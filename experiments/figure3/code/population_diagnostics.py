"""Evaluate saved fixed-alpha actions under the generating distribution without refitting.

Rates divide by every attempted context. Conditional profit comparisons use the
same contexts for both policies. Standard errors treat replications as independent
clusters, with all 80 contexts and both matched actions kept together.
"""
import argparse
import json
from pathlib import Path

from run_cells import load_source
import numpy as np
import pandas as pd
from scipy.stats import norm


HERE = Path(__file__).resolve().parent
RHOS = (0., .25, .50, .75, 1.)
REPS = range(50, 70)
PROB_TOL = 1e-12
PROFIT_TOL = 1e-8


def read_cells(folder, start=REPS.start, stop=REPS.stop, fitted_alpha=.10):
    selected = {}
    for path in sorted(folder.glob('rho_*_rep_*_*.json')):
        data = json.loads(path.read_text(encoding='utf-8'))
        rho = float(data['rows'][0]['rho'])
        rep = int(data['rows'][0]['replication'])
        if rho not in RHOS or not start <= rep < stop:
            continue
        key = rho, rep
        assert all(row['rho'] == rho and row['replication'] == rep
                   and row['alpha'] == .10 for row in data['rows'])
        for row in data['rows']:
            if 'solve_alpha' in row or fitted_alpha != .10:
                expected_alpha = .10 if row['method'] == 'Oracle' else fitted_alpha
                assert row['solve_alpha'] == expected_alpha
        if key not in selected or data['engine'] == 'original':
            selected[key] = data
    return selected


def contexts_for_cell(source, cfg, cell, rho, rep):
    if 'contexts' in cell:
        contexts = np.asarray(cell['contexts'], dtype=float)
    else:
        rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, int(rho*1000), rep]))
        source.generate_historical_data(rng, cfg.n_train, rho, cfg)
        source.generate_historical_data(rng, cfg.n_validation, rho, cfg)
        contexts = source.correlated_context(rng, cfg.n_test_contexts, cfg.context_dim)
    assert contexts.shape == (cfg.n_test_contexts, cfg.context_dim)
    return contexts


def population_metrics(source, cfg, rho, contexts, prices, orders):
    mu, sd = source._mode1_parameters(prices, contexts, rho, cfg)
    qmin = cfg.profit_target / (prices - cfg.unit_cost)
    threshold = ((cfg.profit_target + (cfg.unit_cost-cfg.salvage_value)*orders)
                 / (prices-cfg.salvage_value))
    # For nonnegative orders and v=100 the required demand is strictly positive.
    # Clipping the latent normal at zero leaves this upper-tail probability intact.
    assert np.all(threshold > 0) and np.all(sd > 0)
    reaches_target_at_sellout = orders >= qmin
    reliability = np.where(reaches_target_at_sellout, norm.sf((threshold-mu)/sd), 0.)
    expected_profit = source.expected_profit_mode1(prices, orders, mu, sd, cfg)
    boundary = np.isclose(orders, qmin, rtol=0., atol=1e-10)
    raw_floating_gate = (prices - cfg.unit_cost) * orders >= cfg.profit_target
    disagreement = reaches_target_at_sellout != raw_floating_gate
    assert not np.any(disagreement & ~boundary)
    return reliability, expected_profit, boundary, disagreement, ~reaches_target_at_sellout


def audit_oracle_optimum(source, cfg, rho, contexts, recorded_profit):
    # Independently maximize the normal-model objective over all feasible prices
    # and the exact feasible quantity interval at each of those prices.
    p = np.tile(cfg.prices, len(contexts))
    x = np.repeat(contexts, len(cfg.prices), axis=0)
    mu, sd = source._mode1_parameters(p, x, rho, cfg)
    qmin = cfg.profit_target/(p-cfg.unit_cost)
    qtail = np.maximum(0., mu+sd*norm.ppf(cfg.main_alpha))
    qmax = ((p-cfg.salvage_value)*qtail-cfg.profit_target)/(cfg.unit_cost-cfg.salvage_value)
    fractile = (p-cfg.unit_cost)/(p-cfg.salvage_value)
    q = np.clip(np.maximum(0., mu+sd*norm.ppf(fractile)), qmin, qmax)
    profit = source.expected_profit_mode1(p, q, mu, sd, cfg)
    best = np.where(qmax >= qmin, profit, -np.inf).reshape(len(contexts), -1).max(axis=1)
    np.testing.assert_allclose(best, recorded_profit, rtol=1e-11, atol=PROFIT_TOL)


def evaluate_cell(source, cfg, cell, rho, rep):
    contexts = contexts_for_cell(source, cfg, cell, rho, rep)
    frames = []
    for method in source.METHODS:
        actions = cell['decisions'][method]
        assert len(actions) == len(contexts)
        ids = [a.get('context_id', a.get('context_index')) for a in actions]
        assert ids == list(range(len(contexts)))
        returned = np.array([a['feasible'] for a in actions], dtype=bool)
        p = np.array([a['price'] for a in actions], dtype=float)
        q = np.array([a['order'] for a in actions], dtype=float)
        assert np.array_equal(returned, np.isfinite(p) & np.isfinite(q))
        assert np.all(q[returned] >= 0) and np.isin(p[returned], cfg.prices).all()
        d = pd.DataFrame(dict(
            rho=rho, replication=rep, context_id=ids, method=method, engine=cell['engine'],
            returned=returned, price=p, order=q, true_reliability=np.nan,
            true_expected_profit=np.nan, at_quantity_boundary=False,
            product_cutoff_disagreement=False, below_minimum_quantity=False))
        if returned.any():
            metrics = population_metrics(source, cfg, rho, contexts[returned], p[returned], q[returned])
            columns = ('true_reliability', 'true_expected_profit', 'at_quantity_boundary',
                       'product_cutoff_disagreement', 'below_minimum_quantity')
            for col, values in zip(columns, metrics):
                d.loc[returned, col] = values
        d['true_feasible'] = d.returned & (d.true_reliability >= 1.-cfg.main_alpha-PROB_TOL)
        if method == 'Oracle':
            assert d.true_feasible.all()
            reported = [a['estimated_reliability'] for a in actions]
            np.testing.assert_allclose(d.true_reliability, reported, rtol=0, atol=5e-12)
            oracle_profit = d.true_expected_profit.to_numpy()
            audit_oracle_optimum(source, cfg, rho, contexts, oracle_profit)
        frames.append(d)
    out = pd.concat(frames, ignore_index=True)
    out['oracle_expected_profit'] = oracle_profit[out.context_id.to_numpy()]
    feasible = out[out.true_feasible]
    assert ((feasible.true_expected_profit-feasible.oracle_expected_profit) <= PROFIT_TOL).all()
    return out


def ratio_by_cluster(numerator, denominator):
    """Ratio of totals and replication-cluster delta-method standard error."""
    a, b = np.asarray(numerator, float), np.asarray(denominator, float)
    if b.sum() == 0:
        return np.nan, np.nan
    r = a.sum()/b.sum()
    if len(a) < 2 or np.count_nonzero(b) < 2:
        return r, np.nan
    se = np.linalg.norm(a-r*b)*np.sqrt(len(a)/(len(a)-1))/b.sum()
    return r, se


def summary_for(method_group):
    all_reps = method_group.groupby('replication').size()
    nrep = len(all_reps)
    returned = method_group[method_group.returned]
    feasible = method_group[method_group.true_feasible]
    byrep = feasible.groupby('replication')
    nreturn = returned.groupby('replication').size().reindex(all_reps.index, fill_value=0)
    nf = byrep.size().reindex(all_reps.index, fill_value=0)
    sums = byrep[['true_expected_profit','oracle_expected_profit']].sum().reindex(all_reps.index, fill_value=0)
    rate, rate_se = ratio_by_cluster(nreturn, all_reps)
    good, good_se = ratio_by_cluster(nf, all_reps)
    share, share_se = ratio_by_cluster(sums.true_expected_profit, sums.oracle_expected_profit)
    return dict(n_reps=nrep, n_contexts=len(method_group), n_returned=len(returned),
                n_true_feasible=len(feasible), n_reps_with_true_feasible=int((nf>0).sum()),
                return_rate=rate, return_rate_se=rate_se, true_feasible_rate=good,
                true_feasible_rate_se=good_se, mean_true_reliability_returned=returned.true_reliability.mean(),
                mean_true_profit_feasible=feasible.true_expected_profit.mean(),
                mean_matched_oracle_profit=feasible.oracle_expected_profit.mean(),
                matched_oracle_profit_share=share, matched_oracle_profit_share_se=share_se,
                n_at_quantity_boundary=int(method_group.at_quantity_boundary.sum()),
                n_product_cutoff_disagreements=int(method_group.product_cutoff_disagreement.sum()),
                n_below_min_quantity=int(method_group.below_minimum_quantity.sum()))


def pairwise(pointwise, methods):
    records = []
    for rho, panel in pointwise.groupby('rho', sort=True):
        reps = sorted(panel.replication.unique())
        feasible = panel[panel.true_feasible].pivot(index=['replication','context_id'],
                                                   columns='method', values='true_expected_profit')
        for competitor in (m for m in methods if m != 'CCW'):
            pair = feasible.reindex(columns=['CCW',competitor]).dropna()
            diff = pair.CCW-pair[competitor]
            per_rep = diff.groupby(level='replication').agg(['sum','count']).reindex(reps, fill_value=0)
            effect, effect_se = ratio_by_cluster(per_rep['sum'], per_rep['count'])
            records.append(dict(rho=rho, competitor=competitor, n_reps=len(reps),
                                n_pairs=len(pair), n_reps_with_pairs=int((per_rep['count']>0).sum()),
                                mean_ccw_profit=pair.CCW.mean(), mean_competitor_profit=pair[competitor].mean(),
                                mean_profit_difference=effect, mean_profit_difference_se=effect_se,
                                n_ccw_wins=int((diff > PROFIT_TOL).sum()),
                                n_ties=int((diff.abs() <= PROFIT_TOL).sum()),
                                n_competitor_wins=int((diff < -PROFIT_TOL).sum())))
    return pd.DataFrame(records)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=HERE.parents[2]/'ccw_endogenous_ccp_experiment.py')
    p.add_argument('--cells', type=Path, default=HERE/'cells')
    p.add_argument('--out', type=Path, default=HERE/'population')
    p.add_argument('--start', type=int, default=REPS.start)
    p.add_argument('--stop', type=int, default=REPS.stop, help='Exclusive replication bound')
    p.add_argument('--fitted-alpha', type=float, default=.10,
                   help='Check the fitted violation level. Population evaluation and the Oracle always use 0.10.')
    p.add_argument('--require-complete', action='store_true')
    args = p.parse_args()
    source = load_source(args.source.resolve())
    cfg = source.ExperimentConfig()
    assert cfg.main_alpha == .10 and cfg.dgp_mode == 1
    if args.stop <= args.start:
        p.error('--stop must exceed --start.')
    cells = read_cells(args.cells, args.start, args.stop, args.fitted_alpha)
    expected = len(RHOS)*(args.stop-args.start)
    if not cells:
        p.error(f'No cells available for replications {args.start} through {args.stop-1}.')
    if args.require_complete and len(cells) != expected:
        p.error(f'Only {len(cells)} of the {expected} requested cells are available.')
    rows = [evaluate_cell(source,cfg,cell,rho,rep) for (rho,rep),cell in sorted(cells.items())]
    points = pd.concat(rows, ignore_index=True)
    summary = pd.DataFrame([dict(rho=rho, method=m, **summary_for(g))
                            for (rho,m),g in points.groupby(['rho','method'],sort=False)])
    pairs = pairwise(points, source.METHODS)
    args.out.mkdir(parents=True, exist_ok=True)
    for name,frame in [('pointwise_population',points),('population_summary',summary),('paired_profit',pairs)]:
        temporary = args.out/(name+'.tmp')
        frame.to_csv(temporary,index=False)
        temporary.replace(args.out/(name+'.csv'))
    print(f'{len(cells)}/{expected} cells, {len(points)} pointwise results, {len(summary)} summary rows.')
    print(summary[['rho','method','n_reps','n_returned','n_true_feasible','matched_oracle_profit_share']].to_string(index=False))


if __name__ == '__main__':
    main()
