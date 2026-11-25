"""Compare nominal and guarded prescriptions on the same fresh test contexts.

Profit differences and Oracle shares use only contexts in which both actions
meet the population target. Coverage and return rates use all attempted contexts.
All changes subtract the nominal value from the guarded value.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t
from population_diagnostics import HERE, PROFIT_TOL, RHOS, ratio_by_cluster


KEYS = ['rho', 'replication', 'context_id', 'method']
COLUMNS = KEYS + ['returned', 'true_feasible', 'price', 'order', 'true_expected_profit',
                  'true_reliability', 'oracle_expected_profit']
METHODS = {'CCW', 'CQ-CCP', 'P-CCP', 'SAA-CCP', 'Oracle'}


def read_arm(folder):
    data = pd.read_csv(folder/'pointwise_population.csv', usecols=COLUMNS)
    if data.duplicated(KEYS).any():
        raise ValueError(f'Duplicate test-context and method key in {folder}')
    for col in ['returned', 'true_feasible']:
        if data[col].dtype != bool:
            raise ValueError(f'Invalid {col} column in {folder}')
    if (data.true_feasible & ~data.returned).any():
        raise ValueError(f'True-feasible action marked as missing in {folder}')
    return data


def paired_rows(nominal, margin):
    data = nominal.merge(margin, on=KEYS, how='outer', validate='one_to_one',
                         suffixes=('_nominal','_margin'), indicator=True)
    if not data['_merge'].eq('both').all():
        raise ValueError('The two arms must cover the same replications, contexts, and methods.')
    # A common constrained Oracle is essential to all normalized comparisons.
    np.testing.assert_allclose(data.oracle_expected_profit_nominal, data.oracle_expected_profit_margin,
                               rtol=1e-11, atol=PROFIT_TOL)
    oracle = data[data.method.eq('Oracle')]
    for col in ['price','order','true_reliability','true_expected_profit']:
        np.testing.assert_allclose(oracle[col+'_nominal'], oracle[col+'_margin'], rtol=1e-11, atol=1e-10)
    assert oracle.true_feasible_nominal.all() and oracle.true_feasible_margin.all()
    return data


def add_change(row, name, numerator, denominator):
    effect, se = ratio_by_cluster(numerator, denominator)
    half_width = t.ppf(.975, len(denominator)-1)*se if np.isfinite(se) else np.nan
    row[name] = effect
    row[name+'_se'] = se
    row[name+'_ci95_low'] = effect-half_width
    row[name+'_ci95_high'] = effect+half_width


def summarize_group(g):
    sizes = g.groupby('replication').size()
    grouped = g.groupby('replication')
    row = dict(n_reps=len(sizes), n_contexts=len(g))
    for key, field in [('return', 'returned'), ('true_feasible', 'true_feasible')]:
        nom = g[field+'_nominal']
        margin = g[field+'_margin']
        row['n_'+field+'_nominal'] = int(nom.sum())
        row['n_'+field+'_margin'] = int(margin.sum())
        row[key+'_rate_nominal'] = nom.mean()
        row[key+'_rate_margin'] = margin.mean()
        row['n_'+key+'_gained'] = int((~nom & margin).sum())
        row['n_'+key+'_lost'] = int((nom & ~margin).sum())
        counts = grouped[[field+'_nominal',field+'_margin']].sum()
        add_change(row, key+'_rate_change', counts.iloc[:,1]-counts.iloc[:,0], sizes)
    joint = g[g.true_feasible_nominal & g.true_feasible_margin].copy()
    row['n_joint_true_feasible'] = len(joint)
    row['n_reps_with_joint_true_feasible'] = joint.replication.nunique()
    joint['difference'] = joint.true_expected_profit_margin-joint.true_expected_profit_nominal
    joint['oracle'] = joint.oracle_expected_profit_nominal
    joint_sums = (joint.groupby('replication')
                  .agg(n=('difference','size'), profit_nominal=('true_expected_profit_nominal','sum'),
                       profit_margin=('true_expected_profit_margin','sum'), diff=('difference','sum'),
                       oracle=('oracle','sum')).reindex(sizes.index,fill_value=0))
    for arm in ['nominal','margin']:
        total = joint_sums['profit_'+arm]
        row['joint_mean_profit_'+arm] = ratio_by_cluster(total,joint_sums.n)[0]
        row['joint_matched_oracle_share_'+arm] = ratio_by_cluster(total,joint_sums.oracle)[0]
    row['joint_mean_oracle_profit'] = ratio_by_cluster(joint_sums.oracle,joint_sums.n)[0]
    add_change(row,'joint_mean_profit_change',joint_sums['diff'],joint_sums.n)
    add_change(row,'joint_oracle_share_change',joint_sums['diff'],joint_sums.oracle)
    add_change(row,'joint_relative_profit_change',joint_sums['diff'],joint_sums.profit_nominal)
    row['joint_margin_wins'] = int((joint.difference > PROFIT_TOL).sum())
    row['joint_nominal_wins'] = int((joint.difference < -PROFIT_TOL).sum())
    row['joint_ties'] = int((joint.difference.abs() <= PROFIT_TOL).sum())
    return row


def main():
    data_dir = HERE/'coeval'
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--nominal',type=Path,default=data_dir/'nominal_results')
    p.add_argument('--margin',type=Path,default=data_dir/'margin_results')
    p.add_argument('--out',type=Path,default=data_dir/'paired_margin_cost.csv')
    p.add_argument('--require-complete',action='store_true',
                   help='Verify the preregistered five levels, 20 replications, 80 contexts, and five methods.')
    args = p.parse_args()
    joined = paired_rows(read_arm(args.nominal), read_arm(args.margin))
    if args.require_complete:
        expected = len(RHOS)*20*80*len(METHODS)
        if (len(joined) != expected or set(joined.rho)!=set(RHOS) or
            set(joined.replication)!=set(range(70,90)) or set(joined.method)!=METHODS or
            set(joined.context_id)!=set(range(80))):
            p.error('The arms are incomplete for the preregistered fresh design.')
    out = pd.DataFrame([dict(rho=rho, method=method, **summarize_group(g))
                        for (rho,method),g in joined.groupby(['rho','method'],sort=False)])
    args.out.parent.mkdir(parents=True,exist_ok=True)
    temporary = args.out.with_suffix('.tmp')
    out.to_csv(temporary,index=False)
    temporary.replace(args.out)
    print(out.loc[out.method.eq('CCW'),['rho','n_contexts','n_true_feasible_nominal',
              'n_true_feasible_margin','n_joint_true_feasible','joint_mean_profit_change',
              'joint_relative_profit_change']].to_string(index=False))


if __name__ == '__main__':
    main()
