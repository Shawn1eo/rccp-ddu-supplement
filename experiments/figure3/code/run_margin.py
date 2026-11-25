"""Paired nominal and tightened empirical chance constraints on fresh replications.

Both arms fit exactly the same models. They differ only in the empirical
violation probability supplied to the four data-driven prescription routines.
Both are evaluated against alpha=0.10 alongside the same alpha=0.10 Oracle.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import time

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "1"
os.environ.setdefault("MPLBACKEND","Agg")

import numpy as np
from threadpoolctl import threadpool_limits

from fast_rerun import _empirical, _load_source, _normalized, _oracle, _select_ccw
from run_cells import clean

HERE = Path(__file__).resolve().parent
ARMS = {"nominal": 0.10, "margin": 0.05}
EVAL_ALPHA = .10
RHOS = (0., .25, .5, .75, 1.)


def make_cells(source_path, rho, rep, check_contexts=0):
    source = _load_source(Path(source_path).resolve())
    rho, rep = float(rho), int(rep)
    cfg = replace(source.ExperimentConfig(),main_alpha=EVAL_ALPHA)
    prices = cfg.prices
    rng = np.random.default_rng(np.random.SeedSequence([cfg.seed,int(1000*rho),rep]))
    train = source.generate_historical_data(rng,cfg.n_train,rho,cfg)
    validation = source.generate_historical_data(rng,cfg.n_validation,rho,cfg)
    contexts = source.correlated_context(rng,cfg.n_test_contexts,cfg.context_dim)
    # Draws are first consumed here to preserve the residual bootstrap RNG
    # state. Only evaluate_policy accesses their values, after both solves.
    eval_draws = rng.standard_normal((len(contexts),cfg.n_eval_mc))
    ccw, crps = _select_ccw(source,train,validation,cfg)
    cq = source.CQModel(train,cfg,seed=10_000 + rep)
    parametric = source.ParametricModel(train,cfg,rng)
    saa = source.SAAModel(train,cfg)
    models = {"CCW":ccw,"CQ-CCP":cq,"P-CCP":parametric,"SAA-CCP":saa,
              "Oracle":source.OracleModel(rho,cfg)}
    count = len(contexts)
    features = cq.features(np.repeat(prices,count),np.tile(contexts,(len(prices),1)))
    raw_quantiles = np.column_stack([model.predict(features) for model in cq.models])
    quantiles = np.maximum.accumulate(raw_quantiles,axis=1)
    fit_by_price = dict(zip(prices,quantiles.reshape(len(prices),count,len(cq.models))))

    def ccw_at(price):
        joint = np.column_stack([np.full(count,price),contexts])
        indices = ccw.nn.kneighbors(_normalized(ccw,joint),return_distance=False)
        return ccw.demands[indices]

    def cq_at(price):
        return np.maximum(0,np.asarray([
            np.interp(cq.probability_grid,cq.levels,q) for q in fit_by_price[price]]))

    def parametric_at(price):
        return np.vstack([parametric.scenarios(price,x) for x in contexts])

    oracle = _oracle(source,cfg,contexts,rho)
    if not all(d.feasible for d in oracle):
        raise ValueError(f"Population Oracle infeasible at {rho=}, {rep=}")
    policies = {}
    for arm, alpha in ARMS.items():
        solve_cfg = replace(cfg,main_alpha=alpha)
        # scenario_decision's explicit alpha is used by SAA. _empirical
        # obtains the same alpha from solve_cfg.main_alpha.
        single = saa.prescribe(contexts[0],alpha)
        policies[arm] = {
            "CCW":_empirical(source,solve_cfg,contexts,ccw_at),
            "CQ-CCP":_empirical(source,solve_cfg,contexts,cq_at),
            "P-CCP":_empirical(source,solve_cfg,contexts,parametric_at),
            "SAA-CCP":[single]*count,"Oracle":oracle}

    cells = {}
    for arm,alpha in ARMS.items():
        rows,details,counts = [],{},{}
        for name in source.METHODS:
            ds = policies[arm][name]
            prescription_alpha = EVAL_ALPHA if name=="Oracle" else alpha
            for i in range(min(count,check_contexts)):
                expected = models[name].prescribe(contexts[i],prescription_alpha)
                assert expected.feasible == ds[i].feasible
                if expected.feasible:
                    np.testing.assert_equal(ds[i].price,expected.price)
                    np.testing.assert_allclose(
                        [ds[i].order,ds[i].estimated_reliability],
                        [expected.order,expected.estimated_reliability],atol=1e-12,rtol=0)
            valid = [d for d in ds if d.feasible]
            counts[name] = len(valid)
            metrics = source.evaluate_policy(ds,contexts,rho,EVAL_ALPHA,cfg,eval_draws)
            rows.append({"rho":rho,"replication":rep,"method":name,"arm":arm,
                         "alpha":EVAL_ALPHA,"solve_alpha":prescription_alpha,
                         "evaluation_alpha":EVAL_ALPHA,
                         "k_ccw":ccw.k,"ccw_action_weight":ccw.action_weight,
                         "ccw_validation_crps":crps,**metrics,
                         "mean_price":float(np.mean([d.price for d in valid])) if valid else None,
                         "mean_order":float(np.mean([d.order for d in valid])) if valid else None})
            details[name] = [{"context_index":i,"context_id":i,"feasible":bool(d.feasible),
                              "price":float(d.price) if d.feasible else None,
                              "order":float(d.order) if d.feasible else None,
                              "estimated_reliability":float(d.estimated_reliability) if d.feasible else None}
                             for i,d in enumerate(ds)]
        cells[arm] = {"engine":"paired_fast","arm":arm,"solve_alpha":alpha,
                      "evaluation_alpha":EVAL_ALPHA,"oracle_solve_alpha":EVAL_ALPHA,
                      "replication":rep,"rho":rho,"rows":rows,"decisions":details,
                      "context_count":count,"contexts":contexts.tolist(),
                      "feasible_counts":counts,"scalar_checks_per_method":min(count,check_contexts)}
    return cells


def write_cell(source_path, outdir, rho, rep, check_contexts):
    name = f"rho_{round(100*rho):03d}_rep_{rep:03d}_paired.json"
    targets = {arm:outdir/arm/name for arm in ARMS}
    if all(path.is_file() for path in targets.values()):
        return rho,rep,"cached"
    start = time.perf_counter()
    with threadpool_limits(limits=1):
        cells = make_cells(source_path,rho,rep,check_contexts)
    for arm,path in targets.items():
        path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():
            continue
        with tempfile.NamedTemporaryFile(mode="w",encoding="utf-8",dir=path.parent,
                                         suffix=".part",delete=False) as out:
            json.dump(clean(cells[arm]),out,allow_nan=False)
            tmp_path = Path(out.name)
        tmp_path.replace(path)
    return rho,rep,round(time.perf_counter()-start,2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--output",type=Path,default=HERE/"coeval")
    parser.add_argument("--start",type=int,default=70)
    parser.add_argument("--stop",type=int,default=90,help="Exclusive replication ID")
    parser.add_argument("--rhos",type=float,nargs="+",default=RHOS)
    parser.add_argument("--workers",type=int,default=3)
    parser.add_argument("--check-contexts",type=int,default=0,
                        help="Compare first n decisions per arm and method with scalar archived code")
    args = parser.parse_args()
    if not 70<=args.start<args.stop<=90 or any(rho not in RHOS for rho in args.rhos):
        parser.error("Use fresh replication IDs 70 through 89 and the five archived intensities.")
    if args.workers<1 or not 0<=args.check_contexts<=80:
        parser.error("Invalid process count or scalar-check count.")
    work = [(args.source.resolve(),args.output.resolve(),rho,rep,args.check_contexts)
            for rep in range(args.start,args.stop) for rho in dict.fromkeys(args.rhos)]
    if args.workers == 1:
        for task in work:
            print(write_cell(*task),flush=True)
        return
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        pending = [executor.submit(write_cell,*task) for task in work]
        for future in as_completed(pending):
            print(future.result(),flush=True)


if __name__ == "__main__":
    main()
