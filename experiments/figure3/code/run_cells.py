"""Fixed-alpha check using the unmodified experiment module and Monte Carlo evaluator."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"

def load_source(path):
    spec = importlib.util.spec_from_file_location("fig3_original", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod

def original(source, rho, rep):
    from dataclasses import replace
    import numpy as np
    m = load_source(source)
    cfg = replace(m.ExperimentConfig(), effective_alphas=(0.10,))
    decisions = {}
    evaluator = m.evaluate_policy

    def record(ds, contexts, rho_value, alpha, config, shocks):
        method = m.METHODS[len(decisions)]
        decisions[method] = [
            {"context_id": index, "feasible": bool(d.feasible),
             "price": float(d.price), "order": float(d.order),
             "estimated_reliability": float(d.estimated_reliability)}
            for index, d in enumerate(ds)
        ]
        return evaluator(ds, contexts, rho_value, alpha, config, shocks)

    m.evaluate_policy = record
    try:
        rows = m.run_one_replication(cfg, rho, rep)
    finally:
        m.evaluate_policy = evaluator
    return {"rows": rows, "decisions": decisions}

def clean(value):
    import numpy as np
    if isinstance(value, dict):
        return {key: clean(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(val) for val in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.int64, np.int32)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value

def write_cell(source, root, rho, rep, engine):
    start = time.perf_counter()
    if engine == "original":
        result = original(source, rho, rep)
    else:
        from fast_rerun import run
        result = run(source, rho, rep)
    result = clean(result)
    result["engine"] = engine
    result["elapsed_seconds"] = time.perf_counter() - start
    dst = root / ("rho_%02d_rep_%03d_%s.json" % (round(rho*10),rep,engine))
    tmp = dst.with_suffix(".part")
    tmp.write_text(json.dumps(result, allow_nan=False), encoding="utf-8")
    tmp.replace(dst)
    return (rho, rep, round(result["elapsed_seconds"], 1), [r.get("policy_feasible") for r in result["rows"]])

if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor, as_completed
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--engine", choices=["original","fast"], default="original")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--start", type=int, required=True)
    p.add_argument("--stop", type=int, required=True, help="Exclusive repetition ID")
    p.add_argument("--rhos", type=float, nargs="+", default=[0,.5,1])
    args = p.parse_args()
    root=args.output.resolve()
    root.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs={}
        for rep in range(args.start,args.stop):
            for rho in args.rhos:
                dst=root/("rho_%02d_rep_%03d_%s.json" % (round(rho*10),rep,args.engine))
                if not dst.is_file():
                    job=pool.submit(write_cell,args.source.resolve(),root,rho,rep,args.engine)
                    jobs[job]=(rho,rep)
        for job in as_completed(jobs):
            print(job.result(), flush=True)
