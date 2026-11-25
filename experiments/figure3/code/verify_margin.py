"""Validate paired risk levels against the original experiment for two independent cells."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

from run_cells import clean, load_source


METHODS = ("CCW", "CQ-CCP", "P-CCP", "SAA-CCP", "Oracle")
RHOS = (0.25, 0.75)
ALPHAS = (0.10, 0.05)
REPLICATION = 70


def reference_name(rho):
    return f"rho_{int(100*rho):02d}_rep_{REPLICATION:03d}_original.json"


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    tmp.write_text(json.dumps(clean(data), indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def generate(source, directory, rho):
    source = Path(source)
    mod = load_source(source)
    config = replace(mod.ExperimentConfig(), effective_alphas=ALPHAS)
    evaluator = mod.evaluate_policy
    actions = {str(alpha): {} for alpha in ALPHAS}

    def record(decisions, contexts, rho_value, alpha, cfg, shocks):
        dest = actions[str(alpha)]
        method = METHODS[len(dest)]
        dest[method] = [
            dict(context_index=i, feasible=bool(d.feasible), price=d.price,
                 order=d.order, estimated_reliability=d.estimated_reliability)
            for i, d in enumerate(decisions)
        ]
        return evaluator(decisions, contexts, rho_value, alpha, cfg, shocks)

    mod.evaluate_policy = record
    start = time.perf_counter()
    try:
        rows = mod.run_one_replication(config, rho, REPLICATION)
    finally:
        mod.evaluate_policy = evaluator
    assert all(tuple(arm) == METHODS for arm in actions.values())
    assert all(len(ds) == config.n_test_contexts for arm in actions.values() for ds in arm.values())
    dst = directory / reference_name(rho)
    write_json(dst, dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                         rho=rho, replication=REPLICATION, elapsed_seconds=time.perf_counter()-start,
                         effective_alphas=ALPHAS, rows=rows, decisions=actions))
    return dst


def run_references(source, root):
    directory = root / "reference"
    directory.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(generate, source, directory, rho) for rho in RHOS]
        for job in as_completed(jobs):
            print(job.result(), flush=True)


def reproduce_evaluation_inputs(mod, rho, cfg):
    import numpy as np
    rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, int(1000*rho), REPLICATION]))
    mod.generate_historical_data(rng, cfg.n_train, rho, cfg)
    mod.generate_historical_data(rng, cfg.n_validation, rho, cfg)
    contexts = mod.correlated_context(rng, cfg.n_test_contexts, cfg.context_dim)
    draws = rng.standard_normal((cfg.n_test_contexts, cfg.n_eval_mc))
    return contexts, draws


def record_error(errors, issues, key, expected, observed, identifier):
    if expected is None or observed is None or isinstance(expected, (str, bool)):
        difference = float(expected != observed or type(expected) != type(observed))
    else:
        difference = abs(expected - observed)
    errors[key] = max(difference, errors.get(key, 0.0))
    if difference > 1e-9:
        issues.append(dict(at=identifier, field=key, expected=expected, actual=observed))


def verify(source, root):
    import numpy as np
    from threadpoolctl import threadpool_limits
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    mod = load_source(source)
    cfg = mod.ExperimentConfig()
    comparisons = []
    issues = []
    with threadpool_limits(limits=1):
        for rho in RHOS:
            reference = json.loads((root / "reference" / reference_name(rho)).read_text(encoding="utf-8"))
            assert reference["source_sha256"] == digest
            rows = {(r["method"], r["alpha"]): r for r in reference["rows"]}
            contexts, draws = reproduce_evaluation_inputs(mod, rho, cfg)
            for arm in ("nominal", "margin"):
                name = f"rho_{round(100*rho):03d}_rep_{REPLICATION:03d}_paired.json"
                raw = json.loads((root / arm / name).read_text(encoding="utf-8"))
                assert raw["arm"] == arm and raw["rho"] == rho and raw["replication"] == REPLICATION
                np.testing.assert_array_equal(contexts, raw["contexts"])
                actual_rows = {r["method"]: r for r in raw["rows"]}
                assert set(actual_rows) == set(raw["decisions"]) == set(METHODS)
                for method in METHODS:
                    alpha = 0.05 if arm == "margin" and method != "Oracle" else 0.1
                    expected_ds = reference["decisions"][str(alpha)][method]
                    actual_ds = raw["decisions"][method]
                    assert len(expected_ds) == len(actual_ds) == cfg.n_test_contexts
                    errors = {}
                    group = f"{rho}/{arm}/{method}"
                    for i, (d, other) in enumerate(zip(expected_ds, actual_ds)):
                        assert d["context_index"] == other["context_index"] == other["context_id"] == i
                        for field in ("feasible", "price", "order", "estimated_reliability"):
                            record_error(errors, issues, "decision_" + field,
                                         d[field], other[field], f"{group}/{i}")
                    actual_row = actual_rows[method]
                    assert actual_row["solve_alpha"] == alpha and actual_row["evaluation_alpha"] == 0.1
                    expected_row = rows[method, alpha].copy()
                    if alpha != 0.1:
                        ds = [mod.Decision(d["price"], d["order"], d["estimated_reliability"], True)
                              if d["feasible"] else mod.Decision() for d in expected_ds]
                        expected_row.update(clean(mod.evaluate_policy(ds, contexts, rho, 0.1, cfg, draws)))
                    expected_row["alpha"] = 0.1
                    for field, value in expected_row.items():
                        record_error(errors, issues, "row_" + field, value, actual_row[field], group)
                    comparisons.append(dict(rho=rho, replication=REPLICATION, arm=arm, method=method,
                                            reference_solve_alpha=alpha, evaluation_alpha=0.1,
                                            n_contexts=len(expected_ds), max_absolute_error=errors))
    report = dict(source_sha256=digest, absolute_tolerance=1e-9, passed=not issues,
                  action_comparisons=sum(row["n_contexts"] for row in comparisons),
                  comparisons=comparisons, failures=issues)
    write_json(root / "margin_engine_check.json", report)
    if issues:
        raise AssertionError(issues[:3])
    print("PASS", report["action_comparisons"], "context/method/arm actions", flush=True)


if __name__ == "__main__":
    default = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["reference", "verify"])
    parser.add_argument("--root", type=Path, default=default / "coeval")
    parser.add_argument("--source", type=Path, default=default.parents[2] / "ccw_endogenous_ccp_experiment.py")
    args = parser.parse_args()
    task = run_references if args.action == "reference" else verify
    task(args.source.resolve(), args.root.resolve())
