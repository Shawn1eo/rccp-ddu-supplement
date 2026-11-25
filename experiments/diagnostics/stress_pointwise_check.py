"""Reproduce the pointwise feasibility check at the three displayed DDU levels."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import norm

from ccw_endogenous_ccp_experiment import (
    CCWModel,
    CQModel,
    ExperimentConfig,
    OracleModel,
    ParametricModel,
    SAAModel,
    _mode1_parameters,
    correlated_context,
    generate_historical_data,
    select_ccw_hyperparameters,
)


RISK_LEVEL = 0.10
DDU_LEVELS = (0.0, 0.5, 1.0)
REPLICATION = 0
THRESHOLD_TOLERANCE = 1e-12


def population_reliability(decision, context, rho, config):
    if not decision.feasible:
        return None
    price = float(decision.price)
    quantity = float(decision.order)
    if quantity < config.profit_target / (price - config.unit_cost):
        return 0.0
    threshold = (
        config.profit_target + (config.unit_cost - config.salvage_value) * quantity
    ) / (price - config.salvage_value)
    if threshold <= 0:
        return 1.0
    mean, scale = _mode1_parameters(np.array([price]), context[None, :], rho, config)
    return float(norm.sf((threshold - mean[0]) / scale[0]))


def check_replication(config, rho, replication=REPLICATION):
    rng = np.random.default_rng(np.random.SeedSequence(
        [config.seed, int(1000 * rho), replication]
    ))
    training = generate_historical_data(rng, config.n_train, rho, config)
    validation = generate_historical_data(rng, config.n_validation, rho, config)
    contexts = correlated_context(rng, config.n_test_contexts, config.context_dim)
    # Advance the original evaluation RNG before drawing parametric scenarios.
    rng.standard_normal((config.n_test_contexts, config.n_eval_mc))
    neighbors, weight, _ = select_ccw_hyperparameters(training, validation, config)
    models = {
        "CCW": CCWModel(training, config, neighbors, weight),
        "CQ-CCP": CQModel(training, config, seed=10_000 + replication),
        "P-CCP": ParametricModel(training, config, rng),
        "SAA-CCP": SAAModel(training, config),
        "Oracle": OracleModel(rho, config),
    }
    for method, model in models.items():
        for index, context in enumerate(contexts):
            decision = model.prescribe(context, RISK_LEVEL)
            reliability = population_reliability(decision, context, rho, config)
            meets = (
                None if reliability is None
                else int(reliability >= 1 - RISK_LEVEL - THRESHOLD_TOLERANCE)
            )
            yield {
                "rho": f"{rho:g}", "alpha": RISK_LEVEL, "replication": replication,
                "method": method, "context_index": index,
                **{f"x{j+1}": float(value) for j, value in enumerate(context)},
                "returned": int(decision.feasible),
                "price": float(decision.price) if decision.feasible else None,
                "order": float(decision.order) if decision.feasible else None,
                "population_reliability": reliability, "meets_requirement": meets,
            }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replication", type=int, default=REPLICATION,
                        help="Zero-based replication index in the original seed schedule.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = ExperimentConfig()
    if not 0 <= args.replication < config.n_replications:
        parser.error(f"--replication must be between 0 and {config.n_replications - 1}.")
    output_path = args.output
    if output_path is None:
        filename = ("stress_pointwise_check_20261119.csv" if args.replication == REPLICATION
                    else f"stress_pointwise_check_replication_{args.replication}.csv")
        output_path = Path(__file__).resolve().parent / "assets/data" / filename
    rows = []
    for rho in DDU_LEVELS:
        rows.extend(check_replication(config, rho, args.replication))
        print(f"Completed replication={args.replication} at rho={rho:g}", flush=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
