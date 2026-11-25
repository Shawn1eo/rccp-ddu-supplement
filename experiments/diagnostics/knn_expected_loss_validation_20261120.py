#!/usr/bin/env python3
"""Reproduce a kNN expected-loss diagnostic on the existing CDF histories.

This separate diagnostic retains the price/context grids, random-number streams,
and kNN neighborhoods of theory_validation.py. It evaluates the manuscript's
newsvendor loss for censored nonnegative demand on 31 quantities from 0 to 110.
No probability data, plots, or manuscript files are modified.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import time


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "assets" / "data"
STEM = "knn_expected_loss_20261120"
os.environ.setdefault(
    "MPLCONFIGDIR", str(ROOT / ".matplotlib_cache_knn_expected_loss_20261120")
)

import numpy as np
import pandas as pd
import scipy
from scipy.integrate import quad
from scipy.special import ndtr
from scipy.stats import norm, t as student_t

import theory_validation as probability


QUANTITY_GRID = np.linspace(0.0, 110.0, 31)
UNIT_COST = 5.0
SALVAGE_VALUE = 2.0


def exact_expected_leftover(quantities, mean, standard_deviation):
    """E[(q-max(D,0))_+] for D~Normal(mean, standard_deviation**2)."""
    quantities = np.asarray(quantities, dtype=float)
    z_quantity = (quantities - mean) / standard_deviation
    z_zero = -mean / standard_deviation
    normalizer = 1.0 / np.sqrt(2.0 * np.pi)
    density_quantity = normalizer * np.exp(-0.5 * z_quantity**2)
    density_zero = normalizer * np.exp(-0.5 * z_zero**2)
    result = (
        (quantities - mean) * ndtr(z_quantity)
        + mean * ndtr(z_zero)
        + standard_deviation * (density_quantity - density_zero)
    )
    # The mathematical expression is zero at q=0. Avoid roundoff cancellation.
    return np.where(quantities == 0.0, 0.0, np.maximum(result, 0.0))


def empirical_expected_leftover(sorted_raw_demand, quantities):
    """Compute all quantities after one sort and one cumulative sum."""
    nonnegative_demand = np.maximum(sorted_raw_demand, 0.0)
    prefix = np.r_[0.0, np.cumsum(nonnegative_demand)]
    counts = np.searchsorted(nonnegative_demand, quantities, side="right")
    return (quantities * counts - prefix[counts]) / len(nonnegative_demand)


def validate_leftover_formula():
    """Check the censoring atom and cumulative-sum calculation independently."""
    quadrature_cases = [
        (-8.0, 3.0, 4.0),
        (0.0, 1.0, 0.0),
        (0.0, 1.0, 2.5),
        (18.0, 4.0, 31.0),
        (62.0, 8.0, 110.0),
    ]
    quadrature_errors = []
    for mean, sd, quantity in quadrature_cases:
        reference = quantity * ndtr(-mean / sd)
        integral, _ = quad(
            lambda demand: (quantity - demand) * norm.pdf(
                (demand - mean) / sd
            ) / sd,
            0.0, quantity, epsabs=1e-11, epsrel=1e-11,
        )
        reference += integral
        exact = float(exact_expected_leftover(quantity, mean, sd))
        quadrature_errors.append(abs(exact - reference))
    sample = np.array([-2.0, 0.0, 3.0, 3.0, 7.0])
    brute_force = np.maximum(
        QUANTITY_GRID[:, None] - np.maximum(sample[None, :], 0.0), 0.0
    ).mean(axis=1)
    fast = empirical_expected_leftover(np.sort(sample), QUANTITY_GRID)
    empirical_error = float(np.max(np.abs(fast - brute_force)))
    assert max(quadrature_errors) < 1e-10
    assert empirical_error < 1e-12
    return {
        "maximum_independent_quadrature_discrepancy": max(quadrature_errors),
        "maximum_prefix_sum_vs_brute_force_discrepancy": empirical_error,
        "cases": [list(case) for case in quadrature_cases],
    }


def exact_query_losses():
    price, context = np.meshgrid(
        probability.PRICE_GRID, probability.CONTEXT_GRID, indexing="ij"
    )
    price = price.ravel()
    context = context.ravel()
    mean, sd = probability.demand_parameters(price, context)
    leftover = exact_expected_leftover(
        QUANTITY_GRID[None, :], mean[:, None], sd[:, None]
    )
    losses = (
        (price[:, None] - SALVAGE_VALUE) * leftover
        - (price[:, None] - UNIT_COST) * QUANTITY_GRID[None, :]
    )
    return price, context, losses


def one_replication(sample_size, replication, queries, cdf_truth, prices,
                    contexts, exact_losses, archived_probability_error):
    started = time.perf_counter()
    rng = np.random.default_rng(np.random.SeedSequence(
        [probability.SEED, sample_size, replication]
    ))
    pairs = rng.uniform(
        0.0, 1.0, size=(sample_size, probability.CONDITIONING_DIMENSION)
    )
    history_prices = 10.0 + 20.0 * pairs[:, 0]
    history_contexts = 2.0 * pairs[:, 1] - 1.0
    mean, sd = probability.demand_parameters(
        history_prices, history_contexts
    )
    raw_demand = mean + sd * rng.standard_normal(sample_size)
    neighbors = probability.tuning(sample_size)[0]
    tree = probability.cKDTree(pairs)
    cutoffs = tree.query(queries, k=[neighbors], workers=1)[0][:, 0]
    groups = tree.query_ball_point(
        queries, np.nextafter(cutoffs, np.inf), workers=1
    )
    counts = np.array([len(group) for group in groups], dtype=int)
    assert counts.min() >= neighbors
    maximum_loss_error = -1.0
    maximum_cdf_error = 0.0
    worst_query = 0
    worst_quantity = 0
    worst_estimated_loss = 0.0
    for query_index, group in enumerate(groups):
        sorted_raw = np.sort(raw_demand[np.asarray(group, dtype=int)])
        leftovers = empirical_expected_leftover(sorted_raw, QUANTITY_GRID)
        estimated_loss = (
            (prices[query_index] - SALVAGE_VALUE) * leftovers
            - (prices[query_index] - UNIT_COST) * QUANTITY_GRID
        )
        errors = np.abs(estimated_loss - exact_losses[query_index])
        quantity_index = int(np.argmax(errors))
        if errors[quantity_index] > maximum_loss_error:
            maximum_loss_error = float(errors[quantity_index])
            worst_query = query_index
            worst_quantity = quantity_index
            worst_estimated_loss = float(estimated_loss[quantity_index])
        # Independently recover the old CDF statistic on the unchanged raw D.
        empirical_cdf = np.searchsorted(
            sorted_raw, probability.THRESHOLD_GRID, side="right"
        ) / len(sorted_raw)
        maximum_cdf_error = max(
            maximum_cdf_error,
            float(np.max(np.abs(empirical_cdf - cdf_truth[query_index]))),
        )
    recreation_error = abs(maximum_cdf_error - archived_probability_error)
    assert recreation_error < 1e-12, (
        sample_size, replication, maximum_cdf_error, archived_probability_error
    )
    return {
        "sample_size": sample_size,
        "replication": replication,
        "method": "kNN",
        "max_expected_loss_error": maximum_loss_error,
        "query_count": len(queries),
        "quantity_count": len(QUANTITY_GRID),
        "neighbors": neighbors,
        "minimum_cluster_size": int(counts.min()),
        "mean_cluster_size": float(counts.mean()),
        "maximum_cluster_size": int(counts.max()),
        "worst_price": float(prices[worst_query]),
        "worst_context": float(contexts[worst_query]),
        "worst_quantity": float(QUANTITY_GRID[worst_quantity]),
        "estimated_loss_at_worst": worst_estimated_loss,
        "exact_loss_at_worst": float(exact_losses[worst_query, worst_quantity]),
        "recreated_probability_error": maximum_cdf_error,
        "archived_probability_error": archived_probability_error,
        "probability_recreation_discrepancy": recreation_error,
        "seconds": time.perf_counter() - started,
    }


def summarize(raw):
    summary = raw.groupby(["sample_size", "method"], as_index=False).agg(
        replications=("replication", "size"),
        complete_replications=("max_expected_loss_error", "count"),
        mean_error=("max_expected_loss_error", "mean"),
        standard_deviation=("max_expected_loss_error", "std"),
        median_error=("max_expected_loss_error", "median"),
        minimum_cluster_size=("minimum_cluster_size", "min"),
        mean_cluster_size=("mean_cluster_size", "mean"),
        mean_seconds=("seconds", "mean"),
        maximum_probability_recreation_discrepancy=(
            "probability_recreation_discrepancy", "max"
        ),
    )
    summary["standard_error"] = (
        summary.standard_deviation / np.sqrt(summary.complete_replications)
    )
    summary["ci95_half_width"] = (
        student_t.ppf(0.975, summary.complete_replications - 1)
        * summary.standard_error
    )
    summary["theory_shape"] = (
        np.log(summary.sample_size) / summary.sample_size
    ) ** probability.RATE_EXPONENT
    return summary


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    protected_paths = [
        ROOT / "theory_validation.py",
        OUTPUT / "contextual_probability_20261119_replications.csv",
        OUTPUT / "contextual_probability_20261119_summary.csv",
    ]
    before_hashes = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in protected_paths
    }
    archived = pd.read_csv(protected_paths[1])
    archived = archived.loc[archived.method.eq("kNN")]
    expected_keys = {
        (sample_size, replication)
        for sample_size in probability.SAMPLE_SIZES
        for replication in range(probability.REPLICATIONS)
    }
    archive_lookup = {
        (int(row.sample_size), int(row.replication)): float(row.max_probability_error)
        for row in archived.itertuples()
    }
    assert set(archive_lookup) == expected_keys
    validation = validate_leftover_formula()
    queries, cdf_truth = probability.query_grid()
    prices, contexts, exact_losses = exact_query_losses()
    rows = []
    started = time.perf_counter()
    for sample_size in probability.SAMPLE_SIZES:
        for replication in range(probability.REPLICATIONS):
            rows.append(one_replication(
                sample_size, replication, queries, cdf_truth, prices, contexts,
                exact_losses, archive_lookup[sample_size, replication],
            ))
        print(
            f"Finished N={sample_size:,}, {probability.REPLICATIONS} replications, "
            f"{time.perf_counter() - started:.2f}s elapsed",
            flush=True,
        )
    raw = pd.DataFrame(rows)
    summary = summarize(raw)
    assert len(raw) == len(expected_keys) == 360
    assert raw.max_expected_loss_error.notna().all()
    assert (summary.complete_replications == probability.REPLICATIONS).all()
    # Empirical and exact losses have the same deterministic procurement term.
    error_range_bound = (
        float(max(probability.PRICE_GRID)) - SALVAGE_VALUE
    ) * float(max(QUANTITY_GRID))
    assert (raw.max_expected_loss_error <= error_range_bound + 1e-10).all()
    for path in protected_paths:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == before_hashes[
            str(path.relative_to(ROOT))
        ]
    raw.to_csv(OUTPUT / f"{STEM}_replications.csv", index=False)
    summary.to_csv(OUTPUT / f"{STEM}_summary.csv", index=False)
    design = {
        "seed": probability.SEED,
        "sample_sizes": list(probability.SAMPLE_SIZES),
        "replications": probability.REPLICATIONS,
        "method": "kNN",
        "conditioning_dimension": probability.CONDITIONING_DIMENSION,
        "historical_pairs": "Identical to theory_validation.py: independent scaled uniform price and context",
        "raw_conditional_demand": "D|p,x ~ Normal(90-2p+12x, (6+2x)^2)",
        "loss_demand": "Y=max(D,0); raw histories and original CDF data are unchanged",
        "unit_cost": UNIT_COST,
        "salvage_value": SALVAGE_VALUE,
        "price_grid": probability.PRICE_GRID.tolist(),
        "context_grid": probability.CONTEXT_GRID.tolist(),
        "quantity_grid": QUANTITY_GRID.tolist(),
        "quantity_grid_reason": "31 equally spaced inventory levels including 0 and110; inventory capacity110 for this diagnostic, aligned with the probability threshold range",
        "loss": "(p-s)*(q-max(D,0))_+ - (p-c)*q",
        "exact_expected_leftover": "(q-mu)*Phi((q-mu)/sigma)+mu*Phi(-mu/sigma)+sigma*(phi((q-mu)/sigma)-phi(-mu/sigma))",
        "exact_expected_loss": "(p-s)*exact_expected_leftover-(p-c)*q",
        "knn_rule": "Use existing tuning(N)[0]; cKDTree kth distance, nextafter toward infinity, then query_ball_point to retain cutoff ties",
        "sampling": "SeedSequence([theory_validation.SEED,N,replication]); same uniform pair draw followed by same standard-normal draws",
        "error": "Maximum absolute empirical-minus-exact expected-loss error over 15 prices x9 contexts x31 quantities",
        "error_units": "Absolute loss units, without normalization",
        "error_range_bound": error_range_bound,
        "confidence_interval": "Two-sided95% Student-t interval for the mean of60 replication maximum-grid errors",
        "implementation": "Sort raw demand once percluster; censor sorted values atzero and use cumulative sums and binary search for all31 quantities",
        "interpretation": "Fixed-grid expected-loss diagnostic, not a certified continuum-uniform finite-sample error bound",
        "elapsed_seconds": time.perf_counter() - started,
        "source_and_probability_hashes": before_hashes,
        "independent_validation": validation,
        "maximum_probability_recreation_discrepancy": float(
            raw.probability_recreation_discrepancy.max()
        ),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
    }
    (OUTPUT / f"{STEM}_design.json").write_text(
        json.dumps(design, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False), flush=True)
    print(f"Results written under {OUTPUT / STEM}", flush=True)


if __name__ == "__main__":
    main()
