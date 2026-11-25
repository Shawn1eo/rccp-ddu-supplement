#!/usr/bin/env python3
"""Reproduce the contextual CCW probability-estimation diagnostic.

Run with Python 3.10 or later and numpy, scipy, pandas, matplotlib, scikit-learn.
The two-dimensional conditioning pair contains price and one context variable.
The diagnostic compares conditional CDFs on a fixed grid, using the same data
for kNN, LSA, and CART in each replication. With r_N=(log(N)/N)^(1/6), the
probability fits are 1-exp(-C*r_N) for kNN and C*r_N for LSA. Both use weights
sqrt(N) and nonnegative coefficients. An optional expected-loss summary adds
a kNN panel with a separately fitted C*r_N curve. These curves describe the
fixed-grid observations. The exponent stays fixed.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.spatial import cKDTree
from scipy.special import ndtr, stdtr
from scipy.stats import t as student_t
from scipy.optimize import minimize_scalar
from sklearn.tree import DecisionTreeRegressor


SEED = 20261119
SAMPLE_SIZES = (1000, 2000, 5000, 10000, 30000, 100000)
REPLICATIONS = 60
CONDITIONING_DIMENSION = 2
RATE_EXPONENT = 1.0 / (2.0 * (CONDITIONING_DIMENSION + 1))
K_MULTIPLIER = 1.0
H_MULTIPLIER = 0.6
PRICE_GRID = np.linspace(10.0, 30.0, 15)
CONTEXT_GRID = np.linspace(-1.0, 1.0, 9)
THRESHOLD_GRID = np.linspace(0.0, 110.0, 61)
FILE_STEM = "contextual_probability_20261119"
METHODS = ("kNN", "LSA", "CART")
SHOCK_LAWS = ("normal", "lognormal", "student_t3")
LOSS_ERROR_BOUND = 3080.0


def demand_parameters(price: np.ndarray, context: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the demand location and positive shock scale."""
    location = 90.0 - 2.0 * price + 12.0 * context
    scale = 6.0 + 2.0 * context
    return location, scale


def tuning(sample_size: int) -> tuple[int, float, int]:
    """Balance the displayed theorem terms for a two-dimensional pair."""
    rate_shape = (np.log(sample_size) / sample_size) ** RATE_EXPONENT
    neighbors = int(np.ceil(K_MULTIPLIER * sample_size * rate_shape**CONDITIONING_DIMENSION))
    bandwidth = H_MULTIPLIER * rate_shape
    minimum_leaf_size = max(2, int(np.ceil(neighbors / 2.0)))
    return min(sample_size, neighbors), bandwidth, minimum_leaf_size


def query_grid(shock_law: str = "normal") -> tuple[np.ndarray, np.ndarray]:
    """Use equal coordinate ranges for price and the meaningful context."""
    price, context = np.meshgrid(PRICE_GRID, CONTEXT_GRID, indexing="ij")
    price, context = price.ravel(), context.ravel()
    pairs = np.column_stack(((price - 10.0) / 20.0, (context + 1.0) / 2.0))
    location, scale = demand_parameters(price, context)
    z = (THRESHOLD_GRID[None, :] - location[:, None]) / scale[:, None]
    if shock_law == "normal":
        truth = ndtr(z)
    elif shock_law == "lognormal":
        log_z = np.full_like(z, -np.inf)
        np.log(z, where=z > 0, out=log_z)
        truth = ndtr(log_z)
    elif shock_law == "student_t3":
        truth = stdtr(3, z)
    else:
        raise ValueError(f"Unknown shock law: {shock_law}")
    return pairs, truth


def cdf_error(demand: np.ndarray, groups: list[np.ndarray], truth: np.ndarray) -> dict[str, float | int]:
    """Evaluate all thresholds by sorting demands once per neighborhood."""
    counts = np.array([len(indices) for indices in groups], dtype=int)
    errors = []
    worst_query = -1
    worst_threshold = -1
    maximum_error = -1.0
    for query_index, (indices, exact_probabilities) in enumerate(zip(groups, truth)):
        if len(indices) == 0:
            continue
        sorted_demand = np.sort(demand[indices])
        estimated = np.searchsorted(sorted_demand, THRESHOLD_GRID, side="right") / len(indices)
        absolute_error = np.abs(estimated - exact_probabilities)
        threshold_index = int(np.argmax(absolute_error))
        error = float(absolute_error[threshold_index])
        errors.append(error)
        if error > maximum_error:
            maximum_error = error
            worst_query = query_index
            worst_threshold = threshold_index
    return {
        "max_probability_error": max(errors) if counts.min() > 0 else np.nan,
        "available_queries": int(np.count_nonzero(counts)),
        "query_count": len(groups),
        "minimum_cluster_size": int(counts.min()),
        "mean_cluster_size": float(counts.mean()),
        "maximum_cluster_size": int(counts.max()),
        "worst_price": float(PRICE_GRID[worst_query // len(CONTEXT_GRID)]),
        "worst_context": float(CONTEXT_GRID[worst_query % len(CONTEXT_GRID)]),
        "worst_demand_threshold": float(THRESHOLD_GRID[worst_threshold]),
    }


def one_replication(sample_size: int, replication: int, seed: int,
                    queries: np.ndarray, truth: np.ndarray, *, shock_law: str = "normal") -> list[dict]:
    """Draw one history and evaluate the three methods on that same history."""
    sequence = np.random.SeedSequence([seed, sample_size, replication])
    rng = np.random.default_rng(sequence)
    pairs = rng.uniform(0.0, 1.0, size=(sample_size, CONDITIONING_DIMENSION))
    price = 10.0 + 20.0 * pairs[:, 0]
    context = 2.0 * pairs[:, 1] - 1.0
    location, scale = demand_parameters(price, context)
    if shock_law == "normal":
        shock = rng.standard_normal(sample_size)
    elif shock_law == "lognormal":
        shock = rng.lognormal(size=sample_size)
    elif shock_law == "student_t3":
        shock = rng.standard_t(3, size=sample_size)
    else:
        raise ValueError(f"Unknown shock law: {shock_law}")
    demand = location + scale * shock
    neighbors, bandwidth, minimum_leaf_size = tuning(sample_size)
    rows = []

    started = time.perf_counter()
    spatial_tree = cKDTree(pairs)
    spatial_setup_seconds = time.perf_counter() - started

    for method in METHODS:
        started = time.perf_counter()
        if method == "kNN":
            cutoff_distances = spatial_tree.query(queries, k=[neighbors], workers=1)[0][:, 0]
            # Include all cutoff ties, matching Equation (7a) in the manuscript.
            radii = np.nextafter(cutoff_distances, np.inf)
            groups = spatial_tree.query_ball_point(queries, radii, workers=1)
            setup_seconds = spatial_setup_seconds
            leaf_count = np.nan
        elif method == "LSA":
            groups = spatial_tree.query_ball_point(queries, bandwidth, workers=1)
            setup_seconds = spatial_setup_seconds
            leaf_count = np.nan
        else:
            # Fit the usual regression CART to raw demand. Leaf membership defines
            # the CCW empirical distribution, without an extra residual model.
            cart = DecisionTreeRegressor(
                criterion="squared_error", min_samples_leaf=minimum_leaf_size,
                random_state=seed + replication, splitter="best",
            )
            cart.fit(pairs, demand)
            training_leaves = cart.apply(pairs)
            query_leaves = cart.apply(queries)
            leaf_groups = {
                leaf: np.flatnonzero(training_leaves == leaf)
                for leaf in np.unique(query_leaves)
            }
            groups = [leaf_groups[leaf] for leaf in query_leaves]
            setup_seconds = 0.0
            leaf_count = cart.get_n_leaves()
        groups = [np.asarray(indices, dtype=int) for indices in groups]
        diagnostics = cdf_error(demand, groups, truth)
        rows.append({
            "sample_size": sample_size,
            "replication": replication,
            "method": method,
            **diagnostics,
            "neighbors": neighbors if method == "kNN" else np.nan,
            "bandwidth": bandwidth if method == "LSA" else np.nan,
            "minimum_leaf_size": minimum_leaf_size if method == "CART" else np.nan,
            "leaf_count": leaf_count,
            "seconds": time.perf_counter() - started + setup_seconds,
        })
    return rows


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    summary = raw.groupby(["sample_size", "method"], as_index=False).agg(
        replications=("replication", "size"),
        complete_replications=("max_probability_error", "count"),
        mean_error=("max_probability_error", "mean"),
        standard_deviation=("max_probability_error", "std"),
        median_error=("max_probability_error", "median"),
        minimum_cluster_size=("minimum_cluster_size", "min"),
        mean_cluster_size=("mean_cluster_size", "mean"),
        mean_seconds=("seconds", "mean"),
    )
    summary["standard_error"] = summary.standard_deviation / np.sqrt(summary.complete_replications)
    quantiles = student_t.ppf(0.975, summary.complete_replications - 1)
    summary["ci95_half_width"] = quantiles * summary.standard_error
    summary["theory_shape"] = (np.log(summary.sample_size) / summary.sample_size) ** RATE_EXPONENT
    summary["fitted_rate_reference"] = np.nan
    for method, coefficient in fit_rate_coefficients(summary).items():
        indices = summary.index[summary.method.eq(method)]
        summary.loc[indices, "fitted_rate_reference"] = probability_fit(
            method, summary.loc[indices, "sample_size"].to_numpy(float), coefficient
        )
    return summary


def fit_saturating_coefficient(sample_sizes: np.ndarray, means: np.ndarray,
                               scale: float = 1.0) -> float:
    """Fit scale * (1-exp(-C*r_N)) using weights sqrt(N)."""
    shape = (np.log(sample_sizes) / sample_sizes) ** RATE_EXPONENT
    weights = np.sqrt(sample_sizes)
    normalized_means = means / scale
    implied_scales = -np.log1p(-np.minimum(normalized_means, 1.0 - 1e-15)) / shape
    upper = max(1.0, 2.0 * float(implied_scales.max()))
    fit = minimize_scalar(
        lambda coefficient: np.dot(
            weights, (-np.expm1(-coefficient * shape) - normalized_means) ** 2
        ),
        bounds=(0.0, upper), method="bounded", options={"xatol": 1e-12},
    )
    return float(fit.x)


def fit_rate_coefficients(summary: pd.DataFrame) -> dict[str, float]:
    """Fit saturated kNN and linear LSA curves using weight sqrt(N_i)."""
    coefficients = {}
    for method in ("kNN", "LSA"):
        panel = summary.loc[summary.method.eq(method)].sort_values("sample_size")
        sample_sizes = panel.sample_size.to_numpy(dtype=float)
        shape = (np.log(sample_sizes) / sample_sizes) ** RATE_EXPONENT
        means = panel.mean_error.to_numpy(dtype=float)
        if method == "kNN":
            coefficient = fit_saturating_coefficient(sample_sizes, means)
        else:
            weighted_shape = np.sqrt(sample_sizes) * shape
            coefficient = np.dot(weighted_shape, means) / np.dot(weighted_shape, shape)
        coefficients[method] = max(0.0, float(coefficient))
    return coefficients


def probability_fit(method: str, sample_sizes: np.ndarray, coefficient: float) -> np.ndarray:
    shape = (np.log(sample_sizes) / sample_sizes) ** RATE_EXPONENT
    if method == "kNN":
        return -np.expm1(-coefficient * shape)
    return coefficient * shape


def fit_objective_models(summary: pd.DataFrame) -> dict:
    sample_sizes = summary.sample_size.to_numpy(float)
    means = summary.mean_error.to_numpy(float)
    shape = (np.log(sample_sizes) / sample_sizes) ** RATE_EXPONENT
    weights = np.sqrt(sample_sizes)
    coefficient = float(np.dot(weights * shape, means) / np.dot(weights * shape, shape))
    saturated_coefficient = fit_saturating_coefficient(sample_sizes, means, LOSS_ERROR_BOUND)
    models = {
        "linear": (coefficient, coefficient * shape),
        "saturated": (saturated_coefficient,
                      -LOSS_ERROR_BOUND * np.expm1(-saturated_coefficient * shape)),
    }
    result = {"selected_form": "C*r_N", "fixed_error_range_bound": LOSS_ERROR_BOUND}
    for name, (constant, predictions) in models.items():
        residuals = predictions - means
        result[name] = {
            "coefficient": constant,
            "rmse": float(np.sqrt(np.mean(residuals ** 2))),
            "weighted_rmse": float(np.sqrt(np.dot(weights, residuals ** 2) / weights.sum())),
            "absolute_residuals": np.abs(residuals).tolist(),
        }
    return result


def plot_metadata(summary: pd.DataFrame, objective: pd.DataFrame | None = None,
                  objective_path: Path | None = None) -> dict:
    metadata = {
        "reference": "Probability: kNN uses 1-exp(-C*r_N), LSA uses C*r_N, with r_N=(log(N)/N)^(1/6). Each nonnegative coefficient is fitted to all sample-size means by weighted least squares with weights sqrt(N_i).",
        "reference_fit_forms": {"kNN": "1-exp(-C*r_N)", "LSA": "C*r_N"},
        "reference_fit_coefficients": fit_rate_coefficients(summary),
        "reference_fit_sample_sizes": sorted(int(n) for n in summary.sample_size.unique()),
        "reference_fit_objective": "minimize sum_i sqrt(N_i) * (mean_error_i - fitted_curve(C,r_i))^2 over C>=0, separately by method",
        "reference_fit_weights": {str(int(n)): float(np.sqrt(n)) for n in sorted(summary.sample_size.unique())},
        "reference_data_source": "sample_size and mean_error columns; previously archived baseline_scaled_reference values are unused",
        "plot_axes": "linear x and y axes; the probability panel includes a linear small-sample inset",
        "observations": "Mean errors, connecting lines, and 95% confidence intervals for kNN, LSA, and CART",
        "interpretation": "fixed-grid descriptive fits. The kNN probability curve is bounded by one and asymptotic to C*r_N as r_N tends to zero; it is a descriptive same-order model. CART remains an empirical comparator.",
        "objective_panel": None,
    }
    if objective is not None:
        metadata["objective_panel"] = {
            "summary_csv": str(objective_path),
            "methods": ["kNN"],
            "weights": "sqrt(N_i)",
            "sample_sizes": objective.sample_size.astype(int).tolist(),
            "model_comparison": fit_objective_models(objective),
        }
    return metadata


def make_plot(summary: pd.DataFrame, output: Path, stem: str,
              objective: pd.DataFrame | None = None) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9.5,
        "axes.labelsize": 9.5, "legend.fontsize": 8.2,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    colors = {"kNN": "#2166ac", "LSA": "#b35806", "CART": "#4d4d4d"}
    markers = {"kNN": "o", "LSA": "^", "CART": "s"}
    coefficients = fit_rate_coefficients(summary)
    if objective is None:
        fig, ax = plt.subplots(figsize=(6.7, 3.7))
        fig.subplots_adjust(left=0.11, right=0.98, bottom=0.17, top=0.82)
    else:
        fig, (ax, loss_axis) = plt.subplots(1, 2, figsize=(8.5, 3.8))
        fig.subplots_adjust(left=0.075, right=0.98, bottom=0.17, top=0.82, wspace=0.30)
    small_samples = summary.loc[summary.sample_size.le(10000)]
    panels = [(ax, summary, 4.1, 1.4)]
    if not small_samples.empty:
        inset_rect = [0.51, 0.60, 0.465, 0.365] if objective is None else [0.45, 0.63, 0.52, 0.32]
        inset = ax.inset_axes(inset_rect)
        panels.append((inset, small_samples, 2.5, 0.9))
    for axis, data, marker_size, line_width in panels:
        for method in METHODS:
            panel = data.loc[data.method.eq(method)].sort_values("sample_size")
            axis.errorbar(panel.sample_size, panel.mean_error, yerr=panel.ci95_half_width,
                          color=colors[method], marker=markers[method], markersize=marker_size,
                          capsize=2.0, linewidth=line_width,
                          linestyle="-", label=method)
        for method in ("kNN", "LSA"):
            panel = data.loc[data.method.eq(method)].sort_values("sample_size")
            reference_sizes = np.geomspace(panel.sample_size.min(), panel.sample_size.max(), 500)
            axis.plot(reference_sizes, probability_fit(method, reference_sizes, coefficients[method]),
                      color=colors[method], linestyle="--", linewidth=line_width,
                      label=f"{method} weighted fit")
        axis.set_xscale("linear")
        axis.set_yscale("linear")
        axis.grid(which="major", color="#dddddd", linewidth=0.5)
    ax.set_xlabel(r"Historical sample size $N$")
    ax.set_ylabel("Mean maximum-grid CDF error")
    ax.set_xlim(-4000, 105000)
    if objective is None:
        ax.set_ylim(0.25, 1.04)
        ax.set_xticks(np.arange(0, 100001, 20000))
        ax.xaxis.set_major_formatter(lambda value, _: f"{value:,.0f}")
        ax.set_yticks(np.arange(0.3, 1.01, 0.1))
    else:
        ax.set_ylim(0.25, 0.86)
        ax.set_xticks([0, 50000, 100000], ["0", "50,000", "100,000"])
        ax.set_yticks(np.arange(0.3, 0.81, 0.1))
        ax.set_title("(a) Probability estimation", pad=9)
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.2f}")
    ax.tick_params(direction="out", labelsize=8)
    handles, labels = ax.get_legend_handles_labels()
    legend_order = [labels.index(label) for label in (
        "kNN", "LSA", "CART", "kNN weighted fit", "LSA weighted fit"
    )]
    fig.legend([handles[i] for i in legend_order], [labels[i] for i in legend_order],
              frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.53, 0.99),
              columnspacing=1.25, handlelength=1.9)
    if not small_samples.empty:
        inset.set_xlim(500, 10500)
        inset.set_ylim(0.38, 1.00 if objective is None else 0.82)
        inset.set_xticks([1000, 2000, 5000, 10000], ["1", "2", "5", "10"])
        inset.set_yticks(np.arange(0.4, 1.01 if objective is None else 0.81, 0.2))
        inset.yaxis.set_major_formatter(lambda value, _: f"{value:.1f}")
        inset.text(0.97, 0.94, r"$N$ (thousands)", transform=inset.transAxes,
                   ha="right", va="top", fontsize=6.5)
        inset.tick_params(direction="out", labelsize=6.3, pad=1.2)
    if objective is not None:
        loss_models = fit_objective_models(objective)
        coefficient = loss_models["linear"]["coefficient"]
        loss_axis.errorbar(objective.sample_size, objective.mean_error,
                           yerr=objective.ci95_half_width, color=colors["kNN"],
                           marker=markers["kNN"], markersize=4.1, capsize=2.0,
                           linewidth=1.4, linestyle="none")
        sizes = np.geomspace(objective.sample_size.min(), objective.sample_size.max(), 500)
        shape = (np.log(sizes) / sizes) ** RATE_EXPONENT
        loss_axis.plot(sizes, coefficient * shape, color=colors["kNN"], linestyle="--", linewidth=1.4)
        loss_axis.set_xscale("linear")
        loss_axis.set_yscale("linear")
        loss_axis.set_xlim(-4000, 105000)
        loss_axis.set_xticks([0, 50000, 100000], ["0", "50,000", "100,000"])
        loss_axis.set_ylim(150, 400)
        loss_axis.set_yticks([150, 200, 250, 300, 350, 400])
        loss_axis.set_xlabel(r"Historical sample size $N$")
        loss_axis.set_ylabel("Mean maximum-grid expected-loss error")
        loss_axis.set_title("(b) Expected-loss estimation", pad=9)
        loss_axis.tick_params(direction="out", labelsize=8)
        loss_axis.grid(which="major", color="#dddddd", linewidth=0.5)
    fig.savefig(output / f"{stem}.pdf", metadata={"Title": "Contextual CCW estimation diagnostics"})
    fig.savefig(output / f"{stem}.png", dpi=220)
    plt.close(fig)


def load_objective_summary(path: Path | None,
                           probability: pd.DataFrame) -> tuple[pd.DataFrame | None, Path | None]:
    if path is None:
        return None, None
    objective = pd.read_csv(path)
    objective = objective.loc[objective.method.eq("kNN")].sort_values("sample_size")
    expected_sizes = sorted(probability.sample_size.unique())
    if objective.sample_size.tolist() != expected_sizes:
        raise ValueError("Expected-loss and probability summaries must use the same sample sizes.")
    probability_replications = probability.loc[probability.method.eq("kNN")].sort_values("sample_size").replications
    if not np.array_equal(objective.replications.to_numpy(), probability_replications.to_numpy()):
        raise ValueError("Expected-loss and probability summaries must use the same replication counts.")
    return objective, path


def write_design(output: Path, stem: str, args: argparse.Namespace, summary: pd.DataFrame,
                  objective: pd.DataFrame | None = None, objective_path: Path | None = None) -> None:
    design = {
        "seed": args.seed, "sample_sizes": args.sample_sizes,
        "replications": args.replications,
        "conditioning_dimension": CONDITIONING_DIMENSION,
        "historical_pairs": "P~Uniform[10,30], X~Uniform[-1,1], independent",
        "conditional_demand": "D|P=p,X=x ~ Normal(90-2p+12x, (6+2x)^2), untruncated",
        "scaled_pair": "((p-10)/20, (x+1)/2), Euclidean distance",
        "price_grid": PRICE_GRID.tolist(), "context_grid": CONTEXT_GRID.tolist(),
        "threshold_grid": THRESHOLD_GRID.tolist(),
        "error": "max over all 15*9 conditioning pairs and 61 thresholds of |Fhat-F|",
        "truth": "F(d|p,x)=Phi((d-(90-2p+12x))/(6+2x))",
        "knn": "k=ceil(N*(log(N)/N)^(1/3)); include all observations tied at cutoff",
        "lsa": "h=0.6*(log(N)/N)^(1/6); closed Euclidean ball",
        "cart": "squared-error demand regression CART; min_samples_leaf=ceil(k/2); other sklearn defaults",
        "cart_selection": "leaf-size schedule fixed before simulation; no validation or tuning on evaluation probabilities",
        "tuning_by_sample_size": {
            n: {"knn_neighbors": tuning(n)[0], "lsa_bandwidth": tuning(n)[1],
                "cart_minimum_leaf_size": tuning(n)[2]} for n in args.sample_sizes
        },
        "rate_derivation": "a_N order sqrt(log(N)/N); kNN error order N*a_N/k+(k/N)^(1/d_c); LSA error order a_N/h^d_c+h; balanced order (log(N)/N)^(1/(2*(d_c+1)))",
        "confidence_interval": "two-sided 95% Student t interval for mean replication maximum-grid error",
        "sampling": "independent histories across (N,replication), shared across methods; SeedSequence([seed,N,replication])",
        "empty_clusters": "record unavailable queries; max-grid error missing if any query unavailable; report complete replication count",
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "pandas": pd.__version__,
                     "matplotlib": matplotlib.__version__, "scikit_learn": sklearn.__version__},
    }
    design.update(plot_metadata(summary, objective, objective_path))
    (output / f"{stem}_design.json").write_text(json.dumps(design, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "assets" / "data")
    parser.add_argument("--replications", type=int, default=REPLICATIONS)
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=list(SAMPLE_SIZES))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--stem", default=FILE_STEM)
    parser.add_argument("--objective-summary", type=Path,
                        help="Explicitly add a panel from a kNN expected-loss summary.")
    parser.add_argument("--plot-only", action="store_true",
                        help="Refit and regenerate PDF, PNG, and metadata from existing summaries.")
    args = parser.parse_args()
    if args.plot_only:
        summary_path = args.output_dir / f"{args.stem}_summary.csv"
        summary = pd.read_csv(summary_path)
        objective, objective_path = load_objective_summary(args.objective_summary, summary)
        make_plot(summary, args.output_dir, args.stem, objective)
        metadata_path = args.output_dir / f"{args.stem}_design.json"
        if metadata_path.exists():
            design = json.loads(metadata_path.read_text(encoding="utf-8"))
            design.update(plot_metadata(summary, objective, objective_path))
            metadata_path.write_text(json.dumps(design, indent=2) + "\n", encoding="utf-8")
        print(f"Weighted least-squares coefficients: {fit_rate_coefficients(summary)}", flush=True)
        if objective is not None:
            print(f"Expected-loss fit comparison: {fit_objective_models(objective)}", flush=True)
        print(f"Regenerated PDF and PNG from {summary_path}", flush=True)
        return
    if args.replications < 2 or any(n < 2 for n in args.sample_sizes):
        parser.error("Use at least two replications and sample sizes of at least two.")
    args.sample_sizes = sorted(set(args.sample_sizes))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    queries, truth = query_grid()
    rows = []
    started = time.perf_counter()
    for sample_size in args.sample_sizes:
        for replication in range(args.replications):
            rows.extend(one_replication(sample_size, replication, args.seed, queries, truth))
        print(f"Finished N={sample_size:,}, {args.replications} replications, "
              f"{time.perf_counter() - started:.1f}s elapsed", flush=True)
    raw = pd.DataFrame(rows)
    summary = summarize(raw)
    raw.to_csv(args.output_dir / f"{args.stem}_replications.csv", index=False)
    summary.to_csv(args.output_dir / f"{args.stem}_summary.csv", index=False)
    objective, objective_path = load_objective_summary(args.objective_summary, summary)
    write_design(args.output_dir, args.stem, args, summary, objective, objective_path)
    make_plot(summary, args.output_dir, args.stem, objective)
    print(summary[["sample_size", "method", "mean_error", "ci95_half_width",
                   "complete_replications", "minimum_cluster_size"]].to_string(index=False))
    print(f"Results written to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
