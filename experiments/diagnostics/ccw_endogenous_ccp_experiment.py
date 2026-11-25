#!/usr/bin/env python3
"""Reproducible CCW experiments for chance-constrained endogenous uncertainty.

This script implements the price-setting newsvendor in RCCP-DDU-POMS and
compares CCW, CQ-CCP, P-CCP, global SAA-CCP, and a true-distribution Oracle.
The default DGP is deliberately nonparametric/misspecified for the P-CCP
benchmark: it contains price-context interactions and conditional scale
heterogeneity absent from P-CCP's additive location-scale specification.

Design safeguards added in this version
---------------------------------------
1. Endogeneity strength is varied without collapsing the demand level:
       m_rho(p,x) = m(p0,x) + rho [m(p,x)-m(p0,x)],
   with p0=10 by default.  Hence rho=0 is price-exogenous at a non-degenerate
   demand level and rho=1 is exactly the manuscript DGP.
2. An estimated-infeasible contextual policy is recorded as infeasible; it is
   never replaced by a least-violating action or included in the main figures.
3. The CCW neighbourhood size and the relative price weight in the joint
   (p,x) distance are selected using a held-out empirical CRPS.
4. Mode 2 includes its quadratic mean term from the manuscript.
5. The main figure fixes alpha=0.10.  The risk-return frontier has no single
   reliability target line because each point has a different alpha.

Quick executable check:
    MPLCONFIGDIR=/tmp/mpl python ccw_endogenous_ccp_experiment.py --smoke-test
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Iterable, Sequence

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


METHODS = ("CCW", "CQ-CCP", "P-CCP", "SAA-CCP", "Oracle")
COLORS = {"CCW": "#0072B2", "CQ-CCP": "#D55E00", "P-CCP": "#009E73",
          "SAA-CCP": "#CC79A7", "Oracle": "#222222"}
ALPHA_MARKERS = ("o", "s", "^", "v", "D", "P")


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 20260805
    n_replications: int = 50
    n_train: int = 6000
    n_validation: int = 1200
    n_test_contexts: int = 80
    n_eval_mc: int = 2500
    # The DGP uses five observed predictive covariates.  Adding artificial,
    # irrelevant coordinates to a distance estimator is not a fair test of
    # action--context overlap.
    context_dim: int = 5
    price_low: float = 10.0
    price_high: float = 30.0
    # Historical actions and candidate decisions share this support.  With
    # 41 price levels, 6,000 observations provide tail-estimation overlap;
    # the former 200-point grid provided only about 15 observations/price.
    price_step: float = 0.5
    unit_cost: float = 5.0
    salvage_value: float = 2.0
    profit_target: float = 100.0
    endogeneity_reference_price: float = 10.0
    effective_alphas: tuple[float, ...] = (0.02, 0.05, 0.10, 0.15, 0.20, 0.30)
    main_alpha: float = 0.10
    rhos: tuple[float, ...] = (0.0, 0.25, 0.50, 0.75, 1.0)
    k_candidates: tuple[int, ...] = (100, 150, 225)
    action_weight_candidates: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 12.0)
    cq_levels: tuple[float, ...] = (0.01, 0.02, 0.05, 0.10, 0.20, 0.30,
                                    0.40, 0.50, 0.60, 0.70, 0.80, 0.90,
                                    0.95, 0.98, 0.99)
    dgp_mode: int = 1
    dgp_family: str = "misspecified"  # {"misspecified", "parametric"}; primary case is interaction-misspecified

    @property
    def prices(self) -> np.ndarray:
        return np.round(np.arange(self.price_low, self.price_high + 1e-9,
                                  self.price_step), 10)


@dataclass(frozen=True)
class Decision:
    price: float = np.nan
    order: float = np.nan
    estimated_reliability: float = np.nan
    feasible: bool = False


def correlated_context(rng: np.random.Generator, n: int, dim: int) -> np.ndarray:
    covariance = 0.5 ** np.abs(np.subtract.outer(np.arange(dim), np.arange(dim)))
    return rng.multivariate_normal(np.zeros(dim), covariance, size=n)


def _mean_base(p: np.ndarray, x: np.ndarray, mode: int, cfg: ExperimentConfig) -> np.ndarray:
    beta_x = np.array([-2, -1, 0, 1, 2][:cfg.context_dim], dtype=float) / np.sqrt(10)
    mean = 200.0 - 10.0 * p + x @ beta_x
    if mode == 2:
        mean = mean - 0.2 * p ** 2
    if cfg.dgp_family == "misspecified":
        # Smooth price-context interactions vanish at p0=10, so rho=0 remains
        # a non-degenerate exogenous benchmark while rho=1 is not additive.
        dp = p - cfg.endogeneity_reference_price
        # Deliberately non-additive, decision-dependent response.  Its effect is
        # multiplied by rho below, so CCW's advantage is tested only as
        # endogeneity grows; an additive P-CCP cannot represent this interaction.
        mean += -4.0 * dp * np.tanh(x[:, 0]) + 5.0 * np.sin(0.60 * dp) * np.tanh(x[:, 1])
    return mean


def _sd_base(p: np.ndarray, x: np.ndarray, cfg: ExperimentConfig) -> np.ndarray:
    gamma_x = np.ones(cfg.context_dim, dtype=float) / np.sqrt(5)
    sd = 20.0 - 2.1 * p + 0.2 * p ** 2 + x @ gamma_x
    if cfg.dgp_family == "misspecified":
        dp = np.abs(p - cfg.endogeneity_reference_price)
        sd += 2.0 * dp / (1.0 + np.exp(-x[:, 2]))
    return sd


def _mode1_parameters(p: np.ndarray, x: np.ndarray, rho: float,
                      cfg: ExperimentConfig) -> tuple[np.ndarray, np.ndarray]:
    """Location and scale for the corrected Mode-1 interpolation."""
    p = np.asarray(p, dtype=float).reshape(-1)
    p0 = np.full_like(p, cfg.endogeneity_reference_price)
    mu0 = _mean_base(p0, x, mode=1, cfg=cfg)
    sd0 = _sd_base(p0, x, cfg)
    mu = mu0 + rho * (_mean_base(p, x, mode=1, cfg=cfg) - mu0)
    sd = np.maximum(1e-5, sd0 + rho * (_sd_base(p, x, cfg) - sd0))
    return mu, sd


def demand_from_dgp(p: np.ndarray, x: np.ndarray, u: np.ndarray, rho: float,
                    cfg: ExperimentConfig) -> np.ndarray:
    """Demand samples. Rows correspond to price/context pairs; columns to draws."""
    p = np.asarray(p, dtype=float).reshape(-1)
    x = np.asarray(x, dtype=float)
    p0 = np.full_like(p, cfg.endogeneity_reference_price)
    if cfg.dgp_mode == 1:
        mu, sd = _mode1_parameters(p, x, rho, cfg)
        return np.maximum(0.0, mu[:, None] + sd[:, None] * u)
    # Relationship Mode 2 in the manuscript, including beta'_1 p^2.
    mu0 = _mean_base(p0, x, mode=2, cfg=cfg)
    mu = mu0 + rho * (_mean_base(p, x, mode=2, cfg=cfg) - mu0)
    neg0, negp = 20.0 - 2.1 * p0, 20.0 - 2.1 * p
    pos0, posp = 0.2 * p0 ** 2, 0.2 * p ** 2
    a_neg, a_pos = neg0 + rho * (negp - neg0), pos0 + rho * (posp - pos0)
    return np.maximum(0.0, mu[:, None] + a_neg[:, None] * np.minimum(u, 0)
                      + a_pos[:, None] * np.maximum(u, 0))


def generate_historical_data(rng: np.random.Generator, n: int, rho: float,
                             cfg: ExperimentConfig) -> pd.DataFrame:
    price = rng.choice(cfg.prices, size=n)
    context = correlated_context(rng, n, cfg.context_dim)
    demand = demand_from_dgp(price, context, rng.standard_normal((n, 1)), rho, cfg)[:, 0]
    data = pd.DataFrame({"p": price, "demand": demand})
    for j in range(cfg.context_dim):
        data[f"x{j}"] = context[:, j]
    return data


def profit(price: np.ndarray, order: np.ndarray, demand: np.ndarray,
           cfg: ExperimentConfig) -> np.ndarray:
    return ((price[..., None] - cfg.unit_cost) * order[..., None]
            - (price[..., None] - cfg.salvage_value)
            * np.maximum(order[..., None] - demand, 0.0))


def scenario_decision(samples: np.ndarray, price: float, alpha: float,
                      cfg: ExperimentConfig) -> tuple[float, float, float] | None:
    """Optimal q and estimated performance for a fixed p under a sample law.

    Returns None when the estimated chance constraint has no feasible order
    quantity.  In particular, no least-violating fallback is used.
    """
    samples = np.asarray(samples, dtype=float)
    q_min = cfg.profit_target / (price - cfg.unit_cost)
    q_alpha = float(np.quantile(samples, alpha, method="higher"))
    q_max = ((price - cfg.salvage_value) * q_alpha - cfg.profit_target) / (cfg.unit_cost - cfg.salvage_value)
    if q_max < q_min - 1e-10:
        return None
    critical_ratio = (price - cfg.unit_cost) / (price - cfg.salvage_value)
    q_unconstrained = float(np.quantile(samples, critical_ratio, method="linear"))
    order = float(np.clip(q_unconstrained, q_min, q_max))
    pi = profit(np.array([price]), np.array([order]), samples.reshape(1, -1), cfg)[0]
    return order, float(pi.mean()), float((pi >= cfg.profit_target).mean())


def best_from_sampler(context: np.ndarray, alpha: float, prices: np.ndarray,
                      sampler: Callable[[float, np.ndarray], np.ndarray],
                      cfg: ExperimentConfig) -> Decision:
    candidates: list[tuple[float, float, float, float]] = []
    for price in prices:
        result = scenario_decision(sampler(float(price), context), float(price), alpha, cfg)
        if result is not None:
            order, expected_profit, estimated_reliability = result
            candidates.append((float(price), order, expected_profit, estimated_reliability))
    if not candidates:
        return Decision()
    price, order, _, estimated_reliability = max(candidates, key=lambda z: z[2])
    return Decision(price, order, estimated_reliability, True)


class CCWModel:
    def __init__(self, data: pd.DataFrame, cfg: ExperimentConfig, k: int,
                 action_weight: float = 1.0):
        self.cfg, self.k = cfg, min(int(k), len(data))
        self.action_weight = float(action_weight)
        self.xcols = [f"x{j}" for j in range(cfg.context_dim)]
        z = data[["p", *self.xcols]].to_numpy(dtype=float)
        self.location, self.scale = z.mean(axis=0), z.std(axis=0) + 1e-8
        normalized_z = (z - self.location) / self.scale
        # A separate multiplicative factor for p preserves the joint (p,x)
        # neighbourhood while preventing ten context coordinates from masking
        # economically material price differences.
        normalized_z[:, 0] *= self.action_weight
        self.nn = NearestNeighbors(n_neighbors=self.k).fit(normalized_z)
        self.demands = data["demand"].to_numpy(dtype=float)

    def scenarios(self, price: float, context: np.ndarray) -> np.ndarray:
        query = (np.r_[price, context] - self.location) / self.scale
        query[0] *= self.action_weight
        idx = self.nn.kneighbors(query.reshape(1, -1), return_distance=False)[0]
        return self.demands[idx]

    def prescribe(self, context: np.ndarray, alpha: float) -> Decision:
        return best_from_sampler(context, alpha, self.cfg.prices, self.scenarios, self.cfg)

    def validation_crps(self, validation: pd.DataFrame) -> float:
        """Empirical CRPS on held-out (p,x,d) observations for selecting k."""
        score = 0.0
        for row in validation.itertuples(index=False):
            context = np.array([getattr(row, f"x{j}") for j in range(self.cfg.context_dim)])
            samples = self.scenarios(float(row.p), context)
            first = np.mean(np.abs(samples - row.demand))
            second = 0.5 * np.mean(np.abs(samples[:, None] - samples[None, :]))
            score += first - second
        return score / len(validation)


def select_ccw_hyperparameters(train: pd.DataFrame, validation: pd.DataFrame,
                               cfg: ExperimentConfig) -> tuple[int, float, dict[tuple[int, float], float]]:
    """Jointly choose k and the price-coordinate weight by held-out CRPS."""
    scores: dict[tuple[int, float], float] = {}
    for action_weight in cfg.action_weight_candidates:
        for k in cfg.k_candidates:
            scores[(k, action_weight)] = CCWModel(train, cfg, k, action_weight).validation_crps(validation)
    selected_k, selected_weight = min(scores, key=scores.get)
    return selected_k, selected_weight, scores


class ParametricModel:
    """P-CCP benchmark: additive conditional mean plus *global* residual law.

    This is the residual-SAA parametric benchmark, not a second flexible
    conditional-distribution learner.  Thus it is correctly specified in the
    parametric calibration case and deliberately misspecified in the primary
    price--context interaction case.
    """
    def __init__(self, data: pd.DataFrame, cfg: ExperimentConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.xcols = [f"x{j}" for j in range(cfg.context_dim)]
        x = self.features(data["p"].to_numpy(), data[self.xcols].to_numpy())
        self.mean_model = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(x, data["demand"])
        residual = data["demand"].to_numpy() - self.mean_model.predict(x)
        self.residual_draws = rng.choice(residual, size=1600, replace=True)

    def features(self, price: np.ndarray, context: np.ndarray) -> np.ndarray:
        p = np.asarray(price, dtype=float).reshape(-1, 1)
        return np.column_stack([p, p ** 2, context])

    def scenarios(self, price: float, context: np.ndarray) -> np.ndarray:
        x = self.features(np.array([price]), context.reshape(1, -1))
        mean = self.mean_model.predict(x)[0]
        return np.maximum(0.0, mean + self.residual_draws)

    def prescribe(self, context: np.ndarray, alpha: float) -> Decision:
        return best_from_sampler(context, alpha, self.cfg.prices, self.scenarios, self.cfg)


class CQModel:
    """Conditional-quantile CCP using a monotone grid of fitted quantiles."""
    def __init__(self, data: pd.DataFrame, cfg: ExperimentConfig, seed: int):
        self.cfg, self.levels = cfg, np.asarray(cfg.cq_levels)
        self.xcols = [f"x{j}" for j in range(cfg.context_dim)]
        x = self.features(data["p"].to_numpy(), data[self.xcols].to_numpy())
        self.models = [
            GradientBoostingRegressor(loss="quantile", alpha=float(level), n_estimators=100,
                max_depth=2, learning_rate=0.05, min_samples_leaf=15, random_state=seed + i).fit(x, data["demand"])
            for i, level in enumerate(self.levels)
        ]
        self.probability_grid = np.linspace(self.levels[0], self.levels[-1], 1001)

    def features(self, price: np.ndarray, context: np.ndarray) -> np.ndarray:
        p = np.asarray(price, dtype=float).reshape(-1, 1)
        return np.column_stack([p, p ** 2, context])

    def scenarios(self, price: float, context: np.ndarray) -> np.ndarray:
        x = self.features(np.array([price]), context.reshape(1, -1))
        qgrid = np.array([model.predict(x)[0] for model in self.models])
        qgrid = np.maximum.accumulate(qgrid)  # removes quantile crossing
        return np.maximum(0.0, np.interp(self.probability_grid, self.levels, qgrid))

    def prescribe(self, context: np.ndarray, alpha: float) -> Decision:
        return best_from_sampler(context, alpha, self.cfg.prices, self.scenarios, self.cfg)


class SAAModel:
    def __init__(self, data: pd.DataFrame, cfg: ExperimentConfig):
        self.demands, self.cfg = data["demand"].to_numpy(dtype=float), cfg

    def scenarios(self, price: float, context: np.ndarray) -> np.ndarray:
        return self.demands

    def prescribe(self, context: np.ndarray, alpha: float) -> Decision:
        return best_from_sampler(context, alpha, self.cfg.prices, self.scenarios, self.cfg)


def expected_profit_mode1(price: float, order: float, mu: float, sd: float,
                          cfg: ExperimentConfig) -> float:
    """Exact E[profit] when D=max(0, Normal(mu,sd))."""
    a0, aq = -mu / sd, (order - mu) / sd
    f0, fq = norm.cdf(a0), norm.cdf(aq)
    expected_leftover = ((order - mu) * fq + mu * f0
                         + sd * (norm.pdf(aq) - norm.pdf(a0)))
    return (price - cfg.unit_cost) * order - (price - cfg.salvage_value) * expected_leftover


class OracleModel:
    """Analytic Oracle for the default truncated-normal (Mode-1) DGP."""
    def __init__(self, rho: float, cfg: ExperimentConfig):
        if cfg.dgp_mode != 1:
            raise ValueError("The analytic Oracle is implemented for dgp_mode=1 only.")
        self.rho, self.cfg = rho, cfg

    def prescribe(self, context: np.ndarray, alpha: float) -> Decision:
        candidates: list[tuple[float, float, float, float]] = []
        for price in self.cfg.prices:
            mu, sd = _mode1_parameters(np.array([price]), context.reshape(1, -1), self.rho, self.cfg)
            mu, sd = float(mu[0]), float(sd[0])
            q_min = self.cfg.profit_target / (price - self.cfg.unit_cost)
            demand_q_alpha = max(0.0, mu + sd * norm.ppf(alpha))
            q_max = ((price - self.cfg.salvage_value) * demand_q_alpha - self.cfg.profit_target) / (self.cfg.unit_cost - self.cfg.salvage_value)
            if q_max < q_min - 1e-10:
                continue
            critical_ratio = (price - self.cfg.unit_cost) / (price - self.cfg.salvage_value)
            q_unconstrained = max(0.0, mu + sd * norm.ppf(critical_ratio))
            order = float(np.clip(q_unconstrained, q_min, q_max))
            threshold = (self.cfg.profit_target + (self.cfg.unit_cost - self.cfg.salvage_value) * order) / (price - self.cfg.salvage_value)
            reliability = float(norm.sf((threshold - mu) / sd))
            objective = expected_profit_mode1(float(price), order, mu, sd, self.cfg)
            candidates.append((float(price), order, objective, reliability))
        if not candidates:
            return Decision()
        price, order, _, reliability = max(candidates, key=lambda z: z[2])
        return Decision(price, order, reliability, True)


def evaluate_policy(decisions: Sequence[Decision], context: np.ndarray, rho: float,
                    alpha: float, cfg: ExperimentConfig, u: np.ndarray) -> dict[str, float]:
    feasible = np.array([d.feasible for d in decisions], dtype=bool)
    base = {
        "estimated_context_feasibility": float(feasible.mean()),
        "infeasible_context_rate": float(1.0 - feasible.mean()),
        "policy_feasible": bool(feasible.all()),
    }
    if not feasible.all():
        return {**base, "expected_profit": np.nan, "reliability": np.nan,
                "true_coverage": np.nan, "shortfall": np.nan,
                "estimated_reliability": np.nan, "calibration_mae": np.nan}
    price = np.array([d.price for d in decisions])
    order = np.array([d.order for d in decisions])
    estimated = np.array([d.estimated_reliability for d in decisions])
    demand = demand_from_dgp(price, context, u, rho, cfg)
    pi = profit(price, order, demand, cfg)
    reliability_by_context = (pi >= cfg.profit_target).mean(axis=1)
    return {
        **base,
        "expected_profit": float(pi.mean()),
        "reliability": float(reliability_by_context.mean()),
        "true_coverage": float(np.mean(reliability_by_context >= 1.0 - alpha)),
        "shortfall": float(np.mean(np.maximum(1.0 - alpha - reliability_by_context, 0.0))),
        "estimated_reliability": float(estimated.mean()),
        "calibration_mae": float(np.mean(np.abs(estimated - reliability_by_context))),
    }


def run_one_replication(cfg: ExperimentConfig, rho: float, rep: int) -> list[dict]:
    rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, int(1000 * rho), rep]))
    train = generate_historical_data(rng, cfg.n_train, rho, cfg)
    validation = generate_historical_data(rng, cfg.n_validation, rho, cfg)
    context = correlated_context(rng, cfg.n_test_contexts, cfg.context_dim)
    u_eval = rng.standard_normal((cfg.n_test_contexts, cfg.n_eval_mc))
    selected_k, selected_weight, crps = select_ccw_hyperparameters(train, validation, cfg)
    models = {
        "CCW": CCWModel(train, cfg, selected_k, selected_weight),
        "CQ-CCP": CQModel(train, cfg, seed=10_000 + rep),
        "P-CCP": ParametricModel(train, cfg, rng),
        "SAA-CCP": SAAModel(train, cfg),
        "Oracle": OracleModel(rho, cfg),
    }
    rows: list[dict] = []
    for alpha in cfg.effective_alphas:
        oracle_decisions = [models["Oracle"].prescribe(x, alpha) for x in context]
        if not all(d.feasible for d in oracle_decisions):
            raise RuntimeError(
                f"Oracle is infeasible at rho={rho}, alpha={alpha}. "
                "Adjust the DGP, profit target, or decision range before comparing methods.")
        for name, model in models.items():
            decisions = oracle_decisions if name == "Oracle" else [model.prescribe(x, alpha) for x in context]
            metrics = evaluate_policy(decisions, context, rho, alpha, cfg, u_eval)
            valid = [d for d in decisions if d.feasible]
            rows.append({
                "rho": rho, "replication": rep, "method": name, "alpha": alpha,
                "k_ccw": selected_k, "ccw_action_weight": selected_weight,
                "ccw_validation_crps": crps[(selected_k, selected_weight)],
                **metrics,
                "mean_price": float(np.mean([d.price for d in valid])) if valid else np.nan,
                "mean_order": float(np.mean([d.order for d in valid])) if valid else np.nan,
            })
    return rows


def _aggregate(results: pd.DataFrame) -> pd.DataFrame:
    summary = results.groupby(["rho", "method", "alpha"], as_index=False).agg(
        expected_profit=("expected_profit", "mean"), reliability=("reliability", "mean"),
        profit_se=("expected_profit", "sem"), reliability_se=("reliability", "sem"),
        true_coverage=("true_coverage", "mean"), shortfall=("shortfall", "mean"),
        calibration_mae=("calibration_mae", "mean"), calibration_mae_se=("calibration_mae", "sem"),
        policy_feasible_rate=("policy_feasible", "mean"),
        estimated_context_feasibility=("estimated_context_feasibility", "mean"),
        infeasible_context_rate=("infeasible_context_rate", "mean"),
        n_replications=("replication", "size"),
        n_successful_replications=("expected_profit", "count"),
    )
    summary["reliability_lcb"] = summary["reliability"] - 1.96 * summary["reliability_se"]
    summary["reliability_margin"] = summary["reliability"] - (1.0 - summary["alpha"])
    summary["empirically_risk_adequate"] = summary["reliability_lcb"] >= (1.0 - summary["alpha"])
    # Performance statistics are conditional on a complete estimated policy.
    # The denominator is exposed explicitly; never interpret these means as
    # unconditional performance when a replication is infeasible.
    summary["conditional_performance"] = True
    return summary


def _axes_grid(n: int, width: float = 4.1, height: float = 3.65):
    cols, rows = 3, int(np.ceil(n / 3))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * width, rows * height), squeeze=False)
    return fig, axes.ravel()


def plot_fixed_alpha_comparison(summary: pd.DataFrame, cfg: ExperimentConfig, output: Path) -> None:
    z = summary[np.isclose(summary.alpha, cfg.main_alpha)]
    fig, axes = _axes_grid(len(cfg.rhos))
    target = 1.0 - cfg.main_alpha
    for ax, rho in zip(axes, cfg.rhos):
        panel = z[z.rho == rho]
        plotted = panel.dropna(subset=["reliability", "expected_profit"])
        xlo = max(0.0, min(target - 0.08, plotted.reliability.min() - 0.03) if len(plotted) else target - .08)
        ax.axvspan(xlo, target, color="#FEE8E7", alpha=.75, zorder=0)
        ax.axvline(target, color="0.35", ls="--", lw=1)
        for method in METHODS:
            point = plotted[plotted.method == method]
            if len(point):
                row = point.iloc[0]
                adequate = bool(row.empirically_risk_adequate)
                ax.errorbar(row.reliability, row.expected_profit, xerr=1.96 * row.reliability_se,
                            yerr=1.96 * row.profit_se, fmt="o", ms=6, color=COLORS[method],
                            mfc=COLORS[method] if adequate else "white", mew=1.8,
                            capsize=2.5, label=method)
                ax.annotate(f"{100 * row.policy_feasible_rate:.0f}%", (row.reliability, row.expected_profit),
                            xytext=(4, 4), textcoords="offset points", fontsize=7, color=COLORS[method])
        excluded = panel.loc[panel.expected_profit.isna(), "method"].tolist()
        if excluded:
            ax.text(.02, .03, "Estimated infeasible:\n" + ", ".join(excluded), transform=ax.transAxes,
                    fontsize=8, va="bottom", color="#8B0000")
        ax.set(title=rf"Endogeneity $\rho={rho:g}$", xlabel="True reliability", ylabel="True expected profit")
        ax.grid(alpha=.25)
    for ax in axes[len(cfg.rhos):]: ax.remove()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(METHODS), frameon=False)
    fig.suptitle(rf"Risk--return comparison at $\alpha={cfg.main_alpha:g}$  (labels: estimated-policy feasibility; hollow: true-risk LCB below target)", y=.995)
    fig.tight_layout(rect=(0, .10, 1, .95))
    fig.savefig(output / "main_fixed_alpha_comparison.png", dpi=300)
    plt.close(fig)


def plot_risk_return_frontier(summary: pd.DataFrame, cfg: ExperimentConfig, output: Path) -> None:
    """Full frontier: colour denotes method and marker denotes alpha."""
    fig, axes = _axes_grid(len(cfg.rhos))
    marker_by_alpha = {
        alpha: ALPHA_MARKERS[i % len(ALPHA_MARKERS)]
        for i, alpha in enumerate(cfg.effective_alphas)
    }
    for ax, rho in zip(axes, cfg.rhos):
        panel = summary[summary.rho == rho].dropna(subset=["reliability", "expected_profit"])
        for method in METHODS:
            method_data = panel[panel.method == method].sort_values("alpha")
            if method_data.empty:
                continue
            ax.plot(method_data.reliability, method_data.expected_profit, color=COLORS[method], lw=1.4, alpha=.85)
            for i, (_, row) in enumerate(method_data.iterrows()):
                ax.errorbar(row.reliability, row.expected_profit, xerr=1.96 * row.reliability_se,
                            yerr=1.96 * row.profit_se, fmt=marker_by_alpha[row.alpha],
                            color=COLORS[method], ms=5, capsize=1.8, label=method if i == 0 else None)
                if row.policy_feasible_rate < 1.0:
                    ax.annotate(f"{100 * row.policy_feasible_rate:.0f}%", (row.reliability, row.expected_profit),
                                xytext=(3, 3), textcoords="offset points", fontsize=6, color=COLORS[method])
        ax.set(title=rf"Endogeneity $\rho={rho:g}$", xlabel="True reliability", ylabel="True expected profit")
        ax.ticklabel_format(axis="x", style="plain", useOffset=False)
        ax.grid(alpha=.25)
    for ax in axes[len(cfg.rhos):]: ax.remove()
    method_handles = [Line2D([0], [0], color=COLORS[m], marker="o", lw=1.5, label=m) for m in METHODS]
    alpha_handles = [Line2D([0], [0], color="0.2", marker=marker_by_alpha[a], lw=0,
                            label=rf"$\alpha={a:g}$") for a in cfg.effective_alphas]
    fig.legend(method_handles, METHODS, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(.5, .035), title="Method (colour)")
    fig.legend(alpha_handles, [rf"$\alpha={a:g}$" for a in cfg.effective_alphas], loc="lower center",
               ncol=6, frameon=False, bbox_to_anchor=(.5, .105), title=r"Risk level (marker)")
    fig.tight_layout(rect=(0, .16, 1, 1))
    fig.savefig(output / "risk_return_frontier_by_endogeneity.png", dpi=300)
    plt.close(fig)


def plot_high_endogeneity_pareto(summary: pd.DataFrame, cfg: ExperimentConfig, output: Path) -> None:
    """Paper-ready Pareto figure: emphasize the regimes that identify CCW."""
    rhos = (0.5, 0.75, 1.0)
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.9), squeeze=False)
    marker_by_alpha = {
        alpha: ALPHA_MARKERS[i % len(ALPHA_MARKERS)]
        for i, alpha in enumerate(cfg.effective_alphas)
    }
    for ax, rho in zip(axes.ravel(), rhos):
        panel = summary[summary.rho.eq(rho)].dropna(subset=["reliability", "expected_profit"])
        for method in METHODS:
            d = panel[panel.method.eq(method)].sort_values("alpha")
            if d.empty:
                continue
            ax.plot(d.reliability, d.expected_profit, color=COLORS[method], lw=1.5,
                    alpha=.9 if method in ("CCW", "Oracle") else .45)
            for _, row in d.iterrows():
                ax.scatter(row.reliability, row.expected_profit, marker=marker_by_alpha[row.alpha], s=55,
                           facecolors=COLORS[method] if row.empirically_risk_adequate else "white",
                           edgecolors=COLORS[method], linewidths=1.5, zorder=4)
        ax.set(title=rf"Endogeneity $\rho={rho:g}$", xlabel="True reliability", ylabel="True expected profit")
        ax.grid(alpha=.22)
    method_handles = [Line2D([0], [0], color=COLORS[m], marker="o", lw=1.5, label=m) for m in METHODS]
    alpha_handles = [Line2D([0], [0], color="0.2", marker=marker_by_alpha[a], lw=0,
                            label=rf"$\alpha={a:g}$") for a in cfg.effective_alphas]
    fig.legend(method_handles, METHODS, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(.5, .01), title="Method (colour)")
    fig.legend(alpha_handles, [rf"$\alpha={a:g}$" for a in cfg.effective_alphas], loc="lower center", ncol=6,
               frameon=False, bbox_to_anchor=(.5, .11), title=r"Risk level (marker)")
    fig.suptitle(r"High-endogeneity risk--return frontiers (filled: reliability LCB meets that point's target)", y=1.01)
    fig.tight_layout(rect=(0, .20, 1, .96))
    fig.savefig(output / "high_endogeneity_pareto_frontier.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_diagnostics(summary: pd.DataFrame, cfg: ExperimentConfig, output: Path) -> None:
    main = summary[np.isclose(summary.alpha, cfg.main_alpha) & (summary.method != "Oracle")]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.9))
    for method in [m for m in METHODS if m != "Oracle"]:
        z = main[main.method == method].sort_values("rho")
        axes[0].plot(z.rho, z.reliability_margin, marker="o", color=COLORS[method], label=method)
        axes[1].errorbar(z.rho, z.calibration_mae, yerr=1.96 * z.calibration_mae_se,
                         marker="o", capsize=2.5, color=COLORS[method], label=method)
    axes[0].set(xlabel=r"Endogeneity strength $\rho$", ylabel=rf"Reliability margin $R-(1-\alpha)$ at $\alpha={cfg.main_alpha:g}$")
    axes[0].axhline(0.0, color="0.35", ls="--", lw=1, label="risk target")
    axes[1].set(xlabel=r"Endogeneity strength $\rho$", ylabel="Decision-relevant probability MAE")
    for ax in axes:
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "feasibility_and_probability_diagnostics.png", dpi=300)
    plt.close(fig)


def aggregate_and_plot(results: pd.DataFrame, cfg: ExperimentConfig, output: Path) -> None:
    results.to_csv(output / "replication_results.csv", index=False)
    summary = _aggregate(results)
    summary.to_csv(output / "pareto_summary.csv", index=False)
    plot_fixed_alpha_comparison(summary, cfg, output)
    plot_risk_return_frontier(summary, cfg, output)
    plot_high_endogeneity_pareto(summary, cfg, output)
    plot_diagnostics(summary, cfg, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default="ccw_experiment_output")
    parser.add_argument("--plot-summary", default=None,
                        help="Regenerate all figures from an existing pareto_summary.csv without rerunning experiments.")
    parser.add_argument("--replications", type=int, default=None)
    parser.add_argument("--smoke-test", action="store_true", help="Small configuration for an executable check.")
    args = parser.parse_args()
    cfg = ExperimentConfig()
    if args.plot_summary:
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        summary = pd.read_csv(args.plot_summary)
        plot_fixed_alpha_comparison(summary, cfg, output)
        plot_risk_return_frontier(summary, cfg, output)
        plot_high_endogeneity_pareto(summary, cfg, output)
        plot_diagnostics(summary, cfg, output)
        print(f"Figures regenerated in: {output.resolve()}")
        return
    if args.replications is not None:
        cfg = replace(cfg, n_replications=args.replications)
    if args.smoke_test:
        cfg = replace(cfg, n_replications=1, n_train=400, n_validation=90, n_test_contexts=10,
                      n_eval_mc=400, price_step=.5, rhos=(0.0, 0.5, 1.0),
                      effective_alphas=(.05, .10, .30), k_candidates=(25, 45),
                      cq_levels=(.02, .05, .10, .20, .40, .60, .80, .90, .95, .98))
    if cfg.main_alpha not in cfg.effective_alphas:
        raise ValueError("main_alpha must be one of effective_alphas.")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for rho in cfg.rhos:
        for rep in range(cfg.n_replications):
            print(f"rho={rho:g}, replication={rep + 1}/{cfg.n_replications}", flush=True)
            rows.extend(run_one_replication(cfg, rho, rep))
    aggregate_and_plot(pd.DataFrame(rows), cfg, output)
    print(f"Completed. Results and figures are in: {output.resolve()}")


if __name__ == "__main__":
    main()
