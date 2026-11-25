#!/usr/bin/env python3
"""Repeat the fixed-alpha decision stress test with three demand shocks.

Use the original location and scale, estimator settings, and seed schedule.
Report exact population means conditional on full coverage of the fitted
constraints. Normal, lognormal, and t3 scenarios share normal quantile draws
and historical prices and contexts by replication and DDU intensity.
"""
from __future__ import annotations

import argparse
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr, stdtrit
from scipy.stats import lognorm, norm, t
from threadpoolctl import threadpool_limits

import ccw_endogenous_ccp_experiment as stress


LAWS = {"normal": norm, "lognormal": lognorm(s=1), "t3": t(df=3)}
ALPHA = 0.10
RHOS = (0.0, 0.5, 1.0)
REPLICATIONS = 6


def shock_draw(normal_draw, name):
    if name == "normal":
        return normal_draw
    if name == "lognormal":
        return np.exp(normal_draw)
    return stdtrit(3, ndtr(normal_draw))


def truncated_moment(z, name):
    """First moment of U below z, with no conditioning on U <= z."""
    if name == "normal":
        return -norm.pdf(z)
    if name == "lognormal":
        return np.where(z > 0, math.exp(.5) * ndtr(np.log(np.maximum(z, np.finfo(float).tiny)) - 1), 0)
    return -(3 + z**2) * t.pdf(z, df=3) / 2


def population_profit(price, order, mu, scale, name, cfg):
    zero = -mu / scale
    upper = (order - mu) / scale
    law = LAWS[name]
    leftover = ((order - mu) * law.cdf(upper) + mu * law.cdf(zero)
                - scale * (truncated_moment(upper, name) - truncated_moment(zero, name)))
    return (price - cfg.unit_cost) * order - (price - cfg.salvage_value) * leftover


def population_reliability(price, order, mu, scale, name, cfg):
    threshold = (cfg.profit_target + (cfg.unit_cost - cfg.salvage_value) * order) / (price - cfg.salvage_value)
    return np.where(order >= cfg.profit_target / (price - cfg.unit_cost) - 1e-10,
                    LAWS[name].sf((threshold - mu) / scale), 0)


def demand_history(rng, n, rho, name, cfg):
    prices = rng.choice(cfg.prices, size=n)
    contexts = stress.correlated_context(rng, n, cfg.context_dim)
    normal = rng.standard_normal((n, 1))
    demand = stress.demand_from_dgp(prices, contexts, shock_draw(normal, name), rho, cfg)[:, 0]
    data = pd.DataFrame(contexts, columns=[f"x{i}" for i in range(cfg.context_dim)])
    data["p"] = prices
    data["demand"] = demand
    return data


def ccw_scenarios(model, queries):
    normalized = (queries - model.location) / model.scale
    normalized[:, 0] *= model.action_weight
    indices = model.nn.kneighbors(normalized, return_distance=False)
    return model.demands[indices]


def select_ccw(train, validation, queries, cfg):
    """Compute identical held-out CRPS using the sorted empirical formula."""
    validation_joint = validation[["p", *[f"x{i}" for i in range(cfg.context_dim)]]].to_numpy()
    observed = validation.demand.to_numpy()[:, None]
    scores = {}
    models = {}
    for weight in cfg.action_weight_candidates:
        model = stress.CCWModel(train, cfg, max(cfg.k_candidates), weight)
        models[weight] = model
        neighbors = ccw_scenarios(model, validation_joint)
        for k in cfg.k_candidates:
            samples = neighbors[:, :k]
            ordered = np.sort(samples, axis=1)
            ranks = 2 * np.arange(k) - k + 1
            spread = np.sum(ordered * ranks, axis=1) / k**2
            scores[k, weight] = float((np.abs(samples - observed).mean(axis=1) - spread).mean())
    k, weight = min(scores, key=scores.get)
    return ccw_scenarios(models[weight], queries)[:, :k], k, weight, scores[k, weight]


def best_by_context(feasible, expected_profit, order, estimated_reliability, cfg):
    """Select the first profit-maximizing feasible price at each context."""
    available = feasible.any(axis=1)
    rows = np.arange(len(available))
    idx = np.argmax(np.where(feasible, expected_profit, -np.inf), axis=1)
    return available, cfg.prices[idx], order[rows, idx], estimated_reliability[rows, idx]


def empirical_policy(samples, n_contexts, cfg):
    """Batched equivalent of scenario_decision followed by best_from_sampler."""
    samples = np.asarray(samples).reshape(-1, len(cfg.prices), samples.shape[-1])
    prices = cfg.prices[None, :]
    minimum = cfg.profit_target / (prices - cfg.unit_cost)
    alpha_quantile = np.quantile(samples, ALPHA, axis=2, method="higher")
    maximum = ((prices - cfg.salvage_value) * alpha_quantile - cfg.profit_target) / (cfg.unit_cost - cfg.salvage_value)
    feasible = maximum >= minimum - 1e-10
    ranks = (samples.shape[2] - 1) * (prices - cfg.unit_cost) / (prices - cfg.salvage_value)
    lower, upper = np.floor(ranks).astype(int), np.ceil(ranks).astype(int)
    sorted_samples = np.sort(samples, axis=2)
    lower_value = np.take_along_axis(sorted_samples, lower[:, :, None], axis=2)[:, :, 0]
    upper_value = np.take_along_axis(sorted_samples, upper[:, :, None], axis=2)[:, :, 0]
    quantile = lower_value + (ranks - lower) * (upper_value - lower_value)
    order = np.clip(quantile, minimum, maximum)
    profit = stress.profit(prices, order, samples, cfg)
    policy = best_by_context(feasible, profit.mean(axis=-1), order, (profit >= cfg.profit_target).mean(axis=-1), cfg)
    if len(policy[0]) == 1 and n_contexts != 1:
        return tuple(np.repeat(x, n_contexts) for x in policy)
    return policy


def oracle_policy(mu, scale, name, cfg):
    prices = cfg.prices[None, :]
    minimum = cfg.profit_target / (prices - cfg.unit_cost)
    law = LAWS[name]
    alpha_q = np.maximum(0, mu + scale * law.ppf(ALPHA))
    maximum = ((prices - cfg.salvage_value) * alpha_q - cfg.profit_target) / (cfg.unit_cost - cfg.salvage_value)
    feasible = maximum >= minimum - 1e-10
    beta = (prices - cfg.unit_cost) / (prices - cfg.salvage_value)
    order = np.clip(np.maximum(0, mu + scale * law.ppf(beta)), minimum, maximum)
    profit = population_profit(prices, order, mu, scale, name, cfg)
    reliability = population_reliability(prices, order, mu, scale, name, cfg)
    return best_by_context(feasible, profit, order, reliability, cfg)


def replicate(name, rho, replication):
    with threadpool_limits(limits=1):
        cfg = stress.ExperimentConfig()
        rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, int(1000 * rho), replication]))
        train = demand_history(rng, cfg.n_train, rho, name, cfg)
        valid = demand_history(rng, cfg.n_validation, rho, name, cfg)
        contexts = stress.correlated_context(rng, cfg.n_test_contexts, cfg.context_dim)
        # Keep the residual bootstrap aligned with the original RNG sequence.
        rng.standard_normal((cfg.n_test_contexts, cfg.n_eval_mc))
        n_ctx, n_prices = len(contexts), len(cfg.prices)
        queries = np.column_stack([np.tile(cfg.prices, n_ctx), np.repeat(contexts, n_prices, axis=0)])
        mu, scale = stress._mode1_parameters(queries[:, 0], queries[:, 1:], rho, cfg)
        mu = mu.reshape(n_ctx, n_prices)
        scale = scale.reshape(n_ctx, n_prices)
        ccw, selected_k, selected_weight, crps = select_ccw(train, valid, queries, cfg)
        cq = stress.CQModel(train, cfg, seed=10_000 + replication)
        features = cq.features(queries[:, 0], queries[:, 1:])
        values = np.maximum.accumulate(np.column_stack([m.predict(features) for m in cq.models]), axis=1)
        cq_samples = np.maximum(0, np.array([np.interp(cq.probability_grid, cq.levels, row) for row in values]))
        parametric = stress.ParametricModel(train, cfg, rng)
        means = parametric.mean_model.predict(parametric.features(queries[:, 0], queries[:, 1:]))
        p_samples = np.maximum(0, means[:, None] + parametric.residual_draws)
        saa = np.broadcast_to(train.demand.to_numpy(), (n_prices, cfg.n_train))
        policies = {
            "CCW": empirical_policy(ccw, n_ctx, cfg),
            "CQ-CCP": empirical_policy(cq_samples, n_ctx, cfg),
            "P-CCP": empirical_policy(p_samples, n_ctx, cfg),
            "SAA-CCP": empirical_policy(saa, n_ctx, cfg),
            "Oracle": oracle_policy(mu, scale, name, cfg),
        }
        rows = []
        for method, (available, prices, order, estimated) in policies.items():
            m, s = stress._mode1_parameters(prices, contexts, rho, cfg)
            complete = available.all()
            if method == "Oracle" and not complete:
                raise ValueError(f"Incomplete Oracle policy: {name=}, {rho=}, {replication=}")
            profit = population_profit(prices, order, m, s, name, cfg)
            reliability = population_reliability(prices, order, m, s, name, cfg)
            rows.append({"shock": name, "rho": rho, "alpha": ALPHA, "replication": replication,
                         "method": method, "complete_policy": bool(complete),
                         "decision_availability": float(available.mean()),
                         "expected_profit": float(profit.mean()) if complete else np.nan,
                         "reliability": float(reliability.mean()) if complete else np.nan,
                         "k_ccw": selected_k, "ccw_action_weight": selected_weight,
                         "ccw_validation_crps": crps})
        return rows


def summarize(records):
    out = records.groupby(["shock", "rho", "alpha", "method"], as_index=False).agg(
        n=("replication", "size"), complete=("complete_policy", "sum"),
        mean_availability=("decision_availability", "mean"),
        mean_profit=("expected_profit", "mean"), se_profit=("expected_profit", "sem"),
        mean_reliability=("reliability", "mean"), se_reliability=("reliability", "sem"))
    critical = t.ppf(.975, out.complete - 1)
    out["profit_halfwidth95"] = critical * out.se_profit
    out["reliability_halfwidth95"] = critical * out.se_reliability
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replications", type=int, default=REPLICATIONS)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "assets" / "data")
    args = parser.parse_args()
    if args.replications <= 0 or args.workers <= 0:
        parser.error("Replications and workers must be positive.")
    tasks = [(name, rho, rep) for rep in range(args.replications) for rho in RHOS for name in LAWS]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(replicate, *task): task for task in tasks}
        for completed in as_completed(pending):
            rows.extend(completed.result())
            print("Completed", pending[completed], flush=True)
    records = pd.DataFrame(rows).sort_values(["shock", "rho", "replication", "method"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records.to_csv(args.output_dir / "stress_shock_robustness_replications.csv", index=False)
    summarize(records).to_csv(args.output_dir / "stress_shock_robustness_summary.csv", index=False)


if __name__ == "__main__":
    main()
