"""One fixed-alpha replication of the archived experiment, with batched queries.

run(source_path, rho, rep) returns the original five summary rows plus the
decisions at every context. No decisions or rows are shared across replications.
"""
from __future__ import annotations

import hashlib
import importlib.util
from functools import lru_cache
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits


@lru_cache(maxsize=4)
def _load_source(path):
    key = hashlib.sha256(str(path).encode()).hexdigest()[:16]
    name = "_rccp_figure3_source_" + key
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _normalized(model, joint):
    query = (joint - model.location) / model.scale
    query[:, 0] *= model.action_weight
    return query


def _select_ccw(source, train, validation, cfg):
    columns = ["p", *[f"x{i}" for i in range(cfg.context_dim)]]
    joint = validation[columns].to_numpy(dtype=float)
    observed = validation["demand"].to_numpy()[:, None]
    scores = {}
    max_k = max(cfg.k_candidates)
    for weight in cfg.action_weight_candidates:
        model = source.CCWModel(train, cfg, max_k, weight)
        query = _normalized(model, joint)
        distances, indices = model.nn.kneighbors(query, n_neighbors=max_k+1)
        for k in cfg.k_candidates:
            nearest = indices[:, :k]
            # Query the requested k directly if crossing an exact distance tie.
            ties = distances[:,k-1] == distances[:,k]
            if ties.any():
                nearest = nearest.copy()
                nearest[ties] = model.nn.kneighbors(query[ties], n_neighbors=k,
                                                    return_distance=False)
            samples = model.demands[nearest]
            sorted_samples = np.sort(samples, axis=1)
            ranks = 2 * np.arange(k) - k + 1
            spread = np.sum(sorted_samples * ranks, axis=1) / k**2
            row_scores = np.abs(samples - observed).mean(axis=1) - spread
            scores[k, weight] = sum(row_scores.tolist()) / len(row_scores)
    best = min(scores, key=scores.get)
    # The sorted identity is exact in real arithmetic. Resolve any numerical
    # near-tie with the reference calculation, in its original iteration order.
    close = [pair for pair in scores
             if abs(scores[pair]-scores[best]) <= 1e-11 * max(1,abs(scores[best]))]
    if len(close) > 1:
        for k, weight in close:
            scores[k,weight] = source.CCWModel(train,cfg,k,weight).validation_crps(validation)
        best = min(close, key=scores.get)
    return source.CCWModel(train, cfg, *best), scores[best]


def _empirical(source, cfg, contexts, sample_at_price):
    """Apply the original quantile/feasibility formulas to one price at a time."""
    n = len(contexts)
    incumbent = np.full(n,-np.inf)
    decisions = [source.Decision() for _ in contexts]
    for price in cfg.prices:
        samples = np.ascontiguousarray(sample_at_price(price), dtype=float)
        minimum = cfg.profit_target / (price - cfg.unit_cost)
        q_alpha = np.quantile(samples, cfg.main_alpha, axis=1, method="higher")
        maximum = ((price - cfg.salvage_value) * q_alpha - cfg.profit_target) / (
            cfg.unit_cost - cfg.salvage_value)
        feasible = maximum >= minimum - 1e-10
        if not feasible.any():
            continue
        ratio = (price - cfg.unit_cost) / (price - cfg.salvage_value)
        order = np.clip(np.quantile(samples, ratio, axis=1, method="linear"),minimum,maximum)
        profit = source.profit(np.full(n,price), order, samples, cfg)
        objective = profit.mean(axis=1)
        improve = feasible & (objective > incumbent)
        for index in np.flatnonzero(improve):
            decisions[index] = source.Decision(
                float(price), float(order[index]),
                float((profit[index] >= cfg.profit_target).mean()), True)
        incumbent[improve] = objective[improve]
    return decisions


def _policies(source, cfg, train, ccw, contexts, rng, rho, rep):
    count = len(contexts)
    prices = cfg.prices
    # Query all fitted trees together, but construct interpolated scenarios
    # one price at a time to avoid retaining a 80 x 41 x 1001 working array.
    cq = source.CQModel(train, cfg, seed=10_000+rep)
    features = cq.features(np.repeat(prices, count), np.tile(contexts,(len(prices),1)))
    fits = np.maximum.accumulate(np.column_stack(
        [model.predict(features) for model in cq.models]), axis=1)
    fit_by_price = dict(zip(prices, fits.reshape(len(prices),count,len(cq.models))))
    parametric = source.ParametricModel(train, cfg, rng)
    saa = source.SAAModel(train,cfg)

    def ccw_at(price):
        joint = np.column_stack([np.full(count,price), contexts])
        indices = ccw.nn.kneighbors(_normalized(ccw,joint), return_distance=False)
        return ccw.demands[indices]

    def cq_at(price):
        return np.maximum(0,np.asarray([
            np.interp(cq.probability_grid,cq.levels,q) for q in fit_by_price[price]]))

    def parametric_at(price):
        # One-row Ridge prediction preserves the reference floating-point
        # values exactly, including threshold comparisons for tied outcomes.
        return np.vstack([parametric.scenarios(price,x) for x in contexts])

    policies = {"CCW": _empirical(source,cfg,contexts,ccw_at),
                "CQ-CCP": _empirical(source,cfg,contexts,cq_at),
                "P-CCP": _empirical(source,cfg,contexts,parametric_at)}
    # This model's demand scenarios are the same for every context.
    single = saa.prescribe(contexts[0],cfg.main_alpha)
    policies["SAA-CCP"] = [single]*count
    policies["Oracle"] = _oracle(source,cfg,contexts,rho)
    if not all(decision.feasible for decision in policies["Oracle"]):
        raise RuntimeError(f"Oracle infeasible at rho={rho}, alpha={cfg.main_alpha}")
    return policies


def _oracle(source,cfg,contexts,rho):
    prices = cfg.prices[None,:]
    shape = (len(contexts),len(cfg.prices))
    # Single-pair calls reproduce the reference inner products, irrespective
    # of the BLAS implementation. The CDF/PDF calls are vectorized.
    mus = np.empty(shape)
    scales = np.empty(shape)
    for i,x in enumerate(contexts):
        for j,p in enumerate(cfg.prices):
            mu,sd = source._mode1_parameters(np.array([p]),x.reshape(1,-1),rho,cfg)
            mus[i,j],scales[i,j] = mu[0],sd[0]
    minimum = cfg.profit_target/(prices-cfg.unit_cost)
    alpha_q = np.maximum(0,mus + scales*source.norm.ppf(cfg.main_alpha))
    maximum = ((prices-cfg.salvage_value)*alpha_q-cfg.profit_target)/(cfg.unit_cost-cfg.salvage_value)
    feasible = maximum >= minimum - 1e-10
    ratio = (prices-cfg.unit_cost)/(prices-cfg.salvage_value)
    order = np.clip(np.maximum(0,mus+scales*source.norm.ppf(ratio)),minimum,maximum)
    threshold = (cfg.profit_target+(cfg.unit_cost-cfg.salvage_value)*order)/(prices-cfg.salvage_value)
    reliability = source.norm.sf((threshold-mus)/scales)
    objective = source.expected_profit_mode1(prices,order,mus,scales,cfg)
    idx = np.argmax(np.where(feasible,objective,-np.inf),axis=1)
    return [source.Decision(float(prices[0,j]),float(order[i,j]),float(reliability[i,j]),True)
            if feasible[i,j] else source.Decision() for i,j in enumerate(idx)]


def run(source_path, rho, rep):
    """Reproduce all five fixed-alpha methods with the archived RNG schedule."""
    source = _load_source(Path(source_path).resolve())
    rho,rep = float(rho),int(rep)
    with threadpool_limits(limits=1):
        cfg = source.ExperimentConfig()
        rng = np.random.default_rng(np.random.SeedSequence([cfg.seed,int(1000*rho),rep]))
        train = source.generate_historical_data(rng,cfg.n_train,rho,cfg)
        validation = source.generate_historical_data(rng,cfg.n_validation,rho,cfg)
        contexts = source.correlated_context(rng,cfg.n_test_contexts,cfg.context_dim)
        draws = rng.standard_normal((len(contexts),cfg.n_eval_mc))
        ccw,crps = _select_ccw(source,train,validation,cfg)
        policies = _policies(source,cfg,train,ccw,contexts,rng,rho,rep)
        rows, details, counts = [],{},{}
        for name in source.METHODS:
            decisions = policies[name]
            valid = [d for d in decisions if d.feasible]
            metrics = source.evaluate_policy(decisions,contexts,rho,cfg.main_alpha,cfg,draws)
            rows.append({"rho":rho, "replication":rep,"method":name,"alpha":cfg.main_alpha,
                         "k_ccw":ccw.k,"ccw_action_weight":ccw.action_weight,
                         "ccw_validation_crps":crps,**metrics,
                         "mean_price":float(np.mean([d.price for d in valid])) if valid else np.nan,
                         "mean_order":float(np.mean([d.order for d in valid])) if valid else np.nan})
            counts[name] = len(valid)
            details[name] = [{"context_index":i,"feasible":bool(d.feasible),
                              "price":float(d.price) if d.feasible else None,
                              "order":float(d.order) if d.feasible else None,
                              "estimated_reliability":float(d.estimated_reliability) if d.feasible else None}
                             for i,d in enumerate(decisions)]
    return {"rows":rows,"decisions":details,"context_count":len(contexts),
            "feasible_counts":counts,"contexts":contexts.tolist()}
