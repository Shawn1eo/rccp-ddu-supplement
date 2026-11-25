"""Recalculate selected manuscript statistics from archived rows, without fitting models."""

import csv
from collections import defaultdict
from math import isclose
from pathlib import Path
from statistics import mean


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def check(condition, message):
    if not condition:
        raise ValueError(message)


def check_number(actual, expected, label):
    check(isclose(actual, float(expected), rel_tol=1e-10, abs_tol=1e-12),
          f"{label}: calculated {actual}; archived {expected}")


def check_cdf():
    folder = EXPERIMENTS / "diagnostics" / "assets" / "data"
    groups = defaultdict(dict)
    for row in read_rows(folder / "contextual_probability_20261119_replications.csv"):
        key = (int(row["sample_size"]), row["method"])
        rep = int(row["replication"])
        check(rep not in groups[key], f"Repeated CDF row: {key}, {rep}")
        check(int(row["available_queries"]) == int(row["query_count"]) == 135,
              f"Incomplete CDF grid: {key}, {rep}")
        groups[key][rep] = float(row["max_probability_error"])

    summaries = read_rows(folder / "contextual_probability_20261119_summary.csv")
    sizes = (1000, 2000, 5000, 10000, 30000, 100000)
    keys = {(n, m) for n in sizes for m in ("kNN", "LSA", "CART")}
    check(set(groups) == keys and len(summaries) == len(keys), "Unexpected CDF design")
    checked = set()
    for row in summaries:
        key = int(row["sample_size"]), row["method"]
        check(set(groups[key]) == set(range(60)), f"Missing CDF replication: {key}")
        check_number(mean(groups[key].values()), row["mean_error"], f"CDF mean {key}")
        checked.add(key)
    check(checked == keys, "Incomplete CDF summary")
    print("Figure 2: all 18 mean maximum CDF errors agree across 1,080 method-replication records.")


def check_decisions():
    folder = EXPERIMENTS / "figure3" / "paired_reps70_89" / "margin_results"
    groups = defaultdict(dict)
    for row in read_rows(folder / "pointwise_population.csv"):
        key = float(row["rho"]), row["method"]
        case = int(row["replication"]), int(row["context_id"])
        check(case not in groups[key], f"Repeated decision row: {key}, {case}")
        returned = row["returned"] == "True"
        feasible = returned and float(row["true_reliability"]) >= 0.90 - 1e-12
        check(feasible == (row["true_feasible"] == "True"), f"Feasibility mismatch: {key}, {case}")
        groups[key][case] = (float(row["true_expected_profit"]),
                             float(row["oracle_expected_profit"])) if feasible else None

    intensities = (0.0, 0.25, 0.5, 0.75, 1.0)
    methods = ("CCW", "CQ-CCP", "P-CCP", "SAA-CCP", "Oracle")
    cases = {(rep, ctx) for rep in range(70, 90) for ctx in range(80)}
    keys = {(rho, method) for rho in intensities for method in methods}
    summaries = read_rows(folder / "population_summary.csv")
    check(set(groups) == keys and len(summaries) == len(keys), "Unexpected Figure 3 design")
    checked = set()
    ccw = []
    for row in summaries:
        key = float(row["rho"]), row["method"]
        check(set(groups[key]) == cases, f"Missing test contexts: {key}")
        eligible = [pair for pair in groups[key].values() if pair is not None]
        coverage = len(eligible) / len(cases)
        check_number(coverage, row["true_feasible_rate"], f"Figure 3 coverage {key}")
        if eligible:
            method_profit, oracle_profit = zip(*eligible)
            share = sum(method_profit) / sum(oracle_profit)
            check_number(share, row["matched_oracle_profit_share"], f"Figure 3 profit share {key}")
            if key[1] == "CCW":
                ccw.append((coverage, share))
        else:
            check(row["matched_oracle_profit_share"] == "", f"Profit share with no qualifying contexts: {key}")
        checked.add(key)
    check(checked == keys and len(ccw) == len(intensities), "Incomplete Figure 3 summary")
    coverage, share = zip(*ccw)
    print("Figure 3: all 25 coverage and matched-profit cells agree across 40,000 method-context records.")
    print(f"  CCW coverage {min(coverage):.2%}–{max(coverage):.2%}; matched Oracle profit {min(share):.2%}–{max(share):.2%}.")


def check_runtime():
    times = defaultdict(list)
    for row in read_rows(EXPERIMENTS / "runtime" / "timings.csv"):
        if int(row["sample_size"]) == 6000:
            times[(int(row["mode"]), int(row["strategy"]))].append(float(row["seconds"]))
    reductions = []
    for mode in (1, 2, 3):
        base, accelerated = times[(mode, 3)], times[(mode, 4)]
        check(len(base) == len(accelerated) == 4, f"Missing archived timings: mode {mode}")
        reductions.append(100 * (1 - mean(accelerated) / mean(base)))
    check([round(r) for r in reductions] == [72, 73, 70], "Runtime reductions differ from reported range")
    print("Figure 5: at N=6,000, SS–BD–Acc reduces mean time relative to SS–BD by " +
          ", ".join(f"{r:.2f}%" for r in reductions) + " (modes 1–3).")


if __name__ == "__main__":
    check_cdf()
    check_decisions()
    check_runtime()
