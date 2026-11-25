"""Repeat the contextual CDF diagnostic with lognormal and Student-t shocks."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import pandas as pd

from theory_validation import (
    REPLICATIONS, SAMPLE_SIZES, SEED, SHOCK_LAWS, one_replication, query_grid, summarize,
)


DEFAULT_SHOCKS = tuple(shock for shock in SHOCK_LAWS if shock != "normal")
OUTPUT_STEM = "contextual_probability_shock_robustness_20261120"
_cached_grid = lru_cache(maxsize=len(SHOCK_LAWS))(query_grid)


def run_replication(job: tuple[str, int, int]) -> list[dict]:
    shock_law, sample_size, replication = job
    queries, truth = _cached_grid(shock_law)
    results = one_replication(sample_size, replication, SEED, queries, truth, shock_law=shock_law)
    return [{"shock_law": shock_law, **row} for row in results]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parent / "assets" / "data",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least one")

    jobs = [(shock, n, j) for n in SAMPLE_SIZES for shock in DEFAULT_SHOCKS for j in range(REPLICATIONS)]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for batch in executor.map(run_replication, jobs, chunksize=4):
            rows.extend(batch)
            if batch[0]["replication"] == REPLICATIONS - 1:
                print(f'{batch[0]["shock_law"]}: complete N={batch[0]["sample_size"]:,}', flush=True)

    raw = pd.DataFrame(rows)
    summaries = []
    for shock_law, group in raw.groupby("shock_law", sort=False):
        summary = summarize(group)
        summary.insert(0, "shock_law", shock_law)
        summaries.append(summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.output_dir / f"{OUTPUT_STEM}_replications.csv", index=False)
    pd.concat(summaries, ignore_index=True).to_csv(args.output_dir / f"{OUTPUT_STEM}_summary.csv", index=False)


if __name__ == "__main__":
    main()
