"""How often each independent replication clears the descriptive 85% coverage reference."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t


def calculate(input_file, output_dir):
    point = pd.read_csv(input_file)
    assert not point.duplicated(['rho', 'method', 'replication', 'context_id']).any()
    assert point['true_feasible'].isin([True, False]).all()
    reps = (point.groupby(['rho', 'method', 'replication'])
            .agg(n_contexts=('true_feasible', 'size'), n_returned=('returned', 'sum'),
                 n_truly_feasible=('true_feasible', 'sum')).reset_index())
    assert len(reps) == 500 and reps.n_contexts.eq(80).all()
    reps['coverage'] = reps.n_truly_feasible / reps.n_contexts
    reps['at_least_85_pct'] = 100*reps.n_truly_feasible >= 85*reps.n_contexts
    summaries = []
    for (rho, method), group in reps.groupby(['rho', 'method']):
        n = len(group)
        assert n == 20
        mean = group.coverage.mean()
        se = group.coverage.std(ddof=1) / np.sqrt(n)
        radius = t.ppf(.975, n-1)*se
        summaries.append(dict(rho=rho, method=method, replications=n, n_at_least_85_pct=int(group.at_least_85_pct.sum()),
                              coverage_mean=mean, coverage_se=se, coverage_t95_low=max(0.,mean-radius),
                              coverage_t95_high=min(1.,mean+radius), min_contexts_feasible=int(group.n_truly_feasible.min()),
                              median_contexts_feasible=group.n_truly_feasible.median(),
                              max_contexts_feasible=int(group.n_truly_feasible.max())))
    output_dir.mkdir(parents=True, exist_ok=True)
    reps.to_csv(output_dir/'coverage_by_replication.csv', index=False)
    pd.DataFrame(summaries).to_csv(output_dir/'coverage_stability.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pointwise_population', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    calculate(args.pointwise_population, args.out)
