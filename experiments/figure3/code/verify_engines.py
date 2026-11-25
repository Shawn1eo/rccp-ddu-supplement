"""Compare independent evaluations of the same seed by the two implementations."""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import matplotlib
import numpy
import pandas
import scipy
import sklearn


def compare(left, right):
    if left is None or right is None:
        assert left is None and right is None
        return 0
    if isinstance(left, (str, bool)) or isinstance(right, (str, bool)):
        assert left == right
        return 0
    return abs(left - right)


def verify(cells, source):
    report = []
    for path in sorted(cells.glob('*_original.json')):
        batch_path = path.with_name(path.name.replace('_original.json', '_fast.json'))
        if not batch_path.is_file():
            continue
        original = json.loads(path.read_text(encoding='utf-8'))
        batched = json.loads(batch_path.read_text(encoding='utf-8'))
        first = original['rows'][0]
        entry = dict(rho=first['rho'], replication=first['replication'], n_decisions=0)
        differences = {}
        for method, decisions in original['decisions'].items():
            other = batched['decisions'][method]
            assert len(decisions) == len(other)
            for a, b in zip(decisions, other):
                assert a['context_id'] == b['context_index']
                for name in ('feasible', 'price', 'order', 'estimated_reliability'):
                    key = 'decision_' + name
                    differences[key] = max(differences.get(key, 0), compare(a[name], b[name]))
                entry['n_decisions'] += 1
        other_rows = {r['method']: r for r in batched['rows']}
        for row in original['rows']:
            other = other_rows[row['method']]
            assert set(other) == set(row)
            for key in row:
                delta = compare(row[key], other[key])
                differences[key] = max(differences.get(key, 0), delta)
        entry['max_absolute_difference'] = differences
        assert max(differences.values()) < 1e-9, entry
        report.append(entry)
    assert report, "Generate at least one seed with both engines before comparing"
    return dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                python=platform.python_version(), numpy=numpy.__version__,
                pandas=pandas.__version__, sklearn=sklearn.__version__,
                scipy=scipy.__version__, matplotlib=matplotlib.__version__, comparisons=report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cells', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.write_text(json.dumps(verify(args.cells, args.source), indent=2), encoding='utf-8')
