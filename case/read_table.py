"""Display the reported JD case and optionally compare an archived workbook."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean


def read_reported():
    table_path = Path(__file__).with_name("reported-results.csv")
    with table_path.open(encoding="utf-8-sig", newline="") as table:
        rows = list(csv.DictReader(table))
    results = {}
    for row in rows:
        key = int(row["profit_target_RMB"]), float(row["original_setting_label"])
        results[key] = tuple(float(row[name]) for name in (
            "reported_profit_ratio", "reference_reliability", "ccw_reliability"
        ))
    expected = {(v, p) for v in (0, 5, 10) for p in (0.1, 0.2, 0.5, 0.9)}
    if set(results) != expected or len(rows) != len(expected):
        raise ValueError("Expected one reported row for each of the twelve settings.")
    return results


def summarize_workbook(path):
    from openpyxl import load_workbook

    book = load_workbook(path, read_only=True, data_only=True)
    groups = defaultdict(list)
    try:
        rows = book.active.values
        headings = next(rows)
        for values in rows:
            row = dict(zip(headings, values))
            if row["originalObj"] in (0, None):
                continue
            key = int(row["v"]), float(row["alpha"])
            groups[key].append((row["kNNobj"] / row["originalObj"],
                                row["originRate"], row["kNNrate"]))
    finally:
        book.close()
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-workbook", type=Path,
                        help="Compare a locally held CaseStudy.xlsx (requires openpyxl).")
    args = parser.parse_args()
    reported = read_reported()
    print("Reported results (the manuscript labels the setting as required probability)")
    print("v   setting     LR    OFR    CFR")
    for (target, setting), (ratio, reference, ccw) in sorted(reported.items()):
        print(f"{target:2}    {setting:.1f}   {ratio:.3f}  {reference:.3f}  {ccw:.3f}")

    if args.historical_workbook:
        groups = summarize_workbook(args.historical_workbook)
        print("\nHistorical workbook grouped by its numeric setting label")
        print("v   label    rows     LR    OFR    CFR   matches to 3 decimals")
        for key, values in sorted(reported.items()):
            target, label = key
            if key not in groups:
                print(f"{target:2}    {label:.1f}       0   (missing)")
                continue
            records = groups[key]
            avg = tuple(mean(column) for column in zip(*records))
            matches = all(f"{a:.3f}" == f"{b:.3f}" for a, b in zip(avg, values))
            numbers = "  ".join(f"{value:.3f}" for value in avg)
            print(f"{target:2}    {label:.1f}    {len(records):4}   {numbers}   {matches}")


if __name__ == "__main__":
    main()
