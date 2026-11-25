"""Run one historical timing design cell and print its reported solve time."""
import argparse
import random
import sys
from pathlib import Path

import gurobipy as gp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "code"))
from DataGeneration import DataGeneration
from Scheme import scheme1, scheme2, scheme3, scheme4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, required=True)
    parser.add_argument("--strategy", type=int, choices=(1, 2, 3, 4), required=True)
    parser.add_argument("--mode", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if args.sample_size < 4:
        parser.error("Use a sample size of at least four for the neighborhood rules.")

    np.random.seed(args.seed)
    random.seed(args.seed)
    gp.setParam("Threads", 1)
    prices = [10 + 0.1 * index for index in range(200)]
    history, context = DataGeneration(args.sample_size, prices, 1, args.mode, 10)
    solver = (scheme1, scheme2, scheme3, scheme4)[args.strategy - 1]
    result = solver(history, context, prices, 5, 2, 0.7, 1500, 5000)
    seconds = result[0] if isinstance(result, tuple) else result
    print("sample_size,strategy,mode,seed,seconds")
    print(f"{args.sample_size},{args.strategy},{args.mode},{args.seed},{seconds}")


if __name__ == "__main__":
    main()
