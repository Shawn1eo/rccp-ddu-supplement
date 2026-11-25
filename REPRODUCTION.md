# Reproducing the numerical experiments

Use Python 3.10 and run the commands below from the repository root unless a command changes directories. The synthetic experiments use generated observations. The archived code and records identify the seeds, tuning rules and shared evaluation contexts. Figure numbers refer to the accompanying IJOC manuscript.

```sh
python -m pip install -r experiments/requirements.txt
```

The recorded scientific stack uses Python 3.10.7. The requirements file specifies the package versions used to check this archive. The archived optimization benchmark additionally used Gurobi 9.5.2 and a solver license.

| Manuscript result | Input and output records | What can be reproduced |
| --- | --- | --- |
| Figure 2, conditional CDF error | `experiments/diagnostics/assets/data/contextual_probability_20261119_*` | Six sample sizes, 60 histories per size, three methods, 1,080 method–history records. The summary and fits regenerate the figure. |
| Figure 3, feasibility and profit | `experiments/figure3/paired_reps70_89/` | 100 independent replication–intensity cells per arm and 40,000 method–context records per arm. Saved decisions regenerate the population evaluations and figure. |
| Figure 4, profit–reliability frontier | `experiments/frontier/` | The original image and layout script regenerate the printed figure. The simulator can produce a fresh frontier. Original numeric coordinates, replication-level frontier results and their original summary CSV are unavailable. |
| Figure 5, computation times | `experiments/runtime/timings.csv` and `runtime_comparison_linear.jpg` | 361 archived runs support numerical summaries. The solver modules and a single-cell driver are included. |
| Archived pooled-residual comparison | `experiments/baseline/` | All 16 archived setting means, including six methods' reliability measures and the workbook's objective entries. |
| Alternative shock distributions | `experiments/diagnostics/assets/data/*shock_robustness*.csv` | CDF errors for two additional shock laws and the three-law decision comparison. Original replication rows and summary tables are included. |
| Expected-loss and pointwise checks | `experiments/diagnostics/assets/data/knn_expected_loss_*` and `stress_pointwise_check*.csv` | 360 kNN loss-error records, plus two early checks on separate replication IDs. |

## Conditional CDF and expected-loss diagnostics

`theory_validation.py` generates the normal history from seed 20261119, evaluates each estimate on the 15 × 9 × 61 grid and computes Student intervals. Let `r=(log(N)/N)^(1/6)`. The Figure 2 files fit `1-exp(-C*r)` to kNN and `C*r` to LSA, minimizing squared errors with weights `sqrt(N)`. The fitted coefficients are 3.528093 and 1.419865. CART uses empirical leaf weights.

```sh
python experiments/diagnostics/theory_validation.py --output-dir work/cdf
python experiments/diagnostics/cdf_shock_robustness.py --workers 2 --output-dir work/cdf-shocks
```

For a quick redraw, copy `contextual_probability_20261119_summary.csv` from the archive to `work/cdf/`, then run the first command with `--plot-only`. The three-law CDF table combines normal results from the first command with lognormal and Student results from the second. Column `max_probability_error` contains the maximum error for each method and history, and `mean_error` averages those maxima at a fixed sample size.

The archived normal summary includes a legacy column, `baseline_scaled_reference`. The current script writes `fitted_rate_reference` and calculates the fitted lines directly from the six observed means.

The expected-loss companion evaluates the same history using a 31-quantity grid and censored nonnegative demand. It reads the two archived normal CDF CSVs from its own `assets/data/` folder. To run it, copy the entire `experiments/diagnostics/` folder to a scratch directory and run `python knn_expected_loss_validation_20261120.py` inside the copy. The script tests its analytic expectations and writes the new records alongside the input CSVs.

## Figure 3

In the paired comparison, all methods use the same 80 test contexts for each replication and intensity. The four estimated models impose a fitted reliability of 0.95 in the margin arm and 0.90 in the nominal arm. Population evaluation uses the 0.90 requirement for both arms, and the Oracle solves at 0.90 throughout. Each arm uses 6,000 training and 1,200 validation observations per cell. The fixed design is in `margin_design.json`. The independent nominal run on IDs 50–69 is retained as a sensitivity check.

In `population_summary.csv`, `true_feasible_rate` divides truly feasible returned decisions by all attempted contexts. `matched_oracle_profit_share` compares total true expected profit with total Oracle profit on the contexts where that method returned a truly feasible decision. A method can have zero feasible returns, in which case its profit share is unavailable.

The following commands re-evaluate saved decisions in both arms and redraw the figure. Change `--cells` to the newly fitted cell directories to evaluate a fresh run. A full refit starts with `python code/run_margin.py --source code/ccw_endogenous_ccp_experiment.py --output fresh --workers 3` in the same directory.

```sh
cd experiments/figure3
python code/population_diagnostics.py --source code/ccw_endogenous_ccp_experiment.py --cells paired_reps70_89/margin --out fresh/margin_results --start 70 --stop 90 --fitted-alpha .05 --require-complete
python code/population_diagnostics.py --source code/ccw_endogenous_ccp_experiment.py --cells paired_reps70_89/nominal --out fresh/nominal_results --start 70 --stop 90 --fitted-alpha .10 --require-complete
python code/paired_margin_cost.py --nominal fresh/nominal_results --margin fresh/margin_results --out fresh/paired_margin_cost.csv --require-complete
python code/coverage_stability.py fresh/margin_results/pointwise_population.csv fresh/margin_results
python code/fig3_margin.py fresh/margin_results/population_summary.csv --paired-nominal fresh/nominal_results/population_summary.csv --outdir fresh
cd ../..
```

The manuscript uses `experiments/figure3/fig3_margin.pdf`. Its SHA-256 is `ff6c417736c6b31fc13eefac1d569981db172b9ca8e979afa2cd795bff308d69`, matching the manuscript file named `stress_test.pdf`. The original implementation checks compared 2,000 pointwise actions in the nominal run and 1,600 in the paired run. Details are in `nominal_reps50_69/results/implementation_check.json` and `paired_reps70_89/margin_engine_check.json`.

Archive verification regenerated all 3,600 normal and alternative-shock CDF and kNN expected-loss records from the seeds, excluding elapsed time from numerical comparisons. Re-evaluating the 200 saved Figure 3 cells reproduced all 80,000 method–context results, the population summaries and the within-arm paired-profit tables. The rendered Figure 2 and Figure 3 PNGs matched the archived images pixel for pixel.

## Frontier and shock diagnostics

The archived frontier uses 50 replications per DDU intensity across six violation tolerances. It aggregates results for a method when it returns a decision for all 80 contexts of a replication. This inclusion rule differs from Figure 3. The archived drawing identifies methods by color. The large-type layout preserves the plotted data regions of the source PNG.

```sh
python experiments/frontier/layout_archive_fig4.py experiments/frontier/high_endogeneity_pareto_frontier.png work/fig4_large.pdf
python experiments/diagnostics/ccw_endogenous_ccp_experiment.py --output-dir work/frontier
python experiments/diagnostics/stress_shock_robustness.py --workers 2 --output-dir work/stress-shocks
python experiments/diagnostics/stress_pointwise_check.py --replication 0 --output work/pointwise-rep0.csv
```

Create `work/` before running the layout command. The simulator runs all five intensity levels for the frontier and supports `--smoke-test` for a small execution check. Its frontier plotting routine assigns a fixed shape to each violation tolerance. Figure 3 includes the byte-preserved solver version used for the paired action checks.

For the shock comparison, the 270 records comprise three shock laws × three intensities × six replications × five methods. An empty profit or reliability field marks a replication without a complete policy for that method. At an intensity of one, summing CCW and Oracle expected profit over the same complete CCW replications gives ratios of 0.975 (normal), 0.848 (lognormal) and 0.972 (Student).

## Archived timing benchmark

`timings.csv` contains the positive-sample-size rows from `ExperimentResults.xlsx` with descriptive column names. The workbook's single zero-initialization row was removed. Strategies 1–4 correspond to the original formulation, SS, SS–BD and SS–BD–Acc. Modes 1–3 select normal, lognormal and Student shocks. Each row gives a sample size, strategy, mode, seed and time in seconds. Missing combinations have no recorded run.

At sample size 6,000, each of strategies 3 and 4 has four observations in every mode. Their mean times are 3494.58 and 962.27 seconds in mode 1, 9770.59 and 2680.17 in mode 2, and 1664.90 and 499.36 in mode 3. The reductions are 72.46%, 72.57% and 70.01%. Archived source modules use different neighborhood sizes in `scheme3` and `scheme4`, so these contrasts describe the recorded implementations. The exact source revision used on the benchmark server is unverified.

To run one cell with Gurobi installed and licensed, use the following command. The driver preserves the retained batch design, fixes the solver to one thread, and seeds NumPy and Python's tie-breaking RNG. The historical batch driver seeded only NumPy. Machine load, solver version, implementation and tie-breaking affect wall-clock time.

```sh
python experiments/runtime/run_timing.py --sample-size 100 --strategy 4 --mode 1 --seed 0
```

## Archived pooled-residual diagnostic

`archived_setting_means.csv` transcribes the 16 rows of `performanceComparison_parametric.xlsx`. The workbook groups rows by profit target with merged cells, so the CSV repeats the target on each row. In the original headings, `to_` denotes the archived objective and `tc_` denotes estimated reliability. The suffix `emp` corresponds to the orthogonal matching pursuit column in the reported comparison. The accompanying `baseline_reliability_summary.csv` contains the same six reliability columns with the paper's labels. Replication-level records that reproduce these means and their generating code revision were not identified.
