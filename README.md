# Contextual Chance Constraints under Decision-Dependent Uncertainty

**[Read the Online Appendix (PDF, 9 pages)](Online-Appendix.pdf)**

The paper’s “Online Appendix” is the PDF titled *Computational Supplement*.

| Section | PDF pages | What you’ll find | LaTeX source |
|---|---|---|---|
| **S1 · Experimental Design and Reproducibility** | 1–5 | How the synthetic experiments vary price dependence and demand shocks, and how they measure estimation error, decision feasibility, expected loss, and runtime. | [S1 source](supplement/experiments.tex) |
| **S2 · JD.com Evaluation Conventions** | 5–6 | How the study screens sales data, estimates demand, and sets costs and candidate prices. It also explains how to interpret the reported feasibility rates and profit ratios. | [S2 source](supplement/case-study.tex) |
| **S3 · Cluster Matching** | 6–7 | How k-nearest neighbors and local sample average select historical observations for a decision and context, and how a k-d tree precomputes the local sample average matches. | [S3 source](supplement/cluster-matching.tex) |
| **S4 · Solution by Logic-Based Benders Decomposition** | 7–8 | How the master problem selects a candidate decision, how the subproblem evaluates it, and how feasibility cuts, optimality cuts, and the optimality gap govern the search. | [S4 source](supplement/benders.tex) |
| **S5 · Verification of Sufficient Conditions** | 8–9 | Examples showing when the paper’s covering condition (Assumption 4) and Lipschitz smoothness condition (Assumption 2) hold. | [S5 source](supplement/sufficient-conditions.tex) |

**Quick links to the experimental setup in S1**

| Section | Where to look in the PDF | LaTeX source |
|---|---|---|
| **S1.1 · Archived Pooled-Residual Diagnostic** | Starts on **page 1**. At the top of **page 2**, Table S1 reports reliability across the 16 baseline settings, including the CART result. | [S1.1 source](supplement/experiments.tex#L5) |
| **S1.2 · Controlled DDU Stress Test** | Starts on **page 2**, immediately below Table S1. Equation (S2) gives the demand location and scale functions. Equation (S3) introduces the strength of price dependence. | [S1.2 source](supplement/experiments.tex#L34) |

This research archive accompanies the November 25, 2026 manuscript by Xiangting Liu, Shengran Wang, Kaile Yan and Zhi-Hai Zhang. It contains the computational formulations for contextual cluster weights, synthetic experiment designs and outputs, and the reported JD.com case results. The proof appendix is distributed with the manuscript.

## Cite

Liu, X., S. Wang, K. Yan and Z.-H. Zhang (2026). *Contextual Chance Constraints under Decision-Dependent Uncertainty*. Manuscript and accompanying research archive, November 25, 2026. The journal repository identifier and the publication DOIs are pending.

## Replicate the results

The following commands require only Python 3. They read the supplied records without a solver or model training.

```sh
python scripts/check_reported.py
python case/read_table.py
```

The first command recalculates Figure 2 mean CDF errors, Figure 3 feasibility rates and matched profit shares, and the Figure 5 runtime reductions at 6,000 observations. The second displays the twelve reported rows in the JD.com case table. The full instructions include environment details and commands for regenerating decisions, diagnostics and plots. [Reproduction instructions](REPRODUCTION.md)

| Material | Contents |
|---|---|
| [Computational supplement](supplement/computational-supplement.pdf) · [LaTeX source](supplement/computational-supplement.tex) | S1 numerical designs and robustness experiments, S2 JD.com data and evaluation, S3 cluster matching, S4 Benders decomposition, S5 sufficient-condition examples. |
| [Conditional estimation and robustness](experiments/diagnostics/) | Simulators, designs, replicate-level records and summaries for the CDF, expected-loss, pointwise and alternative-shock experiments. |
| [Figure 3](experiments/figure3/) | Fitted 0.95 and nominal 0.90 comparisons, saved decisions, population evaluations and plots. The paired design compares both fitted requirements on the same contexts. |
| [Figure 4](experiments/frontier/) | Archived frontier image and the layout used for the manuscript. The historical numeric coordinates and replicate-level CSV for this figure have not been recovered. |
| [Runtime and solver implementations](experiments/runtime/) | Archived algorithms, one-cell benchmark driver and 361 run times underlying the Figure 5 comparison. |
| [Pooled-residual comparison](experiments/baseline/) | Archived means for 16 settings in Section S1. |
| [JD.com case](case/README.md) | Data access, reported results, the archived Figure 6 and historical evaluation code. The historical workbook covers nine settings and does not regenerate the paper's twelve-row table. |

## License

The authors release their code under MIT and their documentation, synthetic data and simulation results under CC BY 4.0. Third-party materials retain their original terms. JD.com supplies its underlying data through the MSOM challenge. [License terms](LICENSE) · [Authors](AUTHORS) · [JD.com access and attribution](case/README.md)
