# JD.com case study

The main paper compares a CCW price and quantity decision with the historical decision under a fitted distribution of daily demand. The CSV contains all twelve settings of the paper's case-study table to the precision printed in the November 25, 2026 manuscript. Each record gives the profit target in RMB, the required-probability label printed as 1−α, the reported CCW-to-historical profit ratio, and the historical and CCW target-attainment rates. The three printed historical rates apply to all four settings at their respective targets. [Reported table](reported-results.csv) · [Computational supplement, Section S2](../supplement/computational-supplement.pdf)

## Read the reported results

Run the following command from the repository root using Python 3. It uses the standard library. The CSV reproduces the reported table by transcription. Recomputing estimates from individual decisions requires the evaluation records behind that table.

```sh
python case/read_table.py
```

## Obtain the source data

The underlying March 2018 orders and SKU attributes are supplied by JD.com through the 2020 MSOM Data-Driven Research Challenge. MSOM directed members to the challenge page for access. Obtain the data from the provider under its current terms. Cite Max Shen, Christopher S. Tang, Di Wu, Rong Yuan and Wei Zhou, *JD.com: Transaction-Level Data for the 2020 MSOM Data Driven Research Challenge*, *Manufacturing & Service Operations Management* 26(1), 2–10. [Original dataset description](https://doi.org/10.1287/msom.2020.0900) · [MSOM access announcement](https://connect.informs.org/communities/community-home/digestviewer/viewthread?MessageKey=17aed2fe-eca1-4eee-9cd0-75d200a2dc8f) · [Challenge access](https://connect.informs.org/msom/events/datadriven2020)

The reported preprocessing aggregates orders by SKU and day and merges the product attributes and weekend indicator. The available local daily-sales file has 2,683 rows for 90 SKUs. A retained filtered file has 410 rows for 14 SKUs. A separate local scores file has only three of its 90 recorded scores at or above 0.88, so the retained files do not establish a single reproducible screening run. Transaction and SKU-level files, predictions, fitted model weights and the original challenge tables are excluded from this repository.

## Historical evaluation code and provenance

[regression.py](historical/regression.py) records a PyCaret demand-model comparison. [CaseStudy.py](historical/CaseStudy.py) records a later evaluation driver using a saved model and pooled residuals. The driver imports the archived clustering implementations in [Scheme.py](../experiments/runtime/code/Scheme.py). The original raw-orders-to-daily-table script and a matched model, split, and evaluation output for the reported table have not been identified.

| Item | Retained implementation |
|---|---|
| Evaluation sample | The driver samples 80 rows using seed 42 and makes the full screened frame available for neighborhood construction. It does not identify the paper's held-out split. |
| Prices and costs | The driver sets cost to the row's final unit price divided by 1.021 and salvage to 0.997 times cost. Its 35 candidate prices start at the integer part of that row's final price minus 2, increase in steps of 0.1 and are filtered to exceed cost. |
| Feasibility and profit | The kNN routine indexes sorted demand at `-ceil(k*alpha)`. The driver counts draws attaining the target and adds realized profit only for those draws, dividing both totals by all residual draws. |
| Inclusion | The driver skips solver exceptions, rows with a zero selected price for any of its three methods, and rows with historical objective below 10. These rules and the workbook cannot establish how the publication table handled missing decisions. |

The archived workbook titled `CaseStudy.xlsx` contains 246 rows covering nine of the twelve settings. For example, at target 0 and setting 0.1 the mean of the 28 row-level kNN-to-historical objective ratios is 1.218, whereas the reported ratio is 6.242. None of the nine common rows reproduces all three reported statistics to three decimals. The paper's displayed 0.90 reliability row gives CCW mean rates of 0.912, 0.903 and 0.902. Interpreting the table uses its printed required-probability heading. The exact code and observations that produced those numbers remain unverified.

For authorized local holders of that workbook, the following command prints the grouped means alongside a match flag. It requires `openpyxl`. The workbook audited here has SHA-256 `af99cdf3efc5032fadf23e93f9d2e95dbf733e16e5e9779c032712a516da566a`.

```sh
python case/read_table.py --historical-workbook /path/to/CaseStudy.xlsx
```

## Figure 6

The archived histogram prints a mean ratio of 1.96. The original November 25 manuscript reported that the ratio exceeded one in 88.4% of selected observations. The exact observations behind the plot are unavailable. The retained plotting script filters the historical workbook to a target of RMB 10 and pools the available setting labels. The workbook contains no observations for target 10 with label 0.9. The available records therefore cannot establish which reliability setting or denominator produced that percentage. [Archived figure](figure6.png) · [Historical plotting script](historical/CaseAnalysis.py)
