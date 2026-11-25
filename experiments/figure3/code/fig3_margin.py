"""Figure 3 for the fixed safety-margin experiment, replications 70--89.

Usage:
  python fig3_margin.py coeval/margin_results/population_summary.csv \
      --outdir coeval/margin_figures \
      --paired-nominal coeval/nominal_results/population_summary.csv

The optional comparison CSV uses both arms' coverage on the common set of
attempted test contexts. Profit shares in panel B instead use the selected
method's own truly feasible contexts, matched to the public-target Oracle.
Keep plot_population.py alongside this file.
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np

from plot_population import COLORS, LINESTYLES, MARKERS, METHODS, RHOS, _panel, read_summary


def write_paired_coverage(path, margin, nominal):
    fields = ["rho", "method", "n_test_contexts", "n_true_feasible_nominal",
              "n_true_feasible_margin", "coverage_nominal", "coverage_margin",
              "coverage_change_pp"]
    with path.open("w", newline="", encoding="utf8") as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        for rho in RHOS:
            a, b = nominal[rho,"CCW"], margin[rho,"CCW"]
            if a[0] != b[0]:
                raise ValueError("Arm denominators differ at rho={}".format(rho))
            writer.writerow(dict(
                rho=rho, method="CCW", n_test_contexts=b[0],
                n_true_feasible_nominal=a[1], n_true_feasible_margin=b[1],
                coverage_nominal=a[2]/100, coverage_margin=b[2]/100,
                coverage_change_pp=round(b[2]-a[2], 8),
            ))


def render(summary, outdir, paired_nominal=None):
    rows = read_summary(summary, expect_reps=20)
    outdir.mkdir(exist_ok=True, parents=True)
    if paired_nominal is not None:
        write_paired_coverage(outdir/"ccw_paired_coverage.csv", rows,
                              read_summary(paired_nominal, expect_reps=20))
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "mathtext.fontset":"dejavusans",
        "pdf.fonttype": 42, "ps.fonttype": 42, "text.color":"#171B1F",
    })
    fig = plt.figure(figsize=(6.5,2.25),facecolor="white")
    left = fig.add_axes([.110,.292,.374,.563])
    right = fig.add_axes([.596,.292,.381,.563])
    _panel(left,rows,2)
    _panel(right,rows,4)
    left.set_title("(a) Feasible decisions",loc="left",fontsize=9.6,fontweight="bold",pad=7)
    right.set_title("(b) Profit on feasible decisions",loc="left",fontsize=9.6,fontweight="bold",pad=7)
    left.set_ylabel("Test contexts with a\nfeasible decision (%)",fontsize=9,labelpad=4)
    right.set_ylabel("Expected profit\n(% of Oracle)",fontsize=9,labelpad=3)
    left.set_ylim(-4.5,106)
    left.set_yticks([0,25,50,75,100])
    limits = []
    for row in rows.values():
        if np.isfinite(row[4]):
            se = row[5] if np.isfinite(row[5]) else 0.0
            limits.extend([row[4]-se,row[4]+se])
    if limits:
        span = max(2., max(limits)-min(limits))
        right.set_ylim(min(limits)-.10*span,max(limits)+.12*span)
        right.yaxis.set_major_locator(MaxNLocator(nbins=5,steps=[1,2,2.5,5,10]))
    else:
        right.text(.5,.5,"No truly feasible decisions",transform=right.transAxes,
                   ha="center",va="center",fontsize=8)
    fig.text(.536,.142,r"Price-dependence intensity, $\rho$",ha="center",va="center",fontsize=9)
    handles=[Line2D([],[],color=c,ls=l,marker=m,ms=4.5,lw=1.0,label=name)
             for c,l,m,name in zip(COLORS,LINESTYLES,MARKERS,METHODS)]
    fig.legend(handles=handles,loc="center",bbox_to_anchor=(.5,.054),ncol=5,
               frameon=False,fontsize=8.5,handlelength=1.55,
               handletextpad=.40,columnspacing=1.40,borderpad=0)
    output = outdir/"fig3_margin"
    fig.savefig(output.with_suffix(".pdf"),metadata={"Title":"Fixed safety margin: true feasibility and profit"})
    fig.savefig(output.with_suffix(".png"),dpi=400)
    plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("summary",type=Path)
    parser.add_argument("--outdir",type=Path)
    parser.add_argument("--paired-nominal",type=Path)
    args=parser.parse_args()
    render(args.summary,args.outdir or args.summary.resolve().parent,args.paired_nominal)
