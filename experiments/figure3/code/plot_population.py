"""Plot population coverage and matched-Oracle profit from the audited summary.

Use: python plot_population.py population/population_summary.csv --outdir population --expect-reps 20
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


RHOS = (0., .25, .5, .75, 1.)
METHODS = ("CCW", "CQ-CCP", "P-CCP", "SAA-CCP", "Oracle")
COLORS = ("#006FA6", "#C45000", "#00865E", "#AC5593", "#33363D")
MARKERS = ("o", "s", "^", "D", "X")
LINESTYLES = ("-", (0,(4,2)), (0,(4,1.5,1,1.5)), (0,(1,1.4)), (0,(4,3)))


def _float(text):
    return float(text) if text and text.strip() else np.nan


def read_summary(path, expect_reps=None):
    rows = {}
    with open(path, encoding="utf-8-sig", newline="") as infile:
        for row in csv.DictReader(infile):
            rho, method = float(row["rho"]), row["method"]
            key = (rho, method)
            if key in rows:
                raise ValueError(f"Duplicate summary for {key}")
            total = int(row["n_contexts"])
            count = int(row["n_true_feasible"])
            if expect_reps is not None and int(row["n_reps"]) != expect_reps:
                raise ValueError(f"Expected {expect_reps} replications for {key}")
            coverage = float(row["true_feasible_rate"])
            profit = _float(row["matched_oracle_profit_share"])
            if total <= 0 or not 0 <= count <= total:
                raise ValueError(f"Invalid coverage denominator or count for {key}")
            if not np.isclose(coverage, count/total, atol=1e-7, rtol=0):
                raise ValueError(f"Rate inconsistent with its counts for {key}")
            if (count > 0) != np.isfinite(profit):
                raise ValueError(f"Matched profit availability inconsistent for {key}")
            rows[key] = (total, count, 100*coverage, 100*_float(row["true_feasible_rate_se"]),
                         100*profit, 100*_float(row["matched_oracle_profit_share_se"]))
    expected = {(rho,method) for rho in RHOS for method in METHODS}
    if expected != set(rows):
        raise ValueError(f"Expected all five methods at each of {RHOS}. "
                         f"Missing {expected-set(rows)}; extra {set(rows)-expected}")
    for rho in RHOS:
        if len({rows[rho,method][0] for method in METHODS}) != 1:
            raise ValueError(f"Methods have different evaluation sets at rho={rho}")
    return rows


def _panel(ax, rows, metric_index):
    centers = np.arange(len(RHOS))
    offsets = np.linspace(-.21,.21,len(METHODS))
    for j, method in enumerate(METHODS):
        values = np.array([rows[rho,method][metric_index] for rho in RHOS])
        errors = np.array([rows[rho,method][metric_index+1] for rho in RHOS])
        x = centers+offsets[j]
        # A NaN interrupts the line at an empty feasible set; no interpolation
        # is drawn across an unobserved pair of nonadjacent intensities.
        ax.plot(x,values,color=COLORS[j],linestyle=LINESTYLES[j],
                linewidth=1.12 if j==0 else .85,zorder=2)
        valid = np.isfinite(values) & np.isfinite(errors) & (errors > 0)
        if valid.any():
            ax.errorbar(x[valid],values[valid],yerr=errors[valid],fmt="none",
                        ecolor=COLORS[j],elinewidth=.7,capsize=1.8,
                        capthick=.7,zorder=3)
        ax.plot(x,values,linestyle="none",marker=MARKERS[j],ms=4.5 if j==0 else 4.15,
                color=COLORS[j],mec=COLORS[j],mew=.6,zorder=4)
    ax.set_xticks(centers)
    ax.set_xticklabels(("0","0.25","0.50","0.75","1"))
    ax.set_xlim(-.36,4.36)
    ax.tick_params(labelsize=8.35,length=2.6,width=.6,pad=2.5,
                   colors="#252A2D")
    ax.set_axisbelow(True)
    ax.grid(axis="y",linewidth=.48,color="#D9DDE0")
    for name in ("top","right"):
        ax.spines[name].set_visible(False)
    for name in ("bottom","left"):
        ax.spines[name].set_linewidth(.64)
        ax.spines[name].set_color("#4C5359")


def _counts_table(fig, rows):
    """Counts supply every denominator for the conditional profit comparison."""
    col_centers = (.356,.50,.643,.785,.925)
    fig.text(.043,.329,"Truly feasible decisions",weight="bold",
             ha="left",va="center",fontsize=8.65)
    for x,label in zip(col_centers,("0","0.25","0.50","0.75","1")):
        fig.text(x,.329,r"$\rho="+label+"$",ha="center",va="center",fontsize=8.65)
    fig.add_artist(Line2D([.042,.984],[.302,.302],
                          color="#A1A9AD",lw=.55,transform=fig.transFigure))
    for j,(method,y) in enumerate(zip(METHODS,(.277,.228,.179,.130,.081))):
        fig.add_artist(Line2D([.048,.090],[y,y],color=COLORS[j],
                              lw=1.10,ls=LINESTYLES[j],transform=fig.transFigure))
        fig.add_artist(Line2D([.069],[y],color=COLORS[j],ls="none",
                              marker=MARKERS[j],markersize=4.6,
                              transform=fig.transFigure))
        fig.text(.102,y,method,ha="left",va="center",fontsize=8.8)
        for x,rho in zip(col_centers,RHOS):
            count = rows[rho,method][1]
            fig.text(x,y,f"{count:,}",ha="center",va="center",fontsize=8.65)


def render(summary_path, outdir, expect_reps=None):
    rows = read_summary(summary_path, expect_reps)
    outdir.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,
                         "pdf.fonttype":42,"ps.fonttype":42,
                         "text.color":"#15191D","axes.labelcolor":"#15191D"})
    fig = plt.figure(figsize=(6.5,3.46),facecolor="white")
    left = fig.add_axes([.087,.498,.402,.383])
    right = fig.add_axes([.579,.498,.402,.383])
    for ax,label,title,i in [(left,"Test contexts (%)","(a) Decisions meeting the 90% target",2),
                             (right,"Profit (% of matched Oracle)","(b) Profit on truly feasible decisions",4)]:
        _panel(ax,rows,i)
        ax.set_ylabel(label,fontsize=9,labelpad=3)
        ax.set_title(title,loc="left",fontsize=9.15,fontweight="bold",pad=6.2)
    left.set_ylim(-5,105)
    left.set_yticks([0,25,50,75,100])
    vals = np.array([(rec[4],rec[5]) for rec in rows.values() if np.isfinite(rec[4])])
    lower = np.min(vals[:,0]-np.nan_to_num(vals[:,1]))
    upper = np.max(vals[:,0]+np.nan_to_num(vals[:,1]))
    span = max(2.,upper-lower)
    right.set_ylim(lower-.09*span,upper+.10*span)
    right.yaxis.set_major_locator(MaxNLocator(nbins=5,steps=[1,2,2.5,5,10]))
    fig.text(.535,.389,r"Price-dependence intensity, $\rho$",ha="center",
             va="center",fontsize=9.2)
    _counts_table(fig,rows)
    fig.text(.5,.023,"Bars show ±1 SE, clustered by replication. "
             "Profit shares use each method’s truly feasible contexts.",
             ha="center",va="center",fontsize=8.0)
    output = outdir/"fig3_feasible_profit"
    fig.savefig(output.with_suffix(".pdf"),metadata={"Title":
        "Population feasibility and profit relative to the matched Oracle"})
    fig.savefig(output.with_suffix(".png"),dpi=400)
    plt.close(fig)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary",type=Path)
    parser.add_argument("--outdir",type=Path)
    parser.add_argument("--expect-reps",type=int)
    args = parser.parse_args()
    render(args.summary,args.outdir or args.summary.resolve().parent,args.expect_reps)
