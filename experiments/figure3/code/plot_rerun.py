"""Plot the fixed-alpha independent replication check from its summary CSV.

Usage: python plot_rerun.py summary.csv output_directory
Required CSV fields: rho, method, n_reps, n_complete, mean_context_return_rate,
policy_feasible_rate, expected_profit, profit_se, reliability, reliability_se.
Requires NumPy and Matplotlib. The charts and their lettering are vector PDF.
"""

import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
import numpy as np


METHODS = ("CCW", "CQ-CCP", "P-CCP", "SAA-CCP", "Oracle")
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#222222")
MARKERS = ("o", "s", "^", "D", "v")
RHOS = (0.0, 0.5, 1.0)
STUDY_LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)
TARGET = 0.90
WIDTH = 6.5
RETURN_CMAP = LinearSegmentedColormap.from_list(
    "rates", ["#F7F9FB", "#D6E5EB", "#87AFBF", "#376F87", "#1D4D66"]
)


def number(value):
    if value is None or str(value).strip().lower() in ("", "na", "nan", "null", "none"):
        return math.nan
    return float(value)


def read_summary(csv_path):
    data = {}
    with csv_path.open(encoding="utf-8-sig", newline="") as source:
        for record in csv.DictReader(source):
            if "alpha" in record and not np.isclose(number(record["alpha"]), 0.10):
                continue
            rho, method = float(record["rho"]), record["method"].strip()
            if rho not in STUDY_LEVELS or method not in METHODS:
                continue
            key = (rho, method)
            if key in data:
                raise ValueError("Duplicate record for {}".format(key))
            r = {key: number(record[key]) for key in
                 ("n_reps", "n_complete", "policy_feasible_rate", "mean_context_return_rate",
                  "expected_profit", "profit_se", "reliability", "reliability_se")}
            if not 0 <= r["n_complete"] <= r["n_reps"] or r["n_reps"] <= 0:
                raise ValueError("Invalid replication denominators for {}".format(key))
            for field in ("n_complete", "n_reps"):
                if not r[field].is_integer():
                    raise ValueError("Counts must be integers")
                r[field] = int(r[field])
            for field in ("policy_feasible_rate", "mean_context_return_rate"):
                if not 0 <= r[field] <= 1:
                    raise ValueError("Invalid {} for {}".format(field, key))
            if not np.isclose(r["policy_feasible_rate"], r["n_complete"] / r["n_reps"], atol=1e-6):
                raise ValueError("Complete-run count and fraction disagree for {}".format(key))
            if r["policy_feasible_rate"] > r["mean_context_return_rate"] + 1e-6:
                raise ValueError("Whole-policy return cannot exceed context-level return")
            if r["n_complete"]:
                if not (0 <= r["reliability"] <= 1 and np.isfinite(r["expected_profit"])):
                    raise ValueError("Missing conditional performance for {}".format(key))
            for field in ("profit_se", "reliability_se"):
                if np.isfinite(r[field]) and r[field] < 0:
                    raise ValueError("Negative SE for {}".format(key))
            data[key] = r
    available_levels = {r for r, _ in data}.union(RHOS)
    missing = [(r, m) for r in available_levels for m in METHODS if (r, m) not in data]
    if missing:
        raise ValueError("Missing aggregate records: {}".format(missing))
    return data


def pct(rate):
    if rate == 0:
        return "0%"
    if rate == 1:
        return "100%"
    raw = 100 * rate
    digits = 2 if raw > 99.9 or raw < 0.1 else 1
    text = ("{:.%df}" % digits).format(raw).rstrip("0").rstrip(".")
    if text == "100":
        return ">99.99%"
    if text == "0":
        return "<0.01%"
    return text + "%"


def draw_scatter(ax, rows):
    finite_x, finite_y = [], []
    for i, r in enumerate(rows):
        if not r["n_complete"]:
            continue
        x, y = r["reliability"], r["expected_profit"]
        has_x_se = r["n_complete"] > 1 and np.isfinite(r["reliability_se"])
        has_y_se = r["n_complete"] > 1 and np.isfinite(r["profit_se"])
        dx = 1.96 * r["reliability_se"] if has_x_se else 0.0
        dy = 1.96 * r["profit_se"] if has_y_se else 0.0
        finite_x.extend((x-dx, x+dx))
        finite_y.extend((y-dy, y+dy))
        # A missing SE produces neither an error bar nor a lower-bound classification.
        color = COLORS[i]
        filled = has_x_se and x - dx >= TARGET
        ax.errorbar(
            x, y, xerr=dx if has_x_se else None, yerr=dy if has_y_se else None,
            fmt=MARKERS[i], ms=5.0, mew=0.9, mfc=color if filled else "white",
            mec=color, ecolor=color, elinewidth=0.8, capsize=2,
            zorder=4+i/10, clip_on=False,
        )
        if not has_x_se:
            ax.plot(x, y, marker=".", ms=1.8, color=color, zorder=6)
    # At rho=0 every method can lie within a few ten-thousandths of 100%.
    # Use the actual horizontal spread there, with explicit percentage ticks.
    near_one = bool(finite_x and min(finite_x) > .985 and max(finite_x)-min(finite_x) < .01)
    lo = min(finite_x) if near_one else min([TARGET] + finite_x)
    hi = max(finite_x) if near_one else max([TARGET] + finite_x)
    span = max(hi-lo, 2.0e-5 if near_one else .025)
    lo, hi = lo-0.07*span, hi+0.07*span
    if lo <= TARGET <= hi:
        ax.axvline(TARGET,color="#656971",linestyle=(0,(3,2)),linewidth=.75,zorder=1)
    ax.set_xlim(lo, hi)
    ticks = MaxNLocator(nbins=3 if near_one else 4, steps=[1, 2, 2.5, 5, 10]).tick_values(lo,hi)
    ticks = [t for t in ticks if lo <= t <= hi and 0 <= t <= 1.000001]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["{:g}".format(round(100*t, 6)) if not near_one else
                        ("{:.5f}".format(100*t)).rstrip("0").rstrip(".") for t in ticks])
    if finite_y:
        low, high = min(finite_y), max(finite_y)
        dy = max(high-low, 0.008*max(abs(high),1))
        ax.set_ylim(low-0.10*dy, high+0.14*dy)
    else:
        ax.set_yticks([])
        ax.text(.5,.5,"No complete\nreplications", transform=ax.transAxes,
                ha="center", va="center", fontsize=8)
    if finite_y:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.tick_params(labelsize=8, length=2.5, width=.6, pad=2)
    for side in ("top","right"):
        ax.spines[side].set_visible(False)
    for side in ("left","bottom"):
        ax.spines[side].set_linewidth(.6)
    ax.grid(axis="y", linewidth=.4, color="#DDE0E3", zorder=0)
    ax.set_axisbelow(True)


def figure_three(data):
    height = 2.78
    fig = plt.figure(figsize=(WIDTH, height))
    for i, rho in enumerate(RHOS):
        ax = fig.add_axes([(0.60+2.10*i)/WIDTH, 1.40/height, 1.61/WIDTH, 1.08/height])
        draw_scatter(ax, [data[rho,m] for m in METHODS])
        ax.set_title(r"$\rho = " + "{:g}".format(rho) + "$", fontsize=9.5, pad=4)
    fig.text(.013, 1.94/height, "True expected profit", ha="center", va="center",
             rotation=90, fontsize=9)
    fig.text(.52, 1.05/height, "True reliability (%)", ha="center", fontsize=9)

    fig.text(.035,.84/height,"Replications returning a decision at every context",
             fontsize=8.6, va="center")
    tab = fig.add_axes([.035,.015,.945,.69/height])
    tab.set_xlim(0,1); tab.set_ylim(0,1); tab.axis("off")
    tab.plot([0,1],[.73,.73],color="#8B9299",linewidth=.55)
    tab.text(.007,.89,r"$\rho$",va="center",fontsize=8.6)
    for i,m in enumerate(METHODS):
        x = .185+.179*i
        tab.plot(x-.061,.89,marker=MARKERS[i],ms=4.0,color=COLORS[i],clip_on=False)
        tab.text(x-.045,.89,m,ha="left",va="center",fontsize=8.1)
    for j,rho in enumerate(RHOS):
        y = .596-.225*j
        tab.text(.005,y,"{:g}".format(rho),va="center",fontsize=8.1)
        for i,m in enumerate(METHODS):
            r=data[rho,m]
            text = "{}/{} ({})".format(r["n_complete"],r["n_reps"],pct(r["policy_feasible_rate"]))
            tab.text(.185+.179*i,y,text,ha="center",va="center",fontsize=7.9)
    return fig


def diagnostics(data):
    fig = plt.figure(figsize=(WIDTH, 2.45))
    levels = sorted({r for r, _ in data})
    titles = ("Contexts with a decision", "Replications with decisions at all contexts")
    for j,field in enumerate(("mean_context_return_rate","policy_feasible_rate")):
        values = np.array([[data[rho,m][field] for rho in levels] for m in METHODS])
        ax = fig.add_axes([.135+.44*j,.115,.399,.67])
        ax.imshow(values, vmin=0, vmax=1, cmap=RETURN_CMAP, aspect="auto", interpolation="none")
        ax.set_title(titles[j], fontsize=9, pad=7)
        ax.xaxis.tick_top()
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels(["{:g}".format(rho) for rho in levels],fontsize=8.5)
        ax.set_xlabel(r"Price dependence $\rho$",fontsize=8.5,labelpad=4)
        ax.set_yticks(range(len(METHODS)))
        ax.set_yticklabels(METHODS if j==0 else [""]*len(METHODS), fontsize=8.5)
        ax.tick_params(length=0,pad=4)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for k in np.arange(0.5, len(METHODS), 1.0):
            ax.axhline(k,color="white",lw=1.0)
        for k in np.arange(0.5,len(levels),1.0):
            ax.axvline(k,color="white",lw=1.0)
        for a,m in enumerate(METHODS):
            for b,rho in enumerate(levels):
                rate=values[a,b]
                r=data[rho,m]
                label=pct(rate)
                if j==1:
                    label+="\n({}/{})".format(r["n_complete"], r["n_reps"])
                ax.text(b,a,label,ha="center",va="center",fontsize=8.6 if j==0 else 7.8,
                        linespacing=.96,
                        color="white" if rate>=.64 else "#142B36")
    return fig


def main():
    parser=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("summary",type=Path)
    parser.add_argument("output_dir",type=Path)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    data=read_summary(args.summary)
    plt.rcParams.update({"font.family":"DejaVu Sans","pdf.fonttype":42,
                         "mathtext.fontset":"dejavusans","axes.unicode_minus":False})
    for figure, stem in ((figure_three(data),"fig3_rerun"),(diagnostics(data),"feasibility_diagnostics")):
        figure.savefig(str(args.output_dir/(stem+".pdf")))
        plt.close(figure)


if __name__=="__main__":
    main()
