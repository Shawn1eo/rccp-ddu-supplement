"""Reproduce Figure 4 at print size from its archived bitmap.

Usage: python plot_fig4.py high_endogeneity_pareto_frontier.png fig4_large.pdf
Requires Pillow, numpy, and matplotlib. Source data areas are embedded as
unresampled image crops; all text and the method key are redrawn as vectors.
Tick centers refer to pixel centers in the original 3868 by 1209 PNG.
"""

import argparse
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image


SOURCE_HASH = "55d17d706e07362d4f823e7d2cc5dd3cb33e04711da743e3b9b94f3f8eb3ab27"
WIDTH_IN = 6.5
HEIGHT_IN = 1.84
PLOT_WIDTH_IN = 1.74
PLOT_BOTTOM_IN = 0.72

# Pillow crop coordinates are half open. Each crop includes the original
# enclosing frame as well as every pixel of the data region.
PANELS = (
    {
        "box": (196, 304, 1254, 805), "left_in": 0.53, "rho": "0.50",
        "xpos": (361, 535, 710, 884, 1058, 1232),
        "xtext": ("0.5", "0.6", "0.7", "0.8", "0.9", "1.0"),
        "ypos": (758, 582, 405), "ytext": ("200", "400", "600"),
    },
    {
        "box": (1479, 304, 2536, 805), "left_in": 2.61, "rho": "0.75",
        "xpos": (1580, 1742, 1904, 2065, 2227, 2389),
        "xtext": ("0.70", "0.75", "0.80", "0.85", "0.90", "0.95"),
        "ypos": (791, 683, 574, 466, 357), "ytext": ("460", "480", "500", "520", "540"),
    },
    {
        "box": (2762, 304, 3819, 805), "left_in": 4.68, "rho": "1.00",
        "xpos": (2802, 3053, 3304, 3555, 3806),
        "xtext": ("0.6", "0.7", "0.8", "0.9", "1.0"),
        "ypos": (743, 621, 500, 379), "ytext": ("400", "425", "450", "475"),
    },
)
METHODS = (
    ("CCW", "#0072b2"), ("CQ-CCP", "#d55e00"), ("P-CCP", "#009e73"),
    ("SAA-CCP", "#cc79a7"), ("Oracle", "#222222"),
)


def draw_panel(fig, archive, panel):
    left, top, right, bottom = panel["box"]
    width, height = right - left, bottom - top
    height_in = PLOT_WIDTH_IN * height / width
    ax = fig.add_axes([
        panel["left_in"] / WIDTH_IN, PLOT_BOTTOM_IN / HEIGHT_IN,
        PLOT_WIDTH_IN / WIDTH_IN, height_in / HEIGHT_IN,
    ])
    ax.imshow(
        np.asarray(archive.crop(panel["box"])), extent=(0, 1, 0, 1),
        origin="upper", interpolation="none", resample=False, aspect="auto",
    )
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.set_xticks([(x - left + 0.5) / width for x in panel["xpos"]])
    ax.set_xticklabels(panel["xtext"])
    ax.set_yticks([(bottom - y - 0.5) / height for y in panel["ypos"]])
    ax.set_yticklabels(panel["ytext"])
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontstretch("condensed")
    ax.tick_params(length=2.4, width=0.55, pad=2.1, labelsize=8)
    ax.set_title(r"$\rho = " + panel["rho"] + "$", fontsize=9.3, pad=3.5)
    return height_in


def render(source, output):
    if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_HASH:
        raise ValueError("The input does not match the archived source figure.")
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5, "pdf.fonttype": 42,
        "pdf.compression": 9, "image.composite_image": False,
        "mathtext.fontset": "dejavusans",
    })
    fig = plt.figure(figsize=(WIDTH_IN, HEIGHT_IN))
    with Image.open(source) as source_image:
        archive = source_image.convert("RGB")
        for panel in PANELS:
            height_in = draw_panel(fig, archive, panel)
    fig.text(
        0.102 / WIDTH_IN, (PLOT_BOTTOM_IN + height_in / 2) / HEIGHT_IN,
        "True expected profit", fontsize=9, ha="center", va="center", rotation=90,
    )
    fig.text(0.5, 0.385 / HEIGHT_IN, "True reliability", fontsize=9, ha="center", va="center")
    handles = [
        Line2D([], [], color=color, marker="o", ms=4.4, lw=1.15, label=method)
        for method, color in METHODS
    ]
    fig.legend(
        handles=handles, loc="center", bbox_to_anchor=(0.5, 0.135 / HEIGHT_IN),
        frameon=False, ncol=5, fontsize=8.5, handlelength=1.6,
        handletextpad=0.45, columnspacing=1.5, borderaxespad=0,
    )
    fig.savefig(str(output), metadata={"Title": "Profit and reliability at higher DDU intensities"})
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    render(args.source, args.output)
