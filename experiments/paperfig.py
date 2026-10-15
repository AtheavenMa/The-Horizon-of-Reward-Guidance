"""paperfig.py -- venue-agnostic paper-figure configuration.

Usage:
    from paperfig import use_venue, W
    use_venue("iclr")
    fig, ax = plt.subplots(figsize=(W("col", 0.8), 2.1))

Layout numbers come from the style-file sources / official format instructions.
For this project the relevant one is iclr2027_conference.sty: \textwidth 5.5 true in,
Letter, 10/11 pt Times.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from cycler import cycler

# ---------------------------------------------------------------- venues
VENUES = {
    # single column
    "iclr":    dict(col=5.5,   text=5.5,   cols=1, body=10, fig_min=None, family="times"),
    "neurips": dict(col=5.5,   text=5.5,   cols=1, body=10, fig_min=None, family="times"),
    "tmlr":    dict(col=6.5,   text=6.5,   cols=1, body=10, fig_min=None, family="cmbright"),
    "lncs":    dict(col=4.803, text=4.803, cols=1, body=10, fig_min=None, family="cm"),
    # two column
    "aaai":    dict(col=3.3,   text=6.975, cols=2, body=10, fig_min=9,    family="times"),
    "icml":    dict(col=3.25,  text=6.75,  cols=2, body=10, fig_min=None, family="times"),
    "cvpr":    dict(col=3.281, text=6.875, cols=2, body=10, fig_min=None, family="times"),
    "acl":     dict(col=3.031, text=6.299, cols=2, body=10, fig_min=None, family="times"),
    "ijcai":   dict(col=3.375, text=7.0,   cols=2, body=10, fig_min=None, family="times"),
    "acm":     dict(col=3.333, text=7.0,   cols=2, body=9,  fig_min=None, family="libertine"),
    "ieee":    dict(col=3.5,   text=7.16,  cols=2, body=10, fig_min=None, family="times"),
}

FONT_FAMILIES = {
    "times":     dict(family="serif",
                      serif=["TeX Gyre Termes", "Times New Roman", "Nimbus Roman",
                             "Liberation Serif", "STIXGeneral"],
                      # ICLR/NeurIPS/ICML/AAAI/CVPR load `times` for text only and
                      # leave math in Computer Modern, so figure math matches the
                      # body when mathtext stays "cm" rather than "stix".
                      mathtext="cm"),
    "cm":        dict(family="serif", serif=["cmr10", "CMU Serif", "DejaVu Serif"], mathtext="cm"),
    "cmbright":  dict(family="sans-serif", sans=["CMU Bright", "DejaVu Sans"], mathtext="cm"),
    "libertine": dict(family="serif", serif=["Linux Libertine O", "STIXGeneral"], mathtext="stix"),
}

# Paul Tol "bright" -- colour-blind safe.
TOL_BRIGHT = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377", "#BBBBBB"]

# Project-level semantic colours (see figures.py).
INK   = "#222222"   # emphasised / primary curve
BLUE  = "#4477AA"
RED   = "#CC3311"
TEAL  = "#009988"
GREY  = "#8A8A8A"   # reference lines, secondary information
FAINT = "#BFBFBF"   # gridlines, ideal/unreachable references

_current = {"venue": None}


def make_rc(venue, font_size=None, minor_ticks=False, grid="y", color_cycle=TOL_BRIGHT):
    v = VENUES[venue]
    fs = font_size or v["fig_min"] or (v["body"] - 1)
    ff = FONT_FAMILIES[v["family"]]
    rc = {
        "figure.figsize": (v["col"], v["col"] * 0.62),
        "figure.dpi": 150,
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.02,
        "figure.constrained_layout.w_pad": 0.02,
        # fonts
        "font.family": ff["family"],
        "mathtext.fontset": ff["mathtext"],
        "axes.formatter.use_mathtext": True,
        # sizes: body - 1 pt everywhere, no exceptions
        "font.size": fs, "axes.labelsize": fs, "axes.titlesize": fs,
        "xtick.labelsize": fs, "ytick.labelsize": fs,
        "legend.fontsize": fs, "legend.title_fontsize": fs,
        # line weights: nothing below 0.5 pt
        "axes.linewidth": 0.6, "lines.linewidth": 1.2,
        "lines.markersize": 3.0, "lines.markeredgewidth": 0.6,
        "grid.linewidth": 0.5, "patch.linewidth": 0.6,
        # ticks
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
        "xtick.minor.visible": minor_ticks, "ytick.minor.visible": minor_ticks,
        "xtick.top": False, "ytick.right": False,
        "xtick.major.pad": 2.5, "ytick.major.pad": 2.5,
        "xtick.color": "#444444", "ytick.color": "#444444",
        "xtick.labelcolor": "#222222", "ytick.labelcolor": "#222222",
        # spines
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#444444", "axes.labelcolor": "#222222",
        # grid
        "axes.grid": bool(grid), "axes.grid.axis": grid or "both",
        "grid.color": "#C8C8C8", "grid.alpha": 0.6, "grid.linestyle": "-",
        "axes.axisbelow": True,
        # legend
        "legend.frameon": False, "legend.handlelength": 1.8,
        "legend.handletextpad": 0.5, "legend.columnspacing": 1.2,
        "legend.labelspacing": 0.3, "legend.borderaxespad": 0.2,
        # colours
        "axes.prop_cycle": cycler(color=color_cycle),
        # output
        "savefig.format": "pdf", "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42, "ps.fonttype": 42,   # no Type 3 bitmap fonts
    }
    if "serif" in ff:
        rc["font.serif"] = ff["serif"]
    if "sans" in ff:
        rc["font.sans-serif"] = ff["sans"]
    return rc


def use_venue(venue, **kw):
    plt.rcParams.update(make_rc(venue, **kw))
    _current["venue"] = venue
    return VENUES[venue]


def W(kind="col", frac=1.0):
    """Width in inches for the current venue. kind: 'col' | 'text'."""
    return VENUES[_current["venue"]][kind] * frac


if __name__ == "__main__":
    for name, v in VENUES.items():
        print(f"{name:8s} col={v['col']:.3f}in text={v['text']:.3f}in "
              f"cols={v['cols']} body={v['body']}pt")
