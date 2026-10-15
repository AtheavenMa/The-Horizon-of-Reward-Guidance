"""figures.py -- regenerate every paper figure from the stored result files.

All numerical content is read back from results/*.csv and results/*.json, so the
plots are byte-for-byte the same data the verification scripts produced; only the
presentation changes.  Two curves have no stored CSV and are reconstructed in
closed form from the recorded parameters (see clipped_lag and query_energy).

Run:  python code/figures.py --out results --venue iclr
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paperfig import use_venue, W, BLUE, RED, TEAL, GREY, FAINT, INK

FS = None  # filled in by main(); the single font size used everywhere


# ------------------------------------------------------------------ helpers
def end_label(ax, x, y, text, color, dx=0.012, **kw):
    """Label a curve at its right end, in the curve's own colour (PDS step 4)."""
    xlo, xhi = ax.get_xlim()
    if ax.get_xscale() == "log":
        pos = x * (xhi / xlo) ** dx
    else:
        pos = x + (xhi - xlo) * dx
    return ax.text(pos, y, text, color=color, va="center", ha="left",
                   clip_on=False, **kw)


def hline(ax, y, text, color=GREY, ls=(0, (5, 2)), tx=0.012, va="bottom", lw=0.9):
    ax.axhline(y, color=color, ls=ls, lw=lw, zorder=1.5)
    xlo, xhi = ax.get_xlim()
    ax.text(xlo + (xhi - xlo) * tx, y, text, color=color, va=va, ha="left",
            fontsize=FS)


def crossing(x, y, level):
    """First x where y crosses `level` downwards, by linear interpolation."""
    y = np.asarray(y, float)
    idx = np.where((y[:-1] >= level) & (y[1:] < level))[0]
    if len(idx) == 0:
        return None
    i = idx[0]
    t = (y[i] - level) / (y[i] - y[i + 1])
    return x[i] + t * (x[i + 1] - x[i])


def save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf",
                metadata={"Creator": "", "Producer": "", "CreationDate": None})
    png = out / "png"
    png.mkdir(parents=True, exist_ok=True)
    fig.savefig(png / f"{name}.png", dpi=400)
    plt.close(fig)
    print(f"  wrote {name}.pdf  (+ png/{name}.png)")


# ------------------------------------------------------------------ Figure 1
def fig_closed_loop(res: Path, out: Path):
    d = np.genfromtxt(res / "closed_loop.csv", delimiter=",", names=True)
    meta = json.loads((res / "verification_results.json").read_text())["five_output_closed_loop"]
    gamma = meta["required_progress_normalized"]
    deadline = meta["source_only_exact_deadline"]
    t = d["time"]

    # No end-label margin: the two curve labels sit in the empty interior, so the
    # x-limits are exactly the data range and the figure needs 0.8, not 1.0, of
    # the text width.
    fig, ax = plt.subplots(figsize=(W("text", 0.8), 2.05))
    ax.grid(False)
    ax.set_xlim(0, 15)
    ax.set_ylim(-0.30, 1.14)

    # the uncertified region, as a background band rather than a fourth line
    ax.fill_between([0, 15], -0.30, gamma, color=RED, alpha=0.05, edgecolor="none", zorder=0)

    ax.plot([0, 15], [1.0, 1.0], color=FAINT, ls=(0, (1, 2)), lw=0.9, zorder=1.5)
    ax.text(15, 1.02, "Pointwise optimum ($\\delta=0$)", color="#8f8f8f",
            va="bottom", ha="right", fontsize=FS)
    ax.plot([0, 15], [gamma, gamma], color=GREY, ls=(0, (5, 2)), lw=0.9, zorder=1.5)
    ax.text(15, gamma - 0.03, "Required progress", color=GREY,
            va="top", ha="right", fontsize=FS)

    ax.plot([deadline, deadline], [-0.30, 1.0], color=GREY, ls=(0, (1, 2)),
            lw=0.9, zorder=1.5)
    ax.annotate("no-query deadline $t=5$", xy=(deadline, -0.27),
                xytext=(deadline + 0.25, -0.27), color=GREY,
                fontsize=FS, va="bottom", ha="left")

    ax.plot(t, d["source_certificate_normalized"], color=RED, lw=1.7, zorder=3)
    ax.plot(t, d["post_query_certificate_normalized"], color=BLUE, lw=1.3,
            ls=(0, (4.5, 1.6)), zorder=3)

    ax.plot([deadline], [gamma], marker="o", ms=3.4, mfc="white", mec=RED,
            mew=1.0, zorder=4)

    ax.text(7.6, 0.885, "After 1,200 targeted labels", color=BLUE,
            fontsize=FS, va="bottom", ha="left")
    ax.text(6.3, 0.03, "Source-only certificate", color=RED,
            fontsize=FS, va="bottom", ha="left")

    ax.set_xticks([0, 4, 5, 10, 15])
    ax.set_xlabel("Gradient-flow time")
    ax.set_ylabel("Gain $/\\,\\epsilon^{2}$")
    ax.spines["bottom"].set_bounds(0, 15)
    save(fig, out, "closed_loop")


# ------------------------------------------------------------------ Figure 2
def fig_ordinary_pg(res: Path, out: Path):
    d = np.genfromtxt(res / "ordinary_pg.csv", delimiter=",", names=True)
    t, phi = d["time"], d["phi"]
    lo, hi = d["safe_lower"], d["safe_upper"]
    contact = 0.0078358743

    y0 = 0.84160
    fig, ax = plt.subplots(figsize=(W("col", 0.8), 2.05))
    ax.grid(False)
    ax.set_xlim(0, t[-1])
    ax.set_ylim(y0, 0.84525)

    # everything below the moving boundary has left the safe interval
    ax.fill_between(t, y0, lo, color=RED, alpha=0.055, edgecolor="none", zorder=0)
    ax.plot(t, lo, color=TEAL, lw=1.2, ls=(0, (4.5, 1.6)), zorder=3)
    ax.plot(t, phi, color=INK, lw=1.7, zorder=4)

    ax.axvline(contact, color=GREY, ls=(0, (1, 2)), lw=0.9, zorder=1.5)
    ax.plot([contact], [np.interp(contact, t, phi)], marker="o", ms=3.8,
            mfc="white", mec=INK, mew=1.0, zorder=5)

    ax.set_yticks([0.842, 0.843, 0.844, 0.845])
    ax.set_xticks([0, 0.005, 0.01, 0.015])
    ax.set_xticklabels(["0", "0.005", "0.010", "0.015"])

    ax.text(0.00715, 0.84512, "Moving lower boundary", color=TEAL,
            ha="right", va="top", fontsize=FS)
    ax.text(0.01755, 0.84292, "$\\dot\\phi=.04$, the maximum permitted speed", color=INK,
            ha="right", va="top", fontsize=FS)
    ax.text(0.01755, 0.84168, "below the safe interval", color="#C08A84",
            ha="right", va="bottom", fontsize=FS)
    ax.annotate("contact\n$t=.0078$", xy=(contact, 0.84168),
                xytext=(contact - 0.0004, 0.84168), color=GREY,
                fontsize=FS, ha="right", va="bottom")

    ax.set_xlabel("Policy-gradient flow time")
    ax.set_ylabel("Reward coefficient $\\phi$")
    save(fig, out, "ordinary_pg")


# ------------------------------------------------------------------ Figure 3
def fig_instance_complexity(res: Path, out: Path):
    """(a) cost against T: large spread at fixed T.  (b) the same 14 bounds
    against d_T: they collapse onto one d_T^{-2} line, so d_T is primitive."""
    d = np.genfromtxt(res / "binary_gap_bounds.csv", delimiter=",", names=True)
    deltas = [-0.2, 0.0, 0.2]
    colors = ["#93BBDE", "#3B8AC4", "#0B3D62"]      # sequential: delta* is ordered
    marks = ["v", "o", "^"]
    lss = [(0, (1.6, 1.4)), (0, ()), (0, (4.5, 1.6))]
    mk = dict(ms=3.4, mfc="white", mew=1.0, zorder=3)

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(W("text"), 2.0), sharey=True)
    for a in (ax, bx):
        a.set_yscale("log")
        a.grid(True, axis="y")

    # ---- (a) against the horizon
    ax.set_xlim(4.2, 26)
    for ds, c, m, ls in zip(deltas, colors, marks, lss):
        r = d[d["true_delta"] == ds]
        o = np.argsort(r["T"])
        ax.plot(r["T"][o], r["quadratic_label_lower"][o], color=c, ls=ls,
                lw=1.3, marker=m, mec=c, **mk)
    # direct labels in the empty interior instead of a right-hand margin
    ax.text(14.4, 1788.7, "$\\delta_\\star=0.2$", color=colors[2], ha="right", va="center")
    ax.text(21.5, 900, "$\\delta_\\star=0$", color=colors[1], ha="center", va="bottom")
    ax.text(21, 118, "$\\delta_\\star=-0.2$", color=colors[0], ha="center", va="top")
    # spread at a single horizon
    at15 = d[d["T"] == 15.0]["quadratic_label_lower"]
    lo, hi = at15.min(), at15.max()
    xb = 15.6
    ax.plot([xb, xb], [lo, hi], color=GREY, lw=0.7, zorder=2)
    for yy in (lo, hi):
        ax.plot([xb - 0.25, xb], [yy, yy], color=GREY, lw=0.7, zorder=2)
    ax.text(xb + 0.35, 215, f"{hi / lo:.0f}$\\times$", color=GREY,
            ha="left", va="center")
    ax.set_xticks([5, 10, 15, 20, 25])
    ax.spines["bottom"].set_bounds(5, 25)
    ax.set_xlabel("Required horizon $T$")
    ax.set_ylabel("Necessary expected labels")
    ax.set_title("(a) Against the horizon", loc="left")

    # ---- (b) against the dynamic gap
    const = np.median(d["quadratic_label_lower"] * d["gap"] ** 2)
    g = np.geomspace(0.115, 1.35, 50)
    bx.plot(g, const / g ** 2, color=FAINT, lw=2.4, zorder=1, solid_capstyle="butt")
    bx.text(0.36, const / 0.36 ** 2 * 1.6, "$\\propto d_T^{-2}$", color=GREY,
            ha="left", va="bottom")
    for ds, c, m in zip(deltas, colors, marks):
        r = d[d["true_delta"] == ds]
        bx.plot(r["gap"], r["quadratic_label_lower"], ls="none", marker=m,
                mec=c, **mk)
    bx.set_xscale("log")
    bx.set_xlim(0.115, 1.35)
    bx.set_xticks([0.2, 0.5, 1.0])
    bx.set_xticklabels(["0.2", "0.5", "1"])
    bx.xaxis.set_minor_formatter(plt.NullFormatter())
    bx.set_xlabel("Distance to dynamic infeasibility $d_T$")
    bx.set_title("(b) Against the dynamic gap", loc="left")
    save(fig, out, "instance_complexity")


# ------------------------------------------------------------------ Figure 4
def fig_query_energy(res: Path, out: Path):
    m = json.loads((res / "verification_results.json").read_text())["rank_one_design"]
    r0, floor, ncrit = (m["initial_energy"], m["limiting_energy"],
                        m["minimum_continuous_count_for_half_available_reduction"])
    # exact closed form  E(n) = r0 - n x^2/(1 + n R)  recovered from the record:
    #   r0 - floor = x^2/R  and  ncrit = 1/R
    R = 1.0 / ncrit
    x2 = (r0 - floor) * R
    n = np.arange(0, 201)
    e = r0 - n * x2 / (1.0 + n * R)

    # 99% of the total decrease is over by n = 24, so plotting to n = 200 would
    # spend seven eighths of the width on a flat line.  The check runs to 200;
    # the axis stops at 60 and the dashed line carries the n -> inf limit.
    nmax = 40
    y0 = 0.836
    fig, ax = plt.subplots(figsize=(W("col", 0.55), 1.62))
    ax.grid(False)
    ax.set_xlim(-1.2, nmax)
    ax.set_ylim(y0, 1.42)

    ax.fill_between([-1.2, nmax], y0, floor, color=GREY, alpha=0.10,
                    edgecolor="none", zorder=0)
    ax.axhline(floor, color=GREY, ls=(0, (5, 2)), lw=1.0, zorder=2)
    ax.plot(n, e, color=BLUE, lw=1.7, zorder=3)

    ax.text(nmax / 2, 0.860, "unreachable with this pair alone", color="#8f8f8f",
            fontsize=FS, va="center", ha="center")
    ax.annotate(f"residual floor {floor:.3f}\n($n\\to\\infty$)", xy=(12, floor),
                xytext=(15, 0.955), color=GREY, fontsize=FS, va="bottom", ha="left",
                arrowprops=dict(arrowstyle="-", color=GREY, lw=0.6,
                                shrinkA=1, shrinkB=1))
    ax.text(8, 1.24, "Exact inverse-design\nenergy", color=BLUE, fontsize=FS,
            va="top", ha="left")
    ax.plot([0], [e[0]], marker="o", ms=3.4, mfc="white", mec=BLUE, mew=1.0, zorder=4)

    ax.set_xticks([0, 10, 20, 30, 40])
    ax.set_yticks([0.9, 1.0, 1.1, 1.2, 1.3, 1.4])
    ax.set_xlabel("Repeated comparisons on one pair")
    ax.set_ylabel("Directional uncertainty")
    ax.spines["bottom"].set_bounds(0, nmax)
    save(fig, out, "query_energy")


# ------------------------------------------------------------------ Figure 5
def fig_full_support(res: Path, out: Path):
    d = np.genfromtxt(res / "weak_identification.csv", delimiter=",", names=True)
    eps = d["epsilon"]

    fig, ax = plt.subplots(figsize=(W("col", 0.8), 2.10))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, axis="y")
    ax.set_xlim(7e-4, 3.2e-1)
    ax.set_ylim(2e-6, 2.0)

    ax.plot(eps, d["prediction_risk"], color=BLUE, lw=1.4, ls="-",
            marker="o", ms=3.2, mfc="white", mew=1.0, zorder=3)
    ax.plot(eps, d["identified_gain_variance"], color=RED, lw=1.4,
            ls=(0, (4.5, 1.6)), marker="s", ms=3.0, mfc="white", mew=1.0, zorder=3)

    ax.text(1.4e-2, 0.13, "OOD prediction risk  ($\\propto\\epsilon^{-1}$)", color=BLUE,
            fontsize=FS, va="bottom", ha="center")
    ax.text(1.4e-2, 9.0e-6, "Identified-gain estimation variance  ($\\approx$ const.)",
            color=RED, fontsize=FS, va="bottom", ha="center")

    ax.set_xlabel("Source mismatch probability")
    ax.set_ylabel("Squared error")
    save(fig, out, "full_support_separation")


# ------------------------------------------------------------------ Figure 6
def fig_actual_adaptation(res: Path, out: Path):
    # closed form from verify_v06.clipped_plot: v=.12, rho=.04, k=1, a0=.4
    v, rho, k, a0 = 0.12, 0.04, 1.0, 0.4
    ts = np.log(v / (v - rho)) / k
    t = np.linspace(0, 7, 1401)
    ideal = (v - rho) * t
    clipped = np.where(t <= ts, v / k * (-np.expm1(-k * t)),
                       rho / k + (v - rho) * (t - ts))
    t_ideal = (a0) / (v - rho)
    t_clip = ts + (a0 - rho / k) / (v - rho)

    fig, ax = plt.subplots(figsize=(W("col", 0.8), 2.10))
    ax.grid(False)
    ax.set_xlim(0, 7.05)
    ax.set_ylim(0, 0.62)

    ax.axhline(a0, color=GREY, ls=(0, (5, 2)), lw=1.0, zorder=1.5)
    ax.text(0.12, a0 - 0.014, "Maximum certifiable lag $a_0=.4$", color=GREY,
            fontsize=FS, ha="left", va="top")

    (l1,) = ax.plot(t, ideal, color=GREY, lw=1.2, ls=(0, (4.5, 1.6)), zorder=3,
                    label="Ideal rate-budget controller")
    (l2,) = ax.plot(t, clipped, color=BLUE, lw=1.7, zorder=4,
                    label="Clipped gradient calibration")
    for tc, c in ((t_ideal, GREY), (t_clip, BLUE)):
        ax.plot([tc], [a0], marker="o", ms=3.4, mfc="white", mec=c, mew=1.0, zorder=5)
    ax.legend(handles=[l2, l1], loc="upper left", bbox_to_anchor=(-0.012, 1.04),
              handlelength=2.2)

    # the two flows are visually indistinguishable at this scale; the inset is
    # what shows that the unsaturated phase costs the clipped flow 0.09 of time
    axins = ax.inset_axes([0.545, 0.085, 0.36, 0.34])
    axins.plot(t, ideal, color=GREY, lw=1.1, ls=(0, (4.5, 1.6)))
    axins.plot(t, clipped, color=BLUE, lw=1.5)
    axins.axhline(a0, color=GREY, ls=(0, (5, 2)), lw=0.8)
    for tc, c in ((t_ideal, GREY), (t_clip, BLUE)):
        axins.plot([tc], [a0], marker="o", ms=3.2, mfc="white", mec=c, mew=1.0, zorder=5)
    axins.set_xlim(4.80, 5.12)
    axins.set_ylim(0.3885, 0.4105)
    axins.set_xticks([t_clip, t_ideal])
    axins.set_xticklabels([f"{t_clip:.2f}", f"{t_ideal:.2f}"], fontsize=FS - 1)
    axins.tick_params(axis="x", length=2, pad=1.5)
    axins.set_yticks([])
    axins.grid(False)
    for s in axins.spines.values():
        s.set_visible(True); s.set_linewidth(0.5); s.set_color("#AAAAAA")
    ax.indicate_inset_zoom(axins, edgecolor="#AAAAAA", lw=0.6, alpha=0.9)

    ax.set_xticks([0, 2, 4, 6])
    ax.set_xlabel("Policy-gradient flow time")
    ax.set_ylabel("Uncompensated lag")
    ax.spines["bottom"].set_bounds(0, 7)
    save(fig, out, "actual_adaptation")


# ------------------------------------------------------------------ Figure 7
def fig_discrete_resources(res: Path, out: Path):
    d = np.genfromtxt(res / "discrete_resources.csv", delimiter=",", names=True)
    meta = json.loads((res / "discrete_resources.json").read_text())
    gamma = meta["gamma"]
    ms = sorted(set(int(v) for v in d["reward_steps"]))
    colors = ["#93BBDE", "#4E92C2", "#26678F", "#0B3D62"]
    lss = [(0, (1.6, 1.4)), (0, (4.5, 1.6)), (0, (6, 1.6, 1.2, 1.6)), (0, ())]

    fig, ax = plt.subplots(figsize=(W("col", 0.8), 2.15))
    ax.grid(False)
    ax.set_xlim(0, 18.6)
    ax.set_ylim(-0.90, 1.10)

    ax.fill_between([0, 15], -0.90, gamma, color=RED, alpha=0.05, edgecolor="none", zorder=0)
    ax.plot([0, 15], [gamma, gamma], color=GREY, ls=(0, (5, 2)), lw=0.9, zorder=1.5)
    ax.text(10.6, gamma + 0.02, "Required progress", color=GREY,
            va="bottom", ha="left", fontsize=FS)

    for m, c, ls in zip(ms, colors, lss):
        r = d[d["reward_steps"] == m]
        o = np.argsort(r["time"])
        x, y = r["time"][o], r["worst_normalized_gain"][o]
        lw = 1.7 if m == ms[-1] else 1.2
        ax.plot(x, y, color=c, ls=ls, lw=lw, zorder=3)
        end_label(ax, x[-1], y[-1], f"$m={m}$", c, dx=0.012)
        xc = crossing(x, y, gamma)
        if xc is not None:
            ax.plot([xc], [gamma], marker="o", ms=3.2, mfc="white", mec=c,
                    mew=1.0, zorder=4)

    ax.text(6.4, 1.04, "$m$: reward steps per policy step", color="#8f8f8f",
            fontsize=FS, ha="left", va="bottom")
    ax.set_xticks([0, 5, 10, 15])
    ax.set_xlabel("Policy time (fixed step size 0.05)")
    ax.set_ylabel("Worst-case finite-step gain")
    ax.spines["bottom"].set_bounds(0, 15)
    save(fig, out, "discrete_resources")


# ------------------------------------------------------------------ main
def main():
    global FS
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("results"))
    p.add_argument("--res", type=Path, default=Path("results"))
    p.add_argument("--venue", default="iclr")
    a = p.parse_args()
    use_venue(a.venue, grid="y")
    FS = plt.rcParams["font.size"]
    print(f"venue={a.venue}  col={W('col'):.3f}in  font={FS:g}pt")
    for f in (fig_closed_loop, fig_ordinary_pg, fig_instance_complexity,
              fig_query_energy, fig_full_support, fig_actual_adaptation,
              fig_discrete_resources):
        f(a.res, a.out)


if __name__ == "__main__":
    main()
