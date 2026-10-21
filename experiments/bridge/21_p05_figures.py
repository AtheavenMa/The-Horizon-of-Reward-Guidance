#!/usr/bin/env python3
"""The three Work Plan 05 figures. Reads completed outputs and does not rerun experiments."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
ROOT = Path(__file__).resolve().parents[2]


def load_curves(folder, methods, seeds):
    out = {}
    for method in methods:
        rows = []
        for seed in seeds:
            path = folder / f"{method}_seed{seed}.json"
            if path.exists():
                rows.append(json.loads(path.read_text())["returns"])
        out[method] = np.array(rows, dtype=float)
    return out


def main() -> None:
    weak = ROOT / "outputs/horizon_bridge/p3_dst_live_weak"
    curves = load_curves(weak, ("S", "U", "I", "M", "O"), range(8))
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    colors = {"S": "#546e7a", "U": "#00897b", "I": "#1565c0", "M": "#ef6c00", "O": "#c62828"}
    labels = {"S": "source robust", "U": "uniform robust", "I": "targeted robust", "M": "targeted MLE", "O": "true-reward oracle"}
    x = np.arange(curves["S"].shape[1])
    for method in ("S", "U", "I", "M", "O"):
        mean = curves[method].mean(0)
        sd = curves[method].std(0, ddof=1)
        ax.plot(x, mean, color=colors[method], label=labels[method], lw=1.6)
        ax.fill_between(x, mean - sd, mean + sd, color=colors[method], alpha=0.15)
    for t in (20, 40, 60, 80):
        ax.axvline(t, color="#bdbdbd", lw=0.8, ls="--")
    ax.set_xlabel("policy update")
    ax.set_ylabel("exact true return")
    ax.set_title("Live Deep Sea Treasure, weak source, 8 seeds")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(weak / "p3_curves.png", dpi=140)
    plt.close(fig)

    # Mechanism: phi path of one representative comparison and P2 return gaps.
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    for method, color in (("S", "#546e7a"), ("I", "#1565c0"), ("M", "#ef6c00"), ("O", "#c62828")):
        phis = np.array([json.loads((weak / f"{method}_seed{s}.json").read_text())["phis"] for s in range(8)])
        axes[0].plot(phis.mean(0), color=color, label=labels[method])
    axes[0].axhline(0.375, color="#c62828", lw=0.8, ls=":")
    axes[0].set_title("Deployed phi")
    axes[0].set_xlabel("policy update")
    axes[0].legend(frameon=False, fontsize=8)
    p2 = json.loads((ROOT / "outputs/horizon_bridge/p2_corrected_forks/summary.json").read_text())["summary"]
    names = ["source MLE", "uniform MLE", "Fisher MLE", "Fisher robust", "old fit", "oracle"]
    keys = ["source_mle", "uniform_mle", "fisher_mle", "fisher_robust", "fisher_old_fit", "oracle"]
    vals = [p2[f"early|{key}"]["return_increase_mean"] for key in keys]
    axes[1].bar(np.arange(len(names)), vals, color="#1565c0")
    axes[1].set_xticks(np.arange(len(names)), names, rotation=30, ha="right")
    axes[1].set_ylabel("exact return increase")
    axes[1].set_title("Early checkpoint, 40 exact updates")
    fig.tight_layout()
    fig.savefig(ROOT / "outputs/horizon_bridge/p2_corrected_forks/p2_mechanism.png", dpi=140)
    plt.close(fig)
    print("wrote figures")


if __name__ == "__main__":
    main()
