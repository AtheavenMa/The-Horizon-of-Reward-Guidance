"""Figures for Work Plan 07. Numbers come from seed-level JSON already on disk."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_minigrid(root: Path, dest: Path) -> None:
    rows = []
    for path in sorted((root / "minigrid").glob("*_seed*.json")):
        if path.name.startswith("oracle40k"):
            continue
        rows.append(json.loads(path.read_text()))
    names = ["source_only", "new_32", "extra_fit_same_source", "oracle"]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    width = 0.2
    for offset, name in enumerate(names):
        values = [row["final"]["true_utility"] for row in rows if row["name"] == name]
        ax.bar(np.arange(len(values)) + offset * width, values, width=width, label=name)
    ax.set_ylabel("True utility")
    ax.set_xlabel("Policy seed")
    ax.set_xticks(np.arange(3) + 1.5 * width, ["0", "1", "2"])
    ax.legend(fontsize=7)
    ax.set_title("Empty-8x8 matched forks")
    fig.tight_layout()
    dest.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest / "minigrid_forks.png", dpi=140)
    plt.close(fig)


def plot_human(root: Path, dest: Path) -> None:
    names = ["human_256", "human_1024", "human_4096", "compute_256", "uniform_1024"]
    vals = []
    for name in names:
        row = json.loads((root / "human_tldr" / f"{name}.json").read_text())
        vals.append(row["accuracy"])
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.bar(range(len(names)), vals, color="0.35")
    ax.axhline(0.5, color="0.6", lw=0.8)
    ax.set_xticks(range(len(names)), names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Human validation accuracy")
    ax.set_ylim(0.4, 0.7)
    fig.tight_layout()
    dest.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest / "human_ladder.png", dpi=140)
    plt.close(fig)


def plot_pythia(root: Path, dest: Path) -> None:
    main = json.loads((root / "pythia_1b" / "main.json").read_text())
    labels = [f"{row['method'].split('_')[0]} s{row['seed']}" for row in main["rows"]]
    values = [row["anchor_rms"] for row in main["rows"]]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.bar(range(len(labels)), values, color="0.35")
    ax.axhline(0.05, color="0.5", ls="--", lw=0.8)
    ax.set_xticks(range(len(labels)), labels, fontsize=8)
    ax.set_ylabel("Fixed-anchor RMS")
    fig.tight_layout()
    dest.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest / "pythia_anchor_rms.png", dpi=140)
    plt.close(fig)
