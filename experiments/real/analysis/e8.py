"""Matched-budget summary across the runs that actually finished."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.real.common.records import ROOT, run_dir, write_run


def run_e8() -> dict:
    e1 = _e1_summary()
    e4 = _json(ROOT / "experiments/horizon_real/E4/summary.json")
    e6 = _json(ROOT / "experiments/horizon_real/E6/summary.json")
    e7 = _json(ROOT / "experiments/horizon_real/E7/summary.json")
    rows = []
    if e4:
        rows.append(_budget_row("Walker Walk", e4, "normal", "diagnosis"))
        rows.append(_budget_row("Cheetah Run", [r for r in e4 if r.get("domain") == "cheetah"], "normal", "diagnosis"))
    if e6:
        base = [r["verified_utility"] for r in e6 if r.get("method") == "Fixed-feature reward baseline"]
        comp = [r["verified_utility"] for r in e6 if r.get("method") == "Policy-relevant selection"]
        rows.append({
            "task": "verifiable instructions / Qwen2.5-1.5B",
            "baseline": "Fixed-feature reward baseline",
            "comparison": "Policy-relevant selection",
            "labels_matched": True,
            "compute_matched": True,
            "difference": _mean_diff(comp, base),
            "replicated": len(base) >= 2 and len(comp) >= 2,
        })
    if e7 and isinstance(e7, list) and e7 and "overlap" in e7[0]:
        base = [r["overlap"] for r in e7 if r.get("method") == "Fixed learned reward"]
        comp = [r["overlap"] for r in e7 if r.get("method") == "Fixed-period new feedback"]
        rows.append({
            "task": "TL;DR / Pythia-1B",
            "baseline": "Fixed learned reward",
            "comparison": "Fixed-period new feedback",
            "labels_matched": True,
            "compute_matched": True,
            "difference": _mean_diff(comp, base),
            "replicated": len(base) >= 2 and len(comp) >= 2,
        })
    dest = run_dir("E8")
    write_run(dest, {"sources": ["E1", "E4", "E6", "E7"]}, rows, [], {"status": "complete", "e1": e1})
    (dest / "summary.json").write_text(json.dumps({"e1": e1, "rows": rows}, indent=2) + "\n")
    _figures(dest, e1, e4, e6)
    print(json.dumps(rows, indent=2), flush=True)
    return {"rows": rows, "e1": e1}


def _e1_summary() -> dict:
    out = {}
    for name in ("Deep-Sea-Treasure", "Resource-Gathering"):
        path = ROOT / "experiments/horizon_real/E1" / name / "summary.json"
        if path.exists():
            out[name] = json.loads(path.read_text())
    return out


def _json(path: Path):
    if not path.exists():
        return []
    return json.loads(path.read_text())


def _budget_row(task: str, runs: list[dict], base: str, comp: str) -> dict:
    b = [r["final_true_return"] for r in runs if r.get("mode") == base and task.split()[0].lower() in r.get("domain", task.lower())]
    c = [r["final_true_return"] for r in runs if r.get("mode") == comp and task.split()[0].lower() in r.get("domain", task.lower())]
    if task.startswith("Walker"):
        b = [r["final_true_return"] for r in runs if r.get("domain") == "walker" and r.get("mode") == base]
        c = [r["final_true_return"] for r in runs if r.get("domain") == "walker" and r.get("mode") == comp]
    return {
        "task": task,
        "baseline": base,
        "comparison": comp,
        "labels_matched": True,
        "compute_matched": True,
        "difference": _mean_diff(c, b),
        "replicated": len(b) >= 2 and len(c) >= 2,
        "baseline_mean": float(np.mean(b)) if b else None,
        "comparison_mean": float(np.mean(c)) if c else None,
    }


def _mean_diff(comp, base) -> str:
    if not comp or not base:
        return "unavailable"
    diff = float(np.mean(comp) - np.mean(base))
    spread = float(np.std(comp + base, ddof=1)) if len(comp) + len(base) > 2 else float("nan")
    return f"{diff:.4g} (sd {spread:.4g})"


def _figures(dest: Path, e1: dict, e4: list, e6: list) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    for name, color in (("Deep-Sea-Treasure", "C0"), ("Resource-Gathering", "C1")):
        path = ROOT / "experiments/horizon_real/E1" / name / "seed1" / "metrics.csv"
        if not path.exists():
            continue
        rounds, cert, vq = [], [], []
        with path.open() as handle:
            for row in csv.DictReader(handle):
                rounds.append(int(float(row["round"])))
                cert.append(float(row["original_certificate"]))
                vq.append(float(row["candidate_restricted_value"]))
        ax.plot(rounds, cert, marker="o", color=color, label=f"{name} certificate")
        ax.plot(rounds, vq, marker="s", linestyle="--", color=color, label=f"{name} candidate value")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xlabel("checkpoint")
    ax.set_ylabel("finite-step gain")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(dest / "certificate_vs_candidate.pdf")
    plt.close(fig)

    if e4:
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        modes = sorted({r["mode"] for r in e4 if r.get("domain") == "walker"})
        means, labels = [], []
        for mode in modes:
            vals = [r["final_true_return"] for r in e4 if r.get("domain") == "walker" and r["mode"] == mode]
            if vals:
                means.append(float(np.mean(vals)))
                labels.append(mode)
        ax.bar(labels, means)
        ax.set_ylabel("held-out true return")
        ax.set_title("")
        fig.tight_layout()
        fig.savefig(dest / "walker_methods.pdf")
        plt.close(fig)

    if e6 and "verified_utility" in e6[0]:
        fig, ax = plt.subplots(figsize=(6.2, 3.2))
        methods = []
        for row in e6:
            if row["method"] not in methods:
                methods.append(row["method"])
        means = []
        for method in methods:
            vals = [r["verified_utility"] for r in e6 if r["method"] == method]
            means.append(float(np.mean(vals)) if vals else 0.0)
        ax.barh(methods, means)
        ax.set_xlabel("held-out verified utility")
        fig.tight_layout()
        fig.savefig(dest / "instruction_utility.pdf")
        plt.close(fig)
