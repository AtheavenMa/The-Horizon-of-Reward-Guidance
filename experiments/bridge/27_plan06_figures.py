#!/usr/bin/env python3
"""R5: at most three figures from the recorded Plan 06 outputs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.bridge.pref_common import DST, TRUE_W, select_phi

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/horizon_bridge/r5_paper_results"


def r1_curves():
    """One matched label seed, exact population steps, shared checkpoint."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("r1", ROOT / "experiments/bridge/22_guidance_controller_parity.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    dst = DST()
    logits = module.checkpoint_logits("h3_5")
    phi = 0.5
    c_s, d_s = module.load_h4_source(dst)
    import mo_gymnasium as mo_gym
    gym_env = mo_gym.make("deep-sea-treasure-v0")
    c_pool, d_pool = module.pool_for(dst, gym_env, logits, 5)
    pack = module.labels_for(c_s, d_s, c_pool, d_pool, 210)
    gym_env.close()
    curves = {}
    specs = {
        "U_source": ("lookahead", pack["source"], None),
        "U_updated": ("lookahead", pack["updated"], None),
        "MLE": ("lookahead", pack["mle"], pack["w_hat"]),
        "A_true": ("lookahead", pack["true"], TRUE_W),
        "fixed true-w": ("fixed", pack["true"], TRUE_W),
    }
    for name, (mode, bounds, target) in specs.items():
        series = module.rollout_exact(dst, logits.copy(), phi, bounds, mode, TRUE_W if target is None else target, 40, 0.02)
        curves[name] = [dst.true_return(dst.mu(logits))] + [row["true_return"] for row in series]
    return curves


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    curves = r1_curves()
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for name, series in curves.items():
        ax.plot(series, label=name, linewidth=1.8)
    ax.set_xlabel("Exact policy updates")
    ax.set_ylabel("True return")
    ax.set_title("Same one-step controller, H3 update 5")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "r1_same_controller.png", dpi=140)
    plt.close(fig)

    replay = json.loads((ROOT / "outputs/horizon_bridge/r2_dst_paths/replay.json").read_text())
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    marks = (0, 10, 20, 40, 60, 80, 120, 160)
    for method, label in (("S", "source robust"), ("I", "targeted robust"), ("M", "MLE tracker"), ("O", "fixed true-w")):
        group = [row for row in replay if row["method"] == method]
        mean = [float(np.mean([row["returns_at"][str(t)] for row in group])) for t in marks]
        ax.plot(marks, mean, label=label, linewidth=1.8)
    for t in (20, 40, 60, 80):
        ax.axvline(t, color="0.85", linewidth=0.8)
    ax.annotate("M ahead", xy=(40, 0.627), xytext=(48, 0.60), fontsize=8, arrowprops={"arrowstyle": "->"})
    ax.annotate("S ahead", xy=(160, 0.683), xytext=(112, 0.70), fontsize=8, arrowprops={"arrowstyle": "->"})
    ax.set_xlabel("Live policy updates")
    ax.set_ylabel("True return")
    ax.set_title("Replayed 160-update DST paths")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "r2_replay_curves.png", dpi=140)
    plt.close(fig)

    ident = json.loads((ROOT / "outputs/horizon_bridge/r4_rg_identification/summary.json").read_text())["designs"]
    live_path = ROOT / "outputs/horizon_bridge/r4_rg_robust/live_summary.json"
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
    names = ["weak_source", "weak_uniform", "weak_trace", "weak_mineig"]
    labels = ["weak", "uniform", "trace", "min-eig"]
    axes[0].bar(np.arange(len(names)) - 0.15, [ident[n]["eig_min"] for n in names], width=0.3, label="min")
    axes[0].bar(np.arange(len(names)) + 0.15, [ident[n]["eig_max"] for n in names], width=0.3, label="max")
    axes[0].set_xticks(range(len(names)), labels, fontsize=8)
    axes[0].set_ylabel("Fisher eigenvalue")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_title("Two preference directions")
    if live_path.exists():
        live = json.loads(live_path.read_text())["methods"]
        order = ["source", "uniform", "trace", "mineig", "simplex", "oracle", "lookahead"]
        present = [name for name in order if name in live]
        axes[1].bar(range(len(present)), [live[name]["final_mean"] for name in present], color="0.35")
        axes[1].set_xticks(range(len(present)), present, rotation=40, ha="right", fontsize=8)
        axes[1].set_ylabel("Final true return")
        axes[1].set_title("Eight-run RG comparison")
    else:
        axes[1].set_title("RG comparison not recorded")
    fig.tight_layout()
    fig.savefig(OUT / "r4_identification.png", dpi=140)
    plt.close(fig)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
