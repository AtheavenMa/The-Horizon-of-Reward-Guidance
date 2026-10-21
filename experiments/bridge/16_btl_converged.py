#!/usr/bin/env python3
"""P1: converged Bradley-Terry fits and a fair query pool on Deep Sea Treasure."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import csv
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

from experiments.bridge.common import write_summary
from experiments.bridge.pref_common import (
    BETA,
    DST,
    ROOT,
    TRUE_W,
    confidence_interval,
    fisher_at,
    fit_mle,
    fit_newton,
    fit_steps,
    heldout_scores,
    jsonable,
    kkt_residual,
    label_bits,
    live_batch,
    make_pairs,
    nll,
    pair_terms,
    scripted_treasure,
    verify_against_h3,
)


def load_h4():
    spec = importlib.util.spec_from_file_location("h4", ROOT / "experiments/bridge/15_dst_preference_forks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def regenerate_old(dst: DST) -> dict:
    h4 = load_h4()
    features = h4.collect_features(dst.model, dst.low, dst.high, 480, 7100)
    diff = features[:, None, :] - features[None, :, :]
    # Rebuild pairs exactly as 15: adjacent pairing of the 480 trajectories.
    # 15 uses pair_terms on consecutive feature rows. Replicate that helper.
    c_all, d_all = None, None
    # Read pair_terms from the module.
    paired = h4.pair_terms(features)
    diff, c_all, d_all = paired
    dev = slice(0, 40)
    y_dev = h4.label(c_all[dev], d_all[dev], np.random.default_rng(7200))
    w_dev = h4.fit_mle(c_all[dev], d_all[dev], y_dev)
    rest = np.arange(40, len(d_all))
    source_idx = rest[np.argsort(np.abs(d_all[rest]))][:32]
    remaining = [i for i in rest if i not in set(source_idx.tolist())]
    info_order = sorted(
        remaining,
        key=lambda i: -float((d_all[i] ** 2) * h4.sigmoid(c_all[i] + w_dev * d_all[i]) * (1 - h4.sigmoid(c_all[i] + w_dev * d_all[i]))),
    )
    info_idx = np.array(info_order[:64])
    y_source = h4.label(c_all[source_idx], d_all[source_idx], np.random.default_rng(100))
    y_info = h4.label(c_all[info_idx[:32]], d_all[info_idx[:32]], np.random.default_rng(150))
    c = np.concatenate([c_all[source_idx], c_all[info_idx[:32]]])
    d = np.concatenate([d_all[source_idx], d_all[info_idx[:32]]])
    y = np.concatenate([y_source, y_info])
    mle = fit_mle(c, d, y)
    newton = fit_newton(c, d, y)
    old8 = fit_steps(c, d, y, 8)
    old80 = fit_steps(c, d, y, 80)
    h4_mle = h4.fit_mle(c, d, y)
    h4_80 = h4.fit_steps(c, d, y, 80)
    if abs(mle - 0.3363513883086601) > 1e-6 or abs(old80 - 0.766343105432656) > 1e-6:
        raise AssertionError((mle, old80, h4_mle, h4_80))
    if abs(mle - h4_mle) > 1e-6 or abs(newton - mle) > 1e-5:
        raise AssertionError((mle, newton, h4_mle))
    return {
        "w_dev": float(w_dev),
        "mle": mle,
        "newton": newton,
        "old8": old8,
        "old80": old80,
        "nll_mle": nll(mle, c, d, y),
        "nll_old80": nll(old80, c, d, y),
        "nll_old8": nll(old8, c, d, y),
        "kkt_mle": kkt_residual(mle, c, d, y),
        "kkt_newton": kkt_residual(newton, c, d, y),
        "kkt_old80": kkt_residual(old80, c, d, y),
        "score_mle": float(np.sum(d * (y - (1 / (1 + np.exp(-(c + mle * d))))))),
        "abs_error_mle": abs(mle - TRUE_W),
        "abs_error_old80": abs(old80 - TRUE_W),
        "source_median_fisher": float(np.median(fisher_at(c_all[source_idx], d_all[source_idx], w_dev))),
        "interval": confidence_interval(c, d, y),
        "extra_newton_after_mle": fit_newton(c, d, y, w0=mle),
    }


def collect_diverse(dst: DST, gym_env, seeds: dict, per_policy: int) -> np.ndarray:
    policies = {
        "uniform": np.zeros((dst.n, 4)),
        "treasure": scripted_treasure(dst),
    }
    for update in (5, 20):
        payload = json.loads((ROOT / f"outputs/horizon_bridge/h3_dst_oracle/checkpoint_update_{update}.json").read_text())
        policies[f"ckpt{update}"] = np.array(payload["logits"], dtype=np.float64)
    rng_pol = np.random.default_rng(seeds["random_policies"])
    for i in range(4):
        policies[f"random{i}"] = rng_pol.normal(size=(dst.n, 4))
    feats = []
    for name, logits in policies.items():
        episodes, _steps = live_batch(gym_env, dst, logits, per_policy, np.random.default_rng(seeds[name]))
        feats.extend([ep["ret"] for ep in episodes])
    raw = np.stack(feats)
    return (raw - dst.low) / dst.scale


def summarize_rows(rows: list[dict]) -> dict:
    def mean(key):
        return float(np.mean([row[key] for row in rows]))

    covered = sum(row["covers_true"] for row in rows)
    return {
        "seeds": len(rows),
        "old8_mean": mean("old8"),
        "old80_mean": mean("old80"),
        "mle_mean": mean("mle"),
        "newton_gap_mean": mean("newton_gap"),
        "post_gap_mean": mean("post_gap"),
        "kkt_mle_mean": mean("kkt_mle"),
        "kkt_old80_mean": mean("kkt_old80"),
        "nll_train_mle": mean("nll_mle"),
        "nll_train_old80": mean("nll_old80"),
        "heldout_nll": mean("heldout_nll"),
        "heldout_brier": mean("heldout_brier"),
        "heldout_info_nll": mean("heldout_info_nll"),
        "width_mean": mean("width"),
        "coverage": f"{covered}/{len(rows)}",
        "coverage_rate": covered / len(rows),
        "abs_error": mean("abs_error"),
        "source_fisher_median": mean("source_fisher"),
        "new_fisher_median": mean("new_fisher"),
        "source_width_mean": mean("source_width"),
    }


def one_condition(c_s, d_s, c_pool, d_pool, held, held_info, seeds, regime: str, rule: str) -> list[dict]:
    """rule is source, uniform, or fisher. Uniform indices ignore labels."""
    rows = []
    available = np.arange(len(d_pool))
    if regime == "mixed":
        source_local = np.random.default_rng(8400).choice(len(d_pool), size=32, replace=False)
        available = np.array([i for i in available if i not in set(source_local.tolist())])
    else:
        source_local = None
    uniform_idx = np.random.default_rng(8300 if regime == "weak" else 8401).choice(available, size=32, replace=False)
    for seed in seeds:
        rng = np.random.default_rng(10_000 + seed if regime == "mixed" else seed)
        if regime == "weak":
            y_s = label_bits(c_s, d_s, rng)
            c_src, d_src = c_s, d_s
        else:
            y_s = label_bits(c_pool[source_local], d_pool[source_local], rng)
            c_src, d_src = c_pool[source_local], d_pool[source_local]
        w_source = fit_mle(c_src, d_src, y_s)
        if rule == "source":
            c, d, y = c_src, d_src, y_s
            new_fisher = 0.0
        else:
            if rule == "fisher":
                rank = fisher_at(c_pool[available], d_pool[available], w_source)
                chosen = available[np.argsort(-rank)[:32]]
            else:
                chosen = uniform_idx
            y_new = label_bits(c_pool[chosen], d_pool[chosen], np.random.default_rng(seed + (1500 if regime == "mixed" else 500)))
            c = np.concatenate([c_src, c_pool[chosen]])
            d = np.concatenate([d_src, d_pool[chosen]])
            y = np.concatenate([y_s, y_new])
            new_fisher = float(np.median(fisher_at(c_pool[chosen], d_pool[chosen], w_source)))
        mle = fit_mle(c, d, y)
        newton = fit_newton(c, d, y, w0=0.5)
        after = fit_newton(c, d, y, w0=mle)
        old8 = fit_steps(c, d, y, 8)
        old80 = fit_steps(c, d, y, 80)
        interval = confidence_interval(c, d, y, alpha=0.05)
        if interval is None:
            covers, width = False, float("nan")
        else:
            width = float(interval[1] - interval[0])
            covers = interval[0] - 1e-9 <= TRUE_W <= interval[1] + 1e-9
        scores = heldout_scores(*held, mle)
        info_scores = heldout_scores(*held_info, mle)
        rows.append({
            "regime": regime,
            "rule": rule,
            "seed": int(seed),
            "old8": old8,
            "old80": old80,
            "mle": mle,
            "newton_gap": abs(newton - mle),
            "post_gap": abs(after - mle),
            "kkt_mle": kkt_residual(mle, c, d, y),
            "kkt_old80": kkt_residual(old80, c, d, y),
            "nll_mle": nll(mle, c, d, y),
            "nll_old80": nll(old80, c, d, y),
            "heldout_nll": scores["nll"],
            "heldout_brier": scores["brier"],
            "heldout_info_nll": info_scores["nll"],
            "width": width,
            "covers_true": bool(covers),
            "abs_error": abs(mle - TRUE_W),
            "source_fisher": float(np.median(fisher_at(c_src, d_src, w_source))),
            "new_fisher": new_fisher,
            "source_width": (confidence_interval(c_src, d_src, y_s) or (0.0, 1.0))[1]
            - (confidence_interval(c_src, d_src, y_s) or (0.0, 1.0))[0],
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/horizon_bridge/p1_btl_repair")
    args = parser.parse_args()
    started = time.perf_counter()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    dst = DST()
    check = verify_against_h3(dst)
    old = regenerate_old(dst)
    import mo_gymnasium as mo_gym
    gym_env = mo_gym.make("deep-sea-treasure-v0")
    pool_seeds = {"random_policies": 8100, "uniform": 8110, "treasure": 8111, "ckpt5": 8112, "ckpt20": 8113}
    for i in range(4):
        pool_seeds[f"random{i}"] = 8120 + i
    held_seeds = {"random_policies": 8200, "uniform": 8210, "treasure": 8211, "ckpt5": 8212, "ckpt20": 8213}
    for i in range(4):
        held_seeds[f"random{i}"] = 8220 + i
    pool = collect_diverse(dst, gym_env, pool_seeds, 32)
    held_feat = collect_diverse(dst, gym_env, held_seeds, 64)
    rng_pool = np.random.default_rng(8150)
    left, right = make_pairs(len(pool), 1024, rng_pool)
    c_pool, d_pool = pair_terms(pool, left, right, BETA)
    rng_held = np.random.default_rng(8250)
    h_left, h_right = make_pairs(len(held_feat), 512, rng_held)
    c_h, d_h = pair_terms(held_feat, h_left, h_right, BETA)
    y_h = label_bits(c_h, d_h, np.random.default_rng(8260))
    info_rank = np.argsort(-fisher_at(c_h, d_h, 0.5))[:128]
    held = (c_h, d_h, y_h)
    held_info = (c_h[info_rank], d_h[info_rank], y_h[info_rank])

    # Old weak source pairs, regenerated with the original construction.
    h4 = load_h4()
    features = h4.collect_features(dst.model, dst.low, dst.high, 480, 7100)
    _diff, c_all, d_all = h4.pair_terms(features)
    rest = np.arange(40, len(d_all))
    source_idx = rest[np.argsort(np.abs(d_all[rest]))][:32]
    c_s, d_s = c_all[source_idx], d_all[source_idx]

    seeds = list(range(100, 110))
    specs = [
        ("weak", "source"),
        ("weak", "uniform"),
        ("weak", "fisher"),
        ("mixed", "source"),
        ("mixed", "uniform"),
        ("mixed", "fisher"),
    ]
    conditions = {}
    flat = []
    for regime, rule in specs:
        rows = one_condition(c_s, d_s, c_pool, d_pool, held, held_info, seeds, regime, rule)
        conditions[f"{regime}_{rule}"] = {"summary": summarize_rows(rows), "rows": rows}
        flat.extend(rows)
    gym_env.close()
    with (out / "seeds.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0].keys()))
        writer.writeheader()
        writer.writerows(flat)
    plot_p1(out, old, conditions)
    payload = {
        "dp_check": check,
        "old_extra32": old,
        "heldout_pairs": 512,
        "heldout_informative_pairs": 128,
        "candidate_pairs": int(len(d_pool)),
        "conditions": {key: value["summary"] for key, value in conditions.items()},
        "causal": causal_text(old, conditions),
    }
    write_summary(out, jsonable(payload), started)


def causal_text(old: dict, conditions: dict) -> str:
    weak_u = conditions["weak_uniform"]["summary"]
    weak_f = conditions["weak_fisher"]["summary"]
    mixed_s = conditions["mixed_source"]["summary"]
    return (
        f"On the original extra32 batch the 80-step mean gradient stays at {old['old80']:.6f} "
        f"with train NLL {old['nll_old80']:.4f}, while both bounded minimization and damped Newton "
        f"reach {old['mle']:.6f} with train NLL {old['nll_mle']:.4f}. "
        f"A further Newton step from that MLE moves it by {abs(old['extra_newton_after_mle'] - old['mle']):.3e}. "
        f"Weak-source median Fisher is {conditions['weak_source']['summary']['source_fisher_median']:.4g}; "
        f"fair uniform new labels have median Fisher {weak_u['new_fisher_median']:.4g} and "
        f"Fisher selection {weak_f['new_fisher_median']:.4g}. "
        f"Mixed source alone has mean absolute error {mixed_s['abs_error']:.4f} and mean interval width {mixed_s['width_mean']:.4f}."
    )


def plot_p1(out: Path, old: dict, conditions: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    axes[0].bar(["8 steps", "80 steps", "MLE", "Newton"], [old["old8"], old["old80"], old["mle"], old["newton"]], color=["#bdbdbd", "#9e9e9e", "#1565c0", "#2e7d32"])
    axes[0].axhline(TRUE_W, color="#c62828", lw=1, label="teacher 0.375")
    axes[0].set_ylabel("fitted w")
    axes[0].set_title("Original extra32 batch")
    axes[0].legend(frameon=False, fontsize=8)
    labels = ["weak src", "weak unif", "weak Fisher", "mixed src", "mixed unif", "mixed Fisher"]
    keys = ["weak_source", "weak_uniform", "weak_fisher", "mixed_source", "mixed_uniform", "mixed_fisher"]
    nlls = [conditions[key]["summary"]["heldout_nll"] for key in keys]
    widths = [conditions[key]["summary"]["width_mean"] for key in keys]
    x = np.arange(len(labels))
    axes[1].bar(x - 0.18, nlls, width=0.36, label="held-out NLL", color="#1565c0")
    ax2 = axes[1].twinx()
    ax2.bar(x + 0.18, widths, width=0.36, label="interval width", color="#ef6c00")
    axes[1].set_xticks(x, labels, rotation=30, ha="right")
    axes[1].set_title("Same pool, 10 label seeds")
    axes[1].set_ylabel("held-out NLL")
    ax2.set_ylabel("interval width")
    fig.tight_layout()
    fig.savefig(out / "p1_fit.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
