#!/usr/bin/env python3
"""R1: the same one-step lookahead for every preference set."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.bridge.common import write_summary
from experiments.bridge.pref_common import (
    BETA,
    DST,
    GAMMA_POS,
    ROOT,
    TRUE_W,
    audit_select,
    confidence_interval,
    fisher_at,
    fit_mle,
    jsonable,
    label_bits,
    live_batch,
    make_pairs,
    pair_terms,
    reinforce,
    scripted_treasure,
    select_phi,
)


def load_h4_source(dst: DST):
    spec = importlib.util.spec_from_file_location("h4", ROOT / "experiments/bridge/15_dst_preference_forks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    features = module.collect_features(dst.model, dst.low, dst.high, 480, 7100)
    _diff, c_all, d_all = module.pair_terms(features)
    rest = np.arange(40, len(d_all))
    source_idx = rest[np.argsort(np.abs(d_all[rest]))][:32]
    return c_all[source_idx], d_all[source_idx]


def checkpoint_logits(name: str) -> np.ndarray:
    if name == "h3_5":
        path = ROOT / "outputs/horizon_bridge/h3_dst_oracle/checkpoint_update_5.json"
        return np.array(json.loads(path.read_text())["logits"], dtype=np.float64)
    if name == "h3_20":
        path = ROOT / "outputs/horizon_bridge/h3_dst_oracle/checkpoint_update_20.json"
        return np.array(json.loads(path.read_text())["logits"], dtype=np.float64)
    update = 20 if name == "learned_20" else 60
    path = ROOT / f"outputs/horizon_bridge/p3_dst_live_weak/checkpoint_S_seed0_update{update}.json"
    return np.array(json.loads(path.read_text())["logits"], dtype=np.float64)


def phi_old_for(name: str) -> float:
    if name.startswith("h3"):
        # These checkpoints were trained with the true reward and have no saved guidance phi.
        return 0.5
    update = 20 if name == "learned_20" else 60
    phis = json.loads((ROOT / "outputs/horizon_bridge/p3_dst_live_weak/S_seed0.json").read_text())["phis"]
    return float(phis[update])


def pool_for(dst: DST, gym_env, logits: np.ndarray, salt: int):
    rng = np.random.default_rng(90_000 + salt)
    groups = [
        live_batch(gym_env, dst, logits, 64, rng)[0],
        live_batch(gym_env, dst, np.zeros((dst.n, 4)), 32, rng)[0],
        live_batch(gym_env, dst, scripted_treasure(dst), 32, rng)[0],
    ]
    episodes = [ep for group in groups for ep in group]
    feats = np.stack([(ep["ret"] - dst.low) / dst.scale for ep in episodes])
    left, right = make_pairs(len(feats), 256, rng)
    return pair_terms(feats, left, right, BETA)


def labels_for(c_s, d_s, c_pool, d_pool, seed: int):
    y_s = label_bits(c_s, d_s, np.random.default_rng(seed))
    u_source = confidence_interval(c_s, d_s, y_s, alpha=0.05) or (0.0, 1.0)
    # Pair identities are frozen before the label bits, at the neutral weight.
    uniform_idx = np.random.default_rng(8300).choice(len(d_pool), size=32, replace=False)
    fisher_idx = np.argsort(-fisher_at(c_pool, d_pool, 0.5))[:32]
    y_f = label_bits(c_pool[fisher_idx], d_pool[fisher_idx], np.random.default_rng(seed + 700))
    c = np.concatenate([c_s, c_pool[fisher_idx]])
    d = np.concatenate([d_s, d_pool[fisher_idx]])
    y = np.concatenate([y_s, y_f])
    u_updated = confidence_interval(c, d, y, alpha=0.05) or (0.0, 1.0)
    w_hat = fit_mle(c, d, y)
    return {
        "source": u_source,
        "updated": u_updated,
        "mle": (w_hat, w_hat),
        "true": (TRUE_W, TRUE_W),
        "w_hat": w_hat,
        "uniform_idx": uniform_idx,
    }


def rollout_exact(dst: DST, logits, phi, bounds, mode: str, target: float, steps: int, budget: float):
    series = []
    for _ in range(steps):
        window = (max(0.0, phi - budget), min(1.0, phi + budget))
        if mode == "lookahead":
            chosen, _worst, _true = select_phi(dst, logits, bounds, window, phi)
        elif mode == "track":
            chosen = float(np.clip(target, window[0], window[1]))
        else:
            chosen = float(target)
        nxt, mu, _grad = dst.exact_step(logits, chosen)
        mu2 = dst.mu(nxt)
        series.append({
            "phi": chosen,
            "true_return": dst.true_return(mu2),
            "true_gain": dst.true_return(mu2) - dst.true_return(mu),
        })
        logits, phi = nxt, chosen
    return series


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("exact", "sampled"), default="exact")
    parser.add_argument("--out", default="outputs/horizon_bridge/r1_controller_parity")
    args = parser.parse_args()
    started = time.perf_counter()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    dst = DST()
    import mo_gymnasium as mo_gym
    gym_env = mo_gym.make("deep-sea-treasure-v0")
    c_s, d_s = load_h4_source(dst)
    classes = {
        "h3_5": list(range(210, 218)),
        "h3_20": list(range(210, 215)),
        "learned_20": list(range(210, 215)),
        "learned_60": list(range(210, 215)),
    }
    if args.stage == "sampled":
        run_sampled(dst, gym_env, c_s, d_s, out, started)
        return
    rows = []
    futures = []
    audits = []
    for name, seeds in classes.items():
        logits0 = checkpoint_logits(name)
        phi0 = phi_old_for(name)
        c_pool, d_pool = pool_for(dst, gym_env, logits0, {"h3_5": 5, "h3_20": 20, "learned_20": 120, "learned_60": 160}[name])
        init_return = dst.true_return(dst.mu(logits0))
        for seed in seeds:
            pack = labels_for(c_s, d_s, c_pool, d_pool, seed)
            window = (max(0.0, phi0 - 0.02), min(1.0, phi0 + 0.02))
            chosen = {}
            gains = {}
            for key in ("source", "updated", "mle", "true"):
                audit = audit_select(dst, logits0, pack[key], window, phi0)
                audits.append({"checkpoint": name, "seed": seed, "set": key, **audit})
                chosen[key] = audit["phi"]
                gains[key] = audit["true_gain"]
            # Information-only: changing U does not change theta. Then one exact step does.
            frozen_return = dst.true_return(dst.mu(logits0))
            one = {}
            for key in ("source", "updated", "mle", "true"):
                nxt, mu, _g = dst.exact_step(logits0, chosen[key])
                one[key] = dst.true_return(dst.mu(nxt)) - dst.true_return(mu)
            rows.append({
                "checkpoint": name,
                "seed": seed,
                "phi_old": phi0,
                "source_width": pack["source"][1] - pack["source"][0],
                "updated_width": pack["updated"][1] - pack["updated"][0],
                "w_hat": pack["w_hat"],
                "phi_source": chosen["source"],
                "phi_updated": chosen["updated"],
                "phi_mle": chosen["mle"],
                "phi_true": chosen["true"],
                "gain_source": gains["source"],
                "gain_updated": gains["updated"],
                "gain_mle": gains["mle"],
                "gain_true": gains["true"],
                "oracle_violation": max(0.0, max(gains["source"], gains["updated"], gains["mle"]) - gains["true"]),
                "dense_violation": max(item["violation"] for item in audits if item["checkpoint"] == name and item["seed"] == seed),
                "info_only_return_gap": 0.0,
                "one_step_gap_updated_minus_source": one["updated"] - one["source"],
                "frozen_return": frozen_return,
            })
            if name in ("h3_5", "learned_60"):
                specs = {
                    "source": ("lookahead", pack["source"], pack["w_hat"]),
                    "updated": ("lookahead", pack["updated"], pack["w_hat"]),
                    "mle": ("lookahead", pack["mle"], pack["w_hat"]),
                    "true": ("lookahead", pack["true"], TRUE_W),
                    "mle_tracker": ("track", pack["mle"], pack["w_hat"]),
                    "fixed_true": ("fixed", pack["true"], TRUE_W),
                }
                for mode, (kind, bounds, target) in specs.items():
                    if mode in ("true", "fixed_true") and seed != seeds[0]:
                        continue
                    series = rollout_exact(dst, logits0.copy(), phi0, bounds, kind, target, 40, 0.02)
                    final = series[-1]["true_return"] - init_return
                    futures.append({
                        "checkpoint": name,
                        "seed": seed,
                        "mode": mode,
                        "return_gain": final,
                        "final_phi": series[-1]["phi"],
                        "final_return": series[-1]["true_return"],
                        "min_true_gain": min(item["true_gain"] for item in series),
                    })
    reference = {(item["checkpoint"]): item["return_gain"] for item in futures if item["mode"] == "true"}
    for item in futures:
        item["minus_true"] = item["return_gain"] - reference[item["checkpoint"]]
    gym_env.close()
    _write(out, rows, futures, audits, started)


def _write(out, rows, futures, audits, started) -> None:
    with (out / "one_step_seeds.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    if futures:
        with (out / "futures_seeds.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(futures[0].keys()))
            writer.writeheader()
            writer.writerows(futures)
    summary = {"stage_runtime_seconds": time.perf_counter() - started, "one_step": _mean(rows), "futures": _mean(futures, keys=("checkpoint", "mode")), "max_dense_violation": max(row["dense_violation"] for row in rows), "max_oracle_violation": max(row["oracle_violation"] for row in rows)}
    write_summary(out, jsonable(summary), started)


def _mean(rows, keys=("checkpoint",)):
    grouped = {}
    for row in rows:
        key = "|".join(str(row[name]) for name in keys)
        grouped.setdefault(key, []).append(row)
    out = {}
    numeric = [name for name, value in rows[0].items() if isinstance(value, (int, float)) and name not in ("seed",)]
    for key, group in grouped.items():
        out[key] = {"n": len(group)}
        for name in numeric:
            values = [row[name] for row in group if row.get(name) is not None]
            if values:
                out[key][name] = float(np.mean(values))
    return out


def run_sampled(dst, gym_env, c_s, d_s, out, started) -> None:
    logits0 = checkpoint_logits("h3_5")
    phi0 = 0.5
    c_pool, d_pool = pool_for(dst, gym_env, logits0, 5)
    rows = []
    for seed in (210, 211, 212):
        pack = labels_for(c_s, d_s, c_pool, d_pool, seed)
        for mode, bounds, target in (
            ("source", pack["source"], None),
            ("updated", pack["updated"], None),
            ("mle", pack["mle"], pack["w_hat"]),
            ("true", pack["true"], TRUE_W),
        ):
            logits = logits0.copy()
            phi = phi0
            rng = np.random.default_rng(110_000 + seed + ord(mode[0]))
            ret0 = dst.true_return(dst.mu(logits))
            for _k in range(40):
                window = (max(0.0, phi - 0.02), min(1.0, phi + 0.02))
                chosen, _worst, _true = select_phi(dst, logits, bounds, window, phi)
                rollouts, _steps = live_batch(gym_env, dst, logits, 256, rng)
                logits = reinforce(logits, rollouts, np.array([chosen, 1.0 - chosen]), dst.low, dst.high)
                phi = chosen
            rows.append({"seed": seed, "mode": mode, "return_gain": dst.true_return(dst.mu(logits)) - ret0, "final_phi": phi})
    existing = json.loads((out / "summary.json").read_text()) if (out / "summary.json").exists() else {}
    existing["sampled"] = rows
    gym_env.close()
    write_summary(out, jsonable(existing), started)


if __name__ == "__main__":
    main()
