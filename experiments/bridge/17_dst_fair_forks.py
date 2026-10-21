#!/usr/bin/env python3
"""P2: matched-checkpoint comparison of converged MLE and robust guidance."""
from __future__ import annotations

import argparse
import csv
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
    best_phi,
    confidence_interval,
    fisher_at,
    fit_mle,
    fit_newton,
    fit_steps,
    jsonable,
    label_bits,
    live_batch,
    make_pairs,
    pair_terms,
    reinforce,
    scripted_treasure,
    worst_gain,
)


def load_logits(update: int) -> np.ndarray:
    payload = json.loads((ROOT / f"outputs/horizon_bridge/h3_dst_oracle/checkpoint_update_{update}.json").read_text())
    return np.array(payload["logits"], dtype=np.float64)


_SOURCE_TERMS = None


def source_batch(dst: DST, seed: int):
    """The original low-sensitivity source pairs, with an independent label seed."""
    global _SOURCE_TERMS
    if _SOURCE_TERMS is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location("h4", ROOT / "experiments/bridge/15_dst_preference_forks.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        features = module.collect_features(dst.model, dst.low, dst.high, 480, 7100)
        _diff, c_all, d_all = module.pair_terms(features)
        rest = np.arange(40, len(d_all))
        source_idx = rest[np.argsort(np.abs(d_all[rest]))][:32]
        _SOURCE_TERMS = (c_all[source_idx], d_all[source_idx])
    c, d = _SOURCE_TERMS
    y = label_bits(c, d, np.random.default_rng(seed))
    interval = confidence_interval(c, d, y, alpha=0.05) or (0.0, 1.0)
    return c, d, y, interval


def candidate_pool(dst: DST, gym_env, logits: np.ndarray, seed: int):
    rng = np.random.default_rng(50_000 + seed)
    parts = [
        live_batch(gym_env, dst, logits, 64, rng)[0],
        live_batch(gym_env, dst, np.zeros((dst.n, 4)), 32, rng)[0],
        live_batch(gym_env, dst, scripted_treasure(dst), 32, rng)[0],
    ]
    episodes = [ep for group in parts for ep in group]
    feats = np.stack([(ep["ret"] - dst.low) / dst.scale for ep in episodes])
    left, right = make_pairs(len(feats), 256, rng)
    return pair_terms(feats, left, right, BETA)


def run_exact(dst: DST, logits, phi, mode, bounds_source, bounds_used, target, steps, budget):
    series = []
    for _k in range(steps):
        if mode == "oracle":
            chosen = TRUE_W
        elif mode == "robust":
            if bounds_used is None:
                chosen = phi
            else:
                window = (max(0.0, phi - budget), min(1.0, phi + budget))
                chosen, _value = best_phi(dst, logits, bounds_used, window)
        else:
            chosen = float(np.clip(target, phi - budget, phi + budget))
        nxt, mu, _grad = dst.exact_step(logits, chosen)
        mu2 = dst.mu(nxt)
        series.append({
            "phi": chosen,
            "true_return": dst.true_return(mu2),
            "true_gain": dst.true_return(mu2) - dst.true_return(mu),
            "source_certificate": worst_gain(dst, mu, mu2, bounds_source),
            "updated_certificate": None if bounds_used is None else worst_gain(dst, mu, mu2, bounds_used),
        })
        logits, phi = nxt, chosen
    return series


def mean_or_none(values):
    clean = [v for v in values if v is not None]
    return None if not clean else float(np.mean(clean))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/horizon_bridge/p2_corrected_forks")
    parser.add_argument("--stage", choices=("core", "learned", "sampled"), default="core")
    args = parser.parse_args()
    started = time.perf_counter()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    dst = DST()
    import mo_gymnasium as mo_gym
    gym_env = mo_gym.make("deep-sea-treasure-v0")
    if args.stage == "sampled":
        run_sampled(dst, gym_env, out, started)
        return
    if args.stage == "learned":
        run_learned(dst, gym_env, out, started)
        return

    checkpoints = {"early": load_logits(5), "later": load_logits(20)}
    plans = {
        "early": list(range(110, 118)),
        "later": list(range(110, 115)),
    }
    rows = []
    curves = {}
    pools = {}
    for name, logits0 in checkpoints.items():
        pools[name] = candidate_pool(dst, gym_env, logits0, 5 if name == "early" else 20)
        init_return = dst.true_return(dst.mu(logits0))
        for seed in plans[name]:
            c_s, d_s, y_s, u_source = source_batch(dst, seed)
            w_source = fit_mle(c_s, d_s, y_s)
            c_pool, d_pool = pools[name]
            uniform_idx = np.random.default_rng(8300).choice(len(d_pool), size=32, replace=False)
            rank = fisher_at(c_pool, d_pool, w_source)
            fisher_idx = np.argsort(-rank)[:32]
            y_u = label_bits(c_pool[uniform_idx], d_pool[uniform_idx], np.random.default_rng(seed + 500))
            y_f = label_bits(c_pool[fisher_idx], d_pool[fisher_idx], np.random.default_rng(seed + 700))
            packs = {
                "source": (c_s, d_s, y_s, u_source),
                "uniform": (
                    np.concatenate([c_s, c_pool[uniform_idx]]),
                    np.concatenate([d_s, d_pool[uniform_idx]]),
                    np.concatenate([y_s, y_u]),
                    confidence_interval(np.concatenate([c_s, c_pool[uniform_idx]]), np.concatenate([d_s, d_pool[uniform_idx]]), np.concatenate([y_s, y_u])) or (0.0, 1.0),
                ),
                "fisher": (
                    np.concatenate([c_s, c_pool[fisher_idx]]),
                    np.concatenate([d_s, d_pool[fisher_idx]]),
                    np.concatenate([y_s, y_f]),
                    confidence_interval(np.concatenate([c_s, c_pool[fisher_idx]]), np.concatenate([d_s, d_pool[fisher_idx]]), np.concatenate([y_s, y_f])) or (0.0, 1.0),
                ),
            }
            phi_init, init_gain = best_phi(dst, logits0, u_source, (0.0, 1.0))
            specs = [
                ("source_mle", "source", "point", 0.02, False),
                ("source_mle_extra", "source", "point", 0.02, True),
                ("uniform_mle", "uniform", "point", 0.02, False),
                ("fisher_mle", "fisher", "point", 0.02, False),
                ("fisher_robust", "fisher", "robust", 0.02, False),
                ("fisher_old_fit", "fisher", "point", 0.02, False),
            ]
            if name == "early":
                specs.extend([
                    ("fisher_mle_b05", "fisher", "point", 0.05, False),
                    ("fisher_robust_b05", "fisher", "robust", 0.05, False),
                ])
            for mode, key, kind, budget, extra in specs:
                c, d, y, bounds = packs[key]
                if mode == "fisher_old_fit":
                    target = fit_steps(c, d, y, 8)
                else:
                    target = fit_mle(c, d, y)
                    if extra:
                        target = fit_newton(c, d, y, w0=target)
                # Point deployment follows the fitted target. The updated certificate
                # is still evaluated on the batch interval, which the point rule does not read.
                series = run_exact(dst, logits0.copy(), phi_init, kind, u_source, bounds, target, 40, budget)
                final = series[-1]
                row = {
                    "checkpoint": name,
                    "seed": seed,
                    "mode": mode,
                    "budget": budget,
                    "init_return": init_return,
                    "phi_init": phi_init,
                    "init_source_gain": init_gain,
                    "uncertified_at_init": bool(init_gain < GAMMA_POS),
                    "return_increase": final["true_return"] - init_return,
                    "final_return": final["true_return"],
                    "final_phi": final["phi"],
                    "final_source": final["source_certificate"],
                    "final_updated": final["updated_certificate"],
                    "target": target,
                }
                rows.append(row)
                curves[f"{name}_{seed}_{mode}"] = [item["true_return"] for item in series]
        # One oracle, shared by every seed because it ignores labels.
        phi_init, init_gain = best_phi(dst, logits0, (0.0, 1.0), (0.0, 1.0))
        series = run_exact(dst, logits0.copy(), phi_init, "oracle", (0.0, 1.0), (TRUE_W, TRUE_W), TRUE_W, 40, 1.0)
        rows.append({
            "checkpoint": name,
            "seed": 0,
            "mode": "oracle",
            "budget": None,
            "init_return": init_return,
            "phi_init": TRUE_W,
            "init_source_gain": init_gain,
            "uncertified_at_init": False,
            "return_increase": series[-1]["true_return"] - init_return,
            "final_return": series[-1]["true_return"],
            "final_phi": TRUE_W,
            "final_source": None,
            "final_updated": None,
            "target": TRUE_W,
        })
    gym_env.close()
    _write_rows(out, rows, curves, started, stage="core")


def _write_rows(out: Path, rows, curves, started, stage: str) -> None:
    path = out / f"{stage}_seeds.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    grouped = {}
    for row in rows:
        grouped.setdefault((row["checkpoint"], row["mode"]), []).append(row)
    summary = {}
    for key, group in grouped.items():
        ref = [r for r in rows if r["checkpoint"] == key[0] and r["mode"] == "fisher_mle"]
        ref_mean = float(np.mean([r["return_increase"] for r in ref])) if ref else None
        summary["|".join(map(str, key))] = {
            "n": len(group),
            "return_increase_mean": float(np.mean([r["return_increase"] for r in group])),
            "return_increase_sd": float(np.std([r["return_increase"] for r in group], ddof=1)) if len(group) > 1 else 0.0,
            "minus_fisher_mle": None if ref_mean is None else float(np.mean([r["return_increase"] for r in group]) - ref_mean),
            "final_phi_mean": float(np.mean([r["final_phi"] for r in group])),
            "final_source_mean": mean_or_none([r["final_source"] for r in group]),
            "final_updated_mean": mean_or_none([r["final_updated"] for r in group]),
            "uncertified_at_init": bool(group[0]["uncertified_at_init"]),
            "init_source_gain": float(group[0]["init_source_gain"]),
        }
    payload = {"stage": stage, "summary": summary}
    if stage == "core":
        existing = {}
    else:
        existing_path = out / "summary.json"
        existing = json.loads(existing_path.read_text()) if existing_path.exists() else {}
        existing.setdefault("stages", {})[stage] = summary
        payload = existing
        payload["stage_added"] = stage
    write_summary(out, jsonable(payload if stage == "core" else payload), started)


def run_sampled(dst, gym_env, out, started) -> None:
    summary = json.loads((out / "summary.json").read_text())
    logits0 = load_logits(5)
    rows = []
    for seed in (110, 111, 112):
        c_s, d_s, y_s, u_source = source_batch(dst, seed)
        w_source = fit_mle(c_s, d_s, y_s)
        c_pool, d_pool = candidate_pool(dst, gym_env, logits0, 5)
        rank = fisher_at(c_pool, d_pool, w_source)
        chosen = np.argsort(-rank)[:32]
        y_f = label_bits(c_pool[chosen], d_pool[chosen], np.random.default_rng(seed + 700))
        c = np.concatenate([c_s, c_pool[chosen]])
        d = np.concatenate([d_s, d_pool[chosen]])
        y = np.concatenate([y_s, y_f])
        target = fit_mle(c, d, y)
        bounds = confidence_interval(c, d, y) or (0.0, 1.0)
        phi_init, _ = best_phi(dst, logits0, u_source, (0.0, 1.0))
        for mode in ("fisher_mle", "fisher_robust"):
            logits = logits0.copy()
            phi = phi_init
            rng = np.random.default_rng(70_000 + seed + (0 if mode == "fisher_mle" else 17))
            ret0 = dst.true_return(dst.mu(logits))
            for _k in range(40):
                if mode == "fisher_robust":
                    chosen_phi, _ = best_phi(dst, logits, bounds, (max(0.0, phi - 0.02), min(1.0, phi + 0.02)))
                else:
                    chosen_phi = float(np.clip(target, phi - 0.02, phi + 0.02))
                rollouts, _steps = live_batch(gym_env, dst, logits, 256, rng)
                logits = reinforce(logits, rollouts, np.array([chosen_phi, 1.0 - chosen_phi]), dst.low, dst.high)
                phi = chosen_phi
            rows.append({
                "seed": seed,
                "mode": mode,
                "return_increase": dst.true_return(dst.mu(logits)) - ret0,
                "final_phi": phi,
            })
    summary["sampled"] = rows
    gym_env.close()
    write_summary(out, jsonable(summary), started)


def run_learned(dst, gym_env, out, started) -> None:
    rows = []
    base = ROOT / "outputs/horizon_bridge/p3_dst_live_weak"
    for update in (20, 60):
        payload = json.loads((base / f"checkpoint_S_seed0_update{update}.json").read_text())
        logits0 = np.array(payload["logits"], dtype=np.float64)
        init_return = dst.true_return(dst.mu(logits0))
        c_pool, d_pool = candidate_pool(dst, gym_env, logits0, 600 + update)
        for seed in range(310, 315):
            c_s, d_s, y_s, u_source = source_batch(dst, seed)
            w_source = fit_mle(c_s, d_s, y_s)
            rank = fisher_at(c_pool, d_pool, w_source)
            chosen = np.argsort(-rank)[:32]
            y_f = label_bits(c_pool[chosen], d_pool[chosen], np.random.default_rng(seed + 700))
            c = np.concatenate([c_s, c_pool[chosen]])
            d = np.concatenate([d_s, d_pool[chosen]])
            y = np.concatenate([y_s, y_f])
            target = fit_mle(c, d, y)
            bounds = confidence_interval(c, d, y) or (0.0, 1.0)
            phi_init, init_gain = best_phi(dst, logits0, u_source, (0.0, 1.0))
            for mode, kind, used, aim in (
                ("fisher_mle", "point", u_source, target),
                ("fisher_robust", "robust", bounds, target),
            ):
                series = run_exact(dst, logits0.copy(), phi_init, kind, u_source, used, aim, 40, 0.02)
                rows.append({
                    "checkpoint": f"learned_{update}",
                    "seed": seed,
                    "mode": mode,
                    "budget": 0.02,
                    "init_return": init_return,
                    "phi_init": phi_init,
                    "init_source_gain": init_gain,
                    "uncertified_at_init": bool(init_gain < GAMMA_POS),
                    "return_increase": series[-1]["true_return"] - init_return,
                    "final_return": series[-1]["true_return"],
                    "final_phi": series[-1]["phi"],
                    "final_source": series[-1]["source_certificate"],
                    "final_updated": series[-1]["updated_certificate"],
                    "target": aim,
                })
    gym_env.close()
    _write_rows(out, rows, {}, started, stage="learned")


if __name__ == "__main__":
    main()
