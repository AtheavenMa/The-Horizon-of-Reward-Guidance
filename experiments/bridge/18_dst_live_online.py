#!/usr/bin/env python3
"""P3: live Gym online preference learning on Deep Sea Treasure."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.bridge.common import write_summary
from experiments.bridge.pref_common import (
    DST,
    ROOT,
    TRUE_W,
    jsonable,
    live_batch,
    reinforce,
    frozen_heldout,
    run_online,
    softmax,
)


METHODS = ("S", "U", "I", "M", "O")


def raw_grad(dst: DST, gym_env, logits, weight, n, rng) -> np.ndarray:
    episodes, _steps = live_batch(gym_env, dst, logits, n, rng)
    probs = softmax(logits)
    feats = np.stack([((ep["ret"] - dst.low) / dst.scale) for ep in episodes])
    returns = feats @ weight
    baseline = float(returns.mean())
    grad = np.zeros_like(logits)
    for ep, value in zip(episodes, returns):
        advantage = float(value - baseline)
        for state, action in ep["steps"]:
            one = np.zeros(4)
            one[action] = 1.0
            grad[state] += advantage * (one - probs[state])
    return grad / max(n, 1)


def cosine(a, b) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return 0.0
    return float(np.sum(a * b) / denom)


def gradient_quality(dst: DST, gym_env) -> dict:
    rows = []
    logits = np.zeros((dst.n, 4))
    for name, steps in (("uniform", 0), ("oracle10", 10)):
        if steps:
            rng = np.random.default_rng(7)
            for _ in range(steps):
                episodes, _ = live_batch(gym_env, dst, logits, 256, rng)
                logits = reinforce(logits, episodes, np.array([TRUE_W, 1.0 - TRUE_W]), dst.low, dst.high)
        exact = dst.gradient(softmax(logits), TRUE_W)
        for n in (256, 512):
            estimates = [raw_grad(dst, gym_env, logits, np.array([TRUE_W, 1.0 - TRUE_W]), n, np.random.default_rng(80 + n + 10 * k)) for k in range(3)]
            rows.append({
                "policy": name,
                "episodes": n,
                "cosine_mean": float(np.mean([cosine(g, exact) for g in estimates])),
                "relative_error_mean": float(np.mean([np.linalg.norm(g - exact) / max(np.linalg.norm(exact), 1e-8) for g in estimates])),
            })
    cos256 = np.mean([row["cosine_mean"] for row in rows if row["episodes"] == 256])
    cos512 = np.mean([row["cosine_mean"] for row in rows if row["episodes"] == 512])
    # The declared default is 256. Move to 512 only if 256 is a poor direction
    # and 512 repairs it by a clear margin.
    chosen = 512 if cos256 < 0.5 and cos512 > cos256 + 0.2 else 256
    return {"rows": rows, "chosen_episodes": int(chosen), "cosine_256": float(cos256), "cosine_512": float(cos512)}


def live_check(dst: DST, gym_env, logits, n=2000, seed=11) -> dict:
    rng = np.random.default_rng(seed)
    episodes, _ = live_batch(gym_env, dst, logits, n, rng)
    mean = np.mean([ep["ret"] for ep in episodes], axis=0)
    mu = dst.mu(logits)
    se = np.std([ep["ret"] for ep in episodes], axis=0, ddof=1) / np.sqrt(n)
    return {"gap": (mean - mu).tolist(), "se": se.tolist(), "within_4se": bool(np.all(np.abs(mean - mu) <= 4 * np.maximum(se, 1e-8)))}


def aggregate(rows: list[dict]) -> dict:
    out = {}
    for method in METHODS:
        group = [row for row in rows if row["method"] == method]
        if not group:
            continue
        final = np.array([row["final_return"] for row in group])
        ref = {row["seed"]: row["final_return"] for row in rows if row["method"] == "S"}
        paired = np.array([row["final_return"] - ref[row["seed"]] for row in group])
        out[method] = {
            "n": len(group),
            "return_0": float(np.mean([row["return_0"] for row in group])),
            "return_40_mean": float(np.mean([row["return_40"] for row in group])),
            "return_40_sd": float(np.std([row["return_40"] for row in group], ddof=1)) if len(group) > 1 else 0.0,
            "return_80_mean": float(np.mean([row["return_80"] for row in group])),
            "return_160_mean": float(np.mean(final)),
            "return_160_sd": float(np.std(final, ddof=1)) if len(group) > 1 else 0.0,
            "paired_vs_S_mean": float(np.mean(paired)),
            "paired_vs_S_sd": float(np.std(paired, ddof=1)) if len(group) > 1 else 0.0,
            "auc_mean": float(np.mean([row["auc"] for row in group])),
            "final_phi_mean": float(np.mean([row["final_phi"] for row in group])),
            "heldout_nll_mean": None if group[0]["heldout_nll"] is None else float(np.mean([row["heldout_nll"] for row in group])),
            "final_width_mean": None if group[0]["final_width"] is None else float(np.mean([row["final_width"] or 0.0 for row in group])),
            "new_labels": group[0]["new_labels"],
            "episodes_per_update": group[0]["episodes"],
            "training_episodes": group[0]["episodes"] * group[0]["updates"],
        }
    return out


def slim(row: dict) -> dict:
    keep = dict(row)
    keep.pop("logits", None)
    keep.pop("snapshots", None)
    return keep


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("pilot", "confirm"), required=True)
    parser.add_argument("--source", choices=("weak", "mixed"), default="weak")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    dst = DST()
    import mo_gymnasium as mo_gym
    gym_env = mo_gym.make("deep-sea-treasure-v0")
    if args.stage == "pilot":
        quality = gradient_quality(dst, gym_env)
        rows = []
        for seed in (900, 901, 902):
            for method in ("S", "I", "M"):
                path = out / f"pilot_{method}_seed{seed}.json"
                if path.exists():
                    rows.append(json.loads(path.read_text()))
                    continue
                row = run_online(
                    dst, gym_env, seed=seed, method=method, regime="weak", updates=40,
                    episodes=quality["chosen_episodes"], budget=0.02,
                    feedback_at=(20, 40), label_cap=64,
                )
                path.write_text(json.dumps(jsonable(slim(row))))
                rows.append(slim(row))
                print(method, seed, row["final_return"], flush=True)
        check = live_check(dst, gym_env, np.zeros((dst.n, 4)))
        gym_env.close()
        payload = {"stage": "pilot", "gradient": quality, "live_check_uniform": check, "pilot": aggregate(rows)}
        (out / "pilot.json").write_text(json.dumps(jsonable(payload), indent=2) + "\n")
        write_summary(out, jsonable(payload), started)
        return

    pilot_payload = json.loads((ROOT / "outputs/horizon_bridge/p3_dst_live_weak/pilot.json").read_text())
    quality = pilot_payload["gradient"]
    episodes = int(quality["chosen_episodes"])
    held = frozen_heldout(dst, gym_env)
    seeds = list(range(8)) if args.source == "weak" else list(range(4))
    rows = []
    for seed in seeds:
        for method in METHODS:
            path = out / f"{method}_seed{seed}.json"
            if path.exists():
                row = json.loads(path.read_text())
            else:
                row = run_online(
                    dst, gym_env, seed=seed, method=method, regime=args.source, updates=160,
                    episodes=episodes, budget=0.02, held=held,
                    save_at=(20, 60) if method == "S" and seed == 0 else (),
                )
                if args.source == "weak" and method == "S" and seed == 0:
                    for update, logits in row["snapshots"].items():
                        (out / f"checkpoint_S_seed0_update{update}.json").write_text(json.dumps({
                            "update": int(update),
                            "logits": logits,
                            "note": "Source-only learned policy. The teacher weight is not stored.",
                        }))
                path.write_text(json.dumps(jsonable(slim(row))))
                print(args.source, method, seed, row["final_return"], flush=True)
            rows.append(slim(row) if "logits" in row else row)
    payload = {
        "stage": "confirm",
        "source": args.source,
        "episodes_per_update": episodes,
        "gradient_rule": quality,
        "pilot": pilot_payload.get("pilot"),
        "live_check_uniform": pilot_payload.get("live_check_uniform"),
        "methods": aggregate(rows),
    }
    if args.source == "weak":
        check_logits = np.array(json.loads((out / "checkpoint_S_seed0_update20.json").read_text())["logits"])
        payload["live_check_learned"] = live_check(dst, gym_env, check_logits)
    gym_env.close()
    write_summary(out, jsonable(payload), started)


if __name__ == "__main__":
    main()
