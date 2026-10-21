#!/usr/bin/env python3
"""R2: replay the 160-update DST trajectories and separate state from guidance."""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.bridge.common import write_summary
from experiments.bridge.pref_common import (
    DST,
    TRUE_W,
    frozen_heldout,
    gain,
    jsonable,
    outcome_distribution,
    run_online,
    select_phi,
    softmax,
)


SAVE_AT = (0, 10, 20, 40, 60, 80, 120, 160)
ROOT = Path(__file__).resolve().parents[2]


def replay_one(job: tuple[int, str]) -> dict:
    seed, method = job
    import mo_gymnasium as mo_gym
    from experiments.bridge.pref_common import DST, frozen_heldout, run_online

    dst = DST()
    gym_env = mo_gym.make("deep-sea-treasure-v0")
    held = frozen_heldout(dst, gym_env) if method != "O" else None
    row = run_online(
        dst, gym_env, seed=seed, method=method, regime="weak", updates=160,
        episodes=256, budget=0.02, held=held, save_at=SAVE_AT, trace_info=True,
    )
    saved_path = ROOT / f"outputs/horizon_bridge/p3_dst_live_weak/{method}_seed{seed}.json"
    saved = json.loads(saved_path.read_text())
    gap = float(np.max(np.abs(np.array(saved["returns"]) - np.array(row["returns"]))))
    phi_gap = float(np.max(np.abs(np.array(saved["phis"]) - np.array(row["phis"]))))
    folder = ROOT / "outputs/horizon_bridge/r2_dst_paths/checkpoints"
    folder.mkdir(parents=True, exist_ok=True)
    arrays = {f"u{k}": np.array(v, dtype=np.float64) for k, v in row["snapshots"].items()}
    np.savez_compressed(folder / f"{method}_seed{seed}.npz", **arrays)
    checks = {}
    for update, payload in row["trace"].items():
        logits = arrays[f"u{update}"]
        dist = outcome_distribution(dst, logits)
        pi = softmax(logits)
        cache = dst.backward(pi)
        g_true = dst.gradient(pi, TRUE_W, cache=cache)
        g_phi = dst.gradient(pi, float(payload["phi"]), cache=cache)
        checks[update] = {
            "phi": payload["phi"],
            "w_hat": payload["w_hat"],
            "lo": payload["lo"],
            "hi": payload["hi"],
            "source_lo": payload.get("source_lo", payload["lo"]),
            "source_hi": payload.get("source_hi", payload["hi"]),
            "mu": payload["mu"],
            "return": row["returns"][int(update)],
            "treasure": dist["bins"],
            "expected_length": dist["expected_length"],
            "grad_norm_true": float(np.linalg.norm(g_true)),
            "grad_norm_phi": float(np.linalg.norm(g_phi)),
            "cosine": _cosine(g_true, g_phi),
        }
    gym_env.close()
    return {
        "seed": seed,
        "method": method,
        "return_gap": gap,
        "phi_gap": phi_gap,
        "returns": row["returns"],
        "phis": row["phis"],
        "gains": row["gains"],
        "checks": checks,
        "final_return": row["final_return"],
        "auc": row["auc"],
        "heldout_nll": row["heldout_nll"],
    }


def _cosine(a, b) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 1.0
    return float(np.dot(a.ravel(), b.ravel()) / denom)


def segment(returns, start, stop) -> float:
    return float(returns[stop] - returns[start])


def exact_future(dst: DST, logits, phi, bounds, steps: int) -> dict:
    start = dst.true_return(dst.mu(logits))
    for _ in range(steps):
        window = (max(0.0, phi - 0.02), min(1.0, phi + 0.02))
        chosen, _worst, _true = select_phi(dst, logits, bounds, window, phi)
        logits, _mu, _g = dst.exact_step(logits, chosen)
        phi = chosen
    final = dst.true_return(dst.mu(logits))
    return {"return_gain": final - start, "final_return": final, "final_phi": phi}


def constant_sweep(dst: DST) -> list[dict]:
    rows = []
    for phi in (0.0, 0.25, 0.375, 0.5, 0.75, 1.0):
        logits = np.zeros((dst.n, 4))
        for _ in range(160):
            logits, _mu, _g = dst.exact_step(logits, phi)
        mu = dst.mu(logits)
        rows.append({
            "phi": phi,
            "final_return": dst.true_return(mu),
            "mu": mu.tolist(),
            "treasure": outcome_distribution(dst, logits)["bins"],
        })
    return rows


def positive_controls(dst: DST) -> dict:
    def exact(phi, steps):
        logits = np.zeros((dst.n, 4))
        for _ in range(steps):
            logits, _mu, _g = dst.exact_step(logits, phi)
        return dst.true_return(dst.mu(logits))

    def lookahead(steps):
        logits = np.zeros((dst.n, 4))
        phi = 0.5
        for _ in range(steps):
            window = (max(0.0, phi - 0.02), min(1.0, phi + 0.02))
            phi, _w, _t = select_phi(dst, logits, (TRUE_W, TRUE_W), window, phi)
            logits, _mu, _g = dst.exact_step(logits, phi)
        return {"final_return": dst.true_return(dst.mu(logits)), "final_phi": phi}

    return {
        "exact_true_w_160": exact(TRUE_W, 160),
        "exact_true_w_320": exact(TRUE_W, 320),
        "exact_A_true_160": lookahead(160),
    }


def main() -> None:
    started = time.perf_counter()
    out = ROOT / "outputs/horizon_bridge/r2_dst_paths"
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(seed, method) for seed in range(8) for method in ("S", "U", "I", "M", "O")]
    with ProcessPoolExecutor(max_workers=6) as pool:
        rows = list(pool.map(replay_one, jobs))
    max_gap = max(row["return_gap"] for row in rows)
    if max_gap > 1e-9:
        raise SystemExit(f"replay disagrees with P3 by {max_gap}")
    dst = DST()
    forks = []
    swaps = []
    for row in rows:
        if row["seed"] > 2 or row["method"] not in ("S", "I", "M", "O"):
            continue
        loaded = np.load(out / "checkpoints" / f"{row['method']}_seed{row['seed']}.npz")
        for update in (40, 80, 120):
            check = row["checks"][str(update)]
            logits = loaded[f"u{update}"]
            phi = float(check["phi"])
            source_bounds = (float(check["source_lo"]), float(check["source_hi"]))
            updated = (float(check["lo"]), float(check["hi"])) if check["lo"] is not None else source_bounds
            w_hat = float(check["w_hat"])
            controllers = {
                "A_source": source_bounds,
                "A_updated": updated,
                "A_mle": (w_hat, w_hat),
                "A_true": (TRUE_W, TRUE_W),
            }
            window = (max(0.0, phi - 0.02), min(1.0, phi + 0.02))
            for name, bounds in controllers.items():
                chosen, _worst, true_gain = select_phi(dst, logits, bounds, window, phi)
                swaps.append({
                    "seed": row["seed"], "method": row["method"], "update": update,
                    "controller": name, "phi": chosen, "true_one_step": true_gain,
                })
                future = exact_future(dst, logits, phi, bounds, 40)
                forks.append({
                    "seed": row["seed"], "origin": row["method"], "update": update,
                    "controller": name, **future,
                })
    summary = {
        "replay_max_return_gap": max_gap,
        "replay_max_phi_gap": max(row["phi_gap"] for row in rows),
        "methods": _method_table(rows),
        "mechanisms": _mechanism_table(rows),
        "swaps": _group_mean(swaps, ("origin" if False else "method", "update", "controller"), "true_one_step"),
        "forks": _group_mean(forks, ("origin", "update", "controller"), "return_gain"),
        "constant_phi": constant_sweep(dst),
        "positive_controls": positive_controls(dst),
        "stage_runtime_seconds": time.perf_counter() - started,
    }
    # The grouped swap key uses method. Keep the raw rows for the paper table.
    (out / "swaps.json").write_text(json.dumps(jsonable(swaps), indent=2) + "\n")
    (out / "forks.json").write_text(json.dumps(jsonable(forks), indent=2) + "\n")
    slim = [{k: v for k, v in row.items() if k != "returns"} | {"returns_at": {str(t): row["returns"][t] for t in SAVE_AT}} for row in rows]
    (out / "replay.json").write_text(json.dumps(jsonable(slim), indent=2) + "\n")
    write_summary(out, jsonable(summary), started)


def _method_table(rows):
    out = {}
    for method in ("S", "U", "I", "M", "O"):
        group = [row for row in rows if row["method"] == method]
        finals = np.array([row["returns"][160] for row in group])
        out[method] = {
            "n": len(group),
            "return_40": float(np.mean([row["returns"][40] for row in group])),
            "return_80": float(np.mean([row["returns"][80] for row in group])),
            "return_160": float(np.mean(finals)),
            "sd_160": float(np.std(finals, ddof=1)),
            "auc": float(np.mean([row["auc"] for row in group])),
            "gain_0_40": float(np.mean([segment(row["returns"], 0, 40) for row in group])),
            "gain_40_80": float(np.mean([segment(row["returns"], 40, 80) for row in group])),
            "gain_80_160": float(np.mean([segment(row["returns"], 80, 160) for row in group])),
        }
    return out


def _mechanism_table(rows):
    out = {}
    for method in ("S", "I", "M", "O"):
        group = [row for row in rows if row["method"] == method]
        out[method] = {}
        for update in (40, 80, 120, 160):
            checks = [row["checks"][str(update)] for row in group]
            short = np.mean([item["treasure"].get("0.7", 0.0) for item in checks])
            deep = np.mean([sum(prob for key, prob in item["treasure"].items() if key not in ("0.7", "truncated")) for item in checks])
            out[method][str(update)] = {
                "phi": float(np.mean([item["phi"] for item in checks])),
                "w_hat": float(np.mean([item["w_hat"] for item in checks])),
                "width": float(np.mean([0.0 if item["lo"] is None else item["hi"] - item["lo"] for item in checks])),
                "short_treasure": float(short),
                "deeper_treasure": float(deep),
                "truncation": float(np.mean([item["treasure"].get("truncated", 0.0) for item in checks])),
                "cosine_true_vs_deployed": float(np.mean([item["cosine"] for item in checks])),
                "grad_norm_true": float(np.mean([item["grad_norm_true"] for item in checks])),
            }
    return out


def _group_mean(rows, keys, value):
    grouped = {}
    for row in rows:
        key = "|".join(str(row[name]) for name in keys)
        grouped.setdefault(key, []).append(row[value])
    return {key: {"n": len(values), "mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0} for key, values in grouped.items()}


if __name__ == "__main__":
    main()
