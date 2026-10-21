#!/usr/bin/env python3
"""R3: at a frozen DST policy, separate new preference information from the action it causes."""
from __future__ import annotations

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
    TRUE_W,
    build_source,
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
    worst_gain,
)


ROOT = Path(__file__).resolve().parents[2]
ALPHA = 0.01


def new_pool(dst, gym_env, logits, salt, w_hat):
    rng = np.random.default_rng(120_000 + salt)
    groups = [
        live_batch(gym_env, dst, logits, 64, rng)[0],
        live_batch(gym_env, dst, np.zeros((dst.n, 4)), 32, rng)[0],
        live_batch(gym_env, dst, scripted_treasure(dst), 32, rng)[0],
    ]
    feats = np.stack([(ep["ret"] - dst.low) / dst.scale for group in groups for ep in group])
    left, right = make_pairs(len(feats), 256, rng)
    c, d = pair_terms(feats, left, right, BETA)
    fisher_idx = np.argsort(-fisher_at(c, d, float(w_hat)))[:32]
    uniform_idx = np.random.default_rng(130_000 + salt).choice(len(d), size=32, replace=False)
    return c, d, fisher_idx, uniform_idx


def future(dst, logits, phi, bounds, steps):
    start = dst.true_return(dst.mu(logits))
    phis = []
    for _ in range(steps):
        window = (max(0.0, phi - 0.02), min(1.0, phi + 0.02))
        chosen, _worst, _true = select_phi(dst, logits, bounds, window, phi)
        logits, _mu, _g = dst.exact_step(logits, chosen)
        phi = chosen
        phis.append(chosen)
    return dst.true_return(dst.mu(logits)) - start, phis


def main() -> None:
    started = time.perf_counter()
    replay = json.loads((ROOT / "outputs/horizon_bridge/r2_dst_paths/replay.json").read_text())
    parents = [row for row in replay if row["method"] == "S" and row["seed"] in (0, 1, 2)]
    if len(parents) != 3:
        raise SystemExit("need replayed source-only seeds 0, 1 and 2")
    dst = DST()
    import mo_gymnasium as mo_gym
    gym_env = mo_gym.make("deep-sea-treasure-v0")
    rows = []
    sampled = []
    for parent in parents:
        ckpt = np.load(ROOT / f"outputs/horizon_bridge/r2_dst_paths/checkpoints/S_seed{parent['seed']}.npz")
        source = build_source(dst, gym_env, parent["seed"], "weak")
        for update in (20, 40, 60, 80):
            logits0 = ckpt[f"u{update}"]
            phi0 = float(parent["checks"][str(update)]["phi"])
            w_hat0 = float(parent["checks"][str(update)]["w_hat"])
            c_pool, d_pool, fisher_idx, uniform_idx = new_pool(dst, gym_env, logits0, 1000 * parent["seed"] + update, w_hat0)
            rules = {"none": None, "fisher": fisher_idx, "uniform": uniform_idx}
            if update != 20:
                rules.pop("none")
            for rule, idx in rules.items():
                draws = [None] if idx is None else list(range(8))
                for draw in draws:
                    if idx is None:
                        c = source["c"]
                        d = source["d"]
                        y = source["y"]
                        updated = source["interval"] or (0.0, 1.0)
                        w_hat = source["w_hat"]
                    else:
                        y_new = label_bits(c_pool[idx], d_pool[idx], np.random.default_rng(140_000 + 1000 * draw + update))
                        c = np.concatenate([source["c"], c_pool[idx]])
                        d = np.concatenate([source["d"], d_pool[idx]])
                        y = np.concatenate([source["y"], y_new])
                        updated = confidence_interval(c, d, y, alpha=ALPHA) or (0.0, 1.0)
                        w_hat = fit_mle(c, d, y)
                    u_source = source["interval"] or (0.0, 1.0)
                    window = (max(0.0, phi0 - 0.02), min(1.0, phi0 + 0.02))
                    phi_s, worst_s, true_s = select_phi(dst, logits0, u_source, window, phi0)
                    phi_u, worst_u, true_u = select_phi(dst, logits0, updated, window, phi0)
                    phi_m, _worst_m, true_m = select_phi(dst, logits0, (w_hat, w_hat), window, phi0)
                    # Frozen-path certificate: the same next policy, so the return cannot change.
                    nxt, mu, _g = dst.exact_step(logits0, phi0)
                    frozen_true = dst.true_return(dst.mu(nxt)) - dst.true_return(mu)
                    frozen_source = worst_gain(dst, mu, dst.mu(nxt), u_source)
                    frozen_updated = worst_gain(dst, mu, dst.mu(nxt), updated)
                    gain_s, path_s = future(dst, logits0, phi0, u_source, 20)
                    gain_u, path_u = future(dst, logits0, phi0, updated, 20)
                    gain_m, path_m = future(dst, logits0, phi0, (w_hat, w_hat), 20)
                    rows.append({
                        "seed": parent["seed"],
                        "update": update,
                        "rule": rule,
                        "draw": -1 if draw is None else draw,
                        "width_old": float(u_source[1] - u_source[0]),
                        "width_new": float(updated[1] - updated[0]),
                        "w_hat": w_hat,
                        "phi_source": phi_s,
                        "phi_updated": phi_u,
                        "phi_mle": phi_m,
                        "phi_changed": abs(phi_u - phi_s) > 1e-8,
                        "endpoint_slack": float(min(abs(phi_u - window[0]), abs(phi_u - window[1]))),
                        "one_step_source": true_s,
                        "one_step_updated": true_u,
                        "one_step_mle": true_m,
                        "one_step_difference": true_u - true_s,
                        "frozen_return_gap": 0.0,
                        "frozen_true_gain": frozen_true,
                        "certificate_old": frozen_source,
                        "certificate_new": frozen_updated,
                        "source_above_gamma": bool(worst_s >= GAMMA_POS),
                        "gain20_source": gain_s,
                        "gain20_updated": gain_u,
                        "gain20_mle": gain_m,
                        "gain20_difference": gain_u - gain_s,
                        "info_only_return_equals_source": gain_u == gain_s,
                    })
                    if update == 40 and rule == "fisher" and draw in (0, 1, 2):
                        for mode, bounds in (("source", u_source), ("updated", updated)):
                            logits = logits0.copy()
                            phi = phi0
                            rng = np.random.default_rng(150_000 + 10 * parent["seed"] + draw)
                            ret0 = dst.true_return(dst.mu(logits))
                            for _k in range(20):
                                chosen, _w, _t = select_phi(dst, logits, bounds, (max(0.0, phi - 0.02), min(1.0, phi + 0.02)), phi)
                                rollouts, _steps = live_batch(gym_env, dst, logits, 256, rng)
                                logits = reinforce(logits, rollouts, np.array([chosen, 1.0 - chosen]), dst.low, dst.high)
                                phi = chosen
                            sampled.append({
                                "seed": parent["seed"], "draw": draw, "mode": mode,
                                "return_gain": dst.true_return(dst.mu(logits)) - ret0,
                            })
    gym_env.close()
    out = ROOT / "outputs/horizon_bridge/r3_information_forks"
    out.mkdir(parents=True, exist_ok=True)
    (out / "rows.json").write_text(json.dumps(jsonable(rows)) + "\n")
    any_certified = any(row["source_above_gamma"] for row in rows if row["rule"] == "none" or row["rule"] == "fisher")
    summary = {
        "alpha": ALPHA,
        "gamma_pos": GAMMA_POS,
        "initial_information": "sufficient at some source checkpoint" if any_certified else "INITIAL INFORMATION INSUFFICIENT",
        "groups": _groups(rows),
        "sampled": sampled,
        "max_frozen_return_gap": 0.0,
        "stage_runtime_seconds": time.perf_counter() - started,
    }
    write_summary(out, jsonable(summary), started)


def _groups(rows):
    out = {}
    for row in rows:
        key = f"{row['update']}|{row['rule']}"
        out.setdefault(key, []).append(row)
    result = {}
    for key, group in out.items():
        result[key] = {
            "n": len(group),
            "parents": len({row["seed"] for row in group}),
            "width_old": float(np.mean([row["width_old"] for row in group])),
            "width_new": float(np.mean([row["width_new"] for row in group])),
            "phi_changed_rate": float(np.mean([row["phi_changed"] for row in group])),
            "one_step_difference": float(np.mean([row["one_step_difference"] for row in group])),
            "gain20_difference": float(np.mean([row["gain20_difference"] for row in group])),
            "gain20_source": float(np.mean([row["gain20_source"] for row in group])),
            "gain20_updated": float(np.mean([row["gain20_updated"] for row in group])),
            "gain20_mle": float(np.mean([row["gain20_mle"] for row in group])),
            "certificate_old": float(np.mean([row["certificate_old"] for row in group])),
            "certificate_new": float(np.mean([row["certificate_new"] for row in group])),
            "source_certified_rate": float(np.mean([row["source_above_gamma"] for row in group])),
        }
    return result


if __name__ == "__main__":
    main()
