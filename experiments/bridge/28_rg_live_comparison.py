#!/usr/bin/env python3
"""R4C: live Resource-Gathering under matched preference designs.

The outer-region controller is included only when R4B produced a nonempty cover
tighter than the simplex. The full-simplex arm is the conservative reference.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.bridge.common import write_summary
from experiments.bridge.pref_common import jsonable, softmax


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cosine(a, b) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 1.0 if denom == 0 else float(np.dot(a.ravel(), b.ravel()) / denom)


def gradient_pilot(rg, model, env) -> dict:
    rows = {}
    policies = {"uniform": np.zeros((model.n, 4))}
    logits = np.zeros((model.n, 4))
    rng = np.random.default_rng(3)
    for _ in range(40):
        episodes, _steps = rg.live_batch(env, model, logits, 256, rng)
        logits = rg.reinforce(logits, episodes, rg.W_STAR)
    policies["oracle40"] = logits
    for name, logits in policies.items():
        exact = model.gradient(softmax(logits), rg.W_STAR)
        rows[name] = {}
        for n in (256, 512, 1024, 2048):
            episodes, _steps = rg.live_batch(env, model, logits, n, np.random.default_rng(11))
            # The sampled gradient is the REINFORCE update times the episode count.
            trial = rg.reinforce(logits, episodes, rg.W_STAR)
            sampled = (trial - logits) * max(n, 1)
            rows[name][str(n)] = {"cosine": cosine(exact, sampled), "episodes": n}
    return rows


def project(current, proposal, budget=0.02):
    clipped = np.clip(proposal, current - budget, current + budget)
    clipped = np.clip(clipped, 0.0, 1.0)
    if clipped.sum() <= 1.0:
        return clipped
    lo, hi = 0.0, 1.0
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        cand = current + mid * (clipped - current)
        if cand.sum() <= 1.0:
            lo = mid
        else:
            hi = mid
    return np.clip(current + lo * (clipped - current), 0.0, 1.0)


def candidates(current):
    out = []
    for da in np.linspace(-0.02, 0.02, 5):
        for db in np.linspace(-0.02, 0.02, 5):
            out.append(project(current, current + np.array([da, db])))
    return out


def worst_simplex(model, rg, logits, cand):
    weight = np.array([cand[0], cand[1], 1.0 - cand.sum()])
    mu = model.values(softmax(logits))
    nxt = logits + model.gradient(softmax(logits), weight)
    mu2 = model.values(softmax(nxt))
    delta = (mu2 - mu) / rg.SCALE
    gains = []
    for w in ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)):
        full = np.array([w[0], w[1], 1.0 - w[0] - w[1]])
        gains.append(float(full @ delta))
    return min(gains)


def best_lookahead(model, rg, logits, current, objective):
    best_w, best_value = current.copy(), -1e100
    for cand in candidates(current):
        value = objective(logits, cand)
        if value > best_value:
            best_w, best_value = cand, value
    return best_w, best_value


def run_job(job):
    seed, method, episodes = job
    rg = load("20_resource_gathering")
    model = rg.RG()
    import mo_gymnasium as mo_gym
    env = mo_gym.make("resource-gathering-v0")
    c0, d0, y0, steps = rg.collect_source(env, model, seed)
    batches = [(c0, d0, y0)]
    fitted, _soft, gap = rg.fit_simplex(c0, d0, y0)
    deployed = np.array([1.0 / 3.0, 1.0 / 3.0])
    logits = np.zeros((model.n, 4))
    rng = np.random.default_rng(800_000 + 100 * seed + (sum(ord(ch) for ch in method) % 97))
    mu0 = model.values(softmax(logits))
    curve = [rg.scalarize(mu0, rg.W_STAR)]
    acquired = 0
    rule = {"uniform": "uniform", "trace": "fisher", "mineig": "mineig", "simplex": "mineig", "lookahead": "mineig"}.get(method)
    for k in range(1, 101):
        if rule and k in (40, 80) and acquired < 64:
            c, d, y, extra = collect_rule(rg, env, model, logits, seed, k, fitted, rule)
            steps += extra
            batches.append((c, d, y))
            acquired += len(y)
            c_all = np.concatenate([item[0] for item in batches])
            d_all = np.concatenate([item[1] for item in batches])
            y_all = np.concatenate([item[2] for item in batches])
            fitted, _soft, gap = rg.fit_simplex(c_all, d_all, y_all)
        if method == "oracle":
            chosen = rg.W_STAR[:2].copy()
            weight = rg.W_STAR
        elif method == "lookahead":
            def true_gain(logits_now, cand, model=model, rg=rg):
                weight_now = np.array([cand[0], cand[1], 1.0 - cand.sum()])
                mu = model.values(softmax(logits_now))
                nxt = logits_now + model.gradient(softmax(logits_now), weight_now)
                return rg.scalarize(model.values(softmax(nxt)), rg.W_STAR) - rg.scalarize(mu, rg.W_STAR)
            chosen, _value = best_lookahead(model, rg, logits, deployed, true_gain)
            weight = np.array([chosen[0], chosen[1], 1.0 - chosen.sum()])
        elif method == "simplex":
            chosen, _value = best_lookahead(model, rg, logits, deployed, lambda logits_now, cand: worst_simplex(model, rg, logits_now, cand))
            weight = np.array([chosen[0], chosen[1], 1.0 - chosen.sum()])
        elif method == "exact_true":
            chosen = rg.W_STAR[:2].copy()
            weight = rg.W_STAR
        else:
            chosen = project(deployed, fitted)
            weight = np.array([chosen[0], chosen[1], 1.0 - chosen.sum()])
        if method == "exact_true":
            logits = logits + model.gradient(softmax(logits), weight)
        else:
            rollouts, extra = rg.live_batch(env, model, logits, episodes, rng)
            steps += extra
            logits = rg.reinforce(logits, rollouts, weight)
        deployed = chosen
        curve.append(rg.scalarize(model.values(softmax(logits)), rg.W_STAR))
    mu = model.values(softmax(logits))
    c_all = np.concatenate([item[0] for item in batches])
    d_all = np.concatenate([item[1] for item in batches])
    y_all = np.concatenate([item[2] for item in batches])
    p = 1.0 / (1.0 + np.exp(-np.clip(c_all + d_all @ fitted, -40, 40)))
    fish = (d_all * (p * (1 - p))[:, None]).T @ d_all
    eigs = np.linalg.eigvalsh(fish)
    env.close()
    return {
        "seed": seed,
        "method": method,
        "episodes": episodes,
        "return_0": curve[0],
        "final_return": curve[-1],
        "mu_delta": (mu - mu0).tolist(),
        "final_fit": fitted.tolist(),
        "w_error": float(np.linalg.norm(np.array([fitted[0], fitted[1], 1 - fitted.sum()]) - rg.W_STAR)),
        "eig_min": float(eigs[0]),
        "new_labels": acquired,
        "fit_gap": float(gap),
    }


def collect_rule(rg, env, model, logits, seed, event, w_hat, rule):
    if rule != "mineig":
        c, d, y, steps, _fisher = rg.collect_feedback(env, model, logits, seed, event, w_hat, "uniform" if rule == "uniform" else "fisher")
        return c, d, y, steps
    rng = np.random.default_rng(600_000 + 1000 * seed + event)
    current, s1 = rg.live_batch(env, model, logits, 48, rng)
    uniform, s2 = rg.live_batch(env, model, np.zeros((model.n, 4)), 16, rng)
    feats = np.stack([((ep["ret"] - rg.LOW) / rg.SCALE) for ep in current + uniform])
    left, right = rg.make_pairs(len(feats), 128, rng)
    c, d = rg.design(feats[left] - feats[right])
    spec = importlib.util.spec_from_file_location("r4a", Path(__file__).resolve().parent / "25_rg_identification.py")
    r4a = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r4a)
    chosen = r4a.select_indices(c, d, w_hat, "mineig", 32, rng)
    y = (np.random.default_rng(700_000 + 1000 * seed + event).random(32) < rg.sigmoid(c[chosen] + d[chosen] @ rg.W_STAR[:2])).astype(float)
    return c[chosen], d[chosen], y, s1 + s2


def main() -> None:
    started = time.perf_counter()
    rg = load("20_resource_gathering")
    model = rg.RG()
    import mo_gymnasium as mo_gym
    env = mo_gym.make("resource-gathering-v0")
    pilot = gradient_pilot(rg, model, env)
    env.close()
    # Larger batches did not materially improve the RG cosine in P5. Keep 256 unless a larger batch gains at least 0.2.
    cos256 = pilot["uniform"]["256"]["cosine"]
    cos2048 = pilot["uniform"]["2048"]["cosine"]
    episodes = 2048 if cos2048 > cos256 + 0.2 else 256
    methods = ["source", "uniform", "trace", "mineig", "simplex", "oracle", "lookahead", "exact_true"]
    jobs = [(seed, method, episodes) for method in methods for seed in range(8)]
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(run_job, jobs))
    out = Path("outputs/horizon_bridge/r4_rg_robust")
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "gradient_pilot": pilot,
        "episodes": episodes,
        "methods": _agg(rows),
        "rows": rows,
        "stage_runtime_seconds": time.perf_counter() - started,
    }
    (out / "live_rows.json").write_text(json.dumps(jsonable(rows)) + "\n")
    # Keep the R4B summary beside this live summary.
    (out / "live_summary.json").write_text(json.dumps(jsonable(payload), indent=2) + "\n")
    print(json.dumps(jsonable(payload["methods"]), indent=2))


def _agg(rows):
    out = {}
    for method in sorted({row["method"] for row in rows}):
        group = [row for row in rows if row["method"] == method]
        finals = np.array([row["final_return"] for row in group])
        delta = np.mean([row["mu_delta"] for row in group], axis=0)
        out[method] = {
            "n": len(group),
            "final_mean": float(finals.mean()),
            "final_sd": float(finals.std(ddof=1)),
            "mu_delta": delta.tolist(),
            "w_error": float(np.mean([row["w_error"] for row in group])),
            "eig_min": float(np.mean([row["eig_min"] for row in group])),
        }
    return out


if __name__ == "__main__":
    main()
