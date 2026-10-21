#!/usr/bin/env python3
"""H3: exact finite-horizon evaluation of deep-sea-treasure-v0."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.bridge.common import write_summary

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "experiments/horizon_real/E1/Deep-Sea-Treasure/frozen/frozen.json"
H = 20
N_ACTIONS = 4


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    return p


def build_model():
    import mo_gymnasium as mo_gym
    env = mo_gym.make("deep-sea-treasure-v0")
    raw = env.unwrapped
    sea = np.array(raw.sea_map, dtype=float)
    valid = []
    for r in range(sea.shape[0]):
        for c in range(sea.shape[1]):
            if raw._is_valid_state(np.array([r, c])):
                valid.append((r, c))
    index = {state: i for i, state in enumerate(valid)}
    # Transitions come from the installed step function, including staying on an illegal move.
    trans = []
    for state in valid:
        actions = []
        for action in range(N_ACTIONS):
            raw.current_state = np.array(state, dtype=np.int32)
            nxt, reward, terminated, truncated, _ = raw.step(action)
            reward = np.asarray(reward, dtype=np.float64).reshape(-1)
            nxt_key = (int(nxt[0]), int(nxt[1]))
            actions.append({
                "reward": reward,
                "next": None if terminated or truncated else index[nxt_key],
                "done": bool(terminated or truncated),
            })
        trans.append(actions)
    return {
        "states": valid,
        "index": index,
        "trans": trans,
        "start": index[(0, 0)],
        "map_name": raw.map_name,
        "spec_max_steps": None if env.spec is None else env.spec.max_episode_steps,
        "sea_shape": list(sea.shape),
    }


def scalarize(mu: np.ndarray, weight: np.ndarray, low: np.ndarray, high: np.ndarray) -> float:
    return float(weight @ ((mu - low) / (high - low)))


def exact_gradient(model, logits: np.ndarray, coef: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    n = len(model["states"])
    probs = softmax(logits)
    vector = np.zeros((H + 1, n, 2))
    q_scalar = np.zeros((H, n, N_ACTIONS))
    v_scalar = np.zeros((H, n))
    for t in range(H - 1, -1, -1):
        for s in range(n):
            acc = np.zeros(2)
            for a in range(N_ACTIONS):
                item = model["trans"][s][a]
                future = np.zeros(2) if item["done"] else vector[t + 1, item["next"]]
                total = item["reward"] + future
                q_scalar[t, s, a] = float(coef @ total)
                acc += probs[s, a] * total
            vector[t, s] = acc
            v_scalar[t, s] = float(coef @ acc)
    grad = np.zeros_like(logits)
    occ = np.zeros((H + 1, n))
    occ[0, model["start"]] = 1.0
    for t in range(H):
        for s in range(n):
            if occ[t, s] == 0.0:
                continue
            for a in range(N_ACTIONS):
                adv = q_scalar[t, s, a] - v_scalar[t, s]
                grad[s, a] += occ[t, s] * probs[s, a] * adv
                item = model["trans"][s][a]
                if not item["done"]:
                    occ[t + 1, item["next"]] += occ[t, s] * probs[s, a]
    mu = vector[0, model["start"]].copy()
    return grad, mu, float(coef @ mu)


def optimal_value(model, coef: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(model["states"])
    vector = np.zeros((H + 1, n, 2))
    action = np.zeros((H, n), dtype=int)
    for t in range(H - 1, -1, -1):
        for s in range(n):
            best, best_a = -1e100, 0
            best_vec = np.zeros(2)
            for a in range(N_ACTIONS):
                item = model["trans"][s][a]
                future = np.zeros(2) if item["done"] else vector[t + 1, item["next"]]
                total = item["reward"] + future
                score = float(coef @ total)
                if score > best:
                    best, best_a, best_vec = score, a, total
            vector[t, s] = best_vec
            action[t, s] = best_a
    return vector[0, model["start"]].copy(), action


def monte_carlo(model, probs: np.ndarray, n_episodes: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    totals = []
    start = model["start"]
    for _ in range(n_episodes):
        s = start
        total = np.zeros(2)
        for _t in range(H):
            a = int(rng.choice(N_ACTIONS, p=probs[s]))
            item = model["trans"][s][a]
            total += item["reward"]
            if item["done"]:
                break
            s = item["next"]
        totals.append(total)
    arr = np.stack(totals)
    mean = arr.mean(axis=0)
    se = arr.std(axis=0, ddof=1) / np.sqrt(len(arr))
    return {"mean": mean, "se": se}


def shortest_treasure(model) -> np.ndarray:
    """Deterministic one-step descent into the nearest treasure. Elsewhere uniform."""
    logits = np.zeros((len(model["states"]), N_ACTIONS))
    start = model["index"][(0, 0)]
    logits[start, 1] = 20.0  # action 1 is down in the installed environment
    return logits


def train_exact(model, coef, lr: float, updates: int, start: np.ndarray) -> dict:
    logits = start.copy()
    series = []
    for k in range(updates):
        grad, _before, _ = exact_gradient(model, logits, coef)
        logits = logits + lr * grad
        _after_grad, mu, _ = exact_gradient(model, logits, coef)
        series.append({"update": k + 1, "mu": mu.tolist(), "grad_norm": float(np.linalg.norm(grad))})
    return {"logits": logits, "series": series}


def rollout_episode(model, probs, rng) -> dict:
    s = model["start"]
    steps = []
    total = np.zeros(2)
    for _ in range(H):
        a = int(rng.choice(N_ACTIONS, p=probs[s]))
        item = model["trans"][s][a]
        steps.append((s, a, probs[s].copy()))
        total += item["reward"]
        if item["done"]:
            break
        s = item["next"]
    return {"steps": steps, "ret": total}


def train_sampled(model, coef, lr: float, updates: int, n_traj: int, seed: int, low, high, weight) -> dict:
    rng = np.random.default_rng(seed)
    logits = np.zeros((len(model["states"]), N_ACTIONS))
    series = []
    for k in range(updates):
        probs = softmax(logits)
        episodes = [rollout_episode(model, probs, rng) for _ in range(n_traj)]
        feats = np.stack([((ep["ret"] - low) / (high - low)) for ep in episodes])
        returns = feats @ weight
        baseline = float(returns.mean())
        grad = np.zeros_like(logits)
        for ep, value in zip(episodes, returns):
            adv = float(value - baseline)
            for s, a, prob in ep["steps"]:
                one = np.zeros(N_ACTIONS)
                one[a] = 1.0
                grad[s] += adv * (one - prob)
        step = lr * grad / max(len(episodes), 1)
        norm = np.linalg.norm(step)
        if norm > 1.0:
            step *= 1.0 / norm
        logits = logits + step
        _g, mu, _ = exact_gradient(model, logits, coef)
        series.append({"update": k + 1, "mu": mu.tolist(), "return": scalarize(mu, weight, low, high)})
    return {"series": series, "final_return": series[-1]["return"]}


def live_gap(model, logits, mu_dp) -> list[float]:
    """Monte Carlo in the installed environment, not in the extracted table."""
    import mo_gymnasium as mo_gym
    env = mo_gym.make("deep-sea-treasure-v0")
    rng = np.random.default_rng(7)
    probs = softmax(logits)
    totals = []
    for _ in range(2000):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        total = np.zeros(2)
        for _t in range(H):
            key = (int(obs[0]), int(obs[1]))
            action = int(rng.choice(N_ACTIONS, p=probs[model["index"][key]]))
            obs, reward, terminated, truncated, _ = env.step(action)
            total += np.asarray(reward, dtype=np.float64).reshape(-1)
            if terminated or truncated:
                break
        totals.append(total)
    mean = np.mean(totals, axis=0)
    return (mean - mu_dp).tolist()


def main() -> None:
    started = time.perf_counter()
    frozen = json.loads(FROZEN.read_text())
    low = np.array(frozen["low"], dtype=np.float64)
    high = np.array(frozen["high"], dtype=np.float64)
    weight = np.array(frozen["w_star"], dtype=np.float64)
    coef = weight / (high - low)
    model = build_model()
    down = model["trans"][model["start"]][1]
    if not down["done"] or abs(down["reward"][0] - 0.7) > 1e-5 or abs(down["reward"][1] + 1.0) > 1e-8:
        raise AssertionError(down)
    # H=20 already reaches the deepest convex-front treasure (19 steps). Do not extend.
    uniform = np.zeros((len(model["states"]), N_ACTIONS))
    g0, mu0, j0 = exact_gradient(model, uniform, coef)
    if np.linalg.norm(g0) < 1e-8:
        raise AssertionError("exact gradient is zero at the uniform policy")
    fd_gaps = []
    eps = 1e-6
    for s, a in ((model["start"], 1), (model["start"], 3), (min(3, len(model["states"]) - 1), 0)):
        plus, minus = uniform.copy(), uniform.copy()
        plus[s, a] += eps
        minus[s, a] -= eps
        _, _, jp = exact_gradient(model, plus, coef)
        _, _, jm = exact_gradient(model, minus, coef)
        fd_gaps.append(abs((jp - jm) / (2 * eps) - g0[s, a]))
    if max(fd_gaps) > 1e-6:
        raise AssertionError(fd_gaps)

    mc_uniform = monte_carlo(model, softmax(uniform), 4000, 0)
    if np.any(np.abs(mc_uniform["mean"] - mu0) > 4 * mc_uniform["se"]):
        raise AssertionError((mc_uniform["mean"], mu0, mc_uniform["se"]))
    short = shortest_treasure(model)
    _, mu_short, j_short = exact_gradient(model, short, coef)
    mc_short = monte_carlo(model, softmax(short), 2000, 1)
    if np.any(np.abs(mc_short["mean"] - mu_short) > 4 * np.maximum(mc_short["se"], 1e-8)):
        raise AssertionError((mc_short["mean"], mu_short, mc_short["se"]))

    mu_star, _ = optimal_value(model, coef)
    j_star = scalarize(mu_star, weight, low, high)
    # Development: 20 exact steps at lr=1. Extend only if the return does not rise.
    trial = train_exact(model, coef, 1.0, 20, uniform)
    j_final = scalarize(np.array(trial["series"][-1]["mu"]), weight, low, high)
    rationale = "20 exact updates at learning rate 1 improved on the uniform return."
    updates, lr = 20, 1.0
    if j_final <= scalarize(mu0, weight, low, high) + 1e-8:
        trial = train_exact(model, coef, 1.0, 60, uniform)
        updates, lr = 60, 1.0
        rationale = "20 updates at learning rate 1 did not improve, so the frozen run uses 60 updates at the same rate."
        j_final = scalarize(np.array(trial["series"][-1]["mu"]), weight, low, high)
    gains = []
    prev = scalarize(mu0, weight, low, high)
    curve = [{"update": 0, "return": prev, "mu": mu0.tolist()}]
    for row in trial["series"]:
        current = scalarize(np.array(row["mu"]), weight, low, high)
        gains.append(current - prev)
        curve.append({"update": row["update"], "return": current, "gain": current - prev, "grad_norm": row["grad_norm"]})
        prev = current
    positive = [g for g in gains if g > 0]
    gamma_pos = 0.1 * float(np.median(positive)) if positive else 0.0
    # Checkpoints chosen on this development curve only.
    # Update 1 is a single step above uniform. The early checkpoint is update 5,
    # after the return has moved and before the later, best development checkpoint.
    early_index = 5
    later_index = int(np.argmax([row["return"] for row in curve]))
    logits = uniform.copy()
    saved = {}
    for row in trial["series"]:
        grad, _, _ = exact_gradient(model, logits, coef)
        logits = logits + lr * grad
        if row["update"] in (early_index, later_index):
            saved[row["update"]] = logits.copy()
    # Partly trained policy must agree with Monte Carlo.
    mid_logits = saved.get(early_index, logits)
    _, mu_mid, _ = exact_gradient(model, mid_logits, coef)
    mc_mid = monte_carlo(model, softmax(mid_logits), 2000, 2)
    if np.any(np.abs(mc_mid["mean"] - mu_mid) > 4 * mc_mid["se"]):
        raise AssertionError((mc_mid["mean"], mu_mid, mc_mid["se"]))

    sampled = [train_sampled(model, coef, lr, updates, 512, seed, low, high, weight) for seed in (0, 1, 2)]
    out = ROOT / "outputs/horizon_bridge/h3_dst_oracle"
    out.mkdir(parents=True, exist_ok=True)
    for index, arr in saved.items():
        payload = {
            "update": index,
            "logits": arr.tolist(),
            "states": model["states"],
            "horizon": H,
            "normalization_low": low.tolist(),
            "normalization_high": high.tolist(),
            "env_id": "deep-sea-treasure-v0",
            "map_name": model["map_name"],
            "learning_rate": lr,
            "note": "Teacher weight is not stored in this checkpoint.",
        }
        (out / f"checkpoint_update_{index}.json").write_text(json.dumps(payload))
    j_uniform = scalarize(mu0, weight, low, high)
    payload = {
        "horizon": H,
        "horizon_rationale": "The installed convex front's deepest treasure is 19 steps, so H=20 reaches it. H was not extended.",
        "n_states": len(model["states"]),
        "map_name": model["map_name"],
        "spec_max_episode_steps": model["spec_max_steps"],
        "normalization_low": low.tolist(),
        "normalization_high": high.tolist(),
        "finite_difference_gaps": fd_gaps,
        "mc_uniform_gap": (mc_uniform["mean"] - mu0).tolist(),
        "mc_short_gap": (mc_short["mean"] - mu_short).tolist(),
        "mc_mid_gap": (mc_mid["mean"] - mu_mid).tolist(),
        "uniform_mu": mu0.tolist(),
        "uniform_return": j_uniform,
        "short_mu": mu_short.tolist(),
        "short_return": scalarize(mu_short, weight, low, high),
        "mc_uniform_se": mc_uniform["se"].tolist(),
        "mc_short_se": mc_short["se"].tolist(),
        "mc_mid_se": mc_mid["se"].tolist(),
        "live_env_gap": live_gap(model, uniform, mu0),
        "optimal_return": j_star,
        "optimal_gap_uniform": j_star - j_uniform,
        "exact_updates": updates,
        "exact_lr": lr,
        "exact_rationale": rationale,
        "exact_final_return": curve[-1]["return"],
        "exact_improvement": curve[-1]["return"] - j_uniform,
        "exact_gap": j_star - curve[-1]["return"],
        "gamma_pos": gamma_pos,
        "positive_gains": positive,
        "early_update": early_index,
        "later_update": later_index,
        "early_return": curve[early_index]["return"],
        "later_return": curve[later_index]["return"],
        "sampled_final": [row["final_return"] for row in sampled],
        "sampled_improvement": [row["final_return"] - j_uniform for row in sampled],
        "curve": curve,
        "evaluator_weight": weight.tolist(),
    }
    write_summary(out, _clean(payload), started)


def _clean(value):
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


if __name__ == "__main__":
    main()
