#!/usr/bin/env python3
"""P5: Resource-Gathering with a two-dimensional preference parameter."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.bridge.common import write_summary
from experiments.bridge.pref_common import jsonable, softmax

ROOT = Path(__file__).resolve().parents[2]
H = 40
N_ACTIONS = 4
BETA = 1.0
W_STAR = np.array([0.2, 0.2, 0.6])
LOW = np.array([-1.0, 0.0, 0.0])
HIGH = np.array([0.0, 1.0, 1.0])
SCALE = HIGH - LOW
LR = 1.0
CLIP = 1.0
BUDGET = 0.02


def index(x, y, gold, gem) -> int:
    return ((int(x) * 5 + int(y)) * 2 + int(gold)) * 2 + int(gem)


def decode(state: int):
    gem = state % 2
    gold = (state // 2) % 2
    y = (state // 4) % 5
    x = state // 20
    return x, y, gold, gem


class RG:
    def __init__(self):
        import mo_gymnasium as mo_gym
        env = mo_gym.make("resource-gathering-v0")
        raw = env.unwrapped
        self.map = np.array(raw.map)
        self.spec_cap = None if env.spec is None else env.spec.max_episode_steps
        env.close()
        n = 100
        # Two branches cover the deterministic move and the enemy coin-flip.
        self.p = np.zeros((n, N_ACTIONS, 2))
        self.reward = np.zeros((n, N_ACTIONS, 2, 3))
        self.nxt = np.zeros((n, N_ACTIONS, 2), dtype=np.int64)
        self.done = np.ones((n, N_ACTIONS, 2), dtype=bool)
        dirs = ((-1, 0), (1, 0), (0, -1), (0, 1))
        for state in range(n):
            x, y, gold, gem = decode(state)
            for action, (dx, dy) in enumerate(dirs):
                nx, ny = x + dx, y + dy
                if not (0 <= nx < 5 and 0 <= ny < 5):
                    nx, ny = x, y
                cell = str(self.map[nx, ny])
                ng, nm = gold, gem
                if cell == "R1":
                    ng = 1
                elif cell == "R2":
                    nm = 1
                nxt = index(nx, ny, ng, nm)
                if cell in ("E1", "E2"):
                    self.p[state, action] = (0.9, 0.1)
                    self.nxt[state, action, 0] = nxt
                    self.done[state, action, 0] = False
                    self.reward[state, action, 1] = (-1.0, 0.0, 0.0)
                    self.done[state, action, 1] = True
                elif cell == "H":
                    self.p[state, action, 0] = 1.0
                    self.reward[state, action, 0] = (0.0, ng, nm)
                    self.done[state, action, 0] = True
                else:
                    self.p[state, action, 0] = 1.0
                    self.nxt[state, action, 0] = nxt
                    self.done[state, action, 0] = False
        self.start = index(4, 2, 0, 0)
        self.n = n

    def values(self, pi: np.ndarray) -> np.ndarray:
        value = np.zeros((self.n, 3))
        for _ in range(H):
            future = value[self.nxt] * (~self.done)[..., None]
            branch = self.p[..., None] * (self.reward + future)
            q = branch.sum(axis=2)
            value = np.sum(pi[..., None] * q, axis=1)
        return value[self.start].copy()

    def gradient(self, pi: np.ndarray, weight: np.ndarray) -> np.ndarray:
        coef = weight / SCALE
        q_vec = np.zeros((H, self.n, N_ACTIONS, 3))
        value = np.zeros((self.n, 3))
        for t in range(H - 1, -1, -1):
            future = value[self.nxt] * (~self.done)[..., None]
            branch = self.p[..., None] * (self.reward + future)
            q_vec[t] = branch.sum(axis=2)
            value = np.sum(pi[..., None] * q_vec[t], axis=1)
        occ = np.zeros((H, self.n))
        occ[0, self.start] = 1.0
        for t in range(H - 1):
            mass = occ[t][:, None, None] * pi[:, :, None] * self.p * (~self.done)
            np.add.at(occ[t + 1], self.nxt, mass)
        qs = q_vec @ coef
        vs = np.sum(pi * qs, axis=-1)
        adv = qs - vs[..., None]
        return np.sum(occ[:, :, None] * pi[None] * adv, axis=0)


def scalarize(mu: np.ndarray, weight: np.ndarray) -> float:
    return float(weight @ ((mu - LOW) / SCALE))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def design(diff: np.ndarray):
    # logit = beta * (D3 + w1 (D1-D3) + w2 (D2-D3))
    c = BETA * diff[:, 2]
    d = np.column_stack([BETA * (diff[:, 0] - diff[:, 2]), BETA * (diff[:, 1] - diff[:, 2])])
    return c, d


def nll(w, c, d, y) -> float:
    eta = c + d @ w
    return float(np.sum(np.logaddexp(0.0, eta) - y * eta))


def fit_simplex(c, d, y) -> tuple[np.ndarray, np.ndarray, float]:
    if len(y) == 0 or float(np.sum(d * d)) < 1e-12:
        return np.array([1.0 / 3.0, 1.0 / 3.0]), np.array([1.0 / 3.0, 1.0 / 3.0]), 0.0

    def objective(z):
        return nll(np.clip(z, 0.0, 1.0), c, d, y)

    cons = {"type": "ineq", "fun": lambda z: 1.0 - z[0] - z[1]}
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    first = minimize(objective, np.array([0.3, 0.3]), method="SLSQP", bounds=bounds, constraints=cons, tol=1e-12)
    def soft(z):
        p = np.exp(z - np.max(z))
        p /= np.sum(p)
        return nll(p[:2], c, d, y)
    second = minimize(soft, np.zeros(3), method="BFGS")
    p = np.exp(second.x - np.max(second.x))
    p /= np.sum(p)
    return first.x, p[:2], abs(nll(first.x, c, d, y) - nll(p[:2], c, d, y))


def score_region_ok(c, d, y, alpha=0.05) -> tuple[bool, float]:
    """Check a continuous solver against a simplex grid. This is a diagnostic, not a formal grid certificate."""
    if float(np.sum(d * d)) < 1e-12:
        return False, 1.0
    limits = np.sqrt(0.5 * np.log(4.0 / alpha) * np.sum(d * d, axis=0))

    def feasible(w) -> bool:
        p = sigmoid(c + d @ w)
        score = d.T @ (y - p)
        return bool(np.all(np.abs(score) <= limits + 1e-8))

    direction = np.array([0.2, -0.5])
    cons = [
        {"type": "ineq", "fun": lambda z: 1.0 - z[0] - z[1]},
        {"type": "ineq", "fun": lambda z, sign=1, j=0: sign * (limits[j] - _score(z, c, d, y)[j])},
    ]
    # Rebuild constraints explicitly.
    def constraints():
        out = [{"type": "ineq", "fun": lambda z: 1.0 - float(z[0] + z[1])}]
        for j in range(2):
            out.append({"type": "ineq", "fun": lambda z, j=j: float(limits[j] - _score(z, c, d, y)[j])})
            out.append({"type": "ineq", "fun": lambda z, j=j: float(limits[j] + _score(z, c, d, y)[j])})
        return out

    starts = [np.array(p) for p in ((0.2, 0.2), (0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.3, 0.3))]
    best = None
    for start in starts:
        fit = minimize(lambda z: float(direction @ z), start, method="SLSQP", bounds=[(0, 1), (0, 1)], constraints=constraints(), tol=1e-10)
        if fit.success and (best is None or fit.fun < best):
            best = float(fit.fun)
    grid_vals = []
    for a in np.linspace(0, 1, 21):
        for b in np.linspace(0, 1 - a, 21):
            w = np.array([a, b])
            if feasible(w):
                grid_vals.append(float(direction @ w))
    if best is None or not grid_vals:
        return False, 1.0
    gap = abs(best - min(grid_vals))
    return gap < 0.02, gap


def _score(z, c, d, y):
    p = sigmoid(c + d @ np.asarray(z, dtype=float))
    return d.T @ (y - p)


def live_batch(env, model: RG, logits, n, rng):
    probs = softmax(logits)
    episodes = []
    steps = 0
    for _ in range(n):
        obs, _info = env.reset(seed=int(rng.integers(1, 2**31 - 1)))
        total = np.zeros(3)
        trace = []
        for _t in range(H):
            state = index(obs[0], obs[1], obs[2], obs[3])
            action = int(rng.choice(N_ACTIONS, p=probs[state]))
            obs, reward, terminated, truncated, _info = env.step(action)
            total += np.asarray(reward, dtype=np.float64).reshape(-1)[:3]
            trace.append((state, action))
            steps += 1
            if terminated or truncated:
                break
        episodes.append({"steps": trace, "ret": total})
    return episodes, steps


def reinforce(logits, episodes, weight):
    probs = softmax(logits)
    feats = np.stack([((ep["ret"] - LOW) / SCALE) for ep in episodes])
    returns = feats @ weight
    baseline = float(returns.mean())
    grad = np.zeros_like(logits)
    for ep, value in zip(episodes, returns):
        advantage = float(value - baseline)
        for state, action in ep["steps"]:
            one = np.zeros(N_ACTIONS)
            one[action] = 1.0
            grad[state] += advantage * (one - probs[state])
    step = LR * grad / max(len(episodes), 1)
    norm = float(np.linalg.norm(step))
    if norm > CLIP:
        step *= CLIP / norm
    return logits + step


def project_ball(current, proposal):
    clipped = np.clip(proposal, current - BUDGET, current + BUDGET)
    clipped = np.clip(clipped, 0.0, 1.0)
    if clipped.sum() > 1.0:
        # Scale back toward the current feasible point.
        lo, hi = 0.0, 1.0
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            cand = current + mid * (clipped - current)
            if cand.sum() <= 1.0 and np.all(cand >= -1e-12):
                lo = mid
            else:
                hi = mid
        clipped = np.clip(current + lo * (clipped - current), 0.0, 1.0)
    return clipped


def make_pairs(n, k, rng):
    left = rng.integers(0, n, size=k)
    right = rng.integers(0, n, size=k)
    right = np.where(left == right, (right + 1) % n, right)
    flip = rng.random(k) < 0.5
    return np.where(flip, right, left), np.where(flip, left, right)


def collect_source(env, model, seed, regime="weak"):
    rng = np.random.default_rng(500_000 + seed)
    episodes, steps = live_batch(env, model, np.zeros((model.n, N_ACTIONS)), 64, rng)
    feats = np.stack([((ep["ret"] - LOW) / SCALE) for ep in episodes])
    left, right = make_pairs(len(feats), 128, rng)
    diff = feats[left] - feats[right]
    c, d = design(diff)
    sens = np.sum(d * d, axis=1)
    chosen = np.argsort(sens)[:32] if regime == "weak" else rng.choice(len(d), size=32, replace=False)
    y = (rng.random(32) < sigmoid(c[chosen] + d[chosen] @ W_STAR[:2])).astype(float)
    return c[chosen], d[chosen], y, steps


def collect_feedback(env, model, logits, seed, event, w_hat, rule, n_labels=32):
    rng = np.random.default_rng(600_000 + 1000 * seed + event)
    current, s1 = live_batch(env, model, logits, 48, rng)
    uniform, s2 = live_batch(env, model, np.zeros((model.n, N_ACTIONS)), 16, rng)
    episodes = current + uniform
    feats = np.stack([((ep["ret"] - LOW) / SCALE) for ep in episodes])
    left, right = make_pairs(len(feats), 128, rng)
    c, d = design(feats[left] - feats[right])
    if rule == "uniform":
        chosen = rng.choice(len(d), size=n_labels, replace=False)
    else:
        p = sigmoid(c + d @ w_hat)
        info = np.sum(d * d, axis=1) * p * (1.0 - p)
        chosen = np.argsort(-info)[:n_labels]
    y = (np.random.default_rng(700_000 + 1000 * seed + event).random(n_labels) < sigmoid(c[chosen] + d[chosen] @ W_STAR[:2])).astype(float)
    return c[chosen], d[chosen], y, s1 + s2, float(np.median(np.sum(d[chosen] * d[chosen], axis=1)))


def run(model, env, seed, method, episodes, robust_ok, held=None):
    c0, d0, y0, steps = collect_source(env, model, seed)
    batches = [(c0, d0, y0)]
    fitted, _w_soft, gap = fit_simplex(c0, d0, y0)
    deployed = np.array([1.0 / 3.0, 1.0 / 3.0])
    logits = np.zeros((model.n, N_ACTIONS))
    rng = np.random.default_rng(800_000 + 100 * seed + ord(method[0]))
    curve = [scalarize(model.values(softmax(logits)), W_STAR)]
    mus = [model.values(softmax(logits)).tolist()]
    acquired = 0
    for k in range(1, 101):
        if method in ("fisher", "uniform", "robust") and k in (40, 80) and acquired < 64:
            rule = "uniform" if method == "uniform" else "fisher"
            c, d, y, extra, _fisher = collect_feedback(env, model, logits, seed, k, fitted, rule)
            steps += extra
            batches.append((c, d, y))
            acquired += 32
            c_all = np.concatenate([item[0] for item in batches])
            d_all = np.concatenate([item[1] for item in batches])
            y_all = np.concatenate([item[2] for item in batches])
            fitted, _w2, gap = fit_simplex(c_all, d_all, y_all)
        if method == "oracle":
            chosen = W_STAR[:2].copy()
        elif method == "robust" and robust_ok:
            c_all = np.concatenate([item[0] for item in batches])
            d_all = np.concatenate([item[1] for item in batches])
            y_all = np.concatenate([item[2] for item in batches])
            limits = np.sqrt(0.5 * np.log(4.0 / 0.05) * np.sum(d_all * d_all, axis=0))
            best_w, best_value = deployed.copy(), -1e100
            for da in np.linspace(-BUDGET, BUDGET, 5):
                for db in np.linspace(-BUDGET, BUDGET, 5):
                    cand = project_ball(deployed, deployed + np.array([da, db]))
                    weight = np.array([cand[0], cand[1], 1.0 - cand.sum()])
                    nxt = logits + LR * model.gradient(softmax(logits), weight)
                    mu_now = model.values(softmax(logits))
                    mu_next = model.values(softmax(nxt))
                    worst = _worst((mu_next - mu_now) / SCALE, c_all, d_all, y_all, limits)
                    if worst > best_value:
                        best_w, best_value = cand, worst
            chosen = best_w
        else:
            chosen = project_ball(deployed, fitted)
        weight = W_STAR if method == "oracle" else np.array([chosen[0], chosen[1], 1.0 - chosen.sum()])
        rollouts, extra = live_batch(env, model, logits, episodes, rng)
        steps += extra
        logits = reinforce(logits, rollouts, weight)
        deployed = chosen
        mu = model.values(softmax(logits))
        curve.append(scalarize(mu, W_STAR))
        mus.append(mu.tolist())
    c_all = np.concatenate([item[0] for item in batches])
    d_all = np.concatenate([item[1] for item in batches])
    y_all = np.concatenate([item[2] for item in batches])
    held_c, held_d, held_y = None, None, None
    return {
        "seed": seed,
        "method": method,
        "return_0": curve[0],
        "final_return": curve[-1],
        "curve": curve,
        "mu_0": mus[0],
        "mu_final": mus[-1],
        "final_w": None if method == "oracle" else deployed.tolist(),
        "final_fit": None if method == "oracle" else fitted.tolist(),
        "fit_gap": gap,
        "labels": 0 if method == "oracle" else 32 + acquired,
        "w_error": None if method == "oracle" else float(np.linalg.norm(np.array([fitted[0], fitted[1], 1 - fitted.sum()]) - W_STAR)),
        "heldout_nll": None if held is None or method == "oracle" else nll(fitted, held[0], held[1], held[2]) / max(len(held[2]), 1),
        "gym_steps": steps,
    }


def _worst(delta, c, d, y, limits) -> float:
    # delta is the normalized 3-vector gain. w3 = 1-w1-w2, so the objective is affine in (w1, w2).
    # gain = w·delta = delta3 + (delta1-delta3) w1 + (delta2-delta3) w2
    coef = np.array([delta[0] - delta[2], delta[1] - delta[2]])
    base = float(delta[2])

    def constraints():
        out = [{"type": "ineq", "fun": lambda z: 1.0 - float(z[0] + z[1])}]
        for j in range(2):
            out.append({"type": "ineq", "fun": lambda z, j=j: float(limits[j] - _score(z, c, d, y)[j])})
            out.append({"type": "ineq", "fun": lambda z, j=j: float(limits[j] + _score(z, c, d, y)[j])})
        return out

    best = None
    for start in ((0.2, 0.2), (0.0, 0.0), (0.5, 0.2), (0.2, 0.5)):
        fit = minimize(lambda z: base + float(coef @ z), np.array(start), method="SLSQP", bounds=[(0, 1), (0, 1)], constraints=constraints(), tol=1e-8)
        if fit.success and (best is None or fit.fun < best):
            best = float(fit.fun)
    if best is None:
        return base + float(coef @ np.array([0.2, 0.2]))
    return best


def self_check(model: RG) -> dict:
    import mo_gymnasium as mo_gym
    env = mo_gym.make("resource-gathering-v0")
    raw = env.unwrapped
    env.reset(seed=0)
    deaths = 0
    trials = 2000
    for _ in range(trials):
        raw.current_pos = np.array([1, 1], dtype=np.int32)
        raw.has_gold = 0
        raw.has_gem = 0
        _obs, reward, _term, _trunc, _info = raw.step(3)
        deaths += int(np.asarray(reward).reshape(-1)[0] < -0.5)
    rate = deaths / trials
    if not 0.07 < rate < 0.13:
        raise AssertionError(rate)
    # Down from home stays on H and terminates with a zero resource reward.
    raw.current_pos = np.array([4, 2], dtype=np.int32)
    raw.has_gold = 0
    raw.has_gem = 0
    _obs, reward, terminated, _trunc, _info = raw.step(1)
    reward = np.asarray(reward, dtype=float).reshape(-1)
    if not terminated or np.max(np.abs(reward)) > 1e-8:
        raise AssertionError((terminated, reward))
    pi = softmax(np.zeros((model.n, N_ACTIONS)))
    mu = model.values(pi)
    episodes, _ = live_batch(env, model, np.zeros((model.n, N_ACTIONS)), 2000, np.random.default_rng(1))
    mean = np.mean([ep["ret"] for ep in episodes], axis=0)
    se = np.std([ep["ret"] for ep in episodes], axis=0, ddof=1) / np.sqrt(len(episodes))
    if np.any(np.abs(mean - mu) > 4 * np.maximum(se, 1e-8)):
        raise AssertionError((mean, mu, se))
    env.close()
    return {"enemy_death_rate": rate, "uniform_mu": mu.tolist(), "uniform_gap": (mean - mu).tolist(), "spec_cap": model.spec_cap, "horizon": H}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/horizon_bridge/p5_resource_gathering")
    args = parser.parse_args()
    started = time.perf_counter()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    model = RG()
    check = self_check(model)
    import mo_gymnasium as mo_gym
    env = mo_gym.make("resource-gathering-v0")
    # Gradient-quality pilot at the uniform policy, true preference.
    exact = model.gradient(softmax(np.zeros((model.n, N_ACTIONS))), W_STAR)
    cosines = {}
    for n in (256, 512):
        estimates = []
        for k in range(2):
            episodes, _ = live_batch(env, model, np.zeros((model.n, N_ACTIONS)), n, np.random.default_rng(90 + n + k))
            # Reconstruct the averaged score-function gradient.
            probs = softmax(np.zeros((model.n, N_ACTIONS)))
            feats = np.stack([((ep["ret"] - LOW) / SCALE) for ep in episodes])
            returns = feats @ W_STAR
            baseline = float(returns.mean())
            grad = np.zeros_like(probs)
            for ep, value in zip(episodes, returns):
                advantage = float(value - baseline)
                for state, action in ep["steps"]:
                    one = np.zeros(N_ACTIONS)
                    one[action] = 1.0
                    grad[state] += advantage * (one - probs[state])
            grad /= n
            denom = np.linalg.norm(grad) * np.linalg.norm(exact)
            estimates.append(0.0 if denom < 1e-12 else float(np.sum(grad * exact) / denom))
        cosines[n] = float(np.mean(estimates))
    episodes = 512 if cosines[256] < 0.5 and cosines[512] > cosines[256] + 0.2 else 256
    # One development oracle trajectory of 40 updates decides nothing about preference arms.
    dev_logits = np.zeros((model.n, N_ACTIONS))
    dev_rng = np.random.default_rng(3)
    dev_curve = [scalarize(model.values(softmax(dev_logits)), W_STAR)]
    t1 = time.perf_counter()
    for _ in range(40):
        rollouts, _ = live_batch(env, model, dev_logits, episodes, dev_rng)
        dev_logits = reinforce(dev_logits, rollouts, W_STAR)
        dev_curve.append(scalarize(model.values(softmax(dev_logits)), W_STAR))
    dev_seconds = time.perf_counter() - t1
    # Synthetic region check on a random fixed design, before the policy runs.
    rng = np.random.default_rng(4)
    diff = rng.normal(size=(64, 3))
    c, d = design(diff)
    y = (rng.random(64) < sigmoid(c + d @ W_STAR[:2])).astype(float)
    robust_ok, region_gap = score_region_ok(c, d, y)
    held_eps, _ = live_batch(env, model, np.zeros((model.n, N_ACTIONS)), 160, np.random.default_rng(4242))
    held_feats = np.stack([((ep["ret"] - LOW) / SCALE) for ep in held_eps])
    h_left, h_right = make_pairs(len(held_feats), 256, np.random.default_rng(4243))
    held_c, held_d = design(held_feats[h_left] - held_feats[h_right])
    held_y = (np.random.default_rng(4244).random(len(held_c)) < sigmoid(held_c + held_d @ W_STAR[:2])).astype(float)
    held = (held_c, held_d, held_y)
    methods = ["oracle", "source", "fisher", "uniform"] + (["robust"] if robust_ok else [])
    rows = []
    for seed in range(5):
        for method in methods:
            path = out / f"{method}_seed{seed}.json"
            if path.exists():
                rows.append(json.loads(path.read_text()))
                continue
            row = run(model, env, seed, method, episodes, robust_ok, held)
            path.write_text(json.dumps(jsonable(row)))
            rows.append(row)
            print(method, seed, row["final_return"], flush=True)
    env.close()
    summary = {}
    for method in methods:
        group = [row for row in rows if row["method"] == method]
        final = np.array([row["final_return"] for row in group])
        mu = np.mean([row["mu_final"] for row in group], axis=0)
        mu0 = np.mean([row["mu_0"] for row in group], axis=0)
        summary[method] = {
            "n": len(group),
            "return_0": float(np.mean([row["return_0"] for row in group])),
            "final_mean": float(final.mean()),
            "final_sd": float(final.std(ddof=1)),
            "mu_change": (mu - mu0).tolist(),
            "w_error_mean": None if method == "oracle" else float(np.mean([row["w_error"] for row in group])),
            "heldout_nll_mean": None if method == "oracle" else float(np.mean([row["heldout_nll"] for row in group])),
            "fit_gap_mean": None if method == "oracle" else float(np.mean([row["fit_gap"] for row in group])),
            "labels": group[0]["labels"],
        }
    write_summary(out, jsonable({
        "self_check": check,
        "cosine": cosines,
        "episodes_per_update": episodes,
        "development_oracle_40": {"start": dev_curve[0], "end": dev_curve[-1], "seconds": dev_seconds},
        "robust_region_checked": robust_ok,
        "robust_region_gap": region_gap,
        "robust_status": "numerically checked union-bound region" if robust_ok else "UNRESOLVED",
        "methods": summary,
        "note": "Horizon is the frozen cap 40. The installed wrapper cap is 100 and is not the evaluator.",
    }), started)


if __name__ == "__main__":
    main()
