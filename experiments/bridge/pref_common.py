"""Shared exact evaluation and converged Bradley-Terry fitting for Work Plan 05.

The learned policy never receives the teacher weight. That weight is used only
when drawing preference labels and when the evaluator reports true return.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar

ROOT = Path(__file__).resolve().parents[2]
TRUE_W = 0.375
BETA = 5.0
GAMMA_POS = 0.0004196952089500494
H = 20
N_ACTIONS = 4
LR = 1.0
CLIP = 1.0


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments/bridge" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=-1, keepdims=True)
    return p


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


class DST:
    """Vectorized finite-horizon model extracted from the installed environment."""

    def __init__(self):
        h3 = _load("h3_dst", "14_dst_exact_tabular.py")
        frozen = json.loads((ROOT / "experiments/horizon_real/E1/Deep-Sea-Treasure/frozen/frozen.json").read_text())
        model = h3.build_model()
        n = len(model["states"])
        reward = np.zeros((n, N_ACTIONS, 2))
        nxt = np.zeros((n, N_ACTIONS), dtype=np.int64)
        done = np.zeros((n, N_ACTIONS), dtype=bool)
        for s in range(n):
            for a in range(N_ACTIONS):
                item = model["trans"][s][a]
                reward[s, a] = item["reward"]
                done[s, a] = item["done"]
                nxt[s, a] = 0 if item["done"] else item["next"]
        self.h3 = h3
        self.model = model
        self.n = n
        self.reward = reward
        self.nxt = nxt
        self.done = done
        self.start = model["start"]
        self.index = model["index"]
        self.low = np.array(frozen["low"], dtype=np.float64)
        self.high = np.array(frozen["high"], dtype=np.float64)
        self.scale = self.high - self.low
        self.states = model["states"]

    def coef(self, w: float) -> np.ndarray:
        return np.array([w, 1.0 - w], dtype=np.float64) / self.scale

    def scalarize(self, mu: np.ndarray, w: float) -> float:
        weight = np.array([w, 1.0 - w])
        return float(weight @ ((mu - self.low) / self.scale))

    def true_return(self, mu: np.ndarray) -> float:
        return self.scalarize(mu, TRUE_W)

    def values(self, pi: np.ndarray) -> np.ndarray:
        value = np.zeros((self.n, 2))
        done = self.done
        nxt = self.nxt
        reward = self.reward
        for _ in range(H):
            future = value[nxt] * (~done)[..., None]
            q = reward + future
            value = np.sum(pi[..., None] * q, axis=1)
        return value[self.start].copy()

    def backward(self, pi: np.ndarray):
        q = np.zeros((H, self.n, N_ACTIONS, 2))
        value = np.zeros((self.n, 2))
        done = self.done
        nxt = self.nxt
        reward = self.reward
        for t in range(H - 1, -1, -1):
            future = value[nxt] * (~done)[..., None]
            q[t] = reward + future
            value = np.sum(pi[..., None] * q[t], axis=1)
        occ = np.zeros((H, self.n))
        occ[0, self.start] = 1.0
        for t in range(H - 1):
            mass = occ[t][:, None] * pi * (~done)
            np.add.at(occ[t + 1], nxt, mass)
        return q, occ

    def gradient(self, pi: np.ndarray, w: float, cache=None) -> np.ndarray:
        q, occ = self.backward(pi) if cache is None else cache
        coef = self.coef(w)
        qs = q @ coef
        vs = np.sum(pi * qs, axis=-1)
        adv = qs - vs[..., None]
        grad = np.sum(occ[:, :, None] * pi[None] * adv, axis=0)
        return grad

    def mu(self, logits: np.ndarray) -> np.ndarray:
        return self.values(softmax(logits))

    def exact_step(self, logits: np.ndarray, w: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pi = softmax(logits)
        mu = self.values(pi)
        grad = self.gradient(pi, w)
        return logits + LR * grad, mu, grad


def gain(env: DST, mu: np.ndarray, mu2: np.ndarray, w: float) -> float:
    return env.scalarize(mu2, w) - env.scalarize(mu, w)


def worst_gain(env: DST, mu: np.ndarray, mu2: np.ndarray, bounds: tuple[float, float]) -> float:
    return min(gain(env, mu, mu2, bounds[0]), gain(env, mu, mu2, bounds[1]))


def best_phi(env: DST, logits: np.ndarray, bounds: tuple[float, float], window: tuple[float, float]) -> tuple[float, float]:
    """Continuous one-dimensional search of the worst-case exact one-step gain.

    The policy gradient at a fixed policy is affine in the reward weight, so
    every candidate reuses one backward pass. The next policy's value is not
    affine, so each candidate still gets its own evaluation.
    """
    lo, hi = float(window[0]), float(window[1])
    pi = softmax(logits)
    mu = env.values(pi)
    cache = env.backward(pi)

    def objective(phi: float) -> float:
        grad = env.gradient(pi, float(phi), cache=cache)
        return worst_gain(env, mu, env.mu(logits + LR * grad), bounds)

    if hi - lo < 1e-12:
        return lo, float(objective(lo))
    seeds = np.linspace(lo, hi, 9)
    best = float(max(seeds, key=objective))
    polished = minimize_scalar(lambda x: -objective(float(x)), bounds=(lo, hi), method="bounded", options={"xatol": 1e-6})
    if objective(float(polished.x)) >= objective(best):
        best = float(polished.x)
    return best, float(objective(best))


def pointwise(env: DST, logits: np.ndarray, bounds: tuple[float, float]) -> float:
    _, value = best_phi(env, logits, bounds, (0.0, 1.0))
    return value


def select_phi(env: DST, logits: np.ndarray, bounds: tuple[float, float], window: tuple[float, float], phi_old: float | None = None) -> tuple[float, float, float]:
    """Shared one-step controller A_U.

    Returns the chosen phi, its worst-case exact gain over ``bounds``, and the
    exact true-preference gain of that same population step. Equal worst-case
    values break toward the current phi, then toward the smaller phi.
    """
    lo, hi = float(window[0]), float(window[1])
    pi = softmax(logits)
    mu = env.values(pi)
    cache = env.backward(pi)
    anchor = float(phi_old) if phi_old is not None else 0.5 * (lo + hi)

    def evaluate(phi: float) -> tuple[float, float]:
        grad = env.gradient(pi, float(phi), cache=cache)
        mu2 = env.mu(logits + LR * grad)
        worst = worst_gain(env, mu, mu2, bounds)
        true = gain(env, mu, mu2, TRUE_W)
        return worst, true

    if hi - lo < 1e-12:
        worst, true = evaluate(lo)
        return lo, worst, true
    grid = np.linspace(lo, hi, 65)
    scored = [(float(phi),) + evaluate(float(phi)) for phi in grid]
    # Local polish inside the best bracket. The worst-case objective is not concave.
    best_index = max(range(len(scored)), key=lambda i: (scored[i][1], -abs(scored[i][0] - anchor), -scored[i][0]))
    left = scored[max(0, best_index - 1)][0]
    right = scored[min(len(scored) - 1, best_index + 1)][0]
    if right - left > 1e-12:
        polished = minimize_scalar(lambda x: -evaluate(float(x))[0], bounds=(left, right), method="bounded", options={"xatol": 1e-7})
        scored.append((float(polished.x),) + evaluate(float(polished.x)))
    chosen = max(scored, key=lambda row: (row[1], -abs(row[0] - anchor), -row[0]))
    return chosen


def audit_select(env: DST, logits: np.ndarray, bounds: tuple[float, float], window: tuple[float, float], phi_old: float, picks: int = 201) -> dict:
    """Independent dense grid. A positive violation means a feasible phi beat the selected one."""
    chosen, worst, true = select_phi(env, logits, bounds, window, phi_old)
    lo, hi = float(window[0]), float(window[1])
    pi = softmax(logits)
    mu = env.values(pi)
    cache = env.backward(pi)
    best_other = -1e100
    best_phi = chosen
    for phi in np.linspace(lo, hi, picks):
        grad = env.gradient(pi, float(phi), cache=cache)
        mu2 = env.mu(logits + LR * grad)
        value = worst_gain(env, mu, mu2, bounds)
        if value > best_other:
            best_other, best_phi = value, float(phi)
    return {
        "phi": chosen,
        "worst": worst,
        "true_gain": true,
        "dense_phi": best_phi,
        "dense_worst": best_other,
        "violation": float(best_other - worst),
    }


def outcome_distribution(env: DST, logits: np.ndarray) -> dict:
    """Exact probability of each terminating treasure, plus truncation at the horizon."""
    pi = softmax(logits)
    mass = np.zeros(env.n)
    mass[env.start] = 1.0
    bins: dict[str, float] = {}
    lengths = []
    for t in range(H):
        step = mass[:, None] * pi
        done = env.done
        arriving = step * done
        if np.any(arriving):
            for state, action in zip(*np.nonzero(arriving > 1e-16)):
                treasure = float(env.reward[state, action, 0])
                key = f"{treasure:.1f}"
                bins[key] = bins.get(key, 0.0) + float(arriving[state, action])
                lengths.append((t + 1, float(arriving[state, action])))
        nxt_mass = np.zeros(env.n)
        np.add.at(nxt_mass, env.nxt, step * (~done))
        mass = nxt_mass
    bins["truncated"] = float(mass.sum())
    expected_length = sum(length * prob for length, prob in lengths) + H * bins["truncated"]
    return {"bins": bins, "expected_length": float(expected_length), "truncation": bins["truncated"]}


def nll(w: float, c: np.ndarray, d: np.ndarray, y: np.ndarray) -> float:
    eta = c + float(w) * d
    return float(np.sum(np.logaddexp(0.0, eta) - y * eta))


def score(w: float, c: np.ndarray, d: np.ndarray, y: np.ndarray) -> float:
    return float(np.sum(d * (y - sigmoid(c + float(w) * d))))


def nll_grad(w: float, c: np.ndarray, d: np.ndarray, y: np.ndarray) -> float:
    return -score(w, c, d, y)


def identifiable(d: np.ndarray) -> bool:
    return float(np.sum(d * d)) > 1e-12


def fit_mle(c: np.ndarray, d: np.ndarray, y: np.ndarray) -> float:
    if not identifiable(d):
        return 0.5
    fit = minimize_scalar(lambda w: nll(float(w), c, d, y), bounds=(0.0, 1.0), method="bounded", options={"xatol": 1e-14})
    return float(fit.x)


def fit_newton(c: np.ndarray, d: np.ndarray, y: np.ndarray, w0: float = 0.5) -> float:
    if not identifiable(d):
        return 0.5
    w = float(np.clip(w0, 0.0, 1.0))
    for _ in range(40):
        eta = c + w * d
        p = sigmoid(eta)
        grad = float(np.sum(d * (p - y)))
        hess = float(np.sum((d * d) * p * (1.0 - p)))
        if hess < 1e-14:
            break
        step = grad / hess
        # Damped projection onto [0, 1].
        alpha = 1.0
        base = nll(w, c, d, y)
        moved = w
        for _trial in range(12):
            cand = float(np.clip(w - alpha * step, 0.0, 1.0))
            if nll(cand, c, d, y) <= base + 1e-12:
                moved = cand
                break
            alpha *= 0.5
        if abs(moved - w) < 1e-12:
            w = moved
            break
        w = moved
    return w


def kkt_residual(w: float, c: np.ndarray, d: np.ndarray, y: np.ndarray) -> float:
    grad = nll_grad(w, c, d, y)
    scale = 1.0 + float(np.sum(np.abs(d)))
    if w <= 1e-8:
        return float(max(-grad, 0.0) / scale)
    if w >= 1.0 - 1e-8:
        return float(max(grad, 0.0) / scale)
    return abs(grad) / scale


def fit_steps(c: np.ndarray, d: np.ndarray, y: np.ndarray, steps: int, w0: float = 0.5) -> float:
    """Historical under-damped mean gradient. Diagnostic only."""
    w = w0
    if len(d) == 0:
        return w
    for _ in range(steps):
        grad = float(np.mean(d * (y - sigmoid(c + w * d))))
        w = float(np.clip(w + grad, 0.0, 1.0))
    return w


def confidence_interval(c: np.ndarray, d: np.ndarray, y: np.ndarray, alpha: float = 0.05) -> tuple[float, float] | None:
    if not identifiable(d):
        return (0.0, 1.0)
    limit = float(np.sqrt(0.5 * np.log(2.0 / alpha) * np.sum(d * d)))

    def lower() -> float | None:
        if score(0.0, c, d, y) <= limit:
            return 0.0
        if score(1.0, c, d, y) > limit:
            return None
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if score(mid, c, d, y) > limit:
                lo = mid
            else:
                hi = mid
        return hi

    def upper() -> float | None:
        if score(1.0, c, d, y) >= -limit:
            return 1.0
        if score(0.0, c, d, y) < -limit:
            return None
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if score(mid, c, d, y) < -limit:
                hi = mid
            else:
                lo = mid
        return lo

    lo, hi = lower(), upper()
    if lo is None or hi is None or lo > hi + 1e-8:
        return None
    return (float(lo), float(hi))


def intersect(intervals: list[tuple[float, float] | None]) -> tuple[float, float] | None:
    lo, hi = 0.0, 1.0
    for item in intervals:
        if item is None:
            return None
        lo = max(lo, item[0])
        hi = min(hi, item[1])
        if lo > hi + 1e-10:
            return None
    return (lo, hi)


def pair_terms(features: np.ndarray, left: np.ndarray, right: np.ndarray, beta: float = BETA):
    diff = features[left] - features[right]
    c = beta * diff[:, 1]
    d = beta * (diff[:, 0] - diff[:, 1])
    return c, d


def fisher_at(c: np.ndarray, d: np.ndarray, w: float) -> np.ndarray:
    p = sigmoid(c + w * d)
    return (d * d) * p * (1.0 - p)


def label_bits(c: np.ndarray, d: np.ndarray, rng: np.random.Generator, w: float = TRUE_W) -> np.ndarray:
    return (rng.random(len(d)) < sigmoid(c + w * d)).astype(np.float64)


def heldout_scores(c: np.ndarray, d: np.ndarray, y: np.ndarray, w: float) -> dict:
    p = sigmoid(c + w * d)
    eta = c + w * d
    loss = float(np.mean(np.logaddexp(0.0, eta) - y * eta))
    brier = float(np.mean((p - y) ** 2))
    accuracy = float(np.mean((p >= 0.5) == (y >= 0.5)))
    return {"nll": loss, "brier": brier, "accuracy": accuracy, "abs_error": abs(w - TRUE_W)}


def make_pairs(n_traj: int, n_pairs: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    left = rng.integers(0, n_traj, size=n_pairs)
    right = rng.integers(0, n_traj, size=n_pairs)
    clash = left == right
    right[clash] = (right[clash] + 1) % n_traj
    # Fixed orientation: smaller index is the first argument when the coin is 0.
    flip = rng.random(n_pairs) < 0.5
    a = np.where(flip, right, left)
    b = np.where(flip, left, right)
    return a.astype(np.int64), b.astype(np.int64)


def live_batch(env, dst: DST, logits: np.ndarray, n: int, rng: np.random.Generator):
    """Collect complete episodes by calling the installed Gym reset and step."""
    probs = softmax(logits)
    episodes = []
    gym_steps = 0
    for _ in range(n):
        obs, _ = env.reset(seed=int(rng.integers(1, 2**31 - 1)))
        steps = []
        total = np.zeros(2)
        for _t in range(H):
            key = (int(obs[0]), int(obs[1]))
            state = dst.index[key]
            action = int(rng.choice(N_ACTIONS, p=probs[state]))
            obs, reward, terminated, truncated, _ = env.step(action)
            total += np.asarray(reward, dtype=np.float64).reshape(-1)[:2]
            steps.append((state, action))
            gym_steps += 1
            if terminated or truncated:
                break
        episodes.append({"steps": steps, "ret": total})
    return episodes, gym_steps


def reinforce(logits: np.ndarray, episodes, weight: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    probs = softmax(logits)
    feats = np.stack([((ep["ret"] - low) / (high - low)) for ep in episodes])
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


def scripted_treasure(dst: DST) -> np.ndarray:
    """Prefer down, then right. This is a fixed explorer, not the teacher policy."""
    logits = np.zeros((dst.n, N_ACTIONS))
    for s, (r, c) in enumerate(dst.states):
        logits[s, 1] = 4.0
        logits[s, 3] = 2.0
    return logits


def verify_against_h3(dst: DST) -> dict:
    h3 = dst.h3
    uniform = np.zeros((dst.n, N_ACTIONS))
    coef = dst.coef(TRUE_W)
    g_ref, mu_ref, _ = h3.exact_gradient(dst.model, uniform, coef)
    mu = dst.mu(uniform)
    g = dst.gradient(softmax(uniform), TRUE_W)
    gap_mu = float(np.max(np.abs(mu - mu_ref)))
    gap_g = float(np.max(np.abs(g - g_ref)))
    if gap_mu > 1e-8 or gap_g > 1e-8:
        raise AssertionError((gap_mu, gap_g))
    ckpt = json.loads((ROOT / "outputs/horizon_bridge/h3_dst_oracle/checkpoint_update_20.json").read_text())
    logits = np.array(ckpt["logits"], dtype=np.float64)
    g_ref, mu_ref, _ = h3.exact_gradient(dst.model, logits, coef)
    gap_mu2 = float(np.max(np.abs(dst.mu(logits) - mu_ref)))
    gap_g2 = float(np.max(np.abs(dst.gradient(softmax(logits), TRUE_W) - g_ref)))
    if gap_mu2 > 1e-8 or gap_g2 > 1e-8:
        raise AssertionError((gap_mu2, gap_g2))
    return {"uniform_mu_gap": gap_mu, "uniform_grad_gap": gap_g, "checkpoint_mu_gap": gap_mu2, "checkpoint_grad_gap": gap_g2}


def build_source(dst: DST, gym_env, seed: int, regime: str) -> dict:
    """32 source comparisons. Weak pairs are the lowest-|d| pairs in a fresh pool."""
    rng = np.random.default_rng(100_000 + seed)
    episodes, steps = live_batch(gym_env, dst, np.zeros((dst.n, N_ACTIONS)), 48, rng)
    extra, steps2 = live_batch(gym_env, dst, scripted_treasure(dst), 16, rng)
    episodes.extend(extra)
    feats = np.stack([(ep["ret"] - dst.low) / dst.scale for ep in episodes])
    left, right = make_pairs(len(feats), 128, rng)
    c, d = pair_terms(feats, left, right)
    if regime == "weak":
        chosen = np.argsort(np.abs(d))[:32]
    elif regime == "mixed":
        chosen = rng.choice(len(d), size=32, replace=False)
    else:
        raise ValueError(regime)
    y = label_bits(c[chosen], d[chosen], rng)
    interval = confidence_interval(c[chosen], d[chosen], y, alpha=0.01)
    return {
        "c": c[chosen],
        "d": d[chosen],
        "y": y,
        "interval": interval,
        "w_hat": fit_mle(c[chosen], d[chosen], y),
        "median_fisher": float(np.median(fisher_at(c[chosen], d[chosen], 0.5))),
        "gym_steps": steps + steps2,
    }


def build_feedback(dst: DST, gym_env, logits: np.ndarray, seed: int, method: str, event: int, w_hat: float, n_labels: int) -> dict:
    rng = np.random.default_rng(300_000 + 1000 * seed + 20 * event + ord(method[0]))
    current, s1 = live_batch(gym_env, dst, logits, 64 if n_labels <= 32 else 128, rng)
    uniform, s2 = live_batch(gym_env, dst, np.zeros((dst.n, N_ACTIONS)), 32 if n_labels <= 32 else 64, rng)
    explorer, s3 = live_batch(gym_env, dst, scripted_treasure(dst), 32 if n_labels <= 32 else 64, rng)
    episodes = current + uniform + explorer
    feats = np.stack([(ep["ret"] - dst.low) / dst.scale for ep in episodes])
    n_pairs = 256 if n_labels <= 32 else 512
    left, right = make_pairs(len(feats), n_pairs, rng)
    c, d = pair_terms(feats, left, right)
    if method == "U":
        chosen = rng.choice(len(d), size=n_labels, replace=False)
    else:
        rank = fisher_at(c, d, w_hat)
        chosen = np.argsort(-rank)[:n_labels]
    # Labels are drawn only after the indices above are fixed.
    y = label_bits(c[chosen], d[chosen], np.random.default_rng(400_000 + 1000 * seed + 20 * event + ord(method[0])))
    interval = confidence_interval(c[chosen], d[chosen], y, alpha=0.01)
    return {
        "c": c[chosen],
        "d": d[chosen],
        "y": y,
        "interval": interval,
        "median_fisher": float(np.median(fisher_at(c[chosen], d[chosen], w_hat))),
        "gym_steps": s1 + s2 + s3,
        "pool_pairs": int(n_pairs),
    }


def run_online(
    dst: DST,
    gym_env,
    *,
    seed: int,
    method: str,
    regime: str,
    updates: int,
    episodes: int,
    budget: float,
    feedback_at: tuple[int, ...] = (20, 40, 60, 80),
    labels_per_event: int = 32,
    label_cap: int = 128,
    held=None,
    save_at: tuple[int, ...] = (),
    trace_info: bool = False,
) -> dict:
    source = build_source(dst, gym_env, seed, regime)
    batches = [(source["c"], source["d"], source["y"], source["interval"])]
    logits = np.zeros((dst.n, N_ACTIONS))
    phi = TRUE_W if method == "O" else 0.5
    w_hat = source["w_hat"]
    acquired = 0
    gym_steps = source["gym_steps"]
    curve = []
    mu = dst.mu(logits)
    ret = dst.true_return(mu)
    curve.append({"update": 0, "return": ret, "phi": phi, "gain": 0.0, "source_certificate": None, "updated_certificate": None})
    policy_rng = np.random.default_rng(200_000 + 1000 * seed + ord(method[0]))
    feedback_rows = []
    snapshots = {}
    trace = {}
    if 0 in save_at:
        snapshots[0] = logits.copy()
        if trace_info:
            trace[0] = {"phi": phi, "w_hat": w_hat, "lo": None if source["interval"] is None else source["interval"][0], "hi": None if source["interval"] is None else source["interval"][1], "mu": mu.tolist()}
    for k in range(1, updates + 1):
        if method != "O" and method != "S" and k in feedback_at and acquired < label_cap:
            take = min(labels_per_event, label_cap - acquired)
            event = build_feedback(dst, gym_env, logits, seed, "I" if method == "M" else method, k, w_hat, take)
            gym_steps += event["gym_steps"]
            batches.append((event["c"], event["d"], event["y"], event["interval"]))
            acquired += take
            c = np.concatenate([item[0] for item in batches])
            d = np.concatenate([item[1] for item in batches])
            y = np.concatenate([item[2] for item in batches])
            w_hat = fit_mle(c, d, y)
            feedback_rows.append({"update": k, "labels": acquired, "fisher": event["median_fisher"], "w_hat": w_hat, "pool_pairs": event["pool_pairs"]})
        usable = intersect([item[3] for item in batches]) if method != "O" else (TRUE_W, TRUE_W)
        if method == "O":
            chosen = TRUE_W
        elif method == "M":
            chosen = float(np.clip(w_hat, phi - budget, phi + budget))
        elif usable is None:
            chosen = phi
        else:
            window = (max(0.0, phi - budget), min(1.0, phi + budget))
            chosen, _planned = best_phi(dst, logits, usable, window)
        weight = np.array([chosen, 1.0 - chosen])
        rollouts, steps = live_batch(gym_env, dst, logits, episodes, policy_rng)
        gym_steps += steps
        nxt = reinforce(logits, rollouts, weight, dst.low, dst.high)
        mu2 = dst.mu(nxt)
        ret2 = dst.true_return(mu2)
        source_bounds = batches[0][3]
        source_cert = None if source_bounds is None else worst_gain(dst, mu, mu2, source_bounds)
        updated_cert = None if usable is None or method == "O" else worst_gain(dst, mu, mu2, usable)
        bounded_value = None
        full_value = None
        if k in (1, 20, 40, 80, 160) and usable is not None and method != "O":
            _unused, bounded_value = best_phi(dst, logits, usable, (max(0.0, phi - budget), min(1.0, phi + budget)))
            full_value = pointwise(dst, logits, usable)
        curve.append({
            "update": k,
            "return": ret2,
            "phi": chosen,
            "gain": ret2 - ret,
            "source_certificate": source_cert,
            "updated_certificate": updated_cert,
            "bounded_value": bounded_value,
            "pointwise_value": full_value,
            "grad_norm": float(np.linalg.norm(nxt - logits)),
            "w_hat": w_hat,
            "width": None if usable is None else float(usable[1] - usable[0]),
        })
        logits, phi, mu, ret = nxt, chosen, mu2, ret2
        if k in save_at:
            snapshots[k] = logits.copy()
            if trace_info:
                source_bounds = batches[0][3]
                trace[k] = {
                    "phi": float(chosen),
                    "w_hat": float(w_hat),
                    "lo": None if usable is None else float(usable[0]),
                    "hi": None if usable is None else float(usable[1]),
                    "source_lo": None if source_bounds is None else float(source_bounds[0]),
                    "source_hi": None if source_bounds is None else float(source_bounds[1]),
                    "mu": mu2.tolist(),
                }
    held_nll = None
    if held is not None and method != "O":
        held_nll = heldout_scores(held[0], held[1], held[2], w_hat)["nll"]
    returns = np.array([row["return"] for row in curve], dtype=np.float64)
    return {
        "seed": seed,
        "method": method,
        "regime": regime,
        "episodes": episodes,
        "updates": updates,
        "budget": budget,
        "labels": 32 + acquired,
        "new_labels": acquired,
        "final_return": float(returns[-1]),
        "return_0": float(returns[0]),
        "return_40": float(returns[min(40, updates)]),
        "return_80": float(returns[min(80, updates)]),
        "return_160": float(returns[min(160, updates)]),
        "auc": float(np.trapezoid(returns, dx=1.0)),
        "final_phi": float(phi),
        "final_w": float(w_hat),
        "final_width": curve[-1]["width"],
        "heldout_nll": held_nll,
        "gym_steps": gym_steps,
        "source_fisher": source["median_fisher"],
        "source_width": None if source["interval"] is None else float(source["interval"][1] - source["interval"][0]),
        "feedback": feedback_rows,
        "returns": returns.tolist(),
        "phis": [row["phi"] for row in curve],
        "gains": [row["gain"] for row in curve],
        "source_certificates": [row["source_certificate"] for row in curve],
        "updated_certificates": [row["updated_certificate"] for row in curve],
        "logits": logits.tolist(),
        "snapshots": {str(k): v.tolist() for k, v in snapshots.items()},
        "trace": {str(k): v for k, v in trace.items()},
    }


def frozen_heldout(dst: DST, gym_env):
    """512 held-out comparisons. Pair identities and labels are fixed before any learner run."""
    seeds = {"random_policies": 8200, "uniform": 8210, "treasure": 8211, "ckpt5": 8212, "ckpt20": 8213}
    for i in range(4):
        seeds[f"random{i}"] = 8220 + i
    policies = {"uniform": np.zeros((dst.n, N_ACTIONS)), "treasure": scripted_treasure(dst)}
    for update in (5, 20):
        payload = json.loads((ROOT / f"outputs/horizon_bridge/h3_dst_oracle/checkpoint_update_{update}.json").read_text())
        policies[f"ckpt{update}"] = np.array(payload["logits"], dtype=np.float64)
    rng_pol = np.random.default_rng(seeds["random_policies"])
    for i in range(4):
        policies[f"random{i}"] = rng_pol.normal(size=(dst.n, N_ACTIONS))
    feats = []
    for name, logits in policies.items():
        episodes, _steps = live_batch(gym_env, dst, logits, 64, np.random.default_rng(seeds[name]))
        feats.extend([ep["ret"] for ep in episodes])
    raw = (np.stack(feats) - dst.low) / dst.scale
    left, right = make_pairs(len(raw), 512, np.random.default_rng(8250))
    c, d = pair_terms(raw, left, right)
    y = label_bits(c, d, np.random.default_rng(8260))
    return c, d, y


def jsonable(value):
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value
