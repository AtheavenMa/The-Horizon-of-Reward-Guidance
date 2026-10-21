#!/usr/bin/env python3
"""R4B: conservative outer cover for the two-dimensional BTL score region.

A local solver point is an upper bound on the worst-case minimum. The triangle
cover is a lower bound. The hidden teacher weight is never a fallback.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.bridge.common import write_summary
from experiments.bridge.pref_common import jsonable


def load_rg():
    spec = importlib.util.spec_from_file_location("p5", Path(__file__).resolve().parent / "20_resource_gathering.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def limits_of(d, alpha):
    return np.sqrt(0.5 * np.log(4.0 / alpha) * np.sum(d * d, axis=0))


def score(w, c, d, y):
    p = sigmoid(c + d @ w)
    return d.T @ (y - p)


def feasible(w, c, d, y, limits) -> bool:
    if w[0] < -1e-9 or w[1] < -1e-9 or w[0] + w[1] > 1.0 + 1e-9:
        return False
    return bool(np.all(np.abs(score(w, c, d, y)) <= limits + 1e-7))


def span(tri, c, d, y):
    logits = c[None, :] + tri @ d.T
    p_lo = sigmoid(logits.min(axis=0))
    p_hi = sigmoid(logits.max(axis=0))
    resid_lo = y - p_hi
    resid_hi = y - p_lo
    out = []
    for j in range(2):
        coef = d[:, j]
        low = np.minimum(coef * resid_lo, coef * resid_hi).sum()
        high = np.maximum(coef * resid_lo, coef * resid_hi).sum()
        out.append((float(low), float(high)))
    return out


def classify(tri, c, d, y, limits):
    if len(y) == 0:
        return "in"
    boxes = span(tri, c, d, y)
    for (lo, hi), limit in zip(boxes, limits):
        if hi < -limit or lo > limit:
            return "out"
        if lo < -limit - 1e-9 or hi > limit + 1e-9:
            return "maybe"
    return "in"


def children(tri):
    a, b, c = tri
    ab, bc, ca = 0.5 * (a + b), 0.5 * (b + c), 0.5 * (c + a)
    return (np.stack([a, ab, ca]), np.stack([b, bc, ab]), np.stack([c, ca, bc]), np.stack([ab, bc, ca]))


def gain_at(coef, base, w) -> float:
    return float(base + coef @ w)


def cover(c, d, y, limits, coef, base, target_gap=1e-4, max_depth=8):
    root = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    queue = [(root, 0)]
    kept = []
    tested = 0
    while queue:
        tri, depth = queue.pop()
        tested += 1
        kind = classify(tri, c, d, y, limits)
        if kind == "out":
            continue
        if kind == "maybe" and depth < max_depth:
            queue.extend((child, depth + 1) for child in children(tri))
            continue
        kept.append(tri)
    if not kept:
        return {"nonempty": False, "tested": tested, "lower": None, "upper": None, "gap": None, "vertices": 0, "fallback": "empty", "audit_failures": 0}
    vertices = np.unique(np.concatenate(kept, axis=0).round(10), axis=0)
    lower = min(gain_at(coef, base, w) for w in vertices)
    feasible_values = []
    for w in vertices:
        if feasible(w, c, d, y, limits):
            feasible_values.append(gain_at(coef, base, w))
    # A feasible local solution is an upper bound, never a certificate, and never falls back to the teacher.
    starts = [np.array(p) for p in ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.3, 0.3), (0.5, 0.2))]
    cons = [
        {"type": "ineq", "fun": lambda z: 1.0 - float(z[0] + z[1])},
        {"type": "ineq", "fun": lambda z: float(limits[0] - score(z, c, d, y)[0])},
        {"type": "ineq", "fun": lambda z: float(limits[0] + score(z, c, d, y)[0])},
        {"type": "ineq", "fun": lambda z: float(limits[1] - score(z, c, d, y)[1])},
        {"type": "ineq", "fun": lambda z: float(limits[1] + score(z, c, d, y)[1])},
    ]
    for start in starts:
        fit = minimize(lambda z: gain_at(coef, base, np.asarray(z, dtype=float)), start, method="SLSQP", bounds=[(0, 1), (0, 1)], constraints=cons, tol=1e-10)
        if fit.success and feasible(np.asarray(fit.x, dtype=float), c, d, y, limits):
            feasible_values.append(float(fit.fun))
    if not feasible_values:
        upper = None
        gap = None
    else:
        upper = float(min(feasible_values))
        gap = float(upper - lower)
    simplex = [gain_at(coef, base, w) for w in root]
    return {
        "nonempty": True,
        "tested": tested,
        "lower": float(lower),
        "upper": upper,
        "gap": gap,
        "vertices": int(len(vertices)),
        "fallback": "none",
        "simplex_lower": float(min(simplex)),
        "target_met": gap is not None and gap <= target_gap,
        "numeric_only": True,
        "audit_failures": _audit(c, d, y, limits, coef, base, lower),
    }


def _audit(c, d, y, limits, coef, base, lower) -> int:
    rng = np.random.default_rng(0)
    failures = 0
    for _ in range(200):
        a = float(rng.random())
        b = float(rng.random() * (1.0 - a))
        w = np.array([a, b])
        if feasible(w, c, d, y, limits) and gain_at(coef, base, w) + 1e-7 < lower:
            failures += 1
    return failures


def self_test():
    rng = np.random.default_rng(1)
    # Full simplex: no data. Affine min is the best vertex.
    coef = np.array([0.3, -0.2])
    base = 0.1
    full = cover(np.zeros(0), np.zeros((0, 2)), np.zeros(0), np.zeros(2), coef, base)
    # Concentrated design around (0.2, 0.2).
    w_true = np.array([0.2, 0.2])
    d = rng.normal(size=(80, 2))
    c = -d @ w_true + rng.normal(scale=0.05, size=80)
    y = (rng.random(80) < sigmoid(c + d @ w_true)).astype(float)
    limits = limits_of(d, 0.05)
    tight = cover(c, d, y, limits, coef, base)
    # Every numerically feasible random point must sit above the lower bound.
    violations = 0
    checked = 0
    for _ in range(400):
        a = rng.random()
        b = rng.random() * (1.0 - a)
        w = np.array([a, b])
        if feasible(w, c, d, y, limits):
            checked += 1
            if gain_at(coef, base, w) + 1e-8 < tight["lower"]:
                violations += 1
    # A region that cannot contain the simplex.
    d_out = np.array([[8.0, 0.0]])
    c_out = np.array([-20.0])
    y_out = np.array([1.0])
    empty = cover(c_out, d_out, y_out, np.array([0.01, 0.01]), coef, base, max_depth=4)
    return {"full": full, "tight": tight, "feasible_checked": checked, "lower_violations": violations, "empty_nonempty": empty["nonempty"]}


def main() -> None:
    started = time.perf_counter()
    tests = self_test()
    if tests["lower_violations"] or tests["full"]["gap"] is None or tests["full"]["gap"] > 1e-8 or tests["empty_nonempty"] or tests["tight"].get("audit_failures"):
        raise SystemExit(f"outer cover self-test failed: {tests}")
    rg = load_rg()
    model = rg.RG()
    import mo_gymnasium as mo_gym
    env = mo_gym.make("resource-gathering-v0")
    spec = importlib.util.spec_from_file_location("r4a", Path(__file__).resolve().parent / "25_rg_identification.py")
    r4a = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r4a)
    policies = {
        "uniform": np.zeros((model.n, 4)),
        "gold": r4a.behavior_logits(model, "gold"),
        "diamond": r4a.behavior_logits(model, "diamond"),
        "avoid": r4a.behavior_logits(model, "avoid"),
    }
    rng = np.random.default_rng(9100)
    feats = []
    for logits in policies.values():
        episodes, _steps = rg.live_batch(env, model, logits, 64, rng)
        feats.extend((ep["ret"] - rg.LOW) / rg.SCALE for ep in episodes)
    feats = np.stack(feats)
    left, right = rg.make_pairs(len(feats), 512, rng)
    c_all, d_all = rg.design(feats[left] - feats[right])
    neutral = np.array([1.0 / 3.0, 1.0 / 3.0])
    designs = {}
    source = r4a.select_indices(c_all, d_all, neutral, "low", 32, np.random.default_rng(9400))
    mixed = r4a.select_indices(c_all, d_all, neutral, "uniform", 32, np.random.default_rng(9432))
    pool = np.setdiff1d(np.arange(len(d_all)), source)
    designs["weak_source"] = source
    designs["weak_uniform"] = np.concatenate([source, pool[r4a.select_indices(c_all[pool], d_all[pool], neutral, "uniform", 64, np.random.default_rng(9500))]])
    designs["weak_trace"] = np.concatenate([source, pool[r4a.select_indices(c_all[pool], d_all[pool], neutral, "trace", 64, np.random.default_rng(9500))]])
    designs["weak_mineig"] = np.concatenate([source, pool[r4a.select_indices(c_all[pool], d_all[pool], neutral, "mineig", 64, np.random.default_rng(9500))]])
    y_bits = np.random.default_rng(9600)
    rows = []
    logits = np.zeros((model.n, 4))
    mu = model.values(rg.softmax(logits))
    # A fixed candidate step toward the true weight, used only to score the bound. The bound itself does not use it as a fallback.
    nxt = logits + model.gradient(rg.softmax(logits), rg.W_STAR)
    delta = (model.values(rg.softmax(nxt)) - mu) / rg.SCALE
    coef = np.array([delta[0] - delta[2], delta[1] - delta[2]])
    base = float(delta[2])
    for name, idx in designs.items():
        y = (y_bits.random(len(idx)) < rg.sigmoid(c_all[idx] + d_all[idx] @ rg.W_STAR[:2])).astype(float)
        limits = limits_of(d_all[idx], 0.05)
        t0 = time.perf_counter()
        result = cover(c_all[idx], d_all[idx], y, limits, coef, base)
        result["design"] = name
        result["seconds"] = time.perf_counter() - t0
        result["labels"] = int(len(idx))
        rows.append(result)
    simplex = cover(np.zeros(0), np.zeros((0, 2)), np.zeros(0), np.zeros(2), coef, base)
    simplex["design"] = "full_simplex"
    rows.append(simplex)
    env.close()
    out = Path("outputs/horizon_bridge/r4_rg_robust")
    out.mkdir(parents=True, exist_ok=True)
    payload = {"self_test": tests, "rows": rows, "stage_runtime_seconds": time.perf_counter() - started, "verification": "numeric outer cover, not outward-rounded interval arithmetic"}
    write_summary(out, jsonable(payload), started)


if __name__ == "__main__":
    main()
