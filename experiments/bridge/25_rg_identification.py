#!/usr/bin/env python3
"""R4A: two-direction preference coverage on the existing Resource-Gathering map."""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.bridge.common import write_summary
from experiments.bridge.pref_common import jsonable, softmax


def load_rg():
    spec = importlib.util.spec_from_file_location("p5", Path(__file__).resolve().parent / "20_resource_gathering.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def behavior_logits(model, goal: str) -> np.ndarray:
    """Teacher-blind policies read from the map. Enemies can be blocked."""
    blocked = set()
    targets = []
    home = None
    for x in range(5):
        for y in range(5):
            cell = str(model.map[x, y])
            if cell == "H":
                home = (x, y)
            elif cell == "R1" and goal in ("gold", "avoid"):
                targets.append((x, y))
            elif cell == "R2" and goal == "diamond":
                targets.append((x, y))
            elif cell in ("E1", "E2") and goal == "avoid":
                blocked.add((x, y))
    if goal == "avoid" and not targets:
        targets = [home]
    logits = np.zeros((model.n, 4))
    dirs = ((-1, 0), (1, 0), (0, -1), (0, 1))
    for state in range(model.n):
        x, y, gold, gem = model_decode(state)
        goal_now = home if (gold or gem or not targets) else targets[0]
        nxt = _bfs((x, y), goal_now, blocked)
        if nxt is None:
            continue
        for action, (dx, dy) in enumerate(dirs):
            if (x + dx, y + dy) == nxt or ((x + dx, y + dy) not in ((nx, ny) for nx in range(5) for ny in range(5)) and (x, y) == nxt):
                logits[state, action] = 6.0
        step = (nxt[0] - x, nxt[1] - y)
        for action, delta in enumerate(dirs):
            if delta == step:
                logits[state, action] = 6.0
    return logits


def model_decode(state: int):
    gem = state % 2
    gold = (state // 2) % 2
    y = (state // 4) % 5
    x = state // 20
    return x, y, gold, gem


def _bfs(start, goal, blocked):
    if start == goal:
        return start
    from collections import deque
    dirs = ((-1, 0), (1, 0), (0, -1), (0, 1))
    seen = {start}
    queue = deque([(start, None)])
    while queue:
        (x, y), first = queue.popleft()
        for dx, dy in dirs:
            nxt = (x + dx, y + dy)
            if not (0 <= nxt[0] < 5 and 0 <= nxt[1] < 5) or nxt in blocked or nxt in seen:
                continue
            seen.add(nxt)
            step = nxt if first is None else first
            if nxt == goal:
                return step
            queue.append((nxt, step))
    return None


def fisher_eigs(c, d, w):
    p = 1.0 / (1.0 + np.exp(-np.clip(c + d @ w, -40, 40)))
    weight = p * (1.0 - p)
    mat = (d * weight[:, None]).T @ d
    eigs = np.linalg.eigvalsh(mat)
    return eigs, mat


def select_indices(c, d, w, rule, k, rng):
    if rule == "uniform":
        return rng.choice(len(d), size=k, replace=False)
    if rule == "low":
        return np.argsort(np.sum(d * d, axis=1))[:k]
    if rule == "trace":
        p = 1.0 / (1.0 + np.exp(-np.clip(c + d @ w, -40, 40)))
        info = np.sum(d * d, axis=1) * p * (1.0 - p)
        return np.argsort(-info)[:k]
    # Sequential minimum-eigenvalue greedy. A tiny ridge only breaks exact ties.
    chosen = []
    remaining = set(range(len(d)))
    p = 1.0 / (1.0 + np.exp(-np.clip(c + d @ w, -40, 40)))
    info = p * (1.0 - p)
    mat = np.eye(2) * 1e-8
    for _ in range(k):
        best_i, best_eig = None, -1.0
        for i in remaining:
            cand = mat + info[i] * np.outer(d[i], d[i])
            eig = float(np.linalg.eigvalsh(cand)[0])
            if eig > best_eig + 1e-12 or (abs(eig - best_eig) <= 1e-12 and (best_i is None or i < best_i)):
                best_i, best_eig = i, eig
        chosen.append(best_i)
        remaining.remove(best_i)
        mat = mat + info[best_i] * np.outer(d[best_i], d[best_i])
    return np.array(chosen, dtype=int)


def grid_region(rg, c, d, y, alpha):
    limits = np.sqrt(0.5 * np.log(4.0 / alpha) * np.sum(d * d, axis=0))
    inside = []
    for a in np.linspace(0, 1, 21):
        for b in np.linspace(0, max(1.0 - a, 0.0), 21):
            w = np.array([a, b])
            p = 1.0 / (1.0 + np.exp(-np.clip(c + d @ w, -40, 40)))
            score = d.T @ (y - p)
            if np.all(np.abs(score) <= limits + 1e-8):
                inside.append(w)
    if not inside:
        return {"feasible": 0, "covers_true": False, "box": None}
    pts = np.stack(inside)
    true = rg.W_STAR[:2]
    return {
        "feasible": int(len(inside)),
        "covers_true": bool(np.min(np.linalg.norm(pts - true, axis=1)) <= 0.05),
        "box": [float(pts[:, 0].min()), float(pts[:, 0].max()), float(pts[:, 1].min()), float(pts[:, 1].max())],
    }


def main() -> None:
    started = time.perf_counter()
    rg = load_rg()
    model = rg.RG()
    import mo_gymnasium as mo_gym
    env = mo_gym.make("resource-gathering-v0")
    policies = {
        "uniform": np.zeros((model.n, 4)),
        "gold": behavior_logits(model, "gold"),
        "diamond": behavior_logits(model, "diamond"),
        "avoid": behavior_logits(model, "avoid"),
    }
    rng = np.random.default_rng(9100)
    feats = []
    origins = []
    for name, logits in policies.items():
        episodes, _steps = rg.live_batch(env, model, logits, 64, rng)
        for ep in episodes:
            feats.append((ep["ret"] - rg.LOW) / rg.SCALE)
            origins.append(name)
    feats = np.stack(feats)
    left, right = rg.make_pairs(len(feats), 512, rng)
    c_all, d_all = rg.design(feats[left] - feats[right])
    # Held-out sets use disjoint episodes.
    bal_idx = [i for i, name in enumerate(origins) if name != "uniform"]
    nat_idx = [i for i, name in enumerate(origins) if name == "uniform"]
    held = {}
    for key, idx in (("balanced", bal_idx), ("natural", nat_idx)):
        sub = np.array(idx)
        h_left, h_right = rg.make_pairs(len(sub), 128, np.random.default_rng(9200 if key == "balanced" else 9201))
        diff = feats[sub[h_left]] - feats[sub[h_right]]
        hc, hd = rg.design(diff)
        hy = (np.random.default_rng(9300).random(len(hd)) < rg.sigmoid(hc + hd @ rg.W_STAR[:2])).astype(float)
        held[key] = (hc, hd, hy)
    designs = {
        "weak_source": ("low", 32, None),
        "mixed_source": ("uniform", 32, None),
        "weak_uniform": ("uniform", 64, "low"),
        "weak_trace": ("trace", 64, "low"),
        "weak_mineig": ("mineig", 64, "low"),
        "mixed_mineig": ("mineig", 64, "uniform"),
    }
    rows = []
    neutral = np.array([1.0 / 3.0, 1.0 / 3.0])
    for name, (rule, k, source_rule) in designs.items():
        if source_rule is None:
            idx = select_indices(c_all, d_all, neutral, rule, k, np.random.default_rng(9400 + k))
            extra = np.array([], dtype=int)
        else:
            source_idx = select_indices(c_all, d_all, neutral, source_rule, 32, np.random.default_rng(9400))
            pool = np.setdiff1d(np.arange(len(d_all)), source_idx)
            extra = select_indices(c_all[pool], d_all[pool], neutral, rule, k, np.random.default_rng(9500))
            extra = pool[extra]
            idx = np.concatenate([source_idx, extra])
        for rep in range(20):
            bits = np.random.default_rng(10_000 + 100 * rep + (sum(ord(ch) for ch in name) % 97))
            y = (bits.random(len(idx)) < rg.sigmoid(c_all[idx] + d_all[idx] @ rg.W_STAR[:2])).astype(float)
            slsqp, soft, gap = rg.fit_simplex(c_all[idx], d_all[idx], y)
            eigs, _mat = fisher_eigs(c_all[idx], d_all[idx], slsqp)
            region = grid_region(rg, c_all[idx], d_all[idx], y, 0.05)
            full = np.array([slsqp[0], slsqp[1], 1.0 - slsqp.sum()])
            rows.append({
                "design": name,
                "rep": rep,
                "labels": int(len(idx)),
                "eig_min": float(eigs[0]),
                "eig_max": float(eigs[1]),
                "condition": float(eigs[1] / max(eigs[0], 1e-12)),
                "effective_rank": float((eigs.sum() ** 2) / max(np.sum(eigs ** 2), 1e-12)),
                "mle_l2": float(np.linalg.norm(full - rg.W_STAR)),
                "solver_gap": float(gap),
                "nll_balanced": rg.nll(slsqp, *held["balanced"]) / len(held["balanced"][2]),
                "nll_natural": rg.nll(slsqp, *held["natural"]) / len(held["natural"][2]),
                "feasible_grid": region["feasible"],
                "covers_true": region["covers_true"],
                "box": region["box"],
            })
    env.close()
    # Finite-difference the analytic policy gradient once.
    logits = np.zeros((model.n, 4))
    analytic = model.gradient(softmax(logits), rg.W_STAR)
    eps = 1e-5
    base = rg.scalarize(model.values(softmax(logits)), rg.W_STAR)
    rng_fd = np.random.default_rng(7)
    picks = rng_fd.choice(logits.size, size=3, replace=False)
    gaps = []
    for pos in picks:
        trial = logits.copy()
        trial.ravel()[pos] += eps
        gaps.append(((rg.scalarize(model.values(softmax(trial)), rg.W_STAR) - base) / eps, float(analytic.ravel()[pos])))
    out = Path("outputs/horizon_bridge/r4_rg_identification")
    summary = {
        "designs": _aggregate(rows),
        "finite_difference_spots": gaps,
        "gradient_norm": float(np.linalg.norm(analytic)),
        "stage_runtime_seconds": time.perf_counter() - started,
    }
    (out).mkdir(parents=True, exist_ok=True)
    (out / "rows.json").write_text(json.dumps(jsonable(rows)) + "\n")
    write_summary(out, jsonable(summary), started)


def _aggregate(rows):
    out = {}
    for name in sorted({row["design"] for row in rows}):
        group = [row for row in rows if row["design"] == name]
        out[name] = {"n": len(group)}
        for key in ("eig_min", "eig_max", "condition", "effective_rank", "mle_l2", "solver_gap", "nll_balanced", "nll_natural", "feasible_grid"):
            out[name][key] = float(np.mean([row[key] for row in group]))
        out[name]["coverage"] = float(np.mean([row["covers_true"] for row in group]))
    return out


if __name__ == "__main__":
    main()
