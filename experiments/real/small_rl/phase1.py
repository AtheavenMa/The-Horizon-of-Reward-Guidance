"""E1–E3 on MO-Gymnasium.

The teacher weight w_star labels trajectory pairs and is never passed to the
reward fit, the policy update, or the query selector. Finite-step gain is the
change in independent evaluation return after one tabular policy update.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.real.common.records import ROOT, load_checkpoint, run_dir, save_checkpoint, unpack_json, write_run

CHI2_95_1DF = 3.841458820694124


def _env(env_id: str):
    import mo_gymnasium as mo_gym
    return mo_gym.make(env_id)


def _key(obs: np.ndarray) -> tuple:
    arr = np.asarray(obs, dtype=np.float64).reshape(-1)
    if np.all(np.equal(arr, np.round(arr))):
        return tuple(int(x) for x in arr)
    return tuple(np.round(arr, 3).tolist())


def rollout(env, policy: dict, rng: np.random.Generator, n_actions: int, max_steps: int) -> dict:
    obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
    steps = []
    total = None
    for _ in range(max_steps):
        key = _key(obs)
        logits = policy.get(key)
        if logits is None:
            logits = np.zeros(n_actions, dtype=np.float64)
        z = logits - logits.max()
        prob = np.exp(z)
        prob /= prob.sum()
        action = int(rng.choice(n_actions, p=prob))
        nxt, reward, terminated, truncated, _ = env.step(action)
        reward = np.asarray(reward, dtype=np.float64).reshape(-1)
        if total is None:
            total = np.zeros_like(reward)
        total += reward
        steps.append((key, action, logits.copy(), prob.copy()))
        obs = nxt
        if terminated or truncated:
            break
    assert total is not None
    return {"steps": steps, "ret": total}


def collect(env, policy, rng, n_episodes: int, n_actions: int, max_steps: int) -> list[dict]:
    return [rollout(env, policy, rng, n_actions, max_steps) for _ in range(n_episodes)]


def normalize(ret: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return (ret - low) / np.maximum(high - low, 1e-6)


def mean_feature(env, policy, rng, n: int, n_actions: int, max_steps: int, low, high) -> tuple[np.ndarray, np.ndarray]:
    feats = [normalize(ep["ret"], low, high) for ep in collect(env, policy, rng, n, n_actions, max_steps)]
    arr = np.stack(feats)
    return arr.mean(axis=0), arr.std(axis=0, ddof=1) / np.sqrt(len(arr))


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -40, 40)
    return 1 / (1 + np.exp(-x))


def label_pairs(diffs: np.ndarray, w_star: np.ndarray, beta: float, rng: np.random.Generator) -> np.ndarray:
    """Evaluator-only. Returns binary labels and does not return w_star."""
    prob = sigmoid(beta * diffs @ w_star)
    return rng.random(len(diffs)) < prob


def loglik(diffs: np.ndarray, labels: np.ndarray, w: np.ndarray, beta: float) -> float:
    score = beta * diffs @ w
    # stable binary log-likelihood
    return float(np.sum(labels * score - np.logaddexp(0, score)))


def fit_reward(diffs: np.ndarray, labels: np.ndarray, beta: float, n_steps: int, step_cap: float, w0: np.ndarray | None = None) -> np.ndarray:
    dim = diffs.shape[1]
    w = np.zeros(dim) if w0 is None else w0.astype(np.float64).copy()
    for _ in range(n_steps):
        score = beta * diffs @ w
        resid = sigmoid(score) - labels
        grad = beta * diffs.T @ resid / max(len(diffs), 1)
        step = -0.5 * grad
        norm = np.linalg.norm(step)
        if norm > step_cap:
            step *= step_cap / norm
        w = w + step
        w = np.clip(w, -2, 2)
    return w


def simplex_grid(dim: int, n: int = 11) -> np.ndarray:
    if dim == 2:
        a = np.linspace(0, 1, n)
        return np.stack([a, 1 - a], axis=1)
    if dim == 3:
        pts = []
        for i in range(n):
            for j in range(n - i):
                a, b = i / (n - 1), j / (n - 1)
                pts.append((a, b, 1 - a - b))
        return np.asarray(pts, dtype=np.float64)
    raise ValueError(dim)


def confidence_set(diffs: np.ndarray, labels: np.ndarray, beta: float, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lls = np.array([loglik(diffs, labels, w, beta) for w in grid])
    best = lls.max()
    keep = 2 * (best - lls) <= CHI2_95_1DF
    if not np.any(keep):
        return grid[:0], lls
    return grid[keep], lls


def coverage(diffs: np.ndarray, w_star: np.ndarray, beta: float, grid: np.ndarray, rng: np.random.Generator, reps: int = 40) -> float:
    hits = 0
    star = w_star / w_star.sum()
    idx = int(np.argmin(np.linalg.norm(grid - star, axis=1)))
    for _ in range(reps):
        labels = label_pairs(diffs, w_star, beta, rng).astype(np.float64)
        kept, _ = confidence_set(diffs, labels, beta, grid)
        if len(kept) and np.min(np.linalg.norm(kept - grid[idx], axis=1)) < 1e-8:
            hits += 1
    return hits / reps


def policy_update(policy: dict, episodes: list[dict], phi: np.ndarray, low, high, lr: float, n_actions: int) -> dict:
    """One tabular softmax update. The terminal return is phi dotted with the normalized episode return."""
    updated = {k: v.copy() for k, v in policy.items()}
    returns = []
    scored = []
    for ep in episodes:
        feat = normalize(ep["ret"], low, high)
        g = float(phi @ feat)
        returns.append(g)
        scored.append((ep, g))
    if not scored:
        return updated
    baseline = float(np.mean(returns))
    grads: dict[tuple, np.ndarray] = {}
    for ep, g in scored:
        adv = g - baseline
        for key, action, _logits, prob in ep["steps"]:
            grad = grads.get(key)
            if grad is None:
                grad = np.zeros(n_actions)
                grads[key] = grad
            one = np.zeros(n_actions)
            one[action] = 1
            grad += adv * (one - prob)
    scale = max(len(scored), 1)
    for key, grad in grads.items():
        logits = updated.get(key, np.zeros(n_actions)).copy()
        step = lr * grad / scale
        norm = np.linalg.norm(step)
        if norm > 1.0:
            step *= 1.0 / norm
        updated[key] = logits + step
    return updated


def delta_mu(env, base, updated, rng, n_eval, n_actions, max_steps, low, high) -> tuple[np.ndarray, np.ndarray]:
    mu0, se0 = mean_feature(env, base, rng, n_eval, n_actions, max_steps, low, high)
    mu1, se1 = mean_feature(env, updated, rng, n_eval, n_actions, max_steps, low, high)
    return mu1 - mu0, np.sqrt(se0**2 + se1**2)


def worst_gain(delta: np.ndarray, weights: np.ndarray) -> float:
    if len(weights) == 0:
        return float("nan")
    return float(np.min(weights @ delta))


def best_candidate(delta_by_phi: list[tuple[np.ndarray, np.ndarray]], weights: np.ndarray) -> tuple[float, int]:
    scores = [worst_gain(delta, weights) for _phi, delta in delta_by_phi]
    if not scores or np.all(np.isnan(scores)):
        return float("nan"), -1
    idx = int(np.nanargmax(scores))
    return scores[idx], idx


def sample_diffs(feats: np.ndarray, n_pairs: int, rng: np.random.Generator) -> np.ndarray:
    i = rng.integers(0, len(feats), size=n_pairs)
    j = rng.integers(0, len(feats), size=n_pairs)
    same = i == j
    j = np.where(same, (j + 1) % len(feats), j)
    return feats[i] - feats[j]


def freeze_development(env_id: str, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    env = _env(env_id)
    n_actions = int(env.action_space.n)
    max_steps = 40 if "resource" in env_id else 20
    episodes = collect(env, {}, rng, 60, n_actions, max_steps)
    rets = np.stack([ep["ret"] for ep in episodes])
    low = rets.min(axis=0)
    high = rets.max(axis=0)
    feats = np.stack([normalize(ep["ret"], low, high) for ep in episodes])
    dim = feats.shape[1]
    grid_w = simplex_grid(dim, 9 if dim == 2 else 6)
    beta_grid = [1.0, 2.0, 5.0, 10.0]
    target = 0.20
    best = None
    diffs = sample_diffs(feats, 200, rng)
    for beta in beta_grid:
        for w in grid_w:
            if np.min(w) < 0.15:
                continue
            prob = sigmoid(beta * diffs @ w)
            med = float(np.median(np.abs(prob - 0.5)))
            gap = abs(med - target)
            if best is None or gap < best[0]:
                best = (gap, beta, w.copy(), med)
    assert best is not None
    _gap, beta, w_star, med = best
    # Development finite-step gains use the frozen teacher, then set gamma_pos.
    gains = _development_gains(env, rng, n_actions, max_steps, low, high, w_star, beta, grid_w)
    positive = [g for g in gains if g > 1e-4]
    if positive:
        gamma_pos = 0.1 * float(np.median(positive))
        gamma_note = "0.1 times median positive development finite-step gain"
    else:
        gamma_pos = 0.01
        gamma_note = "development had no positive finite-step gain; gamma_pos fixed at 0.01 before confirmatory runs"
    cov = coverage(diffs[:48], w_star, beta, grid_w, rng, reps=30)
    return {
        "env_id": env_id,
        "low": low.tolist(),
        "high": high.tolist(),
        "w_star": w_star.tolist(),
        "beta": beta,
        "pair_median_abs_half": med,
        "gamma": 0.0,
        "gamma_pos": gamma_pos,
        "gamma_note": gamma_note,
        "coverage_development": cov,
        "certificate_formal": bool(cov >= 0.90),
        "n_actions": n_actions,
        "max_steps": max_steps,
        "dim": dim,
        "positive_development_gains": positive,
    }


def _development_gains(env, rng, n_actions, max_steps, low, high, w_star, beta, grid_w) -> list[float]:
    policy: dict = {}
    gains = []
    episodes = collect(env, policy, rng, 16, n_actions, max_steps)
    feats = np.stack([normalize(ep["ret"], low, high) for ep in episodes])
    diffs = sample_diffs(feats, 32, rng)
    labels = label_pairs(diffs, w_star, beta, rng).astype(np.float64)
    phi = fit_reward(diffs, labels, beta, n_steps=8, step_cap=0.2)
    for candidate in grid_w[:: max(len(grid_w) // 4, 1)]:
        updated = policy_update(policy, episodes, candidate, low, high, lr=0.4, n_actions=n_actions)
        delta, _se = delta_mu(env, policy, updated, rng, 8, n_actions, max_steps, low, high)
        gains.append(float(w_star @ delta))
    _ = phi
    return gains


def run_seed(frozen: dict, seed: int, rounds: int, n_episodes: int, n_pairs: int, n_eval: int, out: Path) -> dict:
    rng = np.random.default_rng(seed)
    env = _env(frozen["env_id"])
    low = np.asarray(frozen["low"], dtype=np.float64)
    high = np.asarray(frozen["high"], dtype=np.float64)
    w_star = np.asarray(frozen["w_star"], dtype=np.float64)
    beta = float(frozen["beta"])
    n_actions = int(frozen["n_actions"])
    max_steps = int(frozen["max_steps"])
    grid = simplex_grid(frozen["dim"], 11 if frozen["dim"] == 2 else 6)
    q_grid = grid  # fixed finite candidate family Q
    policy: dict = {}
    episodes = collect(env, policy, rng, n_episodes, n_actions, max_steps)
    feats = np.stack([normalize(ep["ret"], low, high) for ep in episodes])
    diffs = sample_diffs(feats, n_pairs, rng)
    labels = label_pairs(diffs, w_star, beta, rng).astype(np.float64)
    phi = fit_reward(diffs, labels, beta, n_steps=10, step_cap=0.15)
    u0, _ = confidence_set(diffs, labels, beta, grid)
    rows = []
    events = []
    events.append({"type": "source_labels", "n": int(len(labels)), "positive": int(labels.sum())})
    loss_round = None
    for round_id in range(rounds):
        # Candidate updates share the trajectories collected under the current policy.
        batch = collect(env, policy, rng, n_episodes, n_actions, max_steps)
        deltas = []
        for candidate in q_grid:
            updated = policy_update(policy, batch, candidate, low, high, lr=0.5, n_actions=n_actions)
            delta, se = delta_mu(env, policy, updated, rng, n_eval, n_actions, max_steps, low, high)
            deltas.append((candidate, delta, se))
        learned_delta, learned_se = delta_mu(
            env, policy, policy_update(policy, batch, phi, low, high, lr=0.5, n_actions=n_actions),
            rng, n_eval, n_actions, max_steps, low, high,
        )
        packed = [(c, d) for c, d, _se in deltas]
        v_q, v_idx = best_candidate(packed, u0)
        true_gain = float(w_star @ learned_delta)
        cert = worst_gain(learned_delta, u0)
        cert0 = worst_gain(learned_delta, u0)  # gamma = 0 uses the same point estimate
        row = {
            "seed": seed,
            "round": round_id,
            "original_certificate": cert,
            "certificate_gamma0": cert0,
            "true_finite_step_gain": true_gain,
            "candidate_restricted_value": v_q,
            "true_gain_se": float(np.linalg.norm(learned_se)),
            "phi0": float(phi[0]),
            "phi1": float(phi[1]) if len(phi) > 1 else 0.0,
            "n_confidence": int(len(u0)),
        }
        rows.append(row)
        events.append({"type": "checkpoint", "round": round_id, "certificate": cert, "true_gain": true_gain, "V_Q": v_q})
        if loss_round is None and cert < frozen["gamma_pos"]:
            loss_round = round_id
        ckpt = {
            "policy_json": json.dumps(_policy_dump(policy)),
            "phi": phi,
            "diffs": diffs,
            "labels": labels,
            "round": np.array([round_id]),
            "seed": np.array([seed]),
            "learned_delta": learned_delta,
        }
        save_checkpoint(out, f"seed{seed}_round{round_id}", ckpt)
        # One learning round on the learned reward, then new trajectories. True reward is not a training input.
        policy = policy_update(policy, batch, phi, low, high, lr=0.5, n_actions=n_actions)
        new_eps = collect(env, policy, rng, max(8, n_episodes // 2), n_actions, max_steps)
        new_feats = np.stack([normalize(ep["ret"], low, high) for ep in new_eps])
        extra = sample_diffs(new_feats, max(8, n_pairs // 4), rng)
        extra_labels = label_pairs(extra, w_star, beta, rng).astype(np.float64)
        # Source-only certificate above uses the original U0. New labels are recorded but not folded into U0.
        diffs = np.concatenate([diffs, extra])
        labels = np.concatenate([labels, extra_labels])
        phi = fit_reward(diffs, labels, beta, n_steps=4, step_cap=0.1, w0=phi)
        events.append({"type": "added_labels", "round": round_id, "n": int(len(extra_labels))})
    status = {
        "status": "complete",
        "loss_round": loss_round,
        "censored": loss_round is None,
        "gamma_pos": frozen["gamma_pos"],
        "gamma": 0.0,
        "certificate_formal": frozen["certificate_formal"],
    }
    write_run(out, {**frozen, "seed": seed, "rounds": rounds, "n_pairs": n_pairs}, rows, events, status)
    return {"rows": rows, "status": status, "frozen": frozen}


def _policy_dump(policy: dict) -> list[dict]:
    return [{"s": list(k), "logits": v.tolist()} for k, v in policy.items()]


def load_policy(blob: str, n_actions: int) -> dict:
    policy = {}
    for item in json.loads(blob):
        policy[tuple(item["s"])] = np.asarray(item["logits"], dtype=np.float64)
    _ = n_actions
    return policy


def summarize_task(rows_by_seed: list[list[dict]], frozen: dict) -> dict:
    loss_rounds = []
    v_pos = 0
    true_at_loss = []
    for rows in rows_by_seed:
        loss = next((r for r in rows if r["original_certificate"] < frozen["gamma_pos"]), None)
        if loss is None:
            loss_rounds.append("censored")
        else:
            loss_rounds.append(loss["round"])
            if loss["candidate_restricted_value"] >= frozen["gamma_pos"]:
                v_pos += 1
            true_at_loss.append(loss["true_finite_step_gain"])
    numeric = [x for x in loss_rounds if x != "censored"]
    median = "censored" if not numeric else float(np.median(numeric))
    if any(x == "censored" for x in loss_rounds) and numeric:
        median = f"{median} (some censored)"
    return {
        "seeds": len(rows_by_seed),
        "loss": median,
        "fraction_positive_V": v_pos / max(len(rows_by_seed), 1),
        "true_gain_at_loss": float(np.median(true_at_loss)) if true_at_loss else "censored",
        "coverage": frozen["coverage_development"],
        "formal": frozen["certificate_formal"],
        "gamma_pos": frozen["gamma_pos"],
        "gamma_note": frozen["gamma_note"],
    }


def run_e1(stage: str) -> dict:
    results = {}
    for env_id, name in (
        ("deep-sea-treasure-v0", "Deep-Sea-Treasure"),
        ("resource-gathering-v0", "Resource-Gathering"),
    ):
        frozen_path = run_dir("E1", name, "frozen")
        frozen_file = frozen_path / "frozen.json"
        if frozen_file.exists() and stage == "confirm":
            frozen = json.loads(frozen_file.read_text())
        else:
            frozen = freeze_development(env_id, seed=0)
            frozen_file.write_text(json.dumps(frozen, indent=2) + "\n")
        if stage == "pilot":
            seeds = [1, 2]
            rounds, episodes, pairs, n_eval = 2, 12, 32, 8
        else:
            seeds = [1, 2, 3, 4, 5]
            rounds, episodes, pairs, n_eval = 4, 16, 48, 24
        per_seed = []
        for seed in seeds:
            out = run_dir("E1", name, f"seed{seed}")
            result = run_seed(frozen, seed, rounds, episodes, pairs, n_eval, out)
            per_seed.append(result["rows"])
        summary = summarize_task(per_seed, frozen)
        summary_path = run_dir("E1", name)
        (summary_path / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        results[name] = summary
        print(name, json.dumps(summary), flush=True)
    return results


def run_e2() -> dict:
    """Five branches from two saved E1 checkpoints. Policy-update budget stays one step."""
    name = "Deep-Sea-Treasure"
    frozen = json.loads((run_dir("E1", name, "frozen") / "frozen.json").read_text())
    # Choose rounds on development seed 1 logs: early, and the loss round if present.
    seed_rows = []
    for seed in (1, 2, 3):
        metrics = _read_metrics(run_dir("E1", name, f"seed{seed}") / "metrics.csv")
        if metrics:
            seed_rows.append(metrics)
    if not seed_rows:
        raise RuntimeError("E2 requires E1 checkpoints")
    loss_rounds = [int(r["round"]) for rows in seed_rows for r in rows if float(r["original_certificate"]) < frozen["gamma_pos"]]
    early = 0
    late = loss_rounds[0] if loss_rounds else int(seed_rows[0][-1]["round"])
    late_label = "near suspected loss" if loss_rounds else "later matched-progress checkpoint; E1 showed no loss"
    out_rows = []
    for seed in (1, 2, 3):
        for round_id, label in ((early, "before loss"), (late, late_label)):
            ckpt_path = run_dir("E1", name, f"seed{seed}") / "checkpoints" / f"seed{seed}_round{round_id}.npz"
            if not ckpt_path.exists():
                continue
            out_rows.extend(_fork(frozen, seed, round_id, label, ckpt_path))
    dest = run_dir("E2", name)
    write_run(dest, {"source": "E1 Deep-Sea-Treasure", "rounds": [early, late], "late_label": late_label}, out_rows, [], {"status": "complete"})
    (dest / "summary.json").write_text(json.dumps({"n_rows": len(out_rows), "late_label": late_label}, indent=2) + "\n")
    print("E2 rows", len(out_rows), flush=True)
    return {"rows": out_rows, "late_label": late_label}


def _fork(frozen: dict, seed: int, round_id: int, label: str, ckpt_path: Path) -> list[dict]:
    rng = np.random.default_rng(10_000 + seed * 10 + round_id)
    env = _env(frozen["env_id"])
    low = np.asarray(frozen["low"])
    high = np.asarray(frozen["high"])
    w_star = np.asarray(frozen["w_star"])
    beta = float(frozen["beta"])
    n_actions = int(frozen["n_actions"])
    max_steps = int(frozen["max_steps"])
    blob = np.load(ckpt_path, allow_pickle=False)
    policy = load_policy(str(blob["policy_json"]), n_actions)
    phi = np.asarray(blob["phi"], dtype=np.float64)
    diffs = np.asarray(blob["diffs"], dtype=np.float64)
    labels = np.asarray(blob["labels"], dtype=np.float64)
    grid = simplex_grid(frozen["dim"], 11 if frozen["dim"] == 2 else 6)
    batch = collect(env, policy, rng, 12, n_actions, max_steps)
    new_feats = np.stack([normalize(ep["ret"], low, high) for ep in batch])
    new_diffs = sample_diffs(new_feats, 16, rng)
    new_labels = label_pairs(new_diffs, w_star, beta, rng).astype(np.float64)
    branches = [
        ("Original", diffs, labels, 4, False),
        ("More reward updates, same data", diffs, labels, 16, False),
        ("New feedback, matched updates", np.concatenate([diffs, new_diffs]), np.concatenate([labels, new_labels]), 4, True),
        ("New feedback + more updates", np.concatenate([diffs, new_diffs]), np.concatenate([labels, new_labels]), 16, True),
        ("Stronger guidance search, same data", diffs, labels, 4, False),
    ]
    rows = []
    for name, d, y, steps, added in branches:
        phi_b = fit_reward(d, y, beta, n_steps=steps, step_cap=0.1, w0=phi)
        u_new, _ = confidence_set(d, y, beta, grid)
        u_old, _ = confidence_set(diffs, labels, beta, grid)
        # Updated certificate on the unchanged trajectory uses the saved learned delta.
        learned_delta = np.asarray(blob["learned_delta"], dtype=np.float64)
        updated_policy = policy_update(policy, batch, phi_b, low, high, lr=0.5, n_actions=n_actions)
        delta, _se = delta_mu(env, policy, updated_policy, rng, 8, n_actions, max_steps, low, high)
        if name.startswith("Stronger"):
            v, _ = best_candidate([(c, delta_mu(env, policy, policy_update(policy, batch, c, low, high, lr=0.5, n_actions=n_actions), rng, 6, n_actions, max_steps, low, high)[0]) for c in grid[::2]], u_old)
        else:
            v = worst_gain(delta, u_new)
        rows.append({
            "seed": seed,
            "checkpoint_round": round_id,
            "checkpoint_label": label,
            "branch": name,
            "new_pairs": 16 if added else 0,
            "reward_updates": steps,
            "original_certificate": worst_gain(learned_delta, u_old),
            "updated_certificate_unchanged": worst_gain(learned_delta, u_new),
            "true_finite_step_gain": float(w_star @ delta),
            "candidate_value": v,
        })
    return rows


def run_e3() -> dict:
    import time
    name = "Deep-Sea-Treasure"
    frozen = json.loads((run_dir("E1", name, "frozen") / "frozen.json").read_text())
    rows = []
    for seed in (1, 2, 3):
        ckpt_path = run_dir("E1", name, f"seed{seed}") / "checkpoints" / f"seed{seed}_round0.npz"
        rows.extend(_e3_seed(frozen, name, seed, ckpt_path, time))
    dest = run_dir("E3", name)
    write_run(dest, {"checkpoint": "round0", "added": 16, "beta": float(frozen["beta"]), "seeds": [1, 2, 3]}, rows, [], {"status": "complete"})
    print("E3", json.dumps(rows), flush=True)
    return {"rows": rows}


def _e3_seed(frozen: dict, name: str, seed: int, ckpt_path: Path, time) -> list[dict]:
    blob = np.load(ckpt_path, allow_pickle=False)
    diffs = np.asarray(blob["diffs"], dtype=np.float64)
    labels = np.asarray(blob["labels"], dtype=np.float64)
    phi = np.asarray(blob["phi"], dtype=np.float64)
    learned_delta = np.asarray(blob["learned_delta"], dtype=np.float64)
    w_star = np.asarray(frozen["w_star"])
    beta = float(frozen["beta"])
    grid = simplex_grid(frozen["dim"], 11)
    rng = np.random.default_rng(123 + seed)
    env = _env(frozen["env_id"])
    low = np.asarray(frozen["low"])
    high = np.asarray(frozen["high"])
    n_actions = int(frozen["n_actions"])
    max_steps = int(frozen["max_steps"])
    policy = load_policy(str(blob["policy_json"]), n_actions)
    # Fixed unlabeled pool of feature differences from training-eligible data.
    pool = diffs.copy()
    rng.shuffle(pool)
    u0, _ = confidence_set(diffs, labels, beta, grid)
    # Direction a is the executed feature change of the saved update.
    a = learned_delta
    v_mat = _info_matrix(diffs, beta)
    rows = []
    for selector in ("Uniform", "Generic uncertainty", "Policy-relevant uncertainty"):
        started = time.perf_counter()
        chosen = _select(pool, selector, labels, phi, beta, a, v_mat, rng, k=16)
        query_s = time.perf_counter() - started
        y = label_pairs(chosen, w_star, beta, rng).astype(np.float64)
        d_new = np.concatenate([diffs, chosen])
        y_new = np.concatenate([labels, y])
        u_new, _ = confidence_set(d_new, y_new, beta, grid)
        cert0 = worst_gain(learned_delta, u0)
        cert1 = worst_gain(learned_delta, u_new)
        width0 = _set_width(u0)
        width1 = _set_width(u_new)
        phi_b = fit_reward(d_new, y_new, beta, n_steps=8, step_cap=0.1, w0=phi)
        branch_rng = np.random.default_rng(900 + seed * 10 + len(rows))
        batch = collect(env, policy, branch_rng, 12, n_actions, max_steps)
        updated = policy_update(policy, batch, phi_b, low, high, lr=0.5, n_actions=n_actions)
        delta, _se = delta_mu(env, policy, updated, branch_rng, 8, n_actions, max_steps, low, high)
        rows.append({
            "task": name,
            "checkpoint": f"seed{seed}_round0",
            "selector": selector,
            "added_pairs": 16,
            "certificate_change_unchanged": cert1 - cert0,
            "uncertainty_change": width0 - width1,
            "true_finite_step_gain": float(w_star @ delta),
            "query_seconds": query_s,
            "reward_shift": float(np.linalg.norm(phi_b - phi)),
        })
    return rows


def protocol_checks() -> dict:
    """Section 2.3 checks that do not rerun the paper reproduction."""
    import hashlib
    name = "Deep-Sea-Treasure"
    frozen = json.loads((run_dir("E1", name, "frozen") / "frozen.json").read_text())
    path = run_dir("E1", name, "seed1") / "checkpoints" / "seed1_round0.npz"
    first = np.load(path, allow_pickle=False)
    second = np.load(path, allow_pickle=False)
    digest = lambda blob: hashlib.sha256(np.asarray(blob["policy_json"]).tobytes()).hexdigest()
    zero = zero_update_check(frozen)
    learner_keys = set(first.files)
    import subprocess
    baseline_log = (ROOT / "experiments/horizon_real/baseline.log").read_text() if (ROOT / "experiments/horizon_real/baseline.log").exists() else ""
    reference_clean = subprocess.run(["git", "diff", "--quiet", "--", "results/reference"], cwd=ROOT, check=False).returncode == 0
    report = {
        "pytest_passed": "38 passed" in baseline_log,
        "reproduction_reference_unchanged": reference_clean,
        "reproduction_note": "experiments/reproduce.py --all passed the figure suite and failed one inherited numeric field: adaptive_BTL.maximum_error_radius_ratio 0.46907304790731597 versus reference 0.4263795829396613. results/reference was not edited.",
        "restore_hash_match": digest(first) == digest(second),
        "restore_hash": digest(first),
        "zero_update_gain_norm": zero["gain_norm"],
        "zero_update_parameter_step_norm": zero["parameter_step_norm"],
        "zero_update_within_tolerance": abs(zero["gain_norm"]) < 1e-8 and zero["parameter_step_norm"] < 1e-8,
        "teacher_weight_absent_from_checkpoint": "w_star" not in learner_keys,
        "checkpoint_keys": sorted(learner_keys),
    }
    dest = run_dir("protocol")
    (dest / "checks.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return report


def _info_matrix(diffs: np.ndarray, beta: float) -> np.ndarray:
    # Design matrix of comparison features, regularized.
    return beta**2 * (diffs.T @ diffs) / max(len(diffs), 1) + 1e-3 * np.eye(diffs.shape[1])


def _select(pool, selector, labels, phi, beta, a, v_mat, rng, k: int) -> np.ndarray:
    if selector == "Uniform":
        idx = rng.choice(len(pool), size=min(k, len(pool)), replace=False)
        return pool[idx]
    if selector == "Generic uncertainty":
        pred = np.abs(sigmoid(beta * pool @ phi) - 0.5)
        idx = np.argsort(pred)[:k]  # closest to 0.5
        return pool[idx]
    # Policy-relevant score from the work plan: omega (a^T V^{-1} d)^2 / (1 + omega d^T V^{-1} d)
    try:
        v_inv = np.linalg.inv(v_mat)
    except np.linalg.LinAlgError:
        v_inv = np.linalg.pinv(v_mat)
    omega = 1.0
    scores = []
    for d in pool:
        vd = v_inv @ d
        va = float(a @ vd)
        scores.append(omega * va**2 / (1 + omega * float(d @ vd)))
    idx = np.argsort(scores)[-k:]
    return pool[idx]


def _set_width(weights: np.ndarray) -> float:
    if len(weights) == 0:
        return float("nan")
    return float(np.max(weights[:, 0]) - np.min(weights[:, 0]))


def _read_metrics(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    import csv
    with path.open() as handle:
        return list(csv.DictReader(handle))


def zero_update_check(frozen: dict) -> dict:
    rng = np.random.default_rng(7)
    env = _env(frozen["env_id"])
    low = np.asarray(frozen["low"])
    high = np.asarray(frozen["high"])
    n_actions = int(frozen["n_actions"])
    max_steps = int(frozen["max_steps"])
    policy: dict = {}
    episodes = collect(env, policy, rng, 8, n_actions, max_steps)
    phi = np.zeros(frozen["dim"])
    updated = policy_update(policy, episodes, phi, low, high, lr=0.0, n_actions=n_actions)
    # Same environment seeds for both evaluations. A zero step must not change the action distribution.
    mu0, _se0 = mean_feature(env, policy, np.random.default_rng(11), 6, n_actions, max_steps, low, high)
    mu1, _se1 = mean_feature(env, updated, np.random.default_rng(11), 6, n_actions, max_steps, low, high)
    delta = mu1 - mu0
    return {"parameter_step_norm": _policy_step_norm(policy, updated, n_actions), "gain_norm": float(np.linalg.norm(delta))}


def _policy_step_norm(base: dict, updated: dict, n_actions: int) -> float:
    total = 0.0
    for key in set(base) | set(updated):
        a = base.get(key, np.zeros(n_actions))
        b = updated.get(key, np.zeros(n_actions))
        total += float(np.sum((a - b) ** 2))
    return float(np.sqrt(total))
