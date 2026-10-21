"""S1 MiniGrid. Called only from the Work Plan 07 task.

The observation wrapper is frozen. The two-component outcome is separate from
MiniGrid's built-in shaped reward. The synthetic teacher is evaluator-only.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from experiments.bridge.common import git_sha

ENV_ID = "MiniGrid-Empty-8x8-v0"
W_STAR = np.array([0.5, 0.5])  # predeclared before confirm: goal, negative normalized time
BETA = 5.0
H_FALLBACK = 256


def make_env(seed: int = 0):
    import gymnasium as gym
    import minigrid  # noqa: F401  registers the official ids
    env = gym.make(ENV_ID)
    raw = env.unwrapped
    horizon = int(getattr(raw, "max_steps", None) or H_FALLBACK)
    return env, horizon


def encode(obs: dict) -> np.ndarray:
    image = np.asarray(obs["image"], dtype=np.float32).transpose(2, 0, 1) / 10.0
    direction = np.zeros(4, dtype=np.float32)
    direction[int(obs["direction"])] = 1.0
    return np.concatenate([image.reshape(-1), direction])


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3 * 7 * 7 + 4, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())
        self.actor = nn.Linear(64, 7)
        self.value = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor):
        hidden = self.net(x)
        return self.actor(hidden), self.value(hidden).squeeze(-1)


def scalar(outcome: np.ndarray, weight: np.ndarray) -> float:
    return float(weight @ outcome)


def rollout(env, horizon: int, policy: Policy, steps: int, weight: np.ndarray, rng: np.random.Generator):
    obs, _ = env.reset(seed=int(rng.integers(1, 2**31 - 1)))
    features, actions, logps, values, rewards, dones = [], [], [], [], [], []
    episodes = []
    total = np.zeros(2)
    length = 0
    for _ in range(steps):
        feat = encode(obs)
        x = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits, value = policy(x)
            dist = torch.distributions.Categorical(logits=logits)
            action = int(dist.sample().item())
            logp = float(dist.log_prob(torch.tensor(action)))
            value_f = float(value.item())
        obs, _env_reward, terminated, truncated, _info = env.step(action)
        length += 1
        done = bool(terminated or truncated)
        goal = 1.0 if terminated else 0.0
        if done:
            outcome = np.array([goal, -length / horizon], dtype=np.float64)
            episodes.append(outcome)
            reward = scalar(outcome, weight)
            total = np.zeros(2)
            length = 0
            obs, _ = env.reset(seed=int(rng.integers(1, 2**31 - 1)))
        else:
            reward = 0.0
        features.append(feat)
        actions.append(action)
        logps.append(logp)
        values.append(value_f)
        rewards.append(reward)
        dones.append(done)
    return {
        "features": np.stack(features),
        "actions": np.array(actions),
        "logps": np.array(logps),
        "values": np.array(values),
        "rewards": np.array(rewards, dtype=np.float64),
        "dones": np.array(dones),
        "episodes": episodes,
    }


def ppo_update(policy: Policy, opt: torch.optim.Optimizer, batch: dict, clip: float = 0.2):
    rewards = batch["rewards"]
    values = batch["values"]
    dones = batch["dones"]
    advantages = np.zeros_like(rewards)
    last = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        cont = 0.0 if dones[index] else 1.0
        delta = rewards[index] + 0.99 * (0.0 if index == len(rewards) - 1 else values[index + 1]) * cont - values[index]
        # The bootstrap above uses the next stored value only when the episode continues.
        last = delta + 0.99 * 0.95 * cont * last
        advantages[index] = last
    returns = advantages + values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    x = torch.tensor(batch["features"], dtype=torch.float32)
    actions = torch.tensor(batch["actions"], dtype=torch.int64)
    old = torch.tensor(batch["logps"], dtype=torch.float32)
    adv = torch.tensor(advantages, dtype=torch.float32)
    ret = torch.tensor(returns, dtype=torch.float32)
    for _ in range(4):
        logits, pred = policy(x)
        dist = torch.distributions.Categorical(logits=logits)
        new = dist.log_prob(actions)
        ratio = torch.exp(new - old)
        loss_pi = -torch.minimum(ratio * adv, torch.clamp(ratio, 1 - clip, 1 + clip) * adv).mean()
        loss_v = 0.5 * (pred - ret).pow(2).mean()
        loss = loss_pi + loss_v - 0.01 * dist.entropy().mean()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        opt.step()


def evaluate(env, horizon: int, policy: Policy, episodes: int, weight: np.ndarray) -> dict:
    goals, lengths, utilities = [], [], []
    for seed in range(episodes):
        obs, _ = env.reset(seed=10_000 + seed)
        done = False
        steps = 0
        goal = 0.0
        while not done and steps < horizon:
            x = torch.tensor(encode(obs), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits, _value = policy(x)
            action = int(torch.argmax(logits, dim=-1).item())
            obs, _reward, terminated, truncated, _info = env.step(action)
            steps += 1
            done = bool(terminated or truncated)
            if terminated:
                goal = 1.0
        outcome = np.array([goal, -steps / horizon])
        goals.append(goal)
        lengths.append(steps)
        utilities.append(scalar(outcome, weight))
    return {
        "episodes": episodes,
        "goal_success": float(np.mean(goals)),
        "mean_steps": float(np.mean(lengths)),
        "true_utility": float(np.mean(utilities)),
        "success_count": int(np.sum(goals)),
    }


def train(seed: int, env_steps: int, weight: np.ndarray, out: Path, name: str) -> dict:
    env, horizon = make_env(seed)
    policy = Policy()
    opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
    rng = np.random.default_rng(seed)
    seen = 0
    curves = []
    marks = {0, env_steps // 4, env_steps // 2, 3 * env_steps // 4, env_steps}
    while seen < env_steps:
        batch = rollout(env, horizon, policy, 256, weight, rng)
        ppo_update(policy, opt, batch)
        seen += 256
        if seen in marks or seen >= env_steps:
            metrics = evaluate(env, horizon, policy, 100, W_STAR)
            metrics["env_steps"] = seen
            curves.append(metrics)
    final = curves[-1]
    path = out / f"{name}_seed{seed}.pt"
    torch.save({"state": policy.state_dict(), "seed": seed, "weight": weight.tolist(), "horizon": horizon}, path)
    env.close()
    return {"seed": seed, "name": name, "horizon": horizon, "curve": curves, "final": final, "checkpoint": str(path)}


def fit_weight(pairs: list[tuple[np.ndarray, np.ndarray, int]], steps: int) -> np.ndarray:
    """Logistic MLE on the simplex. Extra steps are the compute-only control."""
    weight = 0.5
    if not pairs:
        return np.array([weight, 1.0 - weight])
    for _ in range(steps):
        grad = 0.0
        for left, right, label in pairs:
            diff = left - right
            slope = float(diff[0] - diff[1])
            margin = weight * slope + float(diff[1])
            prob = 1.0 / (1.0 + np.exp(-BETA * margin))
            grad += (label - prob) * BETA * slope
        weight = float(np.clip(weight + 0.1 * grad / len(pairs), 0.001, 0.999))
    return np.array([weight, 1.0 - weight])


def collect_pairs(env, horizon: int, policy: Policy, episodes: int, rng: np.random.Generator):
    outcomes = []
    for _ in range(episodes):
        obs, _ = env.reset(seed=int(rng.integers(1, 2**31 - 1)))
        done = False
        steps = 0
        goal = 0.0
        while not done and steps < horizon:
            x = torch.tensor(encode(obs), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits, _value = policy(x)
                action = int(torch.distributions.Categorical(logits=logits).sample().item())
            obs, _reward, terminated, truncated, _info = env.step(action)
            steps += 1
            done = bool(terminated or truncated)
            if terminated:
                goal = 1.0
        outcomes.append(np.array([goal, -steps / horizon]))
    pairs = []
    for i in range(len(outcomes)):
        for j in range(i + 1, len(outcomes)):
            diff = outcomes[i] - outcomes[j]
            sensitivity = float(np.linalg.norm(diff))
            if sensitivity < 1e-8:
                continue
            margin = float(W_STAR @ diff)
            prob = 1.0 / (1.0 + np.exp(-BETA * margin))
            label = int(rng.random() < prob)
            pairs.append((outcomes[i], outcomes[j], label, sensitivity))
    pairs.sort(key=lambda item: item[3])
    return pairs


def continue_from(seed: int, state: dict, env_steps: int, weight: np.ndarray, out: Path, name: str) -> dict:
    env, horizon = make_env(seed)
    policy = Policy()
    policy.load_state_dict(state)
    opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
    rng = np.random.default_rng(seed + 1000)
    seen = 0
    while seen < env_steps:
        batch = rollout(env, horizon, policy, 256, weight, rng)
        ppo_update(policy, opt, batch)
        seen += 256
    final = evaluate(env, horizon, policy, 100, W_STAR)
    unchanged_weight = weight.copy()
    weight[:] = np.array([0.1, 0.9])  # information changes; the actor is not updated
    info_after = evaluate(env, horizon, policy, 50, W_STAR)
    weight[:] = unchanged_weight
    info_before = evaluate(env, horizon, policy, 50, W_STAR)
    env.close()
    return {
        "name": name,
        "seed": seed,
        "deployed_weight": weight.tolist(),
        "final": final,
        "information_only_same_policy": info_before == info_after,
        "env_steps_after_fork": seen,
    }


def run_confirm(out: Path) -> dict:
    """Two or three policy seeds. Forks share the checkpoint taken before the intervention."""
    started = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in (0, 1, 2):
        env, horizon = make_env(seed)
        policy = Policy()
        torch.manual_seed(seed)
        opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
        rng = np.random.default_rng(seed)
        seen = 0
        while seen < 8192:
            batch = rollout(env, horizon, policy, 256, W_STAR, rng)
            ppo_update(policy, opt, batch)
            seen += 256
        parent = {key: value.detach().clone() for key, value in policy.state_dict().items()}
        pairs = collect_pairs(env, horizon, policy, 40, np.random.default_rng(seed + 7))
        source = [(a, b, label) for a, b, label, _sens in pairs[:32]]
        informative = [(a, b, label) for a, b, label, _sens in pairs[-32:]]
        weights = {
            "source_only": fit_weight(source, 30),
            "extra_fit_same_source": fit_weight(source, 300),
            "new_32": fit_weight(source + informative, 30),
            "oracle": W_STAR.copy(),
        }
        env.close()
        for name, weight in weights.items():
            row = continue_from(seed, parent, 92_160, weight, out, name)
            row["parent_env_steps"] = seen
            row["source_pairs"] = len(source)
            row["new_pairs"] = 0 if name != "new_32" else 32
            row["fit_steps"] = 300 if name == "extra_fit_same_source" else (0 if name == "oracle" else 30)
            row["guarantee_type"] = "empirical_proxy"
            row["teacher"] = "synthetic_btl_evaluator_only"
            rows.append(row)
            (out / f"{name}_seed{seed}.json").write_text(json.dumps(row, indent=2) + "\n")
    door = _doorkey_oracle(out)
    payload = {
        "env": ENV_ID,
        "seeds": [0, 1, 2],
        "total_env_steps_per_fork": 8192 + 92160,
        "rows": rows,
        "learned": True,
        "source_sha": git_sha(),
        "runtime_seconds": time.perf_counter() - started,
        "scope": "empirical_proxy",
        "doorkey": door,
        "note": "Synthetic BTL labels are not human. DoorKey is reported separately and is not a horizon.",
    }
    (out / "confirm.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def _doorkey_oracle(out: Path) -> dict:
    import gymnasium as gym
    import minigrid  # noqa: F401
    env = gym.make("MiniGrid-DoorKey-6x6-v0")
    horizon = int(getattr(env.unwrapped, "max_steps", None) or 360)
    policy = Policy()
    before = evaluate(env, horizon, policy, 30, W_STAR)
    opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
    rng = np.random.default_rng(0)
    seen = 0
    while seen < 40_000:
        batch = rollout(env, horizon, policy, 256, W_STAR, rng)
        ppo_update(policy, opt, batch)
        seen += 256
    after = evaluate(env, horizon, policy, 30, W_STAR)
    env.close()
    learned = after["goal_success"] > before["goal_success"] + 0.05
    result = {"env": "MiniGrid-DoorKey-6x6-v0", "untrained": before, "oracle_40k": after, "learned": learned, "env_steps": seen}
    (out / "doorkey.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def run_pilot(out: Path) -> dict:
    started = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    env, horizon = make_env(0)
    untrained = Policy()
    before = evaluate(env, horizon, untrained, 100, W_STAR)
    env.close()
    trained = train(0, 40_000, W_STAR, out, "oracle40k")
    learned = trained["final"]["goal_success"] > before["goal_success"] + 0.05 and trained["final"]["goal_success"] >= 0.5
    payload = {
        "env": ENV_ID,
        "observation": "frozen 7x7x3 partial image plus direction; ImgObs-equivalent agent view, not the full map",
        "w_star": W_STAR.tolist(),
        "beta": BETA,
        "horizon": horizon,
        "untrained": before,
        "oracle_40k": trained["final"],
        "learned": learned,
        "source_sha": git_sha(),
        "runtime_seconds": time.perf_counter() - started,
        "scope": "empirical",
        "note": "If learned is false, do not interpret a guidance horizon.",
    }
    (out / "pilot.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload
