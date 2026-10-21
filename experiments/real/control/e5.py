"""Meta-World Push and Drawer Open with the same preference SAC pieces as E4.

Task names come from the installed Meta-World package.
"""
from __future__ import annotations

import json

import numpy as np
import torch

from experiments.real.common.records import run_dir, write_run
from experiments.real.control.pebble import Actor, MLP, TwinQ, _disagreement, _reward_update, _sac_update


def run_e5(stage: str) -> dict:
    try:
        import metaworld
    except Exception as exc:
        dest = run_dir("E5")
        status = {"status": "BLOCKED", "reason": f"metaworld import failed: {type(exc).__name__}: {exc}"}
        write_run(dest, {"stage": stage}, [], [], status)
        (dest / "summary.json").write_text(json.dumps(status, indent=2) + "\n")
        print(status, flush=True)
        return status
    names = _task_names(metaworld)
    push = _pick(names, ("push-v3", "push-v2", "push"))
    drawer = _pick(names, ("drawer-open-v3", "drawer-open-v2", "drawer-open", "drawer"))
    print("metaworld tasks", push, drawer, flush=True)
    steps = 2000 if stage == "pilot" else 6000
    seeds = [1] if stage == "pilot" else [1, 2, 3]
    results = []
    plan = [
        (push, ["baseline", "extra_updates", "extra_feedback", "diagnosis"]),
        (drawer, ["baseline", "extra_feedback"]),
    ]
    for task_name, methods in plan:
        if task_name is None:
            results.append({"task": None, "status": "BLOCKED", "reason": "task id missing", "available": names[:12]})
            continue
        for method in methods:
            for seed in seeds:
                row = _run_task(metaworld, task_name, method, seed, steps)
                results.append(row)
                print(row["task"], row["method"], row["success"], row["return"], flush=True)
    dest = run_dir("E5")
    (dest / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    write_run(dest, {"steps": steps, "push": push, "drawer": drawer}, results, [], {"status": "complete"})
    return {"runs": results}


def _task_names(metaworld) -> list[str]:
    for attr in ("ALL_V3_ENVIRONMENTS", "ALL_V2_ENVIRONMENTS"):
        table = getattr(metaworld, attr, None)
        if isinstance(table, dict) and table:
            return sorted(table)
    try:
        import metaworld.envs.mujoco.env_dict as env_dict
        for attr in ("ALL_V3_ENVIRONMENTS", "ALL_V2_ENVIRONMENTS"):
            table = getattr(env_dict, attr, None)
            if isinstance(table, dict) and table:
                return sorted(table)
    except Exception:
        pass
    return []


def _pick(names: list[str], needles: tuple[str, ...]) -> str | None:
    for needle in needles:
        exact = [name for name in names if name == needle]
        if exact:
            return exact[0]
    return None


def _make(metaworld, task_name: str, seed: int):
    benchmark = metaworld.ML1(task_name, seed=seed)
    env = benchmark.train_classes[task_name]()
    env.set_task(benchmark.train_tasks[seed % len(benchmark.train_tasks)])
    return env


def _reset(env, seed: int):
    if hasattr(env, "reset"):
        try:
            out = env.reset(seed=seed)
        except TypeError:
            out = env.reset()
    else:
        out = env.reset()
    return out[0] if isinstance(out, tuple) else out


def _run_task(metaworld, task_name: str, method: str, seed: int, steps: int) -> dict:
    env = _make(metaworld, task_name, seed)
    eval_env = _make(metaworld, task_name, seed + 1000)
    obs = np.asarray(_reset(env, seed), dtype=np.float32).reshape(-1)
    obs_dim = obs.shape[0]
    act_dim = int(np.asarray(env.action_space.shape).reshape(-1)[0])
    low = np.asarray(env.action_space.low, dtype=np.float32)
    high = np.asarray(env.action_space.high, dtype=np.float32)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    actor = Actor(obs_dim, act_dim).to(dev)
    critic = TwinQ(obs_dim, act_dim).to(dev)
    target = TwinQ(obs_dim, act_dim).to(dev)
    target.load_state_dict(critic.state_dict())
    reward_nets = torch.nn.ModuleList([MLP(obs_dim + act_dim, 1) for _ in range(3)]).to(dev)
    opt_a = torch.optim.Adam(actor.parameters(), lr=3e-4)
    opt_c = torch.optim.Adam(critic.parameters(), lr=3e-4)
    opt_r = torch.optim.Adam(reward_nets.parameters(), lr=3e-4)
    capacity = 20_000
    buf_s = np.zeros((capacity, obs_dim), np.float32)
    buf_a = np.zeros((capacity, act_dim), np.float32)
    buf_sp = np.zeros((capacity, obs_dim), np.float32)
    buf_true = np.zeros(capacity, np.float32)
    buf_done = np.zeros(capacity, np.float32)
    n = 0
    pairs = []
    segment = []
    pending = None
    log_alpha = torch.tensor(0.0, device=dev, requires_grad=True)
    opt_alpha = torch.optim.Adam([log_alpha], lr=3e-4)
    target_entropy = -float(act_dim)
    for t in range(steps):
        if t < 400:
            action = rng.uniform(low, high).astype(np.float32)
        else:
            with torch.no_grad():
                sample, _ = actor.sample(torch.as_tensor(obs, device=dev).unsqueeze(0))
            action = np.clip(sample.squeeze(0).cpu().numpy(), low, high)
        step = env.step(action)
        if len(step) == 5:
            nxt, reward, terminated, truncated, info = step
            done = bool(terminated or truncated)
        else:
            nxt, reward, done, info = step
            done = bool(done)
        nxt = np.asarray(nxt, dtype=np.float32).reshape(-1)
        idx = n % capacity
        buf_s[idx], buf_a[idx], buf_sp[idx] = obs, action, nxt
        buf_true[idx] = float(reward)
        buf_done[idx] = float(done)
        n += 1
        segment.append((obs.copy(), action.copy()))
        if len(segment) == 16:
            if pending is None:
                pending = segment
            else:
                label = 1.0 if buf_true[(np.arange(n - 16, n)) % capacity].sum() > buf_true[(np.arange(n - 32, n - 16)) % capacity].sum() else 0.0
                left = np.concatenate([np.stack([s for s, _a in pending]), np.stack([a for _s, a in pending])], axis=1)
                right = np.concatenate([np.stack([s for s, _a in segment]), np.stack([a for _s, a in segment])], axis=1)
                pairs.append((left, right, label))
                pending = None
            segment = []
        obs = nxt
        if done:
            obs = np.asarray(_reset(env, seed + t + 1), dtype=np.float32).reshape(-1)
        if t >= 400 and t % 2 == 0 and n > 256:
            _sac_update(actor, critic, target, reward_nets, opt_a, opt_c, buf_s, buf_a, buf_sp, buf_done, n, capacity, dev, log_alpha, opt_alpha, target_entropy)
        query_every = 300 if method == "extra_feedback" else 600
        if t > 0 and t % query_every == 0 and len(pairs) > 8:
            epochs = 4 if method == "extra_updates" else 1
            if method == "diagnosis":
                epochs = 1 if _disagreement(reward_nets, pairs[-8:], dev) > 0.05 else 4
            _reward_update(reward_nets, opt_r, pairs, dev, epochs=epochs)
    score, success = _evaluate(eval_env, actor, dev, seed + 1000)
    out = run_dir("E5", task_name, method, f"seed{seed}")
    write_run(
        out,
        {"task": task_name, "method": method, "seed": seed, "steps": steps},
        [{"return": score, "success": success, "pairs": len(pairs)}],
        [{"type": "ai_teacher", "pairs": len(pairs)}],
        {"status": "complete"},
    )
    return {"task": task_name, "method": method, "seed": seed, "steps": steps, "pairs": len(pairs), "success": success, "return": score}


def _evaluate(env, actor, dev, seed: int) -> tuple[float, float]:
    returns, successes = [], []
    for ep in range(3):
        obs = np.asarray(_reset(env, seed + ep), dtype=np.float32).reshape(-1)
        total, success = 0.0, 0.0
        for _ in range(80):
            with torch.no_grad():
                mu, _scale = actor.dist(torch.as_tensor(obs, device=dev).unsqueeze(0))
                action = torch.tanh(mu).squeeze(0).cpu().numpy()
            action = np.clip(action, env.action_space.low, env.action_space.high)
            step = env.step(action)
            if len(step) == 5:
                obs, reward, terminated, truncated, info = step
                done = terminated or truncated
            else:
                obs, reward, done, info = step
            obs = np.asarray(obs, dtype=np.float32).reshape(-1)
            total += float(reward)
            success = max(success, float(info.get("success", 0.0)))
            if done:
                break
        returns.append(total)
        successes.append(success)
    return float(np.mean(returns)), float(np.mean(successes))
