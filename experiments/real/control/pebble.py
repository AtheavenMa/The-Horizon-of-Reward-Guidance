"""SAC with a Bradley–Terry reward ensemble.

The environment reward is used only by the teacher and by held-out evaluation.
The actor and critics train on the ensemble reward.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import softplus

from experiments.real.common.records import run_dir, write_run


def flat_obs(obs: dict) -> np.ndarray:
    return np.concatenate([np.asarray(obs[k], dtype=np.float32).ravel() for k in obs])


class MLP(nn.Module):
    def __init__(self, din: int, dout: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(din, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, dout),
        )

    def forward(self, x):
        return self.net(x)


class Actor(nn.Module):
    def __init__(self, din: int, act: int):
        super().__init__()
        self.body = MLP(din, 2 * act)

    def dist(self, obs):
        mu, raw = self.body(obs).chunk(2, dim=-1)
        scale = softplus(raw) + 1e-4
        return mu, scale

    def sample(self, obs):
        mu, scale = self.dist(obs)
        eps = torch.randn_like(mu)
        pre = mu + scale * eps
        action = torch.tanh(pre)
        logp = (-0.5 * ((pre - mu) / scale).pow(2) - torch.log(scale) - 0.5 * np.log(2 * np.pi)).sum(-1)
        logp = logp - torch.log(1 - action.pow(2) + 1e-6).sum(-1)
        return action, logp


class TwinQ(nn.Module):
    def __init__(self, din: int, act: int):
        super().__init__()
        self.q1 = MLP(din + act, 1)
        self.q2 = MLP(din + act, 1)

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


def train_preference_sac(domain: str, task: str, steps: int, seed: int, mode: str, device: str = "cuda") -> dict:
    from dm_control import suite
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env = suite.load(domain, task, task_kwargs={"random": seed})
    eval_env = suite.load(domain, task, task_kwargs={"random": seed + 1000})
    spec = env.action_spec()
    ts = env.reset()
    obs_dim = flat_obs(ts.observation).shape[0]
    act_dim = spec.shape[0]
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    actor = Actor(obs_dim, act_dim).to(dev)
    critic = TwinQ(obs_dim, act_dim).to(dev)
    target = TwinQ(obs_dim, act_dim).to(dev)
    target.load_state_dict(critic.state_dict())
    reward_nets = nn.ModuleList([MLP(obs_dim + act_dim, 1) for _ in range(3)]).to(dev)
    opt_a = torch.optim.Adam(actor.parameters(), lr=3e-4)
    opt_c = torch.optim.Adam(critic.parameters(), lr=3e-4)
    opt_r = torch.optim.Adam(reward_nets.parameters(), lr=3e-4)
    capacity = 50_000
    buf_s = np.zeros((capacity, obs_dim), np.float32)
    buf_a = np.zeros((capacity, act_dim), np.float32)
    buf_sp = np.zeros((capacity, obs_dim), np.float32)
    buf_true = np.zeros(capacity, np.float32)
    buf_done = np.zeros(capacity, np.float32)
    n = 0
    pairs: list[tuple[np.ndarray, np.ndarray, float]] = []
    segment: list[tuple[np.ndarray, np.ndarray]] = []
    pending = None
    obs = flat_obs(ts.observation)
    ep_ret = 0.0
    eval_curve = []
    log_alpha = torch.tensor(0.0, device=dev, requires_grad=True)
    opt_alpha = torch.optim.Adam([log_alpha], lr=3e-4)
    target_entropy = -float(act_dim)
    for t in range(steps):
        if t < 1000:
            action = rng.uniform(spec.minimum, spec.maximum).astype(np.float32)
        else:
            with torch.no_grad():
                a, _ = actor.sample(torch.as_tensor(obs, device=dev).unsqueeze(0))
            action = a.squeeze(0).cpu().numpy()
            action = np.clip(action, spec.minimum, spec.maximum)
        ts = env.step(action)
        true_r = 0.0 if ts.reward is None else float(ts.reward)
        nxt = flat_obs(ts.observation)
        done = float(ts.last())
        idx = n % capacity
        buf_s[idx], buf_a[idx], buf_sp[idx], buf_true[idx], buf_done[idx] = obs, action, nxt, true_r, done
        n += 1
        segment.append((obs.copy(), action.copy()))
        ep_ret += true_r
        if len(segment) == 16:
            if pending is None:
                pending = segment
            else:
                # Teacher sees true returns of the stored transitions. The learner receives only the bit.
                label = _segment_label(pending, segment, buf_true, n, capacity)
                pairs.append((np.stack([s for s, _a in pending]), np.stack([s for s, _a in segment]), label))
                # store actions too
                pairs[-1] = (
                    np.concatenate([np.stack([s for s, _a in pending]), np.stack([a for _s, a in pending])], axis=1),
                    np.concatenate([np.stack([s for s, _a in segment]), np.stack([a for _s, a in segment])], axis=1),
                    label,
                )
                if mode == "extra_labels" and n >= 48:
                    old_idx = np.arange(n - 48, n - 32) % capacity
                    new_idx = np.arange(n - 16, n) % capacity
                    pairs.append((
                        np.concatenate([buf_s[old_idx], buf_a[old_idx]], axis=1),
                        np.concatenate([buf_s[new_idx], buf_a[new_idx]], axis=1),
                        1.0 if buf_true[new_idx].sum() > buf_true[old_idx].sum() else 0.0,
                    ))
                pending = None
            segment = []
        obs = nxt
        if ts.last():
            ts = env.reset()
            obs = flat_obs(ts.observation)
            ep_ret = 0.0
        if t >= 1000 and t % 2 == 0 and n > 256:
            _sac_update(actor, critic, target, reward_nets, opt_a, opt_c, buf_s, buf_a, buf_sp, buf_done, n, capacity, dev, log_alpha, opt_alpha, target_entropy)
        query_every = 400 if mode == "extra_labels" else 800
        if t > 0 and t % query_every == 0 and len(pairs) > 8:
            epochs = 1
            if mode == "extra_updates":
                epochs = 4
            elif mode == "diagnosis":
                disagree = _disagreement(reward_nets, pairs[-8:], dev)
                epochs = 1 if disagree > 0.05 else 4
            _reward_update(reward_nets, opt_r, pairs, dev, epochs=epochs)
        if t > 0 and t % 2000 == 0:
            score = _evaluate(eval_env, actor, spec, dev, episodes=3)
            eval_curve.append({"step": t, "true_return": score, "pairs": len(pairs)})
    final = _evaluate(eval_env, actor, spec, dev, episodes=5)
    out = run_dir("E4", f"{domain}_{task}", mode, f"seed{seed}")
    rows = eval_curve + [{"step": steps, "true_return": final, "pairs": len(pairs), "seed": seed, "mode": mode}]
    write_run(
        out,
        {"domain": domain, "task": task, "steps": steps, "seed": seed, "mode": mode, "task_id": f"dm_control:{domain}-{task}"},
        rows,
        [{"type": "pairs", "n": len(pairs)}],
        {"status": "complete", "final_true_return": final},
    )
    ckpt = out / "checkpoints" / "agent.pt"
    torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(), "reward": reward_nets.state_dict()}, ckpt)
    return {"final_true_return": final, "pairs": len(pairs), "curve": eval_curve, "path": str(out)}


def _segment_label(seg_a, seg_b, buf_true, n, capacity) -> float:
    # Compare the mean stored environment reward over the latest transitions of equal length.
    # This uses the evaluator buffer, not the policy loss.
    length = len(seg_a)
    if n < 2 * length:
        return 1.0
    # The two segments were just written, so they occupy the last 2*length slots.
    end = n
    b = buf_true[(np.arange(end - length, end)) % capacity].sum()
    a = buf_true[(np.arange(end - 2 * length, end - length)) % capacity].sum()
    return 1.0 if b > a else 0.0


def _reward_update(nets, opt, pairs, dev, epochs: int) -> None:
    if len(pairs) < 4:
        return
    opt.zero_grad()
    take = pairs[-64:]
    loss = 0
    for _ in range(epochs):
        opt.zero_grad()
        total = 0
        for sa, sb, y in take:
            a = torch.as_tensor(sa, dtype=torch.float32, device=dev)
            b = torch.as_tensor(sb, dtype=torch.float32, device=dev)
            pred_a = torch.stack([net(a).sum() for net in nets]).mean()
            pred_b = torch.stack([net(b).sum() for net in nets]).mean()
            logit = pred_b - pred_a
            target = torch.tensor(y, device=dev)
            total = total + torch.nn.functional.binary_cross_entropy_with_logits(logit, target)
        (total / len(take)).backward()
        opt.step()
        loss = float(total.detach())
    _ = loss


def _disagreement(nets, pairs, dev) -> float:
    vals = []
    with torch.no_grad():
        for sa, sb, _y in pairs:
            a = torch.as_tensor(sa, dtype=torch.float32, device=dev)
            scores = torch.stack([net(a).sum() for net in nets])
            vals.append(float(scores.var()))
    return float(np.mean(vals)) if vals else 0.0


def _predicted_reward(nets, obs, action) -> torch.Tensor:
    x = torch.cat([obs, action], dim=-1)
    preds = torch.stack([net(x).squeeze(-1) for net in nets])
    return preds.mean(0)


def _sac_update(actor, critic, target, reward_nets, opt_a, opt_c, buf_s, buf_a, buf_sp, buf_done, n, capacity, dev, log_alpha, opt_alpha, target_entropy):
    size = min(n, capacity)
    idx = np.random.randint(0, size, size=256)
    obs = torch.as_tensor(buf_s[idx], device=dev)
    action = torch.as_tensor(buf_a[idx], device=dev)
    nxt = torch.as_tensor(buf_sp[idx], device=dev)
    done = torch.as_tensor(buf_done[idx], device=dev)
    with torch.no_grad():
        reward = _predicted_reward(reward_nets, obs, action)
        next_a, next_logp = actor.sample(nxt)
        q1_t, q2_t = target(nxt, next_a)
        alpha = log_alpha.exp()
        backup = reward + (1 - done) * 0.99 * (torch.min(q1_t, q2_t) - alpha * next_logp)
    q1, q2 = critic(obs, action)
    loss_c = ((q1 - backup).pow(2) + (q2 - backup).pow(2)).mean()
    opt_c.zero_grad()
    loss_c.backward()
    opt_c.step()
    new_a, logp = actor.sample(obs)
    q1_pi, q2_pi = critic(obs, new_a)
    loss_a = (log_alpha.exp().detach() * logp - torch.min(q1_pi, q2_pi)).mean()
    opt_a.zero_grad()
    loss_a.backward()
    opt_a.step()
    loss_alpha = -(log_alpha * (logp.detach() + target_entropy)).mean()
    opt_alpha.zero_grad()
    loss_alpha.backward()
    opt_alpha.step()
    with torch.no_grad():
        for p, tp in zip(critic.parameters(), target.parameters()):
            tp.mul_(0.995).add_(p, alpha=0.005)


def _evaluate(env, actor, spec, dev, episodes: int) -> float:
    scores = []
    for _ in range(episodes):
        ts = env.reset()
        total = 0.0
        while not ts.last():
            obs = torch.as_tensor(flat_obs(ts.observation), device=dev).unsqueeze(0)
            with torch.no_grad():
                mu, _scale = actor.dist(obs)
                action = torch.tanh(mu).squeeze(0).cpu().numpy()
            action = np.clip(action, spec.minimum, spec.maximum)
            ts = env.step(action)
            total += 0.0 if ts.reward is None else float(ts.reward)
        scores.append(total)
    return float(np.mean(scores))


def run_e4(stage: str) -> dict:
    steps = 4000 if stage == "pilot" else 30000
    seeds = [1] if stage == "pilot" else [1, 2, 3]
    modes = ["normal"] if stage == "pilot" else ["normal", "extra_updates", "extra_labels", "diagnosis"]
    results = []
    for mode in modes:
        for seed in seeds:
            print(f"E4 walker {mode} seed {seed} steps {steps}", flush=True)
            row = train_preference_sac("walker", "walk", steps, seed, mode)
            row.update({"mode": mode, "seed": seed, "domain": "walker"})
            results.append({k: v for k, v in row.items() if k != "curve"})
            print(row["final_true_return"], "pairs", row["pairs"], flush=True)
    if stage == "confirm":
        for seed in (1, 2):
            print(f"E4 cheetah normal/diagnosis seed {seed}", flush=True)
            for mode in ("normal", "diagnosis"):
                row = train_preference_sac("cheetah", "run", steps, seed, mode)
                row.update({"mode": mode, "seed": seed, "domain": "cheetah"})
                results.append({k: v for k, v in row.items() if k != "curve"})
                print("cheetah", mode, row["final_true_return"], flush=True)
    dest = run_dir("E4")
    (dest / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    return {"runs": results}
