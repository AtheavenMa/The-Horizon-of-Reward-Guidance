"""F1 MiniGrid control with the parent optimizer restored.

DoorKey stays the archived NOT LEARNED result. No new sweep.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.bridge.common import git_sha

ROOT = Path(__file__).resolve().parents[2]


def _grid():
    spec = importlib.util.spec_from_file_location("grid07", ROOT / "experiments/large_scale/07_minigrid.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parent(grid, seed: int, steps: int):
    env, horizon = grid.make_env(seed)
    policy = grid.Policy()
    torch.manual_seed(seed)
    opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
    rng = np.random.default_rng(seed)
    seen = 0
    while seen < steps:
        batch = grid.rollout(env, horizon, policy, 256, grid.W_STAR, rng)
        grid.ppo_update(policy, opt, batch)
        seen += 256
    return env, horizon, policy, opt, rng, seen


def _continue(grid, env, horizon, policy, opt, rng, steps: int, weight: np.ndarray):
    seen = 0
    while seen < steps:
        batch = grid.rollout(env, horizon, policy, 256, weight, rng)
        grid.ppo_update(policy, opt, batch)
        seen += 256
    final = grid.evaluate(env, horizon, policy, 100, grid.W_STAR)
    return final, seen


def run(out: Path, stage: str = "confirm") -> dict:
    started = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    grid = _grid()
    parent_steps = 1024 if stage == "pilot" else 8192
    fork_steps = 1024 if stage == "pilot" else 92160
    seeds = [0] if stage == "pilot" else [0, 1]
    rows = []
    archived = json.loads((ROOT / "experiments/horizon_scale_07/minigrid/doorkey.json").read_text())
    door = {
        "env": "MiniGrid-DoorKey-6x6-v0",
        "status": "NOT LEARNED",
        "learned": False,
        "source": "experiments/horizon_scale_07/minigrid/doorkey.json",
        "oracle_40k_goal_success": archived.get("oracle_40k", {}).get("goal_success"),
        "claim": "DoorKey is removed from any Horizon claim. No new sweep was run.",
    }
    (out / "doorkey.json").write_text(json.dumps(door, indent=2) + "\n")
    for seed in seeds:
        done = out / f"seed{seed}.json"
        if done.exists() and stage == "confirm":
            rows.extend(json.loads(done.read_text()))
            continue
        env, horizon, policy, opt, rng, seen = _parent(grid, seed, parent_steps)
        parent_policy = copy.deepcopy(policy.state_dict())
        parent_opt = copy.deepcopy(opt.state_dict())
        parent_rng = copy.deepcopy(rng.bit_generator.state)
        pairs = grid.collect_pairs(env, horizon, policy, 20 if stage == "pilot" else 40, np.random.default_rng(seed + 7))
        source = [(a, b, label) for a, b, label, _sens in pairs[:32]]
        informative = [(a, b, label) for a, b, label, _sens in pairs[-32:]]
        weights = {
            "oracle": grid.W_STAR.copy(),
            "source_only": grid.fit_weight(source, 30),
            "new_32": grid.fit_weight(source + informative, 30),
            "extra_fit_same_source": grid.fit_weight(source, 300),
        }
        seed_rows = []
        for name, weight in weights.items():
            policy.load_state_dict(parent_policy)
            opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
            opt.load_state_dict(parent_opt)
            rng.bit_generator.state = parent_rng
            final, used = _continue(grid, env, horizon, policy, opt, rng, fork_steps, weight)
            row = {
                "name": name,
                "seed": seed,
                "optimizer_restored": True,
                "parent_env_steps": seen,
                "fork_env_steps": used,
                "deployed_weight": weight.tolist(),
                "final": final,
                "new_pairs": 32 if name == "new_32" else 0,
                "guarantee_type": "empirical_proxy",
                "teacher": "synthetic_btl_evaluator_only",
            }
            seed_rows.append(row)
            print(name, seed, final["goal_success"], final["mean_steps"], flush=True)
        env.close()
        (out / f"seed{seed}.json").write_text(json.dumps(seed_rows, indent=2) + "\n")
        rows.extend(seed_rows)
    diagnosis = {"status": "not_run"}
    if stage == "confirm":
        archived_weight = np.array([0.230, 0.770])
        env, horizon, policy, opt, rng, seen = _parent(grid, 2, parent_steps)
        parent_policy = copy.deepcopy(policy.state_dict())
        parent_opt = copy.deepcopy(opt.state_dict())
        pairs = grid.collect_pairs(env, horizon, policy, 40, np.random.default_rng(9))
        source = [(a, b, label) for a, b, label, _sens in pairs[:32]]
        fitted = grid.fit_weight(source, 300)
        diagnosis_rows = []
        for name, weight in (("archived_seed2_weight", archived_weight), ("refit_same_labels", fitted)):
            policy.load_state_dict(parent_policy)
            fresh = torch.optim.Adam(policy.parameters(), lr=3e-4)
            fresh.load_state_dict(parent_opt)
            final, used = _continue(grid, env, horizon, policy, fresh, np.random.default_rng(2), 8192, weight)
            diagnosis_rows.append({"name": name, "weight": weight.tolist(), "fork_env_steps": used, "final": final, "optimizer_restored": True})
        env.close()
        diagnosis = {
            "status": "short_diagnostic",
            "note": "WP07 extra-fit seed 2 reached 0% with weight [0.230, 0.770] and a reinitialized optimizer. This probe restores the optimizer and compares that archived weight with a refit on 8192 steps, not a full 92160-step seed.",
            "rows": diagnosis_rows,
        }
        (out / "seed2_diagnosis.json").write_text(json.dumps(diagnosis, indent=2) + "\n")
    payload = {
        "status": "complete",
        "stage": stage,
        "rows": rows,
        "doorkey": door,
        "seed2_diagnosis": diagnosis,
        "runtime_seconds": time.perf_counter() - started,
        "source_sha": git_sha(),
        "guarantee_type": "empirical_proxy",
        "passed": True,
    }
    (out / f"{stage}.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload
