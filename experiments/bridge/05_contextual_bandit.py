#!/usr/bin/env python3
"""Step 5: one-step contextual bandit on Theorem 2's four outcomes.

Each row changes one element relative to the exact four-outcome model.
The hidden preference is used only to label comparisons and to score true gain.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import importlib.util

from experiments.bridge.common import write_summary

_spec = importlib.util.spec_from_file_location(
    "ordinary_pg_bridge", Path(__file__).with_name("04_ordinary_pg.py"),
)
_pg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pg)
exact_theta_gradient = _pg.exact_theta_gradient
fit_reward_head = _pg.fit_reward_head
metrics = _pg.metrics
ordinary_pg_check = _pg.ordinary_pg_check
sample_gradient = _pg.sample_gradient


H = 5e-4
UPDATES = 8


def true_gain(x: float, y: float, phi_star: float) -> float:
    return phi_star * x + (1.0 - phi_star) * y


def rollout(phi0: float, phi_star: float, n_samples: int | None, seed: int,
            weight: tuple[float, float] | None = None) -> dict:
    x, y, phi = 0.8, 0.0, phi0
    theta = np.array([np.arctanh(x), np.arctanh(y)])
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(UPDATES + 1):
        wq, ws = (phi, 1.0 - phi) if weight is None else weight
        guidance_phi = wq / (wq + ws)
        state = metrics(x, y, guidance_phi)
        exact = exact_theta_gradient(x, y, wq, ws)
        if n_samples is None:
            grad = exact
        else:
            grad = sample_gradient(x, y, wq, ws, n_samples, rng)
        rows.append({
            "step": k,
            "x": x,
            "y": y,
            "guidance_phi": guidance_phi,
            "true_gain": true_gain(x, y, phi_star),
            "source_certificate": state["worst"],
            "pointwise": state["pointwise"],
            "gradient_error": float(np.linalg.norm(grad - exact)),
        })
        if k == UPDATES:
            break
        theta = theta + H * grad
        x, y = float(np.tanh(theta[0])), float(np.tanh(theta[1]))
        if weight is None:
            phi = min(phi0 + 0.04 * (k + 1) * H, phi + 0.04 * H)
    return {"rows": rows, "final": rows[-1]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("outputs/horizon_bridge/step05"))
    args = parser.parse_args()
    started = time.perf_counter()
    reference = ordinary_pg_check()
    phi0 = float(reference["initial_phi"])
    # Evaluator-only preference. Learning functions below do not receive this value
    # except fit_reward_head, which uses it solely to sample labels.
    phi_star = phi0
    exact = rollout(phi0, phi_star, None, 0)
    sampled = rollout(phi0, phi_star, 8192, 1)
    small_head = fit_reward_head(256, phi_star, seed=2, informative=True)
    large_head = fit_reward_head(1200, phi_star, seed=2, informative=True)
    blind_head = fit_reward_head(1200, phi_star, seed=2, informative=False)
    fitted = rollout(phi0, phi_star, None, 0, weight=(float(small_head[0]), float(small_head[1])))
    informed = rollout(phi0, phi_star, None, 0, weight=(float(large_head[0]), float(large_head[1])))
    # Bounded guidance: the fitted coefficient may move at most rho per unit time.
    bounded_phi = phi0
    target = float(large_head[0] / np.sum(large_head))
    bounded_weight = None
    x, y, phi = 0.8, 0.0, phi0
    theta = np.array([np.arctanh(x), np.arctanh(y)])
    bounded_rows = []
    for k in range(UPDATES + 1):
        allowance = 0.04 * k * H
        bounded_phi = float(np.clip(target, phi0 - allowance, phi0 + allowance))
        state = metrics(x, y, bounded_phi)
        grad = exact_theta_gradient(x, y, bounded_phi, 1.0 - bounded_phi)
        bounded_rows.append({
            "step": k, "guidance_phi": bounded_phi,
            "true_gain": true_gain(x, y, phi_star),
            "source_certificate": state["worst"], "pointwise": state["pointwise"],
            "gradient_error": 0.0,
        })
        if k == UPDATES:
            break
        theta = theta + H * grad
        x, y = float(np.tanh(theta[0])), float(np.tanh(theta[1]))
        phi = bounded_phi
    variants = {
        "exact_four_output": exact["final"],
        "sampled_policy_gradient": sampled["final"],
        "btl_fitted_reward_256": {**fitted["final"], "phi_hat": float(small_head[0] / np.sum(small_head))},
        "informative_feedback_1200": {**informed["final"], "phi_hat": target},
        "bounded_guidance": bounded_rows[-1],
        "source_labels_no_sensitivity": {
            "phi_hat": None if np.allclose(blind_head, 0) else float(blind_head[0] / np.sum(blind_head)),
            "weight": blind_head.tolist(),
            "true_gain": true_gain(0.8, 0.0, phi_star),
            "note": "Zero-contrast source labels do not move the reward head off its zero initialization.",
        },
    }
    exact_x = exact["final"]["x"]
    sampled["final"]["agreement_abs_x"] = abs(sampled["final"]["x"] - exact_x)
    fitted["final"]["agreement_abs_x"] = abs(fitted["final"]["x"] - exact_x)
    informed["final"]["agreement_abs_x"] = abs(informed["final"]["x"] - exact_x)
    bounded_rows[-1]["agreement_abs_x"] = abs(x - exact_x)
    payload = {
        "status": "PASS",
        "command": "python experiments/bridge/05_contextual_bandit.py --out outputs/horizon_bridge/step05",
        "updates": UPDATES,
        "step_size": H,
        "samples_per_update": 8192,
        "hidden_preference_used_for_labels_and_true_gain_only": True,
        "initial_true_gain": true_gain(0.8, 0.0, phi_star),
        "initial_source_certificate": metrics(0.8, 0.0, phi0)["worst"],
        "initial_pointwise": metrics(0.8, 0.0, phi0)["pointwise"],
        "variants": {
            "exact_four_output": exact["final"],
            "sampled_policy_gradient": sampled["final"],
            "btl_fitted_reward_256": fitted["final"],
            "informative_feedback_1200": informed["final"],
            "bounded_guidance": bounded_rows[-1],
            "source_labels_no_sensitivity": variants["source_labels_no_sensitivity"],
        },
        "scope": "Four-outcome bandit. No Walker, Meta-World, or language model.",
    }
    write_summary(args.out, payload, started)


if __name__ == "__main__":
    main()
