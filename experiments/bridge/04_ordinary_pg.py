#!/usr/bin/env python3
"""Step 4: finite ordinary Euclidean PG, then a sampled score-function gradient.

The continuous reference is experiments/strengthening_checks.py. Sampling is
the only change in the later rows.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from experiments.bridge.common import write_summary
from experiments.strengthening_checks import ordinary_pg_check


GAMMA = 0.06
RHO = 0.04
MARGIN = 0.001
BOX = {"x": (0.795, 0.81), "y": (-0.005, 0.02), "phi": (0.835, 0.86)}


def omega(z: float) -> float:
    return (1.0 - z * z) ** 2


def metrics(x: float, y: float, phi: float) -> dict:
    a, b = omega(x), omega(y)
    lower = (b + GAMMA) / (2 * a + b)
    upper = (2 * b - GAMMA) / (a + 2 * b)
    worst = min(2 * a * phi - b * (1 - phi), -a * phi + 2 * b * (1 - phi))
    return {
        "omega_q": a, "omega_s": b, "lower": lower, "upper": upper,
        "worst": worst, "pointwise": a * b / (a + b),
        "inside_box": BOX["x"][0] <= x <= BOX["x"][1] and BOX["y"][0] <= y <= BOX["y"][1]
        and BOX["phi"][0] <= phi <= BOX["phi"][1],
    }


def exact_theta_gradient(x: float, y: float, weight_q: float, weight_s: float) -> np.ndarray:
    return np.array([weight_q * (1.0 - x * x), weight_s * (1.0 - y * y)])


def enumerated_gradient(x: float, y: float, weight_q: float, weight_s: float) -> np.ndarray:
    total = np.zeros(2)
    for q in (-1.0, 1.0):
        for s in (-1.0, 1.0):
            probability = 0.25 * (1.0 + x * q) * (1.0 + y * s)
            reward = weight_q * q + weight_s * s
            score = np.array([
                (1.0 - x * x) * q / (1.0 + x * q),
                (1.0 - y * y) * s / (1.0 + y * s),
            ])
            total += probability * reward * score
    return total


def sample_gradient(x: float, y: float, weight_q: float, weight_s: float,
                    n: int, rng: np.random.Generator) -> np.ndarray:
    q = np.where(rng.random(n) < (1.0 + x) / 2.0, 1.0, -1.0)
    s = np.where(rng.random(n) < (1.0 + y) / 2.0, 1.0, -1.0)
    reward = weight_q * q + weight_s * s
    score_q = (1.0 - x * x) * q / (1.0 + x * q)
    score_s = (1.0 - y * y) * s / (1.0 + y * s)
    return np.array([float(np.mean(reward * score_q)), float(np.mean(reward * score_s))])


def euler_reduced(h: float, phi0: float, tmax: float = 0.02) -> dict:
    x, y, phi = 0.8, 0.0, phi0
    contact = None
    last_certified = None
    min_point_in_box = math.inf
    rows = []
    for k in range(int(round(tmax / h)) + 1):
        t = k * h
        state = metrics(x, y, phi)
        if state["inside_box"]:
            min_point_in_box = min(min_point_in_box, state["pointwise"])
        if state["worst"] >= GAMMA - 1e-15:
            last_certified = t
        elif contact is None:
            contact = t
        rows.append((t, x, y, phi, state["worst"], state["pointwise"]))
        if k * h >= tmax - 1e-15:
            break
        x = x + h * state["omega_q"] * phi
        y = y + h * state["omega_s"] * (1.0 - phi)
        phi = phi + h * RHO
    return {"contact": contact, "last_certified": last_certified,
            "min_pointwise_in_box": min_point_in_box, "rows": rows}


def euler_theta(h: float, phi0: float, n_samples: int | None, seed: int, tmax: float = 0.02,
                weight: tuple[float, float] | None = None) -> dict:
    x, y, phi = 0.8, 0.0, phi0
    theta = np.array([np.arctanh(x), np.arctanh(y)])
    rng = np.random.default_rng(seed)
    contact = None
    max_error = 0.0
    min_point = math.inf
    rows = []
    steps = int(round(tmax / h))
    for k in range(steps + 1):
        t = k * h
        wq, ws = (phi, 1.0 - phi) if weight is None else weight
        state = metrics(x, y, phi if weight is None else wq / (wq + ws))
        point_phi = phi if weight is None else wq / (wq + ws)
        state = metrics(x, y, point_phi)
        min_point = min(min_point, state["pointwise"])
        if contact is None and state["worst"] < GAMMA - 1e-15:
            contact = t
        exact = exact_theta_gradient(x, y, wq, ws)
        if n_samples is None:
            grad = exact
            error = 0.0
        else:
            grad = sample_gradient(x, y, wq, ws, n_samples, rng)
            error = float(np.linalg.norm(grad - exact))
            max_error = max(max_error, error)
        rows.append((t, x, y, point_phi, state["worst"], state["pointwise"], error))
        if k == steps:
            break
        theta = theta + h * grad
        x, y = (float(np.tanh(theta[0])), float(np.tanh(theta[1])))
        if weight is None:
            phi = phi + h * RHO
    return {"contact": contact, "max_gradient_error": max_error, "min_pointwise": min_point, "rows": rows}


def checkpoint_errors(x: float, y: float, phi: float, n: int, reps: int = 20) -> list[float]:
    exact = exact_theta_gradient(x, y, phi, 1.0 - phi)
    return [
        float(np.linalg.norm(
            sample_gradient(x, y, phi, 1.0 - phi, n, np.random.default_rng(i)) - exact
        ))
        for i in range(reps)
    ]


def required_batch(x: float, y: float, phi: float) -> dict:
    exact = exact_theta_gradient(x, y, phi, 1.0 - phi)
    enumerated = enumerated_gradient(x, y, phi, 1.0 - phi)
    if np.linalg.norm(exact - enumerated) > 1e-12:
        raise AssertionError((exact, enumerated))
    probe = np.vstack([
        sample_gradient(x, y, phi, 1.0 - phi, 1, np.random.default_rng(1000 + i))
        for i in range(4000)
    ])
    sigma = float(np.std(np.linalg.norm(probe - exact, axis=1), ddof=1))
    n = int(math.ceil((2.0 * sigma / MARGIN) ** 2))
    observed = checkpoint_errors(x, y, phi, n)
    formula_n = n
    # The one-sample formula is only a start. Enlarge until the measured
    # checkpoint error itself is below the 0.001 slack.
    while max(observed) > MARGIN and n < 8_000_000:
        n *= 2
        observed = checkpoint_errors(x, y, phi, n)
    return {
        "sigma_one_sample": sigma,
        "formula_n": formula_n,
        "accuracy_controlled_n": n,
        "formula_checkpoint_errors_not_recomputed_here": True,
        "accurate_checkpoint_max_error": max(observed),
        "accurate_checkpoint_mean_error": float(np.mean(observed)),
        "exact_gradient": exact.tolist(),
        "margin_met": max(observed) <= MARGIN,
    }


def fit_reward_head(n_pairs: int, phi_star: float, seed: int, informative: bool) -> np.ndarray:
    """Pairwise logistic head. phi_star labels informative pairs and is not returned to the caller as a parameter."""
    rng = np.random.default_rng(seed)
    x, y = 0.8, 0.0

    def draw() -> np.ndarray:
        q = 1.0 if rng.random() < (1.0 + x) / 2.0 else -1.0
        s = 1.0 if rng.random() < (1.0 + y) / 2.0 else -1.0
        return np.array([q, s])

    diffs, labels = [], []
    for _ in range(n_pairs):
        a, b = draw(), draw()
        diff = a - b
        if informative:
            logit = phi_star * diff[0] + (1.0 - phi_star) * diff[1]
            label = float(rng.random() < expit(logit))
        else:
            label = float(rng.random() < 0.5)
            diff = np.zeros(2)
        diffs.append(diff)
        labels.append(label)
    design = np.vstack(diffs)
    response = np.array(labels)

    def loss(w: np.ndarray) -> float:
        z = design @ w
        return float(np.sum(np.logaddexp(0.0, z) - response * z) + 1e-6 * np.dot(w, w))

    fit = minimize(loss, np.zeros(2), method="BFGS")
    if not fit.success:
        raise RuntimeError(fit.message)
    return fit.x


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("outputs/horizon_bridge/step04"))
    args = parser.parse_args()
    started = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=True)
    reference = ordinary_pg_check()
    phi0 = float(reference["initial_phi"])
    batch = required_batch(0.8, 0.0, phi0)
    small_n = 64
    accurate_n = max(batch["accuracy_controlled_n"], small_n + 1)
    reduced = {h: euler_reduced(h, phi0) for h in (5e-4, 2.5e-4)}
    small = [euler_theta(5e-4, phi0, small_n, seed, tmax=0.012) for seed in range(8)]
    accurate = [euler_theta(5e-4, phi0, accurate_n, seed, tmax=0.012) for seed in range(4)]
    exact_theta = euler_theta(5e-4, phi0, None, 0)
    hidden = phi0
    weight = fit_reward_head(1200, hidden, seed=0, informative=True)
    if weight[0] + weight[1] <= 0:
        raise AssertionError(weight)
    phi_hat = float(weight[0] / (weight[0] + weight[1]))
    # Fitted coefficient replaces the known initial phi. The maximal-speed schedule is otherwise unchanged.
    btl_runs = [euler_theta(5e-4, phi_hat, accurate_n, seed, tmax=0.002) for seed in range(2)]
    checkpoint_errors = {
        "small": [float(np.linalg.norm(sample_gradient(0.8, 0.0, phi0, 1 - phi0, small_n, np.random.default_rng(i))
                                        - exact_theta_gradient(0.8, 0.0, phi0, 1 - phi0)))
                  for i in range(20)],
        "accurate": [float(np.linalg.norm(sample_gradient(0.8, 0.0, phi0, 1 - phi0, accurate_n, np.random.default_rng(i))
                                           - exact_theta_gradient(0.8, 0.0, phi0, 1 - phi0)))
                     for i in range(20)],
    }

    def contact_stats(runs: list[dict]) -> dict:
        values = [row["contact"] for row in runs]
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "values": values,
            "max_gradient_error": float(max(row["max_gradient_error"] for row in runs)),
            "min_pointwise": float(min(row["min_pointwise"] for row in runs)),
        }

    with (args.out / "ordinary_pg_finite.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["mesh", "time", "x", "y", "phi", "worst_gain", "pointwise"])
        for h, run in reduced.items():
            for row in run["rows"]:
                writer.writerow([h, *row])
    table = {
        "continuous": {
            "contact": reference["numerical_maximal_speed_boundary_time"],
            "all_controls_upper": reference["analytic_all_controls_boundary_time_upper"],
            "pointwise_lower_in_box": reference["analytic_pointwise_optimum_lower_in_box"],
            "initial_worst": reference["initial_worst_gain"],
        },
        "finite": {str(h): {"contact": run["contact"], "last_certified": run["last_certified"],
                            "min_pointwise_in_box": run["min_pointwise_in_box"]}
                   for h, run in reduced.items()},
        "exact_theta_euler_h_5e-4": {"contact": exact_theta["contact"], "min_pointwise": exact_theta["min_pointwise"]},
        "batch": batch,
        "checkpoint_error_mean": {key: float(np.mean(value)) for key, value in checkpoint_errors.items()},
        "checkpoint_error_max": {key: float(np.max(value)) for key, value in checkpoint_errors.items()},
        "sampled_small": contact_stats(small),
        "sampled_accurate": contact_stats(accurate),
        "btl_weight": weight.tolist(),
        "btl_phi_hat": phi_hat,
        "btl_phi_error": phi_hat - phi0,
        "btl_fed": contact_stats(btl_runs),
        "margin": MARGIN,
    }
    payload = {
        "status": "PASS",
        "command": "python experiments/bridge/04_ordinary_pg.py --out outputs/horizon_bridge/step04",
        "table": table,
        "scope": "Theorem 2 ordinary Euclidean PG. Sampling replaces only the expected gradient.",
    }
    write_summary(args.out, payload, started)


if __name__ == "__main__":
    main()
