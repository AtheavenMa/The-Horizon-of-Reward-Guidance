#!/usr/bin/env python3
"""CPU bridge from the existing five-output Horizon construction to finite-step updates.

This is NOT a new RL benchmark. It checks exact finite-step gain, certificate
crossing, policy probabilities and the original fixed-batch BTL example.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.linalg import helmert
from scipy.special import logit, expit


def one_mesh(dt: float, horizon: float = 15.0) -> dict:
    eps, d, coupling = 0.04, 4.0, 0.75
    v, rho, g, gamma, eta0 = 0.12, 0.04, 1.0, 0.6, 1.0
    H = helmert(5, full=False).T
    theta = np.zeros(4)
    n = int(round(horizon / dt))
    assert abs(n * dt - horizon) < 1e-9
    rows = []
    max_identity_error, max_c_error = 0.0, 0.0
    min_probability, min_M_eigenvalue = 1.0, float("inf")
    for k in range(n + 1):
        t = k * dt
        c = coupling * theta[2]
        phi = rho * t
        p = np.ones(5) / 5.0 + eps * H @ theta
        K = np.array([[1.0, 0.0], [-c, 1.0]])
        M = np.block([[d * np.eye(2), K], [K.T, d * np.eye(2)]])
        min_probability = min(min_probability, float(p.min()))
        min_M_eigenvalue = min(min_M_eigenvalue, float(np.linalg.eigvalsh(M).min()))
        max_c_error = max(max_c_error, abs(c - v * t))
        source_gain = g - eta0 * abs(phi - c)
        pointwise_gain = g  # phi=c is allowed; c <= 1.8 < 2
        true_gain = g  # chosen true delta_star=0
        rows.append([k, t, c, phi, source_gain, pointwise_gain, true_gain])
        if k == n:
            break
        velocity = eps * M @ np.array([0.0, 0.0, 1.0, phi])
        theta_next = theta + dt * velocity
        p_next = np.ones(5) / 5.0 + eps * H @ theta_next
        min_probability = min(min_probability, float(p_next.min()))
        delta_p = p_next - p
        for delta in (-1.0, 0.0, 1.0):
            r = H[:, 0] + delta * H[:, 1]
            normalized_step_gain = float(r @ delta_p / (eps * eps * dt))
            max_identity_error = max(
                max_identity_error,
                abs(normalized_step_gain - (1 + delta * (phi - c))),
            )
        theta = theta_next
    assert min_probability > 0, min_probability
    assert min_M_eigenvalue > 0, min_M_eigenvalue
    assert max_c_error < 1e-9, max_c_error
    assert max_identity_error < 1e-9, max_identity_error
    failed = [r for r in rows if r[4] < gamma - 1e-10]
    first_failed = failed[0][1] if failed else None
    assert first_failed is not None
    assert 5.0 < first_failed <= 5.0 + dt + 1e-9
    return {
        "dt": dt, "rows": rows,
        "first_uncertified": first_failed, "expected_exact_horizon": 5.0,
        "max_identity_error": max_identity_error, "max_c_error": max_c_error,
        "min_probability": min_probability,
        "min_M_eigenvalue": min_M_eigenvalue,
        "source_gain_at_T15": rows[-1][4],
        "true_gain_at_T15": rows[-1][6],
        "pointwise_gain_at_T15": rows[-1][5],
    }


def feedback_reference() -> dict:
    """Representative fixed-N BTL interval on a frozen trajectory."""
    n, wins, alpha, q = 1200, 600, 0.05, math.sqrt(2 / 3)
    radius = math.sqrt(math.log(2 / alpha) / (2 * n))
    p_lo = max(float(expit(-q)), wins / n - radius)
    p_hi = min(float(expit(q)), wins / n + radius)
    lo, hi = float(logit(p_lo) / q), float(logit(p_hi) / q)
    assert lo <= 0 <= hi
    c_T, phi_T = 0.12 * 15, 0.04 * 15
    source = 1.0 - abs(phi_T - c_T)
    after = min(1.0 + lo * (phi_T - c_T), 1.0 + hi * (phi_T - c_T))
    assert source < 0.6 < after
    return {
        "comparison": "informative fixed Bradley-Terry pair, sensitivity sqrt(2/3)",
        "label_count": n, "positive_labels": wins, "confidence": 1 - alpha,
        "confidence_interval_delta": [lo, hi], "T": 15.0,
        "source_gain_at_T": source, "updated_gain_same_trajectory_at_T": after,
        "gamma": 0.6,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("outputs/horizon_bridge/step01"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    results = [one_mesh(dt) for dt in (0.2, 0.1, 0.05, 0.025)]
    with (args.out / "finite_steps.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dt", "step", "time", "c", "phi", "source_certificate", "pointwise_gain", "true_gain"])
        for x in results:
            for row in x["rows"]:
                writer.writerow([x["dt"], *row])
    compact = [{k: v for k, v in x.items() if k != "rows"} for x in results]
    payload = {
        "status": "PASS",
        "scope": "exact five-output finite-step reference; not a new RL benchmark",
        "mesh": compact,
        "feedback": feedback_reference(),
    }
    (args.out / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
