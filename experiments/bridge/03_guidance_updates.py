#!/usr/bin/env python3
"""Step 3: published clipped calibration, then information on that fixed path.

The policy update is the published five-output Euler step. Preference labels
do not move phi. A separate robust controller may use a confidence set, never
the hidden true preference.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import math

import numpy as np
from scipy.linalg import helmert
from scipy.special import expit, logit

from experiments.bridge.common import write_summary
from experiments.discrete_resource_check import run as published_discrete
from horizon import btl_confidence_interval


H_POLICY, ETA, G_CLIP, GAMMA, ETA0 = 0.05, 0.01, 0.2, 0.6, 1.0
MAX_TIME = 15.0
EPS, DIAGONAL, COUPLING = 0.04, 4.0, 0.75
V = COUPLING * EPS * DIAGONAL


def worst(phi: float, c: float, lo: float, hi: float) -> float:
    return 1.0 + min(lo * (phi - c), hi * (phi - c))


def first_loss(times: list[float], certs: list[float]) -> float | None:
    for t, value in zip(times, certs):
        if value < GAMMA - 1e-12:
            return t
    return None


def clipped_trajectory(m: int) -> dict:
    """Same loop as experiments/discrete_resource_check.py, one m at a time."""
    theta = np.zeros(4)
    phi = 0.0
    H = helmert(5, full=False).T
    rows = []
    for k in range(round(MAX_TIME / H_POLICY)):
        t = k * H_POLICY
        c = COUPLING * theta[2]
        phi_before = phi
        for _ in range(m):
            grad = float(np.clip(phi - c, -G_CLIP, G_CLIP))
            phi = float(np.clip(phi - ETA * grad, -2.0, 2.0))
        move = abs(phi - phi_before)
        K = np.array([[1.0, 0.0], [-c, 1.0]])
        M = np.block([[DIAGONAL * np.eye(2), K], [K.T, DIAGONAL * np.eye(2)]])
        velocity = EPS * M @ np.array([0.0, 0.0, 1.0, phi])
        p = 0.2 + EPS * H @ theta
        theta = theta + H_POLICY * velocity
        p_next = 0.2 + EPS * H @ theta
        if min(p.min(), p_next.min()) <= 0 or np.linalg.eigvalsh(M).min() <= 0:
            raise AssertionError("published policy left the valid set")
        rows.append({"time": t, "c": float(c), "phi": float(phi), "move": float(move)})
    return {"rows": rows, "budget": m * ETA * G_CLIP}


def robust_trajectory(m: int, lo: float, hi: float) -> dict:
    """Same per-step movement budget. Target maximizes the worst certified gain."""
    theta = np.zeros(4)
    phi = 0.0
    H = helmert(5, full=False).T
    rows = []
    budget = ETA * G_CLIP
    for k in range(round(MAX_TIME / H_POLICY)):
        t = k * H_POLICY
        c = COUPLING * theta[2]
        phi_before = phi
        for _ in range(m):
            window = (phi - budget, phi + budget)
            candidates = [window[0], window[1]]
            if window[0] <= c <= window[1]:
                candidates.append(c)
            phi = float(np.clip(max(candidates, key=lambda z: worst(z, c, lo, hi)), -2.0, 2.0))
        move = abs(phi - phi_before)
        K = np.array([[1.0, 0.0], [-c, 1.0]])
        M = np.block([[DIAGONAL * np.eye(2), K], [K.T, DIAGONAL * np.eye(2)]])
        velocity = EPS * M @ np.array([0.0, 0.0, 1.0, phi])
        theta = theta + H_POLICY * velocity
        rows.append({"time": t, "c": float(c), "phi": float(phi), "move": float(move)})
    return {"rows": rows, "budget": m * budget}


def ideal_loss(m: int) -> float | None:
    budget = m * ETA * G_CLIP
    for k in range(round(MAX_TIME / H_POLICY)):
        ideal_phi = min(V * H_POLICY * k, budget * k)
        if 1.0 - ETA0 * abs(ideal_phi - V * H_POLICY * k) < GAMMA - 1e-12:
            return k * H_POLICY
    return None


def summarize(name: str, m: int, path: dict, lo: float, hi: float, published: dict | None) -> dict:
    times = [row["time"] for row in path["rows"]]
    source = [worst(row["phi"], row["c"], -1.0, 1.0) for row in path["rows"]]
    updated = [worst(row["phi"], row["c"], lo, hi) for row in path["rows"]]
    realized = max(row["move"] for row in path["rows"])
    out = {
        "method": name, "m": m, "movement_budget": path["budget"],
        "source_only_first_loss": first_loss(times, source),
        "after_feedback_first_loss": first_loss(times, updated),
        "realized_max_movement": realized,
        "ideal_envelope_first_loss": ideal_loss(m),
        "phi_at_last_step": path["rows"][-1]["phi"],
        "c_at_last_step": path["rows"][-1]["c"],
        "terminal_source_certificate": source[-1],
        "terminal_updated_certificate": updated[-1],
    }
    if published is not None:
        if abs((published["first_uncertified_policy_step_time"] or -1) - (out["source_only_first_loss"] or -1)) > 1e-9:
            raise AssertionError((published, out))
        if abs(published["observed_max_displacement"] - realized) > 1e-9:
            raise AssertionError((published["observed_max_displacement"], realized))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("outputs/horizon_bridge/step03"))
    args = parser.parse_args()
    started = time.perf_counter()
    published = published_discrete(args.out / "published_reuse")
    by_m = {row["reward_steps_per_policy_step"]: row for row in published["runs"]}
    # Reference informative interval from the frozen 600/1200 batch. It contains 0.
    lo, hi = -0.19245962944052997, 0.19245962944053022
    table = []
    raw = []
    for m in (0, 1, 2, 3):
        path = clipped_trajectory(m)
        item = summarize(
            "no_calibration" if m == 0 else "clipped_calibration",
            m, path, lo, hi, by_m[m],
        )
        table.append(item)
        for row in path["rows"]:
            raw.append([m, "clipped", row["time"], row["c"], row["phi"], row["move"],
                        worst(row["phi"], row["c"], -1.0, 1.0),
                        worst(row["phi"], row["c"], lo, hi)])
    robust = robust_trajectory(1, lo, hi)
    clipped_m1 = clipped_trajectory(1)
    path_gap = max(abs(a["phi"] - b["phi"]) for a, b in zip(robust["rows"], clipped_m1["rows"]))
    robust_row = summarize("robust_same_budget", 1, robust, lo, hi, None)
    robust_row["max_abs_phi_gap_versus_clipped_m1"] = path_gap
    robust_row["interval"] = [lo, hi]
    robust_row["note"] = (
        "The interval contains 0, so the robust target is c and this controller spends the full movement budget. "
        "Clipped calibration uses step eta*clip(phi-c), which is smaller than the budget while the lag is below G. "
        "The phi paths therefore differ: robust loss matches the ideal envelope, clipped loss is earlier."
    )
    table.append(robust_row)
    # Hidden delta generates labels. The controller receives only the fitted interval.
    hidden_delta = 0.4
    sensitivity = math.sqrt(2.0 / 3.0)
    wins = int(np.random.default_rng(0).binomial(1200, float(expit(sensitivity * hidden_delta))))
    fitted = btl_confidence_interval(
        successes=wins, labels=1200, sensitivity=sensitivity, uncertainty_radius=1.0,
    )
    side_lo, side_hi = float(fitted.lower_delta), float(fitted.upper_delta)
    delta_hat = float(np.clip(logit(min(max(wins / 1200, 1e-12), 1 - 1e-12)) / sensitivity, -1.0, 1.0))
    side = robust_trajectory(1, side_lo, side_hi)
    side_gap = max(abs(a["phi"] - b["phi"]) for a, b in zip(side["rows"], clipped_m1["rows"]))
    btl = summarize("btl_fitted_information_and_guidance", 1, side, side_lo, side_hi, None)
    btl["delta_hat"] = delta_hat
    btl["wins"] = wins
    btl["interval"] = [side_lo, side_hi]
    btl["phi_final"] = side["rows"][-1]["phi"]
    btl["max_abs_phi_gap_versus_clipped_m1"] = side_gap
    btl["note"] = (
        "delta_hat is the BTL point estimate from seed 0. phi is the guidance state and is not delta_hat. "
        "The controller sees only the confidence interval. Hidden delta is not an input to the update."
    )
    table.append(btl)
    for row in side["rows"]:
        raw.append([1, "btl_onesided", row["time"], row["c"], row["phi"], row["move"],
                    worst(row["phi"], row["c"], -1.0, 1.0),
                    worst(row["phi"], row["c"], side_lo, side_hi)])
    with (args.out / "guidance.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["m", "controller", "time", "c", "phi", "move", "source_certificate", "updated_certificate"])
        writer.writerows(raw)
    no_info = summarize("no_information_labels", 1, clipped_m1, -1.0, 1.0, None)
    payload = {
        "status": "PASS",
        "command": "python experiments/bridge/03_guidance_updates.py --out outputs/horizon_bridge/step03",
        "published_identity_error": published["maximum_exact_step_identity_error"],
        "informative_interval": [lo, hi],
        "table": table,
        "no_information_control": no_info,
        "no_ambiguity_m0_terminal_true_gain": 1.0,
        "scope": "Clipped calibration reused from discrete_resource_check.py. Labels do not retrain the policy.",
    }
    write_summary(args.out, payload, started)


if __name__ == "__main__":
    main()
