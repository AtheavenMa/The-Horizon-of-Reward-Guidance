#!/usr/bin/env python3
"""Step 2: fixed-N Bradley-Terry labels on the frozen five-output trajectory.

Policy, phi(t)=0.04 t and c(t)=0.12 t are not updated. The only new object is
a confidence set for the scalar preference.
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
from scipy.special import expit, logit
from scipy.stats import binom

from horizon import btl_confidence_interval, certify_btl_batch

from experiments.bridge.common import write_summary


Q = math.sqrt(2.0 / 3.0)
G, GAMMA, ETA0 = 1.0, 0.6, 1.0
V, RHO, T = 0.12, 0.04, 15.0
ARRIVAL = 4.0
SOURCE = (-1.0, 1.0)
SEEDS = tuple(range(10))


def lag(t: float) -> float:
    return RHO * t - V * t


def worst_certificate(t: float, lo: float, hi: float) -> float:
    gap = lag(t)
    return G + min(lo * gap, hi * gap)


def independent_interval(successes: int, labels: int, sensitivity: float) -> tuple[float, float]:
    """Same Hoeffding truncation as horizon.btl_confidence_interval, written out."""
    radius = math.sqrt(math.log(2 / 0.05) / (2 * labels))
    p = successes / labels
    p_lo = max(float(expit(-sensitivity)), p - radius)
    p_hi = min(float(expit(sensitivity)), p + radius)
    if p_lo > p_hi:
        raise ValueError("empty interval")
    return float(logit(p_lo) / sensitivity), float(logit(p_hi) / sensitivity)


def library_interval(successes: int, labels: int, sensitivity: float = Q) -> tuple[float, float] | None:
    try:
        got = btl_confidence_interval(
            successes=successes, labels=labels, sensitivity=sensitivity, uncertainty_radius=ETA0,
        )
    except ValueError:
        return None
    lo, hi = independent_interval(successes, labels, sensitivity)
    if abs(got.lower_delta - lo) > 1e-12 or abs(got.upper_delta - hi) > 1e-12:
        raise AssertionError((got, lo, hi))
    return float(got.lower_delta), float(got.upper_delta)


def point_estimate(successes: int, labels: int, sensitivity: float = Q) -> float:
    p = min(max(successes / labels, 1e-12), 1 - 1e-12)
    return float(np.clip(logit(p) / sensitivity, -ETA0, ETA0))


def first_strict_loss(lo: float, hi: float, dt: float = 0.025) -> float | None:
    n = int(round(T / dt))
    for k in range(n + 1):
        t = k * dt
        if worst_certificate(t, lo, hi) < GAMMA - 1e-10:
            return t
    return None


def path_minimum(lo: float, hi: float, arrival: float | None) -> float:
    grid = np.linspace(0.0, T, 601)
    values = []
    for t in grid:
        a, b = (SOURCE if arrival is not None and t < arrival else (lo, hi))
        values.append(worst_certificate(float(t), a, b))
    return float(min(values))


def binomial_summary(labels: int, true_delta: float, sensitivity: float = Q) -> dict:
    p = float(expit(sensitivity * true_delta))
    ks = np.arange(labels + 1)
    pmf = binom.pmf(ks, labels, p)
    terminal = np.empty(labels + 1)
    upper = np.empty(labels + 1)
    one_sided = np.empty(labels + 1, dtype=bool)
    empty = np.empty(labels + 1, dtype=bool)
    for k in ks:
        interval = library_interval(int(k), labels, sensitivity)
        empty[k] = interval is None
        if interval is None:
            terminal[k] = -np.inf
            upper[k] = np.inf
        else:
            lo, hi = interval
            terminal[k] = worst_certificate(T, lo, hi)
            upper[k] = hi
        one_sided[k] = certify_btl_batch(
            successes=int(k), labels=labels, critical_delta=1.0 / 3.0, sensitivity=sensitivity,
        ).certified
    certify = terminal >= GAMMA - 1e-12
    # The one-sided critical value must match a direct Hoeffding threshold.
    pc = float(expit(sensitivity / 3.0))
    radius = math.sqrt(math.log(1 / 0.05) / (2 * labels))
    threshold = int((labels * (pc - radius)) // 1)
    direct = ks <= threshold
    if not np.array_equal(one_sided, direct):
        raise AssertionError("library one-sided test disagrees with the binomial threshold")
    median_k = int(np.searchsorted(np.cumsum(pmf), 0.5))
    median_interval = library_interval(median_k, labels, sensitivity)
    if median_interval is None:
        raise AssertionError("median binomial outcome has an empty confidence set")
    lo_m, hi_m = median_interval
    return {
        "win_probability": p,
        "median_successes": median_k,
        "median_delta_hat": point_estimate(median_k, labels, sensitivity),
        "median_interval": [lo_m, hi_m],
        "median_terminal_certificate": float(terminal[median_k]),
        "exact_probability_terminal_certificate_meets_gamma": float(pmf[certify].sum()),
        "exact_probability_one_sided_test_certifies": float(pmf[one_sided].sum()),
        "exact_probability_empty_confidence_set": float(pmf[empty].sum()),
        "mean_terminal_certificate": float(pmf[~empty] @ terminal[~empty] / pmf[~empty].sum()),
        "mean_upper_endpoint": float(pmf[~empty] @ upper[~empty] / pmf[~empty].sum()),
    }


def smoke(labels: int, true_delta: float, sensitivity: float = Q) -> list[dict]:
    p = float(expit(sensitivity * true_delta))
    rows = []
    for seed in SEEDS:
        wins = int(np.random.default_rng(seed).binomial(labels, p))
        interval = library_interval(wins, labels, sensitivity)
        rows.append({
            "seed": seed, "wins": wins, "delta_hat": point_estimate(wins, labels, sensitivity),
            "interval": None if interval is None else list(interval),
            "terminal_certificate": None if interval is None else worst_certificate(T, *interval),
        })
    return rows


def negative_controls() -> dict:
    """Step 1 controls that must not recreate the source-only loss."""
    no_ambiguity = [worst_certificate(t, 0.0, 0.0) for t in (0.0, 5.0, 15.0)]
    tracked = [G - ETA0 * abs((V * t) - (V * t)) for t in (0.0, 5.0, 15.0)]
    return {
        "eta0_zero_gain_at_0_5_15": no_ambiguity,
        "rho_equal_v_source_gain_at_0_5_15": tracked,
        "either_recreates_original_loss": any(x < GAMMA for x in no_ambiguity + tracked),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("outputs/horizon_bridge/step02"))
    args = parser.parse_args()
    started = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=True)
    reference = library_interval(600, 1200)
    reference_cert = worst_certificate(T, *reference)
    if abs(reference[0] + 0.19245962944052997) > 1e-12 or abs(reference_cert - 0.7690484446713638) > 1e-12:
        raise AssertionError((reference, reference_cert))
    source_cert = worst_certificate(T, *SOURCE)
    rows = [
        {
            "name": "source_only", "labels": 0, "true_delta": 0.0,
            "delta_hat": None, "interval": list(SOURCE),
            "terminal_certificate": source_cert,
            "path_minimum_feedback_at_4": path_minimum(*SOURCE, None),
            "first_strict_loss": first_strict_loss(*SOURCE),
            "exact_probability_terminal_certificate_meets_gamma": 0.0,
        },
        {
            "name": "source_comparisons_no_delta_sensitivity", "labels": 1200, "true_delta": 0.0,
            "delta_hat": None, "interval": list(SOURCE),
            "terminal_certificate": source_cert,
            "path_minimum_feedback_at_4": path_minimum(*SOURCE, ARRIVAL),
            "first_strict_loss": first_strict_loss(*SOURCE),
            "exact_probability_terminal_certificate_meets_gamma": 0.0,
            "note": "q=0 leaves the likelihood flat on delta, so the source set [-1,1] is unchanged.",
        },
    ]
    informative = {}
    smoke_rows = []
    for n in (64, 256, 1200):
        exact = binomial_summary(n, 0.0)
        lo, hi = exact["median_interval"]
        informative[str(n)] = exact
        rows.append({
            "name": "informative", "labels": n, "true_delta": 0.0,
            "delta_hat": exact["median_delta_hat"], "interval": [lo, hi],
            "terminal_certificate": exact["median_terminal_certificate"],
            "path_minimum_feedback_at_4": path_minimum(lo, hi, ARRIVAL),
            "first_strict_loss": first_strict_loss(lo, hi),
            "exact_probability_terminal_certificate_meets_gamma": exact[
                "exact_probability_terminal_certificate_meets_gamma"
            ],
            "exact_probability_one_sided_test_certifies": exact[
                "exact_probability_one_sided_test_certifies"
            ],
        })
        smoke_rows.extend({"labels": n, **item} for item in smoke(n, 0.0))
    infeasible = binomial_summary(1200, 0.4)
    lo, hi = infeasible["median_interval"]
    rows.append({
        "name": "fixed_reference_600_of_1200", "labels": 1200, "true_delta": 0.0,
        "delta_hat": point_estimate(600, 1200), "interval": list(reference),
        "terminal_certificate": reference_cert,
        "path_minimum_feedback_at_4": path_minimum(*reference, ARRIVAL),
        "first_strict_loss": first_strict_loss(*reference),
        "exact_probability_terminal_certificate_meets_gamma": None,
        "note": "Deterministic published count, not a random draw.",
    })
    rows.append({
        "name": "infeasible_true_delta", "labels": 1200, "true_delta": 0.4,
        "delta_hat": infeasible["median_delta_hat"], "interval": [lo, hi],
        "terminal_certificate": infeasible["median_terminal_certificate"],
        "path_minimum_feedback_at_4": path_minimum(lo, hi, ARRIVAL),
        "first_strict_loss": first_strict_loss(lo, hi),
        "exact_probability_terminal_certificate_meets_gamma": infeasible[
            "exact_probability_terminal_certificate_meets_gamma"
        ],
        "exact_probability_one_sided_test_certifies": infeasible[
            "exact_probability_one_sided_test_certifies"
        ],
    })
    if rows[-1]["exact_probability_terminal_certificate_meets_gamma"] > 0.05:
        raise AssertionError("infeasible preference falsely certifies T=15 too often")
    with (args.out / "preference.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "name", "labels", "true_delta", "delta_hat", "interval_lo", "interval_hi",
            "terminal_certificate", "first_strict_loss",
            "exact_probability_terminal_certificate_meets_gamma",
        ])
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "name": row["name"], "labels": row["labels"], "true_delta": row["true_delta"],
                "delta_hat": row["delta_hat"], "interval_lo": row["interval"][0],
                "interval_hi": row["interval"][1],
                "terminal_certificate": row["terminal_certificate"],
                "first_strict_loss": row["first_strict_loss"],
                "exact_probability_terminal_certificate_meets_gamma": row[
                    "exact_probability_terminal_certificate_meets_gamma"
                ],
            })
    payload = {
        "status": "PASS",
        "command": "python experiments/bridge/02_preference_feedback.py --out outputs/horizon_bridge/step02",
        "frozen_trajectory": {"phi": "0.04 t", "c": "0.12 t", "delta_star_informative": 0.0},
        "sensitivity": Q,
        "feedback_arrival": ARRIVAL,
        "true_gain_at_T15": G,
        "negative_controls": negative_controls(),
        "rows": rows,
        "informative_exact": informative,
        "infeasible_exact": infeasible,
        "smoke_seeds": smoke_rows,
        "scope": "Information on a frozen trajectory. No policy or phi update.",
    }
    write_summary(args.out, payload, started)


if __name__ == "__main__":
    main()
