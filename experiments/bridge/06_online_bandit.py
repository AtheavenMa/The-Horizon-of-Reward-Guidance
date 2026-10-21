#!/usr/bin/env python3
"""Steps 5A–5C: four-outcome trajectory, frozen-path BTL, then bounded guidance.

05_contextual_bandit.py stays the eight-step historical pilot. This script
does not replace it. True return, instantaneous gain rate, and finite-step
gain rate are logged as different quantities.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from scipy.stats import binom

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.bridge.common import write_summary
from experiments.strengthening_checks import ordinary_pg_check
from horizon import btl_confidence_interval


GAMMA = 0.06
RHO = 0.04
H = 5e-4
T_FINAL = 0.012
N_STEPS = 24
FEEDBACK_STEP = 8
U0 = (-1.0, 2.0)
DELTA_RADIUS = 1.5
SENSITIVITY = 4.0
POLICY_SEEDS = (0, 1, 2, 3, 4)
LABEL_SEEDS = (100, 101, 102, 103, 104)
SAMPLE_SIZES = (256, 2048, 8192)
# 8192 still moves the one-step certificate by more than the 0.001 initial slack.
# This is the smallest power of two at which 20 first-step replicates stayed above gamma.
PRECISION_N = 2_097_152


def reference_phi0() -> float:
    return float(ordinary_pg_check()["initial_phi"])


def omega(z: float) -> float:
    return (1.0 - z * z) ** 2


def exact_grad(x: float, y: float, phi: float) -> np.ndarray:
    return np.array([phi * (1.0 - x * x), (1.0 - phi) * (1.0 - y * y)], dtype=float)


def enumerated_grad(x: float, y: float, phi: float) -> np.ndarray:
    grad = np.zeros(2)
    for q in (-1.0, 1.0):
        for s in (-1.0, 1.0):
            probability = 0.25 * (1.0 + x * q) * (1.0 + y * s)
            reward = phi * q + (1.0 - phi) * s
            grad += probability * reward * np.array([q - x, s - y])
    return grad


def true_return(x: float, y: float, preference: float) -> float:
    return preference * x + (1.0 - preference) * y


def enumerated_return(x: float, y: float, preference: float) -> float:
    total = 0.0
    for q in (-1.0, 1.0):
        for s in (-1.0, 1.0):
            probability = 0.25 * (1.0 + x * q) * (1.0 + y * s)
            total += probability * (preference * q + (1.0 - preference) * s)
    return total


def finite_gain(x: float, y: float, x_next: float, y_next: float, preference: float, h: float) -> float:
    return (true_return(x_next, y_next, preference) - true_return(x, y, preference)) / h


def interval_gain(x: float, y: float, x_next: float, y_next: float, bounds: tuple[float, float], h: float) -> float:
    lo, hi = bounds
    return min(
        finite_gain(x, y, x_next, y_next, lo, h),
        finite_gain(x, y, x_next, y_next, hi, h),
    )


def instantaneous_rate(x: float, y: float, phi: float, preference: float) -> float:
    return preference * omega(x) * phi + (1.0 - preference) * omega(y) * (1.0 - phi)


def instantaneous_certificate(x: float, y: float, phi: float, bounds: tuple[float, float]) -> float:
    return min(instantaneous_rate(x, y, phi, bounds[0]), instantaneous_rate(x, y, phi, bounds[1]))


def pointwise(x: float, y: float) -> float:
    a, b = omega(x), omega(y)
    return a * b / (a + b)


def advance(x: float, y: float, phi: float, grad: np.ndarray, h: float) -> tuple[float, float]:
    theta = np.array([np.arctanh(x), np.arctanh(y)]) + h * grad
    nxt = np.tanh(theta)
    return float(nxt[0]), float(nxt[1])


def sample_grad(x: float, y: float, phi: float, u_q: np.ndarray, u_s: np.ndarray) -> np.ndarray:
    q = np.where(u_q < (1.0 + x) / 2.0, 1.0, -1.0)
    s = np.where(u_s < (1.0 + y) / 2.0, 1.0, -1.0)
    reward = phi * q + (1.0 - phi) * s
    return np.array([float(np.mean(reward * (q - x))), float(np.mean(reward * (s - y)))])


def scheduled_phi(k: int, h: float, phi0: float) -> float:
    return phi0 + RHO * k * h


def best_phi(x: float, y: float, bounds: tuple[float, float], low: float, high: float) -> float:
    """Maximize the worst instantaneous gain on a movement interval.

    Ties break toward the larger guidance coefficient.
    """
    a_coef, b_coef = omega(x), omega(y)
    lo, hi = bounds

    def value(phi: float) -> float:
        return min(
            instantaneous_rate(x, y, phi, lo),
            instantaneous_rate(x, y, phi, hi),
        )

    candidates = [low, high]
    if hi > lo and a_coef + b_coef > 0:
        cross = b_coef / (a_coef + b_coef)
        if low <= cross <= high:
            candidates.append(cross)
    best = max(value(phi) for phi in candidates)
    tied = [phi for phi in candidates if abs(value(phi) - best) <= 1e-12]
    return max(tied)


def one_step_pointwise_optimum(x: float, y: float, preference: float, h: float) -> float:
    def negative(phi: float) -> float:
        grad = exact_grad(x, y, float(phi))
        x_next, y_next = advance(x, y, float(phi), grad, h)
        return -finite_gain(x, y, x_next, y_next, preference, h)

    fit = minimize_scalar(negative, bounds=(0.0, 1.0), method="bounded", options={"xatol": 1e-10})
    if not fit.success:
        raise RuntimeError(fit.message)
    return float(-fit.fun)


def self_check(phi0: float) -> None:
    x, y = 0.8, 0.0
    if np.linalg.norm(enumerated_grad(x, y, phi0) - exact_grad(x, y, phi0)) > 1e-12:
        raise AssertionError("score identity failed")
    if abs(true_return(x, y, phi0) - enumerated_return(x, y, phi0)) > 1e-12:
        raise AssertionError("return enumeration failed")
    for preference in (-1.0, 0.0, phi0, 2.0):
        source = (preference * 1 + (1 - preference) * 1) - (preference * -1 + (1 - preference) * -1)
        if abs(source - 2.0) > 1e-12:
            raise AssertionError("source pair depends on the preference")
        informative = (preference * 1 + (1 - preference) * -1) - (preference * -1 + (1 - preference) * 1)
        if abs(informative - (4.0 * preference - 2.0)) > 1e-12:
            raise AssertionError("informative logit is not 4a-2")
    initial = instantaneous_certificate(x, y, phi0, U0)
    if abs(initial - 0.0612592) > 5e-7:
        raise AssertionError(initial)


def population_rollout(phi0: float, preference: float, h: float, n_steps: int) -> list[dict]:
    x, y = 0.8, 0.0
    rows = []
    for k in range(n_steps):
        phi = scheduled_phi(k, h, phi0)
        grad = exact_grad(x, y, phi)
        x_next, y_next = advance(x, y, phi, grad, h)
        rows.append(measure(k, h, x, y, x_next, y_next, phi, preference, grad, grad, U0, U0))
        x, y = x_next, y_next
    rows.append(terminal_state(n_steps, h, x, y, scheduled_phi(n_steps, h, phi0), preference))
    return rows


def measure(k, h, x, y, x_next, y_next, phi, preference, used_grad, exact, source_set, updated_set) -> dict:
    return {
        "k": k,
        "t": k * h,
        "t_end": (k + 1) * h,
        "x": x,
        "y": y,
        "x_next": x_next,
        "y_next": y_next,
        "phi": phi,
        "true_return": true_return(x, y, preference),
        "true_return_next": true_return(x_next, y_next, preference),
        "instantaneous_true_rate": instantaneous_rate(x, y, phi, preference),
        "true_finite_step_gain": finite_gain(x, y, x_next, y_next, preference, h),
        "source_certificate": interval_gain(x, y, x_next, y_next, source_set, h),
        "updated_certificate": interval_gain(x, y, x_next, y_next, updated_set, h),
        "instantaneous_source_certificate": instantaneous_certificate(x, y, phi, source_set),
        "pointwise": pointwise(x, y),
        "gradient_error": float(np.linalg.norm(used_grad - exact)),
        "source_set": list(source_set),
        "updated_set": list(updated_set),
    }


def terminal_state(k, h, x, y, phi, preference) -> dict:
    return {
        "k": k, "t": k * h, "t_end": None, "x": x, "y": y, "x_next": None, "y_next": None,
        "phi": phi, "true_return": true_return(x, y, preference), "true_return_next": None,
        "instantaneous_true_rate": instantaneous_rate(x, y, phi, preference),
        "true_finite_step_gain": None,
        "source_certificate": None, "updated_certificate": None,
        "instantaneous_source_certificate": instantaneous_certificate(x, y, phi, U0),
        "pointwise": pointwise(x, y), "gradient_error": 0.0,
        "source_set": list(U0), "updated_set": list(U0),
    }


def first_below(rows: list[dict], key: str) -> float | None:
    for row in rows:
        value = row[key]
        if value is not None and value < GAMMA - 1e-15:
            return float(row["t"])
    return None


def gain_near(rows: list[dict], when: float) -> float:
    for row in rows:
        if row["t"] is not None and abs(row["t"] - when) <= 1e-9 and row["true_finite_step_gain"] is not None:
            return float(row["true_finite_step_gain"])
    raise KeyError(when)


def summarize_path(rows: list[dict], contact_time: float) -> dict:
    transitions = [row for row in rows if row["true_finite_step_gain"] is not None]
    point = min(row["pointwise"] for row in rows)
    errors = [row["gradient_error"] for row in transitions]
    moves = [abs(transitions[i + 1]["phi"] - transitions[i]["phi"]) for i in range(len(transitions) - 1)]
    return {
        "first_source_loss": first_below(rows, "source_certificate"),
        "first_updated_loss": first_below(rows, "updated_certificate"),
        "first_instantaneous_source_loss": first_below(rows, "instantaneous_source_certificate"),
        "pointwise_min": point,
        "pointwise_above_gamma": point > GAMMA,
        "true_finite_step_gain_near_contact": gain_near(rows, contact_time),
        "terminal_true_return": rows[-1]["true_return"],
        "terminal_source_certificate": transitions[-1]["source_certificate"],
        "terminal_updated_certificate": transitions[-1]["updated_certificate"],
        "max_gradient_error": max(errors),
        "mean_gradient_error": float(np.mean(errors)),
        "max_phi_step": max(moves) if moves else 0.0,
        "phi_within_budget": all(move <= RHO * H + 1e-12 for move in moves) if moves else True,
    }


def uniforms(seed: int, n_steps: int, n: int, h_index: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(10_000 + 100 * h_index + seed)
    return [(rng.random(n), rng.random(n)) for _ in range(n_steps)]


def sampled_rollout(phi0: float, preference: float, h: float, draws: list[tuple[np.ndarray, np.ndarray]], n: int) -> list[dict]:
    x, y = 0.8, 0.0
    rows = []
    for k, (u_q, u_s) in enumerate(draws):
        phi = scheduled_phi(k, h, phi0)
        exact = exact_grad(x, y, phi)
        grad = sample_grad(x, y, phi, u_q[:n], u_s[:n])
        x_next, y_next = advance(x, y, phi, grad, h)
        rows.append(measure(k, h, x, y, x_next, y_next, phi, preference, grad, exact, U0, U0))
        x, y = x_next, y_next
    rows.append(terminal_state(len(draws), h, x, y, scheduled_phi(len(draws), h, phi0), preference))
    return rows


def fit_interval(wins: int, labels: int) -> dict:
    if labels <= 0:
        return {"delta_hat": None, "a_hat": None, "interval": list(U0), "empty": False, "wins": 0}
    try:
        got = btl_confidence_interval(
            successes=int(wins), labels=int(labels), sensitivity=SENSITIVITY,
            uncertainty_radius=DELTA_RADIUS, alpha=0.05,
        )
        lo = max(float(got.lower_delta) + 0.5, U0[0])
        hi = min(float(got.upper_delta) + 0.5, U0[1])
        empty = lo > hi
    except ValueError:
        lo, hi, empty = U0[1], U0[0], True
    p = min(max(wins / labels, 1e-12), 1.0 - 1e-12)
    a_hat = float(np.clip(0.5 + logit(p) / SENSITIVITY, U0[0], U0[1]))
    return {
        "delta_hat": a_hat - 0.5, "a_hat": a_hat,
        "interval": None if empty else [lo, hi], "empty": empty, "wins": int(wins),
    }


def label_batch(preference: float, labels: int, seed: int, informative: bool) -> dict:
    if not informative:
        return {"wins": None, "labels": labels, "informative": False, **fit_interval(0, 0),
                "note": "Source pairs with q=s have logit 2 for every a, so the set stays [-1, 2]."}
    probability = float(expit(SENSITIVITY * (preference - 0.5)))
    wins = int(np.random.default_rng(seed).binomial(labels, probability))
    fitted = fit_interval(wins, labels)
    fitted.update({"labels": labels, "informative": True, "win_probability": probability, "seed": seed})
    return fitted


def apply_interval(rows: list[dict], updated: tuple[float, float], h: float, start_k: int) -> list[dict]:
    out = []
    for row in rows:
        item = dict(row)
        if item["x_next"] is not None and item["k"] >= start_k:
            item["updated_set"] = list(updated)
            item["updated_certificate"] = interval_gain(
                item["x"], item["y"], item["x_next"], item["y_next"], updated, h,
            )
        out.append(item)
    return out


def binomial_information(rows: list[dict], preference: float, labels: int, h: float) -> dict:
    probability = float(expit(SENSITIVITY * (preference - 0.5)))
    counts = np.arange(labels + 1)
    pmf = binom.pmf(counts, labels, probability)
    terminal = np.empty(labels + 1)
    covered = np.empty(labels + 1, dtype=bool)
    certified = np.empty(labels + 1, dtype=bool)
    loss = []
    for wins in counts:
        fitted = fit_interval(int(wins), labels)
        if fitted["empty"]:
            terminal[wins] = -np.inf
            covered[wins] = False
            certified[wins] = False
            loss.append(None)
            continue
        updated = tuple(fitted["interval"])
        covered[wins] = updated[0] - 1e-12 <= preference <= updated[1] + 1e-12
        edited = apply_interval(rows, updated, h, FEEDBACK_STEP)
        terminal[wins] = edited[-2]["updated_certificate"]
        certified[wins] = terminal[wins] >= GAMMA - 1e-12
        loss.append(first_below(edited, "updated_certificate"))
    finite = np.isfinite(terminal)
    return {
        "win_probability": probability,
        "exact_coverage": float(pmf[covered].sum()),
        "exact_probability_terminal_certificate_meets_gamma": float(pmf[certified].sum()),
        "mean_terminal_certificate": float(pmf[finite] @ terminal[finite] / pmf[finite].sum()),
        "empty_probability": float(pmf[~finite].sum()),
    }


def controller_rollout(phi0, preference, h, draws, n, labels, mode: str, rho: float = RHO) -> tuple[list[dict], dict]:
    x, y = 0.8, 0.0
    phi_prev = phi0
    learned = U0
    fitted = {"a_hat": None, "interval": list(U0), "wins": None, "empty": False}
    rows = []
    max_move = 0.0
    for k in range(len(draws) if draws is not None else N_STEPS):
        if k == FEEDBACK_STEP and mode != "source_only":
            fitted = labels
            if not fitted["empty"]:
                learned = tuple(fitted["interval"])
        if k == 0:
            phi = phi0
        elif k < FEEDBACK_STEP or mode == "frozen":
            phi = scheduled_phi(k, h, phi0)
        else:
            low = max(0.0, phi_prev - rho * h)
            high = min(1.0, phi_prev + rho * h)
            if mode == "point_estimate":
                phi = float(np.clip(fitted["a_hat"], low, high))
            else:
                phi = best_phi(x, y, learned if mode != "source_only" else U0, low, high)
        move = abs(phi - phi_prev)
        max_move = max(max_move, move)
        exact = exact_grad(x, y, phi)
        if draws is None:
            grad = exact
        else:
            u_q, u_s = draws[k]
            grad = sample_grad(x, y, phi, u_q[:n], u_s[:n])
        x_next, y_next = advance(x, y, phi, grad, h)
        updated = learned if k >= FEEDBACK_STEP and mode != "source_only" else U0
        row = measure(k, h, x, y, x_next, y_next, phi, preference, grad, exact, U0, updated)
        row["phi_move"] = move
        row["a_hat"] = fitted.get("a_hat")
        rows.append(row)
        phi_prev = phi
        x, y = x_next, y_next
    rows.append(terminal_state(len(rows), h, x, y, phi_prev, preference))
    return rows, {"max_phi_move": max_move, "within_budget": max_move <= rho * h + 1e-12, "fit": fitted}


def write_rows(path: Path, rows: list[dict]) -> None:
    fields = ["branch", "seed", "n", "h", "k", "t", "t_end", "x", "y", "phi", "true_return",
              "true_finite_step_gain", "instantaneous_true_rate", "source_certificate",
              "updated_certificate", "pointwise", "gradient_error", "a_hat"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def stage_5a(out: Path, phi0: float, preference: float) -> dict:
    exact = population_rollout(phi0, preference, H, N_STEPS)
    contact = summarize_path(exact, 0.008)["first_source_loss"]
    if contact is None or abs(contact - 0.008) > 1.5 * H:
        raise AssertionError(f"population path did not cross near 0.008: {contact}")
    half = population_rollout(phi0, preference, H / 2, int(round(T_FINAL / (H / 2))))
    table = []
    raw = []
    for label, rows, h, n, seed in (
        ("exact_population", exact, H, 0, 0),
        ("exact_population_half", half, H / 2, 0, 0),
    ):
        stats = summarize_path(rows, contact)
        table.append({"update": label, "h": h, "n": "exact", "seed": seed, **stats})
        for row in rows:
            raw.append({"branch": label, "seed": seed, "n": n, "h": h, **row})
    sampled = {str(n): [] for n in (*SAMPLE_SIZES, PRECISION_N)}
    for seed in POLICY_SEEDS:
        draws = uniforms(seed, N_STEPS, PRECISION_N)
        for n in (*SAMPLE_SIZES, PRECISION_N):
            rows = sampled_rollout(phi0, preference, H, draws, n)
            sampled[str(n)].append(summarize_path(rows, contact))
            for row in rows:
                raw.append({"branch": f"sampled_{n}", "seed": seed, "n": n, "h": H, **row})
    packed = {}
    for n, runs in sampled.items():
        losses = [run["first_source_loss"] for run in runs]
        gains = [run["true_finite_step_gain_near_contact"] for run in runs]
        finite_losses = [item for item in losses if item is not None]
        packed[n] = {
            "losses": losses,
            "mean_loss": None if not finite_losses else float(np.mean(finite_losses)),
            "loss_std": None if len(finite_losses) < 2 else float(np.std(finite_losses)),
            "censored": sum(item is None for item in losses),
            "max_steps_from_exact": max(abs((loss if loss is not None else T_FINAL + H) - contact) / H for loss in losses),
            "true_gain_mean": float(np.mean(gains)),
            "true_gain_values": gains,
            "max_gradient_error": max(run["max_gradient_error"] for run in runs),
            "mean_gradient_error": float(np.mean([run["mean_gradient_error"] for run in runs])),
            "pointwise_above_gamma": all(run["pointwise_above_gamma"] for run in runs),
            "pointwise_min": min(run["pointwise_min"] for run in runs),
        }
    write_rows(out / "steps.csv", raw)
    (out / "exact_reference.json").write_text(json.dumps({
        "contact": contact,
        "summary": summarize_path(exact, contact),
        "half": summarize_path(half, contact),
    }, indent=2) + "\n")
    return {
        "contact_time": contact,
        "exact": summarize_path(exact, contact),
        "half": summarize_path(half, contact),
        "sampled": packed,
        "initial_source_certificate": instantaneous_certificate(0.8, 0.0, phi0, U0),
        "initial_true_return": true_return(0.8, 0.0, preference),
        "continuous_contact": ordinary_pg_check()["numerical_maximal_speed_boundary_time"],
        "all_controls_upper": ordinary_pg_check()["analytic_all_controls_boundary_time_upper"],
    }


def stage_5b(out: Path, phi0: float, preference: float) -> dict:
    exact = population_rollout(phi0, preference, H, N_STEPS)
    base = summarize_path(exact, 0.008)
    source = label_batch(preference, 512, seed=0, informative=False)
    rows_out = []
    per_seed = []
    binomial = {}
    for labels in (32, 128, 512):
        binomial[str(labels)] = binomial_information(exact, preference, labels, H)
        for seed in LABEL_SEEDS:
            fitted = label_batch(preference, labels, seed, informative=True)
            if fitted["empty"]:
                edited = exact
                stats = {"first_updated_loss": None, "terminal_updated_certificate": None}
            else:
                edited = apply_interval(exact, tuple(fitted["interval"]), H, FEEDBACK_STEP)
                stats = summarize_path(edited, base["first_source_loss"])
            pre = [row["updated_certificate"] for row in edited if row["k"] < FEEDBACK_STEP and row["updated_certificate"] is not None]
            original = [row["source_certificate"] for row in exact if row["source_certificate"] is not None]
            if pre != [row["source_certificate"] for row in exact if row["k"] < FEEDBACK_STEP and row["source_certificate"] is not None]:
                raise AssertionError("feedback changed a pre-arrival certificate")
            covered = (not fitted["empty"]) and fitted["interval"][0] <= preference <= fitted["interval"][1]
            per_seed.append({
                "labels": labels, "seed": seed, "a_hat": fitted["a_hat"], "interval": fitted["interval"],
                "covers_a_star": covered, "wins": fitted["wins"],
                "original_loss": base["first_source_loss"],
                "updated_loss": stats["first_updated_loss"],
                "terminal_updated": stats["terminal_updated_certificate"],
                "terminal_true_gain_changed": False,
                "true_gain_near_contact": base["true_finite_step_gain_near_contact"],
            })
            for row in edited:
                rows_out.append({"branch": f"informative_{labels}", "seed": seed, "n": 0, "h": H, **row})
    write_rows(out / "steps.csv", rows_out)
    frozen_gain = base["true_finite_step_gain_near_contact"]
    return {
        "frozen_original_loss": base["first_source_loss"],
        "frozen_terminal_source_certificate": base["terminal_source_certificate"],
        "frozen_true_finite_step_gain_near_contact": frozen_gain,
        "source_repetition": source,
        "per_seed": per_seed,
        "binomial": binomial,
        "true_gain_identical_across_information": True,
    }


def stage_5c(out: Path, phi0: float, preference: float, labels: int) -> dict:
    exact = population_rollout(phi0, preference, H, N_STEPS)
    contact = summarize_path(exact, 0.008)["first_source_loss"]
    shared = [label_batch(preference, labels, seed, True) for seed in LABEL_SEEDS]
    primary = shared[0]
    if primary["empty"]:
        raise AssertionError("primary label batch has an empty confidence set")
    draws = {seed: uniforms(seed, N_STEPS, PRECISION_N) for seed in POLICY_SEEDS}
    branches = {
        "source_only_bounded_robust": ("source_only", None),
        "information_bounded_robust": ("robust", primary),
        "information_bounded_point": ("point_estimate", primary),
    }
    raw = []
    summary = {}
    for name, (mode, batch) in branches.items():
        runs = []
        for seed in POLICY_SEEDS:
            rows, audit = controller_rollout(
                phi0, preference, H, draws[seed], PRECISION_N, batch or {"a_hat": None, "interval": list(U0), "empty": False}, mode,
            )
            stats = summarize_path(rows, contact)
            stats["within_budget"] = audit["within_budget"]
            stats["max_phi_move"] = audit["max_phi_move"]
            runs.append(stats)
            for row in rows:
                raw.append({"branch": name, "seed": seed, "n": PRECISION_N, "h": H, **row})
        summary[name] = aggregate(runs)
    info_runs = []
    for batch in shared:
        edited = apply_interval(exact, tuple(batch["interval"]), H, FEEDBACK_STEP)
        stats = summarize_path(edited, contact)
        stats["a_hat"] = batch["a_hat"]
        stats["interval"] = batch["interval"]
        stats["covers_a_star"] = batch["interval"][0] <= preference <= batch["interval"][1]
        info_runs.append(stats)
        for row in edited:
            raw.append({"branch": "information_only_frozen", "seed": batch["seed"], "n": 0, "h": H, **row})
    population, population_audit = controller_rollout(
        phi0, preference, H, None, 0, primary, "robust",
    )
    population_stats = summarize_path(population, contact)
    population_stats["within_budget"] = population_audit["within_budget"]
    for row in population:
        raw.append({"branch": "population_exact_robust", "seed": 0, "n": 0, "h": H, **row})
    known, _ = controller_rollout(phi0, preference, H, None, 0, {
        "a_hat": preference, "interval": [preference, preference], "empty": False, "wins": None,
    }, "robust")
    fast, _ = controller_rollout(phi0, preference, H, None, 0, {
        "a_hat": None, "interval": list(U0), "empty": False, "wins": None,
    }, "robust", rho=100.0)
    write_rows(out / "steps.csv", raw)
    plot_branches(out / "certificates.png", exact, population, known)
    return {
        "labels": labels,
        "primary_label_seed": primary["seed"],
        "primary_fit": primary,
        "sampled": summary,
        "information_only": aggregate(info_runs) | {"per_seed": info_runs},
        "population_exact_robust": population_stats,
        "controls": {
            "known_preference_updated_loss": summarize_path(known, contact)["first_updated_loss"],
            "known_preference_source_loss": summarize_path(known, contact)["first_source_loss"],
            "known_preference_terminal_updated": summarize_path(known, contact)["terminal_updated_certificate"],
            "large_rho_source_loss": summarize_path(fast, contact)["first_source_loss"],
            "large_rho_terminal_source": summarize_path(fast, contact)["terminal_source_certificate"],
        },
        "exact_reference_loss": contact,
        "exact_true_gain_near_contact": summarize_path(exact, contact)["true_finite_step_gain_near_contact"],
    }


def aggregate(runs: list[dict]) -> dict:
    def values(key):
        return [run[key] for run in runs]

    losses = values("first_source_loss")
    updated = values("first_updated_loss")
    return {
        "seeds": len(runs),
        "source_losses": losses,
        "updated_losses": updated,
        "source_loss_mean": float(np.mean([item for item in losses if item is not None])) if any(losses) else None,
        "updated_loss_censored": sum(item is None for item in updated),
        "true_gain_near_contact_mean": float(np.mean(values("true_finite_step_gain_near_contact"))),
        "true_gain_near_contact": values("true_finite_step_gain_near_contact"),
        "terminal_updated": values("terminal_updated_certificate"),
        "terminal_source": values("terminal_source_certificate"),
        "pointwise_above_gamma": all(values("pointwise_above_gamma")),
        "within_budget": all(run.get("within_budget", True) for run in runs),
        "max_gradient_error": max(values("max_gradient_error")),
    }


def plot_branches(path: Path, exact: list[dict], population: list[dict], known: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    def series(rows, key):
        xs, ys = [], []
        for row in rows:
            if row["t_end"] is None:
                continue
            xs.append(row["t_end"])
            ys.append(row[key])
        return xs, ys
    plt.figure(figsize=(6.8, 3.8))
    plt.plot(*series(exact, "source_certificate"), label="Exact path, source set")
    plt.plot(*series(population, "updated_certificate"), label="Population robust, learned set")
    plt.plot(*series(known, "updated_certificate"), label="Known preference, no ambiguity")
    plt.axhline(GAMMA, ls="--", label="Required gain 0.06")
    plt.axvline(0.004, ls=":", label="Feedback arrival")
    plt.xlabel("Policy time")
    plt.ylabel("Finite-step gain rate")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("5A", "5B", "5C"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--labels", type=int, default=128)
    args = parser.parse_args()
    started = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=True)
    phi0 = reference_phi0()
    preference = phi0
    self_check(phi0)
    if args.stage == "5A":
        payload = stage_5a(args.out, phi0, preference)
        command = "python experiments/bridge/06_online_bandit.py --stage 5A --out outputs/horizon_bridge/step05a"
    elif args.stage == "5B":
        payload = stage_5b(args.out, phi0, preference)
        command = "python experiments/bridge/06_online_bandit.py --stage 5B --out outputs/horizon_bridge/step05b"
    else:
        payload = stage_5c(args.out, phi0, preference, args.labels)
        command = "python experiments/bridge/06_online_bandit.py --stage 5C --out outputs/horizon_bridge/step05c"
    payload.update({
        "status": "PASS",
        "stage": args.stage,
        "command": command,
        "phi0": phi0,
        "a_star_evaluator_only": preference,
        "gamma": GAMMA,
        "rho": RHO,
        "h": H,
        "steps": N_STEPS,
        "feedback_step": FEEDBACK_STEP,
        "source_set": list(U0),
        "scope": "Four-outcome Theorem 2 bridge. No Walker, Meta-World, or language run.",
    })
    write_summary(args.out, payload, started)


if __name__ == "__main__":
    main()
