#!/usr/bin/env python3
"""G4: one delayed binary transition on the four-outcome policy. Sixteen outcomes."""
from __future__ import annotations

import csv
import importlib.util
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.bridge.common import write_summary


def load_online():
    path = Path(__file__).resolve().parent / "06_online_bandit.py"
    spec = importlib.util.spec_from_file_location("online_bandit_g4", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M = load_online()
ROOT = Path(__file__).resolve().parents[2]
LAM = 0.2
KAPPA = 0.05
A_STAR = None  # filled from the frozen reference after import


def outcomes(x: float, y: float, zeta: float, lam: float = LAM):
    packed = []
    th = math.tanh(zeta)
    for q in (-1.0, 1.0):
        for s in (-1.0, 1.0):
            p_qs = 0.25 * (1.0 + x * q) * (1.0 + y * s)
            p_z1 = 0.5 * (1.0 + lam * q * s)
            for z, p_z in ((0.0, 1.0 - p_z1), (1.0, p_z1)):
                tilt = th if z == 1.0 else 0.0
                for u in (-1.0, 1.0):
                    p_u = 0.5 * (1.0 + u * tilt)
                    packed.append((p_qs * p_z * p_u, q, s, z, u))
    return packed


def expectation_zu(x: float, y: float, zeta: float, lam: float = LAM) -> float:
    return math.tanh(zeta) * (1.0 + lam * x * y) / 2.0


def true_return(x: float, y: float, zeta: float, preference: float, kappa: float, lam: float = LAM) -> float:
    return preference * x + (1.0 - preference) * y + kappa * expectation_zu(x, y, zeta, lam)


def logit_difference(first, second, preference: float, kappa: float) -> float:
    q1, s1, z1, u1 = first
    q2, s2, z2, u2 = second
    return preference * (q1 - q2) + (1.0 - preference) * (s1 - s2) + kappa * (z1 * u1 - z2 * u2)


def gradient(x: float, y: float, zeta: float, phi: float, kappa: float, lam: float = LAM) -> np.ndarray:
    total = np.zeros(3)
    th = math.tanh(zeta)
    sech2 = 1.0 - th * th
    for probability, q, s, z, u in outcomes(x, y, zeta, lam):
        reward = phi * q + (1.0 - phi) * s + kappa * z * u
        score_zeta = 0.0 if z == 0.0 else u * sech2 / (1.0 + u * th)
        total += probability * reward * np.array([q - x, s - y, score_zeta])
    return total


def rates(x: float, y: float, zeta: float, phi: float, kappa: float) -> tuple[float, float, float]:
    gq, gs, gz = gradient(x, y, zeta, phi, kappa)
    dx = (1.0 - x * x) * gq
    dy = (1.0 - y * y) * gs
    base = (1.0 + LAM * x * y) / 2.0
    th = math.tanh(zeta)
    dzu = (1.0 - th * th) * gz * base + th * (LAM / 2.0) * (dx * y + x * dy)
    return dx, dy, dzu


def instantaneous(x: float, y: float, zeta: float, phi: float, preference: float, kappa: float) -> float:
    dx, dy, dzu = rates(x, y, zeta, phi, kappa)
    return preference * dx + (1.0 - preference) * dy + kappa * dzu


def advance(x: float, y: float, zeta: float, grad: np.ndarray, h: float) -> tuple[float, float, float]:
    theta = np.array([np.arctanh(x), np.arctanh(y), zeta]) + h * grad
    return float(np.tanh(theta[0])), float(np.tanh(theta[1])), float(theta[2])


def finite_gain(x, y, zeta, x2, y2, zeta2, preference, kappa, h) -> float:
    return (true_return(x2, y2, zeta2, preference, kappa) - true_return(x, y, zeta, preference, kappa)) / h


def interval_gain(x, y, zeta, x2, y2, zeta2, bounds, kappa, h) -> float:
    return min(
        finite_gain(x, y, zeta, x2, y2, zeta2, bounds[0], kappa, h),
        finite_gain(x, y, zeta, x2, y2, zeta2, bounds[1], kappa, h),
    )


def instantaneous_certificate(x, y, zeta, phi, bounds, kappa) -> float:
    return min(
        instantaneous(x, y, zeta, phi, bounds[0], kappa),
        instantaneous(x, y, zeta, phi, bounds[1], kappa),
    )


def best_phi(x, y, zeta, kappa, bounds, low, high) -> float:
    def value(phi: float, preference: float) -> float:
        return instantaneous(x, y, zeta, phi, preference, kappa)

    def worst(phi: float) -> float:
        return min(value(phi, bounds[0]), value(phi, bounds[1]))

    slope0 = value(1.0, bounds[0]) - value(0.0, bounds[0])
    slope1 = value(1.0, bounds[1]) - value(0.0, bounds[1])
    candidates = [low, high]
    if abs(slope0 - slope1) > 1e-12:
        cross = (value(0.0, bounds[1]) - value(0.0, bounds[0])) / (slope0 - slope1)
        if low - 1e-12 <= cross <= high + 1e-12:
            candidates.append(float(np.clip(cross, low, high)))
    best = max(worst(phi) for phi in candidates)
    tied = [phi for phi in candidates if abs(worst(phi) - best) <= 1e-12]
    return max(tied)


def _grid():
    return [i / 200 for i in range(201)]


def pointwise_value(x, y, zeta, kappa) -> float:
    # The endpoint derivatives are affine in phi, so the maximum is attained
    # at an endpoint or at their crossing. Evaluating the grid only checks that.
    analytic = instantaneous_certificate(x, y, zeta, best_phi(x, y, zeta, kappa, M.U0, 0.0, 1.0), M.U0, kappa)
    gridded = max(instantaneous_certificate(x, y, zeta, phi, M.U0, kappa) for phi in _grid())
    if analytic + 1e-8 < gridded:
        raise AssertionError((analytic, gridded))
    return analytic


def self_check(phi0: float) -> None:
    total = sum(item[0] for item in outcomes(0.8, 0.0, 0.3))
    if abs(total - 1.0) > 1e-12 or any(item[0] < -1e-15 for item in outcomes(-0.4, 0.2, -0.7)):
        raise AssertionError("outcome probabilities are not a distribution")
    enumerated = 0.0
    for probability, q, s, z, u in outcomes(0.3, -0.2, 0.4):
        enumerated += probability * z * u
    if abs(enumerated - expectation_zu(0.3, -0.2, 0.4)) > 1e-12:
        raise AssertionError("closed-form delayed term disagrees with enumeration")
    matched = logit_difference((1, -1, 1, 1), (-1, 1, 1, 1), phi0, KAPPA)
    if abs(matched - (4.0 * phi0 - 2.0)) > 1e-12:
        raise AssertionError("matched informative logit changed")
    mismatched = logit_difference((1, -1, 1, 1), (-1, 1, 1, -1), phi0, KAPPA)
    if abs(mismatched - (4.0 * phi0 - 2.0 + 2.0 * KAPPA)) > 1e-12:
        raise AssertionError("delayed-reward offset is wrong")
    g0 = gradient(0.8, 0.0, 0.0, phi0, 0.0)
    if np.linalg.norm(g0[:2] - M.exact_grad(0.8, 0.0, phi0)) > 1e-12 or abs(g0[2]) > 1e-12:
        raise AssertionError("zero-kappa gradient does not recover the four-output gradient")
    for kappa in (0.0, KAPPA):
        g = gradient(0.8, 0.2, 0.15, phi0, kappa)
        base = true_return(0.8, 0.2, 0.15, phi0, kappa)
        eps = 1e-6
        for index, (tq, ts, tz) in enumerate((
            (np.arctanh(0.8), np.arctanh(0.2), 0.15),
        )):
            probes = [
                (tq + eps, ts, tz), (tq - eps, ts, tz),
                (tq, ts + eps, tz), (tq, ts - eps, tz),
                (tq, ts, tz + eps), (tq, ts, tz - eps),
            ]
        numerical = []
        specs = [(eps, 0, 0), (-eps, 0, 0), (0, eps, 0), (0, -eps, 0), (0, 0, eps), (0, 0, -eps)]
        values = []
        for dq, ds, dz in specs:
            x = math.tanh(np.arctanh(0.8) + dq)
            y = math.tanh(np.arctanh(0.2) + ds)
            values.append(true_return(x, y, 0.15 + dz, phi0, kappa))
        numerical = np.array([
            (values[0] - values[1]) / (2 * eps),
            (values[2] - values[3]) / (2 * eps),
            (values[4] - values[5]) / (2 * eps),
        ])
        if np.linalg.norm(numerical - g) > 1e-8:
            raise AssertionError((kappa, numerical, g))
        # Affinity in phi of each endpoint derivative.
        for preference in M.U0:
            v0 = instantaneous(0.8, 0.2, 0.15, 0.0, preference, kappa)
            v1 = instantaneous(0.8, 0.2, 0.15, 1.0, preference, kappa)
            mid = instantaneous(0.8, 0.2, 0.15, 0.3, preference, kappa)
            if abs(mid - (0.7 * v0 + 0.3 * v1)) > 1e-9:
                raise AssertionError("instantaneous rate is not affine in phi")
    if abs(pointwise_value(0.8, 0.0, 0.0, 0.0) - M.pointwise(0.8, 0.0)) > 1e-8:
        raise AssertionError("zero-kappa pointwise value changed")
    if abs(instantaneous(0.8, 0.0, 0.0, phi0, 2.0, 0.0) - M.instantaneous_rate(0.8, 0.0, phi0, 2.0)) > 1e-10:
        raise AssertionError("zero-kappa instantaneous certificate changed")


def rollout(phi0: float, kappa: float, draws, n: int | None, labels: dict | None, react: bool) -> list[dict]:
    x, y, zeta = 0.8, 0.0, 0.0
    phi_prev = phi0
    bounds = M.U0
    rows = []
    for k in range(M.N_STEPS):
        if labels is not None and k >= M.FEEDBACK_STEP and not labels.get("empty", False):
            bounds = tuple(labels["interval"])
        if k == 0 or not react:
            phi = M.scheduled_phi(k, M.H, phi0)
        else:
            low = max(0.0, phi_prev - M.RHO * M.H)
            high = min(1.0, phi_prev + M.RHO * M.H)
            active = bounds if k >= M.FEEDBACK_STEP else M.U0
            phi = best_phi(x, y, zeta, kappa, active, low, high)
        exact = gradient(x, y, zeta, phi, kappa)
        if draws is None:
            used = exact
        else:
            used = sample_gradient(x, y, zeta, phi, kappa, draws[k][0], draws[k][1], draws[k][2], draws[k][3], n)
        x2, y2, z2 = advance(x, y, zeta, used, M.H)
        xp, yp, zp = advance(x, y, zeta, exact, M.H)
        cert_bounds = bounds if k >= M.FEEDBACK_STEP and labels is not None else M.U0
        rows.append({
            "k": k, "t": k * M.H, "x": x, "y": y, "zeta": zeta, "phi": phi,
            "x_next": x2, "y_next": y2, "zeta_next": z2,
            "true_return": true_return(x, y, zeta, phi0, kappa),
            "true_finite_step_gain": finite_gain(x, y, zeta, x2, y2, z2, phi0, kappa, M.H),
            "source_certificate": interval_gain(x, y, zeta, x2, y2, z2, M.U0, kappa, M.H),
            "updated_certificate": interval_gain(x, y, zeta, x2, y2, z2, cert_bounds, kappa, M.H),
            "population_step": interval_gain(x, y, zeta, xp, yp, zp, M.U0, kappa, M.H),
            "instantaneous": instantaneous_certificate(x, y, zeta, phi, M.U0, kappa),
            "pointwise": pointwise_value(x, y, zeta, kappa),
            "gradient_error": float(np.linalg.norm(used - exact)),
            "phi_move": abs(phi - phi_prev),
        })
        x, y, zeta, phi_prev = x2, y2, z2, phi
    return rows


def sample_gradient(x, y, zeta, phi, kappa, u_q, u_s, u_z, u_u, n: int) -> np.ndarray:
    u_q, u_s, u_z, u_u = u_q[:n], u_s[:n], u_z[:n], u_u[:n]
    q = np.where(u_q < (1.0 + x) / 2.0, 1.0, -1.0)
    s = np.where(u_s < (1.0 + y) / 2.0, 1.0, -1.0)
    z = np.where(u_z < (1.0 + LAM * q * s) / 2.0, 1.0, 0.0)
    th = math.tanh(zeta)
    tilt = np.where(z > 0.5, th, 0.0)
    u = np.where(u_u < (1.0 + tilt) / 2.0, 1.0, -1.0)
    reward = phi * q + (1.0 - phi) * s + kappa * z * u
    score_zeta = np.where(z > 0.5, u * (1.0 - th * th) / (1.0 + u * th), 0.0)
    return np.array([
        float(np.mean(reward * (q - x))),
        float(np.mean(reward * (s - y))),
        float(np.mean(reward * score_zeta)),
    ])


def draws_for(seed: int, n: int):
    rng = np.random.default_rng(40_000 + seed)
    return [(rng.random(n), rng.random(n), rng.random(n), rng.random(n)) for _ in range(M.N_STEPS)]


def first_below(rows: list[dict], key: str) -> float | None:
    for row in rows:
        if row[key] < M.GAMMA - 1e-15:
            return float(row["t"])
    return None


def gain_at(rows: list[dict], when: float | None) -> float | None:
    if when is None:
        return None
    for row in rows:
        if abs(row["t"] - when) <= 1e-9:
            return float(row["true_finite_step_gain"])
    return None


def describe(name: str, rows: list[dict], grad_source: str, n, extra: dict | None = None) -> dict:
    loss = first_below(rows, "source_certificate")
    at = rows[0] if loss is None else next(row for row in rows if abs(row["t"] - loss) <= 1e-9)
    payload = {
        "name": name,
        "gradient": grad_source,
        "n": n,
        "first_source_loss": loss,
        "pointwise_at_loss": at["pointwise"],
        "pointwise_above_gamma": at["pointwise"] > M.GAMMA,
        "true_gain_at_loss": gain_at(rows, loss),
        "min_true_gain": min(row["true_finite_step_gain"] for row in rows),
        "mean_gradient_error": float(np.mean([row["gradient_error"] for row in rows])),
        "terminal_source": rows[-1]["source_certificate"],
        "terminal_updated": rows[-1]["updated_certificate"],
        "terminal_phi": rows[-1]["phi"],
        "terminal_zeta": rows[-1]["zeta_next"],
        "terminal_x": rows[-1]["x_next"],
        "terminal_y": rows[-1]["y_next"],
        "min_pointwise": min(row["pointwise"] for row in rows),
        "population_loss": first_below(rows, "population_step"),
        "max_phi_move": max(row["phi_move"] for row in rows),
    }
    if extra:
        payload.update(extra)
    return payload


def main() -> None:
    started = time.perf_counter()
    phi0 = M.reference_phi0()
    global A_STAR
    A_STAR = phi0
    self_check(phi0)
    out = ROOT / "outputs/horizon_bridge/step09_g4"
    out.mkdir(parents=True, exist_ok=True)

    zero = rollout(phi0, 0.0, None, None, None, False)
    reference = [row for row in M.population_rollout(phi0, phi0, M.H, M.N_STEPS) if row["x_next"] is not None]
    gaps = [max(abs(a["x_next"] - b["x_next"]), abs(a["y_next"] - b["y_next"]), abs(a["source_certificate"] - b["source_certificate"])) for a, b in zip(zero, reference)]
    if max(gaps) > 1e-9 or max(abs(row["zeta_next"]) for row in zero) > 1e-12:
        raise AssertionError("zero-kappa path does not reproduce Step 5A")
    if first_below(zero, "source_certificate") != 0.008:
        raise AssertionError("zero-kappa loss time moved")

    exact = rollout(phi0, KAPPA, None, None, None, False)
    labels = {seed: M.label_batch(phi0, 128, seed, True) for seed in M.LABEL_SEEDS}
    frozen = {seed: rollout(phi0, KAPPA, None, None, labels[seed], False) for seed in M.LABEL_SEEDS}
    sampled = {}
    guided = {}
    for n in (256, 2048):
        sampled[n] = []
        for seed in (0, 1, 2):
            sampled[n].append(rollout(phi0, KAPPA, draws_for(seed, n), n, None, False))
    for seed in (0, 1, 2):
        guided[seed] = rollout(phi0, KAPPA, draws_for(seed, 2048), 2048, labels[100], True)
    source_guided = [rollout(phi0, KAPPA, draws_for(seed, 2048), 2048, None, True) for seed in (0, 1, 2)]

    loss_row = next(row for row in exact if abs(row["t"] - (first_below(exact, "source_certificate") or -1)) <= 1e-9) if first_below(exact, "source_certificate") is not None else None
    diagnosis = {
        "source_certificate_ever_nonempty": any(row["source_certificate"] >= M.GAMMA for row in exact),
        "pointwise_ever_below_gamma": any(row["pointwise"] < M.GAMMA for row in exact),
        "scheduled_phi_matches_source_robust": max(
            abs(a["phi"] - b["phi"]) for a, b in zip(exact, rollout(phi0, KAPPA, None, None, None, True))
        ),
    }
    if loss_row is not None:
        diagnosis["pointwise_at_loss"] = loss_row["pointwise"]
        diagnosis["instantaneous_at_loss"] = loss_row["instantaneous"]

    frozen_summary = []
    for seed, rows in frozen.items():
        frozen_summary.append({
            "seed": seed,
            "interval": labels[seed]["interval"],
            "source_loss": first_below(rows, "source_certificate"),
            "updated_loss": first_below(rows, "updated_certificate"),
            "terminal_updated": rows[-1]["updated_certificate"],
            "suffix_certified": all(row["updated_certificate"] >= M.GAMMA - 1e-15 for row in rows if row["k"] >= M.FEEDBACK_STEP),
            "true_gain_gap_vs_exact": max(abs(a["true_finite_step_gain"] - b["true_finite_step_gain"]) for a, b in zip(rows, exact)),
        })

    variants = [
        describe("kappa0", zero, "exact", None, {"match_gap": max(gaps)}),
        describe("two_step_exact", exact, "exact", None),
    ]
    for n, paths in sampled.items():
        losses = [first_below(path, "source_certificate") for path in paths]
        pops = [first_below(path, "population_step") for path in paths]
        variants.append({
            **describe(f"sampled_{n}", paths[0], "sampled", n),
            "seeds": [0, 1, 2],
            "first_source_loss_by_seed": losses,
            "population_loss_by_seed": pops,
            "mean_gradient_error": float(np.mean([row["gradient_error"] for path in paths for row in path])),
            "true_gain_at_reference_0_008": [gain_at(path, 0.008) for path in paths],
        })
    variants.append({
        "name": "frozen_btl",
        "gradient": "exact",
        "n": 128,
        "labels": frozen_summary,
    })
    variants.append({
        "name": "bounded_guidance",
        "gradient": "sampled",
        "n": 2048,
        "label_seed": 100,
        "interval": labels[100]["interval"],
        "by_seed": [describe(f"guided_{seed}", guided[seed], "sampled", 2048) for seed in (0, 1, 2)],
        "phi_gap_vs_source_controller": [
            max(abs(a["phi"] - b["phi"]) for a, b in zip(guided[seed], source_guided[seed])) for seed in (0, 1, 2)
        ],
    })

    with (out / "steps.csv").open("w", newline="") as handle:
        fieldnames = list(exact[0])
        writer = csv.DictWriter(handle, fieldnames=["variant", "seed", *fieldnames])
        writer.writeheader()
        for row in zero:
            writer.writerow({"variant": "kappa0", "seed": 0, **row})
        for row in exact:
            writer.writerow({"variant": "exact", "seed": 0, **row})
        for n, paths in sampled.items():
            for seed, path in zip((0, 1, 2), paths):
                for row in path:
                    writer.writerow({"variant": f"sampled_{n}", "seed": seed, **row})
        for seed, path in frozen.items():
            for row in path:
                writer.writerow({"variant": "frozen_btl", "seed": seed, **row})
        for seed, path in guided.items():
            for row in path:
                writer.writerow({"variant": "guided", "seed": seed, **row})

    def clean(obj):
        if isinstance(obj, dict):
            return {key: clean(value) for key, value in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [clean(value) for value in obj]
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        return obj

    write_summary(out, clean({"variants": variants, "diagnosis": diagnosis, "frozen_btl": frozen_summary}), started)


if __name__ == "__main__":
    main()
