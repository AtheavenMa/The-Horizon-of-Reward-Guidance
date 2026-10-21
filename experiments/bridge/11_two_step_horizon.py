#!/usr/bin/env python3
"""G4.1: extend the same two-step MDP from 24 updates to 32. Does not rewrite G4."""
from __future__ import annotations

import csv
import importlib.util
import math
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.bridge.common import write_summary


def load_two_step():
    path = Path(__file__).resolve().parent / "10_two_step_mdp.py"
    spec = importlib.util.spec_from_file_location("two_step_mdp_g41", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


T = load_two_step()
M = T.M
ROOT = Path(__file__).resolve().parents[2]
N_STEPS = 32
N_SAMPLE = 2048
POLICY_SEEDS = (0, 1, 2)
XATOL = 1e-12


def scheduled_phi(k: int, h: float, phi0: float) -> float:
    return phi0 + M.RHO * k * h


def gradient_basis(x: float, y: float, zeta: float, kappa: float) -> tuple[np.ndarray, np.ndarray]:
    """Reward is affine in phi, so the score gradient is g0 + phi * g1."""
    g0 = T.gradient(x, y, zeta, 0.0, kappa)
    g1 = T.gradient(x, y, zeta, 1.0, kappa) - g0
    return g0, g1


def finite_worst(x, y, zeta, grad, bounds, kappa, h) -> float:
    x2, y2, z2 = T.advance(x, y, zeta, grad, h)
    return T.interval_gain(x, y, zeta, x2, y2, z2, bounds, kappa, h)


def finite_pointwise(x, y, zeta, kappa, h) -> dict:
    g0, g1 = gradient_basis(x, y, zeta, kappa)
    probe = g0 + 0.3 * g1
    if np.linalg.norm(probe - T.gradient(x, y, zeta, 0.3, kappa)) > 1e-10:
        raise AssertionError("score gradient is not affine in phi")
    phis = np.linspace(0.0, 1.0, 2001)
    grads = g0 + phis[:, None] * g1
    x2 = np.tanh(np.arctanh(x) + h * grads[:, 0])
    y2 = np.tanh(np.arctanh(y) + h * grads[:, 1])
    z2 = zeta + h * grads[:, 2]
    delayed_now = math.tanh(zeta) * (1.0 + T.LAM * x * y) / 2.0
    delayed_next = np.tanh(z2) * (1.0 + T.LAM * x2 * y2) / 2.0
    base = ((y2 - y) + kappa * (delayed_next - delayed_now)) / h
    slope = ((x2 - x) - (y2 - y)) / h
    worst = np.minimum(base + M.U0[0] * slope, base + M.U0[1] * slope)
    best_grid = float(worst.max())

    def worst_scalar(phi: float) -> float:
        grad = g0 + float(phi) * g1
        return finite_worst(x, y, zeta, grad, M.U0, kappa, h)

    check_phi = float(phis[int(np.argmax(worst))])
    if abs(worst_scalar(check_phi) - best_grid) > 1e-9:
        raise AssertionError((worst_scalar(check_phi), best_grid, check_phi))
    fit = minimize_scalar(lambda phi: -worst_scalar(float(phi)), bounds=(0.0, 1.0), method="bounded", options={"xatol": XATOL})
    if not fit.success:
        raise RuntimeError(fit.message)
    refined = float(worst_scalar(fit.x))
    if refined + 1e-8 < best_grid:
        raise AssertionError((refined, best_grid))
    # The vectorized return must match the enumerated finite-step gain at the refined phi.
    if abs(refined - worst_scalar(float(fit.x))) > 0.0:
        raise AssertionError("refined value did not repeat")
    return {
        "value": max(refined, best_grid),
        "phi": float(fit.x),
        "grid_gap": refined - best_grid,
        "xatol": XATOL,
    }


def exact_path(phi0: float, kappa: float, n_steps: int, h: float) -> list[dict]:
    x, y, zeta = 0.8, 0.0, 0.0
    rows = []
    for k in range(n_steps):
        phi = scheduled_phi(k, h, phi0)
        grad = T.gradient(x, y, zeta, phi, kappa)
        x2, y2, z2 = T.advance(x, y, zeta, grad, h)
        pointwise = finite_pointwise(x, y, zeta, kappa, h)
        rows.append({
            "k": k,
            "t": k * h,
            "x": x,
            "y": y,
            "zeta": zeta,
            "phi": phi,
            "x_next": x2,
            "y_next": y2,
            "zeta_next": z2,
            "source_certificate": T.interval_gain(x, y, zeta, x2, y2, z2, M.U0, kappa, h),
            "true_finite_step_gain": T.finite_gain(x, y, zeta, x2, y2, z2, phi0, kappa, h),
            "finite_step_pointwise": pointwise["value"],
            "finite_step_phi": pointwise["phi"],
            "pointwise_grid_gap": pointwise["grid_gap"],
            "instantaneous_pointwise": T.pointwise_value(x, y, zeta, kappa),
            "min_probability": min(item[0] for item in T.outcomes(x, y, zeta)),
        })
        x, y, zeta = x2, y2, z2
    return rows


def draws(seed: int, n_steps: int, n: int):
    rng = np.random.default_rng(40_000 + seed)
    return [(rng.random(n), rng.random(n), rng.random(n), rng.random(n)) for _ in range(n_steps)]


def sampled_path(phi0: float, kappa: float, seed: int, n_steps: int, h: float) -> list[dict]:
    x, y, zeta = 0.8, 0.0, 0.0
    rows = []
    for k, sample in enumerate(draws(seed, n_steps, N_SAMPLE)):
        phi = scheduled_phi(k, h, phi0)
        exact = T.gradient(x, y, zeta, phi, kappa)
        used = T.sample_gradient(x, y, zeta, phi, kappa, *sample, N_SAMPLE)
        x2, y2, z2 = T.advance(x, y, zeta, used, h)
        xp, yp, zp = T.advance(x, y, zeta, exact, h)
        rows.append({
            "k": k,
            "t": k * h,
            "x": x,
            "y": y,
            "zeta": zeta,
            "phi": phi,
            "x_next": x2,
            "y_next": y2,
            "zeta_next": z2,
            "population_x_next": xp,
            "population_y_next": yp,
            "population_zeta_next": zp,
            "source_certificate": T.interval_gain(x, y, zeta, x2, y2, z2, M.U0, kappa, h),
            "population_step": T.interval_gain(x, y, zeta, xp, yp, zp, M.U0, kappa, h),
            "true_finite_step_gain": T.finite_gain(x, y, zeta, x2, y2, z2, phi0, kappa, h),
            "gradient_error": float(np.linalg.norm(used - exact)),
        })
        x, y, zeta = x2, y2, z2
    return rows


def first_below(rows: list[dict], key: str) -> dict | None:
    for row in rows:
        if row[key] < M.GAMMA - 1e-15:
            return row
    return None


def previous_certified(rows: list[dict], key: str) -> dict | None:
    last = None
    for row in rows:
        if row[key] < M.GAMMA - 1e-15:
            return last
        last = row
    return last


def apply_interval(rows: list[dict], bounds: tuple[float, float] | None, kappa: float, h: float, realized: bool) -> list[dict]:
    out = []
    for row in rows:
        item = dict(row)
        if row["k"] < M.FEEDBACK_STEP or bounds is None:
            item["updated_certificate"] = row["source_certificate"]
        elif realized:
            item["updated_certificate"] = T.interval_gain(
                row["x"], row["y"], row["zeta"], row["x_next"], row["y_next"], row["zeta_next"], bounds, kappa, h,
            )
        else:
            item["updated_certificate"] = T.interval_gain(
                row["x"], row["y"], row["zeta"], row["x_next"], row["y_next"], row["zeta_next"], bounds, kappa, h,
            )
        out.append(item)
    return out


def certified(rows: list[dict], key: str, after_only: bool) -> bool:
    chosen = [row for row in rows if row["t"] >= M.FEEDBACK_STEP * M.H - 1e-12 or not after_only]
    return all(row[key] >= M.GAMMA - 1e-15 for row in chosen)


def compare_prefix(fresh: list[dict], stored: list[dict]) -> float:
    keys = ("x", "y", "zeta", "phi", "source_certificate", "true_finite_step_gain")
    worst = 0.0
    for new, old in zip(fresh, stored):
        for key in keys:
            worst = max(worst, abs(float(new[key]) - float(old[key])))
    return worst


def load_stored(variant: str, seed: str = "0") -> list[dict]:
    path = ROOT / "outputs/horizon_bridge/step09_g4/steps.csv"
    with path.open() as handle:
        rows = [row for row in csv.DictReader(handle) if row["variant"] == variant and row["seed"] == seed]
    rows.sort(key=lambda row: int(row["k"]))
    return rows


def loss_pack(rows: list[dict], key: str = "source_certificate") -> dict:
    loss = first_below(rows, key)
    previous = previous_certified(rows, key)
    if loss is None:
        return {
            "first_loss": None,
            "last_certified_start": rows[-1]["t"] if rows else None,
            "source_gain_at_loss": None,
            "finite_step_pointwise_at_loss": None,
            "instantaneous_pointwise_at_loss": None,
            "true_gain_at_loss": None,
            "phi": None,
            "x": None,
            "y": None,
            "zeta": None,
            "min_probability": None,
            "last_step_source_gain": rows[-1]["source_certificate"],
            "censored": True,
        }
    return {
        "first_loss": loss["t"],
        "step": loss["k"],
        "last_certified_start": None if previous is None else previous["t"],
        "source_gain_at_loss": loss["source_certificate"],
        "finite_step_pointwise_at_loss": loss.get("finite_step_pointwise"),
        "instantaneous_pointwise_at_loss": loss.get("instantaneous_pointwise"),
        "true_gain_at_loss": loss["true_finite_step_gain"],
        "phi": loss["phi"],
        "x": loss["x"],
        "y": loss["y"],
        "zeta": loss["zeta"],
        "min_probability": loss.get("min_probability"),
        "last_step_source_gain": rows[-1]["source_certificate"],
        "censored": False,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(exact: list[dict], updated: dict[int, list[dict]], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 3.6))
    times = [row["t"] for row in exact]
    axis.plot(times, [row["source_certificate"] for row in exact], label="source certificate")
    for seed, rows in updated.items():
        axis.plot(times, [row["updated_certificate"] for row in rows], lw=0.8, label=f"updated seed {seed}")
    axis.axhline(M.GAMMA, color="black", lw=0.8, label="required gain")
    axis.axvline(0.012, color="gray", ls="--", lw=0.8, label="new step t=0.012")
    axis.set_xlabel("time")
    axis.set_ylabel("finite-step source gain")
    axis.set_title("kappa=0.05 exact path")
    axis.legend(fontsize=7, ncol=2)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def main() -> None:
    started = time.perf_counter()
    phi0 = M.reference_phi0()
    T.self_check(phi0)
    out = ROOT / "outputs/horizon_bridge/step10_g4_extension"
    out.mkdir(parents=True, exist_ok=True)
    h = M.H

    zero = exact_path(phi0, 0.0, N_STEPS, h)
    delayed = exact_path(phi0, T.KAPPA, N_STEPS, h)
    zero_gap = compare_prefix(zero[: M.N_STEPS], load_stored("kappa0"))
    delayed_gap = compare_prefix(delayed[: M.N_STEPS], load_stored("exact"))
    if zero_gap > 1e-12 or delayed_gap > 1e-12:
        raise AssertionError((zero_gap, delayed_gap))

    step25 = delayed[24]
    if abs(step25["t"] - 0.012) > 1e-15:
        raise AssertionError(step25["t"])
    ambiguous = abs(step25["source_certificate"] - M.GAMMA) < 1e-8
    labels = {seed: M.label_batch(phi0, 128, seed, True) for seed in M.LABEL_SEEDS}
    source_batches = {seed: M.label_batch(phi0, 128, seed, False) for seed in M.LABEL_SEEDS}
    informative = {}
    source_only_feedback = {}
    for seed, batch in labels.items():
        if batch["empty"] or batch["interval"] is None:
            raise RuntimeError(seed)
        informative[seed] = apply_interval(delayed, tuple(batch["interval"]), T.KAPPA, h, False)
        gain_gap = max(abs(a["true_finite_step_gain"] - b["true_finite_step_gain"]) for a, b in zip(informative[seed], delayed))
        if gain_gap > 0.0 or any(a["phi"] != b["phi"] or a["x"] != b["x"] or a["zeta"] != b["zeta"] for a, b in zip(informative[seed], delayed)):
            raise AssertionError("labels changed the exact trajectory")
    for seed, batch in source_batches.items():
        edited = apply_interval(delayed, None, T.KAPPA, h, False)
        if any(abs(a["updated_certificate"] - b["source_certificate"]) > 0.0 for a, b in zip(edited, delayed)):
            raise AssertionError("source comparisons changed the certificate")
        source_only_feedback[seed] = edited
        if list(batch["interval"]) != list(M.U0):
            raise AssertionError(batch["interval"])

    sampled = {seed: sampled_path(phi0, T.KAPPA, seed, N_STEPS, h) for seed in POLICY_SEEDS}
    sample_gaps = {
        seed: compare_prefix(sampled[seed][: M.N_STEPS], load_stored("sampled_2048", str(seed)))
        for seed in POLICY_SEEDS
    }
    if max(sample_gaps.values()) > 1e-12:
        raise AssertionError(sample_gaps)
    sampled_feedback = {
        (policy, label): apply_interval(sampled[policy], tuple(labels[label]["interval"]), T.KAPPA, h, True)
        for policy in POLICY_SEEDS
        for label in M.LABEL_SEEDS
    }
    for (policy, label), rows in sampled_feedback.items():
        if any(abs(a["true_finite_step_gain"] - b["true_finite_step_gain"]) > 0.0 for a, b in zip(rows, sampled[policy])):
            raise AssertionError("labels changed a sampled trajectory")

    def feedback_row(name: str, groups: dict, label_count: int) -> dict:
        originals, updated_losses, terminals, entire, suffix = [], [], [], [], []
        intervals = []
        for key, rows in groups.items():
            originals.append(first_below(rows, "source_certificate"))
            updated = first_below(rows, "updated_certificate")
            updated_losses.append(None if updated is None else updated["t"])
            terminals.append(rows[-1]["updated_certificate"])
            entire.append(certified(rows, "updated_certificate", False))
            suffix.append(certified(rows, "updated_certificate", True))
            intervals.append(rows[M.FEEDBACK_STEP]["updated_certificate"])
        return {
            "name": name,
            "labels": label_count,
            "pairings": len(groups),
            "original_first_loss": [None if row is None else row["t"] for row in originals],
            "updated_first_loss": updated_losses,
            "updated_last_step": terminals,
            "entire_fraction": float(np.mean(entire)),
            "suffix_fraction": float(np.mean(suffix)),
            "true_gain_changed": False,
        }

    exact_source = {0: apply_interval(delayed, None, T.KAPPA, h, False)}
    plot(delayed, informative, out / "certificates.png")
    exact_rows = []
    for name, rows in (("kappa0", zero), ("kappa005", delayed)):
        for row in rows:
            exact_rows.append({"model": name, **row})
    write_csv(out / "exact_steps.csv", exact_rows)
    sampled_rows = []
    for seed, rows in sampled.items():
        for row in rows:
            sampled_rows.append({
                "seed": seed, "label_seed": "", "evidence": "source", "k": row["k"], "t": row["t"],
                "x": row["x"], "y": row["y"], "zeta": row["zeta"], "phi": row["phi"],
                "source_certificate": row["source_certificate"], "updated_certificate": row["source_certificate"],
                "population_step": row["population_step"], "true_finite_step_gain": row["true_finite_step_gain"],
                "gradient_error": row["gradient_error"],
            })
    for (policy, label), rows in sampled_feedback.items():
        base = sampled[policy]
        for row, origin in zip(rows, base):
            sampled_rows.append({
                "seed": policy, "label_seed": label, "evidence": "informative", "k": row["k"], "t": row["t"],
                "x": origin["x"], "y": origin["y"], "zeta": origin["zeta"], "phi": origin["phi"],
                "source_certificate": row["source_certificate"], "updated_certificate": row["updated_certificate"],
                "population_step": origin["population_step"], "true_finite_step_gain": row["true_finite_step_gain"],
                "gradient_error": origin["gradient_error"],
            })
    write_csv(out / "sampled_steps.csv", sampled_rows)

    population_losses = [None if first_below(rows, "population_step") is None else first_below(rows, "population_step")["t"] for rows in sampled.values()]
    realized_losses = [None if first_below(rows, "source_certificate") is None else first_below(rows, "source_certificate")["t"] for rows in sampled.values()]
    informative_summary = feedback_row("informative", informative, 128)
    unresolved = len(set(informative_summary["updated_first_loss"])) > 1 and None in informative_summary["updated_first_loss"]

    write_summary(out, {
        "config": {
            "n_steps": N_STEPS, "h": h, "kappa": T.KAPPA, "lambda": T.LAM,
            "phi0": phi0, "gamma": M.GAMMA, "optimizer_xatol": XATOL,
            "half_step_run": False, "half_step_reason": "crossing margin is larger than 1e-8" if not ambiguous else "ambiguous",
            "n512_run": unresolved,
            "bounded_controller_run": False,
            "bounded_controller_reason": "N=128 did not change the robust direction in G3 or G4; repeating that controller is not a new question",
        },
        "agreement": {"kappa0": zero_gap, "kappa005": delayed_gap, "sampled_2048": sample_gaps},
        "kappa0_24": loss_pack(zero[:24]),
        "kappa005_24": loss_pack(delayed[:24]),
        "kappa005_25": loss_pack(delayed[:25]),
        "kappa005_32": loss_pack(delayed),
        "kappa0_32": loss_pack(zero),
        "prediction_check": {
            "t_0_0115": delayed[23]["source_certificate"],
            "t_0_0120": delayed[24]["source_certificate"],
            "true_gain_t_0_0120": delayed[24]["true_finite_step_gain"],
            "finite_pointwise_t_0_0120": delayed[24]["finite_step_pointwise"],
        },
        "max_pointwise_grid_gap": max(row["pointwise_grid_gap"] for row in zero + delayed),
        "feedback": {
            "source": feedback_row("source", exact_source, 0),
            "repeated_source": feedback_row("repeated_source", source_only_feedback, 128),
            "informative": informative_summary,
            "intervals": {str(seed): labels[seed]["interval"] for seed in labels},
        },
        "sampled": {
            "realized_first_loss": realized_losses,
            "population_first_loss": population_losses,
            "mean_gradient_error": float(np.mean([row["gradient_error"] for rows in sampled.values() for row in rows])),
            "informative": feedback_row("sampled_informative", sampled_feedback, 128),
            "true_gain_at_0_012": [next(row["true_finite_step_gain"] for row in rows if abs(row["t"] - 0.012) <= 1e-12) for rows in sampled.values()],
        },
    }, started)


if __name__ == "__main__":
    main()
