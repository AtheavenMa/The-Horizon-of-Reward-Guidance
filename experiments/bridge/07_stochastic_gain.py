#!/usr/bin/env python3
"""G1: population-step counterfactual versus the realized REINFORCE step.

Reads the committed Step 5A/5C traces. Does not rewrite them.
"""
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.bridge.common import write_summary


def load_online():
    path = Path(__file__).resolve().parent / "06_online_bandit.py"
    spec = importlib.util.spec_from_file_location("online_bandit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M = load_online()
ROOT = Path(__file__).resolve().parents[2]
GAMMA = M.GAMMA
U0 = M.U0
NEW_SEEDS = tuple(range(20, 30))


def read_rows(path: Path) -> list[dict]:
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in list(row.items()):
            if value == "" or value is None:
                row[key] = None
            else:
                try:
                    row[key] = float(value) if key not in {"branch"} else value
                except ValueError:
                    pass
        if row.get("k") is not None:
            row["k"] = int(row["k"])
        if row.get("seed") is not None:
            row["seed"] = int(row["seed"])
        if row.get("n") is not None:
            row["n"] = int(row["n"])
    return rows


def transitions(rows: list[dict]) -> list[dict]:
    return [row for row in rows if row.get("source_certificate") is not None]


def counterfactual(row: dict) -> tuple[float, float, float]:
    step = float(row.get("h", M.H))
    grad = M.exact_grad(row["x"], row["y"], row["phi"])
    x_next, y_next = M.advance(row["x"], row["y"], row["phi"], grad, step)
    pop = M.interval_gain(row["x"], row["y"], x_next, y_next, U0, step)
    inst = M.instantaneous_certificate(row["x"], row["y"], row["phi"], U0)
    true = M.finite_gain(row["x"], row["y"], x_next, y_next, M.reference_phi0(), step)
    return pop, inst, true


def first_time(rows: list[dict], key: str) -> float | None:
    for row in rows:
        value = row.get(key)
        if value is not None and value < GAMMA - 1e-15:
            return float(row["t"])
    return None


def score_covariance(x: float, y: float, phi: float) -> tuple[np.ndarray, np.ndarray]:
    mean = np.zeros(2)
    second = np.zeros((2, 2))
    for q in (-1.0, 1.0):
        for s in (-1.0, 1.0):
            probability = 0.25 * (1.0 + x * q) * (1.0 + y * s)
            reward = phi * q + (1.0 - phi) * s
            score = reward * np.array([q - x, s - y])
            mean += probability * score
            second += probability * np.outer(score, score)
    return mean, second - np.outer(mean, mean)


def endpoint_se(x: float, y: float, phi: float, preference: float, n: int, cov: np.ndarray) -> float:
    direction = np.array([(1.0 - x * x) * preference, (1.0 - y * y) * (1.0 - preference)])
    variance = float(direction @ cov @ direction) / n
    return math.sqrt(max(variance, 0.0))


def attach(rows: list[dict]) -> list[dict]:
    out = []
    for row in transitions(rows):
        pop, inst, true_pop = counterfactual(row)
        item = dict(row)
        item["population_step"] = pop
        item["instantaneous"] = inst
        item["population_true_gain"] = true_pop
        out.append(item)
    return out


def pack(name: str, groups: dict[int, list[dict]], n: int, se_ratio: float) -> dict:
    realized, population, instantaneous, true_gain = [], [], [], []
    gaps = []
    for seed in sorted(groups):
        rows = groups[seed]
        realized.append(first_time(rows, "source_certificate"))
        population.append(first_time(rows, "population_step"))
        instantaneous.append(first_time(rows, "instantaneous"))
        near = [row["true_finite_step_gain"] for row in rows if abs(row["t"] - 0.008) <= 1e-9]
        true_gain.append(near[0] if near else None)
        gaps.extend(abs(row["source_certificate"] - row["population_step"]) for row in rows if row["k"] == 0)
    return {
        "name": name,
        "n": n,
        "seeds": sorted(groups),
        "first_realized": realized,
        "first_population": population,
        "first_instantaneous": instantaneous,
        "se_over_margin": se_ratio,
        "true_gain_at_0_008": true_gain,
        "first_step_abs_gap_mean": float(np.mean(gaps)) if gaps else 0.0,
    }


def fmt_times(values: list[float | None]) -> str:
    shown = ["censored" if value is None else f"{value:.4g}" for value in values]
    return ", ".join(shown)


def replay_matches(phi0: float) -> dict:
    """Regenerate the stored sampler and compare certificates. Does not overwrite traces."""
    stored = read_rows(ROOT / "outputs/horizon_bridge/step05a/steps.csv")
    by_key = {}
    for row in stored:
        if row["branch"].startswith("sampled_"):
            by_key[(row["branch"], row["seed"], row["k"])] = row["source_certificate"]
    worst = 0.0
    checked = 0
    for seed in M.POLICY_SEEDS:
        # Step 5A drew one long stream and used prefixes, so the paired s-sample
        # is not the stream produced by asking for the smaller length directly.
        draws = M.uniforms(seed, M.N_STEPS, M.PRECISION_N)
        for n in (*M.SAMPLE_SIZES, M.PRECISION_N):
            fresh = M.sampled_rollout(phi0, phi0, M.H, draws, n)
            branch = f"sampled_{n}"
            for row in fresh:
                if row["source_certificate"] is None:
                    continue
                key = (branch, seed, row["k"])
                worst = max(worst, abs(row["source_certificate"] - by_key[key]))
                checked += 1
    if worst > 1e-9:
        raise AssertionError(f"replay disagrees with the committed trace by {worst}")
    return {"checked_transitions": checked, "max_abs_certificate_gap": worst}


def new_smoke(phi0: float) -> dict:
    realized, population, true_gain = [], [], []
    for seed in NEW_SEEDS:
        draws = M.uniforms(seed, M.N_STEPS, 2048)
        rows = attach(M.sampled_rollout(phi0, phi0, M.H, draws, 2048))
        for row in rows:
            row["source_certificate"] = row["source_certificate"]
        realized.append(first_time(rows, "source_certificate"))
        population.append(first_time(rows, "population_step"))
        true_gain.append(next(row["true_finite_step_gain"] for row in rows if abs(row["t"] - 0.008) <= 1e-9))
    return {
        "seeds": list(NEW_SEEDS),
        "n": 2048,
        "first_realized": realized,
        "first_population": population,
        "true_gain_at_0_008": true_gain,
    }


def plot_example(rows: list[dict], se_rows: list[dict], path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    times = [row["t"] for row in rows]
    axes[0].plot(times, [row["population_step"] for row in rows], label="population step")
    axes[0].plot(times, [row["source_certificate"] for row in rows], label="realized step")
    axes[0].axhline(GAMMA, color="black", lw=0.8, label="required gain")
    axes[0].axvline(0.00783587, color="gray", ls="--", lw=0.8, label="continuous contact")
    axes[0].set_xlabel("time")
    axes[0].set_ylabel("source finite-step gain")
    axes[0].set_title("seed 0, 8192 samples")
    axes[0].legend(fontsize=8)
    axes[1].bar([row["label"] for row in se_rows], [row["ratio"] for row in se_rows], color="#4c78a8")
    axes[1].set_ylabel("initial SE / margin")
    axes[1].set_title("worst endpoint a=2")
    axes[1].tick_params(axis="x", labelrotation=20)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def main() -> None:
    started = time.perf_counter()
    phi0 = M.reference_phi0()
    M.self_check(phi0)
    out = ROOT / "outputs/horizon_bridge/step06_g1"
    out.mkdir(parents=True, exist_ok=True)

    mean, cov = score_covariance(0.8, 0.0, phi0)
    if np.linalg.norm(mean - M.exact_grad(0.8, 0.0, phi0)) > 1e-12:
        raise AssertionError("enumerated score mean is not the population gradient")
    one_sample = endpoint_se(0.8, 0.0, phi0, 2.0, 1, cov)
    margin = M.instantaneous_certificate(0.8, 0.0, phi0, U0) - GAMMA
    if abs(one_sample - 0.8550) > 5e-3:
        raise AssertionError(f"one-sample scale {one_sample} is not near 0.8550")

    step5a = attach(read_rows(ROOT / "outputs/horizon_bridge/step05a/steps.csv"))
    step5c = attach(read_rows(ROOT / "outputs/horizon_bridge/step05c/steps.csv"))
    grouped: dict[str, dict[int, list[dict]]] = {}
    for row in step5a:
        grouped.setdefault(row["branch"], {}).setdefault(row["seed"], []).append(row)

    se_by_n = {}
    for n in (256, 2048, 8192, M.PRECISION_N):
        se_by_n[n] = endpoint_se(0.8, 0.0, phi0, 2.0, n, cov) / margin

    rows_out = [
        pack("exact", grouped["exact_population"], 0, 0.0),
        pack("256", grouped["sampled_256"], 256, se_by_n[256]),
        pack("2048", grouped["sampled_2048"], 2048, se_by_n[2048]),
        pack("8192", grouped["sampled_8192"], 8192, se_by_n[8192]),
        pack("2097152", grouped["sampled_2097152"], M.PRECISION_N, se_by_n[M.PRECISION_N]),
    ]
    half = pack("exact_half", grouped["exact_population_half"], 0, 0.0)
    source_only = [row for row in step5c if row["branch"] == "source_only_bounded_robust"]
    source_groups = {}
    for row in source_only:
        source_groups.setdefault(row["seed"], []).append(row)
    side = pack("5c_source_only", source_groups, M.PRECISION_N, se_by_n[M.PRECISION_N])

    replay = replay_matches(phi0)
    smoke = new_smoke(phi0)
    example = grouped["sampled_8192"][0]
    se_rows = [{"label": f"n={n}", "ratio": se_by_n[n]} for n in (256, 2048, 8192, M.PRECISION_N)]
    plot_example(example, se_rows, out / "population_vs_realized.png")

    with (out / "recomputed.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["branch", "seed", "k", "t", "source_certificate", "population_step", "instantaneous", "true_finite_step_gain"],
        )
        writer.writeheader()
        for row in step5a:
            writer.writerow({key: row.get(key) for key in writer.fieldnames})

    write_summary(out, {
        "one_sample_sd_a2": one_sample,
        "instantaneous_margin": margin,
        "analytic_se_over_margin": se_by_n,
        "table": rows_out,
        "half_step_exact": half,
        "step5c_source_only": side,
        "replay_check": replay,
        "new_2048_seeds": smoke,
        "figure": str(out / "population_vs_realized.png"),
    }, started)


if __name__ == "__main__":
    main()
