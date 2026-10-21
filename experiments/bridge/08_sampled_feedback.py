#!/usr/bin/env python3
"""G2: Bradley–Terry labels on frozen sampled states. Actions do not change."""
from __future__ import annotations

import csv
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.bridge.common import write_summary


def load_online():
    path = Path(__file__).resolve().parent / "06_online_bandit.py"
    spec = importlib.util.spec_from_file_location("online_bandit_g2", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M = load_online()
ROOT = Path(__file__).resolve().parents[2]
N_PREFIX = 8192
LABELS = 128


def replay_stored(phi0: float, seed: int) -> list[dict]:
    draws = M.uniforms(seed, M.N_STEPS, M.PRECISION_N)
    rows = M.sampled_rollout(phi0, phi0, M.H, draws, N_PREFIX)
    return [row for row in rows if row["x_next"] is not None]


def suffix_draws(seed: int, n_steps: int, n: int):
    rng = np.random.default_rng(20_000 + seed)
    return [(rng.random(n), rng.random(n)) for _ in range(n_steps)]


def hybrid_path(phi0: float, seed: int) -> list[dict]:
    exact = [row for row in M.population_rollout(phi0, phi0, M.H, M.FEEDBACK_STEP) if row["x_next"] is not None]
    x, y = exact[-1]["x_next"], exact[-1]["y_next"]
    rows = list(exact)
    for i, (u_q, u_s) in enumerate(suffix_draws(seed, M.N_STEPS - M.FEEDBACK_STEP, N_PREFIX)):
        k = M.FEEDBACK_STEP + i
        phi = M.scheduled_phi(k, M.H, phi0)
        exact_grad = M.exact_grad(x, y, phi)
        grad = M.sample_grad(x, y, phi, u_q, u_s)
        x_next, y_next = M.advance(x, y, phi, grad, M.H)
        rows.append(M.measure(k, M.H, x, y, x_next, y_next, phi, phi0, grad, exact_grad, M.U0, M.U0))
        x, y = x_next, y_next
    return rows


def gains(row: dict, bounds: tuple[float, float]) -> tuple[float, float]:
    grad = M.exact_grad(row["x"], row["y"], row["phi"])
    x_pop, y_pop = M.advance(row["x"], row["y"], row["phi"], grad, M.H)
    population = M.interval_gain(row["x"], row["y"], x_pop, y_pop, bounds, M.H)
    realized = M.interval_gain(row["x"], row["y"], row["x_next"], row["y_next"], bounds, M.H)
    return population, realized


def annotate(rows: list[dict], updated: tuple[float, float] | None, informative: bool) -> list[dict]:
    out = []
    for row in rows:
        item = dict(row)
        if row["k"] < M.FEEDBACK_STEP or not informative:
            population, realized = gains(row, M.U0)
            item["active_set"] = list(M.U0)
        elif updated is None:
            population, realized = -np.inf, -np.inf
            item["active_set"] = None
        else:
            population, realized = gains(row, updated)
            item["active_set"] = list(updated)
        item["population_step"] = population
        item["realized_step"] = realized
        out.append(item)
    return out


def first_loss(rows: list[dict], key: str, after: bool) -> float | None:
    for row in rows:
        in_window = row["t"] >= M.FEEDBACK_STEP * M.H - 1e-12 if after else row["t"] < M.FEEDBACK_STEP * M.H - 1e-12
        if in_window and row[key] is not None and row[key] < M.GAMMA - 1e-15:
            return float(row["t"])
    return None


def certified(rows: list[dict], key: str, after_only: bool) -> bool:
    chosen = [row for row in rows if (row["t"] >= M.FEEDBACK_STEP * M.H - 1e-12) or not after_only]
    return all(row[key] is not None and row[key] >= M.GAMMA - 1e-15 for row in chosen)


def true_changed(rows: list[dict], reference: list[dict]) -> bool:
    for row, base in zip(rows, reference):
        if abs(row["true_finite_step_gain"] - base["true_finite_step_gain"]) > 1e-12:
            return True
    return False


def fmt_pair(values: list[float | None]) -> str:
    shown = []
    for value in values:
        text = "censored" if value is None else f"{value:.4g}"
        if text not in shown:
            shown.append(text)
    return " / ".join(shown) if False else ", ".join(shown)


def unique_text(values: list[float | None]) -> str:
    shown = []
    for value in values:
        text = "censored" if value is None else f"{value:.4g}"
        if text not in shown:
            shown.append(text)
    return ", ".join(shown)


def summarize_condition(name: str, evidence: str, labels: int, records: list[dict]) -> dict:
    return {
        "trajectory": name,
        "evidence": evidence,
        "labels": labels,
        "pairings": len(records),
        "population_before": unique_text([row["pop_before"] for row in records]),
        "population_after": unique_text([row["pop_after"] for row in records]),
        "realized_before": unique_text([row["real_before"] for row in records]),
        "realized_after": unique_text([row["real_after"] for row in records]),
        "suffix_fraction": float(np.mean([row["suffix"] for row in records])),
        "entire_fraction": float(np.mean([row["entire"] for row in records])),
        "population_suffix_fraction": float(np.mean([row["pop_suffix"] for row in records])),
        "true_gain_changed": any(row["changed"] for row in records),
        "terminal_updated": [row["terminal_updated"] for row in records],
        "intervals": [row["interval"] for row in records],
    }


def evaluate(rows: list[dict], reference: list[dict], updated: tuple[float, float] | None, interval, informative: bool) -> dict:
    annotated = annotate(rows, updated, informative)
    return {
        "pop_before": first_loss(annotated, "population_step", False),
        "pop_after": first_loss(annotated, "population_step", True),
        "real_before": first_loss(annotated, "realized_step", False),
        "real_after": first_loss(annotated, "realized_step", True),
        "suffix": certified(annotated, "realized_step", True),
        "entire": certified(annotated, "realized_step", False),
        "pop_suffix": certified(annotated, "population_step", True),
        "changed": true_changed(annotated, reference),
        "terminal_updated": annotated[-1]["realized_step"],
        "terminal_population": annotated[-1]["population_step"],
        "interval": interval,
        "rows": annotated,
    }


def main() -> None:
    started = time.perf_counter()
    phi0 = M.reference_phi0()
    M.self_check(phi0)
    out = ROOT / "outputs/horizon_bridge/step07_g2"
    out.mkdir(parents=True, exist_ok=True)

    stored = {seed: replay_stored(phi0, seed) for seed in M.POLICY_SEEDS}
    hybrids = {seed: hybrid_path(phi0, seed) for seed in M.POLICY_SEEDS}
    informative = {
        (labels, seed): M.label_batch(phi0, labels, seed, True)
        for labels in (LABELS, 512)
        for seed in M.LABEL_SEEDS
    }
    source_labels = {seed: M.label_batch(phi0, LABELS, seed, False) for seed in M.LABEL_SEEDS}

    def interval_of(batch: dict) -> tuple[float, float] | None:
        if batch["empty"] or batch["interval"] is None:
            return None
        return tuple(batch["interval"])

    conditions = []
    series = []

    def add(name, evidence, labels, path_map, batches):
        records = []
        for policy_seed, rows in path_map.items():
            chosen = batches if isinstance(batches, dict) else {0: batches}
            for label_seed, batch in chosen.items():
                updated = None if batch is None else interval_of(batch)
                # Source evidence keeps U0. An empty learned interval does not certify.
                if evidence != "informative":
                    updated = None
                result = evaluate(
                    rows, rows, updated, None if batch is None else batch.get("interval"), evidence == "informative",
                )
                result["policy_seed"] = policy_seed
                result["label_seed"] = label_seed
                records.append(result)
                for row in result["rows"]:
                    series.append({
                        "trajectory": name, "evidence": evidence, "labels": labels,
                        "policy_seed": policy_seed, "label_seed": label_seed,
                        "k": row["k"], "t": row["t"],
                        "population_step": row["population_step"], "realized_step": row["realized_step"],
                        "true_finite_step_gain": row["true_finite_step_gain"],
                    })
        conditions.append(summarize_condition(name, evidence, labels, records))
        return records

    sampled_source = add("sampled_from_start", "original_source", 0, stored, {0: None})
    add("sampled_from_start", "repeated_source", LABELS, stored, source_labels)
    informative_128 = add(
        "sampled_from_start", "informative", LABELS, stored,
        {seed: informative[(LABELS, seed)] for seed in M.LABEL_SEEDS},
    )
    add("exact_prefix_sampled_suffix", "original_source", 0, hybrids, {0: None})
    add(
        "exact_prefix_sampled_suffix", "informative", LABELS, hybrids,
        {seed: informative[(LABELS, seed)] for seed in M.LABEL_SEEDS},
    )

    unresolved = not (
        all(row["pop_suffix"] for row in informative_128) or not any(row["pop_suffix"] for row in informative_128)
    )
    # A split across pairings is unresolved. A uniform answer is resolved.
    if any(row["interval"] is None for row in informative_128):
        unresolved = True
    if unresolved:
        add(
            "sampled_from_start", "informative", 512, stored,
            {seed: informative[(512, seed)] for seed in M.LABEL_SEEDS},
        )
    else:
        conditions.append({
            "trajectory": "optional_precision",
            "evidence": "informative",
            "labels": 512,
            "pairings": 0,
            "population_before": "NOT RUN",
            "population_after": "NOT RUN",
            "realized_before": "NOT RUN",
            "realized_after": "NOT RUN",
            "suffix_fraction": None,
            "entire_fraction": None,
            "population_suffix_fraction": None,
            "true_gain_changed": False,
            "terminal_updated": [],
            "intervals": [],
            "reason": "N=128 already gives the same post-feedback population answer on every pairing",
        })

    # The first three policy and label seeds are the predeclared pilot cross.
    pilot = [row for row in informative_128 if row["policy_seed"] <= 2 and row["label_seed"] <= 102]
    with (out / "series.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(series[0]))
        writer.writeheader()
        writer.writerows(series)

    write_summary(out, {
        "conditions": conditions,
        "pilot_3x3_population_suffix": float(np.mean([row["pop_suffix"] for row in pilot])),
        "pilot_3x3_realized_suffix": float(np.mean([row["suffix"] for row in pilot])),
        "pilot_3x3_entire": float(np.mean([row["entire"] for row in pilot])),
        "n512_run": unresolved,
    }, started)


if __name__ == "__main__":
    main()
