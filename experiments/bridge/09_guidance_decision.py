#!/usr/bin/env python3
"""G3: when a Bradley–Terry interval changes the bounded guidance direction."""
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
    spec = importlib.util.spec_from_file_location("online_bandit_g3", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M = load_online()
ROOT = Path(__file__).resolve().parents[2]
SIZES = (128, 512, 2048, 4096)


def checkpoint(phi0: float) -> dict:
    rows = [row for row in M.population_rollout(phi0, phi0, M.H, M.FEEDBACK_STEP) if row["x_next"] is not None]
    last = rows[-1]
    x, y = last["x_next"], last["y_next"]
    phi_now = M.scheduled_phi(M.FEEDBACK_STEP - 1, M.H, phi0)
    low = max(0.0, phi_now - M.RHO * M.H)
    high = min(1.0, phi_now + M.RHO * M.H)
    wq, ws = M.omega(x), M.omega(y)
    return {
        "x": x, "y": y, "phi": phi_now, "low": low, "high": high,
        "a_crit": ws / (wq + ws), "omega_q": wq, "omega_s": ws,
    }


def predict(bounds: tuple[float, float], state: dict) -> str:
    if bounds[1] < state["a_crit"] - 1e-12:
        return "down"
    if bounds[0] > state["a_crit"] + 1e-12:
        return "up"
    if state["a_crit"] > state["phi"] + 1e-12:
        return "up"
    if state["a_crit"] < state["phi"] - 1e-12:
        return "down"
    return "stay"


def direction(chosen: float, phi_now: float) -> str:
    if chosen > phi_now + 1e-15:
        return "up"
    if chosen < phi_now - 1e-15:
        return "down"
    return "stay"


def choose(mode: str, bounds: tuple[float, float], a_hat: float, low: float, high: float, x: float, y: float) -> float:
    if mode == "point":
        return float(np.clip(a_hat, low, high))
    active = M.U0 if mode == "source" else bounds
    return M.best_phi(x, y, active, low, high)


def population_path(state: dict, mode: str, bounds: tuple[float, float], a_hat: float) -> list[dict]:
    x, y, phi_prev = state["x"], state["y"], state["phi"]
    rows = []
    for k in range(M.FEEDBACK_STEP, M.N_STEPS):
        low = max(0.0, phi_prev - M.RHO * M.H)
        high = min(1.0, phi_prev + M.RHO * M.H)
        phi = choose(mode, bounds, a_hat, low, high, x, y)
        grad = M.exact_grad(x, y, phi)
        x_next, y_next = M.advance(x, y, phi, grad, M.H)
        cert_bounds = M.U0 if mode == "source" else bounds
        rows.append({
            "k": k, "t": k * M.H, "phi": phi,
            "direction": direction(phi, phi_prev),
            "source_certificate": M.interval_gain(x, y, x_next, y_next, M.U0, M.H),
            "updated_certificate": M.interval_gain(x, y, x_next, y_next, cert_bounds, M.H),
            "true_gain": M.finite_gain(x, y, x_next, y_next, M.reference_phi0(), M.H),
            "x": x, "y": y,
        })
        x, y, phi_prev = x_next, y_next, phi
    return rows


def sampled_path(state: dict, mode: str, bounds: tuple[float, float], a_hat: float, seed: int) -> list[dict]:
    rng = np.random.default_rng(30_000 + seed)
    x, y, phi_prev = state["x"], state["y"], state["phi"]
    rows = []
    for k in range(M.FEEDBACK_STEP, M.N_STEPS):
        u_q, u_s = rng.random(8192), rng.random(8192)
        low = max(0.0, phi_prev - M.RHO * M.H)
        high = min(1.0, phi_prev + M.RHO * M.H)
        phi = choose(mode, bounds, a_hat, low, high, x, y)
        grad = M.sample_grad(x, y, phi, u_q, u_s)
        x_next, y_next = M.advance(x, y, phi, grad, M.H)
        rows.append({
            "k": k, "phi": phi, "direction": direction(phi, phi_prev),
            "true_gain": M.finite_gain(x, y, x_next, y_next, M.reference_phi0(), M.H),
            "source_certificate": M.interval_gain(x, y, x_next, y_next, M.U0, M.H),
        })
        x, y, phi_prev = x_next, y_next, phi
    return rows


def mean(values: list[float]) -> float:
    return float(np.mean(values))


def main() -> None:
    started = time.perf_counter()
    phi0 = M.reference_phi0()
    M.self_check(phi0)
    state = checkpoint(phi0)
    out = ROOT / "outputs/horizon_bridge/step08_g3"
    out.mkdir(parents=True, exist_ok=True)

    source_path = population_path(state, "source", M.U0, phi0)
    if source_path[0]["direction"] != "up":
        raise AssertionError("source robust controller did not move upward")
    if abs(source_path[0]["phi"] - (state["phi"] + M.RHO * M.H)) > 1e-12:
        raise AssertionError("source robust controller did not spend the full step")

    per_seed = []
    summary_rows = [{
        "labels": 0, "evidence": "source", "seeds": 1,
        "below_crit": 0, "upper_min": 2.0, "upper_max": 2.0,
        "a_crit": state["a_crit"],
        "source_direction": "up", "learned_direction": "up", "point_direction": "up",
        "phi_end_gap": 0.0,
        "terminal_updated": source_path[-1]["source_certificate"],
        "true_gain_gap": 0.0,
        "prediction_matches": True,
    }]

    batches = {}
    for labels in (*SIZES, 8192):
        for seed in M.LABEL_SEEDS:
            batches[(labels, seed)] = M.label_batch(phi0, labels, seed, True)

    def collect(labels: int) -> list[dict]:
        found = []
        for seed in M.LABEL_SEEDS:
            batch = batches[(labels, seed)]
            if batch["empty"] or batch["interval"] is None:
                raise RuntimeError(f"empty interval at N={labels} seed {seed}")
            bounds = tuple(batch["interval"])
            predicted = predict(bounds, state)
            learned = population_path(state, "learned", bounds, batch["a_hat"])
            point = population_path(state, "point", bounds, batch["a_hat"])
            if learned[0]["direction"] != predicted:
                raise AssertionError((labels, seed, predicted, learned[0]["direction"], bounds, state["a_crit"]))
            learned_dirs = [row["direction"] for row in learned]
            found.append({
                "labels": labels, "seed": seed, "a_hat": batch["a_hat"],
                "lower": bounds[0], "upper": bounds[1],
                "below": bounds[1] < state["a_crit"] - 1e-12,
                "predicted": predicted,
                "learned_direction": learned[0]["direction"],
                "learned_up_steps": learned_dirs.count("up"),
                "learned_down_steps": learned_dirs.count("down"),
                "point_direction": point[0]["direction"],
                "phi_end_gap": learned[-1]["phi"] - source_path[-1]["phi"],
                "point_phi_end_gap": point[-1]["phi"] - source_path[-1]["phi"],
                "terminal_updated": learned[-1]["updated_certificate"],
                "terminal_source": learned[-1]["source_certificate"],
                "true_gain_gap": mean([row["true_gain"] for row in learned]) - mean([row["true_gain"] for row in source_path]),
                "min_updated": min(row["updated_certificate"] for row in learned),
            })
        return found

    by_n = {labels: collect(labels) for labels in SIZES}
    flipped = {labels: sum(row["below"] for row in rows) for labels, rows in by_n.items()}
    ran_8192 = max(flipped.values()) == 0
    if ran_8192:
        by_n[8192] = collect(8192)
        flipped[8192] = sum(row["below"] for row in by_n[8192])

    promising = None
    for labels in (*SIZES, 8192):
        if labels in flipped and flipped[labels] >= 4:
            promising = labels
            break
    if promising is None:
        for labels in (*SIZES, 8192):
            if labels in flipped and flipped[labels] > 0:
                promising = labels
                break

    sampled = []
    if promising is not None:
        for seed in (0, 1, 2):
            batch = batches[(promising, M.LABEL_SEEDS[0])]
            # One shared label batch, three paired policy seeds.
            bounds = tuple(batch["interval"])
            source_s = sampled_path(state, "source", M.U0, phi0, seed)
            learned_s = sampled_path(state, "learned", bounds, batch["a_hat"], seed)
            sampled.append({
                "policy_seed": seed, "label_seed": M.LABEL_SEEDS[0], "labels": promising,
                "source_direction": source_s[0]["direction"],
                "learned_direction": learned_s[0]["direction"],
                "phi_end_gap": learned_s[-1]["phi"] - source_s[-1]["phi"],
                "true_gain_gap": mean([row["true_gain"] for row in learned_s]) - mean([row["true_gain"] for row in source_s]),
            })

    control_n = promising or (8192 if ran_8192 else 4096)
    control = population_path(state, "source", M.U0, phi0)
    control_gap = control[-1]["phi"] - source_path[-1]["phi"]

    for labels, rows in by_n.items():
        per_seed.extend(rows)
        summary_rows.append({
            "labels": labels, "evidence": "informative", "seeds": len(rows),
            "below_crit": sum(row["below"] for row in rows),
            "upper_min": min(row["upper"] for row in rows),
            "upper_max": max(row["upper"] for row in rows),
            "a_crit": state["a_crit"],
            "source_direction": "up",
            "learned_direction": ",".join(sorted({row["learned_direction"] for row in rows})),
            "learned_paths": [
                {"seed": row["seed"], "up": row["learned_up_steps"], "down": row["learned_down_steps"]}
                for row in rows
            ],
            "point_direction": ",".join(sorted({row["point_direction"] for row in rows})),
            "phi_end_gap": mean([row["phi_end_gap"] for row in rows]),
            "terminal_updated": mean([row["terminal_updated"] for row in rows]),
            "true_gain_gap": mean([row["true_gain_gap"] for row in rows]),
            "prediction_matches": True,
            "sampled": labels == promising,
        })
    summary_rows.append({
        "labels": control_n, "evidence": "repeated_source", "seeds": 5,
        "below_crit": 0, "upper_min": 2.0, "upper_max": 2.0,
        "a_crit": state["a_crit"], "source_direction": "up", "learned_direction": "up",
        "point_direction": "up", "phi_end_gap": control_gap,
        "terminal_updated": control[-1]["source_certificate"],
        "true_gain_gap": 0.0, "prediction_matches": True,
        "note": "Source pairs have logit 2 for every a, so U stays [-1, 2].",
    })

    with (out / "decisions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_seed[0]))
        writer.writeheader()
        writer.writerows(per_seed)

    write_summary(out, {
        "checkpoint": state,
        "summary": summary_rows,
        "flipped_counts": flipped,
        "ran_8192": ran_8192,
        "promising_n": promising,
        "sampled": sampled,
        "source_terminal_certificate": source_path[-1]["source_certificate"],
        "source_mean_true_gain": mean([row["true_gain"] for row in source_path]),
    }, started)


if __name__ == "__main__":
    main()
