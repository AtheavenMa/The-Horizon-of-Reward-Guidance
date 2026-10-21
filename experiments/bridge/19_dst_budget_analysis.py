#!/usr/bin/env python3
"""P4: one-factor label, movement-budget and timing sweeps on the P3 learner."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.bridge.common import write_summary
from experiments.bridge.pref_common import DST, ROOT, frozen_heldout, jsonable, run_online


def t_interval(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if n > 1 else 0.0
    half = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)) if n > 1 else 0.0
    return {"mean": mean, "sd": sd, "n": n, "ci95": [mean - half, mean + half], "interval": "paired Student t, df=n-1"}


def load_p3(method: str, seed: int) -> dict:
    path = ROOT / "outputs/horizon_bridge/p3_dst_live_weak" / f"{method}_seed{seed}.json"
    return json.loads(path.read_text())


def main() -> None:
    started = time.perf_counter()
    out = ROOT / "outputs/horizon_bridge/p4_budget_timing"
    out.mkdir(parents=True, exist_ok=True)
    dst = DST()
    import mo_gymnasium as mo_gym
    gym_env = mo_gym.make("deep-sea-treasure-v0")
    held = frozen_heldout(dst, gym_env)
    pilot = json.loads((ROOT / "outputs/horizon_bridge/p3_dst_live_weak/pilot.json").read_text())
    episodes = int(pilot["gradient"]["chosen_episodes"])
    specs = {
        "labels_32": dict(budget=0.02, feedback_at=(20,), labels_per_event=32, label_cap=32),
        "labels_64": dict(budget=0.02, feedback_at=(20, 40), labels_per_event=32, label_cap=64),
        "b0": dict(budget=0.0, feedback_at=(20, 40, 60, 80), labels_per_event=32, label_cap=128),
        "b01": dict(budget=0.01, feedback_at=(20, 40, 60, 80), labels_per_event=32, label_cap=128),
        "b05": dict(budget=0.05, feedback_at=(20, 40, 60, 80), labels_per_event=32, label_cap=128),
        "late": dict(budget=0.02, feedback_at=(60, 80, 100, 120), labels_per_event=32, label_cap=128),
        "oneshot": dict(budget=0.02, feedback_at=(20,), labels_per_event=128, label_cap=128),
    }
    fresh = {}
    for name, spec in specs.items():
        rows = []
        for seed in range(5):
            path = out / f"{name}_seed{seed}.json"
            if path.exists():
                rows.append(json.loads(path.read_text()))
                continue
            row = run_online(
                dst, gym_env, seed=seed, method="I", regime="weak", updates=160,
                episodes=episodes, held=held, **spec,
            )
            slim = {k: v for k, v in row.items() if k not in ("logits", "snapshots")}
            path.write_text(json.dumps(jsonable(slim)))
            rows.append(slim)
            print(name, seed, slim["final_return"], flush=True)
        fresh[name] = rows
    gym_env.close()
    reference = [load_p3("I", seed) for seed in range(5)]
    source_only = [load_p3("S", seed) for seed in range(5)]
    uniform = [load_p3("U", seed) for seed in range(8)]
    targeted = [load_p3("I", seed) for seed in range(8)]
    mixed_s = [json.loads(p.read_text()) for p in sorted((ROOT / "outputs/horizon_bridge/p3_dst_live_mixed").glob("S_seed*.json"))]
    mixed_i = [json.loads(p.read_text()) for p in sorted((ROOT / "outputs/horizon_bridge/p3_dst_live_mixed").glob("I_seed*.json"))]

    def finals(rows):
        return np.array([row["final_return"] for row in rows])

    def certs(rows, key):
        vals = []
        for row in rows:
            series = row[key]
            vals.append(series[-1] if series[-1] is not None else np.nan)
        return np.array(vals, dtype=float)

    comparisons = {
        "labels_0": t_interval(finals(source_only) - finals(reference)),
        "labels_32": t_interval(finals(fresh["labels_32"]) - finals(reference)),
        "labels_64": t_interval(finals(fresh["labels_64"]) - finals(reference)),
        "labels_128": t_interval(finals(reference) - finals(reference)),
        "b0": t_interval(finals(fresh["b0"]) - finals(reference)),
        "b01": t_interval(finals(fresh["b01"]) - finals(reference)),
        "b02": t_interval(finals(reference) - finals(reference)),
        "b05": t_interval(finals(fresh["b05"]) - finals(reference)),
        "early": t_interval(finals(reference) - finals(reference)),
        "late": t_interval(finals(fresh["late"]) - finals(reference)),
        "oneshot": t_interval(finals(fresh["oneshot"]) - finals(reference)),
        "uniform_minus_targeted_8": t_interval(finals(uniform) - finals(targeted)),
    }
    levels = {
        "labels": {
            "0": float(finals(source_only).mean()),
            "32": float(finals(fresh["labels_32"]).mean()),
            "64": float(finals(fresh["labels_64"]).mean()),
            "128": float(finals(reference).mean()),
        },
        "budget": {
            "0": float(finals(fresh["b0"]).mean()),
            "0.01": float(finals(fresh["b01"]).mean()),
            "0.02": float(finals(reference).mean()),
            "0.05": float(finals(fresh["b05"]).mean()),
        },
        "timing": {
            "early": float(finals(reference).mean()),
            "late": float(finals(fresh["late"]).mean()),
            "oneshot": float(finals(fresh["oneshot"]).mean()),
        },
    }
    payload = {
        "episodes_per_update": episodes,
        "reference": "P3 weak targeted robust, seeds 0-4, b=0.02, 128 labels at 20/40/60/80",
        "levels": levels,
        "difference_from_reference": comparisons,
        "source_certificate_final_mean": {
            "S": float(np.nanmean(certs(source_only, "source_certificates"))),
            "I": float(np.nanmean(certs(reference, "source_certificates"))),
            "b0": float(np.nanmean(certs(fresh["b0"], "source_certificates"))),
        },
        "updated_certificate_final_mean": {
            "I": float(np.nanmean(certs(reference, "updated_certificates"))),
            "b0": float(np.nanmean(certs(fresh["b0"], "updated_certificates"))),
            "labels_32": float(np.nanmean(certs(fresh["labels_32"], "updated_certificates"))),
        },
        "final_phi": {
            "b0": float(np.mean([row["final_phi"] for row in fresh["b0"]])),
            "b01": float(np.mean([row["final_phi"] for row in fresh["b01"]])),
            "b02": float(np.mean([row["final_phi"] for row in reference])),
            "b05": float(np.mean([row["final_phi"] for row in fresh["b05"]])),
            "late": float(np.mean([row["final_phi"] for row in fresh["late"]])),
            "oneshot": float(np.mean([row["final_phi"] for row in fresh["oneshot"]])),
        },
        "mixed_available": len(mixed_s) > 0 and len(mixed_i) > 0,
        "mixed_I_minus_S": None if not mixed_s else t_interval(finals(mixed_i) - finals(mixed_s)),
        "weak_I_minus_S_8": t_interval(finals(targeted) - np.array([load_p3("S", s)["final_return"] for s in range(8)])),
    }
    write_summary(out, jsonable(payload), started)


if __name__ == "__main__":
    main()
