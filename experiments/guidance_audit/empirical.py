"""Empirical audits. They never emit an exact support-horizon certificate."""
from __future__ import annotations

import math


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def audit_empirical(checkpoints: list[dict], preference_batches: list[dict], heldout_pairs: list[dict], policy_evaluations: list[dict]) -> dict:
    correct = [1.0 if int(pair["correct"]) else 0.0 for pair in heldout_pairs]
    nlls = [float(pair["nll"]) for pair in heldout_pairs if "nll" in pair]
    briers = [float(pair["brier"]) for pair in heldout_pairs if "brier" in pair]
    utilities = [float(item["utility"]) for item in policy_evaluations if item.get("utility") is not None]
    anchor_movement = None
    if len(checkpoints) >= 2 and all("anchor_margins" in item for item in checkpoints[:2]):
        old, new = checkpoints[0]["anchor_margins"], checkpoints[1]["anchor_margins"]
        if len(old) == len(new) and old:
            anchor_movement = math.sqrt(sum((a - b) ** 2 for a, b in zip(old, new)) / len(old))
    adaptive = any(batch.get("adaptive") for batch in preference_batches)
    return {
        "guarantee_type": "empirical_proxy",
        "formal_horizon": "not_certified",
        "original_certified_gain": "not_available",
        "updated_certified_gain": "not_available",
        "heldout_accuracy": _mean(correct),
        "heldout_nll": _mean(nlls),
        "heldout_brier": _mean(briers),
        "heldout_n": len(heldout_pairs),
        "mean_utility": _mean(utilities),
        "anchor_rms_movement": anchor_movement,
        "new_label_count": int(sum(int(batch.get("new_label_count", 0)) for batch in preference_batches)),
        "reward_optimizer_steps": int(sum(int(batch.get("reward_optimizer_steps", 0)) for batch in preference_batches)),
        "adaptive_interval_adjustment": "missing_time_uniform_adjustment" if adaptive else "not_applicable",
        "status": "complete" if heldout_pairs else "incomplete",
    }


def select_pairs(features: list[float], count: int) -> list[int]:
    """Teacher-blind selection. The caller must not pass a hidden reward."""
    order = sorted(range(len(features)), key=lambda index: abs(features[index]), reverse=True)
    return order[:count]
