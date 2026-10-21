"""Exact finite audits. These do not apply to a neural reward model."""
from __future__ import annotations

import math

from horizon import analyze_horizon, certified_gain, support_horizon


def _dot(probabilities, rewards) -> float:
    if len(probabilities) != len(rewards):
        raise ValueError("outcome probabilities and rewards must have the same length")
    total = 0.0
    mass = 0.0
    for probability, reward in zip(probabilities, rewards):
        total += float(probability) * float(reward)
        mass += float(probability)
    if abs(mass - 1.0) > 1e-8:
        raise ValueError(f"outcome probabilities must sum to 1, got {mass}")
    return total


def computed_certificate(item: dict, compatible_reward_set: dict) -> dict:
    """Recompute a finite gain from supplied probabilities. Never trust a stored certificate alone."""
    kind = compatible_reward_set.get("kind")
    probabilities = item.get("outcome_probabilities")
    if probabilities is None or kind is None:
        return {"status": "REPORTED_UNVERIFIED", "source_certificate": None}
    if kind == "affine_interval":
        base = item.get("reward_base")
        slope = item.get("reward_slope")
        lo, hi = compatible_reward_set["interval"]
        if base is None or slope is None:
            return {"status": "NOT_VERIFIED", "source_certificate": None}
        left = _dot(probabilities, [b + float(lo) * s for b, s in zip(base, slope)])
        right = _dot(probabilities, [b + float(hi) * s for b, s in zip(base, slope)])
        return {"status": "COMPUTED", "source_certificate": min(left, right), "endpoint_gains": [left, right]}
    if kind == "simplex":
        vertices = item.get("reward_at_vertices")
        if not vertices or len(vertices) != 3:
            return {"status": "NOT_VERIFIED", "source_certificate": None}
        gains = [_dot(probabilities, vertex) for vertex in vertices]
        return {"status": "COMPUTED", "source_certificate": min(gains), "vertex_gains": gains}
    return {"status": "NOT_VERIFIED", "source_certificate": None}


def anchor_movement(before, after, budget: float, epsilon: float) -> dict:
    """RMS of pairwise-margin change. Missing anchors are unverified, not a pass."""
    if before is None or after is None or len(before) == 0 or len(before) != len(after):
        return {"status": "BUDGET_UNVERIFIED", "passed": False, "rms": None}
    squared = [(float(a) - float(b)) ** 2 for a, b in zip(after, before)]
    rms = math.sqrt(sum(squared) / len(squared))
    return {
        "status": "COMPUTED",
        "rms": rms,
        "budget": float(budget),
        "epsilon": float(epsilon),
        "passed": rms <= float(budget) + float(epsilon),
    }


def scalar_horizon_report(gain, required_gain, uncertainty_radius, drift_speed, adaptation_speed, target_horizon: float) -> dict:
    """Pointwise feasibility and trajectory feasibility for the scalar model only."""
    report = analyze_horizon(
        gain=gain,
        required_gain=required_gain,
        uncertainty_radius=uncertainty_radius,
        drift_speed=drift_speed,
        adaptation_speed=adaptation_speed,
        target_horizon=target_horizon,
    )
    return {
        "support_horizon": report.support_horizon,
        "pointwise_feasible": bool(report.pointwise_feasible),
        "trajectory_feasible": bool(report.trajectory_feasible),
        "assumption": "scalar linear moving target under analyze_horizon; not an LLM checkpoint",
    }


def audit_fixed_finite(checkpoints: list[dict], compatible_reward_set: dict, step_rule: str, movement_budget: float) -> dict:
    """Read stored exact states. A label-only change cannot alter true return."""
    if step_rule not in {"population_exact", "recorded_exact"}:
        raise ValueError("step_rule must name an exact population or recorded exact update")
    gamma = float(compatible_reward_set["required_gain"])
    rows = []
    first_loss = None
    for item in checkpoints:
        true_return = item.get("true_return")
        source = item.get("source_certificate")
        updated = item.get("updated_certificate")
        if true_return is None:
            true_return = "not_available"
        computed = computed_certificate(item, compatible_reward_set)
        if computed["status"] == "COMPUTED" and isinstance(source, (int, float)):
            if abs(float(source) - computed["source_certificate"]) > 1e-6:
                computed["status"] = "REJECTED_STORED_CERTIFICATE"
                computed["stored_source_certificate"] = float(source)
        row = {
            "checkpoint_step": int(item["checkpoint_step"]),
            "true_return": true_return,
            "phi": item.get("phi"),
            "source_certificate": computed["source_certificate"] if computed.get("source_certificate") is not None else source,
            "updated_certificate": updated,
            "population_gain": item.get("population_gain"),
            "sampled_gain": item.get("sampled_gain"),
            "movement_budget": float(movement_budget),
            "guarantee_type": "exact_finite" if computed["status"] == "COMPUTED" else "REPORTED_UNVERIFIED",
            "verification_status": computed["status"],
            "computed_source_certificate": computed.get("source_certificate"),
        }
        if computed.get("source_certificate") is not None and updated is None:
            certificate = computed["source_certificate"]
        else:
            certificate = source if updated is None else updated
        if isinstance(certificate, (int, float)) and certificate < gamma and first_loss is None:
            first_loss = int(item["checkpoint_step"])
        rows.append(row)
    # Information-only pairs share a policy id and must share the true return.
    by_policy = {}
    for item in checkpoints:
        key = item.get("policy_id")
        if key is None or item.get("true_return") is None:
            continue
        by_policy.setdefault(key, set()).add(float(item["true_return"]))
    leaks = {key: sorted(values) for key, values in by_policy.items() if len(values) > 1}
    if leaks:
        raise ValueError(f"label update changed true return on a frozen policy: {leaks}")
    return {
        "guarantee_type": "exact_finite",
        "rows": rows,
        "first_unsupported_checkpoint": "censored" if first_loss is None else first_loss,
        "censored": first_loss is None,
        "movement_budget": float(movement_budget),
        "pointwise_available": compatible_reward_set.get("pointwise_available"),
        "path_feasible": compatible_reward_set.get("path_feasible"),
        "flag_status": "REPORTED_UNVERIFIED" if "kind" not in compatible_reward_set else "computed_rows",
    }


def scalar_reference() -> dict:
    value = support_horizon(1.0, 0.6, 1.0, 0.12, 0.04)
    return {"name": "scalar_exact_horizon", "expected": 5.0, "observed": value, "passed": abs(value - 5.0) < 1e-12}


def frozen_policy_label_check() -> dict:
    """Same policy, wider then narrower set. Return stays; the certificate can change."""
    shared = {"policy_id": "frozen-0", "checkpoint_step": 0, "true_return": 0.5, "phi": 0.0, "population_gain": 0.1}
    wide = {**shared, "source_certificate": certified_gain(1.0, 1.0, 4.0, 0.16, 0.12)}
    narrow = {**shared, "source_certificate": certified_gain(1.0, 0.2, 4.0, 0.16, 0.12), "updated_certificate": certified_gain(1.0, 0.2, 4.0, 0.16, 0.12)}
    audit = audit_fixed_finite(
        [wide, narrow],
        {"required_gain": 0.6, "pointwise_available": True, "path_feasible": False},
        "recorded_exact",
        0.04,
    )
    changed_certificate = narrow["updated_certificate"] != wide["source_certificate"]
    same_return = audit["rows"][0]["true_return"] == audit["rows"][1]["true_return"]
    return {"passed": bool(changed_certificate and same_return), "source": wide["source_certificate"], "updated": narrow["updated_certificate"]}
