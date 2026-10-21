"""Checkpoint forks and the five scoped suggestions. None is a neural theorem."""
from __future__ import annotations


SUGGESTIONS = {
    "collect_more_comparisons",
    "optimize_existing_reward_more",
    "reward_change_budget_limited",
    "policy_optimization_limited",
    "no_supported_action",
}


def compare_interventions(parent_checkpoint: dict, branches: list[dict]) -> dict:
    parent_step = int(parent_checkpoint["checkpoint_step"])
    parent_policy = parent_checkpoint.get("policy_sha")
    rows = []
    for branch in branches:
        if int(branch.get("parent_step", parent_step)) != parent_step:
            raise ValueError("a fork does not start from the declared parent checkpoint")
        if parent_policy is not None and branch.get("parent_policy_sha") not in (None, parent_policy):
            raise ValueError("fork policy state does not match the parent")
        rows.append({
            "name": branch["name"],
            "new_human_labels": int(branch.get("new_human_labels", 0)),
            "new_ai_labels": int(branch.get("new_ai_labels", 0)),
            "reward_optimizer_steps": int(branch.get("reward_optimizer_steps", 0)),
            "policy_updates": int(branch.get("policy_updates", 0)),
            "utility": branch.get("utility"),
            "anchor_rms": branch.get("anchor_rms"),
            "guarantee_type": branch.get("guarantee_type", "empirical_proxy"),
            "warning": "no formal neural support horizon",
        })
    return {"parent_step": parent_step, "branches": rows}


def recommend(measurement: dict) -> dict:
    """Return one suggestion plus the reason. Neural outputs carry a no-guarantee warning."""
    if measurement.get("privileged_reward_cannot_move"):
        choice = "policy_optimization_limited"
        reason = "a privileged reference reward does not move the policy under this budget"
    elif measurement.get("fit_not_converged"):
        choice = "optimize_existing_reward_more"
        reason = "the fit on the same observed labels has not converged"
    elif measurement.get("desired_movement", 0) > measurement.get("movement_budget", 0) and measurement.get("movement_budget") is not None:
        choice = "reward_change_budget_limited"
        reason = "the reward change that separates actions exceeds the anchor movement bound"
    elif measurement.get("predictions_unresolved") and measurement.get("informative_pairs_available"):
        choice = "collect_more_comparisons"
        reason = "predictions on the predeclared query pool are empirically unresolved and informative pairs exist"
    else:
        choice = "no_supported_action"
        reason = "noise, mismatch, or a missing confidence model blocks a stronger recommendation"
    if choice not in SUGGESTIONS:
        raise AssertionError(choice)
    return {
        "suggestion": choice,
        "reason": reason,
        "guarantee_type": measurement.get("guarantee_type", "empirical_proxy"),
        "warning": "no-guarantee" if measurement.get("guarantee_type", "empirical_proxy") == "empirical_proxy" else None,
    }
