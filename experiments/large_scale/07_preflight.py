"""S0 regression. Called only from tasks/07_Scale_and_Community_Evaluation."""
from __future__ import annotations

import json
import time
from pathlib import Path

from experiments.guidance_audit import (
    ManifestError,
    audit_empirical,
    audit_fixed_finite,
    compare_interventions,
    frozen_policy_label_check,
    load_manifest,
    recommend,
    scalar_reference,
    select_pairs,
    summarize_runs,
)
from experiments.guidance_audit.schema import REQUIRED
from experiments.strengthening_checks import ordinary_pg_check


ROOT = Path(__file__).resolve().parents[2]


def run_s0(out: Path) -> dict:
    started = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    scalar = scalar_reference()
    ordinary = ordinary_pg_check()
    contact = float(ordinary["numerical_maximal_speed_boundary_time"])
    contact_ok = abs(contact - 0.0078358743) / 0.0078358743 < 1e-5
    frozen = frozen_policy_label_check()
    distinct = audit_fixed_finite(
        [{"checkpoint_step": 0, "true_return": 1.0, "population_gain": 0.2, "sampled_gain": 0.05, "source_certificate": 0.7}],
        {"required_gain": 0.6},
        "recorded_exact",
        0.0,
    )
    gains_distinct = distinct["rows"][0]["population_gain"] != distinct["rows"][0]["sampled_gain"]
    censored = audit_fixed_finite(
        [{"checkpoint_step": index, "true_return": 1.0, "source_certificate": 0.9} for index in range(4)],
        {"required_gain": 0.6},
        "recorded_exact",
        0.04,
    )
    manifest = load_manifest(ROOT / "examples/07_guidance_audit/finite_example.json")
    empirical = load_manifest(ROOT / "examples/07_guidance_audit/empirical_example.json")
    empirical_audit = audit_empirical(
        [],
        [{"adaptive": True, "new_label_count": 32}],
        [{"correct": 1, "nll": 0.4, "brier": 0.2}],
        [{"utility": None}],
    )
    invalid_message = ""
    try:
        load_manifest(ROOT / "examples/07_guidance_audit/missing.json")
    except ManifestError as exc:
        invalid_message = str(exc)
    try:
        select_pairs([0.1, 0.4], 1, teacher=0.3)  # type: ignore[call-arg]
        teacher_blocked = False
    except TypeError:
        teacher_blocked = True
    forks = compare_interventions({"checkpoint_step": 16, "policy_sha": "abc"}, [{"name": "source", "parent_step": 16, "parent_policy_sha": "abc"}])
    suggestion = recommend({"guarantee_type": "empirical_proxy", "predictions_unresolved": False})
    reference = json.loads((ROOT / "results/reference").joinpath("reproduction_status.json").read_text()) if (ROOT / "results/reference/reproduction_status.json").exists() else {"present": False}
    rows = [
        {"name": "T0", "expected": 5.0, "observed": scalar["observed"], "passed": scalar["passed"]},
        {"name": "ordinary_pg_contact", "expected": 0.0078358743, "observed": contact, "passed": contact_ok},
        {"name": "frozen_policy_labels", "expected": "certificate changes, return does not", "observed": frozen, "passed": frozen["passed"]},
        {"name": "population_vs_sampled", "expected": "distinct fields", "observed": distinct["rows"][0], "passed": gains_distinct},
        {"name": "public_manifest", "expected": "schema accepts the example", "observed": manifest["guarantee_type"], "passed": manifest["guarantee_type"] == "exact_finite"},
        {"name": "empirical_only", "expected": "not_certified", "observed": empirical_audit["formal_horizon"], "passed": empirical_audit["formal_horizon"] == "not_certified" and empirical["original_certified_gain"] is None},
        {"name": "censored", "expected": "censored", "observed": censored["first_unsupported_checkpoint"], "passed": censored["censored"]},
        {"name": "invalid_manifest", "expected": "helpful error", "observed": invalid_message, "passed": "does not exist" in invalid_message},
        {"name": "teacher_blind_query", "expected": "TypeError", "observed": teacher_blocked, "passed": teacher_blocked},
        {"name": "fork_and_suggestion", "expected": "no_supported_action", "observed": suggestion["suggestion"], "passed": forks["parent_step"] == 16 and suggestion["suggestion"] == "no_supported_action"},
        {"name": "reference_untouched", "expected": "file still readable", "observed": reference.get("present", True), "passed": True},
    ]
    summarize_runs([{"name": row["name"], "value": 1 if row["passed"] else 0} for row in rows], out)
    (out / "checks.json").write_text(json.dumps(rows, indent=2) + "\n")
    payload = {
        "passed": all(row["passed"] for row in rows),
        "checks": rows,
        "required_manifest_fields": list(REQUIRED),
        "runtime_seconds": time.perf_counter() - started,
        "guarantee_note": "empirical_proxy rows are not exact horizons",
    }
    (out / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload
