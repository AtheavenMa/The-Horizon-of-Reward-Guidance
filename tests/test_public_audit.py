"""Public-facing finite audit and seed comparison smoke tests."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from experiments.guidance_audit.__main__ import main
from experiments.guidance_audit.compare_stats import ComparisonError, paired_comparison
from experiments.guidance_audit.finite import anchor_movement, audit_fixed_finite, computed_certificate
from experiments.guidance_audit.schema import load_manifest


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples" / "08_guidance_audit"


def test_recomputed_finite_changes_when_probability_changes():
    record = load_manifest(EXAMPLES / "finite_computed.json")
    item = record["checkpoints"][0]
    first = computed_certificate(item, record["compatible_reward_set"])
    assert first["status"] == "COMPUTED"
    assert first["source_certificate"] == pytest.approx(7.5)

    changed = copy.deepcopy(item)
    changed["outcome_probabilities"][0] += 1 / 16
    changed["outcome_probabilities"][15] -= 1 / 16
    second = computed_certificate(changed, record["compatible_reward_set"])
    assert second["source_certificate"] != pytest.approx(first["source_certificate"])


def test_untrue_stored_certificate_is_flagged():
    record = load_manifest(EXAMPLES / "intentionally_false_certificate.json")
    result = audit_fixed_finite(
        record["checkpoints"], record["compatible_reward_set"],
        record.get("step_rule", "recorded_exact"), 0.0
    )
    row = result["rows"][0]
    assert row["verification_status"] == "REJECTED_STORED_CERTIFICATE"
    assert row["computed_source_certificate"] == pytest.approx(0.0)


def test_actual_anchor_budget_is_checked():
    assert anchor_movement([0.0, 0.0], [0.02, -0.02], 0.05, 0.0)["passed"]
    assert not anchor_movement([0.0, 0.0], [0.02, -0.02], 0.0, 0.0)["passed"]
    assert anchor_movement(None, [0.0], 0.05, 0.0)["status"] == "BUDGET_UNVERIFIED"


def test_seed_paired_statistics_and_evaluator_parity():
    baseline = [{"seed": 0, "fixed_eval": 0.5, "evaluator": "heldout"},
                {"seed": 1, "fixed_eval": 0.6, "evaluator": "heldout"}]
    treatment = [{"seed": 0, "fixed_eval": 0.51, "evaluator": "heldout"},
                 {"seed": 1, "fixed_eval": 0.63, "evaluator": "heldout"}]
    result = paired_comparison(baseline, treatment, "fixed_eval")
    assert result["n_seeds"] == 2
    assert result["paired_mean"] == pytest.approx(0.02)
    assert result["interval_95"] is not None

    treatment[1]["evaluator"] = "another_head"
    with pytest.raises(ComparisonError, match="evaluators differ"):
        paired_comparison(baseline, treatment, "fixed_eval")


def test_public_cli_runs_without_model_dependencies(tmp_path, capsys):
    manifest = EXAMPLES / "finite_computed.json"
    assert main(["validate", "--manifest", str(manifest)]) == 0
    assert main(["audit", "--manifest", str(manifest), "--out", str(tmp_path)]) == 0
    report = json.loads((tmp_path / "audit.json").read_text())
    assert report["rows"][0]["computed_source_certificate"] == pytest.approx(7.5)
    assert report["rows"][0]["verification_status"] == "COMPUTED"


def test_reader_first_homepages_and_five_minute_tutorial():
    for name in ("README.md", "README_cn.md"):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert "arXiv coming soon" in text
        assert "docs/index.html" in text
        assert "assets/tour_preview.png" in text
        assert "docs/RESULTS.md" in text
        assert "Work Plan 08" not in text
    html = (ROOT / "demo/horizon_playground.html").read_text(encoding="utf-8")
    assert "<svg" in html and 'id="reset"' in html and 'id="feedback"' in html
    assert "https://cdn." not in html
