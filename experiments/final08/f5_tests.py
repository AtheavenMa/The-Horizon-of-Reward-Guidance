"""F5 checks. Imported by the Work Plan 08 task; this file does not load Torch."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.guidance_audit.compare_stats import ComparisonError, paired_comparison
from experiments.guidance_audit.finite import anchor_movement, audit_fixed_finite, scalar_reference
from experiments.guidance_audit.schema import load_manifest


ROOT = Path(__file__).resolve().parents[2]


def _uniform(n: int) -> list[float]:
    return [1.0 / n] * n


class AuditTests(unittest.TestCase):
    def test_scalar_horizon(self):
        row = scalar_reference()
        self.assertTrue(row["passed"])
        self.assertAlmostEqual(row["observed"], 5.0, places=12)

    def test_perturbing_probabilities_changes_the_gain(self):
        base = _uniform(16)
        rewards = [float(i) for i in range(16)]
        item = {"checkpoint_step": 0, "outcome_probabilities": base, "reward_base": rewards, "reward_slope": [0.0] * 16}
        other = dict(item)
        tilted = base[:]
        tilted[0] += 0.1
        tilted[-1] -= 0.1
        other["outcome_probabilities"] = tilted
        setting = {"required_gain": 0.0, "kind": "affine_interval", "interval": [0.0, 0.0]}
        first = audit_fixed_finite([item], setting, "recorded_exact", 0.0)
        second = audit_fixed_finite([other], setting, "recorded_exact", 0.0)
        self.assertEqual(first["rows"][0]["verification_status"], "COMPUTED")
        self.assertNotEqual(first["rows"][0]["source_certificate"], second["rows"][0]["source_certificate"])

    def test_false_certificate_is_rejected(self):
        item = {
            "checkpoint_step": 0,
            "source_certificate": 0.99,
            "outcome_probabilities": _uniform(4),
            "reward_base": [0.0, 0.0, 0.0, 0.0],
            "reward_slope": [0.0, 0.0, 0.0, 0.0],
        }
        audit = audit_fixed_finite([item], {"required_gain": 0.0, "kind": "affine_interval", "interval": [-1.0, 1.0]}, "recorded_exact", 0.0)
        self.assertEqual(audit["rows"][0]["verification_status"], "REJECTED_STORED_CERTIFICATE")
        self.assertAlmostEqual(audit["rows"][0]["source_certificate"], 0.0)

    def test_simplex_uses_vertices_only(self):
        item = {
            "checkpoint_step": 0,
            "outcome_probabilities": [0.5, 0.5],
            "reward_at_vertices": [[0.0, 1.0], [1.0, 0.0], [0.2, 0.2]],
        }
        audit = audit_fixed_finite([item], {"required_gain": 0.0, "kind": "simplex"}, "recorded_exact", 0.0)
        self.assertAlmostEqual(audit["rows"][0]["source_certificate"], 0.2)

    def test_non_simplex_is_not_a_grid_minimum(self):
        item = {"checkpoint_step": 0, "outcome_probabilities": [1.0], "reward_base": [1.0], "reward_slope": [0.0]}
        audit = audit_fixed_finite([item], {"required_gain": 0.0, "kind": "arbitrary_polytope"}, "recorded_exact", 0.0)
        self.assertEqual(audit["rows"][0]["verification_status"], "NOT_VERIFIED")

    def test_budget(self):
        self.assertFalse(anchor_movement([0.0, 0.0], [1.0, 0.0], 0.0, 1e-4)["passed"])
        self.assertEqual(anchor_movement(None, [0.0], 0.0, 1e-4)["status"], "BUDGET_UNVERIFIED")
        self.assertTrue(anchor_movement([0.0, 0.0], [0.0, 0.0], 0.05, 1e-4)["passed"])

    def test_historical_manifest_is_unverified(self):
        payload = load_manifest(ROOT / "examples/07_guidance_audit/finite_example.json")
        self.assertIn("migration_warning", payload)
        audit = audit_fixed_finite(payload["checkpoints"], payload["compatible_reward_set"], "recorded_exact", 0.04)
        self.assertEqual(audit["rows"][0]["verification_status"], "REPORTED_UNVERIFIED")

    def test_paired_interval_and_mismatch(self):
        base = [{"seed": 0, "fixed_eval": 0.2, "evaluator": "rouge"}, {"seed": 1, "fixed_eval": 0.4, "evaluator": "rouge"}]
        treat = [{"seed": 0, "fixed_eval": 0.3, "evaluator": "rouge"}, {"seed": 1, "fixed_eval": 0.5, "evaluator": "rouge"}]
        result = paired_comparison(base, treat, "fixed_eval")
        self.assertEqual(result["n_seeds"], 2)
        self.assertAlmostEqual(result["paired_mean"], 0.1)
        self.assertIsNotNone(result["interval_95"])
        alone = paired_comparison(base[:1], treat[:1], "fixed_eval")
        self.assertIsNone(alone["interval_95"])
        with self.assertRaises(ComparisonError):
            paired_comparison(
                [{"seed": 0, "fixed_eval": 1, "evaluator": "rouge"}],
                [{"seed": 0, "fixed_eval": 1, "evaluator": "teacher"}],
                "fixed_eval",
            )

    def test_cpu_import_has_no_torch(self):
        import experiments.guidance_audit as package
        source = Path(package.__file__).read_text()
        self.assertNotIn("import torch", source)


def run(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(AuditTests)
    buffer = []

    class Collect(unittest.TextTestResult):
        def addSuccess(self, test):
            super().addSuccess(test)
            buffer.append({"test": str(test), "status": "PASS"})

        def addFailure(self, test, err):
            super().addFailure(test, err)
            buffer.append({"test": str(test), "status": "FAIL", "error": self._exc_info_to_string(err, test)})

        def addError(self, test, err):
            super().addError(test, err)
            buffer.append({"test": str(test), "status": "FAIL", "error": self._exc_info_to_string(err, test)})

    result = unittest.TextTestRunner(resultclass=Collect, verbosity=1).run(suite)
    payload = {"passed": result.wasSuccessful(), "tests": buffer, "failures": len(result.failures) + len(result.errors)}
    (out / "f5_tests.json").write_text(json.dumps(payload, indent=2) + "\n")
    example = ROOT / "examples/08_guidance_audit/finite_computed.json"
    if example.exists():
        from experiments.guidance_audit.finite import audit_fixed_finite as audit
        manifest = load_manifest(example)
        smoked = audit(manifest["checkpoints"], manifest["compatible_reward_set"], manifest["step_rule"], 0.0)
        (out / "example_audit.json").write_text(json.dumps(smoked, indent=2) + "\n")
        payload["example_status"] = smoked["rows"][0]["verification_status"]
        (out / "f5_tests.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload
