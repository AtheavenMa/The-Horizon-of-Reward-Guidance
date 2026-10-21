"""Public reward-guidance audit. Exact, fixed-design, and empirical results stay separate."""
from experiments.guidance_audit.decisions import compare_interventions, recommend
from experiments.guidance_audit.empirical import audit_empirical, select_pairs
from experiments.guidance_audit.finite import audit_fixed_finite, frozen_policy_label_check, scalar_reference
from experiments.guidance_audit.reporting import summarize_runs
from experiments.guidance_audit.schema import ManifestError, load_manifest

__all__ = [
    "ManifestError",
    "audit_empirical",
    "audit_fixed_finite",
    "compare_interventions",
    "frozen_policy_label_check",
    "load_manifest",
    "recommend",
    "scalar_reference",
    "select_pairs",
    "summarize_runs",
]
