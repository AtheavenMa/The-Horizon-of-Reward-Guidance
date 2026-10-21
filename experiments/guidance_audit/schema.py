"""Versioned manifest checks. Missing fields fail loudly; nothing is invented."""
from __future__ import annotations

import json
from pathlib import Path

SCHEMA_VERSION = 2
ACCEPTED_SCHEMA_VERSIONS = {1, 2}
GUARANTEE_TYPES = {"exact_finite", "statistical_fixed_design", "empirical_proxy"}
SOURCE_TYPES = {"human", "model_judge", "synthetic_btl", "verifier"}
STATUSES = {"complete", "censored", "incomplete", "invalid"}

REQUIRED = (
    "schema_version",
    "experiment_id",
    "task_id",
    "source_type",
    "evaluator_type",
    "guarantee_type",
    "dataset_id",
    "checkpoint_step",
    "new_label_count",
    "status",
)


class ManifestError(ValueError):
    """A user-supplied record cannot be audited as written."""


def load_manifest(path: str | Path) -> dict:
    file = Path(path)
    if not file.is_file():
        raise ManifestError(f"manifest file does not exist: {file}")
    try:
        payload = json.loads(file.read_text())
    except json.JSONDecodeError as exc:
        raise ManifestError(f"manifest is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ManifestError("manifest must be a JSON object")
    missing = [name for name in REQUIRED if name not in payload]
    if missing:
        raise ManifestError("manifest is missing required fields: " + ", ".join(missing))
    version = int(payload["schema_version"])
    if version not in ACCEPTED_SCHEMA_VERSIONS:
        raise ManifestError(
            f"unsupported schema_version {payload['schema_version']}; expected one of {sorted(ACCEPTED_SCHEMA_VERSIONS)}"
        )
    if version == 1:
        payload = dict(payload)
        payload["migration_warning"] = (
            "schema_version 1 is accepted as a historical manifest. Stored certificates are not recomputed."
        )
    if payload["guarantee_type"] not in GUARANTEE_TYPES:
        raise ManifestError(
            f"guarantee_type must be one of {sorted(GUARANTEE_TYPES)}; got {payload['guarantee_type']!r}"
        )
    if payload["source_type"] not in SOURCE_TYPES:
        raise ManifestError(f"source_type must be one of {sorted(SOURCE_TYPES)}")
    if payload["status"] not in STATUSES:
        raise ManifestError(f"status must be one of {sorted(STATUSES)}")
    if payload["guarantee_type"] == "exact_finite" and payload["evaluator_type"] not in {"exact_finite", "exact"}:
        raise ManifestError("an exact_finite guarantee requires an exact evaluator, not a neural or human proxy")
    if payload["guarantee_type"] == "empirical_proxy" and payload.get("original_certified_gain") not in (None, "not_available"):
        raise ManifestError("an empirical neural or human manifest cannot assert an exact certified gain")
    steps = payload.get("checkpoints")
    if isinstance(steps, list) and steps:
        order = [int(item["checkpoint_step"]) for item in steps]
        if order != sorted(order):
            raise ManifestError("checkpoint ordering is not nondecreasing")
    return payload
