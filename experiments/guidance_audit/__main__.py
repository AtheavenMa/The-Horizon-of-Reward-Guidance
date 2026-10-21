"""CLI: validate, audit, compare. Hugging Face and MiniGrid are not imported here."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.guidance_audit.finite import audit_fixed_finite
from experiments.guidance_audit.reporting import summarize_runs
from experiments.guidance_audit.schema import load_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit a reward-guidance manifest without inventing a certificate.")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate", help="Check a manifest and print its guarantee type.")
    validate.add_argument("--manifest", required=True)
    audit = sub.add_parser("audit", help="Run the exact finite audit described by a manifest.")
    audit.add_argument("--manifest", required=True)
    audit.add_argument("--out", required=True)
    compare = sub.add_parser("compare", help="Paired seed comparison of one shared metric.")
    compare.add_argument("--baseline", required=True)
    compare.add_argument("--treatment", required=True)
    compare.add_argument("--metric", default="fixed_eval")
    compare.add_argument("--pair-by", default="seed")
    compare.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    if args.command == "validate":
        payload = load_manifest(args.manifest)
        print(json.dumps({"experiment_id": payload["experiment_id"], "guarantee_type": payload["guarantee_type"], "status": payload["status"]}, indent=2))
        return 0
    if args.command == "audit":
        payload = load_manifest(args.manifest)
        if payload["guarantee_type"] != "exact_finite":
            print(json.dumps({"guarantee_type": payload["guarantee_type"], "formal_horizon": "not_certified", "status": "empirical_only"}, indent=2))
            return 0
        result = audit_fixed_finite(
            payload["checkpoints"],
            payload["compatible_reward_set"],
            payload.get("step_rule", "recorded_exact"),
            float(payload.get("adaptation_budget_value") or 0.0),
        )
        folder = Path(args.out)
        summarize_runs([{"name": "first_unsupported", "value": 0 if result["censored"] else result["first_unsupported_checkpoint"]}, {"name": "rows", "value": len(result["rows"])}], folder)
        (folder / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"censored": result["censored"], "first_unsupported_checkpoint": result["first_unsupported_checkpoint"]}, indent=2))
        return 0
    from experiments.guidance_audit.compare_stats import ComparisonError, write_comparison
    baseline = Path(args.baseline)
    treatment = Path(args.treatment)
    if not baseline.exists() or not treatment.exists():
        print("NON-COMPARABLE: baseline and treatment folders must already exist", file=sys.stderr)
        return 2
    try:
        result = write_comparison(baseline, treatment, args.metric, Path(args.out or baseline.parent / "public_compare"), args.pair_by)
    except ComparisonError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
