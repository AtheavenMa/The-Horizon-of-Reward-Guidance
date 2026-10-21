"""Work Plan 08 entry. Do not launch experiment modules from the shell.

  python tasks/08_Final_Milestone_and_Paper_Freeze/__main__.py run F0
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "experiments" / "horizon_final_08"


def _run(name: str, function: str, folder: Path, **kwargs):
    module = importlib.import_module(f"experiments.final08.{name}")
    folder.mkdir(parents=True, exist_ok=True)
    result = getattr(module, function)(folder, **kwargs)
    print(json.dumps({key: result.get(key) for key in ("status", "passed", "runtime_seconds") if isinstance(result, dict)}, default=str))
    return 0 if not isinstance(result, dict) or result.get("passed", True) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Work Plan 08 task entry.")
    parser.add_argument("command", choices=["run"])
    parser.add_argument("target", choices=["F0", "F1", "F2", "F3", "F4", "F5", "F6"])
    parser.add_argument("--stage", default="confirm", choices=["pilot", "confirm"])
    parser.add_argument("--part", default="all", choices=["all", "human", "logistic", "minigrid"])
    args = parser.parse_args(argv)
    if args.target == "F0":
        return _run("f0_audit", "run", OUT / "audited_baselines")
    if args.target == "F1":
        code = 0
        if args.part in {"all", "logistic"}:
            code |= _run("f1_human", "run_logistic", OUT / "human_final_holdout", stage=args.stage)
        if args.part in {"all", "human"}:
            code |= _run("f1_human", "run_models", OUT / "human_final_holdout", stage=args.stage)
        if args.part in {"all", "minigrid"}:
            code |= _run("f1_minigrid", "run", OUT / "minigrid_controls", stage=args.stage)
        return code
    if args.target == "F2":
        return _run("f2_pythia", "run", OUT / "pythia_bounded", stage=args.stage)
    if args.target == "F3":
        return _run("f3_rewardbench", "run", OUT / "rb2_corrected", stage=args.stage)
    if args.target == "F4":
        return _run("f4_forks", "run", OUT / "checkpoint_forks", stage=args.stage)
    if args.target == "F5":
        return _run("f5_tests", "run", OUT / "public_smoke")
    if args.target == "F6":
        return _run("f6_pack", "run", OUT / "figures")
    raise SystemExit(args.target)


if __name__ == "__main__":
    raise SystemExit(main())
