"""Entry point for work plan 01.

Run from the repository root:
  python tasks/01_Reward_Guidance_Experiments/__main__.py run baseline
  python tasks/01_Reward_Guidance_Experiments/__main__.py run E1 --stage pilot
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run_baseline() -> int:
    import pytest
    code = pytest.main(["-q", str(ROOT / "tests")])
    if code != 0:
        return int(code)
    completed = subprocess.run(
        [sys.executable, str(ROOT / "experiments" / "reproduce.py"), "--all"],
        cwd=ROOT,
        check=False,
    )
    return int(completed.returncode)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["run"])
    parser.add_argument("target", choices=["baseline", "checks", "E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "all"])
    parser.add_argument("--stage", default="confirm", choices=["pilot", "confirm"])
    args = parser.parse_args()
    if args.target == "baseline":
        return run_baseline()
    if args.target == "checks":
        from experiments.real.small_rl.phase1 import protocol_checks
        report = protocol_checks()
        ok = report["restore_hash_match"] and report["zero_update_within_tolerance"] and report["teacher_weight_absent_from_checkpoint"] and report["reproduction_reference_unchanged"] and report["pytest_passed"]
        return 0 if ok else 1
    if args.target == "E1":
        from experiments.real.small_rl.phase1 import run_e1
        run_e1(args.stage)
        return 0
    if args.target == "E2":
        from experiments.real.small_rl.phase1 import run_e2
        run_e2()
        return 0
    if args.target == "E3":
        from experiments.real.small_rl.phase1 import run_e3
        run_e3()
        return 0
    if args.target == "E4":
        from experiments.real.control.pebble import run_e4
        run_e4(args.stage)
        return 0
    if args.target == "E5":
        from experiments.real.control.e5 import run_e5
        run_e5(args.stage)
        return 0
    if args.target == "E6":
        from experiments.real.language.e6 import run_e6
        run_e6(args.stage)
        return 0
    if args.target == "E7":
        from experiments.real.language.e7 import run_e7
        run_e7(args.stage)
        return 0
    if args.target == "E8":
        from experiments.real.analysis.e8 import run_e8
        run_e8()
        return 0
    raise SystemExit(f"{args.target} is not wired yet")


if __name__ == "__main__":
    raise SystemExit(main())
