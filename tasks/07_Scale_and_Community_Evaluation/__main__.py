"""Entry point for Work Plan 07. Run from the repository root.

  python tasks/07_Scale_and_Community_Evaluation/__main__.py run S0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "experiments" / "horizon_scale_07"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Work Plan 07 task entry. Do not call experiment modules from the shell.")
    parser.add_argument("command", choices=["run"])
    parser.add_argument("target", choices=["S0", "S1", "S2", "S3", "S4", "S5", "S6"])
    parser.add_argument("--stage", default="confirm", choices=["pilot", "confirm"])
    args = parser.parse_args(argv)
    if args.target == "S0":
        import importlib.util
        spec = importlib.util.spec_from_file_location("s0", ROOT / "experiments/large_scale/07_preflight.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result = module.run_s0(OUT / "preflight")
        print({"passed": result["passed"], "runtime_seconds": result["runtime_seconds"]})
        return 0 if result["passed"] else 1
    if args.target == "S1":
        import importlib.util
        spec = importlib.util.spec_from_file_location("s1", ROOT / "experiments/large_scale/07_minigrid.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        folder = OUT / "minigrid"
        result = module.run_pilot(folder) if args.stage == "pilot" else module.run_confirm(folder)
        print({"learned": result.get("learned"), "runtime_seconds": result.get("runtime_seconds")})
        return 0
    if args.target == "S2":
        import importlib.util
        spec = importlib.util.spec_from_file_location("s2", ROOT / "experiments/large_scale/07_human_tldr.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        folder = OUT / "human_tldr"
        if args.stage == "pilot":
            result = module.freeze(folder)
            result["frozen_feature"] = module.frozen_feature(folder)
        else:
            result = module.train_ladder(folder)
        print({key: result.get(key) for key in ("nested_pairs", "train_valid_doc_overlap", "runtime_seconds", "status")})
        return 0
    if args.target == "S4":
        import importlib.util
        spec = importlib.util.spec_from_file_location("s4", ROOT / "experiments/large_scale/07_rewardbench.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        folder = OUT / "rb2_4b"
        if args.stage == "pilot":
            result = module.freeze_rows(folder)
        else:
            result = module.score(folder)
        print({"rows_frozen": result.get("rows_frozen"), "status": result.get("status"), "runtime_seconds": result.get("runtime_seconds")})
        return 0
    if args.target == "S3":
        import importlib.util
        spec = importlib.util.spec_from_file_location("s3", ROOT / "experiments/large_scale/07_pythia.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        folder = OUT / "pythia_1b"
        result = module.run_gate(folder) if args.stage == "pilot" else module.run_main(folder)
        print({"passed": result.get("passed"), "status": result.get("status"), "runtime_seconds": result.get("runtime_seconds")})
        return 0 if result.get("passed", True) else 1
    if args.target == "S5":
        import importlib.util
        spec = importlib.util.spec_from_file_location("s5", ROOT / "experiments/large_scale/07_pythia.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result = module.run_forks(OUT / "checkpoint_forks")
        print({"status": result.get("status"), "runtime_seconds": result.get("runtime_seconds")})
        return 0
    raise SystemExit(f"{args.target} is not wired yet")


if __name__ == "__main__":
    raise SystemExit(main())
