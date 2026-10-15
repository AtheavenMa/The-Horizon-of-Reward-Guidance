#!/usr/bin/env python3
"""Run the original numerical ledger and regenerate all seven paper figures.

Usage: python experiments/reproduce.py --all
Outputs are isolated in outputs/paper_reproduction by default. Original
reference CSV/JSON under results/reference/ is never modified.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "results" / "reference"


def assert_equivalent(expected: Any, observed: Any, path: str = "root") -> None:
    """Compare paper ledger semantically, tolerating platform floating-point noise."""
    if isinstance(expected, dict):
        if not isinstance(observed, dict):
            raise AssertionError(f"{path}: type mismatch")
        keys = expected.keys() - {"environment"}
        if keys != observed.keys() - {"environment"}:
            raise AssertionError(f"{path}: key mismatch")
        for key in keys:
            assert_equivalent(expected[key], observed[key],f"{path}.{key}")
    elif isinstance(expected, list):
        if not isinstance(observed,list) or len(expected) != len(observed):
            raise AssertionError(f"{path}: sequence mismatch")
        for i,(a,b) in enumerate(zip(expected,observed)):
            assert_equivalent(a,b,f"{path}[{i}]")
    elif isinstance(expected,(int,float)) and not isinstance(expected,bool):
        if not isinstance(observed,(int,float)) or isinstance(observed,bool):
            raise AssertionError(f"{path}: numeric mismatch")
        if not math.isclose(float(expected),float(observed),rel_tol=1e-6,abs_tol=2e-7):
            raise AssertionError(f"{path}: {observed} vs reference {expected}")
    elif expected != observed:
        raise AssertionError(f"{path}: {observed!r} vs {expected!r}")


def check_reproduction(output: Path) -> dict[str,int]:
    counters={"json":0,"csv":0,"figures":0}
    for source in sorted(REFERENCE.iterdir()):
        generated=output/source.name
        if not generated.exists():
            raise AssertionError(f"missing {generated}")
        if source.suffix==".json":
            assert_equivalent(json.loads(source.read_text()),json.loads(generated.read_text()),source.name)
            counters["json"]+=1
        elif source.suffix==".csv":
            with source.open(newline="") as f:
                a=list(csv.reader(f))
            with generated.open(newline="") as f:
                b=list(csv.reader(f))
            if len(a)!=len(b):
                raise AssertionError(f"{source.name}: number of rows differs")
            for i,(ra,rb) in enumerate(zip(a,b)):
                if len(ra)!=len(rb):
                    raise AssertionError(f"{source.name}:{i}: number of fields differs")
                for j,(sa,sb) in enumerate(zip(ra,rb)):
                    try:
                        x,y=float(sa),float(sb)
                    except ValueError:
                        if sa!=sb:
                            raise AssertionError(f"{source.name}:{i}:{j}: {sa!r} != {sb!r}")
                    else:
                        if math.isnan(x) and math.isnan(y):
                            continue
                        if not math.isclose(x,y,rel_tol=1e-6,abs_tol=2e-7):
                            raise AssertionError(f"{source.name}:{i}:{j}: {x} != {y}")
            counters["csv"]+=1
    names=["closed_loop","ordinary_pg","instance_complexity","query_energy",
           "full_support_separation","actual_adaptation","discrete_resources"]
    for name in names:
        for path in [output/f"{name}.pdf",output/"png"/f"{name}.png"]:
            if not path.is_file() or path.stat().st_size<1000:
                raise AssertionError(f"missing or empty figure {path}")
        counters["figures"]+=1
    return counters


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all",action="store_true",help="run entire numerical and figure pipeline")
    parser.add_argument("--out",type=Path,default=ROOT/"outputs"/"paper_reproduction")
    args=parser.parse_args()
    if not args.all:
        parser.error("pass --all to explicitly run all paper checks")
    out=args.out.resolve()
    if out == REFERENCE.resolve() or REFERENCE.resolve() in out.parents:
        parser.error("refusing to overwrite frozen reference outputs")
    out.mkdir(parents=True,exist_ok=True)
    commands=[
        [sys.executable,str(ROOT/"experiments"/"verify_v10.py"),"--out",str(out)],
        [sys.executable,str(ROOT/"plotting"/"figures.py"),"--res",str(out),"--out",str(out),"--venue","iclr"]
    ]
    for cmd in commands:
        subprocess.run(cmd,cwd=ROOT,check=True)
    summary=check_reproduction(out)
    print("\nPASS: full numerical ledger and figure reproduction",summary)
    (out/"reproduction_status.json").write_text(json.dumps({"status":"PASS","checks":summary},indent=2))


if __name__ == "__main__":
    main()
