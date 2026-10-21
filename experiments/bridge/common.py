"""Shared bookkeeping for the theorem-anchored bridge. Not a new benchmark."""
from __future__ import annotations

import json
import resource
import subprocess
import time
from pathlib import Path


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def rss_kib() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def write_summary(path: Path, payload: dict, started: float) -> None:
    payload = dict(payload)
    payload["source_sha"] = git_sha()
    payload["runtime_seconds"] = time.perf_counter() - started
    payload["max_rss_kib"] = rss_kib()
    path.mkdir(parents=True, exist_ok=True)
    (path / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: payload[k] for k in payload if k != "rows"}, indent=2))
