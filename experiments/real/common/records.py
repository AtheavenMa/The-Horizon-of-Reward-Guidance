"""Per-run artifacts required by the work plan, section 2.1."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
RUN_ROOT = ROOT / "experiments" / "horizon_real"


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def run_dir(experiment: str, *parts: str) -> Path:
    path = RUN_ROOT / experiment
    for part in parts:
        path = path / part
    path.mkdir(parents=True, exist_ok=True)
    (path / "checkpoints").mkdir(exist_ok=True)
    return path


def write_run(path: Path, config: dict, metrics_rows: list[dict], events: list[dict], status: dict) -> None:
    metadata = {
        "git_sha": git_sha(),
        "python": os.sys.version.split()[0],
        "numpy": np.__version__,
        "config_keys": sorted(config.keys()),
    }
    (path / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (path / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (path / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    (path / "run_status.json").write_text(json.dumps(status, indent=2) + "\n")
    if metrics_rows:
        keys: list[str] = []
        seen = set()
        for row in metrics_rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        lines = [",".join(keys)]
        for row in metrics_rows:
            lines.append(",".join(_cell(row.get(k, "")) for k in keys))
        (path / "metrics.csv").write_text("\n".join(lines) + "\n")
    else:
        (path / "metrics.csv").write_text("")


def _cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.8g}"
    text = str(value)
    if any(ch in text for ch in ",\n\""):
        return '"' + text.replace('"', "'") + '"'
    return text


def save_checkpoint(path: Path, name: str, payload: dict) -> Path:
    dest = path / "checkpoints" / name
    np.savez_compressed(dest, **_pack(payload))
    return dest.with_suffix(".npz")


def load_checkpoint(path: Path) -> dict:
    data = np.load(path, allow_pickle=False)
    return {key: data[key] for key in data.files}


def _pack(payload: dict) -> dict:
    packed = {}
    for key, value in payload.items():
        if isinstance(value, (dict, list)):
            packed[key] = np.array(json.dumps(value))
        else:
            packed[key] = np.asarray(value)
    return packed


def unpack_json(value: np.ndarray) -> Any:
    return json.loads(str(value))
