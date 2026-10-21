"""Seed-paired comparison. Different reward heads are not one utility."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path


_T95 = {1: None, 2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}


class ComparisonError(ValueError):
    """The two runs cannot be compared as paired replicates."""


def _load_rows(folder: Path) -> list[dict]:
    rows = []
    for path in sorted(folder.glob("*.json")):
        if path.name in {"summary.json", "manifest.json"}:
            continue
        payload = json.loads(path.read_text())
        if isinstance(payload, dict) and "seed" in payload:
            rows.append(payload)
        elif isinstance(payload, dict) and "rows" in payload:
            rows.extend(payload["rows"])
    csv_path = folder / "metrics.csv"
    if csv_path.exists():
        with csv_path.open() as handle:
            rows.extend(list(csv.DictReader(handle)))
    return rows


def _metric(row: dict, metric: str):
    if metric not in row or row[metric] in (None, ""):
        return None
    return float(row[metric])


def paired_comparison(baseline: list[dict], treatment: list[dict], metric: str, pair_by: str = "seed") -> dict:
    by_base = {}
    by_treat = {}
    for row in baseline:
        by_base[str(row[pair_by])] = row
    for row in treatment:
        by_treat[str(row[pair_by])] = row
    keys = sorted(set(by_base) & set(by_treat))
    deltas = []
    used = []
    for key in keys:
        left, right = by_base[key], by_treat[key]
        if left.get("policy_sha") and right.get("policy_sha") and left["policy_sha"] != right["policy_sha"] and pair_by != "seed":
            raise ComparisonError("NON-COMPARABLE: policy hashes differ")
        if left.get("evaluator") and right.get("evaluator") and left["evaluator"] != right["evaluator"]:
            raise ComparisonError("NON-COMPARABLE: evaluators differ")
        a, b = _metric(left, metric), _metric(right, metric)
        if a is None or b is None:
            continue
        deltas.append(b - a)
        used.append(key)
    n = len(deltas)
    if n == 0:
        raise ComparisonError("NON-COMPARABLE: no aligned rows contain the metric")
    mean = sum(deltas) / n
    if n < 2:
        interval = None
        sd = None
    else:
        var = sum((value - mean) ** 2 for value in deltas) / (n - 1)
        sd = math.sqrt(var)
        critical = _T95.get(n, 1.96)
        half = critical * sd / math.sqrt(n)
        interval = [mean - half, mean + half]
    return {
        "metric": metric,
        "pair_by": pair_by,
        "n_seeds": n,
        "paired_ids": used,
        "deltas": deltas,
        "paired_mean": mean,
        "sample_sd": sd,
        "interval_95": interval,
        "interval_method": None if interval is None else "paired_t",
        "fraction_positive_seeds": sum(1 for value in deltas if value > 0) / n,
        "note": "Seed-level interval only. Prompt-level bootstrap is a different unit and is not this number.",
    }


def write_comparison(baseline: Path, treatment: Path, metric: str, out: Path, pair_by: str = "seed") -> dict:
    result = paired_comparison(_load_rows(baseline), _load_rows(treatment), metric, pair_by)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    with (out / "paired_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pair_id", "delta"])
        writer.writeheader()
        for key, delta in zip(result["paired_ids"], result["deltas"]):
            writer.writerow({"pair_id": key, "delta": delta})
    return result
