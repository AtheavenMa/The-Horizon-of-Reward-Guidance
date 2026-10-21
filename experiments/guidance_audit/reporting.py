"""Stable CSV, JSON, and one small plot."""
from __future__ import annotations

import csv
import json
from pathlib import Path


def summarize_runs(records: list[dict], folder: Path) -> dict:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "summary.json").write_text(json.dumps({"records": records}, indent=2) + "\n")
    fields = sorted({key for record in records for key in record})
    with (folder / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key) for key in fields})
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 3.2))
        names = [str(record.get("name", index)) for index, record in enumerate(records)]
        values = [record.get("value") for record in records]
        numeric = [float(value) if isinstance(value, (int, float)) else 0.0 for value in values]
        ax.bar(range(len(names)), numeric, color="0.35")
        ax.set_xticks(range(len(names)), names, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Recorded value")
        fig.tight_layout()
        fig.savefig(folder / "summary.png", dpi=120)
        plt.close(fig)
    except Exception as exc:  # plotting is optional for the CPU smoke
        (folder / "plot_error.txt").write_text(str(exc))
    return {"json": str(folder / "summary.json"), "csv": str(folder / "summary.csv")}
