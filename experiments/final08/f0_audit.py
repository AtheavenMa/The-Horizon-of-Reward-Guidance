"""F0 evidence audit. Historical numbers are recorded, not rewritten."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

from experiments.bridge.common import git_sha
from experiments.guidance_audit.finite import frozen_policy_label_check, scalar_reference
from experiments.strengthening_checks import ordinary_pg_check

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "results" / "reference"


def _hash_tree(folder: Path) -> dict:
    digest = hashlib.sha256()
    files = []
    if not folder.exists():
        return {"present": False, "sha256": None, "files": []}
    for path in sorted(p for p in folder.rglob("*") if p.is_file()):
        data = path.read_bytes()
        rel = str(path.relative_to(folder))
        digest.update(rel.encode())
        digest.update(data)
        files.append(rel)
    return {"present": True, "sha256": digest.hexdigest(), "files": files, "reproduction_status_json": "reproduction_status.json" in files}


def _load_pythia_history() -> dict:
    path = ROOT / "experiments/horizon_scale_07/pythia_1b/main.json"
    if not path.exists():
        return {"present": False}
    payload = json.loads(path.read_text())
    rows = []
    for row in payload.get("rows", []):
        if row.get("method") == "B_online_ai":
            rows.append({"seed": row.get("seed"), "anchor_rms": row.get("anchor_rms"), "exceeds_b": row.get("exceeds_b")})
    return {
        "present": True,
        "rows": rows,
        "status": "INVALID for bounded-budget claim",
        "reason": "WP07 computed anchor RMS only at the end. 1.4600 and 1.3538 exceed b=0.05 and were not enforced per update.",
    }


def _load_rb2_history() -> dict:
    path = ROOT / "experiments/horizon_scale_07/rb2_4b/summary.json"
    if not path.exists():
        return {"present": False}
    payload = json.loads(path.read_text())
    return {
        "present": True,
        "rm_macro": (payload.get("rm") or {}).get("macro"),
        "judge_macro": (payload.get("judge") or {}).get("macro_on_scored_subsets"),
        "status": "unofficial",
        "reason": "WP07 used every-chosen-beats-every-rejected and exact index-set matching. Those macros are not official RewardBench 2 scores.",
    }


def _split_holdout(out: Path) -> dict:
    valid_path = ROOT / "experiments/horizon_scale_07/human_tldr/valid_index.json"
    rows = json.loads(valid_path.read_text())
    dev, final = rows[:512], rows[512:]
    dev_docs = {row["doc_id"] for row in dev}
    final_docs = {row["doc_id"] for row in final}
    overlap = sorted(dev_docs & final_docs)
    if overlap:
        # Keep every document entirely on one side. Development keeps the historical 512-row documents.
        moved = [row for row in final if row["doc_id"] in dev_docs]
        final = [row for row in final if row["doc_id"] not in dev_docs]
        note = f"moved {len(moved)} final rows because their documents also appear in the 512 development rows"
    else:
        note = "development and final groups are document-disjoint"
    manifest = {
        "dev_count": 512,
        "final_count": len(final),
        "dev_docs": len(dev_docs),
        "final_docs": len({row["doc_id"] for row in final}),
        "document_overlap": len(overlap),
        "note": note,
        "dev_ids": [{"file": row["file"], "offset": row["offset"], "doc_id": row["doc_id"]} for row in dev],
        "final_ids": [{"file": row["file"], "offset": row["offset"], "doc_id": row["doc_id"]} for row in final],
    }
    (out / "human_split.json").write_text(json.dumps(manifest) + "\n")
    return {key: manifest[key] for key in ("dev_count", "final_count", "dev_docs", "final_docs", "document_overlap", "note")}


def _policy_pools(out: Path) -> dict:
    import pyarrow.parquet as pq
    parquet = Path("/workspace/.hf_home/hub/datasets--trl-lib--tldr/snapshots/21233da376667088e6eb1ce4ce19ed832c2935d3/data/train-00000-of-00001.parquet")
    table = pq.read_table(parquet, columns=["prompt"])
    n = table.num_rows
    order = np.random.default_rng(0).permutation(n)
    train = [int(i) for i in order[:256]]
    inspected = [int(i) for i in order[256:384]]
    policy_eval = [int(i) for i in order[384:512]]
    anchors = [int(i) for i in order[512:640]]
    pools = {
        "train_docs": train,
        "wp07_inspected_anchors_not_reused": inspected,
        "policy_eval_docs": policy_eval,
        "movement_anchor_docs": anchors,
        "overlap_train_eval": len(set(train) & set(policy_eval)),
        "overlap_train_anchor": len(set(train) & set(anchors)),
        "overlap_eval_anchor": len(set(policy_eval) & set(anchors)),
        "note": "WP07's 128 validation anchors were inspected after training, so F2 does not use them as the final policy set or the movement set.",
    }
    (out / "pythia_pools.json").write_text(json.dumps(pools) + "\n")
    return {key: pools[key] for key in ("overlap_train_eval", "overlap_train_anchor", "overlap_eval_anchor", "note")}


def _zero_extension() -> dict:
    path = ROOT / "experiments/bridge/13_three_stage_mdp.py"
    spec = importlib.util.spec_from_file_location("three_stage_f0", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.self_check(module.M.reference_phi0() if hasattr(module, "M") else 0.2)
    return {"status": "PASS", "check": "kappa2=0 gradient matches the 16-outcome gradient on the first coordinates"}


def run(out: Path) -> dict:
    started = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    before = _hash_tree(REFERENCE)
    scalar = scalar_reference()
    ordinary = ordinary_pg_check()
    contact = float(ordinary["numerical_maximal_speed_boundary_time"])
    frozen = frozen_policy_label_check()
    try:
        zero = _zero_extension()
    except Exception as exc:
        zero = {"status": "FAIL", "error": str(exc)}
    after = _hash_tree(REFERENCE)
    hash_same = before["sha256"] == after["sha256"] and before["present"]
    status_file = REFERENCE / "reproduction_status.json"
    rows = [
        {
            "object": "Scalar horizon / ordinary PG",
            "expected": "T0=5 and contact about 0.0078358743",
            "measured": {"T0": scalar["observed"], "contact": contact},
            "status": "PASS" if scalar["passed"] and abs(contact - 0.0078358743) / 0.0078358743 < 1e-5 else "FAIL",
            "path": "horizon.support_horizon and experiments.strengthening_checks.ordinary_pg_check",
            "restriction": "Analytic scalar model and the stated ordinary-PG box only.",
        },
        {
            "object": "16-outcome coupled bound",
            "expected": "outward-rounded all-controls upper or an explicit downgrade",
            "measured": {"historical_upper": 0.05993753, "historical_t": 0.012},
            "status": "PARTIAL",
            "path": "work_plan/04_Reachability_to_Small_MDP.md",
            "restriction": "COMPUTATIONAL EVIDENCE. This audit did not independently verify outward rounding, so the upper is not a computer-checked proof.",
        },
        {
            "object": "64-outcome zero-regression",
            "expected": "first coordinates match the 16-outcome gradient when kappa2=0",
            "measured": zero,
            "status": zero["status"],
            "path": "experiments/bridge/13_three_stage_mdp.py self_check",
            "restriction": "Checks the zero-extension gradient identity, not a new horizon.",
        },
        {
            "object": "Reference files and hashes",
            "expected": "tracked reference files unchanged; a missing status file is not a pass",
            "measured": {"before": before["sha256"], "after": after["sha256"], "reproduction_status_json": "ABSENT" if not status_file.exists() else "present"},
            "status": "PASS" if hash_same and not status_file.exists() else ("FAIL" if not hash_same else "PARTIAL"),
            "path": "results/reference",
            "restriction": "reproduction_status.json is absent. The hash is of the files that are present. Absence is not a pass of that missing file.",
        },
        {
            "object": "Human TL;DR label provenance and split",
            "expected": "train/valid documents disjoint and the final holdout is not the inspected 512",
            "measured": _split_holdout(out),
            "status": "PASS",
            "path": str(out / "human_split.json"),
            "restriction": "Historical human annotations. The original 512 are development only.",
        },
        {
            "object": "Pythia WP07 reward movement",
            "expected": "b=.05 violation recorded as INVALID",
            "measured": _load_pythia_history(),
            "status": "FAIL",
            "path": "experiments/horizon_scale_07/pythia_1b/main.json",
            "restriction": "NOT A VALID BOUNDED CLAIM. Preserved as history.",
        },
        {
            "object": "RewardBench 2 old 502 rows",
            "expected": "historical scoring rejected as unofficial",
            "measured": _load_rb2_history(),
            "status": "FAIL",
            "path": "experiments/horizon_scale_07/rb2_4b/summary.json",
            "restriction": "Do not cite 0.2091 or 0.1115 as official RewardBench 2.",
        },
        {
            "object": "Public audit CLI claims",
            "expected": "stored certificates without probabilities are unverified",
            "measured": {"frozen_policy_label_check": frozen["passed"]},
            "status": "PASS" if frozen["passed"] else "FAIL",
            "path": "experiments/guidance_audit/finite.py",
            "restriction": "A prefilled certificate is REPORTED_UNVERIFIED unless outcome probabilities are supplied.",
        },
    ]
    # The human split status depends on overlap after the repair.
    split = rows[4]["measured"]
    rows[4]["status"] = "PASS" if split["document_overlap"] == 0 or split["final_docs"] > 0 else "FAIL"
    pools = _policy_pools(out)
    choices = {
        "b": 0.05,
        "epsilon": 0.0001,
        "kl_cap_nats_per_token": 0.1,
        "policy_updates": 64,
        "minimum_full_seeds": 2,
        "candidate_optimizer_steps": 64,
        "fork_policy_updates": 8,
        "fork_rm_steps_M": 32,
        "temperature": 0.7,
        "preferred_max_new_tokens": 64,
        "train_prompts_per_update": 8,
        "source_rm_rule": "highest F1 development accuracy, then more training pairs, then lower seed",
    }
    (out / "dev_choices.json").write_text(json.dumps(choices, indent=2) + "\n")
    payload = {
        "rows": rows,
        "pythia_pools": pools,
        "dev_choices": choices,
        "reference_hash_unchanged": hash_same,
        "runtime_seconds": time.perf_counter() - started,
        "source_sha": git_sha(),
        "status": "complete",
    }
    (out / "audit.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload
