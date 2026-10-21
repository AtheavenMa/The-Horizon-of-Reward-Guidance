"""F3 RewardBench 2 scoring with the official aggregation rule.

The 0.5B model is a TL;DR reward head, not an official RewardBench classifier.
Aggregation follows rewardbench.utils.reroll_and_score_dataset and process_single_model.
This is a protocol-compatible replica, not a leaderboard submission.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from experiments.bridge.common import git_sha

ROOT = Path(__file__).resolve().parents[2]
PARQUET = Path("/workspace/.hf_home/hub/datasets--allenai--reward-bench-2/snapshots/7ff08853b0d5686e79b13fda8677024f566a104a/data/test-00000-of-00001.parquet")
FROZEN = ROOT / "experiments/horizon_scale_07/rb2_4b/frozen_rows.json"


def _official_nontie(scores: list[float]) -> float:
    """First completion is the first chosen text. Partial credit if it ties the maximum."""
    maximum = max(scores)
    winners = sum(score == maximum for score in scores)
    return (1.0 / winners) if scores[0] == maximum else 0.0


def _independent_nontie(scores: list[float]) -> float:
    best = max(range(len(scores)), key=lambda index: (scores[index], -index))
    if scores[0] < scores[best]:
        return 0.0
    tied = sum(score == scores[0] for score in scores)
    return 1.0 / tied


def _ties_score(rows: list[dict]) -> dict:
    """rows need id like 'ref:3' or 'tied:3', scores in chosen-then-rejected order, and num_correct."""
    grouped: dict[tuple[str, int], list[tuple[bool, float]]] = {}
    for row in rows:
        kind, prompt_id = row["id"].split(":")
        bucket = grouped.setdefault((kind, int(prompt_id)), [])
        for index, score in enumerate(row["scores"]):
            bucket.append((index < int(row["num_correct"]), float(score)))

    def stats(samples):
        correct = [score for flag, score in samples if flag]
        incorrect = [score for flag, score in samples if not flag]
        best_c, worst_c, best_i = max(correct), min(correct), max(incorrect)
        within = best_c - worst_c if len(correct) > 1 else None
        gap = worst_c - best_i
        return gap > 0, within, gap

    ref, tied = {}, {}
    for (kind, prompt_id), samples in grouped.items():
        (ref if kind == "ref" else tied)[prompt_id] = stats(samples)
    ref_acc = float(np.mean([item[0] for item in ref.values()])) if ref else 0.0
    tied_acc = float(np.mean([item[0] for item in tied.values()])) if tied else 0.0
    shared = sorted(set(ref) & set(tied))
    if not shared:
        return {"overall": 0.5 * tied_acc + 0.5 * ref_acc, "ref_accuracy": ref_acc, "tied_accuracy": tied_acc, "prompts": 0}
    within = np.array([tied[pid][1] for pid in shared], dtype=float)
    gap_t = np.array([tied[pid][2] for pid in shared], dtype=float)
    gap_r = np.array([ref[pid][2] for pid in shared], dtype=float)
    preferred = float(np.mean(gap_t > within))
    preferred_hard = float(np.mean(np.minimum(gap_r, gap_t) > within))
    margin = np.tanh(np.minimum(gap_r, gap_t) / within - 1)
    margin = float(np.nan_to_num(margin, nan=0.0).mean())
    overall = 0.30 * tied_acc + 0.30 * ref_acc + 0.20 * preferred + 0.20 * preferred_hard + 0.01 * margin
    return {"overall": float(overall), "ref_accuracy": ref_acc, "tied_accuracy": tied_acc, "correctness_preferred": preferred, "correctness_preferred_hard": preferred_hard, "prompts": len(shared)}


def _fixture() -> dict:
    cases = [
        [1.0, 0.0, 0.0, 0.0],
        [0.2, 0.2, 0.1],
        [0.0, 1.0, 0.5],
        [3.0, 3.0, 3.0],
        [-1.0, -0.5],
        [0.5, 0.5, 0.5, 0.4],
        [2.0, 2.0, 1.0, 2.0],
        [0.1, 0.2, 0.3, 0.4],
    ]
    mismatches = [index for index, scores in enumerate(cases) if _official_nontie(scores) != _independent_nontie(scores)]
    return {"cases": len(cases), "mismatches": mismatches, "passed": not mismatches}


def _pin() -> str:
    try:
        text = subprocess.check_output(["git", "ls-remote", "https://github.com/allenai/reward-bench.git", "HEAD"], text=True, timeout=30)
        return text.split()[0]
    except Exception as exc:
        return f"unpinned:{exc}"


def _score_rows(rows: list[dict], adapter: Path) -> dict:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", torch_dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(base, adapter / "adapter")
    model.eval()
    head = torch.nn.Linear(model.config.hidden_size, 1, bias=False).to(device="cuda", dtype=torch.bfloat16)
    head.load_state_dict(torch.load(adapter / "head.pt", map_location="cuda"))
    head.eval()
    table = pq.read_table(PARQUET, columns=["id", "prompt", "chosen", "rejected", "subset", "num_correct"])
    scored = []
    truncated = 0
    for meta in rows:
        index = int(meta["row"])
        prompt = table.column("prompt")[index].as_py()
        chosen = list(table.column("chosen")[index].as_py())
        rejected = list(table.column("rejected")[index].as_py())
        texts = [f"Post:\n{prompt}\n\nSummary:\n{text}" for text in chosen + rejected]
        batch = tokenizer(texts, truncation=True, max_length=512, padding=True, return_tensors="pt")
        truncated += int((batch["attention_mask"].sum(1) >= 512).sum())
        batch = {key: value.to(model.device) for key, value in batch.items()}
        with torch.no_grad():
            hidden = model(**batch, output_hidden_states=True).hidden_states[-1]
            loc = batch["attention_mask"].sum(1) - 1
            values = head(hidden[torch.arange(len(texts), device=model.device), loc].to(head.weight.dtype)).float().view(-1)
        scores = [float(value) for value in values]
        scored.append({
            "row": index,
            "id": table.column("id")[index].as_py(),
            "subset": meta["subset"],
            "num_correct": int(table.column("num_correct")[index].as_py()),
            "n_candidates": len(texts),
            "scores": scores,
            "nontie_result": None if meta["subset"] == "Ties" else _official_nontie(scores),
        })
    del model, head
    torch.cuda.empty_cache()
    by = {}
    for name in sorted({row["subset"] for row in scored if row["subset"] != "Ties"}):
        vals = [row["nontie_result"] for row in scored if row["subset"] == name]
        by[name] = float(np.mean(vals)) if vals else None
    ties = _ties_score([row for row in scored if row["subset"] == "Ties"])
    by["Ties"] = ties["overall"]
    return {
        "per_subset": by,
        "macro_on_502": float(np.mean(list(by.values()))),
        "ties_detail": ties,
        "truncated_completions": truncated,
        "rows": [{key: row[key] for key in ("row", "id", "subset", "num_correct", "n_candidates", "nontie_result")} for row in scored],
        "context_cap_tokens": 512,
        "character_truncation": "none",
    }


def run(out: Path, stage: str = "confirm") -> dict:
    started = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    fixture = _fixture()
    pin = _pin()
    if stage == "pilot" or not fixture["passed"]:
        payload = {"status": "pilot" if fixture["passed"] else "invalid", "fixture": fixture, "reward_bench_commit": pin, "passed": fixture["passed"], "runtime_seconds": time.perf_counter() - started}
        (out / "fixture.json").write_text(json.dumps(payload, indent=2) + "\n")
        return payload
    if os.environ.get("HORIZON_GPU") != "1":
        return {"status": "waiting_for_gpu", "passed": False, "fixture": fixture, "runtime_seconds": 0}
    frozen = json.loads(FROZEN.read_text())["rows"]
    adapter = ROOT / "experiments/horizon_scale_07/human_tldr/human_4096"
    selected = ROOT / "experiments/horizon_final_08/human_final_holdout/models_confirm.json"
    if selected.exists():
        name = json.loads(selected.read_text()).get("source_selection", {}).get("selected", {}).get("name")
        if name:
            adapter = ROOT / "experiments/horizon_final_08/human_final_holdout" / name
    scored = _score_rows(frozen, adapter)
    vllm = False
    try:
        import importlib.util
        vllm = importlib.util.find_spec("vllm") is not None
    except Exception:
        vllm = False
    historical = json.loads((ROOT / "experiments/horizon_scale_07/rb2_4b/summary.json").read_text())
    historical_macros = {"rm": (historical.get("rm") or {}).get("macro"), "judge": (historical.get("judge") or {}).get("macro_on_scored_subsets"), "status": "unofficial historical"}
    payload = {
        "status": "complete",
        "protocol": "protocol-compatible independent replica",
        "official_leaderboard_submission": False,
        "reward_bench_commit": pin,
        "fixture": fixture,
        "scalar_rm": scored,
        "frozen_4b_judge": {
            "status": "NOT RUN",
            "reason": "The official generative V2 script scores with vLLM. vLLM is not installed, so the frozen 4B judge was not given a homemade score.",
            "vllm_installed": vllm,
        },
        "qwen15_transfer": "NOT RUN",
        "historical_macros_not_official": historical_macros,
        "runtime_seconds": time.perf_counter() - started,
        "source_sha": git_sha(),
        "passed": True,
        "guarantee_type": "empirical_proxy",
    }
    # Do not store every score vector's prompt. Row ids and results only.
    (out / "summary.json").write_text(json.dumps({key: value for key, value in payload.items() if key != "scalar_rm"}, indent=2) + "\n")
    (out / "scalar_rm.json").write_text(json.dumps(scored, indent=2) + "\n")
    (out / "summary.json").write_text(json.dumps({**{key: value for key, value in payload.items() if key != "scalar_rm"}, "scalar_rm_path": "scalar_rm.json", "macro_on_502": scored["macro_on_502"], "per_subset": scored["per_subset"]}, indent=2) + "\n")
    return payload
