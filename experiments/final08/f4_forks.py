"""F4 forks from F2 checkpoints that contain AdamW state.

Parents are A_source and B_bounded at steps 16 and 48. Each arm reloads that
actor and optimizer before it changes anything. M and the policy continuation
were frozen in F0.
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.bridge.common import git_sha
from experiments.final08.f2_pythia import (
    B,
    EPS,
    _fit_candidate,
    _generate,
    _kl,
    _line_search,
    _load_policy,
    _load_trainable,
    _margins,
    _pythia07,
    _quality,
    _raw_margins,
    _rms,
    _texts,
    _token_logprobs,
    _trainable,
)

ROOT = Path(__file__).resolve().parents[2]
PYTHIA = ROOT / "experiments/horizon_final_08/pythia_bounded"
POOLS = ROOT / "experiments/horizon_final_08/audited_baselines/pythia_pools.json"
HUMAN = ROOT / "experiments/horizon_scale_07/human_tldr"
CHOICES = ROOT / "experiments/horizon_final_08/audited_baselines/dev_choices.json"
ARMS = ("K", "C", "H", "A", "B", "Z", "T")


def _choices() -> dict:
    payload = json.loads(CHOICES.read_text())
    return int(payload["fork_rm_steps_M"]), int(payload["fork_policy_updates"])


def _parents(stage: str) -> list[Path]:
    names = []
    for method in ("A_source", "B_bounded"):
        for seed in (0, 1):
            for step in (16, 48):
                names.append(PYTHIA / f"{method}_seed{seed}_step{step}.pt")
    if stage == "pilot":
        names = names[:1]
    return [path for path in names if path.exists()]


def _install(policy, blob) -> torch.optim.Optimizer:
    named = dict(policy.named_parameters())
    with torch.no_grad():
        for name, value in blob["lora"].items():
            named[name].copy_(value.to(named[name].device))
    opt = torch.optim.AdamW((param for param in policy.parameters() if param.requires_grad), lr=5e-5)
    opt.load_state_dict(copy.deepcopy(blob["optimizer"]))
    try:
        torch.set_rng_state(blob["torch_rng"])
        if blob.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(blob["cuda_rng"])
        restored = True
    except Exception:
        restored = False
    return opt, restored


def _same_outputs(policy, tokenizer, prompts) -> bool:
    first = _generate(policy, tokenizer, prompts[:4], 123, 32, False)
    second = _generate(policy, tokenizer, prompts[:4], 123, 32, False)
    return [row["text"] for row in first] == [row["text"] for row in second]


def _score_summaries(rm, head, tokenizer, prompts, summaries) -> np.ndarray:
    dummy = [""] * len(prompts)
    # A margin against an empty summary is not the summary score. Score directly.
    texts = [f"Post:\n{prompt}\n\nSummary:\n{summary}" for prompt, summary in zip(prompts, summaries)]
    batch = tokenizer(texts, truncation=True, max_length=512, padding=True, return_tensors="pt")
    batch = {key: value.to(rm.device) for key, value in batch.items()}
    with torch.no_grad():
        hidden = rm(**batch, output_hidden_states=True).hidden_states[-1]
        index = batch["attention_mask"].sum(1) - 1
        values = head(hidden[torch.arange(len(texts), device=rm.device), index].to(head.weight.dtype)).float().view(-1)
    return values.cpu().numpy()


def _continue(policy, opt, tokenizer, prompts, score_fn, steps: int, seed: int, max_new: int) -> dict:
    stopped = 0
    last_kl = 0.0
    for step in range(1, steps + 1):
        batch = prompts[((step * 8) % max(len(prompts) - 8, 1)):][:8]
        generated = _generate(policy, tokenizer, batch, seed * 100 + step, max_new, True)
        scores = np.array(score_fn(batch, [row["text"] for row in generated]), dtype=np.float64)
        advantages = scores - scores.mean()
        kls = [_kl(policy, row["prompt_ids"], row["new_ids"]) for row in generated]
        last_kl = float(np.mean(kls)) if kls else 0.0
        if last_kl > 0.1:
            stopped += 1
            continue
        policy.train()
        opt.zero_grad()
        for row, advantage in zip(generated, advantages):
            logp = _token_logprobs(policy, row["prompt_ids"], row["new_ids"]).sum()
            (-(float(advantage) * logp) / max(len(generated), 1)).backward()
        torch.nn.utils.clip_grad_norm_((param for param in policy.parameters() if param.requires_grad), 1.0)
        opt.step()
    return {"kl_steps_stopped": stopped, "last_train_kl": last_kl}


def _pair_rows(which: str) -> tuple[list[str], list[str], list[str], list[bool]]:
    import importlib.util
    spec = importlib.util.spec_from_file_location("human07", ROOT / "experiments/large_scale/07_human_tldr.py")
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    nested = json.loads((HUMAN / "nested_docs.json").read_text())
    uniform = json.loads((HUMAN / "uniform_docs.json").read_text())
    source_docs = set(nested["4096"])
    docs = sorted(source_docs) if which == "source" else sorted(doc for doc in uniform if doc not in source_docs)
    records = legacy._pairs_for(HUMAN, docs)[:32]
    prompts, left, right, prefer = [], [], [], []
    for record in records:
        post, a, b, choice = legacy._load_pair(record)
        prompts.append(post)
        left.append(a)
        right.append(b)
        prefer.append(choice == 0)
    return prompts, left, right, prefer


def _bounded_fit(rm, head, tokenizer, prompts, left, right, prefer, anchors, steps: int) -> dict:
    anchor_prompts, anchor_left, anchor_right = anchors
    old = _trainable(rm, head)
    initial = _raw_margins(rm, head, tokenizer, anchor_prompts, anchor_left, anchor_right)
    _fit_candidate(rm, head, tokenizer, prompts, left, right, prefer, steps)
    cand = _trainable(rm, head)
    decision = _line_search(rm, head, tokenizer, old, cand, anchor_prompts, anchor_left, anchor_right, initial)
    decision["rm_steps"] = steps
    decision["label_count"] = len(prompts)
    if decision["per_event_rms"] > B + EPS:
        _load_trainable(rm, head, old)
        decision["applied"] = False
        decision["status"] = "BLOCKED"
    else:
        decision["applied"] = True
        decision["status"] = "APPLIED"
    return decision


def _holdout(rm, head, tokenizer) -> dict:
    from experiments.final08.f1_human import _batched_eval, _legacy, _rows
    legacy = _legacy()
    rows = _rows("final")
    temperature = 1.7
    selected = ROOT / "experiments/horizon_final_08/human_final_holdout/human_4098_seed0/metrics.json"
    if selected.exists():
        temperature = float(json.loads(selected.read_text()).get("temperature_frozen_on_dev", temperature))
    metrics, _, _ = _batched_eval(rm, head, tokenizer, rows, rm.device, legacy, temperature)
    return {key: metrics[key] for key in ("accuracy", "nll", "brier", "ece", "heldout_n")}


def _regen_anchors(seed: int, prompts: list[str], max_new: int) -> tuple[list[dict], list[dict], bool]:
    policy, tokenizer = _load_policy(100 + seed)
    left = _generate(policy, tokenizer, prompts, 50_000 + seed, max_new, True)
    right = _generate(policy, tokenizer, prompts, 60_000 + seed, max_new, True)
    del policy
    torch.cuda.empty_cache()
    recorded = json.loads((PYTHIA / f"anchors_seed{seed}.json").read_text())
    matched = [row["hash"] for row in left] == recorded["left"] and [row["hash"] for row in right] == recorded["right"]
    return left, right, matched


def run(out: Path, stage: str = "confirm") -> dict:
    started = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    parents = _parents(stage)
    if len(parents) < (1 if stage == "pilot" else 8):
        payload = {"status": "BLOCKED", "passed": False, "reason": "F2 optimizer checkpoints at steps 16 and 48 are incomplete.", "found": [path.name for path in parents], "runtime_seconds": 0}
        (out / f"{stage}.json").write_text(json.dumps(payload, indent=2) + "\n")
        return payload
    m_steps, policy_steps = _choices()
    if stage == "pilot":
        m_steps, policy_steps = 2, 2
    pools = json.loads(POOLS.read_text())
    train, _ = _texts(pools["train_docs"])
    eval_prompts, eval_refs = _texts(pools["policy_eval_docs"])
    anchor_prompts, _ = _texts(pools["movement_anchor_docs"])
    max_new = 32 if stage == "pilot" else 64
    if stage == "pilot":
        train, eval_prompts, eval_refs, anchor_prompts = train[:8], eval_prompts[:8], eval_refs[:8], anchor_prompts[:8]
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    human = ROOT / "experiments/horizon_final_08/human_final_holdout/human_4098_seed0"
    rm_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    if rm_tokenizer.pad_token is None:
        rm_tokenizer.pad_token = rm_tokenizer.eos_token
    base_rm = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", torch_dtype=torch.bfloat16, device_map="cuda")
    rm = PeftModel.from_pretrained(base_rm, human / "adapter")
    head = torch.nn.Linear(rm.config.hidden_size, 1, bias=False).to(device="cuda", dtype=torch.float32)
    head.load_state_dict({key: value.float() for key, value in torch.load(human / "head.pt", map_location="cpu").items()})
    rm.eval()
    head.eval()
    source_pairs = _pair_rows("source")
    hidden_pairs = _pair_rows("hidden")
    legacy = _pythia07()
    teacher = teacher_tokenizer = None
    anchors = {}
    rows = []
    policy, tokenizer = _load_policy(0)
    for path in parents:
        blob = torch.load(path, map_location="cpu")
        seed = int(path.name.split("_seed")[1].split("_")[0])
        if seed not in anchors:
            left, right, matched = _regen_anchors(seed, anchor_prompts, max_new)
            anchors[seed] = ([row["text"] for row in left], [row["text"] for row in right], matched)
        anchor_left, anchor_right, matched = anchors[seed]
        anchor_pack = (anchor_prompts, anchor_left, anchor_right)
        opt, rng_ok = _install(policy, blob)
        identical = _same_outputs(policy, tokenizer, train)
        _load_trainable(rm, head, blob["reward"])
        parent_quality = _quality(policy, tokenizer, eval_prompts, eval_refs, 7000 + int(blob["step"]), max_new)
        info_after = _generate(policy, tokenizer, train[:4], 123, max_new, False)
        # Touch the reward, then decode again with no actor update.
        _load_trainable(rm, head, blob["reward"])
        info_again = _generate(policy, tokenizer, train[:4], 123, max_new, False)
        no_actor_identical = [row["text"] for row in info_after] == [row["text"] for row in info_again]
        parent_nll = _holdout(rm, head, rm_tokenizer)
        if teacher is None:
            teacher, teacher_tokenizer = legacy._load_teacher()
        online_left = _generate(policy, tokenizer, train[:32], 11_000 + seed * 10 + int(blob["step"]), max_new, True)
        online_right = _generate(policy, tokenizer, train[:32], 21_000 + seed * 10 + int(blob["step"]), max_new, True)
        left_s = legacy._teacher_scores(teacher, teacher_tokenizer, train[:32], [row["text"] for row in online_left])
        right_s = legacy._teacher_scores(teacher, teacher_tokenizer, train[:32], [row["text"] for row in online_right])
        online = (train[:32], [row["text"] for row in online_left], [row["text"] for row in online_right], [a >= b for a, b in zip(left_s, right_s)])
        both = (
            hidden_pairs[0] + online[0],
            hidden_pairs[1] + online[1],
            hidden_pairs[2] + online[2],
            hidden_pairs[3] + online[3],
        )
        for arm in ARMS:
            slot = out / f"{path.stem}_{arm}.json"
            if slot.exists() and stage == "confirm":
                rows.append(json.loads(slot.read_text()))
                print(path.name, arm, "cached", flush=True)
                continue
            opt, rng_ok = _install(policy, blob)
            _load_trainable(rm, head, blob["reward"])
            t0 = time.perf_counter()
            movement = {"mode": "none", "per_event_rms": 0.0, "applied": False, "rm_steps": 0, "label_count": 0, "status": "UNCHANGED", "proposal_rms": 0.0}
            label_origin = "none"
            if arm == "C":
                movement = _bounded_fit(rm, head, rm_tokenizer, *source_pairs, anchor_pack, m_steps)
                label_origin = "same human source pairs already in the 4096-document pool"
            elif arm == "H":
                movement = _bounded_fit(rm, head, rm_tokenizer, *hidden_pairs, anchor_pack, m_steps)
                label_origin = "human TRAIN pairs outside the 4096-document source pool"
            elif arm == "A":
                movement = _bounded_fit(rm, head, rm_tokenizer, *online, anchor_pack, m_steps)
                label_origin = "32 on-policy pairs labeled by the frozen AI teacher"
            elif arm == "B":
                movement = _bounded_fit(rm, head, rm_tokenizer, *both, anchor_pack, m_steps * 2)
                label_origin = "the same 32 hidden human pairs and the same 32 AI pairs"
            elif arm == "Z":
                old = _trainable(rm, head)
                initial = _raw_margins(rm, head, rm_tokenizer, anchor_prompts, anchor_left, anchor_right)
                _fit_candidate(rm, head, rm_tokenizer, *online, m_steps)
                proposed = _raw_margins(rm, head, rm_tokenizer, anchor_prompts, anchor_left, anchor_right)
                _load_trainable(rm, head, old)
                deployed = _raw_margins(rm, head, rm_tokenizer, anchor_prompts, anchor_left, anchor_right)
                movement = {"mode": "b0", "per_event_rms": _rms(deployed, initial), "proposal_rms": _rms(proposed, initial), "applied": False, "rm_steps": m_steps, "label_count": 32, "status": "NOT_DEPLOYED"}
                label_origin = "same 32 AI pairs as A; candidate was not deployed"
            elif arm == "T":
                label_origin = "frozen AI teacher scores, no learned reward update"
                movement = {"mode": "teacher", "per_event_rms": None, "applied": False, "rm_steps": 0, "label_count": 0, "status": "ORACLE"}
            fit_seconds = time.perf_counter() - t0
            if arm == "T":
                def score_fn(prompts, summaries, _teacher=teacher, _tok=teacher_tokenizer):
                    return np.array(legacy._teacher_scores(_teacher, _tok, prompts, summaries), dtype=np.float64)
            else:
                def score_fn(prompts, summaries, _rm=rm, _head=head, _tok=rm_tokenizer):
                    return _score_summaries(_rm, _head, _tok, prompts, summaries)
            train_info = _continue(policy, opt, tokenizer, train, score_fn, policy_steps, 80 + seed, max_new)
            after = _quality(policy, tokenizer, eval_prompts, eval_refs, 7100 + int(blob["step"]), max_new)
            if arm == "T":
                nll = None
            elif not movement.get("applied"):
                nll = parent_nll
            else:
                nll = _holdout(rm, head, rm_tokenizer)
            row = {
                "parent": path.name,
                "parent_step": int(blob["step"]),
                "seed": seed,
                "arm": arm,
                "optimizer_restored": True,
                "rng_restored": rng_ok,
                "repeat_decode_identical": identical,
                "no_actor_update_identical": no_actor_identical,
                "anchor_hashes_match_f2": matched,
                "label_origin": label_origin,
                "rm_steps": movement.get("rm_steps", 0),
                "label_count": movement.get("label_count", 0),
                "movement": {key: movement.get(key) for key in ("mode", "alpha", "per_event_rms", "proposal_rms", "applied", "status", "weight_interpolation_stable", "architecture")},
                "parent_rouge_l": parent_quality["rouge_l"],
                "after_rouge_l": after["rouge_l"],
                "delta": after["rouge_l"] - parent_quality["rouge_l"],
                "reference_kl": after["reference_kl_mean_per_token"],
                "mean_tokens": after["mean_tokens"],
                "train_kl_stopped": train_info["kl_steps_stopped"],
                "human_holdout": nll,
                "fit_seconds": fit_seconds,
                "evaluator": "rouge_l_vs_dataset_completion",
                "guarantee_type": "empirical_proxy",
            }
            slot.write_text(json.dumps(row, indent=2) + "\n")
            rows.append(row)
            print(path.name, arm, round(row["delta"], 4), movement.get("status"), flush=True)
        del blob
        torch.cuda.empty_cache()
    if teacher is not None:
        del teacher
        torch.cuda.empty_cache()
    by_parent = {}
    for row in rows:
        by_parent.setdefault(row["parent"], {})[row["arm"]] = row["delta"]
    for row in rows:
        keep = by_parent.get(row["parent"], {}).get("K")
        row["paired_delta_vs_K"] = None if keep is None else row["delta"] - keep
    budget_ok = all(
        (row["movement"].get("per_event_rms") is None) or row["movement"]["per_event_rms"] <= B + EPS
        for row in rows
    )
    payload = {
        "status": "complete" if stage == "confirm" else "pilot",
        "rows": rows,
        "M": m_steps,
        "policy_updates": policy_steps,
        "b": B,
        "epsilon": EPS,
        "anchor_hash_match": {str(seed): anchors[seed][2] for seed in anchors},
        "runtime_seconds": time.perf_counter() - started,
        "source_sha": git_sha(),
        "passed": bool(rows) and budget_ok and all(row["repeat_decode_identical"] and row["optimizer_restored"] and row["no_actor_update_identical"] for row in rows),
        "guarantee_type": "empirical_proxy",
    }
    (out / f"{stage}.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload
