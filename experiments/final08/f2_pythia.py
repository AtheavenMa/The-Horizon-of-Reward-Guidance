"""F2 budget-respecting Pythia-1B run.

The deployed reward changes only when the anchor RMS of that event is within b+epsilon.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from experiments.bridge.common import git_sha

ROOT = Path(__file__).resolve().parents[2]
POOLS = ROOT / "experiments/horizon_final_08/audited_baselines/pythia_pools.json"
CHOICES = ROOT / "experiments/horizon_final_08/audited_baselines/dev_choices.json"
SFT = "cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr"
TEACHER = "cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr"
TLDR = Path("/workspace/.hf_home/hub/datasets--trl-lib--tldr/snapshots/21233da376667088e6eb1ce4ce19ed832c2935d3/data/train-00000-of-00001.parquet")
B = 0.05
EPS = 1e-4
DEPLOY = {"mode": "weights"}


def _pythia07():
    spec = importlib.util.spec_from_file_location("pythia07", ROOT / "experiments/large_scale/07_pythia.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _texts(indices: list[int]) -> tuple[list[str], list[str]]:
    import pyarrow.parquet as pq
    table = pq.read_table(TLDR, columns=["prompt", "completion"])
    prompts = [table.column("prompt")[i].as_py() for i in indices]
    refs = [table.column("completion")[i].as_py() for i in indices]
    return prompts, refs


def _rouge_l(hyp: str, ref: str) -> float:
    a, b = hyp.split(), ref.split()
    if not a or not b:
        return 0.0
    prev = [0] * (len(b) + 1)
    for token in a:
        cur = [0]
        for j, other in enumerate(b, start=1):
            cur.append(prev[j - 1] + 1 if token == other else max(cur[-1], prev[j]))
        prev = cur
    overlap = prev[-1]
    precision, recall = overlap / len(a), overlap / len(b)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _boundary_report(tokenizer) -> dict:
    prompt, summary = "Post: token boundary.\n\nSummary:", " A short ending"
    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    summary_ids = tokenizer(summary, add_special_tokens=False)["input_ids"]
    direct = prompt_ids + summary_ids
    glued = tokenizer(prompt + summary, add_special_tokens=True)["input_ids"]
    separate_prompt = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    return {
        "direct_equals_glued": direct == glued,
        "separate_prompt_length": len(separate_prompt),
        "direct_length": len(direct),
        "policy_loss_uses": "token ids of the prompt concatenated with generated continuation ids",
    }


def _load_policy(seed: int):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(SFT)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(SFT, torch_dtype=torch.bfloat16, device_map="cuda")
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, target_modules=["query_key_value"], bias="none", task_type="CAUSAL_LM"))
    return model, tokenizer


def _generate(model, tokenizer, prompts: list[str], seed: int, max_new: int, sample: bool) -> list[dict]:
    model.eval()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    batch = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=384)
    batch = {key: value.to(model.device) for key, value in batch.items()}
    model.config.use_cache = True
    kwargs = {"max_new_tokens": max_new, "do_sample": sample, "pad_token_id": tokenizer.pad_token_id}
    if sample:
        kwargs["temperature"] = 0.7
    with torch.no_grad():
        gen = model.generate(**batch, **kwargs)
    model.config.use_cache = False
    width = batch["input_ids"].shape[1]
    rows = []
    for i, prompt in enumerate(prompts):
        mask = batch["attention_mask"][i].bool()
        prompt_ids = batch["input_ids"][i][mask].tolist()
        new_ids = gen[i, width:].tolist()
        text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        rows.append({"prompt": prompt, "prompt_ids": prompt_ids, "new_ids": new_ids, "text": text, "hash": _hash(text)})
    return rows


def _token_logprobs(model, prompt_ids: list[int], new_ids: list[int]) -> torch.Tensor:
    if not new_ids:
        return torch.zeros(0, device=model.device)
    ids = torch.tensor([prompt_ids + new_ids], device=model.device)
    start = len(prompt_ids)
    logits = model(input_ids=ids).logits[0, start - 1:-1]
    target = ids[0, start:]
    return torch.log_softmax(logits.float(), dim=-1).gather(-1, target.unsqueeze(-1)).squeeze(-1)


def _kl(model, prompt_ids, new_ids) -> float:
    if not new_ids:
        return 0.0
    current = _token_logprobs(model, prompt_ids, new_ids)
    with model.disable_adapter():
        reference = _token_logprobs(model, prompt_ids, new_ids)
    return float((current - reference).detach().mean())


def _trainable(model, head) -> dict:
    return {
        "lora": {name: param.detach().cpu().clone() for name, param in model.named_parameters() if param.requires_grad},
        "head": {name: param.detach().cpu().clone() for name, param in head.state_dict().items()},
    }


def _load_trainable(model, head, state) -> None:
    named = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in state["lora"].items():
            named[name].copy_(value.to(named[name].device))
        head.load_state_dict({name: value.to(head.weight.device) for name, value in state["head"].items()})


def _mix(old, cand, alpha: float) -> dict:
    mixed = {"lora": {}, "head": {}}
    for name, value in old["lora"].items():
        mixed["lora"][name] = (1 - alpha) * value + alpha * cand["lora"][name]
    for name, value in old["head"].items():
        mixed["head"][name] = (1 - alpha) * value + alpha * cand["head"][name]
    return mixed


def _raw_margins(rm, head, tokenizer, prompts, left, right) -> np.ndarray:
    def score(texts):
        batch = tokenizer(texts, truncation=True, max_length=512, padding=True, return_tensors="pt")
        batch = {key: value.to(rm.device) for key, value in batch.items()}
        with torch.no_grad():
            hidden = rm(**batch, output_hidden_states=True).hidden_states[-1]
            index = batch["attention_mask"].sum(1) - 1
            values = head(hidden[torch.arange(len(texts), device=rm.device), index].to(head.weight.dtype)).view(-1)
        return values.float().cpu().numpy()
    left_text = [f"Post:\n{p}\n\nSummary:\n{s}" for p, s in zip(prompts, left)]
    right_text = [f"Post:\n{p}\n\nSummary:\n{s}" for p, s in zip(prompts, right)]
    return score(left_text) - score(right_text)


def _margins(rm, head, tokenizer, prompts, left, right) -> np.ndarray:
    if DEPLOY.get("mode") != "two_model_mixture":
        return _raw_margins(rm, head, tokenizer, prompts, left, right)
    _load_trainable(rm, head, DEPLOY["old"])
    old = _raw_margins(rm, head, tokenizer, prompts, left, right)
    _load_trainable(rm, head, DEPLOY["cand"])
    cand = _raw_margins(rm, head, tokenizer, prompts, left, right)
    _load_trainable(rm, head, DEPLOY["old"])
    alpha = float(DEPLOY["alpha"])
    return old + alpha * (cand - old)


def _rms(new, old) -> float:
    return float(np.sqrt(np.mean((np.asarray(new) - np.asarray(old)) ** 2)))


def _fit_candidate(rm, head, tokenizer, prompts, left, right, prefer_left, steps: int) -> None:
    rm.train()
    head.train()
    opt = torch.optim.AdamW([param for param in list(rm.parameters()) + list(head.parameters()) if param.requires_grad], lr=1e-4)
    order = list(range(len(prompts)))
    seen = 0
    while seen < steps:
        for index in order:
            chosen, rejected = (left[index], right[index]) if prefer_left[index] else (right[index], left[index])
            texts = [f"Post:\n{prompts[index]}\n\nSummary:\n{chosen}", f"Post:\n{prompts[index]}\n\nSummary:\n{rejected}"]
            batch = tokenizer(texts, truncation=True, max_length=512, padding=True, return_tensors="pt")
            batch = {key: value.to(rm.device) for key, value in batch.items()}
            hidden = rm(**batch, output_hidden_states=True).hidden_states[-1]
            index_tok = batch["attention_mask"].sum(1) - 1
            values = head(hidden[torch.arange(2, device=rm.device), index_tok].to(head.weight.dtype)).view(2)
            loss = -torch.nn.functional.logsigmoid(values[0] - values[1])
            opt.zero_grad()
            loss.backward()
            opt.step()
            seen += 1
            if seen >= steps:
                break
    rm.eval()
    head.eval()


def _line_search(rm, head, tokenizer, old, cand, prompts, left, right, base) -> dict:
    def at(alpha):
        _load_trainable(rm, head, _mix(old, cand, alpha))
        return _raw_margins(rm, head, tokenizer, prompts, left, right)
    cand_margins = _raw_margins(rm, head, tokenizer, prompts, left, right)
    zero = at(0.0)
    full = at(1.0)
    weight_ok = _rms(zero, base) <= EPS and _rms(full, cand_margins) <= 1e-3
    if not weight_ok:
        _load_trainable(rm, head, old)
        gap = _rms(full, base)
        alpha = 0.0 if gap <= EPS else min(1.0, (B + EPS) / gap)
        DEPLOY.clear()
        DEPLOY.update({"mode": "two_model_mixture", "alpha": alpha, "old": old, "cand": cand})
        return {"mode": "two_model_mixture", "alpha": alpha, "per_event_rms": alpha * gap, "accepted": True, "weight_interpolation_stable": False, "architecture": "two_model_mixture"}
    lo, hi = 0.0, 1.0
    best = 0.0
    for _ in range(8):
        mid = (lo + hi) / 2
        if _rms(at(mid), base) <= B + EPS:
            best = mid
            lo = mid
        else:
            hi = mid
    accepted_margins = at(best)
    _load_trainable(rm, head, _mix(old, cand, best))
    DEPLOY.clear()
    DEPLOY["mode"] = "weights"
    return {
        "mode": "weight_interpolation",
        "alpha": best,
        "per_event_rms": _rms(accepted_margins, base),
        "accepted": True,
        "weight_interpolation_stable": True,
    }


def _quality(model, tokenizer, prompts, refs, seed: int, max_new: int) -> dict:
    rows = _generate(model, tokenizer, prompts, seed, max_new, sample=False)
    sampled = _generate(model, tokenizer, prompts, seed + 17, max_new, sample=True)
    rouge = [_rouge_l(row["text"], ref) for row, ref in zip(rows, refs)]
    lengths = [len(row["text"].split()) for row in rows]
    unique = len({tuple(row["text"].split()[i:i + 2]) for row in rows for i in range(max(len(row["text"].split()) - 1, 0))}) 
    tokens = sum(max(len(row["text"].split()) - 1, 0) for row in rows)
    kl = float(np.mean([_kl(model, row["prompt_ids"], row["new_ids"]) for row in rows])) if rows else 0.0
    return {
        "rouge_l": float(np.mean(rouge)),
        "mean_tokens": float(np.mean(lengths)) if lengths else 0.0,
        "empty_rate": float(np.mean([row["text"] == "" for row in rows])),
        "distinct_bigram": float(unique / tokens) if tokens else 0.0,
        "reference_kl_mean_per_token": kl,
        "sampled_rouge_l": float(np.mean([_rouge_l(row["text"], ref) for row, ref in zip(sampled, refs)])),
        "evaluator": "rouge_l_vs_dataset_completion",
        "decoding": "deterministic primary; temperature 0.7 sampled secondary",
    }


def _save_checkpoint(path: Path, model, opt, rm_state, step: int, numpy_state) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step": step,
        "lora": {name: param.detach().cpu() for name, param in model.named_parameters() if "lora_" in name},
        "optimizer": opt.state_dict(),
        "reward": rm_state,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
        "numpy_rng": numpy_state,
        "sft": SFT,
    }, path)


def _run_arm(seed, method, train_prompts, eval_prompts, eval_refs, anchor_prompts, anchor_left, anchor_right, rm, head, rm_tokenizer, out: Path, max_new: int, updates: int, replay=None) -> dict:
    policy, tokenizer = _load_policy(seed)
    opt = torch.optim.AdamW((param for param in policy.parameters() if param.requires_grad), lr=5e-5)
    DEPLOY.clear()
    DEPLOY["mode"] = "weights"
    init_reward = _trainable(rm, head)
    _load_trainable(rm, head, init_reward)
    base = _margins(rm, head, rm_tokenizer, anchor_prompts, anchor_left, anchor_right)
    origin = np.array(base, copy=True)
    events = []
    curve = [_quality(policy, tokenizer, eval_prompts, eval_refs, 5000 + seed, max_new)]
    curve[0]["step"] = 0
    new_labels = 0
    stopped = 0
    replay = replay or {}
    for step in range(1, updates + 1):
        if method == "B_bounded" and (step in (16, 32, 48) or (updates < 16 and step == min(4, updates))):
            event = _reward_event(policy, tokenizer, rm, head, rm_tokenizer, train_prompts, anchor_prompts, anchor_left, anchor_right, base, init_reward, seed, step, max_new, enforce=True)
            events.append(event)
            new_labels += 32
            if event["mode"] == "two_model_mixture":
                # The deployed score is the mixture. Keep both states and score with it below.
                pass
        elif method == "C_b0" and step in replay:
            _load_trainable(rm, head, init_reward)
            now = _margins(rm, head, rm_tokenizer, anchor_prompts, anchor_left, anchor_right)
            events.append({"step": step, "mode": "b0", "per_event_rms": _rms(now, base), "applied": False, "new_ai_labels": 32})
            new_labels += 32
            base = now
        start = ((step * 8) % max(len(train_prompts) - 8, 1))
        batch = train_prompts[start:start + 8]
        rows = _generate(policy, tokenizer, batch, seed * 1000 + step, max_new, sample=True)
        if method == "E_teacher":
            legacy = _pythia07()
            teacher, teacher_tokenizer = legacy._load_teacher()
            scores = np.array(legacy._teacher_scores(teacher, teacher_tokenizer, batch, [row["text"] for row in rows]), dtype=np.float64)
            del teacher
            torch.cuda.empty_cache()
        else:
            scores = np.array(_margins(rm, head, rm_tokenizer, batch, [row["text"] for row in rows], [""] * len(rows)), dtype=np.float64)
        advantages = scores - scores.mean()
        kls = [_kl(policy, row["prompt_ids"], row["new_ids"]) for row in rows]
        mean_kl = float(np.mean(kls)) if kls else 0.0
        if mean_kl > 0.1:
            stopped += 1
        else:
            policy.train()
            opt.zero_grad()
            for row, advantage in zip(rows, advantages):
                logp = _token_logprobs(policy, row["prompt_ids"], row["new_ids"]).sum()
                (-(float(advantage) * logp) / max(len(rows), 1)).backward()
            torch.nn.utils.clip_grad_norm_((param for param in policy.parameters() if param.requires_grad), 1.0)
            opt.step()
        if step in (0, 16, 32, 48, 64) or step == updates:
            quality = _quality(policy, tokenizer, eval_prompts, eval_refs, 5000 + seed, max_new)
            quality["step"] = step
            quality["reference_kl_on_train_batch"] = mean_kl
            curve.append(quality)
            _save_checkpoint(out / f"{method}_seed{seed}_step{step}.pt", policy, opt, _trainable(rm, head), step, None)
            print(method, seed, step, quality["rouge_l"], "kl", round(mean_kl, 4), flush=True)
    final_margins = _margins(rm, head, rm_tokenizer, anchor_prompts, anchor_left, anchor_right)
    del policy
    torch.cuda.empty_cache()
    return {
        "seed": seed,
        "method": method,
        "new_ai_labels": new_labels,
        "updates": updates,
        "events": events,
        "max_per_event_rms": max([event.get("per_event_rms", 0.0) for event in events], default=0.0),
        "cumulative_rms_vs_initial": _rms(final_margins, origin),
        "curve": curve,
        "fixed_eval": curve[-1]["rouge_l"] if curve else None,
        "evaluator": "rouge_l_vs_dataset_completion",
        "kl_steps_stopped": stopped,
        "guarantee_type": "empirical_proxy",
        "empirical_reuse_endpoint": "CENSORED",
        "censor_reason": "No DEV-frozen failure of independent ROUGE-L was declared as a neural support horizon, and none is inferred.",
    }


def _reward_event(policy, tokenizer, rm, head, rm_tokenizer, train_prompts, anchor_prompts, anchor_left, anchor_right, base, init_reward, seed, step, max_new, enforce: bool) -> dict:
    if DEPLOY.get("mode") == "two_model_mixture":
        return {"step": step, "mode": "two_model_mixture", "per_event_rms": 0.0, "applied": False, "status": "BLOCKED", "new_ai_labels": 0, "reason": "The deployed reward is already a two-model mixture, so another single-LoRA update was not applied."}
    legacy = _pythia07()
    start = ((step * 32) % max(len(train_prompts) - 32, 1))
    batch = train_prompts[start:start + 32]
    left = _generate(policy, tokenizer, batch, 10_000 + step, max_new, True)
    right = _generate(policy, tokenizer, batch, 20_000 + step, max_new, True)
    teacher, teacher_tokenizer = legacy._load_teacher()
    left_s = legacy._teacher_scores(teacher, teacher_tokenizer, batch, [row["text"] for row in left])
    right_s = legacy._teacher_scores(teacher, teacher_tokenizer, batch, [row["text"] for row in right])
    del teacher
    torch.cuda.empty_cache()
    old = _trainable(rm, head)
    initial = _margins(rm, head, rm_tokenizer, anchor_prompts, anchor_left, anchor_right)
    cand_model_state = copy.deepcopy(old)
    _load_trainable(rm, head, cand_model_state)
    _fit_candidate(rm, head, rm_tokenizer, batch, [row["text"] for row in left], [row["text"] for row in right], [a >= b for a, b in zip(left_s, right_s)], 64)
    cand = _trainable(rm, head)
    if not enforce:
        _load_trainable(rm, head, cand)
        moved = _margins(rm, head, rm_tokenizer, anchor_prompts, anchor_left, anchor_right)
        return {"step": step, "mode": "unbounded", "per_event_rms": _rms(moved, initial), "applied": True, "new_ai_labels": 32, "label_source": "AI feedback"}
    decision = _line_search(rm, head, rm_tokenizer, old, cand, anchor_prompts, anchor_left, anchor_right, initial)
    decision.update({"step": step, "new_ai_labels": 32, "label_source": "AI feedback", "initial_margins_rms_vs_source": _rms(initial, base)})
    if decision["per_event_rms"] > B + EPS:
        _load_trainable(rm, head, old)
        decision["applied"] = False
        decision["status"] = "BLOCKED"
    else:
        decision["applied"] = True
        decision["status"] = "APPLIED"
    return decision


def _holdout_nll(rm, head, tokenizer, out: Path) -> dict:
    """Score R0 and each bounded step-64 reward on the frozen human FINAL split."""
    from experiments.final08.f1_human import _batched_eval, _legacy, _rows
    legacy = _legacy()
    rows = _rows("final")
    temperature = 1.7
    selected = ROOT / "experiments/horizon_final_08/human_final_holdout/human_4098_seed0/metrics.json"
    if selected.exists():
        temperature = float(json.loads(selected.read_text()).get("temperature_frozen_on_dev", temperature))
    report = {"split": "F1_FINAL", "n": len(rows), "temperature_frozen_on_dev": temperature, "selection_used_final": False}
    source, _, _ = _batched_eval(rm, head, tokenizer, rows, rm.device, legacy, temperature)
    report["R0"] = {key: source[key] for key in ("accuracy", "nll", "brier", "ece", "heldout_n")}
    for seed in (0, 1):
        path = out / f"B_bounded_seed{seed}_step64.pt"
        if not path.exists():
            report[f"B_seed{seed}"] = None
            continue
        blob = torch.load(path, map_location="cpu")
        _load_trainable(rm, head, blob["reward"])
        scored, _, _ = _batched_eval(rm, head, tokenizer, rows, rm.device, legacy, temperature)
        report[f"B_seed{seed}_step64"] = {key: scored[key] for key in ("accuracy", "nll", "brier", "ece", "heldout_n")}
    return report


def run(out: Path, stage: str = "confirm") -> dict:
    started = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    legacy = _pythia07()
    tokenizer = __import__("transformers").AutoTokenizer.from_pretrained(SFT)
    boundary = _boundary_report(tokenizer)
    (out / "token_boundary.json").write_text(json.dumps(boundary, indent=2) + "\n")
    if os.environ.get("HORIZON_GPU") != "1" or not torch.cuda.is_available():
        payload = {"status": "waiting_for_gpu", "passed": False, "runtime_seconds": time.perf_counter() - started, "token_boundary": boundary}
        (out / f"{stage}.json").write_text(json.dumps(payload, indent=2) + "\n")
        return payload
    pools = json.loads(POOLS.read_text())
    train, _ = _texts(pools["train_docs"])
    eval_prompts, eval_refs = _texts(pools["policy_eval_docs"])
    anchor_prompts, _ = _texts(pools["movement_anchor_docs"])
    max_new = 64
    updates = 8 if stage == "pilot" else 64
    seeds = [0] if stage == "pilot" else [0, 1]
    methods = ["B_bounded", "C_b0"] if stage == "pilot" else ["A_source", "B_bounded", "C_b0"]
    if stage == "pilot":
        train, eval_prompts, eval_refs, anchor_prompts = train[:8], eval_prompts[:8], eval_refs[:8], anchor_prompts[:8]
    human = _selected_human()
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    rm_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    if rm_tokenizer.pad_token is None:
        rm_tokenizer.pad_token = rm_tokenizer.eos_token
    base_rm = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", torch_dtype=torch.bfloat16, device_map="cuda")
    rm = PeftModel.from_pretrained(base_rm, human / "adapter")
    head = torch.nn.Linear(rm.config.hidden_size, 1, bias=False).to(device="cuda", dtype=torch.float32)
    head.load_state_dict({key: value.float() for key, value in torch.load(human / "head.pt", map_location="cpu").items()})
    rm.eval()
    head.eval()
    source_reward = _trainable(rm, head)
    rows = []
    replays = {}
    for seed in seeds:
        pending = []
        for method in methods:
            done = out / f"{method}_seed{seed}.json"
            if done.exists() and stage == "confirm" and int(json.loads(done.read_text()).get("updates", 0)) == updates:
                continue
            pending.append(method)
        if not pending:
            for method in methods:
                rows.append(json.loads((out / f"{method}_seed{seed}.json").read_text()))
            continue
        policy, policy_tokenizer = _load_policy(100 + seed)
        left = _generate(policy, policy_tokenizer, anchor_prompts, 50_000 + seed, max_new, True)
        right = _generate(policy, policy_tokenizer, anchor_prompts, 60_000 + seed, max_new, True)
        del policy
        torch.cuda.empty_cache()
        (out / f"anchors_seed{seed}.json").write_text(json.dumps({"left": [row["hash"] for row in left], "right": [row["hash"] for row in right]}) + "\n")
        for method in methods:
            done = out / f"{method}_seed{seed}.json"
            if done.exists() and stage == "confirm":
                prior = json.loads(done.read_text())
                if int(prior.get("updates", 0)) == updates:
                    rows.append(prior)
                    continue
            _load_trainable(rm, head, source_reward)
            row = _run_arm(seed, method, train, eval_prompts, eval_refs, anchor_prompts, [item["text"] for item in left], [item["text"] for item in right], rm, head, rm_tokenizer, out, max_new, updates, replays.get(seed))
            row["max_new_tokens"] = max_new
            row["label_source"] = "AI feedback" if method != "A_source" else "human source RM, no new labels"
            row["historical_wp07_B"] = "NOT A VALID BOUNDED CLAIM"
            (done).write_text(json.dumps(row, indent=2) + "\n")
            rows.append(row)
            if method == "B_bounded":
                replays[seed] = {event["step"]: event for event in row["events"]}
    if stage == "confirm":
        e_path = out / "E_teacher_seed0.json"
        if e_path.exists() and int(json.loads(e_path.read_text()).get("updates", 0)) >= 8:
            rows.append(json.loads(e_path.read_text()))
        else:
            _load_trainable(rm, head, source_reward)
            row = _run_arm(0, "E_teacher", train, eval_prompts, eval_refs, anchor_prompts, [""] * len(anchor_prompts), [""] * len(anchor_prompts), rm, head, rm_tokenizer, out, max_new, 16, None)
            row["max_new_tokens"] = max_new
            row["label_source"] = "frozen AI teacher, development diagnostic, not human"
            row["historical_wp07_B"] = "NOT A VALID BOUNDED CLAIM"
            e_path.write_text(json.dumps(row, indent=2) + "\n")
            rows.append(row)
        _load_trainable(rm, head, source_reward)
        holdout = _holdout_nll(rm, head, rm_tokenizer, out)
        for seed in seeds:
            policy, _tokenizer = _load_policy(seed)
            opt = torch.optim.AdamW((param for param in policy.parameters() if param.requires_grad), lr=5e-5)
            path = out / f"init_seed{seed}_step0.pt"
            _save_checkpoint(path, policy, opt, source_reward, 0, None)
            del policy, opt
            torch.cuda.empty_cache()
    else:
        holdout = None
    payload = {
        "status": "complete",
        "stage": stage,
        "rows": rows,
        "b": B,
        "epsilon": EPS,
        "max_new_tokens": max_new,
        "token_boundary": boundary,
        "source_rm": str(human),
        "runtime_seconds": time.perf_counter() - started,
        "source_sha": git_sha(),
        "passed": all(row.get("max_per_event_rms", 0.0) <= B + EPS or row["method"] in {"E_teacher", "A_source", "C_b0"} for row in rows),
        "guarantee_type": "empirical_proxy",
        "human_holdout": holdout,
        "D_unbounded": {"status": "NOT RUN", "reason": "The bounded arm did not persist candidate response texts or AI label bits, so an unbounded fork would not be the same pairs."},
        "checkpoint_gaps": {
            "live_step0_files": False,
            "step0_files": "init_seed*_step0.pt reconstructed from the seeded LoRA initialization and a fresh AdamW; this is not a live mid-run RNG snapshot",
            "numpy_rng": "not saved; the field is null",
            "generated_pair_texts": "not saved; anchor response hashes were saved",
            "C_label_replay": "C checked that deployed margins stayed at R0. It did not load B's response texts.",
        },
    }
    (out / f"{stage}.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def _selected_human() -> Path:
    path = ROOT / "experiments/horizon_final_08/human_final_holdout/models_confirm.json"
    if path.exists():
        selected = json.loads(path.read_text()).get("source_selection", {}).get("selected")
        if selected:
            return ROOT / "experiments/horizon_final_08/human_final_holdout" / selected["name"]
    return ROOT / "experiments/horizon_scale_07/human_tldr/human_256"
