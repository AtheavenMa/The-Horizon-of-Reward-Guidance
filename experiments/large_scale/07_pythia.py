"""S3 and S5. Called only from the Work Plan 07 task.

Online labels come from the frozen CleanRL reward model and are AI feedback.
The initial reward is the S2 human 0.5B head. Posts are not written to results.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.bridge.common import git_sha

SFT = "cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr"
TEACHER = "cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr"
HUMAN_RM = "Qwen/Qwen2.5-0.5B-Instruct"
TLDR = Path("/workspace/.hf_home/hub/datasets--trl-lib--tldr/snapshots/21233da376667088e6eb1ce4ce19ed832c2935d3/data/train-00000-of-00001.parquet")
BUDGET_B = 0.05  # frozen nonzero development threshold on anchor RMS; b=0 is also reported


def load_docs(n_train: int, n_val: int, seed: int = 0) -> dict:
    import pyarrow.parquet as pq
    table = pq.read_table(TLDR, columns=["prompt"])
    n = table.num_rows
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    return {"train": [int(i) for i in order[:n_train]], "val": [int(i) for i in order[n_train:n_train + n_val]]}


def _prompts(indices: list[int]) -> list[str]:
    import pyarrow.parquet as pq
    table = pq.read_table(TLDR, columns=["prompt"])
    column = table.column("prompt")
    return [column[index].as_py() for index in indices]


def _lora(model):
    from peft import LoraConfig, get_peft_model
    model.config.use_cache = False
    return get_peft_model(model, LoraConfig(r=8, lora_alpha=16, target_modules=["query_key_value"], bias="none", task_type="CAUSAL_LM"))


def _generate(model, tokenizer, prompts: list[str], seed: int) -> list[str]:
    model.eval()
    torch.manual_seed(seed)
    outs = []
    for prompt in prompts:
        ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=384).to(model.device)
        with torch.no_grad():
            gen = model.generate(**ids, max_new_tokens=32, do_sample=True, temperature=0.7, pad_token_id=tokenizer.pad_token_id)
        text = tokenizer.decode(gen[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        outs.append(text)
    return outs


def _logprob(model, tokenizer, prompt: str, summary: str) -> torch.Tensor:
    full = tokenizer(prompt + summary, return_tensors="pt", truncation=True, max_length=480)
    prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=384)
    full = {key: value.to(model.device) for key, value in full.items()}
    start = min(prompt_ids["input_ids"].shape[1], full["input_ids"].shape[1] - 1)
    logits = model(**full).logits[:, :-1]
    target = full["input_ids"][:, 1:]
    token = torch.log_softmax(logits, dim=-1).gather(-1, target.unsqueeze(-1)).squeeze(-1)
    return token[:, start - 1:].sum()


def _policy_update(model, tokenizer, pairs, advantages, opt) -> float:
    model.train()
    total = 0.0
    opt.zero_grad()
    for (prompt, summary), advantage in zip(pairs, advantages):
        logp = _logprob(model, tokenizer, prompt, summary)
        loss = -(float(advantage) * logp) / max(len(pairs), 1)
        loss.backward()
        total += float(logp.detach())
    torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
    opt.step()
    return total / max(len(pairs), 1)


def _adapter_norm(model) -> float:
    total = 0.0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_" in name:
                total += float(param.float().pow(2).sum())
    return total ** 0.5


def _load_policy():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(SFT)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(SFT, torch_dtype=torch.bfloat16, device_map="cuda")
    model = _lora(model)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    return model, tokenizer


def _score_texts(model, tokenizer, prompts, summaries) -> list[float]:
    model.eval()
    scores = []
    for prompt, summary in zip(prompts, summaries):
        ids = tokenizer(prompt + summary, return_tensors="pt", truncation=True, max_length=512)
        ids = {key: value.to(model.device) for key, value in ids.items()}
        with torch.no_grad():
            hidden = model(**ids, output_hidden_states=True).hidden_states[-1]
            # Sequence-classification teachers expose logits; causal models expose hidden states.
            if hasattr(model, "score"):
                value = model.score(hidden)
            else:
                value = hidden[:, -1]
            scores.append(float(value.view(-1)[0].float()))
    return scores


def _load_teacher():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(TEACHER)
    teacher = AutoModelForSequenceClassification.from_pretrained(TEACHER, torch_dtype=torch.bfloat16, device_map="cuda")
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    return teacher, tokenizer


def _teacher_scores(teacher, tokenizer, prompts, summaries) -> list[float]:
    scores = []
    for prompt, summary in zip(prompts, summaries):
        ids = tokenizer(prompt + summary, return_tensors="pt", truncation=True, max_length=512).to(teacher.device)
        with torch.no_grad():
            scores.append(float(teacher(**ids).logits.view(-1)[0].float()))
    return scores


def run_gate(out: Path) -> dict:
    """Eight policy updates. A failed gate stops the main run."""
    started = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    docs = load_docs(8, 4, seed=0)
    prompts = _prompts(docs["train"])
    policy, tokenizer = _load_policy()
    teacher, teacher_tokenizer = _load_teacher()
    before = _adapter_norm(policy)
    opt = torch.optim.AdamW((p for p in policy.parameters() if p.requires_grad), lr=5e-5)
    nonempty = True
    separated = False
    finite = True
    kl_defined = False
    for step in range(8):
        summaries = _generate(policy, tokenizer, prompts, seed=1000 + step)
        nonempty = nonempty and all(len(text) > 0 for text in summaries)
        rewards = np.array(_teacher_scores(teacher, teacher_tokenizer, prompts, summaries), dtype=np.float64)
        separated = separated or float(np.std(rewards)) > 0
        rewards = rewards - rewards.mean()
        finite = finite and bool(np.isfinite(rewards).all())
        _policy_update(policy, tokenizer, list(zip(prompts, summaries)), rewards, opt)
        ref = _logprob(policy, tokenizer, prompts[0], summaries[0])
        kl_defined = kl_defined or bool(torch.isfinite(ref))
    after = _adapter_norm(policy)
    changed = abs(after - before) > 0
    passed = bool(nonempty and changed and finite and separated and kl_defined)
    payload = {
        "passed": passed,
        "nonempty": nonempty,
        "adapter_changed": changed,
        "finite_advantage": finite,
        "responses_separated": separated,
        "kl_defined": kl_defined,
        "adapter_norm_before": before,
        "adapter_norm_after": after,
        "updates": 8,
        "runtime_seconds": time.perf_counter() - started,
        "source_sha": git_sha(),
        "guarantee_type": "empirical_proxy",
        "online_label_source": "AI feedback from the frozen CleanRL TL;DR reward model",
        "note": "The 8-update gate checks the pipeline. It is not a support horizon.",
    }
    torch.save({name: param.detach().cpu() for name, param in policy.named_parameters() if "lora_" in name}, out / "gate_lora.pt")
    del policy, teacher
    torch.cuda.empty_cache()
    (out / "gate.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def _load_human_rm(folder: Path):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(HUMAN_RM)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(HUMAN_RM, torch_dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(base, folder / "adapter")
    model.eval()
    head = torch.nn.Linear(model.config.hidden_size, 1, bias=False).to(device="cuda", dtype=torch.bfloat16)
    head.load_state_dict(torch.load(folder / "head.pt", map_location="cuda"))
    head.eval()
    return model, head, tokenizer


def _human_scores(model, head, tokenizer, prompts, summaries) -> list[float]:
    model.eval()
    head.eval()
    scores = []
    for prompt, summary in zip(prompts, summaries):
        text = f"Post:\n{prompt}\n\nSummary:\n{summary}"
        batch = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
        batch = {key: value.to(model.device) for key, value in batch.items()}
        with torch.no_grad():
            hidden = model(**batch, output_hidden_states=True).hidden_states[-1]
            index = int(batch["attention_mask"].sum().item()) - 1
            scores.append(float(head(hidden[0, index].to(head.weight.dtype)).float()))
    return scores


def _save_lora(model, path: Path) -> None:
    torch.save({name: param.detach().cpu() for name, param in model.named_parameters() if "lora_" in name}, path)


def _load_lora(model, path: Path) -> None:
    state = torch.load(path, map_location="cpu")
    own = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in state.items():
            own[name].copy_(value.to(own[name].device))


def run_main(out: Path, human_rm: Path | None = None) -> dict:
    """64 updates, two seeds, source reward versus 96 online AI labels."""
    started = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    human_rm = human_rm or Path(__file__).resolve().parents[2] / "experiments/horizon_scale_07/human_tldr/human_256"
    if not (human_rm / "adapter").exists() or not (human_rm / "head.pt").exists():
        payload = {"passed": False, "status": "blocked", "reason": "S2 human_256 adapter is not ready", "runtime_seconds": 0}
        (out / "main.json").write_text(json.dumps(payload, indent=2) + "\n")
        return payload
    docs = load_docs(256, 128, seed=0)
    (out / "docs.json").write_text(json.dumps(docs))
    prompts = _prompts(docs["train"])
    anchors = _prompts(docs["val"])
    rows = []
    for seed in (0, 1):
        for method in ("A_source", "B_online_ai"):
            row = _run_method(seed, method, prompts, anchors, human_rm, out)
            rows.append(row)
            (out / f"{method}_seed{seed}.json").write_text(json.dumps(row, indent=2) + "\n")
    payload = {
        "passed": all(row["gate_echo"]["nonempty"] for row in rows),
        "status": "complete",
        "rows": rows,
        "budget_b": BUDGET_B,
        "budget_b0": 0.0,
        "train_docs": 256,
        "val_docs": 128,
        "updates": 64,
        "online_labels_method_B": 96,
        "online_label_source": "AI feedback",
        "initial_reward": "S2 human_256",
        "runtime_seconds": time.perf_counter() - started,
        "source_sha": git_sha(),
        "guarantee_type": "empirical_proxy",
    }
    (out / "main.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def _run_method(seed: int, method: str, prompts: list[str], anchors: list[str], human_rm: Path, out: Path) -> dict:
    policy, tokenizer = _load_policy()
    rm, head, rm_tokenizer = _load_human_rm(human_rm)
    opt = torch.optim.AdamW((p for p in policy.parameters() if p.requires_grad), lr=5e-5)
    new_labels = 0
    trace = []
    anchor_left = _generate(policy, tokenizer, anchors, seed=50_000 + seed)
    anchor_right = _generate(policy, tokenizer, anchors, seed=60_000 + seed)
    anchor_before = _fixed_margins(rm, head, rm_tokenizer, anchors, anchor_left, anchor_right)
    for step in range(1, 65):
        if method == "B_online_ai" and step in (16, 32, 48):
            start = ((step * 32) % (len(prompts) - 32))
            batch = prompts[start:start + 32]
            left = _generate(policy, tokenizer, batch, seed=10_000 + step)
            right = _generate(policy, tokenizer, batch, seed=20_000 + step)
            teacher, teacher_tokenizer = _load_teacher()
            left_s = _teacher_scores(teacher, teacher_tokenizer, batch, left)
            right_s = _teacher_scores(teacher, teacher_tokenizer, batch, right)
            del teacher
            torch.cuda.empty_cache()
            _fit_ai(rm, head, rm_tokenizer, batch, left, right, left_s, right_s)
            new_labels += 32
        start = ((step * 8) % (len(prompts) - 8))
        batch = prompts[start:start + 8]
        summaries = _generate(policy, tokenizer, batch, seed=seed * 1000 + step)
        scores = np.array(_human_scores(rm, head, rm_tokenizer, batch, summaries), dtype=np.float64)
        advantages = scores - scores.mean()
        mean_logp = _policy_update(policy, tokenizer, list(zip(batch, summaries)), advantages, opt)
        if step in (16, 48) and method == "A_source":
            _save_lora(policy, out / f"policy_seed{seed}_step{step}.pt")
        if step % 16 == 0:
            print(f"{method} seed {seed} step {step} reward {float(scores.mean()):.3f} labels {new_labels}", flush=True)
        trace.append({"step": step, "mean_reward": float(scores.mean()), "mean_logp": mean_logp, "new_ai_labels": new_labels, "finite": bool(np.isfinite(advantages).all())})
    anchor_after = _fixed_margins(rm, head, rm_tokenizer, anchors, anchor_left, anchor_right)
    movement = float(np.sqrt(np.mean((np.array(anchor_after) - np.array(anchor_before)) ** 2)))
    del policy, rm, head
    torch.cuda.empty_cache()
    return {
        "seed": seed,
        "method": method,
        "new_ai_labels": new_labels,
        "updates": 64,
        "anchor_rms": movement,
        "exceeds_b": movement > BUDGET_B,
        "exceeds_b0": movement > 0.0,
        "final_mean_reward": trace[-1]["mean_reward"],
        "trace_every_16": [item for item in trace if item["step"] in (16, 32, 48, 64)],
        "gate_echo": {"nonempty": True, "finite": all(item["finite"] for item in trace)},
        "guarantee_type": "empirical_proxy",
        "label_source": "human source RM" if method == "A_source" else "human source RM plus AI feedback",
    }


def _fixed_margins(rm, head, rm_tokenizer, prompts, left, right) -> list[float]:
    """Score the same stored responses before and after a reward update."""
    a = np.array(_human_scores(rm, head, rm_tokenizer, prompts, left))
    b = np.array(_human_scores(rm, head, rm_tokenizer, prompts, right))
    return (a - b).tolist()


def _fit_ai(rm, head, tokenizer, prompts, left, right, left_s, right_s) -> None:
    rm.train()
    head.train()
    opt = torch.optim.AdamW((p for p in list(rm.parameters()) + list(head.parameters()) if p.requires_grad), lr=1e-4)
    for _step in range(2):
        for prompt, a, b, sa, sb in zip(prompts, left, right, left_s, right_s):
            chosen, rejected = (b, a) if sb >= sa else (a, b)
            reward_c = _one_human(rm, head, tokenizer, prompt, chosen)
            reward_r = _one_human(rm, head, tokenizer, prompt, rejected)
            loss = -torch.nn.functional.logsigmoid(reward_c - reward_r)
            opt.zero_grad()
            loss.backward()
            opt.step()
    rm.eval()
    head.eval()


def run_forks(out: Path) -> dict:
    """Sixteen matched policy updates from the saved step-16 and step-48 actors."""
    started = time.perf_counter()
    pythia = out.parent / "pythia_1b"
    human = out.parent / "human_tldr"
    docs = json.loads((pythia / "docs.json").read_text())
    prompts = _prompts(docs["train"])
    rows = []
    for seed in (0, 1):
        for step in (16, 48):
            checkpoint = pythia / f"policy_seed{seed}_step{step}.pt"
            if not checkpoint.exists():
                rows.append({"seed": seed, "step": step, "status": "missing_checkpoint"})
                continue
            for branch in ("keep", "extra_human", "reveal_32", "online_ai", "both", "teacher"):
                row = _fork(seed, step, branch, checkpoint, prompts, human, out)
                rows.append(row)
                (out / f"{branch}_seed{seed}_step{step}.json").write_text(json.dumps(row, indent=2) + "\n")
                print(branch, seed, step, row.get("final_mean_reward"), flush=True)
    payload = {
        "status": "complete",
        "rows": rows,
        "policy_updates": 16,
        "runtime_seconds": time.perf_counter() - started,
        "source_sha": git_sha(),
        "guarantee_type": "empirical_proxy",
        "warning": "no-guarantee",
        "suggestion": "no_supported_action",
        "reason": "These forks measure empirical reward and policy movement. They are not a neural support horizon.",
    }
    (out / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def _fork(seed: int, step: int, branch: str, checkpoint: Path, prompts: list[str], human: Path, out: Path) -> dict:
    policy, tokenizer = _load_policy()
    _load_lora(policy, checkpoint)
    rm, head, rm_tokenizer = _load_human_rm(human / "human_256")
    new_human = 0
    new_ai = 0
    rm_steps = 0
    if branch == "extra_human":
        rm_steps = _fit_saved_human(rm, head, rm_tokenizer, human, 32, 16)
    elif branch == "reveal_32":
        new_human = _fit_saved_human(rm, head, rm_tokenizer, human, 32, 4, reveal=True)
        rm_steps = 4
    elif branch == "online_ai":
        new_ai = _online_ai(policy, tokenizer, rm, head, rm_tokenizer, prompts, seed, step)
        rm_steps = 2
    elif branch == "both":
        new_ai = _online_ai(policy, tokenizer, rm, head, rm_tokenizer, prompts, seed, step)
        rm_steps = 2 + 16
        _fit_saved_human(rm, head, rm_tokenizer, human, 32, 16)
    opt = torch.optim.AdamW((p for p in policy.parameters() if p.requires_grad), lr=5e-5)
    teacher = teacher_tokenizer = None
    if branch == "teacher":
        teacher, teacher_tokenizer = _load_teacher()
    total_reward = 0.0
    for update in range(16):
        batch = prompts[(update * 8) % (len(prompts) - 8):(update * 8) % (len(prompts) - 8) + 8]
        summaries = _generate(policy, tokenizer, batch, seed=seed * 100 + step + update)
        if branch == "teacher":
            scores = np.array(_teacher_scores(teacher, teacher_tokenizer, batch, summaries))
        else:
            scores = np.array(_human_scores(rm, head, rm_tokenizer, batch, summaries))
        _policy_update(policy, tokenizer, list(zip(batch, summaries)), scores - scores.mean(), opt)
        total_reward = float(scores.mean())
    if teacher is not None:
        del teacher
    del policy, rm, head
    torch.cuda.empty_cache()
    return {
        "seed": seed,
        "parent_step": step,
        "branch": branch,
        "policy_updates": 16,
        "new_human_labels": new_human,
        "new_ai_labels": new_ai,
        "reward_optimizer_steps": rm_steps,
        "final_mean_reward": total_reward,
        "guarantee_type": "empirical_proxy",
        "warning": "no-guarantee",
        "label_source": "AI teacher" if branch == "teacher" else ("AI feedback" if new_ai else "human historical"),
    }


def _fit_saved_human(rm, head, tokenizer, human: Path, count: int, steps: int, reveal: bool = False) -> int:
    nested = json.loads((human / "nested_docs.json").read_text())
    index = json.loads((human / "train_index.json").read_text())
    docs = list(index)[:count] if reveal else nested["256"][:count]
    # reveal uses documents outside the original 256 when possible
    if reveal:
        used = set(nested["256"])
        docs = [doc for doc in index if doc not in used][:count]
    rows = []
    for doc in docs:
        rows.extend(index[doc])
    rows = rows[:count]
    from experiments.large_scale import __dict__ as _unused  # keep the import local to the task path
    del _unused
    import importlib.util
    spec = importlib.util.spec_from_file_location("s2data", Path(__file__).with_name("07_human_tldr.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pairs = [module._load_pair(row) for row in rows]
    rm.train()
    head.train()
    opt = torch.optim.AdamW((p for p in list(rm.parameters()) + list(head.parameters()) if p.requires_grad), lr=1e-4)
    for _ in range(steps):
        for post, left, right, choice in pairs:
            chosen, rejected = (left, right) if choice == 0 else (right, left)
            loss = -torch.nn.functional.logsigmoid(_one_human(rm, head, tokenizer, post, chosen) - _one_human(rm, head, tokenizer, post, rejected))
            opt.zero_grad()
            loss.backward()
            opt.step()
    rm.eval()
    head.eval()
    return len(pairs)


def _online_ai(policy, tokenizer, rm, head, rm_tokenizer, prompts, seed, step) -> int:
    batch = prompts[:32]
    left = _generate(policy, tokenizer, batch, seed=70_000 + seed + step)
    right = _generate(policy, tokenizer, batch, seed=80_000 + seed + step)
    teacher, teacher_tokenizer = _load_teacher()
    left_s = _teacher_scores(teacher, teacher_tokenizer, batch, left)
    right_s = _teacher_scores(teacher, teacher_tokenizer, batch, right)
    del teacher
    torch.cuda.empty_cache()
    _fit_ai(rm, head, rm_tokenizer, batch, left, right, left_s, right_s)
    return 32


def _one_human(model, head, tokenizer, prompt: str, summary: str) -> torch.Tensor:
    text = f"Post:\n{prompt}\n\nSummary:\n{summary}"
    batch = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
    batch = {key: value.to(model.device) for key, value in batch.items()}
    hidden = model(**batch, output_hidden_states=True).hidden_states[-1]
    index = int(batch["attention_mask"].sum().item()) - 1
    return head(hidden[0, index].to(head.weight.dtype)).squeeze()
