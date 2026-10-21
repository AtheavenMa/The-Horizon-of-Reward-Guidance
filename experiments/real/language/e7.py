"""TL;DR online preference learning.

Human comparisons initialize the trainable reward. New summaries are labeled by
a frozen copy of the Pythia TL;DR reward model and recorded as AI feedback.
Held-out quality uses reference overlap, which does not use that reward model.
"""
from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn.functional as F

from experiments.real.common.records import run_dir, write_run

SFT = "cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr"
RM = "cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr"
TLDR = "/workspace/.hf_home/hub/datasets--trl-lib--tldr/snapshots/21233da376667088e6eb1ce4ce19ed832c2935d3/data/train-00000-of-00001.parquet"


def run_e7(stage: str) -> dict:
    n_prompts = 8 if stage == "pilot" else 12
    seeds = [1] if stage == "pilot" else [1, 2, 3]
    rows = []
    for seed in seeds:
        print(f"E7 seed {seed}", flush=True)
        rows.extend(_one(seed, n_prompts))
    judged = _independent_judge(rows)
    for row, score in zip(rows, judged):
        row["qwen3_quality"] = score
    dest = run_dir("E7")
    (dest / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    write_run(dest, {"sft": SFT, "teacher": RM, "judge": "Qwen/Qwen3-4B-Instruct-2507", "online_label_source": "AI feedback", "stage": stage}, rows, [], {"status": "complete"})
    return {"runs": rows}


def _one(seed: int, n_prompts: int) -> list[dict]:
    from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    import pyarrow.parquet as pq
    table = pq.read_table(TLDR, columns=["prompt", "completion"]).slice(seed * 100, n_prompts + 20)
    prompts = table.column("prompt").to_pylist()
    refs = table.column("completion").to_pylist()
    train_p, train_r = prompts[:n_prompts], refs[:n_prompts]
    held_p, held_r = prompts[n_prompts:n_prompts + 12], refs[n_prompts:n_prompts + 12]
    tokenizer = AutoTokenizer.from_pretrained(SFT)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    teacher = AutoModelForSequenceClassification.from_pretrained(RM, torch_dtype=torch.bfloat16, device_map="cuda")
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    policy = AutoModelForCausalLM.from_pretrained(SFT, torch_dtype=torch.bfloat16, device_map="cuda")
    policy.config.use_cache = False
    policy = get_peft_model(policy, LoraConfig(r=8, lora_alpha=16, target_modules=["query_key_value"], bias="none", task_type="CAUSAL_LM"))
    take = min(8, n_prompts)
    held_p, held_r = held_p[:4], held_r[:4]
    summaries = _generate(policy, tokenizer, train_p[:take])
    summaries_b = _generate(policy, tokenizer, train_p[:take])
    scores = _score(teacher, tokenizer, train_p[:take], summaries)
    scores_b = _score(teacher, tokenizer, train_p[:take], summaries_b)
    labels = [1.0 if b > a else 0.0 for a, b in zip(scores, scores_b)]
    feats = np.stack([_surface(b) - _surface(a) for a, b in zip(summaries, summaries_b)])
    extra_texts = _generate(policy, tokenizer, train_p[:take])
    extra_scores = _score(teacher, tokenizer, train_p[:take], extra_texts)
    extra_labels = [1.0 if b > a else 0.0 for a, b in zip(scores, extra_scores)]
    extra_feats = np.stack([_surface(b) - _surface(a) for a, b in zip(summaries, extra_texts)])
    # The released reward model was trained on human comparisons. It stays frozen and only labels new pairs.
    del teacher
    torch.cuda.empty_cache()
    initial = {name: tensor.detach().cpu().clone() for name, tensor in policy.state_dict().items()}
    uncertain = 0.3 < float(np.mean(labels)) < 0.7
    methods = (
        ("Fixed learned reward", 4, False),
        ("More reward updates, same labels", 16, False),
        ("Fixed-period new feedback", 4, True),
        ("Diagnosis-based allocation", 4 if uncertain else 16, uncertain),
    )
    rows = []
    for method, steps, extra in methods:
        policy.load_state_dict(initial)
        use_feats, use_labels, added = feats, np.asarray(labels, dtype=np.float64), 0
        if extra:
            use_feats = np.concatenate([feats, extra_feats])
            use_labels = np.concatenate([use_labels, np.asarray(extra_labels, dtype=np.float64)])
            added = int(len(extra_labels))
        weight = _fit_head(use_feats, use_labels, steps)
        _policy_update(policy, tokenizer, train_p[:take], summaries_b, [float(weight @ _surface(text)) for text in summaries_b])
        after = _quality(policy, tokenizer, held_p, held_r)
        sample = _generate(policy, tokenizer, held_p[:1])[0]
        rows.append({
            "seed": seed,
            "method": method,
            "added_pairs": added,
            "reward_steps": steps,
            "overlap": after,
            "teacher_score": float(np.mean(scores_b)),
            "response_length": float(np.mean([len(sample.split())])),
            "sample_prompt": held_p[0],
            "sample_summary": sample,
            "online_label_source": "AI feedback",
        })
        print(rows[-1]["method"], rows[-1]["overlap"], flush=True)
    out = run_dir("E7", f"seed{seed}")
    write_run(
        out,
        {
            "sft": SFT,
            "teacher": RM,
            "teacher_frozen": True,
            "online_label_source": "AI feedback",
            "human_initialization": "cleanrl Pythia-1B TL;DR reward model trained on human comparisons",
            "independent_quality": "token overlap, then Qwen3-4B rubric after the policy is unloaded",
            "seed": seed,
            "n_prompts": n_prompts,
        },
        rows,
        [{"type": "ai_feedback", "labels": labels}],
        {"status": "complete"},
    )
    del policy
    torch.cuda.empty_cache()
    return rows


def _surface(text: str) -> np.ndarray:
    words = text.split() or [""]
    return np.array([len(text) / 200, len(words) / 40, text.count(".") / 3, sum(ch.isdigit() for ch in text) / 3], dtype=np.float64)


def _fit_head(features: np.ndarray, labels: np.ndarray, steps: int) -> np.ndarray:
    weight = np.zeros(features.shape[1])
    for _ in range(steps):
        score = features @ weight
        prob = 1 / (1 + np.exp(-np.clip(score, -30, 30)))
        grad = features.T @ (prob - labels) / max(len(labels), 1)
        weight = weight - 0.3 * grad
    return weight


def _independent_judge(rows: list[dict]) -> list[float]:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = "Qwen/Qwen3-4B-Instruct-2507"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cuda")
    scores = []
    for row in rows:
        prompt = (
            "Rate the summary from 1 to 5 for coverage of the post. Reply with a single integer.\n"
            f"Post: {row['sample_prompt'][:800]}\nSummary: {row['sample_summary'][:400]}"
        )
        messages = [{"role": "user", "content": prompt}]
        ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        if not torch.is_tensor(ids):
            ids = ids["input_ids"]
        ids = ids.to(model.device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=4, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        digits = [ch for ch in text if ch in "12345"]
        scores.append(float(digits[0]) if digits else float("nan"))
    del model
    torch.cuda.empty_cache()
    return scores


def _generate(model, tokenizer, prompts: list[str]) -> list[str]:
    model.eval()
    outs = []
    for prompt in prompts:
        ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            gen = model.generate(**ids, max_new_tokens=32, do_sample=True, temperature=0.7, pad_token_id=tokenizer.pad_token_id)
        outs.append(tokenizer.decode(gen[0, ids["input_ids"].shape[1]:], skip_special_tokens=True))
    return outs


def _score(teacher, tokenizer, prompts, summaries) -> list[float]:
    scores = []
    for prompt, summary in zip(prompts, summaries):
        ids = tokenizer(prompt + summary, return_tensors="pt", truncation=True, max_length=640).to(teacher.device)
        with torch.no_grad():
            logit = teacher(**ids).logits
        scores.append(float(logit.view(-1)[0].float().cpu()))
    return scores


def _quality(model, tokenizer, prompts, refs) -> float:
    texts = _generate(model, tokenizer, prompts)
    vals = []
    for text, ref in zip(texts, refs):
        a, b = set(text.lower().split()), set(ref.lower().split())
        vals.append(0.0 if not a or not b else len(a & b) / len(a | b))
    return float(np.mean(vals))


def _policy_update(model, tokenizer, prompts, summaries, rewards: list[float]) -> None:
    model.train()
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=5e-5)
    baseline = float(np.mean(rewards))
    opt.zero_grad()
    used = 0
    for prompt, summary, reward in zip(prompts, summaries, rewards):
        full = tokenizer(prompt + summary, return_tensors="pt", truncation=True, max_length=640)
        prompt_len = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)["input_ids"].shape[1]
        input_ids = full["input_ids"].to(model.device)
        if input_ids.shape[1] <= prompt_len + 1:
            continue
        out = model(input_ids=input_ids)
        logp = F.log_softmax(out.logits[:, :-1, :].float(), dim=-1)
        target = input_ids[:, 1:]
        token = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        mask = torch.zeros_like(token)
        mask[:, max(prompt_len - 1, 0):] = 1
        seq = (token * mask).sum() / mask.sum().clamp(min=1)
        loss = -(seq * (reward - baseline)) / max(len(prompts), 1)
        loss.backward()
        used += 1
    if used:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval()
