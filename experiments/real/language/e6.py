"""Verifiable instruction preferences with a hidden checker utility.

Training prompts are generated here. The public IFEval input file is not used.
The reward head sees surface features only, not the checker weights.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import torch

from experiments.real.common.records import run_dir, write_run

WORDS = ["amber", "cinder", "harbor", "lumen", "marble", "willow", "quartz", "nectar"]
TOPICS = ["a morning walk", "a small repair", "a library visit", "a late train", "a kitchen shelf", "a paper boat"]
WEIGHTS = np.array([0.4, 0.2, 0.2, 0.2], dtype=np.float64)


def build_split(n: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        word = WORDS[int(rng.integers(0, len(WORDS)))]
        topic = TOPICS[int(rng.integers(0, len(TOPICS)))]
        n_sent = int(rng.integers(2, 4))
        end = "." if rng.random() < 0.5 else "!"
        prompt = (
            f"Write about {topic}. Include the word {word}. "
            f"Use exactly {n_sent} sentences. End the answer with {end} "
            f"Include one integer."
        )
        rows.append({"id": f"{seed}-{i}", "prompt": prompt, "word": word, "n_sent": n_sent, "end": end})
    return rows


def checks(text: str, spec: dict) -> np.ndarray:
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    has_word = float(spec["word"].lower() in text.lower())
    sent_ok = float(len(sentences) == spec["n_sent"])
    end_ok = float(text.rstrip().endswith(spec["end"]))
    number_ok = float(bool(re.search(r"\d", text)))
    return np.array([has_word, sent_ok, end_ok, number_ok], dtype=np.float64)


def utility(text: str, spec: dict) -> float:
    return float(WEIGHTS @ checks(text, spec))


def surface(text: str) -> np.ndarray:
    chars = max(len(text), 1)
    words = text.split() or [""]
    return np.array([
        chars / 200,
        text.count(",") / 5,
        sum(ch.isdigit() for ch in text) / 3,
        sum(ch.isupper() for ch in text) / chars,
        len(set(w.lower() for w in words)) / len(words),
    ], dtype=np.float64)


def fit_linear(features: np.ndarray, labels: np.ndarray, steps: int) -> np.ndarray:
    w = np.zeros(features.shape[1])
    for _ in range(steps):
        score = features @ w
        prob = 1 / (1 + np.exp(-np.clip(score, -30, 30)))
        grad = features.T @ (prob - labels) / len(labels)
        step = -0.4 * grad
        norm = np.linalg.norm(step)
        if norm > 0.2:
            step *= 0.2 / norm
        w = w + step
    return w


METHODS = (
    "Fixed-feature reward baseline",
    "Neural reward baseline",
    "More reward updates, same data",
    "Extra feedback, same updates",
    "Policy-relevant selection",
)


def run_e6(stage: str) -> dict:
    # The planned 100/30/50 split is reduced here so one seed stays well under an hour.
    # Every method in a seed sees the same prompts, the same initial completions, and one policy step.
    n_train = 12 if stage == "pilot" else 24
    n_eval = 6 if stage == "pilot" else 8
    seeds = [1] if stage == "pilot" else [1, 2, 3]
    results = []
    for seed in seeds:
        print(f"E6 seed {seed}", flush=True)
        train = build_split(n_train, seed)
        held = build_split(n_eval, seed + 200)
        results.extend(_one_seed(train, held, seed))
    dest = run_dir("E6")
    (dest / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    write_run(
        dest,
        {"model": "Qwen/Qwen2.5-1.5B-Instruct", "teacher": "synthetic Bradley-Terry", "weights": WEIGHTS.tolist(), "stage": stage, "n_train": n_train, "n_eval": n_eval},
        results,
        [],
        {"status": "complete"},
    )
    return {"runs": results}


def _one_seed(train, held, seed: int) -> list[dict]:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cuda")
    base.config.use_cache = False
    base.gradient_checkpointing_enable()
    model = get_peft_model(
        base,
        LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, target_modules=["q_proj", "v_proj"], bias="none", task_type="CAUSAL_LM"),
    )
    rng = np.random.default_rng(seed)
    prompts = [row["prompt"] for row in train]
    texts_a = _generate(model, tokenizer, prompts, 0.8)
    texts_b = _generate(model, tokenizer, prompts, 0.9)
    feats, labels, events = [], [], []
    for spec, a, b in zip(train, texts_a, texts_b):
        gap = utility(b, spec) - utility(a, spec)
        prefer_b = float(rng.random() < 1 / (1 + np.exp(-4 * gap)))
        feats.append(surface(b) - surface(a))
        labels.append(prefer_b)
        events.append({"source": "synthetic Bradley-Terry", "gap": gap, "label": prefer_b, "id": spec["id"]})
    feat_m = np.stack(feats)
    lab = np.asarray(labels, dtype=np.float64)
    initial = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    rows = []
    for method in METHODS:
        model.load_state_dict(initial)
        chosen, steps, added = _method_data(method, feat_m, lab, train, texts_b, model, tokenizer, rng)
        reward_of = _fit_reward(method, chosen, lab if method != "Extra feedback, same updates" else None, steps, feat_m, lab, train, texts_a, texts_b, rng, model, tokenizer)
        before_lp = _mean_logprob(model, tokenizer, train[:4], texts_b[:4])
        _policy_update(model, tokenizer, train, texts_b, reward_of)
        after_lp = _mean_logprob(model, tokenizer, train[:4], texts_b[:4])
        texts_h = _generate(model, tokenizer, [row["prompt"] for row in held], 0.7)
        comp = np.stack([checks(text, spec) for text, spec in zip(texts_h, held)])
        util = float(np.mean(comp @ WEIGHTS))
        acc = _preference_accuracy(feat_m[-6:], lab[-6:], reward_of)
        row = {
            "seed": seed,
            "method": method,
            "initial_pairs": int(len(lab)),
            "added_pairs": added,
            "reward_steps": steps,
            "verified_utility": util,
            "pass_word": float(comp[:, 0].mean()),
            "pass_sentences": float(comp[:, 1].mean()),
            "pass_ending": float(comp[:, 2].mean()),
            "pass_number": float(comp[:, 3].mean()),
            "heldout_preference_accuracy": acc,
            "logprob_change": after_lp - before_lp,
            "response_length": float(np.mean([len(t.split()) for t in texts_h])),
        }
        print(row, flush=True)
        rows.append(row)
    out = run_dir("E6", f"seed{seed}")
    write_run(
        out,
        {"model": model_id, "seed": seed, "teacher": "synthetic Bradley-Terry on hidden checkers", "weights": WEIGHTS.tolist(), "n_train": len(train), "n_eval": len(held)},
        rows,
        events,
        {"status": "complete"},
    )
    del model, base
    torch.cuda.empty_cache()
    return rows


def _method_data(method, feat_m, lab, train, texts_b, model, tokenizer, rng):
    if method == "More reward updates, same data":
        return feat_m, 24, 0
    if method == "Policy-relevant selection":
        idx = _relevant_idx(feat_m, 16)
        return feat_m[idx], 8, 0
    if method == "Extra feedback, same updates":
        return feat_m, 8, 16
    return feat_m, 8, 0


def _fit_reward(method, chosen, _lab, steps, feat_m, lab, train, texts_a, texts_b, rng, model, tokenizer):
    if method == "Extra feedback, same updates":
        extra_specs = train[:16]
        extra_b = _generate(model, tokenizer, [row["prompt"] for row in extra_specs], 1.0)
        extra_feat, extra_lab = [], []
        for spec, a, b in zip(extra_specs, texts_a[:16], extra_b):
            gap = utility(b, spec) - utility(a, spec)
            extra_lab.append(float(rng.random() < 1 / (1 + np.exp(-4 * gap))))
            extra_feat.append(surface(b) - surface(a))
        chosen = np.concatenate([feat_m, np.stack(extra_feat)])
        labels = np.concatenate([lab, np.asarray(extra_lab)])
        return _as_reward(method, fit_linear(chosen, labels, steps))
    labels = lab if method != "Policy-relevant selection" else lab[_relevant_idx(feat_m, len(chosen))]
    if method == "Neural reward baseline":
        return _fit_neural(chosen, labels, steps)
    return _as_reward(method, fit_linear(chosen, labels, steps))


def _as_reward(method: str, weight: np.ndarray):
    def reward(text: str) -> float:
        return float(weight @ surface(text))
    reward.kind = "linear"
    reward.weight = weight
    return reward


def _fit_neural(features: np.ndarray, labels: np.ndarray, steps: int):
    torch.manual_seed(0)
    layer1 = torch.zeros(features.shape[1], 8, dtype=torch.float32)
    torch.nn.init.normal_(layer1, std=0.1)
    layer2 = torch.zeros(8, dtype=torch.float32)
    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)
    for _ in range(steps):
        hidden = torch.tanh(x @ layer1)
        score = hidden @ layer2
        prob = torch.sigmoid(score)
        delta = ((prob - y) / len(y)).detach()
        layer2 = layer2 - 0.2 * (hidden.T @ delta)
        layer1 = layer1 - 0.2 * (x.T @ (delta[:, None] * (1 - hidden**2) * layer2))
    w1 = layer1.detach().numpy()
    w2 = layer2.detach().numpy()

    def reward(text: str) -> float:
        hidden = np.tanh(surface(text) @ w1)
        return float(hidden @ w2)
    reward.kind = "neural"
    return reward


def _relevant_idx(feat_m: np.ndarray, k: int) -> np.ndarray:
    a = feat_m.mean(axis=0)
    gram = feat_m.T @ feat_m / max(len(feat_m), 1) + 1e-3 * np.eye(feat_m.shape[1])
    v_inv = np.linalg.pinv(gram)
    omega = 1.0
    scores = []
    for d in feat_m:
        vd = v_inv @ d
        scores.append(omega * float(a @ vd) ** 2 / (1 + omega * float(d @ vd)))
    return np.argsort(scores)[-min(k, len(feat_m)):]


def _preference_accuracy(features: np.ndarray, labels: np.ndarray, reward_of) -> float:
    if reward_of.kind != "linear":
        return float("nan")
    pred = (features @ reward_of.weight) > 0
    return float(np.mean(pred == (labels > 0.5)))


def _generate(model, tokenizer, prompts: list[str], temperature: float) -> list[str]:
    texts = []
    model.eval()
    for start in range(0, len(prompts), 4):
        chunk = prompts[start:start + 4]
        messages = [[{"role": "user", "content": prompt}] for prompt in chunk]
        encoded = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", padding=True, return_dict=True,
        )
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        with torch.no_grad():
            out = model.generate(
                **encoded, max_new_tokens=32, do_sample=True, temperature=temperature, pad_token_id=tokenizer.pad_token_id,
            )
        lengths = encoded["attention_mask"].sum(dim=1)
        for row, length in zip(out, lengths):
            texts.append(tokenizer.decode(row[int(length):], skip_special_tokens=True))
    return texts


def _prompt_ids(tokenizer, prompt: str) -> torch.Tensor:
    messages = [{"role": "user", "content": prompt}]
    encoded = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    if not torch.is_tensor(encoded):
        encoded = encoded["input_ids"]
    return encoded


def _mean_logprob(model, tokenizer, specs, texts) -> float:
    model.eval()
    vals = []
    for spec, text in zip(specs, texts):
        prompt_ids = _prompt_ids(tokenizer, spec["prompt"])
        full = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        # Response-only log probability, conditioned on the prompt.
        input_ids = torch.cat([prompt_ids, full["input_ids"][:, 1:]], dim=1).to(model.device)
        prompt_len = prompt_ids.shape[1]
        if input_ids.shape[1] <= prompt_len + 1:
            continue
        with torch.no_grad():
            logits = model(input_ids=input_ids).logits[:, :-1, :].float()
            logp = torch.log_softmax(logits, dim=-1)
            target = input_ids[:, 1:]
            token = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
            vals.append(float(token[:, prompt_len - 1:].mean().cpu()))
    return float(np.mean(vals)) if vals else 0.0


def _policy_update(model, tokenizer, specs, texts, reward_of) -> None:
    model.train()
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    rewards = [float(reward_of(text)) for text in texts]
    baseline = float(np.mean(rewards))
    opt.zero_grad()
    used = 0
    for spec, text, reward in zip(specs, texts, rewards):
        prompt_ids = _prompt_ids(tokenizer, spec["prompt"])
        full = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        input_ids = torch.cat([prompt_ids, full["input_ids"][:, 1:]], dim=1).to(model.device)
        prompt_len = prompt_ids.shape[1]
        if input_ids.shape[1] <= prompt_len + 1:
            continue
        out = model(input_ids=input_ids)
        logp = torch.log_softmax(out.logits[:, :-1, :].float(), dim=-1)
        target = input_ids[:, 1:]
        token = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        seq = token[:, max(prompt_len - 1, 0):].mean()
        loss = -(seq * (reward - baseline)) / max(len(specs), 1)
        loss.backward()
        used += 1
    if used:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval()
