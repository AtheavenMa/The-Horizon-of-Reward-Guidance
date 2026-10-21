"""F1 human reward evidence. The inspected 512 pairs stay development-only."""
from __future__ import annotations

import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np

from experiments.bridge.common import git_sha

ROOT = Path(__file__).resolve().parents[2]
SPLIT = ROOT / "experiments/horizon_final_08/audited_baselines/human_split.json"
HISTORY = ROOT / "experiments/horizon_scale_07/human_tldr"


def _legacy():
    spec = importlib.util.spec_from_file_location("human07", ROOT / "experiments/large_scale/07_human_tldr.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows(which: str) -> list[dict]:
    payload = json.loads(SPLIT.read_text())
    return payload["dev_ids" if which == "dev" else "final_ids"]


def _metrics(correct, nlls, briers, probs, labels, lengths, rewards, truncated, rows) -> dict:
    by_doc: dict[str, list[float]] = {}
    for record, value in zip(rows, correct):
        by_doc.setdefault(record["doc_id"], []).append(value)
    docs = list(by_doc)
    rng = np.random.default_rng(0)
    boots = []
    if docs:
        for _ in range(1000):
            draw = rng.choice(len(docs), size=len(docs), replace=True)
            vals = [item for index in draw for item in by_doc[docs[int(index)]]]
            boots.append(float(np.mean(vals)))
    corr = None
    if len(rewards) > 2 and np.std(rewards) > 0 and np.std(lengths) > 0:
        corr = float(np.corrcoef(rewards, lengths)[0, 1])
    probs_a = np.array(probs)
    labels_a = np.array(labels)
    ece = 0.0
    bins = []
    for low in np.linspace(0, 0.9, 10):
        high = low + 0.1
        mask = (probs_a >= low) & (probs_a < high if high < 1 else probs_a <= 1)
        if mask.any():
            gap = abs(float(probs_a[mask].mean()) - float(labels_a[mask].mean()))
            ece += gap * float(mask.mean())
            bins.append({"bin": float(low), "count": int(mask.sum()), "gap": gap})
    return {
        "heldout_n": len(rows),
        "accuracy": float(np.mean(correct)) if correct else None,
        "accuracy_ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))] if boots else None,
        "nll": float(np.mean(nlls)) if nlls else None,
        "brier": float(np.mean(briers)) if briers else None,
        "ece": float(ece),
        "calibration_bins": bins,
        "length_correlation": corr,
        "truncation_rate": truncated / max(2 * len(rows), 1),
        "documents": len(docs),
        "ci_unit": "document bootstrap, not independent seeds",
    }


def _score_logistic(weight, center, scale, records, legacy) -> dict:
    correct, nlls, briers, probs, labels, lengths, rewards = [], [], [], [], [], [], []
    for record in records:
        feat, choice = legacy._features(record)
        score = float(weight @ ((np.array(feat) - center) / scale))
        prob = 1 / (1 + np.exp(-np.clip(score, -20, 20)))
        correct.append(float((score > 0) == (choice == 1)))
        target = prob if choice == 1 else 1 - prob
        nlls.append(float(-np.log(max(target, 1e-8))))
        briers.append((prob - choice) ** 2)
        probs.append(prob)
        labels.append(choice)
        lengths.extend([feat[0], 0])
        rewards.extend([score, -score])
    return _metrics(correct, nlls, briers, probs, labels, lengths, rewards, 0, records)


def run_logistic(out: Path, stage: str = "confirm") -> dict:
    started = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    legacy = _legacy()
    nested = json.loads((HISTORY / "nested_docs.json").read_text())
    final_rows = _rows("final")
    dev_rows = _rows("dev")
    targets = [("logistic_257", nested["256"])] if stage == "pilot" else [("logistic_257", nested["256"]), ("logistic_4098", nested["4096"])]
    seeds = [0] if stage == "pilot" else [0, 1, 2]
    results = []
    for name, docs in targets:
        train = legacy._pairs_for(HISTORY, docs)
        if stage == "pilot":
            train = train[:32]
        raw, labels = [], []
        for record in train:
            feat, choice = legacy._features(record)
            raw.append(feat)
            labels.append(choice)
        raw_a = np.array(raw, dtype=np.float64)
        center, scale = raw_a.mean(0), raw_a.std(0) + 1e-6
        x = (raw_a - center) / scale
        y = np.array(labels, dtype=np.float64)
        for seed in seeds:
            weight = np.zeros(2)
            rng = np.random.default_rng(seed)
            order = rng.permutation(len(x))
            passes = 2 if stage == "pilot" else 20
            for _ in range(passes):
                for index in order:
                    score = float(weight @ x[index])
                    prob = 1 / (1 + np.exp(-np.clip(score, -20, 20)))
                    weight -= 0.05 * (prob - y[index]) * x[index]
            metrics = _score_logistic(weight, center, scale, final_rows if stage == "confirm" else dev_rows[:64], legacy)
            metrics.update({"name": name, "seed": seed, "train_pairs": len(train), "passes": passes, "label_source": "human"})
            results.append(metrics)
    payload = {
        "status": "complete",
        "stage": stage,
        "results": results,
        "equal_compute": "not a neural equal-compute contrast; this is the length/overlap baseline",
        "runtime_seconds": time.perf_counter() - started,
        "source_sha": git_sha(),
        "guarantee_type": "empirical_proxy",
    }
    (out / f"logistic_{stage}.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def _batched_eval(backbone, head, tokenizer, rows, device, legacy, temperature: float = 1.0):
    import torch
    backbone.eval()
    head.eval()
    correct, nlls, briers, probs, labels, lengths, rewards = [], [], [], [], [], [], []
    truncated = 0
    margins = []
    all_choices = []
    batch_size = 16
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            chunk = rows[start:start + batch_size]
            loaded = [legacy._load_pair(record) for record in chunk]
            texts, choices, lens = [], [], []
            for post, left, right, choice in loaded:
                texts.append(f"Post:\n{post}\n\nSummary:\n{left}")
                texts.append(f"Post:\n{post}\n\nSummary:\n{right}")
                choices.append(choice)
                lens.extend([len(left), len(right)])
            all_choices.extend(choices)
            batch = tokenizer(texts, truncation=True, max_length=512, padding=True, return_tensors="pt")
            truncated += int((batch["attention_mask"].sum(1) >= 512).sum())
            batch = {key: value.to(device) for key, value in batch.items()}
            hidden = backbone(**batch, output_hidden_states=True).hidden_states[-1]
            index = batch["attention_mask"].sum(1) - 1
            values = head(hidden[torch.arange(len(texts), device=device), index].to(head.weight.dtype)).float().view(-1, 2)
            for pair, choice, length_pair in zip(values, choices, zip(lens[0::2], lens[1::2])):
                margin = float((pair[1] - pair[0]) / temperature)
                prob = float(torch.sigmoid(torch.tensor(margin)))
                chosen_margin = margin if choice == 1 else -margin
                correct.append(1.0 if chosen_margin > 0 else 0.0)
                nlls.append(float(-torch.nn.functional.logsigmoid(torch.tensor(chosen_margin))))
                briers.append((prob - choice) ** 2)
                probs.append(prob)
                labels.append(choice)
                margins.append(float(pair[1] - pair[0]))
                rewards.extend([float(pair[0]), float(pair[1])])
                lengths.extend(length_pair)
    metrics = _metrics(correct, nlls, briers, probs, labels, lengths, rewards, truncated, rows)
    metrics["temperature"] = temperature
    return metrics, margins, all_choices


def _apply_temperature(margins, choices, rows, temperature: float) -> dict:
    import torch
    correct, nlls, briers, probs, labels = [], [], [], [], []
    for margin, choice, record in zip(margins, choices, rows):
        scaled = margin / temperature
        prob = float(torch.sigmoid(torch.tensor(scaled)))
        chosen_margin = scaled if choice == 1 else -scaled
        correct.append(1.0 if chosen_margin > 0 else 0.0)
        nlls.append(float(-torch.nn.functional.logsigmoid(torch.tensor(chosen_margin))))
        briers.append((prob - choice) ** 2)
        probs.append(prob)
        labels.append(choice)
    dummy = _metrics(correct, nlls, briers, probs, labels, [0, 0] * len(rows), [0, 0] * len(rows), 0, rows)
    return dummy


def _fit_temperature(margins, choices) -> float:
    import torch
    best_t, best_nll = 1.0, None
    for temperature in np.linspace(0.5, 2.0, 16):
        nlls = []
        for margin, choice in zip(margins, choices):
            chosen = (margin / float(temperature)) if choice == 1 else -(margin / float(temperature))
            nlls.append(float(-torch.nn.functional.logsigmoid(torch.tensor(chosen))))
        nll = float(np.mean(nlls))
        if best_nll is None or nll < best_nll:
            best_t, best_nll = float(temperature), nll
    return best_t


def _fit_one(backbone, head, tokenizer, rows, device, legacy, epochs: int, seed: int, train_lora: bool) -> int:
    import torch
    backbone.train()
    for name, param in backbone.named_parameters():
        param.requires_grad_("lora_" in name and train_lora)
    head.train()
    opt = torch.optim.AdamW([param for param in list(backbone.parameters()) + list(head.parameters()) if param.requires_grad], lr=1e-4)
    seen = 0
    for epoch in range(epochs):
        order = np.random.default_rng(seed * 1009 + epoch).permutation(len(rows))
        for index in order:
            post, left, right, choice = legacy._load_pair(rows[int(index)])
            texts = [f"Post:\n{post}\n\nSummary:\n{left}", f"Post:\n{post}\n\nSummary:\n{right}"]
            batch = tokenizer(texts, truncation=True, max_length=512, padding=True, return_tensors="pt")
            batch = {key: value.to(device) for key, value in batch.items()}
            hidden = backbone(**batch, output_hidden_states=True).hidden_states[-1]
            index_tok = batch["attention_mask"].sum(1) - 1
            values = head(hidden[torch.arange(2, device=device), index_tok].to(head.weight.dtype)).view(2)
            chosen = values[0] if choice == 0 else values[1]
            rejected = values[1] if choice == 0 else values[0]
            loss = -torch.nn.functional.logsigmoid(chosen - rejected)
            opt.zero_grad()
            loss.backward()
            opt.step()
            seen += 1
            if seen % 200 == 0:
                print(f"seed {seed} steps {seen} loss {float(loss.detach()):.4f}", flush=True)
    return seen


def _select(results: list[dict]) -> dict:
    ranked = [row for row in results if row.get("split") == "dev" and row.get("train_lora", True) and not str(row.get("name", "")).startswith("extra")]
    ranked.sort(key=lambda row: (-float(row["accuracy"]), -int(row["train_pairs"]), int(row["seed"])))
    return {"rule": "highest development accuracy, then more training pairs, then lower seed", "selected": {key: ranked[0][key] for key in ("name", "seed", "accuracy", "train_pairs")}} if ranked else {"selected": None}


def run_models(out: Path, stage: str = "confirm") -> dict:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    if os.environ.get("HORIZON_GPU") != "1" or not torch.cuda.is_available():
        payload = {"status": "waiting_for_gpu", "passed": False, "runtime_seconds": 0, "reason": "HORIZON_GPU=1 is required and the GPU was not released for this process"}
        (out / f"models_{stage}.json").write_text(json.dumps(payload, indent=2) + "\n")
        return payload
    legacy = _legacy()
    nested = json.loads((HISTORY / "nested_docs.json").read_text())
    dev_rows, final_rows = _rows("dev"), _rows("final")
    if stage == "pilot":
        jobs = [("pilot_257", nested["256"][:8], 1, 0)]
        dev_rows, final_rows = dev_rows[:32], final_rows[:32]
    else:
        jobs = []
        for seed in (0, 1, 2):
            jobs.append((f"human_257_seed{seed}", nested["256"], 4, seed))
            jobs.append((f"human_1024_seed{seed}", nested["1024"], 2, seed))
            jobs.append((f"human_4098_seed{seed}", nested["4096"], 1, seed))
            jobs.append((f"extra_257_seed{seed}", nested["256"], 16, seed))
    tokenizer = AutoTokenizer.from_pretrained(legacy.MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    backbone = AutoModelForCausalLM.from_pretrained(legacy.MODEL_ID, torch_dtype=torch.bfloat16)
    backbone.config.use_cache = False
    backbone = get_peft_model(backbone, LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    head = torch.nn.Linear(backbone.config.hidden_size, 1, bias=False).to(dtype=torch.bfloat16)
    device = torch.device("cuda")
    backbone.to(device)
    head.to(device)
    init_path = HISTORY / "init" / "reward_init.pt"
    if init_path.exists():
        saved = torch.load(init_path, map_location="cpu")
        legacy._restore(backbone, head, saved["lora"], saved["head"])
    init_lora = {name: param.detach().cpu().clone() for name, param in backbone.named_parameters() if param.requires_grad}
    init_head = {name: param.detach().cpu().clone() for name, param in head.state_dict().items()}
    results = []
    for name, docs, epochs, seed in jobs:
        folder = out / name
        done = folder / "metrics.json"
        if done.exists() and stage == "confirm":
            results.append(json.loads(done.read_text()))
            continue
        legacy._restore(backbone, head, init_lora, init_head)
        torch.manual_seed(seed)
        rows = legacy._pairs_for(HISTORY, docs)
        t0 = time.perf_counter()
        steps = _fit_one(backbone, head, tokenizer, rows, device, legacy, epochs, seed, True)
        dev_metrics, dev_margins, dev_choices = _batched_eval(backbone, head, tokenizer, dev_rows, device, legacy, 1.0)
        best_t = _fit_temperature(dev_margins, dev_choices)
        final_metrics, final_margins, final_choices = _batched_eval(backbone, head, tokenizer, final_rows, device, legacy, 1.0)
        calibrated = _apply_temperature(final_margins, final_choices, final_rows, best_t)
        final_metrics["nll"] = calibrated["nll"]
        final_metrics["brier"] = calibrated["brier"]
        final_metrics["ece"] = calibrated["ece"]
        final_metrics["accuracy_ci95"] = calibrated["accuracy_ci95"]
        final_metrics["temperature"] = best_t
        folder.mkdir(parents=True, exist_ok=True)
        torch.save(head.state_dict(), folder / "head.pt")
        backbone.save_pretrained(folder / "adapter")
        metrics = {
            "name": name,
            "seed": seed,
            "train_pairs": len(rows),
            "epochs": epochs,
            "optimizer_steps": steps,
            "gpu_seconds": time.perf_counter() - t0,
            "temperature_frozen_on_dev": best_t,
            "dev": dev_metrics,
            "final": final_metrics,
            "accuracy": dev_metrics["accuracy"],
            "split": "dev",
            "train_lora": True,
            "label_source": "human",
            "equal_compute_note": "extra_257 uses 16 epochs so its optimizer steps match one pass over 4098 pairs. The examples are not the same, so this is a step-normalized contrast.",
            "guarantee_type": "empirical_proxy",
        }
        (folder / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        results.append(metrics)
        print(name, "dev", dev_metrics["accuracy"], "final", final_metrics["accuracy"], flush=True)
    payload = {
        "status": "complete",
        "stage": stage,
        "results": results,
        "source_selection": _select(results) if stage == "confirm" else {"selected": None, "reason": "pilot is not a selection run"},
        "historical_wp07_dev512": "see experiments/horizon_scale_07/human_tldr/summary.json; not confirmatory",
        "runtime_seconds": time.perf_counter() - started,
        "source_sha": git_sha(),
        "guarantee_type": "empirical_proxy",
        "passed": True,
    }
    (out / f"models_{stage}.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload
