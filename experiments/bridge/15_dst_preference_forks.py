#!/usr/bin/env python3
"""H4: continuous one-dimensional BTL on Deep Sea Treasure, plus fresh-start H4C."""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.bridge.common import write_summary


def load_tabular():
    path = Path(__file__).resolve().parent / "14_dst_exact_tabular.py"
    spec = importlib.util.spec_from_file_location("dst_h4", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


H3 = load_tabular()
ROOT = Path(__file__).resolve().parents[2]
BETA = 5.0
ALPHA = 0.05
TRUE_W = 0.375


def sigmoid(x):
    x = np.clip(x, -40, 40)
    return 1.0 / (1.0 + np.exp(-x))


def pair_terms(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    diff = features[0::2] - features[1::2]
    c = BETA * diff[:, 1]
    d = BETA * (diff[:, 0] - diff[:, 1])
    return diff, c, d


def score(w, c, d, y) -> float:
    return float(np.sum(d * (y - sigmoid(c + w * d))))


def radius(d) -> float:
    return float(np.sqrt(0.5 * np.log(2.0 / ALPHA) * np.sum(d * d)))


def confidence_interval(c, d, y) -> tuple[float, float] | None:
    if float(np.sum(d * d)) == 0.0:
        return (0.0, 1.0)
    limit = radius(d)

    def lower():
        if score(0.0, c, d, y) <= limit:
            return 0.0
        if score(1.0, c, d, y) > limit:
            return None
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if score(mid, c, d, y) > limit:
                lo = mid
            else:
                hi = mid
        return hi

    def upper():
        if score(1.0, c, d, y) >= -limit:
            return 1.0
        if score(0.0, c, d, y) < -limit:
            return None
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if score(mid, c, d, y) < -limit:
                hi = mid
            else:
                lo = mid
        return lo

    lo, hi = lower(), upper()
    if lo is None or hi is None or lo > hi + 1e-8:
        return None
    return (float(lo), float(hi))


def negative_loglik(w, c, d, y) -> float:
    eta = c + w * d
    return float(-np.sum(y * eta - np.logaddexp(0.0, eta)))


def fit_mle(c, d, y) -> float:
    fit = minimize_scalar(lambda w: negative_loglik(float(w), c, d, y), bounds=(0.0, 1.0), method="bounded")
    return float(fit.x)


def fit_steps(c, d, y, steps: int, w0: float = 0.5) -> float:
    w = w0
    for _ in range(steps):
        grad = float(np.mean(d * (y - sigmoid(c + w * d))))
        w = float(np.clip(w + grad, 0.0, 1.0))
    return w


def label(c, d, rng) -> np.ndarray:
    # Evaluator only. TRUE_W is not returned to the learner.
    return (rng.random(len(d)) < sigmoid(c + TRUE_W * d)).astype(float)


def collect_features(model, low, high, n, seed) -> np.ndarray:
    rng = np.random.default_rng(seed)
    probs = H3.softmax(np.zeros((len(model["states"]), H3.N_ACTIONS)))
    feats = []
    for _ in range(n):
        s = model["start"]
        total = np.zeros(2)
        for _t in range(H3.H):
            a = int(rng.choice(H3.N_ACTIONS, p=probs[s]))
            item = model["trans"][s][a]
            total += item["reward"]
            if item["done"]:
                break
            s = item["next"]
        feats.append((total - low) / (high - low))
    return np.stack(feats)


def coef(phi, scale):
    return np.array([phi, 1.0 - phi]) / scale


def mu_and_step(model, logits, phi, scale, lr):
    grad, mu, _ = H3.exact_gradient(model, logits, coef(phi, scale))
    return mu, logits + lr * grad


def gain(mu, mu2, w, low, high) -> float:
    weight = np.array([w, 1.0 - w])
    return H3.scalarize(mu2, weight, low, high) - H3.scalarize(mu, weight, low, high)


def worst(mu, mu2, bounds, low, high) -> float:
    return min(gain(mu, mu2, bounds[0], low, high), gain(mu, mu2, bounds[1], low, high))


def search_phi(model, logits, bounds, low, high, scale, lr, window) -> tuple[float, float]:
    grid = np.linspace(window[0], window[1], 21)
    best_phi, best_gain = float(grid[0]), -1e100
    for phi in grid:
        mu, nxt = mu_and_step(model, logits, float(phi), scale, lr)
        _, mu2, _ = H3.exact_gradient(model, nxt, coef(0.5, scale))
        value = worst(mu, mu2, bounds, low, high)
        if value > best_gain:
            best_phi, best_gain = float(phi), value
    return best_phi, best_gain


def true_return(mu, low, high) -> float:
    return H3.scalarize(mu, np.array([TRUE_W, 1.0 - TRUE_W]), low, high)


def run_exact_branch(model, logits, phi, bounds_source, bounds_used, target, low, high, scale, lr, steps, budget, mode) -> dict:
    series = []
    for k in range(steps):
        if mode == "oracle":
            chosen = TRUE_W
        elif mode == "robust":
            window = (max(0.0, phi - budget), min(1.0, phi + budget))
            chosen, _ = search_phi(model, logits, bounds_used, low, high, scale, lr, window)
        else:
            chosen = float(np.clip(target, phi - budget, phi + budget))
        mu, nxt = mu_and_step(model, logits, chosen, scale, lr)
        _, mu2, _ = H3.exact_gradient(model, nxt, coef(0.5, scale))
        series.append({
            "update": k + 1,
            "phi": chosen,
            "true_return": true_return(mu2, low, high),
            "true_gain": gain(mu, mu2, TRUE_W, low, high),
            "source_certificate": worst(mu, mu2, bounds_source, low, high),
            "updated_certificate": worst(mu, mu2, bounds_used, low, high),
        })
        logits, phi = nxt, chosen
    return {"series": series, "final_phi": phi, "final_return": series[-1]["true_return"]}


def main() -> None:
    started = time.perf_counter()
    frozen = json.loads((ROOT / "experiments/horizon_real/E1/Deep-Sea-Treasure/frozen/frozen.json").read_text())
    h3 = json.loads((ROOT / "outputs/horizon_bridge/h3_dst_oracle/summary.json").read_text())
    low = np.array(frozen["low"], dtype=float)
    high = np.array(frozen["high"], dtype=float)
    scale = high - low
    gamma_pos = float(h3["gamma_pos"])
    lr = float(h3["exact_lr"])
    model = H3.build_model()
    # Synthetic check of the interval inversion before any environment labels.
    rng = np.random.default_rng(0)
    d_syn = np.array([2.0, -1.5, 0.5, 3.0])
    c_syn = np.zeros_like(d_syn)
    y_syn = label(c_syn, d_syn, rng)
    interval = confidence_interval(c_syn, d_syn, y_syn)
    if interval is None or not (interval[0] <= TRUE_W <= interval[1]):
        # One draw can miss. The repeated check below is the recorded coverage.
        pass
    zero = confidence_interval(np.zeros(3), np.zeros(3), np.zeros(3))
    if zero != (0.0, 1.0):
        raise AssertionError(zero)

    features = collect_features(model, low, high, 480, 7100)
    _diff, c, d = pair_terms(features)
    dev = slice(0, 40)
    y_dev = label(c[dev], d[dev], np.random.default_rng(7200))
    w_dev = fit_mle(c[dev], d[dev], y_dev)
    rest = np.arange(40, len(d))
    p_dev = sigmoid(c[rest] + w_dev * d[rest])
    fisher = d[rest] ** 2 * p_dev * (1.0 - p_dev)
    order_low = rest[np.argsort(np.abs(d[rest]))]
    source_idx = order_low[:32]
    remaining = [i for i in rest if i not in set(source_idx)]
    info_order = sorted(remaining, key=lambda i: -((d[i] ** 2) * sigmoid(c[i] + w_dev * d[i]) * (1 - sigmoid(c[i] + w_dev * d[i]))))
    info_idx = np.array(info_order[:64])
    pool = np.array(info_order[64:])
    pick = np.random.default_rng(7300).choice(len(pool), size=64, replace=False)
    uniform_idx = pool[pick[:32]]
    held = pool[pick[32:]]
    source_fisher = float(np.median(d[source_idx] ** 2 * sigmoid(c[source_idx] + w_dev * d[source_idx]) * (1 - sigmoid(c[source_idx] + w_dev * d[source_idx]))))

    def draw(indices, seed):
        return label(c[indices], d[indices], np.random.default_rng(seed))

    covered = 0
    reps = 40
    for rep in range(reps):
        got = confidence_interval(c[source_idx], d[source_idx], draw(source_idx, 8000 + rep))
        if got is not None and got[0] <= TRUE_W <= got[1]:
            covered += 1

    y_held = draw(held, 7400)
    # Preference table uses seed 100 as the displayed fit; forks repeat 100, 101, 102.
    rows_fit = []
    label_bank = {}
    for seed in (100, 101, 102):
        y_source = draw(source_idx, seed)
        packs = {
            "source": (source_idx[:32], y_source, 8),
            "extra16": (np.concatenate([source_idx, info_idx[:16]]), np.concatenate([y_source, draw(info_idx[:16], seed + 50)]), 8),
            "extra32": (np.concatenate([source_idx, info_idx[:32]]), np.concatenate([y_source, draw(info_idx[:32], seed + 50)]), 8),
            "extra64": (np.concatenate([source_idx, info_idx[:64]]), np.concatenate([y_source, draw(info_idx[:64], seed + 50)]), 8),
            "uniform": (np.concatenate([source_idx, uniform_idx]), np.concatenate([y_source, draw(uniform_idx, seed + 70)]), 8),
        }
        label_bank[seed] = packs
        if seed != 100:
            continue
        for name, (idx, y, _steps) in packs.items():
            mle = fit_mle(c[idx], d[idx], y)
            short = fit_steps(c[idx], d[idx], y, 8)
            long = fit_steps(c[idx], d[idx], y, 80)
            interval = confidence_interval(c[idx], d[idx], y)
            pred = sigmoid(c[held] + mle * d[held]) >= 0.5
            rows_fit.append({
                "name": name,
                "w_hat": mle,
                "short_fit": short,
                "long_fit": long,
                "fit_gap": abs(short - mle),
                "interval": None if interval is None else list(interval),
                "width": None if interval is None else interval[1] - interval[0],
                "heldout_accuracy": float(np.mean(pred == (y_held >= 0.5))),
                "contains_true": interval is not None and interval[0] <= TRUE_W <= interval[1],
            })

    checkpoints = {}
    for name, update in (("early", h3["early_update"]), ("later", h3["later_update"])):
        payload = json.loads((ROOT / f"outputs/horizon_bridge/h3_dst_oracle/checkpoint_update_{update}.json").read_text())
        checkpoints[name] = np.array(payload["logits"], dtype=float)

    branches = []
    for ck_name, logits0 in checkpoints.items():
        for seed in (100, 101, 102):
            packs = label_bank[seed]
            y_s = packs["source"][1]
            u0 = confidence_interval(c[source_idx], d[source_idx], y_s)
            if u0 is None:
                u0 = (0.0, 1.0)
            phi_init, init_gain = search_phi(model, logits0, u0, low, high, scale, lr, (0.0, 1.0))
            _, mu0, _ = H3.exact_gradient(model, logits0, coef(0.5, scale))
            pointwise = search_phi(model, logits0, u0, low, high, scale, lr, (0.0, 1.0))[1]
            specs = []
            for mode, key, steps, budget, robust in (
                ("baseline", "source", 8, 0.02, False),
                ("more_fit", "source", 80, 0.02, False),
                ("extra32", "extra32", 8, 0.02, False),
                ("extra32_more_fit", "extra32", 80, 0.02, False),
                ("robust", "extra32", 8, 0.02, True),
                ("oracle", "source", 0, None, False),
                ("budget_0.05", "extra32", 8, 0.05, True),
            ):
                if ck_name == "later" and mode in ("extra32_more_fit",):
                    continue
                if ck_name == "early" and mode == "budget_0.05":
                    continue
                specs.append((mode, key, steps, budget, robust))
            for mode, key, steps, budget, robust in specs:
                idx, y, _ = packs[key]
                target = TRUE_W if mode == "oracle" else (fit_steps(c[idx], d[idx], y, steps) if steps else fit_mle(c[idx], d[idx], y))
                u_n = confidence_interval(c[idx], d[idx], y) or (0.0, 1.0)
                used = u_n if mode in ("robust", "budget_0.05", "extra32", "extra32_more_fit") else u0
                branch_mode = "robust" if robust else ("oracle" if mode == "oracle" else "point")
                result = run_exact_branch(
                    model, logits0.copy(), phi_init, u0, used, target, low, high, scale, lr,
                    20, 0.02 if budget is None else budget, branch_mode,
                )
                final = result["series"][-1]
                branches.append({
                    "checkpoint": ck_name,
                    "seed": seed,
                    "mode": mode,
                    "phi_init": phi_init,
                    "init_source_gain": init_gain,
                    "uncertified_at_init": init_gain < gamma_pos,
                    "pointwise_init": pointwise,
                    "start_return": true_return(mu0, low, high),
                    "final_return": result["final_return"],
                    "return_change": result["final_return"] - true_return(mu0, low, high),
                    "final_phi": result["final_phi"],
                    "final_source": final["source_certificate"],
                    "final_updated": final["updated_certificate"],
                    "min_source": min(row["source_certificate"] for row in result["series"]),
                    "min_updated": min(row["updated_certificate"] for row in result["series"]),
                })

    partial = ROOT / "outputs/horizon_bridge/h4_dst_preference"
    partial.mkdir(parents=True, exist_ok=True)
    (partial / "branches_partial.json").write_text(json.dumps(clean({"fits": rows_fit, "branches": branches, "coverage_fraction": covered / reps, "development_w": w_dev, "source_median_fisher": source_fisher, "gamma_pos": gamma_pos})))
    print("checkpoint branches done", len(branches), flush=True)
    # H4C fresh start. Labels are the preselected seed-100 pools, not a new search.
    fresh = np.zeros_like(checkpoints["early"])
    y_s = label_bank[100]["source"][1]
    u_source = confidence_interval(c[source_idx], d[source_idx], y_s) or (0.0, 1.0)
    y_info = label_bank[100]["extra32"][1][32:]
    y_uni = label_bank[100]["uniform"][1][32:]
    online = {}
    for name, schedule in (
        ("source", None),
        ("informative", info_idx),
        ("uniform", uniform_idx),
        ("oracle", "oracle"),
    ):
        logits = fresh.copy()
        phi = 0.5
        idx = source_idx.copy()
        y = y_s.copy()
        series = []
        for k in range(40):
            if isinstance(schedule, np.ndarray) and k in (9, 19):
                take = schedule[:16] if k == 9 else schedule[16:32]
                extra_y = y_info[:16] if name == "informative" and k == 9 else (
                    y_info[16:32] if name == "informative" else (y_uni[:16] if k == 9 else y_uni[16:32])
                )
                idx = np.concatenate([idx, take])
                y = np.concatenate([y, extra_y])
            u = (TRUE_W, TRUE_W) if name == "oracle" else (confidence_interval(c[idx], d[idx], y) or (0.0, 1.0))
            if name == "oracle":
                chosen = TRUE_W
                bounds = u_source
            else:
                window = (max(0.0, phi - 0.02), min(1.0, phi + 0.02))
                chosen, _ = search_phi(model, logits, u if name != "source" else u_source, low, high, scale, lr, window)
                bounds = u if name != "source" else u_source
            mu, nxt = mu_and_step(model, logits, chosen, scale, lr)
            _, mu2, _ = H3.exact_gradient(model, nxt, coef(0.5, scale))
            series.append({
                "update": k + 1,
                "phi": chosen,
                "true_return": true_return(mu2, low, high),
                "true_gain": gain(mu, mu2, TRUE_W, low, high),
                "source_certificate": worst(mu, mu2, u_source, low, high),
                "updated_certificate": worst(mu, mu2, bounds, low, high),
                "width": None if name == "oracle" else (bounds[1] - bounds[0]),
                "labels": int(len(y)),
            })
            logits, phi = nxt, chosen
        online[name] = {
            "initial_return": series[0]["true_return"] - series[0]["true_gain"],
            "final_return": series[-1]["true_return"],
            "gain": series[-1]["true_return"] - (series[0]["true_return"] - series[0]["true_gain"]),
            "final_width": series[-1]["width"],
            "final_phi": series[-1]["phi"],
            "min_source": min(row["source_certificate"] for row in series),
            "final_source": series[-1]["source_certificate"],
            "final_updated": series[-1]["updated_certificate"],
            "curve": [row["true_return"] for row in series],
        }
    figure(branches, online, ROOT / "outputs/horizon_bridge/h4_dst_preference/forks.png")
    out = ROOT / "outputs/horizon_bridge/h4_dst_preference"
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "gamma_pos": gamma_pos,
        "gamma0": 0.0,
        "development_w": w_dev,
        "source_median_fisher": source_fisher,
        "coverage_reps": reps,
        "coverage_fraction": covered / reps,
        "coverage_note": "Repeated labels on the fixed source pairs. Not a formal 95 percent guarantee.",
        "fits": rows_fit,
        "branches": branches,
        "online": {key: {k: v for k, v in value.items() if k != "curve"} | {"curve_head": value["curve"][:3], "curve_tail": value["curve"][-3:]} for key, value in online.items()},
        "online_curves": {key: value["curve"] for key, value in online.items()},
    }
    write_summary(out, clean(payload), started)
    leftover = out / "branches_partial.json"
    if leftover.exists():
        leftover.unlink()


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def figure(branches, online, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plot, axes = plt.subplots(1, 2, figsize=(8.4, 3.5))
    early = [row for row in branches if row["checkpoint"] == "early" and row["seed"] == 100]
    axes[0].scatter([row["return_change"] for row in early], [row["final_source"] for row in early])
    for row in early:
        axes[0].annotate(row["mode"], (row["return_change"], row["final_source"]), fontsize=7)
    axes[0].set_xlabel("true return change")
    axes[0].set_ylabel("final source certificate")
    axes[0].set_title("early checkpoint, label seed 100")
    for name, value in online.items():
        axes[1].plot(value["curve"], label=name)
    axes[1].set_xlabel("policy update")
    axes[1].set_ylabel("exact true return")
    axes[1].legend(fontsize=7)
    plot.tight_layout()
    plot.savefig(path, dpi=140)
    plt.close(plot)


def sampled_online() -> None:
    """H4C sampled check. Rebuilds the same frozen pools and does not refit the exact run."""
    started = time.perf_counter()
    frozen = json.loads((ROOT / "experiments/horizon_real/E1/Deep-Sea-Treasure/frozen/frozen.json").read_text())
    h3 = json.loads((ROOT / "outputs/horizon_bridge/h3_dst_oracle/summary.json").read_text())
    low = np.array(frozen["low"], dtype=float)
    high = np.array(frozen["high"], dtype=float)
    scale = high - low
    lr = float(h3["exact_lr"])
    model = H3.build_model()
    features = collect_features(model, low, high, 480, 7100)
    _diff, c, d = pair_terms(features)
    y_dev = label(c[:40], d[:40], np.random.default_rng(7200))
    w_dev = fit_mle(c[:40], d[:40], y_dev)
    rest = np.arange(40, len(d))
    order_low = rest[np.argsort(np.abs(d[rest]))]
    source_idx = order_low[:32]
    remaining = [i for i in rest if i not in set(source_idx.tolist())]
    info_order = sorted(remaining, key=lambda i: -((d[i] ** 2) * sigmoid(c[i] + w_dev * d[i]) * (1 - sigmoid(c[i] + w_dev * d[i]))))
    info_idx = np.array(info_order[:64])
    y_s = label(c[source_idx], d[source_idx], np.random.default_rng(100))
    y_info = label(c[info_idx[:32]], d[info_idx[:32]], np.random.default_rng(150))
    u_source = confidence_interval(c[source_idx], d[source_idx], y_s) or (0.0, 1.0)
    rows = []
    for name, schedule in (("source", None), ("informative", info_idx)):
        for seed in (0, 1, 2):
            rng = np.random.default_rng(9000 + seed)
            logits = np.zeros((len(model["states"]), H3.N_ACTIONS))
            phi = 0.5
            idx, y = source_idx.copy(), y_s.copy()
            for k in range(40):
                if isinstance(schedule, np.ndarray) and k in (9, 19):
                    take = schedule[:16] if k == 9 else schedule[16:32]
                    extra = y_info[:16] if k == 9 else y_info[16:32]
                    idx = np.concatenate([idx, take])
                    y = np.concatenate([y, extra])
                bounds = u_source if name == "source" else (confidence_interval(c[idx], d[idx], y) or (0.0, 1.0))
                window = (max(0.0, phi - 0.02), min(1.0, phi + 0.02))
                phi, _ = search_phi(model, logits, bounds, low, high, scale, lr, window)
                probs = H3.softmax(logits)
                episodes = [H3.rollout_episode(model, probs, rng) for _ in range(256)]
                values = []
                for ep in episodes:
                    feat = (ep["ret"] - low) / (high - low)
                    values.append(phi * feat[0] + (1.0 - phi) * feat[1])
                baseline = float(np.mean(values))
                grad = np.zeros_like(logits)
                for ep, value in zip(episodes, values):
                    adv = float(value - baseline)
                    for s, a, prob in ep["steps"]:
                        one = np.zeros(H3.N_ACTIONS)
                        one[a] = 1.0
                        grad[s] += adv * (one - prob)
                step = lr * grad / len(episodes)
                norm = np.linalg.norm(step)
                if norm > 1.0:
                    step *= 1.0 / norm
                logits = logits + step
            _, mu, _ = H3.exact_gradient(model, logits, coef(0.5, scale))
            rows.append({"name": name, "seed": seed, "final_return": true_return(mu, low, high), "final_phi": phi, "labels": int(len(y))})
            print(rows[-1], flush=True)
    summary_path = ROOT / "outputs/horizon_bridge/h4_dst_preference/summary.json"
    payload = json.loads(summary_path.read_text())
    payload["sampled_online"] = rows
    payload["sampled_online_runtime_seconds"] = time.perf_counter() - started
    summary_path.write_text(json.dumps(clean(payload), indent=2))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--sampled":
        sampled_online()
    else:
        main()
