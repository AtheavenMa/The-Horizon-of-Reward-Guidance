#!/usr/bin/env python3
"""H2: one extra binary stage on the verified two-step MDP. Sixty-four outcomes."""
from __future__ import annotations

import csv
import importlib.util
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


def load_two_step():
    path = Path(__file__).resolve().parent / "10_two_step_mdp.py"
    spec = importlib.util.spec_from_file_location("two_step_h2", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


T = load_two_step()
M = T.M
ROOT = Path(__file__).resolve().parents[2]
LAM2 = 0.2
KAPPA2 = 0.025
KAPPA = T.KAPPA
H = M.H
GAMMA = M.GAMMA


def pack():
    rows = [(q, s, z, u, b, v)
            for q in (-1.0, 1.0) for s in (-1.0, 1.0)
            for z in (0.0, 1.0) for u in (-1.0, 1.0)
            for b in (0.0, 1.0) for v in (-1.0, 1.0)]
    return [np.array([row[i] for row in rows]) for i in range(6)]


Q, S, Z, U, B, V = pack()


def probabilities(x, y, zeta, xi):
    p_qs = 0.25 * (1.0 + x * Q) * (1.0 + y * S)
    p_z = 0.5 * (1.0 + (2.0 * Z - 1.0) * T.LAM * Q * S)
    th = np.tanh(zeta)
    p_u = 0.5 * (1.0 + Z * U * th)
    p_b = 0.5 * (1.0 + (2.0 * B - 1.0) * LAM2 * U * (2.0 * Z - 1.0))
    w = xi * (2.0 * B - 1.0)
    p_v = 0.5 * (1.0 + V * np.tanh(w))
    return p_qs * p_z * p_u * p_b * p_v


def scores(zeta, xi):
    th = float(np.tanh(zeta))
    sech2 = 1.0 - th * th
    score_z = np.where(Z > 0.5, U * sech2 / (1.0 + U * th), 0.0)
    w = xi * (2.0 * B - 1.0)
    tw = np.tanh(w)
    score_x = V * (1.0 - tw * tw) * (2.0 * B - 1.0) / (1.0 + V * tw)
    return np.stack([Q - 0.0, S - 0.0, score_z, score_x], axis=1)


def gradient(x, y, zeta, xi, phi, kappa2) -> np.ndarray:
    probability = probabilities(x, y, zeta, xi)
    reward = phi * Q + (1.0 - phi) * S + KAPPA * Z * U + kappa2 * B * V
    direction = scores(zeta, xi)
    direction[:, 0] = Q - x
    direction[:, 1] = S - y
    return probability @ (reward[:, None] * direction)


def expectation_bv(x, y, zeta, xi) -> float:
    probability = probabilities(x, y, zeta, xi)
    return float(probability @ (B * V))


def true_return(x, y, zeta, xi, preference, kappa2) -> float:
    return (
        preference * x + (1.0 - preference) * y
        + KAPPA * T.expectation_zu(x, y, zeta)
        + kappa2 * expectation_bv(x, y, zeta, xi)
    )


def advance(x, y, zeta, xi, grad, h):
    return (
        float(np.tanh(np.arctanh(x) + h * grad[0])),
        float(np.tanh(np.arctanh(y) + h * grad[1])),
        float(zeta + h * grad[2]),
        float(xi + h * grad[3]),
    )


def finite_gain(state, nxt, preference, kappa2, h) -> float:
    return (true_return(*nxt, preference, kappa2) - true_return(*state, preference, kappa2)) / h


def interval_gain(state, nxt, bounds, kappa2, h) -> float:
    return min(finite_gain(state, nxt, bounds[0], kappa2, h), finite_gain(state, nxt, bounds[1], kappa2, h))


def instantaneous(x, y, zeta, xi, phi, preference, kappa2) -> float:
    guided = gradient(x, y, zeta, xi, phi, kappa2)
    valued = gradient(x, y, zeta, xi, preference, kappa2)
    return float(guided @ valued)


def self_check(phi0: float) -> None:
    for state in ((0.8, 0.0, 0.0, 0.0), (-0.2, 0.3, -0.4, 0.2)):
        total = float(probabilities(*state).sum())
        if abs(total - 1.0) > 1e-12 or np.any(probabilities(*state) < -1e-15):
            raise AssertionError(("probability", total))
    enumerated = expectation_bv(0.3, -0.2, 0.4, 0.25)
    if abs(enumerated - (np.tanh(0.25) * probabilities(0.3, -0.2, 0.4, 0.0) @ B)) > 1e-12:
        raise AssertionError("E[bv] factorization")
    old = T.gradient(0.8, 0.2, 0.15, phi0, KAPPA)
    got = gradient(0.8, 0.2, 0.15, 0.3, phi0, 0.0)
    if np.linalg.norm(got[:3] - old) > 1e-10 or abs(got[3]) > 1e-10:
        raise AssertionError(("kappa2=0 gradient", got, old))
    zero_xi = gradient(0.8, 0.0, 0.0, 0.0, phi0, 0.0)
    if abs(zero_xi[3]) > 1e-12 or np.linalg.norm(zero_xi[:3] - T.gradient(0.8, 0.0, 0.0, phi0, KAPPA)) > 1e-10:
        raise AssertionError(("regression gradient", zero_xi, T.gradient(0.8, 0.0, 0.0, phi0, KAPPA)))
    informative = (1.0, -1.0, 1.0, 1.0, 1.0, -1.0)
    other = (-1.0, 1.0, 1.0, 1.0, 1.0, -1.0)
    logit = phi0 * (informative[0] - other[0]) + (1.0 - phi0) * (informative[1] - other[1])
    if abs(logit - (4.0 * phi0 - 2.0)) > 1e-12:
        raise AssertionError("matched informative logit changed")
    source = (1.0, 1.0, 1.0, -1.0, 0.0, 1.0)
    source_other = (-1.0, -1.0, 1.0, -1.0, 0.0, 1.0)
    source_logit = 0.2 * (source[0] - source_other[0]) + 0.8 * (source[1] - source_other[1])
    if abs(source_logit - 2.0) > 1e-12:
        raise AssertionError("source logit is not independent of the preference")
    eps = 1e-6
    base = np.array([np.arctanh(0.8), np.arctanh(-0.2), 0.15, -0.1])
    for kappa2 in (0.0, KAPPA2):
        analytic = gradient(0.8, -0.2, 0.15, -0.1, phi0, kappa2)
        numerical = []
        for index in range(4):
            plus, minus = base.copy(), base.copy()
            plus[index] += eps
            minus[index] -= eps
            jp = true_return(np.tanh(plus[0]), np.tanh(plus[1]), plus[2], plus[3], phi0, kappa2)
            jm = true_return(np.tanh(minus[0]), np.tanh(minus[1]), minus[2], minus[3], phi0, kappa2)
            numerical.append((jp - jm) / (2 * eps))
        if np.linalg.norm(np.array(numerical) - analytic) > 1e-7:
            raise AssertionError((kappa2, numerical, analytic))
        rate = instantaneous(0.8, -0.2, 0.15, -0.1, phi0, 2.0, kappa2)
        moved = advance(0.8, -0.2, 0.15, -0.1, gradient(0.8, -0.2, 0.15, -0.1, phi0, kappa2), eps)
        finite = finite_gain((0.8, -0.2, 0.15, -0.1), moved, 2.0, kappa2, eps)
        if abs(rate - finite) > 1e-5:
            raise AssertionError(("instantaneous", rate, finite))


def step_certificate(state, phi, kappa2, h, bounds):
    grad = gradient(*state, phi, kappa2)
    nxt = advance(*state, grad, h)
    return interval_gain(state, nxt, bounds, kappa2, h), nxt, grad


def finite_pointwise(state, kappa2, h) -> float:
    def worst(phi):
        value, _, _ = step_certificate(state, float(phi), kappa2, h, M.U0)
        return -value

    grid = np.linspace(0.0, 1.0, 1001)
    values = np.array([-worst(phi) for phi in grid])
    fit = minimize_scalar(worst, bounds=(0.0, 1.0), method="bounded", options={"xatol": 1e-12})
    return float(max(values.max(), -fit.fun))


def rollout(phi_seq, kappa2, h, active, grads=None) -> list[dict]:
    state = (0.8, 0.0, 0.0, 0.0)
    rows = []
    for k, phi in enumerate(phi_seq):
        exact_gain, exact_next, exact_grad = step_certificate(state, float(phi), kappa2, h, M.U0)
        used = exact_grad if grads is None else grads(k, state, float(phi))
        realized_next = advance(*state, used, h)
        population_next = exact_next
        rows.append({
            "k": k,
            "t": k * h,
            "x": state[0], "y": state[1], "zeta": state[2], "xi": state[3],
            "phi": float(phi),
            "source_certificate": interval_gain(state, realized_next, M.U0, kappa2, h),
            "population_certificate": exact_gain,
            "updated_certificate": interval_gain(state, realized_next, active(k), kappa2, h),
            "true_finite_step_gain": finite_gain(state, realized_next, M.reference_phi0(), kappa2, h),
            "instantaneous": min(
                instantaneous(*state, float(phi), M.U0[0], kappa2),
                instantaneous(*state, float(phi), M.U0[1], kappa2),
            ),
            "gradient_error": float(np.linalg.norm(used - exact_grad)),
            "x_next": realized_next[0], "y_next": realized_next[1],
            "zeta_next": realized_next[2], "xi_next": realized_next[3],
        })
        state = realized_next
    return rows


def scheduled(n, phi0, h):
    phi = np.empty(n)
    phi[0] = phi0
    for k in range(1, n):
        phi[k] = min(1.0, phi[k - 1] + M.RHO * h)
    return phi


def first_below(rows, key):
    for row in rows:
        if row[key] < GAMMA - 1e-15:
            return float(row["t"])
    return None


def at_time(rows, when, key):
    if when is None:
        return rows[-1][key]
    for row in rows:
        if abs(row["t"] - when) <= 1e-12:
            return row[key]
    return None


def sample_gradient(state, phi, kappa2, draws):
    x, y, zeta, xi = state
    uq, us, uz, uu, ub, uv = draws
    q = np.where(uq < (1.0 + x) / 2.0, 1.0, -1.0)
    s = np.where(us < (1.0 + y) / 2.0, 1.0, -1.0)
    z = np.where(uz < (1.0 + T.LAM * q * s) / 2.0, 1.0, 0.0)
    th = np.tanh(zeta)
    tilt = np.where(z > 0.5, th, 0.0)
    u = np.where(uu < (1.0 + tilt) / 2.0, 1.0, -1.0)
    p_b = 0.5 * (1.0 + LAM2 * u * (2.0 * z - 1.0))
    b = np.where(ub < p_b, 1.0, 0.0)
    w = xi * (2.0 * b - 1.0)
    v = np.where(uv < 0.5 * (1.0 + np.tanh(w)), 1.0, -1.0)
    reward = phi * q + (1.0 - phi) * s + KAPPA * z * u + kappa2 * b * v
    sech2 = 1.0 - th * th
    score_z = np.where(z > 0.5, u * sech2 / (1.0 + u * th), 0.0)
    tw = np.tanh(w)
    score_x = v * (1.0 - tw * tw) * (2.0 * b - 1.0) / (1.0 + v * tw)
    return np.array([
        float(np.mean(reward * (q - x))),
        float(np.mean(reward * (s - y))),
        float(np.mean(reward * score_z)),
        float(np.mean(reward * score_x)),
    ])


def draw_stream(seed, steps, n):
    rng = np.random.default_rng(60_000 + seed)
    return [(rng.random(n), rng.random(n), rng.random(n), rng.random(n), rng.random(n), rng.random(n)) for _ in range(steps)]


def robust_phi(state, kappa2, h, bounds, low, high) -> float:
    grid = np.linspace(low, high, 21)
    scores = [step_certificate(state, float(phi), kappa2, h, bounds)[0] for phi in grid]
    return float(grid[int(np.argmax(scores))])


def describe(name, rows, pointwise_at) -> dict:
    loss = first_below(rows, "source_certificate")
    return {
        "name": name,
        "first_source_loss": loss,
        "pointwise_at_loss": pointwise_at,
        "true_gain_at_loss": at_time(rows, loss, "true_finite_step_gain"),
        "terminal_source": rows[-1]["source_certificate"],
        "terminal_updated": rows[-1]["updated_certificate"],
        "terminal_xi": rows[-1]["xi_next"],
        "terminal_zeta": rows[-1]["zeta_next"],
        "terminal_phi": rows[-1]["phi"],
        "mean_gradient_error": float(np.mean([row["gradient_error"] for row in rows])),
        "population_loss": first_below(rows, "population_certificate"),
        "min_pointwise_recorded": pointwise_at,
    }


def main() -> None:
    started = time.perf_counter()
    phi0 = float(M.reference_phi0())
    self_check(phi0)
    out = ROOT / "outputs/horizon_bridge/h2_three_stage"
    out.mkdir(parents=True, exist_ok=True)

    def source(_k):
        return M.U0

    zero_phi = scheduled(32, phi0, H)
    zero = rollout(zero_phi, 0.0, H, source)
    reference = list(csv.DictReader((ROOT / "outputs/horizon_bridge/step10_g4_extension/exact_steps.csv").open()))
    ref = [row for row in reference if row["model"] == "kappa005"]
    gap = max(abs(a["source_certificate"] - float(b["source_certificate"])) for a, b in zip(zero, ref))
    if gap > 1e-8 or max(abs(row["xi_next"]) for row in zero) > 1e-12:
        raise AssertionError(("regression", gap, zero[-1]["xi_next"]))
    zero_loss = first_below(zero, "source_certificate")
    zero_row = zero[-1] if zero_loss is None else zero[int(round(zero_loss / H))]
    zero_point = finite_pointwise((zero_row["x"], zero_row["y"], zero_row["zeta"], zero_row["xi"]), 0.0, H)

    main_phi = scheduled(48, phi0, H)
    exact = rollout(main_phi, KAPPA2, H, source)
    loss = first_below(exact, "source_certificate")
    loss_state = exact[-1] if loss is None else exact[int(round(loss / H))]
    pointwise = finite_pointwise((loss_state["x"], loss_state["y"], loss_state["zeta"], loss_state["xi"]), KAPPA2, H)
    # The same maximal-upward sequence H1 searched, transferred for 32 steps.
    transferred = rollout(scheduled(32, phi0, H), KAPPA2, H, source)

    labels = {seed: M.label_batch(phi0, 128, seed, True) for seed in M.LABEL_SEEDS}
    source_labels = M.label_batch(phi0, 128, 100, False)

    def learned_active(seed):
        interval = tuple(labels[seed]["interval"]) if not labels[seed]["empty"] else None

        def active(k, interval=interval):
            if k < M.FEEDBACK_STEP or interval is None:
                return M.U0
            return interval
        return active

    frozen = {}
    for seed in M.LABEL_SEEDS:
        frozen[seed] = rollout(main_phi, KAPPA2, H, learned_active(seed))
    frozen_source = rollout(main_phi, KAPPA2, H, source)

    sampled_summary = []
    for n in (512, 2048):
        for seed in (0, 1, 2):
            draws = draw_stream(seed, 32, n)
            def grads(k, state, phi, draws=draws, n=n):
                return sample_gradient(state, phi, KAPPA2, draws[k])
            rows = rollout(scheduled(32, phi0, H), KAPPA2, H, source, grads)
            sampled_summary.append({
                "n": n, "seed": seed,
                "realized_loss": first_below(rows, "source_certificate"),
                "population_loss": first_below(rows, "population_certificate"),
                "true_gain_at_0_012": at_time(rows, 0.012, "true_finite_step_gain") if any(abs(r["t"] - 0.012) < 1e-12 for r in rows) else None,
                "terminal_source": rows[-1]["source_certificate"],
                "mean_gradient_error": float(np.mean([r["gradient_error"] for r in rows])),
                "terminal_xi": rows[-1]["xi_next"],
            })

    # Adaptive arm: exact scheduled control through feedback, then the H1 robust window.
    batch = labels[100]
    interval = tuple(batch["interval"])
    phi_adapt = []
    state = (0.8, 0.0, 0.0, 0.0)
    current = phi0
    adapt_rows = []
    for k in range(48):
        if k < M.FEEDBACK_STEP:
            phi = float(main_phi[k])
        else:
            phi = robust_phi(state, KAPPA2, H, interval, max(0.0, current - M.RHO * H), min(1.0, current + M.RHO * H))
        phi_adapt.append(phi)
        gain, nxt, _ = step_certificate(state, phi, KAPPA2, H, interval if k >= M.FEEDBACK_STEP else M.U0)
        adapt_rows.append({"phi": phi, "certificate": gain, "true": finite_gain(state, nxt, phi0, KAPPA2, H)})
        state = nxt
        current = phi
    phi_gap = float(np.max(np.abs(np.array(phi_adapt) - main_phi)))
    guided = []
    for seed in (0, 1, 2):
        draws = draw_stream(1000 + seed, 32, 2048)
        state = (0.8, 0.0, 0.0, 0.0)
        current = phi0
        rows = []
        for k in range(32):
            if k < M.FEEDBACK_STEP:
                phi = float(scheduled(32, phi0, H)[k])
            else:
                phi = robust_phi(state, KAPPA2, H, interval, max(0.0, current - M.RHO * H), min(1.0, current + M.RHO * H))
            used = sample_gradient(state, phi, KAPPA2, draws[k])
            nxt = advance(*state, used, H)
            rows.append({
                "phi": phi,
                "source": interval_gain(state, nxt, M.U0, KAPPA2, H),
                "updated": interval_gain(state, nxt, interval if k >= M.FEEDBACK_STEP else M.U0, KAPPA2, H),
                "true": finite_gain(state, nxt, phi0, KAPPA2, H),
            })
            state = nxt
            current = phi
        ref_phi = scheduled(32, phi0, H)
        guided.append({
            "seed": seed,
            "phi_gap": float(np.max(np.abs([row["phi"] for row in rows] - ref_phi))),
            "realized_loss": first_below([{"t": i * H, "source_certificate": row["source"]} for i, row in enumerate(rows)], "source_certificate"),
            "updated_loss": first_below([{"t": i * H, "updated_certificate": row["updated"]} for i, row in enumerate(rows)], "updated_certificate"),
            "terminal_true": rows[-1]["true"],
            "terminal_updated": rows[-1]["updated"],
        })

    with (out / "exact_steps.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(exact[0].keys()))
        writer.writeheader()
        writer.writerows(exact)

    figure(ref, zero, exact, out / "three_stage.png")
    payload = {
        "regression_gap_vs_g41": gap,
        "regression_loss": first_below(zero, "source_certificate"),
        "regression_pointwise_at_recorded_step": zero_point,
        "kappa2": KAPPA2,
        "exact_loss": loss,
        "exact_pointwise_at_loss_or_end": pointwise,
        "exact_true_gain": at_time(exact, loss, "true_finite_step_gain"),
        "exact_terminal_source": exact[-1]["source_certificate"],
        "exact_terminal_xi": exact[-1]["xi_next"],
        "exact_terminal_zeta": exact[-1]["zeta_next"],
        "exact_censored": loss is None,
        "transferred_h1_schedule_loss": first_below(transferred, "source_certificate"),
        "transferred_terminal_source": transferred[-1]["source_certificate"],
        "source_labels_interval": source_labels["interval"],
        "frozen_source_loss": first_below(frozen_source, "source_certificate"),
        "frozen": [{
            "seed": seed,
            "interval": labels[seed]["interval"],
            "source_loss": first_below(rows, "source_certificate"),
            "updated_loss": first_below(rows, "updated_certificate"),
            "terminal_updated": rows[-1]["updated_certificate"],
            "suffix_certified": all(row["updated_certificate"] >= GAMMA - 1e-15 for row in rows if row["k"] >= M.FEEDBACK_STEP),
        } for seed, rows in frozen.items()],
        "sampled": sampled_summary,
        "guided_sampled": guided,
        "adaptive_phi_gap": phi_gap,
        "adaptive_paths_identical": phi_gap < 1e-12,
        "adaptive_terminal_true_gap": float(adapt_rows[-1]["true"] - exact[-1]["true_finite_step_gain"]),
        "rows_exact_zero": describe("kappa0", zero, zero_point),
        "rows_exact": describe("kappa2", exact, pointwise),
    }
    write_summary(out, clean(payload), started)


def clean(value):
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def figure(reference, zero, exact, path) -> None:
    plot, axis = plt.subplots(figsize=(7.2, 3.6))
    axis.plot([float(row["t"]) for row in reference], [float(row["source_certificate"]) for row in reference], label="16-outcome kappa=0.05")
    axis.plot([row["t"] for row in zero], [row["source_certificate"] for row in zero], ls="--", label="64-outcome kappa2=0")
    axis.plot([row["t"] for row in exact], [row["source_certificate"] for row in exact], label="64-outcome kappa2=0.025")
    axis.axhline(GAMMA, color="black", lw=0.8)
    axis.set_xlabel("time")
    axis.set_ylabel("source finite-step gain")
    axis.legend(fontsize=8)
    plot.tight_layout()
    plot.savefig(path, dpi=140)
    plt.close(plot)


if __name__ == "__main__":
    main()
