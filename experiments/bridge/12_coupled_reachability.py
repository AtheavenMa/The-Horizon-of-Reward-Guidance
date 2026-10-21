#!/usr/bin/env python3
"""H1: fixed-path corridor, coupled guidance search, and an outer enclosure.

The scheduled G4.1 path is a reference, not an assumption that it is optimal.
A solver that finds no better path does not prove every admissible path fails.
"""
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
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.bridge.common import write_summary
from horizon import interval_feasible


def load_two_step():
    path = Path(__file__).resolve().parent / "10_two_step_mdp.py"
    spec = importlib.util.spec_from_file_location("two_step_h1", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


T = load_two_step()
M = T.M
ROOT = Path(__file__).resolve().parents[2]
GAMMA = M.GAMMA
H = M.H
RHO = M.RHO


def pack_outcomes():
    records = [(q, s, z, u) for q in (-1.0, 1.0) for s in (-1.0, 1.0) for z in (0.0, 1.0) for u in (-1.0, 1.0)]
    return [np.array([row[i] for row in records], dtype=float) for i in range(4)]


Q, S, Z, U = pack_outcomes()


def probabilities(x: float, y: float, zeta: float) -> np.ndarray:
    p_qs = 0.25 * (1.0 + x * Q) * (1.0 + y * S)
    p_z = 0.5 * (1.0 + (2.0 * Z - 1.0) * T.LAM * Q * S)
    th = np.tanh(zeta)
    p_u = 0.5 * (1.0 + Z * U * th)
    return p_qs * p_z * p_u


def gradient_np(x: float, y: float, zeta: float, phi: float, kappa: float) -> np.ndarray:
    probability = probabilities(x, y, zeta)
    reward = phi * Q + (1.0 - phi) * S + kappa * Z * U
    th = float(np.tanh(zeta))
    sech2 = 1.0 - th * th
    score_z = np.where(Z > 0.5, U * sech2 / (1.0 + U * th), 0.0)
    return np.array([
        float(np.sum(probability * reward * (Q - x))),
        float(np.sum(probability * reward * (S - y))),
        float(np.sum(probability * reward * score_z)),
    ])


def advance_np(x: float, y: float, zeta: float, grad: np.ndarray, h: float) -> tuple[float, float, float]:
    return (
        float(np.tanh(np.arctanh(x) + h * grad[0])),
        float(np.tanh(np.arctanh(y) + h * grad[1])),
        float(zeta + h * grad[2]),
    )


def gains_after(x, y, zeta, x2, y2, z2, kappa, h, bounds) -> float:
    now = np.array([T.true_return(x, y, zeta, a, kappa) for a in bounds])
    nxt = np.array([T.true_return(x2, y2, z2, a, kappa) for a in bounds])
    return float(np.min((nxt - now) / h))


def step_certificate(x, y, zeta, phi, kappa, h, bounds) -> tuple[float, tuple[float, float, float]]:
    grad = gradient_np(x, y, zeta, phi, kappa)
    nxt = advance_np(x, y, zeta, grad, h)
    return gains_after(x, y, zeta, *nxt, kappa, h, bounds), nxt


def rollout(phi: np.ndarray, kappa: float, h: float, active) -> dict:
    x, y, zeta = 0.8, 0.0, 0.0
    certificates = []
    source = []
    states = [(x, y, zeta)]
    for k, value in enumerate(phi):
        bounds = active(k)
        certificate, (x, y, zeta) = step_certificate(x, y, zeta, float(value), kappa, h, bounds)
        certificates.append(certificate)
        source.append(step_certificate(states[-1][0], states[-1][1], states[-1][2], float(value), kappa, h, M.U0)[0])
        states.append((x, y, zeta))
    certificates = np.array(certificates)
    source = np.array(source)
    return {
        "phi": np.array(phi, dtype=float),
        "certificate": certificates,
        "source_certificate": source,
        "min_slack": float(np.min(certificates - GAMMA)),
        "first_loss": first_loss(certificates, h),
        "source_first_loss": first_loss(source, h),
        "states": states,
    }


def first_loss(values: np.ndarray, h: float):
    below = np.flatnonzero(values < GAMMA - 1e-15)
    if len(below) == 0:
        return None
    return float(below[0] * h)


def project(phi: np.ndarray, budget: float) -> np.ndarray:
    out = np.array(phi, dtype=float)
    out[0] = M.reference_phi0() if abs(out[0] - M.reference_phi0()) > 0 else out[0]
    out[0] = float(M.reference_phi0())
    for k in range(1, len(out)):
        out[k] = float(np.clip(out[k], out[k - 1] - budget, out[k - 1] + budget))
        out[k] = float(np.clip(out[k], 0.0, 1.0))
    return out


def feasible_sets(states: list[tuple[float, float, float]], kappa: float, h: float, active) -> list[list[tuple[float, float]]]:
    phis = np.linspace(0.0, 1.0, 20001)
    sets = []
    for k, (x, y, zeta) in enumerate(states):
        bounds = active(k)
        values = np.empty(len(phis))
        # Affine score gradient: one pair of basis gradients, then a vector of phis.
        g0 = gradient_np(x, y, zeta, 0.0, kappa)
        g1 = gradient_np(x, y, zeta, 1.0, kappa) - g0
        tq = np.arctanh(x) + h * (g0[0] + phis * g1[0])
        ts = np.arctanh(y) + h * (g0[1] + phis * g1[1])
        tz = zeta + h * (g0[2] + phis * g1[2])
        x2, y2, z2 = np.tanh(tq), np.tanh(ts), tz
        now = {a: T.true_return(x, y, zeta, a, kappa) for a in bounds}
        for endpoint in bounds:
            delayed = T.KAPPA * np.tanh(z2) * (1.0 + T.LAM * x2 * y2) / 2.0
            base_now = T.KAPPA * np.tanh(zeta) * (1.0 + T.LAM * x * y) / 2.0
            gain = (endpoint * (x2 - x) + (1.0 - endpoint) * (y2 - y) + delayed - base_now) / h
            if endpoint == bounds[0]:
                values = gain
            else:
                values = np.minimum(values, gain)
        ok = values >= GAMMA - 1e-12
        pieces = []
        start = None
        for index, flag in enumerate(ok):
            if flag and start is None:
                start = index
            if start is not None and (not flag or index == len(ok) - 1):
                stop = index if flag and index == len(ok) - 1 else index - 1
                lo = float(phis[start])
                hi = float(phis[stop])
                if start > 0 and not ok[start - 1]:
                    lo = float(refine_edge(phis[start - 1], phis[start], x, y, zeta, kappa, h, bounds, True))
                if stop < len(phis) - 1 and not ok[stop + 1]:
                    hi = float(refine_edge(phis[stop], phis[stop + 1], x, y, zeta, kappa, h, bounds, False))
                if hi >= lo:
                    pieces.append((lo, hi))
                start = None
        sets.append(merge_pieces(pieces))
    return sets


def refine_edge(left, right, x, y, zeta, kappa, h, bounds, entering: bool) -> float:
    for _ in range(40):
        mid = 0.5 * (left + right)
        value, _ = step_certificate(x, y, zeta, mid, kappa, h, bounds)
        inside = value >= GAMMA - 1e-12
        if inside == entering:
            right = mid
        else:
            left = mid
    return 0.5 * (left + right)


def merge_pieces(pieces: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not pieces:
        return []
    merged = [pieces[0]]
    for lo, hi in pieces[1:]:
        if lo <= merged[-1][1] + 1e-8:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def fixed_path_reachability(pieces: list[list[tuple[float, float]]], budget: float, phi0: float) -> dict:
    single = all(len(item) <= 1 for item in pieces)
    disconnected = any(len(item) > 1 for item in pieces)
    if not single:
        return {"called_interval_feasible": False, "disconnected": True, "feasible": None, "first_infeasible_index": None,
                "note": "At least one frozen state has a disconnected feasible set, so the single-interval helper does not apply."}
    if any(len(item) == 0 for item in pieces):
        return {"called_interval_feasible": False, "disconnected": False, "feasible": False,
                "first_infeasible_index": next(i for i, item in enumerate(pieces) if len(item) == 0),
                "note": "A frozen state has an empty one-step feasible set."}
    lower = [item[0][0] for item in pieces]
    upper = [item[0][1] for item in pieces]
    result = interval_feasible(lower, upper, [budget] * (len(pieces) - 1), initial_value=phi0)
    return {
        "called_interval_feasible": True,
        "disconnected": disconnected,
        "feasible": bool(result.feasible),
        "first_infeasible_index": result.first_infeasible_index,
        "witness_head": None if result.witness_path is None else [float(v) for v in result.witness_path[:4]],
        "witness": None if result.witness_path is None else [float(v) for v in result.witness_path],
    }


def optimize_coupled(k_steps: int, budget: float, kappa: float, h: float, active, phi0: float, starts: list[np.ndarray]) -> dict:
    best = None
    reports = []
    for start in starts:
        seed = project(np.resize(start, k_steps), budget)

        def objective(free):
            phi = project(np.concatenate([[phi0], free]), budget)
            traced = rollout(phi, kappa, h, active)
            return -traced["min_slack"]

        fit = minimize(objective, seed[1:], method="SLSQP", bounds=[(0.0, 1.0)] * (k_steps - 1),
                       options={"ftol": 1e-14, "maxiter": 80, "disp": False})
        phi = project(np.concatenate([[phi0], fit.x]), budget)
        traced = rollout(phi, kappa, h, active)
        reports.append({"success": bool(fit.success), "min_slack": traced["min_slack"], "first_loss": traced["first_loss"]})
        if best is None or traced["min_slack"] > best["min_slack"]:
            best = traced
    return {"best": best, "starts": reports}


def schedule(k_steps: int, phi0: float, rate: float, h: float) -> np.ndarray:
    phi = np.empty(k_steps)
    phi[0] = phi0
    for k in range(1, k_steps):
        phi[k] = float(np.clip(phi[k - 1] + rate, 0.0, 1.0))
    return phi


def greedy(k_steps: int, budget: float, kappa: float, h: float, active, phi0: float) -> np.ndarray:
    x, y, zeta = 0.8, 0.0, 0.0
    phi = np.empty(k_steps)
    current = phi0
    for k in range(k_steps):
        lo = max(0.0, current - budget)
        hi = min(1.0, current + budget)
        window = np.linspace(lo, hi, 21)
        scores = [step_certificate(x, y, zeta, float(value), kappa, h, active(k))[0] for value in window]
        current = float(window[int(np.argmax(scores))])
        phi[k] = current
        _, (x, y, zeta) = step_certificate(x, y, zeta, current, kappa, h, active(k))
    return phi


class Iv:
    def __init__(self, lo: float, hi: float | None = None):
        if hi is None:
            hi = lo
        lo, hi = float(lo), float(hi)
        if lo > hi:
            lo, hi = hi, lo
        self.lo = lo
        self.hi = hi

    def __add__(self, other):
        other = iv(other)
        return Iv(np.nextafter(self.lo + other.lo, -np.inf), np.nextafter(self.hi + other.hi, np.inf))

    def __sub__(self, other):
        other = iv(other)
        return Iv(np.nextafter(self.lo - other.hi, -np.inf), np.nextafter(self.hi - other.lo, np.inf))

    def __mul__(self, other):
        other = iv(other)
        products = (self.lo * other.lo, self.lo * other.hi, self.hi * other.lo, self.hi * other.hi)
        return Iv(np.nextafter(min(products), -np.inf), np.nextafter(max(products), np.inf))

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return iv(other) - self

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = iv(other)
        if other.lo <= 0.0 <= other.hi:
            raise ZeroDivisionError(other.lo, other.hi)
        return self * Iv(1.0 / other.hi, 1.0 / other.lo)

    def tanh(self):
        return Iv(np.nextafter(np.tanh(self.lo), -np.inf), np.nextafter(np.tanh(self.hi), np.inf))

    def width(self) -> float:
        return self.hi - self.lo


def iv(value) -> Iv:
    if isinstance(value, Iv):
        return value
    return Iv(float(value), float(value))


def enclosure_step(x: Iv, y: Iv, zeta: Iv, phi: Iv, kappa: float, h: float) -> tuple[Iv, Iv, Iv, float]:
    th = zeta.tanh()
    sech2 = Iv(1.0) - th * th
    g = [Iv(0.0), Iv(0.0), Iv(0.0)]
    upper_gains = []
    # Probabilities and the two endpoint gains are enclosed together.
    endpoint_next = {a: Iv(0.0) for a in M.U0}
    endpoint_now = {a: Iv(0.0) for a in M.U0}
    for q, s, z, u in zip(Q, S, Z, U):
        p_qs = Iv(0.25) * (Iv(1.0) + x * q) * (Iv(1.0) + y * s)
        p_z = Iv(0.5) * (Iv(1.0) + (2.0 * z - 1.0) * T.LAM * q * s)
        p_u = Iv(0.5) * (Iv(1.0) + z * u * th)
        probability = p_qs * p_z * p_u
        reward = phi * q + (1.0 - phi) * s + kappa * z * u
        score_z = Iv(0.0) if z == 0.0 else (Iv(u) * sech2) / (Iv(1.0) + Iv(u) * th)
        weighted = probability * reward
        g[0] = g[0] + weighted * (Iv(q) - x)
        g[1] = g[1] + weighted * (Iv(s) - y)
        g[2] = g[2] + weighted * score_z
        for preference in M.U0:
            term = preference * q + (1.0 - preference) * s + kappa * z * u
            endpoint_now[preference] = endpoint_now[preference] + probability * term
    # Next-state enclosure uses the enclosed gradient. arctanh is applied at the
    # current interval; tanh is monotone.
    def arctanh_iv(value: Iv) -> Iv:
        if value.lo <= -1.0 or value.hi >= 1.0:
            raise ValueError(f"state interval left (-1, 1): {value.lo}, {value.hi}")
        return Iv(np.nextafter(np.arctanh(value.lo), -np.inf), np.nextafter(np.arctanh(value.hi), np.inf))

    x2 = (arctanh_iv(x) + g[0] * h).tanh()
    y2 = (arctanh_iv(y) + g[1] * h).tanh()
    z2 = zeta + g[2] * h
    th2 = z2.tanh()
    for q, s, z, u in zip(Q, S, Z, U):
        p_qs = Iv(0.25) * (Iv(1.0) + x2 * q) * (Iv(1.0) + y2 * s)
        p_z = Iv(0.5) * (Iv(1.0) + (2.0 * z - 1.0) * T.LAM * q * s)
        p_u = Iv(0.5) * (Iv(1.0) + z * u * th2)
        probability = p_qs * p_z * p_u
        for preference in M.U0:
            term = preference * q + (1.0 - preference) * s + kappa * z * u
            endpoint_next[preference] = endpoint_next[preference] + probability * term
    for preference in M.U0:
        gain = (endpoint_next[preference] - endpoint_now[preference]) * (1.0 / h)
        upper_gains.append(gain.hi)
    # sup min(g-1, g2) <= min(sup g-1, sup g2)
    return x2, y2, z2, float(min(upper_gains))


def _neg_iv(value) -> Iv:
    value = iv(value)
    return Iv(np.nextafter(-value.hi, -np.inf), np.nextafter(-value.lo, np.inf))


class _Dual:
    """Forward-mode derivative with interval values. Seeds are the four inputs."""

    def __init__(self, value, dx=0.0, dy=0.0, dz=0.0, dp=0.0):
        self.v = value if isinstance(value, Iv) else iv(value)
        self.dx, self.dy, self.dz, self.dp = map(iv, (dx, dy, dz, dp))

    def __add__(self, other):
        other = _as_dual(other)
        return _Dual(self.v + other.v, self.dx + other.dx, self.dy + other.dy, self.dz + other.dz, self.dp + other.dp)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        other = _as_dual(other)
        return _Dual(self.v - other.v, self.dx - other.dx, self.dy - other.dy, self.dz - other.dz, self.dp - other.dp)

    def __rsub__(self, other):
        return _as_dual(other) - self

    def __mul__(self, other):
        other = _as_dual(other)
        return _Dual(
            self.v * other.v,
            self.dx * other.v + self.v * other.dx,
            self.dy * other.v + self.v * other.dy,
            self.dz * other.v + self.v * other.dz,
            self.dp * other.v + self.v * other.dp,
        )

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = _as_dual(other)
        inv = Iv(1.0) / other.v
        inv2 = inv * inv
        return self * _Dual(inv, _neg_iv(other.dx) * inv2, _neg_iv(other.dy) * inv2, _neg_iv(other.dz) * inv2, _neg_iv(other.dp) * inv2)

    def tanh(self):
        th = self.v.tanh()
        sech2 = Iv(1.0) - th * th
        return _Dual(th, sech2 * self.dx, sech2 * self.dy, sech2 * self.dz, sech2 * self.dp)


def _as_dual(value) -> _Dual:
    return value if isinstance(value, _Dual) else _Dual(value)


def _arctanh_dual(value: _Dual) -> _Dual:
    if value.v.lo <= -1.0 or value.v.hi >= 1.0:
        raise ValueError("arctanh domain")
    base = Iv(np.nextafter(np.arctanh(value.v.lo), -np.inf), np.nextafter(np.arctanh(value.v.hi), np.inf))
    coef = Iv(1.0) / (Iv(1.0) - value.v * value.v)
    return _Dual(base, coef * value.dx, coef * value.dy, coef * value.dz, coef * value.dp)


def _endpoint_gain_dual(x: _Dual, y: _Dual, zeta: _Dual, phi: _Dual, preference: float, kappa: float, h: float) -> _Dual:
    th = zeta.tanh()
    sech2 = _Dual(1.0) - th * th
    gq, gs, gz = _Dual(0.0), _Dual(0.0), _Dual(0.0)
    for q, s, z, u in zip(Q, S, Z, U):
        probability = (
            _Dual(0.25) * (_Dual(1.0) + x * q) * (_Dual(1.0) + y * s)
            * _Dual(0.5) * (_Dual(1.0) + (2.0 * z - 1.0) * T.LAM * q * s)
            * _Dual(0.5) * (_Dual(1.0) + z * u * th)
        )
        reward = phi * q + (_Dual(1.0) - phi) * s + kappa * z * u
        score_z = _Dual(0.0) if z == 0.0 else (_Dual(u) * sech2) / (_Dual(1.0) + _Dual(u) * th)
        gq = gq + probability * reward * (_Dual(q) - x)
        gs = gs + probability * reward * (_Dual(s) - y)
        gz = gz + probability * reward * score_z
    x2 = (_arctanh_dual(x) + gq * h).tanh()
    y2 = (_arctanh_dual(y) + gs * h).tanh()
    z2 = zeta + gz * h

    def ret(xx, yy, zz):
        return _Dual(preference) * xx + (1.0 - preference) * yy + kappa * zz.tanh() * (_Dual(1.0) + T.LAM * xx * yy) * _Dual(0.5)

    return (ret(x2, y2, z2) - ret(x, y, zeta)) * (1.0 / h)


def _mean_value_upper(box, preference: float, kappa: float, h: float) -> tuple[float, _Dual]:
    x, y, zeta, phi = box
    center = ((x.lo + x.hi) / 2.0, (y.lo + y.hi) / 2.0, (zeta.lo + zeta.hi) / 2.0, (phi.lo + phi.hi) / 2.0)
    exact, _ = step_certificate(*center, kappa, h, (preference, preference))
    dual = _endpoint_gain_dual(
        _Dual(x, 1.0, 0.0, 0.0, 0.0),
        _Dual(y, 0.0, 1.0, 0.0, 0.0),
        _Dual(zeta, 0.0, 0.0, 1.0, 0.0),
        _Dual(phi, 0.0, 0.0, 0.0, 1.0),
        preference, kappa, h,
    )
    radii = ((x.hi - x.lo) / 2.0, (y.hi - y.lo) / 2.0, (zeta.hi - zeta.lo) / 2.0, (phi.hi - phi.lo) / 2.0)
    remainder = 0.0
    for slope, radius in zip((dual.dx, dual.dy, dual.dz, dual.dp), radii):
        remainder += max(abs(slope.lo), abs(slope.hi)) * radius
    return float(np.nextafter(exact + remainder, np.inf)), dual


def propagate_boxes(k_steps: int, budget: float, kappa: float, h: float, phi0: float) -> list[tuple[Iv, Iv, Iv, Iv]]:
    x, y, zeta = Iv(0.8), Iv(0.0), Iv(0.0)
    boxes = []
    for k in range(k_steps):
        phi = Iv(max(0.0, phi0 - k * budget), min(1.0, phi0 + k * budget))
        boxes.append((x, y, zeta, phi))
        x, y, zeta, _ = enclosure_step(x, y, zeta, phi, kappa, h)
    return boxes


def mean_value_upper_bounds(boxes, kappa: float, h: float) -> list[float]:
    bounds = []
    for box in boxes:
        upper_low, _ = _mean_value_upper(box, M.U0[0], kappa, h)
        upper_high, _ = _mean_value_upper(box, M.U0[1], kappa, h)
        # sup min(f, g) <= min(sup f, sup g), and each sup uses a mean-value remainder.
        bounds.append(float(min(upper_low, upper_high)))
    return bounds


def derivative_spot_check(box, kappa: float, h: float) -> float:
    """Largest finite-difference derivative that falls outside the interval enclosure."""
    x, y, zeta, phi = box
    _, dual = _mean_value_upper(box, M.U0[1], kappa, h)
    enclosed = (dual.dx, dual.dy, dual.dz, dual.dp)
    worst = 0.0
    eps = 1e-6
    corners = []
    for sx in (x.lo, x.hi):
        for sy in (y.lo, y.hi):
            for sz in (zeta.lo, zeta.hi):
                for sp in (phi.lo, phi.hi):
                    corners.append((sx, sy, sz, sp))
    for base in corners[::max(1, len(corners) // 4)]:
        for index in range(4):
            plus, minus = list(base), list(base)
            plus[index] += eps
            minus[index] -= eps
            slope = (
                step_certificate(*plus, kappa, h, (M.U0[1], M.U0[1]))[0]
                - step_certificate(*minus, kappa, h, (M.U0[1], M.U0[1]))[0]
            ) / (2 * eps)
            gap = 0.0
            if slope < enclosed[index].lo - 1e-4:
                gap = enclosed[index].lo - slope
            if slope > enclosed[index].hi + 1e-4:
                gap = slope - enclosed[index].hi
            worst = max(worst, gap)
    return float(worst)


def propagate_enclosure(k_steps: int, budget: float, kappa: float, h: float, phi0: float) -> list[float]:
    x, y, zeta = Iv(0.8), Iv(0.0), Iv(0.0)
    bounds = []
    for k in range(k_steps):
        phi = Iv(max(0.0, phi0 - k * budget), min(1.0, phi0 + k * budget))
        x, y, zeta, upper = enclosure_step(x, y, zeta, phi, kappa, h)
        bounds.append(upper)
    return bounds


def clean(value):
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def main() -> None:
    started = time.perf_counter()
    phi0 = float(M.reference_phi0())
    if np.linalg.norm(gradient_np(0.8, 0.0, 0.0, phi0, T.KAPPA) - T.gradient(0.8, 0.0, 0.0, phi0, T.KAPPA)) > 1e-10:
        raise AssertionError("vectorized gradient disagrees with the 16-outcome enumerator")
    out = ROOT / "outputs/horizon_bridge/h1_reachability"
    out.mkdir(parents=True, exist_ok=True)
    kappa = T.KAPPA
    budget = RHO * H

    def source_active(_k: int):
        return M.U0

    scheduled = project(schedule(32, phi0, budget, H), budget)
    reference = rollout(scheduled, kappa, H, source_active)
    if reference["first_loss"] != 0.012:
        raise AssertionError(reference["first_loss"])
    stored = list(csv_source())
    gap = max(abs(reference["source_certificate"][k] - stored[k]) for k in range(32))
    if gap > 1e-9:
        raise AssertionError(gap)

    pieces = feasible_sets(reference["states"][:-1], kappa, H, source_active)
    fixed = fixed_path_reachability(pieces, budget, phi0)
    prefix = fixed_path_reachability(pieces[:24], budget, phi0)
    widths = [None if not item else item[-1][1] - item[0][0] for item in pieces]
    loss_piece = pieces[24]
    reachable_at_loss = phi0 + 24 * budget
    families = family_sweep(scheduled, budget, kappa, H, source_active, phi0)

    starts = [
        scheduled,
        schedule(32, phi0, -budget, H),
        np.full(32, phi0),
        greedy(32, budget, kappa, H, source_active, phi0),
    ]
    rng = np.random.default_rng(20261019)
    for _ in range(4):
        increments = rng.uniform(-budget, budget, 31)
        path = np.empty(32)
        path[0] = phi0
        path[1:] = phi0 + np.cumsum(increments)
        starts.append(path)
    if prefix.get("witness"):
        padded = np.array(scheduled, dtype=float)
        padded[: len(prefix["witness"])] = prefix["witness"]
        starts.append(padded)

    rows = []
    optimized = {}
    for steps, rate, name in (
        (24, RHO, "u0_k24"),
        (25, RHO, "u0_k25"),
        (32, RHO, "u0_k32"),
        (32, 0.0, "u0_rho0"),
        (32, 0.08, "u0_rho08"),
    ):
        this_budget = rate * H
        print(f"optimizing {name}", flush=True)
        found = optimize_coupled(steps, this_budget, kappa, H, source_active, phi0, [item[:steps] for item in starts])
        optimized[name] = found["best"]
        fixed_here = fixed_path_reachability(pieces[:steps], this_budget, phi0)
        row = pack_row(name, steps, rate, found["best"], fixed_here, source_active)
        row["phi_gap_vs_schedule"] = float(np.max(np.abs(found["best"]["phi"] - scheduled[:steps])))
        print(name, row["min_slack"], row["first_loss"], flush=True)
        rows.append(row)

    batch = M.label_batch(phi0, 128, 100, True)
    learned = tuple(batch["interval"])

    def learned_active(k: int):
        return M.U0 if k < M.FEEDBACK_STEP else learned

    learned_pieces = feasible_sets(reference["states"][:-1], kappa, H, learned_active)
    learned_fixed = fixed_path_reachability(learned_pieces, budget, phi0)
    found = optimize_coupled(32, budget, kappa, H, learned_active, phi0, [item for item in starts])
    optimized["u128"] = found["best"]
    row = pack_row("u128", 32, RHO, found["best"], learned_fixed, learned_active)
    row["phi_gap_vs_schedule"] = float(np.max(np.abs(found["best"]["phi"] - scheduled)))
    row["fixed_path"] = {key: value for key, value in learned_fixed.items() if key != "witness"}
    rows.append(row)

    boxes = propagate_boxes(32, budget, kappa, H, phi0)
    natural = propagate_enclosure(32, budget, kappa, H, phi0)
    tight = mean_value_upper_bounds(boxes, kappa, H)
    derivative_gap = derivative_spot_check(boxes[24], kappa, H)
    if derivative_gap > 0:
        raise AssertionError(derivative_gap)
    controls = random_controls(100, 32, budget, phi0, rng)
    violations = []
    state_violations = 0
    for control in controls:
        traced = rollout(control, kappa, H, source_active)
        if np.any(traced["certificate"] > np.array(tight) + 1e-9):
            violations.append(float(np.max(traced["certificate"] - np.array(tight))))
        for k, (sx, sy, sz) in enumerate(traced["states"][:-1]):
            bx, by, bz, bp = boxes[k]
            if not (bx.lo - 1e-12 <= sx <= bx.hi + 1e-12 and by.lo - 1e-12 <= sy <= by.hi + 1e-12 and bz.lo - 1e-12 <= sz <= bz.hi + 1e-12):
                state_violations += 1
            if not (bp.lo - 1e-12 <= control[k] <= bp.hi + 1e-12):
                state_violations += 1
    if state_violations or violations:
        raise AssertionError((state_violations, violations[:3]))
    proved_time = next((k * H for k, upper in enumerate(tight) if upper < GAMMA - 1e-12), None)

    figure(reference, optimized["u0_k32"], tight, out / "reachability.png")
    payload = {
        "reference_first_loss": reference["first_loss"],
        "reference_min_slack": reference["min_slack"],
        "stored_gap": gap,
        "fixed_path": fixed,
        "fixed_path_prefix_24": {key: value for key, value in prefix.items() if key != "witness"},
        "prefix_witness_head": None if not prefix.get("witness") else prefix["witness"][:4],
        "feasible_piece_counts": [len(item) for item in pieces],
        "feasible_set_at_loss": loss_piece,
        "reachable_phi_at_loss": reachable_at_loss,
        "fixed_path_phi_gap": None if not loss_piece else float(loss_piece[0][0] - reachable_at_loss),
        "feasible_widths_head": widths[:8],
        "families": families,
        "learned_interval": list(learned),
        "learned_fixed_path_feasible": learned_fixed["feasible"],
        "rows": rows,
        "natural_enclosure_at_0_012": natural[24],
        "mean_value_upper": tight,
        "derivative_spot_check_gap": derivative_gap,
        "first_proved_all_controls_loss": proved_time,
        "upper_bound_at_0_012": tight[24],
        "upper_bound_width_at_0_012": float(tight[24] - reference["certificate"][24]),
        "crosscheck_controls": len(controls),
    }
    write_summary(out, clean(payload), started)


def family_sweep(scheduled, budget, kappa, h, active, phi0) -> list[dict]:
    rows = []

    def add(name, phi, rate):
        traced = rollout(project(np.array(phi, dtype=float), rate), kappa, h, active)
        rows.append({"name": name, "min_slack": traced["min_slack"], "first_loss": traced["first_loss"]})

    add("maximal", scheduled, budget)
    for alpha in (0.0, 0.25, 0.5, 0.75):
        add(f"constant_fraction_{alpha}", schedule(32, phi0, alpha * budget, h), budget)
    for switch in (8, 16, 24):
        sprint = np.empty(32)
        coast = np.empty(32)
        sprint[0] = coast[0] = phi0
        for k in range(1, 32):
            sprint[k] = np.clip(sprint[k - 1] + (budget if k <= switch else 0.0), 0.0, 1.0)
            coast[k] = np.clip(coast[k - 1] + (0.0 if k <= switch else budget), 0.0, 1.0)
        add(f"sprint_then_coast_{switch}", sprint, budget)
        add(f"coast_then_sprint_{switch}", coast, budget)
    return rows


def pack_row(name, steps, rate, traced, fixed, active) -> dict:
    if isinstance(fixed, dict):
        fixed = {key: value for key, value in fixed.items() if key != "witness"}
    return {
        "name": name,
        "steps": steps,
        "end_time": (steps - 1) * H,
        "rho": rate,
        "min_slack": traced["min_slack"],
        "first_loss": traced["first_loss"],
        "source_first_loss": traced["source_first_loss"],
        "final_phi": float(traced["phi"][-1]),
        "fixed_path": fixed,
    }


def csv_source():
    import csv
    path = ROOT / "outputs/horizon_bridge/step10_g4_extension/exact_steps.csv"
    with path.open() as handle:
        for row in csv.DictReader(handle):
            if row["model"] == "kappa005":
                yield float(row["source_certificate"])


def random_controls(count, steps, budget, phi0, rng) -> list[np.ndarray]:
    paths = []
    for _ in range(count):
        increments = rng.uniform(-budget, budget, steps - 1)
        path = np.empty(steps)
        path[0] = phi0
        path[1:] = phi0 + np.cumsum(increments)
        paths.append(project(path, budget))
    return paths


def figure(reference, best, enclosure, path: Path) -> None:
    times = np.arange(len(reference["certificate"])) * H
    plot, axis = plt.subplots(figsize=(7.2, 3.6))
    axis.plot(times, reference["certificate"], label="scheduled")
    axis.plot(np.arange(len(best["certificate"])) * H, best["certificate"], label="best coupled path found")
    if enclosure is not None:
        axis.plot(times, enclosure, ls="--", label="verified enclosure upper bound")
    axis.axhline(GAMMA, color="black", lw=0.8, label="required gain")
    axis.set_xlabel("time")
    axis.set_ylabel("source finite-step gain")
    axis.legend(fontsize=8)
    plot.tight_layout()
    plot.savefig(path, dpi=140)
    plt.close(plot)


if __name__ == "__main__":
    main()
