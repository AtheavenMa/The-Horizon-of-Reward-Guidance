"""Reproduce focused checks for the v0.5-review response.

These checks complement the local analytic derivations in the included paper PDF.
They do not establish novelty or validate neural-RM behavior.
Requires NumPy and SciPy; no GPU or external data.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import expit


def ordinary_pg_check() -> dict[str, Any]:
    """A full-support ordinary-Euclidean-PG robust transversality example."""
    gamma, rho = 0.06, 0.04
    x0, y0 = 0.8, 0.0
    A0 = (1 - x0*x0)**2
    ell0 = (1 + gamma)/(2*A0 + 1)
    phi0 = ell0 + 0.001

    # Analytic enclosure of a neighborhood, not a sampled-grid certificate.
    xl, xu, yl, yu, pl, pu = .795, .81, -.005, .02, .835, .86
    Amin, Amax = (1-xu*xu)**2, (1-xl*xl)**2
    Bmin, Bmax = (1-max(abs(yl), abs(yu))**2)**2, 1.
    assert 2*Amin > gamma
    positive_numerator_lower = 8*xl*(1-xu*xu)*Amin*pl*(Bmin+gamma)
    negative_numerator_upper = (2*Amax-gamma)*4*yu*Bmax*(1-pl)
    ell_dot_lower = (positive_numerator_lower-negative_numerator_upper)/(2*Amax+Bmax)**2
    kappa_lower = Amin*Bmin/(Amin+Bmin)
    fx_upper, fy_upper = Amax*pu, Bmax*(1-pl)
    exit_time_lower = min((xu-x0)/fx_upper, (yu-y0)/fy_upper,
                          (phi0-pl)/rho, (pu-phi0)/rho)
    fail_time_upper = .001/(ell_dot_lower-rho)
    assert ell_dot_lower > rho
    assert kappa_lower > gamma
    assert fail_time_upper < exit_time_lower

    def f(_t: float, z: np.ndarray) -> list[float]:
        x, y, phi = z
        return [(1-x*x)**2*phi, (1-y*y)**2*(1-phi), rho]

    def event(_t: float, z: np.ndarray) -> float:
        x, y, phi = z
        A, B = (1-x*x)**2, (1-y*y)**2
        return phi-(B+gamma)/(2*A+B)

    event.terminal = True
    event.direction = -1
    sol = solve_ivp(f, (0, .1), (x0, y0, phi0), events=event,
                    rtol=1e-11, atol=1e-13, max_step=1e-3)
    if not sol.success or not len(sol.t_events[0]):
        raise RuntimeError("Failed to locate the maximal-speed path boundary.")
    boundary_time = float(sol.t_events[0][0])
    assert boundary_time <= fail_time_upper
    return {
        "initial_means": [x0, y0], "initial_phi": phi0,
        "required_gain": gamma, "reward_speed_bound": rho,
        "initial_worst_gain": min(2*A0*phi0-(1-phi0), -A0*phi0+2*(1-phi0)),
        "initial_pointwise_optimum": A0/(A0+1),
        "initial_safe_interval": [ell0, (2-gamma)/(A0+2)],
        "analytic_ell_dot_lower_in_box": ell_dot_lower,
        "analytic_pointwise_optimum_lower_in_box": kappa_lower,
        "analytic_minimum_probability_in_box": (1-xu)*(1-max(abs(yl), abs(yu)))/4,
        "analytic_exit_time_lower": exit_time_lower,
        "analytic_all_controls_boundary_time_upper": fail_time_upper,
        "numerical_maximal_speed_boundary_time": boundary_time,
        "qualifier": "All-controls bound is local to the stated initial point and box; not a general global optimal-control theorem."
    }


def clipped_calibration_check() -> dict[str, float]:
    """Check a realizable clipped quadratic-gradient rule, not arbitrary BTL retraining."""
    v, rho, k, width = .12, .04, 1.0, .4
    ts = np.log(v/(v-rho))/k
    deadline = ts + (width-rho/k)/(v-rho)
    assert width > rho/k
    def f(t: float, z: np.ndarray) -> list[float]:
        return [float(np.clip(k*(v*t-z[0]), -rho, rho))]
    def event(t: float, z: np.ndarray) -> float:
        return v*t-z[0]-width
    event.terminal = True
    event.direction = 1
    sol = solve_ivp(f, (0, 8), [0.], events=event,
                    rtol=1e-11, atol=1e-13, max_step=.001)
    observed = float(sol.t_events[0][0])
    assert abs(observed-deadline) < 1e-7
    return {"saturation_time": float(ts), "analytic_deadline": float(deadline),
            "numerical_deadline": observed, "ideal_speed_ball_deadline": width/(v-rho)}


def bounded_gradient_and_kl_check() -> dict[str, float]:
    rng = np.random.default_rng(20260918)
    max_step_ratio, max_kl_excess = 0., -float("inf")
    for _ in range(10000):
        phi = rng.uniform(-1, 1)
        d = rng.uniform(-4, 4, size=16)
        c = rng.uniform(-2, 2, size=16)
        Y = rng.integers(0, 2, size=16)
        grad = float(np.mean((expit(c+phi*d)-Y)*d))
        eta = .01
        newphi = float(np.clip(phi-eta*grad, -1, 1))
        max_step_ratio = max(max_step_ratio, abs(newphi-phi)/(eta*4))
        # Stable logistic-family KL: KL(P_a || P_b).
        a, shift = rng.normal(), rng.normal()
        kl = np.logaddexp(0., a+shift)-np.logaddexp(0., a)-expit(a)*shift
        max_kl_excess = max(max_kl_excess, float(kl-shift*shift/8))
    assert max_step_ratio <= 1+1e-12
    assert max_kl_excess <= 1e-12
    return {"largest_observed_step_over_bound": max_step_ratio,
            "largest_kl_minus_quadratic_bound": max_kl_excess,
            "trials": 10000}


def main() -> None:
    out = {"ordinary_pg": ordinary_pg_check(),
           "clipped_calibration": clipped_calibration_check(),
           "gradient_and_kl": bounded_gradient_and_kl_check()}
    path = Path(__file__).parent/'verification.json'
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
