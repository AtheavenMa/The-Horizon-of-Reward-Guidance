"""Small, auditable scalar tools for The Limits of Reward Guidance under Finite Adaptation.

These routines implement only the models and assumptions stated in the paper.
They are not an RLHF trainer, a certificate for a neural reward model, or a
solver for general coupled policy--reward control systems.

The scalar model is j_delta(t) = gain + delta * (phi(t) - drift_speed*t),
with delta in [-uncertainty_radius, uncertainty_radius], phi(0)=0, and
|d phi / dt| <= adaptation_speed. Amplitude constraints are assumed not to bind.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil, exp, inf, isfinite, log, sqrt
from typing import Sequence


def _finite(name: str, value: float, *, minimum: float | None = None,
            strict: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite real number")
    value = float(value)
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and (value <= minimum if strict else value < minimum):
        op = ">" if strict else ">="
        raise ValueError(f"{name} must be {op} {minimum}")
    return value


@dataclass(frozen=True)
class HorizonResult:
    support_horizon: float
    minimum_required_speed: float
    target_horizon: float
    pointwise_feasible: bool
    trajectory_feasible: bool
    source_certificate_at_target: float
    required_gain: float
    # "True" means the theorem's amplitude-nonbinding assumption was made.
    amplitude_nonbinding_assumed: bool = True


def support_horizon(gain: float, required_gain: float,
                    uncertainty_radius: float, drift_speed: float,
                    adaptation_speed: float) -> float:
    """Exact scalar support horizon; inf when drift can be matched.

    Preconditions: 0 < required_gain < gain, uncertainty_radius > 0,
    drift_speed and adaptation_speed nonnegative, no binding amplitude limit.
    The certificate still holds at a finite boundary T0 (with equality).
    """
    g = _finite("gain", gain, minimum=0, strict=True)
    gamma = _finite("required_gain", required_gain, minimum=0, strict=True)
    if gamma >= g:
        raise ValueError("required_gain must be strictly less than gain")
    eta = _finite("uncertainty_radius", uncertainty_radius, minimum=0, strict=True)
    v = _finite("drift_speed", drift_speed, minimum=0)
    rho = _finite("adaptation_speed", adaptation_speed, minimum=0)
    return inf if rho >= v else (g - gamma) / (eta * (v - rho))


def minimum_required_speed(gain: float, required_gain: float,
                           uncertainty_radius: float, drift_speed: float,
                           target_horizon: float) -> float:
    """Minimum scalar reward velocity to certify every t in [0,T]."""
    # Reuse validation including required_gain < gain.
    support_horizon(gain, required_gain, uncertainty_radius, drift_speed, 0.)
    T = _finite("target_horizon", target_horizon, minimum=0, strict=True)
    return max(0., float(drift_speed) - (gain-required_gain)/(uncertainty_radius*T))


def certified_gain(gain: float, uncertainty_radius: float, time: float,
                   reward_coefficient: float, drift_speed: float) -> float:
    """Worst-case instantaneous gain over the symmetric scalar source set."""
    g = _finite("gain", gain)
    eta = _finite("uncertainty_radius", uncertainty_radius, minimum=0)
    t = _finite("time", time, minimum=0)
    phi = _finite("reward_coefficient", reward_coefficient)
    v = _finite("drift_speed", drift_speed, minimum=0)
    return g - eta*abs(phi-v*t)


def analyze_horizon(*, gain: float, required_gain: float,
                    uncertainty_radius: float, drift_speed: float,
                    adaptation_speed: float, target_horizon: float) -> HorizonResult:
    """Summary for the scalar linear-target theorem, not a generic RL policy."""
    T = _finite("target_horizon", target_horizon, minimum=0, strict=True)
    T0 = support_horizon(gain,required_gain,uncertainty_radius,drift_speed,adaptation_speed)
    req = minimum_required_speed(gain,required_gain,uncertainty_radius,drift_speed,T)
    # Admissible maximally tracking path phi=min(v,rho)t.
    velocity = min(drift_speed,adaptation_speed)
    attained = certified_gain(gain,uncertainty_radius,T,velocity*T,drift_speed)
    return HorizonResult(T0, req, T, True,
                         attained >= required_gain - 1e-12*max(1.,abs(gain)),
                         attained, float(required_gain))


@dataclass(frozen=True)
class CorridorResult:
    feasible: bool
    witness_path: tuple[float, ...] | None
    first_infeasible_index: int | None


def interval_feasible(lower: Sequence[float], upper: Sequence[float],
                      displacement_budgets: Sequence[float],
                      *, initial_value: float | None = None) -> CorridorResult:
    """Exact feasibility and witness for a *discrete*, externally given corridor.

    For intervals [lower[k], upper[k]], seek x[k] therein such that
    |x[k+1]-x[k]| <= displacement_budgets[k]. If initial_value is given,
    x[0] is fixed. No assertion is made about times *between* these nodes.
    """
    if len(lower) == 0 or len(lower) != len(upper):
        raise ValueError("lower and upper must have the same nonzero length")
    if len(displacement_budgets) != len(lower)-1:
        raise ValueError("need exactly n-1 displacement budgets")
    lows = tuple(_finite(f"lower[{i}]",v) for i,v in enumerate(lower))
    highs = tuple(_finite(f"upper[{i}]",v) for i,v in enumerate(upper))
    budgets = tuple(_finite(f"displacement_budgets[{i}]",v,minimum=0)
                    for i,v in enumerate(displacement_budgets))
    if any(lo > hi for lo,hi in zip(lows,highs)):
        raise ValueError("every interval must have lower <= upper")
    if initial_value is not None:
        init = _finite("initial_value",initial_value)
        if not lows[0] <= init <= highs[0]:
            return CorridorResult(False,None,0)
        starts = (init, init)
    else:
        starts = (lows[0], highs[0])
    reachable = [starts]
    for i, s in enumerate(budgets,1):
        left = max(lows[i],reachable[-1][0]-s)
        right = min(highs[i],reachable[-1][1]+s)
        if left > right + 1e-12:
            return CorridorResult(False,None,i)
        if left > right:
            left = right  # only numerical roundoff
        reachable.append((left,right))
    # Backward recover a path. The interval of states that can reach the
    # chosen successor is [successor-s, successor+s].
    path = [0.]*len(lows)
    path[-1] = (reachable[-1][0]+reachable[-1][1])/2.
    for i in range(len(lows)-2,-1,-1):
        a,b=reachable[i]
        path[i]=min(max(path[i+1],a),b)
        if abs(path[i]-path[i+1]) > budgets[i] + 1e-10:
            raise AssertionError("internal witness construction error")
    return CorridorResult(True,tuple(path),None)


def _sigmoid(x: float) -> float:
    return 1/(1+exp(-x)) if x >= 0 else exp(x)/(1+exp(x))


def _logit(p: float) -> float:
    return log(p/(1-p))


def _binary_kl(p: float, q: float) -> float:
    return p*log(p/q)+(1-p)*log((1-p)/(1-q))


@dataclass(frozen=True)
class BTLBounds:
    gap_to_boundary: float
    necessary_expected_labels: float
    sufficient_fixed_labels: int
    information_lower_bound: float
    curvature_lower_bound: float
    alpha: float


def btl_sample_bounds(*, gap_to_boundary: float, alpha: float = 0.05,
                      sensitivity: float = 1., max_sensitivity: float | None = None,
                      uncertainty_radius: float = 1., intercept: float = 0.) -> BTLBounds:
    """Paper's KL necessary *expected* count and Hoeffding sufficient batch.

    Assumes Bernoulli(sigmoid(intercept + q*delta)), |delta| <= eta, a
    fixed informative pure comparison of positive q, and adaptive queries
    with absolute sensitivity <= Q. Lower counts are expectations, not a
    per-dataset stopping guarantee. The returned upper count is sufficient
    only under the stated bounded-parameter and reliability conditions.
    """
    gap = _finite("gap_to_boundary",gap_to_boundary,minimum=0,strict=True)
    a = _finite("alpha",alpha,minimum=0,strict=True)
    if a >= .25:
        raise ValueError("alpha must be < 1/4 for the stated comparison")
    q = _finite("sensitivity",sensitivity,minimum=0,strict=True)
    Q = q if max_sensitivity is None else _finite("max_sensitivity",max_sensitivity,minimum=0,strict=True)
    if Q < q:
        raise ValueError("max_sensitivity must cover sensitivity")
    eta = _finite("uncertainty_radius",uncertainty_radius,minimum=0,strict=True)
    c = _finite("intercept",intercept)
    zmax = max(abs(c-q*eta),abs(c+q*eta))
    prob = _sigmoid(zmax)
    kappa = prob*(1-prob)
    if kappa <= 0:
        raise ValueError("parameter bound is too large for stable curvature")
    info_lower = 8*_binary_kl(1-a,a)/(gap*gap)
    n_lower = info_lower/(Q*Q)
    n_upper = ceil(2*log(1/a)/(kappa*q*gap)**2)
    return BTLBounds(gap,n_lower,n_upper,info_lower,kappa,a)


@dataclass(frozen=True)
class BTLCertificate:
    certified: bool
    successes: int
    labels: int
    critical_win_probability: float
    empirical_win_probability: float
    one_sided_probability_radius: float
    acceptance_threshold_wins: int


def certify_btl_batch(*, successes: int, labels: int,
                      critical_delta: float, sensitivity: float,
                      alpha: float = .05, intercept: float = 0.) -> BTLCertificate:
    """Fixed-N, one-sided test of delta < critical_delta with q > 0.

    Certificate is the observed Hoeffding rejection of the infeasible null
    delta >= critical_delta. This assumes precommitted n, independent labels
    from one fixed comparison, and the declared logistic model; no optional
    stopping or adaptive-query guarantee is claimed.
    """
    if isinstance(labels,bool) or not isinstance(labels,int) or labels<=0:
        raise ValueError("labels must be a positive integer")
    if isinstance(successes,bool) or not isinstance(successes,int) or not 0<=successes<=labels:
        raise ValueError("successes must be an integer between 0 and labels")
    q = _finite("sensitivity",sensitivity,minimum=0,strict=True)
    delta_c = _finite("critical_delta",critical_delta)
    c = _finite("intercept",intercept)
    a = _finite("alpha",alpha,minimum=0,strict=True)
    if a>=.5:
        raise ValueError("alpha must be < 1/2")
    pc = _sigmoid(c+q*delta_c)
    radius=sqrt(log(1/a)/(2*labels))
    threshold=int((labels*(pc-radius))//1)
    return BTLCertificate(successes <= threshold,successes,labels,pc,successes/labels,radius,threshold)


@dataclass(frozen=True)
class BTLInterval:
    lower_delta: float
    upper_delta: float
    probability_radius: float
    confidence: float


def btl_confidence_interval(*, successes: int, labels: int,
                            sensitivity: float, uncertainty_radius: float,
                            alpha: float = .05, intercept: float = 0.) -> BTLInterval:
    """Two-sided fixed-N Hoeffding interval truncated to the known source set.

    No guarantees for optional stopping; comparisons must be i.i.d. from the
    same fixed logistic pair and independent of the stopping decision.
    """
    # Reuse integer validation without interpreting the test result.
    certify_btl_batch(successes=successes,labels=labels,critical_delta=0.,
                      sensitivity=sensitivity,alpha=alpha,intercept=intercept)
    eta=_finite("uncertainty_radius",uncertainty_radius,minimum=0,strict=True)
    q=_finite("sensitivity",sensitivity,minimum=0,strict=True)
    c=_finite("intercept",intercept)
    a=_finite("alpha",alpha,minimum=0,strict=True)
    rad=sqrt(log(2/a)/(2*labels))
    p=successes/labels
    pmin=max(_sigmoid(c-q*eta),p-rad)
    pmax=min(_sigmoid(c+q*eta),p+rad)
    if pmin > pmax:
        # Empty confidence set can occur for inconsistent observations; do
        # not silently present a certified interval.
        raise ValueError("confidence interval has empty intersection with source set")
    return BTLInterval((_logit(pmin)-c)/q,(_logit(pmax)-c)/q,rad,1-a)
