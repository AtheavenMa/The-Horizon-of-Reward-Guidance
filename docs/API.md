# Python API reference

The public module [`horizon.py`](../horizon.py) provides scalar horizon analysis, interval feasibility, and Bradley–Terry certification utilities using the Python standard library.

## `analyze_horizon(...)`

```python
from horizon import analyze_horizon

r = analyze_horizon(
    gain=1.0, required_gain=0.6, uncertainty_radius=1.0,
    drift_speed=0.12, adaptation_speed=0.04,
    target_horizon=15.0,
)
assert r.support_horizon == 5.0
assert r.pointwise_feasible
assert not r.trajectory_feasible
```

All arguments are keyword-only, finite real numbers. `gain > required_gain > 0`, `uncertainty_radius > 0`, `drift_speed >= 0`, `adaptation_speed >= 0`, `target_horizon > 0`. The returned frozen `HorizonResult` contains `support_horizon`, `minimum_required_speed`, `target_horizon`, `pointwise_feasible`, `trajectory_feasible`, `source_certificate_at_target`, `required_gain`, and a flag naming the **nonbinding amplitude assumption**.

For the scalar linear moving-target model, the horizon is `math.inf` when adaptation can match drift and reward-amplitude constraints are nonbinding.

## `support_horizon(...)` and `minimum_required_speed(...)`

```python
from horizon import support_horizon, minimum_required_speed

T0 = support_horizon(1.0, 0.6, 1.0, 0.12, 0.04)
rho = minimum_required_speed(1.0, 0.6, 1.0, 0.12, 15.0)
```

These expressions are exact under the specified parameter and initialization assumptions.

## `certified_gain(...)`

```python
from horizon import certified_gain

gain_t5 = certified_gain(1.0, 1.0, 5.0, 0.04 * 5.0, 0.12)
assert abs(gain_t5 - 0.6) < 1e-12
```

Computes $g-\eta_0|\phi(t)-vt|$ at a specified time and reward coefficient.

## `interval_feasible(...)`

```python
from horizon import interval_feasible

result = interval_feasible(
    lower=[0.0, 0.1, 0.4],
    upper=[0.0, 0.4, 0.8],
    displacement_budgets=[0.2, 0.4],
    initial_value=0.0,
)
assert result.feasible
print(result.witness_path)
```

Given **discrete** intervals of equal length and exactly `n-1` budgets, tests whether a path exists with `|phi[k+1]-phi[k]| <= budget[k]`. Returns a frozen `CorridorResult` containing `feasible`, `witness_path` (or `None`), and `first_infeasible_index` (or `None`). An optional `initial_value` fixes the first point.

The witness covers the supplied discrete time nodes.

## `btl_sample_bounds(...)`

```python
from horizon import btl_sample_bounds

b = btl_sample_bounds(
    gap_to_boundary=0.2, alpha=0.05,
    sensitivity=1.0, max_sensitivity=1.0,
    uncertainty_radius=1.0,
)
print(b.necessary_expected_labels, b.sufficient_fixed_labels)
```

Arguments `gap_to_boundary>0`, `0<alpha<1/4`, fixed pure-comparison sensitivity `q>0`, max adaptive query sensitivity `Q>=q`, source interval radius `eta>0`, and fixed logistic `intercept`. Returns the KL information lower bound, **expected** necessary number of labels given max sensitivity, and a conservative sufficient **fixed** count for the identified pure comparison.

## `certify_btl_batch(...)`

```python
from math import sqrt
from horizon import certify_btl_batch

r = certify_btl_batch(
    successes=600, labels=1200, critical_delta=1/3,
    sensitivity=sqrt(2/3), alpha=0.05,
)
assert r.certified
assert r.acceptance_threshold_wins == 638
```

Tests against null $\delta\ge\delta_c$ for **independent observations** from one fixed pure BTL pair with positive `sensitivity`. Rejecting the null is evidence of feasibility in this one-dimensional model at the fixed declared error bound. The test uses a fixed sample size and a precommitted comparison pair.

## `btl_confidence_interval(...)`

```python
from math import sqrt
from horizon import btl_confidence_interval

i = btl_confidence_interval(
    successes=600, labels=1200,
    sensitivity=sqrt(2/3), uncertainty_radius=1., alpha=.05,
)
print(i.lower_delta, i.upper_delta)  # approximately -0.19246, +0.19246
```

Constructs the fixed-size two-sided Hoeffding interval, intersected with the known source-compatible scalar interval. An empty intersection raises `ValueError`.
