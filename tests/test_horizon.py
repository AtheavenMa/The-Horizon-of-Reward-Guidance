from math import inf
import pytest
from horizon import analyze_horizon, certified_gain, minimum_required_speed, support_horizon


def test_exact_theorem_instance():
    assert support_horizon(1.,.6,1.,.12,.04)==pytest.approx(5.)
    assert minimum_required_speed(1.,.6,1.,.12,15.)==pytest.approx(.0933333333333)
    r=analyze_horizon(gain=1.,required_gain=.6,uncertainty_radius=1.,
                      drift_speed=.12,adaptation_speed=.04,target_horizon=15.)
    assert r.pointwise_feasible and not r.trajectory_feasible
    assert r.support_horizon==pytest.approx(5.)
    assert r.source_certificate_at_target==pytest.approx(-.2)


def test_boundary_is_still_certified():
    for t, expected in [(0,1.),(5,.6),(10,.2)]:
        assert certified_gain(1,1,t,.04*t,.12)==pytest.approx(expected)
    r=analyze_horizon(gain=1.,required_gain=.6,uncertainty_radius=1.,
                      drift_speed=.12,adaptation_speed=.04,target_horizon=5.)
    assert r.trajectory_feasible
    with pytest.raises(ValueError):
        support_horizon(1.,1.,1.,.12,.04)


def test_fast_adaptation_and_zero_drift():
    assert support_horizon(1,.6,1,.12,.2)==inf
    assert support_horizon(1,.6,1,0,0)==inf
    for rho in (.12,.2):
        result=analyze_horizon(gain=1.,required_gain=.6,uncertainty_radius=1.,
                               drift_speed=.12,adaptation_speed=rho,target_horizon=100.)
        assert result.trajectory_feasible
        assert result.source_certificate_at_target == pytest.approx(1.)


@pytest.mark.parametrize("params",[(1.,.6,0.,.12,.04),(1.,.6,1.,-.1,.04),
                                   (.6,.6,1.,.12,.04),(1.,.6,1.,.12,-.01),
                                   (1.,-.6,1.,.12,.04)])
def test_invalid_paper_assumptions(params):
    with pytest.raises((ValueError,TypeError)):
        support_horizon(*params)


def test_invalid_target():
    with pytest.raises(ValueError):
        minimum_required_speed(1,.6,1,.12,0)
    with pytest.raises(ValueError):
        certified_gain(1,1,-1,0,.12)
