import pytest
from experiments.strengthening_checks import ordinary_pg_check,clipped_calibration_check


def test_local_all_controls_bound():
    r=ordinary_pg_check()
    assert r["initial_worst_gain"]>.06
    assert r["analytic_pointwise_optimum_lower_in_box"]>.06
    assert r["analytic_minimum_probability_in_box"]>0
    assert r["analytic_ell_dot_lower_in_box"]>.04
    assert 0<r["numerical_maximal_speed_boundary_time"]<r["analytic_all_controls_boundary_time_upper"]
    assert r["analytic_all_controls_boundary_time_upper"]<r["analytic_exit_time_lower"]
    assert r["numerical_maximal_speed_boundary_time"]==pytest.approx(.0078358743,rel=1e-5)


def test_clipped_gradient_startup_cost():
    r=clipped_calibration_check()
    assert r["analytic_deadline"]==pytest.approx(4.9054651081081655)
    assert r["numerical_deadline"]==pytest.approx(r["analytic_deadline"],abs=1e-7)
    assert r["ideal_speed_ball_deadline"]==pytest.approx(5.)
