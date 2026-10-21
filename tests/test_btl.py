import math
import pytest
from horizon import btl_confidence_interval,btl_sample_bounds,certify_btl_batch


def test_paper_two_sided_batch():
    i=btl_confidence_interval(successes=600,labels=1200,sensitivity=math.sqrt(2/3),
                              uncertainty_radius=1.,alpha=.05)
    assert i.lower_delta==pytest.approx(-.19245962944052997)
    assert i.upper_delta==pytest.approx(.19245962944053022)
    assert 1-.19245962944053022*1.2==pytest.approx(.7690484446713637)
    # The full paper closed loop includes the pre-query lower certificate at t=4.
    assert 1-.08*4==pytest.approx(.68)


def test_one_sided_fixed_batch_rule():
    q=math.sqrt(2/3)
    delta_c=.4/(.08*15)
    r=certify_btl_batch(successes=600,labels=1200,sensitivity=q,
                        critical_delta=delta_c,alpha=.05)
    assert r.acceptance_threshold_wins==638
    assert r.certified
    assert certify_btl_batch(successes=638,labels=1200,sensitivity=q,
                             critical_delta=delta_c,alpha=.05).certified
    assert not certify_btl_batch(successes=639,labels=1200,sensitivity=q,
                                 critical_delta=delta_c,alpha=.05).certified


def test_complexity_gap_scaling():
    b1=btl_sample_bounds(gap_to_boundary=.2)
    b2=btl_sample_bounds(gap_to_boundary=.1)
    assert b2.necessary_expected_labels==pytest.approx(4*b1.necessary_expected_labels)
    assert b2.sufficient_fixed_labels==pytest.approx(4*b1.sufficient_fixed_labels,abs=4.)
    assert b1.sufficient_fixed_labels > b1.necessary_expected_labels


def test_reject_outside_parameter_assumptions():
    for kwargs in [dict(gap_to_boundary=-1),dict(gap_to_boundary=.1,alpha=.3),
                   dict(gap_to_boundary=.1,sensitivity=2,max_sensitivity=1)]:
        with pytest.raises(ValueError):btl_sample_bounds(**kwargs)
    for successes,labels in [(-1,10),(11,10),(1,0),(1.5,10)]:
        with pytest.raises(ValueError):
            certify_btl_batch(successes=successes,labels=labels,critical_delta=.3,sensitivity=1.)
