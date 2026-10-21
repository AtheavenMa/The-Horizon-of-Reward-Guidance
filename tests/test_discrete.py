from pathlib import Path
import pytest
from experiments.discrete_resource_check import run


def test_original_discrete_resource_model(tmp_path:Path):
    r=run(tmp_path)
    assert len(r["runs"])==4
    times=[x["first_uncertified_policy_step_time"] for x in r["runs"]]
    assert times[:3]==pytest.approx([3.35,4.55,7.85])
    assert times[3] is None
    assert r["maximum_exact_step_identity_error"]<1e-10
    assert min(x["minimum_output_probability"] for x in r["runs"])>.1
    assert all(x["minimum_preconditioner_eigenvalue"]>0 for x in r["runs"])
