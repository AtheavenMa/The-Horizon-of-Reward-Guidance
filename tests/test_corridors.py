import numpy as np
import pytest
from scipy.optimize import linprog
from horizon import interval_feasible


def test_feasible_witness():
    r=interval_feasible([0,0.1,0.4,0.9],[0,0.6,0.9,1.1],[.25,.4,.5],initial_value=0)
    assert r.feasible and r.witness_path is not None
    assert r.witness_path[0]==0
    for i,x in enumerate(r.witness_path):
        assert [0,.1,.4,.9][i]-1e-10 <= x <= [0,.6,.9,1.1][i]+1e-10
    assert all(abs(b-a)<=s+1e-10 for a,b,s in zip(r.witness_path,r.witness_path[1:],[.25,.4,.5]))


def test_pointwise_feasible_not_trajectory_feasible():
    r=interval_feasible([0.,2.,4.],[0.,2.,4.],[1.,1.],initial_value=0)
    assert not r.feasible and r.first_infeasible_index==1
    assert r.witness_path is None


def test_start_is_fixed():
    assert not interval_feasible([1.,1.],[2.,2.],[1.],initial_value=0).feasible
    assert interval_feasible([1.,1.],[2.,2.],[1.],initial_value=1.5).feasible


def test_bad_inputs():
    with pytest.raises(ValueError):interval_feasible([],[],[])
    with pytest.raises(ValueError):interval_feasible([0.,1.],[1.],[])
    with pytest.raises(ValueError):interval_feasible([0.],[1.],[1.])
    with pytest.raises(ValueError):interval_feasible([2.],[1.],[])
    with pytest.raises(ValueError):interval_feasible([0.,0.],[1.,1.],[-1.])


def test_randomized_against_independent_linear_program():
    rng=np.random.default_rng(2191)
    for trial in range(50):
        n=7
        increments=rng.uniform(.02,.5,n-1)
        centers=rng.normal(size=n).cumsum()*.15
        width=rng.uniform(.01,.35,n)
        lo=centers-width;hi=centers+width
        lo[0]=hi[0]=centers[0]
        if trial%3==0:lo[-1]+=2.;hi[-1]+=2.
        q=np.zeros((2*(n-1),n))
        for i in range(n-1):
            q[2*i,i]=-1;q[2*i,i+1]=1
            q[2*i+1,i]=1;q[2*i+1,i+1]=-1
        target=linprog(np.zeros(n),A_ub=q,b_ub=np.repeat(increments,2),
                       bounds=list(zip(lo,hi)),method="highs").success
        found=interval_feasible(lo.tolist(),hi.tolist(),increments.tolist(),initial_value=float(centers[0]))
        assert found.feasible==target
        if found.feasible:
            assert np.all(np.abs(np.diff(found.witness_path))<=increments+1e-9)
