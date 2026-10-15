"""Added in v1.0: discrete resource experiment, not a new general theorem.

The five-outcome model and source ambiguity are inherited from v0.6.
Policy takes a fixed Euler step. Guidance performs m actual clipped quadratic
calibration steps before each policy step; these are NOT BTL likelihood steps.
All observed returns are exact for this affine probability model.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
from scipy.linalg import helmert
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    eps, diagonal, coupling = .04, 4., .75
    h, eta, G, gamma, eta0, max_time = .05, .01, .2, .6, 1., 15.
    v = coupling * eps * diagonal
    H = helmert(5, full=False).T
    rows, summaries = [], []
    plt.figure(figsize=(6.7, 3.65))
    max_identity_error = 0.
    for m in (0, 1, 2, 3):
        theta = np.zeros(4)
        phi = 0.
        prev_phi = 0.
        gains, times = [], []
        min_prob, min_eig = 1., float('inf')
        first_failed = None
        max_move = 0.
        for k in range(round(max_time / h)):
            t = k*h
            c = coupling*theta[2]
            assert abs(c-v*t) < 5e-12
            for _ in range(m):
                grad = float(np.clip(phi-c, -G, G))
                phi = float(np.clip(phi-eta*grad, -2., 2.))
            move = abs(phi-prev_phi)
            max_move = max(max_move, move)
            assert move <= m*eta*G + 1e-12
            prev_phi = phi
            K=np.array([[1., 0.], [-c, 1.]])
            M=np.block([[diagonal*np.eye(2),K],[K.T,diagonal*np.eye(2)]])
            coeff=np.array([0.,0.,1.,phi])
            velocity=eps*M@coeff
            p=.2+eps*H@theta
            next_theta=theta+h*velocity
            next_p=.2+eps*H@next_theta
            assert np.min(p)>0 and np.min(next_p)>0
            min_prob=min(min_prob,float(p.min()),float(next_p.min()))
            min_eig=min(min_eig,float(np.linalg.eigvalsh(M).min()))
            actual=[]
            for delta in (-1.,0.,1.):
                r=H[:,0]+delta*H[:,1]
                gain=float(r@(next_p-p)/(eps**2*h))
                expected=1.+delta*(phi-c)
                err=abs(gain-expected)
                max_identity_error=max(max_identity_error,err)
                assert err<1e-10
                actual.append(gain)
            worst=1.-eta0*abs(phi-c)
            if first_failed is None and worst < gamma-1e-12:
                first_failed=t
            times.append(t);gains.append(worst)
            rows.append([m,k,t,c,phi,move,worst,*actual,min_prob,min_eig])
            theta=next_theta
        step_budget=m*eta*G
        ideal_failure=None
        for k in range(round(max_time/h)):
            ideal_phi=min(v*h*k,step_budget*k)
            ideal_worst=1.-eta0*abs(ideal_phi-v*h*k)
            if ideal_worst < gamma-1e-12:
                ideal_failure=k*h
                break
        plt.plot(times,gains,label=f'{m} reward steps / policy step')
        summaries.append({'reward_steps_per_policy_step':m,
            'max_reward_displacement_per_policy_step':step_budget,
            'capacity_per_policy_time':step_budget/h,
            'observed_max_displacement':max_move,
            'first_uncertified_policy_step_time':first_failed,
            'ideal_budget_envelope_first_uncertified_time':ideal_failure,
            'minimum_output_probability':min_prob,
            'minimum_preconditioner_eigenvalue':min_eig})
    plt.axhline(gamma,ls='--',label='Required normalized progress')
    plt.xlabel('Policy time (fixed step size 0.05)')
    plt.ylabel('Normalized worst-case finite-step gain')
    plt.legend(fontsize=8,loc='lower left')
    plt.tight_layout();plt.savefig(out/'discrete_resources.png',dpi=180,bbox_inches='tight');plt.close()
    with (out/'discrete_resources.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['reward_steps','iteration','time','c','phi','reward_displacement',
            'worst_normalized_gain','gain_delta_minus1','gain_delta_0','gain_delta_1','min_prob_so_far','min_M_eigenvalue_so_far']);w.writerows(rows)
    result={'version_added':'1.0','inherited_model':'v0.6 five-output policy',
        'h_policy':h,'reward_step_size':eta,'gradient_clip':G,'g':1.,'gamma':gamma,'source_radius':eta0,
        'policy_drift_v':v,'time_window':[0.,max_time],
        'iterations_per_run':round(max_time/h),'deterministic':True,
        'runs':summaries,'maximum_exact_step_identity_error':max_identity_error,
        'scope':'Exact stepwise mean-gain test in an affine finite-output policy; not continuous interpolation, BTL fitting, PPO, or a neural experiment.'}
    (out/'discrete_resources.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path('results'))
    print(json.dumps(run(p.parse_args().out),indent=2))
