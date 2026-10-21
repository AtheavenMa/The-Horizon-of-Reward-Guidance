"""Independent numerical checks for the v0.6 finite-model theory.

Run: python code/verify_v06.py --out results
No network, GPU, training corpus, or gold labels beyond the declared synthetic models.
Analytic proofs will be in the forthcoming arXiv preprint; numerical checks do not replace them.
"""
from __future__ import annotations
import argparse
import csv
import itertools
import json
import math
from pathlib import Path
from typing import Any
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import linprog
from scipy.special import expit, xlogy
from scipy.stats import binom
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import legacy_checks as old
import strengthening_checks as extra


def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches='tight')
    plt.close()


def weighted_corridors(rng: np.random.Generator) -> dict[str, Any]:
    """Nonuniform discrete budgets: all-pairs criterion vs independent LP."""
    counts = [0, 0]
    for trial in range(180):
        n = 9
        s = rng.uniform(.01, .4, n-1)
        metric = np.abs(np.r_[0, np.cumsum(s)][:, None]-np.r_[0, np.cumsum(s)][None, :])
        path = np.r_[0, np.cumsum(rng.uniform(-1, 1, n-1)*s)]
        width = rng.uniform(.005, .25, n)
        lo, hi = path-width, path+width
        if trial % 3 == 0:
            j = rng.integers(2, n)
            lo[j] += 1.5
            hi[j] += 1.5
        lo[0] = hi[0] = 0.
        criterion = bool(np.all(lo[:, None] <= hi[None, :]+metric+1e-11))
        Q, rhs = [], []
        for k in range(n-1):
            row = np.zeros(n); row[k+1] = 1; row[k] = -1
            Q.extend([row, -row]); rhs.extend([s[k], s[k]])
        fit = linprog(np.zeros(n), A_ub=np.array(Q), b_ub=np.array(rhs),
                      bounds=list(zip(lo, hi)), method='highs')
        assert criterion == fit.success
        if criterion:
            env = np.max(lo[:, None]-metric, axis=0)
            assert np.all(env >= lo-1e-10) and np.all(env <= hi+1e-10)
            assert np.all(np.abs(np.diff(env)) <= s+1e-10)
        counts[int(criterion)] += 1
    return {'trials': 180, 'feasible': counts[1], 'infeasible': counts[0], 'mismatches': 0}


def ordinary_pg_plot(out: Path) -> None:
    rho, gam = .04, .06
    a0 = (1-.8**2)**2
    phi0 = (1+gam)/(2*a0+1)+.001
    def rhs(t: float, z: np.ndarray) -> list[float]:
        x,y,p=z
        return [(1-x*x)**2*p, (1-y*y)**2*(1-p), rho]
    t=np.linspace(0,.018,401)
    sol=solve_ivp(rhs,(0,t[-1]),[.8,0,phi0],t_eval=t,rtol=1e-11,atol=1e-13,max_step=.0001)
    if not sol.success: raise RuntimeError(sol.message)
    x,y,phi=sol.y
    a=(1-x*x)**2;b=(1-y*y)**2
    lower=(b+gam)/(2*a+b);upper=(2*b-gam)/(a+2*b)
    gain=np.minimum(2*a*phi-b*(1-phi),-a*phi+2*b*(1-phi))
    point=a*b/(a+b)
    np.savetxt(out/'ordinary_pg.csv',np.c_[t,x,y,phi,lower,upper,gain,point],delimiter=',',
               header='time,x,y,phi,safe_lower,safe_upper,worst_gain,pointwise_optimum',comments='')
    plt.figure(figsize=(6.5,3.6))
    plt.plot(t,phi,label='Actual reward: maximum permitted speed')
    plt.plot(t,lower,'--',label='Moving lower boundary')
    plt.axvline(.0078358743,ls=':',label='Observed boundary contact')
    plt.xlabel('Policy-gradient flow time');plt.ylabel('Reward coefficient')
    plt.legend(fontsize=8);savefig(out/'ordinary_pg.png')


def clipped_plot(out: Path) -> None:
    v,rho,k=.12,.04,1.
    ts=np.log(v/(v-rho))/k
    t=np.linspace(0,7,500)
    e=np.where(t<=ts,v/k*(-np.expm1(-k*t)),rho/k+(v-rho)*(t-ts))
    plt.figure(figsize=(6.5,3.6))
    plt.plot(t,(v-rho)*t,label='Ideal rate-budget controller')
    plt.plot(t,e,label='Clipped gradient calibration')
    plt.axhline(.4,ls='--',label='Maximum source-certified lag')
    plt.xlabel('Policy-gradient flow time');plt.ylabel('Uncompensated lag')
    plt.legend(fontsize=8);savefig(out/'actual_adaptation.png')


def kl_bern(p: float, q: float) -> float:
    return float(xlogy(p,p/q)+xlogy(1-p,(1-p)/(1-q)))


def binary_gap_checks(out: Path) -> dict[str, Any]:
    rho,v,Delta,eta0,alpha=.04,.12,.4,1.,.05
    q=math.sqrt(2/3)
    kerr=kl_bern(1-alpha,alpha)
    rows=[]
    for T in [5.1,6.,10.,15.,25.]:
        a=(v-rho)*T;dc=Delta/a;pc=expit(q*dc)
        for ds in [0.,-.2,.2]:
            if dc-ds<1e-10: continue
            gap=dc-ds; ps=expit(q*ds); pg=pc-ps
            lower=8*kerr/(q*q*gap*gap)
            exact_kl_lower=kerr/kl_bern(ps,pc)
            N=math.ceil(2*math.log(1/alpha)/(pg*pg))
            rad=math.sqrt(math.log(1/alpha)/(2*N))
            threshold=math.floor(N*(pc-rad)+1e-10)
            power=float(binom.cdf(threshold,N,ps))
            boundary_accept=float(binom.cdf(threshold,N,pc))
            assert power>=1-alpha-1e-10
            assert boundary_accept<=alpha+1e-10
            assert lower<=exact_kl_lower+1e-7
            rows.append([T,ds,dc,gap,lower,exact_kl_lower,N,power,boundary_accept])
    with (out/'binary_gap_bounds.csv').open('w') as f:
        wr=csv.writer(f);wr.writerow(['T','true_delta','critical_delta','gap','quadratic_label_lower',
            'exact_single_pair_kl_lower','hoeffding_upper_N','power','null_boundary_accept']);wr.writerows(rows)
    plt.figure(figsize=(6.5,3.8))
    for ds in [0.,-.2,.2]:
        r=np.array([a for a in rows if a[1]==ds])
        plt.plot(r[:,0],r[:,4],marker='o',label=f'Lower bound, true delta = {ds:g}')
    plt.yscale('log');plt.xlabel('Required horizon T');plt.ylabel('Necessary expected labels')
    plt.legend(fontsize=8);savefig(out/'instance_complexity.png')
    # Concrete fixed-N one-sided certification at T=15 and delta*=0.
    T,N=15.,1200
    dc=Delta/((v-rho)*T);pc=expit(q*dc)
    rad=math.sqrt(math.log(1/alpha)/(2*N))
    threshold=math.floor(N*(pc-rad)+1e-10)
    power=float(binom.cdf(threshold,N,.5))
    null_accept=float(binom.cdf(threshold,N,pc))
    # Source-like query law is independent of delta, hence exact zero KL.
    source_kl=kl_bern(float(expit(math.sqrt(2))),float(expit(math.sqrt(2))))
    return {'fixed_design_checks':len(rows),'smallest_power':min(r[7] for r in rows),
        'largest_boundary_accept_probability':max(r[8] for r in rows),
        'one_sided_1200_labels':{'T':T,'alpha':alpha,'threshold_wins':threshold,'power_at_zero':power,
        'accept_probability_at_boundary':null_accept,'probability_radius':rad},
        'source_query_KL':source_kl,
        'qualifier':'The table concerns fixed-N tests; unknown-gap sequential stopping uses a separate anytime construction.'}


def connected(block: list[int], edges: list[tuple[int,int]]) -> bool:
    if len(block)<=1:return True
    seen={block[0]}; allowed=set(block)
    for _ in block:
        for i,j in edges:
            if i in allowed and j in allowed and ((i in seen) or (j in seen)):
                seen.update([i,j])
    return seen==allowed


def restricted_graph_cost(W: np.ndarray, edges: list[tuple[int,int]]) -> int | None:
    n=W.shape[0];best=-1
    for blocks in old.partitions(n):
        if all(np.max(np.abs(np.sum(W[list(bl)],axis=0)))<1e-9 and connected(list(bl),edges) for bl in blocks):
            best=max(best,len(blocks))
    return None if best<0 else n-best


def brute_restricted(W: np.ndarray, edges: list[tuple[int,int]]) -> int | None:
    n=W.shape[0]
    for k in range(len(edges)+1):
        for es in itertools.combinations(edges,k):
            parent=list(range(n))
            def root(i):
                while parent[i]!=i:i=parent[i]
                return i
            for i,j in es:parent[root(i)]=root(j)
            groups={}
            for i in range(n):groups.setdefault(root(i),[]).append(i)
            if all(np.max(np.abs(np.sum(W[bl],axis=0)))<1e-9 for bl in groups.values()):return k
    return None


def restricted_checks(rng: np.random.Generator)->dict[str,Any]:
    feasible=0
    for _ in range(36):
        n=5;d=int(rng.integers(1,3));W=rng.integers(-3,4,(n,d)).astype(float);W[-1]=-W[:-1].sum(axis=0)
        H=[(i,j) for i in range(n) for j in range(i+1,n) if rng.random()<.5]
        a,b=restricted_graph_cost(W,H),brute_restricted(W,H)
        assert a==b
        feasible+=int(a is not None)
    example=restricted_graph_cost(np.array([[1.],[2.],[4.],[-7.]]),[(0,1),(1,2),(2,3)])
    impossible=restricted_graph_cost(np.array([[1.],[2.],[4.],[-7.]]),[(0,1),(2,3)])
    assert example==3 and impossible is None
    return {'random_cases':36,'feasible':feasible,'infeasible':36-feasible,'mismatches':0,
            'one_dimensional_path_cost':example,'disconnected_example':'impossible',
            'priority_note':'Single-demand complete-graph core is equivalent to prior debt-settlement theory (Verhoeff 2004); not claimed new.'}


def full_support_check(rng:np.random.Generator,out:Path)->dict[str,Any]:
    N,sig,gamma,reps=2000,1.,.1,50000
    rows=[]
    for eps in [.001,.002,.005,.01,.02,.05,.1,.2]:
        variances=np.array([sig**2/(2*N*(1-eps)),sig**2/(2*N*eps)])
        samples=rng.normal(size=(reps,2))*np.sqrt(variances)
        predicted=float(variances.sum());mc=float(np.mean(np.sum(samples**2,axis=1)))
        gainvar=sig**2*gamma**2/(N*(1-eps));mcv=float(np.mean((np.sqrt(2)*gamma*samples[:,0])**2))
        rows.append([eps,predicted,mc,gainvar,mcv])
    np.savetxt(out/'weak_identification.csv',rows,delimiter=',',header='epsilon,prediction_risk,MC_prediction_risk,identified_gain_variance,MC_gain_variance',comments='')
    z=np.array(rows)
    plt.figure(figsize=(6.5,3.6));plt.loglog(z[:,0],z[:,1],marker='o',label='OOD prediction risk')
    plt.loglog(z[:,0],z[:,3],marker='s',label='Identified-gain estimation variance')
    plt.xlabel('Source mismatch probability (positive support)');plt.ylabel('Squared estimation error')
    plt.legend(fontsize=8);savefig(out/'full_support_separation.png')
    return {'replicates_per_condition':reps,'conditions':len(rows),
        'max_relative_error_prediction':float(np.max(np.abs(z[:,2]/z[:,1]-1))),
        'max_relative_error_gain':float(np.max(np.abs(z[:,4]/z[:,3]-1))),
        'feedback_model':'Fixed-design Gaussian scores, not binary preferences.'}


def main()->None:
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=Path('results'))
    args=parser.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(20260918)
    results={'version':'0.6','seed':20260918,
        'ordinary_pg':extra.ordinary_pg_check(),
        'actual_clipped_algorithm':extra.clipped_calibration_check(),
        'gradient_and_KL':extra.bounded_gradient_and_kl_check(),
        'weighted_discrete_corridors':weighted_corridors(rng),
        'binary_instance_bounds':binary_gap_checks(out),
        'restricted_query_graphs':restricted_checks(rng),
        'legacy_corridors':old.check_corridors(rng),
        'legacy_complete_graphs':old.check_graphs(rng),
        'multiple_adversaries':old.check_nonsmooth(),
        'five_output_closed_loop':old.check_closed_loop(out),
        'rank_one_design':old.check_rank_one(rng,out),
        'adaptive_BTL':old.check_btl_anytime(rng),
        'feature_interface':old.check_feature_bridge(rng),
        'full_support':full_support_check(rng,out)}
    ordinary_pg_plot(out);clipped_plot(out)
    (out/'verification_results.json').write_text(json.dumps(results,indent=2,ensure_ascii=False),encoding='utf8')
    print(json.dumps(results,indent=2,ensure_ascii=False))

if __name__=='__main__':main()
