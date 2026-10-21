#!/usr/bin/env python3
"""Inherited v0.5 regression tests (source preserved; output is rerun in v0.6).
Reproducible finite-model checks. Numerical checks do not replace proofs.
Requires numpy, scipy, matplotlib; no GPU or network.
"""
from __future__ import annotations
import argparse,itertools,json,math
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import linprog,minimize
from scipy.special import expit,logit
from scipy.stats import binom
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def partitions(n):
 b=[]
 def go(k):
  if k==n: yield tuple(tuple(x) for x in b); return
  for x in b:
   x.append(k);yield from go(k+1);x.pop()
  b.append([k]);yield from go(k+1);b.pop()
 yield from go(0)

def graph_cost_partition(W):
 W=np.asarray(W,dtype=float)
 if W.ndim==1:W=W[:,None]
 c=len(W);best=None
 for P in partitions(c):
  if all(np.linalg.norm(W[list(x)].sum(0))<1e-10 for x in P):
   if best is None or len(P)>len(best):best=P
 return c-len(best),best

def graph_cost_enum(W):
 W=np.asarray(W,dtype=float)
 if W.ndim==1:W=W[:,None]
 c=len(W);edges=list(itertools.combinations(range(c),2))
 for k in range(c):
  for es in itertools.combinations(edges,k):
   p=list(range(c))
   def root(i):
    while p[i]!=i:p[i]=p[p[i]];i=p[i]
    return i
   for i,j in es:p[root(i)]=root(j)
   groups={}
   for i in range(c):groups.setdefault(root(i),[]).append(i)
   if all(np.linalg.norm(W[g].sum(0))<1e-10 for g in groups.values()):return k
 raise AssertionError('spanning tree should suffice')

def check_graphs(rng):
 cases=[]
 for w in [[1,2,4,-7],[1,-1,2,-2],[1,1,-2,0],[1,2,4,8,-15]]:
  a,P=graph_cost_partition(w);assert a==graph_cost_enum(w)
  cases.append(dict(demand=w,rank=1,exact_pairs=a,partition=P))
 for _ in range(16):
  w=rng.integers(-3,4,4);w=np.r_[w,-w.sum()]
  a,_=graph_cost_partition(w);assert a==graph_cost_enum(w)
 witness=[[8,9,13],[8,10,12],[9,10,11],[10,10,10]]
 assert all(sum(x)==30 for x in witness)
 assert all(30/4<a<30/2 for x in witness for a in x)
 return dict(examples=cases,random_instances_checked=16,three_partition_reduction_witness=witness,min_pairs_in_witness=12)

def check_corridors(rng):
 counts={'feasible':0,'infeasible':0}
 for _ in range(150):
  t=np.linspace(0,1,13);v=float(rng.uniform(.1,2.5));center=rng.normal(size=len(t)).cumsum()*.13
  width=rng.uniform(.05,.4,len(t));lo=center-width;hi=center+width;lo[0]=hi[0]=center[0]
  dist=np.abs(t[:,None]-t[None,:]);viol=np.max(lo[:,None]-hi[None,:]-v*dist)
  expected=bool(viol<=1e-10);A=[];bb=[]
  for i in range(len(t)-1):
   r=np.zeros(len(t));r[i+1]=1;r[i]=-1;A.extend([r,-r]);bb.extend([v*(t[i+1]-t[i])]*2)
  sol=linprog(np.zeros(len(t)),A_ub=np.array(A),b_ub=bb,bounds=list(zip(lo,hi)),method='highs')
  assert sol.success==expected,(viol,sol.message)
  if expected:
   z=np.max(lo[None,:]-v*dist,axis=1)
   assert np.all(z>=lo-1e-9) and np.all(z<=hi+1e-9)
   assert np.all(np.abs(np.diff(z))<=v*np.diff(t)+1e-9)
  counts['feasible' if expected else 'infeasible']+=1
 return counts

def check_nonsmooth():
 u=np.linspace(-2,2,4001);val=float(np.max(np.minimum(-1+u,-1-u)));assert abs(val+1)<1e-12
 return dict(correct_max_derivative=val,incorrect_single_adversary_formula=1.)

def helmert(n):
 V=np.zeros((n,n-1))
 for j in range(n-1):
  V[:j+1,j]=1/np.sqrt((j+1)*(j+2));V[j+1,j]=-(j+1)/np.sqrt((j+1)*(j+2))
 return V

def check_closed_loop(out):
 eps=.04;g=1.;eta=1.;margin=.6;speed=.04;drift=.12;d=4.;k=drift/(eps*d);V=helmert(5);T=15.
 def M(th):
  c=k*th[2];K=np.array([[g,0.],[-c,1.]])
  return np.block([[d*np.eye(2),K],[K.T,d*np.eye(2)]])
 def rhs(t,y):
  return np.r_[eps*M(y[:4])@np.array([0.,0.,1.,y[4]]),speed]
 tt=np.linspace(0,T,601);sol=solve_ivp(rhs,(0,T),np.zeros(5),t_eval=tt,rtol=1e-11,atol=1e-13);assert sol.success
 th=sol.y[:4].T;phi=sol.y[4];c=k*th[:,2];lag=c-phi;probs=.2+eps*th@V.T
 before=g-eta*np.abs(lag);exit=(g-margin)/(drift-speed)
 assert np.max(np.abs(c-drift*tt))<1e-9 and np.max(np.abs(phi-speed*tt))<1e-9
 assert probs.min()>0;mineig=min(np.linalg.eigvalsh(M(x))[0] for x in th);assert mineig>0
 q=np.sqrt(2/3);conf=.05;n=1200;wins=600;rad=math.sqrt(math.log(2/conf)/(2*n))
 lo=max(expit(-q),wins/n-rad);hi=min(expit(q),wins/n+rad)
 dlo=float(logit(lo)/q);dhi=float(logit(hi)/q);query_time=4.
 bound=np.where(tt<query_time,1.,max(abs(dlo),abs(dhi)));after=g-bound*np.abs(lag);assert after.min()>margin
 threshold=(g-margin)/float(lag[-1]);p_lo=expit(-q*threshold);p_hi=expit(q*threshold)
 wl=math.ceil(n*(p_lo+rad));wh=math.floor(n*(p_hi-rad))
 pr=float(binom.cdf(wh,n,.5)-binom.cdf(wl-1,n,.5)) if wh>=wl else 0.
 np.savetxt(out/'closed_loop.csv',np.c_[tt,c,phi,before,after,np.min(probs,axis=1)],delimiter=',',header='time,required_coefficient,reward_coefficient,source_certificate_normalized,post_query_certificate_normalized,min_probability',comments='')
 fig=plt.figure(figsize=(7.6,4.6));ax=fig.add_subplot(111)
 ax.plot(tt,before,label='Source-only certificate');ax.plot(tt,after,label='After 1,200 targeted binary labels')
 ax.plot(tt,np.ones_like(tt),':',label='Actual gain: delta = 0');ax.axhline(margin,linestyle='--',label='Required progress')
 ax.axvline(exit,linestyle=':',label='Exact no-query deadline')
 ax.set(xlabel='Gradient-flow time',ylabel='Gain divided by epsilon squared',title='A feasible reward exists, but adaptation cannot keep up')
 ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/'closed_loop.png',dpi=180);plt.close(fig)
 return dict(epsilon=eps,shift_speed=drift,reward_speed=speed,source_only_exact_deadline=exit,pointwise_value_normalized=g,horizon=T,minimum_policy_probability=float(probs.min()),minimum_preconditioner_eigenvalue=float(mineig),query_labels=n,illustrative_query_wins=wins,query_delta_interval=[dlo,dhi],query_time=query_time,post_query_min_certificate_normalized=float(after.min()),fixed_N_recovery_probability_at_delta_zero=pr,required_progress_normalized=margin,feedback_model='Fixed-size batch of independent BTL labels; no claim of zero query latency.')

def check_rank_one(rng,out):
 A=rng.normal(size=(6,6));V=A.T@A+np.eye(6);d=rng.normal(size=6);b=rng.normal(size=6);Vi=np.linalg.inv(V)
 r0=float(b@Vi@b);x=float(d@Vi@b);R=float(d@Vi@d);err=0.;vals=[];ns=np.arange(201)
 for n in ns:
  actual=float(b@np.linalg.solve(V+n*np.outer(d,d),b));formula=r0-n*x*x/(1+n*R);err=max(err,abs(actual-formula));vals.append(actual)
 assert err<1e-10
 reduction=.5*x*x/R;ncrit=reduction/(x*x-reduction*R);floor=r0-x*x/R
 fig=plt.figure(figsize=(7.6,4.5));ax=fig.add_subplot(111);ax.plot(ns,vals,label='Exact inverse-design energy')
 ax.axhline(floor,linestyle='--',label='Residual after infinitely many labels on this pair')
 ax.set(xlabel='Repeated comparisons on one pair',ylabel='Directional uncertainty energy',title='One pair can leave an irreducible task-relevant residual')
 ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/'query_energy.png',dpi=180);plt.close(fig)
 return dict(maximum_identity_error=err,initial_energy=r0,limiting_energy=floor,minimum_continuous_count_for_half_available_reduction=ncrit)

def check_btl_anytime(rng):
 n=4;Q=helmert(n);dim=n-1;S=1.;lam=1.;delta=.05;pairs=list(itertools.combinations(range(n),2));ds=np.array([Q[i]-Q[j] for i,j in pairs])
 D=math.sqrt(2)*S;kap=float(expit(D)*(1-expit(D)));true=np.array([.25,-.15,.1]);runs=100;steps=40;covered=0;max_ratio=0.
 for _ in range(runs):
  X=[];Y=[];last=np.zeros(dim);ok=True
  for t in range(1,steps+1):
   H=lam*np.eye(dim)+sum((np.outer(x,x) for x in X),start=np.zeros((dim,dim)))
   vals=np.array([x@np.linalg.solve(H,x)+.02*abs(x@last) for x in ds]);choices=np.flatnonzero(vals>=vals.max()-1e-12)
   xnew=ds[int(rng.choice(choices))];X.append(xnew);Y.append(float(rng.random()<expit(xnew@true)))
   if t%5:continue
   x=np.array(X);y=np.array(Y)
   def fun(z):return float(np.logaddexp(0,x@z).sum()-y@(x@z)+.5*kap*lam*(z@z))
   def jac(z):return x.T@(expit(x@z)-y)+kap*lam*z
   sol=minimize(fun,last,jac=jac,method='SLSQP',constraints=[{'type':'ineq','fun':lambda z:S*S-z@z,'jac':lambda z:-2*z}],options={'ftol':1e-11,'maxiter':200})
   if not sol.success:
    sol=minimize(fun,np.zeros(dim),jac=jac,method='SLSQP',constraints=[{'type':'ineq','fun':lambda z:S*S-z@z,'jac':lambda z:-2*z}],options={'ftol':1e-8,'maxiter':500})
   z=sol.x;gz=jac(z)
   mult=max(0.,-float(gz@z)/(z@z)) if np.linalg.norm(z)>S-1e-5 else 0.
   kkt=float(np.linalg.norm(gz+mult*z))
   assert np.linalg.norm(z)<=S+1e-6 and kkt<2e-4,(sol.message,kkt,np.linalg.norm(z))
   last=z;H=lam*np.eye(dim)+x.T@x
   logterm=.5*np.linalg.slogdet(H)[1]-.5*dim*math.log(lam)+math.log(1/delta)
   beta=math.sqrt(2*logterm)/(2*kap)+math.sqrt(lam)*S;er=last-true;ratio=float(math.sqrt(er@H@er)/beta)
   max_ratio=max(max_ratio,ratio)
   if ratio>1+1e-9:ok=False
  covered+=ok
 return dict(runs=runs,labels_per_run=steps,checkpoints_per_run=8,empirical_simultaneous_coverage=covered/runs,nominal_coverage=1-delta,maximum_error_radius_ratio=max_ratio,warning='Conservative bound: high coverage is not evidence of tightness.')

def check_feature_bridge(rng):
 n=7;k=3;m=2;logits=rng.normal(size=n);p=np.exp(logits-logits.max());p/=p.sum()
 psi=rng.normal(size=(n,k));chi=rng.normal(size=(n,m));w=rng.normal(size=k);phi=rng.normal(size=m)
 D=np.diag(p)-np.outer(p,p);Cpsi=D@psi;Cchi=D@chi;v=Cchi@phi;a=Cpsi.T@v;exact=(D@psi@w)@v
 assert abs(exact-w@a)<1e-10
 h=1e-6
 def prob(t):
  z=logits+t*v;q=np.exp(z-z.max());return q/q.sum()
 fd=psi.T@(prob(h)-prob(-h))/(2*h)
 return dict(gradient_identity_error=float(abs(exact-w@a)),feature_JVP_finite_difference_error=float(np.linalg.norm(fd-a)))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',default=str(Path(__file__).resolve().parents[1]/'results'));args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(20260918)
 res=dict(status='Finite-model numerical checks; see proofs for claims.',graph_completion=check_graphs(rng),finite_horizon_corridor=check_corridors(rng),nonunique_adversary=check_nonsmooth(),closed_loop=check_closed_loop(out),rank_one_query=check_rank_one(rng,out),adaptive_BTL_confidence=check_btl_anytime(rng),matrix_free_bridge=check_feature_bridge(rng))
 (out/'verification_results.json').write_text(json.dumps(res,indent=2,ensure_ascii=False));print(json.dumps(res,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
