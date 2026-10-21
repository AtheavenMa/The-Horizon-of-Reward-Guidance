"""Paper E1: how a fixed-size BTL batch narrows the scalar source set."""
from __future__ import annotations
import argparse,json,math,sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from horizon import btl_confidence_interval,certify_btl_batch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=Path("outputs/demo_feedback"))
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    n,k=1200,600
    q=math.sqrt(2/3)
    interval=btl_confidence_interval(successes=k,labels=n,sensitivity=q,uncertainty_radius=1.,alpha=.05)
    critical_delta=(1-.6)/((.12-.04)*15)
    fixed=certify_btl_batch(successes=k,labels=n,sensitivity=q,critical_delta=critical_delta,alpha=.05)
    t=np.linspace(0,15,601)
    lag=(.12-.04)*t
    # The paper waits until t=4 for the illustrative batch, not instantaneous querying.
    source=1-np.abs(lag)
    post=np.where(t<4.,source,1-np.maximum(abs(interval.lower_delta),abs(interval.upper_delta))*np.abs(lag))
    fig,ax=plt.subplots(figsize=(7.5,4))
    ax.plot(t,source,label="Source-only certificate")
    ax.plot(t,post,label="After 1,200 independent BTL labels",lw=2)
    ax.axhline(.6,color="0.4",ls="--",label="Required progress")
    ax.axvline(4.,color="0.6",ls=":",label="Feedback becomes available")
    ax.set(xlabel="Policy time",ylabel="Worst-case normalized gain")
    ax.legend(fontsize=8);fig.tight_layout();fig.savefig(a.out/"demo_output.png",dpi=180);plt.close(fig)
    result={"labels":n,"successes":k,"alpha":.05,"query_time":4.,
            "two_sided_delta_interval":[interval.lower_delta,interval.upper_delta],
            "post_query_min_certificate":float(min(post)),"required_progress":.6,
            "fixed_batch_one_sided_test_certifies_T15":fixed.certified,
            "scope":"Fixed independent pure BTL comparisons, paper illustrative data; no neural RLHF training."}
    (a.out/"summary.json").write_text(json.dumps(result,indent=2))
    print(f"95% delta interval: [{interval.lower_delta:.6f}, {interval.upper_delta:.6f}]")
    print(f"Min certificate with feedback: {min(post):.6f}")
    print(f"One-sided T=15 certificate: {fixed.certified}")

if __name__=="__main__":main()
