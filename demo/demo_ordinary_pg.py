"""Illustrate local loss of a reward certificate under ordinary Euclidean PG."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.strengthening_checks import ordinary_pg_check


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=Path("outputs/demo_ordinary_pg"))
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    result=ordinary_pg_check()  # includes analytic all-controls enclosure
    x0,y0=.8,0.
    phi0=result["initial_phi"]
    t=np.linspace(0,.018,500)
    rho=.04
    def f(_,z):
        x,y,phi=z
        return [(1-x*x)**2*phi,(1-y*y)**2*(1-phi),rho]
    solution=solve_ivp(f,(0,t[-1]),[x0,y0,phi0],t_eval=t,rtol=1e-11,atol=1e-13,max_step=1e-4)
    if not solution.success:raise RuntimeError(solution.message)
    x,y,phi=solution.y
    aq=(1-x*x)**2;by=(1-y*y)**2
    lower=(by+.06)/(2*aq+by)
    pointwise=aq*by/(aq+by)
    fig,ax=plt.subplots(figsize=(7.5,4))
    ax.plot(t,phi,lw=2,label="Maximum-speed reward coefficient")
    ax.plot(t,lower,ls="--",label="Moving safe lower boundary")
    ax.axvline(result["numerical_maximal_speed_boundary_time"],ls=":",color="0.5",label="Observed contact")
    ax.set(xlabel="Ordinary policy-gradient time",ylabel="Guidance reward coefficient")
    ax.legend(fontsize=8);fig.tight_layout();fig.savefig(a.out/"demo_output.png",dpi=180);plt.close(fig)
    summary={"analytic_all_controls_failure_upper":result["analytic_all_controls_boundary_time_upper"],
             "observed_maximal_speed_contact":result["numerical_maximal_speed_boundary_time"],
             "pointwise_value_lower_over_proof_box":result["analytic_pointwise_optimum_lower_in_box"],
             "required_gain":.06,"min_pointwise_on_plotted_path":float(min(pointwise)),
             "scope":"Local full-support finite Bernoulli-policy system; NOT a global optimal-control result."}
    (a.out/"summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=="__main__":main()
