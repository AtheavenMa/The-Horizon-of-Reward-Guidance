"""Minimal analytic demo: pointwise guidance versus a finite support horizon."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from horizon import analyze_horizon, certified_gain


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=Path("outputs/demo_horizon"))
    parser.add_argument("--target-horizon",type=float,default=15.)
    a=parser.parse_args()
    r=analyze_horizon(gain=1.,required_gain=.6,uncertainty_radius=1.,
                     drift_speed=.12,adaptation_speed=.04,target_horizon=a.target_horizon)
    a.out.mkdir(parents=True,exist_ok=True)
    ts=np.linspace(0,a.target_horizon,401)
    cert=[certified_gain(1.,1.,float(t),.04*float(t),.12) for t in ts]
    fig,ax=plt.subplots(figsize=(7.5,4))
    ax.plot(ts,np.ones(len(ts)),":",color="0.4",label="Pointwise optimum")
    ax.plot(ts,cert,lw=2,label="Best admissible source-only trajectory")
    ax.axhline(.6,ls="--",color="0.4",label="Required progress")
    if r.support_horizon<=a.target_horizon:
        ax.axvline(r.support_horizon,ls=":",color="0.5",label="Exact boundary")
    ax.set(xlabel="Policy time",ylabel="Worst-case normalized gain")
    ax.legend(fontsize=8,loc="best")
    fig.tight_layout();fig.savefig(a.out/"demo_output.png",dpi=180);plt.close(fig)
    summary={"gain":1.,"required_gain":.6,"uncertainty_radius":1.,
             "drift_speed":.12,"adaptation_speed":.04,
             "target_horizon":r.target_horizon,"support_horizon":r.support_horizon,
             "minimum_required_speed":r.minimum_required_speed,
             "pointwise_feasible":r.pointwise_feasible,
             "trajectory_feasible":r.trajectory_feasible,
             "source_certificate_at_target":r.source_certificate_at_target,
             "assumptions":"Scalar symmetric source set and nonbinding amplitude."}
    (a.out/"summary.json").write_text(json.dumps(summary,indent=2))
    print(f"Exact source-only support horizon: {r.support_horizon:.3f}")
    print(f"Minimum reward speed for T={r.target_horizon:g}: {r.minimum_required_speed:.6f}")
    print(f"Pointwise feasible: {r.pointwise_feasible}; trajectory feasible: {r.trajectory_feasible}")
    print(f"Saved {a.out/'demo_output.png'}")

if __name__ == "__main__":main()
