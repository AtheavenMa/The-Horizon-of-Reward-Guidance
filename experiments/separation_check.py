"""Exact four-outcome calculation for Appendix E; no sampling or fitting claim."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np


def run(out: Path) -> dict:
    y=np.array([[1,1],[1,-1],[-1,1],[-1,-1]],dtype=float)
    p=np.array([.45,.45,.05,.05])
    src=np.array([[1.,1.],[-1.,-1.]])
    target=np.array([1.,-1.])
    fitted=np.linalg.lstsq(src,target,rcond=None)[0]
    assert np.allclose(fitted,[.5,.5])
    omega=np.array([(1-.8**2)**2,1.])
    adaptive=np.array([omega[1],omega[0]])/omega.sum()
    answer={'source_least_squares_coefficients':fitted.tolist(),'dynamic_coefficients':adaptive.tolist()}
    for name,w in [('target_A',np.array([2.,-1.])),('target_B',np.array([-1.,2.]))]:
        reward=y@w
        answer[name]={'source_predictor_MSE':float(p@((y@fitted-reward)**2)),
            'dynamic_guidance_MSE':float(p@((y@adaptive-reward)**2)),
            'static_true_gain':float(w@(omega*fitted)),
            'dynamic_true_gain':float(w@(omega*adaptive))}
    risk=.25*float(p@((y@np.array([3.,-3.]))**2))
    answer['two_target_prediction_lower_bound']=risk
    assert abs(risk-4.5)<1e-12
    assert answer['target_A']['static_true_gain']<0
    assert min(answer[t]['dynamic_true_gain'] for t in ['target_A','target_B'])>0
    out.mkdir(parents=True,exist_ok=True)
    (out/'prediction_guidance_separation.json').write_text(json.dumps(answer,indent=2),encoding='utf8')
    return answer

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path('results'))
    print(json.dumps(run(p.parse_args().out),indent=2))
