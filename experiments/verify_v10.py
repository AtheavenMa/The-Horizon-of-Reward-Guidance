"""Reproduce the v1.0 paper: inherited v0.6 checks + one new discrete experiment."""
from __future__ import annotations
import argparse, hashlib, json, platform, subprocess, sys
from pathlib import Path
import numpy, scipy, matplotlib
from discrete_resource_check import run
from separation_check import run as separation


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=Path('results'))
    args=parser.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    subprocess.run([sys.executable,str(Path(__file__).with_name('verify_v06.py')),'--out',str(out)],check=True)
    inherited=json.loads((out/'verification_results.json').read_text())
    new=run(out)
    result={'paper_version':'1.0','inherited_v06':inherited,'added_discrete_resource_experiment':new,
        'recomputed_prediction_guidance_separation':separation(out),
        'environment':{'python':platform.python_version(),'numpy':numpy.__version__,'scipy':scipy.__version__,'matplotlib':matplotlib.__version__},
        'note':'Synthetic finite-model checks. Graphical illustrations and Monte Carlo are not substitutes for proofs or a priority review.'}
    (out/'paper_results_v10.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print('\nAll v1.0 verification blocks completed.')
if __name__=='__main__':main()
