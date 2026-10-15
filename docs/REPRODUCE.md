# Reproducing the manuscript's numerical experiments

## 1. Environment and provenance

All experiments operate on **declared synthetic or finite models**, and the provided scripts require no GPU, API token, model downloads, annotation data, or network connection at runtime.

The supplied reference ledger reports the following original environment:

| Component | Version recorded |
|---|---|
| Python | 3.13.5 |
| NumPy | 2.3.5 |
| SciPy | 1.17.0 |
| Matplotlib | 3.10.8 |
| Pseudorandom seed | `20260918` |

The declared software dependencies support Python **3.10+**. Monte Carlo figures may vary slightly under different library versions, so the regression checks use explicit numeric tolerances rather than requiring byte-identical PNG or PDF files. The numeric source files are versioned and immutable from the reproduction command's perspective.

## 2. Install

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements_dev.txt
python -m pip install -e . --no-deps
```

On Windows PowerShell, the activation command is `.venv\Scripts\Activate.ps1`.

## 3. Quick analytical checks and demos

```bash
python -m pytest -q
python demo/demo.py
python demo/demo_feedback.py
python demo/demo_ordinary_pg.py
```

The scalar demonstration produces $T_0=5$. The feedback example uses 600/1200 binary responses and produces a two-sided preference interval of approximately $[-0.19246,0.19246]$ and a post-query trajectory minimum of 0.682. The ordinary-PG demonstration reports an all-controls local analytic time upper bound of about 0.010006 and numerical maximal-speed contact at 0.007836.

Each demo writes a `summary.json` and `demo_output.png` inside its own `outputs/demo_*/` directory. Custom destinations use `--out`.

## 4. Complete numerical reproduction

Run exactly:

```bash
python experiments/reproduce.py --all
```

Under the hood, this executes the preserved original scripts in order:

```bash
python experiments/verify_v10.py --out outputs/paper_reproduction
python plotting/figures.py --res outputs/paper_reproduction --out outputs/paper_reproduction --venue iclr
```

Then it checks newly generated files against the **frozen** [`results/reference/`](../results/reference/) values. The reference location is not a writable output destination; attempts to overwrite it are rejected by the wrapper.

Success prints something of the form:

```text
PASS: full numerical ledger and figure reproduction {'json': 4, 'csv': 5, 'figures': 7}
```

And writes `outputs/paper_reproduction/reproduction_status.json`, together with five generated CSV files, four generated JSON ledgers, seven high-resolution PNGs in `png/`, and seven PDF figures. The wrapper validates nested numeric results with a tolerance of `rtol=1e-6` and `atol=2e-7`, accounting for solver and Monte Carlo floating-point differences on supported environments. A changed scientific parameter or missing result is a failure.

**Why two plotting entry points?** The original experiment directory retains `figures.py` and `paperfig.py` as provenance copies. The public reproduction command uses `plotting/figures.py` for one predictable figure-generation location. Both use the same paper plotting logic.

## 5. Paper claims to code and data

| Manuscript component | Numerical implementation | Frozen outputs | Verification type |
|---|---|---|---|
| Exact scalar moving target, feedback-extended five-output example | `experiments/legacy_checks.py` (`check_closed_loop`) | `closed_loop.csv`, `verification_results.json` | Analytic identity and finite-model ODE |
| Local ordinary Euclidean PG separation | `experiments/strengthening_checks.py` (`ordinary_pg_check`) | `ordinary_pg.csv`, `verification_results.json` | Analytic neighborhood enclosure + numerical maximal-speed path |
| Moving discrete corridors | `experiments/verify_v06.py` (`weighted_corridors`), `experiments/legacy_checks.py` | `verification_results.json` | Independent linear-program comparisons |
| BTL fixed-design complexity | `experiments/verify_v06.py` (`binary_gap_checks`) | `binary_gap_bounds.csv`, `verification_results.json` | Exact binomial checks and KL lower-bound calculations |
| Clipped calibration startup loss | `experiments/strengthening_checks.py` (`clipped_calibration_check`) | `verification_results.json` | Exact ODE versus numerical ODE |
| Actual finite reward update resource E4 | `experiments/discrete_resource_check.py` | `discrete_resources.csv`, `discrete_resources.json` | 300 explicit finite policy steps × 4 reward-update settings |
| Prediction accuracy versus useful guidance E5 | `experiments/separation_check.py` | `prediction_guidance_separation.json` | Independent analytic recomputation |
| Query information energy | `experiments/legacy_checks.py` (`check_rank_one`) | `verification_results.json` | Rank-one analytic equality versus matrix solve |
| Weak Gaussian identification | `experiments/verify_v06.py` (`full_support_check`) | `weak_identification.csv` | 8 Gaussian conditions with 50k draws per condition |
| All seven paper visuals | `plotting/figures.py` | PNG/PDF files in reproduction output | Figure regeneration from data or declared analytic formula |

Important: `figures.py` reconstructs the **analytic clipped-lag and query-energy curves** from saved parameters when no matching per-point CSV exists. Those two curves are not represented as independent raw observations.

## 6. Paper availability

The full manuscript and source are not distributed in this repository. **arXiv coming soon.** The frozen numerical ledger, figures, experiment scripts, and reproduction commands remain unchanged. No arXiv identifier, accepted venue, or publication status is asserted before the public preprint is available.

## 7. CI and release check

[`tests.yml`](../.github/workflows/tests.yml) runs test and package-build matrices on Python 3.10, 3.12, and 3.13, plus a separate Python 3.13 job that runs the entire numerical-and-figure pipeline. A workflow file alone is not proof of a successful GitHub-hosted run; after pushing to GitHub, check the Actions tab and review failing environments independently.

To run a build locally:

```bash
python -m build --wheel --sdist
```

After creating a repository, publish verified code from a clean branch/PR, check GitHub Actions, and tag a first release only after checks pass. The root `LICENSE` governs original research software and documentation, not the forthcoming manuscript.
