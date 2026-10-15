# Contributing

Contributions are welcome when they preserve the distinction between theorems, numerical evidence, and experiments not yet run. This repository is a small research artifact, not a general-purpose RLHF training framework.

## Development setup

```bash
python -m pip install -r requirements_dev.txt
python -m pip install -e . --no-deps
python -m pytest -q
```

For changes affecting the manuscript's numerical evidence:

```bash
python experiments/reproduce.py --all
```

## Ground rules

1. **Do not edit the frozen reference ledger** to make new tests pass. Generate outputs in `outputs/` and describe any intentional scientific change separately.
2. **Keep claims within their proven model assumptions.** A scalar theorem is not a neural reward-model guarantee; a discrete corridor test is not a continuous-time solver.
3. **Keep the unreleased manuscript separate.** The repository does not distribute the PDF or TeX source before the arXiv preprint. Do not silently change the numerical figures or authorship metadata.
4. **Add unit tests** for new mathematical cases, boundary conditions and parameter validation. Numerical comparisons should use justified tolerances.
5. **Distinguish model data**: some experiments use independent binary Bradley–Terry comparisons; the weak-identification study uses fixed-design Gaussian scores. Do not swap their assumptions.
6. **Keep CPU-only demos reasonably small.** Heavier or downloaded-model adaptations belong in optional, clearly scoped directories and should never be prerequisites of `horizon.py`.
7. **Check licenses and attribution** for any external code, image, checkpoint or data before publication.

For substantive scientific changes, open an issue describing the intended mathematical claim, whether it is proved or only experimentally observed, the manuscript section it concerns, and the new verification commands.
