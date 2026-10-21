# Guidance audit quickstart

Run from the repository root:

```bash
python tasks/07_Scale_and_Community_Evaluation/__main__.py run S0
```

That checks the scalar horizon, the ordinary policy-gradient contact, a frozen-policy label update, and a missing manifest. Results are written to `experiments/horizon_scale_07/`, which is the deployed location of the paths this plan calls `outputs/horizon_scale_07/`.

The package CLI can also validate one manifest:

```bash
python -m experiments.guidance_audit validate --manifest examples/07_guidance_audit/finite_example.json
```

An empirical manifest returns `not_certified`. It does not invent a support horizon.
