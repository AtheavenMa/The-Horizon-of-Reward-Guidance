# Audit quickstart

No GPU and no model download.

```bash
python -m experiments.guidance_audit validate --manifest examples/08_guidance_audit/finite_computed.json
python -m experiments.guidance_audit audit --manifest examples/08_guidance_audit/finite_computed.json --out experiments/horizon_final_08/public_smoke/
python -m experiments.guidance_audit compare --baseline experiments/horizon_final_08/public_smoke/pythia_source --treatment experiments/horizon_final_08/public_smoke/pythia_bounded_arm --metric fixed_eval --pair-by seed --out experiments/horizon_final_08/public_smoke/paired
```

`intentionally_false_certificate.json` is rejected or corrected. A non-simplex set without a validated LP returns `NOT_VERIFIED`. The neural Pythia files are empirical records, not finite certificates. See `docs/BENCHMARK_CARD_08.md`.
