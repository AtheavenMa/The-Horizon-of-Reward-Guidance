# Reproduce the milestone

The task entry is the only launcher:

```bash
python tasks/08_Final_Milestone_and_Paper_Freeze/__main__.py run F0
python tasks/08_Final_Milestone_and_Paper_Freeze/__main__.py run F5
python tasks/08_Final_Milestone_and_Paper_Freeze/__main__.py run F6
```

F0 and F5 are CPU. F6 reads the JSON files and rewrites the tables and figures. F1, F2, F3 and F4 require `HORIZON_GPU=1` and the cached Qwen2.5-0.5B, Pythia-1B SFT, CleanRL reward model, TL;DR parquet and RewardBench 2 parquet. Do not start them on a GPU that another job is using.

CPU audit, with no Torch import from the package entry:

```bash
python -m experiments.guidance_audit validate --manifest examples/08_guidance_audit/finite_computed.json
python -m experiments.guidance_audit audit --manifest examples/08_guidance_audit/finite_computed.json --out experiments/horizon_final_08/public_smoke/
python -m experiments.guidance_audit compare --baseline experiments/horizon_final_08/public_smoke/pythia_source --treatment experiments/horizon_final_08/public_smoke/pythia_bounded_arm --metric fixed_eval --pair-by seed --out experiments/horizon_final_08/public_smoke/paired
```

The human FINAL numbers to match are in `experiments/horizon_final_08/human_final_holdout/final_metrics_corrected.json`. The Pythia per-event RMS values are in `pythia_bounded/B_bounded_seed0.json` and `B_bounded_seed1.json`. A restored checkpoint must contain an AdamW state; `B_bounded_seed0_step16.pt` does.
