# Real-learning experiments

Commands are started from the task, not by calling these modules directly:

```bash
python tasks/01_Reward_Guidance_Experiments/__main__.py run baseline
python tasks/01_Reward_Guidance_Experiments/__main__.py run E1 --stage pilot
python tasks/01_Reward_Guidance_Experiments/__main__.py run E1 --stage confirm
python tasks/01_Reward_Guidance_Experiments/__main__.py run E2
python tasks/01_Reward_Guidance_Experiments/__main__.py run E3
python tasks/01_Reward_Guidance_Experiments/__main__.py run E4 --stage confirm
python tasks/01_Reward_Guidance_Experiments/__main__.py run E5 --stage confirm
python tasks/01_Reward_Guidance_Experiments/__main__.py run E6 --stage confirm
python tasks/01_Reward_Guidance_Experiments/__main__.py run E7 --stage confirm
python tasks/01_Reward_Guidance_Experiments/__main__.py run E8
python tasks/01_Reward_Guidance_Experiments/__main__.py run checks
```

Run records are written to `experiments/horizon_real/`. Checkpoints under those trees are gitignored.
The original numerical ledger in `results/reference/` is not modified.
