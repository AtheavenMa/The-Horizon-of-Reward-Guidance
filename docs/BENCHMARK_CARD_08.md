# Benchmark card 08

**What it is.** A CPU audit for finite reward-guidance certificates, plus the empirical records of one human TL;DR reward fit and one budgeted Pythia-1B run.

**Who should use it.** Someone who already has outcome probabilities and reward vectors, or two folders of per-seed JSON with a shared metric. It does not train a model.

**Comparison unit.** Finite certificates use the supplied outcomes. Paired tests use the training seed. Document bootstraps are a different unit and are not the seed interval.

**Data.** Human comparisons are OpenAI `summarize_from_feedback` Reddit TL;DR. Policy prompts are `trl-lib/tldr`. External rows are `allenai/reward-bench-2`, pinned in `rb2_corrected/summary.json`. The repository stores row IDs, hashes and metrics, not a redistributed copy of those corpora.

**Metrics.** Scalar horizon and finite gain when the vectors are present. Human pairwise accuracy, NLL, Brier and ECE. Pythia ROUGE-L against the dataset completion, anchor RMS, and reference KL to the frozen SFT.

**What it does not guarantee.** A neural support horizon, an official RewardBench 2 rank, a human judgment of live summaries, or a certificate for a reward head whose anchors were not saved.

**Invalid if reused as successes.** WP07 Pythia end-of-run RMS 1.460 and 1.354. WP07 RewardBench macros 0.209 and 0.112. DoorKey as a learned control.
