# Experiments for the manuscript

This note is the experimental section to copy into the paper. The repository has no LaTeX manuscript, and this file does not mean the PDF has been revised.

## Exact and finite controls

In the scalar model with common gain 1, required gain 0.6, uncertainty half-width 1, target speed 0.12 and adaptation speed 0.04, the source-only support horizon is exactly 5. The ordinary policy-gradient contact value remains 0.0078358743. These statements are theorems or direct numerical checks of those theorems, not properties of a language-model checkpoint.

The 16-outcome coupled model first loses trajectory feasibility at time 0.012 even though a pointwise certificate is still available. The reported all-controls upper bound 0.05993753 was not re-proved with outward rounding here, so it stays computational evidence. The 64-outcome self-check only confirms that a zero extension reproduces the 16-outcome gradient.

## Same state, different guidance

At the early H3 checkpoint, the information-only return gap is 0: the frozen policy is unchanged by the label update. Continuing from that state, source guidance later gains 0.0528 while updated robust, clipped MLE and true-weight one-step guidance gain 0.0791. The 160-update Deep Sea Treasure means over 8 seeds are source 0.683, uniform robust 0.683, informative robust 0.678, clipped MLE 0.651 and true weight 0.638. Deeper-treasure probability at step 160 is about 0.46, 0.41, 0.13 and 0.008 for source, informative robust, MLE and true weight respectively. These are finite-horizon empirical paths. They are not a theorem about language models.

## Human preferences

The reward model is Qwen2.5-0.5B-Instruct with LoRA rank 8 on `q_proj` and `v_proj`, plus a linear head. Labels are human Reddit TL;DR comparisons. The 512 previously inspected pairs are development only. Document overlap reduced the remaining holdout from 1,536 rows to 995 rows; intervals are document bootstraps. Temperature 1.7 was chosen on development margins of the selected run and then applied to the holdout. Selection used development accuracy only: `human_4098_seed0` at 0.635.

Final accuracy is about 0.505 for 257 pairs, 0.549 for 1,024 and 0.563 for 4,098, each over 3 seeds. A length-and-overlap logistic regression trained on 4,098 pairs scores about 0.583 on the same 995 rows. Extra epochs on the 257-pair pool, with optimizer steps matched to one pass over 4,098, do not reach the 4,098 development accuracy. The previously reported final NLL in `metrics.json` used only the last evaluation batch. The full-split NLL is in `final_metrics_corrected.json` and is about 0.68 for the 4,098-pair models at the frozen temperature.

MiniGrid-Empty-8x8, started from a shared policy and AdamW state, reaches the goal on all 100 evaluation episodes for oracle, source, 32 new pairs and extra fitting, on seeds 0 and 1. Mean episode length is 11 or 14. DoorKey-6x6 remains unlearned. A short probe did not explain the archived extra-fit seed that scored zero.

## Budgeted Pythia-1B updates

The policy is the cached CleanRL Pythia-1B SFT, with LoRA rank 8. The deployed reward starts at the development-selected 0.5B human head. Arm A takes 64 policy updates with that fixed reward and no new labels. Arm B inserts 32 AI-teacher comparisons before steps 16, 32 and 48. Each candidate is fit for 64 AdamW steps and then weight-interpolated onto the frozen anchor set of 128 pairs. The largest accepted step is 0.0497, below 0.05 plus the predeclared tolerance 1e-4. Cumulative movement from the original head is about 0.11, which is a different quantity and is not bounded by 0.05. Arm C keeps the original reward; its anchor margins do not move. It does not replay B's response texts. Arm D was not run, because those texts were not saved. Arm E, a 16-step frozen teacher on one seed, moves ROUGE-L from 0.161 to 0.161.

The evaluation is deterministic ROUGE-L against the dataset completion on 128 held-out documents, with a temperature-0.7 sample as a secondary score. Generation length is 64 tokens. Reference KL is the mean per continuation token of the log-probability gap to the frozen SFT. No update was stopped by the 0.1 nat/token cap. The paired ROUGE-L difference of B minus A is about −0.002. The human holdout NLL of the updated heads stays at 0.683. The empirical reuse endpoint is censored: no development failure threshold was declared, and none is inferred.

## External scoring and forks

On the precommitted 502 RewardBench 2 rows, including all Ties, an independent implementation of the official nontie rule matched a hand-built fixture on 8 score vectors. The TL;DR head then scores a subset macro of 0.218. Five hundred eighty completions hit the 512-token context cap. This is a protocol-compatible replica of a summarization reward head, not an official leaderboard entry. The frozen 4B judge was not scored, because the official generative script requires vLLM and vLLM is not installed.

Eight Pythia checkpoints at steps 16 and 48, from arms A and B and seeds 0 and 1, restore the actor and AdamW moments. From each parent the arms keep training, fit the same human source pairs, reveal 32 held-out human pairs, label 32 new on-policy AI pairs, combine those labels, fit without deploying, or optimize against the teacher. Every accepted reward move stays within the event budget, and a no-actor re-decode is identical. Mean ROUGE-L differences from the keep arm are within about ±0.005. The forks do not identify a quality mechanism.

## What not to write

Do not call the Pythia result a formal support horizon. Do not cite the old end-of-run RMS values 1.46 and 1.35 as a valid bounded run. Do not cite RewardBench macros 0.209 and 0.112 as official. Do not call AI teacher labels human. Do not say the paper PDF has been updated.
