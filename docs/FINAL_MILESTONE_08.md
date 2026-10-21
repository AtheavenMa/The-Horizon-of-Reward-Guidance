# Final milestone 08

Status: **MILESTONE_COMPLETE_SCOPED**.

The scalar source-only support horizon remains 5 in the stated model. The 16-outcome coupled trajectory still first loses feasibility at 0.012; its upper value 0.05993753 is computational evidence, because this audit did not repeat the outward-rounding proof. On the recorded H3 state, changing only the preference information leaves the frozen return unchanged, while the later guidance choice changes the path. The 160-update DST runs keep the previously recorded separation between robust and true-weight controllers.

More human TL;DR comparisons raise development accuracy of a Qwen2.5-0.5B LoRA reward head. On the repaired 995-pair FINAL split, that head does not beat a length-and-overlap logistic regression. Empty-8x8 succeeds for every restored-optimizer arm that was run. DoorKey stays unlearned.

The new Pythia-1B runs enforce a per-event anchor RMS of 0.05 by weight interpolation. All six applied events stay at or below 0.0498. Held-out ROUGE-L does not improve relative to the fixed source reward. The paired mean of B minus A is -0.0017 with a 95% paired-t interval [-0.016, 0.012] on two seeds. A 16-step frozen AI teacher also leaves ROUGE-L unchanged, so the null is not evidence for a formal neural support horizon. The run is censored at step 64.

The 502-row RewardBench 2 replica uses the pinned nontie rule and scores 0.218 macro. It is not an official leaderboard number. The 4B judge, the full 1,865 rows and the Qwen2.5-1.5B transfer were not run. Nine CPU audit tests pass, including rejection of a false stored certificate.

Base SHA when these JSON files were written: `d5377ae30d16e30c6e40b5f0027bd149dae18ed2`. The pack commit is the revision that contains this file. Spot check: finite status `COMPUTED`, restored AdamW tensors 32, archived FINAL accuracy matches the corrected full split: True.
