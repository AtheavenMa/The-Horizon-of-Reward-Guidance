# Reviewer audit 08

- The scalar theorem and the ordinary policy-gradient contact are separate from the 16-outcome numerical bound and from every neural run.
- The 16-outcome upper bound is labeled computational evidence. It is not claimed as a checked interval proof.
- The H3 information-only return gap is zero. Later path gains are a different comparison, with the controller allowed to move.
- The true preference weight is used only by arms named as privileged: DST true weight, H3 true-weight one-step, MiniGrid oracle, and Pythia arm E. Those arms are not the deployed human reward.
- F4 reloads AdamW moments. A greedy decode repeated without an actor update matches. WP07 S5, which rebuilt the optimizer, is not used.
- Every applied Pythia reward event has anchor RMS at most 0.0498. Cumulative movement from the original head is about 0.11 and is reported separately. The historical RMS above 1 is marked invalid.
- Policy loss uses prompt token IDs concatenated with generated continuation IDs. Reference KL is the mean gap to the frozen SFT on those continuation tokens. Quality is ROUGE-L on a fixed 128-document set, not the training reward.
- Development and FINAL human documents do not overlap. FINAL has 995 pairs. Model selection used development accuracy. Online labels are AI feedback.
- The 4B judge was not scored. The 502-row macro is a protocol-compatible subset. Ties are included. 580 completions hit the 512-token cap.
- Changing the example probabilities changes the computed certificate. A false stored certificate is replaced by the computed value. The paired CLI returns no interval when N is below 2.
- DoorKey is unlearned. Empty success is saturated, so mean steps are the secondary number. The archived extra-fit failure on seed 2 was not explained.
- Figure numbers are read from the JSON files named in `FINAL_REPRODUCE_08.md`.
