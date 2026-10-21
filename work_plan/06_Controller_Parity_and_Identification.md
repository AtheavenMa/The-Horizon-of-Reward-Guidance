# 06 Controller Parity and Preference Identification

**Project:** The Limits of Reward Guidance under Finite Adaptation  
**Target:** ICML 2027  
**Prepared:** 2026-10-19  
**Status:** COMPLETE. R1–R5 ran on CPU. Result cells below are measured. A negative or unresolved contrast is left as measured.  
**Verified starting commit:** 03e6a7318556701d35118f7b9c1e207cd5449440  
**Existing research record:** [Work Plan 05](05_Online_Preference_Optimization.md)  
**Primary output:** This SAME work plan, updated with actual tables, figures, code revisions, runtime and raw-data paths.

> **Instruction for the executing agent:** Execute this plan with the repository's existing CPU/Gym/BTL infrastructure. A longer execution is authorized when justified by real scientific contrasts. Do not request approval for normal engineering choices. Work from the frozen existing results, but do NOT change them or select new favorable labels, seeds, thresholds, or teacher weights to make new results positive. A negative result is useful when its mechanism is diagnosed. After running, fill THIS file, commit source/results/filled plan and report the commit.

## 1. What this experiment is trying to establish

### 1.1 Paper-level question

The paper distinguishes (i) a suitable guidance reward at a particular policy state, (ii) a reward trajectory that can actually track a moving useful set under finite adaptation, and (iii) a statistical certificate based on the source and newly obtained binary preference comparisons.

The controlled bridge already showed an exact scalar horizon, a four-output ordinary-PG contact, finite adaptation, sampled-gradient noise, a 16-outcome two-stage MDP with an all-controls computational upper bound, and a 64-outcome three-stage extension. Work Plans 04 and 05 then moved to real tabular MO-Gymnasium RL.

The next bottleneck is NOT lack of another neural-network benchmark. We need to answer:

**When a preference learner gets extra information but true return does not rise, did the information fail, did its confidence set fail to influence guidance, did the controller lack movement capacity, or did the selected guidance lead to a different long-term policy basin?**

The last plan exposed a confound: the source/robust method uses a model-aided one-step exact guidance optimizer while the named true-reward oracle mostly deploys a fixed known coefficient. Both access the same environment evaluator only for measurement, but their CONTROL ACTIONS are chosen by different procedures. This is not a fair way to attribute long-term return differences to preference information alone.

Separately, the three-objective Resource-Gathering study used a numerical confidence-region optimizer whose local-search and fallback behavior are not a formal lower bound on worst-compatible gain. The next plan must fix that before claiming robust certification.

### 1.2 Existing code and measured references

| Existing source | Precisely what exists | What not to infer |
|---|---|---|
| [Work Plan 05](05_Online_Preference_Optimization.md) | P1–P5 filled results and pilot choices | Marked COMPLETE does not mean every empirical interpretation was positive |
| [pref_common.py](../experiments/bridge/pref_common.py) | DST Gym rollout, exact DP, one-step model-aided robust phi, actual REINFORCE, BTL, updated intervals | Source robust and fixed true-w oracle do NOT share a common guidance decision rule |
| [18_dst_live_online.py](../experiments/bridge/18_dst_live_online.py) | 160-step live Gym P3 and pilot; 8 weak and 4 mixed independent runs | Its per-seed compact JSONs omit complete learned logits at most checkpoints and selected raw BTL label records |
| [17_dst_fair_forks.py](../experiments/bridge/17_dst_fair_forks.py) | Early/later and learned-policy forks, MLE/robust, paired sampled check | The point-estimate target and robust one-step lookahead are different controllers, not only different information |
| [P3 weak results](../outputs/horizon_bridge/p3_dst_live_weak/summary.json) | S/U/I/M/O final returns, 160 policy updates | Oracle fixed true reward is NOT oracle guidance lookahead |
| [P3 mixed results](../outputs/horizon_bridge/p3_dst_live_mixed/summary.json) | Four complete runs per method | I–S has substantial variability; no broad improvement claim |
| [P4 results](../outputs/horizon_bridge/p4_budget_timing/summary.json) | Phi-budget and feedback-timing comparisons | Larger b can change finite-horizon optimization; not just a statistical certificate |
| [20_resource_gathering.py](../experiments/bridge/20_resource_gathering.py) | Three-reward environment, live episodes, exact 100-state finite model, 2D BTL, sampled PG and experimental robust branch | Synthetic comparison of a local solver to a grid is NOT a verified global worst-case certificate |
| [P5 results](../outputs/horizon_bridge/p5_resource_gathering/summary.json) | Oracle learns; preference-based runs variable; robust nearly frozen | It is not evidence that valid two-dimensional robust guidance cannot work |

**Frozen DST evidence from P3 weak source:**

| Method in Work Plan 05 | Final exact true return, mean over 8 independent runs |
|---|---:|
| Source-only model-aided robust S | 0.6830690935 |
| Fair uniform feedback robust U | 0.6826537247 |
| Targeted feedback robust I | 0.6777084703 |
| Targeted feedback, correct MLE target tracking M | 0.6514160215 |
| Fixed true-w PG reference O | 0.6378988062 |

At policy update 40, M was about 0.62683 and S about 0.60694. By update 160, M was about 0.65142 and S about 0.68307. This is an observed reversal under the prior two different deployment rules. It does not, by itself, establish that reward ambiguity is preferable to knowing the true reward.

P2 also showed that at the early oracle-developed checkpoint, **converged** informative MLE tracking led to mean 40-step true-return increase 0.06855 versus 0.06126 for informed robust one-step guidance. At the learned update-60 checkpoint, the gap was larger (0.06045 versus 0.03878). This already defeats a blanket claim that robust is always better than MLE.

P4 found that b=0 led to final mean return 0.64242 while b=.02 gave 0.68122 and b=.05 gave 0.71038 (five-run comparison, with high variability at b=.05). Feedback label count above 32 had little additional effect in that specific run. These results motivate aligning controller budgets and studying policy state distribution, not another unstructured label sweep.

**Frozen Resource-Gathering evidence (5 runs each, 100 updates):** known true-reward oracle mean 0.58207, source-only mean 0.22617, Fisher learned-weight mean 0.46040 with SD 0.16975, uniform learned-weight mean 0.43527 with SD 0.21715, experimental robust mean 0.21708. Fisher and uniform learned preferences still had weak held-out log loss near 0.69. They do not establish accurate two-dimensional preference identification.

### 1.3 Stage order

| Stage | One primary change | Question | New evidence |
|---|---|---|---|
| **R1 Controller parity** | All compared information sets use the SAME model-aided bounded phi optimizer | What is the effect of preference information independent of guidance algorithm? | Same-state exact gains, valid numerical controller parity, paired forks |
| **R2 Long-policy-path diagnosis** | Observe and branch the actual 160-update DST policy states | Why do the 40-step rankings reverse at 160? | Checkpoint causal interventions, true-vector return, occupancy and gradient responses |
| **R3 Information-to-action causality** | Give the same evolving policy histories controlled fresh preference interventions | At which feedback times does new information actually change phi and subsequent true gains? | Same-pool information-only and action-enabled forks |
| **R4 Three-objective repair** | Correct 2D preference coverage, global robust lower bound and illegal fallback | Was the Resource-Gathering outcome due to unidentifiable data, overly conservative valid uncertainty, or invalid numerical controller? | Rank-sensitive labels, truthful numeric certification, paired real-RL comparisons |
| **R5 Consolidation** | Translate only supported causal contrasts into simple paper figures/tables | What new claims survive fair baselines and both tasks? | Three clean figures, a short limitations table, reproducible manifest |

R1–R3 all use the SAME already installed DST environment. R4 reuses the EXISTING Resource-Gathering task. **Do not introduce Walker, Meta-World, LLM PPO, another large dataset or a new reward-network architecture in Work Plan 06.**

## 2. Experimental definitions, shared baselines and scientific safeguards

### 2.1 Same controller means same available operations

Let theta represent the existing time-homogeneous tabular softmax policy; its exact vector return is mu(theta) from the already verified 20-step DST DP. For a deployed guidance coefficient phi in [0,1], the EXACT population-PG update is

\[
\theta^+(\phi)=\theta+\eta\nabla_\theta J_\phi(\theta),\quad
J_w(\theta)=w f_1(\theta)+(1-w)f_2(\theta).
\]

For ANY nonempty preference information set U subset [0,1], the **same bounded one-step lookahead controller** chooses

\[
A_U(\theta,\phi_{\rm old})\in
\arg\max_{\phi\in [0,1]\cap[\phi_{\rm old}-b,\phi_{\rm old}+b]}
\min_{w\in U}\left[J_w(\theta^+(\phi))-J_w(\theta)\right].
\]

Use the same policy step size eta, gradient, DP oracle, search tolerance, phi budget b and tie break for EVERY A_U branch. The environment model available for the one-step lookahead is IDENTICAL in each branch. The only changing dimension is U.

Important U choices:

- U_source: original source confidence interval, e.g. [0,1] in weak initial data.
- U_updated: statistically computed confidence interval after genuine new BTL labels.
- U_MLE={w_hat}: singleton using the CONVERGED trained BTL coefficient.
- U_true={w_star}: singleton using the hidden true preference **ONLY** as a privileged lookahead control. This is a fairer control than a fixed true-w gradient, but is still a *myopic one-step* oracle, NOT a globally optimal 160-step guidance sequence.
- A point-estimate **target tracker** (move phi toward w_hat) and the old fixed true-w policy are SECONDARY deployment controls, explicitly distinguished from the shared A_U rule.

Keep the actual single-step policy update sampled or exact according to a named arm. Exact one-step foresight for choosing phi followed by a sampled REINFORCE update is a **model-assisted guidance** method, not model-free end-to-end preference RL. State this clearly in the eventual paper.

### 2.2 Consistency property required of the one-step solver

At a fixed theta and phi_old, and for any feasible candidate phi_alt,

\[
J_{\star}(\theta^+(A_{U_{\rm true}}))-J_{\star}(\theta)
\ \ge\
J_{\star}(\theta^+(\phi_{\rm alt}))-J_{\star}(\theta)
\]

up to the declared numeric optimization error, because U_true is a singleton true preference and the control window is shared. This property holds **only for the same theta/phi_old and exact population update**. It does not compare full trajectories after their states diverge and does not promise the true oracle will have the highest return at update 160.

For a numeric implementation, evaluate endpoints, multiple interior bracket intervals and refined extrema of the one-dimensional EXACT finite-step objective, not just a single local minimize_scalar call or a 21-point grid. Validate the selected value against a dense independent grid plus selected random feasible candidates; report the maximum observed violation of the oracle inequality. If the numeric solver disagrees, fix the solver before downstream comparisons.

### 2.3 What to report for every DST branch

At each policy update and every significant checkpoint record:

1. Policy theta, previous and newly deployed phi, exactly enforced |delta phi|<=b; the no-movement and source-only baselines.
2. Current true return J_star(theta), exact true finite-step gain and exact two-component return vector f(theta).
3. Source-only robust gain on the ACTUALLY EXECUTED policy change; updated robust gain on the same change; if relevant, prospective population-step counterfactual and ex-post sampled-step gain distinctly.
4. Unconstrained finite-step pointwise value max_phi C(phi;U) and constrained bounded-window value; record whether the controller reaches the relevant feasibility region.
5. BTL new labels, true labels observed (bits only), pool composition, candidate Fisher information, bounded MLE, confidence endpoints and coverage event against the hidden teacher for evaluator only.
6. Policy-gradient norm, angle between oracle reward and current deployed-guidance gradients, and state-visitation/treasure-reaching distribution.
7. First source-certificate violation, first updated-certificate violation after feedback, and whether the entire sequence was certified or only the suffix. **Never retroactively repair a previous failure.**
8. The evaluation time horizon 20, learning updates 160 (or otherwise declared branch length), labels, actual Gym episodes and CPU cost. Do not transfer the four-output gamma=.06 to normalized DST policy-update units; use gamma=0 and H3's development gamma_pos=0.0004196952 only as specifically labeled.

### 2.4 Non-negotiable reuse and independence

- Preserve every source and output under Work Plans 01–05, especially the negative P3 long-run findings and P5 exploratory robust failures.
- The P3 per-seed compact JSON stores returns/phis/gains but not all full policy logits and original label bits. Therefore **replay** old seed settings to make new checkpoints, and first confirm that replay reproduces the old returns at updates 40/80/160. Do not invent policy states by interpolating saved returns.
- Replicate label randomness and policy rollout randomness independently in new complete seeds. A number of branches sharing the SAME preference labels does not create that number of statistically independent preference datasets. Report dependence.
- Shared-controller forks must start from the same theta, phi_old, policy learning rate, preference labels, environment transition model and rollout streams. Changing only U is an information-only intervention; changing theta is a state intervention; changing b is a capacity intervention. Do not vary two of these in one headline comparison.
- Keep the true teacher w_star inaccessible to all deployable arms, including query selectors. Only oracle controls and evaluators can access it.
- If a natural experiment produces no difference, identify the specific invariant (same phi choice, bound width, gradient saturation, no compatible preferences, or lack of behavior coverage) and preserve the result as a finding.
- No metric may be called a **formal certificate** if an optimization solver returns a merely feasible/local numeric point instead of the correct min or a valid global bound.
- Negative or null outcomes are not excuses to change the environment until the mechanism is resolved. The next plan changes the analysis and information/deployment interfaces while keeping the environments fixed.

## 3. R1 — Fair Controller Parity and Full-Information Oracle

**Objective:** Remove the most serious model-access and optimization-confounding issue from Work Plan 05.

### R1A. One-step exact-controller comparison (small CPU task)

1. Reuse pref_common.py::DST, its policy gradients, H3's exact DP evaluator and Work Plan 05 corrected BTL MLE/score intervals. Implement a COMMON lookahead function that accepts only theta, phi_old, the preference-set endpoints and b.
2. At the frozen H3 policy checkpoints after oracle-developed updates 5 and 20, and at P3 replayed source-only learned-policy checkpoints 20 and 60, create a test matrix of policy states. Keep prior phi_old as actually saved/replayed, not arbitrary or hidden true w.
3. From the SAME source/targeted label batch for each label seed, evaluate A_source, A_updated, A_MLE, A_true, direct MLE tracker and true-w tracker on precisely the SAME initial theta and phi_old. Use b=.02, with b=.05 only as a bounded-capacity sensitivity.
4. Record chosen phi, phi-window, predicted exact true one-step gain, source/updated certificate, best pointwise gain over [0,1], and the movement left to reach a qualifying phi. The true oracle should be maximal for exact true one-step gain in the shared window; report solver residual.
5. Start with three label seeds as a numerical smoke test, then 8 seeds for early and 5 for later/learned checkpoints. Labels are regenerated independently and candidate pairs come from the SAME checkpoint pool. Do not cherry-pick a label batch that makes A_updated differ.
6. As a small counterfactual, keep theta fixed and change just U from source to updated, with NO policy update. Certificate value may change but true policy return and gradients must not. Then actually apply one exact policy update with the changed phi to measure the causal policy effect.

### R1B. Short paired futures with exact and sampled policy updates

7. Clone each shared initial theta, phi_old and observed labels. Roll each A_U branch forward for **40 exact policy updates**, recalculating the next phi by the identical A_U rule at each resulting theta. Keep the nominal learning rate LR=1, update schedule and b=.02 in every branch.
8. Compare final true return, minimum source certificate, updated certificate, phi time series, and whether source/updated information alone changed the controller. A_true is the matched privileged **one-step optimizer**, not a 40-step globally optimal reward path.
9. On the three most informative matched labels/checkpoint conditions, repeat the 40-update futures with **256 or 512 sampled Gym policy episodes/update** and common seeded streams where possible. Evaluate learned policy using exact DST DP, with independent teacher-label seeds.
10. To compare a fixed true-w policy and an A_true lookahead, explicitly label **optimizer access** as the changed cause. If A_true performs better, do not attribute it to data; both know true w.

**R1 results, stage A (one row per frozen checkpoint class; individual seed CSV required):**

| Checkpoint class | Label seeds | U_source info | U_updated info | phi chosen by A_source / A_updated / A_MLE / A_true | Exact one-step true gain for each | Max oracle inequality violation | Source/updated certificate | Notes |
|---|---:|---|---|---|---|---|---|---|
| H3 oracle-developed update 5 | 8 | width 1, the whole [0,1] | mean width 0.503 | 0.520 / 0.480 / 0.480 / 0.480 | 0.004648 / 0.005374 / 0.005374 / 0.005374 | 0 | −0.001917 / +0.002922 | phi_old=0.5 because this checkpoint has no saved guidance phi. alpha=0.05. Source certificate is below gamma_pos |
| H3 oracle-developed update 20 | 5 | width 1 | mean width 0.582 | 0.520 / 0.496 / 0.480 / 0.480 | 0.000904 / 0.001003 / 0.001068 / 0.001068 | 0 | −0.000474 / +0.000398 | Updated certificate is positive but still below gamma_pos=0.000420 |
| P3 source-only learned update 20 | 5 | width 1 | mean width 0.495 | 0.602 / 0.595 / 0.590 / 0.590 | 0.001974 / 0.002080 / 0.002151 / 0.002151 | 0 | +0.001974 / +0.001997 | Saved seed-0 logits. phi_old=0.6099. A positive worst gain with width 1 means every weight likes this step; it does not identify w |
| P3 source-only learned update 60 | 5 | width 1 | mean width 0.411 | 0.373 / 0.373 / 0.382 / 0.398 | 0.001089 / 0.001089 / 0.001091 / 0.001096 | 0 | +0.001089 / +0.001089 | Updated labels do not move the robust phi |

**R1 results, stage B:**

| Starting checkpoint | Same one-step solver, preference set | Policy updates | Label seeds | True return gain at 40 | Final phi | Source/updated first violation | Difference vs A_true | Sampled validation | Notes |
|---|---|---:|---:|---|---|---|---|---|---|
| Early H3 | U_source | 40 | 8 label draws, one shared policy | +0.05284 | 0.525 | source certificate −0.00192 at the first step | −0.02624, the same path for every draw | seeds 210–212 mean +0.0526 | U=[0,1], so the eight draws are not eight policies |
| Same | U_updated | 40 | 8 | +0.07908 | 0.000 | updated certificate +0.00292 | 0 | sampled mean +0.0788 | Same path as A_true. Drives phi to 0 inside the budget |
| Same | singleton fitted MLE | 40 | 8 | +0.07908 | 0.000 | singleton | 0 | sampled mean +0.0790 | Same path as A_true |
| Same | singleton true w (privileged) | 40 | 1 exact, matched by the rows above | +0.07908 | 0.000 |  | reference | sampled mean +0.0789 | One-step lookahead, not a 40-step optimum. Final phi 0 |
| Same | direct fitted MLE tracker (old-style) | 40 | 8 | +0.07296 | 0.373 |  | −0.00612, 95% t interval [−0.00747, −0.00477] | not run | Moves toward the fitted weight by at most 0.02 |
| Same | fixed true-w tracker | 40 | 1 | +0.07339 | 0.375 |  | −0.00569 | not run | Privileged: phi is set to 0.375 with no budget |
| P3 learned update 60 | U_source / U_updated / MLE / true | 40 | 5 | +0.02825 / +0.02825 / +0.07053 / +0.07443 | 0.277 / 0.277 / 0.746 / 0.658 | source and updated certificates stay equal | MLE minus A_true −0.00391, interval [−0.01156, +0.00375] crosses 0 | not run | Shrinking U does not change the robust action. The MLE contrast is unresolved |

**Possible R1 outcomes and their meaning:**

- A_true one-step always maximizes the exact gain at the same state, while long-run A_true is overtaken by source A_U: the cause is POLICY-STATE TRAJECTORY / nonconvex optimization, not missing information or unfair model access.
- Correct MLE and robust outcomes converge under the same solver: the earlier point-versus-robust contrast was primarily DEPLOYMENT ALGORITHM rather than statistics.
- New BTL labels change the updated certificate but A_updated equals A_source at the same state: additional information had NO action consequence under this speed budget. That is not a failed fitting experiment.
- A_updated changes phi and the exact same-state true gain changes in the predicted direction: an information-to-action mechanism has been directly isolated.
- Numerically computed A_true loses the single-step true-gain comparison against another feasible phi: the scalar solver is wrong; fix before making any long-horizon claim.


## 4. R2 — Explain the 40-to-160 Update Reversal

**Objective:** Explain why the preference-informed MLE method was ahead at 40 updates but lost at 160, and why source-only robust guided policy ends with a higher true return than a fixed true-w ordinary policy-gradient reference. Do not infer the answer from final return curves alone.

### R2A. Deterministic replay and FULL policy-state recording

1. Reuse 18_dst_live_online.py and pref_common.py, with exactly the Work Plan 05 environment version, normalization, teacher, sampled-gradient rule (256 episodes/update), phi budget .02, label events 20/40/60/80, 160 policy updates, seed-generator formulas and the fixed 20-step horizon.
2. **First** replay seeds 0 and 1 for S, I, M and O, and compare exact-DP evaluator true returns at updates 0/40/80/160 against P3 saved JSON. Record maximum replay discrepancy (target <=1e-9 if the same deterministic RNG streams/libraries are available). If the machine/library differs, report it and identify the exact changed component rather than inventing missing theta states.
3. Complete the replay for the **8 old weak-source seeds** of S/U/I/M/O if the initial replay agrees. The original run was only about 132 seconds of CPU wall time, so this is an economical, justified reproduction; do not substitute linear interpolation from saved return curves.
4. At policy updates 0,10,20,40,60,80,120,160 save full logits, phi, the BTL MLE/interval, complete comparison pair identifiers/features/label bits, policy RNG state, optimizer step and exact mu(theta). Use separate output folders, never overwrite the Work Plan 05 per-seed files.
5. Recompute every saved checkpoint's return independently with the H3 exact evaluator; verify live Gym trajectory-reward consistency for a representative start, mid and final policy. Record full metadata/seed/installed package versions.
6. For each checkpoint compute an EXACT probability distribution over terminal outcomes using the finite-horizon transition model. Report the probability of ending at each treasure type, truncation/no-treasure probability, episode length and occupancy of important cells. This distribution is a more informative outcome description than only one scalar return.
7. Save the exact policy gradient at phi=w_star and at the actual deployed phi, their norm/cosine, action entropy and state occupancy. Distinguish "true-w gradient is small at the current checkpoint" from "the true-w gradient would never have learned another policy from a different initialization".
8. Report actual source and updated certificates as a time series with gamma=0 and gamma_pos from H3. Since the source set is often [0,1], early sampled updates may already have negative source certificates. Do not invent a finite positive source horizon if the initial update was uncertified.

### R2B. Split the reversal into early learning and long-run path effects

9. Compute paired per-seed return changes for three fixed segments **0–40**, **40–80**, **80–160** and the area under the full learning curve. The result should reveal whether the initial MLE advantage came from better early movement, and when/why the original robust trajectory overtook it.
10. At every saved checkpoint for each policy method, evaluate the next exact population PG step under the shared controllers A_source, A_updated, A_MLE and A_true, with the SAME current theta and existing phi_old. This is a diagnostic one-step swap, not a separate full online rollout. Record each exact true gain and the within-state oracle optimality gap.
11. Restore at least 3 independent policy seeds at checkpoints 40,80,120 and fork the SAME theta/phi, replaying one consistent preference batch at that checkpoint under the controller alternatives for the remaining 40 updates. Use exact PG first, then a small sampled follow-up where the exact contrast is meaningful. This separates the effect of PRESENT controller choice from the accumulated policy-state history.
12. For a complementary **state dependence** test, apply A_true lookahead to checkpoint policies originally reached by S, I, M and O, without resetting them to uniform. If the gains from perfect-information lookahead differ across those theta states, much of the long-run result is about optimization-path history, not just reward parameter quality.
13. Run a small **same-policy-class** positive control: ordinary true-w PG for 160 and 320 updates, and model-aided A_true lookahead for 160, from the same uniform initial logits and nominal adaptation rules. With the fixed true-w policy, phi can start at true w only if clearly labeled PRIVILEGED initialization. For A_true, start at the same phi0=.5 and apply b=.02. Neither baseline is globally optimal over arbitrary 160-step reward paths; the time-dependent DP optimum 0.77902 is a separate, larger policy-class reference.
14. As a cheap diagnostic (not a new confirmed main benchmark), use a small predeclared constant-guidance sweep phi in {0,.25,.375,.5,.75,1}, 160 exact policy-gradient updates from uniform. Record final J_star and treasure-distribution changes. This exposes whether a fixed known-true gradient followed a particular basin while alternative reward shaping visits another.
15. If a longer 320-update source/true-w comparison is run, preserve its true return without assuming either policy ought to dominate merely because it knew w_star. Long-run optimization and the one-step oracle inequality are logically different.

**R2 stage results:**

| Method | Replayed complete seeds | Max discrepancy versus old P3 returns | J_star at 0/40/80/160 | Segment gains 0–40 / 40–80 / 80–160 | End treasure/outcome distribution | True-w vs deployed-gradient cosine | Notes |
|---|---:|---|---|---|---|---|---|
| Source-only S | 8 | 0 | 0.49994 / 0.60694 / 0.65654 / 0.68307 | +0.10699 / +0.04960 / +0.02653 | at 160: short 0.542, deeper 0.456, truncated 0.002 | 0.351 | Width stays 1. Phi ends at 0.222. Replay matches P3 returns and phis exactly |
| Uniform feedback U | 8 | 0 | 0.49994 / 0.60787 / 0.65681 / 0.68265 | +0.10793 / +0.04894 / +0.02584 | same basin as S |  | Final 0.68265, sd 0.00385. Not distinguishable from S |
| Targeted robust I | 8 | 0 | 0.49994 / 0.60855 / 0.65307 / 0.67771 | +0.10861 / +0.04452 / +0.02463 |  |  | Final 0.67771, sd 0.01103. Below S |
| Targeted MLE M | 8 | 0 | 0.49994 / 0.62683 / 0.64414 / 0.65142 | +0.12689 / +0.01731 / +0.00727 | at 40 short 0.878; at 160 short 0.873, deeper 0.126 | 0.998 at 160 | Ahead at update 40, then the gradient norm falls to 0.006 and the gain stalls |
| Fixed true-w O | 8 | 0 | 0.49994 / 0.62425 / 0.63334 / 0.63790 | +0.12430 / +0.00909 / +0.00456 | at 160 short 0.991, deeper 0.008 | 1 | Phi fixed at 0.375. Same short-treasure basin as exact true-w PG, final 0.63794 |
| New A_true full-info lookahead, same control budget | 1 exact population run | new, not a P3 replay | final 0.63996, phi 0.000 |  | short-treasure basin, same as phi=0 | 1 by construction | Deterministic. Not 8 sampled runs. Exact true-w for 320 updates only reaches 0.64019 |

**R2 mechanism table:**

| Fixed current-policy checkpoint | Method originally producing theta | Source / MLE / true full-info lookahead one-step true gain | Controller target phi movement | 40-step counterfactual final true return | Difference explained by current policy state? | Notes |
|---|---|---|---|---|---|---|
| Update 40 | S | one-step 0.00180 / 0.00180 / 0.00179 / 0.00187 | A_source and A_updated pick the same phi | next-40 gain 0.0480 / 0.0480 / 0.0469 / 0.0326 | Yes. Staying with A_source gains more than attaching A_true | 3 parent seeds. S still has deeper-treasure probability 0.285 |
| Update 40 | I | 0.00182 / 0.00182 / 0.00189 / 0.00189 | source and updated agree | 0.0485 / 0.0485 / 0.0315 / 0.0319 | Yes | Targeted labels have not changed the robust action |
| Update 40 | M | 0.00173 / 0.00178 / 0.00178 / 0.00178 | small | 0.0299 / 0.0332 / 0.0256 / 0.0331, sd about 0.04 | The policy is already near the short-treasure basin | Seed spread is large; the mean gain is below S |
| Update 40 | O | 0.00042 / 0.00047 / 0.00045 / 0.00047 | A_true moves phi, the one-step gap is 0.00005 | 0.0044 / 0.0117 / 0.0064 / 0.0117 | Yes. A better controller cannot recover much return from this state | Gradient at the true weight is already small |
| Update 80 | S |  | A_source equals A_updated | next-40 gain 0.0174 / 0.0174 / 0.0146 / 0.0648 | Attaching A_true now helps this later S state | The early path, not the final controller, created the gap |
| Update 80 | M / O |  |  | M about 0.007 to 0.009; O A_true 0.0043 | Both remain in the low-gain basin | |
| Update 120 | S / O |  | source equals updated on S | S: A_source 0.0078 versus A_true 0.0587. O: A_true 0.0023 | Same split as update 80 | |
| Uniform initialization | Constant phi in {0, 0.25, 0.375, 0.5, 0.75, 1} for 160 exact steps |  | final returns 0.640 / 0.639 / 0.638 / 0.643 / 0.735 / 0.697 | phi=0.75 reaches deeper treasure with probability 0.974. phi=0.375 stays on the short treasure with probability 0.991 | The better basin is available, and it is not the true-weight policy | Exact true-w 160 is 0.63794. A_true from phi_0=0.5 also ends at phi 0 and return 0.63996 |

**Possible scientific results:**

- A_true is one-step best on each shared checkpoint but loses the final 160-step race to source A_U: consistent with **nonconvex path dependence / myopic versus long-horizon reward control**, not "ignorance is better than true reward."
- An alternative fixed guidance parameter beats fixed true-w PG under the same policy class and optimization budget: an empirical reward-shaping effect. Quantify path and treasure distribution; no universal theorem follows.
- Source-only beats others solely because its previous theta states differ: rewards altered the route through policy space. The final return ranking cannot be explained by preference quality alone.
- Once all methods use identical A_U, the reversal vanishes: previous result was a controller mismatch.
- The replay cannot reproduce original runs with identical seeds: classify this as a reproducibility discrepancy, preserve the old record, and run all controlled new comparisons under one explicitly frozen new implementation. Do not patch historical JSON.

## 5. R3 — Identify When Extra Online Feedback Has a Causal Policy Effect

**Objective:** The P3 Fisher selector narrows U far more than fair uniform, yet does not reliably improve long-run true return. Determine where the information-to-action connection breaks. This stage uses matched checkpoint interventions, not yet another unrelated label-budget grid.

### R3A. Controlled information-only branches at frozen policy states

1. Use new saved R2 policy and phi checkpoints at updates 20,40,60,80 (5 seed identities initially, extend to 8 if the measured difference needs it). Each checkpoint is the SAME theta and phi for all methods in its local comparison.
2. Collect one genuinely NEW unlabeled candidate pool per checkpoint: 64 current-policy, 32 uniform explorer, 32 fixed treasure-seeking explorer full Gym trajectories, and about 256 candidate comparison pairs, using the same setup as prior P3. All selection rules may access the same pair features and previous labels but not the hidden true preference.
3. From this single candidate pool select 32 Fisher informative pairs and 32 fair uniform pairs. Draw conditional-independent teacher labels after selection using independent label RNG. Also retain no-new-label control. Cache the candidate features and label bits so the replay is exact. With the same candidate pool, the two selectors need not pick the same pairs.
4. Fit the CONVERGED pooled BTL point estimate w_hat. Construct one source-only interval and one updated interval using the documented conditional fixed-N or simultaneous method; separately report if statistical coverage fails for the hidden teacher.
5. At the identical theta and phi, compute original versus updated certificate and **make NO policy update**. The true return MUST be identical across information-only branches. This measures pure statistical certification.
6. Then run ONE exact policy update using the common A_U controller: replace only U_source by U_updated (Fisher or uniform) and keep the same movement window, exact gradient model and current theta. Record phi movement, actual true gain, source/updated certificate and free/bounded pointwise values. This measures whether the newly acquired information changes a deployed action.
7. If the confidence interval becomes narrower but phi does not move, record the active constraint and slack. In particular, differentiate a robust optimizer unchanged because the same endpoint remains active from a controller physically blocked by the b=.02 movement window.

### R3B. Matched short-horizon effects after information arrival

8. From identical parent states, create three 20-update continuations:
   - Source-only U_source and original policy guidance.
   - Information-only new U for certification, with original guidance and EXACT same policy transitions.
   - New U plus a bounded A_U guidance controller and actual policy updates.
   A fourth branch uses singleton trained MLE under the same A_U optimizer as a deployment control.
9. First run population-exact steps, then 3 independent REINFORCE sample streams at 256 or 512 full Gym trajectories/update. Use common action RNG streams where practical and exact DP evaluation, but do NOT count paired forks sharing the same parent label data as independent preference-learning trials.
10. Retain the usual observation: information-only improves a certificate and leaves true gain unchanged. Distinguish first source failure BEFORE the intervention, first updated failure AFTER it, whole path and post-feedback suffix separately. Extra labels do not retrospectively repair ex-post failures.
11. Add one **already certified initial state** diagnostic if available: select a checkpoint using only a predeclared, true-label-blind source certificate on a development policy and gamma_pos, then test the moving-support claim from that checkpoint. If source certification is absent at every allowable checkpoint, report INITIAL INFORMATION INSUFFICIENT. Do not cherry-pick theta/w/positive gamma until a horizon appears.
12. Run 8 independent label sets and at least 3 parent-policy seeds for the most relevant contrast. Report the dependence structure, e.g. 3 parent policies x 8 label draws, not "24 independent policy learning experiments."

**R3 result table:**

| Shared checkpoint/time | Query rule | Added labels | Independent parent policies / label draws | Width old to new | Certificate change on frozen path | A_U guidance phi change? | One-step exact true gain difference | 20-step true return difference versus source | Notes |
|---|---|---:|---|---|---|---|---|---|---|
| Update 20 | No labels | 0 | 3 parent policies | 1.00 to 1.00 | frozen-step certificate +0.00182 | no | 0 | 0 | Source set is still [0,1]. The positive worst gain means this step helps every weight. P3's update-1 source certificate was negative, so this is not a repaired horizon |
| Same | Fisher from common pool | 32 | 3 parents × 8 label draws | 1.00 to 0.760 | +0.00182 to +0.00185 | phi changes on 58% of draws | +0.00002 | −0.00002 | Return does not move. Singleton MLE's 20-step gain is +0.0092 versus source, positive on all 3 parents and all 8 draws |
| Same | Fair uniform same pool | 32 | 3 × 8 | 1.00 to 0.942 | unchanged at +0.00182 | phi changes on 29% of draws | 0 | 0 | Less movement than Fisher, and still no return change |
| Update 40 | Fisher vs fair uniform | 32 | 3 × 8 | Fisher 0.631; uniform 0.915 | Fisher +0.00162 to +0.00166 | Fisher moves phi on 71% of draws; uniform on 13% | 0 | Fisher −0.00021; uniform 0 | Sampled REINFORCE, 9 streams: source +0.02983, updated +0.02922 |
| Update 60 | Fisher vs fair uniform | 32 | 3 × 8 | Fisher 0.758; uniform 0.969 | +0.00108 to +0.00109 | Fisher 46%; uniform 4% | 0 | Fisher +0.00005 | MLE minus source has parent means +0.016, +0.010, +0.009, but some label draws are negative |
| Update 80 | Fisher vs fair uniform | 32 | 3 × 8 | Fisher 0.747; uniform 0.952 | +0.00063 to +0.00064 | Fisher 50%; uniform 13% | 0 | 0 | Robust information still does not change the 20-step return. Later MLE gaps are not uniform across draws |

**R3 finer diagnosis:**

| Observation at a given matched checkpoint | Confirmed by what number? | What this establishes | Follow-up needed? |
|---|---|---|---|
| U shrinks but the source set was already the whole interval | Width stays 1.00 until new labels; P3 update 1 was uncertified | The original labels did not identify w. A later positive worst gain is a property of the policy step | No horizon is reported |
| U shrinks and A_U sometimes changes phi | Fisher changes phi on 46–71% of draws. The 20-step return gap stays inside 0.0002 | A different endpoint inside the budget need not change true return | None for the robust controller |
| Singleton MLE does change the next 20 exact steps at update 20 | +0.0092, range +0.0079 to +0.0100, all 3 parents | A point-estimate controller can move return when the robust set does not | Later checkpoints are mixed: parent means stay positive, individual draws cross zero |
| Sampled and exact agree that Fisher versus source does not raise return | Update-40 sampled gap −0.00061 | The null is not an exact-versus-sampled artifact | No |
| These 24 rows are 3 policies times 8 label draws | Dependence is reported as such | They are not 24 independent policy experiments | No |

**Possible results:** The most valuable P3-negative outcome may be that new BTL labels are accurate, but the source-only robust controller already chooses a phi that traverses a better region of policy space under this optimizer. A direct same-state switch and following 20-step fork can establish whether this is the cause, without claiming that less information is universally preferable.


## 6. R4 — Repair Three-Objective Identification and Robust Optimization

**Objective:** Revisit the already-installed MO-Gymnasium Resource-Gathering task without the invalid implication that a local SLSQP worst-case solution is a formal certificate. Determine whether the observed weak preference learning arises from lack of independent preference directions, unreliably computed confidence sets, or genuinely conservative reward guidance.

### 6.1 Frozen task and the previous counterexample

Keep the exact environment and normalization in 20_resource_gathering.py:

- Three reward channels with low [-1,0,0] and high [0,1,1].
- True BTL teacher preference [0.2,0.2,0.6], beta=1; do not change to an easier teacher.
- Tabular policy with 100 modeled states, four actions and horizon 40.
- Initial per-policy-update guidance-coordinate budget b=.02, and the already established 100-update live Gym training protocol.
- Existing source / Fisher / uniform / oracle runs and all results preserved.

The previous source-only mean was 0.22617; Fisher-targeted learned MLE mean 0.46040 with substantial SD 0.16975, uniform learned MLE mean 0.43527 with SD 0.21715, and known-true-reward oracle mean 0.58207. Experimental robust mean was 0.21708. These are **measured old results**, not new outcomes and not proof that validated robust learning fails.

**Current implementation issues that MUST be resolved before R4 certification:**

1. score_region_ok performs a solver/grid comparison on a synthetic random BTL design; it does not globally verify robust minimization on the ACTUAL Resource-Gathering comparison pool.
2. The _worst method uses local SLSQP and, when no valid solve is found, falls back to objective evaluation at [0.2,0.2], the actual hidden teacher's first two coordinates. This MUST BE REMOVED from the new code. Failure of local optimization cannot justify using the teacher as a generic worst-case value.
3. The current scalar information selector scores total Fisher trace. In two free preference dimensions it can concentrate information on one direction and leave the orthogonal direction unidentified.
4. The sampled gradient cosine against exact PG was approximately .45 at 256 and 512 episodes/update. Sample-count escalation alone was not shown to fix that. This is a real separate policy-gradient noise consideration.
5. A numerical grid or a feasible local minimizer supplies an **upper bound** on the true worst-case MINIMUM; using that value as a lower bound to certify progress can be unsound.

### R4A. Genuine two-direction preference information

Let w=(w1,w2,w3), w3=1-w1-w2, with the continuous simplex \(\Delta_2=\{w_1,w_2\ge0,\ w_1+w_2\le1\}\). For trajectory feature difference D=(D1,D2,D3), define

\[
c_i=\beta D_{i,3},\quad
d_i=\beta(D_{i,1}-D_{i,3},D_{i,2}-D_{i,3}),\quad
P(Y_i=1\mid w)=\sigma(c_i+d_i^\top (w_1,w_2)).
\]

1. Reuse RG, its exact finite model, live_batch and original convex 2D BTL optimizer from 20_resource_gathering.py. Validate model-state encoding and raw reward transitions against multiple held-out live Gym policies, not only the uniform one. Cross-check its analytic policy gradient with finite differences in multiple policy states and compare sampled REINFORCE gradients under equal reward scaling.
2. Before new preference labels, build a shared, **teacher-blind** unlabeled trajectory pool with behavior policies designed to represent the three known objectives: a uniform explorer, a policy biased toward resource/gold collection, a policy biased toward gem/diamond collection, and one that avoids enemy risk where possible. Their behavior logic must be stated and derived from the documented environment map, NOT the hidden teacher weights. Save visitation, reward-vector histograms, resource-acquisition and death frequency.
3. Compute the matrix \(F=\sum_i p_i(1-p_i)d_i d_i^\top\) at the current PREVIOUSLY FIT w_hat for each candidate preference query. Report both eigenvalues, effective rank and condition number. A larger trace alone is not evidence that BOTH directions became identifiable.
4. Use the SAME candidate pool at a matched checkpoint to compare 64 additional labels with:
   - fair uniform selection from all candidate pairs;
   - the old trace-Fisher greedy rule;
   - a **minimum-eigenvalue greedy rule**: sequentially choose a candidate improving the smallest eigenvalue of \(F_{\rm prior}+\sum_{selected}p_i(1-p_i)d_i d_i^\top\), with a small fixed ridge to break exact ties only. The real teacher does not enter the score;
   - the original low-sensitivity 32-label source, and a separate mixed-source baseline selected uniformly from the full candidate pool.
5. Use 20 independently drawn TEACHER-label repetitions on each fixed candidate/query design to quantify estimator uncertainty separately from costly policy learning. These are label-only replications, NOT 20 independent policy runs. Hold back a disjoint set of balanced objective-tradeoff comparisons for BTL log loss/Brier and a naturally sampled comparison set for ecological coverage.
6. Fit a **converged 2D simplex BTL MLE** using constrained convex optimization, and independently verify its KKT residual/likelihood via another optimizer. Log parameter error and whether a boundary solution is genuinely supported or merely numerical.
7. Record w error, held-out log loss, two Fisher eigenvalues, and fixed-N confidence-region width/support in several directions. If the source/mixed labels cannot span both preference directions, report that as identification failure, not as a neural-model problem.
8. If the new behavior policies still produce virtually no tradeoffs along a reward component, identify which component cannot be inferred from actually accessible trajectories. Do not change the teacher to make it learnable.

**R4A results table:**

| Source/query design, same candidate pool | Independent label repetitions | Additional labels | Fisher eigenvalues min/max | MLE L2 preference error | Held-out balanced/natural BTL NLL | Confidence region support or width | True weight covered? | Notes |
|---|---:|---:|---|---|---|---|---|---|
| Original weak source | 20 | 0 | 0 / 0 | L2 0.327 | balanced 0.676, natural 0.698 | 441/441 grid points, the whole simplex | Vacuous | Both eigenvalues are zero. The fit stays at (1/3, 1/3). SLSQP and the softmax solver agree |
| Mixed source | 20 | 0 | 1.14 / 12.47, condition 11.0, rank 1.18 | L2 0.493 | 0.671 / 0.701 | 410 grid points | Not informative | One large direction, a weak second |
| Weak + fair uniform | 20 | 64 | 1.71 / 30.38, condition 17.8, rank 1.11 | L2 0.473 | 0.666 / 0.701 | 405 grid points | Not informative | Uniform labels do not identify both components |
| Weak + old trace Fisher | 20 | 64 | 0 / 74.5, rank 1.00 | L2 0.175, the smallest error | 0.657 / 0.695, the best held-out scores | 325 grid points | A wide band, not a point | The orthogonal component cannot be inferred. Trace is not two-dimensional identification |
| Weak + minimum-eigenvalue selection | 20 | 64 | 6.94 / 9.74, condition 1.40, rank 1.94 | L2 0.428 | 0.669 / 0.696 | 385 grid points | Still a large set | Both eigenvalues are nonzero, and the weight error and held-out NLL are worse than trace |
| Mixed + minimum-eigenvalue selection | 20 | 64 | 7.77 / 23.78, condition 3.06, rank 1.59 | L2 0.328 | 0.665 / 0.696 | 397 grid points | Still a large set | Same pattern. Finite-difference checks of the analytic policy gradient match to about 1e-9 |

### R4B. A genuinely conservative robust lower bound over the continuous simplex

The scalarized finite-step return gain for a fixed policy update and unknown preference is affine in w:

\[
G(w)=w_1\Delta f_1+w_2\Delta f_2+(1-w_1-w_2)\Delta f_3.
\]

For each predeclared independent BTL label batch B, define the score vector

\[
S_B(w)=\sum_{i\in B}d_i[Y_i-\sigma(c_i+d_i^\top w)]
\]

and a conservative conditional fixed-batch region

\[
U_B(\alpha_B)=\left\{w\in\Delta_2:
|S_{B,j}(w)|\le
\sqrt{\frac12\log(4/\alpha_B)\sum_{i\in B}d_{ij}^2}
\text{ for }j=1,2\right\}.
\]

For online source + two feedback batches, predeclare \(\alpha_B=0.05/3\) for each and intersect the three regions for a conservative simultaneous set under independent conditional BTL labels and selection-before-labeling. Keep pooled MLE fitting separate from this compatibility-set calculation. If no labels contain a direction, that dimension of the region may remain broad.

**Required robust global calculation:**

1. Implement an independent two-dimensional **outer-cover** solver for the *actual* resulting confidence region. Begin with a triangulation of the entire simplex. For every triangle, use interval bounds on each score component to determine whether any point in that triangle COULD satisfy all score inequalities. Discard a triangle only when at least one score constraint is IMPOSSIBLE throughout that entire triangle; keep every ambiguous triangle. Subdivide the ambiguous triangles as needed.
2. Because the union O of retained triangles is a SUPERSET of the true feasible U, for affine gain G,
   
   \[
   L_{\rm robust}=\min_{w\in O}G(w)\ \le\ \min_{w\in U}G(w).
   \]

   The minimum over each retained triangle is at a vertex, so L_robust is cheap once the outer cover is built. It is a **conservative LOWER bound on worst-compatible gain**, whereas a locally feasible SLSQP solution is an UPPER bound on that minimum. Do not swap the inequality.
3. Compute a feasible-point upper bound using points explicitly verified to satisfy the ACTUAL continuous score constraints (for example an independent local solver with feasibility checks). Report upper minus lower as the numeric optimization uncertainty.
4. Use reliably conservative interval arithmetic/rounding for the score enclosures if calling the lower bound computer-verified. Otherwise label it a numerical conservative approximation with stated epsilon and do NOT make a formal mathematical certification claim. A coarse grid, SLSQP success flag, or agreement on a synthetic design is not sufficient.
5. Never use [0.2,0.2], W_STAR or any evaluator-only preference as a fallback for an optimizer failure. If the confidence region cannot be safely resolved, use the **FULL SIMPLEX** as a transparent, valid but potentially extremely conservative set: its worst compatible gain is exactly the minimum among the three simplex vertices. Log this fallback and its consequence. If the intended region is empty, separately report invalid/empty region and why the fallback was needed.
6. Test the outer-cover code on synthetic convex and disconnected diagnostic score regions, on no-information full simplex, on singleton-like concentrated designs, and on all relevant actual label batches. For every numerically found feasible w, verify \(L_{\rm robust}\le G(w)\). More random feasible points are consistency checks, not proof.
7. Build the bounded guidance selector from the exact finite-horizon **model-based population** policy-gradient update and this lower bound. Candidate guidance weights themselves lie in the same 2D simplex and must move under the same per-coordinate maximum step b=.02 as the point-estimate competitor. Use a documented two-dimensional search with a clear lower-bound objective; distinguish a feasible candidate controller from a globally optimal controller if outer optimization is incomplete.
8. Compare same-checkpoint controller A_source, A_updated, singleton converged MLE and singleton true w using the SAME model lookahead and same movement budget as the DST R1 concept. Separately compare direct MLE target tracking. This separates label information from controller/model access.

**R4B diagnostic table:**

| Real BTL batch / policy checkpoint | Region nonempty? | Triangles/bounds tested | Safe lower bound on worst gain | Numerically feasible upper bound | Upper−lower gap | Fallback count | Formally computer-verified or numerical only? | Notes |
|---|---|---:|---|---|---|---:|---|---|
| Weak source, first policy | yes, the whole simplex | 1 | −3.268e-5 | −3.268e-5 | 0 | 0 | numerical enclosure, not formal interval arithmetic | The score test does not remove any vertex |
| Weak+uniform, same policy | yes | 593 | −3.268e-5 | −3.268e-5 | 0 | 0 | numerical only | Subdividing does not raise the lower bound above the worst simplex vertex |
| Weak+trace Fisher, same policy | yes | 1169 | −3.268e-5 | −3.268e-5 | 0 | 0 | numerical only | Same lower bound |
| Weak+minimum-eigenvalue, same policy | yes, this draw keeps the whole simplex | 1 | −3.268e-5 | −3.268e-5 | 0 | 0 | numerical only | A nonzero Fisher eigenvalue does not by itself shrink this score region |
| Synthetic concentrated design | yes | 49 | −0.099219 | −0.098154 | 0.001065 | 0 | numerical only | 0 violations on 400 feasible random points. The 1e-4 target was not met; the achieved gap is reported |
| Full-simplex fallback reference | always | 1 | −3.268e-5 | −3.268e-5 | 0 | n/a | exact affine minimization | The real batches do not improve on this reference. The teacher weight is never a fallback |

### R4C. Measured policy-learning consequences in the SAME Resource-Gathering task

9. Reuse the existing exact 100-state model for evaluator true return and the actual Gym environment for policy rollouts. Verify the extracted model and analytic policy gradients against fresh live rollouts and finite differences before interpreting a new method.
10. Run a small **gradient precision diagnostic** with 256, 512, 1024 and 2048 complete live episodes/update at two fixed policy states. Use exact model gradient as reference. The previous cosine near .45 at 256/512 is a concern. Choose 512/1024/2048 based on gradient direction and actual throughput BEFORE comparing preference algorithms; keep this choice fixed for all methods. If even large batches remain noisy, report that and add exact-gradient control arms rather than repeatedly increasing sample size without purpose.
11. Complete **8 independent full policy-and-label seeds** of 100 updates (not 8 separate policy streams sharing the same teacher labels) for source-only MLE, same-pool uniform MLE, trace-Fisher MLE, min-eigenvalue MLE, and known-true-w control. Add the new safe robust outer-set controller only if the region solver and numerical worst-case gap are meaningful; otherwise run and label the full-simplex conservative fallback separately.
12. Use the SAME 32 source +64 new labels (two scheduled batches) and full source/query candidate pool composition for every new-feedback method. Select informative pairs using only previous fitted information. Do not change teacher w_star=[.2,.2,.6] or BTL beta=1.
13. At each policy update record exact three-component return vector, true scalar return and true one-step gain, current deployed guidance, fitted 2D BTL weight and info eigenvalues. For the robust controller, report BOTH conservative worst-case progress lower bound and actual true gain.
14. If robust guidance does not learn while point MLE learns, inspect whether the valid compatibility set is too broad to reward resource-seeking, whether its maximin action is cautious, whether the guidance movement budget binds, or whether the sampled gradient is too noisy. Any one of these is a useful grounded limitation.
15. Report the results under b=.02 only, with one b=.05 development sensitivity if the movement budget is clearly binding. Do not silently alter reward scaling or introduce an entirely new neural architecture.

**R4C policy results:**

| Real RG method, same 3-channel teacher | Independent full policy+label runs | Actual new labels | Episodes per policy update | Initial/final true scalar return mean ± SD | 3 reward component changes | BTL error and min Fisher eigenvalue | Robust certificate / solver status | Notes |
|---|---:|---:|---:|---|---|---|---|---|
| Source-only MLE | 8 | 0 | 2048 | 0.2059 → 0.2266 ± 0.0009 | death +0.018, gold −0.001, diamond +0.029 | L2 0.327, min eigenvalue 0 | no region tighter than the simplex | Unidentified. The deployed weight stays near (1/3, 1/3) |
| Fair uniform MLE | 8 | 64 | 2048 | 0.310 ± 0.118 | −0.009 / +0.013 / +0.172 | L2 0.547, min eigenvalue 0.679 | point-estimate tracker | Overlaps the other preference arms |
| Trace-Fisher MLE | 8 | 64 | 2048 | 0.392 ± 0.193 | +0.024 / −0.003 / +0.303 | L2 0.485, min eigenvalue 2.50 | point-estimate tracker | Higher mean than uniform, and the seed sd covers the gap |
| Min-eigenvalue MLE | 8 | 64 | 2048 | 0.423 ± 0.178 | +0.007 / +0.004 / +0.358 | L2 0.409, min eigenvalue 2.48 | point-estimate tracker | Minus uniform is about +0.113 with a rough SE of 0.075. Unresolved |
| Safe outer-region robust, if meaningful | not run as a separate arm |  |  |  |  |  | the cover was not tighter than the simplex | Running it would repeat the full-simplex arm |
| Full-simplex conservative fallback | 8 | 64 collected, not used by the set | 2048 | 0.2145 ± 0.0003 | +0.010 / +0.007 / +0.009 | min eigenvalue 2.91 on the unused labels | valid and too conservative | Does not learn. This replaces the old teacher-coordinate fallback |
| Fixed true-w ordinary PG | 8 | oracle-only | 2048 | 0.5873 ± 0.0112 | +0.018 / −0.000 / +0.630 | n/a | privileged | Agrees with the exact-gradient control, 0.5889 |
| Full-information model-aided guidance oracle | 8 | oracle-only | 2048 | 0.7545 ± 0.0011 | +0.052 / −0.012 / +0.901 | n/a | privileged one-step lookahead | Same controller family as A_true. It is not ordinary PG, and it is not a multi-step optimum |

**Possible R4 conclusions:**

- The old trace-Fisher method had high total information but a very small minimum eigenvalue, and min-eigenvalue queries repair w estimation: the problem was **multi-dimensional identifiability**.
- Correctly inferred preferences increase policy return with a proper deployment controller while source-only does not: meaningful second-task preference-learning evidence.
- Source pairs and candidate pool contain too few resource tradeoffs, even with selected queries: the environment's available preference data are uninformative in one direction; report its action/visitation cause.
- Correctly certified region is broad and robust guidance stalls, while the MLE takes useful but uncertain actions: legitimate **statistical robustness-versus-true-return tradeoff**, provided the numeric worst-case bound is conservative.
- Full-simplex robust fallback stalls but tighter valid sets help: the old near-static robust result was about uncertainty geometry, not solver correctness.
- A numerical solver fails: keep actual MLE/return results; never present an unverified global worst-case value as certified.

## 7. R5 — Publishable Evidence and Limitations, Not Another Benchmark

**Objective:** Use R1–R4 to decide exactly what deserves a main-paper experimental claim. Create at most three readable figures, one small main comparison table and a clear note about unsupported scope. No figure should be interpreted as a new experiment.

### R5A. Simple main figures

1. **Fair guidance controller:** Plot exact true return over policy updates under U_source, U_updated, singleton MLE and singleton true-w ALL controlled by A_U, together with the fixed-true-w ordinary PG reference. The legend should distinguish model-assisted lookahead from fixed reward deployment; include a small table of same-state one-step optimum inequality.
2. **DST path explanation:** Plot actual 160-step true-return curves for S/I/M/fixed true-w/A_true using the **replayed** policy states. Mark the predeclared feedback updates, show at most one uncertainty ribbon/paired-effect panel, and annotate the first 40-step-to-160-step ordering reversal and its measured treasure/outcome-distribution cause. A tiny inset of phi can help only if it materially explains the effect.
3. **Three-objective identification:** Show the two BTL information eigenvalues for uniform/trace/min-eigenvalue query strategies, with a small adjacent real-policy return or robust lower-bound summary. The caption must state whether the 2D robust result is formally computer-verified or merely conservative numeric.

Use compact paper-friendly typography and clear axis units. Do NOT create 20 redundant debug figures or bury a null result.

### R5B. Claims matrix

| Candidate paper claim | Required support | YES / PARTIAL / NO | Numeric evidence and file | Exact valid scope / caveat |
|---|---|---|---|---|
| Source-only and full-information guidance can be compared fairly under SAME exact one-step controller | R1 matched A_U, oracle inequality | YES | Max oracle and dense-grid violations are 0. `r1_controller_parity/summary.json` | Early H3: updated, MLE and A_true share the +0.079 path; source stays at +0.053. Learned update 60: shrinking U does not change the robust path. A_true is one-step lookahead, not a long-horizon optimum |
| Known true reward coefficient is not necessarily best for finite-horizon NONCONVEX policy training | R2 fixed true-w versus A_true and source, policy-state analysis | YES | Exact true-w 0.63794, A_true 0.63996, constant phi=0.75 reaches 0.735, source path 0.683. `r2_dst_paths/summary.json` | This is a basin result. It does not say that an unidentified reward is a better reward. Both true-w and A_true from a uniform start enter the short-treasure basin |
| New BTL labels improve certification, but may not change guidance decisions | R3 same theta, same phi, new interval | PARTIAL | Fisher cuts width from 1.00 to about 0.63–0.76 and changes phi on up to 71% of draws, while the 20-step return gap stays inside 0.0002. `r3_information_forks/summary.json` | On H3 update 5 the same kind of shrinkage does change the 40-step return, because the source set is the whole interval and its certificate is negative |
| New preference information changes control and true progress under an explicitly stated condition | R3 selected matched forks | PARTIAL | At update 20 the singleton MLE gains +0.0092 over source on all 3 parents and all 8 draws. Robust updated-versus-source is about 0 | The condition is the singleton MLE controller at the update-20 source policy, not the robust set, and not the later checkpoints |
| 160-step DST return reversal has a located mechanism | R2 policy occupancy/gradient/path and swap experiment | YES | M is ahead at update 40 (0.627 versus 0.607) and behind at 160 (0.651 versus 0.683). M's short-treasure probability is 0.878 at update 40; S still has deeper-treasure probability 0.285 and keeps a larger gradient | Forks agree. On the update-40 S state, continuing A_source gains +0.048 over the next 40 steps; A_true gains +0.033. On the stuck O state, A_true gains only +0.012 |
| Two-dimensional reward preference can be learned from trajectory comparisons in Resource-Gathering | R4 minimum eigen and held-out log loss, true teacher error | NO | Min-eigen raises the small eigenvalue from 0 to 6.94 in the label study, but its L2 error is 0.428 and its held-out NLL is worse than trace. Online return 0.423 ± 0.178 overlaps uniform and trace | Trace identifies one direction only. The orthogonal component remains unidentified. Live weight error stays near 0.4 |
| RG robust worst-case certificates are globally conservative | R4 continuous-region outer solver and bounded support gap | PARTIAL | Real-batch lower bounds equal the three-vertex simplex minimum, gap 0, 0 audit violations. `r4_rg_robust/summary.json` | Numeric enclosure, not outward-rounded interval arithmetic. The synthetic concentrated gap is 0.001065, short of the 1e-4 target. The simplex controller's true return stays at 0.2145 |
| Any DST feedback/robust performance advantage generalizes to RG | R4 controlled 8 independent full runs | NO | Source 0.227, simplex robust 0.215, min-eigen 0.423 ± 0.178, ordinary true-w 0.587, privileged lookahead 0.755. `r4_rg_robust/live_summary.json` | The only large RG gain is the privileged model-aided lookahead. Preference information does not reproduce the DST source-versus-true-w ranking |

### R5C. What MUST remain separate

- Full-information **one-step** lookahead is not a globally optimal multi-update policy/reward controller.
- Exact model-based guidance is not model-free preference RL. It accesses environment dynamics beyond the BTL comparison labels.
- P3 recorded a NEGATIVE mean 160-step targeted-feedback gain under one specified learner. Do not rewrite that negative result or replace it with newly selected favorable runs.
- The H1 all-controls result belongs to the constructed **16-outcome finite model**; it is not automatically true for DST/RG.
- Realized sampled finite-step certificates differ from population-gradient counterfactuals; do not use one without naming it.
- A label confidence set can fail to cover the teacher on a rare sample; a single empirical coverage fraction is not a mathematical guarantee.
- A local optimizer optimum for a 2D unknown reward set is NOT by itself a globally valid lower bound. Use a mathematically supported outer cover or mark the certificate unverified.
- A source dataset too weak to certify progress at initialization cannot create a positive support horizon that later expires.
- A faster final return with different guidance policies does not imply one policy is uniformly better under the true reward across all training budgets.

## 8. Runtime, Statistical Discipline and Resource Allocation

The exact two-task models are small. The existing P3 main weak source run (8 seeds ×5 methods ×160 updates at 256 live Gym episodes/update) took about 132 seconds in its recorded environment, while old P5 was about 310 seconds. R1 and R2 exact DP studies should be inexpensive compared with the original GPU experiments. R4 outer subdivision/robust optimization may take longer; choose computational precision based on the scale of the actual robust-gain margin.

**New plan compute outline:**

| Stage | First development/pilot | Main intended replication | CPU/GPU and likely bottleneck | Measured pilot sec | Measured full sec | Peak RAM / VRAM | Notes |
|---|---|---|---|---|---|---|---|
| R1 controller parity | 3 label seeds, 4 checkpoints, one-step exact | 8 early +5 later/learned, 40 exact steps + 3 sampled checks | CPU DP and phi scalar search | included in the exact run | exact 80.01; sampled 21.18 | 124028 KiB | No GPU |
| R2 policy path | Replay seeds 0–1 first, exact checkpoint validation | 8 seed P3 replay across S/U/I/M/O; 40-step fork and small true-w control suite | CPU live Gym, exact occupancy | the full replay is the check; max return gap 0 | 235.91 | 98876 KiB | Six worker processes for the replay |
| R3 feedback causality | 1 parent checkpoint, source/fisher/uniform paired info-only | 3 parent policies ×8 label draws at selected times, exact+3 sampled short forks | CPU BTL and 20-step live/DP |  | 508.28 | 96848 KiB | |
| R4A preference information | 20 label-only replications on a frozen real trajectory-pair pool | Uniform vs trace vs min-eigen, weak and mixed source | CPU 2D convex BTL |  | 1.17 | 97640 KiB | |
| R4B global lower bound | One actual BTL batch, coarse outer cover | Actual batches plus a synthetic concentrated set | CPU triangle subdivision |  | 0.46 | 97140 KiB | Not the longest stage. The real regions collapsed to the simplex |
| R4C live RG learning | cosine at 256/512/1024/2048 on two policies | 8 seeds × 8 methods × 100 updates | CPU live Gym and exact DP | uniform cosine 0.429 at 256 and 0.804 at 2048, so 2048 was frozen | 661.34 | not recorded | Includes the exact-gradient true-w control |
| R5 paper results |  | three figures | CPU plots |  | a few seconds |  | `27_plan06_figures.py` |

**Numerical target for R4B:** begin with a conservative outer-versus-feasible-point robust-gain gap target of 1e-4 per policy update (in normalized task units), and tighten only where the claimed margin to gamma is comparable. If the set is broad and this is too expensive, report the achieved safe bound and gap; avoid fake precision. Formal interval verification requires a documented reliable outward-rounded implementation, not just a grid.

**Statistical reporting:** For each paired DST/RG method comparison, report all seed-level differences, mean, SD and a transparent paired 95% interval (t or bootstrap, stating which). Eight runs still give limited precision. If a confidence interval crosses zero, report that the effect is unresolved, not that it is absent or positive. Label-only replications on frozen candidate pools are NOT independent policy replications.

**Do not invent runtime estimates.** Record actual measured CPU/RAM and tasks' statuses; allow longer execution for a scientifically necessary 2D solver but do not spend hours on a result that cannot improve a bound. If an individual task is stalled, continue the other independent experiments, list the exact blocker and use the conservative full-simplex controller in place of an invalid local worst-case certificate.

Recorded environment: numpy 2.5.4, scipy 1.18.1, gymnasium 1.4.0, mo-gymnasium 1.3.2. All of R1–R5 ran on CPU. The uniform-policy cosine was 0.429 at 256 episodes and 0.804 at 2048, so R4C froze 2048 before the preference comparison. The exact-gradient true-w arm (0.5889) agrees with the 2048-episode ordinary true-w arm (0.5873).

## 9. Proposed Code and Output Layout

The following paths are implementation assignments; they do NOT already exist merely because they are in this document.

~~~text
experiments/bridge/22_guidance_controller_parity.py
experiments/bridge/23_dst_path_replay.py
experiments/bridge/24_dst_information_forks.py
experiments/bridge/25_rg_identification.py
experiments/bridge/26_rg_robust_outer.py
experiments/bridge/27_plan06_figures.py
experiments/bridge/28_rg_live_comparison.py

outputs/horizon_bridge/r1_controller_parity/
outputs/horizon_bridge/r2_dst_paths/
outputs/horizon_bridge/r3_information_forks/
outputs/horizon_bridge/r4_rg_identification/
outputs/horizon_bridge/r4_rg_robust/
outputs/horizon_bridge/r5_paper_results/
~~~

Suggested entrypoints. Implement reasonably equivalent CLI if a more modular library is needed, but keep one clear reproduction command per stage.

~~~bash
python experiments/bridge/22_guidance_controller_parity.py --stage exact --out outputs/horizon_bridge/r1_controller_parity
python experiments/bridge/22_guidance_controller_parity.py --stage sampled --out outputs/horizon_bridge/r1_controller_parity
python experiments/bridge/23_dst_path_replay.py --stage replay --out outputs/horizon_bridge/r2_dst_paths
python experiments/bridge/23_dst_path_replay.py --stage diagnose --out outputs/horizon_bridge/r2_dst_paths
python experiments/bridge/24_dst_information_forks.py --out outputs/horizon_bridge/r3_information_forks
python experiments/bridge/25_rg_identification.py --out outputs/horizon_bridge/r4_rg_identification
python experiments/bridge/26_rg_robust_outer.py --out outputs/horizon_bridge/r4_rg_robust
python experiments/bridge/27_plan06_figures.py --out outputs/horizon_bridge/r5_paper_results
~~~

Save compact per-seed JSON/CSV and enough real checkpoint metadata to re-evaluate the exact curves; don't commit huge episode dumps or model weights. Keep labels/pair IDs for reproducible query decisions. Use atomic seed checkpoints for longer RL runs. Each stage must update a summary.json, seed CSV and the results in THIS plan. Historical outputs/results/reference and Work Plans 01–05 remain unchanged.

**Result manifest, fill after execution:**

| Stage | Summary path | Raw seed/checkpoint files | Figure if useful | Source commit SHA | Reproduction command | Notes |
|---|---|---|---|---|---|---|
| R1 | `outputs/horizon_bridge/r1_controller_parity/summary.json` | `one_step_seeds.csv`, `futures_seeds.csv` | `r5_paper_results/r1_same_controller.png` | filled in this commit | `python experiments/bridge/22_guidance_controller_parity.py --stage exact` then `--stage sampled` | alpha=0.05. H3 phi_old=0.5 |
| R2 | `outputs/horizon_bridge/r2_dst_paths/summary.json` | `replay.json`, `forks.json`, `swaps.json`, `checkpoints/*.npz` | `r5_paper_results/r2_replay_curves.png` | filled in this commit | `python experiments/bridge/23_dst_path_replay.py` | Replay gap against P3 is 0. A_true is one exact run |
| R3 | `outputs/horizon_bridge/r3_information_forks/summary.json` | `rows.json` |  | filled in this commit | `python experiments/bridge/24_dst_information_forks.py` | alpha=0.01. 3 parents × 8 draws |
| R4A | `outputs/horizon_bridge/r4_rg_identification/summary.json` | `rows.json` | `r5_paper_results/r4_identification.png` | filled in this commit | `python experiments/bridge/25_rg_identification.py` | 20 label repetitions, not 20 policies |
| R4B | `outputs/horizon_bridge/r4_rg_robust/summary.json` | the summary rows | the same identification figure | filled in this commit | `python experiments/bridge/26_rg_robust_outer.py` | Numeric cover. Teacher fallback count is 0 |
| R4C | `outputs/horizon_bridge/r4_rg_robust/live_summary.json` | `live_rows.json` | the same identification figure | filled in this commit | `python experiments/bridge/28_rg_live_comparison.py` | 2048 episodes after the cosine pilot |
| R5 | this work plan | the three png files | `r1_same_controller.png`, `r2_replay_curves.png`, `r4_identification.png` | filled in this commit | `python experiments/bridge/27_plan06_figures.py` | |

## 10. Completion Status and Execution Handoff

**The completed version of THIS Work Plan 06 is the primary handoff.** An extra report is optional. No results are prefilled for the new experiments; historical Work Plan 05 references earlier in the document are for context only.

| Experiment | NOT RUN / PILOT / COMPLETE / PARTIAL / BLOCKED | Independent policy seeds / label draws | Key observed result | Raw evidence and numerical confidence | Reason for any unresolved claim |
|---|---|---|---|---|---|
| R1 same-controller oracle and information sets | COMPLETE | 8 and 5 label draws on shared checkpoints; sampled seeds 210–212 | Oracle inequality holds. Information changes the early 40-step return and does not change it at learned update 60 | Violation 0. Learned-60 MLE minus A_true crosses 0 | The MLE-versus-A_true gap at update 60 is unresolved |
| R2 replay and long-horizon trajectory diagnosis | COMPLETE | 8 policy seeds × 5 methods; forks use seeds 0–2 | Replay gap 0. The 40-versus-160 reversal is a short-treasure basin, visible in the outcome distribution and in the state forks | Segment gains and treasure probabilities above | A_true was one exact run because the population recursion is deterministic |
| R3 information-to-action and timing forks | COMPLETE | 3 parent policies × 8 label draws, plus a no-label control | Robust updates shrink the interval and can move phi without moving true return. Singleton MLE raises the next 20 steps at update 20 | 20-step robust gap inside 0.0002. MLE gap +0.0092 with no negative draw at update 20 | Later MLE draws cross zero. The source set remains [0,1]; P3 was uncertified at update 1 |
| R4A 2D identification and query-direction coverage | COMPLETE | 20 label repetitions on fixed designs | Min-eigen covers both Fisher directions. It does not win on weight error or held-out NLL | Eigenvalues and NLL in the R4A table | Grid coverage of the true weight is vacuous while hundreds of simplex points remain feasible |
| R4B valid robust 2D lower bound / full-simplex fallback | COMPLETE | one label draw per design, plus synthetic checks | The real-batch lower bound equals the full simplex. Audit violations are 0. No teacher fallback | Gap 0 on the real batches; synthetic gap 0.001065 | Not a formal computer-verified certificate |
| R4C RG real policy comparison | COMPLETE | 8 independent policy-and-label seeds per method | Preference methods overlap. The simplex robust arm does not learn. Privileged lookahead reaches 0.755 | Means and sds in the R4C table | Min-eigen minus uniform is unresolved. The outer-region arm was not separate from the simplex |
| R5 paper-facing synthesis and figures | COMPLETE |  | Three figures and the claims matrix | `outputs/horizon_bridge/r5_paper_results/` | |

**Execute in this order:**

1. **R1:** implement the common A_U bounded exact one-step controller. Verify its singleton true-preference oracle dominance at identical theta/phi. Only then repeat the same-state information-set comparison and short policy futures.
2. **R2:** replay and store the OLD full 160-step policy states for the same predeclared seeds, verifying original results before analyzing checkpoint/treasure distributions or conducting causal state/controller forks. Continue R2 independently if the rest of R1 is still running.
3. **R3:** use identical parent policies and query candidate pools to separate statistical information, controller choice and downstream true return. Keep pre-feedback source certification separate from post-feedback suffix.
4. **R4A:** create genuinely two-directional RG comparison data and test uniform/trace/min-eigenvalue rules with correct convex BTL fitting before rerunning long policies.
5. **R4B:** remove oracle-dependent numeric fallback and compute a valid conservative robust bound or openly label an unresolved numeric gap. Do not make the full R4C workload contingent on proving a difficult global optimizer bound; run the MLE and oracle controls in parallel.
6. **R4C:** evaluate corrected RG MLE, source/uniform/Fisher/min-eigen and conservative robust controllers under matched Gym interaction budgets, independently regenerated label and policy seeds, and exact/evaluated three-component returns.
7. **R5:** fill all tables, report real wall time and solver precision, create at most three interpretable figures, and mark each paper claim YES/PARTIAL/NO with its precise scope. Push source, small raw outputs and THIS updated Markdown to GitHub.

**No intermediate user approval is needed for implementation details.** Do not silently abandon new experiments because a preliminary contrast is null; isolate the variable and finish the interpretable part. Do not replace the old negative Work Plan 05 results with positive-looking new seeds. The intended scientific endpoint is a clear separation among (a) preference information, (b) model-based guidance optimization, (c) finite reward parameter movement, (d) stochastic/nonconvex policy trajectory effects, and (e) multidimensional reward identification.

**Success criterion:** A skeptical ICML reviewer should be able to read the filled R1–R4 tables and immediately see which claims are measurements, which are conservative guarantees, and which are still unsupported. A well-diagnosed negative result is a success; a flattering result with unmatched control access is not.
