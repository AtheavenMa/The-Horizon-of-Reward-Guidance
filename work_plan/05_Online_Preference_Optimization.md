# 05 Online Preference Optimization

**Project:** The Limits of Reward Guidance under Finite Adaptation  
**Target:** ICML 2027  
**Prepared:** 2026-10-19  
**Status:** COMPLETE — P1–P5 were run from this protocol. Result cells below are measured. The homepage README was not changed.  
**Starting commit:** b48180d3921edf21f46fbc5c9070f0c4f3b7418b

> **Executing research agent:** This is the complete experimental assignment once the following sections have been read. The paper is supplied separately by the owner. Use the existing repository, execute the experiments, fill THIS same Markdown document with measured numbers, and commit code, small outputs and the updated plan to GitHub. Ordinary numerical/engineering adjustments are delegated to you; do not stop to request staged approval. Multi-hour jobs are welcome, but don't run experiments whose failure cannot be explained. Preserve unfavorable results and never fabricate values.

## 1. Overall Objective and the Existing Evidence

### 1.1 Scientific motivation

The paper distinguishes (a) pointwise availability of a reward that induces sufficient progress, (b) a reward-parameter trajectory starting from the prescribed initialization and bounded adaptation speed, and (c) statistical confidence from preference labels that such progress is guaranteed. Theorem 1 and Theorem 2 already provide precise constructions; the preceding Work Plans 02–04 connected them to sampled gradient updates, sequential finite models and a small real RL environment.

The next scientific question is more practical: **when new pairwise preferences are obtained, does their information actually improve a learned reward, change finite-budget deployed guidance, and increase true policy return?**

Do not conflate confidence-set contraction, changed phi, and higher evaluator true return. They are different outcomes and must be reported separately. A failed source certificate does not mean true policy improvement became negative.

### 1.2 Actual existing sources

| Existing file | Reuse | Do not inherit |
|---|---|---|
| [Work Plan 04](04_Reachability_to_Small_MDP.md) | Completed H1–H4 tables and previous observations | Previous conclusions are not P1–P5 result cells |
| [14_dst_exact_tabular.py](../experiments/bridge/14_dst_exact_tabular.py) | MO-Gymnasium Deep-Sea-Treasure extraction, exact finite-horizon DP, policy gradient and sampled training | No claim that the time-dependent DP optimal policy is necessarily reachable by the time-homogeneous softmax learner |
| [15_dst_preference_forks.py](../experiments/bridge/15_dst_preference_forks.py) | Preference-pair regeneration, BTL likelihood, confidence interval, checkpoint forks, fresh-start 40-update learning | Its fixed-step BTL fitting is inaccurate, its uniform-query comparator uses a different pool, and its future labels were drawn from a previously fixed trajectory pool |
| [H4 summary](../outputs/horizon_bridge/h4_dst_preference/summary.json) | Old point-estimate discrepancy, return curves and original saved conditions | Three sampled policy seeds share one preference-label set; they are not three independent end-to-end feedback runs |
| [H3 summary](../outputs/horizon_bridge/h3_dst_oracle/summary.json) | Exact DP reference, development threshold and checkpoints at policy updates 5 and 20 | Original 20-step oracle PG did not yet reach the time-indexed DP optimum |
| [Frozen DST teacher](../experiments/horizon_real/E1/Deep-Sea-Treasure/frozen/frozen.json) | w_star=(0.375,0.625), beta=5, low=[0,-20], high=[15.100000381469727,-1], 20-step horizon | Ignore its old E1 coarse-grid formal confidence claims |
| [Frozen Resource-Gathering teacher](../experiments/horizon_real/E1/Resource-Gathering/frozen/frozen.json) | w_star=(0.2,0.2,0.6), beta=1, cap 40 | This three-objective task requires two-dimensional preference inference, not a one-dimensional grid |

### 1.3 Measured references (not new results)

Deep-Sea-Treasure is the installed convex map with 72 enumerated states, 4 discrete actions, finite learning/evaluation horizon 20. Initial uniform policy true return 0.49994372476520715; 20 exact true-reward policy-gradient updates reach 0.607583007273581; time-indexed DP optimum is 0.7790170719066627. Frozen development gamma_pos for the H3 setup is 0.0004196952089500494 **per policy update**; also report gamma=0.

Old 40-update fresh-start comparison: 32 original source labels only led to true return 0.60767129069595; 32 additional informative labels led to 0.6236138236044841; the old uniform-label arm led to 0.60767129069595; evaluator-only true-reward control led to 0.6243692767294623. Three sampled policy seeds preserved the source/informative ordering, but reused the same preference labels.

**Important fitting error:** with extra32 informative labels and old label seed 100, bounded likelihood optimizer estimates w_hat around 0.3363513883, whereas the old first-order optimizer even after 80 updates returned 0.7663431054. True teacher preference is w_star=0.375. The old fitted-point baseline is therefore not reliable.

**Important query-control weakness:** the original 32 source comparisons were deliberately selected for low preference sensitivity, median Fisher information 0. The old uniform new-label candidates came from a leftover low-information subset rather than the SAME pool available to the informative selector. Work Plan 05 must fix both confounders before using the old positive difference in a paper claim.

### 1.4 Experiment sequence

| Stage | Main change | Key scientific question |
|---|---|---|
| **P1** Correct BTL / fair query design | Replace nonconverged fitting; make uniform and informative queries use the same candidate pool | Was previous robust superiority partly an unfair optimizer or data comparison? |
| **P2** Matched-checkpoint deployment | Correct MLE versus confidence-set robust guidance with same labels/policy state | Does robust guidance beat a CONVERGED point estimate? |
| **P3** Actual live-environment online learning | Collect new policy trajectories and preference comparisons during training | Does targeted feedback improve true policy return over a fair uniform selector? |
| **P4** Controlled resource variation | Change just label count, timing or phi movement budget | What resource is actually binding? |
| **P5** One second real task | Resource-Gathering with 3 reward dimensions and 2D preference parameters | How much transfers beyond the 2-objective DST task? |

This plan deliberately permits longer independent runs, but does not restart Walker, Meta-World or LLM PPO while the real tabular causal comparison still has unresolved components.

## 2. Common Experimental Definition and Controls

### 2.1 Policy, evaluator and guidance

- Use actual MO-Gymnasium deep-sea-treasure-v0 with 20-step truncation, exactly as H3. The learned policy is the same time-homogeneous state-by-action softmax table. Its evaluator is exact finite-horizon DP from H3, and the training phase uses actual observed environment trajectories when P3 is run.
- Retain normalized two-dimensional trajectory return features f(tau)=(raw_return-low)/(high-low), with the frozen low/high vectors.
- Teacher preference w_star=0.375 is accessible only to the Bradley–Terry labeling process and independent true-return evaluator.
- Deployed guidance phi is a separate scalar in [0,1], determining training reward R_phi=phi*f1+(1-phi)*f2. It cannot teleport to the true w_star or the fitted w_hat.
- Primary phi movement limit b=0.02 **per policy update**, with only one planned sensitivity b=0.05 in P2 and the small P4 b sweep. Do not equate b with the continuous-time rho=0.04 from the four-output theorem.
- From an actual policy update pi to pi_next, exact true gain is J_star(pi_next)-J_star(pi), not current absolute J_star. Source certificate means minimum scalarized gain over the original preference compatibility U_source; updated certificate means minimum over the new U_N. The min is attained at the two interval endpoints after the policy update is fixed.
- Report two progress thresholds: gamma=0 and H3's frozen development gamma_pos=0.0004196952089500494. If a checkpoint's best available first step is already below gamma, write "uncertified from initialization", not "lost certification later". For a 160-update learning curve, late learning-rate saturation may naturally eliminate positive per-step progress; diagnose this with exact oracle gradient/pointwise values.

### 2.2 Genuine Bradley–Terry reward fitting

For two trajectory feature vectors f_i and f'_i, define D=f_i-f'_i and

\[
c_i=\beta D_{i,2},\qquad d_i=\beta(D_{i,1}-D_{i,2}),\qquad
P(Y_i=1|w)=\sigma(c_i+d_i w),\quad w\in[0,1],\ \beta=5.
\]

The scalar convex negative log likelihood is

\[
\ell(w)=\sum_i[\log(1+\exp(c_i+d_iw))-Y_i(c_i+d_iw)].
\]

A true converged MLE uses bounded scalar minimization plus an independently implemented monotone score-root/damped-Newton check; interior stationary residual must be small relative to data scale, and boundary solutions must obey KKT signs. If all d_i=0, preference is unidentifiable; label it as such instead of treating a default coefficient as learned.

Use held-out BTL log loss, Brier score, predicted-probability calibration, true parameter error (evaluator only), and valid uncertainty interval width. Thresholded accuracy alone is secondary: all H4 heads previously achieved 0.78125 on one small easy held-out set.

For a **fixed-N preference batch selected before its labels**, use a continuous confidence interval by score inversion, not an E1 coarse simplex grid:

\[
S_B(w)=\sum_{i\in B}d_i[Y_i-\sigma(c_i+d_iw)],
\quad
R_B(\alpha)=\sqrt{\tfrac12\log(2/\alpha)\sum_{i\in B}d_i^2},
\]

\[
U_B(\alpha)=\{w\in[0,1]:|S_B(w)|\le R_B(\alpha)\}.
\]

S_B is nonincreasing in w, allowing interval endpoints by root finding. All d=0 implies U=[0,1]. Fixed-batch alpha=.05 is acceptable in P1/P2.

**For the adaptive online P3 run**, future trajectories/pairs depend on earlier labels. To obtain a conservative whole-run simultaneous compatibility set, select each event's batch BEFORE reading its labels, compute its conditional fixed-N interval with alpha=.01, also compute the initial-batch interval at alpha=.01, and intersect the five intervals across the initial batch and four updates (union bound total alpha at most .05). Fit the pooled MLE on all observed labels for point deployment, but do not claim an unadjusted pooled fixed-design interval is time-uniform. If the intersection is empty, record an unavailable certificate and preserve the previous deployed guidance if necessary; do not silently call the empty interval a valid guarantee.

A conservative confidence set remaining broad despite useful MLE labels is a **meaningful statistical limitation**, not a training-code failure. Show certificate improvements separately from actual policy return.

### 2.3 Fair query selection and evaluation

For any candidate pair, rank informativeness using only PREVIOUSLY FIT w_hat and known observable D:

\[
I_i=d_i^2\sigma(c_i+d_i\widehat w)[1-\sigma(c_i+d_i\widehat w)].
\]

The targeted selector takes the most informative distinct candidate pairs; the uniform control samples from the FULL SAME unlabeled pair pool. Do not exclude high-information candidates from the uniform candidate set. Pair orientation and RNG are fixed without seeing labels. No selector reads the hidden teacher weight or teacher probability.

At a shared checkpoint, create one candidate pool and use it for ALL query strategies. During diverging end-to-end online policies the actual pools will differ because the policies differ. P2's frozen-checkpoint forks isolate the query/deployment decision under literally identical data.

Independently freeze a held-out evaluation pool of at least 512 trajectory comparisons from a diverse set of reference/initial/exploratory policies; its labels and pair IDs never enter training or query selection. Report both held-out all-pair log loss and a prespecified informative subset.

The P3 live trajectory generator must ACTUALLY call Gym reset and step while the policy changes. The old H4C collected all future comparison pairs before training. A new P3 result must collect the preference-query candidates at the scheduled NEW policy checkpoints.

### 2.4 Practical execution

- Preserve results/reference, existing G1–G4.1, H1–H4, and historical E1–E8. New code uses its own numbered files and output folders.
- Record policy logits, phi, optimizer state, pair IDs, raw comparison bits or reproducible seeds, current confidence set and exact-DP evaluator return. Large full trajectories stay under /workspace, not Git.
- Independent full seeds vary **both** teacher/label generation AND policy rollouts. Re-running policy seed alone with the same label set is not independent preference learning.
- Run a development throughput/gradient-quality pilot, freeze configs, then complete the specified independent runs. Do not select successful runs or thresholds according to confirmation results.
- Every experiment has an exact or independently measured reference. If a result is null, identify whether BTL information, deployment rate, policy gradient or environment coverage is the missing link, instead of escalating unrelated model complexity.
- CPU first for exact DP/BTL; the 5090 may be used if batched simulation actually speeds up the real workload. Multi-hour execution is acceptable. No GPU-heavy LLM weights are needed.

## 3. P1 — Repair the BTL Reward Learner and Candidate Pool

**Objective:** Replace the unreliable old point estimate by a converged, independently checked BTL optimizer, then compare query strategies from an identical unlabeled pool.

**Source:** Regenerate existing H4 data using 15_dst_preference_forks.py functions and original seeds, plus build an independent new diverse pool. Do not edit old H4 outputs.

**Actions:**

1. Recreate old seed-100 source, extra16, extra32, extra64 and uniform data exactly. Confirm the original H4 MLE (extra32 around 0.3363513883) and old fixed-gradient estimate (around 0.7663431054); any mismatch here indicates changed data/normalization, not a scientific model effect.
2. Implement a new bounded MLE solver and a genuinely convergent first-order/Newton alternative. Compare NLL, numerical score/KKT, estimated w, and true teacher error on each SAME batch. Use undertrained 8-/80-step old iterations only as historical diagnostics.
3. Build one common unlabeled candidate pool from observed real trajectories under current policy and fixed non-oracle exploration/reference policies. A fixed development estimate of w (NOT true w_star) ranks Fisher pairs. The uniform strategy samples over the entire pool, not left-over items.
4. Compare two initial data regimes, both with 32 source labels:
   - **Weak initial source:** old low-abs(d) pool, median Fisher close to zero. This isolates insufficient source information.
   - **Mixed initial source:** 32 random comparison pairs from an actually diverse full candidate pool. This tests whether the gain persists when initial data already contain tradeoff information.
5. Compare source alone, source+32 fair uniform, and source+32 Fisher-informed; all use independent teacher labels with seeds 100–109 (10 seeds) and the same query budgets. Use 512+ disjoint held-out comparisons and report NLL/Brier, coverage, width and information statistics.
6. At fixed labels, compare an underoptimized fit, a converged MLE, and further optimization AFTER convergence. Once optimal, extra compute should not create information or change U; if it changes the deployed policy, report the numerical cause.
7. Return a brief causal finding, not just an RMSE table: was H4's old point baseline inaccurate, was its uniform query control unfair, and how important is source informativeness?

**P1 result table** (per-seed CSV required):

| Initial source | New comparison rule | Total labels | Independent label seeds | Old fit / converged MLE | MLE loss and KKT agreement | Held-out BTL NLL / Brier | Confidence interval width and coverage | Information level | Notes |
|---|---|---:|---:|---|---|---|---|---|---|
| Weak | source only | 32 | 10 | 0.456 / 0.429 | Train NLL 21.326 vs 21.324; Newton gap 2.3e-8. Scaled KKT 0.041 because the likelihood is flat | 0.654 / 0.220 | width 1.00; coverage 10/10 only because the interval is [0,1] | median Fisher 0 | Unidentified. The point estimate is not a reward |
| Weak | fair uniform full pool | 64 | 10 | 0.281 / 0.281 | Train NLL identical; KKT ~1e-9; post-convergence move 5.6e-9 | 0.564 / 0.193 | width 0.929; 10/10 | new-pair median Fisher 0.253 | Mean absolute error 0.150 |
| Weak | Fisher from same pool | 64 | 10 | 0.670 / 0.342 | Train NLL 61.38 vs 35.66; old KKT 0.331, MLE KKT ~1e-9 | 0.563 / 0.193 | width 0.591; 10/10 | new-pair median Fisher 7.07 | Mean absolute error 0.114. Held-out NLL ties the fair uniform arm |
| Mixed | source only | 32 | 10 | 0.356 / 0.342 | Train NLL 12.91 vs 12.86 | 0.557 / 0.192 | width 0.802; 10/10 | source median Fisher 0.791 | Mean absolute error 0.086. Already partly identified |
| Mixed | fair uniform full pool | 64 | 10 | 0.335 / 0.335 | Solvers agree; KKT ~1e-9 | 0.553 / 0.190 | width 0.660; 10/10 | new-pair median Fisher 0.787 | Mean absolute error 0.048. Best held-out NLL in the table |
| Mixed | Fisher from same pool | 64 | 10 | 0.999 / 0.396 | Train NLL 108.9 vs 33.48; old fit sits on the boundary | 0.557 / 0.192 | width 0.332; 10/10 | new-pair median Fisher 10.37 | Mean absolute error 0.029. Narrower interval, no held-out NLL gain over uniform |
| Original H4 seed-100 diagnostic | old extra32 | 64 | 1 | old 0.766343 / MLE 0.336352 (verified) | Train NLL 64.847 vs 41.578; MLE KKT 6.0e-9, old KKT 0.513 | interval [0.0258, 0.5147] contains 0.375 | width 0.489 | source median Fisher 0 | Damped Newton agrees with bounded minimization to 5e-9. One further Newton step moves w by 4.9e-9 |

**Possible conclusions:** correct MLE closes the previous apparent robust advantage; robust remains better despite correct MLE; fair uniform labels prove similarly informative; or mixed source starts already well identified. Each outcome identifies a different cause, and no new benchmark is needed to understand it.

## 4. P2 — Fair Matched-Checkpoint Guidance Comparison

**Objective:** Test whether robust confidence-set deployment has benefits over the **properly fitted MLE point estimate**, holding preference data, policy and adaptation budget fixed.

**Reference states:** H3 early policy checkpoint at oracle-development update 5 and later at update 20. Mark these as oracle-developed policy-state controls, NOT end-to-end learned starting policies. After P3 implementation, repeat the essential fork using two SOURCE-ONLY learned-policy checkpoints at P3 updates 20 and 60.

**Implementation:**

1. Restore identical policy logits, training optimizer state and exact H3 evaluation for each condition. Infer U_source from the SAME 32-label initial batch, and set shared phi_init by source-only robust one-step optimization over phi in [0,1], not hidden w_star.
2. For each label seed and fixed checkpoint, build exactly one complete unlabeled candidate pool. Select either 32 fair uniform or 32 Fisher informative pairs. Save and reuse the SAME 32 informative labels for the converged MLE and robust deployment branches.
3. Compare source-only converged MLE; source-only further converged fitting (extra compute with no labels); uniform+32 converged MLE; informative+32 converged MLE; informative+32 robust compatibility-set guidance; old underoptimized fit clearly labeled DIAGNOSTIC; evaluator-only oracle.
4. MLE controller moves phi toward w_hat by at most 0.02 per policy update. Robust controller searches the permitted phi interval for the best EXACT one-step worst-compatible true-return-vector gain over its learned U_N; it cannot read true w_star. Use deterministic continuous 1D search with endpoints and refined candidate maxima. The old 21-point candidate grid alone is not enough to claim numerical optimality.
5. First evaluate one exact population policy update from each shared checkpoint. Then run 40 exact policy updates per branch, with exact DP after every step. Report phi paths, source/updated certificates, actual true return changes, unconstrained pointwise value and bounded-window best value.
6. Complete 8 independent label seeds on the early checkpoint and 5 on the later. Three paired sampled-gradient seeds for the most important difference (informative MLE versus informative robust) are sufficient after exact comparison.
7. Run one sensitivity b=.05 for only the two primary information-aware methods, to distinguish a binding phi adaptation budget.
8. When P3 has supplied true learned-policy checkpoints at updates 20 and 60, run the same corrected MLE/robust comparison from these saved states. This separates deployment effects from a special oracle-developed starting policy.
9. If newly informative MLE and robust differ little after P1 repair, preserve that as the scientific conclusion: the older robust superiority may have been caused by the poorly fitted competitor.

**P2 table** (each row backed by seed-level data):

| Checkpoint | Same-pool query and reward deployment | New labels | Independent label seeds | Phi budget | Exact true return increase over 40 updates | Difference from converged informative MLE | Original/updated certificate at end | Pointwise vs bounded guidance | Sampled verification | Notes |
|---|---|---:|---:|---|---|---|---|---|---|---|
| Oracle-developed early | source MLE, fully converged | 0 | 8 | .02 | 0.07527 (sd 0.00292) | +0.00672 | -0.000592 / -0.000592 | init robust gain 0.001206, certified for gamma_pos |  | Final phi 0.054. The flat source MLE is driven downward |
| Same | source MLE, more fitting on same labels | 0 | 8 | .02 | 0.07527 (sd 0.00292) | +0.00672 | same to 1e-12 | same init |  | Extra Newton steps do not change the policy |
| Same | fair uniform new labels, MLE | 32 | 8 | .02 | 0.06466 (sd 0.00959) | -0.00389 | -0.000624 / -0.000624 |  |  | Final phi 0.439 |
| Same | Fisher new labels, converged MLE | 32 | 8 | .02 | 0.06855 (sd 0.00682) | 0 | -0.000498 / +0.000075 |  | reference | Final phi 0.389 |
| Same | SAME Fisher labels, robust guidance | 32 | 8 | .02 | 0.06126 (sd 0.00955) | -0.00729 | +0.000714 / +0.000902 | robust keeps a positive certificate and a lower true return | 3 sampled seeds, robust−MLE = +0.00461, +0.00168, +0.00287 | Exact comparison does not favor robust. The sampled sign differs and the gaps are small |
| Same | old unoptimized fitting, diagnostic | 32 | 8 | .02 | 0.06824 (sd 0.00667) | -0.00031 | -0.000496 / +0.000078 |  |  | Final phi 0.399. Forty budget-limited steps hide much of the bad target |
| Same | oracle-only true preference | n/a | 1 | oracle | 0.07339 | +0.00484 | not a source certificate |  |  | phi fixed at 0.375 |
| Same, sensitivity | Fisher MLE / robust at b=.05 | 32 | 8 | .05 | 0.07039 / 0.06463 | +0.00184 / -0.00392 | MLE updated +0.000089; robust updated +0.000769 | same final MLE phi 0.389 as b=.02 |  | Larger budget does not unlock a new MLE target. Robust phi moves from 0.379 to 0.316 |
| Oracle-developed later | converged MLE vs robust | 32 | 5 | .02 | 0.01874 vs 0.01578 | robust −0.00297 | MLE -0.000233 / -0.000012; robust +0.000031 / +0.000199 | init gain 0.000230, below gamma_pos: uncertified from initialization |  | Not a certificate that expired later |
| Learned-policy checkpoint 20 | converged MLE vs robust | 32 | 5 | .02 | 0.07734 vs 0.07876 | robust +0.00142 | MLE -0.000141; robust +0.000950 / +0.001158 | init gain 0.001974, certified |  | Difference is inside the seed sd of about 0.024 |
| Learned-policy checkpoint 60 | converged MLE vs robust | 32 | 5 | .02 | 0.06045 vs 0.03878 | robust −0.02166 | MLE -0.000141; robust +0.000538 / +0.000538 | init gain 0.001089, certified |  | Robust again protects the certificate and lowers true return |

**Interpretation:** an improvement in updated certificate alone is information value, not true return gain. A phi target the optimizer cannot reach is a deployment budget bottleneck. If the fully converged MLE matches robust, state this directly. If only oracle-developed checkpoints show a difference, state its dependence on state distribution.

## 5. P3 — Actual Live-Environment Online Preference RL (Main Long Run)

This is the central multi-hour experiment: genuine policy-environment interactions, preference comparisons from fresh current-policy trajectories, a trained BTL head, and bounded reward guidance.

### 5.1 Predeclared primary protocol

| Factor | Value |
|---|---|
| Environment | Actual Gym/MO-Gymnasium deep-sea-treasure-v0 |
| Policy class | H3 time-homogeneous tabular softmax, fresh uniform logits |
| Initial guidance | phi_init=0.5, no oracle policy warm start |
| Evaluation and training episode horizon | 20 time steps |
| Number of policy updates | **160**, rather than previous 40 |
| Primary training batch | **256 complete LIVE environment episodes per update** (pilot 256 vs 512 only for gradient-estimator quality) |
| Policy update | Sampled REINFORCE, same declared scalar reward, baseline and step size across all learner methods |
| Phi adaptation budget | b=0.02 per policy update, all non-oracle arms |
| Initial source labels | 32, regenerated independently per full teacher/label seed |
| New labels | 32 per acquisition, **four events**, 128 new / 160 total |
| Query times | Immediately before policy updates 20, 40, 60, 80 |
| Fresh candidate pool per event | **128 new REAL trajectories**: 64 current-policy, 32 uniform explorer, 32 predefined treasure-seeking/reference explorer |
| Pair candidates | About 256 unlabeled pairs from those new trajectories; fixed orientation/seed, selected BEFORE BTL labels are drawn |
| Query strategies | Fisher informative (previous w_hat only) or fair uniform over the full candidate pool |
| Evaluation each update | Exact DP vector return and evaluator-only true scalar return; regular live rollout consistency checks |
| Main weak-source confirmation | **8 independent full teacher-label AND policy-rollout seeds** |
| Mixed-source generality check | **4 independent full seeds**, same method/step/feedback budgets |
| Checkpoints | Updates 0, 20, 40, 60, 80, 120, 160 with logits, phi, BTL state and RNG |

At 8 independent weak-source runs, 5 methods, 160 updates and 256 actual episodes/update, the primary workload is **1,638,400 complete policy-training episodes**, plus feedback-pool episodes and the later mixed-source runs. This is PLANNED work, not a measured runtime. Running for multiple hours is authorized. Record actual throughput first, rather than inventing an ETA.

### 5.2 Method definitions

| Method | Preference data / update rule | Deployment |
|---|---|---|
| S | Source-only: 32 initial labels, no further feedback | Bounded robust over U_source |
| U | 32 initial + 128 FAIR UNIFORM new labels from fresh candidate pools | Bounded robust over U_updated |
| I | 32 initial + 128 Fisher-selected new labels | Bounded robust over U_updated |
| M | SAME targeted query rule and total label budget as I, but train a **converged pooled BTL MLE** | Move phi toward w_hat under b |
| O | Evaluator-only true reward preference, not a deployable learner | Oracle policy-gradient control, explicitly labeled |

At the same checkpoint, I and M may share one candidate pool and exact comparison bits. As their full online policy trajectories diverge, future live candidate pools will differ; P2, not P3, establishes the isolated same-data deployment comparison.

The uniform comparator **must not** use leftover candidates after removal of the Fisher-rich pairs. Uniform sampling from the whole pool is essential.

### 5.3 Exact real interaction loop

For every actual policy update:

1. Use the current deployed phi to scalarize the observed reward vector. Call Gym reset and step to collect 256 new policy trajectories, compute a genuine score-function policy gradient and update the real policy logits. Use the same training episodes, number of policy updates, learning rate and gradient step clipping across learner methods.
2. Calculate the updated policy's exact finite-horizon expected two-dimensional return by the independent H3 DP evaluator. The hidden teacher w_star can appear ONLY in this evaluator and BTL labeling process, never in the learner.
3. At the four scheduled feedback updates, **before the next policy update**, collect 128 NEW real Gym episodes (64 current policy / 32 uniform / 32 fixed explorer); form about 256 unlabeled pair candidates, choose 32 using the prescribed uniform or Fisher method, THEN draw independent fixed-teacher BTL binary labels.
4. Refit the correct pooled BTL likelihood. Record w_hat, per-batch and overall confidence set, held-out BTL loss, total and newly acquired labels. Apply the declared bounded phi rule to the NEXT policy update.
5. Save per-step exact true return, true gain, source and updated certificates, full pointwise value if numerically available, bounded-window best value, phi, gradient norm, acquired labels, distinct trajectory IDs, and current policy checkpoint.
6. Use disjoint policy, query-candidate and teacher-label random-number streams per independent run. Common random numbers may be used within a matched comparison, but do not treat dependent branches as independent replications.
7. Source-only S must not silently acquire new labels. Uniform U and informative I/M receive exactly the same LABEL COUNT and scheduled opportunities.
8. At feedback arrival report the entire-path and post-feedback certificates separately. Previously lost ex-post certification cannot be retroactively restored by later labels.
9. Confirm at a few saved policies that independent live-environment rollout returns agree with the DP evaluator within measured sampling uncertainty. If not, repair the model/time-limit mismatch before analyzing the value curve.

**Adaptive interval scope:** preference queries depend on earlier observations, so for a conservative whole-run compatibility set, freeze each round's candidate choices before collecting its labels. Use conditional alpha=.01 for the initial fixed-N source batch and each of four feedback batches; intersect the five per-batch intervals. A union bound gives conservative simultaneous coverage under the stated BTL noise/conditional independence assumptions. Pool all acquired labels for MLE fitting, but do not call an unadjusted pooled fixed-N interval a valid time-uniform certificate. If the conservative set remains broad, that is a measured statistical limitation; log which controller decisions change or do not change.

### 5.4 Execution phases

**Development:** 3 new full seed identities, short 20–40 update pilots on S, I and M. Benchmark actual environment episode speed, reward head fitting time and DP time. Compare gradient-quality at 256 and 512 sampled episodes against exact DP gradient at the same checkpoint; freeze 256 unless estimator variance specifically justifies 512. Do not pick based on whether I beats S on development.

**Confirmation:** full 160-update trials with 8 independent weak-source runs and 4 independent mixed-source runs, all five methods. Preserve all seed signs. Use resumable atomic checkpoints; run jobs in parallel only if 12-core memory/throughput measurements support it.

**What if learning saturates?** Report exact true-return curves, pointwise progress and oracle gradient. Late source certificate loss may arise because even the best policy improvement vanishes as the policy converges; that is different from the moving-support mechanism.

**P3 main result table (one row per method and source regime):**

| Initial source | Method | Independent full seeds | Extra labels | Actual training episodes | Exact true return at 0 | Exact return at 40 / 80 / 160 | Final paired gain vs S | Return area under learning curve | Held-out BTL NLL / interval width | Final phi | Notes |
|---|---|---:|---:|---:|---|---|---|---|---|---|---|
| Weak | S source-only robust | 8 | 0 | 327,680 | 0.499944 | 0.60694 / 0.65654 / 0.68307 (sd 0.00471) | reference | 101.85 | 0.613 / 1.00 | 0.222 | Source Fisher 0. Certificate negative from update 1 |
| Weak | U fair uniform robust | 8 | 128 | 327,680 | 0.499944 | 0.60787 / 0.65681 / 0.68265 (sd 0.00385) | -0.00042 (sd 0.00711) | 101.92 | 0.555 / 0.735 | 0.221 | Same phi path as S. Labels improve NLL, not return |
| Weak | I targeted robust | 8 | 128 | 327,680 | 0.499944 | 0.60855 / 0.65307 / 0.67771 (sd 0.01103) | -0.00536 (sd 0.01425) | 101.45 | 0.555 / 0.407 | 0.231 | Fisher median about 5–7 on seed 0. Width shrinks; return does not rise |
| Weak | M targeted converged MLE | 8 | 128 | 327,680 | 0.499944 | 0.62683 / 0.64414 / 0.65142 (sd 0.03743) | -0.03165 (sd 0.03881) | 100.98 | 0.554 / 0.412 | 0.361 | Phi leaves S at update 1 and approaches 0.375 |
| Weak | O true-reward oracle | 8 | oracle-only | 327,680 | 0.499944 | 0.62425 / 0.63334 / 0.63790 (sd 0.00017) | -0.04517 (sd 0.00465) | 99.88 | n/a | 0.375 | Saturates below the one-step treasure 0.642 and the DP optimum 0.779 |
| Mixed | S source-only robust | 4 | 0 | 163,840 | 0.499944 | 0.60780 / 0.65612 / 0.68181 (sd 0.00418) | reference | 101.80 | 0.561 / 0.900 | 0.222 | Source already has some sensitivity |
| Mixed | U fair uniform robust | 4 | 128 | 163,840 | 0.499944 | 0.60765 / 0.65766 / 0.68427 (sd 0.00396) | +0.00246 (sd 0.00599) | 102.02 | 0.556 / 0.749 | 0.219 |  |
| Mixed | I targeted robust | 4 | 128 | 163,840 | 0.499944 | 0.60981 / 0.65102 / 0.68761 (sd 0.03319) | +0.00579 (sd 0.03581) | 101.53 | 0.555 / 0.372 | 0.336 | One wide seed dominates the mean |
| Mixed | M targeted converged MLE | 4 | 128 | 163,840 | 0.499944 | 0.62530 / 0.63387 / 0.63809 (sd 0.00030) | -0.04372 (sd 0.00443) | 99.93 | 0.554 / 0.399 | 0.367 | Lands on the oracle plateau |
| Mixed | O true-reward oracle | 4 | oracle-only | 163,840 | 0.499944 | 0.62448 / 0.63343 / 0.63795 (sd 0.00016) | -0.04386 (sd 0.00414) | 99.89 | n/a | 0.375 | Same plateau as the weak oracle |

**P3 causal diagnosis table:**

| Question | Exact observation needed | Result | Supported interpretation | Notes |
|---|---|---|---|---|
| Does the environment permit policy learning? | Oracle exact-return curve, gradient, optimality gap | YES, with a ceiling. Return rises from 0.49994 to 0.63790 by update 160. Gradient cosine at 256 episodes is 0.930 | The sampled learner improves and then flattens below the one-step treasure 0.642 and the DP optimum 0.779 | 512 episodes raise cosine only to 0.964, so the batch stays 256. A uniform live check and the update-20 learned policy both lie inside 4 Monte Carlo standard errors of the DP mean |
| Are new labels informative? | Pair Fisher statistics, independent BTL NLL, interval widths | PARTIAL. Weak source Fisher is 0 and width stays 1. Targeted and uniform new labels both cut held-out NLL from 0.613 to 0.555. Targeted width is 0.407 versus uniform 0.735 | New comparisons contain preference information. Fisher shrinks the interval more than fair uniform and does not improve held-out NLL further | Seed-0 targeted Fisher medians are 5.61, 6.78, 4.98, 5.01. Uniform medians are 0.16, 0.07, 1.97, 0.09 |
| Does new preference knowledge change phi? | First phi-path divergence and size/direction | The robust arms do not move away from source-only. The MLE arm does | U and I finish at phi 0.221 and 0.231, beside S at 0.222. M is already 0.03 away at update 1 and finishes at 0.361 | The first divergence is the unidentifiable source MLE, before any new label |
| Does fair query selection matter? | I vs U at matched label/episode budget, P2 same-pool contrast | NO for return. Weak I−U is inside noise; 8-seed U−I mean +0.00495, 95% t interval [-0.00255, +0.01244] | On this controller, selecting the high-Fisher pairs does not change the deployed policy enough to change return | P1 shows the same split: Fisher wins on width and parameter error, not on held-out NLL |
| Does robust beat converged MLE? | I vs M and P2 matched labels/checkpoint | Not as a reward-estimation result. Live I finishes 0.026 above M, but P2 exact robust finishes 0.007 below the same-label MLE | Live M tracks the true weight, and that weight is a slower optimization problem for this learner. The matched exact fork does not give robust a return advantage | Learned-policy update 60 repeats the exact result: robust return is 0.022 lower and its certificate stays positive |
| Did certification expire before new feedback? | Pre-feedback and post-feedback gain curves | NO. The source certificate is negative at update 1 on 8/8 seeds, before the first feedback at update 20 | This is uncertified from initialization. The source interval is the whole [0,1] | True return is rising throughout. A negative certificate is not a negative true gain |
| Is adaptation budget binding? | Bounded-window vs full pointwise gain, phi target distance | YES. See P4: b=0 stays at phi 0.5 and loses 0.039 return; b=0.01 loses 0.009. Both paired intervals exclude 0 | The robust controller wants to move, and the per-update cap determines how far it gets | b=0.05 has a higher mean and a confidence interval that still includes 0 |
| Does mixed initial source weaken the effect? | Mixed-source P3 results, information already available | The return gaps stay null. Mixed I−S is +0.00579 with interval [-0.051, +0.063] | A somewhat informative source does not create a targeted-return advantage, and the MLE still falls onto the oracle plateau | Mixed source width starts at 0.900 rather than 1 |

**Interpretation after a null result:** Identify in order BTL fit quality, query candidate sensitivity, confidence-set movement, phi decisions, exact true-gradient response and baseline learnability. A lower target return is not allowed to remain an uninterpreted "failed benchmark" outcome.


## 6. P4 — Information, Acquisition Time and Reward Adaptation Budgets

**Scientific objective:** After the P3 learner and reward model are working, identify WHY a method does or does not benefit from feedback. Do not run a huge factorial grid: vary one resource at a time, holding DST, policy initialization, evaluation and the other costs fixed.

### 6.1 New preference-label budget alone

On the weak-source condition, compare **0 / 32 / 64 / 128** newly acquired targeted labels, each with the same 160 policy updates and phi movement budget b=.02. Acquire 32 labels at each successive scheduled event (updates 20,40,60,80) until that method's permitted count is exhausted. Include the uniform-128 budget from P3 as the equal-budget selector comparison.

Start with 5 independently generated end-to-end teacher/policy seeds. Each budget comparison uses independent live trajectories and exact evaluator returns. Report cumulative label counts, conditional query informativeness, fitted w error, confidence set width and true policy return at 40,80,160.

A larger label budget is deliberately NOT label-matched. A correct finding might be that the first 32 labels already cross a controller decision boundary and later labels make little difference.

### 6.2 Guidance movement budget alone

Keep targeted query rule, 32 initial +128 new labels, scheduled acquisition events and 160 updates fixed. Compare **b=0, 0.01, 0.02, 0.05** per policy update, with five independent full seeds. Preserve the original phi0=.5 for all methods.

At b=0 new labels may improve the certificate without changing the actual policy. At larger b, record the desired phi target, distance from deployed phi, actual phi movement, updated certificate and true return. If b=.05 outperforms .02, it suggests deployment capacity was limiting. If all b values choose essentially the same phi, the reward movement budget is not the binding factor in this environment.

### 6.3 Feedback arrival time alone

Keep exactly 128 new targeted labels, 160 updates and b=.02 while comparing:

- **Early:** four batches of 32 at 20/40/60/80 (P3 reference).
- **Late:** four batches of 32 at 60/80/100/120.
- **One shot:** one batch of 128 at update 20 from a sufficiently large, newly collected live trajectory candidate pool.

Five independent seed pairs, measured full curves. One-shot acquisition differs in candidate policy distribution from multiple rounds; log the difference, and do not attribute everything to clock time.

### 6.4 Statistics and conclusion

For paired runs report actual final return difference and area-under-return-curve difference, mean, SD and a simple confidence interval (paired bootstrap or t interval, with method specified). Five to eight independent runs support a **limited-precision effect estimate**, not a universal success claim. No favorable-seed selection or replacing a negative seed.

**P4 result table:**

| One changed resource | Actual compared settings | Independent full runs per setting | Label and policy-update budgets | Final true-return difference and uncertainty | Source/updated certificate effect | Binding mechanism inferred | Notes |
|---|---|---:|---|---|---|---|---|
| New BTL labels | 0 / 32 / 64 / 128 | 5 | 160 updates, different label budgets | Means 0.68098 / 0.68122 / 0.68122 / 0.68122. Paired difference of 0 versus 128 is -0.00024, 95% t interval [-0.00621, +0.00572]. Differences among 32, 64 and 128 are below 1e-7 | Final source certificate stays near 0 for 128 labels and is -0.00121 at b=0 | Not binding. Once the robust set is wide, more labels do not cross a decision boundary | 0 reuses P3 S seeds 0–4. 128 reuses P3 I seeds 0–4 |
| Reward phi movement | b=0 / .01 / .02 / .05 | 5 | 160 updates; same 160 preference labels | Means 0.64242 / 0.67189 / 0.68122 / 0.71038. Versus b=.02: -0.03880 [-0.04486, -0.03273]; -0.00933 [-0.01299, -0.00568]; +0.02916 [-0.00815, +0.06647] | b=0 final source certificate -0.00121, updated -0.000565. Final phi is exactly 0.5 | Binding below .02. At b=0 the certificate can improve in principle while the policy cannot move; here the frozen phi 0.5 also has a worse certificate | b=.05 mean is higher but the five-seed interval includes 0. Final phi 0.278 versus 0.223 at b=.02 |
| New-feedback timing | early / late / one-shot | 5 | all 160 total labels and 160 updates | Late−early +0.00145 [-0.00255, +0.00545]. One-shot−early -0.00772 [-0.05415, +0.03870] | No detectable certificate separation | Not binding at this precision | One-shot uses 256 trajectories and 512 candidate pairs at update 20, so the pool is not the same object as four smaller rounds |
| Query rule | targeted vs entire-pool uniform | P3 8 weak-source runs | 160 labels each | Uniform−targeted +0.00495 [-0.00255, +0.01244] | Targeted width 0.407 versus uniform 0.735 | Information changes, deployment does not | Held-out NLL is 0.555 for both |
| Source informativeness | weak vs mixed | P3 8 vs 4 | same feedback/updates | Mixed targeted−source +0.00579 [-0.05119, +0.06278]. Weak targeted−source -0.00536 [-0.01727, +0.00655] | Mixed source width 0.900 versus weak 1.00 | The initial source is less empty and still does not move return | These are separate seed groups, not paired |

**Possible exact conclusions:**

- Preference coverage is already sufficient: mixed-source confidence interval initially narrow; further labels need not change policy.
- Labels add useful information but b=0 prevents action: certificate improves, policy return does not.
- New labels change phi, but true reward gain is nearly identical: the actual response is small or learning saturated.
- Timing matters more than total count: one-shot versus staged feedback changes when the policy becomes responsive.
- A larger budget only helps because an unconstrained reward deployment is otherwise unreachable.

## 7. P5 — A Second Real Environment with Three Reward Channels

**Scientific objective:** Extend the successful or well-diagnosed two-reward DST pipeline to the already used **MO-Gymnasium resource-gathering-v0** environment with a genuine **two-dimensional** unknown preference parameter. This is a scope/generalization experiment, not permission to call the old incorrect 3D confidence grid formal.

### 7.1 Frozen task setup

Use original repository parameters:

- Resource-Gathering environment id: resource-gathering-v0; episode cap 40.
- Return normalization low = [-1,0,0], high = [0,1,1].
- Evaluator-only/teacher preference weights = [0.2, 0.2, 0.6].
- Bradley–Terry beta=1.
- 4 discrete actions as in old E1; inspect the actual observation/state and transition semantics rather than assuming DST's fixed grid model applies.

### 7.2 First make this task learnable without preference estimation

1. Establish the installed environment's observable state, transition randomness, termination, reward-vector channels and task difficulty. If the finite-horizon Markov transition model can be enumerated cheaply, implement a genuine exact 3D-return DP evaluator and exact true-reward policy gradient. Otherwise make a **separate held-out live-environment** policy evaluator with enough trajectories and reported precision. DO NOT claim DP accuracy if no exact model exists.
2. Train from a fresh policy using the evaluator-only known preference as an oracle positive control. Start with a 40-update development pilot using 256–512 trajectories/update and extend to **100 updates** under a frozen optimizer configuration, with five independent seeds on the selected setup. Report all three reward components, scalar true return, and an attainable reference performance/gap.
3. If the oracle does not improve, diagnose visited-state coverage, rare-reward frequencies, gradient estimator error, behavior-policy exploration and normalization. Report what the task actually permits. Do not claim BTL feedback caused a failure when the positive control cannot learn.

### 7.3 Genuine convex two-parameter BTL

Let \(w=(w_1,w_2,1-w_1-w_2)\) with \(w_1,w_2\ge0\), \(w_1+w_2\le1\). For a normalized 3-vector comparison difference \(D\),

\[
\operatorname{logit}P(Y=1\mid w)
=\beta[D_3+w_1(D_1-D_3)+w_2(D_2-D_3)].
\]

Fit the convex logistic likelihood with a **continuous simplex-constrained optimizer**, and verify it with a second solver or KKT checks. Log held-out negative log likelihood, w error (evaluator only), predicted comparison probabilities and information matrix eigenvalues. A model receiving information along only one preference direction must explicitly report that the remaining direction is unidentified.

For fixed predeclared query batches, a conservative confidence-region diagnostic may use the two coordinate score inequalities

\[
|S_j(w)|\le\sqrt{\tfrac12\log(4/\alpha)\sum_i d_{ij}^2},
\qquad j=1,2,
\]

intersected with the continuous simplex, with \(d_{ij}=\beta(D_j-D_3)\). This follows by a two-coordinate union bound conditional on a fixed set of comparisons. The resulting region need not be a simple rectangle. **Do not discretize the simplex and claim a formally exact certificate**; if a valid continuous-region robust optimization is not yet numerically tractable, still complete the real MLE/policy experiments, and mark the formal confidence certificate UNRESOLVED.

### 7.4 Modest live online comparison

From the same fresh policy and declared query strategy, run:

1. Source-only with 32 original source comparisons, no further labels.
2. Source+64 Fisher-targeted informative comparisons, obtained in two 32-label batches from NEW live trajectories during policy learning.
3. Source+64 uniform comparisons from each full candidate pool, not an artificially low-information leftover pool.
4. Evaluator-only true-preference oracle control.
5. If continuous-region robust selection is actually solved and its numerical extrema checked, a fifth robust-confidence-set deployment arm on the SAME labels as targeted point estimate.

Use **100 policy updates**, initial 256–512 real Gym trajectories/update from a development gradient-quality comparison, and **five independent complete teacher-label plus policy-rollout runs**. Policy scalar return should be evaluated with a valid independent evaluator; report its uncertainty if based on Monte Carlo. Preserve vector rewards so reader can see which of the three objectives drive performance.

**P5 result table:**

| Resource-Gathering condition | Actual source + new labels | Independent teacher/policy seeds | Policy updates / episodes per update | True scalar return initial/final with uncertainty | Three-objective vector change | Held-out BTL log loss / w error | Robust continuous confidence region? | Notes |
|---|---|---:|---|---|---|---|---|---|
| Uniform starting policy | n/a | 1 | 0 | 0.20587. Vector [-0.0678, 0.0173, 0.0266] | n/a | n/a | n/a | Live 2000-episode gap is inside 4 standard errors. Enemy death rate on a forced step is 0.105 |
| True-reward oracle learner | oracle-only | 5 | 100 / 256 | 0.20587 → 0.58207 (sd 0.03352) | death +0.0173, gold −0.0002, diamond +0.6213 | n/a | oracle only | A 40-update development run only reaches 0.22531. The gain appears later and is almost entirely the diamond channel |
| Source-only learned reward | 32+0 | 5 | 100 / 256 | 0.20587 → 0.22617 (sd 0.00314) | death +0.0183, gold −0.0009, diamond +0.0280 | 0.686 / 0.327 | not used | Two solvers agree. The low-sensitivity source does not identify the weight |
| Fisher-selected feedback | 32+64 | 5 | 100 / 256 | 0.20587 → 0.46040 (sd 0.16975) | death +0.0049, gold +0.0038, diamond +0.4213 | 0.686 / 0.378 | diagnostic only | Seeds 0.382, 0.589, 0.223, 0.456, 0.651. Held-out NLL does not move |
| Fair uniform feedback | 32+64 | 5 | 100 / 256 | 0.20587 → 0.43527 (sd 0.21715) | death +0.0095, gold +0.0028, diamond +0.3782 | 0.690 / 0.808 | diagnostic only | Seeds 0.312, 0.681, 0.209, 0.320, 0.655. Overlaps Fisher completely |
| Robust guidance if certified 2D solver available | 32+64 | 5 | 100 / 256 | 0.20587 → 0.21708 (sd 0.00080) | death +0.0105, gold +0.0051, diamond +0.0134 | 0.686 / 0.285 | PARTIAL | A continuous SLSQP worst case matched a 21-point simplex grid within 0.00187 on a diagnostic direction. This is a union-bound diagnostic, not a formally exact certificate. The arm does not learn |

**Interpretations:** If the same effect appears in 3 objectives and 2D preference learning, generality increases. If only the 2D BTL fitting fails while oracle policy learning works, the new difficulty is preference identification. If the oracle does not learn, report that environmental learning limitation and do not draw any preference-learning conclusion from it. A scalar gain under w_star does not imply all three individual objectives improved.

## 8. Final Paper-Facing Evidence

### 8.1 What can be claimed only if actually measured?

| Candidate finding | Required measured evidence | Status YES / PARTIAL / NO | Supporting output | Scope / qualification |
|---|---|---|---|---|
| Correct BTL fit removes reward-optimization confounding | P1 two converged solvers, likelihood and held-out NLL | YES | `outputs/horizon_bridge/p1_btl_repair/summary.json` | On the original extra32 batch the old fit stays at 0.766343 with train NLL 64.85, while bounded minimization and damped Newton both reach 0.336352 with train NLL 41.58. A further Newton step moves w by 4.9e-9. The same stall reappears on every Fisher-selected batch and disappears on fair uniform batches |
| Targeted feedback increases true preference information over a FAIR uniform candidate-pool comparator | P1, P2 same-pool and P3 separate online data | PARTIAL | P1 summary and `p3_dst_live_weak/summary.json` | Fisher reduces interval width and parameter error. Held-out NLL ties fair uniform in P1 and in P3 (both 0.555). It does not raise true return |
| Robust guidance outperforms fully fitted point-estimate guidance | P2 same labels/checkpoint, correct MLE and exact DP | NO | `p2_corrected_forks/summary.json` | Early exact robust return is 0.00729 below the same-label MLE, while its certificate stays positive. Learned update 60 repeats the return gap at 0.0217. The live run has the opposite return gap because the MLE deploys a weight this optimizer learns slowly |
| Interactive online new feedback improves true policy return | P3 actual live Gym rollouts, independent teacher-label and policy seeds | NO | P3 weak and mixed summaries | Weak targeted−source is -0.00536 with 95% interval [-0.0173, +0.0066]. Uniform is tied with source. The true-reward oracle itself finishes at 0.63790, below source-only robust at 0.68307 |
| Adaptation speed and label allocation have distinct consequences | P4 one-factor controls | YES | `p4_budget_timing/summary.json` | Label counts 32, 64 and 128 produce the same return to 1e-7. Freezing phi at 0.5 costs 0.0388 return, and the paired interval excludes 0. Feedback timing does not |
| Findings hold beyond two-objective DST | P5 Resource-Gathering oracle + correct 2D BTL | NO | `p5_resource_gathering/summary.json` | The oracle learns, from 0.20587 to 0.58207, almost entirely through the diamond channel. Ninety-six preference labels do not reduce held-out NLL or identify the two-dimensional weight. The robust arm stays at 0.217 |
| Updated source certification is distinguished from actual true policy improvement | Per-update source/updated certificate and true exact return | YES | P3 per-seed curves and P2 fork table | The weak source certificate is negative at update 1 on 8/8 seeds while true return rises for the rest of training. In P2, robust can improve the certificate and reduce true return on the same labels |

P1–P4 are tabular policy + BTL preference experiments, not end-to-end human RLHF or a theorem for arbitrary neural reward architectures. P5 may be Monte Carlo-evaluated if exact Resource Gathering DP proves impractical. Label its evaluation method and precision accurately.

### 8.2 Clear candidate figures, no unnecessary plot inflation

Aim at **three** publishable figures:

1. **P1:** Old BTL gradient fitting versus converged MLE; source and informed BTL log loss, confidence interval and source informativeness.
2. **P3:** 160-update exact true-return learning curves for S/U/I/M plus oracle, marking feedback arrival at 20/40/60/80; show independent-run variability without clutter.
3. **P2/P4 mechanism:** Original versus updated certificate, actual phi path and matched true gain with correct reward estimates.

The second environment may be summarized in a small table before adding another figure. A large plot collection is not a substitute for interpreting the results.

### 8.3 Completion status (fill after running)

| Stage | Status NOT RUN / PILOT / COMPLETE / BLOCKED | Independent teacher-label and policy runs | Key quantitative finding | Source of result | Notes |
|---|---|---|---|---|---|
| P1 corrected BTL and fair query pool | COMPLETE | 10 label seeds × 6 conditions, plus the seed-100 diagnostic | The old 0.766 versus 0.336 gap is an optimizer failure. Fair uniform matches Fisher on held-out NLL | `16_btl_converged.py` | Weak-source coverage of 10/10 is vacuous |
| P2 converged MLE versus robust checkpoint forks | COMPLETE | 8 early, 5 later, 5 learned-at-20, 5 learned-at-60, plus 3 sampled pairs | Converged robust guidance does not beat the converged MLE on exact true return | `17_dst_fair_forks.py` | Later oracle-developed checkpoint is uncertified at initialization |
| P3 long live DST online preference RL | COMPLETE | 8 weak × 5 methods and 4 mixed × 5 methods, 160 updates | New labels improve held-out NLL and do not improve true return. The true-reward arm saturates lower than source-only robust | `18_dst_live_online.py` | 256 episodes kept after cosine 0.930 versus 0.964 |
| P4 feedback / adaptation resource analysis | COMPLETE | 5 runs for each new setting; P3 supplies the matched reference cells | Phi movement binds. Label count and timing do not | `19_dst_budget_analysis.py` | Paired Student t intervals, df = n−1 |
| P5 3-objective Resource Gathering | COMPLETE | 5 seeds × 5 arms, 100 updates | Oracle learns the diamond channel. Two-dimensional BTL does not identify the teacher, and robust deployment does not learn | `20_resource_gathering.py` | Gradient cosine is only 0.45 at both 256 and 512, so the batch stays 256. The oracle still improves |

## 9. Workload, Runtime and Resource Plan

**Current worker:** one RTX 5090 (32 GB), 12 CPU cores, 62 GB system RAM, approximately 450 GB /workspace persistent disk. No 4B/7B model weights should be downloaded for this plan. Exact DP and convex BTL likelihood optimization are CPU-oriented. The live environment interaction, not a large neural model, may dominate elapsed time.

| Stage | Initial intended workload | CPU/GPU execution | Pilot measured runtime | Estimated full runtime FROM the pilot | Actual full runtime | Peak RAM / VRAM | Bottleneck and response |
|---|---|---|---|---|---|---|---|
| P1 | Old-pool regeneration, 10 label seeds per query/source regime, 512+ held-out pairs | CPU |  |  | 0.70 s | 133 MB RAM, no GPU | Closed-form fits. No change required |
| P2 | 8 early and 5 late labels seeds, 40 exact steps per fork, paired sampled confirmation | CPU DP |  |  | core 16.08 s; sampled check 5.05 s; learned-policy forks 6.58 s | 124 MB | Vectorized DP is 0.27 ms per evaluation |
| P3 development | 3 runs × 20–40 actual Gym policy updates; gradient-quality 256/512 comparison | CPU | 10.32 s for 3 seeds × 3 methods × 40 updates | about 3 minutes for the weak confirmation | 10.32 s | 96 MB | 256 episodes already have cosine 0.930 |
| P3 weak confirmation | 8 independent × 5 methods × 160 × 256 live episodes, about 1.64 million policy episodes | one CPU core |  | about 3 minutes from the 40-update pilot | 132.42 s |  | Live Gym was 37,000 episodes/s. The exact robust search, not the environment, dominates |
| P3 mixed confirmation | 4 independent × 5 methods × 160 × 256, about 0.82 million policy episodes | CPU |  | about 70 s | 66.60 s |  | Same learner |
| P4 | One-factor sweeps, five independent runs per method/config | CPU, reuse working sampler |  |  | 148.89 s |  | Reused P3 seeds 0–4 for the 0-label and 128-label reference cells |
| P5 | Five-run Resource Gathering, 100 updates at 256–512 episodes/update | CPU. GPU was not faster for this tabular model and was not used | 40 oracle updates in 1.76 s, return only 0.225 | a few minutes | 310.40 s | 101 MB | The robust arm's continuous worst-case solve is the slow step. Cosine did not improve from 256 to 512 |

These are purposely MULTI-HOUR candidate jobs. They are not time estimates that have already been verified. First run a real short timing/gradient pilot, record measured episodes/second and peak memory, choose a batch size based on gradient quality, and calculate actual expected wall time. Do not quietly replace 160 updates with 20 or change the 8 independent confirmations to 3 simply to finish sooner.

If live Gym stepping is slow, favor parallel independent jobs under the 12-core and RAM limits, avoiding redundant repeated DP calls when policy logits do not change. A vectorized exact transition simulator can be used for analytical diagnostics, but the main P3/P5 feedback candidate episodes and policy training episodes must actually be collected from the current evolving environment interface to qualify as the stated live experiment.

The executor may run multi-hour jobs and resume interrupted seeds. If capacity becomes temporarily limiting, complete P1/P2 and the **full P3 weak-source independent confirmations** before spending heavily on P4/P5. Existing completed runs must not be discarded; partial status is acceptable with clear seed counts.

## 10. New Source/Output Structure

The following are proposed names, **not** already implemented files:

~~~text
experiments/bridge/16_btl_converged.py
experiments/bridge/17_dst_fair_forks.py
experiments/bridge/18_dst_live_online.py
experiments/bridge/19_dst_budget_analysis.py
experiments/bridge/20_resource_gathering.py

outputs/horizon_bridge/p1_btl_repair/
outputs/horizon_bridge/p2_corrected_forks/
outputs/horizon_bridge/p3_dst_live_weak/
outputs/horizon_bridge/p3_dst_live_mixed/
outputs/horizon_bridge/p4_budget_timing/
outputs/horizon_bridge/p5_resource_gathering/
~~~

Implement reasonable run entrypoints with these names or documented equivalents. Suggested commands:

~~~bash
python experiments/bridge/16_btl_converged.py --out outputs/horizon_bridge/p1_btl_repair
python experiments/bridge/17_dst_fair_forks.py --out outputs/horizon_bridge/p2_corrected_forks
python experiments/bridge/18_dst_live_online.py --stage pilot --out outputs/horizon_bridge/p3_dst_live_weak
python experiments/bridge/18_dst_live_online.py --stage confirm --source weak --out outputs/horizon_bridge/p3_dst_live_weak
python experiments/bridge/18_dst_live_online.py --stage confirm --source mixed --out outputs/horizon_bridge/p3_dst_live_mixed
python experiments/bridge/19_dst_budget_analysis.py --out outputs/horizon_bridge/p4_budget_timing
python experiments/bridge/20_resource_gathering.py --out outputs/horizon_bridge/p5_resource_gathering
~~~

Keep a seed-specific status/config/metrics and small checkpoint per actual complete run; write a single aggregate JSON and CSV per P1–P5 stage, so the filled Work Plan does not depend on untracked giant archives. Commit only small research artifacts. The original H3/H4 and earlier Work Plan 01–04 files remain unchanged.

**Result paths to fill:**

| Stage | Main result summary JSON/CSV | Reproduction script and actual args | Figures | Committed SHA | Notes |
|---|---|---|---|---|---|
| P1 | `outputs/horizon_bridge/p1_btl_repair/summary.json`, `seeds.csv` | `python experiments/bridge/16_btl_converged.py --out outputs/horizon_bridge/p1_btl_repair` | `p1_fit.png` | this commit |  |
| P2 | `outputs/horizon_bridge/p2_corrected_forks/summary.json`, `core_seeds.csv`, `learned_seeds.csv` | `python experiments/bridge/17_dst_fair_forks.py --stage core`, then `--stage sampled`, then `--stage learned` | `p2_mechanism.png` | this commit | Learned checkpoints come from P3 source-only seed 0 |
| P3 weak source | `outputs/horizon_bridge/p3_dst_live_weak/summary.json` and `*_seed*.json` | `python experiments/bridge/18_dst_live_online.py --stage pilot` then `--stage confirm --source weak` | `p3_curves.png` | this commit | Also `pilot.json` and source-only checkpoints at updates 20 and 60 |
| P3 mixed source | `outputs/horizon_bridge/p3_dst_live_mixed/summary.json` | `--stage confirm --source mixed --out outputs/horizon_bridge/p3_dst_live_mixed` | same curve figure is the weak run | this commit |  |
| P4 | `outputs/horizon_bridge/p4_budget_timing/summary.json` | `python experiments/bridge/19_dst_budget_analysis.py` | mechanism panel in the P2 figure | this commit |  |
| P5 | `outputs/horizon_bridge/p5_resource_gathering/summary.json` | `python experiments/bridge/20_resource_gathering.py --out outputs/horizon_bridge/p5_resource_gathering` | none; the table is the record | this commit | Horizon 40 is the frozen cap. The wrapper cap is 100 |

## 11. Exact Execution Instructions and Handoff

1. **Read the newest owner-supplied paper and the already completed Work Plan 04**, H3/H4 code and latest outputs. Understand the differences among hidden preference w_star, fitted BTL w_hat, U_source and U_updated, deployed phi, policy optimizer state, certificate, and true evaluator return.
2. **Execute P1 first.** Correct reward fitting and rebuild a common query pool. The decisive old value 0.766343 versus converged MLE 0.336351 on the same extra32 batch must be explained by direct likelihood and optimization comparisons. Run weak and mixed source conditions, with fair uniform selection from the same unlabeled pool.
3. **Execute P2 core oracle-developed checkpoints** with fully converged MLE versus robust guidance using identical preference-label bits and a fixed guidance movement budget. Use actual 40-update DP return/phi/certificate curves. The learned-policy checkpoint continuation can be completed after P3.
4. **Execute P3 as the main longer experiment.** Implement genuinely live Gym policy learning and fresh preference collection at scheduled updates. Run the pilot, then the full 160-update, 8-seed weak-source and 4-seed mixed-source comparisons S/U/I/M/O. Save resumable seed checkpoints and real runtime. Every independent run must redraw BOTH labels and policy randomness.
5. **Finish P2 learned-policy checkpoints and P4.** Analyze the specific effect of new labels, the phi budget and feedback timing, not a huge full-factorial grid. Report paired independent-seed effect sizes and uncertainty, with null effects preserved.
6. **Execute P5** on the already known three-objective Resource Gathering environment. Establish a learning oracle; fit continuous 2D BTL; run 5 independent real preference-policy replications. If a formal 2D compatibility-set optimizer cannot be validated, say so and keep the preference/policy results without a fabricated certificate.
7. **Fill all result tables in THIS same Work Plan 05**, including actual runtime and result paths, and push source and compact results to GitHub. Return the commit and a short factual research summary. No incremental user approval is required for normal technical decisions.

**Intended result:** a scientifically interpretable, longer-duration sequence able to determine whether (i) reward fitting was the true bottleneck, (ii) robust guidance has an advantage beyond correctly fitted point estimates, (iii) FAIR and genuinely live new preference comparisons improve actual policy return, (iv) feedback timing and adaptation resources interact, and (v) these results persist or fail in a second three-objective environment.

**Completion principle:** Do not optimize the experimental design merely to produce favorable scores. Optimize it so that even a negative or null measured effect has a precise, useful explanation.
