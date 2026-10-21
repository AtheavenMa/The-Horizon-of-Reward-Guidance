# 08 Final Milestone and Paper Freeze

**Project:** The Limits of Reward Guidance under Finite Adaptation  
**Target:** ICML 2027 empirical revision and one public community release  
**Prepared:** 2026-10-20  
**Starting main commit:** 2d47ea0f76d61772b41508c0014adc832e3aa8af  
**Status:** MILESTONE_COMPLETE_SCOPED. F0–F6 have been run. Tables below are filled from `experiments/horizon_final_08/`. The bounded Pythia arm respects each event at b=.05 and does not improve fixed ROUGE-L. Full RewardBench 2, the 4B judge and the 1.5B transfer were not run. No Work Plan 09.  
**Role of this work plan:** FINAL MILESTONE of Work Plans 01–08. Close the present empirical expansion, repair the scientifically material deficiencies, produce a useful public research tool, and freeze a defensible paper-facing result set. Do not create Work Plan 09 as the default next step.

> **Executing research agent:** This is an executable scientific assignment, not an invitation to design more experiments. You have authority to fix normal implementation and dependency problems, use the recorded 1×RTX 5090/12 CPU/62 GB RAM workstation when actually available, and execute a final approximately 10-hour bounded campaign. First verify current resources and repository state. Complete F0–F6 with actual code, data and independent evaluation. Preserve every historic WP01–07 result as an archived baseline, including unfavorable outcomes. Do NOT rewrite the theory or manuscript to fit an attractive observation, do NOT call an empirical neural diagnostic a formal support horizon, do NOT claim a run is complete if the declared budget is violated, and do NOT announce a release without passing the release acceptance gates. Fill THIS SAME work plan, commit source/small raw results and final documentation, and provide the exact final commit and reproduction instructions. Routine decisions do not require asking the owner.

## 0. Milestone Scope and Closure Decision

### 0.1 Why this is the endpoint

The work to date has moved from exact support-horizon theory, through finite and stochastic policy updates, to a 16-outcome coupled reachability example, a 64-outcome sequential task, real tabular DST/Resource-Gathering, authentic human preference fitting, a neural MiniGrid policy and preliminary 0.5B/1B/4B experiments.

The remaining value is in making these results **valid, comparable and reusable**, not in adding another untested task family. After WP08, the default state is **FROZEN / PAPER READY AT THE SUPPORTED SCOPE**, or **FROZEN WITH EXPLICIT QUALIFICATIONS**. A blocking scientific failure may force a downgrade of a claim, but it is NOT permission to enlarge the experiment indefinitely.

**Milestone definition:** The final result is accepted when:
1. No paper-level finding depends on an unverified or violated guidance bound, an inconsistent reward scale, a leaked label/test split, a false baseline, or an unrepeatable model checkpoint.
2. The original theory and small controlled results are preserved, and the paper has a compact, ranked, falsifiable contribution story with provenance for each number.
3. A real human-labeled 0.5B RM study and an actual 1B policy study have been properly evaluated; negative findings remain negative.
4. The independent RewardBench 2 external study is accurately scored or explicitly excluded from numerical external claims.
5. A third-party user can run a CPU finite audit and a proper paired comparison through the public API without prefilled, unverified certificate values.
6. Manuscript-ready figures/tables, limitations, replication commands and a frozen Git revision are available.
7. No further benchmark expansion is necessary to honestly present the established scientific conclusions.

**Deliberately out of scope after closure:** new Walker/Meta-World/SAC sweeps, untested LLM PPO implementations, additional data/model families solely to obtain a positive return, a new general neural support-horizon theorem without proof, or a 4B model-training claim inferred from frozen 4B judging. The original 4B LoRA stretch and UltraFeedback fitting are optional, not required to finalize a scientifically honest paper.

### 0.2 Frozen evidence ledger from Work Plans 01–07

These are RECORDED INPUTS to WP08, not WP08 measurements. Verify the underlying files rather than manually transplanting summary prose.

| Scientific/evaluation object | Historical observation | Source | Qualification to preserve |
|---|---|---|---|
| Original exact scalar support | Source-only T0=5; 1,200 fixed-design informative BTL comparisons illustrate certification through T=15 | [docs/PAPER_RESULTS.md](../docs/PAPER_RESULTS.md), results/reference | Analytic statement for the specified scalar model |
| Ordinary Euclidean PG | Local boundary contact 0.00783587; all-admissible local analytic bound in its specified box | [Work Plan 02](02_Theory_to_Training_Bridge.md) | Not a general PPO/Adam statement |
| Stochastic policy updates | Population source loss around 0.008; realized sampled steps can lose earlier for measured variance reasons; new labels change unchanged-trajectory certification | [Work Plan 03](03_Stochastic_Guidance_Transfer.md) | Population and realized ex-post gains have different meanings |
| Coupled finite MDP | At t=0.012, source value about 0.05993540, numerical all-controls upper about 0.05993753, gamma=.06, pointwise max about .11417 | [Work Plan 04](04_Reachability_to_Small_MDP.md) | Numerical all-controls bound on the 16-outcome discrete finite model; requires interval-rounding audit to call computer-verified |
| DST real RL | 160-update source-only mean 0.68307, informed robust 0.67771, converged MLE 0.65142, ordinary true-w PG 0.63790 across eight sampled seeds | [Work Plan 05](05_Online_Preference_Optimization.md) | No general benefit of more preferences; initial-information setting matters |
| DST matched controller and policy basin | Same-state full-information one-step inequality holds numerically; 160-step return reversal correlates with short-vs-deep treasure occupancy | [Work Plan 06](06_Controller_Parity_and_Identification.md) | Long-run nonconvex path dependence, not globally optimal long-run control |
| RG multidimensional preferences | Minimum information eigenvalue and true-weight estimation are not monotonically linked; full-simplex robust is conservative and scores poorly; privileged lookahead improves | [Work Plan 06](06_Controller_Parity_and_Identification.md) | No valid general multidimensional advantage established |
| Authentic TL;DR HUMAN comparisons | Qwen2.5-0.5B trained RM held-out 512-pair accuracy: .5059 on 257 train pairs, .5508 on 1,024, .5996 on 4,098; extra same-257 fitting ≈.502 | [WP07 S2 results](../experiments/horizon_scale_07/human_tldr/summary.json) | Only one main large-data model seed; length correlation increases; human labels historically collected offline |
| MiniGrid neural policy | Empty-8x8 oracle learns; DoorKey-6x6 40k oracle does not | [WP07 S1 results](../experiments/horizon_scale_07/minigrid/confirm.json) | No DoorKey support-horizon claim; Empty is nearly saturated in successful arms |
| Pythia-1B LoRA | 64 actor updates × 2 seeds; 96 new AI-teacher comparisons in refreshed arm; reported anchor RMS 1.460 and 1.354 versus planned b=.05 | [WP07 S3 results](../experiments/horizon_scale_07/pythia_1b/main.json) | THIS ARM DID NOT SATISFY THE DEPLOYMENT BUDGET. Reward means across changing reward heads are not a fixed true-quality evaluator |
| RewardBench 2 on frozen 4B | 502/1,865 predeclared rows; prior 4B exact-index-set macro .1115 and 0.5B strict macro .2091 | [WP07 S4 results](../experiments/horizon_scale_07/rb2_4b/summary.json) | Prior rule is NOT the official multiresponse/Ties metric; do not cite either as official RB2 performance |
| WP07 community audit | Manifest/CLI load; some prefilled finite gains read without independent verification; compare CLI only checked directory existence | [guidance_audit](../experiments/guidance_audit/) | Prototype, not yet a trustworthy community auditing library |

### 0.3 Five scientific contributions to defend and what WP08 can add

The final manuscript should defend a small number of credible scientific contributions:

**C1. Exact finite preference support horizon.** Existing original theoretical theorem; WP08 adds checked reproduction and a precise boundary on its applicability. NO new theorem is claimed.

**C2. Reachability versus pointwise usefulness.** Existing exact ordinary-PG and finite coupled-MDP evidence. WP08 audits the numerical bound and makes the distinction accessible through a reproducible finite audit.

**C3. Preference information versus bounded reward movement.** Existing BTL and same-checkpoint controls show information alone may improve certification but not actual policy quality. WP08 attempts a genuinely budget-respecting neural-RM control and a fair information-versus-optimization comparison.

**C4. Guidance choice affects long-run optimization paths.** Existing DST short/deeper treasure result and matched-controller forks; WP08 preserves matched gradient/model-access context and gives a compact empirical mechanism statement, NOT a universal superiority or global basin theorem.

**C5. Community reward-reuse evaluation across scale.** A real-human preference RM, real neural policy training and a usable audit framework, with explicit empirical/model-mismatch warning types and a benchmark card.

**What cannot be asserted yet:** Extra neural preference labels provably extend a neural support horizon; 4B is a trained reward model; 4B RB2 macro values are official; Pythia method B meets b=.05; improving AI-teacher reward scores proves human summary preference improvement; more information always improves final return. If F0–F6 cannot repair these, they stay outside the final scientific claims.

## 1. Single Final Experimental Campaign and Precommitted Gates

### 1.1 Stage dependencies and outputs

| Stage | Purpose | Mandatory deliverable | Stop/continue decision |
|---|---|---|---|
| **F0 Evidence audit and frozen inputs** | Repair contradictory records, check implementation/provenance, freeze seeds/evaluators | Read-only baseline ledger, frozen manifest and test protocol | Invalid historic values tagged; continue repairs without silently editing history |
| **F1 Human RM and neural-RL control validity** | Replicate human-label evidence and repair MiniGrid train/fork controls | Validated full human holdout, reliable positive-control and honest confidence | If trained RM fails, do not make language policy-quality claims from its score |
| **F2 Correct bounded 1B policy/RM training** | Enforce functional reward adaptation IN THE UPDATE, restore true checkpoints, use independent evaluation | Main WP08 scientific experiment with 2–3 independent seeds and actual movement accounting | Violations make run INVALID and require code repair before any conclusion |
| **F3 Official external and independent scale** | Correct RewardBench 2 evaluation; test 1.5B transfer if throughput fits | Correct protocol results with reproducible row-level scores; optional 1.5B LoRA | Full leaderboard and optional model must not block validated main WP08 |
| **F4 Matched feedback versus optimization** | Separate new labels, same-data RM fitting, guidance budget, policy effect | Genuine same-checkpoint causal forks and paired fixed-evaluator results | Bad checkpoint restoration invalidates causal comparison |
| **F5 Public API and benchmark hardening** | Replace trust-me certificates/empty comparison action with actual algorithms | Installable package, independently recomputed finite audit, seed-paired CLI and tests | No community-ready claim if an external user's data are not actually analyzed |
| **F6 Paper and repository freeze** | Final claims, figures, compact paper text, QA/replication and release | Final paper pack, signed-off claims matrix, frozen Git revision/tag when available | Stop: mark ready or scoped ready; no automatic WP09 |

### 1.2 Execution budget, actual use audited after each stage

Target one **bounded ten-hour workstation campaign** using the prior single RTX 5090 if available. Time is a planning allocation and can be moved between stages based on measured throughput, not retrospective wishful thinking.

| Stage | Target wall-clock | Priority | Critical cost |
|---|---:|---|---|
| F0 | 0.50 h | Must | CPU source/checkpoint/provenance audit |
| F1 | 1.50 h | Must | GPU human RM 2–3 fitting seeds + CPU neural control repairs |
| F2 | 3.00 h | Highest | Real Pythia-1B policy and reward updates with matched evaluation |
| F3 | 1.25 h | High | Official RB2 metric and frozen 4B scorer; short Qwen1.5B transfer if feasible |
| F4 | 1.00 h | Highest | Saved full-state checkpoint forks, independent scoring |
| F5 | 1.25 h | Must | Community CLI, regression/external-user tests |
| F6 | 1.00 h | Must | Paper pack and freeze/repro audit |
| Reserve | 0.50 h | Conditional | Actual OOM, failed reload, independent rerun |
| **Total** | **10.00 h** | | |

**Budget priorities:** Protect F2's actual valid budget, fixed evaluation, F4 causal fork and F5 public tool. If F3 full 1,865-row RB2 or Qwen1.5B would consume these resources, run the minimum correctly audited external subset and record the scope; do not abandon the central paper. A method that uses more RM steps or new comparison labels must separately report GPU seconds/tokens and label counts. A fixed ten-hour budget does NOT mean pretending all possible stretch experiments finished.

### 1.3 Frozen pre-run evaluation and seed contract

1. Before ANY new final test, fix exact repo SHA, dataset and model revisions, tokenizer, any license restrictions, RNG seeds, train/dev/held-out prompt indices, normalization, length cap, full saved checkpoints and evaluation scripts. Record them in a machine-readable run manifest.
2. Training data: existing frozen TL;DR Reddit training documents and original valid1 split; no document or summary overlapping the independent human validation pool, no after-the-fact label revelation before querying.
3. Human RM main endpoints: initial 257 / 1,024 / 4,098 labeled training comparisons and 512 development / a disjoint final held-out set drawn from the already frozen 2,048 validation comparisons. If the old 512 have already influenced method selection, treat them as development and reserve the OTHER 1,536 for the final frozen test. Do not silently present the originally inspected 512 as unseen final evidence.
4. Pythia final policy evaluation: freeze 128 TL;DR validation prompts from original S3's 128 anchors ONLY if not previously tuned on; otherwise freeze a new disjoint held-out document group before F2. Keep a separate 128-pair reward movement anchor pool, with no overlap with training or final evaluation, and store the actual response-pair hashes.
5. Each full training replicate independently seeds actor initialization/LoRA, generation sampler, candidate selection, preference-teacher coin flips if stochastic, RM training and evaluation. Policy/checkpoint forks share the exact parent actor/optimizer/RNG and controlled candidate responses. Declare dependence among forks.
6. All methods compared on fixed human validation use the SAME evaluator and same prompt/response identities. Updating the Reward Model changes its own score scale; **never compare the method-specific raw reward mean across methods as an independent policy-quality outcome.**
7. Do not choose positive thresholds, b, KL cap, reference metrics, response truncation or RewardBench 2 judge prompt from the final held-out scores. Development choices are frozen before confirm.
8. Source human comparisons are historical annotations. Online labels in S3/F2 remain from the FROZEN independent CleanRL AI teacher and must be described as model-generated labels. No human-on-policy result is invented.
9. Training progression should be independently reproducible with restored AdamW (or other actual optimizer) momentum, scheduler, policy LoRA, RM adapter and head, tokenizer, CPU/CUDA RNG, known split IDs and event count. If only actor LoRA was saved historically, old WP07 forks are NOT considered a fully matched restored-optimizer causal result.

## 2. F0 — Evidence Audit, Cross-Plan Consolidation, and Formal Bound Check

**Objective:** Repair the documentation and verification pipeline before spending another GPU hour. Preserve historical outcomes and mark invalid ones rather than overwriting history.

### F0A. Known correctness defects to close

1. WP07 header states PLAN ONLY despite filled results. Do not rewrite the archived historical record just to improve presentation; create this plan's authoritative status ledger and optionally a tiny non-destructive notice pointing to WP08 final status.
2. Preflight run_s0 contains a reference_untouched row that sets passed=True even when its observed value was false. Replace the vacuous test by hashing existing tracked reference files before/after run, verifying the expected fixture exists and asserting hash invariance. If the old expected reproduction_status.json is absent, explicitly say it is absent; do not pass a nonexistent file check.
3. WP07 finite audit currently trusts prefilled source/updated certificates and pointwise_available/path_feasible flags. Add a computed/verified status and remove all implied formal verification from unverifiable prefilled rows.
4. Keep all scientific phrases aligned with their actual scope: source trajectory loss versus any-control loss; pointwise value versus control horizon; population versus stochastic realized update; source versus updated BTL confidence; original theorem versus neural proxy.
5. Numerical H1 all-controls t=.012 upper 0.05993753 is from a custom reachable-set bound. Re-evaluate its stated interval bounds, arithmetic rounding, covering argument and first failure. If outward-rounded upper bounding is not independently verified, label COMPUTATIONAL EVIDENCE, not computer-verified proof. Preserve the analytic ordinary-PG theorem independently.
6. Fix the original noisy E1 RL and language pilot claims in the final paper pack: learning failures and inconclusive comparisons are explicit negative/limited outcomes, not silently dropped.

**F0 acceptance:** exact T0=5; ordinary-PG contact about .0078358743; frozen label update keeps policy return invariant; exact finite 16-outcome recorded loss and 64-outcome zero-extension reproduction checked; reference hashes unchanged; no visible false passing assertion. Any mismatch is a recorded BLOCKER for that result, not a license to retune a target.

### F0B. Evidence provenance schema

Every paper-eligible table row needs:

- claim_id; theorem/empirical status; scope exact_finite / statistical_fixed_design / empirical_proxy;
- policy/environment/model and revision; reward head version; true evaluator identity;
- original preference set and new comparison labels with source/arrival; confidence construction and alpha when meaningful;
- exact policy or sampled optimizer and step-size/time units; reward adaptation budget definition and VERIFIED actual maximum movement;
- independent seed IDs; paired data unit; evaluation metric and error bar; test-set selection; exact source file plus script and full commit hash;
- first unsupported checkpoint and censoring; any invalid status and unfulfilled assumptions.

**F0 table:**

| Object audited | Expected property | Actually measured/verified | Status PASS / FAIL / PARTIAL | Final evidence path | Restriction on paper claims |
|---|---|---|---|---|---|
| Scalar horizon / ordinary PG | Existing analytic and numeric anchors |  |  |  | |
| 16-outcome coupled bound | Actual valid all-controls bound or downgraded numerical evidence |  |  |  | |
| 64-outcome zero-regression | First two policy stages unchanged |  |  |  | |
| Reference files and hashes | No historical mutation; check must not trivially pass |  |  |  | |
| Human TL;DR label provenance and split | Train/valid document disjoint, true human labels |  |  |  | |
| Pythia WP07 reward movement | Historical b=.05 violation recorded as INVALID for bounded-budget claim |  |  |  | |
| RewardBench 2 old 502 rows | Historical scoring rejected as unofficial |  |  |  | |
| Public audit CLI claims | Calculated versus reported values distinguishable |  |  |  | |

## 3. F1 — Repair Real-Human Preference Evidence and Neural Positive Controls

**Objective:** Make the existing 0.5B human data result statistically readable and avoid letting MiniGrid's saturation/failure distort the paper.

### F1A. Human TL;DR reward-model evaluation freeze

1. Reuse only [07_human_tldr.py](../experiments/large_scale/07_human_tldr.py)'s released HUMAN comparison data, its frozen Reddit group split and Qwen2.5-0.5B-Instruct scalar LoRA/head design. Do not call new AI feedback human.
2. Verify train documents and the untouched final human comparisons are grouped/disjoint. Split frozen valid1 2,048 rows into previously used 512 DEV and 1,536 FINAL TEST by stored stable row IDs. If documents overlap between these two evaluation groups, fix by document-group partition and report revised counts, without allowing any final-test examples in fitting, calibration or hyperparameter selection.
3. The old 257, 1,024, 4,098 comparisons are **nested and not independent**. Train independent seeds (minimum 3 for source and 1,024; minimum 2 and target 3 for 4,098) from the SAME frozen starting 0.5B base and head-initialization policy. Keep the original 512-only old results as historical development comparisons.
4. Evaluate the actual 0.5B frozen full validation split (or final disjoint group) using pairwise accuracy, NLL, Brier, calibration/ECE, response-length correlation and prompt truncation fraction. Use paired **comparison-group/document** intervals and per-training-seed spread; do not treat every pair on one post as an independent model seed.
5. Include a simple source-only logistic baseline using actual observed length/response features. Compare fixed-feature-only versus trainable LoRA with identical data splits. State whether 0.5B's gain is beyond length heuristics; if not, narrow the claim.
6. Equalize optimization work between source-only extra-fitting and new-information conditions as closely as feasible. Record both actual reward optimizer steps/tokens and wall GPU seconds. More 257-pair epochs is NOT a controlled equal-compute contrast to 4,098 separate examples unless explicitly normalized.
7. Test score temperature calibration on DEV only, freeze it, then score FINAL unchanged. No additional validation prompt from final may choose model iterations, ensemble composition or selection method.
8. Save only small derived numeric pair indices/hashes, per-seed metrics and permitted LoRA/head checkpoints; document base-model and dataset licenses before claiming redistributability.

**Acceptable scientific finding:** More authentic human comparisons improve a real learned RM relative to its source condition, with uncertainty and limits. It does NOT automatically imply this RM is useful on every new policy distribution or prove a neural support horizon.

**F1 human result table:**

| Reward learner / source | HUMAN train comparisons | RM fit seeds | Human FINAL comparison count | Accuracy ± appropriate CI | NLL / Brier / calibration | Length/truncation confounds | Equal-compute counterpart | Claim status |
|---|---:|---:|---:|---|---|---|---|---|
| Simple length/feature logistic baseline | 257, then 4,098 | 3 CPU | 995 | 257: 0.527 (0.410–0.588); 4098: 0.583 (0.572–0.591) | 4098 NLL 0.680 (0.673–0.685) | length/overlap features; seed 2 of 257 falls to 0.410 | CPU baseline, not a neural contrast | FINAL logistic is at least as high as the 0.5B |
| Qwen-0.5B fitted RM source | 257 | 3 | 995 | 0.505 (0.489–0.517) | NLL 0.800 (0.797–0.805) | see metrics.json length correlation | extra epochs below | modest, near chance |
| Qwen-0.5B fitted RM medium | 1024 | 3 | 995 | 0.549 (0.543–0.555) | NLL 0.724 (0.716–0.732) | document-bootstrap CI |  | above 257, below 4098 |
| Qwen-0.5B fitted RM large | 4098 | 3 | 995 | 0.563 (0.543–0.574) | NLL 0.682 (0.673–0.689); ECE about 0.036–0.047 | length correlation about 0.39 on seed 0; truncation 0.003 | DEV selected seed 0 at 0.635 | does not beat the 4098 logistic on FINAL |
| Same 257 labels, more optimizer steps | 257 | 3 | 995 | 0.527 (0.517–0.543) | NLL 0.909 (0.885–0.929) | step-normalized, not the same examples | not equal to 4098 | extra epochs do not reproduce the 4098 DEV accuracy |
| Historical WP07 512 development comparison | archived only | 1/2 as originally run | 512 DEV | 0.506 / 0.551 / 0.600 | recorded | length correlation up to .318 | not confirmatory | historical DEV only |

### F1B. Neural Gym reference, limited to one meaningful control

1. Keep WP07 Empty-8x8 learned reference, actual 7×7×3 partial image and named true goal/time objective; test one fresh two- or three-seed repetition from a shared parent with **entire policy AND optimizer state** and matched evaluator seeds. Extra-fitting seed 2 previously failed (0% goal completion) while new 32 BTL pairs succeeded; examine first whether altered reward weight, reused/reinitialized PPO optimizer or policy RNG caused it.
2. Test conditional positive-reference learning with a fixed 100-episode held-out evaluation in exactly the same reward units. Saturated 100% success on Empty need not be a headline paper result; track mean steps as a secondary measure.
3. DoorKey-6x6 failed to learn under the first 40k-step oracle. Do not run another large benchmark sweep. One short development repair/longer oracle check is permitted if an independently validated configuration exists; otherwise record NOT LEARNED and remove DoorKey from any Horizon claim.
4. Do not spend GPU time manufacturing an empirical horizon for a policy that has already converged or has insufficient data at initialization.

**F1 Gym result table:**

| Task / arm | Independent policy seeds | Shared policy + optimizer checkpoint verified? | Train env steps | Success / mean steps with uncertainty | Source vs updated preference effect | What can the paper honestly claim? |
|---|---:|---|---:|---|---|---|
| Empty-8x8, oracle | 2 | yes, AdamW state restored | 8192 parent + 92160 | oracle seed 0: success 1.00, steps 14, seed 1: success 1.00, steps 11 | n/a | success is saturated; mean steps are the secondary measure |
| Empty-8x8, source versus 32 BTL | 2 | yes | same | source_only seed 0: success 1.00, steps 11, seed 1: success 1.00, steps 11; new_32 seed 0: success 1.00, steps 11, seed 1: success 1.00, steps 11 | both succeed; steps 11 on both seeds | not a horizon result |
| Empty-8x8, more fitting same labels | 2 | yes | same | extra_fit_same_source seed 0: success 1.00, steps 14, seed 1: success 1.00, steps 11 | no collapse on these two seeds | the archived seed-2 failure was not isolated; a short 8192-step probe of both the archived weight and a refit also scored 0 |
| DoorKey learnability-only diagnostic | 0 new | n/a | archived 40k oracle | success 0 | no Horizon inference | NOT LEARNED; removed from the Horizon claim |

## 4. F2 — Real Budget-Respecting Pythia-1B Policy and Reward Training

**This is the single most important missing neural experiment.** It determines whether bounded reward-model adaptation can actually be implemented on a 1B policy run and whether newly obtained model-generated preferences change an independently measured downstream result.

### 4.1 The specific WP07 defect to repair, not relabel

The existing code in experiments/large_scale/07_pythia.py trains a candidate RM directly in _fit_ai(), then computes an anchor RMS only at the END of the full training trajectory. Its observed RMS was 1.4600 and 1.3538 with a declared budget b=.05. No runtime clipping or rollback kept a single reward adaptation within the limit. The recorded value was merely a post-hoc diagnostic. This previous run cannot support any limited-movement neural statement.

**Required invariant at EVERY scheduled reward update:**

\[
d_A(R_{k+1},R_k)
= \sqrt{\frac1{|A|}\sum_{(x,y^+,y^-)\in A}
      \left( \left[R_{k+1}(x,y^+)-R_{k+1}(x,y^-)\right]
            -\left[R_k(x,y^+)-R_k(x,y^-)\right]\right)^2}
\le b+\epsilon.
\]

Here A is ONE fixed, frozen and pair-ID-hashed set of 128 prompt/response pairs, and the score scale/temperature is fixed on development. Primary b=.05 in the original WP07 anchor-score units; epsilon a small declared floating-point verification tolerance such as 1e-4, NOT a post-hoc margin enlarged to rescue failures. A secondary b=.2 may be tested only after the primary b=.05 experiment is valid, to determine whether .05 suppresses all useful movement.

**Enforcement choices, in order:**

1. Keep old deployed RM R_old frozen; train a separate candidate R_candidate on exactly the newly revealed 32 AI comparisons. Do not mutate the live deployed RM before measuring proposal size.
2. Create a candidate deployed model by interpolating trainable adapter/head weights. Because neural outputs are nonlinear in weights, evaluate the ACTUAL R_alpha on every anchor pair at multiple alpha values and use a verified line search to accept the largest valid alpha in [0,1]. Check alpha=0 equals R_old exactly and alpha=1 matches R_candidate numerically; save the true accepted anchor movement.
3. If weight interpolation is numerically unstable or changes the score scale, use an explicit **functional score interpolation**
   
   \[
   R_{\rm deploy,\alpha}(x,y)=R_{\rm old}(x,y)+\alpha\left[R_{\rm candidate}(x,y)-R_{\rm old}(x,y)\right].
   \]
   
   This exactly bounds anchor-pair margin movement by alpha times the candidate margin movement. Implement real scoring in the policy update with the resulting explicit mixture and log the extra inference/storage cost. A two-model mixture is a different deployed RM architecture and must be marked as such. Do not call it a single LoRA adapter.
4. If neither approach can honor b, the update is NOT applied and the run is marked BLOCKED/INVALID for the bounded-adaptation claim; continue with the source-only control and other independent tasks. Never set exceeds_b=true and then nevertheless report a bounded positive result.
5. Verify b=0 produces EXACTLY unchanged deployed RM pair margins at all checkpoints. Report both **per-event movement** and cumulative movement relative to the original source head, which are distinct. Anchors constrain only the specified finite response-pair set, not every possible response.

### 4.2 Make policy updates a controlled, independent-learning comparison

**Main policy:** existing cached cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr, with LoRA initialized from the same frozen SFT checkpoint. Keep the same task/domain across arms. The source 0.5B human-trained RM comes from a clearly selected F1 checkpoint, chosen by DEV only.

**Main full training arms:**

| Arm | Available preference information | Reward deployment | Policy budget | Purpose |
|---|---|---|---|---|
| A Source-only | Human TRAIN pairs available at initialization; 0 new on-policy comparisons | Fixed source RM R0 | 64 LoRA updates | Baseline |
| B Bounded new AI feedback | Same R0 plus 32 on-policy AI-teacher pairs just before steps 16/32/48, total 96 | Newly fitted RM with enforced b=.05 PER EVENT | 64 same updates | PRIMARY information + bounded movement experiment |
| C Information-only / b=0 | Same 96 AI pairs as matched B whenever the pair pool is shared or replayed | Deployed R remains R0 with zero measured margin change | 64 same updates | New labels cannot affect training without deployment |
| D Unbounded extra feedback | Same pair identities and AI label bits as B where possible | Candidate RM deployed without movement constraint | Short matched fork only, NOT the main scientific bounded comparison | Diagnoses whether the b limit is binding |
| E True reward oracle | Frozen independent AI teacher, explicitly privileged | Uses fixed teacher directly | Development-only matched 8–16 steps | Establish policy optimizer can produce an objective response |

A and B and C are main methods, minimum **two fully independent training seeds**, target three if actual worker throughput allows. B and C may share B's generated label identities in a controlled same-checkpoint fork; whole independent online policy trajectories diverge and their candidate pools cannot be assumed identical after the first intervention. Distinguish those comparisons.

**Train-policy correctness:**

1. Save actor LoRA, optimizer (AdamW momentum and learning rate/scheduler), frozen SFT reference identity, reward-head deployment state, all RNG states and generated pairs at policy steps 0/16/32/48/64. Save actual per-response token log probabilities and response masks. Old WP07 checkpoints saved only trainable actor weights at 16/48 and do not qualify as complete optimizer-state restoration.
2. Validate completion-token masking: the current _logprob concatenates prompt + summary text and chooses a boundary from a separately tokenized prompt; test adversarial token-boundary/BOS cases and compare to direct token-ID concatenation so no prompt tokens enter policy loss inadvertently.
3. Add explicit reference KL using the SAME frozen SFT reference on sampled continuation tokens, with documented mean-per-token versus sum-per-sequence definition. The old mean_logp is NOT a reference KL metric. Use an actual KL penalty or stopping/constraint mechanism with a frozen DEV-chosen limit (initial planned cap .1 nats/token only if the metric and training remain meaningful); report actual excess and any stopped policy steps.
4. Record episode/completion truncation, empty/refusal rate, output diversity and response length. The old implementation generated only 32 new tokens, below the originally planned 96–128. Choose one realistic DEV-tested maximum of 64 or 96 tokens if throughput allows, then fix that number identically in all arms; explicitly record if the run must retain 32.
5. Use independent policy/AI-label RNG per full seed. Fix 256 training document identities and 128 independent held-out document identities as stated in F0, with group hashes. During confirm do not choose the best seed based on a favorable teacher score.
6. Use a new RUN folder under experiments/horizon_final_08/, leaving all WP07 models, results and label records read-only. Run two 64-update primary seeds first; the third seed is higher priority than a new optional 4B LoRA.
7. Exact batch/optimizer compute is not necessarily equal between R0 scoring and candidate/mixed reward scoring. Log generated tokens, model forward passes, LoRA optimizer steps and wall/GPU seconds, as well as new preference labels, for each method.

### 4.3 Independent evaluation, before and after real policy training

The first paper-eligible neural performance quantity is **NOT** the trained RM's own method-specific mean reward.

For each of the 0/16/32/48/64 actor checkpoints and each full seed:

- Generate summaries on ONE fixed held-out 128-document prompt set under predeclared deterministic and sampled decoding settings. Freeze the sampled generation seeds and pair document IDs. Save per-document numerical scores and response hashes.
- Evaluate every arm under the SAME unchanged metric suite: reference summary overlap (e.g. ROUGE-L or a declared deterministic text metric), length, diversity, formatting/coverage checks and reference KL. A fixed independent frozen teacher can be reported as an additional **AI-judge** metric, never described as human quality.
- The independent 4B judge may be added as a SECONDARY score only if its protocol has passed F3 parser/order robustness diagnostics. The known faulty WP07 judge is not automatically an independent gold evaluator.
- Evaluate R0 and each refreshed RM on the SAME fixed human-labeled preference holdout from F1; report pair ranking/NLL and pair-score calibration. This is static human RM generalization, not a new human evaluation of live summaries.
- On a separate fixed anchor pair set, measure old versus new RM margin drift and reversal fraction. A newly fitted RM can improve teacher agreement yet worsen held-out HUMAN preference accuracy; record both.
- Report paired policy-seed differences and prompt-level paired bootstrap CIs clearly nested, with no false inflation of the number of independent training seeds.
- Compute a strictly labeled EMPIRICAL reuse endpoint only if a DEV-frozen criterion and repeated checkpoint measurements support it. If usefulness does not fail by update 64, report CENSORED. Do not infer a formal neural support horizon.

**Primary F2 result table:**

| Policy and reward arm | Independent full seeds | New human / AI pairs | Real policy optimizer steps and generated tokens | Max PER-EVENT anchor movement vs b=.05 | Reference KL / response length | Fixed-evaluator score at 0,16,32,48,64 | Original/updated RM held-out human NLL | Actual paired gain vs A with uncertainty | Status |
|---|---:|---|---|---|---|---|---|---|---|
| A source human RM, fixed | 2 | 0 / 0 | 64 LoRA steps, 64 new tokens | 0 | per-token KL within about ±0.01; mean length about 46–47 words | seed 0 [0:0.161, 16:0.165, 32:0.163, 48:0.164, 64:0.166]; seed 1 [0:0.161, 16:0.161, 32:0.158, 48:0.159, 64:0.162] | R0 accuracy 0.578, NLL 0.683 on 995, float32 deployment | reference | valid fixed-RM control |
| B online AI feedback, BOUNDED | 2 | 0 / 96 AI | 64; three events, 64 candidate AdamW steps each | max event 0.0497; cumulative vs R0 0.109 and 0.113 | no KL step stopped at the 0.1 cap | seed 0 [0:0.161, 16:0.162, 32:0.165, 48:0.166, 64:0.163]; seed 1 [0:0.161, 16:0.158, 32:0.158, 48:0.159, 64:0.161] | step-64 accuracy 0.583 and 0.582; NLL 0.683 and 0.683 | mean -0.0017; 95% paired t [-0.016, 0.012]; N=2 | per-event budget held; no ROUGE gain |
| C online AI labels, b=0 | 2 independent R0 runs | 0 / 96 counted, texts not replayed | 64 | EXACT 0 | same definition | seed 0 [0:0.161, 16:0.167, 32:0.162, 48:0.158, 64:0.158]; seed 1 [0:0.161, 16:0.158, 32:0.161, 48:0.156, 64:0.161] | same R0 | not a matched label fork | margins unchanged; label bits were not reloaded |
| D unbounded RM refresh, diagnostic | 0 |  |  |  |  |  |  |  | NOT RUN: B did not persist the candidate texts |
| E frozen AI teacher policy control | 1 DEV seed, 16 steps | privileged AI teacher | 16 | n/a | KL 0.009 at step 16 | 0.161 to 0.161 | n/a | no meaningful ROUGE change | oracle-only; optimization, not only b, is a bottleneck |
| Historical WP07 B | 2 archived | 0 / 96 | 64 | 1.460 / 1.354 END-TO-END, exceeds b | not logged | method-specific training RM only | not evaluated | NOT A VALID BOUNDED CLAIM | historical |

**Mandatory outcome interpretation:**

- If B obeys b and beats A on independent fixed evaluation: a positive empirical bounded-adaptation result with label/seed/task scope. It is NOT the original theorem applied to neural adapters.
- If B obeys b, new AI labels improve model fitting, but final true/reference quality is unchanged: information may have been useful only to the AI teacher, or the functional movement budget limited the policy response.
- If D improves on a matched parent while B does not: the declared b was binding on those anchors, but generalization beyond the anchors remains unproved.
- If even E cannot change fixed evaluation quality under 16 actual LoRA updates: policy/KL/optimization is the main bottleneck, so do not claim a Reward Guidance support limit.
- If no method beats A, retain the negative result. Do not replace A with an intentionally undertrained RM or move the evaluator threshold.

## 5. F3 — Correct RewardBench 2 Scoring and One Independent Model-Family Probe

### 5.1 Exact source of the WP07 external problem

The previous 502-row RewardBench 2 subset used (i) every chosen response beating every rejected response for a scalar RM, and (ii) a frozen Qwen3-4B prompted judge returning a set of supposedly tied-best indices with exact set matching, 400-character candidate truncation, 800-character prompt truncation and a brittle integer parser. Its recorded macros .2091/.1115 are **not official RewardBench 2 scores** and are NOT suitable for citation as such.

Use authoritative source and implementation:
- Dataset card: https://huggingface.co/datasets/allenai/reward-bench-2
- Official repo instructions and scoring: https://github.com/allenai/reward-bench
- RewardBench 2 evaluator scripts: https://github.com/allenai/reward-bench/blob/main/scripts/run_v2.py and https://github.com/allenai/reward-bench/blob/main/scripts/run_generative_v2.py

The six subsets are Factuality, Precise Instruction Following, Math, Safety, Focus and Ties. RewardBench 2 scores chosen versus rejected responses and treats Ties with a special validity/margin calculation. Generative models can require ratings-based evaluation for the many-candidate Ties subset. Follow the EXACT OFFICIAL implementation and pinned revision, not the WP07 set-match proxy.

### 5.2 Benchmark repairs and parity tests

1. Pin an actual official reward-bench repo commit and inspect its V2 dataset loader, Ties scorer, category aggregation and generative-judge prompting. Run a small FIRST-PARTY fixture on 6–12 selected DEV examples and compare per-row/subset results to an independently implemented metric on the same fixed scores. Do not use the held-out 502 labels to optimize prompts or thresholds.
2. Preserve the precommitted 502 row IDs from WP07. Load the original FULL completions and proper conversation/prompt boundaries; eliminate arbitrary 400-character truncation (document any tokenizer context cap and which rows are affected). For scalar RM, compute a score for every candidate completion, apply official correctness/margin rules and include all 102 Ties.
3. For the frozen Qwen3-4B judge, use the official generative V2 protocol or a clearly separate validated ratings-based adaptation that handles Ties. Precommit the parser, answer positional randomization (two orderings if practical), invalid-response treatment and temperature. A judge prompt with exact index-set scoring is a historical ablation, not the main protocol.
4. Report each row's ID, subset, number of candidates, validation/parse status and score. Compute six category means, official-style macro and prompt-level uncertainty on the fixed 502 only. If official code cannot be invoked, publish a clearly labeled **protocol-compatible independent replica** after fixture parity, NOT an official leaderboard submission.
5. If throughput permits, extend to all 1,865 test rows with the exact same frozen scorer/template. On a 502-row subset, do NOT claim official full RewardBench 2 leaderboard performance. The previously displayed .1115 is retained only as a historical invalid-protocol diagnostic.
6. Do not automatically treat an instruction-tuned 4B LM-as-judge and a 0.5B trained scalar reward model as the same type of scorer or a pure parameter-count scaling law. The 4B is INFERENCE ONLY unless actual pairwise training steps run. Report this explicitly.
7. Record independent errors: position bias, invalid parsing, incomplete score vectors, truncated candidates, loss of ties ranking and absent baseline score calibration. If errors exceed a DEV-frozen validity threshold, downgrade the method's numerical paper eligibility instead of doctoring the format.
8. Use external test data strictly once for confirmation; do not retrain any model on these rows or hand-select a favorable subset after looking at their scores.

**F3 external table:**

| Evaluator and role | Data size and exact test subset | Ties metric and per-row parity | Invalid/parsing/truncation rate | Factuality / Precise IF / Math / Safety / Focus / Ties | Macro and appropriate uncertainty | Correctly official/protocol-compatible/subset? | Paper-eligible? |
|---|---|---|---|---|---|---|---|
| Original WP07 scalar 0.5B | archived 502 | wrong scoring, diagnostic only |  | old .2091 macro | not official | invalid historical | NO |
| Original WP07 4B judge | archived 502 | exact-set proxy, wrong protocol |  | old .1115 macro | not official | invalid historical | NO |
| F3 0.5B human TL;DR-trained RM | same 502 row IDs, full completions | fixture 8/8 matched the official nontie rule; Ties use the vendored overall formula | 580 completions hit the 512-token cap; no character truncation | 0.375 / 0.263 / 0.253 / 0.250 / 0.163 / 0.008 | macro 0.218 on the 502-row subset only | protocol-compatible independent replica, commit 59f6ff7005a9 | subset diagnostic only; this head is a TL;DR scorer |
| F3 frozen Qwen3-4B judge | same 502 | official path needs vLLM |  |  |  | NOT RUN | NO homemade score |
| Full 1,865 optional | not run |  |  |  |  | NOT RUN | NO |

### 5.3 Second policy family as a realistic, bounded optional confirmation

The old WP07 plan promised Qwen2.5-1.5B-Instruct Policy LoRA but never executed it. Make ONE attempt in F3 if primary F2 validity, first-party RB2 scoring and resource measurements are secure.

- Reuse the E6 synthetic/verifiable instruction task BUT repair its previous 24-train/8-eval tiny scale and one-policy-step restriction. Freeze a new 128-train/64-validation group-disjoint generated task set and 32–48 genuine actor LoRA updates, using two methods and at least one reproducible seed; add second seed only if time permits. Keep 2–4 independent verifier components visible, not one opaque utility.
- Fixed source BTL labels and new verifier-generated synthetic BTL comparison pairs are tagged SYNTHETIC, never human TL;DR. Use a bounded functional score movement definition compatible with F2 only if the deployed RM actually supports measuring it.
- Report actual generated response length, per-component verifier pass rates, fixed validation utility, KL, policy changes, label/optimizer work and negative outcomes. If verifier tasks remain unsolved, do not claim cross-family generalization.
- A frozen 4B judge on RB2 is a valid model-SCALE check only for its declared scoring protocol; it does NOT substitute for an actually trained Qwen1.5B policy. If this independent policy run is not feasible, mark F3 transfer NOT RUN and do not let it block an honest final milestone.

**F3 second-family row:**

| Model/task | Real LoRA policy steps | Label source and total | Seeds | Fixed verifier utility before/after | KL, output length, controls | Valid functional budget? | Status and interpretation |
|---|---:|---|---:|---|---|---|---|
| Qwen2.5-1.5B verifiable instruction reference | 0 | no training | 0 |  |  | n/a | NOT RUN |
| Qwen2.5-1.5B source-only | 0 | synthetic source | 0 |  |  |  | NOT RUN |
| Qwen2.5-1.5B informative feedback | 0 | synthetic new BTL | 0 |  |  |  | NOT RUN; it does not block the scoped milestone |

## 6. F4 — Close the Causal Resource Experiment with Fully Restored Checkpoint Forks

**Objective:** Answer the practical question "more preference information, more optimization on existing information, or insufficient reward movement?" while all methods share the same actual actor, optimizer, reward scale and evaluation procedure.

### 6.1 What was wrong with the previous S5 forks

The archived WP07 S5 experiment used actor LoRA snapshots at 16/48 but recreated the AdamW optimizer rather than restoring its momentum and RNG. The new-human, extra-fit and AI-label arms used different RM optimizer-step counts, and their reported final_mean_reward came from whichever reward/teacher model each branch deployed. A teacher score near zero and an RM score near −6 cannot be used as a fair causal quality comparison.

Do not "repair" this by simply rescaling the existing terminal scores. Re-execute the controlled comparison with valid checkpoints and fixed independent evaluation.

### 6.2 Experimental design

Parents: F2 valid source-only and bounded-new-feedback Pythia policies at actual saved steps 16 and 48, with independent policy seed identities. Restore:

- actor LoRA and full policy optimizer/scheduler state, frozen SFT model and exact previous RNG;
- current deployed source/updated reward checkpoint or explicit functional mixture, its own optimizer state, score calibration and current anchor movement budget;
- exactly the same cached candidate summaries, prompt IDs, online comparison-label bits, tokenizer and training/evaluation random streams before a branch changes its named variable.

**Paired branches at a parent state:**

| Method code | Newly revealed comparison information | RM fitting budget | Guidance movement | Policy continuation |
|---|---|---|---|---|
| K keep | zero | zero | none | same 8–16 real LoRA steps |
| C compute-only | zero | fixed +M steps on SAME human source data | same b=.05 | same |
| H historical-human | 32 previously hidden EXISTING human TRAIN pairs; NO new human annotations | fixed +M steps on expanded data | same b=.05 | same |
| A online-AI | 32 newly generated policy response pairs labeled by frozen CleanRL teacher | fixed +M steps on expanded data | same b=.05 | same |
| B both | same 32 H or A labels, specifically reusing their stored bits | +2M steps total | same b=.05 | same |
| Z no-deployment | same label pool as paired H/A | train new RM but b=0 | no new reward deployed | same |
| T privileged teacher | teacher reward directly, evaluator-only/model oracle | named no-RM baseline | no learned b assertion | same |

Choose M in development based on observed RM convergence and GPU time. Count actual optimizer STEPS (one optimization step, not an entire pass over 32 pairs), processed tokens and GPU seconds precisely. Training labels and examples may differ between C/H/A: this is unavoidable but quantified. H and A are separate arms because historical HUMAN labels and ON-POLICY AI feedback are not interchangeable. For query selector comparisons, pool identities are frozen before label reveal, and all candidate query sets have the same opportunity count.

For all branches keep the actor continuation batch/token budget, policy updates, reference KL calculation and frozen evaluation set identical. Report N=128 held-out documents and independent policy seeds. Run parents 16 and 48 from at least TWO independently trained F2 seeds; the third seed if F2 has produced it. Methods that share the same parent seed are PAIRED, not independent additional training replicates.

**Primary endpoints:**

1. Fixed independent held-out utility after the actual continuation minus the PARENT checkpoint's same-evaluator utility.
2. Fixed 128-summary behavior metrics and reference KL; RM held-out human pairwise NLL on the same frozen comparison rows, plus score drift on the same fixed anchor response pairs.
3. Whether new feedback changes the actual deployed guidance after b clipping. Report both proposal movement and accepted movement, and the rejected/line-search fraction.
4. A no-actor-update information-only fork must yield identical generated outputs/held-out utility under fixed decoding; confidence/ranking may change, policy quality MUST NOT.
5. Per-seed paired differences and 95% intervals based on correct nesting (policy seed as independent training replicate; prompt-level CIs separately), and actual resource costs.
6. A null or inconsistent independent quality difference must be reported as unresolved; no reranking branches after seeing final scores.

**F4 result table:**

| Parent checkpoint / seed group | Arm | Label origin and count | Actual RM fit steps / tokens / GPU sec | Accepted max per-event movement | Independent fixed-evaluator quality delta from parent | RM human held-out NLL | KL / response length | Paired effect vs K | Supported mechanism and caveat |
|---|---|---|---|---|---|---|---|---|---|
| 16 and 48, seeds 0 and 1, parents A and B | K | see row files | M=32 except B uses 64 and K/T use 0 | see movement field; every applied event ≤ b+.0001 | mean ROUGE delta vs K 0.0000 (0.000–0.000) | scored when the reward changed; unchanged arms keep the parent NLL | KL recorded per row | paired with the parent, not a new training replicate | null on ROUGE-L |
| 16 and 48, seeds 0 and 1, parents A and B | C | see row files | M=32 except B uses 64 and K/T use 0 | see movement field; every applied event ≤ b+.0001 | mean ROUGE delta vs K -0.0019 (-0.005–0.002) | scored when the reward changed; unchanged arms keep the parent NLL | KL recorded per row | paired with the parent, not a new training replicate | null on ROUGE-L |
| 16 and 48, seeds 0 and 1, parents A and B | H | see row files | M=32 except B uses 64 and K/T use 0 | see movement field; every applied event ≤ b+.0001 | mean ROUGE delta vs K -0.0010 (-0.003–0.002) | scored when the reward changed; unchanged arms keep the parent NLL | KL recorded per row | paired with the parent, not a new training replicate | null on ROUGE-L |
| 16 and 48, seeds 0 and 1, parents A and B | A | see row files | M=32 except B uses 64 and K/T use 0 | see movement field; every applied event ≤ b+.0001 | mean ROUGE delta vs K -0.0003 (-0.004–0.003) | scored when the reward changed; unchanged arms keep the parent NLL | KL recorded per row | paired with the parent, not a new training replicate | null on ROUGE-L |
| 16 and 48, seeds 0 and 1, parents A and B | B | see row files | M=32 except B uses 64 and K/T use 0 | see movement field; every applied event ≤ b+.0001 | mean ROUGE delta vs K -0.0009 (-0.003–0.005) | scored when the reward changed; unchanged arms keep the parent NLL | KL recorded per row | paired with the parent, not a new training replicate | null on ROUGE-L |
| 16 and 48, seeds 0 and 1, parents A and B | Z | see row files | M=32 except B uses 64 and K/T use 0 | see movement field; every applied event ≤ b+.0001 | mean ROUGE delta vs K -0.0013 (-0.003–0.003) | scored when the reward changed; unchanged arms keep the parent NLL | KL recorded per row | paired with the parent, not a new training replicate | null on ROUGE-L |
| 16 and 48, seeds 0 and 1, parents A and B | T | see row files | M=32 except B uses 64 and K/T use 0 | see movement field; every applied event ≤ b+.0001 | mean ROUGE delta vs K -0.0008 (-0.005–0.003) | scored when the reward changed; unchanged arms keep the parent NLL | KL recorded per row | paired with the parent, not a new training replicate | null on ROUGE-L |

### 6.3 Small finite/theory connection to keep the paper coherent

Repeat only the required WP03/WP04 SAME-POLICY information-only control and the DST WP06 matched-controller example as frozen fixtures for the F4 reporting pipeline:

- exact policy trajectory, updated BTL compatibility set, changed certificate but UNCHANGED true progress;
- changed actual guidance may have small, zero or negative future true-return difference under the shared controller; trace the active uncertainty endpoint and b movement;
- 16-outcome coupled all-controls upper bound is a special finite result and should not be carried into Pythia;
- nonconvex long-run DST path result should be called a distinct empirical finding, not an explanation for all language-policy behavior.

The main paper needs a **single connected narrative**, not a disconnected pile of benchmark tasks.

## 7. F5 — Make the Public Guidance Audit a Real Recomputing Tool

**Objective:** Close the gap between a schema checker and a trustworthy reusable research instrument.

### 7.1 Direct known limitations that must become tested features

The existing experiments/guidance_audit/finite.py accepts already-filled source_certificate, updated_certificate, pointwise_available and path_feasible, and echoes the values as though the finite audit established them. The existing CLI compare subcommand only validates that two folders exist; it computes no paired differences. Those are useful scaffolds, but not sufficient for a community research artifact.

**Required behavior A — independently recomputed small finite audit**

1. Support a versioned self-contained exact finite payload with at least known outcome probabilities and reward vectors for every policy checkpoint BEFORE/AFTER the declared policy step, preference-set endpoint(s) or explicitly specified finite reward polytope, the required gain and a declared time/step duration. Compute J_w(pi_before), J_w(pi_after), per-step gain and minimum compatible gain from the actual supplied vectors; do not require a hidden true weight.
2. For a 1D affine preference interval, min compatible finite-step gain MUST be recomputed at the interval endpoints from supplied outcome values. For 2D/full simplex, compute the exact affine worst gain over the three vertices WHEN the set really is the entire simplex. For arbitrary non-simplex uncertainty, implement a validated LP/convex or conservative outer-bound route or explicitly return NOT_VERIFIED; never choose arbitrary candidate-grid minimum as a certified worst case.
3. For the existing scalar moving-target parameters, allow direct invocation of horizon.support_horizon/interval_feasible with provided source set, drift and b, and output pointwise feasibility versus trajectory feasibility distinctly. A computed trajectory-wide bound must state the precise model assumptions; do not apply it to arbitrary recorded LLM checkpoints.
4. When a user passes only stored source_certificate or path_feasible from a historical log without the underlying state/reward evaluator, mark them REPORTED_UNVERIFIED, never independently computed exact_finite. Keep compatibility with version-1 WP07 manifests by a clear migration adapter or warning; use schema_version 2 if necessary and test both paths.
5. Verify movement |phi_{k+1}-phi_k|<=b numerically from supplied actual deployed parameters. For an empirical neural RM functional movement, require anchor pair score vectors before and after each event, recompute RMS and compare to declared b and epsilon. On missing anchors, return BUDGET_UNVERIFIED, not pass.
6. A source certificate is computed from U_source, an updated certificate from U_updated. They are not interchangeable for the first unsupported time. Report source first loss and after-feedback suffix first loss separately, plus whether earlier actual realized steps have already broken continuous certification.
7. Fixed-N BTL interval calculation may be offered only when preference parameter class/query design/independent binary-label assumptions are explicit; otherwise return empirical_only. No time-uniform probability claim from many uncorrected fixed 95% intervals.

**Required behavior B — a real paired-comparison CLI**

8. Implement a CLI that reads TWO machine-readable per-seed result directories or CSV/JSON files and aligns independent training seeds, experiment conditions, parent checkpoint IDs and common evaluation prompt IDs. Mismatched source policy hashes, different metrics or unequal reference evaluators must raise explicit exceptions or report NON-COMPARABLE.
9. Calculate per-seed delta, paired mean, sample SD, N, clearly labeled 95% paired t or bootstrap interval and fraction of positive independent SEED effects. If N<2, return mean with interval unavailable rather than invent one. Distinguish seed-level uncertainty from within-seed prompt bootstrap.
10. Compare original and feedback-updated RM behavior under identical held-out human comparison identities where possible, and actor quality on fixed held-out prompts. Do NOT aggregate reward values from different uncalibrated heads as comparable scores.
11. Include cost columns when available: label count/source, RM optimizer steps, generated token counts, GPU seconds and per-event reward movement. Absent data become missing, not zero.
12. Output deterministic result files paired_comparison.csv, summary.json and one optional simple chart, plus a note on what cannot be inferred.

**Required behavior C — external use and nontrivial tests**

13. Run at least two TRUE computed finite numerical examples, using explicit 16-outcome data and the scalar exact reference, in a clean CPU environment. Alter the input policy probabilities by hand and confirm the recalculated certificate changes; a pass-through implementation would fail this test.
14. Include an adversarial test where a user supplies a plausible but FALSE stored certificate inconsistent with the underlying probabilities. The tool must reject/downgrade the certificate or display corrected calculated value. Similarly make b=0 with changed neural anchor scores fail the budget validator.
15. Include matched-seed comparison tests with known paired arithmetic, a missing seed, mismatched policy SHA, an independent unit of repeated prompts, an uncorrected adaptive BTL interval, no-availability of true reward and a censored trajectory.
16. Verify importability from outside repository root with no GPU/large HF dependencies and optional extras installation. A user should be able to replace the example manifest with their own CSV/JSON, run the CLI, and obtain an actual computed result.
17. Preserve existing python horizon.py and original tests, keep historic result records, and avoid dependency upgrades just to make the example pass.

### 7.2 Public CLI/API acceptance interface

The names are illustrative; write runnable commands with real --help, not illustrative strings unsupported by code.

~~~bash
python -m experiments.guidance_audit validate --manifest examples/08_guidance_audit/finite_computed.json
python -m experiments.guidance_audit audit --manifest examples/08_guidance_audit/finite_computed.json --out experiments/horizon_final_08/public_smoke/
python -m experiments.guidance_audit compare --baseline experiments/horizon_final_08/pythia_source/ --treatment experiments/horizon_final_08/pythia_bounded/ --metric fixed_eval --pair-by seed --out experiments/horizon_final_08/public_compare/
python -m pytest -q tests/guidance_audit/
~~~

If not using pytest, implement a documented standard-library unittest runner with the same tested behavior. Ensure the public CPU command succeeds without first importing Torch or downloading Qwen.

### 7.3 What the released evaluation package claims

The package should be called an **audit and evaluation tool**, with three explicit result scopes:

- VERIFIED_FINITE: independently calculated exact finite return and compatibility-set minima with correct source data and known assumptions;
- VALID_FIXED_BTL: conditional/fixed-N statistical interval with all required assumptions and sample identifiers, distinguished from whole-path simultaneous coverage;
- EMPIRICAL_ONLY: neural RM scores, human-label held-out measurements, model-judged preferences, sampled policy performance, RM functional anchor budget and descriptive first criterion failure. No formal neural horizon claim.

It may recommend collect_more_comparisons, fit_existing_comparisons_more, reward_change_budget_limited, policy_optimization_limited or no_supported_action only when the actual recorded evidence supports the chosen rule. A response named "no_supported_action" is entirely valid.

**F5 community acceptance table:**

| Public feature | Exact test / independent expected observation | Measured test result | PASS / PARTIAL / FAIL | Path and git revision | Meaning for another researcher |
|---|---|---|---|---|---|
| Real computed finite returns and robust gains | Perturb outcome probabilities; computed values must change | 9 unittest rows passed, including the perturbation test | PASS | experiments/final08/f5_tests.py at base d5377ae30d16 plus this pack | A stored certificate is recomputed |
| Historical unverified values downgraded | False stored certificate rejected or labeled reported_only | false 0.99 certificate corrected; schema-1 manifest stays REPORTED_UNVERIFIED | PASS | examples/08_guidance_audit/ | Old logs are not silently certified |
| Actual movement budget verification | b=0 + changed anchor -> fail; bounded event -> pass | unit test passed; F2 events stayed ≤ 0.0501 | PASS | experiments/guidance_audit/finite.py | Missing anchors return BUDGET_UNVERIFIED |
| Proper source/updated and censored horizon fields | Same frozen path, label-only change | scalar T0 recomputed; neural reuse endpoint left CENSORED | PASS | finite.py | No neural support horizon is inferred |
| Real paired CSV statistical comparison | Known seed-level paired numbers, valid t/boot CI | F2 B−A mean {facts['paired']['paired_mean']:.4f}, N=2, interval reported | PASS | compare_stats.py | N<2 does not invent an interval |
| Independence, missing data and hash checks | Mismatch not silently accepted | evaluator mismatch raises ComparisonError | PASS | compare_stats.py | Different heads are not averaged |
| CPU clean import / 5-minute demo | No Torch/HF/GPU required | the audit package source has no torch import; tests passed | PASS | experiments/guidance_audit/ | CPU users can audit a finite manifest |
| Optional Gym and real 0.5B RM adapter | Real file input, honest empirical_only result | MiniGrid and the 0.5B head are empirical_proxy | PARTIAL | experiments/horizon_final_08/ | Not a finite certificate |
| Licenses and sample provenance | Third-party data not rehosted without rights | TL;DR, RewardBench 2 and model weights stay in the local cache | PASS | docs/BENCHMARK_CARD_08.md | Row hashes and IDs are what the repo stores |
| Paper results regenerated from benchmark records | All headline numbers match committed metrics | this pack reads the JSON files and writes the tables | PASS | experiments/final08/f6_pack.py | Headline numbers are not typed by hand |

## 8. F6 — Manuscript Evidence, Community Release and FINAL FREEZE

**Objective:** The work is temporarily finished after this milestone. Do not leave an endless new-task backlog. A reader should be able to understand the scientific contribution, independently reproduce its supported evidence and know the limitations without further explanatory chats.

### 8.1 Paper contribution hierarchy and exact placement

The original paper's analytic structure should remain intact. Work Plan 08 produces the empirical/material needed to strengthen that manuscript, not a replacement of its theorem.

**Main-paper narrative, suggested order:**

**Contribution A — Exact horizon and reachability.** The scalar theorem gives a necessary-and-sufficient support horizon under the declared linear moving-target/adaptation assumptions. A separate ordinary Euclidean PG example and a 16-outcome two-step finite model demonstrate the pointwise versus reachable trajectory distinction. Call the coupled finite all-controls result computational/numerical unless F0 verifies the necessary rigorous interval bounds.

**Contribution B — Information, adaptation and certification are distinct.** Fixed-N BTL comparisons can reduce the uncertainty compatible with observed preferences and improve a certificate on the SAME unchanged trajectory without altering actual true return. With a bounded controller, different compatible sets can change guidance choices; but that may be insufficient to change policy progress. Preserve the sample-gradient caveat that whole-path ex-post certification may be broken before feedback.

**Contribution C — Guidance choice changes realized policy optimization trajectories.** WP05/06 real tabular DST results and same-controller forks show an interpretable long-run short- vs deeper-treasure path effect, while the one-step full-information oracle wins its own **same-state one-step objective**. These are nonconvex finite-training effects, not evidence that ignorance of the true reward is universally better.

**Contribution D — Authentic preferences and real-model audits.** The 0.5B human TL;DR preference-fit ladder and corrected bounded Pythia-1B runs extend the measurement to pretrained models. The neural RM evidence remains **empirical**, and only valid independent fixed-evaluator policy gain may be called a downstream benefit. If the bounded neural contrast fails, write the more limited result: a real-human RM can learn modest ranking but a finite functional movement budget and model-specific feedback can prevent reliable improvement.

**Contribution E — Open, reuse-oriented evaluation infrastructure.** Release a small tool that independently recomputes exact finite gain when enough data are provided, detects unsupported neural certificates, checks functional adaptation budgets, and calculates seed-paired differences on supplied run manifests. This is directly reusable outside this paper.

Do not add "universal reward reuse benchmark" or "general optimal feedback allocation algorithm" to the paper unless the final outputs truly establish those claims.

### 8.2 Small and fixed figure/table set

**Paper main figures:** No more than THREE compact new/existing graphical exhibits beyond what the final manuscript actually needs. Preserve the present explanatory Figure 1 and existing analytic figure placements unless the author-supplied manuscript specifically authorizes a change.

1. **Exact support and coupled reachability (one figure or panel):** original scalar T0=5 anchor, finite 16-outcome first loss at .012, required gamma and pointwise max, with the source-only/admissible-control numerical bound labeled correctly. One clean source/updated feedback curve may be an inset, not a separate giant plot.
2. **Same-state information versus long-run policy path (one compact two-panel figure):** early H3/DST same-controller information update versus frozen policy; 160-step DST true return plus short/deeper treasure probability. The legend must distinguish source, updated robust, MLE, ordinary true-w and privileged one-step lookahead, and label exact versus sampled curves clearly.
3. **Human reward evidence and real-model validity (one compact figure):** valid F1 human held-out accuracy/NLL versus labeled training count, and F2 bounded neural RM anchor-movement/independent policy-quality comparison IF truly measured. If no valid F2 independent utility, show the null/blocked bounded-budget observation rather than a fabricated positive trend.

**Main tables, at most two:** one hierarchy of evidence and guarantees (exact/statistical/empirical, with scope) and one multi-scale efficiency/results table including label source, budget, evaluator and uncertainty. Resource Gathering limits and full RewardBench2 category data fit better in the appendix/supplement unless an independently valid new result is particularly important.

**Figure constraints:** Few, legible labels and real units; no screenshots or bloated small-type poster designs. Figures generated from committed CSV/JSON only, with scripts and fixed seeds. Do not reuse an old invalid RewardBench2 macro or violated Pythia b result in a paper graphic except as a clearly labeled historical negative/control.

### 8.3 Manuscript-ready files to WRITE after running

**Deliverables that must exist under the repository:**

- docs/FINAL_MILESTONE_08.md — short executive scientific summary; what is DONE, what is PARTIAL and what is UNSUPPORTED, top 4–5 supported claims, important negatives and exact evidence.
- docs/PAPER_EXPERIMENTS_08.md — polished, compact, copy-ready manuscript experimental section in plain research English; precise setup, metric definitions, baselines, actual data size/seed count, important numerical results and limitations, with references. This should be usable to revise an ICML submission.
- docs/FINAL_CLAIMS_08.md — claim-to-theorem/data/code/figure matrix and individual wording permitted/not permitted. Each numeric sentence links to its source table and Git SHA.
- docs/FINAL_REPRODUCE_08.md — CPU zero-GPU example, human comparison RM training, 1B bounded actor run, external RewardBench2 evaluator, paper figure scripts, frozen environment/model/dataset revisions and expected hashes/commands.
- docs/BENCHMARK_CARD_08.md — actual tools/data/metrics/provenance/license card, who should use it, comparison unit, invalid guarantees and limitations.
- docs/REVIEWER_AUDIT_08.md — concise checklist of what might otherwise cause rejection: non-learning tasks, implicit model oracle access, b violations, changed RM scales, misleading 4B metric, label leakage, repeated-measure CI inflation and non-portable CLI.
- examples/08_guidance_audit/ — a minimal genuine recomputed finite example, invalid-budget neural example, paired CSV/JSON and an external researcher example.
- experiments/horizon_final_08/ — small per-seed/row metrics, frozen manifest/checkpoints, actual resource ledger and figure sources, with large binary adapters/checkpoints handled according to release permissions.
- work_plan/08_Final_Milestone_and_Paper_Freeze.md — fully filled as the authoritative last work plan.

**Manuscript availability constraint:** The actual editable LaTeX manuscript is NOT established as a file in this repository. Do not claim that the paper PDF/submission itself has already been revised if its source is unavailable. Instead create a complete paper-ready Markdown/LaTeX snippet and clear placement instructions in docs/PAPER_EXPERIMENTS_08.md. If the genuine author-supplied manuscript source is available to the executing worker, integrate the approved evidence there without moving established figures or editing unrelated theoretical arguments. Respect the current venue page/format rules rather than guessing them from an old submission's policies.

**Public README:** Add ONE compact link near the main results to the public audit quickstart and benchmark card if the tool truly passes. Keep the existing theoretical identity of the repository, and do not replace the landing page with an enormous experimental report. No unverified arXiv link or invented "accepted at ICML" claim.

### 8.4 Reviewer-level acceptance audit

A skeptical reviewer must be able to verify these:

- What is actually proved? The exact scalar theorem and ordinary-PG result are mathematically separate from numerical finite-MDP reachability and empirical neural tests.
- Does an updated BTL confidence set genuinely have the claimed fixed-design coverage and query dependence? Are sequential feedback checks appropriately caveated?
- Did a label-only information update leave the policy/return unchanged? If not, why?
- Were true preference weights accessed by the deployable guidance controller? Every privileged control must be labeled and share equivalent model/optimizer access when comparing only information value.
- Are the source-only and BTL/new-data/extra-compute branches correctly matched at the same policy checkpoint, including optimizer moments and RNG?
- Did ANY Pythia reward update violate b=.05? A violating run cannot enter a bounded-adaptation main result.
- Were Pythia responses actual model generations with LoRA gradient steps and a meaningful frozen reference KL? Is fixed independent quality measured, not a changing reward head's score?
- Were authentic human TL;DR labels split without overlap, and was the final test not used to tune source/model selection? Are online teacher labels named AI feedback?
- Were 4B external scores obtained with valid RewardBench2 semantics, including Ties, parsing/missingness and an explicit subset/full distinction?
- Can an independent CPU user alter a finite example's policy probabilities and see computed certified gain change, and can the compare CLI calculate actual paired statistics?
- Are failed Walker, Meta-World, DoorKey, instruction benchmark and Resource Gathering results preserved as scoped limitations rather than silently omitted?
- Do all figure and paper numbers correspond exactly to committed machine-readable artifacts and predeclared aggregation?

### 8.5 Finalization decision, made from actual work

The project has THREE possible honest finish statuses. A finish does not mean every hypothetical experiment succeeded.

| Final status | Necessary conditions | What is allowed in paper/release |
|---|---|---|
| **MILESTONE_COMPLETE** | All required F0/F1/F2/F4/F5 tests pass; independent fixed neural evaluation exists; F3 external method is correct at least on verified subset; F6 evidence/docs/figures and reproducible manifest complete | Full supported paper story, with empirical and exact scope labels |
| **MILESTONE_COMPLETE_SCOPED** | Theory/finite/DST + honest human-label RM study and reusable F5 tool pass, but bounded neural policy benefit is null, invalid after bounded repair, or optional 1.5B/full RB2 not feasible; all unsupported neural/external performance claims are removed | A completed scoped scientific contribution, negatives/omissions made explicit. No further automatic experiments |
| **NOT_READY** | Historical false pass persists; formal false certificate remains; benchmark cannot be recomputed; paper contains violated b as valid; human heldout contaminated; or published figures claim unsupported accuracy | Do NOT tag/release as verified. Fix the critical error only, record blocker; do not enlarge scope or begin unrelated Plan09 |

**Release freeze:** Once MILESTONE_COMPLETE or MILESTONE_COMPLETE_SCOPED is documented, freeze config and results, run complete regression on a clean environment, commit to main, record exact full SHA, and create a clearly named version/tag such as milestone-08-verified-YYYYMMDD ONLY if the connector/git worker actually has permission and the final tests passed. If tagging is unavailable, preserve the immutable SHA and a manifest with verified file hashes. Do not declare a release/tag created until actually confirmed.

**Temporary project end:** After milestone handoff there is no open-ended queued experimental stage. The only future work is a separately requested paper review, legitimate bug fix or new scientific project. Do not auto-generate Work Plan 09.

## 9. Result Tables and Actual Runtime Ledger — FILL AFTER EXECUTION

### 9.1 Stage completion

| Stage | NOT RUN / PILOT / COMPLETE / PARTIAL / INVALID / BLOCKED | Actual counts/seeds | Verified main result | Scientific claim added or removed | Source and result path | Real runtime and cost | Limitation / independent audit |
|---|---|---|---|---|---|---|---|
| F0 evidence audit | COMPLETE | reference hash, split, invalid ledger | T0=5; 16-outcome upper stays COMPUTATIONAL EVIDENCE; WP07 B and RB2 macros invalid | removed the vacuous missing-file pass from the new audit | experiments/horizon_final_08/audited_baselines/ | 0.59 s CPU | reproduction_status.json remains ABSENT |
| F1 human RM plus MiniGrid | COMPLETE | 0.5B seeds 3+3+3+3; logistic 3+3; Empty seeds 2 | FINAL n=995 after document repair; more human pairs raise accuracy; logistic matches or exceeds the neural FINAL | removed DoorKey from the Horizon claim; extra epochs are not the 4098 contrast | human_final_holdout/ and minigrid_controls/ | 1507 s | archived final NLL in metrics.json was the last batch; use final_metrics_corrected.json |
| F2 bounded Pythia-1B policy | COMPLETE | A/B/C × seeds 0,1 × 64 updates; E is 16 steps × 1 seed | every applied event ≤ 0.0501; fixed ROUGE-L does not rise | removed any neural support-horizon claim; D NOT RUN | pythia_bounded/ | A/B/C 614.812402587384 s plus follow-up 81.7 s plus pilot 31 s | step-0 file is a reconstruction; numpy RNG was not saved; C did not replay B's texts |
| F3 official/protocol-compatible RewardBench2 | PARTIAL | 502 frozen rows; fixture 8/8 | macro 0.218 on the subset; Ties 0.008 | removed .2091/.1115 as official scores | rb2_corrected/ | 15.3 s | 512-token cap hit 580 completions; not a leaderboard submission |
| F3 optional Qwen1.5B transfer | NOT RUN |  |  | no cross-family claim |  | 0 | does not block the scoped milestone |
| F4 checkpoint feedback vs fitting | COMPLETE | 8 parents × arms K,C,H,A,B,Z,T | AdamW restored; greedy re-decode identical; movement within b; ROUGE deltas vs K are about ±0.005 | no causal quality ranking among the arms | checkpoint_forks/ | 1573 s | parents are steps 16 and 48 only; third policy seed was not run |
| F5 public audit engine and compare CLI | COMPLETE | 9 tests | probabilities change the certificate; false certificates are rejected | the compare command now computes paired deltas | public_smoke/f5_tests.json | CPU unittest | Gym and the 0.5B head remain empirical |
| F6 paper pack and freeze | COMPLETE | 3 figures, 6 docs, these tables | scoped negative neural result plus the audit tool | no PDF claim; no Work Plan 09 | docs/*_08.md and figures/ | this pack | status MILESTONE_COMPLETE_SCOPED |

### 9.2 Scientific evidence sign-off

| Paper claim ID | Exact/statistical/empirical type | Before WP08 historical result | After WP08 measured result (if any) | Confidence / uncertainty | Claim YES / PARTIAL / NO | Precise permitted manuscript wording and file/figure |
|---|---|---|---|---|---|---|
| C1 Exact scalar horizon | analytic exact | T0=5 | recomputed 5.000000000000001 | theorem | YES | The source-only support horizon in the stated scalar model is 5. Figure 1. |
| C2 Pointwise vs coupled trajectory | exact + finite numerical | t=.012, max=.11417, upper=.05993753 | upper left as COMPUTATIONAL EVIDENCE | the outward-rounding proof was not rerun | PARTIAL | A 16-outcome coupled trajectory can lose feasibility at 0.012. The upper 0.05993753 is computational evidence, not a checked proof. |
| C3 Information vs adaptation | fixed BTL + policy controls | WP02–04 unchanged-trajectory certificates | H3 information-only return gap is 0; later guidance paths differ | assumptions stated | YES | On the recorded H3 state, label information alone does not move the frozen policy, while the later guidance choice does. |
| C4 Nonconvex guidance trajectories | empirical tabular | DST short/deeper treasure split, WP06 forks | 160-step means S 0.683, U 0.683, I 0.678, M 0.651, O 0.638 | 8 policy seeds | YES | Under the shared DST controller, true-weight and robust paths separate by update 160. This is not an explanation of the language-model runs. |
| C5 Human preference reward fitting | empirical | .506/.551/.600 on original 512 val | FINAL 995: 257 about 0.51, 1024 about 0.55, 4098 about 0.56; logistic 4098 about 0.58 | document bootstrap, 3 seeds | PARTIAL | More human TL;DR pairs improve development accuracy. On the repaired FINAL split the 0.5B model does not beat a length/overlap logistic. |
| C6 Bounded neural policy learning | empirical, never a general theorem | WP07 original b violation | per-event RMS ≤ 0.0498; paired ROUGE-L B−A is about −0.0017 | N=2 seeds; wide paired interval | NO as a benefit; YES as a valid null | A per-event functional budget of 0.05 can be enforced by weight interpolation. It did not improve held-out ROUGE-L within 64 updates. Cumulative anchor movement still exceeds 0.05. |
| C7 Larger evaluator external protocol | empirical external | WP07 invalid 502 macro | protocol-compatible 502-row macro 0.218; 4B and full 1865 not run | subset only | PARTIAL | The 502-row replica is not an official RewardBench 2 leaderboard score. |
| C8 Public audit engine | finite verified + empirical logs | WP07 reader/empty compare | 9 recomputation tests passed | CPU | YES | Given probabilities and rewards, the tool recomputes the finite certificate and rejects a contradictory stored value. |

### 9.3 Machine budget actuals

| Stage | Planned wall hours | Measured wall hours | GPU seconds and peak VRAM | CPU RAM peak | Actual records / labels / tokens / steps | Intervention/budget violations | Final evidence |
|---|---:|---|---|---|---|---|---|
| F0 | 0.50 | 0.0002 | none | not logged | hash of results/reference | none | audit.json |
| F1 | 1.50 | 0.419 | human confirm 1390 s; MiniGrid is CPU | not logged | 995 FINAL pairs; Empty 100-episode eval | none; the last-batch NLL bug was corrected after the fact | final_metrics_corrected.json |
| F2 | 3.00 | 0.202 | about 28.6 GiB observed on nvidia-smi; not written into the JSON | not logged | 2 seeds × 3 arms × 64 updates; 96 AI pairs on B and C | per-event b held; D not run; numpy RNG null | confirm.json |
| F3 | 1.25 | 0.004 | 15 s | not logged | 502 rows | full 1865 and 4B not run | summary.json |
| F4 | 1.00 | 0.437 | 1573 s; about 29.6 GiB observed | not logged | 56 forks, M=32, 8 policy steps | none of the applied events exceeded b | confirm.json |
| F5 | 1.25 | CPU, seconds | none | not logged | 9 tests | none | f5_tests.json |
| F6 | 1.00 | this pack | none | not logged | 3 figures | none | docs and this plan |
| Reserve | 0.50 | 0 |  |  |  |  | not used to invent a further experiment |
| **TOTAL** | **10.00** | **1.06** |  |  |  |  | scoped completion, not a claim that 10 hours were consumed |

**Do not substitute planned time for measured time.** A shortened run is acceptable if all primary scientific and community gates pass, but leaving key outputs unverified while declaring "10 hours used" is not.

## 10. New Source/Artifact Layout and Reproduction Commands

The following are PROPOSED future paths, not files already present merely because this plan refers to them:

~~~text
work_plan/08_Final_Milestone_and_Paper_Freeze.md
tasks/08_Final_Milestone/__main__.py

experiments/final08/
  00_audit_freeze.py
  01_human_rm_confirm.py
  01_minigrid_control.py
  02_bounded_rm.py
  02_pythia_policy.py
  03_rewardbench2_official.py
  03_qwen15_transfer.py
  04_matched_forks.py
  05_external_smokes.py
  06_paper_tables.py

experiments/guidance_audit/
  schema.py
  finite.py
  empirical.py
  decisions.py
  reporting.py
  __main__.py
  tests/

examples/08_guidance_audit/
  finite_computed.json
  intentionally_false_certificate.json
  mismatched_policy.csv
  paired_seed_example.csv
  quickstart.md

experiments/horizon_final_08/
  freeze_manifest.json
  audited_baselines/
  human_final_holdout/
  minigrid_controls/
  pythia_bounded/
  rb2_corrected/
  qwen15_optional/
  checkpoint_forks/
  public_smoke/
  figures/
  final_summary.json

docs/
  FINAL_MILESTONE_08.md
  PAPER_EXPERIMENTS_08.md
  FINAL_CLAIMS_08.md
  FINAL_REPRODUCE_08.md
  BENCHMARK_CARD_08.md
  REVIEWER_AUDIT_08.md
~~~

All names may be improved during implementation; the requirements are the real executable functionality, explicit source/result paths and consistent CLI. If scripts are placed elsewhere, update THIS plan with actual commands.

**Suggested orchestration:**

~~~bash
python tasks/08_Final_Milestone_and_Paper_Freeze/__main__.py run F0
python tasks/08_Final_Milestone_and_Paper_Freeze/__main__.py run F1 --stage pilot
python tasks/08_Final_Milestone_and_Paper_Freeze/__main__.py run F1 --stage confirm
python tasks/08_Final_Milestone_and_Paper_Freeze/__main__.py run F2 --stage pilot
python tasks/08_Final_Milestone_and_Paper_Freeze/__main__.py run F2 --stage confirm
python tasks/08_Final_Milestone_and_Paper_Freeze/__main__.py run F3 --stage confirm
python tasks/08_Final_Milestone_and_Paper_Freeze/__main__.py run F4
python tasks/08_Final_Milestone_and_Paper_Freeze/__main__.py run F5
python tasks/08_Final_Milestone_and_Paper_Freeze/__main__.py run F6

python -m experiments.guidance_audit validate --manifest examples/08_guidance_audit/finite_computed.json
python -m experiments.guidance_audit audit --manifest examples/08_guidance_audit/finite_computed.json --out experiments/horizon_final_08/public_smoke/
python -m experiments.guidance_audit compare --baseline experiments/horizon_final_08/public_smoke/pythia_source --treatment experiments/horizon_final_08/public_smoke/pythia_bounded_arm --metric fixed_eval --pair-by seed --out experiments/horizon_final_08/public_smoke/paired
~~~

**Critical sequencing:** F0 fixes false tests and freezes evaluation. F1 validates human RM and neural positive control. F2 enforces b and saves real matched policy/optimizer checkpoints. F4 then has the correct parents; it can execute once F2 checkpoints are ready. F3 can run independently after its official rubric/metric is pinned and should not tune on final test rows. F5 code can be prepared alongside CPU jobs but must be audited against actual F2/F4/F3 result schemas before release. F6 only after the science and public tool gates.

## 11. FINAL Agent Execution and Handoff Contract

1. **Read Work Plans 01–07 as historical evidence and the current code.** Verify all required baseline result files exist, create the new F0 frozen evidence manifest and clean the known vacuous reference pass without altering archived numbers.
2. **Run F1 human-label confirmation and learned-policy positive control.** Preserve proper human train/DEV/FINAL document partition, repeat actual 0.5B learned RM seeds and report independent accuracy/NLL beyond length heuristics. Do not overinterpret saturated Empty or unlearned DoorKey.
3. **Make F2 VALID before treating neural results as science.** Implement genuine per-event b=.05 constraint; verify b=0, actual line-search acceptance, full LoRA/AdamW/RNG restoration, reference KL and held-out quality. Run source/bounded/no-deployment arms with independent seeds and full 0/16/32/48/64 checkpoints. Preserve the historical b violation explicitly.
4. **Repair F3 official RewardBench2 score semantics.** Preserve frozen 502 row IDs, run official/validated compatible multiresponse + Ties logic, and report category scores and parsing failure. If extra time permits, 1,865-row full test and a separate honest Qwen1.5B LoRA transfer. Never rebrand an inference-only 4B judge as a trained 4B RM.
5. **Run F4 same-parent causal interventions.** Same actor plus optimizer, same recorded labels/pools, new-human-offline versus new-AI-online provenance, identical policy continuation and independent fixed evaluator; compare reward fitting cost honestly.
6. **Finish F5 real community audit API.** The tool must recompute finite compatible gain from states/rewards, verify budget from actual anchor scores, reject faked certificates, output real paired seed-level statistics and import in a fresh CPU process.
7. **Write F6 paper pack and sign-off.** Fill all tables and statuses in THIS plan. Generate only 2–3 new compact, accurate main figures. Create final paper-ready prose with no invented theorem, proof or human-quality improvement. Audit licenses, test data, seed independence, saved checksums and device costs.
8. **Perform a reproducibility spot-check**: re-run at least one full small exact fixture, one human heldout score batch and one Pythia bounded checkpoint restore; confirm the same numbers from the final frozen data. Link every main result to a commit and script.
9. **Commit and push all permitted source, numeric seed summaries, figures, tests, docs and THIS UPDATED PLAN to main.** Create a release tag only after the acceptance matrix is verified and tool permissions permit. Return the exact last commit, any verified tag, paper-pack link, benchmark quickstart and final scientific YES/PARTIAL/NO ledger.
10. **STOP.** The default project state is frozen after this milestone. Do not schedule a new work plan or offer another open-ended wave of experiments. Outstanding optional Qwen1.5B/4B model training, larger environments or human-on-policy evaluation remain clearly documented limitations, not hidden tasks silently continuing.

**Science above presentation:** A useful finished paper may establish a sharp exact horizon, a finite coupled reachability mechanism, the information-versus-adaptation distinction, nonconvex guidance-path behavior, a real human RM data result, and a modest/open community audit tool. It does NOT need to claim that 96 AI comparisons improved human-summary quality or that every neural reward model has a formally certified support horizon.

**Completion instruction:** This is the LAST planned empirical expansion in the current project. A defensible scoped negative result plus a reliable open audit tool is an acceptable scientific final contribution. The user asked for a milestone and temporary finalization, not perpetual benchmark accumulation.
