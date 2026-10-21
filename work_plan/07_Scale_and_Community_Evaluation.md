# 07 Scale and Community Evaluation

**Project:** The Limits of Reward Guidance under Finite Adaptation  
**Research target:** ICML 2027  
**Prepared:** 2026-10-19  
**Status:** PLAN ONLY. S0–S6 have not run. The counts and wall-time allocations below are targets, not measured results.  
**Verified starting commit:** 520b9affa92435d347a213a78440a34482692aef  
**Execution target:** approximately TEN HOURS on the existing single workstation, after a throughput and VRAM smoke test  
**Main deliverable:** THIS SAME completed Markdown with the real experiment results, plus an independently useful public reward-guidance evaluation package and a reproducible benchmark manifest.

> **Executing research agent:** Run S0–S6 in scientific order. The user has authorized a larger approximately 10-hour experiment with realistic task and model sizes, not a set of toy synthetic diagrams. You may independently fix routine engineering issues, select batch/microbatch size from measured throughput and VRAM, save and resume long jobs, and use the existing GPU. Do not request repetitive approvals. **Never silently shrink a planned result, invent a number, tune a held-out test, or describe a numerical/empirical diagnostic as a formal theorem.** For expensive experiments use the predeclared gate: first pass the positive controls, then spend the GPU time. Fill the result tables IN THIS FILE after running and push code and small results to main. The desired result is a useful scientific and community artifact even if empirical improvement is absent.

## 1. Why Work Plan 07 Is Different

### 1.1 What Work Plans 01–06 already established

The paper's original mathematical contribution is a support horizon for preference-guided policy improvement under limited reward adaptation, together with separation of pointwise reward feasibility, reachable reward trajectories and the information available from preference comparisons. Do not edit the theoretical claims.

The completed experimental chain is:

| Evidence stage | Established fact from committed work | Limitation that remains |
|---|---|---|
| WP01 original E1–E8 | Real RL and open-model pilots exposed nonlearning environments and negligible LLM policy effects | Large benchmark breadth without a learnable control made many results inconclusive |
| WP02–03 | Exact five/four-output controls, finite PG, real BTL batches, stochastic ex-post gain; a 16-outcome two-step MDP lost scheduled certification at t=.012 while pointwise gain was about .114 | Theorem-level statements remain restricted to specified finite constructions |
| WP04 | 16-outcome coupled reachability produced an all-controls numerical bound at t=.012; 64-outcome three-stage model and exact tabular RL were implemented | The global finite bound was specific to that model, not a neural policy |
| WP05 | Real live Deep-Sea-Treasure with 160 updates and corrected BTL; new feedback improved information but did not consistently improve return | Comparison of label information and controller access needed adjustment |
| WP06 | Matched one-step lookahead, policy-path replay and 2D preference identification; verified nonconvex path dependence and conservatism of full-simplex robust guidance | The community cannot yet run the analysis on its own Gym/LLM checkpoints; neural scale and human-labeled data evidence are weak |

**Concrete WP06 facts:** in DST, the fixed true-reward ordinary PG ended near 0.638 after 160 updates, while source-only guidance ended near 0.683. Exact trajectory replay showed that the former took the nearest small-treasure basin and the latter retained probability mass on deeper treasures. The same-policy-state privileged one-step oracle was never beaten at that particular state, but this does NOT imply its entire long-run trajectory is globally optimal. In three-objective Resource Gathering, source-only remained near 0.227, point-fit baselines varied widely, full-simplex robust nearly stopped learning and privileged lookahead reached about 0.755. More Fisher information did not guarantee the best weight estimate.

### 1.2 The next three scientific questions

**Q1. How long can a learned preference/reward model be reused as policy distributions change?** Measure held-out preference ranking, calibration and realized downstream gains at multiple real policy checkpoints. Quantify when these disagree. This is a practical model-monitoring question, NOT a claim that every neural reward model has a formally certified support horizon.

**Q2. When should a researcher spend budget on additional preference pairs rather than more optimization of the SAME reward labels?** Use matched checkpoints, identical candidate pools, comparable training compute, exact knowledge of what the labels represent, and a learning-positive reference. Return a descriptive or validated decision rule, or a measured failure of the rule.

**Q3. Do observations survive a gradual increase in model, data and trajectory complexity?** Start with already verified finite models, move to a standard neural-policy Gym environment, then authentic released human preference comparisons, pretrained 0.5B–1.5B policy/reward models, and finally a 4B-scale independent evaluation. A 4B frozen evaluator is NOT a 4B trained policy.

### 1.3 Direct contributions to the community (required, not optional decorations)

The intended public artifacts must make this useful without the paper:

1. **Reusable Python evaluation API and CLI** that accepts policy/reward checkpoints, dated preference batches and logged update budgets; returns reproducible original/updated preference metrics, policy improvement trajectories, first unsupported time, label/model-compute cost and warning flags about invalid guarantees.
2. **Standardized audit manifest + benchmark records** for at least one small exact model, one actual Gym neural-policy task and one public preference-model or LLM task. Include all experiment settings, source/version provenance, label source, independent evaluation, raw seed-level records and an explicit guarantee type.
3. **A five-minute quickstart** that works on CPU for the exact/Gym reference, plus a single-GPU example for a real pretrained model. Publish run commands, a compact JSON output example, and a model/dataset scorecard.
4. **Meaningful comparisons rather than a leaderboard of meaningless numbers:** correct in-domain held-out preference likelihood, cross-domain human-preference performance, empirical policy utility, and cost-to-extend-reuse; clearly mark unavailable measurements.

**Release destination:** the existing repository and docs first. Hugging Face dataset/Space publishing is a separate OPTIONAL final action only if user-authorized credentials/permissions and dataset reuse rights are present. Do not assert that a Hub upload or external promotion happened merely because local files were prepared.

### 1.4 Scope of guarantees, stated BEFORE collecting numbers

There are three distinct meanings of a success report. The result JSON MUST name exactly one:

| Scope name in manifest | Meaning | Valid statement | Invalid substitution |
|---|---|---|---|
| exact_finite | Full known discrete model, exact policy step, fully specified source set; numerical/theoretical conditions independently checked | Exact gain, policy-specific contact; all-controls only with a valid reachability bound | Do not extend the original theorem to another environment automatically |
| statistical_fixed_design | Fixed-N independent Bradley–Terry labels, predefined comparisons, model/parameter assumptions satisfied; correctly constructed confidence region | Declared fixed-batch coverage and worst compatible gain for that parameter class | No sequentially valid 95% entire-path claim from repeated uncorrected 95% intervals |
| empirical_proxy | Neural network rewards, human pair annotations with unknown noise process, or model/feature misspecification | Held-out preference ranking, calibrated empirical uncertainty, realized gain with honest error bars | Never label OOD RM accuracy or empirical return as a rigorous neural reward support horizon |

For neural reward models, implement a **first empirical loss of a specified benchmark criterion** only when predeclared (e.g., held-out pairwise accuracy falls below a frozen reference level or a paired policy update ceases to show independent utility improvement); label censored/no crossing appropriately. Do not manufacture a gamma crossing by selecting the threshold from final checkpoints.

## 2. Real-Scale Ladder and Resource Contract

### 2.1 Sizes planned, each with a reference and an information/control contrast

| Level | Actual environment/data/model | Training/evaluation scale target | What genuinely changes? | Required positive/reference control |
|---|---|---|---|---|
| L0 exact | Existing four-output, two-stage (16 outcomes), three-stage (64 outcomes), DST/RG exact tabular models | Archived exact traces and 20/100/160 update checkpoints | New public audit adapter only; no original experiment rerun | Existing deterministic results within tolerance |
| L1 standard RL | Farama MiniGrid Empty-8x8 and DoorKey-6x6 with a neural policy | 40k-step pilot then 100k–200k environment steps per selected task; 2–3 policy seeds and matched guidance forks | Learned features, stochastic environment interaction and longer trajectories | Identically implemented true-objective PPO/REINFORCE must learn before horizon interpretations |
| L2 authentic preference data | OpenAI human-labeled TL;DR summary comparisons; AI-labeled UltraFeedback is a SEPARATE data track | Fixed source 256, additional 768/3,840 for human data totals 1,024/4,096; 1,000 held-out human preference comparisons where available | Real, historically collected human labels rather than fabricated BTL feedback | Bounded/LoRA reward learner beats chance or shows a correctly diagnosed failure |
| L3 pretrained policy updates | Pythia-1B TL;DR SFT policy and Qwen2.5-1.5B-Instruct second task/model where throughput permits | Two policy/reward seeds; 48–96 actual LoRA policy updates per main condition; at least 256 prompt identities and fixed evaluation prompts | Actual model weight updates, response distributions and RM reuse drift | Measurable adapter/held-out change, bounded policy KL, nondegenerate generated responses |
| L4 larger model | Qwen3-4B-Instruct-2507 frozen scorer/judge on external preference or verification data, with optional 4B LoRA smoke | 384–600 balanced external evaluation rows initially, up to all 1,865 RewardBench 2 test rows if measured throughput permits; optional small 4B adapter | A genuinely larger model and external response distribution | Correct RewardBench 2 multi-response/ties scoring; do not call it a trained 4B RM unless trained |

The table describes a **scale ladder**, not five equally large full-factorial benchmark sweeps. Under a ten-hour single-GPU target, guarantee science/reproducibility quality first. Extra 4B training and full RewardBench 2 are explicitly capacity-gated; if not run, record "NOT RUN at 10-hour limit" with measured extrapolated cost, not a fictional completion.

### 2.2 Actual hardware and spending target

The last recorded worker had **one RTX 5090 with about 32 GB VRAM, 12 CPU cores, 62 GB RAM**, and a local Hugging Face cache for Pythia-1B TL;DR and Qwen models. Verify device, free VRAM, model cache SHAs, storage and throughput AGAIN in the execution environment. These are historical machine details, not live resource availability guarantees.

**Ten-hour wall-clock budget including model loads, downloads, experiment execution and report generation:**

| Stage | Target wall time | Primary compute | Planned exit product |
|---|---:|---|---|
| S0 audit schema, load checks, scale preflight | 0.50 h | CPU + short GPU load | Working CLI, verified dependency/model versions, throughput ledger |
| S1 small-to-neural Gym reference | 0.75 h | CPU, GPU only if faster | One learnable real neural-policy baseline + matched policy forks |
| S2 public human/AI preference data + small RM | 1.25 h | CPU preprocessing, single-GPU 0.5B scoring/LoRA | Human-labeled reward-fit and fixed holdout, manifest for all dataset sources |
| S3 real 1B–1.5B policy/reward LoRA updates | 3.00 h | Single GPU main experiment | 2 seeds, meaningful multiple policy updates, same-checkpoint reward/data interventions |
| S4 4B external scale evaluation | 1.50 h | Single GPU batched scorer | Disjoint independent-size scorecard + domain breakdown |
| S5 matched budgets and reuse diagnosis | 1.25 h | Cached scorings + limited extra GPU | Frozen-checkpoint label-versus-compute decision table, confidence/validity tags |
| S6 community packaging, reproduction and reports | 1.25 h | CPU | API tests, quickstart, benchmark card, result tables, license/provenance, commit |
| Reserve for real OOM/reload/precision issues | 0.50 h | Contingency | Explicit actual-vs-planned time audit |
| **TOTAL** | **10.00 h** | one machine | Publicly reusable result, even if secondary scale is censored |

**Budget logic:** S0 and S6 are ESSENTIAL. S2's held-out human evaluation is ESSENTIAL. S3 must demonstrate real model policy updates; if it cannot pass an objective learnability check, report the failure, complete reward-head evaluations, and redirect the remaining GPU time to measurable S4 rather than rerunning broken update steps. Do not shorten independent evaluation, hide negative seeds, replace human labels with AI labels unannounced, or spend the whole GPU session downloading unrelated model weights.

### 2.3 Predeclared throughput/VRAM gates (measure, then freeze)

Before committing to any large loop, measure one train step and one evaluation step with the actual model loaded. Record batch size, input tokens, generated tokens, GPU peak allocated/reserved memory, wall seconds per step, device version, quantization mode and whether checkpoint save/reload agrees on a test batch.

- Start with BF16, LoRA rank 8 or 16 and gradient checkpointing. Quantized 4-bit adapters are allowed only if explicitly labeled and verified with the current CUDA/Transformers/PEFT versions. Do not assume an arbitrary full-precision 4B policy + several 4B reward heads can fit together in 32 GB.
- Actor/reward models may be scored serially with cached tokenized responses, explicit unload/reload and stable scoring caches. If a 4B evaluator alone exceeds the memory/time budget, perform the same predeclared external evaluation with the smaller scorer and record the 4B arm as capacity-censored.
- Freeze the primary seed/model/batch/length and prompt split after the measured development pilot; do not adjust based on held-out RewardBench 2 accuracy or late true-return scores.
- Use a GPU quota and a separate token/label budget. A method with more reward-training steps is NOT compute-matched to a method with more actual feedback labels unless actual model token-FLOP/seconds are also reported. Make matched-label and matched-compute comparisons separately.

## 3. S0 — Audit API, Provenance and Small-Model Regression

### 3.1 Public interface that researchers can actually run

Create a small importable subpackage, proposed path experiments/guidance_audit/, with the following functional interfaces. Names may be adjusted, but capabilities must be preserved:

- load_manifest(path) -> validated versioned task/method/checkpoint/pair records, with descriptive errors for missing labels, invalid checkpoint ordering and mismatched tokenizer/reward norms;
- audit_fixed_finite(checkpoints, compatible_reward_set, step_rule, movement_budget) -> original/updated compatible gains, first observed loss, censored flag, pointwise and path feasibility where truly computable;
- audit_empirical(checkpoints, preference_batches, heldout_pairs, policy_evaluations) -> held-out BTL scores, calibration, reward-model drift, actual utility differences, budget/label costs and explicit "not certified" statuses;
- compare_interventions(parent_checkpoint, same_data_extra_train, new_pairs_same_train, both, reference) -> per-seed paired results with one variable changed at a time;
- summarize_runs(manifest_or_folder) -> stable human-readable CSV, JSON and a small plot with named metrics.

**Manifest minimum fields:** manifest schema version, task ID, paper-relevant purpose, model family/id/revision/tokenizer SHA, dataset ID/revision/origin/license and original sample indices or hashes, label provenance ("human", "model judge", "synthetic BTL", "verifier"), data split, exact vs empirical evaluator, policy algorithm, checkpoint update and RNG, reward model checkpoint and calibration, source/new pair IDs, pair timing, queried label count, actually spent reward steps, model/device/token counts, guidance phi or functional score parameters, adaptation budget units, measured true return/verifier utility, uncertainty/coverage method and failure/status flag.

**Critical design principle:** accept logs from OTHER researchers. Do not hard-code theta=two Bernoulli means or require access to the hidden true reward. If a required return or oracle-compatible set is unavailable, the API must return an honest not_available/empirical_only field, not make up a certificate.

### 3.2 Required unit and regression tests

1. Existing support_horizon, interval_feasible and fixed-N BTL tests. Reproduce the exact scalar T0=5 and original ordinary-PG t≈0.007836 references; preserve results/reference.
2. Saved 16-outcome/64-outcome traces as read-only fixtures. Check source vs updated gain on identical frozen policy states: label update cannot alter true return when the policy does not change.
3. A four-output sampled step: population-exact and realized sampled gains must be logged under DIFFERENT field names; no guarantee leakage.
4. Confidence coverage testing for fixed designed BTL; adaptive preference data must flag missing time-uniform adjustment. An invalid/empty confidence set cannot silently become a certified source set.
5. No hidden teacher parameter or evaluator target allowed in policy guidance or active query code; test by stubbing teacher access and verifying execution still works.
6. JSON schema and CLI smoke tests against a small artificial user-supplied checkpoint file and pairwise label CSV.
7. Show at least one correctly **censored** no-loss trajectory and one intentionally invalid model/label config that fails with a helpful error.

**S0 results table:**

| Test / interface | Small source or fixture | Expected check | Observed | Status | Code/test path |
|---|---|---|---|---|---|
| Scalar exact horizon | horizon.py / 5-output fixture | T0=5 | 5.000000000000001 | PASS | experiments/guidance_audit/finite.py |
| Ordinary PG contact | fixed four-output fixture | around .007836 | 0.007835874324472896 | PASS | experiments/large_scale/07_preflight.py |
| Fixed trajectory, more labels | fixed preference pair batch | certification changes, true return not | source 0.68 to updated 0.936; true return stays 0.5 | PASS | experiments/guidance_audit/finite.py |
| Stochastic versus population | recorded checkpoint fields | values are distinct | population_gain 0.2, sampled_gain 0.05 | PASS | experiments/horizon_scale_07/preflight/checks.json |
| Public non-privileged checkpoint | examples/07_guidance_audit/finite_example.json | schema and CLI work | exact_finite accepted; missing file raises ManifestError | PASS | experiments/guidance_audit/__main__.py |
| Honest empirical-only status | neural RM scores without valid BTL model | no formal horizon claim | formal_horizon not_certified | PASS | experiments/guidance_audit/empirical.py |

The plan's `outputs/horizon_scale_07/` paths are stored at `experiments/horizon_scale_07/` with the same subdirectory names. S0 also checked a censored trajectory, an invalid manifest, and a teacher-blind query. Wall time 0.28 s, CPU only.

## 4. S1 — Actual Neural-Policy RL Beyond Exact Tabular Dynamics

**Research purpose:** The community should be able to monitor reward guidance with a stochastic neural policy on a well-known task, without assuming access to a fully enumerable transition model.

### 4.1 Real tasks and reward information

Use official Farama MiniGrid, starting at MiniGrid-Empty-8x8-v0. Progress to MiniGrid-DoorKey-6x6-v0 only after the oracle training checkpoint demonstrates actual learning; these are distinct levels of exploration difficulty, not changes to the theorem. Use a single explicitly frozen observation wrapper across every algorithm branch (e.g. ImgObsWrapper and a compact CNN, or a full-observation wrapper if it is scientifically justified), and record partial observability.

Create a transparent, externally evaluated **two-component outcome** from real trajectory events:

\[
f(\tau)=(\mathbf{1}\{\text{goal reached}\},\ -\,\text{steps}/H)
\]

where the second entry is negative cost, H is the declared fixed environment maximum length. For DoorKey, report optional door-opening/key-acquisition success separately as task diagnostics, NOT an unannounced hidden third preference dimension. Note the built-in shaped reward already includes a time discount; keep the new vector objective separate from the underlying env reward and apply uniformly.

Set a fixed, evaluator-only synthetic BTL teacher on the normalized two-component vector, beta and weight PREDECLARED on development data; label this synthetic preference, NOT human. The policy only gets labels and the currently deployed guidance weight. Keep the known teacher weight for positive control and independent evaluation.

### 4.2 Exactly the transition to learnable neural RL

1. Run 40k environment-step development with a policy optimizer known to learn MiniGrid (a small CNN actor-critic/PPO, or a validated published starter implementation with frozen hyperparameters). Record mean success and success-count uncertainty on held-out env seeds.
2. If the true-weight oracle cannot improve above the untrained policy on Empty-8x8, diagnose task/observation/optimization first. Do NOT interpret source certificate or reward feedback. Skip expensive variants on DoorKey until a learnability control works.
3. Once learning is established, continue to 100k–200k env steps for the selected task, **2 policy seeds** as primary and third if measured throughput permits. On each seed compare from the SAME saved policy checkpoint: source-only learned preference, 32 new informative BTL pairs, extra BTL optimization on the same source pairs, and the privileged true-weight control.
4. All preference-pair candidates come from actual environment trajectories: maintain a comparable candidate pool per shared checkpoint, use uniform vs Fisher-information selection with same label count, and never read the hidden teacher in query selection.
5. Store policy/network, optimizer and RNG checkpoints at 0,25%,50%,75%,100% of the selected training budget. Evaluate on at least 100 deterministic validation episode seeds per checkpoint, report success rate, trajectory length, empirical true two-component expected utility and sampled confidence intervals. Count new labels and policy/environment steps separately.
6. At the same saved checkpoint, do a frozen-policy information-only test (cert/uncertainty without any new policy update), then one policy update with controlled guidance choice. Use an empirical intervention unless the fixed-feature score model and fixed-design confidence conditions actually justify a statistical statement.
7. Report if larger scale changes which bottleneck matters: source uncertainty, adaptation capacity, policy network optimization or environment exploration. Do not force an expiring support horizon if the source set is already informative or if the policy stops learning.

**S1 table:**

| Environment / observation | Policy size, env steps | Seed count | Method and label budget | Held-out goal success | Steps-to-success | Evaluator true utility | Original/updated diagnostic and scope | Notes |
|---|---|---:|---|---|---|---|---|---|
| Empty-8x8, learned reference | 40,192 steps | 1 development seed | true-weight oracle | 0.00 untrained, 1.00 after | 256 untrained, 11 after | -0.50 untrained, 0.479 after | empirical | 100 eval episodes. Partial 7x7 image. |
| Empty-8x8, real learning | 8,192 shared + 92,160 fork | 3 | source-only, 32 new BTL, extra fitting, oracle | source/new/oracle 1.00 on 3 seeds; extra-fit 1.00, 1.00, 0.00 | source/new/oracle 11; extra-fit 11, 14, 256 | source/new/oracle 0.479; extra-fit 0.479, 0.473, -0.50 | empirical_proxy | Synthetic BTL, not human. Extra optimizer steps moved one seed off the goal. |
| DoorKey-6x6 if learned control works | 40,192 oracle only | 1 | true-weight oracle | 0.00 before and after | 360 before and after | -0.50 both | empirical | Not learned, so no guidance interpretation and no DoorKey forks. |
| Shared-checkpoint information-only | same actor, weight overwritten, no update | 12 paired checks | unchanged policy | identical | identical | identical | empirical, not a certificate | All 12 checks matched. |

**Positive and negative findings:** a well-trained neural policy may benefit from informative feedback, or may be unaffected because source information already suffices, or may require reward adaptation beyond the existing budget. All are reportable IF learning and evaluation have passed. A DoorKey failure with a nonlearning oracle is a benchmark/optimizer limitation, not a claimed refutation of the guidance-horizon theory.


## 5. S2 — Authentic Preference Comparisons and a Trained 0.5B Reward Model

**Research purpose:** The strongest gap in Work Plans 01–06 is that almost all preference-learning evidence uses synthetic Bradley–Terry labels or frozen AI-model feedback. This stage must demonstrate that the public tooling works with released REAL HUMAN preferences on natural language, then stress-test model reuse and cross-domain generalization without fabricating online human annotations.

### 5.1 Public sources and exactly what their labels mean

**Primary authentic human preference track:** [openai/summarize_from_feedback](https://huggingface.co/datasets/openai/summarize_from_feedback), the comparisons configuration. Its public comparisons have train and validation splits in which annotators choose between summaries. The axis Likert scores are a DISTINCT measurement type, not binary preference labels. Use the train comparisons for fitting and train-revealed feedback, validation comparisons as untouched held-out human labels. Verify the actual schema, source subdomain (Reddit vs CNN/DM, etc.) and license/terms on the worker; log real sample counts and any exclusions.

**Secondary general-assistant preference track:** [trl-lib/ultrafeedback_binarized](https://huggingface.co/datasets/trl-lib/ultrafeedback_binarized), with 62,135 listed train and 1,000 listed test examples. These are released binarized model-judged feedback, **NOT newly collected human comparisons**. They support realistic broad-domain reward fitting and low-cost cross-domain model checks but must be tagged AI/generated-preference source, with its upstream provenance checked. Do not combine its accuracy with human-label accuracy as though the annotation process were identical.

**Independent external human benchmark:** [allenai/reward-bench-2](https://huggingface.co/datasets/allenai/reward-bench-2), a held-out test dataset with approximately 1,865 prompts and six categories (Factuality, Precise Instruction Following, Math, Safety, Focus, Ties). It has chosen and rejected RESPONSE LISTS, typically best-of-four; the Ties subset needs its special ranking/margin rules. Use the official evaluation semantics and preferably its official scoring script. RewardBench 2 is TEST-ONLY here, never for model selection or reward fitting. A stratified subset is NOT a full official RewardBench 2 leaderboard number.

**Dataset licensing requirement:** Use published dataset IDs/revisions and sample indices/hashes in reproducible manifests. Do not indiscriminately copy/release copyrighted posts, human annotator identifiers, model-generated third-party completions or the full RewardBench 2 data under the project's own license. Release loaders/references and derived aggregate metrics only until rights have been checked. No personally identifying human annotator fields in public per-pair data.

### 5.2 Freeze the preference-data ladder before fitting

For the human summarization comparison track:

| Scale | Human source pairs used to fit RM | Newly revealed EXISTING human pairs | Evaluation | What it tests |
|---|---:|---:|---|---|
| Small | 256 | 0 | fixed 1,000 validation pairs (if available after filters) | Basic real-human-label RM fit |
| Medium | 256 | +768 from unused TRAIN pool, total 1,024 | SAME frozen validation pairs | Information increase, no new human labeling claim |
| Large | 256 | +3,840 from unused TRAIN pool, total 4,096 | SAME frozen validation pairs | Data scale and reward-model reuse |
| Compute-only | 256 | 0 | SAME frozen validation pairs | More optimizer steps on identical data |
| Uniform-query control | 256 | same number of new pairs | SAME frozen validation pairs | Fair random reveal from the same labeled-but-hidden TRAIN candidate pool |

**Predeclared leakage controls:** Group documents/prompts by normalized source text and original comparison ID; ensure none of the exact chosen/rejected completions or document groups are on both sides. Preserve the original dataset train/validation separation, remove near duplicates conservatively, and report how many rows survived. The public data already contain labels, but the query selector must hide a candidate's label until its predeclared query is chosen. Do not describe these as live newly annotated human comparisons; they are **simulated information acquisition from a historically human-labeled pool**.

For the AI UltraFeedback track, a development-scale 1,024/4,096 preference-pair fit may be run only after the primary human track works. Keep its test independent and use RewardBench 2 solely for external final ranking diagnostics. If doing both tracks in S2 exceeds measured quota, prioritize the human track and complete the open-source loader for the other.

### 5.3 Actual 0.5B learned reward, not just one logit score

**Model:** Qwen2.5-0.5B-Instruct (Apache-2.0 model card) with a trained sequence-level scalar reward head. Use its real pretrained weights. If scoring of long TL;DR prompts exceeds S2's GPU allocation, truncate at a PREDECLARED token limit based on development length histograms, record fraction truncated and use the identical treatment in every method. A frozen-feature linear/logistic BTL head is the cheap positive control; the actual 0.5B LoRA scalar RM is the main medium-scale result.

1. Load tokenizer, model and precise pinned revision. Implement reward score(summarization prompt, candidate summary), including masking and a stable response-ending score token/head definition. Do not accidentally score prompt padding or treat full-sequence log-probability as a learned reward classifier.
2. Fit a scalar BTL pairwise ranking loss on 256 train human comparisons, then from the SAME INITIAL base checkpoint fit separate 1,024 and 4,096 conditions. Compare fully converged fitting on the 256 source only, and 768/3,840 new human labels. Hold optimizer steps/data passes and GPU tokens visibly separate.
3. Use a genuinely trainable LoRA reward-model adapter or scalar score head; check nonzero gradient, head initialization, train/held-out leakage, monotonic loss improvement and reward score scale calibration. Preserve the old WP05 fixed-step BTL failure as a regression test; do not use an underoptimized head as the main baseline.
4. Report **held-out HUMAN comparison accuracy**, negative log-likelihood, Brier, ECE/calibration, pairwise margin distribution, reward-score collapse, length/prompt-truncation correlations and worst-subgroup measurements when metadata permit. A flat accuracy score on easy pairs is not a sufficient evaluation.
5. Score a fixed pool of reward-model anchor comparisons from the source and held-out groups before/after each fitting stage; save score differences and a score-normalized functional drift statistic. Center and calibrate rewards on a frozen anchor set so movement units are comparable across checkpoints.
6. At a held-out human-label checkpoint, compare extra training with SAME 256 pairs to new TRAIN pairs with equal total reward optimizer wall time or tokens; list any imbalance. New data will normally require additional token passes; record them instead of asserting cost-matching that did not happen.
7. Optional general assistant track: train a separate 0.5B reward adapter on 1,024 then 4,096 UltraFeedback train pairs, and score a predeclared 600-row balanced RewardBench 2 subset WITHOUT training on RewardBench 2. The evaluator must follow its multiresponse/Ties protocol, rather than reducing each example to one arbitrary chosen-rejected pair.
8. Avoid raw-text rehosting in benchmark outputs until source licensing permits. Use dataset ID, revision, stable example index/hash and saved numerical scores.

### 5.4 Small-model result tables

**Human preference model:**

| Model / reward training condition | Human TRAIN pairs | GPU reward steps / tokens | Independent fitting seeds | Human validation N | Human pair ranking / NLL / Brier | Score drift on fixed anchors | Prompt truncation rate | Notes |
|---|---:|---|---:|---:|---|---|---|---|
| Fixed-feature logistic BTL | 257 | CPU, 20 passes | 2 | 2048 | seed0 0.564 / 0.685 / 0.246; seed1 0.567 / 0.681 / 0.244 | not a neural score | 0, features are lengths | length and overlap only; human labels |
| Actual Qwen-0.5B reward head, source | 257 | 1,028 GPU steps | 2 | 512 | 0.506 and 0.494 / 1.044 and 1.077 / 0.337 | length corr 0.117 | 0.002 | human train, valid1 held out; 256 does not beat chance |
| 0.5B reward head, same 256 labels extra fitting | 257 | 4,112 GPU steps | 2 | 512 | 0.502 and 0.496 / 1.278 and 1.270 | length corr not the gain | 0.002 | more steps on the same labels do not help |
| 0.5B reward head, medium human | 1,024 | 2,048 GPU steps | 1 | 512 | 0.551 / 0.836 / 0.286 | length corr 0.196 | 0.002 | nested reveal from the train pool |
| 0.5B reward head, large human | 4,098 | 4,098 GPU steps | 1 | 512 | 0.600 / 0.680 / 0.238 | length corr 0.318 | 0.002 | nested reveal; empirical only |
| Same-pool uniform label reveal | 1,024 | 2,068 GPU steps | 1 | 512 | 0.543 / 0.911 | similar to nested 1,024 | 0.002 | separate document draw from the same train pool |
| General-assistant UltraFeedback/RewardBench 2, if run | 1,024/4,096 AI pairs |  | 1 | stratified 600 external | proper multiresponse metric |  |  | separate domain and label source |

**Important limitations:** Human comparisons are static historical data and may not describe the current on-policy response distribution. A neural reward head's empirical confidence/margin is NOT a fixed-N BTL compatible-reward proof without an adequate, separately validated model and sampling assumptions.

## 6. S3 — Actual 1B–1.5B Policy Training and Reward Reuse

**Research purpose:** This is the main GPU experiment. The original E6/E7 large-language pilots used just a few prompts and ONE LoRA policy step. S3 must execute many actual optimizer steps over a realistic response pool, with enough checkpoint history to measure whether source reward information remains useful as the policy changes.

### 6.1 Models and task assignment

**Primary policy:** the existing cached TL;DR SFT policy, cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr, about 1B parameters. It has a task-aligned public SFT starting point, previously exercised in experiments/real/language/e7.py. Train a fresh LoRA adapter from an explicitly saved initial state. Do NOT overwrite the cached base or E7 history.

**Training reward:** the trained human-comparison Qwen-0.5B scalar reward from S2, calibrated on a frozen source anchor pool. When new on-policy comparisons are required, the ONLY automated teacher allowed in this ten-hour run is a FROZEN independent reward/judge, e.g. cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr, previously used in E7, and its labels are explicitly **AI reward-model feedback**, not human labels. The teacher must NEVER be called the human ground truth. It may have training overlap with the same historical human corpus; document the overlap risk. Use an independent human validation set only for offline reward-evaluation statements.

**Second policy/model family:** Qwen2.5-1.5B-Instruct, previously loaded successfully in E6. On a disjoint verifiable-instruction task it may be run after the 1B primary result and S4 allocation are secure, with 1 predeclared seed and 32–48 actual LoRA policy updates, then 2 seeds if measured throughput permits. Its synthetic verifier-based BTL labels and true scripted utility MUST be distinguished from authentic TL;DR human comparisons. This is a meaningful different-family / 1.5B generalization probe, not a replacement for the real human track.

**Scope control:** The open-source SFT and RM are separate models. One GPU may load them serially or concurrently only after measuring VRAM. No vLLM assumption, no automatic package upgrade that breaks the existing E6/E7 code. Use the already pinned transformers/peft stack where compatible.

### 6.2 Policy training contract

For the Pythia-1B primary task:

- Use at least 256 distinct train prompt/document identities where preprocessing permits, plus a separate frozen 128-prompt validation subset never shown to the policy optimizer. Split by document/post ID and normalized content hash, not just response row.
- Run an actual on-policy or periodically refreshed response-level policy-gradient LoRA algorithm, e.g. REINFORCE with a centered baseline and KL-to-frozen-SFT constraint, or a properly tested TRL PPO-style trainer. Do not relabel DPO as explicit reward-guidance learning; DPO is an optional separate algorithmic baseline only if actually executed.
- Plan **64 policy updates**, at least 8 distinct train prompts per update, two sampled candidate summaries/prompt where feasible, maximum new response length predeclared after checking tokenizer and GPU throughput (initial target 96–128 tokens). At the same checkpoints, log generated tokens, optimizer steps, LoRA gradient norm, reference KL, output length, truncation/degenerate output fraction and reward scores.
- At least **two independent policy + label RNG seeds** for the TWO principal methods. Do not count each sampled completion/checkpoint as an independent seed.
- Method A = source-only reward: initial human-trained RM, no new online pairs, same policy update count.
- Method B = online reward refresh: same source RM, fixed feedback opportunities immediately before policy updates 16, 32, 48; select 32 new candidate pairs each time from ACTUAL current-policy outputs, score them with the FROZEN AI teacher, then update/replace the deployed RM using the bounded protocol below. Total new AI-labeled pairs = 96. Query selection has no access to hidden teacher score until after candidate IDs are selected.
- Optional method C = extra RM optimization on the SAME historical human pairs with no new labels, used as a same-data compute control. Prefer a matched-checkpoint fork instead of a full third 64-update run if GPU time is tight.
- Optional targeted-vs-uniform comparison: at one checkpoint construct exactly the same 128 candidate pairs for both selectors before revealing the teacher scores. Compare data efficiency/BTL fit; then run only the most important downstream fork.
- Preserve all post-update policy checkpoints at 0,16,32,48,64, with full trainable adapter state plus tokenization/preference data revision. Restore to EXACT same policy and optimizer state for counterfactual forks.

### 6.3 Functional reward adaptation budget, not arbitrary LoRA parameter distance

Large neural reward models have different parameter scales, so Euclidean distance between their LoRA tensors does NOT supply a comparable reward-guidance speed. Implement an auditable **functional movement budget** on a fixed anchor set of K=128 response pairs:

\[
d(R_{\rm new},R_{\rm old})=
\sqrt{\frac{1}{K}\sum_{i=1}^K
\{[R_{\rm new}(x_i^+)-R_{\rm new}(x_i^-)]
-[R_{\rm old}(x_i^+)-R_{\rm old}(x_i^-)]\}^2}.
\]

Predeclare scores as pairwise margins and center/calibrate against the initial anchor batch to eliminate arbitrary additive shifts. This is an **empirical functional difference**, not a theorem-level parameter norm on a continuous feature space.

After fitting a candidate new reward head, deploy either a bounded interpolation in pairwise scores or a stable checkpoint interpolation with actually verified anchor differences so that d<=b per reward-update event. Primary b selected and frozen in development to give nonzero yet finite movement; include one no-adaptation b=0 control. Report the resulting actual score drift and any clipping, and do not assume this budget bounds reward changes OFF the anchor set. If a model-parameter interpolation violates score bound, shrink the step with an actual line search and log the achieved budget.

### 6.4 Evaluation that is independent of the training reward

There is **no free on-policy human evaluation** in this plan. The following are separate:

1. **Static human preference evaluation:** on the held-out original human-labeled comparison rows in S2, measure the trained RM ranking/NLL. This evaluates the REWARD MODEL on a fixed human-labelled distribution.
2. **Fresh policy output quality:** evaluate 128 fixed held-out TL;DR prompts using a fixed reference-summary metric (e.g. overlap, length and coverage diagnostics) and a separate frozen 4B judge if S4 supplies one. These are automated metrics, NOT new human preferences. Report uncertainty paired by PROMPT, plus independent training-seed variability.
3. **Teacher-model scores:** useful as an optimization diagnostic but never count teacher score alone as proof of real human quality improvement. Include disagreement with the S2 human holdout and reward-score-vs-response-length diagnostics.
4. **Heldout verifiable instruction task:** for Qwen1.5B, use a new disjoint prompt/template split with programmatic correctness checks from E6 ideas, not the public IFEval test set as training data. Report all 2–4 requirements separately, not a single opaque judge score.
5. **Reward reuse diagnostics across policy checkpoints:** apply initial fixed RM, refreshed RM and the independent judge to the SAME saved generated response pairs; calculate preference reversal fractions, margin change, calibration shift on fixed human labels and whether bounded reward updates actually change policy improvement under a frozen checkpoint. This is the key Horizon-related measurement.

### 6.5 Objective learnability and stability gates

Before running four primary LoRA training traces:

1. Run one initial SFT/reference policy generation and one 8-update LoRA development pilot. Verify prompts lead to nonempty summaries, adapter parameters change, training loss/advantage is finite, reward head differentiates responses, and KL can be kept inside a development-frozen reference bound (initial target mean KL <=0.1 nats/token if numerically appropriate; document the actual definition).
2. If learning fails despite valid data, test a 16-update evaluator-only fixed teacher reward control on the SAME starting policy, with identical optimizer/KL budget. If even that does not produce a measurable reward or verification change, do not interpret any failure as horizon/support exhaustion.
3. Freeze the actual 64-step learning rate, batch, reward scale and reference KL after development. If OOM occurs, lower microbatch, shorten development-chosen context length uniformly across ALL conditions, use gradient accumulation or serial RM scoring and log the deviation.
4. Report at each saved checkpoint **the best alternative bounded reward deployment action** evaluated on the SAME frozen candidate response pool, but label this as candidate-restricted empirical pointwise capability, not a global optimizer proof.
5. Use matched exact policy checkpoint forks for new AI feedback vs more fitting on the SAME human data. Both begin with identical actor weights, optimizer state, current RM anchor scoring and cached candidate responses. Do not confuse additional AI labels with newly collected human labels.

### 6.6 Actual Pythia/Qwen result table

| Task/model and learned policy method | Independent policy+label seeds | Human train comparisons used to fit initial RM | New ON-POLICY AI preference labels | Policy LoRA steps / generated tokens | Initial/final independent validation score | Static held-out human RM preference metric | Reference KL and output length | Reward anchor movement realized | Notes |
|---|---:|---:|---:|---|---|---|---|---|---|
| TL;DR Pythia-1B SFT, no policy update | n/a | 257 | 0 | 0 |  |  |  | n/a | gate passed in 21.9 s; adapter norm 6.548 to 6.553 over 8 updates |
| TL;DR Pythia-1B, source-only RM | 2 | 257 human | 0 | 64 | last-batch mean reward -6.51 and -8.35 | human RM, not a new human eval | logprob stored, not a KL certificate | anchor RMS 0.0 and 0.0 | fixed 128 response pairs; b=0.05 not exceeded |
| TL;DR Pythia-1B, online AI-feedback reward refresh | 2 | same 257 | 96 AI | 64 | last-batch mean reward -5.38 and -5.61 | same human-initialized RM after AI updates | not a human quality score | anchor RMS 1.46 and 1.35 | AI labels from the frozen CleanRL RM; exceeds b=0.05 |
| TL;DR Pythia-1B, same-data extra fitting fork | 2, at steps 16 and 48 | same 257, 32 of them refit | 0 | 16 | last-batch human-RM score stays near -6 to -7, like the keep branch | not a new human eval | same 16 updates and 32 new tokens | not separately remeasured | extra fitting did not separate from keep |
| TL;DR Pythia-1B, fixed AI teacher oracle control | 2, at steps 16 and 48 | 0 | AI teacher scores the policy | 16 | teacher-scale means near 0, not comparable to the human-RM scale | n/a | privileged | n/a | different numeric scale |
| Verifiable instruction Qwen2.5-1.5B, frozen base |  |  |  |  |  |  |  |  | NOT RUN. The 1B result and the 4B subset used the slot. |
| Verifiable instruction Qwen2.5-1.5B, actual LoRA policy |  |  |  |  |  |  |  |  | NOT RUN. Same reason. |

**Possible findings:** Additional AI feedback makes RM predictions on current-policy completions more stable but not the independent reference quality; existing human-trained RM remains useful for all tested checkpoints; new feedback changes reward margins but policy KL budget blocks a different action; or a learned reward becomes stale when policies produce outputs outside its source comparison distribution. Each is a scientifically interpretable, community-useful result when data sources and evaluation are honest.

## 7. S4 — 4B Scale and RewardBench 2 Independent Evaluation

**Research purpose:** Establish at least one genuine 4B-scale check on an externally recognizable multi-domain benchmark and provide a scalable public scoring adapter. This is not a claim that a 4B policy was fully trained in ten hours.

### 7.1 Frozen larger model and clear roles

Use Qwen/Qwen3-4B-Instruct-2507, already used as an E7 quality judge and previously cached according to WP01. Verify current cache revision/license and measured GPU load. Two possible modes:

- **Required if resources allow:** a FROZEN 4B comparative judge that receives a fixed rubric and the pair/multiple responses. It is an LLM-as-judge scorer, NOT an explicitly trained scalar reward model. Keep model sampling deterministic (temperature 0 where supported) and randomize order to measure positional bias.
- **Optional, capacity-gated:** a small 4B scalar reward LoRA/head trained on the same 1,024/4,096 human/AI preference data used in S2, with one seed and pinned revision. Only call it a trained 4B RM if its weights ACTUALLY receive pairwise loss updates, nonzero gradients and held-out metric logging.

The required community value can be delivered by the first mode and the public scorer API even if optional 4B training is not feasible. Do not present inference-only scaling as training-scale confirmation.

### 7.2 RewardBench 2 protocol matters

RewardBench 2 has SIX categories including Ties and frequently has multiple chosen/rejected completions per prompt. A correct evaluator must load list-valued chosen/rejected responses, apply the official criterion for chosen over rejected and Ties ranking/margin, and calculate per-category and macro-average scores in the declared exact or subset protocol. Do not flatten all rows into one arbitrarily chosen positive/negative pair and call it RewardBench 2 accuracy.

**Primary quota:** predeclare a stratified sample of roughly 384–600 of the 1,865 listed test rows, with seed and original row IDs frozen BEFORE any model predictions; preserve all six category counts including the 102-row Ties group. Evaluate 4B and the best already working smaller scorer on the SAME rows. Report model scoring prompt template, response ordering, parsing failures, refusal/invalid rates and average inference tokens/example.

**Stretch only if measured GPU throughput permits:** evaluate ALL 1,865 RewardBench 2 rows with the official scorer semantics. If run on a subset, label it subset results, not an official full leaderboard result. If a comparison uses a 0.5B learned RM vs 4B prompted judge, report the algorithmic difference; it is not a pure parameter-count scaling law.

### 7.3 Real evaluation and cross-size robustness

1. Compare source-only RM versus refreshed RM and frozen 4B judge on EXACT same external held-out row identities. Report category-wise correctness, ties metric, positional consistency, agreement/disagreement and a transparent confidence interval over independent prompt rows. No training/threshold selection on these test rows.
2. On a small fixed cross-domain set from TL;DR, compare the 0.5B human-trained reward head, 1B task-specific frozen teacher and 4B judge on the same responses; report ranking correlations, length preference, failure cases and domain specificity. These are different reward functions, not necessarily a shared true scalar.
3. On policy-generated TL;DR held-out prompts, use the frozen 4B judge only as a secondary independent AI metric with a precommitted rubric and blinded method order. Do not report its relative scores as human approval or assume this judge is perfectly reliable.
4. Record VRAM, throughput and optional adapter train iterations if performed. If 4B loading or RewardBench2 multiresponse scoring cannot finish within the declared quota, report the actual completed row IDs/coverage, restore clean model unloading and complete the smaller-scored external set rather than faking a 4B result.

**S4 result table:**

| Reward scorer / judge | Architecture and train/inference role | Params | Test rows and category coverage | Full or subset protocol | Per-category + aggregate RewardBench 2 metric | Positional consistency / invalid parsing | GPU sec, VRAM peak | Notes |
|---|---|---:|---|---|---|---|---|---|
| 0.5B general RM if S2 AI data trained | scalar RM, trained | 0.5B | none | not run |  |  |  | NOT RUN. UltraFeedback was not fit. |
| 0.5B TL;DR RM | scalar RM, trained on human TL;DR | 0.5B | 502 frozen rows, all six subsets including 102 Ties | subset, not a leaderboard number | macro 0.209. Ties 0.029, Math 0.225, Precise IF 0.238, Focus 0.125, Factuality 0.350, Safety 0.288 | strict all-chosen-above-all-rejected rule | part of the 96 s combined run | human_4096 head. Domain shift from TL;DR. |
| Pythia-1B frozen TL;DR reward teacher | task-specific scalar RM, already pretrained | ~1B | not scored on RewardBench 2 | not a leaderboard number |  |  |  | Used only as the AI labeler in S3 and S5. |
| Qwen3-4B-Instruct-2507 | frozen prompted comparative judge | ~4B | same 502 rows; full 1,865 not scored | subset exact-set match, not the official script | macro 0.112. Ties 0.294, Math 0.013, Precise IF 0.000, Focus 0.200, Factuality 0.000, Safety 0.163 | temperature 0, order seed 7; a reply must name exactly the chosen indices | 96 s for the 0.5B plus 4B pass, about 8 GB while the 4B was loaded | Prompted judge, not a trained 4B RM. |
| Qwen3-4B scalar reward LoRA, optional | explicitly trained RM if done | ~4B | SAME predeclared rows | subset/full clearly marked |  |  |  | NOT RUN until weights trained |


## 8. S5 — Matched-Budget Feedback Versus Reward Optimization and Reuse Monitoring

**Objective:** Turn the distinction among preference information, reward fitting and finite reward adaptation into an actionable measurement on ACTUAL pretrained-policy checkpoints. S5 is a targeted causal experiment, not a new huge model grid.

### 8.1 Reuse and progress measurements at a frozen checkpoint

For a frozen policy checkpoint, original reward head R0, updated head R1 and its observed preference batches, log these independently:

1. **Offline preference validity:** frozen held-out human pairwise accuracy/NLL/Brier for each RM; cross-domain AI-labeled scores are separately tagged.
2. **Policy distribution drift:** generated response length, reference KL, reward-margin reversal fraction and output feature/behavior distribution across actual policy checkpoints. Old R0 and new R1 must score the SAME cached responses for a matched scorer comparison.
3. **Functional reward movement:** the RMS change of pairwise margin on the PREDECLARED 128 anchor comparison pairs. Enforce the stated functional adaptation budget for actual deployed reward checkpoints and report violations.
4. **One-step real policy effect:** restore one exact actor/optimizer checkpoint and execute the SAME actual policy update with different deployed reward variants and matched candidate completions. Evaluate reference/verification utility independently after the update.
5. **Information-only effect:** change preference evidence and retrain/swap a scoring head without applying any policy update. The existing actor's independent output metrics must stay constant. Any changed policy behavior here is a bug or non-frozen evaluation.
6. **Censored empirical reuse endpoint:** predeclare an empirically measurable criterion using development-only data, including its score direction, uncertainty method and candidate policy checkpoints. If no loss appears, report no observed loss inside the evaluated interval, NOT an infinite formal support horizon.

These are not one interchangeable "guarantee" number. The API records a separate type/scope for every metric. Neural reward estimates never receive the paper's scalar exact-horizon theorem by implication.

### 8.2 Fixed same-checkpoint interventions

Use the real saved Pythia policy checkpoints at updates 16 and 48. Freeze actor weights, optimizer state, candidate summaries, evaluation prompt IDs, source RM, held-out evaluation data, reward scale and sampling RNG as far as possible. Then compare:

| Branch | New preference information | RM training | Policy continuation | Single changed cause |
|---|---|---|---|---|
| Keep source RM | 0 | unchanged | 8–16 LoRA updates | reference |
| More fitting, same human source pairs | 0 | +16 RM optimizer steps on SAME 256/1024 original labels | same updates | compute only |
| New historic HUMAN train comparisons | reveal 32 previously hidden TRAIN annotations, from a frozen candidate pool | +16 RM steps, matched token/compute budget as possible | same updates | offline information |
| New ONLINE AI-labeled comparisons | 32 pairs from actual actor-generated completions, fixed frozen teacher | +16 RM steps | same updates | on-policy AI information |
| Both new feedback and more RM computation | exactly the same 32 labels as its corresponding feedback branch | +32 RM steps | same updates | interaction of feedback and optimization |
| Frozen teacher positive control | privileged model feedback, explicitly marked AI oracle | no trainable RM needed | same updates | tests whether the policy update can move |

**Budget accounting:** Match the number of policy updates, candidate generated tokens and actor optimizer steps EXACTLY. Reward-head learning can have different wall time and token throughput when examples differ; report both actual RM tokens/steps and GPU seconds. Do not describe a comparison as matched-compute unless recorded GPU/token costs actually agree. Old human TRAIN annotations revealed later are not on-policy human feedback; do not mix them in the same provenance category as AI-labeled on-policy responses.

**Data selection:** draw/choose indices before revealing the associated hidden label. For uniform vs policy-relevant selection use the SAME candidate pool. Query-selection features/weights must come from prior data, not the evaluator label or frozen 4B external test set.

**Evaluation:** use independent same-document held-out generated summaries, reference overlap/coverage, repeat-rate, response length, reference KL and a blinded independent frozen 4B judge if S4 completes. An increase in the frozen AI-teacher score by itself is insufficient to claim better human-aligned summary quality. Report independent seed-level differences (minimum two paired policy seeds) and paired prompt-level uncertainty without pretending individual prompts are independent training replications.

### 8.3 A community-facing resource diagnostic, not an unsupported oracle

The audit tool may return one of the following explicitly scoped **suggestions**, with a numerical reason and a no-guarantee warning when neural models are used:

- collect_more_comparisons: reward predictions on a predeclared candidate/query distribution remain empirically unresolved; suitable informative pairs are available and evidence indicates the information may change a guidance choice.
- optimize_existing_reward_more: the fit on the SAME observed labels has demonstrably not converged; do not prescribe more steps after the MLE/adapter fit stabilizes.
- reward_change_budget_limited: the reward change that distinguishes policy choices exceeds the measured functional movement bound on the anchor set.
- policy_optimization_limited: even a privileged/reference reward controller cannot yield nondegenerate policy progress under the specified training budget.
- no_supported_action: measurement noise, reward-model mismatch, poor OOD performance or lack of a valid confidence model precludes a stronger recommendation.

The tool must not claim a predicted numerical extension to a neural policy's exact support horizon unless assumptions and a separately valid theorem actually support that estimate.

**S5 result table:**

| Actual policy/checkpoint | Reward information branch | New historical human / on-policy AI labels | Extra RM steps and GPU tokens | Held-out preference NLL difference | Anchor RMS functional movement | 8–16-step independent policy utility difference | Formal/empirical scope | Recommended action or no evidence | Notes |
|---|---|---|---|---|---|---|---|---|---|
| Exact finite reference | source then real fixed-N BTL | fixed synthetic labels | n/a |  |  |  | exact/statistical |  | |
| MiniGrid neural checkpoint | original vs 32 BTL | synthetic |  |  |  |  | empirical |  | |
| Qwen-0.5B RM frozen actor | source vs revealed old human TRAIN labels | human: 32 |  |  |  | not an actor update | empirical RM only |  | |
| Pythia-1B update 16 | keep source | 0 | 0 |  | 0 |  | empirical |  | |
| Same checkpoint | fit existing source more | 0 | 16 |  |  |  | empirical |  | |
| Same checkpoint | reveal historical human source comparisons | human:32, offline | 16 |  |  |  | empirical |  | |
| Same checkpoint | new frozen-teacher feedback | AI:32, on-policy | 16 |  |  |  | empirical |  | |
| Pythia update 48 | repeat matched essential branches | provenance separated |  |  |  |  | empirical |  | |
| Qwen1.5B learned instruction policy if available | synthetic verifier/BTL |  |  |  |  |  | empirical |  | |

**Interpretation:** Improvements may be reward-predictive only, action-changing only, return-changing only, or absent. Each case supports a useful monitoring diagnostic if the data and evaluation are correct. For neural models, none of these alone constitutes a new formal support-horizon theorem.

## 9. S6 — Public API, Benchmark Package and Release-Quality Audit

**Objective:** The code must help other researchers evaluate their OWN reward models, preference data, policy checkpoints and adaptation/feedback budgets. It must be genuinely usable outside this repository, not just a new set of paper figures.

### 9.1 Required artifact and minimal interface

Create an importable evaluation package with no implicit local checkpoint path. Public functions (names may differ but functionality must be present):

- load_manifest(path): validated versioned manifest, with model/data provenance, compatibility set when available, seed/checkpoint IDs and clear errors.
- audit_finite(checkpoints, policy_step_rule, preference_set, movement_budget): exact/valid gain and reachability diagnostics with scope assumptions explicitly documented.
- audit_empirical(checkpoints, preference_pairs, heldout_results): neural reward ranking, drift, realized update impact, cost and censored empirical endpoints; must not fabricate certificate values.
- compare_interventions(parent_checkpoint, branch_outputs): paired results with strict requirements for shared starting state, candidate pool, labels and optimizer state.
- export_scorecard(runs): stable JSON/CSV/short report with provenance, uncertainty, measured runtime and status.

Provide a CPU-only executable example using existing 16-outcome/64-outcome saved traces, and a Gym example from MiniGrid. A real LLM example must identify the actual downloaded model adapter and reward head revision rather than requiring any unseen local hidden filesystem path.

**Proposed CLI; finalize syntax during implementation:**

~~~bash
python -m experiments.guidance_audit validate --manifest examples/07_guidance_audit/finite_example.json
python -m experiments.guidance_audit audit --manifest examples/07_guidance_audit/finite_example.json --out outputs/horizon_scale_07/public_smoke/
python -m experiments.guidance_audit compare --baseline outputs/horizon_scale_07/s3_llm/source/ --treatment outputs/horizon_scale_07/s3_llm/online_ai_feedback/
~~~

All commands must have real --help, helpful failure messages and at least one passing smoke/regression test. Imports of Hugging Face, GPU libraries and MiniGrid should be optional so that the exact reference example can run in a clean CPU environment.

### 9.2 Versioned result schema

A file suitable for release needs, at minimum:

~~~yaml
schema_version: 1
experiment_id: human_readable_unique_id
task_id: exact_or_official_environment_or_dataset
source_type: human | model_judge | synthetic_btl | verifier
evaluator_type: exact_finite | heldout_human | verifier | model_judge
guarantee_type: exact_finite | statistical_fixed_design | empirical_proxy
policy_model_id: public_or_custom
policy_revision: full_hash_or_null
reward_model_id: public_or_custom
reward_revision: full_hash_or_null
dataset_id: public_id_or_fixture
dataset_revision: full_hash_or_null
label_pair_ids: array_of_actual_ids
checkpoint_step: integer
policy_optimizer: named_algorithm_and_hyperparams
reward_optimizer_steps: integer
new_label_count: integer
feedback_time: timestamp_or_step
adaptation_budget_unit: parameter_or_functional_anchor
adaptation_budget_value: numeric_or_null
independent_true_return: number_or_null
original_certified_gain: number_or_null
updated_certified_gain: number_or_null
empirical_heldout_nll: number_or_null
first_unsupported_checkpoint: integer_or_censored_or_null
gpu_seconds: numeric_or_null
status: complete | censored | incomplete | invalid
~~~

This is an interface design example, NOT an experimental result. A formal non-null certificate additionally requires original uncertainty-set definition, model class, alpha and coverage method, exact/verified policy update, source and updated label schedule, and validated tested time scope. Reject an empirical-only neural manifest that improperly asserts an exact theorem guarantee.

### 9.3 Public documentation and outputs

**Five-minute tutorial:** A minimal CPU demo then a short real Gym demo, and a one-GPU 0.5B preference-RM example with optional dependencies. Make the tutorial as interactive and concise as reasonable, with a command and one interpretable result per section. Keep extended derivations in a separate technical reference rather than in the public-facing README.

**Benchmark manifest and model card:** Dataset/model IDs and pinned revisions; license/provenance; original train/dev/test source IDs and hashes; label origin human/AI/verifier/synthetic; details of policies/reward heads; score definitions; budgets and actual wall time; seed-level results; independent evaluator and caveats about confidence/certificates. Do not rehost third-party texts or model weights before verifying redistribution permissions. Expose only derived numeric metrics and permitted IDs where appropriate.

**At most three paper figures:** a multi-scale empirical reuse scorecard, one matched information-versus-RM-compute fork, and one clearly drawn policy/RM distribution-drift view. Existing exact theorem figures remain separate. Generate every number from machine-readable seed-level results, not manual spreadsheet entry.

**Publishing:** Updating this GitHub repository and its docs is in scope. Hugging Face Space/dataset/model uploads are NOT implicit permission to use unavailable credentials and are optional only when authorized separately. The executor may prepare Hub-compatible cards and upload commands as files without claiming that it performed an upload.

### 9.4 Release acceptance table

| Public community deliverable | YES / PARTIAL / NO | Actual smoke test or usage check | Path/commit | Distinct benefit to another researcher | Blocker if incomplete |
|---|---|---|---|---|---|
| Importable audit Python API | YES | S0 checks passed | experiments/guidance_audit | Compare a manifest without inventing a certificate | |
| CLI and schema validation | YES | task S0 plus the package CLI | experiments/guidance_audit/__main__.py | Machine-readable guarantee flags | |
| 5-minute CPU reference | YES | S0 finished in 0.28 s | experiments/horizon_scale_07/preflight | No-GPU check of T0 and the ordinary-PG contact | |
| Real Gym neural-policy example | YES | Empty-8x8 oracle learned; DoorKey did not | experiments/horizon_scale_07/minigrid | A learned neural policy, with a failed DoorKey control left uninterpreted | |
| Real HUMAN TL;DR comparison RM fitting | YES | 0.506 to 0.600 as human pairs grow from 257 to 4,098 | experiments/horizon_scale_07/human_tldr | Human labels, validation untouched | |
| Actual Pythia-1B LoRA policy trajectory | YES | 64 updates, two seeds, two methods | experiments/horizon_scale_07/pythia_1b | Source reward stays fixed; 96 AI labels move it | |
| Qwen1.5B independent policy transfer | NO |  |  | Second model family, if run | NOT RUN |
| Frozen 4B external reward/quality scorer | PARTIAL | 502-row subset, exact-set judge, not the official script | experiments/horizon_scale_07/rb2_4b | A frozen 4B judge on a predeclared subset | Full 1,865 and the official script were not run |
| Standardized seed-level benchmark manifest | YES | JSON under experiments/horizon_scale_07 | same directory | Seed-level numbers for the three figures | |
| Documentation/tests/license audit | PARTIAL | this file filled; no new license audit | work_plan/07_Scale_and_Community_Evaluation.md | The path mapping and the censored rows are written down | |

## 10. Results, Hypotheses and Compute Ledger (EXECUTOR MUST FILL)

### 10.1 Main scientific claim decision table

| Scientific hypothesis or community objective | Required experiment/evidence | YES / PARTIAL / NO / UNSUPPORTED | Actual result and uncertainty | Precise scope/limitation |
|---|---|---|---|---|
| A learned preference RM's useful lifetime may shorten as policy output distributions change | S2 human-heldout score history + S3 actual policy checkpoint drift + S5 same-checkpoint true policy intervention | PARTIAL | Source-RM anchor RMS stayed 0 across 64 policy updates. 96 AI labels moved those fixed anchors by 1.46 and 1.35. That is reward change, not a proof that the old reward expired. | empirical_proxy. Not a neural support horizon. |
| More reward fitting and more preference labels solve different limitations | S2/S5 same-data extra optimizer versus new-pair reveal, matched checkpoints | YES on S2, unresolved on S5 | On the 0.5B head, extra epochs on the same 257 pairs stayed near 0.50, while 4,098 nested human pairs reached 0.600. Pythia forks did not separate keep, extra fit, and new labels on the last batch. | S5 scores are one minibatch, not a fixed validation set. |
| A finite functional reward movement budget can prevent an observed information improvement from changing policy behavior | S3/S5 measured anchor movement and bounded alternative deployment | PARTIAL | b=0.05 was frozen before the confirm run. Method A did not exceed it. Method B did, with RMS about 1.4. The last-batch policy reward did not show a clean matched gain. | The budget is an empirical threshold, not a theorem. |
| Real human comparison data materially improve reward identification | S2 trained actual 0.5B RM evaluated on untouched HUMAN comparisons | YES, modestly | Validation accuracy 0.506, 0.551, 0.600 at 257, 1,024, and 4,098 human pairs. A length and overlap logistic baseline was about 0.56 on 2,048 pairs. | Neural validation N is 512. Length correlation rose to 0.318. |
| Preference/guidance reuse measurements transfer from exact systems to neural-policy RL | S0/S1 exact versus empirical distinction and learnability control | PARTIAL | Exact T0 is 5 and the ordinary-PG contact is 0.00783587. Empty-8x8 learned. DoorKey did not, so it was not interpreted. | Neural rows are empirical_proxy. |
| Neural RM/policy findings survive authentic 1B-1.5B weight updates | S3 nonzero LoRA updates, seed level and independent evaluation | PARTIAL | The 1B LoRA ran 64 updates on two seeds. The 8-step gate changed the adapter norm from 6.548 to 6.553. The 1.5B run was not done. | Qwen2.5-1.5B is NOT RUN. |
| At least one independent 4B-scale evaluation is completed with the correct external benchmark semantics | S4 frozen evaluator and reported category coverage | PARTIAL | The frozen Qwen3-4B judge and the 0.5B RM scored the same 502 predeclared rows, including all 102 Ties. Full 1,865 was not scored. | Exact-set match is not the official RewardBench 2 script and is not a leaderboard number. |
| Community researchers can audit their own checkpoints without author-specific hidden paths | S0/S6 external manifest + CLI tests, no mandatory GPU | YES | experiments/guidance_audit loads a manifest, refuses a missing file, and keeps empirical results uncertified. | Run it through tasks/07_Scale_and_Community_Evaluation. |

**No universal positive-result requirement:** Stable RM reuse, no preference utility improvement, or model-scorer disagreement all have research value if the deployment/feedback variables and evaluation are controlled.

### 10.2 Completion table

| Stage | NOT RUN / PILOT / COMPLETE / PARTIAL / BLOCKED | Actual tasks/models and sizes | Independent policy/model seeds | New labels and their TRUE source | Measured core result | Result path and source revision | Why scope was reduced/censored |
|---|---|---|---|---|---|---|---|
| S0 audit API and preflight | COMPLETE | CPU audit of horizon.py and ordinary PG | 0 | none | T0=5; contact 0.00783587 | experiments/horizon_scale_07/preflight | Results live here rather than outputs/horizon_scale_07. |
| S1 neural MiniGrid | COMPLETE | Empty-8x8 PPO; DoorKey oracle did not learn | 3 on Empty, 1 on DoorKey | synthetic BTL, not human | Empty oracle 0 to 1.00 success. Extra-fit failed on seed 2. | experiments/horizon_scale_07/minigrid | DoorKey forks not run because the oracle did not learn. |
| S2 TL;DR human comparison / 0.5B RM | COMPLETE | Qwen2.5-0.5B-Instruct LoRA reward head | 2 on 257, 1 on larger rungs | human comparisons, train only | 0.506 / 0.551 / 0.600 at 257 / 1024 / 4098 | experiments/horizon_scale_07/human_tldr | Validation N is 512 of a frozen 2048, not the full valid1 split. |
| S2 optional UltraFeedback general RM | NOT RUN |  |  |  |  |  | Human track used the slot. AI labels were not mixed in. |
| S3 Pythia-1B real policy updates | COMPLETE | cached Pythia-1B SFT LoRA, 64 updates | 2 | 0 or 96 AI labels from the frozen CleanRL RM | anchor RMS 0 versus 1.46 and 1.35 | experiments/horizon_scale_07/pythia_1b | Last-batch reward is not a held-out human score. |
| S3 Qwen2.5-1.5B second policy | NOT RUN |  |  |  |  |  | Left for after the 1B result and the 4B subset. Not started. |
| S4 4B scorer / external RewardBench 2 | PARTIAL | frozen Qwen3-4B judge and 0.5B human RM | 1 scoring pass | no new labels; test rows only | subset macro 0.112 judge, 0.209 RM | experiments/horizon_scale_07/rb2_4b | 502 of 1865. Not the official script. 4B LoRA NOT RUN. |
| S5 matched resource and reuse diagnostics | COMPLETE | forks at steps 16 and 48 | 2 | 0, 32 historical human, or 32 AI | human-RM branches stay near -6 to -7; teacher is a different scale | experiments/horizon_scale_07/checkpoint_forks | Suggestion is no_supported_action, with a no-guarantee warning. |
| S6 community package and tests | COMPLETE | three figures and the filled tables |  |  | figures from the seed JSON | experiments/horizon_scale_07/public_smoke | No Hugging Face upload. |

### 10.3 Real measured machine/resource ledger

| Stage | Planned wall-clock h | Actual wall-clock h | Measured GPU seconds | Peak GPU VRAM and CPU RAM | Actual train/pair/eval sizes | Main deviation or optimization | Final status |
|---|---:|---|---|---|---|---|---|
| S0 | 0.50 | 0.00008 | 0 | CPU | 11 checks | none | COMPLETE |
| S1 | 0.75 | about 0.2, CPU | 0 | CPU | 40k pilot plus about 100k forks, 3 seeds | DoorKey stopped after a nonlearning 40k oracle | COMPLETE |
| S2 | 1.25 | 0.36 | 1295 s on GPU | about 3 GB | 257 to 4098 human pairs; 512 validation pairs | one discarded run used gradient checkpointing and did not fit | COMPLETE |
| S3 | 3.00 | 0.21 | 772 s | about 4 GB | 256 train docs, 128 anchors, 64 updates, 2 seeds | anchors are fixed responses | COMPLETE |
| S4 | 1.50 | 0.027 | 96 s | about 8 GB while the 4B judge was loaded | 502 rows, six subsets | full 1865 and the official script not run | PARTIAL |
| S5 | 1.25 | 0.28 | 1006 s | about 5 GB | 16 updates, 6 branches, 2 seeds, 2 checkpoints | teacher scores are a different scale | COMPLETE |
| S6 | 1.25 |  | 0 | CPU | 3 figures | no Hub upload | COMPLETE |
| Reserve | 0.50 |  |  |  | GPU was held by another job at the start; this run waited |  |  |
| **Total** | **10.00** | about 1.1 measured GPU hours plus the CPU MiniGrid run |  |  |  | shorter than the plan because the 1.5B run and full RewardBench were not done |  |

A reported 10-hour experimental budget is a **target**; do not invent durations or claim full completion if a download/training stage was blocked. Record meaningful partial results and the dependency/cost of unfinished work.

## 11. Proposed Source/Artifact Layout and Executor Command Plan

Paths listed here are PLANNED implementation targets, not proof they already exist.

~~~text
work_plan/07_Scale_and_Community_Evaluation.md

experiments/guidance_audit/
  __init__.py
  __main__.py
  schema.py
  finite.py
  empirical.py
  decisions.py
  reporting.py
  tests/
examples/07_guidance_audit/
  finite_example.json
  finite_checkpoint.csv
  minigrid_example.json
  quickstart.md

experiments/large_scale/
  07_preflight.py
  07_minigrid.py
  07_tldr_human_data.py
  07_rm_05b.py
  07_policy_1b.py
  07_qwen_15b.py
  07_rewardbench2_4b.py
  07_checkpoint_forks.py
  07_release.py
  requirements_scale07.txt

outputs/horizon_scale_07/
  preflight/
  finite_regression/
  minigrid/
  human_tldr/
  ultrafeedback_optional/
  pythia_1b/
  qwen_15b_optional/
  rb2_4b/
  checkpoint_forks/
  public_smoke/
  release/
~~~

**Command interface to implement (reasonable equivalent CLI is acceptable; describe exact executed commands in results):**

~~~bash
python experiments/large_scale/07_preflight.py --out outputs/horizon_scale_07/preflight
python -m experiments.guidance_audit validate --manifest examples/07_guidance_audit/finite_example.json
python -m experiments.guidance_audit audit --manifest examples/07_guidance_audit/finite_example.json --out outputs/horizon_scale_07/finite_regression
python experiments/large_scale/07_minigrid.py --stage pilot --out outputs/horizon_scale_07/minigrid
python experiments/large_scale/07_minigrid.py --stage confirm --out outputs/horizon_scale_07/minigrid
python experiments/large_scale/07_tldr_human_data.py --out outputs/horizon_scale_07/human_tldr
python experiments/large_scale/07_rm_05b.py --stage train --out outputs/horizon_scale_07/human_tldr
python experiments/large_scale/07_policy_1b.py --stage pilot --out outputs/horizon_scale_07/pythia_1b
python experiments/large_scale/07_policy_1b.py --stage confirm --out outputs/horizon_scale_07/pythia_1b
python experiments/large_scale/07_rewardbench2_4b.py --out outputs/horizon_scale_07/rb2_4b
python experiments/large_scale/07_checkpoint_forks.py --out outputs/horizon_scale_07/checkpoint_forks
python experiments/large_scale/07_release.py --out outputs/horizon_scale_07/release
~~~

### 11.1 Dependency and work-priority policy

- **Critical path:** S0 validation/preflight -> S2 authentic human-label RM -> S3 actual 1B actor updates -> S5 matched checkpoint forks -> S6 reusable community interface/tests. Do not sacrifice the critical path to optional 4B LoRA or full 1,865-row external evaluation.
- **Independent inexpensive path:** S1 neural MiniGrid reference may proceed while dataset preparation/LoRA loading is configured. It is meaningful only after a trained true-reward positive control; it must not block the human/LLM critical path if PPO cannot learn.
- **Scale verification path:** S4 frozen 4B evaluation should run on a precommitted stratified subset if the real throughput meets its planned 1.5-hour slot. If 4B loading would prevent finishing S3 or S6, explicitly mark the optional step PARTIAL/CENSORED rather than jeopardizing the main contribution.
- **At every hard failure:** record the exact exception/input condition and preserve run status, then continue independent work; an invalid or partial label pipeline must NOT silently swap in synthetic labels and still claim a real human test.
- **Release minimum (even if GPU blocks):** working CPU CLI, validated manifest/schema, 5-minute exact example, provenance and scorecard of successfully measured stages, full reproduction instructions. The release is scientifically useful even if expensive model arms are incomplete.

### 11.2 Stop/continue gates that preserve scientific meaning

| Stage | Must be demonstrated before expensive continuation | When NOT demonstrated | What is still worth finishing |
|---|---|---|---|
| S1 MiniGrid | Known-target policy learns, environment observations/rewards audited | Do not assert a guidance Horizon in a nonlearning task | Small positive-control diagnosis, raw logs, API adapter |
| S2 0.5B RM | Labels are truly human on heldout, optimizer updates weights, NLL/accuracy finite and splits clean | No new S3 RM-derived policy performance claim | Dataset loader, frozen-head baseline and truthful failure note |
| S3 1B policy | Nonzero trainable adapter update, KL controlled, independent response metric nondegenerate | Do not burn main GPU allocation repeating a broken full PPO loop | Held-out RM drift measurements and reproducibility packaging |
| S4 4B benchmark | Correct multiresponse/ties semantics, model loads and per-row timing known | No "official RewardBench 2" score from broken subset | Small externally audited scorer, row coverage and omission reason |
| S5 causal feedback | Same parent checkpoint and optimizer restored exactly | Do not claim causal data versus fitting effect | Information-only measurement and dependency-mismatch report |
| S6 package | Tests pass from clean CPU env, values trace back to machine-readable records | Do not claim community-ready release | Publish source status as incomplete with exact blockers |

## 12. Handoff to the Running Research Agent

1. **Start from verified main commit** recorded at top. Read Work Plans 01–06 and the actual source methods in experiments/bridge, experiments/real/language, horizon.py and docs/API.md. Preserve previous results/reference and all E1–E8/H1–H4/P1–P5/R1–R5 results unchanged.
2. **S0 first.** Implement the audit schema/API skeleton, test exact regression references and measure GPU/CPU/model availability. Pin data and model revisions and write a real throughput ledger.
3. **S1 small neural MDP.** Pass a learnability gate on MiniGrid Empty, then optional DoorKey and one frozen-checkpoint feedback intervention, without treating an untrained policy as a horizon result.
4. **S2 real human preference work.** Download/load permitted public comparison datasets, audit group-disjoint splits, train a real 0.5B RM on 256 then 1,024/4,096 HUMAN train pairs with matched reward steps and honest heldout comparison metrics. UltraFeedback and RewardBench 2 remain separate label/test categories.
5. **S3 true model weights.** Train Pythia-1B LoRA actor for nontrivial multiple updates from frozen SFT, with source-only human-trained RM versus bounded on-policy AI-teacher feedback. Record actor training, KL, independent utility, RM anchor movement and checkpoints; run small Qwen1.5B transfer only if feasible.
6. **S4 larger model.** Use frozen 4B scorer on the predeclared external multiresponse test subset with correct Ties semantics, explicitly differentiating a prompted frozen judge from an actual trained scalar RM. Expand to full benchmark only if measured time allows.
7. **S5 actual resource experiment.** Fork exactly saved policy checkpoints and compare extra reward optimization on SAME data versus revealed historical human train labels versus new on-policy AI labels. Match policy budgets, measure actual reward-training cost and independent evaluator quality.
8. **S6 reproducibility/public benefit.** Finalize CPU and GPU examples, schema tests, result manifests, documentation and three concise figures. Expose which guarantee types are valid and which results remain empirical. Do not rehost unlicensed raw human data.
9. **Fill THIS document**, including every completed experiment table, actual times/VRAM, seed counts, measured human/AI provenance, actual training updates, gaps/limitations and reproducible paths. Every omitted 4B/large run must be explicitly NOT RUN or CENSORED with a pilot-grounded reason.
10. **Commit and push main**, with code, compact seed-level numeric results, evaluator scripts, tests, docs and completed Work Plan 07. Return commit SHA and direct public file links. Do not announce a Hugging Face release unless separately authorized and actually performed.

**Research-success standard:** This is a community-facing, reproducible evaluation of reward-guidance reuse across exact finite systems, stochastic neural RL and real open language models, grounded in authentic human preference data. A correct finding of stable reuse or no performance benefit is acceptable; a made-up neural horizon, invalid human-label claim, or unmatched-control performance advantage is not.

**Finish criteria:** the plan is implementable without a new round of design discussion; a worker can tell precisely what to run, when to stop, what to store, how to distinguish human versus AI feedback, and what another researcher can reuse. Results only become COMPLETE after actual execution.
