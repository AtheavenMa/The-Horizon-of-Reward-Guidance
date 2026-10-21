# 01 Reward Guidance Experiments

**Project:** The Limits of Reward Guidance under Finite Adaptation  
**Purpose:** ICML 2027 resubmission, real-learning experiments  
**Status:** Result cells filled from `experiments/horizon_real`. Unresolved estimates stay unresolved.  
**Prepared:** 2026-10-18  
**File naming:** Proposed sequential name 01_Reward_Guidance_Experiments.md. The project owner may approve a final rename; do not rename automatically.  
**Repository:** https://github.com/AtheavenMa/The-Limits-of-Reward-Guidance-under-Finite-Adaptation

> **Primary instruction for the executing agent:** Implement the experiments, fill the tables in THIS document, fill the paths in Section 7, and return the completed document. Additional analysis is optional and cannot replace the filled plan. Do not edit the research claims to match desirable results.

## 0. Locked execution decisions

These decisions fix scale, models, thresholds, and artifact paths. They do not change the Section 3 steps, table columns, or scientific definitions.

- **Scale.** Run all of E1–E8. Key results prefer 5 independent seeds. A minimal pilot must succeed before the confirmatory seeds start. If one run is expected to exceed about one hour, accelerate the implementation without changing the comparison; if it is still too long, continue with later experiments and return to it. Do not drop an experiment.
- **Order.** E1, E2, E3, then E4 (Walker Walk, then Cheetah Run), E6, E5 (Push, then Drawer Open), E7, E8.
- **Required gain.** Report `gamma = 0` and `gamma_pos = 0.1` times the median positive development finite-step gain. Freeze `gamma_pos` before confirmatory seeds. If development runs have no positive gain, check learning, reward scale, and whether the policy is updating before choosing another documented handling. Do not retune the threshold on confirmatory results.
- **Models.** E6 policy: `Qwen/Qwen2.5-1.5B-Instruct`. E7 policy: Pythia-1B SFT `cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr`. E8 second family: `Qwen/Qwen3-4B-Instruct-2507`.
- **Control stack.** Reimplement SAC, Bradley–Terry reward learning, a reward ensemble, teacher pairwise comparisons, replay, and checkpoint forks in current PyTorch. Do not restore the 2021 B-Pref environment. Record the deviation in Notes.
- **E7 labels.** Initialize the reward from human comparisons in `openai/summarize_from_feedback`. Score new online summaries with the fixed Pythia TL;DR reward model `cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr` and mark those labels AI feedback. Score final quality with `Qwen/Qwen3-4B-Instruct-2507`. Do not call automatic labels human annotation. Human blind evaluation is unavailable.
- **Artifacts.** Write run outputs under `experiments/horizon_real/E1` … `E8/`, not `outputs/horizon_real/`. Commit aggregate CSV, JSON, and figures. Do not commit `checkpoints/` or model weights.

## 1. Paper and Experiment Overview

### 1.1 Research question and current evidence

The paper asks how long existing preference information can **certify policy improvement** while the policy changes and the guidance reward has finite adaptation capability. The central distinction is:

1. **Pointwise feasibility:** a reward setting that guarantees progress is available at one policy state.
2. **Trajectory feasibility:** an admissible reward-parameter trajectory, starting at the declared initialization, can maintain that guarantee.
3. **Statistical certification:** available preference feedback contains enough information to support the guarantee.

These are different from out-of-distribution reward prediction accuracy or eventual task return. Failure of a certificate does **not** by itself imply that true policy return has fallen.

Read the current manuscript before implementation, especially Sections 3–8 and Appendices A, B, C, D, E and F. The principal results to respect are:

- **Theorem 1, Section 4.2:** exact scalar support horizon under a moving reward target and a parameter-speed limit. The theorem is about the admissible class, not the actual trajectory of every reward optimizer.
- **Theorem 2, Section 5:** a local loss-of-certificate construction under ordinary Euclidean policy gradient, even though pointwise capability remains sufficient.
- **Theorem 3, Section 6:** Bradley–Terry preference complexity governed by distance to the dynamic feasibility boundary in the declared scalar model.
- **Section 7:** in the general case, excluding dynamically infeasible preferences does not itself guarantee that the remaining preferences share a common control.
- **Appendix D:** finite-policy-change and score-vector-product interfaces are available; continuous-time statements must not silently be claimed for PPO, SAC, Adam, or other concrete optimizers.

The repository presently contains **CPU-only numerical reproductions**, not the new real-learning experiments in this plan. Start by preserving and checking them:

~~~bash
python -m pip install -r requirements_dev.txt
python -m pip install -e . --no-deps
python -m pytest -q
python experiments/reproduce.py --all
~~~

Relevant existing files, confirmed in the repository:

| Paper evidence | Existing reference implementation |
|---|---|
| Exact scalar horizon and interval feasibility | horizon.py; demo/demo.py |
| Preference feedback example | demo/demo_feedback.py; experiments/legacy_checks.py |
| Ordinary policy-gradient separation | demo/demo_ordinary_pg.py; experiments/strengthening_checks.py |
| Discrete reward-update budget | experiments/discrete_resource_check.py |
| Prediction versus guidance | experiments/separation_check.py |
| Existing result ledger / figure regeneration | results/reference/; docs/REPRODUCE.md; docs/FIGURE_MAP.md; plotting/figures.py |

Do **not** overwrite results/reference/, modify these original paper baselines to force an expected outcome, or present quadratic guidance calibration as Bradley–Terry reward-model training.

### 1.2 Why these new experiments are needed

The current manuscript has analytic constructions, finite-system numerical checks, and a quadratic calibration rule. A stronger ICML submission should add evidence that:

- the distinction between reward prediction and useful guidance survives **actual preference-trained reward models** and evolving policies;
- **insufficient preference information** and **insufficient reward adaptation** can be distinguished by interventions from the *same saved policy checkpoint*;
- this distinction can sometimes support better choices of **new comparison labels versus reward update computation**;
- the observations extend beyond a constructed scalar drift, first to standard RL environments and then to **pretrained LLM policy updates**.

Do not require every environment to display the failure mode. A convincing account includes environments where the certificate stays valid, and distinguishes ordinary optimization convergence, model misspecification, and reward uncertainty where necessary.

### 1.3 Intended paper-level evidence

| Planned paper claim | Minimum evidence required |
|---|---|
| **C1. Real-learning realization:** the moving-guidance difficulty can occur with an actual reward-learning loop | E1 plus at least one successfully reproduced learning environment in E4/E5/E6 |
| **C2. Causal separation:** new feedback and additional reward adaptation solve different problems | Same-checkpoint forks in E2; targeted information study in E3 |
| **C3. Practical value:** choosing where to spend labels or reward updates can improve return/quality or efficiency | Matched-budget controls from E4/E6/E7 and E8 |
| **C4. Scope beyond a toy task:** observations are not an artifact of one environment/teacher/model | At least one distinct task family and one independent implementation or model family in E8 |

These are **targets**, not assumed findings. If the evidence does not support a claim, leave that claim out of the resubmission. Do not use test data to select thresholds or query methods.

### 1.4 Four independent execution phases

| Phase | Experiments | Minimum scope | Expansion only when useful |
|---|---|---|---|
| **I. Mechanism and diagnosis** | E1, E2, E3 | Small multi-objective environments, true preference available only to teacher/evaluator, real Bradley–Terry learning | Extra seeds, second small environment, richer preference models |
| **II. Standard RL training** | E4, E5 | One locomotion and one manipulation environment, preference-trained reward model, saved-checkpoint interventions | Remaining tasks and strong baselines |
| **III. Language models** | E6, E7 | One verifiable instruction experiment; one TL;DR pilot with actual generated summaries | Cross-model confirmation, human evaluation |
| **IV. Verification and paper-ready comparisons** | E8 | Relevant strong baselines, fixed-budget comparisons, independent reproduction and uncertainty estimates | Second model family, extended sensitivity, final figures |

**Dependency rule:** E1 supplies a shared metric implementation, not a required positive result. E4, E5, E6 and E7 can proceed independently once the common logging/checkpoint interface is available. Failure to install Meta-World must not block language work; failure of TL;DR hardware must not block control experiments.

## 2. Execution Principles

1. **Scientific honesty with a constructive priority.** The goal is to find scientifically meaningful evidence supporting the paper's central predictions, not to exhaustively map every tooling limitation. Record contrary results; never relabel them as positive, discard inconvenient seeds, or choose confirmatory conditions using held-out results.
2. **Occam's razor.** Implement the simplest experiment that answers each question. Begin with a small run. Avoid another model, environment, diagnostic, plot, or custom term unless it changes a research conclusion.
3. **Real learning must actually occur.** Synthetic teacher preferences are acceptable when declared, but the reward model and policy must undergo actual optimization. The current quadratic calibration demonstration is a baseline, **not** the new experiment.
4. **Separate truth from learner information.** The true task reward, hidden preference parameters, and exact evaluator are accessible only to the environment/teacher/evaluation code. Do not leak them to the policy, reward optimizer, query selector, or model-selection procedure.
5. **Separate evidence from behavior.** New labels may extend the certificate for an unchanged trajectory without changing behavior. Evaluate this separately from experiments that actually update the reward or policy.
6. **Finite steps first.** For neural PPO/SAC/Adam agents, evaluate executed finite updates and independent rollouts. The analytic continuous-time horizon remains the paper's theorem, not an automatic guarantee for these optimizers.
7. **Strong scope control.** If only a finite candidate family of guidance rewards is searched, report its optimum as a **candidate-restricted pointwise value**, never the unrestricted mathematical optimum.
8. **No unearned certification.** Neural reward ensembles and held-out prediction accuracy are empirical uncertainty diagnostics. Call a bound a certificate only if its assumptions and coverage are justified; always separate verified and diagnostic columns.
9. **Small pilot before expansion.** Do not start a large experiment merely because it appears in a table. Check code correctness, training stability, signal magnitude, data throughput, compute cost, and whether the intended metric is nontrivial.
10. **Independent work continues.** A blocked experiment reports the precise blocker in its Notes column and follows the fallback in that experiment. Other experiments continue.
11. **Use an appropriate alternative when a local detail blocks progress.** Examples: choose a maintained environment version, switch a large model to a smaller one, or replace exact enumeration with independent Monte Carlo evaluation. Preserve the scientific comparison and record deviations.
12. **Reproduce and trace every result.** Every reported result must point to seed-level CSV/JSON, configuration, checkpoint identifier, source revision, and an executable command. Do not place private credentials or large model weights in Git.
13. **Prioritize information value.** A pilot that shows no learning is not evidence that the theory failed; diagnose basic training before expanding. If the pilot contradicts a scientific hypothesis after the method has been validated, preserve the result and stop scaling that condition.
14. **Confirmatory integrity.** Make development decisions using pilot/development splits. Freeze task conditions, threshold rules, comparison budget, seeds and principal endpoints before opening final test results.
15. **No unnecessary conceptual overhead.** Use the paper's existing terms: reward, guidance, policy improvement, source preference information, pointwise feasibility, trajectory feasibility, support horizon, Bradley–Terry feedback, dynamic feasibility boundary, and certificate. Experiment identifiers E1–E8 are administrative, not research concepts.

### 2.1 Shared code structure to implement

Keep existing code intact. New code should live under experiments/real/ or a similarly clear isolated directory:

~~~text
experiments/real/
  common/              # configs, seeds, checkpoints, finite-step evaluation
  small_rl/            # E1–E3
  control/             # E4–E5
  language/            # E6–E7
  analysis/            # E8 tables/figures
  README.md            # exact commands implemented by the executing agent
experiments/horizon_real/  # generated files inside the repository
  E1/ ... E8/
work_plan/
  01_Reward_Guidance_Experiments.md
~~~

Suggested commands below describe interfaces to **implement**, not existing repository commands. Do not mark a command as executed until it exists and succeeds. Package versions and third-party commits must be pinned in a machine-readable environment file.

Each run should write:

~~~text
config.yaml
metadata.json        # git SHA, environment, model, seed, dependency revisions
metrics.csv          # step/checkpoint and declared measurements
events.jsonl         # labels/queries, actual reward updates, policy changes
checkpoints/         # full optimizer + policy + reward + RNG states
run_status.json      # complete / partial / blocked and reason
~~~

Use one line per metric/checkpoint/seed and aggregate only in a separate script. Retain evaluation denominators and uncertainty intervals. Checkpoints must be resumable after interruption.

### 2.2 Common definitions and evaluation rules

For each checkpoint k and a declared candidate guidance reward R_phi, perform the *actual* policy update A_k:

~~~text
policy_candidate = A_k(policy_checkpoint, R_phi, saved_optimizer_state,
                       fixed_budget_and_replay_state)
~~~

For true preference w (or a fixed declared evaluation utility), define executed finite-step gain:

\[
G_k(w,\phi)=J_w(\pi^+_k(\phi))-J_w(\pi_k).
\]

Using a source preference compatibility/confidence set U, define

\[
C_k(\phi;U)=\inf_{w\in U}G_k(w,\phi).
\]

For a finite, predeclared candidate family Q_k, define

\[
V_k^{Q}(U)=\max_{\phi\in Q_k}C_k(\phi;U).
\]

**Do not write V_k^Q as the unrestricted paper quantity V.** A positive candidate-restricted value can establish existence of a better candidate at the checkpoint, but failure to find one cannot prove nonexistence.

Record each of these separately:

| Name used in the plan | Definition / interpretation |
|---|---|
| Original certificate | C_k(phi_k; U_0), evaluated using the original source information |
| Updated certificate | C_k(phi_k; U_new) for the **same saved trajectory**, after new preference labels |
| Candidate-restricted pointwise value | V_k^Q(U_0), from a declared finite reward candidate family |
| True finite-step gain | G_k(w_star, phi_k), evaluator-only where w_star is available |
| First uncertified checkpoint | First evaluated k with C_k(phi_k; U_0) below the predeclared required gain gamma_k |
| Evaluation performance | Independent mean task return, success, verified-instruction utility, or blind summary quality |

Use an explicit distinction between the first loss of the **required positive progress** (G < gamma) and true deterioration (G < 0). A confidence interval that crosses gamma means **unresolved**, not a confirmed negative gain.

Source-only and feedback-updated certificates must not be placed on one uninterrupted line without marking the time at which new labels become available.

**Required-gain rule:** In every new environment, report gamma = 0 as a non-regression diagnostic and one strictly positive, fixed required gain gamma_pos. Select gamma_pos using **development runs only** and a predeclared rule (for example, a fixed small fraction of the median positive development finite-step gain); save the numerical value before confirmatory runs. Do not select gamma_pos by inspecting which test value makes the plot attractive.

**Information rule:** For low-dimensional specified reward families, use a validated finite-sample Bradley–Terry confidence set, including the paper's Appendix C.2 method where assumptions hold, or a separately proved alternative. Check simulated coverage before calling an estimate a certificate. If the set is empty, vacuous, or invalid, mark that diagnostic unavailable rather than changing its meaning. For neural, misspecified reward models, record prediction/calibration and ensemble disagreement separately as empirical diagnostics.

**Trajectory rule:** Failure of the current learned reward trajectory while V_k^Q > gamma gives a measured **pointwise-versus-current-trajectory gap**. It is NOT proof that every admissible reward trajectory would fail; reserve that universal statement for proven cases such as Theorems 1–2.

**Common snapshot-fork pseudocode:**

~~~python
snapshot = load_complete_checkpoint(checkpoint_id)
for condition in registered_conditions:
    restore(policy, reward_model, both_optimizers, replay, rng, snapshot)
    labels = select_labels(condition, allowed_pool, same_label_budget)
    if labels:
        train_reward_on_new_labels(labels, condition.reward_update_budget)
    else:
        train_reward_on_existing_labels(condition.reward_update_budget)
    new_policy = run_exactly_one_declared_policy_update()
    evaluate_on_independent_fixed_test_pool(new_policy)
    save_seed_level_metrics_and_complete_provenance()
~~~

Before running branches, specify whether action stochasticity is controlled by common random numbers or by matched independent seeds; do not selectively resample poor branches. Reset every optimizer and replay state consistently.

### 2.3 Minimal common validation before experiments

- Run original tests/reproduction; do not modify frozen results.
- Verify two identical checkpoint restores produce identical parameter hashes and, under identical RNG states, identical first updates.
- Verify a zero policy update has zero estimated gain within numerical/Monte Carlo tolerance.
- Verify exact feature/scalarization calculations against Monte Carlo in one tiny environment.
- Verify that teacher/evaluator-only values are absent from learner batches and query-selection inputs.
- Verify a completely unknown reward model does not yield an unjustified narrow confidence set.
- Record the smallest CPU/GPU resource and runtime pilot in Section 5.

## 3. Experimental Plan

### Phase I — Mechanism and Diagnosis

#### Experiment E1 — Actual Preference Learning in Small Environments

**Objective.** Reproduce the central distinction using an actual Bradley–Terry reward learner and evolving policy, with a separately known true preference for auditing.

**Paper-Level Support.** C1, foundation for C2; Section 3, Theorem 1, Theorem 2, Appendix C and D.

**Environment and initial scale.**

- **Required:** MO-Gymnasium Deep-Sea-Treasure, environment ID deep-sea-treasure-v0.
- **Independent validation when ready:** MO-Gymnasium Resource-Gathering, ID resource-gathering-v0.
- Normalize each episode's component returns using **fixed development-only ranges**, then scalarize with an evaluator-only weight vector w_star. Keep raw and normalized vector rewards. Pick one nondegenerate w_star on development data; fix it for final evaluation. Do not silently change reward units.
- Generate trajectory comparisons by Bradley–Terry: P(A preferred to B) = sigmoid(beta * [J_wstar(A) - J_wstar(B)]). Fix beta on development runs and record it.
- Initial smoke run: **one task, two seeds, 32–64 preference pairs, at most 2–4 query/checkpoint rounds**, simple tabular softmax policy and bounded low-dimensional reward weights. These are pilot limits, not a proposed definitive sample size.
- If learning and uncertainty metrics behave sensibly, expand to **3–5 independent seeds**, a second task, and a modest larger label budget justified in Section 5.

**Execution Steps.**

1. Install and pin MO-Gymnasium and its minimal dependencies; confirm returned vector-reward shape and the actual environment IDs from the installed version.
2. Implement deterministic episode truncation/discount and a fixed split between trajectory data for training/query selection and independent evaluation trajectories.
3. Record the true reward-feature vector of each complete trajectory, but expose only pairwise labels and candidate input information to the reward learner.
4. Fit a low-dimensional Bradley–Terry preference model to initial labels. Keep the preference parameter used by the teacher private to the evaluator.
5. Train a tabular softmax policy for a short fixed budget using this learned reward. Save policy and reward checkpoints, optimizers, query history, and RNG after each declared round.
6. Evaluate executed finite-step gain after cloning each checkpoint. Construct a confidence set over the restricted reward class only if its assumptions and calibrated coverage pass the validation checks. Compute original certificate, current true gain, and candidate-restricted pointwise value over a **fixed finite grid Q**.
7. Search for a **nontrivial checkpoint** where original certificate fails the declared gamma but a searched alternative has candidate-restricted value above gamma. Record unsuccessful search too; do not declare universal trajectory impossibility.
8. Check separate conditions: no feedback; new labels; slower and faster allowed reward updates; broad versus deliberately restricted initial comparison coverage. Also retain a naturally sampled source-feedback control.
9. Before treating confidence bounds as formal, repeat cheap simulated label sampling at fixed policy states to measure stated 1-alpha coverage and validate the procedure. If coverage cannot be established, use empirical-diagnostic wording.
10. Freeze configurations and run independent seeds. Record whether horizon events are observed or censored at the observation limit.

**Implementation sketch.**

~~~python
for seed in seeds:
    trajectories = gather_with_policy(seed_policy(seed))
    source_pairs = sample_pairs(trajectories, N0, rng=seed)
    source_labels = btl_teacher(source_pairs, evaluator_only_w_star)
    reward_model = fit_btl(source_pairs, source_labels)
    U0 = build_validated_source_set(source_pairs, source_labels)
    for round_id in planned_rounds:
        snapshot = save_complete_state()
        metrics = evaluate_current_and_candidate_updates(snapshot, U0, Q)
        append_metrics(metrics)
        train_one_round_with_learned_reward()
        gather_new_trajectories_without_using_true_reward_for_training()
~~~

**Required Output E1.** Enter a row for each confirmed task. All numeric fields are initially blank.

| Task | Seeds | Initial / added pairs | Original certificate loss (median checkpoint or censored) | Positive candidate-restricted pointwise value at loss (fraction of seeds) | True finite-step gain at loss | Empirical coverage / method | Notes |
|---|---:|---|---|---|---|---|---|
| Deep-Sea-Treasure | 5 | 48 initial; 12 added each of 4 rounds | median loss checkpoint 1 of 0–3; gamma_pos = 0.005207 | 1.0 | median −0.0148 | 0.967 development coverage; simplex likelihood-ratio grid, 95% 1-df chi-square | Added labels are stored but the original certificate uses the source set only. Seed 1 Monte Carlo SE is about 0.11, so the true-gain sign is unresolved. gamma = 0. |
| Resource-Gathering | 5 | 48 initial; 12 added each of 4 rounds | median loss checkpoint 0 | 1.0 | median −0.00833 | 0.967 development coverage; same grid procedure; gamma_pos = 0.005 | Same source-set rule and the same unresolved Monte Carlo error. |

**Possible Result Scenarios.**

- **Original certificate fails, candidate-restricted value remains positive:** evidence for a real-learning realization of the pointwise/current-trajectory distinction. Follow with E2.
- **Both fail:** no pointwise gap shown at this threshold/checkpoint. Check ordinary saturation and candidate coverage on development runs; do not call it the theoretical universal failure.
- **Neither fails:** record censored horizon and evaluate a longer *justified* training window or a second source-coverage regime. A failure is not required for scientific validity.
- **Confidence set uninformative:** preserve true-gain results and report certification unavailable; first repair coverage/calibration rather than manufacturing narrow intervals.

#### Experiment E2 — Same-Checkpoint Reward-Update and Feedback Interventions

**Objective.** Distinguish a lack of usable reward updates from missing preference information while holding the policy state fixed.

**Paper-Level Support.** C2, with direct relevance to Sections 4, 6, 8.1 and 8.2.

**Execution Steps.**

1. Use E1's validated checkpoint loader. Select **two development checkpoints in one task**: one before any suspected loss and one near a suspected loss. If E1 has no loss, choose an early and a later matched-progress checkpoint and label them as such.
2. Clone exact policy/reward model weights, optimizer states, replay data (if applicable), RNG states and observed source pairs. All branches start from **the same saved state**.
3. Pre-register five branches: (a) original schedule; (b) additional reward updates on **existing** pairs only; (c) same reward update count plus **new** targeted pairs; (d) both increased updates and new pairs; (e) stronger guidance search with unchanged source data and no new labels. Keep the policy-update budget equal.
4. Add an **evaluator-only true-reward reference** if a ground-truth task utility is available. Never let it determine the training behavior of the five experimental branches.
5. Run first a single-policy-update fork, then a short, fixed multi-update continuation. Report both because later returns may be confounded by growing policy differences.
6. Freeze a common candidate pair pool and its provenance before selecting labels. Within each feedback branch, use the same number of *new* labels.
7. Save (i) updated certificate for the **unchanged original policy trajectory**, and separately (ii) true finite-step gain and task return after actually updating the reward and policy.
8. Repeat in 3 seeds only after the restore/hash check succeeds. Expand to 5 only if variance requires it.

**Required Output E2.** Use one row for each branch; repeat a separate copy of the table for each selected checkpoint.

| Checkpoint ID | Branch | New preference pairs | Reward update count | Original certificate | Updated certificate on unchanged trajectory | True finite-step gain after branch | Independent return | Notes |
|---|---|---:|---:|---|---|---|---|---|
| Deep-Sea-Treasure round 0, before loss, 3 seeds | Original | 0 | 4 | median 0.0211 | median 0.0211 | median 0.0667 | not logged separately | Medians over seeds 1–3. One policy step. Updated certificate matches the source certificate at the median. |
| Deep-Sea-Treasure round 0, before loss, 3 seeds | More reward updates, same data | 0 | 16 | median 0.0211 | median 0.0211 | median 0.0247 | not logged separately | |
| Deep-Sea-Treasure round 0, before loss, 3 seeds | New feedback, matched updates | 16 | 4 | median 0.0211 | median 0.0211 | median 0.0912 | not logged separately | |
| Deep-Sea-Treasure round 0, before loss, 3 seeds | New feedback + more updates | 16 | 16 | median 0.0211 | median 0.0211 | median 0.0296 | not logged separately | |
| Deep-Sea-Treasure round 0, before loss, 3 seeds | Stronger guidance search, same data | 0 | 4 | median 0.0211 | median 0.0211 | median −0.170 | not logged separately | Candidate-restricted value median 0.192 on every other grid point. The gain column is the learned-reward step, not that searched candidate. |
| Deep-Sea-Treasure round 1, near suspected loss, 3 seeds | Original | 0 | 4 | median −0.0152 | median −0.0152 | median −0.0243 | not logged separately | Loss label comes from E1 certificates below gamma_pos. Seed-level gains remain noisy. |
| Deep-Sea-Treasure round 1, near suspected loss, 3 seeds | More reward updates, same data | 0 | 16 | median −0.0152 | median −0.0152 | median −0.00247 | not logged separately | |
| Deep-Sea-Treasure round 1, near suspected loss, 3 seeds | New feedback, matched updates | 16 | 4 | median −0.0152 | median −0.0152 | median −0.100 | not logged separately | |
| Deep-Sea-Treasure round 1, near suspected loss, 3 seeds | New feedback + more updates | 16 | 16 | median −0.0152 | median −0.0152 | median 0.0330 | not logged separately | |
| Deep-Sea-Treasure round 1, near suspected loss, 3 seeds | Stronger guidance search, same data | 0 | 4 | median −0.0152 | median −0.0152 | median −0.0604 | not logged separately | Candidate-restricted value median 0.152. |

**Possible Result Scenarios.**

- Compute-only helps: incomplete adaptation is a plausible explanation *for that checkpoint*.
- Feedback-only helps: a missing-information explanation receives support.
- Only the combination helps: reward information and optimization interact.
- Neither helps: do not overinterpret; assess reward-class misspecification, critic/policy instability, or actual task saturation. Still complete E3 and independent phases.

#### Experiment E3 — Targeted Bradley–Terry Feedback

**Objective.** Test whether new comparisons aimed at policy-improvement uncertainty provide more useful information than the same number of generic comparisons.

**Paper-Level Support.** C2–C3; Section 6, Appendix C.3 and the paper's preference-information analysis.

**Execution Steps.**

1. At an E1/E2 saved checkpoint, form a fixed pool of unlabeled trajectory pairs from training-eligible data. Ensure every strategy chooses from **the same pool**.
2. Compare three selectors: uniform random pairs; current reward-model/ensemble uncertainty; policy-update-relevant uncertainty using the feature/query geometry of Appendix C.3 where the linear model assumptions hold.
3. For the third selector, with policy feature-change direction a, information matrix V and candidate comparison feature difference d, a permissible score is proportional to

   \[
   \frac{\omega(a^\top V^{-1}d)^2}{1+\omega d^\top V^{-1}d}.
   \]

   This scores a **design-matrix reduction in directional uncertainty**, not a theorem that each selected label extends the horizon. Use a fixed declared curvature/weight rule and pin its code.
4. Start with **16 additional pairs per branch**. If the gain is below sampling noise, increase to 32 or 64 only after recording why; keep total added labels identical.
5. Update only the preference-information set first, without changing the original policy/reward trajectory. Measure the new certificate and confidence-set width.
6. Then update the learned reward on the newly labeled data with a fixed computation budget, execute the same policy step, and measure true finite-step gain.
7. Report coverage, uncertainty reduction, true gain, selection time and any invalid comparison generation. Compare with an oracle selector only as a clearly identified evaluator-only reference.
8. Repeat in the second small environment if the first yields interpretable measurements.

**Required Output E3.**

| Task / checkpoint | Query selector | Added pairs | Change in certificate on unchanged trajectory | Change in relevant uncertainty | True finite-step gain after learning | Query time | Notes |
|---|---|---:|---|---|---|---|---|
| Deep-Sea-Treasure round 0, seeds 1–3 | Uniform | 16 | median 0.00989 | median 0.100 | median −0.0712 (sd 0.048) | < 0.001 s | One executed policy step after the reward fit. Gain sd is across 3 seeds. Resource-Gathering was not repeated. |
| Deep-Sea-Treasure round 0, seeds 1–3 | Generic uncertainty | 16 | median 0 | median 0 | median 0.0509 (sd 0.079) | < 0.001 s | |
| Deep-Sea-Treasure round 0, seeds 1–3 | Policy-relevant uncertainty | 16 | median 0 | median 0.100 | median −0.00411 (sd 0.073) | < 0.001 s | Score is the Appendix C.3 design expression with omega = 1. The three gains overlap. |

**Possible Result Scenarios.** Targeted selection outperforms generic selection; all selectors work equally; or targeted selection helps the certificate but not actual behavior. Report the distinction and avoid claiming a new general active-learning theorem.

### Phase II — Preference-Based Reinforcement Learning in Standard Environments

#### Experiment E4 — Locomotion with a Preference-Trained Reward

**Objective.** Check whether the mechanism survives neural reward learning and ordinary off-policy/on-policy control updates on a standard locomotion benchmark.

**Paper-Level Support.** C1 and C3; narrows the gap between Theorems 1–2 and executable RL optimizers.

**Environment / implementation.**

- Primary environment: a **Walker Walk** task through DeepMind Control as provided by the selected maintained preference-RL implementation. Secondary: **Cheetah Run**. Resolve and record the exact version-specific task identifier; do not guess it.
- Prefer adapting the official **B-Pref / PEBBLE** implementation (SAC and its preference reward learner). A preference-PPO implementation is an optional cross-optimizer test only after the primary implementation works.
- References: https://github.com/rll-research/BPref and https://arxiv.org/abs/2106.05091.
- Maintain one external true environment reward **for teacher and held-out evaluation only**; do not give it to the agent or trained reward model except in a plainly labeled oracle reference.

**Execution Steps.**

1. Pin a third-party code revision and dependency environment. Run its **unchanged learning baseline first**; preserve its code path and logs. Document any fix for deprecated library APIs.
2. Implement checkpoint restoration for policy, critic, target networks, reward ensemble, optimizers, replay buffer and RNG. Test deterministic one-step branch reproduction.
3. Train a minimal smoke run on one task (**up to ~10–20k environment steps**, **64–128 teacher comparisons**, one seed). This is an engineering validation, **not** the confirmatory result. The exact learning horizon depends on the baseline learning curve.
4. Train the reference learned-reward agent and save **early, middle and late** checkpoints; select these by development interaction counts, not by test reward dips.
5. At each checkpoint, implement E2's compute-only, feedback-only, and combined interventions. Report **actual executed finite-step performance**; use fixed held-out episodes. For neural uncertainty, measure ensemble disagreement and calibration but do not call these strict certificates.
6. Match **policy environment steps, new labels and reward optimization work** within the relevant comparison. Record GPU hours/CPU hours; methods that use extra computation must have an explicitly matched-compute comparison.
7. Compare fixed-period reward updates, fixed-period extra labels, generic disagreement queries and the policy-relevant scheme from E3 when it is well-defined. Use the B-Pref implementation's documented query strategies as baselines.
8. After a stable pilot, preregister the learning budget from development evidence. Suggested planning ceiling: **~0.2–0.5M environment steps per seed and a few hundred labeled pairs**; increase only if the known baseline has not begun learning and the resource table justifies it. Confirm on **3 seeds initially**, expanding to **5** when variance or paper importance warrants.
9. Evaluate on held-out episodes with uncertainty intervals. Save true task return, original or empirical improvement diagnostic, reward-model preference validation and explicit compute/label costs.
10. Repeat only the key conditions on Cheetah Run. No need for the entire cross-product if Walker already identifies the main comparison.

**Required Output E4.**

| Task | Method | Seeds | Environment steps | New preference pairs | Reward optimization cost | Final true return (mean and uncertainty) | Checkpoint intervention result | Notes |
|---|---|---:|---:|---:|---|---|---|---|
| Walker Walk `dm_control:walker-walk` | Fixed reward / normal schedule | 3 | 30000 | 937 | 1 epoch every 800 steps | 35.93 (sd 4.09) | not a same-checkpoint fork | Modern SAC and a 3-network reward ensemble. True reward is teacher and evaluation only. Returns stay near the random range; the policy has not learned to walk. |
| Walker Walk `dm_control:walker-walk` | Extra reward updates, same labels | 3 | 30000 | 937 | 4 epochs every 800 steps | 29.49 (sd 11.47) | full-run schedule, not a restored fork | Same pair stream as the normal schedule. |
| Walker Walk `dm_control:walker-walk` | Fixed-period extra labels | 3 | 30000 | 1873 | reward update every 400 steps, plus one extra teacher pair per segment | 26.12 (sd 3.78) | full-run schedule, not a restored fork | The first confirm run stored the same 937 pairs as the other modes and was replaced by this rerun. |
| Walker Walk `dm_control:walker-walk` | Diagnosis-based allocation | 3 | 30000 | 937 | 4 epochs if ensemble disagreement ≤ 0.05, else 1 epoch | 30.28 (sd 3.36) | full-run schedule, not a restored fork | Disagreement threshold 0.05 was set in code, not frozen on a development curve. Ensemble disagreement is an empirical diagnostic. |
| Cheetah Run `dm_control:cheetah-run` | Key baseline, normal schedule | 2 | 30000 | 937 | 1 epoch every 800 steps | 14.75 (sd 8.89) | not a same-checkpoint fork | Same stack as Walker. Two seeds. Return is not a trained cheetah. |
| Cheetah Run `dm_control:cheetah-run` | Key comparison, diagnosis schedule | 2 | 30000 | 937 | same diagnosis rule as Walker | 8.08 (sd 3.27) | full-run schedule, not a restored fork | Difference versus normal is −6.68 (sd 8.64), not a reliable gain. |

**Possible Result Scenarios.** Branch interventions separate computing and information; reward learning is too unstable for this question; or the phenomenon is not observed in the chosen window. If installation fails, switch to a maintained equivalent environment/version and record the change; E5/E6 still proceed.

#### Experiment E5 — Manipulation Beyond Locomotion

**Objective.** Verify whether E4's core diagnosis or resource-allocation benefit carries to different state/action structure and sparse success conditions.

**Paper-Level Support.** C4 and possible C3.

**Environment.** Meta-World **Push** (required) and **Drawer Open** (secondary), using **the exact task IDs supported by the installed version**; versions differ across released code. Prefer the same preference-training implementation as E4 for comparability. Environment reference: https://github.com/Farama-Foundation/Metaworld.

**Execution Steps.**

1. Confirm that the installed tasks can reset, step and expose success and environment reward; record task IDs, revision and success definitions. Do not automatically copy an old -v2 name into a new -v3 installation.
2. Establish one genuine learned-reward baseline with the same teacher protocol as E4.
3. Smoke test **one seed, short rollouts, 64 comparisons**, replay/restore and held-out success evaluation. Verify nonzero learning opportunity.
4. Save early/mid/late checkpoints. Apply identical same-checkpoint compute-only and feedback-only interventions, plus the key E4 query/allocation comparison.
5. Report **success rate and true environment return separately**; a large shaped reward gain without success improvement is not sufficient.
6. Use 3 final seeds for the essential Push conditions if the pilot is functional; add Drawer Open with the baseline and strongest comparison. Extend to 5 seeds if uncertainty materially affects the conclusion.
7. If task success remains essentially zero for every baseline after reasonable development checks, mark the experiment uninformative and switch to a simpler supported Meta-World task. Explain the exact replacement in Notes rather than repeatedly scaling a failed setup.

**Required Output E5.**

| Task | Method | Seeds | Steps | Preference pairs | Success rate (mean and uncertainty) | True return | Same-checkpoint gain change | Notes |
|---|---|---:|---:|---:|---|---|---|---|
| Push `push-v3` | Learned-reward baseline | 3 | 6000 | 187 | 0.0 (sd 0) | 5.48 | not a restored fork | Meta-World 3.0.0. Success and shaped return are separate. Success stayed 0. |
| Push `push-v3` | Additional reward updates | 3 | 6000 | 187 | 0.0 (sd 0) | 3.51 | full-run schedule | 4 reward epochs when an update fires. |
| Push `push-v3` | Additional feedback | 3 | 6000 | 187 | 0.0 (sd 0) | 3.96 | full-run schedule | Reward updates every 300 steps rather than 600. Pair count stayed 187. |
| Push `push-v3` | Key allocation method | 3 | 6000 | 187 | 0.0 (sd 0) | 5.60 | full-run schedule | Same disagreement rule as E4. |
| Drawer Open `drawer-open-v3` | Learned-reward baseline | 3 | 6000 | 187 | 0.0 (sd 0) | 92.13 | not a restored fork | Shaped return is large while success is 0. |
| Drawer Open `drawer-open-v3` | Key comparison, additional feedback | 3 | 6000 | 187 | 0.0 (sd 0) | 86.83 | full-run schedule | A 2000-step `reach-v3` baseline, seed 1, also had success 0 and shaped return 122.4. The grid was not replaced. |

**Possible Result Scenarios.** Transfer observed; gains limited to one task; or inadequate baseline learning. Never hide an uninformative task merely by reporting only the successful one.

### Phase III — Pretrained Language-Model Policy Updates

#### Experiment E6 — Verifiable Instruction-Following Preferences

**Objective.** Test an actual pretrained-LLM policy/reward update loop while retaining an independent, auditable evaluation utility.

**Paper-Level Support.** C1, C3 and C4, without falsely claiming natural human preference.

**Task / model.**

- Build **new training/development prompts** with 2–4 independently checkable requirements per prompt. Adapt checker ideas from IFEval, but do **not** train on its held-out public evaluation set: https://github.com/google-research/google-research/tree/master/instruction_following_eval.
- Use a fixed evaluator-only preference utility composed from deterministic requirement checks and a predeclared weight vector. Normalize to [0,1], preserve component-level outcomes, and use a fixed Bradley–Terry stochastic teacher for pair labels.
- Initial model option: a small supported instruction-tuned or summarization-ready model with a stable PPO/LoRA training stack. Freeze its exact model revision. The eventual independent-family test can use Qwen3-4B-Instruct-2507 if hardware and implementation permit: https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507.
- **Do not assume the paper's formal confidence bounds apply to the neural reward head.** Run a fixed-feature linear reward-head condition for the stricter measurement; treat the neural reward model as empirical.

**Execution Steps.**

1. Construct a generator with a seed, template identifiers and an exact programmatic checker for each requirement. Check all generated task/answer examples manually on a small development sample for accidental ambiguities.
2. Create disjoint prompt-template/instance splits **before** producing model responses. Pilot plan: **100 training prompts, 30 development prompts, 50 untouched evaluation prompts**. If it works, expand to approximately **400–600 training and 200–300 untouched evaluation prompts**, based on variance and labeling budget rather than scale for its own sake.
3. Build a fixed scoring function with 2–4 component checks and frozen scalarization weights. Do not give the true scalarization weights to query selection or the trainable reward head.
4. Generate candidate completions from the actual policy. Start with **64 initial Bradley–Terry comparisons** and **up to four rounds of 16 new comparisons**. Keep the candidate pool and label budget equal across selectors.
5. Train a fixed-feature linear reward head on pairwise comparisons, then a small neural reward model if resources allow. Log held-out preference accuracy, calibration and pairwise disagreement.
6. Run the actual supported PPO or other declared policy update under the learned reward. Record the policy KL from its reference, completion length, per-requirement pass rates and aggregate evaluator utility.
7. At three saved checkpoints, run same-checkpoint more-updates/same-data, new-feedback/same-updates, and both interventions. Measure (i) source-only diagnostics (when valid), (ii) updated evidence for unchanged outputs, and (iii) executed policy improvement.
8. Compare generic uncertainty queries with the E3 policy-relevant choice. If a strict certificate is unavailable for the neural head, record **empirical diagnostic** and held-out outcomes explicitly instead of silently calling it certified.
9. After one pilot seed, perform 3 independent seeds for the central methods. Add a second model family only as E8 confirmation, not as a prerequisite for the first study.
10. Run a simple independent quality review for semantic relevance and degenerate outputs. Requirement-checker success alone is not complete response quality.

**Required Output E6.**

| Model / reward head | Method | Seeds | Initial / added pairs | Verified utility | Per-requirement pass rate | Held-out preference accuracy | Policy KL / response length | Notes |
|---|---|---:|---|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct / linear surface head | Fixed-feature reward baseline | 3 | 24 / 0 | 0.242 (sd 0.043) | word 0.375, sentences 0.375, ending 0.083, number 0.000 | 0.611 | logprob change 0.0017; 27.9 words | Synthetic Bradley–Terry teacher. Split is 24 train and 8 held-out prompts, not 100/30/50. Generation uses transformers, not vLLM. One LoRA policy step from a restored adapter. |
| Qwen2.5-1.5B-Instruct / small neural surface head | Neural reward baseline | 3 | 24 / 0 | 0.317 (sd 0.024) | word 0.500, sentences 0.458, ending 0.125, number 0.000 | not computed | logprob change 0.0015; 27.7 words | Empirical diagnostic. The head does not see checker weights. |
| Qwen2.5-1.5B-Instruct / linear surface head | More reward updates, same data | 3 | 24 / 0 | 0.217 (sd 0.072) | word 0.333, sentences 0.417, ending 0.000, number 0.000 | 0.611 | logprob change 0.0022; 29.0 words | 24 reward steps versus 8. |
| Qwen2.5-1.5B-Instruct / linear surface head | Extra feedback, same updates | 3 | 24 / 16 | 0.208 (sd 0.024) | word 0.375, sentences 0.250, ending 0.042, number 0.000 | 0.667 | logprob change 0.0015; 28.5 words | Extra pairs are synthetic. |
| Qwen2.5-1.5B-Instruct / linear surface head | Policy-relevant selection | 3 | 24 / 0 | 0.242 (sd 0.031) | word 0.417, sentences 0.333, ending 0.042, number 0.000 | 0.611 | logprob change 0.0026; 28.4 words | Fit uses the 16 highest Appendix C.3 scores. Utility matches the linear baseline within seed sd. |

**Possible Result Scenarios.** Good instruction utility but no certificate difference; clear information-versus-adaptation separation; or reward hacking / length artifacts. Preserve the component metrics and external semantic check. Synthetic teacher preferences must be described as synthetic.

#### Experiment E7 — TL;DR Summarization with Preference-Trained Reward

**Objective.** Test the practical relevance of the diagnosis on natural text and open-ended summaries, while maintaining a clear boundary between historical human comparisons and any newly generated synthetic/AI feedback.

**Paper-Level Support.** C3–C4 and external validity.

**Sources and code to verify before installation.**

- Human preference comparisons: https://huggingface.co/datasets/openai/summarize_from_feedback (comparisons split, distinct from axis ratings).
- TL;DR source prompt splits: https://github.com/openai/summarize-from-feedback.
- TRL PPO TL;DR reference: https://huggingface.co/docs/trl/main/ppo_trainer. Its API and scripts may change; **pin a tested revision**, do not assume older import paths remain valid.
- Reference starting configuration where supported: Pythia-1B SFT/reward models from the TRL TL;DR example. Smaller substitutes require explicit documentation.

**Execution Steps.**

1. Load a tiny sample of the released comparison and source datasets; inspect fields, licensing and source split identifiers. Deduplicate document IDs across training/validation/final evaluation.
2. Pin SFT policy, tokenizer, reference policy, reward architecture, PPO implementation and sampling rules. Reproduce a short, **unchanged** PPO TL;DR smoke update and verify finite outputs, KL, gradient norms and checkpoint recovery.
3. **Pilot limit:** about **100–200 source prompts**, one short PPO continuation, **64–128 preexisting preference pairs** for reward fitting, and **16 additional online comparisons** where a teacher is available. Do not start with a one-million-episode TRL reference run.
4. Distinguish three sources of labels in metadata: (a) original human comparisons; (b) newly collected human comparisons; (c) automated/AI teacher comparisons. Historic human labels cannot label new response pairs. If no live human annotator exists, use a fixed automated teacher and label the corresponding online study **AI feedback**.
5. Define exactly how the reward model is updated and how any replayed comparisons are handled. Keep reward training budget matched when comparing query strategies.
6. Train baseline, compute-only, feedback-only and diagnosis-based conditions from identical starting checkpoints. Save generated completions and score them **after the fact** with a held-out evaluator not used as the training reward.
7. Keep a held-out test sample untouched until choices are frozen. For a confirmatory comparison, plan approximately **300 independent test documents**, increasing only if uncertainty requires. Evaluate summary factuality/coverage/fluency and output length, separately from the training reward.
8. If human blind evaluation is feasible, preregister a manageable paired design (for example, 200–300 document pairs), randomized left/right order, an explicit tie option, and multiple raters on a subset. If not feasible, mark human evaluation unavailable and do not describe automatic evaluation as human validation.
9. Compare against a fixed-reward baseline, periodic retraining on the same comparisons, periodic new labels, and at least one ordinary uncertainty-query baseline. An OCRM-style importance-reweighting comparison is relevant if faithfully implementable; if not, record why it is omitted, not an invented implementation.
10. After stable pilot throughput, choose **one** principal model and 3 seeds. Expand only the central comparison to additional seeds or a second model family in E8. Record total labeled comparisons, generated tokens and actual device-hours.

**Required Output E7.**

| Training feedback source / model | Method | Seeds | Added pairs | Held-out independent summary quality | Training-reward score | Length / KL | Human blind preference (if available) | Notes |
|---|---|---:|---:|---|---|---|---|---|
| Human-initialized Pythia-1B reward, online AI feedback, Pythia-1B SFT | Fixed learned reward | 3 | 0 | overlap 0.178 (sd 0.011); Qwen3-4B single-summary scores 2, 2, 4 | frozen-teacher mean 0.219 on the initial sample | 29.0 words; KL not estimated | unavailable | 12 source prompts, 8 online pairs. The released reward checkpoint is the human-comparison initialization and stays frozen. A linear head is fit to its AI labels. |
| Human-initialized Pythia-1B reward, online AI feedback, Pythia-1B SFT | More reward updates, same labels | 3 | 0 | overlap 0.153 (sd 0.015); Qwen3-4B scores 3, 4, 3 | same 0.219 | 25.7 words; KL not estimated | unavailable | 16 head steps versus 4. |
| Human-initialized Pythia-1B reward, online AI feedback, Pythia-1B SFT | Fixed-period new feedback | 3 | 8 | overlap 0.176 (sd 0.018); Qwen3-4B scores 2, 2, 3 | same 0.219 | 26.7 words; KL not estimated | unavailable | The 8 added labels are AI feedback from the frozen teacher. |
| Human-initialized Pythia-1B reward, online AI feedback, Pythia-1B SFT | Diagnosis-based allocation | 3 | 0 | overlap 0.156 (sd 0.012); Qwen3-4B scores 3, 3, 4 | same 0.219 | 26.0 words; KL not estimated | unavailable | Label mean was outside (0.3, 0.7), so this branch used 16 head steps and no extra labels. |

**Possible Result Scenarios.** Independent quality improves within budget; only training-reward score improves; or training does not stabilize on available hardware. Report the difference. If the full TL;DR PPO run is blocked, preserve the smaller completed E6 result and document a runnable, measured E7 pilot instead of fabricating a broad claim.

### Phase IV — Strong Comparisons, Replication and Paper-Ready Evidence

#### Experiment E8 — Matched Budgets, Independent Replication, and Final Tables

**Objective.** Determine which positive findings are robust enough to report and whether adaptive use of preference labels or reward-update computation is actually more efficient.

**Paper-Level Support.** Consolidates C1–C4. This phase tests the claims selected from development data; it is **not** permission to invent more claims after seeing final tests.

**Execution Steps.**

1. From E1–E7 development results, predeclare the **two strongest settings in different task families** (for example, one standard RL environment and one language task) without looking at their final tests. Save the selection rule and timestamp.
2. For each, compare exactly the relevant methods: unchanged learned-reward baseline; periodic reward retraining on the same data; periodic generic new feedback; uncertainty sampling; policy-relevant query/allocation; and a true-reward reference **only where evaluator truth exists**.
3. For a clean 2-by-2 factor comparison at one checkpoint, vary query **timing** (fixed versus diagnosis-triggered) and query **selection** (generic versus policy-relevant) while matching number of labels and measured computation.
4. Run all confirmatory seeds and never drop non-learning or adverse seeds without an independently documented engineering failure. If testing a second model family is affordable, rerun only the two primary methods and the external evaluator, not every ablation.
5. Plot learning curves against **environment steps or generated tokens** and separately against **new comparison labels**, reward updates and device-hours. Prevent a slow/no-update policy from being described as having a better horizon solely because its clock advances differently.
6. Report for certificate events: first crossing of gamma, **censored** if no crossing in the observation window; fraction of runs without loss at each checkpoint; true finite-step gain at crossing; and the candidate-restricted pointwise value when available.
7. Report uncertainty using the **independent training seed as the unit**; checkpoints are repeated measures, not independent seeds. For document-level results, account for repeated responses on the same document. Provide intervals and raw seed-level results.
8. For repeated certificate checks, implement simultaneous/time-uniform coverage or preregistered error allocation. Do not give every checkpoint a separate uncorrected 95% interval and call the entire trajectory 95%-certified.
9. Include at least one **positive control** (mechanism or task demonstrably learnable) and one **negative control** (well-covered initial preferences or an environment where no loss is expected). Use development data to choose these, and report both.
10. Audit source truth access, held-out isolation, optimizer restoration, reward scaling, stochastic teacher, candidate family, cost accounting, and all paper-ready statements.
11. Generate a concise set of final paper candidate visuals **from the same data tables**: (a) true return/quality versus budget, (b) same-checkpoint intervention comparison, (c) original versus feedback-updated guarantee on an unchanged path when justified, (d) cross-task summary. Prefer 3–4 figures over a large redundant catalog.
12. Freeze the completed results in THIS work plan. A separate detailed run report is optional.

**Required Output E8.**

| Task family / model | Baseline | Comparison | Label budget matched? | Compute budget matched? | Difference in independent return/quality with uncertainty | Effect replicated? | Notes |
|---|---|---|---|---|---|---|---|
| Walker Walk / preference SAC | normal schedule | diagnosis schedule | yes, 937 pairs | yes, 30000 steps; diagnosis may use 4 reward epochs | −5.65 return (sd 5.14) | no | These two families were named after the confirmatory runs existed. Walker has not learned the gait. |
| Verifiable instructions / Qwen2.5-1.5B | linear reward baseline | policy-relevant selection | yes, 24 pairs, 0 added | yes, one policy step and 8 reward steps | 0.000 utility (sd 0.041) | no | Positive control for the certificate gap remains Deep-Sea-Treasure in E1, not this comparison. |
| TL;DR / Pythia-1B | fixed learned reward | fixed-period new AI feedback | no: the comparison adds 8 AI pairs | yes, one policy step and 4 head steps | −0.0016 reference overlap (sd 0.017) | no | Qwen3-4B judged one summary per seed. Human blind evaluation is unavailable. |

**Possible Result Scenarios.**

- **Replicated quality/efficiency improvement:** report an empirical benefit with its task/model scope, and keep analytic claims scoped to their theorems.
- **Mechanism replicated but no downstream gain:** report mechanism results; do not claim a new practical training advantage.
- **Only one environment succeeds:** report that limited result if scientifically meaningful, with appropriate uncertainty; do not silently generalize across domains.
- **Empirical indicators contradict formal small-model findings:** check which formal assumptions no longer hold and separate theorem verification from neural observations.

## 4. Required Deliverables

### 4.1 The completed plan is the primary deliverable

The executing agent must **return this Markdown document with ALL tables filled**. An experiment that cannot run must have status BLOCKED or NOT RUN, a concise factual blocker, the alternate route tried, and a result path if anything was produced. An unrun experiment must not have an invented zero or plausible-looking metric.

For every reported value: specify mean/uncertainty and unit in the cell or the Notes field; add an actual artifact path in Section 7. The Notes column immediately following each result row is **mandatory**; use it to record protocol deviations, issues and comparability limitations, otherwise leave it empty.

Do not turn the main deliverable into a narrative progress report. Do not write novel technical terminology. Use manuscript language for scientific quantities and ordinary descriptive language for engineering.

### 4.2 Completion and status table

| Experiment | Status (NOT RUN / PILOT / COMPLETE / BLOCKED) | Validated seeds / runs | Primary result in Section 3 filled? | Relative result artifacts recorded in Section 7? | Notes |
|---|---|---|---|---|---|
| E1 | COMPLETE | 5 seeds × 2 tasks | yes | yes | Development seed 0 froze w_star, beta, and gamma_pos. |
| E2 | COMPLETE | 3 seeds × 2 checkpoints × 5 branches | yes | yes | Not expanded to 5 seeds. Gain differences are inside the Monte Carlo noise. |
| E3 | COMPLETE | 3 seeds × 3 selectors | yes | yes | Resource-Gathering was not repeated. |
| E4 | COMPLETE | Walker 3 seeds × 4 schedules; Cheetah 2 seeds × 2 schedules | yes | yes | Extra-label Walker was rerun after the first pass failed to add pairs. Schedules are not same-checkpoint forks. |
| E5 | COMPLETE | Push 3×4; Drawer Open 3×2; one reach check | yes | yes | Success is 0. The result is uninformative for manipulation success. |
| E6 | COMPLETE | 3 seeds × 5 methods | yes | yes | Reduced prompt split. Synthetic teacher. |
| E7 | COMPLETE | 3 seeds × 4 methods | yes | yes | Online labels are AI feedback. Human blind evaluation is unavailable. |
| E8 | COMPLETE | comparisons over the finished E4, E6, and E7 seeds | yes | yes | Family choice was not locked before the confirmatory seeds. |

### 4.3 Minimal required electronic outputs for each completed run

- Full config and random seed, exact code commit and third-party library revisions.
- Training data identities/split assignments; preference pair IDs, their label **source** and query timing.
- Training checkpoints including optimizer and RNG states; metadata sufficient to rerun branches.
- Seed-level metric CSV or JSON (not merely screenshots or aggregated points).
- Independent evaluator outputs and uncertainty calculations.
- For each claimed guarantee: the exact source information set, confidence procedure, coverage assumption, required progress threshold and tested time/checkpoint scope.
- Pilot timing and memory estimate plus observed full-run time; warnings or deviations if any.
- Final graph/table generation command; image/PDF outputs may be supplied as secondary artifacts.

### 4.4 Paper claim acceptance checks

| Claim | Evidence available? (YES / NO / PARTIAL) | Matching table / figure ID | Key limitation or deviation | Notes |
|---|---|---|---|---|
| C1 Real-learning realization | PARTIAL | E1 table; `experiments/horizon_real/E8/certificate_vs_candidate.pdf` | True finite-step gain at the loss checkpoint is unresolved because the Monte Carlo SE is about 0.11 | Deep-Sea-Treasure: original certificate falls below gamma_pos while the candidate-restricted value stays positive on 5/5 seeds. That is not a loss of true return. |
| C2 Feedback versus adaptation | PARTIAL | E2 and E3 tables | Updated certificates did not move at the reported medians, and branch gains overlap | The five branches and three selectors were executed. They do not separate adaptation from information. |
| C3 Budget or return/quality benefit | NO | E4, E6, E7, and E8 tables; `walker_methods.pdf`, `instruction_utility.pdf` | Walker and Cheetah returns stay near random; instruction and TL;DR differences sit inside the seed sd | No comparison shows a reliable return or quality gain from extra feedback or extra reward updates. |
| C4 Different task/model family | NO | E4–E7 tables | Locomotion and manipulation did not learn; the second language family does not reproduce the E1 certificate gap | More than one family was run. The E1 pattern was not recovered there. |

**Acceptance condition:** A claim is YES only when its evidence actually exists and has passed protocol auditing. A weak or negative result does not justify rewriting the experiment until it appears successful.

## 5. Runtime and Resource Assessment

### 5.1 Pre-run hardware and stability inventory — fill before training

| Machine / phase | CPU / RAM | GPU model and count / VRAM | Free disk / data download size | OS / Python / major library revisions | Test run command and status | Notes |
|---|---|---|---|---|---|---|
| Phase I | 12 CPU / 62 GB RAM | RTX 5090 was left idle | workspace volume; models stay in `/workspace/.hf_home` | Linux, Python 3.12.15, numpy 2.5.4, mo-gymnasium 1.3.2 | `python tasks/01_Reward_Guidance_Experiments/__main__.py run E1 --stage confirm` completed | CPU only. |
| Phase II | 12 CPU / 62 GB RAM | 1× RTX 5090, 32607 MiB | same volume | torch 2.11.0+cu128, dm-control 1.0.49, metaworld 3.0.0, mujoco 3.16.0 | E4 confirm exit 0; E5 confirm exit 0 | Driver 570.133.07. No display server; GLFW warned and the environments still stepped. |
| Phase III | 12 CPU / 62 GB RAM | 1× RTX 5090, 32607 MiB | Qwen2.5-1.5B, Pythia-1B SFT and reward, Qwen3-4B already downloaded | transformers 5.20.1, peft 0.21.2 | E6 and E7 confirm exit 0 | vLLM is not installed. Generation uses transformers. Installing vLLM was skipped so it would not replace torch. |
| Phase IV | 12 CPU / 62 GB RAM | GPU not required | figures under `experiments/horizon_real/E8/` | matplotlib 3.11.2 | `run E8` exit 0 | Reads finished summaries only. |

**Expected order of magnitude only:** E1–E3 should be CPU-friendly, with moderate RAM and tiny checkpoint files. E4–E5 will likely benefit from one accelerator for policy/reward training, while environment stepping may remain CPU-bound. E6–E7 have the highest VRAM, checkpoint and generation costs; do **not** commit to a specific card or VRAM size before a measured model-load, forward, backward and generation pilot.

### 5.2 Mandatory pilot method

Before any experiment's main run:

1. Measure installation/model-download success and exact downloaded bytes.
2. Measure a warm-up **forward, backward, optimizer step and checkpoint save/restore** on the intended real batch and sequence length.
3. For RL, time at least a short environment rollout and one reward update; for LLMs, time prompt generation, a preference pair evaluation, reward fit and one policy update.
4. Record peak GPU allocated/reserved memory, CPU RSS, disk usage, and tokens or environment steps per second.
5. Estimate full runtime using measured work units, including evaluation, preference generation and checkpoint I/O. Add an explicit uncertainty margin, not an invented precise finish time.
6. If a pilot is unstable, fix the reproducible crash/NaN, reduce task or model size without violating the intended scientific comparison, and rerun the short pilot.
7. If no device can hold the intended model, first try LoRA/PEFT, mixed precision, gradient checkpointing, smaller microbatch with gradient accumulation, reduced generation/sequence length **if the task remains valid**, sequential reward/policy placement, or CPU offload. Confirm equivalence of effective batch/policy-update protocol where claimed.
8. If a single run is expected to exceed **~1 hour**, identify bottlenecks and cheap speedups. If a phase is expected to exceed **~10 hours**, explicitly reconsider parallel CPU environment collection, batching generations, caching immutable features, checkpoint interval, and whether to reduce an unjustified experiment grid. Do not spend hours optimizing a task whose entire run is already short.
9. Never use an accelerator merely to rerun cheap deterministic analytic checks.

### 5.3 Runtime prediction and observation log

Fill the table **before** full scale with estimates, and update actuals afterward. Prefer ranges. Record total costs for all seeds, not only one best run.

| Experiment | Small pilot size | Pilot wall time | Estimated time per seed | Estimated full total | Actual full total | Peak GPU VRAM / CPU RAM | Peak disk | Bottleneck / mitigation | Notes |
|---|---|---|---|---|---|---|---|---|---|
| E1 | 5 seeds were the confirm run; the earlier 2-seed pilot was superseded | under 1 min | seconds | under 1 min | under 1 min for both tasks | CPU; RAM not separately logged | small npz checkpoints | none | gamma_pos frozen in `frozen.json` before seeds 1–5. |
| E2 | 2 checkpoints, 5 branches, 3 seeds | under 1 min | seconds | under 1 min | under 1 min | CPU | small | none | |
| E3 | 16 pairs, 3 selectors, 3 seeds | under 1 min | seconds | under 1 min | under 1 min | CPU | small | none | |
| E4 | confirm used 30000 steps after a 4000-step pilot whose directory was overwritten | not retained | about 1.4 min at 30000 steps | about 20 min | 21.8 min for 16 runs | 5090 used; allocated MiB not logged; 62 GB RAM | agent checkpoints gitignored | environment stepping | Do not read the overwritten pilot number back out of the confirm metrics. |
| E5 | 6000-step confirm; one 2000-step reach check | reach check 9 s | about 20 s | about 6 min | 5.2 min for 18 runs | 5090; allocated MiB not logged | small | environment stepping | |
| E6 | 24 train / 8 eval prompts, not 100/30/50 | the confirm run is the measured run | about 25 s | about 2 min | 80 s for 3 seeds | 5090; allocated MiB not logged | no policy checkpoint saved | batched generation of 32 new tokens | vLLM not used. |
| E7 | 12 prompts, 8 online pairs, not 100–200 prompts | the confirm run is the measured run | about 15 s plus one shared judge | about 2 min | 55 s for 3 seeds and the Qwen3-4B judge | policy and teacher loaded together, then the judge | no policy checkpoint saved | short generations | |
| E8 | two task families plus the TL;DR row | under 1 min | n/a | under 1 min | about 1 s | CPU | three PDFs | none | |

### 5.4 Suggested low-cost execution order

1. E1 smoke, while another worker sets up E4 and E6 environments.
2. E2/E3 as soon as E1 snapshots can be restored, without waiting for a positive E1 result.
3. E4 and E5 can proceed in parallel if hardware allows; otherwise prioritize E4 and defer the optional second task.
4. E6 can proceed with a different GPU from E4. E7 requires its own stability and throughput gate.
5. E8 begins only after the source experiments have interpretable **development** results and a frozen final-test plan.

The plan is not a promise of background/asynchronous execution. Record real measurements; no unmeasured runtime guarantees.

## 6. Terminology Reference

This section defines only terms needed to execute and report the paper's experiments.

| Paper term | Plain-language interpretation / necessary definition |
|---|---|
| **Guidance reward** R_phi | Reward used to induce a policy update; need not minimize reward prediction error. |
| **True reward / preference** w_star or r_star | Task utility known to a synthetic teacher/evaluator; hidden from the learner. In human-feedback tasks the true utility is generally unknown. |
| **Bradley–Terry preference feedback** | A binary comparison with probability sigmoid(beta times the two trajectory/response utility difference), with its teacher model declared. |
| **Source preference information** U_0 | Reward parameters compatible with the original preference evidence under the stated information model. |
| **Updated preference information** U_new | The corresponding set after additional comparisons. Only justified confidence procedures may be called certificates. |
| **Certified gain** | Worst compatible improvement over the stated reward class. It differs from observed true return. |
| **Required progress** gamma | Predeclared threshold for improvement; use gamma=0 and a fixed positive threshold as separate diagnostics. |
| **Pointwise feasibility** | A qualifying reward setting exists at one current policy state. For finite search, write candidate-restricted value. |
| **Trajectory feasibility** | An admissible path beginning from its initialization can maintain the guarantee. It is stronger than pointwise feasibility. |
| **Support horizon** | Last duration/checkpoint interval satisfying the original guidance guarantee; record censoring when loss is not observed. |
| **Finite reward adaptation** | Restricted amount the reward parameters can move through the declared training process. A bound on capability does not specify what an optimizer will do. |
| **Dynamic feasibility boundary** | Boundary between preference alternatives that can and cannot support a target guidance horizon in the declared model. |
| **Preference-label complexity** | Number of comparisons needed for statistical certification under stated model assumptions; do not transplant the paper's scalar d_T scaling unchanged to arbitrary LLMs. |
| **Executed finite-step gain** G_k | Difference in independent evaluation utility between the actual updated and saved policy for a declared update. |
| **Candidate-restricted pointwise value** V_k^Q | Best worst-case gain among a fixed, finite set Q of reward candidates; not proof of a global optimum. |
| **Checkpoint intervention** | A controlled comparison of alternative reward/data updates starting from exactly the same saved training state. |
| **Censored support horizon** | No failure observed before the run ends; the final checkpoint is **not** automatically the true failure time. |
| **Empirical diagnostic** | Predictive or ensemble measurement without proven coverage of the true preference; not a formal certificate. |

For the main manuscript symbols and the distinctions among prediction, guidance and preference learning, always prefer manuscript Sections 3–7 and Appendices C–E over paraphrases here.

## 7. Result Location

**This section is filled when experiments run.** Use repository-relative paths under `experiments/horizon_real/`. Do not invent filenames. The executing agent fills every row, including BLOCKED / NOT RUN as appropriate. Large checkpoints stay in those run directories and are not committed.

| Experiment | Result description | File / Folder Path | Config / code revision | Notes |
|---|---|---|---|---|
| E1 | Per-seed metrics, frozen development config, summaries | `experiments/horizon_real/E1/` | git `8078a36` plus this working tree | Checkpoints under `checkpoints/` are not committed. |
| E2 | 30 branch rows | `experiments/horizon_real/E2/Deep-Sea-Treasure/metrics.csv` | same tree | |
| E3 | 9 selector rows | `experiments/horizon_real/E3/Deep-Sea-Treasure/metrics.csv` | same tree | |
| E4 | Walker and Cheetah summaries and per-run metrics | `experiments/horizon_real/E4/summary.json` | same tree | `agent.pt` stays in each run's `checkpoints/` directory. |
| E5 | Push, Drawer Open, and the reach check | `experiments/horizon_real/E5/summary.json` | same tree | Task ids `push-v3`, `drawer-open-v3`, `reach-v3`. |
| E6 | Five methods, three seeds | `experiments/horizon_real/E6/summary.json` | same tree | Teacher is synthetic. |
| E7 | Four methods, three seeds, Qwen3-4B scores | `experiments/horizon_real/E7/summary.json` | same tree | Online labels are marked AI feedback. |
| E8 | Matched-budget rows and three figures | `experiments/horizon_real/E8/summary.json`, `certificate_vs_candidate.pdf`, `walker_methods.pdf`, `instruction_utility.pdf` | same tree | Protocol checks: `experiments/horizon_real/protocol/checks.json`. |

### Final handoff

Return, in priority order:

1. **This same completed 01_Reward_Guidance_Experiments.md**, with the Section 3 result tables, Section 4 status/claim tables, Section 5 runtime tables and Section 7 artifact paths filled.
2. The corresponding code and reproducible data/aggregate results at the recorded revision and paths.
3. Optionally, a brief supporting execution report for unresolved blockers or protocol changes. **The report never substitutes for the completed plan.**

A deviation from the prescribed procedure **must appear in the Notes cell for that result**. If no deviation or issue occurred, a blank Notes cell is correct. No speculative scientific interpretation is required from the executing agent.

---

## References and verified implementation starting points

- Current manuscript: *The Limits of Reward Guidance under Finite Adaptation*, especially Sections 3–8, Appendices A–F (source manuscript supplied by project owner).
- Current repository reproduction: [docs/REPRODUCE.md](../docs/REPRODUCE.md), [docs/FIGURE_MAP.md](../docs/FIGURE_MAP.md), [docs/PAPER_RESULTS.md](../docs/PAPER_RESULTS.md).
- [MO-Gymnasium Deep-Sea-Treasure](https://mo-gymnasium.farama.org/environments/deep-sea-treasure/) and [Resource-Gathering](https://mo-gymnasium.farama.org/environments/resource-gathering/).
- [B-Pref code and preference-based RL teacher baselines](https://github.com/rll-research/BPref).
- [Meta-World](https://github.com/Farama-Foundation/Metaworld).
- [IFEval original implementation](https://github.com/google-research/google-research/tree/master/instruction_following_eval).
- [OpenAI human-feedback summary comparison dataset](https://huggingface.co/datasets/openai/summarize_from_feedback) and [released TL;DR sources](https://github.com/openai/summarize-from-feedback).
- [TRL PPO/TL;DR reference](https://huggingface.co/docs/trl/main/ppo_trainer); version-pin before execution.
- [Qwen3-4B-Instruct-2507 model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507).

These are starting points, not a guarantee that old and new dependency versions interoperate. Pin the actually tested versions and record necessary fixes.
