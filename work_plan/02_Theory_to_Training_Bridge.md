# 02 Theory to Training Bridge

**Project:** The Limits of Reward Guidance under Finite Adaptation  
**Target:** ICML 2027  
**Status:** Steps 0–5 and follow-ups 5A–5C are completed. Earlier rows are unchanged. 5A–5C are filled below.  
**Repository:** https://github.com/AtheavenMa/The-Limits-of-Reward-Guidance-under-Finite-Adaptation  
**Date:** 2026-10-18

> **Main rule:** Begin with the exact constructions and numbers already established by the paper. Add only ONE new uncertainty, optimizer, or environment change at a time. The output of every experiment must identify a definite result or a specific failed premise. No new Walker, Meta-World, Qwen, or TL;DR run until the small scientific bridge is understood.

## 1. Paper and Experiment Overview

The paper is about how long original preference information can certify policy improvement while the policy and its guidance reward change. The theoretical conclusions already exist. New experiments should extend them, **not** attempt to rediscover the result in unrelated benchmarks.

The central distinction is:

- **Pointwise feasibility:** At a given policy state, a reward setting can guarantee the required policy improvement.
- **Trajectory feasibility:** A reward-parameter trajectory starting at the given initialization can maintain that guarantee within its adaptation budget.
- **Statistical certification:** The available preference comparisons contain enough information to establish the guarantee.

The manuscript's exact scalar model (Theorem 1, Section 4.2) is

\[
j_\delta(t)=g+\delta[\phi(t)-vt],
\quad\delta\in[-\eta_0,\eta_0],\quad
|\dot\phi|\le\rho<v,
\]

and its support horizon is

\[
T_0=\frac{g-\gamma}{\eta_0(v-\rho)}.
\]

**Important:** Losing the certificate does not mean losing true performance. When the true preference is \(\delta_\star=0\), actual normalized gain can remain \(1\) while the source-only certificate expires.

Use the owner's updated paper PDF for Sections 3–8 and Appendices A–F. The PDF will be provided separately.

### 1.1 Existing work is the starting point

| Already existing and verified | Code to reuse | Scientific purpose |
|---|---|---|
| Scalar support horizon and finite-sample BTL tools | [horizon.py](../horizon.py) | Exact expectations and valid fixed-N preference intervals |
| Five-output endogenous policy and targeted labels | [experiments/legacy_checks.py](../experiments/legacy_checks.py), check_closed_loop | Exact finite policy mechanism without invented external drift |
| Discrete finite-step reward calibration | [experiments/discrete_resource_check.py](../experiments/discrete_resource_check.py) | Actual clipped guidance updates and resource budget |
| Ordinary Euclidean policy-gradient example | [experiments/strengthening_checks.py](../experiments/strengthening_checks.py), ordinary_pg_check | Exact result not dependent on a specially designed preconditioner |
| Original frozen numerical results | [results/reference/](../results/reference/), [docs/REPRODUCE.md](../docs/REPRODUCE.md) | Provenance and reference values |
| Prior E1–E8 engineering run | [experiments/real/](../experiments/real/), [experiments/horizon_real/](../experiments/horizon_real/) | Historical pilot only, do not overwrite or scale by default |
| New first executable bridge | [experiments/bridge/01_finite_policy.py](../experiments/bridge/01_finite_policy.py) | Step 1 below, runnable on CPU |

**Why the old broad experiment failed to answer the paper's question:** E1 used a noisy maximum over reward candidates and a discretized confidence set that need not include the true preference. E4/E5 did not learn normal RL behavior. E6/E7 ran a single reward-weighted policy-gradient step rather than complete PPO. Their negative or inconclusive scores do not distinguish the cause. Preserve those records; do not expand them yet.

### 1.2 Fixed exact parameters — do not retune

For the five-output model use:

\[
\epsilon=.04,\quad g=1,\quad\gamma=.6,\quad
\eta_0=1,\quad v=.12,\quad\rho=.04,\quad T_0=5,\quad T=15.
\]

Original source information leaves \(\delta\in[-1,1]\). The informative Bradley–Terry comparison in Appendix A.3 has sensitivity \(q=\sqrt{2/3}\). For the known reference \(\delta_\star=0\), the representative batch is 600 positive labels out of 1200.

For the separate ordinary Euclidean PG example, preserve \(x_0=.8,y_0=0,\phi_0\simeq.84280432,\gamma=.06,\rho=.04\); the existing maximal-speed trajectory touches the feasible boundary near \(t=.007836\), with an analytic local all-controls bound near \(t=.010006\).

### 1.3 Minimal dependency graph

~~~text
5-output theorem and existing code
    |
    +-- Step 1: only change continuous -> finite policy steps
    |       |
    |       +-- Step 2: only add finite BTL labels / confidence interval
    |               |
    |               +-- Step 3: only replace ideal guidance motion
    |                   with the existing clipped optimizer
    |
4-output ordinary-PG theorem and existing code
    |
    +-- Step 4A: exact finite policy updates
            |
            +-- Step 4B: sampled policy gradient
                    |
                    +-- Step 5: smallest real learning interface
                             (same four outcomes, actual sampled actions)

Only AFTER these bridges: a separate small MDP / Deep-Sea-Treasure study,
then a verified learning SAC benchmark, and only then possibly LLMs.
~~~

## 2. Execution Principles

1. **Change one thing at a time.** Do not simultaneously change state/action space, policy parameterization, feedback mechanism, guidance optimizer and evaluator.
2. **Every non-ideal result must have a diagnosis.** Compare against exact population calculations, a known working optimizer, or a known positive/negative control. An uninterpretable low benchmark reward is not a result.
3. **Do not confuse preference and guidance parameters.** BTL training estimates the unknown preference \(\delta\). The guidance coefficient \(\phi\) determines how the policy updates. Updating BTL parameters does not automatically move \(\phi\) toward the moving certified set.
4. **No statistical guesses when exact sums are available.** Four/five-output spaces are sufficiently small for exact probability/gain calculations. Add finite samples only when explicitly investigating sampling.
5. **Keep theory scope intact.** Universal trajectory impossibility is proved only under declared assumptions. Searching a finite number of reward candidates establishes a candidate-restricted value, not the unrestricted pointwise optimum.
6. **Negative controls are useful, not failed experiments.** No ambiguity and sufficient adaptation speed are predetermined controls where loss should not occur. Uninformative preference comparisons should not resolve \(\delta\).
7. **Use the smallest informative run.** Begin with deterministic calculations or a fixed 10-seed sampling smoke test. Increase sample count to resolve a quantified error, not merely to seek a favorable sign.
8. **No manipulation of outcomes.** Retain contrary results and the original parameters. A result outside the theory's conditions is an identified scope difference, not evidence to discard.
9. **Independent branches can continue.** An error in the five-output preference estimator does not block the separate exact ordinary-PG validation. But do not move a broken branch to a more complex environment.
10. **The completed plan is the delivery.** Fill all tables and result locations. Notes are mandatory only for deviations, limitations or unexpected issues.

## 3. Sequential Experiments

### Step 0 — Verify existing exact anchors

**Purpose.** Preserve an authoritative starting point.

**Execution.**

1. Run the existing tests, demos and relevant numerical reference checks, recording exact commands and source commit. Do not overwrite results/reference.
2. Record the previously observed full-reproduction mismatch involving adaptive_BTL.maximum_error_radius_ratio (recorded 0.469073 versus reference 0.426380); check it independently, but do not conflate it with the core horizon identity.
3. Verify these known reference quantities:
   - Scalar source-only horizon: 5.
   - Updated source information from 600/1200 informative BTL labels certifies the SAME trajectory through T=15.
   - Ordinary Euclidean PG maximal-speed contact: approximately .007836.
   - Finite resource experiment with reward update counts m=0,1,2,3: first uncertified time 3.35, 4.55, 7.85, and no failure before 15, respectively.
4. If a deterministic assertion fails, isolate the smallest input and code expression responsible. Do not inject Monte Carlo noise before it is corrected.

| Exact anchor | Expected | Observed | Status | Notes |
|---|---|---|---|---|
| Theorem 1 exact T0 | 5.000 | 5.000 | PASS | `support_horizon(1, .6, 1, .12, .04)` |
| Original finite five-output policy valid | p_i > 0 and M positive definite | minimum probability 0.12322 to 0.12329; minimum eigenvalue 1.75464 | PASS | Step 1 mesh, all four step sizes |
| BTL feedback extends fixed trajectory | T=15 certified for reference 600/1200 | interval [-0.19246, 0.19246]; terminal certificate 0.76905; path minimum with feedback at t=4 is 0.682 | PASS | True gain at delta=0 remains 1. Source-only terminal certificate is -0.200 |
| Ordinary PG maximal-speed contact | approximately .007836 | 0.00783587 | PASS | All-controls local upper bound 0.010006 |
| Discrete reward steps m=0/1/2/3 | 3.35 / 4.55 / 7.85 / censored | 3.35 / 4.55 / 7.85 / no loss before 15 | PASS | Exact step identity error 7.73e-13. `results/reference` unchanged |

### Step 1 — Finite policy steps in the same five-output construction

**Only change:** Integrate the same finite policy system using discrete Euler steps rather than continuous time. Keep all theory parameters and the ideal guidance path unchanged.

**Run now:**

~~~bash
python experiments/bridge/01_finite_policy.py --out outputs/horizon_bridge/step01
~~~

The script was written and independently run on CPU when this plan was prepared; **rerun it in the repository as a reproducibility check**. It records a small CSV plus JSON, and verifies probability positivity, preconditioner positivity, and exact normalized finite-step gain for \(\delta=-1,0,1\). It does NOT claim to train a reward model.

Steps:

1. Use the very same five-output \(p_\theta\) and \(M(\theta)\) already implemented in experiments/discrete_resource_check.py.
2. Fix \(\phi(t)=.04t\) and endogenous \(c(t)=.12t\).
3. Sweep only policy step sizes \(h\in\{.2,.1,.05,.025\}\).
4. At each step compute exact source-only certificate, pointwise optimum, true gain at \(\delta_\star=0\), \(p_\theta\) and preconditioner eigenvalue.
5. Check first **strict** loss of guarantee after \(T_0=5\); equality at 5 is still certified.
6. Negative controls: \(\eta_0=0\) or adaptation capable of tracking at \(\rho\ge v\); neither creates the original loss on the chosen ideal path.
7. Calculate fixed-N reference 600/1200 BTL posterior confidence interval but do not count it yet as a trained-model result.

**Independently measured initial anchor results:**

| h | First strict loss | Exact gain identity error (upper magnitude) |
|---:|---:|---:|
| .200 | 5.200 | 1.2e-13 |
| .100 | 5.100 | 3.0e-13 |
| .050 | 5.050 | 6.3e-13 |
| .025 | 5.025 | 1.3e-12 |

At T=15, source-only certificate = -0.2; true gain = 1; pointwise gain = 1. For the reference 600/1200 informative feedback, the **terminal** updated certificate = approximately .76905 > .6. Do not confuse terminal updated certificate with the **minimum over the entire path** before feedback arrives (the original paper reports approximately .682).

**Fill from actual repository run:**

| h | First loss | Minimum output probability | Minimum M eigenvalue | Max finite-step identity error | T=15 true gain | Notes |
|---|---|---|---|---|---|---|
| .200 | 5.200 | 0.123216 | 1.754638 | 1.356e-13 | 1 | Source certificate at T=15 is -0.200. Pointwise gain is 1 |
| .100 | 5.100 | 0.123259 | 1.754638 | 2.983e-13 | 1 | |
| .050 | 5.050 | 0.123280 | 1.754638 | 6.032e-13 | 1 | |
| .025 | 5.025 | 0.123291 | 1.754638 | 1.248e-12 | 1 | eta0=0 and rho=v both keep gain 1 through T=15 |

**If mismatch:** the error is in the finite simulator/parameter translation, not in a new learned-reward effect. Repair before Step 2.

### Step 2 — Actual finite Bradley–Terry feedback on a FROZEN trajectory

**Only change:** Replace exact knowledge of a preference interval with confidence bounds inferred from actual binary feedback. Do NOT change policy dynamics, \(\phi(t)\), or their trajectories.

**Why this is essential:** Separate *information insufficiency* from any subsequent issue in the policy/reward optimizer.

Steps:

1. Hold Step 1's successful five-output trajectory fixed. Original source comparisons leave \(\delta\in[-1,1]\).
2. Use the informative Appendix A.3 pair, known sensitivity \(q=\sqrt{2/3}\), true \(\delta_\star=0\), binary labels \(Y\sim\operatorname{Bernoulli}(1/2)\). Keep source and informative comparisons distinct.
3. Use \(N\in\{64,256,1200\}\). Start with 10 predetermined seeds as a code smoke test, then use **exact binomial probabilities** rather than interpreting 10 successes as precise coverage estimates.
4. Fit \(\delta\) with the scalar BTL likelihood. Cross-check the solution against the exact binomial likelihood, and compute the valid fixed-N confidence set with horizon.btl_confidence_interval / horizon.certify_btl_batch. Do not discretize a simplex, do not recycle the defective prior E1 confidence grid.
5. Compute the updated worst-case certificate at T=15 on the SAME exact policy/reward trajectory. Report updated certificate along the full trajectory separately, including the specified feedback arrival time.
6. Fixed reference arm: 600/1200 positive labels must yield approximately \([-0.19246,0.19246]\) and terminal certificate approximately .76905.
7. **No-information arm:** 1200 repeats of a source comparison that has no \(\delta\) sensitivity. Its \(\delta\) uncertainty cannot shrink.
8. **Infeasible-preference control:** choose \(\delta_\star>1/3\) (the T=15 dynamic boundary for this scalar setup). More informative labels should prevent falsely certifying T=15, not manufacture feasibility.
9. Preserve equal sample-size comparisons; do not allow different optimization or policy steps across conditions.

**Fill:**

| Preference evidence | N | True delta | Estimated delta / valid interval | T=15 certificate | Exact acceptance probability or result | Notes |
|---|---:|---:|---|---|---|---|
| Original source only | 0 | 0 | [-1, 1] | -0.200 | 0 | True gain remains 1. First strict loss 5.025 on the h=.025 grid |
| Source comparisons, no delta information | 1200 | 0 | [-1, 1] | -0.200 | 0 | Sensitivity 0 leaves the likelihood flat, so the source set does not shrink |
| Informative comparisons | 64 | 0 | median 0; [-0.86603, 0.86603] | -0.03924 | 0.05171 | One-sided test of delta<1/3 has probability 0.08432. Same binomial probabilities, different test |
| Informative comparisons | 256 | 0 | median 0; [-0.41990, 0.41990] | 0.49612 | 0.28693 | Median path first loses at 11.925. One-sided probability 0.37737 |
| Informative comparisons | 1200 | 0 | median 0; [-0.19246, 0.19246] | 0.76905 | 0.97683 | No loss before 15. Path minimum with feedback arriving at t=4 is 0.682, not the terminal value |
| Fixed reference, 600/1200 | 1200 | 0 | [-0.19246, 0.19246] | 0.76905 | deterministic published count | Library interval matches an independent Hoeffding calculation |
| Dynamically infeasible true delta | 1200 | 0.400 | median 0.3995; [0.20441, 0.59977] | 0.28027 | 0.0001325 | False certification of T=15 stays far below 0.05. Median path loses at 8.35 |

**Result interpretation:** A random fixed batch missing the requested certificate is expected at some sample sizes, with a predicted probability. If the exact binomial and implemented fixed-batch test disagree, the new statistical implementation is faulty. If a valid information set contracts but the same fixed trajectory remains uncertifiable, the target or the remaining uncertainty is too demanding; this is a definite conclusion.

### Step 3 — Only now replace ideal guidance with ACTUAL finite adaptation

**Only change:** Use the existing clipped quadratic guidance-calibration procedure rather than the ideal maximum-speed path, while preserving the SAME five-output policy and source uncertainty.

**Do not conflate two different optimizations:** The BTL objective fits the unknown true preference coefficient \(\delta\). The guidance optimizer moves \(\phi\) in response to the current policy and the certified feasible region. A stationary BTL reward fit is not automatically a moving-target controller.

Steps:

1. Reuse experiments/discrete_resource_check.py exactly: \(h=.05,\eta=.01,G=.2,\gamma=.6\).
2. At every policy step compute the actual endogenous \(c(\theta)\), and perform \(m\in\{0,1,2,3\}\) projected, clipped calibration steps in \(\phi\), using the published update. Take the same finite policy step afterward.
3. Check the original results: first uncertified times 3.35, 4.55, 7.85 and no loss before 15; record maximum observed movement and allowed movement \(m\eta G\).
4. Compare the actual optimizer's horizon to the ideal displacement envelope. An optimizer can underuse its budget, so a shorter realized horizon does not contradict the theory.
5. With each trajectory now **fixed**, apply Step 2's informative preference batch and recompute updated certificates. This cleanly tests the effect of information without retraining the policy.
6. Separately implement one **robust-gain guidance controller** constrained to the same movement budget; it may use the observed BTL confidence set but must not read the hidden true preference. Compare it against clipped quadratic calibration on the identical finite policy state.
7. Then, and only then, fit a scalar BTL reward preference model and let the guidance controller use its confidence set; report \(\hat\delta\) and \(\phi\) separately.
8. Negative controls: no preference ambiguity; sufficient maximum adaptation speed; and insufficient information despite many repeated source-only labels.

**Fill:**

| Guidance method | Reward steps m / movement budget | Source-only first loss | After-feedback first loss on SAME trajectory | Realized movement | Ideal envelope | Notes |
|---|---|---|---|---|---|---|
| No calibration | 0 / 0 | 3.35 | none before 15 | 0 | 3.35 | The 600/1200 interval raises the terminal certificate from -0.794 to 0.655. True gain at delta=0 stays 1 |
| Existing clipped calibration | 1 / 0.002 | 4.55 | none before 15 | 0.002 | 5.05 | Early steps use eta times the lag, so the budget is not spent while the lag is below 0.2 |
| Existing clipped calibration | 2 / 0.004 | 7.85 | none before 15 | 0.004 | 10.05 | Same published identity, error 7.73e-13 |
| Existing clipped calibration | 3 / 0.006 | none before 15 | none before 15 | 0.005992 | none before 15 | Realized movement is slightly inside the budget |
| Robust guidance controller, same budget | 1 / 0.002 | 5.05 | none before 15 | 0.002 | 5.05 | Spends the full budget toward c. Source-only loss matches the envelope. Maximum phi gap versus clipped m=1 is 0.0366 |
| BTL-fitted information and guidance | 1 / 0.002 | 5.05 | 9.95 | 0.002 | 5.05 | Seed 0, hidden delta 0.4 used only to draw labels. delta hat=0.3078, interval [0.1142, 0.5053], final phi=0.600. Repeating source-only labels leaves the m=1 loss at 4.55 |

**If there is no improvement:** Report whether the missing step was information, movement budget or optimizer realization. These are distinct quantities already measurable with exact five-output probabilities. Do not jump to SAC.

### Step 4 — Existing ordinary Euclidean PG, then one sampling perturbation

**This is the second exact theorem anchor**, not an unrelated benchmark.

**4A. Deterministic ordinary PG.**

1. Use the *unchanged* two-coordinate Bernoulli policy from Theorem 2, its four outputs, \(r_a=a q+(1-a)s\), and \(R_\phi=\phi q+(1-\phi)s\).
2. Verify the known safe interval, pointwise value and initialization using experiments/strengthening_checks.py.
3. Replace continuous integration with finite steps \(h=5e-4\) and \(h=2.5e-4\). Evaluate the exact four-outcome expected gradient and normalized true/certified gains.
4. Verify numerical boundary contact converges toward .007836, the analytical local all-controls upper bound is approximately .010006, and the pointwise optimum remains above .06 in the specified box.
5. If this fails, fix the deterministic PG discretization before adding data.

**4B. Only replace the exact expected gradient by a sample-based gradient.**

1. Keep the identical four outcomes, features, optimizer, initial conditions and evaluation formulas.
2. At one fixed checkpoint, estimate the score-function policy gradient from a small number of sampled actions. Compare its error directly against the exact enumerated gradient, including the boundary-moving direction.
3. Choose the sample size according to whether gradient error is smaller than the **known analytical margin**, not because a generic benchmark uses some standard episode count.
4. Run short sampled trajectories and measure contact-time variability. Use exact enumerated values as independent assessment, paired random numbers when appropriate, and actual uncertainty estimates.
5. Fit BTL preferences from Step 2 only after the sampled gradient behaves correctly. Only after that may a small reward head replace the known fixed feature representation.

**Fill:**

| Model/update | Exact or sampled gradient | Sample size | Boundary contact | Pointwise value > gamma? | Gradient error | Notes |
|---|---|---:|---|---|---|---|
| Existing theorem reference | exact continuous | n/a | 0.00783587 | yes; box lower bound 0.10575 | none | All-controls local upper bound 0.010006 |
| Finite PG, first mesh | exact, h=5e-4 | n/a | first strict loss 0.008; last certified 0.0075 | yes; minimum in the box 0.11278 | 0 | Bracket (0.0075, 0.008] contains the continuous contact |
| Finite PG, half mesh | exact, h=2.5e-4 | n/a | first strict loss 0.008; last certified 0.00775 | yes; minimum in the box 0.11278 | 0 | Bracket shrinks to (0.00775, 0.008] |
| Sampled PG, small batch | sampled | 64 | mean 0.008, sd 0.000559, range 0.007 to 0.009 | yes | mean 0.0901; checkpoint max 0.186; path max 0.342 | Eight seeds. Error exceeds the 0.001 slack |
| Sampled PG, accuracy-controlled batch | sampled | 4795680 | 0.008 on all 4 seeds | yes | checkpoint max 0.000738; path max 0.000798 | One-sample formula gave n=149865 with measured max error 0.00421. N was doubled until the measured error was below 0.001 |
| BTL-fed sampled PG | sampled | 4795680 | 0 | yes; pointwise 0.11455 | path max 0.000571 | 1200 labels give phi hat=0.780, 0.062 below phi0=0.843, already outside the safe interval. The gradient is accurate; the fitted coefficient is not |

**If exact PG works but sampled PG does not:** the new source of difficulty is gradient sampling precision, not the validity of the exact construction. If both work, this is the actual bridge from proven policy dynamics to stochastic policy training.

### Step 5 — Original four-output contextual bandit (completed historical pilot; not yet a closed loop)

**Purpose.** Introduce genuine sampled trajectories and preference-trained reward into the **same four-outcome model**, rather than changing to Walker.

1. Wrap the exact four outcomes from Theorem 2 in a one-step contextual bandit with identical policy and reward features.
2. Test the wrapper against exact four-outcome value and gradient first.
3. Train the actual policy with sampled REINFORCE/score-function updates. Fix the same teacher preferences.
4. Fit a simple BTL reward model from source comparison labels; separately add informative comparison labels. Never leak the true preference parameter to learning.
5. Compare exact versus sampled gains and certificates at corresponding checkpoints.
6. Only change one additional element at a time: finite preference labels, finite \(\phi\) optimizer, neural reward approximator.
7. Keep no-ambiguity and sufficient-speed controls.
8. Stop at small sample sizes while the exact model supplies ground truth. Only add more data to meet a stated measurement-precision target.

| Bandit variant | Ground-truth reference | Samples/updates | True gain | Source-only certificate | Candidate pointwise value | Agreement with exact model | Notes |
|---|---|---|---|---|---|---|---|
| Exact four-output model | yes | 8 updates, h=5e-4 | 0.67471 | 0.06061 | 0.11434 | exact | Initial true gain 0.67424 and initial certificate 0.06126. Hidden preference is evaluator-only |
| Sampled policy gradient | yes | 8192 samples, 8 updates | 0.67471 | 0.06062 | 0.11434 | absolute x gap 1.07e-6 | Last gradient error 0.0120, above the 0.001 margin. The short state still matches |
| BTL-fitted reward | yes | 256 informative labels, then 8 exact updates | 0.67455 | 0.00932 | 0.11449 | absolute x gap 1.66e-4 | phi hat=0.802. Certificate falls below 0.06 while true gain barely moves |
| New informative feedback | yes | 1200 informative labels, then 8 exact updates | 0.67466 | 0.03599 | 0.11439 | absolute x gap 5.63e-5 | phi hat=0.823, still outside the initial safe interval. Zero-contrast source labels leave the reward head at 0 |
| Bounded guidance updates | yes | 1200-label target, movement at most rho | 0.67471 | 0.06021 | 0.11434 | absolute x gap 7.23e-8 | The speed limit keeps guidance inside the safe interval that the unconstrained fitted jump leaves |


### Step 5A — Complete the original four-output trajectory before adding learning uncertainty

**Status:** RUN. This step replaces the scientific interpretation of the short eight-step Step 5 pilot and leaves that pilot's original results unchanged.

**Objective:** First verify the actual finite-step robust gain and true improvement over a trajectory that really reaches the boundary at approximately t = 0.007836. Do not add any newly fitted reward or new task at this stage.

**What remains fixed:** Theorem 2's Bernoulli policy over (q,s) in {-1,1} squared, the true reward family r_a(q,s) = a q + (1-a)s with a in [-1,2], guidance reward R_phi = phi q + (1-phi)s, x0 = 0.8, y0 = 0, phi0 = 0.8428043202, rho = 0.04, gamma = 0.06, and the original source information that cannot distinguish a. Choose the evaluator-only true preference a_star = phi0 as in the existing Step 5 pilot. This choice is fixed from the previous implementation, NOT retuned to produce a desired result.

**Critical corrections to the old Step 5 code:**

1. Existing true_gain(x,y,phi_star) actually equals the policy's **current true return** J_a(x,y), not improvement. Rename that logged quantity to true_return. Do not erase the old output; write corrected results to a NEW directory.
2. Record the policy's actual finite-step **gain per unit policy time**: (J_a(theta_next) - J_a(theta_current)) / h. Report separately the exact instantaneous true progress rate under the population vector field. Cumulative true return and finite-step gain are not interchangeable.
3. Use exactly h = 0.0005 initially. Run at least **24 policy updates** through t = 0.012 (rather than the previous eight updates ending at t = 0.004). The original continuous boundary is near 0.007836 and the observed exact finite-step crossing near 0.008. Extend to 30 updates only if needed to characterize the post-contact behavior.
4. Start ALL variants from the same (theta0, phi0) and prevent any instantaneous BTL-fitted replacement of phi0.
5. When using a fitted reward head, the actual policy guidance must remain R_phi = phi q + (1-phi)s with phi in [0,1]; do not accidentally use unnormalized two-head coefficients that change the overall policy-gradient scale.
6. Recompute the SOURCE robust certificate using a in [-1,2] without fitting a coarse parameter grid. Because gain is affine in a, evaluating the interval endpoints is exact. Compute the source-only certificate even after additional feedback, for a fair original-information comparison.
7. Compute both (i) the exact **prospective expected gain** for the declared population policy update, and (ii) the exact worst-case gain **after a realized sampled update**. The second is an ex-post measurement and is not automatically an advance guarantee for a future stochastic update.
8. The unrestricted pointwise benchmark is available analytically in the continuous model:
   
   \[
   \omega_q=(1-x^2)^2,\quad\omega_s=(1-y^2)^2,\qquad
   V=\frac{\omega_q\omega_s}{\omega_q+\omega_s}.
   \]

   If evaluating **finite-step** pointwise feasibility, optimize the exact executed finite-step gain over phi in [0,1] (one-dimensional continuous search or an analytic solution), not the noisy maximum across independently simulated candidates.

**Exact metrics for every step k:**

\[
J_a(x_k,y_k)=a x_k+(1-a)y_k,
\quad
g^{\rm true}_{k,h}=\frac{J_{a_\star}(x_{k+1},y_{k+1})-J_{a_\star}(x_k,y_k)}{h},
\]

\[
C_{k,h}(U)=\min_{a\in U}
\frac{J_a(x_{k+1},y_{k+1})-J_a(x_k,y_k)}{h}.
\]

For the population continuous policy gradient, compare with

\[
g_a(x,y,\phi)=a\,\omega_q\phi+(1-a)\omega_s(1-\phi).
\]

Both formulas have the same units as the main-paper required gain gamma = 0.06. Do NOT compare a current true return such as 0.6747 with gamma = 0.06.

**Minimal experimental sequence:**

1. Reuse the importable formulas in experiments/bridge/04_ordinary_pg.py. Implement a new script experiments/bridge/06_online_bandit.py; keep 05_contextual_bandit.py intact as the historical pilot.
2. Implement one exact model transition and one sampled REINFORCE transition from the same current state. A sampled update uses the actual guidance R_phi (not hidden true reward). Always evaluate finite-step gains by exact enumeration of the FOUR outcome probabilities, so reward evaluation contributes no Monte Carlo noise.
3. First run a population-exact trajectory with phi(t) = phi0 + rho*t. Verify the analytic safe boundary, initial worst gain approximately 0.0612592, continuous contact 0.00783587, and the exact finite-step first violation around 0.008.
4. Then use sampled policy gradients with n = 256, 2048, 8192 observations per policy update, initially 5 fixed seeds (a development-level comparison, not a statistical proof). Use common pre-generated uniform draws for q and s in matched branches; record both the sampled gradient and its exact expectation at every state.
5. Compare the first failure times, the fraction remaining certified, the true finite-step gain rates, and the policy state errors against exact dynamics. If the N=8192 first-loss distribution is within 1–2 policy steps of the exact reference, do not increase the batch further. If not, report the gradient and certificate error together and increase only the sampling precision necessary to decide whether the difference is numerical, not millions of samples by default.
6. Also run a half-step exact population trajectory h = 0.00025 to establish that the numerical crossing resolution is understood. A half-step stochastic repeat is optional, not required until there is a discrepancy.

**Implement a clean logged sample gradient:**

~~~python
# At each actual policy checkpoint, not a new environment
q = 2 * (u_q < (1 + x) / 2) - 1
s = 2 * (u_s < (1 + y) / 2) - 1
reward = phi * q + (1 - phi) * s
grad_theta = mean(reward[:, None] * stack([q-x, s-y], axis=1))
theta_next = theta + h * grad_theta
# Derive x_next,y_next = tanh(theta_next), and evaluate
# ALL FOUR outcomes exactly for both true and robust gains.
~~~

**Fill Table 5A:**

| Policy update | h | Samples/update | Seeds | First original-certificate loss | Pointwise value remains > gamma? | True finite-step gain near contact | Max / mean exact-gradient error | Notes |
|---|---:|---:|---:|---|---|---|---|---|
| Exact population, reference | 0.0005 | exact | 1 | 0.008 | yes; minimum 0.11355 | 0.11603 | 0 | The step starting at 0.0075 is still 0.06003. Continuous contact is 0.00783587. Initial certificate 0.061259. True return 0.674 is not this gain |
| Exact population, half step | 0.00025 | exact | 1 | 0.008 | yes; minimum 0.11355 | 0.11604 | 0 | Last certified step starts at 0.00775, so the crossing lies in (0.00775, 0.008] |
| Sampled REINFORCE | 0.0005 | 256 | 5 | 0, 0.0005, 0, 0.0005, 0.0005 | yes; minimum 0.11351 | mean 0.11366 | 0.151 / 0.0484 | The 0.001 initial slack is smaller than the gradient error, so the certificate fails on the first or second step |
| Sampled REINFORCE | 0.0005 | 2048 | 5 | 0.001, 0.0005, 0, 0.001, 0.0015 | yes; minimum 0.11354 | mean 0.11349 | 0.0441 / 0.0175 | True gain at t=0.008 remains near the exact value |
| Sampled REINFORCE | 0.0005 | 8192 | 5 | 0.001, 0.0005, 0, 0.001, 0.002 | yes; minimum 0.11355 | mean 0.11523 | 0.0264 / 0.00862 | Still 16 steps earlier than the exact loss. Not within the planned 1–2 steps |
| Sampled REINFORCE, precision check | 0.0005 | 2097152 | 5 | 0.003, 0.0035, 0.002, 0.003, 0.004 | yes; minimum 0.11355 | mean 0.11593 | 0.00165 / 0.000555 | Added because 8192 did not resolve the loss time. True gain now matches 0.11603; the certificate still trips while its slack is below the remaining gradient error |

**Possible outcomes and what they mean:**

- Exact crosses near 0.008 and sampled paths cluster nearby: **sampling of the real policy update preserves the mechanism**.
- Exact crosses, but sampled paths scatter: quantified policy-gradient noise is the new factor. Report gradient-error versus boundary error, and do not blame preference learning.
- Exact does not cross or leaves the theorem's box: the implementation/model has changed. Fix this exact reference before interpreting stochastic branches.
- All sampled paths stay above gamma beyond the analytic all-controls upper bound: verify that phi movement, reward family, step units and ex-post/prospective gains are consistent before claiming a genuine difference.

### Step 5B — Online Bradley–Terry feedback, with the policy trajectory held FIXED

**Status:** RUN. This step introduces only statistical preference information, not a new guidance optimizer.

**Objective:** Demonstrate how actually observed informative preference labels alter the certificate while **the exact same policy trajectory and guidance coefficients are retained**. This is the ordinary-PG counterpart of the already successful five-output Step 2.

**Source comparison must match Theorem 2, not the previous artificial zero-feature shortcut:**

- True reward r_a(q,s) = a q + (1-a)s and original compatibility U0 = [-1,2].
- Source pairs compare outcomes with q = s, such as (+1,+1) versus (-1,-1). Their BTL logit is 2 and **does not depend on a**. Repeating these comparisons cannot distinguish a.
- Informative pairs compare (+1,-1) versus (-1,+1). Their BTL logit is **4a - 2**. Choose a fixed orientation and label convention, verify the probability with direct reward differences.
- Hidden true a_star remains the already declared Step 5 value phi0. It is accessible to the teacher and the exact evaluator only, never the policy-gradient optimizer or guidance selector.

**How to fit and certify with minimum new machinery:** Parameterize delta = a - 0.5 in [-1.5,1.5], so the informative comparison logit is 4*delta. Reuse the existing horizon.btl_confidence_interval(successes, labels, sensitivity=4, uncertainty_radius=1.5, alpha=0.05) and convert the endpoints back to a by adding 0.5. Intersect with U0. This is a fixed-N confidence set, not an ensemble uncertainty proxy or a discretized simplex.

**Execution:**

1. Select fixed feedback arrival **t_a = 0.004**, eight policy steps for h=0.0005, before the original theoretical contact. Use identical source-only trajectories from Step 5A; freeze theta, phi and the policy-gradient RNG stream. Source information before the arrival remains U0.
2. Draw source BTL comparisons, confirm that the likelihood over a is flat. Use this control once, at fixed N.
3. For informative pairs use N = 32, 128, 512 (small-to-large), starting with five fixed independent label seeds. Do NOT change the policy trajectory or its random action draws in response to these labels in 5B.
4. For each fixed batch, compute a_hat from the exact scalar binomial likelihood, U_N from the fixed-N interval, and at every subsequent step compute C_{k,h}(U0) and C_{k,h}(U_N) on the IDENTICAL executed transition. The reported confidence set must contain a_star on the valid-coverage event; report any uncovered run explicitly.
5. Evaluate the terminal certificate at t=0.012 and the earliest violation after feedback. Pre-feedback certificates MUST be identical between all conditions.
6. Calculate exact binomial acceptance probabilities for the selected fixed-N design if they can be evaluated cheaply (N<=512), as in Step 2. If N=512 still fails to provide sufficient confidence in most outcomes, measure the remaining interval and decide whether 1200 labels is a necessary single escalation. Do not vary the true teacher coefficient to make success easier.
7. Repeat source-only comparisons at equal N. Their information cannot narrow a, regardless of how many times they are sampled.
8. On the fixed trajectory also log actual true finite-step gain under a_star. It is unchanged by an information-only intervention. An improved certificate **without a change in true gain** is the exact positive finding sought here.

**Fill Table 5B:**

| Information source | New labels | First possible update time | BTL estimate / interval for a | Original certificate loss | Updated certificate at t=0.012 | Updated certificate loss / censored | Exact true gain changed? | Notes |
|---|---:|---|---|---|---|---|---|---|
| Original source comparisons | 0 | none | [-1, 2] | 0.008 | 0.05939 | 0.008 | no | Frozen population path. True finite-step gain at t=0.008 is 0.11603 |
| Repeated source comparisons | 512 | 0.004 | [-1, 2] | 0.008 | 0.05939 | 0.008 | no | Pairs with q=s have logit 2 for every a, so the set does not shrink |
| Informative comparisons | 32 | 0.004 | five seeds all cover 0.843; seed 100 is 0.735 in [0.479, 1.287] | 0.008 | 0.0941, 0.1014, 0.0594, 0.0594, 0.0941 | 3 of 5 censored; 2 lose at 0.008 | no | Exact probability of a terminal certificate at least 0.06 is 0.314. The two failures are intervals truncated at 2 |
| Informative comparisons | 128 | 0.004 | five seeds all cover 0.843; seed 100 is 0.796 in [0.650, 1.012] | 0.008 | 0.1075, 0.1101, 0.0949, 0.1004, 0.1075 | 5 of 5 censored | no | Exact probability 0.992. Exact coverage of a_star is 0.999. Sufficient for Step 5C |
| Informative comparisons | 512 | 0.004 | five seeds all cover 0.843; seed 100 is 0.821 in [0.740, 0.921] | 0.008 | 0.1119, 0.1092, 0.1094, 0.1116, 0.1112 | 5 of 5 censored | no | Exact probability 1. No escalation to 1200 labels |

For stochastic labels, provide per-seed values and an aggregate (fraction of seeds meeting gamma), NOT just the friendliest batch.

**Possible outcomes:**

- Informative labels narrow a and raise the certificate while source pairs do not: directly supports the information side in a new, ordinary-PG system.
- Labels shrink uncertainty but cannot certify the whole target: compute exact remaining worst-case margin; it is a measurable sample-size or feasibility issue.
- The certificate or true gain changes before t_a, or the true gain changes solely due to adding labels to a frozen path: branch leakage / implementation bug, not a scientific failure.

### Step 5C — Actual closed-loop preference learning and BOUNDED reward guidance

**Status:** RUN. The exact path from 5A crosses at 0.008, and 5B shows that 128 informative labels certify that frozen path.

**Objective:** Combine in one actual training trajectory: sampled policy-gradient updates, finite binary BTL preference learning, and gradual reward guidance adaptation. Make the new information affect a deployed guidance optimizer **without ever replacing the given initial phi0 instantaneously**.

**Core invariant:**

\[
\phi(0)=0.8428043202,\quad
0\le\phi_k\le1,\quad
|\phi_{k+1}-\phi_k|\le\rho h,\quad
\rho=0.04,\ h=0.0005.
\]

A BTL parameter estimate a_hat and the deployed guidance phi_k must remain separate variables.

**Controller: choose the simplest valid operation, not another neural optimizer.**

For each checkpoint, use current state x,y and current preference interval U. For each candidate phi in the permitted one-step movement interval, compute the worst-case **prospective expected gain rate**

\[
L_U(x,y,\phi)=\min_{a\in U}
\{a\omega_q\phi+(1-a)\omega_s(1-\phi)\}.
\]

Select the maximizer over the one-dimensional permitted interval, which is a concave piecewise-linear function: evaluate the endpoints and the crossing of the two endpoint reward lines if it lies inside. Tie-breaking must be fixed. This selection uses known four-outcome policy dynamics and statistically learned U, **not the hidden a_star**.

**Why this is a genuine bridge:** The preference interval comes from observed BTL labels, phi is chosen by a real bounded control update, and the policy step is estimated from actual sampled actions. Exact enumeration is used only to validate/measure the resulting update. This does NOT claim that unconstrained BTL likelihood training automatically gives the optimal guidance path.

**Step-by-step execution:**

1. Create an independent new script experiments/bridge/06_online_bandit.py. Provide explicit modes 5A, 5B, 5C, all using one shared four-output implementation and shared saved state. Preserve historical 05_contextual_bandit.py and old files untouched.
2. At t=0 save theta0, phi0, U0, random generators and any optimizer state. All branches start with the same theta0/phi0. Keep original source comparisons on q=s with no a-information.
3. Run through t_a=0.004 using the **same sampled REINFORCE updates** and original source U0. At t_a, save one exact checkpoint, with all RNG states, before any new label is processed.
4. Clone that checkpoint into these essential branches:
   - **Source-only guidance:** no new informative labels; online bounded guidance controller using U0.
   - **Information-only:** new N informative labels shrink U, but **continue the original frozen guidance and policy trajectory from 5A**; report updated certificate without changing behavior (the 5B arm).
   - **New information + bounded guidance:** same N labels; choose phi each step by the bounded robust controller above; actual sampled policy update follows.
   - **BTL fitted coefficient + bounded deployment:** use the same BTL labels to estimate a_hat, set target phi_target = clip(a_hat,0,1), but move toward that target with |delta_phi|<=rho*h. This separates a conventional point-estimate reward deployment from uncertainty-aware guidance. It is a comparator, not a replacement for the robust controller.
5. Use N=128 as the first feedback budget (5B has already measured its information value); increase to N=512 only if the measured confidence interval or run variability requires it. Use **the same exact new labels** in both learned-information branches. If the fixed-N interval excludes a_star on a rare sampled event, report it and keep the seed; do not silently resample.
6. Use exactly h=0.0005, 24 updates (to t=0.012) and n=8192 sampled actions/update initially. Run the population-exact version of the SAME controller first. Then five matched seeds of the sampled update. Reduce n only if 5A established no material loss of the target conclusion at a lower n.
7. Generate random action outcomes using a stored per-step sequence of uniforms so comparisons are reproducible and paired where possible. Do not reuse feedback labels as policy rewards. Each sampled policy update is REINFORCE on R_phi, while the hidden true a is used only by the teacher and evaluator.
8. At every checkpoint log theta, x, y, current phi, BTL N/wins, a_hat, U0, U_N, phi movement, exact population PG, sampled PG, instantaneous true gain rate, **executed finite-step true gain rate**, original source-only certificate, updated certificate on the SAME realized transition, unrestricted continuous pointwise value and one-step population pointwise optimum if needed.
9. Do not classify certificate failure based on **current J_a**. All progress comparisons use instantaneous or finite-step gain *per unit policy time*, not absolute true return.
10. Compare first original-source and updated-certificate crossing times, true finite-step gain, candidate feasibility and BTL information quality across the matched branches. A larger certified horizon need not yield visibly larger total return in this very short local model; report them separately.
11. Check both essential controls: no preference ambiguity means no original uncertainty-driven loss; source-only comparisons cannot shrink U; if rho is made sufficiently large in a separate control, the original speed limitation is removed. Do not build a large new experiment grid.
12. If the bounded robust controller and fitted-reward controller differ, identify whether the difference is from the compatibility interval or the BTL point estimate; do not imply optimizer learning advantage merely because they use different information.
13. If no difference remains beyond Monte Carlo noise, retain the exact population result, show the measured error term, and adjust sample count based on the already measured 5A precision—not by switching to a new benchmark.

**Pseudocode:**

~~~python
theta, phi, U = theta0, phi0, U_source
feedback_step = 8  # t=0.004, with h=0.0005
for k in range(24):
    x, y = tanh(theta)
    if k == feedback_step and method_uses_feedback:
        labels = fixed_BTL_labels_from_same_saved_checkpoint()
        a_hat, U = fit_one_parameter_BTL_and_fixedN_interval(labels)

    if k == 0:
        phi = phi0  # first policy update has EXACT initial guidance
    elif k < feedback_step:
        phi = predeclared_original_phi_at(k*h)  # common pre-feedback path
    else:
        low_phi = max(0, phi - rho*h)
        high_phi = min(1, phi + rho*h)
        if method == "bounded_robust":
            phi = argmax_piecewise_linear_L(U, x, y, low_phi, high_phi)
        elif method == "bounded_point_estimate":
            phi = clip(a_hat, low_phi, high_phi)
        else:
            phi = predeclared_original_phi_at(k*h)  # unchanged-path arm

    sampled_gradient = reinforce_sample_gradient(theta, phi, n, same_uniforms[k])
    theta_next = theta + h * sampled_gradient
    exact_metric_record = enumerate_all_four_outputs(
        theta, theta_next, U_source, U, a_star_evaluator_only
    )
    theta = theta_next
~~~

**Important timing convention:** Ensure the published phi0 is the guidance parameter of the first policy step. The movement constraint applies BETWEEN successive steps; do not permit an extra adaptation jump at k=0 or feedback arrival.

**Fill Table 5C:**

| Branch | Preference labels | Guidance update | Seeds | First original-certificate loss | First updated-certificate loss | True finite-step gain near contact | Phi movement within rho*h? | Notes |
|---|---:|---|---:|---|---|---|---|---|
| Original source only, bounded robust | 0 | restricted to U0 | 5 | 0.003, 0.0035, 0.002, 0.003, 0.004 | same times, because U stays U0 | mean 0.11593 | yes | Sample size 2097152, not 8192: 5A showed that 8192 moves this loss by 16 steps. The population schedule still loses at 0.008 |
| Information-only, original frozen trajectory | 128 | no changed trajectory | 5 | 0.008 | censored for all 5; terminals 0.1075, 0.1101, 0.0949, 0.1004, 0.1075 | 0.11603 | yes | Exact 5A path and five label seeds. Labels do not move phi or the true gain |
| New information + bounded robust | 128 | restricted to U_N | 5 | 0.003, 0.0035, 0.002, 0.003, 0.004 | four early losses remain; the 0.004 loss is censored after feedback. Terminal updated certificates are about 0.1075 | mean 0.11593 | yes | This interval contains the robust target, so the speed limit reproduces the original upward schedule. Phi matches the source-only branch |
| New information + bounded point estimate | 128 | toward a_hat=0.796, speed limited | 5 | same pre-feedback times as the robust branch | same early pattern; terminal updated certificates are about 0.1074 | mean 0.11595 | yes | After t=0.004 this branch decreases phi while robust increases it. The phi gap at the last step is 0.00064, and the terminal source certificate falls to about 0.0583 |
| Population-exact control reference | 128 | same robust controller, no sample noise | 1 | 0.008 | censored; terminal updated certificate 0.10749 | 0.11603 | yes | Phi matches the original schedule to 3e-16. A known singleton preference keeps the updated certificate and moves the source loss to 0.0065. Raising rho to 100 removes the source loss; its terminal source certificate is 0.1135 |

**Possible outcomes and interpretation:**

- Original source-only certificate expires near the known horizon, while informative labels preserve an updated certificate on the unchanged trajectory: **the theory-to-feedback link has been established**.
- The bounded robust controller differs from clipped or point-estimate deployment while respecting exactly the same rho: evidence of optimizer/controller realization under fixed resources, with scope limited to this four-output model.
- Additional information certifies the same trajectory but does not improve true return: **expected information-only effect**, not a failed method.
- New feedback changes phi but the source-only certificate still expires: source data remain unchanged; that is why the source-only diagnostic remains a distinct metric.
- BTL information is accurate but guidance performance worsens: the new issue is how the feedback is deployed, not preference estimation or the original theorem.
- A branch violates phi movement budget, source labels change the true evaluator or the initial state differs: invalid implementation, repair before new experiments.
- Gradient sampling noise dominates the improvement threshold: increase n only enough to resolve the fixed gamma comparison; exact population branch remains the known reference.

### Step 5A–5C final deliverables and acceptance

**Primary deliverable:** Return THIS updated Markdown plan with every 5A–5C table filled; preserve all Step 0–5 original rows exactly as historic outputs, correcting interpretations only in the new results. An optional short execution note cannot replace the filled plan. In every table the last Notes column must describe any deviation or limitation; leave it blank when none.

**Expected source and output layout (scripts marked to implement, not yet present):**

~~~text
experiments/bridge/06_online_bandit.py   # shared 5A / 5B / 5C
outputs/horizon_bridge/step05a/
outputs/horizon_bridge/step05b/
outputs/horizon_bridge/step05c/
~~~

**Suggested CLI interface to implement:**

~~~bash
python experiments/bridge/06_online_bandit.py --stage 5A --out outputs/horizon_bridge/step05a
python experiments/bridge/06_online_bandit.py --stage 5B --out outputs/horizon_bridge/step05b
python experiments/bridge/06_online_bandit.py --stage 5C --out outputs/horizon_bridge/step05c
~~~

All three modes must save a single run summary plus **per-step, per-seed raw results**, configuration, numerical parameters, random seeds, code revision and measured wall time. Do not just print the final mean. Include 5A's exact references in 5C's diagnostic plots/tables.

**Runtime and compute:** The system has only four outputs. Exact evaluation should be cheap on CPU. The default n=8192, 24 updates, five seeds, and three actual training branches require only a few million simulated action samples, not an LLM or SAC training run. Record measured seconds and peak CPU RAM; if the program is unexpectedly slow, vectorize the four-outcome sampling/score-gradient computation before switching to GPU. There is no reason to rent a larger accelerator for this step.

**One-step stopping conditions (not a request for approval):**

- 5A: if the population model does not reproduce the known boundary, fix it before 5B; if stochastic estimates are noisy, measure and isolate the error, not another model.
- 5B: if the confidence set is not valid or the source comparisons unexpectedly reveal a, fix the preference-label model before 5C.
- 5C: if either model fits but differs from the exact prediction, identify the new component. Do not hide failed assumptions by selecting another true preference/initialization.
- Once the loop is scientifically coherent, return the completed plan. **Do not start another Deep-Sea-Treasure, Walker, Meta-World or LLM run as part of this update.**

**Fill the new result locations:**

| Step | Main finding | Raw data / summary path | Source revision | Measured runtime / peak memory | Notes |
|---|---|---|---|---|---|
| 5A | Exact finite-step loss starts at 0.008 while true gain stays 0.116. Sampling noise, not the true gain, moves the certificate | `outputs/horizon_bridge/step05a/` | 7807ac1 | 4.05 s, 1.61 GB RSS | The precision row uses 2097152 samples |
| 5B | Informative labels raise the frozen-path certificate. Source pairs do not, and the true gain does not change | `outputs/horizon_bridge/step05b/` | 7807ac1 | 0.016 s, 99 MB RSS | N=128 already has exact success probability 0.992 |
| 5C | On the exact path, bounded robust guidance with this interval follows the original schedule; the point estimate moves the other way. Sampling still trips the source certificate before feedback | `outputs/horizon_bridge/step05c/` | 7807ac1 | 11.67 s, 3.91 GB RSS | Figure `certificates.png`. No Walker, Meta-World, or language run |

**Terminology clarification:** For Step 5 and its successors, write **true return** for J_a(theta), **true instantaneous gain rate** for the population derivative, and **true finite-step gain rate** for (J_a(theta_next)-J_a(theta))/h. A model that records only J_a cannot claim to have compared progress to the threshold gamma. The fitted preference a_hat and the deployed guidance phi are never the same state variable.

### Later expansion, NOT part of the present run

After completion of Steps 5A–5C, the **next** plan may consider Deep-Sea-Treasure with exact small-MDP policy evaluation, followed by a continuous control SAC experiment. Before testing Horizon in Walker, a trusted SAC baseline **using the true environment reward** must demonstrate real learning. If that basic control does not work, the failure is implementation or training quality, not a Horizon result.

No new Meta-World, PPO or LLM experiment is authorized by this plan. Complete Steps 5A–5C first. Those require a bridge result and a separate scientific question.

## 4. Interpretation Matrix: Every Outcome Must Say What Failed

| Observed outcome | Definite interpretation if prior controls hold | Action |
|---|---|---|
| Exact five-output finite-step identity fails | Wrong simulator/equation/assumption translation | Repair exact implementation |
| Step 1 identity holds and detected loss moves with h | Resolution of discrete detection, not new phenomenon | Record convergence |
| Fixed-N BTL interval/test conflicts with exact binomial | Inference implementation or fixed-sample assumption error | Repair estimator |
| Informative pairs shrink uncertainty; source-only pairs do not | Evidence identifies the missing preference direction | Move to guidance update |
| Added labels change certificate but not actual gain | Correct *information-only* effect on frozen path | Keep this important result |
| Larger movement budget helps envelope but optimizer lags | Optimizer-specific startup or movement shortfall | Compare realized movement |
| Exact ordinary PG works; sampled estimate fails | Gradient sampling/finite-step noise | Quantify needed precision |
| Sampled PG works; learned reward does not | Preference model/data fit issue | Fix BTL/reward component |
| All above work; natural RL task does not learn | New control/benchmark implementation issue | True-reward oracle learning first |
| Natural task learns but has no certificate loss | The required mechanism is not active in that task/window | Report this scope result honestly |

The requirement is **not** that every possible real environment must yield a positive effect. The requirement is that every experiment is constructed so its outcomes identify a scientifically meaningful condition.

## 5. Deliverables and Runtime

**Primary deliverable: this same completed Markdown file.** Fill all result tables, and fill the paths below. An optional supplementary execution report is allowed but never replaces the completed plan. Notes must be filled for protocol deviations, otherwise may remain empty.

For every step, store exact configuration, raw per-step data, summary JSON, source SHA, executed command, runtime and memory measurements. Do not replace the old 01 results or edit results/reference. Suggested output structure:

~~~text
experiments/bridge/
  01_finite_policy.py
  02_preference_feedback.py
  03_guidance_updates.py
  04_ordinary_pg.py
  05_contextual_bandit.py          # historical pilot, preserve unchanged
  06_online_bandit.py
outputs/horizon_bridge/
  step01/
  step02/
  step03/
  step04/
  step05/
  step05a/
  step05b/
  step05c/
~~~

**Resources:** Steps 0–3 use only tiny four/five-output systems and should normally need CPU, not a rented 5090. Step 4 begins with exact CPU expectations. Step 5 can use CPU or the existing GPU. Run a few seconds of each new procedure first, recording throughput, RAM and solver stability; optimize tasks forecast to exceed an hour. Do not burn rented GPU time on unrelated benchmarks while basic assumptions remain unchecked.

| Step | Executed command | Runtime | CPU RAM / GPU VRAM | Summary path | Status | Notes |
|---|---|---|---|---|---|---|
| 0 | `python -m pytest -q`; `ordinary_pg_check`; `python experiments/discrete_resource_check.py --out outputs/horizon_bridge/step00_discrete` | pytest 2.33s; adaptive BTL call 0.54s | CPU only, no GPU | `outputs/horizon_bridge/step00/summary.json` | PASS | Fresh `check_btl_anytime` radius 0.439032. Earlier full reproduction recorded 0.469073. Reference remains 0.426380 and was not edited |
| 1 | `python experiments/bridge/01_finite_policy.py --out outputs/horizon_bridge/step01` | 0.273s | CPU only | `outputs/horizon_bridge/step01/summary.json` | PASS | Published script does not record RSS |
| 2 | `python experiments/bridge/02_preference_feedback.py --out outputs/horizon_bridge/step02` | 0.017s | 97 MB RSS, no GPU | `outputs/horizon_bridge/step02/summary.json` | PASS | |
| 3 | `python experiments/bridge/03_guidance_updates.py --out outputs/horizon_bridge/step03` | 0.260s | 100 MB RSS, no GPU | `outputs/horizon_bridge/step03/summary.json` | PASS | |
| 4 | `python experiments/bridge/04_ordinary_pg.py --out outputs/horizon_bridge/step04` | 18.19s | 305 MB RSS, no GPU | `outputs/horizon_bridge/step04/summary.json` | PASS | |
| 5 | `python experiments/bridge/05_contextual_bandit.py --out outputs/horizon_bridge/step05` | 0.014s | 79 MB RSS, no GPU | `outputs/horizon_bridge/step05/summary.json` | PASS | Historical eight-step pilot; its logged true gain is true return |
| 5A | `python experiments/bridge/06_online_bandit.py --stage 5A --out outputs/horizon_bridge/step05a` | 4.05s | 1.61 GB RSS, no GPU | `outputs/horizon_bridge/step05a/summary.json` | PASS | Includes the 2097152 precision check |
| 5B | `python experiments/bridge/06_online_bandit.py --stage 5B --out outputs/horizon_bridge/step05b` | 0.016s | 99 MB RSS, no GPU | `outputs/horizon_bridge/step05b/summary.json` | PASS | |
| 5C | `python experiments/bridge/06_online_bandit.py --stage 5C --out outputs/horizon_bridge/step05c --labels 128` | 11.67s | 3.91 GB RSS, no GPU | `outputs/horizon_bridge/step05c/summary.json` | PASS | |

**Result Location — fill after execution**

| Step | Main result | File or folder | Source commit | Notes |
|---|---|---|---|---|
| 0 | Anchors match. Adaptive BTL radius does not match the frozen reference | `outputs/horizon_bridge/step00/` | a6d00ff | `results/reference` was not modified |
| 1 | Finite-step identity holds. First strict loss is 5+h. True gain at T=15 is 1 | `outputs/horizon_bridge/step01/` | a6d00ff | |
| 2 | Informative labels shrink delta. Source-only labels do not. Delta 0.4 does not certify T=15 | `outputs/horizon_bridge/step02/` | a6d00ff | Script added with this completed plan |
| 3 | Information, movement budget, and clipped-step size are separate. Robust m=1 meets the envelope at 5.05; clipped m=1 loses at 4.55 | `outputs/horizon_bridge/step03/` | a6d00ff | |
| 4 | Exact finite PG brackets 0.007836. Sampling matches once the gradient error is below 0.001. A 1200-label reward head does not | `outputs/horizon_bridge/step04/` | a6d00ff | |
| 5 | Eight-step bandit matches the exact four-outcome model. An unconstrained fitted phi loses the certificate while true gain stays near 0.675 | `outputs/horizon_bridge/step05/` | a6d00ff | No Walker, Meta-World, or language run |

### Unambiguous execution order

1. Run Step 0 reference checks and the new Step 1 script. Compare with exact numbers.
2. Implement Step 2 using the existing horizon.py BTL functions and exact binomial probabilities; **do not modify the policy path**.
3. Implement Step 3 by reusing the published discrete calibration and finite policy dynamics. Distinguish updated information from realized guidance movement.
4. Independently reproduce the existing ordinary Euclidean PG construction, then add only gradient sampling.
5. Preserve the existing Step 5 result as a short historical pilot. It ends at t=0.004 and its logged true_gain is actually current true return.
6. Run Step 5A (24+ steps, true finite-step gain), then 5B (BTL information on the same frozen trajectory), then 5C (joint BTL / bounded guidance / sampled PG).
7. Return **this completed plan**, including the newly filled 5A–5C tables, raw results and deviations.

## 6. Terminology

- **Guidance reward \(\phi\):** reward parameter actually inducing a policy change; not interchangeable with true preference \(\delta\) or its BTL estimate.
- **Source-only certificate:** worst compatible gain under original source feedback, fixed when asking when that information expires.
- **Updated certificate:** recomputation after new informative preference labels, possibly on an unchanged policy trajectory.
- **Pointwise feasibility:** existence of a certifying reward at one state; finite candidate search provides only a candidate-restricted statement.
- **Trajectory feasibility:** maintaining the certificate from the initial reward under an allowed path, not just solving each state independently.
- **Support horizon:** latest interval retaining required robust gain; discrete detection uses the first strictly violating step.
- **Actual optimizer versus movement envelope:** actual \(\phi\) updates versus their maximum allowed displacement.
- **True finite-step gain:** independently measured change in task utility under the evaluator-only true preference, distinct from the certificate.
- **BTL confidence set:** preference uncertainty supported by fixed-N binary comparisons and a stated coverage guarantee, not a coarse empirical parameter grid.
