# 03 Stochastic Guidance Transfer

**Project:** The Limits of Reward Guidance under Finite Adaptation  
**Purpose:** ICML 2027, the next controlled transition after the completed four-output online bridge  
**Prepared:** 2026-10-19  
**Status:** G1–G4 results preserved; G4.1 exact two-step extension completed and filled below  
**Verified starting commit:** 04193e61998298991b4e544578c5eef37865ec83  
**Repository:** https://github.com/AtheavenMa/The-Limits-of-Reward-Guidance-under-Finite-Adaptation

> **Executing agent: this document is the assignment and the final report.** Implement G1–G4 in order, fill every preallocated table and Section 7, and return the completed Markdown. Do not submit an alternative narrative report instead. This work plan advances only from already verified constructions. Do NOT restart Walker, Meta-World, Qwen, TL;DR, PPO, or a large benchmark as part of this assignment.

## 1. Paper and Experiment Overview

### 1.1 Research question

The paper establishes that continued preference-guided improvement is different from having a suitable guidance reward at any individual policy state. Under a finite rate of reward adaptation, reward settings that guarantee progress can outrun the guidance optimizer. In the scalar model, Theorem 1 gives an exact support horizon. Theorem 2 exhibits the separation locally under ordinary Euclidean policy gradient. Theorem 3 studies the Bradley–Terry information required to certify a longer horizon.

This plan **does not re-prove those theorems**. It asks what happens when one moves from the paper's exact population policy update to a genuinely sampled policy update, and then adds enough preference data to influence *which* reward update should be taken.

The experiments must keep distinct:

- **Pointwise feasibility:** whether a qualifying guidance reward exists at a current state.
- **Trajectory feasibility:** whether the reward can follow the moving certified region from its declared initialization and adaptation rate.
- **Statistical certification:** what the source or newly collected preference labels establish.
- **Realized stochastic policy change:** what one finite Monte Carlo batch actually causes.

These last two objects cannot be interchanged. The original continuous population theorem does NOT claim that every finite-sample policy-gradient step has the same lower bound.

### 1.2 What the latest bridge actually established

The preceding Work Plan [02 Theory to Training Bridge](02_Theory_to_Training_Bridge.md) and committed [Step 5A–5C code](../experiments/bridge/06_online_bandit.py) report:

| Existing result | Committed evidence | Correct interpretation |
|---|---|---|
| Theorem 2 ordinary-PG continuous contact at about 0.00783587 | outputs/horizon_bridge/step04/summary.json | Analytic/ODE reference |
| Exact finite-step four-output policy loses original certificate at time 0.008 | outputs/horizon_bridge/step05a/summary.json | Population-policy finite-step realization |
| Sampled REINFORCE updates lose the **ex-post** source certificate much earlier | same file, sample sizes 256, 2048, 8192, 2097152 | A distinct stochastic phenomenon; not yet a failure of the population model |
| Informative fixed-N Bradley–Terry labels shrink the preference interval | outputs/horizon_bridge/step05b/summary.json | 128 labels certify the frozen population path through 0.012 in 5/5 specified label samples |
| Repeated source comparisons do not identify the unknown preference | same file | Negative information control passes |
| BTL + bounded reward guidance + sampled policy updates were jointly executed | outputs/horizon_bridge/step05c/summary.json | Correct implementation direction, but 4/5 sampled seeds lose the ex-post source certificate **before** feedback arrives |
| The BTL informed robust guidance is effectively the same path as source-only robust guidance at N=128 | same file and source code | Evidence of improved information/certification, NOT demonstrated improvement of the guidance controller or true policy performance |

The current paper has valuable evidence for the **information-only benefit** and exact finite-step separation. It does not yet have a demonstrated practical return advantage from deciding to collect labels or change a controller.

### 1.3 New finding from independent review of saved Step 5A/5C traces

This is the most important reason for the next plan.

The old code logs:

\[
C^{\mathrm{real}}_{k,h}(U_0)=
\inf_{a\in U_0}\frac{J_a(\theta_k+h\widehat g_k)-J_a(\theta_k)}{h},
\]

where \(\widehat g_k\) is one sampled REINFORCE gradient. This is an **ex-post gain on the realized random step**.

An independent recomputation from outputs/horizon_bridge/step05a/steps.csv and step05c/steps.csv also evaluated the *population-gradient counterfactual at the SAME sampled state*:

\[
C^{\mathrm{pop}}_{k,h}(U_0)=
\inf_{a\in U_0}
\frac{J_a(\theta_k+h g_k)-J_a(\theta_k)}{h},
\quad
g_k=\mathbb E[\widehat g_k\mid\theta_k,\phi_k].
\]

This population-step counterfactual is NOT exactly the conditional expectation of the nonlinear random-step gain. Preserve the distinction.

The recomputation finds:

| Samples per actual policy update | Observed first ex-post certificate loss, five seeds | First population-step counterfactual loss at the same sampled states |
|---:|---|---|
| 256 | 0–0.0005 | 0.0075–0.008 |
| 2048 | 0–0.0015 | 0.008 |
| 8192 | 0–0.002 | 0.008 |
| 2097152 | 0.002–0.004 | 0.008 |

These are **preliminary independently recomputed values** from committed trace CSV, NOT results of G1. Implement and save the recomputation as part of G1 so the executing agent can verify the numbers.

The mechanism can be understood without expensive training. At the initial point \(x=.8,y=0,\phi=.84280432\), the worst source preference is \(a=2\), with population gain rate \(0.0612592\) against required \(\gamma=.06\). The margin is only \(0.0012592\). Under the actual REINFORCE score estimator, exact enumeration of its four possible single-sample outcomes gives a standard deviation of approximately \(0.8550\) for the **first-order worst-preference gain-rate estimator**. Thus its standard error is approximately \(0.00945\) for 8192 actions and \(0.000590\) for 2097152 actions. These are analytical first-order variance estimates, not a claim of exact coverage or an exact finite-step random-gain distribution.

This explains why more than two million samples can still yield early **first** ex-post violations across 24 repeated updates. The correct next experiment is to separate the two quantities, NOT repeatedly double the sample size.

### 1.4 Scope and stage order

| Stage | Change from verified source | Principal question | Scientific output |
|---|---|---|---|
| G1. Population versus realized gains | **No new environment or reward model**; re-evaluate existing stochastic trace correctly | Does the stochastic path remain near the analytic support horizon when measured with its population policy update? | Exact four-outcome variance, conditional reference and realized-gain comparison |
| G2. Feedback on sampled states | Add finite BTL information to the **same sampled trajectory**, without changing behavior | Can we separate information-based certification of the new policy state from early stochastic update noise? | Evidence-only effect, pre-feedback and post-feedback loss treated separately |
| G3. Reward guidance decision | Use enough informative labels for the fitted preference set to change the direction of the **bounded** guidance controller | When does new information alter behavior, as opposed to merely raising a certificate? | Explicit threshold, gradual control change, true gain comparison |
| G4. Small sequential environment | Add exactly one short, enumerable transition and delayed reward to the proven four-output system | Is the separation visible when a policy produces a short trajectory rather than a one-step outcome? | A genuine two-step finite MDP with exact evaluation and sampled-gradient cross-check |

**Gate:** G1–G3 are independently implementable using the existing four outcomes. G4 should begin after their common metric definitions are verified. G4 does not depend on achieving a positive performance advantage in G3.

## 2. Execution Principles

1. **Start with the paper's code and fixed parameters.** For G1–G3, keep \(x_0=.8,y_0=0,\phi_0=.8428043202,\rho=.04,\gamma=.06,h=.0005\), 24 updates and the original preference class \(a\in[-1,2]\). Do not change the environment or threshold to find a convenient crossing.
2. **Change one cause per experiment.** A loss in a stochastic policy update must not be attributed to BTL reward learning if no learned preference has affected the update yet.
3. **Keep the theorem's comparison explicit.** A sampled ex-post certificate is not the population-policy certificate. Report both, at the same state, rather than choosing one based on which looks favorable.
4. **No unjustified sample escalation.** The old 2097152-samples/update run already shows that estimating every realized gain with extreme precision is an inefficient research strategy. Use exact four-outcome sums to explain the stochastic discrepancy. Initial stochastic comparisons use 256/2048/8192 samples and a small number of seeds; increase only if a specific planned contrast is unresolved.
5. **Keep BTL preference and deployed reward guidance separate.** The hidden \(a_\star\), its fitted estimate \(\widehat a\), the confidence interval \(U_N\), and deployed \(\phi\) are four different objects.
6. **Reward amplitude/normalization matter.** Every policy gradient used in G1–G3 must be generated by \(R_\phi(q,s)=\phi q+(1-\phi)s\). Do not use an unnormalized fitted two-parameter head that changes the update scale.
7. **Distinguish evidence from action.** Updating \(U_0\to U_N\) without changing the policy or \(\phi\) can improve the certificate but cannot change true return. Record it as a separate positive finding.
8. **A past violation cannot be retroactively undone.** If an ex-post update failed at \(t=0.003\) and feedback arrives at \(t=0.004\), a higher certificate at \(t=0.010\) does not mean the whole path was certified. Record pre-feedback history, post-feedback interval and whole-path status separately.
9. **Use genuine held-out evaluation in larger environments, but exact sums now.** Exact enumeration makes G1–G4 cheap and removes unnecessary Monte Carlo noise from evaluation.
10. **A scientifically informative negative is acceptable; an uninterpretable experiment is not.** Unexpected results must have a comparison that identifies the failed premise or newly introduced variable. Never fabricate or select favorable seeds.
11. **Preserve everything already submitted.** Do not overwrite outputs/horizon_bridge/step00–step05c, experiments/horizon_real or results/reference.
12. **Finish this plan by filling it.** The most important deliverable is this very Markdown with all tables, statuses and actual file locations. Extra prose reports are optional, not replacements.

### 2.1 Experimental definitions used throughout

For the independent Bernoulli policy,

\[
\pi_\theta(q,s)=\tfrac14(1+xq)(1+ys),\quad
x=\tanh\theta_q,\ y=\tanh\theta_s,\quad(q,s)\in\{-1,1\}^2.
\]

Guidance reward and true preference reward:

\[
R_\phi(q,s)=\phi q+(1-\phi)s,
\quad
r_a(q,s)=a q+(1-a)s,\quad a\in[-1,2].
\]

Exact population gradient in \(\theta\):

\[
g_k=\big(\phi_k(1-x_k^2),\ (1-\phi_k)(1-y_k^2)\big).
\]

One-sample score gradient from a genuine sampled output is

\[
Z(q,s)=R_{\phi_k}(q,s)(q-x_k,\ s-y_k),
\quad
\widehat g_k=\frac1n\sum_{i=1}^{n}Z(q_i,s_i).
\]

Policy update uses \(\theta_{k+1}=\theta_k+h\widehat g_k\); true return \(J_a=a x+(1-a)y\). A **finite-step gain rate** is \([J_a(\theta_{k+1})-J_a(\theta_k)]/h\), never the absolute return \(J_a\).

Define three distinct checkpoint quantities:

\[
C^{\mathrm{real}}_k(U)
 = \inf_{a\in U}
 \frac{J_a(\theta_k+h\widehat g_k)-J_a(\theta_k)}{h},
\]

\[
C^{\mathrm{pop}}_k(U)
 = \inf_{a\in U}
 \frac{J_a(\theta_k+h g_k)-J_a(\theta_k)}{h},
\]

\[
C^{\mathrm{inst}}_k(U)
 = \inf_{a\in U}
 \big[a\omega_q\phi_k+(1-a)\omega_s(1-\phi_k)\big],
\quad \omega_q=(1-x_k^2)^2,\ \omega_s=(1-y_k^2)^2.
\]

For this four-output reward family all three worst-case problems minimize an expression affine in \(a\), so evaluate the **two interval endpoints exactly**. \(C^{\mathrm{pop}}\) is a population-gradient counterfactual at the sampled state and \(C^{\mathrm{inst}}\) is a continuous-time reference. Neither automatically certifies each realized random update.

The original adaptation condition is \(|\phi_{k+1}-\phi_k|\le\rho h=0.00002\), with the prescribed initial \(\phi_0\). The pointwise population optimum and safe interval come from Section 5 / Theorem 2 and must be recorded independently of stochastic sampling.

## 3. Experimental Plan

### Experiment G1 — Identify the source of early stochastic certificate loss

**Objective.** Separate theoretical population dynamics from the random step actually applied by sampled REINFORCE. Explain the early loss *quantitatively*, using the already saved trajectories before deciding whether any new runs are needed.

**Paper-Level Support.** Supplies the missing empirical bridge from Theorem 2 to real sampled policy updates. It does not create a new general stochastic theorem.

**Steps:**

1. Start with existing outputs/horizon_bridge/step05a/steps.csv and step05c/steps.csv. Do not rerun or change historical outputs.
2. Implement experiments/bridge/07_stochastic_gain.py with a mode that reads those traces; for each recorded state, reconstruct \(\theta=\mathrm{arctanh}(x,y)\), compute \(g_k\) from the exact four-outcome expectation, and calculate \(C^{\mathrm{pop}}_k(U_0)\) and \(C^{\mathrm{inst}}_k(U_0)\).
3. Compare their first crossings of \(\gamma=.06\) with the stored **ex-post** \(C^{\mathrm{real}}_k(U_0)\). In this recorded five-seed case, the population-step reference should first fail near 0.008 across sample sizes. This checks that the stored sampled states did not drift far from the original theorem.
4. Independently enumerate the four outcome probabilities at each state, and compute the mean and covariance matrix of \(Z(q,s)\). For each endpoint \(a=-1,2\), set

   \[
   v_a=((1-x^2)a,\ (1-y^2)(1-a)),
   \qquad
   \mathrm{SE}_a(n)=\sqrt{v_a^\top\operatorname{Cov}(Z)v_a/n}.
   \]

   This approximates the standard error of the corresponding *first-order finite-step gain rate*.
5. Check the initial worst endpoint \(a=2\) and reproduce the expected approximately \(0.8550/\sqrt n\) scale. Report whether the difference between measured ex-post and population gain agrees in order of magnitude with that scale. Do not interpret a Gaussian approximation as a finite-sample confidence guarantee.
6. Plot **only one** clear trajectory example showing \(C^{\mathrm{pop}}\), \(C^{\mathrm{real}}\), \(\gamma\) and the original contact; optionally a second panel/table for the SE-to-margin ratio. Do not produce many visually redundant plots.
7. As a minimal additional smoke test if needed, run 10 NEW stochastic seed paths with n=2048 at h=.0005 for 24 steps using the existing exact environment and evaluate both metrics. The point is to verify repeatability, not to obtain 100% positive outcomes.
8. Compute the exact true finite-step gain rate under \(a_\star=0.8428043202\) for the same path, separately from the worst-case source certificate. Verify whether true gain remains positive even when the ex-post source certificate has failed.
9. Mark each first loss as population, instantaneous or realized. Do not conflate an update that happens to have a small realized gain with the statement that **all** admissible guidance controls are infeasible.

**Deliverable G1:**

| Samples per policy update | Seeds | First realized source loss | First population-step loss at same state | First instantaneous loss | Initial worst-endpoint SE / margin | True finite-step gain near 0.008 | Notes |
|---:|---:|---|---|---|---|---|---|
| Exact gradient | 1 | 0.008 | 0.008 | 0.008 | 0 | 0.116030 | Committed trace replays exactly. One-sample SD at a=2 is 0.855038; instantaneous margin is 0.0012592 |
| 256 | 5 existing | 0, 0.0005, 0, 0.0005, 0.0005 | 0.008, 0.0075, 0.008, 0.008, 0.0075 | 0.008 ×5 | 42.44 | mean 0.11366, all in 0.100–0.123 | First-step mean \|realized−population\| 0.0177; analytic SE 0.0534 |
| 2048 | 5 existing | 0.001, 0.0005, 0, 0.001, 0.0015 | 0.008 ×5 | 0.008 ×5 | 15.00 | mean 0.11349, all in 0.111–0.118 | Gap 0.00939 versus SE 0.0189. Ten new seeds 20–29: population loss 0.008 on 10/10, realized loss 0–0.003 |
| 8192 | 5 existing | 0.001, 0.0005, 0, 0.001, 0.002 | 0.008 ×5 | 0.008 ×5 | 7.502 | mean 0.11523, all in 0.112–0.118 | Gap 0.00778 versus SE 0.00945 |
| 2097152 | 5 existing | 0.003, 0.0035, 0.002, 0.003, 0.004 | 0.008 ×5 | 0.008 ×5 | 0.469 | mean 0.11593, all in 0.1158–0.1161 | Gap 0.000236 versus SE 0.000590. The Step 5C source-only path gives these same five times |

**Possible outcomes:**

- Population reference crosses at about 0.008, realized step much earlier, and analytic SE explains the scale: the earlier negative result is specifically **sampled policy-gradient noise in ex-post updates**, not wrong population theory.
- Both population and realized references move to earlier times: the sampled policy state drifted away from the exact trajectory; compare x,y,phi and gradient error to the expected state dynamics.
- Analytic variance and empirical variation differ markedly: examine sampling independence, the score function, reward scaling or metric code before moving on.

### Experiment G2 — New preference information on a sampled policy trajectory

**Objective.** Transfer the successful Step 5B information-only finding from the deterministic path to actual sampled-policy *states*, without changing actions, policy states or guidance parameters when adding labels.

**Paper-Level Support.** This is the immediate extension of the paper's information mechanism to actual stochastic policy updates, with the important caveat that a previous realized loss cannot be undone.

**Only change from G1:** A statistically justified updated Bradley–Terry preference interval \(U_N\) is computed from labels.

**Steps:**

1. Reuse the same sampled REINFORCE trajectories already stored for n=8192 in Step 5A. New sampling is not necessary for the first result.
2. Fix feedback arrival \(t_a=.004\), true hidden \(a_\star=.8428043202\), source set \(U_0=[-1,2]\), and informative comparisons whose logit is \(4a-2\). The source comparison is (q,s)=(1,1) versus (-1,-1), whose logit is 2 and reveals no a-information.
3. Generate N=128 informative labels independently from five predeclared label seeds. Use existing horizon.btl_confidence_interval with delta=a-0.5, sensitivity=4, uncertainty_radius=1.5. Reuse original Step 5B validated fitting/inference code.
4. For each sampled state trajectory and label interval, do NOT change policy transitions. Compute original and updated \(C^{\mathrm{pop}}\), \(C^{\mathrm{inst}}\), and the already realized \(C^{\mathrm{real}}\) from **the same actions**.
5. For each pair of sampled policy seed and preference-label seed, record separately:

   - first loss BEFORE feedback arrival;
   - first loss AFTER feedback arrival, conditional on starting that post-feedback interval;
   - whether the ENTIRE [0,.012] realized trajectory remained certified;
   - whether the post-feedback suffix [.004,.012] remained certified;
   - terminal source and updated certificates;
   - whether exact true finite-step gain changed (it must not, as trajectory is frozen).

6. Begin with 3 policy seeds × 3 label seeds, then complete the five-seed pairings if results or variance make it useful. Do not treat all crossed pairings as independent policy-training runs.
7. Include three information conditions: original source; 128 repeated source comparisons; 128 informative comparisons. Add N=512 informative comparisons **only if** 128 produces an unresolved updated certificate in the sampled policy states. Do not repeatedly tune the true preference or required gain.
8. To isolate policy sampling AFTER a certifiable history, also run one clearly labeled **controlled hybrid**: population-exact trajectory through feedback at .004, then sampled REINFORCE on the saved exact state for the remaining 16 steps, using the same label batches. This is a controlled transition, not evidence of an entirely sampled-from-start certified path.
9. Save seed-level time series. Never claim that the new labels restored a certificate already violated before their arrival.

**Deliverable G2:**

| Trajectory type | Preference evidence | Labels | Policy × label seeds | Population-step loss before / after feedback | Realized loss before / after feedback | Certified post-feedback suffix (fraction) | Certified entire path (fraction) | True gain changed? | Notes |
|---|---|---:|---:|---|---|---|---|---|---|
| Sampled from start | Original source | 0 | 5 | censored / 0.008 | 0–0.002 / 0.004 or 0.005 | 0/5 | 0/5 | no | Stored n=8192 prefixes. Population suffix also fails at 0.008 |
| Sampled from start | Repeated source pairs | 128 | 5×5 | censored / 0.008 | 0–0.002 / 0.004 or 0.005 | 0/25 | 0/25 | no | Every interval stays [−1, 2]. Certificates match the source row |
| Sampled from start | Informative pairs | 128 | 5×5 | censored / censored | 0–0.002 / censored | 25/25 | 0/25 | no | The 3×3 pilot subset is the same: suffix 9/9, entire path 0/9. Pre-feedback realized losses remain |
| Exact prefix, sampled suffix | Original source | 0 | 5 | censored / 0.008 | censored / 0.004 | 0/5 | 0/5 | no | Suffix RNG is 20000+policy seed, n=8192. The first sampled step loses the source certificate |
| Exact prefix, sampled suffix | Informative pairs | 128 | 5×5 | censored / censored | censored / censored | 25/25 | 25/25 | no | Controlled hybrid only. Not a path that was sampled from the start |
| Optional precision check | Informative pairs | 512 | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | no | N=128 already gave the same post-feedback answer on every pairing |

**Possible outcomes:**

- Population-step updated certificate stays positive through the target, but individual realized steps can still violate it: **information deficiency is reduced, sampling risk remains a separate issue**.
- Realized post-feedback suffix remains certified more often with new labels, while whole-path success is low because losses occurred before arrival: this is a useful time-specific finding, not a rescued continuous past.
- Even the exact-prefix sampled-suffix arm fails frequently: the remaining stochastic gradient variation, not historical contamination, is sufficient to explain failure.
- Source-only comparisons change the inferred a-interval: implementation/query construction error.

### Experiment G3 — Make preference information change the guidance decision

**Objective.** The current N=128 experiment makes a certificate larger, but the robust guidance controller chooses essentially the same trajectory as the source-only controller. Identify, **analytically first**, when enough new information changes the direction of optimal bounded guidance, then verify that change in a real BTL update.

**Paper-Level Support.** Distinguishes improved certification from a changed reward-guidance decision. This is a necessary intermediate step before claiming any practical optimizer advantage.

**A pre-existing mathematical threshold—not a new learned concept.**

For the population gain \(a\omega_q\phi+(1-a)\omega_s(1-\phi)\), its slope with respect to phi is

\[
\frac{\partial g_a}{\partial\phi}=a\omega_q-(1-a)\omega_s.
\]

The sign changes at

\[
a_{\mathrm{crit}}(x,y)=\frac{\omega_s}{\omega_q+\omega_s}.
\]

At the initial x=.8,y=0, this is about 0.88527. The fixed hidden true \(a_\star=.84280432\) is slightly smaller. With a broad source interval spanning this threshold, the best robust guidance near the current phi uses the moving pointwise balance; with a sufficiently narrow interval whose **upper endpoint is less than** a_crit, the robust population gain is locally decreasing in phi. Thus the bounded controller can change from moving phi upward to moving it downward. This threshold is mathematically determined before labels are drawn; do not pick favorable labels after seeing the outcome.

**Steps:**

1. At the exact saved feedback checkpoint t=.004 from Step 5A, compute x,y, \(\omega_q,\omega_s,a_{\rm crit}\) and the allowed single-step phi interval \([\phi-\rho h,\phi+\rho h]\). The source robust controller's chosen movement provides the fixed comparison.
2. Reuse the same source/informative comparison definitions and exact fixed-N BTL interval calculation from G2.
3. Predict analytically for every observed confidence interval whether its upper endpoint is below a_crit (or whether the one-step robust argmax changes for another precisely calculable reason). Record the predicted direction of the robust controller BEFORE running it.
4. Begin with N=128, 512, 2048, 4096 informative comparisons and five fixed label seeds. These labels are cheap scalar Bernoulli samples. If 4096 does not produce an interval crossing the threshold in the fixed seeds, allow one predeclared escalation to N=8192; record the observed upper endpoint. Do not change a_star or cherry-pick a favorable seed.
5. For each label set, compare source-only bounded robust, learned-interval bounded robust, and bounded point-estimate deployment, all from the EXACT same theta and phi at the feedback checkpoint. Always enforce \(|\Delta\phi|\le.00002\) per policy step.
6. First run **population-exact** policy updates for each controller from t=.004 to .012. This provides a high-precision diagnosis of any controller-action difference without sampled policy-gradient noise. The true preference remains evaluator-only.
7. Record whether the controller's direction changed, actual phi_end, the change in the population certified gain under U_N, original source certified gain under U_0, and the evaluator true gain rate. Small effects are acceptable: the original adaptation budget permits only 0.00002 movement per step, so a large return increase is not an appropriate expectation.
8. Only after a predicted and observed population controller difference is confirmed, run **paired sampled policy updates** for the two essential controllers (one fixed promising N and 3 policy seeds initially, n=8192). Report noise/uncertainty; do not expect a tiny true return difference to be statistically resolved by a handful of trajectories.
9. If reward information changes the direction but not the measurable true gain beyond noise, that is still a precise decision result. Do NOT claim an improvement in expected or realized return without its actual evidence.
10. Add one simple negative control: repeating original source pairs at the same N leaves U_0 unchanged and cannot trigger an information-driven decision change.

**Deliverable G3:**

| BTL labels N | Label seeds | Upper interval endpoint versus a_crit | Source robust direction | Learned robust direction | Phi_end difference | Updated population certificate | True population finite-step gain difference | Sampled agreement if run | Notes |
|---:|---:|---|---|---|---|---|---|---|---|
| 0 source | none | 2, above 0.885663 | up | up | 0 | 0.059390 | 0 | not run | Checkpoint x=0.800436, y=0.000629, phi=0.842944. Later rows match this first-step prediction |
| 128 informative | 5 | 0.95848–1.26951, 0/5 below | up | up, 16/16 steps | 0 | 0.104089 | 0 | not run | Point estimate moves down on 3/5 seeds and up on 2/5. Robust action is unchanged |
| 512 informative | 5 | 0.92060–0.97650, 0/5 below | up | up, 16/16 steps | 0 | 0.110661 | 0 | not run | Same point-estimate split, 3 down and 2 up |
| 2048 informative | 5 | 0.885845–0.933326, 0/5 below at the checkpoint | up | up on the first step | mean −9.6e-5 | 0.112956 | +1.92e-6 | not run | Seed 103, upper 0.885845, turns down after 4 upward steps. The other four stay up |
| 4096 informative | 5 | 0.85627–0.88500, 5/5 below | up | down, 16/16 steps | −0.000640 | 0.114336 | +1.673e-5 | 3/3 policy seeds match the direction and the phi gap; realized true-gain gaps are 1.66e-5–1.68e-5 | Label seed 100, n=8192. Decision change only. The return gap is the slope over a 0.00064 guidance move |
| 8192 if justified | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | 4096 already crossed the threshold on all five predeclared seeds |
| Same-N source comparisons | 5, at N=4096 | stays 2 | up | up | 0 | 0.059390 | 0 | not run | Source pairs have logit 2 for every preference, so the set stays [−1, 2] |

**Possible outcomes:**

- The confidence interval moves entirely below the predicted critical preference, and robust guidance moves phi in the predicted direction: information now has a **mechanistic decision consequence**, beyond merely narrowing the certificate.
- The interval crosses the critical value and guidance does not change: this is expected for an information-insufficient batch, not failed optimization.
- A decision change has negligible true-return effect: quantify the small permissible phi movement and true-gain slope; do not scale a new benchmark to manufacture a larger effect.
- Predicted sign and executed optimal phi disagree: an implementation or objective/normalization mismatch. The exact one-dimensional maximization can be verified without a neural network.

### Experiment G4 — One controlled extension to a two-step finite MDP

**Objective.** Move beyond a one-step bandit without losing exact ground truth. Add one finite transition and a delayed reward to the same four-outcome construction. This is the first NEW environment in this plan; no continuous control or LLM is needed yet.

**Paper-Level Support.** A modest, scientifically interpretable external-validity experiment. It tests whether the same distinction can be evaluated in a short sequential decision problem. It does NOT automatically inherit Theorem 2's exact contact time or universal all-controls result once new dynamics are added.

**Environment design: retain the old model when the new term is zero.**

Initial independent Bernoulli actions are the SAME q,s from Theorem 2, generated by pi_theta. Then a binary state z is sampled:

\[
P(z=1\mid q,s)=\frac{1+\lambda q s}{2}.
\]

A second action u in {-1,1} is selected from a small conditional policy

\[
P_\zeta(u\mid z)=\frac{1+u\tanh(\zeta z)}{2}.
\]

Total preference reward and guidance reward:

\[
r_a(q,s,z,u)=a q+(1-a)s+\kappa z u,\qquad
R_\phi(q,s,z,u)=\phi q+(1-\phi)s+\kappa z u.
\]

Choose **lambda=0.2 and kappa=0.05** as the initial nonzero trial, not a large hyperparameter sweep. For the **zero-kappa reference** the second-stage reward vanishes and the original two-coordinate theta gradient is recovered; use this as the exact bridge. At nonzero kappa, the policy is genuinely sequential and the delayed reward affects policy learning. The extra policy parameter zeta starts at 0. It may be optimized only in G4, not retroactively inserted into G1–G3.

The true preference coefficient a_star stays the same as in G1–G3. The source pairs (q=s) are still independent of a; if a comparison contains a different z,u outcome, its known delayed-reward difference is included as a **known Bradley–Terry logit offset**. Informative comparisons distinguish a through q ≠ s. Avoid any teacher-label information leakage.

**Steps:**

1. Build experiments/bridge/10_two_step_mdp.py without importing a large RL benchmark. Enumerate all **16** outcomes (q,s,z,u). Verify probabilities sum to 1 and nonnegative, and analytic expected returns equal direct enumeration.
2. **Zero-kappa regression:** use the original source policy x=.8,y=0 and initial phi0. Confirm the four-output true reward, exact theta gradient, source certificate and pointwise value match Step 5A. The second policy's reward-gradient contribution should vanish when kappa=0. This isolates compatibility with the old result.
3. Turn on the delayed reward with the one predeclared lambda/kappa pair. Keep the same initial policy and true preference. Compute exact gradients with score vectors for **both** theta and zeta, and verify them against finite differences of exact expected guidance return (in addition to enumeration).
4. Apply the same learning steps h=.0005 and a finite bounded phi budget. Initially use 24 updates with exact gradients. Record source-only gain and candidate pointwise value. Do not assume the loss must still be exactly at 0.008: the new delayed reward changes the dynamics.
5. If a candidate pointwise-versus-trajectory separation is not present, calculate whether (a) the source robust certificate is nonempty, (b) the controller can track the feasible set, and (c) pointwise capability itself fell. These exact calculations make the negative outcome interpretable. Do not blindly retune kappa or the true preference to force loss.
6. Replace only exact policy gradients with sampled complete **two-step trajectories**, using 256 then 2048 trajectories/update and 3 fixed seed runs. Since the space has 16 outcomes, exact evaluation remains available. Recompute G1's population-step versus ex-post separation.
7. Add one batch of 128 genuine informative BTL labels from complete trajectories at t=.004. The label model must include the known delayed reward offset when needed. First evaluate information on an unchanged trajectory; only then allow the bounded guidance controller to react.
8. Explicitly save the exact seed-level checkpoint, initial and final theta/zeta, phi, source/updated intervals, true cumulative return, true finite-step gain rate and pointwise feasibility. The first failed conclusion must be attributable to the new delayed reward/transition or stochastic sampling, not an unverified SAC or PPO implementation.
9. If the baseline zero-kappa regression, exact two-step gradients and the population-versus-sampled comparison all succeed, document that this creates a useful next starting point for **a separate future small MDP / Deep-Sea-Treasure plan**. Do not start Walker, Qwen or Meta-World merely because G4 completed.

**Small G4 result table:**

| Environment variant | Gradient source | Trajectories/update | First source-certificate loss or censored | Pointwise > gamma at loss? | Updated certificate with BTL labels | True finite-step gain near loss | Exact-versus-sampled error | Notes |
|---|---|---:|---|---|---|---|---|---|
| kappa=0 (exact four-output regression) | exact | n/a | 0.008 | yes, 0.113945 | not applied | 0.116030 | path gap versus Step 5A below 1e-18 | Zeta stays 0 and its gradient is 0. Initial source certificate 0.061233 |
| lambda=.2, kappa=.05 (two-step) | exact | n/a | censored through 0.012; terminal source certificate 0.060015 | yes, minimum pointwise 0.114229 | not applied | no loss; minimum true gain 0.116339 | finite-difference residual below 1e-8 | Initial source certificate 0.061858. The delayed term lifts the certificate by about 6.2e-4, enough to cover the old 6.1e-4 deficit. Pointwise never falls below 0.06, and the source robust controller matches the schedule |
| Same two-step process | sampled | 256 | 0, 0, 0.0005 | yes, 0.115356 at the initial state | not applied | at t=0.008: 0.1119, 0.1008, 0.1248 | mean gradient error 0.0607 | Seeds 0–2. Population-step loss is censored on all three |
| Same two-step process | sampled | 2048 | 0.0005, 0, 0 | yes, 0.1153 | not applied | at t=0.008: 0.1192, 0.1225, 0.1174 | mean gradient error 0.0207 | Population-step loss is censored on all three. Realized loss is sampling noise against a margin of about 0.0019 |
| Same two-step, unchanged path | informative BTL | 128 labels | censored, same exact path | yes | terminal updated 0.09557–0.11071; suffix certified 5/5 | unchanged, gap 0 | 0, path is exact | Five label seeds. Information raises the certificate and does not move the policy |
| Same two-step, bounded guidance | sampled | 2048 and 128 labels | realized source loss 0.0005, 0, 0 | yes | terminal updated 0.10437, 0.10295, 0.11089 | same trajectories as the source controller | mean gradient error about 0.021 | Label seed 100. Phi gap versus the source robust controller is 0 on 3/3 seeds, because this interval still contains the crossing. Post-feedback updated certificates stay above 0.102 |

**Possible outcomes:**

- The zero-kappa model matches exactly and the nonzero delayed reward retains a pointwise/trajectory gap: a controlled extension beyond the original system.
- Zero-kappa matches, but the nonzero delayed reward removes the separation: determine whether feasible intervals move more slowly or pointwise progress becomes impossible. This clearly identifies the new environmental effect.
- Exact two-step works, sampled two-step differs: stochastic policy gradient is the new cause, measurable against exact enumeration.
- The delayed reward comparator cannot be learned by the chosen policy: verify the zeta score gradient in the 16-outcome enumerator; do not immediately adopt another algorithm.


### Experiment G4.1 — Extend the same two-step MDP to its first observed support loss

**Status: COMPLETE.** This is the immediate, minimal continuation of G4. All prior G1–G4 results, files, and completed tables remain unchanged.

**Objective.** G4's delayed reward kept the exact original certificate barely above 0.06 at its last executed starting time t=0.0115. Add one exact policy step to establish the first source-only loss, then extend the same system to t=0.016 and independently measure (i) the source certificate, (ii) pointwise feasibility, (iii) true finite-step gain, and (iv) the effect of existing BTL labels on the SAME trajectory.

**Scientific claim that this experiment can support.** A controlled addition of a second-stage transition and delayed reward shifts the observed loss time of the CURRENT reward guidance trajectory, while pointwise capability remains. The original Theorem 2 all-admissible-trajectories result cannot automatically be transferred to this modified MDP. Do not claim universal dynamic impossibility here without a separate proof or reachability computation.

**Keep these facts fixed from the existing repository.**

| Factor | Value, unchanged |
|---|---|
| First-stage policy | Independent Bernoulli q,s in {-1,1}, initial x=0.8 and y=0 |
| Second-stage dynamics | P(z=1 given q,s)=(1+0.2 q s)/2, P(u given z)=(1+u tanh(zeta*z))/2 |
| Second-stage policy parameter | Initial zeta=0, updated with the original ordinary policy gradient |
| Reward family | r_a=a q+(1-a)s+kappa z u; guidance R_phi=phi q+(1-phi)s+kappa z u |
| Preference uncertainty | Source U0=[-1,2]; evaluator-only true a_star=original phi0 |
| Delayed reward | kappa=0.05, with kappa=0 as exact four-output control |
| Guidance | phi0=0.8428043202033038, phi_k=phi0+0.04*k*h |
| Time step and requirement | h=0.0005; gamma=0.06; no retuning |
| New feedback | Original five label seeds 100–104, N=128 fixed-N informative BTL pairs, arrival at t=0.004 |
| Observation window | Original 24 steps to time 0.012; new 25th exact update first; finally 32 steps to time 0.016 |

**New step count is the ONLY change in the primary exact experiment.** Do not change kappa, lambda, policy initialization, reward, true preference, gamma, optimizer, or feedback arrival.

#### G4.1A — Extend the exact trajectory and locate the crossing

1. Reuse the existing 16-outcome enumerator, exact score-gradient and finite-step return calculations in experiments/bridge/10_two_step_mdp.py. Its current rollout uses global M.N_STEPS=24. Implement a small separate driver **experiments/bridge/11_two_step_horizon.py** (name proposed; create it as part of execution) with a local n_steps argument. Do not change the global in historical modules, overwrite step09_g4 files, or replay unrelated benchmarks.
2. Run the original exact models for kappa=0 and kappa=0.05, each for 32 policy updates, from exactly the saved initial policy state. Require numerical agreement between the first 24 executed steps and the committed G4 exact CSV and summary on x,y,zeta,phi, source gain, and true finite-step gain. If this check fails, the new driver is wrong or has changed the experiment; repair it before proceeding.
3. At each start time t_k=k*h compute the original robust finite-step gain using the **exact** population gradient:

   \[
   C_{k,h}(U_0)=\min_{a\in\{-1,2\}}
   \frac{J_a(\theta_k+h\,\nabla_\theta J_{R_{\phi_k}}(\theta_k))-J_a(\theta_k)}{h}.
   \]

   Evaluate the two endpoints, without fitting a grid over true preference.
4. Evaluate **finite-step pointwise feasibility**, independently of the scheduled phi:

   \[
   V_{k,h}=\max_{\phi\in[0,1]}\min_{a\in\{-1,2\}}
   \frac{J_a(\theta_k+h\,\nabla_\theta J_{R_\phi}(\theta_k))-J_a(\theta_k)}{h}.
   \]

   Use deterministic one-dimensional optimization of exact 16-outcome expected returns. Search endpoints plus interior candidates and refine; report tolerance and cross-check against the existing instantaneous pointwise value. Do not confuse the instantaneous V with finite-step V_{k,h}, and do not maximize noisy Monte Carlo candidates.
5. Find the first **strict** C_{k,h}<gamma and record: its starting time and step number, previous certified time, source certificate, finite-step pointwise value, instantaneous pointwise value, true finite-step gain, phi, x,y,zeta, and the minimum outcome probability. Record true gain at the actual evaluator-only a_star, never J_a itself as a gain rate.
6. The primary result is the first 25th update (k=24, starts t=0.012). Complete 32 updates only to show what follows after crossing; if no crossing occurs, its censoring is explicit, not a reason to modify model parameters.
7. A half-step reference at h=0.00025 with the same total time is optional only if exact discretization precision makes the crossing location ambiguous.

**Pre-run analytical/numerical reference, NOT a completed G4.1 result.** Using the CURRENT source equations, an independent calculation predicts:

| Checkpoint/start time | Expected source certificate |
|---|---:|
| t=0.0115, already observed in G4 | 0.0600147411 |
| t=0.0120, new 25th update | 0.0599353993 |

At t=0.0120 it predicts true finite-step gain approximately **0.11629425** and numerical finite-step pointwise value approximately **0.11416757**, both exceeding gamma=0.06. These numbers are reference expectations only. The executing agent must recompute them from the code and enter the observed results below.

#### G4.1B — Same extended trajectory, informative preference feedback

1. Reuse the five existing 128-label BTL batches/intervals from G4 (seeds 100–104) without selecting a new favorable sample.
2. Use matched second-stage outcomes (z,u) in informative trajectory comparison pairs so their Bradley–Terry logit remains 4a-2. If any second-stage outcomes differ, the known kappa*(z1*u1-z2*u2) term must be included as a fixed logit offset; otherwise the preference data do not match the declared model.
3. Compute a source-only certificate along the new 32-step exact path. Then recompute the updated certificate on **the identical saved path** using U0 prior to t=0.004 and the corresponding valid BTL confidence interval afterward. Neither theta, zeta, phi, nor any true gain is allowed to change in this information-only experiment.
4. Keep the uninformative source-pair control; its comparisons depend on no unknown a and must leave U0 unchanged.
5. Report original first source loss; first updated-certificate loss/censored; final executed-step updated certificate (step starting t=0.0155); fraction of the 5 fixed label seeds that certify the entire trajectory, and fraction certifying the post-feedback suffix. If new information arrives after an earlier failure in some sampled continuation, do not retroactively call the whole history certified.
6. If a fixed N=128 batch does not certify the longer window, report its interval and exact remaining worst-case gain. Use N=512 only to answer a specific unresolved statistical question, not simply to guarantee a positive-looking result.

#### G4.1C — Minimal stochastic check of the extension

1. After exact G4.1A succeeds, run **three predeclared policy seeds** with 2,048 sampled complete two-step trajectories per policy update, now continuing through 32 updates at the same h. Reuse the original G4 stochastic gradient estimator and seed protocol.
2. Evaluate each visited sampled-policy state in TWO ways: (a) gain of its actually executed random policy step; (b) gain if the exact population gradient had been used at that same state. Keep their first-loss times separate.
3. Apply the **same** BTL information to each fixed sampled path, measuring the updated certificate without changing its random actions. Optionally run the bounded guidance controller at one fixed label batch, matched to a source-only branch with common random numbers. Only do this if it answers an additional control-choice question rather than repeating G3.
4. Do not repeat the old millions-of-samples escalation. If realized losses appear earlier while population-step counterfactual losses remain near the exact G4.1 time, the G1 sampling-noise explanation still applies in the sequential model.

**Results to fill after execution — exact horizon and pointwise value.**

| Model | Number of exact updates | First source certificate loss | Last certified start time | Source gain at loss | Finite-step pointwise value at loss | True finite-step gain at loss | Last-step source gain | Notes |
|---|---:|---|---|---|---|---|---|---|
| kappa=0 existing reference | 24 | 0.008 | 0.0075 | 0.059947 | 0.113932 | 0.116030 | 0.059390 | Prefix matches the committed G4 path with gap 0. Zeta remains 0 |
| kappa=0.05 existing reference | 24 | censored | 0.0115 | no loss | no loss | no loss | 0.0600147411 | Prefix gap 0. The last executed step is still certified |
| kappa=0.05, first added update | 25 | 0.012 | 0.0115 | 0.0599353993 | 0.11416756 | 0.11629425 | 0.0599353993 | This is step k=24. The deficit below 0.06 is 6.46e-5, so no half-step was run. Minimum outcome probability 0.009916. Instantaneous pointwise value 0.114180 |
| kappa=0.05, complete continuation | 32 | 0.012 | 0.0115 | 0.0599353993 | 0.11416756 | 0.11629425 | 0.059382 | Extending the window does not move the first loss |
| kappa=0, extended reference | 32 | 0.008 | 0.0075 | 0.059947 | 0.113932 | 0.116030 | 0.058758 | The four-output loss time stays at 0.008 |

**Results to fill after execution — feedback and sampling.**

| Experiment | BTL labels | Policy seeds / label seeds | Original source first loss | Updated first loss or censored | Updated last-step certificate | Entire path certified? | True finite-step gain changed? | Notes |
|---|---:|---|---|---|---|---|---|---|
| Extended exact path, source only | 0 | 1 / 0 | 0.012 | 0.012 | 0.059382 | no | no | The source loss is after the feedback time, so the post-feedback suffix is also uncertified |
| Extended exact path, uninformative source comparisons | 128 | 1 / 5 | 0.012 | 0.012 | 0.059382 | no, 0/5 | no | Every source-pair batch leaves the interval at [−1, 2] |
| Extended exact path, informative BTL | 128 | 1 / 5 | 0.012 | censored on 5/5 | 0.095110–0.110322 | yes, 5/5 | no | Labels do not move theta, zeta, phi, or true gain. N=512 was not required |
| Extended sampled path, source only | 0 | 3 / 0 | 0.0005, 0, 0 | 0.0005, 0, 0 | 0.05266, 0.08408, 0.06803 | no | n/a | Population-step loss is 0.012 on 3/3. Mean gradient error 0.0213. True gain at t=0.012 is 0.1187, 0.1113, 0.1120 |
| Extended sampled path, informative BTL without behavior changes | 128 | 3 / 5 | 0.0005 or 0 | same pre-feedback times | 0.09337–0.11206 | entire path 0/15; post-feedback suffix 15/15 | no | The recorded loss is the realized step before t=0.004. Labels do not undo it and do not change the actions |

**Possible result scenarios and their scientific conclusions.**

- **First source loss at t=0.012 while finite-step pointwise value stays above 0.06:** the sequential modification delayed the loss for the current guidance trajectory without removing pointwise feasibility. Do not claim all admissible reward trajectories fail.
- **No failure by t=0.016:** quantify how much the second-stage term lifted the certificate and changed its slope; report the censored horizon. This is still a well-defined environmental effect.
- **Source certificate fails while true finite-step gain remains positive:** exactly separates loss of a guarantee from loss of actual improvement.
- **Feedback preserves the updated certificate on the unchanged path:** information improves the guarantee, not policy performance. State this explicitly.
- **Sampled realized steps fail early while population-step evaluation agrees with the 32-step reference:** the stochastic effect previously isolated by G1 remains active in the new two-step setting.
- **Pointwise feasibility also fails:** the new sequential model no longer exhibits the same pointwise/current-trajectory gap at that checkpoint; record both values rather than retuning kappa.

**Execution and resources.** This is an exact 16-outcome model; CPU is sufficient. Prioritize the first 25th exact update, then 32-step completion, then five existing BTL intervals, finally the three-seed 2,048-sample comparison. The additional computation should be small compared with the earlier 5090 experiments. Record actual runtime and peak CPU RAM; investigate vectorization only if the run is unexpectedly slow.

**Expected new deliverables (not yet created).** Add experiments/bridge/11_two_step_horizon.py and outputs/horizon_bridge/step10_g4_extension/ containing at least config/metadata, exact and sampled step-level CSV, summary.json, a simple source-versus-updated certificate plot if useful, and a source revision. Do not overwrite outputs/horizon_bridge/step09_g4 or its filled Work Plan 03 table.

**Primary completion instruction.** Fill BOTH G4.1 tables in this document from real runs and return this completed same Markdown work plan. A supplementary report is optional and never substitutes for the filled tables.

## 4. Required Deliverables

### 4.1 The completed plan is the primary result

The executing agent must return THIS SAME FILE with G1–G4 tables fully populated, Section 5 resources filled, and Section 7 paths filled. New results must be obtained by actually executing the code. An experiment not run is marked NOT RUN or BLOCKED, never assigned invented values. A deviation belongs in the corresponding row's Notes cell.

The executor does not need to write a separate scientific essay. At the end of the run, it should be sufficient to read the filled tables to understand which layers worked, which comparisons were causal, and which next environment is justified.

### 4.2 Completion table

| Experiment | Status (NOT RUN / PILOT / COMPLETE / BLOCKED) | Seeds / evaluations | Core table filled? | Exact reference implemented? | Notes |
|---|---|---|---|---|---|
| G1 | COMPLETE | 5 stored seeds at 4 sample sizes, plus 10 new seeds at n=2048 | yes | yes | Population loss stays at 0.0075–0.008. Realized loss is earlier and tracks the analytic standard error |
| G2 | COMPLETE | 5 policy × 5 label seeds, plus the exact-prefix hybrid | yes | yes | N=512 was not needed. Whole-path certification and post-feedback certification are reported separately |
| G3 | COMPLETE | 5 label seeds at N=128, 512, 2048, 4096; sampled check is 3 policy seeds at N=4096 | yes | yes | N=8192 was not needed. The robust direction changes only after the upper endpoint falls below a_crit |
| G4 | COMPLETE | exact regression, 3 sampled seeds at two sizes, 5 label seeds, 3 guided seeds | yes | yes | Zero kappa reproduces Step 5A. Nonzero kappa does not lose the exact source certificate inside 24 steps |
| G4.1 | COMPLETE | 32 exact steps, 5 label seeds, 3 sampled seeds at n=2048 | yes | yes | The first strict loss is the 25th update. The half-step and N=512 were not required |

### 4.3 Candidate paper findings

Do not presume they will all be supported. Check which actually survive the experiments.

| Candidate finding | Evidence required | YES / PARTIAL / NO | Result file / figure | Notes |
|---|---|---|---|---|
| Population policy horizon remains near the deterministic theoretical reference along sampled states | G1 population counterfactual at actual sampled theta/phi | YES | outputs/horizon_bridge/step06_g1/ | Population-step loss is 0.008 on 18 of 20 stored sampled paths and 0.0075 on the other two, both at n=256. Instantaneous loss is 0.008 on all of them. Ten new seeds agree |
| Finite stochastic updates may lose ex-post guarantees earlier for a quantified statistical reason | G1 analytic variance and actual realized updates | YES | outputs/horizon_bridge/step06_g1/population_vs_realized.png | One-sample SD 0.855038. The first-step certificate gap is the same order as 0.8550/sqrt(n), and smaller than that Gaussian endpoint SE. True finite-step gain at t=0.008 stays positive |
| New BTL information raises certification of the same sampled-policy trajectory | G2 unchanged trajectory, source versus informative comparison | YES | outputs/horizon_bridge/step07_g2/ | Informative N=128 certifies the post-feedback suffix on 25/25 pairings and changes no true gain. Repeated source pairs do nothing. The whole sampled-from-start path stays uncertified because the realized loss happened before the labels arrived |
| Enough preference information changes bounded guidance decisions | G3 threshold crossing and executed controller action | YES | outputs/horizon_bridge/step08_g3/ | At the checkpoint, a_crit=0.885663. N=4096 puts all five upper endpoints below it and the robust controller reverses for all 16 remaining steps. The population true-gain gap is +1.673e-5. That is not a practical return claim |
| Mechanism remains interpretable in a sequential finite environment | G4 exact and sampled two-step extension | YES | outputs/horizon_bridge/step09_g4/ | Zero kappa matches Step 5A. Kappa=0.05 lifts the exact source certificate by about 6.2e-4, so it no longer expires by t=0.012, while pointwise value stays near 0.114. Sampled steps still fail ex post. N=128 raises the certificate and does not change phi |
| Delayed reward shifts the observed first loss while finite-step pointwise feasibility remains | G4.1 exact 32-step comparison, pointwise optimization and frozen BTL labels | YES | outputs/horizon_bridge/step10_g4_extension/ | Kappa=0 still loses at 0.008. Kappa=0.05 first loses at 0.012, where finite-step pointwise value is 0.114168 and true finite-step gain is 0.116294. This is the scheduled trajectory, not an all-controls theorem. Informative labels certify that frozen path on 5/5 seeds without changing true gain |

The paper should not claim a practical return improvement simply because a confidence set narrowed or guidance action changed.

## 5. Runtime and Resource Assessment

The reported Step 5A–5C runs completed on CPU and used a tiny four-outcome sampler; the recorded earlier sample count of 2097152/update was **disproportionately large relative to the research question**. G1–G4 should be achievable on the existing CPU. Do not rent a larger GPU for this plan.

Before any new execution, record an actual 1–5 second smoke test where appropriate, memory usage, estimated full runtime and whether exact enumeration is the computational bottleneck (it should not be). Vectorize the small four-/sixteen-outcome calculations and reuse stored traces.

Suggested initial work:

- G1: mostly re-analysis of existing files; optional 10 new runs with only 2048 samples/update.
- G2: start with 3 × 3 crossed policy/label seeds; 128 labels, no new trajectory required for the first test.
- G3: analytic controller decision and exact two-coordinate policy updates first; 5 label seeds at 4 sample sizes; only one paired sampled follow-up if meaningful.
- G4: exact 16-outcome enumerator, 24-step short runs, 3 seeds and two small sample sizes. Do not scale to thousands of training episodes before validating exact gradients.

**Fill resource/time table:**

| Experiment | Reused inputs or new samples | Smoke runtime | Full runtime estimate | Actual full runtime | Peak CPU RAM / GPU VRAM | Bottleneck / speedup | Notes |
|---|---|---|---|---|---|---|---|
| G1 | Existing Step 5A/5C CSV, analytic 4-outcome variance, 10 new n=2048 paths | same run | seconds | 4.91 s | 1.63 GB RAM, no GPU | Replaying the stored 2,097,152-sample stream once per seed | Enumeration of four outcomes is not the cost |
| G2 | Frozen sampled paths, 128 BTL labels | same run | seconds | 1.20 s | 0.85 GB RAM, no GPU | Same stored-stream replay | The 5×5 label cross itself is negligible |
| G3 | 128–4096 BTL labels, exact controls, one n=8192 sampled check | same run | seconds | 0.36 s | 98 MB RAM, no GPU | None | No escalation to 8192 labels |
| G4 | 16-outcome exact evaluation and small sampled runs | same run | seconds | 3.96 s | 100 MB RAM, no GPU | Pointwise grid is a few seconds and was not optimized further | No GPU and no benchmark environment |
| G4.1 | 32 exact two-step updates, five existing 128-label BTL batches, 3 × 2048 sampled updates | same run | seconds | 0.78 s | 134 MB RAM, no GPU | Vectorized 16-outcome finite-step search | No half-step and no N=512 escalation |

If any one task is expected to exceed one hour, identify the concrete bottleneck and whether simplification or batch vectorization can preserve the scientific comparison. Do not spend hours engineering an optimization for a seconds-long test.

## 6. Terminology Reference

- **Source preference set \(U_0\):** the original interval [-1,2] of true reward preference coefficients a compatible with the fixed source comparisons.
- **Updated preference set \(U_N\):** statistically valid interval after N new informative Bradley–Terry comparisons. It may reduce uncertainty without changing policy behavior.
- **True preference \(a_\star\):** a fixed evaluator-/teacher-only coefficient. Never hand it to the reward guidance controller.
- **Learned preference \(\widehat a\):** estimate from binary comparison labels, not the current deployed guidance coefficient.
- **Guidance coefficient \(\phi\):** controls the reward used in the policy-gradient update; moves within the declared speed budget.
- **Population-step counterfactual \(C^{\mathrm{pop}}\):** finite-step gain if the actual policy's current exact expected gradient is used. It is not the expected value of the nonlinear realized update, and not a pathwise stochastic guarantee.
- **Instantaneous certificate \(C^{\mathrm{inst}}\):** analytic continuous-time progress lower bound from the paper.
- **Realized-step certificate \(C^{\mathrm{real}}\):** worst compatible gain actually induced by one stochastic sampled update. Early failure is possible even when population guidance remains positive.
- **Pointwise feasibility:** an eligible reward is available at the current policy state. It is separate from tracking feasibility.
- **Support horizon:** maximal duration over which a specified guarantee holds; in a sampled experiment state exactly which guarantee and whether it concerns each realized step, a population reference or a post-feedback suffix.
- **Finite-step true gain rate:** actual difference in true expected policy return divided by policy step size h; not absolute true return.
- **Bradley–Terry comparison:** binary preference observation with a stated logistic logit; use the known intercept when the new delayed reward contributes a term independent of a.

## 7. Result Location — leave empty until run

Store new work separately. Suggested implementation files, none of which exist merely because this plan mentions them:

~~~text
experiments/bridge/07_stochastic_gain.py
experiments/bridge/08_sampled_feedback.py
experiments/bridge/09_guidance_decision.py
experiments/bridge/10_two_step_mdp.py

outputs/horizon_bridge/step06_g1/
outputs/horizon_bridge/step07_g2/
outputs/horizon_bridge/step08_g3/
outputs/horizon_bridge/step09_g4/
~~~

Do not overwrite the earlier 06_online_bandit.py or its Step 5A/5B/5C recorded results. If a correction is required in common code, create a named new experiment implementation or document the exact change and source SHA.

| Experiment | Result summary | Raw data / summary path | Code revision | Notes |
|---|---|---|---|---|
| G1 | Population-step loss stays near 0.008; realized loss is earlier and matches the four-outcome score variance in order of magnitude | outputs/horizon_bridge/step06_g1/summary.json | experiments/bridge/07_stochastic_gain.py | Figure: population_vs_realized.png. Historical Step 5A/5C files were not rewritten |
| G2 | Informative labels certify the post-feedback suffix and do not repair a realized loss from before t=0.004 | outputs/horizon_bridge/step07_g2/summary.json | experiments/bridge/08_sampled_feedback.py | Series: step07_g2/series.csv |
| G3 | The robust controller reverses only when the upper endpoint is below a_crit=0.885663, which first happens at N=4096 | outputs/horizon_bridge/step08_g3/summary.json | experiments/bridge/09_guidance_decision.py | Seed-level decisions: step08_g3/decisions.csv |
| G4 | Zero kappa reproduces the four-output loss at 0.008. Kappa=0.05 keeps the exact source certificate just above 0.06 through t=0.012 | outputs/horizon_bridge/step09_g4/summary.json | experiments/bridge/10_two_step_mdp.py | Matched (z,u) comparisons keep logit 4a−2. A mismatched pair adds the known offset 2 kappa |
| G4.1 | Exact source loss moves from 0.008 at kappa=0 to 0.012 at kappa=0.05, while finite-step pointwise value stays at 0.114168 | outputs/horizon_bridge/step10_g4_extension/summary.json | experiments/bridge/11_two_step_horizon.py | Figure: certificates.png. The committed G4 files were not overwritten; the first 24 steps agree with gap 0 |

## 8. Immediate Execution Sequence

1. **Start G1 by reading existing committed Step 5A/5C traces, not generating new model data.** Confirm that the exact-population reference first loses near t=.008 at the actual sampled states and explain the earlier realized loss with exact four-outcome score variance.
2. For G2, add BTL labels to the SAME sampled policy trajectories, measuring whole-path versus post-feedback certification separately. Preserve the true gains and sample random draws.
3. For G3, derive the guidance decision threshold from the existing reward and policy functions. Evaluate the change of optimal bounded phi as preference intervals become informative. Use exact population policy updates before small stochastic confirmation.
4. For G4, extend the same four-output policy into a 16-trajectory two-step finite MDP. Exact zero-kappa must reproduce the old system before delayed reward and a second policy action are introduced.
5. **Immediate next task G4.1A:** extend the SAME two-step MDP by only one exact policy update first, then complete 32 exact steps without changing model parameters. Measure source certificate, finite-step pointwise optimum and true gain at the first loss.
6. **Then G4.1B/C:** apply the existing five fixed-N BTL batches to the 32-step frozen path, followed by at most three seeds of 2048-sample continuation. Fill the G4.1 tables and result paths.
7. Return this work plan fully completed. **Do not begin Walker, Meta-World or LLM runs in this work plan.** The next expansion decision will use the G4.1 results.
