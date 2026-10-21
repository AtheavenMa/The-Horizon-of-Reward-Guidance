# 04 Reachability to Small MDP

**Project:** The Limits of Reward Guidance under Finite Adaptation  
**Target:** ICML 2027  
**Prepared:** 2026-10-19  
**Status:** COMPLETE — H1–H4 were executed on 2026-10-19. The tables below are filled from those runs.  
**Verified starting commit:** 86c7f1170cbac768bfba456576fd7041799e85f5  
**Repository:** https://github.com/AtheavenMa/The-Limits-of-Reward-Guidance-under-Finite-Adaptation

> **Research objective:** Extend the existing exact theorem-to-training bridge one difficulty at a time. First distinguish loss of the current guidance trajectory from loss for ALL allowed guidance trajectories. Then extend the existing two-stage 16-outcome environment into a small three-stage 64-outcome model. Finally reuse the repository's existing MO-Gymnasium Deep-Sea-Treasure code to conduct a real tabular policy-learning and preference-learning study with exact evaluation. No uncontrolled leap to continuous control or LLMs.

> **Executor:** Work independently. Complete the four experiments in the prescribed scientific order, use reasonable implementation substitutions when necessary, and fill THIS exact Markdown file. Do not interrupt the work to request routine hyperparameter or engineering approvals. A negative effect is acceptable if it identifies the newly introduced assumption; an experiment whose failure could have six unrelated causes is not. Keep actual results, including unfavorable ones. Do not invent measurements.

## 1. Paper Overview and Established Evidence

### 1.1 What the paper establishes

The manuscript distinguishes:

1. **Pointwise feasibility.** For each policy state, a reward parameter exists whose policy update meets the required robust progress.
2. **Trajectory feasibility.** An admissible sequence of deployed reward parameters, starting at a specified initialization and moving no faster than its adaptation budget, can keep meeting the requirement as the policy evolves.
3. **Statistical certification.** The preference comparisons available to the learner identify a sufficiently small uncertainty set for the claimed progress guarantee.

Theoretical anchors already implemented in the repository include the exact scalar support horizon, the endogenous five-output example, an ordinary Euclidean policy-gradient local separation, and instance-level Bradley–Terry information bounds. Read the current paper supplied separately by the project owner before translating results into theorem language. The manuscript itself is not in this public repo.

The following quantities are **not interchangeable**:

- Original-information source certificate and certificate after extra BTL labels.
- Best reward at the CURRENT policy state and existence of a continuous/reachable reward path.
- Exact population-gradient policy change and the actual change caused by a sampled gradient.
- Finite-step true gain rate and the absolute true return.
- One optimizer's failure and a theorem about ALL admissible optimizers.

### 1.2 Exact inputs that already exist

| Source | What can be reused | What must not be assumed |
|---|---|---|
| [horizon.py](../horizon.py) | Scalar theory, fixed-N BTL interval and interval reachability helpers | The scalar theorem is not automatically a theorem for a coupled nonlinear MDP |
| [experiments/bridge/06_online_bandit.py](../experiments/bridge/06_online_bandit.py) | Four-output policy, exact and sampled gradients, original phi and source preference set | Ex-post stochastic certificate is not a population certificate |
| [experiments/bridge/10_two_step_mdp.py](../experiments/bridge/10_two_step_mdp.py) | 16-outcome sequential policy, delay/transition model, 3D exact gradient | A failed scheduled trajectory does not show all control paths fail |
| [experiments/bridge/11_two_step_horizon.py](../experiments/bridge/11_two_step_horizon.py) | 32-step exact/sampled extension; direct pointwise value search | Previous first-loss data were measured only on the scheduled guidance path |
| [outputs/horizon_bridge/step10_g4_extension/summary.json](../outputs/horizon_bridge/step10_g4_extension/summary.json) | Frozen G4.1 measured comparison and BTL outcomes | Do not overwrite these records or treat pre-run predictions as new results |
| [experiments/real/small_rl/phase1.py](../experiments/real/small_rl/phase1.py) | MO-Gymnasium environments, rollout and trajectory-feature extraction, BTL teacher, tabular softmax policy, checkpoint format | Old Monte Carlo gain estimates, coarse-grid LR confidence sets and max-of-noisy-candidates are NOT valid evidence |
| [experiments/horizon_real/E1/Deep-Sea-Treasure/frozen/frozen.json](../experiments/horizon_real/E1/Deep-Sea-Treasure/frozen/frozen.json) | Previously fixed preference teacher w_star, beta and feature scaling for that exact task | Prior experiment's certificates and loss times cannot be imported into a new study |
| [work_plan/01_Reward_Guidance_Experiments.md](01_Reward_Guidance_Experiments.md), [02](02_Theory_to_Training_Bridge.md), [03](03_Stochastic_Guidance_Transfer.md) | Research motivation, completed tables, historical failures, result locations | Do not rerun the old all-at-once E1–E8 grid |

### 1.3 Frozen measured facts from G4.1

The *existing* two-stage model has:

\[
x_0=0.8,\quad y_0=0,\quad \zeta_0=0,\quad
\phi_0=0.8428043202033038,\quad
h=0.0005,\quad \rho=0.04,\quad \gamma=0.06.
\]

The reward family is \(r_a(q,s,z,u)=a q+(1-a)s+\kappa zu\), with \(a\in U_0=[-1,2]\). The deployed guidance is \(R_\phi=\phi q+(1-\phi)s+\kappa zu\).

The second-stage environment has \(P(z=1\mid q,s)=(1+0.2qs)/2\), and the second action \(u\) depends on z via the already implemented \(\zeta\)-policy.

**Observed, do not rerun by default:**

| Previously measured metric | Four-output / kappa=0 | Two-stage / kappa=0.05 |
|---|---:|---:|
| Current scheduled guidance trajectory first loses the source certificate | 0.008 | 0.012 |
| Source robust finite-step gain at the loss | 0.05994738 | 0.05993540 |
| Finite-step pointwise value at loss | 0.11393229 | 0.11416756 |
| True finite-step gain at loss | 0.11602959 | 0.11629425 |
| 128 informative BTL labels on the same frozen exact two-stage path | n/a | 5/5 label batches certify it through t=0.016 |
| Source-only / uninformative comparison | n/a | Does not prevent the scheduled path from losing its certificate |

The existing G3 experiment found a preference-information threshold that reverses **which guidance action is chosen**, but the measured true gain improvement was only around 1.67e-5. That is an interesting mechanism, **not** proof of a large performance advantage.

The exact-current-state calculation suggests that at t=0.012 a guidance increment beyond the remaining permitted movement would be needed to restore the current progress requirement. This motivates H1. It does NOT settle whether another earlier reward trajectory, which would produce a different policy state, could have remained certified.

### 1.4 Next four experiments and the controlled progression

| Experiment | One main new factor | Main question | Starting point |
|---|---|---|---|
| **H1 Coupled reachability** | Allow all bounded guidance-control sequences, not just the previous scheduled one | Is the loss optimizer-specific, or can trajectory infeasibility be established for the coupled policy-reward system? | The unchanged 16-outcome, 32-step G4.1 model |
| **H2 Three-stage exact MDP** | Add exactly one Markov transition and one learned binary action | Does the measured separation survive a modest increase in sequential depth? | H1's 16-outcome model and existing dynamics |
| **H3 Deep-Sea-Treasure exact tabular learning** | Replace constructed 16/64-outcome environment with one small real benchmark | Can a real tabular policy improve under exact, independently evaluated reward updates? | Existing E1 MO-Gymnasium code and frozen teacher |
| **H4 Actual preference learning and checkpoint forks** | Train reward from BTL comparisons and deploy it with finite adaptation | What do additional feedback and extra reward computation change, holding the policy checkpoint fixed? | The known-learning H3 policy and exact evaluator |

H1 and H2 are CPU-exact. H3/H4 are small tabular RL, not deep continuous control. Run H1 first. H2's model code can be prepared independently of any ambiguity about an H1 global infeasibility proof. H3 can set up the exact benchmark evaluator while H1/H2 execute.

## 2. Research and Engineering Principles

1. **One conceptual change at a time.** Preserve the original four-output → 16-outcome → 64-outcome progression before using a new external environment.
2. **Every experiment has a known reference.** In H1, the previous scheduled path; in H2, setting the new delayed-reward coefficient to zero; in H3, exact tabular dynamic programming and a true-reward oracle; in H4, the same fully restored policy checkpoint.
3. **Do not use noisy rollouts for quantities available exactly.** Enumerate 16/64 outcomes or use tabular finite-horizon dynamic programming. Sampling is a separately labeled learning condition.
4. **Do not claim a universal negative from failed numerical search.** An exactly evaluated feasible control sequence proves that a feasible sequence exists. Failure of SLSQP, sampled-control search or a grid is only evidence about the searched family. A universal claim needs a valid outer reachable-set bound or another mathematical argument.
5. **No artificial "pass/fail testing game."** The core deliverable is a scientific distinction and numerical results. A failed oracle or a noisy gradient must identify the precise component and a useful remedy; independent tasks continue.
6. **Keep labels honest.** Source-only, synthetic BTL and independently evaluated true reward must be marked as such. No learned reward receives the hidden true preference coefficient.
7. **Fair branch comparisons.** Policy state, optimizer state, replay/batch, time, random draws, and preference budget are fixed except for the specifically named change.
8. **Do not optimize for favorable confirmatory results.** Use development runs to select a numerical step size or learner configuration. Freeze the chosen configuration before seed-level results. Null effects must remain.
9. **Preserve original experiments.** Never overwrite G1–G4.1, historical E1–E8, or results/reference. New source and outputs live in separately numbered paths below.
10. **No meaningless compute.** A few CPU exact traces and a small tabular rollout pilot come before any additional 5090 work. If runtime grows beyond roughly one hour, identify the actual bottleneck and a scientifically equivalent optimization.
11. **Finish independently.** The executor should implement the code, run experiments, update and push the result files, and fill this same work plan without requiring approvals for ordinary adjustments.
12. **The observed outcome matters more than the preselected shape.** Do not rename an absence of certificate loss as an experimental failure if the experiment establishes that the source set was already sufficient or the chosen environment did not develop a moving bottleneck.

## 3. Detailed Experiment Design

### Experiment H1 — Coupled policy-reward reachability in the EXISTING two-stage MDP

**Scientific objective:** Determine how much can be said about *all* allowed guidance paths in the model where G4.1 observed the currently deployed path lose its source certificate at t=0.012.

The coupled dynamics are

\[
\theta_{k+1}=\theta_k+h\,g(\theta_k,\phi_k),\qquad
\phi_0=\phi_{\rm init},\qquad
|\phi_{k+1}-\phi_k|\le\rho h,\qquad
0\le\phi_k\le1.
\]

Here theta comprises both original Bernoulli policy coordinates and zeta; g is the exact 16-outcome expected gradient. The original information set is \(U_0=[-1,2]\). The source certificate for an actual finite population step is

\[
C_k(U_0)=
\min_{a\in U_0}\frac{J_a(\theta_{k+1})-J_a(\theta_k)}{h}.
\]

This value is exactly evaluated at the two endpoints of U0 because J is affine in a. The horizon of a **specific** guidance sequence is distinct from the maximal possible horizon over admissible sequences.

**Known reference:** The previously executed maximal-upward schedule already uses the full positive speed budget at each step. Its first source loss is t=0.012. It is tempting but not justified to assume that schedule is globally optimal, because changing phi earlier also changes theta.

**H1A — Fixed-policy-path feasibility.**

1. Load the exact theta_k from G4.1. At each k, hold theta_k fixed and solve the scalar one-step robust gain condition \(C_k(\theta_k,\phi)\ge\gamma\) with the finite-step 16-outcome model. Compute the full feasible phi interval (it may be empty or disconnected; detect and report if the expected interval structure does not hold). Use a continuous one-dimensional solver, not a coarse candidate reward grid.
2. Run the existing horizon.interval_feasible helper on this time sequence with fixed phi0 and movement budget \(\rho h\). Report the first unreachable step and an explicit feasible phi witness through the last reachable step.
3. **Label this result FIXED-POLICY-PATH only.** Changing the guidance would generally change the actual theta trajectory, so this computation cannot on its own prove coupled-system impossibility.

**H1B — Actual coupled guidance-control search.**

4. Parameterize the actual increments as \(d_k\in[-\rho h,\rho h]\), with phi0 fixed, and simulate the full coupled theta dynamics with each control path. Use a small multi-start constrained solver (for example SLSQP on increment controls and minimum slack variable) to solve

   \[
   \max_{\phi_{0:K}}\,\min_{0\le k<K}(C_k(U_0)-\gamma).
   \]

   Enforce the true difference constraints, [0,1] box and exact initial value.
5. Start with K=24 (through t=0.012), then K=25 and K=32. Warm starts: existing maximal-upward schedule, source-only greedy robust controller, a feasible path from H1A if available, and a few reproducible interior controls. Do not declare infeasibility merely because an optimizer returns no witness.
6. Save the best *independently re-evaluated* full control path and its minimum exact robust gain. If it meets gamma at all time steps, it is a valid numerical **witness of feasibility** for the modeled finite system. If it does not, report the strongest found trajectory and the search gap, not a universal impossibility claim.
7. Repeat once with the already recorded BTL confidence interval from fixed label seed 100 after feedback arrival at step 8. This distinguishes an information-set change from added adaptation capability. An already failed source-only prefix stays recorded separately.
8. Add one rate-control comparison \(\rho=0\) and one larger budget, e.g. \(\rho=0.08\), **without changing the actual task or preference class**. These are controlled capacity comparisons, not unexplained hyperparameter searches.

**H1C — Outer reachable-set bound for a stronger negative conclusion.**

9. Independently propagate conservative boxes for the reachable theta and phi through the 32 exact finite steps. The phi box at time k is bounded by \(\phi_0\pm k\rho h\), intersected with [0,1]. Obtain theta boxes by interval extension of the existing exact gradient and transition map. Tighten with small box splitting or analytic Jacobian/Lipschitz bounds only if the first bound is too loose.
10. To exclude ALL controls at a given checkpoint, compute a **valid upper bound**

   \[
   B_k\ \ge\ \sup_{\mathrm{admissible\ controls\ through}\ k} C_k(U_0).
   \]

   A computational method that enumerates a finite sample of controls yields a **lower**, not an upper, bound on this supremum and does NOT establish impossibility. Valid upper bounds may use verified interval enclosures with outward rounding, or mathematically justified Lipschitz remainder bounds.
11. If \(B_k<\gamma\), report a certified all-controls failure at k **within the explicitly defined discrete finite model**. If bounds overlap gamma, report INCONCLUSIVE with their numeric width. Still report the best observed guidance sequence and its first loss. Do not spend unbounded time forcing the interval method to succeed.
12. Cross-check each purported bound against 50–100 independently generated admissible controls. This is a consistency check; it is not the proof by itself.

**H1 final table — fill actual numeric cells:**

| Information / control condition | Horizon K or end time | Fixed-path reachability (YES/NO) | Best coupled min certificate minus gamma | Best coupled first loss / censored | Valid all-controls upper bound at candidate loss | Coupled all-controls infeasibility proved? | Notes |
|---|---|---|---|---|---|---|---|
| Original scheduled guidance, U0, rho=.04 | 32 / .016 | NO. First unreachable index 24, t=0.012 | -0.000618171780482013 | 0.012 | 0.05993752641004983 | YES, at t=0.012 | Reference. Feasible phi sets are single intervals. At the loss step the feasible phi starts at 0.8433357484810311 and the reachable phi ends at 0.8432843202033038, a gap of 5.14282777273e-5. |
| Optimized guidance, U0, rho=.04 | 24 / .012 | YES | +1.4741116276584254e-05 | censored | 0.06001668894398318 at the last included step | NO. A witness exists | The best slack equals the maximal-upward schedule. `interval_feasible` returns a prefix witness. |
| Optimized guidance, U0, rho=.04 | 25 / .0125 | NO, index 24 | -6.460070972152954e-05 | 0.012 | 0.05993752641004983 | YES, at t=0.012 | SLSQP and the slower speed families do not beat the schedule. Phi gap versus the schedule is 0. |
| Optimized guidance, U0, rho=.04 | 32 / .016 | NO, index 24 | -0.000618171780482013 | 0.012 | 0.05993752641004983 | YES, at t=0.012 | Best found path is the schedule. Constant-fraction and coast/sprint families lose at 0.009 to 0.012. |
| Optimized guidance, U0, rho=0 | 32 / .016 | NO, index 18, t=0.009 | -0.0013952791718074287 | 0.009 | the admissible set is one path | YES for rho=0 only | The only admissible control keeps phi at phi0. This does not prove the rho=.04 claim. |
| Optimized guidance, U0, rho=.08 | 32 / .016 | YES | +0.0001589339748187535 | censored | not a failure | NO. A witness exists | Doubling the movement budget keeps the maximal schedule certified. Same task and preference class. |
| Post-feedback U_128, rho=.04 | 32 / .016 | YES | +0.0012931109182923728 | censored on the updated set; source prefix still loses at 0.012 | the U0 bound above still applies to the source certificate | NO for the updated certificate | Seed 100 interval [0.6499217036622524, 1.0118023713175432] after step 8. The optimized phi path matches the schedule. |

**Possible outcomes:**

- The source-only optimized trajectory still loses at or before t=.012, and a valid reachable-set upper bound is below gamma: stronger evidence (possibly a numerical proof with declared enclosures) of all-controls failure.
- A different source-only trajectory remains certified through .012 or .016: **the old failure was optimizer/path-specific**. This is a decisive and useful new result; do not suppress it.
- No feasible trajectory found, but no valid upper bound obtained: unknown global status. The fixed-path result and best found trajectory still quantify the observed gap and the unresolved optimization question.
- Extra BTL information allows a new feasible path while source-only information does not: information and adaptation are separated in the coupled setting.

### Experiment H2 — Three-stage, 64-outcome sequential extension

**Scientific objective:** Increase actual decision depth from two to three while preserving an EXACT finite evaluator and the original preference ambiguity.

**Only new environment element:** Add one binary state and one binary policy-controlled terminal action to the already verified two-stage model. The original q,s,z,u variables and first two policy stages remain unchanged.

Let the new binary state be b in {0,1} and the new action v in {-1,1}:

\[
P(b=1\mid z,u)=\tfrac12[1+\lambda_2 u(2z-1)],\qquad
P_\xi(v\mid b)=\tfrac12[1+v\tanh(\xi(2b-1))].
\]

Use predeclared \(\lambda_2=0.2,\ \xi_0=0\). Extend the total reward by exactly one preference-independent delayed term:

\[
r_a(q,s,z,u,b,v)
 = a q+(1-a)s+0.05zu+\kappa_2 b v,
\]

\[
R_\phi(q,s,z,u,b,v)
 = \phi q+(1-\phi)s+0.05zu+\kappa_2 b v.
\]

Use \(\kappa_2=0.025\) for the main extension and \(\kappa_2=0\) as the exact regression. There are \(4\times2\times2\times2\times2=64\) complete trajectories. The policy parameters are the existing three plus \(\xi\). Neither \(a_\star\) nor source U0 changes.

**H2A — Exact zero-extension and nonzero extension.**

1. Build a 64-outcome enumerator using the exact [10_two_step_mdp.py](../experiments/bridge/10_two_step_mdp.py) implementation as the prefix. Do not write unrelated new dynamics. At \(\kappa_2=0\), the added v,b reward term vanishes and the old theta/zeta gradients must be recovered exactly; the xi gradient must be zero.
2. Verify probability normalization, nonnegative probabilities, direct vector expected reward, all four policy score derivatives against finite differences, and the preference BTL logit on matched delayed outcomes.
3. Fix \(\phi_0=.8428043202033038,\rho=.04,h=.0005,\gamma=.06,a_\star=\phi_0,U_0=[-1,2]\). Do not transfer a new arbitrary threshold from another task.
4. Run 32 exact population updates for the \(\kappa_2=0\) regression and 48 updates (to t=.024) for the new \(\kappa_2=.025\) environment. If the new environment is still certified through 48 updates, record censoring and how the extra delayed term changes source gain. Do not tune kappa to force a crossing.
5. At each step report finite-step source certificate, finite-step true gain, finite-step pointwise optimum over phi in [0,1], instantaneous benchmark, xi/zeta, phi and current policy state. Exact policy expectation must not be replaced by sampled rewards.
6. If H1 obtains a feasible or infeasible control witness, check whether applying **that same guidance control sequence** in H2 has comparable behavior. This is a controlled transfer of a policy, not an all-controls theorem for H2.

**H2B — Actual sampled trajectories and preference feedback.**

7. Only after exact results are understood, sample complete three-stage trajectories for three fixed policy seeds, first 512 and then 2048 trajectories per policy update, over 32 updates. All four policy gradients are learned from the same actual sampled trajectory and deployed guidance reward.
8. Evaluate every observed stochastic policy checkpoint with the exact 64-outcome population-step counterfactual and the realized step. This gives an explanation if sample noise causes an early ex-post failure.
9. At t=.004, collect 128 informative BTL comparison labels from matched delayed outcomes where the unknown preference enters exactly through q,s. On the exact frozen trajectory, compare source-only and updated certificates. Source comparison pairs with q=s but otherwise matched outcomes remain uninformative in a.
10. In one separate adaptive arm, apply the updated U_N to the same bounded robust guidance controller used in H1, retaining the same rho and the same state before feedback. Compare controller movement and true return; do not claim improvement if the chosen phi paths are identical.
11. If adding the new term destroys the old separation, directly identify whether the **pointwise optimum**, the moving safe set, or the actual available movement budget is responsible. That is the scientifically useful result.

**H2 table:**

| Three-stage variant | Exact/sampled | Trajectories per update | Seeds | Source first loss / censored | Finite pointwise value at loss | True gain at loss | Updated certificate after 128 BTL labels | New terminal-policy effect | Notes |
|---|---|---:|---:|---|---|---|---|---|---|
| kappa2=0 regression | exact | n/a | 1 | 0.012 | 0.11416756461612998 | 0.11629425477344135 | source labels stay on [-1, 2] | xi gradient and terminal xi are 0 | Certificate gap versus the saved 16-outcome kappa=0.05 path is 4.44e-13. |
| kappa2=.025, 48-step run | exact | n/a | 1 | 0.013 | 0.11422655845994001 | 0.11636056703423492 | see the frozen-path row | terminal xi 0.0003000088090302, terminal zeta 0.0006001816885314 | Not censored. The same maximal schedule transferred from H1 also loses at 0.013. Pointwise value stays near 0.114. |
| kappa2=.025, 32-step run | sampled | 512 | 3 | realized 0.0015, 0.0015, 0.003; population-step 0.0125, 0.013, 0.013 | not a population pointwise | true gain at t=0.012: 0.11831304154830136, 0.10376104398157615, 0.1202136017055544 | not applied | terminal xi stays below 0.0004 | Realized loss is early. The exact step at the sampled state stays near 0.013. |
| kappa2=.025, 32-step run | sampled | 2048 | 3 | realized 0.0005, 0.0005, 0.0015; population-step 0.013, 0.013, 0.0125 | not a population pointwise | true gain at t=0.012: 0.11971720946490905, 0.1268437086945351, 0.117761226164248 | not applied | mean gradient error about 0.030 | Same separation: certificate noise, true gain stays above 0.11. |
| kappa2=.025, frozen path + labels | exact | 128 BTL pairs | 5 label seeds | 0.013 on 5/5 | 0.11422655845994001 | 0.11636056703423492 | terminal updated 0.09435316024131168 to 0.10971053424113641; updated loss censored 5/5; post-feedback suffix certified 5/5 | policy unchanged | Seed 100 interval [0.6499217036622524, 1.0118023713175432]. Source pairs with q=s keep [-1, 2]. |
| kappa2=.025, BTL-guided policy | sampled | 2048 + 128 labels | 3 | realized 0, 0.001, 0.0025 | unchanged because phi does not move | terminal true gain 0.11644321876502772, 0.11868453745167429, 0.1099268879587445 | terminal updated 0.10563308627964396, 0.10540105240530728, 0.10140416898107851; first updated loss equals the pre-feedback realized loss | phi gap versus the schedule is 0 on 3/3 | No improvement claim. The N=128 interval still contains the maximal-speed choice. |

**Result interpretation:** The 64-outcome model is the exact bridge between the already studied 16-outcome sequential task and a new tabular benchmark. It shows what the third decision adds, instead of changing the model class and optimizer simultaneously.

### Experiment H3 — Real small multi-objective MDP with an EXACT tabular evaluator

**Scientific objective:** Replace the custom 16/64-outcome environment with a recognizable, already used small RL environment while preserving exactly computable expected returns and the same explicit reward-learning problem.

**First and ONLY initial external environment:** MO-Gymnasium [deep-sea-treasure-v0](https://mo-gymnasium.farama.org/environments/deep-sea-treasure/). This is a small discrete grid task with two reward components, treasure and time penalty, and four discrete moves. The **existing repository already trained a tabular softmax policy on it** in E1. Do not add a new benchmark family at the same time.

**Keep the previous teacher:** Read experiments/horizon_real/E1/Deep-Sea-Treasure/frozen/frozen.json at execution time. Its development-frozen value was w_star=(0.375,0.625), BTL beta=5.0 and a fixed affine normalization of vector returns. Carry these forward as a chosen teacher setup, but DO NOT carry forward E1's coarse-grid confidence set or its noisy "horizon" measurement.

**H3A — Build and establish the exact model, not a new black box.**

1. Reuse the installed environment and its discrete state/action encoding. Create a finite-horizon tabular transition/reward representation from the **actual version pinned on the machine**. Initial development horizon H=20, matching the old E1 truncation. If a longer horizon H=40 is needed to make the intended treasure outcomes reachable, decide this on development and freeze it; clearly report the chosen horizon. Never compare policies evaluated at different horizons as if on the same task.
2. Enumerate accessible grid states/actions and the terminal/truncation behavior; derive each transition from the installed environment's source or a controlled state-setting simulator mechanism, not an invented map. Include the time index in exact finite-horizon evaluation. Record reward vector [treasure,time_penalty] and all normalization constants. The time step cap is part of the environment definition.
3. Define a time-homogeneous tabular softmax policy with logits for each reachable state and each of four actions, optionally including time in the policy only if the same class is used in all comparison branches. Reuse the old E1 rollout/softmax code wherever compatible.
4. Implement exact finite-horizon dynamic programming to evaluate the **2D expected return vector** mu(pi) of any supplied policy. Verify direct dynamic-programming return against simulated Monte Carlo estimates for three fixed policies: uniform, a shortest-path-to-an-accessible-treasure policy, and one learned/partly trained policy. The Monte Carlo estimate needs to agree within its measured uncertainty; the exact DP result is primary.
5. Implement the finite-horizon tabular policy gradient of J_phi(pi)=phi*mu1+(1-phi)*mu2 and compare a small number of logits with independent centered finite differences. Ensure the gradient is nonzero where learning is possible. Separate true environment vector reward from the preference weight used to scalarize it.
6. Compute the optimal finite-horizon value for the frozen true preference via an exact dynamic-programming oracle. This proves the chosen task has a useful attainable policy and supplies a quantitative optimality gap, **without giving the true preference to the eventual learned-reward agent**.

**H3B — Positive learning control before any preference learner.**

7. From a fixed softmax initialization, train the tabular policy with **exact policy gradients using the true frozen preference** for an oracle-only positive control. Begin with 20 updates and inspect learning; permit 60 updates or a modest learning-rate adjustment only when a developer-visible baseline fails to improve relative to its exact initial value. Record the rationale and freeze the final configuration before the other branches.
8. Independently train the same policy with sampled REINFORCE using the **same oracle preference** for 3 fixed seeds, initially 256 or 512 trajectories per policy update. Compare its value curve with the exact policy-gradient curve. Use exact DP to evaluate both after each update; source preference uncertainty is not introduced yet.
9. Include an exact optimal-policy DP reference. If the direct oracle policy improves but sampled policy learning does not, the new difficulty is policy-gradient variance/coverage. If even the exact oracle policy fails while an optimal gap exists, identify optimizer initialization, parameterization, step size or gradient correctness before blaming BTL.
10. Save one full-state policy checkpoint EARLY in learning and one later checkpoint chosen on **development progress**. These are inputs to H4; do not choose them using H4's confirmatory preference results. Store current policy logits, optimizer state, normalization, environment version, time horizon and random state.

**H3 evaluation quantities:**

- Exact vector return mu(pi), evaluated with full finite-horizon DP.
- Exact true scalar J_{w_star}(pi), measured only by evaluator.
- True finite policy-update gain J_{w_star}(pi_new)-J_{w_star}(pi_old).
- Exact population-gradient counterfactual and realized sampled-update gain separately.
- Optimality gap J_star - J(pi), train/rollout steps and number of policy updates.
- Pointwise robust gain under the currently declared reward-information set only when such a set has actually been defined; do not borrow H1's U0 or gamma.

**H3 table:**

| Learning condition | Horizon, reachable states | Seeds | Policy updates / sampled trajectories | Initial exact true return | Final exact true return | Improvement vs initial | Gap to exact optimal policy | Exact-vs-sampled discrepancy | Notes |
|---|---|---:|---|---|---|---|---|---|---|---|
| Uniform initial policy, exact DP | 20, 72 states | 1 | 0 | 0.49994372476520715 | 0.49994372476520715 | 0 | 0.27907334714145554 | table Monte Carlo and a 2000-episode live-env rollout agree within their standard errors | Installed convex map. Deepest front treasure is 19 steps, so H stays 20. Wrapper cap is 100 and is not the horizon used here. |
| Optimal true-preference DP policy | 20, 72 states | 1 | DP | 0.7790170719066627 | 0.7790170719066627 |  | 0 | none | Time-dependent optimum. One-step descent into the nearest treasure scores 0.6423841047518433, between uniform and this optimum. |
| Oracle true-reward, exact policy gradient | 20, 72 states | 1 | 20 updates, learning rate 1 | 0.49994372476520715 | 0.607583007273581 | 0.1076392825083739 | 0.17143406463308164 | none | Twenty steps improve the return. The last gain is still +0.0016251579394054305 and the final return is still below the one-step treasure. The budget was frozen because the run beat its initial value. |
| Oracle true-reward, sampled REINFORCE | 20, 72 states | 3 | 20 updates, 512 trajectories | 0.49994372476520715 | 0.6067842940964885, 0.6070446172518287, 0.6074354886061718 | 0.10684056933128133, 0.10710089248662152, 0.10749176384096465 | 0.1722327778101742, 0.171972454654834, 0.1715815833004909 | final gap versus exact PG is about 0.0008, 0.0005, 0.0001 | Same learning rate and update count. Sampling is not the obstacle at this size. |
| Frozen early full checkpoint for H4 | 20, 72 states | 1 | update 5 | 0.49994372476520715 | 0.5529640648097348 | 0.05302034004452765 | 0.2260530070969279 | none | Chosen on the development curve, before any H4 label result. Logits only; the teacher weight is not in the file. |
| Frozen later full checkpoint for H4 | 20, 72 states | 1 | update 20 | 0.49994372476520715 | 0.607583007273581 | 0.1076392825083739 | 0.17143406463308164 | none | Best development return inside the frozen 20-update budget. |

**If a learning baseline underperforms:** Use the **already available exact gradient and optimality gap** to determine whether the difficulty is sparse reward, poor exploration, wrong model, poor initialization or insufficient optimization. A reference soft policy iteration can be a separate algorithmic positive control if basic PG cannot improve; do not silently replace the experimental PG policy and call the two algorithms equivalent.

### Experiment H4 — Real BTL reward learning and matched-checkpoint interventions

**Scientific objective:** After H3 demonstrates real policy learning on the same task, train the reward preference parameter from actual comparison labels, and separate new feedback from extra optimization work under a finite guidance-update budget. This is the modest real-environment test the earlier E1–E3 lacked.

**Fixed environment and teacher:** Same Deep-Sea-Treasure version, finite horizon, normalized return features and hidden w_star/beta as H3. The teacher samples independent Bradley–Terry binary labels; it never passes w_star to the learner or the guidance controller.

**H4A — Real preference pool and continuous one-parameter inference.**

1. Construct a fixed development/confirmation split of trajectory pairs. Generate actual environment trajectories under a documented exploratory behavior policy. For each trajectory record the 2D normalized return feature f(tau) using the frozen H3 normalization. Keep pair identities and originating policy checkpoint/trajectory batch. Freeze the unlabeled pair pool before seeing its binary labels.
2. For a comparison pair, write D_i=f(tau_i)-f(tau'_i), scalar parameter w in [0,1] and

   \[
   P(Y_i=1\mid w)=\sigma(c_i+w d_i),
   \quad c_i=\beta D_{i,2},\quad
   d_i=\beta(D_{i,1}-D_{i,2}).
   \]

   Train a real scalar BTL reward head with logistic likelihood (e.g. bounded 1D numerical optimization or gradient updates). Check it against direct likelihood optimization. Label source-only, informative and teacher/oracle data separately.
3. Do NOT reuse phase1.py's discretized simplex confidence_set. Instead use an explicit continuous one-dimensional fixed-N confidence interval. One conservative construction for a **preselected, label-independent** pool is to invert the bounded score

   \[
   S(w)=\sum_i d_i[Y_i-\sigma(c_i+w d_i)]
   \]

   with
   \[
   |S(w)|\le \sqrt{\tfrac12\log(2/\alpha)\sum_i d_i^2},
   \quad \alpha=.05.
   \]

   Conditional on the chosen pairs and independent Bernoulli labels, this follows from Hoeffding's inequality. S(w) is nonincreasing in w, so solve the interval endpoints continuously in [0,1] with bracketing/root finding, NOT a coarse simplex grid. If all d_i are zero, the entire [0,1] remains compatible. Report that this confidence method is conservative and applies to the described fixed-N pair design, not arbitrary adaptive stopping.
4. Verify that the true teacher preference belongs to the confidence set in a controlled repeated-label development check, and record the exact covered fraction, without claiming a 95% formal guarantee merely from a small number of simulation seeds.
5. Select the **source comparison pool** by low preference sensitivity |d_i|, and an **informative pool** by prespecified high scalar Fisher information d_i^2 p_i(1-p_i), using an estimate based only on previous/dev data rather than hidden w_star. If low-|d| comparisons are not truly uninformative, report their measured information; do not claim they have zero sensitivity.
6. Begin with **32 initial source comparison labels**, then additional 16, 32 or 64 informative labels. If 32 source labels already provide a tight enough interval, report that fact rather than forcing an artificial insufficiency condition. Keep uniform random pool selection as an information control.
7. Separate two distinct resources: (i) likelihood optimizer iterations on the SAME labels, and (ii) NEW independently generated preference labels. More fitting steps cannot add statistical information after the likelihood has already converged. This is an expected and meaningful null control.

**H4B — Same-checkpoint policy and reward intervention.**

**Initial guidance choice, common to all checkpoint branches:** Starting from the same saved checkpoint, first infer U0 from the 32 source labels. Then set phi_init to the deterministic maximizer of **exact one-step worst compatible gain over U0** at that checkpoint (with phi in [0,1]). This uses no hidden w_star. Save that phi_init once and restore it for EVERY branch from that checkpoint. If even the best initial phi has certificate below the chosen gamma, say "not certified at initialization"; do not manufacture a later source-horizon loss. Do not simply reuse the oracle's hidden preference as initial guidance.


8. Restore each of the two H3 checkpoints, including the policy logits, optimizer state, environment and RNG. All branches from a given checkpoint begin with identical policy state and declared guidance phi.
9. Guidance is a **separate deployed scalar parameter** phi in [0,1], not the hidden preference w_star and not automatically the BTL point estimate. Use a default bounded change |phi_{k+1}-phi_k| <= b per **policy update**, with b=0.02 and a single sensitivity check b=0.05. This tabular b is a budget in training-iteration units; it is NOT numerically equated to the theoretical continuous-time rho=0.04 without converting time units.
10. For each policy checkpoint compute exact vector return and exact policy gradient. Define a candidate reward choice as maximizing worst-case **exact** one-step improvement over the current continuous confidence interval, subject to the current phi movement budget. In one dimension this can be solved with a deterministic scalar search. Hold fixed the policy update budget per branch.
11. Compare these branches, using the same checkpoint and comparable update steps:

    - **Baseline:** initial labels, current fitted reward/phi.
    - **More reward fitting, same labels:** extra BTL likelihood optimizer iterations, NO additional pairs. Same policy update budget.
    - **Extra informative feedback, matched fitting:** add 32 informative pairs, but the same number of total fitting iterations as the baseline branch.
    - **New feedback + extra fitting:** same extra pairs plus higher optimizer iterations.
    - **Robust guidance with updated confidence set:** use U_N to choose a bounded phi maximizing exact worst compatible policy improvement; use same total policy updates as the main comparison.
    - **Oracle true preference** as an evaluator-only positive control, explicitly marked as not a usable learner.

12. Start with **one exact population-gradient policy step**, evaluated exactly by DP. This isolates the effects of reward parameters without policy sampling. Continue each branch for a modest 20 policy updates, recording all intermediate actual phi and policy states. Only after results are meaningful, repeat actual sampled REINFORCE with matched random draws for 3 seeds.
13. Measure source certificate using the INITIAL source information U_0 at every checkpoint (do not overwrite it after new labels), updated certificate using U_N, true exact gain under w_star, and current pointwise value. If the source initial certificate is already below the chosen positive gamma, label it **uncertified from initialization**, not a positive horizon that later failed.
14. Report two gain thresholds: gamma=0, and a positive gamma fixed from the **development** exact-oracle finite-step gains (suggested 10% of the median positive gain, as in prior work). Never choose gamma from final results; report task units and step sizes. This measured gamma is task-specific and not comparable numerically with the four-output 0.06.
15. The actual pointwise value is the best feasible one-step robust gain at the SAME policy checkpoint with phi free in [0,1]. Calculate it with deterministic 1D optimization, not a maximum over noisy simulations. Do not claim universal trajectory impossibility in Deep-Sea-Treasure unless separately established.
16. If a branch shows no difference, identify the relevant cause using the existing exact quantities: already-tight preference set; reward MLE already converged; phi path unchanged by the new confidence interval; limited phi budget; policy gradient nearly zero; or absence of source-only certification. Do not just label the whole experiment failed.


**H4C — An actual ONLINE preference-learning run from a fresh policy.**

The checkpoint forks above are a mechanism experiment and may start from a policy checkpoint produced by an **oracle true-reward control**. Therefore do not describe those forks as an end-to-end learner trained without oracle initialization. Add the following very small independent experiment:

1. Reset the tabular policy to the SAME development-frozen softmax initialization used by H3. Fix deployed guidance phi_init=0.5 (not the hidden teacher w_star), with per-policy-update movement at most b=0.02.
2. Use 32 previously selected source comparison labels to fit the scalar BTL preference head and U0. Every policy update uses the actual deployed R_phi and the **exact** tabular policy gradient initially. The true w_star is inaccessible to training; evaluate its resulting return separately with exact DP.
3. Run 40 policy updates as a modest fixed trajectory. Predefine the only information acquisitions: 16 fresh informative comparisons immediately before update 10 and another 16 before update 20, reusing the frozen unlabeled-pair pool and query rule. Do not draw additional labels based on final-test results. Keep the source-only branch on exactly the same label budget as its original 32 source labels throughout.
4. Branches from the SAME initial checkpoint: source-only and no new feedback; two fixed scheduled batches of informative feedback; a uniform-query control with the same count of fresh labels; and an oracle true-preference control clearly labeled oracle-only. All preference-driven branches use the identical phi movement budget and the same number of policy updates.
5. Train/refresh the BTL estimate from observed labels only, deploy it with speed-limited phi updates, and save time series of phi, exact true return, gain per update, source and updated confidence sets, preference accuracy and label counts. No oracle policy warm start is permitted in this H4C experiment.
6. If the learned runs do not improve true return, the H3 true-reward exact-gradient run and continuous BTL fit reveal whether the cause is (a) information quality, (b) deployment choice, (c) reward movement capacity, or (d) lack of positive policy gradient. This is a concrete result; do not switch tasks just because the learned agent is not better.
7. If the exact-gradient online result is informative, make the modest additional check with sampled policy gradients, 3 predeclared seeds and approximately 256–512 trajectories per update, evaluated with exact DP. Do not immediately expand to long deep-RL episodes.

**H4C online-learning result table:**

| Full online condition, all start from fresh policy | Initial / new labels | Policy updates | Phi movement bound | Initial exact true return | Final exact true return | Exact true gain vs initialization | Preference-interval final width | Source / updated certificate conclusion | Notes |
|---|---|---:|---:|---|---|---|---|---|---|
| Source-only learned reward | 32 / 0 | 40 | 0.02 | 0.49994372476520715 | 0.60767129069595 | +0.1077275659307429 | 1 | source certificate reaches a minimum of -0.0024546114070957414 and ends at +0.0018933020727303906 | phi starts at 0.5 and ends at 0.466. The source interval never leaves [0, 1]. No oracle policy warm start. |
| Informative feedback at updates 10 and 20 | 32 / 32 | 40 | 0.02 | 0.49994372476520715 | 0.6236138236044841 | +0.12367009883927699 | 0.4888275366978845 | final source certificate -0.0019719281262031846; final updated certificate +0.00043845927047470035 | phi ends at 0.06. True return approaches the oracle. The 16+16 labels were fixed before this run. |
| Uniform new-feedback control at identical dates | 32 / 32 | 40 | 0.02 | 0.49994372476520715 | 0.60767129069595 | +0.1077275659307429 | 1 | identical to the source-only row | phi path matches the source-only path. Uninformative labels do not move the controller. |
| Oracle w_star reward, separate upper control | oracle-only | 40 | n/a | 0.49994372476520715 | 0.6243692767294623 | +0.12442555196425514 | n/a | not a learner certificate | phi is held at 0.375. Evaluator-only upper control. |
| Sampled-policy verification if informative | 32 / 32 | 40 | 0.02 | 0.49994372476520715 | source seeds 0–2: 0.6082157945728542, 0.6094362387528305, 0.6084764525182795. Informative seeds 0–2: 0.6238135160224499, 0.6236214586082176, 0.6209669113488772 | source about +0.108 to +0.110; informative about +0.121 to +0.124 | source interval stays width 1; informative uses the same 64 labels as the exact run | the ordering matches the exact run on 3/3 seeds | 256 trajectories per update. Final phi is about 0.46 for source and 0.06 for informative, the same split as the exact controller. |

**H4D — Small, justified replication.**


17. For matched-checkpoint branches, use the already fixed two checkpoints and 3 independent preference-label seeds at first. Keep H4C's fresh-start online trajectories separate from those oracle-checkpoint branches. Increase to 5 seeds only after the main comparison is numerically stable and the effect size makes a stronger estimate worthwhile.
18. If the first 32 extra pairs produce the expected statistical distinction, do not perform a large automatic label sweep. If the confidence interval still spans the relevant decision boundary, run the predetermined 16/32/64 budget comparison to locate it. Record selection rule and cumulative labels clearly.
19. Resource-Gathering is an **optional second task after this Work Plan**, not part of the default grid. It has 3 reward components, and its confidence set needs a genuinely multidimensional treatment; do not copy the one-dimensional 2-objective interval there.
20. Do not start Walker, Meta-World or LLM PPO. Their old baselines did not learn reliably, and this plan is specifically designed to establish real small-task learning first.

**H4 table — freeze/development details and real preference fit:**

| Task/checkpoint | Source pairs | Extra informative pairs | Teacher beta and hidden w (evaluator only) | BTL fitted w_hat | Valid continuous confidence interval U | Preference accuracy on independent held-out comparisons | Measured width / information gain | Notes |
|---|---:|---:|---|---|---|---|---|---|
| Deep-Sea-Treasure, source pool | 32 | 0 | beta 5, hidden w=0.375 | 0.43148180250320634 | [0, 1] | 0.78125 | width 1; median Fisher of the selected source pairs is 0 | Label seed 100. Eight gradient steps reach 0.4978467411205311 and eighty reach 0.48127183314495264; neither equals the bounded MLE. Forty repeated label draws cover the true w on 40/40 only because the interval is the whole segment. |
| Deep-Sea-Treasure, extra 16 | 32 | 16 | beta 5, hidden w=0.375 | 0.3714702326252165 | [0, 0.5841863803556153] | 0.78125 | width 0.5841863803556153 | Contains 0.375. The fixed-step gradient iteration stalls at 0.6777571299695081, away from this MLE. |
| Deep-Sea-Treasure, extra 32 | 32 | 32 | beta 5, hidden w=0.375 | 0.3363513883086601 | [0.025833379021802164, 0.5146609157196866] | 0.78125 | width 0.4888275366978845 | Contains 0.375. Eight and eighty gradient steps both stop at 0.766343105432656. |
| Deep-Sea-Treasure, extra 64 if needed | 32 | 64 | beta 5, hidden w=0.375 | 0.3433169979600479 | [0.09436794462759628, 0.517050275129972] | 0.78125 | width 0.42268233050237575 | The interval still covers 0.375 and is not a sharp decision. No further label sweep. |
| Uniform-query control | 32 | 32 | beta 5, hidden w=0.375 | 0.7755605202918432 | [0, 1] | 0.78125 | width 1 | The short fit, the long fit and the MLE agree. Uniform pairs do not shrink the interval. |

**H4 table — actual matched-checkpoint interventions:**

| Source checkpoint | Intervention | Added pairs | Reward fitting iterations | Phi movement budget per policy update | Exact true return change over 20 policy updates | Source certificate loss / none at start / censored | Updated certificate loss | Final phi / pointwise value | Sampled gain (if run) | Notes |
|---|---|---:|---:|---|---|---|---|---|---|---|
| Early | Same-label baseline | 0 | 8 | 0.02 | seed 100: +0.0461895073151; seeds 101 and 102: +0.040446 and +0.049739 | positive at initialization (0.00108780437712, above gamma_pos); final source certificate -0.000827356234328 | same as source, because no new labels | phi 0.497846741121; initial pointwise 0.00108780437712 | exact DP only | phi_init=0.70 is the robust maximizer over U0=[0,1], shared by every early branch. |
| Early | More fitting, same labels | 0 | 80 | 0.02 | seed 100: +0.0472754887182; seed 101: -0.022396; seed 102: +0.052569 | final source -0.000957448074424 on seed 100 | same as source | phi 0.481271833145 on seed 100; seed 101 drives phi to 1 | exact DP only | Eighty steps still miss the MLE. One of three seeds loses return. |
| Early | +32 informative labels | 32 | 8 | 0.02 | seed 100: +0.00121222902527; the other two are +0.000028 and +0.011119 | final source -0.00512059992067 | final updated -0.00470766030376 | phi 0.766343105433 | exact DP only | This deploys the stalled gradient fit, not the MLE 0.3363513883086601. |
| Early | +32 labels and more fitting | 32 | 80 | 0.02 | identical to the 8-step row on 3/3 seeds | identical | identical | identical | exact DP only | Extra gradient steps do not move this fit. |
| Early | Updated-set bounded robust guidance | 32 | interval, not a point fit | 0.02 | +0.0525694962716 on 3/3 | final source -0.0021219348725 | final updated +0.00126520530123, still above gamma_pos | phi 0.30; initial pointwise 0.00108780437712 | exact DP only | The interval controller, not the stalled point estimate, is what moves the return. |
| Early | Oracle preference (positive control) | 0 | n/a | n/a | +0.0610580373016 | final source -0.000817863497053 | same number; the oracle does not use a learned set | phi 0.375 | exact DP only | Evaluator-only. Not a usable learner. |
| Later | Same-label baseline | 0 | 8 | 0.02 | seed 100: +0.0120691968139; seeds 101 and 102: +0.009975 and +0.013477 | not certified at initialization for gamma_pos: best initial gain is 8.7994465146e-05, which is above 0 and below 0.0004196952089500494 | same as source | phi 0.497846741121; initial pointwise equals that initial gain | exact DP only | phi_init=0.65. Do not read the later path as a horizon that expired after a positive-gamma certificate. |
| Later | More fitting, same labels | 0 | 80 | 0.02 | seed 100: +0.0124842684216; seed 101: -0.017894; seed 102: +0.015047 | uncertified at initialization for gamma_pos | same as source | phi 0.481271833145 on seed 100 | exact DP only | Same optimization gap as the early checkpoint. |
| Later | +32 informative labels | 32 | 8 | 0.02 | seed 100: -0.00329970518994; seeds 101 and 102: -0.003819 and +0.000355 | uncertified at initialization for gamma_pos | final updated -0.00148875231677 on seed 100 | phi 0.766343105433 | exact DP only | The stalled point estimate does not improve the later policy. |
| Later | Updated-set bounded robust guidance | 32 | interval | 0.02 | +0.0150469306856 on 3/3 | uncertified at initialization for gamma_pos; final source -0.00069324961859 | final updated +0.000434038979338, just above gamma_pos | phi 0.25 | exact DP only | Same qualitative split as the early checkpoint: updated certificate stays positive, source certificate does not. |
| Later | Adaptation-budget sensitivity | 32 | interval | 0.05 | +0.0204608491333 on 3/3 | uncertified at initialization for gamma_pos | final updated +0.000351386643336 | phi 0 | exact DP only | A larger tabular budget is not the continuous-time rho=0.04. Return rises further and phi is driven to the boundary. |

**Interpretation matrix for the new task:**

| What was observed? | What it establishes, assuming prior reference works | What to do next |
|---|---|---|
| DP optimal policy works; exact oracle PG learns; BTL agent does not | Bottleneck is reward information/fit/deployment, not environment learnability | Compare exact likelihood, intervals and phi update |
| Exact oracle PG learns; sampled oracle PG does not | Sampling/exploration variance is the new factor | Compare gradient estimator with exact finite-horizon PG before adding BTL |
| More reward-fitting iterations help only when optimization is incomplete | Optimization error, not additional preference information | Quantify loss convergence; compare at equal fitting accuracy |
| New informative BTL pairs improve certificate but not actual return | Valid information-only effect | Preserve; do not claim a performance win |
| New informative BTL pairs change phi but not true gain | Guidance choice has changed, but its downstream effect is too small here | Report phi movement and exact gain derivative |
| New information improves exact true policy gain under matched updates | Concrete effect of informed reward deployment in this task | Confirm with 3 seeds, frozen checkpoints and exact DP |
| Pointwise gain stays available but deployed path fails | Current trajectory gap in a new tabular MDP | Do not automatically claim all-controls infeasibility |
| No source-only certificate at initial checkpoint | Original preference information insufficient already at initialization | Report as information shortfall; do not manufacture a horizon |
| True-return oracle does not improve | Optimizer/evaluator/initialization issue, not preference learning evidence | Correct reference policy first; other independent experiments continue |

## 4. Final Deliverables and Paper-Level Findings

### 4.1 Primary handoff

**The completed version of this exact 04_Reachability_to_Small_MDP.md file is the central deliverable.** The executor fills the H1–H4 tables above plus the summary tables below and Section 7. A separate narrative report is optional only. All numbers come from actual runs; no invented outcomes, no silent omission of negative controls.

Implement new code in separately named files. Commit source, configs, aggregate CSV/JSON, small tables and at most a few figures. Keep large checkpoint blobs and transient caches out of Git; store their reproducible metadata/path and optionally a tiny policy-logit checkpoint if suitable.

### 4.2 Completion/status table

| Experiment | Status (NOT RUN / PILOT / COMPLETE / BLOCKED) | Primary result | Seeds / task size | Exact reference available? | Result table filled? | Notes |
|---|---|---|---|---|---|---|
| H1 coupled reachability | COMPLETE | At rho=0.04 every admissible control loses the source certificate at t=0.012. The proved upper bound is 0.05993752641004983. | 32 steps, 16 outcomes, multi-start SLSQP plus 100 cross-check controls | yes, saved G4.1 path | yes | rho=0.08 still has a certified witness. |
| H2 three-stage extension | COMPLETE | kappa2=0.025 moves the scheduled loss from 0.012 to 0.013. Pointwise value stays 0.11422655845994001. | 64 outcomes; sampled 512 and 2048, seeds 0–2; 5 label seeds | yes, kappa2=0 matches G4.1 to 4.44e-13 | yes | The guided sampled phi path matches the schedule. |
| H3 exact tabular Deep-Sea-Treasure | COMPLETE | Exact and sampled oracle policy gradient both raise true return by about 0.108. The DP optimum is 0.7790170719066627. | H=20, 72 states, 3 sampled seeds, 512 trajectories | yes, finite-horizon DP | yes | gamma_pos frozen at 0.0004196952089500494 before H4. |
| H4 BTL reward learning and checkpoint forks | COMPLETE | Fresh-start informative feedback reaches 0.6236138236044841, next to the oracle 0.6243692767294623. The uniform control does not. | 3 label seeds on two checkpoints; H4C is one exact trajectory per branch plus a sampled check | yes, H3 DP | yes | Source comparisons have median Fisher 0, so U0 stays [0, 1]. |

### 4.3 Paper-level candidate claims

Use PARTIAL or NO when needed. A code path that executes does not by itself establish the claim.

| Candidate contribution | Status (YES / PARTIAL / NO) | Supporting result/figure | What exactly was shown? | Remaining qualification |
|---|---|---|---|---|
| All-admissible trajectory loss rather than only the chosen scheduled path | YES | outputs/horizon_bridge/h1_reachability/reachability.png | At rho=0.04, every admissible control in the 32-step 16-outcome model has source finite-step gain at most 0.05993752641004983 at t=0.012, below 0.06. The scheduled path is the best one found, and its phi gap to the optimizer is 0. | The bound is a mean-value enclosure of the finite-step model. It is not a claim about rho=0.08, where the maximal schedule stays certified, and it is not a claim about the 64-outcome model. |
| Robust progress separation persists in a longer sequential environment | YES for the scheduled path | outputs/horizon_bridge/h2_three_stage/three_stage.png | With kappa2=0.025 the scheduled source loss moves from 0.012 to 0.013. Finite-step pointwise value there is 0.11422655845994001 and the true gain is 0.11636056703423492. | This is the scheduled trajectory. H2 does not contain an all-controls proof. |
| Source versus informative BTL feedback can extend certification under real sampled policy updates | PARTIAL | H2 frozen-path rows and the sampled guided arm | On the exact frozen 64-outcome path, 128 informative labels certify the post-feedback suffix on 5/5 seeds while the source loss stays at 0.013. On sampled paths the robust controller does not move, and the first realized loss remains the pre-feedback one. | Whole-path certification of a from-start sampled trajectory was not obtained. |
| Small standard tabular policy learns when the reward is known | YES | H3 exact and sampled curves | Exact policy gradient raises true return from 0.49994372476520715 to 0.607583007273581 in 20 updates. Three sampled seeds finish within about 0.001 of that value. The DP optimum is 0.7790170719066627. | The same 20-update budget does not catch the one-step treasure policy at 0.6423841047518433. The gradient is still nonzero at the stop. |
| Actual learned BTL reward and finite guidance movement produce interpretable policy changes in a benchmark | PARTIAL | H4 matched-checkpoint table | The robust interval controller moves phi and raises early-checkpoint return by 0.0525694962716, with a positive updated certificate. The stalled point estimate near 0.766 does not. | The source interval is [0, 1]. At the later checkpoint the positive gamma is already missed at initialization. |
| Online BTL reward learning from a fresh policy actually improves or fails in a diagnosed way | YES | H4C table | From uniform initialization and phi=0.5, informative batches at updates 10 and 20 raise exact true return to 0.6236138236044841, next to the oracle's 0.6243692767294623. The uniform-label control matches the source-only run at 0.60767129069595. Three sampled seeds keep the same ordering. | The source pool's median Fisher information is 0, so its interval stays [0, 1]. No oracle policy warm start is used. |
| Extra feedback helps beyond extra reward optimization at matched resources | PARTIAL | Early +32 rows versus the robust row | Eight and eighty gradient steps on the same +32 labels produce the same policy and the same small return change. The interval controller on those labels produces a larger return change. | This is not "more optimizer iterations help." The gradient iteration has not reached the MLE. One source-label seed loses return when fitting is extended. |

### 4.4 Figures — fewer, clearer and tied to paper claims

A final maximum of three **candidate** figures is enough, if supported:

1. **H1 reachability**: a single clear graphic of scheduled certificate, best feasible found trajectory, movement budget and any proved upper bound. Do not draw an upper bound unless actually valid.
2. **H2 depth bridge**: existing 16-outcome vs new 64-outcome source certificate and pointwise curve, with exact G4.1 reference.
3. **H3/H4 real small task**: a compact matched-checkpoint return-versus-certificate comparison. Show identical starting checkpoints and label budgets.

Additional tables are better than an explosion of small confusing figures.

## 5. Hardware, Resource and Runtime Assessment

The available worker configuration is **1 RTX 5090 32GB, 12 CPU cores, 62GB RAM and about 450GB workspace disk**. H1/H2 are small enough for CPU and direct numerical optimization. H3/H4 should generally be CPU-first because tabular DP and BTL optimization are small. Use the 5090 only if measured batched simulations provide a real speedup. No new large model checkpoints should be downloaded.

Planned initial computational scale, adjustable based on meaningful results rather than arbitrary stopping limits:

| Experiment | Initial scale | CPU/GPU plan | Expected bottleneck to measure | Potential economical optimization |
|---|---|---|---|---|
| H1 | 32 policy steps, 16 outcomes, 3 horizon scopes, 4–8 solver starts | CPU | Coupled trajectory optimization and verified interval enclosure | Exact gradient/Jacobian and warm starts; bound at only relevant steps |
| H2 | 64 outcomes, 48 exact updates, 3 seeds × 2 sampled sizes for 32 steps | CPU | Python outcome-loop overhead | Vectorized enumeration and trajectory sampling |
| H3 | One discrete MO-Gymnasium task, horizon 20 (or development-frozen 40), 20–60 oracle updates, 3 sampled seeds | CPU | Repeated finite-horizon DP and rollout simulation | Precompute transitions and batched DP |
| H4 | Two frozen checkpoints, 3 label seeds, 32 source + 16/32/64 extra pairs, about 20 policy updates per branch | CPU; GPU only if valuable | Preference pool generation and branch replication | Cache fixed pairs, exact feature returns and policy baselines |

**Runtime recording:**

| Experiment | Measured pilot runtime | Estimated full runtime | Actual full runtime | Peak CPU RAM | Peak GPU VRAM | Disk use | Bottleneck / change | Notes |
|---|---|---|---|---|---|---|---|---|
| H1 | the enclosure and family sweep are about 1 s | a few minutes of SLSQP | 105.91702284291387 s | 138256 KiB | none | summary, csv-free json, one png | multi-start SLSQP | CPU only. The mean-value bound replaced a loose natural interval extension whose value at t=0.012 was 0.122. |
| H2 | self-check under 1 s | seconds | 1.1888811942189932 s | 143744 KiB | none | exact_steps.csv, summary, one png | none material | CPU only. Vectorized 64-outcome expectations. |
| H3 | one DP call is about 0.014 s | seconds | 3.0148733546957374 s | 59864 KiB | none | summary and two logit checkpoints | python DP and 2000 live episodes | CPU only. No GPU. |
| H4 | checkpoint branches finish before the fresh-start loop | a few minutes | 197.05147571396083 s exact, then 137.03049924224615 s for the sampled H4C check | 131536 KiB | none | summary and one png | one-dimensional search over exact policy gradients | CPU only. Sampled H4C uses 256 trajectories per update and agrees with the exact ordering. |

**Execution strategy:** Prototype the exact small case first, reuse caches, and avoid allocating GPU hours before the policy evaluator and learning controls work. If one stage is technically blocked, keep working on independent implementation tasks; record the specific blocker and an equivalent resolution rather than abandoning the work plan.

## 6. Terminology

- **Exact population gradient:** finite-policy gradient computed from the known finite outcome/environment model, not Monte Carlo.
- **Ex-post realized gain:** gain caused by the actual sampled gradient and policy step. It can be lower than its exact population-step counterfactual.
- **Source-only certificate:** worst compatible gain using ONLY the original preference information U0.
- **Updated certificate:** same quantity after collecting informative comparisons and replacing U0 by U_N. It can improve without any policy change.
- **Pointwise value:** best worst-compatible gain obtainable by freely choosing the current guidance reward at the current state. It ignores the path needed to reach that reward.
- **Fixed-policy-path reachability:** feasibility of a rate-limited reward sequence when the policy states are artificially held to a reference path. This is not coupled trajectory feasibility.
- **Coupled trajectory reachability:** feasibility when the guidance history changes policy states and both must satisfy their joint update equations.
- **Admissible-control witness:** a complete guidance sequence that satisfies the fixed initialization, speed limits, exact policy dynamics and every stated gain requirement. A valid witness proves existence.
- **Verified all-controls upper bound:** a bound that covers every admissible control history. An optimizer's best-found value is never by itself such a bound.
- **Reward head estimate:** trained Bradley–Terry preference weight; it is not necessarily the deployed guidance phi.
- **Reward adaptation budget:** constraint on deployed phi changes, distinct from the number of optimizer iterations fitted on existing labels.
- **Finite-horizon tabular DP:** exact expected reward vector and gradient for a known small Markov transition system with fixed policy and horizon.
- **Initial insufficient information:** source uncertainty is too wide to certify the starting policy step. This is different from certification being valid initially and later expiring.
- **Censored horizon:** no certificate loss observed within the finite evaluated number of policy updates; not infinite support.

## 7. Result Location — initially blank; executor fills after real runs

Suggested names (these files do **not** exist merely because this plan refers to them):

~~~text
experiments/bridge/12_coupled_reachability.py
experiments/bridge/13_three_stage_mdp.py
experiments/bridge/14_dst_exact_tabular.py
experiments/bridge/15_dst_preference_forks.py

outputs/horizon_bridge/h1_reachability/
outputs/horizon_bridge/h2_three_stage/
outputs/horizon_bridge/h3_dst_oracle/
outputs/horizon_bridge/h4_dst_preference/
~~~

Existing source reference experiments must remain unchanged under outputs/horizon_bridge/step00 through step10_g4_extension and experiments/horizon_real/E1–E8. Prefer small new artifacts and a clearly filled summary table to excessively verbose logs.

| Experiment | Actual result description | Actual file / folder path | Git revision | Notes |
|---|---|---|---|---|
| H1 | Coupled search and a mean-value all-controls bound for the saved 16-outcome model | experiments/bridge/12_coupled_reachability.py and outputs/horizon_bridge/h1_reachability/ | this commit | Does not rewrite step10_g4_extension. |
| H2 | 64-outcome regression, 48-step extension, sampled gradients and frozen BTL | experiments/bridge/13_three_stage_mdp.py and outputs/horizon_bridge/h2_three_stage/ | this commit | kappa2=0 reproduces the G4.1 kappa=0.05 certificate. |
| H3 | Exact tabular Deep Sea Treasure DP, oracle policy gradient and two checkpoints | experiments/bridge/14_dst_exact_tabular.py and outputs/horizon_bridge/h3_dst_oracle/ | this commit | Checkpoints store logits, not the teacher weight. |
| H4 | Continuous BTL interval, matched checkpoint forks and fresh-start H4C | experiments/bridge/15_dst_preference_forks.py and outputs/horizon_bridge/h4_dst_preference/ | this commit | Does not reuse phase1.py's simplex confidence grid. |

## 8. Mandatory Execution Sequence and Handoff

1. **Start H1 immediately from the committed 32-step 16-outcome G4.1 implementation.** First reproduce the scheduled loss at t=.012, then test alternative bounded phi histories and the full coupled optimization. Seek a valid outer bound; if it is inconclusive, distinguish known feasible witnesses from unproved impossibility.
2. **Run H2 with kappa2=0 before kappa2=.025.** This guarantees the original two-stage model is retained when the third-stage reward is absent. Only then add the extra state/action and small-sample preference learning.
3. **Build H3 Deep-Sea-Treasure exact finite-horizon model.** Reuse the old environment and teacher, discard noisy E1 certificate logic. Prove a useful policy exists via optimal DP, then verify a working oracle policy update and sampled-policy comparison before adding preferences.
4. **Execute H4 as two separate kinds of real learning.** First train real scalar BTL reward estimates and compare extra fitting versus new labels using matched H3 checkpoints. Then run the separate H4C learner from a FRESH policy with no oracle initialization and a fixed online-feedback schedule. Use exact DP to evaluate both.
5. **Fill all result tables in THIS Work Plan 04 and push them with reproducible code and small results.** Return the commit, the completed plan and the most relevant compact figure/summary.
6. **Do not launch Walker, Meta-World, or full LLM PPO here.** A later Work Plan 05 can decide whether H4 supports a larger task or instead identifies an important limitation in source information/optimizer dynamics.

## 9. References

- Repository's original exact dynamics and preference analysis: [README.md](../README.md), [horizon.py](../horizon.py), [11_two_step_horizon.py](../experiments/bridge/11_two_step_horizon.py).
- Completed sequential evidence: [Work Plan 03](03_Stochastic_Guidance_Transfer.md).
- Existing MO-Gymnasium implementation: [experiments/real/small_rl/phase1.py](../experiments/real/small_rl/phase1.py).
- Official MO-Gymnasium Deep-Sea-Treasure environment specification: https://mo-gymnasium.farama.org/environments/deep-sea-treasure/
- Task package: https://github.com/Farama-Foundation/MO-Gymnasium

**Completion instruction to the research agent:** Execute, document true measured outcomes, and update THIS SAME Work Plan 04. The purpose is to obtain a connected sequence of interpretable results, not to mechanically fill a benchmark grid or force a positive effect.
