# Scope, limitations, and safeguards against misinterpretation

The project investigates how **existing preference information and finite reward adaptation** can jointly certify continued improvement along a policy path. Its contribution is a theoretical distinction and explicit finite-model numerical evidence.

## What is established by the paper

1. An **exact necessary-and-sufficient scalar horizon** under the specified symmetric source set, linear target drift, initial reward coefficient, and nonbinding amplitude limit.
2. A **moving-interval reachability criterion**, plus explicit finite-policy constructions and a **local** ordinary Euclidean policy-gradient example where pointwise feasibility persists while all allowed reward trajectories lose certification.
3. An **instance-level preference-query complexity** result under bounded Bradley–Terry binary comparisons and an informative pure comparison, with lower and upper orders set by distance to the infeasible boundary.
4. Numerical checks of finite-output models, actual clipped quadratic calibration, Gaussian weak-identification comparisons, and the corresponding deterministic or Monte Carlo quantities.

## What is **not** established or implemented

- No **neural reward model training** or language-model fine-tuning; no PPO, DPO, GRPO or neural RLHF benchmark is included.
- No claim that **pointwise** availability implies a controller can track the corresponding reward states.
- No guarantee that a **particular optimizer** attains the maximum feasible speed. The clipped-calibration experiment explicitly demonstrates startup loss below an ideal speed envelope.
- No universal solver for the **coupled** endogenous policy/reward dynamics. The scalar API and discrete-corridor feasibility function assume their own declared models.
- No real human preference collection; BTL data used in the demos are synthetic or illustrative fixed counts.
- No optional-stopping validity of the **fixed-N** hypothesis test or Hoeffding confidence interval; the original code's anytime statistical check is separate.
- No empirical claim of universal neural-model safety or actual policy robustness based on these finite-system tests.
- No assertion that numerical plots themselves prove all-controls theorems, novelty, or a complete literature priority review.

## Reading the figures responsibly

The paper figures illustrate the model's consequence given its assumptions. The displayed post-query extension is achieved by **narrowing the source-compatible preference set**. It does **not** change the physical adaptation speed or the plotted policy/reward trajectory. The Ordinary PG visual shows a maximal-speed control and accompanies, but does not replace, an analytic *all-controls* local proof. The Gaussian risk figure compares **different squared-error quantities**, not two algorithms competing on one metric.

For experimental provenance, see [Paper Results](PAPER_RESULTS.md) and [Reproduction](REPRODUCE.md). The complete preprint is **arXiv coming soon**.
