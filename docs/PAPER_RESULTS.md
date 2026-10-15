# Paper results and numerical evidence

Numerical results from [`results/reference/paper_results_v10.json`](../results/reference/paper_results_v10.json) and the accompanying CSV files, organized by experiment.

## E1: five-output moving certificate with targeted BTL feedback

| Property | Recorded value |
|---|---:|
| Finite-output model | 5 outputs, full support in the displayed trajectory |
| Common normalized gain $g$ | 1.0 |
| Required normalized gain $\gamma$ | 0.6 |
| Source uncertainty $\eta_0$ | 1.0 |
| Target drift $v$ | 0.12 |
| Reward speed $\rho$ | 0.04 |
| Exact source-only support horizon $T_0$ | **5.0** |
| Evaluated extension horizon $T$ | 15.0 |
| Feedback available at | $t=4.0$ |
| Independent targeted BTL comparisons | 1,200 |
| Representative positive labels | 600 |
| Query-updated preference interval | $[-0.19246,0.19246]$ |
| Minimum certificate on illustrated post-query trajectory | **0.6820** |
| Minimum output probability | 0.12330 |
| Minimum preconditioner eigenvalue | 1.75464 |

The source-only certificate falls below the required gain beyond $t=5$ even as the pointwise optimum remains available. With the representative fixed-size feedback batch, the compatibility set shrinks, and the *same* trajectory stays certified through $t=15$. See [Figure 1](FIGURE_MAP.md#figure-1--closed_loop).

This example uses a fixed simulated feedback batch and keeps the policy/reward trajectory unchanged.

## E2: local ordinary Euclidean policy-gradient example

| Quantity | Recorded value |
|---|---:|
| Initial policy means | $(x,y)=(0.8,0.0)$ |
| Initial coefficient $\phi$ | 0.84280432 |
| Required gain | 0.06 |
| Allowed coefficient speed | 0.04 |
| Initial certified worst gain | 0.0612592 |
| Initial pointwise optimum | 0.11473088 |
| Analytic lower pointwise optimum throughout proof box | **0.10575039** |
| Analytic lower bound on safe-boundary velocity | 0.13993981 |
| Analytic lower bound on output probability | 0.04655 |
| Upper bound for all-controls boundary loss within the box | **0.01000602** |
| Numerical maximal-speed boundary contact | 0.00783587 |

The analytic local bound covers all admissible controls; the numerical trajectory gives the contact time. See [Figure 2](FIGURE_MAP.md#figure-2--ordinary_pg).

## E3: clipped reward-calibration startup loss

| Quantity | Time |
|---|---:|
| Clipping becomes saturated | 0.40546511 |
| Ideal speed-envelope deadline | **5.00000000** |
| Exact clipped-calibration deadline | **4.90546511** |
| Independent numerical ODE deadline | 4.90546511 |

The experiment follows a clipped **quadratic guidance calibration** flow. Startup transient effects shorten the realized duration relative to the ideal speed-envelope controller. See [Figure 6](FIGURE_MAP.md#figure-6--actual_adaptation).

## E4: actual finite reward updates per policy step

Both models and step rules are explicitly fixed. The policy uses Euler step size 0.05, reward-calibration learning rate 0.01, clipping magnitude 0.2, and 300 policy steps across the 15-unit window. Only the number of reward updates before each policy step changes.

| Reward updates $m$ | Maximum possible displacement per step | Observed first uncertified time | Ideal capacity-envelope time |
|---:|---:|---:|---:|
| 0 | 0.000 | 3.35 | 3.35 |
| 1 | 0.002 | 4.55 | 5.05 |
| 2 | 0.004 | 7.85 | 10.05 |
| 3 | 0.006 | Not within 15 | Not within 15 |

Across the runs, the smallest output probability is **0.11686156**, the smallest preconditioner eigenvalue is **1.75964264**, and the maximum finite-step mean-gain identity discrepancy is **$7.06\times10^{-13}$**. These checks validate the numerical construction. See [Figure 7](FIGURE_MAP.md#figure-7--discrete_resources).

## E5: prediction accuracy and useful guidance can differ

The supplied code independently recomputes the paper's four-output example. Source feedback leaves a direction of true reward unidentified, but a guidance reward can make positive progress for either of two source-compatible targets.

| Target true reward | Static source fit: OOD MSE | Dynamic guidance: OOD MSE | Static progress | Dynamic progress |
|---|---:|---:|---:|---:|
| $r_A$ | 4.5000 | 2.4853 | -0.3704 | **0.1147** |
| $r_B$ | 4.5000 | 7.1085 | 0.9352 | **0.1147** |

Dynamic guidance has higher OOD MSE for $r_B$ while preserving a positive common improvement direction. Data: [`prediction_guidance_separation.json`](../results/reference/prediction_guidance_separation.json). Function: `experiments/separation_check.py`.

## Instance-level binary preference queries

Under the paper's informative fixed pure comparison and target error probability $\alpha=0.05$, numerical checks evaluate target horizons $T\in\{5.1,6,10,15,25\}$ and three preference locations, keeping only strictly feasible conditions.

| Diagnostic | Recorded value |
|---|---:|
| Feasible fixed-design checks | 14 |
| Smallest exact power at the sufficient batch sizes | **0.9912333** |
| Largest infeasible-boundary acceptance probability | **0.0076684** |
| Fixed test for $T=15$, $N=1,200$: threshold positive labels | **638** |
| Fixed test at truth $\delta=0$: acceptance probability | 0.9869058 |
| Fixed test at feasibility boundary: acceptance probability | 0.0065814 |

The fixed-N one-sided test and the two-sided source-set update serve different purposes. [Figure 3](FIGURE_MAP.md#figure-3--instance_complexity) shows the relationship between label complexity and distance to the dynamic feasibility boundary.

## Additional checks inherited from the v0.6 verification suite

| Object | Scope | Recorded result |
|---|---|---|
| Discrete nonuniform moving corridors | 180 cases | 120 feasible, 60 infeasible, **0 LP mismatches** |
| Piecewise corridor regression | 150 cases | 100 feasible, 50 infeasible |
| Reward-gradient displacement and logistic KL inequalities | 10,000 random cases | No violations |
| Restricted comparison pools | 36 random pools | 28 feasible, 8 infeasible, **0 exhaustive-search mismatches** |
| Adaptive BTL finite simulation | 100 trajectories, 40 labels, 8 checkpoints | Empirical simultaneous coverage 1.0 |
| Jacobian-vector interface | Numerical identities | Identity error $1.73\times10^{-18}$; finite-difference discrepancy $5.32\times10^{-11}$ |
| Weak identification with Gaussian scores | 8 conditions, 50,000 replicates each | Maximum relative errors approximately 1.33% (prediction) and 1.61% (identified gain) |

### Notes on auxiliary figures

[Figure 4](FIGURE_MAP.md#figure-4--query_energy) shows repeated-comparison directional information with an irreducible residual when the query direction is insufficient. [Figure 5](FIGURE_MAP.md#figure-5--full_support_separation) uses **fixed-design Gaussian scores**, not BTL labels, and compares two *different* squared-error quantities. 

For complete numeric precision and output generation, see [`results/reference/`](../results/reference) and [Reproduction](REPRODUCE.md).
