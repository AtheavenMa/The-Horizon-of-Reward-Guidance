# Figure map: all seven manuscript figures

These are the **actual figures generated from the supplied numerical code and ledgers**, not newly invented diagrams. The PNG versions in [`assets/figures/`](../assets/figures) are high-resolution displays. `plotting/figures.py` creates publication-format PDF and PNG copies from saved output data; it does not simulate new observations.

| Manuscript figure | Source data / calculation | Generator | Display |
|---|---|---|---|
| **1**. [Certificate expiration and feedback](#figure-1--closed_loop) | `closed_loop.csv`, `verification_results.json` | `fig_closed_loop` | [`closed_loop.png`](../assets/figures/closed_loop.png) |
| **2**. [Ordinary PG safe boundary contact](#figure-2--ordinary_pg) | `ordinary_pg.csv` | `fig_ordinary_pg` | [`ordinary_pg.png`](../assets/figures/ordinary_pg.png) |
| **3**. [Preference-label complexity](#figure-3--instance_complexity) | `binary_gap_bounds.csv` | `fig_instance_complexity` | [`instance_complexity.png`](../assets/figures/instance_complexity.png) |
| **4**. [Query information energy](#figure-4--query_energy) | Rank-one-design JSON and recorded parameters; plotted curve reconstructed analytically | `fig_query_energy` | [`query_energy.png`](../assets/figures/query_energy.png) |
| **5**. [Weak Gaussian identification](#figure-5--full_support_separation) | `weak_identification.csv` | `fig_full_support` | [`full_support_separation.png`](../assets/figures/full_support_separation.png) |
| **6**. [Clipped calibration versus ideal speed budget](#figure-6--actual_adaptation) | Analytic curve reconstructed from recorded clipped-calibration parameters | `fig_actual_adaptation` | [`actual_adaptation.png`](../assets/figures/actual_adaptation.png) |
| **7**. [Discrete reward update resources](#figure-7--discrete_resources) | `discrete_resources.csv`, `discrete_resources.json` | `fig_discrete_resources` | [`discrete_resources.png`](../assets/figures/discrete_resources.png) |

All generators are in [`plotting/figures.py`](../plotting/figures.py), with shared style in `plotting/paperfig.py`. Relative data paths are resolved by `--res` and output paths by `--out`.

## Figure 1 — closed_loop

![Figure 1: source-only and updated preference certificate](../assets/figures/closed_loop.png)

Pointwise rewards continue to be available, but the source-only certificate of the specified reward trajectory reaches its boundary at $t=5$. A representative fixed BTL batch of 1,200 labels narrows the compatible preference set, keeping the same reward trajectory certified through $t=15$. The confidence set updates only after feedback becomes available at $t=4$.

## Figure 2 — ordinary_pg

![Figure 2: ordinary PG boundary contact](../assets/figures/ordinary_pg.png)

The moving lower safe boundary catches the maximal-speed reward coefficient under ordinary Euclidean policy gradient. The numerical contact time is approximately 0.0078. An analytic local proof, not the image alone, establishes failure for **all** bounded-speed reward controls in the stated neighborhood.

## Figure 3 — instance_complexity

![Figure 3: expected label lower bound versus target horizon](../assets/figures/instance_complexity.png)

Instance-dependent necessary expected label lower bounds for requested horizons and different feasible true preferences, under the specified Bradley–Terry query class. Increasing horizon need not have one universal quadratic cost: the primitive difficulty parameter is distance to the dynamic infeasibility boundary.

## Figure 4 — query_energy

![Figure 4: directional uncertainty under repeated queries](../assets/figures/query_energy.png)

A repeated comparison can reduce directional uncertainty but leave a nonzero asymptotic residual if its feature difference does not span all task-relevant unknown directions. This curve is reconstructed from the recorded rank-one model parameters rather than a separate CSV series.

## Figure 5 — full_support_separation

![Figure 5: weak Gaussian identification](../assets/figures/full_support_separation.png)

With fixed-design Gaussian ratings and nonzero source support, OOD prediction risk becomes sensitive to a weak direction while estimation variance of an identified common gain stays controlled. **These are not the same metric for two compared algorithms.**

## Figure 6 — actual_adaptation

![Figure 6: calibration startup loss](../assets/figures/actual_adaptation.png)

An explicit clipped quadratic-calibration rule does not immediately use its maximum speed budget. The lag reaches the certification boundary at approximately 4.9055 rather than the ideal-envelope time 5. This curve is analytically reconstructed from recorded parameters.

## Figure 7 — discrete_resources

![Figure 7: finite reward updates](../assets/figures/discrete_resources.png)

Actual clipped calibration steps change the first uncertified policy-step time under fixed Euler policy updates. Increasing $m$ changes the number of reward optimization steps, not the underlying policy step size or the clipping threshold.

For interpretation and all reported numerical checks, see [Paper Results](PAPER_RESULTS.md) and [Scope](SCOPE.md).
