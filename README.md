# ⏳ The Limits of Reward Guidance under Finite Adaptation

🌍 Languages: [English](README.md) | [中文](README_cn.md)

👉 **TL;DR.** Preference feedback can be sufficient to certify a policy update without identifying the reward completely. As the policy changes, however, the reward that certifies improvement may move faster than the deployed reward can adapt. **The Limits of Reward Guidance under Finite Adaptation** studies how long existing preference information can support continued policy improvement, when additional feedback helps, and what happens during actual policy training.

<p align="center">
  <b>📄 arXiv coming soon</b> &nbsp; | &nbsp;
  <a href="https://atheavenma.github.io/The-Limits-of-Reward-Guidance-under-Finite-Adaptation/"><b>🎞️ 5-minute tutorial</b></a> &nbsp; | &nbsp;
  <a href="#-main-results"><b>📊 Main results</b></a> &nbsp; | &nbsp;
  <a href="#-quickstart"><b>🚀 Quickstart</b></a>
</p>

## 🎥 Overview

<p align="center">
  <img src="assets/figures/fig1.png" alt="Existing preference information and additional feedback certify different durations of progress on an illustrated trajectory." width="850"/>
</p>

**Figure 1.** (a) The region of rewards that guarantee progress moves with the policy. The fastest reachable reward reaches its boundary at \(T_0=5\) and leaves it afterward. (b) The guarantee from the original preference data reaches the required progress at \(T_0\), although suitable rewards remain available at each time. Adding 1,200 targeted preference labels extends the guarantee along the same trajectory without changing the policy.

The central question is whether reward parameters can continue to maintain certified progress as the policy changes. A suitable reward can exist at each policy state even when no admissible reward trajectory can follow it from the given initialization.

---

## 🎞️ 5-Minute Tutorial

<p align="center">
  <a href="https://atheavenma.github.io/The-Limits-of-Reward-Guidance-under-Finite-Adaptation/"><img src="assets/tour_preview.png" alt="Three-slide preview: cover, exact support horizon, and finite policy-training results" width="900"/></a>
</p>

💻 Click to open the [tutorial](https://atheavenma.github.io/The-Limits-of-Reward-Guidance-under-Finite-Adaptation/) in your browser. The [self-contained HTML](docs/index.html) also runs locally.

---

## ✨ Highlights

- ⏳ **A finite support horizon.** We derive the exact duration over which existing preference information can certify progress in the scalar model.
- ⚖️ **Pointwise feasibility is not trajectory feasibility.** The distinction also occurs under ordinary Euclidean policy gradient, where limited reward adaptation can fail even while a suitable reward remains available at each state.
- 📏 **The value of feedback depends on the instance.** The information needed to certify a longer horizon scales with the distance to the dynamic feasibility boundary.
- 🧭 **Reward guidance affects policy learning.** Deep-Sea-Treasure experiments show that different guidance rewards can lead to different behaviors after finite training.
- ⚠️ **More information need not improve the final policy.** Human preference data improve reward estimation, but bounded reward updates using additional AI feedback do not produce a confirmed gain in the Pythia-1B experiment.

---

## 📐 What the Horizon Measures

In the scalar model, the true normalized progress under a preference parameter $`\delta`$ is

$$j_{\delta}(t)=g+\delta\,[\phi(t)-vt],\qquad
\delta\in[-\eta_0,\eta_0],\qquad \phi(0)=0,\quad |\dot\phi|\le\rho.$$

Here $`g`$ is the common gain identified by existing preference information, $`vt`$ is the moving target reward coefficient, $`\phi(t)`$ is the deployed guidance coefficient, $`\eta_0`$ bounds unresolved preference uncertainty, and $`\rho`$ bounds reward adaptation speed. The minimum guaranteed progress is

$$L(t)=g-\eta_0|\phi(t)-vt|.$$

For $`0<\gamma<g`$, $`0\le\rho<v`$, and with nonbinding reward-amplitude constraints, the **exact** support horizon for the existing preference information is

$$\boxed{T_0=\frac{g-\gamma}{\eta_0(v-\rho)}}.$$

The admissible path $`\phi(t)=\rho t`$ attains this bound. In contrast, if the instantaneous reward target $`\phi=vt`$ is allowed, pointwise optimization always attains $`g`$. Therefore pointwise feasibility need not imply dynamically maintainable certification.

Three distinct notions matter:

| Concept | Question |
|---|---|
| **Pointwise feasibility** | Is a certifying reward available at the current policy state? |
| **Trajectory feasibility** | Can a bounded-speed reward path remain certified from its initialization? |
| **Statistical certification** | Can preference feedback establish feasibility for the desired horizon? |

The paper additionally analyzes finite reward-update resources, local ordinary-policy-gradient separation, and the information needed for longer-horizon certification.

The same model also gives an instance-dependent preference-label requirement. For a target horizon beyond the original guarantee, the relevant quantity is the distance \(d_T\) to the dynamic feasibility boundary, with label complexity of order \(d_T^{-2}\log(1/\alpha)\) under the stated Bradley–Terry assumptions. Figure 2 illustrates why the target horizon alone is not enough to determine the number of labels.

---

## 📊 Main Results

<details>
<summary>📈 Figure 2. Preference label requirements depend on the distance to dynamic infeasibility</summary>

<p align="center"><a href="assets/figures/fig2.svg"><img src="assets/figures/fig2.png" alt="Label lower bounds for longer horizons and distance to the feasibility boundary." width="780"/></a></p>

Two feasible preference instances can require very different numbers of labels for the same target horizon. At \(T=15\), the quadratic lower bounds range from approximately **112 to 1,789 comparisons** across the three illustrated preferences. The dependence follows \(d_T^{-2}\), where \(d_T\) is the distance to the dynamic feasibility boundary.

These are bounds on the information needed for certification, not measurements of policy return.

</details>

<details>
<summary>🧭 Figure 3. Certified progress can fail under ordinary policy gradient</summary>

<p align="center"><a href="assets/figures/fig3.svg"><img src="assets/figures/fig3.png" alt="The deployed reward reaches the moving feasibility boundary while pointwise certified progress remains available." width="780"/></a></p>

A reward coefficient moving at the maximum permitted speed reaches the boundary of the certifying interval at approximately **0.007836**. The pointwise optimum remains above the required progress. The local analysis bounds the loss of certification for every admissible reward trajectory in the stated system by time **0.010006**.

The numerical contact time and the analytic bound apply to different statements.

</details>

<details>
<summary>🧩 Figure 4. Reward guidance changes the policy trajectory</summary>

<p align="center"><a href="assets/figures/fig4.svg"><img src="assets/figures/fig4.png" alt="Final true returns and deeper-treasure probabilities in Deep-Sea-Treasure." width="780"/></a></p>

After **160 updates** in Deep-Sea-Treasure, guidance based on the original preference information reaches deeper treasures with probability **45.6%**, compared with **0.8%** for ordinary policy gradient using the true reward in the recorded setting. Their mean true returns are **0.683** and **0.638**, respectively.

These methods also differ in how guidance is chosen. The result shows dependence on the policy trajectory, not that uncertain rewards are generally better than the true reward.

</details>

<details>
<summary>🔬 Figure 5. More human preference data improve reward estimation</summary>

<p align="center"><a href="assets/figures/fig5.svg"><img src="assets/figures/fig5.png" alt="Human preference accuracy and negative log-likelihood across training data sizes." width="780"/></a></p>

A Qwen2.5-0.5B reward model improves from **50.5% to 56.3%** mean preference accuracy as the human training comparisons increase from **257 to 4,098**. Its negative log-likelihood also decreases. On the independent final test, however, a simple length-and-overlap logistic regression reaches **58.3%**.

Better preference prediction does not itself establish improved policy guidance.

</details>

<details>
<summary>🎯 Figure 6. Additional feedback under finite reward adaptation</summary>

<p align="center"><a href="assets/figures/fig6.svg"><img src="assets/figures/fig6.png" alt="Bounded reward movement and held-out summarization quality during Pythia-1B training." width="780"/></a></p>

During **64 Pythia-1B policy updates**, a reward model incorporates **96 additional AI-labeled comparisons**. All six reward-update events satisfy the prescribed movement limit of **0.05** on the fixed comparison set. Yet the final held-out ROUGE-L difference relative to using the fixed reward is **−0.0017** (95% paired interval **[−0.0159, 0.0124]**, two training seeds).

The result separates successful reward adaptation from a confirmed improvement in policy performance. It does not establish a formal support horizon for a neural network.

</details>

For experimental settings, statistical qualifications, and source data, see [Main Results](docs/RESULTS.md) and [Experiments for the manuscript](docs/PAPER_EXPERIMENTS_08.md).

---

## 🔢 Finite Reward Update Resources

The original experiments also measure what happens when the reward receives a limited number of optimization steps before each policy update. These are clipped quadratic reward-calibration updates, not additional preference labels. The policy step size is 0.05, and the other optimizer settings remain fixed.

<table align="center">
  <thead>
    <tr>
      <th align="center">Reward updates per policy step</th>
      <th align="center">First time progress is no longer certified</th>
      <th align="center">Ideal capacity limit</th>
    </tr>
  </thead>
  <tbody>
    <tr><td align="center">0</td><td align="center">3.35</td><td align="center">3.35</td></tr>
    <tr><td align="center">1</td><td align="center">4.55</td><td align="center">5.05</td></tr>
    <tr><td align="center">2</td><td align="center">7.85</td><td align="center">10.05</td></tr>
    <tr><td align="center">3</td><td align="center">Not within 15</td><td align="center">Not within 15</td></tr>
  </tbody>
</table>

The finite update rule can fall behind the ideal adaptation budget because it does not instantly attain its maximum permitted speed. This is a separate question from how many comparisons are needed to reduce preference uncertainty. [Full numerical evidence](docs/PAPER_RESULTS.md).

---

## ⚙️ Install

Python 3.10+.

~~~bash
git clone https://github.com/AtheavenMa/The-Limits-of-Reward-Guidance-under-Finite-Adaptation.git
cd The-Limits-of-Reward-Guidance-under-Finite-Adaptation
python -m pip install -e ".[demo]"
~~~

## 🚀 Quickstart

Run an exact scalar example:

~~~python
from horizon import analyze_horizon

result = analyze_horizon(
    gain=1.0,
    required_gain=0.6,
    uncertainty_radius=1.0,
    drift_speed=0.12,
    adaptation_speed=0.04,
    target_horizon=15.0,
)
print(result.support_horizon)      # 5.0
print(result.pointwise_feasible)   # True
print(result.trajectory_feasible)  # False
~~~

For an executable plot, run:

~~~bash
python demo/demo_feedback.py
~~~

To check a finite model using your own probabilities and rewards, use the [reward-guidance audit](docs/AUDIT.md):

~~~bash
horizon-audit audit --manifest examples/08_guidance_audit/finite_computed.json --out outputs/audit_demo
~~~

The audit computes finite-model progress from supplied outcomes. It does not turn empirical neural-model measurements into mathematical guarantees.

---

## 🗺️ Figure Map

| Figure | Main question | Files |
|---|---|---|
| Figure 1 | How long can existing preference information maintain certification? | [PNG](assets/figures/fig1.png) · [SVG](assets/figures/fig1.svg) · [PDF](assets/figures/fig1.pdf) |
| Figure 2 | How much feedback is needed beyond the horizon? | [PNG](assets/figures/fig2.png) · [SVG](assets/figures/fig2.svg) · [PDF](assets/figures/fig2.pdf) |
| Figure 3 | Can bounded reward adaptation keep up with policy gradient? | [PNG](assets/figures/fig3.png) · [SVG](assets/figures/fig3.svg) · [PDF](assets/figures/fig3.pdf) |
| Figure 4 | How does reward guidance change policy behavior? | [PNG](assets/figures/fig4.png) · [SVG](assets/figures/fig4.svg) · [PDF](assets/figures/fig4.pdf) |
| Figure 5 | How does human feedback affect reward estimation? | [PNG](assets/figures/fig5.png) · [SVG](assets/figures/fig5.svg) · [PDF](assets/figures/fig5.pdf) |
| Figure 6 | Do bounded reward updates improve a trained policy? | [PNG](assets/figures/fig6.png) · [SVG](assets/figures/fig6.svg) · [PDF](assets/figures/fig6.pdf) |

The original numerical experiments and their historical figure index are preserved in [Figure Provenance](docs/FIGURE_MAP.md). The editable main-result figure sources are in [plotting](plotting/render_main_figures.py).

---

## 🔬 Reproduction

~~~bash
python -m pytest -q
python experiments/reproduce.py --all
~~~

The original mathematical checks and numerical figures remain reproducible. The earlier reference suite also checks the discrete reward update experiment, the prediction-versus-guidance example, four JSON files, five CSV files, and seven historical numerical figures against frozen reference values. Generated files are stored in `outputs/paper_reproduction/` without overwriting those references. The newer human-preference, RL, and language-model results have separate [experiment records](docs/RESULTS.md) and [reproduction instructions](docs/FINAL_REPRODUCE_08.md).

## 🙏 Appreciation

We appreciate the research community's contributions to preference learning, policy optimization, safe control, and information theory.

## 📚 Citation

📄 arXiv coming soon. Until the public manuscript citation is available, please use [CITATION.cff](CITATION.cff) to cite the research software.

MIT licensed. [Contributing](CONTRIBUTING.md) · [Changelog](CHANGELOG.md)
