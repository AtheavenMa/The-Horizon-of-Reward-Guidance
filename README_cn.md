# ⏳ The Horizon of Reward Guidance

🌍 Languages: [English](README.md) | [中文](README_cn.md)

👉 **TL;DR.** 偏好反馈可能已经足以支持一次有保证的策略更新，却仍然无法确定奖励函数的所有方向。随着策略持续变化，能够保证进展的奖励参数集合可能移动得比奖励模型的适应速度更快。**The Horizon of Reward Guidance** 研究现有偏好信息在有限奖励适应能力下能够维持多久的有效指导，证明逐点可行不代表整条轨迹可行，并量化认证更长时间所需的额外二元偏好反馈。

<p align="center">
  <b>📄 arXiv coming soon</b> &nbsp; | &nbsp;
  <a href="#-快速上手"><b>🚀 快速上手</b></a> &nbsp; | &nbsp;
  <a href="#-演示"><b>🧪 演示</b></a>
</p>

## 🎥 Overview

<p align="center">
  <img src="assets/figures/closed_loop.png" alt="Reward guidance 证书在 t=5 到期，增加 1200 个二元偏好标签后可认证到 t=15" width="850"/>
</p>

**图 1.** 每个时刻仍然存在能够提供合适指导的奖励参数，但从固定初始点出发、满足有限适应速度的奖励轨迹，其仅依赖源信息的证书在 **t=5** 到达临界值。论文的五输出有限模型展示：额外获得一批代表性的 **1,200 个二元偏好标签**后，同一条轨迹可以被认证到 **t=15**。这不是神经网络奖励模型的实际训练实验。

---

## ✨ Highlights

- 🎯 **精确的标量支持期限。** 在移动目标与有限适应速度的假设下，给出持续鲁棒进展的必要充分条件。
- 🔄 **逐点可行不意味着轨迹可行。** 每个策略状态都可能存在合格奖励，但从给定初始状态出发，没有任何速度受限的奖励路径能持续保持认证。
- 📈 **标准策略梯度下同样可能失效。** 局部反例适用于普通 Euclidean Policy Gradient 与 full-support policy，而不只是人工构造的预条件矩阵。
- 🧮 **偏好标签复杂度。** 在有界 Bradley–Terry 二元比较模型中，认证成本取决于真实偏好到动态不可行边界的距离。
- 🧪 **可复现的理论数值实验。** 仅需 CPU；包含核心数学接口、三个 Demo、原始数值验证脚本、论文全部七张图及其数据，无需模型权重或外部 API。

---

## 💡 为什么 Reward Guidance 有 Horizon？

现有偏好数据能够排除部分奖励参数，却不一定能区分所有真正不同的奖励函数。在某个策略状态，所有兼容偏好可能都认可某个更新方向，因此当前指导可以被认证。但在后续策略状态，相同的未知方向可能变得重要。

关键是要区分三个问题：

| 层次 | 核心问题 | 无法推出的结论 |
|---|---|---|
| **Pointwise Feasibility** | 当前状态是否存在合格奖励？ | 是否能够从现有奖励参数出发及时到达 |
| **Trajectory Feasibility** | 给定初始奖励及速度预算，是否存在全程合格的路径？ | 指定的奖励优化器是否一定会跟随它 |
| **Statistical Certification** | 现有偏好数据能否排除使目标期限不可行的偏好参数？ | 奖励模型的实际适应速度是否提高 |

因此，Reward Guidance 的失效可能是 **轨迹无法跟上移动约束**，也可能是 **当前偏好信息不足以证明它能跟上**。额外标签能解决第二个问题，但不能自动解决第一个问题。

---

## 📐 数学定义

标量模型的真实归一化进展写为

$$j_\delta(t)=g+\delta[\phi(t)-vt],\quad
\delta\in[-\eta_0,\eta_0],\quad
\phi(0)=0,\quad |\dot\phi|\le\rho.$$

其中 $g$ 为源信息已经识别的共同进展，$vt$ 为随策略变化的目标奖励系数，$\phi(t)$ 为实际指导系数，$\eta_0$ 描述偏好未识别方向的范围，$\rho$ 是奖励参数允许的最大适应速度。

对所有兼容偏好取最差情况，得到证书：

$$L(t)=g-\eta_0|\phi(t)-vt|.$$

当 $0<\gamma<g$、$0\le\rho<v$ 且奖励幅度约束不会先成为瓶颈时，精确支持期限为

$$\boxed{T_0=\frac{g-\gamma}{\eta_0(v-\rho)}}.$$

轨迹 $\phi(t)=\rho t$ 能达到这个上界。与此同时，如果允许每个时刻独立选择 $\phi=vt$，逐点最优值始终为 $g$。这就产生了论文的核心分离：**逐点有解，但全程没有可到达的解。**

针对更长的目标期限 $T$，论文还证明在给定条件下，Bradley–Terry 比较的标签复杂度为

$$\boxed{N=\Theta\!\left(d_T^{-2}\log(1/\alpha)\right)},\qquad
d_T=\frac{g-\gamma}{(v-\rho)T}-\delta_\star>0.$$

请参阅 [理论与假设说明](docs/THEORY.md)，了解模型来源、适用条件与定理边界。

---

## 🔎 主要结果

### 1. 偏好反馈延长认证期限

| 量 | 论文实例 |
|---|---:|
| 仅依赖源偏好的精确支持期限 | **5.0** |
| 新增二元偏好标签 | 1,200 |
| 示例正标签数 | 600 |
| 查询后的目标认证期限 | **15.0** |
| 查询后沿轨迹的最低证书 | **0.682** |
| 要求达到的归一化进展 | 0.600 |

该图中反馈在 $t=4$ 时变得可用。标签是**说明性的固定批次**，并不是来自真实标注人员的数据。

### 2. 实际奖励更新资源

| 每个策略步骤的奖励更新次数 $m$ | 首次失去认证的时间 | 理想预算对应时间 |
|---:|---:|---:|
| 0 | 3.35 | 3.35 |
| 1 | 4.55 | 5.05 |
| 2 | 7.85 | 10.05 |
| 3 | 15 内未失证 | 15 内未失证 |

该实验使用**截断的二次奖励校准更新**，并非 Bradley–Terry 奖励似然训练。策略 Euler 步长固定为 0.05，只改变每次策略更新前执行的奖励更新次数。

### 3. 标准 Policy Gradient 的局部反例

论文在 full-support policy 内给出严格局部分析：所考虑区域内的 pointwise optimum 始终大于 0.10575，而要求的最小进展为 0.06。尽管如此，所有满足速度限制的奖励路径都会在不超过约 0.010006 的时间内失去证书。数值上，采用最大允许奖励更新速度的路径约在 0.007836 时接触安全边界。

其全控制路径结论来自解析证明，而不是从单一数值轨迹推断而来。

更多结果、实验定义和辅助验证请参阅 [论文数值结果](docs/PAPER_RESULTS.md)。

---

## ⚙️ Install

Python **3.10+**。核心 [`horizon.py`](horizon.py) 只使用 Python 标准库，演示和完整实验另外需要 NumPy、SciPy、Matplotlib。

```bash
git clone https://github.com/AtheavenMa/The-Horizon-of-Reward-Guidance.git
cd The-Horizon-of-Reward-Guidance
python -m pip install -r requirements.txt
```

也可以在当前仓库目录内执行 `python -m pip install -e .` 安装核心模块，完整开发依赖则执行 `python -m pip install -r requirements_dev.txt`。

---

## 🚀 快速上手

```python
from horizon import analyze_horizon

result = analyze_horizon(
    gain=1.0,
    required_gain=0.6,
    uncertainty_radius=1.0,
    drift_speed=0.12,
    adaptation_speed=0.04,
    target_horizon=15.0,
)

print(result.support_horizon)            # 5.0
print(round(result.minimum_required_speed, 6))  # 0.093333
print(result.pointwise_feasible)         # True
print(result.trajectory_feasible)        # False
```

更多接口见 [API 文档](docs/API.md)：离散时刻的区间可行性用 `interval_feasible()` 检查；有界 Bradley–Terry 模型下的样本量分析、固定样本单侧检验和双侧置信区间分别由 `btl_sample_bounds()`、`certify_btl_batch()` 与 `btl_confidence_interval()` 提供。

这些都是带有明确假设的数学工具，不是对任意实际 RLHF 系统的通用认证。

---

## 🧪 演示

**Demo 1：精确标量 Horizon**

```bash
python demo/demo.py
```

<p align="center"><img src="assets/demo_horizon.png" alt="标量 Horizon 演示" width="750"/></p>

输出 $T_0=5$，并显示 pointwise feasibility 与 trajectory feasibility 为什么不同。保存 `outputs/demo_horizon/demo_output.png` 和 `summary.json`。

**Demo 2：通过偏好反馈延长认证**

```bash
python demo/demo_feedback.py
```

<p align="center"><img src="assets/demo_feedback.png" alt="偏好反馈示例" width="750"/></p>

使用 600/1200 个固定二元标签，计算置信区间及查询后 0.682 的最低证书，并单独检查针对 $T=15$ 的固定样本单侧检验。

**Demo 3：普通 Policy Gradient 的局部失效**

```bash
python demo/demo_ordinary_pg.py
```

<p align="center"><img src="assets/demo_ordinary_pg.png" alt="普通 PG 安全边界接触示例" width="750"/></p>

绘制允许最大更新速度的奖励路径与移动安全边界，同时打印解析的 all-controls 上界。数值图本身不承担证明。

---

## 📊 论文图表索引

七张原始论文图、对应输入 CSV/JSON、图像生成函数及结果解释已经汇总于 [Figure Map](docs/FIGURE_MAP.md)。绘图代码会从冻结的数值数据重建图像，不会在绘图时重新模拟或修改结果。对于没有逐点 CSV 的解析曲线，文档明确说明数据由已记录参数重建。

---

## 🔬 完整复现

```bash
python experiments/reproduce.py --all
python -m pytest -q
```

第一条命令运行完整原始验证流程，重新生成七张图，并将新得到的 **4 份 JSON、5 份 CSV、7 张图**与 `results/reference/` 中的冻结数据逐项比较。输出在 `outputs/paper_reproduction/`，不会覆盖参考文件。

详细环境、随机种子、命令及论文结果映射见 [Reproduction Guide](docs/REPRODUCE.md)。

---

## ⚠️ 研究范围与预印本状态

- 已复现的是有限输出和标量数学模型的分析与数值实验，**没有**实际神经网络 Reward Model 训练、PPO/GRPO 或 LLM benchmark。
- 速度预算描述的是允许的最大能力，不代表某个具体优化器能自动达到。
- 固定样本 BTL 检验不能在未重新论证有效性的情况下用于任意停止或任意自适应选样。
- 本仓库暂不公开论文正文 PDF 或 LaTeX 源码。**arXiv coming soon.** 现有全部图表、结果、Demo 和数值验证代码继续公开。
- 数值验证支持论文给出的具体构造，不能替代定理证明，也不代表真实神经奖励模型的经验验证。

请参阅完整的 [Scope](docs/SCOPE.md) 和 [Third-party notices](THIRD_PARTY_NOTICES.md)。

---

## 🙏 Appreciation

感谢相关偏好学习、强化学习、安全控制、信息理论等研究工作的作者。具体引文和数学来源以随附论文参考文献为准。

## 📜 Citation & License

**论文预印本：arXiv coming soon。** 当前仓库不提供论文正文。完整作者名单、发表状态与正式引用信息以之后公开的 arXiv 预印本或会议记录为准。[`CITATION.cff`](CITATION.cff) 描述的是**软件工件**，不用于确定论文的完整作者名单。

仓库原创代码和项目文档采用 [MIT License](LICENSE)；未来公开论文的著作权与发表许可单独处理，根目录 MIT 许可证不对论文或第三方内容作重新授权。


开发与发布流程见 [Contributing](CONTRIBUTING.md)、[Changelog](CHANGELOG.md) 和 [发布检查清单](RELEASE_CHECKLIST.md)。
