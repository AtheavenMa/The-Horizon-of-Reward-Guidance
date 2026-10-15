# Theory and assumptions

This file explains the paper's objects and derives the scalar equations implemented by `horizon.py`. The complete manuscript, including full proofs and precise regularity conditions, will be available with the forthcoming preprint (**arXiv coming soon**). The software does not substitute for the paper's theorems.

## 1. What is being certified?

Let a guidance reward with parameter $\phi$ induce a policy-velocity vector $b=B_\theta\phi$. The true reward vector $r$ is not known exactly; source preference observations restrict it to a declared compatibility set $\mathcal U$. The worst-case instantaneous improvement in true reward is

$$L_{\mathcal U}(b)=\inf_{r\in\mathcal U}r^\top b.$$

A valid instantaneous certificate requires $L_{\mathcal U}(B_\theta\phi)\ge\gamma$ for the desired $\gamma>0$. A *pointwise* certificate may choose a fresh $\phi$ independently at each state. A *trajectory* certificate requires one admissible path starting from a specified initialization, obeying its finite speed or displacement budget, and maintaining this inequality for every time up to the target horizon.

The latter is the subject of the paper: the certified set can remain nonempty while moving too rapidly for the deployed reward to remain in it.

## 2. Theorem 1: exact scalar support horizon

In the scalar model,

$$j_\delta(t)=g+\delta(\phi(t)-vt),\quad
\delta\in[-\eta_0,\eta_0],\quad
\phi(0)=0,\quad |\dot\phi|\le\rho.$$

With $0<\gamma<g$, $\eta_0>0$, and $v>\rho\ge0$, the worst compatible gain is obtained by minimizing over the unknown $\delta$:

$$\inf_{|\delta|\le\eta_0}j_\delta(t)=g-\eta_0|\phi(t)-vt|.$$

Because $\phi(0)=0$ and $\dot\phi\le\rho$, every allowed path has $\phi(t)\le\rho t$. As the target moves at speed $v>\rho$, its lag satisfies $vt-\phi(t)\ge(v-\rho)t$. Consequently every admissible path has certified gain at most

$$g-\eta_0(v-\rho)t.$$

The path $\phi(t)=\rho t$ attains this upper bound for all $t$. Requiring at least $\gamma$ yields the exact longest support duration:

$$\boxed{T_0=\frac{g-\gamma}{\eta_0(v-\rho)}}.$$

**Interpretation of equality:** the progress guarantee *still holds* at $t=T_0$; it becomes false strictly afterward. When $\rho\ge v$, the target can be tracked and the speed limitation alone imposes no finite horizon, assuming nonbinding amplitude constraints.

Solving the same condition for a desired duration $T>0$ gives the minimum speed budget

$$\boxed{\rho_{\min}(T)=\max\{0,\ v-(g-\gamma)/(\eta_0T)\}}.$$

All of these identities hold for the declared scalar model. A finite $|\phi|$ bound can create a new limitation that is **not** represented by `support_horizon`.

## 3. Moving feasible intervals and discrete reachability

Suppose certification at time $t$ is equivalent to $\phi(t)\in[\ell(t),u(t)]$. For an externally specified continuous interval, the paper gives a cross-time condition:

$$\ell(s)\le u(t)+\rho|s-t|\quad\text{for all }s,t\in[0,T],$$

with the prescribed singleton initialization, as the criterion for existence of an absolutely continuous path with $|\dot\phi|\le\rho$.

In discrete time, suppose intervals at time nodes $k=0,\ldots,n-1$ are $[\ell_k,u_k]$ and allowed successive displacement is $s_k\ge0$. The exact reachable interval can be computed recursively as

$$R_0=[\ell_0,u_0],\qquad
R_{k+1}=[\ell_{k+1},u_{k+1}]\cap(R_k+[-s_k,s_k]).$$

An empty intersection implies infeasibility. If all intersections are nonempty, backward selection yields a witness path. This is what `interval_feasible()` implements and what the test suite compares to an independent linear program. **It does not certify what happens between sample nodes** and it is not a solver for the endogenous coupled policy–reward system.

## 4. Theorem 2: ordinary Euclidean policy gradient

For binary coordinates $(q,s)\in\{-1,1\}^2$, consider

$$p_\theta(q,s)=\tfrac14(1+xq)(1+ys),\quad
x=\tanh(\theta_q),\ y=\tanh(\theta_s).$$

With ordinary Euclidean policy gradient and guidance reward $R_\phi=\phi q+(1-\phi)s$,

$$\dot x=(1-x^2)^2\phi,\qquad
\dot y=(1-y^2)^2(1-\phi).$$

Writing $\omega_q=(1-x^2)^2$ and $\omega_s=(1-y^2)^2$, the robust safe interval for the paper's source-compatible reward class is

$$\ell(x,y)=\frac{\omega_s+\gamma}{2\omega_q+\omega_s},\quad
u(x,y)=\frac{2\omega_s-\gamma}{\omega_q+2\omega_s}.$$

The pointwise optimum remains $V=\omega_q\omega_s/(\omega_q+\omega_s)$. The proof exhibits a full-support interior initial state where $V>\gamma$ in a neighborhood, but the lower safe boundary moves more quickly than any permissible reward path. The analytic local all-controls upper bound is $0.0100060$, and a maximal-speed numerical trajectory touches the boundary at approximately $0.00783587$.

This is a **local counterexample**. The numerical path does not by itself prove failure for all possible controls; the paper establishes the latter through an analytic neighborhood and speed bound.

## 5. Theorem 3: instance-level Bradley–Terry complexity

For a target horizon $T>T_0$, define $\Delta=g-\gamma$, $a_T=(v-\rho)T$, critical preference $\delta_c=\Delta/a_T$, and feasible true preference $\delta_\star<\delta_c$. The gap to infeasibility is

$$d_T=\delta_c-\delta_\star>0.$$

Under the paper's bounded adaptive binary-comparison class,

$$\Pr(Y_i=1\mid\mathcal F_{i-1})=\sigma(c_i+q_i\delta),\qquad |q_i|\le Q,$$

any reliable certification procedure must acquire information about alternatives near $\delta_c$. The paper's KL argument supplies a lower bound on expected squared query sensitivity, giving an expected-label lower bound of order $d_T^{-2}\log(1/\alpha)$ for bounded $Q$.

If the query pool contains a fixed pure comparison with positive sensitivity $q$, and the parameter is bounded to a region where $\sigma'(c+q\delta)\ge\kappa>0$, a sufficiently large fixed batch obeys

$$N\ge\frac{2\log(1/\alpha)}{(\kappa qd_T)^2}.$$

The upper and lower bounds match in order for fixed comparison bounds. In the code:

- `btl_sample_bounds` computes the KL expected-count lower bound and the fixed-batch Hoeffding sufficient count under declared bounds;
- `certify_btl_batch` checks a *precommitted*, one-sided test against $\delta\ge\delta_c$;
- `btl_confidence_interval` constructs a *two-sided*, fixed-batch interval for the scalar parameter, truncated to its source set.

The one-sided certification test and two-sided source-set narrowing serve different purposes and have different log-factors. Neither routine supports optional stopping or arbitrary adaptive comparison selection. The paper treats more general sequential comparisons separately; that implementation is retained in the original numerical checks, not exposed as a universal public certification API.

## 6. Finite resource versus actual algorithm

A finite step displacement envelope is a statement of available capacity, not a guarantee that a specific optimizer achieves the envelope. The paper illustrates this distinction using clipped quadratic calibration, which loses certification earlier than an ideal controller with the same maximal speed. A separate experiment allows $m=0,1,2,3$ actual clipped calibration steps per policy step, changing only the number of updates. These are not Bradley–Terry training steps.

See [Paper Results](PAPER_RESULTS.md) for measurements and [Scope](SCOPE.md) for what is not claimed.
