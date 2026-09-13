# SuperGLM Roadmap Additions — Numerical and Function-Space Frontier

**Status:** Research dossier / living roadmap addition
**Source technical review:** 2026-09-09
**Repository alignment:** `origin/master` at `7d054022` (C3+C1 merged in PR #379).
**Scope:** Numerical optimization, scalable linear algebra, adaptive function spaces, and structured covariate types.

This reviewed addition extends the [original C1–C14 dossier](2026-09-superglm-feature-roadmap-dossier.md) with C15–C26. The [main roadmap](../ROADMAP.md) records current priorities and delivered evidence. Readiness labels and the computational programme below are research recommendations; promotion requires a separately scoped decision. Example API names are proposals.

This document records research directions rather than implementation commitments. A candidate marked `READY` is ready to **scope or prototype**, not automatically ready to ship. Anything that changes the estimator, uses stochastic approximations, or introduces data-driven active sets needs an explicit inference and certification story before becoming a default SuperGLM path.

The central observation is that C1 improves row-side basis-product work and compressed support contraction, while retaining row likelihood and aggregation passes. SuperGLM is already strong in the large-$n$, moderate-$p$ regime, but adaptive smooths, random effects, functional predictors, graph effects, and richer interactions can push the coefficient dimension $p$ from hundreds into thousands or tens of thousands. At that point, even a perfectly compressed data side can leave Hessian storage and factorization as the new bottleneck.

The current architecture is repeated penalized WLS/Newton inside REML/EFS, with proximal Newton block-coordinate descent for nonsmooth selection penalties. The merged distributional path supports grouped/discrete execution with observed coefficient curvature. Dense coefficient factorizations, retained histories and other whole-fit allocations remain scaling limits; current evidence does not establish bounded whole-fit memory or remove all dependence on $n$. See the [C3+C1 implementation evidence](2026-09-c3-c1-completion-evidence.md) and [discrete performance report](2026-09-discrete-performance-report.md).

The next mathematical generation of SuperGLM should therefore attack three dimensions simultaneously:

$$
\boxed{
\text{rows }n
\quad\times\quad
\text{coefficients }p
\quad\times\quad
\text{complexity of the function space}
}
$$

C1 addresses the row-side cost of this interaction. The candidates below investigate coefficient dimension and representation efficiency, including whether their algorithms can compose with C1.

## Candidate summary

| Candidate | Idea | Main payoff | Reviewed status |
| --- | --- | --- | --- |
| **C15** | Matrix-free large-$p$ EFS/REML backend | Avoid $O(p^2)$ Hessian storage and large direct factorizations | **RESEARCH / NEXT prototype** |
| **C16** | Adaptive hierarchical splines with error control | Spend basis resolution only where it is needed | **RESEARCH — high value** |
| **C17** | Exploit Fisher orthogonality + observed-geometry preconditioning | Reduce cross-predictor coupling in LSS solves | **RESEARCH** |
| **C18** | Multigrid + recycled Krylov subspaces | Reduce iteration growth as spline dimension increases | **RESEARCH**; recycling becomes **READY** after C15 |
| **C19** | qEFS + safeguarded Anderson acceleration | Faster, more principled smoothing iterations | **qEFS: READY to benchmark; Anderson: RESEARCH** |
| **C20** | Progressive exactification / inexact Newton | Avoid full-data/full-precision work when far from convergence | **RESEARCH** |
| **C21** | Sparse-grid / Smolyak interaction smooths | Higher-order smooth interactions without full tensor explosion | **RESEARCH — high predictive upside** |
| **C22** | Trend filtering + graph-structured effects | Local adaptivity, thresholds, structured categorical borrowing | **GraphEffect: READY-ish; trend variants: RESEARCH** |
| **C23** | Functional + compositional predictors | Curves, telemetry, histories, and mixtures as native covariates | **Functional-linear: READY-ish; richer forms: RESEARCH** |
| **C24** | HODLR / HSS Hessian compression | Fast solves/log-dets if hidden hierarchical rank exists | **SPECULATIVE — test first** |
| **C25** | Continuation / branch following | Robustness and branch diagnostics for difficult LSS likelihoods | **RESEARCH** |
| **C26** | Semismooth Newton / primal-dual active-set backend | Fast specialist solver for structured nonsmooth penalties | **RESEARCH / NEXT when C22 requires it** |

The first several are more consequential than adding another generic optimizer.

---

## C15 — Matrix-free large-$p$ SuperGLM backend

### Pitch

Generalize SuperGLM from "assemble a penalized Hessian, then factorize it" to an **operator backend** that can solve Newton/Fisher systems and smoothing trace problems without materializing the full $p\times p$ matrix.

This is not completely new territory inside the repository: the existing [SCOP performance prototype blueprint](../guide/scop-performance-prototype.md#phase-3-controlled-approximation) already lists inexact Newton solves, matrix-free `H @ v`, block preconditioners, and exact fallback as staged targets. C15 is the proposal to turn that idea into a general scalar/LSS backend and add the smoothing-criterion machinery needed for production REML/EFS.

### Core algebra

For a $K$-predictor LSS model, write the penalized coefficient curvature blockwise as

$$
H_{jk}
=
X_j^\top W_{jk}X_k + S_{jk}.
$$

A Newton step does not intrinsically require explicit $H$. It requires the action of $H$ on a vector:

$$
(Hv)_j
=
\sum_k
X_j^\top
\left[
w_{jk}\odot(X_kv_k)
\right]
+
(Sv)_j.
$$

That is a linear operator. The Newton equation

$$
H\delta=-g
$$

can then be solved iteratively:

- **PCG** when the operator is symmetric positive definite;
- **MINRES**, truncated Newton, or a trust-region Krylov method when observed curvature is indefinite;
- direct Cholesky/QR should remain the default for small and moderate systems where it is cheaper and more deterministic.

A fixed coefficient threshold such as "$p>5000$ means matrix-free" should not be hard-coded from intuition. The backend switch should be benchmark-driven and depend on $p$, sparsity/structure, estimated condition number, factorization fill, and the relative cost of one Hessian-vector product.

### Why this is now credible

Zimmermann, Collarin, Wood and Ziel presented a 2026 large-GAM method combining generalized Fellner–Schall, Hutch++ stochastic trace estimation, and PCG while avoiding explicit Hessian formation and Cholesky. Their conference abstract reports fits with more than one million observations and more than 20,000 coefficients in a little over half an hour. It does **not** establish a universal $O(np)$ complexity claim, so SuperGLM should not repeat that stronger statement without a derivation for its own operator. [Zimmermann et al. 2026][1]

### Smoothing traces

EFS/LAML-style methods require quantities of the form

$$
\operatorname{tr}(H^{-1}S_j).
$$

Those can be estimated from repeated solves with $H$ and random probe vectors instead of materializing $H^{-1}$. Hutch++ gives improved query complexity for PSD trace-estimation problems relative to classical Hutchinson estimators. [Meyer et al. 2021][2]

The exact implementation detail matters: $H^{-1}S_j$ is not generally represented as a symmetric PSD matrix even when $H$ is SPD and $S_j$ is PSD. The estimator should therefore be derived for a symmetric congruent operator or otherwise justified explicitly rather than assuming every Hutch++ theorem transfers verbatim.

If an exact-LAML path or diagnostic needs

$$
\log |H|
=
\operatorname{tr}(\log H),
$$

stochastic Lanczos quadrature is a natural matrix-free candidate for SPD $H$. [Ubaru, Chen & Saad 2017][3]

### Preconditioning

A useful hierarchy is

$$
P^{-1}
\approx
\underbrace{D^{-1}}_{\text{term/predictor blocks}}
+
\underbrace{\text{coarse spline correction}}_{\text{multigrid}}
+
\underbrace{\text{spectral correction}}_{\text{optional low-rank/Nyström component}}.
$$

Randomized Nyström preconditioning is established for regularized SPD systems and is worth testing when a small spectral subspace dominates conditioning. [Frangella, Tropp & Udell 2023][4]

### Certification and reproducibility

Stochastic traces introduce a new failure mode for SuperGLM: **optimizer noise can masquerade as smoothing convergence**. A production design should therefore include:

- fixed/common random probes across nearby $\lambda$ evaluations;
- deterministic seeds in reproducibility mode;
- increasing probe counts as the outer gradient shrinks;
- an estimated Monte Carlo error on every stochastic trace;
- a rule that the stochastic error must be small relative to the smoothing-gradient tolerance;
- an exact or higher-accuracy cleanup/certification route when feasible.

The principle should be:

> approximate linear algebra may determine how we reach the solution; it must not silently weaken what `converged` or `certified` means.

### Open question: can C1 and C15 compose?

This is the important research problem.

C1 compresses the row-side algebra through marginal discretization. C15 avoids forming large coefficient-space matrices. A naive matrix-free Hessian-vector product can reintroduce a full $O(np)$ data pass on every Krylov iteration, partly undoing C1's benefit.

The research question is therefore:

$$
\boxed{
\text{Can the marginal-discrete representation apply }v\mapsto Hv
\text{ cheaply without forming all }H_{jk}\text{ blocks?}
}
$$

If yes, SuperGLM could attack both the large-$n$ and large-$p$ barriers in one solver architecture. If no, C1 and C15 should remain separate size-regime backends.

---

## C16 — Adaptive hierarchical splines as statistical mesh refinement

### Pitch

Treat the spline space like an adaptive PDE discretization: start with a coarse nested basis and introduce finer-scale basis functions only where the current fit leaves statistically meaningful unresolved structure.

The goal is not simply "automatic knot selection." The more ambitious target is an adaptive numerical approximation to a **predeclared fine model**, with an explicit bound on the error introduced by not activating the entire basis.

### Local refinement indicator

Let $A$ denote the active basis and $B$ a candidate set of finer child functions. At the current active optimum, let $g_B$ be the score in the omitted directions. Under a local quadratic model, the conditional curvature of $B$ after allowing the active coefficients to readjust is the Schur complement

$$
C_B
=
H_{BB}
-
H_{BA}H_{AA}^{-1}H_{AB}.
$$

When $C_B$ is positive definite, the local quadratic model predicts an improvement of approximately

$$
\Delta_B
\approx
\frac12
g_B^\top C_B^{-1}g_B.
$$

This is a useful **refinement indicator**:

> after the existing fit is allowed to readjust, how much local objective improvement appears to remain in this finer function space?

It is **not**, by itself, an a-posteriori error certificate. Different omitted blocks interact, the likelihood may be nonquadratic, smoothing parameters may reoptimize, and LSS objectives can be locally nonconvex.

### What a real certificate would require

Define a fixed, sufficiently rich space $V_{\max}$ and a fixed penalized objective $F(\beta)$ on that space. The adaptive algorithm should only change how the fixed problem is represented and solved; it must not silently change the underlying penalty each time the mesh is refined.

If $F$ is differentiable and $m$-strongly convex on the identifiable subspace, then a full residual at an adaptively truncated solution $\tilde\beta$ can support bounds of the form

$$
\|\tilde\beta-\beta^\star\|
\le
\frac{\|\nabla F(\tilde\beta)\|}{m},
$$

and

$$
F(\tilde\beta)-F(\beta^\star)
\le
\frac{\|\nabla F(\tilde\beta)\|^2}{2m}.
$$

That is much closer to the PDE notion of a certified discretization error than simply requiring

$$
\sum_B\Delta_B<\varepsilon.
$$

For nonconvex LSS fits, any analogous result would normally be **local to a certified branch/neighbourhood**, not a global guarantee.

A useful SuperGLM-facing target would eventually be something like:

> numerical basis-truncation uncertainty is less than 0.03 posterior standard deviations over the requested prediction domain.

That is a research target, not a currently justified guarantee.

### Existing statistical precedent

Automatic knot selection for GAMs remains active research. Carrizosa, Guerrero and Durbán (2026) extend adaptive-spline ideas to GAMs using an adaptive-ridge approximation and a customized Fellner–Schall scheme, obtaining substantially smaller bases in their examples. That supports the *direction*, but not the stronger hierarchical certification proposed here. [Carrizosa, Guerrero & Durbán 2026][5]

### Why it matters

A conventional large basis allocates resolution everywhere. An adaptive hierarchy can spend coefficients where the function is difficult:

$$
V_0\subset V_1\subset V_2\subset\cdots
$$

while activating only selected descendants. This can plausibly improve both runtime and predictive resolution, especially for localized curvature that ordinary globally regularized splines represent inefficiently.

### Status

**RESEARCH — high value.** The local refinement indicator is reasonable to prototype. The certification layer requires substantially more theory, especially once $\lambda$ is estimated jointly.

---

## C17 — Exploit Fisher orthogonality and precondition observed LSS geometry

### Correction to the original proposal

The initial draft asked whether Tweedie LSS might admit a new near-orthogonal parameterization. That is the wrong starting point.

For the standard Tweedie parameterization, mean coefficients are Fisher-orthogonal to the variance-side block $(\phi,p)$. [Dunn & Smyth 2005][6] This does not justify assuming mutual orthogonality of dispersion and power: retain their coupled expected-information block unless its zeros are established for the chosen likelihood and coordinates. Delong, Lindholm and Wüthrich establish mean-side orthogonality for their joint count/amount likelihood; their fixed-$p$ information matrix does not settle every information term involving $p$. [Delong, Lindholm & Wüthrich 2021][7]

So the interesting SuperGLM problem is **not** "discover Tweedie orthogonal coordinates." It is:

1. exploit known expected-Fisher orthogonality computationally;
2. understand why **observed** LSS curvature can still be strongly coupled;
3. derive or import useful orthogonal coordinates for families that are not already orthogonal.

### Two geometries should be measured separately

Let $H_{\text{obs}}$ denote observed penalized curvature and $I_F$ expected Fisher information. Define block-normalized coupling diagnostics such as

$$
R_{\text{obs}}
=
D_{\text{obs}}^{-1/2}
H_{\text{obs}}
D_{\text{obs}}^{-1/2},
$$

and

$$
R_F
=
D_F^{-1/2}
I_F
D_F^{-1/2},
$$

where $D$ contains the predictor-diagonal blocks. These inverse square roots require every diagonal block to be positive definite. For singular or indefinite blocks, first specify an identifiable subspace or an SPD regularization; the diagnostic then refers to that choice. The full observed matrix may still be indefinite even with SPD diagonal blocks.

Then

$$
\chi_{\text{obs}}=\|R_{\text{obs}}-I\|_2
$$

can be used as a **diagnostic of normalized observed coupling**. A large value suggests that block-diagonal preconditioning may be weak; it does not by itself prove that a block method will converge badly.

For a correctly implemented Tweedie expected-information geometry with separate predictor penalties, $R_F$ should expose the separation of the mean from the variance-side block, without assuming that all variance predictors are mutually uncoupled. If observed curvature remains strongly coupled, that gap itself is useful information.

### Hybrid Fisher / observed Newton

A promising solver policy is:

- use Fisher geometry as a stable block preconditioner or early-iteration metric;
- introduce observed-curvature corrections near the solution;
- use trust-region control when the observed Hessian becomes indefinite;
- record how much of the final step is caused by off-diagonal observed curvature.

This resembles using a statistically meaningful metric rather than a generic diagonal scaling.

### Other families

Classical parameter orthogonality is due to Cox and Reid. [Cox & Reid 1987][8] Recent work continues to derive global or local orthogonal parameterizations for multi-parameter distributions. [Shen, Li & Tong 2025][9] Huet and Prosdocimi (2026) specifically derive orthogonal parameterizations for extreme-value families including the GPD. [Huet & Prosdocimi 2026][10]

That makes the worthwhile family-by-family questions:

- Generalized Gamma: can a useful global or near-global orthogonal map be derived?
- GPD: does the new orthogonal parameterization materially improve additive scale/shape fitting?
- Two-piece families: can location/scale/skewness be separated enough to improve Newton geometry?
- Negative-binomial LSS: what expected cross-information remains under the chosen parameterization and links?

### Schur-complement solves

For a block system

$$
H=
\begin{pmatrix}
A&B\\
B^\top&C
\end{pmatrix},
$$

the Schur complement

$$
S=C-B^\top A^{-1}B
$$

gives $\det H=\det A\,\det S$ when $A$ is invertible. If $H$ is SPD, then $A$ and $S$ are SPD and the ordinary positive-curvature identity is

$
\log\det H
=
\log\det A+\log\det S.
$

For nonsingular indefinite systems, use log absolute determinants and retain determinant signs separately.

If the mean predictor is very large and scale/shape predictors are smaller, this can be substantially more attractive than treating every coefficient identically. Fisher orthogonality or near-orthogonality can make the cross-block $B$ smaller and strengthen block preconditioners.

### Status

**RESEARCH.** The potentially original contribution is an automatic expected-vs-observed geometry controller for distributional GAMs, not rediscovering Tweedie orthogonality.

---

## C18 — Multigrid and Krylov recycling

### Multigrid

Spline spaces naturally possess resolution levels. A nested sequence such as

$$
V_0\subset V_1\subset\cdots\subset V_L
$$

can support the usual multigrid pattern:

$$
\text{fine relaxation}
\rightarrow
\text{restrict residual}
\rightarrow
\text{coarse solve}
\rightarrow
\text{prolongate correction}.
$$

Multigrid-preconditioned CG has already been developed for tensor-product spline smoothing, with grid-independent convergence under the assumptions studied there. [Takacs & Takacs 2019/2021][11]

The important caveat is that those results do **not** automatically transfer to changing IRLS/LSS weights, arbitrary constraints, or indefinite observed curvature. SuperGLM must benchmark whether the coarse operator remains a good approximation as $W$ changes.

If C16 exists, one hierarchical representation could support three roles:

$$
\boxed{
\text{representation}
+
\text{adaptive refinement}
+
\text{linear-solver preconditioning}
}
$$

That architectural reuse is attractive.

### Krylov recycling

SuperGLM repeatedly solves related systems:

$$
H_t x_t=b_t,
\qquad
H_{t+1}\approx H_t.
$$

IRLS weights and smoothing parameters often move gradually. Recycling methods retain selected subspaces from previous Krylov solves and use them in later systems. Parks et al. developed this specifically for sequences in which both the matrix and right-hand side may change. [Parks et al. 2006][12]

Candidate recycled directions include:

- difficult Ritz vectors;
- weakly identified null-space directions;
- strongly correlated smooth combinations;
- slow cross-predictor modes in LSS.

### Status

**Multigrid: RESEARCH.**
**Krylov recycling: READY to prototype once C15 provides a general iterative operator backend.**

---

## C19 — qEFS and safeguarded fixed-point acceleration

EFS defines an outer fixed-point map on log smoothing parameters:

$$
\rho_{k+1}=T(\rho_k),
\qquad
\rho=\log\lambda.
$$

That makes modern fixed-point acceleration relevant.

### qEFS / structured secant updates

The 2026 structured-secant work of Krause, Borst and van Rij develops a limited-memory quasi-Newton variant of EFS for general smooth models, reducing the need for expensive higher-order likelihood derivatives while retaining an EFS target under its stated conditions. [Krause, Borst & van Rij 2026][13]

This deserves a first-class roadmap item rather than being mentioned only as competitor context.

**Status: READY to reproduce and benchmark clean-room from the paper.**

### Anderson acceleration

Given residuals

$$
r_i=T(\rho_i)-\rho_i,
$$

Anderson acceleration forms a small least-squares extrapolation from recent fixed-point iterates. There is theory showing improved local rates for linearly convergent fixed-point iterations under suitable assumptions. [Evans et al. 2020][14]

Raw Anderson is not safe enough for SuperGLM. A candidate design should include:

- small memory, e.g. $m=3$ to $8$;
- conditioning checks on the residual-history least-squares problem;
- restart when the history becomes nearly singular;
- objective/residual acceptance;
- fallback to the unaccelerated EFS step;
- no claim of global convergence merely because the local fixed-point Jacobian looks stable.

Safeguarded Anderson schemes using residual tests, conditioning controls and memory restarts are established in optimization. [Garstka, Cannon & Goulart 2022][15]

**Status: RESEARCH.** Cheap to prototype, but not a drop-in theorem for EFS.

### Local Jacobian diagnostics

Near a candidate fixed point,

$$
\delta_{k+1}
\approx
J_T(\rho^\star)\delta_k.
$$

The spectral radius of $J_T$ is a useful **local diagnostic**:

- $\rho(J_T)<1$: locally attracting fixed-point iteration;
- eigenvalues close to $1$: slow/near-neutral smoothing directions;
- $\rho(J_T)>1$: locally unstable plain fixed-point iteration.

This can inform whether acceleration or a trust-region/Newton rescue is sensible. It is not a global convergence certificate.

---

## C20 — Progressive exactification and inexact Newton

### Pitch

Do not spend full-data, full-precision work when the current iterate is obviously far from the final solution.

There are two independent approximation budgets.

### Adaptive sample size

Early Newton/trust-region iterations can use a controlled subset and increase the sample size as stationarity improves. Adaptive-sample trust-region methods have global-convergence theory for finite-sum optimization and can eventually transition to full-batch evaluation. [Mohr & Stein 2019][16]

For insurance data, uniform subsampling is usually a poor default. Candidate sampling should preserve important information through stratification or importance weights based on some combination of:

- exposure / case weight;
- response or tail region;
- approximate leverage;
- curvature contribution;
- important portfolio segments.

### Adaptive likelihood precision

For families involving a series, quadrature, CDF inversion, root solve, or expensive special functions, allow a numerical tolerance

$$
\varepsilon_{\ell,k}
$$

to tighten with optimizer progress.

For example, **illustratively rather than as fixed defaults**:

$$
10^{-4}
\rightarrow
10^{-8}
\rightarrow
10^{-12}.
$$

The forcing rule should be tied to gradient/step accuracy, not hard-coded decimal stages.

### What "exact final cleanup" does and does not guarantee

The principle should be:

> approximations may guide the path, but final stationarity and inference are evaluated on the full-data/full-precision objective.

For a convex problem, a sufficiently accurate final solve can recover the same unique optimizer as an exact-from-start path.

For a nonconvex LSS model, this statement is weaker: an approximate path can enter a different basin and converge to a different legitimate stationary branch. Full-data cleanup preserves the **objective definition**, not necessarily the same branch that an exact path from another initialization would have reached.

That caveat should be explicit.

### Status

**RESEARCH.** Particularly attractive for expensive LSS likelihoods and future transformation/copula models.

---

## C21 — Higher-dimensional smooth interactions without full tensor explosion

Higher-dimensional tensor products eventually suffer basis growth resembling

$$
M^d
$$

for $d$ dimensions and $M$ marginal resolution.

### Sparse-grid / Smolyak smooths

A sparse-grid hierarchy retains tensor levels satisfying a restriction such as

$$
\ell_1+\cdots+\ell_d\le L
$$

instead of all combinations in the full tensor grid.

Under the mixed-smoothness conditions for which sparse grids are designed, the number of degrees of freedom can scale like

$$
O(N(\log N)^{d-1})
$$

rather than $O(N^d)$ for a corresponding full grid. Bungartz and Griebel give the classical approximation theory and make the required mixed-regularity assumptions explicit. [Bungartz & Griebel 2004][17]

The attraction for SuperGLM is that the model can remain **linear in coefficients**:

$$
\eta=X\beta.
$$

That preserves the *possibility* of:

- quadratic smoothness penalties;
- REML/EFS smoothing;
- covariance calculations;
- matrix-free Hessian products;
- compatibility with a discrete assembly strategy.

Those properties are **not automatic**. Rank constraints, centering, anisotropic penalties, tensor hierarchy bookkeeping, and C1-compatible cross-products still need to be derived.

A possible user-facing term is

```text
SparseTensor(x1, x2, x3, x4, level=..., adaptive=True)
```

with anisotropic refinement and strong heredity.

C16 and C21 could eventually share a refinement engine.

### Failure mode

Sparse grids are strongest for functions with suitable mixed regularity. Non-axis-aligned ridges, sharp diagonal boundaries, and other adversarial structures can defeat the efficiency advantage. Benchmarks must therefore include such cases rather than only smooth separable test functions.

### Factorized spline interactions

Rügamer (2024) factorizes higher-order tensor-product spline interactions so that all nonlinear interactions can be represented at costs proportional to a model without explicit interaction enumeration under the proposed factorization. [Rügamer 2024][18]

That is highly interesting for prediction and structure discovery. The trade-off is different from sparse grids:

- factorization introduces nonconvex optimization;
- latent factors create non-identifiability/rotation issues;
- ordinary GAM effect inference does not transfer automatically.

A sensible division is therefore:

$$
\boxed{\text{sparse grid} \approx \text{candidate inferential model}}
$$

and

$$
\boxed{\text{factorized spline interactions} \approx \text{predictive / discovery model}}
$$

The factorized model could screen interactions, followed by an inferential refit using ordinary or sparse-grid terms.

### Status

**RESEARCH — high predictive upside.**

---

## C22 — Locally adaptive and graph-structured effects

This candidate should be split conceptually because its two halves have different inference stories.

### C22a — Trend-filtering smooths

A conventional quadratic spline penalty such as

$$
\lambda\int[f''(x)]^2\,dx
$$

encourages globally smooth changes in curvature.

Trend filtering instead uses an $\ell_1$ penalty on discrete derivatives, schematically

$$
\lambda\|D^{k+1}f\|_1.
$$

The result is an adaptively piecewise-polynomial effect:

- $k=0$: piecewise constant;
- $k=1$: piecewise linear;
- $k=2$: piecewise quadratic.

Additive trend filtering has minimax-optimal rates over the corresponding bounded-variation additive classes, and the literature emphasizes its local adaptivity relative to smoothing splines. [Sadhanala & Tibshirani 2019][19]

For pricing, a particularly interesting structured variant is monotone trend filtering:

$$
f'(x)\ge0,
\qquad
\|D^2f\|_1\ \text{penalized}.
$$

That can produce a monotone piecewise-linear tariff curve whose slope changes are data-adaptive.

#### Inference caveat

This is a major difference from ordinary penalized splines. The $\ell_1$ penalty creates a data-dependent active set / knot pattern. A standard inverse-Hessian Wald covariance conditional on that active set is **not automatically valid post-selection inference**.

Before trend filtering becomes an "inferential GAM term," SuperGLM needs one of:

- explicit post-selection/selective inference;
- sample splitting;
- bootstrap or other calibrated uncertainty;
- a prediction-only label;
- or a second-stage smooth refit with clearly stated inferential semantics.

So `TrendFilter` should not be marked simply READY for the same inference contract as a normal spline.

### C22b — Graph effects

A structured categorical variable can be represented by a graph $G$.

For a quadratic graph smoother,

$$
\beta^\top L_G\beta
=
\sum_{(u,v)\in E}
w_{uv}(\beta_u-\beta_v)^2,
$$

where $L_G$ is the graph Laplacian. This is a quadratic penalty and fits much more naturally into SuperGLM's existing penalized-linear-algebra and REML machinery.

Examples include:

- manufacturer $\rightarrow$ model $\rightarrow$ variant taxonomies;
- geographical adjacency;
- occupation or industry taxonomies;
- repair-component hierarchies.

A stronger locally adaptive version uses graph trend filtering:

$$
\lambda
\sum_{(u,v)\in E}
w_{uv}|\beta_u-\beta_v|.
$$

Connected levels can then fuse exactly while genuine boundaries remain. Graph trend filtering was designed to provide local adaptivity beyond $\ell_2$ graph smoothers. [Wang et al. 2016][20]

A possible API is

```text
GraphEffect(vehicle_model, graph=vehicle_taxonomy)
GraphTrend(area, graph=adjacency)
```

`GraphEffect` is the lower-risk feature because it retains a quadratic penalty. `GraphTrend` inherits the nonsmooth/post-selection issues of trend filtering.

### Status

- **Quadratic `GraphEffect`: READY-ish to scope.**
- **`TrendFilter` / `GraphTrend`: RESEARCH for inferential use; potentially earlier for prediction/structure discovery.**

---

## C23 — Functional and compositional predictors

### Functional predictors

A functional predictor is a curve $x_i(t)$ rather than a scalar.

Examples:

- journey telemetry;
- monthly claim-development history;
- warranty sensor trajectories;
- utilization/load curves;
- temporal exposure profiles.

A functional linear term has the form

$$
\eta_i
=
\int x_i(t)\beta(t)\,dt.
$$

After basis expansion and numerical integration this is still linear in the coefficient vector, so it is a natural fit for SuperGLM's existing penalty machinery.

A richer functional GAM uses

$$
\eta_i
=
\int F(x_i(t),t)\,dt,
$$

with a bivariate smooth $F$. Functional generalized additive model theory already exists. [McLean et al. 2014][21]

A practical implementation must specify:

- common versus irregular observation grids;
- quadrature rules;
- missing segments;
- measurement error;
- centering/identifiability;
- whether the functional design can be discretized or cached efficiently.

**Functional-linear terms are READY-ish to scope. Full FGAM terms remain RESEARCH.**

### Compositional predictors

A composition lies on a simplex rather than ordinary Euclidean space. For strictly positive components,

$$
x_j>0,
\qquad
\sum_jx_j=1,
$$

log-ratio/Bayes-space geometry is the natural starting point; treating all parts as independent unconstrained regressors is generally inappropriate.

Generalized functional additive mixed-model work has already incorporated finite and functional compositional covariates using geometry-respecting transformations and constrained basis representations. [Eckardt, Mateu & Greven 2024][22]

A production feature must address zeros explicitly. Many real insurance compositions contain structural or sampling zeros, so a strict $x_j>0$ API without a declared zero-handling model would be insufficient.

**Status: RESEARCH/READY depending on scope.** A simple finite-composition linear effect is much closer than a nonlinear functional-composition term.

---

## C24 — Test whether SuperGLM Hessians have hierarchical low rank

This remains speculative and should stay cheap until evidence appears.

HODLR/HSS methods exploit matrices whose off-diagonal blocks have low numerical rank recursively. PDE inverse problems provide examples where this can replace dense cubic operations with near-log-linear hierarchical operations. [Hartland et al. 2023][23]

That is precedent, not evidence that a GAM/LSS Hessian has the same structure.

### Experiment first

Order coefficients by

$$
\text{predictor}
\rightarrow
\text{term}
\rightarrow
\text{local spline support}
$$

and recursively measure off-diagonal numerical rank at tolerances such as

$$
10^{-3},\ 10^{-5},\ 10^{-7},\ 10^{-9}.
$$

Test across:

- scalar GAMs;
- dense LSS;
- tensor interactions;
- graph penalties;
- random-effect-heavy systems.

If rank stays roughly bounded or grows slowly with block size, investigate HODLR/HSS arithmetic.

If rank grows proportionally with block size, kill the idea.

### Status

**SPECULATIVE — test first.**

---

## C25 — Continuation and branch following for difficult distributional likelihoods

For GPD, generalized Gamma, coupled LSS and future copula models, the likelihood surface can become difficult enough that initialization and path dependence matter.

Introduce a homotopy parameter

$$
\tau\in[0,1],
$$

with an easy problem at $\tau=0$ and the requested model at $\tau=1$.

For example,

$$
\eta_{\text{shape},\tau}
=
\tau\,\eta_{\text{shape}},
$$

or gradually introduce interaction, dependence, tail-shape, or constraint complexity.

Then solve

$$
F(\beta,\lambda,\tau)=0
$$

with predictor-corrector continuation. Pseudo-arclength continuation can follow a branch through a fold where naive parameter continuation fails.

### What this buys statistically

Continuation can expose multiple stationary branches that ordinary restart logic may hide.

If two distinct locally stable branches are found, SuperGLM has positive evidence that uniqueness is not established and can report that explicitly.

The converse is important:

> following one branch successfully does **not** prove that no other branch exists.

Continuation is a branch-exploration and robustness tool, not an automatic global uniqueness certificate.

### Status

**RESEARCH.** More valuable for robustness/certification than raw speed.

---

## C26 — Semismooth Newton / primal-dual active-set backend

C22 introduces structured nonsmooth penalties for which the current proximal Newton BCD path may not be the best specialist solver.

Semismooth Newton and augmented-Lagrangian/primal-dual methods exploit generalized derivatives and "second-order sparsity" in structured $\ell_1$ problems. There is direct precedent for sparse-Hessian semismooth Newton augmented-Lagrangian methods for general $\ell_1$ trend filtering. [Liu & Zhang 2023][24]

This does **not** imply that one semismooth Newton implementation should replace BCD everywhere.

A sensible solver policy is:

- ordinary smooth quadratic penalties $\rightarrow$ direct/iterative Newton;
- sparse-group selection where BCD is strong $\rightarrow$ keep proximal Newton BCD;
- fused/trend/graph-trend penalties with favorable generalized-Jacobian structure $\rightarrow$ benchmark SSN/augmented-Lagrangian or primal-dual active-set methods.

### Requirements

- exact KKT residuals;
- warm starts along $\lambda$ paths;
- active-set diagnostics;
- deterministic fallback;
- no covariance claim until the nonsmooth/post-selection inference semantics are defined.

### Status

**RESEARCH / NEXT when C22 moves from model idea to implementation.**

---

## Strengthen C3 — Adaptive regularized Newton controller

This is prospective controller research beyond the [merged C3 follow-through](2026-09-pragmatic-convergence.md). It does not replace the delivered convergence evidence or automatically reopen resolved stress cases. Scope a reproduced remaining limitation before promoting regularization, trust-region/cubic steps or further curvature reuse.

### Geometry-dependent step policy

When ordinary Newton geometry is well behaved, use Newton.

When curvature is poorly conditioned but essentially positive, use a regularized system such as

$$
H+\gamma I.
$$

When meaningful negative curvature is present, use a trust-region or cubic-regularized model:

$$
m(s)
=
g^\top s
+
\frac12s^\top Hs
+
\frac{\sigma}{3}\|s\|^3.
$$

Recent cubic-regularization work includes scalable subspace methods, which are relevant as inspiration for large coefficient systems. [Cartis, Shao & Tansley 2025][25]

### Lazy curvature reuse

If curvature changes little between iterations, do not automatically rebuild/factorize it. "Lazy Hessian" methods have modern convergence theory in other second-order settings. [Doikov et al. 2023][26]

This is inspiration rather than a drop-in theorem for IRLS/LSS. SuperGLM's weights, constraints and penalties create a different changing-curvature problem, so the reuse criterion must be derived and benchmarked in-model.

The existing fREML philosophy already demonstrates the broader idea: reuse expensive geometry when the statistical quantities that generated it have barely moved.

---

## Revised computational programme

The roadmap is stronger if C1 is not described as the only flagship. There are three complementary fronts.

### Front A — Compress observations

**C1: discrete marginal multi-predictor assembly**

Target: huge $n$ with moderate $p$.

### Front B — Stop materializing high-dimensional algebra

**C15 + C18**, supported by C3/C19.

Target: large $p$ through operator Hessians, iterative solves, stochastic traces, multilevel preconditioning, and recycled subspaces.

### Front C — Stop wasting coefficients

**C16 + C21 + C22**, with C26 as a solver enabler.

Target: improve representational efficiency and predictive performance through local adaptivity, sparse higher-order interactions, and graph structure.

These fronts should share infrastructure where the mathematics permits, but they should not be forced into one backend if the complexity regimes differ.

## Research questions worth attacking directly

### R1 — Can C1 and C15 be composed?

Can a marginal-discrete design provide efficient Hessian-vector products for coupled LSS models without either:

- reconstructing $O(p^2)$ cross-product blocks, or
- reverting to a full $O(np)$ row pass at every Krylov iteration?

A positive answer would attack large $n$ and large $p$ simultaneously.

### R2 — Can adaptive spline truncation be certified relative to a fixed virtual basis?

Predeclare a large $V_{\max}$, solve only an adaptive subset, and bound the difference between the adaptive numerical solution and the full-basis penalized optimum in units relevant to inference.

This is much stronger than automatic knot placement.

### R3 — Can expected Fisher geometry systematically precondition observed LSS curvature?

Tweedie supplies a known orthogonal test case. The research contribution is to turn expected-vs-observed geometry into an automatic solver diagnostic/controller and extend useful parameterizations to families such as GPD and generalized Gamma.

## Numerical moonshot

The computational moonshot is:

$$
\boxed{
\text{large }n
+
\text{large }p
+
\text{multiple predictors}
+
\text{automatic smoothing}
+
\text{full likelihood inference}
}
$$

with exact or explicitly quantified approximation error.

The predictive moonshot is:

$$
\boxed{
\text{adaptive hierarchical main effects}
+
\text{sparse-grid interactions}
+
\text{graph-structured factors}
+
\text{distributional likelihood}
}
$$

That remains recognizably additive and interpretable while acquiring several capabilities that make tree models strong: local adaptivity, sharp changes, structured categorical sharing, and richer interactions.

---

## References

[1]: https://trr391.tu-dortmund.de/events/conferences/2026/abstracts/ "Zimmermann, Collarin, Wood & Ziel (2026), Scalable estimation of large generalized additive models: Extended Fellner-Schall via Hutch++ and conjugate gradients"
[2]: https://arxiv.org/abs/2010.09649 "Meyer et al. (2021), Hutch++: Optimal Stochastic Trace Estimation"
[3]: https://epubs.siam.org/doi/10.1137/16M1104974 "Ubaru, Chen & Saad (2017), Fast Estimation of tr(f(A)) via Stochastic Lanczos Quadrature"
[4]: https://epubs.siam.org/doi/10.1137/21M1466244 "Frangella, Tropp & Udell (2023), Randomized Nyström Preconditioning"
[5]: https://arxiv.org/abs/2607.21083 "Carrizosa, Guerrero & Durbán (2026), Automatic knot selection in smooth additive models"
[6]: https://gksmyth.github.io/pubs/tweediepdf-series-preprint.pdf "Dunn & Smyth (2005), Series evaluation of Tweedie exponential dispersion model densities"
[7]: https://link.springer.com/article/10.1007/s13385-021-00264-3 "Delong, Lindholm & Wüthrich (2021), Making Tweedie's compound Poisson model more accessible"
[8]: https://doi.org/10.1111/j.2517-6161.1987.tb01422.x "Cox & Reid (1987), Parameter Orthogonality and Approximate Conditional Inference"
[9]: https://arxiv.org/abs/2501.08093 "Shen, Li & Tong (2025), A note on parameter orthogonality for multi-parameter distributions"
[10]: https://arxiv.org/abs/2602.16283 "Huet & Prosdocimi (2026), Orthogonal parametrisations of Extreme-Value distributions"
[11]: https://arxiv.org/abs/1901.00654 "Takacs & Takacs, A Multigrid Preconditioner for Tensor Product Spline Smoothing"
[12]: https://epubs.siam.org/doi/10.1137/040607277 "Parks et al. (2006), Recycling Krylov Subspaces for Sequences of Linear Systems"
[13]: https://arxiv.org/abs/2606.26804 "Krause, Borst & van Rij (2026), Structured Secant Methods to Select Smoothing Parameters For General Smooth Models"
[14]: https://epubs.siam.org/doi/10.1137/19M1245384 "Evans et al., A Proof That Anderson Acceleration Improves the Convergence Rate in Linearly Converging Fixed-Point Methods"
[15]: https://doi.org/10.23919/ECC55457.2022.9838320 "Garstka, Cannon & Goulart (2022), Safeguarded Anderson acceleration for parametric nonexpansive operators"
[16]: https://arxiv.org/abs/1910.03294 "Mohr & Stein (2019), An Adaptive Sample Size Trust-Region Method for Finite-Sum Minimization"
[17]: https://doi.org/10.1017/S0962492904000182 "Bungartz & Griebel (2004), Sparse grids"
[18]: https://proceedings.mlr.press/v238/ruegamer24a.html "Rügamer (2024), Scalable Higher-Order Tensor Product Spline Models"
[19]: https://doi.org/10.1214/19-AOS1833 "Sadhanala & Tibshirani (2019), Additive Models with Trend Filtering"
[20]: https://www.jmlr.org/papers/v17/15-147.html "Wang, Sharpnack, Smola & Tibshirani (2016), Trend Filtering on Graphs"
[21]: https://pmc.ncbi.nlm.nih.gov/articles/PMC3982924/ "McLean et al. (2014), Functional Generalized Additive Models"
[22]: https://doi.org/10.1093/jrsssc/qlae016 "Eckardt, Mateu & Greven (2024), Generalized functional additive mixed models with (functional) compositional covariates"
[23]: https://arxiv.org/abs/2301.03644 "Hartland et al. (2023), Hierarchical off-diagonal low-rank approximation of Hessians in inverse problems, with application to ice sheet model initialization"
[24]: https://www.crwweb.net/info/1090/6158.htm "Liu & Zhang (2023), Sparse Hessian based semismooth Newton augmented Lagrangian algorithm for general L1 trend filtering, Pacific Journal of Optimization 19(2):187–204"
[25]: https://arxiv.org/abs/2501.09734 "Cartis, Shao & Tansley (2025), Random Subspace Cubic-Regularization Methods, with Applications to Low-Rank Functions"
[26]: https://proceedings.mlr.press/v202/doikov23a.html "Doikov et al. (2023), Second-Order Optimization with Lazy Hessians"
