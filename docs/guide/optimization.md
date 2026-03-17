# How To Think About REML and the Solvers

If the optimization stack feels like too many acronyms piled on top of each other, the clean mental model is:

- `IRLS` turns a nonlinear GLM problem into a sequence of weighted least-squares problems.
- `PIRLS` is that same idea after adding penalties.
- when `selection_penalty > 0`, the penalized weighted least-squares subproblem is nonsmooth, so it needs an inner solver
- this repo uses proximal Newton block coordinate descent for that inner solve
- `REML` sits outside all of that and chooses the smoothing parameters

So the stack is not:

`REML versus PIRLS versus BCD`

It is:

`REML outside PIRLS outside a block solver`

## Contents

Use this like a chaptered reference: skim sections `1` to `3` for the big picture, then
jump to the solver layer you care about.

- [1. The story in one ladder](#1-the-story-in-one-ladder)
    - [1.1 OLS](#11-ols)
    - [1.2 WLS](#12-wls)
    - [1.3 IRLS](#13-irls)
    - [1.4 P-IRLS](#14-p-irls)
    - [1.5 P-IRLS plus sparse penalties](#15-p-irls-plus-sparse-penalties)
    - [1.6 REML and fREML](#16-reml-and-freml)
- [2. Solver map](#2-solver-map)
- [3. Why outer and inner solvers are not actually crazy](#3-why-outer-and-inner-solvers-are-not-actually-crazy)
- [4. Why IRLS exists](#4-why-irls-exists)
- [5. What PIRLS means here](#5-what-pirls-means-here)
- [6. Why an inner solver is needed](#6-why-an-inner-solver-is-needed)
    - [6.1 Smooth case: selection_penalty = 0](#61-smooth-case-selection_penalty-0)
    - [6.2 Nonsmooth case: selection_penalty > 0](#62-nonsmooth-case-selection_penalty-0)
- [7. What "proximal Newton BCD" means](#7-what-proximal-newton-bcd-means)
    - [7.1 Proximal](#71-proximal)
    - [7.2 Newton](#72-newton)
    - [7.3 BCD](#73-bcd)
- [8. Why this choice is reasonable](#8-why-this-choice-is-reasonable)
    - [8.1 Why not plain coordinate descent?](#81-why-not-plain-coordinate-descent)
    - [8.2 Why not proximal gradient or FISTA?](#82-why-not-proximal-gradient-or-fista)
    - [8.3 Why not L-BFGS or L-BFGS-B?](#83-why-not-l-bfgs-or-l-bfgs-b)
    - [8.4 Why not ADMM?](#84-why-not-admm)
    - [8.5 Why not a full proximal Newton solve over all coefficients at once?](#85-why-not-a-full-proximal-newton-solve-over-all-coefficients-at-once)
- [9. Performance and discretization](#9-performance-and-discretization)
    - [9.1 Where the time really goes](#91-where-the-time-really-goes)
    - [9.2 Discretization: what it does and what it does not do](#92-discretization-what-it-does-and-what-it-does-not-do)
- [10. Where REML comes in](#10-where-reml-comes-in)
- [11. Why REML has to be outside PIRLS](#11-why-reml-has-to-be-outside-pirls)
- [12. What the REML objective is doing](#12-what-the-reml-objective-is-doing)
    - [12.1 REML gradient and Hessian](#121-reml-gradient-and-hessian)
- [13. The three REML paths in this repo](#13-the-three-reml-paths-in-this-repo)
    - [13.1 Direct REML (Newton)](#131-direct-reml-newton)
    - [13.2 Discrete cached-W REML (fREML)](#132-discrete-cached-w-reml-freml)
    - [13.3 EFS REML (Fellner-Schall for sparse models)](#133-efs-reml-fellner-schall-for-sparse-models)
- [14. Compared to other software](#14-compared-to-other-software)
- [15. Why "REML on top of proximal Newton BCD" is a coherent design](#15-why-reml-on-top-of-proximal-newton-bcd-is-a-coherent-design)
- [16. Reading order in the code](#16-reading-order-in-the-code)
- [17. Appendix: demystifying the implementation-heavy parts](#17-appendix-demystifying-the-implementation-heavy-parts)
    - [17.1 Cached W: what is actually being cached?](#171-cached-w-what-is-actually-being-cached)
    - [17.2 Why the "dead group" logic exists](#172-why-the-dead-group-logic-exists)
    - [17.3 QR: why is there QR code in `metrics.py` if the fitter uses Cholesky/eigendecomposition?](#173-qr-why-is-there-qr-code-in-metricspy-if-the-fitter-uses-choleskyeigendecomposition)
    - [17.4 Kernels: what are they really doing?](#174-kernels-what-are-they-really-doing)
    - [17.5 A good way to read this code without getting lost](#175-a-good-way-to-read-this-code-without-getting-lost)

## 1. The story in one ladder

One good way to make the stack feel intuitive is to view it as a historical build-up rather than a pile of acronyms.

### 1.1 OLS

Start with ordinary least squares: `min_beta ||y - X beta||^2`

```
# ── OLS: one solve, no iteration ──────────────────────────────
β = solve(XᵀX, Xᵀy)            # normal equations: (XᵀX)β = Xᵀy
```

This is the easy world: quadratic objective, one global solve, no iteration.

### 1.2 WLS

Then you realize observations should not contribute equally. WLS introduces a diagonal weight matrix \(\mathbf{W} = \text{diag}(w_1, \ldots, w_n)\) that controls how much each observation influences the fit.

```
# ── WLS: same as OLS but with observation weights ─────────────
# W = diag(w₁, ..., wₙ)       # diagonal weight matrix
β = solve(XᵀWX, XᵀWy)          # weighted normal equations
```

Still linear algebra, still one solve. The only change is that \(\mathbf{X}^\top\mathbf{X}\) becomes \(\mathbf{X}^\top\mathbf{W}\mathbf{X}\) and \(\mathbf{X}^\top\mathbf{y}\) becomes \(\mathbf{X}^\top\mathbf{W}\mathbf{y}\).

**Where do the weights come from?** Three common sources, each with a different interpretation:

| Source | Weight | Meaning |
|--------|--------|---------|
| Inverse-variance | \(w_i = 1 / \text{Var}(y_i)\) | More precise observations contribute more — this is the classical GLS motivation |
| Frequency / exposure | \(w_i = e_i\) (e.g. policy-years) | Observation \(i\) represents \(e_i\) units of exposure, so its expected count scales linearly: \(\text{E}[y_i] = e_i \cdot \lambda_i\) |
| IRLS working weights | \(w_i = e_i \cdot (\mathrm{d}\mu/\mathrm{d}\eta)^2 / V(\mu_i)\) | Local curvature of the GLM log-likelihood — this is the weight that makes the WLS subproblem equivalent to one Fisher-scoring step |

In the IRLS ladder below, the third type is what appears. The weights are not fixed — they are recomputed at each iteration from the current fit. That is the whole point of "iteratively **reweighted**" least squares.

In insurance pricing, the frequency/exposure weights \(e_i\) enter as a multiplier on the IRLS working weights. A policy observed for 3 years contributes 3x the information of a policy observed for 1 year. The product \(w_i = e_i \cdot (\mathrm{d}\mu/\mathrm{d}\eta)^2 / V(\mu_i)\) combines both the exposure and the local curvature into a single diagonal weight.

### 1.3 IRLS

Then you move from Gaussian least squares to a GLM. The likelihood is not quadratic anymore.

The classic trick: near the current fit, replace the GLM with a weighted least-squares problem.

```
# ── IRLS: iteratively reweighted least squares ────────────────
# g       = link function (e.g. log, logit)
# g⁻¹     = inverse link (e.g. exp, expit)
# V(μ)    = variance function (e.g. μ for Poisson, μ(1−μ) for Binomial)

η = X β₀                             # linear predictor
μ = g⁻¹(η)                           # predicted mean on response scale

for t in 1, 2, ...:
    dμ/dη = (g⁻¹)′(η)               # derivative of inverse link
    Wᵢ = wᵢ ⋅ (dμ/dη)² / V(μ)       # working weights (Fisher information)
    zᵢ = ηᵢ + (yᵢ − μᵢ) / (dμ/dη)  # working response (pseudo-data)

    β = solve(XᵀWX, XᵀWz)           # solve one WLS problem

    η = Xβ                            # update linear predictor
    μ = g⁻¹(η)                        # update predicted mean

    if converged(deviance):            # check relative deviance change
        break
```

For canonical links the working weights simplify:

- Poisson/log: `W_i = w_i ⋅ mu_i`
- Binomial/logit: `W_i = w_i ⋅ mu_i ⋅ (1 - mu_i)`

And for Gamma/log (not canonical — the canonical link is inverse, but log is the standard choice in practice):

- Gamma/log: `W_i = w_i` (constant — one of the reasons Gamma/log converges fast)

This is the first place the "outer loop / inner solve" idea appears naturally:

- outer loop updates `W` and `z` (the local quadratic approximation)
- inner solve solves the current WLS problem

### 1.4 P-IRLS

Now add a smooth quadratic penalty `S = sum_j lambda_j S_j` (block-diagonal smoothing matrix):

```
# ── P-IRLS: penalized IRLS ───────────────────────────────────
# S = block-diagonal smoothing penalty (one block per spline term)
# λⱼ controls the roughness penalty for term j

for t in 1, 2, ...:
    W, z = working_quantities(η, μ, y)

    β = solve(XᵀWX + S, XᵀWz)      # penalized normal equations

    η = Xβ
    μ = g⁻¹(η)
    if converged(deviance):
        break
```

Still one `p x p` linear solve per iteration. The penalty `S` regularises the spline coefficients, preventing overfitting without needing to choose knots carefully.

### 1.5 P-IRLS plus sparse penalties

Now add group lasso or sparse group lasso. The subproblem becomes:

```
# ── P-IRLS + group lasso ─────────────────────────────────────
# Objective at each IRLS step:
#   ½ ‖W¹ᐟ²(z − Xβ)‖²                   ← data fit
#   + ½ βᵀSβ                              ← smoothness penalty (quadratic)
#   + λ₁ Σ_g wg ‖βg‖₂                    ← group lasso (nonsmooth!)
```

The last term is not differentiable at zero — it has a kink where entire coefficient groups hit the origin.

This is the point where the subproblem stops being "just linear algebra" because the penalty is nonsmooth at zero.

That is why you need a dedicated inner coefficient solver.

### 1.6 REML and fREML

Finally, ask a higher-level question:

> not just "what coefficients fit best for this penalty?"
> but "what smoothness penalty should I use?"

That creates a new outer optimization problem over the smoothing parameters.

```text
# ── REML: conceptual outer loop ───────────────────────────────
# Goal: choose smoothness parameters λ, not just coefficients β

initialize λ

repeat:
    β̂, μ, W = solve the inner GLM at current λ
        # IRLS / PIRLS gives the fitted coefficients and local geometry

    H = XᵀWX + S(λ)
        # penalized Hessian at the current fit

    obj, grad, hess = evaluate REML criterion from β̂, W, H
        # "is this amount of smoothness too wiggly or too stiff?"

    λ = update_smoothing_parameters(obj, grad, hess)
        # Newton / Fellner-Schall / damped step on log λ

until λ and the REML objective stabilize
```

```text
# ── fREML: same outer goal, fewer expensive refreshes ─────────
# Key idea: if W is nearly unchanged, reuse weighted summaries

initialize λ

repeat:
    β̂, μ, W = run a full inner fit
    cache XᵀWX, XᵀWz, XᵀW1, ΣW, ΣWz

    repeat a few cheap smoothing updates:
        hold W fixed
        change only S(λ)
        re-solve from cached summaries
        update λ

    if λ changed enough that W is no longer trustworthy:
        refresh with another full inner fit

until λ stabilizes
```

!!! note "Conceptually"
    `REML` is "fit coefficients, score the current smoothness, update the smoothness, repeat."
    `fREML` is the same outer idea, but engineered to reduce repeated full data passes
    by reusing weighted summaries when the IRLS geometry has not changed much.

!!! tip "Recommended workflow for spline-based GAM models"
    Use `fit_reml()` with `select=True` on your spline terms. REML estimates a separate smoothing parameter per term, and `select=True` adds a double-penalty decomposition (linear + wiggly subgroups) that lets REML shrink irrelevant terms all the way to zero — mgcv-style automatic term selection without needing `selection_penalty > 0`.

So the rough historical ladder is:

`OLS -> WLS -> IRLS -> P-IRLS -> nonsmooth inner solver -> REML/fREML`

That is the story of how the stack grows. The rest of this guide does not try
to retell every rung in full detail. It just zooms in on the three transitions
that matter for this repo:

- how IRLS turns a GLM into a penalized WLS problem
- why the nonsmooth PIRLS subproblem needs its own inner solver
- why REML has to sit outside the coefficient fit

## 2. Solver map

| Layer | Main question | Solves for | Why it exists | In this repo |
|---|---|---|---|---|
| GLM likelihood | How should the mean depend on the predictors? | `beta`, intercept | The base statistical model | `fit()`, `fit_reml()` |
| IRLS / PIRLS | What quadratic surrogate should represent the GLM right now? | working weights `W`, working response `z` | GLM loss is not quadratic | `src/superglm/solvers/pirls.py`, `src/superglm/solvers/irls_direct.py` |
| Inner coefficient solver | For fixed `W` and `z`, what coefficients solve the penalized WLS problem? | `beta`, intercept | penalties may be nonsmooth | direct solve or proximal Newton BCD |
| REML / fREML | How smooth should each smooth term be? | per-term `lambda_j` | the inner fit does not choose smoothness for you | `src/superglm/reml_optimizer.py` |

!!! tip "If you remember one thing"
    `IRLS` fits coefficients for fixed penalties. `REML` fits the penalties themselves.

## 3. Why outer and inner solvers are not actually crazy

At first glance, nested solvers look like engineering excess. But they usually mean the model has genuinely separable layers.

Here the layers are real:

- the GLM likelihood is nonlinear, so it needs repeated local quadratic updates
- the sparse penalty is nonsmooth, so the weighted least-squares subproblem needs a prox-capable inner solver
- the smoothness parameters are hyperparameters of the penalized fit, so they need their own outer optimization

So the nesting is not arbitrary complexity. It is the optimization structure implied by the model.

## 4. Why IRLS exists

A GLM maximises a log-likelihood that is nonlinear in \(\boldsymbol{\beta}\). IRLS replaces it with a sequence of WLS problems by taking a local Fisher-scoring approximation at each iteration.

The working weights encode the local curvature of the GLM likelihood:

$$
W_i = w_i \cdot \frac{(\mathrm{d}\mu_i / \mathrm{d}\eta_i)^2}{V(\mu_i)}
$$

The working response linearises the residual around the current fit:

$$
z_i = \eta_i + \frac{y_i - \mu_i}{\mathrm{d}\mu_i / \mathrm{d}\eta_i}
$$

The P-IRLS update (with smoothing penalty \(\mathbf{S}\)) solves the augmented system for the intercept \(\alpha\) and coefficients \(\boldsymbol{\beta}\) simultaneously:

$$
\begin{pmatrix} \sum W_i & \mathbf{1}^\top \mathbf{W} \mathbf{X} \\ \mathbf{X}^\top \mathbf{W} \mathbf{1} & \mathbf{X}^\top \mathbf{W} \mathbf{X} + \mathbf{S} \end{pmatrix} \begin{pmatrix} \alpha \\ \boldsymbol{\beta} \end{pmatrix} = \begin{pmatrix} \sum W_i z_i \\ \mathbf{X}^\top \mathbf{W} \mathbf{z} \end{pmatrix}
$$

That is the whole IRLS idea. Each outer iteration does four things:

1. compute \(\mathbf{W}\) and \(\mathbf{z}\) from the current fit
2. solve one penalized weighted least-squares problem
3. update \(\eta\) and \(\mu\)
4. stop when the deviance stabilizes

So the only reason IRLS feels complicated at first is that the weights are
endogenous. Once \(\mathbf{W}\) and \(\mathbf{z}\) are fixed, the subproblem is
just weighted least squares again.

## 5. What PIRLS means here

In the GAM literature, `PIRLS` usually means "penalized IRLS": IRLS plus a smooth quadratic penalty.

In this repo, that basic idea is still true, but there is an extra wrinkle:

- the spline smoothness penalty is quadratic
- the group-lasso style selection penalty is nonsmooth

So "PIRLS" in the code really means:

> build the penalized weighted least-squares subproblem, then hand it to the appropriate inner solver

That is why the repo has two inner paths:

- `fit_irls_direct()` when the subproblem is smooth enough to solve directly
- `_fit_pirls_inner()` when the subproblem includes a nonsmooth group penalty

```
# ── PIRLS dispatch ─────────────────────────────────────────────
# At fit time, the solver is chosen automatically:

if selection_penalty == 0:
    # All penalties are quadratic (smoothing only)
    # → one dense (p+1)×(p+1) solve per IRLS iteration
    result, H⁻¹ = fit_irls_direct(X, y, W, S)

else:
    # Group lasso / sparse group lasso penalty is nonsmooth
    # → proximal Newton BCD inner loop per IRLS iteration
    result = fit_pirls(X, y, W, S, penalty)
```

## 6. Why an inner solver is needed

For fixed \(\mathbf{W}\) and \(\mathbf{z}\), the penalized subproblem is:

$$
\min_{\boldsymbol{\beta}} \; \frac{1}{2} \|\mathbf{W}^{1/2}(\mathbf{z} - \alpha\mathbf{1} - \mathbf{X}\boldsymbol{\beta})\|^2 + \frac{1}{2}\boldsymbol{\beta}^\top \mathbf{S} \boldsymbol{\beta} + P(\boldsymbol{\beta})
$$

Now split by penalty type.

### 6.1 Smooth case: selection_penalty = 0

If there is no group lasso penalty, \(P(\boldsymbol{\beta}) = 0\) and the whole objective is quadratic. The solution is a single linear system:

$$
\boldsymbol{\beta} = (\mathbf{X}^\top \mathbf{W} \mathbf{X} + \mathbf{S})^{-1} \mathbf{X}^\top \mathbf{W} \mathbf{z}
$$

```
# ── Direct IRLS: smooth penalties only ─────────────────────────
# S = block-diagonal smoothing penalty (one block per spline)
# Uses gram-based ops: XᵀWX via per-group gram + cross_gram
#   → O(n_bins ⋅ K²) for discretized groups, not O(n ⋅ p²)

η = α + Xβ
μ = g⁻¹(η)

for t in 1, 2, ...:
    W, z = working_quantities(η, μ, y)

    # Form augmented (p+1)×(p+1) system from cached summaries
    XᵀWX, XᵀW1, XᵀWz = block_xtwx_rhs(groups, W, Wz)

    M = [[ΣWᵢ,   XᵀW1  ],           # intercept row
         [XᵀW1,  XᵀWX + S]]          # coefficient rows
    rhs = [ΣWᵢzᵢ, XᵀWz]

    α, β = solve(M, rhs)              # one eigh or Cholesky

    η = α + Xβ
    μ = g⁻¹(η)
    if converged(deviance): break
```

The repo uses this in `src/superglm/solvers/irls_direct.py`. Typical iteration counts: 5-7 for Poisson/log, even with `select=True` double-penalty terms.

### 6.2 Nonsmooth case: selection_penalty > 0

If \(\lambda_1 > 0\), the penalty includes a group lasso term:

$$
P(\boldsymbol{\beta}) = \lambda_1 \sum_g w_g \|\boldsymbol{\beta}_g\|_2
$$

This is not differentiable at \(\boldsymbol{\beta}_g = \mathbf{0}\) — it has a kink where entire coefficient groups hit the origin.

That means a plain Newton or Cholesky solve is no longer enough. You need a method that can handle:

- exact zeros (entire groups removed)
- whole groups turning on and off
- warm starts across IRLS and lambda paths

That is what the inner proximal solver is for.

## 7. What "proximal Newton BCD" means

This name sounds more complicated than the actual algorithm.

For one PIRLS outer step, the code loops over coefficient groups and does:

1. hold \(\mathbf{W}\) and \(\mathbf{z}\) fixed
2. update the intercept in closed form
3. for each group \(g\), compute the local gradient and block Hessian
4. take a Newton step for that block
5. apply the proximal operator of the group penalty
6. update the residual cheaply

The gradient and Hessian for group \(g\) are:

$$
\nabla_g = -\mathbf{X}_g^\top \mathbf{W} \mathbf{r}, \qquad \mathbf{H}_g = \mathbf{X}_g^\top \mathbf{W} \mathbf{X}_g
$$

where \(\mathbf{r} = \mathbf{z} - \alpha\mathbf{1} - \mathbf{X}\boldsymbol{\beta}\) is the working residual. The Newton direction is \(\mathbf{d}_g = \mathbf{H}_g^{-1} \nabla_g\), and the proximal operator enforces group sparsity:

$$
\boldsymbol{\beta}_g^{\text{new}} = \text{prox}_{\lambda_1 w_g / L_g}\!\left(\boldsymbol{\beta}_g - \mathbf{d}_g\right) = \left(1 - \frac{\lambda_1 w_g / L_g}{\|\boldsymbol{\beta}_g - \mathbf{d}_g\|_2}\right)_+ (\boldsymbol{\beta}_g - \mathbf{d}_g)
$$

where \(L_g\) is the maximum eigenvalue of \(\mathbf{H}_g\) (local Lipschitz constant).

```
# ── Proximal Newton BCD: one PIRLS outer step ─────────────────
# W, z fixed from the current IRLS approximation
# r = z − α1 − Xβ (working residual)

for inner in 1, 2, ..., max_bcd:
    # Intercept update (closed form, unpenalised)
    δ = ΣWᵢrᵢ / ΣWᵢ
    α += δ
    r -= δ

    # BCD cycle: one Newton+prox step per group
    for g in groups:
        βg_old = βg
        ∇g     = −Xgᵀ(W ⊙ r)               # group gradient
        dg     = chol_solve(Hg, ∇g)          # Newton direction: Hg⁻¹∇g
        βg_cand = βg_old − dg                # candidate (no penalty)
        βg_new  = prox(βg_cand, λ₁wg/Lg)    # group soft-threshold

        # Cheap residual update (no full matvec)
        if βg_new ≠ βg_old:
            r -= Xg(βg_new − βg_old)
            βg = βg_new

    if max|β − β_prev| < tol: break
```

The words in the name mean:

### 7.1 Proximal

The penalty is handled by its proximal operator rather than by pretending it is smooth. For ordinary group lasso, the prox is block soft-thresholding. That gives exact zeros.

### 7.2 Newton

The update uses local curvature, not just a gradient. The block Hessian \(\mathbf{H}_g\) makes this much more aggressive than a first-order proximal-gradient method.

### 7.3 BCD

Block coordinate descent. The coefficient vector is partitioned by feature groups, and the algorithm updates one group at a time. That is natural here because the design matrix, penalty, and group activation/deactivation are all group-structured.

See `src/superglm/solvers/pirls.py`.

So the solver is not doing anything exotic. It is just:

> take a good quadratic step for one group, then enforce the nonsmooth penalty exactly

## 8. Why this choice is reasonable

For this repo's problem shape, proximal Newton BCD is a defensible choice.

### 8.1 Why not plain coordinate descent?

Plain coordinate descent is strongest when the penalty is coordinate-separable and the updates are scalar, like glmnet-style lasso.

Your problem is group-structured, so block updates are a more natural fit than scalar updates.

### 8.2 Why not proximal gradient or FISTA?

Those are simpler, but they only use first-order information. For IRLS subproblems, you already have a strong quadratic structure available. Ignoring that curvature usually means more iterations.

### 8.3 Why not L-BFGS or L-BFGS-B?

Those are good smooth optimizers. The moment you want exact group sparsity, they are awkward:

- you either smooth the penalty
- or use subgradient tricks
- or give up exact zeros during the solve

That is a bad trade here because exact zero groups are the point of the penalty.

### 8.4 Why not ADMM?

ADMM is a very useful splitting method, especially when:

- the problem is huge
- the structure is awkward
- there are multiple coupled constraints
- distributed or decomposed computation matters

But it also brings:

- another tuning parameter
- primal and dual residual bookkeeping
- more splitting machinery than the current problem really needs
- weaker appeal when the groups are modest and the local Newton blocks are cheap

For this repo, ADMM would be a plausible alternative, but not an obviously better one.

### 8.5 Why not a full proximal Newton solve over all coefficients at once?

You could try it, but then you lose the cheap group-local structure and make each step heavier. The block version keeps the good part of Newton without turning every update into a large global solve.

## 9. Performance and discretization

### 9.1 Where the time really goes

It is easy to fixate on the scary-looking `O(p^3)` solve cost and miss where the practical runtime often lives.

A useful split is:

1. building weighted summaries like `X'WX`, `X'Wz`, and `X'W1`
2. solving the resulting `p x p` system once those summaries are available

In spline-heavy GLM fitting, especially with repeated PIRLS and REML iterations, the repeated observation-level aggregation work is often the more painful part.

So the real performance question is often:

> how do I avoid redoing expensive data passes and weighted Gram construction over and over?

not just:

> how do I shrink the asymptotic cost of the final dense solve?

### 9.2 Discretization: what it does and what it does not do

Discretization is easy to misunderstand, so it is worth being explicit.

#### 9.2.1 What it does not do

It does **not** fundamentally reduce the coefficient dimension `p`.

So if you end up solving a dense `p x p` system, discretization does not magically change that asymptotic solve into a tiny problem just by itself.

The coefficient-space solve is still a coefficient-space solve.

#### 9.2.2 What it does do

It compresses repeated basis work on the observation side.

For discretized spline groups, the code stores:

- a basis evaluated at `n_bins` support points or bin centers
- an observation-to-bin index

instead of repeatedly treating every observation as having its own distinct basis row.

That means operations like:

- `X'WX`
- `X'W`
- `X'Wz`

can be formed by:

1. aggregating weights by bin
2. doing small dense linear algebra on the binned basis
3. scattering or gathering only when needed

That is exactly what `DiscretizedSSPGroupMatrix` in `src/superglm/group_matrix.py`
is built for. Here `SSP` just means the spline is stored in the repo's
reparameterized penalized form: sparse basis `B`, small dense transform
`R_inv`, and penalty matrix `omega`, rather than one fully materialized dense
design block.

#### 9.2.3 Why this matters in practice

For discretized spline groups, the expensive matrix-building work can move from repeated \(O(n)\) passes on the full basis to something much closer to:

- one \(O(n)\) aggregation by bin (weighted bincounts)
- then \(O(n_{\text{bins}} \cdot K^2)\) dense work in basis space

where \(n_{\text{bins}} \ll n\) (typically 256 bins vs 678k observations) and \(K\) is the local basis size for that group (typically 9-13).

So discretization helps most with:

- weighted Gram construction (\(\mathbf{X}_g^\top \mathbf{W} \mathbf{X}_g\))
- repeated PIRLS updates
- repeated REML objective evaluations
- cached-\(\mathbf{W}\) continuation in the fast REML path

#### 9.2.4 Why the p x p solve is often still fine

In many models here, \(p\) is not enormous. It is often tens or low hundreds, not millions.

That means a Cholesky or eigendecomposition of a \(p \times p\) matrix is not free, but it is often not the part that kills you.

The real win is usually:

- reduce repeated data passes
- reduce repeated basis aggregation
- cache weighted summaries when \(\mathbf{W}\) is unchanged

That is the whole point of the discrete cached-\(\mathbf{W}\) REML path.

## 10. Where REML comes in

Everything above assumes the smoothness penalties are already known.

That is the next big question:

> how large should the smoothing penalties be?

The inner solve does not answer that. It only answers:

> for this value of \(\lambda\), what is the best coefficient vector?

REML answers the higher-level question:

> which \(\lambda\) values make the smooths appropriately flexible without overfitting?

That is why REML must be an outer loop.

## 11. Why REML has to be outside PIRLS

The REML objective depends on the fitted coefficients and the local weighted geometry.

For one candidate set of smoothing parameters, you need:

1. the fitted \(\hat{\boldsymbol{\beta}}(\boldsymbol{\lambda})\)
2. the current IRLS weights \(\mathbf{W}\)
3. the penalized Hessian \(\mathbf{H} = \mathbf{X}^\top \mathbf{W} \mathbf{X} + \mathbf{S}(\boldsymbol{\lambda})\)

You only get those after solving the inner GLM problem.

So the nesting is inevitable:

- inner fit gives you \(\hat{\boldsymbol{\beta}}\), \(\mathbf{W}\), and \(\mathbf{H}\)
- outer REML uses those to score the smoothing parameters
- then proposes new smoothing parameters
- then reruns the inner fit

That is why `fit_reml()` is "refit-heavy". It is solving a different optimization problem on top of the coefficient problem.

## 12. What the REML objective is doing

In this repo, `reml_laml_objective()` in `src/superglm/reml_optimizer.py` evaluates a Laplace-approximate REML criterion (Wood 2011).

This section stays intentionally short. A full REML derivation turns into a paper-sized detour;
the main thing this guide needs is the role each term plays in the outer optimization.

For a known-scale family (\(\phi = 1\), e.g. Poisson):

$$
\mathcal{V}(\boldsymbol{\rho}) = -\ell(\hat{\boldsymbol{\beta}}) + \tfrac{1}{2}\hat{\boldsymbol{\beta}}^\top \mathbf{S} \hat{\boldsymbol{\beta}} + \tfrac{1}{2}\log|\mathbf{H}| - \tfrac{1}{2}\log|\mathbf{S}|_+
$$

where \(\boldsymbol{\rho} = \log\boldsymbol{\lambda}\). For estimated-scale families (Gamma, Tweedie), a \(\phi\)-profiled version replaces the NLL with a scale-free deviance term.

Here \(|\mathbf{S}|_+\) means the pseudo-determinant of \(\mathbf{S}\): the product of its strictly positive eigenvalues only.
The `+` matters because spline penalties usually have a null space, so \(\mathbf{S}\) is often rank-deficient. For example,
constant or linear directions may be left unpenalized, which gives exact zero eigenvalues that should be excluded from the
log-determinant term. In the code, this is the same idea as `log_det_omega_plus` and `cached_logdet_s_plus(...)` in
`src/superglm/reml.py`.

The four terms:

| Term | Meaning |
|------|---------|
| \(-\ell(\hat{\boldsymbol{\beta}})\) | Fit the data (negative log-likelihood) |
| \(\hat{\boldsymbol{\beta}}^\top \mathbf{S} \hat{\boldsymbol{\beta}}\) | Penalize wiggliness |
| \(\log|\mathbf{H}|\) | Account for posterior concentration (Laplace correction) |
| \(-\log|\mathbf{S}|_+\) | Adjust for penalty scale using only penalized directions |

REML is not "just another regularizer". It is an empirical-Bayes criterion for picking the smoothness parameters — it integrates out \(\boldsymbol{\beta}\) via a Laplace approximation and optimizes the resulting marginal likelihood.

### 12.1 REML gradient and Hessian

The partial gradient with respect to \(\rho_j = \log\lambda_j\) at fixed \(\mathbf{W}\) is:

$$
\frac{\partial \mathcal{V}}{\partial \rho_j} = \frac{1}{2}\left(\lambda_j \left(\frac{1}{\phi}\hat{\boldsymbol{\beta}}_g^\top \boldsymbol{\Omega}_g \hat{\boldsymbol{\beta}}_g + \text{tr}(\mathbf{H}^{-1}_{[g,g]} \boldsymbol{\Omega}_g)\right) - r_g\right)
$$

where \(\boldsymbol{\Omega}_g\) is the penalty basis for group \(g\) and \(r_g\) is the penalty rank. Setting the gradient to zero gives the Fellner-Schall fixed-point update:

$$
\lambda_j^{\text{new}} = \frac{r_g}{\frac{1}{\phi}\hat{\boldsymbol{\beta}}_g^\top \boldsymbol{\Omega}_g \hat{\boldsymbol{\beta}}_g + \text{tr}(\mathbf{H}^{-1}_{[g,g]} \boldsymbol{\Omega}_g)}
$$

The numerator is the penalty rank (degrees of freedom available). The denominator balances the current coefficient energy (\(\hat{\boldsymbol{\beta}}\) term) against the posterior uncertainty (trace term). When the signal is strong, the \(\hat{\boldsymbol{\beta}}\) term dominates and \(\lambda\) stays small. When the signal is weak, the trace dominates and \(\lambda\) grows large.

For the Newton path, the Hessian is:

$$
\frac{\partial^2 \mathcal{V}}{\partial \rho_i \partial \rho_j} = -\frac{1}{2}\text{tr}(\mathbf{H}^{-1}\mathbf{S}_i \mathbf{H}^{-1}\mathbf{S}_j) + [\text{diagonal correction}]
$$

with an optional first-order \(W(\boldsymbol{\rho})\) correction that accounts for \(\mathrm{d}\mathbf{W}/\mathrm{d}\boldsymbol{\rho}\) via the implicit function theorem.

## 13. The three REML paths in this repo

### 13.1 Direct REML (Newton)

When `selection_penalty = 0`, the inner problem is smooth. The repo uses direct IRLS plus a damped Newton outer loop on \(\boldsymbol{\rho} = \log\boldsymbol{\lambda}\), with Armijo line search.

That is `optimize_direct_reml()`.

```
# ── Direct REML: Newton on log-λ ──────────────────────────────
# Phase 1: fixed-point warmup (3 iterations)
# Phase 2: Newton with gradient, Hessian, line search

# Bootstrap: fit once at minimal penalty, take one FP step
β̂, H⁻¹ = fit_irls_direct(X, y, W, S=10⁻⁴)
for j in each group:
    ρⱼ = log(rⱼ / (β̂gᵀ Ωg β̂g + tr(H⁻¹[g,g] Ωg)))

for iter in 1, 2, ..., max_reml_iter:
    λ = exp(ρ)
    β̂, H⁻¹, XᵀWX = fit_irls_direct(X, y, W, S(λ))

    if iter ≤ 3:
        # Fixed-point warmup (cheap, no Hessian needed)
        for j in each group:
            ρⱼ = log(rⱼ / (β̂gᵀΩgβ̂g + tr(H⁻¹[g,g] Ωg)))

    else:
        # Newton phase: gradient + Hessian + W(ρ) correction
        ∇ = reml_gradient(β̂, H⁻¹, λ)
        ∇ += w_correction(dW/dη, dβ̂/dρ)     # IFT
        ℋ = reml_hessian(H⁻¹, λ, ∇)
        δ = −solve(ℋ, ∇)                     # Newton step
        δ = clip(δ, −5, 5)                    # trust region

        # Armijo line search: refit at trial ρ
        for step in 1.0, 0.5, 0.25, ...:
            ρ_trial = ρ + step ⋅ δ
            obj_trial = reml_objective(ρ_trial)
            if obj_trial ≤ obj + 10⁻⁴ ⋅ step ⋅ (∇ᵀδ):
                ρ = ρ_trial; break

    if |projected ∇| < tol: break
```

This is the cleanest path mathematically. The \(W(\boldsymbol{\rho})\) correction (enabled after warmup) accounts for the fact that changing \(\boldsymbol{\lambda}\) changes \(\hat{\boldsymbol{\beta}}\), which changes \(\boldsymbol{\mu}\), which changes \(\mathbf{W}\).

In this repo: `src/superglm/reml_optimizer.py::optimize_direct_reml`.

### 13.2 Discrete cached-W REML (fREML)

When `discrete=True`, the expensive part is repeatedly rebuilding \(\mathbf{X}^\top \mathbf{W} \mathbf{X}\) from \(n\) observations.

So the repo caches the weighted summaries after each IRLS convergence and does multiple analytical \(\boldsymbol{\lambda}\) updates before refreshing \(\mathbf{W}\):

```
# ── Cached-W fREML: minimize data passes ──────────────────────
# Key insight: changing S(λ) with fixed W is O(p³),
#              but refreshing W requires O(n) data passes.

for w_iter in 1, 2, ...:
    # Expensive: full IRLS to convergence (data passes through n obs)
    β̂, H⁻¹, XᵀWX = fit_irls_direct(X, y, W, S(λ))
    cache(XᵀWX, XᵀWz, XᵀW1, ΣW, ΣWz)              # save for reuse

    # Cheap inner loop: analytical λ updates at fixed W
    for inner in 1, 2, ..., 30:
        S_new = build_penalty(λ)
        # Re-solve p×p system from cached summaries (no data pass!)
        M = [[ΣW, XᵀW1], [XᵀW1, XᵀWX + S_new]]
        α, β̂ = cholesky_solve(M, [ΣWz, XᵀWz])
        H⁻¹ = invert(XᵀWX + S_new)

        # Fellner-Schall fixed-point update
        for j in each group:
            λⱼ = rⱼ / (β̂gᵀΩgβ̂g + tr(H⁻¹[g,g] Ωg))
            if dead_group(β̂gᵀΩgβ̂g ≪ tr):
                λⱼ → ceiling                  # snap degenerate

        if max|Δρ| < 0.01: break

    if only_dead_groups_changed:
        skip_irls_next_iter = True             # analytical continuation

    if converged: break
```

The mental model is:

- if \(\mathbf{W}\) changes, the local geometry changed and you need fresh weighted summaries (expensive)
- if only \(\mathbf{S}(\boldsymbol{\lambda})\) changes, you can reuse the same weighted geometry and do cheap \(p \times p\) algebra (free)

In this repo: `src/superglm/reml_optimizer.py::optimize_discrete_reml_cached_w`.

### 13.3 EFS REML (Fellner-Schall for sparse models)

When `selection_penalty > 0`, the inner fit uses BCD and the active set can change. The repo uses the generalized Fellner-Schall (EFS) fixed-point update from Wood & Fasiolo (2017) instead of the exact Newton outer loop.

That is `optimize_efs_reml()`.

This Anderson step is internal to the EFS log-\(\lambda\) update. It is not a
separate user-tunable acceleration setting on the PIRLS solver.

```
# ── EFS REML: fixed-point with Anderson acceleration ──────────
# Used when selection_penalty > 0 (group lasso + REML smoothing)

for iter in 1, 2, ...:
    if large_change:
        # Full tier: rebuild design matrix + PIRLS with BCD
        dm = rebuild_design_matrix(λ)          # R_inv depends on λ
        β̂ = fit_pirls(X, y, W, S, penalty)    # BCD inner solver
        XᵀWX = build_xtwx(W)                  # cache for reuse
    else:
        # Cheap tier: re-invert cached XᵀWX + S only (O(p³))
        H⁻¹ = invert(XᵀWX + S(λ))

    # EFS fixed-point update (same formula as Fellner-Schall)
    for j in each group:
        if ‖β̂g‖ < ε: continue                 # L₁ killed this group
        λⱼ = rⱼ / (β̂gᵀΩgβ̂g + tr(H⁻¹[g,g] Ωg))
        λⱼ = clamp(λⱼ, max_log_step=5)

    # Anderson(1) acceleration on log-λ scale
    λ = anderson_accelerate(λ, λ_new)

    if max|Δlog λ| < tol: break
```

So when both sparsity and smoothness selection are active:

- PIRLS + BCD handles coefficient sparsity (which features survive)
- EFS REML handles smoothness (how wiggly the survivors are)

In this repo: `src/superglm/reml_optimizer.py::optimize_efs_reml`.

## 14. Compared to other software

There is no single universal "meta" that every library follows, because libraries are solving slightly different problems.

But for spline-based GAM smoothness selection, `REML` and `fREML` are absolutely mainstream choices, especially in the `mgcv` tradition.

At the same time:

- many libraries do cross-validation instead of REML
- many penalized GLM libraries focus on coefficient regularization and do not try to estimate smoothness parameters at all
- some libraries support splines but stop at fixed lambdas or grid search

So the right way to think about it is:

!!! note "A practical rule"
    `REML` and `fREML` are standard, high-quality choices for spline smoothness estimation. They are not the universal default for every penalized regression library.

That is why comparing directly to something like `glum` is only partly fair:

- `glum` is primarily about penalized GLMs
- this repo is trying to do penalized GLMs and GAM-style smoothing parameter estimation

Those overlap, but they are not the same target.

If you want the broad "story of the field" summary, it is roughly:

- classical GLM software focused on IRLS for coefficient fitting
- spline GAM software added P-IRLS plus GCV, UBRE, and REML for smoothness selection
- large-`n` GAM software added fREML, discretization, and cached weighted summaries to avoid repeated full data passes
- sparse penalized GLM software focused on coordinate or block-descent paths with fixed regularization parameters
- this repo is trying to hybridize those traditions: GAM-style smoothing selection plus sparse-group style coefficient selection

## 15. Why "REML on top of proximal Newton BCD" is a coherent design

The whole stack becomes much easier to reason about once you separate the jobs:

- `IRLS` handles nonlinearity of the GLM likelihood
- proximal Newton BCD handles the nonsmooth sparse penalty inside the penalized WLS subproblem
- `REML` handles smoothness selection outside the coefficient fit

Each layer solves a different problem.

That means the architecture is not overengineered for the sake of it. It is just the natural decomposition of:

1. nonlinear likelihood
2. nonsmooth coefficient penalty
3. unknown smoothness parameters

## 16. Reading order in the code

If you want to understand the implementation from top to bottom, read in this order:

1. `src/superglm/model/fit_ops.py::fit`
2. `src/superglm/solvers/pirls.py::_fit_pirls_inner`
3. `src/superglm/solvers/irls_direct.py::fit_irls_direct`
4. `src/superglm/model/fit_ops.py::fit_reml`
5. `src/superglm/reml_optimizer.py::reml_laml_objective`
6. `src/superglm/reml_optimizer.py::optimize_direct_reml`
7. `src/superglm/reml_optimizer.py::optimize_discrete_reml_cached_w`
8. `src/superglm/reml_optimizer.py::optimize_efs_reml`

!!! tip "Three questions to keep in your head"
    `IRLS`: what weighted least-squares problem is the GLM pretending to be right now?

    `Inner solver`: for that weighted least-squares problem, how do I solve the coefficient optimization with the penalties I actually have?

    `REML`: after I know how to fit coefficients for fixed penalties, how do I choose the smoothness penalties themselves?

## 17. Appendix: demystifying the implementation-heavy parts

This section is about the parts that often feel "agent-generated" or hard to parse when you first read the source.

### 17.1 Cached W: what is actually being cached?

The cached-\(\mathbf{W}\) idea in the fast REML path is simpler than it looks.

For one fixed set of IRLS weights \(\mathbf{W}\), the coefficient problem only depends on the data through a few weighted summaries:

| Cached quantity | Size | Cost to build |
|---|---|---|
| \(\mathbf{X}^\top \mathbf{W} \mathbf{X}\) | \(p \times p\) | \(O(n \cdot p^2)\) or \(O(n_{\text{bins}} \cdot K^2)\) |
| \(\mathbf{X}^\top \mathbf{W} \mathbf{z}\) | \(p\) | \(O(n \cdot p)\) |
| \(\mathbf{X}^\top \mathbf{W} \mathbf{1}\) | \(p\) | \(O(n \cdot p)\) |
| \(\sum W_i\) | scalar | \(O(n)\) |
| \(\sum W_i z_i\) | scalar | \(O(n)\) |

Those are exactly the quantities saved out of `fit_irls_direct()` for reuse.

Once you have them, changing the smoothing penalty only changes the penalty matrix \(\mathbf{S}(\boldsymbol{\lambda})\). It does **not** force you to revisit all \(n\) observations immediately.

So with fixed \(\mathbf{W}\), the augmented system is just:

$$
\begin{pmatrix} \sum W_i & \mathbf{X}^\top \mathbf{W} \mathbf{1} \\ \mathbf{X}^\top \mathbf{W} \mathbf{1} & \mathbf{X}^\top \mathbf{W} \mathbf{X} + \mathbf{S}(\boldsymbol{\lambda}) \end{pmatrix} \begin{pmatrix} \alpha \\ \boldsymbol{\beta} \end{pmatrix} = \begin{pmatrix} \sum W_i z_i \\ \mathbf{X}^\top \mathbf{W} \mathbf{z} \end{pmatrix}
$$

That means you can try new lambdas by doing one \(O(p^3)\) Cholesky solve on cached summaries, instead of rebuilding all weighted summaries from scratch at \(O(n)\) cost.

#### 17.1.1 When is that cache valid?

Only while \(\mathbf{W}\) is effectively unchanged.

That is why the fast REML path alternates between:

- expensive IRLS refreshes, which update \(\mathbf{W}\) (\(O(n)\) per iteration)
- cheap analytical continuation steps, which keep \(\mathbf{W}\) fixed and only change \(\mathbf{S}(\boldsymbol{\lambda})\) (\(O(p^3)\) per step)

The cache is not magic. It is just:

> if the observation-side geometry has not changed, do not recompute it

### 17.2 Why the "dead group" logic exists

In the fast REML path, some groups are detected as effectively dead: their coefficient energy is tiny relative to the trace term.

The practical interpretation is:

- that smooth component is already basically gone
- pushing its lambda higher will not materially change the fitted mean
- so it will not materially change `W`

That is why the code can sometimes keep nudging those lambdas upward without a full IRLS refresh.

It is a pragmatic shortcut, not a new statistical principle.

### 17.3 QR: why is there QR code in `metrics.py` if the fitter uses Cholesky/eigendecomposition?

Because the QR code is mostly for diagnostics and inference, not for the hot fitting path.

The main QR block is in `src/superglm/metrics.py`. It builds an augmented matrix:

$$
\mathbf{A} = \begin{pmatrix} \mathbf{W}^{1/2} \mathbf{X}_a \\ \mathbf{S}^{1/2} \end{pmatrix}
$$

so that \(\mathbf{A}^\top \mathbf{A} = \mathbf{X}^\top \mathbf{W} \mathbf{X} + \mathbf{S}\). Then a reduced QR decomposition gives \(\mathbf{A} = \mathbf{Q}\mathbf{R}\), which implies:

$$
\mathbf{R}^\top \mathbf{R} = \mathbf{A}^\top \mathbf{A} = \mathbf{X}^\top \mathbf{W} \mathbf{X} + \mathbf{S}
$$

So QR is being used as a numerically stable way to factor the penalized normal equations when computing:

- coefficient covariance: \(\hat{\phi} \cdot (\mathbf{X}^\top \mathbf{W} \mathbf{X} + \mathbf{S})^{-1}\)
- leverage and effective degrees of freedom
- inference summaries (SEs, Wood smooth tests)

It is not the main coefficient-fitting method for the core solver.

#### 17.3.1 Why use QR there?

Because for diagnostics, you often care more about stability than raw speed.

The QR path is basically saying:

> I want a reliable factorization of the penalized weighted design, even if the active design is a bit awkward or nearly rank-deficient

There is also pivoted QR in `src/superglm/wood_pvalue.py`, but that is again an inference thing, not a fitting thing. There QR is used to move a smooth term into a numerically stable test space before doing Wood-style significance calculations.

### 17.4 Kernels: what are they really doing?

The kernels in `src/superglm/group_matrix.py` look intimidating because they are low-level and optimized, but conceptually they are mostly just "fast weighted aggregation."

#### 17.4.1 `_fused_bincount_2`

This computes binwise sums of `W` and `Wz` in one pass.

Instead of:

- pass once for `W`
- pass again for `Wz`

it does both together.

#### 17.4.2 `_weighted_bincount_2d`

This aggregates a dense matrix by bin with weights, without materializing the big temporary `W[:, None] * M`.

So it is mostly an allocation-avoidance kernel.

#### 17.4.3 `_csr_weighted_bincount`

Same idea, but for sparse CSR matrices.

It avoids densifying sparse groups just to do weighted aggregation.

#### 17.4.4 `_disc_disc_2d_hist`

This builds a 2D weighted histogram for two discretized groups.

That lets the code compute cross-products between two discretized groups without reconstructing full observation-level matrices.

#### 17.4.5 `_block_xtwx_rhs`

This is the main orchestration routine.

Its job is:

- build `X'WX`
- build `X'W`
- build `X'Wz`

block by block, using the cheapest kernel available for each group-pair type.

So if you want one sentence for the whole kernel layer, it is:

> these kernels are just specialized ways of building weighted cross-products and weighted right-hand sides without materializing expensive temporary matrices

### 17.5 A good way to read this code without getting lost

If you try to understand the kernels line by line first, the code will feel cryptic.

A better order is:

1. Understand the mathematical target:
   - `X'WX`
   - `X'Wz`
   - `X'W1`
2. Understand why discretization replaces observation-level work with bin-level work.
3. Only then read the kernels as implementation details for those targets.

Do the same with the QR code:

1. first understand that it wants `(X'WX + S)^{-1}`
2. then understand the identity `A'A = X'WX + S`
3. only then read the QR factorization

That order makes the source much less opaque.
