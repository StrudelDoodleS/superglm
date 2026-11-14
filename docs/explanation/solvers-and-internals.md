# How To Think About REML and the Solvers

If the optimization stack feels like too many acronyms piled on top of each other, the clean mental model is:

- `IRLS` turns a nonlinear GLM problem into a sequence of weighted least-squares problems.
- `PIRLS` is that same idea after adding penalties.
- when `selection_penalty > 0`, the penalized weighted least-squares subproblem is nonsmooth, so it needs an inner solver
- this repo uses proximal Newton block coordinate descent for that inner solve
- when a term carries a shape rule (monotone, convex, a cap), the weighted least-squares subproblem gains linear inequality constraints, so each step is a small quadratic program
- `REML` sits outside all of that and chooses the smoothing parameters

So the stack is not:

`REML versus PIRLS versus BCD`

It is:

`REML outside PIRLS outside a block solver`

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

Then you realize observations should not contribute equally. WLS introduces a diagonal weight matrix $\mathbf{W} = \text{diag}(w_1, \ldots, w_n)$ that controls how much each observation influences the fit.

```
# ── WLS: same as OLS but with observation weights ─────────────
# W = diag(w₁, ..., wₙ)       # diagonal weight matrix
β = solve(XᵀWX, XᵀWy)          # weighted normal equations
```

Still linear algebra, still one solve. The only change is that $\mathbf{X}^\top\mathbf{X}$ becomes $\mathbf{X}^\top\mathbf{W}\mathbf{X}$ and $\mathbf{X}^\top\mathbf{y}$ becomes $\mathbf{X}^\top\mathbf{W}\mathbf{y}$.

**Where do the weights come from?** Weighted-model notation is overloaded.
Three common sources have different interpretations:

| Source | Weight | Meaning |
|--------|--------|---------|
| Inverse-variance | $w_i = 1 / \text{Var}(y_i)$ | More precise observations contribute more — this is the classical GLS motivation |
| Case / frequency | $a_i = e_i$, for a Poisson *rate* row | Observation $i$ contributes $e_i$ likelihood units; for fixed feature geometry, integer $a_i$ is likelihood-level row replication, and $\text{E}[y_i]=\lambda_i$ remains a rate |
| IRLS working weights | $W_i = a_i \cdot (\mathrm{d}\mu/\mathrm{d}\eta)^2 / V(\mu_i)$ | Local curvature of the GLM log-likelihood — this is the weight that makes the WLS subproblem equivalent to one Fisher-scoring step |

In the IRLS ladder below, the third type is what appears. The weights are not fixed — they are recomputed at each iteration from the current fit. That is the whole point of "iteratively **reweighted**" least squares.

For the Poisson rate encoding, exposure $a_i=e_i$ multiplies the IRLS
curvature, so a three-year rate row contributes three times the information of
a one-year row. It does not multiply the conditional mean automatically. For
a raw count response, put `log(exposure)` in the offset so that
$\text{E}[\text{count}_i]=e_i\lambda_i$. Whether `sample_weight` is the inverse-variance row above or a replication
count is declared per model by `weight_semantics`, and defaults to the former;
see [Families & Dispersion](families-and-weights.md#weight-semantics).

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
# Key idea: hold W fixed within a step, and try new λ on
# weighted summaries instead of on the rows

initialize λ

repeat:
    β̂, W = one IRLS step at the current λ
    cache the weighted summaries of the rows at this W

    take one Newton step on log λ for the REML criterion
        of the working model that W defines
    line-search that step on the cached summaries:
        each trial λ is a small p x p solve, no pass over the rows

until λ and the REML objective stabilize
```

```{admonition} Conceptually
:class: note
`REML` is "fit coefficients, score the current smoothness, update the smoothness, repeat."
`fREML` is the same outer idea, but engineered to reduce repeated full data passes:
it updates the weights and the smoothing parameters together, one step each,
and tries candidate smoothing parameters on cached weighted summaries.
This is Wood's performance-oriented iteration (Wood 2011, section 6.2).
```

```{admonition} Which curvature the binned route uses
:class: note
For a family and link whose link is not canonical, the Laplace REML
criterion uses the observed information, not the expected (Fisher)
information (Wood 2011). The exact route (`discrete=False`) always does.

The binned route (`discrete=True`) runs mgcv-style performance iteration
(Wood et al. 2017). Each step fits a weighted working linear model, and a
weighted model needs non-negative weights. So the binned route uses the
observed information, in its optimiser, its final refit and the objective
it reports, only for pairs whose log-likelihood is concave in the linear
predictor and whose observed rows superglm forms so that they stay
non-negative in floating point: Gamma/log, Tweedie/log, NB2/log, and
binomial probit and cloglog. For these pairs the binned and exact routes
publish the same criterion. They do not reach exactly the same optimum,
because performance iteration holds the working weights fixed within a
step and has no term for how they change with the smoothing parameters.

Every other pair uses Fisher information throughout on the binned route:

- For the canonical pairs (Poisson/log, binomial/logit,
  Gaussian/identity, Gamma/inverse) Fisher and observed are the same.
- Gaussian/log and binomial/cauchit have observed weights that can go
  negative, so their binned fits keep the expected information that
  penalized quasi-likelihood uses (Wood, Pya and Säfken 2016).
- Binomial/log is concave, but its maximum can sit on the boundary of the
  mean space, where probabilities are clipped and rows with an event carry
  no curvature, so its binned fit stays on Fisher.
- Other non-canonical built-in pairs, such as Poisson/sqrt, Gamma/identity
  and Tweedie with a non-canonical power link, are not yet declared and
  stay on Fisher.
- A custom family or link also uses Fisher on the binned route.

For all of these, the binned route's REML objective is the Fisher one, not
the exact route's.
```

```{admonition} Recommended workflow for spline-based GAM models
:class: tip
Use `fit_reml()` with `select=True` on your spline terms. REML estimates a separate smoothing parameter per term, and `select=True` adds a double-penalty decomposition (linear + wiggly subgroups) that lets REML shrink irrelevant terms all the way to zero — mgcv-style automatic term selection without needing `selection_penalty > 0`.
```

```{admonition} Current multi-penalty guard rails
:class: note
The exact and discrete REML paths support shared-block multi-penalty terms
such as tensor margins and `Spline(m=(...))`. The sparse-additive
`selection_penalty > 0` path also accepts shared-block multi-penalty
terms. For tuples compatible with the selected spline class,
`select=True + m=(...)` produces a null-space component plus one component
per derivative order, with separate REML lambdas; per-class order limits
still apply. Tensor interactions with multi-order spline parents remain
unsupported.
```

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
| IRLS / PIRLS | What quadratic surrogate should represent the GLM right now? | working weights `W`, working response `z` | GLM loss is not quadratic | `src/superglm/solvers/pirls.py`, `src/superglm/solvers/irls_direct/` |
| Inner coefficient solver | For fixed `W` and `z`, what coefficients solve the penalized WLS problem? | `beta`, intercept | penalties may be nonsmooth, and shape rules add inequality constraints | direct solve, proximal Newton BCD, or an active-set QP (`src/superglm/solvers/constrained_qp.py`) |
| REML / fREML | How smooth should each smooth term be? | per-term `lambda_j` | the inner fit does not choose smoothness for you | `src/superglm/reml/` |

```{admonition} If you remember one thing
:class: tip
`IRLS` fits coefficients for fixed penalties. `REML` fits the penalties themselves.
```

## 3. Why outer and inner solvers are not actually crazy

At first glance, nested solvers look like engineering excess. But they usually mean the model has genuinely separable layers.

Here the layers are real:

- the GLM likelihood is nonlinear, so it needs repeated local quadratic updates
- the sparse penalty is nonsmooth, so the weighted least-squares subproblem needs a prox-capable inner solver
- the smoothness parameters are hyperparameters of the penalized fit, so they need their own outer optimization

So the nesting is not arbitrary complexity. It is the optimization structure implied by the model.

## 4. Why IRLS exists

A GLM maximises a log-likelihood that is nonlinear in $\boldsymbol{\beta}$. IRLS replaces it with a sequence of WLS problems by taking a local Fisher-scoring approximation at each iteration.

The working weights encode the local curvature of the GLM likelihood:

$$
W_i = w_i \cdot \frac{(\mathrm{d}\mu_i / \mathrm{d}\eta_i)^2}{V(\mu_i)}
$$

The working response linearises the residual around the current fit:

$$
z_i = \eta_i + \frac{y_i - \mu_i}{\mathrm{d}\mu_i / \mathrm{d}\eta_i}
$$

The P-IRLS update (with smoothing penalty $\mathbf{S}$) solves the augmented system for the intercept $\alpha$ and coefficients $\boldsymbol{\beta}$ simultaneously:

$$
\begin{pmatrix} \sum W_i & \mathbf{1}^\top \mathbf{W} \mathbf{X} \\ \mathbf{X}^\top \mathbf{W} \mathbf{1} & \mathbf{X}^\top \mathbf{W} \mathbf{X} + \mathbf{S} \end{pmatrix} \begin{pmatrix} \alpha \\ \boldsymbol{\beta} \end{pmatrix} = \begin{pmatrix} \sum W_i z_i \\ \mathbf{X}^\top \mathbf{W} \mathbf{z} \end{pmatrix}
$$

The code does not form this bordered system. It profiles the intercept out
first: every column is centred on its weighted mean, which leaves a
$p \times p$ system for $\boldsymbol{\beta}$ alone, and the intercept follows
from the weighted means. That is the Schur complement of the intercept row,
so it gives the same answer, and centring keeps the rounding of each column
in proportion to its spread rather than to its size.

That is the whole IRLS idea. Each outer iteration does four things:

1. compute $\mathbf{W}$ and $\mathbf{z}$ from the current fit
2. solve one penalized weighted least-squares problem
3. update $\eta$ and $\mu$
4. stop when the penalized deviance stabilizes (the default,
   `convergence="deviance"`; `"coefficients"` stops on the coefficient change
   instead). A step the line search had to reject never counts as converged.

So the only reason IRLS feels complicated at first is that the weights are
endogenous. Once $\mathbf{W}$ and $\mathbf{z}$ are fixed, the subproblem is
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

if any term carries a shape rule, or selection_penalty == 0:
    # All penalties are quadratic (smoothing only)
    # → one p×p solve per IRLS iteration, or one small QP
    #   when shape rules add inequality constraints
    result, H⁻¹ = fit_irls_direct(X, y, W, S)

else:
    # Group lasso / sparse group lasso penalty is nonsmooth
    # → proximal Newton BCD inner loop per IRLS iteration
    result = fit_pirls(X, y, W, S, penalty)
```

## 6. Why an inner solver is needed

For fixed $\mathbf{W}$ and $\mathbf{z}$, the penalized subproblem is:

$$
\min_{\boldsymbol{\beta}} \; \frac{1}{2} \|\mathbf{W}^{1/2}(\mathbf{z} - \alpha\mathbf{1} - \mathbf{X}\boldsymbol{\beta})\|^2 + \frac{1}{2}\boldsymbol{\beta}^\top \mathbf{S} \boldsymbol{\beta} + P(\boldsymbol{\beta})
$$

Now split by penalty type.

### 6.1 Smooth case: selection_penalty = 0

If there is no group lasso penalty, $P(\boldsymbol{\beta}) = 0$ and the whole objective is quadratic. The solution is a single linear system:

$$
\boldsymbol{\beta} = (\mathbf{X}^\top \mathbf{W} \mathbf{X} + \mathbf{S})^{-1} \mathbf{X}^\top \mathbf{W} \mathbf{z}
$$

```
# ── Direct IRLS: smooth penalties only ─────────────────────────
# S = block-diagonal smoothing penalty (one block per spline)
# XᵀWX is formed from per-group Grams and cross-Grams
#   → O(n_bins ⋅ K²) for discretized groups, not O(n ⋅ p²)

η = α + Xβ
μ = g⁻¹(η)

for t in 1, 2, ...:
    W, z = working weights and working response at (η, μ, y)

    # Profile the intercept out: centre the columns on their
    # weighted means x̄ = XᵀW1 / ΣW, and z on z̄ = ΣWz / ΣW
    X̃ᵀWX̃ = XᵀWX − ΣW ⋅ x̄x̄ᵀ           # centred Gram, p×p
    X̃ᵀWz̃ = XᵀWz − ΣW ⋅ x̄z̄

    β = solve(X̃ᵀWX̃ + S, X̃ᵀWz̃)       # Cholesky, or a rank-certified factor
    α = z̄ − x̄ᵀβ

    η = α + Xβ                         # line search if the step raised
    μ = g⁻¹(η)                         #   the penalized deviance
    if converged(deviance): break
```

The repo uses this in `src/superglm/solvers/irls_direct/`. The solve step has
four forms there, one per kind of model:

- the Gram step above, the default (`direct_solve="gram"`, or `"auto"` for
  most models);
- a QR factor of the weighted, centred design stacked on the penalty's square
  root (`direct_solve="qr"`), backward-stable but $O(n p^2)$ per iteration,
  for smaller datasets;
- block elimination for large `RandomEffect` chains and factor-smooth
  (`basis="fs"` or `"sz"`) terms, which never forms their dense Gram
  (`direct_solve="auto"` picks it from the terms and their sizes);
- an active-set quadratic program when a term carries a shape rule
  (section 6.3).

### 6.2 Nonsmooth case: selection_penalty > 0

If $\lambda_1 > 0$, the penalty includes a group lasso term:

$$
P(\boldsymbol{\beta}) = \lambda_1 \sum_g w_g \|\boldsymbol{\beta}_g\|_2
$$

This is not differentiable at $\boldsymbol{\beta}_g = \mathbf{0}$ — it has a kink where entire coefficient groups hit the origin.

That means a plain Newton or Cholesky solve is no longer enough. You need a method that can handle:

- exact zeros (entire groups removed)
- whole groups turning on and off
- warm starts across IRLS and lambda paths

That is what the inner proximal solver is for.

### 6.3 Constrained case: shape rules

A shape rule (`Constraint.fit.increasing`, `convex`, a cap, a flat stretch,
`max_step`) is a set of linear inequalities on a term's coefficients: for a
monotone P-spline, every first difference of its coefficients has one sign;
for a convex spline of degree three or lower, a set of linear conditions on
its coefficients that holds exactly when its curvature does. With $\mathbf{W}$ and
$\mathbf{z}$ fixed, the penalized least-squares step becomes

$$
\min_{\boldsymbol{\beta}} \; \tfrac{1}{2}\boldsymbol{\beta}^\top \mathbf{H} \boldsymbol{\beta} - \mathbf{g}^\top \boldsymbol{\beta}
\quad \text{subject to} \quad \mathbf{A}\boldsymbol{\beta} \ge \mathbf{b},
\qquad \mathbf{H} = \tilde{\mathbf{X}}^\top \mathbf{W} \tilde{\mathbf{X}} + \mathbf{S}
$$

a convex quadratic program. superglm solves it with a primal active-set
method (Nocedal and Wright, *Numerical Optimization*, 2nd ed., 2006,
Algorithm 16.3): keep a working set of constraints held at equality, solve the
equality-constrained problem on it, and either step until a new constraint
blocks or release the constraint whose multiplier has the wrong sign. Every
shape-constrained spline kind (`PSpline`, `BSplineSmooth`,
`CubicRegressionSpline`) fits on this one solver, in
`src/superglm/solvers/constrained_qp.py`. The SCOP reparameterization
(Pya and Wood 2015), which releases up to 0.41.0 used for shape-constrained
P-splines, is retired; a model saved with it still loads and predicts.

The textbook loop needs a few guards to be safe in floating point:

- **Independent working rows.** In exact arithmetic a row that depends on
  the working rows cannot block a step; rounding can make it appear to. The
  working rows' span is held as a QR factor that is updated as rows enter and
  leave (Golub and Van Loan 2013, section 6.5), and a row that would not raise
  its rank enters by replacing one (Gill and Wong 2015).
- **Anti-cycling.** On a degenerate face, a release can be undone by a zero
  step that blocks on the same row, and the working sets cycle. Once a
  working set repeats with the objective unchanged, releases switch to
  Bland's least-index rule (Bland 1977), which cannot cycle.
- **Warm starts.** Each IRLS iteration starts from the previous one's working
  set, so a fit's later steps usually need few or no changes to it. A warm
  solve that fails is solved again from a cold start.
- **A rank-deficient $\mathbf{H}$** is truncated by the shared rank policy,
  never regularized, and the result is accepted only after a check of
  feasibility, multiplier signs, stationarity and complementarity.

With `fit_reml()`, the smoothing parameters are chosen without the shape
rules, and the model is then refit with them at those smoothing parameters.
See [Monotone and curvature constraints](../how-to/constrain-a-smooth.md)
for what that means for intervals and degrees of freedom.

## 7. What "proximal Newton BCD" means

This name sounds more complicated than the actual algorithm.

For one PIRLS outer step, the code loops over coefficient groups and does:

1. hold $\mathbf{W}$ and $\mathbf{z}$ fixed
2. update the intercept in closed form
3. for each group $g$, compute the local gradient and block Hessian
4. minimize the group's local quadratic model plus its penalty
5. update the residual cheaply

The gradient and Hessian for group $g$, including the group's share of the
smoothing penalty, are:

$$
\nabla_g = -\mathbf{X}_g^\top \mathbf{W} \mathbf{r} + (\mathbf{S}\boldsymbol{\beta})_g, \qquad \mathbf{H}_g = \mathbf{X}_g^\top \mathbf{W} \mathbf{X}_g + \mathbf{S}_{gg}
$$

where $\mathbf{r} = \mathbf{z} - \alpha\mathbf{1} - \mathbf{X}\boldsymbol{\beta}$ is the working residual. Step 4 then solves

$$
\boldsymbol{\beta}_g^{\text{new}} = \arg\min_{\mathbf{b}} \; \nabla_g^\top(\mathbf{b} - \boldsymbol{\beta}_g) + \tfrac12 (\mathbf{b} - \boldsymbol{\beta}_g)^\top \mathbf{H}_g (\mathbf{b} - \boldsymbol{\beta}_g) + P_g(\mathbf{b})
$$

and how it does so depends on the penalty:

- **Group lasso and group elastic net** ($P_g = \lambda_1 w_g \|\mathbf{b}\|_2$,
  plus a ridge part for the elastic net): this block problem is solved
  exactly. The group is zero when $\|\mathbf{H}_g\boldsymbol{\beta}_g - \nabla_g\|_2 \le \lambda_1 w_g$;
  otherwise the eigendecomposition of $\mathbf{H}_g$ reduces the optimality
  condition to one scalar equation in $\|\mathbf{b}\|_2$, solved by a root
  search. This is the block coordinate descent of Qin, Scheinberg and
  Goldfarb (*Mathematical Programming Computation* 5(2), 2013).
- **Ridge:** the block problem is quadratic, so it is one solve with
  $\mathbf{H}_g + \lambda_1\mathbf{I}$, factored once per IRLS step.
- **Other penalties, such as sparse group lasso:** one proximal-gradient
  step, with step length $1/L_g$ where $L_g$ is the largest eigenvalue of
  $\mathbf{H}_g$. For sparse group lasso the prox is soft-thresholding of each
  coordinate followed by group soft-thresholding (Simon, Friedman, Hastie and
  Tibshirani 2013).

```
# ── Proximal Newton BCD: one PIRLS outer step ─────────────────
# W, z fixed from the current IRLS approximation
# r = z − α1 − Xβ (working residual)
# Hg = XgᵀWXg + Sgg, formed (and factored) once per outer step

for inner in 1, 2, ..., max_bcd:
    # Intercept update (closed form, unpenalised)
    δ = ΣWᵢrᵢ / ΣWᵢ
    α += δ
    r -= δ

    # BCD cycle: one block minimization per group
    for g in groups:
        if active_set and g was zero and passed its KKT check: skip

        βg_old = βg
        ∇g     = −Xgᵀ(W ⊙ r) + (Sβ)g           # group gradient
        βg_new = block_minimizer(Hg, ∇g, βg_old, penalty)

        # Cheap residual update (no full matvec)
        if βg_new ≠ βg_old:
            r -= Xg(βg_new − βg_old)
            βg = βg_new

    every 5th sweep: recompute r from scratch (stops rounding drift)
    if max|β − β_prev| < tol / 100: break
```

The words in the name mean:

### 7.1 Proximal

The penalty is handled by its proximal operator rather than by pretending it is smooth. For ordinary group lasso, the prox is block soft-thresholding. That gives exact zeros.

### 7.2 Newton

The update uses local curvature, not just a gradient. The block Hessian $\mathbf{H}_g$ makes this much more aggressive than a first-order proximal-gradient method.

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

That is exactly what `DiscretizedSSPGroupMatrix` (imported from
`superglm.group_matrix`, defined in
`src/superglm/_group_matrix/_group_matrix_discretized.py`) is built for. Here `SSP` just means the spline is stored in the repo's
reparameterized penalized form: sparse basis `B`, small dense transform
`R_inv`, and penalty matrix `omega`, rather than one fully materialized dense
design block.

#### 9.2.3 Why this matters in practice

For discretized spline groups, the expensive matrix-building work can move from repeated $O(n)$ passes on the full basis to something much closer to:

- one $O(n)$ aggregation by bin (weighted bincounts)
- then $O(n_{\text{bins}} \cdot K^2)$ dense work in basis space

where $n_{\text{bins}} \ll n$ (typically 256 bins vs 678k observations) and $K$ is the local basis size for that group (typically 9-13).

So discretization helps most with:

- weighted Gram construction ($\mathbf{X}_g^\top \mathbf{W} \mathbf{X}_g$)
- repeated PIRLS updates
- repeated REML objective evaluations
- cached-$\mathbf{W}$ continuation in the fast REML path

#### 9.2.4 Why the p x p solve is often still fine

In many models here, $p$ is not enormous. It is often tens or low hundreds, not millions.

That means a Cholesky or eigendecomposition of a $p \times p$ matrix is not free, but it is often not the part that kills you.

The real win is usually:

- reduce repeated data passes
- reduce repeated basis aggregation
- cache weighted summaries when $\mathbf{W}$ is unchanged

That is the whole point of the discrete cached-$\mathbf{W}$ REML path.

## 10. Where REML comes in

Everything above assumes the smoothness penalties are already known.

That is the next big question:

> how large should the smoothing penalties be?

The inner solve does not answer that. It only answers:

> for this value of $\lambda$, what is the best coefficient vector?

REML answers the higher-level question:

> which $\lambda$ values make the smooths appropriately flexible without overfitting?

That is why REML must be an outer loop.

## 11. Why REML has to be outside PIRLS

The REML objective depends on the fitted coefficients and the local weighted geometry.

For one candidate set of smoothing parameters, you need:

1. the fitted $\hat{\boldsymbol{\beta}}(\boldsymbol{\lambda})$
2. the current IRLS weights $\mathbf{W}$
3. the penalized Hessian $\mathbf{H} = \mathbf{X}^\top \mathbf{W} \mathbf{X} + \mathbf{S}(\boldsymbol{\lambda})$

You only get those after solving the inner GLM problem.

So the nesting is inevitable:

- inner fit gives you $\hat{\boldsymbol{\beta}}$, $\mathbf{W}$, and $\mathbf{H}$
- outer REML uses those to score the smoothing parameters
- then proposes new smoothing parameters
- then reruns the inner fit

That is why `fit_reml()` is "refit-heavy". It is solving a different optimization problem on top of the coefficient problem.

## 12. What the REML objective is doing

In this repo, `reml_laml_objective()` in `src/superglm/reml/objective.py` evaluates a Laplace-approximate REML criterion (Wood 2011).

This section stays intentionally short. A full REML derivation turns into a paper-sized detour;
the main thing this guide needs is the role each term plays in the outer optimization.

For a known-scale family ($\phi = 1$, e.g. Poisson):

$$
\mathcal{V}(\boldsymbol{\rho}) = -\ell(\hat{\boldsymbol{\beta}}) + \tfrac{1}{2}\hat{\boldsymbol{\beta}}^\top \mathbf{S} \hat{\boldsymbol{\beta}} + \tfrac{1}{2}\log|\mathbf{H}| - \tfrac{1}{2}\log|\mathbf{S}|_+
$$

where $\boldsymbol{\rho} = \log\boldsymbol{\lambda}$. For estimated-scale families (Gamma, Tweedie), a $\phi$-profiled version replaces the NLL with a scale-free deviance term.

Here $|\mathbf{S}|_+$ means the pseudo-determinant of $\mathbf{S}$: the product of its strictly positive eigenvalues only.
The `+` matters because spline penalties usually have a null space, so $\mathbf{S}$ is often rank-deficient. For example,
constant or linear directions may be left unpenalized, which gives exact zero eigenvalues that should be excluded from the
log-determinant term. In the code, this is the same idea as `log_det_omega_plus` and `cached_logdet_s_plus(...)` in
`src/superglm/reml/penalty_algebra.py`.

The four terms:

| Term | Meaning |
|------|---------|
| $-\ell(\hat{\boldsymbol{\beta}})$ | Fit the data (negative log-likelihood) |
| $\hat{\boldsymbol{\beta}}^\top \mathbf{S} \hat{\boldsymbol{\beta}}$ | Penalize wiggliness |
| $\log\lvert\mathbf{H}\rvert$ | Account for posterior concentration (Laplace correction) |
| $-\log\lvert\mathbf{S}\rvert_+$ | Adjust for penalty scale using only penalized directions |

REML is not "just another regularizer". It is an empirical-Bayes criterion for picking the smoothness parameters — it integrates out $\boldsymbol{\beta}$ via a Laplace approximation and optimizes the resulting marginal likelihood.

### 12.1 REML gradient and Hessian

The partial gradient with respect to $\rho_j = \log\lambda_j$ at fixed $\mathbf{W}$ is:

$$
\frac{\partial \mathcal{V}}{\partial \rho_j} = \frac{1}{2}\left(\lambda_j \left(\frac{1}{\phi}\hat{\boldsymbol{\beta}}_g^\top \boldsymbol{\Omega}_g \hat{\boldsymbol{\beta}}_g + \text{tr}(\mathbf{H}^{-1}_{[g,g]} \boldsymbol{\Omega}_g)\right) - r_g\right)
$$

where $\boldsymbol{\Omega}_g$ is the penalty basis for group $g$ and $r_g$ is the penalty rank. Setting the gradient to zero gives the Fellner-Schall fixed-point update:

$$
\lambda_j^{\text{new}} = \frac{r_g}{\frac{1}{\phi}\hat{\boldsymbol{\beta}}_g^\top \boldsymbol{\Omega}_g \hat{\boldsymbol{\beta}}_g + \text{tr}(\mathbf{H}^{-1}_{[g,g]} \boldsymbol{\Omega}_g)}
$$

The numerator is the penalty rank (degrees of freedom available). The denominator balances the current coefficient energy ($\hat{\boldsymbol{\beta}}$ term) against the posterior uncertainty (trace term). When the signal is strong, the $\hat{\boldsymbol{\beta}}$ term dominates and $\lambda$ stays small. When the signal is weak, the trace dominates and $\lambda$ grows large.

For the Newton path, the Hessian is:

$$
\frac{\partial^2 \mathcal{V}}{\partial \rho_i \partial \rho_j} = -\frac{1}{2}\text{tr}(\mathbf{H}^{-1}\mathbf{S}_i \mathbf{H}^{-1}\mathbf{S}_j) + [\text{diagonal correction}]
$$

with an optional first-order $W(\boldsymbol{\rho})$ correction that accounts for $\mathrm{d}\mathbf{W}/\mathrm{d}\boldsymbol{\rho}$ via the implicit function theorem.

## 13. The REML paths in this repo

Both paths below search $\boldsymbol{\rho} = \log\boldsymbol{\lambda}$ inside
a box. The box is not a fixed range of $\lambda$: it is
$[10^{-6}, 10^{10}]$ times the fit's *information unit*, the mean Fisher
information one likelihood row carries at the intercept-only fit, rounded to
a power of two (`src/superglm/_lambda_box.py`). A penalty $\lambda\mathbf{S}$
is weighed against the data curvature $\mathbf{X}^\top\mathbf{W}\mathbf{X}$,
so multiplying every prior weight by $c$ multiplies the optimal $\lambda$ by
$c$ too, while predictions and EDF stay where they were (Wood, Pya and Säfken
2016, section 3.1). A box in the fit's own units moves with the optimum, so
the same model fits the same way in any units. Rounding the unit to a power of
two makes a rescaling by $2^k$ move the bounds exactly, and the ceiling never
falls below $10^{10}$, since it acts as the working infinity for a term that
should be smoothed away. When a smoothing parameter finishes on a bound and
the gradient presses against it, the fit warns and names the term.

### 13.1 Direct REML (Newton)

When `selection_penalty = 0`, the inner problem is smooth. The repo uses direct IRLS plus a damped Newton outer loop on $\boldsymbol{\rho} = \log\boldsymbol{\lambda}$, with Armijo line search.

That is `optimize_direct_reml()`.

A cold exact fit with at least eight estimated smoothing parameters, no random-effect or factor-smooth term and a `max_reml_iter` of at least 10 starts from the cached-W fixed point when its family and link are Poisson/log, binomial with the logit, probit or cloglog link, Gamma/log, Tweedie/log or NB2/log with a fixed theta: the pairs whose cached-weights working model carries the exact route's curvature, where the start was measured to pay. The discrete path's cached-weights REML loop runs first on the same exact design, and the Newton loop below starts from its smoothing parameters and coefficients instead of the bootstrap. That fixed point lacks the working weights' dependence on the smoothing parameters, so the first Newton step moves every direction whose curvature puts a minimum within the step bound; only from the next iteration are flat directions held. On such a fit an accepted decrease within the rounding of the two objective evaluations counts as no decrease, so the dead-search certificate decides that stop. A fit warm-started with a `lambda2_init` mapping starts from those values and skips it, as do `estimate_p`'s power-search candidates and the fits of an NB2 model whose theta is estimated (`theta="auto"`).

Any other cold fit takes its first step from a bootstrap fit. Main effects are almost unpenalized there ($\lambda = 10^{-4}$) and interaction components start at $\lambda = 1$. A random effect starts at its levels' mean information, $\lambda = \operatorname{tr}(A)/q$ for its $q$ levels, with $A$ the intercept-centred Fisher Gram at the intercept-only fit given the offset: under the diagonal approximation $\mathrm{edf}_g = A_{gg}/(A_{gg} + \lambda)$ that puts the typical level's effective degrees of freedom at one half, the start Wood, Pya & Säfken (2016, §3.1) and mgcv's `initial.sp` describe. Where that fit has no valid mode (it does not converge, its intercept is infinite, or it holds a weighted row at a clip past which the Fisher weight grows), the random effect keeps the almost unpenalized $10^{-4}$. $A$ carries the units of the prior weights, and of the response to the power $2 - p$ under a log link with variance $\mu^p$, as the REML optimum $\phi/\sigma_b^2$ does, so the start is the same number of log units from the optimum in any units. Almost unpenalized, a random-effect level that its own rows do not identify (the parent of a nested effect, whose indicators its children's span, or a level whose responses are all zero under a log link) has a posterior variance of order $1/\lambda$, so the fixed-point step from that fit returns about $\lambda$ times a constant and leaves the Newton loop to climb many log units. Near the optimum, where the Newton model predicts a decrease $P$ below the stop resolution $\tau(1 + |\mathcal{V}|)$ and the dead-search certificate already holds, a shorter step accepted after a stationary trial rejected the full step, with a decrease above $2P\alpha(1 - (1 - c)\alpha)$ (the most a quadratic criterion that rejected the full step allows, $c$ the Armijo constant) and within the stop resolution, was decided by the error the inner solves leave in the objective. When steps at that fraction, a linear rate, cannot reach the score bar in the iterations left, it is discarded and the certificate decides that stop.

```
# ── Direct REML: Newton on log-λ ──────────────────────────────
# Start: the cached-W pre-solve (above), or one bootstrap fit
# Then: Newton with gradient, Hessian, W(ρ) correction, line search

# Bootstrap: fit once at small penalty (λ = 10⁻⁴ for main effects,
#   1 for interactions, tr(A)/q for random effects, all clipped to
#   the box), then take one Fellner-Schall step from it
β̂, H⁻¹ = fit_irls_direct(X, y, W, S(λ_boot))
for j in each group:
    ρⱼ = log(rⱼ / (β̂gᵀ Ωg β̂g + tr(H⁻¹[g,g] Ωg)))

for iter in 1, 2, ..., max_reml_iter:
    λ = exp(ρ)
    β̂, H⁻¹, XᵀWX = fit_irls_direct(X, y, W, S(λ))

    ∇ = reml_gradient(β̂, H⁻¹, λ)
    ∇ += w_correction(dW/dη, dβ̂/dρ)     # implicit function theorem
    ℋ = reml_hessian(H⁻¹, λ, ∇)

    hold directions whose gradient and curvature are both negligible
    if the gradient, the objective change or the Newton step left
       is below the stop resolution: stop

    δ = −solve(ℋ, ∇) on the directions not held   # Newton step
    if max|δ| > 5: δ ⋅= 5 / max|δ|                 # step bound

    # Armijo line search: refit at trial ρ
    for step in 1.0, 0.5, 0.25, ...:
        ρ_trial = clip(ρ + step ⋅ δ, box)
        obj_trial = reml_objective(ρ_trial)
        if obj_trial ≤ obj + 10⁻⁴ ⋅ step ⋅ (∇ᵀδ):
            ρ = ρ_trial; break
    if no trial was accepted: the dead-search certificate decides the stop
```

This is the cleanest path mathematically. The $W(\boldsymbol{\rho})$ correction, used from the first Newton step, accounts for the fact that changing $\boldsymbol{\lambda}$ changes $\hat{\boldsymbol{\beta}}$, which changes $\boldsymbol{\mu}$, which changes $\mathbf{W}$.

Its gradient term is $\tfrac12\operatorname{tr}(\mathbf{H}^{-1}\mathbf{X}^\top\mathbf{D}_j\mathbf{X})$, with $\mathbf{D}_j$ the change of the weights along $\rho_j$. It equals $\tfrac12\sum_r d_{jr} h_r$, where $h_r = \mathbf{x}_r^\top\mathbf{H}^{-1}\mathbf{x}_r$ is row $r$'s leverage (Wood 2011, section 3.5.1). One pass over the rows therefore serves every smoothing parameter. The exact Newton Hessian also needs $\operatorname{tr}(\mathbf{H}^{-1}\mathbf{X}^\top\mathbf{D}_i\mathbf{X}\,\mathbf{H}^{-1}\mathbf{X}^\top\mathbf{D}_j\mathbf{X})$, which costs one $p \times p$ weighted cross-product per smoothing parameter. On the dense solver, a model with four or more smoothing parameters and no random-effect or factor-smooth term leaves that product out of the outer Hessian once those cross-products would cost about two seconds a step ($m\,n\,p^2 \ge 10^{11}$ multiply-adds). It keeps the terms the same pass forms. Binomial/logit on the dense solver
takes this route at any size, since its derivative Grams cost several
leverage passes. The gradient stays exact, so the fit stops at the same optimum; only the Newton steps that reach it can differ.

In this repo: `src/superglm/reml/direct.py::optimize_direct_reml`.

### 13.2 Discrete cached-W REML (fREML)

When `discrete=True`, the expensive part is repeatedly rebuilding $\mathbf{X}^\top \mathbf{W} \mathbf{X}$ from $n$ observations.

So the repo runs performance-oriented iteration (Wood 2011, section 6.2;
the scheme mgcv's `bam` uses): each outer iteration takes one IRLS step,
which refreshes $\mathbf{W}$ and caches the weighted summaries, then one
Newton step on $\log\boldsymbol{\lambda}$ for the REML criterion of the
working linear model that $\mathbf{W}$ defines. The line search tries its
candidate $\boldsymbol{\lambda}$ on the cached summaries alone:

```
# ── Cached-W fREML: minimize data passes ──────────────────────
# Key insight: changing S(λ) with fixed W is O(p³),
#              but refreshing W requires O(n) data passes.

for poi_iter in 1, 2, ..., max_reml_iter:
    # One IRLS step at the current λ (one pass over the n rows)
    β̂, H⁻¹, XᵀWX = fit_irls_direct(X, y, W, S(λ), max_iter=1)
    cache(centred XᵀWX, centred XᵀWz, weighted column means, ΣW)

    obj = reml_objective(β̂, H, λ)          # the working model's REML
    ∇, ℋ = reml_gradient_and_hessian(β̂, H⁻¹, λ)

    hold flat directions; stop if the gradient, objective change or
        Newton step left is below the stop resolution, with the
        working model settled
    δ = −solve(ℋ, ∇);  if max|δ| > 5: δ ⋅= 5 / max|δ|

    # Line search on the cached system: no data pass per trial
    for step in 1.0, 0.5, 0.25, ...:
        λ_trial = exp(clip(ρ + step ⋅ δ, box))
        β̂_trial = cholesky_solve(centred XᵀWX + S(λ_trial), centred XᵀWz)
        if reml_objective(trial) is low enough: accept; break
```

The mental model is:

- if $\mathbf{W}$ changes, the local geometry changed and you need fresh weighted summaries (expensive)
- if only $\mathbf{S}(\boldsymbol{\lambda})$ changes, you can reuse the same weighted geometry and do cheap $p \times p$ algebra (free)

The working model holds $\mathbf{W}$ fixed within a step and has no term for
how $\mathbf{W}$ moves with $\boldsymbol{\lambda}$, so this path stops near,
not exactly at, the exact REML optimum. Where the REML surface is flat in a
noise direction, it may also stop at a nearby stationary point with a large
but not maximal $\lambda$. A tensor term whose margins share a block takes a
damped Newton step on the pair instead of the plain step bound.

The same loop has a second job. On an exact fit (`discrete=False`) that
meets the pre-solve rule in section 13.1, it runs first on the exact design
to give the Newton loop its start.

In this repo: `src/superglm/reml/discrete.py::optimize_discrete_reml_cached_w`.

### 13.3 REML and a selection penalty

`fit_reml()` does not combine smoothness selection with group-lasso sparsity: it
refuses `selection_penalty > 0` before any work starts. Choose which terms survive
with `fit()` or `fit_path()` at fixed smoothing, then choose smoothness with
`fit_reml()` on the survivors.

## 14. Compared to other software

There is no single universal "meta" that every library follows, because libraries are solving slightly different problems.

But for spline-based GAM smoothness selection, `REML` and `fREML` are absolutely mainstream choices, especially in the `mgcv` tradition.

At the same time:

- many libraries do cross-validation instead of REML
- many penalized GLM libraries focus on coefficient regularization and do not try to estimate smoothness parameters at all
- some libraries support splines but stop at fixed lambdas or grid search

So the right way to think about it is:

```{admonition} A practical rule
:class: note
`REML` and `fREML` are standard, high-quality choices for spline smoothness estimation. They are not the universal default for every penalized regression library.
```

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
3. `src/superglm/solvers/irls_direct/_fit.py::fit_irls_direct`, then the step
   it picks in `_dense.py`, `_qr.py`, `_qp.py` or `_structured.py`
4. `src/superglm/solvers/constrained_qp.py::solve_constrained_qp`, for
   shape-constrained terms
5. `src/superglm/model/fit_ops.py::fit_reml`
6. `src/superglm/reml/objective.py::reml_laml_objective`
7. `src/superglm/reml/direct.py::optimize_direct_reml`
8. `src/superglm/reml/discrete.py::optimize_discrete_reml_cached_w`

```{admonition} Three questions to keep in your head
:class: tip
`IRLS`: what weighted least-squares problem is the GLM pretending to be right now?

`Inner solver`: for that weighted least-squares problem, how do I solve the coefficient optimization with the penalties I actually have?

`REML`: after I know how to fit coefficients for fixed penalties, how do I choose the smoothness penalties themselves?
```

## 17. Appendix: demystifying the implementation-heavy parts

This section is about the parts that often feel "agent-generated" or hard to parse when you first read the source.

### 17.1 Cached W: what is actually being cached?

The cached-$\mathbf{W}$ idea in the fast REML path is simpler than it looks.

For one fixed set of IRLS weights $\mathbf{W}$, the coefficient problem only depends on the data through a few weighted summaries:

| Cached quantity | Size | Cost to build |
|---|---|---|
| $\mathbf{X}^\top \mathbf{W} \mathbf{X}$ | $p \times p$ | $O(n \cdot p^2)$ or $O(n_{\text{bins}} \cdot K^2)$ |
| $\mathbf{X}^\top \mathbf{W} \mathbf{z}$ | $p$ | $O(n \cdot p)$ |
| $\mathbf{X}^\top \mathbf{W} \mathbf{1}$ | $p$ | $O(n \cdot p)$ |
| $\sum W_i$ | scalar | $O(n)$ |
| $\sum W_i z_i$ | scalar | $O(n)$ |

`fit_irls_direct()` saves these in their centred form, the one it solves
(section 4): the centred Gram, the centred right-hand side, the weighted
column means $\mathbf{X}^\top\mathbf{W}\mathbf{1}/\sum W_i$, the weighted
mean of $\mathbf{z}$, and $\sum W_i$. A model with large random-effect or
factor-smooth terms caches their block-eliminated system instead of a dense
Gram.

Once you have them, changing the smoothing penalty only changes the penalty matrix $\mathbf{S}(\boldsymbol{\lambda})$. It does **not** force you to revisit all $n$ observations immediately.

So with fixed $\mathbf{W}$, the augmented system is just:

$$
\begin{pmatrix} \sum W_i & \mathbf{X}^\top \mathbf{W} \mathbf{1} \\ \mathbf{X}^\top \mathbf{W} \mathbf{1} & \mathbf{X}^\top \mathbf{W} \mathbf{X} + \mathbf{S}(\boldsymbol{\lambda}) \end{pmatrix} \begin{pmatrix} \alpha \\ \boldsymbol{\beta} \end{pmatrix} = \begin{pmatrix} \sum W_i z_i \\ \mathbf{X}^\top \mathbf{W} \mathbf{z} \end{pmatrix}
$$

That means you can try new lambdas by doing one $O(p^3)$ Cholesky solve on cached summaries, instead of rebuilding all weighted summaries from scratch at $O(n)$ cost.

#### 17.1.1 When is that cache valid?

Only while $\mathbf{W}$ is effectively unchanged.

That is why the fast REML path alternates between:

- an IRLS step, which updates $\mathbf{W}$ ($O(n)$, once per outer iteration)
- line-search trials, which keep $\mathbf{W}$ fixed and only change $\mathbf{S}(\boldsymbol{\lambda})$ ($O(p^3)$ per trial)

The cache lives for one outer iteration: the next IRLS step replaces it.

The cache is not magic. It is just:

> if the observation-side geometry has not changed, do not recompute it

### 17.2 Why some smoothing parameters are held still

Some smoothing components are effectively dead: their coefficient energy
$\hat{\boldsymbol{\beta}}_g^\top\boldsymbol{\Omega}_g\hat{\boldsymbol{\beta}}_g$
is tiny next to the trace term. The practical reading is:

- that smooth component is already basically gone
- pushing its lambda higher will not materially change the fitted mean
- so the REML criterion is flat in that direction

Two pieces of code act on this.

- **At the start.** With `select=True`, a null-space component whose
  bootstrap coefficient energy is below a tenth of its trace term starts at
  the box's ceiling rather than at its fixed-point value, since almost any
  $\lambda$ is a fixed point there.
- **At every Newton step.** A direction whose gradient and curvature are
  both negligible at the stop resolution is held still for that step
  (`freeze_flat_directions` in `src/superglm/reml/convergence.py`), so a flat
  direction neither absorbs the step bound nor keeps the fit from stopping.
  A direction that is flat but concave, a shoulder, takes a bounded step
  downhill instead.

Holding a flat direction is Wood's "working infinity" (Wood 2008, section
3.4), not a new statistical principle. The stop test leaves a held direction
out only while its gradient stays below the hold's bar; a direction whose
gradient grows past it is live again. An accepted stop must also pass a
second-order check, so a concave shoulder cannot pass for a minimum.

### 17.3 QR: where is it used if the fitter uses Cholesky?

The default fit solves the centred normal equations by Cholesky, falling back
to a rank-certified factor when the Gram cannot certify its own rank. QR
appears in three other places, each where stability matters more than speed.

**The QR solve option.** With `direct_solve="qr"`, each IRLS step factors the
stacked matrix

$$
\mathbf{A} = \begin{pmatrix} \mathbf{W}^{1/2} \tilde{\mathbf{X}} \\ \mathbf{S}^{1/2} \end{pmatrix}
$$

so that $\mathbf{A}^\top \mathbf{A} = \tilde{\mathbf{X}}^\top \mathbf{W} \tilde{\mathbf{X}} + \mathbf{S}$. A reduced QR decomposition $\mathbf{A} = \mathbf{Q}\mathbf{R}$ then gives

$$
\mathbf{R}^\top \mathbf{R} = \mathbf{A}^\top \mathbf{A} = \tilde{\mathbf{X}}^\top \mathbf{W} \tilde{\mathbf{X}} + \mathbf{S}
$$

without ever squaring the condition number of the design, as forming the
Gram does. The price is the materialised $n \times p$ design and $O(np^2)$
work per iteration, so this is an option for smaller, near-collinear data
(`src/superglm/solvers/irls_direct/_qr.py`).

**The data rank.** Deciding which coefficients the data can estimate
(section 17.3.2) factors a thin block of rows by QR when the Gram alone cannot
decide.

**Smooth-term tests.** `src/superglm/stats/wood_pvalue.py` uses pivoted QR to
move a smooth term into a numerically stable test space before Wood's
significance calculation.

Coefficient covariance, leverage and EDF come from the inverse of the
penalized Hessian that the fit's last solve leaves behind
(`src/superglm/model/state_ops.py`), not from a separate QR pass.

#### 17.3.1 Why use QR there?

Because for those jobs, you often care more about stability than raw speed.

The QR path is basically saying:

> I want a reliable factorization of the penalized weighted design, even if the active design is a bit awkward or nearly rank-deficient

#### 17.3.2 Which coefficients the data can estimate

After the fit, superglm decides which coefficients the data alone can
identify. This is the *data rank*. A coefficient outside it, such as a spline
basis function over a corner of a tensor term with no rows, is reported as not
estimable, and its standard error is shown as missing. Its fitted value and the
model's predictions do not depend on this decision, which is made after the
coefficients are solved.

The decision is the numerical rank of the weighted design
$\mathbf{W}^{1/2}\mathbf{X}$. Columns are scaled to unit length, and a
direction counts when its singular value is at least $\sqrt{\epsilon}$ of the
largest.

- The weighted Gram $\mathbf{X}^\top\mathbf{W}\mathbf{X}$ is already formed
  during the fit. When its eigenvalues are clearly on one side of that cut, it
  decides on its own.
- Some directions are exactly zero by construction, such as a tensor cell
  whose rows sit on fewer distinct points than it has basis functions. These
  are read off the term's table of distinct rows, without touching the rows.
- Otherwise only the few *grey* directions, the eigenvectors with the smallest
  eigenvalues, are checked against the rows. The rows are projected onto those
  $k$ directions and that $n \times k$ block is factored by QR. This costs
  about $nk^2$, not the $np^2$ of factoring every row. When every such
  direction is one of the exact zeros above, $k = 0$ and no row is read.
- A perturbation bound for symmetric block matrices (Li and Li, 2005) confirms
  that the other directions cannot change the answer. Where it cannot confirm
  this, the fit factors every row instead.

`metrics()` on rows, weights or offsets other than the fit's and a fit's
rebuilt inference state decide the data rank the same way.

On the freMTPL2 frequency book with ten interaction pairs, $p = 1535$. Of
its 230 directions outside the data rank, 164 are read off the tables, and
$k = 178$ directions are checked against the rows.

### 17.4 Separation: checking that a finite answer exists

IRLS assumes the likelihood has a finite maximum. Sometimes it does not. A
categorical level whose responses are all zero under a log-link Poisson, or
all 0 or all 1 under a binomial, has its maximum at $\eta = -\infty$ or
$+\infty$. IRLS then drifts until the objective stops changing and returns
fitted values collapsed to the boundary, while the rank checks still look
healthy. This is nonexistence of the exponential-family MLE (Albert and
Anderson 1984; Fienberg and Rinaldo 2012).

So the design build scans for it before the solver runs
(`src/superglm/diagnostics/separation.py`).

- **Indicator blocks.** For categoricals and crossed categoricals, the
  directions along which the likelihood keeps rising are exactly the cells
  that have exposure and only boundary responses, so a counting pass over the
  rows is exact.
- **Spline-by-factor levels.** In general finding those directions is a
  linear program (Konis 2007): find the rows some direction $\mathbf{a}$ with
  $\mathbf{D}\mathbf{a} \ge 0$ moves strictly. Solving one program per row
  costs one program per distinct row. Instead, one dual program first
  certifies the rows that no such direction can move: by Stiemke's
  alternative, a row is held exactly when some $\mathbf{y} \ge 0$ that is
  positive on it has $\mathbf{D}^\top\mathbf{y} = 0$. Per-row programs then run
  only on the rows that certificate leaves open. On one 3,000-row binomial
  fit whose levels did not separate, the 3,008 per-row programs had taken
  14.8 of its 19.8 seconds; the certificate decides those rows in one
  program.

What the fit does with a separated level (`separation="warn"`, `"error"` or
`"ignore"`) is described in
[the spline-by-factor separation migration note](../development/migrations/spline-factor-separation-hold.md).

### 17.5 Post-fit shape repair and its intercept search

`apply_shape_postfit()` changes a fitted term's curve after the fit, so the
intercept has to be estimated again with the other coefficients held. That
is a one-dimensional problem: minimize the deviance over a shift of the
linear predictor.

- The search looks for the root of the intercept's score with Fisher-scoring
  steps, kept inside the bracket the score's signs give and bisecting where
  a step would leave it (safeguarded Newton; Press et al., *Numerical
  Recipes*, 3rd ed., 2007, section 9.4). Where the family's unit deviance is
  convex in the linear predictor (a canonical link, and a declared list of
  others, such as Gamma/log and binomial probit and cloglog), that root is
  the minimum.
- Where it is not convex, a root can be higher than the start, so the result
  is compared with the start, and if it is higher the search runs again on
  the deviance itself.
- A run that stops on a maximum, where the Fisher step is zero, hands over
  to a search with no derivatives: it brackets a minimum by steps that grow
  outward, then refines it with Brent's method (Brent, *Algorithms for
  Minimization Without Derivatives*, 1973, chapter 5).

On a saturated link the deviance can be flat in the shift to the last bits of
float64. There the derivative-free search compares deviances only by more
than their rounding: it walks outward until one side falls by more than that,
and keeps the start if neither does. Otherwise the side it left by would be
chosen by those last bits, which differ from one BLAS build to another.

The rest of the repair, the projection of the curve onto the shape in the
fit's penalized metric, is described in
[Post-fit repair](../how-to/constrain-a-smooth.md#post-fit-repair).

### 17.6 Compiled kernels: why the first fit is slower

Most of a fit's time goes into a few kinds of loop over the rows: summing
weights into bins, forming weighted cross-products block by block, evaluating
a family's likelihood and its derivatives row by row. superglm writes those
loops in Python and compiles them to machine code with Numba the first time a
process needs them.

So a first fit in a fresh environment includes some compile time, a few
seconds for a typical model. Numba saves what it compiles to an on-disk
cache, and later processes load it from there instead of compiling again.
Two things follow:

- On a laptop or a long-lived server, you pay the compile once, the first
  time each kind of fit runs.
- On machines that start from a clean disk every time, such as job clusters,
  put the cache on persistent storage, and optionally call
  `superglm.warmup()` in the start-up script to compile everything ahead of
  the first fit. [Starting fast on ephemeral clusters](../how-to/start-fast-on-ephemeral-clusters.md)
  shows how.

The kernels are an implementation detail: they compute the same quantities
the sections above describe, and nothing about a fit's answer depends on
whether they were compiled in this process or loaded from the cache. For a
walk-through of each kernel family, which solver step calls it, and how the
cache and `warmup()` work, see
[Compiled kernels](../development/internals/compiled-kernels.md).

### 17.7 A good way to read this code without getting lost

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
