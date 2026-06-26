# Matrix Operations: What They Are and Why They Exist

This guide is for the moment where `fit_reml()` starts looking like a wall of
letters: \(\mathbf{X}\), \(\mathbf{B}\), \(\boldsymbol{\Omega}\), \(\mathbf{R}^{-1}\),
\(\mathbf{W}\), \(\mathbf{z}\), \(\mathbf{H}\), eigendecompositions, QR, Cholesky,
SVD, traces, log-determinants, and Newton steps.

The important point is that these operations are not arbitrary. They answer a
small number of practical questions:

1. **How do we represent a pricing model as fast linear algebra?**
2. **How do we fit a nonlinear GLM by solving repeated weighted least-squares problems?**
3. **How do we choose smoothness parameters \(\boldsymbol{\lambda}\) automatically?**
4. **How do we keep the solve numerically stable when splines, sparse factors, and near-duplicate columns appear?**

For day-to-day modelling, you do not manually build these matrices. SuperGLM
builds them from `features=`, `interactions=`, `sample_weight=`, and `offset=`.
But understanding what the matrices mean makes the solver code much less
mysterious.

## The one-line model

On the link scale, SuperGLM fits

\[
\boldsymbol{\eta} = \alpha\mathbf{1} + \mathbf{X}\boldsymbol{\beta} + \mathbf{o},
\qquad
\boldsymbol{\mu} = g^{-1}(\boldsymbol{\eta})
\]

where:

- \(\alpha\) is the intercept.
- \(\mathbf{X}\) is the design matrix made from spline, categorical, numeric, polynomial, and interaction blocks.
- \(\boldsymbol{\beta}\) is the coefficient vector.
- \(\mathbf{o}\) is an optional offset, such as \(\log(\text{exposure})\).
- \(g^{-1}\) maps the linear predictor back to the response scale, for example `exp` for a log-link Poisson or Gamma model.

The direct REML path repeatedly solves a penalised weighted least-squares problem:

\[
\min_{\alpha, \boldsymbol{\beta}}
\frac{1}{2}\left\|\mathbf{W}^{1/2}
\left(\mathbf{z} - \alpha\mathbf{1} - \mathbf{X}\boldsymbol{\beta}\right)
\right\|^2
+
\frac{1}{2}\boldsymbol{\beta}^{\top}\mathbf{S}(\boldsymbol{\lambda})\boldsymbol{\beta}
\]

Then REML sits outside that fit and asks: **which smoothness parameters
\(\boldsymbol{\lambda}\) make the smooths neither too wiggly nor too flat?**

## Mental ladder

The solver stack is easiest to read as a ladder:

```text
OLS
  -> WLS
    -> IRLS
      -> penalised IRLS
        -> REML over smoothing parameters
          -> final refit after the final lambda values are known
```

The important split is:

```text
IRLS / PIRLS: choose coefficients beta for the current lambdas
REML:         choose lambdas using the fitted beta and local curvature
```

So `fit_reml()` is not "one big solver". It is a nested process:

```text
repeat over REML steps:
    pick current lambdas
    run an inner GLM fit at those lambdas
    inspect beta, Hessian, EDF, log-determinants
    update lambdas
final rebuild and refit at the chosen lambdas
```

## Symbols and objects

| Symbol / object | Meaning | Why it exists |
|---|---|---|
| \(n\) | Number of training rows | Determines row-scale work. Large \(n\) is why grouped/discrete ops matter. |
| \(p\) | Number of solver-space coefficients | Determines dense solve size. Cholesky/SVD/QR mostly scale with \(p\). |
| \(\alpha\) | Intercept | Carries the reference level / base rate. It is usually unpenalised. |
| \(\boldsymbol{\beta}\) | Solver-space coefficients | The unknowns solved in each inner fit. |
| \(\mathbf{B}\) | Raw spline basis matrix, shape \(n \times K\) | Evaluates spline basis functions before solver reparameterisation. |
| \(\boldsymbol{\Omega}\) | Raw spline penalty matrix | Measures roughness. Large \(\mathbf{b}^{\top}\boldsymbol{\Omega}\mathbf{b}\) means a wiggly spline. |
| \(\mathbf{R}^{-1}\) | SSP transform | Reparameterises splines so the solver sees a better-conditioned basis. |
| \(\mathbf{X}_g\) | One feature/group block | Lets SuperGLM operate blockwise: one spline, categorical, numeric, or interaction at a time. |
| \(\mathbf{X}\) | Virtual full design matrix \([\mathbf{X}_1 | \cdots | \mathbf{X}_G]\) | Conceptual full model matrix. Often not materialised. |
| \(\mathbf{W}\) | IRLS working weights | Local curvature of the GLM likelihood. Stored as a vector, not a dense diagonal matrix. |
| \(\mathbf{z}\) | IRLS working response | The pseudo-response that makes the current GLM step look like WLS. |
| \(\mathbf{S}(\boldsymbol{\lambda})\) | Block-diagonal smoothing penalty | Adds roughness penalties to spline coefficient blocks. |
| \(\mathbf{X}^{\top}\mathbf{W}\mathbf{X}\) | Weighted Gram matrix | Data curvature / information for the current IRLS step. |
| \(\mathbf{H}=\mathbf{X}^{\top}\mathbf{W}\mathbf{X}+\mathbf{S}\) | Penalised Hessian | Main curvature matrix used for coefficient solves, EDF, covariance-like quantities, and REML. |
| \(\mathbf{H}^{-1}\) | Inverse or pseudo-inverse of \(\mathbf{H}\) | Used for standard errors, EDF, REML traces, and sensitivity to penalties. |
| \(\phi\) | Dispersion / scale | Needed by some families and REML formulas. For Poisson it is fixed at 1. |
| \(\rho_j=\log \lambda_j\) | Log smoothing parameter | REML optimises on the log scale so \(\lambda_j\) remains positive. |

## Matrix operation definitions

### `matvec`: multiply by a design block

`matvec(beta_g)` computes

\[
\mathbf{X}_g\boldsymbol{\beta}_g
\]

without necessarily building \(\mathbf{X}_g\) as a dense array.

Why it exists: prediction and residual updates need \(\mathbf{X}\boldsymbol{\beta}\).
For a categorical term this can be a lookup. For a discrete spline it can be
"evaluate at bin support points, then index back to rows". That is much cheaper
than multiplying a full dense \(n \times p\) matrix.

### `rmatvec`: multiply by the transpose

`rmatvec(w)` computes

\[
\mathbf{X}_g^{\top}\mathbf{w}
\]

Why it exists: gradients and right-hand sides need transpose products, for
example \(\mathbf{X}^{\top}\mathbf{W}\mathbf{z}\). For categoricals this is a
weighted `bincount`; for splines it is a basis transpose multiply, often with an
\(\mathbf{R}^{-\top}\) sandwich.

### `gram`: weighted self-product

`gram(W)` computes

\[
\mathbf{X}_g^{\top}\operatorname{diag}(\mathbf{W})\mathbf{X}_g
\]

Why it exists: this is the curvature block for one feature. It is the object you
need to solve the WLS normal equations. For categoricals it is diagonal. For
discrete splines it can be computed from binned weights instead of all rows.

### `cross_gram`: weighted product between two blocks

`cross_gram(gm_i, gm_j, W)` computes

\[
\mathbf{X}_i^{\top}\operatorname{diag}(\mathbf{W})\mathbf{X}_j
\]

Why it exists: the full Hessian has off-diagonal blocks. A spline and a
categorical are not independent inside the solve; their columns can be
correlated, so the solver needs the cross-block curvature too.

### Trace

\[
\operatorname{tr}(\mathbf{A}) = \sum_i A_{ii}
\]

Why it exists: traces measure total influence / degrees of freedom. In REML,
terms such as \(\operatorname{tr}(\mathbf{H}^{-1}\boldsymbol{\Omega}_j)\) ask:
"how much does penalty component \(j\) affect the fitted model?"

### Log-determinant

\[
\log|\mathbf{H}|
\]

Why it exists: determinants measure volume / uncertainty. A larger determinant
means the local quadratic bowl is tighter in more directions. REML uses
log-determinants to account for how much model complexity remains after
penalisation.

## Decomposition methods: what they are used for

This is the part that usually feels like magic. It is not magic; each
decomposition is used because it exposes a useful structure.

| Method | What it does | Why SuperGLM uses it |
|---|---|---|
| Cholesky | Writes a symmetric positive definite matrix as \(\mathbf{L}\mathbf{L}^{\top}\). | Fast default for solving \((\mathbf{X}^{\top}\mathbf{W}\mathbf{X}+\mathbf{S})\boldsymbol{\beta}=\text{rhs}\). |
| Pivoted Cholesky | Cholesky with column/row pivoting. | Handles nearly rank-deficient systems better than plain Cholesky. |
| SVD | Writes \(\mathbf{A}=\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^{\top}\). | Slow but robust fallback. Also reveals rank and safely ignores near-zero directions. |
| QR | Writes \(\mathbf{A}=\mathbf{Q}\mathbf{R}\) with orthonormal \(\mathbf{Q}\). | Builds constraint-respecting bases and gives a stable least-squares solve when normal equations are ill-conditioned. |
| Eigendecomposition | Writes symmetric \(\mathbf{A}=\mathbf{U}\boldsymbol{\Lambda}\mathbf{U}^{\top}\). | Splits penalty null/wiggle spaces, builds square-root penalties, and stabilises REML Newton Hessians. |
| Newton step | Uses gradient and Hessian to propose a parameter update. | REML has a smooth objective over \(\boldsymbol{\rho}=\log\boldsymbol{\lambda}\); Newton is much faster than blind trial-and-error. |
| Line search | Try a proposed step, shrink it if the objective does not improve. | The local quadratic approximation can be wrong; line search prevents bad jumps. |

## Why QR appears

### 1. Identifiability constraints

A spline can accidentally duplicate the intercept. For example, a constant
spline contribution and an intercept both explain the same direction. If that is
left in the model, \(\mathbf{H}\) can become singular or poorly identified.

SuperGLM builds a constraint direction, then uses QR to find an orthonormal basis
for the remaining allowed subspace.

Conceptually:

```text
constraint c says: this coefficient direction should not be fitted
QR(c.T) gives an orthonormal coordinate system
keep the columns orthogonal to c
fit the spline in that reduced basis
```

Why QR is a good method here: it gives a numerically stable orthogonal basis. The
point is not to "fit by QR". The point is to remove the forbidden direction
cleanly before fitting.

### 2. Natural spline boundary constraints

Natural splines impose constraints such as

\[
f''(x_{\min}) = 0, \qquad f''(x_{\max}) = 0.
\]

Those are linear constraints on the raw spline coefficients. QR again gives a
basis for the coefficient space that satisfies the constraints. Once the basis
has been projected, the solver only sees valid natural-spline shapes.

### 3. Optional `direct_solve="qr"`

The default direct solve builds normal-equation style matrices, roughly
\(\mathbf{X}^{\top}\mathbf{W}\mathbf{X}\). This is fast, especially with grouped
and discrete matrix operations, but normal equations can amplify conditioning
problems.

`direct_solve="qr"` instead solves the augmented least-squares system more
directly:

\[
\begin{bmatrix}
\mathbf{W}^{1/2}\mathbf{X}_{aug} \\
\mathbf{L}
\end{bmatrix}
\boldsymbol{\theta}
\approx
\begin{bmatrix}
\mathbf{W}^{1/2}\mathbf{z} \\
\mathbf{0}
\end{bmatrix}
\]

where \(\mathbf{X}_{aug}\) includes the intercept and \(\mathbf{L}^{\top}\mathbf{L}=\mathbf{S}\).

Why this is useful: QR is backward-stable for least squares. Why it is not the
default: it materialises the design matrix, which can defeat the performance
benefit of grouped and discrete matrix ops on large data.

## Why eigendecomposition appears

### 1. Splitting a spline penalty into null and wiggle spaces

A spline penalty matrix \(\boldsymbol{\Omega}\) is usually positive semidefinite,
not positive definite. Some directions have zero penalty. For a cubic-like
smooth, those are usually the low-order polynomial directions, such as constant
and linear components. The positive-eigenvalue directions are the wiggly part.

Eigendecomposition exposes this split:

\[
\boldsymbol{\Omega} = \mathbf{U}\boldsymbol{\Lambda}\mathbf{U}^{\top}
\]

- zero / tiny eigenvalues: null space
- positive eigenvalues: penalised wiggle space

Why SuperGLM does this: `select=True` needs to let REML shrink more than just the
wiggle. Without this split, the unpenalised null space can remain in the model
even when the feature is not useful. The decomposition lets SuperGLM build
separate penalty components, so REML can shrink the linear/null part and the
wiggle part separately.

### 2. Building a square-root penalty for QR

The penalised objective contains

\[
\boldsymbol{\beta}^{\top}\mathbf{S}\boldsymbol{\beta}.
\]

To solve it as augmented least squares, we need a matrix \(\mathbf{L}\) such that

\[
\mathbf{L}^{\top}\mathbf{L}=\mathbf{S}.
\]

If \(\mathbf{S}\) is semidefinite, eigendecomposition is a safe way to build that
square root: keep non-negative eigenvalues, take square roots, and leave exactly
unpenalised directions alone.

### 3. Stabilising the REML Newton step

REML optimises over \(\boldsymbol{\rho}=\log\boldsymbol{\lambda}\). The Newton
proposal is roughly

\[
\Delta \boldsymbol{\rho} = -\mathbf{G}^{-1}\mathbf{g}
\]

where \(\mathbf{g}\) is the REML gradient and \(\mathbf{G}\) is the REML Hessian.
In finite precision, or when a smoothing parameter is weakly identified,
\(\mathbf{G}\) can be indefinite or nearly singular. Eigendecomposition lets the
optimizer floor tiny eigenvalues and build a positive definite approximation
before solving for the step. The line search then checks whether the proposed
step actually improves the objective.

## Why Cholesky, pivoted Cholesky, and SVD are all present

The fast path is Cholesky because \(\mathbf{H}\) should usually be symmetric
positive definite after adding the penalty:

\[
\mathbf{H}=\mathbf{X}^{\top}\mathbf{W}\mathbf{X}+\mathbf{S}.
\]

But real pricing data can create hard cases:

- sparse categorical levels
- duplicated or near-duplicated features
- too many knots for the exposure pattern
- very small working weights
- severe collinearity between main effects and interactions
- penalty null spaces that leave some directions weakly identified

So the direct solve uses a hierarchy:

```text
try Cholesky
if that is unstable, try pivoted Cholesky
if that is still unstable, use SVD as the robust fallback
```

The practical meaning is:

- Cholesky is the normal fast route.
- Pivoted Cholesky tries to rescue nearly rank-deficient systems.
- SVD is the "I still need a defensible answer" fallback.

If SVD happens repeatedly, that is usually a modelling/design warning, not just a
linear algebra detail. It often means the model has redundant columns, overly
thin categories, too many knots, or a feature/intercept identifiability issue.

## Why SSP reparameterisation exists

Raw spline bases can be numerically awkward. Some basis columns may have very
different scales. Some may be highly correlated. The same \(\lambda\) value can
mean different things depending on the raw column scaling.

SSP reparameterisation builds a transform \(\mathbf{R}^{-1}\) from an initial
weighted Gram-plus-penalty matrix:

\[
\mathbf{M}
= \frac{\mathbf{B}^{\top}\mathbf{W}_0\mathbf{B}}{\sum_i W_{0i}}
+ \lambda_{init}\boldsymbol{\Omega}
+ \epsilon\mathbf{I}
\]

Then, using a Cholesky factor \(\mathbf{R}\), the solver sees

\[
\mathbf{X}_{spline}=\mathbf{B}\mathbf{R}^{-1},
\qquad
\boldsymbol{\Omega}_{ssp}=\mathbf{R}^{-\top}\boldsymbol{\Omega}\mathbf{R}^{-1}.
\]

Why this helps: it preconditions the spline block. The solver works in a more
balanced coordinate system, which makes REML and IRLS less sensitive to raw basis
scaling.

The important implementation detail is that the dense matrix
\(\mathbf{B}\mathbf{R}^{-1}\) is often logical rather than materialised:

```text
matvec(beta):      B @ (R_inv @ beta)
rmatvec(w):        R_inv.T @ (B.T @ w)
gram(W):           R_inv.T @ (B.T @ diag(W) @ B) @ R_inv
```

## Why the final rebuild/refit happens

This is easy to miss: SSP depends on \(\boldsymbol{\lambda}\). During REML, the
smoothing parameters change. Once REML has chosen the final values, SuperGLM
rebuilds the design using the final \(\mathbf{R}^{-1}\), maps coefficients across
basis parameterisations, and refits.

Why this exists: the final public model should be fitted in the final solver
basis, not in an old provisional SSP basis from an earlier \(\lambda\).

## The direct REML path in plain English

For the common pricing-GAM path:

```python
model = SuperGLM(
    family="poisson",
    selection_penalty=0.0,
    features=features,
)
model.fit_reml(X, y, sample_weight=exposure)
```

SuperGLM does roughly this:

```text
1. Build feature blocks
   - splines become basis + penalty + SSP transform
   - categoricals become integer-coded lookup blocks
   - interactions become structured product blocks

2. Start with initial smoothing parameters lambda

3. For each REML outer step:
   a. build S(lambda)
   b. run IRLS until the GLM coefficients stabilise
      - compute eta, mu
      - compute W and z
      - build blockwise XtWX, XtW1, XtWz
      - solve the augmented penalised WLS system
   c. compute H, H_inv, EDF, logdet pieces
   d. compute REML objective, gradient, Hessian
   e. update log(lambda) by a damped Newton step with line search

4. Rebuild SSP with the final lambdas

5. Final refit and canonicalise public prediction state
```

## Using SuperGLM: common pricing snippets

### Frequency as rate target with exposure weights

Use this when your target is claim frequency, i.e. claims per unit exposure.

```python
import numpy as np
from superglm import Categorical, Numeric, Spline, SuperGLM

X_train = train[["DrivAge", "VehAge", "BonusMalus", "Area", "LogDensity"]]
y_train = train["ClaimCount"].to_numpy() / train["Exposure"].to_numpy()
w_train = train["Exposure"].to_numpy()

features = {
    "DrivAge": Spline(kind="ps", k=14, knot_strategy="quantile_rows", select=True),
    "VehAge": Spline(kind="cr", k=10, knot_strategy="quantile_rows", select=True),
    "BonusMalus": Spline(kind="cr", k=12, knot_strategy="quantile_tempered"),
    "Area": Categorical(base="most_exposed"),
    "LogDensity": Numeric(),
}

model = SuperGLM(
    family="poisson",
    selection_penalty=0.0,
    features=features,
)

model.fit_reml(
    X_train,
    y_train,
    sample_weight=w_train,
    max_reml_iter=30,
)

rate_pred = model.predict(score_df[X_train.columns])
```

Here, `sample_weight` means exposure / frequency weight. A row with 0.5 exposure
contributes half as much information as a row with 1.0 exposure.

### Claim count target with log-exposure offset

Use this when your target is raw claim count and you want the model to estimate a
rate while the offset carries exposure.

```python
import numpy as np
from superglm import Categorical, Numeric, Spline, SuperGLM

X_train = train[["DrivAge", "VehAge", "Area", "LogDensity"]]
y_count = train["ClaimCount"].to_numpy()
exposure = np.clip(train["Exposure"].to_numpy(), 1e-12, None)
offset = np.log(exposure)

features = {
    "DrivAge": Spline(kind="ps", k=14, select=True),
    "VehAge": Spline(kind="cr", k=10, select=True),
    "Area": Categorical(base="most_exposed"),
    "LogDensity": Numeric(),
}

model = SuperGLM(
    family="poisson",
    selection_penalty=0.0,
    features=features,
)
model.fit_reml(X_train, y_count, offset=offset, max_reml_iter=30)

score_exposure = np.clip(score_df["Exposure"].to_numpy(), 1e-12, None)
pred_count = model.predict(score_df[X_train.columns], offset=np.log(score_exposure))
pred_rate = pred_count / score_exposure
```

Do not accidentally use exposure twice. For a basic frequency model, use either
rate target + exposure weight, or count target + log-exposure offset. Only add
both when they have distinct meanings in your modelling setup.

### Large data: discrete REML

Use `discrete=True` when row count is large and spline bases would otherwise
force repeated full-row matrix work.

```python
model = SuperGLM(
    family="poisson",
    selection_penalty=0.0,
    features=features,
    discrete=True,
    n_bins=256,
)
model.fit_reml(X_train, y_train, sample_weight=w_train, max_reml_iter=30)
```

Discrete mode keeps basis values at support/bin points and uses row-to-bin
indices. That is why `gram(W)` can aggregate weights by bin first rather than
multiplying every original row every time.

### Interaction screening

Interactions increase coefficient count and cross-Gram work. For candidate
searches, use the faster candidate mode first, then refit the chosen model in the
full mode.

```python
model = SuperGLM(
    family="poisson",
    selection_penalty=0.0,
    features=features,
    interactions=[("DrivAge", "VehAge"), ("DrivAge", "Area")],
    discrete=True,
    n_bins=256,
)

model.fit_reml(
    X_train,
    y_train,
    sample_weight=w_train,
    interaction_mode="fast_candidate",
    max_reml_iter=20,
)
```

### When to try the QR solver

Use the QR solve path as a diagnostic or small-data stability option when the
Gram/Cholesky route repeatedly falls back to SVD.

```python
model = SuperGLM(
    family="poisson",
    selection_penalty=0.0,
    features=features,
    direct_solve="qr",
)
model.fit_reml(X_train, y_train, sample_weight=w_train)
```

This is not the default because it materialises the weighted augmented design.
It is useful when you want stability more than speed, or when you are trying to
confirm that a warning is a conditioning issue rather than a modelling bug.

### Inspect the fitted model

```python
print(model.summary(detail="compact"))

metrics = model.metrics(X_train, y_train, sample_weight=w_train)
print(metrics.summary())

age = model.term_inference("DrivAge", with_se=True)
print(age.edf)
print(age.spline.interior_knots)

rels = model.relativities(with_se=True)
area_rels = rels["Area"]

telemetry = model.training_telemetry()
reml = model.reml_diagnostics()
```

Useful mental mapping:

- `summary()` tells you what was fitted.
- `metrics()` tells you how the model performed on a dataset.
- `term_inference()` tells you what one term looks like.
- `relativities()` gives deployment/reporting-friendly term tables.
- `training_telemetry()` and `reml_diagnostics()` are plain Python payloads for audit/logging systems.

### Export rating tables

```python
model.export_rating_tables(
    "rating_tables.xlsx",
    X_train,
    y_train,
    sample_weight=w_train,
    n_bins=150,
)
```

For source-aware term offsets, pass the raw source value as well as the link-scale
offset:

```python
term_months = train["term_months"].to_numpy()
offset = np.log(term_months / 12.0)

model.fit_reml(X_train, y_train, sample_weight=w_train, offset=offset)

model.export_rating_tables(
    "rating_tables.xlsx",
    X_train,
    y_train,
    sample_weight=w_train,
    offset=offset,
    offset_source=term_months,
    offset_name="Term",
)
```

The fitted model still scores with the link-scale offset. The exported rating
table can use the raw deployment value, such as `Term = 12` or `Term = 36`, when
that raw value maps cleanly to a multiplier.

## Choosing the right fit path

| Modelling need | Use |
|---|---|
| Normal pricing GAM with splines | `selection_penalty=0.0` + `fit_reml()` |
| Let smooth terms shrink toward zero in REML | `Spline(..., select=True)` |
| Very large row count | `discrete=True`, often `n_bins=256` as a starting point |
| Candidate interaction search | `fit_reml(..., interaction_mode="fast_candidate")`, then full refit |
| Group-lasso sparse screening | `fit()` or `fit_path()` with `selection_penalty > 0` |
| Repeated SVD fallback / conditioning diagnosis | try `direct_solve="qr"` on a smaller fit, then simplify or stabilise the design |
| Deployment to native Python scoring | pickle the fitted `SuperGLM` object |
| Deployment to workbook/rating-table workflow | `model.export_rating_tables(...)` |

## Debugging solver warnings by meaning

| Symptom | What it usually means | Practical response |
|---|---|---|
| Repeated SVD fallback | \(\mathbf{H}\) is nearly singular or badly conditioned | Check duplicate features, sparse factor levels, too many knots, or interactions that duplicate main effects. |
| Huge coefficient movement but little deviance movement | A weakly identified direction is moving | Prefer deviance convergence, simplify the term, or inspect thin levels. |
| QR works but Gram solve struggles | Normal equations amplified conditioning | Use QR as a check; then fix the design or keep QR for small fits. |
| EDF looks too high | Smoothness too low or basis too flexible | Use REML, `select=True`, fewer knots, or inspect exposure density. |
| EDF collapses near zero | REML selected heavy shrinkage | This can be correct; inspect the term plot and business meaning before forcing it back in. |
| Interaction fit is much slower | Cross-Grams and coefficient count grew | Screen interactions, use `discrete=True`, or reduce candidate set. |

## One-screen summary

```text
Model:
    eta = alpha + X beta + offset
    mu  = inverse_link(eta)

Spline:
    x -> knots -> raw basis B
    B -> roughness penalty Omega
    QR removes constraint / intercept-duplicate directions
    eig splits null vs wiggle directions for select=True
    SSP builds R_inv so solver sees X_g = B @ R_inv

Categorical:
    levels -> integer codes
    matvec is lookup
    rmatvec and gram are bincounts

IRLS:
    current eta, mu -> working W and z
    build XtWX, XtW1, XtWz blockwise
    solve augmented penalised WLS system

Robust solve:
    Cholesky -> pivoted Cholesky -> SVD
    optional QR solve materialises weighted augmented design

REML:
    S(lambda) = blockdiag(lambda_j Omega_j)
    H = XtWX + S(lambda)
    objective balances fit, beta.T S beta, log|H|, and log|S|_+
    gradient/Hessian ask how the objective changes if log(lambda_j) moves
    Newton step is eig-stabilised and line-searched

Finalisation:
    rebuild SSP at final lambdas
    map beta to the new basis
    final refit
    canonicalise prediction/export state
```

The high-level reason for all of this: SuperGLM needs to fit flexible tariff
curves, keep them stable, estimate smoothness automatically, support audit-ready
term views, and still run fast enough on insurance-pricing data. The matrix
operations are the machinery that makes those requirements compatible.
