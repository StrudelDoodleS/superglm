# Severity families: declared laws with generated derivatives, and incomplete-gamma bounds

2026-10-31, against master `084cc96d` and PR 545 at `90b0eb46` (unmerged, held). Design
note only: no production code changes. The prototype behind every measurement is in
[`severity-family-prototype/`](severity-family-prototype/), and the commands that produced
the numbers are listed at the end.

## Recommendation

1. **Generate severity families from a declared log density, at build time.** SymPy
   differentiates the declaration and emits numba row kernels. The kernels are checked
   in, with a `--check` test that fails when they are stale. superglm already works
   this way for fourth derivatives (`tools/generate_lss_fourth_derivatives.py`, PR #497).
   This extends it from derivative blocks to the whole row kernel. SymPy stays a dev
   dependency; at runtime only numba and scipy run.
2. **Put the numerical care in a small library of stable primitives, not in each
   family.** Examples are `softplus`, `log1mexp` and the gamma normaliser. Each primitive
   has a derivative rule, a numba body that handles its own tails, and an mpmath oracle.
   Declarations must be written in log space with these primitives.
   - Written that way, every generated derivative of four test families and the gamma
     (orders 0 to 4, plus log survival and log CDF to order 2) is within 15 u of a
     60-digit reference, in the body and out to 1e-100 tails. The one exception is
     benign (section 2.4).
   - The same laws written in textbook form lose up to 1e4 u in the body, or return
     ±inf in the tails.
   - The generated oracle grid flags the difference with no hand-written test.
3. **Each new family needs about 10 declared lines, against about 1,850 production lines
   for the hand-written `GammaLS`.** One generic adapter implements the family contract
   for all of them. Censoring and truncation come free for any family with a closed-form
   CDF: the generated log-tail channels feed PR 545's window combiner.
4. **No JAX, and no finite or complex-step differences** (section 2.6).
5. **Keep hand-written code for what a declaration cannot carry:** prior-weight
   (reproductive) laws, Fisher information, certified expected shortfall, kinks, and
   series-defined densities such as the Tweedie.
6. **Issue 544 stage 2 (gamma bounds) should adopt the standard method.** That is Gil,
   Segura and Temme (2012) on the log scale, carrying a Taylor jet in the shape through
   its series and continued fraction. The sweep's float64 version of this
   (section 3.4) is accurate to about 1e-13 typically at every order to 4, and to
   2.5e-10 at worst for orders 1 and 2. Build it as one jet primitive, so the gamma
   and generalized-gamma bounds also come through the generator. The generalized gamma
   additionally needs Temme's uniform expansion. That part is standard but untested in
   the shape.

**Measured speed.** The comparison is `GammaLS` with only its order-0-2 row kernel
swapped for the generated one; plan, links, orders 3-4 and every protocol stay
hand-written.

| | Hand-written `GammaLS` | Generated kernel |
|---|---|---|
| row kernel, order 2, 10^6 rows, one thread | 628 ns/row | 153 ns/row |
| row kernel, order 0 (backtracking screens) | 123 ns/row | 34 ns/row |
| complete `fit_reml`, 200,000 rows, `mean ~ s(x1) + s(x2)`, `scale ~ s(x3)`, default threads, three alternating runs | 4.76, 4.54, 4.46 s | 3.38, 3.29, 3.30 s |
| of which likelihood evaluation (27 calls) | 2.07, 2.00, 1.96 s | 0.89, 0.85, 0.86 s |
| outer iterations, coefficient fits, inner iterations | 6, 7, 13 | 6, 7, 13 |
| log-likelihood, EDF | -1,927,989.361974, 13.986938 | identical to the printed digits |

- **Predicted parameters agree to 3.7e-14.** The row kernels agree to 6e-16 (value),
  5e-12 (score) and 3e-11 (Hessian), relative to each row's largest entry.
- **Neither kernel is the more accurate on the rows where they differ most.** Those
  rows sit where the CV score passes through zero, and against mpmath both are equally
  far off relative to that tiny row scale.
- **The generated kernels of the new families cost 30-68 ns/row at order 2 in scale
  form.** The mean-form Weibull, which calls polygamma, costs 150. Log tails cost
  22-191 ns/row, and orders 3 and 4 together 37-412.

## 1. What the problem is

Each row contributes `l(y; theta)` with two or three natural parameters. The fit needs
the value and the packed score and Hessian in `theta`, and optionally the third and
fourth derivatives. Censored and truncated rows need the same channels for `log S`,
`log F` and `log(S(a) - S(b))`. In a numerical analyst's terms:

- **The object.** A closed-form scalar function of three to four real variables, plus a
  few special functions. Its derivatives are needed to fourth order at about 10^6 points
  per evaluation.
- **The textbook answer.** Symbolic or source-transform differentiation with
  common-subexpression elimination, compiled once. Reverse-mode AD pays off with many
  inputs and one output; here every point has few inputs and needs every
  higher-order partial.
- **The difficulty is not the calculus. It is floating point in the tails.** The
  oracle runs below show that differentiating a textbook formula correctly is not
  enough. `1 - (1 + x^c)^(-k)` loses the Burr XII left tail. Higher derivatives of
  `log1p(y/scale)` overflow at large `y`. The direct gamma `a log a - lgamma(a)`
  loses four digits at a coefficient of variation of 0.02. Writing the law well is the
  hard part, and that is why the stability work belongs in reusable primitives.

What superglm already has:

- the SymPy fourth-derivative generator above. Its design rule is that every special
  function a family needs is an argument computed by the kernel's stable routines, and
  only the chain and product rules are generated;
- `tests/test_family_derivatives_symbolic.py`, a SymPy and mpmath oracle for every
  family's derivative channels;
- `families._base.natural_curvature_direction` and
  `natural_curvature_second_direction`, which compose a family's natural third and
  fourth derivatives with its links;
- in PR 545, `kernels/_response_tails.log_window_channels`, which turns the two ends'
  `log T` channels into the window's log probability, score and Hessian.

The declared path assembles these pieces. It is not a new kind of machinery.

## 2. Half 1: the declared path

### 2.1 What a family declares

The whole per-family input for Burr XII, from
[`declarations.py`](severity-family-prototype/declarations.py):

```python
def _burr_z(y, scale, c):
    return c * log(y / scale)

BURR12 = Declaration(
    name="burr12",
    parameters=("scale", "c", "k"),
    log_density=lambda y, scale, c, k: (
        log(c) + log(k) - log(y) + _burr_z(y, scale, c) - (k + 1) * softplus(_burr_z(y, scale, c))
    ),
    log_survival=lambda y, scale, c, k: -k * softplus(_burr_z(y, scale, c)),
    log_cdf=lambda y, scale, c, k: log1mexp(k * softplus(_burr_z(y, scale, c))),
    scipy_law=lambda scale, c, k: stats.burr12(c=c, d=k, scale=scale),
)
```

Weibull, log-logistic and Lomax take 10 lines each, and the gamma in `GammaLS`'s own
(mean, coefficient of variation) parameters takes 8. The mean-parameterised Weibull,
`log scale = log mean - lgamma(1 + 1/shape)`, takes 15. That shows mean forms with
gamma functions work too, which matters because actuarial relativities sit on the mean.

### 2.2 What is generated

[`sevgen.py`](severity-family-prototype/sevgen.py) produces three things.

**One numba module per family:**
- the fused density kernel at orders 0, 1 and 2;
- the natural third and fourth derivatives, packed;
- the log survival and log CDF at orders 0 and 2;
- the parameter-independent carrier. These are the additive terms free of every
  parameter (`-log y`), split off as the hand-written families split them.

Common subexpressions are shared within each kernel. Generation takes 0.03-0.2 s per
family for derivation and 0.04-0.22 s for printing, and produces 113-248 lines.

**A generic adapter** ([`declared_family.py`](severity-family-prototype/declared_family.py),
142 lines, written once):
- log links on positive parameters;
- refusal of non-unit prior weights, because the law is not reproductive, as
  `LogNormalLS` does;
- frequency weights as replication;
- initialisation from a scipy fit on a subsample;
- the mean, CDF and quantile from scipy.

**An oracle check** ([`oracle_check.py`](severity-family-prototype/oracle_check.py)). Every
declaration gets this test unchanged. It uses three independent references:
1. scipy's `logpdf`, `logsf` and `logcdf`, which shows the declaration is scipy's law;
2. the declaration's own SymPy derivatives evaluated by mpmath at 60 digits, which
   checks the derivation, the printed code and the primitives;
3. Richardson central differences of scipy's `logpdf`, which checks the derivatives
   with no SymPy involved.

Points cover the body (quantiles 0.001 to 0.999) and the tails (`F` or `S` of 1e-12,
1e-30 and 1e-100), with shapes from 0.25 to 25 and coefficients of variation from 0.02
to 5.

### 2.3 The primitive library

[`sevprims.py`](severity-family-prototype/sevprims.py) defines eight primitives. Each is a
SymPy function with an `fdiff` rule, a numba body, and an mpmath reference. So every
generated derivative is again a product of primitives.

| Primitive | Value | Derivative | Tail handling in the numba body |
|---|---|---|---|
| `softplus(z)` | `log(1 + e^z)` | `expit(z)` | split at `z = 0` |
| `expit(z)` | `1/(1 + e^-z)` | `expit(z) expit(-z)` | split at `z = 0` |
| `log1mexp(t)` | `log(1 - e^-t)` | `invexpm1(t)` | Mächler's switch at `log 2` |
| `invexpm1(t)` | `1/(e^t - 1)` | `-r(1 + r)` | `expm1` |
| `gdev(y, m)` | `y/m - 1 - log(y/m)` | `(y - m)/(y m)`, `(m - y)/m^2` | series in `(y - m)/m` near 1 |
| `gnorm(a)` | `a log a - a - lgamma(a)` | `-psires(0, a)` | Stirling for `a >= 10` |
| `psires(n, a)` | `psi^(n)(a)` less the n-th derivative of `log a` | `psires(n + 1, a)` | asymptotic residual for `a >= 10` |
| `polygamma`, `digamma`, `lgamma` | — | SymPy's own rules | scipy `psi` via its Cython entry point; recurrence plus asymptotic series |

Against mpmath, every primitive is within 6e-15 relative error for `a` from 0.01 to 1e6
and `t` from -0.9 to 5.

The rule each primitive follows is the one the hand-written kernels already follow:
take the singular or cancelling part out analytically, and evaluate the regular
remainder directly.
- `GammaLS`'s scaled digamma residual `a (psi(a) - log a)` is this, and so is PR 545's
  Mills-ratio continued fraction for `x + r`.
- `gdev` exists because the measurement found a failure. With the one-argument
  `dev(t)`, `t = (y - m)/m`, the derivative `t/(1 + t)` divided by zero when `y/m`
  underflowed below `u`: `1 + t` was exactly 0 at a coefficient of variation of 5 and
  the 1e-12 quantile.
- The numba row loops use `error_model="numpy"`, so a non-finite channel marks its row
  invalid instead of raising.

SymPy's own rewriter, `sympy.codegen.rewriting.optims_c99`, is the alternative to a
primitive library. It rewrites patterns, `exp(x) - 1` into `expm1(x)` and
`log(1 + x)` into `log1p(x)`, and only where the expression literally has that shape
([SymPy codegen documentation](https://docs.sympy.org/latest/modules/codegen.html)).
It cannot reach the cancellations that differentiation creates, such as `1 - t(1 + r(t))`
below. Primitives with derivative rules can.

### 2.4 Accuracy, measured

Worst error over the grid, in units of `u = 2^-53`.
- **Log-likelihood values:** absolute below one and relative above, which is what a fit
  sums.
- **Derivative channels:** relative to the larger of the row's own largest entry and
  the parameter cell's body-point scale.
- **"link H":** the Hessian in `eta = log theta`, which is what a log-link solve reads.
- **Format:** body / tail.

| Family | log f | score | Hessian | link H | 3rd | 4th | log S | log S link H | log F | log F link H |
|---|---|---|---|---|---|---|---|---|---|---|
| Weibull | 2.2 / 7.2 | 4.0 / 8.5 | 4.1 / 11 | 3.8 / 8.6 | 4.4 / 10 | 5.6 / 13 | 2.3 / 6.5 | 3.6 / 7.9 | 1.8 / 1.3 | 6.0 / **230** |
| Weibull, mean form | 1.8 / 8.4 | 4.7 / 10 | 4.7 / 13 | 4.9 / 12 | 6.9 / 12 | 5.3 / 15 | 3.3 / 8.6 | 5.1 / 12 | 2.0 / 1.3 | 6.7 / **280** |
| Burr XII | 1.9 / 2.4 | 1.3 / 2.5 | 1.2 / 2.6 | 1.6 / 3.2 | 1.5 / 2.8 | 3.5 / 1.3 | 2.1 / 1.5 | 1.6 / 1.4 | 1.7 / 2.0 | 12 / **230** |
| log-logistic | 4.9 / 2.8 | 2.1 / 1.6 | 2.1 / 0.5 | 1.6 / 1.6 | 3.1 / 0.6 | 3.5 / 0.6 | 2.1 / 1.3 | 1.3 / 0.7 | 2.5 / 1.3 | 2.0 / 0.8 |
| Lomax | 1.9 / 15 | 1.0 / 1.5 | 1.0 / 1.0 | 2.2 / 1.5 | 0.7 / 0.7 | 1.1 / 1.1 | 1.5 / 1.4 | 1.3 / 1.4 | 1.9 / 1.6 | 6.3 / 6.4 |
| gamma (mean, CV) | 2.0 / 7.3 | 3.0 / 3.1 | 3.5 / 3.5 | 2.4 / 3.4 | 3.2 / 3.3 | 2.2 / 3.3 | — | — | — | — |

**The independent finite-difference check agrees.** It differences scipy's `logpdf`
directly, with no SymPy. The score agrees to 1.4e-12 to 1.8e-8 relative, and the
Hessian to 1.2e-9 to 5.7e-7, with the gamma at CV 0.02 the loosest. That is the
differencing error itself, so it confirms the formulas without being able to resolve
them to `u`.

**The bold entries are the one cancellation the primitives do not remove.** In the far
left tail, `d^2 log F/dk d(scale)` contains `1 - t(1 + 1/expm1(t))`, which is about
`-t/2` as `t -> 0`. It costs at most 280 u, or 3e-14 relative, on rows whose `F` is
1e-100. Nothing a user reads moves. A primitive for `t/(1 - e^-t) - 1` (a Bernoulli
series) would remove it, if a stricter gate ever wants that.

**The same laws in textbook form**, through the same generator and the same oracle,
with the mpmath reference at 400 digits so it resolves its own cancellation:

| Declaration | log f | score | Hessian | 3rd / 4th | log S | log F | log F link H |
|---|---|---|---|---|---|---|---|
| gamma, `a log a - lgamma(a) + (a - 1) log y - a log mu - a y/mu` | 1.0e4 / 1.3e3 | 1.0e4 / 1.0e3 | 9.8e3 / 2.8e2 | 9.5e3 / 7.4e2 | — | — | — |
| Burr XII, `log(1 + x^c)` and `log(1 - (1 + x^c)^-k)` | 2.4 / 1.8 | 1.4 / 2.4 | 55 / 4.2e2 | 1.2e3 / **inf** | 2.2 / 1.2 | 2.9e2 / **inf** | 2.0e4 / **inf** |
| Lomax, `log(1 + y/scale)` | 6.3 / 2.2 | 4.0 / 1.5 | 1.0 / 1.1 | 0.7 / **inf** | 24 / 7.2 | 3.2e3 / **inf** | 5.5e4 / **inf** |
| Lomax, `log1p(y/scale)` | 1.4 / 1.1 | 0.7 / 1.5 | 1.0 / 1.1 | 0.7 / **inf** | 0.9 / 1.3 | 0.8 / 0.4 | 7.1 / 6.2 |

- **inf** means the generated kernel returned a non-finite channel: `log 0` in the left
  tail, or `y^4` overflowing in a fourth derivative at the 1e-100 right tail.
- The `log1p` form of Lomax is accurate to order 2, but overflows at orders 3 and 4.
  Writing it as `softplus(log(y/scale))`, which is Burr XII at `c = 1`, bounds every
  derivative.
- The gamma's direct normaliser loses four digits at CV 0.02 (`a = 2500`).

**scipy's own log functions are not a likelihood engine.**
- `gamma.logpdf` is 5.7e3 u off at CV 0.02.
- `burr12.logcdf` is 290 u off in the body and returns `-inf` at `F = 1e-100`.
- `fisk.logsf` (log-logistic) is 31 u off in the body and non-finite in the deep
  right tail.

So the likelihood, censoring and truncation must use the generated tails. scipy serves
the prediction side (`cdf`, `ppf`, `mean`), where a probability-scale value is read
directly.

**Censoring and truncation from the same declaration.**
[`window_check.py`](severity-family-prototype/window_check.py) feeds the generated
`log S` or `log F` channels at a window's two ends through PR 545's documented
combination. That combination is reproduced from its stated formulas, because the code
is unmerged. It compares the result with SymPy's derivatives of `log(S(a) - S(b))`
evaluated at 100 digits. Errors are relative to each window's own largest entry:

| Window (probabilities) | Weibull | Burr XII | log-logistic | Lomax |
|---|---|---|---|---|
| (0.5, 0.9), (0.9, 0.999), (0.999, 1 - 1e-9): value / score / Hessian | ≤ 3.1 / 6.0 / 5.4 | ≤ 3.3 / 2.9 / 4.9 | ≤ 1.5 / 2.5 / 3.4 | ≤ 1.8 / 2.6 / 6.1 |
| (1e-9, 1e-3): value / score / Hessian | 0.4 / 0.8 / **1.7e3** | 1.0 / 0.7 / **4.9e2** | 0.6 / 2.8 / 3.7 | 0.2 / 0.5 / 1.2 |
| (0.3, 0.3 + 1e-4): value / score / Hessian | 1.3e3 / 1.4e4 / 1.0e4 | 1.3e3 / 5.8e3 / 1.8e4 | 4.7e2 / 6.5e3 / 7.6e3 | 3.6e2 / 4.4e3 / 8.7e3 |

- **The left-tail Hessians in bold** (at most 1.9e-13 relative) are the log-CDF
  cancellation above.
- **The narrow window loses about `u/width` (1e4 u, or 1e-12).** That is the differencing of two
  log tails, as PR 545's docstring states. It is why PR 545 integrates any window
  narrower than 0.05, on the density's own scale, by three-point Gauss-Legendre
  quadrature of the density, which this check did not apply.
- **That quadrature needs only the density channels,** which are generated too. So a
  declared family gets the whole `ResponseWindowFamily` contract from its declaration
  and PR 545's shared code, with no per-family bounds code.

**Declared families fitted through `SuperLSS`.**
[`fit_declared.py`](severity-family-prototype/fit_declared.py) fits 100,000 simulated rows
whose true log scale is `8 + 0.4 sin(2 pi x1)`. It uses the generic adapter and no
family-specific code.

| | Weibull (true shape 0.8) | Burr XII (true c 1.5, k 2.0) |
|---|---|---|
| intercept-only parameters against scipy maximum likelihood (BFGS on `scipy.stats` logpdf, `gtol` 1e-10) | within 4.4e-8 | within 4.2e-7 |
| intercept-only log-likelihood, superglm / scipy | -910,722.9714437132 / -910,722.9714437132 | -877,741.6229305915 / -877,741.6229305919 |
| `fit_reml`, `scale ~ s(x1)`: converged, EDF | yes, 11.16 | yes, 12.59 |
| fitted shape parameters | 0.8004 | 1.490, 2.024 |
| worst relative error of the fitted scale curve | 1.7% | 2.7% |
| wall time (4 cores) | 0.51 s | 0.81 s |

Where the parameters differ, scipy's optimiser stopped short: superglm's
log-likelihood is equal or higher.

### 2.5 Development cost

| | Production lines | Test lines | History |
|---|---|---|---|
| `GammaLS` (hand-written) | 570 family + 1,278 kernel | about 5,500 (`test_gamma_ls_family`, `_vertical_slice`, `_joint_infinity_face`, `_series_execution`, `_working_arithmetic`, `_fisher_data_reuse`, `_gamma_lss_oracles`) | 8 commits touch the kernel, including PRs #491 and #497, which added orders 3 and 4 |
| `GeneralizedGammaLSS` (hand-written) | 561 + 2,066 | about 1,700 | 9 commits |
| `GeneralizedParetoLSS` (hand-written) | 452 + 636 | about 1,340 | 6 commits |
| PR 545 bounds, per family | 115-118 family + 25-144 kernel | shared 1,406-line suite | shared 207 + 340 lines |
| Declared family | 8-15 declared lines, plus a scipy mapping and a starting rule (one line each in the prototype; probably 15-25 lines with names, links and supports in production) | the shared oracle grid, plus one public fit test | — |
| Declared path, one-time | generator 367 + primitives 257 + adapter 142 + oracle 247 = 1,013 prototype lines | — | — |

The one-time cost is about one hand-written kernel. Every family after the first is
nearly free. That includes orders 3 and 4, which took two PRs to hand-write for the
existing families. What a new family still costs is judgement: choosing a log-space
form, its parameter domains and its starting values, and reading the oracle grid when
it flags a channel.

### 2.6 The alternatives, measured

A separate sweep measured the alternatives on the gamma (mean, CV) row. It used 10^6
rows on one pinned core through `benchboard run-timed`, and 21 rows against a 50-digit
reference: 15 typical and 6 adversarial, with CV of 0.02 and 5 and `y/mu` from 1e-6 to
1e3. Its harness is separate from this note's, so compare its rows with each other, not
with section 2.4's.

| Method | ns per row | Worst relative error, Hessian (typical / adversarial) |
|---|---|---|
| one log-density evaluation | 28 | — |
| numba closed form, textbook | 67 | 3.7e-13 / 1.6e-10 |
| numba closed form, with stable helpers | 98 | 1.6e-14 / 2.1e-15 |
| JAX `jit(vmap(grad + hessian))`, including transfer | 258 | 8.7e-13 / 1.5e-11 |
| central differences (13 evaluations) | 156 | 1.5e-5 / 2.9e-3 |
| Richardson differences (25 evaluations) | 298 | 1.0e-7 / 2.6e-6 |
| complex step for the score, then differences (16 complex evaluations) | — | 3.2e-10 / 4.0e-8 |
| hyper-dual numbers | not timed | 2.5e-13 / 4.7e-11 |

**JAX.**
- **Dependency cost:**
  - about 340 MB installed (jaxlib alone is 322 MB), against 189 MB for numba and
    llvmlite;
  - a 0.45 s cold import;
  - a 0.7-0.8 s recompile for every new input length, unless rows are padded to fixed
    sizes;
  - no Windows ARM64 or macOS x86-64 wheels in the current release.
- **float64** needs the process-wide `jax_enable_x64` flag, which a library cannot own.
  The scoped context manager is thread-local, and `jnp.asarray` of a float64 array
  silently returns float32.
- **Interop:** `@njit` cannot call JAX, so it would be a separate Python-level batch
  step.
- **Accuracy:** JAX's own `log_ndtr` is 2e-11 off at `x = -20`, where scipy's is
  1e-15. `gammainc` has no second derivative in the shape: it raises
  `NotImplementedError`.
- **Speed:** on one core it is 2.6-3.9 times slower than a numba closed form.
- **AD differentiates the formula as written,** so it inherits the textbook gamma's
  four-digit loss at CV 0.02, like any other route. The AD literature says the same:
  Baydin et al. (2018), Hückelheim et al. (2023), and Carpenter et al. (2015) on Stan,
  whose built-in densities use analytic partials for this reason.

**Finite differences** give Hessians near 1e-5 to 1e-7, 3e-3 at worst, at 2-6 times the
cost of a closed form. Newton on such a Hessian converges, but the standard errors and
EDF would carry that error. superglm already differences orders 3 and 4 with a
certificate, and that remains the fallback. It is not a way to get orders 1 and 2.

**The complex step** gives exact first derivatives, but not second ones. Exact second
derivatives need hyper-dual or multicomplex arithmetic (Fike and Alonso 2011; Lantoine
et al. 2012), which carry the same psi and trigamma burden as the closed form. scipy's
`gammaln`, `polygamma`, `gammainc` and `gammaincc` reject complex input, and numba
has no complex `lgamma`. So the complex step cannot reach the gamma families at all.

**Hand-written code stays accurate and fast,** but costs the lines in section 2.5 for
every family. A generated kernel is the same closed form, without the hand algebra.

**The comparison's main finding is the same as this note's:** accuracy comes from the
cancellation-free reformulation (the helpers for `log a - psi(a)`, `1/a - psi'(a)` and
`log r + 1 - r`), not from how the derivatives are produced. SymPy generation with a
primitive layer gives exactly those helpers, at closed-form speed.

### 2.7 What the declared path does not cover

- **Prior-weight laws.** `GammaLS` and `TweedieLSS` reweight the law itself (shape
  `w a`, dispersion `phi/w`). A declaration could carry that substitution as a field,
  but the prototype refuses non-unit prior weights, as `LogNormalLS` and
  `GeneralizedGammaLSS` do.
- **Expected information.** `ExpectedInformationFamily` needs expectations, generally
  not closed form. Declared families use observed curvature with the documented
  nearest-semidefinite fallback.
- **Certified expected shortfall.** scipy's `expect` is quadrature, not a certified tail
  mean, so `ExpectedShortfallFamily` needs a declared closed form. The Lomax and
  generalized Pareto have one; the Weibull's is an incomplete gamma, which is stage 2.
- **Kinks.** The two-piece families are only `C^1`. A generated third derivative there
  is wrong at the join, and the differenced route is the correct one. A declaration
  must state that it is smooth everywhere its links reach. The generator cannot
  check this.
- **Non-closed-form CDFs.** The gamma and generalized gamma CDFs are incomplete gamma
  functions. They join through one jet primitive (section 3.6).
- **Arithmetic hardening.** `GammaLS` refuses a channel that is not representable,
  using exact `frexp` products. The generated kernels instead mark the row invalid
  and leave the line search to back away. The oracle grid is the evidence that this
  happens only outside the representable range, not inside it.
- **Compile time.** Without numba's on-disk cache, a family's eight kernels compile in
  0.9-1.4 s in all ([`compile_time.py`](severity-family-prototype/compile_time.py),
  one core). Orders 0-2 alone take 0.3-0.6 s. That is a first-fit cost of the same kind
  PR #502 cut. Checked-in kernels with `cache=True` and `warmup()` coverage pay it once
  per install. The prototype compiles with `cache=False` because it writes its modules
  at run time.

## 3. Half 2: issue 544 stage 2, censoring and truncation for the gamma families

### 3.1 The problem

For `GammaLS`, a censored or truncated row needs `log P(a, x)`, `log Q(a, x)` or
`log(P(a, x_hi) - P(a, x_lo))`. Here P and Q are the regularised lower and upper
incomplete gamma functions, `a = 1/cv^2` is the shape, and `x = y a/mu`. Both vary per
row.

- Every partial involving `x` is closed form, because `dP/dx` is the gamma density.
- Only the pure shape derivatives `d^k log P/da^k` and `d^k log Q/da^k` need an
  algorithm: to order 2 at least, and order 4 for the analytic endgame.
- The generalized gamma uses the same functions. Write its shape as `q`, to keep it
  apart from the upper function Q. For `q > 0`, `F(y) = P(1/q^2, e^{q w}/q^2)` with
  `w = (log y - location)/sigma`. For `q < 0`, P and Q swap. At `q = 0` the law is
  log-normal.

### 3.2 The standard method and its assumptions

The sources were read in full unless marked otherwise in the list at the end.

**Evaluating P and Q in float64 is settled.**
- **Gautschi (1979, ACM TOMS 5(4)).** One function is primary and the other comes by
  complement, which loses at most one bit. Q is primary when `a <= x + 1/4`.
  - P comes from the all-positive series `x^a e^-x sum x^n/Gamma(a + n + 1)`.
  - Q comes from Legendre's continued fraction for `x > 1.5`, and from a split `u + v`
    below that.
  - He says the method strains for `a ~ x >> 1`. He gives no derivatives in `a`.
- **Gil, Segura and Temme (2012, SIAM J. Sci. Comput.; arXiv:1306.1754).** This is the
  current standard. It has four domains:
  - the Taylor series for P;
  - Gautschi's `u + v` for small `x`;
  - the continued fraction for Q;
  - Temme's uniform asymptotic expansion for `a >= 12` and `0.30 a <= x <= 2.35 a`.
    Near the diagonal the continued fraction "becomes rather slow", and the expansion
    needs at most 25 terms.

  They write the prefactor `x^a e^-x/Gamma(a + 1)` as `exp(-a eta^2/2)/(sqrt(2 pi a) Gamma*(a))`
  and test the scaled functions. That is exactly the log-scale form a likelihood needs.
  They report a maximum relative error of 1.7e-15 on `(0, 1]^2`, 7.9e-13 on `(0, 500]^2`
  and 4e-14 for the uniform expansion. They give no derivatives in `a`.
- **DiDonato and Morris.** The method is in ACM TOMS 12(4), 1986 (abstract only; not
  accessible). The Fortran is Algorithm 654, ACM TOMS 13(3), 1987, read in full. It
  targets 14 digits and is built on Temme's expansions (DLMF 8.25).

**Derivatives in the shape: order 1 is settled; orders 3 and 4 have a method but no
float64 routine.**
- **Moore (1982), Algorithm AS 187, Applied Statistics 31(3).** Orders 1 and 2 by series
  and continued fraction. Only its R wrappers' documentation was available (VGAM, GPL-3;
  heavy). They add that it can overflow when `x >> a`.
- **Figurnov, Mohamed and Mnih (2018, arXiv:1805.08498), appendices B and C.**
  Forward-mode differentiation of Bhattacharjee's AS 32 as implemented in Eigen, at about
  twice the CDF's cost. Its 5.4e-13 is the mean absolute error of the reparameterisation
  gradient on bulk samples, not a pointwise relative error for `dP/da`.
- **Stan Math, JAX, TensorFlow and Eigen** document order 1 only. Stan states an
  absolute accuracy of 1e-10 on `a` in `[0, 20]`, `z` in `[0, 30]`. Its documentation
  says the upper-function gradient uses an asymptotic series rather than Gautschi's
  continued fraction.
- **Johansson (2021, arXiv:2110.10583), read in part.** This is the one published
  arbitrary-order method. It substitutes `a -> a + X` and works in truncated power
  series `C[[X]]/X^(j+1)`, with rigorous truncation bounds. It is arbitrary precision
  (Arb), not float64.

**Corrections to PR 545's 2026-10-30 sweep:**
- Gautschi gives no shape derivatives. Stan differentiated his formulas.
- Figurnov's figure is a mean absolute error on samples, not a tail accuracy.
- The DiDonato and Morris method paper and Algorithm 654 are two separate citations.
- Boost documents `lgamma_p` and `lgamma_q` (log P and log Q), which scipy does not
  expose. Boost's `gamma_p_derivative` is the derivative in `x`, not in `a`.

### 3.3 What existing software provides

Licences come from LICENSE files, package metadata or CRAN pages.

| Software | Licence | In the shape `a` | May we |
|---|---|---|---|
| SciPy 1.18 | BSD-3-Clause | nothing; no log Q | read, adapt |
| Boost.Math | BSL-1.0 | nothing (`gamma_p_derivative` is in `x`); `lgamma_p`/`lgamma_q` give log P, log Q | read, adapt |
| Stan Math | BSD-3-Clause | order 1 | read, adapt |
| JAX, TensorFlow | Apache-2.0 | order 1 | read, adapt |
| Eigen | MPL-2.0 | order 1 | read, reimplement |
| mpmath 1.3 | BSD-3-Clause | any order through `diff`: the test oracle | read, adapt, oracle |
| FLINT/Arb | LGPL-3.0 | arbitrary order, arbitrary precision | read, reimplement |
| VGAM `pgamma.deriv` | GPL-3 | orders 1-2 (AS 187) | black-box oracle only |
| R `pgamma` | GPL | value and `log.p` only | documentation only |
| heavy `pgamma.deriv` | archived from CRAN 2022; treat as GPL | orders 0-2 | black-box oracle only |
| Cephes, Abergel-Moisan code, GSL | not established | values only | do not open |

No package offers a float64 shape derivative beyond order 2.

### 3.4 Accuracy and cost, measured by the sweep

The sweep carried order-4 Taylor jets in `a` through the series and the Legendre
continued fraction, on the log scale. The prefactor `a log x - x - lgamma(a)` stays
analytic, and its jet uses polygamma. The complement uses Mächler's `log1mexp`
switch. It was tested at 42 points against mpmath, with `a` from 1e-2 to 1e4 and points
in both tails, on the series/continued-fraction switch line and on the diagonal.

| Order | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| worst relative error | 5.9e-11 | 6.4e-11 | 2.5e-10 | 3.7e-9 | 1.6e-8 |

- **Typical error is 1e-16 to 1e-13 at every order.** Every worst case is at `a = 1e4`
  within `a ± sqrt(a)`, which is a coefficient of variation of 0.01.
- **Far tails are exact to rounding.** At `a = 10`, `x = 800`, `log Q = -752.63` is right
  to 2e-16 at every order. `scipy.stats.gamma.logsf` returns `-inf` there.
- **At large `a` the value error grows like `a u`**, from cancellation in the
  prefactor. GST's `eta`/`Gamma*` form removes it.
- **Two failure modes were found and fixed:**
  - stopping the continued fraction on a per-coefficient change of `u` stops too early
    for orders 3 and 4. Stopping on each jet coefficient's own relative change fixes it,
    1.7e-10 to 2e-13;
  - just above the diagonal at large `a`, the continued fraction is ill-conditioned at
    order 4 (1.5e-7 at `a = 1e4`, `x = a + 1`, however many iterations). Hence the switch
    line `x = a + max(1, sqrt(a))`.
- **Cost:**
  - jets need 5-40% more terms than a value-only evaluation;
  - each term costs about `(K + 1)(K + 2)/2` multiply-adds per jet product. That is about
    2-3 times a value at order 1, matching Figurnov, 6 times at order 2 and 15 times at
    order 4;
  - near the diagonal the series needs about `9 sqrt(a)` terms (809 at `a = 1e4`),
    against at most 25 for the uniform expansion.

### 3.5 Regime map and recommendation

| Region | Method | Status |
|---|---|---|
| `x <= a + max(1, sqrt(a))` | series for log P on jets; log Q by `log1mexp` | measured: at most 2.5e-10 at orders up to 2 |
| `x > a + max(1, sqrt(a))` | Legendre continued fraction for log Q on jets, stopped on each coefficient's relative change; log P by complement | measured: tails to 1e-16 |
| `a >= 12`, `0.30 a <= x <= 2.35 a` | GST uniform expansion (their eq. 5.1) on jets, with the prefactor in the `eta`/`Gamma*` form | standard for values; **not built or tested in the shape**. Cuts about `9 sqrt(a)` terms to 25, and should remove the `a u` value loss and the order-3 and order-4 loss near the diagonal |
| windows | pick the tail, then `log T_big + log1mexp(log T_big - log T_small)` on jets: PR 545's combiner | not measured; Gauss-Legendre for narrow windows, as PR 545 already does |
| `a < 1e-2` | Gautschi's `u + v` | only if the shape range widens (coefficient of variation above 10) |

**For `GammaLS`,** the first two rows are enough. Severity coefficients of variation of
0.3-3 give `a` from 0.1 to 11, and censoring at a policy limit sits in the
continued-fraction tail.

**For `GeneralizedGammaLSS`, the uniform expansion is required, not optional.** With
`a = 1/q^2` and `x = a e^{q w}`, every `|q| < 0.29` puts `a` at or above 12. Then
`x - a` is about `w sqrt(a)`, which is the diagonal. Temme's `eta` there satisfies
`eta^2/2 = e^{qw} - 1 - qw`, so `eta sqrt(a) -> ±w` as `q -> 0`. The expansion's leading
term is `erfc(eta sqrt(a/2))/2`, which tends to the normal tail. So the uniform expansion
is also how the truncated likelihood reaches the log-normal limit smoothly. PR 545's risk
list names that limit, and this is the standard tool for it.

The sweep found nothing published or tested for four things:
- a float64 routine for orders 3 and 4;
- the uniform expansion differentiated in `a`;
- shape derivatives of a narrow window `log(P(x_hi) - P(x_lo))`;
- `a < 1e-2` with small `x`.

Those four are where superglm would be first. Everything else is adoption.

### 3.6 How it joins the generator

Package the engine as one jet primitive, `logq_jet(a, x) -> (J0, ..., J4)` with
`Jn = d^n log Q/da^n`, and the same for log P.

- Its SymPy rule in `a` is `J(n) -> J(n + 1)`.
- Its rule in `x` is closed form: `d log Q/dx = -h`, with
  `h = exp((a - 1) log x - x - lgamma(a) - J0)`, the gamma hazard. SymPy differentiates
  `h` through the `J` rules, so every mixed partial is generated.
- The printer calls the engine once per row and indexes the tuple, rather than once per
  `Jn`.

With this primitive, the gamma and generalized-gamma log tails are declared in a few
lines, like the Weibull's, and the window combiner does the rest. The Weibull's
expected shortfall, `scale Gamma(1 + 1/k, s^k)/S`, uses the same engine.

## 4. Staged plan and gates

Every stage meets the AGENTS.md performance-evidence rule. Fit time must be equal to or
faster than master, on a complete fit, with held-out metrics through `predict()`.

**Stage 0: PR 545 merges.** The bounds API is held by Max. Stages 1 and 2 do not need
it; stage 3 onwards does.

**Stage 1: the generator and primitives in the repository, with no family switched.**
- `tools/generate_severity_families.py` holds the declarations and SymPy. It writes
  checked-in numba modules under `src/superglm/distributional/kernels/`, with a
  `--check` mode and a test that the output is current, exactly as the
  fourth-derivative generator works.
- The primitives become one kernel module. Each has an mpmath test over its range.
- **Gate:**
  - every primitive is within 32 u relative of mpmath (the prototype measured at most
    25 u, at the `a = 10` series switch);
  - the 8-line gamma declaration regenerates `GammaLS`'s orders 0-2;
  - `tests/test_family_derivatives_symbolic.py`'s existing tolerances hold on the
    regenerated kernel;
  - no production family changes.

**Stage 2: the generic adapter and the first new families.** These are Weibull (scale
and mean forms), log-logistic, Lomax and Burr XII.
- **Tests:**
  - one parametrised oracle test over every declaration, as in `oracle_check.py`;
  - one public-fit test per family: intercept-only against scipy's maximum likelihood,
    plus one smooth;
  - pickle round trips, and protocol presence and absence.
- **Orders 3 and 4** come from the generated natural derivatives, through
  `natural_curvature_direction` and `natural_curvature_second_direction`.
- **Evidence checklist items the generator covers for every family:** the independent
  row and derivative oracle, derivative-order laziness, the packed channel order and
  width, and frequency-weight replication. Configuration admission, initialisation
  refusals, the public fit and pickling stay per family, but are one line each in a
  parametrised test.
- **Gate:**
  - the oracle grid is within 16 u everywhere outside the documented left-tail log-CDF
    entries;
  - intercept-only fits match scipy's maximum likelihood to 1e-6 in the parameters;
  - `warmup()` covers the new kernels, and first-fit compile time is reported;
  - held-out log score and CRPS through `predict()` on the freMTPL2 severity rerun,
    against `GammaLS`, `LogNormalLS` and `GeneralizedGammaLSS`. This is evidence for
    the families' value, not a gate on the method.
- **Max decides:** the public names (the LS/LSS convention gives `WeibullLS`,
  `LogLogisticLS`, `LomaxLS`, `BurrLSS`), and whether the scale or the mean form is
  each family's default.

**Stage 3: bounds for declared families,** after PR 545 merges.
- Declared `log_survival` and `log_cdf` give `ResponseWindowFamily` through
  `log_window_channels` and its narrow-window quadrature.
- Bounded plans refuse analytic orders 3 and 4 and expected information, as PR 545's
  contract requires.
- **Gate:**
  - the window oracle (`window_check.py`) holds on the PR 545 code rather than on the
    prototype's copy of its formula;
  - an issue-544-style recovery test passes: a truncated body and claims censored at a
    limit recover the truth within sampling error;
  - a regression test fails on the complete-observation path;
  - the PIT of bounded rows is uniform.

**Stage 4: issue 544 stage 2, the incomplete-gamma jet engine.**
- **4a. Series and continued fraction on order-4 jets** (sections 3.4-3.5), in numba,
  as the `logq_jet`/`logp_jet` primitive.
  - **Gate:** an mpmath grid over `a` from 1e-2 to 1e4, with `x` across both tails and
    the diagonal. Orders 0-2 within 1e-12 relative for `a <= 1e3`, and within 5e-10
    up to 1e4. Orders 3 and 4 within 1e-7.
  - **Cost** is per row at orders 0 and 2 on 10^6 rows, against PR 545's estimate of
    1-3 µs per row.
- **4b. `GammaLS` bounds.** Its log tails are declared through the jet primitive.
  - **Acceptance:** issue 544's own criteria, a fitted mean of 5,000 and CV of 1.2
    within sampling error. Also the issue's 200,000-row truncated fit time against the
    complete fit.
- **4c. Temme's uniform expansion on jets,** for `a >= 12` near the diagonal.
  - **Gate:** the regions agree to 1e-12 across their boundaries, and the
    generalized-gamma `q -> 0` window matches `LogNormalLS`'s to 1e-10.
- **4d. `GeneralizedGammaLSS` bounds:** the `q` sign switch, and `q = 0` as a
  removable point. The gate is 4b's plus the `q`-limit tests.

**Stage 5: optimisation, after the methods are in.**
- Value-only evaluation when only order 0 is asked for.
- One digamma and trigamma per distinct shape when the scale predictor takes few
  values.
- Skip rows whose window probability rounds to one.
- numba's on-disk cache for the generated modules.

**Stage 6, a separate decision: replace a hand-written kernel with its generated
equivalent** where that is faster and gives the same outputs. `GammaLS` is the measured
candidate (section 2.4 and the speed results above). A declaration can carry a
reproductive prior-weight law as a substitution, `cv -> cv/sqrt(w)` for the gamma, so
`GammaLS`'s prior-weight law need not block this.
- **Gate:** equal or faster wall time at default threads on complete fits, on every
  route `GammaLS` takes: dense, discrete, and prior and frequency weights.
  Predictions within 1e-10. Every hand-written refusal either kept or shown
  unreachable.

## 5. Discrepancies and follow-ups

- **The task brief says censoring is "an explicit `censored=` flag".** PR 545 at
  `90b0eb46` takes `censoring=(lower, upper)` pairs. A row equal to its bound is
  censored on that side (`observation.resolve_response_bounds`). If a flag is planned,
  it is not on that head.
- **"The Research section's note on the gamma families"** exists only in PR 545's edit
  of `notes/ROADMAP.md`. Master's roadmap has no such note.
- **scipy's log tails fail in places** (section 2.4). Any code that censors through
  scipy's `logsf` or `logcdf` would have to refuse those rows or silently mis-weight
  them.
- **`GammaLS`'s hand-written row kernel is the slowest part of its fit.** Its
  likelihood evaluation took 2.0 s of a 4.6 s fit above. The generated equivalent
  takes 0.87 s with the same iterations and the same predictions. That is stage 6. It
  is a separate change, and it needs the every-route gate there before it could land.
- **Both gamma kernels recompute digamma and trigamma on every row,** even when the
  shape is constant. PR 545's note found this for the hand-written kernel. It is
  stage 5's distinct-shape item, and it applies to either kernel.
- **Process.** The oracle checks for section 2.4 ran single-threaded outside
  benchboard. They were light mpmath work, but the machine rule covers them. Every
  timing and fit ran through benchboard.

## Runs behind this note

Everything runs from `notes/research/severity-family-prototype/` with the dev
environment (`uv sync --python 3.13 --extra dev`). `OUT` is any scratch directory.

| Script | What it measures | How it ran |
|---|---|---|
| `oracle_check.py OUT` | section 2.4 tables: every channel against scipy, 60-digit (400 for textbook forms) mpmath and Richardson differences | one thread, outside benchboard (light mpmath work; disclosed here because the machine rule covers it) |
| `bench_kernel.py OUT` | ns per row on 10^6 rows, best of 7 | `benchboard run-timed`, all seven thread pools pinned to 1 |
| `fit_compare.py OUT 3` | 200,000-row `GammaLS` `fit_reml`, hand-written against generated kernel, three alternating pairs | `benchboard run-timed`, default threads |
| `fit_declared.py OUT` | declared Weibull and Burr XII through `SuperLSS`, against scipy maximum likelihood | `benchboard heavy`, 4 cores |
| `window_check.py OUT` | generated log tails through PR 545's documented combiner, against 100-digit mpmath | `benchboard heavy`, one core |
| `compile_time.py OUT` | SymPy and numba compile seconds per family | `benchboard heavy`, one core |
| `alternatives/` (`speed.py`, `accuracy.py`, `complex_step.py`, `jax_facts.py`, `pypi_wheels.py`, with their JSON and logs) | section 2.6: JAX, differences, complex step and hyper-dual against a numba closed form | timing through `benchboard run-timed` on one pinned core; JAX 0.12.0.1 in a separate venv, never in the project lock |

The incomplete-gamma measurements in section 3.4 come from the literature sweep's own
scripts, committed with their outputs in
[`incomplete_gamma/`](severity-family-prototype/incomplete_gamma/):
- `jet_check.py` is the first version, with the switch at `x = a`;
- `v2_check.py` is the switch line and stopping rule of section 3.5 (the table);
- `cf_diag.py` diagnoses the continued fraction near the diagonal.

Run `python incomplete_gamma/jet_check.py` directly. `v2_check.py` and `cf_diag.py`
take the folder that holds `jet_check.py` as their argument. They are light mpmath work
and ran outside benchboard.

## Sources

Read in full unless marked.

- Gautschi, W. (1979). A computational procedure for incomplete gamma functions. ACM
  TOMS 5(4).
- Gil, A., Segura, J. and Temme, N. M. (2012). Efficient and accurate algorithms for the
  computation and inversion of the incomplete gamma function ratios. SIAM J. Sci.
  Comput. 34(6), doi:10.1137/120872553. arXiv:1306.1754.
- DiDonato, A. R. and Morris, A. H. (1986). Computation of the incomplete gamma function
  ratios and their inverse. ACM TOMS 12(4) (abstract only). Algorithm 654, ACM TOMS
  13(3), 1987.
- Moore, R. J. (1982). Algorithm AS 187: derivatives of the incomplete gamma integral.
  Applied Statistics 31(3) (abstract and R wrapper documentation only).
- Figurnov, M., Mohamed, S. and Mnih, A. (2018). Implicit reparameterization gradients.
  arXiv:1805.08498.
- Johansson, F. (2021). arXiv:2110.10583 (sections 1, 3.1-3.2 and 5).
- Abergel, R. and Moisan, L. (2020). Algorithm 1006. ACM TOMS 46(1) (abstract only).
- Mächler, M. (2012). Accurately computing log(1 - exp(-|a|)). Rmpfr package vignette.
- DLMF sections 8.12 and 8.25.
- Wood, S. N., Pya, N. and Säfken, B. (2016). Smoothing parameter and model selection
  for general smooth models. JASA 111.
- Documentation, read in part: SciPy `special`; Boost.Math incomplete gamma and its
  derivatives; Stan Math `grad_reg_inc_gamma` and `grad_reg_lower_inc_gamma`; JAX
  `lax.igamma_grad_a`; SymPy `codegen.rewriting`; VGAM and heavy `pgamma.deriv`
  (documentation only; GPL); R `GammaDist`.
- Automatic differentiation and finite differences:
  - Carpenter et al. (2015), the Stan Math library, arXiv:1509.07164, read in part;
  - Kristensen et al. (2016), TMB, arXiv:1509.00660, read in part;
  - Margossian (2019), arXiv:1811.05031, read in part;
  - Hückelheim et al. (2023), arXiv:2305.07546;
  - Baydin et al. (2018), arXiv:1502.05767, read in part.
- Squire and Trapp (1998), SIAM Review 40(1): abstract, and the formula in a secondary
  source.
- Martins, Sturdza and Alonso (2003), ACM TOMS 29(3): abstract only.
- Fike and Alonso (2011), AIAA 2011-886: abstract and a secondary source.
- Lantoine, Russell and Dargent (2012), ACM TOMS 38(3): bibliographic record only.
- Gill, Murray, Saunders and Wright (1983): through NAG's e04xaf documentation only.
- JAX 0.12 package metadata and PyPI wheel listings.
