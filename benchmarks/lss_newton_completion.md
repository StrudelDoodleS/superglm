# SuperLSS curvature refresh and smoothing completion

SuperLSS could exhaust its Newton budget because a negative-curvature verdict
from an earlier fit kept blocking completion after accepted BFGS steps. The
solver now refreshes that verdict when its other stop checks pass. The mixed
Gaussian and GPD reproductions finish within their existing budgets. The
default outer method and practical stopping policy stay unchanged.

This follows the thread-sensitive parity assertion found during the predictor
API work. That assertion compared independent practical REML endpoints too
tightly. Fixing the assertion did not establish that the two endpoints had
identical uncertainty estimates. This comparison measures that difference
directly and checks the existing stricter search.

## The stale curvature verdict

The initial full Hessian can have positive diagonal entries and an indefinite
active block. The step selector can use that matrix after regularization,
while retaining its negative-curvature verdict. Subsequent BFGS steps update
the inverse memory and evaluate new gradients. Previously, they could keep
that original verdict indefinitely and never request another full Hessian.

Both reproductions reached points where the gradient, objective-change and
remaining-gain checks passed. The old curvature flag was their only failed
stop condition. A separate Hessian evaluation at the final baseline fit gave
a positive active block in both cases:

| Case | Smallest active eigenvalue | Norm of reported Hessian certificate |
| --- | ---: | ---: |
| Mixed Gaussian | 0.013538 | `3.28e-9` |
| GPD | 0.088976 | `5.22e-5` |

The repair disables reuse of the BFGS approximation for the next step when
those other checks pass and the retained verdict is negative. It preserves
the inverse memory for fallback. The existing step path evaluates the current
Hessian using the current gradient workspace, applies its usual safeguards,
and accepts or rejects the trial under the existing objective rule. A new
gradient evaluation then judges the accepted point. Exhausted budgets and
unresolved gradient certificates keep their existing precedence.

This is a scheduling repair. It does not accept a modified BFGS matrix as
evidence that the actual Hessian became positive, add a new stopping formula,
or claim a certified terminal Hessian. The first-order fallback contract still
allows stationary completion when a Hessian is unavailable or untrusted.

## Scope and reproduction

The production baseline is release 0.32.0 at
`087d5983d1cba9823014c73eb8400b593a8c9217`. The
[driver](lss_newton_completion.py) compares these public options:

```python
# Default practical search.
model.fit_reml(X, y, outer="efs", practical_reml=True)

# Stricter smoothing completion using the existing Newton option.
model.fit_reml(X, y, outer="efs+newton", practical_reml=False)
```

The recorded measurements used the
[driver at commit `681939e6`](https://github.com/StrudelDoodleS/superglm/blob/681939e63084553ad82aa7e49c09642138245d34/benchmarks/lss_newton_completion.py),
whose hash is retained in the receipt. Use that revision of the driver when
replaying the comparison against 0.32.0. The current driver uses the family-bound
predictor API introduced after these measurements; the fixtures and fitting
options remain the same. The recorded timings describe the measured revisions.

Each case uses identical data, terms, weights, offsets and starting-policy
settings across the two routes. The outward-boundary control supplies an
explicit start, cap, step size and tolerances. The other cases use the public
defaults. The Newton phase can use BFGS updates after its initial Hessian; the
name does not mean it recomputes an exact Hessian for every step.

Run a worker in a fresh process:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run --no-sync python benchmarks/lss_newton_completion.py \
  --case gaussian-correlated --route newton-strict \
  --data data/ --out /tmp/lss-completion/gaussian-newton.json
```

Repeat each case and route separately. Exclude the first warm-up cycle, then
alternate which route runs first. Add `--profile` for a separate cProfile and
phase-timing run. The driver records source and input hashes, package versions,
native thread pools, process activity, complete-fit wall and CPU time, process
peak RSS through fitting, diagnostics, and the actual terminal coefficient
execution backend. RSS includes imports and fixture construction and excludes
subsequent prediction/export work. The NPZ
beside each receipt stores natural parameters, linear predictors, conditional
coefficient covariance, and conditional standard errors on the link scale.

The correlated Gaussian case reproduces the original 160-row API fixture.
The crossed control separates the two smooth covariates on a 20 by 20 grid.
The mixed Gaussian case has 4,096 rows and many small terms. The Gamma case
uses 22,450 training policies with positive aggregate claims from freMTPL2.
The NB2 book case uses the first 20,000 training policies in sorted ID order
with log exposure as an offset. Both book cases exclude every tenth policy
for evaluation and reuse the existing benchmark's covariate clipping rules.
The separate NB2 simulation has a finite size parameter and 3,000 observations.
The existing Tweedie stress case has 10,000 observations and 192 coefficients;
the GPD case has 1,401 excesses and 98 coefficients.

## Complete-fit measurements

The [receipt](lss_newton_completion_receipt.json) records input/source checks,
individual timings, numerical differences and profile call counts. Runs used
Python 3.13, NumPy 2.5.2, SciPy 1.18.0 and one thread in the configured native
pools. Fresh workers selected the recorded checkout through `PYTHONPATH` and
verified the imported solver paths and file hashes for the repair comparison.
Profiles ran separately from the uninstrumented timing comparisons.

After one excluded warm-up cycle, the affected cases had two paired repeats,
alternating which source ran first. Times and RSS below are medians. These are
descriptive measurements of these configurations, with limited repetition.

| Case, `efs+newton` | Baseline stop | Repaired stop | Newton steps, before / after | Wall seconds, before / after | Peak process MiB, before / after |
| --- | --- | --- | ---: | ---: | ---: |
| Mixed Gaussian | Budget exhausted | Stationary | 20 / 13 | 1.453 / 1.219 | 433.4 / 438.8 |
| GPD | Budget exhausted | Stationary | 20 / 6 | 2.335 / 1.398 | 402.6 / 405.5 |

The measured time ranges were 1.407-1.499 versus 1.136-1.301 seconds for the
Gaussian case, and 2.285-2.384 versus 1.370-1.427 seconds for GPD. Peak process
RSS increased by 5.5 MiB and 2.9 MiB respectively. One additional Hessian pass
reduced subsequent fits and derivative passes. Relative to the longer baseline Newton trajectories,
the largest conditional standard-error change was 0.079% for Gaussian and
0.077% for GPD. The largest link-prediction shifts were 0.0037 and 0.0010 of
their respective conditional standard errors. These comparisons describe the
returned fits; the unconverged baseline is not an accuracy oracle.
Here the SE percentage is `100 * abs(SE_before / SE_repaired - 1)`, maximized
over the common evaluation rows for each parameter.

The correlated Gaussian and Tweedie controls returned identical numerical
arrays and retained the same iteration and derivative-pass counts. Five
paired timing repeats gave Gaussian medians of 1.005 versus 1.111 seconds,
with ranges 0.942-1.125 versus 0.986-1.276. Tweedie medians were 7.076 versus
7.316 seconds, with ranges 6.931-7.311 versus 6.863-7.534. Those control
medians increased, with overlapping ranges and no added numerical work.
The measurements do not establish a general speedup or a stable control-path
regression. Peak process RSS stayed within 0.3 MiB for these controls.
Separate plain-EFS controls for the two affected cases returned identical
saved arrays before and after the patch.

## Policy comparison

The initial comparison used the unmodified baseline and two measured repeats
after a warm-up cycle. This is separate from the repair comparison above.

| Case | Default EFS outcome | Baseline strict Newton outcome | Median wall seconds, EFS / Newton |
| --- | --- | --- | ---: |
| Correlated Gaussian | Practical plateau | Stationary | 1.528 / 1.096 |
| Crossed Gaussian | Objective rejected | Stationary | 0.728 / 0.497 |
| Mixed Gaussian | Practical plateau | Budget exhausted | 0.906 / 1.468 |
| Outward Gaussian | Practical plateau | Exact face, `lambda_change` | 0.022 / 0.043 |
| Gamma book | Practical plateau | Stationary | 1.600 / 1.719 |
| NB2 simulation | Practical plateau | Stationary | 0.294 / 0.366 |
| NB2 book | Initial coefficient fit refused | Same refusal | 11.323 / 11.470 |
| Tweedie stress | Later coefficient refit refused | Stationary | 89.242 / 7.780 |
| GPD tail | Unresolved cap | Budget exhausted | 1.290 / 2.551 |

The NB2 book raises `NegativeBinomialPoissonBoundaryError` before smoothing
starts. This records the solver's refusal; it does not independently prove
that the generating process is Poisson. The retained Tweedie coefficient fit
is converged; the outer search refuses a later trial coefficient fit.

Comparing practical-EFS predictions with strict-Newton predictions at the same
evaluation rows gives these largest conditional standard-error differences:

| Case and parameter | Maximum change in conditional link SE |
| --- | ---: |
| Correlated Gaussian location | 3.24% |
| Crossed Gaussian location | 0.41% |
| Gamma mean / scale | 0.082% / 0.084% |
| NB2 simulation mean | 0.0045% |
| Tweedie power | 1.26% |
| Mixed Gaussian scale, repaired Newton | 11.69% |
| GPD shape, repaired Newton | 18.95% |

The mixed Gaussian and GPD comparisons use the repaired stationary runs.
These are sensitivity measurements of conditional link standard errors, not
interval coverage tests. A covariance-matrix norm alone would obscure this
parameter- and row-specific behavior.
The percentage is `100 * abs(SE_EFS / SE_strict - 1)`, maximized over the common
evaluation rows. The stricter fit supplies the denominator, without being
assumed to have correct interval coverage.

## Profiling and reuse

The additional Hessian uses the gradient workspace already prepared at that
accepted fit. `LamlDerivativeWorkspace.matches()` checks the fit, family,
likelihood plan, lambdas, derivative step and predictor-matrix identities.
The next accepted fit creates a new workspace. No new cache or retained array
was added.

| Profile operation | Gaussian before / after | GPD before / after |
| --- | ---: | ---: |
| Gradient passes | 21 / 14 | 21 / 7 |
| Hessian passes | 1 / 2 | 1 / 2 |
| Coefficient-solver calls | 30 / 23 | 42 / 21 |
| Accepted-endpoint reuse attempts | 28 / 21 | 40 / 19 |

Hessian evaluation reuses the gradient's stencils and leverage blocks. Its
additional pair contractions consume those results without repeating the
gradient pass. Coefficient trials continue to use `_DenseObservedReuseSession`
and warm starts from the accepted fit. That session can reuse likelihood and
data-curvature work while changing the penalty, subject to its existing
identity, retained-state and prediction checks. The ordinary controls retain
one Hessian pass each. Their call counts do not indicate discarded work added
by the repair.

The four profiled comparison cases recorded `distributional-chunked-v1` as the
terminal coefficient backend. The outward Gaussian policy control used
`distributional-dense-v1`; the NB2 book refusal produced no terminal backend.
These labels identify the terminal fit, not every internal operation. The
separate profiles confirm the bounded predictor and
existing coefficient-reuse paths. These fixtures do not measure ten-million-row
memory behavior.

## What the stopping rules mean

Practical stopping checks sustained small changes in the objective and fitted
natural parameters. It retains its pressure evidence and reports
`smoothing_certified_=False`. The Newton route checks the existing numerical
stationarity contract for the profiled objective in log smoothing parameters.
Passing that contract does not prove a global minimum, unique solution or
correct interval coverage. The public guide describes the limitations of its
[derivative certificates](../docs/models/distributional.md#how-smoothing-parameters-are-chosen).

The mathematical reason that an objective plateau is insufficient follows from
Taylor's theorem. Near a stationary point,

\[
L(\theta + d)-L(\theta) \approx \tfrac12 d^\top H d.
\]

A small eigenvalue of the Hessian permits substantial movement with a small
objective change. For example, `L(b) = 1e-8 * b**2` changes by only `1e-8`
between `b=0` and `b=1`, even in exact arithmetic. If an objective is strongly
convex with a known curvature lower bound `m > 0`, its gradient gives a bound
`L(theta) - L* <= ||gradient||**2 / (2*m)`. The bound becomes weak when `m` is
small. We have not established that global assumption for these profiled
SuperLSS objectives. See [Boyd and Vandenberghe, chapter 9](https://web.stanford.edu/~boyd/cvxbook/bv_cvxslides.pdf).

The existing Hessian trust test compares diagonal entries with derivative
certificates for step selection. It does not prove spectral positivity. Such
a claim would need the smallest active eigenvalue to exceed the certificate's
matrix norm and eigensolver error, and would still depend on that certificate
enclosing the relevant error. The reported refinement indicators are not
complete error enclosures. Likewise, the capped or ridged `remaining_gain`
probe is a local-model stopping heuristic, not a bound on further objective
improvement. The refresh repair preserves these documented limits. The
underlying outer algorithm distinguishes curvature assessment from step
regularization in [Wood, Pya and Säfken, algorithm 4(c-d)](https://arxiv.org/pdf/1511.03864).

Roundoff sensitivity is a separate question. A condition number measures how
input perturbations can affect a result; a stable linear solve can still have
a sensitive solution. See [Higham's condition-number explanation](https://nhigham.com/2020/03/19/what-is-a-condition-number/).
Neither statement establishes that Newton will always improve a fit.

## The outward practical exception

Commit `7d054022853a1315801ffb77ccb828551c0cc2cd` deliberately allowed a
sustained outward practical plateau to finish before Newton. The public guide
documented this exception, although `should_hand_off()` described its own
decisions too broadly. Its docstring now distinguishes the caller's practical
stop from stops submitted to the handoff helper.

The 40-row random-effect control demonstrates why the exception matters.
After three accepted outward steps, the finite model meets the practical
criteria. Forcing a handoff gives these outcomes:

| Outer budget | Existing practical policy | Forced Newton handoff |
| --- | --- | --- |
| 3 | Practical convergence | Unresolved cap, not converged |
| 4 | Practical convergence | Face-validation budget exhausted |
| 20 | Practical convergence | Exact face, `lambda_change` |

At budget three, the returned objective is identical. The forced route needs
more budget to decide and validate the boundary; it has not disproved the
practical evidence. At the larger budget, the Newton pass reports cap pressure
and takes zero Newton steps before the existing boundary machinery selects an
exact face. This also shows why zero Newton iterations alone cannot establish
that Newton was never called.

The practical-policy regression checks this policy under both `outer` options and all three
budgets. A forced-handoff mutation must fail it. Existing practical-stop tests
compare natural parameters and covariance with the strict face solution.
Existing Newton tests cover unavailable gradients, retained accepted fits,
BFGS fallback, and recovery after a released cap. These behaviors remain intact.

## Regression coverage

Both real curvature-refresh regressions failed against the unfixed solver
because it evaluated only one Hessian and exhausted its budget. They now check
a Hessian at a later accepted state, reuse of that state's gradient workspace,
stationary completion within the existing budget, and terminal gradient
authority. The initial negative eigenvalue must be separated from the
derivative certificate and a conservative relative matrix perturbation.

Fault injection after the actual initial indefinite Hessian covers unavailable,
untrusted, and persistently indefinite refreshes. The tests check continued
BFGS fallback, bounded work, and terminal derivative provenance. An unavailable
or persistently indefinite refresh in this fixture exhausts the existing
budget without claiming convergence. An untrusted Hessian can still end under
the existing first-order stationarity contract.

## Decision

Keep the default and the documented outward practical exception. To investigate
sensitivity, fit the same model with `outer="efs+newton"` and
`practical_reml=False`, inspect its completion reason and diagnostics, and
compare predictions and conditional standard errors at common evaluation rows.
An unconverged stricter run is another candidate, not a reference answer.

The mixed Gaussian and GPD failures needed fresh curvature, rather than a
larger iteration budget. The NB2 book refusal happens before smoothing starts,
so this repair cannot resolve that reproduction. The broader policy comparison
still includes costs and model-dependent changes in uncertainty. It supports
the focused repair and explicit stricter comparisons, rather than a universal
default change.
