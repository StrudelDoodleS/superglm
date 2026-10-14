# Migration: exact REML fits in 0.40

*Ships in 0.40.0.*

Three changes to how an exact smoothing search (`fit_reml` with
`discrete=False`) runs and stops. Binned fits (`discrete=True`) take only the
third.

## 1. Weight derivatives in one pass

### What changed

Each iteration needs how the fit's working weights move with every smoothing
parameter. It used to form one weighted cross-product over all rows per
smoothing parameter. It now takes one pass over the rows for all of them
(Wood 2011, section 3.5.1). The gradient is the same, so the search stops at
the same optimum.

### Who is affected

- **Large exact fits**: those with at least four smoothing parameters where
  smoothing parameters x rows x coefficients squared reaches 1e11.
- **Not affected**: binned fits, Gaussian with the identity link, binomial
  with the logit link (its own one-pass gradient, as before), fits with a
  `RandomEffect` or `FactorSmooth` term, and `w_correction_order=2`.

### What moves

- **Iteration counts**: the outer Hessian leaves out one second-order term, so
  a fit can take a different number of iterations (3 pairs on the freMTPL2
  book: 12 before, 14 after).
- **The last digits of the smoothing parameters**: within the stopping
  tolerance (1 pair: REML objective 0.0059 lower, EDF 77.36 to 77.27, test
  deviance +0.04).
- **Time**: 1 pair 141 s to 25 s, 3 pairs 913 s to 106 s, one thread.

### Getting the old behaviour back

There is no switch for the old route. `w_correction_order=2` still forms the
per-parameter products, with second-order terms added, and is slower.

## 2. A cold exact fit starts from the cached-weights solution

### What changed

Before its Newton search, a cold exact fit first runs the binned path's
cheaper cached-weights search on the same exact design, and starts the
Newton search from where that stops (mgcv's `outerPIsteps`, run to
convergence). The Newton search's gradient and stopping test are unchanged.

### Who is affected

The fit must have all of these:

- no warm start (`lambda2_init` is not a mapping);
- at least eight smoothing parameters to estimate;
- Poisson with the log link, the only family measured so far (others keep
  the plain start until they are measured);
- no `RandomEffect` or `FactorSmooth` term;
- `max_reml_iter` of at least 10 (so not `interaction_mode="fast_candidate"`).

It is also off inside `estimate_p`'s power search.

### What moves

On the freMTPL2 frequency book (corrected claim counts, cold fits, one
thread):

| | Before | 0.40 |
|---|---|---|
| 4 gated pairs: time | 115 s | 61 s |
| 4 gated pairs: REML objective | 64,773.720 | 64,773.704 |
| 4 gated pairs: test deviance | 32,538.24 | 32,538.30 |
| 10 pairs: time | 365 s | 189 s |
| 10 pairs: REML objective | 64,919.021 | 64,919.030 |
| 10 pairs: test deviance | 32,546.39 | 32,546.44 |

- **Smoothing parameters**: those of nearly flat terms stop elsewhere along
  the flat region, which depends on the start. The objective moves by about
  0.01.
- **Iterations**: `reml_diagnostics()["n_reml_iter"]` counts the Newton
  search only: 6 on both models above, against 12 (4 gated pairs) and 14
  (10 pairs) before. `["n_presolve_iter"]` counts the cached-weights updates
  before it (9 and 12).

### Getting the old behaviour back

Pass `lambda2_init` as a mapping (for example a previous fit's
`reml_diagnostics()["lambdas"]`). A warm start skips the cached-weights search.

## 3. A search no longer stops on a concave shoulder

### What changed

Far up the flat end of a smooth that the data do need, the criterion still
falls toward the smooth's optimum, with negative curvature. The search used
to hold such a direction as flat and report convergence there. It now steps
off it, and the tolerance stop is refused while one is left. Two stops can
still happen beside one: a line search that rejects every trial certifies on
its first-order precision floor, as before, and a direction whose negative
curvature is under half or over twice its gradient keeps the old hold.

### Who is affected

- **Warm starts and `lambda2_init` values far above the optimum**, on both
  exact and binned fits.
- **Cold fits**: unchanged, bit for bit, on the cases measured.

### What moves

- **A warm start that used to report convergence** now iterates further. It
  may reach the cold fit's answer, a different stationary point, or the
  iteration cap. At the cap it says so with a `ConvergenceWarning`.
- **Measured**: 4 gated pairs started with every smoothing parameter at 1e4
  used to report convergence 1.89 worse than the cold fit. It now ends 1.05
  worse, one iteration short of certifying at the default cap of 20; it
  certifies at iteration 21.

### Getting the old behaviour back

There is none: the old stop was not a minimum. To avoid the longer search,
start nearer the optimum, for example from a cold fit's smoothing parameters.

## Verification trail

- `tests/test_reml_leverage_hessian.py`: the one-pass weight derivatives
  against the per-parameter route.
- `tests/test_reml.py` (`TestExactPresolve`, `TestExactPresolveScope`): where
  the cached-weights start runs, and that every excluded case is bitwise the
  fit without it.
- `tests/test_reml.py` (`TestREMLWarmStart`) and
  `tests/test_reml_convergence.py`: a fit started on a concave shoulder
  returns to the cold optimum on both engines.
