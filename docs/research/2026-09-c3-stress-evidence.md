# C3 stress convergence evidence, 2026-09-08

Both original stress fixtures reach the API's certified first-order stationary fits using the existing
`outer="efs+newton"` option with practical stopping disabled. No new optimizer,
shape penalty, relaxed coefficient tolerance, or endpoint-authority change was
needed. Default EFS still fails on these fixtures; this is an explicit limitation
of that route, not a claim that every starting point or optimizer converges.

## Reproduction and controls

The frozen source is v0.31.0,
`8962c4520cad948aa20c480a238b7bb1e276e9cd`, production-tree hash
`b5dabb2445f0e25e927250f7d07f079c4364578ff077a5bca19bbd1e95e9e5c2`.
The implementation refresh records HEAD
`a433ab88a1b7fb26c88af2b69fccc0d24b051862` and its complete production tree,
including uncommitted files, as
`ee937668f83c46d8e6d0990f4d0781186a2d98531000afbb564eecd687b377af`.
All 17 refreshed receipts pass source/helper stability and one-thread runtime
checks. The recovered pure generators in
[`benchmarks/_c3_c1_fixtures.py`](../../benchmarks/_c3_c1_fixtures.py) preserve the
original random draw order and were checked for exact array equality against
AST-extracted original definitions. No old assembler or solver monkeypatch is
used. The cases are:

- Correlated Tweedie: `fixture(10000, mix=.75, family="tweedie")`, seed 5915,
  three predictors, 192 coefficients.
- GPD threshold excesses: `marked_book(10000, tail=True)`, seed 9506, training
  claims exceeding 1000, excess response, 1,401 rows and 98 coefficients.

The historical failure-probe controls are `max_reml_iter=60` and
`acceleration="multisecant"`, with default EFS and practical stopping. The
successful replay uses:

```python
model.fit_reml(
    frame, y, sample_weight=weights,
    outer="efs+newton", practical_reml=False,
    initial_lambda=0.01, max_reml_iter=100,
    acceleration="multisecant",
)
```

Starts 0.01, 0.1, and 1 were tested without changing the default `reml_tol=1e-6`
or `inner_tol=1e-7`. The final source-bound refresh uses the same Python
environment and one BLAS, OpenMP, and Numba thread, with source selection through
`PYTHONPATH` and an explicit `--source` assertion. Thread settings and runtime
counts are recorded. The optional mpmath reference uses an ephemeral environment
with the same numerical-library versions. These were potentially concurrent
numerical probes: **time and memory benefit are unmeasured here**. The separate
complete-fit harness owns performance evidence.

| Source/case | Outer route | Start | Outcome | Negative LAML | EDF |
|---|---|---:|---|---:|---:|
| Baseline/current GPD | EFS | 0.1 | `lambda_cap_unresolved` | 10804.724537124 | 24.178870 |
| Baseline/current GPD | strict EFS + Newton | 0.01 | certified `stationary` | 10804.676915949 | 24.608578 |
| Baseline/current GPD | strict EFS + Newton | 0.1 | certified `stationary` | 10804.676992369 | 24.683169 |
| Baseline/current GPD | strict EFS + Newton | 1 | certified `stationary` | 10804.676237803 | 24.637047 |
| Baseline/current Tweedie | EFS | 0.1 | `coefficient_not_converged` | 9663.177673923 | 44.143246 |
| Baseline/current Tweedie | strict EFS + Newton | 0.01 | certified `stationary` | 9663.174369049 | 44.176908 |
| Baseline/current Tweedie | strict EFS + Newton | 0.1 | certified `stationary` | 9663.178098707 | 44.184318 |
| Baseline/current Tweedie | strict EFS + Newton | 1 | certified `stationary` | 9663.177084444 | 44.194233 |

The observed negative-LAML differences between starts are below the configured
stationarity bar and consistent with stopping sensitivity. Their cause is not
established by that comparison, and they do not establish an identical optimum.
Here "certified" names the existing API's first-order log-lambda stopping and
authority contract: the computed projected LAML score and the propagated
Richardson-refinement indicators separately meet the objective-scaled bar,
`reml_tol * (1 + abs(objective))`. The indicators do not enclose all derivative
error, including coefficient-mode and linear-solve errors. These two checks do
not prove a bound on the exact gradient, interval-certified convergence, a local
minimum, a global optimum, or forward error. They also do not mean a zero gradient.

The terminal projected norms at starts 0.01, 0.1, and 1 are approximately
0.006813, 0.009473, and 0.002426 for GPD, against bars approximately 0.010806;
for Tweedie they are 0.0001129, 0.003244, and 0.007608, against bars approximately
0.009664. These are the stored terminal values, not the last pre-step history
gradient. The compact receipts retain every terminal component and refinement
indicator needed to repeat the reported comparisons.

## Failure classification

The EFS GPD replay retains 21 coefficient fits and refuses the
`shape:z#wiggle` cap. Its endpoint assessment explicitly requests coefficient
tolerance `1e-12`; the last assessment fit reports `line_search_failed` and
relative score approximately `2.56e-9`. That failed authority is preserved. The
successful Newton trajectories stop at large finite penalties and do not need
to authorize an infinity face. GPD's existing shape link restricts shape to
`(0, 1)`: reaching a smoothing cap does not diagnose unbounded likelihood, and
adding a shape penalty would change the statistical model.

The EFS Tweedie replay retains 80 coefficient fits and reports
`coefficient_not_converged`. Strict Newton follows a different smoothing
trajectory and reaches stationary results. This does not justify accepting an
uncertified sub-ULP coefficient move, using an unbounded paired likelihood
difference as authority, or relaxing endpoint KKT checks. Those safeguards are
unchanged.

## Across-start predictions and conditional uncertainty

For each pair of starts, the comparison computes per-parameter row RMS of
`(prediction_a - prediction_b) / prediction_a`. Conditional linear-predictor
standard errors are `sqrt(diag(X_k Cov(beta_k) X_k.T))`, compared by row RMS of
`(SE_a - SE_b) / SE_a`. The table gives the largest pairwise RMS in each column;
these are measured sensitivities, not numerical acceptance tolerances.

| Case/parameter | Prediction relative RMS | Conditional link SE relative RMS |
|---|---:|---:|
| GPD scale | 0.1893% | 0.1531% |
| GPD shape | 1.2655% | 1.3184% |
| Tweedie mean | 0.0001673% | 0.0024446% |
| Tweedie dispersion | 0.0070393% | 0.0091934% |
| Tweedie power | 0.0104289% | 0.1098023% |

The GPD shape uncertainty sensitivity is material enough to report explicitly.
These comparisons concern conditional covariance; they do not establish
across-start equality of a smoothing-uncertainty correction. Individual
smoothing parameters, especially near-flat directions, need not agree.

## Independent references and their limits

At the three GPD stationary fits, the whole-fit likelihood agrees with
`scipy.stats.genpareto.logpdf` to at most `1.82e-12`. At sampled fitted rows,
60-decimal-digit differentiation of the literal GPD log density with mpmath
gives maximum `abs(error)/(1+abs(reference))` of `6.49e-16` for the natural score
and `3.42e-15` for the natural Hessian.

At sampled Tweedie fitted rows, the installed R mgcv 1.9.3 public `ldTweedie`
callable, with prior weights represented by `phi/weight`, agrees with normalized
production log densities to at most `2.40e-14`. This executes the installed
reference implementation without reading or copying its GPL source.

The GPD start-0.01 LAML gradient was also compared to central differences of
independently refitted objectives at log-lambda steps 0.02, 0.01, and 0.005,
followed by Richardson extrapolation. Five ordinary directions agree within
the observed refinement resolution. Two tighter coefficient probes refuse
with `line_search_failed`; they are recorded as refusals. The nearly flat
`shape:z#wiggle` direction has analytic derivative about `-9.18e-9`, while
refitted-objective subtraction fluctuates around `1e-6`. That finite-difference
probe does not resolve the direction and is not an independent certificate for
it. No failed probe was converted into a passing check.

## Durable checks and receipts

[`tests/test_c3_stress_stationarity.py`](../../tests/test_c3_stress_stationarity.py)
replays both exact fixtures under strict Newton at start 0.01. It checks fresh
projected-score and derivative-certificate authority using the configured
objective-scaled bar, and mutates the published score authority to demonstrate
that invalid result replay is rejected. Both tests pass on baseline and current
source. They are marked `slow` and explicitly run with:

```sh
uv run pytest tests/test_c3_stress_stationarity.py -q -m slow
```

[`benchmarks/c3_c1_stress_diagnosis.py`](../../benchmarks/c3_c1_stress_diagnosis.py)
replays either fixture and writes raw JSON, NPZ, and SHA256 receipts. It requires
`--source /absolute/path/to/checkout` and refuses an imported package outside
that source tree. Before importing production code, it records the Git HEAD,
hash of every production Python file (including untracked modules), combined
production-tree hash, and diagnostic, fixture, and provenance-helper hashes.
It checks source and helper stability again after the fit and optional probes.
The receipt exposes the terminal coefficient-fit index, computed projected
gradient, full terminal gradient, refinement indicators, smoothing configuration,
and stationarity bar. These terminal fields supersede the earlier receipts;
a last pre-step history gradient cannot establish the terminal stopping claim.
Optional `--oracle` runs the refitted-LAML probes; `--row-reference` requires
installed R/mgcv for Tweedie or mpmath for GPD.

The source-bound generated evidence is under
`benchmarks/results/c3-c1-stress-v2/`, with full histories, parameter and covariance
arrays, independent reference rows, and `manifest.sha256.json`. The original
`c3-c1-stress/` directory is preserved. The intermediate pass without full thread
metadata is archived as `c3-c1-stress-v2-prethreadmeta/` and is not the final
evidence set. These directories follow the ignored generated-results policy.

[`benchmarks/c3_c1_stress_summary.py`](../../benchmarks/c3_c1_stress_summary.py)
checks artifact hashes, source-tree hashes and stability, imported roots, and
runtime thread counts without importing SuperGLM. For stationary receipts it
recomputes the bar from the recorded configuration and objective, then checks
the published terminal norm and indicators against that bar and verifies the
selected terminal coefficient fit converged. This checks the reported API
contract independently; it does not supply a new derivative-error enclosure.
The compact, source-bound selection is
[`benchmarks/c3_c1_stress_receipt.json`](../../benchmarks/c3_c1_stress_receipt.json).
To reproduce it from the full local receipts:

```sh
uv run python benchmarks/c3_c1_stress_summary.py \
    benchmarks/results/c3-c1-stress-v2 \
    --out benchmarks/c3_c1_stress_receipt.json
```

The evidence closes C3 through existing optimizer controls, while preserving
the explicit EFS and finite-difference limitations above.
