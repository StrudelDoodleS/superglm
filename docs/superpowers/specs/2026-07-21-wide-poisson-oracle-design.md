# Fast Wide-Poisson POI Authority Design

## Purpose

Replace the five-minute `test_wide_poisson_poi_quality` execution with a focused
test of the same production workload and numerical contract. The replacement
must continue to protect the discrete Performance Oriented Iteration (POI) path
on a wide, flat REML problem without recomputing an expensive exact-fit oracle
on every test run.

This is a test-only change. It does not alter production fitting, tolerances,
solver dispatch, or numerical algorithms.

## Existing Contract

The existing deterministic fixture contains:

- 50,000 Poisson observations;
- 20 independent cubic regression splines;
- 20 interior knots per spline;
- three signal terms and 17 noise terms;
- 420 fitted design columns and 20 smoothing parameters;
- an ordinary exact `fit_reml(discrete=False)` fit;
- a discrete cached-working-system `fit_reml(discrete=True)` fit.

It asserts that both fits converge, their deviances differ by at most 0.1%
relative, and their fitted-mean vectors have correlation of at least 0.999.
The distinctive coverage is the many-penalty, mostly-noise REML geometry. The
50,000-row POI workload itself is inexpensive; the live exact comparison is
not.

## Measured Cause

Measurements used one process and one thread per numerical runtime on commit
`28b04a385dda38cbae1131ba625bcb6e0261f16d` with Python 3.14.4, NumPy 2.4.2,
SciPy 1.17.1, and pandas 3.0.1.

| Route | Complete fit | REML weight correction | Outer iterations |
|---|---:|---:|---:|
| Exact, original fixture | 231.71 s | 205.95 s | 9 |
| Discrete POI, original fixture | 2.28 s | 0 s | 9 |

The exact and discrete results were already much closer than the documented
contract:

- exact deviance: 54,426.173838495466;
- discrete deviance: 54,426.71777640764;
- relative deviance difference: approximately 0.001%;
- exact EDF: 49.25185295070702;
- discrete EDF: 49.244820521030064;
- EDF difference: approximately 0.007;
- 64-point prediction-probe correlation: 0.9999791937.

Reducing the live fixture to 1,000 rows and two interior knots per spline made
the exact/discrete pair run in approximately 5.6 seconds, but reduced the
design from 420 columns to 60. That is a useful negative alternative, not the
chosen design, because it no longer exercises the original wide basis.

## Chosen Design

### Routine test

The replacement test will construct the same deterministic 50,000-row,
20-spline, 20-knot fixture and run only the production discrete POI fit. It will
compare that result with a compact, committed oracle generated once by the
known-good exact fit.

The oracle will live in a small human-readable fixture under `tests/fixtures/`.
It will contain:

- provenance: source commit and dependency versions;
- fixture-shape metadata;
- exact terminal deviance;
- exact terminal EDF;
- exact terminal REML/LAML objective;
- exact fitted means at 64 deterministic, evenly spaced training-row indices.

The fixture will not contain coefficient arrays, raw data, smoothing
parameters, or a large prediction vector. Noise-term smoothing parameters are
deliberately unsuitable oracle values because the protected REML surface is
flat in those directions.

### Assertions

The routine test will verify all of the following:

1. The discrete fit converges.
2. The built design still has 420 fitted columns.
3. The result still contains 20 fitted smoothing parameters.
4. POI-specific profile counters show that the discrete cached-working-system
   route executed: `reml_w_correction_s == 0.0`,
   `reml_n_analytical_iters > 0`, `reml_n_linesearch_full_evals > 0`, and
   `reml_n_outer_iter == n_reml_iter`.
5. The terminal deviance remains within the existing 0.1% relative contract
   of the exact oracle.
6. The terminal EDF remains within 0.25 of the exact oracle.
7. The terminal objective remains within 0.1% relative of the exact oracle.
8. The 64 deterministic probe predictions retain correlation of at least
   0.999 with the exact oracle.
9. The outer iteration count is between one and 15 inclusive.

The existing deviance and prediction tolerances will not be widened. New EDF,
objective, design-shape, smoothing-parameter-count, iteration, and dispatch
assertions strengthen the old test.

### Exact-solver responsibility

The wide test treats the exact fit as an oracle; it is not the primary exact
weight-correction test. Exact REML derivatives, working-weight corrections,
observed geometry, finite-difference agreement, rank handling, and terminal
publication remain covered by their focused test modules. Removing the live
exact fit from this one integration test therefore separates responsibilities:

- focused exact tests validate exact-solver mathematics;
- the frozen wide oracle validates POI fit quality on the original difficult
  geometry;
- focused cached-system tests validate individual discrete algebra and
  fallback behaviour.

## Oracle Maintenance

Oracle changes require an intentional numerical review. A maintainer must not
update reference values merely because the test fails.

When an approved exact-solver correction legitimately changes the oracle:

1. run the deterministic fixture with `discrete=False` on the reviewed exact
   implementation;
2. record the source commit and dependency versions;
3. compare the new exact result with the prior oracle and explain every
   material difference;
4. run the discrete POI fit and confirm that the public quality contract still
   holds;
5. update the compact fixture in the same reviewed change.

The test helper will keep fixture construction in one place so exact oracle
regeneration cannot silently use different data or spline settings.

## CI and Duration Data

After implementing the replacement, regenerate or update `.test_durations` so
`pytest-split` no longer treats this node as a 232-second test. Verify exact
four-way collection coverage and run all four shards. The expected routine
test time is approximately 2–4 seconds locally, subject to machine variance.
Timing is evidence for the redesign, not a brittle assertion inside pytest.

## Validation

Validation must include:

- a red/green regression cycle for the new oracle assertions;
- the focused cached-W validation module;
- focused exact REML derivative and observed-geometry modules;
- all four duration-balanced pytest shards with no duplicate or missing node;
- Ruff check and format check;
- supported-Python CI execution;
- comparison of final test duration with the 232.6-second recorded baseline;
- confirmation that no production source file changed.

## Non-goals

This change will not:

- optimise or alter exact REML weight correction;
- alter POI, IRLS, PIRLS, Tabmat, or design-matrix code;
- weaken numerical tolerances;
- reduce the original row count, feature count, knot count, or noise-term
  count;
- delete focused exact or discrete unit tests;
- add a general oracle framework;
- add a timing assertion to a correctness test.
