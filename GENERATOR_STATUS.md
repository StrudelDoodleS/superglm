# Generator Status — Round 2

## Summary
All 5 reviewer findings from round 1 have been addressed. 4 new tests added. Full test suite passes with 0 failures.

## Requirements Completed

### Group 1: Diagnostic Plots
| Req | Description | Status |
|-----|-------------|--------|
| R1 | `plot_diagnostics()` with 2x2 panel layout (resid vs fitted, Q-Q, scale-location, leverage) | PASS |
| R2 | Works for all 6 families (Poisson, Gaussian, Gamma, Binomial, NB2, Tweedie) | PASS |
| R3 | `residual_type` forwarded to Panel 1; Panels 2-4 use standardized deviance | PASS |
| R4 | Suptitle shows family name and link function | PASS |

### Group 2: Actuarial Validation Toolkit
| Req | Description | Status |
|-----|-------------|--------|
| R5 | `superglm.validation` module with 4 public functions accepting y_obs, y_pred, sample_weight, exposure | PASS |
| R6 | `lift_chart` with equal-exposure quantile bins and LiftChartResult dataclass | PASS |
| R7 | `double_lift_chart` binned by model A, shows both models' A/E ratios | PASS |
| R8 | `lorenz_curve` with gini_model, gini_perfect, gini_ratio | PASS |
| R9 | `loss_ratio_chart` with optional feature_values binning | PASS |
| R10 | `sample_weight` correctly used in all binning and aggregation | PASS |
| R11 | Gini via trapezoidal rule: `1 - 2 * AUC(lorenz)` | PASS |

### Group 3: Model Adequacy Tests
| Req | Description | Status |
|-----|-------------|--------|
| R12 | `superglm.model_tests` module with 4 public functions | PASS |
| R13 | `zero_inflation_index` for Poisson and NB2, ValueError for others | PASS |
| R14 | `score_test_zi` van den Broek (1995) score test | PASS |
| R15 | `dispersion_test` Cameron & Trivedi (1990) with greater/less/two-sided | PASS |
| R16 | `vuong_test` Vuong (1989) with none/aic/bic corrections | PASS |

## Reviewer Findings Addressed (Round 1 -> Round 2)

1. **[HIGH] Lorenz curve diagonal** — `cum_loss_share_ordered` now set to `cum_exp_m` instead of `linspace(0,1,n+1)`. Plot "Random" line changed to `[0,1]->[0,1]` (always a straight diagonal). Regression test added: `test_lorenz_nonuniform_exposure_diagonal` asserts `cum_loss_share_ordered == cum_exposure_share`.

2. **[MEDIUM] sample_weight silently ignored** — `score_test_zi` and `dispersion_test` now raise `NotImplementedError("sample_weight is not yet supported...")` when `sample_weight` is not None. Tests `TestScoreTestZIWeights` and `TestDispersionTestWeights` verify this.

3. **[LOW] Dead code** — Removed the dead `denom = np.sum(p0 * (1 - p0) - mu**2 * p0)` line that was immediately overwritten.

4. **[LOW] zero_inflation_index extra kwargs** — Acknowledged as acceptable deviation. The function takes raw arrays (not a model), so `family` and `theta` kwargs are necessary for the computation.

5. **[LOW] Vuong test weight handling** — Replaced `m = m * w` followed by unweighted `mean(m)` / `std(m)` with proper weighted mean `sum(w*m)/sum(w)` and weighted variance `sum(w*(m-mean)^2)/sum(w)`.

## Tests Added/Modified

### New tests in round 2 (4):
- `test_validation.py::TestLorenzCurveGini::test_lorenz_nonuniform_exposure_diagonal`
- `test_validation.py::TestLorenzCurveGini::test_lorenz_nonuniform_exposure_gini_bounds`
- `test_model_tests.py::TestScoreTestZIWeights::test_weights_raise`
- `test_model_tests.py::TestDispersionTestWeights::test_weights_raise`

### Test infrastructure:
- Added `autouse` `_close_figures` fixture to `test_validation.py`

### All tests (cumulative):
| File | Tests | Result |
|------|-------|--------|
| `tests/test_plot_diagnostics.py` | 17 tests (T1-T5) | All pass |
| `tests/test_validation.py` | 27 tests (T6-T12 + 2 new) | All pass |
| `tests/test_model_tests.py` | 24 tests (T13-T26 + 2 new) | All pass |

## Files Changed (Round 2)
- `src/superglm/validation.py` — Lorenz curve diagonal fix (lines 373, 385)
- `src/superglm/model_tests.py` — NotImplementedError for sample_weight in score_test_zi/dispersion_test, dead code removal, Vuong weighted mean/std
- `tests/test_validation.py` — 2 new tests, autouse plt.close() fixture
- `tests/test_model_tests.py` — 2 new tests

## Known Gaps / Deferred Items
- `sample_weight` in `score_test_zi` and `dispersion_test` raises `NotImplementedError` rather than implementing weighted versions. Implementing weighted score tests requires careful derivation of the weighted variance terms — deferred to a future enhancement.
- `zero_inflation_index` has `family`/`theta` kwargs beyond PLAN spec — necessary and acceptable.

## All Existing Tests Pass
Yes — 1519 passed, 75 skipped, 0 failures, 2 warnings (pre-existing).
