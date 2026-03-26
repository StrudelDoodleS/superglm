# Harness Progress

## Task
Add three feature groups per FEATURE_PLAN.md: (1) model.plot_diagnostics() R-style 4-panel residual plots, (2) actuarial lift toolkit in superglm.validation (lift_chart, double_lift_chart, lorenz_curve, loss_ratio_chart with Gini — CAS RPM 2016 methodology), (3) model adequacy tests in superglm.model_tests (zero_inflation_index, score_test_zi van den Broek 1995, dispersion_test Cameron-Trivedi 1990, vuong_test 1989 with AIC/BIC corrections). Read FEATURE_PLAN.md for full spec.

## Status
Round: 1 / 4
Verdict: PENDING

## Completed
- **R1-R4 (Group 1: Diagnostic plots):** `plot_diagnostics()` method on SuperGLM, 4-panel layout (resid vs fitted, Q-Q, scale-location, leverage with Cook's D contours), all 6 families, lowess smoothers, suptitle with family/link, residual_type forwarding, figsize support.
- **R5-R11 (Group 2: Validation toolkit):** `superglm.validation` module with `lift_chart`, `double_lift_chart`, `lorenz_curve`, `loss_ratio_chart`. All return frozen dataclass results. Gini via trapezoidal rule. Equal-weight quantile binning. `sample_weight` and `exposure` support. `ax` parameter pattern.
- **R12-R16 (Group 3: Model adequacy tests):** `superglm.model_tests` module with `zero_inflation_index` (Poisson/NB2), `score_test_zi` (van den Broek 1995), `dispersion_test` (Cameron-Trivedi 1990), `vuong_test` (Vuong 1989 with none/aic/bic corrections). All return frozen dataclass results.
- **Exports:** All new types and functions exported from `superglm.__init__`.
- **Tests T1-T26:** All 64 new tests pass.
- **Full suite:** 1515 passed, 75 skipped, 0 failures.

## Outstanding
(none — all requirements implemented and tested)

## Round History
- **Round 1:** All three groups (R1-R16) implemented, all tests (T1-T26) pass, full suite green. Files changed: `src/superglm/plotting/diagnostics.py` (new), `src/superglm/validation.py` (new), `src/superglm/model_tests.py` (new), `src/superglm/model/api.py` (new method), `src/superglm/__init__.py` (exports), `tests/test_plot_diagnostics.py` (new), `tests/test_validation.py` (new), `tests/test_model_tests.py` (new).
