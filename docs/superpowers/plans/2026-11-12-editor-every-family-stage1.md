# Every family in the editor, stage 1: the display follows the link — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** every scalar-response SuperGLM family × link and every scalar SuperLSS family opens and runs every editor step, each curve drawn in a quantity true for its link, with a test that walks every exported family.

**Architecture:** one shared module decides a term's display from its fitted link (`relativity`, `inverse_theta`, new `additive`); parameter names come from `ParameterSpec` instead of an editor table; `open.py` drops its family allow-list; the distribution panel and Explain tab keep their browser maths for the four families that have it and answer every other family with one sentence until stage 2.

**Tech Stack:** Python 3.12–3.14 (numpy, pandas), browser ES modules checked by `tsc -p jsconfig.json`, node:test, pytest + Playwright.

**Spec:** `docs/superpowers/specs/2026-11-12-editor-every-family-design.md` (sections 5 and 8; section 6 is stage 2, section 7 stage 3). Approved by Max 2026-11-12 ("yes on both recommendations").

## Global Constraints

- Nothing a log-link model shows today changes: every log-link SuperGLM and today's four SuperLSS families (LogNormalLS mean form, GammaLS, NegativeBinomialLS, TweedieLSS) keep every label, number and payload key. `tests/editor_frontend/fixtures/display_transforms.json`'s existing entries stay byte-identical.
- No editor module decides by a family's class or name, except the browser fast-path table in `src/superglm/editor/lss/panel.py` (`_FAST_PATHS`), which only picks browser maths.
- Saved sessions store link-scale effects (`edited_log_effect`); no change to what is stored.
- User text follows the vocabulary rule: Mean, CV, Overdispersion (chip 1/θ), Dispersion, Power stay as they are; no "Average" or "Spread" as a parameter's name; plain language, no status jargon.
- No drag does new per-frame server work (editor-smoothness hard limit).
- Every focused test run, browser run and suite run goes through `benchboard heavy <agent>-<job> -- timeout N ...`; the two browser suites run as separate pytest commands with `--run-browser`.
- Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Never push; the orchestrator pushes.

## Review Focus

1. A Gaussian/identity SuperGLM with a response in the thousands: Increase must move a selection by a visible, sensible amount (about what ×1.05 would on an average row), not by 0.049 units. → Task 3 test `test_additive_increase_moves_an_average_row_by_about_five_percent`.
2. A drag below zero on an additive curve: the browser must not clamp at 1e-12 (`interactions.js` and `ordered_spline.js` call `Math.max(1e-12, …)` today). → Task 4 node test `additive drag keeps negative values`.
3. A SuperLSS parameter fitted on an overridden link (log scale on a mean that defaults to log, identity on a scale): opens, and its curve kind follows the fitted link, not the default. → Task 5 test `test_overridden_link_draws_by_the_fitted_link`.
4. A model saved by master (pickle with the old `ParameterSpec`) loads and opens. → Task 1 test `test_parameter_spec_pickled_without_labels_loads`.
5. A TweedieLSS with terms on the power: opens; power curves are additive on the bounded-power link scale; the panel still draws (its browser maths reads the power per row). → Task 5 test `test_tweedie_power_terms_open_and_draw`.

---

### Task 1: `ParameterSpec` carries the name a user reads

**Files:**
- Modify: `src/superglm/distributional/family.py` (`ParameterSpec`, ~line 239)
- Modify: every built-in family's `ParameterSpec(...)` calls under `src/superglm/distributional/families/` (gamma, log_normal, negative_binomial, tweedie, gaussian, generalized_gamma, generalized_pareto, log_logistic, lomax, two_piece, dirichlet, `_declared.py` if it builds specs)
- Test: `tests/test_distributional_parameter_labels.py` (new, small)

**Interfaces:**
- Produces: `ParameterSpec.label: str | None = None`, `ParameterSpec.description: str | None = None`, `ParameterSpec.quantity: str | None = None` (frozen dataclass fields after `curvature`). `__post_init__` accepts `None` or a non-empty string for each.

- [ ] **Step 1: Write the failing tests**
  - `test_every_built_in_parameter_has_a_label`: for each exported SuperLSS family (instantiate DirichletLSS with `components=("a","b","c")`), every `parameter.label` is a non-empty string and `parameter.description` a sentence ending in ".".
  - `test_today_s_editor_labels_are_declared`: `GammaLS` → mean "Mean", scale "CV"; `LogNormalLS` → scale label "CV", quantity "σ"; `NegativeBinomialLS` theta → "Overdispersion"; `TweedieLSS` → "Mean", "Dispersion", "Power".
  - `test_parameter_spec_pickled_without_labels_loads`: `old = object.__new__(ParameterSpec)`; `old.__dict__.update(name=..., default_link=..., role=..., support=..., curvature="observed")` with none of the new keys (what master pickles); `loaded = pickle.loads(pickle.dumps(old))`; assert `loaded.label is None and loaded.quantity is None`. Also open a GammaLS fit pickled that way in the editor.
  - `test_label_must_be_text`: `ParameterSpec(..., label="")` raises `ValueError`.
- [ ] **Step 2: Run** `benchboard heavy <agent>-t1 --cores 1 -- timeout 300 uv run pytest tests/test_distributional_parameter_labels.py -q`. Expected: FAIL (unknown keyword `label`).
- [ ] **Step 3: Implement** the three fields and set them on every built-in parameter. Labels to use (copy exactly): mean "Mean"; GammaLS scale "CV"; LogNormalLS scale "CV" quantity "σ"; NegativeBinomialLS theta "Overdispersion" (description "1/θ: variance = mean + mean²/θ.") quantity "overdispersion"; TweedieLSS dispersion "Dispersion", power "Power"; location "Location"; GaussianLS scale "Standard deviation" quantity "σ"; GeneralizedGamma scale "σ" quantity "σ", shape "Shape"; GeneralizedPareto scale "Scale", shape "Tail index"; Lomax and LogLogistic scale "Scale", shape "Shape"; two-piece scale "Scale" quantity "σ", skew "Skew"; Dirichlet components: the component's name. Descriptions: one plain sentence each; reuse the editor's current `_FULL_NAMES` text for the five it has.
- [ ] **Step 4: Run** the test file and `tests/test_lss_*family*` smoke: `benchboard heavy <agent>-t1 -- timeout 600 uv run pytest -n 4 tests/test_distributional_parameter_labels.py tests/test_lss_serialization*.py -q`. Expected: PASS.
- [ ] **Step 5: Commit** `Distributional families declare the name a user reads for each parameter`.

### Task 2: one display module, decided by the fitted link

**Files:**
- Create: `src/superglm/editor/display.py` (moves `DisplayTransform`, `RELATIVITY`, `INVERSE_THETA`, `ParameterDisplay`, `parameter_display`, `y_label_for`, `display_for`, `FoldTermDisplay`, `fold_chart_display`, `directed_operation` out of `lss/display.py`)
- Modify: `src/superglm/editor/lss/display.py` (keeps `lss_history_fields`, `lss_term_fields`; re-exports the moved names so `superglm.editor.lss` imports keep working)
- Modify: `src/superglm/editor/payloads.py` (SuperGLM `y_label` from the display), `src/superglm/editor/cv.py` (`_plain_display` uses the model's link), `src/superglm/editor/widget.py` (imports)
- Test: `tests/test_editor_display.py` (new); extend `tests/test_editor_lss_payloads.py` fixture writer for the `additive` entry

**Interfaces:**
- Consumes: Task 1's `ParameterSpec.label/description/quantity/role`.
- Produces:
  - `Kind = Literal["relativity", "inverse_theta", "additive"]`; `ADDITIVE = DisplayTransform("additive")`; `DisplayTransform.to_display/from_display/increasing/no_effect` (`no_effect` is 1.0 for the multiplicative kinds, 0.0 for additive); `DisplayTransform.to_payload() -> {"kind": ..., "link": str}` where `link` is a short link name for labels (`"log"`, `"floored_log"`, `"logit"`, `"identity"`, else the link class name without `Link`, lower-cased).
  - `transform_for_link(link: Link, *, reciprocal: bool = False) -> DisplayTransform`: `LogLink` or a shifted log (`superglm.links._is_shifted_log_link`) or `LowerBoundedLogLink` → relativity (inverse_theta when `reciprocal`); `LogitLink` → relativity; everything else → additive. Classify by type/probe, never by family.
  - `display_for(model, key: str) -> DisplayTransform`: SuperGLM → `transform_for_link(model._link)`; SuperLSS view → the fitted link of the key's parameter (`view._fitted.layout.predictors[i].link`), `reciprocal=spec.role == "size"`.
  - `parameter_display(spec: ParameterSpec, *, average: bool) -> ParameterDisplay` with `label = spec.label or spec.name.replace("_", " ").capitalize()`, `chip` ("1/θ" for role `size`, else the label), `role` (`"average"` when `average` else `"spread"`), `colour`, `full_name` (description or label).
  - `average_parameter(family) -> str`: the parameter with role `mean`, else role `location`, else the first declared.
  - `y_label_for(transform, quantity: str) -> str`: relativity → `"relativity"` when quantity is "mean" on a SuperGLM, else `f"relativity of the {quantity}"` (σ without "the": `"relativity of σ"`); logit → `f"relativity of the odds"` (SuperGLM) / `f"relativity of the odds of the {quantity}"`; inverse_theta → `"relativity of the overdispersion"`; additive on identity → `f"effect on the {quantity}"`; additive otherwise → `f"effect on the {link} scale"`.

- [ ] **Step 1: Write the failing tests** in `tests/test_editor_display.py`:
  - `test_transform_follows_the_link` parametrized over `LogLink`, `LowerBoundedLogLink(0.01)`, `LogitLink`, `IdentityLink`, `InverseLink`, `ProbitLink`, `CloglogLink`, `SqrtLink`, `BoundedLogitLink(-0.9, 0.9)`, `BoundedPowerLink(1.05, 1.95)` → expected kinds `relativity, relativity, relativity, additive × 7`.
  - `test_size_role_shows_the_reciprocal`: log link with `reciprocal=True` → `inverse_theta`.
  - `test_additive_maps_are_identity`: `ADDITIVE.to_display([-3, 0, 2.5]) == [-3, 0, 2.5]`, `from_display` the same, `no_effect == 0.0`, `increasing is True`.
  - `test_glm_identity_payload_shows_the_effect`: Gaussian/identity SuperGLM (reuse `tests.test_editor.editor_model`): `session_payload(session)[term]["y"]` equals `term.edited_log_effect` to 1e-12 and `y_label == "effect on the mean"`; display kind `additive`.
  - `test_glm_logit_payload_is_an_odds_relativity`: Binomial/logit: `y == exp(effect)`, `y_label == "relativity of the odds"`.
  - `test_glm_log_payload_is_unchanged`: Poisson/log: `y_label == "relativity"`, `display["kind"] == "relativity"`.
  - `test_cv_fold_items_follow_the_glm_link`: Gaussian/identity fold items' `fit` equal the payload's `original_y`.
  - Add an `"additive"` entry to `display_transforms.json` (effects, display, probes, effects_from_probes) in the fixture writer test; existing entries unchanged.
- [ ] **Step 2: Run** `benchboard heavy <agent>-t2 -- timeout 600 uv run pytest -n 4 tests/test_editor_display.py -q` → FAIL (module missing).
- [ ] **Step 3: Implement** the module and its use in `payloads.py` (SuperGLM `y_label` and `display` key: add `"display": transform.to_payload()` only when the kind is not `relativity`, so a log-link SuperGLM payload keeps `GLM_TERM_KEYS` exactly), `cv.py`, `widget.py`, `lss/display.py`. LSS labels come from Task 1's specs; the editor's `_LABELS`, `_CHIPS`, `_FULL_NAMES`, `_ROLES`, `SPREAD_FAMILIES` and `_LOGNORMAL_Y_LABEL` tables are deleted.
- [ ] **Step 4: Run** the new file plus every existing editor test that imports display names: `benchboard heavy <agent>-t2 -- timeout 900 uv run pytest -n 8 tests/test_editor_display.py tests/test_editor_lss_*.py tests/test_editor_cv.py -q` → PASS, with `test_editor_lss_payloads.py`'s fixture check still byte-identical for the old entries.
- [ ] **Step 5: Commit** `Editor: a term's display follows its fitted link`.

### Task 3: tools on an additive curve (Python)

**Files:**
- Modify: `src/superglm/editor/widget.py` (`_operate` shift_up/shift_down; `_average_relativity` already maps through the display)
- Modify: `src/superglm/editor/session.py` only if a tool assumes a positive display (check `linear_interpolate`, `set_values`, `isotonic`, `smooth`, snaps)
- Modify: `src/superglm/editor/payloads.py` (`_impact_payload`'s `weighted_mean_relativity` → also `weighted_mean_display` when additive; the existing key stays for relativity kinds)
- Modify: `src/superglm/editor/lss/rules.py`, `src/superglm/editor/shapes.py` if they map caps or shaped ranges through `exp`
- Test: `tests/test_editor_display.py`

**Interfaces:**
- Consumes: Task 2's `display_for`, `ADDITIVE`.
- Produces: `EditorSession.additive_step(term: str) -> float`: 0.05 × the exposure-weighted mean of |η| of the term's linear predictor over the training rows (SuperGLM: `model` linear predictor; SuperLSS: that parameter's η), cached per fit identity (`self._fit_identity(self.model)`); falls back to 0.05 × (max − min) of the term's fitted curve when that mean is 0, and to 0.05 when both are 0. `widget._operate` uses `±additive_step` on an additive term and `log(1.05)`/`log(0.95)` otherwise.

- [ ] **Step 1: Write the failing tests:**
  - `test_additive_increase_moves_an_average_row_by_about_five_percent`: Gaussian/identity SuperGLM on y ≈ 3000 + effects; Increase on one selected point raises its effect by `additive_step`, which lies within 1% of `0.05 * np.average(np.abs(model.predict(X)), weights=w)`... (for identity, η = prediction).
  - `test_log_increase_is_unchanged`: Poisson/log Increase is `log(1.05)` on the effect.
  - `test_every_tool_runs_on_an_additive_curve`: on Gaussian/identity, for each of Average, Straighten selection, Smooth, Make increasing, Make decreasing, Snap highest/lowest, Move (`set_values` to a negative display value), undo: the edited effect is finite and the payload's `y` equals the effect; Move to −50 stores −50 (no clamp).
- [ ] **Step 2: Run** them → FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `tests/test_editor_display.py tests/test_editor.py -k "increase or average or straighten or smooth or isotonic or snap"` → PASS.
- [ ] **Step 5: Commit** `Editor: every tool works on an additive curve`.

### Task 4: the browser draws additive curves

**Files:**
- Modify: `src/superglm/editor/app/chart/display.js` (kind `additive`; `noEffect(spec)`; `reachableDisplay` floors only multiplicative kinds; `showsRelativity` unchanged meaning)
- Modify: `src/superglm/editor/app/api/contracts.js` (`DisplaySpec` kind union and `link`; the comment "SuperLSS terms only" becomes "absent means relativity")
- Modify: `src/superglm/editor/app/interactions.js` (lines ~601, ~818, ~948: replace `Math.max(1e-12, …)` with `reachableDisplay(spec, …)`), `src/superglm/editor/app/chart/ordered_spline.js` (~76, same), `src/superglm/editor/app/chart.js` (baseline line ~240 at `noEffect(term.display)`; axis title from `term.y_label`), `src/superglm/editor/app/chart/sparkline.js` (bar baseline `noEffect(term.display)`), `src/superglm/editor/app/views/context_bar.js` (~23: an additive term's readouts say "+12.3" / "−4", not "×1.05"), `src/superglm/editor/app/distribution/preview.js` (unchanged maths; confirm it only runs for fast-path families)
- Test: `tests/editor_frontend/display.test.js` (reads the new `additive` fixture entry), `tests/editor_frontend/sparkline.test.js`, `tests/editor_frontend/context_bar.test.js`, `tests/editor_frontend/interactions.test.js`

**Interfaces:**
- Consumes: payload `term.display = {kind: "additive", link: "identity"}` from Task 2.
- Produces: `noEffect(spec): number` exported from `chart/display.js`.

- [ ] **Step 1: Write the failing node tests:** `additive maps are identity and unclamped` (fixture-driven); `additive drag keeps negative values` (a drag from 2 to −3 leaves `term.y` at −3); `an additive sparkline's bars stand on zero`; `an additive context readout adds`.
- [ ] **Step 2: Run** `benchboard heavy <agent>-t4 --cores 1 -- timeout 300 npm run check:frontend` → FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the same command → PASS (typecheck and every node test).
- [ ] **Step 5: Run the browser check** `benchboard heavy <agent>-t4b --cores 2 -- timeout 900 uv run pytest tests/test_editor_browser.py --run-browser -q -k "relativity or drag or baseline"` and, separately, `tests/editor/ --run-browser -q -k "lss or chart"` → PASS. Add one browser test in `tests/editor/test_editor_workspace_browser.py`: `test_identity_link_chart_draws_effects_around_zero` (Gaussian/identity model: the zero line's y equals `sy(0)`, axis title "effect on the mean", a drag below zero holds).
- [ ] **Step 6: Commit** `Editor: the browser draws and edits additive curves`.

### Task 5: every scalar SuperLSS family opens

**Files:**
- Modify: `src/superglm/editor/lss/open.py` (delete the family allow-list, the log-normal form refusal, the Tweedie-power refusal and `_overridden_link`; keep the rest in order; module docstring updated)
- Modify: `src/superglm/editor/lss/panel.py` (`_FAMILIES` → `_FAST_PATHS: dict[type, str]` with a predicate for LogNormalLS mean form; `fast_path(family) -> str | None`; `_parameters` uses Task 2's `parameter_display` and `average_parameter`; `link_spec` returns `{"kind": "identity"}`, `{"kind": "logit"}`, `{"kind": "bounded_logit", lower, upper}` for `BoundedLogitLink` too, and `{"kind": "other"}` instead of raising; a family with no fast path answers `kind: "empty"` with `empty_reason = NOT_DRAWN_YET`)
- Modify: `src/superglm/editor/lss/risk.py` (`_family_name` → `fast_path`; no fast path → `{"kind": "unavailable", "reason": NOT_DRAWN_YET, ...}` keeping `features` and `parameters` so the drivers table still lists terms)
- Modify: `src/superglm/editor/app/views/risk_panel.js`, `risk_drivers.js` (an `unavailable` reply shows its reason; drivers show "+" for an additive parameter), `distribution_panel.js` (unknown family → its `empty_reason`)
- Modify: `src/superglm/editor/lss/view.py`, `lss/terms.py`, `lss/apply.py`, `lss/refit.py`, `lss/rules.py`, `lss/validation.py` wherever they read a parameter by name or assume a log link (`rg -n "\"scale\"|\"theta\"|\"mean\"|LogLink|np\.exp" src/superglm/editor/lss`)
- Test: `tests/test_editor_lss_open.py` (the refusal tests for those four conditions become open tests), `tests/test_editor_lss_panel.py`, `tests/test_editor_lss_risk.py`

**Interfaces:**
- Consumes: Tasks 1–4.
- Produces: `NOT_DRAWN_YET = "The distribution panel does not draw {family} models yet; the rest of the editor works for them."` (formatted with the class name), `fast_path(family) -> str | None`.

- [ ] **Step 1: Write the failing tests:** `test_every_scalar_family_opens` (each of GaussianLS, GeneralizedGammaLSS, GeneralizedParetoLSS, LogLogisticLSS, LomaxLSS, TwoPieceLogNormalLSS, TwoPieceNormalLSS, LogNormalLS(parametrisation="location" or whichever non-mean forms exist) fits on 400 rows and opens; payload builds; panel answers `NOT_DRAWN_YET`; risk answers `unavailable`); `test_overridden_link_draws_by_the_fitted_link`; `test_tweedie_power_terms_open_and_draw` (power with `s("age", k=4)`: opens, power terms are additive, the panel payload's kind is `numeric`); `test_gaussian_location_is_blue_and_additive` (role `average`, kind `additive`, `y_label == "effect on the location"`); `test_lomax_scale_is_the_average_parameter`; `test_rating_table_of_an_additive_parameter_gives_the_library_s_reason` (GaussianLS location: the editor's rating-table route answers the library's refusal sentence, not a table and not a crash).
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `benchboard heavy <agent>-t5 -- timeout 1200 uv run pytest -n 8 tests/test_editor_lss_*.py -q` → PASS; then `npm run check:frontend`; then the LSS browser files `tests/editor/test_editor_lss_*browser.py --run-browser` → PASS.
- [ ] **Step 5: Commit** `Editor: every scalar SuperLSS family opens`.

### Task 6: the walker, the source rule, docs

**Files:**
- Create: `tests/test_editor_every_family.py`
- Create: `tests/test_editor_names_no_family.py`
- Modify: `docs/tutorials/edit-a-model-in-the-browser.md` (which families open; what a curve shows per link, in the table of spec section 5, plain words; the Explain/panel sentence for families without browser maths), `docs/api/editor.md` if it lists families
- Modify: `.test_durations` via `scripts/update_test_durations.py` if `tests/test_ci_contracts.py` asks

**Interfaces:**
- Consumes: everything above.

- [ ] **Step 1: Write the walker** (spec section 8): `EXAMPLES: dict[str, Example]` keyed by class name, each with natural parameters for an intercept-only law (e.g. GammaLS mean 1000, scale 0.6) or a `sample(rng, n)` for a family without `quantile`; `test_every_exported_family_has_an_example` (enumerates `superglm.distributional.families.__all__` classes that have `parameters`, and `superglm.distributions` `Distribution` classes; a missing one fails with "add <Name> to EXAMPLES in tests/test_editor_every_family.py"); `test_family_works_in_the_editor[<case>]` for every SuperLSS family (Dirichlet excluded with the stage-3 note, asserting it still refuses with the vector-response sentence) and every SuperGLM case of spec section 8 (default links plus Gaussian/log, Gamma/inverse, Binomial/log, Binomial/probit, Binomial/cloglog, Poisson/sqrt). Each case: 300 rows, `s("x", k=5)` and `cat("g")` on the first parameter, `s("x", k=4)` on the second, intercepts elsewhere; drives open → `session_payload` → shift on each term → undo → `widget._state()` → `session.to_model()` → the Final fit path `widget._final_fit_for_export()` → joblib export → `fold_term_items` on a 2-fold `cross_validate` for one case per response kind → validation scores → panel and risk payloads (SuperLSS) → rating table where the link is log.
- [ ] **Step 2: Write the source rule:** `test_editor_names_no_family`: no file under `src/superglm/editor/` (py and js) contains any exported family class name (`GammaLS`, `LogNormalLS`, …, `Poisson`, `Gamma(`, `Tweedie`, …) or a family string (`"gamma"`, `"tweedie"`, `"lognormal_mean"`, `"negative_binomial"`) outside `lss/panel.py`'s `_FAST_PATHS` block and `app/distribution/` + the views that dispatch on fast-path names; the allowed places are listed in the test with one comment each.
- [ ] **Step 3: Run** `benchboard heavy <agent>-t6 -- timeout 1500 uv run pytest -n 8 tests/test_editor_every_family.py tests/test_editor_names_no_family.py -q` → PASS; report the walker's wall time on `-n 8`. If it exceeds 60 s, mark the heaviest cases `slow` and keep one per link kind and per family in the quick set.
- [ ] **Step 4: Update docs; run** `uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/` and `tests/test_private_docstring_length.py tests/test_ci_contracts.py`.
- [ ] **Step 5: Commit** `Editor: a test walks every family through the editor`.

### Orchestrator, after Task 6

Quick suite (`scripts/run_test_suite.py -m "not slow and not browser and not docs"`), both browser suites, `npm run check:frontend`, a whole-branch review (one Opus reviewer + `sol-review`), then the PR with `release:minor`.
