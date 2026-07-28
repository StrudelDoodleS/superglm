# Weighted Refit and Holdout Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve sample weights in public refit diagnostics and give holdout term-drop diagnostics an explicit, validated evaluation-weight and evaluation-offset contract.

**Architecture:** Keep the public `SuperGLM` façade and `model.explain_ops` delegation layers intact. Resolve training-versus-validation argument ownership once in `diagnostics.term_diagnostics`, validate evaluation vectors before scoring, and retain the existing compact one-dimensional contribution path for RE, FS, and SZ terms.

**Tech Stack:** Python 3.10+, NumPy, pandas, pytest, Ruff, uv.

---

### Task 1: Preserve weights in `refit_unpenalised()`

**Files:**
- Modify: `tests/test_refit.py`
- Modify: `src/superglm/inference/_term_model_ops.py:312-358`

- [ ] **Step 1: Write a failing weighted-refit regression**

Add a deterministic Gaussian case whose high-weight rows imply a different
line from the unweighted rows:

```python
def test_refit_unpenalised_preserves_nonuniform_sample_weight():
    x = np.linspace(-2.0, 2.0, 160)
    y = 0.6 + 0.9 * x
    y[x > 0.8] += 2.5
    sample_weight = np.where(x > 0.8, 0.05, 3.0)
    X = pd.DataFrame({"x": x})

    model = SuperGLM(
        family="gaussian",
        features={"x": Numeric()},
        selection_penalty=0.0,
    ).fit(X, y, sample_weight=sample_weight)

    weighted = model.refit_unpenalised(X, y, sample_weight=sample_weight)
    unweighted = model.refit_unpenalised(X, y)

    np.testing.assert_allclose(weighted.result.beta, model.result.beta, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(
        weighted.result.intercept,
        model.result.intercept,
        rtol=1e-9,
        atol=1e-9,
    )
    assert not np.allclose(weighted.result.beta, unweighted.result.beta, rtol=1e-5, atol=1e-5)
```

- [ ] **Step 2: Run the regression and verify RED**

Run:

```bash
uv run pytest tests/test_refit.py::test_refit_unpenalised_preserves_nonuniform_sample_weight -q
```

Expected: FAIL because `weighted` and `unweighted` are identical after
`sample_weight` is deleted.

- [ ] **Step 3: Forward the weight to the cloned fit**

In `refit_unpenalised()`, remove `del sample_weight` and fit with both public
row vectors:

```python
new_model.fit(
    X,
    y,
    sample_weight=sample_weight,
    offset=offset,
)
```

- [ ] **Step 4: Verify GREEN and the existing refit surface**

Run:

```bash
uv run pytest tests/test_refit.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add tests/test_refit.py src/superglm/inference/_term_model_ops.py
git commit -m "Preserve weights in unpenalised refits"
```

### Task 2: Preserve weights in refit-mode term-drop diagnostics

**Files:**
- Modify: `tests/test_diagnostics.py`
- Modify: `src/superglm/diagnostics/term_diagnostics.py:158-175`

- [ ] **Step 1: Write a failing weighted `drop1()` parity regression**

Add a two-term Gaussian model so every reduced model is a genuine refit:

```python
def test_refit_drop_diagnostics_forwards_nonuniform_sample_weight():
    rng = np.random.default_rng(20260727)
    n = 220
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    y = 0.4 + 0.8 * x - 0.35 * z + rng.normal(scale=0.25, size=n)
    y[x > 1.0] += 1.5
    sample_weight = np.where(x > 1.0, 0.08, 2.5)
    X = pd.DataFrame({"x": x, "z": z})
    model = SuperGLM(
        family="gaussian",
        features={"x": Numeric(), "z": Numeric()},
        selection_penalty=0.0,
    ).fit(X, y, sample_weight=sample_weight)

    expected = model.drop1(X, y, sample_weight=sample_weight)
    unweighted = model.drop1(X, y)
    actual = model.term_drop_diagnostics(
        X,
        y,
        sample_weight=sample_weight,
        mode="refit",
    )

    columns = ["feature", "deviance_reduced", "delta_deviance", "statistic", "p_value"]
    pd.testing.assert_frame_equal(
        actual[columns].reset_index(drop=True),
        expected[columns].reset_index(drop=True),
        rtol=1e-10,
        atol=1e-10,
    )
    assert not np.allclose(
        actual["deviance_reduced"],
        unweighted["deviance_reduced"],
        rtol=1e-6,
        atol=1e-6,
    )
```

Import `Numeric` from `superglm` in this test module.

- [ ] **Step 2: Run the regression and verify RED**

Run:

```bash
uv run pytest tests/test_diagnostics.py::test_refit_drop_diagnostics_forwards_nonuniform_sample_weight -q
```

Expected: FAIL because `_drop_term_refit()` currently matches the unweighted
`drop1()` result.

- [ ] **Step 3: Forward both refit row vectors**

Change `_drop_term_refit()` to call:

```python
drop1_df = model.drop1(
    X,
    y,
    sample_weight=sample_weight,
    offset=offset,
)
```

- [ ] **Step 4: Verify GREEN**

Run:

```bash
uv run pytest tests/test_diagnostics.py::test_refit_drop_diagnostics_forwards_nonuniform_sample_weight -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add tests/test_diagnostics.py src/superglm/diagnostics/term_diagnostics.py
git commit -m "Preserve weights in refit diagnostics"
```

### Task 3: Add the explicit holdout weight and offset API

**Files:**
- Modify: `tests/test_diagnostics.py`
- Modify: `src/superglm/model/api.py:1433-1463`
- Modify: `src/superglm/model/explain_ops.py:215-240`
- Modify: `src/superglm/diagnostics/term_diagnostics.py:126-224`

- [ ] **Step 1: Write the failing offset-aware holdout regression**

Create separate training and validation portfolios with unequal row counts:

```python
def test_holdout_drop_uses_validation_weights_and_offset():
    rng = np.random.default_rng(20260728)
    n_train, n_val = 240, 80
    x_train = rng.normal(size=n_train)
    z_train = rng.normal(size=n_train)
    x_val = rng.normal(size=n_val)
    z_val = rng.normal(size=n_val)
    exposure_train = rng.uniform(0.2, 2.5, size=n_train)
    exposure_val = rng.uniform(0.1, 3.0, size=n_val)
    offset_train = np.log(exposure_train)
    offset_val = np.log(exposure_val)
    weights_train = rng.uniform(0.5, 2.0, size=n_train)
    weights_val = rng.uniform(0.4, 2.3, size=n_val)
    y_train = rng.poisson(
        np.exp(0.2 + 0.55 * x_train - 0.3 * z_train + offset_train)
    )
    y_val = rng.poisson(np.exp(0.2 + 0.55 * x_val - 0.3 * z_val + offset_val))
    X_train = pd.DataFrame({"x": x_train, "z": z_train})
    X_val = pd.DataFrame({"x": x_val, "z": z_val})
    model = SuperGLM(
        family="poisson",
        features={"x": Numeric(), "z": Numeric()},
        selection_penalty=0.0,
    ).fit(
        X_train,
        y_train,
        sample_weight=weights_train,
        offset=offset_train,
    )

    actual = model.term_drop_diagnostics(
        X_train,
        y_train,
        sample_weight=weights_train,
        offset=offset_train,
        mode="holdout",
        X_val=X_val,
        y_val=y_val,
        sample_weight_val=weights_val,
        offset_val=offset_val,
    )

    eta = model.result.intercept + offset_val
    contributions = {}
    for group in model._groups:
        contribution = X_val[group.feature_name].to_numpy() * model.result.beta[group.sl][0]
        contributions[group.feature_name] = contribution
        eta += contribution
    mu_full = model._link.inverse(stabilize_eta(eta, model._link))
    dev_full = np.sum(weights_val * model._distribution.deviance_unit(y_val, mu_full))
    expected = []
    for name, contribution in contributions.items():
        mu_drop = model._link.inverse(stabilize_eta(eta - contribution, model._link))
        dev_drop = np.sum(weights_val * model._distribution.deviance_unit(y_val, mu_drop))
        expected.append({"feature": name, "delta_deviance": dev_drop - dev_full})

    pd.testing.assert_frame_equal(
        actual.reset_index(drop=True),
        pd.DataFrame(expected).reset_index(drop=True),
        rtol=2e-10,
        atol=2e-10,
    )
```

Import `stabilize_eta` from `superglm.links`. Apply `clip_mu()` in the reference
if required by the fitted family, matching production's distribution boundary.

- [ ] **Step 2: Run the regression and verify RED**

Run:

```bash
uv run pytest tests/test_diagnostics.py::test_holdout_drop_uses_validation_weights_and_offset -q
```

Expected: `TypeError` because `sample_weight_val` and `offset_val` do not yet
exist.

- [ ] **Step 3: Add keyword propagation through the public façade**

Add `sample_weight_val=None` and `offset_val=None` after `y_val` in:

- `SuperGLM.term_drop_diagnostics()`;
- `model.explain_ops.term_drop_diagnostics()`;
- `diagnostics.term_diagnostics.term_drop_diagnostics()`.

Forward them by keyword at each layer:

```python
return _term_drop_diagnostics(
    model,
    X,
    y,
    sample_weight,
    offset,
    mode=mode,
    X_val=X_val,
    y_val=y_val,
    sample_weight_val=sample_weight_val,
    offset_val=offset_val,
)
```

- [ ] **Step 4: Resolve same-object fallback without value guessing**

Before converting `X_val`, compute:

```python
same_validation_rows = X_val is X and y_val is y

if sample_weight_val is None:
    if same_validation_rows:
        sample_weight_val = sample_weight
    elif sample_weight is not None:
        raise ValueError(
            "separate validation rows require sample_weight_val; "
            "training sample_weight is not reused by length"
        )

if offset_val is None:
    if same_validation_rows:
        offset_val = offset
    elif offset is not None or getattr(model, "_fit_used_offset", False):
        raise ValueError(
            "offset-based holdout diagnostics on separate validation rows "
            "require offset_val"
        )

if offset_val is None and getattr(model, "_fit_used_offset", False):
    raise ValueError(
        "holdout diagnostics for an offset-fitted model require offset_val "
        "or the matching training offset on same-object validation rows"
    )
```

Validation-specific arguments take precedence because these branches execute
only when the corresponding validation argument is `None`.

- [ ] **Step 5: Validate holdout vectors and include the offset**

Use `_finite_vector` from `superglm.model.input_validation` and
`validate_response` from `superglm.distributions`:

```python
n_val = len(X_val)
y_arr = _finite_vector(
    "y_val",
    y_val,
    n_val,
    require_nonempty=True,
    check_finite=False,
)
validate_response(y_arr, model._distribution)

w = (
    np.ones(n_val, dtype=np.float64)
    if sample_weight_val is None
    else _finite_vector("sample_weight_val", sample_weight_val, n_val)
)
if np.any(w < 0.0):
    raise ValueError("sample_weight_val must be nonnegative")
if not np.any(w > 0.0):
    raise ValueError("sample_weight_val must not be all zero")

offset_arr = (
    np.zeros(n_val, dtype=np.float64)
    if offset_val is None
    else _finite_vector("offset_val", offset_val, n_val)
)
eta_raw = np.full(n_val, model.result.intercept, dtype=np.float64)
eta_raw += offset_arr
```

Rename `_drop_term_holdout()` parameters to `sample_weight_val` and
`offset_val`, and pass the resolved values from the wrapper.

- [ ] **Step 6: Verify GREEN**

Run:

```bash
uv run pytest tests/test_diagnostics.py::test_holdout_drop_uses_validation_weights_and_offset -q
```

Expected: pass.

- [ ] **Step 7: Commit**

```bash
git add tests/test_diagnostics.py src/superglm/model/api.py src/superglm/model/explain_ops.py src/superglm/diagnostics/term_diagnostics.py
git commit -m "Add explicit holdout diagnostic geometry"
```

### Task 4: Lock down ambiguity, validation, and compact structured scoring

**Files:**
- Modify: `tests/test_diagnostics.py`
- Modify: `tests/test_structured_diagnostics.py`
- Modify: `src/superglm/diagnostics/term_diagnostics.py`

- [ ] **Step 1: Write failing compatibility and ambiguity tests**

Cover these calls individually:

```python
def _fit_offset_diagnostic_model():
    rng = np.random.default_rng(20260729)
    n = 120
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    exposure = rng.uniform(0.2, 2.5, size=n)
    offset = np.log(exposure)
    weights = rng.uniform(0.5, 2.0, size=n)
    y = rng.poisson(np.exp(0.15 + 0.45 * x - 0.2 * z + offset))
    X = pd.DataFrame({"x": x, "z": z})
    model = SuperGLM(
        family="poisson",
        features={"x": Numeric(), "z": Numeric()},
        selection_penalty=0.0,
    ).fit(X, y, sample_weight=weights, offset=offset)
    return model, X, y, weights, offset


def test_holdout_same_objects_reuse_training_weight_and_offset():
    model, X, y, weights, offset = _fit_offset_diagnostic_model()
    actual = model.term_drop_diagnostics(
        X,
        y,
        sample_weight=weights,
        offset=offset,
        mode="holdout",
        X_val=X,
        y_val=y,
    )
    explicit = model.term_drop_diagnostics(
        X,
        y,
        sample_weight=weights,
        offset=offset,
        mode="holdout",
        X_val=X,
        y_val=y,
        sample_weight_val=weights,
        offset_val=offset,
    )
    pd.testing.assert_frame_equal(actual, explicit)


def test_holdout_separate_rows_reject_training_weight_fallback(fitted_model):
    model, X, y, weights = fitted_model
    with pytest.raises(ValueError, match="sample_weight_val"):
        model.term_drop_diagnostics(
            X,
            y,
            sample_weight=weights,
            mode="holdout",
            X_val=X.copy(),
            y_val=y.copy(),
        )


def test_holdout_offset_fit_requires_validation_offset():
    model, X, y, weights, offset = _fit_offset_diagnostic_model()
    with pytest.raises(ValueError, match="offset_val"):
        model.term_drop_diagnostics(
            X,
            y,
            sample_weight=weights,
            offset=offset,
            mode="holdout",
            X_val=X.copy(),
            y_val=y.copy(),
            sample_weight_val=weights.copy(),
        )
```

Run the three tests and confirm each fails for the missing contract rather than
fixture setup.

- [ ] **Step 2: Write failing vector-validation tests**

Parametrize invalid `sample_weight_val` inputs:

```python
@pytest.mark.parametrize(
    ("weights", "message"),
    [
        (np.ones((4, 1)), "one-dimensional"),
        (np.ones(3), "length 4"),
        (np.array([1.0, np.nan, 1.0, 1.0]), "finite"),
        (np.array([1.0, -1.0, 1.0, 1.0]), "nonnegative"),
        (np.zeros(4), "all zero"),
    ],
)
def test_holdout_validates_sample_weight_val(fitted_model, weights, message):
    model, X, y, _ = fitted_model
    X_val = X.iloc[:4].copy()
    y_val = np.asarray(y[:4])
    with pytest.raises(ValueError, match=message):
        model.term_drop_diagnostics(
            X,
            y,
            mode="holdout",
            X_val=X_val,
            y_val=y_val,
            sample_weight_val=weights,
        )
```

Add corresponding offset checks:

```python
@pytest.mark.parametrize(
    ("offset_val", "message"),
    [
        (np.ones((4, 1)), "one-dimensional"),
        (np.ones(3), "length 4"),
        (np.array([0.0, np.inf, 0.0, 0.0]), "finite"),
    ],
)
def test_holdout_validates_offset_val(fitted_model, offset_val, message):
    model, X, y, _ = fitted_model
    with pytest.raises(ValueError, match=message):
        model.term_drop_diagnostics(
            X,
            y,
            mode="holdout",
            X_val=X.iloc[:4].copy(),
            y_val=np.asarray(y[:4]),
            offset_val=offset_val,
        )


def test_holdout_validates_y_val_length(fitted_model):
    model, X, y, _ = fitted_model
    with pytest.raises(ValueError, match="y_val must have length 4"):
        model.term_drop_diagnostics(
            X,
            y,
            mode="holdout",
            X_val=X.iloc[:4].copy(),
            y_val=np.asarray(y[:3]),
        )
```

Run the parametrized tests and verify RED.

- [ ] **Step 3: Complete minimal validation behavior**

Adjust only error ordering or messages needed for the failing tests. Do not
introduce generic cross-validation abstractions or alter fit-time validation.

- [ ] **Step 4: Extend the structured compactness regression**

Change `_dense_holdout_drop_reference()` to accept an `offset` vector and add it
to both full and dropped raw predictors. In
`test_holdout_drop_uses_compact_structured_score()`:

- use `X.copy()` and `y.copy()` as separate validation objects;
- pass non-uniform `sample_weight_val`;
- pass a finite nonzero `offset_val`;
- retain the `transform()` sentinel and exactly-one-`score()` assertion.

Run:

```bash
uv run pytest tests/test_structured_diagnostics.py -q
```

Expected: RE, FS, and SZ cases pass without expanded matrices.

- [ ] **Step 5: Run the complete focused diagnostic surface**

Run:

```bash
uv run pytest tests/test_refit.py tests/test_diagnostics.py tests/test_structured_diagnostics.py tests/test_dataframe_boundary_diagnostics.py tests/test_import_compat.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add tests/test_diagnostics.py tests/test_structured_diagnostics.py src/superglm/diagnostics/term_diagnostics.py
git commit -m "Validate holdout diagnostic geometry"
```

### Task 5: Document the public contract and verify the release surface

**Files:**
- Modify: `src/superglm/model/api.py`
- Modify: `src/superglm/diagnostics/term_diagnostics.py`
- Modify: `docs/api/diagnostics.md`

- [ ] **Step 1: Expand public docstrings**

Document:

- training/refit ownership of `sample_weight` and `offset`;
- holdout ownership of `sample_weight_val` and `offset_val`;
- same-object fallback;
- explicit rejection for ambiguous separate portfolios;
- offset inclusion in the validation predictor.

Remove the duplicated `sample_weight, sample_weight` typo in
`term_importance()` while editing the diagnostics module.

- [ ] **Step 2: Add a diagnostics guide example**

In `docs/api/diagnostics.md`, add a concise separate-validation example:

```python
drop = model.term_drop_diagnostics(
    X_train,
    y_train,
    sample_weight=weight_train,
    offset=np.log(exposure_train),
    mode="holdout",
    X_val=X_validation,
    y_val=y_validation,
    sample_weight_val=weight_validation,
    offset_val=np.log(exposure_validation),
)
```

State that validation vectors are never inferred by matching lengths.

- [ ] **Step 3: Format and lint the touched surface**

Run:

```bash
uv run ruff format src/superglm/model/api.py src/superglm/model/explain_ops.py src/superglm/inference/_term_model_ops.py src/superglm/diagnostics/term_diagnostics.py tests/test_refit.py tests/test_diagnostics.py tests/test_structured_diagnostics.py
uv run ruff check src/superglm/model/api.py src/superglm/model/explain_ops.py src/superglm/inference/_term_model_ops.py src/superglm/diagnostics/term_diagnostics.py tests/test_refit.py tests/test_diagnostics.py tests/test_structured_diagnostics.py
```

Expected: format unchanged after the first command and no lint findings.

- [ ] **Step 4: Run the full exact-head suite**

Run:

```bash
uv run pytest tests/ -q
```

Expected: no failures.

- [ ] **Step 5: Run packaging and smoke gates**

Run:

```bash
uv build
uv run python run_test.py
```

Expected: wheel and sdist build successfully and the end-to-end smoke script
prints `END-TO-END COMPLETE`.

- [ ] **Step 6: Commit documentation**

```bash
git add src/superglm/model/api.py src/superglm/diagnostics/term_diagnostics.py docs/api/diagnostics.md
git commit -m "Document weighted diagnostic evaluation"
```

### Task 6: Publish the exact head for review

**Files:**
- Modify remotely: PR `#165` body validation section

- [ ] **Step 1: Verify the committed diff and branch state**

Run:

```bash
git diff --check origin/master...HEAD
git status --short --branch
git log -6 --oneline
```

Expected: no whitespace errors, clean worktree, and the five focused commits
above the rebased structured-credibility history.

- [ ] **Step 2: Push normally**

Run:

```bash
git push origin feature/structured-credibility
```

Expected: push succeeds without force because the rebase was already published
before implementation.

- [ ] **Step 3: Update exact-head validation evidence**

Replace stale `bf6d0b4` wording in PR #165's validation section with the new
head SHA and the observed full-suite, focused-suite, Ruff, build, and smoke
results. Do not change release scope or mark the PR ready.

- [ ] **Step 4: Request exact-head Codex review**

Post a new `@codex review` comment that calls out:

- weighted refit parity;
- separate validation weight/offset ownership;
- same-object fallback;
- offset-aware deviance;
- RE/FS/SZ compact scoring.

- [ ] **Step 5: Resolve actionable review threads and verify CI**

Address only technically verified findings, resolve completed inline threads,
and rerun the exact relevant checks. Finish with:

```bash
gh pr checks 165
git status --short --branch
```

Expected: all checks pass, the worktree is clean, and PR #165 remains draft.
