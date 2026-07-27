# Structured Diagnostics Compactness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep large random-effect inference and RE/FS/SZ diagnostics on compact covariance and scoring paths while making unsupported unpenalised variance-component refits fail intentionally.

**Architecture:** Extend the existing exact prediction scorer with a term-local coefficient entry point, then make diagnostics consume the cached prediction plan instead of rebuilding expanded design matrices. Route random-effect uncertainty through selected covariance diagonals and share one REML-only term preflight between model-comparison helpers.

**Tech Stack:** Python 3.10+, NumPy, pandas, pytest, Matplotlib, Ruff, MyPy, SuperGLM structured covariance accessors and prediction plans.

---

## File map

- Modify `src/superglm/inference/_term_covariance.py`
  - Dispatch random-effect pointwise uncertainty to selected covariance
    diagonals before any principal-block request.
- Modify `src/superglm/model/base.py`
  - Add the canonical exact scorer for a prediction-plan term and a
    term-local coefficient vector.
- Modify `src/superglm/diagnostics/term_diagnostics.py`
  - Score importance and holdout drops through the prediction plan.
- Modify `src/superglm/inference/_term_model_ops.py`
  - Centralise REML-only term discovery and reject unsupported unpenalised
    refits at entry.
- Modify `tests/test_random_effect_inference.py`
  - Cover the 256-coefficient covariance boundary through public surfaces.
- Create `tests/test_structured_diagnostics.py`
  - Prove RE, FS, and SZ diagnostics never call their expanded transforms and
    preserve their prior small-model results.
- Modify `tests/test_refit.py`
  - Verify the public `refit_unpenalised()` variance-component contract.

### Task 1: Route large random-effect SEs through selected diagonals

**Files:**
- Modify: `tests/test_random_effect_inference.py`
- Modify: `src/superglm/inference/_term_covariance.py:120-166`

- [ ] **Step 1: Write the failing 270-level public-surface regression**

Extend the covariance import in `tests/test_random_effect_inference.py`:

```python
from superglm.inference.covariance import (
    StructuredCovarianceAccessor,
    StructuredSlopeCovarianceAccessor,
)
```

Add this test beside `test_random_effect_generic_term_surfaces_are_consistent`:

```python
def test_large_random_effect_generic_term_surfaces_use_covariance_diagonal(monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, model, _, _, exposure = _fit_pair(
        fit_dense=False,
        n_levels=270,
        max_reml_iter=2,
    )
    report = model.random_effects("broker", exposure=exposure)
    covariance, _ = model._coef_covariance
    assert isinstance(covariance, StructuredSlopeCovarianceAccessor)

    def fail_selected_block(indices):
        del indices
        raise AssertionError("full covariance block requested")

    monkeypatch.setattr(covariance, "selected_block", fail_selected_block)

    term = model.term_inference("broker")
    assert term.se_log_relativity is not None
    np.testing.assert_allclose(
        term.se_log_relativity,
        report.table["posterior_se"].to_numpy(),
        rtol=2e-10,
        atol=2e-10,
    )

    relativities = model.relativities(with_se=True)["broker"]
    np.testing.assert_allclose(
        relativities["se_log_relativity"].to_numpy(),
        report.table["posterior_se"].to_numpy(),
        rtol=2e-10,
        atol=2e-10,
    )

    plot_data = model.plot_data(terms="broker")
    assert plot_data["terms"][0]["name"] == "broker"
    axes = model.plot(terms="broker")
    assert axes is not None
    plt.close("all")
```

- [ ] **Step 2: Run the regression and verify the block request fails**

Run:

```bash
rtk pytest tests/test_random_effect_inference.py::test_large_random_effect_generic_term_surfaces_use_covariance_diagonal
```

Expected: FAIL with `AssertionError: full covariance block requested`.

- [ ] **Step 3: Dispatch `RandomEffect` before materialising `Cov_g`**

In `feature_se_from_cov()` in
`src/superglm/inference/_term_covariance.py`, place the random-effect branch
immediately after `indices` is built:

```python
    indices = np.concatenate([np.arange(ag.start, ag.end) for ag in active_subs])

    if isinstance(spec, RandomEffect):
        from superglm.inference.covariance import covariance_selected_diagonal

        variance = covariance_selected_diagonal(Cov_active, indices)
        return cast(NDArray, np.sqrt(np.maximum(variance, 0.0)))

    Cov_g = Cov_active[np.ix_(indices, indices)]
```

Delete the later branch that extracts `np.diag(Cov_g)` for
`RandomEffect`. Do not alter block covariance handling for any other spec.

- [ ] **Step 4: Run focused inference and plotting tests**

Run:

```bash
rtk pytest tests/test_random_effect_inference.py -k "generic_term_surfaces or large_random_effect"
```

Expected: PASS, including the 270-level pointwise plotting surfaces.

- [ ] **Step 5: Commit the covariance fix**

```bash
rtk git add src/superglm/inference/_term_covariance.py tests/test_random_effect_inference.py
rtk git commit -m "Use compact random-effect covariance diagonals"
```

### Task 2: Add canonical local term scoring and compact term importance

**Files:**
- Create: `tests/test_structured_diagnostics.py`
- Modify: `src/superglm/model/base.py:336-353`
- Modify: `src/superglm/diagnostics/term_diagnostics.py:20-114`

- [ ] **Step 1: Create structured diagnostic fixtures and a failing importance test**

Create `tests/test_structured_diagnostics.py`:

```python
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from superglm import (
    FactorSmooth,
    LambdaPolicy,
    Numeric,
    RandomEffect,
    Spline,
    SuperGLM,
)


def _fit_structured_diagnostic_case(
    basis: str,
) -> tuple[SuperGLM, pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(20260727)
    n_levels = 5
    repeats = 24
    codes = np.repeat(np.arange(n_levels), repeats)
    rng.shuffle(codes)
    x = rng.uniform(-1.0, 1.0, size=len(codes))
    z = rng.normal(size=len(codes))
    labels = np.array([f"level-{code}" for code in codes], dtype=object)

    if basis == "re":
        effects = np.array([-0.45, -0.1, 0.2, 0.35, 0.0])
        y = 0.4 + 0.2 * z + effects[codes] + rng.normal(scale=0.12, size=len(codes))
        X = pd.DataFrame({"z": z, "group": labels})
        model = SuperGLM(
            family="gaussian",
            features={
                "z": Numeric(),
                "group": RandomEffect(lambda_policy=LambdaPolicy.fixed(1.3)),
            },
            selection_penalty=0.0,
            direct_solve="structured",
        )
    else:
        amplitudes = np.array([0.65, -0.4, 0.25, -0.55, 0.05])
        if basis == "sz":
            amplitudes -= amplitudes.mean()
        y = (
            0.4
            + 0.2 * z
            + amplitudes[codes] * (x + 0.25 * x**2)
            + rng.normal(scale=0.12, size=len(codes))
        )
        X = pd.DataFrame({"x": x, "z": z, "group": labels})
        policies = {"wiggle": LambdaPolicy.fixed(1.3)}
        if basis == "fs":
            policies |= {
                "null_0": LambdaPolicy.fixed(0.8),
                "null_1": LambdaPolicy.fixed(1.1),
            }
        features = {"z": Numeric()}
        if basis == "sz":
            features["x"] = Spline(
                n_knots=5,
                lambda_policy=LambdaPolicy.fixed(1.0),
            )
        model = SuperGLM(
            family="gaussian",
            features=features,
            interactions=[
                FactorSmooth(
                    "x",
                    group="group",
                    basis=basis,
                    k=6,
                    lambda_policy=policies,
                )
            ],
            selection_penalty=0.0,
            direct_solve="structured",
        )

    model.fit_reml(X, y, max_reml_iter=2, runtime_validation="skip")
    return model, X, y


@pytest.fixture(scope="module", params=["re", "fs", "sz"])
def structured_diagnostic_case(request):
    return _fit_structured_diagnostic_case(request.param)


def _structured_spec(model):
    if "group" in model._specs:
        return model._specs["group"]
    return next(
        spec
        for spec in model._interaction_specs.values()
        if isinstance(spec, FactorSmooth)
    )


def test_term_importance_uses_compact_structured_score(
    structured_diagnostic_case,
    monkeypatch,
):
    model, X, _ = structured_diagnostic_case
    expected = model.term_importance(X)
    spec = _structured_spec(model)

    def fail_transform(*args, **kwargs):
        del args, kwargs
        raise AssertionError("structured transform must not be called")

    monkeypatch.setattr(spec, "transform", fail_transform)
    actual = model.term_importance(X)

    pd.testing.assert_frame_equal(actual, expected, rtol=2e-11, atol=2e-11)
```

- [ ] **Step 2: Run the importance regression and verify expanded transforms are used**

Run:

```bash
rtk pytest tests/test_structured_diagnostics.py::test_term_importance_uses_compact_structured_score
```

Expected: three FAIL cases with
`AssertionError: structured transform must not be called`.

- [ ] **Step 3: Add the term-local scorer and migrate importance**

In `src/superglm/model/base.py`, split exact term scoring into a local-beta
entry point:

```python
def _score_prediction_term_local_exact(
    term: dict[str, Any],
    X: EagerFrame,
    beta: NDArray,
) -> NDArray[np.floating]:
    """Score one canonical term from its term-local coefficient vector."""
    beta = np.asarray(beta, dtype=np.float64).ravel()
    expected_width = len(term["beta_idx"])
    if beta.shape != (expected_width,):
        raise ValueError(
            f"term {term['name']!r} requires {expected_width} coefficients, "
            f"got {len(beta)}"
        )
    if term["kind"] == "feature":
        return _score_feature(term["spec"], X.column_array(term["name"]), beta)

    left_name, right_name = term["parent_names"]
    return _score_interaction(
        term["spec"],
        X.column_array(left_name),
        X.column_array(right_name),
        beta,
    )


def _score_prediction_term_exact(
    term: dict[str, Any],
    X: EagerFrame,
    beta_all: NDArray,
) -> NDArray[np.floating]:
    """Score one canonical term exactly on the requested rows."""
    return _score_prediction_term_local_exact(
        term,
        X,
        beta_all[term["beta_idx"]],
    )
```

In `term_importance()` in
`src/superglm/diagnostics/term_diagnostics.py`, resolve prediction-plan terms
once:

```python
    from superglm.model import base

    plan = base._prediction_plan(model)
    terms_by_name = {
        term["name"]: term
        for term in (*plan["features"], *plan["interactions"])
    }
```

Replace the `spec.transform()` / `ispec.transform()` branch with:

```python
        term = terms_by_name.get(g.feature_name)
        if term is None:
            raise RuntimeError(
                f"prediction plan does not define fitted term {g.feature_name!r}"
            )
        term_indices = np.asarray(term["beta_idx"], dtype=np.intp)
        group_positions = (term_indices >= g.start) & (term_indices < g.end)
        if np.count_nonzero(group_positions) != g.size:
            raise RuntimeError(
                f"prediction plan coefficient layout disagrees with group {g.name!r}"
            )
        term_beta = np.zeros(len(term_indices), dtype=np.float64)
        term_beta[group_positions] = beta[term_indices[group_positions]]
        eta_g = base._score_prediction_term_local_exact(term, frame, term_beta)
```

Preserve the zero-norm fast path and every existing output field.

- [ ] **Step 4: Run importance, prediction, and select-group regressions**

Run:

```bash
rtk pytest tests/test_structured_diagnostics.py::test_term_importance_uses_compact_structured_score
rtk pytest tests/test_diagnostics.py tests/test_select.py -k "term_importance or predict"
```

Expected: PASS. The structured sentinel is never called, and existing
per-`GroupSlice` output remains unchanged.

- [ ] **Step 5: Commit canonical scoring and compact importance**

```bash
rtk git add src/superglm/model/base.py src/superglm/diagnostics/term_diagnostics.py tests/test_structured_diagnostics.py
rtk git commit -m "Score term importance without expanded matrices"
```

### Task 3: Make holdout drop diagnostics subtract compact contributions

**Files:**
- Modify: `tests/test_structured_diagnostics.py`
- Modify: `src/superglm/diagnostics/term_diagnostics.py:170-234`

- [ ] **Step 1: Add the failing compact holdout regression**

Append to `tests/test_structured_diagnostics.py`:

```python
def test_holdout_drop_uses_compact_structured_score(
    structured_diagnostic_case,
    monkeypatch,
):
    model, X, y = structured_diagnostic_case
    weights = np.linspace(0.7, 1.3, len(y))
    expected = model.term_drop_diagnostics(
        X,
        y,
        sample_weight=weights,
        mode="holdout",
        X_val=X,
        y_val=y,
    )
    spec = _structured_spec(model)
    original_score = spec.score
    scored_betas = []

    def fail_transform(*args, **kwargs):
        del args, kwargs
        raise AssertionError("structured transform must not be called")

    def record_score(*args, **kwargs):
        beta = np.asarray(args[-1], dtype=np.float64)
        scored_betas.append(beta.copy())
        return original_score(*args, **kwargs)

    monkeypatch.setattr(spec, "transform", fail_transform)
    monkeypatch.setattr(spec, "score", record_score)
    actual = model.term_drop_diagnostics(
        X,
        y,
        sample_weight=weights,
        mode="holdout",
        X_val=X,
        y_val=y,
    )

    pd.testing.assert_frame_equal(actual, expected, rtol=2e-11, atol=2e-11)
    from superglm.model import base

    plan = base._prediction_plan(model)
    term = next(
        term
        for term in (*plan["features"], *plan["interactions"])
        if term["spec"] is spec
    )
    assert len(scored_betas) == 1
    np.testing.assert_array_equal(
        scored_betas[0],
        model.result.beta[term["beta_idx"]],
    )
```

- [ ] **Step 2: Run the holdout regression and verify expanded transforms are used**

Run:

```bash
rtk pytest tests/test_structured_diagnostics.py::test_holdout_drop_uses_compact_structured_score
```

Expected: three FAIL cases with
`AssertionError: structured transform must not be called`.

- [ ] **Step 3: Replace full-design rebuilding with full-minus-term scoring**

Rewrite `_drop_term_holdout()` in
`src/superglm/diagnostics/term_diagnostics.py` around the existing validation
and deviance calculation:

```python
    from superglm.distributions import clip_mu
    from superglm.links import stabilize_eta
    from superglm.model import base

    beta = model.result.beta
    dist = model._distribution
    w = sample_weight if sample_weight is not None else np.ones(len(y_val))
    y_arr = np.asarray(y_val, dtype=np.float64)

    plan = base._prediction_plan(model)
    terms = [*plan["features"], *plan["interactions"]]
    eta_raw = np.full(len(X_val), model.result.intercept, dtype=np.float64)
    contributions: dict[str, NDArray[np.floating]] = {}
    for term in terms:
        contribution = base._score_prediction_term_exact(term, X_val, beta)
        contributions[term["name"]] = contribution
        eta_raw += contribution

    eta_full = stabilize_eta(eta_raw, model._link)
    mu_full = clip_mu(model._link.inverse(eta_full), dist)
    dev_full = float(np.sum(w * dist.deviance_unit(y_arr, mu_full)))

    rows = []
    for term in terms:
        eta_drop = stabilize_eta(
            eta_raw - contributions[term["name"]],
            model._link,
        )
        mu_drop = clip_mu(model._link.inverse(eta_drop), dist)
        dev_drop = float(np.sum(w * dist.deviance_unit(y_arr, mu_drop)))
        rows.append(
            {
                "feature": term["name"],
                "delta_deviance": dev_drop - dev_full,
            }
        )

    return pd.DataFrame(rows)
```

Delete `beta_zeroed`, the repeated `transform()` loops, `blocks`,
`np.hstack()`, and the `seen_features` scan. Keep exact scoring and
stabilisation after subtraction.

- [ ] **Step 4: Run compactness and ordinary numerical-equivalence tests**

Run:

```bash
rtk pytest tests/test_structured_diagnostics.py
rtk pytest tests/test_diagnostics.py tests/test_dataframe_boundary_diagnostics.py -k "holdout or term_drop"
```

Expected: PASS for RE, FS, SZ, pandas, Polars, and existing strong-term
deviance assertions.

- [ ] **Step 5: Commit compact holdout scoring**

```bash
rtk git add src/superglm/diagnostics/term_diagnostics.py tests/test_structured_diagnostics.py
rtk git commit -m "Keep holdout diagnostics on compact scoring paths"
```

### Task 4: Define the unpenalised-refit variance-component contract

**Files:**
- Modify: `tests/test_refit.py`
- Modify: `src/superglm/inference/_term_model_ops.py:210-235,311-350`

- [ ] **Step 1: Write the failing public preflight regression**

Extend imports in `tests/test_refit.py`:

```python
from superglm import LambdaPolicy, RandomEffect, SuperGLM
```

Add:

```python
def test_refit_unpenalised_rejects_variance_component_terms_at_entry(monkeypatch):
    rng = np.random.default_rng(20260727)
    codes = np.repeat(np.arange(6), 12)
    X = pd.DataFrame(
        {"group": np.array([f"group-{code}" for code in codes], dtype=object)}
    )
    y = rng.normal(size=len(codes))
    model = SuperGLM(
        family="gaussian",
        features={
            "group": RandomEffect(lambda_policy=LambdaPolicy.fixed(1.0)),
        },
        selection_penalty=0.0,
        direct_solve="structured",
    ).fit_reml(X, y, runtime_validation="skip")

    def fail_clone(*args, **kwargs):
        del args, kwargs
        raise AssertionError("model clone requested")

    monkeypatch.setattr(model, "_clone_without_features", fail_clone)
    with pytest.raises(
        NotImplementedError,
        match=r"refit_unpenalised\(\).*variance-component.*group",
    ):
        model.refit_unpenalised(X, y)
```

- [ ] **Step 2: Run the regression and verify it reaches cloning**

Run:

```bash
rtk pytest tests/test_refit.py::test_refit_unpenalised_rejects_variance_component_terms_at_entry
```

Expected: FAIL with `AssertionError: model clone requested`.

- [ ] **Step 3: Share REML-term discovery and reject at method entry**

Add above `drop1()` in `src/superglm/inference/_term_model_ops.py`:

```python
def _requires_reml_term_names(model) -> list[str]:
    """Return configured terms that require variance-component fitting."""
    configured_terms = (
        [(name, model._specs[name]) for name in model._feature_order]
        + [
            (name, model._interaction_specs[name])
            for name in model._interaction_order
        ]
    )
    return [
        name
        for name, spec in configured_terms
        if getattr(spec, "requires_reml", False)
    ]
```

Replace `drop1()`'s local comprehension with:

```python
    reml_only_terms = _requires_reml_term_names(model)
```

In `refit_unpenalised()`, keep the fitted check first, then add:

```python
    reml_only_terms = _requires_reml_term_names(model)
    if reml_only_terms:
        raise NotImplementedError(
            "refit_unpenalised() does not support variance-component terms "
            f"{reml_only_terms!r}; an ordinary unpenalised fit cannot preserve "
            "their REML variance-component contract."
        )
```

Move `del sample_weight` below this preflight. Do not invent a fixed-lambda
ordinary-fit path.

- [ ] **Step 4: Run refit and drop-one contract tests**

Run:

```bash
rtk pytest tests/test_refit.py
rtk pytest tests/test_random_effect_inference.py -k "drop1 or refit"
```

Expected: PASS. Ordinary models still refit, while REML-only models fail
before cloning with the method-specific message.

- [ ] **Step 5: Commit the explicit refit contract**

```bash
rtk git add src/superglm/inference/_term_model_ops.py tests/test_refit.py
rtk git commit -m "Reject unpenalised variance-component refits early"
```

### Task 5: Run final quality, numerical, packaging, and review gates

**Files:**
- Verify all touched source and tests.
- Update no files unless a gate finds a concrete defect.

- [ ] **Step 1: Format and lint only the touched files**

Run:

```bash
rtk ruff format src/superglm/inference/_term_covariance.py src/superglm/model/base.py src/superglm/diagnostics/term_diagnostics.py src/superglm/inference/_term_model_ops.py tests/test_random_effect_inference.py tests/test_structured_diagnostics.py tests/test_refit.py
rtk ruff check src/superglm/inference/_term_covariance.py src/superglm/model/base.py src/superglm/diagnostics/term_diagnostics.py src/superglm/inference/_term_model_ops.py tests/test_random_effect_inference.py tests/test_structured_diagnostics.py tests/test_refit.py
```

Expected: no formatting changes after the first pass and no lint errors.

- [ ] **Step 2: Run static typing on touched source**

Run:

```bash
rtk mypy src/superglm/inference/_term_covariance.py src/superglm/model/base.py src/superglm/diagnostics/term_diagnostics.py src/superglm/inference/_term_model_ops.py
```

Expected: success with no new type errors.

- [ ] **Step 3: Run the focused structured and generic surface suites**

Run:

```bash
rtk pytest tests/test_random_effect_inference.py tests/test_structured_diagnostics.py tests/test_factor_smooth_inference.py tests/test_factor_smooth_sz_inference.py tests/test_diagnostics.py tests/test_dataframe_boundary_diagnostics.py tests/test_refit.py
```

Expected: all selected tests pass.

- [ ] **Step 4: Run the full repository and package gates**

Run:

```bash
rtk pytest tests/
rtk proxy uv build
rtk proxy uv run python run_test.py
```

Expected: full suite passes, wheel and source distribution build, and the
repository smoke test succeeds.

- [ ] **Step 5: Inspect the exact final patch and commit any gate-only edits**

Run:

```bash
rtk git diff --check
rtk git status --short
rtk git diff master...HEAD --stat
```

If formatting or a verified gate defect changed files, commit only those exact
files:

```bash
rtk git add src/superglm/inference/_term_covariance.py src/superglm/model/base.py src/superglm/diagnostics/term_diagnostics.py src/superglm/inference/_term_model_ops.py tests/test_random_effect_inference.py tests/test_structured_diagnostics.py tests/test_refit.py
rtk git commit -m "Polish compact structured diagnostics"
```

Expected: clean worktree after the final commit. Then push the exact head,
request fresh independent and Codex review, resolve only verified actionable
threads, and rerun required CI on that same SHA.
