# Credibility Release-Safety Review Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make PR #165 release-safe by enforcing structured-backend topology, preventing silent deployment omissions, completing random-effect public API integration, retaining backend-neutral report support, guarding reporting provenance, and removing the normal large-\(n\) discrete factor-smooth dense transient.

**Architecture:** Keep the validated Schur/SZ numerical core unchanged. Put unsupported-layout decisions in backend selection, deployment decisions in an export preflight, public-name decisions in design construction, and compact reporting support in a backend-neutral model state built during terminal REML finalization. Refactor only factor-smooth marginal construction: derive the existing natural parameterization from QR `R`, stream the symmetric/default discrete case, and retain a reduced-memory compatibility path where separate null-policy semantics could change.

**Tech Stack:** Python 3.10+, NumPy, SciPy sparse/LAPACK, pandas, Tabmat-backed grouped matrices, pytest, Ruff, mypy, cProfile, GitHub Actions.

**Design:** `docs/superpowers/specs/2026-07-26-credibility-release-safety-review-design.md`

**Constraints:** Do not touch LSS. Do not merge or publish PR #165. Prefix every shell command with `rtk`.

---

## File Structure

### New files

- `src/superglm/model/reporting_state.py`
  - Owns immutable RE/FS support records, the backend-neutral reporting-state
    container, and final-fit sufficient-statistic construction.
- `benchmarks/profile_factor_smooth_construction.py`
  - Isolated wall-time, peak-RSS, and cProfile harness for exact/discrete FS/SZ
    marginal construction.

### Modified implementation files

- `src/superglm/solvers/_structured/selection.py`
  - Enforces the one-dominant-FS topology before cost selection.
- `src/superglm/solvers/_structured/state.py`
  - Imports compatibility support types from model reporting state.
- `src/superglm/solvers/structured.py`
  - Retains existing compatibility re-exports.
- `src/superglm/export/rating_tables.py`
  - Rejects unsupported RE/FS/SZ exports before payload work.
- `src/superglm/dm_builder.py`
  - Enforces cross-namespace term names and fitted group-name uniqueness.
- `src/superglm/model/base.py`
  - Initializes/projects reporting state and validates explicit/auto-resolved
    term namespaces.
- `src/superglm/model/fit_state.py`
  - Carries and invalidates reporting state transactionally.
- `src/superglm/model/fit_ops.py`
  - Clears reporting state and captures training-geometry fingerprints.
- `src/superglm/model/fit_data_guard.py`
  - Adds response-aware training-row verification.
- `src/superglm/model/reml_finalize.py`
  - Builds reporting state for every final backend and shares dominant moments
    with structured state.
- `src/superglm/features/random_effect.py`
  - Exposes conventional relativity reconstruction keys.
- `src/superglm/features/factor_smooth.py`
  - Uses `R`-only natural algebra and bounded streamed QR where safe.
- `src/superglm/inference/_term_covariance.py`
  - Returns all-level RE uncertainty.
- `src/superglm/inference/_term_ops.py`
  - Produces categorical-shaped RE term inference.
- `src/superglm/inference/_term_model_ops.py`
  - Rejects `drop1` for REML-only terms.
- `src/superglm/inference/random_effects.py`
  - Reads backend-neutral support and enforces training-only row provenance.
- `src/superglm/inference/factor_smooths.py`
  - Reads backend-neutral support.

### Modified tests and documentation

- `tests/test_structured_allocations.py`
- `tests/test_factor_smooth_structured_parity.py`
- `tests/test_rating_table_export.py`
- `tests/test_factor_smooth_feature.py`
- `tests/test_factor_smooth_discrete.py`
- `tests/test_interactions.py`
- `tests/test_plot_api.py`
- `tests/test_drop1.py`
- `tests/test_random_effect_inference.py`
- `tests/test_factor_smooth_inference.py`
- `tests/test_factor_smooth_sz_inference.py`
- `tests/test_fit_data_guard.py`
- `docs/getting-started/quickstart.md`
- `docs/guide/credibility.md`
- PR #165 body and review comments

---

### Task 1: Enforce Structured Topology Before Assembly

**Files:**
- Modify: `src/superglm/solvers/_structured/selection.py`
- Modify: `tests/test_structured_allocations.py`
- Modify: `tests/test_factor_smooth_structured_parity.py`

- [ ] **Step 1: Write low-level failing selection tests**

Add a compact factory and the topology assertions to
`tests/test_structured_allocations.py`:

```python
def _factor_smooth_matrix(n: int, *, levels: int, k: int = 5) -> FactorSmoothGroupMatrix:
    x = np.linspace(-1.0, 1.0, n)
    basis = np.column_stack([x**power for power in range(k)])
    return FactorSmoothGroupMatrix(
        sp.csr_matrix(basis),
        np.arange(n, dtype=np.intp) % levels,
        levels,
        natural_map=np.eye(k),
        levels=tuple(f"level-{index}" for index in range(levels)),
        repeated_penalty_components=(("wiggle", np.eye(k)),),
    )


def test_structured_selection_rejects_multiple_factor_smooths_before_assembly():
    matrices: list[GroupMatrix] = [
        _factor_smooth_matrix(80, levels=8),
        _factor_smooth_matrix(80, levels=6),
    ]
    groups = _groups(matrices)

    auto = select_structured_group(matrices, groups, mode="auto")
    assert auto.group_index is None
    assert "at most one FactorSmooth" in auto.fallback_reason
    assert groups[0].name in auto.fallback_reason
    assert groups[1].name in auto.fallback_reason

    with pytest.raises(ValueError, match="at most one FactorSmooth"):
        select_structured_group(matrices, groups, mode="structured")


def test_factor_smooth_is_dominant_candidate_even_when_random_effect_is_wider():
    matrices: list[GroupMatrix] = [
        RandomEffectGroupMatrix(np.arange(300) % 50, n_levels=50),
        _factor_smooth_matrix(300, levels=8),
    ]
    groups = _groups(matrices)

    selected = select_structured_group(matrices, groups, mode="structured")

    assert selected.group_index == 1
    assert selected.group_name == groups[1].name
```

- [ ] **Step 2: Run the low-level tests and verify the old dispatcher fails**

Run:

```bash
rtk pytest tests/test_structured_allocations.py -k "multiple_factor_smooths or dominant_candidate" -q
```

Expected: both new tests fail because the current selector chooses the widest
structured block without inspecting factor-smooth topology.

- [ ] **Step 3: Implement topology-aware candidate selection**

Replace the current single `candidates` selection in
`select_structured_group()` with:

```python
    factor_smooth_indices = [
        index
        for index, matrix in enumerate(group_matrices)
        if isinstance(matrix, FactorSmoothGroupMatrix)
    ]
    random_effect_indices = [
        index
        for index, matrix in enumerate(group_matrices)
        if isinstance(matrix, RandomEffectGroupMatrix)
    ]
    if len(factor_smooth_indices) > 1:
        names = [groups[index].name for index in factor_smooth_indices]
        return _selection_failure(
            "the structured backend supports at most one FactorSmooth term; "
            f"found {names!r}",
            mode,
        )
    if factor_smooth_indices:
        dominant_index = factor_smooth_indices[0]
    elif random_effect_indices:
        dominant_index = max(
            random_effect_indices,
            key=lambda index: group_matrices[index].shape[1],
        )
    else:
        return _selection_failure(
            "the model has no RandomEffect or FactorSmooth term",
            mode,
        )
```

Keep the existing geometry consistency, singularity, and cost checks after
this selection.

- [ ] **Step 4: Add end-to-end auto/forced regression tests**

In `tests/test_factor_smooth_structured_parity.py`, add:

```python
def _multi_structured_data():
    rng = np.random.default_rng(20260726)
    n = 600
    g1 = np.arange(n) % 8
    g2 = np.arange(n) % 6
    re = np.arange(n) % 50
    x1 = rng.uniform(-1.0, 1.0, n)
    x2 = rng.uniform(-1.0, 1.0, n)
    y = 0.4 + 0.3 * x1 - 0.2 * x2 + rng.normal(scale=0.15, size=n)
    X = pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "g1": np.array([f"a-{v}" for v in g1], dtype=object),
            "g2": np.array([f"b-{v}" for v in g2], dtype=object),
            "re": np.array([f"r-{v}" for v in re], dtype=object),
        }
    )
    return X, y


def test_two_factor_smooths_auto_fall_back_and_forced_structured_rejects():
    X, y = _multi_structured_data()

    def model(mode: str) -> SuperGLM:
        return SuperGLM(
            family="gaussian",
            features={},
            interactions=[
                FactorSmooth(
                    "x1",
                    group="g1",
                    k=5,
                    lambda_policy=LambdaPolicy.fixed(1.0),
                ),
                FactorSmooth(
                    "x2",
                    group="g2",
                    k=5,
                    lambda_policy=LambdaPolicy.fixed(1.0),
                ),
            ],
            selection_penalty=0.0,
            direct_solve=mode,
        )

    gram = model("gram").fit_reml(X, y, runtime_validation="skip")
    auto = model("auto").fit_reml(X, y, runtime_validation="skip")

    assert auto.result.direct_backend == "gram"
    assert "at most one FactorSmooth" in auto.result.direct_fallback_reason
    np.testing.assert_allclose(auto.predict(X), gram.predict(X), atol=2e-8)

    with pytest.raises(ValueError, match="at most one FactorSmooth"):
        model("structured").fit_reml(X, y, runtime_validation="skip")


@pytest.mark.parametrize("basis", ["fs", "sz"])
def test_single_factor_smooth_dominates_wider_random_effect(basis: str):
    X, y = _multi_structured_data()

    def model(mode: str) -> SuperGLM:
        features = {
            "re": RandomEffect(lambda_policy=LambdaPolicy.fixed(1.1)),
        }
        if basis == "sz":
            features["x1"] = Spline(
                n_knots=5,
                lambda_policy=LambdaPolicy.fixed(1.2),
            )
        return SuperGLM(
            family="gaussian",
            features=features,
            interactions=[
                FactorSmooth(
                    "x1",
                    group="g1",
                    basis=basis,
                    k=5,
                    lambda_policy=LambdaPolicy.fixed(0.9),
                )
            ],
            selection_penalty=0.0,
            direct_solve=mode,
        )

    gram = model("gram").fit_reml(X, y, runtime_validation="skip")
    auto = model("auto").fit_reml(X, y, runtime_validation="skip")
    structured = model("structured").fit_reml(X, y, runtime_validation="skip")

    assert auto.result.direct_backend == "structured"
    assert structured.result.direct_backend == "structured"
    np.testing.assert_allclose(auto.predict(X), gram.predict(X), atol=3e-8)
    np.testing.assert_allclose(structured.predict(X), gram.predict(X), atol=3e-8)
    assert auto.result.deviance == pytest.approx(gram.result.deviance, abs=2e-8)
    assert structured.result.deviance == pytest.approx(gram.result.deviance, abs=2e-8)
```

- [ ] **Step 5: Run topology and parity tests**

Run:

```bash
rtk pytest tests/test_structured_allocations.py tests/test_factor_smooth_structured_parity.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
rtk git add src/superglm/solvers/_structured/selection.py tests/test_structured_allocations.py tests/test_factor_smooth_structured_parity.py
rtk git commit -m "Guard structured factor-smooth topology"
```

---

### Task 2: Make Public Term and Group Names Globally Unique

**Files:**
- Modify: `src/superglm/dm_builder.py`
- Modify: `src/superglm/model/base.py`
- Modify: `tests/test_factor_smooth_feature.py`
- Modify: `tests/test_plot_api.py`
- Modify: `tests/test_interactions.py`

- [ ] **Step 1: Write failing explicit/generated collision tests**

Add to `tests/test_factor_smooth_feature.py`:

```python
def test_explicit_factor_smooth_name_cannot_collide_with_main_feature():
    with pytest.raises(ValueError, match="risk.*main feature.*interaction"):
        SuperGLM(
            family="gaussian",
            features={"risk": Numeric()},
            interactions=[FactorSmooth("age", group="broker", name="risk")],
        )
```

Add to `tests/test_plot_api.py`:

```python
def test_generated_interaction_name_cannot_collide_with_main_feature():
    X = pd.DataFrame(
        {
            "a": np.linspace(0.0, 1.0, 40),
            "b": np.linspace(1.0, 2.0, 40),
            "a:b": np.linspace(-1.0, 1.0, 40),
        }
    )
    y = np.linspace(0.2, 1.1, 40)
    model = SuperGLM(
        family="gaussian",
        features={"a": Numeric(), "b": Numeric(), "a:b": Numeric()},
        interactions=[("a", "b")],
        selection_penalty=0.0,
    )

    with pytest.raises(ValueError, match="a:b.*main feature.*interaction"):
        model.fit(X, y)
```

- [ ] **Step 2: Run collision tests and verify failure**

```bash
rtk pytest tests/test_factor_smooth_feature.py tests/test_plot_api.py -k "name_cannot_collide" -q
```

Expected: the explicit constructor currently succeeds and the generated
interaction fails too late or aliases the main name.

- [ ] **Step 3: Add reusable namespace/group validators**

In `src/superglm/dm_builder.py`, add:

```python
def validate_term_name_namespace(
    specs: dict[str, FeatureSpec],
    interaction_specs: dict[str, Any],
) -> None:
    collisions = sorted(set(specs).intersection(interaction_specs))
    if collisions:
        names = ", ".join(repr(name) for name in collisions)
        raise ValueError(
            f"Term name(s) {names} are registered as both a main feature and an interaction."
        )


def validate_fitted_group_names(groups: list[GroupSlice]) -> None:
    counts: dict[str, int] = {}
    for group in groups:
        counts[group.name] = counts.get(group.name, 0) + 1
    duplicates = sorted(name for name, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"Generated fitted group names must be unique; found {duplicates!r}.")
```

Call `validate_term_name_namespace()`:

- after explicit main features and explicit interactions are owned in
  `init_model()`;
- inside `add_interaction()` immediately after the factory resolves `iname`;
- after auto-detection and after pending interactions resolve.

Call `validate_fitted_group_names(groups)` immediately before constructing the
final `DesignMatrix`.

- [ ] **Step 4: Add a generated subgroup collision test**

Add to `tests/test_interactions.py`. A decomposed tensor emits the fitted
subgroups `a:b:bilinear` and `a:b:wiggly`; the first must not alias a main
feature with that public name:

```python
def test_generated_fitted_group_names_cannot_alias():
    X = pd.DataFrame(
        {
            "a": np.linspace(-1.0, 1.0, 80),
            "b": np.linspace(0.0, 2.0, 80),
            "a:b:bilinear": np.linspace(-2.0, 2.0, 80),
        }
    )
    y = 0.4 + np.sin(X["a"].to_numpy()) + 0.2 * X["b"].to_numpy()
    model = SuperGLM(
        family="gaussian",
        features={
            "a": Spline(n_knots=5),
            "b": Spline(n_knots=5),
            "a:b:bilinear": Numeric(),
        },
        selection_penalty=0.0,
    )
    model._add_interaction("a", "b", decompose=True)

    with pytest.raises(ValueError, match="group names must be unique.*a:b:bilinear"):
        model.fit(X, y)
```

- [ ] **Step 5: Run namespace tests and nearby design tests**

```bash
rtk pytest tests/test_factor_smooth_feature.py tests/test_plot_api.py tests/test_interactions.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
rtk git add src/superglm/dm_builder.py src/superglm/model/base.py tests/test_factor_smooth_feature.py tests/test_plot_api.py tests/test_interactions.py
rtk git commit -m "Enforce globally unique term names"
```

---

### Task 3: Reject Unsupported Rating-Table Exports Before Work Starts

**Files:**
- Modify: `src/superglm/export/rating_tables.py`
- Modify: `tests/test_rating_table_export.py`

- [ ] **Step 1: Write failing preflight tests**

Add:

```python
@pytest.mark.parametrize("basis", [None, "fs", "sz"])
def test_rating_table_export_rejects_structured_terms_before_impact(
    basis: str | None,
    monkeypatch,
):
    rng = np.random.default_rng(20260726)
    n = 160
    x = rng.uniform(-1.0, 1.0, n)
    codes = np.arange(n) % 5
    X = pd.DataFrame(
        {
            "x": x,
            "group": np.array([f"g-{code}" for code in codes], dtype=object),
        }
    )
    y = 0.5 + 0.3 * x + rng.normal(scale=0.15, size=n)
    if basis is None:
        features = {
            "x": Spline(k=6, lambda_policy=LambdaPolicy.fixed(1.0)),
            "group": RandomEffect(lambda_policy=LambdaPolicy.fixed(1.0)),
        }
        interactions = None
        expected = "group"
    else:
        features = {"x": Spline(k=6, lambda_policy=LambdaPolicy.fixed(1.0))}
        interactions = [
            FactorSmooth(
                "x",
                group="group",
                basis=basis,
                k=5,
                lambda_policy=LambdaPolicy.fixed(1.0),
            )
        ]
        expected = f"x:group:{basis}"
    model = SuperGLM(
        family="gaussian",
        features=features,
        interactions=interactions,
        selection_penalty=0.0,
        direct_solve="gram",
    ).fit_reml(X, y, runtime_validation="skip")

    monkeypatch.setattr(
        model,
        "discretization_impact",
        lambda *_args, **_kwargs: pytest.fail("impact analysis ran before export preflight"),
    )
    with pytest.raises(NotImplementedError, match=expected):
        build_rating_table_payload(model, X, y)
```

Add one model containing both RE and FS and assert both names occur in the
single exception string:

```python
def test_rating_table_export_reports_every_unsupported_structured_term():
    rng = np.random.default_rng(32)
    n = 180
    X = pd.DataFrame(
        {
            "x": rng.uniform(-1.0, 1.0, n),
            "group": np.array([f"g-{index % 5}" for index in range(n)], dtype=object),
            "broker": np.array([f"b-{index % 9}" for index in range(n)], dtype=object),
        }
    )
    y = 0.5 + 0.3 * X["x"].to_numpy() + rng.normal(scale=0.15, size=n)
    model = SuperGLM(
        family="gaussian",
        features={
            "x": Spline(k=6, lambda_policy=LambdaPolicy.fixed(1.0)),
            "broker": RandomEffect(lambda_policy=LambdaPolicy.fixed(1.0)),
        },
        interactions=[
            FactorSmooth(
                "x",
                group="group",
                k=5,
                lambda_policy=LambdaPolicy.fixed(1.0),
            )
        ],
        selection_penalty=0.0,
        direct_solve="gram",
    ).fit_reml(X, y, runtime_validation="skip")

    with pytest.raises(NotImplementedError) as exc_info:
        build_rating_table_payload(model, X, y)
    message = str(exc_info.value)
    assert "broker" in message
    assert "x:group:fs" in message
```

- [ ] **Step 2: Run tests and verify the silent omission**

```bash
rtk pytest tests/test_rating_table_export.py -k "rejects_structured_terms" -q
```

Expected: RE export currently returns a payload; FS/SZ errors occur only after
later reconstruction work.

- [ ] **Step 3: Implement whole-model export preflight**

Near the rating-table builder, add:

```python
def _unsupported_structured_export_terms(model) -> list[str]:
    from superglm.features.factor_smooth import FactorSmooth
    from superglm.features.random_effect import RandomEffect

    unsupported = [
        name for name in model._feature_order if isinstance(model._specs[name], RandomEffect)
    ]
    unsupported.extend(
        name
        for name in model._interaction_order
        if isinstance(model._interaction_specs[name], FactorSmooth)
    )
    return unsupported


def _preflight_rating_table_terms(model) -> None:
    unsupported = _unsupported_structured_export_terms(model)
    if unsupported:
        raise NotImplementedError(
            "Rating-table export does not yet support conditional or population-only "
            f"RandomEffect/FactorSmooth terms {unsupported!r}; no payload was produced."
        )
```

Call `_preflight_rating_table_terms(model)` immediately after the fitted-model
check in `build_rating_table_payload()`, before frame conversion or impact
analysis.

- [ ] **Step 4: Run all export tests**

```bash
rtk pytest tests/test_rating_table_export.py -q
```

Expected: PASS with no behavior changes for supported models.

- [ ] **Step 5: Commit**

```bash
rtk git add src/superglm/export/rating_tables.py tests/test_rating_table_export.py
rtk git commit -m "Reject unsupported structured rating exports"
```

---

### Task 4: Complete Safe Random-Effect Public API Integration

**Files:**
- Modify: `src/superglm/features/random_effect.py`
- Modify: `src/superglm/inference/_term_covariance.py`
- Modify: `src/superglm/inference/_term_ops.py`
- Modify: `src/superglm/inference/_term_model_ops.py`
- Modify: `tests/test_random_effect_inference.py`
- Modify: `tests/test_drop1.py`

- [ ] **Step 1: Write one public-surface integration regression**

Add to `tests/test_random_effect_inference.py`:

```python
import pickle


def test_random_effect_generic_term_surfaces_are_consistent():
    _, model, X, _, _ = _fit_pair(fit_dense=False, max_reml_iter=2)
    group = next(group for group in model._groups if group.name == "broker")

    term = model.term_inference("broker")
    assert term.kind == "categorical"
    assert term.levels == model._specs["broker"]._levels
    np.testing.assert_allclose(term.log_relativity, model.result.beta[group.sl])
    assert term.centering_mode == "population_zero"

    rel = model.relativities(with_se=True)["broker"]
    assert list(rel["level"]) == term.levels
    np.testing.assert_allclose(rel["log_relativity"], term.log_relativity)

    selected = model.plot_data(terms="broker", ci="pointwise")
    default = model.plot_data(ci=None)
    assert selected["terms"][0]["name"] == "broker"
    assert any(item["name"] == "broker" for item in default["terms"])

    axes = model.plot(terms="broker", ci=None)
    assert axes is not None
    default_axes = model.plot(ci=None)
    assert default_axes is not None

    assert model.summary()["fit"]["n_obs"] == len(X)
    assert np.all(np.isfinite(model.predict(X.iloc[:5])))
    restored = pickle.loads(pickle.dumps(model))
    restored_term = restored.term_inference("broker")
    np.testing.assert_allclose(restored_term.log_relativity, term.log_relativity)
```

Add to `tests/test_drop1.py`:

```python
def test_drop1_rejects_variance_component_models_before_refitting():
    X = pd.DataFrame({"group": np.repeat(["a", "b", "c"], 20)})
    y = np.tile([0.2, 0.7, 1.1], 20)
    model = SuperGLM(
        family="gaussian",
        features={"group": RandomEffect(lambda_policy=LambdaPolicy.fixed(1.0))},
        selection_penalty=0.0,
    ).fit_reml(X, y, runtime_validation="skip")

    with pytest.raises(NotImplementedError, match="drop1.*variance-component.*REML"):
        model.drop1(X, y)
```

- [ ] **Step 2: Run and verify current accidental failures**

```bash
rtk pytest tests/test_random_effect_inference.py tests/test_plot_api.py tests/test_drop1.py -k "generic_term_surfaces or variance_component" -q
```

Expected: term/plot/relativity calls fail with `TypeError`/`KeyError`, and
`drop1` reaches a reduced `.fit()` before failing.

- [ ] **Step 3: Extend `RandomEffect.reconstruct()`**

Return all established keys:

```python
    effects = {
        level: float(value)
        for level, value in zip(self._levels, np.asarray(beta).ravel(), strict=True)
    }
    return {
        "levels": self._levels.copy(),
        "effects": effects,
        "log_relativities": effects.copy(),
        "relativities": {level: float(np.exp(value)) for level, value in effects.items()},
    }
```

- [ ] **Step 4: Add all-level covariance and term inference**

In `feature_se_from_cov()`, import `RandomEffect`, return `len(spec._levels)`
zeros for an inactive term, and return `sqrt(diag(Cov_g))` for active RE
coefficients.

In `term_inference()`, add this branch before ordinary categorical dispatch:

```python
    elif isinstance(spec, RandomEffect):
        raw = spec.reconstruct(beta_combined)
        levels = raw["levels"]
        log_rels = np.asarray([raw["effects"][level] for level in levels])
        rels = np.exp(log_rels)
        se = ci_lo = ci_hi = None
        if with_se and active and Cov_active is not None:
            assert active_groups_cov is not None
            se = feature_se_from_cov(
                name,
                Cov_active,
                active_groups_cov,
                result,
                groups,
                specs,
                interaction_specs,
            )
            ci_lo = _safe_exp(log_rels - z_alpha * se)
            ci_hi = _safe_exp(log_rels + z_alpha * se)
        return _recenter_term(
            TermInference(
                name=name,
                kind="categorical",
                active=active,
                levels=levels,
                log_relativity=log_rels,
                relativity=rels,
                se_log_relativity=se,
                ci_lower=_maybe_array(ci_lo),
                ci_upper=_maybe_array(ci_hi),
                absorbs_intercept=False,
                centering_mode="population_zero",
                edf=edf,
                smoothing_lambda=lam,
                alpha=alpha,
            ),
            centering,
        )
```

- [ ] **Step 5: Reject unsupported `drop1` at entry**

Immediately after the fitted-model check:

```python
    reml_only_terms = [
        name
        for name, spec in (
            [(name, model._specs[name]) for name in model._feature_order]
            + [(name, model._interaction_specs[name]) for name in model._interaction_order]
        )
        if getattr(spec, "requires_reml", False)
    ]
    if reml_only_terms:
        raise NotImplementedError(
            "drop1() does not support variance-component terms "
            f"{reml_only_terms!r}; boundary-aware REML comparison requires "
            "a dedicated model-comparison contract."
        )
```

- [ ] **Step 6: Run generic API, plotting, and drop-one suites**

```bash
rtk pytest tests/test_random_effect_inference.py tests/test_term_inference.py tests/test_plot_api.py tests/test_drop1.py tests/test_drop1_weights.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
rtk git add src/superglm/features/random_effect.py src/superglm/inference/_term_covariance.py src/superglm/inference/_term_ops.py src/superglm/inference/_term_model_ops.py tests/test_random_effect_inference.py tests/test_drop1.py
rtk git commit -m "Integrate random effects with term APIs"
```

---

### Task 5: Retain Backend-Neutral Reporting Support

**Files:**
- Create: `src/superglm/model/reporting_state.py`
- Modify: `src/superglm/solvers/_structured/state.py`
- Modify: `src/superglm/model/reml_finalize.py`
- Modify: `src/superglm/model/base.py`
- Modify: `src/superglm/model/fit_state.py`
- Modify: `src/superglm/model/fit_ops.py`
- Modify: `src/superglm/inference/random_effects.py`
- Modify: `src/superglm/inference/factor_smooths.py`
- Modify: `tests/test_random_effect_inference.py`
- Modify: `tests/test_factor_smooth_inference.py`
- Modify: `tests/test_factor_smooth_sz_inference.py`

- [ ] **Step 1: Write forced/automatic Gram released-state regressions**

Add to `tests/test_random_effect_inference.py` (the `pickle` import was added in
Task 4):

```python
@pytest.mark.parametrize("direct_solve", ["gram", "auto"])
def test_released_random_effect_report_is_backend_neutral(direct_solve: str):
    rng = np.random.default_rng(81)
    codes = np.repeat(np.arange(12), 10)
    X = pd.DataFrame({"group": np.array([f"g-{code}" for code in codes], dtype=object)})
    y = rng.normal(size=len(codes))
    model = SuperGLM(
        family="gaussian",
        features={"group": RandomEffect(lambda_policy=LambdaPolicy.fixed(1.2))},
        selection_penalty=0.0,
        direct_solve=direct_solve,
        retain_fit_state=False,
    ).fit_reml(X, y, runtime_validation="skip")

    assert model.result.direct_backend == "gram"
    report = model.random_effects("group")
    restored = pickle.loads(pickle.dumps(model)).random_effects("group")
    pd.testing.assert_frame_equal(restored.table, report.table)
```

Add to `tests/test_factor_smooth_inference.py`. Four FS levels at `k=6`, plus
the numeric main effect, remain below the 32-coefficient auto crossover:

```python
@pytest.mark.parametrize("direct_solve", ["gram", "auto"])
def test_released_fs_report_is_backend_neutral(direct_solve: str):
    X, y = _data()
    keep = X["segment"].isin(["segment-0", "segment-1", "segment-2", "segment-3"])
    X = X.loc[keep].reset_index(drop=True)
    y = y[keep.to_numpy()]
    model = _model(
        discrete=False,
        retain_fit_state=False,
        direct_solve=direct_solve,
    ).fit_reml(
        X,
        y,
        max_reml_iter=2,
        runtime_validation="skip",
    )
    assert model.result.direct_backend == "gram"
    report = model.factor_smooth("x:segment:fs", grid=9)
    restored = pickle.loads(pickle.dumps(model)).factor_smooth("x:segment:fs", grid=9)
    pd.testing.assert_frame_equal(restored.table, report.table)
    pd.testing.assert_frame_equal(restored.curves, report.curves)
```

Add `import pickle` and this separate SZ regression to
`tests/test_factor_smooth_sz_inference.py`. Two SZ levels keep its two global
smooths plus deviation block below the same crossover:

```python
import pickle


@pytest.mark.parametrize("direct_solve", ["gram", "auto"])
def test_released_sz_report_is_backend_neutral(direct_solve: str):
    X, y = _data()
    keep = X["segment"].isin(["alpha", "beta"])
    X = X.loc[keep].reset_index(drop=True)
    y = y[keep.to_numpy()]
    model = _fit(
        _model(
            direct_solve=direct_solve,
            retain_fit_state=False,
        ),
        X,
        y,
    )
    assert model.result.direct_backend == "gram"
    report = model.factor_smooth("x:segment:sz", grid=9)
    restored = pickle.loads(pickle.dumps(model)).factor_smooth("x:segment:sz", grid=9)
    pd.testing.assert_frame_equal(restored.table, report.table)
    pd.testing.assert_frame_equal(restored.curves, report.curves)
```

- [ ] **Step 2: Run released-state tests and verify backend-dependent failure**

```bash
rtk pytest tests/test_random_effect_inference.py tests/test_factor_smooth_inference.py tests/test_factor_smooth_sz_inference.py -k "backend_neutral" -q
```

Expected: reports raise the current “retain_fit_state or structured” error for
Gram-backed models.

- [ ] **Step 3: Create backend-neutral immutable support types**

Move the existing validation bodies into
`src/superglm/model/reporting_state.py` and add:

```python
@dataclass(frozen=True)
class ReportingSupportState:
    support_totals: dict[
        str,
        StructuredLevelSupport | FactorSmoothLevelSupport,
    ]

    def __post_init__(self) -> None:
        object.__setattr__(self, "support_totals", dict(self.support_totals))
```

Keep `StructuredLevelSupport` as the compatibility class name. In
`solvers/_structured/state.py`, import both support records from the new module
so `solvers.structured` continues to re-export the same symbols.

- [ ] **Step 4: Implement one final-fit support builder**

In `model/reporting_state.py`, implement:

```python
def build_reporting_support_state(
    *,
    dm,
    groups,
    result,
    distribution,
    link,
    sample_weight: NDArray,
    y: NDArray,
    offset: NDArray,
    retain_fit_state: bool,
    information_by_group_index: dict[int, NDArray] | None = None,
) -> ReportingSupportState | None:
    structured_indices = [
        index
        for index, matrix in enumerate(dm.group_matrices)
        if isinstance(matrix, RandomEffectGroupMatrix | FactorSmoothGroupMatrix)
    ]
    if not structured_indices:
        return None
    full_eta = stabilize_eta(dm.matvec(result.beta) + result.intercept + offset, link)
    mu = clip_mu(link.inverse(full_eta), distribution)
    variance = np.maximum(distribution.variance(mu), _VARIANCE_FLOOR)
    derivative = link.deriv_inverse(full_eta)
    working_weights = sample_weight * derivative**2 / variance
    supplied = information_by_group_index or {}
    totals = {}
    for index in structured_indices:
        matrix = dm.group_matrices[index]
        group = groups[index]
        information = supplied.get(index)
        if isinstance(matrix, FactorSmoothGroupMatrix):
            if information is None:
                information, _xtw, _rhs = matrix.factor_smooth_sufficient_stats(
                    working_weights,
                    np.zeros_like(working_weights),
                )
            totals[group.name] = FactorSmoothLevelSupport(
                count=np.bincount(matrix.codes, minlength=matrix.n_levels),
                fit_weight=np.bincount(
                    matrix.codes, weights=sample_weight, minlength=matrix.n_levels
                ),
                information=information,
            )
            continue
        if information is None:
            information = matrix.rmatvec(working_weights)
        unpooled = None
        if not retain_fit_state:
            from superglm.inference.random_effects import (
                vectorized_conditional_unpooled_effect,
            )

            base_eta = full_eta - result.beta[group.sl][matrix.codes]
            unpooled = vectorized_conditional_unpooled_effect(
                codes=matrix.codes,
                n_levels=matrix.n_levels,
                y=y,
                sample_weight=sample_weight,
                base_eta=base_eta,
                distribution=distribution,
                link=link,
                initial=result.beta[group.sl],
            )
        totals[group.name] = StructuredLevelSupport(
            count=np.bincount(matrix.codes, minlength=matrix.n_levels),
            fit_weight=np.bincount(
                matrix.codes, weights=sample_weight, minlength=matrix.n_levels
            ),
            information=information,
            unpooled_effect=unpooled,
        )
    return ReportingSupportState(totals)
```

Use the project’s exact `NDArray` annotations and imports.

- [ ] **Step 5: Refactor REML finalization to build support for every backend**

Add a small helper in `reml_finalize.py`:

```python
def _structured_information_by_group(cache: dict) -> dict[int, np.ndarray]:
    system = cache.get("structured_system")
    if isinstance(system, ScalarStructuredSystem):
        return {system.dominant_group_index: system.operator.d}
    if isinstance(system, BlockStructuredSystem | SumToZeroBlockStructuredSystem):
        return {system.dominant_group_index: system.operator.D}
    return {}
```

Immediately after `maybe_qp_passthrough_refit()`, build support from the
terminal coefficient state whenever rows will be released, regardless of
backend, and for structured fits whose retained linear-system compatibility
state also owns support. This avoids adding a final row pass to ordinary
retained Gram fits:

```python
    structured_terminal = not qp_passthrough and isinstance(
        final_factor,
        (
            ProfiledScalarSchurFactor
            | ProfiledBlockSchurFactor
            | ProfiledSumToZeroBlockFactor
        ),
    )
    reporting_state = (
        build_reporting_support_state(
            dm=model._dm,
            groups=model._groups,
            result=final_pirls,
            distribution=model._distribution,
            link=model._link,
            sample_weight=sample_weight,
            y=y,
            offset=offset_arr,
            retain_fit_state=model._retain_fit_state,
            information_by_group_index=_structured_information_by_group(final_cache),
        )
        if not model._retain_fit_state or structured_terminal
        else None
    )
    structured_linear_state = (
        _build_structured_linear_system_state(
            factor=final_factor,
            data_operator=final_xtwx,
            cache=final_cache,
            support_totals=(
                {} if reporting_state is None else reporting_state.support_totals
            ),
        )
        if use_direct and not qp_passthrough
        else None
    )
```

Remove row arrays, support construction, and the now-unused
`model`/`sample_weight`/`y`/`offset_arr`/`result` parameters from
`_build_structured_linear_system_state()`. At the existing final publication
point, publish both:

```python
    model._reporting_support_state = reporting_state
    model._linear_system_state = structured_linear_state
```

- [ ] **Step 6: Carry and invalidate the new model state**

Initialize `_reporting_support_state = None` in constructor/materialization,
clear it in `_clear_fit_inference_caches()` and coefficient-state invalidation,
and add it beside `_linear_system_state` in `_FIT_PROJECTION_NAMES`.

Update both report modules:

```python
    # inference/random_effects.py
    reporting = getattr(model, "_reporting_support_state", None)
    if isinstance(reporting, ReportingSupportState):
        support = reporting.support_totals.get(group.name)
        if isinstance(support, StructuredLevelSupport):
            return support

    # inference/factor_smooths.py
    reporting = getattr(model, "_reporting_support_state", None)
    if isinstance(reporting, ReportingSupportState):
        support = reporting.support_totals.get(group.name)
        if isinstance(support, FactorSmoothLevelSupport):
            return support
```

Keep the old `StructuredLinearSystemState.support_totals` lookup second for
compatibility, then the retained-design fallback.

- [ ] **Step 7: Run released-state, transaction, and pickle tests**

```bash
rtk pytest tests/test_random_effect_inference.py tests/test_factor_smooth_inference.py tests/test_factor_smooth_sz_inference.py tests/test_fit_state_retention.py tests/test_fit_transactions.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
rtk git add src/superglm/model/reporting_state.py src/superglm/solvers/_structured/state.py src/superglm/model/reml_finalize.py src/superglm/model/base.py src/superglm/model/fit_state.py src/superglm/model/fit_ops.py src/superglm/inference/random_effects.py src/superglm/inference/factor_smooths.py tests/test_random_effect_inference.py tests/test_factor_smooth_inference.py tests/test_factor_smooth_sz_inference.py
rtk git commit -m "Retain backend-neutral credibility support"
```

---

### Task 6: Enforce Training-Only Random-Effect Report Rows

**Files:**
- Modify: `src/superglm/model/fit_data_guard.py`
- Modify: `src/superglm/model/fit_ops.py`
- Modify: `src/superglm/inference/random_effects.py`
- Modify: `tests/test_fit_data_guard.py`
- Modify: `tests/test_random_effect_inference.py`

- [ ] **Step 1: Write failing response/geometry mismatch tests**

Add:

```python
def test_explicit_random_effect_rows_must_reproduce_training_inputs():
    _, model, X, y, exposure = _fit_pair(
        retain_fit_state=False,
        fit_dense=False,
        max_reml_iter=2,
    )
    weights = np.ones(len(y))
    offset = np.log(exposure)

    report = model.random_effects(
        "broker",
        X=X.copy(),
        y=y.copy(),
        sample_weight=weights,
        offset=offset,
    )
    assert report.diagnostics["row_source"] == "fit"

    changed_y = y.copy()
    changed_y[0] += 1.0
    with pytest.raises(ValueError, match="must reproduce fitted training rows"):
        model.random_effects(
            "broker",
            X=X,
            y=changed_y,
            sample_weight=weights,
            offset=offset,
        )

    evaluation = X.iloc[: len(X) // 2]
    with pytest.raises(ValueError, match="must reproduce fitted training rows"):
        model.random_effects(
            "broker",
            X=evaluation,
            y=y[: len(evaluation)],
            sample_weight=weights[: len(evaluation)],
            offset=offset[: len(evaluation)],
        )

    changed_weights = weights.copy()
    changed_weights[0] = 2.0
    with pytest.raises(ValueError, match="must reproduce fitted training rows"):
        model.random_effects(
            "broker",
            X=X,
            y=y,
            sample_weight=changed_weights,
            offset=offset,
        )

    changed_offset = offset.copy()
    changed_offset[0] += 0.1
    with pytest.raises(ValueError, match="must reproduce fitted training rows"):
        model.random_effects(
            "broker",
            X=X,
            y=y,
            sample_weight=weights,
            offset=changed_offset,
        )
```

Add this response-aware guard unit to `tests/test_fit_data_guard.py` and import
`FitGeometryGuard` there:

```python
def test_geometry_guard_can_require_the_training_response():
    X = pd.DataFrame({"x": np.linspace(-1.0, 1.0, 12)})
    y = 0.5 + 0.2 * X["x"].to_numpy()
    weights = np.linspace(0.5, 1.5, len(X))
    offset = np.linspace(-0.1, 0.1, len(X))
    guard = FitGeometryGuard.capture(
        X,
        y,
        weights,
        offset,
        columns=("x",),
    )

    assert guard.matches(X.copy(), weights.copy(), offset.copy())
    assert guard.matches_training(
        X.copy(),
        y.copy(),
        weights.copy(),
        offset.copy(),
    )
    changed = y.copy()
    changed[0] += 1.0
    assert not guard.matches_training(X, changed, weights, offset)
```

- [ ] **Step 2: Run and verify arbitrary evaluation rows are currently accepted**

```bash
rtk pytest tests/test_random_effect_inference.py -k "must_reproduce_training_inputs" -q
```

Expected: changed response or subset rows currently produce a mixed-provenance
table instead of raising.

- [ ] **Step 3: Extend `FitGeometryGuard` with response identity**

Add `y_digest: bytes`, accept `y` in `capture()`, and implement:

```python
    def matches_training(self, X, y, sample_weight, offset) -> bool:
        try:
            n_y, y_digest = _numeric_vector_digest(y)
            return bool(
                n_y == self.n_rows
                and y_digest == self.y_digest
                and self.matches(X, sample_weight, offset)
            )
        except (AttributeError, IndexError, KeyError, TypeError, ValueError, OverflowError):
            return False
```

Capture `n_y, y_digest` alongside weights/offset and require all lengths to
match the frame.

In `_prime_fit_caches()`, construct this constant-size guard for retained and
released fits:

```python
    model._fit_geometry_guard = FitGeometryGuard.capture(
        X_ref,
        y_arr,
        model._fit_weights,
        np.zeros(len(y_arr)) if model._fit_offset is None else model._fit_offset,
        columns=tuple(guard_columns),
    )
```

- [ ] **Step 4: Guard explicit report rows before prediction**

After normalizing frame, response, weights, and offset in `_reporting_rows()`:

```python
    if explicit:
        guard = getattr(model, "_fit_geometry_guard", None)
        if guard is None or not guard.matches_training(
            frame,
            y_values,
            weights,
            offset_values,
        ):
            raise ValueError(
                "random-effect reporting rows must reproduce fitted training rows, "
                "response, sample_weight, and offset; out-of-time evaluation "
                "diagnostics are not supported by random_effects()."
            )
```

Add `"row_source": "fit"` to result diagnostics.

- [ ] **Step 5: Run guard, metrics, pickle, and report tests**

```bash
rtk pytest tests/test_fit_data_guard.py tests/test_fit_state_retention.py tests/test_random_effect_inference.py tests/test_metrics.py -q
```

Expected: PASS and no regression in compact-fit metrics, which still call
`FitGeometryGuard.matches()` without response comparison.

- [ ] **Step 6: Commit**

```bash
rtk git add src/superglm/model/fit_data_guard.py src/superglm/model/fit_ops.py src/superglm/inference/random_effects.py tests/test_fit_data_guard.py tests/test_random_effect_inference.py
rtk git commit -m "Guard random-effect reporting provenance"
```

---

### Task 7: Remove the Normal Large-\(n\) Factor-Smooth Dense Transient

**Files:**
- Modify: `src/superglm/features/factor_smooth.py`
- Modify: `tests/test_factor_smooth_feature.py`
- Modify: `tests/test_factor_smooth_discrete.py`
- Create: `benchmarks/profile_factor_smooth_construction.py`
- Modify: `docs/guide/credibility.md`

- [ ] **Step 1: Write failing construction-policy and algebra tests**

Add to `tests/test_factor_smooth_feature.py`:

```python
def _built_discrete_spec(*, basis="fs", m=2, lambda_policy=None):
    x = np.linspace(-2.0, 2.0, 5000)
    group = np.array([f"g-{index % 20}" for index in range(len(x))], dtype=object)
    spec = FactorSmooth(
        "x",
        group="group",
        basis=basis,
        k=max(6, m + 4),
        m=m,
        lambda_policy=lambda_policy,
    )
    info = spec.build_discrete(x, group, {}, 256)
    return spec, info


def test_default_fs_and_sz_use_streamed_marginal_qr():
    fs, fs_info = _built_discrete_spec()
    sz, sz_info = _built_discrete_spec(basis="sz")
    assert fs._marginal_build_backend == "streamed_tsqr"
    assert sz._marginal_build_backend == "streamed_tsqr"
    assert fs_info.factor_smooth_basis is None
    assert fs_info.factor_smooth_basis_unique.shape == (256, fs.k)
    assert sz_info.factor_smooth_basis_unique.shape == (256, sz.k)


def test_asymmetric_or_high_order_fs_uses_dense_compatibility_qr():
    asymmetric = {
        "wiggle": LambdaPolicy.fixed(1.0),
        "null_0": LambdaPolicy.fixed(0.7),
        "null_1": LambdaPolicy.fixed(1.3),
    }
    custom, _ = _built_discrete_spec(lambda_policy=asymmetric)
    high_order, _ = _built_discrete_spec(m=3)
    assert custom._marginal_build_backend == "dense_qr_compat"
    assert high_order._marginal_build_backend == "dense_qr_compat"
```

Add this reduced-QR reference and compatibility-path regression in the same
test module:

```python
def _legacy_natural_parameterization(basis, penalty, *, rank):
    import scipy.linalg as la

    X = np.asarray(basis, dtype=np.float64)
    _Q, R = np.linalg.qr(X, mode="reduced")
    R_inv = la.solve_triangular(R, np.eye(R.shape[0]), lower=False)
    transformed = R_inv.T @ penalty @ R_inv
    eigenvalues, eigenvectors = la.eigh(
        0.5 * (transformed + transformed.T),
        driver="evr",
    )
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    natural_map = R_inv @ eigenvectors
    natural_map[:, :rank] /= np.sqrt(eigenvalues[:rank])
    natural_basis = X @ natural_map
    penalized_scale = 1.0 / np.sqrt(np.mean(natural_basis[:, :rank] ** 2))
    natural_map[:, :rank] *= penalized_scale
    null_dim = X.shape[1] - rank
    if null_dim:
        null_scale = 1.0 / np.sqrt(np.mean(natural_basis[:, rank:] ** 2))
        natural_map[:, rank:] *= null_scale
    wiggle = np.zeros((X.shape[1], X.shape[1]))
    wiggle[np.arange(rank), np.arange(rank)] = penalized_scale**2
    components = [("wiggle", wiggle)]
    for null_index in range(null_dim):
        component = np.zeros_like(wiggle)
        coordinate = rank + null_index
        component[coordinate, coordinate] = 1.0
        components.append((f"null_{null_index}", component))
    return natural_map, tuple(components)


def test_dense_qr_compat_matches_legacy_transform_and_penalties():
    x = np.linspace(-2.0, 2.0, 5000)
    group = np.arange(len(x), dtype=np.intp) % 20
    policies = {
        "wiggle": LambdaPolicy.fixed(1.0),
        "null_0": LambdaPolicy.fixed(0.7),
        "null_1": LambdaPolicy.fixed(1.3),
    }
    spec = FactorSmooth(
        "x",
        group="group",
        k=6,
        m=2,
        lambda_policy=policies,
    )
    spec.build_discrete(x, group, {}, 256)
    raw = np.asarray(spec._spline._raw_basis_matrix(x), dtype=np.float64)
    penalty = np.asarray(spec._spline._build_penalty(), dtype=np.float64)
    expected_map, expected_components = _legacy_natural_parameterization(
        raw,
        penalty,
        rank=spec.k - spec.m,
    )

    np.testing.assert_allclose(spec._natural_map, expected_map, atol=2e-12)
    assert [name for name, _ in spec._base_penalty_components] == [
        name for name, _ in expected_components
    ]
    for (_, actual), (_, expected) in zip(
        spec._base_penalty_components,
        expected_components,
        strict=True,
    ):
        np.testing.assert_allclose(actual, expected, atol=2e-12)
```

- [ ] **Step 2: Run tests and verify no construction policy exists**

```bash
rtk pytest tests/test_factor_smooth_feature.py -k "marginal_qr or compatibility_qr" -q
```

Expected: FAIL because every discrete build currently materializes the full
dense marginal and no backend marker exists.

- [ ] **Step 3: Refactor natural parameterization to consume only `R`**

Replace the basis-consuming `_natural_parameterization()` with:

```python
def _natural_parameterization_from_r(
    R: NDArray,
    penalty: NDArray,
    *,
    rank: int,
    n_rows: int,
) -> tuple[NDArray, tuple[tuple[str, NDArray], ...]]:
    R_array = np.asarray(R, dtype=np.float64)
    S = np.asarray(penalty, dtype=np.float64)
    if R_array.ndim != 2 or R_array.shape[0] != R_array.shape[1]:
        raise ValueError("factor-smooth QR factor must be square")
    if S.shape != R_array.shape:
        raise ValueError("factor-smooth QR factor and penalty dimensions do not agree")
    k = R_array.shape[0]
    if n_rows < k or np.linalg.matrix_rank(R_array) < k:
        raise ValueError(
            "FactorSmooth marginal basis is rank deficient; use more distinct numeric "
            "values or a smaller k, or choose a suitable non-smooth feature."
        )
    R_inv = la.solve_triangular(R_array, np.eye(k), lower=False)
    transformed_penalty = R_inv.T @ S @ R_inv
    eigenvalues, eigenvectors = la.eigh(
        0.5 * (transformed_penalty + transformed_penalty.T),
        driver="evr",
    )
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    positive = eigenvalues[:rank]
    if rank < 1 or rank > k or np.any(positive <= 0.0):
        raise ValueError("FactorSmooth marginal penalty has an invalid numerical rank")
    natural_map = R_inv @ eigenvectors
    natural_map[:, :rank] /= np.sqrt(positive)
    penalized_scale = np.sqrt(n_rows * rank / np.sum(1.0 / positive))
    natural_map[:, :rank] *= penalized_scale
    null_dim = k - rank
    if null_dim:
        natural_map[:, rank:] *= np.sqrt(n_rows)

    wiggle = np.zeros((k, k), dtype=np.float64)
    wiggle[np.arange(rank), np.arange(rank)] = penalized_scale**2
    components: list[tuple[str, NDArray]] = [("wiggle", wiggle)]
    for null_index in range(null_dim):
        component = np.zeros_like(wiggle)
        coordinate = rank + null_index
        component[coordinate, coordinate] = 1.0
        components.append((f"null_{null_index}", component))
    return natural_map, tuple(components)
```

Do not retain any `n_rows x k` array in this function.

- [ ] **Step 4: Add bounded TSQR and compatibility builders**

Use:

```python
_MARGINAL_QR_CHUNK_ROWS = 65_536


def _combine_qr_r(current: NDArray | None, basis_chunk: sp.csr_matrix) -> NDArray:
    chunk_r = np.linalg.qr(basis_chunk.toarray(), mode="r")
    if current is None:
        return chunk_r
    return np.linalg.qr(np.vstack((current, chunk_r)), mode="r")
```

Add `FactorSmooth._streaming_safe()`:

```python
    if self.basis == "sz":
        return True
    if self.m > 2:
        return False
    if self.m <= 1:
        return True
    if self._lambda_policy is None or isinstance(self._lambda_policy, LambdaPolicy):
        return True
    policies = [
        self._lambda_policy.get(f"null_{index}", LambdaPolicy.estimate())
        for index in range(self.m)
    ]
    return all(policy == policies[0] for policy in policies[1:])
```

Refactor marginal spline setup into one helper. For the streaming path,
evaluate basis chunks, combine `R`, and retain sparse chunks only for exact
`build()`. For discrete construction, retain no row basis and evaluate only
`basis_unique` after `_discretize_column()`.

For the compatibility path, materialize `raw_dense` once, call
`np.linalg.qr(raw_dense, mode="r")`, and never request `Q`, a row-scale SVD, or
`natural_basis`.

Set `_marginal_build_backend` to `"streamed_tsqr"` or `"dense_qr_compat"`.

- [ ] **Step 5: Add fit-level fixed-policy and mgcv parity gates**

Add this exact/discrete fit regression to
`tests/test_factor_smooth_discrete.py`; it exercises both the streamed,
component-symmetric path and the asymmetric compatibility path without
confounding the comparison with lambda optimization:

```python
def _fixed_policy_model(*, discrete: bool, lambda_policy) -> SuperGLM:
    return SuperGLM(
        family="poisson",
        features={},
        interactions=[
            FactorSmooth(
                "x",
                group="segment",
                k=6,
                lambda_policy=lambda_policy,
            )
        ],
        selection_penalty=0.0,
        discrete=discrete,
        n_bins=256,
        direct_solve="gram",
    )


@pytest.mark.parametrize(
    ("lambda_policy", "backend"),
    [
        (LambdaPolicy.fixed(1.0), "streamed_tsqr"),
        (
            {
                "wiggle": LambdaPolicy.fixed(1.0),
                "null_0": LambdaPolicy.fixed(0.7),
                "null_1": LambdaPolicy.fixed(1.3),
            },
            "dense_qr_compat",
        ),
    ],
    ids=["symmetric", "asymmetric"],
)
def test_factor_smooth_marginal_backend_preserves_exact_discrete_fit(
    lambda_policy,
    backend: str,
) -> None:
    X, y, weights, offset = _data()
    exact = _fixed_policy_model(
        discrete=False,
        lambda_policy=lambda_policy,
    ).fit_reml(
        X,
        y,
        sample_weight=weights,
        offset=offset,
        runtime_validation="skip",
    )
    discrete = _fixed_policy_model(
        discrete=True,
        lambda_policy=lambda_policy,
    ).fit_reml(
        X,
        y,
        sample_weight=weights,
        offset=offset,
        runtime_validation="skip",
    )

    assert exact._interaction_specs["x:segment:fs"]._marginal_build_backend == backend
    assert discrete._interaction_specs["x:segment:fs"]._marginal_build_backend == backend
    np.testing.assert_allclose(
        discrete.predict(X),
        exact.predict(X),
        rtol=2e-5,
        atol=2e-6,
    )
    assert discrete.result.deviance == pytest.approx(exact.result.deviance, rel=2e-6)
```

The symmetric TSQR natural coordinates may differ from the historical reduced
QR only by signed permutation in the two-dimensional `m=2` null space; the
fit-level prediction/deviance assertion is the invariant.

Then run:

```bash
rtk pytest tests/test_factor_smooth_feature.py tests/test_factor_smooth_discrete.py tests/test_factor_smooth_mgcv_parity.py tests/test_factor_smooth_sz_mgcv_parity.py -q
```

Expected: PASS without loosening existing pinned mgcv tolerances.

- [ ] **Step 6: Create the isolated construction profiler**

Create `benchmarks/profile_factor_smooth_construction.py` as this complete
standalone CLI:

```python
"""Profile large-row factor-smooth marginal construction."""

from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import resource
import time

import numpy as np

from superglm import FactorSmooth


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--levels", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--basis", choices=("fs", "sz"), required=True)
    parser.add_argument("--bins", type=int, required=True)
    parser.add_argument("--profile", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    x = np.linspace(-2.0, 2.0, args.rows, dtype=np.float64)
    group = np.arange(args.rows, dtype=np.intp) % args.levels
    spec = FactorSmooth(
        "x",
        group="group",
        basis=args.basis,
        k=args.k,
    )

    profile = cProfile.Profile() if args.profile else None
    started = time.perf_counter()
    if profile is None:
        info = spec.build_discrete(x, group, {}, args.bins)
    else:
        info = profile.runcall(spec.build_discrete, x, group, {}, args.bins)
    elapsed = time.perf_counter() - started

    peak_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    print(
        json.dumps(
            {
                "rows": args.rows,
                "basis": args.basis,
                "backend": spec._marginal_build_backend,
                "elapsed_s": elapsed,
                "peak_rss_mib": peak_mib,
                "support_shape": list(info.factor_smooth_basis_unique.shape),
            },
            sort_keys=True,
        )
    )
    if profile is not None:
        pstats.Stats(profile).strip_dirs().sort_stats("cumulative").print_stats(30)


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Run the one-million-row time/RSS/cProfile gate**

```bash
rtk proxy /usr/bin/time -f "maxrss_kib=%M elapsed=%e" .venv/bin/python benchmarks/profile_factor_smooth_construction.py --rows 1000000 --levels 100 --k 10 --basis fs --bins 256 --profile
rtk proxy /usr/bin/time -f "maxrss_kib=%M elapsed=%e" .venv/bin/python benchmarks/profile_factor_smooth_construction.py --rows 1000000 --levels 100 --k 10 --basis sz --bins 256
```

Expected:

- backend is `streamed_tsqr`;
- FS peak RSS is materially below the measured 810 MiB baseline and no worse
  than 450 MiB on the same host;
- marginal setup is no slower than the measured 1.62 s baseline;
- call-stack output contains no full-row SVD, QR `Q`, or transformed `n x k`
  basis construction.

- [ ] **Step 8: Document qualified large-\(n\) behavior**

Add to the current-scope section of `docs/guide/credibility.md`:

```markdown
- Discrete FS/SZ constructs the normal `m <= 2` marginal in bounded QR
  chunks before evaluating the final support basis. FS declarations with
  asymmetric null-component policies or `m > 2` retain a reduced-memory dense
  compatibility construction so those custom penalty coordinates do not
  change silently.
```

- [ ] **Step 9: Commit**

```bash
rtk git add src/superglm/features/factor_smooth.py tests/test_factor_smooth_feature.py tests/test_factor_smooth_discrete.py benchmarks/profile_factor_smooth_construction.py docs/guide/credibility.md
rtk git commit -m "Stream large factor-smooth marginals"
```

---

### Task 8: Compatibility Documentation and Complete Validation

**Files:**
- Verify/modify: `docs/getting-started/quickstart.md`
- Modify: PR #165 body
- No LSS files

- [ ] **Step 1: Verify the shorthand migration note**

Confirm the quickstart explicitly states that `splines=` emits a
`FutureWarning`, remains functional during the compatibility window, and
migrates to explicit feature specifications. If any clause is absent, use this exact
text:

```markdown
Auto-detection remains available for compatibility, but explicit feature
specs are the canonical API. Passing the legacy `splines` keyword emits a
`FutureWarning`; calls continue to run during the 0.15 compatibility window,
including with `n_knots`, `degree`, and `categorical_base`. Warning-as-error
environments should migrate named smooths to
`features={"column": Spline(kind="ps")}` before upgrading.
```

- [ ] **Step 2: Run touched-surface formatting and static checks**

```bash
rtk proxy uv run ruff format src/superglm/solvers/_structured/selection.py src/superglm/solvers/_structured/state.py src/superglm/export/rating_tables.py src/superglm/dm_builder.py src/superglm/model/base.py src/superglm/model/fit_state.py src/superglm/model/fit_ops.py src/superglm/model/fit_data_guard.py src/superglm/model/reporting_state.py src/superglm/model/reml_finalize.py src/superglm/features/random_effect.py src/superglm/features/factor_smooth.py src/superglm/inference/_term_covariance.py src/superglm/inference/_term_ops.py src/superglm/inference/_term_model_ops.py src/superglm/inference/random_effects.py src/superglm/inference/factor_smooths.py tests/test_structured_allocations.py tests/test_factor_smooth_structured_parity.py tests/test_rating_table_export.py tests/test_factor_smooth_feature.py tests/test_interactions.py tests/test_plot_api.py tests/test_drop1.py tests/test_random_effect_inference.py tests/test_factor_smooth_inference.py tests/test_factor_smooth_sz_inference.py tests/test_fit_data_guard.py tests/test_factor_smooth_discrete.py benchmarks/profile_factor_smooth_construction.py
rtk ruff check src/ tests/ benchmarks/profile_factor_smooth_construction.py
rtk mypy src/
```

Expected: all commands pass.

- [ ] **Step 3: Run focused numerical and deployment gates**

```bash
rtk pytest tests/test_structured_allocations.py tests/test_factor_smooth_structured_parity.py tests/test_factor_smooth_sz_reml.py tests/test_factor_smooth_mgcv_parity.py tests/test_factor_smooth_sz_mgcv_parity.py tests/test_random_effect_inference.py tests/test_factor_smooth_inference.py tests/test_factor_smooth_sz_inference.py tests/test_rating_table_export.py tests/test_plot_api.py tests/test_drop1.py tests/test_fit_state_retention.py -q
```

Expected: PASS.

- [ ] **Step 4: Run the full local suite**

```bash
rtk pytest tests/
```

Expected: no failures; record exact passed/skipped counts.

- [ ] **Step 5: Run release/package verification**

```bash
rtk proxy uv build
rtk proxy uv run python scripts/verify_release_artifacts.py
rtk proxy uv run python scripts/check_api_docs.py
rtk proxy uv run python run_test.py
```

Expected: wheel/sdist verification, API-doc checks, and smoke test pass.

- [ ] **Step 6: Audit scope and forbidden paths**

```bash
rtk git diff --check origin/master...HEAD
rtk proxy git diff --name-only origin/master...HEAD
rtk proxy git diff --name-only 3c4321c..HEAD | rtk grep "lss|LSS"
```

Expected: diff check passes; the final command reports zero matches.

- [ ] **Step 7: Update the PR body compatibility/release section**

State explicitly:

- RE/FS/SZ rating-table export fails loudly in 0.15.0;
- `splines=` still runs but emits `FutureWarning`;
- default discrete FS/SZ marginal construction is bounded-memory, with the
  asymmetric-policy/high-order FS compatibility qualification;
- validation counts and one-million-row peak RSS/time are from the exact head.

- [ ] **Step 8: Commit any documentation-only adjustment**

If Step 1 changed the quickstart:

```bash
rtk git add docs/getting-started/quickstart.md
rtk git commit -m "Document credibility release contracts"
```

If no local file changed, do not create an empty commit.

---

### Task 9: Push, Review, Resolve, and Prove the Exact Final Head

**Files:**
- GitHub PR #165
- No merge or publication

- [ ] **Step 1: Verify local history and push the exact source state**

```bash
rtk git status --short --branch
rtk git log --oneline origin/feature/structured-credibility..HEAD
rtk git push origin feature/structured-credibility
```

Expected: clean worktree and successful push.

- [ ] **Step 2: Obtain an independent exact-head code review**

Use the existing final-code-review capability against `HEAD`, explicitly
requesting checks for:

- structured selection/assembly coherence;
- silent/partial export;
- random-effect public API semantics;
- support-state ownership and released-state pickle behavior;
- training-row provenance;
- TSQR null-coordinate policy safety;
- large-\(n\) allocations;
- LSS scope.

Do not implement a review suggestion until reproducing it or proving it from
source.

- [ ] **Step 3: Address every justified independent finding test-first**

For each actionable finding:

1. add/reproduce a failing test;
2. implement the smallest design-consistent fix;
3. run its targeted suite;
4. commit with an imperative subsystem-scoped subject;
5. repeat the relevant full/numerical gate if solver behavior changed.

- [ ] **Step 4: Push reviewed fixes and request a fresh Codex review**

Use the GitHub review-comment workflow and post a new PR comment that tags
Codex and names the exact SHA. Ask specifically for terminal-fit/report-state
coherence, unsupported layout fallback, deployment preflight, provenance, and
FS construction parity.

Do not reuse an old approval for a changed head.

- [ ] **Step 5: Allow at least 15 minutes for Codex**

Poll the PR review state at intervals no longer than 60 seconds while providing
progress updates. Do not use one blocking `sleep 900`. Continue safe local
inspection between polls.

- [ ] **Step 6: Reproduce and fix every actionable Codex comment**

Use the GitHub address-comments workflow for thread-level context. For each
comment, record:

- reproduced and fixed;
- already fixed on the exact head;
- rejected with source/test evidence;
- non-actionable product suggestion deferred by the approved spec.

Resolve a thread only after the corresponding committed fix is pushed.

- [ ] **Step 7: Re-tag Codex after any material fix**

Repeat Steps 4–6 until the exact current head receives no actionable findings.
Each changed SHA requires a fresh review request and its own wait window.

- [ ] **Step 8: Require green CI on the exact final SHA**

```bash
rtk gh pr checks 165
rtk gh pr view 165 --json headRefOid,isDraft,mergeStateStatus,reviewDecision,statusCheckRollup
```

Expected:

- `headRefOid` equals local `HEAD`;
- every required check succeeds;
- PR remains draft;
- no merge or publication occurs.

If checks are still pending, poll `rtk gh pr checks 165` at intervals no longer
than 60 seconds, yielding progress between polls; do not use a single blocking
watch or sleep.

- [ ] **Step 9: Run the completion audit**

For each numbered review finding and every design validation gate, cite the
authoritative test, benchmark, source diff, PR thread, or CI check proving it.
Confirm:

- all justified findings are fixed;
- the deprecation decision is documented rather than removed;
- exact-head independent and Codex reviews are clean;
- every actionable thread is resolved;
- the branch is clean and synchronized;
- LSS is untouched;
- PR #165 remains draft, unmerged, and unpublished.

Only then mark the active goal complete.
