# Mixed-Type Interaction Screening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend PSST (`SuperGLM.screen_interactions`) beyond spline×spline to spline×cat, numeric×cat, cat×cat, and numeric×numeric pairs — plus spline-mode OrderedCategorical margins, which first requires making OC a working interaction parent on the fit side.

**Architecture:** Per-kind margin adapters feed the existing cell-table → score/curvature → overlap-profiling → penalized-statistic pipeline. Categorical margins are gridded margins with an (L, L−1) contrast menu and zero penalty (no new kernels). Numeric margins never grid: z-weighted moment channels through plain bincounts make those pairs exact. A stateless resolver at the interaction input boundary maps spline-mode OC parents to (inner Spline, mapped scores).

**Tech Stack:** Python 3.12+, numpy/scipy/pandas, pytest via `uv run pytest`.

**Spec:** `docs/superpowers/specs/2026-07-31-mixed-interaction-screening-design.md` (committed on this branch). Read it before starting any task.

## Global Constraints

- Branch: `mixed-interaction-screening`, cut from `origin/master` @ c1b1339. All work stays on it.
- `docs/superpowers/` is **gitignored** — commit plan/spec docs with `git add -f`.
- Never cite or reference R package source code (mgcv/scam internals, file:line, identifiers) in code, comments, docs, or commit messages. Naming a package in passing in prose is fine; published papers are fine.
- Ranking-only language everywhere: the statistic is never a p-value and must not be described as one.
- Deferred and OUT of this plan: spline×numeric screening, `SplineNumeric` term class, Polynomial margins, step-mode-OC as interaction parent (gets an explicit error), 3-way sweeps, fusion-penalty ladders.
- Commit messages follow house style: sentence-case imperative first line, no `feat:`/`fix:` prefixes, footer `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Test runner: `uv run pytest <path> -x -q` (never filter CI to required checks; Python 3.10 numerics divergences are a known non-issue — this repo floors at 3.12).
- Do not modify `pair_cell_moments`, `pair_score_curvature`, `pair_overlap_moments`, `penalized_score_statistic`, or `working_score` — every extension composes around them; their release pins must keep passing untouched.

---

### Task 1: OC interaction-parent resolver + build-path enabler

Spline-mode `OrderedCategorical` is dispatched as a "spline" parent by `_spec_kind` but every interaction build raises `TypeError: Expected a spline spec` because the builders receive the OC spec and raw labels. Fix: resolve `(spec, column)` at the interaction assembly boundary in `build_design_matrix`. Step-mode OC gets an explicit `NotImplementedError`. OC-parented pairs are gated out of the discrete-tensor/spline-cat build paths (their support is ≤K score points — nothing to compress).

**Files:**
- Modify: `src/superglm/features/ordered_categorical.py` (add module-level function at end of file)
- Modify: `src/superglm/dm_builder.py` (interaction loop ~line 945; `should_discretize_tensor_interaction`; `should_discretize_spline_categorical_interaction`)
- Test: `tests/test_ordered_categorical_interactions.py` (new file)

**Interfaces:**
- Consumes: `OrderedCategorical` internals (`basis`, `_spline`, `_grouping`, `_known_levels`, `_map_to_numeric`), `_grouping_labels`, `_validate_categorical_levels` (both already imported in `ordered_categorical.py`).
- Produces: `resolve_interaction_parent(spec, x) -> tuple[Any, NDArray]` in `superglm.features.ordered_categorical` — identity for non-OC specs **and for `spec=None`** (FactorSmooth group columns have no feature spec); Tasks 2 and 6 import it.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ordered_categorical_interactions.py`:

```python
"""OrderedCategorical as an interaction parent: build-side enabler."""

import numpy as np
import pandas as pd
import pytest

from superglm import Categorical, SuperGLM
from superglm.features.ordered_categorical import (
    OrderedCategorical,
    resolve_interaction_parent,
)
from superglm.features.spline import Spline, _SplineBase

BANDS = ["18-25", "26-35", "36-45", "46-55", "56+"]


def _frame(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "age_band": rng.choice(BANDS, n),
            "region": rng.choice(list("ABCD"), n),
            "power": rng.uniform(20.0, 200.0, n),
        }
    )
    band_effect = df["age_band"].map({b: v for b, v in zip(BANDS, [0.4, 0.1, 0.0, 0.1, 0.3])})
    eta = -1.5 + band_effect + 0.002 * df["power"]
    y = rng.poisson(np.exp(eta)).astype(np.float64)
    return df, y


def _oc():
    return OrderedCategorical(order=BANDS, basis=Spline(kind="ps", n_knots=4))


def test_resolver_is_identity_for_non_oc_and_none():
    x = np.array([1.0, 2.0])
    spec = Spline(kind="ps", n_knots=4)
    assert resolve_interaction_parent(spec, x) == (spec, x)
    assert resolve_interaction_parent(None, x) == (None, x)


def test_resolver_maps_oc_to_inner_spline_and_scores():
    spec = _oc()
    labels = np.array(["18-25", "56+", "36-45"], dtype=object)
    eff_spec, eff_x = resolve_interaction_parent(spec, labels)
    assert isinstance(eff_spec, _SplineBase)
    assert eff_spec is spec._spline
    expected = [spec._level_to_value[v] for v in labels]
    np.testing.assert_allclose(eff_x, expected)


def test_resolver_rejects_step_mode():
    with pytest.warns(FutureWarning):
        spec = OrderedCategorical(order=BANDS, basis="step")
    with pytest.raises(NotImplementedError, match="step"):
        resolve_interaction_parent(spec, np.array(BANDS, dtype=object))


def test_resolver_rejects_unseen_levels():
    spec = _oc()
    with pytest.raises(ValueError, match="unseen|levels"):
        resolve_interaction_parent(spec, np.array(["99+"], dtype=object))


def test_oc_categorical_interaction_fits():
    df, y = _frame()
    model = SuperGLM(
        family="poisson",
        features={"age_band": _oc(), "region": Categorical()},
        interactions=[("age_band", "region")],
    )
    model.fit_reml(df, y)
    assert np.isfinite(model._result.effective_df)


def test_oc_spline_tensor_interaction_fits():
    df, y = _frame()
    model = SuperGLM(
        family="poisson",
        features={"age_band": _oc(), "power": Spline(kind="ps", n_knots=5)},
        interactions=[("age_band", "power")],
    )
    model.fit_reml(df, y)
    assert np.isfinite(model._result.effective_df)


def test_oc_tensor_fit_matches_manual_score_mapping():
    # The OC×spline fit must equal the same model fitted on the scores directly.
    df, y = _frame()
    oc = _oc()
    model_oc = SuperGLM(
        family="poisson",
        features={"age_band": _oc(), "power": Spline(kind="ps", n_knots=5)},
        interactions=[("age_band", "power")],
    )
    model_oc.fit_reml(df, y)

    df_num = df.copy()
    df_num["age_band"] = df_num["age_band"].map(oc._level_to_value)
    model_num = SuperGLM(
        family="poisson",
        features={"age_band": Spline(kind="ps", n_knots=4), "power": Spline(kind="ps", n_knots=5)},
        interactions=[("age_band", "power")],
    )
    model_num.fit_reml(df_num, y)
    np.testing.assert_allclose(
        model_oc._result.deviance, model_num._result.deviance, rtol=1e-6
    )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_ordered_categorical_interactions.py -x -q`
Expected: FAIL — `ImportError: cannot import name 'resolve_interaction_parent'`.

- [ ] **Step 3: Implement the resolver**

Append to `src/superglm/features/ordered_categorical.py` (module level, after the class):

```python
def resolve_interaction_parent(spec: Any, x: NDArray) -> tuple[Any, NDArray]:
    """Resolve one interaction parent (spec, column) for assembly.

    Identity for every spec — including ``None``, which FactorSmooth group
    columns carry — except spline-mode OrderedCategorical, which
    contributes its inner Spline on the mapped numeric scores, applying
    the same grouping, level validation, and score mapping its own
    ``build``/``transform`` apply.  Step-mode OC cannot parent an
    interaction: the deprecated one-hot geometry has no marginal smooth.
    """
    if not isinstance(spec, OrderedCategorical):
        return spec, x
    if spec.basis != "spline" or spec._spline is None:
        raise NotImplementedError(
            "OrderedCategorical with basis='step' is deprecated and cannot parent "
            "an interaction; use basis=Spline(...) for a smoothed ordinal parent "
            "or a Categorical feature for unsmoothed level effects."
        )
    x = np.asarray(x).ravel()
    if spec._grouping is not None:
        x = _grouping_labels(x)
        valid = spec._known_levels | set(spec._grouping.grouped_levels)
        _validate_categorical_levels(x, valid)
        x = np.array(
            [spec._grouping.original_to_group.get(v, v) for v in x], dtype=object
        )
    else:
        _validate_categorical_levels(x, spec._known_levels)
    return spec._spline, spec._map_to_numeric(x)
```

(The grouped branch mirrors `OrderedCategorical.transform`, which is the permissive
form valid at both build and predict time.)

- [ ] **Step 4: Wire the resolver into `build_design_matrix`**

In `src/superglm/dm_builder.py`, the interaction loop currently reads:

```python
    for iname in interaction_order:
        ispec = interaction_specs[iname]
        p1, p2 = ispec.parent_names
        x1 = X.column_array(p1)
        x2 = X.column_array(p2)
```

Replace with:

```python
    from superglm.features.ordered_categorical import resolve_interaction_parent

    for iname in interaction_order:
        ispec = interaction_specs[iname]
        p1, p2 = ispec.parent_names
        spec1, x1 = resolve_interaction_parent(specs.get(p1), X.column_array(p1))
        spec2, x2 = resolve_interaction_parent(specs.get(p2), X.column_array(p2))
        if spec1 is specs.get(p1) and spec2 is specs.get(p2):
            parent_specs = specs
        else:
            parent_specs = {**specs, p1: spec1, p2: spec2}
```

(put the import at module top with the other `superglm.features` imports, not
inside the loop). Then, in the SAME loop body, change the two generic build
calls to use the resolved view — `ispec.build(x1, x2, parent_specs,
sample_weight=sample_weight)` and each `ispec.build_discrete(x1, x2,
parent_specs, ...)` — while the three `should_discretize_*` gate calls keep
receiving the ORIGINAL `specs` (the OC gate added in Step 5 lives there and
must see the OC spec, not the resolved spline).

- [ ] **Step 5: Gate OC parents out of the discrete build paths**

In `src/superglm/dm_builder.py`, add as the FIRST lines of both
`should_discretize_tensor_interaction` and
`should_discretize_spline_categorical_interaction`:

```python
    from superglm.features.ordered_categorical import OrderedCategorical

    if any(
        isinstance(specs.get(p), OrderedCategorical)
        for p in getattr(ispec, "parent_names", ())
    ):
        # OC margins live on <= n_levels score points; there is nothing for
        # fit-time discretization to compress, and the fast-discrete predict
        # metadata reads raw columns as float64, which label data cannot be.
        return False
```

(Both functions are called as `(ispec, specs, model_discrete)` at their call
sites in the interaction loop; write the guard in terms of each function's
own parameter names for the interaction spec and the parent-spec dict.)

- [ ] **Step 6: Run the new tests and the regression suites**

Run: `uv run pytest tests/test_ordered_categorical_interactions.py -x -q`
Expected: PASS (the two predict-dependent behaviors are Task 2; this file has no predict tests yet).

Run: `uv run pytest tests/test_interaction_screening.py -q` and the interaction/design-matrix suites: `uv run pytest tests/ -q -k "interaction or dm_builder or factor_smooth"`.
Expected: PASS — the resolver is identity for every existing configuration.

- [ ] **Step 7: Commit**

```bash
git add src/superglm/features/ordered_categorical.py src/superglm/dm_builder.py tests/test_ordered_categorical_interactions.py
git commit -m "Let spline-mode OrderedCategorical parent an interaction

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: OC prediction path

`predict` scores interactions with raw parent columns (`model/base.py`,
`_score_prediction_term_local_exact`), so an OC-parented interaction would feed
labels to a spline-built term. Resolve the columns through the same resolver at
scoring time. The fast-discrete predict path is already gated off by Task 1
Step 5 (metadata compile consults `should_discretize_tensor_interaction`).

**Files:**
- Modify: `src/superglm/model/base.py` (`_build_prediction_plan` interaction term dict ~line 228; `_score_prediction_term_local_exact` ~line 340)
- Test: `tests/test_ordered_categorical_interactions.py` (extend)

**Interfaces:**
- Consumes: `resolve_interaction_parent` from Task 1.
- Produces: interaction term dicts carry `"parent_specs": tuple(model._specs.get(p) for p in parent_names)`; exact scoring resolves through it. No public API change.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_ordered_categorical_interactions.py`:

```python
def test_oc_interaction_predict_round_trips_training_frame():
    df, y = _frame()
    model = SuperGLM(
        family="poisson",
        features={"age_band": _oc(), "region": Categorical()},
        interactions=[("age_band", "region")],
    )
    model.fit_reml(df, y)
    mu_train = model.predict(df)
    assert mu_train.shape == (len(df),)
    assert np.all(np.isfinite(mu_train)) and np.all(mu_train > 0)


def test_oc_tensor_predict_matches_manual_score_mapping():
    df, y = _frame()
    oc = _oc()
    model_oc = SuperGLM(
        family="poisson",
        features={"age_band": _oc(), "power": Spline(kind="ps", n_knots=5)},
        interactions=[("age_band", "power")],
    )
    model_oc.fit_reml(df, y)
    new = df.iloc[:200].copy()
    pred_oc = model_oc.predict(new)

    df_num = df.copy()
    df_num["age_band"] = df_num["age_band"].map(oc._level_to_value)
    model_num = SuperGLM(
        family="poisson",
        features={"age_band": Spline(kind="ps", n_knots=4), "power": Spline(kind="ps", n_knots=5)},
        interactions=[("age_band", "power")],
    )
    model_num.fit_reml(df_num, y)
    new_num = new.copy()
    new_num["age_band"] = new_num["age_band"].map(oc._level_to_value)
    np.testing.assert_allclose(pred_oc, model_num.predict(new_num), rtol=1e-8)


def test_oc_interaction_predict_rejects_unseen_level():
    df, y = _frame()
    model = SuperGLM(
        family="poisson",
        features={"age_band": _oc(), "region": Categorical()},
        interactions=[("age_band", "region")],
    )
    model.fit_reml(df, y)
    bad = df.iloc[:5].copy()
    bad.loc[bad.index[0], "age_band"] = "99+"
    with pytest.raises(ValueError, match="unseen|levels"):
        model.predict(bad)


def test_oc_interaction_added_post_hoc_refits():
    # Exercises the config-template deepcopy path (the editor-clone contract):
    # add_interaction stores a deep-copied template that the next fit rebuilds.
    df, y = _frame()
    model = SuperGLM(
        family="poisson",
        features={"age_band": _oc(), "region": Categorical()},
    )
    model.fit_reml(df, y)
    model._add_interaction("age_band", "region")
    model.fit_reml(df, y)
    mu = model.predict(df.iloc[:50])
    assert np.all(np.isfinite(mu)) and np.all(mu > 0)


def test_oc_interaction_survives_discrete_mode():
    df, y = _frame()
    model = SuperGLM(
        family="poisson",
        features={"age_band": _oc(), "power": Spline(kind="ps", n_knots=5)},
        interactions=[("age_band", "power")],
        discrete=True,
    )
    model.fit_reml(df, y)
    mu = model.predict(df.iloc[:100])
    assert np.all(np.isfinite(mu))
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_ordered_categorical_interactions.py -x -q`
Expected: the predict tests FAIL (labels reach numeric code — a dtype/`TypeError`/`ValueError` inside scoring, or a mapping mismatch).

- [ ] **Step 3: Stash parent specs in the prediction plan and resolve at scoring**

In `src/superglm/model/base.py`, where `_build_prediction_plan` assembles the
interaction term dict (the block containing `"parent_names":
tuple(model._interaction_specs[name].parent_names)`), add one key:

```python
                "parent_specs": tuple(
                    model._specs.get(p)
                    for p in model._interaction_specs[name].parent_names
                ),
```

In `_score_prediction_term_local_exact`, replace the interaction branch:

```python
    left_name, right_name = term["parent_names"]
    return _score_interaction(
        term["spec"],
        X.column_array(left_name),
        X.column_array(right_name),
        beta,
    )
```

with:

```python
    from superglm.features.ordered_categorical import resolve_interaction_parent

    left_name, right_name = term["parent_names"]
    left_spec, right_spec = term.get("parent_specs", (None, None))
    _, left = resolve_interaction_parent(left_spec, X.column_array(left_name))
    _, right = resolve_interaction_parent(right_spec, X.column_array(right_name))
    return _score_interaction(term["spec"], left, right, beta)
```

(hoist the import to module top; `term.get` keeps any cached plan built before
this change working — identity path for `None`).

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_ordered_categorical_interactions.py -x -q`
Expected: PASS.

Run: `uv run pytest tests/ -q -k "predict or interaction"`
Expected: PASS (identity resolution leaves every existing path bit-identical).

- [ ] **Step 5: Commit**

```bash
git add src/superglm/model/base.py tests/test_ordered_categorical_interactions.py
git commit -m "Resolve OrderedCategorical parents on the interaction predict path

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Screening pair kinds, eligibility, and validation

Teach `screening_ops` which features are screenable, what kind each pair is,
which kinds are deferred, and extend fitted-pair exclusion from
`TensorInteraction` to every interaction class plus `FactorSmooth`. No new
statistics yet — the sweep still computes only `ti` pairs; other kinds raise
`NotImplementedError` from the dispatch point so this task is honestly
testable and the next tasks each delete one arm of that error.

**Files:**
- Modify: `src/superglm/model/screening_ops.py`
- Test: `tests/test_mixed_interaction_screening.py` (new file)

**Interfaces:**
- Consumes: fitted spec internals (`_levels`, `_non_base`, `_base_level`, `_grouping`, `_known_levels`; OC `basis`/`_spline`).
- Produces (module-private, used by Tasks 4–6):
  - `_margin_kind(spec) -> str | None` — `"spline" | "categorical" | "numeric" | None`; spline-mode OC → `"spline"`; step-OC, `Polynomial`, `RandomEffect`, <2-level `Categorical` → `None`.
  - `_pair_kind(kind_a, kind_b) -> str | None` — `"ti" | "spline_cat" | "numeric_cat" | "cat_cat" | "numeric_numeric"`; `None` for deferred combos (spline+numeric).
  - `_validated_pairs(candidates, margin_kinds, fitted_pairs) -> list[tuple[str, str]]` (signature changes: takes the name→kind dict instead of `spline_names`).
  - Result frame gains a `kind` column (position 3, after `feature_b`).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_mixed_interaction_screening.py`:

```python
"""Mixed-type PSST: eligibility, pair kinds, and per-kind screening."""

import numpy as np
import pandas as pd
import pytest

from superglm import Categorical, SuperGLM
from superglm.features.numeric import Numeric
from superglm.features.ordered_categorical import OrderedCategorical
from superglm.features.polynomial import Polynomial
from superglm.features.spline import Spline

BANDS = ["18-25", "26-35", "36-45", "46-55", "56+"]


def _mixed_frame(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "age": rng.uniform(18.0, 80.0, n),
            "power": rng.uniform(20.0, 200.0, n),
            "region": rng.choice(list("ABCD"), n),
            "brand": rng.choice(["B1", "B2", "B3"], n),
            "bm": rng.uniform(0.5, 2.0, n),
            "band": rng.choice(BANDS, n),
        }
    )
    return df, rng


def _fit_mixed(df, y, **kwargs):
    model = SuperGLM(
        family="poisson",
        features={
            "age": Spline(kind="ps", n_knots=6),
            "power": Spline(kind="ps", n_knots=6),
            "region": Categorical(),
            "brand": Categorical(),
            "bm": Numeric(),
            "band": OrderedCategorical(order=BANDS, basis=Spline(kind="ps", n_knots=4)),
        },
        **kwargs,
    )
    model.fit_reml(df, y)
    return model


def _null_y(df, rng):
    return rng.poisson(np.exp(-1.5 + 0.004 * df["age"]), len(df)).astype(np.float64)


def test_default_sweep_covers_every_eligible_kind():
    df, rng = _mixed_frame()
    y = _null_y(df, rng)
    model = _fit_mixed(df, y)
    table = model.screen_interactions(df, y)
    got = {
        (frozenset((a, b)), kind)
        for a, b, kind in zip(table["feature_a"], table["feature_b"], table["kind"])
    }
    # spot-check one pair of every kind
    assert (frozenset(("age", "power")), "ti") in got
    assert (frozenset(("age", "band")), "ti") in got          # OC screens as a spline margin
    assert (frozenset(("age", "region")), "spline_cat") in got
    assert (frozenset(("bm", "region")), "numeric_cat") in got
    assert (frozenset(("region", "brand")), "cat_cat") in got
    # numeric_numeric needs two Numerics; single bm pairs with nothing numeric
    assert not any(k == "numeric_numeric" for _, k in got)
    # deferred: spline x numeric absent from the default sweep
    assert (frozenset(("age", "bm"))) not in {p for p, _ in got}


def test_candidates_rejects_deferred_and_ineligible_kinds():
    df, rng = _mixed_frame()
    y = _null_y(df, rng)
    model = _fit_mixed(df, y)
    with pytest.raises(ValueError, match="deferred"):
        model.screen_interactions(df, y, candidates=[("age", "bm")])
    df2 = df.assign(poly=np.linspace(0.0, 1.0, len(df)))
    model2 = SuperGLM(
        family="poisson",
        features={"age": Spline(kind="ps", n_knots=6), "poly": Polynomial(degree=2)},
    )
    model2.fit_reml(df2, y)
    with pytest.raises(ValueError, match="screenable|eligible"):
        model2.screen_interactions(df2, y, candidates=[("age", "poly")])


def test_fitted_pairs_of_every_class_are_excluded():
    df, rng = _mixed_frame()
    y = _null_y(df, rng)
    model = _fit_mixed(df, y, interactions=[("region", "brand"), ("age", "region")])
    table = model.screen_interactions(df, y)
    pairs = {frozenset((a, b)) for a, b in zip(table["feature_a"], table["feature_b"])}
    assert frozenset(("region", "brand")) not in pairs
    assert frozenset(("age", "region")) not in pairs
    with pytest.raises(ValueError, match="already fitted"):
        model.screen_interactions(df, y, candidates=[("region", "brand")])


def test_factor_smooth_pair_is_excluded():
    from superglm import FactorSmooth

    df, rng = _mixed_frame()
    y = _null_y(df, rng)
    model = SuperGLM(
        family="poisson",
        features={"age": Spline(kind="ps", n_knots=6, m=2), "region": Categorical()},
        interactions=[FactorSmooth("age", group="region", basis="fs", kind="ps", k=5)],
        selection_penalty=0.0,
    )
    model.fit_reml(df, y)
    table = model.screen_interactions(df, y)
    pairs = {frozenset((a, b)) for a, b in zip(table["feature_a"], table["feature_b"])}
    assert frozenset(("age", "region")) not in pairs
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_mixed_interaction_screening.py -x -q`
Expected: FAIL — no `kind` column / `KeyError: 'kind'`, and categorical pairs absent from the sweep.

- [ ] **Step 3: Implement kinds, eligibility, exclusion, and dispatch skeleton**

In `src/superglm/model/screening_ops.py`:

1. Add imports: `from superglm.features.categorical import Categorical,
   _grouping_labels, _validate_categorical_levels`, `from
   superglm.features.numeric import Numeric`, `from
   superglm.features.ordered_categorical import OrderedCategorical`.

2. Add module-level helpers:

```python
_DEFERRED_KIND_HINT = (
    "spline x numeric screening is deferred until a varying-coefficient "
    "interaction term exists; respec the Numeric parent as a Spline to screen "
    "the pair as ti(), or see the screening guide. Polynomial margins are "
    "likewise deferred."
)


def _margin_kind(spec) -> str | None:
    """Classify a fitted spec for screening; None means not screenable."""
    if isinstance(spec, _SplineBase):
        return "spline"
    if isinstance(spec, OrderedCategorical):
        if spec.basis == "spline" and spec._spline is not None:
            return "spline"
        return None
    if isinstance(spec, Categorical):
        return "categorical" if len(spec._levels) >= 2 else None
    if isinstance(spec, Numeric):
        return "numeric"
    return None


_PAIR_KINDS = {
    frozenset(("spline",)): "ti",
    frozenset(("spline", "categorical")): "spline_cat",
    frozenset(("numeric", "categorical")): "numeric_cat",
    frozenset(("categorical",)): "cat_cat",
    frozenset(("numeric",)): "numeric_numeric",
}


def _pair_kind(kind_a: str, kind_b: str) -> str | None:
    return _PAIR_KINDS.get(frozenset((kind_a, kind_b)))
```

3. Replace `_validated_pairs` with the kind-aware version (same error style):

```python
def _validated_pairs(candidates, margin_kinds, fitted_pairs):
    if candidates is None:
        return [
            pair
            for pair in combinations(margin_kinds, 2)
            if _pair_kind(margin_kinds[pair[0]], margin_kinds[pair[1]]) is not None
            and frozenset(pair) not in fitted_pairs
        ]
    pairs = []
    for raw in candidates:
        pair = tuple(raw)
        if len(pair) != 2 or pair[0] == pair[1] or not set(margin_kinds).issuperset(pair):
            raise ValueError(
                "candidates entries must pair two distinct screenable fitted "
                f"features; got {raw!r} (screenable features: "
                f"{sorted(margin_kinds)})"
            )
        if _pair_kind(margin_kinds[pair[0]], margin_kinds[pair[1]]) is None:
            raise ValueError(
                f"candidates entry {raw!r} pairs kinds "
                f"({margin_kinds[pair[0]]}, {margin_kinds[pair[1]]}) with no "
                f"refit target — {_DEFERRED_KIND_HINT}"
            )
        if frozenset(pair) in fitted_pairs:
            raise ValueError(
                f"candidates entry {raw!r} is already fitted as an interaction; "
                "screening profiles only the parent mains and cannot re-screen it"
            )
        pairs.append(pair)
    return pairs
```

4. In `screen_interactions`, replace the `spline_names` block with:

```python
    margin_kinds = {
        name: kind
        for name in model._feature_order
        if (kind := _margin_kind(model._specs.get(name))) is not None
    }
    fitted_pairs = {
        frozenset(spec.parent_names)
        for spec in getattr(model, "_interaction_specs", {}).values()
        if hasattr(spec, "parent_names")
    }
    pairs = _validated_pairs(candidates, margin_kinds, fitted_pairs)
```

Keep the `select=True` guard, applying it to spline margins only — for OC
margins consult the inner spline:

```python
    def _select_flag(name):
        spec = model._specs[name]
        if isinstance(spec, OrderedCategorical):
            spec = spec._spline
        return getattr(spec, "select", False)

    selected = sorted(
        {name for pair in pairs for name in pair if _select_flag(name)}
    )
```

5. Per-pair dispatch skeleton: at the top of the pair loop compute
`kind = _pair_kind(margin_kinds[feat_a], margin_kinds[feat_b])` and raise
`NotImplementedError(f"screening kind {kind!r} lands in a later task")` for
anything but `"ti"`. Add `"kind"` to `_RESULT_COLUMNS` after `"feature_b"` and
to every `rows.append` (the `ti` arm appends `kind`). The `_raw` prefetch loop
must only touch spline/numeric margins for now (categorical raw columns are
object dtype — Task 4 adds their path), so guard the prefetch with
`if margin_kinds[name] != "categorical"`.

6. `_marginal_width_estimate` gains the exact non-spline widths:

```python
def _marginal_width_estimate(spec) -> int:
    if isinstance(spec, OrderedCategorical) and spec._spline is not None:
        spec = spec._spline
    if isinstance(spec, Categorical):
        return max(len(spec._levels) - 1, 1)
    if isinstance(spec, Numeric):
        return 1
    n_knots = getattr(spec, "n_knots", None)
    if n_knots is not None:
        return max(int(n_knots), 1)
    return 1
```

(keep the existing docstring, extending it with one line: categorical and
numeric widths are exact, not estimates).

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_mixed_interaction_screening.py -x -q`
Expected: the eligibility/validation/exclusion tests in this file PASS except
`test_default_sweep_covers_every_eligible_kind` (blocked on
`NotImplementedError` for non-ti kinds) — mark it
`@pytest.mark.xfail(reason="kinds land in tasks 4-6", strict=True)` for now;
Task 5 removes the marker.

Run: `uv run pytest tests/test_interaction_screening.py -q`
Expected: PASS — spline-only models see identical behavior plus the new
constant `kind == "ti"` column (fix any existing test asserting the exact
column list by updating it to include `kind`).

- [ ] **Step 5: Commit**

```bash
git add src/superglm/model/screening_ops.py tests/test_mixed_interaction_screening.py tests/test_interaction_screening.py
git commit -m "Resolve screening pair kinds and eligibility for mixed margins

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Categorical margins — cat_cat and spline_cat

Categorical margins are gridded margins with an (L, L−1) contrast menu and no
penalty; they flow through the existing kernels verbatim. `tensor_penalty(S,
0)` already yields `kron(S, I)` for the spline_cat block, and
`penalized_score_statistic` with an all-zero `S` already returns the
unpenalized Rao statistic — evaluate such pairs at a single rung.

**Files:**
- Modify: `src/superglm/model/screening_ops.py`
- Modify: `src/superglm/model/api.py` (screen_interactions docstring: kinds table)
- Test: `tests/test_interaction_screening.py` (margin exactness pins, reusing `_dense_row_kronecker`)
- Test: `tests/test_mixed_interaction_screening.py` (end-to-end power + df pins)

**Interfaces:**
- Consumes: Task 3's `_margin_kind`/`_pair_kind` dispatch; existing kernels unchanged.
- Produces (used by Task 6): `_categorical_codes(spec, x_raw) -> tuple[NDArray, int]` (dense codes over ALL fitted levels, L) and `_contrast_menu(spec) -> NDArray` of shape (L, L−1); the both-gridded pair path parameterized by per-margin `(codes, n, menu, S)`.

- [ ] **Step 1: Write the failing exactness pins**

Append to `tests/test_interaction_screening.py`:

```python
def test_contrast_menu_kron_is_the_pair_indicator_block():
    # kron of contrast menus on the level-pair grid == CategoricalInteraction's
    # non-base pair indicator columns, row for row.
    rng = np.random.default_rng(7)
    n, L1, L2 = 400, 4, 3
    codes_a = rng.integers(0, L1, n)
    codes_b = rng.integers(0, L2, n)
    menu_a = np.zeros((L1, L1 - 1)); menu_a[1:, :] = np.eye(L1 - 1)  # base = level 0
    menu_b = np.zeros((L2, L2 - 1)); menu_b[1:, :] = np.eye(L2 - 1)
    X = _dense_row_kronecker(codes_a, codes_b, menu_a, menu_b)
    expected = np.column_stack(
        [
            (codes_a == i).astype(float) * (codes_b == j).astype(float)
            for i in range(1, L1)
            for j in range(1, L2)
        ]
    )
    np.testing.assert_allclose(X, expected)


def test_cell_assembly_exact_for_contrast_menus():
    rng = np.random.default_rng(8)
    n, L1, L2 = 500, 5, 4
    codes_a = rng.integers(0, L1, n)
    codes_b = rng.integers(0, L2, n)
    menu_a = np.zeros((L1, L1 - 1)); menu_a[1:, :] = np.eye(L1 - 1)
    menu_b = np.zeros((L2, L2 - 1)); menu_b[1:, :] = np.eye(L2 - 1)
    score = rng.normal(size=n)
    w = rng.uniform(0.1, 3.0, n)
    S_cell, W_cell = pair_cell_moments(codes_a, codes_b, L1, L2, score, w)
    U, V = pair_score_curvature(menu_a, menu_b, S_cell, W_cell)
    X = _dense_row_kronecker(codes_a, codes_b, menu_a, menu_b)
    np.testing.assert_allclose(U, X.T @ score, rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(V, X.T @ (X * w[:, None]), rtol=1e-12, atol=1e-9)
```

(match the existing file's import style — `pair_cell_moments` and
`pair_score_curvature` are already imported there.)

Append to `tests/test_mixed_interaction_screening.py`:

```python
def test_cat_cat_planted_table_ranks_first_with_exact_df():
    df, rng = _mixed_frame(n=20000, seed=3)
    boost = ((df["region"] == "B") & (df["brand"] == "B2")).astype(float)
    y = rng.poisson(np.exp(-1.3 + 0.5 * boost)).astype(np.float64)
    model = _fit_mixed(df, y)
    table = model.screen_interactions(df, y)
    top = table.iloc[0]
    assert {top["feature_a"], top["feature_b"]} == {"region", "brand"}
    assert top["kind"] == "cat_cat"
    # (L1-1)(L2-1) = 3*2; unpenalized rung reports the achieved rank as edf0
    assert top["edf0"] == pytest.approx(6.0, abs=0.26)
    assert top["lambda0"] == 0.0
    assert top["z"] > 10.0
    row = table[table["kind"] == "cat_cat"].iloc[0]
    assert row["n_cells"] == 4 * 3
    assert not row["approx"]


def test_spline_cat_planted_deviation_curve_ranks_first():
    df, rng = _mixed_frame(n=20000, seed=4)
    bend = np.where(
        df["region"] == "C", np.sin((df["age"] - 18.0) / 62.0 * np.pi) * 0.4, 0.0
    )
    y = rng.poisson(np.exp(-1.3 + bend)).astype(np.float64)
    model = _fit_mixed(df, y)
    table = model.screen_interactions(df, y)
    top = table.iloc[0]
    assert {top["feature_a"], top["feature_b"]} == {"age", "region"}
    assert top["kind"] == "spline_cat"
    assert top["z"] > 8.0


def test_two_level_factor_pairs_are_legal():
    df, rng = _mixed_frame(n=6000, seed=12)
    df = df.assign(fuel=rng.choice(["diesel", "petrol"], len(df)))
    y = _null_y(df, rng)
    model = SuperGLM(
        family="poisson",
        features={"fuel": Categorical(), "brand": Categorical()},
    )
    model.fit_reml(df, y)
    table = model.screen_interactions(df, y)
    row = table.iloc[0]
    assert row["kind"] == "cat_cat"
    assert row["edf0"] == pytest.approx(2.0, abs=0.26)  # (2-1)*(3-1)
    assert np.isfinite(row["z"])


def test_spline_cat_confirms_by_refit():
    df, rng = _mixed_frame(n=20000, seed=4)
    bend = np.where(
        df["region"] == "C", np.sin((df["age"] - 18.0) / 62.0 * np.pi) * 0.4, 0.0
    )
    y = rng.poisson(np.exp(-1.3 + bend)).astype(np.float64)
    base = _fit_mixed(df, y)
    dev0 = base._result.deviance
    confirm = _fit_mixed(df, y, interactions=[("age", "region")])
    assert dev0 - confirm._result.deviance > 50.0
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_interaction_screening.py -q -k "contrast" ; uv run pytest tests/test_mixed_interaction_screening.py -q -k "cat_cat or spline_cat"`
Expected: the pure-kernel pins PASS already (they only exercise shipped
kernels — that is the point: the kernels need no change); the end-to-end
tests FAIL on `NotImplementedError` from Task 3's dispatch.

- [ ] **Step 3: Implement categorical margins in the sweep**

In `src/superglm/model/screening_ops.py`:

1. Raw access for object columns — replace the single `_raw` with:

```python
    raw_x: dict[str, np.ndarray] = {}

    def _raw_numeric(name):
        if name not in raw_x:
            x = frame.column_array(name, dtype=np.float64)
            if not np.all(np.isfinite(x)):
                raise ValueError(
                    f"screen_interactions requires finite covariates; {name!r} "
                    "contains non-finite values"
                )
            raw_x[name] = x
        return raw_x[name]

    def _raw_object(name):
        key = ("obj", name)
        if key not in raw_x:
            raw_x[key] = np.asarray(frame.column_array(name)).ravel()
        return raw_x[key]
```

2. Categorical margin helpers (module level):

```python
def _categorical_codes(spec, x_raw):
    """Dense 0-based codes over ALL fitted levels, in ``spec._levels`` order.

    Applies the same grouping collapse and unseen-level validation the fitted
    spec applies, so the screen sees exactly the mains' level geometry.
    """
    x = np.asarray(x_raw).ravel()
    if spec._grouping is not None:
        x = _grouping_labels(x)
        valid = set(spec._grouping.all_original_levels) | set(spec._grouping.grouped_levels)
        _validate_categorical_levels(x, valid)
        x = pd.Series(x).map(lambda v: spec._grouping.original_to_group.get(v, v)).to_numpy()
    else:
        _validate_categorical_levels(x, set(spec._levels))
    codes = pd.Categorical(x, categories=spec._levels).codes.astype(np.intp)
    if codes.size and codes.min() < 0:
        raise ValueError(
            f"column for {spec!r} contains levels outside the fitted set"
        )
    return codes, len(spec._levels)


def _contrast_menu(spec):
    """(L, L-1) treatment-contrast identity; the base level's row is zero."""
    levels = list(spec._levels)
    menu = np.zeros((len(levels), len(levels) - 1))
    for j, lev in enumerate(spec._non_base):
        menu[levels.index(lev), j] = 1.0
    return menu
```

3. Generalize the support/menu caches per margin kind. `_support(name,
binned)` keeps its shape for spline margins; add:

```python
    def _margin(name, binned):
        """(codes, n, menu, S) for one margin; binned applies to splines only."""
        kind = margin_kinds[name]
        spec = model._specs[name]
        if kind == "categorical":
            key = ("cat", name)
            if key not in support_cache:
                codes, n_levels = _categorical_codes(spec, _raw_object(name))
                support_cache[key] = (codes, n_levels, _contrast_menu(spec), None)
            return support_cache[key]
        s = _support(name, binned)
        menu, S = _one_marginal(name, binned)
        return s["codes"], s["n"], menu, S
```

4. Rework the pair loop's both-gridded arm to use `_margin` for each side and
dispatch by `kind`:

- `ti`: unchanged flow (both margins spline; binning loop as today).
- `spline_cat`: spline margin keeps the binning/width loop; the categorical
  margin never bins and contributes `S=None`. Build
  `S_ti = tensor_penalty(S_spline, np.zeros((menu_cat.shape[1],) * 2))` with
  the spline margin FIRST (order the pair internally as (spline, cat) for
  assembly; report `feature_a`/`feature_b` in the original order).
- `cat_cat`: both margins from `_categorical_codes`; `S_ti = None`.

Budget gates: reuse `_within_budget` with the categorical width `L−1`; the
binnable list must only ever contain spline margins (categorical margins are
excluded by kind, whatever their L).

5. Single-rung evaluation for unpenalized blocks: replace the ladder loop
condition —

```python
        penalized = S_ti is not None and bool(np.any(S_ti))
        for budget in budgets if penalized else budgets[:1]:
```

(`penalized_score_statistic` ignores `edf0` when `S_ti` is falsy and reports
achieved rank + `lambda0=0`.)

6. `approx` and `_pair_refits_discrete`: only spline margins contribute —
categorical margins never bin and never discretize lossily; give
`_pair_refits_discrete` an early `False` when either margin kind is not
`"spline"`... except BOTH-spline pairs, i.e. keep its current body for `ti`
pairs and return `False` for `spline_cat`/`cat_cat` (a `SplineCategorical`
refit discretizes only its spline margin; mark those rows `approx` only when
the spline margin was binned at screen time — the bin_flag mechanism covers
that already).

7. `n_cells` for these kinds: the true grid `n_a_support * L` /
`L1 * L2` (falls out of the existing `n_a * n_b` reporting).

8. Remove the Task 3 `NotImplementedError` arm for `spline_cat`/`cat_cat`
(leave it for `numeric_cat`/`numeric_numeric` until Task 5).

9. Update the `screen_interactions` docstring in `src/superglm/model/api.py`
and the module docstring of `screening_ops.py`: one short paragraph — the
sweep now covers `ti`, `spline_cat`, `numeric_cat`, `cat_cat`,
`numeric_numeric` (last two arriving in this release series), rank by `z`
across kinds, `kind` names the refit constructor family, unpenalized kinds
report achieved rank as `edf0` with `lambda0=0`.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_mixed_interaction_screening.py -q -k "cat" ; uv run pytest tests/test_interaction_screening.py -q`
Expected: PASS (including the untouched spline-only suite).

- [ ] **Step 5: Commit**

```bash
git add src/superglm/model/screening_ops.py src/superglm/model/api.py tests/test_interaction_screening.py tests/test_mixed_interaction_screening.py
git commit -m "Screen categorical margins: cat_cat and spline_cat kinds

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Numeric margins — numeric_cat and numeric_numeric

Numeric margins never grid: five bincount channels over the other margin's
cells (`s`, `s·z`, `w`, `w·z`, `w·z²`) give the exact score, curvature, and
overlap moments for the `z ⊗ contrasts` probe; numeric×numeric reduces to
plain dot products. New focused module + integration.

**Files:**
- Create: `src/superglm/screening/_numeric_margin.py`
- Modify: `src/superglm/screening/__init__.py` (export the two functions)
- Modify: `src/superglm/model/screening_ops.py` (dispatch arms)
- Test: `tests/test_interaction_screening.py` (exactness vs dense reference)
- Test: `tests/test_mixed_interaction_screening.py` (end-to-end)

**Interfaces:**
- Consumes: Task 4's `_margin` provider for the gridded side.
- Produces:
  - `numeric_pair_moments(codes_g, n_g, menu_g, z, score, working_weights) -> tuple[U, V, C, M, u_m]` — probe `menu_g[codes] * z[:, None]`, overlap span `[1 | menu_g[codes] | z]`, shapes `U (k,)`, `V (k, k)`, `C (k + 2, k)`, `M (k + 2, k + 2)`, `u_m (k + 2,)`.
  - `numeric_numeric_moments(z1, z2, score, working_weights) -> tuple[U, V, C, M, u_m]` — probe `z1*z2`, span `[1 | z1 | z2]`, shapes `U (1,)`, `V (1, 1)`, `C (3, 1)`, `M (3, 3)`, `u_m (3,)`.

- [ ] **Step 1: Write the failing exactness tests**

Append to `tests/test_interaction_screening.py`:

```python
def test_numeric_pair_moments_match_dense_assembly():
    from superglm.screening import numeric_pair_moments

    rng = np.random.default_rng(11)
    n, L = 600, 5
    codes = rng.integers(0, L, n)
    menu = np.zeros((L, L - 1)); menu[1:, :] = np.eye(L - 1)
    z = rng.uniform(-2.0, 3.0, n)
    score = rng.normal(size=n)
    w = rng.uniform(0.1, 3.0, n)

    U, V, C, M, u_m = numeric_pair_moments(codes, L, menu, z, score, w)

    X_T = menu[codes] * z[:, None]                      # probe block
    X_o = np.column_stack([np.ones(n), menu[codes], z])  # overlap span
    np.testing.assert_allclose(U, X_T.T @ score, rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(V, X_T.T @ (X_T * w[:, None]), rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(M, X_o.T @ (X_o * w[:, None]), rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(C, X_o.T @ (X_T * w[:, None]), rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(u_m, X_o.T @ score, rtol=1e-12, atol=1e-9)


def test_numeric_numeric_moments_match_dense_assembly():
    from superglm.screening import numeric_numeric_moments

    rng = np.random.default_rng(12)
    n = 500
    z1 = rng.uniform(-1.0, 2.0, n)
    z2 = rng.uniform(0.5, 1.5, n)
    score = rng.normal(size=n)
    w = rng.uniform(0.1, 3.0, n)
    U, V, C, M, u_m = numeric_numeric_moments(z1, z2, score, w)
    X_T = (z1 * z2)[:, None]
    X_o = np.column_stack([np.ones(n), z1, z2])
    np.testing.assert_allclose(U, X_T.T @ score, rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(V, X_T.T @ (X_T * w[:, None]), rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(M, X_o.T @ (X_o * w[:, None]), rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(C, X_o.T @ (X_T * w[:, None]), rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(u_m, X_o.T @ score, rtol=1e-12, atol=1e-9)
```

Append to `tests/test_mixed_interaction_screening.py`:

```python
def test_numeric_cat_planted_slope_ranks_first_with_exact_df():
    df, rng = _mixed_frame(n=20000, seed=5)
    slope = np.where(df["region"] == "D", 0.35, 0.0)
    y = rng.poisson(np.exp(-1.6 + slope * (df["bm"] - 1.0))).astype(np.float64)
    model = _fit_mixed(df, y)
    table = model.screen_interactions(df, y)
    top = table.iloc[0]
    assert {top["feature_a"], top["feature_b"]} == {"bm", "region"}
    assert top["kind"] == "numeric_cat"
    assert top["edf0"] == pytest.approx(3.0, abs=0.26)   # L-1 with L=4
    assert top["n_cells"] == 4
    assert not top["approx"]
    assert top["z"] > 6.0


def test_numeric_numeric_planted_product_ranks_first():
    df, rng = _mixed_frame(n=20000, seed=6)
    df = df.assign(dens=rng.uniform(0.0, 1.0, len(df)))
    y = rng.poisson(
        np.exp(-1.6 + 0.3 * (df["bm"] - 1.25) * (df["dens"] - 0.5))
    ).astype(np.float64)
    model = SuperGLM(
        family="poisson",
        features={"bm": Numeric(), "dens": Numeric()},
    )
    model.fit_reml(df, y)
    table = model.screen_interactions(df, y)
    top = table.iloc[0]
    assert top["kind"] == "numeric_numeric"
    assert {top["feature_a"], top["feature_b"]} == {"bm", "dens"}
    assert top["edf0"] == pytest.approx(1.0, abs=0.01)
    assert top["n_cells"] == 1
    assert top["z"] > 5.0
```

Also delete the `xfail` marker Task 3 placed on
`test_default_sweep_covers_every_eligible_kind` — with all v1 kinds landed it
must pass (OC margins pass through the spline arm already because
`_margin_kind` maps them there; Task 6 pins their specific behavior).

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_interaction_screening.py -q -k numeric ; uv run pytest tests/test_mixed_interaction_screening.py -q`
Expected: FAIL — `ImportError` for the new module, `NotImplementedError` end-to-end.

- [ ] **Step 3: Implement the numeric-margin module**

Create `src/superglm/screening/_numeric_margin.py`:

```python
"""Numeric-margin sufficient statistics for mixed-type screening.

A numeric covariate enters every v1 probe LINEARLY (``z * contrast`` slopes,
``z1 * z2`` products), so the pair needs no joint grid: z-weighted moments
accumulated over the other margin's cells are the complete sufficient
statistics, exact at any cardinality.  Channels: ``s``, ``s*z``, ``w``,
``w*z``, ``w*z**2`` (and the symmetric set for two numerics).  Exactness is
pinned against the dense assembly in tests/test_interaction_screening.py.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def numeric_pair_moments(
    codes_g: NDArray,
    n_g: int,
    menu_g: NDArray,
    z: NDArray,
    score: NDArray,
    working_weights: NDArray,
) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
    """Moments for probe ``menu_g[codes] * z`` with overlap ``[1 | menu | z]``.

    Returns ``(U, V, C, M, u_m)`` with ``k = menu_g.shape[1]`` probe columns
    and overlap width ``q = 1 + k + 1``.
    """
    codes_g = np.asarray(codes_g, dtype=np.intp)
    menu_g = np.asarray(menu_g, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    score = np.asarray(score, dtype=np.float64)
    w = np.asarray(working_weights, dtype=np.float64)
    if not (codes_g.shape == z.shape == score.shape == w.shape):
        raise ValueError("codes, z, score, and working weights must share one row dimension")
    if codes_g.size and (int(codes_g.min()) < 0 or int(codes_g.max()) >= n_g):
        raise ValueError("codes_g fall outside [0, n_g)")

    def cell(v):
        return np.bincount(codes_g, weights=v, minlength=n_g)

    s0, s1 = cell(score), cell(score * z)
    w0, w1, w2 = cell(w), cell(w * z), cell(w * z * z)

    k = menu_g.shape[1]
    q = 1 + k + 1
    U = menu_g.T @ s1
    V = menu_g.T @ (menu_g * w2[:, None])

    M = np.empty((q, q), dtype=np.float64)
    sl = slice(1, 1 + k)
    M[0, 0] = w0.sum()
    M[0, sl] = w0 @ menu_g
    M[0, -1] = w1.sum()
    M[sl, 0] = M[0, sl]
    M[-1, 0] = M[0, -1]
    M[sl, sl] = menu_g.T @ (menu_g * w0[:, None])
    M[sl, -1] = menu_g.T @ w1
    M[-1, sl] = M[sl, -1]
    M[-1, -1] = w2.sum()

    C = np.empty((q, k), dtype=np.float64)
    C[0] = menu_g.T @ w1
    C[sl] = menu_g.T @ (menu_g * w1[:, None])
    C[-1] = menu_g.T @ w2

    u_m = np.empty(q, dtype=np.float64)
    u_m[0] = s0.sum()
    u_m[sl] = menu_g.T @ s0
    u_m[-1] = s1.sum()
    return U, V, C, M, u_m


def numeric_numeric_moments(
    z1: NDArray,
    z2: NDArray,
    score: NDArray,
    working_weights: NDArray,
) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
    """Moments for probe ``z1 * z2`` with overlap span ``[1 | z1 | z2]``."""
    z1 = np.asarray(z1, dtype=np.float64)
    z2 = np.asarray(z2, dtype=np.float64)
    score = np.asarray(score, dtype=np.float64)
    w = np.asarray(working_weights, dtype=np.float64)
    if not (z1.shape == z2.shape == score.shape == w.shape):
        raise ValueError("z1, z2, score, and working weights must share one row dimension")
    p = z1 * z2
    U = np.array([p @ score])
    V = np.array([[(p * p) @ w]])
    ones = np.ones_like(z1)
    span = (ones, z1, z2)
    M = np.array([[ (a * b) @ w for b in span] for a in span])
    C = np.array([[(a * p) @ w] for a in span])
    u_m = np.array([a @ score for a in span])
    return U, V, C, M, u_m
```

Export both from `src/superglm/screening/__init__.py` alongside the existing
names.

- [ ] **Step 4: Wire the dispatch arms**

In `screen_interactions`'s pair loop, replace the remaining
`NotImplementedError` arms:

```python
        if kind == "numeric_numeric":
            z1, z2 = _raw_numeric(feat_a), _raw_numeric(feat_b)
            U, V, C, M, u_m = numeric_numeric_moments(z1, z2, score, working_weights)
            S_ti, n_cells, approx = None, 1, False
        elif kind == "numeric_cat":
            num_name, cat_name = (
                (feat_a, feat_b) if margin_kinds[feat_a] == "numeric" else (feat_b, feat_a)
            )
            codes_g, n_g, menu_g, _ = _margin(cat_name, False)
            U, V, C, M, u_m = numeric_pair_moments(
                codes_g, n_g, menu_g, _raw_numeric(num_name), score, working_weights
            )
            S_ti, n_cells, approx = None, n_g, False
```

then fall through to the SAME statistic/ladder/rows tail the gridded kinds
use (single rung, since `S_ti is None`). Restructure the loop tail so all
kinds share one "evaluate budgets → normalize → append row" block; the
gridded kinds set `(U, V, C, M, u_m, S_ti, n_cells, approx)` from their
existing path.

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_interaction_screening.py -q ; uv run pytest tests/test_mixed_interaction_screening.py -q`
Expected: PASS, including the un-xfailed default-sweep test.

- [ ] **Step 6: Commit**

```bash
git add src/superglm/screening/_numeric_margin.py src/superglm/screening/__init__.py src/superglm/model/screening_ops.py tests/test_interaction_screening.py tests/test_mixed_interaction_screening.py
git commit -m "Screen numeric margins exactly via z-moment channels

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: OrderedCategorical margins in the screen

OC margins ride the spline arm on their mapped scores. What is left: feed the
mapped scores into the support cache, use the inner spline for menus, widths,
`select`, and discretization consultation, and pin behavior end-to-end
(depends on Tasks 1–2 for the confirmatory refit to exist).

**Files:**
- Modify: `src/superglm/model/screening_ops.py`
- Test: `tests/test_mixed_interaction_screening.py`

**Interfaces:**
- Consumes: `resolve_interaction_parent` (Task 1), `_margin`/`_support`/`_one_marginal` (Tasks 3–4).
- Produces: no new symbols — OC margins are spline margins after resolution.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_mixed_interaction_screening.py`:

```python
def test_oc_margin_screens_as_spline_and_confirms():
    df, rng = _mixed_frame(n=20000, seed=8)
    band_idx = df["band"].map({b: i for i, b in enumerate(BANDS)}).to_numpy()
    ramp = (band_idx / 4.0) * (df["power"] - 110.0) / 90.0 * 0.35
    y = rng.poisson(np.exp(-1.5 + ramp)).astype(np.float64)
    model = _fit_mixed(df, y)
    table = model.screen_interactions(df, y)
    top = table.iloc[0]
    assert {top["feature_a"], top["feature_b"]} == {"band", "power"}
    assert top["kind"] == "ti"
    assert top["z"] > 6.0
    confirm = _fit_mixed(df, y, interactions=[("band", "power")])
    assert model._result.deviance - confirm._result.deviance > 50.0


def test_oc_cat_pair_is_spline_cat_kind():
    df, rng = _mixed_frame(n=8000, seed=9)
    y = _null_y(df, rng)
    model = _fit_mixed(df, y)
    table = model.screen_interactions(df, y, candidates=[("band", "region")])
    assert list(table["kind"]) == ["spline_cat"]
    assert np.isfinite(table["z"]).all()
    # 5 score points x 4 levels
    assert int(table["n_cells"].iloc[0]) == 5 * 4


def test_oc_cat_planted_deviation_confirms():
    df, rng = _mixed_frame(n=20000, seed=13)
    band_idx = df["band"].map({b: i for i, b in enumerate(BANDS)}).to_numpy()
    bend = np.where(df["region"] == "A", (band_idx / 4.0 - 0.5) * 0.5, 0.0)
    y = rng.poisson(np.exp(-1.4 + bend)).astype(np.float64)
    model = _fit_mixed(df, y)
    table = model.screen_interactions(df, y)
    top = table.iloc[0]
    assert {top["feature_a"], top["feature_b"]} == {"band", "region"}
    assert top["kind"] == "spline_cat"
    assert top["z"] > 6.0
    confirm = _fit_mixed(df, y, interactions=[("band", "region")])
    assert model._result.deviance - confirm._result.deviance > 30.0


def test_oc_select_inner_spline_raises_upfront():
    df, rng = _mixed_frame(n=4000, seed=10)
    y = _null_y(df, rng)
    model = SuperGLM(
        family="poisson",
        features={
            "band": OrderedCategorical(
                order=BANDS, basis=Spline(kind="ps", n_knots=4, select=True)
            ),
            "power": Spline(kind="ps", n_knots=5),
        },
    )
    model.fit_reml(df, y)
    with pytest.raises(ValueError, match="select"):
        model.screen_interactions(df, y, candidates=[("band", "power")])
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_mixed_interaction_screening.py -q -k oc`
Expected: FAIL — the support path calls `frame.column_array(name,
dtype=np.float64)` on label data (`ValueError`/`TypeError`), or menus built
from the OC wrapper rather than the inner spline.

- [ ] **Step 3: Route OC margins through resolution**

In `screen_interactions`:

1. Add an effective-margin resolution next to the caches:

```python
    from superglm.features.ordered_categorical import resolve_interaction_parent

    def _margin_source(name):
        """(effective_spec, x_values) for a spline-kind margin."""
        spec = model._specs[name]
        if isinstance(spec, OrderedCategorical):
            eff_spec, x = resolve_interaction_parent(spec, _raw_object(name))
            if not np.all(np.isfinite(x)):
                raise ValueError(
                    f"screen_interactions requires finite covariates; {name!r} "
                    "maps to non-finite scores"
                )
            return eff_spec, x
        return spec, _raw_numeric(name)
```

2. `_support(name, binned)` uses `_margin_source(name)[1]` instead of
`_raw(name)`; `_one_marginal` and `_marginal_width_estimate` use
`_margin_source(name)[0]` (the inner spline) when building menus —
`TensorInteraction._marginal_from_spec(eff_spec, x, None, support=..., counts=...)`.
The `_select_flag` helper from Task 3 already consults the inner spline.

3. `_pair_refits_discrete` consults the effective spec:
`should_discretize(eff_spec, model_discrete)` and
`resolve_discrete_n_bins(name, eff_spec, n_bins_config)` — but per Task 1's
gate, OC-parented refits never discretize, so short-circuit: if either spec
is `OrderedCategorical`, return `False` (matches
`should_discretize_tensor_interaction`).

4. The prefetch loop from Task 3 prefetches `_margin_source` for spline
margins and `_raw_object` for categorical margins, so all input validation
still happens before any statistics.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_mixed_interaction_screening.py -q ; uv run pytest tests/test_ordered_categorical_interactions.py -q ; uv run pytest tests/test_interaction_screening.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/superglm/model/screening_ops.py tests/test_mixed_interaction_screening.py
git commit -m "Screen OrderedCategorical margins on their mapped scores

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: Per-kind null gauntlet, measured floors, release pins

Re-run the null methodology per kind, record the measured floors for the
docs, and pin generous bounds as release gates. The measurement script lives
in `benchmarks/` (not `scripts/`, which is release tooling).

**Files:**
- Create: `benchmarks/screening_null_floors.py`
- Test: `tests/test_mixed_interaction_screening.py` (bounded-null pins)

**Interfaces:**
- Consumes: the full mixed sweep from Tasks 3–6.
- Produces: a printed per-kind `max |z|` table (used by Task 8's docs) and pinned null bounds.

- [ ] **Step 1: Write the bounded-null release pins (failing only if the machinery is miscalibrated)**

Append to `tests/test_mixed_interaction_screening.py`:

```python
@pytest.mark.parametrize("seed", range(4))
def test_mixed_null_z_stays_bounded_poisson(seed):
    df, rng = _mixed_frame(n=8000, seed=100 + seed)
    y = _null_y(df, rng)
    model = _fit_mixed(df, y)
    table = model.screen_interactions(df, y)
    finite = table["z"][np.isfinite(table["z"])]
    assert (finite < 10.0).all()


@pytest.mark.parametrize("seed", range(4))
def test_mixed_null_z_stays_bounded_dispersed_gaussian(seed):
    df, rng = _mixed_frame(n=8000, seed=200 + seed)
    y = rng.normal(loc=1.0 + 0.01 * df["age"], scale=10.0, size=len(df))
    model = SuperGLM(
        family="gaussian",
        features={
            "age": Spline(kind="ps", n_knots=6),
            "region": Categorical(),
            "brand": Categorical(),
            "bm": Numeric(),
        },
    )
    model.fit_reml(df, y)
    table = model.screen_interactions(df, y)
    finite = table["z"][np.isfinite(table["z"])]
    assert (finite < 10.0).all()
```

The 10.0 bound mirrors the shipped release-gate reading ("treat the
release-gate bound of 10 as generous") — these tests are gates, not floor
measurements.

- [ ] **Step 2: Run the pins**

Run: `uv run pytest tests/test_mixed_interaction_screening.py -q -k null`
Expected: PASS. If any kind breaches 10 on a pure null, STOP and debug that
kind's moments (an exactness bug, not a tuning problem — the dense-assembly
pins from Tasks 4–5 are where to look first).

- [ ] **Step 3: Add the real-book end-to-end sanity test (skips when data absent)**

Append to `tests/test_mixed_interaction_screening.py` (top of file gains
`from . import _datasets` next to the other imports):

```python
FREQ_SKIP = pytest.mark.skipif(
    _datasets.load_freq() is None, reason="freMTPL2freq.parquet not available"
)


def _fremtpl_features():
    return {
        "DrivAge": Spline(kind="ps", n_knots=8),
        "VehAge": Spline(kind="ps", n_knots=6),
        "BonusMalus": Numeric(),
        "VehBrand": Categorical(),
        "Region": Categorical(),
    }


@FREQ_SKIP
def test_fremtpl_mixed_sweep_end_to_end():
    df = _datasets.load_freq().sample(80_000, random_state=0).reset_index(drop=True)
    y = df["ClaimNb"].to_numpy(dtype=np.float64)
    exposure = df["Exposure"].to_numpy(dtype=np.float64)
    model = SuperGLM(family="poisson", features=_fremtpl_features())
    model.fit_reml(df, y, sample_weight=exposure)
    table = model.screen_interactions(df, y, sample_weight=exposure)
    # every v1 kind this feature set can produce shows up and computes
    assert {"ti", "spline_cat", "numeric_cat", "cat_cat"} <= set(table["kind"])
    assert np.isfinite(table["z"]).any()
    # the queue is workable on a real book: the top pair refits and improves
    top = table.iloc[0]
    confirm = SuperGLM(
        family="poisson",
        features=_fremtpl_features(),
        interactions=[(top["feature_a"], top["feature_b"])],
    )
    confirm.fit_reml(df, y, sample_weight=exposure)
    assert confirm._result.deviance < model._result.deviance
```

Run: `uv run pytest tests/test_mixed_interaction_screening.py -q -k fremtpl`
Expected: PASS locally (the parquet is in `data/`); SKIP on machines without it.

- [ ] **Step 4: Write the floor-measurement benchmark**

Create `benchmarks/screening_null_floors.py`:

```python
"""Measure per-kind null z floors for the screening docs.

Run:  uv run python benchmarks/screening_null_floors.py [--seeds 40]

Prints max |z| per screening kind over a battery of null datasets spanning
families (Poisson, Bernoulli-like binomial, gamma, dispersed Gaussian),
correlated parents, exposure spread, and rare-level factors.  The maxima go
into docs/guide/screening.md verbatim; they are documentation of measured
noise floors, not calibrated quantiles.
"""

import argparse

import numpy as np
import pandas as pd

from superglm import Categorical, SuperGLM
from superglm.features.numeric import Numeric
from superglm.features.ordered_categorical import OrderedCategorical
from superglm.features.spline import Spline

BANDS = ["18-25", "26-35", "36-45", "46-55", "56+"]


def _frame(n, rng):
    region_p = np.array([0.55, 0.25, 0.15, 0.05])  # includes a rare level
    age = rng.uniform(18.0, 80.0, n)
    df = pd.DataFrame(
        {
            "age": age,
            # correlated continuous parent
            "power": np.clip(1.5 * age + rng.normal(0.0, 25.0, n), 20.0, 220.0),
            "region": rng.choice(list("ABCD"), n, p=region_p),
            "brand": rng.choice(["B1", "B2", "B3"], n),
            "bm": rng.uniform(0.5, 2.0, n),
            "band": rng.choice(BANDS, n),
        }
    )
    exposure = rng.uniform(0.05, 1.0, n)
    return df, exposure


def _features():
    return {
        "age": Spline(kind="ps", n_knots=6),
        "power": Spline(kind="ps", n_knots=6),
        "region": Categorical(),
        "brand": Categorical(),
        "bm": Numeric(),
        "band": OrderedCategorical(order=BANDS, basis=Spline(kind="ps", n_knots=4)),
    }


def _null_response(df, exposure, family, rng):
    eta = -1.5 + 0.004 * df["age"] + 0.1 * (df["region"] == "B")
    if family == "poisson":
        return rng.poisson(exposure * np.exp(eta)).astype(np.float64), exposure
    if family == "gamma":
        mu = np.exp(eta)
        return rng.gamma(2.0, mu / 2.0), None
    if family == "binomial":
        p = 1.0 / (1.0 + np.exp(-eta))
        return rng.binomial(1, p).astype(np.float64), None
    mu = eta
    return rng.normal(mu, 10.0), None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=40)
    parser.add_argument("--n", type=int, default=8000)
    args = parser.parse_args()

    maxima: dict[str, float] = {}
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        df, exposure = _frame(args.n, rng)
        for family in ("poisson", "gamma", "binomial", "gaussian"):
            y, w = _null_response(df, exposure, family, rng)
            model = SuperGLM(family=family, features=_features())
            try:
                model.fit_reml(df, y, sample_weight=w)
                table = model.screen_interactions(df, y, sample_weight=w)
            except Exception as err:  # a failed null fit is data, not a crash
                print(f"seed={seed} family={family}: skipped ({err})")
                continue
            for kind, group in table.groupby("kind"):
                z = group["z"][np.isfinite(group["z"])]
                if len(z):
                    maxima[kind] = max(maxima.get(kind, -np.inf), float(z.max()))
    print("\nmax null z per kind over the battery:")
    for kind in sorted(maxima):
        print(f"  {kind:16s} {maxima[kind]:6.2f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the benchmark and record the numbers**

Run: `uv run python benchmarks/screening_null_floors.py --seeds 40`
Expected: a table of per-kind maxima, each comfortably under 10. Copy the
printed maxima into the Task 8 docs work (and paste the table into the plan's
execution notes for the record). If a kind's floor lands above ~6, say so
explicitly in the docs floor table — do not round it down.

- [ ] **Step 6: Commit**

```bash
git add benchmarks/screening_null_floors.py tests/test_mixed_interaction_screening.py
git commit -m "Pin per-kind null bounds, real-book sanity, and the floor battery

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 8: Documentation

Rewrite the screening guide for mixed kinds with the measured floors, and
document OC as an interaction parent. Run the API-doc checker.

**Files:**
- Modify: `docs/guide/screening.md`
- Modify: `docs/guide/interactions.md`

**Interfaces:**
- Consumes: Task 7's measured per-kind floors (from the benchmark output recorded in the execution notes).

- [ ] **Step 1: Update `docs/guide/screening.md`**

Keep the existing structure and voice; changes:

1. Intro: "ranks every candidate pair of fitted spline features" becomes a
   short list of the five kinds with one-line geometry each, and states that
   OC (spline mode) margins screen on their mapped scores.
2. Add a "Pair kinds" table mirroring the spec's (kind, probe, penalty, df,
   refit target) — including the note that `spline_cat` rows can also be
   confirmed as `FactorSmooth` when pooling is wanted.
3. "Reading the output": add `kind`; state that `statistic/edf0/lambda0`
   describe the winning rung for penalized kinds and the achieved rank with
   `lambda0=0` for unpenalized kinds; keep "rank by z, and only z"; add the
   per-kind floor table with the MEASURED numbers from Task 7 and a sentence
   that low-df kinds carry heavier null tails so their floors differ.
4. "Measured limits": add that categorical margins never bin and numeric
   margins are exact at any cardinality (no `approx` from those margins);
   rare-level cells screen weak rather than false-positive.
5. Provenance: extend with one sentence each — classical Rao score tests for
   the unpenalized kinds; the penalized-smooth score-test line (Lin 1997;
   Zhang & Lin 2003); ordered-factor scores (Graubard & Korn 1987;
   Gertheiss & Tutz 2009; Azzalini 2023/2024); complementary reluctant
   interaction inference (Yu et al. 2019; Huang et al. 2025). No R source
   references.

- [ ] **Step 2: Update `docs/guide/interactions.md`**

In "Auto-detected interaction types", add a sentence under the table:
spline-mode `OrderedCategorical` parents participate as splines on their
mapped level scores (so OC+Categorical builds `SplineCategorical`, OC+Spline
builds `TensorInteraction`); step-mode OC parents are rejected. Mention that
OC-parented tensors always use the exact prediction path.

- [ ] **Step 3: Check docs and run the full suite**

Run: `uv run python scripts/check_api_docs.py && uv run pytest tests/ -q`
Expected: both clean.

- [ ] **Step 4: Commit**

```bash
git add docs/guide/screening.md docs/guide/interactions.md
git commit -m "Document mixed-kind screening and OC interaction parents

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Execution notes

- Tasks 1→2 and 3→4→5→6 are strict sequences; Task 3 can start in parallel
  with Task 2 (different files) but merge order must keep the suite green.
- Task 7 depends on all kinds; Task 8 depends on Task 7's measured numbers.
- If a planted-signal threshold in a test proves seed-fragile, loosen the
  z threshold before touching the statistics — the dense-assembly pins are
  the correctness authority, the end-to-end thresholds are smoke.
- Append the Task 7 benchmark table and any disposition notes to this plan
  file at completion (house style: the narrative, not the checkboxes, is the
  record).

### Task 7 — measured null floors

> **SUPERSEDED 2026-08-01 — the tables and dispositions in this subsection were
> measured OFF the exposure contract and are kept only as the record.** The
> benchmark's Poisson arm passed claim COUNTS as `y` alongside
> `sample_weight=exposure`, where the library's contract is
> `Var(y) = phi * V(mu) / w` — i.e. an exposure-weighted `y` is a RATE. That
> under-estimated `phi` on the Poisson arm and inflated every Poisson `z` in
> the battery. See "Task 7 (round 2) — measured null floors, on-contract"
> below for the numbers that hold.

`uv run python benchmarks/screening_null_floors.py --seeds 40` (n=8000,
4 families x 40 seeds = 160 fits, 3520 screened rows, 2m10s wall):

```
max null z per kind over the battery:
  kind              rows   max z  max|z|  mean z   p90 z     probe df  approx
  cat_cat            480    7.23    7.23    0.12    1.46   2.0-  6.0      0%
  numeric_cat        960    7.64    7.64    0.09    1.44   1.0-  3.0      0%
  numeric_numeric    160    4.91    4.91   -0.01    1.14   1.0-  1.0      0%
  spline_cat        1440    7.11    7.11    0.66    2.25   2.0- 16.0      0%
  ti                 480    7.31    7.31    0.67    2.18   2.0- 16.0     33%

null tail by exact probe df (unpenalized kinds; low df = heavier tail):
  kind                df  rows   max z  max|z|   p90 z
  numeric_numeric    1.0   160    4.91    4.91    1.14
  numeric_cat        1.0   320    7.64    7.64    1.06
  numeric_cat        2.0   320    4.45    4.45    1.42
  numeric_cat        3.0   320    5.71    5.71    1.83
  cat_cat            2.0   160    7.23    7.23    1.45
  cat_cat            3.0   160    5.71    5.71    1.31
  cat_cat            6.0   160    3.34    3.34    1.57

ordered-categorical margins vs plain spline margins (spline kinds only):
  kind         margins     rows   max z  max|z|  mean z   p90 z     cells
  ti           plain        160    7.31    7.31    0.58    1.89   1899008
  ti           oc           320    5.40    5.40    0.72    2.27     38697
  spline_cat   plain        960    5.69    5.69    0.71    2.28     23218
  spline_cat   oc           480    7.11    7.11    0.54    2.03        15

max null z by family (is the floor one family's artifact?):
  kind              poisson    gamma binomial gaussian
  cat_cat              7.23     3.20     2.90     3.18
  numeric_cat          7.64     4.66     4.17     7.53
  numeric_numeric      4.84     3.40     2.87     4.91
  spline_cat           7.11     5.34     5.53     4.97
  ti                   5.40     3.09     3.49     7.31

the 6 largest single rows in the battery:
  z= 7.64  numeric_cat      fuel x dens            seed=27  family=poisson   edf0=1.0
  z= 7.53  numeric_cat      fuel x bm              seed=15  family=gaussian  edf0=1.0
  z= 7.31  ti               age x power            seed=37  family=gaussian  edf0=2.0
  z= 7.23  cat_cat          brand x fuel           seed=6   family=poisson   edf0=2.0
  z= 7.11  spline_cat       fuel x band            seed=10  family=poisson   edf0=2.0
  z= 6.21  spline_cat       region x band          seed=13  family=poisson   edf0=12.0

diagnostics:
  fits attempted            160
  fits that failed          0
  rows screened             3520
  non-finite z rows         0   (refusals or degenerate margins)
  widest numeric_cat factor 4 levels; the gate (L+2)^2 <= max_cells=5000000 admits L <= 2234
  warnings raised           0 in 0 distinct forms

  (n=8000 rows per dataset, 4 families x 40 seeds)
```

Dispositions for Task 8's docs (SUPERSEDED — see round 2 below):

- **The floor is ~7.6, not the ~4.5 `docs/guide/screening.md` currently
  claims.** The old figure predates the mixed kinds and came from a far
  smaller battery; a maximum grows with the number of draws, so this is a
  sample-size correction, not a regression. Four of five kinds land above
  6, which the plan already said must be stated rather than rounded down.
  "Treat `z` below 4-5 as noise-level" needs to become a per-df reading.
- **The tail is a function of the probe's df, not of the kind.** Five of the
  six largest rows sit at `edf0 <= 2`; within `cat_cat` the maximum falls
  monotonically with df (7.23 / 5.71 / 3.34 at df 2 / 3 / 6). Rank a 1-df
  `numeric_cat` on a 2-level factor against a 16-df `ti` and the low-df row
  wins on noise alone. Do NOT phrase this as "`numeric_numeric` is the
  heavy kind": it measured the *lowest* maximum of any kind (4.91), because
  it contributes one pair per sweep and a max over 160 draws is smaller
  than a max over 320 of the same distribution.
- **OC margins do not move the floor.** Against the plain spline margins of
  the same kind the four bulk gaps are: `spline_cat` mean 0.17 and p90 0.25,
  `ti` mean 0.14 and p90 0.38 — within 0.2 on the means and within 0.4 on
  the p90s. Every one of those gaps flips sign between the two kinds (OC is
  *lower* on both `spline_cat` statistics, *higher* on both `ti` ones), as
  do the maxima (`spline_cat` OC 7.11 vs plain 5.69, `ti` OC 5.40 vs plain
  7.31). A difference with no consistent direction across kinds is
  sampling noise, not a 5-point grid inflating anything.
- **No family drives the floor.** Poisson tops three kinds and Gaussian the
  other two; gamma and binomial stay milder (2.9-5.5) throughout.
- **The `numeric_cat` budget gate never fires here.** `(L+2)^2 <=
  max_cells` admits L <= 2234 at the default; the widest factor in the
  battery is 4 levels. The "policy, not law" flag from Task 5 stands
  unrevised — nothing in the measured-floors pass touched it.
- **The release pins do not cover every measured configuration — do not
  describe them as gating all kinds.** Both pin models run on frames with no
  2-level factor and no second `Numeric`, so what they actually bound is:
  `ti` (Poisson pin only), `spline_cat`, `numeric_cat` at df 2 and 3, and
  `cat_cat` at df 6 alone. The three heaviest-tailed configurations the
  battery measured are therefore measured but **not** gated — `numeric_cat`
  at df=1 (7.64, the largest row in the battery), `cat_cat` at df=2 (7.23)
  and df=3 (5.71), and the `numeric_numeric` kind in its entirety. The
  benchmark carries a 2-level factor and a second `Numeric` precisely
  because the pins do not.
- **Caution the release gate's headroom.** The pins bound `z < 10` and the
  battery reached 7.64 over 3520 rows. The floor rises with sweep width, so
  a wide book screened in one pass draws more null rows than this whole
  battery; 10 is generous for a handful of pairs and thinner for hundreds.

### Task 7 (round 2) — measured null floors, on-contract

**2026-08-01, final whole-branch review.** The round-1 battery above was
measured off the exposure contract: `benchmarks/screening_null_floors.py`'s
Poisson arm returned `rng.poisson(exposure * exp(eta))` — claim COUNTS — as
`y` while passing `sample_weight=exposure`. Under `Var(y) = phi * V(mu) / w`
an exposure-weighted response is a RATE, so the counts form under-estimated
`phi` and inflated every Poisson `z` in the battery, worst at high df. The arm
now returns `counts / exposure` with the same weight; nothing else about the
battery changed.

Measured on the 80k freMTPL2 sample the guide's worked example uses, with the
same feature set: the off-contract form (counts as `y`, `sample_weight` the
exposure) estimates `phi = 0.5557`, while BOTH on-contract forms — the rate
form (`y = ClaimNb/Exposure`, `sample_weight` the exposure) and the offset
form (`y = ClaimNb`, `offset = log(Exposure)`) — estimate `phi = 2.5193`, and
they agree with each other to ten significant figures. The off-contract form
was low by a factor of 4.53. The statistic is reported on the `T / phi` scale,
so that factor multiplies it directly, and `z = (T/phi - edf0)/sqrt(2*edf0)`
carries the excess through — worst at high `edf0`, where a 4.53x multiple of
an `edf0`-sized statistic is a large absolute shift. Same command, same seeds:

```
max null z per kind over the battery:
  kind              rows   max z  max|z|  mean z   p90 z     probe df  approx
  cat_cat            480    3.98    3.98   -0.01    1.20   2.0-  6.0      0%
  numeric_cat        960    7.53    7.53    0.00    1.19   1.0-  3.0      0%
  numeric_numeric    160    4.91    4.91   -0.07    1.07   1.0-  1.0      0%
  spline_cat        1440    5.53    5.53    0.42    1.82   2.0- 16.0      0%
  ti                 480    7.31    7.31    0.42    1.56   2.0- 16.0     33%

null tail by exact probe df (unpenalized kinds; low df = heavier tail):
  kind                df  rows   max z  max|z|   p90 z
  numeric_numeric    1.0   160    4.91    4.91    1.07
  numeric_cat        1.0   320    7.53    7.53    0.83
  numeric_cat        2.0   320    4.45    4.45    1.13
  numeric_cat        3.0   320    4.78    4.78    1.62
  cat_cat            2.0   160    3.18    3.18    1.17
  cat_cat            3.0   160    3.98    3.98    1.28
  cat_cat            6.0   160    2.90    2.90    1.14

ordered-categorical margins vs plain spline margins (spline kinds only):
  kind         margins     rows   max z  max|z|  mean z   p90 z     cells
  ti           plain        160    7.31    7.31    0.35    1.49   1899008
  ti           oc           320    4.25    4.25    0.45    1.58     38697
  spline_cat   plain        960    5.34    5.34    0.46    1.80     23218
  spline_cat   oc           480    5.53    5.53    0.35    1.85        15

max null z by family (is the floor one family's artifact?):
  kind              poisson    gamma binomial gaussian
  cat_cat              3.98     3.20     2.90     3.18
  numeric_cat          3.17     4.66     4.17     7.53
  numeric_numeric      3.74     3.40     2.87     4.91
  spline_cat           4.08     5.34     5.53     4.97
  ti                   3.56     3.09     3.49     7.31

the 6 largest single rows in the battery:
  z= 7.53  numeric_cat      fuel x bm              seed=15  family=gaussian  edf0=1.0
  z= 7.31  ti               age x power            seed=37  family=gaussian  edf0=2.0
  z= 5.53  spline_cat       region x band          seed=15  family=binomial  edf0=3.0
  z= 5.34  spline_cat       age x brand            seed=12  family=gamma     edf0=2.0
  z= 4.97  spline_cat       age x fuel             seed=3   family=gaussian  edf0=2.0
  z= 4.93  spline_cat       brand x band           seed=12  family=gaussian  edf0=2.0

diagnostics:
  fits attempted            160
  fits that failed          0
  rows screened             3520
  non-finite z rows         0   (refusals or degenerate margins)
  widest numeric_cat factor 4 levels; the gate (L+1)^2 <= max_cells=5000000 admits L <= 2235
  warnings raised           0 in 0 distinct forms

  (n=8000 rows per dataset, 4 families x 40 seeds)
```

Dispositions, revised against the on-contract numbers:

- **The floor is 7.53, and only two of five kinds clear 6** (`numeric_cat`
  7.53 and `ti` 7.31; `spline_cat` 5.53, `numeric_numeric` 4.91, `cat_cat`
  3.98). Round 1's "four of five above 6" was the Poisson inflation: the
  Poisson column fell from 7.23/7.64/4.84/7.11/5.40 to 3.98/3.17/3.74/4.08/
  3.56 once `phi` was estimated on-contract. The headline correction against
  the pre-branch guide still stands — the old "never exceeded ~4.5" came from
  a smaller splines-only battery, and 7.53 supersedes it.
- **The df story survives, but only at the extreme, and it is not monotone.**
  All six of the largest rows sit at `edf0 <= 3` and the largest of all is the
  1-df `numeric_cat`, 2.6 clear of every other configuration in the by-df
  table. But neither kind is monotone now (`numeric_cat` 7.53/4.45/4.78 at df
  1/2/3; `cat_cat` 3.18/3.98/2.90 at df 2/3/6), so round 1's "within `cat_cat`
  the maximum falls monotonically with df" is retired. Still do NOT phrase
  this as "`numeric_numeric` is the heavy kind": at 4.91 it sits below the
  `numeric_cat` maximum on the same 1 df, on the fewest draws.
- **One family DOES carry the floor now, and it is the dispersed Gaussian.**
  Round 1's "no family drives the floor" is retired: Gaussian tops three of
  five kinds and holds both maxima above 6, while no other family reached
  beyond 5.53 anywhere (Poisson at most 4.08, gamma 5.34, binomial 5.53).
  Quote the maxima as Gaussian-driven; the per-kind floor, being a max over
  all four families, stays the conservative reading for any single one.
- **OC margins still do not move the floor**, on smaller gaps than round 1
  measured: within 0.11 on the means and 0.09 on the p90s, with the means
  flipping sign between kinds (OC lower on `spline_cat`, higher on `ti`) and
  the maxima disagreeing with the means on both (`spline_cat` OC 5.53 vs plain
  5.34; `ti` OC 4.25 vs plain 7.31). The p90s are the one statistic where OC
  is above plain on both kinds, by 0.05 and 0.09 against p90s of 1.5-1.9 —
  small enough to read as sampling noise, and stated rather than dropped.
- **The `numeric_cat` budget gate never fires here** (unchanged), but the
  diagnostic that reported it was wrong and is fixed: the shipped gate is
  `(k_g + 2)^2 <= max_cells` on the gridded margin's contrast count, i.e.
  `(L + 1)^2` in level terms, admitting `L <= 2235` — not the `(L + 2)^2` /
  2234 the benchmark printed. `docs/guide/screening.md` already carried the
  corrected form; the benchmark now mirrors it.
- **The release pins now cover every measured configuration.** The Gaussian
  bounded-null pin gained a second `Numeric` (`dens`) and a 2-level factor
  (`fuel`), so the union of the two pins covers `spline_cat`, `cat_cat` at df
  2/3/6, `numeric_cat` at df 1/2/3 and `numeric_numeric` — with `ti` still in
  the Poisson pin only. Round 1's "the three heaviest-tailed configurations
  are measured but not gated" is therefore retired. Measured over the pins'
  own 8 sweeps: Poisson max 4.27, Gaussian max 4.58, against the `z < 10`
  bound.
- **The gate's headroom caution stands** on the new number: the pins bound
  `z < 10` and the battery reached 7.53 over 3520 rows. A floor is a maximum,
  so it grows with sweep width — 10 is generous for a handful of pairs and
  thinner for hundreds.
