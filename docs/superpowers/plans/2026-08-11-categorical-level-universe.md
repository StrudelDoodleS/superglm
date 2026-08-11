# Bound Level Universes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. This plan is executed via the Workflow orchestrator: one agent per task, strictly sequential, each task commits.

**Goal:** Give every categorical-family term a bound level universe (declared / dtype / full-frame CV bind), pin declared-but-empty levels to base instead of erroring, and add an `unseen="error"|"base"` predict policy — per the approved spec `docs/superpowers/specs/2026-08-11-categorical-level-universe-design.md`.

**Architecture:** Universe state lives on the term spec (resolved at `__init__` or adopted at build); a new `LevelBinding` value flows from `cross_validate` through `ModelConfig.level_bindings` → `materialize` → `build_design_matrix`, which applies it via two `hasattr`-guarded spec hooks (`adopt_dtype_categories`, `apply_level_binding`) so no `build()` signature changes. The kernel is untouched: empty/unseen levels route to the existing `-1` sink of `CategoricalGroupMatrix`.

**Tech Stack:** Python 3.10+, numpy, pandas, narwhals (polars optional), pytest.

## Global Constraints

- Branch: `categorical-level-universe` (already exists, tracks origin/master post-0.25.0). Commit after every task; never push from a task agent.
- **No version bump, no `pyproject.toml` changes** (release convention: bump at release).
- Default-path behavior must not change: plain `fit` on a raw object column with no `levels=`, no categorical dtype, no binding = today's exact behavior (per-fit inference, same predict-time unseen `ValueError` message).
- All new warnings are `UserWarning` via `warnings.warn`, once per fit/predict call, never per row; messages name the feature (`context`), the levels (`sorted(..., key=str)`), and the fix.
- NaN/None in a data column stays a hard error everywhere; NaN in a `levels=` source is a hard error.
- Follow existing idioms: local `import pandas as pd` inside methods in `categorical.py`; `key=str` sorts in error paths; O(n) bincount over dict loops; comment density of the surrounding file (constraint-comments only).
- Run targeted tests with `python -m pytest <file>::<test> -x -q`; a task is done only when its listed tests pass AND `python -m pytest tests/test_categorical_level_validation.py tests/test_categorical_ux.py -q` stays green (existing contract surface).
- GPL sources (mgcv/scam/CRAN) must not be opened. sklearn/pandas/patsy source and docs are fine.

---

### Task 1: Level-source resolution helper

**Files:**
- Create: `src/superglm/features/_level_source.py`
- Test: `tests/test_level_source.py` (new)

**Interfaces:**
- Produces: `resolve_level_source(source, *, context: str = "") -> list` — the single normalization used by every term that accepts `levels=`. Also `LEVEL_SOURCE_KINDS: tuple[str, ...] = ("declared",)` is NOT needed — source-kind strings are owned by the term (Task 3).
- Consumes: nothing from other tasks.

Semantics (spec §3.1):
- `list`/`tuple` → exactly those labels, order preserved.
- `pd.Series`/`np.ndarray`/polars `Series`: if the pandas dtype is `CategoricalDtype` → `dtype.categories.tolist()` (dtype order); if a polars/narwhals `Enum`-typed Series → its dtype categories; otherwise → sorted observed uniques (`sorted(pd.unique(values).tolist(), key=str)` — `key=str` because mixed int/str labels have no natural order).
- `pd.CategoricalDtype` → `.categories.tolist()`.
- NaN/None anywhere in the resolved labels → `ValueError` ("a level cannot be a missing value").
- Duplicate labels after resolution → `ValueError` naming the duplicates.
- Fewer than 2 labels → `ValueError` (mirrors the existing `>= 2 levels` build check).
- A fitted sklearn encoder (object with a `categories_` attribute) → `TypeError` whose message says: pass `encoder.categories_[0]` for the single feature instead.
- Anything else → `TypeError` naming the accepted shapes.

- [ ] **Step 1: Write the failing tests**

```python
"""tests/test_level_source.py"""
import numpy as np
import pandas as pd
import pytest

from superglm.features._level_source import resolve_level_source


def test_list_preserves_order():
    assert resolve_level_source(["b", "a", "c"]) == ["b", "a", "c"]


def test_tuple_preserves_order():
    assert resolve_level_source(("z", "y")) == ["z", "y"]


def test_object_series_sorted_uniques():
    s = pd.Series(["b", "a", "b", "c"])
    assert resolve_level_source(s) == ["a", "b", "c"]


def test_numpy_array_sorted_uniques():
    assert resolve_level_source(np.array(["b", "a", "b"])) == ["a", "b"]


def test_mixed_type_array_sorts_by_str():
    # int beside str must not crash the sort
    out = resolve_level_source(np.array([2, "MISSING", 1], dtype=object))
    assert set(out) == {1, 2, "MISSING"} and len(out) == 3


def test_categorical_series_uses_dtype_categories_and_order():
    s = pd.Series(pd.Categorical(["a", "b"], categories=["c", "b", "a"]))
    assert resolve_level_source(s) == ["c", "b", "a"]  # declared-but-unobserved 'c' kept


def test_categorical_dtype_direct():
    dt = pd.CategoricalDtype(["x", "y", "z"])
    assert resolve_level_source(dt) == ["x", "y", "z"]


def test_nan_in_source_raises():
    with pytest.raises(ValueError, match="missing value"):
        resolve_level_source(pd.Series(["a", None, "b"]))


def test_duplicate_labels_raise():
    with pytest.raises(ValueError, match="duplicate"):
        resolve_level_source(["a", "a", "b"])


def test_singleton_raises():
    with pytest.raises(ValueError, match=">= 2"):
        resolve_level_source(["only"])


def test_fitted_encoder_rejected_with_guidance():
    class FakeEncoder:
        categories_ = [np.array(["a", "b"])]

    with pytest.raises(TypeError, match=r"categories_\[0\]"):
        resolve_level_source(FakeEncoder())


def test_unsupported_type_rejected():
    with pytest.raises(TypeError, match="levels"):
        resolve_level_source(42)


def test_context_prefixes_errors():
    with pytest.raises(ValueError, match=r"\[vehicle_group\]"):
        resolve_level_source(["a", "a"], context="vehicle_group")


def test_polars_enum_series_uses_declared_categories():
    pl = pytest.importorskip("polars")
    s = pl.Series("g", ["a", "b"], dtype=pl.Enum(["c", "b", "a"]))
    assert resolve_level_source(s) == ["c", "b", "a"]
```

- [ ] **Step 2: Run to verify failure** — `python -m pytest tests/test_level_source.py -x -q`. Expected: ImportError (module does not exist).

- [ ] **Step 3: Implement `src/superglm/features/_level_source.py`**

```python
"""Normalize a user-supplied level universe into a plain list of labels.

Accepted shapes are deliberately exactly three (spec 2026-08-11, §3.1):
an explicit sequence, a data column (Series/array), or a CategoricalDtype.
Encoder objects are rejected with the one-line recipe instead of being
half-supported.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _fail(msg: str, context: str, exc: type[Exception] = ValueError) -> None:
    raise exc(f"[{context}] {msg}" if context else msg)


def resolve_level_source(source: Any, *, context: str = "") -> list:
    import pandas as pd

    if hasattr(source, "categories_"):
        _fail(
            "levels= does not accept fitted encoder objects; pass the "
            "vocabulary itself, e.g. levels=encoder.categories_[0] for a "
            "single-feature encoder.",
            context,
            TypeError,
        )

    if isinstance(source, pd.CategoricalDtype):
        labels = source.categories.tolist()
    elif isinstance(source, (list, tuple)):
        labels = list(source)
    elif isinstance(source, (pd.Series, np.ndarray)) or (
        hasattr(source, "to_numpy") and hasattr(source, "dtype")
    ):
        if isinstance(source, pd.Series) and isinstance(source.dtype, pd.CategoricalDtype):
            labels = source.dtype.categories.tolist()
        else:
            declared = getattr(getattr(source, "dtype", None), "categories", None)
            if declared is not None:
                # polars Enum dtype exposes declared categories
                labels = list(declared)
            else:
                values = np.asarray(
                    source.to_numpy() if hasattr(source, "to_numpy") else source
                ).ravel()
                labels = sorted(pd.unique(values).tolist(), key=str)
    else:
        _fail(
            "levels= accepts a list/tuple of labels, a data column "
            "(pandas/polars Series or numpy array), or a pandas "
            f"CategoricalDtype; got {type(source).__name__}.",
            context,
            TypeError,
        )

    if any(v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v) for v in labels):
        _fail("levels= contains a missing value; a level cannot be NaN or None.", context)
    seen: set = set()
    dupes = [v for v in labels if v in seen or seen.add(v)]
    if dupes:
        _fail(f"levels= contains duplicate labels: {sorted(set(dupes), key=str)}.", context)
    if len(labels) < 2:
        _fail(f"levels= needs >= 2 labels, got {len(labels)}.", context)
    return labels
```

Note: `pd.isna(v)` on a scalar is safe here (labels are scalars, not arrays); it catches `pd.NA`/`pd.NaT` beside the narrow float test, matching the "a level cannot be missing" contract. The polars branch works because a polars `Series` has `to_numpy` and `dtype`, and only its `Enum` dtype has `.categories`.

- [ ] **Step 4: Run to verify pass** — `python -m pytest tests/test_level_source.py -x -q`. Expected: all pass (polars test skips if polars absent).

- [ ] **Step 5: Commit**

```bash
git add src/superglm/features/_level_source.py tests/test_level_source.py
git commit -m "Add level-source normalization for declared universes"
```

---

### Task 2: Frame boundary — surface dtype-declared categories

**Files:**
- Modify: `src/superglm/_frame.py` (add one method to `EagerFrame`, near `column_kind` at `:90`)
- Test: `tests/test_dataframe_boundary.py` (append a new test class)

**Interfaces:**
- Produces: `EagerFrame.column_declared_categories(name) -> list | None` — declared categories when the column dtype carries them (pandas `CategoricalDtype`, polars `Enum`), else `None`. plain polars `Categorical` (no declared universe) → `None`.
- Consumes: nothing from other tasks. Task 4 calls this from the build loop.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_dataframe_boundary.py`)

```python
class TestDeclaredCategories:
    def test_pandas_categorical_dtype_declared_categories(self):
        import pandas as pd
        from superglm._frame import as_eager_frame

        df = pd.DataFrame(
            {"g": pd.Categorical(["a", "b"], categories=["a", "b", "c"]), "x": [1.0, 2.0]}
        )
        frame = as_eager_frame(df)
        assert frame.column_declared_categories("g") == ["a", "b", "c"]

    def test_pandas_object_column_returns_none(self):
        import pandas as pd
        from superglm._frame import as_eager_frame

        frame = as_eager_frame(pd.DataFrame({"g": ["a", "b"]}))
        assert frame.column_declared_categories("g") is None

    def test_declared_categories_survive_take_rows(self):
        import pandas as pd
        from superglm._frame import as_eager_frame
        import numpy as np

        df = pd.DataFrame(
            {"g": pd.Categorical(["a", "b", "a"], categories=["a", "b", "c"])}
        )
        sliced = as_eager_frame(as_eager_frame(df).take_rows(np.array([0, 2])))
        assert sliced.column_declared_categories("g") == ["a", "b", "c"]

    def test_polars_enum_declared_categories(self):
        pl = pytest.importorskip("polars")
        from superglm._frame import as_eager_frame

        df = pl.DataFrame({"g": pl.Series(["a", "b"], dtype=pl.Enum(["a", "b", "c"]))})
        assert as_eager_frame(df).column_declared_categories("g") == ["a", "b", "c"]

    def test_polars_plain_categorical_returns_none(self):
        pl = pytest.importorskip("polars")
        from superglm._frame import as_eager_frame

        df = pl.DataFrame({"g": pl.Series(["a", "b"], dtype=pl.Categorical)})
        assert as_eager_frame(df).column_declared_categories("g") is None
```

- [ ] **Step 2: Run to verify failure** — `python -m pytest tests/test_dataframe_boundary.py::TestDeclaredCategories -x -q`. Expected: AttributeError.

- [ ] **Step 3: Implement** — add to `EagerFrame` after `column_dtype` (`_frame.py:124`), mirroring the backend split of `column_kind`:

```python
    def column_declared_categories(self, name: object) -> list | None:
        """Return dtype-declared categories, or None when the dtype has none.

        Only dtypes that DECLARE a universe qualify (pandas CategoricalDtype,
        polars Enum). A plain polars Categorical is an encoding, not a
        declaration, and returns None.
        """
        if self.backend == "pandas":
            dtype = cast(pd.DataFrame, self.native)[cast(Any, name)].dtype
            if isinstance(dtype, pd.CategoricalDtype):
                return dtype.categories.tolist()
            return None
        dtype = self._polars_schema[cast(str, name)]
        if isinstance(dtype, nw.Enum):
            return list(dtype.categories)
        return None
```

Verify the narwhals `Enum` attribute first: `python -c "import narwhals as nw, polars as pl; import narwhals.stable.v1 as nws; d = nw.from_native(pl.DataFrame({'g': pl.Series(['a'], dtype=pl.Enum(['a','b']))})).schema['g']; print(type(d), d.categories)"` — if the schema object's attribute is named differently in the pinned narwhals version, adapt the accessor (the test is the contract, not the attribute name). Check how `_polars_schema` types come back (they are narwhals dtypes per `column_kind`'s `isinstance(dtype, nw.String | nw.Categorical | nw.Enum)` at `_frame.py:116`).

- [ ] **Step 4: Run to verify pass** — `python -m pytest tests/test_dataframe_boundary.py -x -q` (whole file: prove no regression in the boundary suite).

- [ ] **Step 5: Commit**

```bash
git add src/superglm/_frame.py tests/test_dataframe_boundary.py
git commit -m "Surface dtype-declared categories at the frame boundary"
```

---

### Task 3: Categorical core — declared universe, pins, unseen policy

**Files:**
- Modify: `src/superglm/features/categorical.py`
- Test: `tests/test_categorical_levels.py` (new)

**Interfaces:**
- Consumes: `resolve_level_source` (Task 1).
- Produces (relied on by Tasks 4, 5, 8, 9):
  - `Categorical.__init__(self, base="most_exposed", grouping=None, *, levels=None, unseen="error")`
  - `Categorical.adopt_dtype_categories(categories: list) -> None` — sets the universe iff none declared; records `_level_source = "dtype"`.
  - `Categorical.apply_level_binding(binding) -> None` — `binding` has `.levels: tuple` and `.base` (level name or `None`); adopts levels iff no universe yet (`_level_source = "full-frame"`); pins `self._pinned_base = binding.base` iff `self.base == "most_exposed"` and binding.base is not None.
  - `Categorical.resolve_binding(values, sample_weight) -> LevelBinding` — pure (no self-mutation): full-column universe + resolved `most_exposed` base, for `cross_validate` (Task 5). Import `LevelBinding` from `superglm.types` (Task 4 creates it; in THIS task define the method but guard the import inside the method body with a local `from superglm.types import LevelBinding` — Task 4 lands the type; to keep Task 3 self-contained, create the type here if absent, see Step 3).
  - Post-fit state: `_levels` (full universe incl. pinned), `_non_base` (emitted columns only), `_pinned_levels: list`, `_base_fallback: tuple[str, str] | None` (requested, used), `_level_source: str` in `{"declared","dtype","full-frame","inferred"}`.
  - `reconstruct()` gains keys: `"pinned_levels"`, `"level_source"`, `"base_fallback"`; pinned levels appear in `relativities` at 1.0 / `log_relativities` at 0.0.
  - `unseen` attribute readable by derived terms (Task 8).

Behavioral contract (spec §3.1, §3.2, §3.3):
1. Universe present → `build` codes via `pd.Categorical(x, categories=universe)`; `-1` codes are split into missing (narrow per-element test → existing missing-values error) vs out-of-universe (→ `ValueError` "outside the declared level universe", naming both sets, sorted `key=str`).
2. Effective observation = count when unweighted, else total `sample_weight` per level; levels with zero effective weight (and ≠ base) are pinned: no design column, codes remapped to `-1`, one `UserWarning` naming them, recorded in `_pinned_levels`.
3. Resolved base with zero effective weight → deterministic fallback (most-exposed observed by weight; first observed in universe order when unweighted), `UserWarning`, `_base_fallback = (requested, used)`.
4. `base="first"` with a declared universe = first-declared (universe order), not alphabetical.
5. Grouped + declared: `levels=` declares the RAW universe; every declared raw level must be covered by `grouping.original_to_group` else `ValueError`; the working universe is the grouping-image of the declared raws, first-occurrence order.
6. `score`/`transform` with `unseen="error"`: exactly today's behavior. With `unseen="base"`: skip the unseen validation, missing-values check still runs, out-of-universe rows get zero contribution, one `UserWarning` with levels + affected row count.
7. The `score` mapping is wraparound-proof regardless of policy: `-1` codes must never fancy-index from the tail.
8. No universe (status quo): byte-identical behavior to current code, `_level_source == "inferred"`.

- [ ] **Step 1: Write the failing tests** — `tests/test_categorical_levels.py`:

```python
"""Bound level universes on Categorical (spec 2026-08-11)."""
import warnings

import numpy as np
import pandas as pd
import pytest

from superglm.features.categorical import Categorical


def _build(spec, x, w=None):
    return spec.build(np.asarray(x, dtype=object), sample_weight=w)


class TestDeclaredUniverse:
    def test_declared_unobserved_level_is_known_and_pinned(self):
        spec = Categorical(base="first", levels=["a", "b", "c"])
        with pytest.warns(UserWarning, match=r"pinned to base.*'c'"):
            info = _build(spec, ["a", "b", "a"])
        assert spec._levels == ["a", "b", "c"]
        assert spec._pinned_levels == ["c"]
        assert spec._non_base == ["b"]          # no column for 'c'
        assert info.n_cols == 1

    def test_pinned_level_scores_as_base(self):
        spec = Categorical(base="first", levels=["a", "b", "c"])
        with pytest.warns(UserWarning):
            _build(spec, ["a", "b", "a"])
        eta = spec.score(np.asarray(["a", "c", "b"], dtype=object), np.array([0.7]))
        assert eta == pytest.approx([0.0, 0.0, 0.7])

    def test_fit_data_outside_universe_errors(self):
        spec = Categorical(base="first", levels=["a", "b"])
        with pytest.raises(ValueError, match="outside the declared level universe"):
            _build(spec, ["a", "b", "ROGUE"])

    def test_missing_values_still_error_before_universe_check(self):
        spec = Categorical(base="first", levels=["a", "b"])
        with pytest.raises(ValueError, match="missing values"):
            _build(spec, ["a", None, "b"])

    def test_declared_order_defines_base_first(self):
        spec = Categorical(base="first", levels=["z", "a"])
        _build(spec, ["a", "z"])
        assert spec._base_level == "z"

    def test_no_universe_is_status_quo(self):
        spec = Categorical(base="first")
        info = _build(spec, ["b", "a", "b"])
        assert spec._levels == ["a", "b"]
        assert spec._level_source == "inferred"
        assert info.n_cols == 1

    def test_levels_source_series(self):
        spec = Categorical(base="first", levels=pd.Series(["b", "a", "b"]))
        assert spec._declared_levels == ["a", "b"]
        assert spec._level_source == "declared"


class TestZeroWeightAndBaseFallback:
    def test_zero_weight_level_is_pinned(self):
        spec = Categorical(base="first", levels=["a", "b", "c"])
        with pytest.warns(UserWarning, match=r"pinned to base.*'c'"):
            _build(spec, ["a", "b", "c"], w=np.array([1.0, 1.0, 0.0]))
        assert spec._pinned_levels == ["c"]

    def test_empty_declared_base_falls_back_deterministically(self):
        spec = Categorical(base="c", levels=["a", "b", "c"])
        with pytest.warns(UserWarning, match="fall"):
            _build(spec, ["a", "b", "b"], w=np.array([1.0, 2.0, 2.0]))
        assert spec._base_level == "b"          # most exposed observed
        assert spec._base_fallback == ("c", "b")

    def test_empty_base_fallback_unweighted_first_observed(self):
        spec = Categorical(base="c", levels=["a", "b", "c"])
        with pytest.warns(UserWarning, match="fall"):
            _build(spec, ["b", "a", "b"])
        assert spec._base_level == "a"          # first observed in universe order

    def test_most_exposed_ignores_pinned_levels(self):
        spec = Categorical(levels=["a", "b", "c"])
        with pytest.warns(UserWarning):
            _build(spec, ["a", "b"], w=np.array([1.0, 5.0]))
        assert spec._base_level == "b"


class TestUnseenPolicy:
    def _fitted(self):
        spec = Categorical(base="first", unseen="base")
        _build(spec, ["a", "b", "a"])
        return spec

    def test_unseen_base_routes_to_zero_with_warning(self):
        spec = self._fitted()
        with pytest.warns(UserWarning, match=r"NOVEL.*2 row"):
            eta = spec.score(np.asarray(["a", "NOVEL", "NOVEL"], dtype=object), np.array([0.5]))
        assert eta == pytest.approx([0.0, 0.0, 0.0])

    def test_unseen_base_transform_zero_rows(self):
        spec = self._fitted()
        with pytest.warns(UserWarning):
            T = spec.transform(np.asarray(["b", "NOVEL"], dtype=object))
        assert T.tolist() == [[1.0], [0.0]]

    def test_unseen_error_is_default_and_unchanged(self):
        spec = Categorical(base="first")
        _build(spec, ["a", "b"])
        with pytest.raises(ValueError, match="unseen categorical levels"):
            spec.score(np.asarray(["NOVEL"], dtype=object), np.array([0.5]))

    def test_unseen_base_missing_values_still_error(self):
        spec = self._fitted()
        with pytest.raises(ValueError, match="missing values"):
            spec.score(np.asarray(["a", None], dtype=object), np.array([0.5]))

    def test_invalid_unseen_rejected(self):
        with pytest.raises(ValueError, match="unseen"):
            Categorical(unseen="ignore")

    def test_wraparound_guard_no_negative_indexing(self):
        # base='b' is the LAST level: a wrapped -1 would grab beta of the
        # last non-base level instead of 0. Assert exact zeros.
        spec = Categorical(base="b", unseen="base")
        _build(spec, ["a", "b", "a"])
        with pytest.warns(UserWarning):
            eta = spec.score(np.asarray(["NOVEL"], dtype=object), np.array([3.14]))
        assert eta == pytest.approx([0.0])


class TestGroupedDeclared:
    def test_grouping_must_cover_declared_universe(self):
        from superglm.features.grouping import LevelGrouping

        grouping = LevelGrouping({"a": "grp", "b": "grp"})
        with pytest.raises(ValueError, match="not covered by the grouping"):
            Categorical(levels=["a", "b", "c"], grouping=grouping)


class TestAdoptionHooks:
    def test_adopt_dtype_categories_when_unset(self):
        spec = Categorical(base="first")
        spec.adopt_dtype_categories(["a", "b", "c"])
        assert spec._declared_levels == ["a", "b", "c"]
        assert spec._level_source == "dtype"

    def test_adopt_does_not_override_declared(self):
        spec = Categorical(base="first", levels=["x", "y"])
        spec.adopt_dtype_categories(["a", "b"])
        assert spec._declared_levels == ["x", "y"]
        assert spec._level_source == "declared"

    def test_apply_level_binding_levels_and_base(self):
        from superglm.types import LevelBinding

        spec = Categorical()          # most_exposed, no universe
        spec.apply_level_binding(LevelBinding(levels=("a", "b"), base="b"))
        assert spec._declared_levels == ["a", "b"]
        assert spec._level_source == "full-frame"
        _build(spec, ["a", "b"], w=np.array([9.0, 1.0]))
        assert spec._base_level == "b"          # pinned wins over fold exposure

    def test_binding_base_ignored_for_explicit_base(self):
        from superglm.types import LevelBinding

        spec = Categorical(base="a", levels=["a", "b"])
        spec.apply_level_binding(LevelBinding(levels=("a", "b"), base="b"))
        _build(spec, ["a", "b"])
        assert spec._base_level == "a"

    def test_resolve_binding_pure(self):
        spec = Categorical()
        binding = spec.resolve_binding(
            np.asarray(["a", "b", "b"], dtype=object), np.array([1.0, 3.0, 3.0])
        )
        assert list(binding.levels) == ["a", "b"] and binding.base == "b"
        assert spec._levels == [] and spec._declared_levels is None  # untouched


class TestReconstruct:
    def test_reconstruct_reports_pins_and_source(self):
        spec = Categorical(base="first", levels=["a", "b", "c"])
        with pytest.warns(UserWarning):
            _build(spec, ["a", "b", "a"])
        rec = spec.reconstruct(np.array([0.7]))
        assert rec["pinned_levels"] == ["c"]
        assert rec["level_source"] == "declared"
        assert rec["base_fallback"] is None
        assert rec["relativities"]["c"] == pytest.approx(1.0)
        assert rec["log_relativities"]["c"] == pytest.approx(0.0)
```

- [ ] **Step 2: Run to verify failure** — `python -m pytest tests/test_categorical_levels.py -x -q`. Expected: TypeError on `levels=` kwarg.

- [ ] **Step 3: Implement in `categorical.py`.** Shape of the change (adapt to the file, keep its idioms; current code shown at `categorical.py:141-263`):

`__init__` (replaces `:153-158`):

```python
    def __init__(
        self,
        base: str = "most_exposed",
        grouping=None,
        *,
        levels=None,
        unseen: str = "error",
    ):
        from superglm.features._level_source import resolve_level_source

        if unseen not in ("error", "base"):
            raise ValueError(f"unseen must be 'error' or 'base', got {unseen!r}")
        self.base = base
        self.unseen = unseen
        self._grouping = grouping
        self._declared_levels: list | None = (
            None if levels is None else resolve_level_source(levels, context="Categorical")
        )
        self._level_source: str = "declared" if levels is not None else "inferred"
        if self._declared_levels is not None and grouping is not None:
            uncovered = [
                lev
                for lev in self._declared_levels
                if str(lev) not in grouping.original_to_group
            ]
            if uncovered:
                raise ValueError(
                    f"levels= contains labels not covered by the grouping: "
                    f"{sorted(uncovered, key=str)}."
                )
        self._levels: list = []
        self._base_level: str = ""
        self._non_base: list = []
        self._pinned_levels: list = []
        self._base_fallback: tuple | None = None
        self._pinned_base = None
```

(Verify `LevelGrouping.original_to_group` keys are the stringified raw labels — `_grouping_labels` casts through `astype(str)` at `categorical.py:98`; match that.)

Adoption hooks + binding (new methods):

```python
    def adopt_dtype_categories(self, categories: list) -> None:
        """Adopt a dtype-declared universe unless one is already declared."""
        if self._declared_levels is None:
            from superglm.features._level_source import resolve_level_source

            self._declared_levels = resolve_level_source(list(categories), context="Categorical")
            self._level_source = "dtype"

    def apply_level_binding(self, binding) -> None:
        """Adopt a full-frame binding: universe if unset, base pin if unpinned."""
        if self._declared_levels is None and binding.levels is not None:
            self._declared_levels = list(binding.levels)
            self._level_source = "full-frame"
        if binding.base is not None and self.base == "most_exposed":
            self._pinned_base = binding.base

    def resolve_binding(self, values, sample_weight=None):
        """Compute this spec's full-frame binding without mutating the spec."""
        import copy

        from superglm.types import LevelBinding

        probe = copy.deepcopy(self)
        import warnings as _warnings

        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            probe.build(values, sample_weight=sample_weight)
        return LevelBinding(levels=tuple(probe._levels), base=probe._base_level)
```

(`resolve_binding` reuses `build` on a throwaway copy so grouping, sorting, exposure and NaN checks stay single-sourced. If Task 4 has not landed when this task runs, add `LevelBinding` to `superglm/types.py` here — a frozen dataclass `LevelBinding(levels: tuple, base: object | None = None)` — and Task 4 will keep it.)

`build` core (replaces `:174-223`): keep the no-universe branch byte-identical (factorize path). Universe branch:

```python
        universe = self._working_universe()   # declared raws mapped through grouping, or None
        if universe is None:
            codes, uniques = pd.factorize(x, sort=True)
            if (codes == -1).any():
                raise ValueError("Categorical column contains missing values (NaN or None).")
            self._levels = uniques.tolist()
        else:
            codes = pd.Categorical(x, categories=universe).codes.astype(np.intp, copy=False)
            if (codes == -1).any():
                bad = x[codes == -1]
                if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in bad):
                    raise ValueError("Categorical column contains missing values (NaN or None).")
                raise ValueError(
                    f"Training data contains levels outside the declared level universe: "
                    f"{sorted(set(bad.tolist()), key=str)}. Declared: "
                    f"{sorted(universe, key=str)}."
                )
            self._levels = list(universe)
```

`_working_universe()` is a small private method: `None` if `_declared_levels is None`; else the grouping image in first-occurrence order when `_grouping` is not None (`dict.fromkeys(self._grouping.original_to_group[str(lev)] for lev in self._declared_levels)`), else `list(self._declared_levels)`.

Effective-weight observation mask, pins, base (replaces `:189-207`):

```python
        n_levels_total = len(self._levels)
        if sample_weight is not None:
            effective = np.bincount(codes, weights=sample_weight, minlength=n_levels_total)
        else:
            effective = np.bincount(codes, minlength=n_levels_total).astype(np.float64)
        observed = effective > 0.0

        requested = self._pinned_base if self._pinned_base is not None else self.base
        # resolve requested base -> level name (universe order defines 'first')
        if requested == "most_exposed":
            base_level = (
                self._levels[int(np.argmax(effective))]
                if sample_weight is not None
                else next(lev for i, lev in enumerate(self._levels) if observed[i])
            )
        elif requested == "first":
            base_level = self._levels[0]
        elif requested in self._levels:
            base_level = requested
        else:
            raise ValueError(f"Base '{requested}' not found in levels: {self._levels}")

        if not observed[self._levels.index(base_level)]:
            fallback = (
                self._levels[int(np.argmax(effective))]
                if sample_weight is not None
                else next(lev for i, lev in enumerate(self._levels) if observed[i])
            )
            warnings.warn(
                f"Categorical base level '{base_level}' has no effective training "
                f"rows in this fit; falling back to '{fallback}'. Coefficient "
                f"identity changes; predictions do not.",
                UserWarning,
                stacklevel=2,
            )
            self._base_fallback = (base_level, fallback)
            base_level = fallback
        self._base_level = base_level

        self._pinned_levels = [
            lev
            for i, lev in enumerate(self._levels)
            if not observed[i] and lev != self._base_level
        ]
        if self._pinned_levels:
            warnings.warn(
                f"Categorical level(s) {sorted(self._pinned_levels, key=str)} have no "
                f"effective training rows and are pinned to base for this fit "
                f"(zero contribution). They remain known levels.",
                UserWarning,
                stacklevel=2,
            )
        self._non_base = [
            lev
            for i, lev in enumerate(self._levels)
            if lev != self._base_level and observed[i]
        ]
```

Remap (adapts `:212-221`): pinned levels and base both map to `-1`; only `_non_base` levels get column indices 0..k-1. Keep the `>= 2 levels` check against `len(self._levels)` before base selection (note: with a universe it can now also trip at `resolve_level_source` time — both fine). Wait: bincount on unweighted path — the current `most_exposed`+no-weight branch demotes to "first"; the new code above uses first-OBSERVED for unweighted `most_exposed`, which for the no-universe path (everything observed, `factorize` output) is exactly `self._levels[0]` — the current behavior, preserved. Add `import warnings` at module top (module currently has none — top-level import is the codebase norm for `warnings`, check e.g. `random_effect.py` conventions; if the file style keeps stdlib imports at top, put it there).

`score` (replaces `:234-248`) — validation honors the policy; the mapping is wraparound-proof for every path:

```python
    def score(self, x: NDArray, beta: NDArray[np.floating]) -> NDArray[np.floating]:
        import pandas as pd

        x = _resolve_categorical_labels(
            x,
            self._grouping,
            known_levels=set(self._levels) if self.unseen == "error" else None,
        )
        if self.unseen != "error":
            # missing values must still be rejected even when unseen levels are allowed
            _validate_missing_only(x)

        codes = np.asarray(
            pd.Categorical(x, categories=self._levels).codes, dtype=np.intp
        )
        if (codes < 0).any():
            novel = sorted(set(np.asarray(x)[codes < 0].tolist()), key=str)
            n_rows = int((codes < 0).sum())
            warnings.warn(
                f"Routing {n_rows} row(s) with novel categorical level(s) {novel} "
                f"to the base level (unseen='base').",
                UserWarning,
                stacklevel=2,
            )
        level_effects = np.zeros(len(self._levels) + 1, dtype=np.float64)
        for i, lev in enumerate(self._non_base):
            level_effects[self._levels.index(lev)] = float(beta[i])
        # -1 codes (novel under unseen='base') land on the appended zero slot,
        # never on the tail level.
        return level_effects[np.where(codes >= 0, codes, len(self._levels))]
```

`_validate_missing_only(x)` is a tiny module-level helper extracting the narrow per-element missing test from `_validate_categorical_levels` (`:71-75`) so both call sites share it (DRY — refactor `_validate_categorical_levels` to call it too). `transform` (`:225-232`) gets the same treatment: `known_levels=... if self.unseen == "error" else None`, plus the missing-only check and the novel-level warning; its equality-mask construction already yields all-zero rows for unknowns — correct for base routing. NOTE for grouped specs with `unseen="base"`: `_resolve_categorical_labels` with `known_levels=None` still validates raw labels against `grouping.all_original_levels` (`:125-129`) — a novel RAW level under a grouping therefore still errors in v1. Add one test asserting that documented limitation (novel raw label + grouping + `unseen="base"` → the grouping-domain `ValueError`), so the boundary is pinned, not accidental.

`reconstruct` (extends `:250-262`): after the loop, `for lev in self._pinned_levels: log_rels[lev] = 0.0; relativities[lev] = 1.0`, and add the three new keys to the returned dict.

- [ ] **Step 4: Run to verify pass** — `python -m pytest tests/test_categorical_levels.py -x -q` then the guard set: `python -m pytest tests/test_categorical_level_validation.py tests/test_categorical_ux.py tests/test_grouping_label_types.py -q`. Also the contract test: `python -m pytest tests/test_theory_invariants.py -k unseen -q`.

- [ ] **Step 5: Commit**

```bash
git add src/superglm/features/categorical.py tests/test_categorical_levels.py
git commit -m "Categorical: declared level universe, zero-count pins, unseen policy"
```

---

### Task 4: Thread bindings through ModelConfig and the build loop

**Files:**
- Modify: `src/superglm/types.py` (add `LevelBinding` if Task 3 did not), `src/superglm/model/fit_state.py`, `src/superglm/model/base.py` (the `model_build_design_matrix` call), `src/superglm/dm_builder.py` (feature loop, before `spec.build` at `~:997`)
- Test: `tests/test_categorical_levels.py` (extend), `tests/test_dataframe_boundary.py` (extend)

**Interfaces:**
- Consumes: `EagerFrame.column_declared_categories` (Task 2), `adopt_dtype_categories` / `apply_level_binding` hooks (Task 3).
- Produces (relied on by Task 5):
  - `superglm.types.LevelBinding` — `@dataclass(frozen=True)` with `levels: tuple` and `base: object | None = None`.
  - `ModelConfig.level_bindings: tuple[tuple[Hashable, LevelBinding], ...] | None = None` — new defaulted field; `__getattr__` fallback returns `None` for old pickles (extend the existing `features_explicit` pattern at `fit_state.py:102-109`); `capture` reads `getattr(model, "_level_bindings", None)`; `materialize` writes `"_level_bindings"` into the work model dict; `with_value(level_bindings=...)` works via the existing `replace` path.
  - `build_design_matrix(..., level_bindings: dict | None = None)` — new keyword-only param, default `None`; `model/base.py`'s call site passes `dict(model._level_bindings) if getattr(model, "_level_bindings", None) else None`.
  - In the per-feature build loop, immediately before `spec.build(x_col, ...)`:

```python
            declared = X.column_declared_categories(feature_name)
            if declared is not None and hasattr(spec, "adopt_dtype_categories"):
                spec.adopt_dtype_categories(declared)
            if level_bindings is not None and hasattr(spec, "apply_level_binding"):
                binding = level_bindings.get(feature_name)
                if binding is not None:
                    spec.apply_level_binding(binding)
```

(Adapt the variable names to the loop's actual ones around `dm_builder.py:997`; the loop variable for the feature name and the `X` frame are in scope — verify by reading `build_design_matrix` from `:838`.)

- [ ] **Step 1: Write the failing tests** (append to `tests/test_categorical_levels.py`):

```python
class TestEndToEndDtypeUniverse:
    def test_categorical_dtype_column_declares_universe_through_fit(self):
        import pandas as pd
        from superglm import SuperGLM
        from superglm.features import Categorical

        rng = np.random.default_rng(0)
        g = pd.Categorical(
            rng.choice(["a", "b"], size=200), categories=["a", "b", "c"]
        )
        X = pd.DataFrame({"g": g, "x": rng.normal(size=200)})
        y = rng.poisson(1.0, size=200).astype(float)
        model = SuperGLM(family="poisson", features={"g": Categorical(base="first")})
        with pytest.warns(UserWarning, match="pinned to base"):
            model.fit(X, y)
        spec = model._specs["g"]
        assert spec._levels == ["a", "b", "c"]
        assert spec._pinned_levels == ["c"]
        assert spec._level_source == "dtype"
        # predict on a frame containing the declared-but-unfitted level: no error
        Xp = pd.DataFrame({"g": pd.Categorical(["c", "a"], categories=["a", "b", "c"]),
                           "x": [0.0, 0.0]})
        mu = model.predict(Xp)
        assert np.isfinite(mu).all()

    def test_level_bindings_flow_through_config(self):
        import pandas as pd
        from superglm import SuperGLM
        from superglm.features import Categorical
        from superglm.types import LevelBinding

        rng = np.random.default_rng(1)
        X = pd.DataFrame({"g": rng.choice(["a", "b"], size=100).astype(object)})
        y = rng.poisson(1.0, size=100).astype(float)
        model = SuperGLM(family="poisson", features={"g": Categorical(base="first")})
        model._config = model._config.with_value(
            level_bindings=(("g", LevelBinding(levels=("a", "b", "z"), base=None)),)
        )
        with pytest.warns(UserWarning, match="pinned to base"):
            model.fit(X, y)
        assert model._specs["g"]._levels == ["a", "b", "z"]
        assert model._specs["g"]._level_source == "full-frame"

    def test_config_pickle_roundtrip_without_bindings(self):
        import pickle
        from superglm import SuperGLM

        model = SuperGLM(family="poisson")
        state = pickle.loads(pickle.dumps(model._config))
        assert state.level_bindings is None
```

- [ ] **Step 2: Run to verify failure** — `python -m pytest tests/test_categorical_levels.py::TestEndToEndDtypeUniverse -x -q`.

- [ ] **Step 3: Implement.** `types.py`: add near `GroupInfo`:

```python
@dataclass(frozen=True)
class LevelBinding:
    """A resolved level universe (and optional base pin) for one feature."""

    levels: tuple
    base: object | None = None
```

`fit_state.py`: add the field after `retain_fit_state` with default `None`; extend `__getattr__` to return `None` for `"level_bindings"`; in `capture` add `level_bindings=copy.deepcopy(getattr(model, "_level_bindings", None))`; in `materialize` add `"_level_bindings": copy.deepcopy(self.level_bindings),` to the dict (`:202-264`). `constructor_kwargs` does NOT expose it (it is fit-machinery intent, not constructor API). `model/base.py`: find the `build_design_matrix(...)` call inside `model_build_design_matrix` (~`:958`) and pass `level_bindings=`. `dm_builder.py`: add the parameter and the loop hook. `SuperGLM.__init__` does not grow an argument; `_level_bindings` never appears on user-constructed models (only via config).

- [ ] **Step 4: Run to verify pass** — `python -m pytest tests/test_categorical_levels.py tests/test_dataframe_boundary.py -q` plus a broad sanity slice: `python -m pytest tests/test_api.py -q -x`.

- [ ] **Step 5: Commit**

```bash
git add src/superglm/types.py src/superglm/model/fit_state.py src/superglm/model/base.py src/superglm/dm_builder.py tests/test_categorical_levels.py
git commit -m "Thread level bindings through ModelConfig into the build loop"
```

---

### Task 5: cross_validate binds on the full frame

**Files:**
- Modify: `src/superglm/model_selection.py`
- Test: `tests/test_cross_validate.py` (append)

**Interfaces:**
- Consumes: `LevelBinding`, config threading (Task 4), `resolve_binding` (Task 3).
- Produces: `_resolve_level_bindings(model, frame, sample_weight) -> dict[Hashable, LevelBinding]` (module-private), used before the fold loop at `model_selection.py:353`; each fold clone gets `est._config = est._config.with_value(level_bindings=tuple(bindings.items()))` right after `_clone_model` (`:376`).

Rules:
- Explicit-features models: for every template spec with a `resolve_binding` method (Categorical, RandomEffect, FactorSmooth after Task 7), compute the binding from `frame.column_array(name)` + full `sample_weight`. Specs whose universe is already declared still get a binding (harmless: `apply_level_binding` only fills gaps), because the binding ALSO pins `most_exposed` base.
- `features=None` models: run the existing auto-detection on the full frame to discover which columns become categorical specs, then bind those. Find the auto-detect entry (`auto_detect_features`, `dm_builder.py:270`) and call it with the same arguments the fit path uses (read the call inside `build_design_matrix`; reuse, do not reimplement classification). The detected spec objects are used only to call `resolve_binding` and are then discarded.
- Zero cat-family features → empty dict → skip the config rewrite entirely (no behavior change for numeric-only models).
- `OrderedCategorical` is NOT bound here (declared via `order=`; Task 6 handles its fold behavior).

- [ ] **Step 1: Write the failing tests** (append to `tests/test_cross_validate.py`, matching its existing fixture style — read the file's helpers first):

```python
class TestFullFrameLevelBinding:
    def _rare_level_frame(self, n=120, seed=0):
        rng = np.random.default_rng(seed)
        g = rng.choice(["a", "b"], size=n).astype(object)
        g[0] = "rare"                      # exactly one row: pigeonhole-guaranteed
        X = pd.DataFrame({"g": g, "x": rng.normal(size=n)})
        y = rng.poisson(1.2, size=n).astype(float)
        return X, y

    def test_rare_level_no_longer_kills_folds(self):
        from sklearn.model_selection import KFold
        from superglm import SuperGLM, cross_validate
        from superglm.features import Categorical

        X, y = self._rare_level_frame()
        model = SuperGLM(family="poisson", features={"g": Categorical(base="first")})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)   # expected pin warnings
            res = cross_validate(
                model, X, y, cv=KFold(n_splits=5, shuffle=True, random_state=0),
                error_score="raise",
            )
        assert len(res.fold_scores) == 5                    # adapt attr to CrossValidationResult
        assert all(np.isfinite(s) for s in res.fold_scores)

    def test_folds_share_universe_and_base(self):
        from sklearn.model_selection import KFold
        from superglm import SuperGLM, cross_validate
        from superglm.features import Categorical

        X, y = self._rare_level_frame()
        w = np.ones(len(y))
        model = SuperGLM(family="poisson", features={"g": Categorical()})  # most_exposed
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            res = cross_validate(
                model, X, y, sample_weight=w,
                cv=KFold(n_splits=4, shuffle=True, random_state=1),
                return_estimators=True, error_score="raise",
            )
        specs = [est._specs["g"] for est in res.estimators]
        universes = {tuple(s._levels) for s in specs}
        bases = {s._base_level for s in specs}
        assert len(universes) == 1 and len(bases) == 1
        assert set(next(iter(universes))) == {"a", "b", "rare"}

    def test_auto_detect_path_binds_too(self):
        from sklearn.model_selection import KFold
        from superglm import SuperGLM, cross_validate

        X, y = self._rare_level_frame()
        model = SuperGLM(family="poisson")                  # features=None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            res = cross_validate(
                model, X, y, cv=KFold(n_splits=5, shuffle=True, random_state=0),
                error_score="raise",
            )
        assert all(np.isfinite(s) for s in res.fold_scores)

    def test_user_model_not_mutated(self):
        from sklearn.model_selection import KFold
        from superglm import SuperGLM, cross_validate
        from superglm.features import Categorical

        X, y = self._rare_level_frame()
        model = SuperGLM(family="poisson", features={"g": Categorical(base="first")})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            cross_validate(model, X, y, cv=KFold(n_splits=3, shuffle=True, random_state=0))
        assert model._config.level_bindings is None
        assert model._specs["g"]._levels == []
```

Adapt `res.fold_scores` / `res.estimators` to the actual `CrossValidationResult` attributes (read the dataclass near `model_selection.py:180`; use whatever the existing tests in the file use — likely a records list or dict of arrays).

- [ ] **Step 2: Run to verify failure** — `python -m pytest tests/test_cross_validate.py::TestFullFrameLevelBinding -x -q`. Expected: the rare-level test fails (fold error or NaN), the shared-universe test fails.

- [ ] **Step 3: Implement `_resolve_level_bindings` + the two-line fold hook.** Before the fold loop (after `frame = as_eager_frame(X)` and weight validation):

```python
    level_bindings = _resolve_level_bindings(model, frame, sample_weight)
```

and inside the try, after `est = _clone_model(model)`:

```python
            if level_bindings:
                est._config = est._config.with_value(
                    level_bindings=tuple(level_bindings.items())
                )
```

`_resolve_level_bindings` sketch:

```python
def _resolve_level_bindings(model, frame, sample_weight):
    """Bind level universes and most_exposed bases on the full pre-split frame.

    Sharing the level SET across folds is R factor semantics -- the vocabulary
    is a property of the data column, not the training subset. No target
    information crosses folds.
    """
    templates: list[tuple[Any, Any]] = list(model._config.feature_templates)
    if not templates:
        templates = _auto_detected_templates(model, frame)   # features=None path
    bindings: dict[Any, Any] = {}
    for name, spec in templates:
        if not hasattr(spec, "resolve_binding"):
            continue
        values = frame.column_array(name)
        declared = frame.column_declared_categories(name)
        probe = copy.deepcopy(spec)
        if declared is not None and hasattr(probe, "adopt_dtype_categories"):
            probe.adopt_dtype_categories(declared)
        bindings[name] = probe.resolve_binding(values, sample_weight)
    return bindings
```

`_auto_detected_templates` calls the same auto-detection the fit path uses — read `build_design_matrix`'s use of `auto_detect_features` (`dm_builder.py:270-305`) and reproduce the call with the model's config (`categorical_base`, splines settings) on the full frame, returning `(name, spec)` pairs, cat-family only. If `auto_detect_features` needs `y`/weights, pass what `cross_validate` has. Keep it defensive: any column whose kind is not "categorical"/"boolean" contributes nothing.

- [ ] **Step 4: Run to verify pass** — `python -m pytest tests/test_cross_validate.py -q` (whole file).

- [ ] **Step 5: Commit**

```bash
git add src/superglm/model_selection.py tests/test_cross_validate.py
git commit -m "cross_validate: bind level universes and base on the full frame"
```

---

### Task 6: OrderedCategorical — pin thin specials, count declared smooth levels

**Files:**
- Modify: `src/superglm/features/ordered_categorical.py`
- Test: `tests/test_ordered_categorical_specials.py` (append), `tests/test_cross_validate.py` (append one integration test)

**Interfaces:**
- Consumes: warning conventions (Task 3's message shapes).
- Produces: post-fit state `_pinned_specials: list` on `OrderedCategorical`; `reconstruct()` (or its reporting equivalent — find how OC reports; mirror `Categorical.reconstruct`'s new keys) exposes `pinned_specials`.

Behavior changes (spec §3.6), at the two hard-error sites the exploration pinned:
1. `ordered_categorical.py:868-874` ("Special level(s) ... never observed in the training data") and `:879-889` (zero-weight variant): replace raise with pin — drop that special's indicator column, record in `_pinned_specials`, emit one `UserWarning` ("pinned to zero contribution for this fit ... remains a known level"). The special stays in `_known_levels`/`_ordered_levels` (predict-time validation still accepts it; its contribution is zero). Downstream column bookkeeping (coefficient slicing, penalty blocks, display ordering) must shrink consistently — follow how the indicator columns are assembled after `:868` and keep every parallel list aligned.
2. `_require_two_smooth_levels` (`:194-201`): count DECLARED smooth levels (`order=`/`values=` domain) rather than observed ones. An observed count of 1 with a declared count >= 2 proceeds (the penalty bridges empty positions); a declared count < 2 still errors at construction.
3. Genuinely out-of-domain data (not in `_known_levels`) keeps erroring exactly as today (`:836-843` build; predict path) — `tests/test_ordered_categorical_specials.py:493-503` must stay green.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_ordered_categorical_specials.py`, reusing its existing model/data helpers — read the file first; the tests below show required behavior, adapt construction to the file's fixtures):

```python
class TestThinSpecialsPin:
    def test_declared_special_with_no_rows_pins_not_errors(self, oc_model_factory):
        # fixture-style: build the file's standard specials model but with a
        # special label that appears in specials= and NOT in the data
        model, X, y = oc_model_factory(specials=["MISSING", "GHOST"], data_specials=["MISSING"])
        with pytest.warns(UserWarning, match=r"GHOST.*zero contribution"):
            model.fit(X, y)
        spec = next(s for s in model._specs.values() if hasattr(s, "_pinned_specials"))
        assert spec._pinned_specials == ["GHOST"]

    def test_pinned_special_predicts_zero_contribution(self, oc_model_factory):
        model, X, y = oc_model_factory(specials=["MISSING", "GHOST"], data_specials=["MISSING"])
        with pytest.warns(UserWarning):
            model.fit(X, y)
        Xg = X.copy()
        Xg.iloc[0, Xg.columns.get_loc("band")] = "GHOST"
        mu = model.predict(Xg)
        assert np.isfinite(mu).all()

    def test_zero_weight_special_pins(self, oc_model_factory):
        model, X, y = oc_model_factory(specials=["MISSING"], data_specials=["MISSING"])
        w = np.ones(len(y)); w[X["band"] == "MISSING"] = 0.0
        with pytest.warns(UserWarning, match=r"MISSING.*zero contribution"):
            model.fit(X, y, sample_weight=w)

    def test_unseen_level_still_errors(self, oc_model_factory):
        model, X, y = oc_model_factory(specials=["MISSING"], data_specials=["MISSING"])
        with pytest.warns(UserWarning) if False else warnings.catch_warnings():
            model.fit(X, y)
        Xb = X.copy(); Xb.iloc[0, Xb.columns.get_loc("band")] = "NEVER_DECLARED"
        with pytest.raises(ValueError, match="unseen"):
            model.predict(Xb)
```

If the file has no factory fixture, write `oc_model_factory` in the test class using the file's existing construction idiom (it builds `SuperGLM` models with an `OrderedCategorical(order=[...], specials=[...])` feature — copy a working construction from an existing test and parameterize the specials). The CV integration test in `tests/test_cross_validate.py`:

```python
    def test_thin_special_survives_cv(self):
        # one special row: some folds lack it entirely
        from sklearn.model_selection import KFold
        from superglm import SuperGLM, cross_validate
        from superglm.features import OrderedCategorical

        rng = np.random.default_rng(3)
        band = rng.choice(["b1", "b2", "b3"], size=100).astype(object)
        band[0] = "MISSING"
        X = pd.DataFrame({"band": band})
        y = rng.poisson(1.0, size=100).astype(float)
        model = SuperGLM(
            family="poisson",
            features={"band": OrderedCategorical(order=["b1", "b2", "b3"], specials=["MISSING"])},
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            res = cross_validate(
                model, X, y, cv=KFold(n_splits=5, shuffle=True, random_state=0),
                error_score="raise",
            )
        assert all(np.isfinite(s) for s in res.fold_scores)
```

(Adapt `OrderedCategorical(order=..., specials=...)` argument names to the real signature at `ordered_categorical.py:340-355`.)

- [ ] **Step 2: Run to verify failure** — the pin tests raise the current hard error; the CV test dies on a special-less fold.

- [ ] **Step 3: Implement.** Read `ordered_categorical.py:600-900` fully before editing. Replace the raises at `:868-874` and `:879-889` with the pin path; adjust `_require_two_smooth_levels` call sites to pass the declared domain. Keep `_known_levels` untouched (the pinned special is still known). Ensure the indicator-column assembly, penalty block sizes, and the display list `_special_display` stay mutually consistent when a special is dropped — grep for every use of the specials count in the file.

- [ ] **Step 4: Run to verify pass** — `python -m pytest tests/test_ordered_categorical_specials.py tests/test_cross_validate.py -q`.

- [ ] **Step 5: Commit**

```bash
git add src/superglm/features/ordered_categorical.py tests/test_ordered_categorical_specials.py tests/test_cross_validate.py
git commit -m "OrderedCategorical: pin thin specials, count declared smooth levels"
```

---

### Task 7: RandomEffect and FactorSmooth gain universe sources

**Files:**
- Modify: `src/superglm/features/random_effect.py`, `src/superglm/features/factor_smooth.py`
- Test: `tests/test_random_effect.py` (append), `tests/test_factor_smooth_feature.py` (append)

**Interfaces:**
- Consumes: `resolve_level_source` (Task 1), `LevelBinding` (Task 4); the dm_builder hooks (Task 4) call these specs' new methods automatically via `hasattr`.
- Produces: on both classes — `levels=None` constructor kwarg, `adopt_dtype_categories`, `apply_level_binding` (levels only — these terms have no base to pin: pass `base=None` in their bindings), `resolve_binding`, `_level_source`.

Semantics: penalized terms — a declared-but-unobserved level gets a COLUMN (not a pin); its coefficient shrinks to 0 through the ridge/penalty naturally (spec §3.2). So `build` with a declared universe:

```python
        if self._declared_levels is not None:
            self._levels = list(self._declared_levels)
            codes = pd.Index(self._levels).get_indexer(values).astype(np.intp, copy=False)
            if (codes < 0).any():
                bad = values[codes < 0]
                if np.any(pd.isna(bad)):
                    raise ValueError("RandomEffect column contains missing values (NaN or None).")
                raise ValueError(
                    f"Training data contains levels outside the declared level universe: "
                    f"{sorted(set(bad.tolist()), key=str)}."
                )
        else:
            codes, uniques = pd.factorize(values, sort=True)
            ...existing...
```

`n_cols=len(self._levels)` now includes empty levels — verify `RandomEffectGroupMatrix` accepts codes that never reach some bins (it does: bincount-based; its explicit out-of-range check at `_group_matrix_core.py:150-151` is about codes >= n_cols, unaffected). Existing `unseen=` policies unchanged. FactorSmooth: apply the same declared-universe pattern at `_factorize_group` (`factor_smooth.py:201-212`); check what depends on `n_levels` (basis blocks per level — an empty level's block is all-zero and penalized; if the basis construction indexes rows by level presence, verify an empty level cannot produce a degenerate block — if it can, pin THAT case out with a clear error and record it in the task notes rather than shipping a crash).

- [ ] **Step 1: Write failing tests** (append; follow each file's existing style):

```python
# tests/test_random_effect.py
class TestDeclaredUniverse:
    def test_declared_unobserved_level_gets_shrunk_coefficient(self):
        from superglm.features.random_effect import RandomEffect

        spec = RandomEffect(levels=["a", "b", "ghost"])
        info = spec.build(np.asarray(["a", "b", "a"], dtype=object))
        assert info.n_cols == 3
        assert spec._levels == ["a", "b", "ghost"]

    def test_fit_data_outside_declared_universe_errors(self):
        from superglm.features.random_effect import RandomEffect

        spec = RandomEffect(levels=["a", "b"])
        with pytest.raises(ValueError, match="outside the declared level universe"):
            spec.build(np.asarray(["a", "ROGUE"], dtype=object))

    def test_score_on_declared_unobserved_level_uses_its_beta(self):
        from superglm.features.random_effect import RandomEffect

        spec = RandomEffect(levels=["a", "b", "ghost"])
        spec.build(np.asarray(["a", "b"], dtype=object))
        eta = spec.score(np.asarray(["ghost"], dtype=object), np.array([0.1, 0.2, 0.3]))
        assert eta == pytest.approx([0.3])
```

Plus an end-to-end REML fit test: declared 3-level universe, 2 observed, `fit_reml`, assert the fitted effect for the ghost level is finite (pulled to ~0 by the variance component) and prediction on a ghost row works without the population fallback. FactorSmooth: mirror with its fixture style from `tests/test_factor_smooth_feature.py`.

- [ ] **Step 2: Verify failure** — TypeError on `levels=`.

- [ ] **Step 3: Implement** both classes: `levels=` via `resolve_level_source(..., context="RandomEffect")`, the two adoption hooks (levels only), `resolve_binding` returning `LevelBinding(levels=tuple(...), base=None)` (universe = declared/dtype/sorted-observed of the full column — reuse a deepcopy-probe build like `Categorical.resolve_binding`).

- [ ] **Step 4: Verify pass** — `python -m pytest tests/test_random_effect.py tests/test_factor_smooth_feature.py -q`.

- [ ] **Step 5: Commit**

```bash
git add src/superglm/features/random_effect.py src/superglm/features/factor_smooth.py tests/test_random_effect.py tests/test_factor_smooth_feature.py
git commit -m "RandomEffect/FactorSmooth: declared level universes (penalty absorbs empties)"
```

---

### Task 8: Derived interaction terms inherit universe, pins, and policy

**Files:**
- Modify: `src/superglm/features/interaction.py`, `src/superglm/model/screening_ops.py`
- Test: `tests/test_categorical_levels.py` (append a `TestDerivedTerms` class)

**Interfaces:**
- Consumes: parent `Categorical` state (Task 3): `_levels` (universe), `_non_base` (emitted), `_pinned_levels`, `unseen`.
- Produces: derived terms whose build/validate use the parent's bound universe; predict-time unseen handling honors the parent's `unseen` policy.

The four copy sites (`interaction.py:315-316`, `:730-732`, `:803-806`, and the PolynomialCategorical equivalent) already copy `_non_base`/`_base_level`, so pinned levels are automatically absent from derived columns. Required changes:
1. Everywhere a derived term validates prediction values against a fitted level set (`:441-446` SplineCategorical, `:847-858` CategoricalInteraction, and the analogous sites), the KNOWN set must be the parent's full universe `_levels` (so declared-but-pinned levels validate fine), and the unseen branch must honor the copied parent policy: copy `self._cat_unseen = cat_spec.unseen` at build; on `unseen == "base"`, skip the hard validation, route unknown codes to zero contribution for the categorical axis (the interaction contributes nothing on those rows), warn once.
2. `screening_ops.py:390-396` (the negative-code guard in `screen_interactions`): keep the guard, but the message should distinguish "outside the declared universe" from the old "outside the fitted level set" — use `spec._levels` (which is now the universe) and keep raising: screening runs at fit time, where out-of-universe data is always an error.

- [ ] **Step 1: Failing tests** — in `TestDerivedTerms`: build a small model with `interactions=[("g", "x")]` (spline-by-categorical — copy the construction idiom from an existing interaction test in `tests/`, e.g. grep `SplineCategorical` usages) where `g` has a declared universe with one empty level; assert (a) fit succeeds with the pin warning, (b) predict on a row with the pinned level works and the interaction contributes zero on it, (c) with parent `unseen="base"`, predict on a NOVEL level works with the warning; with default policy it raises.

- [ ] **Step 2: Verify failure.**

- [ ] **Step 3: Implement** per the two points above. Read each derived class's build/validate before editing; the parent spec object is available at build time (`cat_spec`), so copying `unseen` is one line per site.

- [ ] **Step 4: Verify pass** — `python -m pytest tests/test_categorical_levels.py -q` plus the interaction suites: `python -m pytest tests/test_screening_worth_gate.py -q` and whichever `tests/test_*interaction*` files exist (`ls tests | grep -i interact`).

- [ ] **Step 5: Commit**

```bash
git add src/superglm/features/interaction.py src/superglm/model/screening_ops.py tests/test_categorical_levels.py
git commit -m "Derived categorical terms inherit universe, pins, and unseen policy"
```

---

### Task 9: Reporting, docs, changelog

**Files:**
- Modify: `src/superglm/model/report_ops.py` (only if summary needs explicit wiring — `reconstruct()` may already flow; verify), `docs/guide/features.md`, `docs/guide/interactions.md`, `CHANGELOG.md`
- Test: `tests/test_design_summary.py` or the summary test file that exercises `reconstruct` output (grep `reconstruct(` in tests; append there)

**Interfaces:**
- Consumes: `reconstruct()` keys from Task 3 (`pinned_levels`, `level_source`, `base_fallback`), `_pinned_specials` from Task 6.

Steps:
1. Test that a fitted model's summary/reporting path surfaces `pinned_levels` and `level_source` for a Categorical with a pinned level (find where `reconstruct` output lands — `model.summary(...)` levels display comes from `superglm.inference.summary_levels`; if the new keys don't flow, add them to the level display rows; assert on the summary object, not stdout).
2. Docs — `docs/guide/features.md:93-119` (Categorical section): document `levels=` (three accepted shapes + the `encoder.categories_[0]` recipe + `pd.get_dummies` explicitly NOT consumed), `unseen=`, zero-count pin semantics, the dtype channel, and a short "CV and level universes" paragraph pointing at `cross_validate`'s full-frame bind; update the `RandomEffect` section (`:123-135`) for `levels=`; update the specials section (`:163-207`) for the pin-instead-of-error change. `docs/guide/interactions.md:128` area: one paragraph on inheritance. Keep the guide's voice; examples must be runnable.
3. `CHANGELOG.md`: add an Unreleased entry (match the file's existing format exactly): the feature, the two deliberate behavior changes (CV folds that previously NaN-scored now complete; `pd.Categorical` declared categories now honored — previously silently stripped), the specials relaxation. **No version bump.**

- [ ] **Steps: write the summary test (failing), implement wiring if needed, write docs, changelog, run** `python -m pytest tests/test_design_summary.py tests/test_documentation_examples.py -q` (the latter guards runnable doc examples if it covers the guide — check; if docs examples are not auto-tested, run the new examples by hand in a scratch script).

- [ ] **Commit**

```bash
git add -A src/superglm docs/guide CHANGELOG.md tests
git commit -m "Reporting, guide docs, and changelog for level universes"
```

---

### Task 10: Full-suite verification + equivalence control

**Files:**
- Test: `tests/test_categorical_levels.py` (append `TestEquivalenceControl`)

Steps:
1. Equivalence control (spec §6.11): one frame, every level observed; three models — `levels=["a","b","c"]` declared (sorted order), dtype-carried (`pd.Categorical` with the same categories), and inferred (raw object column) — assert identical `_levels`, identical design (`spec.transform` outputs byte-equal), and identical fitted coefficients after `fit` (same seed/data). This is the guard against the universe machinery perturbing the already-correct path.
2. Full suite: `python -m pytest tests/ -x -q -p no:cacheprovider` (respect the repo's normal invocation — check `pyproject.toml` / CI config for the canonical command and markers; exclude nothing). Python 3.10-specific numeric divergence is a known non-signal (do not chase it here; it is not in the required checks).
3. Fix anything red that this branch broke (bisect by reverting the suspect commit locally if unclear — but FIX, don't revert, in the final state). Failures that reproduce at the merge-base are pre-existing: note them, don't fix them here.
4. Commit test additions and any fixes.

```bash
git add -A tests src/superglm
git commit -m "Equivalence control and full-suite fixes for level universes"
```

---

### Task 11 (addendum, spec §9): public `bind_levels`

**Files:**
- Create: `src/superglm/model/binding_ops.py` (relocated from `model_selection.py`)
- Modify: `src/superglm/model/api.py` (new method on `SuperGLM`), `src/superglm/model_selection.py` (import from new module + merge rule), `docs/guide/features.md` (train/val/test section — the file was rewritten by Task 9; read it first), `CHANGELOG.md` (extend the Task 9 entry)
- Test: `tests/test_categorical_levels.py` (append `TestBindLevels`)

**Interfaces:**
- Consumes: `_resolve_level_bindings` / `_auto_detected_templates` (Task 5), `LevelBinding` + config threading (Task 4).
- Produces: `SuperGLM.bind_levels(X, sample_weight=None) -> self`; `superglm.model.binding_ops.resolve_level_bindings(model, frame, sample_weight)` (public-ish home of the Task 5 helper, `model_selection` re-imports it).

- [ ] **Step 1: Write the failing tests** (append to `tests/test_categorical_levels.py`):

```python
class TestBindLevels:
    def _outer_frame(self, n=150, seed=7):
        rng = np.random.default_rng(seed)
        g = rng.choice(["a", "b"], size=n).astype(object)
        g[-1] = "holdout_only"
        X = pd.DataFrame({"g": g, "x": rng.normal(size=n)})
        y = rng.poisson(1.1, size=n).astype(float)
        return X, y

    def test_bind_then_manual_holdout(self):
        from superglm import SuperGLM
        from superglm.features import Categorical

        X, y = self._outer_frame()
        model = SuperGLM(
            family="poisson", features={"g": Categorical(base="first")}
        ).bind_levels(X)
        Xtr, ytr, Xho = X.iloc[:-1], y[:-1], X.iloc[[-1]]
        with pytest.warns(UserWarning, match="pinned to base"):
            model.fit(Xtr, ytr)
        assert model._specs["g"]._levels == ["a", "b", "holdout_only"]
        assert model._specs["g"]._level_source == "full-frame"
        assert np.isfinite(model.predict(Xho)).all()

    def test_returns_self_for_chaining(self):
        from superglm import SuperGLM
        from superglm.features import Categorical

        X, _ = self._outer_frame()
        model = SuperGLM(family="poisson", features={"g": Categorical()})
        assert model.bind_levels(X) is model

    def test_declared_universe_violation_fails_at_bind_time(self):
        from superglm import SuperGLM
        from superglm.features import Categorical

        X, _ = self._outer_frame()   # contains 'holdout_only'
        model = SuperGLM(
            family="poisson", features={"g": Categorical(levels=["a", "b"])}
        )
        with pytest.raises(ValueError, match="outside the declared level universe"):
            model.bind_levels(X)

    def test_most_exposed_base_pinned_from_outer_frame(self):
        from superglm import SuperGLM
        from superglm.features import Categorical

        rng = np.random.default_rng(11)
        g = np.array(["a"] * 40 + ["b"] * 60, dtype=object)
        X = pd.DataFrame({"g": g})
        y = rng.poisson(1.0, size=100).astype(float)
        w = np.ones(100)
        model = SuperGLM(family="poisson", features={"g": Categorical()})
        model.bind_levels(X, sample_weight=w)     # outer winner: 'b'
        # training slice where 'a' dominates: outer pin must still win
        sl = np.r_[0:40, 40:50]
        model.fit(X.iloc[sl], y[sl], sample_weight=w[sl])
        assert model._specs["g"]._base_level == "b"

    def test_rebind_replaces(self):
        from superglm import SuperGLM
        from superglm.features import Categorical

        X1 = pd.DataFrame({"g": np.array(["a", "b"], dtype=object)})
        X2 = pd.DataFrame({"g": np.array(["a", "b", "c"], dtype=object)})
        model = SuperGLM(family="poisson", features={"g": Categorical()})
        model.bind_levels(X1).bind_levels(X2)
        bound = dict(model._config.level_bindings)
        assert set(bound["g"].levels) == {"a", "b", "c"}

    def test_features_none_auto_detect_path(self):
        from superglm import SuperGLM

        X, y = self._outer_frame()
        model = SuperGLM(family="poisson").bind_levels(X)
        with pytest.warns(UserWarning, match="pinned to base"):
            model.fit(X.iloc[:-1], y[:-1])
        assert np.isfinite(model.predict(X.iloc[[-1]])).all()

    def test_cross_validate_respects_existing_bindings(self):
        from sklearn.model_selection import KFold
        from superglm import SuperGLM, cross_validate
        from superglm.features import Categorical

        X, y = self._outer_frame()
        model = SuperGLM(
            family="poisson", features={"g": Categorical(base="first")}
        ).bind_levels(X)
        # CV only sees the slice WITHOUT the holdout level; universe must
        # still be the full-frame one on every fold estimator
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            res = cross_validate(
                model, X.iloc[:-1], y[:-1],
                cv=KFold(n_splits=3, shuffle=True, random_state=0),
                return_estimators=True, error_score="raise",
            )
        for est in res.estimators:   # adapt attribute per CrossValidationResult
            assert "holdout_only" in est._specs["g"]._levels

    def test_missing_feature_column_errors(self):
        from superglm import SuperGLM
        from superglm.features import Categorical

        model = SuperGLM(family="poisson", features={"g": Categorical()})
        with pytest.raises(ValueError, match="missing required columns"):
            model.bind_levels(pd.DataFrame({"other": [1.0, 2.0]}))
```

(Adapt `res.estimators` to the real `CrossValidationResult` attribute, same as Task 5 did.)

- [ ] **Step 2: Verify failure** — AttributeError: no `bind_levels`.

- [ ] **Step 3: Implement.**
  1. Create `src/superglm/model/binding_ops.py`; MOVE `_resolve_level_bindings` and `_auto_detected_templates` there from `model_selection.py` (public names `resolve_level_bindings`, `auto_detected_templates`); `model_selection.py` imports them (keep behavior identical — Task 5's tests are the guard). Add `require_columns` validation for explicit-features models: every cat-family feature name must be a column of the frame.
  2. `api.py`, near `clone_unfitted` (`api.py:218`):

```python
    def bind_levels(self, X, sample_weight=None):
        """Bind categorical level universes from the outermost frame.

        Runs the same pre-pass ``cross_validate`` applies to its own input:
        for every categorical-family feature, resolve the level universe and
        any unresolved ``most_exposed`` base from ``X``, and store the result
        on this model's configuration. Call with the FULL frame before any
        train/val/test carve; every subsequent fit on any slice shares one
        universe. Explicit ``levels=`` on a term always wins; re-calling
        replaces the stored bindings. Returns ``self`` so construction
        chains: ``model = SuperGLM(...).bind_levels(df)``.
        """
        from superglm._frame import as_eager_frame
        from superglm.model.binding_ops import resolve_level_bindings

        frame = as_eager_frame(X)
        bindings = resolve_level_bindings(self, frame, sample_weight)
        stored = tuple(bindings.items()) if bindings else None
        self._level_bindings = stored
        self._config = self._config.with_value(level_bindings=stored)
        return self
```

  3. Merge rule in `cross_validate` (the pre-pass call from Task 5):

```python
    existing = dict(model._config.level_bindings or ())
    computed = _resolve_level_bindings(model, frame, sample_weight)
    level_bindings = {**computed, **existing}   # outer/user bindings win
```

  4. Docs: read the Task 9 version of `docs/guide/features.md`; add a "True holdouts and level universes" subsection with the bind-then-split example and the one-line sklearn bridge ("this is fit-the-encoder-on-full-data, minus the encoder object"). CHANGELOG: extend the existing Unreleased entry with one line for `bind_levels`.

- [ ] **Step 4: Verify pass** — `python -m pytest tests/test_categorical_levels.py tests/test_cross_validate.py -q` (the second file guards the relocation + merge rule).

- [ ] **Step 5: Commit**

```bash
git add src/superglm/model/binding_ops.py src/superglm/model/api.py src/superglm/model_selection.py tests/test_categorical_levels.py docs/guide/features.md CHANGELOG.md
git commit -m "Public bind_levels: one-call full-frame universe binding"
```

---

## Self-review (run after writing, before execution)

- Spec coverage: §3.1→T1/T3/T4, §3.2→T3/T6/T7, §3.3→T3/T8, §3.4→T2/T4, §3.5→T5, §3.6→T6/T7/T8, §3.7→T3/T6 messages, §3.8→T9, §5→T9 changelog + T10 equivalence, §6→tests throughout. No gap found.
- Type consistency: `LevelBinding(levels: tuple, base: object | None)` used identically in T3/T4/T5/T7; hook names `adopt_dtype_categories` / `apply_level_binding` / `resolve_binding` identical in T3/T4/T5/T7; state names `_declared_levels`/`_level_source`/`_pinned_levels`/`_base_fallback` identical in T3/T8/T9.
- Placeholders: tasks 6–9 direct the implementer to read regions before editing and to adapt fixture idioms — with the behavioral contract pinned by literal test code in every task. No "add error handling"-style steps remain.
