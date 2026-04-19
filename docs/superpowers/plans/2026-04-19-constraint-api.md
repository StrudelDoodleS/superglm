# Constraint API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the public `monotone=` / `monotone_mode=` spline API with a single typed `constraint=` API based on `Constraint.fit.increasing`-style tokens, while keeping the current solver behavior intact.

**Architecture:** Add a small public constraint-token module that produces immutable `ConstraintSpec` values, normalize those tokens into the existing internal `spec.monotone` / `spec.monotone_mode` fields, and remove the old public arguments from spline constructors and factories. The first pass touches API validation, docs, and tests only; it does not change the SCOP/QP/postfit solver routing.

**Tech Stack:** Python 3.10+, dataclasses or enum-like immutable value objects, MkDocs docs, pytest, ruff

---

## File Map

- Create: `src/superglm/features/constraint.py`
  Purpose: public `Constraint` namespace plus immutable `ConstraintSpec` token type.
- Modify: `src/superglm/features/__init__.py`
  Purpose: export `Constraint`.
- Modify: `src/superglm/__init__.py`
  Purpose: export `Constraint` at the package root.
- Modify: `src/superglm/features/_spline_config.py`
  Purpose: normalize `constraint=` tokens, reject old public args, preserve current internal monotone fields.
- Modify: `src/superglm/features/_spline_factory.py`
  Purpose: replace `monotone` / `monotone_mode` parameters with `constraint`.
- Modify: `src/superglm/features/spline.py`
  Purpose: replace public signatures and docstrings with `constraint=`.
- Modify: `src/superglm/features/ordered_categorical.py`
  Purpose: update docs/examples to refer to `Spline(constraint=...)`.
- Modify: `docs/guide/monotone.md`
  Purpose: show the new API and document reserved `convex` / `concave`.
- Modify: `docs/guide/features.md`
  Purpose: update examples away from the removed public args.
- Modify: `docs/api/features.md`
  Purpose: update factory guidance and examples to the new API.
- Create: `tests/test_constraint_api.py`
  Purpose: focused API/validation tests for `Constraint`.
- Modify: `tests/test_spline_factory.py`
  Purpose: cover `constraint=` in the public `Spline(...)` factory.
- Modify: `tests/test_monotone_fit.py`
  Purpose: migrate monotone examples/tests to the new API.
- Modify: `tests/test_ordered_categorical.py`
  Purpose: verify ordered categorical spline-basis delegation works with `constraint=`.

### Task 1: Add Public Constraint Tokens

**Files:**
- Create: `src/superglm/features/constraint.py`
- Modify: `src/superglm/features/__init__.py`
- Modify: `src/superglm/__init__.py`
- Test: `tests/test_constraint_api.py`

- [ ] **Step 1: Write the failing API tests**

Create `tests/test_constraint_api.py` with:

```python
from __future__ import annotations

import pytest

from superglm import Constraint


def test_constraint_fit_increasing_is_constraint_spec():
    token = Constraint.fit.increasing
    assert token.mode == "fit"
    assert token.kind == "increasing"


def test_constraint_postfit_decreasing_is_constraint_spec():
    token = Constraint.postfit.decreasing
    assert token.mode == "postfit"
    assert token.kind == "decreasing"


@pytest.mark.parametrize(
    "token",
    [
        Constraint.fit.convex,
        Constraint.fit.concave,
        Constraint.postfit.convex,
        Constraint.postfit.concave,
    ],
)
def test_reserved_tokens_exist(token):
    assert token.kind in {"convex", "concave"}
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
uv run pytest tests/test_constraint_api.py -q
```

Expected: FAIL with `ImportError` or `AttributeError` because `Constraint` does not exist yet.

- [ ] **Step 3: Implement the public token module**

Create `src/superglm/features/constraint.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConstraintSpec:
    mode: str
    kind: str


class _ConstraintKindNamespace:
    def __init__(self, mode: str):
        self.increasing = ConstraintSpec(mode=mode, kind="increasing")
        self.decreasing = ConstraintSpec(mode=mode, kind="decreasing")
        self.convex = ConstraintSpec(mode=mode, kind="convex")
        self.concave = ConstraintSpec(mode=mode, kind="concave")


class _ConstraintNamespace:
    fit = _ConstraintKindNamespace("fit")
    postfit = _ConstraintKindNamespace("postfit")


Constraint = _ConstraintNamespace()

__all__ = ["Constraint", "ConstraintSpec"]
```

Then export it in `src/superglm/features/__init__.py`:

```python
from superglm.features.constraint import Constraint
```

and add `"Constraint"` to `__all__`.

Then export it in `src/superglm/__init__.py`:

```python
from superglm.features.constraint import Constraint
```

and add `"Constraint"` to `__all__`.

- [ ] **Step 4: Run the test to verify it passes**

Run:

```bash
uv run pytest tests/test_constraint_api.py -q
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_constraint_api.py src/superglm/features/constraint.py src/superglm/features/__init__.py src/superglm/__init__.py
git commit -m "feat: add public constraint tokens"
```

### Task 2: Normalize `constraint=` and Reject the Old API

**Files:**
- Modify: `src/superglm/features/_spline_config.py`
- Modify: `src/superglm/features/_spline_factory.py`
- Modify: `src/superglm/features/spline.py`
- Test: `tests/test_constraint_api.py`

- [ ] **Step 1: Add the failing normalization tests**

Append to `tests/test_constraint_api.py`:

```python
import pytest

from superglm import Constraint, PSpline


def test_constraint_normalizes_to_internal_monotone_fields():
    spec = PSpline(n_knots=8, constraint=Constraint.fit.increasing)
    assert spec.monotone == "increasing"
    assert spec.monotone_mode == "fit"


def test_old_monotone_args_are_rejected():
    with pytest.raises(TypeError):
        PSpline(n_knots=8, monotone="increasing", monotone_mode="fit")


@pytest.mark.parametrize(
    "token",
    [
        Constraint.fit.convex,
        Constraint.fit.concave,
        Constraint.postfit.convex,
        Constraint.postfit.concave,
    ],
)
def test_reserved_constraint_kinds_raise_not_implemented(token):
    with pytest.raises(NotImplementedError):
        PSpline(n_knots=8, constraint=token)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
uv run pytest tests/test_constraint_api.py -q
```

Expected: FAIL because spline constructors do not yet accept `constraint=` and still accept the old args.

- [ ] **Step 3: Implement normalization and removal of the old public args**

In `src/superglm/features/_spline_config.py`, replace the public initialization arguments:

```python
    monotone: str | None,
    monotone_mode: str,
```

with:

```python
    constraint,
```

and add normalization logic:

```python
from superglm.features.constraint import ConstraintSpec


def _normalize_constraint(constraint):
    if constraint is None:
        return None, "postfit"
    if not isinstance(constraint, ConstraintSpec):
        raise TypeError(
            "constraint must be None or a Constraint token such as "
            "Constraint.fit.increasing"
        )
    if constraint.kind in {"convex", "concave"}:
        raise NotImplementedError(
            f"{constraint.kind} constraints are not implemented yet."
        )
    if constraint.kind not in ("increasing", "decreasing"):
        raise ValueError(f"Unsupported constraint kind: {constraint.kind!r}")
    if constraint.mode not in ("fit", "postfit"):
        raise ValueError(f"Unsupported constraint mode: {constraint.mode!r}")
    return constraint.kind, constraint.mode
```

Then inside `initialize_spec(...)`:

```python
    monotone, monotone_mode = _normalize_constraint(constraint)
    spec.monotone = monotone
    spec.monotone_mode = monotone_mode
```

In `src/superglm/features/_spline_factory.py`, replace public parameters:

```python
    monotone: str | None = None,
    monotone_mode: str = "postfit",
```

with:

```python
    constraint=None,
```

and forward `constraint=constraint` into class construction. Remove the old public monotone-arg validation from this factory layer.

In `src/superglm/features/spline.py`, update every public constructor/factory signature to replace:

```python
    monotone: str | None = None,
    monotone_mode: str = "postfit",
```

with:

```python
    constraint=None,
```

and forward `constraint=constraint` into `_spline_config.initialize_spec(...)` and `_spline_factory.Spline(...)`.

Do **not** remove internal `spec.monotone` / `spec.monotone_mode` fields; they remain the normalized internal representation.

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
uv run pytest tests/test_constraint_api.py -q
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_constraint_api.py src/superglm/features/_spline_config.py src/superglm/features/_spline_factory.py src/superglm/features/spline.py
git commit -m "feat: normalize constraint tokens for splines"
```

### Task 3: Update Factory and Ordered Categorical Surface

**Files:**
- Modify: `tests/test_spline_factory.py`
- Modify: `tests/test_ordered_categorical.py`
- Modify: `src/superglm/features/ordered_categorical.py`

- [ ] **Step 1: Add failing tests for factory and ordered categorical integration**

Add to `tests/test_spline_factory.py`:

```python
from superglm import Constraint


def test_spline_factory_accepts_constraint_token():
    s = Spline(kind="ps", n_knots=8, constraint=Constraint.fit.increasing)
    assert s.monotone == "increasing"
    assert s.monotone_mode == "fit"
```

Add to `tests/test_ordered_categorical.py`:

```python
from superglm import Constraint, OrderedCategorical, PSpline


def test_ordered_categorical_spline_basis_inherits_constraint():
    spec = OrderedCategorical(
        order=["a", "b", "c", "d"],
        basis=PSpline(n_knots=3, constraint=Constraint.fit.increasing),
    )
    assert spec._spline.monotone == "increasing"
    assert spec._spline.monotone_mode == "fit"
```

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run:

```bash
uv run pytest tests/test_spline_factory.py tests/test_ordered_categorical.py -q -k "constraint or inherits_constraint"
```

Expected: FAIL until all factory/documentation surfaces are migrated.

- [ ] **Step 3: Update ordered categorical docs/examples**

In `src/superglm/features/ordered_categorical.py`, change the docstring lines that currently mention:

```python
Spline(monotone="increasing")
```

to:

```python
PSpline(constraint=Constraint.fit.increasing)
```

and update the surrounding prose so it no longer claims monotone support is a future release.

Also update any remaining factory/doc examples in the file from the old public args to `constraint=...`.

- [ ] **Step 4: Run the targeted tests again**

Run:

```bash
uv run pytest tests/test_spline_factory.py tests/test_ordered_categorical.py -q -k "constraint or inherits_constraint"
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_spline_factory.py tests/test_ordered_categorical.py src/superglm/features/ordered_categorical.py
git commit -m "docs: update factory and ordered categorical constraint examples"
```

### Task 4: Migrate Monotone Fit Tests and Public Docs

**Files:**
- Modify: `tests/test_monotone_fit.py`
- Modify: `docs/guide/monotone.md`
- Modify: `docs/guide/features.md`
- Modify: `docs/api/features.md`

- [ ] **Step 1: Convert one representative test first**

In `tests/test_monotone_fit.py`, convert one existing monotone constructor call from:

```python
PSpline(n_knots=8, monotone="increasing", monotone_mode="fit")
```

to:

```python
PSpline(n_knots=8, constraint=Constraint.fit.increasing)
```

and one postfit example to:

```python
constraint=Constraint.postfit.increasing
```

- [ ] **Step 2: Run the smallest targeted test subset**

Run:

```bash
uv run pytest tests/test_monotone_fit.py -q -k "scop or qp or summary_shows_engine"
```

Expected: FAIL at first on remaining old public call sites or imports.

- [ ] **Step 3: Finish the migration of docs and monotone tests**

Update the remaining public docs/examples:

- `docs/guide/monotone.md`
- `docs/guide/features.md`
- `docs/api/features.md`

and the monotone-facing tests in `tests/test_monotone_fit.py` so they consistently use:

```python
constraint=Constraint.fit.increasing
constraint=Constraint.fit.decreasing
constraint=Constraint.postfit.increasing
constraint=Constraint.postfit.decreasing
```

Also add one explicit doc/example note that:

```python
Constraint.fit.convex
Constraint.fit.concave
Constraint.postfit.convex
Constraint.postfit.concave
```

exist as reserved tokens but currently raise `NotImplementedError`.

- [ ] **Step 4: Run the migrated subset again**

Run:

```bash
uv run pytest tests/test_monotone_fit.py -q -k "scop or qp or summary_shows_engine"
uv run --group docs mkdocs build -q
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_monotone_fit.py docs/guide/monotone.md docs/guide/features.md docs/api/features.md
git commit -m "docs: migrate monotone API to constraint tokens"
```

### Task 5: Full Verification and Old-API Rejection Sweep

**Files:**
- Modify: any remaining public call sites discovered by grep
- Test: repo-wide targeted monotone/API checks

- [ ] **Step 1: Search for remaining old public API call sites**

Run:

```bash
rg -n "monotone=|monotone_mode=" src tests docs
```

Expected: only internal normalized-field usage remains. Public examples/tests/constructors should no longer rely on the old API.

- [ ] **Step 2: Add one explicit old-API rejection test if not already covered**

If `tests/test_constraint_api.py` does not already cover both factory and class rejection, add:

```python
def test_spline_factory_old_monotone_args_are_rejected():
    with pytest.raises(TypeError):
        Spline(kind="ps", n_knots=8, monotone="increasing", monotone_mode="fit")
```

- [ ] **Step 3: Run the verification commands**

Run:

```bash
uv run pytest tests/test_constraint_api.py tests/test_spline_factory.py tests/test_ordered_categorical.py tests/test_monotone_fit.py -q
uv run ruff check src/superglm tests/test_constraint_api.py tests/test_spline_factory.py tests/test_ordered_categorical.py tests/test_monotone_fit.py
uv run --group docs mkdocs build -q
```

Expected: all commands PASS

- [ ] **Step 4: Commit the final cleanup if needed**

```bash
git add src/superglm/features/spline.py src/superglm/features/_spline_factory.py src/superglm/features/_spline_config.py src/superglm/features/ordered_categorical.py src/superglm/features/constraint.py src/superglm/features/__init__.py src/superglm/__init__.py tests/test_constraint_api.py tests/test_spline_factory.py tests/test_ordered_categorical.py tests/test_monotone_fit.py docs/guide/monotone.md docs/guide/features.md docs/api/features.md
git commit -m "feat: replace monotone args with constraint tokens"
```
