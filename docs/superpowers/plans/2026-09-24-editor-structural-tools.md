# Editor Structural Tools Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Execution for this plan:** the Workflow in the last section (chosen by Max, 2026-09-24). Each workflow agent executes its tasks' steps in order.

**Goal:** Add Refresh from Python, Set reference and refit, the Breaks tool with Transform and refit, one Restore stack with Revert to original model, and a solid original-model line to the SuperGLM editor, all on one shared structural-step path.

**Architecture:** Every structural change (collapse, ungroup, transform, set reference) becomes one call through a shared session path: build a fresh replacement spec, refit a clone, push a `StructuralStep`, put the refit in force. A shared widget path returns the atomic transition envelope. Collapse and ungroup move onto that path first, which deletes their duplicated plumbing and the widget's parallel info stack; the new operations are thin callers. The browser gets focused modules (`breaks.js`, `chart/break_overlay.js`, `views/breaks_controls.js`), and `main.js` only wires them.

**Tech Stack:** Python 3.13 via uv, the existing FastAPI editor server, NumPy; vanilla ES modules with JSDoc `// @ts-check` (tsc `checkJs`), `node:test`; Playwright (Python) for browser tests.

**Spec:** `docs/superpowers/specs/2026-09-23-editor-structural-tools-design.md`

## Global Constraints

- Work only in `/home/max/projects/superglm/.worktrees/editor-structural-tools` on `feat/editor-structural-tools`. Never `cd` to the repository root. Before every commit run `git rev-parse --abbrev-ref HEAD` and confirm the branch; `git add` explicit paths only; never push.
- The browser only ever sees intentional messages. Raise `EditorValueError` / `EditorTypeError` (`src/superglm/editor/errors.py`) with fixed text; never put backend exception text in one.
- Nothing intrudes: no dialog, prompt or banner in response to a selection. Each new action is its own icon (24×24 stroke SVG in the style of `index.html`) with `aria-label`, `data-popover-title`, `data-popover-body` and a Help entry. The only interruption is the existing structural-confirm dialog.
- Every structural step changes one term and pushes exactly one `StructuralStep`. Restore pops exactly one. Only Revert and distribution re-profiling clear the stack.
- User-facing strings are exactly as written in this plan.
- **Code shape (Max, standing):** no guard that cannot fire; no nested loops in Python (vectorise or restructure); nesting depth ≤ 3; helpers of one job, roughly ≤ 40 lines; names say what the thing is; comments say why, never narrate; no flags nobody sets; no copy of an existing path — extend the shared one; the diff is as small as the behaviour requires. Report source LOC and test LOC separately.
- **Tests:** tolerances derive from the fit tolerance (`model._tol`) and conditioning, never magic numbers; no wall-clock assertions; every regression test gets a mutation check (break the fix, watch the test fail, restore).
- Run suites with xdist: `uv run pytest -n 14 <paths> -q`. Never run the full suite serially.
- Frontend: `npm run check:frontend` passes; no state outside `app/state/store.js`; pointer-drag detail stays in `app/interactions.js`.
- Do not touch `pyproject.toml`'s version, `superglm.__version__` or `uv.lock`'s version pin.
- `docs/superpowers/` is gitignored: plan and spec edits need `git add -f`.
- No real-dataset names in committed text.

## Review Focus

1. **A monotone-constrained source term.** Transforming a term whose spline carries `Constraint.fit.increasing`: the Spline form carries the constraint; the Piecewise and Polynomial forms refuse with a named message. A constraint is a market rule and must never be dropped silently. → Task 4, `test_transform_carries_a_monotone_constraint_to_a_spline_and_refuses_it_elsewhere`.
2. **Crafted `/transform_term` requests.** Breaks outside the fitted range, on the first or last band, unknown bands, duplicates or unsorted input, breaks sent with a polynomial, an out-of-range degree, per-segment degrees on a numeric axis: each gives a 400 with an intentional message and leaves the model, revision and stack unchanged. → Task 4, `test_transform_refuses_invalid_requests_without_changing_anything`.
3. **Set reference on awkward levels.** A numeric-valued categorical (levels `1, 2, 3`, displayed `"1", "2", "3"`) resolves to the native value; an original level inside a collapsed group resolves to its group label; a special level is refused. → Task 3 tests.
4. **A mixed sequence across two terms.** Collapse A, transform B, set reference on A, ungroup A, then Restore four times: each step returns exactly (bitwise) to the predictions before it. → Task 5, `test_restore_walks_a_mixed_two_term_sequence_back_exactly`.
5. **A structural change made from the notebook.** `session.replace_with_collapsed_levels(...)` in Python, then Refresh: the Restore icon, the chart and the reference chip all reflect it. → Task 9, `test_refresh_pulls_a_notebook_side_structural_change`.

---

## File Structure

| File | Change | Responsibility |
|---|---|---|
| `src/superglm/editor/_types.py` | modify | `StructuralStep` — one undoable structural change |
| `src/superglm/editor/session.py` | modify | `structure_history`; `_refit_replacing`, `_push_structure`; collapse/ungroup on them; `replace_with_reference_level`, `replace_with_transformed_term`, `revert_to_reference_model` |
| `src/superglm/editor/collapse.py` | modify | level operations: `rebuilt_ordered_spec` (fresh ordered spec, optional new basis), `reference_feature_spec`, `ungroup_label` |
| `src/superglm/editor/transform.py` | create | `transformed_feature_spec`, break/degree validation, `transform_payload` for the browser |
| `src/superglm/editor/widget.py` | modify | `_structural_step`; every structural operation on it; drop the parallel info stack; stale offset-refit guard |
| `src/superglm/editor/server.py` | modify | routes `/set_reference`, `/transform_term`, `/revert_to_original`, `/restore_structure` (replaces `/uncollapse_levels`) |
| `src/superglm/editor/payloads.py` | modify | `structure_history_payload`; per-term `reference` and `transform`; original line re-anchored at the current reference |
| `src/superglm/editor/summaries.py`, `reports.py` | modify | drop the dead `collapse` field; compact note carries the breaks note |
| `src/superglm/model/report_ops.py` | modify | `editor_break_terms` from specs marked by the editor |
| `src/superglm/inference/summary.py` | modify | `editor_break_notes`; `_editor_notes` includes it |
| `src/superglm/export/summary.py` | modify | reuse `_editor_notes` instead of a duplicate block and constants |
| `src/superglm/plotting/editor_style.py`, `editor/app/styles.css` | modify | solid, wider, semi-transparent original-model line |
| `src/superglm/editor/app/breaks.js` | create | pure break-draft logic (node-tested) |
| `src/superglm/editor/app/chart/svg.js` | create | `el` / `line` / `text` SVG helpers moved out of `chart.js` for sharing |
| `src/superglm/editor/app/chart/break_overlay.js` | create | draws breaks, labels, remove handles and degree chips |
| `src/superglm/editor/app/views/breaks_controls.js` | create | the Breaks controls in the chart action bar |
| `src/superglm/editor/app/{index.html, chart.js, interactions.js, main.js, summary.js}` | modify | markup, one overlay call, the breaks gestures, wiring, descriptors |
| `src/superglm/editor/app/views/{app_bar, context_bar, tool_rail, structural_confirm, help_content}.js` | modify | Refresh/Revert icons, reference chip, Breaks mode, confirm copy, Help |
| `src/superglm/editor/app/state/{store, actions}.js`, `api/contracts.js` | modify | `breakDraftByTerm`, `refreshFromPython`, typedefs |
| `tests/test_editor_structure.py` | create | Python tests for the stack, reference, transform, revert, notes |
| `tests/editor_frontend/breaks.test.js` | create | node tests for `breaks.js` |
| `tests/editor/test_editor_structure_browser.py` | create | one Playwright case per feature |
| existing tests | modify | only assertions that pin replaced semantics (listed per task) |
| `docs/tutorials/edit-a-model-in-the-browser.md`, `docs/development/internals/editor-frontend.md` | modify | user docs; "Add a structural operation" recipe |

---

### Task 1: One structural step in the session

**Files:**
- Modify: `src/superglm/editor/_types.py`
- Modify: `src/superglm/editor/session.py` (`__init__` `collapse_history`; `reprofile_distribution`; `refit_with_collapsed_levels` … `replace_in_force_model`, currently lines ~814–1041)
- Modify: `src/superglm/editor/collapse.py` (metadata of `collapsed_feature_spec` and `ungrouped_feature_spec`)
- Create: `tests/test_editor_structure.py`
- Modify: `tests/test_editor.py` (only the assertions listed in Step 6)

**Interfaces:**
- Produces: `StructuralStep(previous_model, operation: str, term: str | None, label: str)` (frozen dataclass, `superglm.editor._types`).
- Produces: `EditorSession.structure_history: list[StructuralStep]` (replaces `collapse_history`).
- Produces: `EditorSession._refit_replacing(term, build, *, X=None, y=None, sample_weight=None, offset=None, method="auto", lambda1=..., lambda2=..., **fit_kwargs) -> model`, where `build(X_ref) -> (replacement_spec, metadata: dict)`. Stamps `model._editor_step = metadata` (renamed from `_editor_level_collapse`); every metadata dict carries a `"label"`.
- Produces: `EditorSession._push_structure(model, *, operation, term, label) -> model`.
- Produces: `collapse.ungroup_label(term_name: str, levels: list[str]) -> str`.
- `can_uncollapse_levels()` and `uncollapse_levels()` keep their public names and now mean "the latest structural step".

- [ ] **Step 1: Write the failing tests** in the new file `tests/test_editor_structure.py`:

```python
"""Editor structural steps: one stack, set reference, transform, revert."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from superglm import Categorical, Numeric, OrderedCategorical, Polynomial, Spline, SuperGLM
from superglm.editor import EditorSession


@pytest.fixture
def region_model():
    rng = np.random.default_rng(20260924)
    region = rng.choice(["A", "B", "C", "D"], 600, p=[0.3, 0.3, 0.2, 0.2])
    x = rng.uniform(0.0, 10.0, 600)
    effects = {"A": 0.0, "B": 0.15, "C": 0.2, "D": -0.1}
    y = 0.4 + np.array([effects[r] for r in region]) + 0.05 * x + rng.normal(0.0, 0.05, 600)
    X = pd.DataFrame({"region": region, "x": x})
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        spline_penalty=0.1,
        features={"region": Categorical(base="first"), "x": Spline(n_knots=6)},
    )
    model.fit(X, y)
    return model, X


def test_every_structural_step_pushes_one_restorable_entry(region_model):
    model, _ = region_model
    session = EditorSession.from_model(model, terms=["region"])
    opened = session.model

    session.select_levels("region", ["B", "C"])
    collapsed = session.replace_with_collapsed_levels("region", method="fit")
    assert [s.operation for s in session.structure_history] == ["collapse_levels"]
    assert session.structure_history[-1].previous_model is opened
    assert session.structure_history[-1].label == "collapse B + C in region"

    session.select_levels("region", ["B", "C"])
    session.replace_with_ungrouped_levels("region", method="fit")
    # The ungroup removes the last group, so the pre-collapse fit is reused ...
    assert session.model is opened
    # ... but it is still a step of its own: Restore undoes it.
    assert [s.operation for s in session.structure_history] == [
        "collapse_levels",
        "ungroup_levels",
    ]
    assert session.structure_history[-1].label == "ungroup B, C in region"
    assert session.uncollapse_levels() is collapsed
    assert session.uncollapse_levels() is opened
    assert not session.can_uncollapse_levels()
```

- [ ] **Step 2: Run it and watch it fail**

Run: `uv run pytest tests/test_editor_structure.py -q`
Expected: FAIL — `AttributeError: 'EditorSession' object has no attribute 'structure_history'`.

- [ ] **Step 3: Implement**

In `_types.py`, add after `EditRecord`:

```python
@dataclass(frozen=True)
class StructuralStep:
    """One undoable structural change: the model that was in force before it."""

    previous_model: Any
    operation: str
    term: str | None
    label: str
```

In `collapse.py`, add the label helper and put a `"label"` into both metadata dicts:

```python
def ungroup_label(term_name: str, levels: list[str]) -> str:
    return f"ungroup {', '.join(levels)} in {term_name}"
```

`collapsed_feature_spec` metadata gains `"label": f"collapse {' + '.join(selected_levels)} in {term.name}"`; `ungrouped_feature_spec` metadata gains `"label": ungroup_label(term.name, selected_levels)`.

In `session.py`:

1. `__init__`: replace `self.collapse_history: list[Any] = []` with `self.structure_history: list[StructuralStep] = []` (import `StructuralStep` from `superglm.editor._types`).
2. `reprofile_distribution`: `self.collapse_history.clear()` → `self.structure_history.clear()`.
3. Replace `refit_with_collapsed_levels`, `replace_with_collapsed_levels`, `refit_with_ungrouped_levels`, `replace_with_ungrouped_levels`, `can_uncollapse_levels` and `uncollapse_levels` with the following (`_ungroup_restores_reference_model`, `_has_collapsed_level_groups_after_replacement` and `_model_has_collapsed_level_groups` stay as they are):

```python
    def refit_with_collapsed_levels(self, term: str, *, group_label: str | None = None, **refit_kwargs: Any):
        """Collapse selected categorical levels and refit a full model copy.

        ``refit_kwargs`` are ``X``, ``y``, ``sample_weight``, ``offset``,
        ``method``, ``lambda1``, ``lambda2`` and fit keywords.
        """
        editable = self._require_term(term)
        idx = self._require_selection(term)
        return self._refit_replacing(
            term,
            lambda X_ref: collapsed_feature_spec(
                self.model, editable, idx, X=X_ref, group_label=group_label
            ),
            **refit_kwargs,
        )

    def replace_with_collapsed_levels(self, term: str, **kwargs: Any):
        """Collapse selected levels, refit, and make the refit the in-force edit model."""
        refit_model = self.refit_with_collapsed_levels(term, **kwargs)
        return self._push_structure(
            refit_model,
            operation="collapse_levels",
            term=term,
            label=refit_model._editor_step["label"],
        )

    def refit_with_ungrouped_levels(self, term: str, **refit_kwargs: Any):
        """Remove selected levels from collapsed groups and refit a model copy."""
        editable = self._require_term(term)
        idx = self._require_selection(term)
        return self._refit_replacing(
            term,
            lambda X_ref: ungrouped_feature_spec(self.model, editable, idx, X=X_ref),
            **refit_kwargs,
        )

    def replace_with_ungrouped_levels(self, term: str, **kwargs: Any):
        """Ungroup selected levels and put the result in force as one structural step.

        When this ungroup removes the model's last collapsed group and the
        model before the latest step had none, that earlier fit is exactly the
        result, so it is reused instead of refitting.
        """
        levels = [str(self.terms[term].levels[i]) for i in self._require_selection(term)]
        model = self._pre_collapse_model(term, **kwargs)
        if model is None:
            model = self.refit_with_ungrouped_levels(term, **kwargs)
        return self._push_structure(
            model, operation="ungroup_levels", term=term, label=ungroup_label(term, levels)
        )

    def _pre_collapse_model(self, term: str, **kwargs: Any):
        """The model before the latest step, when this ungroup reproduces it exactly."""
        if not self.structure_history or not self._ungroup_restores_reference_model(term, **kwargs):
            return None
        previous = self.structure_history[-1].previous_model
        return None if self._model_has_collapsed_level_groups(previous) else previous

    def can_uncollapse_levels(self) -> bool:
        """Return whether a structural step can be restored."""
        return bool(self.structure_history)

    def uncollapse_levels(self):
        """Restore the model that was in force before the latest structural step."""
        if not self.structure_history:
            raise RuntimeError("No structural step is available to restore.")
        step = self.structure_history.pop()
        self.replace_in_force_model(step.previous_model)
        return step.previous_model

    def _refit_replacing(
        self,
        term: str,
        build: Callable[[Any], tuple[Any, dict[str, Any]]],
        *,
        X=None,
        y=None,
        sample_weight=None,
        offset=None,
        method: str = "auto",
        lambda1=...,
        lambda2=...,
        **fit_kwargs: Any,
    ):
        """Refit a copy of the model with ``term``'s spec replaced by ``build(X_ref)``.

        ``build`` returns the fresh replacement spec and the step's metadata;
        the metadata, with the resolved fit method, is stamped on the refit.
        """
        if self.model is None:
            raise RuntimeError("Cannot refit without a source model.")
        X_ref, y_ref, sample_weight_ref, base_offset = self._resolve_refit_data(
            X, y, sample_weight, offset
        )
        if y_ref is None:
            raise RuntimeError("Fit response data was not retained on the source model.")
        replacement, metadata = build(X_ref)
        refit_model = clone_with_replaced_feature(
            self.model, term, replacement, lambda1=lambda1, lambda2=lambda2
        )
        metadata["method"] = fit_refit_model(
            self.model,
            refit_model,
            method=method,
            X=X_ref,
            y=y_ref,
            sample_weight=sample_weight_ref,
            offset=base_offset,
            fit_kwargs=fit_kwargs,
        )
        refit_model._editor_step = metadata
        return refit_model

    def _push_structure(self, model, *, operation: str, term: str | None, label: str):
        """Put ``model`` in force as one undoable structural step."""
        self.structure_history.append(StructuralStep(self.model, operation, term, label))
        try:
            self.replace_in_force_model(model)
        except Exception:
            self.structure_history.pop()
            raise
        return model
```

Import `Callable` from `collections.abc` and `ungroup_label` from `superglm.editor.collapse`.

- [ ] **Step 4: Run the new test**

Run: `uv run pytest tests/test_editor_structure.py -q`
Expected: PASS.

- [ ] **Step 5: Mutation check.** In `replace_with_ungrouped_levels`, temporarily replace `self._push_structure(...)` with a bare `self.replace_in_force_model(model); return model`. The test must fail on the history assertion. Restore the code.

- [ ] **Step 6: Update the existing tests that pin the old semantics, and nothing else**

Run: `uv run pytest -n 14 tests/test_editor.py -q`. Expected failures, and their updates:
- `collapse_history` → `structure_history`; where a test compares models, compare `[s.previous_model for s in session.structure_history]`.
- `_editor_level_collapse` → `_editor_step`.
- `test_ungroup_last_collapsed_levels_restores_pre_collapse_in_force_model`: keep `restored is profiled_model`; `can_uncollapse_levels()` is now `True`, and `session.structure_history[-1].operation == "ungroup_levels"`.
- `test_ungroup_pre_collapsed_model_refits_without_history`, `test_ordered_integer_ungroup_pre_collapsed_model_refits_without_history`, `test_final_ungroup_after_partial_ungroup_does_not_restore_stale_history`: the ungroup now pushes a step instead of clearing; assert the pushed step rather than an empty history.

Any other failure is a regression in this task. Fix the code, not the test. List every changed assertion in the commit message.

- [ ] **Step 7: Commit**

```bash
git add src/superglm/editor/_types.py src/superglm/editor/session.py src/superglm/editor/collapse.py tests/test_editor_structure.py tests/test_editor.py
git commit -m "Put every structural edit on one session path and one undo stack"
```

---

### Task 2: One structural step in the widget; `structure_history` in the snapshot; `/restore_structure`

**Files:**
- Modify: `src/superglm/editor/widget.py` (`__init__` lines ~113–120; `_state`; `_summary`; `_refit_offset`; `_collapse_levels` … `_uncollapse_levels`; `_invalidate_refit`)
- Modify: `src/superglm/editor/payloads.py`, `src/superglm/editor/summaries.py`, `src/superglm/editor/reports.py`, `src/superglm/editor/server.py`
- Modify: `src/superglm/editor/app/api/contracts.js`, `app/state/actions.js` (`isEditorSnapshot`), `app/summary.js`, `app/main.js`, `app/views/structural_confirm.js`
- Modify tests: `tests/test_editor.py`, `tests/editor_frontend/{actions,store,structural_confirm,summary}.test.js`, `tests/editor/test_editor_workspace_browser.py`
- Test: `tests/test_editor_structure.py`

**Interfaces:**
- Consumes: Task 1's `structure_history`, `StructuralStep`, `_editor_step`.
- Produces: `EditorWidget._structural_step(operation: str, apply: Callable[[str], Any], *, term: str | None = None, level_display: str = "expanded") -> dict` (the envelope).
- Produces: `EditorWidget._restore_structure(*, level_display="expanded")` (replaces `_uncollapse_levels`).
- Produces: `payloads.structure_history_payload(session) -> {"depth": int, "last": {"operation": str, "term": str | None, "label": str} | None}`, published as `state["structure_history"]`. `state["can_uncollapse_levels"]` and `state["last_collapse"]` are removed.
- Produces: route `POST /restore_structure {level_display}` (replaces `/uncollapse_levels`).
- Produces (JS): typedef `StructureHistory`; `restoreTransition()` in `summary.js` (replaces `uncollapseTransition`), `{name: "restore previous structure", path: "/restore_structure", payload: {}}`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_editor_structure.py`):

```python
from superglm.editor.widget import EditorWidget


def test_state_publishes_the_structure_history(region_model):
    model, _ = region_model
    session = EditorSession.from_model(model, terms=["region"])
    widget = EditorWidget(session)
    try:
        assert widget._state()["structure_history"] == {"depth": 0, "last": None}
        session.select_levels("region", ["B", "C"])
        envelope = widget._collapse_levels("region", "fit")
        assert envelope["state"]["structure_history"] == {
            "depth": 1,
            "last": {
                "operation": "collapse_levels",
                "term": "region",
                "label": "collapse B + C in region",
            },
        }
        assert "last_collapse" not in envelope["state"]
        restored = widget._restore_structure()
        assert restored["timing"]["operation"] == "restore_structure"
        assert restored["state"]["structure_history"]["depth"] == 0
    finally:
        widget.close()


def test_offset_refit_summary_goes_unavailable_after_a_notebook_side_edit(region_model):
    model, _ = region_model
    session = EditorSession.from_model(model, terms=["region", "x"])
    widget = EditorWidget(session)
    try:
        session.select_indices("x", [3, 4])
        session.shift("x", 0.1)
        assert widget._refit_offset("fit")["available"] is True
        session.shift("x", 0.1)  # made in the notebook: never passes through the widget
        assert widget._summary("refit")["available"] is False
    finally:
        widget.close()
```

- [ ] **Step 2: Run and watch both fail**

Run: `uv run pytest tests/test_editor_structure.py -q -k "structure_history or offset_refit"`
Expected: FAIL — `KeyError: 'structure_history'`, and `available` is still `True` after the notebook edit.

- [ ] **Step 3: Implement the Python side**

`payloads.py`:

```python
def structure_history_payload(session) -> dict[str, Any]:
    """The Restore icon's state: how many steps there are and what the last one did."""
    steps = session.structure_history
    if not steps:
        return {"depth": 0, "last": None}
    last = steps[-1]
    return {
        "depth": len(steps),
        "last": {"operation": last.operation, "term": last.term, "label": last.label},
    }
```

`widget.py`:
1. `__init__`: delete `_collapsed_refit_model`, `_collapsed_refit_info`, `_collapse_info_history` and `_in_force_info`; add `self._offset_refit_revision: int | None = None`.
2. `_state`: replace the `"can_uncollapse_levels"` and `"last_collapse"` entries with `"structure_history": structure_history_payload(self.session)`.
3. Add `_structural_step` and rebuild the structural operations on it (delete the old bodies):

```python
    def _structural_step(
        self,
        operation: str,
        apply: Callable[[str], Any],
        *,
        term: str | None = None,
        level_display: str = "expanded",
    ) -> dict[str, Any]:
        """Run one structural session change and return its atomic transition envelope."""
        level_display = validate_level_display(level_display)
        with self._lock:
            operation_start = time.perf_counter()
            if term is not None:
                self._select_term(term)
            target = self.selected_term
            selected_indices = self.session.selection(target).astype(int).tolist()
            selected_levels = self._selected_level_labels(target)
            fit_start = time.perf_counter()
            apply(target)
            fit_end = time.perf_counter()
            self._invalidate_refit()
            self._restore_selection(target, selected_levels, selected_indices)
            self._chart_generation += 1
            return self._structural_transition(
                operation,
                operation_start=operation_start,
                fit_start=fit_start,
                fit_end=fit_end,
                level_display=level_display,
            )

    def _collapse_levels(self, term=None, method="auto", *, level_display="expanded"):
        return self._structural_step(
            "collapse_levels",
            lambda target: self.session.replace_with_collapsed_levels(target, method=method),
            term=term,
            level_display=level_display,
        )

    def _ungroup_levels(self, term=None, method="auto", *, level_display="expanded"):
        return self._structural_step(
            "ungroup_levels",
            lambda target: self.session.replace_with_ungrouped_levels(target, method=method),
            term=term,
            level_display=level_display,
        )

    def _restore_structure(self, *, level_display="expanded"):
        return self._structural_step(
            "restore_structure",
            lambda _target: self.session.uncollapse_levels(),
            level_display=level_display,
        )
```

Restore now keeps the selection by level label, like every other step (it used to clear it). Say so in the commit message.

4. Offset refit: in `_refit_offset`, after storing the refit, set `self._offset_refit_revision = self.session.model_revision`. In `_invalidate_refit`, also set `self._offset_refit_revision = None`. In `_summary`, directly after the superseded-revision check, add:

```python
        with self._lock:
            # A notebook-side edit advances the revision without passing through
            # this widget, so a stored fixed-offset refit can outlive its edits.
            if self._offset_refit_revision != self.session.model_revision:
                self._invalidate_refit()
```

and delete the `collapse_info_override` / `_in_force_info` plumbing from `_summary`.

`summaries.py` and `reports.py`: delete the `collapse_info_override` parameter and the `"collapse"` key; no browser code reads it (`rg '\.collapse\b' src/superglm/editor/app` finds nothing). In `summary_payload`, delete the `else:` branch that reads `widget._collapsed_refit_model` — every caller normalises `source` to `original`, `in_force` or `refit` before it.

`server.py`: replace the `/uncollapse_levels` route with:

```python
    @app.post("/restore_structure")
    def restore_structure(payload: dict[str, Any] = Body(default_factory=dict)) -> Response:
        return _guarded_json(lambda: widget._restore_structure(level_display=_level_display(payload)))
```

- [ ] **Step 4: Implement the browser side of the new snapshot shape**

`api/contracts.js`: add

```js
/**
 * @typedef {Object} StructureHistory
 * @property {number} depth
 * @property {{operation:string, term:string|null, label:string}|null} last
 */
```

and in `EditorSnapshot` replace `can_uncollapse_levels` and `last_collapse` with `@property {StructureHistory} structure_history`.

`state/actions.js` `isEditorSnapshot`: replace the `last_collapse` / `can_uncollapse_levels` checks with `if (!isStructureHistory(value.structure_history)) return false;` and add:

```js
/** @param {unknown} value @returns {boolean} */
function isStructureHistory(value) {
  return isRecord(value) &&
    Number.isInteger(value.depth) && Number(value.depth) >= 0 &&
    (value.last === null || (isRecord(value.last) && typeof value.last.label === "string"));
}
```

`summary.js`: rename `uncollapseTransition` to `restoreTransition` returning `{ name: "restore previous structure", path: "/restore_structure", payload: {} }`.
`main.js`: update the import and the `descriptor.name !== "restore collapsed levels"` check to `"restore previous structure"`. In `updateCollapseAction`, the Restore button's visibility becomes `snapshot.structure_history.depth > 0`. Task 7 moves the button; this visibility rule is final.
`views/structural_confirm.js`: `OPERATION_TITLES["restore previous structure"] = "Restore previous structure"`, replacing the collapse-only entry; the question for it becomes `Restore the model before "${snapshot.structure_history.last.label}"?`.

- [ ] **Step 5: Run and update the pinned tests**

Run: `uv run pytest -n 14 tests/test_editor.py tests/test_editor_structure.py -q` and `npm run check:frontend`.
Expected updates, and only these:
- Tests that read `state["last_collapse"]` / `state["can_uncollapse_levels"]` read `state["structure_history"]`.
- `test_widget_final_ungroup_keeps_collapse_metadata_history_aligned` becomes a check that `structure_history` depth and last label track the steps.
- The info-detachment tests around the old `_in_force_info` (lines ~4835–4895) keep their purpose on the new field: mutating `envelope["state"]["structure_history"]["last"]["label"]` must not change the next `widget._state()`.
- Node fixtures replace `can_uncollapse_levels`/`last_collapse` with `structure_history: { depth: 0, last: null }`.
- `test_editor_workspace_browser.py` references follow the rename.

- [ ] **Step 6: Mutation check.** Remove the new stale-refit block from `_summary`; `test_offset_refit_summary_goes_unavailable_after_a_notebook_side_edit` must fail. Restore it.

- [ ] **Step 7: Commit**

```bash
git add src/superglm/editor/{widget,payloads,summaries,reports,server}.py src/superglm/editor/app/api/contracts.js src/superglm/editor/app/state/actions.js src/superglm/editor/app/summary.js src/superglm/editor/app/main.js src/superglm/editor/app/views/structural_confirm.js tests/test_editor.py tests/test_editor_structure.py tests/editor_frontend tests/editor/test_editor_workspace_browser.py
git commit -m "Run every structural edit through one widget step and publish its history"
```

---

### Task 3: Set reference and refit (Python)

**Files:**
- Modify: `src/superglm/editor/collapse.py` (`_ordered_spec_with_grouping` → `rebuilt_ordered_spec`; add `_pristine_basis`, `reference_feature_spec`, `_fitted_level_label`; update the two existing callers)
- Modify: `src/superglm/editor/session.py`, `widget.py`, `server.py`, `payloads.py`
- Test: `tests/test_editor_structure.py`

**Interfaces:**
- Produces: `collapse.rebuilt_ordered_spec(spec, *, grouping, base, data, basis=None) -> OrderedCategorical` — a fresh, unfitted spec; `basis=None` clones the pristine declared basis.
- Produces: `collapse.reference_feature_spec(model, term: EditableTerm, level: str, *, X) -> (spec, metadata)`.
- Produces: `EditorSession.replace_with_reference_level(term: str, level: str, **refit_kwargs)`.
- Produces: `EditorWidget._set_reference(term: str, level: str, method: str = "auto", *, level_display="expanded")`.
- Produces: route `POST /set_reference {term, level, method?, level_display}`.
- Produces: per-term payload `"reference": {"level": str, "policy": "most_exposed" | "first" | "pinned"} | None`.

- [ ] **Step 1: Write the failing tests**

```python
from superglm.editor.errors import EditorValueError
from superglm.editor.payloads import session_payload


def test_set_reference_keeps_predictions_and_puts_the_level_at_one(region_model):
    model, X = region_model
    session = EditorSession.from_model(model, terms=["region"])
    before = session.model.predict(X)
    session.replace_with_reference_level("region", "C", method="fit")
    region = session.terms["region"]
    assert region.relativity[region.levels.index("C")] == 1.0
    # An unpenalized factor's reference is a reparametrisation: the optimum is
    # unchanged, so predictions agree to the refit's own convergence tolerance.
    np.testing.assert_allclose(session.model.predict(X), before, rtol=10 * model._tol)


def test_set_reference_on_an_ordered_smooth_keeps_predictions():
    rng = np.random.default_rng(20260925)
    bands = [f"B{i}" for i in range(1, 9)]
    band = rng.choice(bands, 800)
    y = 0.3 + 0.05 * np.array([bands.index(b) for b in band]) + rng.normal(0.0, 0.05, 800)
    X = pd.DataFrame({"band": band})
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        spline_penalty=0.1,
        features={"band": OrderedCategorical(order=bands, basis=Spline(kind="ps", k=5), base="first")},
    )
    model.fit(X, y)
    session = EditorSession.from_model(model, terms=["band"])
    before = session.model.predict(X)
    session.replace_with_reference_level("band", "B5", method="fit")
    # Holds because the constant lies in the smoothing penalty's null space
    # (spec §3.4). If this fails, stop and report; do not loosen the tolerance.
    np.testing.assert_allclose(session.model.predict(X), before, rtol=10 * model._tol)


def test_set_reference_changes_the_fit_under_a_selection_penalty(region_model):
    _, X = region_model
    y = region_model[0]._fit_y_ref
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.05,
        features={"region": Categorical(base="first"), "x": Numeric()},
    )
    model.fit(X, y)
    session = EditorSession.from_model(model, terms=["region"])
    # The factor must survive the penalty, or both fits agree trivially. If it
    # is zeroed, lower selection_penalty until it is active; never assert on a
    # dropped factor.
    assert np.ptp(session.terms["region"].original_log_effect) > 0
    before = session.model.predict(X)
    session.replace_with_reference_level("region", "C", method="fit")
    # The hover text says the fit changes here; this pins that it is true.
    assert np.max(np.abs(session.model.predict(X) - before)) > 1e3 * model._tol


def test_set_reference_maps_a_numeric_level_label_to_its_native_value():
    rng = np.random.default_rng(20260926)
    band = rng.choice([1, 2, 3], 400)
    y = 0.5 + 0.1 * band + rng.normal(0.0, 0.05, 400)
    model = SuperGLM(family="gaussian", selection_penalty=0.0, features={"band": Categorical(base="first")})
    model.fit(pd.DataFrame({"band": band}), y)
    session = EditorSession.from_model(model, terms=["band"])
    session.replace_with_reference_level("band", "3", method="fit")
    assert session.model._specs["band"]._base_level == 3


def test_set_reference_on_a_grouped_member_pins_its_group(region_model):
    model, _ = region_model
    session = EditorSession.from_model(model, terms=["region"])
    session.select_levels("region", ["B", "C"])
    session.replace_with_collapsed_levels("region", method="fit")
    session.replace_with_reference_level("region", "B", method="fit")
    assert session.model._specs["region"]._base_level == "B+C"


def test_a_pinned_reference_survives_a_later_collapse(region_model):
    model, _ = region_model
    session = EditorSession.from_model(model, terms=["region"])
    session.replace_with_reference_level("region", "C", method="fit")
    session.select_levels("region", ["A", "B"])
    session.replace_with_collapsed_levels("region", method="fit")
    assert session.model._specs["region"]._base_level == "C"


def test_set_reference_refuses_a_special_level():
    rng = np.random.default_rng(20260927)
    levels = ["0", "1", "2", "3", "4", "5"]
    band = rng.choice(levels, 600)
    y = 0.2 + 0.05 * band.astype(float) + rng.normal(0.0, 0.05, 600)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"band": OrderedCategorical(order=levels, basis=Spline(kind="ps", k=4), specials=["0"])},
    )
    model.fit(pd.DataFrame({"band": band}), y)
    session = EditorSession.from_model(model, terms=["band"])
    with pytest.raises(EditorValueError, match="special level can't be the reference"):
        session.replace_with_reference_level("band", "0", method="fit")
    assert session.structure_history == []


def test_payload_reports_the_reference_and_reanchors_the_original_line(region_model):
    model, _ = region_model
    session = EditorSession.from_model(model, terms=["region", "x"])
    payload = session_payload(session)
    assert payload["region"]["reference"] == {"level": "A", "policy": "first"}
    assert payload["x"]["reference"] is None
    session.replace_with_reference_level("region", "C", method="fit")
    region = session_payload(session)["region"]
    assert region["reference"] == {"level": "C", "policy": "pinned"}
    # A pure reparametrisation: the opened model's curve, re-expressed against
    # the new reference, is the current curve.
    np.testing.assert_allclose(region["original_y"], region["y"], rtol=10 * model._tol)
```

Also add an HTTP test modelled on `test_widget_http_ungroup_levels_returns_transition_envelope` (`tests/test_editor.py`), posting `{"term": "region", "level": "C"}` to `/set_reference` and asserting `timing.operation == "set_reference"`. Add a refusal test modelled on `test_collapse_levels_rejects_terms_used_by_interactions`, matching `used by interaction`.

- [ ] **Step 2: Run and watch them fail**

Run: `uv run pytest tests/test_editor_structure.py -q -k reference`
Expected: FAIL — `AttributeError: ... 'replace_with_reference_level'`.

- [ ] **Step 3: Implement**

`collapse.py`: rename `_ordered_spec_with_grouping` to `rebuilt_ordered_spec`, drop its unused `selected_levels` parameter, make the arguments keyword-only, and add `basis`. Move the pristine-clone block, with its comments, into `_pristine_basis(spec)` unchanged:

```python
def rebuilt_ordered_spec(spec, *, grouping, base, data, basis=None) -> OrderedCategorical:
    """A fresh, unfitted OrderedCategorical like ``spec`` with this grouping and base.

    ``basis`` replaces the inner basis (a transform). By default the pristine
    declared basis is cloned. A fitted spec is never mutated: its resolved base
    is sticky and would silently survive a changed ``base``.
    """
    values, native_base = _ordered_original_values(spec, grouping, data, base)
    specials = list(spec._special_raw) or list(spec._specials)
    source = _pristine_basis(spec) if basis is None else basis
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=re.escape(_CLAMP_WARNING_PREFIX), category=UserWarning)
        return OrderedCategorical(
            values=values,
            basis=source,
            base=native_base,
            grouping=grouping,
            specials=specials or None,
        )
```

Update the two existing callers (`collapsed_feature_spec`, `ungrouped_feature_spec`) to `rebuilt_ordered_spec(spec, grouping=..., base=base, data=values)`, and any other importer that `rg _ordered_spec_with_grouping src tests` finds. Then add:

```python
def reference_feature_spec(model, term: EditableTerm, level: str, *, X) -> tuple[Any, dict[str, Any]]:
    """A fresh spec for ``term`` whose reference is the displayed ``level``."""
    spec = model._specs[term.name]
    if not isinstance(spec, Categorical | OrderedCategorical):
        raise EditorTypeError(
            f"Set reference is only available for categorical terms, got {term.name!r}."
        )
    _require_not_interaction_parent(model, term.name, operation="set the reference level")
    grouping = getattr(spec, "_grouping", None)
    fitted = _fitted_level_label(spec, grouping, term, level)
    if isinstance(spec, OrderedCategorical):
        frame = as_eager_frame(X)
        frame.require_columns((term.name,))
        replacement = rebuilt_ordered_spec(
            spec, grouping=grouping, base=fitted, data=frame.column_array(term.name)
        )
    else:
        # Fitted levels keep their native type (an integer level stays 3, not "3").
        native = {str(value): value for value in spec._levels}[fitted]
        replacement = Categorical(
            base=native, grouping=grouping, levels=spec._declared_levels, unseen=spec.unseen
        )
    metadata = {
        "format": "superglm.editor.reference_level.v1",
        "term": term.name,
        "level": fitted,
        "label": f"set reference of {term.name} to {fitted}",
        "message": f"The reference level of {term.name} was set to {fitted} and the full model was refit.",
    }
    return replacement, metadata


def _fitted_level_label(spec, grouping, term: EditableTerm, level: str) -> str:
    """The fitted level that carries the reference for a displayed ``level``."""
    if level not in (term.levels or []):
        raise EditorValueError(f"{level!r} is not a level of term {term.name!r}.")
    if isinstance(spec, OrderedCategorical) and level in {str(s) for s in spec._specials}:
        raise EditorValueError(
            f"A special level can't be the reference of {term.name!r}; choose an ordered level."
        )
    return level if grouping is None else str(grouping.original_to_group.get(level, level))
```

`session.py`:

```python
    def replace_with_reference_level(self, term: str, level: str, **refit_kwargs: Any):
        """Pin ``level`` as ``term``'s reference, refit, and put the refit in force."""
        editable = self._require_term(term)
        refit_model = self._refit_replacing(
            term,
            lambda X_ref: reference_feature_spec(self.model, editable, level, X=X_ref),
            **refit_kwargs,
        )
        return self._push_structure(
            refit_model, operation="set_reference", term=term, label=refit_model._editor_step["label"]
        )
```

`widget.py`:

```python
    def _set_reference(self, term: str, level: str, method: str = "auto", *, level_display="expanded"):
        return self._structural_step(
            "set_reference",
            lambda target: self.session.replace_with_reference_level(target, level, method=method),
            term=term,
            level_display=level_display,
        )
```

`server.py`:

```python
    @app.post("/set_reference")
    def set_reference(payload: dict[str, Any] = Body(default_factory=dict)) -> Response:
        return _guarded_json(
            lambda: widget._set_reference(
                str(_required(payload, "term")),
                str(_required(payload, "level")),
                str(payload.get("method", "auto")),
                level_display=_level_display(payload),
            )
        )
```

`payloads.py`: add `"reference": _reference_payload(session, name)` to `term_payload`, and re-anchor the opened model's curve for level terms:

```python
def _reference_payload(session, name: str) -> dict[str, str] | None:
    spec = session.model._specs[name]
    level = getattr(spec, "_base_level", "")
    if level == "":
        return None
    policy = spec.base if spec.base in {"most_exposed", "first"} else "pinned"
    return {"level": str(level), "policy": policy}
```

In `_reference_log_effect`, replace the level branch with:

```python
    if term.levels is not None and reference.levels is not None:
        by_level = {level: float(values[i]) for i, level in enumerate(reference.levels)}
        current = _reference_payload(session, name)
        # Express the opened model's curve against the CURRENT reference; a
        # reference that is a new group label has no value in the opened model.
        anchor = by_level.get(current["level"], 0.0) if current else 0.0
        return np.array(
            [
                by_level[level] - anchor if level in by_level else term.original_log_effect[i]
                for i, level in enumerate(term.levels)
            ]
        )
```

- [ ] **Step 4: Run**

Run: `uv run pytest -n 14 tests/test_editor_structure.py tests/test_editor.py -q`
Expected: PASS. (Existing collapse tests exercise `rebuilt_ordered_spec` through its callers.)

- [ ] **Step 5: Mutation check.** Replace the fresh `Categorical(...)` in `reference_feature_spec` with `copy.deepcopy(spec)` and `.base = native`. `test_set_reference_keeps_predictions_and_puts_the_level_at_one` must fail, because the sticky fitted base keeps "A". Restore.

- [ ] **Step 6: Commit**

```bash
git add src/superglm/editor/{collapse,session,widget,server,payloads}.py tests/test_editor_structure.py
git commit -m "Add Set reference and refit on the shared structural path"
```

---

### Task 4: Transform and refit (Python)

**Files:**
- Create: `src/superglm/editor/transform.py`
- Modify: `src/superglm/editor/session.py`, `widget.py`, `server.py`, `payloads.py`
- Test: `tests/test_editor_structure.py`

**Interfaces:**
- Produces: `transform.transformed_feature_spec(model, term: EditableTerm, *, form: str, breaks: list, degrees: list[int] | None = None, degree: int | None = None, X) -> (spec, metadata)`.
- Produces: `transform.EDITOR_BREAKS_ATTRIBUTE = "_editor_chosen_breaks"` — set to `True` on the basis whose breaks the editor placed (Piecewise or knotted Spline). Read by name in Task 5.
- Produces: `transform.transform_payload(spec, term: EditableTerm) -> {"axis": list[str] | None, "piecewise": {"breaks": list, "degrees": list[int]} | None} | None` — `None` for terms that cannot be transformed. `axis` is the ordered band axis (displayed labels without specials) or `None` on a numeric axis.
- Produces: `EditorSession.replace_with_transformed_term(term, *, form, breaks=(), degrees=None, degree=None, **refit_kwargs)`.
- Produces: `EditorWidget._transform_term(term, *, form, breaks, degrees=None, degree=None, method="auto", level_display="expanded")`.
- Produces: route `POST /transform_term {term, form, breaks, degrees?, degree?, method?, level_display}`.
- Produces: per-term payload `"transform"` from `transform_payload`.

- [ ] **Step 1: Write the failing tests**

```python
from superglm import Constraint, Piecewise

BANDS = [f"B{i}" for i in range(1, 9)]


@pytest.fixture
def banded():
    rng = np.random.default_rng(20260928)
    band = rng.choice(BANDS, 1200)
    x = rng.uniform(0.0, 10.0, 1200)
    kink = np.array([min(BANDS.index(b), 4) for b in band]) * 0.08
    y = 0.3 + kink + 0.04 * x + rng.normal(0.0, 0.05, 1200)
    X = pd.DataFrame({"band": band, "x": x})
    features = {
        "band": OrderedCategorical(order=BANDS, basis=Spline(kind="ps", k=5), base="first"),
        "x": Spline(n_knots=8),
    }
    model = SuperGLM(family="gaussian", selection_penalty=0.0, spline_penalty=0.1, features=features)
    model.fit(X, y)
    return model, X, y


TRANSFORM_CASES = [
    ("band", dict(form="piecewise", breaks=["B3", "B6"], degrees=[1, 2, 0]),
     lambda: OrderedCategorical(order=BANDS, basis=Piecewise(breaks=["B3", "B6"], degrees=[1, 2, 0]), base="first")),
    ("band", dict(form="spline", breaks=["B3", "B6"]),
     lambda: OrderedCategorical(order=BANDS, basis=Spline(kind="ps", knots=["B3", "B6"]), base="first")),
    ("band", dict(form="polynomial", breaks=[], degree=2),
     lambda: OrderedCategorical(order=BANDS, basis=Polynomial(degree=2), base="first")),
    ("x", dict(form="piecewise", breaks=[3.0, 6.5]), lambda: Piecewise(breaks=[3.0, 6.5])),
    ("x", dict(form="spline", breaks=[3.0, 6.5]), lambda: Spline(kind="ps", knots=[3.0, 6.5])),
    ("x", dict(form="polynomial", breaks=[], degree=3), lambda: Polynomial(degree=3)),
]


@pytest.mark.parametrize(("term", "request_", "expected_spec"), TRANSFORM_CASES)
def test_transformed_model_predicts_like_a_direct_fit_of_the_same_spec(banded, term, request_, expected_spec):
    model, X, y = banded
    session = EditorSession.from_model(model, terms=[term])
    session.replace_with_transformed_term(term, method="fit", **request_)
    features = {
        "band": OrderedCategorical(order=BANDS, basis=Spline(kind="ps", k=5), base="first"),
        "x": Spline(n_knots=8),
    }
    features[term] = expected_spec()
    direct = SuperGLM(family="gaussian", selection_penalty=0.0, spline_penalty=0.1, features=features)
    direct.fit(X, y)
    # Same spec, same data, same solver: agreement to the fit tolerance checks
    # the whole replacement path against an independent construction.
    np.testing.assert_allclose(session.model.predict(X), direct.predict(X), rtol=10 * model._tol)
    assert session.structure_history[-1].operation == "transform_term"


def test_transform_carries_a_monotone_constraint_to_a_spline_and_refuses_it_elsewhere(banded):
    _, X, y = banded
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        spline_penalty=0.1,
        features={"x": Spline(n_knots=8, constraint=Constraint.fit.increasing)},
    )
    model.fit(X, y)
    session = EditorSession.from_model(model, terms=["x"])
    for form in ("piecewise", "polynomial"):
        with pytest.raises(EditorValueError, match="carries the increasing constraint"):
            session.replace_with_transformed_term(
                "x", form=form, breaks=[5.0] if form == "piecewise" else [], degree=2, method="fit"
            )
    assert session.structure_history == []
    session.replace_with_transformed_term("x", form="spline", breaks=[5.0], method="fit")
    assert session.model._specs["x"].constraint_kind == "increasing"


INVALID_REQUESTS = [
    ("x", dict(form="piecewise", breaks=[11.0]), "inside the fitted range"),
    ("x", dict(form="piecewise", breaks=[5.0, 5.0]), "strictly increasing"),
    ("x", dict(form="piecewise", breaks=[6.0, 3.0]), "strictly increasing"),
    ("x", dict(form="piecewise", breaks=[]), "Add at least one break"),
    ("x", dict(form="piecewise", breaks=[5.0], degrees=[2, 1]), "straight lines"),
    ("x", dict(form="polynomial", breaks=[5.0], degree=2), "has no breaks"),
    ("x", dict(form="polynomial", breaks=[], degree=6), "degree from 1 to 5"),
    ("band", dict(form="piecewise", breaks=["B1"], degrees=[1, 1]), "interior band"),
    ("band", dict(form="piecewise", breaks=["B8"], degrees=[1, 1]), "interior band"),
    ("band", dict(form="piecewise", breaks=["B9"], degrees=[1, 1]), "interior band"),
    ("band", dict(form="piecewise", breaks=["B6", "B3"], degrees=[1, 1, 1]), "strictly increasing"),
    ("band", dict(form="piecewise", breaks=["B3"], degrees=[1]), "one degree per segment"),
    ("band", dict(form="bumps", breaks=["B3"]), "piecewise, spline or polynomial"),
]


@pytest.mark.parametrize(("term", "request_", "message"), INVALID_REQUESTS)
def test_transform_refuses_invalid_requests_without_changing_anything(banded, term, request_, message):
    model, _, _ = banded
    session = EditorSession.from_model(model, terms=[term])
    revision = session.model_revision
    with pytest.raises(EditorValueError, match=message):
        session.replace_with_transformed_term(term, method="fit", **request_)
    assert session.model is model
    assert session.model_revision == revision
    assert session.structure_history == []


def test_a_shape_the_library_refuses_reaches_the_analyst_as_a_fixed_message(banded):
    model, _, _ = banded
    session = EditorSession.from_model(model, terms=["band"])
    with pytest.raises(EditorValueError, match="could not fit this shape"):
        session.replace_with_transformed_term(
            "band", form="piecewise", breaks=["B3", "B6"], degrees=[0, 0, 0], method="fit"
        )


def test_payload_offers_the_current_breaks_for_editing(banded):
    model, _, _ = banded
    session = EditorSession.from_model(model, terms=["band", "x"])
    payload = session_payload(session)
    assert payload["band"]["transform"] == {"axis": BANDS, "piecewise": None}
    assert payload["x"]["transform"] == {"axis": None, "piecewise": None}
    session.replace_with_transformed_term("band", form="piecewise", breaks=["B3", "B6"], degrees=[1, 2, 0], method="fit")
    session.replace_with_transformed_term("x", form="piecewise", breaks=[3.0, 6.5], method="fit")
    payload = session_payload(session)
    assert payload["band"]["transform"]["piecewise"] == {"breaks": ["B3", "B6"], "degrees": [1, 2, 0]}
    assert payload["x"]["transform"]["piecewise"] == {"breaks": [3.0, 6.5], "degrees": [1, 1, 1]}
```

Add an HTTP test posting `{"term": "x", "form": "piecewise", "breaks": [3.0, 6.5]}` to `/transform_term` (envelope, `timing.operation == "transform_term"`), and one posting `{"term": "x", "form": "piecewise", "breaks": [true]}` that expects status 400 with the message `breaks must be a list of band names or numbers.`

- [ ] **Step 2: Run and watch them fail**

Run: `uv run pytest tests/test_editor_structure.py -q -k transform`
Expected: FAIL — `AttributeError: ... 'replace_with_transformed_term'`.

- [ ] **Step 3: Implement `transform.py`**

```python
"""Give one term a stated shape: piecewise polynomial, spline at breaks, or polynomial."""

from __future__ import annotations

from typing import Any

import numpy as np

from superglm._frame import as_eager_frame
from superglm.editor._types import EditableTerm
from superglm.editor.collapse import _require_not_interaction_parent, rebuilt_ordered_spec
from superglm.editor.errors import EditorTypeError, EditorValueError
from superglm.features.constraint import ConstraintSpec
from superglm.features.numeric import Numeric
from superglm.features.ordered_categorical import OrderedCategorical, _spline_kind_name
from superglm.features.piecewise import Piecewise
from superglm.features.polynomial import Polynomial
from superglm.features.spline import Spline, _SplineBase

FORMS = ("piecewise", "spline", "polynomial")
MAX_SEGMENT_DEGREE = 3
MAX_POLYNOMIAL_DEGREE = 5
# Read by model/report_ops.py BY NAME, so the model layer never imports the
# editor: a basis carrying it had its breaks placed in the editor from this
# data, so its tests are conditional on them (spec §3.3).
EDITOR_BREAKS_ATTRIBUTE = "_editor_chosen_breaks"
_NUMERIC_KINDS = (_SplineBase, Numeric, Polynomial, Piecewise)


def transformed_feature_spec(model, term: EditableTerm, *, form, breaks, degrees=None, degree=None, X):
    """A fresh spec that gives ``term`` the requested shape, with the step's metadata."""
    spec = model._specs[term.name]
    ordered = isinstance(spec, OrderedCategorical)
    if not ordered and not isinstance(spec, _NUMERIC_KINDS):
        raise EditorTypeError(f"Breaks need an ordered or numeric axis; {term.name!r} has neither.")
    _require_not_interaction_parent(model, term.name, operation="transform")
    if form not in FORMS:
        raise EditorValueError(f"Choose a piecewise, spline or polynomial form, got {form!r}.")
    source = _source_spline(spec)
    if form == "polynomial":
        basis = _polynomial(term.name, source, breaks, degree)
    else:
        axis = _break_axis(spec, term) if ordered else None
        _validate_breaks(term, breaks, axis)
        basis = _piecewise(term, spec, source, axis, breaks, degrees) if form == "piecewise" else _knotted_spline(source, breaks)
        setattr(basis, EDITOR_BREAKS_ATTRIBUTE, True)
    replacement = _hosted(spec, basis, term.name, X) if ordered else basis
    return replacement, _metadata(term.name, form, breaks, degrees, degree)
```

The helpers, each one job:
- `_source_spline(spec)` returns the spline the term is fitted with — `spec._spline_obj` for an ordered term whose basis is a `_SplineBase`, `spec` itself for a numeric spline — else `None`.
- `_break_axis(spec, term)` returns `[level for level in term.levels if level not in {str(s) for s in spec._specials}]`.
- `_validate_breaks(term, breaks, axis)` raises `EditorValueError` with exactly these messages:
  - `"Add at least one break."` when empty.
  - On an ordered axis: `"Breaks must be interior bands of {name} (not its first or last band)."` when any break is not in `axis[1:-1]`; `"Breaks must be strictly increasing along {name}'s bands."` when their axis positions are not strictly increasing.
  - On a numeric axis: `"Breaks must be finite numbers strictly inside the fitted range {lo:g} to {hi:g}."` (with `lo, hi = min(term.x), max(term.x)`) when any break is non-finite or outside; `"Breaks must be strictly increasing."` when `np.diff` is not all positive.
- `_polynomial(name, source, breaks, degree)` first calls `_refuse_constraint(name, source, "polynomial")`. It then raises `"A polynomial has no breaks; clear them or choose another form."` when breaks are given, and `"Choose a polynomial degree from 1 to 5."` unless `degree` is an int in `[1, MAX_POLYNOMIAL_DEGREE]`. It returns `Polynomial(degree=degree)`.
- `_piecewise(term, spec, source, axis, breaks, degrees)` first calls `_refuse_constraint(term.name, source, "piecewise")`.
  - **Numeric axis:** `degrees` must be `None` or all 1, else `"On a numeric feature, piecewise segments are straight lines; per-segment degrees need an ordered term."`. Returns `Piecewise(breaks=list(breaks), extrapolation=getattr(spec, "extrapolation", "clip"))`.
  - **Ordered axis:** `degrees=None` means every segment linear (`[1] * (len(breaks) + 1)`). Otherwise `degrees` must have `len(breaks) + 1` integer entries, else `"State one degree per segment: {n} breaks make {n + 1} segments."`, and each entry must be in `[0, MAX_SEGMENT_DEGREE]`, else `"Segment degrees run from 0 (flat) to 3 (cubic)."`. Returns `Piecewise(breaks=list(breaks), degrees=list(degrees))`.
  - All-flat and consecutive-flat segments stay the library's call; the session wrapper below turns that refusal into a fixed message.
- `_refuse_constraint(name, source, form)` raises `f"{name} carries the {source.constraint_kind} constraint, which a {form} can't hold. Choose Spline, or remove the constraint first."` when `source` is not `None` and has a `constraint_kind`.
- `_knotted_spline(source, knots)` returns `Spline(kind="cr", knots=list(knots))` when `source` is `None`. Otherwise:

```python
    constraint = (
        None
        if source.constraint_kind is None
        else ConstraintSpec(mode=source.constraint_mode, kind=source.constraint_kind)
    )
    return Spline(
        kind=_spline_kind_name(source),
        knots=list(knots),
        degree=source.degree,
        penalty=source.penalty,
        select=source.select,
        extrapolation=source.extrapolation,
        discrete=source.discrete,
        n_bins=source.n_bins,
        m=source._m_orders if len(source._m_orders) > 1 else source._m_orders[0],
        constraint=constraint,
    )
```

- `_hosted(spec, basis, name, X)` returns `rebuilt_ordered_spec(spec, grouping=getattr(spec, "_grouping", None), base=spec.base, data=<column of X>, basis=basis)`.
- `_metadata(name, form, breaks, degrees, degree)` returns `{"format": "superglm.editor.term_transform.v1", "term", "form", "breaks", "degrees", "degree", "label", "message"}`. The labels are:
  - `f"transform {name} to piecewise ({n} break{'s' if n != 1 else ''})"`
  - `f"transform {name} to spline, knots at {n} break{'s' if n != 1 else ''}"`
  - `f"transform {name} to polynomial, degree {degree}"`

  The message is `f"{name} was transformed in the editor and the full model was refit."`.

`transform_payload(spec, term)`: returns `None` unless the spec is ordered or one of `_NUMERIC_KINDS`; otherwise `{"axis": _break_axis(spec, term) if ordered else None, "piecewise": _current_piecewise(spec, term)}`. `_current_piecewise` returns:
- for a numeric `Piecewise`: `{"breaks": [float(k) for k in spec._knots[1:-1]], "degrees": [1] * (len(spec._knots) - 1)}`;
- for an ordered spec whose `_spline_obj` is a `Piecewise` with stated breaks: `{"breaks": [b if isinstance(b, str) else axis[int(b)] for b in basis.breaks], "degrees": list(basis.degrees or [1] * (len(basis.breaks) + 1))}`;
- `None` otherwise, including int-mode breaks, which have no stated positions.

- [ ] **Step 4: Session, widget, route and payload**

`session.py`:

```python
    def replace_with_transformed_term(self, term: str, *, form: str, breaks=(), degrees=None, degree=None, **refit_kwargs: Any):
        """Give ``term`` a new shape, refit, and put the refit in force."""
        editable = self._require_term(term)
        try:
            refit_model = self._refit_replacing(
                term,
                lambda X_ref: transformed_feature_spec(
                    self.model, editable, form=form, breaks=list(breaks), degrees=degrees, degree=degree, X=X_ref
                ),
                **refit_kwargs,
            )
        except EditorClientError:
            raise
        except ValueError as exc:
            # The library's refusal text is backend text (editor/errors.py): the
            # analyst gets one intentional sentence, Python callers keep the cause.
            raise EditorValueError(_TRANSFORM_REFUSED) from exc
        return self._push_structure(
            refit_model, operation="transform_term", term=term, label=refit_model._editor_step["label"]
        )
```

with `_TRANSFORM_REFUSED = "SuperGLM could not fit this shape. Check that no segment is flat next to another flat one, that each segment has enough bands or data, and that no collapsed group spans a break."`

`widget.py`:

```python
    def _transform_term(self, term, *, form, breaks, degrees=None, degree=None, method="auto", level_display="expanded"):
        return self._structural_step(
            "transform_term",
            lambda target: self.session.replace_with_transformed_term(
                target, form=form, breaks=breaks, degrees=degrees, degree=degree, method=method
            ),
            term=term,
            level_display=level_display,
        )
```

`server.py`: add the route plus two parsers beside `_int` / `_float`:

```python
    @app.post("/transform_term")
    def transform_term(payload: dict[str, Any] = Body(default_factory=dict)) -> Response:
        return _guarded_json(
            lambda: widget._transform_term(
                str(_required(payload, "term")),
                form=str(_required(payload, "form")),
                breaks=_break_list(payload.get("breaks", [])),
                degrees=_optional_int_list(payload.get("degrees"), "degrees"),
                degree=_optional_int(payload.get("degree"), "degree"),
                method=str(payload.get("method", "auto")),
                level_display=_level_display(payload),
            )
        )


def _break_list(value: Any) -> list[str | float]:
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, str | int | float) for item in value
    ):
        raise EditorValueError("breaks must be a list of band names or numbers.")
    return [item if isinstance(item, str) else float(item) for item in value]


def _optional_int_list(value: Any, name: str) -> list[int] | None:
    return None if value is None else [_int(item, name) for item in _list(value, name)]
```

(`_list(value, name)` raises `EditorValueError(f"{name} must be a list.")` for a non-list. Reuse an existing helper if `server.py` has one.)

`payloads.py`: `term_payload["transform"] = transform_payload(session.model._specs[name], term)`.

- [ ] **Step 5: Run**

Run: `uv run pytest -n 14 tests/test_editor_structure.py tests/test_editor.py tests/test_piecewise_editor.py -q`
Expected: PASS. If an oracle case fails, report the numbers; never loosen the tolerance.

- [ ] **Step 6: Mutation checks.**
  1. In `_knotted_spline`, drop `constraint=constraint`; the monotone test must fail.
  2. Remove the `except ValueError` conversion; the fixed-message test must fail (a raw library `ValueError` escapes).

  Restore both.

- [ ] **Step 7: Commit**

```bash
git add src/superglm/editor/transform.py src/superglm/editor/{session,widget,server,payloads}.py tests/test_editor_structure.py
git commit -m "Add Transform and refit: piecewise polynomial, spline at breaks, polynomial"
```

---

### Task 5: Revert to original model; the breaks note in every renderer

**Files:**
- Modify: `src/superglm/editor/session.py`, `widget.py`, `server.py`, `summaries.py`
- Modify: `src/superglm/model/report_ops.py` (next to the `_editor_offset` block, line ~245)
- Modify: `src/superglm/inference/summary.py` (`_editor_notes`, line ~1349)
- Modify: `src/superglm/export/summary.py` (the inline editor-note block, line ~445; the duplicate constants at lines ~85–93)
- Test: `tests/test_editor_structure.py`

**Interfaces:**
- Produces: `EditorSession.revert_to_reference_model() -> model`; `EditorWidget._revert_to_original(*, level_display="expanded")`; route `POST /revert_to_original {level_display}`.
- Produces: `model_info["editor_break_terms"]: list[str]` and `inference.summary.editor_break_notes(info) -> list[str]`.

- [ ] **Step 1: Write the failing tests**

```python
import joblib
import io


def test_revert_returns_the_opened_model_and_clears_every_history(region_model):
    model, X = region_model
    session = EditorSession.from_model(model, terms=["region", "x"])
    session.select_levels("region", ["B", "C"])
    session.replace_with_collapsed_levels("region", method="fit")
    session.select_indices("x", [3, 4])
    session.shift("x", 0.1)
    session.revert_to_reference_model()
    assert session.model is model
    assert session.structure_history == [] and session.history == [] and session.redo_stack == []
    np.testing.assert_array_equal(session.to_model().predict(X), model.predict(X))


def test_restore_walks_a_mixed_two_term_sequence_back_exactly(banded):
    model, X, _ = banded
    territory = np.random.default_rng(20260929).choice(["T1", "T2", "T3", "T4"], len(X))
    X = X.assign(territory=territory)
    y = model._fit_y_ref + 0.05 * (territory == "T3")
    two_term = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        spline_penalty=0.1,
        features={
            "band": OrderedCategorical(order=BANDS, basis=Spline(kind="ps", k=5), base="first"),
            "territory": Categorical(base="first"),
        },
    )
    two_term.fit(X, y)
    session = EditorSession.from_model(two_term, terms=["band", "territory"])
    before_each_step = [session.model.predict(X)]
    session.select_levels("territory", ["T1", "T2"])
    session.replace_with_collapsed_levels("territory", method="fit")
    before_each_step.append(session.model.predict(X))
    session.replace_with_transformed_term("band", form="piecewise", breaks=["B3", "B6"], degrees=[1, 1, 1], method="fit")
    before_each_step.append(session.model.predict(X))
    session.replace_with_reference_level("territory", "T3", method="fit")
    before_each_step.append(session.model.predict(X))
    session.select_levels("territory", ["T1", "T2"])
    session.replace_with_ungrouped_levels("territory", method="fit")
    for expected in reversed(before_each_step):
        session.uncollapse_levels()
        np.testing.assert_array_equal(session.model.predict(X), expected)
    assert not session.can_uncollapse_levels()


def test_breaks_note_reaches_every_renderer_and_survives_export(banded):
    model, X, _ = banded
    session = EditorSession.from_model(model, terms=["band", "x"])
    session.replace_with_transformed_term("band", form="piecewise", breaks=["B3", "B6"], degrees=[1, 2, 0], method="fit")
    sentence = "Breaks for band were placed in the editor from this data."
    assert sentence in str(session.model.summary())
    # A later step on another term keeps the note: the mark lives on the basis.
    session.replace_with_transformed_term("x", form="polynomial", breaks=[], degree=2, method="fit")
    edited = session.to_model()
    assert edited.summary()._info["editor_break_terms"] == ["band"]
    buffer = io.BytesIO()
    joblib.dump(edited, buffer)
    buffer.seek(0)
    assert joblib.load(buffer).summary()._info["editor_break_terms"] == ["band"]
```

Also assert the workbook's Model Summary notes (the builder in `export/summary.py` that produces the notes tuple around line 445) contain the same sentence, and that the editor's compact summary `note` (`summary_payload(widget, "in_force")["note"]`) contains it. Add a negative check: after only a polynomial transform or a set-reference, `"editor_break_terms"` is absent.

- [ ] **Step 2: Run and watch them fail**

Run: `uv run pytest tests/test_editor_structure.py -q -k "revert or mixed or breaks_note"`
Expected: FAIL.

- [ ] **Step 3: Implement**

`session.py`:

```python
    def revert_to_reference_model(self):
        """Put the opened model back in force and clear every history and display reorder."""
        self._level_orders = {}
        self.replace_in_force_model(self.reference_model)
        self.structure_history.clear()
        return self.reference_model
```

`widget.py`:

```python
    def _revert_to_original(self, *, level_display="expanded"):
        return self._structural_step(
            "revert_to_original",
            lambda _target: self.session.revert_to_reference_model(),
            level_display=level_display,
        )
```

`server.py`: a `/revert_to_original` route shaped like `/restore_structure`.

`report_ops.py`, after the `_editor_offset` block:

```python
    # The editor marks a basis whose breaks it placed (editor/transform.py,
    # EDITOR_BREAKS_ATTRIBUTE); read by name so this layer never imports the editor.
    break_terms = [name for name, spec in model._specs.items() if _editor_chose_breaks(spec)]
    if break_terms:
        model_info["editor_break_terms"] = break_terms
```

```python
def _editor_chose_breaks(spec) -> bool:
    return bool(getattr(getattr(spec, "_spline_obj", spec), "_editor_chosen_breaks", False))
```

`inference/summary.py`:

```python
_EDITOR_BREAKS_NOTE = (
    "were placed in the editor from this data. Tests are conditional on those "
    "breaks; judge them on validation deviance."
)


def editor_break_notes(info: dict[str, Any]) -> list[str]:
    """The note for terms whose breaks were placed in the editor."""
    terms = info.get("editor_break_terms") or []
    return [f"Breaks for {', '.join(terms)} {_EDITOR_BREAKS_NOTE}"] if terms else []
```

`_editor_notes` ends with `notes.extend(editor_break_notes(info))`.

`export/summary.py`: replace the inline stale/offset note appends with `notes.extend(_editor_notes(info))` (import `_editor_notes` from `superglm.inference.summary`; keep the `inference_stale` variable for the later branch), and delete the now-unused duplicate `_EDITOR_STALE_NOTE` / `_EDITOR_OFFSET_NOTE` constants if nothing else in the file references them. This deletes the duplicate rather than adding a third copy.

`summaries.py`, in `summary_payload`: `"note": " ".join([_summary_note(source), *editor_break_notes(getattr(summary, "_info", {}))])`.

- [ ] **Step 4: Run**

Run: `uv run pytest -n 14 tests/test_editor_structure.py tests/test_editor.py tests/test_summary*.py tests/test_export*.py -q`
Expected: PASS. Existing summary and workbook tests pin the note text, so they prove the export consolidation left their output unchanged.

- [ ] **Step 5: Mutation checks.**
  1. Drop `setattr(basis, EDITOR_BREAKS_ATTRIBUTE, True)`; the note test must fail.
  2. In `revert_to_reference_model`, drop `self.structure_history.clear()`; the revert test must fail.

  Restore both.

- [ ] **Step 6: Commit**

```bash
git add src/superglm/editor/{session,widget,server,summaries}.py src/superglm/model/report_ops.py src/superglm/inference/summary.py src/superglm/export/summary.py tests/test_editor_structure.py
git commit -m "Add Revert to original model and the editor-chosen breaks note"
```

---

### Task 6: A solid original-model line

**Files:**
- Modify: `src/superglm/editor/app/styles.css` (`.original`, line ~790)
- Modify: `src/superglm/plotting/editor_style.py` (`"original"`, line ~73)
- Modify: `tests/test_lss_editor_style.py` (the `"original"` grammar mapping, line ~43)

- [ ] **Step 1: Update the parity test first.** In `tests/test_lss_editor_style.py`, the `original` entry maps `color → stroke`, `alpha → stroke-opacity`, `width → stroke-width`, and no dash. Run `uv run pytest tests/test_lss_editor_style.py -q`; expect FAIL against the dashed style.
- [ ] **Step 2: Change the style.** `styles.css`: `.original { fill: none; stroke: #8c959f; stroke-opacity: 0.5; stroke-width: 3; }`. `editor_style.py`: `"original": dict(color=(140, 149, 159), alpha=0.5, width=3.0),`. Check every reader of this style (`rg '"original"' src/superglm/plotting`) for an unconditional `["dash"]` read and switch it to `.get("dash")` only where one exists.
- [ ] **Step 3: Run** `uv run pytest -n 14 tests/test_lss_editor_style.py tests/test_lss_plotting.py -q`. Expect PASS.
- [ ] **Step 4: Commit**

```bash
git add src/superglm/editor/app/styles.css src/superglm/plotting/editor_style.py tests/test_lss_editor_style.py
git commit -m "Draw the original-model line solid, wider and semi-transparent"
```

---

### Task 7: Browser — Refresh, Revert, Restore, Set reference, reference chip

**Files:**
- Modify: `src/superglm/editor/app/index.html`, `app/views/app_bar.js`, `app/views/context_bar.js`, `app/views/structural_confirm.js`, `app/views/help_content.js`, `app/state/actions.js`, `app/summary.js`, `app/main.js`, `app/api/contracts.js`, `app/styles/shell.css`
- Test: `tests/editor_frontend/{app_bar,structural_confirm,actions,summary}.test.js`

**Interfaces:**
- Consumes: `structure_history`, per-term `reference`; routes `/set_reference`, `/revert_to_original`, `/restore_structure`; `restoreTransition()`.
- Produces: `actions.refreshFromPython(): Promise<ActionResult>`; `setReferenceTransition(term, level)`, `revertTransition()` in `summary.js`; typedef `TermReference {level:string, policy:'most_exposed'|'first'|'pinned'}` and `TermPayload.reference: TermReference|null`.

Use the frontend-design skill; keep to the existing visual language (tokens in `styles/tokens.css`, icon strokes as in `index.html`).

- [ ] **Step 1: Write the failing node tests.**
  - `actions.test.js`: (a) `refreshFromPython` commits a snapshot with an equal `model_revision` but a changed selection, and calls `scheduleVisibleEvidence(revision, { immediate: true })`; (b) it returns `{ ok: false, skipped: true }` while a mutation is running; (c) a malformed snapshot leaves state unchanged and returns `ok: false`.
  - `structural_confirm.test.js`: (a) `"revert to original model"` requires confirmation when there are manual edits *or* structural steps, with the message `Revert to the original model? This clears 3 manual edits and 2 structural steps, and can't be undone.`; (b) singular forms read `1 manual edit` and `1 structural step`; (c) `"set reference and refit"` behaves like the other structural operations.
  - `app_bar.test.js`: Refresh is disabled while busy; Revert is enabled only when edits or structural steps exist.
  - `summary.test.js`: the descriptors are `{name: "set reference and refit", path: "/set_reference", payload: {term, level, method: "auto"}}` and `{name: "revert to original model", path: "/revert_to_original", payload: {}}`.

  Run `npm run test:frontend`; expect FAIL.
- [ ] **Step 2: Markup** (`index.html`):
  - **App bar,** after Redo: `#revertAction` (aria-label "Revert to original model"; popover "Revert to original model" / "Go back to the model the editor was opened with. Clears every edit and structural step.") and `#refreshAction` (aria-label "Refresh from Python"; popover "Refresh from Python" / "Re-read the Python session and redraw. Use it after changing the session in the notebook. Nothing refits.").
  - **Chart action bar,** right side: `#restoreStructure` (aria-label "Restore previous structure"; popover title "Restore previous structure"; body set at render time to `Undo: <last label>`).
  - **Selection palette,** after Ungroup: `#setReference` (`class="selection-item"`, `data-help-operation="set_reference"`, aria-label "Set reference and refit", `hidden`). Delete `#uncollapseLevels` and its `main.js` code.
  - **Context bar,** after `#termEdf`: `<span id="termReference" class="context-chip" hidden></span>`.
  - **Icons** (24×24, stroke): revert = counter-clockwise arrow into a baseline; refresh = two opposing arcs with arrowheads; restore = counter-clockwise arrow; set reference = a pin on a baseline. Use the spec mockup's shapes as the starting point.
- [ ] **Step 3: Behaviour.**
  - `actions.refreshFromPython()`:

```js
  /** @returns {Promise<ActionResult>} */
  async function refreshFromPython() {
    if (store.getState().request.mutation.status === "running") {
      return skippedMutation("An editor mutation is already running.");
    }
    let candidate;
    try {
      candidate = await client.getState();
    } catch (value) {
      return { ok: false, error: normalizeError(value) };
    }
    if (!isEditorSnapshot(candidate)) {
      return { ok: false, error: new Error("Python returned an editor state this page cannot read.") };
    }
    const snapshot = /** @type {EditorSnapshot} */ (candidate);
    store.update((state) => commitRemote(state, snapshot));
    void Promise.resolve(scheduleVisibleEvidence(snapshot.model_revision, { immediate: true })).catch(() => {});
    return { ok: true, snapshot };
  }
```

  - After a successful refresh, the status line reads `Synced with Python · revision N`.
  - Revert enabled ⇔ `history.active.length + history.redo.length > 0 || structure_history.depth > 0`. Click → `runStructuralRefit(revertTransition())`.
  - Restore visible ⇔ `structure_history.depth > 0`, with popover body `Undo: ${last.label}`. Click → `runStructuralRefit(restoreTransition())`.
  - Set reference visible ⇔ the active term is a categorical or ordered categorical, exactly one displayed level is selected, and that label ≠ `term.reference.level`. Click → `runStructuralRefit(setReferenceTransition(term, label))`.
  - Reference chip: `reference ${level} · ${{most_exposed: "most exposed", first: "first", pinned: "pinned"}[policy]}`, hidden when `term.reference` is `null`.
  - `structural_confirm.js`: add titles `"set reference and refit": "Set reference"`, `"transform and refit": "Transform"`, `"revert to original model": "Revert to original model"`. For revert, require confirmation when edits + structural steps > 0 and use the message from Step 1.
  - `help_content.js` `OPERATION_HELP`:
    - `set_reference`: "Set reference and refit" / "Pin the selected level as the reference (relativity 1.00) and refit. Predictions stay the same unless a selection penalty is on."
    - `restore_structure`: "Restore previous structure" / "Undo the latest collapse, ungroup, transform or reference change."
    - `revert_to_original`: "Revert to original model" / "Go back to the model the editor was opened with."
    - `refresh_from_python`: "Refresh from Python" / "Re-read the Python session after changing it in the notebook."
  - `main.js` gains only the wiring for the above. Any new rendering logic of more than a few lines goes in the view module that owns the element (`app_bar.js`, `context_bar.js`).
- [ ] **Step 4: Run** `npm run check:frontend`. Expect PASS.
- [ ] **Step 5: Commit**

```bash
git add src/superglm/editor/app tests/editor_frontend
git commit -m "Add Refresh, Revert, Restore and Set reference icons and the reference chip"
```

---

### Task 8: Browser — Breaks mode and Transform and refit

**Files:**
- Create: `src/superglm/editor/app/breaks.js`, `app/chart/svg.js`, `app/chart/break_overlay.js`, `app/views/breaks_controls.js`, `tests/editor_frontend/breaks.test.js`
- Modify: `app/chart.js` (move `el`/`line`/`text` to `chart/svg.js`; one overlay call), `app/interactions.js`, `app/views/tool_rail.js`, `app/state/store.js`, `app/api/contracts.js`, `app/index.html`, `app/views/help_content.js`, `app/summary.js`, `app/main.js`, `app/styles/chart.css`, `app/styles/shell.css`
- Test: `tests/editor_frontend/{breaks,tool_rail,store}.test.js`

**Interfaces:**
- Consumes: per-term `transform: {axis: string[]|null, piecewise: {breaks, degrees}|null} | null`; route `/transform_term`.
- Produces (`breaks.js`): `isTransformable(term)`, `initialDraft(term)`, `snapBreak(term, dataX)`, `breakX(term, value)`, `addBreak(term, draft, value)`, `moveBreak(term, draft, index, value)`, `removeBreak(term, draft, index)`, `cycleDegree(term, draft, segment)`, `maxSegmentDegree(term, draft, segment)`, `setPolynomialDegree(draft, degree)`, `draftProblem(term, draft)`, `formHint(draft)`, `degreeName(degree)`, `transformPayload(termName, term, draft)`; constants `MAX_SEGMENT_DEGREE = 3`, `MAX_POLYNOMIAL_DEGREE = 5`.
- Produces: typedef `BreakDraft {form:'piecewise'|'spline'|'polynomial', breaks:Array<string|number>, degrees:number[], degree:number}` — breaks are band labels on an ordered axis and x values on a numeric axis, exactly as the route takes them.
- Produces: view state `breakDraftByTerm: Record<string, BreakDraft>` and `setBreakDraft(state, term, draft|null)` in `store.js`; `transformTransition(payload)` in `summary.js` with name `"transform and refit"`.

Use the frontend-design skill for the overlay and controls; keep the existing visual language.

- [ ] **Step 1: Write the failing node tests** (`breaks.test.js`). Use a fixture ordered term with `levels ["B1".."B8"]`, `x = [0..7]`, `transform.axis = levels`, and a numeric term with `x` from 0 to 10 and `transform.axis = null`. Cases:
  - `snapBreak` on the ordered term: 2.4 → `"B3"`; 0.1 → `"B2"` (never the first band); 7.0 → `"B7"`.
  - `snapBreak` on the numeric term: 3.14159 → `3.14`; −1 → `null`; 10 → `null`.
  - `addBreak` keeps breaks sorted by axis position, splits the segment it lands in, and copies that segment's degree to both halves (a flat segment's right half becomes linear).
  - `moveBreak` never crosses or lands on a neighbour.
  - `removeBreak` merges the two segments at the higher degree.
  - `cycleDegree` cycles 1 → 2 → 3 → 0 within `maxSegmentDegree`. That cap is `min(3, span)` on an ordered axis, where span is the gap between the segment's break positions, and 1 on a numeric axis.
  - `draftProblem` returns exactly: `"Add at least one break."`, `"Give at least one segment a degree: all-flat is a constant."`, `"Two flat segments in a row make one plateau: remove the break between them."`, or `null`.
  - `transformPayload` sends degrees only for an ordered piecewise draft and `{breaks: [], degree}` for polynomial.
  - `initialDraft` loads `transform.piecewise` when present.

  Add to `tool_rail.test.js`: `b` selects `"breaks"`. Add to `store.test.js`: `setBreakDraft(state, term, null)` deletes the draft.

  Run `npm run test:frontend`; expect FAIL.
- [ ] **Step 2: Implement `breaks.js`** — pure functions only, no DOM. Messages and hints exactly: `formHint` returns `"Click the plot to add a break."` (piecewise), `"Knots sit at the breaks."` (spline), `"One polynomial across the whole axis."` (polynomial). `degreeName` maps 0–3 to `flat`, `linear`, `quadratic`, `cubic`.
- [ ] **Step 3: Chart overlay.**
  - Move `el`, `line` and `text` from `chart.js` into `chart/svg.js` and import them in both places. This is a move, not a copy.
  - `drawBreakOverlay(svg, { term, draft, sx, margin, innerW, innerH })` draws, in order:
    - alternate segment shading (`.break-segment`);
    - per break: a dashed line (`.break-line`), a 16-px transparent hit strip with `data-break-index` (`.break-hit`), and a top label (`.break-label`, `tabindex="0"`, `role="slider"`, `aria-valuetext` = the band or value) holding the band/value text and a remove handle (`.break-remove`, `data-break-remove`, `role="button"`, `aria-label="Remove break at <label>"`);
    - for an ordered piecewise draft only: one degree chip per segment (`.degree-chip`, `data-segment`, `role="button"`, `aria-label="Segment <n>: <degree name>. Change degree"`).
  - Breaks are placed at `sx(breakX(term, value))`.
  - `chart.js` gains one call, after the point layer: `if (visualMode === "breaks" && context.breakDraft) drawBreakOverlay(svg, {...})`.
- [ ] **Step 4: Gestures** (`interactions.js`). Add a `mode === "breaks"` branch in `onPointerDown` / `onPointerMove` / `onPointerUp`:
  - down on `[data-break-remove]` → `removeBreak`;
  - down on `[data-segment]` → `cycleDegree`;
  - down on `[data-break-index]` → start `interaction.breakDrag`;
  - any other down inside the plot → `addBreak(snapBreak(...))`.
  - A drag commits through `context.setBreakDraft` only when the snapped value changes. Snaps are discrete, so there are few commits; a comment says so.
  - The hover ghost line is an imperative SVG element managed here, like `interaction.brush`, and is removed on pointer leave and mode change.
  - Keyboard on a focused `.break-label`: ArrowLeft / ArrowRight move one band (ordered) or one display-grid step (numeric) through `moveBreak`; Delete / Backspace remove the break.
- [ ] **Step 5: Controls, mode, wiring.**
  - `index.html`: the tool-rail button (`data-tool="breaks"`, `aria-keyshortcuts="B"`, the spec mockup's icon), and in the chart action bar a `#breaksControls` group (hidden unless the mode is breaks) containing:
    - the form radiogroup `[data-form]` with Piecewise / Spline / Polynomial and their popovers from spec §4.3;
    - `#polynomialDegreeWrap` with −/+ and an `<output>`;
    - `#breaksHint` (`role="status"`, `aria-live="polite"`);
    - `#clearBreaks` "Clear breaks";
    - `#transformTerm`, an icon button with `aria-label="Transform and refit"`, `data-help-operation="transform_term"`, popover "Transform and refit" / "Replace this term with the form and breaks shown, then refit the model. Restore undoes it."
  - `views/breaks_controls.js`: `bindBreaksControls({root, onForm, onDegree, onClear, onTransform})` and `renderBreaksControls({root, draft, problem, busy})`. Transform is disabled when `problem` is set or `busy` is true. The hint shows `problem ?? formHint(draft)` with a `warn` class when there is a problem. Clear is hidden for polynomial.
  - `tool_rail.js`: add `'breaks'` to `ToolMode` and `b: "breaks"` to the shortcuts. The rail renders the Breaks button disabled when `!isTransformable(activeTerm)`, with popover body `Breaks need an ordered or numeric axis.`. If the active term changes to one that can't be transformed while in breaks mode, switch to select.
  - `main.js`: the chart context gets `breakDraft()`, which returns `view.breakDraftByTerm[term] ?? initialDraft(termPayload)`; interactions get `setBreakDraft(draft)`. The chart re-renders when the active term's draft changes, by adding it to `selectChartRenderState`. After a successful transform, `setBreakDraft(state, term, null)`, so the draft re-derives from the new payload.
  - `help_content.js`: `TOOL_HELP.breaks` "Breaks" / "Click the plot to add a break, drag a break to move it, and click × to remove it. Choose a form, then Transform and refit.", and `OPERATION_HELP.transform_term` with the Transform popover text.
- [ ] **Step 6: Run** `npm run check:frontend`. Expect PASS.
- [ ] **Step 7: Commit**

```bash
git add src/superglm/editor/app tests/editor_frontend
git commit -m "Add the Breaks tool and Transform and refit to the editor"
```

---

### Task 9: Browser tests, user docs, and full verification

**Files:**
- Create: `tests/editor/test_editor_structure_browser.py`
- Modify: `docs/tutorials/edit-a-model-in-the-browser.md`, `docs/development/internals/editor-frontend.md`

- [ ] **Step 1: Browser tests** (fixtures `open_editor_page` / `editor_browser_model` from `tests/editor/conftest.py`; mark `browser`). Coordinate on route responses and DOM predicates, never sleeps:
  - `test_breaks_mode_transforms_an_ordered_term`: open with `selected_term="age_band"`; press `b`; click the plot at two x positions; assert two `.break-line`; click a `.degree-chip` and see its label change; click `#transformTerm`; await `/transform_term`. The Restore icon is visible, and `session.model._specs["age_band"]._spline_obj` is a `Piecewise`.
  - `test_set_reference_icon_needs_exactly_one_level`: on `territory`, one selected level shows `#setReference`, two hide it; clicking with one selected makes the chip read `· pinned`.
  - `test_refresh_pulls_a_notebook_side_structural_change` (Review Focus 5): collapse `T01, T02` through `session` in Python, check the page still shows no Restore, click `#refreshAction`, then Restore is visible and the level-group markers are drawn.
  - `test_revert_confirms_and_returns_to_the_opened_model`: after a collapse, `#revertAction` is enabled; clicking shows the confirm with the step count; confirming hides Restore and disables Revert.
- [ ] **Step 2: Docs.**
  - Tutorial: short plain-language sections "Set a reference level", "Give a term a shape with breaks", "Restore and revert", "Refresh from Python", and one sentence where Restore moved. One limit per bullet; no internal names.
  - `editor-frontend.md`: an "Add a structural operation" recipe — spec builder returning `(spec, metadata-with-label)` → `EditorSession._refit_replacing` + `_push_structure` → `EditorWidget._structural_step` → route → a descriptor in `summary.js` → `runStructuralRefit`.
- [ ] **Step 3: Full verification** (report every command's result):

```bash
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
npm run check:frontend
uv run pytest -n 14 tests/ -q -m "not slow"
uv run pytest -n 14 tests/ -q
uv run pytest tests/editor tests/test_editor_browser.py -m browser --run-browser -q
uv run python run_test.py
uv lock --check && uv pip check
git diff --stat origin/master -- src/ ; git diff --stat origin/master -- tests/
```

- [ ] **Step 4: Commit**

```bash
git add tests/editor/test_editor_structure_browser.py docs/tutorials/edit-a-model-in-the-browser.md docs/development/internals/editor-frontend.md
git commit -m "Browser-test the structural tools and document them"
```

---

## Execution: the Workflow

Eight agents run in sequence, each in the worktree, and each commits its own tasks and pushes nothing. Every agent prompt carries the Global Constraints verbatim, including the code-shape rules, and names its tasks.

| # | Agent | Model | Tasks |
|---|---|---|---|
| 1 | backend-core | Opus, effort max | 1, 2 |
| 2 | backend-features | Opus, effort max | 3, 4 |
| 3 | backend-notes | Opus, effort high | 5, 6 |
| 4 | frontend-structure | Fable 5.1 + frontend-design skill | 7 |
| 5 | frontend-breaks | Fable 5.1 + frontend-design skill | 8 |
| 6 | verify-and-docs | Opus, effort high | 9 |
| 7 | critic | Opus, effort max | whole branch vs. spec and plan: spec coverage, code shape (duplication, nesting, dead guards, names), mutation checks re-run, the five Review Focus lines; findings as P1/P2/P3 with `file:line` |
| 8 | repair | Opus, effort max | fixes P1 and material P2 findings, re-runs Task 9 Step 3; skipped when there are none |

Then Max tries it in the real editor. Opening a pull request needs his go-ahead (one PR for the whole branch).
