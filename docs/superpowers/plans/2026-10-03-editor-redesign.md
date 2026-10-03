# Editor Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:**
- staged structural changes with one Refit, keep-reference, click/Shift-click
  ranges, ordered-categorical spline tools;
- inspector search, a Settings tab, git-style history;
- a Cross-validation tab with Run CV and Final fit;
- a rating-table preview, a warm dark theme with the animated comic switch, and
  the fold-plot level-order fix.

**Architecture:**
- **Backend.** The Python editor session keeps a list of pending structural steps
  whose draft specs compose. One batch refit applies them all, carries hand edits
  over, and lands as one undoable step. CV and Final fit run as cancellable
  background jobs off the widget lock.
- **Frontend.** The vanilla-JS app gains a settings module, a pending-aware chart,
  anchor-based selection, a searchable inspector, a `cv` app view and a pill
  theme switch.

**Tech Stack:** Python 3.12–3.14 (dev 3.13), numpy/pandas, FastAPI + uvicorn
(editor server), vanilla ES modules with JSDoc + `tsc` checking, `node --test`,
Playwright browser tests.

**Spec:** `docs/superpowers/specs/2026-10-03-editor-redesign-design.md` (read it
first; decisions D1–D12 are binding). Mock-ups:
https://claude.ai/artifact/Wi5CUShrPvbCRfEJ7ifZxY.

## Global Constraints

- Work only in this worktree:
  `/home/max/projects/superglm/.claude/worktrees/editor-improvements-redesign-96ab8d`.
  Never `cd` to `/home/max/projects/superglm`: that is the main checkout on
  another branch.
- Run Python through the worktree venv as a module:
  `./.venv/bin/python -m pytest …`. `./.venv/bin/pytest` imports the wrong tree.
- Quick suite: `./.venv/bin/python scripts/run_test_suite.py -m "not slow and not browser and not docs"`.
  Frontend: `npm run check:frontend` (run `npm ci` once first). Browser, as
  two separate commands, never one `-n` run:
  `./.venv/bin/python -m pytest tests/test_editor_browser.py -m browser --run-browser -q`
  then `./.venv/bin/python -m pytest tests/editor -m browser --run-browser -q`.
- No new runtime dependencies.
- No framework or build step in the frontend: vanilla ES modules with
  `// @ts-check` JSDoc.
- A browser-facing error is one fixed sentence per refusal kind. Never forward
  backend exception text (`src/superglm/editor/errors.py`).
- No popups, dialogs, banners or suggestions triggered by a selection. A new
  action is a compact SVG icon with a delayed hover popover and a Help entry.
- One linear Undo/Redo history over manual edits, pending structural steps and
  applied steps. "Revert to original model" stays the undo-all.
- Tests assert behaviour, not wall time. Every new behaviour test must fail on
  `origin/master` (155832e8). Each task says how that was demonstrated.
- float64 only. No `longdouble`.
- Fonts: Source Sans 3, IBM Plex Mono, and Bangers (theme-switch label only),
  through the one Google Fonts link in `app/index.html`.
- Every `localStorage` read and write is wrapped in try/catch, and the page
  renders correctly when storage is blocked.
- `prefers-reduced-motion: reduce` disables every animation and transition.
  tokens.css already does this globally; keep it.
- Feature PRs never touch `pyproject.toml` `version`, `superglm.__version__` or
  `uv.lock`'s own version. Release impact is declared later as `release:minor`.
- Never push, and never open a PR, until Max says so.
- Commit after each task with a message ending:
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Never override the git identity.

## Review Focus

1. **Two staged changes on the same term**: collapse B10+B11, then set the
   reference to "B10+B11"; or collapse, then ungroup part of the group. They
   compose, run as **one** fit, and the reference matches the last staged
   intent. A fit-time refusal leaves the pending list and model unchanged and
   says to undo the last waiting change. *Owner: Task A3.*
2. **Interleaved undo/redo**: edit DrivAge, stage a collapse, edit VehAge, then
   Undo ×3. That undoes VehAge, then the staged collapse (no refit, no revision
   bump), then DrivAge. Undo after a Refit brings the changes back as waiting;
   Redo re-applies the Refit without fitting. *Owner: Task A2.*
3. **Integer-typed or numeric-looking levels** (levels `1, 2, 10`, or ints)
   through collapse with keep-reference, ungroup, inspector search and fold-plot
   order. Native types are preserved, and the order is model order, not string
   or row order. *Owners: Tasks B1, H1, E1.*
4. **Blocked storage / private window**: settings fall back to defaults, the
   theme follows the browser, and nothing throws. *Owners: Tasks F1, I2.*
5. **CV data mismatch and cancel**:
   - `cv_data` rows ≠ fold indices → Run CV is disabled with a fixed reason;
   - an older result without a fingerprint → row-count check plus a note;
   - Cancel mid-run → job ends "cancelled", nothing is published;
   - a hand-edited spline whose fold training range is narrower than the
     editor grid → curve clamped at the ends, no NaN.

   *Owner: Tasks G2–G4.*

## Shared interface contract

Every section below uses these names. A section that needs a change states it
under "Contract amendments" at its top.

### Python — `src/superglm/editor/_types.py`

```python
def new_step_id() -> str:
    """7 hex digits, unique within the process (odd-multiplier hash of a counter plus a salt)."""

@dataclass
class EditRecord:                    # existing fields unchanged, plus:
    step_id: str = field(default_factory=new_step_id)
    created_at: float = field(default_factory=time.time)

@dataclass(frozen=True)
class PendingStep:
    operation: str                   # "collapse" | "ungroup" | "set_reference" | "shape"
    term: str
    label: str                       # automatic message, e.g. "Collapse B10 + B11"
    params: dict[str, Any]           # labels only, see below
    draft_spec: Any                  # unfitted replacement spec after this step
    history_position: int            # len(session.history) when staged
    step_id: str = field(default_factory=new_step_id)
    created_at: float = field(default_factory=time.time)

# SessionState gains:   pending: tuple[PendingStep, ...] = ()
# StructuralStep gains: step_id: str = field(default_factory=new_step_id)
#                       created_at: float = field(default_factory=time.time)
```

`params` by operation:
- `collapse`: `{"levels": [label, ...], "group_label": str | None}`
- `ungroup`: `{"levels": [label, ...]}`
- `set_reference`: `{"level": label}`
- `shape`: `{"lo": float | str, "hi": float | str, "degree": int, "join": "tangent" | "kink"}`

### Python — `EditorSession` (session.py)

```python
pending: list[PendingStep]                       # captured in SessionState
step_notes: dict[str, str]                       # step_id -> note; survives undo/redo
def draft_spec(self, term: str): ...             # last pending spec for term, else in-force fitted spec
def stage_structural(self, operation: str, term: str, params: dict[str, Any], *,
                     keep_reference: bool = True) -> PendingStep: ...
def refit_pending(self, *, method: str = "auto") -> StructuralStep: ...
def set_step_note(self, step_id: str, note: str) -> None: ...
def editor_history_records(self) -> list[dict[str, Any]]: ...
    # [{"id", "time", "operation", "term", "message", "note", "status"}], status in {"applied","waiting","edit"}
# Existing replace_with_collapsed_levels / replace_with_ungrouped_levels /
# replace_with_reference_level / replace_with_shaped_range keep their signatures:
# each becomes stage_structural(...) followed by refit_pending().
```

`from_model(..., cv=None, cv_data=None)` and
`edit(model, terms=None, *, n_points=200, centering="native", with_se=True, train_data=None, validation_data=None, test_data=None, cv=None, cv_data=None)`.
`cv_report=` keeps working (D12).

### Python — spec builders

- `collapsed_feature_spec(...)` and `ungrouped_feature_spec(...)` gain keyword
  `draft_spec=None, keep_reference: bool = True`.
- `reference_feature_spec(...)` and `shaped_feature_spec(...)` gain keyword
  `draft_spec=None`.
- `clone_with_replaced_features(model, replacements: dict[str, spec], *, lambda1=..., lambda2=...)`
  is the mapping form. `clone_with_replaced_feature` delegates to it.

### HTTP routes (server.py → widget)

| Route | Body | Returns |
|---|---|---|
| `POST /stage` | `{operation, term, params, keep_reference, level_display}` | structural-transition payload, with `pending` |
| `POST /refit_pending` | `{level_display}` | structural-transition payload |
| `POST /note` | `{id, note}` | `{ok: true, state}` |
| `POST /rating_table` | `{term}` | `{term, available, reason, columns, rows, note}` |
| `POST /job_start` | `{kind: "cv" \| "final_fit"}` | `{job_id}` |
| `POST /job_status` | `{job_id, wait}` | `{job_id, kind, status: "running"\|"done"\|"failed"\|"cancelled", progress: [...], result}` |
| `POST /job_cancel` | `{job_id}` | `{job_id, status}` |

### State payload additions

- Top level, `pending`: `[{id, operation, term, label, params, note, time}]`.
- Per term, `pending`: `{groups: {group_label: [member labels]} | null, ranges: [{lo, hi, degree, join, label}], reference: str | null}`.
- Each timeline entry gains `id`, `time`, `note`, and
  `status: "applied" | "waiting" | "edit"`.
- `report` payload kinds: `"validation"`, `"final"`, `"cv"`.

### Frontend

- `app/views/settings.js` exports:
  - `DEFAULT_SETTINGS` = `{refitEveryChange: false, keepReference: true,
    followBrowserTheme: true, groupsDefault: "expanded", buildDurationMs: 10000,
    showTimings: false}`;
  - `loadSettings()`;
  - `saveSettings(patch)`;
  - `onSettingsChange(listener) -> unsubscribe`.

  It uses one key, `superglm.editor.settings`.
- `api/client.js` gains `stage`, `refitPending`, `setNote`, `ratingTable`,
  `jobStart`, `jobStatus` and `jobCancel`.
- Store: `view.selectionAnchor: number | null`. `view.inspectorPane` takes
  `"settings"` instead of `"advanced"`. `AppView` adds `"cv"`.
- DOM ids:
  - `#refitPendingAction` (top bar, shortcut `R`);
  - `#settingsTab` / `#settingsPane` (replace `#advancedTab` / `#advancedPane`);
  - `#summarySearch`;
  - `#themeSwitch` (replaces `#themeAction`);
  - `#cvTab`;
  - `#termViewToggle` (Chart / Table);
  - `#ratingTableFrame`.

## File map

| File | Responsibility | Tasks |
|---|---|---|
| `src/superglm/plotting/comparison.py` | fold-plot level domain | H1 |
| `src/superglm/editor/collapse.py` | builders: keep-reference, draft-aware, `levels=`/`unseen=` | B1, A1 |
| `src/superglm/editor/shapes.py` | shaped-range builder draft-aware | A1 |
| `src/superglm/editor/_types.py` | step ids, `PendingStep`, state fields | A2 |
| `src/superglm/editor/session.py` | staging, batch refit, carry-over, notes, undo | A2–A4 |
| `src/superglm/editor/payloads.py` | pending + timeline payload, `spline_view` | A4, D1 |
| `src/superglm/editor/widget.py`, `server.py` | routes and jobs wiring | A4, K1, G3 |
| `src/superglm/editor/jobs.py` (new) | cancellable background jobs | G3 |
| `src/superglm/editor/cv.py` (new) | CV payload, stored folds, run CV, final fit | G2–G4 |
| `src/superglm/editor/carry.py` (new) | re-apply edited curves onto another fitted model | G2 |
| `src/superglm/editor/rating_preview.py` (new) | per-term rating-table preview | K1 |
| `src/superglm/editor/controls.py` | handles for ordered spline terms | D1 |
| `src/superglm/model_selection.py` | `n_rows`, `data_fingerprint` | G1 |
| `app/views/settings.js` (new) | settings store + Settings pane | F1 |
| `app/main.js`, `app/state/*`, `app/api/*` | wiring | all frontend tasks |
| `app/interactions.js` | gestures | C1 |
| `app/summary.js` | inspector search/filters/follow | E1–E2 |
| `app/chart.js`, `app/chart/*` | pending overlays, ordered spline curve | A5, D2 |
| `app/reports.js` | cv view | G5 |
| `app/views/theme.js`, `app/styles/dark.css`, `app/styles/*.css` | warm dark + comic switch | I1–I2 |
| `app/index.html` | DOM for all of the above | each frontend task |
| `docs/tutorials/edit-a-model-in-the-browser.md` | user docs | Z1 |

## Phase order

1. **H1, B1**: library and reference. Independent; run first.
2. **A1–A4**: staging backend. A1 → A2 → A3 → A4, strictly sequential.
3. **F1, A5, A6**: settings, then pending UI and the history panel. Needs A4.
4. **C1, E1, E2**: gestures and inspector. Needs F1 for the filters; C1 is
   independent of A.
5. **D1, D2, K1**: ordered spline and rating preview. D independent of A; K1
   needs A4 only for the route pattern.
6. **I1, I2**: dark theme and switch. Needs F1.
7. **G1–G5**: cross-validation. Needs A3 (carry-over semantics) and F1.
8. **Z1**: docs, full suite, ruff, lock checks, one end-to-end timing pair,
   whole-branch review.

Tasks within a phase that touch different files may run in parallel. `main.js`
and `index.html` are shared: frontend tasks touching them run one at a time.

---

## Reconciliation: read before any task

Seven writers drafted the sections below in parallel against the shared
contract. Each section opens with its own **Contract amendments**. Those
amendments are accepted and override the contract above wherever they differ.
Where two sections disagree, this block decides. An executor reads this block,
then its task, then the contract amendments of the task's section.

### Final task order

| Phase | Tasks |
|---|---|
| 0 | S0 |
| 1 | H1, B1 |
| 2 | A1 → A2 → A3 → A4 |
| 3 | F1 → A5 → A5b → A5c → A6 |
| 4 | C1 → C2 → E1 → E2 |
| 5 | D1 → D2 → K1a → K1b |
| 6 | I1, I2 → I3 |
| 7 | G1 → G2 → G3 → G4 → G5 |
| 7b | M1 → M2: **deferred** (Max, 2026-10-03: "We don't have to worry about splitting main js for now"); not run in this build |
| 8 | Z1 |

I1 has no dependencies and may run at any point. Every other frontend task
edits `main.js` or `index.html`, so they run one at a time.

### Decisions on the conflicts

1. **`carry.py` is created once, by A3** (`carried_curve`). G2 Step 3a says
   "Create `src/superglm/editor/carry.py`". Read that as "add to it": append
   G2's `model_with_edited_curves` and its helpers below A3's code, and keep
   A3's `carried_curve` unchanged. If a G2 helper has the same name as an A3
   function, keep A3's and call it.
2. **No `stage`, `refitPending` or `setNote` client methods** (S3 amendment 2).
   Staging and Refit are descriptors in `summary.js`, and the note goes through
   `actions.executeStateMutation`. `ratingTable`, `jobStart`, `jobStatus` and
   `jobCancel` stay client methods, as in K1b and G5.
3. **"Refit after every structural change" means one Undo per change** (decided
   by Claude: Undo takes back the last thing you did).
   - When the setting is on, A5 calls the legacy route for the operation
     (`/collapse_levels`, `/ungroup_levels`, `/set_reference`, `/shape_range`),
     with `keep_reference` from Settings, instead of `/stage` and
     `/refit_pending`.
   - S2's legacy calls stage and refit as one step whose Undo goes straight
     back to the state before the change. They also keep the per-operation
     refusal sentences.
   - A5's browser test `test_refit_after_every_change_stages_then_refits_at_once`
     asserts instead that one change gives:
     - exactly one refit (count `/refit_pending` and legacy-route responses);
     - one new timeline entry, and no entry with `status: "waiting"`;
     - that one Undo restores the term's pre-change levels.

     Rename it `test_refit_after_every_change_is_one_step`.
4. **No p-value chip for a categorical's folded line** (E2). A minimum over its
   levels overstates significance. The folded line shows name, kind, EDF and
   the waiting chip. Spline and ordered-spline lines keep their term-level p
   chip. Drop E2's "min" chip code and its test assertions. Add one assertion
   that a categorical's folded line has no `.summary-p-chip`.
5. **Ordered-spline handles sit where numeric-spline handles do.** A handle's
   value is its coefficient, so it can sit off the curve; D1/D2 are as
   written. The mock-up drew them on the curve. Consistency with numeric
   splines wins, so the drag rule stays the same.
6. **Handles stay on while a change to the same term waits.** As with every
   manual edit on that term, D2's rule drops such edits at Refit, and Undo
   restores them.
7. **A waiting set-reference shows in the reference chip.** Add this to A5c.
   - **Test** (`tests/editor_frontend/context_bar.test.js`, a new file headed
     `// @ts-nocheck` like its neighbours):

     ```js
     import test from "node:test";
     import assert from "node:assert/strict";
     import { renderContextBar } from "../../src/superglm/editor/app/views/context_bar.js";

     function nodes() {
       const node = () => ({ textContent: "", hidden: false, dataset: {} });
       return { kindNode: node(), edfNode: node(), referenceNode: node(), statusNode: node() };
     }

     test("a waiting reference change shows in the reference chip", () => {
       const n = nodes();
       renderContextBar(n, {
         name: "VehBrand",
         term: {
           kind: "categorical", term_type: "categorical", effective_df: 10, n_points: 11,
           reference: { level: "B2", policy: "kept" },
           pending: { groups: null, ranges: [], reference: "B10 + B11" },
         },
         selectionSize: 0,
       });
       assert.equal(n.referenceNode.textContent, "reference B10 + B11 · waiting");
       assert.equal(n.referenceNode.dataset.waiting, "true");
     });

     test("without a waiting reference the chip shows the fitted one", () => {
       const n = nodes();
       renderContextBar(n, {
         name: "VehBrand",
         term: { kind: "categorical", effective_df: 10, n_points: 11,
           reference: { level: "B2", policy: "kept" } },
         selectionSize: 0,
       });
       assert.equal(n.referenceNode.textContent, "reference B2 · kept");
       assert.equal(n.referenceNode.dataset.waiting, "false");
     });
     ```

     It fails on master: there is no `pending` handling and no `"kept"`
     policy, so the chip text is `reference B2 · undefined`.
   - **Implementation**, in `app/views/context_bar.js` (after B1 has added
     `kept: "kept"` to `REFERENCE_POLICY`). Replace:

     ```js
       const reference = term.reference;
       referenceNode.hidden = !reference;
       referenceNode.textContent = reference
         ? `reference ${reference.level} · ${REFERENCE_POLICY[reference.policy]}`
         : "";
     ```

     with:

     ```js
       const reference = term.reference;
       const waitingReference = term.pending ? term.pending.reference : null;
       referenceNode.hidden = !reference && !waitingReference;
       referenceNode.textContent = waitingReference
         ? `reference ${waitingReference} · waiting`
         : reference
           ? `reference ${reference.level} · ${REFERENCE_POLICY[reference.policy]}`
           : "";
       referenceNode.dataset.waiting = waitingReference ? "true" : "false";
     ```

     Add to `app/styles/shell.css`:

     ```css
     .context-chip[data-waiting="true"] {
       background: var(--sig-weak-bg);
       color: var(--sig-weak-fg);
     }
     ```

   A waiting ungroup needs nothing new: it changes `pending.groups`, which A5c
   already draws.
8. **Python stays modular** (Max reviews the Python; see the spec's
   maintainability note).
   - The bodies of `stage_structural`, `refit_pending`, `draft_spec`,
     `undo_target`, `redo_target` and `timeline_items`, and their private
     helpers from A2/A3, live in a new `src/superglm/editor/staging.py`. Each
     is a module-level function taking the session as its first argument
     (`def stage_structural(session, operation, term, params, *, keep_reference=True, X=None)`).
   - `session.py` keeps one-line methods that delegate, so the contract's
     method names and signatures are unchanged.
   - Apply A2/A3's Step 3 code into `staging.py`, renaming `self` to
     `session`. Tests and call sites are unaffected.
   - A2's commit adds `staging.py` to its `git add` line, and a test in A2
     asserts `superglm.editor.staging.stage_structural` exists, so a reviewer
     sees the split.
   - CV and jobs already live in `cv.py` and `jobs.py`.
9. **The fold colours** are `var(--trace-k)` in G5, with I1's re-stepped dark
   values (S7 amendment 6). If G5 draws exposure at 0.35 in light, its dark
   value is 0.55, scoped to `:root[data-theme="dark"]`.
10. **Bangers offline.** The switch label's font stack is
    `Bangers, Impact, "Arial Black", sans-serif`. I2 also sets the label's
    `max-width` to the free side of the track with `overflow: hidden`, so
    NIGHT never runs under the knob when Bangers fails to load.
11. **Running the browser tests.** Always run `tests/test_editor_browser.py`
    and `tests/editor` as separate commands, never in one `-n` run. On master,
    running `tests/editor` first in one process fails 10 tests in the other
    file. This replaces the single browser command in Global Constraints.

### Task S0: Baseline

**Files:** none changed.

- [ ] **Step 1: Install the frontend tools once.**

  Run: `npm ci`.

  Expected: `node_modules/` exists, and `npm run check:frontend` passes on the
  untouched tree.

- [ ] **Step 2: Record the baseline.**

  ```bash
  ./.venv/bin/python scripts/run_test_suite.py -m "not slow and not browser and not docs"
  ./.venv/bin/python -m pytest tests/test_editor_browser.py -m browser --run-browser -q
  ./.venv/bin/python -m pytest tests/editor -m browser --run-browser -q
  ```

  Expected: all pass. Note the counts; Z1 compares against them.

### Pre-existing problems found while planning (not fixed here)

These are recorded for follow-up, unless a task says otherwise.

- `cross_validate(return_estimators=True)` crashes on an integer-coded
  categorical. **Fixed in H1** (its test needs it).
- openpyxl writes workbook floats to 16 significant digits, not exact float64
  (K1a).
- The ordering of the two browser files matters (item 11).
- The light palette fails the separability rule that I1 applies to the dark
  one (S7).
- Saved editor sessions do not keep waiting changes or notes (S2).
- `retained_fit_dataset` reads `_fit_weights` while `_resolve_refit_data` reads
  `_fit_sample_weight_ref` (S6).

---


# Section S1 — Level order in fold plots (H1) and keep the reference (B1)

Both tasks are Phase 1 and touch disjoint files, so they can run in parallel.
Every snippet below was applied to a copy of `origin/master` (155832e8) and run:
the new tests fail on master as stated and pass with the change, and the
neighbouring files listed in each Step 4 stay green.

## Contract amendments

1. `EditorSession.refit_with_collapsed_levels(term, *, group_label=None, keep_reference: bool = True, **refit_kwargs)`
   and `EditorSession.refit_with_ungrouped_levels(term, *, keep_reference: bool = True, **refit_kwargs)`
   gain `keep_reference`. *Reason:* B1 lands before A2's `stage_structural`, and
   the existing routes need the switch now. `replace_with_collapsed_levels(term, **kwargs)`
   and `replace_with_ungrouped_levels(term, **kwargs)` keep their contract
   signatures; `keep_reference` reaches the refit methods through `**kwargs`.
   When A3 rewrites them as `stage_structural(...)` + `refit_pending()`, it must
   pop `keep_reference` from `kwargs` and pass it to `stage_structural`, or it
   reaches `SuperGLM.fit()` as an unknown keyword.
2. `EditorWidget._collapse_levels(...)` and `EditorWidget._ungroup_levels(...)` gain
   keyword `keep_reference: bool = True`. `POST /collapse_levels` and
   `POST /ungroup_levels` accept an optional boolean body field `keep_reference`,
   which defaults to true. A value that is not a JSON boolean is refused with the
   fixed sentence `keep_reference must be true or false.` The helper
   `server._keep_reference(payload) -> bool` is the one A4 reuses for `POST /stage`.
   *Reason:* D3 says keep-reference is on by default, and D8 says a setting that
   affects the backend travels with each request.
3. In the state payload, `terms[name].reference.policy` can now be `"kept"`, so it
   takes `"most_exposed" | "first" | "pinned" | "kept"`. The `TermReference.policy`
   union in `app/api/contracts.js` gains `'kept'`. *Reason:* spec §B says the chip
   reads "reference B2 · kept".
4. New name `superglm.editor.collapse.KEPT_REFERENCE_ATTRIBUTE = "_editor_kept_reference"`.
   A collapse or ungroup builder run with `keep_reference=True` sets this attribute
   on the replacement spec, and `_reference_payload` reads it. It follows
   the existing `EDITOR_CHOSEN_SHAPE_ATTRIBUTE` precedent in `editor/shapes.py`.
   *Reason:* the mark lives on the spec object, so it moves with undo and redo
   and with draft composition without needing extra session state. A
   `PendingStep.draft_spec` built with `keep_reference=True` therefore reads
   "kept" after the Refit, with no further work in phase A.

## Notes for later tasks

- **A1.** B1 already handles the spec §A bullet "Ungroup to no grouping leaves a
  string base; map it back to the native type" for a plain `Categorical`, through
  `_native_level` in `ungrouped_feature_spec`, on both keep paths. The ordered
  path already had this mapping in `_ordered_original_values`. A1 should drop
  that bullet.
- **A1, draft-aware builders.** `_in_force_reference(spec)` is the seam. It
  returns `spec._base_level` when the spec is fitted, and `spec.base` when it is
  not. When a draft exists:
  - If the draft's declared base is concrete (an earlier kept or set-reference
    step), pass the draft.
  - If the draft's declared base is symbolic (an earlier step with
    `keep_reference=False`), pass the in-force fitted spec, so the reference that
    is kept is the one actually in force.
- **A5 and F1.**
  - The collapse and ungroup Help bodies in `app/views/help_content.js` still say
    only "... and refit the model". Whoever rewrites them for staging should add
    one sentence on keep-reference.
  - The frontend sends no `keep_reference` yet. It defaults to true on the
    server, and F1's setting adds it to the request body.
- **G (CV tab).** `curve_similarity[term]["domain"]["levels"]` is in model level
  order after H1, so the "Relativities by fold" chart can use it directly.

---

### Task H1: Fold-plot level domain in model order

Spec §4 H and Review Focus item 3. `_shared_level_domain` puts an unordered
`Categorical` in the order its rows first appear (`drop_duplicates()`). Measured on
freMTPL2, that gave Area A, B, E, D, C, F. `build_cv_curve_similarity`
(`plotting/curve_similarity.py:105`) gets its domain from the same function
through `_build_term_comparison_data`, so `cross_validate(..., return_estimators=True)`
(`model_selection.py:542`) and `CrossValidationResult.plot_terms_by_fold`
(`model_selection.py:68`) are both fixed here.

**Found while verifying (needed for the integer case):** on master, an
integer-coded `Categorical` makes `_build_term_comparison_data` raise
`ValueError: Encountered unseen categorical levels at predict time: ['1', '10', '2']`.
The domain is text, and the fitted universe is native ints. So on master
`cross_validate(..., return_estimators=True)` crashes for any model with an
integer-coded categorical. The fix scores the column's own native values, which
is what `predict` receives.

"Model order" here means the order the model's own term inference reports, the
same order the editor draws:

| Term | Model order |
|---|---|
| `OrderedCategorical` | `_ordered_levels`, unchanged |
| grouped `Categorical` | `grouping.all_original_levels`, as `inference/_term_helpers.py:368` expands groups |
| other `Categorical` | `spec._levels`, which keeps native order: 1, 2, 10 |

Observed labels the model lacks follow in row order. The ordered path already
behaved this way.

**Files:**
- Modify `src/superglm/plotting/comparison.py:108-135` (`_shared_level_domain`, plus two new helpers next to it)
- Modify `src/superglm/plotting/comparison.py:210-211` (scoring in `_build_term_comparison_data`)
- Test `tests/test_plot_comparison.py:9` (import) and a new test inserted above line 106
- Test `tests/test_curve_similarity.py:127` (an existing assertion pinned row order; update it) and a new test appended at the end of the file

**Interfaces:**
- Consumes the fitted specs' `_ordered_levels`, `_grouping.all_original_levels` and `_levels`.
- Produces:
  - `_model_level_order(spec) -> list[str]`
  - `_shared_level_domain(models: Mapping[str, Any], X: EagerFrame, term: str) -> dict[str, list[str]]`: same signature; the order changes, the set of labels does not
  - `_native_level_values(X: EagerFrame, term: str, labels: list[str]) -> NDArray` (object dtype)
- The payload's `domain["levels"]` stays `list[str]`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_plot_comparison.py`, extend the import on line 9:

```python
from superglm import Categorical, OrderedCategorical, Spline, SuperGLM
```
becomes
```python
from superglm import Categorical, OrderedCategorical, Spline, SuperGLM, collapse_levels
```

Insert this test immediately above
`def test_build_term_comparison_data_can_store_per_label_support(fitted_comparison_models):`
(line 106):

```python
@pytest.mark.parametrize(
    ("rows", "grouped", "expected"),
    [
        (["C", "A", "B"], False, ["A", "B", "C"]),
        (["C", "A", "B"], True, ["A", "B", "C"]),
        # Model order, which is neither row order (10, 1, 2) nor text order (1, 10, 2).
        ([10, 1, 2], False, ["1", "2", "10"]),
    ],
    ids=["text", "grouped", "integer"],
)
def test_unordered_level_domain_follows_the_model_not_the_rows(rows, grouped, expected):
    from superglm.plotting.comparison import _build_term_comparison_data

    rng = np.random.default_rng(20261003)
    region = np.tile(np.asarray(rows), 100)
    y = 0.5 + 0.1 * (region == rows[1]) + rng.normal(0.0, 0.05, region.size)
    X = pd.DataFrame({"region": region})
    grouping = collapse_levels(X["region"], groups={"B+C": ["B", "C"]}) if grouped else None
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"region": Categorical(base="first", grouping=grouping)},
    )
    model.fit(X, y)

    term = _build_term_comparison_data(models={"fit": model}, terms=["region"], X=X)["terms"][0]

    assert term["domain"]["levels"] == expected
    assert term["support"]["levels"] == expected
    # Each label carries its own fitted effect, scored from the column's native values.
    inference = model.term_inference("region", with_se=False)
    assert [str(level) for level in inference.levels] == expected
    np.testing.assert_array_equal(term["series"]["fit"]["link"], inference.log_relativity)
```

In `tests/test_curve_similarity.py`, the polars test pinned row order at line 127.
Replace

```python
    assert similarity["band"]["domain"]["levels"] == list(dict.fromkeys(band))
```
with
```python
    # Model order, not the order the rows first show each level.
    assert similarity["band"]["domain"]["levels"] == ["A", "B", "C"]
```

Append this test at the end of `tests/test_curve_similarity.py`:

```python
def test_build_cv_curve_similarity_scores_integer_coded_levels_in_model_order():
    from superglm.plotting.curve_similarity import build_cv_curve_similarity

    rng = np.random.default_rng(9)
    n = 150
    code = np.tile([10, 1, 2], n // 3)
    w = rng.uniform(0.5, 1.2, n)
    y = rng.poisson(np.exp(-1.0 + 0.2 * (code == 2)) * w).astype(float)
    X = pd.DataFrame({"code": code})

    models = []
    for seed in [1, 2, 3]:
        idx = np.random.default_rng(seed).choice(n, size=int(0.8 * n), replace=False)
        model = SuperGLM(features={"code": Categorical(base="first")})
        model.fit(X.iloc[idx], y[idx], sample_weight=w[idx])
        models.append(model)

    similarity = build_cv_curve_similarity(models=models, X=X, sample_weight=w, n_points=41)

    assert similarity["code"]["domain"]["levels"] == ["1", "2", "10"]
    for label, model in zip(["fold_0", "fold_1", "fold_2"], models, strict=True):
        inference = model.term_inference("code", with_se=False)
        np.testing.assert_array_equal(
            similarity["code"]["curves"]["link"][label], inference.log_relativity
        )
```

- [ ] **Step 2: Run them, expect FAIL**

```bash
./.venv/bin/python -m pytest tests/test_plot_comparison.py tests/test_curve_similarity.py -q
```

Expected on unmodified source (origin/master 155832e8): 5 failed, 10 passed,
4 skipped. The skips are plotly tests; plotly is absent from the local venv and
CI runs them.
- `test_unordered_level_domain_follows_the_model_not_the_rows[text]` and `[grouped]`
  fail with `AssertionError: assert ['C', 'A', 'B'] == ['A', 'B', 'C']`, which is row order.
- `[integer]` fails with `ValueError: Encountered unseen categorical levels at predict time: ['1', '10', '2']`.
- `test_build_cv_curve_similarity_accepts_polars_without_converting_fold_models`
  fails because master returns row order (`['C', ..., 'A']`).
- `test_build_cv_curve_similarity_scores_integer_coded_levels_in_model_order` fails
  with the same `ValueError`.

- [ ] **Step 3: Implement**

In `src/superglm/plotting/comparison.py`, replace `_shared_level_domain` (lines 108-135):

```python
def _shared_level_domain(
    models: Mapping[str, Any],
    X: EagerFrame,
    term: str,
) -> dict[str, list[str]]:
    """Build a shared categorical/ordered level domain."""
    ordered_levels: list[str] | None = None
    for model in models.values():
        spec = model._specs[term]
        if isinstance(spec, OrderedCategorical):
            ordered_levels = [str(level) for level in spec._ordered_levels]
            break

    observed_levels = [
        str(level)
        for level in pd.Series(X.column_array(term), name=term)
        .astype(str)
        .drop_duplicates()
        .tolist()
    ]
    if ordered_levels is None:
        return {"levels": observed_levels}

    merged = [level for level in ordered_levels if level in observed_levels]
    for level in observed_levels:
        if level not in merged:
            merged.append(level)
    return {"levels": merged}
```
with
```python
def _model_level_order(spec) -> list[str]:
    """The level order a fitted level term reports in, as text.

    An ordered term's declared order; a grouped categorical's original levels,
    which is how its term inference expands the groups; otherwise the fitted
    universe, which keeps native order (1, 2, 10, not "1", "10", "2").
    """
    if isinstance(spec, OrderedCategorical):
        return [str(level) for level in spec._ordered_levels]
    grouping = getattr(spec, "_grouping", None)
    if grouping is not None:
        return [str(level) for level in grouping.all_original_levels]
    return [str(level) for level in spec._levels]


def _shared_level_domain(
    models: Mapping[str, Any],
    X: EagerFrame,
    term: str,
) -> dict[str, list[str]]:
    """Build a shared categorical/ordered level domain in model order.

    An ordered term's order wins; otherwise the first model's fitted order.
    Observed labels the model order lacks follow in row order.
    """
    specs = [model._specs[term] for model in models.values()]
    spec = next((s for s in specs if isinstance(s, OrderedCategorical)), specs[0])

    observed_levels = [
        str(level)
        for level in pd.Series(X.column_array(term), name=term)
        .astype(str)
        .drop_duplicates()
        .tolist()
    ]
    observed = set(observed_levels)
    merged = [level for level in _model_level_order(spec) if level in observed]
    placed = set(merged)
    merged.extend(level for level in observed_levels if level not in placed)
    return {"levels": merged}


def _native_level_values(X: EagerFrame, term: str, labels: list[str]) -> NDArray:
    """The column's own values for the domain's text labels, as predict receives them.

    A fitted universe keeps native types, so an integer-coded categorical
    refuses the text "1" as an unseen level.
    """
    native: dict[str, Any] = {}
    for value in pd.Series(X.column_array(term), name=term).drop_duplicates().tolist():
        native.setdefault(str(value), value)
    return np.asarray([native.get(label, label) for label in labels], dtype=object)
```

Then, in `_build_term_comparison_data` (lines 210-211), score native values:

```python
            domain = _shared_level_domain(normalized_models, frame, term)
            levels = np.asarray(domain["levels"], dtype=object)
```
becomes
```python
            domain = _shared_level_domain(normalized_models, frame, term)
            levels = _native_level_values(frame, term, domain["levels"])
```

`_support_payload` keeps grouping exposure by `astype(str)` labels, which
already agree with `domain["levels"]`. Leave it unchanged.

- [ ] **Step 4: Run the tests, expect PASS**

```bash
./.venv/bin/python -m pytest tests/test_plot_comparison.py tests/test_curve_similarity.py -q
./.venv/bin/python -m pytest tests/test_cross_validate.py tests/test_public_api_snapshot.py tests/test_piecewise_editor.py -q -n 8
./.venv/bin/ruff check src/superglm/plotting/comparison.py tests/test_plot_comparison.py tests/test_curve_similarity.py
./.venv/bin/ruff format --check src/superglm/plotting/comparison.py tests/test_plot_comparison.py tests/test_curve_similarity.py
```

Expected: 15 passed and 4 skipped (plotly) for the first command, and 199
passed and 4 skipped (plotly) for the second. `test_polars_comparison_payload_and_support_match_pandas`
stays green, so pandas and polars still agree.

Mutation check, already shown by Step 2: drop `_native_level_values` (score
`np.asarray(domain["levels"], dtype=object)` as before) and both integer cases
fail again with the `ValueError`.

- [ ] **Step 5: Commit**

```bash
git add src/superglm/plotting/comparison.py tests/test_plot_comparison.py tests/test_curve_similarity.py
git commit -m "$(cat <<'MSG'
Fold plots: unordered level domain in model order, scored on native values

_shared_level_domain put an unordered Categorical in row-appearance order
(freMTPL2 Area came out A, B, E, D, C, F). Use the first model's fitted
order: the grouping's original levels for a grouped term, otherwise
spec._levels; an ordered term keeps _ordered_levels; observed labels the
model lacks follow in row order. build_cv_curve_similarity and
plot_terms_by_fold read the same domain.

Score the column's native values for the domain labels: an integer-coded
categorical (levels 1, 2, 10) raised "unseen categorical levels" on master,
which crashed cross_validate(return_estimators=True) for such a model.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
MSG
)"
```

---

### Task B1: Keep the reference through collapse and ungroup

Spec §4 B, D3 and Review Focus item 3. Facts checked on master:

- `collapsed_feature_spec` passes the declared `spec.base` to `_collapsed_base`
  (`collapse.py:95`). A symbolic policy passes through and resolves again at
  fit time. Measured: under `most_exposed`, collapsing C + D moved the
  reference from B to C + D.
- `ungrouped_feature_spec` passes `spec.base` to `_valid_base_after_ungroup`
  (`collapse.py:154`, `546-555`). For a concrete base whose group a partial
  ungroup renamed, that falls back to `selected_levels[0]`, the first level
  pulled out.
- A plain `Categorical` rebuilt with no grouping gets a text base. An
  integer-coded term then fails at fit: `Base '2' not found in levels: [1, 2, 10]`.

The change:
- With `keep_reference=True`, which is the default, the builders start from
  `_in_force_reference(spec)`: the fitted `_base_level`, in its native type.
  - The existing concrete-base branches of `_collapsed_base` then give "the
    level, or the group containing it".
  - An ungroup uses the new `_kept_base_after_ungroup`: the new level holding
    most of the old reference group's members (the collapse rule), with a tie
    going to a level that was not pulled out.
  - The replacement spec carries `KEPT_REFERENCE_ATTRIBUTE`, and the chip reads
    `reference B · kept`.
- With `keep_reference=False`, the code path is today's. The one difference is
  that an ungroup to no grouping maps the base to the column's native value,
  which removes the master crash above.

**Files:**
- Modify `src/superglm/editor/collapse.py`:
  - `:11` (import pandas) and `:24` (new constant);
  - `collapsed_feature_spec` `:27-35` and `:95-103`;
  - `ungrouped_feature_spec` `:116-123` and `:154-160`;
  - new helpers above `_collapsed_base` (`:506`);
  - new helper below `_valid_base_after_ungroup` (`:546-555`).
- Modify `src/superglm/editor/session.py`: `refit_with_collapsed_levels` `:912-938`, `refit_with_ungrouped_levels` `:950-958`.
- Modify `src/superglm/editor/widget.py`: `_collapse_levels` `:971-983`, `_ungroup_levels` `:985-997`.
- Modify `src/superglm/editor/server.py`: routes `:248-266`, new helper above `_evidence_metadata` (`:521`).
- Modify `src/superglm/editor/payloads.py`: import `:12`, `_reference_payload` `:260-267`.
- Modify `src/superglm/editor/app/views/context_bar.js:12-13` and `src/superglm/editor/app/api/contracts.js:46-49`.
- Test `tests/test_editor_structure.py`: insert above `:295` (`test_set_reference_refuses_a_special_level`) and above `:373` (`BANDS = ...`).
- Test `tests/editor/test_editor_structure_browser.py:306-307`: the chip text after a collapse.
- Test `tests/test_editor.py:4873-4875` and `:4915`: two widget stubs that the new `keep_reference=` keyword would break.

**Interfaces:**
- Consumes:
  - fitted spec `_base_level` (native: int 2 on an integer-coded term, `"B+C"` under a grouping);
  - `LevelGrouping.group_to_originals`, `original_to_group` and `grouped_levels`;
  - the column values from `X`.
- Produces:
  - `collapsed_feature_spec(model, term, selected_indices, *, X, group_label=None, keep_reference: bool = True) -> tuple[Any, dict[str, Any]]`
  - `ungrouped_feature_spec(model, term, selected_indices, *, X, keep_reference: bool = True) -> tuple[Any, dict[str, Any]]`
  - `KEPT_REFERENCE_ATTRIBUTE: str`
  - `_in_force_reference(spec) -> Any`
  - `_native_level(label: Any, data) -> Any`
  - `_kept_base_after_ungroup(base: Any, selected_levels: list[str], existing: LevelGrouping, grouping: LevelGrouping) -> str`
  - `EditorSession.refit_with_collapsed_levels(term, *, group_label=None, keep_reference=True, **refit_kwargs)`
  - `EditorSession.refit_with_ungrouped_levels(term, *, keep_reference=True, **refit_kwargs)`
  - `EditorWidget._collapse_levels(term=None, method="auto", *, level_display="expanded", keep_reference=True)` and the same keyword on `_ungroup_levels`
  - `server._keep_reference(payload: dict[str, Any]) -> bool`
  - payload `reference.policy == "kept"`

- [ ] **Step 1: Write the failing tests**

In `tests/test_editor_structure.py`, insert this helper and these tests
immediately above `def test_set_reference_refuses_a_special_level():` (line 295).
The fixture uses deterministic counts:
- B (360) is the most exposed single level;
- C + D (660) outweigh B;
- A + B (540) weigh less than C + D;
- E (100) only exists so a second group can stay in place.

```python
def _exposed_region_session(base: str) -> EditorSession:
    """B is the most exposed level; C + D outweigh it, and A + B weigh less than C + D."""
    rng = np.random.default_rng(20261003)
    region = rng.permutation(np.repeat(["A", "B", "C", "D", "E"], [180, 360, 330, 330, 100]))
    effects = {"A": -0.1, "B": 0.0, "C": 0.15, "D": 0.2, "E": 0.05}
    y = 0.5 + np.array([effects[r] for r in region]) + rng.normal(0.0, 0.05, region.size)
    weight = np.ones(region.size)
    X = pd.DataFrame({"region": region})
    model = SuperGLM(
        family="gaussian", selection_penalty=0.0, features={"region": Categorical(base=base)}
    )
    model.fit(X, y, sample_weight=weight)
    assert model._specs["region"]._base_level == "B", "precondition: B is the reference"
    return EditorSession.from_model(model, terms=["region"], train_data=(X, y, weight))


@pytest.mark.parametrize(
    ("options", "reference"),
    [
        ({}, {"level": "B", "policy": "kept"}),
        ({"keep_reference": False}, {"level": "C+D", "policy": "most_exposed"}),
    ],
    ids=["kept", "off"],
)
def test_a_collapse_keeps_a_most_exposed_reference(options, reference):
    # C + D outweigh B, so most_exposed, chosen again at the refit, moves the reference.
    session = _exposed_region_session("most_exposed")
    session.select_levels("region", ["C", "D"])
    session.replace_with_collapsed_levels("region", method="fit", **options)
    assert session_payload(session)["region"]["reference"] == reference


def test_collapsing_the_reference_makes_its_group_the_reference():
    session = _exposed_region_session("most_exposed")
    session.select_levels("region", ["C", "D"])
    session.replace_with_collapsed_levels("region", method="fit")
    # A + B weigh less than C + D, which most_exposed alone would pick.
    session.select_levels("region", ["A", "B"])
    session.replace_with_collapsed_levels("region", method="fit")
    assert session.model._specs["region"]._base_level == "A+B"


@pytest.mark.parametrize(
    ("pulled", "options", "reference"),
    [
        (["D"], {}, "B+C"),
        (["B"], {}, "C+D"),
        (["B", "C"], {}, "D"),
        # Off, the declared-base rule is unchanged: the first level pulled out.
        (["D"], {"keep_reference": False}, "D"),
    ],
    ids=["majority", "reference-pulled", "tie", "off"],
)
def test_a_partial_ungroup_leaves_the_reference_with_the_levels_that_stay(
    pulled, options, reference
):
    session = _exposed_region_session("B")
    # A + E stays grouped, so no ungroup here can reuse the pre-collapse fit.
    session.select_levels("region", ["A", "E"])
    session.replace_with_collapsed_levels("region", method="fit")
    session.select_levels("region", ["B", "C", "D"])
    session.replace_with_collapsed_levels("region", method="fit")
    assert session.model._specs["region"]._base_level == "B+C+D"

    session.select_levels("region", pulled)
    session.replace_with_ungrouped_levels("region", method="fit", **options)
    assert session.model._specs["region"]._base_level == reference


def test_keeping_the_reference_keeps_integer_levels_native():
    rng = np.random.default_rng(20261004)
    band = rng.permutation(np.repeat([1, 2, 10], [250, 400, 350]))
    y = 0.5 + 0.1 * (band == 10) + rng.normal(0.0, 0.05, band.size)
    weight = np.ones(band.size)
    X = pd.DataFrame({"band": band})
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"band": Categorical(base="most_exposed")},
    )
    model.fit(X, y, sample_weight=weight)
    assert model._specs["band"]._base_level == 2, "precondition: native integer levels"

    setup = EditorSession.from_model(model, terms=["band"], train_data=(X, y, weight))
    setup.select_levels("band", ["1", "10"])
    collapsed = setup.replace_with_collapsed_levels("band", method="fit")
    # 1 + 10 outweigh 2, which stays the reference; a grouped fit spells levels as text.
    assert collapsed._specs["band"]._base_level == "2"

    # A fresh session has no pre-collapse fit to reuse, so the ungroup refits.
    session = EditorSession.from_model(collapsed, terms=["band"], train_data=(X, y, weight))
    session.select_levels("band", ["1", "10"])
    session.replace_with_ungrouped_levels("band", method="fit")
    assert session.model._specs["band"]._base_level == 2
    assert session_payload(session)["band"]["reference"] == {"level": "2", "policy": "kept"}
```

Insert these two HTTP tests immediately above `BANDS = [f"B{i}" for i in range(1, 9)]`
(line 373):

```python
@pytest.mark.parametrize(
    ("body", "reference"),
    [
        ({}, {"level": "B", "policy": "kept"}),
        ({"keep_reference": False}, {"level": "C+D", "policy": "most_exposed"}),
    ],
    ids=["default", "off"],
)
def test_widget_http_collapse_reads_keep_reference_from_the_body(body, reference):
    session = _exposed_region_session("most_exposed")
    widget = session.widget()
    try:
        _post_json(f"{widget.url}/select", {"term": "region", "indices": [2, 3]})
        payload = _post_json(
            f"{widget.url}/collapse_levels", {"term": "region", "method": "fit", **body}
        )
        assert payload["state"]["terms"]["region"]["reference"] == reference
    finally:
        widget.close()


@pytest.mark.parametrize("route", ["collapse_levels", "ungroup_levels"])
def test_widget_http_refuses_a_keep_reference_that_is_not_a_boolean(region_model, route):
    model, _ = region_model
    session = EditorSession.from_model(model, terms=["region"])
    widget = session.widget()
    try:
        session.select_levels("region", ["B", "C"])
        with pytest.raises(urllib.error.HTTPError) as error:
            _post_json(f"{widget.url}/{route}", {"term": "region", "keep_reference": "no"})
        assert error.value.code == 400
        assert json.loads(error.value.read().decode("utf-8")) == {
            "error": "keep_reference must be true or false."
        }
        assert session.model is model and session.structure_history == []
    finally:
        widget.close()
```

In `tests/editor/test_editor_structure_browser.py`, the refresh test collapses
T01 + T02 under `base="first"`. The reference is the same level as before, but
it is now kept. Replace lines 306-307:

```python
        # The first level now sits inside the new group, so the group is the reference.
        assert reference.text_content() == "reference T01+T02 · first"
```
with
```python
        # The reference T01 now sits inside the new group, which keeps it.
        assert reference.text_content() == "reference T01+T02 · kept"
```

In `tests/test_editor.py`, two tests replace `session.replace_with_collapsed_levels`
with stubs that do not accept the keyword the widget now forwards. At line 4873:

```python
        def replace_with_smaller_term(term, *, method="auto"):
            assert term == "region"
            assert method == "fit"
```
becomes
```python
        def replace_with_smaller_term(term, *, method="auto", keep_reference=True):
            assert term == "region"
            assert method == "fit"
            assert keep_reference is True
```

and at line 4915:

```python
            lambda term, *, method="auto": editor_model,
```
becomes
```python
            lambda term, *, method="auto", keep_reference=True: editor_model,
```

- [ ] **Step 2: Run them, expect FAIL**

```bash
./.venv/bin/python -m pytest tests/test_editor_structure.py -q -n 8 -k "keeps_a_most_exposed or makes_its_group or partial_ungroup_leaves or integer_levels_native or keep_reference_from_the_body or not_a_boolean"
./.venv/bin/python -m pytest tests/editor/test_editor_structure_browser.py -k refresh_pulls -m browser --run-browser -q
```

Expected on unmodified source (origin/master 155832e8): 11 failed, 1 passed;
the browser test fails.

| Test | How it fails on master |
|---|---|
| `test_a_collapse_keeps_a_most_exposed_reference[kept]` | `{'level': 'C+D', 'policy': 'most_exposed'} != {'level': 'B', 'policy': 'kept'}`: the measured move |
| `[off]` | `TypeError: SuperGLM.fit() got an unexpected keyword argument 'keep_reference'` |
| `test_collapsing_the_reference_makes_its_group_the_reference` | `'C+D' == 'A+B'` |
| `test_a_partial_ungroup_leaves_the_reference_with_the_levels_that_stay[majority]` | `'D' == 'B+C'`: master's first-pulled fallback |
| `[reference-pulled]` | `'B' == 'C+D'` |
| `[tie]` | `'B' == 'D'` |
| `[off]` | the `TypeError` above |
| `test_keeping_the_reference_keeps_integer_levels_native` | `'1+10' == '2'` |
| `test_widget_http_collapse_reads_keep_reference_from_the_body[default]` | the measured move, through HTTP |
| `test_widget_http_refuses_a_keep_reference_that_is_not_a_boolean[collapse_levels]` | `DID NOT RAISE HTTPError`: master ignores the field |
| `[ungroup_levels]` | the 400 says "Term 'region' does not have collapsed levels." |
| browser `test_refresh_pulls_a_notebook_side_structural_change` | `'reference T01+T02 · first' == 'reference T01+T02 · kept'` |

`test_widget_http_collapse_reads_keep_reference_from_the_body[off]` passes on
master. That is correct: "off" is master's behaviour, and master ignores the
unknown body field. It guards the forwarding. Step 4 gives its mutation check.

The two updated stubs in `tests/test_editor.py` pass on master and after the change.

- [ ] **Step 3: Implement**

`src/superglm/editor/collapse.py`. Import pandas (line 11):

```python
import numpy as np

from superglm._frame import as_eager_frame
```
becomes
```python
import numpy as np
import pandas as pd

from superglm._frame import as_eager_frame
```

Add the constant under `_SYMBOLIC_BASE_POLICIES` (line 24):

```python
_SYMBOLIC_BASE_POLICIES = {"first", "most_exposed"}
```
becomes
```python
_SYMBOLIC_BASE_POLICIES = {"first", "most_exposed"}
# Marks a spec whose reference a collapse or ungroup held in place. The state
# payload reads it to label the reference "kept".
KEPT_REFERENCE_ATTRIBUTE = "_editor_kept_reference"
```

`collapsed_feature_spec` signature and docstring (lines 32-35):

```python
    X,
    group_label: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Return a replacement feature spec that collapses selected levels."""
```
becomes
```python
    X,
    group_label: str | None = None,
    keep_reference: bool = True,
) -> tuple[Any, dict[str, Any]]:
    """Return a replacement feature spec that collapses selected levels.

    ``keep_reference`` holds the reference the in-force fit resolved, or the
    new group when it takes that level in. ``False`` hands the declared base
    to the refit, where a symbolic policy (``most_exposed``, ``first``)
    resolves again and can move the reference.
    """
```

Its base and the mark (lines 95-103):

```python
    base = _collapsed_base(spec.base, selected_levels, label, existing, grouping)

    if isinstance(spec, OrderedCategorical):
        replacement = rebuilt_ordered_spec(spec, grouping=grouping, base=base, data=values)
    else:
        replacement = Categorical(
            base=base,
            grouping=grouping,
        )
```
becomes
```python
    declared = _in_force_reference(spec) if keep_reference else spec.base
    base = _collapsed_base(declared, selected_levels, label, existing, grouping)

    if isinstance(spec, OrderedCategorical):
        replacement = rebuilt_ordered_spec(spec, grouping=grouping, base=base, data=values)
    else:
        replacement = Categorical(
            base=base,
            grouping=grouping,
        )
    if keep_reference:
        setattr(replacement, KEPT_REFERENCE_ATTRIBUTE, True)
```

`ungrouped_feature_spec` signature and docstring (lines 120-123):

```python
    *,
    X,
) -> tuple[Any, dict[str, Any]]:
    """Return a replacement feature spec that removes selected levels from groups."""
```
becomes
```python
    *,
    X,
    keep_reference: bool = True,
) -> tuple[Any, dict[str, Any]]:
    """Return a replacement feature spec that removes selected levels from groups.

    ``keep_reference`` holds the in-force reference: a reference group that
    loses members follows the members that stay. ``False`` hands the declared
    base on, as ``collapsed_feature_spec`` does.
    """
```

Its base, the native mapping and the mark (lines 154-160):

```python
    base = _valid_base_after_ungroup(spec.base, selected_levels, grouping)
    if isinstance(spec, OrderedCategorical):
        replacement = rebuilt_ordered_spec(
            spec, grouping=replacement_grouping, base=base, data=values
        )
    else:
        replacement = Categorical(base=base, grouping=replacement_grouping)
```
becomes
```python
    if keep_reference:
        base = _kept_base_after_ungroup(
            _in_force_reference(spec), selected_levels, existing, grouping
        )
    else:
        base = _valid_base_after_ungroup(spec.base, selected_levels, grouping)
    if isinstance(spec, OrderedCategorical):
        replacement = rebuilt_ordered_spec(
            spec, grouping=replacement_grouping, base=base, data=values
        )
    else:
        # Without a grouping the fit reads native values (3, not "3").
        if replacement_grouping is None:
            base = _native_level(base, values)
        replacement = Categorical(base=base, grouping=replacement_grouping)
    if keep_reference:
        setattr(replacement, KEPT_REFERENCE_ATTRIBUTE, True)
```

Two helpers above `_collapsed_base` (line 506). `_collapsed_base` already
starts with `base = str(base)`, so only its annotation changes:

```python
def _collapsed_base(
    base: str,
```
becomes
```python
def _in_force_reference(spec) -> Any:
    """The reference ``spec``'s fit resolved, in its native type (3, not "3").

    Before a fit there is none, and the declared base stands in.
    """
    level = getattr(spec, "_base_level", "")
    return spec.base if level == "" else level


def _native_level(label: Any, data) -> Any:
    """``label`` as the column spells it; a symbolic policy passes through."""
    if label in _SYMBOLIC_BASE_POLICIES:
        return label
    native: dict[str, Any] = {}
    for raw in pd.unique(np.asarray(data).ravel()).tolist():
        native.setdefault(str(raw), raw)
    return native.get(str(label), label)


def _collapsed_base(
    base: Any,
```

The ungroup rule, below `_valid_base_after_ungroup` (its last line, 555).
`_valid_base_after_ungroup` itself stays as it is: it is the `keep_reference=False` path.

```python
    return selected_levels[0] if selected_levels else "most_exposed"
```
becomes
```python
    return selected_levels[0] if selected_levels else "most_exposed"


def _kept_base_after_ungroup(
    base: Any,
    selected_levels: list[str],
    existing: LevelGrouping,
    grouping: LevelGrouping,
) -> str:
    """The level holding the in-force reference once ``selected_levels`` leave their groups.

    A reference group that loses members follows the new level holding most of
    them (the collapse rule in ``_collapsed_base``); a tie goes to a level that
    was not pulled out. A reference the old grouping does not know keeps the
    declared-base rule.
    """
    base = str(base)
    if base in _SYMBOLIC_BASE_POLICIES or base in grouping.grouped_levels:
        return base
    members = _base_original_members(base, existing)
    if not members:
        return _valid_base_after_ungroup(base, selected_levels, grouping)
    mapped = [str(grouping.original_to_group.get(member, member)) for member in members]
    pulled = set(selected_levels)
    counts = {label: mapped.count(label) for label in dict.fromkeys(mapped)}
    return max(counts, key=lambda label: (counts[label], label not in pulled))
```

`src/superglm/editor/session.py`, `refit_with_collapsed_levels` (lines 912-927):

```python
    def refit_with_collapsed_levels(
        self, term: str, *, group_label: str | None = None, **refit_kwargs: Any
    ):
        """Collapse selected categorical levels and refit a full model copy.

        ``refit_kwargs`` are ``X``, ``y``, ``sample_weight``, ``offset``,
        ``method``, ``lambda1``, ``lambda2`` and fit keywords.
        """
        editable = self._require_term(term)
        idx = self._require_selection(term)
        try:
            return self._refit_replacing(
                term,
                lambda X_ref: collapsed_feature_spec(
                    self.model, editable, idx, X=X_ref, group_label=group_label
                ),
```
becomes
```python
    def refit_with_collapsed_levels(
        self,
        term: str,
        *,
        group_label: str | None = None,
        keep_reference: bool = True,
        **refit_kwargs: Any,
    ):
        """Collapse selected categorical levels and refit a full model copy.

        ``keep_reference`` holds the in-force reference level, or the group
        that takes it in; ``False`` lets the declared base policy choose again.
        ``refit_kwargs`` are ``X``, ``y``, ``sample_weight``, ``offset``,
        ``method``, ``lambda1``, ``lambda2`` and fit keywords.
        """
        editable = self._require_term(term)
        idx = self._require_selection(term)
        try:
            return self._refit_replacing(
                term,
                lambda X_ref: collapsed_feature_spec(
                    self.model,
                    editable,
                    idx,
                    X=X_ref,
                    group_label=group_label,
                    keep_reference=keep_reference,
                ),
```

`refit_with_ungrouped_levels` (lines 950-958):

```python
    def refit_with_ungrouped_levels(self, term: str, **refit_kwargs: Any):
        """Remove selected levels from collapsed groups and refit a model copy."""
        editable = self._require_term(term)
        idx = self._require_selection(term)
        return self._refit_replacing(
            term,
            lambda X_ref: ungrouped_feature_spec(self.model, editable, idx, X=X_ref),
            **refit_kwargs,
        )
```
becomes
```python
    def refit_with_ungrouped_levels(
        self, term: str, *, keep_reference: bool = True, **refit_kwargs: Any
    ):
        """Remove selected levels from collapsed groups and refit a model copy.

        ``keep_reference`` holds the in-force reference; a reference group that
        loses members follows the members that stay.
        """
        editable = self._require_term(term)
        idx = self._require_selection(term)
        return self._refit_replacing(
            term,
            lambda X_ref: ungrouped_feature_spec(
                self.model, editable, idx, X=X_ref, keep_reference=keep_reference
            ),
            **refit_kwargs,
        )
```

`replace_with_collapsed_levels` and `replace_with_ungrouped_levels` stay as they
are. Their `**kwargs` already reach the two methods above, and
`_ungroup_restores_reference_model` reads only `X`, `y`, `sample_weight` and
`offset` from them.

`src/superglm/editor/widget.py`, `_collapse_levels` (lines 971-980):

```python
    def _collapse_levels(
        self,
        term: str | None = None,
        method: str = "auto",
        *,
        level_display: str = "expanded",
    ) -> dict[str, Any]:
        return self._structural_step(
            "collapse_levels",
            lambda target: self.session.replace_with_collapsed_levels(target, method=method),
```
becomes
```python
    def _collapse_levels(
        self,
        term: str | None = None,
        method: str = "auto",
        *,
        level_display: str = "expanded",
        keep_reference: bool = True,
    ) -> dict[str, Any]:
        return self._structural_step(
            "collapse_levels",
            lambda target: self.session.replace_with_collapsed_levels(
                target, method=method, keep_reference=keep_reference
            ),
```

`_ungroup_levels` (lines 985-994):

```python
    def _ungroup_levels(
        self,
        term: str | None = None,
        method: str = "auto",
        *,
        level_display: str = "expanded",
    ) -> dict[str, Any]:
        return self._structural_step(
            "ungroup_levels",
            lambda target: self.session.replace_with_ungrouped_levels(target, method=method),
```
becomes
```python
    def _ungroup_levels(
        self,
        term: str | None = None,
        method: str = "auto",
        *,
        level_display: str = "expanded",
        keep_reference: bool = True,
    ) -> dict[str, Any]:
        return self._structural_step(
            "ungroup_levels",
            lambda target: self.session.replace_with_ungrouped_levels(
                target, method=method, keep_reference=keep_reference
            ),
```

`src/superglm/editor/server.py`, the `/collapse_levels` route body (lines 251-255):

```python
            lambda: widget._collapse_levels(
                None if "term" not in payload else str(payload["term"]),
                str(payload.get("method", "auto")),
                level_display=_level_display(payload),
            )
```
becomes
```python
            lambda: widget._collapse_levels(
                None if "term" not in payload else str(payload["term"]),
                str(payload.get("method", "auto")),
                level_display=_level_display(payload),
                keep_reference=_keep_reference(payload),
            )
```

The `/ungroup_levels` route body (lines 261-265):

```python
            lambda: widget._ungroup_levels(
                None if "term" not in payload else str(payload["term"]),
                str(payload.get("method", "auto")),
                level_display=_level_display(payload),
            )
```
becomes
```python
            lambda: widget._ungroup_levels(
                None if "term" not in payload else str(payload["term"]),
                str(payload.get("method", "auto")),
                level_display=_level_display(payload),
                keep_reference=_keep_reference(payload),
            )
```

The helper goes above `_evidence_metadata` (line 521). It raises inside the
route's `_guarded_json` lambda, so the browser gets the fixed sentence with a 400:

```python
def _evidence_metadata(payload: dict[str, Any]) -> dict[str, Any]:
```
becomes
```python
def _keep_reference(payload: dict[str, Any]) -> bool:
    value = payload.get("keep_reference", True)
    if not isinstance(value, bool):
        raise EditorValueError("keep_reference must be true or false.")
    return value


def _evidence_metadata(payload: dict[str, Any]) -> dict[str, Any]:
```

`src/superglm/editor/payloads.py`, the import (line 12):

```python
from superglm.editor._types import StructuralStep
```
becomes
```python
from superglm.editor._types import StructuralStep
from superglm.editor.collapse import KEPT_REFERENCE_ATTRIBUTE
```

and the policy in `_reference_payload` (lines 266-267):

```python
    policy = spec.base if spec.base in {"most_exposed", "first"} else "pinned"
    return {"level": str(level), "policy": policy}
```
becomes
```python
    if getattr(spec, KEPT_REFERENCE_ATTRIBUTE, False):
        policy = "kept"
    else:
        policy = spec.base if spec.base in {"most_exposed", "first"} else "pinned"
    return {"level": str(level), "policy": policy}
```

`src/superglm/editor/app/views/context_bar.js` (lines 12-13). `tsc` requires the
record to cover the widened union:

```js
  pinned: "pinned",
});
```
becomes
```js
  pinned: "pinned",
  kept: "kept",
});
```

`src/superglm/editor/app/api/contracts.js` (line 48):

```js
 * @property {'most_exposed'|'first'|'pinned'} policy
```
becomes
```js
 * @property {'most_exposed'|'first'|'pinned'|'kept'} policy
```

- [ ] **Step 4: Run the tests, expect PASS**

```bash
./.venv/bin/python -m pytest tests/test_editor_structure.py -q -n 8 -k "keeps_a_most_exposed or makes_its_group or partial_ungroup_leaves or integer_levels_native or keep_reference_from_the_body or not_a_boolean"
./.venv/bin/python -m pytest tests/test_editor.py tests/test_editor_structure.py tests/test_ordered_categorical_specials_editor.py tests/test_piecewise_editor.py tests/test_editor_security.py tests/test_editor_validation_errors.py tests/test_editor_evidence.py tests/test_editor_evaluation_cache.py tests/test_lss_editor_style.py -q -n 12 -m "not browser"
./.venv/bin/python -m pytest tests/editor/test_editor_structure_browser.py -m browser --run-browser -q
npm run check:frontend
./.venv/bin/ruff check src/superglm/editor tests/test_editor_structure.py tests/test_editor.py tests/editor/test_editor_structure_browser.py
./.venv/bin/ruff format --check src/superglm/editor tests/test_editor_structure.py tests/test_editor.py tests/editor/test_editor_structure_browser.py
```

Expected on the copy used to verify this section:
- 12 passed for the first command;
- 598 passed and 2 skipped for the editor files (the skips are plotly-only);
- 9 passed for the browser file;
- 185 passing node tests for the frontend.

`tsc` was not available where this section was verified, so the type check
is still to run. The two JS edits only widen the `TermReference.policy` union
and add the matching record key, and `tsc` checks that both are present.

The existing base tests stay green unchanged:
- `test_regrouping_split_base_group_chooses_valid_remaining_base`
- `test_compact_summary_shows_collapsed_reference_level_group`
- `test_a_pinned_reference_survives_a_later_collapse`
- `test_ungroup_preserves_symbolic_base_policy_to_avoid_display_uplift`: its
  ungroup removes the last group, so the pre-collapse fit is reused
- `test_ordered_integer_ungroup_pre_collapsed_model_refits_without_history`

Mutation checks. Each was run on the verified copy, then reverted:
- In `_kept_base_after_ungroup`, replace the key with `key=counts.__getitem__`.
  Only `...[tie]` fails, `'B' == 'D'`, so the tie-break is load-bearing.
- In `ungrouped_feature_spec`, drop `base = _native_level(base, values)`.
  `test_keeping_the_reference_keeps_integer_levels_native` fails with
  `ValueError: Base '2' not found in levels: [1, 2, 10]`.
- In `server.py`, drop `keep_reference=_keep_reference(payload)` from the
  `/collapse_levels` route. `test_widget_http_collapse_reads_keep_reference_from_the_body[off]`
  fails, because the reference stays B.

- [ ] **Step 5: Commit**

```bash
git add src/superglm/editor/collapse.py src/superglm/editor/session.py src/superglm/editor/widget.py src/superglm/editor/server.py src/superglm/editor/payloads.py src/superglm/editor/app/views/context_bar.js src/superglm/editor/app/api/contracts.js tests/test_editor_structure.py tests/test_editor.py tests/editor/test_editor_structure_browser.py
git commit -m "$(cat <<'MSG'
Editor: keep the reference level when collapsing or ungrouping

A collapse or ungroup rebuilt the term with its declared base, so a
symbolic policy re-resolved at the refit and could move the reference
(most_exposed: collapsing C+D moved it from B to C+D). With
keep_reference (default on, D3) the builders start from the in-force
fit's resolved reference: the level, or the group that takes it in; a
partial ungroup of the reference group follows the members that stay.
An ungroup to no grouping maps the base back to the column's native
type, so integer-coded levels refit. The spec is marked and the chip
reads "reference B · kept".

keep_reference=False keeps today's rule. The session refit methods,
widget and /collapse_levels and /ungroup_levels take it; a non-boolean
body value is refused with a fixed sentence.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
MSG
)"
```

## Risks and open questions

- **Default change in the Python API.** `replace_with_collapsed_levels` and
  `replace_with_ungrouped_levels` now keep the reference by default (D3). Under a
  selection penalty the reference choice changes the fit as well as the
  parametrisation, so a notebook that relied on `most_exposed` re-resolving gets
  a different fit. `keep_reference=False` restores the old behaviour. The PR
  body should say so.
- **The pre-collapse shortcut.** `replace_with_ungrouped_levels` can reuse the
  pre-collapse fit, which keeps that fit's own reference. That agrees with
  keep-reference unless the collapse being undone ran with
  `keep_reference=False`. Left as it is.
- **Shaping drops "kept" (follow-up).** A later shaped range on an ordered term
  rebuilds the spec (`shapes._hosted`) without the mark, so the chip then reads
  "pinned" for the same level.
- **The mark is exported.** Exported models carry `_editor_kept_reference=True`
  on the spec, as they already carry `_editor_chosen_shape`. This is harmless to
  `predict` and to pickling.


## Section S2 — Staged structural changes and Refit: backend (Tasks A1–A4)

Spec §4 "A. Staged structural changes and Refit (core)", decisions D1, D2, D11,
Review Focus 1 and 2. Backend only; the chart, Refit button and History panel
are A5/A6.

**Research gate.** Characterised in spec §3: linear undo over one merged
command history (manual edits, waiting re-specifications, applied refits),
batch re-specification of several terms followed by one penalised refit, and
re-applying an edited curve on an identical display grid. The spec's sweep
(GAM Changer, KDD 2022) found no precedent for staging several structural
changes against one refit; nothing below adds an algorithm beyond that. The
step-id scheme is fixed by the contract; its uniqueness is elementary (an odd
multiplier is a unit modulo 2^28, so `n -> n*M + salt` is a bijection).

**Line numbers** are `origin/master` 155832e8. B1 runs first and edits
`collapse.py`, `session.py`, `widget.py` and `server.py`; anchors below name
the function as well, so an implementer finds the spot after B1 shifts lines.

**Formatting.** Not every code block below is pre-wrapped at 100 columns.
Run `./.venv/bin/python -m ruff format` on each touched file before a Step 4's
`ruff format --check`.

### Contract amendments

1. `PendingStep` also has `metadata: dict[str, Any] = field(default_factory=dict)` (the builder's step metadata, stamped as `_editor_step` on a one-change refit so that contract is unchanged) and `predictor: str | None = None` (spec §5 SuperLSS seam).
2. `SessionState` also has `pending_redo: tuple[PendingStep, ...] = ()`, and the session `pending_redo: list[PendingStep]`: undone waiting changes must redo in time order and cross structural steps the way `redo_stack` does.
3. `StructuralStep` also has `changes: tuple[PendingStep, ...] = ()`, the waiting changes a refit applied, so their ids and notes reach the timeline and the export. A change refitted on its own by a legacy call shares its step's `step_id`.
4. `EditRecord` gains a read-only `label` property (today's `"shift area"` text), so edits, waiting changes and steps all answer `.label`.
5. `stage_structural(..., *, keep_reference=True, X=None)`: `X` lets the legacy `replace_with_*` calls, which accept explicit refit data, build against the frame they will fit.
6. `refit_pending(self, *, method="auto", **refit_kwargs)`: a superset (`X`, `y`, `sample_weight`, `offset`, `lambda1`, `lambda2`, fit keywords), so `replace_with_*` keep their signatures. Returns the refit's `StructuralStep`.
7. New session methods `undo_target()`, `redo_target()` and `timeline_items()`. They are the one source of time order, used by undo/redo, the payloads and the export.
8. `PendingStep.label` is the builder's existing label (`"collapse B10 + B11 in VehBrand"`, `"Line 18–26 in DrivAge"`), the text the Undo popover already shows.
9. Timeline entries gain kind `"pending"` for a change that is waiting, or that a refit applied. The marker stays exactly `{"kind": "marker"}`; tests pin it. Refit step: operation `"refit_pending"`, label `"Refit · N changes"` (`"1 change"`). Carry-over step: operation `"carry_edits"`, label `"Hand edits carried over: area, age"`.
10. `time` is float seconds since the epoch in payloads, and an ISO-8601 UTC string in `editor_history_records()`. Each record also has `"predictor": None`.
11. Per-term `pending.groups` is the draft's whole grouping (groups of two or more) once a waiting collapse or ungroup touches the term: `{}` if none remain, `null` if none of the term's waiting changes is a collapse or ungroup. `ranges` lists the ranges the draft adds or changes. `reference` is the level the draft pins in place of the fitted reference, else `null`.
12. `src/superglm/editor/carry.py` (`carried_curve`) is created in A3, not G2. G2 extends it for fold grids.
13. New fixed sentences: `"No changes are waiting for a refit."`, `"Refit or undo the waiting changes before re-profiling."` (a re-profile is refused while changes wait), `"Unknown history entry."`, `"Unknown structural change: 'x'"`, `"params must be an object."`, `"levels must be a list of level labels."`, `"keep_reference must be true or false."`, `"note must be text."`, `"A note can be at most 2000 characters."`.

### Seams with other tasks

- **B1** (keep-reference) runs first. A1 renames the builders' `spec` to the
  term's draft. B1's read of the in-force reference must then go through
  `_reference_to_keep(fitted, spec, term)` (A1 Step 3d). The A1 test
  `test_keep_reference_keeps_the_waiting_reference_not_the_fitted_one` fails
  until it does. The A3 test `test_collapse_then_partial_ungroup_keeps_the_reference_in_the_majority_group`
  assumes B1's majority rule in `_valid_base_after_ungroup`.
- **A5/A6** (frontend):
  - The legacy routes (`/collapse_levels`, `/ungroup_levels`, `/set_reference`,
    `/shape_range`) keep working. Each one stages and refits at once, keeps
    today's per-operation refusal sentences, and unstages on refusal.
    `/refit_pending` answers any fit-time refusal with the batch sentence.
    "Refit after every structural change" can therefore keep the legacy routes
    and keep the specific sentences.
  - `revertAvailable` (`app/views/app_bar.js`) must also count a `"pending"`
    entry before the marker.
  - `TimelineEntry` in `api/contracts.js` gains `kind: "pending"`, `id`, `time`,
    `note` and `status`.
  - JS route pins in `tests/test_editor.py::test_widget_app_shell_contains_drag_editor`
    belong to A5.
- **G2** imports `superglm.editor.carry.carried_curve` (D5 reuses D2's rule).
- **Z1**: the new test node ids, and one renamed id (A3), count against
  `tests/test_ci_contracts.py::test_duration_manifest_covers_the_non_browser_suite`
  (≥95% recorded durations). Record durations for the whole uncovered set there.

---

### Task A1: Spec builders compose on a waiting draft

**Files:**
- Modify `src/superglm/editor/collapse.py`:
  - `collapsed_feature_spec`, :27-113;
  - `ungrouped_feature_spec`, :116-169;
  - `reference_feature_spec`, :176-209;
  - `clone_with_replaced_feature`, :227-235;
  - new helpers after `_valid_base_after_ungroup`, :546-555.
- Modify `src/superglm/editor/shapes.py`:
  - `shaped_feature_spec`, :97-145;
  - `_numeric_edges`, :200-211;
  - new `_free_geometry` after it.
- Create `tests/test_editor_staging.py`.

**Interfaces:**
- Consumes (B1):
  - `collapsed_feature_spec(..., keep_reference: bool = True)`;
  - `ungrouped_feature_spec(..., keep_reference: bool = True)`.
- Produces:
  - `collapsed_feature_spec(model, term, selected_indices, *, X, group_label=None, draft_spec=None, keep_reference=True) -> tuple[spec, dict]`
  - `ungrouped_feature_spec(model, term, selected_indices, *, X, draft_spec=None, keep_reference=True) -> tuple[spec, dict]`
  - `reference_feature_spec(model, term, level, *, X, draft_spec=None) -> tuple[spec, dict]`
  - `shaped_feature_spec(model, name, *, lo, hi, degree, join="tangent", X, draft_spec=None) -> tuple[spec, dict]`
  - `clone_with_replaced_features(model, replacements: dict[str, Any], *, lambda1=..., lambda2=...)`. It deep-copies each replacement, so a draft is never fitted. `clone_with_replaced_feature` delegates to it.
  - Private: `_reference_to_keep(fitted, spec, term)`, `_rebuilt_categorical(spec, fitted, *, base, grouping, data)`, `_native_levels(spec, fitted, data)`, `shapes._free_geometry(spec)`.

`draft_spec=None` means the fitted spec, so with nothing waiting every builder
behaves as today.

- [ ] **Step 1: Write the failing test.** Create `tests/test_editor_staging.py`:

```python
"""Editor staging: waiting structural changes, one Refit, carried edits, history ids and notes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from superglm import Categorical, Spline, SuperGLM
from superglm.editor import EditorSession
from superglm.editor.collapse import (
    clone_with_replaced_features,
    collapsed_feature_spec,
    reference_feature_spec,
    ungrouped_feature_spec,
)
from superglm.editor.errors import EditorValueError
from superglm.editor.refit import fit_refit_model
from superglm.editor.shapes import shaped_feature_spec

BRANDS = ["B1", "B2", "B10", "B11", "B12"]


@pytest.fixture
def book():
    """A small motor book: a brand factor, an area factor and a driver-age spline."""
    rng = np.random.default_rng(20261003)
    n = 900
    brand = rng.choice(BRANDS, n, p=[0.3, 0.25, 0.15, 0.15, 0.15])
    area = rng.choice(["A", "B", "C", "D"], n)
    age = rng.uniform(18.0, 80.0, n)
    effects = dict(zip(BRANDS, [0.0, 0.1, 0.25, 0.22, -0.1], strict=True))
    y = (
        0.5
        + np.array([effects[b] for b in brand])
        + 0.1 * (area == "C")
        + 0.2 * np.sin(age / 15.0)
        + rng.normal(0.0, 0.05, n)
    )
    X = pd.DataFrame({"brand": brand, "area": area, "age": age})
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        spline_penalty=0.1,
        features={
            "brand": Categorical(base="first"),
            "area": Categorical(base="first"),
            "age": Spline(n_knots=6),
        },
    )
    model.fit(X, y)
    return model, X, y


def _term(model, name):
    return EditorSession.from_model(model, terms=[name]).terms[name]


def _at(term, *labels):
    return np.array([term.levels.index(label) for label in labels], dtype=np.intp)


def test_a_reference_can_name_the_group_a_waiting_collapse_made(book):
    model, X, _ = book
    brand = _term(model, "brand")
    collapsed, _ = collapsed_feature_spec(model, brand, _at(brand, "B10", "B11"), X=X)
    pinned, step = reference_feature_spec(model, brand, "B10+B11", X=X, draft_spec=collapsed)
    assert (pinned.base, step["level"]) == ("B10+B11", "B10+B11")
    assert pinned._grouping.group_to_originals["B10+B11"] == ["B10", "B11"]


def test_keep_reference_keeps_the_waiting_reference_not_the_fitted_one(book):
    model, X, _ = book
    brand = _term(model, "brand")
    assert model._specs["brand"]._base_level == "B1"
    pinned, _ = reference_feature_spec(model, brand, "B2", X=X)
    collapsed, _ = collapsed_feature_spec(
        model, brand, _at(brand, "B10", "B11"), X=X, draft_spec=pinned, keep_reference=True
    )
    assert collapsed.base == "B2"
    again, _ = collapsed_feature_spec(
        model, brand, _at(brand, "B1", "B12"), X=X, draft_spec=collapsed, keep_reference=True
    )
    assert again.base == "B2"
    ungrouped, _ = ungrouped_feature_spec(
        model, brand, _at(brand, "B10"), X=X, draft_spec=again, keep_reference=True
    )
    assert ungrouped.base == "B2"


def test_collapse_and_ungroup_keep_the_level_universe_and_the_unseen_policy():
    rng = np.random.default_rng(20261004)
    area = rng.choice(["A", "B", "C", "D"], 400)
    y = 0.5 + 0.1 * (area == "B") + rng.normal(0.0, 0.05, 400)
    X = pd.DataFrame({"area": area})
    declared = ["A", "B", "C", "D", "E"]
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"area": Categorical(base="first", levels=declared, unseen="base")},
    )
    with pytest.warns(UserWarning, match="pinned to base"):
        model.fit(X, y)
    term = _term(model, "area")
    collapsed, _ = collapsed_feature_spec(model, term, _at(term, "B", "C"), X=X)
    ungrouped, _ = ungrouped_feature_spec(
        model, term, _at(term, "B", "C"), X=X, draft_spec=collapsed
    )
    for spec in (collapsed, ungrouped):
        assert (spec._declared_levels, spec.unseen) == (declared, "base")


def test_ungrouping_to_no_groups_gives_an_integer_reference_its_native_type():
    rng = np.random.default_rng(20261005)
    code = rng.choice([1, 2, 3, 10], 400)
    y = 0.5 + 0.1 * (code == 2) - 0.1 * (code == 10) + rng.normal(0.0, 0.05, 400)
    X = pd.DataFrame({"code": code})
    model = SuperGLM(
        family="gaussian", selection_penalty=0.0, features={"code": Categorical(base=3)}
    )
    model.fit(X, y)
    term = _term(model, "code")
    collapsed, _ = collapsed_feature_spec(model, term, _at(term, "1", "2"), X=X)
    # A grouped design speaks the grouping's labels, which are text.
    assert collapsed.base == "3"
    ungrouped, _ = ungrouped_feature_spec(
        model, term, _at(term, "1", "2"), X=X, draft_spec=collapsed
    )
    assert ungrouped._grouping is None
    assert type(ungrouped.base) is int and ungrouped.base == 3
    refit = clone_with_replaced_features(model, {"code": ungrouped})
    fit_refit_model(model, refit, method="fit", X=X, y=y)
    assert refit._specs["code"]._base_level == 3
    # The draft itself is never fitted: the clone fitted a copy.
    assert ungrouped._levels == []


def test_two_waiting_shapes_on_one_spline_compose_on_the_fitted_knots(book):
    model, X, _ = book
    fitted = model._specs["age"]
    first, _ = shaped_feature_spec(model, "age", lo=30.0, hi=45.0, degree=1, X=X)
    second, step = shaped_feature_spec(
        model, "age", lo=60.0, hi=70.0, degree=0, X=X, draft_spec=first
    )
    assert [(r.lo, r.hi, r.degree) for r in second.polynomial_ranges] == [
        (30.0, 45.0, 1),
        (60.0, 70.0, 0),
    ]
    np.testing.assert_array_equal(second._explicit_knots, fitted.fitted_base_knots)
    assert second._explicit_boundary == fitted.fitted_boundary
    assert step["label"] == "Flat 60–70 in age"
    with pytest.raises(EditorValueError, match="^This range overlaps the Line range 30–45."):
        shaped_feature_spec(model, "age", lo=40.0, hi=50.0, degree=0, X=X, draft_spec=first)
```

- [ ] **Step 2: Run it, expect FAIL.**
  `./.venv/bin/python -m pytest tests/test_editor_staging.py -q`
  - On this tree, and on `origin/master` 155832e8: the module fails to import with
    `ImportError: cannot import name 'clone_with_replaced_features'`.
  - With that import satisfied, master's builders fail each test on its own:
    - `TypeError: ... unexpected keyword argument 'draft_spec'`.
    - The universe test fails `(None, "error") == (declared, "base")`:
      collapse rebuilds `Categorical(base=, grouping=)` only.
    - The draft swapped into `model._specs`: set reference raises `KeyError: 'B10+B11'`,
      because `spec._levels` is `[]` before a fit (probed).
    - A second shape raises `TypeError: cannot unpack non-iterable NoneType`,
      because `fitted_boundary` is None before a fit (probed).
    - The integer ungroup fits with `ValueError: Base '3' not found in levels: [1, 2, 3, 10]` (probed).

- [ ] **Step 3: Implement.**

  **3a. `collapse.py` — one mapping form of the clone.** Replace
  `clone_with_replaced_feature` (:227-235) with:

```python
def clone_with_replaced_features(
    model, replacements: dict[str, Any], *, lambda1=..., lambda2=...
):
    """Clone a model and replace feature specs before fitting.

    Each replacement is deep-copied in, so fitting the clone never touches the
    caller's spec: a waiting step's draft stays unfitted, and a refit that is
    undone and run again fits a fresh copy of the same draft.
    """
    new_model = model._clone_without_features(set(), lambda1=lambda1, lambda2=lambda2)
    for term, replacement in replacements.items():
        new_model._specs[term] = copy.deepcopy(replacement)
    new_model._config = new_model._config.with_value(
        feature_templates=tuple((name, new_model._specs[name]) for name in new_model._feature_order)
    )
    new_model._config_revision += 1
    return new_model


def clone_with_replaced_feature(model, term: str, replacement, *, lambda1=..., lambda2=...):
    """Clone a model and replace one feature spec before fitting."""
    return clone_with_replaced_features(
        model, {term: replacement}, lambda1=lambda1, lambda2=lambda2
    )
```

  **3b. `collapse.py` — helpers.** Insert after `_valid_base_after_ungroup`
  (ends :555, before `_is_identity_grouping`):

```python
def _reference_to_keep(fitted, spec, term: EditableTerm):
    """The reference a keep-reference step keeps, named in ``spec``'s own levels.

    ``spec`` is the term's draft, or ``fitted`` when nothing waits. A draft that
    already names a level or group (a reference an earlier waiting step kept
    or set) keeps it. Otherwise the fitted reference is kept while the draft
    still has that level or group; a draft regrouped by a step staged with
    keep-reference off keeps its own policy.
    """
    if spec is fitted:
        return fitted._base_level
    if str(spec.base) not in _SYMBOLIC_BASE_POLICIES:
        return spec.base
    grouping = getattr(spec, "_grouping", None)
    names = term.levels if grouping is None else grouping.grouped_levels
    held = {str(name) for name in names or []}
    return fitted._base_level if str(fitted._base_level) in held else spec.base


def _native_levels(spec: Categorical, fitted: Categorical, data) -> dict[str, Any]:
    """Each level's native value by its text: declared, then fitted, then seen in ``data``.

    A draft has no fitted ``_levels``, so the in-force spec and the column stand
    in for it. A grouped spec's fitted levels are group labels, not raw ones.
    """
    import pandas as pd

    fitted_levels = fitted._levels if getattr(fitted, "_grouping", None) is None else []
    observed = pd.unique(np.asarray(data).ravel()).tolist()
    native: dict[str, Any] = {}
    for level in chain(spec._declared_levels or [], fitted_levels, observed):
        native.setdefault(str(level), level)
    return native


def _rebuilt_categorical(
    spec: Categorical, fitted: Categorical, *, base, grouping, data
) -> Categorical:
    """A fresh Categorical like ``spec`` with this grouping and base.

    It keeps ``levels=`` and ``unseen=``, which a collapse or ungroup used to
    drop. Grouped, the design speaks the grouping's text labels; ungrouped,
    the base goes back to its native value, so an integer level stays 3, not "3".
    """
    if grouping is None and str(base) not in _SYMBOLIC_BASE_POLICIES:
        base = _native_levels(spec, fitted, data).get(str(base), base)
    return Categorical(
        base=base, grouping=grouping, levels=spec._declared_levels, unseen=spec.unseen
    )
```

  **3c. `collapse.py` — `collapsed_feature_spec`.**
  - Signature: after `group_label: str | None = None,` add `draft_spec=None,`.
    Keep B1's `keep_reference: bool = True` beside it.
  - Append to the docstring:
    `"``draft_spec`` is the term's spec as waiting changes leave it (None: the fitted spec); the collapse is built on it, so changes to one term compose."`
  - Then the body:

```python
    spec = model._specs[term.name]
    if not isinstance(spec, Categorical | OrderedCategorical):
        raise EditorTypeError(
            f"Collapse levels is only available for categorical terms, got {term.name!r}."
        )
```
→
```python
    fitted = model._specs[term.name]
    spec = fitted if draft_spec is None else draft_spec
    if not isinstance(spec, Categorical | OrderedCategorical):
        raise EditorTypeError(
            f"Collapse levels is only available for categorical terms, got {term.name!r}."
        )
```
and
```python
    else:
        replacement = Categorical(
            base=base,
            grouping=grouping,
        )
```
→
```python
    else:
        replacement = _rebuilt_categorical(spec, fitted, base=base, grouping=grouping, data=values)
```

  **`ungrouped_feature_spec`**: add `draft_spec=None,` to the keyword
  parameters, beside B1's `keep_reference`. Append the same docstring
  sentence, with "ungroup" for "collapse". Then:
```python
    spec = model._specs[term.name]
    if not isinstance(spec, Categorical | OrderedCategorical):
        raise EditorTypeError(
            f"Ungroup levels is only available for categorical terms, got {term.name!r}."
        )
```
→
```python
    fitted = model._specs[term.name]
    spec = fitted if draft_spec is None else draft_spec
    if not isinstance(spec, Categorical | OrderedCategorical):
        raise EditorTypeError(
            f"Ungroup levels is only available for categorical terms, got {term.name!r}."
        )
```
and `        replacement = Categorical(base=base, grouping=replacement_grouping)` →
`        replacement = _rebuilt_categorical(spec, fitted, base=base, grouping=replacement_grouping, data=values)`.

  **3d. The B1 seam.** B1 passes the in-force resolved reference, when
  `keep_reference` is true, as the first argument of `_collapsed_base(...)` in
  `collapsed_feature_spec` and of `_valid_base_after_ungroup(...)` in
  `ungrouped_feature_spec`. In both builders, make that argument
  `_reference_to_keep(fitted, spec, term) if keep_reference else spec.base`.
  - A draft's own `_base_level` is `""`, and the fitted reference ignores a
    reference a waiting step set. Either read is wrong once `spec` is the draft.
  - If B1 also maps that value to its native type through
    `{str(v): v for v in spec._levels}`, delete that mapping: `_rebuilt_categorical`
    now maps the final base, and `spec._levels` is empty on a draft.

  **3e. `reference_feature_spec`.** Replace the whole function (:176-209) with:

```python
def reference_feature_spec(
    model, term: EditableTerm, level: str, *, X, draft_spec=None
) -> tuple[Any, dict[str, Any]]:
    """Return a fresh replacement spec whose reference is the displayed ``level``.

    ``draft_spec`` is the term's spec as waiting changes leave it (None: the
    fitted spec), so a reference can name a group a waiting collapse made.
    """
    fitted = model._specs[term.name]
    spec = fitted if draft_spec is None else draft_spec
    if not isinstance(spec, Categorical | OrderedCategorical):
        raise EditorTypeError(
            f"Set reference is only available for categorical terms, got {term.name!r}."
        )
    _require_not_interaction_parent(model, term.name, operation="set the reference level")
    grouping = getattr(spec, "_grouping", None)
    label = _fitted_level_label(spec, grouping, term, level)
    frame = as_eager_frame(X)
    frame.require_columns((term.name,))
    values = frame.column_array(term.name)
    if isinstance(spec, OrderedCategorical):
        replacement = rebuilt_ordered_spec(spec, grouping=grouping, base=label, data=values)
    else:
        # Fitted levels keep their native type (an integer level stays 3, not "3").
        replacement = _rebuilt_categorical(
            spec, fitted, base=label, grouping=grouping, data=values
        )
    metadata = {
        "format": "superglm.editor.reference_level.v1",
        "term": term.name,
        "level": label,
        "label": f"set reference of {term.name} to {label}",
        "message": (
            f"The reference level of {term.name} was set to {label} and the full model was refit."
        ),
    }
    return replacement, metadata
```

  **3f. `shapes.py`.**
  - Signature of `shaped_feature_spec` (:97-99):
    `    model, name: str, *, lo, hi, degree: int, join: str = "tangent", X` →
    `    model, name: str, *, lo, hi, degree: int, join: str = "tangent", X, draft_spec=None`.
  - Before the closing `"""` of its docstring, add:
    `    ``draft_spec`` is the term's spec as waiting changes leave it (None: the`
    `    fitted spec); a numeric draft is the unfitted spline an earlier waiting`
    `    shape built, and keeps the fitted knots and boundary it states.`
  - Then in its body:

```python
    if join == "tangent" and _source_spline(model._specs[name]).degree < 2:
        raise EditorValueError(_LINEAR_TANGENT)
    spec = model._specs[name]
    ordered = isinstance(spec, OrderedCategorical)
```
→
```python
    spec = model._specs[name] if draft_spec is None else draft_spec
    if join == "tangent" and _source_spline(spec).degree < 2:
        raise EditorValueError(_LINEAR_TANGENT)
    ordered = isinstance(spec, OrderedCategorical)
```
and
```python
    else:
        source, knots, boundary = spec, spec.fitted_base_knots, spec.fitted_boundary
```
→
```python
    else:
        source = spec
        knots, boundary = _free_geometry(spec)
```
In `_numeric_edges`:
```python
    boundary = spec.fitted_boundary
    return _snapped_edge(boundary, float(lo), -1), _snapped_edge(boundary, float(hi), 1)
```
→
```python
    boundary = _free_geometry(spec)[1]
    return _snapped_edge(boundary, float(lo), -1), _snapped_edge(boundary, float(hi), 1)
```
and add after `_numeric_edges`:

```python
def _free_geometry(spec) -> tuple[Any, tuple[float, float]]:
    """The base knots and boundary a numeric term keeps when it is shaped.

    A fitted spline reports them. A draft, the unfitted spline an earlier
    waiting shape built (``_shaped_spline``), states the fitted ones it kept.
    """
    if spec.fitted_boundary is not None:
        return spec.fitted_base_knots, spec.fitted_boundary
    return spec._explicit_knots, spec._explicit_boundary
```

  `_unavailable_reason(model, name)` keeps reading the in-force spec: a draft has
  the same degree, penalty orders, constraint and interaction users.

- [ ] **Step 4: Run tests, expect PASS.**
  ```
  ./.venv/bin/python -m pytest tests/test_editor_staging.py tests/test_editor_structure.py tests/test_editor.py tests/test_editor_validation_errors.py tests/test_ordered_categorical_specials_editor.py tests/test_piecewise_editor.py -q -n 8
  ./.venv/bin/python -m ruff check src/superglm/editor tests/test_editor_staging.py
  ./.venv/bin/python -m ruff format --check src/superglm/editor tests/test_editor_staging.py
  ```
  `test_set_reference_maps_a_numeric_level_label_to_its_native_value` and
  `test_unseen_levels_rated_at_the_reference_move_with_it` cover the rewritten
  reference builder on the fitted path.

- [ ] **Step 5: Commit.**
  ```
  git add src/superglm/editor/collapse.py src/superglm/editor/shapes.py tests/test_editor_staging.py
  git commit -m "Editor builders compose on a waiting draft and keep levels= and unseen=

  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
  ```

---

### Task A2: Staging, time-ordered undo and redo, step ids and notes

**Files:**
- Modify `src/superglm/editor/_types.py`:
  - imports, :3-9;
  - `EditRecord`, :54-64;
  - `SessionState`, :66-75;
  - `StructuralStep`, :78-89;
  - new `new_step_id` and `PendingStep`.
- Modify `src/superglm/editor/session.py`:
  - imports, :5-22;
  - constants after `_COLLAPSE_SENTENCES`, :93-96;
  - module helpers after `_causes`, :113-116;
  - `__init__`, :149-154;
  - `reset`, :365-369;
  - `undo`/`redo`, :613-650, plus a new staging block before `to_model`;
  - `reprofile_distribution`, :860-863;
  - `_push_structure`, :1117-1118;
  - `_capture_state`, :1124-1133;
  - `_step_across`, :1135-1145;
  - `replace_in_force_model`, :1169-1192;
  - `_commit`, :1534-1537;
  - `_clear_term_history`, `_trim_term_history` and `_records_without_indices`, :1548-1583.
- Modify `src/superglm/editor/payloads.py`: `undo_redo_payload` and `_next_label`, :130-142.
- Test: `tests/test_editor_staging.py`, appended.

**Interfaces:**
- Consumes: A1's builders (`draft_spec=`) and B1's `keep_reference=`.
- Produces:
  - `_types.new_step_id() -> str`
  - `PendingStep(operation, term, label, params, draft_spec, history_position, step_id, created_at, metadata, predictor)`
  - `EditRecord.step_id`, `EditRecord.created_at`, `EditRecord.label`
  - `SessionState.pending`, `SessionState.pending_redo`
  - `StructuralStep.changes`, `StructuralStep.step_id`, `StructuralStep.created_at`
  - `EditorSession.pending`, `EditorSession.pending_redo`, `EditorSession.step_notes`
  - `draft_spec(term)`
  - `stage_structural(operation, term, params, *, keep_reference=True, X=None) -> PendingStep`
  - `undo_target()`, `redo_target()`
  - `timeline_items() -> tuple[list[tuple[item, status]], list[tuple[item, status]]]`
  - `set_step_note(step_id, note) -> None`
  - `editor_history_records() -> list[dict]`

**Time order.** A waiting change remembers `history_position = len(history)`
when it was staged.
- Undo takes the waiting change when `pending[-1].history_position >= len(history)`.
  Otherwise it takes the latest edit, and only with neither does it cross an
  applied step.
- Redo mirrors it: it takes `pending_redo[-1]` when `redo_stack` is empty, or
  when `pending_redo[-1].history_position <= len(history)`.
- Anything that rewrites the live history (`reset`, a term-scoped undo) moves
  each waiting change back past the records it lost (`_rewrite_history`).
- Staging, like an edit, ends the future: it clears `redo_stack`,
  `pending_redo` and `structure_redo`.
- A waiting change undone or redone bumps neither `model_revision` nor
  `edit_epoch`. Model and curves are unchanged, so evidence stays current.

- [ ] **Step 1: Write the failing test.** Replace the import block of
  `tests/test_editor_staging.py` with:

```python
from __future__ import annotations

import re
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from superglm import Categorical, Spline, SuperGLM
from superglm.editor import EditorSession
from superglm.editor import session as session_module
from superglm.editor._types import new_step_id
from superglm.editor.collapse import (
    clone_with_replaced_features,
    collapsed_feature_spec,
    reference_feature_spec,
    ungrouped_feature_spec,
)
from superglm.editor.errors import EditorKeyError, EditorValueError
from superglm.editor.payloads import undo_redo_payload
from superglm.editor.refit import fit_refit_model
from superglm.editor.shapes import shaped_feature_spec
```

  After `_at`, add:

```python
def _session(model, centering="native"):
    return EditorSession.from_model(model, terms=["brand", "area", "age"], centering=centering)


def _count_fits(monkeypatch) -> list[object]:
    """Every refit the session fits, recorded; each still fits."""
    fits: list[object] = []
    fit = session_module.fit_refit_model

    def counted(*args, **kwargs):
        fits.append(args[1])
        return fit(*args, **kwargs)

    monkeypatch.setattr(session_module, "fit_refit_model", counted)
    return fits
```

  Append:

```python
def test_step_ids_are_seven_hex_digits_unique_in_the_process():
    ids = [new_step_id() for _ in range(10_000)]
    assert len(set(ids)) == len(ids)
    assert all(re.fullmatch(r"[0-9a-f]{7}", step_id) for step_id in ids)


def test_staging_waits_without_fitting_or_moving_the_model_revision(book, monkeypatch):
    model, _, _ = book
    session = _session(model)
    fits = _count_fits(monkeypatch)
    revision, epoch = session.model_revision, session.edit_epoch
    curves = {name: term.edited_log_effect.copy() for name, term in session.terms.items()}

    staged = session.stage_structural("collapse", "brand", {"levels": ["B11", "B10"]})

    assert fits == [] and session.model is model
    assert (session.model_revision, session.edit_epoch) == (revision, epoch)
    for name, curve in curves.items():
        np.testing.assert_array_equal(session.terms[name].edited_log_effect, curve)
    assert session.pending == [staged]
    assert (staged.operation, staged.term, staged.label) == (
        "collapse",
        "brand",
        "collapse B10 + B11 in brand",
    )
    assert staged.params == {"levels": ["B10", "B11"], "group_label": "B10+B11"}
    assert staged.history_position == 0 and re.fullmatch(r"[0-9a-f]{7}", staged.step_id)
    assert session.draft_spec("brand") is staged.draft_spec
    assert session.draft_spec("area") is model._specs["area"]
    assert undo_redo_payload(session) == {"undo": staged.label, "redo": None}


@pytest.mark.parametrize(
    ("operation", "term", "params", "error", "message"),
    [
        ("shape", "age", {"lo": 33.0, "hi": 33.0, "degree": 1}, EditorValueError,
         "Select at least two points to shape a range."),
        ("collapse", "brand", {"levels": ["B10", "B99"]}, EditorKeyError,
         "Unknown level(s) for term 'brand': ['B99']"),
        ("set_reference", "brand", {}, EditorValueError, "Missing required field: level."),
        ("merge", "brand", {}, EditorValueError, "Unknown structural change: 'merge'"),
    ],
)
def test_a_change_the_builder_refuses_is_refused_at_staging(
    book, operation, term, params, error, message
):
    model, _, _ = book
    session = _session(model)
    with pytest.raises(error) as caught:
        session.stage_structural(operation, term, params)
    assert caught.value.public_message == message
    assert session.pending == []


def test_undo_and_redo_follow_time_across_edits_and_waiting_changes(book, monkeypatch):
    model, _, _ = book
    session = _session(model)
    fits = _count_fits(monkeypatch)
    session.select_indices("age", [10, 11])
    session.shift("age", 0.1)
    staged = session.stage_structural("collapse", "brand", {"levels": ["B10", "B11"]})
    session.select_levels("area", ["C"])
    session.shift("area", -0.1)
    age_edit, area_edit = session.history
    assert undo_redo_payload(session) == {"undo": "shift area", "redo": None}

    session.undo()
    assert [record.step_id for record in session.history] == [age_edit.step_id]
    assert session.pending == [staged]
    assert undo_redo_payload(session) == {"undo": staged.label, "redo": "shift area"}
    revision = session.model_revision
    session.undo()
    # A waiting change undone moves nothing: no fit, and no new model revision.
    assert session.pending == [] and session.pending_redo == [staged]
    assert session.model_revision == revision
    session.undo()
    assert session.history == [] and session.edited_terms() == []
    assert undo_redo_payload(session) == {"undo": None, "redo": "shift age"}

    session.redo()
    assert session.edited_terms() == ["age"] and session.pending == []
    revision = session.model_revision
    session.redo()
    assert session.pending == [staged] and session.model_revision == revision
    session.redo()
    assert [record.step_id for record in session.history] == [
        age_edit.step_id,
        area_edit.step_id,
    ]
    assert session.model is model and fits == []


def test_a_new_action_ends_the_future_of_an_undone_waiting_change(book):
    model, _, _ = book
    session = _session(model)
    session.select_levels("area", ["C"])
    session.shift("area", 0.1)
    session.undo()
    staged = session.stage_structural("collapse", "brand", {"levels": ["B10", "B11"]})
    # Staging is a new action: the undone edit's future is gone.
    assert session.redo_stack == []
    session.undo()
    assert session.pending_redo == [staged]
    session.select_levels("area", ["C"])
    session.shift("area", 0.1)
    assert session.pending_redo == []
    session.redo()
    assert session.pending == []


def test_a_reset_keeps_a_waiting_change_in_its_place_among_the_edits(book):
    model, _, _ = book
    session = _session(model)
    session.select_indices("age", [10])
    session.shift("age", 0.1)
    staged = session.stage_structural("collapse", "brand", {"levels": ["B10", "B11"]})
    session.select_levels("area", ["C"])
    session.shift("area", -0.1)
    session.clear_selection("age")
    session.reset("age")

    [area_edit] = session.history
    assert session.pending[0].step_id == staged.step_id
    assert session.pending[0].history_position == 0
    # The area edit came after the waiting collapse, so Undo takes it first.
    assert session.undo_target() is area_edit
    session.undo()
    assert session.undo_target().step_id == staged.step_id


def test_notes_survive_undo_and_redo_and_travel_in_the_history_records(book):
    model, _, _ = book
    session = _session(model)
    session.select_levels("area", ["C"])
    session.shift("area", 0.1)
    [edit] = session.history
    staged = session.stage_structural("collapse", "brand", {"levels": ["B10", "B11"]})
    session.set_step_note(staged.step_id, "  One dealer network  ")
    session.set_step_note(edit.step_id, "Area C rated by hand")
    session.undo().undo().redo().redo()
    assert session.step_notes == {
        staged.step_id: "One dealer network",
        edit.step_id: "Area C rated by hand",
    }

    session.set_step_note(edit.step_id, "")
    records = session.editor_history_records()
    assert [(r["id"], r["operation"], r["status"], r["note"]) for r in records] == [
        (edit.step_id, "shift", "edit", None),
        (staged.step_id, "collapse", "waiting", "One dealer network"),
    ]
    assert (records[1]["message"], records[1]["term"]) == (staged.label, "brand")
    assert records[1]["predictor"] is None
    assert datetime.fromisoformat(records[0]["time"]).tzinfo == UTC
    with pytest.raises(EditorKeyError) as caught:
        session.set_step_note("not-an-id", "x")
    assert caught.value.public_message == "Unknown history entry."


def test_revert_sets_the_waiting_changes_aside_and_undo_brings_them_back(book):
    model, _, _ = book
    session = _session(model)
    staged = session.stage_structural("collapse", "brand", {"levels": ["B10", "B11"]})
    session.revert_to_reference_model()
    assert session.pending == []
    session.undo()
    assert session.pending == [staged] and session.model is model


def test_re_profiling_waits_until_nothing_is_waiting(book):
    model, _, _ = book
    session = _session(model)
    session.stage_structural("collapse", "brand", {"levels": ["B10", "B11"]})
    with pytest.raises(EditorValueError) as caught:
        session.reprofile_distribution("tweedie_p")
    assert caught.value.public_message == "Refit or undo the waiting changes before re-profiling."
```

  Ruff format splits the parametrize tuples over several lines. Run
  `./.venv/bin/python -m ruff format tests/test_editor_staging.py` after
  pasting.

- [ ] **Step 2: Run it, expect FAIL.**
  `./.venv/bin/python -m pytest tests/test_editor_staging.py -q`
  - On this tree, and on `origin/master` 155832e8, the module fails to import:
    `ImportError: cannot import name 'new_step_id'`.
  - With that import satisfied:
    - every staging test fails with `AttributeError: 'EditorSession' object has no attribute 'stage_structural'`;
    - the notes test fails with `AttributeError: ... 'set_step_note'`.
  - Mutation checks:
    - without `_rewrite_history`'s rebase, the reset test fails
      (`undo_target()` is the staged change, not the area edit);
    - with a pending undo that calls `_advance_model_revision`, the time-order
      test fails at `model_revision == revision`.

- [ ] **Step 3: Implement.**

  **3a. `_types.py`.** Imports (:3-9) →

```python
from __future__ import annotations

import itertools
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

# Step ids are seven hex digits, n -> (n * M + salt) mod 2**28. M is odd, so it
# is a unit modulo 2**28 and the map is a bijection: the first 2**28 counters
# give distinct ids. M scatters consecutive steps; the salt differs per process.
_STEP_ID_BITS = 28
_STEP_ID_MULTIPLIER = 0x9E3779B
_STEP_ID_SALT = secrets.randbits(_STEP_ID_BITS)
_step_counter = itertools.count()
_step_counter_lock = threading.Lock()


def new_step_id() -> str:
    """Seven hex digits naming one history entry, unique within the process."""
    with _step_counter_lock:
        n = next(_step_counter)
    return f"{(n * _STEP_ID_MULTIPLIER + _STEP_ID_SALT) % (1 << _STEP_ID_BITS):07x}"
```

  Replace `EditRecord`, `SessionState` and `StructuralStep` (:54-89) with:

```python
@dataclass
class EditRecord:
    """One reversible edit to a term."""

    term: str
    operation: str
    indices: NDArray[np.intp]
    before: NDArray
    after: NDArray
    params: dict[str, Any] = field(default_factory=dict)
    step_id: str = field(default_factory=new_step_id)
    created_at: float = field(default_factory=time.time)

    @property
    def label(self) -> str:
        """The automatic message, e.g. ``"shift area"``."""
        return f"{self.operation.replace('_', ' ')} {self.term}"


@dataclass(frozen=True)
class PendingStep:
    """One structural change waiting for a Refit (spec D1).

    ``params`` name levels, groups and edges by label, as the builder resolved
    them. ``draft_spec`` is the term's unfitted spec after this change, built
    on the term's previous draft; it is never fitted itself, because a refit
    fits a deep copy. ``history_position`` is ``len(history)`` when it was
    staged: its place in time among the live edits. ``metadata`` is the
    builder's step metadata, stamped on a refit that applies this change
    alone. ``predictor`` is reserved for the SuperLSS editor.
    """

    operation: str
    term: str
    label: str
    params: dict[str, Any]
    draft_spec: Any
    history_position: int
    step_id: str = field(default_factory=new_step_id)
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)
    predictor: str | None = None


@dataclass(frozen=True)
class SessionState:
    """The editor state on one side of a structural step."""

    model: Any
    terms: dict[str, EditableTerm]
    selection: dict[str, NDArray[np.intp]]
    level_orders: dict[str, list[str]]
    history: list[EditRecord]
    redo_stack: list[EditRecord]
    pending: tuple[PendingStep, ...] = ()
    pending_redo: tuple[PendingStep, ...] = ()


@dataclass(frozen=True)
class StructuralStep:
    """One structural change on the undo timeline and the state on its far side.

    On the undo stack ``state`` is the state before the step; on the redo
    stack it is the state the step left. ``changes`` are the waiting changes a
    refit applied, oldest first; a change refitted at once on its own is the
    step itself and shares its ``step_id``.
    """

    state: SessionState
    operation: str
    term: str | None
    label: str
    changes: tuple[PendingStep, ...] = ()
    step_id: str = field(default_factory=new_step_id)
    created_at: float = field(default_factory=time.time)
```

  **3b. `session.py` — imports and constants.**
  - `from dataclasses import replace` → add the next line
    `from datetime import UTC, datetime`.
  - `from superglm.editor._types import EditableTerm, EditRecord, SessionState, StructuralStep` →
    `from superglm.editor._types import EditableTerm, EditRecord, PendingStep, SessionState, StructuralStep`.

  After the `_COLLAPSE_SENTENCES` tuple (:93-96) add:

```python
_STAGED_SENTENCES = {"collapse": _COLLAPSE_SENTENCES, "shape": _SHAPE_SENTENCES}
# A waiting change's operation, and the operation its step carries when a
# legacy call refits it at once (``replace_with_*``).
_REFIT_AT_ONCE = {
    "collapse": "collapse_levels",
    "ungroup": "ungroup_levels",
    "set_reference": "set_reference",
    "shape": "shape_range",
}
_PROFILE_WHILE_WAITING = "Refit or undo the waiting changes before re-profiling."
_UNKNOWN_ENTRY = "Unknown history entry."
_NOTE_LIMIT = 2000
```

  After `_causes` (ends :116) add:

```python
def _param(params: dict[str, Any], name: str) -> Any:
    if name not in params:
        raise EditorValueError(f"Missing required field: {name}.")
    return params[name]


def _label_params(operation: str, metadata: dict[str, Any]) -> dict[str, Any]:
    """A waiting change's parameters by label, as its builder resolved them."""
    if operation == "collapse":
        return {"levels": list(metadata["levels"]), "group_label": metadata["group_label"]}
    if operation == "ungroup":
        return {"levels": list(metadata["levels"])}
    if operation == "set_reference":
        return {"level": metadata["level"]}
    return {name: metadata[name] for name in ("lo", "hi", "degree", "join")}


def _in_time_order(records, waiting) -> list:
    """Edits and waiting changes as they happened: one staged after k edits follows the k-th."""
    queue = list(waiting)
    ordered: list = []
    for index, record in enumerate(records):
        while queue and queue[0].history_position <= index:
            ordered.append(queue.pop(0))
        ordered.append(record)
    return [*ordered, *queue]


def _redo_order(n_live: int, records, waiting) -> list:
    """What Redo would put back, in order: ``redo_target``'s rule run forward."""
    records, waiting, ordered = list(records), list(waiting), []
    while records or waiting:
        if waiting and (not records or waiting[-1].history_position <= n_live):
            ordered.append(waiting.pop())
        else:
            ordered.append(records.pop())
            n_live += 1
    return ordered


def _with_status(items, status: str) -> list[tuple[Any, str]]:
    """Each item with its timeline status: an edit is an ``"edit"``, a change ``status``."""
    return [(item, "edit" if isinstance(item, EditRecord) else status) for item in items]
```

  **3c. `__init__`.** After
  `        self.structure_redo: list[StructuralStep] = []` (:154) add:

```python
        # Structural changes waiting for one Refit (spec D1), oldest first, and
        # the ones Undo took back, latest last. Notes are kept by step id, apart
        # from the undo states, so they survive undo and redo.
        self.pending: list[PendingStep] = []
        self.pending_redo: list[PendingStep] = []
        self.step_notes: dict[str, str] = {}
```

  **3d. `reset`** (:365-369):
```python
            self.structure_redo.clear()
            self._advance_model_revision()
        return self

    def shift(
```
→
```python
            self.structure_redo.clear()
            self.pending_redo.clear()
            self._advance_model_revision()
        return self

    def shift(
```

  **3e. `undo` and `redo`, and the staging block.** Replace `undo` and `redo`
  (:613-650) with the code below. It goes up to the blank line before
  `def to_model`.

```python
    def undo(self, term: str | None = None) -> EditorSession:
        """Undo the latest action, in time: an edit, a waiting change or an applied step.

        Undoing a waiting change puts back the term's previous draft; the model
        and the curves are unchanged, so the model revision stays. Undoing an
        applied step puts back the whole editor state from before it, edits and
        waiting changes included, without refitting. ``term`` limits the undo
        to that term's latest edit since the last structural step.
        """
        target = self.undo_target() if term is None else None
        if isinstance(target, PendingStep):
            self.pending_redo.append(self.pending.pop())
            return self
        if isinstance(target, StructuralStep):
            self._step_across(self.structure_history, self.structure_redo)
            return self
        if not self.history:
            return self
        index = self._latest_index(self.history, term)
        if index is None:
            return self
        record = self.history[index]
        self._rewrite_history([None if i == index else kept for i, kept in enumerate(self.history)])
        current = self.terms[record.term].edited_log_effect[record.indices].copy()
        self.terms[record.term].edited_log_effect[record.indices] = record.before
        self.redo_stack.append(record)
        if not np.array_equal(current, record.before):
            self._advance_model_revision()
        return self

    def redo(self, term: str | None = None) -> EditorSession:
        """Redo the latest undone action, in the order Undo took them back."""
        target = self.redo_target() if term is None else None
        if isinstance(target, PendingStep):
            self.pending.append(self.pending_redo.pop())
            return self
        if isinstance(target, StructuralStep):
            self._step_across(self.structure_redo, self.structure_history)
            return self
        if not self.redo_stack:
            return self
        record = self._pop_record(self.redo_stack, term)
        if record is None:
            return self
        current = self.terms[record.term].edited_log_effect[record.indices].copy()
        self.terms[record.term].edited_log_effect[record.indices] = record.after
        self.history.append(record)
        if not np.array_equal(current, record.after):
            self._advance_model_revision()
        return self

    def undo_target(self) -> EditRecord | PendingStep | StructuralStep | None:
        """What a plain :meth:`undo` takes next: the latest edit or waiting change, else a step."""
        if self.pending and self.pending[-1].history_position >= len(self.history):
            return self.pending[-1]
        if self.history:
            return self.history[-1]
        return self.structure_history[-1] if self.structure_history else None

    def redo_target(self) -> EditRecord | PendingStep | StructuralStep | None:
        """What a plain :meth:`redo` puts back next, in the reverse of the order Undo took."""
        if self.pending_redo and (
            not self.redo_stack or self.pending_redo[-1].history_position <= len(self.history)
        ):
            return self.pending_redo[-1]
        if self.redo_stack:
            return self.redo_stack[-1]
        return self.structure_redo[-1] if self.structure_redo else None

    # Structural changes wait for one Refit (spec D1). Each is built on its
    # term's draft, the last waiting change's spec, so changes to a term
    # compose; nothing is fitted until ``refit_pending``.
    def draft_spec(self, term: str):
        """``term``'s spec as the waiting changes leave it: the last one's draft, else the fitted spec."""
        self._require_term(term)
        waiting = self._waiting_draft(term)
        return self.model._specs[term] if waiting is None else waiting

    def stage_structural(
        self,
        operation: str,
        term: str,
        params: dict[str, Any],
        *,
        keep_reference: bool = True,
        X=None,
    ) -> PendingStep:
        """Stage one structural change to wait for :meth:`refit_pending`.

        ``operation`` is ``"collapse"`` (``levels``, optional ``group_label``),
        ``"ungroup"`` (``levels``), ``"set_reference"`` (``level``) or
        ``"shape"`` (``lo``, ``hi``, ``degree``, optional ``join``); levels are
        display labels. A change its builder refuses is refused now, with
        today's sentence. Nothing is fitted: the model, the curves and the model
        revision stay as they are. ``X`` is the frame the refit will read
        (default: the session's refit data).
        """
        editable = self._require_term(term)
        if operation not in _REFIT_AT_ONCE:
            raise EditorValueError(f"Unknown structural change: {operation!r}")
        if not isinstance(params, dict):
            raise EditorValueError("params must be an object.")
        X_ref = self._resolve_refit_data(None, None, None, None)[0] if X is None else X
        try:
            replacement, metadata = self._draft_for(
                operation, editable, params, keep_reference=keep_reference, X=X_ref
            )
        except EditorClientError:
            raise
        except ValueError as exc:
            sentence = _range_refusal(exc, _STAGED_SENTENCES.get(operation, ()))
            if sentence is None:
                raise
            raise EditorValueError(sentence) from exc
        step = PendingStep(
            operation=operation,
            term=term,
            label=str(metadata["label"]),
            params=_label_params(operation, metadata),
            draft_spec=replacement,
            history_position=len(self.history),
            metadata=dict(metadata),
        )
        self.pending.append(step)
        # A new action ends the future of whatever was undone.
        self.redo_stack.clear()
        self.pending_redo.clear()
        self.structure_redo.clear()
        return step

    def timeline_items(self) -> tuple[list[tuple[Any, str]], list[tuple[Any, str]]]:
        """Every action, oldest first, split at the current position, each with its status.

        Done: for each applied step, the edits made before it and the waiting
        changes it applied, as they happened, then the step; then the live
        edits and the changes still waiting. Undone: what Redo would put back,
        in the order it would. A change refitted at once on its own shares its
        step's id and is listed once, as the step. Statuses are ``"edit"``,
        ``"waiting"`` and ``"applied"``.
        """
        done: list[tuple[Any, str]] = []
        for step in self.structure_history:
            applied = [change for change in step.changes if change.step_id != step.step_id]
            done += _with_status(_in_time_order(step.state.history, applied), "applied")
            done.append((step, "applied"))
        done += _with_status(_in_time_order(self.history, self.pending), "waiting")
        undone = _with_status(
            _redo_order(len(self.history), self.redo_stack, self.pending_redo), "waiting"
        )
        for step in reversed(self.structure_redo):
            undone.append((step, "applied"))
            state = step.state
            undone += _with_status(
                _redo_order(len(state.history), state.redo_stack, state.pending_redo), "waiting"
            )
        return done, undone

    def set_step_note(self, step_id: str, note: str | None) -> None:
        """Write ``note`` on the timeline entry ``step_id``; an empty note removes it.

        Notes survive undo and redo, and travel with an exported model
        (:meth:`editor_history_records`).
        """
        done, undone = self.timeline_items()
        if step_id not in {item.step_id for item, _ in (*done, *undone)}:
            raise EditorKeyError(_UNKNOWN_ENTRY)
        text = "" if note is None else str(note).strip()
        if len(text) > _NOTE_LIMIT:
            raise EditorValueError(f"A note can be at most {_NOTE_LIMIT} characters.")
        if text:
            self.step_notes[step_id] = text
        else:
            self.step_notes.pop(step_id, None)

    def editor_history_records(self) -> list[dict[str, Any]]:
        """The timeline up to now, oldest first, as an exported model's ``_editor_history``.

        One dict per edit, waiting change and applied step: id, time (ISO 8601,
        UTC), operation, term, message, note, status and predictor (None until
        the SuperLSS editor names one). What Redo would put back is not history.
        """
        done, _ = self.timeline_items()
        return [
            {
                "id": item.step_id,
                "time": datetime.fromtimestamp(item.created_at, tz=UTC).isoformat(
                    timespec="seconds"
                ),
                "operation": item.operation,
                "term": item.term,
                "message": item.label,
                "note": self.step_notes.get(item.step_id),
                "status": status,
                "predictor": getattr(item, "predictor", None),
            }
            for item, status in done
        ]

    def _waiting_draft(self, term: str):
        """The last waiting change's draft for ``term``, or None when none waits."""
        return next((step.draft_spec for step in reversed(self.pending) if step.term == term), None)

    def _draft_for(self, operation, editable, params, *, keep_reference: bool, X):
        """``operation``'s builder on the term's draft: the replacement spec and its metadata."""
        draft = self._waiting_draft(editable.name)
        if operation == "collapse":
            return collapsed_feature_spec(
                self.model,
                editable,
                self._level_indices(editable, _param(params, "levels")),
                X=X,
                group_label=params.get("group_label"),
                draft_spec=draft,
                keep_reference=keep_reference,
            )
        if operation == "ungroup":
            return ungrouped_feature_spec(
                self.model,
                editable,
                self._level_indices(editable, _param(params, "levels")),
                X=X,
                draft_spec=draft,
                keep_reference=keep_reference,
            )
        if operation == "set_reference":
            level = str(_param(params, "level"))
            return reference_feature_spec(self.model, editable, level, X=X, draft_spec=draft)
        return shaped_feature_spec(
            self.model,
            editable.name,
            lo=_param(params, "lo"),
            hi=_param(params, "hi"),
            degree=_param(params, "degree"),
            join=params.get("join", "tangent"),
            X=X,
            draft_spec=draft,
        )

    def _level_indices(self, editable: EditableTerm, labels) -> NDArray[np.intp]:
        """Display indices of ``labels``, which a waiting change names its levels by."""
        if editable.levels is None:
            raise EditorTypeError(f"Term {editable.name!r} does not expose categorical levels.")
        if not isinstance(labels, list | tuple):
            raise EditorValueError("levels must be a list of level labels.")
        position = {level: index for index, level in enumerate(editable.levels)}
        missing = [label for label in labels if str(label) not in position]
        if missing:
            raise EditorKeyError(f"Unknown level(s) for term {editable.name!r}: {missing}")
        return np.array([position[str(label)] for label in labels], dtype=np.intp)
```

  **3f. `reprofile_distribution`** (:860-863):
```python
        if self.model is None:
            raise RuntimeError("Cannot reprofile without a source model.")
        if self.edited_terms():
```
→
```python
        if self.model is None:
            raise RuntimeError("Cannot reprofile without a source model.")
        if self.pending:
            # The re-profile cannot be undone, and replacing the model would drop
            # the waiting changes with no way back.
            raise EditorValueError(_PROFILE_WHILE_WAITING)
        if self.edited_terms():
```

  **3g. `_push_structure`** (:1117-1118):
```python
        step = StructuralStep(self._capture_state(), operation, term, label)
        step.state.redo_stack.clear()
```
→
```python
        state = replace(self._capture_state(), redo_stack=[], pending_redo=())
        step = StructuralStep(state, operation, term, label)
```

  **3h. `_capture_state`.** After `            redo_stack=list(self.redo_stack),` add:
```python
            pending=tuple(self.pending),
            pending_redo=tuple(self.pending_redo),
```
  **`_step_across`:**
```python
        self.history = step.state.history
        self.redo_stack = step.state.redo_stack
        self._advance_model_revision()
```
→
```python
        self.history = step.state.history
        self.redo_stack = step.state.redo_stack
        self.pending = list(step.state.pending)
        self.pending_redo = list(step.state.pending_redo)
        self._advance_model_revision()
```

  **3i. `replace_in_force_model`** (:1169-1192). Waiting changes, like
  edits, are not carried into a model put in force from outside.
  - After `        old_redo_stack = self.redo_stack` add
    `        old_pending, old_pending_redo = self.pending, self.pending_redo`.
  - After `            self.redo_stack = []` (inside `try`) add
    `            self.pending, self.pending_redo = [], []`.
  - After `            self.redo_stack = old_redo_stack` (inside `except`) add
    `            self.pending, self.pending_redo = old_pending, old_pending_redo`.
  - Append to its docstring: `Edits and waiting changes do not carry over.`

  **3j. `_commit`:**
```python
        self.redo_stack.clear()
        self.structure_redo.clear()
        if changed:
            self._advance_model_revision()

    def _pop_record(
```
→
```python
        self.redo_stack.clear()
        self.structure_redo.clear()
        self.pending_redo.clear()
        if changed:
            self._advance_model_revision()

    def _pop_record(
```

  **3k. History rewrites keep waiting changes in place.** Replace
  `_clear_term_history`, `_trim_term_history` and `_records_without_indices`
  (:1548-1583) with:

```python
    def _latest_index(self, records: list[EditRecord], term: str | None) -> int | None:
        """Index of the latest record, or of ``term``'s latest; None when there is none."""
        if term is None:
            return len(records) - 1 if records else None
        self._require_term(term)
        return next((i for i in range(len(records) - 1, -1, -1) if records[i].term == term), None)

    def _rewrite_history(self, records: list[EditRecord | None]) -> None:
        """Keep ``records``' survivors as the history; waiting changes keep their place.

        ``records`` is the history with each dropped record replaced by None. A
        change staged after k records now follows the survivors among them.
        """
        kept = np.cumsum([0, *(record is not None for record in records)])
        dropped = np.arange(kept.size) - kept

        def moved(step: PendingStep) -> PendingStep:
            shift = int(dropped[min(step.history_position, len(records))])
            return step if shift == 0 else replace(step, history_position=step.history_position - shift)

        self.pending = [moved(step) for step in self.pending]
        self.pending_redo = [moved(step) for step in self.pending_redo]
        self.history = [record for record in records if record is not None]

    def _clear_term_history(self, term: str) -> None:
        self._require_term(term)
        self._rewrite_history([None if record.term == term else record for record in self.history])
        self.redo_stack = [record for record in self.redo_stack if record.term != term]

    def _trim_term_history(self, term: str, reset_indices: NDArray[np.intp]) -> None:
        self._require_term(term)
        reset = set(np.asarray(reset_indices, dtype=np.intp).tolist())
        self._rewrite_history(
            [self._record_without_indices(record, term, reset) for record in self.history]
        )
        trimmed = (self._record_without_indices(record, term, reset) for record in self.redo_stack)
        self.redo_stack = [record for record in trimmed if record is not None]

    @staticmethod
    def _record_without_indices(
        record: EditRecord, term: str, reset: set[int]
    ) -> EditRecord | None:
        """``record`` less ``term``'s reset points, keeping its id; None when nothing is left."""
        if record.term != term:
            return record
        keep = np.array([int(index) not in reset for index in record.indices], dtype=bool)
        if not bool(np.any(keep)):
            return None
        return replace(
            record,
            indices=record.indices[keep].copy(),
            before=record.before[keep].copy(),
            after=record.after[keep].copy(),
            params=dict(record.params),
        )
```

  **3l. `payloads.py`.** Replace `undo_redo_payload` and `_next_label`
  (:130-142) with:

```python
def undo_redo_payload(session) -> dict[str, str | None]:
    """What Undo and Redo would take next, for their popovers; None leaves one disabled."""
    undo, redo = session.undo_target(), session.redo_target()
    return {
        "undo": None if undo is None else undo.label,
        "redo": None if redo is None else redo.label,
    }
```

- [ ] **Step 4: Run tests, expect PASS.**
  ```
  ./.venv/bin/python -m pytest tests/test_editor_staging.py tests/test_editor_structure.py tests/test_editor.py tests/test_editor_validation_errors.py tests/test_ordered_categorical_specials_editor.py tests/test_piecewise_editor.py tests/test_editor_security.py tests/test_editor_evidence.py tests/test_editor_evaluation_cache.py -q -n 8
  ./.venv/bin/python -m ruff check src/superglm/editor tests/test_editor_staging.py
  ./.venv/bin/python -m ruff format --check src/superglm/editor tests/test_editor_staging.py
  ```
  - The legacy `replace_with_*` path is unchanged in this task, so every
    existing structure, timeline and undo test passes as it is.
  - `test_reset_clears_term_history_without_creating_reset_edit` and
    `test_partial_reset_preserves_history_for_unreset_points` cover the
    rewritten history trims.

- [ ] **Step 5: Commit.**
  ```
  git add src/superglm/editor/_types.py src/superglm/editor/session.py src/superglm/editor/payloads.py tests/test_editor_staging.py
  git commit -m "Editor session stages structural changes; Undo follows time; history ids and notes

  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
  ```

---

### Task A3: One Refit for every waiting change, hand edits carried over, legacy calls

**Files:**
- Create `src/superglm/editor/carry.py`.
- Modify `src/superglm/editor/session.py`:
  - imports, :15-22;
  - constants after A2's;
  - `replace_with_collapsed_levels`, `replace_with_ungrouped_levels`,
    `replace_with_reference_level` and `replace_with_shaped_range` (:940-1048,
    as B1 left them);
  - `_push_structure`, :1103-1122;
  - new `refit_pending` after `stage_structural`;
  - new private helpers after `_push_structure`.
- Test: `tests/test_editor_staging.py`, appended.
- Update the tests whose subject D2 changes:
  - `tests/test_editor.py::test_collapse_levels_replaces_in_force_model_and_clears_manual_edits` (:2427-2446), renamed;
  - in `tests/test_editor_structure.py`:
    - `test_undoing_a_collapse_brings_back_the_edits_made_before_it`, :1137-1154;
    - `test_the_timeline_lists_every_action_around_the_current_position`, :1246-1287;
    - `test_a_collapse_keeps_the_edits_made_before_it_on_the_timeline`, :1290-1315.

**Interfaces:**
- Consumes:
  - A1: `clone_with_replaced_features` and the draft-aware builders;
  - A2: `stage_structural`, `PendingStep`, `StructuralStep.changes`, time-ordered undo.
- Produces:
  - `carry.carried_curve(edited: EditableTerm, refitted: EditableTerm) -> NDArray | None`
  - `EditorSession.refit_pending(*, method="auto", **refit_kwargs) -> StructuralStep`
  - `replace_with_collapsed_levels(term, *, group_label=None, keep_reference=True, **refit_kwargs) -> model`
  - `replace_with_ungrouped_levels(term, *, keep_reference=True, **refit_kwargs) -> model`
  - `replace_with_reference_level(term, level, **refit_kwargs) -> model`
  - `replace_with_shaped_range(term, *, lo, hi, degree, join="tangent", **refit_kwargs) -> model`

**Semantics.**

`refit_pending` makes one clone (`clone_with_replaced_features`, one draft
per term: the last waiting change's, which holds the earlier ones) and one
fit, `method="auto"` as today. It pushes one `StructuralStep`:
- operation `"refit_pending"`;
- label `"Refit · N changes"`;
- `changes` = the waiting list;
- its state captured with the waiting list, so Undo brings the changes back
  as waiting and Redo puts the refit back without fitting.

D2 carry-over runs before the refit's result is final:
- each edited term not restructured keeps its rows and `n_points`, so its grid
  and labels are equal (`carried_curve` checks);
- its native curve goes back exactly;
- the carried terms form one further step, `"carry_edits"` /
  `"Hand edits carried over: area, age"`, whose state is the refitted terms,
  so its Undo returns them to the refitted curves;
- edits on restructured terms are dropped; Undo of the refit restores them.

A fit-time `ValueError` leaves the waiting list, model and revision as they
were and raises the batch sentence, with the library error as `__cause__`.

The legacy `replace_with_*` calls are `stage_structural` and then the refit
of everything waiting, as one action:
- the step keeps the state from before staging, so one Undo takes the whole
  call back;
- with one change, the step is that change: same id, the legacy operation
  and label, and the builder's `_editor_step` plus `method`;
- a refusal unstages and keeps today's per-operation sentence mapping;
- the ungroup shortcut, which reuses the fit from before the latest step,
  applies only when nothing is waiting.

- [ ] **Step 1: Write the failing test.**

  Add to the import block of `tests/test_editor_staging.py`, after the
  `shapes` line:
  `from superglm.editor.terms import native_log_effect_values`.

  Append:

```python
def test_waiting_changes_on_two_terms_refit_in_one_fit_and_one_clone(book, monkeypatch):
    model, _, _ = book
    session = _session(model)
    fits = _count_fits(monkeypatch)
    clones: list[list[str]] = []
    clone = session_module.clone_with_replaced_features

    def spied(source, replacements, **kwargs):
        clones.append(sorted(replacements))
        return clone(source, replacements, **kwargs)

    monkeypatch.setattr(session_module, "clone_with_replaced_features", spied)
    session.stage_structural("collapse", "brand", {"levels": ["B10", "B11"]})
    session.stage_structural("set_reference", "brand", {"level": "B10+B11"})
    session.stage_structural("shape", "age", {"lo": 30.0, "hi": 45.0, "degree": 1})

    step = session.refit_pending(method="fit")

    assert len(fits) == 1 and clones == [["age", "brand"]]
    brand = session.model._specs["brand"]
    assert brand._base_level == "B10+B11"
    assert brand._grouping.group_to_originals["B10+B11"] == ["B10", "B11"]
    ranges = session.model._specs["age"].polynomial_ranges
    assert [(r.lo, r.hi, r.degree) for r in ranges] == [(30.0, 45.0, 1)]
    assert (step.operation, step.label, step.term) == ("refit_pending", "Refit · 3 changes", None)
    assert [change.operation for change in step.changes] == ["collapse", "set_reference", "shape"]
    assert session.pending == [] and session.structure_history == [step]


def test_collapse_then_partial_ungroup_keeps_the_reference_in_the_majority_group(
    book, monkeypatch
):
    model, _, _ = book
    session = _session(model)
    fits = _count_fits(monkeypatch)
    session.stage_structural("collapse", "brand", {"levels": ["B1", "B10", "B11"]})
    session.stage_structural("ungroup", "brand", {"levels": ["B11"]})
    session.refit_pending(method="fit")
    spec = session.model._specs["brand"]
    assert len(fits) == 1
    assert spec._grouping.group_to_originals["B1+B10"] == ["B1", "B10"]
    assert spec._base_level == "B1+B10"


def test_a_refused_refit_leaves_the_waiting_changes_and_the_model(book, monkeypatch):
    model, _, _ = book
    session = _session(model)
    with pytest.raises(EditorValueError) as caught:
        session.refit_pending()
    assert caught.value.public_message == "No changes are waiting for a refit."
    staged = session.stage_structural("collapse", "brand", {"levels": ["B10", "B11"]})
    revision = session.model_revision

    def refused(*args, **kwargs):
        raise ValueError("the solver refused")

    monkeypatch.setattr(session_module, "fit_refit_model", refused)
    with pytest.raises(EditorValueError) as caught:
        session.refit_pending(method="fit")
    assert caught.value.public_message == (
        "The refit was refused. Undo the last waiting change and try again."
    )
    assert str(caught.value.__cause__) == "the solver refused"
    assert session.pending == [staged] and session.model is model
    assert session.model_revision == revision and session.structure_history == []


def test_undo_of_a_refit_brings_its_changes_back_waiting_and_redo_fits_nothing(
    book, monkeypatch
):
    model, _, _ = book
    session = _session(model)
    fits = _count_fits(monkeypatch)
    first = session.stage_structural("collapse", "brand", {"levels": ["B10", "B11"]})
    second = session.stage_structural("shape", "age", {"lo": 30.0, "hi": 45.0, "degree": 1})
    session.refit_pending(method="fit")
    refit = session.model

    session.undo()
    assert session.model is model
    assert [step.step_id for step in session.pending] == [first.step_id, second.step_id]
    assert undo_redo_payload(session) == {"undo": second.label, "redo": "Refit · 2 changes"}
    session.redo()
    assert session.model is refit and session.pending == []
    assert len(fits) == 1


def test_hand_edits_on_terms_the_refit_left_alone_are_carried_over(book):
    model, _, _ = book
    session = _session(model)
    session.select_indices("age", [20, 21, 22])
    session.shift("age", 0.2)
    session.select_levels("area", ["C"])
    session.shift("area", -0.1)
    session.select_levels("brand", ["B2"])
    session.shift("brand", 0.05)
    held = {name: session.terms[name].edited_log_effect.copy() for name in ("area", "age")}
    session.stage_structural("collapse", "brand", {"levels": ["B10", "B11"]})

    refit_step = session.refit_pending(method="fit")
    refit = session.model

    carried = session.structure_history[-1]
    assert session.structure_history == [refit_step, carried]
    assert (carried.operation, carried.label) == (
        "carry_edits",
        "Hand edits carried over: area, age",
    )
    assert session.edited_terms() == ["area", "age"]
    # Shown natively, the carried curve is the edited one, bit for bit.
    for name, curve in held.items():
        np.testing.assert_array_equal(session.terms[name].edited_log_effect, curve)
    # brand was restructured: its edit is dropped here and comes back with the refit's Undo.
    brand = session.terms["brand"]
    np.testing.assert_array_equal(brand.edited_log_effect, brand.original_log_effect)

    session.undo()
    assert session.model is refit and session.edited_terms() == []
    session.undo()
    assert session.model is model and session.edited_terms() == ["brand", "area", "age"]
    assert [step.operation for step in session.pending] == ["collapse"]


def test_a_mean_centred_carry_keeps_the_curve_the_model_scores(book):
    model, _, _ = book
    session = EditorSession.from_model(model, terms=["brand", "age"], centering="mean")
    session.select_indices("age", [20, 21, 22])
    session.shift("age", 0.2)
    before = session.terms["age"]
    held = native_log_effect_values(before)
    shift_before = before.metadata["native_original_log_effect"] - before.original_log_effect
    session.stage_structural("collapse", "brand", {"levels": ["B10", "B11"]})
    session.refit_pending(method="fit")

    term = session.terms["age"]
    native = np.asarray(term.metadata["native_original_log_effect"])
    # The carried display is held - (native - display): one rounding each for the
    # shift and the subtraction, two more reading the curve back. Each acts on a
    # value no larger than M = |held| + |native| + |display| to first order, so
    # 4uM bounds the difference; 8uM leaves room for the second-order terms.
    u = np.finfo(np.float64).eps / 2
    bound = 8 * u * (np.abs(held) + np.abs(native) + np.abs(term.original_log_effect))
    assert np.all(np.abs(native_log_effect_values(term) - held) <= bound)
    # Copying the displayed values instead would miss by the refit's change in
    # centring constant, which is far outside that bound.
    assert np.max(np.abs((native - term.original_log_effect) - shift_before)) > bound.max()


def test_a_change_refitted_at_once_is_its_own_step_under_its_own_id(book):
    model, _, _ = book
    session = _session(model)
    session.select_levels("brand", ["B10", "B11"])
    session.replace_with_collapsed_levels("brand", method="fit")
    [step] = session.structure_history
    [change] = step.changes
    assert (step.operation, step.label, step.step_id) == (
        "collapse_levels",
        change.label,
        change.step_id,
    )
    assert session.model._editor_step["label"] == "collapse B10 + B11 in brand"
    done, _ = session.timeline_items()
    assert [item for item, _status in done] == [step]
    session.undo()
    assert session.model is model and session.pending == []


def test_the_ungroup_shortcut_waits_for_nothing_else_to_be_waiting(book, monkeypatch):
    model, _, _ = book
    session = _session(model)
    session.select_levels("brand", ["B10", "B11"])
    session.replace_with_collapsed_levels("brand", method="fit")
    staged = session.stage_structural("shape", "age", {"lo": 30.0, "hi": 45.0, "degree": 1})
    fits = _count_fits(monkeypatch)

    session.select_levels("brand", ["B10", "B11"])
    session.replace_with_ungrouped_levels("brand", method="fit")

    # The fit from before the collapse lacks the waiting shape, so it cannot be reused.
    assert len(fits) == 1 and session.model is not model
    assert session.model._specs["brand"]._grouping is None
    ranges = session.model._specs["age"].polynomial_ranges
    assert [(r.lo, r.hi, r.degree) for r in ranges] == [(30.0, 45.0, 1)]
    assert session.structure_history[-1].label == "Refit · 2 changes"
    # Undo takes the whole call back: the shape waits again, the ungroup is gone.
    session.undo()
    assert [step.step_id for step in session.pending] == [staged.step_id]
```

  Update the four existing tests.

  `tests/test_editor.py` (:2427-2446), whole function →

```python
def test_collapse_levels_replaces_in_force_model_and_carries_edits_on_other_terms(editor_model):
    session = EditorSession.from_model(editor_model, terms=["x_spline", "region"])
    session.select_indices("x_spline", [10, 11, 12])
    session.shift("x_spline", 0.4)
    edited = session.terms["x_spline"].edited_log_effect.copy()
    assert session.edited_terms() == ["x_spline"]

    session.select_levels("region", ["B", "C"])
    refit = session.replace_with_collapsed_levels("region", method="fit")

    assert session.reference_model is editor_model
    assert session.model is refit
    # The collapse left x_spline's rows and grid alone, so its hand edit is
    # carried over the refit (spec D2), as one entry of its own.
    assert session.edited_terms() == ["x_spline"]
    assert session.history == []
    assert session.structure_history[-1].label == "Hand edits carried over: x_spline"
    assert session.selection("x_spline").size == 0
    np.testing.assert_array_equal(session.terms["x_spline"].edited_log_effect, edited)
    grouping = session.model.features["region"]._grouping
    assert grouping.original_to_group["B"] == "B+C"
    assert grouping.original_to_group["C"] == "B+C"
```

  `tests/test_editor_structure.py::test_undoing_a_collapse_brings_back_the_edits_made_before_it`:
```python
    session.replace_with_collapsed_levels("region", method="fit")
    assert session.history == [] and session.edited_terms() == []
    session.undo()
    _assert_state_is(session, before)
```
→
```python
    session.replace_with_collapsed_levels("region", method="fit")
    # x kept its grid, so its smoothing is carried over; region was restructured,
    # so its shift is not. Undo takes the carry-over back, then the collapse.
    assert session.history == [] and session.edited_terms() == ["x"]
    assert [step.operation for step in session.structure_history] == [
        "collapse_levels",
        "carry_edits",
    ]
    session.undo().undo()
    _assert_state_is(session, before)
```

  `test_the_timeline_lists_every_action_around_the_current_position`, whole
  function →

```python
def test_the_timeline_lists_every_action_around_the_current_position(region_model):
    model, _ = region_model
    session = EditorSession.from_model(model, terms=["region", "x"])
    session.select_levels("region", ["D"])
    session.shift("region", 0.1)
    session.replace_with_shaped_range("x", lo=2.0, hi=4.0, degree=1, method="fit")
    # The shape left region alone, so its edit is carried over the refit.
    shape, carried = (step.label for step in session.structure_history)
    assert carried == "Hand edits carried over: region"
    session.select_indices("x", [0, 1])
    session.shift("x", -0.05)
    session.undo()

    assert _outline(session) == [
        ("edit", "shift region", False),
        ("structural", shape, False),
        ("structural", carried, False),
        ("marker", None, None),
        ("edit", "shift x", True),
    ]
    # The entries either side of the marker read as the Undo and Redo popovers do.
    assert undo_redo_payload(session) == {"undo": carried, "redo": "shift x"}
    undone = timeline_payload(session)[-1]

    session.redo()
    assert _outline(session) == [
        ("edit", "shift region", False),
        ("structural", shape, False),
        ("structural", carried, False),
        ("edit", "shift x", False),
        ("marker", None, None),
    ]
    # An edit's hash names its place in the session, whichever side of the marker it is on.
    assert timeline_payload(session)[3]["hash"] == undone["hash"]

    # Undone past the steps, they and the edits after them wait in the order Redo takes them.
    session.select_indices("x", [5, 6])
    session.smooth("x", 0.5)
    session.undo().undo().undo().undo()
    assert _outline(session) == [
        ("edit", "shift region", False),
        ("marker", None, None),
        ("structural", shape, True),
        ("structural", carried, True),
        ("edit", "shift x", True),
        ("edit", "smooth x", True),
    ]
```

  `test_a_collapse_keeps_the_edits_made_before_it_on_the_timeline`:
```python
    assert _outline(session) == [
        ("edit", "shift x", False),
        ("structural", "collapse B + C in region", False),
        ("marker", None, None),
    ]
```
→
```python
    assert _outline(session) == [
        ("edit", "shift x", False),
        ("structural", "collapse B + C in region", False),
        ("structural", "Hand edits carried over: x", False),
        ("marker", None, None),
    ]
```
  In the same test, `    assert _outline(session)[2:] == [` →
  `    assert _outline(session)[3:] == [`.

- [ ] **Step 2: Run it, expect FAIL.**
  ```
  ./.venv/bin/python -m pytest tests/test_editor_staging.py tests/test_editor_structure.py tests/test_editor.py -q -n 8 -k "refit or carried or carry or refitted or ungroup_shortcut or majority or carries_edits or brings_back_the_edits or every_action or keeps_the_edits_made"
  ```
  - New tests fail with `AttributeError: 'EditorSession' object has no attribute 'refit_pending'`.
  - The refitted-at-once test fails with `ValueError: not enough values to unpack`:
    `step.changes` is `()`.
  - On `origin/master` 155832e8 and on this tree, the four updated tests fail at
    `session.edited_terms() == [...]` / the outline: the legacy refit drops
    every hand edit (`replace_in_force_model` rebuilds the terms).
  - Mutation check: copying the display values verbatim makes the mean-centred
    test fail at the bound, by ~2e-6 (probed).

- [ ] **Step 3: Implement.**

  **3a. Create `src/superglm/editor/carry.py`:**

```python
"""Carry hand-edited curves onto a refitted model (spec D2).

A Refit re-specifies some terms and refits all of them. A term it left alone
keeps its rows and ``n_points``, so its display grid and level labels are the
same, and its hand-edited curve can go back onto the refitted model exactly.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from superglm.editor._types import EditableTerm
from superglm.editor.terms import native_log_effect_values


def carried_curve(edited: EditableTerm, refitted: EditableTerm) -> NDArray | None:
    """``edited``'s curve as ``refitted`` would display it, or None when their grids differ.

    The curve kept is the one the model scores, the native one. Shown natively
    it is ``edited``'s own values; shown mean-centred it moves by the refit's
    own centring constant, so that the scored curve does not.
    """
    if edited.size != refitted.size or edited.levels != refitted.levels:
        return None
    if (edited.x is None) != (refitted.x is None):
        return None
    if edited.x is not None and not np.array_equal(edited.x, refitted.x):
        return None
    return native_log_effect_values(edited) - _centring_offset(refitted)


def _centring_offset(term: EditableTerm) -> NDArray | float:
    """``term``'s native curve less its displayed one: zero unless it is centred for display."""
    native = term.metadata.get("native_original_log_effect")
    if native is None:
        return 0.0
    native = np.asarray(native, dtype=np.float64).ravel()
    if native.shape != term.original_log_effect.shape:
        return 0.0
    return native - np.asarray(term.original_log_effect, dtype=np.float64)
```

  **3b. `session.py` — imports and constants.**
  - Add `from superglm.editor.carry import carried_curve` after the `_types` import.
  - Add `clone_with_replaced_features,` to the `superglm.editor.collapse` import
    list, after `clone_with_replaced_feature,`.
  - After A2's `_NOTE_LIMIT = 2000` add:

```python
_REFIT_REFUSED = "The refit was refused. Undo the last waiting change and try again."
_NOTHING_WAITING = "No changes are waiting for a refit."


def _refit_label(count: int) -> str:
    return f"Refit · {count} change{'' if count == 1 else 's'}"
```

  **3c. `refit_pending`.** Insert directly after A2's `stage_structural`:

```python
    def refit_pending(self, *, method: str = "auto", **refit_kwargs: Any) -> StructuralStep:
        """Apply every waiting change in one fit, as one structural step (spec D1).

        Undo of the step brings the changes back as waiting; Redo puts the refit
        back without fitting. Hand edits on terms no change restructured are
        carried over (spec D2) as one more entry. A fit-time refusal leaves the
        waiting changes and the model as they were and raises one fixed
        sentence; Python callers keep the library's error as its cause.
        ``refit_kwargs`` are ``X``, ``y``, ``sample_weight``, ``offset``,
        ``lambda1``, ``lambda2`` and fit keywords.
        """
        try:
            return self._apply_pending(method=method, **refit_kwargs)
        except EditorClientError:
            raise
        except ValueError as exc:
            raise EditorValueError(_REFIT_REFUSED) from exc
```

  **3d. The four legacy calls.** Replace `replace_with_collapsed_levels`,
  `replace_with_ungrouped_levels`, `replace_with_reference_level` and
  `replace_with_shaped_range` with the code below, whatever B1 left in their
  bodies. Leave `refit_with_collapsed_levels`, `refit_with_ungrouped_levels`,
  `_pre_collapse_model`, `_ungroup_restores_reference_model` and
  `_refit_replacing` as they are: they are the non-staging preview API.

```python
    def replace_with_collapsed_levels(
        self,
        term: str,
        *,
        group_label: str | None = None,
        keep_reference: bool = True,
        **refit_kwargs: Any,
    ):
        """Collapse the selected levels and refit at once, as one structural step.

        This is :meth:`stage_structural` followed by the refit of every waiting
        change; Undo takes the whole call back. ``refit_kwargs`` are ``X``,
        ``y``, ``sample_weight``, ``offset``, ``method``, ``lambda1``,
        ``lambda2`` and fit keywords.
        """
        params = {"levels": self._selected_labels(term), "group_label": group_label}
        return self._stage_and_refit(
            "collapse", term, params, keep_reference=keep_reference, **refit_kwargs
        )

    def replace_with_ungrouped_levels(
        self, term: str, *, keep_reference: bool = True, **refit_kwargs: Any
    ):
        """Ungroup the selected levels and refit at once, as one structural step.

        With nothing waiting, an ungroup that removes the model's last collapsed
        group, when the model before the latest step had none, reuses that
        earlier fit instead of refitting: it is exactly the result.
        """
        if not self.pending:
            model = self._pre_collapse_model(term, **refit_kwargs)
            if model is not None:
                # Read after the shortcut has validated the selection against a
                # grouped term, in the sorted order the ungrouped spec uses.
                label = ungroup_label(term, self._selected_labels(term))
                self._put_in_force(
                    model, restructured={term}, operation="ungroup_levels", term=term, label=label
                )
                return model
        params = {"levels": self._selected_labels(term)}
        return self._stage_and_refit(
            "ungroup", term, params, keep_reference=keep_reference, **refit_kwargs
        )

    def replace_with_reference_level(self, term: str, level: str, **refit_kwargs: Any):
        """Pin ``level`` as ``term``'s reference and refit at once, as one structural step."""
        return self._stage_and_refit("set_reference", term, {"level": level}, **refit_kwargs)

    def replace_with_shaped_range(
        self, term: str, *, lo, hi, degree: int, join: str = "tangent", **refit_kwargs: Any
    ):
        """Pin ``term`` to a ``degree`` polynomial on ``[lo, hi]`` and refit at once.

        ``lo`` and ``hi`` are values on a numeric term (snapped outward to
        three significant figures of the fitted span) and band labels on an
        ordered one. ``join`` is ``"tangent"`` or ``"kink"`` (Corner).
        """
        params = {"lo": lo, "hi": hi, "degree": degree, "join": join}
        return self._stage_and_refit("shape", term, params, **refit_kwargs)
```

  **3e. `_push_structure`.** Replace the whole method (A2's version) with:

```python
    def _push_structure(
        self,
        model,
        *,
        operation: str,
        term: str | None,
        label: str,
        level_orders: dict[str, list[str]] | None = None,
        state: SessionState | None = None,
        changes: tuple[PendingStep, ...] = (),
        step_id: str | None = None,
    ):
        """Put ``model`` in force as one structural step on the undo timeline.

        The step keeps the state before it: ``state`` when the caller captured
        it earlier (a change refitted at once is staged after that capture),
        else the live one. Like any new action, it ends the future of whatever
        was undone, so the kept state holds no redo. ``changes`` are the waiting
        changes a refit applied; ``step_id`` names the step after the one change
        it stands for.
        """
        kept = replace(
            self._capture_state() if state is None else state, redo_stack=[], pending_redo=()
        )
        named = {} if step_id is None else {"step_id": step_id}
        step = StructuralStep(kept, operation, term, label, changes=tuple(changes), **named)
        self.replace_in_force_model(model, level_orders=level_orders)
        self.structure_history.append(step)
        self.structure_redo.clear()
        return model
```

  **3f. Refit, carry and legacy helpers.** Insert after `_push_structure`:

```python
    def _apply_pending(
        self,
        *,
        before: SessionState | None = None,
        alone: PendingStep | None = None,
        X=None,
        y=None,
        sample_weight=None,
        offset=None,
        method: str = "auto",
        lambda1=...,
        lambda2=...,
        **fit_kwargs: Any,
    ) -> StructuralStep:
        """Fit every waiting change in one refit and put it in force as one structural step.

        ``before`` and ``alone`` come from a change refitted at once: the step
        keeps the state from before that change was staged and, when it is the
        only change, is that change, under its id, operation and label.
        Nothing changes unless the fit succeeds.
        """
        if not self.pending:
            raise EditorValueError(_NOTHING_WAITING)
        X_ref, y_ref, sample_weight_ref, base_offset = self._resolve_refit_data(
            X, y, sample_weight, offset
        )
        if y_ref is None:
            raise RuntimeError("Fit response data was not retained on the source model.")
        changes = tuple(self.pending)
        # A later change on a term was built on the earlier ones' draft, so the
        # last one holds them all.
        drafts = {change.term: change.draft_spec for change in changes}
        refit_model = clone_with_replaced_features(
            self.model, drafts, lambda1=lambda1, lambda2=lambda2
        )
        method_used = fit_refit_model(
            self.model,
            refit_model,
            method=method,
            X=X_ref,
            y=y_ref,
            sample_weight=sample_weight_ref,
            offset=base_offset,
            fit_kwargs=fit_kwargs,
        )
        if alone is not None and len(changes) == 1 and changes[0] is alone:
            identity = {
                "operation": _REFIT_AT_ONCE[alone.operation],
                "label": alone.label,
                "step_id": alone.step_id,
            }
            refit_model._editor_step = {**alone.metadata, "method": method_used}
        else:
            label = _refit_label(len(changes))
            identity = {"operation": "refit_pending", "label": label, "step_id": None}
            refit_model._editor_step = {
                "format": "superglm.editor.refit.v1",
                "label": label,
                "changes": [dict(change.metadata) for change in changes],
                "method": method_used,
                "message": "The waiting structural changes were applied and the full model was refit.",
            }
        term = next(iter(drafts)) if len(drafts) == 1 else None
        return self._put_in_force(
            refit_model,
            restructured=set(drafts),
            state=before,
            term=term,
            changes=changes,
            **identity,
        )

    def _stage_and_refit(
        self,
        operation: str,
        term: str,
        params: dict[str, Any],
        *,
        keep_reference: bool = True,
        **refit_kwargs: Any,
    ):
        """Stage one change and refit every waiting change at once: the calls before staging.

        The step keeps the state from before the change was staged, so one Undo
        takes the whole call back and any earlier waiting changes wait again. A
        refusal leaves nothing staged; a library range refusal reads as the
        operation's own fixed sentence, as it always has.
        """
        before = self._capture_state()
        future = (list(self.redo_stack), list(self.pending_redo), list(self.structure_redo))
        change = None
        try:
            change = self.stage_structural(
                operation, term, params, keep_reference=keep_reference, X=refit_kwargs.get("X")
            )
            self._apply_pending(before=before, alone=change, **refit_kwargs)
        except BaseException as exc:
            if change is not None and self.pending and self.pending[-1] is change:
                self.pending.pop()
                self.redo_stack, self.pending_redo, self.structure_redo = future
            refused = isinstance(exc, ValueError) and not isinstance(exc, EditorClientError)
            sentence = _range_refusal(exc, _STAGED_SENTENCES.get(operation, ())) if refused else None
            if sentence is None:
                raise
            raise EditorValueError(sentence) from exc
        return self.model

    def _put_in_force(self, model, *, restructured: set[str], **step: Any) -> StructuralStep:
        """Push ``model`` as one structural step, then carry over edits on the terms it left alone."""
        previous = self.terms
        held = [name for name in self.edited_terms() if name not in restructured]
        self._push_structure(model, **step)
        pushed = self.structure_history[-1]
        self._carry_edits({name: previous[name] for name in held})
        return pushed

    def _carry_edits(self, edited: dict[str, EditableTerm]) -> None:
        """Re-apply hand-edited curves over a refit, as one entry Undo takes back (spec D2).

        ``edited`` holds the edited terms the refit did not restructure. Each
        keeps its rows and ``n_points``, so its grid and labels match and its
        curve goes back exactly (``carried_curve``); a term whose grid moved
        anyway is left at the refit. Undo of the entry returns the carried
        terms to the refitted curves; Undo of the refit then puts back every
        edit, those on restructured terms included.
        """
        carried = {}
        for name, term in edited.items():
            curve = carried_curve(term, self.terms[name])
            if curve is not None:
                carried[name] = curve
        if not carried:
            return
        refitted = self._capture_state()
        # A fresh dict of copies: the entry's state keeps the refitted terms.
        self.terms = {name: term.copy() for name, term in self.terms.items()}
        for name, curve in carried.items():
            self.terms[name].edited_log_effect = curve
        label = f"Hand edits carried over: {', '.join(carried)}"
        self.structure_history.append(StructuralStep(refitted, "carry_edits", None, label))
        self._advance_model_revision()

    def _selected_labels(self, term: str) -> list[str]:
        """The selected levels' labels in display order: how a waiting change names them."""
        editable = self._require_term(term)
        idx = np.unique(self._require_selection(term))
        if editable.levels is None:
            raise EditorTypeError(f"Term {term!r} does not expose categorical levels.")
        return [str(editable.levels[int(index)]) for index in idx]
```

  `_put_in_force` passes `state`, `term`, `changes`, `operation`, `label` and
  `step_id` straight through to `_push_structure`.

- [ ] **Step 4: Run tests, expect PASS.**
  ```
  ./.venv/bin/python -m pytest tests/test_editor_staging.py tests/test_editor_structure.py tests/test_editor.py tests/test_editor_validation_errors.py tests/test_ordered_categorical_specials_editor.py tests/test_piecewise_editor.py tests/test_editor_security.py tests/test_editor_evidence.py tests/test_editor_evaluation_cache.py -q -n 8
  ./.venv/bin/python -m pytest tests/editor/test_editor_structure_browser.py tests/editor/test_editor_refit_browser.py -m browser --run-browser -q
  ./.venv/bin/python -m ruff check src/superglm/editor tests/test_editor_staging.py tests/test_editor.py tests/test_editor_structure.py
  ./.venv/bin/python -m ruff format --check src/superglm/editor tests/test_editor_staging.py tests/test_editor.py tests/test_editor_structure.py
  ```
  Existing tests that pin the legacy behaviour must still pass unchanged:
  - one step per legacy call, with its label: `test_every_structural_step_pushes_one_restorable_entry`;
  - one Undo per call: `test_undo_rolls_back_one_collapse_at_a_time`;
  - the `_editor_step` method: `test_auto_level_refits_use_reml_when_source_model_was_reml_fit`;
  - refusal sentences and an unchanged model: `test_library_refusal_reaches_the_browser_as_the_fixed_sentence`,
    `test_a_range_leaving_too_few_values_beside_it_says_to_widen_it` and
    `test_widget_http_shape_range_answers_refusals_with_intentional_messages`;
  - a non-range error keeping its own error: `test_a_refit_failure_that_is_not_a_range_refusal_keeps_its_own_error`;
  - a swap with no fit: `test_undo_and_redo_of_a_step_swap_states_without_refitting`;
  - the dropped redo model being collected: `test_the_timeline_holds_one_state_per_step`.

  The browser structure and refit tests edit only the restructured term, so
  they see no carry-over.

- [ ] **Step 5: Commit.**
  ```
  git add src/superglm/editor/carry.py src/superglm/editor/session.py tests/test_editor_staging.py tests/test_editor.py tests/test_editor_structure.py
  git commit -m "Editor refits every waiting change in one fit and carries hand edits over (D2)

  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
  ```

---

### Task A4: Routes and payloads for staging, Refit and notes; history in the exported model

**Files:**
- Modify `src/superglm/editor/payloads.py`:
  - imports, :5-16;
  - `session_payload`, `"shape"` at :60;
  - `timeline_payload`, `_before_step` and `_after_step`, :71-102;
  - `_timeline_entry`, :105-127;
  - `_edit_label`, :145-146;
  - new `pending_payload` and per-term helpers.
- Modify `src/superglm/editor/shapes.py`: new `waiting_ranges` after `shape_payload` (:44-65).
- Modify `src/superglm/editor/widget.py`:
  - imports, :43-47;
  - `_state`, :181;
  - `_export_bytes` joblib branch, :628-636;
  - new `_stage`, `_refit_pending` and `_set_note` after `_shape_range` (:1014-1032).
- Modify `src/superglm/editor/server.py`:
  - routes after `/shape_range` (:294-306, before `return app` at :308);
  - helpers after `_level_display` (:514-518).
- Modify `src/superglm/editor/persistence.py`:
  - imports, :5;
  - `save_model`, :168-182;
  - new `with_editor_history`.
- Test:
  - `tests/test_editor_staging.py`, appended;
  - `tests/test_editor.py::test_editor_server_declares_fastapi_routes`, :6849-6880;
  - `tests/test_editor.py::test_widget_evidence_and_export_reuse_materialized_model`, :1421-1468.

**Interfaces:**
- Consumes: A2's `timeline_items()`, `pending`, `step_notes`, `set_step_note()`
  and `editor_history_records()`; A3's `refit_pending()`.
- Produces:
  - Routes:
    - `POST /stage {operation, term, params, keep_reference, level_display}` →
      `{state, summary, timing}`, with `timing.operation == "stage"`;
    - `POST /refit_pending {level_display}` → `{state, summary, timing}`;
    - `POST /note {id, note}` → `{ok: true, state}`.
  - `payloads.pending_payload(session) -> list[dict]` (state top-level `pending`).
  - Per-term `pending: {groups, ranges, reference}`.
  - Timeline entries with `id`, `time`, `note` and `status`.
  - `shapes.waiting_ranges(draft, fitted) -> list[dict]`.
  - `persistence.with_editor_history(model, records) -> model copy`.
  - Widget `_stage`, `_refit_pending` and `_set_note`.

- [ ] **Step 1: Write the failing test.** Replace the import block of
  `tests/test_editor_staging.py` with:

```python
from __future__ import annotations

import io
import json
import re
import urllib.error
from datetime import UTC, datetime

import joblib
import numpy as np
import pandas as pd
import pytest

from superglm import Categorical, Spline, SuperGLM
from superglm.editor import EditorSession
from superglm.editor import session as session_module
from superglm.editor._types import new_step_id
from superglm.editor.collapse import (
    clone_with_replaced_features,
    collapsed_feature_spec,
    reference_feature_spec,
    ungrouped_feature_spec,
)
from superglm.editor.errors import EditorKeyError, EditorValueError
from superglm.editor.payloads import timeline_payload, undo_redo_payload
from superglm.editor.refit import fit_refit_model
from superglm.editor.shapes import shaped_feature_spec
from superglm.editor.terms import native_log_effect_values
from tests.test_editor import _post_json
```

  After `_count_fits`, add:

```python
def _refused(url: str, body: dict) -> str:
    """The fixed sentence a 400 answer carries."""
    with pytest.raises(urllib.error.HTTPError) as error:
        _post_json(url, body)
    assert error.value.code == 400
    return json.loads(error.value.read().decode("utf-8"))["error"]
```

  Append:

```python
def test_widget_http_stage_waits_without_fitting_and_says_what_waits(book, monkeypatch):
    model, _, _ = book
    session = _session(model)
    fits = _count_fits(monkeypatch)
    revision = session.model_revision
    widget = session.widget()
    try:
        _post_json(
            f"{widget.url}/stage",
            {
                "operation": "collapse",
                "term": "brand",
                "params": {"levels": ["B10", "B11"]},
                "keep_reference": True,
            },
        )
        _post_json(
            f"{widget.url}/stage",
            {"operation": "set_reference", "term": "area", "params": {"level": "B"}},
        )
        payload = _post_json(
            f"{widget.url}/stage",
            {
                "operation": "shape",
                "term": "age",
                "params": {"lo": 30.0, "hi": 45.0, "degree": 1},
                "level_display": "grouped",
            },
        )
    finally:
        widget.close()

    assert fits == [] and session.model is model
    assert set(payload) == {"state", "summary", "timing"}
    assert payload["timing"]["operation"] == "stage"
    state = payload["state"]
    assert state["model_revision"] == revision
    collapse = session.pending[0]
    assert state["pending"][0] == {
        "id": collapse.step_id,
        "operation": "collapse",
        "term": "brand",
        "label": "collapse B10 + B11 in brand",
        "params": {"group_label": "B10+B11", "levels": ["B10", "B11"]},
        "note": None,
        "time": collapse.created_at,
    }
    assert [entry["term"] for entry in state["pending"]] == ["brand", "area", "age"]
    terms = state["terms"]
    assert terms["brand"]["pending"] == {
        "groups": {"B10+B11": ["B10", "B11"]},
        "ranges": [],
        "reference": None,
    }
    assert terms["area"]["pending"] == {"groups": None, "ranges": [], "reference": "B"}
    assert terms["age"]["pending"] == {
        "groups": None,
        "ranges": [{"lo": 30.0, "hi": 45.0, "degree": 1, "label": "Line", "join": "tangent"}],
        "reference": None,
    }
    assert state["undo_redo"]["undo"] == "Line 30–45 in age"
    assert [(entry["kind"], entry.get("status")) for entry in state["timeline"]] == [
        ("pending", "waiting"),
        ("pending", "waiting"),
        ("pending", "waiting"),
        ("marker", None),
    ]


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (
            {"operation": "shape", "term": "age", "params": {"lo": 33.0, "hi": 33.0, "degree": 1}},
            "Select at least two points to shape a range.",
        ),
        ({"operation": "collapse", "term": "brand", "params": ["B10"]}, "params must be an object."),
        (
            {"operation": "collapse", "term": "brand", "params": {"levels": "B10"}},
            "levels must be a list of level labels.",
        ),
        ({"operation": "merge", "term": "brand", "params": {}}, "Unknown structural change: 'merge'"),
        (
            {
                "operation": "collapse",
                "term": "brand",
                "params": {"levels": ["B10", "B11"]},
                "keep_reference": "yes",
            },
            "keep_reference must be true or false.",
        ),
    ],
)
def test_widget_http_stage_refuses_with_intentional_messages(book, body, message):
    model, _, _ = book
    session = _session(model)
    widget = session.widget()
    try:
        assert _refused(f"{widget.url}/stage", body) == message
    finally:
        widget.close()
    assert session.pending == []


def test_widget_http_refit_pending_applies_every_waiting_change_in_one_step(book, monkeypatch):
    model, _, _ = book
    session = _session(model)
    fits = _count_fits(monkeypatch)
    widget = session.widget()
    try:
        _post_json(
            f"{widget.url}/stage",
            {"operation": "collapse", "term": "brand", "params": {"levels": ["B10", "B11"]}},
        )
        _post_json(
            f"{widget.url}/stage",
            {"operation": "shape", "term": "age", "params": {"lo": 30.0, "hi": 45.0, "degree": 1}},
        )
        payload = _post_json(f"{widget.url}/refit_pending", {"level_display": "expanded"})
    finally:
        widget.close()

    assert len(fits) == 1
    assert payload["timing"]["operation"] == "refit_pending"
    state = payload["state"]
    assert state["pending"] == []
    assert [(e["kind"], e.get("label"), e.get("status")) for e in state["timeline"]] == [
        ("pending", "collapse B10 + B11 in brand", "applied"),
        ("pending", "Line 30–45 in age", "applied"),
        ("structural", "Refit · 2 changes", "applied"),
        ("marker", None, None),
    ]
    assert state["undo_redo"] == {"undo": "Refit · 2 changes", "redo": None}


def test_widget_http_refit_pending_answers_a_refusal_with_the_fixed_sentence(book, monkeypatch):
    model, _, _ = book
    session = _session(model)
    widget = session.widget()

    def refused(*args, **kwargs):
        raise ValueError("solver failed")

    try:
        _post_json(
            f"{widget.url}/stage",
            {"operation": "collapse", "term": "brand", "params": {"levels": ["B10", "B11"]}},
        )
        monkeypatch.setattr(session_module, "fit_refit_model", refused)
        assert _refused(f"{widget.url}/refit_pending", {}) == (
            "The refit was refused. Undo the last waiting change and try again."
        )
    finally:
        widget.close()
    assert len(session.pending) == 1 and session.model is model


def test_widget_http_note_is_written_on_its_entry_and_unknown_ids_are_refused(book):
    model, _, _ = book
    session = _session(model)
    widget = session.widget()
    try:
        _post_json(
            f"{widget.url}/stage",
            {"operation": "collapse", "term": "brand", "params": {"levels": ["B10", "B11"]}},
        )
        step_id = session.pending[0].step_id
        payload = _post_json(f"{widget.url}/note", {"id": step_id, "note": "Thin exposure"})
        assert payload["ok"] is True
        entry, _marker = payload["state"]["timeline"]
        assert (entry["id"], entry["note"]) == (step_id, "Thin exposure")
        assert payload["state"]["pending"][0]["note"] == "Thin exposure"
        assert _refused(f"{widget.url}/note", {"id": "not-an-id", "note": "x"}) == (
            "Unknown history entry."
        )
        assert _refused(f"{widget.url}/note", {"id": step_id, "note": 5}) == "note must be text."
    finally:
        widget.close()


def test_timeline_entries_carry_their_id_time_note_and_status(book):
    model, _, _ = book
    session = _session(model)
    session.select_levels("area", ["C"])
    session.shift("area", 0.1)
    staged = session.stage_structural("collapse", "brand", {"levels": ["B10", "B11"]})
    session.set_step_note(staged.step_id, "One dealer network")
    refit = session.refit_pending(method="fit")
    carry = session.structure_history[-1]
    session.undo()

    [edit] = refit.state.history
    timeline = timeline_payload(session)
    assert [(e["kind"], e.get("id"), e.get("status"), e.get("redo")) for e in timeline] == [
        ("edit", edit.step_id, "edit", False),
        ("pending", staged.step_id, "applied", False),
        ("structural", refit.step_id, "applied", False),
        ("marker", None, None, None),
        ("structural", carry.step_id, "applied", True),
    ]
    assert (timeline[1]["note"], timeline[1]["time"]) == ("One dealer network", staged.created_at)
    assert timeline[3] == {"kind": "marker"}


def test_exported_models_carry_the_history_and_the_session_models_are_left_alone(book, tmp_path):
    model, _, _ = book
    session = _session(model)
    session.select_levels("area", ["C"])
    session.shift("area", 0.1)
    staged = session.stage_structural("collapse", "brand", {"levels": ["B10", "B11"]})
    session.set_step_note(staged.step_id, "B10 and B11 share one dealer network")
    session.refit_pending(method="fit")
    session.stage_structural("set_reference", "area", {"level": "B"})
    expected = session.editor_history_records()

    widget = session.widget()
    try:
        downloaded = joblib.load(io.BytesIO(widget._export_bytes("joblib").data))
    finally:
        widget.close()
    saved = joblib.load(session.save_model(tmp_path / "edited.joblib"))

    for exported in (downloaded, saved):
        assert exported._editor_history == expected
    assert [(record["operation"], record["status"]) for record in expected] == [
        ("shift", "edit"),
        ("collapse", "applied"),
        ("refit_pending", "applied"),
        ("carry_edits", "applied"),
        ("set_reference", "waiting"),
    ]
    assert expected[1]["id"] == staged.step_id
    assert expected[1]["note"] == "B10 and B11 share one dealer network"
    # The export copies; the in-force and the cached edited model are untouched.
    assert session._materialized_edit_model is not None
    assert not hasattr(session.model, "_editor_history")
    assert not hasattr(session._materialized_edit_model, "_editor_history")
```

  `tests/test_editor.py::test_editor_server_declares_fastapi_routes`: after
  `    assert ("/reorder_levels", frozenset({"POST"})) in routes` add

```python
    assert ("/stage", frozenset({"POST"})) in routes
    assert ("/refit_pending", frozenset({"POST"})) in routes
    assert ("/note", frozenset({"POST"})) in routes
```

  `tests/test_editor.py::test_widget_evidence_and_export_reuse_materialized_model`:
```python
    assert downloaded
    assert dumped == [materialized, materialized]
```
→
```python
    assert downloaded
    # Both exports dump a copy of the one materialized model that carries the
    # history (spec D11); the cached model itself is left as it was.
    assert len(dumped) == 2
    assert all(model is not materialized and model._result is materialized._result for model in dumped)
    assert not hasattr(materialized, "_editor_history")
```

- [ ] **Step 2: Run it, expect FAIL.**
  ```
  ./.venv/bin/python -m pytest tests/test_editor_staging.py -q -k "widget_http or timeline_entries or exported_models"
  ./.venv/bin/python -m pytest tests/test_editor.py -q -k "declares_fastapi_routes or reuse_materialized_model"
  ```
  - `/stage`, `/refit_pending` and `/note` answer 404 (`HTTPError: Not Found`),
    so `_post_json` raises and `_refused` sees code 404.
  - The timeline test fails at its first comparison: entries carry no `id` or
    `status`, and the applied change is not listed.
  - The export test fails with `AttributeError: 'SuperGLM' object has no attribute '_editor_history'`.
  - The route test fails at `("/stage", ...)`.
  - The reuse test fails at `model is not materialized`: today the
    materialized model itself is dumped.
  - On `origin/master` 155832e8, the staging module fails earlier, at import
    (`new_step_id`). The two `tests/test_editor.py` tests fail as above.

- [ ] **Step 3: Implement.**

  **3a. `shapes.py`.** After `shape_payload` add:

```python
def waiting_ranges(draft, fitted) -> list[dict[str, Any]]:
    """The ranges ``draft`` adds or changes against the fitted spec, in axis order.

    Each is listed as the palette lists a range in force.
    """
    in_force = {(r.lo, r.hi, r.degree, r.join) for r in _current_ranges(fitted)}
    ranges = _current_ranges(draft)
    if not isinstance(draft, OrderedCategorical):
        ranges = sorted(ranges, key=lambda r: r.lo)
    return [
        {"lo": r.lo, "hi": r.hi, "degree": r.degree, "label": r.label, "join": r.join}
        for r in ranges
        if (r.lo, r.hi, r.degree, r.join) not in in_force
    ]
```

  **3b. `payloads.py`.** Imports (:5-16) →

```python
import hashlib
import json
from typing import Any

import numpy as np

from superglm.editor._types import PendingStep, StructuralStep
from superglm.editor.controls import CONTROL_HANDLE_TERM_TYPES
from superglm.editor.group_display import build_group_display
from superglm.editor.shapes import shape_payload, waiting_ranges
from superglm.editor.terms import term_from_inference
from superglm.features.categorical import Categorical
from superglm.features.ordered_categorical import OrderedCategorical
```

  In `session_payload`, after
  `            "shape": shape_payload(session.model, name, term.metadata.get("shape_support")),`
  add `            "pending": _pending_term_payload(session, name),`.

  Replace `timeline_payload`, `_before_step`, `_after_step` and
  `_timeline_entry` (:71-127) with:

```python
def timeline_payload(session) -> list[dict[str, Any]]:
    """Every action in the session, oldest first, with a marker at the current position.

    The order is the session's own (``EditorSession.timeline_items``). Before
    the marker comes what Undo would take back, latest last; after it, what
    Redo would put back, in the order it would. Each action carries its id,
    its time (seconds since the epoch), its note and its status: ``"edit"``,
    ``"waiting"`` or ``"applied"``. A structural change a Refit applies, or
    that still waits for one, is a ``"pending"`` entry.
    """
    done, undone = session.timeline_items()
    notes = getattr(session, "step_notes", {})
    entries: list[dict[str, Any]] = []
    parent: str | None = None
    for position, pair in enumerate([*done, None, *undone]):
        if pair is None:
            entries.append({"kind": "marker"})
            continue
        item, status = pair
        entry = _timeline_entry(item, parent, redo=position > len(done))
        entry.update(
            id=item.step_id,
            time=float(item.created_at),
            note=notes.get(item.step_id),
            status=status,
        )
        parent = entry.get("hash", parent)
        entries.append(entry)
    return entries


def pending_payload(session) -> list[dict[str, Any]]:
    """The structural changes waiting for a Refit, oldest first."""
    notes = getattr(session, "step_notes", {})
    return [
        {
            "id": step.step_id,
            "operation": step.operation,
            "term": step.term,
            "label": step.label,
            "params": _json_safe(step.params),
            "note": notes.get(step.step_id),
            "time": float(step.created_at),
        }
        for step in getattr(session, "pending", ())
    ]


def _timeline_entry(item, parent_hash: str | None, *, redo: bool) -> dict[str, Any]:
    if isinstance(item, StructuralStep):
        return {
            "kind": "structural",
            "operation": item.operation,
            "term": item.term,
            "label": item.label,
            "redo": redo,
        }
    if isinstance(item, PendingStep):
        return {
            "kind": "pending",
            "operation": item.operation,
            "term": item.term,
            "label": item.label,
            "params": _json_safe(item.params),
            "redo": redo,
        }
    # The hash chains through the edits in timeline order, so it names an edit
    # by its place in the session and survives its moves across the marker.
    return {
        "kind": "edit",
        "operation": str(item.operation),
        "term": str(item.term),
        "label": item.label,
        "n_points": int(np.asarray(item.indices, dtype=np.intp).size),
        "params": _json_safe(item.params),
        "hash": _record_hash(item, parent_hash),
        "redo": redo,
    }


def _pending_term_payload(session, name: str) -> dict[str, Any]:
    """What the term's waiting changes would put in force; None or empty where they change nothing.

    ``groups`` is the draft's whole grouping once a waiting collapse or
    ungroup touches the term; ``ranges`` are the shaped ranges the draft adds
    or changes; ``reference`` is the level or group the draft pins in place of
    the fitted reference.
    """
    waiting = [step for step in getattr(session, "pending", ()) if step.term == name]
    if not waiting:
        return {"groups": None, "ranges": [], "reference": None}
    draft, fitted = waiting[-1].draft_spec, session.model._specs[name]
    regrouped = any(step.operation in {"collapse", "ungroup"} for step in waiting)
    return {
        "groups": _draft_groups(draft) if regrouped else None,
        "ranges": waiting_ranges(draft, fitted),
        "reference": _waiting_reference(draft, fitted),
    }


def _draft_groups(draft) -> dict[str, list[str]]:
    grouping = getattr(draft, "_grouping", None)
    if grouping is None:
        return {}
    members = {
        str(label): [str(member) for member in grouping.group_to_originals.get(label, [])]
        for label in grouping.grouped_levels
    }
    return {label: levels for label, levels in members.items() if len(levels) > 1}


def _waiting_reference(draft, fitted) -> str | None:
    """The level or group a draft pins in place of the fitted reference, or None."""
    if not isinstance(draft, Categorical | OrderedCategorical):
        return None
    base = str(draft.base)
    if base in {"first", "most_exposed"} or base == str(getattr(fitted, "_base_level", "")):
        return None
    return base
```

  Delete `_edit_label` (:145-146); `EditRecord.label` replaces it.

  **3c. `widget.py`.**
  - Import list (:43-47):
    `from superglm.editor.payloads import (pending_payload, session_payload, timeline_payload, undo_redo_payload,)`.
  - In `_state`, after `                "timeline": timeline_payload(self.session),` add
    `                "pending": pending_payload(self.session),`.
  - `_export_bytes`, joblib branch:
```python
            model, revision = self._current_model_for_evidence()
            if model is None:
                raise RuntimeError("Export request was superseded.")
            data, validation = persistence.serialize_validated_model(
                model,
                dataset=default_metrics_dataset(self.session),
            )
```
→
```python
            model, revision = self._current_model_for_evidence()
            if model is None:
                raise RuntimeError("Export request was superseded.")
            with self._lock:
                history = self.session.editor_history_records()
            data, validation = persistence.serialize_validated_model(
                persistence.with_editor_history(model, history),
                dataset=default_metrics_dataset(self.session),
            )
```
  - After `_shape_range` (ends :1032) add:

```python
    def _stage(
        self,
        operation: str,
        term: str,
        params: dict[str, Any],
        *,
        keep_reference: bool = True,
        level_display: str = "expanded",
    ) -> dict[str, Any]:
        """Stage one structural change and return its transition envelope.

        Nothing is fitted, so the model revision, its evidence and any
        fixed-offset refit all stand; only the chart redraws what waits.
        """
        level_display = validate_level_display(level_display)
        with self._lock:
            operation_start = time.perf_counter()
            self._select_term(term)
            stage_start = time.perf_counter()
            self.session.stage_structural(operation, term, params, keep_reference=keep_reference)
            stage_end = time.perf_counter()
            self._chart_generation += 1
            return self._structural_transition(
                "stage",
                operation_start=operation_start,
                fit_start=stage_start,
                fit_end=stage_end,
                level_display=level_display,
            )

    def _refit_pending(self, *, level_display: str = "expanded") -> dict[str, Any]:
        """Refit every waiting change in one fit and return the transition envelope."""
        return self._structural_step(
            "refit_pending",
            lambda _target: self.session.refit_pending(),
            level_display=level_display,
        )

    def _set_note(self, step_id: str, note: str) -> dict[str, Any]:
        """Write a note on one timeline entry; a note changes no model, so nothing is refit."""
        with self._lock:
            self.session.set_step_note(step_id, note)
            return {"ok": True, "state": self._state()}
```

  **3d. `server.py`.** After the `/shape_range` route, before `    return app`:

```python
    @app.post("/stage")
    def stage(payload: dict[str, Any] = Body(default_factory=dict)) -> Response:
        return _guarded_json(
            lambda: widget._stage(
                str(_required(payload, "operation")),
                str(_required(payload, "term")),
                _stage_params(payload),
                keep_reference=_keep_reference(payload),
                level_display=_level_display(payload),
            )
        )

    @app.post("/refit_pending")
    def refit_pending(payload: dict[str, Any] = Body(default_factory=dict)) -> Response:
        return _guarded_json(lambda: widget._refit_pending(level_display=_level_display(payload)))

    @app.post("/note")
    def note(payload: dict[str, Any] = Body(default_factory=dict)) -> Response:
        return _guarded_json(
            lambda: widget._set_note(str(_required(payload, "id")), _note_text(payload))
        )
```

  After `_level_display` add the helpers below. If B1 already added a parser
  for `keep_reference` on `/collapse_levels` and `/ungroup_levels`, keep one of
  the two; they do the same thing.

```python
def _stage_params(payload: dict[str, Any]) -> dict[str, Any]:
    """A /stage body's ``params``, each field checked for its JSON type; the session checks the rest."""
    params = payload.get("params", {})
    if not isinstance(params, dict):
        raise EditorValueError("params must be an object.")
    parsed: dict[str, Any] = {}
    if "levels" in params:
        levels = params["levels"]
        if not isinstance(levels, list) or not all(isinstance(level, str) for level in levels):
            raise EditorValueError("levels must be a list of level labels.")
        parsed["levels"] = list(levels)
    for name in ("group_label", "level"):
        if params.get(name) is not None:
            parsed[name] = str(params[name])
    for name in ("lo", "hi"):
        if name in params:
            parsed[name] = _range_edge(params[name])
    for name in ("degree", "join"):
        if name in params:
            parsed[name] = params[name]
    return parsed


def _keep_reference(payload: dict[str, Any]) -> bool:
    value = payload.get("keep_reference", True)
    if not isinstance(value, bool):
        raise EditorValueError("keep_reference must be true or false.")
    return value


def _note_text(payload: dict[str, Any]) -> str:
    note = payload.get("note")
    if note is None:
        return ""
    if not isinstance(note, str):
        raise EditorValueError("note must be text.")
    return note
```

  `degree` and `join` pass through unparsed, as on `/shape_range`. The
  builder's `_is_shape_degree` and `EDITOR_JOINS` checks give the fixed
  sentences.

  **3e. `persistence.py`.**
  - Add `import copy` before `import io`.
  - After `edited_model_for_export` add:

```python
def with_editor_history(model, records: list[dict[str, Any]]):
    """A shallow copy of ``model`` carrying the editor's timeline as ``_editor_history`` (D11).

    The copy shares every fitted attribute and owns only the new one, so the
    in-force or cached model the editor holds is never changed.
    """
    exported = copy.copy(model)
    exported._editor_history = [dict(record) for record in records]
    return exported
```
  In `save_model`:
```python
    model = edited_model_for_export(session, model_override=model_override)
```
→
```python
    model = with_editor_history(
        edited_model_for_export(session, model_override=model_override),
        session.editor_history_records(),
    )
```

- [ ] **Step 4: Run tests, expect PASS.**
  ```
  ./.venv/bin/python -m pytest tests/test_editor_staging.py tests/test_editor_structure.py tests/test_editor.py tests/test_editor_validation_errors.py tests/test_ordered_categorical_specials_editor.py tests/test_piecewise_editor.py tests/test_editor_security.py tests/test_editor_evidence.py tests/test_editor_evaluation_cache.py -q -n 8
  ./.venv/bin/python -m pytest tests/editor -m browser --run-browser -q
  ./.venv/bin/python -m ruff check src/superglm/editor tests/test_editor_staging.py tests/test_editor.py
  ./.venv/bin/python -m ruff format --check src/superglm/editor tests/test_editor_staging.py tests/test_editor.py
  ```
  Existing tests that cover the export path and the timeline pins:
  - `test_download_serialization_does_not_hold_widget_lock` (`tests/test_editor_evidence.py`):
    the history is read under the lock before the dump, not during it;
  - `test_widget_export_file_discards_superseded_build`;
  - `test_save_model_validation_failure_creates_no_file`;
  - `test_polars_editor_category_collapse_matches_pandas_and_retains_native_frame`:
    the per-term `pending` is equal across backends;
  - the marker pins `{"kind": "marker"}` (`tests/test_editor.py:3880`, `:3893`,
    `:3912`) and `_outline` (kind, label, redo) are unchanged.

  The current history panel renders a `"pending"` entry as a generic step
  chip, so the browser suite is unaffected until A6.

- [ ] **Step 5: Commit.**
  ```
  git add src/superglm/editor/payloads.py src/superglm/editor/shapes.py src/superglm/editor/widget.py src/superglm/editor/server.py src/superglm/editor/persistence.py tests/test_editor_staging.py tests/test_editor.py
  git commit -m "Editor routes for staging, Refit and notes; history travels with the exported model

  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
  ```

---

### Performance note (for Z1)

The fitting code itself is unchanged. N staged changes run one
`fit_refit_model` call instead of N, which the tests count.
- Staging cost is the builders as today, plus one `pd.unique` over the
  term's column in `_native_levels`. It runs only for a Categorical reference,
  or an ungroup that leaves no groups.
- Carry-over copies the `EditableTerm`s once per refit that carries edits.
- Each applied step still keeps its fitted model for the session's life, as
  today. A carry step shares its refit's model, so it adds no model.

### Follow-ups (not in this section)

- `save_session`/`load_session` (JSON artifact `superglm.editor.v1`) persists
  neither waiting changes nor notes.
- A model opened from an export does not read `_editor_history` back into the
  editor.
- `/refit_pending` maps every fit-time `ValueError` to the batch sentence,
  including a library range refusal. The specific range sentence survives only
  through the legacy routes.


# Section S3: Settings, waiting changes and the History panel (frontend)

Tasks, in order: **F1** (Settings tab), **A5** (stage structural changes and
Refit), **A5b** (waiting changes in the feature list, the status line and the
export dialog), **A5c** (waiting changes on the chart), **A6** (git-style
History). All five edit `main.js` and `index.html`, so they run one at a time in
this order, after A4 (routes and payload) and before C1/E1/E2. A5b, A5c and A6
stage through the Python API (`session.stage_structural`) in their browser tests,
so a reviewer can reject any of them without touching A5's wiring.

Line numbers are at `origin/master` 155832e8 and were checked against this
worktree. Later tasks in this section edit files an earlier task already changed,
so every edit is given as an exact old snippet followed by its replacement.

## Contract amendments

These were checked against S2 (A1–A4) and S7 (I1–I3) as they stand now.

1. **The action controller changes:**
   - `StructuralMutationDescriptor` gains `blocking?: boolean`, default `true`. A
     stage runs with `blocking: false`, so there is no busy overlay and nothing
     goes inert.
   - `executeStructuralMutation` asks for no evidence when the revision did not
     change. A stage keeps the revision.
   - A 4xx failure shows the server's fixed sentence. Today every structural
     failure shows "The model change outcome is uncertain", so S2's build-time
     refusals and its batch sentence "The refit was refused. Undo the last
     waiting change and try again." would never reach the analyst.
   - `executeStateMutation` also accepts S2's `/note` answer `{ok: true, state}`
     and commits `state`. With that, `/note` keeps its contract shape, and a
     failed note gets the usual alert and Retry.
2. **`api/client.js` gains no `stage`, `refitPending` or `setNote` methods.**
   - Staging and Refit are descriptors in `summary.js`, next to the existing ones:
     `stageCollapse`, `stageUngroup`, `stageReference`, `stageShapeRange` and
     `refitPendingTransition`. They run through `actions.executeStructuralMutation`.
   - The note runs through `actions.executeStateMutation`.
   - Both reach `client.postJSON`, which is the route the internals doc
     prescribes: "through `api/client.js` and the action controller".
   - `ratingTable`, `jobStart`, `jobStatus` and `jobCancel` belong to other
     sections and are unaffected.
3. **`views/settings.js` exports more than the contract names.** On top of
   `DEFAULT_SETTINGS`, `loadSettings`, `saveSettings` and `onSettingsChange`, it
   exports:
   - `SETTINGS_STORAGE_KEY` and `BUILD_DURATION_RANGE`;
   - `normaliseSettings`, `createSettingsStore({storage})` and
     `formatBuildDuration`;
   - `renderSettingsPane` and `bindSettingsPane`.

   `DEFAULT_SETTINGS` is the contract's, `followBrowserTheme` included. As S7
   assumes, the Settings switch only calls `saveSettings({followBrowserTheme})`.
4. **F1 bridges "Follow the browser" to today's theme icon until I2.**
   - The theme key `superglm.editor.theme` decides, and `followBrowserTheme`
     mirrors it, which is S7's rule 3. The first-paint script keeps reading the
     key alone.
   - To do this, `mountThemeControl` gains `onChange(choice)` and returns
     `{choice(), setChoice(choice), destroy()}`.
   - main.js reconciles the two at startup, saves the setting when the icon
     changes the theme, and sets the theme when the setting changes.
   - I2 deletes `mountThemeControl` and this bridge. The bridge is the block
     marked `// Until I2:` in main.js. I2 also updates F1's
     `test_follow_the_browser_is_the_theme_choice`, which clicks the "Theme: Auto"
     icon.
5. **What the frontend sends to `/stage`** (S2's `_stage_params` accepts it). The
   body is `{operation, term, params, keep_reference, level_display}`. `params`
   by operation:
   - collapse and ungroup: `{levels: [labels]}`, taken from `term.levels` at the
     selected source indices. There is no `group_label`.
   - set_reference: `{level}`, which may be a group label.
   - shape: `{lo, hi, degree, join}`.

   `keep_reference` is a JSON boolean. S2 refuses anything else.
6. **What the frontend reads from S2's payload:**
   - A waiting step is a `kind: "pending"` timeline entry with
     `status: "waiting"`. A step a Refit applied keeps `kind: "pending"` with
     `status: "applied"`. The History reads `status`, never `kind`.
   - The Refit step has operation `refit_pending` and label `Refit · N changes`.
     The carry-over step has operation `carry_edits`.
   - `time` is in Unix seconds.
   - `terms[t].pending.groups` is the draft's whole grouping. The chart draws only
     the groups whose member set differs from a fitted `level_groups` entry, and
     compares members as strings.
   - `pending.ranges[].label` is the shape name, so the tag reads
     "Line · waiting for refit".
7. **Selectors and typedefs:**
   - New selectors `selectPendingSteps(state) -> readonly PendingStep[]` and
     `selectWaitingTerms(state) -> string[]`. E1/E2's Waiting filter and G's
     "Refit first" use them.
   - New typedefs: `StagedOperation`, `StageRequest`, `PendingStep` and
     `TermPending`.
   - New optional fields: `EditorSnapshot.pending?`, `TermPayload.pending?`, and
     `TimelineEntry.{id, time, note, status}`.
   - `inspectorPane` takes `'settings'`.
8. **`planCategoricalAxis` returns `labelsBottom`**, the bottom of the tick-label
   band. The waiting-group bracket row sits there.
9. **New DOM ids:**
   - `#refitPendingCount`, the badge inside `#refitPendingAction`;
   - `#exportPendingNote`;
   - `#settingsTiming`, which replaces `#advancedTiming`;
   - the switches `#settingRefitEveryChange`, `#settingKeepReference`,
     `#settingFollowBrowserTheme` and `#settingShowTimings`, and the radios
     `name="settingGroupsDefault"`.

   `#buildDurationWrap`, `#buildDuration` and `#buildDurationValue` keep their ids.
   They move into the Settings pane.

S2's seams for A5/A6 are all taken up here:
- `revertAvailable` counts a waiting entry (A5);
- `TimelineEntry` gains its fields (A5, because `revertAvailable` reads `status`);
- the route pins in `test_widget_app_shell_contains_drag_editor` (A5 and A6).

## Conventions for every task here

- Node tests: `node --test tests/editor_frontend/*.test.js`.
- Typecheck: `npm run typecheck:frontend`, or `npm run check:frontend` for both.
  It needs `node_modules/`; run `npm ci` once if it is missing. `tsc` covers
  `app/api`, `app/chart`, `app/state`, `app/views` and `tests/editor_frontend`.
- A new test file that uses a fake DOM starts with `// @ts-nocheck`, like its
  neighbours.
- Python: `./.venv/bin/python -m pytest …`. Browser:
  `./.venv/bin/python -m pytest <files> -m browser --run-browser -q`.
- Python test files keep Ruff's 100-column limit. Check them with
  `./.venv/bin/python -m ruff check tests/editor tests/test_editor.py` and
  `./.venv/bin/python -m ruff format --check tests/editor tests/test_editor.py`.
- No test asserts wall time.
- Every new action is a compact control with a delayed hover popover and a Help
  entry. Nothing opens when a selection is made.

---

### Task F1: Settings tab replaces Advanced

**Files:**
- Create:
  - `src/superglm/editor/app/views/settings.js`;
  - `tests/editor_frontend/settings.test.js`;
  - `tests/editor/test_editor_settings_browser.py`.
- Modify:
  - `src/superglm/editor/app/index.html`: the Advanced tab (439–441), the
    Advanced pane (485–495) and the inspector toggle's popover (226);
  - `src/superglm/editor/app/main.js`: the imports (60–61), the build-duration
    nodes (118–119), `advancedTiming` (168), the timing tracker (193–195), the
    theme mount (220–224), `showTimingStatus`/`renderAdvancedTiming` (706–727),
    `updateHandleCount` (1290), `applyTermDefaults` (1313–1322),
    `buildDurationMs`/`updateBuildDurationLabel` (1345–1354) and the slider
    listeners (1506–1507);
  - `src/superglm/editor/app/views/theme.js`: `mountThemeControl` (100–128);
  - `src/superglm/editor/app/views/inspector.js`: `inspectorPane` (161–165);
  - `src/superglm/editor/app/api/contracts.js`: `EditorViewState.inspectorPane` (188);
  - `src/superglm/editor/app/views/help_content.js`: the Theme section (202–208)
    and a new Settings section;
  - `src/superglm/editor/app/styles/panels.css`: the header (1–2), the
    user-select list (190–196) and the Advanced rules (224–235);
  - `docs/development/internals/editor-frontend.md`: line 102.
- Test (modify):
  - `tests/editor_frontend/theme.test.js`;
  - `tests/test_editor.py`: 6616–6617, 6681–6695 and the asset list at 6758–6765;
  - `tests/editor/test_editor_workspace_browser.py`: 1860–1876.

**Interfaces:**
- Consumes: the existing `views/theme.js` choice model (`auto|light|dark`) and the
  store's `view.inspectorPane`.
- Produces:
  ```js
  // views/settings.js
  export const SETTINGS_STORAGE_KEY = "superglm.editor.settings";
  export const BUILD_DURATION_RANGE; // {min: 4000, max: 30000, step: 500}
  export const DEFAULT_SETTINGS;     // {refitEveryChange:false, keepReference:true, followBrowserTheme:true, groupsDefault:"expanded", buildDurationMs:10000, showTimings:false}
  export function normaliseSettings(value: unknown): Readonly<EditorSettings>;
  export function createSettingsStore({storage?}): {load(), save(patch), subscribe(listener) -> unsubscribe};
  export function loadSettings(): Readonly<EditorSettings>;
  export function saveSettings(patch: Partial<EditorSettings>): Readonly<EditorSettings>;
  export function onSettingsChange(listener: (settings) => void): () => void;
  export function formatBuildDuration(ms: number): string; // "10 s", "6.5 s"
  export function renderSettingsPane(nodes, {settings}): void;
  export function bindSettingsPane(nodes, {onToggle(key), onGroupsDefault(value), onBuildDuration(ms)}): {destroy()};
  // views/theme.js
  mountThemeControl({button, root, media, storage?, onChange?}) -> {choice(): ThemeChoice, setChoice(choice: string): void, destroy(): void};
  ```
  main.js gets `renderSettingsView()`. A5 adds the selection-menu label to it. It
  also gets the interim theme bridge marked `// Until I2:`.

- [ ] **Step 1: Write the failing tests**

Create `tests/editor_frontend/settings.test.js`:

```js
// @ts-nocheck

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  BUILD_DURATION_RANGE,
  DEFAULT_SETTINGS,
  SETTINGS_STORAGE_KEY,
  bindSettingsPane,
  createSettingsStore,
  formatBuildDuration,
  normaliseSettings,
  renderSettingsPane,
} from "../../src/superglm/editor/app/views/settings.js";

function memoryStorage(entries = []) {
  const store = new Map(entries);
  return {
    store,
    getItem: (key) => (store.has(key) ? store.get(key) : null),
    setItem: (key, value) => store.set(key, String(value)),
  };
}

const BLOCKED = Object.freeze({
  getItem() { throw new Error("storage disabled"); },
  setItem() { throw new Error("storage disabled"); },
});

test("the defaults are the contract's, frozen, under one key", () => {
  assert.equal(SETTINGS_STORAGE_KEY, "superglm.editor.settings");
  assert.deepEqual(DEFAULT_SETTINGS, {
    refitEveryChange: false,
    keepReference: true,
    followBrowserTheme: true,
    groupsDefault: "expanded",
    buildDurationMs: 10000,
    showTimings: false,
  });
  assert.equal(Object.isFrozen(DEFAULT_SETTINGS), true);
  assert.deepEqual(createSettingsStore({ storage: memoryStorage() }).load(), DEFAULT_SETTINGS);
});

test("a change is saved as one JSON object, reported once, and the other keys are left alone", () => {
  const storage = memoryStorage([
    ["superglm.editor.theme", "dark"],
    ["superglm.editor.shapeJoin", "kink"],
  ]);
  const settings = createSettingsStore({ storage });
  const heard = [];
  const unsubscribe = settings.subscribe((value) => heard.push(value));

  const saved = settings.save({ refitEveryChange: true, buildDurationMs: 6000 });
  assert.deepEqual(saved, { ...DEFAULT_SETTINGS, refitEveryChange: true, buildDurationMs: 6000 });
  assert.deepEqual(JSON.parse(storage.store.get(SETTINGS_STORAGE_KEY)), saved);
  assert.equal(storage.store.get("superglm.editor.theme"), "dark");
  assert.equal(storage.store.get("superglm.editor.shapeJoin"), "kink");
  assert.deepEqual(heard, [saved]);

  // Saving what is already there changes nothing and reports nothing.
  assert.strictEqual(settings.save({ refitEveryChange: true }), saved);
  assert.equal(heard.length, 1);

  unsubscribe();
  settings.save({ showTimings: true });
  assert.equal(heard.length, 1);

  // A later page reads back what this one saved.
  assert.deepEqual(createSettingsStore({ storage }).load(), { ...saved, showTimings: true });
});

test("an old, partial or hand-edited entry keeps its valid fields and defaults the rest", () => {
  const storage = memoryStorage([[SETTINGS_STORAGE_KEY, JSON.stringify({
    refitEveryChange: "yes",
    keepReference: false,
    groupsDefault: "sideways",
    buildDurationMs: 99999,
    retired: true,
  })]]);
  assert.deepEqual(createSettingsStore({ storage }).load(), {
    ...DEFAULT_SETTINGS,
    keepReference: false,
    buildDurationMs: BUILD_DURATION_RANGE.max,
  });
  assert.equal(normaliseSettings({ buildDurationMs: 6234 }).buildDurationMs, 6000);
  assert.equal(normaliseSettings({ buildDurationMs: 100 }).buildDurationMs, BUILD_DURATION_RANGE.min);
  assert.deepEqual(normaliseSettings([true]), DEFAULT_SETTINGS);
  assert.deepEqual(normaliseSettings(null), DEFAULT_SETTINGS);
  const corrupt = memoryStorage([[SETTINGS_STORAGE_KEY, "{not json"]]);
  assert.deepEqual(createSettingsStore({ storage: corrupt }).load(), DEFAULT_SETTINGS);
});

test("blocked storage gives the defaults, and a change lasts for the page without throwing", () => {
  const settings = createSettingsStore({ storage: BLOCKED });
  assert.deepEqual(settings.load(), DEFAULT_SETTINGS);
  const heard = [];
  settings.subscribe((value) => heard.push(value.refitEveryChange));
  assert.doesNotThrow(() => settings.save({ refitEveryChange: true }));
  assert.equal(settings.load().refitEveryChange, true);
  assert.deepEqual(heard, [true]);

  // Storage whose very access throws, as a blocked origin's does.
  const hostile = {
    get getItem() { throw new Error("SecurityError"); },
    get setItem() { throw new Error("SecurityError"); },
  };
  const page = createSettingsStore({ storage: hostile });
  assert.deepEqual(page.load(), DEFAULT_SETTINGS);
  assert.doesNotThrow(() => page.save({ showTimings: true }));
  assert.equal(page.load().showTimings, true);
});

class FakeElement {
  constructor({ tag = "div", dataset = {}, attributes = {}, name = "", value = "", parent = null } = {}) {
    this.tagName = tag.toUpperCase();
    this.dataset = dataset;
    this.attributes = new Map(Object.entries(attributes));
    this.name = name;
    this.value = value;
    this.checked = false;
    this.hidden = false;
    this.textContent = "";
    this.parentNode = parent;
    this.listeners = new Map();
  }

  setAttribute(name, value) {
    this.attributes.set(name, String(value));
  }

  getAttribute(name) {
    return this.attributes.get(name) ?? null;
  }

  closest(selector) {
    if (selector !== '[role="switch"][data-setting]') throw new Error(`fake DOM cannot match ${selector}`);
    for (let node = this; node; node = node.parentNode) {
      if (node.getAttribute("role") === "switch" && node.dataset.setting) return node;
    }
    return null;
  }

  contains(node) {
    for (let current = node; current; current = current.parentNode) {
      if (current === this) return true;
    }
    return false;
  }

  addEventListener(type, listener) {
    const listeners = this.listeners.get(type) ?? new Set();
    listeners.add(listener);
    this.listeners.set(type, listeners);
  }

  removeEventListener(type, listener) {
    this.listeners.get(type)?.delete(listener);
  }

  // Bubbles from `target` to the root, like the real event path.
  emit(type, target = this) {
    const event = { type, target };
    for (let node = target; node; node = node.parentNode) {
      for (const listener of node.listeners.get(type) ?? []) listener(event);
    }
  }
}

class FakeInput extends FakeElement {}

globalThis.Element = FakeElement;
globalThis.HTMLElement = FakeElement;
globalThis.HTMLInputElement = FakeInput;

const SWITCH_KEYS = ["refitEveryChange", "keepReference", "followBrowserTheme", "showTimings"];

function pane() {
  const root = new FakeElement();
  const switches = SWITCH_KEYS.map((setting) => new FakeElement({
    tag: "button",
    dataset: { setting },
    attributes: { role: "switch", "aria-checked": "false" },
    parent: root,
  }));
  const radios = ["expanded", "collapsed"].map((value) => new FakeInput({
    tag: "input",
    name: "settingGroupsDefault",
    value,
    parent: root,
  }));
  root.querySelectorAll = (selector) => {
    if (selector === '[role="switch"][data-setting]') return switches;
    if (selector === 'input[name="settingGroupsDefault"]') return radios;
    throw new Error(`fake DOM cannot match ${selector}`);
  };
  const nodes = {
    root,
    buildDuration: new FakeInput({ tag: "input", value: "10000", parent: root }),
    buildDurationValue: new FakeElement({ parent: root }),
    timing: new FakeElement({ parent: root }),
  };
  const checked = () => Object.fromEntries(
    switches.map((node) => [node.dataset.setting, node.getAttribute("aria-checked")]),
  );
  return { nodes, switches, radios, checked };
}

test("the pane shows each setting", () => {
  const { nodes, radios, checked } = pane();
  renderSettingsPane(nodes, {
    settings: {
      ...DEFAULT_SETTINGS,
      refitEveryChange: true,
      followBrowserTheme: false,
      groupsDefault: "collapsed",
      buildDurationMs: 6500,
    },
  });
  assert.deepEqual(checked(), {
    refitEveryChange: "true",
    keepReference: "true",
    followBrowserTheme: "false",
    showTimings: "false",
  });
  assert.deepEqual(radios.map((radio) => radio.checked), [false, true]);
  assert.equal(nodes.buildDuration.value, "6500");
  assert.equal(nodes.buildDurationValue.textContent, "6.5 s");
  assert.equal(nodes.timing.hidden, true);

  renderSettingsPane(nodes, { settings: { ...DEFAULT_SETTINGS, showTimings: true } });
  assert.equal(nodes.timing.hidden, false);
  assert.equal(checked().followBrowserTheme, "true");
  assert.equal(formatBuildDuration(10000), "10 s");
});

test("with storage blocked the pane renders the defaults", () => {
  const { nodes, radios, checked } = pane();
  renderSettingsPane(nodes, { settings: createSettingsStore({ storage: BLOCKED }).load() });
  assert.deepEqual(checked(), {
    refitEveryChange: "false",
    keepReference: "true",
    followBrowserTheme: "true",
    showTimings: "false",
  });
  assert.deepEqual(radios.map((radio) => radio.checked), [true, false]);
  assert.equal(nodes.buildDurationValue.textContent, "10 s");
});

test("a switch reports its key, a groups choice its value, and the slider saves on release", () => {
  const { nodes, switches, radios } = pane();
  const calls = [];
  const binding = bindSettingsPane(nodes, {
    onToggle: (key) => calls.push(["toggle", key]),
    onGroupsDefault: (value) => calls.push(["groups", value]),
    onBuildDuration: (ms) => calls.push(["build", ms]),
  });

  nodes.root.emit("click", switches[0]);
  nodes.root.emit("click", switches[2]);
  radios[1].checked = true;
  nodes.root.emit("change", radios[1]);
  nodes.buildDuration.value = "7500";
  nodes.buildDuration.emit("input");
  // While the slider moves only its label follows; nothing is saved yet.
  assert.equal(nodes.buildDurationValue.textContent, "7.5 s");
  assert.equal(calls.length, 3);
  nodes.root.emit("change", nodes.buildDuration);

  assert.deepEqual(calls, [
    ["toggle", "refitEveryChange"],
    ["toggle", "followBrowserTheme"],
    ["groups", "collapsed"],
    ["build", 7500],
  ]);
  binding.destroy();
  nodes.root.emit("click", switches[0]);
  assert.equal(calls.length, 4);
});

test("index.html carries one control per setting, and the slider's range is the module's", () => {
  const html = readFileSync(
    new URL("../../src/superglm/editor/app/index.html", import.meta.url),
    "utf8",
  );
  for (const key of SWITCH_KEYS) {
    assert.equal(html.split(`data-setting="${key}"`).length - 1, 1, key);
  }
  assert.equal(html.split('name="settingGroupsDefault"').length - 1, 2);
  const { min, max, step } = BUILD_DURATION_RANGE;
  assert.ok(html.includes(`id="buildDuration" type="range" min="${min}" max="${max}" step="${step}"`));
  assert.ok(html.includes('id="settingsPane"'));
  assert.ok(!html.includes('id="advancedPane"'));
});
```

Append to `tests/editor_frontend/theme.test.js`:

```js
test("the Settings pane reads and sets the choice, and hears every change", () => {
  const storage = memoryStorage();
  const button = new FakeButton();
  const root = { dataset: {} };
  const media = new FakeMedia(true);
  const changes = [];
  const control = mountThemeControl({
    button, root, media, storage, onChange: (choice) => changes.push(choice),
  });
  assert.equal(control.choice(), "auto");

  control.setChoice("light");
  assert.deepEqual(
    [control.choice(), root.dataset.theme, storage.getItem(THEME_STORAGE_KEY), button.dataset.choice],
    ["light", "light", "light", "light"],
  );
  // The control's own click is reported too: Light, with a dark browser, goes to it.
  button.click();
  assert.equal(control.choice(), "dark");
  control.setChoice("auto");
  assert.equal(storage.getItem(THEME_STORAGE_KEY), null);
  control.setChoice("sepia");
  assert.equal(control.choice(), "auto");
  assert.deepEqual(changes, ["light", "dark", "auto"]);

  control.destroy();
  assert.equal(button.listeners.size + media.listeners.size, 0);
});
```

Create `tests/editor/test_editor_settings_browser.py`:

```python
from __future__ import annotations

import json

import pytest

pytest.importorskip("playwright.sync_api")
pytestmark = pytest.mark.browser

SETTINGS_KEY = "superglm.editor.settings"
SWITCHES = (
    "Refit after every structural change",
    "Keep the reference level when collapsing",
    "Follow the browser's light or dark setting",
    "Request timings",
)
# What a private window or a blocked origin does: every storage call throws.
BLOCK_STORAGE = """(() => {
    const refuse = () => { throw new DOMException('The operation is insecure.', 'SecurityError'); };
    Storage.prototype.getItem = refuse;
    Storage.prototype.setItem = refuse;
    Storage.prototype.removeItem = refuse;
})()"""


def _reload(page, term: str) -> None:
    page.reload(wait_until="domcontentloaded")
    page.locator("#chart path.edited").first.wait_for()
    page.wait_for_function(
        "term => document.querySelector('#status')?.dataset.term === term", arg=term
    )


def _settings_pane(page):
    inspector = page.get_by_role("complementary", name="Model inspector")
    inspector.get_by_role("tab", name="Settings").click()
    pane = inspector.get_by_role("tabpanel", name="Settings")
    pane.wait_for(state="visible")
    return pane


def _checked(pane) -> list[str | None]:
    return [
        pane.get_by_role("switch", name=name).get_attribute("aria-checked") for name in SWITCHES
    ]


def test_settings_are_kept_under_one_key_and_take_effect(open_editor_page):
    with open_editor_page(
        selected_term="territory", collapsed_levels=("territory", ("T02", "T03"))
    ) as (page, _session):
        assert page.locator("#groupDisplayMode").input_value() == "expanded"
        pane = _settings_pane(page)
        assert _checked(pane) == ["false", "true", "true", "false"]
        assert page.locator("#settingsTiming").get_attribute("hidden") is not None

        pane.get_by_role("switch", name="Refit after every structural change").click()
        pane.get_by_role("switch", name="Request timings").click()
        pane.get_by_text("Collapsed", exact=True).click()
        page.locator("#buildDuration").evaluate(
            """node => {
                node.value = '6000';
                node.dispatchEvent(new Event('input', { bubbles: true }));
                node.dispatchEvent(new Event('change', { bubbles: true }));
            }"""
        )

        assert _checked(pane) == ["true", "true", "true", "true"]
        assert page.locator("#settingsTiming").get_attribute("hidden") is None
        assert page.locator("#buildDurationValue").text_content() == "6 s"
        stored = json.loads(page.evaluate(f"() => localStorage.getItem('{SETTINGS_KEY}')"))
        assert stored == {
            "refitEveryChange": True,
            "keepReference": True,
            "followBrowserTheme": True,
            "groupsDefault": "collapsed",
            "buildDurationMs": 6000,
            "showTimings": True,
        }
        # The other preferences keep their own keys.
        assert page.evaluate("() => localStorage.getItem('superglm.editor.theme')") is None

        _reload(page, "territory")
        # A grouped term now opens collapsed, and every choice reads back.
        assert page.locator("#groupDisplayMode").input_value() == "collapsed"
        assert page.evaluate("() => document.querySelector('#chart')._scale.displayIsCollapsed")
        pane = _settings_pane(page)
        assert _checked(pane) == ["true", "true", "true", "true"]
        assert pane.get_by_role("radio", name="Collapsed").is_checked()
        assert page.locator("#buildDuration").input_value() == "6000"


def test_follow_the_browser_is_the_theme_choice(open_editor_page):
    with open_editor_page() as (page, _session):
        page.emulate_media(color_scheme="dark")
        page.wait_for_function("() => document.documentElement.dataset.theme === 'dark'")
        pane = _settings_pane(page)
        follow = pane.get_by_role("switch", name="Follow the browser's light or dark setting")
        assert follow.get_attribute("aria-checked") == "true"

        # Off keeps the theme on screen, as a choice that outlives the page.
        follow.click()
        assert follow.get_attribute("aria-checked") == "false"
        assert page.evaluate("() => localStorage.getItem('superglm.editor.theme')") == "dark"
        page.emulate_media(color_scheme="light")
        page.wait_for_function("() => !matchMedia('(prefers-color-scheme: dark)').matches")
        assert page.evaluate("() => document.documentElement.dataset.theme") == "dark"

        # On is Auto again: the theme follows the browser and nothing is stored.
        follow.click()
        page.wait_for_function("() => document.documentElement.dataset.theme === 'light'")
        assert page.evaluate("() => localStorage.getItem('superglm.editor.theme')") is None

        # Choosing a theme in the top bar turns the switch off.
        page.get_by_role("button", name="Theme: Auto").click()
        assert follow.get_attribute("aria-checked") == "false"


def test_settings_render_their_defaults_and_still_work_with_storage_blocked(open_editor_page):
    with open_editor_page() as (page, _session):
        errors: list[str] = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.context.add_init_script(BLOCK_STORAGE)
        page.emulate_media(color_scheme="dark")
        _reload(page, "curve")

        # With nothing remembered, the theme follows the browser.
        assert page.evaluate("() => document.documentElement.dataset.theme") == "dark"
        pane = _settings_pane(page)
        assert _checked(pane) == ["false", "true", "true", "false"]
        assert pane.get_by_role("radio", name="Expanded").is_checked()
        assert page.locator("#buildDurationValue").text_content() == "10 s"

        refit_every = pane.get_by_role("switch", name="Refit after every structural change")
        refit_every.click()
        # The change holds for the page, though nothing could be stored.
        assert refit_every.get_attribute("aria-checked") == "true"
        assert errors == []
```

Update `tests/editor/test_editor_workspace_browser.py:1860-1876`:

```python
def test_inspector_uses_one_slot_for_summary_history_advanced_and_help(open_editor_page):
    with open_editor_page() as (page, _session):
        inspector = page.get_by_role("complementary", name="Model inspector")

        assert inspector.count() == 1
        assert inspector.get_by_role("tab").all_inner_texts() == [
            "Summary",
            "History",
            "Advanced",
            "Help",
        ]

        inspector.get_by_role("tab", name="Advanced").click()
        advanced = inspector.get_by_role("tabpanel", name="Advanced")
        assert advanced.is_visible()
        assert advanced.get_by_label("Build animation duration").is_visible()
        assert page.locator("#buildDurationWrap").count() == 1
```
becomes
```python
def test_inspector_uses_one_slot_for_summary_history_settings_and_help(open_editor_page):
    with open_editor_page() as (page, _session):
        inspector = page.get_by_role("complementary", name="Model inspector")

        assert inspector.count() == 1
        assert inspector.get_by_role("tab").all_inner_texts() == [
            "Summary",
            "History",
            "Settings",
            "Help",
        ]

        inspector.get_by_role("tab", name="Settings").click()
        settings = inspector.get_by_role("tabpanel", name="Settings")
        assert settings.is_visible()
        assert settings.get_by_label("Build animation duration").is_visible()
        assert page.locator("#buildDurationWrap").count() == 1
```

Update the pins in `tests/test_editor.py`. At lines 6616–6617:

```python
    assert 'id="advancedTiming"' in html
    assert "advancedTiming.textContent" in main_js
```
becomes
```python
    assert 'id="settingsTiming"' in html
    assert "settingsTiming.textContent" in main_js
```

At lines 6681–6695:

```python
def test_editor_inspector_has_summary_history_advanced_and_help_tabs():
    root = Path(__file__).resolve().parents[1] / "src/superglm/editor/app"
    html = (root / "index.html").read_text()
    main_js = (root / "main.js").read_text()
    css = (root / "styles/panels.css").read_text()
    history_js_path = root / "history.js"

    assert 'aria-label="Model inspector"' in html
    assert ">Summary</button>" in html
    assert ">History</button>" in html
    assert ">Advanced</button>" in html
    assert ">Help</button>" in html
    assert "historyFrame" in html
    assert "advancedTiming" in html
    assert html.count('id="buildDurationWrap"') == 1
```
becomes
```python
def test_editor_inspector_has_summary_history_settings_and_help_tabs():
    root = Path(__file__).resolve().parents[1] / "src/superglm/editor/app"
    html = (root / "index.html").read_text()
    main_js = (root / "main.js").read_text()
    css = (root / "styles/panels.css").read_text()
    history_js_path = root / "history.js"

    assert 'aria-label="Model inspector"' in html
    assert ">Summary</button>" in html
    assert ">History</button>" in html
    assert "</svg>Settings</button>" in html
    assert "Advanced" not in html
    assert ">Help</button>" in html
    assert "historyFrame" in html
    assert 'id="settingsPane"' in html
    assert "settingsTiming" in html
    assert html.count('id="buildDurationWrap"') == 1
    assert 'from "./views/settings.js"' in main_js
```

In the asset list in `test_widget_serves_editor_app_assets` (6764–6765):

```python
            "views/inspector.js",
            "views/help_drawer.js",
        ]:
```
becomes
```python
            "views/inspector.js",
            "views/help_drawer.js",
            "views/settings.js",
        ]:
```

- [ ] **Step 2: Run them, expect FAIL**

```bash
node --test tests/editor_frontend/settings.test.js tests/editor_frontend/theme.test.js
./.venv/bin/python -m pytest tests/test_editor.py -q -k "structural_refits_show_busy or inspector_has_summary_history or serves_editor_app_assets"
./.venv/bin/python -m pytest tests/editor/test_editor_settings_browser.py tests/editor/test_editor_workspace_browser.py -m browser --run-browser -q -k "settings or inspector_uses_one_slot"
```

The expected failures are the same on `origin/master` 155832e8 and on this branch
before F1:
- `settings.test.js` fails to load with `ERR_MODULE_NOT_FOUND` for `views/settings.js`.
- The new theme test fails with `TypeError: control.choice is not a function`.
- Each Python pin fails its first new assertion: `'id="settingsTiming"' in html`,
  `"</svg>Settings</button>" in html`, and the asset fetch of `views/settings.js`,
  which returns 404.
- Each browser test fails because `get_by_role("tab", name="Settings")` times out:
  the tab is called Advanced.

- [ ] **Step 3: Implement**

Create `src/superglm/editor/app/views/settings.js`:

```js
// @ts-check
// Editor preferences and the Settings pane. The preferences live in this
// browser under one localStorage key, as one JSON object. Storage that is
// blocked or broken never stops the page: a read falls back to the defaults
// and a change lasts for the page. The theme keeps its own key, which the
// first-paint script in index.html reads; "Follow the browser" mirrors it, and
// main.js keeps the two equal.

/** @typedef {"expanded"|"collapsed"} GroupsDefault */
/**
 * @typedef {object} EditorSettings
 * @property {boolean} refitEveryChange refit after every structural change instead of waiting for Refit
 * @property {boolean} keepReference collapsing and ungrouping keep the reference level
 * @property {boolean} followBrowserTheme the theme follows the browser's light or dark setting
 * @property {GroupsDefault} groupsDefault how a grouped term's levels are drawn when it opens
 * @property {number} buildDurationMs how long a Build animation runs
 * @property {boolean} showTimings show the request-timing readout
 */
/** @typedef {"refitEveryChange"|"keepReference"|"followBrowserTheme"|"showTimings"} SettingsFlag */
/** @typedef {Pick<Storage, 'getItem'|'setItem'>} SettingsStorage */
/**
 * @typedef {object} SettingsStore
 * @property {()=>Readonly<EditorSettings>} load
 * @property {(patch:Partial<EditorSettings>)=>Readonly<EditorSettings>} save
 * @property {(listener:(settings:Readonly<EditorSettings>)=>void)=>()=>void} subscribe
 */
/**
 * @typedef {object} SettingsPaneNodes
 * @property {HTMLElement} root the Settings pane
 * @property {HTMLInputElement} buildDuration
 * @property {HTMLElement} buildDurationValue
 * @property {HTMLElement} timing the request-timing readout
 */

export const SETTINGS_STORAGE_KEY = "superglm.editor.settings";

/** The Build animation's range in ms; the slider in index.html states the same. */
export const BUILD_DURATION_RANGE = Object.freeze({ min: 4000, max: 30000, step: 500 });

/** @type {Readonly<EditorSettings>} */
export const DEFAULT_SETTINGS = Object.freeze({
  refitEveryChange: false,
  keepReference: true,
  followBrowserTheme: true,
  groupsDefault: "expanded",
  buildDurationMs: 10000,
  showTimings: false,
});

/** @type {ReadonlySet<string>} */
const SWITCHES = new Set(["refitEveryChange", "keepReference", "followBrowserTheme", "showTimings"]);
const GROUPS_INPUT = "settingGroupsDefault";

/**
 * Settings from a stored value: each field present with a valid value, the
 * default for every other, so an old, partial or hand-edited entry still
 * loads. The Build animation's length is held to its slider's range and step.
 * @param {unknown} value
 * @returns {Readonly<EditorSettings>}
 */
export function normaliseSettings(value) {
  /** @type {Record<string, unknown>} */
  const source = value !== null && typeof value === "object" && !Array.isArray(value)
    ? /** @type {Record<string, unknown>} */ (value)
    : {};
  /** @param {SettingsFlag} key */
  const flag = (key) =>
    typeof source[key] === "boolean" ? Boolean(source[key]) : DEFAULT_SETTINGS[key];
  const groups = source.groupsDefault;
  return Object.freeze({
    refitEveryChange: flag("refitEveryChange"),
    keepReference: flag("keepReference"),
    followBrowserTheme: flag("followBrowserTheme"),
    groupsDefault: groups === "expanded" || groups === "collapsed"
      ? groups
      : DEFAULT_SETTINGS.groupsDefault,
    buildDurationMs: buildDuration(source.buildDurationMs),
    showTimings: flag("showTimings"),
  });
}

/** @param {unknown} value */
function buildDuration(value) {
  if (typeof value !== "number" || !Number.isFinite(value)) return DEFAULT_SETTINGS.buildDurationMs;
  const { min, max, step } = BUILD_DURATION_RANGE;
  return Math.min(max, Math.max(min, Math.round(value / step) * step));
}

/** @param {Readonly<EditorSettings>} left @param {Readonly<EditorSettings>} right */
function sameSettings(left, right) {
  return left.refitEveryChange === right.refitEveryChange &&
    left.keepReference === right.keepReference &&
    left.followBrowserTheme === right.followBrowserTheme &&
    left.groupsDefault === right.groupsDefault &&
    left.buildDurationMs === right.buildDurationMs &&
    left.showTimings === right.showTimings;
}

/** @param {SettingsStorage|undefined} storage @returns {Readonly<EditorSettings>} */
function readStored(storage) {
  try {
    const stored = (storage ?? localStorage).getItem(SETTINGS_STORAGE_KEY);
    return stored === null ? DEFAULT_SETTINGS : normaliseSettings(JSON.parse(stored));
  } catch {
    // Blocked, unusable or corrupt storage: the defaults.
    return DEFAULT_SETTINGS;
  }
}

/** @param {Readonly<EditorSettings>} settings @param {SettingsStorage|undefined} storage */
function writeStored(settings, storage) {
  try {
    (storage ?? localStorage).setItem(SETTINGS_STORAGE_KEY, JSON.stringify(settings));
  } catch {
    // Unusable storage: the settings last this page only.
  }
}

/**
 * A settings store over `storage`, which is localStorage when omitted and is
 * read on first use. It keeps the page's settings itself, so a change holds
 * for the page even where storage refuses it, and it tells its listeners once
 * per change.
 * @param {{storage?:SettingsStorage}} [options]
 * @returns {SettingsStore}
 */
export function createSettingsStore({ storage } = {}) {
  /** @type {Readonly<EditorSettings>|null} */
  let current = null;
  /** @type {Set<(settings:Readonly<EditorSettings>)=>void>} */
  const listeners = new Set();

  function load() {
    if (current === null) current = readStored(storage);
    return current;
  }

  /** @param {Partial<EditorSettings>} patch */
  function save(patch) {
    const previous = load();
    const next = normaliseSettings({ ...previous, ...patch });
    if (sameSettings(next, previous)) return previous;
    current = next;
    writeStored(next, storage);
    for (const listener of [...listeners]) listener(next);
    return next;
  }

  /** @param {(settings:Readonly<EditorSettings>)=>void} listener */
  function subscribe(listener) {
    listeners.add(listener);
    return () => {
      listeners.delete(listener);
    };
  }

  return Object.freeze({ load, save, subscribe });
}

const browserSettings = createSettingsStore();

/** The page's settings, read from this browser's storage on first use. */
export function loadSettings() {
  return browserSettings.load();
}

/** @param {Partial<EditorSettings>} patch */
export function saveSettings(patch) {
  return browserSettings.save(patch);
}

/** @param {(settings:Readonly<EditorSettings>)=>void} listener @returns {()=>void} */
export function onSettingsChange(listener) {
  return browserSettings.subscribe(listener);
}

/** @param {number} ms @returns {string} */
export function formatBuildDuration(ms) {
  const seconds = ms / 1000;
  return `${Number.isInteger(seconds) ? seconds : seconds.toFixed(1)} s`;
}

/** @param {string|undefined} value @returns {value is SettingsFlag} */
function isSwitch(value) {
  return value !== undefined && SWITCHES.has(value);
}

/**
 * Show the settings in the pane: each switch, the groups choice, the Build
 * animation's length, and whether the timing readout shows.
 * @param {SettingsPaneNodes} nodes
 * @param {{settings:Readonly<EditorSettings>}} state
 */
export function renderSettingsPane(nodes, { settings }) {
  for (const element of nodes.root.querySelectorAll('[role="switch"][data-setting]')) {
    if (!(element instanceof HTMLElement)) continue;
    const key = element.dataset.setting;
    if (isSwitch(key)) element.setAttribute("aria-checked", String(settings[key]));
  }
  for (const element of nodes.root.querySelectorAll(`input[name="${GROUPS_INPUT}"]`)) {
    if (element instanceof HTMLInputElement) element.checked = element.value === settings.groupsDefault;
  }
  nodes.buildDuration.value = String(settings.buildDurationMs);
  nodes.buildDurationValue.textContent = formatBuildDuration(settings.buildDurationMs);
  nodes.timing.hidden = !settings.showTimings;
}

/**
 * Bind the pane. A switch reports its setting, the groups choice its value,
 * and the slider its length when released. While it moves, only its label
 * follows. The settings themselves live with the caller, which renders them
 * back.
 * @param {SettingsPaneNodes} nodes
 * @param {{
 *   onToggle:(key:SettingsFlag)=>unknown,
 *   onGroupsDefault:(value:GroupsDefault)=>unknown,
 *   onBuildDuration:(ms:number)=>unknown
 * }} handlers
 * @returns {{destroy:()=>void}}
 */
export function bindSettingsPane(nodes, { onToggle, onGroupsDefault, onBuildDuration }) {
  /** @param {Event} event */
  function onClick(event) {
    const target = event.target instanceof Element
      ? event.target.closest('[role="switch"][data-setting]')
      : null;
    if (!(target instanceof HTMLElement) || !nodes.root.contains(target)) return;
    const key = target.dataset.setting;
    if (isSwitch(key)) onToggle(key);
  }

  /** @param {Event} event */
  function onChange(event) {
    const target = event.target;
    if (target === nodes.buildDuration) {
      onBuildDuration(Number(nodes.buildDuration.value));
      return;
    }
    if (
      target instanceof HTMLInputElement &&
      target.name === GROUPS_INPUT &&
      target.checked &&
      (target.value === "expanded" || target.value === "collapsed")
    ) {
      onGroupsDefault(target.value);
    }
  }

  function onInput() {
    nodes.buildDurationValue.textContent = formatBuildDuration(Number(nodes.buildDuration.value));
  }

  nodes.root.addEventListener("click", onClick);
  nodes.root.addEventListener("change", onChange);
  nodes.buildDuration.addEventListener("input", onInput);
  return Object.freeze({
    destroy() {
      nodes.root.removeEventListener("click", onClick);
      nodes.root.removeEventListener("change", onChange);
      nodes.buildDuration.removeEventListener("input", onInput);
    },
  });
}
```

In `src/superglm/editor/app/views/theme.js`, replace `mountThemeControl` (lines 100–128):

```js
/**
 * Mount the control: apply the remembered choice, cycle it on a click, and
 * follow the browser's setting while the choice is Auto.
 * @param {{button:HTMLElement, root:HTMLElement, media:DarkMedia, storage?:ThemeStorage}} options
 * @returns {{destroy:()=>void}}
 */
export function mountThemeControl({ button, root, media, storage }) {
  let choice = readThemeChoice(storage);

  function render() {
    root.dataset.theme = resolveTheme(choice, media.matches);
    renderThemeControl(button, choice, media.matches);
  }

  function onClick() {
    choice = nextThemeChoice(choice, media.matches);
    storeThemeChoice(choice, storage);
    render();
  }

  button.addEventListener("click", onClick);
  media.addEventListener("change", render);
  render();
  return Object.freeze({
    destroy() {
      button.removeEventListener("click", onClick);
      media.removeEventListener("change", render);
    },
  });
}
```
with
```js
/**
 * Mount the control: apply the remembered choice, cycle it on a click, and
 * follow the browser's setting while the choice is Auto. Settings' "Follow
 * the browser" mirrors the choice through the returned handle, and
 * `onChange` hears every change, the control's own clicks included. I2
 * replaces this control with the DAY/NIGHT switch.
 * @param {{button:HTMLElement, root:HTMLElement, media:DarkMedia, storage?:ThemeStorage,
 *   onChange?:(choice:ThemeChoice)=>void}} options
 * @returns {{choice:()=>ThemeChoice, setChoice:(choice:string)=>void, destroy:()=>void}}
 */
export function mountThemeControl({ button, root, media, storage, onChange = () => {} }) {
  let choice = readThemeChoice(storage);

  function render() {
    root.dataset.theme = resolveTheme(choice, media.matches);
    renderThemeControl(button, choice, media.matches);
  }

  /** @param {string} next */
  function setChoice(next) {
    if (!isThemeChoice(next)) return;
    choice = next;
    storeThemeChoice(choice, storage);
    render();
    onChange(choice);
  }

  function onClick() {
    setChoice(nextThemeChoice(choice, media.matches));
  }

  button.addEventListener("click", onClick);
  media.addEventListener("change", render);
  render();
  return Object.freeze({
    choice: () => choice,
    setChoice,
    destroy() {
      button.removeEventListener("click", onClick);
      media.removeEventListener("change", render);
    },
  });
}
```

In `src/superglm/editor/app/views/inspector.js` (line 162):

```js
  return value === "summary" || value === "history" || value === "advanced" || value === "help"
```
becomes
```js
  return value === "summary" || value === "history" || value === "settings" || value === "help"
```

In `src/superglm/editor/app/api/contracts.js` (line 188):

```js
 * @property {'summary'|'history'|'advanced'|'help'} inspectorPane
```
becomes
```js
 * @property {'summary'|'history'|'settings'|'help'} inspectorPane
```

In `src/superglm/editor/app/index.html`, change the inspector toggle's popover (226):

```html
          data-popover-body="Show or hide the summary, history, advanced controls and help.">
```
becomes
```html
          data-popover-body="Show or hide the summary, history, settings and help.">
```

The Advanced tab (439–441):

```html
          <button id="advancedTab" class="sidepanel-tab" type="button" role="tab"
            data-inspector-tab="advanced" aria-selected="false" aria-controls="advancedPane"
            tabindex="-1">Advanced</button>
```
becomes
```html
          <button id="settingsTab" class="sidepanel-tab" type="button" role="tab"
            data-inspector-tab="settings" aria-selected="false" aria-controls="settingsPane"
            tabindex="-1"><svg class="sidepanel-tab-icon" viewBox="0 0 24 24" aria-hidden="true">
              <circle cx="12" cy="12" r="3"></circle>
              <path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z"></path>
            </svg>Settings</button>
```

The Advanced pane (485–495):

```html
      <div id="advancedPane" class="sidepanel-pane" role="tabpanel"
        aria-labelledby="advancedTab" data-inspector-pane="advanced" hidden>
        <h2>Advanced editor controls</h2>
        <label id="buildDurationWrap" class="build-duration">
          <span>Build animation duration</span>
          <input id="buildDuration" type="range" min="4000" max="30000" step="500"
            value="10000" aria-label="Build animation duration">
          <output id="buildDurationValue">10s</output>
        </label>
        <div id="advancedTiming" class="advanced-timing" aria-live="polite"></div>
      </div>
```
becomes
```html
      <div id="settingsPane" class="sidepanel-pane settings-pane" role="tabpanel"
        aria-labelledby="settingsTab" data-inspector-pane="settings" hidden>
        <h3 class="settings-heading">Refitting</h3>
        <div class="setting-row">
          <div class="setting-text">
            <span id="settingRefitEveryChangeLabel" class="setting-title">Refit after every structural change</span>
            <span id="settingRefitEveryChangeHint" class="setting-hint">Off: Collapse, Ungroup, Set reference and the shapes wait for the Refit button, so you can make several in one go.</span>
          </div>
          <button id="settingRefitEveryChange" class="setting-switch" type="button" role="switch"
            aria-checked="false" aria-labelledby="settingRefitEveryChangeLabel"
            aria-describedby="settingRefitEveryChangeHint" data-setting="refitEveryChange"></button>
        </div>
        <div class="setting-row">
          <div class="setting-text">
            <span id="settingKeepReferenceLabel" class="setting-title">Keep the reference level when collapsing</span>
            <span id="settingKeepReferenceHint" class="setting-hint">The reference stays where it is. Collapse it into a group and that group becomes the reference.</span>
          </div>
          <button id="settingKeepReference" class="setting-switch" type="button" role="switch"
            aria-checked="true" aria-labelledby="settingKeepReferenceLabel"
            aria-describedby="settingKeepReferenceHint" data-setting="keepReference"></button>
        </div>
        <h3 class="settings-heading">Display</h3>
        <div class="setting-row">
          <div class="setting-text">
            <span id="settingFollowBrowserThemeLabel" class="setting-title">Follow the browser's light or dark setting</span>
            <span id="settingFollowBrowserThemeHint" class="setting-hint">On until you choose a theme in the top bar.</span>
          </div>
          <button id="settingFollowBrowserTheme" class="setting-switch" type="button" role="switch"
            aria-checked="true" aria-labelledby="settingFollowBrowserThemeLabel"
            aria-describedby="settingFollowBrowserThemeHint" data-setting="followBrowserTheme"></button>
        </div>
        <div class="setting-row">
          <div class="setting-text">
            <span id="settingGroupsDefaultLabel" class="setting-title">Groups shown as</span>
            <span class="setting-hint">How collapsed levels are drawn when a term opens.</span>
          </div>
          <div class="summary-level-segments setting-segments" role="radiogroup"
            aria-labelledby="settingGroupsDefaultLabel">
            <label><input type="radio" name="settingGroupsDefault" value="expanded" checked><span>Expanded</span></label>
            <label><input type="radio" name="settingGroupsDefault" value="collapsed"><span>Collapsed</span></label>
          </div>
        </div>
        <div class="setting-row">
          <div class="setting-text">
            <span class="setting-title">Build animation</span>
            <span class="setting-hint">How long Build takes to add the basis functions.</span>
          </div>
          <label id="buildDurationWrap" class="build-duration">
            <input id="buildDuration" type="range" min="4000" max="30000" step="500"
              value="10000" aria-label="Build animation duration">
            <output id="buildDurationValue" for="buildDuration">10 s</output>
          </label>
        </div>
        <h3 class="settings-heading">Diagnostics</h3>
        <div class="setting-row">
          <div class="setting-text">
            <span id="settingShowTimingsLabel" class="setting-title">Request timings</span>
            <span id="settingShowTimingsHint" class="setting-hint">How long the last refit and each panel took.</span>
            <div id="settingsTiming" class="settings-timing" aria-live="polite" hidden></div>
          </div>
          <button id="settingShowTimings" class="setting-switch" type="button" role="switch"
            aria-checked="false" aria-labelledby="settingShowTimingsLabel"
            aria-describedby="settingShowTimingsHint" data-setting="showTimings"></button>
        </div>
      </div>
```

In `src/superglm/editor/app/main.js`:

Add the settings import above the theme import, and import `resolveTheme` (lines 60–61):

```js
import { bindPopovers } from "./views/popover.js";
import { mountThemeControl } from "./views/theme.js";
```
becomes
```js
import { bindPopovers } from "./views/popover.js";
import {
  bindSettingsPane,
  loadSettings,
  onSettingsChange,
  renderSettingsPane,
  saveSettings
} from "./views/settings.js";
import { mountThemeControl, resolveTheme } from "./views/theme.js";
```

Remove the slider nodes (118–119). The settings view owns them now:

```js
const contribPlay = document.getElementById("contribPlay");
const buildDuration = document.getElementById("buildDuration");
const buildDurationValue = document.getElementById("buildDurationValue");
const resetZoom = document.getElementById("resetZoom");
```
becomes
```js
const contribPlay = document.getElementById("contribPlay");
const resetZoom = document.getElementById("resetZoom");
```

Replace `advancedTiming` (168):

```js
const advancedTiming = document.getElementById("advancedTiming");
```
becomes
```js
const settingsTiming = document.getElementById("settingsTiming");
const settingsNodes = Object.freeze({
  root: document.getElementById("settingsPane"),
  buildDuration: document.getElementById("buildDuration"),
  buildDurationValue: document.getElementById("buildDurationValue"),
  timing: settingsTiming
});
```

Timing tracker (193–195):

```js
const evidenceTiming = createEvidenceTimingTracker({
  onComplete: () => renderAdvancedTiming()
});
```
becomes
```js
const evidenceTiming = createEvidenceTimingTracker({
  onComplete: () => renderTimingReadout()
});
```

Theme mount (220–224):

```js
mountThemeControl({
  button: document.getElementById("themeAction"),
  root: document.documentElement,
  media: window.matchMedia("(prefers-color-scheme: dark)")
});
```
becomes
```js
// Until I2: the theme key decides and "Follow the browser" mirrors it. A
// theme chosen with the icon turns the setting off and Auto turns it on;
// turning the setting on removes the key, and turning it off keeps the theme
// now showing. I2's switch takes this over (S7).
const darkMedia = window.matchMedia("(prefers-color-scheme: dark)");
const themeControl = mountThemeControl({
  button: document.getElementById("themeAction"),
  root: document.documentElement,
  media: darkMedia,
  onChange: (choice) => saveSettings({ followBrowserTheme: choice === "auto" })
});
onSettingsChange((settings) => {
  const choice = themeControl.choice();
  if (settings.followBrowserTheme === (choice === "auto")) return;
  themeControl.setChoice(
    settings.followBrowserTheme ? "auto" : resolveTheme(choice, darkMedia.matches)
  );
});
saveSettings({ followBrowserTheme: themeControl.choice() === "auto" });

// Settings keep their choices in this browser (views/settings.js).
function renderSettingsView() {
  renderSettingsPane(settingsNodes, { settings: loadSettings() });
}

bindSettingsPane(settingsNodes, {
  onToggle: (key) => saveSettings({ [key]: !loadSettings()[key] }),
  onGroupsDefault: (groupsDefault) => saveSettings({ groupsDefault }),
  onBuildDuration: (buildDurationMs) => saveSettings({ buildDurationMs })
});
onSettingsChange(renderSettingsView);
renderSettingsView();
```

Timing readout (706–727):

```js
  renderAdvancedTiming();
  if (summaryNote) summaryNote.textContent = payload.note || "";
}

function renderAdvancedTiming() {
  if (!advancedTiming) return;
  const sections = [];
  if (latestTransitionTiming) sections.push(formatTimingDetails(latestTransitionTiming));
  const evidenceDetails = formatEvidenceTimingDetails(evidenceTiming.durations());
  if (evidenceDetails) sections.push(evidenceDetails);
  const details = sections.filter(Boolean).join(" · ");
  advancedTiming.textContent = latestTimingNote && details
```
becomes
```js
  renderTimingReadout();
  if (summaryNote) summaryNote.textContent = payload.note || "";
}

// Settings › Request timings: the last refit's and each panel's durations.
function renderTimingReadout() {
  if (!settingsTiming) return;
  const sections = [];
  if (latestTransitionTiming) sections.push(formatTimingDetails(latestTransitionTiming));
  const evidenceDetails = formatEvidenceTimingDetails(evidenceTiming.durations());
  if (evidenceDetails) sections.push(evidenceDetails);
  const details = sections.filter(Boolean).join(" · ");
  settingsTiming.textContent = latestTimingNote && details
```

`updateHandleCount` (1289–1291):

```js
  contribPlay.disabled = buildFrame !== null;
  updateBuildDurationLabel();
  basisToggle.setAttribute("aria-pressed", String(Boolean(view.showContrib && canShowContrib)));
```
becomes
```js
  contribPlay.disabled = buildFrame !== null;
  basisToggle.setAttribute("aria-pressed", String(Boolean(view.showContrib && canShowContrib)));
```

`applyTermDefaults` (1313–1322):

```js
  if (
    term.group_display &&
    term.group_display.available &&
    !view.groupModeByTerm[selectedTerm()]
  ) {
    patch.groupModeByTerm = {
      ...view.groupModeByTerm,
      [selectedTerm()]: term.group_display.default_mode || "expanded"
    };
  }
```
becomes
```js
  // A grouped term opens as Settings' "Groups shown as" says.
  if (
    term.group_display &&
    term.group_display.available &&
    !view.groupModeByTerm[selectedTerm()]
  ) {
    patch.groupModeByTerm = {
      ...view.groupModeByTerm,
      [selectedTerm()]: loadSettings().groupsDefault
    };
  }
```

`buildDurationMs` and its label (1345–1354):

```js
function buildDurationMs() {
  return Math.max(500, Number(buildDuration.value) || 10000);
}

function updateBuildDurationLabel() {
  const seconds = buildDurationMs() / 1000;
  buildDurationValue.textContent = Number.isInteger(seconds)
    ? `${seconds}s`
    : `${seconds.toFixed(1)}s`;
}
```
becomes
```js
function buildDurationMs() {
  return loadSettings().buildDurationMs;
}
```

The slider listeners (1504–1509):

```js
contribPlay.addEventListener("click", startContributionBuild);

buildDuration.addEventListener("input", updateBuildDurationLabel);
buildDuration.addEventListener("change", updateBuildDurationLabel);

handleCount.addEventListener("input", () => {
```
becomes
```js
contribPlay.addEventListener("click", startContributionBuild);

handleCount.addEventListener("input", () => {
```

In `src/superglm/editor/app/views/help_content.js`, the Theme section (202–208):

```js
  Object.freeze({
    title: "Theme",
    items: Object.freeze([
      "The theme icon in the application bar cycles through Auto, Light and Dark. Auto follows the browser's light or dark setting, which inside a notebook is not always the notebook's own; a chosen theme wins over it.",
      "The choice is kept through a reload of the page.",
    ]),
  }),
```
becomes
```js
  Object.freeze({
    title: "Theme",
    items: Object.freeze([
      "The theme icon in the application bar cycles through Auto, Light and Dark. Auto follows the browser's light or dark setting, which inside a notebook is not always the notebook's own; a chosen theme wins over it.",
      "The choice is kept through a reload of the page.",
      "Settings › Follow the browser's light or dark setting goes back to Auto. Turning it off keeps the theme now showing.",
    ]),
  }),
  Object.freeze({
    title: "Settings",
    items: Object.freeze([
      "Settings, in the inspector, holds preferences kept in this browser: refit after every structural change, keep the reference level when collapsing, follow the browser's light or dark setting, how groups show when a term opens, the Build animation's length, and request timings.",
      "Where the browser blocks storage, as some private windows do, the choices last until the page closes.",
    ]),
  }),
```

In `src/superglm/editor/app/styles/panels.css`, change the header (lines 1–2):

```css
/* panels.css: feature list, inspector, help, advanced controls, and the
   evidence status rows. The side panels share one paper tint and no border. */
```
becomes
```css
/* panels.css: feature list, inspector, help, settings, and the evidence
   status rows. The side panels share one paper tint and no border. */
```

The user-select list (190–196):

```css
.help-pane,
.advanced-timing,
.summary-frame,
```
becomes
```css
.help-pane,
.settings-timing,
.summary-frame,
```

The Advanced rules (224–235):

```css
#advancedPane h2 {
  margin: 2px 0 12px;
  font-size: 14px;
  font-weight: 600;
}

.advanced-timing {
  margin-top: 12px;
  color: var(--muted);
  font-size: 12px;
  line-height: 1.45;
}
```
become
```css
.sidepanel-tab:has(.sidepanel-tab-icon) {
  display: inline-flex;
  align-items: center;
  gap: 5px;
}

.sidepanel-tab-icon {
  width: 15px;
  height: 15px;
  fill: none;
  stroke: currentColor;
  stroke-width: 1.6;
  stroke-linecap: round;
  stroke-linejoin: round;
}

/* Settings: one card per preference, under small headings. */
.settings-pane {
  gap: 8px;
  overflow-y: auto;
}

.settings-heading {
  margin: 6px 2px 0;
  color: var(--muted);
  font-size: 11.5px;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

.setting-row {
  display: flex;
  align-items: flex-start;
  gap: 12px;
  padding: 10px;
  border-radius: 8px;
  background: var(--surface);
}

.setting-text {
  display: grid;
  flex: 1 1 auto;
  min-width: 0;
  gap: 2px;
}

.setting-title {
  font-size: 13px;
  font-weight: 600;
}

.setting-hint {
  color: var(--muted);
  font-size: 12px;
}

.setting-switch,
.setting-switch:hover,
.setting-switch:active {
  position: relative;
  flex: 0 0 auto;
  width: 34px;
  height: 20px;
  padding: 0;
  border: 0;
  border-radius: 10px;
  background: var(--border-strong);
}

.setting-switch[aria-checked="true"],
.setting-switch[aria-checked="true"]:hover,
.setting-switch[aria-checked="true"]:active {
  background: var(--blue);
}

.setting-switch::after {
  content: "";
  position: absolute;
  top: 2px;
  left: 2px;
  width: 16px;
  height: 16px;
  border-radius: 50%;
  background: var(--surface);
  box-shadow: 0 1px 2px color-mix(in srgb, var(--ink) 25%, transparent);
  transition: left 120ms ease;
}

.setting-switch[aria-checked="true"]::after {
  left: 16px;
}

.settings-pane .setting-segments {
  flex: 0 0 auto;
  width: 172px;
}

.settings-pane .build-duration {
  flex: 0 0 auto;
}

.settings-timing {
  margin-top: 2px;
  color: var(--muted);
  font-size: 12px;
  line-height: 1.45;
}
```

In `docs/development/internals/editor-frontend.md`, lines 102–104:

```
The Advanced pane keeps browser request wait, synchronous store/DOM commit, and the two-frame paint
boundary as separate timings. Metrics, summary, and report completion are recorded independently as
panel evidence timings; they are not folded back into the blocking refit duration.
```
become
```
Settings › Request timings shows browser request wait, synchronous store/DOM commit, and the
two-frame paint boundary as separate timings. Metrics, summary, and report completion are recorded
independently as panel evidence timings; they are not folded back into the blocking refit duration.
```

- [ ] **Step 4: Run tests, expect PASS**

```bash
npm run check:frontend
./.venv/bin/python -m pytest tests/test_editor.py tests/test_lss_editor_style.py -q
./.venv/bin/python -m pytest tests/editor/test_editor_settings_browser.py tests/editor/test_editor_workspace_browser.py tests/editor/test_editor_theme_browser.py tests/editor/test_editor_chart_size_browser.py -m browser --run-browser -q
./.venv/bin/python -m ruff check tests/editor tests/test_editor.py && ./.venv/bin/python -m ruff format --check tests/editor tests/test_editor.py
```

The full `test_editor.py` and `test_editor_workspace_browser.py` run here because
both pin `index.html` and `main.js` text. The chart-size browser tests cover the
Build slider leaving the context bar.

- [ ] **Step 5: Commit**

```bash
git add src/superglm/editor/app/views/settings.js src/superglm/editor/app/views/theme.js \
  src/superglm/editor/app/views/inspector.js src/superglm/editor/app/views/help_content.js \
  src/superglm/editor/app/api/contracts.js src/superglm/editor/app/index.html \
  src/superglm/editor/app/main.js src/superglm/editor/app/styles/panels.css \
  docs/development/internals/editor-frontend.md tests/editor_frontend/settings.test.js \
  tests/editor_frontend/theme.test.js tests/editor/test_editor_settings_browser.py \
  tests/editor/test_editor_workspace_browser.py tests/test_editor.py
git commit -m "Editor: Settings tab replaces Advanced, kept under one storage key

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task A5: Structural changes wait for one Refit

**Files:**
- Modify:
  - `src/superglm/editor/app/state/actions.js`: `recoverMutation` (183–248) and
    `executeStructuralMutation` (421–490);
  - `src/superglm/editor/app/state/selectors.js`, adding `selectPendingSteps`;
  - `src/superglm/editor/app/api/contracts.js`: `TimelineEntry` (21–35), the
    request typedefs (83–100), `TermPayload` (102–122), `EditorSnapshot`
    (128–141) and `StructuralMutationDescriptor` (162–168);
  - `src/superglm/editor/app/summary.js`: the typedef imports (5–7) and the
    descriptors (170–210);
  - `src/superglm/editor/app/shapes.js`: `meetingEdge` (52–65);
  - `src/superglm/editor/app/views/app_bar.js`, the bindings, the render and
    `revertAvailable` (121–132);
  - `src/superglm/editor/app/views/help_content.js`: `OPERATION_HELP` structural
    entries (88–118), `STRUCTURE_HELP` (122–123), Shaped ranges item 1 (165) and
    the Undo items (195);
  - `src/superglm/editor/app/main.js`: the imports, the app-bar wiring, the new
    `runStructuralChange`/`refitPending`, the app-bar render state and the
    structural bindings (1584–1615);
  - `src/superglm/editor/app/index.html`: the app actions (61–62), the selection
    label (334) and the seven structural `aria-label`s (336–385);
  - `src/superglm/editor/app/styles/shell.css`, after `.app-actions` (76–81);
  - `src/superglm/editor/app/styles.css`: `.selection-group-label` (293–303);
  - `docs/development/internals/editor-frontend.md`: lines 97 and 229–230.
- Test (modify):
  - `tests/editor_frontend/actions.test.js`, `app_bar.test.js`, `summary.test.js`,
    `shapes.test.js` and `store.test.js`;
  - `tests/test_editor.py`: 6395–6396, 6558–6559, 6619–6623 and 6632–6633;
  - `tests/editor/test_editor_refit_browser.py`: 155–227 and 386–433, plus two
    new tests;
  - `tests/editor/test_editor_structure_browser.py`: the helpers, 109–112, 192–195,
    225–228, 279–281 and 337–373;
  - `tests/editor/test_editor_workspace_browser.py`: 1480.

**Interfaces:**
- Consumes:
  - A4's `POST /stage` and `POST /refit_pending` (S2), which answer with the
    structural envelope `{state, summary, timing}`. The `/stage` body is in
    amendment 5;
  - top-level `snapshot.pending`;
  - per-term `pending.ranges`;
  - F1's `loadSettings()` and `renderSettingsView()`.
- Produces:
  ```js
  // summary.js
  export function stageCollapse(term: string, levels: readonly string[]): {name, path: "/stage", payload: StageRequest};
  export function stageUngroup(term: string, levels: readonly string[]): {...};
  export function stageReference(term: string, level: string): {...};
  export function stageShapeRange(term, lo, hi, degree, join = "tangent"): {...};
  export function refitPendingTransition(count: number): {name, path: "/refit_pending", payload: {}};
  // state/selectors.js
  export function selectPendingSteps(state): readonly PendingStep[];
  // views/app_bar.js
  bindAppBar({..., refitButton, onRefit});
  renderAppBar({..., refitButton, refitCount, pendingCount});
  // actions: executeStructuralMutation({..., blocking?: boolean})
  // main.js: runStructuralChange(descriptor), refitPending()
  ```

- [ ] **Step 1: Write the failing tests**

Append to `tests/editor_frontend/actions.test.js`:

```js
test("a staged change runs without blocking the page and, at an unchanged revision, asks for no evidence", async () => {
  const envelope = transitionEnvelope(2);
  envelope.state.pending = [{
    id: "a1b2c3d",
    operation: "collapse",
    term: "age",
    label: "Collapse 1 + 2",
    params: { levels: ["1", "2"] },
    note: null,
    time: 1
  }];
  const store = createEditorStore(createInitialEditorState(snapshot(2)));
  /** @type {boolean|undefined} */
  let blockingSeen;
  /** @type {number[]} */
  const scheduled = [];
  const actions = createEditorActions({
    store,
    client: {
      postJSON: async (path, payload) => {
        blockingSeen = store.getState().request.mutation.blocking;
        assert.equal(path, "/stage");
        assert.deepEqual(payload, {
          operation: "collapse", term: "age", params: { levels: ["1", "2"] }
        });
        return envelope;
      },
      getState: async () => { throw new Error("success must not recover through /state"); }
    },
    waitForPaint: async () => {},
    scheduleVisibleEvidence: (revision) => { scheduled.push(revision); }
  });

  const result = await actions.executeStructuralMutation({
    name: "collapse levels",
    path: "/stage",
    payload: { operation: "collapse", term: "age", params: { levels: ["1", "2"] } },
    blocking: false
  });

  assert.deepEqual(result, { ok: true, envelope });
  assert.equal(blockingSeen, false);
  assert.strictEqual(store.getState().remote.snapshot, envelope.state);
  assert.equal(store.getState().request.mutation.status, "idle");
  assert.deepEqual(scheduled, []);
});

test("a refused structural request shows Python's fixed sentence, not an uncertain outcome", async () => {
  const refusal = "The refit was refused. Undo the last waiting change and try again.";
  const store = createEditorStore(createInitialEditorState(snapshot(4)));
  const actions = createEditorActions({
    store,
    client: {
      postJSON: async () => { throw Object.assign(new Error(refusal), { status: 400 }); },
      getState: async () => snapshot(4)
    },
    waitForPaint: async () => {}
  });

  const result = await actions.executeStructuralMutation({
    name: "refit 2 waiting changes",
    path: "/refit_pending",
    payload: {}
  });

  assert.equal(result.ok, false);
  assert.equal(store.getState().request.recovery?.message, refusal);
  assert.equal(store.getState().request.recovery?.retry, null);
  assert.equal(store.getState().remote.snapshot?.model_revision, 4);
});
```

In `tests/editor_frontend/app_bar.test.js`, extend the fake (lines 12–19):

```js
class FakeElement {
  constructor(tagName = "div") {
    this.tagName = tagName.toUpperCase();
    this.dataset = {};
    this.disabled = false;
    this.isContentEditable = false;
    this.listeners = new Map();
  }
```
becomes
```js
class FakeElement {
  constructor(tagName = "div") {
    this.tagName = tagName.toUpperCase();
    this.dataset = {};
    this.disabled = false;
    this.hidden = false;
    this.textContent = "";
    this.isContentEditable = false;
    this.listeners = new Map();
    this.attributes = new Map();
    this.classes = new Set();
    this.classList = {
      toggle: (name, force) => (force ? this.classes.add(name) : this.classes.delete(name)),
      contains: (name) => this.classes.has(name),
    };
  }

  setAttribute(name, value) {
    this.attributes.set(name, String(value));
  }

  getAttribute(name) {
    return this.attributes.get(name) ?? null;
  }
```

In the first test's `bindAppBar` call, add the Refit button and its handler:

```js
    revertButton: new FakeButton(),
    refreshButton: new FakeButton(),
    onView: () => {},
```
becomes
```js
    revertButton: new FakeButton(),
    refreshButton: new FakeButton(),
    refitButton: new FakeButton(),
    onView: () => {},
```
and
```js
    onRevert: () => {},
    onRefresh: () => {},
  });
```
becomes
```js
    onRevert: () => {},
    onRefresh: () => {},
    onRefit: () => {},
  });
```

Both `buttons` objects are identical, so replace them with one replace-all:

```js
  const buttons = {
    undoButton: new FakeButton(),
    redoButton: new FakeButton(),
    revertButton: new FakeButton(),
    refreshButton: new FakeButton(),
  };
```
becomes
```js
  const buttons = {
    undoButton: new FakeButton(),
    redoButton: new FakeButton(),
    revertButton: new FakeButton(),
    refreshButton: new FakeButton(),
    refitButton: new FakeButton(),
    refitCount: new FakeElement("span"),
  };
```
In the two render helpers (lines 138–139 and 167), make these changes:

```js
    canRevert: false,
    busy: false,
    ...overrides,
```
becomes
```js
    canRevert: false,
    busy: false,
    pendingCount: 0,
    ...overrides,
```
and
```js
      root, activeView: "editor", ...buttons, undoLabel, redoLabel, canRevert: false, busy: false,
```
becomes
```js
      root, activeView: "editor", ...buttons, undoLabel, redoLabel, canRevert: false, busy: false,
      pendingCount: 0,
```

Append:

```js
function installDocument(t) {
  const saved = ["document", "Element", "HTMLElement", "HTMLButtonElement"].map(
    (name) => [name, globalThis[name]],
  );
  const documentHub = new FakeElement("document");
  globalThis.document = documentHub;
  globalThis.Element = FakeElement;
  globalThis.HTMLElement = FakeElement;
  globalThis.HTMLButtonElement = FakeButton;
  t.after(() => {
    for (const [name, value] of saved) {
      if (value === undefined) delete globalThis[name];
      else globalThis[name] = value;
    }
  });
  return documentHub;
}

test("Refit shows the waiting count, and only a count enables it", () => {
  const root = new FakeElement("nav");
  const buttons = {
    undoButton: new FakeButton(),
    redoButton: new FakeButton(),
    revertButton: new FakeButton(),
    refreshButton: new FakeButton(),
    refitButton: new FakeButton(),
    refitCount: new FakeElement("span"),
  };
  const render = (pendingCount, busy = false) => renderAppBar({
    root, activeView: "editor", ...buttons, undoLabel: null, redoLabel: null,
    canRevert: false, busy, pendingCount,
  });
  const { refitButton, refitCount } = buttons;

  render(0);
  assert.deepEqual(
    [refitButton.disabled, refitCount.hidden, refitButton.classList.contains("has-pending")],
    [true, true, false],
  );
  assert.equal(refitButton.getAttribute("aria-label"), "Refit, nothing waiting");
  render(2);
  assert.deepEqual(
    [refitButton.disabled, refitCount.hidden, refitCount.textContent,
      refitButton.classList.contains("has-pending")],
    [false, false, "2", true],
  );
  assert.equal(refitButton.getAttribute("aria-label"), "Refit, 2 changes waiting");
  assert.equal(
    refitButton.dataset.popoverBody,
    "Apply 2 changes in one fit. Hand edits on terms whose structure did not change are kept.",
  );
  render(1);
  assert.equal(refitButton.getAttribute("aria-label"), "Refit, 1 change waiting");
  render(1, true);
  assert.equal(refitButton.disabled, true);
});

test("R refits what is waiting, except while typing, with a modifier, in a dialog, or with nothing waiting", (t) => {
  const documentHub = installDocument(t);
  const root = new FakeElement("nav");
  const refitButton = new FakeButton();
  let refits = 0;
  const binding = bindAppBar({
    root,
    undoButton: new FakeButton(),
    redoButton: new FakeButton(),
    revertButton: new FakeButton(),
    refreshButton: new FakeButton(),
    refitButton,
    onView: () => {},
    onUndo: () => {},
    onRedo: () => {},
    onRevert: () => {},
    onRefresh: () => {},
    onRefit: () => { refits += 1; },
  });

  const pressed = documentHub.emit("keydown", { key: "r" });
  assert.equal(refits, 1);
  assert.equal(pressed.defaultPrevented, true);
  documentHub.emit("keydown", { key: "R", shiftKey: true });
  assert.equal(refits, 2);

  // Reload stays the browser's; typing and dialogs keep their keys.
  documentHub.emit("keydown", { key: "r", ctrlKey: true });
  documentHub.emit("keydown", { key: "r", metaKey: true });
  documentHub.emit("keydown", { key: "r", target: new FakeElement("input") });
  documentHub.openDialog = new FakeElement("dialog");
  documentHub.emit("keydown", { key: "r" });
  documentHub.openDialog = null;
  refitButton.disabled = true;
  documentHub.emit("keydown", { key: "r" });
  assert.equal(refits, 2);

  refitButton.disabled = false;
  refitButton.emit("click");
  assert.equal(refits, 3);
  binding.destroy();
  documentHub.emit("keydown", { key: "r" });
  refitButton.emit("click");
  assert.equal(refits, 3);
});
```

In the Revert test, change:

```js
  // A structural step or a distribution re-profile puts another model in force.
  assert.equal(revertAvailable(snapshot({ in_force_is_original: false })), true);
```
to
```js
  // A change waiting for Refit is something Revert takes back too; an undone one is not.
  const waiting = {
    kind: "pending", status: "waiting", label: "collapse B10 + B11 in brand", redo: false,
  };
  assert.equal(revertAvailable(snapshot({ timeline: [waiting, marker] })), true);
  assert.equal(revertAvailable(snapshot({ timeline: [marker, { ...waiting, redo: true }] })), false);
  // A structural step or a distribution re-profile puts another model in force.
  assert.equal(revertAvailable(snapshot({ in_force_is_original: false })), true);
```

In `tests/editor_frontend/summary.test.js`, update the destructured import (lines 11–21):

```js
const {
  collapseTransition,
  refreshSummary,
  renderSummary,
  runDistributionProfile,
  revertTransition,
  setReferenceTransition,
  shapeRangeTransition,
  runOffsetRefit,
  ungroupTransition
} = await import(summaryModulePath);
```
becomes
```js
const {
  refitPendingTransition,
  refreshSummary,
  renderSummary,
  runDistributionProfile,
  revertTransition,
  runOffsetRefit,
  stageCollapse,
  stageReference,
  stageShapeRange,
  stageUngroup
} = await import(summaryModulePath);
```
and replace the two descriptor tests (lines 184–224), from
`test("structural transition descriptors are pure route descriptions", () => {`
through the end of `test("transition descriptor payloads are independent caller-owned values", …)`,
with:

```js
test("structural changes are staged through one route, levels by label", () => {
  assert.deepEqual(stageCollapse("region", ["B", "C"]), {
    name: "collapse levels",
    path: "/stage",
    payload: { operation: "collapse", term: "region", params: { levels: ["B", "C"] } }
  });
  assert.deepEqual(stageUngroup("region", ["B"]), {
    name: "ungroup levels",
    path: "/stage",
    payload: { operation: "ungroup", term: "region", params: { levels: ["B"] } }
  });
  assert.deepEqual(stageReference("region", "B+C"), {
    name: "set reference",
    path: "/stage",
    payload: { operation: "set_reference", term: "region", params: { level: "B+C" } }
  });
  // The join is the toggle's choice; Tangent when no choice is given.
  assert.deepEqual(stageShapeRange("age", 30, 45, 1), {
    name: "make a Line range",
    path: "/stage",
    payload: {
      operation: "shape", term: "age", params: { lo: 30, hi: 45, degree: 1, join: "tangent" }
    }
  });
  assert.deepEqual(stageShapeRange("band", "B2", "B4", 0, "kink").payload.params, {
    lo: "B2", hi: "B4", degree: 0, join: "kink"
  });
  assert.deepEqual(refitPendingTransition(1), {
    name: "refit 1 waiting change",
    path: "/refit_pending",
    payload: {}
  });
  assert.equal(refitPendingTransition(3).name, "refit 3 waiting changes");
  assert.deepEqual(revertTransition(), {
    name: "revert to original model",
    path: "/revert_to_original",
    payload: {}
  });
});

test("transition descriptor payloads are independent caller-owned values", () => {
  const levels = ["B", "C"];
  const first = stageCollapse("region", levels);
  first.payload.params.levels.push("D");
  levels.push("E");

  assert.deepEqual(first.payload.params.levels, ["B", "C", "D"]);
  assert.deepEqual(stageCollapse("region", ["B", "C"]).payload.params, { levels: ["B", "C"] });
});
```

In `tests/editor_frontend/shapes.test.js`, add `STRUCTURE_HELP` to the
`help_content.js` import:

```js
import {
  OPERATION_HELP,
  helpForElement
} from "../../src/superglm/editor/app/views/help_content.js";
```
becomes
```js
import {
  OPERATION_HELP,
  STRUCTURE_HELP,
  helpForElement
} from "../../src/superglm/editor/app/views/help_content.js";
```
Change the disabled-reason test (lines 208–221). Replace `"Line and refit"` with
`"Line"` in its three places:

```js
  assert.equal(OPERATION_HELP.shape_line.title, "Line and refit");
```
becomes `assert.equal(OPERATION_HELP.shape_line.title, "Line");`;
`popoverTitle: "Line and refit",` becomes `popoverTitle: "Line",`; and
`assert.deepEqual(helpForElement(disabled), { title: "Line and refit", body: GROUPED_EDGE });`
becomes `assert.deepEqual(helpForElement(disabled), { title: "Line", body: GROUPED_EDGE });`.
Append:

```js
test("a run next to a waiting range meets it too, so ranges staged back to back leave no sliver", () => {
  const term = {
    ...numeric,
    pending: {
      groups: null,
      reference: null,
      ranges: [{ lo: 35, hi: 50, degree: 1, join: "tangent", label: "Line" }]
    }
  };
  assert.deepEqual(shapeRangeForSelection(term, new Set([1, 2, 3])), { lo: 20, hi: 35 });
});

test("every structural icon's help says the change waits for Refit", () => {
  for (const key of [
    "collapse_levels", "ungroup_levels", "set_reference",
    "shape_flat", "shape_line", "shape_quadratic", "shape_cubic"
  ]) {
    assert.match(OPERATION_HELP[key].body, /waits for Refit/, key);
    assert.doesNotMatch(OPERATION_HELP[key].title, /refit/i, key);
  }
  assert.equal(STRUCTURE_HELP.refit_pending.title, "Refit");
  assert.equal(STRUCTURE_HELP.refit_pending.shortcut, "R");
});
```

In `tests/editor_frontend/store.test.js`, add `selectPendingSteps` to the
selectors destructure:

```js
  selectModelRevision,
  selectMutation,
```
becomes
```js
  selectModelRevision,
  selectMutation,
  selectPendingSteps,
```
In the export-list test, change:

```js
    "selectMutation",
    "selectRenderableTerm",
```
to
```js
    "selectMutation",
    "selectPendingSteps",
    "selectRenderableTerm",
```
Append:

```js
test("the pending selector reads the waiting structural changes, and none without them", () => {
  const confirmed = snapshot(7);
  confirmed.pending = [
    { id: "a1b2c3d", operation: "collapse", term: "age", label: "Collapse 1 + 2", params: {}, note: null, time: 1 },
    { id: "b2c3d4e", operation: "shape", term: "age", label: "Line 1 – 2", params: {}, note: null, time: 2 }
  ];
  assert.strictEqual(selectPendingSteps(createInitialEditorState(confirmed)), confirmed.pending);
  assert.deepEqual(selectPendingSteps(createInitialEditorState(snapshot(7))), []);
  assert.deepEqual(selectPendingSteps(createInitialEditorState()), []);
});
```

Update the pins in `tests/test_editor.py`. In
`test_widget_app_shell_contains_drag_editor` (6395–6396), add the new routes:

```python
        assert "/metrics" in js
        assert "/summary" in js
```
becomes
```python
        assert "/metrics" in js
        assert "/summary" in js
        assert "/stage" in js
        assert "/refit_pending" in js
```
At 6558–6559:

```python
    collapse_start = summary_js.index("export function collapseTransition")
    collapse_end = summary_js.index("export function ungroupTransition", collapse_start)
```
becomes
```python
    collapse_start = summary_js.index("export function stageCollapse")
    collapse_end = summary_js.index("export function stageUngroup", collapse_start)
```
At 6619–6623:

```python
    assert "collapseTransition" in main_js[:refit_start]
    assert "ungroupTransition" in main_js[:refit_start]
    assert "restoreTransition" not in main_js
    assert "runStructuralRefit(collapseTransition(selectedTerm()))" in bindings_source
    assert "runStructuralRefit(ungroupTransition(selectedTerm()))" in bindings_source
```
becomes
```python
    assert "stageCollapse" in main_js[:refit_start]
    assert "stageUngroup" in main_js[:refit_start]
    assert "restoreTransition" not in main_js
    assert "runStructuralChange(stageCollapse(selectedTerm()," in bindings_source
    assert "runStructuralChange(stageUngroup(selectedTerm()," in bindings_source
```
At 6632–6633:

```python
    assert 'path: "/collapse_levels"' in collapse_source
    assert 'payload: { term, method: "auto" }' in collapse_source
```
becomes
```python
    assert 'stageTransition("collapse", term, { levels: [...levels] }' in collapse_source
    assert 'path: "/stage"' in summary_js
```

In `tests/editor/test_editor_refit_browser.py`, add the import after the
`numpy`/`pytest` imports:

```python
import numpy as np
import pytest
```
becomes
```python
import numpy as np
import pytest

from superglm.editor.payloads import session_payload
```

Rewrite the click-and-wait part of
`test_structural_refit_commits_atomically_before_held_metrics`, from line 171
(`        requests: list[object] = []`) to the end of the function (227). The test now
stages first and holds the metrics of the Refit:

```python
        # The collapse waits: staging fits nothing, so nothing blocks the page.
        with page.expect_response(
            lambda response: response.request.method == "POST" and _path(response.url) == "/stage"
        ) as stage_info:
            page.get_by_role("button", name="Collapse", exact=True).click()
        assert stage_info.value.status == 200
        assert page.locator("#appBusyOverlay").is_hidden()
        page.wait_for_function("() => !document.querySelector('#refitPendingAction').disabled")

        requests: list[object] = []
        held_metrics: list[object] = []

        def record_request(request) -> None:
            requests.append(request)

        def hold_metrics(route) -> None:
            held_metrics.append(route)
            page.evaluate("count => { window.__heldMetricRouteCount = count; }", len(held_metrics))

        page.evaluate("window.__heldMetricRouteCount = 0")
        page.on("request", record_request)
        page.route("**/metrics", hold_metrics)
        try:
            with page.expect_request(
                lambda request: request.method == "POST" and _path(request.url) == "/metrics"
            ):
                with page.expect_response(
                    lambda response: (
                        response.request.method == "POST"
                        and _path(response.url) == "/refit_pending"
                    )
                ) as refit_info:
                    page.locator("#refitPendingAction").click()
                    page.locator("#appBusyOverlay").wait_for(state="visible")
                    assert page.locator("#editorView").get_attribute("inert") == ""
                    assert page.evaluate("document.activeElement?.id") == "appBusyAnnouncement"

            assert refit_info.value.status == 200
            page.wait_for_function(
                """revision => {
                    const overlay = document.querySelector('#appBusyOverlay');
                    const chart = document.querySelector('#chart');
                    const summary = document.querySelector('#summaryFrame');
                    const metrics = document.querySelector('#metricGrid');
                    return overlay?.hidden
                        && window.__heldMetricRouteCount === 1
                        && chart?.dataset.modelRevision === revision
                        && summary?.dataset.modelRevision === revision
                        && metrics?.dataset.freshness === 'updating';
                }""",
                arg=str(session.model_revision),
            )

            chart_revision = page.locator("#chart").get_attribute("data-model-revision")
            summary_revision = page.locator("#summaryFrame").get_attribute("data-model-revision")
            assert chart_revision == summary_revision == str(session.model_revision)
            assert page.locator("#appBusyOverlay").is_hidden()
            assert page.locator("#metricGrid").get_attribute("data-freshness") == "updating"
            assert len(held_metrics) == 1
            request_paths = [_path(request.url) for request in requests]
            assert request_paths.count("/refit_pending") == 1
            assert request_paths.count("/state") == 0
        finally:
            for route in held_metrics:
                route.abort()
            page.unroute("**/metrics", hold_metrics)
```

Rewrite the body of
`test_a_structural_refit_over_live_edits_runs_at_once_and_undo_brings_them_back`
from line 399 (`        edited = session.terms[term].edited_log_effect.copy()`) to the end (433):

```python
        edited = session.terms[term].edited_log_effect.copy()
        records = list(session.history)
        model = session.model
        stages: list[object] = []
        page.on(
            "request",
            lambda request: _path(request.url) == "/stage" and stages.append(request),
        )
        collapse = page.get_by_role("button", name="Collapse", exact=True)

        # A change still waits for a running action to finish.
        page.evaluate("window.__superglmTest.setAppBusy(true, 'Testing busy guard', 'Waiting')")
        collapse.evaluate("node => node.click()")
        page.evaluate("() => new Promise(resolve => requestAnimationFrame(() => resolve()))")
        assert stages == []
        page.evaluate("window.__superglmTest.setAppBusy(false)")

        with page.expect_response(
            lambda response: response.request.method == "POST" and _path(response.url) == "/stage"
        ) as response_info:
            collapse.click()
        assert response_info.value.status == 200
        # Staged: the model and the edits stand until Refit.
        assert session.model is model
        assert [id(record) for record in session.history] == [id(record) for record in records]

        with page.expect_response(
            lambda response: (
                response.request.method == "POST" and _path(response.url) == "/refit_pending"
            )
        ) as refit_info:
            page.keyboard.press("r")
        assert refit_info.value.status == 200
        page.locator("#appBusyOverlay").wait_for(state="hidden")
        # Nothing is lost, so nothing asked first. The collapse restructured the
        # edited term, so its edits are set aside (D2).
        assert page.locator("dialog[open]").count() == 0 and len(stages) == 1
        assert session.history == [] and session.edited_terms() == []

        with page.expect_response(
            lambda response: response.request.method == "POST" and _path(response.url) == "/op"
        ):
            page.locator("#undoAction").click()
        _wait_for_editor_idle(page)
        np.testing.assert_array_equal(session.terms[term].edited_log_effect, edited)
        assert [id(record) for record in session.history] == [id(record) for record in records]
        # Undo of the Refit brings the collapse back as waiting.
        assert len(session.pending) == 1
```

Append two tests to `tests/editor/test_editor_refit_browser.py`:

```python
def _stage_response(response) -> bool:
    return response.request.method == "POST" and _path(response.url) == "/stage"


def _refit_response(response) -> bool:
    return response.request.method == "POST" and _path(response.url) == "/refit_pending"


def test_two_staged_collapses_wait_and_one_refit_applies_both(open_editor_page):
    with open_editor_page(selected_term="territory") as (page, session):
        _wait_for_editor_idle(page)
        original = session.model
        revision = session.model_revision
        refit = page.locator("#refitPendingAction")
        assert refit.is_disabled()
        assert refit.get_attribute("aria-label") == "Refit, nothing waiting"
        paths: list[str] = []
        page.on(
            "request",
            lambda request: request.method == "POST" and paths.append(_path(request.url)),
        )

        for levels in (["T02", "T03"], ["T06", "T07"]):
            session.select_levels("territory", levels)
            _reload_editor(page, "territory")
            page.locator("#selectionMenu").wait_for(state="visible")
            assert page.locator("#selectionRefitLabel").text_content() == "Structure"
            with page.expect_response(_stage_response) as staged:
                page.get_by_role("button", name="Collapse", exact=True).click()
            assert staged.value.status == 200
            # Staging fits nothing: the page never blocks and the model stands.
            assert page.locator("#appBusyOverlay").is_hidden()
            assert session.model is original and session.model_revision == revision

        page.wait_for_function(
            "() => document.querySelector('#refitPendingCount')?.textContent === '2'"
        )
        assert refit.get_attribute("aria-label") == "Refit, 2 changes waiting"
        assert len(session.pending) == 2

        with page.expect_response(_refit_response) as refit_info:
            page.keyboard.press("r")
        assert refit_info.value.status == 200
        _wait_for_editor_idle(page)
        assert paths.count("/stage") == 2 and paths.count("/refit_pending") == 1
        assert not {"/collapse_levels", "/ungroup_levels"} & set(paths)
        assert session.pending == []
        groups = session_payload(session)["territory"]["level_groups"]
        assert sorted(group["levels"] for group in groups) == [["T02", "T03"], ["T06", "T07"]]
        page.wait_for_function("() => document.querySelector('#refitPendingAction').disabled")


def test_refit_after_every_change_stages_then_refits_at_once(open_editor_page):
    with open_editor_page(selected_term="territory") as (page, session):
        _wait_for_editor_idle(page)
        inspector = page.get_by_role("complementary", name="Model inspector")
        inspector.get_by_role("tab", name="Settings").click()
        switch = inspector.get_by_role("switch", name="Refit after every structural change")
        switch.click()
        assert switch.get_attribute("aria-checked") == "true"

        session.select_levels("territory", ["T04", "T05"])
        _reload_editor(page, "territory")
        page.locator("#selectionMenu").wait_for(state="visible")
        # With the setting on, the menu's structural row says its icons refit.
        assert page.locator("#selectionRefitLabel").text_content() == "Refit"
        paths: list[str] = []
        page.on(
            "request",
            lambda request: request.method == "POST" and paths.append(_path(request.url)),
        )
        with page.expect_response(_refit_response) as refit_info:
            page.get_by_role("button", name="Collapse", exact=True).click()
        assert refit_info.value.status == 200
        _wait_for_editor_idle(page)

        assert [path for path in paths if path in {"/stage", "/refit_pending"}] == [
            "/stage",
            "/refit_pending",
        ]
        assert session.pending == []
        groups = session_payload(session)["territory"]["level_groups"]
        assert [group["levels"] for group in groups] == [["T04", "T05"]]
```

In `tests/editor/test_editor_structure_browser.py`, change the payload import:

```python
from superglm.editor.payloads import session_payload
```
becomes
```python
from superglm.editor.payloads import session_payload, timeline_payload
```
Add these helpers after `_settled_after_refit`:

```python
def _stage_and_refit(page, icon) -> None:
    """Click a structural icon, which stages the change, then Refit it."""
    with page.expect_response(_posted("/stage")) as staged:
        icon.click()
    assert staged.value.status == 200
    page.wait_for_function("() => !document.querySelector('#refitPendingAction').disabled")
    with page.expect_response(_posted("/refit_pending")) as refitted:
        page.locator("#refitPendingAction").click()
    assert refitted.value.status == 200
    _settled_after_refit(page)


def _timeline_rows(session) -> list[list[str]]:
    """The rows the History pane draws for the session's timeline, top to bottom."""
    rows = []
    for entry in timeline_payload(session):
        if entry["kind"] == "marker":
            rows.append(["history-now", "now"])
            continue
        kind = f"history-item {entry['kind']}" + (" redo" if entry["redo"] else "")
        rows.append([kind, entry["label"]])
    return rows
```

Replace the four click-and-wait blocks with `_stage_and_refit`. In
`test_line_icon_pins_a_run_of_points_and_undo_and_redo_step_across_it` (109–112):

```python
        with page.expect_response(_posted("/shape_range")) as response_info:
            line.click()
        assert response_info.value.status == 200
        _settled_after_refit(page)
```
becomes `        _stage_and_refit(page, line)`.

In `test_quadratic_on_bands_spans_whole_bands_and_cubic_says_why_not` (192–195):

```python
        with page.expect_response(_posted("/shape_range")) as response_info:
            page.locator("#shapeQuadratic").click()
        assert response_info.value.status == 200
        _settled_after_refit(page)
```
becomes `        _stage_and_refit(page, page.locator("#shapeQuadratic"))`.

In `test_back_to_back_runs_give_ranges_that_meet` (225–228):

```python
            with page.expect_response(_posted("/shape_range")) as response_info:
                page.locator(icon).click()
            assert response_info.value.status == 200
            _settled_after_refit(page)
```
becomes `            _stage_and_refit(page, page.locator(icon))`.

In `test_set_reference_icon_needs_exactly_one_level` (279–281):

```python
        with page.expect_response(_posted("/set_reference")) as response_info:
            set_reference.click()
        assert response_info.value.status == 200
```
becomes `        _stage_and_refit(page, set_reference)`.

Replace `test_history_lists_the_session_in_order_and_follows_undo_and_redo` (337–373)
with the version below, which takes its rows from the session's timeline. A6
replaces it again.

```python
def test_history_lists_the_session_in_order_and_follows_undo_and_redo(open_editor_page):
    with open_editor_page() as (page, session):
        _box_select_x(page, 3.0, 5.0)
        with page.expect_response(_posted("/op")):
            page.get_by_role("button", name="Increase selection").click()
        line = page.locator("#shapeLine")
        line.wait_for(state="visible")
        _stage_and_refit(page, line)

        page.locator("#historyTab").click()
        rows = "document.querySelectorAll('#historyFrame .history-list > li')"
        listed = _timeline_rows(session)
        page.wait_for_function(f"() => {rows}.length === {len(listed)}")
        assert _history_rows(page) == listed

        with page.expect_response(_posted("/op")):
            page.keyboard.press("Control+z")
        page.wait_for_function("() => document.querySelector('#historyFrame .history-item.redo')")
        assert _history_rows(page) == _timeline_rows(session)

        with page.expect_response(_posted("/op")):
            page.keyboard.press("Control+Shift+z")
        page.wait_for_function("() => !document.querySelector('#historyFrame .history-item.redo')")
        assert _history_rows(page) == listed
```

In `tests/editor/test_editor_workspace_browser.py:1480`:

```python
        forbidden = {"collapse_levels", "ungroup_levels", "refit_offset"}
```
becomes
```python
        forbidden = {"collapse_levels", "ungroup_levels", "stage", "refit_pending", "refit_offset"}
```

- [ ] **Step 2: Run them, expect FAIL**

```bash
node --test tests/editor_frontend/actions.test.js tests/editor_frontend/app_bar.test.js tests/editor_frontend/summary.test.js tests/editor_frontend/shapes.test.js tests/editor_frontend/store.test.js
./.venv/bin/python -m pytest tests/test_editor.py -q -k "structural_refits_show_busy or app_shell_contains_drag_editor"
./.venv/bin/python -m pytest tests/editor/test_editor_refit_browser.py tests/editor/test_editor_structure_browser.py -m browser --run-browser -q
```

The expected failures:
- **Actions.** The stage test sees `blockingSeen === true` and `scheduled === [2]`.
  The refusal test gets "The model change outcome is uncertain. The operation was
  not retried."
- **App bar.** The R test counts 0 refits. The Refit render test finds
  `refitButton.disabled === false`, because `renderAppBar` ignores the button.
  The Revert test finds `false` for a waiting entry.
- **Summary.** It throws `TypeError: stageCollapse is not a function`.
- **Shapes.** The title is still "Line and refit". The meeting edge gives
  `{lo: 20, hi: 30}`. `STRUCTURE_HELP.refit_pending` is undefined.
- **Store.** `selectPendingSteps is not a function`, and the export list differs.
- **Python pins.** `"stageCollapse" in main_js[:refit_start]` is false, and so
  is `"/stage" in js`.
- **Browser.** `get_by_role("button", name="Collapse", exact=True)` and
  `#refitPendingAction` time out, because master names the icon "Collapse and
  refit" and has no Refit button. On `origin/master` 155832e8 the routes `/stage`
  and `/refit_pending` do not exist either.

- [ ] **Step 3: Implement**

**`src/superglm/editor/app/state/actions.js`**

After `isNonnegativeFiniteNumber` (lines 88–91), add:

```js
/**
 * A 4xx answer is a refusal: Python checked the request and changed nothing,
 * and its message is one of the editor's fixed sentences.
 * @param {unknown} value
 */
function isRefusal(value) {
  if (!(value instanceof Error) || !("status" in value)) return false;
  const status = value.status;
  return typeof status === "number" && status >= 400 && status < 500;
}
```

In `recoverMutation`, change the doc and signature (lines 183–194):

```js
  /**
   * Reconciles one state-only recovery response against the current remote revision. Structural
   * recovery advances only to a newer revision. Ordinary recovery also accepts an equal revision
   * because UI-only state, such as the selected term, does not increment the model revision.
   *
   * @param {unknown} error
   * @param {string} operation
   * @param {MutationDescriptor|null} retry
   * @param {(state:EditorState, snapshot:EditorSnapshot)=>EditorState} [commitRecovered]
   * @returns {Promise<{ok:false, error:Error}>}
   */
  async function recoverMutation(error, operation, retry, commitRecovered = commitRemote) {
```
becomes
```js
  /**
   * Reconciles one state-only recovery response against the current remote revision. Structural
   * recovery advances only to a newer revision. Ordinary recovery also accepts an equal revision
   * because UI-only state, such as the selected term, does not increment the model revision.
   * A refusal's message is shown as sent; any other failed structural request leaves its outcome
   * uncertain and says so.
   *
   * @param {unknown} error
   * @param {string} operation
   * @param {MutationDescriptor|null} retry
   * @param {(state:EditorState, snapshot:EditorSnapshot)=>EditorState} [commitRecovered]
   * @param {boolean} [refused]
   * @returns {Promise<{ok:false, error:Error}>}
   */
  async function recoverMutation(
    error, operation, retry, commitRecovered = commitRemote, refused = false
  ) {
```
and (line 241):

```js
            message: retry ? normalizedError.message : STRUCTURAL_OUTCOME_UNCERTAIN,
```
becomes
```js
            message: retry || refused ? normalizedError.message : STRUCTURAL_OUTCOME_UNCERTAIN,
```

Make three changes in `executeStructuralMutation`. The head (lines 425–445):

```js
  async function executeStructuralMutation({
    name,
    path,
    payload,
    onRequestSettled = () => {},
    onPrimaryCommitted = () => {},
    onPaintSettled = () => {}
  }) {
    if (store.getState().request.mutation.status === "running") {
      return skippedMutation("An editor mutation is already running.");
    }

    const requestPayload = snapshotPayload(payload);
    store.update((state) => ({
      ...state,
      request: {
        ...state.request,
        mutation: { status: "running", operation: name, error: null, blocking: true },
        recovery: null
      }
    }));
```
becomes
```js
  async function executeStructuralMutation({
    name,
    path,
    payload,
    blocking = true,
    onRequestSettled = () => {},
    onPrimaryCommitted = () => {},
    onPaintSettled = () => {}
  }) {
    if (store.getState().request.mutation.status === "running") {
      return skippedMutation("An editor mutation is already running.");
    }

    const previousRevision = store.getState().remote.snapshot?.model_revision ?? -1;
    const requestPayload = snapshotPayload(payload);
    store.update((state) => ({
      ...state,
      request: {
        ...state.request,
        mutation: { status: "running", operation: name, error: null, blocking },
        recovery: null
      }
    }));
```
The request failure (lines 453–456):

```js
    } catch (value) {
      await notifyTimingHook(onRequestSettled);
      return recoverMutation(value, name, null);
    }
```
becomes
```js
    } catch (value) {
      await notifyTimingHook(onRequestSettled);
      return recoverMutation(value, name, null, commitRemote, isRefusal(value));
    }
```
The evidence at the end (lines 480–489):

```js
    finishStructuralMutation(null);
    try {
      void Promise.resolve(scheduleVisibleEvidence(envelope.state.model_revision, {
        immediate: true,
        summaryCommitted: true
      })).catch(() => {});
    } catch {
      // Evidence refresh cannot change an authoritative structural success.
    }
    return { ok: true, envelope };
```
becomes
```js
    finishStructuralMutation(null);
    // A staged change leaves the model, so its evidence, as it was.
    if (envelope.state.model_revision !== previousRevision) {
      try {
        void Promise.resolve(scheduleVisibleEvidence(envelope.state.model_revision, {
          immediate: true,
          summaryCommitted: true
        })).catch(() => {});
      } catch {
        // Evidence refresh cannot change an authoritative structural success.
      }
    }
    return { ok: true, envelope };
```

**`src/superglm/editor/app/state/selectors.js`.** After `selectRenderableTerm`, add:

```js
/** @type {readonly import('../api/contracts.js').PendingStep[]} */
const NO_PENDING = Object.freeze([]);

/**
 * The structural changes waiting for Refit, oldest first.
 * @param {EditorState} state
 * @returns {readonly import('../api/contracts.js').PendingStep[]}
 */
export function selectPendingSteps(state) {
  return selectSnapshot(state)?.pending ?? NO_PENDING;
}
```

**`src/superglm/editor/app/api/contracts.js`.** `TimelineEntry` (lines 21–35)
gets S2's fields now, because `revertAvailable` reads `status`:

```js
/**
 * One action on the session's timeline, oldest first. The "marker" entry is
 * the current position: Undo takes the entry before it, and the `redo`
 * entries after it are what Redo would put back, in that order.
 * @typedef {Object} TimelineEntry
 * @property {"edit"|"structural"|"marker"} kind
 * @property {string} [operation]
 * @property {string|null} [term]
 * @property {string} [label]
 * @property {number} [n_points]
 * @property {Record<string, unknown>} [params]
 * @property {string} [hash]
 * @property {boolean} [redo]
 */
```
becomes
```js
/**
 * One action on the session's timeline, oldest first. The "marker" entry is
 * the current position: Undo takes the entry before it, and the `redo`
 * entries after it are what Redo would put back, in that order. A structural
 * change is a "pending" entry, with status "waiting" until a Refit applies it
 * and "applied" after. Read the status, not the kind.
 * @typedef {Object} TimelineEntry
 * @property {"edit"|"structural"|"pending"|"marker"} kind
 * @property {string} [operation]
 * @property {string|null} [term]
 * @property {string} [label]
 * @property {number} [n_points]
 * @property {Record<string, unknown>} [params]
 * @property {string} [hash]
 * @property {boolean} [redo]
 * @property {string} [id] the step's short id: 7 hex digits
 * @property {number} [time] when the step was made, in Unix seconds
 * @property {string|null} [note] the analyst's note, saved with the exported model
 * @property {"applied"|"waiting"|"edit"} [status]
 */
```
Replace the two request typedefs (lines 83–100):

```js
/**
 * The /shape_range request: the selection's edges as shapeRangeForSelection
 * names them, the degree of the icon chosen, and the join toggle's choice.
 * @typedef {Object} ShapeRangeRequest
 * @property {string} term
 * @property {number|string} lo
 * @property {number|string} hi
 * @property {number} degree
 * @property {"tangent"|"kink"} join
 * @property {string} method
 */
/**
 * The /set_reference request: a displayed level, which may be a group label.
 * @typedef {Object} SetReferenceRequest
 * @property {string} term
 * @property {string} level
 * @property {string} method
 */
```
with
```js
/** @typedef {"collapse"|"ungroup"|"set_reference"|"shape"} StagedOperation */
/**
 * The /stage request: one structural change, with its parameters by label as
 * the session stores them. Collapse and ungroup take ``levels``. Set
 * reference takes a displayed ``level``, which may be a group label. A shape
 * takes ``lo``, ``hi``, ``degree`` and ``join``. main.js adds the Settings
 * that shape the change: ``keep_reference``, and ``level_display`` for the
 * summary.
 * @typedef {Object} StageRequest
 * @property {StagedOperation} operation
 * @property {string} term
 * @property {Record<string, unknown>} params
 * @property {boolean} [keep_reference]
 * @property {string} [level_display]
 */
/**
 * A structural change waiting for Refit. The snapshot lists them oldest first.
 * @typedef {Object} PendingStep
 * @property {string} id
 * @property {StagedOperation} operation
 * @property {string} term
 * @property {string} label
 * @property {Record<string, unknown>} params
 * @property {string|null} note
 * @property {number} time Unix seconds
 */
/**
 * What a term's waiting changes make of it at the next Refit: its groups by
 * label with their member levels, its shaped ranges, and its reference level.
 * @typedef {Object} TermPending
 * @property {Record<string, string[]>|null} groups
 * @property {ShapedRange[]} ranges
 * @property {string|null} reference
 */
```
In `TermPayload` (line 121):

```js
 * @property {TermShape} shape
 */
```
becomes
```js
 * @property {TermShape} shape
 * @property {TermPending|null} [pending]
 */
```
This string occurs only once, in `TermPayload`, so the edit is unambiguous. In
`EditorSnapshot` (line 140):

```js
 * @property {boolean} in_force_is_original
 */
```
becomes
```js
 * @property {boolean} in_force_is_original
 * @property {PendingStep[]} [pending]
 */
```
In `StructuralMutationDescriptor` (lines 162–168):

```js
/**
 * @typedef {MutationDescriptor & {
 *   onRequestSettled?:()=>void|Promise<void>,
 *   onPrimaryCommitted?:()=>void|Promise<void>,
 *   onPaintSettled?:()=>void|Promise<void>
 * }} StructuralMutationDescriptor
 */
```
becomes
```js
/**
 * ``blocking`` is false for a change that fits nothing, such as a stage: no
 * busy overlay, and nothing goes inert.
 * @typedef {MutationDescriptor & {
 *   blocking?:boolean,
 *   onRequestSettled?:()=>void|Promise<void>,
 *   onPrimaryCommitted?:()=>void|Promise<void>,
 *   onPaintSettled?:()=>void|Promise<void>
 * }} StructuralMutationDescriptor
 */
```

**`src/superglm/editor/app/summary.js`.** Change the typedef imports (lines 5–7):

```js
/** @typedef {import('./api/contracts.js').EmptyStructuralRequest} EmptyStructuralRequest */
/** @typedef {import('./api/contracts.js').SetReferenceRequest} SetReferenceRequest */
/** @typedef {import('./api/contracts.js').ShapeRangeRequest} ShapeRangeRequest */
```
becomes
```js
/** @typedef {import('./api/contracts.js').EmptyStructuralRequest} EmptyStructuralRequest */
/** @typedef {import('./api/contracts.js').StageRequest} StageRequest */
```
Replace everything from `export function collapseTransition(term) {` (line 170)
through the closing brace of `shapeRangeTransition` (line 210) with the code below.
`revertTransition`, below it, stays.

```js
/**
 * One structural change, staged: Python builds it and keeps it waiting,
 * drawn on the chart, until Refit applies every waiting change in one fit.
 * ``name`` is what the busy overlay and an alert call it.
 * @param {StageRequest['operation']} operation @param {string} term
 * @param {Record<string, unknown>} params @param {string} name
 * @returns {{name:string, path:string, payload:StageRequest}}
 */
function stageTransition(operation, term, params, name) {
  return { name, path: "/stage", payload: { operation, term, params } };
}

/** @param {string} term @param {readonly string[]} levels the selected levels, by label */
export function stageCollapse(term, levels) {
  return stageTransition("collapse", term, { levels: [...levels] }, "collapse levels");
}

/** @param {string} term @param {readonly string[]} levels the selected levels, by label */
export function stageUngroup(term, levels) {
  return stageTransition("ungroup", term, { levels: [...levels] }, "ungroup levels");
}

/** @param {string} term @param {string} level a displayed level, which may be a group label */
export function stageReference(term, level) {
  return stageTransition("set_reference", term, { level }, "set reference");
}

/**
 * Named for its shape. ``join`` is how the range meets the free curve:
 * "tangent" (the default) or "kink" (Corner).
 * @param {string} term @param {number|string} lo @param {number|string} hi @param {number} degree
 * @param {"tangent"|"kink"} [join]
 */
export function stageShapeRange(term, lo, hi, degree, join = "tangent") {
  return stageTransition(
    "shape", term, { lo, hi, degree, join }, `make a ${SHAPE_NAMES[degree]} range`
  );
}

/**
 * Refit: every waiting change in one fit, and one step on the timeline.
 * @param {number} count how many changes wait, for the busy overlay
 * @returns {{name:string, path:string, payload:EmptyStructuralRequest}}
 */
export function refitPendingTransition(count) {
  return {
    name: `refit ${count} waiting ${count === 1 ? "change" : "changes"}`,
    path: "/refit_pending",
    payload: {}
  };
}
```

**`src/superglm/editor/app/shapes.js`.** Change `meetingEdge` and its doc
(lines 52–65):

```js
/**
 * A numeric run's edge: its end point, or a shaped range's facing edge when
 * no drawn point lies between the two, so back-to-back selections meet
 * instead of leaving the free sliver that snapping each outward would open.
 * Past the first or last point ``next`` is undefined and matches nothing.
 * @param {TermPayload} term @param {number} index @param {-1|1} direction
 */
function meetingEdge(term, index, direction) {
  const x = term.x[index];
  const next = term.x[index + direction];
  const facing = term.shape.ranges.map((range) => Number(direction < 0 ? range.hi : range.lo));
  return facing.find((edge) => (edge - x) * direction > 0 && (next - edge) * direction >= 0) ?? x;
}
```
becomes
```js
/**
 * A numeric run's edge: its end point, or a shaped range's facing edge when
 * no drawn point lies between the two, so back-to-back selections meet
 * instead of leaving the free sliver that snapping each outward would open.
 * A range waiting for Refit is met as one in force is, so ranges staged back
 * to back meet too. Past the first or last point ``next`` is undefined and
 * matches nothing.
 * @param {TermPayload} term @param {number} index @param {-1|1} direction
 */
function meetingEdge(term, index, direction) {
  const x = term.x[index];
  const next = term.x[index + direction];
  const ranges = [...term.shape.ranges, ...(term.pending?.ranges ?? [])];
  const facing = ranges.map((range) => Number(direction < 0 ? range.hi : range.lo));
  return facing.find((edge) => (edge - x) * direction > 0 && (next - edge) * direction >= 0) ?? x;
}
```

**`src/superglm/editor/app/views/app_bar.js`.** Below the `EditorSnapshot` typedef
(line 3), add:

```js

const NOTHING_WAITING =
  "Nothing is waiting. Collapse, Ungroup, Set reference and the shapes wait here for one refit.";
```
`bindAppBar`'s doc and signature (lines 9–20):

```js
 * @param {HTMLButtonElement} options.refreshButton
 * @param {(view:string)=>unknown} options.onView
 * @param {()=>unknown} options.onUndo
 * @param {()=>unknown} options.onRedo
 * @param {()=>unknown} options.onRevert
 * @param {()=>unknown} options.onRefresh
 */
export function bindAppBar({
  root, undoButton, redoButton, revertButton, refreshButton,
  onView, onUndo, onRedo, onRevert, onRefresh,
}) {
```
becomes
```js
 * @param {HTMLButtonElement} options.refreshButton
 * @param {HTMLButtonElement} options.refitButton
 * @param {(view:string)=>unknown} options.onView
 * @param {()=>unknown} options.onUndo
 * @param {()=>unknown} options.onRedo
 * @param {()=>unknown} options.onRevert
 * @param {()=>unknown} options.onRefresh
 * @param {()=>unknown} options.onRefit
 */
export function bindAppBar({
  root, undoButton, redoButton, revertButton, refreshButton, refitButton,
  onView, onUndo, onRedo, onRevert, onRefresh, onRefit,
}) {
```
The document shortcut (lines 51–59):

```js
  /** @param {KeyboardEvent} event */
  function onDocumentKeyDown(event) {
    if (isEditableTarget(event.target) || event.altKey || document.querySelector("dialog[open]")) {
      return;
    }
    const primary = event.ctrlKey || event.metaKey;
    if (!primary) return;
    const key = event.key.toLowerCase();
    if (key === "z" && !event.shiftKey) {
```
becomes
```js
  /** @param {KeyboardEvent} event */
  function onDocumentKeyDown(event) {
    if (isEditableTarget(event.target) || event.altKey || document.querySelector("dialog[open]")) {
      return;
    }
    const key = event.key.toLowerCase();
    if (!(event.ctrlKey || event.metaKey)) {
      // R refits what waits; with nothing waiting the button is disabled.
      if (key === "r" && !event.defaultPrevented && !refitButton.disabled) {
        event.preventDefault();
        onRefit();
      }
      return;
    }
    if (key === "z" && !event.shiftKey) {
```
The listeners (lines 73–74 and 83–84):

```js
  refreshButton.addEventListener("click", onRefresh);
  document.addEventListener("keydown", onDocumentKeyDown);
```
becomes
```js
  refreshButton.addEventListener("click", onRefresh);
  refitButton.addEventListener("click", onRefit);
  document.addEventListener("keydown", onDocumentKeyDown);
```
and
```js
      refreshButton.removeEventListener("click", onRefresh);
      document.removeEventListener("keydown", onDocumentKeyDown);
```
becomes
```js
      refreshButton.removeEventListener("click", onRefresh);
      refitButton.removeEventListener("click", onRefit);
      document.removeEventListener("keydown", onDocumentKeyDown);
```
`renderAppBar`'s doc and signature (lines 95–105):

```js
 * @param {HTMLButtonElement} options.refreshButton
 * @param {string|null} options.undoLabel what Undo would take back; null disables it
 * @param {string|null} options.redoLabel what Redo would put back; null disables it
 * @param {boolean} options.canRevert
 * @param {boolean} options.busy
 */
export function renderAppBar({
  root, activeView, undoButton, redoButton, revertButton, refreshButton,
  undoLabel, redoLabel, canRevert, busy,
}) {
```
becomes
```js
 * @param {HTMLButtonElement} options.refreshButton
 * @param {HTMLButtonElement} options.refitButton
 * @param {HTMLElement} options.refitCount the count badge inside Refit
 * @param {string|null} options.undoLabel what Undo would take back; null disables it
 * @param {string|null} options.redoLabel what Redo would put back; null disables it
 * @param {boolean} options.canRevert
 * @param {boolean} options.busy
 * @param {number} options.pendingCount how many structural changes wait for Refit
 */
export function renderAppBar({
  root, activeView, undoButton, redoButton, revertButton, refreshButton, refitButton, refitCount,
  undoLabel, redoLabel, canRevert, busy, pendingCount,
}) {
```
and its end (lines 117–119):

```js
  revertButton.disabled = !canRevert;
  refreshButton.disabled = busy;
}
```
becomes
```js
  revertButton.disabled = !canRevert;
  refreshButton.disabled = busy;
  renderRefit(refitButton, refitCount, pendingCount, busy);
}

/**
 * Refit is quiet while nothing waits and the one filled action while changes
 * do. Its badge and its name carry the count.
 * @param {HTMLButtonElement} button @param {HTMLElement} count
 * @param {number} pending @param {boolean} busy
 */
function renderRefit(button, count, pending, busy) {
  const waiting = pending > 0;
  const changes = `${pending} ${pending === 1 ? "change" : "changes"}`;
  button.disabled = busy || !waiting;
  button.classList.toggle("has-pending", waiting);
  button.setAttribute("aria-label", waiting ? `Refit, ${changes} waiting` : "Refit, nothing waiting");
  button.dataset.popoverBody = waiting
    ? `Apply ${changes} in one fit. Hand edits on terms whose structure did not change are kept.`
    : NOTHING_WAITING;
  count.hidden = !waiting;
  count.textContent = String(pending);
}
```
`revertAvailable` (lines 121–132):

```js
/**
 * Whether anything differs from the opened model: a live manual edit, or an
 * in-force model a structural step or a distribution re-profile put there.
 * The live edits are the run just before the timeline's marker, so one exists
 * exactly when the entry before the marker is an edit.
 * @param {EditorSnapshot} snapshot
 */
export function revertAvailable(snapshot) {
  const { timeline } = snapshot;
  const marker = timeline.findIndex((entry) => entry.kind === "marker");
  return timeline[marker - 1]?.kind === "edit" || !snapshot.in_force_is_original;
}
```
becomes
```js
/**
 * Whether anything differs from the opened model: a live manual edit, a
 * change waiting for Refit, or an in-force model a structural step or a
 * distribution re-profile put there. The live edits and waiting changes are
 * the run just before the timeline's marker, so one exists exactly when the
 * entry before the marker is an edit or a waiting change.
 * @param {EditorSnapshot} snapshot
 */
export function revertAvailable(snapshot) {
  const { timeline } = snapshot;
  const marker = timeline.findIndex((entry) => entry.kind === "marker");
  const last = timeline[marker - 1];
  return last?.kind === "edit" || last?.status === "waiting" || !snapshot.in_force_is_original;
}
```

**`src/superglm/editor/app/views/help_content.js`.** Above `export const OPERATION_HELP`
(line 40), add:

```js
// A structural change waits for Refit unless Settings says to refit after
// every one.
const WAITS = "It waits for Refit in the top bar, or refits at once when Settings says so.";

```
Replace the structural entries (lines 88–118), from `  collapse_levels: Object.freeze({`
through the `shape_cubic` entry, with:

```js
  collapse_levels: Object.freeze({
    title: "Collapse",
    body: `Combine the selected levels into one group. ${WAITS}`,
  }),
  ungroup_levels: Object.freeze({
    title: "Ungroup",
    body: `Separate the selected grouped levels. ${WAITS}`,
  }),
  set_reference: Object.freeze({
    title: "Set reference",
    body:
      `Pin the selected level as the reference (relativity 1.00). ${WAITS} Predictions stay the same unless a selection penalty is on. Unseen levels rated at the reference move with it.`,
  }),
  shape_flat: Object.freeze({
    title: "Flat",
    body: `Make the selected range flat; the rest of the curve stays smooth. ${WAITS} Undo takes it back.`,
  }),
  shape_line: Object.freeze({
    title: "Line",
    body:
      `Make the selected range a straight line; the rest of the curve stays smooth. ${WAITS} Undo takes it back.`,
  }),
  shape_quadratic: Object.freeze({
    title: "Quadratic",
    body:
      `Make the selected range a quadratic; the rest of the curve stays smooth. ${WAITS} Undo takes it back.`,
  }),
  shape_cubic: Object.freeze({
    title: "Cubic",
    body: `Make the selected range a cubic; the rest of the curve stays smooth. ${WAITS} Undo takes it back.`,
  }),
```
`STRUCTURE_HELP` (lines 122–123):

```js
export const STRUCTURE_HELP = Object.freeze({
  revert_to_original: Object.freeze({
```
becomes
```js
export const STRUCTURE_HELP = Object.freeze({
  refit_pending: Object.freeze({
    title: "Refit",
    body: "Apply every waiting change in one fit. Hand edits on terms whose structure did not change are kept. Undo brings the changes back as waiting.",
    shortcut: "R",
  }),
  revert_to_original: Object.freeze({
```
Shaped ranges, item 1 (line 165):

```js
      "Select a run of points or bands on a spline term, then choose Flat, Line, Quadratic or Cubic. That range is pinned to the shape; the rest of the term stays the fitted smooth.",
```
becomes
```js
      "Select a run of points or bands on a spline term, then choose Flat, Line, Quadratic or Cubic. That range is pinned to the shape at the next Refit; the rest of the term stays the fitted smooth.",
```
Undo, Redo and Revert (line 195):

```js
      "Undoing a step brings back the model, curves, edits and selection from before it, without refitting.",
```
becomes
```js
      "Undoing a step brings back the model, curves, edits and selection from before it, without refitting.",
      "Undo takes back a waiting change without refitting. Undo after a Refit brings its changes back as waiting.",
```

**`src/superglm/editor/app/index.html`.** In the app actions (lines 61–62):

```html
    <div class="app-actions" aria-label="Edit actions">
      <button id="undoAction" class="icon-button" type="button" aria-label="Undo edit"
```
becomes
```html
    <div class="app-actions" aria-label="Edit actions">
      <button id="refitPendingAction" class="refit-action" type="button"
        aria-label="Refit, nothing waiting" aria-keyshortcuts="R" data-popover-title="Refit"
        data-popover-body="Nothing is waiting. Collapse, Ungroup, Set reference and the shapes wait here for one refit."
        disabled>
        <svg class="toolbar-icon" viewBox="0 0 24 24" aria-hidden="true">
          <path d="M20 11a8 8 0 1 0-2.3 5.7"></path>
          <path d="M20 4v7h-7"></path>
          <circle cx="12" cy="12" r="1.6" fill="currentColor"></circle>
        </svg>
        <span class="refit-action-label">Refit</span>
        <span id="refitPendingCount" class="refit-action-count" aria-hidden="true" hidden>0</span>
      </button>
      <span class="app-actions-gap" aria-hidden="true"></span>
      <button id="undoAction" class="icon-button" type="button" aria-label="Undo edit"
```
Make these replacements in the selection menu:
- `<span id="selectionRefitLabel" class="selection-group-label">Refit</span>` (334)
  becomes `<span id="selectionRefitLabel" class="selection-group-label">Structure</span>`.
- `aria-label="Collapse and refit"` (336) becomes `aria-label="Collapse"`.
- `aria-label="Ungroup and refit"` (344) becomes `aria-label="Ungroup"`.
- `aria-label="Set reference and refit"` (352) becomes `aria-label="Set reference"`.
- `aria-label="Flat and refit"` (360) becomes `aria-label="Flat"`.
- `aria-label="Line and refit"` (367) becomes `aria-label="Line"`.
- `aria-label="Quadratic and refit"` (374) becomes `aria-label="Quadratic"`.
- `aria-label="Cubic and refit"` (381) becomes `aria-label="Cubic"`.

**`src/superglm/editor/app/styles/shell.css`.** After the `.app-tabs, .app-actions`
rule (lines 76–81), add:

```css

/* Refit: quiet while nothing waits, the one filled action while changes do. */
.app-actions .refit-action {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  height: 30px;
  padding: 0 10px 0 9px;
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  background: var(--surface);
  color: var(--muted);
  font-weight: 600;
}

.app-actions .refit-action.has-pending,
.app-actions .refit-action.has-pending:hover {
  border-color: var(--blue);
  background: var(--blue);
  color: var(--surface);
}

.refit-action-count {
  min-width: 18px;
  height: 18px;
  padding: 0 5px;
  border-radius: 9px;
  background: color-mix(in srgb, var(--surface) 22%, transparent);
  font-family: var(--font-mono);
  font-size: 11.5px;
  line-height: 18px;
  text-align: center;
}

@media (max-width: 640px) {
  .refit-action-label {
    display: none;
  }
}
```

**`src/superglm/editor/app/styles.css`.** "Structure" needs a wider label column
(line 294):

```css
.selection-group-label {
  width: 34px;
```
becomes
```css
.selection-group-label {
  width: 54px;
```

**`src/superglm/editor/app/main.js`.**

1. Summary imports (lines 30–40):
   ```js
   import {
     collapseTransition,
     renderSummary,
     runDistributionProfile,
     showDistributionProfileDialog,
     runOffsetRefit,
     revertTransition,
     setReferenceTransition,
     shapeRangeTransition,
     ungroupTransition
   } from "./summary.js";
   ```
   becomes
   ```js
   import {
     refitPendingTransition,
     renderSummary,
     runDistributionProfile,
     showDistributionProfileDialog,
     runOffsetRefit,
     revertTransition,
     stageCollapse,
     stageReference,
     stageShapeRange,
     stageUngroup
   } from "./summary.js";
   ```
2. Selector import (lines 14–15):
   ```js
     selectModelRevision,
     selectRenderableTerm,
   ```
   becomes
   ```js
     selectModelRevision,
     selectPendingSteps,
     selectRenderableTerm,
   ```
3. The Refit nodes (line 68):
   ```js
   const refreshAction = document.getElementById("refreshAction");
   ```
   becomes
   ```js
   const refreshAction = document.getElementById("refreshAction");
   const refitPendingAction = document.getElementById("refitPendingAction");
   const refitPendingCount = document.getElementById("refitPendingCount");
   ```
4. The app-bar wiring (lines 213–219):
   ```js
     refreshButton: refreshAction,
     onView: showView,
     onUndo: undo,
     onRedo: redo,
     onRevert: () => runStructuralRefit(revertTransition()),
     onRefresh: refreshFromPython
   });
   ```
   becomes
   ```js
     refreshButton: refreshAction,
     refitButton: refitPendingAction,
     onView: showView,
     onUndo: undo,
     onRedo: redo,
     onRevert: () => runStructuralRefit(revertTransition()),
     onRefresh: refreshFromPython,
     onRefit: refitPending
   });
   ```
5. The selection menu names its structural row by the setting. In F1's
   `renderSettingsView`:
   ```js
   function renderSettingsView() {
     renderSettingsPane(settingsNodes, { settings: loadSettings() });
   }
   ```
   becomes
   ```js
   function renderSettingsView() {
     renderSettingsPane(settingsNodes, { settings: loadSettings() });
     // The selection menu names its structural row by what its icons do.
     selectionRefitLabel.textContent = loadSettings().refitEveryChange ? "Refit" : "Structure";
   }
   ```
6. Staging and Refit, before the structural-refit comment (lines 608–610):
   ```js
   // A structural step loses nothing: Undo puts back the state before it, edits
   // included, so it runs without asking.
   async function runStructuralRefit(descriptor) {
   ```
   becomes
   ```js
   // A structural change waits: Python builds it and keeps it, drawn on the
   // chart, until Refit applies every waiting change in one fit. Nothing is
   // fitted, so nothing blocks the page. With "Refit after every structural
   // change" on in Settings, Refit follows at once.
   async function runStructuralChange(descriptor) {
     if (appBusyActive || store.getState().request.mutation.status !== "idle") return null;
     stopContributionBuild();
     const result = await actions.executeStructuralMutation({
       ...descriptor,
       blocking: false,
       payload: {
         ...descriptor.payload,
         keep_reference: loadSettings().keepReference,
         level_display: selectSummaryLevelDisplay(store.getState())
       }
     });
     if (!result.ok) return null;
     if (!loadSettings().refitEveryChange) return result.envelope;
     return refitPending();
   }

   // The Refit button, its R shortcut and Refit-after-every-change come here.
   async function refitPending() {
     const count = selectPendingSteps(store.getState()).length;
     if (count === 0) return null;
     return runStructuralRefit(refitPendingTransition(count));
   }

   // A structural step loses nothing: Undo puts back the state before it, edits
   // included, so it runs without asking.
   async function runStructuralRefit(descriptor) {
   ```
7. The app-bar render state (lines 913–940):
   ```js
       canRevert: Boolean(snapshot && revertAvailable(snapshot)),
       busy: state.request.mutation.status === "running"
     };
   }
   ```
   becomes
   ```js
       canRevert: Boolean(snapshot && revertAvailable(snapshot)),
       busy: state.request.mutation.status === "running",
       pendingCount: selectPendingSteps(state).length
     };
   }
   ```
   then
   ```js
       next.canRevert === previous.canRevert &&
       next.busy === previous.busy;
   }
   ```
   becomes
   ```js
       next.canRevert === previous.canRevert &&
       next.busy === previous.busy &&
       next.pendingCount === previous.pendingCount;
   }
   ```
   then
   ```js
       canRevert: state.canRevert,
       busy: state.busy
     });
   }
   ```
   becomes
   ```js
       canRevert: state.canRevert,
       busy: state.busy,
       refitButton: refitPendingAction,
       refitCount: refitPendingCount,
       pendingCount: state.pendingCount
     });
   }
   ```
8. Levels by label, before `selectedLevelLabel` (line 1251):
   ```js
   // One displayed level: a single source level, or one whole collapsed group.
   function selectedLevelLabel(term, selection) {
   ```
   becomes
   ```js
   // The selected source levels by label, in axis order: what a staged change names.
   function selectedLevels(term, selection) {
     const levels = Array.isArray(term.levels) ? term.levels : [];
     return [...selection]
       .sort((left, right) => left - right)
       .filter((index) => index >= 0 && index < levels.length)
       .map((index) => String(levels[index]));
   }

   // One displayed level: a single source level, or one whole collapsed group.
   function selectedLevelLabel(term, selection) {
   ```
9. The structural bindings (lines 1584–1615):
   ```js
   if (collapseLevels) {
     collapseLevels.addEventListener("click", async () => {
       await runStructuralRefit(collapseTransition(selectedTerm()));
     });
   }
   if (ungroupLevels) {
     ungroupLevels.addEventListener("click", async () => {
       await runStructuralRefit(ungroupTransition(selectedTerm()));
     });
   }
   if (setReference) {
     setReference.addEventListener("click", async () => {
       const term = currentTerm();
       const label = term ? selectedLevelLabel(term, currentSelection()) : null;
       if (label === null) return;
       await runStructuralRefit(setReferenceTransition(selectedTerm(), label));
     });
   }
   for (const button of shapeButtons) {
     button.addEventListener("click", async () => {
       const term = currentTerm();
       const range = term && shapeRangeForSelection(term, currentSelection());
       if (!range || button.getAttribute("aria-disabled") === "true") return;
       const degree = Number(button.dataset.shapeDegree);
       await runStructuralRefit(
         shapeRangeTransition(
           selectedTerm(), range.lo, range.hi, degree,
           effectiveShapeJoin(shapeJoinChoice, term.shape.joins)
         )
       );
     });
   }
   ```
   becomes
   ```js
   if (collapseLevels) {
     collapseLevels.addEventListener("click", async () => {
       const term = currentTerm();
       if (!term) return;
       await runStructuralChange(stageCollapse(selectedTerm(), selectedLevels(term, currentSelection())));
     });
   }
   if (ungroupLevels) {
     ungroupLevels.addEventListener("click", async () => {
       const term = currentTerm();
       if (!term) return;
       await runStructuralChange(stageUngroup(selectedTerm(), selectedLevels(term, currentSelection())));
     });
   }
   if (setReference) {
     setReference.addEventListener("click", async () => {
       const term = currentTerm();
       const label = term ? selectedLevelLabel(term, currentSelection()) : null;
       if (label === null) return;
       await runStructuralChange(stageReference(selectedTerm(), label));
     });
   }
   for (const button of shapeButtons) {
     button.addEventListener("click", async () => {
       const term = currentTerm();
       const range = term && shapeRangeForSelection(term, currentSelection());
       if (!range || button.getAttribute("aria-disabled") === "true") return;
       const degree = Number(button.dataset.shapeDegree);
       await runStructuralChange(
         stageShapeRange(
           selectedTerm(), range.lo, range.hi, degree,
           effectiveShapeJoin(shapeJoinChoice, term.shape.joins)
         )
       );
     });
   }
   ```

**`docs/development/internals/editor-frontend.md`.** At line 97:

```
There is no successful post-refit `/state` fetch.
```
becomes
```
There is no successful post-refit `/state` fetch.

A staged change (`/stage`) returns the same envelope without fitting. Its revision is unchanged, so
it runs without the blocking overlay and re-requests no evidence. A refused request (HTTP 400) shows
Python's fixed sentence in the alert.
```
At lines 229–230:

```
5. Add a descriptor next to `setReferenceTransition` in `summary.js` and run it through
   `runStructuralRefit` from `main.js`.
```
becomes
```
5. Add a `stage…` descriptor next to `stageReference` in `summary.js` and run it through
   `runStructuralChange` from `main.js`. The change is staged on `/stage` and waits, drawn on the
   chart, until Refit (`/refit_pending`, through `runStructuralRefit`) applies every waiting change
   in one fit.
```

- [ ] **Step 4: Run tests, expect PASS**

```bash
npm run check:frontend
./.venv/bin/python -m pytest tests/test_editor.py tests/test_editor_structure.py -q
./.venv/bin/python -m pytest tests/editor/test_editor_refit_browser.py tests/editor/test_editor_structure_browser.py tests/editor/test_editor_workspace_browser.py tests/editor/test_editor_settings_browser.py tests/test_editor_browser.py -m browser --run-browser -q
./.venv/bin/python -m ruff check tests/editor tests/test_editor.py && ./.venv/bin/python -m ruff format --check tests/editor tests/test_editor.py
```

The workspace browser file must stay green. It covers the 360px selection palette
with the wider label column, the app bar at 900px with the Refit button, and the
summary-toggle guard.

- [ ] **Step 5: Commit**

```bash
git add src/superglm/editor/app/state/actions.js src/superglm/editor/app/state/selectors.js \
  src/superglm/editor/app/api/contracts.js src/superglm/editor/app/summary.js \
  src/superglm/editor/app/shapes.js src/superglm/editor/app/views/app_bar.js \
  src/superglm/editor/app/views/help_content.js src/superglm/editor/app/index.html \
  src/superglm/editor/app/main.js src/superglm/editor/app/styles/shell.css \
  src/superglm/editor/app/styles.css docs/development/internals/editor-frontend.md \
  tests/editor_frontend/actions.test.js tests/editor_frontend/app_bar.test.js \
  tests/editor_frontend/summary.test.js tests/editor_frontend/shapes.test.js \
  tests/editor_frontend/store.test.js tests/test_editor.py \
  tests/editor/test_editor_refit_browser.py tests/editor/test_editor_structure_browser.py \
  tests/editor/test_editor_workspace_browser.py
git commit -m "Editor: structural changes wait for one Refit (R)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task A5b: Waiting changes in the feature list, the status line and the export dialog

**Files:**
- Modify:
  - `src/superglm/editor/app/views/feature_list.js`: `RowState` (7),
    `renderFeatureList` (108–135) and `featureRow` (184–198);
  - `src/superglm/editor/app/views/context_bar.js`: before `renderContextBar`
    (15–22) and the status line (34–37);
  - `src/superglm/editor/app/views/export_dialog.js`: the typedefs (24–43), the
    signature (92) and `openDialog` (137–149);
  - `src/superglm/editor/app/state/selectors.js`, adding `selectWaitingTerms`;
  - `src/superglm/editor/app/main.js`: the export nodes and binding (132,
    525–542), the context-bar calls (806–816 and 979–993) and the feature-list
    render state (835–863);
  - `src/superglm/editor/app/index.html`, the export dialog (552–554);
  - `src/superglm/editor/app/styles/panels.css` (after 115), `styles.css` (after
    258) and `styles/dialogs.css` (before 133).
- Create: `tests/editor_frontend/context_bar.test.js`.
- Test (modify):
  - `tests/editor_frontend/feature_list.test.js`, `export_dialog.test.js` and
    `store.test.js`;
  - `tests/editor/test_editor_structure_browser.py`, with one new test.

**Interfaces:**
- Consumes: `selectPendingSteps` (A5), top-level `snapshot.pending`, and A4's
  `EditorSession.stage_structural` for the browser test.
- Produces:
  ```js
  export function selectWaitingTerms(state): string[];           // each waiting term once
  export function waitingLabel(count: number): string;           // views/context_bar.js
  renderContextBar(nodes, {name, term, selectionSize, note?, pendingCount?});
  renderFeatureList(nodes, {..., waiting?: ReadonlySet<string>});
  export function pendingExportNote(count: number): string;      // views/export_dialog.js
  bindExportDialog({client, nodes: {..., pendingNote?}, saveBlobToFile, pendingCount?});
  ```

- [ ] **Step 1: Write the failing tests**

Create `tests/editor_frontend/context_bar.test.js`:

```js
// @ts-nocheck

import assert from "node:assert/strict";
import test from "node:test";

import {
  renderContextBar,
  waitingLabel,
} from "../../src/superglm/editor/app/views/context_bar.js";

class FakeNode {
  constructor(tagName = "span") {
    this.tagName = tagName.toUpperCase();
    this.dataset = {};
    this.hidden = false;
    this.className = "";
    this.children = [];
    this.text = "";
    this.ownerDocument = { createElement: (tag) => new FakeNode(tag) };
  }

  get textContent() {
    return this.children.length
      ? this.children.map((child) => (typeof child === "string" ? child : child.textContent)).join("")
      : this.text;
  }

  set textContent(value) {
    this.children = [];
    this.text = String(value);
  }

  replaceChildren(...nodes) {
    this.text = "";
    this.children = nodes;
  }
}

const TERM = {
  kind: "categorical",
  term_type: "categorical",
  effective_df: 4,
  n_points: 10,
  reference: null,
  impact: { weighted_mean_relativity: 1.2, selected_weight_share: 0.25 },
};

function render(context) {
  const nodes = {
    nameNode: new FakeNode(),
    kindNode: new FakeNode(),
    edfNode: new FakeNode(),
    referenceNode: new FakeNode(),
    statusNode: new FakeNode(),
  };
  renderContextBar(nodes, { name: "territory", term: TERM, ...context });
  return nodes.statusNode;
}

test("with nothing waiting the status line is the selection sentence", () => {
  const status = render({ selectionSize: 2 });
  assert.equal(
    status.textContent,
    "2 of 10 selected · average edit relativity 1.2x · selected exposure 25%",
  );
  assert.equal(status.dataset.term, "territory");
});

test("changes waiting lead the status line, then the last-refit note or the selection", () => {
  const idle = render({ selectionSize: 0, pendingCount: 2 });
  assert.equal(idle.children[0].className, "status-waiting");
  assert.equal(idle.children[0].textContent, "2 changes waiting for refit");
  assert.equal(
    idle.textContent,
    "2 changes waiting for refit · the curve and metrics are from the last refit",
  );

  const selecting = render({ selectionSize: 3, pendingCount: 1 });
  assert.equal(
    selecting.textContent,
    "1 change waiting for refit · 3 of 10 selected · average edit relativity 1.2x · selected exposure 25%",
  );
  assert.equal(selecting.dataset.term, "territory");
  assert.equal(waitingLabel(1), "1 change waiting for refit");
});
```

Append to `tests/editor_frontend/feature_list.test.js`:

```js
test("a term with a change waiting for refit carries a dot on its row; the others carry none", () => {
  const { render, rows } = fixture();
  render({ waiting: new Set(["region"]) });
  const dots = rows().map(
    (row) => row.children.filter((part) => part.className === "feature-row-waiting"),
  );
  assert.deepEqual(dots.map((found) => found.length), [0, 0, 1, 0]);
  assert.equal(dots[2][0].getAttribute("role"), "img");
  assert.equal(dots[2][0].getAttribute("aria-label"), "Waiting for refit");

  render();
  assert.ok(rows().every((row) => row.children.length === 3));
});
```

Append to `tests/editor_frontend/export_dialog.test.js`:

```js
test("the dialog says how many waiting changes the export leaves out", async () => {
  const fixture = exportFixture();
  const pendingNote = new FakeElement();
  let waiting = 2;
  const binding = bindExportDialog({
    ...fixture.context,
    nodes: { ...fixture.nodes, pendingNote },
    pendingCount: () => waiting,
  });

  await fixture.action.emit("click");
  assert.equal(pendingNote.hidden, false);
  assert.equal(
    pendingNote.textContent,
    "2 waiting changes are not included. The export is the last refit.",
  );

  fixture.dialog.close();
  waiting = 1;
  await fixture.action.emit("click");
  assert.equal(
    pendingNote.textContent,
    "1 waiting change is not included. The export is the last refit.",
  );

  fixture.dialog.close();
  waiting = 0;
  await fixture.action.emit("click");
  assert.equal(pendingNote.hidden, true);
  assert.equal(pendingNote.textContent, "");
  binding.destroy();
});
```

In `tests/editor_frontend/store.test.js`, add `selectWaitingTerms` to the
selectors destructure (after `selectSnapshot`):

```js
  selectSummaryLevelDisplay,
  selectSnapshot
} = selectors;
```
becomes
```js
  selectSummaryLevelDisplay,
  selectSnapshot,
  selectWaitingTerms
} = selectors;
```
In the export-list test, change:

```js
    "selectSummaryLevelDisplay",
    "selectVisibleEvidencePanels"
  ]);
```
to
```js
    "selectSummaryLevelDisplay",
    "selectVisibleEvidencePanels",
    "selectWaitingTerms"
  ]);
```
Append:

```js
test("each term with a change waiting for refit is named once", () => {
  const confirmed = snapshot(7);
  confirmed.pending = [
    { id: "a1b2c3d", operation: "collapse", term: "region", label: "Collapse B + C", params: {}, note: null, time: 1 },
    { id: "b2c3d4e", operation: "shape", term: "age", label: "Line 1 – 2", params: {}, note: null, time: 2 },
    { id: "c3d4e5f", operation: "set_reference", term: "region", label: "Reference B", params: {}, note: null, time: 3 }
  ];
  assert.deepEqual(selectWaitingTerms(createInitialEditorState(confirmed)), ["region", "age"]);
  assert.deepEqual(selectWaitingTerms(createInitialEditorState(snapshot(7))), []);
});
```

Append to `tests/editor/test_editor_structure_browser.py`:

```python
def test_waiting_changes_show_in_the_feature_list_status_line_and_export(open_editor_page):
    with open_editor_page(selected_term="territory") as (page, session):
        session.stage_structural(
            "collapse", "territory", {"levels": ["T02", "T03"], "group_label": None}
        )
        _reload_editor(page, "territory")
        status = page.locator("#status")
        assert status.text_content() == (
            "1 change waiting for refit · the curve and metrics are from the last refit"
        )
        assert page.locator("#status .status-waiting").text_content() == (
            "1 change waiting for refit"
        )
        assert page.locator("#featureList .feature-row-waiting").count() == 1
        assert (
            page.locator('#featureList [data-term="territory"] .feature-row-waiting').count() == 1
        )

        page.locator("#exportAction").click()
        note = page.locator("#exportPendingNote")
        note.wait_for(state="visible")
        assert note.text_content() == (
            "1 waiting change is not included. The export is the last refit."
        )
        page.locator("#exportDialogClose").click()

        # With a selection, the waiting count still leads the line.
        session.select_levels("territory", ["T05"])
        _reload_editor(page, "territory")
        assert status.text_content().startswith("1 change waiting for refit · 1 of ")
```

- [ ] **Step 2: Run them, expect FAIL**

```bash
node --test tests/editor_frontend/context_bar.test.js tests/editor_frontend/feature_list.test.js tests/editor_frontend/export_dialog.test.js tests/editor_frontend/store.test.js
./.venv/bin/python -m pytest tests/editor/test_editor_structure_browser.py -m browser --run-browser -q -k waiting_changes_show
```

The expected failures:
- **Context bar.** `waitingLabel` is not exported, so the import throws
  `SyntaxError`.
- **Feature list.** No row gets a `.feature-row-waiting` child, giving
  `[0, 0, 0, 0]`.
- **Export.** The note stays at `hidden === false` and `textContent === ""`.
- **Store.** `selectWaitingTerms is not a function`.
- **Browser.** On the branch after A5, `#status` reads "0 of 10 selected · …". On
  `origin/master` 155832e8, `session.stage_structural` raises `AttributeError`.

- [ ] **Step 3: Implement**

**`src/superglm/editor/app/state/selectors.js`.** After `selectPendingSteps`, add:

```js
/**
 * Each term with a change waiting for Refit, once, in the order first staged.
 * @param {EditorState} state
 * @returns {string[]}
 */
export function selectWaitingTerms(state) {
  return [...new Set(selectPendingSteps(state).map((step) => step.term))];
}
```

**`src/superglm/editor/app/views/context_bar.js`.** B1 (S1) adds `kept` to
`REFERENCE_POLICY`. These edits leave that table alone. Before `renderContextBar`
(lines 15–22):

```js
/**
 * @param {{nameNode?:HTMLElement|null, kindNode:HTMLElement, edfNode:HTMLElement, referenceNode:HTMLElement, statusNode:HTMLElement}} nodes
 * @param {{name:string, term:TermPayload, selectionSize:number, note?:string}} context
 */
export function renderContextBar(
  { nameNode = null, kindNode, edfNode, referenceNode, statusNode },
  { name, term, selectionSize, note = "" },
) {
```
becomes
```js
/** The status line's note while changes wait and nothing is selected. */
const FROM_LAST_REFIT = "the curve and metrics are from the last refit";

/** @param {number} count */
export function waitingLabel(count) {
  return `${count} ${count === 1 ? "change" : "changes"} waiting for refit`;
}

/**
 * @param {{nameNode?:HTMLElement|null, kindNode:HTMLElement, edfNode:HTMLElement, referenceNode:HTMLElement, statusNode:HTMLElement}} nodes
 * @param {{name:string, term:TermPayload, selectionSize:number, note?:string, pendingCount?:number}} context
 */
export function renderContextBar(
  { nameNode = null, kindNode, edfNode, referenceNode, statusNode },
  { name, term, selectionSize, note = "", pendingCount = 0 },
) {
```
and the status line (lines 34–37):

```js
  const impact = term.impact || {};
  const suffix = note ? ` · ${note}` : "";
  statusNode.textContent = `${selectionSize} of ${term.n_points} selected · average edit relativity ${fmt(impact.weighted_mean_relativity || 1)}x · selected exposure ${fmtPercent(impact.selected_weight_share || 0)}${suffix}`;
  statusNode.dataset.term = name;
```
becomes
```js
  const impact = term.impact || {};
  const suffix = note ? ` · ${note}` : "";
  const selected = `${selectionSize} of ${term.n_points} selected · average edit relativity ${fmt(impact.weighted_mean_relativity || 1)}x · selected exposure ${fmtPercent(impact.selected_weight_share || 0)}${suffix}`;
  if (pendingCount > 0) {
    // While changes wait, the line leads with them: what is drawn is the last refit.
    const waiting = statusNode.ownerDocument.createElement("strong");
    waiting.className = "status-waiting";
    waiting.textContent = waitingLabel(pendingCount);
    statusNode.replaceChildren(waiting, ` · ${selectionSize ? selected : FROM_LAST_REFIT}`);
  } else {
    statusNode.textContent = selected;
  }
  statusNode.dataset.term = name;
```

**`src/superglm/editor/app/views/feature_list.js`.**
- `RowState` (line 7):
  ```js
  /** @typedef {{terms:Record<string, TermPayload>, activeTerm:string, tabStop:string|undefined}} RowState */
  ```
  becomes
  ```js
  /** @typedef {{terms:Record<string, TermPayload>, activeTerm:string, tabStop:string|undefined, waiting:ReadonlySet<string>}} RowState */
  ```
- `renderFeatureList` (lines 108–135):
  ```js
   * @param {{root:HTMLElement, rows:HTMLElement, toggle:HTMLButtonElement, strip:HTMLElement}} nodes
   * @param {{groups:FeatureGroup[], terms:Record<string, TermPayload>, activeTerm:string, query:string, open:boolean}} state
   */
  export function renderFeatureList(
    { root, rows, toggle, strip },
    { groups, terms, activeTerm, query, open },
  ) {
  ```
  becomes
  ```js
   * @param {{root:HTMLElement, rows:HTMLElement, toggle:HTMLButtonElement, strip:HTMLElement}} nodes
   * @param {{groups:FeatureGroup[], terms:Record<string, TermPayload>, activeTerm:string, query:string, open:boolean, waiting?:ReadonlySet<string>}} state
   */
  export function renderFeatureList(
    { root, rows, toggle, strip },
    { groups, terms, activeTerm, query, open, waiting = new Set() },
  ) {
  ```
  then
  ```js
    const rowState = { terms, activeTerm, tabStop };
  ```
  becomes
  ```js
    const rowState = { terms, activeTerm, tabStop, waiting };
  ```
- `featureRow` (lines 184–200):
  ```js
  function featureRow(doc, name, { terms, activeTerm, tabStop }) {
  ```
  becomes
  ```js
  function featureRow(doc, name, { terms, activeTerm, tabStop, waiting }) {
  ```
  and
  ```js
      span(doc, "feature-row-edf", edfLabel(term.effective_df)),
    );
    return row;
  }
  ```
  becomes
  ```js
      span(doc, "feature-row-edf", edfLabel(term.effective_df)),
    );
    if (waiting.has(name)) row.append(waitingDot(doc));
    return row;
  }

  /** An amber dot on a term with a change waiting for Refit. @param {Document} doc */
  function waitingDot(doc) {
    const dot = doc.createElement("span");
    dot.className = "feature-row-waiting";
    dot.setAttribute("role", "img");
    dot.setAttribute("aria-label", "Waiting for refit");
    return dot;
  }
  ```

**`src/superglm/editor/app/views/export_dialog.js`.**
- The nodes typedef (lines 34–35):
  ```js
   * @property {HTMLElement} status
   */
  ```
  becomes
  ```js
   * @property {HTMLElement} status
   * @property {HTMLElement|null} [pendingNote] says how many waiting changes the export leaves out
   */
  ```
  That string occurs once in the file, in `ExportDialogNodes`.
- The context typedef (line 42):
  ```js
   * @property {(blob:Blob, filename:string, metadata:{description:string,accept:Readonly<Record<string,readonly string[]>>})=>Promise<string|null>} saveBlobToFile
   */
  ```
  becomes
  ```js
   * @property {(blob:Blob, filename:string, metadata:{description:string,accept:Readonly<Record<string,readonly string[]>>})=>Promise<string|null>} saveBlobToFile
   * @property {()=>number} [pendingCount] how many structural changes wait for Refit
   */
  ```
- After `successMessage` (ends at line 85), add:
  ```js

  /**
   * What the dialog says while changes wait: the export is the last refit.
   * @param {number} count
   */
  export function pendingExportNote(count) {
    if (count <= 0) return "";
    return `${count} waiting ${count === 1 ? "change is" : "changes are"} not included. The export is the last refit.`;
  }
  ```
- The signature (line 92):
  ```js
  export function bindExportDialog({ client, nodes, saveBlobToFile }) {
  ```
  becomes
  ```js
  export function bindExportDialog({ client, nodes, saveBlobToFile, pendingCount = () => 0 }) {
  ```
- `openDialog` (lines 137–139):
  ```js
    async function openDialog() {
      nodes.status.textContent = "";
      if (nodes.dialog.open) return;
  ```
  becomes
  ```js
    async function openDialog() {
      nodes.status.textContent = "";
      if (nodes.pendingNote) {
        const note = pendingExportNote(pendingCount());
        nodes.pendingNote.textContent = note;
        nodes.pendingNote.hidden = note === "";
      }
      if (nodes.dialog.open) return;
  ```

**`src/superglm/editor/app/index.html`.** In the export dialog (lines 552–554):

```html
      <button id="exportDialogClose" type="button" aria-label="Close export dialog">Close</button>
    </div>
    <fieldset class="export-format-group">
```
becomes
```html
      <button id="exportDialogClose" type="button" aria-label="Close export dialog">Close</button>
    </div>
    <p id="exportPendingNote" class="export-pending-note" role="note" hidden></p>
    <fieldset class="export-format-group">
```

**`src/superglm/editor/app/main.js`.**
1. Selector import:
   ```js
     selectSummaryLevelDisplay,
     selectVisibleEvidencePanels
   } from "./state/selectors.js";
   ```
   becomes
   ```js
     selectSummaryLevelDisplay,
     selectVisibleEvidencePanels,
     selectWaitingTerms
   } from "./state/selectors.js";
   ```
2. The export note node (line 132):
   ```js
   const exportFormatInputs = [...document.querySelectorAll('input[name="exportFormat"]')];
   ```
   becomes
   ```js
   const exportFormatInputs = [...document.querySelectorAll('input[name="exportFormat"]')];
   const exportPendingNote = document.getElementById("exportPendingNote");
   ```
3. `bindExportDialog` (lines 536–542):
   ```js
       openDirectory: exportOpenDirectory instanceof HTMLButtonElement
         ? exportOpenDirectory
         : null,
       status: exportStatus
     },
     saveBlobToFile
   });
   ```
   becomes
   ```js
       openDirectory: exportOpenDirectory instanceof HTMLButtonElement
         ? exportOpenDirectory
         : null,
       status: exportStatus,
       pendingNote: exportPendingNote instanceof HTMLElement ? exportPendingNote : null
     },
     saveBlobToFile,
     pendingCount: () => selectPendingSteps(store.getState()).length
   });
   ```
4. `renderChartWorkspace` (line 815):
   ```js
       { name: selected, term, selectionSize: selection.size, note: collapsedOriginalNote }
   ```
   becomes
   ```js
       {
         name: selected,
         term,
         selectionSize: selection.size,
         note: collapsedOriginalNote,
         pendingCount: selectPendingSteps(editorState).length
       }
   ```
5. `renderSelectionState` (lines 987–992):
   ```js
       {
         name: termName,
         term,
         selectionSize: selection.size,
         note: selectionContextNote(term)
       }
   ```
   becomes
   ```js
       {
         name: termName,
         term,
         selectionSize: selection.size,
         note: selectionContextNote(term),
         pendingCount: selectPendingSteps(store.getState()).length
       }
   ```
6. The feature-list render state (lines 835–863):
   ```js
   // The revision stands in for every row's EDF, which only a refit changes.
   function selectFeatureListRenderState(state) {
     const snapshot = selectSnapshot(state);
     return {
       ready: snapshot !== null,
       catalogueKey: snapshot ? termCatalogueKey(snapshot.terms || {}) : "",
       revision: selectModelRevision(state),
       activeTerm: selectActiveTermName(state)
     };
   }

   function sameFeatureListRenderState(next, previous) {
     return next.ready === previous.ready &&
       next.catalogueKey === previous.catalogueKey &&
       next.revision === previous.revision &&
       next.activeTerm === previous.activeTerm;
   }
   ```
   becomes
   ```js
   // The revision stands in for every row's EDF, which only a refit changes; a
   // stage leaves the revision, so the waiting terms are keyed on their own.
   function selectFeatureListRenderState(state) {
     const snapshot = selectSnapshot(state);
     return {
       ready: snapshot !== null,
       catalogueKey: snapshot ? termCatalogueKey(snapshot.terms || {}) : "",
       revision: selectModelRevision(state),
       activeTerm: selectActiveTermName(state),
       waiting: selectWaitingTerms(state).join("\u0000")
     };
   }

   function sameFeatureListRenderState(next, previous) {
     return next.ready === previous.ready &&
       next.catalogueKey === previous.catalogueKey &&
       next.revision === previous.revision &&
       next.activeTerm === previous.activeTerm &&
       next.waiting === previous.waiting;
   }
   ```
   then
   ```js
       query: featureQuery,
       open: featureListOpen
     });
   }
   ```
   becomes
   ```js
       query: featureQuery,
       open: featureListOpen,
       waiting: new Set(selectWaitingTerms(state))
     });
   }
   ```

**CSS.**
- `styles/panels.css`, after `.feature-row-edf { font-variant-numeric: tabular-nums; }`
  (lines 113–115):
  ```css

  /* A term with a change waiting for Refit carries an amber dot on its name line. */
  .feature-row:has(.feature-row-waiting) {
    position: relative;
  }

  .feature-row:has(.feature-row-waiting) .feature-row-name {
    padding-right: 12px;
  }

  .feature-row-waiting {
    position: absolute;
    top: 11px;
    right: 8px;
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--group-0);
  }
  ```
- `styles.css`, after `#status.is-error { color: var(--danger); }` (lines 256–258):
  ```css
  #status .status-waiting {
    color: var(--sig-weak-fg);
    font-weight: 600;
  }
  ```
- `styles/dialogs.css`, before `.export-fields {` (line 133):
  ```css
  .export-pending-note {
    margin: 0 0 var(--space-3);
    padding: 6px 10px;
    border-radius: var(--radius-md);
    background: var(--sig-weak-bg);
    color: var(--sig-weak-fg);
    font-size: 12px;
    font-weight: 600;
  }
  ```

- [ ] **Step 4: Run tests, expect PASS**

```bash
npm run check:frontend
./.venv/bin/python -m pytest tests/test_editor.py tests/test_lss_editor_style.py -q
./.venv/bin/python -m pytest tests/editor/test_editor_structure_browser.py tests/editor/test_editor_workspace_browser.py tests/editor/test_editor_refit_browser.py -m browser --run-browser -q
./.venv/bin/python -m ruff check tests/editor && ./.venv/bin/python -m ruff format --check tests/editor
```

The workspace browser file asserts many `#status` sentences that start "N of".
With nothing waiting they must be unchanged.

- [ ] **Step 5: Commit**

```bash
git add src/superglm/editor/app/state/selectors.js src/superglm/editor/app/views/context_bar.js \
  src/superglm/editor/app/views/feature_list.js src/superglm/editor/app/views/export_dialog.js \
  src/superglm/editor/app/index.html src/superglm/editor/app/main.js \
  src/superglm/editor/app/styles/panels.css src/superglm/editor/app/styles.css \
  src/superglm/editor/app/styles/dialogs.css tests/editor_frontend/context_bar.test.js \
  tests/editor_frontend/feature_list.test.js tests/editor_frontend/export_dialog.test.js \
  tests/editor_frontend/store.test.js tests/editor/test_editor_structure_browser.py
git commit -m "Editor: waiting changes in the feature list, status line and export dialog

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task A5c: Waiting changes on the chart

**Files:**
- Create:
  - `src/superglm/editor/app/chart/pending_overlay.js`;
  - `tests/editor_frontend/pending_overlay.test.js`.
- Modify:
  - `src/superglm/editor/app/chart.js`: the imports (2), the constants (15), the
    view and axis layout in `drawChart` (70–100), the exposure call (148), the
    title (191), the shape overlay (200), the level groups (230–233),
    `applyPlotClip` (466), `categoricalAxisLayout` (1015–1036), `exposureLayer`
    (1141–1170) and a helper next to `levelGroupColor` (652);
  - `src/superglm/editor/app/chart/geometry.js`: the plan typedef (49–59) and its
    return (239–247);
  - `src/superglm/editor/app/chart/shape_overlay.js`, exporting `labelWidth` (61–62);
  - `src/superglm/editor/app/styles/chart.css`, appended after line 79.
- Test (modify):
  - `tests/editor_frontend/chart_geometry.test.js`;
  - `tests/test_editor.py`, the asset list;
  - `tests/editor/test_editor_structure_browser.py`, with two new tests.

**Interfaces:**
- Consumes: per-term `pending.groups` and `pending.ranges` (amendment 6), and
  `levelGroupColor` in chart.js.
- Produces:
  ```js
  // chart/pending_overlay.js
  export const WAITING_BRACKET_ROW = 20;
  export function pendingGroupMarks(term, view): PendingGroupMark[]; // {label, members, display, slot}
  export function waitingBracketText(mark): string;                  // "B10 + B11 · waiting", "5 levels · waiting"
  export function drawPendingRanges(svg, {term, view, sx, margin, innerW, innerH}): void;
  export function drawPendingGroupRings(svg, marks, {view, sx, sy, color}): void;
  export function drawPendingGroupBrackets(svg, marks, {view, sx, top, left, right, color}): void;
  // chart/geometry.js: CategoricalAxisPlan.labelsBottom
  // chart/shape_overlay.js: export function labelWidth(label)
  ```
  SVG classes:
  - `.pending-range`, `.pending-range-box`, `.pending-range-tag` and
    `.pending-range-label` for a waiting range;
  - `.pending-group-bracket`, `.pending-group-bracket-line`,
    `.pending-group-label` and `.pending-group-ring` for a waiting group;
  - `rect.exposure.waiting` for a member's exposure bar.

- [ ] **Step 1: Write the failing tests**

Create `tests/editor_frontend/pending_overlay.test.js`:

```js
// @ts-nocheck

import assert from "node:assert/strict";
import test from "node:test";

import {
  pendingGroupMarks,
  waitingBracketText,
} from "../../src/superglm/editor/app/chart/pending_overlay.js";

const LEVELS = ["T01", "T02", "T03", "T04", "T05"];
const expanded = { x: LEVELS.map((_, i) => i), displayToSourceIndices: LEVELS.map((_, i) => [i]) };

function waiting(groups, levelGroups = []) {
  return {
    levels: LEVELS.slice(),
    level_groups: levelGroups,
    pending: { groups, ranges: [], reference: null },
  };
}

test("each waiting group takes its members' points, in axis order, with the next palette slots", () => {
  const term = waiting({ "T04+T05": ["T05", "T04"], "T02+T03": ["T02", "T03"] });
  assert.deepEqual(pendingGroupMarks(term, expanded), [
    { label: "T02+T03", members: ["T02", "T03"], display: [1, 2], slot: 0 },
    { label: "T04+T05", members: ["T04", "T05"], display: [3, 4], slot: 1 },
  ]);
});

test("a group the fitted term already has is not waiting, and slots follow the fitted groups", () => {
  const term = waiting(
    { "T01+T02": ["T01", "T02"], "T03+T04": ["T03", "T04"] },
    [{ label: "T01+T02", indices: [1, 0] }],
  );
  assert.deepEqual(pendingGroupMarks(term, expanded), [
    { label: "T03+T04", members: ["T03", "T04"], display: [2, 3], slot: 1 },
  ]);
});

test("drawn collapsed, a waiting group maps through the display's source levels", () => {
  const collapsed = { x: [0, 1, 2, 3], displayToSourceIndices: [[0], [1, 2], [3], [4]] };
  const term = waiting({ "T03+T04": ["T03", "T04"] }, [{ label: "T02+T03", indices: [1, 2] }]);
  assert.deepEqual(pendingGroupMarks(term, collapsed)[0].display, [1, 2]);
});

test("levels that look like numbers match as labels, and unknown or lone members draw nothing", () => {
  const term = {
    levels: ["1", "2", "10"],
    pending: { groups: { "1+10": [1, 10], solo: ["2", "99"] }, ranges: [], reference: null },
  };
  const axis = { x: [0, 1, 2], displayToSourceIndices: [[0], [1], [2]] };
  assert.deepEqual(pendingGroupMarks(term, axis), [
    { label: "1+10", members: ["1", "10"], display: [0, 2], slot: 0 },
  ]);
});

test("a term with nothing waiting, or without levels, draws no group", () => {
  assert.deepEqual(pendingGroupMarks({ levels: LEVELS, pending: null }, expanded), []);
  assert.deepEqual(pendingGroupMarks({ levels: LEVELS }, expanded), []);
  const numeric = { levels: null, pending: { groups: { a: ["x", "y"] }, ranges: [], reference: null } };
  assert.deepEqual(pendingGroupMarks(numeric, expanded), []);
});

test("the bracket names up to three members and counts more", () => {
  assert.equal(waitingBracketText({ members: ["B10", "B11"] }), "B10 + B11 · waiting");
  assert.equal(waitingBracketText({ members: ["A", "B", "C"] }), "A + B + C · waiting");
  assert.equal(waitingBracketText({ members: ["A", "B", "C", "D", "E"] }), "5 levels · waiting");
});
```

Append to `tests/editor_frontend/chart_geometry.test.js`:

```js
test("a taller title row grows the bottom gutter and leaves the labels where they were", () => {
  const labels = ["T01", "T02", "T03"];
  /** @param {number} titleHeight */
  const plan = (titleHeight) => planCategoricalAxis({
    values: [0, 1, 2],
    labels,
    measurements: labels.map((label) => measurement(label)),
    availableWidth: 788,
    svgHeight: 520,
    baseLeft: 76,
    baseBottom: 0,
    titleHeight,
  });
  const plain = plan(14);
  const roomy = plan(34);
  assert.ok(plain.axisY < plain.labelsBottom && plain.labelsBottom < plain.titleY);
  assert.equal(roomy.bottom - plain.bottom, 20);
  assert.equal(roomy.labelsBottom - roomy.axisY, plain.labelsBottom - plain.axisY);
  assert.equal(roomy.titleY - roomy.labelsBottom, plain.titleY - plain.labelsBottom);
});
```

In `tests/test_editor.py`'s asset list (after the line F1 added):

```python
            "views/settings.js",
        ]:
```
becomes
```python
            "views/settings.js",
            "chart/pending_overlay.js",
        ]:
```

Append to `tests/editor/test_editor_structure_browser.py`:

```python
def test_a_waiting_collapse_is_drawn_dashed_with_a_bracket_under_the_axis(open_editor_page):
    with open_editor_page(selected_term="territory") as (page, session):
        session.stage_structural(
            "collapse", "territory", {"levels": ["T02", "T03"], "group_label": None}
        )
        _reload_editor(page, "territory")
        bracket = page.locator("#chart .pending-group-bracket")
        assert bracket.count() == 1
        assert bracket.locator(".pending-group-label").text_content() == "T02 + T03 · waiting"
        assert page.locator("#chart rect.exposure.waiting").count() == 2
        assert page.locator("#chart .pending-group-ring").count() == 2
        # The curve is still the last refit's: nothing is grouped in force yet.
        assert page.locator("#chart .level-group-marker").count() == 0
        # The bracket has its own row between the level labels and the axis title.
        rows = page.evaluate(
            """() => {
                const svg = document.querySelector('#chart');
                const label = svg.querySelector('.pending-group-label').getBBox();
                const title = svg.querySelector('.x-axis-title').getBBox();
                const ticks = Array.from(svg.querySelectorAll('.x-tick-label'), n => n.getBBox());
                return {
                    ticksBottom: Math.max(...ticks.map(box => box.y + box.height)),
                    labelTop: label.y,
                    labelBottom: label.y + label.height,
                    titleTop: title.y,
                };
            }"""
        )
        assert rows["ticksBottom"] <= rows["labelTop"]
        assert rows["labelBottom"] <= rows["titleTop"]


def test_a_waiting_range_is_a_dashed_box_until_refit_pins_it(open_editor_page):
    with open_editor_page() as (page, session):
        session.stage_structural(
            "shape", "curve", {"lo": 3.0, "hi": 5.0, "degree": 1, "join": "tangent"}
        )
        _reload_editor(page, "curve")
        waiting = page.locator("#chart .pending-range")
        assert waiting.count() == 1
        assert (
            waiting.locator(".pending-range-label").text_content().endswith(" · waiting for refit")
        )
        assert page.locator("#chart .shape-range").count() == 0
        [staged] = session_payload(session)["curve"]["pending"]["ranges"]
        extent = page.evaluate(
            """([lo, hi]) => {
                const svg = document.querySelector('#chart');
                const rect = svg.querySelector('.pending-range-box');
                const left = Number(rect.getAttribute('x'));
                return {
                    left,
                    right: left + Number(rect.getAttribute('width')),
                    expected: [svg._scale.sx(lo), svg._scale.sx(hi)],
                };
            }""",
            [staged["lo"], staged["hi"]],
        )
        assert [extent["left"], extent["right"]] == pytest.approx(extent["expected"], abs=1e-9)

        with page.expect_response(_posted("/refit_pending")):
            page.keyboard.press("r")
        _settled_after_refit(page)
        assert waiting.count() == 0
        assert page.locator("#chart .shape-range").count() == 1
```

- [ ] **Step 2: Run them, expect FAIL**

```bash
node --test tests/editor_frontend/pending_overlay.test.js tests/editor_frontend/chart_geometry.test.js
./.venv/bin/python -m pytest tests/test_editor.py -q -k serves_editor_app_assets
./.venv/bin/python -m pytest tests/editor/test_editor_structure_browser.py -m browser --run-browser -q -k "waiting_collapse_is_drawn or waiting_range_is_a_dashed"
```

The expected failures:
- `pending_overlay.test.js` fails with `ERR_MODULE_NOT_FOUND`.
- The geometry test fails, because `plain.labelsBottom` is `undefined` and
  `plain.axisY < undefined` is false.
- The asset fetch returns 404.
- The browser tests find `.pending-group-bracket` and `.pending-range` with count
  0 on the branch after A5b. On `origin/master` 155832e8, `stage_structural` does
  not exist.

- [ ] **Step 3: Implement**

Create `src/superglm/editor/app/chart/pending_overlay.js`:

```js
// @ts-check
// Structural changes waiting for Refit, drawn over the last refit's curve,
// which stays as it is until Refit. A waiting range is a dashed box with a
// "waiting for refit" tag. A waiting group shows its members' exposure bars
// dashed in its group colour (chart.js colours them from pendingGroupMarks),
// rings their points, and is named on a dashed bracket under the axis labels,
// in a row the categorical axis keeps for it.

import { shapeRangeDescription, shapeRangeExtent } from "../shapes.js";
import { labelWidth } from "./shape_overlay.js";
import { el, text } from "./svg.js";

/** @typedef {import('../api/contracts.js').TermPayload} TermPayload */
/** @typedef {import('../shapes.js').DisplayAxis} DisplayAxis */
/** @typedef {(slot:number, alpha?:number)=>string} GroupColor */
/**
 * One waiting group on the displayed axis.
 * @typedef {object} PendingGroupMark
 * @property {string} label the group's label
 * @property {string[]} members its levels, in axis order
 * @property {number[]} display the displayed points it takes in, ascending
 * @property {number} slot its colour in the level-group palette, after the fitted groups'
 */

/** The row a waiting group's bracket takes under the axis labels, in px. */
export const WAITING_BRACKET_ROW = 20;
const BRACKET_DROP = 6;
const BRACKET_TICK = 4;
const BRACKET_LABEL_GAP = 11;
const BRACKET_PAD = 0.35;
const RING_RADIUS = 6.5;
const TAG_INSET = 6;
const TAG_PADDING = 8;
const TAG_HEIGHT = 20;
const TAG_BASELINE = 20;

/**
 * The waiting groups to draw: each group the waiting changes make that the
 * fitted term does not have yet, with its members' displayed points and a
 * palette slot after the fitted groups' slots. Members are matched to the
 * term's levels as strings, so levels that look like numbers match.
 * @param {TermPayload} term @param {DisplayAxis} view
 * @returns {PendingGroupMark[]}
 */
export function pendingGroupMarks(term, view) {
  const groups = term.pending?.groups;
  if (!groups || !Array.isArray(term.levels)) return [];
  const levels = term.levels.map(String);
  const fitted = new Set(
    (term.level_groups ?? []).map((group) => [...group.indices].sort((a, b) => a - b).join(","))
  );
  /** @type {Omit<PendingGroupMark, "slot">[]} */
  const marks = [];
  for (const [label, members] of Object.entries(groups)) {
    const sources = [...new Set(members.map((member) => levels.indexOf(String(member))))]
      .filter((index) => index >= 0)
      .sort((left, right) => left - right);
    if (sources.length < 2 || fitted.has(sources.join(","))) continue;
    /** @type {number[]} */
    const display = [];
    view.displayToSourceIndices.forEach((indices, position) => {
      if (indices.some((index) => sources.includes(index))) display.push(position);
    });
    if (!display.length) continue;
    marks.push({ label, members: sources.map((index) => levels[index]), display });
  }
  marks.sort((left, right) => left.display[0] - right.display[0]);
  const first = term.level_groups?.length ?? 0;
  return marks.map((mark, index) => ({ ...mark, slot: first + index }));
}

/** @param {{members:readonly string[]}} mark */
export function waitingBracketText(mark) {
  const named = mark.members.length <= 3
    ? mark.members.join(" + ")
    : `${mark.members.length} levels`;
  return `${named} · waiting`;
}

/**
 * Each waiting range as a dashed box over the plot, tagged at its top.
 * @param {SVGElement} svg
 * @param {{term:TermPayload, view:DisplayAxis, sx:(v:number)=>number,
 *   margin:{left:number, top:number}, innerW:number, innerH:number}} options
 */
export function drawPendingRanges(svg, { term, view, sx, margin, innerW, innerH }) {
  const ranges = term.pending?.ranges ?? [];
  if (!ranges.length) return;
  const left = margin.left;
  const right = margin.left + innerW;
  const layer = el("g", { class: "pending-layer" });
  // In the document before the tags are made, so each label can be measured.
  svg.appendChild(layer);
  for (const range of ranges) {
    const extent = shapeRangeExtent(term, view, range);
    if (!extent) continue;
    const x0 = Math.max(left, sx(extent[0]));
    const x1 = Math.min(right, sx(extent[1]));
    if (x1 <= x0) continue;
    const name = `${range.label} · waiting for refit`;
    const band = el("g", {
      class: "pending-range",
      "data-popover-title": name,
      "data-popover-body": `${shapeRangeDescription(range)} It applies at the next Refit.`
    });
    band.appendChild(el("rect", {
      class: "pending-range-box", x: x0, y: margin.top, width: x1 - x0, height: innerH
    }));
    layer.appendChild(band);
    const label = text(
      band, x0 + TAG_INSET + TAG_PADDING, margin.top + TAG_BASELINE, name,
      "pending-range-label", "start"
    );
    band.insertBefore(el("rect", {
      class: "pending-range-tag",
      x: x0 + TAG_INSET,
      y: margin.top + TAG_INSET,
      width: labelWidth(label) + TAG_PADDING * 2,
      height: TAG_HEIGHT,
      rx: TAG_HEIGHT / 2,
      ry: TAG_HEIGHT / 2
    }), label);
  }
}

/**
 * A ring in its group's colour around each point a waiting group takes in.
 * @param {SVGElement} svg @param {readonly PendingGroupMark[]} marks
 * @param {{view:{x:number[], y:number[]}, sx:(v:number)=>number, sy:(v:number)=>number,
 *   color:GroupColor}} options
 */
export function drawPendingGroupRings(svg, marks, { view, sx, sy, color }) {
  for (const mark of marks) {
    for (const position of mark.display) {
      svg.appendChild(el("circle", {
        class: "pending-group-ring",
        cx: sx(view.x[position]),
        cy: sy(view.y[position]),
        r: RING_RADIUS,
        style: `stroke: ${color(mark.slot, 0.95)}`
      }));
    }
  }
}

/**
 * A dashed bracket under the axis labels for each waiting group, named for
 * its members. ``top`` is the bottom of the tick labels; ``left`` and
 * ``right`` bound the plot, so a zoom clips the bracket and drops a group
 * zoomed out of view.
 * @param {SVGElement} svg @param {readonly PendingGroupMark[]} marks
 * @param {{view:{x:number[]}, sx:(v:number)=>number, top:number, left:number, right:number,
 *   color:GroupColor}} options
 */
export function drawPendingGroupBrackets(svg, marks, { view, sx, top, left, right, color }) {
  const step = view.x.length > 1 ? Math.abs(sx(view.x[1]) - sx(view.x[0])) : right - left;
  const pad = BRACKET_PAD * step;
  const lineY = top + BRACKET_DROP;
  for (const mark of marks) {
    const first = view.x[mark.display[0]];
    const last = view.x[mark.display[mark.display.length - 1]];
    const x0 = Math.max(left, sx(first) - pad);
    const x1 = Math.min(right, sx(last) + pad);
    if (x1 <= x0) continue;
    const bracket = el("g", {
      class: "pending-group-bracket",
      "data-popover-title": "Waiting for refit",
      "data-popover-body": `${mark.members.join(", ")} become one group at the next Refit.`
    });
    bracket.appendChild(el("path", {
      class: "pending-group-bracket-line",
      d: `M ${x0.toFixed(2)} ${lineY - BRACKET_TICK} V ${lineY} H ${x1.toFixed(2)} V ${lineY - BRACKET_TICK}`,
      style: `stroke: ${color(mark.slot, 0.95)}`
    }));
    const label = text(
      bracket, (x0 + x1) / 2, lineY + BRACKET_LABEL_GAP, waitingBracketText(mark),
      "pending-group-label", "middle"
    );
    label.setAttribute("style", `fill: ${color(mark.slot, 1)}`);
    svg.appendChild(bracket);
  }
}
```

In `src/superglm/editor/app/chart/shape_overlay.js` (lines 61–62):

```js
/** @param {SVGElement} label */
function labelWidth(label) {
```
becomes
```js
/**
 * A drawn label's width: measured where the SVG has layout, else estimated.
 * @param {SVGElement} label
 */
export function labelWidth(label) {
```

In `src/superglm/editor/app/chart/geometry.js`, the plan typedef (lines 55–58):

```js
 * @property {number} titleY
 * @property {number} titleHeight
 * @property {number} maxLabelHeight
```
becomes
```js
 * @property {number} titleY
 * @property {number} titleHeight
 * @property {number} maxLabelHeight
 * @property {number} labelsBottom where the tick labels end, above the title row
```
and the return (lines 239–246):

```js
  return {
    ticks,
    bottom,
    axisY,
    titleY,
    titleHeight,
    maxLabelHeight,
    labelBudget,
  };
```
becomes
```js
  return {
    ticks,
    bottom,
    axisY,
    titleY,
    titleHeight,
    maxLabelHeight,
    labelsBottom: axisY + TICK_OFFSET + maxLabelHeight,
    labelBudget,
  };
```

In `src/superglm/editor/app/chart.js`:
1. Imports (line 2):
   ```js
   import { drawShapeOverlay } from "./chart/shape_overlay.js";
   ```
   becomes
   ```js
   import { drawShapeOverlay } from "./chart/shape_overlay.js";
   import {
     WAITING_BRACKET_ROW,
     drawPendingGroupBrackets,
     drawPendingGroupRings,
     drawPendingRanges,
     pendingGroupMarks
   } from "./chart/pending_overlay.js";
   ```
2. Constants (line 15):
   ```js
   const LENS_HALF_WIDTH = 26;
   ```
   becomes
   ```js
   const LENS_HALF_WIDTH = 26;
   // The x-axis title's own row, as planCategoricalAxis reserves it by default.
   const AXIS_TITLE_HEIGHT = 14;
   ```
3. In `drawChart` (lines 70–73):
   ```js
     const view = resolveDisplayTerm(
       term,
       context.groupDisplayMode ? context.groupDisplayMode() : "expanded"
     );
   ```
   becomes
   ```js
     const view = resolveDisplayTerm(
       term,
       context.groupDisplayMode ? context.groupDisplayMode() : "expanded"
     );
     // A waiting group is named on a bracket under the axis labels, which takes
     // a row of its own. The handles view draws neither groups nor brackets.
     const waitingGroups = visualMode === "handles" && term.controls
       ? []
       : pendingGroupMarks(term, view);
     const bracketRow = waitingGroups.length ? WAITING_BRACKET_ROW : 0;
   ```
   and (lines 97–100)
   ```js
           height,
           baseMargin
         )
       : null;
   ```
   becomes
   ```js
           height,
           baseMargin,
           bracketRow
         )
       : null;
   ```
4. The exposure call (line 148):
   ```js
     exposureLayer(svg, view, sx, margin, innerW, innerH, exposure);
   ```
   becomes
   ```js
     exposureLayer(svg, view, sx, margin, innerW, innerH, exposure, waitingSlots(waitingGroups));
   ```
5. The title (line 191):
   ```js
       categoricalLayout ? categoricalLayout.titleY : height - 12,
   ```
   becomes
   ```js
       categoricalLayout ? categoricalLayout.titleY + bracketRow : height - 12,
   ```
6. The overlays (line 200):
   ```js
     if (!buildActive) drawShapeOverlay(svg, { term, view, sx, margin, innerW, innerH });
   ```
   becomes
   ```js
     if (!buildActive) drawShapeOverlay(svg, { term, view, sx, margin, innerW, innerH });
     if (!buildActive) drawPendingRanges(svg, { term, view, sx, margin, innerW, innerH });
   ```
7. The level groups (lines 230–233):
   ```js
     if (!handlesMode) {
       if (view.displayIsCollapsed) drawCollapsedLevelGroups(svg, view, sx, sy);
       else drawLevelGroups(svg, view, sx, sy);
     }
   ```
   becomes
   ```js
     if (!handlesMode) {
       if (view.displayIsCollapsed) drawCollapsedLevelGroups(svg, view, sx, sy);
       else drawLevelGroups(svg, view, sx, sy);
     }
     if (waitingGroups.length && categoricalLayout) {
       drawPendingGroupRings(svg, waitingGroups, { view, sx, sy, color: levelGroupColor });
       drawPendingGroupBrackets(svg, waitingGroups, {
         view,
         sx,
         top: categoricalLayout.labelsBottom,
         left: margin.left,
         right: margin.left + innerW,
         color: levelGroupColor
       });
     }
   ```
8. `applyPlotClip` (lines 466–467). `".level-group-marker",` also appears at line 845,
   so match it with the line after:
   ```js
       ".level-group-marker",
       ".point",
   ```
   becomes
   ```js
       ".level-group-marker",
       ".pending-group-ring",
       ".point",
   ```
9. `categoricalAxisLayout` (line 1015):
   ```js
   function categoricalAxisLayout(svg, view, xMin, xMax, availableWidth, svgHeight, baseMargin) {
   ```
   becomes
   ```js
   function categoricalAxisLayout(
     svg, view, xMin, xMax, availableWidth, svgHeight, baseMargin, extraRow = 0
   ) {
   ```
   and (lines 1033–1036)
   ```js
       baseLeft: baseMargin.left,
       baseBottom: baseMargin.bottom
     });
   }
   ```
   becomes
   ```js
       baseLeft: baseMargin.left,
       baseBottom: baseMargin.bottom,
       titleHeight: AXIS_TITLE_HEIGHT + extraRow
     });
   }
   ```
10. `exposureLayer` (line 1141):
    ```js
    function exposureLayer(svg, term, sx, margin, innerW, innerH, exposure) {
    ```
    becomes
    ```js
    function exposureLayer(svg, term, sx, margin, innerW, innerH, exposure, waiting = new Map()) {
    ```
    and the bars (lines 1158–1169):
    ```js
        for (let i = 0; i < exposure.y.length; i++) {
          const h = Math.max(1, maxH * exposure.y[i] / maxWeight);
          svg.appendChild(el("rect", {
            x: sx(x[i]) - nominalW / 2,
            y: yBase - h,
            width: nominalW,
            height: h,
            rx: 2,
            ry: 2,
            class: "exposure"
          }));
        }
    ```
    become
    ```js
        for (let i = 0; i < exposure.y.length; i++) {
          const h = Math.max(1, maxH * exposure.y[i] / maxWeight);
          const bar = el("rect", {
            x: sx(x[i]) - nominalW / 2,
            y: yBase - h,
            width: nominalW,
            height: h,
            rx: 2,
            ry: 2,
            class: waiting.has(i) ? "exposure waiting" : "exposure"
          });
          // A level in a waiting group shows its bar dashed, in the group's colour.
          if (waiting.has(i)) {
            const slot = waiting.get(i);
            bar.setAttribute(
              "style",
              `fill: ${levelGroupColor(slot, 0.22)}; stroke: ${levelGroupColor(slot, 0.95)}`
            );
          }
          svg.appendChild(bar);
        }
    ```
11. A helper above `levelGroupColor` (line 650):
    ```js
    // The categorical palettes live in tokens.css, one value per theme: the SVG
    // names a colour by index and the stylesheet supplies it, at the alpha asked.
    function levelGroupColor(index, alpha = 1) {
    ```
    becomes
    ```js
    // The palette slot of each displayed point a waiting group takes in.
    function waitingSlots(marks) {
      const slots = new Map();
      for (const mark of marks) {
        for (const position of mark.display) slots.set(position, mark.slot);
      }
      return slots;
    }

    // The categorical palettes live in tokens.css, one value per theme: the SVG
    // names a colour by index and the stylesheet supplies it, at the alpha asked.
    function levelGroupColor(index, alpha = 1) {
    ```

Append to `src/superglm/editor/app/styles/chart.css`, after the `.shape-range-label`
rule (ends at line 79):

```css

/* Waiting structural changes: dashed, in a group colour, until Refit. */

.pending-range-box {
  fill: color-mix(in srgb, var(--group-0) 6%, transparent);
  stroke: var(--group-0);
  stroke-width: 1.2;
  stroke-dasharray: 5 4;
}

.pending-range-tag {
  fill: var(--sig-weak-bg);
}

.pending-range-label {
  fill: var(--sig-weak-fg);
  font-size: 11px;
  font-weight: 600;
  pointer-events: none;
}

.exposure.waiting {
  fill-opacity: 1;
  stroke-width: 1.3;
  stroke-dasharray: 4 3;
}

.pending-group-ring {
  fill: none;
  stroke-width: 2.2;
  pointer-events: none;
}

.pending-group-bracket-line {
  fill: none;
  stroke-width: 1.4;
  stroke-dasharray: 3 2;
}

.pending-group-label {
  font-size: 10.5px;
  font-weight: 600;
}
```

- [ ] **Step 4: Run tests, expect PASS**

```bash
npm run check:frontend
./.venv/bin/python -m pytest tests/test_editor.py tests/test_lss_editor_style.py -q
./.venv/bin/python -m pytest tests/editor/test_editor_structure_browser.py tests/editor/test_editor_axis_browser.py tests/editor/test_editor_chart_size_browser.py tests/editor/test_editor_workspace_browser.py -m browser --run-browser -q
./.venv/bin/python -m ruff check tests/editor tests/test_editor.py && ./.venv/bin/python -m ruff format --check tests/editor tests/test_editor.py
```

The axis browser tests cover the categorical gutter. With nothing waiting it must
be byte-identical, because `bracketRow` is 0 and `titleHeight` is the default 14.

- [ ] **Step 5: Commit**

```bash
git add src/superglm/editor/app/chart/pending_overlay.js src/superglm/editor/app/chart.js \
  src/superglm/editor/app/chart/geometry.js src/superglm/editor/app/chart/shape_overlay.js \
  src/superglm/editor/app/styles/chart.css tests/editor_frontend/pending_overlay.test.js \
  tests/editor_frontend/chart_geometry.test.js tests/test_editor.py \
  tests/editor/test_editor_structure_browser.py
git commit -m "Editor: waiting groups and ranges drawn dashed on the chart

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task A6: Git-style History with ids, times and notes

**Files:**
- Modify:
  - `src/superglm/editor/app/history.js`, a full rewrite;
  - `src/superglm/editor/app/main.js`: the import (4) and `bindHistory` after
    `openHelp = …` (272);
  - `src/superglm/editor/app/state/actions.js`: `executeStateMutation` (301–309)
    and a helper before `STRUCTURAL_OUTCOME_UNCERTAIN` (133);
  - `src/superglm/editor/app/views/help_content.js`: the History item in "Undo,
    Redo and Revert" (199);
  - `src/superglm/editor/app/styles.css`: the History block (536–649).
- Test (modify):
  - `tests/editor_frontend/history.test.js`, a full rewrite;
  - `tests/editor_frontend/actions.test.js`, with one appended test;
  - `tests/test_editor.py`: 6708–6712, and the route pins A5 added to
    `test_widget_app_shell_contains_drag_editor`;
  - `tests/editor/test_editor_structure_browser.py`: replace the History test and
    its two row helpers.

**Interfaces:**
- Consumes:
  - timeline entries `{kind, id, time, note, status, label, term, operation, redo, hash?}`,
    typed in A5;
  - `POST /note` `{id, note}`, which returns `{ok: true, state}` (S2);
  - A4's `session.step_notes`, `session.pending` and `timeline_payload` for the
    browser test.
- Produces:
  ```js
  // history.js
  export function renderHistory(timeline: TimelineEntry[]|undefined, node: HTMLElement|null): void;
  export function bindHistory(node: HTMLElement, {onNote: (id: string, note: string) => unknown}): {destroy()};
  ```
  The DOM:
  - sections `section.history-section.{waiting,applied,undone}`;
  - rows `li.history-item.{waiting|applied|edit}[.redo][data-step-id]`;
  - in each row, `.history-label`, `.history-meta`, `code.history-id`, `time`,
    `.history-undo-chip`, `button.history-note-edit` (aria-label "Add a note" or
    "Edit note"), `p.history-note`, the editor `.history-note-editor > input.history-note-input`
    (aria-label "Note for this step") and `.history-foot`.

- [ ] **Step 1: Write the failing tests**

Replace `tests/editor_frontend/history.test.js`:

```js
// @ts-nocheck

import assert from "node:assert/strict";
import test from "node:test";

import { renderHistory } from "../../src/superglm/editor/app/history.js";

const at = (hours, minutes) => new Date(2026, 9, 3, hours, minutes).getTime() / 1000;

/** The pane's sections in order, each with its rows' labels, top to bottom. */
function sections(node) {
  return node.innerHTML.split("<section ").slice(1).map((part) => [
    part.match(/class="history-section (\w+)"/)[1],
    [...part.matchAll(/class="history-label">([^<]*)</g)].map((match) => match[1]),
  ]);
}

/** One row's markup, found by its step id. */
function row(node, id) {
  const marker = node.innerHTML.indexOf(`data-step-id="${id}"`);
  const start = node.innerHTML.lastIndexOf("<li", marker);
  return node.innerHTML.slice(start, node.innerHTML.indexOf("</li>", marker));
}

const TIMELINE = [
  { kind: "edit", status: "edit", id: "0a1b2c3", time: at(14, 1), note: null,
    label: "Shift 3 – 5", term: "curve", operation: "shift", redo: false },
  { kind: "structural", status: "applied", id: "1b2c3d4", time: at(14, 3),
    note: "Young-driver tail is noise", label: "Line 18 – 26", term: "DrivAge",
    operation: "shape_range", redo: false },
  { kind: "pending", status: "waiting", id: "2c3d4e5", time: at(14, 5), note: null,
    label: "Collapse B10 + B11", term: "VehBrand", operation: "collapse", redo: false },
  { kind: "marker" },
  { kind: "pending", status: "waiting", id: "3d4e5f6", time: at(14, 6), note: "<b>thin</b>",
    label: "Collapse B13 + B14", term: "VehBrand", operation: "collapse", redo: true },
];

test("waiting changes sit above the applied ones, newest first, and the undone ones below", () => {
  const node = { innerHTML: "" };
  renderHistory(TIMELINE, node);
  assert.deepEqual(sections(node), [
    ["waiting", ["Collapse B10 + B11"]],
    ["applied", ["Line 18 – 26", "Shift 3 – 5"]],
    ["undone", ["Collapse B13 + B14"]],
  ]);
  assert.match(node.innerHTML, /Notes are saved with the exported Python model\./);
});

test("Undo takes the newest step, whichever section it sits in", () => {
  const node = { innerHTML: "" };
  renderHistory(TIMELINE, node);
  assert.equal(node.innerHTML.match(/history-undo-chip/g).length, 1);
  assert.match(row(node, "2c3d4e5"), /Undo takes this/);

  // An edit made after the waiting change is what Undo takes next.
  const editedLast = [
    ...TIMELINE.slice(0, 3),
    { ...TIMELINE[0], id: "4e5f6a7", time: at(14, 7), label: "Smooth 62.7 – 85" },
    { kind: "marker" },
  ];
  renderHistory(editedLast, node);
  assert.match(row(node, "4e5f6a7"), /Undo takes this/);
  assert.doesNotMatch(row(node, "2c3d4e5"), /Undo takes this/);
});

test("each step shows its id, time, term and kind, its note escaped, and a pencil", () => {
  const node = { innerHTML: "" };
  renderHistory(TIMELINE, node);
  const waiting = row(node, "2c3d4e5");
  assert.match(waiting, /class="history-item waiting"/);
  assert.match(waiting, /<code class="history-id">2c3d4e5<\/code>/);
  assert.ok(waiting.includes(
    `<time datetime="${new Date(at(14, 5) * 1000).toISOString()}">14:05</time>`,
  ));
  assert.match(waiting, /class="history-meta">VehBrand · collapse</);
  assert.match(waiting, /aria-label="Add a note"/);

  const applied = row(node, "1b2c3d4");
  assert.match(applied, /class="history-meta">DrivAge · shape</);
  assert.match(applied, /aria-label="Edit note"/);
  assert.match(applied, /<span>Young-driver tail is noise<\/span>/);
  assert.match(row(node, "0a1b2c3"), /class="history-meta">curve · edit</);

  const undone = row(node, "3d4e5f6");
  assert.match(undone, /class="history-item waiting redo"/);
  assert.match(undone, /&lt;b&gt;thin&lt;\/b&gt;/);
});

test("a timeline from before step ids still lists its edits and steps, with no pencil", () => {
  const node = { innerHTML: "" };
  renderHistory([
    { kind: "edit", label: "shift age", hash: "a1b2c3d", n_points: 3, redo: false },
    { kind: "structural", label: "Line 30–45 in age", redo: false },
    { kind: "marker" },
  ], node);
  assert.deepEqual(sections(node), [["applied", ["Line 30–45 in age", "shift age"]]]);
  assert.match(node.innerHTML, /<code class="history-id">a1b2c3d<\/code>/);
  assert.doesNotMatch(node.innerHTML, /history-note-edit/);
});

test("a timeline holding only the marker, or none at all, says nothing happened yet", () => {
  for (const timeline of [[{ kind: "marker" }], undefined]) {
    const node = { innerHTML: "" };
    renderHistory(timeline, node);
    assert.equal(node.innerHTML, '<div class="history-empty">Nothing yet.</div>');
  }
});
```

Append to `tests/editor_frontend/actions.test.js`:

```js
test("a note's answer, the state wrapped as {ok, state}, commits the state it carries", async () => {
  const noted = snapshot(3);
  noted.timeline = [
    {
      kind: "pending", status: "waiting", id: "a1b2c3d", note: "Thin exposure",
      label: "collapse 1 + 2 in age", redo: false
    },
    { kind: "marker" }
  ];
  const store = createEditorStore(createInitialEditorState(snapshot(3)));
  /** @type {number[]} */
  const scheduled = [];
  const actions = createEditorActions({
    store,
    client: {
      postJSON: async (path, payload) => {
        assert.equal(path, "/note");
        assert.deepEqual(payload, { id: "a1b2c3d", note: "Thin exposure" });
        return { ok: true, state: noted };
      },
      getState: async () => { throw new Error("success must not recover through /state"); }
    },
    scheduleVisibleEvidence: (revision) => { scheduled.push(revision); }
  });

  const result = await actions.executeStateMutation({
    name: "note", path: "/note", payload: { id: "a1b2c3d", note: "Thin exposure" }
  });

  assert.deepEqual(result, { ok: true, snapshot: noted });
  assert.strictEqual(store.getState().remote.snapshot, noted);
  assert.deepEqual(scheduled, []);
});
```

In `test_widget_app_shell_contains_drag_editor`, after the lines A5 added:

```python
        assert "/stage" in js
        assert "/refit_pending" in js
```
becomes
```python
        assert "/stage" in js
        assert "/refit_pending" in js
        assert "/note" in js
```

In `tests/test_editor.py`, `test_editor_history_module_renders_the_timeline`
(lines 6708–6712):

```python
    assert "renderHistory" in source
    assert "history-now" in source
    assert "history-chip" in source
    assert "history-hash" in source
```
becomes
```python
    assert "renderHistory" in source
    assert "bindHistory" in source
    assert "history-section" in source
    assert "history-undo-chip" in source
    assert "history-id" in source
```

In `tests/editor/test_editor_structure_browser.py`, delete the helpers
`_history_rows` (lines 86–93) and `_timeline_rows` (added by A5). Add these in
their place:

```python
def _history_sections(page) -> dict[str, list[str]]:
    """The History pane's rows by section, top to bottom."""
    return page.evaluate(
        """() => Object.fromEntries(['waiting', 'applied', 'undone'].map(kind => [
            kind,
            Array.from(
                document.querySelectorAll(`#historyFrame .history-section.${kind} .history-label`),
                node => node.textContent,
            ),
        ]))"""
    )


def _timeline_sections(session) -> dict[str, list[str]]:
    """The sections the session's timeline asks for: newest first, undone in Redo's order."""
    timeline = timeline_payload(session)
    marker = next(i for i, entry in enumerate(timeline) if entry["kind"] == "marker")
    done = timeline[:marker][::-1]
    return {
        "waiting": [entry["label"] for entry in done if entry.get("status") == "waiting"],
        "applied": [entry["label"] for entry in done if entry.get("status") != "waiting"],
        "undone": [entry["label"] for entry in timeline[marker + 1 :]],
    }
```

Replace `test_history_lists_the_session_in_order_and_follows_undo_and_redo` (A5's
version) with:

```python
def test_history_lists_waiting_changes_above_applied_ones_with_ids_and_notes(open_editor_page):
    with open_editor_page() as (page, session):
        _box_select_x(page, 3.0, 5.0)
        with page.expect_response(_posted("/op")):
            page.get_by_role("button", name="Increase selection").click()
        session.stage_structural(
            "shape", "curve", {"lo": 6.0, "hi": 8.0, "degree": 1, "join": "tangent"}
        )
        [step] = session.pending
        _reload_editor(page, "curve")

        page.locator("#historyTab").click()
        page.locator("#historyFrame .history-section.waiting").wait_for()
        assert _history_sections(page) == _timeline_sections(session)
        assert _history_sections(page)["waiting"] == [step.label]
        waiting = page.locator(f'#historyFrame [data-step-id="{step.step_id}"]')
        assert waiting.locator(".history-id").text_content() == step.step_id
        # Undo takes the newest step, which is the waiting one.
        assert page.locator("#historyFrame .history-undo-chip").count() == 1
        assert waiting.locator(".history-undo-chip").count() == 1

        # A note is written in place and saved on Enter.
        note = "Young-driver tail is noise"
        waiting.get_by_role("button", name="Add a note").click()
        field = page.get_by_role("textbox", name="Note for this step")
        field.fill(note)
        with page.expect_request(
            lambda request: request.method == "POST" and urlsplit(request.url).path == "/note"
        ) as note_info:
            field.press("Enter")
        assert note_info.value.post_data_json == {"id": step.step_id, "note": note}
        page.wait_for_function(
            'id => document.querySelector(`[data-step-id="${id}"] .history-note`)',
            arg=step.step_id,
        )
        assert waiting.locator(".history-note").text_content() == note
        assert session.step_notes[step.step_id] == note

        # Escape keeps the note as it was and sends nothing.
        notes: list[object] = []
        page.on(
            "request",
            lambda request: urlsplit(request.url).path == "/note" and notes.append(request),
        )
        waiting.get_by_role("button", name="Edit note").click()
        field.fill("changed my mind")
        field.press("Escape")
        assert notes == []
        assert waiting.locator(".history-note").text_content() == note

        # Undo moves the waiting step under Undone, note and all; Redo puts it back.
        with page.expect_response(_posted("/op")):
            page.keyboard.press("Control+z")
        page.locator("#historyFrame .history-section.undone").wait_for()
        assert _history_sections(page) == _timeline_sections(session)
        undone = page.locator("#historyFrame .history-section.undone .history-note")
        assert undone.text_content() == note
        with page.expect_response(_posted("/op")):
            page.keyboard.press("Control+Shift+z")
        page.locator("#historyFrame .history-section.waiting").wait_for()
        assert _history_sections(page) == _timeline_sections(session)

        # After Refit the step is applied, and its note stays with it.
        with page.expect_response(_posted("/refit_pending")):
            page.keyboard.press("r")
        _settled_after_refit(page)
        page.wait_for_function(
            "() => !document.querySelector('#historyFrame .history-section.waiting')"
        )
        assert _history_sections(page) == _timeline_sections(session)
        assert note in page.locator("#historyFrame .history-section.applied").text_content()
```

- [ ] **Step 2: Run them, expect FAIL**

```bash
node --test tests/editor_frontend/history.test.js tests/editor_frontend/actions.test.js
./.venv/bin/python -m pytest tests/test_editor.py -q -k "history_module_renders or app_shell_contains_drag_editor"
./.venv/bin/python -m pytest tests/editor/test_editor_structure_browser.py -m browser --run-browser -q -k history_lists_waiting
```

The expected failures:
- **Node.** `sections(node)` is `[]`, because the old renderer draws one `<ol>` with
  a "now" marker and no sections.
- **Note action.** The test commits the `{ok, state}` wrapper as the snapshot, so
  `result.snapshot` is not `noted`.
- **Python pins.** `"bindHistory" in source` is false, and so is `"/note" in js`.
- **Browser.** On the branch after A5c, `.history-section.waiting` times out. On
  `origin/master` 155832e8, `stage_structural` does not exist.

- [ ] **Step 3: Implement**

Replace `src/superglm/editor/app/history.js`:

```js
// @ts-check
// The History pane: the session newest first, git-style. The changes waiting
// for Refit come first, then the applied ones, then what Redo would put back.
// Each step shows its short id, its time and an automatic message. A pencil
// writes a note in place, and notes are saved with the exported Python model.

import { escapeHTML } from "./format.js";

/** @typedef {import('./api/contracts.js').TimelineEntry} TimelineEntry */

const PENCIL = '<svg class="history-note-icon" viewBox="0 0 24 24" aria-hidden="true">'
  + '<path d="M4 20h4L19 9l-4-4L4 16z"></path><path d="m14 6 4 4"></path></svg>';
// S2 refuses a longer note with a fixed sentence.
const NOTE_MAX_LENGTH = 2000;
const STATUSES = new Set(["applied", "waiting", "edit"]);

/** @type {Readonly<Record<string, string>>} */
const OPERATION_WORDS = Object.freeze({
  collapse: "collapse",
  collapse_levels: "collapse",
  ungroup: "ungroup",
  ungroup_levels: "ungroup",
  set_reference: "reference",
  shape: "shape",
  shape_range: "shape",
  refit_pending: "refit",
  carry_edits: "edits carried over",
  revert_to_original: "revert",
});

/** The timeline each pane last drew, so a cancelled note can put it back. */
const drawn = new WeakMap();

/**
 * Render the session's timeline: the waiting changes, then the applied ones,
 * newest first, then the undone ones in the order Redo would put them back.
 * The newest step, which Undo takes, says so.
 * @param {TimelineEntry[]|undefined} timeline
 * @param {HTMLElement|null} node
 */
export function renderHistory(timeline, node) {
  if (!node) return;
  const entries = Array.isArray(timeline) ? timeline : [];
  drawn.set(node, entries);
  const marker = entries.findIndex((entry) => entry.kind === "marker");
  const done = marker < 0 ? entries : entries.slice(0, marker);
  const undone = marker < 0 ? [] : entries.slice(marker + 1);
  if (!done.length && !undone.length) {
    node.innerHTML = `<div class="history-empty">Nothing yet.</div>`;
    return;
  }
  const newest = done.at(-1) ?? null;
  const newestFirst = [...done].reverse();
  node.innerHTML = [
    historySection(
      "waiting", "Waiting for refit",
      newestFirst.filter((entry) => entryStatus(entry) === "waiting"), newest
    ),
    historySection(
      "applied", "Applied",
      newestFirst.filter((entry) => entryStatus(entry) !== "waiting"), newest
    ),
    historySection("undone", "Undone", undone, null, "Redo puts back the top one first."),
    `<p class="history-foot">Notes are saved with the exported Python model.</p>`
  ].join("");
}

/**
 * Wire the pane's pencils. A click opens the step's note in place. Enter, or
 * leaving the field, saves it, and Escape keeps the old one. The pane redraws
 * from the next confirmed timeline, and a note that did not save is put back
 * as it was.
 * @param {HTMLElement} node
 * @param {{onNote:(id:string, note:string)=>unknown}} handlers
 * @returns {{destroy:()=>void}}
 */
export function bindHistory(node, { onNote }) {
  /** @param {MouseEvent} event */
  function onClick(event) {
    const button = event.target instanceof Element
      ? event.target.closest(".history-note-edit")
      : null;
    const item = button?.closest("[data-step-id]");
    if (!(item instanceof HTMLElement) || !node.contains(item)) return;
    openNoteEditor(item);
  }

  /** @param {HTMLElement} item */
  function openNoteEditor(item) {
    const id = item.dataset.stepId || "";
    /** @type {TimelineEntry[]} */
    const entries = drawn.get(node) ?? [];
    const current = entries.find((entry) => entry.id === id)?.note ?? "";
    const doc = item.ownerDocument;
    const editor = doc.createElement("label");
    editor.className = "history-note-editor";
    editor.innerHTML = PENCIL;
    const input = doc.createElement("input");
    input.type = "text";
    input.className = "history-note-input";
    input.value = current;
    input.maxLength = NOTE_MAX_LENGTH;
    input.placeholder = "Why? Add a note to this step";
    input.setAttribute("aria-label", "Note for this step");
    editor.append(input);
    const shown = item.querySelector(".history-note");
    if (shown) shown.replaceWith(editor);
    else item.append(editor);
    input.focus();

    let settled = false;
    /** @param {boolean} save */
    const finish = (save) => {
      if (settled) return;
      settled = true;
      const note = input.value.trim();
      if (!save || note === current) {
        renderHistory(drawn.get(node), node);
        return;
      }
      void Promise.resolve(onNote(id, note)).finally(() => {
        if (editor.isConnected) renderHistory(drawn.get(node), node);
      });
    };
    input.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        finish(true);
      } else if (event.key === "Escape") {
        // The note's own Escape: it must not also close the inspector or a popover.
        event.preventDefault();
        event.stopPropagation();
        finish(false);
      }
    });
    input.addEventListener("blur", () => finish(true));
  }

  node.addEventListener("click", onClick);
  return Object.freeze({
    destroy() {
      node.removeEventListener("click", onClick);
    },
  });
}

/**
 * @param {string} kind @param {string} title @param {TimelineEntry[]} entries
 * @param {TimelineEntry|null} newest @param {string} [hint]
 */
function historySection(kind, title, entries, newest, hint = "") {
  if (!entries.length) return "";
  const items = entries.map((entry) => historyItem(entry, entry === newest)).join("");
  const hintHTML = hint ? `<p class="history-section-hint">${hint}</p>` : "";
  return `<section class="history-section ${kind}" aria-label="${title}">`
    + `<h3 class="history-section-title">${title}</h3>${hintHTML}`
    + `<ol class="history-list">${items}</ol></section>`;
}

/** @param {TimelineEntry} entry @param {boolean} takesUndo */
function historyItem(entry, takesUndo) {
  const status = entryStatus(entry);
  const id = typeof entry.id === "string" ? entry.id : "";
  const shownId = id || (typeof entry.hash === "string" ? entry.hash : "");
  const note = typeof entry.note === "string" ? entry.note : "";
  const classes = ["history-item", status, entry.redo ? "redo" : ""].filter(Boolean).join(" ");
  const idAttribute = id ? ` data-step-id="${escapeHTML(id)}"` : "";
  const chip = takesUndo ? `<span class="history-undo-chip">Undo takes this</span>` : "";
  const stamp = `${shownId ? `<code class="history-id">${escapeHTML(shownId)}</code>` : ""}`
    + timeTag(entry.time);
  const noteHTML = note
    ? `<p class="history-note">${PENCIL}<span>${escapeHTML(note)}</span></p>`
    : "";
  return `<li class="${classes}"${idAttribute}><div class="history-row">`
    + `<div class="history-body"><div class="history-label">${escapeHTML(entry.label || "")}</div>`
    + `<div class="history-meta">${escapeHTML(entryMeta(entry, status))}</div></div>`
    + `${chip}<div class="history-stamp">${stamp}</div>${id ? noteButton(note) : ""}</div>`
    + `${noteHTML}</li>`;
}

/**
 * A timeline from before step ids has no status: its edits are edits and
 * its steps applied ones.
 * @param {TimelineEntry} entry @returns {"applied"|"waiting"|"edit"}
 */
function entryStatus(entry) {
  if (entry.status && STATUSES.has(entry.status)) return entry.status;
  return entry.kind === "edit" ? "edit" : "applied";
}

/** @param {TimelineEntry} entry @param {string} status */
function entryMeta(entry, status) {
  const operation = String(entry.operation ?? "");
  const word = status === "edit"
    ? "edit"
    : OPERATION_WORDS[operation] ?? (operation.replaceAll("_", " ") || "step");
  return [entry.term, word].filter(Boolean).join(" · ");
}

/** @param {unknown} seconds */
function timeTag(seconds) {
  if (typeof seconds !== "number" || !Number.isFinite(seconds)) return "";
  const at = new Date(seconds * 1000);
  const clock = `${String(at.getHours()).padStart(2, "0")}:${String(at.getMinutes()).padStart(2, "0")}`;
  return `<time datetime="${at.toISOString()}">${clock}</time>`;
}

/** @param {string} note */
function noteButton(note) {
  const label = note ? "Edit note" : "Add a note";
  return `<button class="history-note-edit icon-button" type="button" aria-label="${label}"`
    + ` data-popover-title="${label}"`
    + ` data-popover-body="Say why. Notes are saved with the exported Python model.">`
    + `${PENCIL}</button>`;
}
```

In `src/superglm/editor/app/state/actions.js`, before `STRUCTURAL_OUTCOME_UNCERTAIN`
(line 133):

```js
const STRUCTURAL_OUTCOME_UNCERTAIN =
  "The model change outcome is uncertain. The operation was not retried.";
```
becomes
```js
/**
 * A state mutation answers with the snapshot itself, or, as /note does, with
 * `{ok: true, state}`.
 * @param {unknown} response
 * @returns {EditorSnapshot}
 */
function stateOf(response) {
  if (isRecord(response) && response.ok === true && isEditorSnapshot(response.state)) {
    return /** @type {EditorSnapshot} */ (response.state);
  }
  return /** @type {EditorSnapshot} */ (response);
}

const STRUCTURAL_OUTCOME_UNCERTAIN =
  "The model change outcome is uncertain. The operation was not retried.";
```
and in `executeStateMutation` (lines 301–309):

```js
    /** @type {EditorSnapshot} */
    let snapshot;
    try {
      snapshot = /** @type {EditorSnapshot} */ (
        await client.postJSON(path, descriptor.payload)
      );
    } catch (value) {
      return recoverMutation(value, name, descriptor);
    }
```
becomes
```js
    /** @type {EditorSnapshot} */
    let snapshot;
    try {
      snapshot = stateOf(await client.postJSON(path, descriptor.payload));
    } catch (value) {
      return recoverMutation(value, name, descriptor);
    }
```

In `src/superglm/editor/app/main.js`, the import (line 4):

```js
import { renderHistory } from "./history.js";
```
becomes
```js
import { bindHistory, renderHistory } from "./history.js";
```
and after `openHelp = () => inspector.open("help");` (line 272):

```js
openHelp = () => inspector.open("help");
```
becomes
```js
openHelp = () => inspector.open("help");

// A History note is saved through the action controller like an edit, so a
// failed save gets the same alert and Retry.
bindHistory(historyFrame, {
  onNote: (id, note) => executeStateMutation("/note", { id, note })
});
```

In `src/superglm/editor/app/views/help_content.js`, "Undo, Redo and Revert"
(line 199):

```js
      "History, in the inspector, lists every edit and step of the session in order. Undo takes the entry above the current-position line; the muted entries below it are what Redo would put back.",
```
becomes
```js
      "History, in the inspector, lists the session newest first: the changes waiting for refit, then the applied ones. Undo takes the entry marked Undo takes this; the muted entries under Undone are what Redo would put back.",
      "Each step has a short id and the time it was made. The pencil adds a note saying why; notes are saved with the exported Python model.",
```

In `src/superglm/editor/app/styles.css`, replace the History block. It starts at
the line `/* History: the session as a timeline, edits and steps on one rail. */`
(536) and runs through the end of the `.history-meta { … }` rule (649, just
before `/* The compact coefficient summary. */`). The replacement:

```css
/* History: newest first, the changes waiting for refit above the applied
   ones and the undone ones below, each step with its id, time and note. */
.history-frame {
  min-height: 0;
  flex: 1 1 auto;
  overflow-y: auto;
  border-top: 1px solid var(--border);
  padding-top: 2px;
}
.history-empty {
  color: var(--muted);
  font-size: 12px;
}
.history-section-title {
  display: flex;
  align-items: center;
  gap: 6px;
  margin: 8px 2px 4px;
  color: var(--muted);
  font-size: 12px;
  font-weight: 600;
}
.history-section-title::before {
  content: "";
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: currentColor;
}
.history-section.waiting .history-section-title {
  color: var(--group-0);
}
.history-section-hint {
  margin: -2px 2px 4px;
  color: var(--muted);
  font-size: 11px;
}
.history-list {
  display: grid;
  gap: 5px;
  list-style: none;
  margin: 0;
  padding: 0;
}
.history-item {
  padding: 8px 9px;
  border: 1px solid var(--hairline);
  border-radius: var(--radius-md);
  background: var(--surface);
}
.history-item.waiting {
  border: 1px dashed var(--group-0);
}
.history-item.redo {
  opacity: 0.55;
}
.history-row {
  display: flex;
  align-items: center;
  gap: 8px;
}
.history-body {
  flex: 1 1 auto;
  min-width: 0;
}
.history-label {
  font-size: 13px;
  font-weight: 600;
  overflow-wrap: anywhere;
}
.history-meta {
  color: var(--muted);
  font-size: 11.5px;
  line-height: 1.35;
  overflow-wrap: anywhere;
}
.history-undo-chip {
  flex: 0 0 auto;
  padding: 0 8px;
  border-radius: 999px;
  background: var(--blue-soft);
  color: var(--blue);
  font-size: 11.5px;
  line-height: 20px;
  white-space: nowrap;
}
.history-stamp {
  display: grid;
  flex: 0 0 auto;
  justify-items: end;
  color: var(--muted);
  font-family: var(--font-mono);
  font-size: 10.5px;
  line-height: 1.35;
}
.history-id {
  font: inherit;
  letter-spacing: 0.01em;
}
.history-note-edit.icon-button {
  flex: 0 0 auto;
  width: 24px;
  height: 24px;
  border-color: transparent;
  background: transparent;
  color: var(--muted);
  opacity: 0.55;
}
.history-item:hover .history-note-edit,
.history-note-edit:focus-visible {
  opacity: 1;
}
.history-note-icon {
  flex: 0 0 auto;
  width: 13px;
  height: 13px;
  fill: none;
  stroke: currentColor;
  stroke-width: 1.6;
  stroke-linecap: round;
  stroke-linejoin: round;
}
.history-note {
  display: flex;
  align-items: center;
  gap: 6px;
  margin: 5px 0 0;
  color: var(--text);
  font-size: 12.5px;
  font-style: italic;
  overflow-wrap: anywhere;
}
.history-note .history-note-icon {
  color: var(--muted);
}
.history-note-editor {
  display: flex;
  align-items: center;
  gap: 6px;
  height: 28px;
  margin-top: 6px;
  padding: 0 8px;
  border: 1px solid var(--blue);
  border-radius: var(--radius-md);
  background: var(--surface);
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--blue) 12%, transparent);
  color: var(--muted);
}
.history-note-input[type="text"] {
  flex: 1 1 auto;
  min-width: 0;
  height: 24px;
  padding: 0;
  border: 0;
  background: transparent;
  color: var(--text);
  font-size: 12.5px;
}
.history-note-input:focus-visible {
  outline: none;
}
.history-foot {
  margin: 8px 2px 0;
  color: var(--muted);
  font-size: 11.5px;
}
```

- [ ] **Step 4: Run tests, expect PASS**

```bash
npm run check:frontend
./.venv/bin/python -m pytest tests/test_editor.py tests/test_lss_editor_style.py -q
./.venv/bin/python -m pytest tests/editor/test_editor_structure_browser.py tests/editor/test_editor_workspace_browser.py tests/editor/test_editor_refit_browser.py -m browser --run-browser -q
./.venv/bin/python -m ruff check tests/editor tests/test_editor.py && ./.venv/bin/python -m ruff format --check tests/editor tests/test_editor.py
```

The workspace browser tests at 890–1063 hold `#historyFrame > *` steady across
selection-only renders. The new markup re-renders only when the timeline changes,
so the identity checks still hold.

- [ ] **Step 5: Commit**

```bash
git add src/superglm/editor/app/history.js src/superglm/editor/app/main.js \
  src/superglm/editor/app/state/actions.js src/superglm/editor/app/views/help_content.js \
  src/superglm/editor/app/styles.css tests/editor_frontend/history.test.js \
  tests/editor_frontend/actions.test.js tests/test_editor.py \
  tests/editor/test_editor_structure_browser.py
git commit -m "Editor: git-style History with ids, times and notes

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

## Risks and open questions for this section

1. **Two Undos with "Refit after every structural change" on.** The spec says
   "stage then refit" (§4 A), so one change is two timeline entries, the waiting
   step and the Refit, and taking it back fully takes two Undos.
   - S2 keeps the legacy routes (`/collapse_levels` and the others). They stage
     and refit in one call, list the change once, and keep the per-operation
     refusal sentences.
   - Switching `runStructuralChange` to those routes when the setting is on would
     give one Undo. That would contradict §4 A's wording, so it is Max's call.
2. **Grouped ordered terms open Expanded by default.** "Groups shown as" now
   decides for every term. With the default Expanded, an ordered term with groups
   opens expanded; today it opens Collapsed. No current test pins the old
   behaviour.
3. **No chart mark for a waiting ungroup or set-reference.** These show only in the
   Refit count, the amber dot, the status line and History. The spec draws only
   groups and ranges.
4. **I2 removes F1's bridge.** S7 replaces `mountThemeControl` with
   `mountThemeSwitch({settings})`. With it, I2 deletes:
   - the `// Until I2:` block in main.js;
   - the `setChoice` test in `theme.test.js`;
   - F1's `test_follow_the_browser_is_the_theme_choice`, which clicks the
     "Theme: Auto" icon.

   S7's old snippets for the theme import and the mount call are master's text,
   so they must be rebased onto F1's.
5. **Typecheck not run here.** `node_modules/` is absent in this worktree.
   - Every edit pair in this section was applied in memory to the current
     sources: 103 pairs, all applied, plus the new modules. All 17 node test
     files then passed through `data:` URL imports, except `interactions.test.js`,
     which the harness cannot load and none of these tasks touch.
   - The HTML and Python-pin assertions were checked against the edited text.
   - `tsc` was not run. The implementer runs `npm run check:frontend` after
     `npm ci`.


## Section S4: Selection gestures (C) and inspector (E)

Tasks: **C1** gestures, **C2** shapes span the selection (D4), **E1** inspector
search, **E2** filters, follows-the-chart and header. They run in this order,
one after another: all four touch `main.js` or `views/help_content.js`. F1
runs before them and touches different lines. E2 reads A4's top-level
`pending` when it exists and works without it.

Everything below was checked on a scratch copy of the worktree. Its app and
tests are byte-identical to `155832e8`
(`git diff 155832e8 HEAD -- src tests package.json jsconfig.json` is empty).
- Each new Node test was run against unmodified `155832e8` code and failed
  there, then passed with the code below.
- Each browser test was run the same way, with the asset loader pointed at the
  patched app.
- `tsc -p jsconfig.json` passed on the patched tree.
- The full `tests/editor` browser suite passed with all four tasks applied
  (75 passed).
- `tests/test_editor_browser.py` also passed (10 passed).
- `tests/test_editor.py` passed with all four tasks applied (its string pins
  included).

### Contract amendments

1. **Selection anchor.** The store field is
   `view.selectionAnchor: {term: string, index: number} | null`, not
   `number | null`.
   - `index` is a **source** index (a position in `term.x` / `term.levels`), not
     a display index.
   - It is a source index because display indices change when the Groups
     control switches between Expanded and Collapsed. A source index keeps its
     meaning, and is mapped through `svg._scale.displayToSourceIndices` when
     used.
   - It carries `term` because the active term changes along several paths:
     feature-list click, Undo/Redo, recovery, `commitRemote`. An anchor that
     names another term is simply ignored, so no reset is needed on any of
     those paths.
2. **Interaction context.** `bindInteractions(context)` gains
   `context.selectionAnchor(): {term, index} | null` and
   `context.setSelectionAnchor(anchor): void`. `main.js` backs both with the
   store.
3. **Test-only hook.** With `?test=1`, `window.__superglmTest.mutationStatus()`
   returns `state.request.mutation.status`.
   - A selection posts without the busy overlay. A click made while that
     mutation is still running is skipped by design (`executeSelectionMutation`).
   - Without the hook, 1 of 12 repeated browser runs lost its second click. With
     the hook, 12 of 12 passed.
4. **State payload, per term:** `edited: boolean`, equal to
   `name in session.edited_terms()`. The Edited filter reads it. The
   `TermPayload` typedef gains `edited?: boolean`.
5. **DOM.**
   - New ids: `#summarySearchCount`, `#summaryFilter`, `#summaryHeader`,
     `#summaryModelChips`, `#summaryTiles`.
   - `#refitOffset`, `#reprofileTweedie` and `#reprofileNb2` move from
     `.summary-controls` into `#summaryHeader`. Their ids and listeners are
     unchanged.
   - `#summarySearch` is as in the shared contract.
6. **Modules.**
   - New `app/views/summary_view.js` (exports listed per task).
   - `summary.js` exports `applySummaryView(nodes, {follow})`.
   - The summary node bundle (`summaryNodes()` in `main.js`) gains optional
     `summarySearchCount`, `summaryHeader`, `summaryModelChips`, `summaryTiles`
     and `summaryView()`.
7. **Waiting chip ownership.** E2 owns the inspector's per-term "N waiting" chip
   and the Waiting filter (spec §4 A, "Other surfaces"). A5 should not add a
   second one.

### Shared verification notes

- `npm run check:frontend` needs `node_modules/`. Run `npm ci` once if it is
  absent; `package-lock.json` is committed and holds dev dependencies only.
- `tests/editor` may run with `-n`. Never put `tests/test_editor_browser.py` in
  the same `-n` run:
  - two sync-Playwright instances end up in one worker, and every test in that
    file errors with "Sync API inside the asyncio loop";
  - the plan header's serial command is fine (85 passed on `155832e8`).
  This section runs the two files separately.

---

### Task C1: Click anchor, Shift-click span, Shift-drag pan, Ctrl/Cmd toggle on any term, click on the curve

**Files:**
- Modify `src/superglm/editor/app/interactions.js`. Edits, with line numbers on
  `155832e8`:
  - interaction state, lines 1–13;
  - pan start, 17–32;
  - Ctrl/Cmd toggle, 110–126;
  - brush start, 135–136;
  - pointermove, 148–152;
  - pointerup pan, 198–202;
  - order-drag release, 245–252;
  - click/box release, 261–278;
  - `hasActiveInteraction`, `cancelActiveInteraction` and
    `isModifierLevelSelection`, 318–353;
  - `beginOrderDrag`, 489–490;
  - `beginPan`, 678–686.
- Modify `src/superglm/editor/app/state/store.js:26-27`.
- Modify `src/superglm/editor/app/api/contracts.js:191-192`.
- Modify `src/superglm/editor/app/main.js`:
  - after `interactionMode`, 394–396;
  - test hook, 702–704;
  - `bindInteractions` call, 1431–1442.
- Modify `src/superglm/editor/app/views/help_content.js`:
  - `TOOL_HELP.select`, 12–16;
  - `HELP_SECTIONS` "Modes", 154–157.
- Test `tests/editor_frontend/interactions.test.js`: harness lines 5–51, plus
  appended tests.
- Test `tests/editor/test_editor_workspace_browser.py`: insert before line 541.

**Interfaces:**
- Consumes:
  - `context.actions.executeSelectionMutation({term, indices}) -> Promise<ActionResult>`;
  - `actions.patchView(patch)`;
  - `svg._scale`: `{sx, sy, x, y, margin, innerW, innerH, displayIsCollapsed, displayToSourceIndices}`, set by `chart.js:269-275`.
- Produces:
  - `context.selectionAnchor(): {term:string, index:number}|null`;
  - `context.setSelectionAnchor(anchor: {term:string, index:number}|null): void`;
  - `EditorViewState.selectionAnchor`;
  - `window.__superglmTest.mutationStatus(): "idle"|"running"|"error"`;
  - browser-test helper `_click_selects(page, point, modifiers=(), timeout=30000) -> None`. E2 reuses it.

The behaviour, on master and after:

| Gesture | `155832e8` | After C1 |
|---|---|---|
| Shift + press | Pans at once, on a point too | Becomes a pan only past the 3-unit click slop. Released inside the slop in Select mode, it selects from the anchor to the point. |
| Ctrl/Cmd on a point | Toggles on pointerdown; only when `term.levels` is non-empty | Decided on release on any term: inside the slop it toggles one point and sets the anchor; past the slop it is a box |
| Plain click on a point | Replaces the selection | Same, and sets the anchor |
| Plain click within 12 px of the curve, between points | Nothing | Picks the nearest display index by x |
| Click on empty space | Nothing | Nothing |
| Middle button | Pans at once | Same |
| Build-animation guard (`main.js:1420-1429`) | — | Kept, untouched |

- [ ] **Step 1: Write the failing tests**

`tests/editor_frontend/interactions.test.js`. The two existing Ctrl-click
tests now need a release, a pointer position, and a context that holds an
anchor. Make four edits to the existing harness.

Edit 1. Replace:
```js
const interactionsModulePath = "../../src/superglm/editor/app/interactions.js";
const { bindInteractions } = await import(interactionsModulePath);
```
with:
```js
const interactionsModulePath = "../../src/superglm/editor/app/interactions.js";
const { bindInteractions } = await import(interactionsModulePath);

// A press draws a brush rectangle; the gestures only need it to exist.
globalThis.document = {
  createElementNS: () => ({ setAttribute() {}, remove() {} })
};
```

Edit 2. Replace:
```js
  const svg = {
    _scale: { displayIsCollapsed, displayToSourceIndices },
    addEventListener(name, listener) {
      listeners.set(name, listener);
    },
    removeEventListener() {},
  };
```
with:
```js
  const svg = {
    _scale: { displayIsCollapsed, displayToSourceIndices },
    viewBox: { baseVal: { x: 0, y: 0, width: 100, height: 100 } },
    addEventListener(name, listener) {
      listeners.set(name, listener);
    },
    removeEventListener() {},
    setPointerCapture() {},
    appendChild() {},
    getScreenCTM() { return null; },
    getBoundingClientRect() { return { left: 0, top: 0, width: 100, height: 100 }; },
  };
```

Edit 3. Replace:
```js
    currentSelection: () => new Set(),
    mode: () => "select",
    selectedTerm: () => "feature",
    actions: {
      async executeSelectionMutation(payload) {
        mutations.push(payload);
      },
    },
```
with:
```js
    currentSelection: () => new Set(),
    selectionAnchor: () => null,
    setSelectionAnchor() {},
    mode: () => "select",
    selectedTerm: () => "feature",
    actions: {
      async executeSelectionMutation(payload) {
        mutations.push(payload);
      },
    },
```

Edit 4. Replace:
```js
    async ctrlClick(displayIndex) {
      await listeners.get("pointerdown")({
        button: 0,
        ctrlKey: true,
        metaKey: false,
        shiftKey: false,
        target: { dataset: { index: String(displayIndex) } },
        preventDefault() {},
      });
    },
```
with:
```js
    async ctrlClick(displayIndex) {
      const event = {
        button: 0,
        pointerId: 1,
        clientX: 10,
        clientY: 10,
        ctrlKey: true,
        metaKey: false,
        shiftKey: false,
        target: { dataset: { index: String(displayIndex) } },
        preventDefault() {},
      };
      await listeners.get("pointerdown")(event);
      await listeners.get("pointerup")(event);
    },
```

Then append to the end of the file:
```js

// SVG units are client pixels here: data x maps to 100 + 10x and relativity
// to 300 - 100y, on a plot 200 wide and 300 tall.
const SPLINE_X = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
const SPLINE_Y = [1, 2.5, 0.5, 2, 1, 2.8, 0.2, 1.5, 1, 2];

function gestureHarness({
  x = SPLINE_X,
  y = SPLINE_Y,
  levels = null,
  selection = [],
  anchor = null,
  displayToSourceIndices = null,
}) {
  const listeners = new Map();
  const mutations = [];
  const zooms = [];
  const state = { selection: new Set(selection), anchor };
  const scale = {
    sx: (value) => 100 + 10 * value,
    sy: (value) => 300 - 100 * value,
    x,
    y,
    xMin: 0, xMax: 20, yMin: 0, yMax: 3,
    baseXMin: 0, baseXMax: 20, baseYMin: 0, baseYMax: 3,
    margin: { left: 100, top: 0 },
    innerW: 200,
    innerH: 300,
    displayIsCollapsed: displayToSourceIndices !== null,
    displayToSourceIndices: displayToSourceIndices ?? x.map((_, i) => [i]),
  };
  const svg = {
    _scale: scale,
    viewBox: { baseVal: { x: 0, y: 0, width: 400, height: 400 } },
    addEventListener(name, listener) { listeners.set(name, listener); },
    removeEventListener() {},
    setPointerCapture() {},
    appendChild() {},
    getScreenCTM() { return null; },
    getBoundingClientRect() { return { left: 0, top: 0, width: 400, height: 400 }; },
  };
  const term = { term_type: levels ? "categorical" : "spline", x, y, levels };
  const context = {
    svg,
    currentTerm: () => term,
    currentSelection: () => new Set(state.selection),
    selectionAnchor: () => state.anchor,
    setSelectionAnchor(next) { state.anchor = next; },
    mode: () => "select",
    selectedTerm: () => "age",
    setZoom(_term, range) { zooms.push(range); },
    clearZoom() {},
    setPreviewTerm() {},
    clearPreviewTerm() {},
    actions: {
      async executeSelectionMutation(payload) {
        mutations.push(payload);
        state.selection = new Set(payload.indices);
        return { ok: true };
      },
    },
  };
  bindInteractions(context);

  const pointAt = (i) => ({ x: scale.sx(x[i]), y: scale.sy(y[i]) });
  // Press, move once, release: on a point when `index` is given, else on
  // whatever lies under `at` (the curve, or empty plot).
  async function gesture({ index = null, at, to = at, keys = {} }) {
    const base = {
      button: 0,
      pointerId: 1,
      shiftKey: false,
      ctrlKey: false,
      metaKey: false,
      ...keys,
      target: { dataset: index === null ? {} : { index: String(index) } },
      preventDefault() {},
    };
    await listeners.get("pointerdown")({ ...base, clientX: at.x, clientY: at.y });
    listeners.get("pointermove")({ ...base, clientX: to.x, clientY: to.y });
    await listeners.get("pointerup")({ ...base, clientX: to.x, clientY: to.y });
  }

  return {
    mutations,
    zooms,
    state,
    pointAt,
    click: (i, keys = {}) => gesture({ index: i, at: pointAt(i), keys }),
    clickAt: (at, keys = {}) => gesture({ at, keys }),
    drag: (i, to, keys = {}) => gesture({ index: i, at: pointAt(i), to, keys }),
  };
}

test("a click on a point selects it alone and makes it the anchor", async () => {
  const chart = gestureHarness({ selection: [7, 8] });

  await chart.click(3);

  assert.deepEqual(chart.mutations, [{ term: "age", indices: [3] }]);
  assert.deepEqual(chart.state.anchor, { term: "age", index: 3 });
});

test("Shift-click selects every point between the anchor and the point by x, whatever their heights", async () => {
  const chart = gestureHarness({});

  await chart.click(2);
  await chart.click(6, { shiftKey: true });
  assert.deepEqual(chart.mutations.at(-1), { term: "age", indices: [2, 3, 4, 5, 6] });

  // The anchor stays put, so another Shift-click spans from it again.
  await chart.click(0, { shiftKey: true });
  assert.deepEqual(chart.mutations.at(-1), { term: "age", indices: [0, 1, 2] });
  assert.deepEqual(chart.state.anchor, { term: "age", index: 2 });
  assert.deepEqual(chart.zooms, []);
});

test("without an anchor on this term a Shift-click selects the point and anchors it", async () => {
  const chart = gestureHarness({ anchor: { term: "other", index: 1 } });

  await chart.click(5, { shiftKey: true });

  assert.deepEqual(chart.mutations, [{ term: "age", indices: [5] }]);
  assert.deepEqual(chart.state.anchor, { term: "age", index: 5 });
});

test("a Shift press pans only once it moves past the click slop", async () => {
  const chart = gestureHarness({});
  const from = chart.pointAt(4);

  await chart.drag(4, { x: from.x + 30, y: from.y }, { shiftKey: true });
  assert.deepEqual(chart.mutations, []);
  assert.ok(chart.zooms.length > 0);

  const panned = chart.zooms.length;
  await chart.drag(4, { x: from.x + 2, y: from.y - 2 }, { shiftKey: true });
  assert.equal(chart.zooms.length, panned);
  assert.deepEqual(chart.mutations, [{ term: "age", indices: [4] }]);
});

test("Ctrl/Cmd-click toggles one point on a spline and moves the anchor", async () => {
  const chart = gestureHarness({ selection: [1, 2] });

  await chart.click(5, { ctrlKey: true });
  assert.deepEqual(chart.mutations.at(-1), { term: "age", indices: [1, 2, 5] });

  await chart.click(2, { metaKey: true });
  assert.deepEqual(chart.mutations.at(-1), { term: "age", indices: [1, 5] });
  assert.deepEqual(chart.state.anchor, { term: "age", index: 2 });
});

test("a click on the curve between points snaps to the nearest point by x; off it nothing changes", async () => {
  // Points 3, 4 and 5 sit at (130, 190), (140, 180) and (150, 180).
  const chart = gestureHarness({ y: [1, 1, 1, 1.1, 1.2, 1.2, 1, 1, 1, 1] });

  // On the line from 3 to 4, nearer 4 by x.
  await chart.clickAt({ x: 137, y: 183 });
  assert.deepEqual(chart.mutations.at(-1), { term: "age", indices: [4] });
  assert.deepEqual(chart.state.anchor, { term: "age", index: 4 });

  // 8 px above the line from 4 to 5, nearer 5 by x.
  await chart.clickAt({ x: 147, y: 172 });
  assert.deepEqual(chart.mutations.at(-1), { term: "age", indices: [5] });

  // 16 px above that line, then empty plot: neither changes the selection.
  await chart.clickAt({ x: 145, y: 164 });
  await chart.clickAt({ x: 250, y: 20 });
  assert.equal(chart.mutations.length, 2);
});

test("on a collapsed display the anchor is a source level and the span selects every source level in it", async () => {
  // Display point 1 is the group b + c; the anchor, source level c, lies in it.
  const chart = gestureHarness({
    x: [0, 1, 2, 3],
    y: [1, 1.2, 0.8, 1.1],
    levels: ["a", "b", "c", "d", "e"],
    anchor: { term: "age", index: 2 },
    displayToSourceIndices: [[0], [1, 2], [3], [4]],
  });

  await chart.click(3, { shiftKey: true });

  assert.deepEqual(chart.mutations, [{ term: "age", indices: [1, 2, 3, 4] }]);
});
```

`tests/editor/test_editor_workspace_browser.py`. Insert the following
immediately before
`def test_select_all_is_incremental_bounded_and_keeps_bounds_behind_points(open_editor_page):`
(line 541), keeping two blank lines on each side:
```python
def _click_selects(page, point, modifiers=(), timeout=30000) -> None:
    """Click, wait for its /select, and let the page settle before the next click.

    A selection posts without the busy overlay, and a click that lands while
    one is still running is skipped, so the next click waits for it.
    """
    with page.expect_response(
        lambda response: is_select_request(response.request), timeout=timeout
    ):
        point.click(modifiers=list(modifiers))
    page.wait_for_function("() => window.__superglmTest?.mutationStatus?.() !== 'running'")


def test_click_shift_click_selects_a_span_and_ctrl_click_toggles_on_a_spline(open_editor_page):
    # Thirty points draw every marker, so the selection palette steps around them.
    with open_editor_page(n_points=30) as (page, session):
        select_chart_tool(page, "Select")

        def point(index: int):
            return page.locator(f'#chart circle.point[data-index="{index}"]')

        _click_selects(page, point(8))
        _click_selects(page, point(14), modifiers=["Shift"], timeout=5000)
        assert session.selection("curve").tolist() == list(range(8, 15))
        # The Shift press did not pan: the chart still shows its whole x range.
        assert page.evaluate(
            "() => { const s = document.querySelector('#chart')._scale;"
            " return s.xMin === s.baseXMin && s.xMax === s.baseXMax; }"
        )

        _click_selects(page, point(11), modifiers=["Control"])
        assert session.selection("curve").tolist() == [8, 9, 10, 12, 13, 14]
        _click_selects(page, point(20), modifiers=["Control"])
        assert session.selection("curve").tolist() == [8, 9, 10, 12, 13, 14, 20]
        # The last Ctrl-click is the anchor: Shift-click 17 spans 17 to 20.
        _click_selects(page, point(17), modifiers=["Shift"])
        assert session.selection("curve").tolist() == [17, 18, 19, 20]
```

- [ ] **Step 2: Run them, expect FAIL**

```bash
node --test tests/editor_frontend/interactions.test.js
./.venv/bin/python -m pytest tests/editor/test_editor_workspace_browser.py -m browser --run-browser -q -k shift_click_selects_a_span
```

Expected on `155832e8`: 7 Node tests fail, the 5 existing ones pass. The
failures:

| Test | Why it fails |
|---|---|
| a click on a point … anchor | `chart.state.anchor` stays `null`: nothing sets an anchor |
| Shift-click … by x | The Shift press calls `beginPan`, so no `/select` is sent. `mutations.at(-1)` is still `[2]`. |
| without an anchor … | `mutations` is `[]` |
| a Shift press pans only … | The in-slop Shift press pans: `zooms.length` is 2, expected 1 |
| Ctrl/Cmd-click toggles … spline | The `term.levels` gate sends the press to the brush, so the selection is replaced: `[5]`, expected `[1, 2, 5]` |
| a click on the curve … | No `data-index`, so no mutation |
| collapsed display … | Pans; `mutations` is `[]` |

The browser test fails at the Shift-click with
`TimeoutError: Timeout 5000ms exceeded while waiting for event "response"`.

- [ ] **Step 3: Implement**

`src/superglm/editor/app/interactions.js`. These edits are the whole change.
They keep the strings that `tests/test_editor.py:6461-6464` and `6493-6494`
pin: `sourceIndicesForDisplayIndex(ices)`, `valuesForSourceIndices`,
`if (!scale.displayIsCollapsed) return [index]`, `scale.displayIsCollapsed`
and `if (scale.displayIsCollapsed) return false`.

Edit C1.1. Replace the first line `export function bindInteractions(context) {` with:
```js
// A press that moves no further than this many SVG units in x and in y is a
// click; past it, a drag. The chart draws at its own CSS-pixel size, so an SVG
// unit is a pixel.
const CLICK_SLOP = 3;
// A click this close to the drawn curve, between its points, picks the point
// nearest by x.
const CURVE_SNAP_DISTANCE = 12;

export function bindInteractions(context) {
```

Edit C1.2. Replace:
```js
    orderDrag: null,
    pendingClickIndex: null
  };
```
with:
```js
    orderDrag: null,
    pendingClickIndex: null,
    toggleClick: false,
    shiftPress: null
  };
```

Edit C1.3. Replace:
```js
    const activeTerm = context.currentTerm();
    if (!activeTerm) return;
    if ((event.shiftKey || event.button === 1) && beginPan(context, interaction, event)) {
      event.preventDefault();
      interaction.pendingClickIndex = null;
      interaction.dragStart = null;
      interaction.pointDrag = null;
      interaction.controlDrag = null;
      clearBoxZoom(interaction);
      clearOrderDropPreview(interaction);
      interaction.orderDrag = null;
      svg.setPointerCapture(event.pointerId);
      return;
    }
    const index = event.target && event.target.dataset ? event.target.dataset.index : undefined;
```
with:
```js
    const activeTerm = context.currentTerm();
    if (!activeTerm) return;
    const index = event.target && event.target.dataset ? event.target.dataset.index : undefined;
    // A middle press pans at once. A Shift press waits: once it moves past the
    // click slop it pans, and released inside it, it is a Shift-click.
    if (event.button === 1 && beginPan(context, interaction, svgPoint(context, event))) {
      event.preventDefault();
      resetPress(interaction);
      svg.setPointerCapture(event.pointerId);
      return;
    }
    if (event.shiftKey && svg._scale) {
      event.preventDefault();
      resetPress(interaction);
      interaction.shiftPress = {
        start: svgPoint(context, event),
        index: index !== undefined ? Number(index) : null
      };
      svg.setPointerCapture(event.pointerId);
      return;
    }
```

Edit C1.4. Replace:
```js
    if (index !== undefined && isModifierLevelSelection(event, context.currentTerm())) {
      event.preventDefault();
      interaction.pendingClickIndex = null;
      interaction.dragStart = null;
      interaction.brush = null;
      clearBoxZoom(interaction);
      clearOrderDropPreview(interaction);
      interaction.orderDrag = null;
      const source = sourceIndicesForDisplayIndex(context, Number(index));
      const indices = toggleSourceSelection(context.currentSelection(), source);
      await context.actions.executeSelectionMutation({
        term: context.selectedTerm(),
        indices
      });
      return;
    }
    if (index !== undefined && beginOrderDrag(context, interaction, event, Number(index))) {
```
with:
```js
    // Ctrl/Cmd marks the press a toggle; it is decided as a click or a box on
    // release, like any other press.
    const toggle = isToggleClick(event);
    if (toggle) event.preventDefault();
    if (!toggle && index !== undefined && beginOrderDrag(context, interaction, event, Number(index))) {
```

Edit C1.5. Replace:
```js
    interaction.pendingClickIndex = index !== undefined ? Number(index) : null;
    interaction.dragStart = svgPoint(context, event);
```
with:
```js
    interaction.pendingClickIndex = index !== undefined ? Number(index) : null;
    interaction.toggleClick = toggle;
    interaction.dragStart = svgPoint(context, event);
```

Edit C1.6. Replace:
```js
  function onPointerMove(event) {
    if (interaction.panDrag) {
      panZoomView(context, interaction, svgPoint(context, event));
      return;
    }
```
with:
```js
  function onPointerMove(event) {
    if (interaction.panDrag) {
      panZoomView(context, interaction, svgPoint(context, event));
      return;
    }
    if (interaction.shiftPress) {
      const press = interaction.shiftPress;
      const point = svgPoint(context, event);
      if (!movedPastClickSlop(press.start, point)) return;
      interaction.shiftPress = null;
      if (beginPan(context, interaction, press.start)) panZoomView(context, interaction, point);
      return;
    }
```

Edit C1.7. Replace:
```js
  async function onPointerUp(event) {
    if (interaction.panDrag) {
      interaction.panDrag = null;
      return;
    }
```
with:
```js
  async function onPointerUp(event) {
    if (interaction.panDrag) {
      interaction.panDrag = null;
      return;
    }
    if (interaction.shiftPress) {
      const press = interaction.shiftPress;
      interaction.shiftPress = null;
      if (context.mode() !== "select") return;
      const target = press.index ?? curveIndexNear(context, press.start);
      if (target !== null) await shiftClickPoint(context, target);
      return;
    }
```

Edit C1.8. Replace:
```js
      if (drag.active && drag.targetIndex !== null) {
        await context.actions.executeStateMutation({
          name: "reorder_levels",
          path: "/reorder_levels",
          payload: { term: context.selectedTerm(), target_index: drag.targetIndex }
        });
      }
      return;
```
with:
```js
      if (drag.active && drag.targetIndex !== null) {
        await context.actions.executeStateMutation({
          name: "reorder_levels",
          path: "/reorder_levels",
          payload: { term: context.selectedTerm(), target_index: drag.targetIndex }
        });
      } else if (!drag.active) {
        // A click on a selected level keeps the selection for dragging and
        // anchors the next Shift-click there.
        context.setSelectionAnchor({ term: context.selectedTerm(), index: drag.pressed });
      }
      return;
```

Edit C1.9. Replace:
```js
    if (!interaction.dragStart) return;
    const point = svgPoint(context, event);
    const moved = Math.abs(point.x - interaction.dragStart.x) > 3 ||
      Math.abs(point.y - interaction.dragStart.y) > 3;
    const displayIndices = moved ? indicesInBox(context, interaction.dragStart, point) : (
      interaction.pendingClickIndex === null ? null : [interaction.pendingClickIndex]
    );
    if (interaction.brush) interaction.brush.remove();
    interaction.brush = null;
    interaction.dragStart = null;
    interaction.pendingClickIndex = null;
    if (displayIndices === null) return;
    const indices = sourceIndicesForDisplayIndices(context, displayIndices);
    await context.actions.executeSelectionMutation({
      term: context.selectedTerm(),
      indices
    });
  }
```
with:
```js
    if (!interaction.dragStart) return;
    const start = interaction.dragStart;
    const point = svgPoint(context, event);
    const pressed = interaction.pendingClickIndex;
    const toggle = interaction.toggleClick;
    if (interaction.brush) interaction.brush.remove();
    interaction.brush = null;
    interaction.dragStart = null;
    interaction.pendingClickIndex = null;
    interaction.toggleClick = false;
    if (movedPastClickSlop(start, point)) {
      await context.actions.executeSelectionMutation({
        term: context.selectedTerm(),
        indices: sourceIndicesForDisplayIndices(context, indicesInBox(context, start, point))
      });
      return;
    }
    // A click on empty space changes nothing.
    const target = pressed ?? curveIndexNear(context, start);
    if (target !== null) await clickPoint(context, target, toggle);
  }
```

Edit C1.10. Replace:
```js
    interaction.orderDrag ||
    interaction.pendingClickIndex !== null
  );
```
with:
```js
    interaction.orderDrag ||
    interaction.shiftPress ||
    interaction.pendingClickIndex !== null
  );
```

Edit C1.11. This replaces the dead modifier gate and adds the gesture helpers.
Replace:
```js
  interaction.orderDrag = null;
  interaction.pendingClickIndex = null;
  if (hadPreview) context.clearPreviewTerm();
}

function isModifierLevelSelection(event, term) {
  return Boolean(
    term &&
    Array.isArray(term.levels) &&
    term.levels.length > 0 &&
    (event.ctrlKey || event.metaKey)
  );
}
```
with:
```js
  interaction.orderDrag = null;
  interaction.pendingClickIndex = null;
  interaction.toggleClick = false;
  interaction.shiftPress = null;
  if (hadPreview) context.clearPreviewTerm();
}

// Forget a press in progress before a pan or a Shift press takes over.
function resetPress(interaction) {
  interaction.pendingClickIndex = null;
  interaction.toggleClick = false;
  interaction.shiftPress = null;
  interaction.dragStart = null;
  interaction.pointDrag = null;
  interaction.controlDrag = null;
  clearBoxZoom(interaction);
  clearOrderDropPreview(interaction);
  interaction.orderDrag = null;
}

// Ctrl or Cmd adds or removes one point, on any term.
function isToggleClick(event) {
  return Boolean(event.ctrlKey || event.metaKey);
}

function movedPastClickSlop(start, point) {
  return Math.abs(point.x - start.x) > CLICK_SLOP || Math.abs(point.y - start.y) > CLICK_SLOP;
}

// A click on one display point selects it alone, or with Ctrl/Cmd toggles it,
// and either way makes it the anchor of the next Shift-click.
async function clickPoint(context, displayIndex, toggle) {
  const term = context.selectedTerm();
  const source = sourceIndicesForDisplayIndices(context, [displayIndex]);
  context.setSelectionAnchor({ term, index: source[0] });
  await context.actions.executeSelectionMutation({
    term,
    indices: toggle ? toggleSourceSelection(context.currentSelection(), source) : source
  });
}

// A Shift-click selects every display point between the anchor and this one
// by x, whatever their heights, and leaves the anchor where it is. With no
// anchor on this term it is a plain click.
async function shiftClickPoint(context, displayIndex) {
  const term = context.selectedTerm();
  const anchor = anchorDisplayIndex(context, term);
  if (anchor === null) {
    await clickPoint(context, displayIndex, false);
    return;
  }
  await context.actions.executeSelectionMutation({
    term,
    indices: sourceIndicesForDisplayIndices(
      context,
      displayIndicesBetween(context, anchor, displayIndex)
    )
  });
}

// The anchor is a source index, so it survives a switch between the expanded
// and collapsed displays; this finds the display point that shows it.
function anchorDisplayIndex(context, term) {
  const anchor = context.selectionAnchor();
  if (!anchor || anchor.term !== term) return null;
  const scale = context.svg._scale || {};
  const count = Array.isArray(scale.x) ? scale.x.length : 0;
  if (!scale.displayIsCollapsed) return anchor.index < count ? anchor.index : null;
  const mapping = Array.isArray(scale.displayToSourceIndices) ? scale.displayToSourceIndices : [];
  const display = mapping.findIndex(
    (source) => Array.isArray(source) && source.map(Number).includes(anchor.index)
  );
  return display >= 0 ? display : null;
}

function displayIndicesBetween(context, from, to) {
  const x = context.svg._scale.x;
  const lo = Math.min(Number(x[from]), Number(x[to]));
  const hi = Math.max(Number(x[from]), Number(x[to]));
  const indices = [];
  for (let i = 0; i < x.length; i++) {
    const value = Number(x[i]);
    if (value >= lo && value <= hi) indices.push(i);
  }
  return indices;
}

// The display point a click on the drawn curve means: the nearest by x, when
// the click is inside the plot and within CURVE_SNAP_DISTANCE of the curve.
function curveIndexNear(context, point) {
  const scale = context.svg._scale;
  if (!scale || !Array.isArray(scale.x) || !scale.x.length) return null;
  const { sx, sy, x, y, margin, innerW, innerH } = scale;
  if (
    point.x < margin.left || point.x > margin.left + innerW ||
    point.y < margin.top || point.y > margin.top + innerH
  ) {
    return null;
  }
  let nearest = 0;
  let distance = Math.hypot(point.x - sx(x[0]), point.y - sy(y[0]));
  for (let i = 1; i < x.length; i++) {
    distance = Math.min(
      distance,
      segmentDistance(point, sx(x[i - 1]), sy(y[i - 1]), sx(x[i]), sy(y[i]))
    );
    if (Math.abs(sx(x[i]) - point.x) < Math.abs(sx(x[nearest]) - point.x)) nearest = i;
  }
  return distance <= CURVE_SNAP_DISTANCE ? nearest : null;
}

function segmentDistance(point, x0, y0, x1, y1) {
  const dx = x1 - x0;
  const dy = y1 - y0;
  const length2 = dx * dx + dy * dy;
  const t = length2 > 0
    ? Math.max(0, Math.min(1, ((point.x - x0) * dx + (point.y - y0) * dy) / length2))
    : 0;
  return Math.hypot(point.x - (x0 + t * dx), point.y - (y0 + t * dy));
}
```

Edit C1.12. In `beginOrderDrag`, replace:
```js
  interaction.orderDrag = {
    start: svgPoint(context, event),
    indices: Array.from(selection).sort((a, b) => a - b),
```
with:
```js
  interaction.orderDrag = {
    start: svgPoint(context, event),
    pressed: index,
    indices: Array.from(selection).sort((a, b) => a - b),
```

Edit C1.13. Replace:
```js
function beginPan(context, interaction, event) {
  // Panning is intentionally chorded behind Shift or middle click so ordinary
  // drag-select remains the default interaction.
  if (!context.svg._scale) return false;
  const scale = context.svg._scale;
  interaction.panDrag = {
    start: svgPoint(context, event),
```
with:
```js
function beginPan(context, interaction, start) {
  // Panning is intentionally chorded behind Shift or middle click so ordinary
  // drag-select remains the default interaction. `start` is where the press
  // began, so a Shift press that turns into a pan keeps the ground it covered.
  if (!context.svg._scale) return false;
  const scale = context.svg._scale;
  interaction.panDrag = {
    start,
```

`src/superglm/editor/app/state/store.js`. Replace:
```js
      preview: null,
      selectionPreview: null
    },
```
with:
```js
      preview: null,
      selectionPreview: null,
      selectionAnchor: null
    },
```

`src/superglm/editor/app/api/contracts.js` (`EditorViewState`). Replace:
```js
 * @property {{term:string, indices:number[]}|null} selectionPreview
 */
```
with:
```js
 * @property {{term:string, indices:number[]}|null} selectionPreview
 * @property {{term:string, index:number}|null} selectionAnchor the point the next
 *   Shift-click spans from: a source index of `term`, set by a click or a Ctrl/Cmd-click
 */
```

`src/superglm/editor/app/main.js`. Three edits.

Edit 1. Replace:
```js
function interactionMode() {
  return store.getState().view.mode;
}
```
with:
```js
function interactionMode() {
  return store.getState().view.mode;
}

function selectionAnchor() {
  return store.getState().view.selectionAnchor;
}

function setSelectionAnchor(anchor) {
  actions.patchView({ selectionAnchor: anchor });
}
```

Edit 2. Replace `  window.__superglmTest = Object.freeze({ setAppBusy });` with the
following. If another task has already extended this object, add only the
`mutationStatus` member.
```js
  window.__superglmTest = Object.freeze({
    setAppBusy,
    // A selection posts without the busy overlay; tests wait on this before
    // the next click, which a running mutation would skip.
    mutationStatus: () => store.getState().request.mutation.status
  });
```

Edit 3. In the `bindInteractions({ ... })` call, replace:
```js
  currentSelection,
  setPreviewTerm: setInteractionPreview,
```
with:
```js
  currentSelection,
  selectionAnchor,
  setSelectionAnchor,
  setPreviewTerm: setInteractionPreview,
```

`src/superglm/editor/app/views/help_content.js`. Two edits.

Edit 1. Replace:
```js
    title: "Select",
    body: "Click points or drag a box to select curve values.",
```
with:
```js
    title: "Select",
    body:
      "Click a point or drag a box to select curve values. Shift-click selects every point from the last one clicked; Ctrl/Cmd-click adds or removes one.",
```

Edit 2. Replace:
```js
  Object.freeze({
    title: "Modes",
    keys: Object.freeze(["select", "move", "zoom", "handles"]),
  }),
```
with:
```js
  Object.freeze({
    title: "Modes",
    keys: Object.freeze(["select", "move", "zoom", "handles"]),
  }),
  Object.freeze({
    title: "Selecting points",
    items: Object.freeze([
      "Click a point to select it. Shift-click another to select every point between the two by position along the axis, whatever their heights.",
      "Ctrl/Cmd-click adds or removes one point, on any term. The point clicked last, with or without Ctrl/Cmd, is where the next Shift-click starts.",
      "A click on the curve between points selects the nearest point; a click on empty space changes nothing.",
      "Drag a box to select the points inside it. Shift-drag pans instead.",
    ]),
  }),
```

The Navigation item "Shift-drag or middle-drag: pan" (`help_content.js:187`) is
still true, so it stays.

- [ ] **Step 4: Run the tests, expect PASS**

```bash
node --test tests/editor_frontend/interactions.test.js
npm run check:frontend
./.venv/bin/python -m pytest tests/test_editor.py -q -k "interactions_map_group_display or blocks_collapsed_reorder"
./.venv/bin/python -m pytest tests/editor -m browser --run-browser -q -n 6
./.venv/bin/python -m pytest tests/test_editor_browser.py -m browser --run-browser -q
```

- The full `tests/editor` run covers the existing Ctrl-click tests:
  - `test_selection_menu_does_not_block_adjacent_modifier_selection` (workspace file, line 534);
  - `test_structural_refit_commits_atomically_before_held_metrics` (refit file, line 166).
  Playwright's `click(modifiers=["Control"])` releases inside the slop, so both
  stay green.
- It also covers the empty-box no-op
  (`test_selection_noop_empty_box_preserves_chart_without_posting`) and the
  collapsed-display source mapping tests.
- **Mutation check, done:**
  - `CURVE_SNAP_DISTANCE = 20` fails the curve test: the 16 px press snaps;
  - `CURVE_SNAP_DISTANCE = 5` fails it too: the 8 px press does not snap;
  - so the test pins 12 px between 8 and 16.

- [ ] **Step 5: Commit**

```bash
git add src/superglm/editor/app/interactions.js src/superglm/editor/app/state/store.js \
  src/superglm/editor/app/api/contracts.js src/superglm/editor/app/main.js \
  src/superglm/editor/app/views/help_content.js tests/editor_frontend/interactions.test.js \
  tests/editor/test_editor_workspace_browser.py
git commit -m "Editor: click anchor, Shift-click span and Ctrl/Cmd toggle on any term

A plain or Ctrl/Cmd click sets a client-side anchor (a source index, so it
survives the collapsed display). Shift-click selects every display point
between the anchor and the point by x; a Shift press pans only once it moves
past the 3-unit click slop. Ctrl/Cmd toggles one point on any term, decided on
release. A click within 12 px of the curve picks the nearest point by x.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task C2: A shape covers the span from the first to the last selected point (D4)

This is separate from C1 because a reviewer can reject D4 on its own: today a
selection with a gap disables the shape icons.

**Files:**
- Modify `src/superglm/editor/app/shapes.js`:
  - line 19, the `NOT_CONTIGUOUS` export;
  - lines 26–37, `selectedRun`;
  - lines 39–47, the `shapeRangeForSelection` doc and call;
  - lines 85–86, `disabledReason`.
- Modify `src/superglm/editor/app/views/help_content.js:165`, the "Shaped ranges"
  first item.
- Test `tests/editor_frontend/shapes.test.js`:
  - import, lines 5–14;
  - tests at 59–62, 87–99 and 119–123.

**Interfaces:**
- Consumes the selection set from `currentSelection()`.
- Produces:
  - `shapeRangeForSelection(term, selectedIndices)` returns
    `{lo, hi}` from `min..max` of any non-empty selection;
  - `shapeButtonState` no longer refuses a selection with a gap;
  - the export `NOT_CONTIGUOUS` is removed. Its only users were `shapes.js` and
    `shapes.test.js`; checked with `grep -rn NOT_CONTIGUOUS src tests`.
- The backend is unchanged. `/shape_range` already takes edges `lo`/`hi` and fits
  the shape to the data between them. A band special inside the span is refused
  with the existing `SPECIAL_LEVEL` message.

- [ ] **Step 1: Write the failing tests**

In `tests/editor_frontend/shapes.test.js`, make four edits.

Edit 1. Replace:
```js
  GROUPED_EDGE,
  NOT_CONTIGUOUS,
  SPECIAL_LEVEL,
```
with:
```js
  GROUPED_EDGE,
  SPECIAL_LEVEL,
```

Edit 2. Replace:
```js
test("a gap in the selection gives no range, nor does an empty one", () => {
  assert.equal(shapeRangeForSelection(numeric, new Set([1, 3])), null);
  assert.equal(shapeRangeForSelection(numeric, new Set()), null);
});
```
with:
```js
test("a selection with gaps spans its first to its last point; an empty one names no range", () => {
  // A Ctrl-clicked selection: the shape covers everything between its ends.
  assert.deepEqual(shapeRangeForSelection(numeric, new Set([4, 1, 3])), { lo: 20, hi: 40 });
  assert.deepEqual(shapeRangeForSelection(ordered(), new Set([4, 1])), { lo: "B2", hi: "B5" });
  assert.equal(shapeRangeForSelection(numeric, new Set()), null);
});
```

Edit 3. Replace:
```js
test("a broken run or a single point disables the icons and says so", () => {
  assert.deepEqual(shapeButtonState(numeric, new Set([1, 3])), {
    visible: true,
    enabled: false,
    reason: NOT_CONTIGUOUS
  });
```
with:
```js
test("a selection with gaps enables the icons; a single point says why not", () => {
  assert.deepEqual(shapeButtonState(numeric, new Set([1, 3])), {
    visible: true,
    enabled: true,
    reason: null
  });
```

Edit 4. Replace:
```js
  assert.equal(shapeButtonState(term, new Set([4, 5]), 1).reason, SPECIAL_LEVEL);
  assert.equal(shapeButtonState(term, new Set([3, 4]), 1).enabled, true);
});
```
with:
```js
  assert.equal(shapeButtonState(term, new Set([4, 5]), 1).reason, SPECIAL_LEVEL);
  assert.equal(shapeButtonState(term, new Set([3, 4]), 1).enabled, true);
  // The span is what is shaped, so a special between two clicked bands counts.
  const inside = { ...ordered(), shape: { ...AVAILABLE, specials: ["B4"] } };
  assert.equal(shapeButtonState(inside, new Set([2, 5]), 1).reason, SPECIAL_LEVEL);
});
```

- [ ] **Step 2: Run them, expect FAIL**

```bash
node --test tests/editor_frontend/shapes.test.js
```

Expected on `155832e8`: 3 tests fail, 15 pass.
- `{4,1,3}` returns `null`, not `{lo: 20, hi: 40}`.
- `{1,3}` is disabled with "Select a continuous run of points."
- The special inside `{2,5}` reports the contiguity message, not `SPECIAL_LEVEL`.

- [ ] **Step 3: Implement**

`src/superglm/editor/app/shapes.js`. Four edits.

Edit 1. Delete the line:
```js
export const NOT_CONTIGUOUS = "Select a continuous run of points.";
```

Edit 2. Replace:
```js
/**
 * The selection as a run of source indices: null unless it is non-empty and
 * consecutive, which is contiguous in the displayed order on the numeric and
 * ordered axes that can take a shape.
 * @param {Set<number>} selectedIndices @returns {[number, number]|null}
 */
function selectedRun(selectedIndices) {
  if (!selectedIndices.size) return null;
  const lo = Math.min(...selectedIndices);
  const hi = Math.max(...selectedIndices);
  return hi - lo + 1 === selectedIndices.size ? [lo, hi] : null;
}

/**
 * The range a contiguous selection names: its x extent on a numeric term,
 * its first and last band labels on an ordered one. A selected collapsed
 * group is selected by all of its source bands, so it contributes them all.
 * @param {TermPayload} term @param {Set<number>} selectedIndices
 * @returns {{lo:number|string, hi:number|string}|null}
 */
export function shapeRangeForSelection(term, selectedIndices) {
  const run = selectedRun(selectedIndices);
```
with:
```js
/**
 * The selection's span as source indices: its first and last selected point,
 * with any gap between them, since a shape covers the whole span and is fitted
 * to the data across it (a Ctrl-clicked selection shapes from end to end).
 * Source order is the displayed order on the numeric and ordered axes that can
 * take a shape. Null when nothing is selected.
 * @param {Set<number>} selectedIndices @returns {[number, number]|null}
 */
function selectedSpan(selectedIndices) {
  if (!selectedIndices.size) return null;
  return [Math.min(...selectedIndices), Math.max(...selectedIndices)];
}

/**
 * The range a selection names, from its first selected point to its last: its
 * x extent on a numeric term, its first and last band labels on an ordered
 * one. A selected collapsed group is selected by all of its source bands, so
 * it contributes them all.
 * @param {TermPayload} term @param {Set<number>} selectedIndices
 * @returns {{lo:number|string, hi:number|string}|null}
 */
export function shapeRangeForSelection(term, selectedIndices) {
  const run = selectedSpan(selectedIndices);
```

Edit 3. Replace:
```js
  const run = selectedRun(selectedIndices);
  if (!run) return NOT_CONTIGUOUS;
```
with:
```js
  const run = selectedSpan(selectedIndices);
  if (!run) return TOO_FEW_POINTS;
```

Edit 4, in `help_content.js`. Replace:
```js
      "Select a run of points or bands on a spline term, then choose Flat, Line, Quadratic or Cubic. That range is pinned to the shape; the rest of the term stays the fitted smooth.",
```
with:
```js
      "Select points or bands on a spline term, then choose Flat, Line, Quadratic or Cubic. The range runs from the first selected point to the last, gaps included, and is pinned to the shape fitted to the data across it; the rest of the term stays the fitted smooth.",
```

`docs/tutorials/edit-a-model-in-the-browser.md:101` still says "select a
continuous run of points". That file belongs to Z1; hand this line to Z1.

- [ ] **Step 4: Run the tests, expect PASS**

```bash
node --test tests/editor_frontend/shapes.test.js
npm run check:frontend
./.venv/bin/python -m pytest tests/editor/test_editor_structure_browser.py -m browser --run-browser -q
```

The structure browser file's shape tests drive contiguous runs and back-to-back
ranges, and stay green.

- [ ] **Step 5: Commit**

```bash
git add src/superglm/editor/app/shapes.js src/superglm/editor/app/views/help_content.js \
  tests/editor_frontend/shapes.test.js
git commit -m "Editor: a shape covers the span from the first to the last selected point

A Ctrl-clicked selection with gaps no longer disables the shape icons: the
range runs from its first to its last point and is fitted to the data across
it (D4). A special band inside the span is still refused.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task E1: Inspector search

**Files:**
- Create `src/superglm/editor/app/views/summary_view.js`.
- Modify `src/superglm/editor/app/summary.js`:
  - imports, lines 1–3;
  - payload cache, line 10;
  - `renderSummary`, 222–242;
  - `renderCompactSummary`, 495–500 and 539;
  - `renderSummaryRows` and `summaryRowGroup`, 566–585;
  - `renderSummaryRow`, 601–611.
- Modify `src/superglm/editor/app/main.js`:
  - import from `./summary.js`, lines 30–40;
  - `./views/theme.js` import, line 61;
  - after line 167;
  - `summaryNodes`, 433–459;
  - after `bindFeatureList`, 1466–1471.
- Modify `src/superglm/editor/app/index.html:449-451`.
- Modify `src/superglm/editor/app/styles.css:511-514`.
- Modify `src/superglm/editor/app/views/help_content.js:179-182`.
- Test: create `tests/editor_frontend/summary_view.test.js`.
- Test: modify `tests/editor_frontend/summary.test.js`, the import at 10–21 and a
  new test before line 263.
- Test: `tests/editor/test_editor_workspace_browser.py`, insert before line 1712.

**Interfaces:**
- Consumes:
  - compact summary rows (`summaries.py:201-241`): `name`, `group`, `level_group`,
    `kind`, `edf`, `p_value`, `sig_class`, `sig_code`;
  - `snapshot.terms` keys.
- Measured on `155832e8`: a `Categorical` over integers `[1, 2, 10]` gives rows
  `bonus[1]`, `bonus[2]`, `bonus[10]`, in model order. A `Polynomial` prints
  under the group `"x_poly P(2)"`, which is not the term name. That is why rows
  map back to editor terms.
- Produces, from `views/summary_view.js`:
  - `DEFAULT_SUMMARY_VIEW`;
  - `summaryRowTerm(row, termNames): string`;
  - `summaryLevelLabel(row, term): string`;
  - `highlightMatches(text, query): string` (HTML);
  - `highlightRowName(name, term, query): string`;
  - `summaryViewModel(rows, view): {sections, termCount, rowCount}`;
  - `summaryCountText(model, query): string`;
  - `bindSummarySearch(search, onQuery): {destroy}`.
- Also produces:
  - from `summary.js`, `applySummaryView(nodes): void`;
  - optional node-bundle fields `summarySearchCount`, `summaryView(): SummaryView`;
  - rows `tr.summary-row[data-term]` and headers `tr.summary-group-row[data-term]`.

Design, so the reviewer can check it against spec §4 E:
- The search is part of the summary markup. Every `renderSummary` call reads
  `nodes.summaryView()`, so it holds after each re-render: refits, profile runs,
  level-display changes and evidence refreshes (`main.js:1152`, `summary.js`).
- Rows outside the search stay in the DOM with `hidden`.
- A keystroke rewrites only `.summary-table tbody`. The "Full summary" iframe is
  never searched, and an open `<details>` stays open.
- The markup cache (`summary.test.js:226`) is refreshed to the full markup that
  view would write, so an identical later render is still a no-op.
- `applySummaryView` acts only on a frame that still holds its last render. This
  keeps a typed query from resurrecting a payload while `main.js:1154` has
  emptied the frame for the other level display.

- [ ] **Step 1: Write the failing tests**

Create `tests/editor_frontend/summary_view.test.js`:
```js
// @ts-nocheck
import assert from "node:assert/strict";
import test from "node:test";

const summaryViewModulePath = "../../src/superglm/editor/app/views/summary_view.js";
const {
  bindSummarySearch,
  highlightMatches,
  highlightRowName,
  summaryCountText,
  summaryRowTerm,
  summaryViewModel
} = await import(summaryViewModulePath);

// Compact rows as Python sends them, in model order.
const ROWS = [
  { name: "Intercept", group: "" },
  { name: "VehAge", group: "VehAge", kind: "spline" },
  { name: "VehBrand[B1]", group: "VehBrand" },
  { name: "VehBrand[B10]", group: "VehBrand" },
  { name: "VehBrand[B2]", group: "VehBrand" },
  // Integer-typed levels arrive as their text, in the model's order 1, 2, 10.
  { name: "bonus[1]", group: "bonus" },
  { name: "bonus[2]", group: "bonus" },
  { name: "bonus[10]", group: "bonus" },
  // A polynomial prints under a decorated group.
  { name: "x_poly[P1]", group: "x_poly P(2)" },
  { name: "x_poly[P2]", group: "x_poly P(2)" }
];
const TERMS = ["VehAge", "VehBrand", "bonus", "x_poly"];

function search(query) {
  return summaryViewModel(ROWS, { query, termNames: TERMS });
}

/** The row names a search shows, in the order shown. */
function shown(query) {
  return search(query).sections.flatMap((section) => section.rows
    .filter((entry) => !entry.hidden)
    .map((entry) => ROWS[entry.index].name));
}

test("each row belongs to its editor term, a decorated polynomial group included", () => {
  assert.equal(summaryRowTerm(ROWS[8], TERMS), "x_poly");
  assert.equal(summaryRowTerm(ROWS[3], TERMS), "VehBrand");
  assert.equal(summaryRowTerm(ROWS[0], TERMS), "Intercept");
  assert.equal(summaryRowTerm({ name: "x:z", group: "x:z" }, TERMS), "x:z");
  const sections = search("").sections;
  assert.deepEqual(sections.map((section) => [section.term, section.label]), [
    ["Intercept", "Intercept"],
    ["VehAge", "VehAge"],
    ["VehBrand", "VehBrand"],
    ["bonus", "bonus"],
    ["x_poly", "x_poly P(2)"]
  ]);
});

test("an empty search shows every row", () => {
  assert.deepEqual(shown(""), ROWS.map((row) => row.name));
  assert.equal(summaryCountText(search(""), ""), "");
});

test("a term-name match keeps all its rows and the intercept leaves while searching", () => {
  assert.deepEqual(shown("veh"), ["VehAge", "VehBrand[B1]", "VehBrand[B10]", "VehBrand[B2]"]);
  assert.equal(summaryCountText(search("veh"), "veh"), "2 terms · 4 rows");
  // Model order is kept, not string order (1, 10, 2).
  assert.deepEqual(shown("BONUS"), ["bonus[1]", "bonus[2]", "bonus[10]"]);
});

test("numeric-looking levels match as text: 1 finds 1 and 10, 10 finds only 10", () => {
  assert.deepEqual(shown("1"), ["VehBrand[B1]", "VehBrand[B10]", "bonus[1]", "bonus[10]", "x_poly[P1]"]);
  assert.equal(summaryCountText(search("1"), "1"), "3 terms · 5 rows");
  assert.deepEqual(shown("10"), ["VehBrand[B10]", "bonus[10]"]);
  // The decorated group's "(2)" is not a level: only the level P2 matches 2.
  assert.deepEqual(shown("2"), ["VehBrand[B2]", "bonus[2]", "x_poly[P2]"]);
  assert.equal(summaryCountText(search(" 10 "), " 10 "), "2 terms · 2 rows");
});

test("a search with no match says so, and its text is taken literally", () => {
  assert.equal(summaryCountText(search("zzz"), "zzz"), "No terms match.");
  assert.deepEqual(shown("("), []);
  assert.deepEqual(shown("."), []);
  assert.equal(search("zzz").sections.every((section) => section.hidden), true);
});

test("matches are marked in the term and the level, and the rest is escaped", () => {
  assert.equal(highlightMatches("<B1>b1", "b1"), "&lt;<mark>B1</mark>&gt;<mark>b1</mark>");
  assert.equal(highlightRowName("bonus[10]", "bonus", "1"), "bonus[<mark>1</mark>0]");
  assert.equal(highlightRowName("VehBrand[B1]", "VehBrand", "veh"), "<mark>Veh</mark>Brand[B1]");
  assert.equal(highlightRowName("VehBrand[B1]", "VehBrand", ""), "VehBrand[B1]");
});

test("the search box reports each input and Escape clears it without closing the inspector", () => {
  const listeners = new Map();
  const search = {
    value: "",
    addEventListener(name, listener) { listeners.set(name, listener); },
    removeEventListener(name) { listeners.delete(name); }
  };
  const queries = [];
  const binding = bindSummarySearch(search, (query) => queries.push(query));
  const key = (name) => {
    const event = {
      key: name,
      prevented: false,
      stopped: false,
      preventDefault() { this.prevented = true; },
      stopPropagation() { this.stopped = true; }
    };
    listeners.get("keydown")(event);
    return event;
  };

  search.value = "veh";
  listeners.get("input")();
  const escape = key("Escape");
  assert.deepEqual(queries, ["veh", ""]);
  assert.equal(search.value, "");
  assert.equal(escape.prevented && escape.stopped, true);

  // An empty box lets Escape through, so a narrow inspector still closes.
  const passed = key("Escape");
  assert.equal(passed.prevented || passed.stopped, false);
  assert.deepEqual(queries, ["veh", ""]);

  binding.destroy();
  assert.equal(listeners.size, 0);
});
```

`tests/editor_frontend/summary.test.js`. This file is type-checked, so the
fixture carries JSDoc.

Edit 1. In the dynamic import list, replace:
```js
const {
  collapseTransition,
  refreshSummary,
```
with:
```js
const {
  applySummaryView,
  collapseTransition,
  refreshSummary,
```

Edit 2. Insert immediately before
`test("expanded compact summary shows group indicators without a membership legend", () => {`:
```js
/** A compact summary with integer-typed levels, as Python prints them. */
function bonusSummary() {
  /**
   * @param {string} name @param {string} group @param {string} sigClass
   * @param {Record<string, unknown>} [extra]
   */
  const row = (name, group, sigClass, extra = {}) => ({
    name, group, kind: "coef", sig_class: sigClass, ...extra
  });
  return {
    available: true,
    label: "Summary",
    html: "",
    compact: {
      model: {},
      level_display: "expanded",
      has_level_groups: false,
      level_groups: [],
      rows: [
        row("region[A]", "region", "sig-reference", { kind: "reference" }),
        row("bonus[1]", "bonus", "sig-reference", { kind: "reference" }),
        row("bonus[2]", "bonus", "sig-none", { coef: 0.1, p_value: 0.2 }),
        row("bonus[10]", "bonus", "sig-medium", { coef: 0.3, p_value: 0.004, sig_code: "**" })
      ]
    }
  };
}

test("the inspector search is reapplied on every render and marks its matches", () => {
  const view = { query: "1", termNames: ["region", "bonus"] };
  const nodes = {
    ...compactSummaryNodes(),
    summarySearchCount: { textContent: "" },
    summaryView: () => view
  };
  const payload = bonusSummary();

  renderSummary(payload, nodes);
  const first = nodes.summaryFrame.innerHTML;
  assert.match(first, /<tr class="summary-row sig-reference" data-term="bonus">/);
  assert.match(first, /bonus\[<mark>1<\/mark>\]/);
  assert.match(first, /bonus\[<mark>1<\/mark>0\]/);
  assert.match(first, /<tr class="summary-row sig-none" data-term="bonus" hidden>/);
  assert.match(first, /<tr class="summary-group-row[^"]*" data-term="region"[^>]* hidden>/);
  assert.equal(nodes.summarySearchCount.textContent, "1 term · 2 rows");

  // A refit sends a new payload; the frame is rebuilt and the search holds.
  const refit = bonusSummary();
  refit.html = "<p>Refitted</p>";
  renderSummary(refit, nodes);
  assert.notEqual(nodes.summaryFrame.innerHTML, first);
  assert.match(nodes.summaryFrame.innerHTML, /<tr class="summary-row sig-none" data-term="bonus" hidden>/);

  // A new query redraws the last payload without fetching it again.
  view.query = "10";
  applySummaryView(nodes);
  assert.match(nodes.summaryFrame.innerHTML, /bonus\[<mark>10<\/mark>\]/);
  assert.match(nodes.summaryFrame.innerHTML, /<tr class="summary-row sig-reference" data-term="bonus" hidden>/);
  assert.equal(nodes.summarySearchCount.textContent, "1 term · 1 row");

  view.query = "";
  applySummaryView(nodes);
  assert.doesNotMatch(nodes.summaryFrame.innerHTML, / hidden>|<mark>/);
  assert.equal(nodes.summarySearchCount.textContent, "");
});
```

`tests/editor/test_editor_workspace_browser.py`. Insert immediately before
`def test_context_bar_reports_term_kind_and_edf(open_editor_page):` (line 1712):
```python
def test_summary_search_filters_rows_survives_a_new_payload_and_escape_clears_it(
    open_editor_page,
):
    with open_editor_page(selected_term="territory") as (page, _session):
        page.wait_for_function(
            """() => document.querySelector('#summaryFrame')?.getAttribute('aria-busy') === 'false'
                && document.querySelectorAll('#summaryFrame .summary-row').length > 0"""
        )
        search = page.get_by_role("searchbox", name="Search terms and levels")

        def shown_rows() -> list[str]:
            return page.locator("#summaryFrame tr.summary-row:not([hidden])").evaluate_all(
                "rows => rows.map(row => row.querySelector('.summary-term span').textContent)"
            )

        # T01..T10 are text: "t1" is in T10 only, and in no term's name.
        search.fill("t1")
        assert shown_rows() == ["territory[T10]"]
        marks = page.locator("#summaryFrame tr.summary-row:not([hidden]) mark")
        assert marks.all_inner_texts() == ["T1"]
        assert page.locator("#summarySearchCount").inner_text() == "1 term · 1 row"

        # A new payload rebuilds the frame, and the search is applied to it.
        grouped = page.get_by_role("group", name="Categorical levels").get_by_role(
            "radio", name="Grouped"
        )
        with page.expect_response(
            lambda response: (
                response.request.method == "POST"
                and response.url.split("?", maxsplit=1)[0].endswith("/summary")
            )
        ):
            grouped.check()
        page.wait_for_function(
            "() => document.querySelector('#summaryFrame')?.getAttribute('aria-busy') === 'false'"
        )
        assert shown_rows() == ["territory[T10]"]

        search.press("Escape")
        assert search.input_value() == ""
        assert page.locator("#summarySearchCount").inner_text() == ""
        assert len([row for row in shown_rows() if row.startswith("territory[")]) == 10
        assert page.locator("#inspector").get_attribute("data-open") == "true"
```

- [ ] **Step 2: Run them, expect FAIL**

```bash
node --test tests/editor_frontend/summary_view.test.js tests/editor_frontend/summary.test.js
./.venv/bin/python -m pytest tests/editor/test_editor_workspace_browser.py -m browser --run-browser -q -k summary_search_filters_rows
```

Expected on `155832e8`:
- `summary_view.test.js` fails to load with
  `ERR_MODULE_NOT_FOUND ... views/summary_view.js`.
- In `summary.test.js`, "the inspector search is reapplied …" fails on its first
  assertion: master rows carry no `data-term`. Every other test passes.
- The browser test fails with
  `Locator.fill: Timeout 30000ms exceeded … get_by_role("searchbox", name="Search terms and levels")`.

- [ ] **Step 3: Implement**

Create `src/superglm/editor/app/views/summary_view.js`:
```js
// @ts-check
// The inspector summary's view of the compact coefficient rows: the term each
// row belongs to, which rows a search keeps, and the marks it draws. Pure but
// for the search box binding; summary.js turns the result into markup.

import { escapeHTML } from "../format.js";

/**
 * The fields of one compact summary row (Python's `_compact_summary_row`)
 * read here.
 * @typedef {Object} SummaryRowLike
 * @property {string} [name]
 * @property {string} [group]
 * @property {string} [level_group]
 */
/**
 * What the inspector shows of the summary. `termNames` are the editor's terms,
 * which a decorated summary group such as "x_poly P(2)" is mapped back to.
 * @typedef {Object} SummaryView
 * @property {string} query
 * @property {readonly string[]} termNames
 */
/**
 * @typedef {Object} SummaryViewRow
 * @property {number} index position in the payload's rows
 * @property {boolean} hidden
 */
/**
 * One term's rows, in payload order. `label` is the group the summary prints
 * ("x_poly P(2)"); `term` is the editor term it belongs to ("x_poly"). The
 * intercept is a section without a header.
 * @typedef {Object} SummaryViewSection
 * @property {string} term
 * @property {string} label
 * @property {boolean} header
 * @property {boolean} hidden
 * @property {SummaryViewRow[]} rows
 */
/**
 * @typedef {Object} SummaryViewModel
 * @property {SummaryViewSection[]} sections
 * @property {number} termCount sections with a row shown, the intercept aside
 * @property {number} rowCount rows shown in those sections
 */

/** @type {Readonly<SummaryView>} */
export const DEFAULT_SUMMARY_VIEW = Object.freeze({ query: "", termNames: Object.freeze([]) });

const INTERCEPT = "Intercept";
const REGEXP_SYNTAX = /[.*+?^${}()|[\]\\]/g;

/**
 * The group a row prints under: its group, or the text before its "[".
 * @param {SummaryRowLike} row
 */
function rowGroupLabel(row) {
  const group = row.group ? String(row.group) : "";
  if (group) return group;
  const name = row.name ? String(row.name) : "";
  const bracket = name.indexOf("[");
  return bracket > 0 ? name.slice(0, bracket) : name;
}

/**
 * The term a summary row belongs to: its group when that is an editor term,
 * else the longest editor term the group decorates ("x_poly P(2)" belongs to
 * "x_poly"), else the group itself.
 * @param {SummaryRowLike} row @param {readonly string[]} termNames
 */
export function summaryRowTerm(row, termNames) {
  const label = rowGroupLabel(row);
  if (termNames.includes(label)) return label;
  let owner = "";
  for (const term of termNames) {
    if (label.startsWith(`${term} `) && term.length > owner.length) owner = term;
  }
  return owner || label;
}

/**
 * The level a row names: the text inside its brackets, "10" in "bonus[10]";
 * empty for a whole-term row.
 * @param {SummaryRowLike} row @param {string} term
 */
export function summaryLevelLabel(row, term) {
  const name = row.name ? String(row.name) : "";
  if (!name.endsWith("]")) return "";
  const prefix = `${term}[`;
  if (name.startsWith(prefix)) return name.slice(prefix.length, -1);
  const bracket = name.indexOf("[");
  return bracket > 0 ? name.slice(bracket + 1, -1) : "";
}

/**
 * A case-insensitive pattern for the search text taken literally, or null when
 * the box is empty. Labels are compared as text: "1" finds levels 1 and 10.
 * @param {string} query @returns {RegExp|null}
 */
function queryPattern(query) {
  const needle = query.trim();
  return needle ? new RegExp(needle.replace(REGEXP_SYNTAX, "\\$&"), "giu") : null;
}

/** @param {string} text @param {RegExp} pattern */
function hasMatch(text, pattern) {
  return text.search(pattern) >= 0;
}

/**
 * `text` escaped for HTML, each match of the search wrapped in <mark>.
 * @param {string} text @param {string} query
 */
export function highlightMatches(text, query) {
  const pattern = queryPattern(query);
  if (!pattern) return escapeHTML(text);
  let html = "";
  let at = 0;
  for (const match of text.matchAll(pattern)) {
    const start = match.index ?? 0;
    html += `${escapeHTML(text.slice(at, start))}<mark>${escapeHTML(match[0])}</mark>`;
    at = start + match[0].length;
  }
  return html + escapeHTML(text.slice(at));
}

/**
 * A row's name with the search marked in the two parts it reads: the term,
 * and the level inside the brackets.
 * @param {string} name @param {string} term @param {string} query
 */
export function highlightRowName(name, term, query) {
  const prefix = `${term}[`;
  if (term && name.startsWith(prefix) && name.endsWith("]")) {
    const level = name.slice(prefix.length, -1);
    return `${highlightMatches(term, query)}[${highlightMatches(level, query)}]`;
  }
  return highlightMatches(name, query);
}

/**
 * Group the rows into term sections, in payload order, and apply the search:
 * a section whose term matches keeps every row; otherwise a row stays when its
 * level or level group matches. While a search is active the intercept, which
 * is no term, is hidden.
 * @param {readonly SummaryRowLike[]} rows @param {SummaryView} view
 * @returns {SummaryViewModel}
 */
export function summaryViewModel(rows, view) {
  /** @type {SummaryViewSection[]} */
  const sections = [];
  rows.forEach((row, index) => {
    const term = summaryRowTerm(row, view.termNames);
    let section = sections.at(-1);
    if (!section || section.term !== term) {
      section = { term, label: rowGroupLabel(row), header: term !== INTERCEPT, hidden: false, rows: [] };
      sections.push(section);
    }
    section.rows.push({ index, hidden: false });
  });

  const pattern = queryPattern(view.query);
  let termCount = 0;
  let rowCount = 0;
  for (const section of sections) {
    if (pattern !== null) {
      const keepAll = section.header && hasMatch(section.term, pattern);
      for (const entry of section.rows) {
        const row = rows[entry.index];
        entry.hidden = !section.header || !(
          keepAll ||
          hasMatch(summaryLevelLabel(row, section.term), pattern) ||
          hasMatch(row.level_group ? String(row.level_group) : "", pattern)
        );
      }
    }
    section.hidden = section.rows.every((entry) => entry.hidden);
    if (section.header && !section.hidden) {
      termCount += 1;
      rowCount += section.rows.filter((entry) => !entry.hidden).length;
    }
  }
  return { sections, termCount, rowCount };
}

/** @param {number} count @param {string} noun */
function counted(count, noun) {
  return `${count} ${noun}${count === 1 ? "" : "s"}`;
}

/**
 * What the search box reports: nothing while it is empty, else the terms and
 * rows found, or that none match.
 * @param {SummaryViewModel} model @param {string} query
 */
export function summaryCountText(model, query) {
  if (!query.trim()) return "";
  if (model.termCount === 0) return "No terms match.";
  return `${counted(model.termCount, "term")} · ${counted(model.rowCount, "row")}`;
}

/**
 * Wire the search box: each input reports the query; Escape clears a box that
 * holds text, and keeps the key from also closing a narrow inspector.
 * @param {HTMLInputElement} search @param {(query:string)=>unknown} onQuery
 * @returns {{destroy:()=>void}}
 */
export function bindSummarySearch(search, onQuery) {
  function onInput() {
    onQuery(search.value);
  }

  /** @param {KeyboardEvent} event */
  function onKeyDown(event) {
    if (event.key !== "Escape" || !search.value) return;
    event.preventDefault();
    event.stopPropagation();
    search.value = "";
    onQuery("");
  }

  search.addEventListener("input", onInput);
  search.addEventListener("keydown", onKeyDown);
  return Object.freeze({
    destroy() {
      search.removeEventListener("input", onInput);
      search.removeEventListener("keydown", onKeyDown);
    },
  });
}
```

`src/superglm/editor/app/summary.js`. Seven edits.

Edit E1.1. Replace `import { SHAPE_NAMES } from "./shapes.js";` with:
```js
import { SHAPE_NAMES } from "./shapes.js";
import {
  DEFAULT_SUMMARY_VIEW,
  highlightMatches,
  highlightRowName,
  summaryCountText,
  summaryViewModel
} from "./views/summary_view.js";
```

Edit E1.2. Replace `const summaryMarkupByFrame = new WeakMap();` with:
```js
const summaryMarkupByFrame = new WeakMap();
// The payload each frame shows, so a change of view can redraw it unfetched.
const summaryPayloadByFrame = new WeakMap();
```

Edit E1.3. Replace the whole of `renderSummary`, lines 222–242:
```js
export function renderSummary(payload, nodes) {
  const { summaryStatus, summaryNote, summaryFrame } = nodes;
  updateDistributionProfileActions(payload, nodes);
  if (!payload.available) {
    summaryStatus.textContent = payload.label || "Summary";
    summaryNote.textContent = "";
    updateSummaryMarkup(
      summaryFrame,
      `<div class="summary-empty">${escapeHTML(payload.error || "Summary unavailable.")}</div>`
    );
    return;
  }
  summaryStatus.textContent = payload.label || "Summary";
  summaryNote.textContent = payload.note || "";
  // Prefer the typed compact payload for the immediate panel. The raw HTML is
  // still included inside the disclosure for full notebook-style detail.
  updateSummaryMarkup(
    summaryFrame,
    payload.compact ? renderCompactSummary(payload) : payload.html || ""
  );
}
```
with:
```js
export function renderSummary(payload, nodes) {
  const { summaryStatus, summaryNote, summaryFrame } = nodes;
  summaryPayloadByFrame.set(summaryFrame, payload);
  updateDistributionProfileActions(payload, nodes);
  if (!payload.available) {
    summaryStatus.textContent = payload.label || "Summary";
    summaryNote.textContent = "";
    updateSummaryMarkup(
      summaryFrame,
      `<div class="summary-empty">${escapeHTML(payload.error || "Summary unavailable.")}</div>`
    );
    renderSearchCount(nodes, "");
    return;
  }
  summaryStatus.textContent = payload.label || "Summary";
  summaryNote.textContent = payload.note || "";
  // Prefer the typed compact payload for the immediate panel. The raw HTML is
  // still included inside the disclosure for full notebook-style detail. The
  // inspector's search is part of the markup, so every render reapplies it.
  const view = summaryViewOf(nodes);
  const viewModel = payload.compact ? compactViewModel(payload.compact, view) : null;
  updateSummaryMarkup(
    summaryFrame,
    viewModel ? renderCompactSummary(payload, viewModel, view.query) : payload.html || ""
  );
  renderSearchCount(nodes, viewModel ? summaryCountText(viewModel, view.query) : "");
}

/**
 * Redraw the summary on show for the inspector's current search. With the
 * compact table in the DOM only its body is rewritten, so an open "Full
 * summary" keeps its frame; otherwise the last payload is rendered again.
 */
export function applySummaryView(nodes) {
  const { summaryFrame } = nodes;
  const payload = summaryPayloadByFrame.get(summaryFrame);
  const shown = summaryMarkupByFrame.get(summaryFrame);
  // Only a frame still holding its last render is redrawn: one emptied while
  // the other level display loads waits for that payload.
  if (!payload || !shown || shown.firstElementChild !== summaryFrame.firstElementChild) return;
  const body = payload.available && payload.compact && typeof summaryFrame.querySelector === "function"
    ? summaryFrame.querySelector(".summary-table tbody")
    : null;
  if (!body) {
    renderSummary(payload, nodes);
    return;
  }
  const view = summaryViewOf(nodes);
  const viewModel = compactViewModel(payload.compact, view);
  body.innerHTML = renderSummaryBody(
    compactRows(payload.compact),
    viewModel,
    payload.compact.has_level_groups === true,
    view.query
  );
  renderSearchCount(nodes, summaryCountText(viewModel, view.query));
  // The frame now holds what a full render for this view writes, so a later
  // render of the same payload and view leaves the DOM alone.
  summaryMarkupByFrame.set(summaryFrame, {
    markup: renderCompactSummary(payload, viewModel, view.query),
    firstElementChild: summaryFrame.firstElementChild
  });
}

function summaryViewOf(nodes) {
  return typeof nodes.summaryView === "function" ? nodes.summaryView() : DEFAULT_SUMMARY_VIEW;
}

function compactRows(compact) {
  return Array.isArray(compact.rows) ? compact.rows : [];
}

function compactViewModel(compact, view) {
  return summaryViewModel(compactRows(compact), view);
}

function renderSearchCount(nodes, text) {
  if (nodes.summarySearchCount) nodes.summarySearchCount.textContent = text;
}
```

Edit E1.4. Replace:
```js
function renderCompactSummary(payload) {
  const compact = payload.compact || {};
  const model = compact.model || {};
  const rows = Array.isArray(compact.rows) ? compact.rows : [];
  const hasLevelGroups = compact.has_level_groups === true;
  const columnCount = hasLevelGroups ? 8 : 7;
```
with:
```js
function renderCompactSummary(payload, viewModel, query) {
  const compact = payload.compact || {};
  const model = compact.model || {};
  const rows = compactRows(compact);
  const hasLevelGroups = compact.has_level_groups === true;
```

Edit E1.5. Replace `          ${renderSummaryRows(rows, hasLevelGroups, columnCount)}` with:
```js
          ${renderSummaryBody(rows, viewModel, hasLevelGroups, query)}
```

Edit E1.6. Replace `renderSummaryRows` and `summaryRowGroup`, lines 566–585:
```js
function renderSummaryRows(rows, hasLevelGroups, columnCount) {
  let previousGroup = "";
  return rows.map((row) => {
    const group = summaryRowGroup(row);
    const showGroup = group && group !== previousGroup && group !== "Intercept";
    previousGroup = group || previousGroup;
    const groupRow = showGroup
      ? `<tr class="summary-group-row"><td colspan="${columnCount}">${escapeHTML(group)}</td></tr>`
      : "";
    return `${groupRow}${renderSummaryRow(row, hasLevelGroups)}`;
  }).join("");
}

function summaryRowGroup(row) {
  const group = row && row.group ? String(row.group) : "";
  if (group) return group;
  const name = row && row.name ? String(row.name) : "";
  const bracket = name.indexOf("[");
  return bracket > 0 ? name.slice(0, bracket) : name;
}
```
with:
```js
// One header row per term, then its rows. Rows and headers outside the
// search stay in the markup, hidden, each tagged with its term.
function renderSummaryBody(rows, viewModel, hasLevelGroups, query) {
  const columnCount = hasLevelGroups ? 8 : 7;
  return viewModel.sections.map((section) => {
    const groupRow = section.header
      ? `<tr class="summary-group-row" data-term="${escapeHTML(section.term)}"${section.hidden ? " hidden" : ""}><td colspan="${columnCount}">${highlightMatches(section.label, query)}</td></tr>`
      : "";
    const sectionRows = section.rows.map((entry) => renderSummaryRow(
      rows[entry.index],
      hasLevelGroups,
      section.term,
      query,
      entry.hidden
    ));
    return groupRow + sectionRows.join("");
  }).join("");
}
```

Edit E1.7. Replace:
```js
function renderSummaryRow(row, hasLevelGroups) {
  // SE cell color is data-driven from Python's significance class. The browser
  // never infers significance from display text.
  const sigClass = safeSigClass(row.sig_class);
  const levelGroupCell = hasLevelGroups
    ? `<td class="summary-level-group">${escapeHTML(row.level_group || "")}</td>`
    : "";
  return `
    <tr class="summary-row ${sigClass}">
      <td class="summary-term">
        <span>${escapeHTML(row.name || "")}</span>
```
with:
```js
function renderSummaryRow(row, hasLevelGroups, term, query, hidden) {
  // SE cell color is data-driven from Python's significance class. The browser
  // never infers significance from display text.
  const sigClass = safeSigClass(row.sig_class);
  const levelGroupCell = hasLevelGroups
    ? `<td class="summary-level-group">${highlightMatches(String(row.level_group || ""), query)}</td>`
    : "";
  return `
    <tr class="summary-row ${sigClass}" data-term="${escapeHTML(term)}"${hidden ? " hidden" : ""}>
      <td class="summary-term">
        <span>${highlightRowName(String(row.name || ""), term, query)}</span>
```

`src/superglm/editor/app/main.js`. Five edits.

Edit 1. Replace:
```js
import {
  collapseTransition,
  renderSummary,
```
with:
```js
import {
  applySummaryView,
  collapseTransition,
  renderSummary,
```

Edit 2. Replace `import { mountThemeControl } from "./views/theme.js";` with:
```js
import { bindSummarySearch } from "./views/summary_view.js";
import { mountThemeControl } from "./views/theme.js";
```

Edit 3. Replace `const summaryFrame = document.getElementById("summaryFrame");` with:
```js
const summaryFrame = document.getElementById("summaryFrame");
const summarySearch = document.getElementById("summarySearch");
const summarySearchCount = document.getElementById("summarySearchCount");
let summaryQuery = "";
```

Edit 4. At the end of `summaryNodes()`, replace:
```js
    summaryStatus,
    summaryNote,
    summaryFrame
  };
}
```
with:
```js
    summaryStatus,
    summaryNote,
    summaryFrame,
    summarySearchCount,
    summaryView
  };
}

// What the inspector shows of the summary; summary.js reapplies it on every
// render, so a refit or a new payload keeps the search.
function summaryView() {
  const snapshot = store.getState().remote.snapshot;
  return {
    query: summaryQuery,
    termNames: snapshot ? Object.keys(snapshot.terms) : []
  };
}
```

Edit 5. Replace:
```js
    storeFeatureListOpen(featureListOpen);
    renderFeatureListState();
  }
});
renderFeatureListState();
```
with:
```js
    storeFeatureListOpen(featureListOpen);
    renderFeatureListState();
  }
});
renderFeatureListState();

bindSummarySearch(summarySearch, (query) => {
  summaryQuery = query;
  applySummaryView(summaryNodes());
});
```

`src/superglm/editor/app/index.html`. Replace:
```html
      <div id="summaryPane" class="sidepanel-pane" role="tabpanel" aria-labelledby="summaryTab"
        data-inspector-pane="summary">
        <div class="summary-controls">
```
with:
```html
      <div id="summaryPane" class="sidepanel-pane" role="tabpanel" aria-labelledby="summaryTab"
        data-inspector-pane="summary">
        <label class="summary-search">
          <svg class="summary-search-icon" viewBox="0 0 16 16" aria-hidden="true">
            <circle cx="7" cy="7" r="4.5"></circle>
            <path d="m10.5 10.5 3.5 3.5"></path>
          </svg>
          <input id="summarySearch" type="search" placeholder="Search terms and levels"
            aria-label="Search terms and levels" autocomplete="off" spellcheck="false">
          <span id="summarySearchCount" class="summary-search-count" aria-live="polite"></span>
        </label>
        <div class="summary-controls">
```

`src/superglm/editor/app/styles.css`. Replace:
```css
.summary-level-segments input:focus-visible + span {
  outline: 2px solid var(--focus);
  outline-offset: -2px;
}
```
with:
```css
.summary-level-segments input:focus-visible + span {
  outline: 2px solid var(--focus);
  outline-offset: -2px;
}
/* The inspector's search: icon, box and count in one field. */
.summary-search {
  display: flex;
  align-items: center;
  gap: 6px;
  height: 30px;
  margin-bottom: 8px;
  padding: 0 8px;
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  background: var(--surface);
  color: var(--muted);
}
.summary-search:focus-within {
  border-color: var(--focus);
  box-shadow: 0 0 0 3px var(--blue-band);
}
.summary-search-icon {
  flex: none;
  width: 15px;
  height: 15px;
  fill: none;
  stroke: currentColor;
  stroke-width: 1.7;
  stroke-linecap: round;
}
.summary-search input[type="search"] {
  flex: 1 1 auto;
  min-width: 0;
  height: 100%;
  padding: 0;
  border: 0;
  outline: none;
  background: transparent;
  color: var(--text);
}
.summary-search-count {
  flex: none;
  font-family: var(--font-mono);
  font-size: 11px;
  white-space: nowrap;
}
.summary-frame mark {
  padding: 0 1px;
  border-radius: 2px;
  background: color-mix(in srgb, var(--yellow) 45%, transparent);
  color: inherit;
}
```

`src/superglm/editor/app/views/help_content.js`. Replace:
```js
  Object.freeze({
    title: "Model structure",
    keys: Object.freeze(Object.keys(STRUCTURE_HELP)),
  }),
```
with:
```js
  Object.freeze({
    title: "Summary",
    items: Object.freeze([
      "The search box at the top of Summary keeps the terms and levels whose names contain the text, ignoring case, and counts what it found. Escape clears it. The full summary below the table is not searched.",
    ]),
  }),
  Object.freeze({
    title: "Model structure",
    keys: Object.freeze(Object.keys(STRUCTURE_HELP)),
  }),
```

- [ ] **Step 4: Run the tests, expect PASS**

```bash
node --test tests/editor_frontend/summary_view.test.js tests/editor_frontend/summary.test.js
npm run check:frontend
./.venv/bin/python -m pytest tests/test_editor.py -q -k "summary"
./.venv/bin/python -m pytest tests/editor -m browser --run-browser -q -n 6
./.venv/bin/python -m pytest tests/test_editor_browser.py -m browser --run-browser -q
```

These neighbours must stay green:
- `test_summary_updating_status_preserves_confirmed_table_nodes`, and
  `test_term_change_does_not_rewrite_unrelated_editor_panels` (it asserts
  `#summaryFrame > *` is the same node after a term change);
- `test_summary_level_display_toggle_*`;
- `test_ordered_summary_has_one_whole_smooth_test_and_no_level_tests`;
- `test_raw_summary_html_is_isolated_in_a_sandboxed_iframe`;
- in `summary.test.js`, "rendering unchanged summary markup preserves the
  existing table DOM" (writes == 1) and "ordinary compact summary remains a
  seven-column table" (seven `<th>`).

- [ ] **Step 5: Commit**

```bash
git add src/superglm/editor/app/views/summary_view.js src/superglm/editor/app/summary.js \
  src/superglm/editor/app/main.js src/superglm/editor/app/index.html \
  src/superglm/editor/app/styles.css src/superglm/editor/app/views/help_content.js \
  tests/editor_frontend/summary_view.test.js tests/editor_frontend/summary.test.js \
  tests/editor/test_editor_workspace_browser.py
git commit -m "Editor inspector: search the summary by term and level

A box at the top of Summary keeps the terms and levels whose names contain the
text (case-insensitive, literal; numeric-looking levels match as text), marks
the matches, counts terms and rows, and clears on Escape. Rows carry data-term.
The search is part of every render, so a refit or a new payload keeps it; a
keystroke rewrites only the table body and the full summary is not searched.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task E2: All / Edited / Waiting filters, follows the chart, and the header layout

Depends on: E1 (`summary_view.js`, `applySummaryView`), C1 (`_click_selects`,
`mutationStatus`). If A4 has landed, it supplies the top-level `pending`.

**Files:**
- Modify `src/superglm/editor/payloads.py`:
  - `session_payload`, lines 27–28;
  - the term dict, line 59.
- Modify `src/superglm/editor/app/api/contracts.js:120-122`, `TermPayload`.
- Modify `src/superglm/editor/app/views/summary_view.js`, as created in E1.
- Modify `src/superglm/editor/app/summary.js`, as left by E1, plus the
  `renderCompactSummary` facts at master lines 499–510 and 521–524.
- Modify `src/superglm/editor/app/main.js`, as left by E1.
- Modify `src/superglm/editor/app/index.html`:
  - after E1's search label;
  - master lines 469–471, which move.
- Modify `src/superglm/editor/app/styles.css`:
  - after E1's block;
  - after master lines 765–767;
  - master lines 792–797.
- Modify `src/superglm/editor/app/views/help_content.js`, E1's "Summary" section.
- Test `tests/test_editor_structure.py`: insert before line 356.
- Test `tests/editor_frontend/summary_view.test.js` and
  `tests/editor_frontend/summary.test.js`.
- Test `tests/editor/test_editor_workspace_browser.py`: insert after E1's test.

**Interfaces:**
- Consumes:
  - per-term `edited` (added here);
  - top-level `pending: [{term, ...}]` from A4. It is read as `snapshot.pending ?? []`.
  - `selectActiveTermName(state)`.
- Produces, from `summary_view.js`:
  - `waitingCounts(pending: readonly {term:string}[]): Record<string, number>`;
  - `bindSummaryFilter(root, onFilter: (filter: "all"|"edited"|"waiting") => unknown): {destroy}`;
  - `renderSummaryFilter(root, filter): void`;
  - `bindSummarySections(frame, onToggle: (term: string, open: boolean) => unknown): {destroy}`.
- `SummaryView` gains optional `filter`, `currentTerm`, `toggled`
  (`ReadonlyMap<string, boolean>`), `edited`, `waiting` and `kinds`.
- `SummaryViewSection` gains `open`, `current`, `kind`, `waiting`, `edf` and `chip`.
- `SummaryViewModel` gains `empty`.
- From `summary.js`: `applySummaryView(nodes, {follow = false} = {})`.
- Per term: `edited`.

Rules, from spec §4 E:

**Following the chart**
- The chart's term section is open. Every other term folds to one line: name,
  kind, EDF, p chip and waiting chip.
- On term change, or when the summary is rewritten, the current line scrolls
  into view (`block: "nearest"`).
- A section the user opens or closes stays so until the chart's term or the
  search changes; `main.js` clears `summaryToggled` then.
- A search opens every section it finds.
- With no current term, every section is open. This is what callers without a
  view get, so they render as before.

**Filters**
- Edited: terms with `edited === true`.
- Waiting: terms with one or more pending steps.
- Filters and search combine.
- An empty result shows "No terms match." in the table.

**The p chip**
- Terms with a whole-term test (spline, piecewise) show that row's p.
- Categoricals have no term-level test. Their chip shows the smallest level p,
  labelled `min`, with the title "Smallest p-value among this term's levels".
- The colour is that row's Python `sig_class`; nothing is re-derived in the
  browser (`summary.js:601-603` keeps the same rule).
- EDF is the sum of the rows' finite `edf`. Python puts a categorical's EDF on
  one row only (measured: `region[B]` carries 2.0).

**Header**
- Family, link and method appear as chips.
- Four tiles: Deviance, AIC, BIC, Total EDF.
- Refit offsets sits beside the chips.
- The header hides while a search is active, as on the approved board.
- Tweedie p, its CI and NB2 theta stay in the frame; the existing
  `summary.test.js` tests pin "[1.4, 1.7]" and "censored".
- Log likelihood leaves the inspector. It is still in the metric grid
  (`metrics.js:7`).
- The Expanded/Grouped fieldset (`<legend>Categorical levels</legend>`, pinned
  by `tests/test_editor.py:6425-6431`) stays in `.summary-controls`.

- [ ] **Step 1: Write the failing tests**

`tests/test_editor_structure.py`. Insert before
`def test_widget_http_set_reference_returns_transition_envelope(region_model):` (line 356):
```python
def test_payload_marks_the_terms_that_carry_hand_edits(region_model):
    model, _ = region_model
    session = EditorSession.from_model(model, terms=["region", "x"])
    assert {name: term["edited"] for name, term in session_payload(session).items()} == {
        "region": False,
        "x": False,
    }
    session.select_indices("x", [3, 4])
    session.shift("x", 0.1)
    payload = session_payload(session)
    assert (payload["x"]["edited"], payload["region"]["edited"]) == (True, False)
    session.undo()
    assert session_payload(session)["x"]["edited"] is False
```

`tests/editor_frontend/summary_view.test.js`.

Edit 1. Replace:
```js
  summaryRowTerm,
  summaryViewModel
} = await import(summaryViewModulePath);
```
with:
```js
  summaryRowTerm,
  summaryViewModel,
  waitingCounts
} = await import(summaryViewModulePath);
```

Then append:
```js

// A small book: a spline with its whole-term test, two categoricals.
const BOOK_ROWS = [
  { name: "Intercept", group: "", kind: "coef", edf: 1, p_value: 1e-6, sig_class: "sig-strong", sig_code: "***" },
  { name: "DrivAge", group: "DrivAge", kind: "spline", edf: 11.2, p_value: 2e-5, sig_class: "sig-strong", sig_code: "***" },
  { name: "VehBrand[B1]", group: "VehBrand", kind: "reference", edf: null, p_value: null, sig_class: "sig-reference" },
  { name: "VehBrand[B2]", group: "VehBrand", kind: "coef", edf: 10, p_value: 0.03, sig_class: "sig-standard", sig_code: "*" },
  { name: "VehBrand[B10]", group: "VehBrand", kind: "coef", edf: null, p_value: 0.004, sig_class: "sig-medium", sig_code: "**" },
  { name: "Area[A]", group: "Area", kind: "reference", edf: null, p_value: null, sig_class: "sig-reference" },
  { name: "Area[B]", group: "Area", kind: "coef", edf: 1, p_value: 0.5, sig_class: "sig-none", sig_code: "" }
];
const BOOK = {
  query: "",
  termNames: ["DrivAge", "VehBrand", "Area"],
  currentTerm: "VehBrand",
  kinds: { DrivAge: "spline", VehBrand: "categorical" }
};

function book(patch = {}) {
  return summaryViewModel(BOOK_ROWS, { ...BOOK, ...patch });
}

function lines(model) {
  return model.sections.filter((section) => section.header && !section.hidden).map((section) => section.term);
}

function openLines(model) {
  return model.sections
    .filter((section) => section.header && !section.hidden && section.open)
    .map((section) => section.term);
}

function bookRows(model) {
  return model.sections.flatMap((section) => section.rows
    .filter((entry) => !entry.hidden)
    .map((entry) => BOOK_ROWS[entry.index].name));
}

test("the chart's term is open, the others fold to a line that still says how they fit", () => {
  const model = book();

  assert.deepEqual(lines(model), ["DrivAge", "VehBrand", "Area"]);
  assert.deepEqual(openLines(model), ["VehBrand"]);
  assert.deepEqual(bookRows(model), ["Intercept", "VehBrand[B1]", "VehBrand[B2]", "VehBrand[B10]"]);
  const [, drivAge, vehBrand, area] = model.sections;
  assert.deepEqual(
    [drivAge.kind, drivAge.edf, drivAge.chip, drivAge.current],
    ["spline", 11.2, { p: 2e-5, sigClass: "sig-strong", sigCode: "***", smallest: false }, false]
  );
  // A categorical has no whole-term test: its line shows the smallest level p.
  assert.deepEqual(
    [vehBrand.edf, vehBrand.chip, vehBrand.current],
    [10, { p: 0.004, sigClass: "sig-medium", sigCode: "**", smallest: true }, true]
  );
  // A term the editor does not show still gets its line.
  assert.deepEqual([area.kind, area.edf, area.chip.p], ["", 1, 0.5]);
});

test("a section opened or closed by hand stays so; without a chart term every section is open", () => {
  const toggled = new Map([["Area", true], ["VehBrand", false]]);
  assert.deepEqual(openLines(book({ toggled })), ["Area"]);
  assert.deepEqual(openLines(book({ currentTerm: "DrivAge" })), ["DrivAge"]);
  assert.deepEqual(openLines(book({ currentTerm: "" })), ["DrivAge", "VehBrand", "Area"]);
});

test("a search opens every section it finds", () => {
  const model = book({ query: "b1" });
  assert.deepEqual(lines(model), ["VehBrand"]);
  assert.deepEqual(bookRows(model), ["VehBrand[B1]", "VehBrand[B10]"]);

  const every = book({ query: "a" });
  assert.deepEqual(openLines(every), ["DrivAge", "VehBrand", "Area"]);
  assert.equal(summaryCountText(every, "a"), "3 terms · 6 rows");
});

test("Edited and Waiting keep the terms with hand edits or changes waiting for refit", () => {
  const edited = book({ filter: "edited", edited: ["DrivAge"] });
  assert.deepEqual(lines(edited), ["DrivAge"]);
  assert.deepEqual(bookRows(edited), []);

  const waiting = book({ filter: "waiting", waiting: { Area: 2 } });
  assert.deepEqual(lines(waiting), ["Area"]);
  assert.equal(waiting.sections[3].waiting, 2);

  // The filter and the search combine.
  const neither = book({ filter: "waiting", waiting: { Area: 2 }, query: "veh" });
  assert.deepEqual(lines(neither), []);
  assert.equal(summaryCountText(neither, "veh"), "No terms match.");
  // An empty result says so in the table too; the plain view never does.
  assert.equal(neither.empty, true);
  assert.equal(book({ filter: "edited" }).empty, true);
  assert.equal(book().empty, false);
});

test("waiting changes are counted per term", () => {
  assert.deepEqual(
    waitingCounts([{ term: "Area" }, { term: "VehBrand" }, { term: "Area" }]),
    { Area: 2, VehBrand: 1 }
  );
  assert.deepEqual(waitingCounts([]), {});
});
```

`tests/editor_frontend/summary.test.js`. Insert immediately before
`test("expanded compact summary shows group indicators without a membership legend", () => {`.
That places it after E1's test; it reuses `bonusSummary()`.
```js
test("the chart's term is open and every other term folds to one line", () => {
  /** @type {import("../../src/superglm/editor/app/views/summary_view.js").SummaryView} */
  const view = {
    query: "",
    termNames: ["region", "bonus"],
    currentTerm: "bonus",
    kinds: { region: "categorical", bonus: "categorical" },
    waiting: { region: 1 }
  };
  const nodes = { ...compactSummaryNodes(), summaryView: () => view };

  renderSummary(bonusSummary(), nodes);
  const markup = nodes.summaryFrame.innerHTML;

  assert.match(
    markup,
    /<tr class="summary-group-row summary-section" data-term="region" data-current="false">/
  );
  assert.match(markup, /data-summary-section="region" aria-expanded="false"/);
  assert.match(markup, /<span class="summary-section-kind">categorical<\/span><span class="summary-waiting">1 waiting<\/span>/);
  assert.match(markup, /<tr class="summary-row sig-reference" data-term="region" hidden>/);
  assert.match(markup, /data-summary-section="bonus" aria-expanded="true"/);
  assert.match(
    markup,
    /<span class="summary-p-chip sig-medium" title="Smallest p-value among this term's levels">min 0\.004 \*\*<\/span>/
  );
  assert.doesNotMatch(markup, /data-term="bonus" hidden/);

  // Opened by hand, a folded term shows its rows; the chart's term may be closed.
  view.toggled = new Map([["region", true], ["bonus", false]]);
  applySummaryView(nodes);
  assert.match(nodes.summaryFrame.innerHTML, /<tr class="summary-row sig-reference" data-term="region">/);
  assert.match(nodes.summaryFrame.innerHTML, /<tr class="summary-row sig-none" data-term="bonus" hidden>/);
});

test("the header shows the model as chips and four tiles and steps aside for a search", () => {
  const view = { query: "", termNames: ["region", "bonus"] };
  const nodes = {
    ...compactSummaryNodes(),
    summaryHeader: { hidden: false },
    summaryModelChips: { innerHTML: "" },
    summaryTiles: { innerHTML: "" },
    summaryView: () => view
  };
  const payload = bonusSummary();
  payload.compact.model = {
    family: "Poisson",
    link: "Log",
    method: "MLE",
    deviance: 445.3,
    aic: 1125.7,
    bic: 1177.3,
    effective_df: 12.93,
    log_likelihood: -549.9
  };

  renderSummary(payload, nodes);

  assert.equal(
    nodes.summaryModelChips.innerHTML,
    '<span class="summary-chip">Poisson</span><span class="summary-chip">Log link</span>'
      + '<span class="summary-chip">MLE</span>'
  );
  const tiles = [...nodes.summaryTiles.innerHTML.matchAll(/<span>([^<]+)<\/span><strong[^>]*>([^<]+)</g)]
    .map((match) => [match[1], match[2]]);
  assert.deepEqual(tiles, [
    ["Deviance", "445.3"], ["AIC", "1125.7"], ["BIC", "1177.3"], ["Total EDF", "12.9"]
  ]);
  assert.doesNotMatch(nodes.summaryFrame.innerHTML, /summary-facts|Deviance|Log lik/);
  assert.equal(nodes.summaryHeader.hidden, false);

  view.query = "bonus";
  applySummaryView(nodes);
  assert.equal(nodes.summaryHeader.hidden, true);
});
```

`tests/editor/test_editor_workspace_browser.py`. Insert after E1's
`test_summary_search_filters_rows_survives_a_new_payload_and_escape_clears_it`:
```python
def test_summary_follows_the_chart_and_filters_edited_and_waiting_terms(
    open_editor_page, choose_feature
):
    with open_editor_page(selected_term="territory") as (page, _session):
        page.wait_for_function(
            """() => document.querySelector('#summaryFrame')?.getAttribute('aria-busy') === 'false'
                && document.querySelector('#summaryFrame tr.summary-section')"""
        )

        def lines() -> list[str]:
            return page.locator("#summaryFrame tr.summary-section:not([hidden])").evaluate_all(
                "rows => rows.map(row => row.dataset.term)"
            )

        def open_lines() -> list[str]:
            return page.locator(
                '#summaryFrame tr.summary-section:not([hidden]) [aria-expanded="true"]'
            ).evaluate_all("buttons => buttons.map(button => button.dataset.summarySection)")

        # The chart's term is open; every other term is one line.
        assert lines() == ["curve", "territory", "age_band", "long_category"]
        assert open_lines() == ["territory"]
        current = page.locator('#summaryFrame tr.summary-section[data-current="true"]')
        assert current.get_attribute("data-term") == "territory"

        # Another term opens from its line and stays open while the chart stays put.
        page.locator('#summaryFrame [data-summary-section="age_band"]').click()
        assert open_lines() == ["territory", "age_band"]

        # The chart moves on: the summary follows it, folds the rest again, and
        # rewrites only its table body.
        summary_child = page.locator("#summaryFrame > *").first.element_handle()
        choose_feature(page, "curve")
        page.wait_for_function("() => document.querySelector('#status')?.dataset.term === 'curve'")
        assert open_lines() == ["curve"]
        assert summary_child.evaluate(
            "node => node === document.querySelector('#summaryFrame > *')"
        )

        # A hand edit on curve: Edited keeps it alone, and nothing is waiting.
        select_chart_tool(page, "Select")
        _click_selects(page, page.locator('#chart circle.point[data-index="5"]'))
        with page.expect_response(
            lambda response: (
                response.request.method == "POST"
                and response.url.split("?", maxsplit=1)[0].endswith("/op")
            )
        ):
            page.locator('button[data-op="shift_up"]').click()
        edited = page.get_by_role("button", name="Edited", exact=True)
        edited.click()
        page.wait_for_function(
            """() => [...document.querySelectorAll('#summaryFrame tr.summary-section:not([hidden])')]
                .map(row => row.dataset.term).join() === 'curve'"""
        )
        assert edited.get_attribute("aria-pressed") == "true"
        page.get_by_role("button", name="Waiting", exact=True).click()
        assert lines() == []
        assert page.locator("#summaryFrame .summary-empty-row").inner_text() == "No terms match."
        page.get_by_role("button", name="All", exact=True).click()
        assert lines() == ["curve", "territory", "age_band", "long_category"]
```

- [ ] **Step 2: Run them, expect FAIL**

```bash
./.venv/bin/python -m pytest tests/test_editor_structure.py -q -k marks_the_terms_that_carry_hand_edits
node --test tests/editor_frontend/summary_view.test.js tests/editor_frontend/summary.test.js
./.venv/bin/python -m pytest tests/editor/test_editor_workspace_browser.py -m browser --run-browser -q -k summary_follows_the_chart
```

Expected, after E1, with E2 not yet applied:
- Python: `KeyError: 'edited'`.
- Node: 7 failures.
  - The `book()` tests: `section.open` and `section.chip` are undefined.
  - "waiting changes are counted": `waitingCounts is not a function`.
  - The two `summary.test.js` tests: no `summary-section` markup, and no chips.
- Browser: `Page.wait_for_function: Timeout 30000ms exceeded`. No
  `tr.summary-section` exists.

On `155832e8` all of them fail too: there is no `summary_view.js` (E1) and no
`edited` key.

- [ ] **Step 3: Implement**

`src/superglm/editor/payloads.py`. Two edits. If A4 has reshaped these lines,
make the same insertions at the equivalent places.

Edit 1. Replace:
```python
    payload: dict[str, dict[str, Any]] = {}
    for name, term in session.terms.items():
```
with:
```python
    payload: dict[str, dict[str, Any]] = {}
    edited = set(session.edited_terms())
    for name, term in session.terms.items():
```

Edit 2. Replace:
```python
            "effective_df": _finite_float(term.metadata.get("edf")),
```
with:
```python
            "effective_df": _finite_float(term.metadata.get("edf")),
            "edited": name in edited,
```

`src/superglm/editor/app/api/contracts.js` (`TermPayload`). Replace:
```js
 * @property {Array<{label:string, indices:number[]}>} [level_groups]
 * @property {TermShape} shape
 */
```
with:
```js
 * @property {Array<{label:string, indices:number[]}>} [level_groups]
 * @property {TermShape} shape
 * @property {boolean} [edited] whether the term carries hand edits
 *   (Python's `EditorSession.edited_terms()`)
 */
```

`src/superglm/editor/app/views/summary_view.js`, editing E1's file.

Edit E2.1. Replace the header comment:
```js
// The inspector summary's view of the compact coefficient rows: the term each
// row belongs to, which rows a search keeps, and the marks it draws. Pure but
// for the search box binding; summary.js turns the result into markup.
```
with:
```js
// The inspector summary's view of the compact coefficient rows: the term each
// row belongs to, which rows a search and a filter keep, which term sections
// are open as the summary follows the chart, and the marks a search draws.
// Pure but for three small bindings; summary.js turns the result into markup.
```

Edit E2.2. Replace:
```js
 * @property {string} [level_group]
 */
/**
 * What the inspector shows of the summary. `termNames` are the editor's terms,
 * which a decorated summary group such as "x_poly P(2)" is mapped back to.
 * @typedef {Object} SummaryView
 * @property {string} query
 * @property {readonly string[]} termNames
 */
```
with:
```js
 * @property {string} [level_group]
 * @property {string} [kind]
 * @property {number|null} [edf]
 * @property {number|null} [p_value]
 * @property {string} [sig_class]
 * @property {string} [sig_code]
 */
/** @typedef {'all'|'edited'|'waiting'} SummaryFilter */
/**
 * What the inspector shows of the summary. `termNames` are the editor's terms,
 * which a decorated summary group such as "x_poly P(2)" is mapped back to.
 * `currentTerm` is the chart's; its section opens and the others fold, unless
 * `toggled` says the analyst opened or closed one. With no current term every
 * section is open. `waiting` counts the changes waiting for refit per term.
 * @typedef {Object} SummaryView
 * @property {string} query
 * @property {readonly string[]} termNames
 * @property {SummaryFilter} [filter]
 * @property {string} [currentTerm]
 * @property {ReadonlyMap<string, boolean>} [toggled]
 * @property {readonly string[]} [edited]
 * @property {Readonly<Record<string, number>>} [waiting]
 * @property {Readonly<Record<string, string>>} [kinds]
 */
/**
 * The p-value a folded section shows: the whole-term test's, or else the
 * smallest of its levels', which `smallest` marks.
 * @typedef {Object} SummaryChip
 * @property {number} p
 * @property {string} sigClass
 * @property {string} sigCode
 * @property {boolean} smallest
 */
```

Edit E2.3. Replace:
```js
 * intercept is a section without a header.
 * @typedef {Object} SummaryViewSection
 * @property {string} term
 * @property {string} label
 * @property {boolean} header
 * @property {boolean} hidden
 * @property {SummaryViewRow[]} rows
 */
/**
 * @typedef {Object} SummaryViewModel
 * @property {SummaryViewSection[]} sections
 * @property {number} termCount sections with a row shown, the intercept aside
 * @property {number} rowCount rows shown in those sections
 */
```
with:
```js
 * intercept is a section without a header, always open.
 * @typedef {Object} SummaryViewSection
 * @property {string} term
 * @property {string} label
 * @property {boolean} header
 * @property {boolean} hidden
 * @property {boolean} open
 * @property {boolean} current
 * @property {string} kind
 * @property {number} waiting
 * @property {number|null} edf
 * @property {SummaryChip|null} chip
 * @property {SummaryViewRow[]} rows
 */
/**
 * @typedef {Object} SummaryViewModel
 * @property {SummaryViewSection[]} sections
 * @property {number} termCount sections with a row shown, the intercept aside
 * @property {number} rowCount rows shown in those sections
 * @property {boolean} empty a search or filter left no term to show
 */
```

Edit E2.4. Replace:
```js
const INTERCEPT = "Intercept";
const REGEXP_SYNTAX = /[.*+?^${}()|[\]\\]/g;
```
with:
```js
const INTERCEPT = "Intercept";
const REGEXP_SYNTAX = /[.*+?^${}()|[\]\\]/g;
// Row kinds that carry a whole-term test (Python's group rows).
const GROUP_TEST_KINDS = new Set(["spline", "piecewise"]);
```

Edit E2.5. Replace the whole of E1's `summaryViewModel`, from its doc comment
`/**\n * Group the rows into term sections, in payload order, and apply the search:`
through its closing `return { sections, termCount, rowCount };\n}`, with:
```js
/**
 * Group the rows into term sections, in payload order, and apply the view.
 * The search: a section whose term matches keeps every row; otherwise a row
 * stays when its level or level group matches; the intercept, which is no
 * term, leaves while searching. The filter keeps every term, the terms with
 * hand edits, or the terms waiting for refit. A section is open when the
 * analyst opened it, else when a search is active, else when it is the chart's
 * term; a folded section shows its line only.
 * @param {readonly SummaryRowLike[]} rows @param {SummaryView} view
 * @returns {SummaryViewModel}
 */
export function summaryViewModel(rows, view) {
  /** @type {SummaryViewSection[]} */
  const sections = [];
  rows.forEach((row, index) => {
    const term = summaryRowTerm(row, view.termNames);
    let section = sections.at(-1);
    if (!section || section.term !== term) {
      section = newSection(term, rowGroupLabel(row), view);
      sections.push(section);
    }
    section.rows.push({ index, hidden: false });
  });

  const pattern = queryPattern(view.query);
  const filter = view.filter ?? "all";
  const edited = new Set(view.edited ?? []);
  let termCount = 0;
  let rowCount = 0;
  for (const section of sections) {
    summarizeSection(section, rows);
    const matched = section.rows.map((entry) => rowMatches(rows[entry.index], section, pattern));
    section.hidden = !passesFilter(section, filter, edited) || !matched.some(Boolean);
    section.open = !section.header || (
      view.toggled?.get(section.term) ?? (pattern !== null || !view.currentTerm || section.current)
    );
    section.rows.forEach((entry, k) => {
      entry.hidden = section.hidden || !matched[k] || !section.open;
    });
    if (section.header && !section.hidden) {
      termCount += 1;
      rowCount += matched.filter(Boolean).length;
    }
  }
  const empty = termCount === 0 && (pattern !== null || filter !== "all");
  return { sections, termCount, rowCount, empty };
}

/**
 * @param {string} term @param {string} label @param {SummaryView} view
 * @returns {SummaryViewSection}
 */
function newSection(term, label, view) {
  return {
    term,
    label,
    header: term !== INTERCEPT,
    hidden: false,
    open: true,
    current: term === view.currentTerm,
    kind: view.kinds?.[term] ?? "",
    waiting: view.waiting?.[term] ?? 0,
    edf: null,
    chip: null,
    rows: []
  };
}

/**
 * Whether a row is in the search; every row is while the box is empty.
 * @param {SummaryRowLike} row @param {SummaryViewSection} section @param {RegExp|null} pattern
 */
function rowMatches(row, section, pattern) {
  if (pattern === null) return true;
  if (!section.header) return false;
  return hasMatch(section.term, pattern) ||
    hasMatch(summaryLevelLabel(row, section.term), pattern) ||
    hasMatch(row.level_group ? String(row.level_group) : "", pattern);
}

/**
 * @param {SummaryViewSection} section @param {SummaryFilter} filter
 * @param {ReadonlySet<string>} edited
 */
function passesFilter(section, filter, edited) {
  if (filter === "edited") return section.header && edited.has(section.term);
  if (filter === "waiting") return section.waiting > 0;
  return true;
}

/** @param {unknown} value @returns {value is number} */
function isFiniteNumber(value) {
  return typeof value === "number" && Number.isFinite(value);
}

/**
 * The figures on a section's line: the EDF its rows carry, and the p-value of
 * its whole-term test or else the smallest of its levels'. Significance comes
 * from Python's class and code for that row; nothing is re-derived here.
 * @param {SummaryViewSection} section @param {readonly SummaryRowLike[]} rows
 */
function summarizeSection(section, rows) {
  const sectionRows = section.rows.map((entry) => rows[entry.index]);
  const edfs = sectionRows.map((row) => row.edf).filter(isFiniteNumber);
  section.edf = edfs.length ? edfs.reduce((sum, value) => sum + value, 0) : null;
  const test = sectionRows.find(
    (row) => GROUP_TEST_KINDS.has(String(row.kind)) && isFiniteNumber(row.p_value)
  );
  /** @type {SummaryRowLike|null} */
  let source = test ?? null;
  if (test) {
    section.kind ||= String(test.kind);
  } else {
    for (const row of sectionRows) {
      if (isFiniteNumber(row.p_value) && (source === null || row.p_value < Number(source.p_value))) {
        source = row;
      }
    }
  }
  section.chip = source === null ? null : {
    p: Number(source.p_value),
    sigClass: String(source.sig_class ?? ""),
    sigCode: String(source.sig_code ?? ""),
    smallest: !test
  };
}

/**
 * The changes waiting for refit per term, from the state's top-level `pending`.
 * @param {readonly {term:string}[]} pending @returns {Record<string, number>}
 */
export function waitingCounts(pending) {
  /** @type {Record<string, number>} */
  const counts = {};
  for (const step of pending) counts[step.term] = (counts[step.term] ?? 0) + 1;
  return counts;
}
```

Edit E2.6. Insert immediately before the doc comment of `bindSummarySearch`
(`/**\n * Wire the search box: each input reports the query; ...`):
```js
/** @param {string|undefined} value @returns {SummaryFilter|null} */
function summaryFilter(value) {
  return value === "all" || value === "edited" || value === "waiting" ? value : null;
}

/**
 * Wire the All / Edited / Waiting buttons inside `root`.
 * @param {HTMLElement} root @param {(filter:SummaryFilter)=>unknown} onFilter
 * @returns {{destroy:()=>void}}
 */
export function bindSummaryFilter(root, onFilter) {
  /** @param {MouseEvent} event */
  function onClick(event) {
    const target = event.target instanceof Element ? event.target : null;
    const button = target?.closest("[data-summary-filter]");
    if (!(button instanceof HTMLElement) || !root.contains(button)) return;
    const filter = summaryFilter(button.dataset.summaryFilter);
    if (filter) onFilter(filter);
  }

  root.addEventListener("click", onClick);
  return Object.freeze({
    destroy() {
      root.removeEventListener("click", onClick);
    },
  });
}

/** @param {HTMLElement} root @param {SummaryFilter} filter */
export function renderSummaryFilter(root, filter) {
  for (const button of root.querySelectorAll("[data-summary-filter]")) {
    if (button instanceof HTMLElement) {
      button.setAttribute("aria-pressed", String(button.dataset.summaryFilter === filter));
    }
  }
}

/**
 * Wire the section lines in the summary frame: a click reports the term and
 * whether its section was open.
 * @param {HTMLElement} frame @param {(term:string, open:boolean)=>unknown} onToggle
 * @returns {{destroy:()=>void}}
 */
export function bindSummarySections(frame, onToggle) {
  /** @param {MouseEvent} event */
  function onClick(event) {
    const target = event.target instanceof Element ? event.target : null;
    const button = target?.closest("[data-summary-section]");
    if (!(button instanceof HTMLElement) || !frame.contains(button)) return;
    onToggle(button.dataset.summarySection ?? "", button.getAttribute("aria-expanded") === "true");
  }

  frame.addEventListener("click", onClick);
  return Object.freeze({
    destroy() {
      frame.removeEventListener("click", onClick);
    },
  });
}

```

`src/superglm/editor/app/summary.js`, editing it as E1 left it.

Edit E2.7. In `renderSummary`'s unavailable branch, replace:
```js
    renderSearchCount(nodes, "");
    return;
  }
```
with:
```js
    renderSearchCount(nodes, "");
    renderSummaryHeader(payload, nodes, "");
    return;
  }
```

Edit E2.8. Replace:
```js
  updateSummaryMarkup(
    summaryFrame,
    viewModel ? renderCompactSummary(payload, viewModel, view.query) : payload.html || ""
  );
  renderSearchCount(nodes, viewModel ? summaryCountText(viewModel, view.query) : "");
}
```
with:
```js
  const written = updateSummaryMarkup(
    summaryFrame,
    viewModel ? renderCompactSummary(payload, viewModel, view.query) : payload.html || ""
  );
  renderSearchCount(nodes, viewModel ? summaryCountText(viewModel, view.query) : "");
  renderSummaryHeader(payload, nodes, view.query);
  if (written && !view.query.trim()) scrollToCurrentSection(summaryFrame);
}
```

Edit E2.9. Replace:
```js
/**
 * Redraw the summary on show for the inspector's current search. With the
 * compact table in the DOM only its body is rewritten, so an open "Full
 * summary" keeps its frame; otherwise the last payload is rendered again.
 */
export function applySummaryView(nodes) {
```
with:
```js
/**
 * Redraw the summary on show for the inspector's current view: search,
 * filter and open sections. With the compact table in the DOM only its body is
 * rewritten, so an open "Full summary" keeps its frame; otherwise the last
 * payload is rendered again. `follow` scrolls the chart's term into view.
 */
export function applySummaryView(nodes, { follow = false } = {}) {
```

Edit E2.10. Replace:
```js
  renderSearchCount(nodes, summaryCountText(viewModel, view.query));
  // The frame now holds what a full render for this view writes, so a later
  // render of the same payload and view leaves the DOM alone.
  summaryMarkupByFrame.set(summaryFrame, {
    markup: renderCompactSummary(payload, viewModel, view.query),
    firstElementChild: summaryFrame.firstElementChild
  });
}
```
with:
```js
  renderSearchCount(nodes, summaryCountText(viewModel, view.query));
  renderSummaryHeader(payload, nodes, view.query);
  // The frame now holds what a full render for this view writes, so a later
  // render of the same payload and view leaves the DOM alone.
  summaryMarkupByFrame.set(summaryFrame, {
    markup: renderCompactSummary(payload, viewModel, view.query),
    firstElementChild: summaryFrame.firstElementChild
  });
  if (follow) scrollToCurrentSection(summaryFrame);
}

// Follow the chart: bring its term's line into view in the summary frame.
function scrollToCurrentSection(summaryFrame) {
  if (typeof summaryFrame.querySelector !== "function") return;
  const line = summaryFrame.querySelector('tr.summary-section[data-current="true"]:not([hidden])');
  if (line) line.scrollIntoView({ block: "nearest" });
}

// Family, link and method as chips and four figures as tiles, beside Refit
// offsets. The header steps aside while a search narrows the table.
function renderSummaryHeader(payload, nodes, query) {
  const { summaryHeader, summaryModelChips, summaryTiles } = nodes;
  const model = payload.available && payload.compact ? payload.compact.model || {} : null;
  if (summaryModelChips) updateSummaryMarkup(summaryModelChips, model ? renderModelChips(model) : "");
  if (summaryTiles) updateSummaryMarkup(summaryTiles, model ? renderModelTiles(model) : "");
  if (summaryHeader) summaryHeader.hidden = query.trim() !== "";
}

function renderModelChips(model) {
  const link = model.link ? `${model.link} link` : "";
  return [model.family, link, model.method]
    .filter((value) => value !== null && value !== undefined && value !== "")
    .map((value) => `<span class="summary-chip">${escapeHTML(value)}</span>`)
    .join("");
}

function renderModelTiles(model) {
  return [
    ["Deviance", model.deviance],
    ["AIC", model.aic],
    ["BIC", model.bic],
    ["Total EDF", model.effective_df]
  ].map(([label, value]) => `<div class="summary-tile"><span>${escapeHTML(label)}</span><strong title="${escapeHTML(formatFullNumber(value))}">${escapeHTML(formatSummaryValue(value))}</strong></div>`).join("");
}
```

Edit E2.11. `updateSummaryMarkup` now reports whether it wrote. Replace:
```js
function updateSummaryMarkup(summaryFrame, markup) {
  const cached = summaryMarkupByFrame.get(summaryFrame);
  if (
    cached?.markup === markup &&
    cached.firstElementChild === summaryFrame.firstElementChild
  ) return;
  summaryFrame.innerHTML = markup;
  summaryMarkupByFrame.set(summaryFrame, {
    markup,
    firstElementChild: summaryFrame.firstElementChild
  });
}
```
with:
```js
// Whether the markup was written; unchanged markup leaves the DOM alone.
function updateSummaryMarkup(summaryFrame, markup) {
  const cached = summaryMarkupByFrame.get(summaryFrame);
  if (
    cached?.markup === markup &&
    cached.firstElementChild === summaryFrame.firstElementChild
  ) return false;
  summaryFrame.innerHTML = markup;
  summaryMarkupByFrame.set(summaryFrame, {
    markup,
    firstElementChild: summaryFrame.firstElementChild
  });
  return true;
}
```

Edit E2.12. In `renderCompactSummary`, replace:
```js
  const hasLevelGroups = compact.has_level_groups === true;
  const facts = [
    ["Family", model.family],
    ["Link", model.link],
    ["Method", model.method],
    ["Total EDF", model.effective_df],
    ["Deviance", model.deviance],
    ["AIC", model.aic],
    ["BIC", model.bic],
    ["Log lik", model.log_likelihood]
  ];
```
with:
```js
  const hasLevelGroups = compact.has_level_groups === true;
  // Family, link, method and the four headline figures are the header's
  // (renderSummaryHeader); a profiled distribution parameter stays here.
  const facts = [];
```

Edit E2.13. Replace:
```js
    <div class="compact-summary">
      <div class="summary-facts">
        ${facts.map(([label, value]) => renderSummaryFact(label, value)).join("")}
      </div>
```
with:
```js
    <div class="compact-summary">
      ${facts.length ? `<div class="summary-facts">
        ${facts.map(([label, value]) => renderSummaryFact(label, value)).join("")}
      </div>` : ""}
```

Edit E2.14. In E1's `renderSummaryBody`, replace:
```js
  const columnCount = hasLevelGroups ? 8 : 7;
  return viewModel.sections.map((section) => {
    const groupRow = section.header
      ? `<tr class="summary-group-row" data-term="${escapeHTML(section.term)}"${section.hidden ? " hidden" : ""}><td colspan="${columnCount}">${highlightMatches(section.label, query)}</td></tr>`
      : "";
```
with:
```js
  const columnCount = hasLevelGroups ? 8 : 7;
  const empty = viewModel.empty
    ? `<tr class="summary-empty-row"><td colspan="${columnCount}">No terms match.</td></tr>`
    : "";
  return empty + viewModel.sections.map((section) => {
    const groupRow = section.header ? renderSectionHeader(section, columnCount, query) : "";
```

Edit E2.15. Insert immediately before `function renderSummaryFact(label, value) {`:
```js
// A term's line: what it is and how it fits, folded or open. Its button opens
// or closes the rows under it.
function renderSectionHeader(section, columnCount, query) {
  const term = escapeHTML(section.term);
  const kind = section.kind
    ? `<span class="summary-section-kind">${escapeHTML(section.kind)}</span>`
    : "";
  const waiting = section.waiting > 0
    ? `<span class="summary-waiting">${section.waiting} waiting</span>`
    : "";
  const edf = section.edf === null
    ? ""
    : `<span class="summary-section-edf">EDF ${escapeHTML(formatSummaryValue(section.edf))}</span>`;
  return `<tr class="summary-group-row summary-section" data-term="${term}" data-current="${section.current}"${section.hidden ? " hidden" : ""}><td colspan="${columnCount}"><button type="button" class="summary-section-toggle" data-summary-section="${term}" aria-expanded="${section.open}"><svg class="summary-chevron" viewBox="0 0 16 16" aria-hidden="true"><path d="m6 4 4 4-4 4"></path></svg><span class="summary-section-name">${highlightMatches(section.label, query)}</span>${kind}${waiting}<span class="summary-section-fill"></span>${edf}${renderPChip(section.chip)}</button></td></tr>`;
}

// The p-value on a term's line: the whole-term test's, or the smallest of its
// levels', labelled "min" so it is not read as a test of the term.
function renderPChip(chip) {
  if (!chip) return "";
  const text = `${chip.smallest ? "min " : ""}${formatP(chip.p)}${chip.sigCode ? ` ${chip.sigCode}` : ""}`;
  const title = chip.smallest
    ? "Smallest p-value among this term's levels"
    : "p-value of the whole-term test";
  return `<span class="summary-p-chip ${safeSigClass(chip.sigClass)}" title="${escapeHTML(title)}">${escapeHTML(text)}</span>`;
}

```

`src/superglm/editor/app/main.js`, editing it as E1 left it.

Edit 1. Replace `import { bindSummarySearch } from "./views/summary_view.js";` with:
```js
import {
  bindSummaryFilter,
  bindSummarySearch,
  bindSummarySections,
  renderSummaryFilter,
  waitingCounts
} from "./views/summary_view.js";
```

Edit 2. Replace:
```js
const summarySearchCount = document.getElementById("summarySearchCount");
let summaryQuery = "";
```
with:
```js
const summarySearchCount = document.getElementById("summarySearchCount");
const summaryFilterNode = document.getElementById("summaryFilter");
const summaryHeader = document.getElementById("summaryHeader");
const summaryModelChips = document.getElementById("summaryModelChips");
const summaryTiles = document.getElementById("summaryTiles");
let summaryQuery = "";
let summaryFilter = "all";
// Sections the analyst opened or closed; cleared when the chart's term or the
// search changes, so the summary goes back to following the chart.
const summaryToggled = new Map();
```

Edit 3. Replace:
```js
    summaryFrame,
    summarySearchCount,
    summaryView
  };
}
```
with:
```js
    summaryFrame,
    summarySearchCount,
    summaryHeader,
    summaryModelChips,
    summaryTiles,
    summaryView
  };
}
```

Edit 4. Replace E1's `summaryView`:
```js
function summaryView() {
  const snapshot = store.getState().remote.snapshot;
  return {
    query: summaryQuery,
    termNames: snapshot ? Object.keys(snapshot.terms) : []
  };
}
```
with:
```js
function summaryView() {
  const state = store.getState();
  const snapshot = state.remote.snapshot;
  const terms = snapshot ? snapshot.terms : {};
  const names = Object.keys(terms);
  return {
    query: summaryQuery,
    termNames: names,
    filter: summaryFilter,
    currentTerm: selectActiveTermName(state),
    toggled: summaryToggled,
    edited: names.filter((name) => terms[name].edited === true),
    waiting: waitingCounts(snapshot?.pending ?? []),
    kinds: Object.fromEntries(
      names.map((name) => [name, terms[name].term_type || terms[name].kind || ""])
    )
  };
}

// What the summary reads from the state besides its payload: which terms
// carry hand edits and which wait for a refit.
function selectSummaryMarks(state) {
  const snapshot = state.remote.snapshot;
  if (!snapshot) return "";
  const edited = Object.keys(snapshot.terms).filter((name) => snapshot.terms[name].edited === true);
  return JSON.stringify([edited, waitingCounts(snapshot.pending ?? [])]);
}
```

Edit 5. Replace:
```js
bindSummarySearch(summarySearch, (query) => {
  summaryQuery = query;
  applySummaryView(summaryNodes());
});
```
with:
```js
bindSummarySearch(summarySearch, (query) => {
  summaryQuery = query;
  summaryToggled.clear();
  applySummaryView(summaryNodes());
});
bindSummaryFilter(summaryFilterNode, (filter) => {
  summaryFilter = filter;
  renderSummaryFilter(summaryFilterNode, filter);
  applySummaryView(summaryNodes());
});
bindSummarySections(summaryFrame, (term, open) => {
  summaryToggled.set(term, !open);
  applySummaryView(summaryNodes());
});
```

Edit 6. Replace
`store.subscribe(selectSelectionState, renderSelectionState, sameSelectionState);`
with:
```js
store.subscribe(selectSelectionState, renderSelectionState, sameSelectionState);
store.subscribe(selectActiveTermName, () => {
  summaryToggled.clear();
  applySummaryView(summaryNodes(), { follow: true });
});
store.subscribe(selectSummaryMarks, () => applySummaryView(summaryNodes()));
```

`src/superglm/editor/app/index.html`. Two edits.

Edit 1. Replace:
```html
          <span id="summarySearchCount" class="summary-search-count" aria-live="polite"></span>
        </label>
        <div class="summary-controls">
```
with:
```html
          <span id="summarySearchCount" class="summary-search-count" aria-live="polite"></span>
        </label>
        <div class="summary-view-row">
          <div id="summaryFilter" class="summary-filter" role="group" aria-label="Show terms">
            <button type="button" data-summary-filter="all" aria-pressed="true">All</button>
            <button type="button" data-summary-filter="edited" aria-pressed="false">Edited</button>
            <button type="button" data-summary-filter="waiting" aria-pressed="false">Waiting</button>
          </div>
          <span class="summary-follow" tabindex="0" data-popover-title="Follows the chart"
            data-popover-body="The chart's term opens here and scrolls into view; the others fold to one line. Open any of them until the chart shows another term.">
            <svg class="summary-follow-icon" viewBox="0 0 16 16" aria-hidden="true">
              <path d="M8 14.5s4.5-4.6 4.5-8a4.5 4.5 0 0 0-9 0c0 3.4 4.5 8 4.5 8z"></path>
              <circle cx="8" cy="6.5" r="1.5"></circle>
            </svg>
            Follows the chart
          </span>
        </div>
        <div id="summaryHeader" class="summary-header">
          <div class="summary-header-row">
            <div id="summaryModelChips" class="summary-model-chips"></div>
            <button id="refitOffset" type="button">Refit offsets</button>
            <button id="reprofileTweedie" type="button" hidden>Re-profile p</button>
            <button id="reprofileNb2" type="button" hidden>Re-estimate theta</button>
          </div>
          <div id="summaryTiles" class="summary-tiles"></div>
        </div>
        <div class="summary-controls">
```

Edit 2. Replace:
```html
          </fieldset>
          <button id="refitOffset" type="button">Refit offsets</button>
          <button id="reprofileTweedie" type="button" hidden>Re-profile p</button>
          <button id="reprofileNb2" type="button" hidden>Re-estimate theta</button>
        </div>
```
with:
```html
          </fieldset>
        </div>
```

`src/superglm/editor/app/styles.css`. Three edits.

Edit 1. Append after E1's `.summary-frame mark { … }` rule:
```css
/* All / Edited / Waiting, and the note that the summary follows the chart. */
.summary-view-row {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 8px;
}
.summary-filter {
  display: inline-flex;
  gap: 2px;
  padding: 2px;
  border-radius: var(--radius-md);
  background: var(--surface-hover);
}
.summary-filter button {
  height: 24px;
  padding: 0 10px;
  border: 0;
  border-radius: var(--radius-sm);
  background: transparent;
  color: var(--muted);
  font-size: 12px;
}
.summary-filter button[aria-pressed="true"] {
  background: var(--surface);
  color: var(--text);
  font-weight: 600;
  box-shadow: 0 1px 2px var(--shadow);
}
.summary-follow {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  margin-left: auto;
  color: var(--muted);
  font-size: 11.5px;
}
.summary-follow-icon {
  width: 13px;
  height: 13px;
  fill: none;
  stroke: currentColor;
  stroke-width: 1.5;
}
/* The model header: chips with Refit offsets beside them, then four tiles. */
.summary-header {
  display: grid;
  gap: 6px;
  margin-bottom: 8px;
}
.summary-header[hidden] {
  display: none;
}
.summary-header-row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px;
}
.summary-model-chips {
  display: flex;
  flex: 1 1 auto;
  flex-wrap: wrap;
  gap: 6px;
  min-width: 0;
}
.summary-chip {
  display: inline-flex;
  align-items: center;
  height: 22px;
  padding: 0 8px;
  border-radius: 11px;
  background: var(--surface);
  color: var(--text);
  font-size: 12px;
  white-space: nowrap;
}
.summary-tiles {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 6px;
}
.summary-tile {
  display: grid;
  min-width: 0;
  padding: 6px 8px;
  border-radius: var(--radius-md);
  background: var(--surface);
}
.summary-tile span {
  color: var(--muted);
  font-size: 11px;
}
.summary-tile strong {
  overflow: hidden;
  font-family: var(--font-mono);
  font-size: 13px;
  font-weight: 500;
  text-overflow: ellipsis;
  white-space: nowrap;
}
```

Edit 2. Replace:
```css
.summary-group-row + .summary-row td {
  border-top: 0;
}
```
with:
```css
.summary-group-row + .summary-row td {
  border-top: 0;
}
/* A term's line in the summary: its whole row is the open/close button. */
.summary-section td {
  padding: 4px 0 2px;
}
.summary-section-toggle {
  display: flex;
  align-items: center;
  gap: 6px;
  width: 100%;
  height: auto;
  padding: 4px 6px;
  border: 0;
  border-radius: var(--radius-md);
  background: transparent;
  color: var(--text);
  font-size: 12px;
  font-weight: 400;
  text-align: left;
}
.summary-section[data-current="true"] .summary-section-toggle {
  background: var(--surface);
  box-shadow: inset 2px 0 0 var(--blue);
}
.summary-chevron {
  flex: none;
  width: 12px;
  height: 12px;
  fill: none;
  stroke: var(--muted);
  stroke-width: 1.8;
  stroke-linecap: round;
  stroke-linejoin: round;
}
.summary-section-toggle[aria-expanded="true"] .summary-chevron {
  transform: rotate(90deg);
}
/* Name and kind give way, with an ellipsis, so the EDF and chips stay whole. */
.summary-section-name,
.summary-section-kind {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.summary-section-name {
  font-weight: 600;
}
.summary-section-kind {
  flex-shrink: 4;
}
.summary-section-kind,
.summary-section-edf {
  color: var(--muted);
  font-size: 11px;
  white-space: nowrap;
}
.summary-section-edf {
  font-family: var(--font-mono);
}
.summary-section-fill {
  flex: 1 1 auto;
}
.summary-waiting,
.summary-p-chip {
  padding: 1px 6px;
  border-radius: 9px;
  font-size: 10.5px;
  white-space: nowrap;
}
.summary-waiting {
  background: var(--sig-weak-bg);
  color: var(--sig-weak-fg);
}
.summary-p-chip {
  font-family: var(--font-mono);
}
.summary-empty-row td {
  padding: 10px 6px;
  color: var(--muted);
  text-align: left;
}
```

Edit 3. Replace:
```css
.se-cell.sig-strong { background: var(--sig-strong-bg); color: var(--sig-strong-fg); }
.se-cell.sig-medium { background: var(--sig-medium-bg); color: var(--sig-medium-fg); }
.se-cell.sig-standard { background: var(--sig-standard-bg); color: var(--sig-standard-fg); }
.se-cell.sig-weak { background: var(--sig-weak-bg); color: var(--sig-weak-fg); }
.se-cell.sig-none { background: var(--sig-none-bg); color: var(--sig-none-fg); }
.se-cell.sig-unknown { background: var(--surface-hover); color: var(--muted); }
```
with:
```css
.se-cell.sig-strong, .summary-p-chip.sig-strong { background: var(--sig-strong-bg); color: var(--sig-strong-fg); }
.se-cell.sig-medium, .summary-p-chip.sig-medium { background: var(--sig-medium-bg); color: var(--sig-medium-fg); }
.se-cell.sig-standard, .summary-p-chip.sig-standard { background: var(--sig-standard-bg); color: var(--sig-standard-fg); }
.se-cell.sig-weak, .summary-p-chip.sig-weak { background: var(--sig-weak-bg); color: var(--sig-weak-fg); }
.se-cell.sig-none, .summary-p-chip.sig-none { background: var(--sig-none-bg); color: var(--sig-none-fg); }
.se-cell.sig-unknown, .summary-p-chip.sig-unknown { background: var(--surface-hover); color: var(--muted); }
```

`src/superglm/editor/app/views/help_content.js`. In E1's "Summary" section,
replace:
```js
      "The search box at the top of Summary keeps the terms and levels whose names contain the text, ignoring case, and counts what it found. Escape clears it. The full summary below the table is not searched.",
    ]),
```
with:
```js
      "The search box at the top of Summary keeps the terms and levels whose names contain the text, ignoring case, and counts what it found. Escape clears it. The full summary below the table is not searched.",
      "All, Edited and Waiting show every term, the terms with hand edits, or the terms with changes waiting for refit.",
      "Summary follows the chart: the chart's term opens and scrolls into view, and the others fold to one line with their kind, EDF and p-value. A categorical term has no whole-term test, so its line shows the smallest p-value among its levels, marked min. Open any term from its line until the chart shows another; a search opens every term it finds.",
    ]),
```

- [ ] **Step 4: Run the tests, expect PASS**

```bash
./.venv/bin/python -m pytest tests/test_editor_structure.py -q -k "payload"
./.venv/bin/python -m pytest tests/test_editor.py -q -k "summary or payload"
node --test tests/editor_frontend/summary_view.test.js tests/editor_frontend/summary.test.js
npm run check:frontend
./.venv/bin/python -m pytest tests/editor -m browser --run-browser -q -n 6
./.venv/bin/python -m pytest tests/test_editor_browser.py -m browser --run-browser -q
uv run ruff check src/superglm/editor/payloads.py tests/test_editor_structure.py tests/editor/test_editor_workspace_browser.py
uv run ruff format --check src/superglm/editor/payloads.py tests/test_editor_structure.py tests/editor/test_editor_workspace_browser.py
```

- `test_term_change_does_not_rewrite_unrelated_editor_panels` must stay green. A
  term change now redraws the summary, but only its `tbody`.
- The three `summary.test.js` Tweedie tests must stay green ("[1.4, 1.7]",
  "censored", "not computed"). Tweedie p, its CI and NB2 theta stay in the frame.

- [ ] **Step 5: Commit**

```bash
git add src/superglm/editor/payloads.py src/superglm/editor/app/api/contracts.js \
  src/superglm/editor/app/views/summary_view.js src/superglm/editor/app/summary.js \
  src/superglm/editor/app/main.js src/superglm/editor/app/index.html \
  src/superglm/editor/app/styles.css src/superglm/editor/app/views/help_content.js \
  tests/test_editor_structure.py tests/editor_frontend/summary_view.test.js \
  tests/editor_frontend/summary.test.js tests/editor/test_editor_workspace_browser.py
git commit -m "Editor inspector: follows the chart, All/Edited/Waiting, chips and tiles

The chart's term is open and scrolled into view; other terms fold to one line
(name, kind, EDF, p chip, waiting chip) and can be opened until the chart
moves. A categorical's chip is the smallest level p, marked min. All, Edited
and Waiting filter the terms; the state payload marks each term edited from
EditorSession.edited_terms(). The header shows family, link and method as
chips, four tiles and Refit offsets beside them.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Risks and follow-ups for this section

**Open question for Max: the categorical p chip.** Neither the mock nor the spec
says what p-value a categorical's folded line shows. The compact summary has no
whole-term test for a categorical (`inference/summary.py` puts group-level Wald
tests on spline and piecewise rows only). E2 shows the smallest level p,
labelled `min`, with an explanatory title. The alternative is no chip on
categoricals.

**Pre-existing items, left alone:**
- `togglePointSelection` (`interactions.js:355-360`) is dead code. It stays,
  as a follow-up.
- `tests/editor/test_editor_workspace_browser.py` and its siblings click twice
  in a row with the same race that needed `mutationStatus`, for example
  `tests/editor/test_editor_refit_browser.py:157-166`. They could adopt
  `_click_selects`.

**For other tasks:**
- A5 / I1:
  - the inspector's waiting chip uses `--sig-weak-bg` / `--sig-weak-fg`; if A5
    defines dedicated waiting tokens, switch `.summary-waiting` to them;
  - I1's warm dark rewrite must keep every `--sig-*` token: the p chip and the
    SE cells share them.
- Z1:
  - the tutorial line about "a continuous run of points" (C2);
  - the help sections "Selecting points" and "Summary" should be mirrored in
    `docs/tutorials/edit-a-model-in-the-browser.md`.


## Section S5 — Ordered categorical spline (D) and rating-table preview (K)

Workstreams §4 D and §4 K of the design spec. Four tasks:

| Task | Delivers | Depends on |
|---|---|---|
| D1 | backend: `spline_view`, handles from the fitted coefficients, moves that set level effects to `B(level positions)·c` | nothing in this plan |
| D2 | frontend: the spline drawn between levels, handles, Contrib/Build on the grid, disabled-with-reason Handles | D1 |
| K1a | backend: `POST /rating_table` and `editor/rating_preview.py` | nothing (uses A4's route pattern only) |
| K1b | frontend: Chart/Table switch and the table view | K1a |

D2 and K1b both edit `main.js`, `index.html`, `api/contracts.js`, `views/help_content.js` and `styles/shell.css`: run them one at a time (any order). Each edit below is anchored on text present at 155832e8; if A4/A5/F1 landed first and moved a line, apply the same edit to the moved text.

**Validation record.** Every step of this section was executed on a scratch copy of the worktree at 00a24891 (= 155832e8 + spec docs). With the code below: the 11 D1 tests, 6 K1a tests, 12 new node tests and 3 new browser tests pass; each fails on 155832e8 as stated in its Step 2; four mutations of D1 (listed in Task D1) each fail a named test; `tests/test_ordered_categorical_specials_editor.py`, `tests/test_piecewise_editor.py`, `tests/test_editor_structure.py`, `tests/test_editor.py` (minus the notebook test, which needs `docs/`), `tests/test_editor_security.py`, `tests/test_editor_validation_errors.py`, `tests/test_editor_evidence.py`, all 197 node tests (node 22, TypeScript 7.0.2 as pinned in `package-lock.json`), `tsc -p jsconfig.json`, and the 88 browser tests (`tests/test_editor_browser.py tests/editor`, in that order) pass; ruff check/format clean. The combined patch is kept beside this file as `S5_validated.patch` for cross-checking only; the code in the tasks is authoritative.

### Contract amendments

1. **Per-term payload `spline_view`** (new key on every term; `null` unless the term is an `OrderedCategorical` whose basis is a spline):

   ```text
   spline_view: null | {
     available: boolean,            // the spec's `basis: "spline"` flag
     reason: string | null,         // fixed sentence when available is false
     x: number[] | null,            // drawing grid on the display axis; level i at x = i,
                                    // ORDERED_SPLINE_GRID_STEPS = 24 points per gap, G = 24 (S-1) + 1
     y: number[] | null,            // current spline exp(B(t) @ c) on the grid (relativity)
     original_y: number[] | null,   // in-force fit's spline on the grid; null once a structural
                                    // step has replaced the opened model (then the chart joins
                                    // the original levels as today)
     level_indices: number[] | null,// display indices of the smooth levels, 0..S-1 (specials follow)
     fits_levels: boolean           // the edited smooth levels lie on `y`
   }
   ```

   The spec's flag `basis: "spline"` is carried as `spline_view.available === true`; no top-level `basis` key is added, because `controls.basis` already means the handle basis rows.
2. **`controls` for an ordered spline term** keeps every existing key and adds `grid_x: number[]` (= `spline_view.x`). `basis` rows run over the term's `n_points` (smooth levels, then specials at 0), so the existing drag preview moves the level dots unchanged. `build_basis` / `build_log_effect` cover only the *live* basis columns (positive mass on the grid), sampled on `grid_x`; `basis_index[i]` indexes `build_basis` rows. Numeric spline and piecewise `controls` are unchanged (no `grid_x`).
3. **`POST /rating_table`** returns `{term, available, reason, columns, rows, formats, note, model_revision}`: `formats: (string | null)[]` (the Excel number format the workbook gives each column) and `model_revision: int` (the revision the block was built from) are added to the header's contract. `rows` cells are `string | number | boolean | null`; `note` is the note the workbook writes above that block (piecewise and ppform blocks) or `null`.
4. **Frontend state and types**: `EditorViewState.termView: "chart" | "table"` (default `"chart"`); typedefs `TermView`, `SplineView`, `RatingTableResponse` in `api/contracts.js`; `TermPayload.spline_view?: SplineView | null`. `api/client.js`: `createEditorClient()` returns `ratingTable(term)` alongside `getState`.
5. **Python**: `superglm.editor.controls` gains `ORDERED_SPLINE_GRID_STEPS`, the four `ORDERED_SPLINE_*` sentences, `OrderedSplineGeometry`, `ordered_spline_geometry`, `least_change_coefficients`, `ordered_handle_columns`, `ordered_control_points`, `ordered_control_after_move`, `spline_fits_levels`. `EditorSession` gains `ordered_spline(term) -> OrderedSplineGeometry | str | None` and `ordered_spline_coefficients(term, geometry) -> NDArray`. `superglm.editor.rating_preview` is new (`PREVIEW_IMPACT_BINS`, `RatingPreview`, `term_rating_table`, `refusal_reason`, four sentences).
6. **`renderToolRail(root, {mode, handlesAvailable, handlesReason?})`**: with a reason, Handles is `aria-disabled="true"` (not `disabled`) and carries `data-popover-title/body`, so the hover reason shows; without one it stays `disabled` as today.
7. **`CONTROL_HANDLE_TERM_TYPES` stays `("spline", "piecewise")`.** The spec says the tuple should "accept the flagged term"; adding `"ordered categorical"` would also admit ordered terms with a Piecewise or Polynomial basis, which have no handles. Both callers (`_require_control_term`, `session_payload`) ask `ordered_spline_geometry` instead, and the tuple's comment says so.

### Research gate (stages 1–3)

**Characterisation.** An ordered term with a spline basis is a univariate spline `f(t) = Σ_j c_j B_j(t)` evaluated at S level positions `t_i` (`_level_to_value`: `linspace(0, 1)` or the user's `values=`), plus free special levels with no position. The handles are the control coefficients `c_j` (a B-spline's control polygon). With `n_knots` clamped to `S−1` the basis usually has `K > S` columns, so the level values do not determine `c`: recovering `c` from levels is an underdetermined linear system and needs a selection rule; with `K < S` (e.g. `Spline(k=5)` over six bands) it is overdetermined.

**What we adopt and what it assumes.**
- Handle values are the fitted coefficients themselves: `c = M β_spline − (b_base · M β_spline)·1`, where `M` is the map `transform` applies (`R_inv`, or the SCOP `Σ[:, null_dim:]`). Subtracting a constant from every coefficient subtracts it from the curve because a B-spline (and a cardinal natural-spline) basis is a partition of unity — measured: signed row sums 1 ± 2e-16 for `ps`, `bs`, `ns`, `cr`, `cr_cardinal`. The result is certified, not assumed: the curve at the level positions must reproduce the reported level effects within a float64 bound (Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., 2002, §§3.1, 3.5); a fit that fails is refused with a fixed sentence. Measured over 15 fits (five kinds, `values=`, specials, `select=True`, REML, extend, two monotone fits): residual/bound ≤ 0.009.
- After an edit, the coefficients are the **least change** from the last handle state that reproduces the edited levels: `c = c_prev + B⁺(e − B c_prev)`, the minimum-norm correction numpy's `lstsq` (LAPACK `gelsd`) returns. With `K ≥ S` and full row rank it interpolates every edit; with `K < S` it is the least-squares spline. `gelsd` is backward stable like Householder QR (Higham 2002, Thm 20.3), which the tests' residual bounds use. The coefficients written by a handle move are kept in the edit record, so Undo/Redo restore the exact handle state.
- Handle x: the basis-weighted centre of each column on the display grid — the rule numeric splines already use first (`_basis_support_centers`, `controls.py:182-192`). The B-spline collocation kernel is totally positive (de Boor, "Total positivity of the spline collocation matrix", *Indiana Univ. Math. J.* 25 (1976) 541–551), so the centres are non-decreasing. Greville abscissae (the usual control sites) were measured and rejected as the fallback: a P-spline's open knot vector puts its end columns' abscissae outside the level range, which stacked two handles on each end level of the browser fixture's `age_band` (x = 0, 0, 2.5, 5, 5; centres give 0.5, 1.17, 2.5, 3.83, 4.5).
- Rating preview: openpyxl (MIT, read) writes a float as `"%.16g"`, so workbook cells hold 16 significant digits, not the float64 round-trip; the preview sends the block's float64 values and the workbook's own number formats, and its test compares to `5e-16|v| + u|cell|`.

**New territory, stated:** none. Both pieces apply the standard methods above.

**Cost measured (not asserted in tests).** Rating payload on the 678,013-row freMTPL2 book (5 terms, 2 splines, 4 BLAS threads): 6.96 s as the export builds it, 1.49 s with `impact_bins=()`; blocks identical (`DataFrame.equals` on every block). A single-term `discretization_impact` (0.84 s) saves little over two terms (0.98 s), so the preview builds the whole payload once per model revision rather than re-deriving one block. `session_payload` on a 60k-row fit with one 9-level ordered spline: 0.73 ms → 1.53 ms process time per state, single thread; geometry 0.2 ms.

---

### Task D1: Ordered spline geometry, handles and `spline_view` (backend)

**Files:**
- Modify: `src/superglm/editor/controls.py` (imports 3-9, `CONTROL_HANDLE_TERM_TYPES` block 11-15, append after `_as_dense_matrix` 238-241)
- Modify: `src/superglm/editor/session.py` (imports 23-24, `control_points` 574-577, `move_control_point` 579-611, before `undo` 613, `_require_control_term` 1396-1402)
- Modify: `src/superglm/editor/payloads.py` (import 13, `session_payload` 34-48, `_controls_payload` 317-349)
- Test: `tests/test_editor_ordered_spline.py` (new)

**Interfaces:**
- Consumes: `OrderedCategorical._basis_spline`, `._smooth_levels`, `._level_to_value`, `._base_level`, `._grouping`, `._split_beta`, `.basis_kind` (`features/ordered_categorical.py:746-777, 944-953, 1482`); inner spline `._basis_matrix`, `.polynomial_ranges`, `._R_inv`, `._scop_Sigma`, `._scop_null_dim`, `._scop_col_means`; `model.result.beta`, `model._groups`; `controls._control_basis_indices`, `_control_handle_limits`, `_as_dense_matrix`; `EditorSession._commit`.
- Produces:
  ```python
  # superglm/editor/controls.py
  ORDERED_SPLINE_GRID_STEPS: int = 24
  ORDERED_SPLINE_GROUPED, ORDERED_SPLINE_SHAPED, ORDERED_SPLINE_POSITIONS, ORDERED_SPLINE_UNAVAILABLE: str
  @dataclass(frozen=True)
  class OrderedSplineGeometry:
      level_index: NDArray[np.intp]; level_basis: NDArray; grid_x: NDArray; grid_basis: NDArray
      handle_x: NDArray; live: NDArray[np.intp]; fitted: NDArray; n_points: int
  def ordered_spline_geometry(model, term: EditableTerm) -> OrderedSplineGeometry | str | None
  def least_change_coefficients(geometry, effects, prior=None) -> NDArray
  def ordered_handle_columns(geometry, n_handles: int | None = None) -> NDArray[np.intp]
  def ordered_control_points(geometry, coefficients, *, n_handles=None) -> dict
  def ordered_control_after_move(geometry, coefficients, handle_index, log_effect, *, n_handles=None) -> tuple[NDArray, int]
  def spline_fits_levels(geometry, term, records) -> bool
  # EditorSession
  def ordered_spline(self, term: str) -> OrderedSplineGeometry | str | None
  def ordered_spline_coefficients(self, term: str, geometry: OrderedSplineGeometry) -> NDArray
  # history: a handle move on an ordered spline commits EditRecord(operation="control_point",
  # indices=smooth levels, params={"handle_index", "log_effect", "basis": "ordered_spline",
  # "basis_index": raw column, "x", "coefficients": list[float]})
  # payload: term["spline_view"] (amendment 1), term["controls"] for ordered splines (amendment 2)
  ```

- [ ] **Step 1: Write the failing test**

Create `tests/test_editor_ordered_spline.py`:

```python
"""Handles, Contrib and Build for an OrderedCategorical with a spline basis.

The handles are the fitted spline's own coefficients, and moving one sets the
smooth levels to ``B(level positions) @ c``.  Every tolerance below is written
in the unit roundoff ``u = 2**-53`` and ``gamma_k = k u / (1 - k u)`` (Higham,
*Accuracy and Stability of Numerical Algorithms*, 2nd ed., 2002, sections 2.2,
3.1 and 3.5), times the magnitudes the computation actually combines.
"""

from __future__ import annotations

import json
import urllib.request

import numpy as np
import pandas as pd
import pytest

from superglm import OrderedCategorical, Piecewise, Spline, SuperGLM
from superglm.editor import EditorSession
from superglm.editor.controls import (
    ORDERED_SPLINE_GRID_STEPS,
    ORDERED_SPLINE_GROUPED,
    ORDERED_SPLINE_SHAPED,
)
from superglm.editor.payloads import session_payload

U = 2.0**-53
SMOOTH = ["1", "2", "3", "4", "5", "6"]
EFFECT = {"1": -0.30, "2": -0.18, "3": -0.05, "4": 0.06, "5": 0.15, "6": 0.20, "MISSING": 0.55}


def _gamma(count: int) -> float:
    return count * U / (1.0 - count * U)


def _fit(basis, *, specials=("MISSING",), seed=20261003):
    """A gaussian fit of one ordered band, with a special level when asked."""
    rng = np.random.default_rng(seed)
    labels = rng.choice(SMOOTH + list(specials), 900)
    X = pd.DataFrame({"band": labels})
    y = np.array([EFFECT[label] for label in labels]) + rng.normal(0.0, 0.15, 900)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={
            "band": OrderedCategorical(order=SMOOTH, specials=list(specials) or None, basis=basis)
        },
    )
    model.fit(X, y)
    return model, X


@pytest.fixture
def wide():
    """Eight basis columns over six levels: the levels do not fix the coefficients."""
    return _fit(Spline(kind="ps", k=8))


@pytest.fixture
def narrow():
    """Five basis columns over six levels, like the browser fixture's ``age_band``."""
    return _fit(Spline(kind="ps", k=5), specials=())


def _spline_parts(model):
    """The pieces an independent evaluation of the fitted curve needs."""
    spec = model._specs["band"]
    inner = spec._basis_spline
    beta = np.concatenate(
        [model.result.beta[g.sl] for g in model._groups if g.feature_name == "band"]
    )
    spline_beta, _ = spec._split_beta(beta)
    positions = np.array([spec._level_to_value[level] for level in spec._smooth_levels])
    basis = inner._basis_matrix(positions).toarray()
    base_row = inner._basis_matrix(np.array([spec._level_to_value[spec._base_level]])).toarray()[0]
    assert inner._R_inv is not None, "precondition: an SSP spline without a SCOP map"
    raw = inner._R_inv @ spline_beta
    fitted = raw - base_row @ raw
    weights = np.abs(inner._R_inv) @ np.abs(spline_beta)
    return basis, base_row, fitted, weights, spline_beta.size


def _curve_bound(basis, base_row, weights, p):
    """How far two float64 evaluations of the base-relative curve at a level can differ.

    ``fl(fl(b_i R) beta) - fl(fl(b_0 R) beta)`` against
    ``fl(b_i fl(R beta)) - fl(b_0 fl(R beta))``: each is within
    ``gamma_{K+p+3} (1 + |b_i|_1) (a_i + a_0)`` of the exact value, with
    ``a_i = |b_i| |R| |beta|``.
    """
    k = basis.shape[1]
    level = np.abs(basis) @ weights
    base = float(np.abs(base_row) @ weights)
    row_norm = np.maximum(1.0, np.sum(np.abs(basis), axis=1))
    return 4.0 * _gamma(2 * (k + p) + 6) * row_norm * (level + base)


def _least_squares_bound(basis, start, coefficients, effects):
    """Residual of the least-change solve when the edited levels are a spline of the basis.

    numpy's ``lstsq`` (LAPACK ``gelsd``) works by orthogonal transformations and
    is backward stable like Householder QR (Higham 2002, Thm 20.3), so on a
    consistent system its residual is within ``gamma`` of
    ``|B|_F |delta| + |r_0|``; the second term covers forming ``B @ start``,
    ``B @ c`` and the differences.
    """
    s, k = basis.shape
    correction = np.linalg.norm(coefficients - start)
    start_residual = np.linalg.norm(effects - basis @ start)
    solve = 4.0 * _gamma(4 * s * k) * (np.linalg.norm(basis) * correction + start_residual)
    forming = np.abs(basis) @ np.abs(coefficients) + np.abs(effects)
    return solve + 4.0 * _gamma(k + 1) * np.linalg.norm(forming)


def _smallest_singular_value(basis):
    """The smallest singular value ``lstsq`` keeps (numpy's default ``rcond``)."""
    values = np.linalg.svd(basis, compute_uv=False)
    return float(values[values > values[0] * max(basis.shape) * np.finfo(np.float64).eps][-1])


@pytest.mark.parametrize("fixture", ["wide", "narrow"])
def test_ordered_spline_handles_start_at_the_fitted_coefficients(fixture, request):
    # Mutation check: recovering the coefficients by least squares on the six
    # level points, as a numeric spline does on its grid, returns the
    # minimum-norm vector instead. On `wide` (eight columns) that misses the
    # fitted coefficients by O(0.1); on master `control_points` refuses the term.
    model, _ = request.getfixturevalue(fixture)
    session = EditorSession.from_model(model, terms=["band"])
    basis, base_row, fitted, weights, p = _spline_parts(model)
    geometry = session.ordered_spline("band")

    controls = session.control_points("band")

    # The session starts from the fit and corrects it by the minimum-norm
    # solution of `B delta = e - B c_fit`, whose right-hand side is the curve's
    # rounding, at most the curve bound in each entry.
    bound = _curve_bound(basis, base_row, weights, p)
    drift = np.linalg.norm(bound) / _smallest_singular_value(basis)
    evaluation = 4.0 * _gamma(2 * (basis.shape[1] + p) + 6) * (weights + float(base_row @ weights))
    live = geometry.live
    np.testing.assert_array_less(
        np.abs(np.asarray(controls["build_log_effect"]) - fitted[live]),
        drift + evaluation[live] + U,
    )
    assert np.all(np.diff(controls["x"]) >= 0.0)
    assert controls["x"][0] >= 0.0 and controls["x"][-1] <= len(SMOOTH) - 1.0


@pytest.mark.parametrize("fixture", ["wide", "narrow"])
def test_spline_view_reproduces_the_fitted_level_effects(fixture, request):
    model, _ = request.getfixturevalue(fixture)
    session = EditorSession.from_model(model, terms=["band"])
    basis, base_row, _, weights, p = _spline_parts(model)
    effects = session.terms["band"].original_log_effect[: len(SMOOTH)]

    view = session_payload(session)["band"]["spline_view"]

    steps = ORDERED_SPLINE_GRID_STEPS
    assert view["available"] is True
    assert view["fits_levels"] is True
    assert len(view["x"]) == steps * (len(SMOOTH) - 1) + 1
    assert view["x"][::steps] == [float(i) for i in range(len(SMOOTH))]
    assert view["level_indices"] == list(range(len(SMOOTH)))
    # exp(curve) at each level against exp(level effect): the curve is within
    # the curve bound of the effect, and each exp rounds within 2u (numpy's exp
    # is within one ulp), so by the mean value theorem the relativities differ
    # by at most exp(e) (2 bound + 8u) while the bound is far below 1.
    expected = np.exp(effects)
    tolerance = expected * (2.0 * _curve_bound(basis, base_row, weights, p) + 8.0 * U)
    for curve in (view["y"], view["original_y"]):
        at_levels = np.asarray(curve)[::steps]
        np.testing.assert_array_less(np.abs(at_levels - expected), tolerance + U)


def test_moving_a_handle_sets_every_smooth_level_to_the_spline(wide):
    model, _ = wide
    session = EditorSession.from_model(model, terms=["band"])
    term = session.terms["band"]
    basis, *_ = _spline_parts(model)
    before = term.edited_log_effect.copy()
    controls = session.control_points("band")
    handle = controls["x"].size // 2
    target = float(controls["log_effect"][handle] + 0.3)

    session.move_control_point("band", handle, target)

    record = session.history[-1]
    moved = np.asarray(record.params["coefficients"])
    assert moved[record.params["basis_index"]] == target
    # Two float64 evaluations of the same dot products b_i . c: each within
    # gamma_K |b_i| |c| of the exact value.
    smooth = term.edited_log_effect[: len(SMOOTH)]
    bound = 2.0 * _gamma(basis.shape[1]) * (np.abs(basis) @ np.abs(moved))
    np.testing.assert_array_less(np.abs(smooth - basis @ moved), bound + U)
    # The special level has no place on the spline: its value is not written.
    np.testing.assert_array_equal(term.edited_log_effect[len(SMOOTH) :], before[len(SMOOTH) :])
    # The next request starts from the moved coefficients, so the handle stays
    # where it was dropped. Mutation check: least change from the FIT instead
    # moves it by 0.3 times the row-space projection's diagonal, O(0.1) here.
    after = session.control_points("band")
    reproduce = np.linalg.norm(bound) / _smallest_singular_value(basis)
    assert abs(after["log_effect"][handle] - target) <= reproduce + 2.0 * U * abs(target)


def test_a_level_edit_keeps_the_spline_through_the_levels_and_a_handle_keeps_the_edit(wide):
    # Mutation check: restarting from the fitted coefficients after a level
    # edit draws a curve that misses level "3" by the 0.2 shift, and the handle
    # move below would then wipe the shift out.
    model, _ = wide
    session = EditorSession.from_model(model, terms=["band"])
    term = session.terms["band"]
    basis, *_ = _spline_parts(model)
    start = session.ordered_spline("band").fitted
    session.select_levels("band", ["3"])
    session.shift("band", 0.2)
    effects = term.edited_log_effect[: len(SMOOTH)].copy()

    view = session_payload(session)["band"]["spline_view"]
    geometry = session.ordered_spline("band")
    current = session.ordered_spline_coefficients("band", geometry)

    assert view["fits_levels"] is True
    bound = _least_squares_bound(basis, start, current, effects)
    assert np.linalg.norm(basis @ current - effects) <= bound

    controls = session.control_points("band")
    handle = controls["x"].size // 2
    column = int(geometry.live[controls["basis_index"][handle]])
    outside = np.flatnonzero(basis[:, column] == 0.0)
    assert outside.size, "precondition: the moved basis function misses some level"
    session.move_control_point("band", handle, float(controls["log_effect"][handle] + 0.3))
    # Off the moved column's support each level is b_i . c again: the edited
    # value up to the least-change residual and one more evaluation.
    moved_bound = bound + 2.0 * _gamma(basis.shape[1]) * float(
        np.max(np.abs(basis) @ np.abs(current))
    )
    np.testing.assert_array_less(
        np.abs(term.edited_log_effect[outside] - effects[outside]), moved_bound + U
    )


def test_handles_are_off_with_a_reason_once_levels_are_grouped(wide):
    model, _ = wide
    session = EditorSession.from_model(model, terms=["band"])
    session.select_levels("band", ["2", "3"])
    session.replace_with_collapsed_levels("band", method="fit")

    payload = session_payload(session)["band"]

    assert payload["controls"] is None
    assert payload["spline_view"]["available"] is False
    assert payload["spline_view"]["reason"] == ORDERED_SPLINE_GROUPED
    with pytest.raises(TypeError, match="grouped"):
        session.control_points("band")


def test_handles_are_off_with_a_reason_once_a_band_is_shaped(wide):
    model, _ = wide
    session = EditorSession.from_model(model, terms=["band"])
    session.replace_with_shaped_range("band", lo="2", hi="4", degree=1, method="fit")

    payload = session_payload(session)["band"]

    assert payload["controls"] is None
    assert payload["spline_view"]["reason"] == ORDERED_SPLINE_SHAPED
    with pytest.raises(TypeError, match="shaped"):
        session.move_control_point("band", 0, 0.0)


def test_an_ordered_term_without_a_spline_basis_gets_no_spline_view():
    model, _ = _fit(Piecewise(breaks=["3"]), specials=())
    session = EditorSession.from_model(model, terms=["band"])

    payload = session_payload(session)["band"]

    assert payload["spline_view"] is None
    assert payload["controls"] is None
    with pytest.raises(TypeError, match="control handles"):
        session.control_points("band")


def test_to_model_moves_predictions_by_exactly_the_handle_edit(wide):
    # The edited levels are what `to_model` consumes (#453): the export moves
    # each smooth level's rows by its level's change, and the special's by none.
    model, X = wide
    session = EditorSession.from_model(model, terms=["band"])
    term = session.terms["band"]
    before = term.edited_log_effect.copy()
    controls = session.control_points("band")
    handle = controls["x"].size // 2
    session.move_control_point("band", handle, float(controls["log_effect"][handle] + 0.3))
    change = term.edited_log_effect - before

    edited = session.to_model()

    delta = np.asarray(edited._predict_eta_exact(X)) - np.asarray(model._predict_eta_exact(X))
    level_of_row = {label: i for i, label in enumerate(term.levels)}
    wanted = change[[level_of_row[str(label)] for label in X["band"]]]
    np.testing.assert_array_less(np.abs(delta - wanted), _projection_bound(model, edited, term, X))


def _projection_bound(model, edited, term, X):
    """Per row, how far the exported prediction change can sit from the edit.

    `_apply_ordered_spline_term` solves the weighted least-squares problem
    ``W^1/2 [1, T] x = W^1/2 t`` on the smooth levels; the targets are a spline
    of the basis, so the system is consistent and the backward-stable solve
    leaves a residual within ``gamma_{4 S n} (|A|_F |x| + |b|)`` (Higham 2002,
    Thm 20.3), divided by the smallest root weight to bound one level. Each
    row's linear predictor then rounds within ``gamma_{p+2}`` of
    ``|a| + |T_row| |beta|``, once per model.
    """
    spec = model._specs["band"]
    groups = [g for g in model._groups if g.feature_name == "band"]
    old_beta = np.concatenate([model.result.beta[g.sl] for g in groups])
    new_beta = np.concatenate([edited.result.beta[g.sl] for g in groups])
    design = np.asarray(spec.transform(X["band"].to_numpy()))
    rows = np.abs(design) @ np.abs(old_beta) + np.abs(design) @ np.abs(new_beta)
    evaluation = _gamma(design.shape[1] + 2) * (
        abs(model.result.intercept) + abs(edited.result.intercept) + rows
    )
    smooth = np.asarray(spec.transform(np.array(spec._smooth_levels, dtype=object)))
    width = spec._split_beta(np.zeros(design.shape[1]))[0].size
    weights = np.maximum(term.weights[: len(SMOOTH)], 1e-12)
    root = np.sqrt(weights)
    A = root[:, None] * np.column_stack([np.ones(len(SMOOTH)), smooth[:, :width]])
    # The solve's intercept is the change less the base shift put back after it.
    shift = spec._base_log_effect(old_beta)
    x = np.concatenate(
        [[edited.result.intercept - model.result.intercept - shift], new_beta[:width]]
    )
    b = root * term.edited_log_effect[: len(SMOOTH)]
    solve = 4.0 * _gamma(4 * A.size) * (np.linalg.norm(A) * np.linalg.norm(x) + np.linalg.norm(b))
    return solve / float(np.min(root)) + 2.0 * evaluation + U


def _post_json(widget, path: str, payload: dict) -> dict:
    request = urllib.request.Request(
        f"{widget.url}{path}",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json", "X-SuperGLM-Editor-Token": widget._token},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def test_widget_moves_an_ordered_spline_handle_over_http(wide):
    model, _ = wide
    session = EditorSession.from_model(model, terms=["band"])
    widget = session.widget()
    try:
        state = _post_json(widget, "/control_count", {"term": "band", "count": 4})
        band = state["terms"]["band"]
        controls = band["controls"]
        assert controls["count"] == 4
        assert len(controls["grid_x"]) == len(band["spline_view"]["x"])
        assert all(len(row) == len(controls["grid_x"]) for row in controls["build_basis"])
        assert all(len(row) == band["n_points"] for row in controls["basis"])
        assert len(controls["build_log_effect"]) == len(controls["build_basis"])

        before = session.terms["band"].edited_log_effect.copy()
        target = float(np.exp(controls["log_effect"][1] + 0.25))
        state = _post_json(widget, "/control", {"term": "band", "handle_index": 1, "value": target})
    finally:
        widget.close()

    after = session.terms["band"].edited_log_effect
    assert np.max(np.abs(after - before)) > 0.0
    assert session.history[-1].params["basis"] == "ordered_spline"
    assert state["terms"]["band"]["controls"]["count"] == 4
```

The bounds, stated: (a) **moved levels** `|e'_i − (B c')_i| ≤ 2 γ_K |b_i|·|c'|` (two evaluations of the same K-term dot product); (b) **curve at the levels** `|f̂(t_i) − e_i| ≤ 4 γ_{2(K+p)+6} max(1, ‖b_i‖₁) (a_i + a_0)`, `a_i = |b_i||M||β|` (two evaluation orders of the base-relative curve, the `γ` count also covering forming `a_i`), then `exp` adds `≤ 2u` relative each side; (c) **least-change residual** `‖B c − e‖₂ ≤ 4 γ_{4SK}(‖B‖_F‖δ‖₂ + ‖r₀‖₂) + 4 γ_{K+1} ‖|B||c| + |e|‖₂`; (d) **handle drift** after the fit or a move: the minimum-norm correction of a right-hand side bounded by (a) or (b), divided by the smallest kept singular value of `B` (a conditioning factor); (e) **export round trip**: the projection residual of (c)'s form on the weighted system, divided by `min √w`, plus `2 γ_{p+2}` of each linear predictor's magnitude.

- [ ] **Step 2: Run it, expect FAIL**

Run: `./.venv/bin/python -m pytest tests/test_editor_ordered_spline.py -q`
Expected on 155832e8: collection error `ImportError: cannot import name 'ORDERED_SPLINE_GRID_STEPS' from 'superglm.editor.controls'`. Behaviourally, master has no `spline_view` key, `session.control_points("band")` raises `TypeError: Term 'band' does not expose spline control handles.`, and the ordered term's `controls` payload is `None` (`payloads.py:325`, `session.py:1398-1401`).

- [ ] **Step 3: Implement**

**3a. `src/superglm/editor/controls.py` — imports and the new types** (replace lines 3-15):

Old:
```python
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from superglm.editor._types import EditableTerm
from superglm.editor.errors import EditorIndexError, EditorTypeError

# Term types whose control handles are recovered from a fitted basis rather than
# drawn as a display-only fallback.  Kept here, where the recovery lives, so the
# two callers that gate on it (`EditorSession._require_control_term` and
# `payloads._controls_payload`) cannot drift apart.
CONTROL_HANDLE_TERM_TYPES = ("spline", "piecewise")
```
New:
```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from superglm.editor._types import EditableTerm
from superglm.editor.errors import EditorIndexError, EditorTypeError

# Term types whose control handles are recovered from a fitted basis rather than
# drawn as a display-only fallback.  Kept here, where the recovery lives, so the
# two callers that gate on it (`EditorSession._require_control_term` and
# `payloads._controls_payload`) cannot drift apart.  An ordered categorical is
# not listed: only one with a spline basis has handles, and
# `ordered_spline_geometry` is the gate both callers ask about it.
CONTROL_HANDLE_TERM_TYPES = ("spline", "piecewise")

# Points per gap between adjacent levels on the grid an ordered spline is drawn
# on; the grid also holds every level position exactly.
ORDERED_SPLINE_GRID_STEPS = 24

# Why an ordered term with a spline basis shows no handles: one fixed sentence
# per reason, shown on the disabled Handles tool.
ORDERED_SPLINE_GROUPED = (
    "Handles are off while levels are grouped. Ungroup them to edit the spline."
)
ORDERED_SPLINE_SHAPED = "Handles are off once a band is shaped. Undo the shape to edit the spline."
ORDERED_SPLINE_POSITIONS = "Handles need each level at its own place on the spline's axis."
ORDERED_SPLINE_UNAVAILABLE = "Handles are not available for this spline."

_UNIT_ROUNDOFF = 2.0**-53


def _gamma(count: int) -> float:
    """Higham's ``gamma_k = k u / (1 - k u)`` for ``k`` roundings, ``u = 2^-53``."""
    product = count * _UNIT_ROUNDOFF
    return product / (1.0 - product) if product < 1.0 else float("inf")


@dataclass(frozen=True)
class OrderedSplineGeometry:
    """An ordered term's fitted spline on its level axis, in display coordinates.

    Smooth level ``i`` sits at display x ``i``; between levels the spline's own
    positions (``linspace(0, 1)`` or the user's ``values=``) map linearly onto
    the display axis.  ``level_basis`` ``(S, K)`` and ``grid_basis`` ``(G, K)``
    are the inner spline's raw basis at the level positions and on a grid of
    ``ORDERED_SPLINE_GRID_STEPS`` points per gap; grid row
    ``ORDERED_SPLINE_GRID_STEPS * i`` is evaluated at level ``i``'s position
    itself.  ``fitted`` are the in-force fit's raw coefficients less the curve
    at the reporting base, so ``level_basis @ fitted`` is the displayed level
    effects: every basis row sums to one, so subtracting a constant from every
    coefficient subtracts it from the curve.  ``live`` lists, in order, the
    columns with positive mass on the grid; only they get handles.
    """

    level_index: NDArray[np.intp]
    level_basis: NDArray
    grid_x: NDArray
    grid_basis: NDArray
    handle_x: NDArray
    live: NDArray[np.intp]
    fitted: NDArray
    n_points: int
```

**3b. `src/superglm/editor/controls.py` — append after `_as_dense_matrix` (end of file, line 241):**

```python


def ordered_spline_geometry(model, term: EditableTerm) -> OrderedSplineGeometry | str | None:
    """The fitted spline of an ordered term with a spline basis, on its level axis.

    ``None`` for every other term.  A fixed sentence instead of the geometry
    when the term's handles are off: a grouping or a shaped band changes what
    the coefficients mean, and a fitted curve that does not reproduce the
    reported level effects to round-off is refused rather than drawn.
    """
    from superglm.features.ordered_categorical import OrderedCategorical

    spec = None if model is None else getattr(model, "_specs", {}).get(term.name)
    if not isinstance(spec, OrderedCategorical) or spec.basis_kind != "spline":
        return None
    if spec._grouping is not None:
        return ORDERED_SPLINE_GROUPED
    inner = spec._basis_spline
    if inner.polynomial_ranges:
        return ORDERED_SPLINE_SHAPED
    smooth = list(spec._smooth_levels)
    if term.levels is None or term.levels[: len(smooth)] != [str(level) for level in smooth]:
        return ORDERED_SPLINE_UNAVAILABLE
    positions = np.asarray([spec._level_to_value[level] for level in smooth], dtype=np.float64)
    if positions.size < 2 or not np.all(np.diff(positions) > 0.0):
        return ORDERED_SPLINE_POSITIONS
    steps = ORDERED_SPLINE_GRID_STEPS
    fractions = np.arange(steps, dtype=np.float64) / steps
    gaps = np.diff(positions)
    grid_positions = np.concatenate(
        [(positions[:-1, None] + gaps[:, None] * fractions).ravel(), positions[-1:]]
    )
    starts = np.arange(positions.size - 1, dtype=np.float64)
    grid_x = np.concatenate([(starts[:, None] + fractions).ravel(), [positions.size - 1.0]])
    base = np.array([spec._level_to_value[spec._base_level]], dtype=np.float64)
    beta = np.concatenate(
        [
            np.asarray(model.result.beta, dtype=np.float64)[group.sl]
            for group in model._groups
            if group.feature_name == term.name
        ]
    )
    spline_beta, _ = spec._split_beta(beta)
    try:
        level_basis = _as_dense_matrix(inner._basis_matrix(positions))
        grid_basis = _as_dense_matrix(inner._basis_matrix(grid_positions))
        base_row = _as_dense_matrix(inner._basis_matrix(base))[0]
    except ValueError:
        # extrapolation="error" and a declared level outside the fitted range.
        return ORDERED_SPLINE_UNAVAILABLE
    coefficient_map = _raw_coefficient_map(inner, spline_beta.size)
    if coefficient_map.shape != (level_basis.shape[1], spline_beta.size):
        return ORDERED_SPLINE_UNAVAILABLE
    raw = coefficient_map @ spline_beta
    fitted = raw - float(base_row @ raw)
    effects = np.asarray(term.original_log_effect, dtype=np.float64)[: positions.size]
    bound = _certification_bound(level_basis, base_row, coefficient_map, spline_beta, inner)
    if not np.all(np.abs(level_basis @ fitted - effects) <= bound):
        return ORDERED_SPLINE_UNAVAILABLE
    centres = _handle_centres(grid_basis, grid_x)
    live = np.flatnonzero(np.isfinite(centres)).astype(np.intp)
    if live.size < 3:
        return ORDERED_SPLINE_UNAVAILABLE
    return OrderedSplineGeometry(
        level_index=np.arange(positions.size, dtype=np.intp),
        level_basis=level_basis,
        grid_x=grid_x,
        grid_basis=grid_basis,
        handle_x=centres,
        live=live,
        fitted=fitted,
        n_points=int(term.size),
    )


def least_change_coefficients(
    geometry: OrderedSplineGeometry,
    effects: NDArray,
    prior: NDArray | list[float] | None = None,
) -> NDArray:
    """The coefficients behind ``effects``: the least change from ``prior`` that reproduces them.

    ``prior`` is the vector the latest handle move wrote, else the fit's.  The
    level points alone cannot say which coefficients they came from -- the
    basis usually has more columns than there are levels -- so the
    minimum-norm correction ``lstsq`` returns keeps every coefficient the
    edits do not need to move.  With no more levels than the basis can
    interpolate, the corrected spline passes through every edited level;
    otherwise it is their least-squares spline.
    """
    start = geometry.fitted if prior is None else np.asarray(prior, dtype=np.float64)
    if start.shape != geometry.fitted.shape:
        start = geometry.fitted
    smooth = np.asarray(effects, dtype=np.float64)[geometry.level_index]
    residual = smooth - geometry.level_basis @ start
    correction = np.linalg.lstsq(geometry.level_basis, residual, rcond=None)[0]
    return np.asarray(start + correction, dtype=np.float64)


def ordered_handle_columns(
    geometry: OrderedSplineGeometry, n_handles: int | None = None
) -> NDArray[np.intp]:
    """The basis columns that carry a handle, thinned like a numeric spline's."""
    return geometry.live[_control_basis_indices(geometry.live.size, n_handles=n_handles)]


def ordered_control_points(
    geometry: OrderedSplineGeometry,
    coefficients: NDArray,
    *,
    n_handles: int | None = None,
) -> dict:
    """Handles for an ordered spline, in the shape ``control_points`` returns.

    ``basis`` rows run over the term's display points (levels, then specials
    at zero), so the browser's drag preview moves the level dots exactly as
    for a numeric spline.  ``build_basis`` rows run over the drawing grid
    ``grid_x``, and ``basis_index`` indexes them.
    """
    handles = ordered_handle_columns(geometry, n_handles)
    coefficients = np.asarray(coefficients, dtype=np.float64)
    basis = np.zeros((handles.size, geometry.n_points), dtype=np.float64)
    basis[:, geometry.level_index] = geometry.level_basis[:, handles].T
    min_handles, max_handles = _control_handle_limits(geometry.live.size)
    return {
        "x": geometry.handle_x[handles].copy(),
        "log_effect": coefficients[handles].copy(),
        "basis_index": np.searchsorted(geometry.live, handles).astype(np.intp),
        "basis": basis,
        "build_basis": np.asarray(geometry.grid_basis[:, geometry.live].T, dtype=np.float64),
        "build_log_effect": coefficients[geometry.live].copy(),
        "grid_x": geometry.grid_x.copy(),
        "min_handles": min_handles,
        "max_handles": max_handles,
    }


def ordered_control_after_move(
    geometry: OrderedSplineGeometry,
    coefficients: NDArray,
    handle_index: int,
    log_effect: float,
    *,
    n_handles: int | None = None,
) -> tuple[NDArray, int]:
    """The coefficients with one handle's set to ``log_effect``, and its column."""
    handles = ordered_handle_columns(geometry, n_handles)
    if handle_index < 0 or handle_index >= handles.size:
        raise EditorIndexError("Control handle index out of range for this term.")
    column = int(handles[handle_index])
    moved = np.asarray(coefficients, dtype=np.float64).copy()
    moved[column] = float(log_effect)
    return moved, column


def spline_fits_levels(geometry: OrderedSplineGeometry, term: EditableTerm, records) -> bool:
    """Whether the spline the coefficients draw passes through the edited levels.

    Decided by construction, not by a tolerance: it does when the basis
    interpolates any level values (full row rank at the level positions), when
    no smooth level is edited, or when the latest edit touching the smooth
    levels was a handle move that wrote all of them.  ``matrix_rank`` uses
    numpy's default cutoff; the answer only picks what the chart draws.
    """
    smooth = geometry.level_index
    if np.linalg.matrix_rank(geometry.level_basis) == smooth.size:
        return True
    if np.array_equal(term.edited_log_effect[smooth], term.original_log_effect[smooth]):
        return True
    for record in reversed(records):
        if record.term != term.name or not np.intersect1d(record.indices, smooth).size:
            continue
        return "coefficients" in record.params and record.indices.size == smooth.size
    return False


def _raw_coefficient_map(inner, width: int) -> NDArray:
    """The matrix taking the fitted coefficients to the raw basis coefficients.

    ``transform`` evaluates ``B @ R_inv`` (or, for a SCOP monotone fit,
    ``(B @ Sigma)[:, null_dim:]`` less a column-mean constant), so the raw
    coefficients are this matrix times beta, up to that constant, which the
    base shift removes.
    """
    sigma = getattr(inner, "_scop_Sigma", None)
    if sigma is not None:
        drop = int(getattr(inner, "_scop_null_dim", 1))
        return np.asarray(sigma, dtype=np.float64)[:, drop:]
    r_inv = getattr(inner, "_R_inv", None)
    if r_inv is None:
        return np.eye(width, dtype=np.float64)
    return np.asarray(r_inv, dtype=np.float64)


def _certification_bound(level_basis, base_row, coefficient_map, beta, inner) -> NDArray:
    """Per level, how far two float64 evaluations of the base-relative curve can differ.

    The reported effect is ``fl(fl(b_i M) beta) - fl(fl(b_0 M) beta)``; ours is
    ``fl(b_i fl(M beta) - fl(b_0 fl(M beta)))``.  Each is within
    ``gamma_{K+p+3} (1 + |b_i|_1) (a_i + a_0)`` of the exact value, where
    ``a_i = |b_i| |M| |beta|`` (Higham 2002, sections 3.1 and 3.5), plus the
    SCOP column-mean constant's magnitude when the fit carries one.  Twice
    that, and a ``gamma`` whose count also covers forming ``a_i`` from
    non-negative terms, keeps the computed bound an upper bound.
    """
    columns = level_basis.shape[1]
    weights = np.abs(coefficient_map) @ np.abs(beta)
    constant = 0.0
    means = getattr(inner, "_scop_col_means", None)
    if getattr(inner, "_scop_Sigma", None) is not None and means is not None:
        constant = float(np.abs(np.asarray(means, dtype=np.float64)) @ np.abs(beta))
    level_magnitude = np.abs(level_basis) @ weights + constant
    base_magnitude = float(np.abs(base_row) @ weights) + constant
    row_norm = np.maximum(1.0, np.sum(np.abs(level_basis), axis=1))
    count = 2 * (columns + beta.size) + 6
    return 4.0 * _gamma(count) * row_norm * (level_magnitude + base_magnitude)


def _handle_centres(grid_basis: NDArray, grid_x: NDArray) -> NDArray:
    """Each column's basis-weighted centre on the display axis; NaN without visible mass.

    The rule a numeric spline's handles use (``_basis_support_centers``), taken
    on the uniform display grid.  The B-spline collocation kernel is totally
    positive (de Boor, "Total positivity of the spline collocation matrix",
    Indiana Univ. Math. J. 25, 1976), so these centres never decrease from one
    column to the next; a cardinal basis's negative lobes are left out of the
    mass.  Greville abscissae are not the fallback here: a P-spline's open knot
    vector puts its end columns' abscissae outside the level range, which
    stacked two handles on each end level of the browser fixture's ``age_band``.
    """
    mass = np.maximum(grid_basis, 0.0)
    totals = np.sum(mass, axis=0)
    centres = np.full(totals.size, np.nan, dtype=np.float64)
    visible = totals > 0.0
    centres[visible] = (mass[:, visible].T @ grid_x) / totals[visible]
    return centres
```

**3c. `src/superglm/editor/session.py` — import** (lines 23-24):

Old:
```python
from superglm.editor.controls import CONTROL_HANDLE_TERM_TYPES, control_curve_after_move
from superglm.editor.controls import control_points as _control_points
```
New:
```python
from superglm.editor.controls import (
    CONTROL_HANDLE_TERM_TYPES,
    OrderedSplineGeometry,
    control_curve_after_move,
    least_change_coefficients,
    ordered_control_after_move,
    ordered_control_points,
    ordered_spline_geometry,
)
from superglm.editor.controls import control_points as _control_points
```

**3d. `session.py` — `control_points` plus the two new public methods** (lines 574-577):

Old:
```python
    def control_points(self, term: str, n_handles: int | None = None) -> dict[str, Any]:
        """Return fixed-x spline control handles for advanced curve editing."""
        editable = self._require_control_term(term)
        return _control_points(self.model, editable, n_handles=n_handles)
```
New:
```python
    def control_points(self, term: str, n_handles: int | None = None) -> dict[str, Any]:
        """Return fixed-x spline control handles for advanced curve editing."""
        editable, geometry = self._require_control_term(term)
        if geometry is not None:
            coefficients = self.ordered_spline_coefficients(term, geometry)
            return ordered_control_points(geometry, coefficients, n_handles=n_handles)
        return _control_points(self.model, editable, n_handles=n_handles)

    def ordered_spline(self, term: str) -> OrderedSplineGeometry | str | None:
        """The fitted spline of an ordered term on its level axis.

        ``None`` unless the term is an ordered categorical with a spline basis;
        a fixed sentence when its handles are off.
        """
        return ordered_spline_geometry(self.model, self._require_term(term))

    def ordered_spline_coefficients(self, term: str, geometry: OrderedSplineGeometry) -> NDArray:
        """The spline coefficients behind an ordered term's current level effects.

        Starts from the coefficients the latest handle move on the term wrote,
        else the fit's, and keeps them wherever the edits since allow
        (``least_change_coefficients``).  Undo and Redo move the history, so
        the handles follow them.
        """
        prior = next(
            (
                record.params["coefficients"]
                for record in reversed(self.history)
                if record.term == term and "coefficients" in record.params
            ),
            None,
        )
        return least_change_coefficients(
            geometry, self._require_term(term).edited_log_effect, prior
        )
```

**3e. `session.py` — `move_control_point`** (lines 587-590):

Old:
```python
        """Move one spline control handle vertically and refit the displayed curve."""
        editable = self._require_control_term(term)
        handle_index = int(handle_index)
        before = editable.edited_log_effect.copy()
```
New:
```python
        """Move one spline control handle vertically and refit the displayed curve."""
        editable, geometry = self._require_control_term(term)
        handle_index = int(handle_index)
        if geometry is not None:
            self._move_ordered_spline_handle(
                term, geometry, handle_index, float(log_effect), n_handles=n_handles
            )
            return self
        before = editable.edited_log_effect.copy()
```

**3f. `session.py` — insert before `def undo` (line 613):**

Old:
```python
    def undo(self, term: str | None = None) -> EditorSession:
        """Undo the latest edit or, with none since it, the latest structural step.
```
New:
```python
    def _move_ordered_spline_handle(
        self,
        term: str,
        geometry: OrderedSplineGeometry,
        handle_index: int,
        log_effect: float,
        *,
        n_handles: int | None,
    ) -> None:
        """Set one coefficient, and every smooth level to the spline it draws.

        The levels become ``B(level positions) @ c``. Special levels have no
        place on the spline, so the edit leaves them alone. The record keeps
        the coefficients, so the next handle starts from them.
        """
        coefficients = self.ordered_spline_coefficients(term, geometry)
        moved, column = ordered_control_after_move(
            geometry, coefficients, handle_index, log_effect, n_handles=n_handles
        )
        smooth = geometry.level_index
        self._commit(
            term,
            "control_point",
            smooth,
            self.terms[term].edited_log_effect[smooth].copy(),
            geometry.level_basis @ moved,
            {
                "handle_index": handle_index,
                "log_effect": log_effect,
                "basis": "ordered_spline",
                "basis_index": column,
                "x": float(geometry.handle_x[column]),
                "coefficients": [float(value) for value in moved],
            },
        )

    def undo(self, term: str | None = None) -> EditorSession:
        """Undo the latest edit or, with none since it, the latest structural step.
```

**3g. `session.py` — `_require_control_term`** (lines 1396-1402):

Old:
```python
    def _require_control_term(self, term: str) -> EditableTerm:
        editable = self._require_term(term)
        if editable.x is None or editable.levels is not None:
            raise EditorTypeError(f"Term {term!r} does not expose spline control handles.")
        if str(editable.metadata.get("term_type", editable.kind)) not in CONTROL_HANDLE_TERM_TYPES:
            raise EditorTypeError(f"Term {term!r} does not expose spline control handles.")
        return editable
```
New:
```python
    def _require_control_term(self, term: str) -> tuple[EditableTerm, OrderedSplineGeometry | None]:
        """The term and, for an ordered spline, its geometry; refuse a term without handles."""
        editable = self._require_term(term)
        geometry = ordered_spline_geometry(self.model, editable)
        if isinstance(geometry, OrderedSplineGeometry):
            return editable, geometry
        if isinstance(geometry, str):
            raise EditorTypeError(geometry)
        if editable.x is None or editable.levels is not None:
            raise EditorTypeError(f"Term {term!r} does not expose spline control handles.")
        if str(editable.metadata.get("term_type", editable.kind)) not in CONTROL_HANDLE_TERM_TYPES:
            raise EditorTypeError(f"Term {term!r} does not expose spline control handles.")
        return editable, None
```

**3h. `src/superglm/editor/payloads.py` — import** (line 13):

Old:
```python
from superglm.editor.controls import CONTROL_HANDLE_TERM_TYPES
```
New:
```python
from superglm.editor.controls import (
    CONTROL_HANDLE_TERM_TYPES,
    OrderedSplineGeometry,
    ordered_control_points,
    spline_fits_levels,
)
```

**3i. `payloads.py` — `session_payload`** (lines 34-48):

Old:
```python
        ci_lower, ci_upper = _ci_payload(term, edit_delta)
        term_payload = {
```
New:
```python
        ci_lower, ci_upper = _ci_payload(term, edit_delta)
        n_handles = None if control_counts is None else control_counts.get(name)
        ordered_controls, spline_view = _ordered_spline_payloads(session, name, term, n_handles)
        term_payload = {
```
Old:
```python
            "controls": _controls_payload(
                session,
                name,
                term,
                None if control_counts is None else control_counts.get(name),
            ),
```
New:
```python
            "controls": (
                ordered_controls
                if spline_view is not None
                else _controls_payload(session, name, term, n_handles)
            ),
            "spline_view": spline_view,
```

**3j. `payloads.py` — split `_controls_payload` and add the ordered builder** (lines 326-349):

Old:
```python
    try:
        controls = session.control_points(name, n_handles=n_handles)
    except TypeError:
        return None
    payload = {
```
New:
```python
    try:
        controls = session.control_points(name, n_handles=n_handles)
    except TypeError:
        return None
    return _control_payload(controls)


def _control_payload(controls: dict[str, Any]) -> dict[str, Any]:
    payload = {
```
Old (end of the same function):
```python
        payload["build_log_effect"] = [
            float(v) for v in np.asarray(controls["build_log_effect"], dtype=np.float64)
        ]
    return payload
```
New:
```python
        payload["build_log_effect"] = [
            float(v) for v in np.asarray(controls["build_log_effect"], dtype=np.float64)
        ]
    if "grid_x" in controls:
        payload["grid_x"] = [float(v) for v in controls["grid_x"]]
    return payload


def _ordered_spline_payloads(
    session, name: str, term, n_handles: int | None
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """``(controls, spline_view)`` for an ordered term with a spline basis.

    ``(None, None)`` for every other term, and no controls with a
    ``spline_view`` that carries the reason when the handles are off.  The
    geometry and the coefficients are worked out once for both.
    """
    geometry = session.ordered_spline(name)
    if geometry is None:
        return None, None
    if not isinstance(geometry, OrderedSplineGeometry):
        return None, {
            "available": False,
            "reason": geometry,
            "x": None,
            "y": None,
            "original_y": None,
            "level_indices": None,
            "fits_levels": False,
        }
    coefficients = session.ordered_spline_coefficients(name, geometry)
    controls = ordered_control_points(geometry, coefficients, n_handles=n_handles)
    # The fitted curve is the opened model's only while no structural step has
    # replaced it; after one, the chart joins the original levels instead.
    in_force_is_original = getattr(session, "reference_model", session.model) is session.model
    original = geometry.grid_basis @ geometry.fitted if in_force_is_original else None
    return _control_payload(controls), {
        "available": True,
        "reason": None,
        "x": [float(v) for v in geometry.grid_x],
        "y": [float(v) for v in np.exp(geometry.grid_basis @ coefficients)],
        "original_y": None if original is None else [float(v) for v in np.exp(original)],
        "level_indices": geometry.level_index.astype(int).tolist(),
        "fits_levels": spline_fits_levels(geometry, term, session.history),
    }
```

Notes for the implementer:
- No cache is added: the geometry is recomputed per request (0.2 ms on a 9-level term) and depends only on the in-force model and the term's original effects. It is not stored in `term.metadata`, because `save_session` JSON-dumps `metadata` (`io.py:27`) and a dataclass would break it.
- `to_model` is untouched: `_apply_ordered_spline_term` (`apply.py:263-339`) consumes the edited level effects exactly as for a level edit, so the #453 intercept carriage is unchanged; the round-trip test pins it.

- [ ] **Step 4: Run tests, expect PASS**

```bash
./.venv/bin/python -m pytest tests/test_editor_ordered_spline.py -q
./.venv/bin/python -m pytest tests/test_ordered_categorical_specials_editor.py tests/test_piecewise_editor.py tests/test_editor_structure.py -q
./.venv/bin/python -m pytest tests/test_editor.py -k "control or ordered or payload or widget_http" -q
./.venv/bin/ruff check src/superglm/editor tests/test_editor_ordered_spline.py
./.venv/bin/ruff format --check src/superglm/editor tests/test_editor_ordered_spline.py
```
Expected: 11 passed in the new file; neighbours all pass. Mutation checks run on the scratch copy (each fails exactly the named test, then revert): start `least_change_coefficients` from zeros instead of `geometry.fitted` → `test_ordered_spline_handles_start_at_the_fitted_coefficients[wide]`; pass `prior=None` in `ordered_spline_coefficients` → `test_moving_a_handle_sets_every_smooth_level_to_the_spline`; drop the `lstsq` correction → `test_a_level_edit_keeps_the_spline_through_the_levels_and_a_handle_keeps_the_edit`; drop the base shift (`fitted = raw`) → 8 of 11 fail (certification refuses the term).

- [ ] **Step 5: Commit**

```bash
git add src/superglm/editor/controls.py src/superglm/editor/session.py src/superglm/editor/payloads.py tests/test_editor_ordered_spline.py
git commit -m "$(cat <<'EOF'
Editor: handles for an ordered categorical with a spline basis

Handles are the fitted spline's own coefficients (certified against the
reported level effects); moving one sets every smooth level to
B(level positions) @ c and leaves specials alone. After other edits the
coefficients are the least change from the last handle state. The payload
gains spline_view: the spline on a 24-points-per-gap grid of the level axis.
Grouped or shaped terms keep handles off with a fixed reason.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task D2: Draw the ordered spline; Handles, Contrib and Build on its grid (frontend)

**Files:**
- Create: `src/superglm/editor/app/chart/ordered_spline.js`
- Modify: `src/superglm/editor/app/chart.js` (import 9, y-range 119-137, lines 222-224, `basisContributions` 684-694, `buildAccumulationCurve` 696-699, `drawActiveBasis` 722-727, `buildContributionEnvelope` 734-741)
- Modify: `src/superglm/editor/app/interactions.js` (line 1, controlDrag 45-53, point drag 69-70, `previewControlCurve` 730-740)
- Modify: `src/superglm/editor/app/views/tool_rail.js` (34-48, 94-95, 113-127, before `isToolMode` 129)
- Modify: `src/superglm/editor/app/main.js` (line 800)
- Modify: `src/superglm/editor/app/api/contracts.js` (TermPayload 105-122)
- Modify: `src/superglm/editor/app/styles/shell.css` (after `.tool-rail button.active`, 168-172)
- Modify: `src/superglm/editor/app/views/help_content.js` (before the "Features" section, 172)
- Test: `tests/editor_frontend/ordered_spline.test.js` (new), `tests/editor_frontend/tool_rail.test.js`, `tests/editor_frontend/interactions.test.js`, `tests/editor/test_editor_ordered_spline_browser.py` (new)

**Interfaces:**
- Consumes: `term.spline_view` (amendment 1), `term.controls.grid_x/build_basis/basis_index` (amendment 2).
- Produces:
  ```js
  // app/chart/ordered_spline.js
  export function splineCurves(term) -> {x:number[], y:number[]|null, originalY:number[]|null, levelIndices:number[]}|null
  export function levelPolyline(x, values, levelIndices) -> {x:number[], y:number[]}
  export function contributionX(term) -> number[]
  export function shiftedCurve(baseCurve, gridRow, deltaLog) -> number[]
  // views/tool_rail.js
  export function renderToolRail(root, {mode, handlesAvailable, handlesReason = null})
  ```
  Gates: no change is needed at `main.js:1281-1308` (`updateHandleCount`) or `main.js:1334-1343` (`canShowContributions`): both read `controls.basis` / `controls.build_basis`, which D1 now sends for the flagged term. The length check that broke them is the one in `chart.js:684-745`, fixed here through `contributionX`.

- [ ] **Step 1: Write the failing tests**

Create `tests/editor_frontend/ordered_spline.test.js`:

```js
// @ts-nocheck

import assert from "node:assert/strict";
import test from "node:test";

import {
  contributionX,
  levelPolyline,
  shiftedCurve,
  splineCurves,
} from "../../src/superglm/editor/app/chart/ordered_spline.js";

function orderedTerm(overrides = {}) {
  return {
    x: [0, 1, 2, 3],
    y: [0.8, 1, 1.2, 1.5],
    levels: ["a", "b", "c", "MISSING"],
    controls: { grid_x: [0, 0.5, 1, 1.5, 2], build_basis: [[1, 0.5, 0, 0, 0]] },
    spline_view: {
      available: true,
      reason: null,
      x: [0, 0.5, 1, 1.5, 2],
      y: [0.8, 0.9, 1, 1.1, 1.2],
      original_y: [0.8, 0.9, 1, 1.1, 1.2],
      level_indices: [0, 1, 2],
      fits_levels: true,
    },
    ...overrides,
  };
}

test("an ordered spline draws its spline on the grid", () => {
  const curves = splineCurves(orderedTerm());

  assert.deepEqual(curves.x, [0, 0.5, 1, 1.5, 2]);
  assert.deepEqual(curves.y, [0.8, 0.9, 1, 1.1, 1.2]);
  assert.deepEqual(curves.levelIndices, [0, 1, 2]);
});

test("levels off the spline are joined instead, and other terms get no spline", () => {
  const term = orderedTerm();
  term.spline_view.fits_levels = false;

  assert.equal(splineCurves(term).y, null);
  assert.equal(splineCurves({ ...term, spline_view: { ...term.spline_view, available: false } }), null);
  assert.equal(splineCurves({ x: [0, 1], y: [1, 1], spline_view: null }), null);
});

test("the level polyline leaves special levels out", () => {
  const term = orderedTerm();

  assert.deepEqual(levelPolyline(term.x, term.y, [0, 1, 2]), { x: [0, 1, 2], y: [0.8, 1, 1.2] });
});

test("basis rows are sampled on the grid for an ordered spline and on x otherwise", () => {
  assert.deepEqual(contributionX(orderedTerm()), [0, 0.5, 1, 1.5, 2]);
  assert.deepEqual(contributionX({ x: [3, 4], controls: { build_basis: [] } }), [3, 4]);
  assert.deepEqual(contributionX({ x: [3, 4], controls: null }), [3, 4]);
});

test("a moved coefficient scales the curve by its basis function", () => {
  const moved = shiftedCurve([1, 2, 4], [0, 0.5, 1], Math.log(2));

  assert.deepEqual(moved.map((value) => Number(value.toFixed(12))), [1, 2 * Math.SQRT2, 8].map(
    (value) => Number(value.toFixed(12))
  ));
});
```

Append to `tests/editor_frontend/tool_rail.test.js`:

```js

test("Handles off with a reason stay hoverable and say why", () => {
  const buttons = ["select", "move", "zoom", "handles", "help"].map(
    (tool) => new FakeButton(tool),
  );
  const root = new FakeEventHub(buttons);
  const modes = [];
  const binding = bindToolRail({
    root,
    shortcutRoot: new FakeEventHub(),
    onMode: (mode) => modes.push(mode),
    onHelp: () => {},
  });
  const reason = "Handles are off once a band is shaped. Undo the shape to edit the spline.";

  renderToolRail(root, { mode: "handles", handlesAvailable: false, handlesReason: reason });

  assert.equal(buttons[3].disabled, false);
  assert.equal(buttons[3].getAttribute("aria-disabled"), "true");
  assert.equal(buttons[3].dataset.popoverBody, reason);
  assert.equal(buttons[0].getAttribute("aria-checked"), "true");
  root.emit("click", { target: buttons[3] });
  assert.deepEqual(modes, []);

  renderToolRail(root, { mode: "handles", handlesAvailable: true, handlesReason: null });
  assert.equal(buttons[3].getAttribute("aria-disabled"), "false");
  assert.equal(buttons[3].dataset.popoverBody, undefined);
  binding.destroy();
});
```

In `tests/editor_frontend/interactions.test.js`, expose the harness term — in `moveHarness`, change

```js
  return {
    previews,
    mutations,
    get clears() { return clears; },
```
to
```js
  return {
    term,
    previews,
    mutations,
    get clears() { return clears; },
```
and append:

```js

test("a handle drag on an ordered spline moves its drawn curve with the level dots", async () => {
  const harness = moveHarness({ mutationResult: { ok: true }, mode: "handles" });
  const term = harness.term;
  term.y = [1, 1, 1, 1];
  term.controls = {
    y: [1],
    log_effect: [0],
    count: 1,
    basis_index: [0],
    basis: [[0.5, 1, 0.5, 0]],
    build_basis: [[0.25, 0.5, 1, 0.5, 0]],
    build_log_effect: [0],
    grid_x: [0, 0.5, 1, 1.5, 2],
  };
  term.spline_view = {
    available: true,
    reason: null,
    x: [0, 0.5, 1, 1.5, 2],
    y: [1, 1, 1, 1, 1],
    original_y: [1, 1, 1, 1, 1],
    level_indices: [0, 1, 2],
    fits_levels: true,
  };

  await harness.drag(0);

  const { preview } = harness.previews.at(-1);
  const deltaLog = Math.log(preview.controls.y[0]);
  assert.notEqual(deltaLog, 0);
  const expected = [0.25, 0.5, 1, 0.5, 0].map((row) => Math.exp(row * deltaLog));
  preview.spline_view.y.forEach((value, index) => {
    assert.ok(Math.abs(value - expected[index]) <= 1e-12 * expected[index]);
  });
  assert.ok(Math.abs(preview.y[1] - Math.exp(deltaLog)) <= 1e-12 * Math.exp(deltaLog));
  assert.equal(preview.y[3], 1);
});

test("a dragged level of an ordered spline is joined, not drawn on the stale spline", async () => {
  const harness = moveHarness({ mutationResult: { ok: true } });
  harness.term.spline_view = {
    available: true, reason: null, x: [0, 1], y: [2, 2], original_y: [2, 2],
    level_indices: [0, 1], fits_levels: true,
  };

  await harness.drag(0);

  assert.equal(harness.previews[0].preview.spline_view.fits_levels, false);
  assert.equal(harness.term.spline_view.fits_levels, true);
});
```

(The `1e-12` relative tolerances compare a JS `exp` product with a JS `exp` of the same expression — two IEEE evaluations a few ulps apart; this is the preview, not a model value.)

Create `tests/editor/test_editor_ordered_spline_browser.py`:

```python
from __future__ import annotations

from urllib.parse import urlsplit

import pytest

from superglm.editor.controls import ORDERED_SPLINE_GRID_STEPS, ORDERED_SPLINE_SHAPED

pytest.importorskip("playwright.sync_api")
pytestmark = pytest.mark.browser

# The browser fixture's age_band: six levels, no specials.
AGE_BANDS = 6


def _posted(path: str):
    return lambda response: (
        response.request.method == "POST" and urlsplit(response.url).path == path
    )


def _path_points(page, selector: str) -> int:
    return page.evaluate(
        "selector => document.querySelector(selector).getAttribute('d').match(/[ML]/g).length",
        selector,
    )


def _handles_tool(page):
    return page.get_by_role("radiogroup", name="Chart tools").get_by_role(
        "radio", name="Handles", exact=True
    )


def test_an_ordered_spline_is_drawn_as_its_spline_with_handles_contrib_and_build(
    open_editor_page,
):
    with open_editor_page(selected_term="age_band") as (page, session):
        grid_points = ORDERED_SPLINE_GRID_STEPS * (AGE_BANDS - 1) + 1
        # The curve between the dots is the spline, not straight segments.
        assert _path_points(page, "#chart path.edited") == grid_points
        assert _path_points(page, "#chart path.original") == grid_points

        _handles_tool(page).click()
        handles = page.locator("#chart .control-handle")
        handles.first.wait_for()
        live = session.ordered_spline("age_band").live.size
        assert handles.count() == live
        # Handles turn Contrib on; each contribution runs over the same grid.
        assert page.locator("#basisToggle").is_visible()
        assert page.locator("#contribPlay").is_visible()
        contributions = page.locator("#chart .basis-contribution")
        assert contributions.count() == live
        assert _path_points(page, "#chart .basis-contribution") == grid_points

        before = session.terms["age_band"].edited_log_effect.copy()
        box = handles.nth(live // 2).bounding_box()
        assert box is not None
        page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        page.mouse.down()
        page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2 - 30, steps=4)
        with page.expect_response(_posted("/control")) as response_info:
            page.mouse.up()
        assert response_info.value.status == 200

        record = session.history[-1]
        assert record.operation == "control_point"
        assert record.params["basis"] == "ordered_spline"
        assert (session.terms["age_band"].edited_log_effect != before).any()
        # After the move the levels lie on the spline again, so it is still drawn.
        page.wait_for_function(
            "n => document.querySelector('#chart path.edited')"
            ".getAttribute('d').match(/[ML]/g).length === n",
            arg=grid_points,
        )


def test_a_shaped_band_turns_handles_off_and_says_why(open_editor_page):
    with open_editor_page(selected_term="age_band") as (page, session):
        session.replace_with_shaped_range(
            "age_band", lo="25-34", hi="45-54", degree=1, method="fit"
        )
        page.reload(wait_until="domcontentloaded")
        page.locator("#chart path.edited").first.wait_for()
        page.wait_for_function(
            "term => document.querySelector('#status')?.dataset.term === term", arg="age_band"
        )

        handles = _handles_tool(page)
        assert handles.get_attribute("aria-disabled") == "true"
        assert handles.get_attribute("data-popover-body") == ORDERED_SPLINE_SHAPED
        handles.hover()
        popover = page.locator("#uiPopover")
        popover.wait_for(state="visible")
        assert ORDERED_SPLINE_SHAPED in popover.inner_text()
        handles.click(force=True)
        assert handles.get_attribute("aria-checked") == "false"
        assert page.locator("#chart .control-handle").count() == 0
```

(The browser fixture's `age_band` is `Spline(kind="ps", k=5)` over six bands, five columns: the `K < S` case, where `fits_levels` holds after a handle move by construction.)

- [ ] **Step 2: Run them, expect FAIL**

```bash
node --test tests/editor_frontend/ordered_spline.test.js tests/editor_frontend/tool_rail.test.js tests/editor_frontend/interactions.test.js
./.venv/bin/python -m pytest tests/editor/test_editor_ordered_spline_browser.py -m browser --run-browser -q
```
Expected on 155832e8 (with D1 in place): `ordered_spline.test.js` cannot load (`ERR_MODULE_NOT_FOUND .../chart/ordered_spline.js`); `not ok - a handle drag on an ordered spline moves its drawn curve with the level dots`, `not ok - a dragged level of an ordered spline is joined, not drawn on the stale spline`, `not ok - Handles off with a reason stay hoverable and say why`; browser: `AssertionError: assert 6 == 121` (the polyline through six levels) and `assert None == 'true'` (no `aria-disabled` on Handles).

- [ ] **Step 3: Implement**

**3a. Create `src/superglm/editor/app/chart/ordered_spline.js`:**

```js
// @ts-check

/**
 * How an ordered categorical with a spline basis is drawn: its spline on a
 * fine grid of the level axis with the level dots on it, and its special
 * levels as dots of their own. Pure helpers, so the chart and the drag preview
 * read the payload's ``spline_view`` the same way.
 */

/** @typedef {import('../api/contracts.js').TermPayload} TermPayload */

/**
 * The spline lines to draw for ``term``, or null to join its points as usual.
 * ``y`` is null while the edited levels are off the spline (a level edit the
 * basis cannot follow, or a level being dragged); ``originalY`` is null once
 * a structural step has replaced the fit the chart compares against.
 *
 * @param {TermPayload} term
 * @returns {{x:number[], y:number[]|null, originalY:number[]|null, levelIndices:number[]}|null}
 */
export function splineCurves(term) {
  const view = term.spline_view;
  if (!view || !view.available || !view.x || !view.level_indices) return null;
  return {
    x: view.x,
    y: view.fits_levels && view.y ? view.y : null,
    originalY: view.original_y ?? null,
    levelIndices: view.level_indices,
  };
}

/**
 * The polyline through the smooth levels only: special levels are not joined.
 *
 * @param {number[]} x
 * @param {number[]} values
 * @param {number[]} levelIndices
 * @returns {{x:number[], y:number[]}}
 */
export function levelPolyline(x, values, levelIndices) {
  return {
    x: levelIndices.map((index) => x[index]),
    y: levelIndices.map((index) => values[index]),
  };
}

/**
 * The x values the rows of ``controls.build_basis`` are sampled at: the
 * drawing grid for an ordered spline, the term's own x otherwise.
 *
 * @param {TermPayload} term
 * @returns {number[]}
 */
export function contributionX(term) {
  const controls = /** @type {{grid_x?:unknown}|null} */ (term.controls);
  return controls && Array.isArray(controls.grid_x) ? /** @type {number[]} */ (controls.grid_x) : term.x;
}

/**
 * The drawn spline after one coefficient moves by ``deltaLog``: each grid
 * value scales by ``exp(row * deltaLog)``, the same rule the drag preview
 * applies to the level dots.
 *
 * @param {number[]} baseCurve relativities on the grid before the drag
 * @param {number[]} gridRow the moved basis function on the grid
 * @param {number} deltaLog the coefficient's change on the log scale
 * @returns {number[]}
 */
export function shiftedCurve(baseCurve, gridRow, deltaLog) {
  return baseCurve.map((value, index) =>
    Math.max(1e-12, value * Math.exp((Number(gridRow[index]) || 0) * deltaLog))
  );
}
```

**3b. `src/superglm/editor/app/api/contracts.js` — `SplineView` and `TermPayload.spline_view`:**

Old (line 105-106):
```js
 * @typedef {Object} TermPayload
 * @property {string} kind
```
New:
```js
 * An ordered categorical with a spline basis, drawn as its spline: the grid
 * ``x`` (level ``i`` at ``i``), the current and fitted curves as relativities,
 * and the display indices of the smooth levels. ``available`` is false, with
 * a fixed ``reason``, while the term's handles are off; ``fits_levels`` is
 * false while the edited levels are off the spline. Null for every other term.
 * @typedef {Object} SplineView
 * @property {boolean} available
 * @property {string|null} reason
 * @property {number[]|null} x
 * @property {number[]|null} y
 * @property {number[]|null} original_y
 * @property {number[]|null} level_indices
 * @property {boolean} fits_levels
 */
/**
 * @typedef {Object} TermPayload
 * @property {string} kind
```
Old (line 121-122):
```js
 * @property {TermShape} shape
 */
```
New:
```js
 * @property {TermShape} shape
 * @property {SplineView|null} [spline_view]
 */
```

**3c. `src/superglm/editor/app/chart.js`:**

Old (line 9):
```js
import { el, line, text } from "./chart/svg.js";
```
New:
```js
import { contributionX, levelPolyline, splineCurves } from "./chart/ordered_spline.js";
import { el, line, text } from "./chart/svg.js";
```
Old (lines 119-137):
```js
  const buildEnvelope = buildActive ? buildContributionEnvelope(term) : [];
  const buildValues = buildEnvelope.flat();
  const previousValues = previous || [];
  const yMinRaw = Math.min(
    ...y,
    ...original,
    ...previousValues,
    ...ciValues,
    ...controlValues,
    ...buildValues
  );
  const yMaxRaw = Math.max(
    ...y,
    ...original,
    ...previousValues,
    ...ciValues,
    ...controlValues,
    ...buildValues
  );
```
New:
```js
  const buildEnvelope = buildActive ? buildContributionEnvelope(term) : [];
  const buildValues = buildEnvelope.flat();
  const previousValues = previous || [];
  // A grouped display never carries a spline: the tools are off for groups.
  const spline = view.displayIsCollapsed ? null : splineCurves(term);
  const splineValues = spline ? [...(spline.y || []), ...(spline.originalY || [])] : [];
  const yMinRaw = Math.min(
    ...y,
    ...original,
    ...previousValues,
    ...ciValues,
    ...controlValues,
    ...buildValues,
    ...splineValues
  );
  const yMaxRaw = Math.max(
    ...y,
    ...original,
    ...previousValues,
    ...ciValues,
    ...controlValues,
    ...buildValues,
    ...splineValues
  );
```
Old (lines 222-224):
```js
  if (!buildActive) path(svg, x, original, sx, sy, "original");
  if (!buildActive && previous) path(svg, x, previous, sx, sy, "previous-edit");
  if (!buildActive) path(svg, x, y, sx, sy, "edited");
```
New:
```js
  if (!buildActive) drawTermLines(svg, { x, y, original, previous, spline, sx, sy });
```
Old (lines 684-692):
```js
function basisContributions(svg, term, sx, sy, buildActive = false) {
  const { basis, logEffects } = contributionComponents(term);
  for (let i = 0; i < basis.length; i++) {
    const row = basis[i];
    if (!Array.isArray(row) || row.length !== term.x.length) continue;
    const beta = Array.isArray(logEffects) ? Number(logEffects[i] || 0) : 0;
    const y = row.map((v) => Math.exp((Number(v) || 0) * beta));
    const contribution = path(svg, term.x, y, sx, sy, "basis-contribution");
```
New:
```js
// An ordered spline draws its curves on the level-axis grid and leaves its
// special levels as lone dots; every other term joins its points.
function drawTermLines(svg, { x, y, original, previous, spline, sx, sy }) {
  const join = (values) => (spline ? levelPolyline(x, values, spline.levelIndices) : { x, y: values });
  const originalLine = spline && spline.originalY
    ? { x: spline.x, y: spline.originalY }
    : join(original);
  path(svg, originalLine.x, originalLine.y, sx, sy, "original");
  if (previous) {
    const previousLine = join(previous);
    path(svg, previousLine.x, previousLine.y, sx, sy, "previous-edit");
  }
  const editedLine = spline && spline.y ? { x: spline.x, y: spline.y } : join(y);
  path(svg, editedLine.x, editedLine.y, sx, sy, "edited");
}

// Basis rows are sampled at contributionX(term): the ordered spline's grid,
// or the term's own x.
function basisContributions(svg, term, sx, sy, buildActive = false) {
  const { basis, logEffects } = contributionComponents(term);
  const gridX = contributionX(term);
  for (let i = 0; i < basis.length; i++) {
    const row = basis[i];
    if (!Array.isArray(row) || row.length !== gridX.length) continue;
    const beta = Array.isArray(logEffects) ? Number(logEffects[i] || 0) : 0;
    const y = row.map((v) => Math.exp((Number(v) || 0) * beta));
    const contribution = path(svg, gridX, y, sx, sy, "basis-contribution");
```
Old (lines 697-699):
```js
  const { basis, logEffects } = contributionComponents(term);
  const x = term.x || [];
  if (!x.length) return { x: [], y: [], activeIndex: -1 };
```
New:
```js
  const { basis, logEffects } = contributionComponents(term);
  const x = contributionX(term) || [];
  if (!x.length) return { x: [], y: [], activeIndex: -1 };
```
Old (lines 725-729):
```js
  const row = basis[index];
  if (!Array.isArray(row) || row.length !== term.x.length) return;
  const beta = Number(logEffects[index] || 0);
  const y = row.map((v) => Math.exp((Number(v) || 0) * beta));
  const active = path(svg, term.x, y, sx, sy, "basis-active");
```
New:
```js
  const row = basis[index];
  const gridX = contributionX(term);
  if (!Array.isArray(row) || row.length !== gridX.length) return;
  const beta = Number(logEffects[index] || 0);
  const y = row.map((v) => Math.exp((Number(v) || 0) * beta));
  const active = path(svg, gridX, y, sx, sy, "basis-active");
```
Old (lines 735-740):
```js
  const { basis, logEffects } = contributionComponents(term);
  const finalEta = finalContributionEta(basis, logEffects, term.x.length);
  const values = [finalEta.map((value) => Math.exp(value))];
  for (let j = 0; j < basis.length; j++) {
    const row = basis[j];
    if (!Array.isArray(row) || row.length !== term.x.length) continue;
```
New:
```js
  const { basis, logEffects } = contributionComponents(term);
  const n = contributionX(term).length;
  const finalEta = finalContributionEta(basis, logEffects, n);
  const values = [finalEta.map((value) => Math.exp(value))];
  for (let j = 0; j < basis.length; j++) {
    const row = basis[j];
    if (!Array.isArray(row) || row.length !== n) continue;
```

**3d. `src/superglm/editor/app/interactions.js`:**

Old (line 1):
```js
export function bindInteractions(context) {
```
New:
```js
import { shiftedCurve } from "./chart/ordered_spline.js";

export function bindInteractions(context) {
```
Old (lines 45-53):
```js
      interaction.controlDrag = {
        term: context.selectedTerm(),
        preview,
        index: i,
        startValue: preview.controls.y[i],
        value: preview.controls.y[i],
        baseY: preview.y.slice(),
        basis: preview.controls.basis ? preview.controls.basis[i] : null
      };
```
New:
```js
      interaction.controlDrag = {
        term: context.selectedTerm(),
        preview,
        index: i,
        startValue: preview.controls.y[i],
        value: preview.controls.y[i],
        baseY: preview.y.slice(),
        basis: preview.controls.basis ? preview.controls.basis[i] : null,
        // An ordered spline also moves its drawn curve, sampled on its grid.
        gridBasis: splineGridRow(preview.controls, i),
        baseCurve: preview.spline_view && Array.isArray(preview.spline_view.y)
          ? preview.spline_view.y.slice()
          : null
      };
```
Old (lines 69-70):
```js
      const affectedIndices = structuralEditSourceIndices(activeTerm, indices);
      const preview = structuredClone(activeTerm);
```
New:
```js
      const affectedIndices = structuralEditSourceIndices(activeTerm, indices);
      const preview = structuredClone(activeTerm);
      // A dragged level leaves the spline until Python redraws it: join the dots.
      if (preview.spline_view) preview.spline_view = { ...preview.spline_view, fits_levels: false };
```
Old (end of `previewControlCurve`, lines 737-740):
```js
  for (let i = 0; i < term.y.length; i++) {
    term.y[i] = Math.max(1e-12, drag.baseY[i] * Math.exp(drag.basis[i] * deltaLog));
  }
}
```
New:
```js
  for (let i = 0; i < term.y.length; i++) {
    term.y[i] = Math.max(1e-12, drag.baseY[i] * Math.exp(drag.basis[i] * deltaLog));
  }
  if (term.spline_view && drag.gridBasis && drag.baseCurve &&
      drag.gridBasis.length === drag.baseCurve.length) {
    term.spline_view.y = shiftedCurve(drag.baseCurve, drag.gridBasis, deltaLog);
  }
}

// The dragged handle's basis function on an ordered spline's drawing grid;
// null for every other term.
function splineGridRow(controls, index) {
  if (!Array.isArray(controls.grid_x) || !Array.isArray(controls.build_basis)) return null;
  const row = controls.build_basis[controls.basis_index ? controls.basis_index[index] : index];
  return Array.isArray(row) ? row : null;
}
```

**3e. `src/superglm/editor/app/views/tool_rail.js`:**

Old (line 38):
```js
      if (element instanceof HTMLButtonElement && !element.disabled) radios.push(element);
```
New:
```js
      if (element instanceof HTMLButtonElement && !isUnavailable(element)) radios.push(element);
```
Old (line 46):
```js
    if (!(element instanceof HTMLButtonElement) || !root.contains(element) || element.disabled) {
```
New:
```js
    if (!(element instanceof HTMLButtonElement) || !root.contains(element) || isUnavailable(element)) {
```
Old (line 95):
```js
    if (!(button instanceof HTMLButtonElement) || button.disabled) return;
```
New:
```js
    if (!(button instanceof HTMLButtonElement) || isUnavailable(button)) return;
```
Old (lines 113-121):
```js
/**
 * @param {HTMLElement} root
 * @param {{mode:ToolMode, handlesAvailable:boolean}} state
 */
export function renderToolRail(root, { mode, handlesAvailable }) {
  const effectiveMode = mode === "handles" && !handlesAvailable ? "select" : mode;
  for (const element of root.querySelectorAll('[role="radio"]')) {
    if (!(element instanceof HTMLButtonElement)) continue;
    if (element.dataset.tool === "handles") element.disabled = !handlesAvailable;
```
New:
```js
/**
 * Handles with a ``handlesReason`` stay focusable and hoverable, marked
 * aria-disabled, so their popover can say why they are off; without one they
 * are plainly disabled.
 *
 * @param {HTMLElement} root
 * @param {{mode:ToolMode, handlesAvailable:boolean, handlesReason?:string|null}} state
 */
export function renderToolRail(root, { mode, handlesAvailable, handlesReason = null }) {
  const effectiveMode = mode === "handles" && !handlesAvailable ? "select" : mode;
  for (const element of root.querySelectorAll('[role="radio"]')) {
    if (!(element instanceof HTMLButtonElement)) continue;
    if (element.dataset.tool === "handles") renderHandlesAvailability(element, handlesAvailable, handlesReason);
```
Old (line 129-130):
```js
/** @param {string|undefined} value @returns {value is ToolMode} */
function isToolMode(value) {
```
New:
```js
/**
 * @param {HTMLButtonElement} element
 * @param {boolean} available
 * @param {string|null} reason
 */
function renderHandlesAvailability(element, available, reason) {
  const explained = !available && Boolean(reason);
  element.disabled = !available && !explained;
  element.setAttribute("aria-disabled", String(!available));
  if (explained && reason) {
    element.dataset.popoverTitle = "Handles";
    element.dataset.popoverBody = reason;
  } else {
    delete element.dataset.popoverTitle;
    delete element.dataset.popoverBody;
  }
}

/** @param {HTMLButtonElement} element */
function isUnavailable(element) {
  return element.disabled || element.getAttribute("aria-disabled") === "true";
}

/** @param {string|undefined} value @returns {value is ToolMode} */
function isToolMode(value) {
```
(`helpForElement` already prefers an element's own `data-popover-body` over `TOOL_HELP` — `help_content.js:230` — so no popover change is needed.)

**3f. `src/superglm/editor/app/main.js` (line 800):**

Old:
```js
  renderToolRail(toolRail, { mode: view.mode, handlesAvailable: Boolean(term.controls) });
```
New:
```js
  renderToolRail(toolRail, {
    mode: view.mode,
    handlesAvailable: Boolean(term.controls),
    handlesReason: term.spline_view?.reason ?? null
  });
```

**3g. `src/superglm/editor/app/styles/shell.css` — after the `.tool-rail button.active` rule (lines 168-172):**

Old:
```css
.tool-rail button.active {
  background: var(--surface);
  color: var(--blue);
  box-shadow: 0 1px 2px var(--shadow);
}
```
New:
```css
.tool-rail button.active {
  background: var(--surface);
  color: var(--blue);
  box-shadow: 0 1px 2px var(--shadow);
}

/* Off with a reason: still hoverable, so its popover can say why. */
.tool-rail button[aria-disabled="true"] {
  cursor: not-allowed;
  opacity: 0.45;
}

.tool-rail button[aria-disabled="true"]:hover {
  background: transparent;
  color: var(--muted);
}
```

**3h. `src/superglm/editor/app/views/help_content.js` — a Help section (before line 172):**

Old:
```js
  Object.freeze({
    title: "Features",
```
New:
```js
  Object.freeze({
    title: "Ordered splines",
    items: Object.freeze([
      "An ordered categorical fitted with a spline is drawn as its spline, with a dot on each level. Special levels are separate dots.",
      "Handles edit it like a numeric spline: each handle is one spline coefficient, and moving it sets every level to the spline through the handles. Special levels do not move. Contrib and Build show the spline's basis.",
      "Handles are off while levels are grouped or a band is shaped; the Handles tool says which.",
    ]),
  }),
  Object.freeze({
    title: "Features",
```

- [ ] **Step 4: Run tests, expect PASS**

```bash
npm ci   # only if node_modules/ is absent (dev-only, from package-lock.json)
npm run check:frontend
./.venv/bin/python -m pytest tests/test_editor_browser.py tests/editor -m browser --run-browser -q
```
Expected: `tsc` clean; node `# fail 0` (185 tests at 155832e8, 193 after this task, 197 with K1b); browser all pass (88 with K1b, 87 without). Run the browser files in this order: on 155832e8 itself, `tests/editor` *before* `tests/test_editor_browser.py` fails 10 tests in `test_editor_browser.py` (pre-existing; see Follow-ups).

- [ ] **Step 5: Commit**

```bash
git add src/superglm/editor/app/chart/ordered_spline.js src/superglm/editor/app/chart.js \
  src/superglm/editor/app/interactions.js src/superglm/editor/app/views/tool_rail.js \
  src/superglm/editor/app/main.js src/superglm/editor/app/api/contracts.js \
  src/superglm/editor/app/styles/shell.css src/superglm/editor/app/views/help_content.js \
  tests/editor_frontend/ordered_spline.test.js tests/editor_frontend/tool_rail.test.js \
  tests/editor_frontend/interactions.test.js tests/editor/test_editor_ordered_spline_browser.py
git commit -m "$(cat <<'EOF'
Editor: draw an ordered spline between its levels, with Handles, Contrib and Build

The chart draws an ordered categorical's spline on its level-axis grid,
with specials as lone dots; contributions and Build run on that grid; a
handle drag moves the drawn curve with the dots. Handles that are off for
a grouped or shaped term stay hoverable and say why.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task K1a: Rating-table preview backend (`/rating_table`)

**Files:**
- Create: `src/superglm/editor/rating_preview.py`
- Modify: `src/superglm/editor/widget.py` (imports 24 and 43-47, `_EXPORT_DEFAULT_FILENAMES` 65-68, `__init__` 128, `_export_bytes` 637-657, before `_export_file` 678)
- Modify: `src/superglm/editor/server.py` (before `/download_export`, 179)
- Test: `tests/test_editor_rating_preview.py` (new)

**Interfaces:**
- Consumes: `training_export_dataset(session)` (`evaluation.py:149`), `EditorWidget._model_materialized_for_dataset` (`widget.py:352-366`), `export.rating_tables.build_rating_table_payload` (`rating_tables.py:1884`) and `_unsupported_structured_export_terms` (101), `export.excel._main_effect_number_format` (367), `_PIECEWISE_NUMBER_FORMAT` (83), `_piecewise_interpolation_note` (289), `_ppform_evaluation_note` (476).
- Produces:
  ```python
  # superglm/editor/rating_preview.py
  PREVIEW_IMPACT_BINS: tuple[int, ...] = ()
  UNSUPPORTED_TERMS, EXPORT_REFUSED, NO_BLOCK, SUPERSEDED: str
  @dataclass(frozen=True)
  class RatingPreview: model_revision: int; payload: Any | None; reason: str | None
  def refusal_reason(model) -> str
  def term_rating_table(preview: RatingPreview, term: str) -> dict[str, Any]
  # EditorWidget
  def _rating_table_payload(self, *, impact_bins: tuple[int, ...] | None = None) -> tuple[RatingTablePayload | None, int]
  def _rating_table(self, term: str) -> dict[str, Any]
  # HTTP: POST /rating_table {term} -> {term, available, reason, columns, rows, formats, note, model_revision}
  ```
  Cache: `EditorWidget._rating_preview: RatingPreview | None`. Owner: the widget. Lifetime: until `session.model_revision` moves (every edit, undo, redo, structural step and Refit advances it; switching terms and staging a pending step do not). Invalidation: compared on every request; built outside the lock like the export and published only if the revision is still current. It holds one `RatingTablePayload` (every main-effect block of one revision). It preserves no rank or error decisions: the blocks are the builder's.

- [ ] **Step 1: Write the failing test**

Create `tests/test_editor_rating_preview.py`:

```python
"""The rating-table preview shows the Excel export's own block for the current term."""

from __future__ import annotations

import io
import json
import urllib.request

import numpy as np
import pandas as pd
import pytest

from superglm import (
    Categorical,
    LambdaPolicy,
    Numeric,
    OrderedCategorical,
    Piecewise,
    RandomEffect,
    Spline,
    SuperGLM,
)
from superglm.editor import EditorSession
from superglm.editor.rating_preview import EXPORT_REFUSED, UNSUPPORTED_TERMS

TERMS = ["x_spline", "x_piece", "region", "band"]


def _frame(seed=20261003, n=600):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "x_spline": rng.uniform(0.0, 10.0, n),
            "x_piece": rng.uniform(0.0, 6.0, n),
            "region": rng.choice(["A", "B", "C"], n),
            "band": rng.choice(["low", "medium", "high"], n),
        }
    )
    eta = (
        -0.6
        + 0.18 * np.sin(X["x_spline"].to_numpy() / 1.8)
        + 0.05 * np.abs(X["x_piece"].to_numpy() - 3.0)
        + 0.25 * (X["region"].to_numpy() == "B")
        + 0.3 * (X["band"].to_numpy() == "high")
    )
    return X, rng.poisson(np.exp(eta)).astype(np.float64)


def _features():
    return {
        "x_spline": Spline(n_knots=6),
        "x_piece": Piecewise(breaks=[3.0]),
        "region": Categorical(base="first"),
        "band": OrderedCategorical(
            order=["low", "medium", "high"], basis=Spline(kind="ps", n_knots=2), base="first"
        ),
    }


@pytest.fixture
def poisson_session():
    X, y = _frame()
    model = SuperGLM(family="poisson", selection_penalty=0.0, features=_features()).fit(X, y)
    return EditorSession.from_model(model, terms=TERMS, train_data=(X, y))


def _workbook_block(data: bytes, term: str):
    """``(columns, rows, formats, note)`` of ``term``'s block on the workbook's sheet."""
    from openpyxl import load_workbook

    sheet = load_workbook(io.BytesIO(data))["Rating Tables"]
    start = next(cell.column for cell in sheet[5] if cell.value == term)
    width = 0
    while sheet.cell(row=7, column=start + width).value not in (None, ""):
        width += 1
        if sheet.cell(row=5, column=start + width).value not in (None, ""):
            break
    columns = [sheet.cell(row=7, column=start + offset).value for offset in range(width)]
    rows, formats, row = [], None, 8
    while sheet.cell(row=row, column=start).value is not None:
        cells = [sheet.cell(row=row, column=start + offset) for offset in range(width)]
        rows.append([cell.value for cell in cells])
        formats = [
            None if cell.number_format == "General" else cell.number_format for cell in cells
        ]
        row += 1
    return columns, rows, formats, sheet.cell(row=6, column=start).value


def _assert_same_cells(sent, written):
    """Text cells match exactly; numbers to the workbook's 16 significant digits.

    openpyxl writes a float as ``"%.16g"``, half a unit in the 16th digit at
    most ``5e-16 |v|`` away, and reading the decimal back rounds once more, by
    ``u`` relative. The preview sends the block's float64 values themselves.
    """
    assert len(sent) == len(written)
    for value, cell in zip(sent, written, strict=True):
        if isinstance(value, str):
            assert value == cell
        else:
            assert abs(value - cell) <= 5e-16 * abs(value) + 2.0**-53 * abs(cell)


def test_the_preview_is_the_workbook_block_for_every_term(poisson_session):
    # On master there is no preview at all: the widget has no `_rating_table`.
    session = poisson_session
    session.select_levels("region", ["B"])
    session.shift("region", 0.2)
    widget = session.widget()
    try:
        workbook = widget._export_bytes("xlsx").data
        previews = {term: widget._rating_table(term) for term in TERMS}
    finally:
        widget.close()

    for term, preview in previews.items():
        columns, rows, formats, note = _workbook_block(workbook, term)
        assert preview["available"] is True
        assert preview["model_revision"] == session.model_revision
        assert preview["columns"] == columns
        assert len(preview["rows"]) == len(rows)
        for sent, written in zip(preview["rows"], rows, strict=True):
            _assert_same_cells(sent, written)
        assert preview["formats"] == formats
        assert preview["note"] == note


def test_the_preview_builds_once_per_revision_and_skips_the_impact_sweep(
    poisson_session, monkeypatch
):
    from superglm.export import rating_tables

    calls = []
    build = rating_tables.build_rating_table_payload

    def counting(*args, **kwargs):
        calls.append(kwargs.get("impact_bins"))
        return build(*args, **kwargs)

    monkeypatch.setattr(rating_tables, "build_rating_table_payload", counting)
    session = poisson_session
    widget = session.widget()
    try:
        widget._rating_table("region")
        widget._rating_table("band")
        assert calls == [()]
        session.select_levels("band", ["high"])
        session.shift("band", 0.1)
        edited = widget._rating_table("band")
        assert calls == [(), ()]
    finally:
        widget.close()
    assert edited["model_revision"] == session.model_revision


def test_without_training_data_the_preview_gives_the_exports_sentence():
    rng = np.random.default_rng(20260802)
    X = pd.DataFrame({"x": rng.normal(size=90)})
    y = rng.poisson(np.exp(0.2 + 0.4 * X["x"].to_numpy())).astype(np.float64)
    model = SuperGLM(
        family="poisson",
        retain_fit_state=False,
        selection_penalty=0.0,
        features={"x": Numeric()},
    ).fit(X, y)
    session = EditorSession.from_model(model, terms=["x"], validation_data=(X[:20], y[:20]))
    widget = session.widget()
    try:
        preview = widget._rating_table("x")
    finally:
        widget.close()

    assert preview["available"] is False
    assert preview["reason"] == (
        "Excel export requires train_data or retained fit data; "
        "validation/test data are not substituted."
    )
    assert preview["rows"] == []


def test_a_refused_export_gives_a_fixed_sentence_not_backend_text():
    # A gaussian identity-link model: the multiplicative workbook refuses it.
    X, y = _frame()
    model = SuperGLM(family="gaussian", selection_penalty=0.0, features=_features()).fit(
        X, np.log1p(y)
    )
    session = EditorSession.from_model(model, terms=TERMS)
    widget = session.widget()
    try:
        preview = widget._rating_table("region")
    finally:
        widget.close()

    assert preview["available"] is False
    assert preview["reason"] == EXPORT_REFUSED


def test_a_model_with_a_random_effect_says_rating_tables_do_not_cover_it():
    rng = np.random.default_rng(20260726)
    codes = np.repeat(np.arange(8), 12)
    X = pd.DataFrame({"x": rng.normal(size=codes.size), "group": [f"g{c}" for c in codes]})
    y = rng.poisson(np.exp(0.1 * X["x"].to_numpy() + 0.05 * codes)).astype(np.float64)
    model = SuperGLM(
        family="poisson",
        features={"x": Numeric(), "group": RandomEffect(lambda_policy=LambdaPolicy.fixed(1.0))},
        selection_penalty=0.0,
        direct_solve="structured",
    ).fit_reml(X, y, runtime_validation="skip")
    session = EditorSession.from_model(model, terms=["x"])
    widget = session.widget()
    try:
        preview = widget._rating_table("x")
    finally:
        widget.close()

    assert preview["available"] is False
    assert preview["reason"] == UNSUPPORTED_TERMS


def test_the_rating_table_route_answers_in_the_contract_shape(poisson_session):
    widget = poisson_session.widget()
    try:
        request = urllib.request.Request(
            f"{widget.url}/rating_table",
            data=json.dumps({"term": "band"}).encode("utf-8"),
            method="POST",
            headers={
                "Content-Type": "application/json",
                "X-SuperGLM-Editor-Token": widget._token,
            },
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    finally:
        widget.close()

    assert set(payload) == {
        "term",
        "available",
        "reason",
        "columns",
        "rows",
        "formats",
        "note",
        "model_revision",
    }
    assert payload["term"] == "band"
    assert payload["columns"] == ["band", "Relativity", "Weight"]
    assert [row[0] for row in payload["rows"]] == ["low", "medium", "high"]
```

The comparison bound, stated: openpyxl writes `"%.16g"` (`openpyxl/compat/strings.py::safe_string`, MIT), so a cell is within half a unit in the 16th significant digit, `≤ 5e-16 |v|`, of the float64 value, and parsing it back rounds once more by `u|cell|`. The piecewise term covers the workbook's one per-block format override and its note row; the counting test is the "counts, not wall time" assertion and the mutation check for both the cache and `impact_bins=()`.

- [ ] **Step 2: Run it, expect FAIL**

Run: `./.venv/bin/python -m pytest tests/test_editor_rating_preview.py -q`
Expected on 155832e8: collection error `ModuleNotFoundError: No module named 'superglm.editor.rating_preview'`; behaviourally there is no `/rating_table` route (404) and no `EditorWidget._rating_table`.

- [ ] **Step 3: Implement**

**3a. Create `src/superglm/editor/rating_preview.py`:**

```python
"""The rating-table preview: one term's block of the Excel export, for the browser.

The widget builds the payload through the same call the workbook export makes
(``EditorWidget._rating_table_payload``); this module picks one term's block
out of it and renders it as JSON, with the cell formats and the note the
workbook gives that block.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

# The preview asks for no discretisation-impact sweep. The builder makes every
# block before it sweeps, and the sweep fills only the workbook's impact sheet,
# so the blocks are the workbook's; the sweep was 5.5 s of the 7.0 s build on
# the 678,013-row freMTPL2 book with two splines.
PREVIEW_IMPACT_BINS: tuple[int, ...] = ()

# One fixed sentence per reason there is no table to show.
UNSUPPORTED_TERMS = (
    "Rating tables do not cover random-effect or factor-smooth terms, so this model has none."
)
EXPORT_REFUSED = "The Excel export refuses this model, so there is no rating table to preview."
NO_BLOCK = "The rating table has no block for this term."
SUPERSEDED = "The model changed while the table was built. Showing the new one next."


@dataclass(frozen=True)
class RatingPreview:
    """One model revision's rating-table payload, or why it has none.

    Owned by the widget and kept until the session's ``model_revision``
    moves, which every edit, undo, redo and structural step does; switching
    terms reuses it.
    """

    model_revision: int
    payload: Any | None
    reason: str | None


def refusal_reason(model) -> str:
    """The fixed sentence for an export the builder refused."""
    from superglm.export.rating_tables import _unsupported_structured_export_terms

    return UNSUPPORTED_TERMS if _unsupported_structured_export_terms(model) else EXPORT_REFUSED


def term_rating_table(preview: RatingPreview, term: str) -> dict[str, Any]:
    """``term``'s main-effect block as ``{term, available, reason, columns, rows, ...}``."""
    if preview.payload is None:
        return _unavailable(term, preview.reason or EXPORT_REFUSED, preview.model_revision)
    block = next((item for item in preview.payload.main_effects if item.name == term), None)
    if block is None:
        return _unavailable(term, NO_BLOCK, preview.model_revision)
    table = block.table
    return {
        "term": term,
        "available": True,
        "reason": None,
        "columns": [str(column) for column in table.columns],
        "rows": [[_cell(value) for value in row] for row in table.itertuples(index=False)],
        "formats": _block_formats(block),
        "note": _block_note(block),
        "model_revision": preview.model_revision,
    }


def _unavailable(term: str, reason: str, model_revision: int) -> dict[str, Any]:
    return {
        "term": term,
        "available": False,
        "reason": reason,
        "columns": [],
        "rows": [],
        "formats": [],
        "note": None,
        "model_revision": model_revision,
    }


def _block_formats(block) -> list[str | None]:
    """The number format the workbook gives each column, in the order it applies them."""
    from superglm.export.excel import _PIECEWISE_NUMBER_FORMAT, _main_effect_number_format

    formats = [
        _main_effect_number_format(block, str(column), offset)
        for offset, column in enumerate(block.table.columns)
    ]
    if block.kind == "piecewise":
        for offset in (1, 2):
            if offset < len(formats):
                formats[offset] = _PIECEWISE_NUMBER_FORMAT
    return formats


def _block_note(block) -> str | None:
    """The note the workbook writes above the block, or None where it writes none."""
    from superglm.export.excel import _piecewise_interpolation_note, _ppform_evaluation_note

    if block.kind == "piecewise":
        return _piecewise_interpolation_note(
            block.table, block.extrapolation or "clip", block.centering_shift
        )
    if block.kind == "continuous_ppform":
        return _ppform_evaluation_note(block.name, block.extrapolation)
    return None


def _cell(value: Any) -> str | int | float | bool | None:
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if isinstance(value, int | np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        number = float(value)
        return number if np.isfinite(number) else None
    if value is None or pd.isna(value):
        return None
    return str(value)
```

(`_block_formats` mirrors `excel._format_main_effect_blocks` then `_annotate_piecewise_blocks`, `excel.py:389-436`; `_block_note` mirrors the three `_annotate_*` passes for the two block kinds an editor term produces. The private `export.excel` helpers are imported, not copied, so the preview cannot drift from the workbook.)

**3b. `src/superglm/editor/widget.py`:**

Old (line 24):
```python
from superglm.editor import persistence
```
New:
```python
from superglm.editor import persistence, rating_preview
```
Old (lines 43-47):
```python
from superglm.editor.payloads import (
    session_payload,
    timeline_payload,
    undo_redo_payload,
)
```
New:
```python
from superglm.editor.payloads import (
    session_payload,
    timeline_payload,
    undo_redo_payload,
)
from superglm.editor.rating_preview import PREVIEW_IMPACT_BINS, RatingPreview
```
Old (lines 65-68):
```python
_EXPORT_DEFAULT_FILENAMES = {
    "joblib": "superglm_edited_model.joblib",
    "xlsx": "superglm_rating_tables.xlsx",
}
```
New:
```python
_EXPORT_DEFAULT_FILENAMES = {
    "joblib": "superglm_edited_model.joblib",
    "xlsx": "superglm_rating_tables.xlsx",
}
_EXCEL_NEEDS_TRAINING_DATA = (
    "Excel export requires train_data or retained fit data; "
    "validation/test data are not substituted."
)
```
Old (line 128):
```python
        self._profile_condition = threading.Condition(threading.RLock())
```
New:
```python
        self._profile_condition = threading.Condition(threading.RLock())
        self._rating_preview: RatingPreview | None = None
```
Old (lines 637-657, the xlsx branch of `_export_bytes`):
```python
        else:
            dataset = training_export_dataset(self.session)
            if dataset is None:
                raise EditorValueError(
                    "Excel export requires train_data or retained fit data; "
                    "validation/test data are not substituted."
                )
            model, revision = self._model_materialized_for_dataset(dataset)
            if model is None:
                raise RuntimeError("Export request was superseded.")
            from superglm.export.excel import write_rating_table_workbook
            from superglm.export.rating_tables import build_rating_table_payload

            payload = build_rating_table_payload(
                model,
                dataset.X,
                dataset.y,
                sample_weight=dataset.sample_weight,
                offset=dataset.offset,
            )
            buffer = io.BytesIO()
```
New:
```python
        else:
            payload, revision = self._rating_table_payload()
            if payload is None:
                raise RuntimeError("Export request was superseded.")
            from superglm.export.excel import write_rating_table_workbook

            buffer = io.BytesIO()
```
Old (line 678):
```python
    def _export_file(
        self,
        format: str,
```
New:
```python
    def _rating_table_payload(self, *, impact_bins: tuple[int, ...] | None = None):
        """The Excel export's rating-table payload for the current revision.

        One path for the workbook and its preview: the training split, the
        model materialised on it, and the builder's defaults, except the
        ``impact_bins`` the preview passes. ``(None, revision)`` when the
        revision moved while the model was materialised.
        """
        dataset = training_export_dataset(self.session)
        if dataset is None:
            raise EditorValueError(_EXCEL_NEEDS_TRAINING_DATA)
        model, revision = self._model_materialized_for_dataset(dataset)
        if model is None:
            return None, revision
        from superglm.export.rating_tables import build_rating_table_payload

        options = {} if impact_bins is None else {"impact_bins": impact_bins}
        payload = build_rating_table_payload(
            model,
            dataset.X,
            dataset.y,
            sample_weight=dataset.sample_weight,
            offset=dataset.offset,
            **options,
        )
        return payload, revision

    def _rating_table(self, term: str) -> dict[str, Any]:
        """``term``'s block of the Excel rating table, for the Table view.

        Built once per model revision, outside the lock like the export, and
        reused while the revision stands.
        """
        with self._lock:
            if term not in self.session.terms:
                raise EditorKeyError(f"Unknown editable term: {term!r}")
            revision = self.session.model_revision
            preview = self._rating_preview
        if preview is None or preview.model_revision != revision:
            preview = self._build_rating_preview(revision)
            with self._lock:
                if preview.model_revision == self.session.model_revision:
                    self._rating_preview = preview
        return rating_preview.term_rating_table(preview, term)

    def _build_rating_preview(self, revision: int) -> RatingPreview:
        try:
            payload, revision = self._rating_table_payload(impact_bins=PREVIEW_IMPACT_BINS)
        except EditorClientError as exc:
            return RatingPreview(revision, None, exc.public_message)
        except (NotImplementedError, OverflowError, ValueError):
            # The builder's refusals carry backend text; the browser gets a
            # fixed sentence and the log keeps the cause.
            _LOGGER.info("The rating-table preview was refused.", exc_info=True)
            return RatingPreview(revision, None, rating_preview.refusal_reason(self.session.model))
        if payload is None:
            return RatingPreview(revision, None, rating_preview.SUPERSEDED)
        return RatingPreview(revision, payload, None)

    def _export_file(
        self,
        format: str,
```
(`EditorClientError` is caught before `ValueError` because `EditorValueError` subclasses both; the no-data case therefore shows the export's own sentence.)

**3c. `src/superglm/editor/server.py` (before line 179):**

Old:
```python
    @app.get("/download_export")
    def download_export(format: str = "joblib", filename: str | None = None) -> Response:
```
New:
```python
    @app.post("/rating_table")
    def rating_table(payload: dict[str, Any] = Body(default_factory=dict)) -> Response:
        return _guarded_json(lambda: widget._rating_table(str(_required(payload, "term"))))

    @app.get("/download_export")
    def download_export(format: str = "joblib", filename: str | None = None) -> Response:
```

- [ ] **Step 4: Run tests, expect PASS**

```bash
./.venv/bin/python -m pytest tests/test_editor_rating_preview.py -q
./.venv/bin/python -m pytest tests/test_editor.py -k "export or excel or rating" tests/test_editor_security.py tests/test_editor_validation_errors.py tests/test_rating_table_export.py -q
./.venv/bin/ruff check src/superglm/editor tests/test_editor_rating_preview.py
./.venv/bin/ruff format --check src/superglm/editor tests/test_editor_rating_preview.py
```
Expected: 6 passed in the new file; the existing export tests (which now go through `_rating_table_payload`) pass unchanged.

- [ ] **Step 5: Commit**

```bash
git add src/superglm/editor/rating_preview.py src/superglm/editor/widget.py src/superglm/editor/server.py tests/test_editor_rating_preview.py
git commit -m "$(cat <<'EOF'
Editor: rating-table preview route built by the Excel export's own path

POST /rating_table returns the current term's main-effect block of the
rating-table payload the workbook export builds, from the same training
split and materialised model, with the workbook's number formats and note.
One build per model revision, without the discretisation-impact sweep,
which only fills the impact sheet. Refusals are fixed sentences.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task K1b: Chart / Table switch and the table view (frontend)

**Files:**
- Create: `src/superglm/editor/app/views/rating_table.js`
- Modify: `src/superglm/editor/app/index.html` (after `.term-context` 187-192; after `<svg id="chart">` 235)
- Modify: `src/superglm/editor/app/main.js` (import 62, consts 87-88, `renderChartWorkspace` 794-805, `renderChartOnly` 819-822, `selectChartRenderState` 872, `sameChartRenderState` 886, after `contribPlay` listener 1504, subscriptions 1618)
- Modify: `src/superglm/editor/app/api/client.js` (82-87), `src/superglm/editor/app/api/contracts.js` (line 4, `EditorViewState` 179-182, before "What Undo and Redo" 123-124), `src/superglm/editor/app/state/store.js` (line 18)
- Modify: `src/superglm/editor/app/styles/chart.css` (before `.ui-popover`, 11), `src/superglm/editor/app/styles/shell.css` (after `.tool-rail svg` block, 174-182), `src/superglm/editor/app/views/help_content.js` (before "Features")
- Test: `tests/editor_frontend/rating_table.test.js` (new), `tests/editor/test_editor_rating_table_browser.py` (new)

**Interfaces:**
- Consumes: `POST /rating_table` (K1a, amendment 3); `selectActiveTermName`, `selectModelRevision` (`state/selectors.js:14`); `editorClient`.
- Produces:
  ```js
  // views/rating_table.js
  export const RATING_TABLE_LOADING, RATING_TABLE_FAILED  // fixed sentences
  export function ratingTableMessage(term, message) -> RatingTableModel
  export function ratingTableModel(response: RatingTableResponse) -> RatingTableModel
  export function formatRatingCell(value, format) -> string
  export function renderRatingTable(frame: HTMLElement, model: RatingTableModel)
  export function renderTermViewToggle(root: HTMLElement, view: TermView)
  export function bindTermViewToggle(root, {onChange}) -> {destroy}
  // api/client.js: createEditorClient() returns {..., ratingTable(term)}
  // store: view.termView "chart" | "table"; DOM #termViewToggle (radiogroup "Term view"), #ratingTableFrame
  ```

- [ ] **Step 1: Write the failing tests**

Create `tests/editor_frontend/rating_table.test.js`:

```js
// @ts-nocheck

import assert from "node:assert/strict";
import test from "node:test";

import {
  RATING_TABLE_FAILED,
  bindTermViewToggle,
  formatRatingCell,
  ratingTableModel,
  renderTermViewToggle,
} from "../../src/superglm/editor/app/views/rating_table.js";

test("cells read as the workbook's number formats show them", () => {
  assert.equal(formatRatingCell(1.0069302040629162, "0.000000"), "1.006930");
  assert.equal(formatRatingCell(12345.678, "#,##0.00"), "12,345.68");
  assert.equal(formatRatingCell(-0.012345678901234567, "0.000000000000"), "-0.012345678901");
  assert.equal(formatRatingCell(0.000123456789012345678, "0.00000000000000E+00"), "1.23456789012346E-04");
  assert.equal(formatRatingCell(4, null), "4");
  assert.equal(formatRatingCell("[18.0, 20.0)", null), "[18.0, 20.0)");
  assert.equal(formatRatingCell(null, "0.000000"), "");
});

test("an available block becomes a header and formatted rows", () => {
  const model = ratingTableModel({
    term: "band",
    available: true,
    reason: null,
    columns: ["band", "Relativity", "Weight"],
    rows: [["low", 1, 210.5], ["high", 1.25, 1530]],
    formats: [null, "0.000000", "#,##0.00"],
    note: null,
    model_revision: 3,
  });

  assert.equal(model.message, null);
  assert.deepEqual(model.header, ["band", "Relativity", "Weight"]);
  assert.deepEqual(model.body, [["low", "1.000000", "210.50"], ["high", "1.250000", "1,530.00"]]);
  assert.deepEqual(model.numeric, [false, true, true]);
});

test("a refused table shows its fixed reason", () => {
  const model = ratingTableModel({
    term: "x", available: false, reason: "No table.", columns: [], rows: [], formats: [],
    note: null, model_revision: 0,
  });
  assert.equal(model.message, "No table.");
  assert.equal(ratingTableModel({ ...model, available: false, reason: null }).message, RATING_TABLE_FAILED);
});

class FakeButton {
  constructor(view) {
    this.dataset = { termView: view };
    this.attributes = new Map();
    this.tabIndex = -1;
    this.classes = new Set();
    this.classList = { toggle: (name, on) => (on ? this.classes.add(name) : this.classes.delete(name)) };
  }

  setAttribute(name, value) { this.attributes.set(name, String(value)); }
  getAttribute(name) { return this.attributes.get(name) ?? null; }
  closest() { return this; }
  focus() { this.focused = true; }
}

globalThis.Element = FakeButton;
globalThis.HTMLElement = FakeButton;
globalThis.HTMLButtonElement = FakeButton;

test("the switch marks one view and reports clicks and arrows", () => {
  const buttons = [new FakeButton("chart"), new FakeButton("table")];
  const listeners = new Map();
  const root = {
    querySelectorAll: () => buttons,
    querySelector: (selector) => {
      if (selector.includes('aria-checked="true"')) {
        return buttons.find((button) => button.getAttribute("aria-checked") === "true") ?? null;
      }
      return buttons.find((button) => selector.includes(`"${button.dataset.termView}"`)) ?? null;
    },
    contains: (node) => buttons.includes(node),
    addEventListener: (name, listener) => listeners.set(name, listener),
    removeEventListener: (name) => listeners.delete(name),
  };
  const views = [];
  bindTermViewToggle(root, { onChange: (view) => views.push(view) });

  renderTermViewToggle(root, "table");
  assert.equal(buttons[1].getAttribute("aria-checked"), "true");
  assert.equal(buttons[1].tabIndex, 0);
  assert.equal(buttons[0].getAttribute("aria-checked"), "false");

  listeners.get("click")({ target: buttons[0] });
  listeners.get("keydown")({ key: "ArrowRight", preventDefault() {} });
  assert.deepEqual(views, ["chart", "chart"]);
  assert.equal(buttons[0].focused, true);
});
```

Create `tests/editor/test_editor_rating_table_browser.py`:

```python
from __future__ import annotations

from urllib.parse import urlsplit

import pytest

pytest.importorskip("playwright.sync_api")
pytestmark = pytest.mark.browser


def _rating_table_for(term: str):
    return lambda response: (
        response.request.method == "POST"
        and urlsplit(response.url).path == "/rating_table"
        and response.request.post_data_json == {"term": term}
    )


def _term_view(page, name: str):
    return page.get_by_role("radiogroup", name="Term view").get_by_role(
        "radio", name=name, exact=True
    )


def test_table_shows_the_terms_rating_table_block_in_place_of_the_chart(
    open_editor_page, choose_feature
):
    with open_editor_page(selected_term="age_band") as (page, _session):
        with page.expect_response(_rating_table_for("age_band")) as response_info:
            _term_view(page, "Table").click()
        block = response_info.value.json()

        frame = page.locator("#ratingTableFrame")
        frame.locator("table.rating-table").wait_for()
        assert page.locator("#chart").is_hidden()
        assert page.locator("#ciToggle").is_hidden()
        assert _term_view(page, "Table").get_attribute("aria-checked") == "true"
        assert block["available"] is True
        assert frame.locator("thead th").all_inner_texts() == block["columns"]
        relativity = block["columns"].index("Relativity")
        weight = block["columns"].index("Weight")
        first = frame.locator("tbody tr").first.locator("td").all_inner_texts()
        assert first[0] == str(block["rows"][0][0])
        assert first[relativity] == f"{block['rows'][0][relativity]:.6f}"
        assert first[weight] == f"{block['rows'][0][weight]:,.2f}"
        assert frame.locator("tbody tr").count() == len(block["rows"])

        with page.expect_response(_rating_table_for("territory")):
            choose_feature(page, "territory")
        page.wait_for_function(
            "() => document.querySelector('#ratingTableFrame thead th')?.textContent"
            " === 'territory'"
        )

        _term_view(page, "Chart").click()
        page.locator("#chart path.edited").first.wait_for()
        assert frame.is_hidden()
        assert page.locator("#chart").is_visible()
```

- [ ] **Step 2: Run them, expect FAIL**

```bash
node --test tests/editor_frontend/rating_table.test.js
./.venv/bin/python -m pytest tests/editor/test_editor_rating_table_browser.py -m browser --run-browser -q
```
Expected on 155832e8 (with K1a in place): node cannot load `views/rating_table.js` (`ERR_MODULE_NOT_FOUND`); browser fails with a 30 s locator timeout waiting for the `radiogroup` named "Term view" (there is no Chart/Table switch).

- [ ] **Step 3: Implement**

**3a. Create `src/superglm/editor/app/views/rating_table.js`:**

```js
// @ts-check

/**
 * The Chart / Table switch and the rating-table view: the current term's
 * block of the Excel rating table, with the number formats and the note the
 * workbook gives it.
 */

/** @typedef {import('../api/contracts.js').RatingTableResponse} RatingTableResponse */
/** @typedef {import('../api/contracts.js').TermView} TermView */

export const RATING_TABLE_LOADING = "Building the rating table...";
export const RATING_TABLE_FAILED = "The rating table could not be built.";

/**
 * What the table view shows: a message, or a header and formatted rows.
 * @typedef {Object} RatingTableModel
 * @property {string} term
 * @property {string|null} message
 * @property {string[]} header
 * @property {string[][]} body
 * @property {boolean[]} numeric
 * @property {string|null} note
 */

/**
 * @param {string} term
 * @param {string} message
 * @returns {RatingTableModel}
 */
export function ratingTableMessage(term, message) {
  return { term, message, header: [], body: [], numeric: [], note: null };
}

/**
 * @param {RatingTableResponse} response
 * @returns {RatingTableModel}
 */
export function ratingTableModel(response) {
  if (!response.available) {
    return ratingTableMessage(response.term, response.reason || RATING_TABLE_FAILED);
  }
  const numeric = response.columns.map((_, column) =>
    response.rows.some((row) => typeof row[column] === "number")
  );
  return {
    term: response.term,
    message: null,
    header: response.columns.slice(),
    body: response.rows.map((row) =>
      row.map((value, column) => formatRatingCell(value, response.formats[column] ?? null))
    ),
    numeric,
    note: response.note,
  };
}

/**
 * One cell as the workbook's number format shows it. The export applies four
 * formats: fixed places ("0.000000", "0.000000000000"), grouped with two
 * places ("#,##0.00") and scientific ("0.00000000000000E+00"); any other
 * cell is shown as it is.
 *
 * @param {string|number|boolean|null} value
 * @param {string|null} format
 * @returns {string}
 */
export function formatRatingCell(value, format) {
  if (value === null) return "";
  if (typeof value !== "number") return String(value);
  if (!Number.isFinite(value)) return "";
  if (format === "#,##0.00") {
    return value.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  }
  const fixed = format ? /^0\.(0+)$/.exec(format) : null;
  if (fixed) return value.toFixed(fixed[1].length);
  const scientific = format ? /^0\.(0+)E\+00$/.exec(format) : null;
  if (scientific) {
    const [mantissa, exponent] = value.toExponential(scientific[1].length).split("e");
    const power = Number(exponent);
    return `${mantissa}E${power < 0 ? "-" : "+"}${String(Math.abs(power)).padStart(2, "0")}`;
  }
  return String(value);
}

/**
 * @param {HTMLElement} frame
 * @param {RatingTableModel} model
 */
export function renderRatingTable(frame, model) {
  const doc = frame.ownerDocument;
  frame.dataset.term = model.term;
  if (model.message !== null) {
    const message = doc.createElement("p");
    message.className = "rating-table-message";
    message.textContent = model.message;
    frame.replaceChildren(message);
    return;
  }
  const table = doc.createElement("table");
  table.className = "rating-table";
  const caption = doc.createElement("caption");
  caption.textContent = `${model.term} · as written to the Excel rating table`;
  const headRow = doc.createElement("tr");
  model.header.forEach((label, column) => {
    const cell = doc.createElement("th");
    cell.scope = "col";
    cell.textContent = label;
    if (model.numeric[column]) cell.className = "numeric";
    headRow.append(cell);
  });
  const head = doc.createElement("thead");
  head.append(headRow);
  const body = doc.createElement("tbody");
  for (const values of model.body) {
    const row = doc.createElement("tr");
    values.forEach((text, column) => {
      const cell = doc.createElement("td");
      cell.textContent = text;
      if (model.numeric[column]) cell.className = "numeric";
      row.append(cell);
    });
    body.append(row);
  }
  table.append(caption, head, body);
  if (!model.note) {
    frame.replaceChildren(table);
    return;
  }
  // The workbook writes a block's note above its header.
  const note = doc.createElement("p");
  note.className = "rating-table-note";
  note.textContent = model.note;
  frame.replaceChildren(note, table);
}

/**
 * @param {HTMLElement} root
 * @param {TermView} view
 */
export function renderTermViewToggle(root, view) {
  for (const button of root.querySelectorAll("[data-term-view]")) {
    if (!(button instanceof HTMLButtonElement)) continue;
    const active = button.dataset.termView === view;
    button.setAttribute("aria-checked", String(active));
    button.tabIndex = active ? 0 : -1;
    button.classList.toggle("active", active);
  }
}

/**
 * @param {HTMLElement} root
 * @param {{onChange:(view:TermView)=>unknown}} options
 * @returns {{destroy:()=>void}}
 */
export function bindTermViewToggle(root, { onChange }) {
  /** @param {MouseEvent} event */
  function onClick(event) {
    const button = event.target instanceof Element ? event.target.closest("[data-term-view]") : null;
    if (!(button instanceof HTMLButtonElement) || !root.contains(button)) return;
    const view = button.dataset.termView;
    if (view === "chart" || view === "table") onChange(view);
  }

  /** @param {KeyboardEvent} event */
  function onKeyDown(event) {
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
    const current = root.querySelector('[data-term-view][aria-checked="true"]');
    /** @type {TermView} */
    const next = current instanceof HTMLElement && current.dataset.termView === "table"
      ? "chart"
      : "table";
    event.preventDefault();
    onChange(next);
    const button = root.querySelector(`[data-term-view="${next}"]`);
    if (button instanceof HTMLElement) button.focus();
  }

  root.addEventListener("click", onClick);
  root.addEventListener("keydown", onKeyDown);
  return Object.freeze({
    destroy() {
      root.removeEventListener("click", onClick);
      root.removeEventListener("keydown", onKeyDown);
    },
  });
}
```

**3b. `src/superglm/editor/app/index.html`:**

Old (lines 191-192):
```html
          <span id="termReference" class="context-chip" hidden></span>
        </div>
```
New:
```html
          <span id="termReference" class="context-chip" hidden></span>
        </div>
        <div id="termViewToggle" class="tool-rail term-view-toggle" role="radiogroup"
          aria-label="Term view">
          <div class="tool-rail-modes">
            <button type="button" role="radio" aria-checked="true" aria-label="Chart"
              data-term-view="chart" data-popover-title="Chart"
              data-popover-body="Show the term's curve.">
              <svg viewBox="0 0 24 24" aria-hidden="true">
                <path d="M4 19h16M5 15l4-5 4 3 6-7"></path>
              </svg>
            </button>
            <button type="button" role="radio" aria-checked="false" aria-label="Table"
              data-term-view="table" tabindex="-1" data-popover-title="Rating table"
              data-popover-body="Show this term's block of the Excel rating table, built by the same code as the export.">
              <svg viewBox="0 0 24 24" aria-hidden="true">
                <rect x="4" y="5" width="16" height="14" rx="1.5"></rect>
                <path d="M4 10h16M4 14.5h16M10 5v14"></path>
              </svg>
            </button>
          </div>
        </div>
```
Old (line 235):
```html
        <svg id="chart" viewBox="0 0 940 520" role="img" aria-label="SuperGLM editable effect"></svg>
```
New:
```html
        <svg id="chart" viewBox="0 0 940 520" role="img" aria-label="SuperGLM editable effect"></svg>
        <section id="ratingTableFrame" class="rating-table-frame" aria-label="Rating table"
          aria-live="polite" hidden></section>
```

**3c. `src/superglm/editor/app/api/client.js` (lines 82-87):**

Old:
```js
  /** @returns {Promise<unknown>} */
  function getState() {
    return requestJSON("/state");
  }

  return { requestJSON, postJSON, requestBlob, getState };
```
New:
```js
  /** @returns {Promise<unknown>} */
  function getState() {
    return requestJSON("/state");
  }

  /** @param {string} term @returns {Promise<unknown>} */
  function ratingTable(term) {
    return postJSON("/rating_table", { term });
  }

  return { requestJSON, postJSON, requestBlob, getState, ratingTable };
```

**3d. `src/superglm/editor/app/api/contracts.js`:**

Old (line 4):
```js
/** @typedef {'select'|'move'|'zoom'|'handles'} EditorMode */
```
New:
```js
/** @typedef {'select'|'move'|'zoom'|'handles'} EditorMode */
/** @typedef {'chart'|'table'} TermView */
```
Old (lines 179-182):
```js
 * @typedef {Object} EditorViewState
 * @property {string} activeTerm
 * @property {AppView} activeView
 * @property {EditorMode} mode
```
New:
```js
 * @typedef {Object} EditorViewState
 * @property {string} activeTerm
 * @property {AppView} activeView
 * @property {EditorMode} mode
 * @property {TermView} termView
```
Old (lines 123-124):
```js
/**
 * What Undo and Redo would take next, edits and structural steps alike; null
```
New:
```js
/**
 * The /rating_table response: the term's main-effect block of the Excel
 * export, with the number format and the note the workbook gives it, and
 * ``available`` false with a fixed ``reason`` when there is no table.
 * @typedef {Object} RatingTableResponse
 * @property {string} term
 * @property {boolean} available
 * @property {string|null} reason
 * @property {string[]} columns
 * @property {Array<Array<string|number|boolean|null>>} rows
 * @property {Array<string|null>} formats
 * @property {string|null} note
 * @property {number} model_revision
 */
/**
 * What Undo and Redo would take next, edits and structural steps alike; null
```

**3e. `src/superglm/editor/app/state/store.js` (lines 17-18):**

Old:
```js
      activeView: "editor",
      mode: "select",
```
New:
```js
      activeView: "editor",
      mode: "select",
      termView: "chart",
```

**3f. `src/superglm/editor/app/main.js`:**

Old (line 62):
```js
import { bindToolRail, renderToolRail } from "./views/tool_rail.js";
```
New:
```js
import {
  RATING_TABLE_FAILED,
  RATING_TABLE_LOADING,
  bindTermViewToggle,
  ratingTableMessage,
  ratingTableModel,
  renderRatingTable,
  renderTermViewToggle
} from "./views/rating_table.js";
import { bindToolRail, renderToolRail } from "./views/tool_rail.js";
```
Old (lines 87-88):
```js
const svg = document.getElementById("chart");
const selectionMenu = document.getElementById("selectionMenu");
```
New:
```js
const svg = document.getElementById("chart");
const selectionMenu = document.getElementById("selectionMenu");
const plotColumn = document.querySelector(".plot-column");
const termViewToggle = document.getElementById("termViewToggle");
const ratingTableFrame = document.getElementById("ratingTableFrame");
```
Old (lines 794-795):
```js
  if (applyTermDefaults(term)) return;
  const selection = view.preview && view.preview.term === selected
```
New:
```js
  if (applyTermDefaults(term)) return;
  const tableView = renderTermView(view.termView);
  const selection = view.preview && view.preview.term === selected
```
Old (lines 804-806):
```js
  updateResetOrderAction(term);
  drawChart(term, selection, chartContext);
  const collapsedOriginalNote = selectionContextNote(term);
```
New:
```js
  updateResetOrderAction(term);
  if (!tableView) drawChart(term, selection, chartContext);
  const collapsedOriginalNote = selectionContextNote(term);
```
Old (lines 819-822):
```js
function renderChartOnly() {
  const state = store.getState();
  const term = currentTerm();
  if (!state.remote.snapshot || !term) return;
```
New:
```js
// Table puts the term's rating-table block where the chart was; the chart
// keeps its mode, zoom and selection for when Chart comes back.
function renderTermView(termView) {
  const tableView = termView === "table";
  renderTermViewToggle(termViewToggle, termView);
  plotColumn.classList.toggle("is-table-view", tableView);
  // An SVG element has no `hidden` property; the attribute is what CSS hides.
  svg.toggleAttribute("hidden", tableView);
  ratingTableFrame.hidden = !tableView;
  if (tableView) stopContributionBuild();
  return tableView;
}

let ratingTableSequence = 0;

function selectRatingTableRequest(state) {
  return {
    table: state.view.termView === "table",
    term: selectActiveTermName(state),
    revision: selectModelRevision(state)
  };
}

function sameRatingTableRequest(next, previous) {
  return next.table === previous.table &&
    next.term === previous.term &&
    next.revision === previous.revision;
}

// One request per term and model revision while Table is shown; a reply
// that a newer request has overtaken is dropped.
async function refreshRatingTable({ table, term, revision }) {
  if (!table || !term || revision < 0) return;
  const sequence = ++ratingTableSequence;
  renderRatingTable(ratingTableFrame, ratingTableMessage(term, RATING_TABLE_LOADING));
  let model;
  try {
    model = ratingTableModel(await editorClient.ratingTable(term));
  } catch {
    model = ratingTableMessage(term, RATING_TABLE_FAILED);
  }
  if (sequence === ratingTableSequence) renderRatingTable(ratingTableFrame, model);
}

function renderChartOnly() {
  const state = store.getState();
  const term = currentTerm();
  if (!state.remote.snapshot || !term || state.view.termView === "table") return;
```
Old (line 872):
```js
    mode: view.mode,
    showCi: view.showCi,
```
New:
```js
    mode: view.mode,
    termView: view.termView,
    showCi: view.showCi,
```
Old (line 886):
```js
    next.mode === previous.mode &&
    next.showCi === previous.showCi &&
```
New:
```js
    next.mode === previous.mode &&
    next.termView === previous.termView &&
    next.showCi === previous.showCi &&
```
Old (line 1504):
```js
contribPlay.addEventListener("click", startContributionBuild);
```
New:
```js
contribPlay.addEventListener("click", startContributionBuild);

bindTermViewToggle(termViewToggle, {
  onChange: (termView) => actions.patchView({ termView })
});
```
Old (line 1618):
```js
store.subscribe(selectChartRenderState, () => renderChartWorkspace(), sameChartRenderState);
```
New:
```js
store.subscribe(selectChartRenderState, () => renderChartWorkspace(), sameChartRenderState);
store.subscribe(selectRatingTableRequest, refreshRatingTable, sameRatingTableRequest);
```
(A failed request shows the fixed `RATING_TABLE_FAILED`, never the server's text; a refused export arrives as `available: false` with K1a's fixed reason.)

**3g. `src/superglm/editor/app/styles/shell.css` — after the `.tool-rail svg { … }` block (ends line 182):**

Old:
```css
.tool-rail svg {
  width: 17px;
  height: 17px;
  fill: none;
  stroke: currentColor;
  stroke-width: 1.7;
  stroke-linecap: round;
  stroke-linejoin: round;
}
```
New:
```css
.tool-rail svg {
  width: 17px;
  height: 17px;
  fill: none;
  stroke: currentColor;
  stroke-width: 1.7;
  stroke-linecap: round;
  stroke-linejoin: round;
}

/* The Chart / Table switch borrows the tool rail's segmented look. */
.term-view-toggle {
  margin-right: 0;
  margin-left: 2px;
}

/* Table view: the rating table takes the chart's place, and the controls
   that only act on the chart step aside. */
.plot-column.is-table-view :is(#handleCountWrap, #basisToggle, #contribPlay, #ciToggle,
  #resetZoom, #selectionMenu) {
  display: none !important;
}
```

**3h. `src/superglm/editor/app/styles/chart.css` — before `.ui-popover {` (line 11):**

Old:
```css
.ui-popover {
```
New:
```css
.rating-table-frame {
  height: 100%;
  min-height: 360px;
  overflow: auto;
  padding: 4px 2px;
}

.rating-table {
  border-collapse: collapse;
  font-variant-numeric: tabular-nums;
  font-size: 13px;
}

.rating-table caption {
  padding: 0 0 6px;
  color: var(--muted);
  text-align: left;
  white-space: nowrap;
}

.rating-table th,
.rating-table td {
  padding: 3px 12px 3px 0;
  border-bottom: 1px solid var(--border);
  text-align: left;
  white-space: nowrap;
}

.rating-table th {
  font-weight: 600;
}

.rating-table .numeric {
  text-align: right;
}

.rating-table td.numeric {
  font-family: var(--font-mono);
}

.rating-table-note,
.rating-table-message {
  max-width: 72ch;
  margin: 0 0 8px;
  color: var(--muted);
}

.ui-popover {
```

**3i. `src/superglm/editor/app/views/help_content.js` — before the "Features" section:**

Old:
```js
  Object.freeze({
    title: "Features",
```
New:
```js
  Object.freeze({
    title: "Rating table",
    items: Object.freeze([
      "The Chart / Table switch above the chart shows the current term's block of the Excel rating table instead of its curve, with the workbook's number formats and note. It is built by the same code as the export, on the training data, and follows every edit.",
      "Interactions are not shown. A model the export refuses shows the reason instead of a table.",
    ]),
  }),
  Object.freeze({
    title: "Features",
```

- [ ] **Step 4: Run tests, expect PASS**

```bash
npm run check:frontend
./.venv/bin/python -m pytest tests/test_editor_browser.py tests/editor -m browser --run-browser -q
./.venv/bin/python -m pytest tests/test_editor.py -k "frontend or asset" -q
```
Expected: `tsc` clean, node `# fail 0`; browser all pass (88 with D2); the asset-serving tests pass (new files are picked up by the package's `rglob`, `tests/test_supply_chain_governance.py:370-374`).

- [ ] **Step 5: Commit**

```bash
git add src/superglm/editor/app/views/rating_table.js src/superglm/editor/app/index.html \
  src/superglm/editor/app/main.js src/superglm/editor/app/api/client.js \
  src/superglm/editor/app/api/contracts.js src/superglm/editor/app/state/store.js \
  src/superglm/editor/app/styles/chart.css src/superglm/editor/app/styles/shell.css \
  src/superglm/editor/app/views/help_content.js tests/editor_frontend/rating_table.test.js \
  tests/editor/test_editor_rating_table_browser.py
git commit -m "$(cat <<'EOF'
Editor: Chart / Table switch showing the term's rating-table block

A two-icon switch in the context bar replaces the chart with the current
term's block of the Excel rating table, formatted as the workbook formats
it, refetched per term and model revision. Chart-only controls step aside
while the table shows.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Decisions taken here (binding on implementers, open to review)

- **Handles sit at the coefficients (the control polygon), as numeric-spline handles do.** The spec binds "moving a handle sets level effects to `B(level positions)·c`", so a handle's value is `c_j`; the mock-up's on-curve handles were illustrative (its handle y was the curve at the Greville site). On a P-spline with a weakly identified end coefficient a handle can sit well off the curve (measured on freMTPL2 `VehPower`: one end handle at relativity 0.2 under a curve near 1.0). Making handles sit on the curve would need a different drag rule (`Δc_j = Δy / B_j(x_j)`) and, for consistency, the same change on numeric splines — a follow-up if Max wants it.
- **A level edit the basis cannot follow (`K < S`) is drawn as the polyline** (`fits_levels: false`), and the next handle move snaps every smooth level onto the least-squares spline — the same as a numeric spline's handle move after a point edit.
- **Handles stay on while a shape or collapse on the term is only waiting** (Task A): the drawn spline and the handles are the in-force fit's; D2's rule drops the hand edits on that term at Refit anyway.

### Follow-ups found (pre-existing; not fixed here)

1. **Browser suite is order-dependent.** On 155832e8, `pytest tests/editor tests/test_editor_browser.py -m browser --run-browser` fails 10 tests in `test_editor_browser.py`; the header's order (`tests/test_editor_browser.py tests/editor`) passes all. Each file is green alone. Likely two sync Playwright instances (the session-scoped `chromium_browser` in `tests/editor/conftest.py` and the one in `tests/test_editor_browser.py`) in one process.
2. **Workbook cells carry 16 significant digits, not the float64 value.** openpyxl serialises floats as `"%.16g"`, so `RatingTableBlock` values reach the `.xlsx` with up to `5e-16` relative error. That is inside the export docstring's measured 4.4e-16–6.2e-16 product error, but the docstring's "exact" wording for categorical blocks and its offset-mapping discussion assume the cell is the float.
3. **Another local `_gamma`/`_UNIT_ROUNDOFF` pair** (`controls.py`) joins the duplicates already in `reml/identified.py`, `reml/multi_penalty.py`, `diagnostics/exact_banding.py`; a shared float64-analysis helper module is the cleanup.


## Phase 6 — Dark theme and theme switch (I1–I3)

Spec §4 I, decisions D9 (gruvbox-family warm dark) and D10 (the comic pill,
animated by CSS keyframes, no starburst; "Follow the browser" lives in
Settings). Three tasks, each rejectable on its own:

- **I1** the warm dark palette, held to 4.5:1 text contrast and separable
  categorical palettes. Touches CSS and one new test only; it does not need F1
  and can run in any phase.
- **I2** the DAY/NIGHT switch and its state: the explicit choice, "Follow the
  browser" through Settings, first paint, blocked storage. Needs F1.
- **I3** the flip animation and the page's cross-fade, with reduced motion.

Every code block below was run in a scratch copy of the package at 155832e8
(plus a stand-in for F1's `settings.js`): `node --test` on all 17 frontend
files, `tsc -p jsconfig.json`, and all 88 editor browser tests passed after
each of I2 and I3. The failure on master and the mutation checks quoted in the
steps are from those runs.

**Research gate (stages 1–2).** The palette work is two textbook checks, not
new territory:

- text legibility is WCAG 2.2 contrast, (L1 + 0.05)/(L2 + 0.05) over relative
  luminance, ≥ 4.5:1 (SC 1.4.3), and marks against their ground ≥ 3:1
  (SC 1.4.11);
- categorical separability is colour difference in OKLab (Ottosson, *A
  perceptual color space for image processing*, 2020; coefficients checked
  against his post, public domain/MIT) under normal vision and under
  protanopia/deuteranopia simulated with Machado, Oliveira and Fernandes,
  *A Physiologically-based Model for Simulation of Color Vision Deficiency*,
  IEEE TVCG 15(6), 2009 (severity-1.0 matrices checked against the authors'
  page), plus the spec's rule that neighbours also differ in lightness.

The gruvbox hexes were read from `colors/gruvbox.vim` in morhetz/gruvbox,
whose README states MIT/X11. Values are cited, no code is copied. No existing
palette validation exists in the repo (searched tests and scripts for
luminance, contrast, OKLab and ΔE), so I1 writes one.

### Contract amendments

1. **`views/theme.js` API.** The cycling API is replaced.
   - Removed: `nextThemeChoice`, `describeThemeControl`, `renderThemeControl`
     and `mountThemeControl`. Only `main.js` and `theme.test.js` used them.
   - Kept: `THEME_STORAGE_KEY`, `isThemeChoice`, `readThemeChoice(storage?)`,
     `storeThemeChoice(choice, storage?)` and
     `resolveTheme(choice, prefersDark)`.
   - Added in I2: `describeThemeSwitch(choice, prefersDark) -> {checked, title, body}`,
     `renderThemeSwitch(button, choice, prefersDark)` and
     `mountThemeSwitch({button, root, media, settings, storage?}) -> {destroy}`.
     `settings` is `{load, save, subscribe}`, which `main.js` builds from F1's
     `loadSettings`, `saveSettings` and `onSettingsChange`.
   - Added in I3: `FLIP_CLASS = "is-flipping"` and `FADE_CLASS = "theme-fading"`.
2. **What I2 assumes of F1's `settings.js`.**
   - `loadSettings().followBrowserTheme` is a boolean.
   - `saveSettings(patch)` calls every `onSettingsChange` listener
     synchronously with the merged settings, in the page that saved too, and
     keeps them in memory when storage is blocked.
   - F1's Settings control only calls `saveSettings({followBrowserTheme})`.
     It never touches the `superglm.editor.theme` key or `<html data-theme>`;
     theme.js owns both.
   - The control's accessible label is the spec's text,
     `Follow the browser's light or dark setting`. The browser test finds it by
     that label.
   - If F1 landed differently, adapt the three-line `settings` adapter in
     `main.js`, not theme.js.
3. **Which store decides.** The `superglm.editor.theme` key decides. A stored
   `light` or `dark` is an explicit choice; no stored value means follow the
   browser. `followBrowserTheme` mirrors the key, and theme.js keeps the two
   equal:
   - on mount it reconciles them: an old explicit choice turns the setting
     off, and a missing key turns it on;
   - a flip stores the theme and turns the setting off;
   - turning the setting on in Settings removes the key;
   - turning the setting off in Settings stores the theme on screen.

   So the first-paint script keeps reading the theme key alone, unchanged, and
   a choice made with today's cycling icon survives the upgrade.
4. **DOM.** `#themeSwitch` is a
   `<button type="button" role="switch" aria-checked aria-label="Dark theme">`
   holding `.theme-switch-label[data-label="day"|"night"]`,
   `.theme-switch-knob` and `.theme-switch-icon[data-icon="sun"|"moon"]`.
   `aria-checked="true"` means Night. In I3, `<html>` carries `.theme-fading`
   while a flip's cross-fade runs.
5. **Tokens.** I2 adds `--switch-track` and `--switch-dots` to `tokens.css`
   and `dark.css`. `editor_style.py` checks only its own subset, so it is
   unaffected.
6. **Board values changed by the validation.** The approved boards fail the
   checks the spec asks for, so I1 re-steps these within gruvbox, holding the
   hue:

   | Token | Board | Shipped | Failed check |
   |---|---|---|---|
   | `--surface-hover` | #3c3836 | #32302f (dark0_soft) | `--muted` on it 4.17:1 |
   | `--blue-soft` | #2f3a4f | #283244 | `--muted` on it 4.11:1 (the checked export card) |
   | `--worse` / `--danger` | #fb4934 | #fb7a6b (the board's sig-none text) | 4.29:1 on the inspector's #282828 |
   | `--trace-2` (fold 3) | #d3869b | #b16286 (neutral purple) | 14.2 OKLab from fold 4 (< 15); lightness 0.026 apart |
   | `--trace-4` (fold 5) | #8ec07c | #689d6a (neutral aqua) | lightness 0.024 from fold 4 |

   G5 must colour fold *k* with `var(--trace-k)`. If G5 draws the CV board's
   lighter exposure (0.35 in light), its dark value is 0.55, scoped to
   `:root[data-theme="dark"]` like I1's exposure rule.

---

### Task I1: Warm dark palette, contrast and separability test

**Files:**
- Modify: `src/superglm/editor/app/styles/dark.css`. Replace the whole file,
  lines 1–104 at 155832e8.
- Test: `tests/editor_frontend/dark_palette.test.js` (new).
- Unchanged, but read: `styles/tokens.css` (the light values the test also
  checks); `styles.css:1063–1064` (`.exposure` and `.exposure-density` at
  `fill-opacity: 0.6`) and `:1158` (`.legend-swatch` at 0.45);
  `plotting/editor_style.py`, which reads only `tokens.css` and `styles.css`
  and no dark token, so it is not restated.

**Interfaces:**
- Consumes: the custom-property names in `tokens.css`, and the slot counts
  `chart.js` cycles (6 group, 12 basis) and `summary.js` cycles (10 trace).
- Produces: the dark token values below, which I2, I3 and G5 consume.

- [ ] **Step 1: Write the failing test.** Create
  `tests/editor_frontend/dark_palette.test.js`. It is type-checked, so it has
  no `@ts-nocheck`.

```js
// The dark palette's legibility and separability, computed from the CSS.
//
// Text: WCAG 2.2 contrast, (L1 + 0.05) / (L2 + 0.05) over relative luminance,
// at least 4.5:1 (success criterion 1.4.3) for every text token on every
// ground a rule sets it on, in both themes.
//
// Categorical palettes, the slots a chart has to tell apart: every mark at
// least 3:1 on the chart's ground (1.4.11), and every two neighbouring slots,
// the wrap from the last to the first included because the chart cycles them,
//   - at least 15 apart in normal vision and 6 apart under protanopia and
//     deuteranopia, as Euclidean distance x100 in OKLab (Ottosson, "A
//     perceptual color space for image processing", 2020), the deficiencies
//     simulated in linear sRGB with Machado, Oliveira and Fernandes, "A
//     Physiologically-based Model for Simulation of Color Vision Deficiency",
//     IEEE TVCG 15(6), 2009, at severity 1.0;
//   - at least 0.06 apart in OKLab lightness, so neighbours stay apart when
//     hue is lost (spec section 4 I: colours that must be told apart also
//     differ in lightness).
// The 6 floor leans on a second channel, which each palette has: the group
// label under the axis, the Build's one highlighted basis, the legends.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

/** @typedef {[number, number, number]} Triple */

/** @param {string} path */
function appFile(path) {
  return readFileSync(new URL(`../../src/superglm/editor/app/${path}`, import.meta.url), "utf8");
}

/**
 * The custom properties declared in the first rule with this selector.
 * @param {string} css @param {string} selector @returns {Map<string, string>}
 */
function declared(css, selector) {
  const start = css.indexOf(`${selector} {`);
  assert.notEqual(start, -1, `no ${selector} rule`);
  const body = css.slice(start, css.indexOf("}", start));
  return new Map([...body.matchAll(/(--[\w-]+):\s*([^;]+);/g)].map((match) => [match[1], match[2].trim()]));
}

const LIGHT = declared(appFile("styles/tokens.css"), ":root");
const DARK = new Map([...LIGHT, ...declared(appFile("styles/dark.css"), ':root[data-theme="dark"]')]);
const THEMES = /** @type {const} */ ([["light", LIGHT], ["dark", DARK]]);

/** @param {Map<string, string>} theme @param {string} token @returns {Triple} sRGB in [0, 1] */
function srgb(theme, token) {
  const value = theme.get(token) ?? "";
  const match = /^#([0-9a-f]{6})$/i.exec(value);
  assert.ok(match, `${token} is ${value || "missing"}, not a six-digit hex colour`);
  const n = Number.parseInt(match[1], 16);
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
}

// The sRGB transfer function inverted (IEC 61966-2-1). WCAG 2.0 wrote the
// knee as 0.03928; no 8-bit channel lies between the two.
/** @param {number} c */
const decode = (c) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);

/** @param {Map<string, string>} theme @param {string} token @returns {Triple} */
function linearRgb(theme, token) {
  const [r, g, b] = srgb(theme, token);
  return [decode(r), decode(g), decode(b)];
}

/** @param {Map<string, string>} theme @param {string} token */
function luminance(theme, token) {
  const [r, g, b] = linearRgb(theme, token);
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

/** @param {Map<string, string>} theme @param {string} a @param {string} b */
function contrast(theme, a, b) {
  const [high, low] = [luminance(theme, a), luminance(theme, b)].sort((x, y) => y - x);
  return (high + 0.05) / (low + 0.05);
}

/** @param {Triple} rgb linear sRGB @returns {Triple} OKLab L, a, b */
function oklab([r, g, b]) {
  const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
  const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
  const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
  return [
    0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s,
    1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s,
    0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s,
  ];
}

// Machado, Oliveira and Fernandes (2009), severity 1.0, on linear sRGB.
/** @type {Record<string, Triple[]>} */
const DEFICIENCIES = {
  protanopia: [[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]],
  deuteranopia: [[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.01182, 0.04294, 0.968881]],
};

/** @param {Triple[]} matrix @param {Triple} rgb @returns {Triple} */
function simulate(matrix, rgb) {
  /** @param {Triple} row */
  const channel = (row) => Math.min(1, Math.max(0, row[0] * rgb[0] + row[1] * rgb[1] + row[2] * rgb[2]));
  return [channel(matrix[0]), channel(matrix[1]), channel(matrix[2])];
}

/** @param {Triple} p @param {Triple} q */
const distance = (p, q) => 100 * Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]);

/** @param {Map<string, string>} theme @param {string} name @returns {string[]} */
function slots(theme, name) {
  const tokens = [];
  while (theme.has(`--${name}-${tokens.length}`)) tokens.push(`--${name}-${tokens.length}`);
  return tokens;
}

// Text tokens and the grounds the stylesheets set them on.
/** @type {[string, string[]][]} */
const TEXT_ON_GROUND = [
  // Body text; the hover ground under hovered rows; .export-format-card when checked.
  ["--text", ["--surface", "--surface-subtle", "--surface-hover", "--blue-soft"]],
  // .history-chip and .se-cell.sig-unknown sit on the hover ground, the
  // checked export card's <small> on blue-soft.
  ["--muted", ["--surface", "--surface-subtle", "--surface-hover", "--blue-soft"]],
  // .tool-rail button.active, .selection-item:focus-visible, .context-bar button[aria-pressed].
  ["--blue", ["--surface", "--surface-subtle", "--blue-soft"]],
  // #status.is-error on the workspace, .summary-table td.advisory-code in the inspector.
  ["--danger", ["--surface", "--surface-subtle"]],
  // Metric and report deltas.
  ["--better", ["--surface", "--surface-subtle"]],
  ["--worse", ["--surface", "--surface-subtle"]],
  // .app-alert and its buttons.
  ["--danger-text", ["--danger-surface", "--surface"]],
  // button.primary and .ui-popover, and the popover's secondary line.
  ["--surface", ["--text", "--primary-hover"]],
  ["--popover-muted", ["--text"]],
  ["--sig-strong-fg", ["--sig-strong-bg"]],
  ["--sig-medium-fg", ["--sig-medium-bg"]],
  ["--sig-standard-fg", ["--sig-standard-bg"]],
  ["--sig-weak-fg", ["--sig-weak-bg"]],
  ["--sig-none-fg", ["--sig-none-bg"]],
];

test("every text token keeps 4.5:1 on every ground it is set on, in both themes", () => {
  const failures = [];
  for (const [themeName, theme] of THEMES) {
    for (const [text, grounds] of TEXT_ON_GROUND) {
      for (const ground of grounds) {
        const ratio = contrast(theme, text, ground);
        if (!(ratio >= 4.5)) failures.push(`${themeName} ${text} on ${ground}: ${ratio.toFixed(2)}:1`);
      }
    }
  }
  assert.deepEqual(failures, []);
});

test("the dark categorical palettes keep neighbours apart in colour and in lightness", () => {
  // chart.js cycles six group and twelve basis colours, summary.js ten trace colours.
  const palettes = /** @type {const} */ ([["group", 6], ["basis", 12], ["trace", 10]]);
  const failures = [];
  for (const [name, count] of palettes) {
    const tokens = slots(DARK, name);
    assert.equal(tokens.length, count, `--${name}-* slots`);
    for (const token of tokens) {
      const ratio = contrast(DARK, token, "--surface");
      if (!(ratio >= 3)) failures.push(`${token} on --surface: ${ratio.toFixed(2)}:1`);
    }
    tokens.forEach((token, i) => {
      const next = tokens[(i + 1) % tokens.length];
      const p = linearRgb(DARK, token);
      const q = linearRgb(DARK, next);
      const normal = distance(oklab(p), oklab(q));
      const deficient = Math.min(
        ...Object.values(DEFICIENCIES).map((matrix) => distance(oklab(simulate(matrix, p)), oklab(simulate(matrix, q)))),
      );
      const lightness = Math.abs(oklab(p)[0] - oklab(q)[0]);
      if (!(normal >= 15)) failures.push(`${token}/${next}: ${normal.toFixed(1)} apart`);
      if (!(deficient >= 6)) failures.push(`${token}/${next}: ${deficient.toFixed(1)} apart under CVD`);
      if (!(lightness >= 0.06)) failures.push(`${token}/${next}: ${lightness.toFixed(3)} apart in lightness`);
    });
  }
  assert.deepEqual(failures, []);
});

test("the dark theme is gruvbox's warm dark, with the edit in its own blue", () => {
  // Spec D9: gruvbox grounds and text (morhetz/gruvbox, MIT/X11).
  const anchors = {
    "--surface": "#1d2021",
    "--surface-subtle": "#282828",
    "--border": "#3c3836",
    "--border-strong": "#7c6f64",
    "--text": "#ebdbb2",
    "--muted": "#a89984",
    "--grey": "#928374",
    "--blue": "#83a8e8",
    "--orange": "#fe8019",
    "--yellow": "#d79921",
    "--group-0": "#fabd2f",
    "--group-1": "#d3869b",
    "--trace-0": "#83a598",
    "--trace-1": "#b8bb26",
  };
  assert.deepEqual(Object.fromEntries(Object.keys(anchors).map((token) => [token, DARK.get(token)])), anchors);
});

test("the exposure strip is more opaque on the dark ground only", () => {
  // The light strip's opacity lives in styles.css and editor_style.py restates it.
  assert.match(appFile("styles.css"), /\.exposure \{[^}]*fill-opacity: 0\.6;/);
  const dark = appFile("styles/dark.css");
  const rule = dark.slice(dark.indexOf(':root[data-theme="dark"] .exposure,'));
  assert.match(rule, /^:root\[data-theme="dark"\] \.exposure,\s*:root\[data-theme="dark"\] \.exposure-density,\s*:root\[data-theme="dark"\] \.legend-swatch \{\s*fill-opacity: 0\.8;\s*\}/);
});
```

- [ ] **Step 2: Run it, expect FAIL.**

```bash
node --test tests/editor_frontend/dark_palette.test.js
```

  Expected on 155832e8: 1 pass, 3 fail.
  - `the dark categorical palettes keep neighbours apart…` lists 26 pairs,
    for example `--group-2/--group-3: 0.020 apart in lightness`,
    `--basis-6/--basis-7: 4.1 apart under CVD` and
    `--trace-4/--trace-5: 12.3 apart`.
  - `the dark theme is gruvbox's warm dark…` shows today's values: `--surface`
    `#15171c`, `--text` `#f2efe6`, `--blue` `#7aa2f7`, and so on.
  - `the exposure strip is more opaque…` fails because master sets 0.4 and has
    no `.legend-swatch` in that rule.
  - The contrast test passes on master. It is the guard the new values must
    also clear, and Step 4 shows it rejecting the board's unadjusted values.

- [ ] **Step 3: Implement.** Replace `src/superglm/editor/app/styles/dark.css`
  with:

```css
/* dark.css: the dark palette, a warm dark. Every colour token in tokens.css
   is restated for a dark ground, keyed on <html data-theme="dark">:
   index.html sets that attribute before first paint and views/theme.js keeps
   it current. It lives apart from tokens.css because
   superglm/plotting/editor_style.py reads that file as the light palette.
   tests/editor_frontend/dark_palette.test.js holds every text pair to 4.5:1
   and keeps neighbours in each categorical palette apart, in lightness too. */

/* Palette: gruvbox by morhetz (github.com/morhetz/gruvbox), MIT/X11. */

:root[data-theme="dark"] {
  color-scheme: dark;
  /* Grounds and text are gruvbox's. The hover ground is its dark0_soft
     rather than dark1, which keeps muted text on it at 4.5:1. */
  --text: #ebdbb2;
  --muted: #a89984;
  --surface: #1d2021;
  --surface-subtle: #282828;
  --surface-hover: #32302f;
  --border: #3c3836;
  --border-strong: #7c6f64;
  --hairline: rgba(235, 219, 178, 0.1);
  --grid: rgba(168, 153, 132, 0.14);
  /* The current edit is this file's own blue, clearer than gruvbox's. */
  --blue: #83a8e8;
  --blue-soft: #283244;
  --blue-band: rgba(131, 168, 232, 0.12);
  --orange: #fe8019;
  --red: #fb4934;
  --yellow: #d79921;
  --yellow-border: #d79921;
  /* Red set as text is a step lighter than gruvbox's red, which falls below
     4.5:1 on the inspector's ground. */
  --danger: #fb7a6b;
  --danger-surface: #3c1f1e;
  --danger-text: #ffb3a7;
  --better: #8ec07c;
  --worse: #fb7a6b;
  --focus: #83a8e8;
  --shadow: rgba(0, 0, 0, 0.55);
  --ink: #000000;
  --primary-hover: #d5c4a1;
  --popover-muted: #504945;
  --grey: #928374;
  --zero: rgba(235, 219, 178, 0.3);
  --ci: rgba(131, 168, 232, 0.18);
  --ci-whisker: rgba(131, 168, 232, 0.6);
  --basis-contribution: rgba(168, 153, 132, 0.4);
  --build-end: #83a8e8;
  --trace-ink: #a89984;
  --sig-strong-bg: #2b3a1f;
  --sig-strong-fg: #b8bb26;
  --sig-medium-bg: #2a3622;
  --sig-medium-fg: #a9c47f;
  --sig-standard-bg: #33361c;
  --sig-standard-fg: #d5c46a;
  --sig-weak-bg: #3d3115;
  --sig-weak-fg: #fabd2f;
  --sig-none-bg: #3c1f1e;
  --sig-none-fg: #fb7a6b;
  /* The categorical palettes draw on gruvbox's bright and neutral hues,
     ordered so neighbours differ in lightness as well as hue. */
  --group-0: #fabd2f;
  --group-1: #d3869b;
  --group-2: #689d6a;
  --group-3: #b8bb26;
  --group-4: #fb4934;
  --group-5: #458588;
  --basis-0: #b8bb26;
  --basis-1: #fb4934;
  --basis-2: #8ec07c;
  --basis-3: #d65d0e;
  --basis-4: #83a598;
  --basis-5: #fabd2f;
  --basis-6: #98971a;
  --basis-7: #458588;
  --basis-8: #fe8019;
  --basis-9: #b16286;
  --basis-10: #d79921;
  --basis-11: #689d6a;
  --trace-0: #83a598;
  --trace-1: #b8bb26;
  --trace-2: #b16286;
  --trace-3: #fe8019;
  --trace-4: #689d6a;
  --trace-5: #fabd2f;
  --trace-6: #fb4934;
  --trace-7: #458588;
  --trace-8: #8ec07c;
  --trace-9: #d65d0e;
}

/* A translucent yellow over near-black composes to a dim gold, so the
   exposure strip, and its legend swatch, are more opaque here than on the
   light ground (0.6 in styles.css). */
:root[data-theme="dark"] .exposure,
:root[data-theme="dark"] .exposure-density,
:root[data-theme="dark"] .legend-swatch {
  fill-opacity: 0.8;
}
```

  Each token and where its value comes from:

  | Token(s) | Value | Source |
  |---|---|---|
  | `--text`, `--muted`, `--trace-ink` | #ebdbb2, #a89984, #a89984 | gruvbox light1, light4 (board) |
  | `--surface`, `--surface-subtle`, `--surface-hover` | #1d2021, #282828, #32302f | dark0_hard, dark0 (board); dark0_soft (re-stepped, amendment 6) |
  | `--border`, `--border-strong`, `--grey` | #3c3836, #7c6f64, #928374 | dark1, dark4, gray (board) |
  | `--hairline`, `--grid`, `--zero` | text 10 %, muted 14 %, text 30 % | board; `--zero` keeps today's dark alpha |
  | `--blue`, `--focus`, `--build-end` | #83a8e8 | the board's edit blue |
  | `--blue-soft`, `--blue-band` | #283244, blue 12 % | re-stepped (amendment 6); board |
  | `--ci`, `--ci-whisker` | blue 18 %, blue 60 % | the board's dark CI band, a step above light's 13 % |
  | `--orange`, `--red`, `--yellow`, `--yellow-border` | #fe8019, #fb4934, #d79921, #d79921 | bright_orange (selection), bright_red, neutral_yellow (exposure) (board) |
  | `--danger`, `--worse` | #fb7a6b | re-stepped (amendment 6) |
  | `--danger-surface`, `--danger-text` | #3c1f1e, #ffb3a7 | the board's sig-none ground; today's dark text, 8.7:1 on it |
  | `--better` | #8ec07c | bright_aqua (board) |
  | `--shadow`, `--ink` | black 55 %, #000000 | the board's scrim; unchanged |
  | `--primary-hover` | #d5c4a1 | light2, a step down from the `--text` fill of `button.primary` |
  | `--popover-muted` | #504945 | dark2, 6.4:1 on the cream popover |
  | `--basis-contribution` | muted 40 % | the board's dashed basis line |
  | `--sig-*` | the board's five pairs | 5.8–7.5:1 |
  | `--group-0…5` | #fabd2f #d3869b #689d6a #b8bb26 #fb4934 #458588 | 0–1 from the board; the rest searched over gruvbox's bright and neutral hues without orange (selection) or neutral yellow (exposure) |
  | `--basis-0…11` | #b8bb26 #fb4934 #8ec07c #d65d0e #83a598 #fabd2f #98971a #458588 #fe8019 #b16286 #d79921 #689d6a | basis-0 is green (`--green` aliases it); exhaustive search over gruvbox |
  | `--trace-0…9` | #83a598 #b8bb26 #b16286 #fe8019 #689d6a #fabd2f #fb4934 #458588 #8ec07c #d65d0e | folds 1–5 from the board, 3 and 5 re-stepped; 5–9 searched including the wrap to 0 |

  Weakest neighbouring pair in each palette, the wrap included (normal OKLab ×100, CVD,
  lightness, worst mark contrast on `--surface`):
  - group: 15.4, 7.3, 0.060, 3.88:1;
  - basis: 15.4, 8.3, 0.071, 3.87:1;
  - trace: 15.9, 7.3, 0.071, 3.87:1.

  The 6–8 CVD band is covered by each palette's second channel: the group
  label under the axis, the Build's one highlighted basis, and the legends.
  The exposure alpha lives in `styles.css` (0.6) and stays there. The dark
  override drops today's bright strip edge, which the more opaque fill makes
  unnecessary; the board has none.

- [ ] **Step 4: Run tests, expect PASS.**

```bash
node --test tests/editor_frontend/dark_palette.test.js   # 4 pass
npm ci                                                   # once, if node_modules/ is missing (CI does the same)
npm run check:frontend
./.venv/bin/python -m pytest tests/editor/test_editor_theme_browser.py -m browser --run-browser -q
./.venv/bin/python -m pytest tests/test_lss_editor_style.py -q
```

  - The existing browser test still passes: it checks a dark ground darker
    than 48 per channel, and #1d2021 is (29, 32, 33).
  - Mutation check: put the board's five values back (amendment 6) and rerun
    the node test. It fails with exactly these, then restore:
    - `dark --muted on --surface-hover: 4.17:1`;
    - `dark --muted on --blue-soft: 4.11:1`;
    - `dark --danger on --surface-subtle: 4.29:1`;
    - `dark --worse on --surface-subtle: 4.29:1`;
    - `--trace-1/--trace-2: 0.060 apart in lightness` (0.0598);
    - `--trace-2/--trace-3: 14.2 apart`;
    - `--trace-2/--trace-3: 0.026 apart in lightness`;
    - `--trace-3/--trace-4: 0.024 apart in lightness`.

- [ ] **Step 5: Commit.**

```bash
git add src/superglm/editor/app/styles/dark.css tests/editor_frontend/dark_palette.test.js
git commit -m "Editor: warm gruvbox dark palette, text held to 4.5:1 and neighbouring hues apart

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task I2: The DAY/NIGHT switch and "Follow the browser" in Settings

**Files:**
- Modify (whole file): `src/superglm/editor/app/views/theme.js`, lines 1–128
  at 155832e8.
- Modify `src/superglm/editor/app/index.html`; line numbers are at 155832e8,
  so match on the text:
  - lines 8–9, the first-paint comment;
  - line 20, the Google Fonts link;
  - lines 111–124, the `#themeAction` button.
- Modify `src/superglm/editor/app/main.js`: line 61 (the import) and lines
  220–224 (the mount).
- Modify `src/superglm/editor/app/styles/shell.css`: insert after the
  `.app-tabs, .app-actions` rule, lines 76–81.
- Modify `src/superglm/editor/app/styles/tokens.css`: insert after line 91,
  `--trace-best`.
- Modify `src/superglm/editor/app/styles/dark.css`: two lines before the
  closing brace of the token rule.
- Modify `src/superglm/editor/app/views/help_content.js`, lines 205–206: the
  Theme help.
- Test (whole file): `tests/editor_frontend/theme.test.js`, lines 1–182.
- Test (whole file): `tests/editor/test_editor_theme_browser.py`, lines 1–53.

**Interfaces:**
- Consumes: from F1, `loadSettings()`, `saveSettings(patch)`,
  `onSettingsChange(listener) -> unsubscribe`, `#settingsTab`, and the Follow
  control (amendment 2). From I1, the dark tokens.
- Produces:
  - `mountThemeSwitch({button, root, media, settings, storage?}) -> {destroy}`;
  - `describeThemeSwitch(choice, prefersDark) -> {checked: boolean, title: string, body: string}`;
  - `renderThemeSwitch(button, choice, prefersDark)`;
  - `#themeSwitch`;
  - the CSS tokens `--switch-track` and `--switch-dots`.

- [ ] **Step 1: Write the failing tests.** Replace
  `tests/editor_frontend/theme.test.js` with:

```js
// @ts-nocheck

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  THEME_STORAGE_KEY,
  describeThemeSwitch,
  mountThemeSwitch,
  readThemeChoice,
  resolveTheme,
  storeThemeChoice,
} from "../../src/superglm/editor/app/views/theme.js";

class FakeSwitch {
  constructor() {
    this.dataset = {};
    this.attributes = new Map();
    this.listeners = new Map();
  }

  setAttribute(name, value) {
    this.attributes.set(name, value);
  }

  addEventListener(type, listener) {
    this.listeners.set(type, listener);
  }

  removeEventListener(type) {
    this.listeners.delete(type);
  }

  click() {
    this.listeners.get("click")?.();
  }

  get checked() {
    return this.attributes.get("aria-checked") === "true";
  }
}

class FakeMedia {
  constructor(matches) {
    this.matches = matches;
    this.listeners = new Map();
  }

  addEventListener(type, listener) {
    this.listeners.set(type, listener);
  }

  removeEventListener(type) {
    this.listeners.delete(type);
  }

  set(matches) {
    this.matches = matches;
    this.listeners.get("change")?.();
  }
}

function memoryStorage() {
  const store = new Map();
  return {
    store,
    getItem: (key) => (store.has(key) ? store.get(key) : null),
    setItem: (key, value) => store.set(key, value),
    removeItem: (key) => store.delete(key),
  };
}

const BLOCKED = {
  getItem() { throw new Error("storage disabled"); },
  setItem() { throw new Error("storage disabled"); },
  removeItem() { throw new Error("storage disabled"); },
};

// The settings store as views/settings.js keeps it: the current settings in
// memory, every listener told synchronously after a save.
function memorySettings(followBrowserTheme) {
  let current = { followBrowserTheme };
  const listeners = new Set();
  return {
    saves: [],
    load: () => ({ ...current }),
    save(patch) {
      current = { ...current, ...patch };
      this.saves.push(patch);
      for (const listener of [...listeners]) listener({ ...current });
      return { ...current };
    },
    subscribe(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    listenerCount: () => listeners.size,
  };
}

function mount({ stored = null, follow = true, prefersDark = false, storage = memoryStorage(), settings } = {}) {
  if (stored !== null) storage.setItem(THEME_STORAGE_KEY, stored);
  const button = new FakeSwitch();
  const root = { dataset: {} };
  const media = new FakeMedia(prefersDark);
  const store = settings ?? memorySettings(follow);
  const control = mountThemeSwitch({ button, root, media, settings: store, storage });
  return { button, root, media, settings: store, storage, control };
}

function appFile(path) {
  return readFileSync(new URL(`../../src/superglm/editor/app/${path}`, import.meta.url), "utf8");
}

test("a stored choice is explicit, and no stored choice or blocked storage follows the browser", () => {
  assert.deepEqual(
    [resolveTheme("auto", false), resolveTheme("auto", true), resolveTheme("light", true), resolveTheme("dark", false)],
    ["light", "dark", "light", "dark"],
  );
  const storage = memoryStorage();
  assert.equal(readThemeChoice(storage), "auto");
  storeThemeChoice("dark", storage);
  assert.deepEqual([...storage.store.entries()], [["superglm.editor.theme", "dark"]]);
  assert.equal(readThemeChoice(storage), "dark");
  storage.store.set("superglm.editor.theme", "sepia");
  assert.equal(readThemeChoice(storage), "auto");
  storeThemeChoice("auto", storage);
  assert.equal(storage.store.size, 0);
  assert.equal(readThemeChoice(BLOCKED), "auto");
  assert.doesNotThrow(() => storeThemeChoice("dark", BLOCKED));
  assert.doesNotThrow(() => storeThemeChoice("auto", BLOCKED));
});

test("the switch is on for Night and says what a click does", () => {
  assert.deepEqual(describeThemeSwitch("auto", false), {
    checked: false,
    title: "Theme: Day",
    body: "Follows the browser's setting. Click for Night; the theme then stays as you set it.",
  });
  assert.deepEqual(describeThemeSwitch("auto", true), {
    checked: true,
    title: "Theme: Night",
    body: "Follows the browser's setting. Click for Day; the theme then stays as you set it.",
  });
  assert.deepEqual(describeThemeSwitch("dark", false), {
    checked: true,
    title: "Theme: Night",
    body: "Click for Day. Settings can follow the browser's setting again.",
  });
  assert.deepEqual(describeThemeSwitch("light", true), {
    checked: false,
    title: "Theme: Day",
    body: "Click for Night. Settings can follow the browser's setting again.",
  });
});

test("a flip stores the theme and stops following the browser", () => {
  const { button, root, media, settings, storage } = mount();
  assert.deepEqual([root.dataset.theme, button.checked, button.dataset.popoverTitle], ["light", false, "Theme: Day"]);
  assert.deepEqual(settings.saves, []);

  button.click();
  assert.deepEqual([root.dataset.theme, button.checked, storage.getItem(THEME_STORAGE_KEY)], ["dark", true, "dark"]);
  assert.deepEqual(settings.saves, [{ followBrowserTheme: false }]);
  // A flipped switch stays put when the browser changes.
  media.set(true);
  media.set(false);
  assert.equal(root.dataset.theme, "dark");

  button.click();
  assert.deepEqual([root.dataset.theme, storage.getItem(THEME_STORAGE_KEY)], ["light", "light"]);
  assert.deepEqual(settings.saves, [{ followBrowserTheme: false }, { followBrowserTheme: false }]);
});

test("while following the browser the switch moves with it", () => {
  const { button, root, media } = mount({ prefersDark: false });
  media.set(true);
  assert.deepEqual([root.dataset.theme, button.checked], ["dark", true]);
  assert.equal(button.dataset.popoverBody, "Follows the browser's setting. Click for Day; the theme then stays as you set it.");
});

test("Settings hands the theme back to the browser, and turning that off keeps the theme on screen", () => {
  const { button, root, media, settings, storage } = mount({ prefersDark: false });
  button.click();
  settings.save({ followBrowserTheme: true });
  assert.deepEqual([root.dataset.theme, storage.getItem(THEME_STORAGE_KEY)], ["light", null]);
  media.set(true);
  assert.equal(root.dataset.theme, "dark");

  settings.save({ followBrowserTheme: false });
  assert.equal(storage.getItem(THEME_STORAGE_KEY), "dark");
  media.set(false);
  assert.equal(root.dataset.theme, "dark");
});

test("the theme key decides, and the setting is brought into line with it on mount", () => {
  // A choice made before the setting existed: still explicit.
  const legacy = mount({ stored: "dark", follow: true, prefersDark: false });
  assert.equal(legacy.root.dataset.theme, "dark");
  assert.deepEqual(legacy.settings.saves, [{ followBrowserTheme: false }]);
  // No stored choice: the browser's theme, whatever the setting said.
  const unset = mount({ stored: null, follow: false, prefersDark: true });
  assert.equal(unset.root.dataset.theme, "dark");
  assert.deepEqual(unset.settings.saves, [{ followBrowserTheme: true }]);
  // In line already: nothing is saved.
  assert.deepEqual(mount({ stored: "light", follow: false }).settings.saves, []);
});

test("with storage blocked the switch follows the browser, flips for the page, and nothing throws", () => {
  const { button, root, media } = mount({ storage: BLOCKED, prefersDark: true });
  assert.equal(root.dataset.theme, "dark");
  assert.doesNotThrow(() => button.click());
  assert.equal(root.dataset.theme, "light");
  media.set(false);
  media.set(true);
  assert.equal(root.dataset.theme, "light");

  // A settings store that keeps nothing echoes its defaults back; the flip holds.
  const listeners = new Set();
  const forgetful = {
    load: () => ({ followBrowserTheme: true }),
    save() {
      for (const listener of listeners) listener({ followBrowserTheme: true });
    },
    subscribe(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
  const page = mount({ storage: BLOCKED, prefersDark: false, settings: forgetful });
  page.button.click();
  assert.equal(page.root.dataset.theme, "dark");
});

test("destroy removes every listener", () => {
  const { button, media, settings, control } = mount();
  control.destroy();
  assert.equal(button.listeners.size + media.listeners.size + settings.listenerCount(), 0);
});

test("index.html paints the stored theme first and carries the switch", () => {
  const html = appFile("index.html");
  assert.ok(html.includes(`localStorage.getItem("${THEME_STORAGE_KEY}")`));
  assert.ok(!html.includes('id="themeAction"'));
  const start = html.indexOf('<button id="themeSwitch"');
  assert.notEqual(start, -1);
  const markup = html.slice(start, html.indexOf("</button>", start));
  for (const part of ['role="switch"', 'aria-checked="false"', 'aria-label="Dark theme"', ">DAY<", ">NIGHT<", 'data-icon="sun"', 'data-icon="moon"']) {
    assert.ok(markup.includes(part), part);
  }
  assert.match(html, /fonts\.googleapis\.com\/css2\?family=Bangers&family=Source\+Sans\+3/);
});

test("the dark palette restates every colour token of the light one and no other", () => {
  const names = (css) => new Set([...css.matchAll(/^\s+(--[\w-]+):/gm)].map((match) => match[1]));
  const light = names(appFile("styles/tokens.css"));
  const dark = names(appFile("styles/dark.css"));
  const layout = /^--(font|radius|space|control|feature|inspector|tooltip)/;
  // An alias of another token follows it in both themes.
  const aliases = new Set(["--green", "--trace-best"]);
  const missing = [...light].filter((name) => !dark.has(name) && !layout.test(name) && !aliases.has(name));
  assert.deepEqual(missing, []);
  assert.deepEqual([...dark].filter((name) => !light.has(name)), []);
});
```

  Replace `tests/editor/test_editor_theme_browser.py` with:

```python
from __future__ import annotations

import json

import pytest

pytest.importorskip("playwright.sync_api")
pytestmark = pytest.mark.browser

THEME = "() => document.documentElement.dataset.theme"
GROUND = "() => getComputedStyle(document.body).backgroundColor"
CURVE = "() => getComputedStyle(document.querySelector('#chart path.edited')).stroke"
STORED = "key => localStorage.getItem(key)"
# styles/dark.css: gruvbox's dark0_hard ground and the editor's own edit blue.
DARK_GROUND = "rgb(29, 32, 33)"
DARK_EDIT = "rgb(131, 168, 232)"
FOLLOW = "Follow the browser's light or dark setting"
BLOCK_STORAGE = """
Object.defineProperty(window, 'localStorage', {
  configurable: true,
  get() { throw new DOMException('The operation is insecure.', 'SecurityError'); },
});
"""


def _channels(colour: str) -> tuple[int, ...]:
    return tuple(int(part) for part in colour.strip("rgba()").split(",")[:3])


def _await_theme(page, theme: str) -> None:
    """The browser delivers a colour-scheme change to the page asynchronously."""
    page.wait_for_function("theme => document.documentElement.dataset.theme === theme", arg=theme)


def test_switch_flips_to_the_warm_dark_and_survives_a_reload(open_editor_page):
    with open_editor_page() as (page, _session):
        page.emulate_media(color_scheme="light")
        _await_theme(page, "light")
        light_ground, light_curve = page.evaluate(GROUND), page.evaluate(CURVE)
        assert min(_channels(light_ground)) > 240
        switch = page.get_by_role("switch", name="Dark theme")
        assert not switch.is_checked()

        switch.click()
        assert page.evaluate(THEME) == "dark"
        assert switch.is_checked()
        assert page.evaluate(GROUND) == DARK_GROUND
        assert page.evaluate(CURVE) == DARK_EDIT
        assert page.evaluate(STORED, "superglm.editor.theme") == "dark"
        settings = json.loads(page.evaluate(STORED, "superglm.editor.settings"))
        assert settings["followBrowserTheme"] is False

        page.reload(wait_until="domcontentloaded")
        # The first-paint script restores the choice before the app loads.
        assert page.evaluate(THEME) == "dark"
        page.locator("#chart path.edited").first.wait_for()
        assert page.get_by_role("switch", name="Dark theme").is_checked()
        assert page.evaluate(GROUND) == DARK_GROUND

        # The browser still says light; the flipped switch wins until flipped back.
        page.get_by_role("switch", name="Dark theme").click()
        assert page.evaluate(THEME) == "light"
        assert page.evaluate(GROUND) == light_ground
        assert page.evaluate(CURVE) == light_curve


def test_following_the_browser_again_hands_it_the_switch(open_editor_page):
    with open_editor_page() as (page, _session):
        page.emulate_media(color_scheme="light")
        _await_theme(page, "light")
        switch = page.get_by_role("switch", name="Dark theme")
        switch.click()
        page.locator("#settingsTab").click()
        follow = page.get_by_label(FOLLOW)
        assert not follow.is_checked()

        follow.click()
        _await_theme(page, "light")
        assert follow.is_checked()
        assert not switch.is_checked()
        assert page.evaluate(STORED, "superglm.editor.theme") is None

        page.emulate_media(color_scheme="dark")
        _await_theme(page, "dark")
        assert switch.is_checked()
        assert page.evaluate(GROUND) == DARK_GROUND


def test_blocked_storage_follows_the_browser_and_still_flips(open_editor_page):
    with open_editor_page() as (page, _session):
        errors: list[str] = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.emulate_media(color_scheme="dark")
        page.add_init_script(BLOCK_STORAGE)
        page.reload(wait_until="domcontentloaded")
        page.locator("#chart path.edited").first.wait_for()
        assert page.evaluate("() => { try { localStorage; return false; } catch { return true; } }")
        assert page.evaluate(THEME) == "dark"
        switch = page.get_by_role("switch", name="Dark theme")
        assert switch.is_checked()

        switch.click()
        assert page.evaluate(THEME) == "light"
        assert not switch.is_checked()
        assert errors == []
```

- [ ] **Step 2: Run them, expect FAIL.**

```bash
node --test tests/editor_frontend/theme.test.js
./.venv/bin/python -m pytest tests/editor/test_editor_theme_browser.py -m browser --run-browser -q
```

  Expected on 155832e8 (and after I1):
  - node fails to load the file:
    `SyntaxError: The requested module '../../src/superglm/editor/app/views/theme.js' does not provide an export named 'describeThemeSwitch'`;
  - all three browser tests fail with
    `TimeoutError: Locator.is_checked: Timeout 30000ms exceeded` (or
    `Locator.click`), because there is no switch named "Dark theme".

- [ ] **Step 3: Implement.**

  3a. Replace `src/superglm/editor/app/views/theme.js` with:

```js
// @ts-check
// The theme switch in the app bar: a comic pill reading DAY or NIGHT. Until
// it is flipped the editor follows the browser's colour-scheme setting, which
// inside a notebook is not always the notebook's own theme; a flip is an
// explicit choice that outlives the page through localStorage when it can,
// and Settings can hand the theme back to the browser. The resolved theme is
// written to <html data-theme>, which styles/dark.css keys the dark palette on
// and the switch's rest position follows; index.html writes the same attribute
// before first paint.
//
// Which store decides: the theme key. A stored "light" or "dark" is an
// explicit choice and no stored value means follow the browser, so the
// first-paint script reads that key alone. The followBrowserTheme setting
// mirrors it, and this module keeps the two equal: it reconciles them on
// mount, a flip clears the setting, and turning the setting on or off in
// Settings removes the key or stores the theme on screen.

/** @typedef {"auto"|"light"|"dark"} ThemeChoice */
/** @typedef {"light"|"dark"} Theme */
/** @typedef {Pick<Storage, 'getItem'|'setItem'|'removeItem'>} ThemeStorage */
/** @typedef {Pick<MediaQueryList, 'matches'|'addEventListener'|'removeEventListener'>} DarkMedia */
/** @typedef {{followBrowserTheme: boolean}} FollowSetting */
/**
 * The part of views/settings.js the switch uses, passed in by main.js.
 * @typedef {{
 *   load: () => FollowSetting,
 *   save: (patch: FollowSetting) => unknown,
 *   subscribe: (listener: (settings: FollowSetting) => void) => () => void,
 * }} ThemeSettings
 */

export const THEME_STORAGE_KEY = "superglm.editor.theme";

/** @param {unknown} value @returns {value is ThemeChoice} */
export function isThemeChoice(value) {
  return value === "auto" || value === "light" || value === "dark";
}

/**
 * The remembered choice, or Auto with no stored choice or unusable storage
 * (private mode, a blocked origin).
 * @param {Pick<ThemeStorage, 'getItem'>} [storage] defaults to localStorage, whose access may itself throw
 * @returns {ThemeChoice}
 */
export function readThemeChoice(storage) {
  try {
    const stored = (storage ?? localStorage).getItem(THEME_STORAGE_KEY);
    return isThemeChoice(stored) ? stored : "auto";
  } catch {
    return "auto";
  }
}

/**
 * Remember the choice; Auto is the absence of one.
 * @param {ThemeChoice} choice @param {ThemeStorage} [storage]
 */
export function storeThemeChoice(choice, storage) {
  try {
    const store = storage ?? localStorage;
    if (choice === "auto") store.removeItem(THEME_STORAGE_KEY);
    else store.setItem(THEME_STORAGE_KEY, choice);
  } catch {
    // Unusable storage: the choice lasts this page only.
  }
}

/** @param {ThemeChoice} choice @param {boolean} prefersDark @returns {Theme} */
export function resolveTheme(choice, prefersDark) {
  if (choice === "auto") return prefersDark ? "dark" : "light";
  return choice;
}

/**
 * What the switch says of itself: on for Night, its popover, what a click does.
 * @param {ThemeChoice} choice @param {boolean} prefersDark
 * @returns {{checked: boolean, title: string, body: string}}
 */
export function describeThemeSwitch(choice, prefersDark) {
  const dark = resolveTheme(choice, prefersDark) === "dark";
  const next = dark ? "Day" : "Night";
  return {
    checked: dark,
    title: `Theme: ${dark ? "Night" : "Day"}`,
    body: choice === "auto"
      ? `Follows the browser's setting. Click for ${next}; the theme then stays as you set it.`
      : `Click for ${next}. Settings can follow the browser's setting again.`,
  };
}

/**
 * Show the choice on the switch: its state and its popover.
 * @param {HTMLElement} button @param {ThemeChoice} choice @param {boolean} prefersDark
 */
export function renderThemeSwitch(button, choice, prefersDark) {
  const { checked, title, body } = describeThemeSwitch(choice, prefersDark);
  button.dataset.choice = choice;
  button.setAttribute("aria-checked", String(checked));
  button.dataset.popoverTitle = title;
  button.dataset.popoverBody = body;
}

/**
 * Mount the switch: apply the remembered choice, flip it on a click, follow
 * the browser while no choice is stored, and keep the follow setting equal to
 * that.
 * @param {{button:HTMLElement, root:HTMLElement, media:DarkMedia, settings:ThemeSettings, storage?:ThemeStorage}} options
 * @returns {{destroy:()=>void}}
 */
export function mountThemeSwitch({ button, root, media, settings, storage }) {
  let choice = readThemeChoice(storage);
  // Set while this module saves the setting, so its own echo is not taken
  // for a change made in Settings.
  let saving = false;

  function mirror() {
    saving = true;
    try {
      settings.save({ followBrowserTheme: choice === "auto" });
    } finally {
      saving = false;
    }
  }

  function render() {
    root.dataset.theme = resolveTheme(choice, media.matches);
    renderThemeSwitch(button, choice, media.matches);
  }

  function onClick() {
    choice = resolveTheme(choice, media.matches) === "dark" ? "light" : "dark";
    storeThemeChoice(choice, storage);
    mirror();
    render();
  }

  function onBrowserChange() {
    if (choice === "auto") render();
  }

  /** @param {FollowSetting} next */
  function onSettingsChange(next) {
    if (saving || next.followBrowserTheme === (choice === "auto")) return;
    choice = next.followBrowserTheme ? "auto" : resolveTheme(choice, media.matches);
    storeThemeChoice(choice, storage);
    render();
  }

  if (settings.load().followBrowserTheme !== (choice === "auto")) mirror();
  button.addEventListener("click", onClick);
  media.addEventListener("change", onBrowserChange);
  const unsubscribe = settings.subscribe(onSettingsChange);
  render();
  return Object.freeze({
    destroy() {
      button.removeEventListener("click", onClick);
      media.removeEventListener("change", onBrowserChange);
      unsubscribe();
    },
  });
}
```

  3b. `index.html`, first-paint comment. Its logic is unchanged (amendment 3).
  Replace:

```html
  // The theme before first paint: the remembered choice, else the browser's
  // setting. views/theme.js takes over once the app loads.
```

  with:

```html
  // The theme before first paint: the remembered choice, else the browser's
  // setting. views/theme.js takes over once the app loads and keeps the
  // "Follow the browser" setting equal to this key, so the key alone decides.
```

  3c. `index.html`, the fonts link. Bangers is for the switch label only.
  Replace:

```html
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
```

  with:

```html
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bangers&family=Source+Sans+3:wght@400;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
```

  3d. `index.html`: replace the whole `#themeAction` button, from
  `<button id="themeAction"` to its `</button>` (14 lines at 155832e8,
  keeping the `app-actions-gap` span before it), with:

```html
      <button id="themeSwitch" type="button" role="switch" aria-checked="false"
        aria-label="Dark theme" data-popover-title="Theme: Day"
        data-popover-body="Follows the browser's setting. Click for Night; the theme then stays as you set it.">
        <span class="theme-switch-label" data-label="day" aria-hidden="true">DAY</span>
        <span class="theme-switch-label" data-label="night" aria-hidden="true">NIGHT</span>
        <span class="theme-switch-knob" aria-hidden="true">
          <svg class="theme-switch-icon" data-icon="sun" viewBox="0 0 24 24">
            <circle cx="12" cy="12" r="4"></circle>
            <path d="M12 2.5V5M12 19v2.5M2.5 12H5M19 12h2.5M5.3 5.3l1.8 1.8M16.9 16.9l1.8 1.8M5.3 18.7l1.8-1.8M16.9 7.1l1.8-1.8"></path>
          </svg>
          <svg class="theme-switch-icon" data-icon="moon" viewBox="0 0 24 24">
            <path d="M20.5 14.5A8.5 8.5 0 0 1 9.5 3.5a8.5 8.5 0 1 0 11 11z"></path>
          </svg>
        </span>
      </button>
```

  3e. `main.js`. Replace:

```js
import { mountThemeControl } from "./views/theme.js";
```

  with:

```js
import { mountThemeSwitch } from "./views/theme.js";
```

  If `main.js` already imports from `"./views/settings.js"` (Task F1), make
  sure that import names `loadSettings`, `onSettingsChange` and
  `saveSettings`, each once. Otherwise add this line after the theme import:

```js
import { loadSettings, onSettingsChange, saveSettings } from "./views/settings.js";
```

  Then replace:

```js
mountThemeControl({
  button: document.getElementById("themeAction"),
  root: document.documentElement,
  media: window.matchMedia("(prefers-color-scheme: dark)")
});
```

  with:

```js
mountThemeSwitch({
  button: document.getElementById("themeSwitch"),
  root: document.documentElement,
  media: window.matchMedia("(prefers-color-scheme: dark)"),
  settings: { load: loadSettings, save: saveSettings, subscribe: onSettingsChange }
});
```

  3f. `styles/shell.css`. The switch's geometry and colours follow the
  approved generator at 1×:
  - the pill is 76 × 32 inside a 2.5 px ink outline, with a 3 px hard shadow;
  - the halftone is 1.1/1.4 px dots on a 5 px grid;
  - the knob is 28 px and travels 44 px;
  - the labels are Bangers 15 px with 0.06 em tracking, at x = 38 (DAY) and
    x = 8 (NIGHT);
  - the icons are 17 px at stroke 2.2.

  The colours are tokens. The ink is `--text`, the shadow `--ink`, the day
  knob `--surface` and the night knob `--blue`. The sun is `--red` (light
  #d6402b) and the moon `--surface` (dark #1d2021). The track and the dots are
  the two new tokens.

  Insert after the `.app-tabs, .app-actions { … gap: 2px; }` rule and its
  blank line:

```css
/* The theme switch: a comic pill, DAY or NIGHT, in the docs site's display
   face. At rest it follows <html data-theme>, which index.html sets before
   first paint, so the pill is right before views/theme.js runs. The id
   outranks the toolbar's ghost-button rules. */
#themeSwitch {
  position: relative;
  flex: none;
  box-sizing: content-box;
  width: 76px;
  height: 32px;
  margin: 0 6px 0 2px;
  padding: 0;
  border: 2.5px solid var(--text);
  border-radius: 999px;
  background-color: var(--switch-track);
  background-image: radial-gradient(circle, var(--switch-dots) 1.1px, transparent 1.4px);
  background-size: 5px 5px;
  box-shadow: 3px 3px 0 var(--ink);
  color: var(--text);
}

.theme-switch-label {
  position: absolute;
  top: 9.5px;
  font: 15px/1 Bangers, Impact, "Arial Black", sans-serif;
  letter-spacing: 0.06em;
}

.theme-switch-label[data-label="day"] {
  left: 38px;
}

.theme-switch-label[data-label="night"] {
  left: 8px;
  opacity: 0;
}

.theme-switch-knob {
  position: absolute;
  top: 2px;
  left: 2px;
  width: 28px;
  height: 28px;
  border: 2.5px solid var(--text);
  border-radius: 50%;
  background-color: var(--surface);
}

.theme-switch-icon {
  position: absolute;
  inset: 0;
  width: 17px;
  height: 17px;
  margin: auto;
  fill: none;
  stroke-width: 2.2;
  stroke-linecap: round;
  stroke-linejoin: round;
}

.theme-switch-icon[data-icon="sun"] {
  stroke: var(--red);
}

.theme-switch-icon[data-icon="moon"] {
  stroke: var(--surface);
  opacity: 0;
}

:root[data-theme="dark"] .theme-switch-knob {
  translate: 44px 0;
  background-color: var(--blue);
}

:root[data-theme="dark"] .theme-switch-label[data-label="day"],
:root[data-theme="dark"] .theme-switch-icon[data-icon="sun"] {
  opacity: 0;
}

:root[data-theme="dark"] .theme-switch-label[data-label="night"],
:root[data-theme="dark"] .theme-switch-icon[data-icon="moon"] {
  opacity: 1;
}
```

  3g. `styles/tokens.css`, after `  --trace-best: var(--trace-1);`:

```css
  /* The theme switch's track and its halftone dots. */
  --switch-track: #fabd2f;
  --switch-dots: rgba(21, 23, 28, 0.2);
```

  3h. `styles/dark.css`, after `  --trace-9: #d65d0e;` (the last token):

```css
  --switch-track: #32302f;
  --switch-dots: rgba(235, 219, 178, 0.22);
```

  3i. `views/help_content.js`, Theme section. Replace the two items:

```js
      "The theme icon in the application bar cycles through Auto, Light and Dark. Auto follows the browser's light or dark setting, which inside a notebook is not always the notebook's own; a chosen theme wins over it.",
      "The choice is kept through a reload of the page.",
```

  with:

```js
      "The DAY / NIGHT switch in the application bar sets the theme. Until you flip it, the editor follows the browser's light or dark setting, which inside a notebook is not always the notebook's own.",
      "A flipped switch keeps its theme through a reload of the page. Follow the browser's light or dark setting, in Settings, hands the theme back to the browser.",
```

  The tutorial's Theme section, `docs/tutorials/edit-a-model-in-the-browser.md:205`,
  is Z1's.

- [ ] **Step 4: Run tests, expect PASS.**

```bash
npm run check:frontend
./.venv/bin/python -m pytest tests/editor/test_editor_theme_browser.py -m browser --run-browser -q
./.venv/bin/python -m pytest tests/test_editor_browser.py tests/editor -m browser --run-browser -q -n 6
./.venv/bin/python -m pytest tests/test_editor.py::test_widget_serves_editor_app_assets tests/test_lss_editor_style.py -q
./.venv/bin/python -m ruff check tests/editor/test_editor_theme_browser.py
./.venv/bin/python -m ruff format --check tests/editor/test_editor_theme_browser.py
```

  - Node: every frontend test passes, and `tsc` is clean.
  - Browser: 3 theme tests pass, and the whole editor browser suite stays
    green. The switch is 89 px with its margins, and the layout tests at
    900–1180 px still pass.
  - Mutation checks on theme.js, each failing exactly one node test:
    - drop `saving ||` from `onSettingsChange`. `with storage blocked…` fails,
      because the forgetful store's echo undoes the flip;
    - delete the mount-time reconcile line, `if (settings.load()… ) mirror();`.
      `the theme key decides…` fails;
    - delete `mirror();` in `onClick`. `a flip stores the theme…` fails.

- [ ] **Step 5: Commit.**

```bash
git add src/superglm/editor/app/views/theme.js src/superglm/editor/app/index.html \
  src/superglm/editor/app/main.js src/superglm/editor/app/styles/shell.css \
  src/superglm/editor/app/styles/tokens.css src/superglm/editor/app/styles/dark.css \
  src/superglm/editor/app/views/help_content.js tests/editor_frontend/theme.test.js \
  tests/editor/test_editor_theme_browser.py
git commit -m "Editor: DAY/NIGHT theme switch; following the browser moves to Settings

A stored choice from the old cycling icon still wins on first paint.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task I3: The flip animation and the page's cross-fade

**Files:**
- Modify `src/superglm/editor/app/views/theme.js` (I2's version): six edits.
- Modify `src/superglm/editor/app/styles/shell.css`: insert after I2's last
  switch rule.
- Test: `tests/editor_frontend/theme.test.js` (I2's version), seven edits.
- Test: `tests/editor/test_editor_theme_browser.py` (I2's version), seven
  edits.

**Interfaces:**
- Consumes: I2's switch DOM, and `mountThemeSwitch`.
- Produces:
  - `FLIP_CLASS = "is-flipping"`, on `#themeSwitch`;
  - `FADE_CLASS = "theme-fading"`, on `<html>`;
  - the keyframes `theme-knob-to-night|day`, `theme-track-to-night|day`,
    `theme-icon-in|out` and `theme-label-in|out`.

**How the restart works.** Each element's keyframe is named for the theme it
reaches, and the rules key on `<html data-theme>`. A click keeps
`.is-flipping` and flips the theme, which changes every animation name, so the
browser restarts them. That is the class-toggle route the brief allows, so
there is no doubled A/B set. A theme change that is not a click comes from
Settings handing the theme back, and it removes the class first. Otherwise the
last flip would replay under the new names. While the browser leads, there is
never a class to remove: only a flip adds one, and a flip ends the following.

The fill mode is `backwards`. It holds the 0% frame through each delay, so the
incoming icon and label stay hidden until they pop. After the end the rest
styles take over, so nothing holds a final frame. The cross-fade transitions
only the grounds, the app bar and the switch. Measured in headless Chromium,
a transition on every element dropped the frame time from 16.7 ms to a mean
of 34–41 ms during the flip; the scoped fade runs at 17–19 ms. The chart's
marks and the panels' contents change at once.

- [ ] **Step 1: Write the failing tests.** Apply these edits, each exact
  text → replacement.

  1. `tests/editor_frontend/theme.test.js`. Replace:

```js
import {
  THEME_STORAGE_KEY,
```

  with:

```js
import {
  FADE_CLASS,
  FLIP_CLASS,
  THEME_STORAGE_KEY,
```

  2. `tests/editor_frontend/theme.test.js`. Replace:

```js
class FakeSwitch {
  constructor() {
    this.dataset = {};
    this.attributes = new Map();
    this.listeners = new Map();
  }
```

  with:

```js
class FakeClassList {
  constructor() {
    this.names = new Set();
  }

  add(name) {
    this.names.add(name);
  }

  remove(name) {
    this.names.delete(name);
  }

  contains(name) {
    return this.names.has(name);
  }
}

class FakeSwitch {
  constructor() {
    this.dataset = {};
    this.attributes = new Map();
    this.classList = new FakeClassList();
    this.listeners = new Map();
  }
```

  3. `tests/editor_frontend/theme.test.js`. Replace:

```js
  get checked() {
    return this.attributes.get("aria-checked") === "true";
  }
}
```

  with:

```js
  animationEnd(animationName) {
    this.listeners.get("animationend")?.({ animationName });
  }

  get checked() {
    return this.attributes.get("aria-checked") === "true";
  }

  get flipping() {
    return this.classList.contains(FLIP_CLASS);
  }
}
```

  4. `tests/editor_frontend/theme.test.js`. Replace:

```js
  const root = { dataset: {} };
```

  with:

```js
  const root = { dataset: {}, classList: new FakeClassList() };
```

  5. `tests/editor_frontend/theme.test.js`. Replace:

```js
function appFile(path) {
```

  with:

```js
const fading = (root) => root.classList.contains(FADE_CLASS);

function appFile(path) {
```

  6. `tests/editor_frontend/theme.test.js`. Replace:

```js
test("destroy removes every listener", () => {
```

  with:

```js
test("a flip plays the switch and fades the page until the knob lands, and nothing else plays", () => {
  const { button, root, media, settings } = mount();
  assert.deepEqual([button.flipping, fading(root)], [false, false]);

  button.click();
  assert.deepEqual([root.dataset.theme, button.flipping, fading(root)], ["dark", true, true]);
  // The fade ends when the knob lands, not when another keyframe does.
  button.animationEnd("theme-icon-in");
  assert.equal(fading(root), true);
  button.animationEnd("theme-knob-to-night");
  assert.equal(fading(root), false);
  // The next flip keeps the class; its keyframes take the other theme's names.
  button.click();
  assert.deepEqual([root.dataset.theme, button.flipping, fading(root)], ["light", true, true]);

  // Settings hands the theme back without motion, and the browser leads without it.
  settings.save({ followBrowserTheme: true });
  assert.deepEqual([button.flipping, fading(root)], [false, false]);
  media.set(true);
  assert.deepEqual([root.dataset.theme, button.flipping, fading(root)], ["dark", false, false]);
});

test("destroy removes every listener", () => {
```

  7. `tests/editor_frontend/theme.test.js`. Replace:

```js
test("the dark palette restates every colour token
```

  with:

```js
test("the flip's keyframes are named for the theme reached, and reduced motion lands them at once", () => {
  const shell = appFile("styles/shell.css");
  for (const name of [
    "theme-knob-to-night", "theme-knob-to-day", "theme-track-to-night", "theme-track-to-day",
    "theme-icon-in", "theme-icon-out", "theme-label-in", "theme-label-out",
  ]) {
    assert.ok(shell.includes(`@keyframes ${name} {`), name);
  }
  // tokens.css zeroes every animation and transition; shell.css drops the switch's delays.
  assert.match(
    appFile("styles/tokens.css"),
    /@media \(prefers-reduced-motion: reduce\) \{\s*\*, \*::before, \*::after \{[^}]*animation-duration: 0\.001ms !important;[^}]*transition-duration: 0\.001ms !important;/,
  );
  assert.match(shell, /@media \(prefers-reduced-motion: reduce\) \{\s*#themeSwitch,\s*#themeSwitch \* \{\s*animation-delay: 0s !important;/);
});

test("the dark palette restates every colour token
```

  8. `tests/editor/test_editor_theme_browser.py`. Replace:

```python
STORED = "key => localStorage.getItem(key)"
```

  with:

```python
STORED = "key => localStorage.getItem(key)"
KNOB = "() => getComputedStyle(document.querySelector('#themeSwitch .theme-switch-knob')).animationName"
FADING = "() => document.documentElement.classList.contains('theme-fading')"
# The flip's keyframes in play on the switch: [name, duration, delay] in seconds.
SWITCH_ANIMATIONS = """() => [...document.querySelectorAll('#themeSwitch, #themeSwitch *')]
  .map((node) => getComputedStyle(node))
  .filter((style) => style.animationName !== 'none')
  .map((style) => [style.animationName, parseFloat(style.animationDuration), parseFloat(style.animationDelay)])"""
```

  9. `tests/editor/test_editor_theme_browser.py`. Replace:

```python
    page.wait_for_function("theme => document.documentElement.dataset.theme === theme", arg=theme)
```

  with:

```python
    page.wait_for_function("theme => document.documentElement.dataset.theme === theme", arg=theme)


def _await_landing(page) -> None:
    """The grounds cross-fade until the knob lands and theme.js ends the fade."""
    page.wait_for_function("() => !document.documentElement.classList.contains('theme-fading')")
```

  10. `tests/editor/test_editor_theme_browser.py`. Replace:

```python
        assert not switch.is_checked()

        switch.click()
        assert page.evaluate(THEME) == "dark"
        assert switch.is_checked()
        assert page.evaluate(GROUND) == DARK_GROUND
```

  with:

```python
        assert not switch.is_checked()
        # Nothing plays on first paint.
        assert page.evaluate(SWITCH_ANIMATIONS) == []

        switch.click()
        assert page.evaluate(THEME) == "dark"
        assert switch.is_checked()
        assert page.evaluate(KNOB) == "theme-knob-to-night"
        assert page.evaluate(FADING)
        _await_landing(page)
        assert page.evaluate(GROUND) == DARK_GROUND
```

  11. `tests/editor/test_editor_theme_browser.py`. Replace:

```python
        assert page.get_by_role("switch", name="Dark theme").is_checked()
        assert page.evaluate(GROUND) == DARK_GROUND

        # The browser still says light; the flipped switch wins until flipped back.
        page.get_by_role("switch", name="Dark theme").click()
        assert page.evaluate(THEME) == "light"
        assert page.evaluate(GROUND) == light_ground
        assert page.evaluate(CURVE) == light_curve
```

  with:

```python
        assert page.get_by_role("switch", name="Dark theme").is_checked()
        # A restored switch does not play.
        assert page.evaluate(SWITCH_ANIMATIONS) == []
        assert page.evaluate(GROUND) == DARK_GROUND

        # The browser still says light; the flipped switch wins until flipped back.
        page.get_by_role("switch", name="Dark theme").click()
        assert page.evaluate(THEME) == "light"
        assert page.evaluate(KNOB) == "theme-knob-to-day"
        _await_landing(page)
        assert page.evaluate(GROUND) == light_ground
        assert page.evaluate(CURVE) == light_curve
        # Each flip restarts the keyframes under the theme it reaches.
        page.get_by_role("switch", name="Dark theme").click()
        assert page.evaluate(KNOB) == "theme-knob-to-night"
```

  12. `tests/editor/test_editor_theme_browser.py`. Replace:

```python
        switch.click()
        page.locator("#settingsTab").click()
```

  with:

```python
        switch.click()
        _await_landing(page)
        page.locator("#settingsTab").click()
```

  13. `tests/editor/test_editor_theme_browser.py`. Replace:

```python
        assert not switch.is_checked()
        assert page.evaluate(STORED, "superglm.editor.theme") is None

        page.emulate_media(color_scheme="dark")
        _await_theme(page, "dark")
        assert switch.is_checked()
        assert page.evaluate(GROUND) == DARK_GROUND
```

  with:

```python
        assert not switch.is_checked()
        assert page.evaluate(STORED, "superglm.editor.theme") is None
        # Handed back, the switch moves without playing.
        assert page.evaluate(SWITCH_ANIMATIONS) == []

        page.emulate_media(color_scheme="dark")
        _await_theme(page, "dark")
        assert switch.is_checked()
        assert page.evaluate(SWITCH_ANIMATIONS) == []
        assert not page.evaluate(FADING)
        assert page.evaluate(GROUND) == DARK_GROUND


def test_reduced_motion_lands_the_flip_at_once(open_editor_page):
    with open_editor_page() as (page, _session):
        page.emulate_media(color_scheme="light", reduced_motion="reduce")
        _await_theme(page, "light")
        page.get_by_role("switch", name="Dark theme").click()
        assert page.evaluate(THEME) == "dark"
        played = page.evaluate(SWITCH_ANIMATIONS)
        # The flip still selects its keyframes; none of them takes any time.
        assert "theme-knob-to-night" in [name for name, _duration, _delay in played]
        assert all(duration < 0.001 and delay == 0 for _name, duration, delay in played)
        _await_landing(page)
        assert page.evaluate(GROUND) == DARK_GROUND
```

  14. `tests/editor/test_editor_theme_browser.py`. Replace:

```python
        assert not switch.is_checked()
        assert errors == []
```

  with:

```python
        assert not switch.is_checked()
        _await_landing(page)
        assert errors == []
```


- [ ] **Step 2: Run them, expect FAIL.**

```bash
node --test tests/editor_frontend/theme.test.js
./.venv/bin/python -m pytest tests/editor/test_editor_theme_browser.py -m browser --run-browser -q
```

  Expected after I2:
  - node:
    `SyntaxError: … does not provide an export named 'FADE_CLASS'`;
  - browser: 2 failed and 2 passed. `test_switch_flips…` fails with
    `assert 'none' == 'theme-knob-to-night'`, and
    `test_reduced_motion_lands_the_flip_at_once` fails with
    `assert 'theme-knob-to-night' in []`.

  On 155832e8 all four browser tests time out, as in I2.

- [ ] **Step 3: Implement.** Apply these edits.

  1. `src/superglm/editor/app/views/theme.js`. Replace:

```js
// Settings removes the key or stores the theme on screen.

/** @typedef
```

  with:

```js
// Settings removes the key or stores the theme on screen.
//
// A flip plays the switch's keyframes and cross-fades the page's grounds
// (styles/shell.css). A change from the browser or from Settings lands
// without motion.

/** @typedef
```

  2. `src/superglm/editor/app/views/theme.js`. Replace:

```js
export const THEME_STORAGE_KEY = "superglm.editor.theme";
```

  with:

```js
export const THEME_STORAGE_KEY = "superglm.editor.theme";
/** On the switch while a flip's keyframes may play. */
export const FLIP_CLASS = "is-flipping";
/** On <html> while the page's grounds cross-fade after a flip. */
export const FADE_CLASS = "theme-fading";
```

  3. `src/superglm/editor/app/views/theme.js`. Replace:

```js
    mirror();
    render();
  }

  function onBrowserChange() {
    if (choice === "auto") render();
  }
```

  with:

```js
    mirror();
    button.classList.add(FLIP_CLASS);
    root.classList.add(FADE_CLASS);
    render();
  }

  // While the browser leads the switch carries no flip: a flip ends the
  // following, and Settings clears the flip when it hands the theme back.
  function onBrowserChange() {
    if (choice === "auto") render();
  }
```

  4. `src/superglm/editor/app/views/theme.js`. Replace:

```js
    storeThemeChoice(choice, storage);
    render();
  }

  if (settings.load()
```

  with:

```js
    storeThemeChoice(choice, storage);
    // Kept, the last flip's keyframes would replay under the new theme's names.
    button.classList.remove(FLIP_CLASS);
    root.classList.remove(FADE_CLASS);
    render();
  }

  /** The knob has landed, and the fade with it. @param {Event} event */
  function onAnimationEnd(event) {
    if (/** @type {AnimationEvent} */ (event).animationName.startsWith("theme-knob-")) {
      root.classList.remove(FADE_CLASS);
    }
  }

  if (settings.load()
```

  5. `src/superglm/editor/app/views/theme.js`. Replace:

```js
  button.addEventListener("click", onClick);
  media.addEventListener
```

  with:

```js
  button.addEventListener("click", onClick);
  button.addEventListener("animationend", onAnimationEnd);
  media.addEventListener
```

  6. `src/superglm/editor/app/views/theme.js`. Replace:

```js
      button.removeEventListener("click", onClick);
      media.removeEventListener
```

  with:

```js
      button.removeEventListener("click", onClick);
      button.removeEventListener("animationend", onAnimationEnd);
      media.removeEventListener
```

  7. `src/superglm/editor/app/styles/shell.css`. Replace:

```css
:root[data-theme="dark"] .theme-switch-label[data-label="night"],
:root[data-theme="dark"] .theme-switch-icon[data-icon="moon"] {
  opacity: 1;
}
```

  with:

```css
:root[data-theme="dark"] .theme-switch-label[data-label="night"],
:root[data-theme="dark"] .theme-switch-icon[data-icon="moon"] {
  opacity: 1;
}

/* The flip, on a click only: theme.js adds .is-flipping. The knob squashes,
   overshoots and settles, the outgoing icon spins out and the incoming one
   in, the label pops, and the halftone drifts four dots, a whole number of
   its period. Each keyframe is named for the theme it reaches, so the next
   click changes the name and the browser restarts it. */
#themeSwitch.is-flipping {
  animation: theme-track-to-day 560ms ease backwards;
}

#themeSwitch.is-flipping .theme-switch-knob {
  animation: theme-knob-to-day 680ms cubic-bezier(0.3, 0.7, 0.2, 1) backwards;
}

:root[data-theme="dark"] #themeSwitch.is-flipping {
  animation-name: theme-track-to-night;
}

:root[data-theme="dark"] #themeSwitch.is-flipping .theme-switch-knob {
  animation-name: theme-knob-to-night;
}

#themeSwitch.is-flipping [data-icon="sun"],
:root[data-theme="dark"] #themeSwitch.is-flipping [data-icon="moon"] {
  animation: theme-icon-in 620ms cubic-bezier(0.3, 0.7, 0.2, 1) 160ms backwards;
}

#themeSwitch.is-flipping [data-icon="moon"],
:root[data-theme="dark"] #themeSwitch.is-flipping [data-icon="sun"] {
  animation: theme-icon-out 420ms cubic-bezier(0.3, 0.7, 0.2, 1) backwards;
}

#themeSwitch.is-flipping [data-label="day"],
:root[data-theme="dark"] #themeSwitch.is-flipping [data-label="night"] {
  animation: theme-label-in 560ms cubic-bezier(0.3, 0.7, 0.2, 1) 260ms backwards;
}

#themeSwitch.is-flipping [data-label="night"],
:root[data-theme="dark"] #themeSwitch.is-flipping [data-label="day"] {
  animation: theme-label-out 240ms cubic-bezier(0.3, 0.7, 0.2, 1) backwards;
}

@keyframes theme-knob-to-night {
  0% { translate: 0 0; scale: 1 1; }
  30% { scale: 1.32 0.78; }
  68% { translate: 51px 0; scale: 0.88 1.12; }
  84% { translate: 41.55px 0; scale: 1.05 0.96; }
  100% { translate: 44px 0; scale: 1 1; }
}

@keyframes theme-knob-to-day {
  0% { translate: 44px 0; scale: 1 1; }
  30% { scale: 1.32 0.78; }
  68% { translate: -7px 0; scale: 0.88 1.12; }
  84% { translate: 2.45px 0; scale: 1.05 0.96; }
  100% { translate: 0 0; scale: 1 1; }
}

@keyframes theme-track-to-night {
  from { background-position: 0 0; }
  to { background-position: 20px 0; }
}

@keyframes theme-track-to-day {
  from { background-position: 0 0; }
  to { background-position: -20px 0; }
}

@keyframes theme-icon-in {
  0% { opacity: 0; transform: rotate(-220deg) scale(0.2); }
  60% { opacity: 1; transform: rotate(18deg) scale(1.15); }
  100% { opacity: 1; transform: rotate(0deg) scale(1); }
}

@keyframes theme-icon-out {
  0% { opacity: 1; transform: rotate(0deg) scale(1); }
  100% { opacity: 0; transform: rotate(220deg) scale(0.2); }
}

@keyframes theme-label-in {
  0% { opacity: 0; transform: scale(0.2) rotate(-14deg); }
  55% { opacity: 1; transform: scale(1.35) rotate(5deg); }
  78% { transform: scale(0.92) rotate(-2deg); }
  100% { opacity: 1; transform: scale(1) rotate(0deg); }
}

@keyframes theme-label-out {
  0% { opacity: 1; transform: scale(1); }
  100% { opacity: 0; transform: scale(0.3) translateY(20%); }
}

/* tokens.css zeroes every duration under reduced motion; the delays go too,
   so a flip lands at once. */
@media (prefers-reduced-motion: reduce) {
  #themeSwitch,
  #themeSwitch * {
    animation-delay: 0s !important;
  }
}

/* The grounds, the app bar and the switch cross-fade with a flip, and only
   then: theme.js puts .theme-fading on <html> for the flip and takes it off
   when the knob lands. The rest changes at once, because a transition on
   every element halves the frame rate while it runs. Reduced motion zeroes
   the fade through tokens.css. */
:root.theme-fading,
:root.theme-fading body,
:root.theme-fading .app-bar,
:root.theme-fading .app-bar *,
:root.theme-fading .feature-list,
:root.theme-fading .editor-workspace,
:root.theme-fading #chart,
:root.theme-fading .inspector {
  transition: background-color 600ms ease, border-color 600ms ease, color 600ms ease,
    box-shadow 600ms ease;
}
```


- [ ] **Step 4: Run tests, expect PASS.**

```bash
npm run check:frontend
./.venv/bin/python -m pytest tests/editor/test_editor_theme_browser.py -m browser --run-browser -q
./.venv/bin/python -m pytest tests/test_editor_browser.py tests/editor -m browser --run-browser -q -n 6
./.venv/bin/python -m ruff check tests/editor/test_editor_theme_browser.py
./.venv/bin/python -m ruff format --check tests/editor/test_editor_theme_browser.py
```

  - Node and `tsc` are clean. The 4 theme browser tests pass, and so does the
    whole editor browser suite.
  - Mutation checks:
    - In shell.css, change `animation-delay: 0s !important` to
      `animation-delay: inherit`. `test_reduced_motion…` fails, with delays of
      0.16 s and 0.26 s left in play.
    - In `onSettingsChange`, delete the two `classList.remove` lines. Then
      `test_following_the_browser_again…` fails, because the last flip replays
      as `theme-track-to-day`, `theme-knob-to-day` and so on. The node test
      `a flip plays the switch…` fails as well.
  - Look at it once. Open the editor, flip both ways, and compare with board
    7b without the starburst. The knob squashes, overshoots past the end and
    settles; the moon or sun spins in after the other spins out; the label
    pops with a tilt. There is no assertion on timing.

- [ ] **Step 5: Commit.**

```bash
git add src/superglm/editor/app/views/theme.js src/superglm/editor/app/styles/shell.css \
  tests/editor_frontend/theme.test.js tests/editor/test_editor_theme_browser.py
git commit -m "Editor: the theme switch's flip animation and the grounds' cross-fade

Keyframes play on a click only; reduced motion lands them at once. The fade is
scoped to the grounds and the app bar: on every element it halved the frame rate.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Risks and follow-ups recorded by this phase

- The board values in amendment 6 change what Max approved. They were
  re-stepped within the same gruvbox hues, but are worth one glance.
- The light palette fails the separability rule I1 applies to dark. Examples:
  `--group-2/--group-3` are 0.019 apart in lightness, and
  `--trace-4/--trace-5` are 12.4 apart (1.9 under CVD). This is a follow-up,
  not this branch, so the test checks only the dark palette.
- The group palette clears the lightness floor by 0.0004 (0.0604). That is
  deterministic, since it is fixed hexes in float64, but any change to group-1
  or group-2 must rerun the test.
- Bangers loads from Google Fonts. Offline, the label falls back to Impact or
  Arial Black, which are wider, so NIGHT may run under the knob.
- F1's exact `onSettingsChange` semantics and Follow label are assumed
  (amendment 2). The browser test's Settings step is the check.
- Local node is v22, while `package.json` asks for ≥ 24. `npm ci` warns but
  works, and every test here ran on v22.


## Section 6: Cross-validation tab (G1–G5)

Spec §4 G, decisions D5, D6, D7 and D12; Review Focus item 5. Five tasks, strictly
in order: each one edits files the next one edits. Every step below was applied in
order to a copy of `origin/master` 155832e8 (with a stand-in for A2's
`session.pending` and `stage_structural`), and at every task boundary `ruff check`,
`ruff format --check`, `tsc -p jsconfig.json`, `node --test`, the focused pytest
files and the Playwright suites were green. Each new test was also run against the
unpatched code; the failure it gives there is quoted in its Step 2.

### Contract amendments

- `CrossValidationResult` gains a third field beside `n_rows` and `data_fingerprint`: `splitter: str | None = None` (the splitter's class name), because the tab's header names the splitter and nothing else records it. All three default to `None`, so results built or pickled before them read `None`.
- The fingerprint is defined exactly: `superglm.model_selection._data_fingerprint(y, sample_weight=None) -> str` is SHA-256 over the row count (8 bytes, little-endian), then `y` and the weights as little-endian float64. `sample_weight=None` hashes as unit weights, which is how every scorer reads it. It stays private (no `superglm.__all__` change).
- `carry.model_with_edited_curves(model, edited, X, y, sample_weight=None, offset=None, *, n_points=200)`: `sample_weight` defaults to `None` and `n_points` is new. The carried curve keeps the edited **shape** exactly and moves by one constant, the exposure-weighted mean of (refit curve − editor's original curve). That constant is the difference between the two fits' centring. Measured: without it, a Final fit whose `most_exposed` reference re-resolved from A to B moved every prediction by 0.36 on the log scale; with it the edit's exposure-weighted change is kept to 64u. An edit that shifted the whole curve keeps its shift. This is the D5 reading; a reviewer should confirm it (see risks).
- `EditorSession.__init__` gains `cv=None, cv_data=None` (`cv_data` already an `EvaluationDataset`; `from_model`/`edit` take the plain tuple). The session exposes `cv` and `cv_check: CVDataCheck(rows, reason, note)`, fixed for its life. A non-`CrossValidationResult` `cv=` raises `TypeError` (people pass the splitter by mistake); `cv_data=` without `cv=` raises `ValueError`.
- New modules: `editor/cv.py` (contents listed in G2 and G4) and `editor/jobs.py` (`JobRunner(*, name, wait_timeout=30.0)` with `start(kind, work, publish) -> job_id`, `status(job_id, *, wait=False)`, `cancel(job_id)`, `latest(kind)` and `close()`; `JobContext.check()/progress(phase, **details)`; `JobCancelledError`, so named for ruff N818).
- Routes: `/job_start` returns the whole status dict, which includes `job_id`. `/job_cancel` returns `{job_id, status, cancel_requested}`. The status dict also carries `error`, `cancel_requested`, `started_at` and `finished_at`. A refusal is HTTP 400 with a fixed sentence. A job whose model changed while it ran ends `failed` with `cv.SUPERSEDED`.
- State payload: top-level `final_fit: {available, stale}`. The report kind `"final"` gains `final_fit: {available, note} | {available, stale, n_rows, splits, carried, pending, summary}`.
- Export format `"final"` (alias `"final_fit"`), default file `superglm_final_model.joblib`, DOM `#exportFinalFit`. `bindExportDialog` takes an optional `finalFitAvailable: () => boolean`.
- Frontend: the cv view is a new `app/views/cv_tab.js` (type-checked, node-tested), plus `app/styles/cv.css`. `reports.js` gets only the dispatch: `renderReport(payload, nodes, cvTab = null)`. `contracts.js` gains `JobKind`, `JobStatus`, `CVFoldRow`, `CVResultPayload`, `CVTermItem`, `CVReportPayload` and `EditorSnapshot.final_fit`. The tab order is Editor · Validation · Cross-validation · Final Fit, as on the board.
- D7 applies to Run CV only, as the spec says. Final fit runs while changes are waiting: it refits the last refit's structure and says "N waiting changes are not included", following the Export precedent.
- Fold colours reuse the existing `--trace-0…9` tokens. I1 should restate `--trace-0…4` with the spec's fold colours (#83a598, #b8bb26, #d3869b, #fe8019, #8ec07c) if the gruvbox folds are wanted; no new tokens, so `editor_style.py`'s token tests are untouched.
- Python tests for G2–G5 go in a new `tests/test_editor_cv.py` with its own small fixture, and the browser test in a new `tests/editor/test_editor_cv_browser.py`. `test_editor.py` only gets its route list and asset list extended.

### Before you start

- Depends on A2/A3: the code reads `session.pending` (D7, the waiting notes), and G4's D7 test stages a change with `session.stage_structural("collapse", "region", {"levels": ["B", "C"], "group_label": None})`. Nothing here needs F1.
- Every "Replace … with" below quotes `origin/master` 155832e8 and gives its master line. Where an earlier task changed the same block, apply the same insertion to the current text. The blocks most likely to have moved: `widget._state()` (A4 adds `pending`); the `bindExportDialog({...})` options in `main.js` and the `bindExportDialog` signature (A5/A6 may add the waiting-changes note); the `_export_bytes` joblib branch (A6 attaches `_editor_history`; the `final` branch shares that code, so the final model carries the history too); and `help_content.js` (C1, F1).
- Run Python as `./.venv/bin/python -m pytest …` from the worktree. The frontend checks need `npm ci` once: this worktree has `package-lock.json` but no `node_modules`.
- Measured cost, for the record: Run CV makes exactly one fit per stored fold and Final fit exactly one fit (G4's tests count `SuperGLM.fit` calls). Re-applying edits costs one `term_inference` per edited term and one prediction pass over each fold's training rows; no solver code changes.

---

### Task G1: `cross_validate` records its rows, data fingerprint and splitter

**Files:**
- Modify: `src/superglm/model_selection.py`: imports (line 5), the `CrossValidationResult`
  docstring and fields (lines 55–66), a new `_data_fingerprint` before `# ── Model cloning`
  (line 112), and the result constructor (lines 551–560)
- Test: `tests/test_cross_validate.py` (append a class at the end of the file, after line 1966)

**Interfaces:**
- Consumes: nothing new.
- Produces: `CrossValidationResult.n_rows: int | None`, `.data_fingerprint: str | None`,
  `.splitter: str | None`; `superglm.model_selection._data_fingerprint(y, sample_weight=None) -> str`.

- [ ] **Step 1: Write the failing test.** Append to `tests/test_cross_validate.py`:

```python


# ── Data fingerprint (the editor's Run CV) ───────────────────────


class TestDataFingerprint:
    """A result records the rows its folds index, so a consumer can replay them."""

    def test_result_records_row_count_fingerprint_and_splitter(self, poisson_data, base_model):
        import hashlib

        df, y, sw = poisson_data
        result = cross_validate(base_model, df, y, cv=SimpleKFold(3), sample_weight=sw)

        expected = hashlib.sha256(len(y).to_bytes(8, "little"))
        expected.update(np.asarray(y, dtype="<f8").tobytes())
        expected.update(np.asarray(sw, dtype="<f8").tobytes())
        assert result.n_rows == len(y)
        assert result.splitter == "SimpleKFold"
        assert result.data_fingerprint == expected.hexdigest()

    def test_fingerprint_reads_no_weights_as_unit_weights_and_sees_row_order(
        self, poisson_data, base_model
    ):
        from superglm.model_selection import _data_fingerprint

        df, y, sw = poisson_data
        unweighted = cross_validate(base_model, df, y, cv=SimpleKFold(2))

        assert unweighted.data_fingerprint == _data_fingerprint(y, np.ones(len(y)))
        assert _data_fingerprint(y, sw) != _data_fingerprint(y[::-1], sw[::-1])
        assert _data_fingerprint(y, sw) != _data_fingerprint(y, 2.0 * sw)

    def test_result_pickled_before_the_fields_existed_reads_none(self):
        # Such a pickle restores without the attributes; the dataclass
        # defaults are class attributes, so the fields read as None.
        old = CrossValidationResult.__new__(CrossValidationResult)
        old.__dict__.update(
            fold_scores=pd.DataFrame(),
            mean_scores={},
            pooled_scores={},
            std_scores={},
            fold_indices=None,
            curve_similarity=None,
            oof_predictions=None,
            estimators=None,
        )
        restored = pickle.loads(pickle.dumps(old))

        assert (restored.n_rows, restored.data_fingerprint, restored.splitter) == (None, None, None)
```

- [ ] **Step 2: Run it, expect FAIL.**

```bash
./.venv/bin/python -m pytest tests/test_cross_validate.py -k DataFingerprint -q
```

Expected on 155832e8: `3 failed`. Two fail with `AttributeError: 'CrossValidationResult'
object has no attribute 'n_rows'` and one with `ImportError: cannot import name
'_data_fingerprint' from 'superglm.model_selection'`.

- [ ] **Step 3: Implement.** In `src/superglm/model_selection.py`:

Edit 1 (master line 5). Replace

```python
import logging
import time
```

with

```python
import hashlib
import logging
import time
```

Edit 2 (master line 55). Replace

```python
    estimators : list or None
        Fitted model per fold. ``None`` unless ``return_estimators=True``.
    """
```

with

```python
    estimators : list or None
        Fitted model per fold. ``None`` unless ``return_estimators=True``.
    n_rows : int or None
        Number of rows the folds index: the length of ``y``.
    data_fingerprint : str or None
        SHA-256 of the response and sample weights the folds were scored on,
        as little-endian float64 bytes after the row count, with unit weights
        standing in for ``sample_weight=None``. Equal fingerprints mean the
        same response and weights in the same row order, which is what lets
        a later consumer, such as the editor's Run CV, replay
        ``fold_indices`` on data it holds.
    splitter : str or None
        Class name of the splitter that drew the folds.

    ``n_rows``, ``data_fingerprint`` and ``splitter`` are ``None`` on a
    result made before they were recorded.
    """
```

Edit 3 (master line 65). Replace

```python
    oof_predictions: NDArray | None = None
    estimators: list | None = None
```

with

```python
    oof_predictions: NDArray | None = None
    estimators: list | None = None
    n_rows: int | None = None
    data_fingerprint: str | None = None
    splitter: str | None = None
```

Edit 4 (master line 112): a module function, placed before the cloning helpers. Replace

```python
# ── Model cloning ────────────────────────────────────────────────
```

with

```python
def _data_fingerprint(y, sample_weight=None) -> str:
    """SHA-256 of the row count, then the response and weights as little-endian float64.

    Unit weights stand in for ``sample_weight=None``, which is how every
    scorer reads it, so an unweighted result matches the same rows supplied
    with explicit ones. The row count goes first so the boundary between the
    two arrays is fixed. Feature columns are not hashed: the response and
    weights identify the rows and their order in one pass over two vectors.
    """
    response = np.asarray(y, dtype=np.float64).ravel()
    weights = (
        np.ones(response.size, dtype=np.float64)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=np.float64).ravel()
    )
    digest = hashlib.sha256(response.size.to_bytes(8, "little"))
    digest.update(np.ascontiguousarray(response, dtype="<f8").tobytes())
    digest.update(np.ascontiguousarray(weights, dtype="<f8").tobytes())
    return digest.hexdigest()


# ── Model cloning ────────────────────────────────────────────────
```

Edit 5 (master line 558): in `cross_validate`'s return; `y` and `sample_weight` are already float64 here. Replace

```python
        oof_predictions=oof,
        estimators=estimators_list,
    )
```

with

```python
        oof_predictions=oof,
        estimators=estimators_list,
        n_rows=n,
        data_fingerprint=_data_fingerprint(y, sample_weight),
        splitter=type(cv).__name__,
    )
```

- [ ] **Step 4: Run tests, expect PASS.**

```bash
./.venv/bin/python -m pytest tests/test_cross_validate.py tests/test_documentation_examples.py tests/test_public_api_snapshot.py -q -n 8
./.venv/bin/ruff check src/superglm/model_selection.py tests/test_cross_validate.py
./.venv/bin/ruff format --check src/superglm/model_selection.py tests/test_cross_validate.py
```

Expected: all pass. In `test_cross_validate.py` that is 84 passed with 3 skipped; the three
skipped are the existing plotly tests, which need plotly installed.

- [ ] **Step 5: Commit.**

```bash
git add src/superglm/model_selection.py tests/test_cross_validate.py
git commit -F - <<'EOF'
cross_validate records its row count, data fingerprint and splitter

CrossValidationResult gains n_rows, data_fingerprint (SHA-256 of the row count, y and
the weights as little-endian float64; None reads as unit weights) and splitter, so the
editor can check that data it holds is the data the folds index before replaying them.
All three default to None, so older results and pickles read None.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```

---

### Task G2: Carried hand edits, stored folds, `cv=`/`cv_data=` and the `cv` report

**Files:**
- Create: `src/superglm/editor/carry.py`, `src/superglm/editor/cv.py` (part 1; G3 and G4 extend it)
- Modify: `src/superglm/editor/session.py`: imports (lines 24, 32, 60),
  `__init__` signature and body (lines 134–143), `from_model` (lines 161–202) and `edit()`
  (lines 1604–1619)
- Modify: `src/superglm/editor/reports.py`: import (line 7) and `report_payload` (line 85)
- Modify: `src/superglm/editor/widget.py`: import (line 25), `__init__` (line 128) and
  `_report` (line 549)
- Test: create `tests/test_editor_cv.py`

**Interfaces:**
- Consumes: G1's `CrossValidationResult.n_rows/data_fingerprint/splitter` and `_data_fingerprint`;
  A2's `session.pending`; `term_offset_values` and `native_log_effect_values` (terms.py:171, 236);
  `EditorSession.to_model` (session.py:652); `plotting.comparison._feature_beta`;
  `plotting.curve_similarity._summarize_against_fold_mean`.
- Produces:
  - `carry.model_with_edited_curves(model, edited, X, y, sample_weight=None, offset=None, *, n_points=200) -> model`
  - `cv.StoredFolds(folds, before_fold=None)`, with `.split()` and `.get_n_splits()`
  - `cv.CVDataCheck(rows, reason=None, note=None)`, `cv.CVRun`, `cv.FinalFit`, `cv.CVTabView`
  - `cv.check_cv_data(cv, cv_rows, fallback) -> CVDataCheck`, `cv.run_cv_reason(session) -> str | None`
  - `cv.final_fit_datasets(session)`, `cv.fold_log_curves(model, terms)`, `cv.fold_term_items(terms, fold_curves)`
  - `cv.capture_cv_view(session, *, run, final_fit)`, `cv.cv_tab_payload(view, *, jobs, request_sequence=None)`
  - `cv.cv_report_payload(widget, *, request_sequence=None)`, plus `JOB_KINDS` and the sentence constants
  - `EditorSession.from_model(..., cv=None, cv_data=None)`, `edit(..., train_data, validation_data, test_data, cv, cv_data)`
  - `session.cv`, `session.cv_check`; `widget._cv_run`, `widget._final_fit`
  - `report_payload(widget, "cv")` and `/report {report: "cv"}`

The tab payload (kind `"cv"`) has these keys: `available, report, title, note, model_revision,
request_sequence, header{supplied, n_folds, splitter, n_rows}, pending, run_cv{available,
reason, note}, final_fit{available, reason, note, done, stale, n_rows}, metrics[{name, label,
lower_is_better}], results[{label, origin, model_revision, stale, folds[{fold, n_train, n_test,
fit_time_s, converged, n_iter, effective_df, scores}], mean, std, pooled}], relativities{available,
origin, stale, note, terms[{name, kind, x, levels, weights, folds[{label, values}], fit, edited,
spread, min_correlation}]}, jobs{cv, final_fit}`.

Each `terms` entry is read on the editor's own grid or levels, with levels in the model's order
(`metadata["native_levels"]`, the order before any display reorder). Every curve is
re-centred on its exposure-weighted mean log. The list is ranked by `spread`, the mean over folds of
`rmse_to_mean` on that relativity scale, and also gives `min_correlation`. A linear (`Numeric`)
term is one slope and has no curve, so it is left out.

- [ ] **Step 1: Write the failing test.** Create `tests/test_editor_cv.py`:

```python
"""The Cross-validation tab: carried edits, stored folds, Run CV and Final fit."""

from __future__ import annotations

import dataclasses
import json
import urllib.error
import urllib.request

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold

from superglm import Categorical, Numeric, Spline, SuperGLM, cross_validate
from superglm.editor import EditorSession

_U = np.finfo(np.float64).eps / 2


def _model() -> SuperGLM:
    return SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        features={
            "age": Spline(n_knots=6),
            "power": Numeric(),
            "region": Categorical(base="first"),
        },
    )


@pytest.fixture(scope="module")
def cv_frame():
    """600 rows: train 0-399, validation 400-499, test 500-599.

    Region's rows meet C first; the model orders its levels A, B, C.
    """
    rng = np.random.default_rng(20261003)
    n = 600
    X = pd.DataFrame(
        {
            "age": rng.uniform(18.0, 80.0, n),
            "power": rng.normal(0.0, 1.0, n),
            "region": rng.choice(["C", "A", "B"], n, p=[0.4, 0.35, 0.25]),
        }
    )
    eta = (
        -0.5
        + 0.2 * np.sin(X["age"].to_numpy() / 12.0)
        + 0.1 * X["power"].to_numpy()
        + np.select([X["region"] == "B", X["region"] == "C"], [0.25, -0.15], 0.0)
    )
    y = rng.poisson(np.exp(eta)).astype(np.float64)
    w = rng.uniform(0.5, 1.5, n)
    return X, y, w


@pytest.fixture(scope="module")
def cv_fit(cv_frame):
    """The model on the train rows, and a 3-fold cross_validate() of it there."""
    X, y, w = cv_frame
    train = slice(0, 400)
    model = _model().fit(X.iloc[train], y[train], sample_weight=w[train])
    supplied = cross_validate(
        _model(),
        X.iloc[train],
        y[train],
        cv=KFold(3, shuffle=True, random_state=0),
        sample_weight=w[train],
        scoring=("deviance", "gini", "nll"),
        return_estimators=True,
    )
    return model, supplied


def _splits(cv_frame):
    X, y, w = cv_frame
    return {
        "train_data": (X.iloc[:400], y[:400], w[:400]),
        "validation_data": (X.iloc[400:500], y[400:500], w[400:500]),
        "test_data": (X.iloc[500:], y[500:], w[500:]),
    }


def _post_json(url: str, payload: dict):
    from superglm.editor.widget import _LIVE_WIDGETS

    origin = url.rsplit("/", 1)[0]
    token = next(widget._token for widget in _LIVE_WIDGETS if widget.url == origin)
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json", "X-SuperGLM-Editor-Token": token},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _post_error(url: str, payload: dict) -> tuple[int, dict]:
    with pytest.raises(urllib.error.HTTPError) as error:
        _post_json(url, payload)
    return error.value.code, json.loads(error.value.read().decode("utf-8"))


# ── Carrying hand edits onto another fit (D5) ────────────────────


def test_carried_spline_edit_is_clamped_past_a_narrower_fold_range(cv_frame, cv_fit):
    from superglm.editor.apply import _as_dense
    from superglm.editor.carry import model_with_edited_curves
    from superglm.plotting.comparison import _feature_beta

    X, y, w = cv_frame
    model, _supplied = cv_fit
    session = EditorSession.from_model(
        model, terms=["age"], train_data=(X.iloc[:400], y[:400], w[:400])
    )
    term = session.terms["age"]
    # A straight line lies in every spline basis, so the carried curve can
    # match it exactly; only its centring constant may move.
    line = 0.01 * term.x
    session.set_values("age", np.arange(term.size), line)
    narrow = X["age"].to_numpy()[:400] < 55.0
    fold = _model().fit(X.iloc[:400][narrow], y[:400][narrow], sample_weight=w[:400][narrow])

    carried = model_with_edited_curves(
        fold,
        {"age": session.terms["age"].copy()},
        X.iloc[:400][narrow],
        y[:400][narrow],
        w[:400][narrow],
        n_points=session.n_points,
    )

    spec = carried._specs["age"]
    lo, hi = spec.fitted_boundary
    beta = _feature_beta(carried, "age")
    curve = spec.score(term.x, beta)
    inside = term.x <= hi
    assert hi < term.x[-1]
    assert np.all(np.isfinite(curve))
    # Past the fold's range the spline holds its end value, as it does at
    # predict time: equal to it up to two dot products' rounding (Higham sec. 3.1).
    end_row = _as_dense(spec.transform(np.array([hi])))[0]
    end_tol = 2 * end_row.size * _U * np.sum(np.abs(end_row * beta))
    assert np.max(np.abs(curve[~inside] - spec.score(np.array([hi]), beta)[0])) <= end_tol
    # Inside it, the edited line plus one constant. A least-squares fit of a
    # representable target is accurate to about cond(design) * n * u * |target|
    # (Higham, Accuracy and Stability of Numerical Algorithms, 2nd ed., sec. 20.1).
    grid = np.linspace(lo, hi, session.n_points)
    design = np.column_stack([np.ones(grid.size), _as_dense(spec.transform(grid))])
    tol = np.linalg.cond(design) * grid.size * _U * np.max(np.abs(line))
    assert np.ptp(curve[inside] - line[inside]) <= tol
    assert np.all(np.isfinite(carried.predict(X.iloc[:400][~narrow])))


def test_carried_level_edit_keeps_its_change_when_the_reference_moves():
    from superglm.editor.carry import model_with_edited_curves

    rng = np.random.default_rng(7)

    def rows(n, shares):
        x = rng.uniform(0.0, 10.0, n)
        region = rng.choice(["A", "B", "C"], n, p=shares)
        eta = -0.4 + 0.1 * np.sin(x) + np.select([region == "B", region == "C"], [0.3, -0.2], 0.0)
        return pd.DataFrame({"x": x, "region": region}), rng.poisson(np.exp(eta)).astype(float)

    X_train, y_train = rows(400, [0.5, 0.3, 0.2])
    X_more, y_more = rows(400, [0.1, 0.7, 0.2])
    X_all = pd.concat([X_train, X_more], ignore_index=True)
    y_all = np.concatenate([y_train, y_more])

    def fit(X, y):
        features = {"x": Spline(n_knots=6), "region": Categorical(base="most_exposed")}
        return SuperGLM(family="poisson", selection_penalty=0.0, features=features).fit(X, y)

    model = fit(X_train, y_train)
    refit = fit(X_all, y_all)
    assert (model._specs["region"]._base_level, refit._specs["region"]._base_level) == ("A", "B")
    session = EditorSession.from_model(model, train_data=(X_train, y_train))
    session.select_levels("region", ["C"])
    session.shift("region", 0.1)
    term = session.terms["region"]

    carried = model_with_edited_curves(
        refit, {"region": term.copy()}, X_all, y_all, n_points=session.n_points
    )

    probe = pd.DataFrame({"x": [5.0] * 3, "region": term.levels})
    log_carried = np.log(carried.predict(probe))
    change = log_carried - np.log(refit.predict(probe))
    edit = term.edited_log_effect - term.original_log_effect
    exposure = np.array([np.sum(X_all["region"] == level) for level in term.levels], dtype=float)
    tol = 64 * _U * max(1.0, np.max(np.abs(log_carried)))
    # The edited relativities hold exactly...
    np.testing.assert_allclose(
        log_carried - log_carried[0],
        term.edited_log_effect - term.edited_log_effect[0],
        rtol=0.0,
        atol=tol,
    )
    # ...and so does the edit's exposure-weighted change, whatever base the refit chose.
    assert abs(np.average(change, weights=exposure) - np.average(edit, weights=exposure)) <= tol


# ── Stored folds and the supplied result's rows ──────────────────


def test_stored_folds_replays_indices_with_a_hook_between_folds(cv_frame, cv_fit):
    from superglm.editor.cv import StoredFolds

    X, y, w = cv_frame
    _model_unused, supplied = cv_fit
    folds = tuple(supplied.fold_indices)
    events = []

    def score(model, X_val, y_val, *, sample_weight=None, offset=None):
        events.append(("score", len(y_val)))
        return {"rows": float(len(y_val))}

    replayed = cross_validate(
        _model(),
        X.iloc[:400],
        y[:400],
        cv=StoredFolds(folds, before_fold=lambda index: events.append(("before", index))),
        sample_weight=w[:400],
        scoring=score,
    )

    expected = []
    for index, (_train, test) in enumerate(folds):
        expected += [("before", index), ("score", len(test))]
    assert events == expected
    for (train, test), (again_train, again_test) in zip(folds, replayed.fold_indices, strict=True):
        np.testing.assert_array_equal(again_train, train)
        np.testing.assert_array_equal(again_test, test)


def test_cv_data_is_checked_against_the_folds(cv_frame, cv_fit):
    from superglm.editor.cv import (
        FINGERPRINT_MISMATCH,
        NO_CV,
        NO_FINGERPRINT,
        ROWS_MISMATCH,
        TRAIN_ROWS_MISMATCH,
    )

    X, y, w = cv_frame
    model, supplied = cv_fit
    rows = (X.iloc[:400], y[:400], w[:400])

    matched = EditorSession.from_model(model, terms=["region"], train_data=rows, cv=supplied)
    assert (matched.cv_check.reason, matched.cv_check.note) == (None, None)
    assert matched.cv_check.rows.n_obs == 400

    fewer = EditorSession.from_model(
        model, terms=["region"], cv=supplied, cv_data=(X.iloc[:399], y[:399], w[:399])
    )
    assert fewer.cv_check.rows is None
    assert fewer.cv_check.reason == ROWS_MISMATCH.format(rows=399, expected=400)

    wider = EditorSession.from_model(
        model, terms=["region"], train_data=(X.iloc[:500], y[:500], w[:500]), cv=supplied
    )
    assert wider.cv_check.reason == TRAIN_ROWS_MISMATCH.format(rows=500, expected=400)

    reordered = (X.iloc[:400][::-1], y[:400][::-1], w[:400][::-1])
    moved = EditorSession.from_model(model, terms=["region"], cv=supplied, cv_data=reordered)
    assert moved.cv_check.reason == FINGERPRINT_MISMATCH

    older = dataclasses.replace(supplied, n_rows=None, data_fingerprint=None)
    noted = EditorSession.from_model(model, terms=["region"], cv=older, cv_data=reordered)
    assert noted.cv_check.reason is None
    assert noted.cv_check.note == NO_FINGERPRINT

    assert EditorSession.from_model(model, terms=["region"]).cv_check.reason == NO_CV
    with pytest.raises(TypeError, match="not a splitter"):
        EditorSession.from_model(model, cv=KFold(3))


def test_edit_takes_split_data_and_a_cv_result(cv_frame, cv_fit):
    from superglm.editor import edit
    from superglm.editor.evaluation import evaluation_datasets

    model, supplied = cv_fit
    session = edit(model, ["region"], cv=supplied, **_splits(cv_frame))

    assert session.cv is supplied
    assert [dataset.name for dataset in evaluation_datasets(session)] == [
        "train",
        "validation",
        "test",
    ]
    assert session.cv_check.reason is None


# ── The tab from a supplied result ───────────────────────────────


def test_cv_report_shows_a_supplied_result_least_stable_first(cv_frame, cv_fit):
    from superglm.plotting.curve_similarity import _summarize_against_fold_mean

    X, _y, _w = cv_frame
    model, supplied = cv_fit
    session = EditorSession.from_model(model, cv=supplied, **_splits(cv_frame))
    widget = session.widget()
    try:
        report = _post_json(f"{widget.url}/report", {"report": "cv"})
    finally:
        widget.close()

    assert report["report"] == "cv"
    assert report["header"] == {"supplied": True, "n_folds": 3, "splitter": "KFold", "n_rows": 400}
    assert [metric["name"] for metric in report["metrics"]] == ["deviance", "gini", "nll"]
    [result] = report["results"]
    assert (result["label"], result["origin"], result["stale"]) == (
        "As supplied",
        "supplied",
        False,
    )
    assert [fold["n_test"] for fold in result["folds"]] == [
        len(t) for _, t in supplied.fold_indices
    ]
    assert result["mean"]["deviance"] == supplied.mean_scores["deviance"]
    assert result["pooled"]["deviance"] == supplied.pooled_scores["deviance"]
    assert report["run_cv"] == {"available": True, "reason": None, "note": None}

    relativities = report["relativities"]
    assert relativities["origin"] == "supplied"
    # A linear term is one slope, so it has no curve to compare.
    assert {item["name"] for item in relativities["terms"]} == {"age", "region"}
    spreads = [item["spread"] for item in relativities["terms"]]
    assert spreads == sorted(spreads, reverse=True)
    region = next(item for item in relativities["terms"] if item["name"] == "region")
    assert list(pd.unique(X["region"])) != ["A", "B", "C"]
    assert region["levels"] == ["A", "B", "C"]
    weights = np.asarray(region["weights"])
    curves = {fold["label"]: np.asarray(fold["values"]) for fold in region["folds"]}
    for values in curves.values():
        log_values = np.log(values)
        assert abs(np.average(log_values, weights=weights)) <= 64 * _U * np.max(np.abs(log_values))
    assert region["spread"] == _summarize_against_fold_mean(curves, weights)["rmse_to_mean"].mean()


def test_cv_report_says_how_to_get_what_is_missing(cv_frame, cv_fit):
    from superglm.editor.cv import NO_CV, NO_ESTIMATORS

    model, supplied = cv_fit
    bare = dataclasses.replace(supplied, estimators=None)
    with_bare = EditorSession.from_model(model, cv=bare, **_splits(cv_frame)).widget()
    without = EditorSession.from_model(model, **_splits(cv_frame)).widget()
    try:
        bare_report = with_bare._report("cv")
        empty_report = without._report("cv")
    finally:
        with_bare.close()
        without.close()

    assert bare_report["relativities"] == {
        "available": False,
        "origin": None,
        "stale": False,
        "note": NO_ESTIMATORS,
        "terms": [],
    }
    assert (empty_report["note"], empty_report["results"]) == (NO_CV, [])
    assert empty_report["run_cv"]["reason"] == NO_CV
    assert empty_report["final_fit"]["available"] is True
```

These cover Review Focus 5: a mismatched row count, a train fallback that does not match,
reordered rows (a different fingerprint), an older result without a fingerprint (row-count check
plus `NO_FINGERPRINT`), and a hand-edited spline on a fold whose range is narrower than the
editor's grid. That last curve is finite, holds its end value past the fold's range and is
the edited line plus a constant inside it. The level-carry test is the anchor's
mutation check: make `carry._anchor` return `0.0` and its last assertion fails by 0.36.

- [ ] **Step 2: Run it, expect FAIL.**

```bash
./.venv/bin/python -m pytest tests/test_editor_cv.py -q
```

Expected on 155832e8, and after G1: `7 failed`. That is `ModuleNotFoundError: No module named
'superglm.editor.carry'` twice, `ModuleNotFoundError: No module named 'superglm.editor.cv'` three
times, and `TypeError: EditorSession.from_model() got an unexpected keyword argument 'cv'` and
`TypeError: edit() got an unexpected keyword argument 'cv'` once each.

- [ ] **Step 3: Implement.**

3a. Create `src/superglm/editor/carry.py`:

```python
"""Put hand-edited curves back on another fitted model of the same structure.

Run CV refits the in-force structure on each fold and Final fit refits it on
train and validation rows together; both then put the hand edits back as
set (D5). Each refit draws its own grid: a fold's numeric range is its own
training range and its level universe is its own. This module carries every
edited curve onto that grid and returns the edited copy that
:meth:`EditorSession.to_model` makes, so the refit is patched by the same
code an editor export uses.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from superglm.editor.terms import native_log_effect_values, term_offset_values

if TYPE_CHECKING:
    from superglm.editor._types import EditableTerm


def model_with_edited_curves(
    model,
    edited: dict[str, EditableTerm],
    X,
    y,
    sample_weight=None,
    offset=None,
    *,
    n_points: int = 200,
):
    """Return a copy of fitted ``model`` with each ``edited`` curve put back as set.

    ``edited`` maps term names to the editor's terms, each a main effect of
    ``model``. A numeric curve is carried by ``np.interp`` on ``x``, held at
    its end values past either end of the editor's grid (a piecewise term
    follows its own extrapolation, as its offset does). A level curve is
    carried by label; a level the editor never showed keeps its refit value.

    The carried curve keeps the edited shape exactly and moves by one
    constant: the exposure-weighted mean of the refit curve minus the
    editor's original curve. That constant is the difference between the
    two fits' centring, a fold's own centring or a reference level that
    re-resolved on more rows, so it does not move predictions; an edit that
    shifted the whole curve keeps its shift.

    ``X``, ``y``, ``sample_weight`` and ``offset`` are the rows ``model`` was
    fitted on. They weight the projection onto the refit's basis and refresh
    the copy's fit statistics. ``model`` itself is not changed.
    """
    from superglm.editor.session import EditorSession

    if not edited:
        return model
    session = EditorSession.from_model(
        model,
        list(edited),
        n_points=n_points,
        centering="native",
        with_se=False,
        train_data=(X, y, sample_weight, offset),
    )
    for name, source in edited.items():
        target = session.terms[name]
        target.edited_log_effect = _carried_values(source, target)
    return session.to_model(X=X, y=y, sample_weight=sample_weight, offset=offset)


def _carried_values(source: EditableTerm, target: EditableTerm) -> NDArray[np.float64]:
    """``source``'s edited curve on ``target``'s grid, in ``target``'s centring."""
    refit = np.asarray(target.edited_log_effect, dtype=np.float64)
    edited = native_log_effect_values(source)
    if source.size == 1:
        # A linear term's one value is its slope, which carries no centring.
        return edited.copy()
    original = _original_curve(source)
    weights = _weights(target)
    if target.levels is not None:
        position = {label: index for index, label in enumerate(source.levels or ())}
        shared = np.array([label in position for label in target.levels], dtype=bool)
        index = [position[label] for label in target.levels if label in position]
        values = refit.copy()
        anchor = _anchor(refit[shared], native_log_effect_values(original)[index], weights[shared])
        values[shared] = edited[index] + anchor
        return values
    x = np.asarray(target.x, dtype=np.float64)
    anchor = _anchor(refit, term_offset_values(original, x), weights)
    return term_offset_values(source, x) + anchor


def _original_curve(term: EditableTerm) -> EditableTerm:
    """``term`` with its edits undone, so the offset helpers read the original curve."""
    unedited = term.copy()
    unedited.edited_log_effect = unedited.original_log_effect.copy()
    return unedited


def _weights(term: EditableTerm) -> NDArray[np.float64]:
    if term.weights is None:
        return np.ones(term.size, dtype=np.float64)
    return np.asarray(term.weights, dtype=np.float64)


def _anchor(refit: NDArray, original: NDArray, weights: NDArray) -> float:
    """Exposure-weighted mean of ``refit - original``; unweighted when no row backs it."""
    if refit.size == 0:
        return 0.0
    if not float(np.sum(weights)) > 0.0:
        weights = np.ones_like(refit)
    return float(np.average(refit - original, weights=weights))


__all__ = ["model_with_edited_curves"]
```

3b. Create `src/superglm/editor/cv.py`. In this task `cv_report_payload` reports
`jobs` as `None`; G3 wires the runner in.

```python
"""The Cross-validation tab: a supplied result, Run CV and Final fit.

``edit(model, cv=result)`` hands the editor a :class:`CrossValidationResult`.
The tab shows its fold scores and, when it kept its fold models, each
term's relativities by fold, least stable first. Run CV replays the
result's own folds on the in-force structure with the hand edits put back
on every fold (D5): :class:`StoredFolds` feeds ``cross_validate`` the
recorded indices, and its per-fold hook is where progress is reported and a
cancel is honoured. Final fit refits the in-force structure on train and
validation rows together, puts the hand edits back, and is offered by
Export (D6).

Every refusal is a fixed sentence (``editor/errors.py``).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from superglm.editor.evaluation import EvaluationDataset, training_export_dataset
from superglm.model_selection import CrossValidationResult, _data_fingerprint
from superglm.plotting.comparison import _feature_beta
from superglm.plotting.curve_similarity import _summarize_against_fold_mean

if TYPE_CHECKING:
    from superglm.editor._types import EditableTerm

JOB_KINDS = ("cv", "final_fit")

NO_CV = (
    "No cross-validation result was supplied. Pass cv=cross_validate(..., "
    "return_estimators=True) to edit()."
)
NO_FOLDS = "This result has no fold indices, so its folds cannot be run again."
NO_ROWS = (
    "Run CV needs the rows the folds were drawn on. Pass cv_data=(X, y, sample_weight) to edit()."
)
ROWS_MISMATCH = "The CV data has {rows:,} rows, but the folds were drawn on {expected:,}."
TRAIN_ROWS_MISMATCH = (
    "The train data has {rows:,} rows, but the folds were drawn on {expected:,}. "
    "Pass cv_data=(X, y, sample_weight) to edit()."
)
FINGERPRINT_MISMATCH = (
    "The CV data's response or weights differ from the data the folds were drawn on. "
    "Pass the same rows, in the same order, as cv_data."
)
NO_FINGERPRINT = (
    "This result was made before cross_validate recorded a data fingerprint, so only "
    "the row count was checked."
)
NO_ESTIMATORS = (
    "Fold curves need the fold models: pass return_estimators=True to cross_validate, "
    "or run CV on the current model."
)
NO_FINAL_ROWS = "Final fit needs train_data, or a model that kept its fit data."
TRAIN_ONLY = "No validation data was supplied, so Final fit uses the train rows only."

_METRICS = (
    ("deviance", "Mean deviance", True),
    ("gini", "Gini", False),
    ("nll", "Negative log-likelihood", True),
)
_FOLD_COLUMNS = ("fold", "n_train", "n_test", "fit_time_s", "converged", "n_iter", "effective_df")


def waiting_sentence(count: int) -> str:
    """'1 change is waiting' or 'N changes are waiting'."""
    return "1 change is waiting" if count == 1 else f"{count} changes are waiting"


def _not_included(count: int) -> str:
    if count == 1:
        return "1 waiting change is not included."
    return f"{count} waiting changes are not included."


@dataclass(frozen=True)
class CVDataCheck:
    """The rows Run CV replays a supplied result's folds on, or why it cannot."""

    rows: EvaluationDataset | None
    reason: str | None = None
    note: str | None = None


@dataclass(frozen=True)
class StoredFolds:
    """A splitter that yields recorded ``(train_idx, test_idx)`` pairs.

    ``cross_validate`` asks for the next fold only after scoring the last, so
    ``before_fold(index)`` runs between folds, before fold ``index`` is
    fitted. Raising there stops the run without fitting that fold.
    """

    folds: tuple[tuple[NDArray[np.intp], NDArray[np.intp]], ...]
    before_fold: Callable[[int], None] | None = None

    def split(self, X=None, y=None, groups=None):
        del X, y, groups
        for index, (train, test) in enumerate(self.folds):
            if self.before_fold is not None:
                self.before_fold(index)
            yield train, test

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        del X, y, groups
        return len(self.folds)


@dataclass(frozen=True)
class CVRun:
    """A finished Run CV, kept on the widget with the revision it ran on."""

    result: CrossValidationResult
    terms: list[dict[str, Any]]
    model_revision: int
    carried: tuple[str, ...]


@dataclass(frozen=True)
class FinalFit:
    """A finished Final fit, kept on the widget with the revision it ran on."""

    model: Any
    model_revision: int
    n_rows: int
    splits: tuple[str, ...]
    carried: tuple[str, ...]
    pending: int


# ── The supplied result and its rows ─────────────────────────────


def check_cv_data(
    cv: CrossValidationResult | None,
    cv_rows: EvaluationDataset | None,
    fallback: EvaluationDataset | None,
) -> CVDataCheck:
    """Decide which rows Run CV replays ``cv``'s folds on.

    ``cv_rows`` is ``cv_data=`` when it was supplied; otherwise ``fallback``,
    the train data, is used when its row count matches the folds. A result
    that records a data fingerprint must match it. An older one gets the
    row-count check and a note saying so.
    """
    if cv is None:
        return CVDataCheck(None, NO_CV)
    if not cv.fold_indices:
        return CVDataCheck(None, NO_FOLDS)
    rows = fallback if cv_rows is None else cv_rows
    if rows is None:
        return CVDataCheck(None, NO_ROWS)
    expected = _expected_rows(cv)
    if rows.n_obs != expected:
        sentence = TRAIN_ROWS_MISMATCH if cv_rows is None else ROWS_MISMATCH
        return CVDataCheck(None, sentence.format(rows=rows.n_obs, expected=expected))
    if cv.data_fingerprint is None:
        return CVDataCheck(rows, note=NO_FINGERPRINT)
    if cv.data_fingerprint != _data_fingerprint(rows.y, rows.sample_weight):
        return CVDataCheck(None, FINGERPRINT_MISMATCH)
    return CVDataCheck(rows)


def _expected_rows(cv: CrossValidationResult) -> int:
    """The recorded row count, or one past the largest index an older result holds."""
    if cv.n_rows is not None:
        return int(cv.n_rows)
    return 1 + max(int(np.max(np.concatenate(fold))) for fold in cv.fold_indices)


def run_cv_reason(session) -> str | None:
    """Why Run CV is disabled now, or None (D7: it waits for Refit)."""
    if session.cv_check.reason is not None:
        return session.cv_check.reason
    if session.pending:
        return f"Refit first: {waiting_sentence(len(session.pending))}."
    return None


def final_fit_datasets(session) -> tuple[EvaluationDataset, ...]:
    """Train and validation rows (D6); the test split stays held out."""
    train = training_export_dataset(session)
    if train is None:
        return ()
    validation = session._evaluation_data.get("validation")
    return (train,) if validation is None else (train, validation)


# ── Relativities by fold ─────────────────────────────────────────


@dataclass(frozen=True)
class _TermGrid:
    """Where a term's fold curves are read: the editor's grid, levels in model order."""

    kind: str
    points: NDArray
    order: NDArray[np.intp]
    labels: list[str] | None


def _term_grid(term: EditableTerm) -> _TermGrid | None:
    if term.levels is not None:
        position = {label: index for index, label in enumerate(term.levels)}
        native = list(term.metadata.get("native_levels", term.levels))
        in_model_order = [level for level in native if str(level) in position]
        labels = [str(level) for level in in_model_order]
        return _TermGrid(
            kind="levels",
            points=np.asarray(in_model_order, dtype=object),
            order=np.asarray([position[label] for label in labels], dtype=np.intp),
            labels=labels,
        )
    if term.x is None or term.size < 2:
        # A linear term is one slope; its fold spread is a coefficient's.
        return None
    return _TermGrid(
        kind="continuous",
        points=np.asarray(term.x, dtype=np.float64),
        order=np.arange(term.size, dtype=np.intp),
        labels=None,
    )


def fold_log_curves(model, terms: Mapping[str, EditableTerm]) -> dict[str, NDArray[np.float64]]:
    """One fold model's log curve for each term, read on the editor's grid or levels.

    The editor's grid can reach past a fold's training range; a spline holds
    its end value there, as it does at predict time. A fold that cannot
    score a term's points, a level it never saw, is left out of that term.
    """
    curves: dict[str, NDArray[np.float64]] = {}
    for name, term in terms.items():
        grid = _term_grid(term)
        if grid is None or name not in model._specs:
            continue
        try:
            values = model._specs[name].score(grid.points, _feature_beta(model, name))
        except (KeyError, ValueError):
            continue
        curves[name] = np.asarray(values, dtype=np.float64)
    return curves


def fold_term_items(
    terms: Mapping[str, EditableTerm],
    fold_curves: Mapping[int, Mapping[str, NDArray[np.float64]]],
) -> list[dict[str, Any]]:
    """One chart entry per term, least stable first.

    Every curve is re-centred on its exposure-weighted mean log and shown as
    a relativity. A term's ``spread`` is the mean over folds of
    ``rmse_to_mean`` on that scale, and ``min_correlation`` the lowest
    ``correlation_to_mean`` (``plotting.curve_similarity``).
    """
    items = []
    for name, term in terms.items():
        grid = _term_grid(term)
        curves = {
            f"Fold {index + 1}": by_term[name]
            for index, by_term in sorted(fold_curves.items())
            if name in by_term
        }
        if grid is not None and curves:
            items.append(_term_item(name, term, grid, curves))
    items.sort(key=lambda item: (-item["spread"], item["name"]))
    return items


def _term_item(name: str, term: EditableTerm, grid: _TermGrid, curves) -> dict[str, Any]:
    weights = (
        np.ones(grid.order.size, dtype=np.float64)
        if term.weights is None
        else np.asarray(term.weights, dtype=np.float64)[grid.order]
    )
    if not float(np.sum(weights)) > 0.0:
        weights = np.ones(grid.order.size, dtype=np.float64)

    def centred(log_values) -> NDArray[np.float64]:
        values = np.asarray(log_values, dtype=np.float64)
        return np.exp(values - np.average(values, weights=weights))

    folds = {label: centred(values) for label, values in curves.items()}
    vs_mean = _summarize_against_fold_mean(folds, weights)
    fit = term.original_log_effect[grid.order]
    edited = term.edited_log_effect[grid.order]
    changed = not np.allclose(edited, fit, rtol=0.0, atol=1e-14)
    return {
        "name": name,
        "kind": grid.kind,
        "x": None if grid.kind == "levels" else grid.points,
        "levels": grid.labels,
        "weights": weights,
        "folds": [{"label": label, "values": values} for label, values in folds.items()],
        "fit": centred(fit),
        "edited": centred(edited) if changed else None,
        "spread": float(vs_mean["rmse_to_mean"].mean()),
        "min_correlation": float(vs_mean["correlation_to_mean"].min()),
    }


# ── The tab's payload ────────────────────────────────────────────


@dataclass(frozen=True)
class CVTabView:
    """What the tab shows, captured under the widget lock."""

    supplied: CrossValidationResult | None
    check: CVDataCheck
    terms: dict[str, EditableTerm]
    run: CVRun | None
    final_fit: FinalFit | None
    model_revision: int
    model_changed: bool
    pending: int
    run_reason: str | None
    final_reason: str | None
    has_validation: bool


def capture_cv_view(session, *, run: CVRun | None, final_fit: FinalFit | None) -> CVTabView:
    """Copy what the tab reads; the caller holds the widget lock."""
    return CVTabView(
        supplied=session.cv,
        check=session.cv_check,
        terms={name: term.copy() for name, term in session.terms.items()},
        run=run,
        final_fit=final_fit,
        model_revision=session.model_revision,
        model_changed=session.model is not session.reference_model or bool(session.edited_terms()),
        pending=len(session.pending),
        run_reason=run_cv_reason(session),
        final_reason=None if final_fit_datasets(session) else NO_FINAL_ROWS,
        has_validation="validation" in session._evaluation_data,
    )


def cv_report_payload(widget, *, request_sequence: int | None = None) -> dict[str, Any]:
    """The ``cv`` report: captured under the widget lock, built outside it."""
    with widget._lock:
        view = capture_cv_view(widget.session, run=widget._cv_run, final_fit=widget._final_fit)
    jobs = {kind: None for kind in JOB_KINDS}
    return cv_tab_payload(view, jobs=jobs, request_sequence=request_sequence)


def cv_tab_payload(
    view: CVTabView,
    *,
    jobs: Mapping[str, dict[str, Any] | None],
    request_sequence: int | None = None,
) -> dict[str, Any]:
    """The tab: header, fold performance, the fold table's rows and relativities by fold."""
    results = []
    if view.supplied is not None:
        results.append(
            _result_payload(view.supplied, "As supplied", "supplied", None, view.model_changed)
        )
    run = view.run
    if run is not None:
        results.append(
            _result_payload(
                run.result,
                "Current model",
                "run",
                run.model_revision,
                run.model_revision != view.model_revision,
            )
        )
    final = view.final_fit
    final_notes = [
        _not_included(view.pending) if view.pending else None,
        None if view.has_validation else TRAIN_ONLY,
    ]
    return {
        "available": True,
        "report": "cv",
        "title": "Cross-validation",
        "note": "" if results else NO_CV,
        "model_revision": view.model_revision,
        "request_sequence": request_sequence,
        "header": _header(view),
        "pending": view.pending,
        "run_cv": {
            "available": view.run_reason is None,
            "reason": view.run_reason,
            "note": view.check.note,
        },
        "final_fit": {
            "available": view.final_reason is None,
            "reason": view.final_reason,
            "note": " ".join(note for note in final_notes if note) or None,
            "done": final is not None,
            "stale": final is not None and final.model_revision != view.model_revision,
            "n_rows": None if final is None else final.n_rows,
        },
        "metrics": [
            {"name": name, "label": label, "lower_is_better": lower}
            for name, label, lower in _METRICS
            if any(name in result["mean"] for result in results)
        ],
        "results": results,
        "relativities": _relativities(view),
        "jobs": dict(jobs),
    }


def _header(view: CVTabView) -> dict[str, Any]:
    supplied = view.supplied
    if supplied is None:
        return {"supplied": False, "n_folds": 0, "splitter": None, "n_rows": None}
    rows = None if view.check.rows is None else view.check.rows.n_obs
    return {
        "supplied": True,
        "n_folds": len(supplied.fold_indices or supplied.fold_scores),
        "splitter": supplied.splitter,
        "n_rows": rows if supplied.n_rows is None else supplied.n_rows,
    }


def _result_payload(
    result: CrossValidationResult,
    label: str,
    origin: str,
    model_revision: int | None,
    stale: bool,
) -> dict[str, Any]:
    names = [name for name, _label, _lower in _METRICS if name in result.fold_scores.columns]
    folds = [
        {
            **{column: record.get(column) for column in _FOLD_COLUMNS},
            "scores": {name: record.get(name) for name in names},
        }
        for record in result.fold_scores.to_dict("records")
    ]
    return {
        "label": label,
        "origin": origin,
        "model_revision": model_revision,
        "stale": bool(stale),
        "folds": folds,
        "mean": {name: result.mean_scores.get(name) for name in names},
        "std": {name: result.std_scores.get(name) for name in names},
        "pooled": {
            name: result.pooled_scores[name] for name in names if name in result.pooled_scores
        },
    }


def _relativities(view: CVTabView) -> dict[str, Any]:
    run = view.run
    if run is not None and run.terms:
        return {
            "available": True,
            "origin": "run",
            "stale": run.model_revision != view.model_revision,
            "note": None,
            "terms": run.terms,
        }
    estimators = [] if view.supplied is None else list(view.supplied.estimators or ())
    curves = {
        index: fold_log_curves(model, view.terms)
        for index, model in enumerate(estimators)
        if model is not None
    }
    if not curves:
        return {
            "available": False,
            "origin": None,
            "stale": False,
            "note": NO_ESTIMATORS,
            "terms": [],
        }
    return {
        "available": True,
        "origin": "supplied",
        "stale": view.model_changed,
        "note": None,
        "terms": fold_term_items(view.terms, curves),
    }
```

3c. `src/superglm/editor/session.py`. The cv imports go in their isort places. The data check
runs once at construction, because its inputs are constructor arguments. Without `cv=` it
returns `NO_CV` before hashing anything, so the sessions `carry.py` builds per fold cost nothing.

Edit 1 (master line 24). Replace

```python
from superglm.editor.controls import control_points as _control_points
from superglm.editor.errors import (
```

with

```python
from superglm.editor.controls import control_points as _control_points
from superglm.editor.cv import check_cv_data
from superglm.editor.errors import (
```

Edit 2 (master line 32). Replace

```python
from superglm.editor.evaluation import coerce_evaluation_data, default_metrics_dataset
```

with

```python
from superglm.editor.evaluation import (
    EvaluationDataset,
    coerce_dataset,
    coerce_evaluation_data,
    default_metrics_dataset,
    training_export_dataset,
)
```

Edit 3 (master line 60). Replace

```python
from superglm.solvers.dispersion import model_weight_semantics
```

with

```python
from superglm.model_selection import CrossValidationResult
from superglm.solvers.dispersion import model_weight_semantics
```

Edit 4 (master line 134). Replace

```python
        evaluation_data: dict[str, Any] | None = None,
        cv_report: Any = None,
    ):
        self.model = model
```

with

```python
        evaluation_data: dict[str, Any] | None = None,
        cv_report: Any = None,
        cv: CrossValidationResult | None = None,
        cv_data: EvaluationDataset | None = None,
    ):
        self.model = model
```

Edit 5 (master line 143). Replace

```python
        self.cv_report = cv_report
```

with

```python
        self.cv_report = cv_report
        if cv is not None and not isinstance(cv, CrossValidationResult):
            raise TypeError(
                "cv= takes the CrossValidationResult that superglm.cross_validate returns, "
                "not a splitter."
            )
        if cv_data is not None and cv is None:
            raise ValueError("cv_data= holds the rows a cv= result's folds index; pass cv= too.")
        self.cv = cv
        # Fixed for the session's life: cv, cv_data and the train split are
        # constructor inputs, so the rows Run CV replays never change.
        self.cv_check = check_cv_data(cv, cv_data, training_export_dataset(self))
```

Edit 6 (master line 171). Replace

```python
        test_data=None,
        cv_report: Any = None,
    ) -> EditorSession:
        """Build an editor session from fitted 1D main-effect inference."""
        if getattr(model, "_result", None) is None:
```

with

```python
        test_data=None,
        cv_report: Any = None,
        cv: CrossValidationResult | None = None,
        cv_data=None,
    ) -> EditorSession:
        """Build an editor session from fitted 1D main-effect inference.

        ``cv`` is a :func:`superglm.cross_validate` result for the
        Cross-validation tab, and ``cv_data`` the ``(X, y[, sample_weight[,
        offset]])`` rows its folds index. Without ``cv_data`` the train data
        is used when its row count matches the folds. ``cv_report`` is the
        older free-form report the Validation tab shows.
        """
        if getattr(model, "_result", None) is None:
```

Edit 7 (master line 183). Replace

```python
            weight_semantics=model_weight_semantics(model),
        )
        names = list(model._feature_order if terms is None else terms)
```

with

```python
            weight_semantics=model_weight_semantics(model),
        )
        cv_rows = coerce_dataset(
            "cv",
            cv_data,
            family=model._distribution,
            weight_semantics=model_weight_semantics(model),
        )
        names = list(model._feature_order if terms is None else terms)
```

Edit 8 (master line 200). Replace

```python
            evaluation_data=evaluation_data,
            cv_report=cv_report,
        )
```

with

```python
            evaluation_data=evaluation_data,
            cv_report=cv_report,
            cv=cv,
            cv_data=cv_rows,
        )
```

Edit 9 (master line 1609). Replace

```python
    centering: str = "native",
    with_se: bool = True,
) -> EditorSession:
    """Create an editor session for a fitted model."""
    return EditorSession.from_model(
        model,
        terms=terms,
        n_points=n_points,
        centering=centering,
        with_se=with_se,
    )
```

with

```python
    centering: str = "native",
    with_se: bool = True,
    train_data=None,
    validation_data=None,
    test_data=None,
    cv: CrossValidationResult | None = None,
    cv_data=None,
) -> EditorSession:
    """Create an editor session for a fitted model.

    The split data are ``(X, y[, sample_weight[, offset]])`` tuples.
    ``cv`` is a :func:`superglm.cross_validate` result for the
    Cross-validation tab and ``cv_data`` the rows its folds index; see
    :meth:`EditorSession.from_model`.
    """
    return EditorSession.from_model(
        model,
        terms=terms,
        n_points=n_points,
        centering=centering,
        with_se=with_se,
        train_data=train_data,
        validation_data=validation_data,
        test_data=test_data,
        cv=cv,
        cv_data=cv_data,
    )
```

3d. `src/superglm/editor/reports.py`:

Edit 1 (master line 7). Replace

```python
from superglm.editor.evaluation import evaluation_datasets
```

with

```python
from superglm.editor.cv import cv_report_payload
from superglm.editor.evaluation import evaluation_datasets
```

Edit 2 (master line 85). Replace

```python
    """Dispatch a named report for the local editor app."""
    if report == "final":
```

with

```python
    """Dispatch a named report for the local editor app."""
    if report == "cv":
        return cv_report_payload(widget, request_sequence=request_sequence)
    if report == "final":
```

3e. `src/superglm/editor/widget.py`. The `cv` report needs no split metrics, so it branches
before them, and it is dropped if the revision moved while it was built:

Edit 1 (master line 25). Replace

```python
from superglm.editor.apply import materialize_edit_request
```

with

```python
from superglm.editor.apply import materialize_edit_request
from superglm.editor.cv import CVRun, FinalFit
```

Edit 2 (master line 128). Replace

```python
        self._profile_condition = threading.Condition(threading.RLock())
```

with

```python
        self._profile_condition = threading.Condition(threading.RLock())
        # The Cross-validation tab's results, each kept with the revision it
        # ran on: a newer revision marks them stale rather than dropping them.
        self._cv_run: CVRun | None = None
        self._final_fit: FinalFit | None = None
```

Edit 3 (master line 549). Replace

```python
            if model_revision is not None and int(model_revision) != current_revision:
                return _superseded_payload(int(model_revision), request_sequence)
        edited_model, revision = self._current_model_for_evidence()
```

with

```python
            if model_revision is not None and int(model_revision) != current_revision:
                return _superseded_payload(int(model_revision), request_sequence)
        if report == "cv":
            payload = report_payload(self, report, request_sequence=request_sequence)
            with self._lock:
                if payload["model_revision"] != self.session.model_revision:
                    return _superseded_payload(payload["model_revision"], request_sequence)
            return payload
        edited_model, revision = self._current_model_for_evidence()
```

- [ ] **Step 4: Run tests, expect PASS.**

```bash
./.venv/bin/python -m pytest tests/test_editor_cv.py -q
./.venv/bin/python -m pytest tests/test_editor.py tests/test_editor_structure.py -q -n 8 -k "cv or report or session"
./.venv/bin/ruff check src/superglm/editor tests/test_editor_cv.py
./.venv/bin/ruff format --check src/superglm/editor tests/test_editor_cv.py
```

Expected: 7 passed in `test_editor_cv.py`. The `cv_report=` tests (`test_editor.py` 5302,
5521, 5585) still pass unchanged (D12): Validation still shows `cv_report` and
`can_run_cv` stays `False` there.

- [ ] **Step 5: Commit.**

```bash
git add src/superglm/editor/carry.py src/superglm/editor/cv.py src/superglm/editor/session.py \
  src/superglm/editor/reports.py src/superglm/editor/widget.py tests/test_editor_cv.py
git commit -F - <<'EOF'
Editor: cv= and cv_data=, carried hand edits and the Cross-validation report

edit()/from_model take a cross_validate() result and the rows its folds index, checked
by row count and data fingerprint (an older result: row count plus a note). carry.py
puts hand-edited curves back on another fit of the same structure, keeping the edited
shape and moving it only by the two fits' centring difference. The cv report shows the
supplied fold scores and relativities by fold on the editor's grid, levels in model
order, least stable first. cv_report= is unchanged.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```

---

### Task G3: Cancellable background jobs and the job routes

**Files:**
- Create: `src/superglm/editor/jobs.py`
- Modify: `src/superglm/editor/widget.py`: import (line 39), `__init__` (after G2's lines), `close()`
  (line 164) and new methods before `_structural_transition` (line 912)
- Modify: `src/superglm/editor/server.py`: three routes after `/profile_distribution/status/{job_id}` (line 244)
- Modify: `src/superglm/editor/cv.py`: `cv_report_payload` reports the runner's latest jobs
- Test: `tests/test_editor_cv.py` (append), `tests/test_editor.py`: route list (line 6875)

**Interfaces:**
- Consumes: the profile-job pattern (widget.py:805–910, a thread plus a job dict under a
  Condition), fixed in two ways. It has a cancel flag checked between steps, and it never holds
  the widget lock while it works: the starter captures inputs under the lock, `work` runs
  unlocked, and `publish` re-takes the lock and stores only if the state is still current
  (the `_current_model_for_evidence` pattern, widget.py:320–350). Once `work` returns, a
  cancel is refused, so a cancelled job never publishes. Finishing a job evicts the earlier
  finished jobs of its kind, and one job per kind runs at a time.
- Produces: `jobs.JobRunner`, `jobs.JobContext`, `jobs.JobCancelledError`;
  `widget._jobs`, `widget._job_starters: dict[kind, () -> (work, publish)]`,
  `widget._job_start(kind)`, `widget._job_status(job_id, *, wait=False)`, `widget._job_cancel(job_id)`;
  `POST /job_start {kind}`, `POST /job_status {job_id, wait}`, `POST /job_cancel {job_id}`.

- [ ] **Step 1: Write the failing test.** In `tests/test_editor_cv.py`, add `import threading`
and a `_get_json` helper:

Replace

```python
import json
import urllib.error
```

with

```python
import json
import threading
import urllib.error
```

Replace

```python
def _post_error(url: str, payload: dict) -> tuple[int, dict]:
```

with

```python
def _get_json(url: str):
    from superglm.editor.widget import _LIVE_WIDGETS

    origin = url.rsplit("/", 1)[0]
    token = next(widget._token for widget in _LIVE_WIDGETS if widget.url == origin)
    request = urllib.request.Request(url, headers={"X-SuperGLM-Editor-Token": token})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _post_error(url: str, payload: dict) -> tuple[int, dict]:
```

Then append:

```python


# ── Background jobs ──────────────────────────────────────────────


def _blocking_job(entered, release, published):
    """A job that stops after its first step until ``release`` is set."""

    def work(context):
        context.progress("fold", fold=1, n_folds=2)
        entered.set()
        assert release.wait(30)
        context.check()
        context.progress("fold", fold=2, n_folds=2)
        return "value"

    def publish(value):
        published.append(value)
        return {"published": value}

    return work, publish


def test_job_runner_cancel_between_steps_publishes_nothing():
    from superglm.editor.jobs import JobRunner

    runner = JobRunner(name="test")
    entered, release, published = threading.Event(), threading.Event(), []
    job_id = runner.start("cv", *_blocking_job(entered, release, published))
    assert entered.wait(30)

    requested = runner.cancel(job_id)
    release.set()
    finished = runner.status(job_id, wait=True)
    runner.close()

    assert requested == {"job_id": job_id, "status": "running", "cancel_requested": True}
    assert finished["status"] == "cancelled"
    assert finished["progress"] == [{"phase": "fold", "fold": 1, "n_folds": 2}]
    assert published == []


def test_job_runner_runs_one_job_per_kind_and_keeps_the_last_of_each():
    from superglm.editor.errors import EditorKeyError, EditorValueError
    from superglm.editor.jobs import JobRunner

    runner = JobRunner(name="test")
    entered, release, published = threading.Event(), threading.Event(), []
    first = runner.start("cv", *_blocking_job(entered, release, published))
    assert entered.wait(30)
    with pytest.raises(EditorValueError, match="already running"):
        runner.start("cv", *_blocking_job(entered, release, published))
    other = runner.start("final_fit", lambda context: "other", lambda value: {"value": value})
    release.set()
    assert runner.status(first, wait=True)["result"] == {"published": "value"}
    assert runner.status(other, wait=True)["status"] == "done"

    second = runner.start("cv", *_blocking_job(entered, release, published))

    assert runner.status(second, wait=True)["status"] == "done"
    with pytest.raises(EditorKeyError):
        runner.status(first)
    assert runner.latest("cv")["job_id"] == second
    assert runner.latest("final_fit")["job_id"] == other
    assert published == ["value", "value"]
    runner.close()


def test_job_runner_reports_fixed_sentences_when_a_job_fails():
    from superglm.editor.errors import EditorValueError
    from superglm.editor.jobs import JobRunner

    def leak(context):
        raise RuntimeError("backend detail that must not reach the browser")

    def refuse(context):
        raise EditorValueError("A sentence written for the browser.")

    runner = JobRunner(name="test")
    leaked = runner.status(runner.start("cv", leak, lambda value: {}), wait=True)
    refused = runner.status(runner.start("final_fit", refuse, lambda value: {}), wait=True)
    runner.close()

    assert (leaked["status"], leaked["error"]) == ("failed", "internal editor error")
    assert (refused["status"], refused["error"]) == (
        "failed",
        "A sentence written for the browser.",
    )


def test_job_routes_run_a_job_off_the_widget_lock(cv_fit):
    model, _supplied = cv_fit
    widget = EditorSession.from_model(model, terms=["region"]).widget()
    entered, release, published = threading.Event(), threading.Event(), []
    widget._job_starters["probe"] = lambda: _blocking_job(entered, release, published)
    try:
        started = _post_json(f"{widget.url}/job_start", {"kind": "probe"})
        assert entered.wait(30)
        # The work is mid-run; a request that takes the widget lock still answers.
        assert _get_json(f"{widget.url}/state")["selected_term"] == "region"
        cancelled = _post_json(f"{widget.url}/job_cancel", {"job_id": started["job_id"]})
        release.set()
        finished = _post_json(
            f"{widget.url}/job_status", {"job_id": started["job_id"], "wait": True}
        )
        unknown = _post_error(f"{widget.url}/job_start", {"kind": "nonsense"})
    finally:
        release.set()
        widget.close()

    assert started["status"] == "running"
    assert cancelled["cancel_requested"] is True
    assert finished["status"] == "cancelled"
    assert published == []
    assert unknown == (400, {"error": "Unknown job kind."})
```

Also extend the route list in `tests/test_editor.py`:

Edit 1 (master line 6875). Replace

```python
    assert ("/profile_distribution/status/{job_id}", frozenset({"GET"})) in routes
```

with

```python
    assert ("/profile_distribution/status/{job_id}", frozenset({"GET"})) in routes
    assert ("/job_start", frozenset({"POST"})) in routes
    assert ("/job_status", frozenset({"POST"})) in routes
    assert ("/job_cancel", frozenset({"POST"})) in routes
```

The jobs here are deterministic: each one blocks on a `threading.Event` the test sets, and
the 30-second waits only bound a hang. Nothing asserts on time. The route test is the proof
that no lock is held: `GET /state` takes the widget lock and still answers while the job's
work is blocked.

- [ ] **Step 2: Run it, expect FAIL.**

```bash
./.venv/bin/python -m pytest tests/test_editor_cv.py -k "job" -q
./.venv/bin/python -m pytest tests/test_editor.py -k declares_fastapi_routes -q
```

Expected: `ModuleNotFoundError: No module named 'superglm.editor.jobs'` (three runner tests),
`AttributeError: 'EditorWidget' object has no attribute '_job_starters'` (the route test), and
`assert ('/job_start', frozenset({'POST'})) in routes` failing. Same on 155832e8.

- [ ] **Step 3: Implement.**

3a. Create `src/superglm/editor/jobs.py`:

```python
"""Cancellable background jobs for the editor app.

Run CV and Final fit take one fit per fold, or one fit on every row, so they
run off the request thread, each on its own daemon thread, and the browser
polls them by id. A job never holds the widget lock while it works: ``work``
runs on state captured beforehand, and ``publish`` takes the lock itself and
stores the result only if that state is still current. A cancel is a flag
the work checks between steps (between folds for Run CV); once the work has
returned, publication can no longer be cancelled, so a cancelled job never
publishes anything. A finished job evicts the finished jobs of its kind
before it, so the runner keeps the last of each kind.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from superglm.editor.errors import EditorClientError, EditorKeyError, EditorValueError
from superglm.editor.io import jsonable

_LOGGER = logging.getLogger(__name__)

_ALREADY_RUNNING = "That job is already running. Wait for it, or cancel it first."
_UNKNOWN_JOB = "Unknown job. It may have finished and been replaced by a newer one."
_INTERNAL_ERROR = "internal editor error"


class JobCancelledError(Exception):
    """Raised by :meth:`JobContext.check` once the job's cancel flag is set."""


@dataclass(frozen=True)
class JobContext:
    """What a job's work sees: its id, a cancel check and a progress channel."""

    job_id: str
    _cancel: threading.Event = field(repr=False)
    _report: Callable[[dict[str, Any]], None] = field(repr=False)

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def check(self) -> None:
        """Raise :class:`JobCancelledError` if a cancel was requested."""
        if self._cancel.is_set():
            raise JobCancelledError

    def progress(self, phase: str, **details: Any) -> None:
        """Append one progress entry the status route returns."""
        self._report({"phase": phase, **details})


class JobRunner:
    """Start, poll and cancel the editor's background jobs."""

    def __init__(self, *, name: str, wait_timeout: float = 30.0) -> None:
        self._name = name
        self._wait_timeout = float(wait_timeout)
        self._condition = threading.Condition(threading.RLock())
        self._jobs: dict[str, dict[str, Any]] = {}
        self._flags: dict[str, threading.Event] = {}
        self._counter = 0
        self._closed = False

    def start(
        self,
        kind: str,
        work: Callable[[JobContext], Any],
        publish: Callable[[Any], dict[str, Any]],
    ) -> str:
        """Run ``work`` on a new thread, then ``publish`` its value; return the job id."""
        with self._condition:
            if self._closed:
                raise RuntimeError("The job runner is closed.")
            if any(
                job["kind"] == kind and job["status"] == "running" for job in self._jobs.values()
            ):
                raise EditorValueError(_ALREADY_RUNNING)
            self._counter += 1
            job_id = f"{kind}-{self._counter}"
            flag = threading.Event()
            self._flags[job_id] = flag
            self._jobs[job_id] = {
                "job_id": job_id,
                "kind": kind,
                "status": "running",
                "progress": [],
                "result": None,
                "error": None,
                "cancel_requested": False,
                "publishing": False,
                "started_at": time.time(),
                "finished_at": None,
            }
        threading.Thread(
            target=self._run,
            args=(job_id, work, publish, flag),
            name=f"superglm-{self._name}-{job_id}",
            daemon=True,
        ).start()
        return job_id

    def status(self, job_id: str, *, wait: bool = False) -> dict[str, Any]:
        """Return a job's status; with ``wait``, first wait until it stops running."""
        with self._condition:
            self._require(job_id)
            if wait:
                self._condition.wait_for(
                    lambda: self._jobs.get(job_id, {}).get("status") != "running",
                    timeout=self._wait_timeout,
                )
            return self._snapshot(self._require(job_id))

    def cancel(self, job_id: str) -> dict[str, Any]:
        """Ask a running job to stop at its next check; a finished job is left as it is."""
        with self._condition:
            job = self._require(job_id)
            if job["status"] == "running" and not job["publishing"]:
                self._flags[job_id].set()
                job["cancel_requested"] = True
                self._condition.notify_all()
            return {
                "job_id": job_id,
                "status": job["status"],
                "cancel_requested": job["cancel_requested"],
            }

    def latest(self, kind: str) -> dict[str, Any] | None:
        """The most recently started job of ``kind``, or None."""
        with self._condition:
            jobs = [job for job in self._jobs.values() if job["kind"] == kind]
            return self._snapshot(jobs[-1]) if jobs else None

    def close(self) -> None:
        """Ask every running job to stop; refuse new ones."""
        with self._condition:
            self._closed = True
            for flag in self._flags.values():
                flag.set()
            self._condition.notify_all()

    def _run(self, job_id, work, publish, flag: threading.Event) -> None:
        context = JobContext(job_id, flag, lambda entry: self._append(job_id, entry))
        try:
            value = work(context)
            with self._condition:
                if flag.is_set():
                    raise JobCancelledError
                self._jobs[job_id]["publishing"] = True
            result = publish(value)
        except JobCancelledError:
            self._finish(job_id, "cancelled")
        except EditorClientError as exc:
            self._finish(job_id, "failed", error=exc.public_message)
        except Exception:
            _LOGGER.exception("Unhandled SuperGLM editor job error.")
            self._finish(job_id, "failed", error=_INTERNAL_ERROR)
        else:
            self._finish(job_id, "done", result=result)

    def _append(self, job_id: str, entry: dict[str, Any]) -> None:
        with self._condition:
            self._jobs[job_id]["progress"].append(jsonable(entry))
            self._condition.notify_all()

    def _finish(self, job_id: str, status: str, *, result=None, error: str | None = None) -> None:
        with self._condition:
            job = self._jobs[job_id]
            job.update(
                status=status,
                result=jsonable(result),
                error=error,
                publishing=False,
                finished_at=time.time(),
            )
            self._flags.pop(job_id, None)
            evicted = [
                other_id
                for other_id, other in self._jobs.items()
                if other_id != job_id
                and other["kind"] == job["kind"]
                and other["status"] != "running"
            ]
            for other_id in evicted:
                del self._jobs[other_id]
            self._condition.notify_all()

    def _require(self, job_id: str) -> dict[str, Any]:
        job = self._jobs.get(str(job_id))
        if job is None:
            raise EditorKeyError(_UNKNOWN_JOB)
        return job

    @staticmethod
    def _snapshot(job: dict[str, Any]) -> dict[str, Any]:
        snapshot = {key: value for key, value in job.items() if key != "publishing"}
        snapshot["progress"] = list(job["progress"])
        return jsonable(snapshot)


__all__ = ["JobCancelledError", "JobContext", "JobRunner"]
```

3b. `src/superglm/editor/widget.py`:

Edit 1 (master line 39). Replace

```python
from superglm.editor.evidence import EvidenceCoordinator, EvidenceKey
from superglm.editor.io import jsonable
```

with

```python
from superglm.editor.evidence import EvidenceCoordinator, EvidenceKey
from superglm.editor.io import jsonable
from superglm.editor.jobs import JobContext, JobRunner
```

Edit 2 (text an earlier G task added). Replace

```python
        self._cv_run: CVRun | None = None
        self._final_fit: FinalFit | None = None
```

with

```python
        self._cv_run: CVRun | None = None
        self._final_fit: FinalFit | None = None
        # Run CV and Final fit: one background job of each kind at a time.
        # A starter captures a job's inputs under the widget lock and returns
        # its work (run off the lock) and its publish step (which re-takes it).
        self._jobs = JobRunner(name=f"editor-{id(self):x}")
        self._job_starters: dict[
            str,
            Callable[[], tuple[Callable[[JobContext], Any], Callable[[Any], dict[str, Any]]]],
        ] = {}
```

Edit 3 (master line 164). Replace

```python
        self._closed = True
        _LIVE_WIDGETS.discard(self)
        self._evidence.close()
```

with

```python
        self._closed = True
        _LIVE_WIDGETS.discard(self)
        self._jobs.close()
        self._evidence.close()
```

Edit 4 (master line 912). Replace

```python
    def _structural_transition(
        self,
```

with

```python
    def _job_start(self, kind: str) -> dict[str, Any]:
        """Capture a job's inputs under the lock, then start it off the lock."""
        starter = self._job_starters.get(kind)
        if starter is None:
            raise EditorValueError("Unknown job kind.")
        with self._lock:
            work, publish = starter()
        return self._jobs.status(self._jobs.start(kind, work, publish))

    def _job_status(self, job_id: str, *, wait: bool = False) -> dict[str, Any]:
        return self._jobs.status(job_id, wait=wait)

    def _job_cancel(self, job_id: str) -> dict[str, Any]:
        return self._jobs.cancel(job_id)

    def _structural_transition(
        self,
```

3c. `src/superglm/editor/server.py`:

Edit 1 (master line 244). Replace

```python
    @app.get("/profile_distribution/status/{job_id}")
    def profile_distribution_status(job_id: str, wait: bool = False) -> Response:
        return _guarded_json(lambda: widget._profile_distribution_status(job_id, wait=wait))
```

with

```python
    @app.get("/profile_distribution/status/{job_id}")
    def profile_distribution_status(job_id: str, wait: bool = False) -> Response:
        return _guarded_json(lambda: widget._profile_distribution_status(job_id, wait=wait))

    @app.post("/job_start")
    def job_start(payload: dict[str, Any] = Body(default_factory=dict)) -> Response:
        return _guarded_json(lambda: widget._job_start(str(_required(payload, "kind"))))

    @app.post("/job_status")
    def job_status(payload: dict[str, Any] = Body(default_factory=dict)) -> Response:
        return _guarded_json(
            lambda: widget._job_status(
                str(_required(payload, "job_id")), wait=payload.get("wait") is True
            )
        )

    @app.post("/job_cancel")
    def job_cancel(payload: dict[str, Any] = Body(default_factory=dict)) -> Response:
        return _guarded_json(lambda: widget._job_cancel(str(_required(payload, "job_id"))))
```

3d. `src/superglm/editor/cv.py`:

Edit 1 (text an earlier G task added). Replace

```python
    jobs = {kind: None for kind in JOB_KINDS}
    return cv_tab_payload(view, jobs=jobs, request_sequence=request_sequence)
```

with

```python
    jobs = {kind: widget._jobs.latest(kind) for kind in JOB_KINDS}
    return cv_tab_payload(view, jobs=jobs, request_sequence=request_sequence)
```

- [ ] **Step 4: Run tests, expect PASS.**

```bash
./.venv/bin/python -m pytest tests/test_editor_cv.py -q
./.venv/bin/python -m pytest tests/test_editor.py -q -n 8 -k "routes or report or profile"
./.venv/bin/ruff check src/superglm/editor tests/test_editor_cv.py tests/test_editor.py
./.venv/bin/ruff format --check src/superglm/editor tests/test_editor_cv.py tests/test_editor.py
```

Expected: 11 passed in `test_editor_cv.py`. The profile-job tests are unchanged and pass.

- [ ] **Step 5: Commit.**

```bash
git add src/superglm/editor/jobs.py src/superglm/editor/widget.py src/superglm/editor/server.py \
  src/superglm/editor/cv.py tests/test_editor_cv.py tests/test_editor.py
git commit -F - <<'EOF'
Editor: cancellable background jobs and the job routes

JobRunner runs one job per kind on a daemon thread. Its work runs off the widget lock
on captured state, a cancel flag is checked between steps, and publication re-takes
the lock and is refused to a cancelled job. A finished job evicts the earlier ones of
its kind. Routes: /job_start, /job_status, /job_cancel; refusals are fixed sentences.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```

---

### Task G4: Run CV and Final fit jobs, and the Final fit export

**Files:**
- Modify: `src/superglm/editor/cv.py`: the import block and constants, then append the Run CV and
  Final fit sections
- Modify: `src/superglm/editor/widget.py`: imports, `_EXPORT_*` (lines 61–68),
  `_normalise_export_format` (line 87), `_safe_export_filename` (line 100), the starters,
  `_state()` (line 184), `_report` (lines 558, 598), `_export_bytes` (line 627),
  `_final_fit_for_export` (before `_export_file`, line 678) and the two job methods
- Modify: `src/superglm/editor/reports.py`: `final_fit_report_payload` (lines 47–73) and
  `report_payload` (lines 76–99)
- Test: `tests/test_editor_cv.py` (append)

**Interfaces:**
- Consumes: G2's `check_cv_data`, `run_cv_reason`, `fold_log_curves`, `fold_term_items`,
  `model_with_edited_curves`; G3's runner. Also `cross_validate` (unchanged): its folds are
  consumed lazily (model_selection.py:407), so `StoredFolds.before_fold` runs between folds.
  Its callable-scorer path is used, so edits are re-applied before scoring.
  `fit_refit_model` and `resolve_refit_method` (refit.py:11, terms.py:250).
- Produces:
  - `cv.CVRunPlan`, `cv.capture_cv_run(session) -> CVRunPlan` (refuses with `run_cv_reason`), and
    `cv.run_cv(plan, context) -> CVRun`
  - `cv.FinalFitPlan`, `cv.capture_final_fit(session)` and `cv.run_final_fit(plan, context) -> FinalFit`
  - `widget._cv_job()` and `widget._final_fit_job()` (job kinds `"cv"` and `"final_fit"`),
    and `widget._final_fit_for_export()`
  - export format `"final"`, state `final_fit`, and the final report's `final_fit` section

Run CV in one paragraph. It takes the in-force model and its fit method (`resolve_refit_method(model, "auto")`),
the rows from `session.cv_check`, and the supplied folds. The scorers are the supplied
result's built-ins among deviance, Gini and NLL, or all three. `cross_validate` fits each
fold; the recorder's `score` puts the hand edits back on the fold model with
`model_with_edited_curves` (D5), on that fold's training rows. It then scores the edited
model with the built-in scorers, and records its fold curves on the editor's grid.
`cross_validate` does not pool a callable scorer's dict, so the pooled deviance and NLL are
summed from the same numerator and denominator parts it pools. With nothing edited, the
scores reproduce the supplied ones exactly (a test).

- [ ] **Step 1: Write the failing test.** Append to `tests/test_editor_cv.py`:

```python


# ── Run CV and Final fit ─────────────────────────────────────────


class _Context:
    """A job context that never cancels and records progress."""

    def __init__(self):
        self.entries = []

    def check(self):
        return None

    def progress(self, phase, **details):
        self.entries.append({"phase": phase, **details})


@pytest.fixture
def fit_rows(monkeypatch):
    """The row count of every SuperGLM.fit call, in order."""
    rows = []
    fit = SuperGLM.fit

    def counted(self, X, y, *args, **kwargs):
        rows.append(len(y))
        return fit(self, X, y, *args, **kwargs)

    monkeypatch.setattr(SuperGLM, "fit", counted)
    return rows


def test_run_cv_reproduces_the_supplied_scores_when_nothing_is_edited(cv_frame, cv_fit, fit_rows):
    from superglm.editor.cv import capture_cv_run, run_cv

    model, supplied = cv_fit
    session = EditorSession.from_model(model, cv=supplied, **_splits(cv_frame))
    context = _Context()

    run = run_cv(capture_cv_run(session), context)

    assert fit_rows == [len(train) for train, _test in supplied.fold_indices]
    assert [entry["fold"] for entry in context.entries if entry["phase"] == "fold"] == [1, 2, 3]
    # The same folds, structure and scorers in this thread: the same numbers.
    for name in ("deviance", "gini", "nll"):
        np.testing.assert_array_equal(run.result.fold_scores[name], supplied.fold_scores[name])
    assert run.result.pooled_scores == supplied.pooled_scores
    assert run.result.splitter == "KFold"


def test_run_cv_job_puts_the_hand_edits_back_on_every_fold(cv_frame, cv_fit, fit_rows):
    model, supplied = cv_fit
    session = EditorSession.from_model(model, cv=supplied, **_splits(cv_frame))
    session.select_levels("region", ["C"])
    session.shift("region", 0.1)
    widget = session.widget()
    try:
        started = _post_json(f"{widget.url}/job_start", {"kind": "cv"})
        finished = _post_json(
            f"{widget.url}/job_status", {"job_id": started["job_id"], "wait": True}
        )
        report = _post_json(f"{widget.url}/report", {"report": "cv"})
    finally:
        widget.close()

    assert finished["status"] == "done"
    assert [entry["fold"] for entry in finished["progress"] if entry["phase"] == "fold"] == [
        1,
        2,
        3,
    ]
    assert len(fit_rows) == 3
    supplied_result, current = report["results"]
    assert (current["label"], current["origin"], current["stale"]) == (
        "Current model",
        "run",
        False,
    )
    for edited_fold, supplied_fold in zip(current["folds"], supplied_result["folds"], strict=True):
        assert edited_fold["scores"]["deviance"] != supplied_fold["scores"]["deviance"]
    assert report["relativities"]["origin"] == "run"
    region = next(item for item in report["relativities"]["terms"] if item["name"] == "region")
    # Every fold carries the edited curve, so the folds agree on region exactly.
    for fold in region["folds"]:
        np.testing.assert_allclose(fold["values"], region["edited"], rtol=64 * _U)
    assert region["spread"] <= 64 * _U


def test_run_cv_is_refused_with_its_reason(cv_frame, cv_fit):
    from superglm.editor.cv import ROWS_MISMATCH

    X, y, w = cv_frame
    model, supplied = cv_fit
    waiting = EditorSession.from_model(model, cv=supplied, **_splits(cv_frame))
    waiting.stage_structural("collapse", "region", {"levels": ["B", "C"], "group_label": None})
    mismatched = EditorSession.from_model(
        model, cv=supplied, cv_data=(X.iloc[:399], y[:399], w[:399])
    )
    seen = {}
    for name, session in {"waiting": waiting, "mismatched": mismatched}.items():
        widget = session.widget()
        try:
            report = widget._report("cv")
            seen[name] = (report, _post_error(f"{widget.url}/job_start", {"kind": "cv"}))
        finally:
            widget.close()

    report, refused = seen["waiting"]
    assert report["run_cv"] == {
        "available": False,
        "reason": "Refit first: 1 change is waiting.",
        "note": None,
    }
    assert refused == (400, {"error": "Refit first: 1 change is waiting."})
    assert report["final_fit"]["note"] == "1 waiting change is not included."
    report, refused = seen["mismatched"]
    assert report["run_cv"]["reason"] == ROWS_MISMATCH.format(rows=399, expected=400)
    assert refused == (400, {"error": report["run_cv"]["reason"]})


def test_run_cv_cancelled_mid_run_publishes_nothing(cv_frame, cv_fit, fit_rows, monkeypatch):
    from superglm.editor.jobs import JobContext

    model, supplied = cv_fit
    widget = EditorSession.from_model(model, cv=supplied, **_splits(cv_frame)).widget()
    progress = JobContext.progress

    def cancel_before_the_second_fold(self, phase, **details):
        progress(self, phase, **details)
        if details.get("fold") == 2:
            widget._job_cancel(self.job_id)

    monkeypatch.setattr(JobContext, "progress", cancel_before_the_second_fold)
    try:
        started = _post_json(f"{widget.url}/job_start", {"kind": "cv"})
        finished = _post_json(
            f"{widget.url}/job_status", {"job_id": started["job_id"], "wait": True}
        )
        report = widget._report("cv")
    finally:
        widget.close()

    assert finished["status"] == "cancelled"
    assert [entry["fold"] for entry in finished["progress"]] == [1, 2]
    assert len(fit_rows) == 1
    assert [result["origin"] for result in report["results"]] == ["supplied"]
    assert report["relativities"]["origin"] == "supplied"
    assert report["jobs"]["cv"]["status"] == "cancelled"


def test_run_cv_result_is_dropped_when_the_model_changes_mid_run(cv_frame, cv_fit, monkeypatch):
    from superglm.editor.cv import SUPERSEDED
    from superglm.editor.jobs import JobContext

    model, supplied = cv_fit
    widget = EditorSession.from_model(model, cv=supplied, **_splits(cv_frame)).widget()
    progress = JobContext.progress

    def edit_during_the_first_fold(self, phase, **details):
        progress(self, phase, **details)
        if details.get("fold") == 1:
            widget._drag("region", [1], delta=0.1)

    monkeypatch.setattr(JobContext, "progress", edit_during_the_first_fold)
    try:
        started = _post_json(f"{widget.url}/job_start", {"kind": "cv"})
        finished = _post_json(
            f"{widget.url}/job_status", {"job_id": started["job_id"], "wait": True}
        )
        report = widget._report("cv")
    finally:
        widget.close()

    assert (finished["status"], finished["error"]) == ("failed", SUPERSEDED)
    assert [result["origin"] for result in report["results"]] == ["supplied"]


def test_final_fit_refits_train_and_validation_and_export_offers_it(cv_frame, cv_fit, fit_rows):
    from superglm.editor.cv import FINAL_NOT_RUN, FINAL_STALE
    from superglm.editor.errors import EditorValueError
    from superglm.editor.persistence import joblib_load_bytes

    model, supplied = cv_fit
    session = EditorSession.from_model(model, cv=supplied, **_splits(cv_frame))
    session.select_levels("region", ["C"])
    session.shift("region", 0.1)
    edited = session.terms["region"].edited_log_effect.copy()
    widget = session.widget()
    try:
        with pytest.raises(EditorValueError) as before:
            widget._export_bytes("final")
        started = _post_json(f"{widget.url}/job_start", {"kind": "final_fit"})
        finished = _post_json(
            f"{widget.url}/job_status", {"job_id": started["job_id"], "wait": True}
        )
        state = _get_json(f"{widget.url}/state")
        section = widget._report("final")["final_fit"]
        exported = widget._export_bytes("final")
        widget._drag("region", [0], delta=0.05)
        with pytest.raises(EditorValueError) as stale:
            widget._export_bytes("final")
    finally:
        widget.close()

    assert before.value.public_message == FINAL_NOT_RUN
    assert finished["status"] == "done"
    # Train and validation rows, one fit; the test split stays held out (D6).
    assert finished["result"]["n_rows"] == 500
    assert fit_rows == [500]
    assert state["final_fit"] == {"available": True, "stale": False}
    assert (section["n_rows"], section["splits"], section["carried"]) == (
        500,
        ["train", "validation"],
        ["region"],
    )
    assert exported.filename == "superglm_final_model.joblib"
    final_model = joblib_load_bytes(exported.data)
    probe = pd.DataFrame({"age": [40.0] * 3, "power": [0.0] * 3, "region": ["A", "B", "C"]})
    log_mu = np.log(final_model.predict(probe))
    # The hand edit is put back as set: the final model's region relativities are the edited ones.
    np.testing.assert_allclose(
        log_mu - log_mu[0],
        edited - edited[0],
        rtol=0.0,
        atol=64 * _U * max(1.0, np.max(np.abs(log_mu))),
    )
    assert stale.value.public_message == FINAL_STALE


def test_run_cv_takes_the_in_force_fit_method_and_the_supplied_scorers(cv_frame, cv_fit):
    from superglm.editor.cv import capture_cv_run

    X, y, w = cv_frame
    model, supplied = cv_fit
    reml = _model().fit_reml(X.iloc[:400], y[:400], sample_weight=w[:400])
    scores = supplied.fold_scores
    deviance_only = dataclasses.replace(supplied, fold_scores=scores.drop(columns=["gini", "nll"]))
    no_builtins = dataclasses.replace(
        supplied, fold_scores=scores.drop(columns=["deviance", "gini", "nll"])
    )

    def plan(fitted, result):
        return capture_cv_run(EditorSession.from_model(fitted, cv=result, **_splits(cv_frame)))

    assert plan(model, supplied).fit_mode == "fit"
    assert plan(reml, supplied).fit_mode == "fit_reml"
    assert plan(model, deviance_only).scoring == ("deviance",)
    assert plan(model, no_builtins).scoring == ("deviance", "gini", "nll")
```

These cover Review Focus 5's cancel case: cancelling mid-run gives `cancelled`, exactly one fit,
and nothing published. They also cover D5 (every fold carries the edit, so the folds agree on
`region` to 64u), D6 (500 = train ∪ validation rows, one fit, the test split held out), D7
(the reason, as a disabled reason and as an HTTP 400) and publish-if-current. Fits are counted,
never timed. The cancel and model-change tests make it deterministic by doing their work
inside the job's own progress callback at a fold boundary.

- [ ] **Step 2: Run it, expect FAIL.**

```bash
./.venv/bin/python -m pytest tests/test_editor_cv.py -k "run_cv or final_fit" -q
```

Expected after G3: the direct tests fail with `ImportError: cannot import name 'capture_cv_run'
from 'superglm.editor.cv'`, or with `'FINAL_NOT_RUN'` or `'SUPERSEDED'`. The job tests get
HTTP 400 `{"error": "Unknown job kind."}`. On 155832e8 every one of them fails earlier, with
`TypeError: ... unexpected keyword argument 'cv'`.

- [ ] **Step 3: Implement.**

3a. `src/superglm/editor/cv.py`: the imports and constants.

Edit 1 (text an earlier G task added). Replace

```python
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from superglm.editor.evaluation import EvaluationDataset, training_export_dataset
from superglm.model_selection import CrossValidationResult, _data_fingerprint
from superglm.plotting.comparison import _feature_beta
```

with

```python
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from superglm._frame import as_eager_frame
from superglm.editor.carry import model_with_edited_curves
from superglm.editor.errors import EditorValueError
from superglm.editor.evaluation import EvaluationDataset, training_export_dataset
from superglm.editor.refit import fit_refit_model
from superglm.editor.terms import resolve_refit_method
from superglm.model_selection import (
    _BUILTIN_SCORERS,
    _POOLED_PARTS,
    CrossValidationResult,
    _data_fingerprint,
    cross_validate,
)
from superglm.plotting.comparison import _feature_beta
```

Edit 2 (text an earlier G task added). Replace

```python
TRAIN_ONLY = "No validation data was supplied, so Final fit uses the train rows only."
```

with

```python
TRAIN_ONLY = "No validation data was supplied, so Final fit uses the train rows only."
SUPERSEDED = "The model changed while the job ran, so its result was not kept. Run it again."
MIXED_FRAMES = "Train and validation data must both be pandas or both be Polars data frames."
FINAL_NOT_RUN = "Run Final fit on all rows, on the Cross-validation tab, first."
FINAL_STALE = "The model changed after the final fit. Run Final fit on all rows again."
```

Edit 3 (text an earlier G task added). Replace

```python
_FOLD_COLUMNS = (
```

with

```python
_DEFAULT_SCORING = tuple(name for name, _label, _lower in _METRICS)
_FOLD_COLUMNS = (
```

3b. Append to the end of `src/superglm/editor/cv.py`:

```python


# ── Run CV ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class CVRunPlan:
    """Everything Run CV reads, captured under the widget lock."""

    model: Any
    model_revision: int
    rows: EvaluationDataset
    folds: tuple[tuple[NDArray[np.intp], NDArray[np.intp]], ...]
    terms: dict[str, EditableTerm]
    edited: dict[str, EditableTerm]
    fit_mode: str
    scoring: tuple[str, ...]
    splitter: str | None
    n_points: int

    def is_current(self, session) -> bool:
        return session.model_revision == self.model_revision and session.model is self.model


def capture_cv_run(session) -> CVRunPlan:
    """Capture Run CV's inputs; refuse with the tab's reason while it is disabled."""
    reason = run_cv_reason(session)
    if reason is not None:
        raise EditorValueError(reason)
    cv = session.cv
    terms = {name: term.copy() for name, term in session.terms.items()}
    supplied = tuple(name for name in _DEFAULT_SCORING if name in cv.fold_scores.columns)
    return CVRunPlan(
        model=session.model,
        model_revision=session.model_revision,
        rows=session.cv_check.rows,
        folds=tuple(
            (np.asarray(train, dtype=np.intp), np.asarray(test, dtype=np.intp))
            for train, test in cv.fold_indices
        ),
        terms=terms,
        edited={name: terms[name] for name in session.edited_terms()},
        fit_mode=resolve_refit_method(session.model, "auto"),
        scoring=supplied or _DEFAULT_SCORING,
        splitter=cv.splitter,
        n_points=session.n_points,
    )


def run_cv(plan: CVRunPlan, context) -> CVRun:
    """Replay the stored folds on the in-force structure with the hand edits put back."""
    recorder = _FoldRecorder(plan, context)
    result = cross_validate(
        plan.model,
        plan.rows.X,
        plan.rows.y,
        cv=StoredFolds(plan.folds, before_fold=recorder.before_fold),
        sample_weight=plan.rows.sample_weight,
        offset=plan.rows.offset,
        fit_mode=plan.fit_mode,
        scoring=recorder.score,
    )
    context.check()
    context.progress("curves")
    return CVRun(
        result=replace(result, pooled_scores=recorder.pooled_scores(), splitter=plan.splitter),
        terms=fold_term_items(plan.terms, recorder.curves),
        model_revision=plan.model_revision,
        carried=tuple(sorted(plan.edited)),
    )


class _FoldRecorder:
    """Run CV's per-fold hooks: progress and cancel between folds, edits before scoring.

    ``cross_validate`` fits each fold and hands the fitted model to
    :meth:`score`, which puts the hand edits back (D5) and computes the
    built-in scores on that edited model. A callable scorer's dict is not
    pooled by ``cross_validate``, so the pooled deviance and NLL are summed
    here from the same numerator and denominator parts it pools.
    """

    def __init__(self, plan: CVRunPlan, context) -> None:
        self._plan = plan
        self._context = context
        self._fold = -1
        self._frame = as_eager_frame(plan.rows.X)
        self._y = np.asarray(plan.rows.y, dtype=np.float64)
        self._totals = {name: [0.0, 0.0] for name in plan.scoring if name in _POOLED_PARTS}
        self.curves: dict[int, dict[str, NDArray[np.float64]]] = {}

    def before_fold(self, index: int) -> None:
        self._context.progress("fold", fold=index + 1, n_folds=len(self._plan.folds))
        self._context.check()
        self._fold = index

    def score(self, model, X, y, *, sample_weight=None, offset=None) -> dict[str, float]:
        if self._plan.edited:
            train = self._plan.folds[self._fold][0]
            model = model_with_edited_curves(
                model,
                self._plan.edited,
                self._frame.take_rows(train),
                self._y[train],
                _take(self._plan.rows.sample_weight, train),
                _take(self._plan.rows.offset, train),
                n_points=self._plan.n_points,
            )
        scores: dict[str, float] = {}
        parts: dict[str, tuple[float, float]] = {}
        for name in self._plan.scoring:
            pooled = _POOLED_PARTS.get(name)
            if pooled is None:
                scorer = _BUILTIN_SCORERS[name]
                scores[name] = float(
                    scorer(model, X, y, sample_weight=sample_weight, offset=offset)
                )
                continue
            numerator, denominator = pooled(model, X, y, sample_weight=sample_weight, offset=offset)
            parts[name] = (numerator, denominator)
            scores[name] = numerator / denominator
        # Only a fold that scored completely joins the pooled totals.
        for name, (numerator, denominator) in parts.items():
            self._totals[name][0] += numerator
            self._totals[name][1] += denominator
        self.curves[self._fold] = fold_log_curves(model, self._plan.terms)
        return scores

    def pooled_scores(self) -> dict[str, float]:
        return {
            name: numerator / denominator
            for name, (numerator, denominator) in self._totals.items()
            if denominator > 0.0
        }


def _take(values, rows: NDArray[np.intp]):
    return None if values is None else np.asarray(values, dtype=np.float64)[rows]


# ── Final fit ────────────────────────────────────────────────────


@dataclass(frozen=True)
class FinalFitPlan:
    """Everything Final fit reads, captured under the widget lock."""

    model: Any
    model_revision: int
    datasets: tuple[EvaluationDataset, ...]
    edited: dict[str, EditableTerm]
    pending: int
    n_points: int

    def is_current(self, session) -> bool:
        return session.model_revision == self.model_revision and session.model is self.model


def capture_final_fit(session) -> FinalFitPlan:
    """Capture Final fit's inputs; refuse when there are no training rows."""
    datasets = final_fit_datasets(session)
    if not datasets:
        raise EditorValueError(NO_FINAL_ROWS)
    return FinalFitPlan(
        model=session.model,
        model_revision=session.model_revision,
        datasets=datasets,
        edited={name: session.terms[name].copy() for name in session.edited_terms()},
        pending=len(session.pending),
        n_points=session.n_points,
    )


def run_final_fit(plan: FinalFitPlan, context) -> FinalFit:
    """Refit the in-force structure on train and validation rows, then put the edits back."""
    X, y, sample_weight, offset = _union_rows(plan.datasets)
    context.progress("fitting", n_rows=int(y.size))
    context.check()
    model = plan.model.clone_unfitted()
    fit_refit_model(
        plan.model,
        model,
        method="auto",
        X=X,
        y=y,
        sample_weight=sample_weight,
        offset=offset,
    )
    context.check()
    if plan.edited:
        context.progress("carrying", terms=sorted(plan.edited))
        model = model_with_edited_curves(
            model, plan.edited, X, y, sample_weight, offset, n_points=plan.n_points
        )
        context.check()
    return FinalFit(
        model=model,
        model_revision=plan.model_revision,
        n_rows=int(y.size),
        splits=tuple(dataset.name for dataset in plan.datasets),
        carried=tuple(sorted(plan.edited)),
        pending=plan.pending,
    )


def _union_rows(datasets: Sequence[EvaluationDataset]):
    """The splits' rows stacked in order: one frame, response, weight and offset."""
    frames = [as_eager_frame(dataset.X) for dataset in datasets]
    backends = {frame.backend for frame in frames}
    if len(backends) > 1:
        raise EditorValueError(MIXED_FRAMES)
    if len(frames) == 1:
        X = frames[0].native
    elif backends == {"pandas"}:
        X = pd.concat([frame.native for frame in frames], ignore_index=True)
    else:
        import polars as pl

        X = pl.concat([frame.native for frame in frames], how="vertical_relaxed")
    y = np.concatenate([np.asarray(dataset.y, dtype=np.float64) for dataset in datasets])
    return X, y, _stacked(datasets, "sample_weight", 1.0), _stacked(datasets, "offset", 0.0)


def _stacked(datasets: Sequence[EvaluationDataset], name: str, fill: float):
    """One column across the splits, ``fill`` where a split lacks it; None if all do."""
    columns = [getattr(dataset, name) for dataset in datasets]
    if all(column is None for column in columns):
        return None
    return np.concatenate(
        [
            np.full(dataset.n_obs, fill) if column is None else np.asarray(column, dtype=float)
            for dataset, column in zip(datasets, columns, strict=True)
        ]
    )
```

3c. `src/superglm/editor/widget.py`:

Edit 1 (text an earlier G task added). Replace

```python
from superglm.editor.cv import CVRun, FinalFit
```

with

```python
from superglm.editor.cv import (
    FINAL_NOT_RUN,
    FINAL_STALE,
    SUPERSEDED,
    CVRun,
    FinalFit,
    capture_cv_run,
    capture_final_fit,
    run_cv,
    run_final_fit,
)
```

Edit 2 (master line 61). Replace

```python
_EXPORT_MEDIA_TYPES = {
    "joblib": "application/octet-stream",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}
_EXPORT_DEFAULT_FILENAMES = {
    "joblib": "superglm_edited_model.joblib",
    "xlsx": "superglm_rating_tables.xlsx",
}
```

with

```python
_EXPORT_MEDIA_TYPES = {
    "joblib": "application/octet-stream",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "final": "application/octet-stream",
}
_EXPORT_DEFAULT_FILENAMES = {
    "joblib": "superglm_edited_model.joblib",
    "xlsx": "superglm_rating_tables.xlsx",
    "final": "superglm_final_model.joblib",
}
_EXPORT_SUFFIXES = {"joblib": ".joblib", "xlsx": ".xlsx", "final": ".joblib"}
```

Edit 3 (master line 87). Replace

```python
    if normalized in {"xlsx", "excel"}:
        return "xlsx"
```

with

```python
    if normalized in {"xlsx", "excel"}:
        return "xlsx"
    if normalized in {"final", "final_fit"}:
        return "final"
```

Edit 4 (master line 100). Replace

```python
    expected = f".{format}"
```

with

```python
    expected = _EXPORT_SUFFIXES[format]
```

Edit 5 (text an earlier G task added). Replace

```python
        self._job_starters: dict[
            str,
            Callable[[], tuple[Callable[[JobContext], Any], Callable[[Any], dict[str, Any]]]],
        ] = {}
```

with

```python
        self._job_starters: dict[
            str,
            Callable[[], tuple[Callable[[JobContext], Any], Callable[[Any], dict[str, Any]]]],
        ] = {"cv": self._cv_job, "final_fit": self._final_fit_job}
```

Edit 6 (master line 184). Replace

```python
                "in_force_is_original": self.session.model is self.session.reference_model,
```

with

```python
                "in_force_is_original": self.session.model is self.session.reference_model,
                # Export offers the Final fit model while it is current (D6).
                "final_fit": {
                    "available": self._final_fit is not None,
                    "stale": self._final_fit is not None
                    and self._final_fit.model_revision != self.session.model_revision,
                },
```

Edit 7 (master line 558). Replace

```python
            datasets = tuple(evaluation_datasets(self.session))
            reference_model = getattr(self.session, "reference_model", self.session.model)
```

with

```python
            datasets = tuple(evaluation_datasets(self.session))
            final_fit = self._final_fit
            reference_model = getattr(self.session, "reference_model", self.session.model)
```

Edit 8 (master line 598). Replace

```python
            request_sequence=request_sequence,
            model_override=summary_model,
        )
```

with

```python
            request_sequence=request_sequence,
            model_override=summary_model,
            final_fit=final_fit,
        )
```

Edit 9 (master line 627). Replace

```python
        validation_scope: str | None = None
        if canonical_format == "joblib":
            model, revision = self._current_model_for_evidence()
```

with

```python
        validation_scope: str | None = None
        if canonical_format in {"joblib", "final"}:
            model, revision = (
                self._final_fit_for_export()
                if canonical_format == "final"
                else self._current_model_for_evidence()
            )
```

Edit 10 (master line 678). Replace

```python
    def _export_file(
        self,
```

with

```python
    def _final_fit_for_export(self):
        """The Final fit model and its revision; refused before a run or once stale."""
        with self._lock:
            final = self._final_fit
            if final is None:
                raise EditorValueError(FINAL_NOT_RUN)
            if final.model_revision != self.session.model_revision:
                raise EditorValueError(FINAL_STALE)
            return final.model, final.model_revision

    def _export_file(
        self,
```

Edit 11 (text an earlier G task added). Replace

```python
    def _job_status(self, job_id: str, *, wait: bool = False) -> dict[str, Any]:
```

with

```python
    def _cv_job(self):
        """Run CV on the in-force structure (D5, D7); the caller holds the lock."""
        plan = capture_cv_run(self.session)

        def publish(run: CVRun) -> dict[str, Any]:
            with self._lock:
                if not plan.is_current(self.session):
                    raise EditorValueError(SUPERSEDED)
                self._cv_run = run
            return {"model_revision": plan.model_revision, "n_folds": len(plan.folds)}

        return (lambda context: run_cv(plan, context)), publish

    def _final_fit_job(self):
        """Final fit on train and validation rows (D6); the caller holds the lock."""
        plan = capture_final_fit(self.session)

        def publish(final: FinalFit) -> dict[str, Any]:
            with self._lock:
                if not plan.is_current(self.session):
                    raise EditorValueError(SUPERSEDED)
                self._final_fit = final
            return {"model_revision": plan.model_revision, "n_rows": final.n_rows}

        return (lambda context: run_final_fit(plan, context)), publish

    def _job_status(self, job_id: str, *, wait: bool = False) -> dict[str, Any]:
```

3d. `src/superglm/editor/reports.py`:

Edit 1 (text an earlier G task added). Replace

```python
from superglm.editor.cv import cv_report_payload
```

with

```python
from superglm.editor.cv import FINAL_NOT_RUN, cv_report_payload
```

Edit 2 (master line 52). Replace

```python
    request_sequence: int | None = None,
    model_override=None,
) -> dict[str, Any]:
    """Return the current in-force model summary and split metrics."""
```

with

```python
    request_sequence: int | None = None,
    model_override=None,
    final_fit=None,
) -> dict[str, Any]:
    """Return the current in-force model summary, split metrics and any Final fit."""
```

Edit 3 (master line 71). Replace

```python
        "summary": summary_payload(widget, "in_force", model_override=model_override),
        "can_run_cv": False,
    }
```

with

```python
        "summary": summary_payload(widget, "in_force", model_override=model_override),
        "can_run_cv": False,
        "final_fit": _final_fit_section(widget, final_fit, model_revision=revision),
    }


def _final_fit_section(widget, final_fit, *, model_revision: int) -> dict[str, Any]:
    """The Final fit on train and validation rows, once one has run (D6)."""
    if final_fit is None:
        return {"available": False, "note": FINAL_NOT_RUN}
    summary = summary_payload(widget, "in_force", model_override=final_fit.model)
    return {
        "available": True,
        "stale": final_fit.model_revision != model_revision,
        "n_rows": final_fit.n_rows,
        "splits": list(final_fit.splits),
        "carried": list(final_fit.carried),
        "pending": final_fit.pending,
        "summary": summary.get("compact"),
    }
```

Edit 4 (master line 82). Replace

```python
    request_sequence: int | None = None,
    model_override=None,
) -> dict[str, Any]:
    """Dispatch a named report for the local editor app."""
```

with

```python
    request_sequence: int | None = None,
    model_override=None,
    final_fit=None,
) -> dict[str, Any]:
    """Dispatch a named report for the local editor app."""
```

Edit 5 (master line 91). Replace

```python
            request_sequence=request_sequence,
            model_override=model_override,
        )
    return validation_report_payload(
```

with

```python
            request_sequence=request_sequence,
            model_override=model_override,
            final_fit=final_fit,
        )
    return validation_report_payload(
```

- [ ] **Step 4: Run tests, expect PASS.**

```bash
./.venv/bin/python -m pytest tests/test_editor_cv.py -q
./.venv/bin/python -m pytest tests/test_editor.py -q -n 8 -k "export or report or save or download"
./.venv/bin/ruff check src/superglm/editor tests/test_editor_cv.py
./.venv/bin/ruff format --check src/superglm/editor tests/test_editor_cv.py
```

Expected: 18 passed in `test_editor_cv.py`. The existing joblib and xlsx export tests pass,
including the filename-suffix tests.

- [ ] **Step 5: Commit.**

```bash
git add src/superglm/editor/cv.py src/superglm/editor/widget.py src/superglm/editor/reports.py \
  tests/test_editor_cv.py
git commit -F - <<'EOF'
Editor: Run CV and Final fit jobs, and the Final fit export

Run CV replays the supplied folds through cross_validate on the in-force structure and
fit method, puts the hand edits back on every fold before scoring (D5), and waits
while changes wait for Refit (D7). Final fit refits on train and validation rows, puts
the edits back and is offered by Export as the Final fit model (D6). Both run as
cancellable jobs and keep their result only if the model did not change meanwhile.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```

---

### Task G5: The Cross-validation tab in the app

**Files:**
- Create: `src/superglm/editor/app/views/cv_tab.js` and `src/superglm/editor/app/styles/cv.css`
- Modify: `app/api/contracts.js` (lines 3, 140, 218), `app/api/client.js` (line 82),
  `app/reports.js` (lines 1, 13, 22, 161), `app/main.js` (lines 44, 197, 539, 767, 1098, 1105),
  `app/index.html` (lines 27, 56, 569), `app/views/export_dialog.js` (lines 10, 42, 78, 92, 137)
  and `app/views/help_content.js` (lines 209, 213)
- Test: create `tests/editor_frontend/cv_tab.test.js` and `tests/editor/test_editor_cv_browser.py`;
  append to `tests/editor_frontend/client.test.js`, `tests/editor_frontend/export_dialog.test.js` and
  `tests/test_editor_cv.py`; modify `tests/test_editor.py` (lines 6726, 6762, 6774) and
  `tests/editor/test_editor_workspace_browser.py` (line 1190, the pinned tab list)

**Interfaces:**
- Consumes: the G2–G4 payloads and routes; `escapeHTML` and `fmt` (format.js); the popover
  system (`data-popover-title`/`-body`, bound on `document`); `selectVisibleEvidencePanels`,
  which already treats any non-editor view as needing the report (selectors.js:83).
- Produces:
  - `AppView` `"cv"` and `#cvTab`
  - `editorClient.jobStart(kind)`, `.jobStatus(jobId, wait=false)` and `.jobCancel(jobId)`
  - from `views/cv_tab.js`: `createCVTab({frame, client, onJobSettled, pause})` returning
    `{render, start, cancel, destroy}`, plus `cvSourceLine`, `cvTabMarkup`, `toolbarMarkup`,
    `performanceMarkup`, `foldTableMarkup`, `relativitiesMarkup`, `termListMarkup`,
    `levelChartMarkup`, `curveChartMarkup`, `filterTerms`, `foldEnvelope`, `niceTicks`,
    `newerJob`, `jobLine` and `JOB_KINDS`
  - `renderReport(payload, nodes, cvTab)`, and `bindExportDialog({..., finalFitAvailable})`

The layout follows board 6. The header row holds the waiting and stale chips and the two
buttons, then a muted line explaining why a button is disabled, then each job's progress line
with Cancel. The three performance cards give mean ± sd · pooled, one dot per fold, and one row
per run ("As supplied", then "Current model" after a Run CV). Then the latest run's fold table,
then "Relativities across folds": a searchable term list (least stable first, spread and min r)
beside the chosen term's chart. A level chart has fold dots, a range whisker and an all-rows-fit
bar per level in model order, over exposure; a curve chart has fold lines, a min–max envelope and
the all-rows fit. An edited curve is drawn dashed. While a job runs, only the header row is
redrawn; typing in the search redraws only the list and chart. Running jobs are picked up again
from `payload.jobs` after a reload.

- [ ] **Step 1: Write the failing tests.**

1a. Create `tests/editor_frontend/cv_tab.test.js`:

```js
// @ts-nocheck

import assert from "node:assert/strict";
import test from "node:test";

import {
  createCVTab,
  curveChartMarkup,
  cvSourceLine,
  cvTabMarkup,
  filterTerms,
  foldEnvelope,
  jobLine,
  levelChartMarkup,
  newerJob,
  niceTicks
} from "../../src/superglm/editor/app/views/cv_tab.js";

function foldRow(index, deviance, gini) {
  return {
    fold: index,
    n_train: 36000,
    n_test: 9000,
    fit_time_s: 1.25,
    converged: true,
    n_iter: 7,
    effective_df: 31.4,
    scores: { deviance, gini }
  };
}

const brand = Object.freeze({
  name: "VehBrand",
  kind: "levels",
  x: null,
  levels: ["B1", "B2", "B10"],
  weights: [3, 2, 1],
  folds: [
    { label: "Fold 1", values: [1.1, 0.9, 1.0] },
    { label: "Fold 2", values: [1.2, 0.8, 1.05] }
  ],
  fit: [1.15, 0.85, 1.02],
  edited: null,
  spread: 0.05,
  min_correlation: 0.9
});

const age = Object.freeze({
  name: "DrivAge",
  kind: "continuous",
  x: [18, 30, 50, 85],
  levels: null,
  weights: [1, 4, 3, 1],
  folds: [
    { label: "Fold 1", values: [1.4, 1.0, 0.9, 1.1] },
    { label: "Fold 2", values: [1.5, 1.05, 0.88, 1.0] }
  ],
  fit: [1.45, 1.02, 0.89, 1.05],
  edited: [1.3, 1.02, 0.89, 1.05],
  spread: 0.01,
  min_correlation: 0.99
});

function cvPayload(overrides = {}) {
  return {
    report: "cv",
    title: "Cross-validation",
    note: "",
    model_revision: 3,
    header: { supplied: true, n_folds: 2, splitter: "KFold", n_rows: 45000 },
    pending: 0,
    run_cv: { available: true, reason: null, note: null },
    final_fit: { available: true, reason: null, note: null, done: false, stale: false, n_rows: null },
    metrics: [
      { name: "deviance", label: "Mean deviance", lower_is_better: true },
      { name: "gini", label: "Gini", lower_is_better: false }
    ],
    results: [{
      label: "As supplied",
      origin: "supplied",
      model_revision: null,
      stale: false,
      folds: [foldRow(0, 0.31, 0.2), foldRow(1, 0.3, 0.22)],
      mean: { deviance: 0.305, gini: 0.21 },
      std: { deviance: 0.005, gini: 0.01 },
      pooled: { deviance: 0.3049 }
    }],
    relativities: { available: true, origin: "supplied", stale: false, note: null, terms: [brand, age] },
    jobs: { cv: null, final_fit: null },
    ...overrides
  };
}

const idle = () => ({
  term: "",
  query: "",
  jobs: { cv: null, final_fit: null },
  errors: { cv: "", final_fit: "" }
});

function count(markup, pattern) {
  return (markup.match(pattern) ?? []).length;
}

test("the source line names the folds, splitter, rows and edit() call", () => {
  assert.equal(
    cvSourceLine(cvPayload()),
    "2 folds · KFold · 45,000 rows · supplied with edit(model, cv=result)"
  );
  assert.equal(
    cvSourceLine(cvPayload({ header: { supplied: false, n_folds: 0, splitter: null, n_rows: null } })),
    "No cross-validation result supplied"
  );
});

test("Run CV is disabled with its reason while changes wait", () => {
  const markup = cvTabMarkup(cvPayload({
    pending: 2,
    run_cv: { available: false, reason: "Refit first: 2 changes are waiting.", note: null }
  }), idle());

  assert.match(markup, /data-cv-start="cv"\s+disabled/);
  assert.doesNotMatch(markup, /data-cv-start="final_fit"\s+disabled/);
  assert.match(markup, /<p class="cv-reason">Refit first: 2 changes are waiting\.<\/p>/);
  assert.match(markup, /2 changes waiting for refit/);
});

test("performance cards show mean ± sd, pooled, and one dot per fold for each run", () => {
  const current = { ...cvPayload().results[0], label: "Current model", origin: "run", pooled: {} };
  const markup = cvTabMarkup(cvPayload({ results: [cvPayload().results[0], current] }), idle());
  const deviance = markup.slice(markup.indexOf("Mean deviance"), markup.indexOf(">Gini"));

  assert.match(deviance, /<strong>0\.3050<\/strong>\s*<span class="cv-card-spread">± 0\.0050 · pooled 0\.3049<\/span>/);
  assert.equal(count(deviance, /<circle /g), 4);
  assert.equal(count(deviance, /class="cv-card-row"/g), 2);
});

test("the fold table lists the latest run's folds and a mean row", () => {
  const markup = cvTabMarkup(cvPayload(), idle());
  const table = markup.slice(markup.indexOf('aria-label="Fold scores"'), markup.indexOf("</table>"));

  assert.equal(count(table, /<tr>/g), 3);
  assert.match(table, /Fold 2<\/td>\s*<td>36,000<\/td><td>9,000<\/td>/);
  assert.match(table, /<th>Deviance<\/th><th>Gini<\/th>/);
  assert.match(table, /Mean ± sd/);
});

test("the term list filters by name, ignoring case, in the server's order", () => {
  assert.deepEqual(filterTerms([brand, age], "").map((term) => term.name), ["VehBrand", "DrivAge"]);
  assert.deepEqual(filterTerms([brand, age], "  drIV ").map((term) => term.name), ["DrivAge"]);
  const markup = cvTabMarkup(cvPayload(), { ...idle(), query: "brand" });
  assert.match(markup, /Veh<mark>Brand<\/mark>/);
  assert.equal(count(markup, /class="cv-term"/g), 1);
  assert.match(cvTabMarkup(cvPayload(), { ...idle(), query: "zzz" }), /No terms match\./);
});

test("a level chart keeps the model's level order with a whisker and dots per fold", () => {
  const markup = levelChartMarkup(brand);

  assert.ok(markup.indexOf(">B1<") < markup.indexOf(">B2<"));
  assert.ok(markup.indexOf(">B2<") < markup.indexOf(">B10<"));
  assert.equal(count(markup, /class="cv-range"/g), 3);
  assert.equal(count(markup, /class="cv-fold-dot"/g), 6);
  assert.equal(count(markup, /class="cv-fit"/g), 3);
  assert.equal(count(markup, /class="exposure"/g), 3);
  assert.equal(count(markup, /class="cv-edited"/g), 0);
});

test("a curve chart draws each fold, the envelope, the fit and the edited curve", () => {
  const markup = curveChartMarkup(age);

  assert.equal(count(markup, /class="cv-fold-line"/g), 2);
  assert.equal(count(markup, /class="cv-envelope"/g), 1);
  assert.equal(count(markup, /class="cv-fit-line"/g), 1);
  assert.equal(count(markup, /class="cv-edited-line"/g), 1);
  assert.doesNotMatch(markup, /NaN/);
});

test("the fold envelope and the axis ticks", () => {
  assert.deepEqual(foldEnvelope(brand), { lo: [1.1, 0.8, 1.0], hi: [1.2, 0.9, 1.05] });
  assert.deepEqual(niceTicks(0.82, 1.21, 5), [0.9, 1, 1.1, 1.2]);
  assert.deepEqual(niceTicks(18, 85, 6), [20, 40, 60, 80]);
});

test("a job line says what the job is doing and how it ended", () => {
  const running = { job_id: "cv-1", kind: "cv", status: "running", progress: [], result: null };
  assert.equal(jobLine("cv", running), "Run CV: starting…");
  assert.equal(
    jobLine("cv", { ...running, progress: [{ phase: "fold", fold: 2, n_folds: 5 }] }),
    "Run CV: fold 2 of 5…"
  );
  assert.equal(jobLine("cv", { ...running, cancel_requested: true }), "Run CV: cancelling after this step…");
  assert.equal(jobLine("cv", { ...running, status: "cancelled" }), "Run CV was cancelled. Nothing was kept.");
  assert.equal(
    jobLine("final_fit", { ...running, kind: "final_fit", status: "done", result: { n_rows: 50000 } }),
    "Final fit finished on 50,000 rows. Export offers it as Final fit model."
  );
  assert.equal(
    jobLine("cv", { ...running, status: "failed", error: "Refit first: 1 change is waiting." }),
    "Run CV failed: Refit first: 1 change is waiting."
  );
});

test("the newer status wins: a later job, then a finished one, then more progress", () => {
  const first = { job_id: "cv-1", status: "running", progress: [{}] };
  const later = { job_id: "cv-3", status: "running", progress: [] };
  assert.equal(newerJob(first, later), later);
  assert.equal(newerJob(later, first), later);
  assert.equal(newerJob(first, { ...first, status: "done" }).status, "done");
  assert.equal(newerJob({ ...first, status: "done" }, first).status, "done");
  assert.equal(newerJob(first, { ...first, progress: [] }), first);
  assert.equal(newerJob(null, first), first);
});

function fakeFrame() {
  return {
    innerHTML: "",
    querySelector: () => null,
    addEventListener() {},
    removeEventListener() {}
  };
}

test("a started job is polled until it settles, then reported once", async () => {
  const statuses = [
    { job_id: "cv-1", kind: "cv", status: "running", progress: [{ phase: "fold", fold: 1, n_folds: 2 }], result: null },
    { job_id: "cv-1", kind: "cv", status: "done", progress: [], result: { n_folds: 2 } }
  ];
  const calls = [];
  const settled = [];
  const frame = fakeFrame();
  const tab = createCVTab({
    frame,
    client: {
      async jobStart(kind) {
        calls.push(["start", kind]);
        return { job_id: "cv-1", kind, status: "running", progress: [], result: null };
      },
      async jobStatus(jobId) {
        calls.push(["status", jobId]);
        return statuses.shift();
      },
      async jobCancel() {
        throw new Error("not called");
      }
    },
    onJobSettled: (kind, job) => settled.push([kind, job.status]),
    pause: async () => {}
  });
  tab.render(cvPayload());

  await tab.start("cv");

  assert.deepEqual(calls, [["start", "cv"], ["status", "cv-1"], ["status", "cv-1"]]);
  assert.deepEqual(settled, [["cv", "done"]]);
  assert.match(frame.innerHTML, /Run CV finished\./);
});

test("cancel posts the running job's id and shows it is stopping", async () => {
  const cancelled = [];
  let release;
  const frame = fakeFrame();
  const running = { job_id: "cv-2", kind: "cv", status: "running", progress: [], result: null };
  const tab = createCVTab({
    frame,
    client: {
      async jobStart() {
        return running;
      },
      async jobStatus() {
        await new Promise((resolve) => { release = resolve; });
        return { ...running, status: "cancelled" };
      },
      async jobCancel(jobId) {
        cancelled.push(jobId);
        return { job_id: jobId, status: "running", cancel_requested: true };
      }
    },
    onJobSettled: () => {},
    pause: async () => {}
  });
  tab.render(cvPayload());
  const started = tab.start("cv");
  await new Promise((resolve) => setImmediate(resolve));

  await tab.cancel("cv");
  const stopping = frame.innerHTML;
  release();
  await started;

  assert.deepEqual(cancelled, ["cv-2"]);
  assert.match(stopping, /Run CV: cancelling after this step…/);
  assert.doesNotMatch(stopping, /data-cv-cancel="cv"/);
  assert.match(frame.innerHTML, /Run CV was cancelled\. Nothing was kept\./);
});

test("a refused start shows the server's sentence", async () => {
  const frame = fakeFrame();
  const tab = createCVTab({
    frame,
    client: {
      async jobStart() {
        throw new Error("Refit first: 1 change is waiting.");
      },
      async jobStatus() {},
      async jobCancel() {}
    },
    onJobSettled: () => {},
    pause: async () => {}
  });
  tab.render(cvPayload());

  await tab.start("cv");

  assert.match(frame.innerHTML, /data-status="failed">Refit first: 1 change is waiting\.<\/p>/);
});
```

1b. Append to `tests/editor_frontend/client.test.js`:

```js

test("client job calls post to the job routes", async () => {
  /** @type {Array<{url:string, body:unknown}>} */
  const requests = [];
  const client = createEditorClient({
    fetchImpl: async (url, options) => {
      requests.push({ url: String(url), body: JSON.parse(String(options?.body)) });
      return new Response(JSON.stringify({ job_id: "cv-1", status: "running" }), { status: 200 });
    }
  });

  await client.jobStart("cv");
  await client.jobStatus("cv-1", true);
  await client.jobCancel("cv-1");

  assert.deepEqual(requests, [
    { url: "/job_start", body: { kind: "cv" } },
    { url: "/job_status", body: { job_id: "cv-1", wait: true } },
    { url: "/job_cancel", body: { job_id: "cv-1" } }
  ]);
});
```

1c. Append to `tests/editor_frontend/export_dialog.test.js`:

```js

test("the Final fit model option follows availability and downloads the final export", async () => {
  const fixture = exportFixture();
  const final = new FakeElement("final");
  fixture.nodes.formatInputs.push(final);
  let available = false;
  const binding = bindExportDialog({ ...fixture.context, finalFitAvailable: () => available });

  await fixture.action.emit("click");
  assert.equal(final.disabled, true);

  available = true;
  fixture.dialog.open = false;
  await fixture.action.emit("click");
  assert.equal(final.disabled, false);
  fixture.joblib.checked = false;
  final.checked = true;
  await final.emit("change");
  assert.equal(fixture.filename.value, "superglm_final_model.joblib");
  await fixture.download.emit("click");
  assert.deepEqual(fixture.blobPaths, [
    "/download_export?format=final&filename=superglm_final_model.joblib",
  ]);

  // Once the model changes, the stale final fit is no longer offered.
  available = false;
  fixture.dialog.open = false;
  await fixture.action.emit("click");
  assert.deepEqual([final.disabled, final.checked, fixture.joblib.checked], [true, false, true]);
  assert.equal(fixture.filename.value, "superglm_edited_model.joblib");
  binding.destroy();
});
```

1d. Append to `tests/test_editor_cv.py`:

```python


# ── The tab in the app ───────────────────────────────────────────


def test_app_has_a_cross_validation_tab_and_a_final_fit_export():
    from pathlib import Path

    import superglm.editor

    root = Path(superglm.editor.__file__).parent / "app"
    html = (root / "index.html").read_text()

    assert html.index('id="validationTab"') < html.index('id="cvTab"') < html.index('id="finalTab"')
    assert 'data-view="cv"' in html
    assert 'id="exportFinalFit" type="radio" name="exportFormat" value="final"' in html
    assert '<link rel="stylesheet" href="/assets/styles/cv.css">' in html
```

1e. Create `tests/editor/test_editor_cv_browser.py`:

```python
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold

from superglm import Categorical, Spline, SuperGLM, cross_validate
from superglm.editor import EditorSession

pytest.importorskip("playwright.sync_api")
pytestmark = pytest.mark.browser


def _model() -> SuperGLM:
    return SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        features={"age": Spline(n_knots=6), "region": Categorical(base="first")},
    )


@pytest.fixture
def cv_widget():
    rng = np.random.default_rng(20261003)
    n = 400
    X = pd.DataFrame({"age": rng.uniform(18.0, 80.0, n), "region": rng.choice(["C", "A", "B"], n)})
    eta = -0.5 + 0.2 * np.sin(X["age"].to_numpy() / 12.0) + 0.2 * (X["region"] == "B")
    y = rng.poisson(np.exp(eta)).astype(np.float64)
    supplied = cross_validate(
        _model(),
        X.iloc[:300],
        y[:300],
        cv=KFold(3, shuffle=True, random_state=0),
        scoring=("deviance", "gini", "nll"),
        return_estimators=True,
    )
    session = EditorSession.from_model(
        _model().fit(X.iloc[:300], y[:300]),
        train_data=(X.iloc[:300], y[:300]),
        validation_data=(X.iloc[300:], y[300:]),
        cv=supplied,
    )
    widget = session.widget()
    try:
        yield widget
    finally:
        widget.close()


def _open_cv_tab(chromium_browser, widget):
    page = chromium_browser.new_page(viewport={"width": 1280, "height": 900})
    page.goto(widget.app_url, wait_until="domcontentloaded")
    page.locator("#chart path.edited").first.wait_for()
    page.locator("#cvTab").click()
    page.locator("#reportFrame .cv-card").first.wait_for()
    return page


def test_cv_tab_renders_a_supplied_result(cv_widget, chromium_browser):
    page = _open_cv_tab(chromium_browser, cv_widget)
    try:
        assert page.locator("#reportTitle").text_content() == "Cross-validation"
        assert page.locator("#reportStatus").text_content() == (
            "3 folds · KFold · 300 rows · supplied with edit(model, cv=result)"
        )
        assert page.locator("#reportFrame .cv-card").count() == 3
        assert page.locator("#reportFrame .cv-fold-table tbody tr").count() == 4
        assert sorted(page.locator("#reportFrame .cv-term-name").all_text_contents()) == [
            "age",
            "region",
        ]
        page.locator('#reportFrame [data-cv-term="region"]').click()
        assert page.locator("#reportFrame .cv-level title").all_text_contents() == ["A", "B", "C"]
        page.locator("#cvTermSearch").fill("reg")
        assert page.locator("#reportFrame .cv-term").count() == 1
        page.locator("#cvTermSearch").press("Escape")
        assert page.locator("#reportFrame .cv-term").count() == 2
    finally:
        page.close()


def test_run_cv_from_the_tab_adds_the_current_model(cv_widget, chromium_browser):
    page = _open_cv_tab(chromium_browser, cv_widget)
    try:
        page.locator('#reportFrame [data-cv-start="cv"]').click()
        page.locator('#reportFrame .cv-card-row[data-origin="run"]').first.wait_for()
        assert page.locator('#reportFrame .cv-card-row[data-origin="run"]').count() == 3
        assert page.locator('#reportFrame .cv-job[data-cv-job="cv"]').text_content() == (
            "Run CV finished. Its folds are shown beside the supplied ones."
        )
    finally:
        page.close()
```

1f. `tests/test_editor.py` asset checks and the pinned tab list in
`tests/editor/test_editor_workspace_browser.py`:

Edit 1 (master line 6726). Replace

```python
        assert '<link rel="stylesheet" href="/assets/styles/dialogs.css">' in shell
```

with

```python
        assert '<link rel="stylesheet" href="/assets/styles/dialogs.css">' in shell
        assert '<link rel="stylesheet" href="/assets/styles/cv.css">' in shell
```

Edit 2 (master line 6762). Replace

```python
            "views/inspector.js",
            "views/help_drawer.js",
        ]:
```

with

```python
            "views/inspector.js",
            "views/help_drawer.js",
            "views/cv_tab.js",
        ]:
```

Edit 3 (master line 6774). Replace

```python
            "styles/panels.css",
            "styles/dialogs.css",
        ]:
```

with

```python
            "styles/panels.css",
            "styles/dialogs.css",
            "styles/cv.css",
        ]:
```

Edit 1 (master line 1190). Replace

```python
        assert tabs.get_by_role("tab").all_inner_texts() == [
            "Editor",
            "Validation",
            "Final Fit",
        ]
```

with

```python
        assert tabs.get_by_role("tab").all_inner_texts() == [
            "Editor",
            "Validation",
            "Cross-validation",
            "Final Fit",
        ]
```

- [ ] **Step 2: Run them, expect FAIL.**

```bash
npm ci   # once; this worktree has no node_modules
node --test tests/editor_frontend/cv_tab.test.js tests/editor_frontend/client.test.js tests/editor_frontend/export_dialog.test.js
./.venv/bin/python -m pytest tests/test_editor_cv.py -k app_has -q
./.venv/bin/python -m pytest tests/editor/test_editor_cv_browser.py -m browser --run-browser -q
```

Expected, measured against 155832e8's sources:
- `cv_tab.test.js` fails to load with `ERR_MODULE_NOT_FOUND ... views/cv_tab.js`.
- `client job calls post to the job routes` fails because `client.jobStart is not a function`.
- The Final fit export test fails at `assert.equal(final.disabled, true)`.
- `test_app_has_...` fails with `ValueError: substring not found` (there is no `id="cvTab"`).
- The browser tests time out waiting for `#cvTab`.
- `test_widget_serves_editor_app_assets` fails on the `cv.css` link.
- The pinned tab list fails once the tab is added.

- [ ] **Step 3: Implement.**

3a. `app/api/contracts.js`:

Edit 1 (master line 3). Replace

```js
/** @typedef {'editor'|'validation'|'final'} AppView */
```

with

```js
/** @typedef {'editor'|'validation'|'cv'|'final'} AppView */
```

Edit 2 (master line 140). Replace

```js
 * @property {boolean} in_force_is_original
 */
```

with

```js
 * @property {boolean} in_force_is_original
 * @property {{available:boolean, stale:boolean}} [final_fit] whether Export can offer the Final fit model
 */
```

Edit 3 (master line 218). Replace

```js
/** @typedef {{panel:EvidencePanel, revision:number, sequence:number}} EvidenceToken */
```

with

```js
/** @typedef {'cv'|'final_fit'} JobKind */
/**
 * A background job's status (/job_start, /job_status). ``progress`` entries
 * carry a ``phase`` ("fold", "curves", "fitting", "carrying") and its details.
 * @typedef {Object} JobStatus
 * @property {string} job_id
 * @property {JobKind} kind
 * @property {'running'|'done'|'failed'|'cancelled'} status
 * @property {Array<Record<string, unknown>>} progress
 * @property {Record<string, unknown>|null} result
 * @property {string|null} [error]
 * @property {boolean} [cancel_requested]
 */
/**
 * @typedef {Object} CVFoldRow
 * @property {number} fold
 * @property {number} n_train
 * @property {number} n_test
 * @property {number|null} fit_time_s
 * @property {boolean} converged
 * @property {number|null} effective_df
 * @property {Record<string, number|null>} scores
 */
/**
 * One cross-validation run: the supplied result, or Run CV on the current model.
 * @typedef {Object} CVResultPayload
 * @property {string} label
 * @property {'supplied'|'run'} origin
 * @property {number|null} model_revision
 * @property {boolean} stale
 * @property {CVFoldRow[]} folds
 * @property {Record<string, number|null>} mean
 * @property {Record<string, number|null>} std
 * @property {Record<string, number>} pooled
 */
/**
 * One term's relativities by fold, each curve re-centred on its
 * exposure-weighted mean log; levels in the model's order.
 * @typedef {Object} CVTermItem
 * @property {string} name
 * @property {'levels'|'continuous'} kind
 * @property {number[]|null} x
 * @property {string[]|null} levels
 * @property {number[]} weights
 * @property {Array<{label:string, values:number[]}>} folds
 * @property {number[]} fit
 * @property {number[]|null} edited
 * @property {number} spread
 * @property {number} min_correlation
 */
/**
 * The /report payload for the Cross-validation tab.
 * @typedef {Object} CVReportPayload
 * @property {'cv'} report
 * @property {string} title
 * @property {string} note
 * @property {number} model_revision
 * @property {{supplied:boolean, n_folds:number, splitter:string|null, n_rows:number|null}} header
 * @property {number} pending
 * @property {{available:boolean, reason:string|null, note:string|null}} run_cv
 * @property {{available:boolean, reason:string|null, note:string|null, done:boolean, stale:boolean, n_rows:number|null}} final_fit
 * @property {Array<{name:string, label:string, lower_is_better:boolean}>} metrics
 * @property {CVResultPayload[]} results
 * @property {{available:boolean, origin:string|null, stale:boolean, note:string|null, terms:CVTermItem[]}} relativities
 * @property {Record<JobKind, JobStatus|null>} jobs
 */
/** @typedef {{panel:EvidencePanel, revision:number, sequence:number}} EvidenceToken */
```

3b. `app/api/client.js`. These are methods on the client object, so the module's exports,
which `client.test.js` pins, do not change. Add them beside any methods earlier tasks added
(`stage`, `refitPending`, …):

Edit 1 (master line 82). Replace

```js
  /** @returns {Promise<unknown>} */
  function getState() {
    return requestJSON("/state");
  }

  return { requestJSON, postJSON, requestBlob, getState };
```

with

```js
  /** @returns {Promise<unknown>} */
  function getState() {
    return requestJSON("/state");
  }

  /** @param {string} kind "cv" or "final_fit" */
  function jobStart(kind) {
    return postJSON("/job_start", { kind });
  }

  /** @param {string} jobId @param {boolean} [wait] wait up to 30 s for the job to stop running */
  function jobStatus(jobId, wait = false) {
    return postJSON("/job_status", { job_id: jobId, wait });
  }

  /** @param {string} jobId */
  function jobCancel(jobId) {
    return postJSON("/job_cancel", { job_id: jobId });
  }

  return { requestJSON, postJSON, requestBlob, getState, jobStart, jobStatus, jobCancel };
```

3c. Create `src/superglm/editor/app/views/cv_tab.js`:

```js
// @ts-check

import { escapeHTML, fmt } from "../format.js";

/** @typedef {import('../api/contracts.js').CVReportPayload} CVReportPayload */
/** @typedef {import('../api/contracts.js').CVResultPayload} CVResultPayload */
/** @typedef {import('../api/contracts.js').CVTermItem} CVTermItem */
/** @typedef {import('../api/contracts.js').JobKind} JobKind */
/** @typedef {import('../api/contracts.js').JobStatus} JobStatus */
/**
 * @typedef {object} JobClient
 * @property {(kind:JobKind)=>Promise<unknown>} jobStart
 * @property {(jobId:string, wait?:boolean)=>Promise<unknown>} jobStatus
 * @property {(jobId:string)=>Promise<unknown>} jobCancel
 */
/**
 * @typedef {object} CVTabState
 * @property {string} term the term whose chart is shown
 * @property {string} query the term search
 * @property {Record<JobKind, JobStatus|null>} jobs the newest status of each job
 * @property {Record<JobKind, string>} errors a refused job request's message
 */

/** @type {readonly JobKind[]} */
export const JOB_KINDS = Object.freeze(["cv", "final_fit"]);
const POLL_MS = 250;
const CHART = Object.freeze({ width: 760, height: 320, left: 56, right: 18, top: 30, bottom: 66 });
const JOB_NAMES = Object.freeze({ cv: "Run CV", final_fit: "Final fit" });
const SHORT_METRICS = Object.freeze({ deviance: "Deviance", gini: "Gini", nll: "NLL" });
const RUN_CV_HELP = "Refit the current structure on each stored fold, put your hand edits back "
  + "exactly as set, and score the held-out rows.";
const FINAL_FIT_HELP = "Refit the current structure on train and validation rows and put your "
  + "hand edits back. The test split stays held out. Export offers the result as Final fit model.";
const ICONS = Object.freeze({
  cv: '<svg class="toolbar-icon" viewBox="0 0 24 24" aria-hidden="true">'
    + '<path d="m12 3 9 5-9 5-9-5z"></path><path d="m3 13 9 5 9-5"></path></svg>',
  final_fit: '<svg class="toolbar-icon" viewBox="0 0 24 24" aria-hidden="true">'
    + '<path d="M20 11a8 8 0 1 0-2.3 5.7"></path><path d="M20 4v7h-7"></path></svg>'
});

/** @param {unknown} value @returns {value is number} */
function isNumber(value) {
  return typeof value === "number" && Number.isFinite(value);
}

/** @param {unknown} value */
function metricText(value) {
  return isNumber(value) ? value.toFixed(4) : "--";
}

/** @param {unknown} value */
function countText(value) {
  return isNumber(value) ? value.toLocaleString("en-US") : "--";
}

/** @param {number} value */
function px(value) {
  return value.toFixed(1);
}

/** @param {number} index */
function foldColour(index) {
  return `var(--trace-${index % 10})`;
}

/** @param {string} value @returns {JobKind} */
function jobKind(value) {
  return value === "final_fit" ? "final_fit" : "cv";
}

/**
 * The header's source line: folds, splitter, rows and where the result came from.
 * @param {CVReportPayload} payload
 */
export function cvSourceLine(payload) {
  const { header } = payload;
  if (!header.supplied) return "No cross-validation result supplied";
  const parts = [`${header.n_folds} folds`];
  if (header.splitter) parts.push(header.splitter);
  if (header.n_rows !== null) parts.push(`${countText(header.n_rows)} rows`);
  parts.push("supplied with edit(model, cv=result)");
  return parts.join(" · ");
}

/**
 * Terms whose name contains ``query``, ignoring case, in the server's
 * order: least stable first.
 * @param {CVTermItem[]} terms @param {string} query
 */
export function filterTerms(terms, query) {
  const needle = query.trim().toLowerCase();
  return needle ? terms.filter((term) => term.name.toLowerCase().includes(needle)) : terms;
}

/**
 * The lowest and highest fold value at each point.
 * @param {CVTermItem} term
 */
export function foldEnvelope(term) {
  /** @param {number} index */
  const column = (index) => term.folds.map((fold) => fold.values[index]).filter(isNumber);
  return {
    lo: term.fit.map((_value, index) => Math.min(...column(index))),
    hi: term.fit.map((_value, index) => Math.max(...column(index)))
  };
}

/**
 * Round tick values covering [lo, hi], about ``target`` of them.
 * @param {number} lo @param {number} hi @param {number} [target]
 * @returns {number[]}
 */
export function niceTicks(lo, hi, target = 5) {
  if (!(hi > lo)) return [lo];
  const raw = (hi - lo) / target;
  const power = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((multiple) => multiple * power)
    .find((candidate) => candidate >= raw) ?? 10 * power;
  const first = Math.ceil(lo / step) * step;
  const count = Math.floor((hi - first) / step + 1e-9) + 1;
  return Array.from({ length: count }, (_unused, index) =>
    Number((first + index * step).toPrecision(12)));
}

/**
 * The newer of two statuses for one job kind: a later job wins, and for the
 * same job a finished status or one with more progress.
 * @param {JobStatus|null} known @param {JobStatus|null} reported
 * @returns {JobStatus|null}
 */
export function newerJob(known, reported) {
  if (!reported) return known;
  if (!known) return reported;
  const order = (/** @type {JobStatus} */ job) => Number(job.job_id.split("-").pop());
  if (order(reported) !== order(known)) return order(reported) > order(known) ? reported : known;
  if (known.status !== "running") return known;
  if (reported.status !== "running") return reported;
  return reported.progress.length >= known.progress.length ? reported : known;
}

/**
 * One line on what a job is doing, or how it ended.
 * @param {JobKind} kind @param {JobStatus|null} job
 */
export function jobLine(kind, job) {
  if (!job) return "";
  const name = JOB_NAMES[kind];
  if (job.status === "cancelled") return `${name} was cancelled. Nothing was kept.`;
  if (job.status === "failed") return `${name} failed: ${job.error || "internal editor error"}`;
  if (job.status === "done") {
    return kind === "final_fit"
      ? `Final fit finished on ${countText(job.result?.n_rows)} rows. Export offers it as Final fit model.`
      : "Run CV finished. Its folds are shown beside the supplied ones.";
  }
  if (job.cancel_requested) return `${name}: cancelling after this step…`;
  const last = job.progress[job.progress.length - 1];
  if (last?.phase === "fold") return `${name}: fold ${last.fold} of ${last.n_folds}…`;
  if (last?.phase === "fitting") return `${name}: fitting ${countText(last.n_rows)} rows…`;
  if (last?.phase === "carrying") return `${name}: putting the hand edits back…`;
  if (last?.phase === "curves") return `${name}: reading the fold curves…`;
  return `${name}: starting…`;
}

/** @param {string} text @param {string} tone */
function chip(text, tone) {
  return `<span class="cv-chip" data-tone="${tone}">${escapeHTML(text)}</span>`;
}

/**
 * @param {JobKind} kind @param {string} label @param {boolean} enabled
 * @param {string} help @param {boolean} primary
 */
function actionButton(kind, label, enabled, help, primary) {
  return `<button type="button" class="cv-action${primary ? " is-primary" : ""}"
    data-cv-start="${kind}" ${enabled ? "" : "disabled"}
    data-popover-title="${escapeHTML(label)}" data-popover-body="${escapeHTML(help)}">
    ${ICONS[kind]}<span>${escapeHTML(label)}</span></button>`;
}

/** @param {JobKind} kind @param {CVTabState} state */
function jobRow(kind, state) {
  const job = state.jobs[kind];
  const error = state.errors[kind];
  if (error) return `<p class="cv-job" data-cv-job="${kind}" data-status="failed">${escapeHTML(error)}</p>`;
  if (!job) return "";
  const cancel = job.status === "running" && !job.cancel_requested
    ? ` <button type="button" class="cv-cancel" data-cv-cancel="${kind}">Cancel</button>`
    : "";
  return `<p class="cv-job" data-cv-job="${kind}" data-status="${job.status}">`
    + `${escapeHTML(jobLine(kind, job))}${cancel}</p>`;
}

/**
 * The header row: what is waiting or stale, the two job buttons, why a
 * button is disabled, and each job's progress.
 * @param {CVReportPayload} payload @param {CVTabState} state
 */
export function toolbarMarkup(payload, state) {
  const chips = [];
  if (payload.pending) {
    chips.push(chip(`${payload.pending} ${payload.pending === 1 ? "change" : "changes"} waiting for refit`, "waiting"));
  }
  if (payload.results.some((result) => result.stale)) {
    chips.push(chip("The model has changed since these folds were fitted", "stale"));
  }
  if (payload.final_fit.done && payload.final_fit.stale) {
    chips.push(chip("The final fit is from an earlier version of the model", "stale"));
  }
  if (payload.run_cv.note) chips.push(chip(payload.run_cv.note, "note"));
  const runEnabled = payload.run_cv.available && state.jobs.cv?.status !== "running";
  const finalEnabled = payload.final_fit.available && state.jobs.final_fit?.status !== "running";
  const reasons = [
    payload.header.supplied ? payload.run_cv.reason : null,
    payload.final_fit.reason,
    payload.final_fit.note
  ].filter((reason) => reason);
  return `
    <section class="cv-toolbar" data-cv-toolbar>
      <div class="cv-chips">${chips.join("")}</div>
      <div class="cv-actions">
        ${actionButton("final_fit", "Final fit on all rows", finalEnabled, payload.final_fit.reason || FINAL_FIT_HELP, false)}
        ${actionButton("cv", "Run CV on current model", runEnabled, payload.run_cv.reason || RUN_CV_HELP, true)}
      </div>
      ${reasons.map((reason) => `<p class="cv-reason">${escapeHTML(reason)}</p>`).join("")}
      <div class="cv-jobs" aria-live="polite">${JOB_KINDS.map((kind) => jobRow(kind, state)).join("")}</div>
    </section>`;
}

/** @param {string} title @param {string} hint */
function sectionHead(title, hint) {
  return `<div class="cv-section-head"><h3>${escapeHTML(title)}</h3>`
    + `<span class="cv-hint">${escapeHTML(hint)}</span></div>`;
}

/**
 * @param {{name:string, label:string, lower_is_better:boolean}} metric
 * @param {CVResultPayload[]} results
 */
function metricCard(metric, results) {
  const values = results.flatMap((result) => result.folds.map((fold) => fold.scores[metric.name]))
    .filter(isNumber);
  const lo = Math.min(...values);
  const span = Math.max(...values) - lo || 1;
  const x = (/** @type {number} */ value) => 10 + ((value - lo) / span) * 200;
  const rows = results.map((result) => {
    const mean = result.mean[metric.name];
    const pooled = result.pooled[metric.name];
    const dots = result.folds.map((fold, index) => {
      const value = fold.scores[metric.name];
      return isNumber(value)
        ? `<circle cx="${px(x(value))}" cy="12" r="4.5" style="fill: ${foldColour(index)}">`
          + `<title>Fold ${index + 1}: ${metricText(value)}</title></circle>`
        : "";
    }).join("");
    const meanTick = isNumber(mean) ? `<path class="cv-mean" d="M${px(x(mean))},3 V21"></path>` : "";
    const pooledText = isNumber(pooled) ? ` · pooled ${metricText(pooled)}` : "";
    return `<div class="cv-card-row" data-origin="${result.origin}">
      <div class="cv-card-value"><span class="cv-card-label">${escapeHTML(result.label)}</span>
        <strong>${metricText(mean)}</strong>
        <span class="cv-card-spread">± ${metricText(result.std[metric.name])}${pooledText}</span></div>
      <svg class="cv-strip" viewBox="0 0 220 24" role="img"
        aria-label="${escapeHTML(`${metric.label} by fold, ${result.label}`)}">
        <path class="cv-strip-axis" d="M10,12 H210"></path>${meanTick}${dots}</svg>
    </div>`;
  }).join("");
  return `<div class="cv-card"><div class="cv-card-title">${escapeHTML(metric.label)}
    <span>· ${metric.lower_is_better ? "lower" : "higher"} is better</span></div>${rows}</div>`;
}

/**
 * One card per metric: mean ± sd and pooled for each run, one dot per fold.
 * @param {CVReportPayload} payload
 */
export function performanceMarkup(payload) {
  const head = sectionHead("Performance across folds", "each dot is a fold; the bar is the mean");
  if (!payload.results.length) {
    return `<section class="report-section">${head}<div class="report-note">${escapeHTML(payload.note)}</div></section>`;
  }
  const cards = payload.metrics.map((metric) => metricCard(metric, payload.results)).join("");
  return `<section class="report-section cv-performance">${head}<div class="cv-cards">${cards}</div></section>`;
}

/**
 * The latest run's folds: rows, scores, EDF, fit time and convergence.
 * @param {CVReportPayload} payload
 */
export function foldTableMarkup(payload) {
  const result = payload.results[payload.results.length - 1];
  if (!result) return "";
  const names = payload.metrics.map((metric) => metric.name);
  const label = (/** @type {string} */ name) =>
    SHORT_METRICS[/** @type {keyof typeof SHORT_METRICS} */ (name)] ?? name;
  const rows = result.folds.map((fold, index) => `<tr>
      <td><span class="cv-fold-swatch" style="background: ${foldColour(index)}"></span>Fold ${fold.fold + 1}</td>
      <td>${countText(fold.n_train)}</td><td>${countText(fold.n_test)}</td>
      ${names.map((name) => `<td>${metricText(fold.scores[name])}</td>`).join("")}
      <td>${isNumber(fold.effective_df) ? fold.effective_df.toFixed(1) : "--"}</td>
      <td>${isNumber(fold.fit_time_s) ? `${fold.fit_time_s.toFixed(2)} s` : "--"}</td>
      <td data-converged="${fold.converged}">${fold.converged ? "yes" : "no"}</td>
    </tr>`).join("");
  const summary = names.map((name) =>
    `<td>${metricText(result.mean[name])} ± ${metricText(result.std[name])}</td>`).join("");
  return `<section class="report-section">
    ${sectionHead("Folds", result.label)}
    <table class="report-table cv-fold-table" aria-label="Fold scores">
      <thead><tr><th>Fold</th><th>Train rows</th><th>Test rows</th>
        ${names.map((name) => `<th>${escapeHTML(label(name))}</th>`).join("")}
        <th>EDF</th><th>Fit time</th><th>Converged</th></tr></thead>
      <tbody>${rows}<tr class="cv-mean-row"><td>Mean ± sd</td><td></td><td></td>${summary}<td></td><td></td><td></td></tr></tbody>
    </table></section>`;
}

/** @param {string} name @param {string} query */
function highlighted(name, query) {
  const needle = query.trim().toLowerCase();
  const at = needle ? name.toLowerCase().indexOf(needle) : -1;
  if (at < 0) return escapeHTML(name);
  const end = at + needle.length;
  return `${escapeHTML(name.slice(0, at))}<mark>${escapeHTML(name.slice(at, end))}</mark>${escapeHTML(name.slice(end))}`;
}

/**
 * @param {CVTermItem[]} terms @param {string} current @param {string} query
 */
export function termListMarkup(terms, current, query) {
  if (!terms.length) return '<div class="report-note">No terms match.</div>';
  return terms.map((term) => `<button type="button" class="cv-term" data-cv-term="${escapeHTML(term.name)}"
      aria-current="${term.name === current}">
      <span class="cv-term-name">${highlighted(term.name, query)}</span>
      <span>${term.spread.toFixed(3)}</span><span>${term.min_correlation.toFixed(2)}</span></button>`).join("");
}

/**
 * @param {CVTermItem} term
 * @param {(value:number)=>number} y
 * @param {number[]} ticks
 */
function frameMarkup(term, y, ticks) {
  const { width, left, right } = CHART;
  return ticks.map((tick) => `<line class="cv-grid" x1="${left}" x2="${width - right}"
      y1="${px(y(tick))}" y2="${px(y(tick))}"></line>
    <text class="cv-tick" x="${left - 6}" y="${px(y(tick) + 4)}" text-anchor="end">${escapeHTML(fmt(tick))}</text>`).join("")
    + `<line class="zero" x1="${left}" x2="${width - right}" y1="${px(y(1))}" y2="${px(y(1))}"></line>`
    + `<text class="cv-axis-title" x="${left}" y="16">${escapeHTML(term.name)} · relativity</text>`;
}

/** @param {CVTermItem} term */
function legendMarkup(term) {
  const items = [
    ["cv-legend-fold", "folds"],
    ["cv-legend-fit", "all-rows fit"],
    ...(term.edited ? [["cv-legend-edited", "edited"]] : []),
    ["cv-legend-range", term.kind === "levels" ? "fold range" : "fold min–max"],
    ["cv-legend-exposure", "exposure"]
  ];
  const { width, right } = CHART;
  return `<g class="cv-legend" transform="translate(${width - right - 110 * items.length}, 8)">`
    + items.map(([kind, text], index) => `<g transform="translate(${index * 110}, 0)">
      <rect class="${kind}" x="0" y="3" width="14" height="6"></rect>
      <text x="19" y="10">${escapeHTML(text)}</text></g>`).join("")
    + "</g>";
}

/** @param {CVTermItem} term */
function yScale(term) {
  const { height, top, bottom } = CHART;
  const values = [...term.folds.flatMap((fold) => fold.values), ...term.fit, ...(term.edited ?? [])]
    .filter(isNumber);
  const lo = Math.min(...values, 1);
  const hi = Math.max(...values, 1);
  const pad = (hi - lo) * 0.1 || 0.05;
  const domain = [lo - pad, hi + pad];
  const y = (/** @type {number} */ value) =>
    height - bottom - ((value - domain[0]) / (domain[1] - domain[0])) * (height - top - bottom);
  return { y, ticks: niceTicks(domain[0], domain[1], 5) };
}

/**
 * A level term: fold dots, the fold range as a whisker and the all-rows fit
 * as a bar at each level, in the model's level order, over exposure.
 * @param {CVTermItem} term
 */
export function levelChartMarkup(term) {
  const { width, height, left, right, top, bottom } = CHART;
  const levels = term.levels ?? [];
  const band = (width - left - right) / Math.max(levels.length, 1);
  const x = (/** @type {number} */ index) => left + (index + 0.5) * band;
  const { y, ticks } = yScale(term);
  const envelope = foldEnvelope(term);
  const base = height - bottom;
  const maxWeight = Math.max(...term.weights, 0) || 1;
  const barWidth = Math.min(band * 0.7, 28);
  const step = Math.min(6, band / (term.folds.length + 2));
  const bars = term.weights.map((weight, index) => {
    const tall = (weight / maxWeight) * (base - top) / 3;
    return `<rect class="exposure" x="${px(x(index) - barWidth / 2)}" y="${px(base - tall)}"
      width="${px(barWidth)}" height="${px(tall)}"></rect>`;
  }).join("");
  const whiskers = levels.map((_level, index) => `<line class="cv-range" x1="${px(x(index))}"
      x2="${px(x(index))}" y1="${px(y(envelope.lo[index]))}" y2="${px(y(envelope.hi[index]))}"></line>`).join("");
  const dots = term.folds.flatMap((fold, foldIndex) => fold.values.map((value, index) =>
    `<circle class="cv-fold-dot" cx="${px(x(index) + (foldIndex - (term.folds.length - 1) / 2) * step)}"
      cy="${px(y(value))}" r="3.6" style="fill: ${foldColour(foldIndex)}">
      <title>${escapeHTML(`${fold.label} · ${levels[index]}: ${fmt(value)}`)}</title></circle>`)).join("");
  /** @param {number[]} values @param {string} kind */
  const levelTicks = (values, kind) => values.map((value, index) => `<line class="${kind}"
      x1="${px(x(index) - 16)}" x2="${px(x(index) + 16)}" y1="${px(y(value))}" y2="${px(y(value))}"></line>`).join("");
  const rotate = levels.length > 10;
  const labels = levels.map((level, index) => `<text class="cv-level" x="${px(x(index))}" y="${base + 16}"
      text-anchor="${rotate ? "end" : "middle"}"${rotate ? ` transform="rotate(-40 ${px(x(index))} ${base + 16})"` : ""}>
      ${escapeHTML(level.length > 14 ? `${level.slice(0, 13)}…` : level)}<title>${escapeHTML(level)}</title></text>`).join("");
  return `<svg class="cv-chart-svg" viewBox="0 0 ${width} ${height}" role="img"
      aria-label="${escapeHTML(`${term.name} relativities by fold`)}">
    ${frameMarkup(term, y, ticks)}${bars}${whiskers}${dots}${levelTicks(term.fit, "cv-fit")}
    ${term.edited ? levelTicks(term.edited, "cv-edited") : ""}${labels}${legendMarkup(term)}</svg>`;
}

/** @param {Array<[number, number]>} points */
function pathData(points) {
  return points.map(([px0, py0], index) => `${index ? "L" : "M"}${px(px0)},${px(py0)}`).join("");
}

/**
 * A numeric term: one line per fold, the folds' min–max envelope and the
 * all-rows fit, over exposure.
 * @param {CVTermItem} term
 */
export function curveChartMarkup(term) {
  const { width, height, left, right, top, bottom } = CHART;
  const xs = term.x ?? [];
  const first = xs[0] ?? 0;
  const last = xs[xs.length - 1] ?? 1;
  const x = (/** @type {number} */ value) =>
    left + ((value - first) / (last - first || 1)) * (width - left - right);
  const { y, ticks } = yScale(term);
  const envelope = foldEnvelope(term);
  const base = height - bottom;
  const maxWeight = Math.max(...term.weights, 0) || 1;
  /** @param {number[]} values @returns {Array<[number, number]>} */
  const points = (values) => xs.map((value, index) => [x(value), y(values[index])]);
  const exposure = pathData([
    [x(first), base],
    ...xs.map((value, index) => /** @type {[number, number]} */ (
      [x(value), base - (term.weights[index] / maxWeight) * (base - top) / 4])),
    [x(last), base]
  ]);
  const band = pathData([...points(envelope.hi), ...points(envelope.lo).reverse()]);
  const lines = term.folds.map((fold, index) => `<path class="cv-fold-line" style="stroke: ${foldColour(index)}"
      d="${pathData(points(fold.values))}"><title>${escapeHTML(fold.label)}</title></path>`).join("");
  const xTicks = niceTicks(first, last, 6).map((tick) => `<text class="cv-tick" x="${px(x(tick))}"
      y="${base + 16}" text-anchor="middle">${escapeHTML(fmt(tick))}</text>`).join("");
  return `<svg class="cv-chart-svg" viewBox="0 0 ${width} ${height}" role="img"
      aria-label="${escapeHTML(`${term.name} relativities by fold`)}">
    ${frameMarkup(term, y, ticks)}<path class="exposure-density" d="${exposure}Z"></path>
    <path class="cv-envelope" d="${band}Z"></path>${lines}
    <path class="cv-fit-line" d="${pathData(points(term.fit))}"></path>
    ${term.edited ? `<path class="cv-edited-line" d="${pathData(points(term.edited))}"></path>` : ""}
    ${xTicks}${legendMarkup(term)}</svg>`;
}

/** @param {CVTermItem} term */
function chartMarkup(term) {
  return term.kind === "levels" ? levelChartMarkup(term) : curveChartMarkup(term);
}

/** @param {CVTermItem[]} terms @param {string} name */
function currentTerm(terms, name) {
  return terms.find((term) => term.name === name) ?? terms[0] ?? null;
}

/**
 * The searchable term list ranked by spread, and the chosen term's chart.
 * @param {CVReportPayload} payload @param {CVTabState} state
 */
export function relativitiesMarkup(payload, state) {
  const relativities = payload.relativities;
  const head = sectionHead(
    "Relativities across folds",
    "levels in the model's order · each curve re-centred on its exposure-weighted mean"
  );
  if (!relativities.available) {
    return `<section class="report-section cv-relativities">${head}
      <div class="report-note">${escapeHTML(relativities.note || "")}</div></section>`;
  }
  const shown = filterTerms(relativities.terms, state.query);
  const current = currentTerm(shown, state.term);
  const origin = relativities.origin === "run"
    ? "From Run CV on the current model, with the hand edits put back on every fold."
    : "From the supplied result's fold models.";
  return `<section class="report-section cv-relativities">${head}
    <p class="cv-hint">${escapeHTML(origin)}${relativities.stale ? " The model has changed since." : ""}</p>
    <div class="cv-rel-layout">
      <div class="cv-term-panel">
        <input id="cvTermSearch" class="cv-search" type="search" placeholder="Search terms"
          aria-label="Search terms" value="${escapeHTML(state.query)}">
        <div class="cv-term-head" aria-hidden="true"><span>Least stable first</span><span>spread</span><span>min r</span></div>
        <div class="cv-term-list" data-cv-term-list>${termListMarkup(shown, current?.name ?? "", state.query)}</div>
        <p class="cv-hint">spread: mean distance of a fold's curve from the fold average. min r: lowest fold correlation with it.</p>
      </div>
      <div class="cv-chart" data-cv-chart>${current ? chartMarkup(current) : ""}</div>
    </div></section>`;
}

/**
 * The whole tab below the report header.
 * @param {CVReportPayload} payload @param {CVTabState} state
 */
export function cvTabMarkup(payload, state) {
  return toolbarMarkup(payload, state) + performanceMarkup(payload) + foldTableMarkup(payload)
    + relativitiesMarkup(payload, state);
}

/**
 * Bind the Cross-validation tab to the report frame: render each ``cv``
 * payload, and start, poll and cancel its two jobs.
 *
 * @param {object} options
 * @param {HTMLElement} options.frame
 * @param {JobClient} options.client
 * @param {(kind:JobKind, job:JobStatus)=>unknown} options.onJobSettled
 * @param {(ms:number)=>Promise<void>} [options.pause]
 */
export function createCVTab({
  frame,
  client,
  onJobSettled,
  pause = (ms) => new Promise((resolve) => { setTimeout(resolve, ms); })
}) {
  /** @type {CVReportPayload|null} */
  let payload = null;
  /** @type {CVTabState} */
  const state = { term: "", query: "", jobs: { cv: null, final_fit: null }, errors: { cv: "", final_fit: "" } };
  /** @type {Set<string>} */
  const polling = new Set();

  function renderAll() {
    if (!payload) return;
    const search = frame.querySelector("#cvTermSearch");
    const caret = search === frame.ownerDocument?.activeElement && search instanceof HTMLInputElement
      ? search.selectionStart
      : null;
    frame.innerHTML = cvTabMarkup(payload, state);
    const next = frame.querySelector("#cvTermSearch");
    if (caret !== null && next instanceof HTMLInputElement) {
      next.focus();
      next.setSelectionRange(caret, caret);
    }
  }

  function renderToolbar() {
    const toolbar = frame.querySelector("[data-cv-toolbar]");
    if (!payload || !toolbar) return renderAll();
    toolbar.outerHTML = toolbarMarkup(payload, state);
  }

  function renderTerms() {
    const list = frame.querySelector("[data-cv-term-list]");
    const chart = frame.querySelector("[data-cv-chart]");
    if (!payload || !list || !chart) return renderAll();
    const shown = filterTerms(payload.relativities.terms, state.query);
    const current = currentTerm(shown, state.term);
    list.innerHTML = termListMarkup(shown, current?.name ?? "", state.query);
    chart.innerHTML = current ? chartMarkup(current) : "";
  }

  /** @param {JobKind} kind @param {unknown} error */
  function refuse(kind, error) {
    state.errors[kind] = error instanceof Error ? error.message : String(error);
    renderToolbar();
  }

  /** @param {JobKind} kind @param {string} jobId */
  async function poll(kind, jobId) {
    if (polling.has(jobId)) return;
    polling.add(jobId);
    try {
      let job = state.jobs[kind];
      while (job?.job_id === jobId && job.status === "running") {
        await pause(POLL_MS);
        state.jobs[kind] = newerJob(state.jobs[kind], /** @type {JobStatus} */ (await client.jobStatus(jobId)));
        job = state.jobs[kind];
        renderToolbar();
      }
      if (job?.job_id === jobId) await onJobSettled(kind, job);
    } catch (error) {
      refuse(kind, error);
    } finally {
      polling.delete(jobId);
    }
  }

  /** @param {JobKind} kind */
  async function start(kind) {
    if (state.jobs[kind]?.status === "running") return;
    state.errors[kind] = "";
    try {
      const job = /** @type {JobStatus} */ (await client.jobStart(kind));
      state.jobs[kind] = job;
      renderToolbar();
      await poll(kind, job.job_id);
    } catch (error) {
      refuse(kind, error);
    }
  }

  /** @param {JobKind} kind */
  async function cancel(kind) {
    const job = state.jobs[kind];
    if (job?.status !== "running") return;
    try {
      await client.jobCancel(job.job_id);
      state.jobs[kind] = { ...job, cancel_requested: true };
      renderToolbar();
    } catch (error) {
      refuse(kind, error);
    }
  }

  /** @param {Event} event */
  function onClick(event) {
    const target = event.target instanceof Element ? event.target : null;
    const starter = target?.closest("[data-cv-start]");
    const canceller = target?.closest("[data-cv-cancel]");
    const termButton = target?.closest("[data-cv-term]");
    if (starter instanceof HTMLButtonElement) void start(jobKind(starter.dataset.cvStart || ""));
    else if (canceller instanceof HTMLElement) void cancel(jobKind(canceller.dataset.cvCancel || ""));
    else if (termButton instanceof HTMLElement) {
      state.term = termButton.dataset.cvTerm || "";
      renderTerms();
    }
  }

  /** @param {Event} event */
  function onInput(event) {
    if (!(event.target instanceof HTMLInputElement) || event.target.id !== "cvTermSearch") return;
    state.query = event.target.value;
    renderTerms();
  }

  /** @param {KeyboardEvent} event */
  function onKeyDown(event) {
    const input = event.target;
    if (event.key !== "Escape" || !(input instanceof HTMLInputElement) || input.id !== "cvTermSearch") return;
    input.value = "";
    state.query = "";
    renderTerms();
  }

  frame.addEventListener("click", onClick);
  frame.addEventListener("input", onInput);
  frame.addEventListener("keydown", onKeyDown);

  return Object.freeze({
    /** @param {CVReportPayload} next */
    render(next) {
      payload = next;
      for (const kind of JOB_KINDS) {
        state.jobs[kind] = newerJob(state.jobs[kind], next.jobs[kind]);
        const job = state.jobs[kind];
        if (job?.status === "running") void poll(kind, job.job_id);
      }
      renderAll();
    },
    start,
    cancel,
    destroy() {
      frame.removeEventListener("click", onClick);
      frame.removeEventListener("input", onInput);
      frame.removeEventListener("keydown", onKeyDown);
    }
  });
}
```

3d. Create `src/superglm/editor/app/styles/cv.css`:

```css
/* cv.css: the Cross-validation tab (views/cv_tab.js). Folds take the trace
   palette in order, the all-rows fit the current-edit blue, exposure the
   chart's yellow; every colour is a token, so dark.css restyles it. */

.cv-toolbar {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  align-items: center;
  gap: 6px 16px;
}
.cv-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}
.cv-chip {
  border-radius: 999px;
  padding: 2px 9px;
  background: var(--surface-subtle);
  color: var(--muted);
  font-size: 12px;
}
.cv-chip[data-tone="waiting"],
.cv-chip[data-tone="stale"] {
  background: var(--sig-weak-bg);
  color: var(--sig-weak-fg);
}
.cv-actions {
  display: flex;
  gap: 8px;
}
.cv-action {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  height: 32px;
  padding: 0 12px;
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  background: var(--surface);
  color: var(--text);
  font-weight: 600;
  cursor: pointer;
}
.cv-action.is-primary {
  border-color: var(--blue);
  background: var(--blue);
  color: var(--surface);
}
.cv-action:not(:disabled):hover {
  background: var(--surface-hover);
}
.cv-action.is-primary:not(:disabled):hover {
  background: var(--blue);
  filter: brightness(1.08);
}
.cv-reason,
.cv-job {
  grid-column: 1 / -1;
  margin: 0;
  color: var(--muted);
  font-size: 12px;
}
.cv-job[data-status="failed"] {
  color: var(--danger);
}
.cv-cancel {
  margin-left: 8px;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--surface);
  color: var(--text);
  font-size: 12px;
  cursor: pointer;
}
.cv-jobs {
  display: contents;
}
.cv-section-head {
  display: flex;
  align-items: baseline;
  gap: 10px;
}
.cv-section-head h3 {
  margin: 0;
  font-size: 13px;
  font-weight: 600;
}
.cv-hint {
  margin: 0;
  color: var(--muted);
  font-size: 12px;
}
.cv-cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 12px;
}
.cv-card {
  display: grid;
  gap: 8px;
  border-radius: var(--radius-lg);
  padding: 12px 14px;
  background: var(--surface-subtle);
}
.cv-card-title {
  color: var(--muted);
  font-size: 12px;
}
.cv-card-value {
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  gap: 6px;
  font-family: var(--font-mono);
}
.cv-card-value strong {
  font-size: 18px;
  font-weight: 500;
}
.cv-card-label {
  font-family: var(--font-sans);
  color: var(--muted);
  font-size: 12px;
}
.cv-card-spread {
  color: var(--muted);
  font-size: 12px;
}
.cv-strip {
  display: block;
  width: 220px;
  height: 24px;
}
.cv-strip-axis {
  stroke: var(--border);
  stroke-width: 2;
}
.cv-mean {
  stroke: var(--text);
  stroke-width: 1.5;
}
.cv-fold-table td {
  font-family: var(--font-mono);
  font-size: 12px;
}
.cv-fold-table td:first-child {
  font-family: var(--font-sans);
}
.cv-fold-swatch {
  display: inline-block;
  width: 8px;
  height: 8px;
  margin-right: 7px;
  border-radius: 50%;
}
.cv-mean-row td {
  font-weight: 600;
}
.cv-rel-layout {
  display: grid;
  grid-template-columns: 250px minmax(0, 1fr);
  align-items: start;
  gap: 16px;
}
.cv-term-panel {
  display: grid;
  gap: 4px;
  border-radius: var(--radius-lg);
  padding: 10px 8px;
  background: var(--surface-subtle);
}
.cv-search {
  height: var(--control-height);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: 0 8px;
  background: var(--surface);
  color: var(--text);
}
.cv-term-head,
.cv-term {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 56px 48px;
  align-items: center;
  padding: 6px 8px;
  text-align: right;
}
.cv-term-head {
  color: var(--muted);
  font-size: 11px;
}
.cv-term-head > :first-child,
.cv-term-name {
  overflow: hidden;
  text-align: left;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.cv-term {
  width: 100%;
  border: 0;
  border-radius: var(--radius-md);
  background: transparent;
  color: var(--text);
  font-family: var(--font-mono);
  font-size: 11.5px;
  cursor: pointer;
}
.cv-term-name {
  font-family: var(--font-sans);
  font-size: 13px;
  font-weight: 600;
}
.cv-term[aria-current="true"] {
  background: var(--surface);
  box-shadow: inset 2px 0 0 var(--blue);
}
.cv-term mark {
  background: var(--blue-soft);
  color: inherit;
}
.cv-chart {
  min-width: 0;
  border-radius: var(--radius-lg);
  padding: 10px 12px;
  background: var(--surface-subtle);
}
.cv-chart-svg {
  display: block;
  width: 100%;
  height: auto;
  font-size: 11px;
}
.cv-grid { stroke: var(--grid); }
.cv-chart-svg .exposure,
.cv-chart-svg .exposure-density { fill-opacity: 0.35; }
.cv-tick,
.cv-level,
.cv-legend text { fill: var(--muted); }
.cv-axis-title { fill: var(--text); font-weight: 600; }
.cv-range { stroke: var(--grey); stroke-width: 2; stroke-opacity: 0.6; }
.cv-fold-dot { fill-opacity: 0.9; }
.cv-fit,
.cv-fit-line { fill: none; stroke: var(--blue); stroke-width: 2.6; }
.cv-edited,
.cv-edited-line { fill: none; stroke: var(--orange); stroke-width: 2; stroke-dasharray: 5 4; }
.cv-fold-line { fill: none; stroke-width: 1.3; stroke-opacity: 0.85; }
.cv-envelope { fill: var(--blue-band); stroke: none; }
.cv-legend-fold { fill: var(--trace-0); }
.cv-legend-fit { fill: var(--blue); }
.cv-legend-edited { fill: var(--orange); }
.cv-legend-range { fill: var(--grey); }
.cv-legend-exposure { fill: var(--yellow); }

@media (max-width: 760px) {
  .cv-toolbar,
  .cv-rel-layout {
    grid-template-columns: minmax(0, 1fr);
  }
}
```

3e. `app/reports.js`. This is the `cv` branch, plus the Final fit section on the Final Fit tab:

Edit 1 (master line 1). Replace

```js
import { escapeHTML, fmt, fmtSigned } from "./format.js";
import { metricDirection } from "./metrics.js";
```

with

```js
import { escapeHTML, fmt, fmtSigned } from "./format.js";
import { metricDirection } from "./metrics.js";
import { cvSourceLine } from "./views/cv_tab.js";
```

Edit 2 (master line 13). Replace

```js
export function renderReport(payload, { reportTitle, reportStatus, reportFrame }) {
  reportTitle.textContent = payload.title || "Report";
  reportStatus.textContent = payload.note || "";
```

with

```js
export function renderReport(payload, { reportTitle, reportStatus, reportFrame }, cvTab = null) {
  reportTitle.textContent = payload.title || "Report";
  if (payload.report === "cv" && cvTab) {
    reportStatus.textContent = cvSourceLine(payload);
    cvTab.render(payload);
    return;
  }
  reportStatus.textContent = payload.note || "";
```

Edit 3 (master line 22). Replace

```js
  const summarySection = payload.report === "final" ? renderFinalSummary(payload.summary) : "";
  reportFrame.innerHTML = `${splitSection}${cvSection}${summarySection}`;
```

with

```js
  const summarySection = payload.report === "final" ? renderFinalSummary(payload.summary) : "";
  const finalFitSection = payload.report === "final" ? renderFinalFit(payload.final_fit) : "";
  reportFrame.innerHTML = `${splitSection}${cvSection}${summarySection}${finalFitSection}`;
```

Edit 4 (master line 161). Replace

```js
function formatValue(value) {
```

with

```js
function renderFinalFit(finalFit) {
  if (!finalFit) return "";
  if (!finalFit.available) {
    return `
      <section class="report-section final-fit">
        <h3>Final Fit on All Rows</h3>
        <div class="report-note">${escapeHTML(finalFit.note || "")}</div>
      </section>
    `;
  }
  const model = finalFit.summary?.model || {};
  const notes = [
    `Refitted on ${Number(finalFit.n_rows).toLocaleString("en-US")} ${finalFit.splits.join(" and ")} rows; the test split stays held out.`,
    finalFit.carried.length ? `Hand edits put back: ${finalFit.carried.join(", ")}.` : "",
    finalFit.stale ? "The model has changed since; run Final fit again before exporting." : ""
  ].filter(Boolean);
  return `
    <section class="report-section final-fit">
      <h3>Final Fit on All Rows</h3>
      ${notes.map((note) => `<div class="report-note">${escapeHTML(note)}</div>`).join("")}
      <table class="report-table" aria-label="Final fit on all rows">
        <tbody>
          <tr><th>Rows</th><td>${escapeHTML(formatValue(finalFit.n_rows))}</td></tr>
          <tr><th>Method</th><td>${escapeHTML(formatValue(model.method))}</td></tr>
          <tr><th>Total EDF</th><td>${escapeHTML(formatValue(model.effective_df))}</td></tr>
          <tr><th>Deviance</th><td>${escapeHTML(formatValue(model.deviance))}</td></tr>
          <tr><th>AIC</th><td>${escapeHTML(formatValue(model.aic))}</td></tr>
          <tr><th>BIC</th><td>${escapeHTML(formatValue(model.bic))}</td></tr>
        </tbody>
      </table>
    </section>
  `;
}

function formatValue(value) {
```

3f. `app/main.js`. If an earlier task added options to `bindExportDialog({...})`, add
`finalFitAvailable` beside them:

Edit 1 (master line 44). Replace

```js
import { bindExportDialog } from "./views/export_dialog.js";
```

with

```js
import { createCVTab } from "./views/cv_tab.js";
import { bindExportDialog } from "./views/export_dialog.js";
```

Edit 2 (master line 197). Replace

```js
const undo = () => actions.executeStateMutation({
```

with

```js
// The Cross-validation tab draws into the report frame. A finished Run CV or
// Final fit changes the tab, and a Final fit also what Export offers.
const cvTab = createCVTab({
  frame: reportFrame,
  client: editorClient,
  onJobSettled: async (kind) => {
    if (kind === "final_fit") await actions.refreshFromPython();
    await refreshActiveReport();
  }
});

const undo = () => actions.executeStateMutation({
```

Edit 3 (master line 539). Replace

```js
    status: exportStatus
  },
  saveBlobToFile
});
```

with

```js
    status: exportStatus
  },
  saveBlobToFile,
  finalFitAvailable: () => {
    const finalFit = store.getState().remote.snapshot?.final_fit;
    return Boolean(finalFit?.available && !finalFit.stale);
  }
});
```

Edit 4 (master line 767). Replace

```js
  const activeView = view === "final" ? "final" : view === "validation" ? "validation" : "editor";
```

with

```js
  const activeView = ["validation", "cv", "final"].includes(view) ? view : "editor";
```

Edit 5 (master line 1098). Replace

```js
function renderReportEvidence(evidence, activeView) {
```

with

```js
const REPORT_TITLES = Object.freeze({
  validation: "Validation Report",
  cv: "Cross-validation",
  final: "Final Fit Report"
});

function renderReportEvidence(evidence, activeView) {
```

Edit 6 (master line 1105). Replace

```js
    renderReport(evidence.payload, { reportTitle, reportStatus, reportFrame });
  } else {
    reportTitle.textContent = activeView === "final" ? "Final Fit Report" : "Validation Report";
```

with

```js
    renderReport(evidence.payload, { reportTitle, reportStatus, reportFrame }, cvTab);
  } else {
    reportTitle.textContent = REPORT_TITLES[activeView] || REPORT_TITLES.validation;
```

3g. `app/index.html`:

Edit 1 (master line 27). Replace

```html
<link rel="stylesheet" href="/assets/styles/dialogs.css">
```

with

```html
<link rel="stylesheet" href="/assets/styles/dialogs.css">
<link rel="stylesheet" href="/assets/styles/cv.css">
```

Edit 2 (master line 56). Replace

```html
          role="tab" aria-selected="false" aria-controls="reportPanel" tabindex="-1">Validation</button>
```

with

```html
          role="tab" aria-selected="false" aria-controls="reportPanel" tabindex="-1">Validation</button>
        <button id="cvTab" class="app-tab" type="button" data-view="cv"
          role="tab" aria-selected="false" aria-controls="reportPanel" tabindex="-1">Cross-validation</button>
```

Edit 3 (master line 569). Replace

```html
            <strong>Excel rating workbook</strong>
            <small>Rating tables and structured fit summary</small>
          </span>
        </label>
```

with

```html
            <strong>Excel rating workbook</strong>
            <small>Rating tables and structured fit summary</small>
          </span>
        </label>
        <label class="export-format-card">
          <input id="exportFinalFit" type="radio" name="exportFormat" value="final"
            aria-label="Final fit model" disabled>
          <span>
            <strong>Final fit model</strong>
            <small>Refitted on train and validation rows, hand edits put back</small>
          </span>
        </label>
```

3h. `app/views/export_dialog.js`:

Edit 1 (master line 10). Replace

```js
  xlsx: Object.freeze({
    filename: "superglm_rating_tables.xlsx",
    description: "Excel rating workbook",
    validationDescription: "Excel rating workbook",
    accept: Object.freeze({
      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": Object.freeze([
        ".xlsx",
      ]),
    }),
  }),
});
```

with

```js
  xlsx: Object.freeze({
    filename: "superglm_rating_tables.xlsx",
    description: "Excel rating workbook",
    validationDescription: "Excel rating workbook",
    accept: Object.freeze({
      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": Object.freeze([
        ".xlsx",
      ]),
    }),
  }),
  final: Object.freeze({
    filename: "superglm_final_model.joblib",
    description: "Final fit model",
    validationDescription: "Validated final fit model",
    accept: Object.freeze({ "application/octet-stream": Object.freeze([".joblib"]) }),
  }),
});
```

Edit 2 (master line 42). Replace

```js
 * @property {(blob:Blob, filename:string, metadata:{description:string,accept:Readonly<Record<string,readonly string[]>>})=>Promise<string|null>} saveBlobToFile
 */
```

with

```js
 * @property {(blob:Blob, filename:string, metadata:{description:string,accept:Readonly<Record<string,readonly string[]>>})=>Promise<string|null>} saveBlobToFile
 * @property {()=>boolean} [finalFitAvailable] whether a current Final fit model exists
 */
```

Edit 3 (master line 78). Replace

```js
function successMessage(message, format, validation) {
  if (format !== "joblib") return message;
```

with

```js
function successMessage(message, format, validation) {
  if (format === "xlsx") return message;
```

Edit 4 (master line 92). Replace

```js
export function bindExportDialog({ client, nodes, saveBlobToFile }) {
  let pending = false;

  /** @returns {ExportFormat} */
  function selectedFormat() {
    const value = nodes.formatInputs.find((input) => input.checked)?.value;
    return value === "xlsx" ? "xlsx" : "joblib";
  }
```

with

```js
export function bindExportDialog({
  client, nodes, saveBlobToFile, finalFitAvailable = () => false,
}) {
  let pending = false;

  /** @returns {ExportFormat} */
  function selectedFormat() {
    const value = nodes.formatInputs.find((input) => input.checked)?.value;
    return value === "xlsx" || value === "final" ? value : "joblib";
  }

  // Export offers the Final fit model only while one is current (D6).
  function syncFinalFit() {
    const finalInput = nodes.formatInputs.find((input) => input.value === "final");
    if (!finalInput) return;
    finalInput.disabled = !finalFitAvailable();
    if (!finalInput.disabled || !finalInput.checked) return;
    finalInput.checked = false;
    const joblib = nodes.formatInputs.find((input) => input.value === "joblib");
    if (joblib) joblib.checked = true;
    normaliseFilename();
  }
```

Edit 5 (master line 137). Replace

```js
  async function openDialog() {
    nodes.status.textContent = "";
```

with

```js
  async function openDialog() {
    syncFinalFit();
    nodes.status.textContent = "";
```

3i. `app/views/help_content.js`. This is the Help entry for the new actions:

Edit 1 (master line 209). Replace

```js
  Object.freeze({
    title: "Exporting",
```

with

```js
  Object.freeze({
    title: "Cross-validation",
    items: Object.freeze([
      "Pass a cross_validate() result to edit(model, cv=result). The tab shows each fold's scores and, when the result kept its fold models (return_estimators=True), every term's relativities by fold, least stable first.",
      "Run CV on current model refits the current structure on the same folds and puts your hand edits back on each fold before scoring. It waits while changes wait for Refit, and needs the rows the folds were drawn on: cv_data=, or train data with the same row count.",
      "Final fit on all rows refits the current structure on train and validation rows and puts your hand edits back; the test split stays held out. Export then offers it as Final fit model.",
      "Both run in the background with a Cancel button. A job whose model changed while it ran is not kept.",
    ]),
  }),
  Object.freeze({
    title: "Exporting",
```

Edit 2 (master line 213). Replace

```js
      "Excel rating workbooks require training or retained fit data and include structured summary tables.",
```

with

```js
      "Excel rating workbooks require training or retained fit data and include structured summary tables.",
      "Final fit model is the latest Final fit on all rows, offered while the model has not changed since.",
```

- [ ] **Step 4: Run tests, expect PASS.**

```bash
npm run check:frontend
./.venv/bin/python -m pytest tests/test_editor_cv.py tests/test_editor.py -q -n 8
./.venv/bin/python -m pytest tests/test_editor_browser.py tests/editor -m browser --run-browser -q -n 6
./.venv/bin/ruff check src/ tests/ && ./.venv/bin/ruff format --check src/ tests/
```

Expected: `tsc` is clean. Node: every test passes (on 155832e8 plus this section, 200 tests:
187 existing and 13 in `cv_tab.test.js`). `test_editor_cv.py`: 19 passed. Browser: all pass,
including the two CV tests and the updated tab list. On a supplied 5-fold result the tab was
checked by eye against board 6, in light and dark.

- [ ] **Step 5: Commit.**

```bash
git add src/superglm/editor/app/views/cv_tab.js src/superglm/editor/app/styles/cv.css \
  src/superglm/editor/app/api/contracts.js src/superglm/editor/app/api/client.js \
  src/superglm/editor/app/reports.js src/superglm/editor/app/main.js src/superglm/editor/app/index.html \
  src/superglm/editor/app/views/export_dialog.js src/superglm/editor/app/views/help_content.js \
  tests/editor_frontend/cv_tab.test.js tests/editor_frontend/client.test.js \
  tests/editor_frontend/export_dialog.test.js tests/editor/test_editor_cv_browser.py \
  tests/editor/test_editor_workspace_browser.py tests/test_editor_cv.py tests/test_editor.py
git commit -F - <<'EOF'
Editor: the Cross-validation tab

A fourth app view on the report panel (board 6). The header shows the source line, the
waiting and stale chips, Run CV and Final fit with progress and Cancel, and why a
button is disabled. Below it: fold performance cards (mean ± sd · pooled, a dot per
fold, supplied beside current), the fold table, and relativities by fold with a
searchable term list ranked by spread. Level charts are in model order; curve charts
show the fold envelope. Export offers the Final fit model while it is current.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```

### Risks and open questions for the reviewer

- **The anchor reading of D5** (see the amendments). Shape exact, level re-anchored, a whole-curve shift kept. The alternative, absolute values, biased a Final fit by 0.36 log when its reference re-resolved. A3's same-rows carry-over (D2) can stay a direct copy; on the same rows the two readings differ only by rounding-sized centring.
- **Pinned levels in a fold.** `apply._apply_categorical_term` refuses to edit a categorical with pinned levels, which happens when a thin level has no training rows in one fold. That fold then fails inside `cross_validate`: it scores NaN, shows "Converged: no" and logs a warning. Rare and visible; a fixed sentence could follow.
- **Memory.** A Final fit keeps one model, and with `retain_fit_state` it keeps the concatenated train ∪ validation frame. A CV run keeps only scores and curves, not the fold models.
- **Supplied fold curves past a fold's range** follow each model's own extrapolation. Splines hold their end value; a `Polynomial` extrapolates, so it can swing at the grid's ends. That is what the fold model predicts there.
- **Publish-if-current drops a long Run CV** if the user keeps editing while it runs. That is by design, and the job line says to run it again.
- The "replays exactly" test asserts bitwise-equal fold scores for two fits in one thread (same folds, structure and scorers). Same-thread determinism is already required; if a platform ever breaks it, the right tolerance is the solver's convergence tolerance.
- `status(wait=True)` holds a server worker for up to 30 s. Only the tests use it; the frontend polls every 250 ms without waiting.


## Phase 7b — Frontend structure (M1, M2): DEFERRED, not run in this build

M1 and M2 run after G5 and before Z1. Every feature task has landed by then, so
the split moves the final code once. They change no behaviour. The proof is
that every existing test passes unchanged; only import lines and asset/route
pins in tests may move.

### Task M1: Split `main.js` into controllers

**Files:**
- Create:
  - `src/superglm/editor/app/controllers/context.js`
  - `src/superglm/editor/app/controllers/evidence.js`
  - `src/superglm/editor/app/controllers/busy.js`
  - `src/superglm/editor/app/controllers/structure.js`
  - `src/superglm/editor/app/controllers/build.js`
  - `src/superglm/editor/app/controllers/render.js`
  - `src/superglm/editor/app/controllers/views.js`
- Modify: `src/superglm/editor/app/main.js`, which becomes the bootstrap: DOM
  references, store/client/actions creation, controller wiring and event
  binding.
- Modify: `tests/test_editor.py`.
  - Add the new files to the fetched-asset list in
    `test_widget_serves_editor_app_assets` (~6746-6767).
  - Move each `main.js` import assertion (~6782-6789) to the module that now
    holds the import.
  - Move each route-string pin (~6373-6398) that names `main.js` to the module
    that now holds the string.
- Modify: `docs/development/internals/editor-frontend.md` (module list).

**Interfaces:**
- Consumes: the final `main.js` after G5.
- Produces:
  - each controller exports one factory, `create<Name>Controller(ctx)`,
    returning an object of the functions listed below;
  - `ctx` is the `AppContext` built once in `main.js`:
    `{dom, store, actions, client, settings, controllers}`. `controllers` is
    filled in as each factory returns, so controllers reach each other only
    through `ctx.controllers`.

**What moves where.** Names are as on 155832e8. A function a feature task
added goes with the group it serves.

| Module | Functions |
|---|---|
| `evidence.js` | `summaryNodes`, `summaryRequestPayload`, `refreshMetricsView`, `refreshSummaryView`, `refreshActiveReport`, `scheduleVisibleEvidence`, `scheduleVisibleEvidenceCatchUp`, `renderMetricsEvidence`, `renderReportEvidence`, `renderSummaryEvidence`, `renderEvidenceFreshness`, `renderSnapshotRevision` |
| `busy.js` | `setAppBusy`, `restoreFocusAfterBusy`, `showTimingStatus`, the Settings timing renderer that F1 renamed from `renderAdvancedTiming`, `formatTimingDetails`, `formatEvidenceTimingDetails`, `formatMilliseconds`, `renderMutationBusy`, `renderRecovery`, `retryFailedMutation` |
| `structure.js` | `runStructuralRefit`, A5's staging helpers, `updateCollapseAction`, `updateShapeActions`, `renderShapeReason`, `selectedLevelLabel`, `selectionTouchesCollapsedGroup`, `updateResetOrderAction`, `updateGroupDisplayControl`, `renderShapeJoin` |
| `build.js` | `canShowContributions`, `buildDurationMs`, `updateBuildDurationLabel`, `startContributionBuild`, `runContributionBuild`, `advanceContributionBuild`, `stopContributionBuild`, `updateHandleCount`, `applyTermDefaults` |
| `render.js` | every `select…RenderState` / `same…RenderState` / `render…State` trio (feature list, chart, history, app bar, active view, selection, interaction), plus `renderChartWorkspace`, `renderChartOnly`, `termCatalogueKey`, `selectionContextNote` |
| `views.js` | `showView`, `renderAppView`, `renderInspectorView`, `syncViewport`, `redrawChartToFit`, `selectFeature`, `refreshFromPython` |
| `main.js` (stays) | `loadState`, `executeStateMutation`, `mutationName`, `undo`, `redo`, the small accessors (`selectedTerm`, `currentTerm`, `currentSelection`, `interactionMode`, `visualMode`, `activeGroupDisplayMode`, the preview/zoom setters), event binding, startup |

- [ ] **Step 1: Pin the current behaviour.** Run every suite and record the
  counts. This is the baseline the split must reproduce exactly.

  ```bash
  npm run check:frontend
  ./.venv/bin/python -m pytest tests/test_editor.py tests/test_editor_structure.py -q
  ./.venv/bin/python -m pytest tests/editor -m browser --run-browser -q
  ./.venv/bin/python -m pytest tests/test_editor_browser.py -m browser --run-browser -q
  ```

  Expected: all pass. Run the two browser commands separately, never in one
  `-n` run.

- [ ] **Step 2: Add `controllers/context.js`.** Use the real factory names that
  `state/store.js`, `state/actions.js` and `api/client.js` export.

  ```js
  // @ts-check
  /** The one object main.js builds and hands to every controller. */

  /**
   * @typedef {object} AppContext
   * @property {Record<string, HTMLElement>} dom   element references looked up once in main.js
   * @property {ReturnType<import("../state/store.js").createStore>} store
   * @property {ReturnType<import("../state/actions.js").createActionController>} actions
   * @property {ReturnType<import("../api/client.js").createEditorClient>} client
   * @property {typeof import("../views/settings.js")} settings
   * @property {Record<string, any>} controllers   filled in by main.js as each factory returns
   */
  export {};
  ```

- [ ] **Step 3: Move one group at a time, `evidence.js` first.** For each group:
  1. Create the module with `// @ts-check`, a one-line header comment saying
     what it owns, and the factory. The bodies are unchanged except that free
     references now read from `ctx`:

     ```js
     // @ts-check
     /** Evidence panels: metrics strip, inspector summary, report views, and their freshness. */

     /** @param {import("./context.js").AppContext} ctx */
     export function createEvidenceController(ctx) {
       const { dom, store, actions, client } = ctx;
       // ...the moved functions...
       return { refreshMetricsView, refreshSummaryView, refreshActiveReport,
         scheduleVisibleEvidence, scheduleVisibleEvidenceCatchUp, renderMetricsEvidence,
         renderReportEvidence, renderSummaryEvidence, renderEvidenceFreshness,
         renderSnapshotRevision, summaryNodes, summaryRequestPayload };
     }
     ```

  2. In `main.js`, replace the moved definitions with
     `const evidence = createEvidenceController(ctx); ctx.controllers.evidence = evidence;`
     and turn each call site into `evidence.<name>(...)`.
  3. Run Step 1's four commands. Expected: the same counts.
  4. Commit:

     ```bash
     git add src/superglm/editor/app/main.js src/superglm/editor/app/controllers tests/test_editor.py
     git commit -m "Editor frontend: move evidence rendering out of main.js

     No behaviour change; suites unchanged.

     Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
     ```

  Then repeat for `busy.js`, `structure.js`, `build.js`, `render.js` and
  `views.js`, in that order, with one commit each.

- [ ] **Step 4: Check the size.** Run `wc -l src/superglm/editor/app/main.js src/superglm/editor/app/controllers/*.js`.
  Expected: `main.js` is under 500 lines and no controller is over 500. If one
  is, split it along the table's sub-groups and record why in the commit.

- [ ] **Step 5: Document.** Add the module list and the `ctx.controllers` rule
  to `docs/development/internals/editor-frontend.md`, then commit.

### Task M2: Type-check every frontend module

**Files:**
- Modify: `jsconfig.json` (`include`).
- Modify: whichever of `app/main.js`, `app/controllers/*.js`,
  `app/interactions.js`, `app/summary.js`, `app/chart.js`, `app/reports.js`,
  `app/shapes.js`, `app/history.js`, `app/metrics.js` and `app/format.js` need
  JSDoc. Annotations only.

**Interfaces:**
- Consumes: M1.
- Produces: `tsc -p jsconfig.json` covers all of `src/superglm/editor/app/**/*.js`.

- [ ] **Step 1: Widen the include.** In `jsconfig.json` replace the five
  `include` entries with:

  ```json
    "include": [
      "src/superglm/editor/app/**/*.js",
      "tests/editor_frontend/**/*.js"
    ]
  ```

- [ ] **Step 2: List the errors.**

  Run: `npm run typecheck:frontend 2>&1 | tee /tmp/tsc-errors.txt | tail -5`.

  Expected: errors, all in files outside the old include.

- [ ] **Step 3: Fix the errors one file at a time, with annotations only.**
  - Add `// @ts-check`.
  - Add `@param`/`@returns`/`@typedef` from `app/api/contracts.js`.
  - Where a DOM lookup can be null, add an `instanceof HTMLElement` guard that
    throws the file's existing missing-element error.
  - Change no other control flow. Use no `any` where a contract type exists.
    Add no `// @ts-ignore`.
  - After each file, run `npm run check:frontend`. Expected: that file is clean
    and the node tests pass.

- [ ] **Step 4: Run the browser suites.** Run the four commands from M1 Step 1.
  Expected: the M1 baseline counts.

- [ ] **Step 5: Commit.**

  ```bash
  git add jsconfig.json src/superglm/editor/app
  git commit -m "Editor frontend: type-check every module

  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
  ```


## Phase 8 — Integration

### Task Z1: User docs, full verification, timing pair, whole-branch review

**Files:**
- Modify: `docs/tutorials/edit-a-model-in-the-browser.md`. Sections to change:
  - "Open the Editor" (line 7): `cv=` and the data arguments;
  - "Select, Move, Zoom, and Handles" (38): click, Shift-click, Ctrl-click, Shift-drag;
  - "Group and Ungroup Categorical Levels" (68), "Set a Reference Level" (82) and
    "Shape a Range" (96): changes wait for Refit, and keep-reference;
  - "Undo, Redo and Revert" (152): waiting steps, notes;
  - "Inspector and Help" (194): search, filters, Settings;
  - "Theme" (205): the switch, "Follow the browser";
  - "Export" (211): the final-fit model, the history notes, waiting changes not
    included;
  - "Keyboard Shortcuts" (224): `R`.

  New sections: "Cross-validation" (after Export) and "Rating-table preview"
  (after Shape a Range).
- Modify: `docs/development/internals/editor-frontend.md`. Record:
  - the new modules (`views/settings.js`, the cv view, the theme switch);
  - the settings key;
  - the pending payload fields;
  - the job routes.
- Test: `tests/test_docs*.py` and the strict off-mode docs build already in CI.
  There are no new test files in this task.

**Interfaces:**
- Consumes: everything from Tasks H1–G5.
- Produces: nothing new.

- [ ] **Step 1: Update the tutorial.** Follow the docs plain-language rule:
  - no status jargon on user pages;
  - one limit per bullet;
  - user messages are neutral and actionable;
  - name every UI control exactly as the Help text and hover popovers name it.

  Add the CV example exactly:

  ```python
  from sklearn.model_selection import KFold
  from superglm import cross_validate
  from superglm.editor import EditorSession

  cv = cross_validate(
      model_template, X_train, y_train, cv=KFold(5, shuffle=True, random_state=7),
      sample_weight=w_train, scoring=("deviance", "gini", "nll"), return_estimators=True,
  )
  session = EditorSession.from_model(
      model, train_data=(X_train, y_train, w_train),
      validation_data=(X_val, y_val, w_val), cv=cv,
  )
  session.widget()
  ```

  Add one sentence each:
  - "Run CV on current model" refits the folds the result came from, with your
    hand edits put back as you set them, and it waits while changes are waiting
    for Refit;
  - "Final fit on all rows" uses the train and validation rows and leaves the
    test rows out.

- [ ] **Step 2: Update the internals page** with the module list, the settings
  key `superglm.editor.settings`, the `pending` payload fields, and the routes
  `/stage`, `/refit_pending`, `/note`, `/rating_table`, `/job_start`,
  `/job_status` and `/job_cancel`.

- [ ] **Step 3: Run the strict docs build.**

  Run: `./.venv/bin/python -m pytest tests -m docs -q`. Then run the off-mode
  build exactly as `.github/workflows/dev-ci.yml` runs it: read the docs job's
  command from that file and run it verbatim.

  Expected: PASS, no warnings.

- [ ] **Step 4: Run the full verification.** Run each and keep the outputs for
  the PR body:

  ```bash
  ./.venv/bin/python scripts/run_test_suite.py
  ./.venv/bin/python -m pytest tests/test_editor_browser.py tests/editor -m browser --run-browser -q
  npm run check:frontend
  ./.venv/bin/python -m ruff check src/ tests/
  ./.venv/bin/python -m ruff format --check src/ tests/
  uv lock --check
  uv pip check
  ./.venv/bin/python run_test.py
  ./.venv/bin/python scripts/verify_release_artifacts.py --help
  ```

  Expected:
  - the suite passes, apart from known dataset skips; report the counts;
  - browser, frontend, ruff and lock all pass;
  - `verify_release_artifacts` lists every new `.js`/`.css` file (run it per its
    `--help` against a built wheel and sdist: `uv build`, then the script).

- [ ] **Step 4b: Record durations for the new tests.**
  `tests/test_ci_contracts.py::test_duration_manifest_covers_the_non_browser_suite`
  needs ≥ 95% of non-browser test ids to carry a recorded duration.
  - Record the whole uncovered set, not only this branch's tests:
    `./.venv/bin/python -m pytest <every uncovered test file> --store-durations -q`.
    Use `--store-durations`, which merges into `.test_durations`. Never use
    `--durations=N`.
  - Then run `./.venv/bin/python -m pytest tests/test_ci_contracts.py -k duration_manifest -q`.
    Expected: PASS.
  - Commit `.test_durations`.

- [ ] **Step 5: Run the end-to-end timing pair.** The fit count is the claim;
  wall time is context only. Run on a quiet machine: pin all thread pools, and
  take the benchboard timing lease if it is in use.
  - Fit the freMTPL2 sample from the mock-ups. The script is the session
    scratchpad's `gen_data.py`; copy its model block into a throwaway script
    outside the repo.
  - **Master:** open the session, then call `replace_with_collapsed_levels` on
    VehBrand three times (three fits).
  - **Branch:** call `stage_structural("collapse", …)` three times, then
    `refit_pending()` once (one fit).
  - Count fits by wrapping `superglm.editor.refit.fit_refit_model` with a
    counter.

  Expected: master 3 fits, branch 1 fit. Report both wall times with the thread
  pinning used. Two runs each is enough; this is not a benchmark block.

- [ ] **Step 6: Run the whole-branch review.** Use `/code-review high` on the
  branch diff against `origin/master` (Opus 5.5, xhigh). Then fix the confirmed
  findings, each with a regression test that fails before the fix.

- [ ] **Step 7: Commit the docs.**

  ```bash
  git add docs/tutorials/edit-a-model-in-the-browser.md docs/development/internals/editor-frontend.md
  git commit -m "Editor docs: staged refit, gestures, inspector, settings, CV tab, rating preview, theme switch

  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
  ```

- [ ] **Step 8: Hand over to Max.** Report:
  - the commits;
  - the suite counts;
  - the fit-count evidence;
  - how to try it: `uv run --with jupyterlab jupyter lab` in the worktree.

  Ask before pushing or opening the PR. One PR, `release:minor`.

  Also propose the roadmap line from spec §8 and add it only on his yes.
