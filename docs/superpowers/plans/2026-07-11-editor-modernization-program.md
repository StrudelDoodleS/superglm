# Editor Modernization Program Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver the approved SuperGLM analyst editor modernization as four independently testable phases without replacing the native JavaScript/SVG frontend.

**Architecture:** Python remains authoritative and serves readable ES modules to the notebook iframe. The program first establishes semantic revisions and a tested browser store, then modernizes the workspace and plot geometry, then makes structural transitions atomic and evidence scoring bounded/off-lock, and finally verifies accessibility, documentation, packaging, and CI as one product.

**Tech Stack:** Python 3.10+, FastAPI, pytest, NumPy/pandas, native ES modules with JSDoc and TypeScript `checkJs`, Node's built-in test runner, SVG, CSS Grid/Flexbox, and Python Playwright.

---

## Program Inputs

- Design: `docs/superpowers/specs/2026-07-10-editor-workspace-modernization-design.md`
- Phase 1: `docs/superpowers/plans/2026-07-11-editor-foundation-state.md`
- Phase 2: `docs/superpowers/plans/2026-07-11-editor-workspace-axis.md`
- Phase 3: `docs/superpowers/plans/2026-07-11-editor-refit-evidence.md`
- Phase 4: `docs/superpowers/plans/2026-07-11-editor-integration-docs.md`
- Worktree: `/home/mhick/python_projects/superglm/.worktrees/editor-refit-timing-debugging`
- Baseline design commit: `53bab60`

## Interfaces Locked Across Phases

Later phases may extend these contracts, but they must not silently rename or reinterpret them:

```text
Python state
  model_revision: integer changed only by prediction/evidence-changing operations
  selected_term, terms, selection, history, can_uncollapse_levels, last_collapse

Structural transition
  { state, summary, timing }
  state and summary describe the same committed model_revision

Browser store
  remote: authoritative Python snapshot
  view: mode/zoom/group-display/inspector/help preferences
  request.mutation: idle | running | error
  request.evidence.<panel>: idle | updating | current | stale | error

Evidence payload
  model_revision, available, and panel-specific aggregate data
  late results are rejected by model_revision plus a per-panel request sequence
```

The SVG renderer remains imperative. Python remains the only numerical scoring engine. Persistent
evaluation caches store scalars and one current edited-model materialization, never copied row data,
dense design matrices, or historical prediction vectors.

## Design Traceability

| Approved requirement | Owning implementation task |
|---|---|
| Native modules, gradual typing, executable tests, Python-friendly structure | Foundation Tasks 1-9 |
| Semantic revision rules and stale response protection | Foundation Tasks 3-7; Refit Tasks 9A/10A |
| One store/action path and rollback after failed previews | Foundation Tasks 5-8; Integration Task 1 |
| Analyst app/context bars, visible Undo/Redo/Save | Workspace Task 5 |
| Select/Move/Zoom/Handles rail and retained SVG operation palette | Workspace Task 6 |
| 350 ms hover, immediate leave, focus/Escape popovers and shared help copy | Workspace Task 4 |
| Summary/History/Advanced/Help in one responsive inspector slot | Workspace Tasks 7-8 |
| Notebook, narrow, and short-window proportions | Workspace Task 8 |
| Display-only categorical ellipsis, full labels, tick density, dynamic gutter | Workspace Tasks 9-10 |
| Atomic `{state, summary, timing}` transition with no successful follow-up `/state` | Refit Tasks 2/3A |
| Confirmation only when a structural refit discards active history | Refit Task 2B |
| One current materialized model without copied row frames/design matrix | Refit Tasks 4/8A |
| Scalar cache, fitted-artifact reuse, metric/report sharing | Refit Tasks 5-6 |
| One off-lock cache miss, same-key de-duplication, latest-revision coalescing | Refit Tasks 7/8A |
| Paint-before-background-evidence, freshness states, selective rendering | Refit Tasks 9A/10A |
| Persistent recovery, inert state, focus/keyboard/reduced motion | Workspace Task 11; Integration Tasks 1-3 |
| Analyst and Python-developer guides, screenshot, navigation, CI | Integration Tasks 4-8 |

Checked prerequisite or superseded tasks inside a phase plan are documentary only. Ralph workers
execute unchecked tasks; the explicit replacement task (`3A`, `8A`, `9A`, or `10A`) is authoritative.

### Task 1: Establish the Baseline and Execution Ledger

**Files:**
- Read: `docs/superpowers/specs/2026-07-10-editor-workspace-modernization-design.md`
- Read: all four phase plans listed above
- Inspect: `src/superglm/editor/`, `src/superglm/editor/app/`, `tests/test_editor.py`
- Preserve: `docs/notebooks/spline diagnostics/`

- [ ] **Step 1: Confirm the branch and worktree scope**

Run:

```bash
rtk git status --short --branch
rtk git log -3 --oneline
```

Expected: branch `feature/editor-refit-timing-debugging`, design commit `53bab60` in recent history,
and the pre-existing `docs/notebooks/spline diagnostics/` directory still untracked and untouched.

- [ ] **Step 2: Read the complete design and every phase plan**

Run:

```bash
rtk read docs/superpowers/specs/2026-07-10-editor-workspace-modernization-design.md
rtk read docs/superpowers/plans/2026-07-11-editor-foundation-state.md
rtk read docs/superpowers/plans/2026-07-11-editor-workspace-axis.md
rtk read docs/superpowers/plans/2026-07-11-editor-refit-evidence.md
rtk read docs/superpowers/plans/2026-07-11-editor-integration-docs.md
```

Expected: no missing plan and no unresolved placeholder.

- [ ] **Step 3: Record a fresh focused baseline**

Run:

```bash
rtk test uv run pytest tests/test_editor.py -q
rtk ruff check src/superglm/editor tests/test_editor.py
```

Expected: both commands pass before implementation begins. If the baseline is red, record the exact
pre-existing failure before changing production code.

### Task 2: Execute Phase 1 — Foundation and State

**Files:**
- Follow: `docs/superpowers/plans/2026-07-11-editor-foundation-state.md`
- Verify: `src/superglm/editor/session.py`
- Verify: `src/superglm/editor/widget.py`
- Verify: `src/superglm/editor/app/state/`
- Verify: `src/superglm/editor/app/api/contracts.js`
- Verify: `tests/editor_frontend/`

- [ ] **Step 1: Execute every unchecked Phase 1 task in order**

Use a fresh implementation subagent for each task and apply the two-stage review required by
`superpowers:subagent-driven-development`: specification compliance first, then code quality.

- [ ] **Step 2: Run the Phase 1 gate**

Run:

```bash
rtk npm run test:frontend
rtk npm run typecheck:frontend
rtk test uv run pytest tests/test_editor.py -q
rtk ruff check src/superglm/editor tests/test_editor.py
```

Expected: semantic revision tests, store/selectors tests, stale-response tests, and the existing
editor suite pass. There is one state/action path and no visual redesign yet.

- [ ] **Step 3: Inspect Phase 1 scope and checkpoint commit**

Run:

```bash
rtk git diff --check
rtk git status --short
rtk git log --oneline 53bab60..HEAD
```

Expected: only Phase 1 files changed, no generated Node/browser output is tracked, and each completed
task has its own focused commit.

### Task 3: Execute Phase 2 — Workspace and Adaptive Axis

**Files:**
- Follow: `docs/superpowers/plans/2026-07-11-editor-workspace-axis.md`
- Verify: `src/superglm/editor/app/index.html`
- Verify: `src/superglm/editor/app/styles/`
- Verify: `src/superglm/editor/app/chart/geometry.js`
- Verify: `src/superglm/editor/app/chart.js`
- Verify: `src/superglm/editor/app/views/`

- [ ] **Step 1: Execute every unchecked Phase 2 task in order**

Preserve the existing operation identifiers and SVG curve-editing behavior. Keep label truncation
display-only and keep full category strings in Python payloads, selection identity, tooltips, and
accessible names.

- [ ] **Step 2: Run the Phase 2 gate**

Run:

```bash
rtk npm run test:frontend
rtk npm run typecheck:frontend
rtk test uv run pytest tests/test_editor_browser.py tests/editor/ -m browser --run-browser -q
rtk test uv run pytest tests/test_editor.py -q
```

Expected: 1180x720, narrow, and short-window tests pass; long category labels remain inside the SVG
and do not intersect the x-axis title; tool popovers obey 350 ms hover/immediate-leave/focus/Escape
behavior; existing edit operations still pass.

- [ ] **Step 3: Inspect Phase 2 scope and checkpoint commits**

Run:

```bash
rtk git diff --check
rtk git status --short
rtk git log --oneline 53bab60..HEAD
```

Expected: no framework, Tailwind, bundler, generated production artifact, or mutated test fixture was
introduced.

### Task 4: Execute Phase 3 — Atomic Refit and Evidence Performance

**Files:**
- Follow: `docs/superpowers/plans/2026-07-11-editor-refit-evidence.md`
- Verify: `src/superglm/editor/evaluation_cache.py`
- Verify: `src/superglm/editor/evidence.py`
- Verify: `src/superglm/editor/metrics.py`
- Verify: `src/superglm/editor/reports.py`
- Verify: `src/superglm/editor/widget.py`
- Verify: `src/superglm/editor/app/state/actions.js`

- [ ] **Step 1: Execute every unchecked Phase 3 task in order**

Do not remove the global refit overlay until cache misses score outside the widget mutation lock.
Keep one active row-scale evidence computation and at most one coalesced latest pending request per
widget.

- [ ] **Step 2: Run the Phase 3 correctness gate**

Run:

```bash
rtk test uv run pytest tests/test_editor_evaluation_cache.py tests/test_editor_evidence.py -q
rtk test uv run pytest tests/test_editor.py -q
rtk npm run test:frontend
rtk npm run typecheck:frontend
```

Expected: cache reuse/invalidation, one materialization per edit epoch, transition consistency,
off-lock scoring, in-flight de-duplication, latest-revision coalescing, and stale browser response
rejection all pass.

- [ ] **Step 3: Run the Phase 3 request-order gate**

Run:

```bash
rtk test uv run pytest tests/test_editor_browser.py tests/editor/ -m browser --run-browser -q
```

Expected: structural responses contain state and summary, no follow-up `/state` is made, chart and
summary commit before delayed metrics, and completing metrics does not redraw the SVG.

### Task 5: Execute Phase 4 — Integration, Accessibility, and Documentation

**Files:**
- Follow: `docs/superpowers/plans/2026-07-11-editor-integration-docs.md`
- Verify: `docs/guide/editor.md`
- Verify: `docs/editor_frontend.md`
- Verify: `mkdocs.yml`
- Verify: `.github/workflows/dev-ci.yml`

- [ ] **Step 1: Execute every unchecked Phase 4 task in order**

Complete browser-level recovery/accessibility coverage, publish analyst and Python-maintainer guides,
capture the deterministic editor screenshot, and wire the dedicated browser job without adding
Playwright to ordinary runtime dependencies.

- [ ] **Step 2: Run the documentation and browser gate**

Run:

```bash
rtk test uv run mkdocs build --strict
rtk test uv run pytest tests/test_editor_browser.py tests/editor/ -m browser --run-browser -q
rtk npm run test:frontend
rtk npm run typecheck:frontend
```

Expected: documentation navigation resolves, the browser suite passes in Chromium, and all frontend
modules pass tests and `checkJs`.

### Task 6: Final Program Verification and Handoff

**Files:**
- Verify: all changed files from `53bab60..HEAD`
- Preserve: `docs/notebooks/spline diagnostics/`

- [ ] **Step 1: Run the complete automated verification matrix**

Run:

```bash
rtk test uv run pytest tests/ -q
rtk ruff check src/ tests/
rtk test uv run ruff format --check src/ tests/
rtk mypy src/
rtk npm run test:frontend
rtk npm run typecheck:frontend
rtk test uv run mkdocs build --strict
```

Expected: every command exits zero. Report any environment-dependent skip explicitly; do not call the
program complete when the dedicated browser suite has not run in an environment with Chromium.

- [ ] **Step 2: Run the editor-specific acceptance suite**

Run:

```bash
rtk test uv run pytest tests/test_editor.py tests/test_editor_evaluation_cache.py tests/test_editor_evidence.py tests/test_editor_browser.py tests/editor/ --run-browser -q
```

Expected: all editor acceptance tests pass, including browser tests when the browser extra and
Chromium are installed.

- [ ] **Step 3: Audit the final diff against the design acceptance criteria**

Run:

```bash
rtk git diff --check 53bab60..HEAD
rtk git diff --stat 53bab60..HEAD
rtk git status --short --branch
```

Expected: no whitespace errors, no unrelated user files, and the only remaining untracked path is the
pre-existing `docs/notebooks/spline diagnostics/` directory unless the user changed the worktree.

- [ ] **Step 4: Use the branch-finishing workflow**

Invoke `superpowers:requesting-code-review`, address confirmed findings, rerun the complete matrix,
then invoke `superpowers:finishing-a-development-branch` to present merge/PR/cleanup choices. Do not
push or open a PR without the user's authorization.
