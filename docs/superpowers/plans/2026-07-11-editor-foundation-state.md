# Editor Foundation and State Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the native editor one typed browser store and one ordinary-action controller, backed by semantic Python model revisions and executable JavaScript/browser tests, without changing the visible workspace or SVG editing engine.

**Architecture:** `EditorSession` remains authoritative and exposes a semantic `model_revision` through every state snapshot. Native ES modules hold immutable `remote`, `view`, and `request` state; pure selectors feed the existing renderers; an injected API client and action controller own ordinary state-returning routes, rollback, duplicate suppression, evidence sequencing, and revision-aware refresh. `main.js` adopts these pieces incrementally while `chart.js`, the structural-refit flow, and the current panel renderers remain operational.

**Tech Stack:** Python 3.10+, FastAPI/Uvicorn, native JavaScript ES modules, JSDoc with TypeScript `checkJs`, Node 24 built-in test runner, pytest, Python Playwright/Chromium, Hatchling package resources.

---

## Scope boundaries

This plan includes only:

- development-only JavaScript type/test tooling and a Python Playwright browser-test job;
- JSDoc contracts, store, selectors, API client, and ordinary-action controller;
- semantic `model_revision`/`edit_epoch` plumbing in `EditorSession` and
  `EditorWidget._state()`;
- immutable local preview/rollback support needed by ordinary drag/control actions;
- incremental `main.js` integration for `/state`, `/term`, `/select`, `/op`, `/drag`, `/control`, `/control_count`, `/reorder_levels`, `/metrics`, and `/report`.

The following remain separate plans: structural transition envelopes for collapse/ungroup/uncollapse, distribution-profile final commits, evaluation caching/workers, workspace layout/tool rail/help, CSS decomposition, adaptive categorical labels, and documentation expansion. Until the structural-transition plan lands, `runStructuralRefit()` and `summary.js` retain their current structural route handling.

## Dependency order

```text
tooling + browser harness
  -> contracts
  -> semantic Python revision
  -> store + selectors
  -> injectable API client
  -> ordinary-action controller
  -> main.js integration
  -> immutable interaction preview + recovery browser coverage
  -> final package/CI verification
```

## File map

**Create**

- `package.json` — development-only frontend commands and dependencies.
- `package-lock.json` — locked TypeScript/Node type dependencies.
- `jsconfig.json` — strict `checkJs` scope for new foundation modules and tests.
- `src/superglm/editor/app/api/contracts.js` — JSDoc payload/state/action contracts.
- `src/superglm/editor/app/api/client.js` — injectable token-aware HTTP client.
- `src/superglm/editor/app/state/store.js` — immutable state transitions and subscriptions.
- `src/superglm/editor/app/state/selectors.js` — pure derived editor values.
- `src/superglm/editor/app/state/actions.js` — ordinary mutation/evidence orchestration.
- `tests/editor_frontend/contracts.test.js`
- `tests/editor_frontend/store.test.js`
- `tests/editor_frontend/client.test.js`
- `tests/editor_frontend/actions.test.js`
- `tests/test_editor_browser.py`

**Modify**

- `.gitignore` — ignore `node_modules/` and Playwright output.
- `pyproject.toml` — Playwright dev dependency and `browser` pytest marker.
- `.github/workflows/ci.yml` — exclude browser tests from the Python matrix and add one frontend/browser job.
- `.github/workflows/dev-ci.yml` — run frontend checks and exclude browser tests from existing Python jobs.
- `src/superglm/editor/session.py` — semantic revision counter and exact bump points.
- `src/superglm/editor/widget.py` — include `model_revision` in `_state()`.
- `src/superglm/editor/app/api.js` — compatibility re-export during migration.
- `src/superglm/editor/app/main.js` — compose store/client/actions and remove ordinary route heuristics.
- `src/superglm/editor/app/interactions.js` — use store-backed mode/zoom/actions and isolate previews from confirmed state.
- `src/superglm/editor/app/metrics.js` — export rendering separately from fetching.
- `src/superglm/editor/app/reports.js` — export rendering separately from fetching.
- `tests/test_editor.py` — revision/state HTTP tests and nested asset assertions.
- `.github/workflows/security.yml` — require nested state/API assets in built wheel and sdist.

## Route policy locked by this plan

| Route/action | Response | Revision effect | Controller behavior |
|---|---|---|---|
| `GET /state` | `EditorSnapshot` | none | initial/recovery snapshot |
| `POST /term` | `EditorSnapshot` | stable | commit remote and active term |
| `POST /select` | `EditorSnapshot` | stable | commit selection |
| `POST /op` `select_all`, `reset_order` | `EditorSnapshot` | stable | commit without evidence refresh |
| `POST /op` coefficient/reset/undo/redo operations | `EditorSnapshot` | increment only if fitted predictions can change | schedule visible evidence when revision differs |
| `POST /drag`, `/control` | `EditorSnapshot` | increment only if values change | clear preview, commit, schedule evidence |
| `POST /control_count`, `/reorder_levels` | `EditorSnapshot` | stable | commit display state only |
| `POST /metrics`, `/report` | existing payload | none | tag locally with starting revision and per-panel sequence; ignore stale completion |

The controller determines evidence invalidation by comparing returned and confirmed revisions. It must not reproduce `REFIT_INVALIDATING_ROUTES` or operation-name refresh lists.

### Task 1: Add development-only JavaScript checks

**Files:**
- Create: `package.json`
- Create: `package-lock.json`
- Create: `jsconfig.json`
- Create: `tests/editor_frontend/contracts.test.js`
- Modify: `.gitignore:1-23`

- [ ] **Step 1: Write a frontend smoke test before declaring the repository an ES-module package**

```javascript
// tests/editor_frontend/contracts.test.js
import assert from "node:assert/strict";
import test from "node:test";

test("frontend tests execute as native ES modules", () => {
  assert.deepEqual({ runner: "node", modules: "esm" }, {
    runner: "node",
    modules: "esm"
  });
});
```

- [ ] **Step 2: Run the repository frontend command and verify the missing package fails**

Run: `rtk npm run check:frontend`

Expected: FAIL because `package.json` and the `check:frontend` script do not exist.

- [ ] **Step 3: Add the package and strict gradual-check configuration**

```json
// package.json
{
  "name": "superglm-editor-development",
  "private": true,
  "type": "module",
  "engines": {
    "node": ">=24"
  },
  "scripts": {
    "test:frontend": "node --test tests/editor_frontend/*.test.js",
    "typecheck:frontend": "tsc -p jsconfig.json",
    "check:frontend": "npm run typecheck:frontend && npm run test:frontend"
  },
  "devDependencies": {
    "@types/node": "^24.13.3",
    "typescript": "^7.0.2"
  }
}
```

```json
// jsconfig.json
{
  "compilerOptions": {
    "allowJs": true,
    "checkJs": true,
    "noEmit": true,
    "strict": true,
    "target": "ES2022",
    "module": "NodeNext",
    "moduleResolution": "NodeNext",
    "lib": ["ES2022", "DOM", "DOM.Iterable"],
    "types": ["node"],
    "skipLibCheck": true
  },
  "include": [
    "src/superglm/editor/app/api/**/*.js",
    "src/superglm/editor/app/state/**/*.js",
    "tests/editor_frontend/**/*.js"
  ]
}
```

Append to `.gitignore`:

```gitignore
# Frontend development
node_modules/
playwright-report/
test-results/
```

- [ ] **Step 4: Generate the lockfile and run the frontend check**

Run: `rtk npm install --package-lock-only`

Expected: `package-lock.json` is created without installing a runtime dependency.

Run: `rtk npm ci && rtk npm run check:frontend`

Expected: PASS, one Node test.

- [ ] **Step 5: Commit the tooling foundation**

```bash
rtk git add package.json package-lock.json jsconfig.json .gitignore tests/editor_frontend/contracts.test.js
rtk git commit -m "Add editor frontend development checks"
```

### Task 2: Add a real-browser test harness without entering the UI redesign

**Files:**
- Create: `tests/test_editor_browser.py`
- Modify: `tests/conftest.py`
- Modify: `pyproject.toml:23-32,87-96`
- Modify: `.github/workflows/ci.yml:16-68`
- Modify: `.github/workflows/dev-ci.yml:15-59`

- [ ] **Step 1: Add a browser-marked characterization test**

```python
# tests/test_editor_browser.py
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from playwright.sync_api import sync_playwright

from superglm import Categorical, Spline, SuperGLM
from superglm.editor import EditorSession


@pytest.fixture
def browser_editor_widget():
    rng = np.random.default_rng(20260711)
    n = 120
    X = pd.DataFrame(
        {
            "age": rng.uniform(18.0, 80.0, n),
            "region": rng.choice(["A", "B", "C"], n),
        }
    )
    y = 0.3 + 0.01 * X["age"].to_numpy() + 0.2 * (X["region"] == "B")
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"age": Spline(n_knots=7), "region": Categorical(base="first")},
    )
    model.fit(X, y)
    widget = EditorSession.from_model(model, terms=["age", "region"]).widget()
    try:
        yield widget
    finally:
        widget.close()


@pytest.mark.browser
def test_editor_browser_loads_authoritative_state(browser_editor_widget):
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1180, "height": 720})
        page.goto(browser_editor_widget.app_url)
        page.locator("#chart .edited").wait_for()
        assert page.locator("#term").input_value() == "age"
        assert page.locator("#status").get_attribute("style") in (None, "")
        browser.close()
```

- [ ] **Step 2: Run the focused browser test and verify the missing dependency fails**

Run: `rtk uv run pytest tests/test_editor_browser.py -m browser --run-browser -q`

Expected: FAIL during collection or launch because Playwright and Chromium are not installed.

- [ ] **Step 3: Add the development dependency and marker**

Add to `dev` in `pyproject.toml`:

```toml
    "playwright>=1.55",
```

Add to the existing marker list:

```toml
    "browser: runs real-browser editor integration tests",
```

Add an explicit opt-in so the repository's ordinary `pytest tests/` command skips browser tests
when Chromium is not installed:

```python
def pytest_addoption(parser):
    parser.addoption("--run-browser", action="store_true", help="run Playwright editor tests")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-browser"):
        return
    skip = pytest.mark.skip(reason="pass --run-browser to run Playwright editor tests")
    for item in items:
        if "browser" in item.keywords:
            item.add_marker(skip)
```

- [ ] **Step 4: Install Chromium and verify the characterization test**

Run: `rtk uv sync --extra dev && rtk uv run playwright install chromium`

Run: `rtk uv run pytest tests/test_editor_browser.py -m browser --run-browser -q`

Expected: PASS, one browser test.

- [ ] **Step 5: Add one dedicated frontend job and keep browsers out of the Python matrix**

Change the existing Python test commands in `ci.yml` and `dev-ci.yml` to include `-m "not browser"`. Add this job to `ci.yml` and the same setup/check steps to the `quick-check` job in `dev-ci.yml`:

```yaml
  frontend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0
      - uses: actions/setup-node@a0853c24544627f65ddf259abe73b1d18a591444
        with:
          node-version: "24"
          cache: npm
      - name: Install uv
        uses: astral-sh/setup-uv@fac544c07dec837d0ccb6301d7b5580bf5edae39
      - name: Set up Python
        run: uv python install 3.13
      - name: Install dependencies
        run: uv sync --python 3.13 --extra dev
      - name: Install frontend dependencies
        run: npm ci
      - name: Check frontend modules
        run: npm run check:frontend
      - name: Install Chromium
        run: uv run playwright install --with-deps chromium
      - name: Browser tests
        run: uv run pytest tests/test_editor_browser.py -m browser --run-browser -q
```

Also add `package.json`, `package-lock.json`, `jsconfig.json`, and
`tests/editor_frontend/**` to the `ci.yml` pull-request path filter so frontend-only changes
cannot skip the job.

- [ ] **Step 6: Commit the browser harness**

```bash
rtk git add pyproject.toml uv.lock .github/workflows/ci.yml .github/workflows/dev-ci.yml tests/conftest.py tests/test_editor_browser.py
rtk git commit -m "Add editor browser test harness"
```

### Task 3: Define runtime-free editor contracts

**Files:**
- Create: `src/superglm/editor/app/api/contracts.js`
- Modify: `tests/editor_frontend/contracts.test.js`

- [ ] **Step 1: Extend the contract test with the required durable state sections**

```javascript
import {
  EVIDENCE_PANELS,
  createEmptyEvidenceState
} from "../../src/superglm/editor/app/api/contracts.js";

test("foundation contracts define all evidence panels", () => {
  assert.deepEqual(EVIDENCE_PANELS, ["metrics", "summary", "report"]);
  assert.deepEqual(createEmptyEvidenceState(), {
    status: "idle",
    revision: null,
    sequence: 0,
    payload: null,
    error: null,
    retry: null
  });
});
```

- [ ] **Step 2: Run the contract test and verify the missing module fails**

Run: `rtk npm run test:frontend`

Expected: FAIL with `ERR_MODULE_NOT_FOUND` for `api/contracts.js`.

- [ ] **Step 3: Add the JSDoc contracts and small runtime constants**

```javascript
// @ts-check

/** @typedef {'editor'|'validation'|'final'} AppView */
/** @typedef {'select'|'move'|'zoom'|'handles'} EditorMode */
/** @typedef {'idle'|'running'|'error'} MutationStatus */
/** @typedef {'idle'|'updating'|'current'|'stale'|'error'} EvidenceStatus */
/** @typedef {'metrics'|'summary'|'report'} EvidencePanel */
/**
 * @typedef {Object} EditorHistory
 * @property {Array<Record<string, unknown>>} active
 * @property {Array<Record<string, unknown>>} redo
 */
/**
 * @typedef {Object} GroupDisplayPayload
 * @property {boolean} available
 * @property {string} default_mode
 * @property {Record<string, unknown>|null} collapsed
 */
/**
 * @typedef {Object} TermPayload
 * @property {string} kind
 * @property {string} term_type
 * @property {number[]} x
 * @property {number[]} y
 * @property {number[]} original_y
 * @property {number[]|null} previous_y
 * @property {string[]|null} levels
 * @property {number} n_points
 * @property {Record<string, unknown>|null} controls
 * @property {GroupDisplayPayload|null} group_display
 * @property {Record<string, unknown>} impact
 */
/**
 * @typedef {Object} EditorSnapshot
 * @property {number} model_revision
 * @property {string} selected_term
 * @property {Record<string, TermPayload>} terms
 * @property {Record<string, number[]>} selection
 * @property {boolean} can_uncollapse_levels
 * @property {Record<string, unknown>|null} last_collapse
 * @property {EditorHistory} history
 */
/**
 * @typedef {Object} EvidenceState
 * @property {EvidenceStatus} status
 * @property {number|null} revision
 * @property {number} sequence
 * @property {unknown} payload
 * @property {string|null} error
 * @property {{path:string, payload:Record<string, unknown>}|null} retry
 */
/**
 * @typedef {Object} EditorState
 * @property {{snapshot:EditorSnapshot|null}} remote
 * @property {{activeTerm:string, activeView:AppView, mode:EditorMode, showCi:boolean, showContrib:boolean, zoomByTerm:Record<string, unknown>, groupModeByTerm:Record<string,string>, inspectorPane:'summary'|'history'|'advanced'|'help', inspectorOpen:boolean, preview:{term:string, payload:TermPayload}|null}} view
 * @property {{mutation:{status:MutationStatus, operation:string|null, error:string|null}, evidence:Record<EvidencePanel,EvidenceState>, recovery:{message:string, retry:{name:string,path:string,payload:Record<string,unknown>}|null}|null, nextSequence:number}} request
 */
/** @typedef {{panel:EvidencePanel, revision:number, sequence:number}} EvidenceToken */
/** @typedef {{ok:true, snapshot:EditorSnapshot}|{ok:false, skipped?:boolean, error:Error}} ActionResult */

export const EVIDENCE_PANELS = /** @type {const} */ (["metrics", "summary", "report"]);

/** @returns {EvidenceState} */
export function createEmptyEvidenceState() {
  return {
    status: "idle",
    revision: null,
    sequence: 0,
    payload: null,
    error: null,
    retry: null
  };
}
```

Keep these typedefs at module scope so `import("./contracts.js").EditorSnapshot`-style JSDoc
imports resolve from the store, selectors, client, actions, and tests. Do not duplicate payload
shapes in consumer modules.

- [ ] **Step 4: Run type and unit checks**

Run: `rtk npm run check:frontend`

Expected: PASS.

- [ ] **Step 5: Commit the contracts**

```bash
rtk git add src/superglm/editor/app/api/contracts.js tests/editor_frontend/contracts.test.js
rtk git commit -m "Define editor frontend state contracts"
```

### Task 4: Add semantic model revision to authoritative snapshots

**Files:**
- Modify: `src/superglm/editor/session.py:48-73,231-245,467-487,880-896,1207-1230`
- Modify: `src/superglm/editor/widget.py:90-108`
- Modify: `tests/test_editor.py:158-177,2432-2445,2838-2868`

- [ ] **Step 1: Add failing session and HTTP revision tests**

```python
def test_editor_model_revision_changes_only_for_prediction_mutations(editor_model):
    session = EditorSession.from_model(editor_model, terms=["x_spline", "region"])
    assert session.model_revision == 0
    assert session.edit_epoch == 0

    session.select_indices("x_spline", [3, 4])
    assert session.model_revision == 0

    session.shift("x_spline", 0.1)
    assert session.model_revision == 1
    assert session.edit_epoch == 1

    session.undo()
    assert session.model_revision == 2
    assert session.edit_epoch == 2

    session.redo()
    assert session.model_revision == 3
    assert session.edit_epoch == 3

    session.select_indices("region", [1])
    session.reorder_levels("region", target_index=0)
    assert session.model_revision == 3


def test_widget_state_exposes_semantic_model_revision(editor_model):
    session = EditorSession.from_model(editor_model, terms=["x_spline"])
    widget = session.widget()
    try:
        initial = _get_json(f"{widget.url}/state")
        _post_json(f"{widget.url}/select", {"term": "x_spline", "indices": [5, 6]})
        selected = _get_json(f"{widget.url}/state")
        changed = _post_json(f"{widget.url}/op", {"operation": "shift_up"})
        assert initial["model_revision"] == selected["model_revision"] == 0
        assert changed["model_revision"] == 1
    finally:
        widget.close()
```

- [ ] **Step 2: Run focused tests and verify the missing property fails**

Run: `rtk uv run pytest tests/test_editor.py::test_editor_model_revision_changes_only_for_prediction_mutations tests/test_editor.py::test_widget_state_exposes_semantic_model_revision -q`

Expected: FAIL with missing `model_revision`.

- [ ] **Step 3: Add the counter and central bump helper**

Add in `EditorSession.__init__` and next to the selection helpers. The materialization slots are
initialized here so the evidence phase can reuse the same invalidation boundary without inventing
another epoch:

```python
self._model_revision = 0
self._edit_epoch = 0
self._materialized_edit_model = None
self._materialized_edit_epoch: int | None = None

@property
def model_revision(self) -> int:
    """Semantic revision for prediction- or fit-evidence-changing state."""
    return self._model_revision

@property
def edit_epoch(self) -> int:
    """Monotonic invalidation token for the current edited-model materialization."""
    return self._edit_epoch

def _advance_model_revision(self) -> None:
    self._model_revision += 1
    self._edit_epoch += 1
    self._materialized_edit_model = None
    self._materialized_edit_epoch = None
```

In `_commit()`, bump only when values differ:

```python
changed = not np.array_equal(before, after)
self.terms[term].edited_log_effect[indices] = after
# Keep the existing EditRecord append and redo-stack clear unchanged.
if changed:
    self._advance_model_revision()
```

In `reset()`, capture `before` before assignment and advance after history trimming only when it
differs from the restored values. In `undo()` and `redo()`, compare the current selected values with
the target record values and advance after applying an actual change. At the end of a successful
`replace_in_force_model()`, call `_advance_model_revision()` exactly once. Selection, term changes,
control-count changes, zoom/group display, and display-only level reordering never advance either
counter.

- [ ] **Step 4: Expose the revision in `EditorWidget._state()`**

Add the first entry of the returned dictionary:

```python
"model_revision": self.session.model_revision,
```

- [ ] **Step 5: Run revision and existing editor tests**

Run: `rtk uv run pytest tests/test_editor.py -q -m "not slow"`

Expected: PASS, including stable revisions for selection/display reorder and increments for edits/undo/redo.

- [ ] **Step 6: Commit semantic revision plumbing**

```bash
rtk git add src/superglm/editor/session.py src/superglm/editor/widget.py tests/test_editor.py
rtk git commit -m "Add semantic editor model revisions"
```

### Task 5: Implement the immutable store and selectors

**Files:**
- Create: `src/superglm/editor/app/state/store.js`
- Create: `src/superglm/editor/app/state/selectors.js`
- Create: `tests/editor_frontend/store.test.js`

- [ ] **Step 1: Write store, selector, and selective-subscription tests**

```javascript
// tests/editor_frontend/store.test.js
import assert from "node:assert/strict";
import test from "node:test";

import {
  commitRemote,
  createEditorStore,
  createInitialEditorState,
  patchView,
  setPreviewTerm
} from "../../src/superglm/editor/app/state/store.js";
import {
  selectActiveTermName,
  selectCurrentSelection,
  selectRenderableTerm
} from "../../src/superglm/editor/app/state/selectors.js";

function snapshot(revision = 0) {
  return {
    model_revision: revision,
    selected_term: "age",
    terms: {
      age: {
        kind: "spline", term_type: "spline", x: [1], y: [1], original_y: [1],
        previous_y: null, levels: null, n_points: 1, controls: null,
        group_display: null, impact: {}
      }
    },
    selection: { age: [0] },
    can_uncollapse_levels: false,
    last_collapse: null,
    history: { active: [], redo: [] }
  };
}

test("store keeps confirmed remote data separate from a chart preview", () => {
  const initial = createInitialEditorState(snapshot());
  const preview = { ...snapshot().terms.age, y: [1.4] };
  const next = setPreviewTerm(initial, "age", preview);
  assert.deepEqual(next.remote.snapshot?.terms.age.y, [1]);
  assert.deepEqual(selectRenderableTerm(next)?.y, [1.4]);
});

test("remote commit clears preview and preserves valid view state", () => {
  let state = createInitialEditorState(snapshot());
  state = patchView(state, { mode: "move", showCi: true });
  state = setPreviewTerm(state, "age", { ...snapshot().terms.age, y: [1.4] });
  state = commitRemote(state, snapshot(1));
  assert.equal(state.view.mode, "move");
  assert.equal(state.view.preview, null);
  assert.equal(selectActiveTermName(state), "age");
  assert.deepEqual(selectCurrentSelection(state), [0]);
});

test("selector subscriptions ignore unrelated state changes", () => {
  const store = createEditorStore(createInitialEditorState(snapshot()));
  let calls = 0;
  store.subscribe(selectActiveTermName, () => { calls += 1; });
  store.update((state) => patchView(state, { showCi: true }));
  assert.equal(calls, 0);
  store.update((state) => patchView(state, { activeTerm: "missing" }));
  assert.equal(calls, 0);
});
```

- [ ] **Step 2: Run the store tests and verify missing modules fail**

Run: `rtk node --test tests/editor_frontend/store.test.js`

Expected: FAIL with `ERR_MODULE_NOT_FOUND`.

- [ ] **Step 3: Implement immutable state and selector subscriptions**

`store.js` must export these exact symbols:

```javascript
// @ts-check
import { createEmptyEvidenceState } from "../api/contracts.js";

/** @typedef {import('../api/contracts.js').EditorSnapshot} EditorSnapshot */
/** @typedef {import('../api/contracts.js').EditorState} EditorState */
/** @typedef {import('../api/contracts.js').TermPayload} TermPayload */

/** @param {EditorSnapshot|null} snapshot @returns {EditorState} */
export function createInitialEditorState(snapshot = null) {
  return {
    remote: { snapshot },
    view: {
      activeTerm: snapshot?.selected_term || "",
      activeView: "editor",
      mode: "select",
      showCi: false,
      showContrib: false,
      zoomByTerm: {},
      groupModeByTerm: {},
      inspectorPane: "summary",
      inspectorOpen: true,
      preview: null
    },
    request: {
      mutation: { status: "idle", operation: null, error: null },
      evidence: {
        metrics: createEmptyEvidenceState(),
        summary: createEmptyEvidenceState(),
        report: createEmptyEvidenceState()
      },
      recovery: null,
      nextSequence: 1
    }
  };
}

/** @param {EditorState} initialState */
export function createEditorStore(initialState) {
  let state = initialState;
  /** @type {Set<any>} Internal erasure; the public subscribe method remains generic. */
  const subscriptions = new Set();

  /** @param {(state: EditorState) => EditorState} updater */
  function update(updater) {
    const previous = state;
    const next = updater(previous);
    if (next === previous) return;
    state = next;
    for (const subscription of subscriptions) {
      const selected = subscription.selector(state);
      if (!subscription.equals(selected, subscription.value)) {
        const oldValue = subscription.value;
        subscription.value = selected;
        subscription.listener(selected, oldValue);
      }
    }
  }

  /**
   * @template T
   * @param {(state: EditorState) => T} selector
   * @param {(value: T, previous: T) => void} listener
   * @param {(left: T, right: T) => boolean} [equals]
   */
  function subscribe(selector, listener, equals = Object.is) {
    const subscription = { selector, listener, equals, value: selector(state) };
    subscriptions.add(subscription);
    return () => subscriptions.delete(subscription);
  }

  return {
    getState: () => state,
    update,
    subscribe
  };
}

/** @param {EditorState} state @param {Partial<EditorState['view']>} patch */
export function patchView(state, patch) {
  return { ...state, view: { ...state.view, ...patch } };
}

/** @param {EditorState} state @param {string} term @param {TermPayload} payload */
export function setPreviewTerm(state, term, payload) {
  return patchView(state, { preview: { term, payload } });
}

/** @param {EditorState} state @param {EditorSnapshot} snapshot */
export function commitRemote(state, snapshot) {
  const activeTerm = snapshot.terms[state.view.activeTerm]
    ? state.view.activeTerm
    : snapshot.selected_term || Object.keys(snapshot.terms)[0] || "";
  return {
    ...state,
    remote: { snapshot },
    view: { ...state.view, activeTerm, preview: null }
  };
}
```

`selectors.js` must export:

```javascript
// @ts-check
/** @typedef {import('../api/contracts.js').EditorState} EditorState */
/** @typedef {import('../api/contracts.js').EvidencePanel} EvidencePanel */
/** @typedef {import('../api/contracts.js').EvidenceState} EvidenceState */

/** @param {EditorState} state */
export const selectSnapshot = (state) => state.remote.snapshot;
/** @param {EditorState} state */
export const selectModelRevision = (state) => state.remote.snapshot?.model_revision ?? -1;
/** @param {EditorState} state */
export function selectActiveTermName(state) {
  const snapshot = selectSnapshot(state);
  if (!snapshot) return "";
  return snapshot.terms[state.view.activeTerm]
    ? state.view.activeTerm
    : snapshot.selected_term || Object.keys(snapshot.terms)[0] || "";
}
/** @param {EditorState} state */
export function selectCurrentTerm(state) {
  const snapshot = selectSnapshot(state);
  return snapshot?.terms[selectActiveTermName(state)] ?? null;
}
/** @param {EditorState} state */
export function selectCurrentSelection(state) {
  const snapshot = selectSnapshot(state);
  return snapshot?.selection[selectActiveTermName(state)] ?? [];
}
/** @param {EditorState} state */
export function selectRenderableTerm(state) {
  const active = selectActiveTermName(state);
  return state.view.preview?.term === active ? state.view.preview.payload : selectCurrentTerm(state);
}
/** @param {EditorState} state */
export function selectGroupDisplayMode(state) {
  const active = selectActiveTermName(state);
  const term = selectCurrentTerm(state);
  return state.view.groupModeByTerm[active] || term?.group_display?.default_mode || "expanded";
}
/** @param {EditorState} state */
export const selectMutation = (state) => state.request.mutation;
/** @param {EvidencePanel} panel @returns {(state: EditorState) => EvidenceState} */
export const selectEvidence = (panel) => (state) => state.request.evidence[panel];
```

- [ ] **Step 4: Run frontend checks**

Run: `rtk npm run check:frontend`

Expected: PASS, including no notifications for unrelated selector values.

- [ ] **Step 5: Commit store and selectors**

```bash
rtk git add src/superglm/editor/app/state/store.js src/superglm/editor/app/state/selectors.js tests/editor_frontend/store.test.js
rtk git commit -m "Add editor browser store and selectors"
```

### Task 6: Add the injectable API client and ordinary-action controller

**Files:**
- Create: `src/superglm/editor/app/api/client.js`
- Create: `src/superglm/editor/app/state/actions.js`
- Create: `tests/editor_frontend/client.test.js`
- Create: `tests/editor_frontend/actions.test.js`
- Modify: `src/superglm/editor/app/api.js:1-37`

- [ ] **Step 1: Write client tests for token propagation and structured errors**

```javascript
// tests/editor_frontend/client.test.js
import assert from "node:assert/strict";
import test from "node:test";
import { createEditorClient, EditorAPIError } from "../../src/superglm/editor/app/api/client.js";

test("client attaches the widget token", async () => {
  let request = null;
  const client = createEditorClient({
    token: "secret",
    fetchImpl: async (url, options) => {
      request = { url, options };
      return new Response(JSON.stringify({ ok: true }), { status: 200 });
    }
  });
  assert.deepEqual(await client.postJSON("/op", { operation: "reset" }), { ok: true });
  assert.ok(request);
  assert.equal(new Headers(request.options.headers).get("X-SuperGLM-Editor-Token"), "secret");
});

test("client preserves status and payload on API errors", async () => {
  const client = createEditorClient({
    fetchImpl: async () => new Response(JSON.stringify({ error: "bad edit" }), { status: 400 })
  });
  await assert.rejects(
    () => client.getState(),
    (error) => error instanceof EditorAPIError && error.status === 400 && error.message === "bad edit"
  );
});
```

- [ ] **Step 2: Write action tests for revision refresh, rollback, duplicate suppression, and stale evidence**

```javascript
// tests/editor_frontend/actions.test.js
import assert from "node:assert/strict";
import test from "node:test";
import { createEditorActions } from "../../src/superglm/editor/app/state/actions.js";
import { createEditorStore, createInitialEditorState } from "../../src/superglm/editor/app/state/store.js";

/** @param {number} revision */
const snapshot = (revision) => ({
  model_revision: revision, selected_term: "age", terms: {}, selection: {},
  can_uncollapse_levels: false, last_collapse: null, history: { active: [], redo: [] }
});

test("successful mutation commits once and schedules evidence only for a new revision", async () => {
  const store = createEditorStore(createInitialEditorState(snapshot(0)));
  const scheduled = [];
  const actions = createEditorActions({
    store,
    client: { postJSON: async () => snapshot(1), getState: async () => snapshot(1) },
    scheduleEvidence: (revision) => scheduled.push(revision)
  });
  const result = await actions.executeStateMutation({ name: "shift", path: "/op", payload: { operation: "shift_up" } });
  assert.equal(result.ok, true);
  assert.equal(store.getState().remote.snapshot.model_revision, 1);
  assert.deepEqual(scheduled, [1]);
});

test("failed mutation restores confirmed state and records retry data", async () => {
  const store = createEditorStore(createInitialEditorState(snapshot(2)));
  const actions = createEditorActions({
    store,
    client: { postJSON: async () => { throw new Error("network down"); }, getState: async () => snapshot(2) }
  });
  const result = await actions.executeStateMutation({ name: "drag", path: "/drag", payload: { term: "age" } });
  assert.equal(result.ok, false);
  assert.equal(store.getState().remote.snapshot.model_revision, 2);
  assert.equal(store.getState().request.recovery.message, "network down");
  assert.equal(store.getState().request.recovery.retry.path, "/drag");
});

test("late evidence cannot replace a newer revision", async () => {
  /** @type {(value: unknown) => void} */
  let resolveEvidence = () => {};
  const store = createEditorStore(createInitialEditorState(snapshot(3)));
  const actions = createEditorActions({
    store,
    client: { postJSON: () => new Promise((resolve) => { resolveEvidence = resolve; }), getState: async () => snapshot(3) }
  });
  const pending = actions.refreshEvidence("metrics", "/metrics", {});
  store.update((state) => ({ ...state, remote: { snapshot: snapshot(4) } }));
  resolveEvidence({ available: true });
  await pending;
  assert.equal(store.getState().request.evidence.metrics.payload, null);
});
```

- [ ] **Step 3: Run the focused tests and verify missing modules fail**

Run: `rtk node --test tests/editor_frontend/client.test.js tests/editor_frontend/actions.test.js`

Expected: FAIL with missing `client.js` and `actions.js`.

- [ ] **Step 4: Implement the injected client and keep `api.js` as a compatibility facade**

`client.js` must export `EditorAPIError`, `createEditorClient`, `editorClient`, `requestJSON`, `postJSON`, and `requestBlob`. The factory accepts `{token, fetchImpl}`; `getState()` calls `requestJSON('/state')`; every request adds `X-SuperGLM-Editor-Token`; JSON errors retain HTTP status and parsed payload. `api.js` becomes:

```javascript
export {
  EditorAPIError,
  editorClient,
  postJSON,
  requestBlob,
  requestJSON
} from "./api/client.js";
```

- [ ] **Step 5: Implement the action-controller interface**

`actions.js` must export `createEditorActions({store, client, scheduleEvidence = () => {}})` with:

```javascript
initialize()
executeStateMutation({name, path, payload})
refreshEvidence(panel, path, payload)
retryMutation()
retryEvidence(panel)
dismissRecovery()
patchView(patch)
```

Import the contract typedefs with JSDoc and type every factory option, method parameter, return
value, callback, and Promise. `npm run typecheck:frontend` must pass without adding `@ts-ignore`,
`@ts-nocheck`, or widening public state/action values to `any`.

`executeStateMutation` must reject a second call while `request.mutation.status === 'running'`, retain the confirmed snapshot, commit returned state, clear preview, compare revisions, and invoke `scheduleEvidence(newRevision)` without awaiting it. On failure it attempts `client.getState()`, falls back to the confirmed snapshot, clears preview, and stores `{message, retry:{name,path,payload}}` without throwing. `refreshEvidence` allocates a per-panel sequence from `request.nextSequence`, captures the current revision, and applies success or failure only when both revision and latest sequence still match.

- [ ] **Step 6: Run all frontend checks**

Run: `rtk npm run check:frontend`

Expected: PASS, including rollback and stale-response tests.

- [ ] **Step 7: Commit client and controller**

```bash
rtk git add src/superglm/editor/app/api.js src/superglm/editor/app/api/client.js src/superglm/editor/app/state/actions.js tests/editor_frontend/client.test.js tests/editor_frontend/actions.test.js
rtk git commit -m "Add editor API client and action controller"
```

### Task 7: Integrate the store into `main.js` for ordinary actions

**Files:**
- Modify: `src/superglm/editor/app/main.js:1-157,353-385,454-507,696-873`
- Modify: `src/superglm/editor/app/metrics.js:1-74`
- Modify: `src/superglm/editor/app/reports.js:1-181`
- Modify: `tests/test_editor.py:3948-4488`
- Modify: `tests/test_editor_browser.py`

- [ ] **Step 1: Add a browser test proving duplicate ordinary actions are suppressed**

Intercept `/op`, delay the first `shift_up` response until two clicks have been issued, click `Select all`, and double-click `button[data-op='shift_up']`. Assert exactly one `shift_up` request reached FastAPI and the returned snapshot was committed. The current UI has no pending-action guard, so this is the browser seam for `request.mutation.status`.

- [ ] **Step 2: Run the test and verify current orchestration fails the new request/commit assertion**

Run: `rtk uv run pytest tests/test_editor_browser.py -m browser --run-browser -q`

Expected: FAIL because the current click handler submits both `shift_up` requests.

- [ ] **Step 3: Replace the ordinary global state and helpers with store/selectors**

Import `createEditorStore`, `createInitialEditorState`, `patchView`, selectors, `editorClient`, and `createEditorActions`. Replace `let state`, `showCi`, `showContrib`, `graphMode`, `zoomState`, and `groupDisplayModeByTerm` with one store. Keep animation frame/timer handles local because they are ephemeral resources, not durable state.

Use these exact adapters during the incremental phase:

```javascript
const store = createEditorStore(createInitialEditorState());
const actions = createEditorActions({
  store,
  client: editorClient,
  scheduleEvidence: () => {
    void refreshMetricsView();
    void refreshSummaryView();
    void refreshActiveReport();
  }
});

const selectedTerm = () => selectActiveTermName(store.getState());
const currentTerm = () => selectRenderableTerm(store.getState());
const currentSelection = () => new Set(selectCurrentSelection(store.getState()));
const activeGroupDisplayMode = () => selectGroupDisplayMode(store.getState());
```

`loadState()` becomes `await actions.initialize()`. `render()` reads `const state = store.getState()` and `const snapshot = state.remote.snapshot`; it must never read a select value, class name, or dialog attribute as state. Subscribe the existing full `render()` once during this task; selective per-view subscriptions come in the workspace plan.

- [ ] **Step 4: Route ordinary handlers through `actions.executeStateMutation`**

Replace `postJSONWithRefresh()` calls for `/term`, `/select`, `/op`, `/drag`, `/control`, `/control_count`, and `/reorder_levels`. Remove `REFIT_INVALIDATING_ROUTES`, `postJSONWithRefresh()`, and `isDisplayOnlyOperation()`. Evidence scheduling comes only from revision comparison. Preserve `runStructuralRefit()` and the three structural functions in `summary.js` unchanged.

- [ ] **Step 5: Separate evidence rendering from fetching**

Export `renderMetricGrid(payload, nodes)` and `renderReport(payload, nodes)` from their existing modules. `refreshMetricsView()` and `refreshActiveReport()` call `actions.refreshEvidence(...)`; store subscriptions pass accepted payloads to those render functions. Remove module-level `metricPayload` so last-confirmed/stale values live only in `request.evidence`.

- [ ] **Step 6: Replace source-string assertions with nested-asset and behavior assertions**

Update `test_widget_serves_editor_app_assets` to request `api/client.js`, `api/contracts.js`, `state/store.js`, `state/selectors.js`, and `state/actions.js`. Remove assertions tied to `postJSONWithRefresh`, `REFIT_INVALIDATING_ROUTES`, and direct legacy imports; the Node and Playwright tests now own those behaviors.

- [ ] **Step 7: Run focused and full checks**

Run: `rtk npm run check:frontend`

Run: `rtk uv run pytest tests/test_editor.py tests/test_editor_browser.py -q -m "not slow"`

Expected: PASS; ordinary edits commit returned state directly and structural refits still use the legacy path.

- [ ] **Step 8: Commit incremental integration**

```bash
rtk git add src/superglm/editor/app/main.js src/superglm/editor/app/metrics.js src/superglm/editor/app/reports.js tests/test_editor.py tests/test_editor_browser.py
rtk git commit -m "Route ordinary editor actions through the browser store"
```

### Task 8: Isolate chart previews and prove failure recovery

**Files:**
- Modify: `src/superglm/editor/app/interactions.js:1-281,545-677`
- Modify: `src/superglm/editor/app/main.js:103-114,696-719`
- Modify: `src/superglm/editor/app/index.html:9-31`
- Modify: `src/superglm/editor/app/styles.css:27-87`
- Modify: `tests/test_editor_browser.py`
- Modify: `tests/editor_frontend/actions.test.js`

- [ ] **Step 1: Add a browser test for failed optimistic drag recovery**

Use Playwright to select a chart point, switch to Move, capture the edited path's `d` attribute, intercept the next `/drag` request with a `500` JSON response, drag the point, and assert that the path returns to its confirmed `d` value. Assert the persistent recovery state is visible and a second drag is possible after dismissal.

- [ ] **Step 2: Run the browser test and verify current in-place mutation fails rollback**

Run: `rtk uv run pytest tests/test_editor_browser.py -m browser --run-browser -q`

Expected: FAIL because `interactions.js` currently mutates `remote.terms[term].y` and control arrays in place.

- [ ] **Step 3: Change the interaction context to explicit store/action callbacks**

`bindInteractions(context)` must consume `mode()`, `currentTerm()`, `currentSelection()`, `setPreviewTerm(term, payload)`, `setZoom(term, range)`, `clearZoom(term)`, and `actions`. It must not read `modeSelect.value`, mutate `context.getState()`, mutate `context.zoomState`, or call a raw `postJSON` callback.

On point/control drag start, clone only the active `TermPayload` with `structuredClone`. Pointer moves mutate that preview copy and call `setPreviewTerm`; pointer-up calls `actions.executeStateMutation`. Successful and failed actions both clear the preview through store transitions. Selection, keyboard undo/redo, and reorder call action-controller methods.

- [ ] **Step 4: Keep transient gesture state local and add listener cleanup**

Brush, point/control drag, pan, box zoom, reorder preview, and pointer-capture state remain inside `bindInteractions`. Register named pointer/wheel/keydown handlers and return:

```javascript
return {
  resetZoomView: () => context.clearZoom(context.selectedTerm()),
  destroy() {
    svg.removeEventListener("pointerdown", onPointerDown);
    svg.removeEventListener("pointermove", onPointerMove);
    svg.removeEventListener("pointerup", onPointerUp);
    svg.removeEventListener("wheel", onWheel);
    document.removeEventListener("keydown", onKeyDown);
  }
};
```

- [ ] **Step 5: Render persistent recovery controls from request state**

Add immediately inside `.app-shell` in `index.html`:

```html
<div id="appAlert" class="app-alert" role="alert" hidden>
  <span id="appAlertMessage"></span>
  <button id="appAlertRetry" type="button">Retry</button>
  <button id="appAlertDismiss" type="button">Dismiss</button>
</div>
```

Add a store subscription in `main.js` that renders `request.recovery`, calls `actions.retryMutation()` for Retry, and calls `actions.dismissRecovery()` for Dismiss. Style the alert as an in-flow, high-contrast row; it must not cover the SVG or disappear on evidence-only errors.

- [ ] **Step 6: Add a unit assertion that confirmed state is unchanged after failure**

Extend `actions.test.js` so a preview payload is installed before a rejected `/drag`; after the result, assert `remote.snapshot` retains its original array identity/content, `view.preview === null`, and recovery retains the retry descriptor.

- [ ] **Step 7: Run frontend, browser, and Python editor tests**

Run: `rtk npm run check:frontend`

Run: `rtk uv run pytest tests/test_editor.py tests/test_editor_browser.py -q -m "not slow"`

Expected: PASS; no browser preview mutates the last-confirmed snapshot.

- [ ] **Step 8: Commit preview isolation**

```bash
rtk git add src/superglm/editor/app/interactions.js src/superglm/editor/app/main.js src/superglm/editor/app/index.html src/superglm/editor/app/styles.css tests/editor_frontend/actions.test.js tests/test_editor_browser.py
rtk git commit -m "Isolate editor previews from confirmed state"
```

### Task 9: Verify source packaging and the foundation boundary

**Files:**
- Modify: `.github/workflows/security.yml:96-133`
- Modify: `tests/test_editor.py:4442-4488`

- [ ] **Step 1: Add a failing archive requirement for nested source modules**

In the archive inspection script, add this helper and call it for each wheel and sdist after reading `names`:

```python
def require_editor_assets(names, archive):
    required = {
        "superglm/editor/app/api/client.js",
        "superglm/editor/app/api/contracts.js",
        "superglm/editor/app/state/store.js",
        "superglm/editor/app/state/selectors.js",
        "superglm/editor/app/state/actions.js",
    }
    missing = {suffix for suffix in required if not any(name.endswith(suffix) for name in names)}
    if missing:
        raise SystemExit(f"Missing editor source assets in {archive}: {sorted(missing)}")
```

- [ ] **Step 2: Build and inspect archives locally**

Run: `rtk uvx --from 'build==1.2.2.post1' python -m build`

Run: `rtk uvx --from 'check-wheel-contents==0.6.3' check-wheel-contents dist/*.whl`

Expected: PASS and nested native modules are present without a generated bundle.

- [ ] **Step 3: Run the complete proportional verification set**

Run: `rtk npm ci && rtk npm run check:frontend`

Run: `rtk uv run ruff check src/ tests/`

Run: `rtk uv run ruff format --check src/ tests/`

Run: `rtk uv run pytest tests/test_editor.py -q -m "not slow"`

Run: `rtk uv run playwright install chromium && rtk uv run pytest tests/test_editor_browser.py -m browser --run-browser -q`

Expected: all commands PASS.

- [ ] **Step 4: Confirm follow-on concerns did not leak into this foundation**

Run: `rtk git diff --name-only 53bab60..HEAD`

Expected: no structural-transition envelope, evaluation-cache/worker, workspace-layout, CSS-module split, or chart-label geometry files are present beyond the foundation paths listed above.

- [ ] **Step 5: Commit package verification**

```bash
rtk git add .github/workflows/security.yml tests/test_editor.py
rtk git commit -m "Verify packaged editor foundation assets"
```

## Handoff to follow-on plans

The next plan may rely on these stable seams:

- `EditorSnapshot.model_revision` is authoritative and semantic.
- confirmed remote payloads are never mutated by previews;
- `createEditorStore()` is the sole durable browser state owner;
- `createEditorActions()` is the sole ordinary mutation/evidence request owner;
- `selectRenderableTerm()` is the chart's preview-aware term boundary;
- evidence responses are protected by revision plus per-panel sequence;
- structural routes still use the legacy summary/refetch path and are ready for a dedicated transition-envelope plan.
