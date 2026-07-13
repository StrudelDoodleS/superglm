# Editor Workspace and Adaptive Axis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the approved analyst workspace, discoverable SVG tools and help, responsive inspector, accessible interaction semantics, and categorical axes whose display-only labels cannot collide with the x-axis title or leave the SVG viewport.

**Architecture:** Keep FastAPI/Python authoritative and keep the existing imperative SVG renderer. Add small native ES modules for view controls and pure axis geometry, with `main.js` remaining the composition root and `interactions.js` retaining transient pointer state. Browser integration is covered through the real widget server with Python Playwright tests; deterministic timing and geometry are covered with Node's built-in test runner.

**Tech Stack:** Python 3.10+, FastAPI, native ES modules, SVG, CSS Grid/Flexbox, Node built-in tests, TypeScript `checkJs`, pytest, Playwright Chromium.

---

## Scope and dependency contract

This plan owns the visible editor shell and plot layout only:

- design tokens and workspace CSS;
- application bar and context bar;
- Select, Move, Zoom, Handles, and Help rail;
- shared delayed popovers and selection-operation help;
- Summary, History, Advanced, and Help inspector panes;
- notebook-width, narrow-width, and short-window behavior;
- adaptive categorical ticks, full-label disclosure, and display-only ellipsis;
- keyboard/focus semantics, busy-state inertness, reduced motion, and text selection;
- pure frontend and real-browser regression coverage.

`docs/superpowers/plans/2026-07-11-editor-foundation-state.md` is a completed prerequisite. Its
contracts take precedence over legacy names retained in explanatory snippets below:

- `createEditorStore()` remains the only durable browser state owner;
- read authoritative data through `selectSnapshot(store.getState())`, never a new `state` global;
- read/write mode, CI, contribution, zoom, group, and inspector choices through `view` plus
  `actions.patchView()`, never `activeMode`, `graphMode`, `showCi`, `zoomState`, or another mutable
  module-level value;
- execute `/op`, `/term`, `/select`, `/drag`, `/control`, `/control_count`, and `/reorder_levels`
  through `actions.executeStateMutation({name, path, payload})`, never restore
  `postJSONWithRefresh()` or direct `postJSON()` calls;
- use `tests/editor_frontend/` and the existing `check:frontend` npm script;
- keep the existing `browser` marker/dependency and CI job; Task 2 expands its fixtures and test
  paths rather than creating a second browser job.

The view modules accept callbacks and plain render inputs, so they do not create a second state
source. When a code example below shows a legacy variable solely to explain a UI transformation,
the implementation worker must apply the exact mapping above in that same step.

The structural-refit/evidence workstream owns response envelopes, model revisions, metric caching,
and evidence workers. This plan only supplies semantic status/alert containers and makes the
primary editor regions inert while the existing busy overlay is active.

## File structure

### Create in this phase

- `src/superglm/editor/app/styles/tokens.css` — complete color, spacing, focus, and motion tokens.
- `src/superglm/editor/app/styles/shell.css` — application bar, context bar, tool rail, workspace grid,
  responsive drawer, and short-window scrolling.
- `src/superglm/editor/app/styles/chart.css` — chart surface, adaptive tick, selection palette, and
  popover rules touched by this work.
- `src/superglm/editor/app/styles/panels.css` — inspector/help/advanced and metric-strip layout.
- `src/superglm/editor/app/views/app_bar.js` — top tabs, visible undo/redo, shortcuts, and roving
  keyboard focus.
- `src/superglm/editor/app/views/context_bar.js` — analyst term metadata and context-control render.
- `src/superglm/editor/app/views/tool_rail.js` — exclusive tool selection and keyboard shortcuts.
- `src/superglm/editor/app/views/help_content.js` — one source of copy for tools, curve operations,
  shortcuts, refits, history, and saving.
- `src/superglm/editor/app/views/popover.js` — delayed pointer disclosure, immediate focus
  disclosure, immediate dismissal, viewport positioning, and Escape behavior.
- `src/superglm/editor/app/views/inspector.js` — inspector tabs, drawer state, responsive media
  changes, close behavior, and focus restoration.
- `src/superglm/editor/app/views/help_drawer.js` — renders the shared help catalog.
- `src/superglm/editor/app/chart/geometry.js` — pure tick selection, measured truncation, rotation,
  and bottom-gutter calculations.
- `tests/editor_frontend/popover.test.js` — deterministic popover timing tests.
- `tests/editor_frontend/chart_geometry.test.js` — deterministic categorical-axis geometry tests.
- `tests/editor/conftest.py` — fitted editor model and real-widget browser factory.
- `tests/editor/test_editor_workspace_browser.py` — semantic workspace and responsive browser tests.
- `tests/editor/test_editor_axis_browser.py` — actual SVG font measurement, clipping, and full-label
  browser tests.

### Modify

- `src/superglm/editor/app/index.html` — semantic workspace markup while preserving existing curve
  operation SVG paths.
- `src/superglm/editor/app/main.js` — compose the new views, render their state, and bind callbacks.
- `src/superglm/editor/app/chart.js` — consume pure axis geometry, measure actual SVG text, draw
  accessible categorical ticks, and reserve a dynamic gutter.
- `src/superglm/editor/app/interactions.js` — read mode through a callback instead of a select
  element and leave pointer gesture state otherwise unchanged.
- `src/superglm/editor/app/styles.css` — remove only rules superseded by the four focused style
  modules; retain untouched report/dialog/profile rules during this pass.
- `tests/test_editor.py` — reduce brittle source-string assertions to asset/server contracts and
  update the nested asset list.
- `package.json` and `jsconfig.json` — extend the foundation include globs for `views/`,
  `chart/geometry.js`, and their `tests/editor_frontend/` tests; do not change scripts or strictness.
- `.github/workflows/ci.yml` and `.github/workflows/dev-ci.yml` — extend the existing single browser
  job to collect `tests/editor/`; do not create another job.

## Dependency order

```text
Task 1 tooling ──> Task 2 browser harness
       │                   │
       ├──> Task 3 tokens/style boundaries ──> Task 5 shell ──> Task 6 tool rail
       │                                           │                 │
       ├──> Task 4 popover/help catalog ───────────┼────────> Task 7 inspector/help
       │                                           │                 │
       └──> Task 9 pure axis geometry ─────────────┴────────> Task 10 SVG axis integration

Tasks 5-10 ──> Task 8 responsive matrix ──> Task 11 accessibility hardening ──> Task 12 final checks
```

Task 1 is already satisfied by the foundation plan and is checked below for traceability; do not
re-run it. Task 9 can run in parallel with Tasks 3-7. Task 10 waits for both the geometry module
and the shared popover. Task 8 is listed after the component work even though each component adds
its own failing browser test first.

### Task 1: Add development-only frontend test and type-check tooling

**Prerequisite status:** Satisfied by the foundation plan. The checked steps document why this phase
has native-module tooling; they are not implementation work and must not alter the foundation's
Node 24, TypeScript 7, strict `checkJs`, npm scripts, browser extra, or lockfile.

**Files:**
- Create: `package.json`
- Create: `package-lock.json`
- Create: `jsconfig.json`
- Modify: `pyproject.toml:23-51,71-86`
- Modify: `uv.lock`

- [x] **Step 1: Create the frontend package manifest**

Create `package.json` with no runtime dependencies or build command:

```json
{
  "name": "superglm-editor-frontend",
  "private": true,
  "type": "module",
  "scripts": {
    "test:frontend": "node --test tests/editor_frontend/*.test.js",
    "check:types": "tsc -p jsconfig.json",
    "check:frontend": "npm run check:types && npm run test:frontend"
  },
  "devDependencies": {
    "@types/node": "^24.0.0",
    "typescript": "^5.9.0"
  }
}
```

- [x] **Step 2: Create the gradual-JavaScript type-check configuration**

Create `jsconfig.json`:

```json
{
  "compilerOptions": {
    "allowJs": true,
    "checkJs": true,
    "noEmit": true,
    "strict": false,
    "noImplicitAny": false,
    "skipLibCheck": true,
    "target": "ES2022",
    "module": "ES2022",
    "moduleResolution": "Bundler",
    "types": ["node"],
    "lib": ["DOM", "DOM.Iterable", "ES2022"]
  },
  "include": [
    "src/superglm/editor/app/views/**/*.js",
    "src/superglm/editor/app/chart/geometry.js",
    "tests/editor_frontend/**/*.test.js"
  ]
}
```

- [x] **Step 3: Generate the development lockfile**

Run:

```bash
rtk npm install --package-lock-only
```

Expected: `package-lock.json` is created and reports zero runtime packages.

- [x] **Step 4: Add the optional Python browser dependency and marker**

Add this dependency group after the existing `docs` group in `pyproject.toml`:

```toml
frontend = [
    "playwright>=1.52",
]
```

Add this entry to `[tool.pytest.ini_options].markers`:

```toml
    "browser: runs the real editor in Playwright Chromium",
```

Then lock it:

```bash
rtk uv lock
```

Expected: `uv.lock` records Playwright without adding it to SuperGLM's runtime dependencies.

- [x] **Step 5: Verify the empty harness**

Run:

```bash
rtk npm exec tsc -- --version
```

Expected: the locked TypeScript version prints successfully. The combined check starts in Task 4,
after the first focused source modules and tests exist.

- [x] **Step 6: Commit the tooling boundary**

```bash
rtk git add package.json package-lock.json jsconfig.json pyproject.toml uv.lock
rtk git commit -m "Add editor frontend test tooling"
```

### Task 2: Add a real-widget Playwright harness

**Files:**
- Create: `tests/editor/conftest.py`
- Create: `tests/editor/test_editor_workspace_browser.py`
- Modify: `.github/workflows/ci.yml:45-76`
- Modify: `.github/workflows/dev-ci.yml:42-65`

- [ ] **Step 1: Create deterministic fitted browser fixtures**

Create `tests/editor/conftest.py`:

```python
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import pandas as pd
import pytest

from superglm import Categorical, Spline, SuperGLM
from superglm.editor import EditorSession


@pytest.fixture(scope="session")
def editor_browser_model() -> SuperGLM:
    rng = np.random.default_rng(20260711)
    territory_levels = [f"T{i:02d}" for i in range(1, 11)]
    long_levels = [
        "MyReallyLongCategoryNameThatWouldNeverFit",
        "CommercialVehicleWithSpecialistUsage",
        "PrivateMotorStandard",
        "AgriculturalMachinery",
        "MotorcycleAndScooter",
        "TaxiAndPrivateHire",
        "FleetLightCommercial",
        "FleetHeavyCommercial",
        "ClassicAndCollectable",
        "TemporaryAdditionalVehicle",
    ]
    n = 500
    curve = rng.uniform(0.0, 10.0, n)
    territory = rng.choice(territory_levels, n)
    long_category = rng.choice(long_levels, n)
    y = (
        0.5
        + 0.12 * np.sin(curve)
        + 0.03 * np.array([territory_levels.index(value) for value in territory])
        + rng.normal(0.0, 0.04, n)
    )
    frame = pd.DataFrame(
        {
            "curve": curve,
            "territory": territory,
            "long_category": long_category,
        }
    )
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        spline_penalty=0.1,
        features={
            "curve": Spline(n_knots=7),
            "territory": Categorical(base="first"),
            "long_category": Categorical(base="first"),
        },
    )
    model.fit(frame, y)
    return model


@pytest.fixture(scope="session")
def chromium_browser():
    playwright_api = pytest.importorskip("playwright.sync_api")
    with playwright_api.sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        yield browser
        browser.close()


@pytest.fixture
def open_editor_page(chromium_browser, editor_browser_model):
    opened: list[tuple[object, object]] = []

    @contextmanager
    def open_page(
        *,
        viewport: dict[str, int] | None = None,
        selected_term: str = "curve",
    ) -> Iterator[tuple[object, EditorSession]]:
        session = EditorSession.from_model(
            editor_browser_model,
            terms=["curve", "territory", "long_category"],
        )
        widget = session.widget()
        page = chromium_browser.new_page(viewport=viewport or {"width": 1180, "height": 720})
        opened.append((page, widget))
        try:
            page.goto(f"{widget.app_url}&test=1", wait_until="networkidle")
            page.locator("#chart .point").first.wait_for()
            if selected_term != "curve":
                page.select_option("#term", selected_term)
                page.locator("#chart .point").first.wait_for()
            yield page, session
        finally:
            page.close()
            widget.close()
            opened.remove((page, widget))

    yield open_page

    for page, widget in reversed(opened):
        page.close()
        widget.close()
```

- [ ] **Step 2: Write and run the real-server smoke test**

Create `tests/editor/test_editor_workspace_browser.py` with the baseline smoke test:

```python
from __future__ import annotations

import pytest

pytest.importorskip("playwright.sync_api")
pytestmark = pytest.mark.browser


def test_real_editor_boots_and_draws_svg(open_editor_page):
    with open_editor_page() as (page, _session):
        assert page.title() == "SuperGLM Editor"
        assert page.locator("#chart").get_attribute("role") == "img"
        assert page.locator("#chart .point").count() > 0
```

Run:

```bash
rtk uv sync --extra dev
rtk uv run playwright install chromium
rtk uv run pytest tests/editor/test_editor_workspace_browser.py --run-browser -q
```

Expected: one browser test passes against `widget.app_url`, including its editor token.

- [ ] **Step 3: Extend the existing Linux/Chromium job**

In the `frontend` job created by the foundation plan, replace only its browser-test command with:

```yaml
      - name: Browser tests
        run: uv run pytest tests/test_editor_browser.py tests/editor/ -m browser --run-browser -q
```

Make the same command change in the existing development-branch browser step. Keep exactly one
Chromium installation/job per workflow; do not add an `editor-browser` job or a second Playwright
dependency group.

- [ ] **Step 4: Commit the browser harness**

```bash
rtk git add tests/editor .github/workflows/ci.yml .github/workflows/dev-ci.yml
rtk git commit -m "Test the editor in a real browser"
```

### Task 3: Establish complete tokens and focused style boundaries

**Files:**
- Create: `src/superglm/editor/app/styles/tokens.css`
- Create: `src/superglm/editor/app/styles/shell.css`
- Create: `src/superglm/editor/app/styles/chart.css`
- Create: `src/superglm/editor/app/styles/panels.css`
- Modify: `src/superglm/editor/app/index.html:6-8`
- Modify: `src/superglm/editor/app/styles.css:1-25,87-226,227-371,927-967,1068-1161`
- Modify: `tests/test_editor.py:3949-4450`

- [ ] **Step 1: Add a failing nested-asset contract test**

In `test_widget_serves_editor_app_assets` in `tests/test_editor.py`, extend the requested asset list
with these exact paths:

```python
        for asset in [
            "styles/tokens.css",
            "styles/shell.css",
            "styles/chart.css",
            "styles/panels.css",
        ]:
            request = urllib.request.Request(f"{widget.url}/assets/{asset}", method="GET")
            with urllib.request.urlopen(request, timeout=5) as response:
                assert response.headers["Content-Type"].startswith("text/css")
                assert response.read()
```

Run:

```bash
rtk uv run pytest tests/test_editor.py::test_widget_serves_editor_app_assets -q
```

Expected: FAIL with a 404 for `styles/tokens.css`.

- [ ] **Step 2: Create the complete token file**

Create `src/superglm/editor/app/styles/tokens.css`:

```css
:root {
  color-scheme: light;
  --text: #24292f;
  --muted: #57606a;
  --surface: #ffffff;
  --surface-subtle: #f6f8fa;
  --surface-hover: #eef2f6;
  --border: #d0d7de;
  --border-strong: #8c959f;
  --grid: rgba(140, 149, 159, 0.22);
  --blue: #0969da;
  --blue-soft: #dbeafe;
  --red: #d1242f;
  --orange: #bf6a02;
  --yellow: #f4d35e;
  --yellow-border: #d8a10f;
  --danger: #b42318;
  --focus: #0969da;
  --shadow: rgba(31, 35, 40, 0.16);
  --radius-sm: 4px;
  --radius-md: 6px;
  --space-1: 4px;
  --space-2: 8px;
  --space-3: 12px;
  --space-4: 16px;
  --tool-rail-width: 44px;
  --inspector-width: 360px;
  --tooltip-delay-ms: 350ms;
}

*, *::before, *::after {
  box-sizing: border-box;
}

html {
  min-width: 0;
  min-height: 100%;
  background: var(--surface);
}

body {
  min-width: 0;
  min-height: 100dvh;
  margin: 0;
  background: var(--surface);
  color: var(--text);
  font: 13px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

button,
select,
input {
  font: inherit;
}

button:focus-visible,
select:focus-visible,
input:focus-visible,
[tabindex="0"]:focus-visible {
  outline: 2px solid var(--focus);
  outline-offset: 2px;
}

[hidden] {
  display: none !important;
}

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    scroll-behavior: auto !important;
    animation-duration: 0.001ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.001ms !important;
  }
}
```

- [ ] **Step 3: Create the focused style files and load order**

Create the other three files with responsibility comments so later tasks append only their owned
selectors:

```css
/* shell.css: app bar, context bar, workspace grid, tool rail, and responsive shell. */
```

```css
/* chart.css: chart surface, SVG axis, selection palette, and shared popover. */
```

```css
/* panels.css: inspector, help, advanced controls, metrics, and reports. */
```

Replace the stylesheet link in `index.html` with this ordered set:

```html
<link rel="stylesheet" href="/assets/styles/tokens.css">
<link rel="stylesheet" href="/assets/styles.css">
<link rel="stylesheet" href="/assets/styles/shell.css">
<link rel="stylesheet" href="/assets/styles/chart.css">
<link rel="stylesheet" href="/assets/styles/panels.css">
```

Remove the old `:root`, universal `box-sizing`, and `body` typography/color declarations from
`styles.css`. Leave its current layout rules until their replacements land in Tasks 5-8.

- [ ] **Step 4: Update the static shell assertion and rerun it**

Replace the single-link assertion in `test_widget_serves_editor_app_assets` with:

```python
        assert '<link rel="stylesheet" href="/assets/styles/tokens.css">' in shell
        assert '<link rel="stylesheet" href="/assets/styles.css">' in shell
        assert '<link rel="stylesheet" href="/assets/styles/shell.css">' in shell
        assert '<link rel="stylesheet" href="/assets/styles/chart.css">' in shell
        assert '<link rel="stylesheet" href="/assets/styles/panels.css">' in shell
```

Run:

```bash
rtk uv run pytest tests/test_editor.py::test_widget_serves_editor_app_assets -q
```

Expected: PASS and all nested CSS assets return `text/css`.

- [ ] **Step 5: Commit the style boundary**

```bash
rtk git add src/superglm/editor/app tests/test_editor.py
rtk git commit -m "Define editor workspace style boundaries"
```

### Task 4: Add shared help copy and delayed popovers

**Files:**
- Create: `src/superglm/editor/app/views/help_content.js`
- Create: `src/superglm/editor/app/views/popover.js`
- Create: `tests/editor_frontend/popover.test.js`
- Modify: `src/superglm/editor/app/index.html:65-201,327-328`
- Modify: `src/superglm/editor/app/main.js:1-115,700-850`
- Modify: `src/superglm/editor/app/styles/chart.css`

- [ ] **Step 1: Write failing deterministic delay tests**

Create `tests/editor_frontend/popover.test.js`:

```javascript
import assert from "node:assert/strict";
import test from "node:test";

import { PopoverDelay, TOOLTIP_SHOW_DELAY_MS } from "../../src/superglm/editor/app/views/popover.js";

function fakeScheduler() {
  let callback = null;
  return {
    setTimer(next, delay) {
      callback = next;
      assert.equal(delay, TOOLTIP_SHOW_DELAY_MS);
      return 1;
    },
    clearTimer() {
      callback = null;
    },
    flush() {
      const next = callback;
      callback = null;
      if (next) next();
    },
  };
}

test("pointer disclosure waits 350ms and pointer dismissal is immediate", () => {
  const scheduler = fakeScheduler();
  const events = [];
  const delay = new PopoverDelay({
    setTimer: scheduler.setTimer,
    clearTimer: scheduler.clearTimer,
    onShow: () => events.push("show"),
    onHide: () => events.push("hide"),
  });
  delay.pointerEnter();
  assert.deepEqual(events, []);
  scheduler.flush();
  assert.deepEqual(events, ["show"]);
  delay.pointerLeave();
  assert.deepEqual(events, ["show", "hide"]);
});

test("keyboard focus shows immediately and Escape hides", () => {
  const scheduler = fakeScheduler();
  const events = [];
  const delay = new PopoverDelay({
    setTimer: scheduler.setTimer,
    clearTimer: scheduler.clearTimer,
    onShow: () => events.push("show"),
    onHide: () => events.push("hide"),
  });
  delay.focus();
  delay.escape();
  assert.deepEqual(events, ["show", "hide"]);
});
```

Run:

```bash
rtk node --test --test-name-pattern "pointer disclosure|keyboard focus" tests/editor_frontend/popover.test.js
```

Expected: FAIL because `views/popover.js` does not exist.

- [ ] **Step 2: Create the shared help catalog**

Create `src/superglm/editor/app/views/help_content.js` with these exported immutable records:

```javascript
// @ts-check

export const TOOL_HELP = Object.freeze({
  select: { title: "Select", body: "Click points or drag a box to select curve values.", shortcut: "V" },
  move: { title: "Move", body: "Drag a selected point or selection to change relativity.", shortcut: "M" },
  zoom: { title: "Zoom", body: "Drag a box to zoom. The mouse wheel zooms in every mode.", shortcut: "Z" },
  handles: { title: "Handles", body: "Edit spline control handles and inspect basis contributions.", shortcut: "H" },
  help: { title: "Help", body: "Open modes, gestures, shortcuts, curve operations, refits, and saving.", shortcut: "?" },
});

export const OPERATION_HELP = Object.freeze({
  shift_up: { title: "Increase selection", body: "Increase selected relativities by 5%." },
  shift_down: { title: "Decrease selection", body: "Decrease selected relativities by 5%." },
  smooth: { title: "Smooth selection", body: "Reduce local variation across the selected relativities." },
  linearise: {
    title: "Straighten selection",
    body: "Interpolate the selected relativities between their first and last points.",
  },
  increasing: { title: "Make increasing", body: "Constrain selected relativities to a non-decreasing sequence." },
  decreasing: { title: "Make decreasing", body: "Constrain selected relativities to a non-increasing sequence." },
  level_left: { title: "Level from left", body: "Set selected relativities to the leftmost selected value." },
  average: { title: "Average selection", body: "Set selected relativities to their exposure-weighted mean (or their unweighted mean when exposure is unavailable)." },
  level_right: { title: "Level from right", body: "Set selected relativities to the rightmost selected value." },
  snap_highest: { title: "Snap to highest", body: "Set selected relativities to the highest selected value." },
  snap_lowest: { title: "Snap to lowest", body: "Set selected relativities to the lowest selected value." },
  collapse_levels: { title: "Collapse and refit", body: "Combine the selected categorical levels and refit the model." },
  ungroup_levels: { title: "Ungroup and refit", body: "Separate the selected grouped levels and refit the model." },
  uncollapse_levels: { title: "Restore collapse", body: "Restore the model state from before the last collapse." },
});

export const HELP_SECTIONS = Object.freeze([
  { title: "Modes", keys: ["select", "move", "zoom", "handles"] },
  { title: "Selection operations", keys: Object.keys(OPERATION_HELP) },
  { title: "Navigation", items: ["Mouse wheel: zoom", "Shift-drag or middle-drag: pan", "Home: reset zoom"] },
  { title: "History", items: ["Ctrl/Cmd+Z: undo", "Ctrl/Cmd+Shift+Z or Ctrl+Y: redo"] },
  { title: "Saving", items: ["Save writes or downloads the current Python-confirmed edited model."] },
]);

export function helpForElement(element) {
  const tool = element.dataset.tool;
  if (tool && TOOL_HELP[tool]) return TOOL_HELP[tool];
  const operation = element.dataset.helpOperation || element.dataset.op;
  if (operation && OPERATION_HELP[operation]) return OPERATION_HELP[operation];
  const title = element.dataset.popoverTitle;
  if (!title) return null;
  return { title, body: element.dataset.popoverBody || "" };
}
```

- [ ] **Step 3: Implement the timing primitive and delegated DOM controller**

Create `src/superglm/editor/app/views/popover.js`:

```javascript
// @ts-check

import { helpForElement } from "./help_content.js";

export const TOOLTIP_SHOW_DELAY_MS = 350;

export class PopoverDelay {
  constructor({ setTimer, clearTimer, onShow, onHide }) {
    this.setTimer = setTimer;
    this.clearTimer = clearTimer;
    this.onShow = onShow;
    this.onHide = onHide;
    this.timer = null;
    this.visible = false;
  }

  pointerEnter() {
    this.cancel();
    this.timer = this.setTimer(() => this.show(), TOOLTIP_SHOW_DELAY_MS);
  }

  pointerLeave() {
    this.cancel();
    this.hide();
  }

  focus() {
    this.cancel();
    this.show();
  }

  escape() {
    this.cancel();
    this.hide();
  }

  show() {
    this.timer = null;
    if (this.visible) return;
    this.visible = true;
    this.onShow();
  }

  hide() {
    if (!this.visible) return;
    this.visible = false;
    this.onHide();
  }

  cancel() {
    if (this.timer === null) return;
    this.clearTimer(this.timer);
    this.timer = null;
  }
}

export function bindPopovers({ root, popover }) {
  let target = null;
  const titleNode = popover.querySelector("[data-popover-heading]");
  const bodyNode = popover.querySelector("[data-popover-description]");
  const delay = new PopoverDelay({
    setTimer: window.setTimeout.bind(window),
    clearTimer: window.clearTimeout.bind(window),
    onShow: () => show(),
    onHide: () => hide(),
  });

  function candidate(node) {
    return node instanceof Element
      ? node.closest("[data-tool], [data-help-operation], [data-popover-title]")
      : null;
  }

  function setTarget(next, immediate) {
    target = next;
    if (!target || !helpForElement(target)) {
      delay.pointerLeave();
    } else if (immediate) {
      delay.focus();
    } else {
      delay.pointerEnter();
    }
  }

  function show() {
    if (!target) return;
    const help = helpForElement(target);
    if (!help) return;
    titleNode.textContent = help.title;
    bodyNode.textContent = help.body;
    popover.hidden = false;
    target.setAttribute("aria-describedby", popover.id);
    const anchor = target.getBoundingClientRect();
    const box = popover.getBoundingClientRect();
    const left = Math.max(8, Math.min(window.innerWidth - box.width - 8, anchor.left + anchor.width / 2 - box.width / 2));
    const below = anchor.bottom + 8;
    const top = below + box.height <= window.innerHeight - 8 ? below : anchor.top - box.height - 8;
    popover.style.left = `${left}px`;
    popover.style.top = `${Math.max(8, top)}px`;
  }

  function hide() {
    if (target) target.removeAttribute("aria-describedby");
    popover.hidden = true;
  }

  root.addEventListener("pointerover", (event) => {
    const next = candidate(event.target);
    if (next && next !== target) setTarget(next, false);
  });
  root.addEventListener("pointerout", (event) => {
    if (!target || target.contains(event.relatedTarget)) return;
    delay.pointerLeave();
    target = null;
  });
  root.addEventListener("focusin", (event) => {
    const next = candidate(event.target);
    if (next) setTarget(next, true);
  });
  root.addEventListener("focusout", (event) => {
    if (!target || target.contains(event.relatedTarget)) return;
    delay.pointerLeave();
    target = null;
  });
  root.addEventListener("keydown", (event) => {
    if (event.key !== "Escape") return;
    delay.escape();
  });

  return { close: () => delay.escape(), isOpen: () => !popover.hidden };
}
```

- [ ] **Step 4: Add the popover markup and preserve every selection SVG path**

Insert this immediately before the module script in `index.html`:

```html
<div id="uiPopover" class="ui-popover" role="tooltip" hidden>
  <strong data-popover-heading></strong>
  <span data-popover-description></span>
</div>
```

On each existing `.selection-item`, remove the native `title` attribute, preserve its SVG child
without changing any `path` data, and add `data-help-operation` matching its operation. For the
current `linearise` button, the exact result is:

```html
<button class="selection-item" data-op="linearise" data-help-operation="linearise"
  type="button" aria-label="Straighten selection">
  <svg class="selection-icon" viewBox="0 0 24 24" aria-hidden="true">
    <path d="M4 18 20 6"></path><path d="M5 17h3v3"></path><path d="M16 4h3v3"></path>
  </svg>
</button>
```

Set each operation button's `aria-label` to the matching `OPERATION_HELP` title: Increase selection,
Decrease selection, Smooth selection, Straighten selection, Make increasing, Make decreasing,
Level from left, Average selection, Level from right, Snap to highest, Snap to lowest, Collapse and
refit, Ungroup and refit, and Restore collapse.

For structural buttons without `data-op`, use `collapse_levels`, `ungroup_levels`, and
`uncollapse_levels` as `data-help-operation` values.

- [ ] **Step 5: Bind and style the delegated popover**

Import and bind it once in `main.js`:

```javascript
import { bindPopovers } from "./views/popover.js";

const uiPopover = document.getElementById("uiPopover");
const popovers = bindPopovers({ root: document, popover: uiPopover });
```

Append to `styles/chart.css`:

```css
.ui-popover {
  position: fixed;
  z-index: 80;
  width: max-content;
  max-width: min(280px, calc(100vw - 16px));
  padding: 8px 10px;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--text);
  color: var(--surface);
  box-shadow: 0 8px 24px var(--shadow);
  pointer-events: none;
}

.ui-popover strong,
.ui-popover span {
  display: block;
}

.ui-popover span {
  margin-top: 2px;
  color: #eaeef2;
  font-size: 12px;
}
```

- [ ] **Step 6: Run frontend checks and commit**

Run:

```bash
rtk npm run check:frontend
rtk uv run pytest tests/test_editor.py::test_widget_app_shell_contains_drag_editor -q
```

Expected: popover tests and type checks pass; the existing selection operations still appear in the
served shell.

Commit:

```bash
rtk git add src/superglm/editor/app tests/editor_frontend
rtk git commit -m "Explain editor icons with delayed popovers"
```

### Task 5: Build the application bar and analyst context bar

**Files:**
- Create: `src/superglm/editor/app/views/app_bar.js`
- Create: `src/superglm/editor/app/views/context_bar.js`
- Modify: `src/superglm/editor/app/index.html:20-77`
- Modify: `src/superglm/editor/app/main.js:12-115,454-520,700-850`
- Modify: `src/superglm/editor/app/interactions.js:258-279,693-701`
- Modify: `src/superglm/editor/app/styles/shell.css`
- Modify: `tests/editor/test_editor_workspace_browser.py`

- [ ] **Step 1: Write failing semantic app-bar tests**

Append to `tests/editor/test_editor_workspace_browser.py`:

```python
def test_application_bar_exposes_views_undo_redo_and_save(open_editor_page):
    with open_editor_page() as (page, _session):
        tabs = page.get_by_role("tablist", name="Editor views")
        assert tabs.get_by_role("tab").all_inner_texts() == ["Editor", "Validation", "Final Fit"]
        assert page.get_by_role("button", name="Undo edit").is_disabled()
        assert page.get_by_role("button", name="Redo edit").is_disabled()
        assert page.get_by_role("button", name="Save edited model").is_visible()


def test_context_bar_reports_term_kind_and_edf(open_editor_page):
    with open_editor_page(selected_term="curve") as (page, _session):
        context = page.get_by_role("region", name="Term context")
        assert context.get_by_label("Term").input_value() == "curve"
        assert "spline" in context.locator("#termKind").inner_text().lower()
        assert "EDF" in context.locator("#termEdf").inner_text()
```

Run:

```bash
rtk uv run pytest tests/editor/test_editor_workspace_browser.py --run-browser -q -k "application_bar or context_bar"
```

Expected: FAIL because the new labels and controls do not exist.

- [ ] **Step 2: Create the application-bar controller**

Create `src/superglm/editor/app/views/app_bar.js`:

```javascript
// @ts-check

export function bindAppBar({ root, undoButton, redoButton, onView, onUndo, onRedo }) {
  const tabs = Array.from(root.querySelectorAll('[role="tab"]'));
  root.addEventListener("click", (event) => {
    const tab = event.target.closest('[role="tab"]');
    if (tab) onView(tab.dataset.view || "editor");
  });
  root.addEventListener("keydown", (event) => {
    const index = tabs.indexOf(event.target);
    if (index >= 0 && ["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) {
      event.preventDefault();
      const next = event.key === "Home"
        ? 0
        : event.key === "End"
          ? tabs.length - 1
          : (index + (event.key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length;
      tabs[next].focus();
      onView(tabs[next].dataset.view || "editor");
    }
  });
  undoButton.addEventListener("click", onUndo);
  redoButton.addEventListener("click", onRedo);
  document.addEventListener("keydown", (event) => {
    if (isEditableTarget(event.target) || event.altKey) return;
    const primary = event.ctrlKey || event.metaKey;
    if (!primary) return;
    const key = event.key.toLowerCase();
    if (key === "z" && !event.shiftKey) {
      event.preventDefault();
      if (!undoButton.disabled) onUndo();
    } else if (key === "y" || (key === "z" && event.shiftKey)) {
      event.preventDefault();
      if (!redoButton.disabled) onRedo();
    }
  });
}

export function renderAppBar({ root, activeView, undoButton, redoButton, canUndo, canRedo }) {
  for (const tab of root.querySelectorAll('[role="tab"]')) {
    const active = tab.dataset.view === activeView;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-selected", String(active));
    tab.tabIndex = active ? 0 : -1;
  }
  undoButton.disabled = !canUndo;
  redoButton.disabled = !canRedo;
}

function isEditableTarget(target) {
  if (!(target instanceof HTMLElement)) return false;
  const tag = target.tagName.toLowerCase();
  return target.isContentEditable || tag === "input" || tag === "select" || tag === "textarea";
}
```

- [ ] **Step 3: Create the context renderer**

Create `src/superglm/editor/app/views/context_bar.js`:

```javascript
// @ts-check

import { fmt, fmtPercent } from "../format.js";

export function renderContextBar({ kindNode, edfNode, statusNode }, { name, term, selectionSize }) {
  const kind = term.term_type || term.kind || "term";
  kindNode.textContent = kind;
  edfNode.textContent = term.effective_df === null || term.effective_df === undefined
    ? "EDF unavailable"
    : `EDF ${fmt(term.effective_df)}`;
  const impact = term.impact || {};
  statusNode.textContent = `${selectionSize} of ${term.n_points} selected · average edit relativity ${fmt(impact.weighted_mean_relativity || 1)}x · selected exposure ${fmtPercent(impact.selected_weight_share || 0)}`;
  statusNode.dataset.term = name;
}
```

- [ ] **Step 4: Replace the top markup**

Replace the existing `.app-tabs` and `.toolbar` opening region with:

```html
<header id="appBar" class="app-bar">
  <div class="app-tabs" role="tablist" aria-label="Editor views">
    <button id="editorTab" class="app-tab active" type="button" data-view="editor"
      role="tab" aria-selected="true" aria-controls="editorView">Editor</button>
    <button id="validationTab" class="app-tab" type="button" data-view="validation"
      role="tab" aria-selected="false" aria-controls="reportPanel" tabindex="-1">Validation</button>
    <button id="finalTab" class="app-tab" type="button" data-view="final"
      role="tab" aria-selected="false" aria-controls="reportPanel" tabindex="-1">Final Fit</button>
  </div>
  <div class="app-actions" aria-label="Edit actions">
    <button id="undoAction" type="button" aria-label="Undo edit" aria-keyshortcuts="Control+Z Meta+Z"
      data-popover-title="Undo" data-popover-body="Undo the most recent edit. Ctrl/Cmd+Z." disabled>Undo</button>
    <button id="redoAction" type="button" aria-label="Redo edit"
      aria-keyshortcuts="Control+Y Control+Shift+Z Meta+Shift+Z"
      data-popover-title="Redo" data-popover-body="Restore the most recently undone edit." disabled>Redo</button>
    <button id="saveModel" class="icon-button" type="button" aria-label="Save edited model"
      data-popover-title="Save" data-popover-body="Save or download the Python-confirmed edited model.">
      <svg class="toolbar-icon" viewBox="0 0 24 24" aria-hidden="true">
        <path d="M5 4h12l2 2v14H5z"></path><path d="M8 4v6h8V4"></path>
        <path d="M8 15h8v5H8z"></path><path d="M14 4v4"></path>
      </svg>
    </button>
  </div>
</header>

<div class="context-bar" role="region" aria-label="Term context">
  <label class="context-term">Term <select id="term"></select></label>
  <span id="termKind" class="context-chip"></span>
  <span id="termEdf" class="context-chip"></span>
  <label id="groupDisplayWrap" class="toolbar-field group-display-toggle">
    <span>Groups</span>
    <select id="groupDisplayMode">
      <option value="expanded">Expanded</option>
      <option value="collapsed">Collapsed</option>
    </select>
  </label>
  <button id="ciToggle" type="button" aria-pressed="false">Reference CI</button>
  <button id="inspectorToggle" type="button" aria-expanded="true" aria-controls="inspector">Inspector</button>
  <span id="status" role="status" aria-live="polite"></span>
</div>
```

Move the current handle-count, contribution, Build, Home, Reset order, and Reset controls into this
temporary chart action bar below; Tasks 6-7 place them permanently. Retain the mode select until
Task 6 so this intermediate commit remains runnable:

```html
<div id="chartActionBar" class="chart-action-bar">
  <label>Mode <select id="mode"><option value="select">Select</option><option value="zoom">Zoom</option><option value="move">Move</option><option value="handles">Handles</option></select></label>
  <label id="handleCountWrap" class="handle-count" hidden>
    <span>Handles</span>
    <input id="handleCount" type="range" min="3" max="12" value="12">
    <output id="handleCountValue">12</output>
  </label>
  <button id="basisToggle" type="button" hidden>Contrib</button>
  <button id="contribPlay" type="button" hidden>Build</button>
  <label id="buildDurationWrap" class="build-duration" hidden>
    <span>Build</span>
    <input id="buildDuration" type="range" min="4000" max="30000" step="500" value="10000">
    <output id="buildDurationValue">10s</output>
  </label>
  <button type="button" data-op="select_all">Select all</button>
  <button id="resetZoom" type="button">Home</button>
  <button id="resetOrder" type="button" data-op="reset_order" hidden>Reset order</button>
  <button type="button" data-op="reset">Reset</button>
</div>
```

- [ ] **Step 5: Wire `main.js` and remove duplicate keyboard handling**

Import `bindAppBar`, `renderAppBar`, and `renderContextBar`. Bind Undo/Redo through the same mutation
function used by keyboard shortcuts:

```javascript
const undo = () => actions.executeStateMutation({
  name: "undo",
  path: "/op",
  payload: { operation: "undo" }
});
const redo = () => actions.executeStateMutation({
  name: "redo",
  path: "/op",
  payload: { operation: "redo" }
});

bindAppBar({
  root: appBar,
  undoButton: undoAction,
  redoButton: redoAction,
  onView: showView,
  onUndo: undo,
  onRedo: redo,
});
```

At the end of `render()`, add:

```javascript
const editorState = store.getState();
const snapshot = selectSnapshot(editorState);
renderAppBar({
  root: appBar,
  activeView: editorState.view.activeView,
  undoButton: undoAction,
  redoButton: redoAction,
  canUndo: Boolean(snapshot?.history.active.length),
  canRedo: Boolean(snapshot?.history.redo.length),
});
renderContextBar(
  { kindNode: termKind, edfNode: termEdf, statusNode },
  { name: selected, term, selectionSize: selection.size },
);
ciToggle.setAttribute("aria-pressed", String(editorState.view.showCi));
```

Delete the document `keydown` block and local `isEditableTarget` function from `interactions.js`;
`app_bar.js` now owns global undo/redo shortcuts.

- [ ] **Step 6: Style the two rows and verify**

Append to `styles/shell.css`:

```css
.app-bar,
.context-bar {
  display: flex;
  min-width: 0;
  align-items: center;
  gap: var(--space-2);
}

.app-bar {
  justify-content: space-between;
  border-bottom: 1px solid var(--border);
}

.app-tabs,
.app-actions {
  display: flex;
  align-items: center;
  gap: var(--space-1);
}

.context-bar {
  min-height: 40px;
  flex-wrap: wrap;
  padding: var(--space-1) 0 var(--space-2);
}

.chart-action-bar {
  display: flex;
  min-width: 0;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px;
  margin-bottom: 8px;
}

.context-term select {
  min-width: 180px;
}

.context-chip {
  padding: 3px 7px;
  border: 1px solid var(--border);
  border-radius: 999px;
  color: var(--muted);
  background: var(--surface-subtle);
}

#status {
  min-width: 220px;
  margin-left: auto;
  color: var(--muted);
  text-align: right;
}
```

Run:

```bash
rtk uv run pytest tests/editor/test_editor_workspace_browser.py --run-browser -q -k "application_bar or context_bar"
rtk npm run check:frontend
```

Expected: both browser tests and frontend checks pass.

- [ ] **Step 7: Commit the analyst header**

```bash
rtk git add src/superglm/editor/app tests/editor/test_editor_workspace_browser.py
rtk git commit -m "Add editor application and context bars"
```

### Task 6: Replace the mode select with an accessible tool rail

**Files:**
- Create: `src/superglm/editor/app/views/tool_rail.js`
- Modify: `src/superglm/editor/app/index.html:78-218`
- Modify: `src/superglm/editor/app/main.js:20-120,466-620,630-760`
- Modify: `src/superglm/editor/app/chart.js:23-31`
- Modify: `src/superglm/editor/app/interactions.js:1-100`
- Modify: `src/superglm/editor/app/styles/shell.css`
- Modify: `tests/editor/test_editor_workspace_browser.py`

- [ ] **Step 1: Write failing tool-rail semantics and operation-preservation tests**

Append:

```python
def test_tool_rail_selects_one_mode_and_exposes_help(open_editor_page):
    with open_editor_page() as (page, _session):
        rail = page.get_by_role("radiogroup", name="Chart tools")
        select = rail.get_by_role("radio", name="Select")
        move = rail.get_by_role("radio", name="Move")
        assert select.get_attribute("aria-checked") == "true"
        move.click()
        assert move.get_attribute("aria-checked") == "true"
        assert select.get_attribute("aria-checked") == "false"
        assert page.get_by_role("button", name="Help").is_visible()


def test_existing_svg_selection_operation_still_posts_linearise(open_editor_page):
    with open_editor_page() as (page, session):
        page.locator("#chart .point").nth(2).click()
        page.locator("#chart .point").nth(3).click(modifiers=["Control"])
        page.get_by_role("button", name="Straighten selection").click()
        page.wait_for_timeout(50)
        assert session.history[-1].operation == "linear_interpolate"
```

Run:

```bash
rtk uv run pytest tests/editor/test_editor_workspace_browser.py --run-browser -q -k "tool_rail or svg_selection"
```

Expected: FAIL because the rail does not exist; after the rail lands, the authoritative session
records the unchanged `linear_interpolate` operation.

- [ ] **Step 2: Create the exclusive mode controller**

Create `src/superglm/editor/app/views/tool_rail.js`:

```javascript
// @ts-check

const MODES = new Set(["select", "move", "zoom", "handles"]);

export function bindToolRail({ root, onMode, onHelp }) {
  root.addEventListener("click", (event) => {
    const button = event.target.closest("[data-tool]");
    if (!button) return;
    if (button.dataset.tool === "help") onHelp();
    else if (MODES.has(button.dataset.tool)) onMode(button.dataset.tool);
  });
  root.addEventListener("keydown", (event) => {
    const radios = Array.from(root.querySelectorAll('[role="radio"]:not(:disabled)'));
    const index = radios.indexOf(event.target);
    if (index < 0 || !["ArrowUp", "ArrowDown", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    const next = event.key === "Home"
      ? 0
      : event.key === "End"
        ? radios.length - 1
        : (index + (event.key === "ArrowDown" ? 1 : -1) + radios.length) % radios.length;
    radios[next].focus();
    onMode(radios[next].dataset.tool);
  });
  document.addEventListener("keydown", (event) => {
    if (event.ctrlKey || event.metaKey || event.altKey || isEditableTarget(event.target)) return;
    const key = event.key.toLowerCase();
    const shortcuts = { v: "select", m: "move", z: "zoom", h: "handles" };
    if (shortcuts[key]) onMode(shortcuts[key]);
    if (event.key === "?") onHelp();
  });
}

export function renderToolRail(root, { mode, handlesAvailable }) {
  for (const button of root.querySelectorAll('[role="radio"]')) {
    const active = button.dataset.tool === mode;
    button.setAttribute("aria-checked", String(active));
    button.tabIndex = active ? 0 : -1;
    button.classList.toggle("active", active);
    if (button.dataset.tool === "handles") button.disabled = !handlesAvailable;
  }
}

function isEditableTarget(target) {
  if (!(target instanceof HTMLElement)) return false;
  const tag = target.tagName.toLowerCase();
  return target.isContentEditable || tag === "input" || tag === "select" || tag === "textarea";
}
```

- [ ] **Step 3: Add the rail while retaining the selection palette SVG markup**

Wrap the plot column and inspector in `<div class="editor-workspace">`. Add this as its first child:

```html
<nav id="toolRail" class="tool-rail" aria-label="Chart tools">
  <div class="tool-rail-modes" role="radiogroup" aria-label="Chart tools">
    <button type="button" role="radio" aria-checked="true" aria-label="Select"
      aria-keyshortcuts="V" data-tool="select">
      <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 3l13 9-6 2-3 6z"></path></svg>
    </button>
    <button type="button" role="radio" aria-checked="false" aria-label="Move"
      aria-keyshortcuts="M" data-tool="move" tabindex="-1">
      <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 2v20M2 12h20M12 2l-3 3m3-3 3 3M22 12l-3-3m3 3-3 3M12 22l-3-3m3 3 3-3M2 12l3-3m-3 3 3 3"></path></svg>
    </button>
    <button type="button" role="radio" aria-checked="false" aria-label="Zoom"
      aria-keyshortcuts="Z" data-tool="zoom" tabindex="-1">
      <svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="10" cy="10" r="6"></circle><path d="M14.5 14.5 21 21M7 10h6M10 7v6"></path></svg>
    </button>
    <button type="button" role="radio" aria-checked="false" aria-label="Handles"
      aria-keyshortcuts="H" data-tool="handles" tabindex="-1">
      <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 17c4-10 12-10 16 0"></path><circle cx="4" cy="17" r="2"></circle><circle cx="12" cy="8" r="2"></circle><circle cx="20" cy="17" r="2"></circle></svg>
    </button>
  </div>
  <button id="helpAction" class="tool-rail-help" type="button" aria-label="Help"
    aria-keyshortcuts="?" data-tool="help">
    <svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="9"></circle><path d="M9.7 9a2.5 2.5 0 1 1 3.4 2.3c-.8.4-1.1.9-1.1 1.7M12 17h.01"></path></svg>
  </button>
</nav>
```

Leave `#selectionMenu` and all of its operation SVG paths inside `.chart-shell` unchanged.

- [ ] **Step 4: Remove DOM mode state and feed interactions through a callback**

Delete `#mode` from `index.html` and delete `modeSelect` queries/listeners from `main.js`. Do not add
an `activeMode`/`graphMode` variable: Phase 1 already owns `store.getState().view.mode`. Pass the
store-backed callbacks required by the Phase 1 interaction interface:

```javascript
const chartContext = {
  svg,
  selectionMenu,
  selectedTerm,
  visualMode: () => store.getState().view.mode,
  zoomState: () => store.getState().view.zoomByTerm,
  showCi: () => store.getState().view.showCi,
  showContrib: () => store.getState().view.showContrib,
  buildProgress: () => buildProgress,
  groupDisplayMode: () => activeGroupDisplayMode(),
};

const interactions = bindInteractions({
  svg,
  mode: () => store.getState().view.mode,
  selectedTerm,
  currentTerm,
  currentSelection,
  hasState: () => selectSnapshot(store.getState()) !== null,
  setPreviewTerm: (term, payload) => store.update(
    (state) => setPreviewTerm(state, term, payload)
  ),
  setZoom: (term, range) => {
    const state = store.getState();
    actions.patchView({ zoomByTerm: { ...state.view.zoomByTerm, [term]: range } });
  },
  clearZoom: (term) => {
    const state = store.getState();
    const zoomByTerm = { ...state.view.zoomByTerm };
    delete zoomByTerm[term];
    actions.patchView({ zoomByTerm });
  },
  render,
  drawChart: (term, selection) => drawChart(term, selection, chartContext),
  actions,
});
```

Replace every `modeSelect.value` read in `interactions.js` with `context.mode()`. In `chart.js`, use
`context.visualMode()` and `context.zoomState()` directly. `startContributionBuild()` sets
`view.mode = "handles"` through `actions.patchView()`.

- [ ] **Step 5: Bind, render, and style the rail**

In `main.js`, define a working pre-inspector fallback before binding the rail:

```javascript
let openHelp = () => inspectorToggle.click();

bindToolRail({
  root: toolRail,
  onMode: (mode) => {
    stopContributionBuild();
    actions.patchView({
      mode,
      showContrib: mode === "handles" && canShowContributions(currentTerm())
    });
    render();
  },
  onHelp: () => openHelp(),
});
```

At the end of `render()`:

```javascript
renderToolRail(toolRail, {
  mode: store.getState().view.mode,
  handlesAvailable: Boolean(currentTerm() && currentTerm().controls),
});
```

Append to `styles/shell.css`:

```css
.tool-rail {
  display: flex;
  width: var(--tool-rail-width);
  min-height: 0;
  flex-direction: column;
  justify-content: space-between;
  padding: 4px;
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  background: var(--surface-subtle);
}

.tool-rail-modes {
  display: grid;
  gap: 4px;
}

.tool-rail button {
  display: grid;
  width: 34px;
  height: 34px;
  place-items: center;
  padding: 0;
}

.tool-rail button.active {
  border-color: var(--blue);
  background: var(--blue-soft);
  color: var(--blue);
}

.tool-rail svg {
  width: 19px;
  height: 19px;
  fill: none;
  stroke: currentColor;
  stroke-width: 1.8;
  stroke-linecap: round;
  stroke-linejoin: round;
}
```

- [ ] **Step 6: Run interaction regressions and commit**

Run:

```bash
rtk uv run pytest tests/editor/test_editor_workspace_browser.py --run-browser -q -k "tool_rail or svg_selection"
rtk uv run pytest tests/test_editor.py -q -k "interactions or chart or app_shell"
rtk npm run check:frontend
```

Expected: tool semantics pass and the existing pointer/operation regressions remain green.

Commit:

```bash
rtk git add src/superglm/editor/app tests/editor tests/test_editor.py
rtk git commit -m "Add the editor chart tool rail"
```

### Task 7: Build the shared inspector, Advanced pane, and Help pane

**Files:**
- Create: `src/superglm/editor/app/views/inspector.js`
- Create: `src/superglm/editor/app/views/help_drawer.js`
- Modify: `src/superglm/editor/app/index.html:218-248`
- Modify: `src/superglm/editor/app/main.js:70-180,387-450,509-620,800-850`
- Modify: `src/superglm/editor/app/styles/panels.css`
- Modify: `tests/editor/test_editor_workspace_browser.py`

- [ ] **Step 1: Write failing inspector and Help tests**

Append:

```python
def test_inspector_uses_one_slot_for_summary_history_advanced_and_help(open_editor_page):
    with open_editor_page() as (page, _session):
        inspector = page.get_by_role("complementary", name="Model inspector")
        assert inspector.get_by_role("tab").all_inner_texts() == ["Summary", "History", "Advanced", "Help"]
        inspector.get_by_role("tab", name="Advanced").click()
        assert inspector.get_by_label("Build animation duration").is_visible()
        page.get_by_role("button", name="Help").click()
        assert inspector.get_by_role("tabpanel", name="Help").is_visible()
        assert "Straighten selection" in inspector.inner_text()


def test_inspector_tabs_support_arrow_keys(open_editor_page):
    with open_editor_page() as (page, _session):
        summary = page.get_by_role("tab", name="Summary")
        summary.focus()
        page.keyboard.press("ArrowRight")
        assert page.get_by_role("tab", name="History").get_attribute("aria-selected") == "true"
```

Run:

```bash
rtk uv run pytest tests/editor/test_editor_workspace_browser.py --run-browser -q -k inspector
```

Expected: FAIL because Advanced and Help panes do not exist.

- [ ] **Step 2: Implement the inspector controller**

Create `src/superglm/editor/app/views/inspector.js`:

```javascript
// @ts-check

export function bindInspector({
  root, toggle, closeButton, scrim, onPanelChange, onOpenChange, isNarrow
}) {
  const tabs = Array.from(root.querySelectorAll('[role="tab"]'));
  let opener = null;

  function open(next = "summary", source = null) {
    opener = source || document.activeElement;
    onPanelChange(next);
    onOpenChange(true);
  }

  function close({ restoreFocus = true } = {}) {
    onOpenChange(false);
    if (restoreFocus && opener instanceof HTMLElement) opener.focus();
  }

  root.addEventListener("click", (event) => {
    const tab = event.target.closest("[data-inspector-tab]");
    if (tab) onPanelChange(tab.dataset.inspectorTab);
  });
  root.addEventListener("keydown", (event) => {
    const index = tabs.indexOf(event.target);
    if (index < 0 || !["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    const next = event.key === "Home"
      ? 0
      : event.key === "End"
        ? tabs.length - 1
        : (index + (event.key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length;
    tabs[next].focus();
    onPanelChange(tabs[next].dataset.inspectorTab);
  });
  toggle.addEventListener("click", () => onOpenChange(root.dataset.open !== "true"));
  closeButton.addEventListener("click", () => close());
  scrim.addEventListener("click", () => close());
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && root.dataset.open === "true" && isNarrow()) close();
  });

  return { open, close };
}

export function renderInspector({ root, toggle, scrim, panel, open, narrow }) {
  root.dataset.open = String(open);
  toggle.setAttribute("aria-expanded", String(open));
  scrim.hidden = !(open && narrow);
  for (const tab of root.querySelectorAll('[role="tab"]')) {
    const active = tab.dataset.inspectorTab === panel;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-selected", String(active));
    tab.tabIndex = active ? 0 : -1;
  }
  for (const pane of root.querySelectorAll('[role="tabpanel"]')) {
    pane.hidden = pane.dataset.inspectorPane !== panel;
  }
}
```

- [ ] **Step 3: Render shared help content**

Create `src/superglm/editor/app/views/help_drawer.js`:

```javascript
// @ts-check

import { HELP_SECTIONS, OPERATION_HELP, TOOL_HELP } from "./help_content.js";

export function renderHelpDrawer(root) {
  root.replaceChildren(...HELP_SECTIONS.map(sectionNode));
}

function sectionNode(section) {
  const sectionElement = document.createElement("section");
  sectionElement.className = "help-section";
  const heading = document.createElement("h3");
  heading.textContent = section.title;
  sectionElement.append(heading);
  const list = document.createElement("dl");
  for (const key of section.keys || []) {
    const entry = TOOL_HELP[key] || OPERATION_HELP[key];
    if (!entry) continue;
    const term = document.createElement("dt");
    term.textContent = entry.shortcut ? `${entry.title} (${entry.shortcut})` : entry.title;
    const description = document.createElement("dd");
    description.textContent = entry.body;
    list.append(term, description);
  }
  for (const item of section.items || []) {
    const term = document.createElement("dt");
    term.textContent = item;
    list.append(term);
  }
  sectionElement.append(list);
  return sectionElement;
}
```

- [ ] **Step 4: Expand the existing sidepanel markup without adding a column**

Rename `#summaryPanel` to `#inspector`, keep its current Summary and History contents, and replace
its tab list with:

```html
<aside id="inspector" class="inspector" aria-label="Model inspector" data-open="true">
  <div class="inspector-head">
    <div class="sidepanel-tabs" role="tablist" aria-label="Model inspector panels">
      <button id="summaryTab" class="sidepanel-tab active" type="button" role="tab"
        data-inspector-tab="summary" aria-selected="true" aria-controls="summaryPane">Summary</button>
      <button id="historyTab" class="sidepanel-tab" type="button" role="tab"
        data-inspector-tab="history" aria-selected="false" aria-controls="historyPane" tabindex="-1">History</button>
      <button id="advancedTab" class="sidepanel-tab" type="button" role="tab"
        data-inspector-tab="advanced" aria-selected="false" aria-controls="advancedPane" tabindex="-1">Advanced</button>
      <button id="helpTab" class="sidepanel-tab" type="button" role="tab"
        data-inspector-tab="help" aria-selected="false" aria-controls="helpPane" tabindex="-1">Help</button>
    </div>
    <button id="inspectorClose" class="inspector-close" type="button" aria-label="Close inspector">Close</button>
  </div>
```

Add `data-inspector-pane="summary"` and `data-inspector-pane="history"` to the existing panes. Add:

```html
<div id="advancedPane" class="sidepanel-pane" role="tabpanel" aria-labelledby="advancedTab"
  data-inspector-pane="advanced" hidden>
  <h2>Advanced editor controls</h2>
  <label id="buildDurationWrap" class="build-duration">
    <span>Build animation duration</span>
    <input id="buildDuration" type="range" min="4000" max="30000" step="500" value="10000"
      aria-label="Build animation duration">
    <output id="buildDurationValue">10s</output>
  </label>
  <div id="advancedTiming" class="advanced-timing" aria-live="polite"></div>
</div>
<div id="helpPane" class="sidepanel-pane help-pane" role="tabpanel" aria-labelledby="helpTab"
  data-inspector-pane="help" hidden></div>
</aside>
<button id="inspectorScrim" class="inspector-scrim" type="button"
  aria-label="Close inspector" hidden></button>
```

Remove the old Build duration control from the toolbar. Keep handle count, Contrib, and Build near
the chart because they are direct Handles-mode actions.

- [ ] **Step 5: Bind the inspector and move detailed timing to Advanced**

Import both `bindInspector` and `renderInspector` from `views/inspector.js`. In `main.js`, initialize:

```javascript
const narrowQuery = window.matchMedia("(max-width: 999px)");
renderHelpDrawer(helpPane);
const inspector = bindInspector({
  root: inspectorNode,
  toggle: inspectorToggle,
  closeButton: inspectorClose,
  scrim: inspectorScrim,
  onPanelChange: (panel) => {
    actions.patchView({ inspectorPane: panel });
    const snapshot = selectSnapshot(store.getState());
    if (panel === "history" && snapshot) renderHistory(snapshot.history, historyFrame);
  },
  onOpenChange: (open) => actions.patchView({ inspectorOpen: open }),
  isNarrow: () => narrowQuery.matches,
});
openHelp = () => inspector.open("help", helpAction);

store.subscribe(
  (state) => ({ pane: state.view.inspectorPane, open: state.view.inspectorOpen }),
  ({ pane, open }) => renderInspector({
    root: inspectorNode,
    toggle: inspectorToggle,
    scrim: inspectorScrim,
    panel: pane,
    open,
    narrow: narrowQuery.matches,
  }),
  (left, right) => left.pane === right.pane && left.open === right.open,
);
renderInspector({
  root: inspectorNode,
  toggle: inspectorToggle,
  scrim: inspectorScrim,
  panel: store.getState().view.inspectorPane,
  open: store.getState().view.inspectorOpen,
  narrow: narrowQuery.matches,
});
```

Replace the detailed `summaryNote` timing assignment in `showTimingStatus()` with:

```javascript
const details = formatTimingDetails(timing);
advancedTiming.textContent = payload.note ? `${payload.note} · ${details}` : details;
summaryNote.textContent = payload.note || "";
```

Delete `showSidepanelPane()` and its direct summary/history listeners; the inspector controller now
owns those tabs.

- [ ] **Step 6: Style the one-slot inspector**

Append to `styles/panels.css`:

```css
.inspector {
  width: var(--inspector-width);
  min-width: 320px;
  min-height: 0;
  display: flex;
  flex-direction: column;
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: 10px;
  background: var(--surface);
}

.inspector-head {
  display: flex;
  align-items: start;
  gap: 6px;
}

.sidepanel-tabs {
  min-width: 0;
  display: flex;
  flex: 1 1 auto;
  overflow-x: auto;
}

.sidepanel-tab.active {
  border-bottom-color: var(--orange);
}

.inspector-close {
  display: none;
}

@media (min-width: 1000px) {
  .editor-workspace:has(> .inspector[data-open="false"]) {
    grid-template-columns: var(--tool-rail-width) minmax(0, 1fr);
  }

  .inspector[data-open="false"] {
    display: none;
  }
}

.help-pane,
.advanced-timing,
.summary-frame,
.history-frame {
  user-select: text;
  -webkit-user-select: text;
}

.help-pane {
  overflow: auto;
}

.help-section h3 {
  margin: 12px 0 5px;
  font-size: 13px;
}

.help-section dl {
  margin: 0;
}

.help-section dt {
  margin-top: 7px;
  font-weight: 600;
}

.help-section dd {
  margin: 1px 0 0;
  color: var(--muted);
}
```

- [ ] **Step 7: Verify and commit**

Run:

```bash
rtk uv run pytest tests/editor/test_editor_workspace_browser.py --run-browser -q -k inspector
rtk npm run check:frontend
```

Expected: all inspector and keyboard-tab tests pass.

Commit:

```bash
rtk git add src/superglm/editor/app tests/editor/test_editor_workspace_browser.py
rtk git commit -m "Add the editor inspector and help drawer"
```

### Task 8: Make notebook, narrow, and short viewports usable

**Files:**
- Modify: `src/superglm/editor/app/styles/shell.css`
- Modify: `src/superglm/editor/app/styles/panels.css`
- Modify: `src/superglm/editor/app/views/inspector.js`
- Modify: `tests/editor/test_editor_workspace_browser.py`

- [ ] **Step 1: Write failing viewport geometry tests**

Append:

```python
def boxes_overlap(first: dict, second: dict) -> bool:
    return not (
        first["x"] + first["width"] <= second["x"]
        or second["x"] + second["width"] <= first["x"]
        or first["y"] + first["height"] <= second["y"]
        or second["y"] + second["height"] <= first["y"]
    )


def test_notebook_view_keeps_chart_and_inspector_side_by_side(open_editor_page):
    with open_editor_page(viewport={"width": 1180, "height": 720}) as (page, _session):
        chart = page.locator("#chart").bounding_box()
        inspector = page.locator("#inspector").bounding_box()
        assert chart["width"] >= 600
        assert inspector["x"] > chart["x"] + chart["width"]
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")


def test_narrow_view_uses_dismissible_inspector_drawer(open_editor_page):
    with open_editor_page(viewport={"width": 900, "height": 720}) as (page, _session):
        inspector = page.locator("#inspector")
        assert inspector.get_attribute("data-open") == "false"
        page.get_by_role("button", name="Help").click()
        assert inspector.get_attribute("data-open") == "true"
        page.keyboard.press("Escape")
        assert inspector.get_attribute("data-open") == "false"


def test_short_window_scrolls_without_chart_metric_overlap(open_editor_page):
    with open_editor_page(viewport={"width": 1180, "height": 540}) as (page, _session):
        chart = page.locator("#chart").bounding_box()
        metrics = page.locator(".metrics-strip").bounding_box()
        assert chart["height"] >= 360
        assert not boxes_overlap(chart, metrics)
        assert page.evaluate("document.documentElement.scrollHeight > window.innerHeight")
```

Run:

```bash
rtk uv run pytest tests/editor/test_editor_workspace_browser.py --run-browser -q -k "notebook_view or narrow_view or short_window"
```

Expected: at least the narrow and short tests fail because the existing page is fixed-height and
overflow-hidden.

- [ ] **Step 2: Install the normal workspace grid**

Append to `styles/shell.css`, replacing superseded `.app-shell`, `#editorView`, `.main`, and
`.plot-column` rules in `styles.css`:

```css
body {
  padding: 12px;
  overflow: auto;
}

.app-shell {
  position: relative;
  width: min(100%, 1400px);
  min-width: 0;
  min-height: calc(100dvh - 24px);
  display: grid;
  grid-template-rows: auto auto minmax(0, 1fr);
  margin: 0 auto;
}

#editorView {
  min-height: 0;
}

.editor-workspace {
  position: relative;
  min-width: 0;
  min-height: 0;
  display: grid;
  grid-template-columns: var(--tool-rail-width) minmax(0, 1fr) minmax(320px, var(--inspector-width));
  gap: 10px;
  align-items: stretch;
}

.plot-column {
  width: auto;
  min-width: 0;
  min-height: 0;
  display: grid;
  grid-template-rows: auto minmax(360px, 1fr) auto;
  gap: 8px;
}

.chart-shell {
  min-width: 0;
  min-height: 360px;
}

#chart {
  width: 100%;
  height: 100%;
  min-height: 360px;
}
```

- [ ] **Step 3: Add narrow drawer and short-window behavior**

Append:

```css
@media (max-width: 999px) {
  .editor-workspace {
    grid-template-columns: var(--tool-rail-width) minmax(0, 1fr);
  }

  .inspector {
    position: fixed;
    top: 8px;
    right: 8px;
    bottom: 8px;
    z-index: 61;
    width: min(390px, calc(100vw - var(--tool-rail-width) - 28px));
    min-width: 0;
    box-shadow: 0 18px 50px rgba(31, 35, 40, 0.24);
    transform: translateX(calc(100% + 20px));
    visibility: hidden;
    transition: transform 140ms ease, visibility 0s linear 140ms;
  }

  .inspector[data-open="true"] {
    transform: translateX(0);
    visibility: visible;
    transition-delay: 0s;
  }

  .inspector-close {
    display: inline-flex;
  }

  .inspector-scrim {
    position: fixed;
    inset: 0;
    z-index: 60;
    width: 100%;
    height: 100%;
    border: 0;
    background: rgba(31, 35, 40, 0.22);
  }
}

@media (max-height: 620px) {
  .app-shell {
    min-height: 0;
  }

  .editor-workspace,
  .plot-column {
    min-height: max-content;
  }

  .plot-column {
    grid-template-rows: auto 360px auto;
  }

}

@media (min-width: 1000px) and (max-height: 620px) {
  .inspector {
    min-height: 430px;
  }
}
```

- [ ] **Step 4: Synchronize the store's drawer default with the media query**

In `main.js`, add the synchronization around the `narrowQuery` created with the inspector:

```javascript
function syncViewport(event = narrowQuery) {
  actions.patchView({ inspectorOpen: !event.matches });
}

narrowQuery.addEventListener("change", syncViewport);
syncViewport();
```

Pass `isNarrow: () => narrowQuery.matches` into the inspector controller so it does not create a
second media query. The controller still owns only the transient opener element used for focus
restoration; the store owns open/pane state.

- [ ] **Step 5: Run the viewport matrix and commit**

Run:

```bash
rtk uv run pytest tests/editor/test_editor_workspace_browser.py --run-browser -q -k "notebook_view or narrow_view or short_window"
```

Expected: all three viewport tests pass, the normal chart is at least 600px wide, and the short page
scrolls rather than overlaps.

Commit:

```bash
rtk git add src/superglm/editor/app tests/editor/test_editor_workspace_browser.py
rtk git commit -m "Make the editor workspace responsive"
```

### Task 9: Implement pure categorical-axis geometry test-first

**Files:**
- Create: `src/superglm/editor/app/chart/geometry.js`
- Create: `tests/editor_frontend/chart_geometry.test.js`

- [ ] **Step 1: Write failing tick-density, truncation, and gutter tests**

Create `tests/editor_frontend/chart_geometry.test.js`:

```javascript
import assert from "node:assert/strict";
import test from "node:test";

import {
  evenlySpacedIndices,
  fitMeasuredLabel,
  planCategoricalAxis,
  rotatedExtent,
} from "../../src/superglm/editor/app/chart/geometry.js";

function measurement(label, widthPerCharacter = 7) {
  const characters = Array.from(label);
  return {
    fullWidth: characters.length * widthPerCharacter,
    prefixWidths: characters.map((_, index) => (index + 1) * widthPerCharacter),
    ellipsisWidth: widthPerCharacter,
    height: 11,
  };
}

test("tick reduction retains first, last, and evenly spaced interior categories", () => {
  assert.deepEqual(evenlySpacedIndices(10, 5), [0, 2, 5, 7, 9]);
  assert.deepEqual(evenlySpacedIndices(3, 5), [0, 1, 2]);
});

test("measured truncation uses a Unicode end ellipsis without changing the source", () => {
  const source = "MyReallyLongCategoryNameThatWouldNeverFit";
  const fitted = fitMeasuredLabel(source, measurement(source), 112);
  assert.equal(fitted, "MyReallyLongCat…");
  assert.equal(source, "MyReallyLongCategoryNameThatWouldNeverFit");
});

test("rotated extent projects width into the bottom gutter", () => {
  const extent = rotatedExtent(100, 11, -45);
  assert.ok(extent.width > 70 && extent.width < 80);
  assert.ok(extent.height > 70 && extent.height < 80);
});

test("categorical layout reserves title space and preserves full labels", () => {
  const labels = Array.from({ length: 10 }, (_, index) => `TerritoryCategoryNumber${index + 1}`);
  const layout = planCategoricalAxis({
    values: labels.map((_, index) => index),
    labels,
    measurements: labels.map(label => measurement(label)),
    availableWidth: 788,
    svgHeight: 520,
    baseLeft: 76,
    baseBottom: 72,
  });
  assert.equal(layout.ticks[0].fullLabel, labels[0]);
  assert.equal(layout.ticks.at(-1).fullLabel, labels.at(-1));
  assert.ok(layout.ticks.some(tick => tick.displayLabel.endsWith("…")));
  assert.ok(layout.bottom > 72);
  assert.ok(layout.titleY > layout.axisY + layout.maxLabelHeight);
  assert.ok(layout.titleY + layout.titleHeight <= 520 - 8);
});
```

Run:

```bash
rtk node --test --test-name-pattern "tick reduction|measured truncation|rotated extent|categorical layout" tests/editor_frontend/chart_geometry.test.js
```

Expected: FAIL because `chart/geometry.js` does not exist.

- [ ] **Step 2: Implement the pure geometry module**

Create `src/superglm/editor/app/chart/geometry.js`:

```javascript
// @ts-check

const MAX_TICKS = 30;
const MIN_ANGLED_SLOT = 56;
const TICK_OFFSET = 18;
const TITLE_GAP = 18;
const OUTER_PAD = 12;

export function evenlySpacedIndices(count, maximum) {
  if (count <= 0 || maximum <= 0) return [];
  if (count <= maximum) return Array.from({ length: count }, (_, index) => index);
  if (maximum === 1) return [0];
  const indices = [];
  for (let position = 0; position < maximum; position++) {
    indices.push(Math.round(position * (count - 1) / (maximum - 1)));
  }
  return Array.from(new Set(indices)).sort((left, right) => left - right);
}

export function fitMeasuredLabel(label, measurement, budget) {
  if (measurement.fullWidth <= budget) return label;
  const characters = Array.from(label);
  const prefixBudget = Math.max(0, budget - measurement.ellipsisWidth);
  let low = 0;
  let high = characters.length;
  while (low < high) {
    const middle = Math.ceil((low + high) / 2);
    const width = middle === 0 ? 0 : measurement.prefixWidths[middle - 1];
    if (width <= prefixBudget) low = middle;
    else high = middle - 1;
  }
  return `${characters.slice(0, low).join("")}…`;
}

export function rotatedExtent(width, height, degrees) {
  const radians = Math.abs(degrees) * Math.PI / 180;
  return {
    width: Math.abs(width * Math.cos(radians)) + Math.abs(height * Math.sin(radians)),
    height: Math.abs(width * Math.sin(radians)) + Math.abs(height * Math.cos(radians)),
  };
}

export function planCategoricalAxis({
  values,
  labels,
  measurements,
  availableWidth,
  svgHeight,
  baseLeft,
  baseBottom,
  titleHeight = 14,
}) {
  const densityLimit = Math.max(2, Math.floor(availableWidth / MIN_ANGLED_SLOT) + 1);
  const indices = evenlySpacedIndices(labels.length, Math.min(MAX_TICKS, densityLimit));
  const slot = availableWidth / Math.max(indices.length - 1, 1);
  const horizontalBudget = Math.max(24, slot - 10);
  const rotate = indices.some(index => measurements[index].fullWidth > horizontalBudget);
  const angle = rotate ? -45 : 0;
  const radians = Math.abs(angle) * Math.PI / 180;
  const edgeBudget = rotate
    ? Math.max(24, (baseLeft - 8 - measurements[indices[0]].height * Math.sin(radians)) / Math.cos(radians))
    : horizontalBudget;
  const rotatedBudget = rotate
    ? Math.max(24, (slot - measurements[indices[0]].height * Math.sin(radians)) / Math.cos(radians))
    : horizontalBudget;
  const labelBudget = Math.min(160, edgeBudget, rotatedBudget);
  const ticks = indices.map(index => {
    const displayLabel = fitMeasuredLabel(labels[index], measurements[index], labelBudget);
    const characters = Array.from(displayLabel.endsWith("…") ? displayLabel.slice(0, -1) : displayLabel);
    const prefixWidth = characters.length
      ? measurements[index].prefixWidths[characters.length - 1]
      : 0;
    const displayWidth = displayLabel.endsWith("…")
      ? prefixWidth + measurements[index].ellipsisWidth
      : measurements[index].fullWidth;
    const extent = rotatedExtent(displayWidth, measurements[index].height, angle);
    return {
      index,
      value: values[index],
      fullLabel: labels[index],
      displayLabel,
      angle,
      anchor: rotate ? "end" : "middle",
      width: extent.width,
      height: extent.height,
    };
  });
  const maxLabelHeight = Math.max(0, ...ticks.map(tick => tick.height));
  const bottom = Math.max(
    baseBottom,
    Math.ceil(TICK_OFFSET + maxLabelHeight + TITLE_GAP + titleHeight + OUTER_PAD),
  );
  const axisY = svgHeight - bottom;
  const titleY = axisY + TICK_OFFSET + maxLabelHeight + TITLE_GAP;
  return { ticks, bottom, axisY, titleY, titleHeight, maxLabelHeight, labelBudget };
}
```

- [ ] **Step 3: Run type and unit checks**

Run:

```bash
rtk npm run check:frontend
```

Expected: all geometry and popover tests pass and TypeScript reports no errors in the new modules.

- [ ] **Step 4: Commit pure axis geometry**

```bash
rtk git add src/superglm/editor/app/chart/geometry.js tests/editor_frontend/chart_geometry.test.js
rtk git commit -m "Plan categorical axis labels from measured text"
```

### Task 10: Integrate measured categorical axes into the SVG renderer

**Files:**
- Create: `tests/editor/test_editor_axis_browser.py`
- Modify: `src/superglm/editor/app/chart.js:1-177,676-699,779-805`
- Modify: `src/superglm/editor/app/styles/chart.css`
- Modify: `tests/test_editor.py:4368-4389`

- [ ] **Step 1: Write failing real-font clipping and string-integrity tests**

Create `tests/editor/test_editor_axis_browser.py`:

```python
from __future__ import annotations

import pytest

pytest.importorskip("playwright.sync_api")
pytestmark = pytest.mark.browser


def test_ten_categories_do_not_intersect_title_or_svg_viewport(open_editor_page):
    with open_editor_page(selected_term="territory") as (page, _session):
        result = page.locator("#chart").evaluate(
            """svg => {
              const root = svg.getBoundingClientRect();
              const title = svg.querySelector('.x-axis-title').getBoundingClientRect();
              const ticks = [...svg.querySelectorAll('.x-tick-label')].map(node => node.getBoundingClientRect());
              return { root, title, ticks };
            }"""
        )
        for tick in result["ticks"]:
            assert tick["left"] >= result["root"]["left"] - 0.5
            assert tick["right"] <= result["root"]["right"] + 0.5
            assert tick["bottom"] < result["title"]["top"]
        assert result["title"]["bottom"] <= result["root"]["bottom"] + 0.5


def test_long_labels_truncate_only_on_screen_and_disclose_exact_value(open_editor_page):
    full = "MyReallyLongCategoryNameThatWouldNeverFit"
    with open_editor_page(selected_term="long_category") as (page, session):
        tick = page.locator(f'.x-tick-label[aria-label="{full}"]')
        assert tick.inner_text().endswith("…")
        assert tick.get_attribute("data-full-label") == full
        tick.focus()
        popover = page.get_by_role("tooltip")
        assert popover.is_visible()
        assert full in popover.inner_text()
        assert full in session.terms["long_category"].levels
        assert all(not level.endswith("…") for level in session.terms["long_category"].levels)
```

Run:

```bash
rtk uv run pytest tests/editor/test_editor_axis_browser.py --run-browser -q
```

Expected: FAIL because the current fixed 72-unit gutter and fixed title position collide.

- [ ] **Step 2: Add actual SVG text measurement without mutating payload labels**

Import geometry at the top of `chart.js`:

```javascript
import { planCategoricalAxis } from "./chart/geometry.js";
```

Add this DOM-only measurement helper before `xTicks()`:

```javascript
function measureCategoricalLabels(svg, labels) {
  const layer = el("g", { class: "axis-measure-layer", "aria-hidden": "true" });
  layer.setAttribute("visibility", "hidden");
  svg.appendChild(layer);
  const probe = text(layer, 0, 0, "", "tick-label", "start");
  const ellipsisWidth = measureText(probe, "…");
  const measurements = labels.map(label => {
    const characters = Array.from(String(label));
    const prefixWidths = characters.map((_, index) => measureText(probe, characters.slice(0, index + 1).join("")));
    probe.textContent = String(label);
    const box = probe.getBBox();
    return {
      fullWidth: prefixWidths.at(-1) || 0,
      prefixWidths,
      ellipsisWidth,
      height: Math.max(1, box.height),
    };
  });
  layer.remove();
  return measurements;
}

function measureText(probe, value) {
  probe.textContent = value;
  return probe.getComputedTextLength();
}
```

- [ ] **Step 3: Compute layout before scales and draw from the plan**

In `drawChart()`, resolve `view` before margins. Replace the fixed margin creation with:

```javascript
const width = 940;
const height = 520;
const baseMargin = { left: 76, right: 76, top: 48, bottom: 72 };
const categoricalLayout = view.levels
  ? planCategoricalAxis({
      values: view.x,
      labels: view.levels.map(String),
      measurements: measureCategoricalLabels(svg, view.levels.map(String)),
      availableWidth: width - baseMargin.left - baseMargin.right,
      svgHeight: height,
      baseLeft: baseMargin.left,
      baseBottom: baseMargin.bottom,
    })
  : null;
const margin = {
  ...baseMargin,
  bottom: categoricalLayout ? categoricalLayout.bottom : baseMargin.bottom,
};
const innerW = width - margin.left - margin.right;
const innerH = height - margin.top - margin.bottom;
```

For categorical ticks, draw only `categoricalLayout.ticks`, filtered to values within the current
`xMin`/`xMax`. Draw each with these exact accessibility attributes:

```javascript
const tickLabel = text(
  svg,
  tickX,
  margin.top + innerH + 18,
  tick.displayLabel,
  tick.angle ? "tick-label x-tick-label angled" : "tick-label x-tick-label",
  tick.anchor,
);
tickLabel.setAttribute("data-full-label", tick.fullLabel);
tickLabel.setAttribute("data-popover-title", "Category");
tickLabel.setAttribute("data-popover-body", tick.fullLabel);
tickLabel.setAttribute("aria-label", tick.fullLabel);
tickLabel.setAttribute("tabindex", "0");
if (tick.angle) {
  tickLabel.setAttribute("transform", `rotate(${tick.angle} ${tickX} ${margin.top + innerH + 18})`);
}
```

Draw the x-axis title at `categoricalLayout.titleY` and add class `x-axis-title`; keep continuous
axes at `height - 20`. Remove `MAX_LEVEL_LABELS` and the categorical branches from old `xTicks()`;
retain its continuous numeric ticks only.

- [ ] **Step 4: Make tick labels focusable and pointer-addressable**

Append to `styles/chart.css`:

```css
.x-tick-label {
  pointer-events: auto;
  cursor: help;
}

.x-tick-label:focus-visible {
  outline: none;
  paint-order: stroke;
  stroke: var(--surface);
  stroke-width: 4px;
  fill: var(--blue);
}

.axis-measure-layer {
  pointer-events: none;
}
```

Narrow the legacy global rule in `styles.css` from `svg text { pointer-events: none; ... }` to:

```css
svg text:not(.x-tick-label) {
  pointer-events: none;
  user-select: none;
  -webkit-user-select: none;
}
```

- [ ] **Step 5: Replace source-string axis assertions with behavior contracts**

Delete `test_editor_chart_labels_ordered_categorical_levels_up_to_thirty` from `tests/test_editor.py`.
Keep `test_editor_chart_points_have_relativity_exposure_tooltips`, and add this static integrity check:

```python
def test_editor_axis_uses_display_only_measured_geometry():
    root = Path(__file__).resolve().parents[1] / "src/superglm/editor/app"
    chart_js = (root / "chart.js").read_text()
    geometry_js = (root / "chart/geometry.js").read_text()

    assert "measureCategoricalLabels" in chart_js
    assert "data-full-label" in chart_js
    assert "planCategoricalAxis" in chart_js
    assert "fitMeasuredLabel" in geometry_js
    assert "term.levels[" not in geometry_js
```

- [ ] **Step 6: Run axis and editor regressions**

Run:

```bash
rtk npm run check:frontend
rtk uv run pytest tests/editor/test_editor_axis_browser.py --run-browser -q
rtk uv run pytest tests/test_editor.py -q -k "axis or chart_points or grouped_display"
```

Expected: pure geometry, actual browser bounds, tooltip disclosure, and source integrity all pass.

- [ ] **Step 7: Commit the clipping fix**

```bash
rtk git add src/superglm/editor/app tests/editor_frontend tests/editor/test_editor_axis_browser.py tests/test_editor.py
rtk git commit -m "Fix categorical axis label clipping"
```

### Task 11: Harden accessibility, busy behavior, and text selection

**Files:**
- Modify: `src/superglm/editor/app/index.html:10-260`
- Modify: `src/superglm/editor/app/main.js:387-407,454-466`
- Modify: `src/superglm/editor/app/styles/tokens.css`
- Modify: `src/superglm/editor/app/styles/shell.css`
- Modify: `src/superglm/editor/app/styles/chart.css`
- Modify: `src/superglm/editor/app/styles/panels.css`
- Modify: `tests/editor/test_editor_workspace_browser.py`

- [ ] **Step 1: Write failing accessibility behavior tests**

Append:

```python
def test_busy_state_makes_editor_regions_inert(open_editor_page):
    with open_editor_page() as (page, _session):
        page.evaluate("window.__superglmTest.setAppBusy(true, 'Testing busy state', 'Waiting')")
        assert page.locator("#editorView").get_attribute("inert") == ""
        assert page.locator("#appBar").get_attribute("inert") == ""
        assert page.locator("#appBusyOverlay").get_attribute("aria-live") == "polite"
        page.evaluate("window.__superglmTest.setAppBusy(false)")
        assert page.locator("#editorView").get_attribute("inert") is None


def test_non_chart_text_is_selectable_and_reduced_motion_disables_spinner(open_editor_page):
    with open_editor_page() as (page, _session):
        assert page.locator("#helpPane").evaluate("node => getComputedStyle(node).userSelect") in {"auto", "text"}
        page.emulate_media(reduced_motion="reduce")
        duration = page.locator(".busy-spinner").evaluate("node => getComputedStyle(node).animationDuration")
        assert duration in {"0s", "0.000001s"}


def test_escape_closes_a_focused_icon_popover_immediately(open_editor_page):
    with open_editor_page() as (page, _session):
        button = page.get_by_role("button", name="Straighten selection")
        button.focus()
        assert page.get_by_role("tooltip").is_visible()
        page.keyboard.press("Escape")
        assert page.get_by_role("tooltip").is_hidden()
```

Run:

```bash
rtk uv run pytest tests/editor/test_editor_workspace_browser.py --run-browser -q -k "busy_state or selectable or escape_closes"
```

Expected: busy inertness fails; the other assertions identify any remaining global selection or
motion leakage.

- [ ] **Step 2: Give status and error regions explicit semantics**

Ensure `index.html` contains these distinct regions:

```html
<div id="status" role="status" aria-live="polite"></div>
<div id="appAlert" class="app-alert" role="alert" aria-live="assertive" hidden>
  <span id="appAlertMessage"></span>
  <button id="appAlertRetry" type="button">Retry</button>
  <button id="appAlertDismiss" type="button">Dismiss</button>
</div>
```

The state/action workstream supplies the retry callback and message. This plan only ensures the
region is persistent, keyboard reachable, and visually distinct.

- [ ] **Step 3: Make actual regions inert during blocking mutations**

Update `setAppBusy()` in `main.js`:

```javascript
function setAppBusy(active, title = "Working...", detail = "") {
  if (!appShell || !appBusyOverlay) return;
  if (appBusyTimer !== null) {
    clearInterval(appBusyTimer);
    appBusyTimer = null;
  }
  for (const region of [appBar, contextBar, editorView, reportPanel]) {
    region.toggleAttribute("inert", active);
  }
  appShell.classList.toggle("is-busy", active);
  appShell.setAttribute("aria-busy", String(active));
  appBusyOverlay.hidden = !active;
  if (!active) return;
  appBusyStarted = performance.now();
  const update = () => {
    const elapsed = performance.now() - appBusyStarted;
    appBusyTitle.textContent = title;
    appBusyDetail.textContent = `${detail || "Refitting model"} · ${formatMilliseconds(elapsed)} elapsed`;
  };
  update();
  appBusyTimer = window.setInterval(update, 250);
}
```

Expose it for the narrowly scoped browser assertion only in development/test mode:

```javascript
if (new URLSearchParams(window.location.search).get("test") === "1") {
  window.__superglmTest = { setAppBusy };
}
```

Change the browser test call to `window.__superglmTest.setAppBusy(...)`; do not expose a generic
mutation API.

- [ ] **Step 4: Limit selection suppression and complete focus/contrast rules**

Remove `user-select: none` from `body`. Add:

```css
.chart-shell,
#chart,
.tool-rail,
.selection-menu {
  user-select: none;
  -webkit-user-select: none;
}

.summary-frame,
.history-frame,
.report-frame,
.help-pane,
.advanced-timing,
.app-alert {
  user-select: text;
  -webkit-user-select: text;
}

.app-alert {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
  padding: 8px 10px;
  border: 1px solid #f1aeb5;
  border-radius: var(--radius-sm);
  background: #fff1f2;
  color: #842029;
}

button:disabled {
  cursor: not-allowed;
  opacity: 0.55;
}
```

Replace hard-coded exposure colors in `styles.css` with `var(--yellow)` and
`var(--yellow-border)`. Replace all remaining `var(--gray)` references with `var(--muted)` or add
`--gray: var(--muted)` only as a temporary compatibility alias until the final grep is clean.

- [ ] **Step 5: Run accessibility checks and commit**

Run:

```bash
rtk uv run pytest tests/editor/test_editor_workspace_browser.py --run-browser -q -k "busy_state or selectable or escape_closes or inspector_tabs"
rtk rg -n 'var\(--orange\)|var\(--text\)|user-select: none' src/superglm/editor/app
rtk npm run check:frontend
```

Expected: browser tests pass; every reported CSS token is defined; selection suppression appears
only on the chart/tool interaction surfaces.

Commit:

```bash
rtk git add src/superglm/editor/app tests/editor/test_editor_workspace_browser.py
rtk git commit -m "Harden editor accessibility semantics"
```

### Task 12: Replace brittle frontend assertions and run the complete workspace gate

**Files:**
- Modify: `tests/test_editor.py:3949-4450`
- Modify: `tests/editor/test_editor_workspace_browser.py`
- Modify: `tests/editor/test_editor_axis_browser.py`
- Modify: `.github/workflows/ci.yml`
- Modify: `.github/workflows/dev-ci.yml`

- [ ] **Step 1: Reduce the giant shell source test to packaging contracts**

In `test_widget_app_shell_contains_drag_editor`, keep only assertions that belong to Python asset
packaging and server boundaries:

```python
        assert '<script type="module" src="/assets/main.js"></script>' in shell
        assert 'role="tablist" aria-label="Editor views"' in shell
        assert 'id="chart"' in shell
        assert 'id="selectionMenu"' in shell
        assert 'id="inspector"' in shell
        assert "X-SuperGLM-Editor-Token" in js
        assert "/drag" in js
        assert "/control" in js
        assert "/metrics" in js
        assert "/summary" in js
```

Delete assertions about exact CSS declarations, absence of Undo/Redo, the old mode select,
`MAX_LEVEL_LABELS`, and exact internal function names now covered by pure or browser tests.

- [ ] **Step 2: Add a final user-facing browser journey**

Append:

```python
def test_analyst_can_discover_edit_undo_redo_help_and_save(open_editor_page):
    with open_editor_page() as (page, _session):
        page.locator("#chart .point").nth(2).click()
        page.get_by_role("button", name="Increase selection").focus()
        assert "Increase selected relativities" in page.get_by_role("tooltip").inner_text()
        page.get_by_role("button", name="Increase selection").click()
        undo = page.get_by_role("button", name="Undo edit")
        assert undo.is_enabled()
        undo.click()
        redo = page.get_by_role("button", name="Redo edit")
        assert redo.is_enabled()
        page.get_by_role("button", name="Help").click()
        assert page.get_by_role("tabpanel", name="Help").is_visible()
        assert page.get_by_role("button", name="Save edited model").is_visible()
```

- [ ] **Step 3: Run focused frontend and browser gates**

Run:

```bash
rtk npm ci
rtk npm run check:frontend
rtk uv sync --extra dev
rtk uv run playwright install chromium
rtk uv run pytest tests/editor/ -q -m browser --run-browser
rtk uv run pytest tests/test_editor.py -q
```

Expected: Node unit/type checks, all workspace/axis browser tests, and the complete editor Python
suite pass.

- [ ] **Step 4: Run repository lint and non-slow regression tests**

Run:

```bash
rtk uv run ruff check src/ tests/
rtk uv run pytest tests/ -q -m "not slow"
```

Expected: Ruff passes and the non-slow suite reports no failures.

- [ ] **Step 5: Check packaging and working-tree scope**

Run:

```bash
rtk uv build
rtk git status --short
rtk git diff --check
```

Expected: the wheel builds with nested JavaScript/CSS assets, `git diff --check` reports no
whitespace errors, and the pre-existing `docs/notebooks/spline diagnostics/` directory remains
untouched and untracked.

- [ ] **Step 6: Commit the browser-backed frontend contract**

```bash
rtk git add tests .github/workflows/ci.yml .github/workflows/dev-ci.yml
rtk git commit -m "Replace editor source checks with browser coverage"
```

## Final acceptance checklist

- [ ] The production editor uses native source modules with no framework, Tailwind, bundler, or
  production Node runtime.
- [ ] Existing SVG paths and all curve operations remain available.
- [ ] The rail exposes Select, Move, Zoom, Handles, and Help with exclusive accessible state.
- [ ] Icon popovers wait 350ms for pointer hover, open immediately on focus, disappear immediately
  on leave, and close with Escape.
- [ ] `linearise` is presented as “Straighten selection” with the approved interpolation copy.
- [ ] Summary, History, Advanced, and Help share one inspector slot.
- [ ] At 1180x720 the inspector is side-by-side; below 1000px it is a dismissible drawer; at short
  heights the page scrolls without chart/metric overlap.
- [ ] Categorical label truncation changes only rendered text, uses a Unicode end ellipsis, retains
  exact model strings, and exposes exact values on hover/focus and point tooltips.
- [ ] Tick labels and the x-axis title remain inside the SVG and never intersect in browser tests.
- [ ] Undo/Redo/Save are visible, tablists support keyboard navigation, busy regions are inert,
  reduced motion is honored, and non-chart text is selectable.
- [ ] Pure frontend, real-browser, editor Python, lint, packaging, and non-slow regression gates pass.
