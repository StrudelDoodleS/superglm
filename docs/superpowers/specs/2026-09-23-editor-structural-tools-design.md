# Editor structural tools: design

- **Date:** 2026-09-23
- **Branch:** `feat/editor-structural-tools`, cut from `origin/master` at `2f1436be`
- **Status:** draft for review
- **Release impact (advisory):** `release:minor` — new editor features and one visible style change

## 1. What ships

Five additions to the browser editor, delivered as one pull request:

1. **Refresh from Python.** An app-bar icon that re-reads the Python session and redraws the
   chart, summary and metrics. Nothing refits.
2. **Set reference and refit.** A selection-palette icon that pins one level of a categorical or
   ordered categorical term as its reference and refits.
3. **Breaks tool, Transform and refit.** A tool-rail mode for placing breaks on the plot, a form
   switch (piecewise polynomial, spline with knots at the breaks, orthogonal polynomial), and an
   icon that replaces the term with that form and refits the model.
4. **Restore previous structure, Revert to original model.** One undo stack for every structural
   step, with its icon in the action bar, and an app-bar icon that returns to the model the editor
   was opened with.
5. **Original-model line style.** The original-model curve becomes a solid, wider, semi-transparent
   grey line instead of a thin dashed one.

## 2. Who it is for, and the rules it follows

The editor is Max's daily pricing workspace and is replacing Emblem in that work. Three rules
govern every addition:

- **Nothing intrudes** (Max, 2026-09-23: "it CANNOT be intrusive or annoying to the user"). No
  prompt, popup or banner appears in response to an ordinary gesture such as selecting levels.
  Each action is its own compact icon in the existing SVG style, named in the delayed hover
  popover and in Help. Caveats live in hover text, Help and the summary, never in a new dialog.
  The one allowed interruption is the existing structural confirmation, shown only when manual
  edit history (or, for Revert, structural steps) would be discarded.
- **Python owns every confirmed change** (`docs/development/internals/editor-frontend.md`). The
  browser holds drafts and view state only; every mutation returns an authoritative snapshot.
- **Existing behaviour stays unless this design names the change.** The chart, exposure bars,
  palette operations and evidence panels are untouched except where §4 says otherwise.

## 3. Research record

### 3.1 What the problems are

- **Transform.** Replace one additive term's basis with (a) a C⁰ piecewise polynomial on fixed,
  stated breaks — the grafted or segmented polynomial (Gallant & Fuller 1973, *JASA* 68:144–147)
  with degree-0 plateau tails (Anderson & Nelson 1975, *Biometrics* 31:303–318); (b) a regression
  spline with fixed knots; or (c) an orthogonal polynomial. Then refit the model by the same
  (penalized) likelihood. **There are no new numerics.** Every basis already exists in the library
  — `Piecewise`, `Spline(knots=...)`, `Polynomial`, and `OrderedCategorical(basis=...)` with
  band-name breaks and per-segment `degrees=` — and each was swept when it shipped (the
  2026-08-06 Piecewise spec and the 2026-08-09 ordered-basis sweep, 22 citations). The editor
  composes them; it does not re-derive them.
- **Reference level.** Changing a factor's baseline category is a reparametrisation of that
  factor's coefficients.
- **Refresh.** Two views of one authoritative store (the notebook and the browser) drift apart.
  This is ordinary cache coherence.

### 3.2 What established practice does

*A Practitioner's Guide to Generalized Linear Models* (Anderson, Feldblum, Modlin, Schirmacher,
Schirmacher and Thandi; CAS study note, 3rd ed., 2007 — the Towers Perrin guide from Emblem's
lineage) recommends exactly the workflow the transform supports:

- §2.16: start with narrowly defined categorical factors, and "if the categorical factor presents
  GLM parameter estimates which appear appropriate for modeling with a polynomial, then the
  polynomial in the variate may be used in place of the categorical factor."
- §2.25–2.28: judge that choice by parameter standard errors, type III tests, consistency with
  time, and common sense.
- §2.103: refitting "using polynomial terms as variates within the GLM" is the more scientific
  form of smoothing.

So the transformed term reports full inference, as practice does (§4.7).

### 3.3 Inference after choosing breaks by eye

Placing a break where the fitted curve visibly bends, then testing for a kink there on the same
data, is post-selection inference. The known-break Wald statistic is N(0,1); the selected-break
statistic behaves like the supremum of a Gaussian process over the candidate locations (Shi &
Lund, arXiv:2511.17942, verified in the 2026-08-06 Piecewise sweep). The library's own Piecewise
reporting contract names this as condition 3 (`inference/coef_tables.py`).

**Measured** (script in Appendix A). The truth is a straight line with no kink, across 4,000
datasets with 19 candidate break locations:

| How the break was chosen | False "significant kink" at p < 0.05 |
|---|---|
| Fixed in advance | 5.0% |
| Placed at the visible bend, tested on the same data | 23.3% |
| Placed on one half of the data, tested on the other half | 5.4% |

The third row is the remedy: held-out evaluation restores the nominal rate. The editor already
carries validation data, and its metrics strip compares the original model with the current one.

**Decision (Max, 2026-09-23):** report standard errors and tests as usual, as Emblem and the
guide above do, and add one line of disclosure. Do not withhold anything.

### 3.4 The reference level and the fit

- **Unpenalized factor:** fitted values do not depend on the reference. Changing it re-expresses
  every relativity against the new level, and the base rate absorbs the difference.
- **Ordered categorical smooth:** invariant when the constant lies in the smoothing penalty's null
  space. This is expected for the difference and derivative penalties used here, but it is pinned
  by a test (§7) rather than assumed.
- **Group lasso under reference coding:** the fit depends on the reference (Gertheiss & Tutz 2010,
  *Ann. Appl. Stat.* 4(4):2150–2180; "A note on coding and standardization of categorical
  variables in (sparse) group lasso regression", *J. Statist. Plann. Inference*, 2019).
  `selection_penalty` is off by default (`model/api.py`), so only an explicit selection penalty
  makes set-reference change the fit.

**Measured on `origin/master`:** with `Categorical()` (default `base="most_exposed"`), collapsing
levels C and D moved the reference from B to the new group C+D. Every displayed relativity
changed; B went from 1.000 to 0.746. This motivates pinning a reference explicitly.

### 3.5 Refresh

The standard options are pull on demand, revalidate on focus, and server push. **Pull on demand**
is chosen (Max, 2026-09-23); the other two are follow-ups (§8).

### 3.6 New territory?

None. Every element composes established methods and existing library features. The one choice
without a textbook answer — withhold or disclose post-selection tests — is settled by Max's
decision in §3.3.

## 4. Design

### 4.1 Refresh from Python

- **Placement:** an icon button in the app bar after Redo, before Export. Accessible name and
  popover title "Refresh from Python"; popover body: "Re-read the Python session and redraw. Use
  it after changing the session in the notebook. Nothing refits." It is disabled while a blocking
  mutation runs.
- **Behaviour:** `GET /state` (existing route), validated as an `EditorSnapshot`, is committed
  through `commitRemote`. A fresh `/state` always carries a newer `state_generation`, so it is
  accepted even at an equal `model_revision` (a notebook-side selection change does not advance
  the revision). The visible evidence panels (metrics, summary, active report) are then
  re-requested immediately. The status line reads "Synced with Python · revision N".
- **Python side:** a fixed-offset refit is stored with the session revision it was computed at.
  The `"refit"` summary source reports it unavailable once the revision has moved. Today a
  notebook-side edit never calls `EditorWidget._invalidate_refit`, so a refreshed view could
  otherwise show an out-of-date offset refit.
- **Not included:** polling, revalidate-on-focus, server push (§8).

Any browser mutation already returns the full authoritative snapshot, so the view is only stale
between a notebook-side change and the next browser action or refresh.

### 4.2 Set reference and refit

- **Placement:** an icon in the selection palette, after Collapse and Ungroup. Accessible name
  and popover title "Set reference and refit"; popover body: "Pin this level as the reference
  (relativity 1.00) and refit. Predictions stay the same unless a selection penalty is on." It is
  visible when the active term is categorical or ordered categorical, exactly one displayed level
  is selected, and that level is not already the reference. Help carries the same text.
- **Seeing the reference:** the context bar gains a chip, "reference ⟨level⟩ · most exposed",
  "· first" or "· pinned", beside the existing kind and edf chips.
- **Python:** `EditorSession.replace_with_reference_level(term, level, *, method="auto", ...)`
  calls `reference_feature_spec(model, term, level)`, which builds the replacement spec. It then
  follows the collapse path: `clone_with_replaced_feature` → `fit_refit_model` → push onto the
  structural stack → `replace_in_force_model`. The refit model is stamped with
  `format: "superglm.editor.reference_level.v1"`.
- **The replacement is built fresh, never as a mutated fitted copy.** A fitted `Categorical`
  keeps its resolved base sticky (`_resolve_base` prefers the fitted `_base_level` over
  `self.base`), and `clone_without_features` deep-copies fitted specs. Setting `.base` on such a
  copy would silently keep the old reference. `collapse.py` already constructs fresh specs; this
  follows it.
- **Level vocabulary:** in the collapsed display a displayed level can be a group label. The base
  is then that group label, as `_collapsed_base` already allows.
- **Refused, with a named `EditorValueError`:** a special level of an ordered categorical
  (specials are free effects outside the ordered axis); a term used by an interaction (as
  collapse refuses); any term without levels.
- **It sticks:** collapse and ungroup already keep a concrete base (`_collapsed_base`,
  `_valid_base_after_ungroup`). Collapsing the reference into a group makes the group the
  reference.

### 4.3 Breaks tool, Transform and refit

**Mode.** A new tool-rail mode, Breaks (shortcut `B`, after Handles). It is enabled for ordered
categorical terms and numeric-axis terms (`Spline`, `Polynomial`, `Piecewise`). On any other term
it is disabled, and its popover says why: "Breaks need an ordered or numeric axis." A linear
`Numeric` term is drawn as a single point, so it has no axis to place a break on (§8). On an
ordered term with collapsed groups, Breaks mode draws the expanded display, because a break names
one band; the Groups control is disabled there.

**Gestures.**

- Click inside the plot to add a break. Drag a break line to move it. Click the × on its label to
  remove it. Breaks cannot cross: a drag is clamped between its neighbours.
- Keyboard: each break is focusable; arrow keys move it one band (ordered) or one grid step
  (numeric); Delete removes it.
- **Snapping:** on an ordered term, to band positions strictly inside the axis (never the first or
  last band). On a numeric term, to a grid of three significant figures of the fitted span (of
  the span, not the value, so an offset axis such as years keeps every position), strictly inside
  the fitted range; the arrow keys move one grid step.
- Entering the mode on a term that is already `Piecewise` (numeric or ordered-hosted) loads its
  current breaks and degrees.

**Draft state.** The breaks, degrees and form being edited form a per-term draft in the browser
store's `view` (like zoom, not like gesture internals). A draft never changes predictions and is
never sent until Transform and refit.

**Action bar in Breaks mode.** A form switch (Piecewise · Spline · Polynomial), a degree stepper
for Polynomial (1–5), an inline hint that doubles as validation text, "Clear breaks", and the
Transform and refit icon (primary). Accessible name and popover title "Transform and refit";
popover body: "Replace this term with the form and breaks shown, then refit the model. Restore
undoes it."

**Degree chips.** For Piecewise on an ordered term, one chip per segment sits above the plot.
Clicking it cycles flat → linear → quadratic → cubic, capped at `min(3, points in the segment −
1)`. The browser mirrors the library's rules for instant feedback — at least one break, not every
segment flat, no two flat segments in a row — and disables the icon with the rule shown inline.
Python remains authoritative.

**What each form becomes.** Other parameters carry over from the source spec wherever the target
has them (order, specials, grouping, base, extrapolation; spline kind, degree, penalty, `select`,
boundary and lambda policy; a numeric `Piecewise`'s `lower` and `upper` pins, and its base knot
while a knot is still there).

| Source term | Form | Replacement spec |
|---|---|---|
| `OrderedCategorical` (any basis) | Piecewise | `OrderedCategorical(..., basis=Piecewise(breaks=[band names], degrees=[...]))` |
| | Spline | `OrderedCategorical(..., basis=Spline(kind=<current kind, else "cr">, knots=[band names]))` |
| | Polynomial | `OrderedCategorical(..., basis=Polynomial(degree=d))` |
| Numeric axis (`Spline`, `Polynomial`, `Piecewise`) | Piecewise | `Piecewise(breaks=[values], extrapolation=<source's, else "clip">)` — straight segments only |
| | Spline | `Spline(kind=<current kind, else "cr">, knots=[values], ...)` |
| | Polynomial | `Polynomial(degree=d)` |

Per-segment degrees stay off the numeric axis because the library refuses them there today
(`features/piecewise.py`): the exported workbook is exact under linear interpolation only at
degree 1. Lifting that is a library and export change (§8).

**Python.** A new `editor/transform.py`, a sibling of `collapse.py`, provides
`transformed_feature_spec(model, term, *, form, breaks, degrees=None, degree=None)`.
`EditorSession.replace_with_transformed_term(term, ...)` follows the same refit path as §4.2 and
stamps `format: "superglm.editor.term_transform.v1"` with the form, breaks (as labels or values),
degrees and the disclosure message (§4.7).

**Route.** `POST /transform_term` takes
`{term, form: "piecewise"|"spline"|"polynomial", breaks: [str|number], degrees?: [int],
degree?: int, level_display}` and returns the existing structural envelope with
`timing.operation = "transform_term"`.

### 4.4 Restore previous structure: one stack

- **One stack.** Collapse, ungroup, transform and set-reference each push exactly one entry (the
  previous in-force model and a short label). Restore pops exactly one. Only Revert (§4.5) and
  distribution re-profiling (unchanged) clear the stack.
- **Manual edits.** Every structural step, Restore included, rebuilds the editable terms from the
  new model, so it clears manual edit history, as collapse does today. The existing confirmation
  appears only when there is history to lose. Set references and structure first, then hand-edit.
- **Ungroup becomes an ordinary step.** Today, when an ungroup removes a term's last collapse, the
  session either pops the pre-collapse model or refits and clears the whole stack. With a shared
  stack, clearing would silently discard transform and reference steps. So ungroup always pushes.
  Reusing an identical earlier fitted model instead of refitting stays as an optimisation. Tests
  that pin the old clearing behaviour are updated, and the change is stated in the PR.
- **Placement.** The Restore icon moves from the selection palette to the action bar and is
  visible in every mode whenever the stack is non-empty. Accessible name "Restore previous
  structure"; its popover names the step it undoes, e.g. "Undo: transform MileageBand to
  piecewise". The palette's contextual Restore is removed.
- **State.** The snapshot gains `structure_history: {depth, last: {operation, term, label} |
  null}`, which drives the icon. It replaces `last_collapse` and `can_uncollapse_levels`, whose
  only consumer was the old palette Restore. The widget's parallel info stack goes with them:
  each step's information lives on its `StructuralStep` and on the refit model's stamp.
- **Names.** The public session methods keep their names (`uncollapse_levels`,
  `can_uncollapse_levels`) with docstrings widened to "the previous structural step". The route
  becomes `POST /restore_structure`; the editor ships its own frontend, so no route alias is kept.

### 4.5 Revert to original model

- **Placement:** an icon in the app bar next to Undo and Redo. Accessible name "Revert to original
  model". It is enabled when anything differs from the opened model: a manual edit on any term,
  a non-empty structural stack, or an in-force model replaced without either (a distribution
  re-profile clears both histories).
- **Action:** the in-force model becomes `session.reference_model` (already fitted, so there is no
  refit). This clears every term's manual history and redo stack, the structural stack, and any
  stored fixed-offset refit.
- **Confirmation:** the existing structural-confirm dialog, extended with a structural-step count,
  e.g. "Revert to the original model? This clears 7 manual edits and 2 structural steps, and
  can't be undone."
- **Route:** `POST /revert_to_original` returns the structural envelope with
  `timing.operation = "revert_to_original"`.

### 4.6 Original-model line style

- `styles.css` `.original`: solid, `stroke: #8c959f; stroke-opacity: 0.5; stroke-width: 3`,
  replacing `stroke-width: 1.7; stroke-dasharray: 7 5`. The legend swatch uses the same class and
  follows automatically. The line stays under the edited curve in draw order.
- `plotting/editor_style.py` `"original"`: `dict(color=(140, 149, 159), alpha=0.5, width=3.0)`,
  keeping the Python plots in step with the editor.
- The values are a starting point for Max to adjust on the real chart.

### 4.7 The summary note

- A transform whose breaks came from the editor stamps the fitted model with `_editor_structure`,
  a dict keyed by term name holding the form, breaks, degrees and message. `model/report_ops.py`
  renders it into the model information exactly as it renders `_editor_offset`. The editor
  summary, the Python `summary()` (`inference/summary.py`) and the workbook's Model Summary
  (`export/summary.py`) therefore all show the same line:
  "Breaks for ⟨term⟩ were chosen in the editor from this data. Tests are conditional on them;
  judge the breaks on validation deviance."
- Polynomial transforms (no breaks) and set-reference steps carry no note.
- The stamp must survive `to_model()` and the `.joblib` export; a test pins both.

## 5. Contracts

- **New routes:** `POST /set_reference` `{term, level, level_display}`, `POST /transform_term`
  (§4.3), `POST /revert_to_original` `{level_display}`, and `POST /restore_structure`
  `{level_display}`, which replaces `/uncollapse_levels`. All are token-guarded, parse untrusted
  JSON explicitly, put the mutation and locking in `EditorWidget`, and return the structural
  envelope from one post-refit lock scope.
- **Snapshot:** `structure_history` (§4.4); the reference per level term (kind and level) for the
  chip in §4.2; `in_force_is_original` for Revert (§4.5).
- **`api/contracts.js`:** `EditorMode` gains `'breaks'`; add the transform draft, the new request
  payloads and `structure_history`.
- **Timing operation names:** `set_reference`, `transform_term`, `revert_to_original`,
  `restore_structure`.

## 6. Errors and refusals

- Browser-side checks are advisory; Python validation is authoritative. The editor checks the
  rules it can state itself (break positions, order and range; segment-degree counts and ranges;
  polynomial degree; constraints) with its own intentional messages. `editor/errors.py` forbids
  passing backend exception text to the browser, so any other `ValueError` from building or
  fitting a transformed spec becomes one fixed message ("SuperGLM could not fit this shape…"),
  chained to the original for Python callers.
- A failed structural refit leaves the stack and the in-force model unchanged (the existing
  `try/except` pattern in the `replace_with_*` methods). The browser uses its existing recovery:
  reconcile from `/state`, then show an alert with Retry and Dismiss.

## 7. Testing

Follows the repository's numerical test policy: invariants rather than roundoff, tolerances
derived from solver tolerance and conditioning, and a mutation check for every regression test.

- **Transform oracle.** For each form, on an ordered term and a numeric term, the
  editor-transformed model's predictions equal those of a model fitted directly with the same
  spec, within a bound derived from the fit's convergence tolerance. This independently checks
  the whole replacement path.
- **Carry-over.** The transform keeps the base, order, specials, grouping and extrapolation.
- **Set reference.** Predictions are unchanged under default settings, on a `Categorical` term and
  an ordered-spline term (tolerance-derived bound). The new reference's relativity is exactly 1.
  Under an explicit selection penalty the fit changes, which demonstrates that the hover text is
  true. A pinned reference survives a later collapse. Specials and interaction parents are
  refused.
- **Mutation check for set reference.** Building the replacement as a mutated fitted copy makes
  the test fail (the sticky base keeps the old reference).
- **One stack.** Collapse → transform → set reference → ungroup, then Restore four times, returns
  exactly to the opened model's predictions, with each intermediate state exact.
- **Revert.** Returns the reference model's predictions exactly and clears every history.
- **Refresh.** A Node test: a snapshot at an equal revision with a changed selection or curve is
  committed, and evidence is scheduled. A Python test: a fixed-offset refit becomes unavailable
  after a notebook-side edit.
- **Summary note.** Present in the editor summary, the Python summary and the workbook for
  editor-chosen breaks; absent for polynomial and set-reference.
- **Style.** `tests/test_lss_editor_style.py::test_chart_grammar_matches_the_editor_css` keeps
  pinning the CSS-to-Python mirror, extended to read the new stroke opacity.
- **Browser (Playwright).** One case per feature: Breaks gestures plus transform; set-reference
  icon visibility and effect; refresh after a Python-side edit; the revert confirmation.
- No wall-clock assertions anywhere in the suite.

## 8. Out of scope (follow-ups)

- Per-segment degrees on the numeric axis. The library refuses them today; lifting that needs an
  export design, probably through the ppform path.
- A live preview of the transformed curve before applying.
- Automatic refresh (polling or revalidate-on-focus).
- Carrying manual edits across structural refits.
- Pinning a `Piecewise` base knot from the editor.
- The same disclosure note for levels grouped by eye through collapse. The same caveat applies;
  this is a consistency follow-up.
- Converting an unordered categorical into an ordered one.
- Breaks on a linear `Numeric` term. It is drawn as a single point, so it first needs an axis over
  its fitted range.

## 9. Decisions

**Max, 2026-09-23:**

- Refresh means re-sync the view with Python; nothing refits.
- A transform applies immediately; Restore undoes it (no preview-then-accept).
- Standard errors and tests are reported as usual, with one line of disclosure (§3.3).
- Set reference is its own icon; no popup from a selection.
- Nothing in the editor may intrude.
- The original-model line becomes a solid, wider grey line with alpha.
- An undo-all-changes control is needed (it became Revert to original model).

**Delegated to Claude (override any):**

- Numeric-axis Piecewise stays degree 1.
- No preview before applying.
- Restore moves to the action bar, with one shared stack; ungroup always pushes.
- Revert is final, behind the existing confirmation.
- Per-segment degree cap of 3; polynomial degree 1–5.
- Numeric breaks round to three significant figures of the fitted span.
- The reference chip goes in the context bar.
- Style values: opacity 0.5, width 3.

## 10. Risks for the plan

- **Ungroup semantics change.** Existing tests pin the clearing behaviour; the PR must say so.
- **Fresh spec construction for every source kind.** Fitted state is sticky: a `Categorical`'s
  base, a `Piecewise` base knot.
- **Ordered-hosted Piecewise with an existing grouping.** The library refuses a collapse group
  that spans a break; that message has to reach the analyst intact.
- **The editor's 24-handle cap** for Piecewise terms with many knots (existing and documented).
- **Distribution re-profiling clears the stack** (existing); confirm this is still wanted now that
  the stack holds more kinds of step.
- **Confirmation copy** for Revert and Set reference.

## Appendix A: the post-selection simulation

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(1)
n, reps = 2000, 4000
candidates = np.linspace(0.05, 0.95, 19)

def kink_t(x, y, c):
    X = np.column_stack([np.ones_like(x), x, np.maximum(x - c, 0.0)])
    beta, rss, *_ = np.linalg.lstsq(X, y, rcond=None)
    cov = rss[0] / (len(y) - 3) * np.linalg.inv(X.T @ X)
    return beta[2] / np.sqrt(cov[2, 2])

p_of = lambda t: 2 * stats.norm.sf(abs(t))
fixed = eyeball = split = 0
for _ in range(reps):
    x = rng.uniform(0, 1, n)
    y = 1.0 + 0.8 * x + rng.normal(0, 1, n)          # no kink anywhere
    fixed += p_of(kink_t(x, y, 0.5)) < 0.05           # break stated in advance
    ts = [kink_t(x, y, c) for c in candidates]
    eyeball += p_of(max(ts, key=abs)) < 0.05          # break at the visible bend
    half = n // 2
    c_star = candidates[np.argmax([abs(kink_t(x[:half], y[:half], c)) for c in candidates])]
    split += p_of(kink_t(x[half:], y[half:], c_star)) < 0.05  # chosen on A, tested on B
print(fixed / reps, eyeball / reps, split / reps)     # 0.050 0.233 0.054
```
