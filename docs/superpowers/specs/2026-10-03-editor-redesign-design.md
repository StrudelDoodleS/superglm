# Editor redesign — design

Date: 2026-10-03. Branch `claude/editor-improvements-redesign-96ab8d`, cut from
`origin/master` at 155832e8 (v0.36.1). Mock-ups (approved by Max 2026-10-03,
"This looks great!", dark theme "looking fire"):
https://claude.ai/artifact/Wi5CUShrPvbCRfEJ7ifZxY — boards 1–7, drawn from a real
Poisson fit on 60,000 freMTPL2 rows and a real 5-fold `cross_validate()`.

This document is for the builders and reviewers. Max reads the one-screen summary
in chat, not this file.

## 1. What users will notice

1. Collapse, ungroup, set reference and the four shapes no longer refit. They
   wait, drawn on the chart as "waiting", and one **Refit** button in the top bar
   applies every waiting change in a single fit. Hand edits on terms whose
   structure did not change survive the Refit.
2. Collapsing never moves the reference level (a setting, on by default).
3. Click a point, Shift-click another: everything between them is selected.
   Ctrl/Cmd-click adds or removes single points on any term. Shift-drag still pans.
4. An ordered categorical with a spline basis gets Handles, Contrib and Build,
   and its spline is drawn between the levels.
5. The inspector has a search box, All / Edited / Waiting filters, follows the
   chart, and has a tidier header. Advanced becomes **Settings**.
6. History is git-style: every step has a short id, an automatic message and an
   optional note; notes travel with the exported Python model.
7. A **Cross-validation** tab, fed by `cv=` a `CrossValidationResult`: fold
   performance, a fold table, relativities by fold ranked by instability, and
   two jobs — **Run CV on current model** and **Final fit on all rows**.
8. A **rating-table preview** of the current term, built by the same code as the
   Excel export.
9. A warm dark theme in the gruvbox family replaces the cool blue-black one, and
   a pill switch replaces the cycling theme icon.
10. Library fix: `plot_terms_by_fold` and the CV curve-similarity diagnostics put
    unordered categorical levels in the model's order, not row order.

## 2. Decisions

| # | Decision | Who |
|---|---|---|
| D1 | Every structural change (collapse, ungroup, set reference, Flat/Line/Quadratic/Cubic) waits for Refit; Settings can restore refit-on-every-change. | Max asked for collapse; extension to all structural steps delegated, one rule |
| D2 | Refit carries hand edits over on terms whose structure did not change (the same curves, re-applied after the fit). Edits on a restructured term are dropped; Undo restores them. Today every hand edit is dropped. | Delegated |
| D3 | Keep the reference when collapsing/ungrouping: on by default, a setting. | Max (toggle in Settings) |
| D4 | A Ctrl-clicked, non-contiguous selection: a shape covers the span first→last selected point and is fitted to the data. | Delegated; shown on board 2 |
| D5 | Run CV and Final fit refit the current structure on each fold / on all rows, then put the hand edits back exactly as set (D2's rule), before scoring or exporting. | Delegated; "held as you set them" |
| D6 | "All rows" for Final fit = train ∪ validation. The test split stays held out. The final model is shown on the Final fit tab and offered by Export; it does not replace the model being edited. | Delegated |
| D7 | Run CV is disabled while changes are waiting ("Refit first"). | Delegated |
| D8 | Settings live in the browser (one localStorage key); backend-affecting settings travel with each request. | Delegated |
| D9 | Dark theme: gruvbox-family warm dark (MIT/X11, morhetz/gruvbox README). Solarized dark rejected: its blue-teal base fights the sand light theme. | Max approved the boards |
| D10 | Theme switch: the **comic pill (variant B)**, animated (board 7b); "Follow the browser" lives in Settings, on until the pill is flipped. | Max picked B and asked for the animation, 2026-10-03 |
| D11 | History notes are saved with the exported Python model. The Excel workbook is unchanged in this round: its block layout is a cross-repo contract with the Airflow builder repo. | Delegated |
| D12 | Superseded: `cv_report=` keeps working (shown in Validation as today); the new tab uses `cv=`. Never break userspace. | Delegated |

## 3. Research gate

**Characterisation.** This is interactive editing of a fitted penalised GLM/GAM:
direct manipulation of shape functions, structural re-specification of terms
followed by a penalised refit, and held-out validation of the edited model.

**Sweep and what we adopt.**

- GAM Changer — Wang, Kale, Nori, Stella, Nunnally, Chau, Vorvoreanu, Wortman
  Vaughan and Caruana, *Interpretability, Then What? Editing Machine Learning
  Models to Reflect Human Knowledge and Values*, KDD 2022, arXiv:2206.15465
  (Max's stated inspiration).
  - Marquee selection in a select mode, a Context Toolbar on selection (we have both).
  - Monotone edit by count-weighted isotonic regression (already adopted, #419).
  - Edits are **committed or discarded** from the status bar (check / cross
    icons). This is the closest precedent for D1: GAM Changer stages one edit;
    we stage structural changes because ours need a penalised refit, which an
    EBM edit does not. That difference is the reason we are outside its design,
    not an omission.
  - A git-style History Panel: each commit has a timestamp, an identifier and an
    auto-generated message the user can edit; participants used it for model
    audit documentation (§5.2.3), and the edited model is saved with its history
    (§A.4). Adopted as item 6.
  - Metric panel scopes (global / selected / slice); our selected-exposure
    readout covers the "selected" scope. Slice scope is a follow-up.
  - GAM Changer has no fold-level validation; the Cross-validation tab goes
    beyond it, using superglm's existing `curve_similarity` diagnostics.
- Multi-select convention — WAI-ARIA Authoring Practices, Listbox pattern:
  Shift+Space "selects contiguous items from the most recently selected item to
  the focused item"; Space toggles one option. Mouse equivalents (Shift-click
  range from the anchor, Ctrl/Cmd-click toggle) follow the same model. Adopted
  for item 3; keyboard range selection is a follow-up.
- Gruvbox palette, MIT/X11 per the morhetz/gruvbox README. Values are used and
  cited, no code copied.

**New territory, stated:** staging several structural re-specifications against
one penalised refit, with an undo history spanning staged and applied steps, has
no published precedent we found (GAM Changer stages single weight edits only).

## 4. Workstreams

Order is dependency order. Each lands as its own commits on this branch.

### H. Level order in fold plots (library)

- `plotting/comparison.py::_shared_level_domain` uses row-appearance order for an
  unordered `Categorical` (`drop_duplicates()` on the column). Use the first
  model's fitted level order (`spec._levels`, as strings), then append any
  observed label it lacks. `OrderedCategorical` keeps `_ordered_levels`.
- Measured on freMTPL2: Area came out A, B, E, D, C, F; VehBrand B1, B10, B11,
  B12, B2, B3, B4, B5, B13, B6, B14.
- Check `build_cv_curve_similarity` takes its domain from the same function.
- Test: rows in order C, A, B → domain A, B, C. Mutation check: fails on master.

### B. Keep the reference (backend)

- `_collapsed_base` (`collapse.py:506`) receives the declared `spec.base`. With a
  symbolic policy (`most_exposed`, `first`) the fresh spec re-resolves at fit
  time and the reference can move. With keep-reference on, pass the in-force
  fitted spec's resolved `_base_level` (native type, via the mapping at
  `collapse.py:196`); the existing concrete-base branches then give "the group
  containing it".
- `_valid_base_after_ungroup` (`collapse.py:546`) falls back to the first level
  pulled out when the base group is renamed by a partial ungroup. Map to the
  group holding most of the old group's members instead (the collapse rule).
- With the setting off, today's behaviour.
- `_reference_payload` then reports a pinned policy; the chip reads "reference
  B2 · kept".
- Tests: collapse under `most_exposed` that measurably moved the reference on
  master (C+D example in the structural-tools programme) keeps it; collapsing the
  reference itself gives its group; partial ungroup of the reference group keeps
  the majority group. Each fails on master.

### A. Staged structural changes and Refit (core)

**Session model.**
- `SessionState` gains `pending: list[PendingStep]`. A `PendingStep` stores the
  operation, term, label, its parameters **as labels** (selected level labels,
  group label, reference level, lo/hi/degree/join), the resulting draft spec, and
  its position relative to manual edits (the length of `history` when staged).
- A term's draft spec is the last pending step's spec for that term, else the
  in-force fitted spec.
- Undo pops whatever happened last, in time: a manual edit, a pending step, or
  an applied step. Popping a pending step restores the previous draft and does
  **not** advance `model_revision`, because model and terms did not change.
- `_capture_state` shares the `terms` dict; a pending step must not be captured
  in a way that later in-place edits can mutate (copy what it keeps).

**Builders become draft-aware.** Each takes the term's draft spec, the in-force
fitted spec and the display term. These break on an unfitted draft today:
- Set reference on a plain categorical reads `spec._levels`, which is `[]` before
  a fit. Take the level universe from the in-force fitted spec and the draft
  grouping.
- A shaped range on a numeric spline reads `fitted_boundary`/`fitted_base_knots`,
  which are None before a fit. Use the draft's explicit `knots=`/`boundary=` when
  present, else the in-force fitted spec's values.
- Collapse and ungroup rebuild `Categorical(base=..., grouping=...)` and drop
  `levels=` and `unseen=`. Carry both. This is a known follow-up from #419;
  composition makes it required.
- Ungroup to no grouping leaves a string base; map it back to the native type.
- The ungroup shortcut that restores a previous fitted model applies only when
  nothing is pending.

**Refit.**
- `clone_with_replaced_feature` takes a mapping, so one clone replaces every
  staged term.
- One fit with `method="auto"`, as today.
- One applied `StructuralStep`, "Refit · N changes". Its saved state holds the
  pending list, so Undo of the Refit brings the changes back **as waiting**.
- Carry-over (D2): for every term not restructured, re-apply its edited curve to
  the refitted model (same grid and labels: same rows, same `n_points`). It is
  recorded as one history entry, "Hand edits carried over: DrivAge, VehAge",
  whose Undo returns those terms to the refitted curves.
- Refusals: build-time refusals (range edges, level labels) happen when the step
  is staged, with today's fixed messages. Fit-time refusals name the batch:
  "The refit was refused. Undo the last waiting change and try again." This uses
  a fixed sentence, because `editor/errors.py` forbids backend text.

**Display while waiting.**
- The payload gains `pending` (one entry per step: term, operation, label, params)
  plus, per term, `pending_groups` and `pending_ranges`.
- Chart:
  - grouped levels in their group colour, dashed bars, and a dashed bracket
    "B10 + B11 · waiting" under the axis;
  - a dashed range box with a "Line · waiting for refit" chip;
  - the curve itself stays the last refit's.
- Other surfaces:
  - the feature list shows an amber dot on the term;
  - the top-bar Refit button shows the count;
  - the inspector shows a "N waiting" chip and the Waiting filter;
  - History shows a "Waiting for refit" section above "Applied";
  - the status line reads "N changes waiting for refit · the curve and metrics
    are from the last refit".
- Metrics are not marked stale, because they are true for the last fit.
- Export while waiting exports the last refit, and the export dialog says
  "N waiting changes are not included".

**Frontend.**
- The structural icons call a new `/stage` (or `stage=true` on the existing
  routes).
- The Refit button calls `/refit_pending`.
- With "Refit after every structural change" on, the frontend stages and then
  refits.
- `R` refits (check it is unused).
- The selection-menu group label reads "Structure"; it reverts to "Refit" while
  the setting is on.

**Git-style history (item 6).**
- Every timeline entry, applied or waiting, gets:
  - a stable short id (7 hex digits from a per-session counter plus a random
    salt);
  - an automatic message ("Collapse B10 + B11", "Line 18 – 26");
  - an optional note, edited inline from a pencil.
- Notes live on the step and survive undo/redo.
- `export` (joblib) attaches the timeline as `model._editor_history` (a list of
  dicts: id, time, operation, term, message, note, applied/waiting).
- Not added to the workbook (D11).

### F. Settings tab

- Replaces Advanced (gear icon, `data-inspector-tab="settings"`). Contents:
  - Refit after every structural change (off);
  - Keep the reference level when collapsing (on);
  - Follow the browser's light or dark setting (on);
  - Groups shown as Expanded or Collapsed (the default when a term opens);
  - Build animation (moved, and now persisted);
  - Request timings (a switch that shows the existing timing readout).
- One module `views/settings.js` with one localStorage key
  `superglm.editor.settings` (a JSON object, try/catch on read and write; renders
  correctly when storage is blocked).
- Existing keys (theme, featureList, shapeJoin) are left alone.

### C. Selection gestures

- Pointerdown on a point no longer pans on Shift. Panning starts only once the
  pointer moves past the existing 3-unit threshold. Releasing under the
  threshold with Shift on a point selects the anchor..point span.
- The anchor is client-side state, set by a plain click or a Ctrl-click.
- The span is every display index between the two x positions, regardless of y.
  It goes through the existing `/select`.
- Ctrl/Cmd-click toggles one point on any term (drop the `term.levels` gate at
  `interactions.js:346`).
- A click on the curve line between points snaps to the nearest grid index by x
  when within 12 px of the curve; a click on empty space changes nothing, as
  today.
- Shapes use the span from the first to the last selected point (D4).
- Help text, `help_content.js` and the pinned strings in `tests/test_editor.py`
  are updated.
- The Build-animation click guard (`main.js:1420`) stays.

### E. Inspector

- **Search:**
  - a box at the top of Summary filters the compact rows by term name and level
    label (case-insensitive substring), highlighting matches with `<mark>`;
  - it shows "N terms · M rows", or "No terms match.", and Esc clears it;
  - it is reapplied after every `updateSummaryMarkup` (the frame is rebuilt each
    render);
  - rows gain `data-term`;
  - the raw-summary iframe is not searched.
- **Filters:** All / Edited / Waiting.
- **Follows the chart:**
  - the current term's section is open and scrolled into view; the others fold
    to a header line (name, kind, EDF, p-value chip, waiting chip);
  - the user can open others until the term changes;
  - a search opens every matching section.
- **Header:** family / link / method as chips, four metric tiles (deviance, AIC,
  BIC, total EDF), and Refit offsets beside them. The Expanded/Grouped radio
  stays.

### D. Ordered categorical with a spline basis

- Payload flag `basis: "spline"` for an `OrderedCategorical` whose basis is a
  spline and which has no shaped ranges. Once a band is shaped, the tools are
  disabled with a hover reason ("goes away", Max).
- Grouped ordered terms are out of scope for this round: disabled with a reason.
- **Handles** come from the **fitted** spline coefficients (`_split_beta`), not
  least squares on the level points. That fit is underdetermined, because
  `n_knots` is clamped to L−1 and the basis often has more columns than levels.
  - Handle x maps from spline position (`_level_to_value`: linspace(0,1), or the
    user's `values=`) to display x, piecewise linear between level indices.
  - Moving a handle sets level effects to B(level positions)·c.
  - Special levels have no position and are untouched.
- **Curve:** draw the spline between levels on a fine grid of positions (24 per
  gap), mapped to display x; level dots sit on it; specials are drawn as
  separate dots.
- **Contrib/Build:** basis contributions on the same fine grid. Row lengths match
  the fine grid, not `term.x` (fix the length check at `chart.js:684`).
- **Gates:**
  - Handles: `controls.py` `CONTROL_HANDLE_TERM_TYPES` and `_require_control_term`
    accept the flagged term.
  - Contrib/Build: `main.js:1281` and `1334`.

### I. Dark theme and theme switch

- Rewrite `styles/dark.css` with the board palette:

  | Token | Value |
  |---|---|
  | Grounds | #1d2021 / #282828 / #3c3836 |
  | Text / muted | #ebdbb2 / #a89984 |
  | Current-edit blue | #83a8e8 |
  | Exposure yellow | #d79921, higher alpha |
  | Selection orange | #fe8019 |
  | Groups | #fabd2f, #d3869b, … |
  | Folds | #83a598, #b8bb26, #d3869b, #fe8019, #8ec07c |
  | Significance tints | as on the board |

- Validate text contrast ≥ 4.5:1 and series separability; keep the dataviz rule
  that colours that must be told apart also differ in lightness.
- Restate in `plotting/editor_style.py` only if it reads dark tokens. Today it
  reads the light palette; its tests stay.
- Theme switch: the comic pill (`role="switch"`, board 7b) replaces the cycling
  icon. It has:
  - an ink outline, a hard offset shadow, a halftone track (a radial-gradient
    dot pattern) and a Bangers DAY/NIGHT label;
  - the font loaded with the existing Google Fonts link (Bangers is already the
    docs display face).
- Animation, played only on the user's click. It is a CSS keyframe animation,
  restarted on each click by alternating two identically defined keyframe names
  (board 7b; Max asked for "css animation" after a transition-only version read
  as instant):
  - the knob travels in 680 ms: squash (scale 1.32 × 0.78), overshoot past the
    end, settle;
  - the outgoing icon spins out (420 ms) and the incoming one spins in with a
    small overshoot (620 ms, 160 ms delay);
  - the label pops in (560 ms, 260 ms delay, scale 0.2 → 1.35 → 1 with a tilt);
  - no starburst: the board 7b version had one and Max rejected it;
  - the track, ink, shadow, bar and page colours cross-fade, and the halftone
    dots drift (560–600 ms).
- `prefers-reduced-motion` turns all of it off. tokens.css already zeroes
  animations and transitions under it.
- The pre-paint script keeps working. The "Follow the browser" setting restores
  Auto.
- The docs site's dark stays as is (follow-up).

### K. Rating-table preview

- A Chart / Table switch in the context bar. Table shows the current term's
  block from `export.rating_tables.build_rating_table_payload` on the same
  dataset and model the Excel export uses (`widget._export_bytes` path), so the
  preview is exactly what the workbook writes.
- Continuous terms show their banded/ppform rows as the export does.
- Interaction blocks are out of scope (main effects only).
- Refused terms (`_preflight_rating_table_terms`) show the export's own refusal
  as a fixed message.

### G. Cross-validation tab

**API.**
- `EditorSession.from_model(..., cv=CrossValidationResult, cv_data=(X, y[, w[, offset]]))`.
- `edit()` gains the same `cv=` and data arguments.
- Without `cv_data`, the train data is used when its row count matches the folds.
- `cross_validate` records `n_rows` and a data fingerprint (SHA-256 over the
  y and weight bytes) on the result. The editor compares them. On a mismatch,
  Run CV is disabled with the reason. An older result without a fingerprint gets
  a row-count check and a one-line note.

**Tab.**
- A fourth app view, `cv`, reusing `#reportPanel`.
- `report_payload` and `renderReport` get a `cv` branch; `AppView`, `showView`
  and the title fallback are extended.

**Content (board 6):**
- Header: folds, splitter, rows and source; a waiting chip; the two buttons.
- Performance cards: mean ± sd, pooled where defined, one dot per fold. After a
  Run CV, "as supplied" and "current" are shown side by side.
- Fold table: train and test rows, metrics, EDF, fit time, converged.
- Relativities by fold:
  - a searchable term list ranked by spread (the mean over folds of
    `rmse_to_mean` on the response scale) with min `correlation_to_mean`;
  - per term, a chart:
    - categorical / ordered: fold dots, a range whisker and an all-rows-fit bar,
      in **model level order**;
    - continuous: fold lines, a min–max envelope and the all-rows fit;
    - both: exposure behind.
  - Every curve is re-centred on its exposure-weighted mean log.
  - Fold curves come from the estimators (`return_estimators=True` or a Run CV).
    Without them the section says how to get them.

**Jobs.**
- Run CV and Final fit reuse the profile-job pattern (thread, job dict, status
  polling) with two fixes:
  - a cancel endpoint and flag, checked between folds;
  - the widget lock is not held across the fit: capture the model, work
    unlocked, publish if still current (`_current_model_for_evidence`).
- Finished jobs are evicted (keep the last of each kind).
- Run CV:
  - stored folds come through a small splitter that yields
    `CrossValidationResult.fold_indices`;
  - the model is the in-force structure (last refit);
  - hand edits are re-applied per fold (D5);
  - `fit_mode` is the in-force method;
  - the scorers are the supplied result's built-in names, else deviance, Gini
    and NLL.
- Final fit: train ∪ validation; hand edits re-applied (D5, D6); shown on Final
  fit and offered in Export as "Final fit model".

## 5. Out of scope, recorded as follow-ups

- **SuperLSS editor (sub-project 2):** a predictor-by-predictor editor, LSS
  visualisations, and whether a grouping is linked across predictors. It needs
  its own design round and a GAMLSS literature sweep. Design A–G so a predictor
  selector can be added: pending steps and history entries carry an optional
  `predictor` key, unused for now.
- **UMAP explorer (sub-project 3):** which question it answers (mispriced
  segments, or level similarity for collapse) is open. `umap-learn` is BSD-3.
- Editable history in the workbook (needs the builder repo checked first).
- GAM Changer's slice-scope metrics.
- A "neutralise" tool (GAM Changer's delete, its participants' favourite).
- Keyboard range selection (Shift+Arrow).
- Natural sort of level labels (B1, B2, …, B10).
- Hold hand edits as offsets during Refit (the D2 alternative).
- Docs-site dark theme.
- The planned `main.js` split.

## 6. Testing

- Python: one focused test per behaviour above, each demonstrated against
  master. That covers composition of staged builders, Undo/Redo over pending and
  applied steps, carry-over, reference pinning, the CV job (folds and hand edits
  re-applied), the Final fit rows, history notes in the export, and the level
  order. Assert counts, not wall time: N staged changes run exactly one fit
  (count fits through a patched `fit_refit_model`).
- Node (`npm run check:frontend`): gestures, settings persistence with blocked
  storage, summary search and filters reapplied after a re-render, pending
  rendering, and the theme switch.
- Browser (`--run-browser`): stage two collapses then Refit; Shift-click range
  on a spline; Ctrl-click on a spline; the CV tab renders from a supplied
  result; dark theme toggle.
- Full suite once at the end, xdist; ruff; `uv lock --check`.

## 7. Performance evidence

No solver, REML or kernel code changes. The batch Refit calls the same fit, once
for N staged changes instead of N times; report the fit count. The CV job runs
`cross_validate` unchanged. One end-to-end timing pair (master vs branch: open,
three collapses, refit) at integration only.

## 8. Delivery

- One branch, one PR when Max says so. Release impact `release:minor`: new
  editor features and a new `cv=` argument; no breaking change, since
  `cv_report=` still works.
- Roadmap: `notes/ROADMAP.md` has no editor entries. Propose one line under
  Current position: "Editor redesign (2026-10, explicitly scoped user work);
  SuperLSS editor next."
