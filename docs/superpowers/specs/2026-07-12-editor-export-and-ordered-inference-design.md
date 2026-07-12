# Editor export and ordered-categorical inference

## Objective

Replace the editor's model-only Save action with one Export workflow that can produce a
validated Python model artifact or the existing deployment-oriented Excel rating workbook.
At the same time, integrate the completed ordered-categorical inference semantics so the
Python summary, Excel workbook, and editor show the same statistical interpretation.

## Existing behavior

- The editor header opens a Save dialog for a joblib model. It can download the artifact or
  write it to a kernel-visible path.
- The joblib path materializes the authoritative edited model but does not load the produced
  artifact back or validate prediction equivalence.
- `export_rating_tables()` already produces rating tables, discretization impact, and a model
  summary. The summary sheet is an ASCII rendering split into successive cells in column A,
  rather than typed spreadsheet data.
- The editor redesign branch does not contain the completed ordered-spline inference work.
  Consequently, its summary still exposes reference-dependent p-values for each displayed
  ordered level.

## User experience

The header action is renamed **Export** and uses an export/download icon and matching delayed
popover help. It opens a single dialog with two format choices:

1. **Python model (`.joblib`)**
2. **Excel rating workbook (`.xlsx`)**

The dialog exposes a format-appropriate filename. Browser download is the primary action.
The existing kernel-path destination remains available as a secondary action so notebook and
remote-kernel workflows do not regress. Export work does not mutate editor state, advance the
model revision, or redraw the chart, summary, or report panels.

The dialog reports preparation and validation progress, the validation scope, success, and
actionable errors. Repeated clicks while an export is running are suppressed.

## Authoritative model and revision handling

Both formats start with the editor's existing materialized-model coordinator. The export
captures a model revision, obtains that exact authoritative in-force model, and checks the
revision again before returning or writing the artifact. A superseded request produces an
explicit error rather than exporting a mixture of revisions.

The materialized model is reused within one request. Excel generation and Python validation do
not independently re-apply the same edits.

## Python model validation

The Python artifact is serialized to bytes and then loaded from those same bytes before it can
be downloaded or written. Validation has two layers, with prediction validation conditional on
having evaluation rows:

1. **Artifact contract validation** always runs. The loaded value must be a fitted SuperGLM
   model with compatible feature metadata, result dimensions, finite fitted parameters, and a
   callable prediction surface.
2. **Prediction validation** runs whenever an editor evaluation dataset is available. It uses a
   deterministic bounded sample of at most 512 rows and compares predictions from the loaded
   artifact with predictions from the authoritative materialized model using tight floating-
   point tolerances. The dataset is used only for validation and is not copied into the artifact
   beyond the model's existing retention policy.

An artifact with no available evaluation rows may still be exported only after contract and
round-trip validation; the UI states that prediction validation was unavailable. Any failed
check blocks the artifact and returns a concise validation error. The implementation must not
log or return raw training rows.

## Excel rating workbook

The editor invokes the existing rating-table payload builder and Excel renderer rather than
creating an editor-specific workbook. Rating tables use explicit training data when supplied,
otherwise retained fit data. Validation or test data are never silently substituted because
their exposure distribution is not the deployment basis. If neither training source exists,
Excel export fails with a message directing the caller to provide `train_data` or retain fit
data.

The renderer accepts both filesystem paths and binary file-like targets so the public
`export_rating_tables()` API and browser download share one implementation. The workbook keeps
its existing sheets and rating-table semantics:

- `Rating Tables`
- `Discretization Impact`
- `Model Summary`

### Structured Model Summary sheet

The ASCII dump is removed. A renderer-independent typed summary payload supplies:

- a model and fit overview;
- information criteria and deviance measures;
- optional distribution-profile estimates;
- term-inference rows; and
- statistical or editor-staleness notes.

The worksheet contains a compact key/value overview followed by a filterable Excel table with
one row per intercept, coefficient, level effect, or whole-smooth test. Columns include, where
applicable:

`Term`, `Group`, `Kind`, `Estimate`, `Std Error`, `Statistic`, `Statistic Type`, `P Value`,
`CI Lower`, `CI Upper`, `EDF`, `Lambda`, `Active`, `Significance`, and `Warning`.

Numbers remain numeric cells with appropriate formats; missing statistics are blank cells.
Headers, frozen panes, filters, widths, and restrained table styling make the sheet readable
without changing its data. This is a normal Excel table, not a pivot table: model statistics
are heterogeneous and non-additive, while a flat typed table remains filterable and can be
pivoted downstream if an analyst needs one.

## Ordered-categorical inference

The implementation integrates the completed `feat/ordered-categorical-inference` behavior at
the shared inference layer; it does not merely hide cells in JavaScript.

For an `OrderedCategorical` backed by `Spline(...)`:

- one Wood-style whole-smooth test represents the term;
- its null is a flat centered smooth;
- the test, EDF, covariance, and scale handling match a directly specified spline;
- the p-value is invariant to the reporting base;
- level rows retain base-relative estimates, standard errors, and confidence intervals; and
- level rows have no z-statistic, p-value, or significance code.

The editor compact payload and Excel payload consume those shared rows, so both display one
global ordered-smooth p-value and blank level p-values. The editor demo is updated from the
deprecated `basis="step"` construction to explicit `basis=Spline(...)` and its selection logic
uses the global term p-value. Legacy step smoothing remains deprecated; independent level
effects belong in `Categorical(...)`.

The ordered-inference integration also retains its corrected reference-level intercept,
weighted `drop1` behavior, canonical spline API warnings, and rating-table base relativity.

## Server and frontend boundaries

- Persistence owns joblib serialization and validation.
- The export package owns typed summary payloads and workbook rendering.
- The editor widget coordinates the authoritative revision and chooses the training and
  validation datasets.
- HTTP routes return either JSON save results or attachment responses with safe filenames and
  correct content types.
- The frontend Export dialog only selects format/destination, initiates the request, saves the
  returned blob, and presents progress or errors.

Legacy model-save/download routes remain available during this change, while the new frontend
uses export-oriented names. Filename validation continues to reject directory traversal.

## Error handling

- No training data: reject Excel export before workbook construction.
- Superseded model revision: reject the request without an artifact.
- Serialization, load, contract, or prediction mismatch: reject Python export and identify the
  failed validation stage without exposing model internals or data.
- Unsupported filename/extension: normalize a missing extension and reject a conflicting one.
- Browser file-picker cancellation: treat it as cancellation, not an application error.
- Kernel-path failures: preserve the requested path in the visible error but do not expose a
  traceback in the browser.

## Verification

Tests cover:

1. joblib round-trip contract and prediction equivalence, including a deliberate mismatch;
2. bounded deterministic validation sampling and the no-data validation status;
3. current-revision enforcement for both export formats;
4. Excel download and kernel-path writing with correct MIME type and filenames;
5. missing-training-data failure for rating-table export;
6. structured workbook values, numeric cell types, blank missing statistics, table/filter
   metadata, and unchanged rating and impact sheets;
7. one global ordered-spline test with non-inferential level rows in Python, editor payloads,
   Excel, and a real browser;
8. Export dialog accessibility, click suppression, successful downloads, cancellation, and
   errors without workspace redraws; and
9. existing public rating-table and model-persistence behavior.

## Out of scope

- A generic exporter plugin registry
- A broad analyst-report workbook with validation charts
- Excel pivot tables or macros
- Exporting raw training, validation, or test rows
- Restoring deprecated ordered step smoothing as a recommended modeling path
