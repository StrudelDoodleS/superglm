# Implicit state machines: what the repository holds, and what to do with them

**Question.** Where does the code keep a lifecycle (a small set of named states and the moves
between them) as scattered flags, status strings and counters, and what one rule serves the
analyst using the editor, the developer and the agents that change the code?

**Answer.** One rule, applied where it pays:

> A lifecycle is one tagged value with one declared table of moves, in the module that owns it.
> That module's function is the only writer, and a test walks the table.

No library. The editor ships plain JavaScript with no build step, and no lifecycle found here
needs more than a flat table. The rule is in `AGENTS.md` (*Code and tests*), and the editor's
pattern is `src/superglm/editor/app/state/machine.js` with `state/mutation.js`.

It is applied only when a change adds a state to a lifecycle or fixes a wrong state in it. The
numeric search loops are listed below and left alone: they are well commented, sit on the path
that decides time to fit, and making them explicit would move no number a user reads.

## Why this rule

The problem is a product type used where a sum type is meant: three booleans can hold eight
combinations when the lifecycle has four states, and the other four are screens nobody designed.
The established answers are the statechart (Harel, *Statecharts: a visual formalism for complex
systems*, 1987): named states, events and transitions, with separate regions for concerns that
vary independently; and, for a request, one type that holds every state the request can be in
instead of loading and error flags (the `RemoteData` type Kris Jenkins described for Elm, as its
ports to other languages record). Both assume a small discrete state space moved by named events.
Every lifecycle in the tables below meets that; a numeric iteration does not, and is excluded.

Statechart libraries add hierarchy and a runtime. Nothing here needs hierarchy, and the editor
takes no runtime dependency, so the table is thirty lines of our own.

## What this pass changed

Two wrong screens were reproduced in a browser on the unchanged code, then fixed by making the
state explicit. Each has a browser test that fails before the fix.

| Wrong screen | Cause | Fix |
|---|---|---|
| Pressing Undo (Ctrl+Z), or saving a History note, during a Free levels fit removed the busy overlay while the fit was still running. | Two writers of "the editor is busy": the store's mutation state, and Free levels calling the overlay directly. Any move of the first cleared the second. | Free levels runs through `actions.holdEditor`, so it is a `running` state like a refit. The overlay has one owner, and Undo and notes wait. |
| In a window wide enough to keep the feature list open, a failed distribution request on one feature left its message and Retry on screen after going back to a feature already loaded. | One variable meant both "the panel has been drawn" and "the panel shows this drawing". A message over the drawing left the second untrue, so the redraw was skipped. | `distributionShown` records whether the drawing is still what the panel shows. |

The mutation lifecycle is now `MUTATION` in `state/mutation.js`: three states (`idle`, `running`,
`error`), six events, one function (`moveMutation`) that every write in `actions.js` goes through.
The store's shape and every status string are unchanged.

## Editor front end (`src/superglm/editor/app`)

| Lifecycle | Where it lives | Encoded as | Standing |
|---|---|---|---|
| The editor's hold and its alert | `request.mutation`, `request.recovery` | was seven object literals written in `actions.js`, and the overlay set directly from `main.js` | **Done here** |
| Profile dialog (Tweedie p, NB theta) | `runDistributionProfile` in `summary.js` | dialog open, a disabled button, a polled status | **Next.** Runs outside the hold while Python keeps its lock for the whole profile, so closing the dialog mid-run leaves a page that looks idle and answers nothing. Run it through `holdEditor`. |
| Fixed-offset refit | `runOffsetRefit` in `summary.js` | a disabled button and `aria-busy` | **Next**, with the profile dialog: the same case. |
| Distribution panel | 13 module variables in `main.js` (`distributionPayload`, `...Drawn`, `...Shown`, `...Stale`, `...InView`, `...RequestDue`, `...Replies`, ...) | request, display, in-view, drag preview and reply cache, all as loose variables in a file the type checker does not read | **Next.** Move to a typed module with its client and drawing passed in, so node tests can drive it. The 16 browser tests guard the move. |
| Evidence panels (metrics, summary, report) | `request.evidence[panel]` | `status` of five strings with revision and sequence | Already one writer (six pure functions in `state/store.js`). What is scattered is the display: "never asked" is `idle` with no payload in three places, and the summary works out its own status in `renderSummaryEvidence`. **Next:** one `evidenceDisplay` selector for the three panels. |
| Retry in progress | `retryInProgress`, `retryRecovery` in `main.js` | two variables beside the store | The alert stays up, disabled, while a retry runs. A `retrying` state in the table would replace both, and changes a status string tests read. Later. |
| Background jobs in the Cross-validation tab | `state.jobs`, `starting`, `polling`, `errors` in `views/cv_tab.js`; `pillSeenJob` in `main.js` | a server status plus three client sets | One tagged phase per job kind. Later. |
| Pointer gestures | the `interaction` object in `interactions.js` | seven nullable fields, at most one set at a time by convention | One `gesture` value with a `kind`. Later; a wrong combination here is a stuck brush or drag. |
| Chart tool | `view.mode`, `view.showContrib` | a typed mode, with forced moves in four functions of `main.js` | One function that decides the forced moves. Later. |
| Contribution build animation | `buildProgress`, `buildFrame` in `main.js` | a number or null, and a frame id or null | Later. |
| Rating table request | `ratingTableSequence` in `main.js` | a counter; latest reply wins | Later, with the distribution panel's request. |
| Selection sent to Python | `latestView`, `unsentView`, `viewTimer`, `viewSends` in `state/actions.js` | four variables in one closure, 13 tests | Leave. |
| History note queue | `noteSave` in `state/actions.js` | one promise chain | Leave. |
| Unsmoothed line | `view.unsmoothed[term].status` | already a typed set of four statuses | Leave. |
| Free levels shown | `view.freeLevels`, `view.showFreeLevels`, `fit_token` | a kept comparison, a toggle and a token match | Leave. |
| Knot gestures | `knot_gestures.js` | typed, one closure, 31 tests | Leave. |
| Busy overlay | `appBusyActive` and three more in `main.js` | one caller in the app after this pass | Leave. |
| Export dialog, inspector, theme, popover, Diagnose tab, notebook refresh | their own modules | one or two flags each, inside one closure | Leave. Found by search, not read line by line. |

`main.js` also keeps the feature search, the summary search and filter, and the open sections in
module variables. They are view state, not lifecycles, but `editor-frontend.md` says view state
belongs in the store.

## Editor back end (`src/superglm/editor`)

Read by a sub-agent; those marked *checked* I confirmed in the code myself. None was run.

| Lifecycle | Where it lives | Problem | Smallest explicit form |
|---|---|---|---|
| Profile job | `_profile_jobs` in `widget.py` | A second job runner with its own words (`complete`, `error` against `done`, `failed`). No guard against a second run, no cancel, ignored by `close()`, and it holds the widget lock for the whole profile (*checked*). | Make it a `JobRunner` kind. |
| "Superseded" replies | 20 `return _superseded_payload(...)` sites in `widget.py` | Four causes share one string. Export raises `RuntimeError` for the same case, which the server answers as "internal editor error" (*checked*). | One private exception raised at each checkpoint and turned into the reply in one place. |
| Edit, waiting and applied timeline | six lists on `EditorSession` (`history`, `redo_stack`, `pending`, `pending_redo`, `structure_history`, `structure_redo`) | "A new action ends what was undone" is written by hand in about five places. | A `Timeline` class that owns the lists. About 250 lines moved, none added. |
| Live or stale results | `model_revision` compared in `widget.py`, `cv.py`, `reports.py` | The same comparison written five times. | One `freshness(stamped, session)` helper. |
| Background jobs | `JobRunner` in `jobs.py` | A superseded job ends as `failed` with a sentence; the page can tell only by the text. | A typed status with a `superseded` end. |
| Open or closed | four separate `_closed` flags (`widget.py`, `jobs.py`, `evidence.py`, `notebook.py`) | After `close()` a direct call raises `RuntimeError` from whichever part notices first. The server has stopped by then, so a user is unlikely to see it. | One state on the widget and one check in the route wrapper. |
| Fit-token caches | `_fit_token`, `_free_levels`, `_unsmoothed` in `widget.py` | Two requests for one term both run the fit. | A small helper keyed by the token. |
| Chart and state generations | counters in `widget.py` | Bumped by hand at seven sites, and not after a New levels choice or a profile. | One `_bump_chart()`. |

Already explicit: the dataclasses in `_types.py`, `EvidenceOutcome.status`, the route classes in
`notebook.py`, and the tool-availability reasons in `lss/availability.py`.

## Library (`src/superglm`, outside the editor)

Read by a sub-agent. Items 4 and 5 I confirmed by reading; none was run.

| # | Lifecycle | Problem | Smallest explicit form | On the fit's hot path |
|---|---|---|---|---|
| 1 | SuperGLM fitted, released, revised | At least 11 wordings of "not fitted" across `model/`. Installing a fit is already one atomic swap. | One `_require_fitted(model, what)`. Changes message text, so tests change. | No |
| 2 | REML stop reason | `REMLResult.termination_reason` is a bare string written at about 20 sites; the PIRLS one is already a `Literal`. No test compares the writers with the tables that read them. | A `Literal` with today's strings, and a test that the reader tables cover it. No runtime change. | No |
| 3 | Inference caches after a refit | The list of caches to drop is written out at 8 sites with different subsets. | One `drop_fit_inference_caches(model)`. | No |
| 4 | SuperLSS fit | `_model` is assigned, then warnings are raised, then `_fit_settings` is assigned by the caller. With warnings turned into errors, the object holds the new fit and the old settings. | Assign all four attributes together, before the warnings. | No |
| 5 | sklearn wrapper | Attributes ending in `_` are set before the fit runs, so a failed refit leaves new columns beside old coefficients. | Assign them at the end of `_fit_common`. | No |
| 6 | Private mode flags on clones (`_in_power_search`, `_nb_theta_search`, `_defer_rank_info`, ...) | Bare attributes read with `getattr(model, "_x", False)`. | One frozen `FitRole` on the workspace. | Read once per fit |
| 7 | SuperLSS outer search, REML outer search, IRLS curvature switch, centring rungs, screening route ladder | Loose variables inside long loops. | **Leave.** Each change here needs timing evidence and moves nothing a user reads. | Yes |

Already explicit: about 25 `Literal` status types (PIRLS termination, the LSS convergence and
endpoint reasons, curvature kinds, fit-report statuses), the frozen fit-state dataclasses, and the
solver-route inventory in `tests/test_no_auto_fallback.py`.

## Order of work

1. Profile and fixed-offset refit through `holdEditor` (front end), then the profile job as a
   `JobRunner` kind (back end). Same defect class as the two fixed here.
2. Distribution panel into a typed, node-tested module.
3. One `evidenceDisplay` selector.
4. The back end's superseded exception, then the `Timeline` class.
5. Library items 2 to 5, each its own small change.

## How this was found

I read the front end's state, action and main modules in full and searched the rest for mutable
variables and status comparisons. Two sub-agents inventoried the Python back end and the library
by reading; their reports name which files were read in full and which were searched. Claims not
marked as checked or reproduced are readings of the code, not observations of a failure.
