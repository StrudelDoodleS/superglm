# Step backoff for the SCOP REML candidate site

**Issue:** #179 (the remaining asymmetry, after #181 fixed the trigger)
**Status:** designed 2026-07-31
**Release:** folds into the unpublished `0.17.0`; no version bump

---

## The asymmetry

`_fit_scop_reml_mode` returns `None` when a coefficient mode fails latent
penalized-score certification (or its inner fit does not converge). Of the five
`require_converged=True` call sites in `reml/scop_efs.py`, four raise on `None`
and one continues:

| site | on `None` |
|---|---|
| line search | `continue` — tries the next damped `alpha` |
| bootstrap | `raise` |
| fixed-lambda | `raise` |
| candidate | `raise` |
| final fit | `raise` |

Same mode, same quality, opposite outcome depending only on which site asked
for it. On NumPy 2.4.1 the reported fixture's first rejection landed at the
candidate site and killed the fit; on 2.4.2 comparable rejections landed in
the line search and were survived.

## What the structure actually says

Reading `optimize_scop_efs_reml` closely sharpens the picture beyond the
issue's table:

- **The candidate fit runs only on outer iteration 1.** From iteration 2
  onward, `retained_mode` (the line search's result) is always non-`None`, so
  the Step-1 fit is skipped. The candidate site's raise is reachable exactly
  once per optimization, warm-started from the bootstrap.
- **Every lambda movement after iteration 1 is safeguarded.** Each EFS
  proposal goes through `_backtrack_scop_efs_candidate`, which fits damped
  trials (`alpha = ±0.5**attempt`), skips uncertifiable ones, and falls back
  to the current certified mode if everything fails.
- **One lambda movement is not: the bootstrap EFS step.** Its result is
  adopted blindly and fit as a hard candidate. That unguarded step is the
  asymmetry's entire reachable surface.
- **The final-fit refit is reachable only when the loop never ran**
  (`max_reml_iter=0`); on every other path the terminal mode is reused from a
  certified line-search state, and `lambdas` always equals its lambdas.

The governing principle, established on the issue: **a failure is recoverable
exactly when a previously certified state exists to fall back to.** The
bootstrap and fixed-lambda sites have none and must stay loud. The candidate
site has one — the mode its lambda step was taken from.

## The change

On candidate-site `None`, do what the line search would have done: **back off
the lambda step toward the certified mode it was taken from**, instead of
aborting.

A new helper, `_backoff_scop_candidate_step(context, origin, proposed_lambdas,
*, reml_iteration)`:

- For `attempt` in `1 .. _SCOP_EFS_MAX_BACKTRACK_ATTEMPTS - 1`, with
  `alpha = 0.5**attempt`: build trial lambdas by geometric interpolation,
  `log λ_trial = log λ_origin + alpha · (log λ_proposed − log λ_origin)` for
  every component present in both dicts whose proposal differs from the
  origin, clipped to `[1e-6, 1e10]` — the line search's exact trial formula
  and clip, with the certified origin in the role of `current`, including its
  positivity/finiteness guard on the endpoints.
- Fit each trial via `_fit_scop_reml_mode(..., phase="candidate",
  reml_iteration=reml_iteration, trial_alpha=alpha, require_converged=True)`,
  warm-started from the origin's coefficients and SCOP state. Each attempt
  gets the full four-rung certification ladder, cold rung included.
- Return the first certified mode together with its trial lambdas; return
  `None` if every attempt fails.

At the candidate site, `optimize_scop_efs_reml` tracks the step's origin — a
new `step_origin` variable holding `boot_mode` before the loop and updated to
`retained_mode` alongside the existing warm-start state in Step 9 (today the
site only fires on iteration 1, so the origin is always the bootstrap; the
variable keeps the helper honest if the loop structure ever changes). On
`None` it calls the helper; on rescue it adopts the returned mode and its
lambdas and the iteration proceeds exactly as if the candidate had certified
there; on exhaustion it raises today's exact error, `"SCOP REML candidate did
not converge to a coefficient mode"`.

The alpha schedule and budget mirror the line search's forward pass: the full
step (`alpha = 1.0`) was the original candidate fit, so the backoff's seven
attempts complete the same eight-trial geometric ladder the line search runs.

### Why the rescue is sound

As `alpha → 0` the trial approaches the origin — the same lambdas, warm-started
from the origin's own coefficients — and the inner solve reproduces a mode
that certified. SCOP active sets make the fitted mode only piecewise-smooth in
lambda, so no intermediate alpha is guaranteed, but the geometric ladder
searches all the way down to `alpha ≈ 0.008`, where the limiting argument
takes over; if certifiability is recoverable by shortening the step at all,
the ladder finds it. This is
Armijo-style backtracking — shrink the step until the trial is acceptable
(Nocedal & Wright, *Numerical Optimization*, §3.1) — with certification in the
role of acceptability, the same shape as trust-region step rejection. It is
the appropriate safeguard here because the generalized Fellner–Schall update
(Wood & Fasiolo 2017) carries no guarantee about the trial point it proposes;
this codebase already acknowledges that by line-searching every other EFS
proposal. The backoff extends the identical safeguard to the one proposal
that bypasses it.

After a rescue the loop continues from a certified mode at intermediate
lambdas: Step 6 recomputes the EFS step from that mode, and all subsequent
movement already has line-search protection. Nothing needs to remember the
failure — step-size control from that point on is owned by the adaptive
`efs_alpha` sign-flip damping and the line search, as it is today.

### What does not change

- **The success path is bit-identical.** The backoff runs only where today's
  code raises. No fit that currently succeeds can change in any way — not
  its trajectory, not its result, not its trace.
- **The bar and the ladder.** Certification, `_scop_mode_tolerance`, and the
  four retry rungs are untouched.
- **The bootstrap and fixed-lambda sites keep raising.** No certified
  predecessor exists; a loud refusal is the only honest outcome.
- **The final-fit site keeps raising.** It is reachable only under
  `max_reml_iter=0`, where the sole fallback is the bootstrap and backing off
  would silently publish lambdas the caller did not ask for. Zero corpus
  presence; recorded here as deliberate.
- **Error semantics on exhaustion.** Same exception, same message, no new
  return path. `converged` and `termination_reason` never describe a rescue
  as anything other than a normal iteration.
- **A rescue is never the returned terminal state without accepted progress
  after it** (added in review — PR #183, Codex P2). The rescued mode is
  chosen for certifiability, not objective merit; no acceptance gate ever
  endorsed it. If the line search cannot accept a single trial from it, the
  ordinary `line_search_stalled` return would publish half a bootstrap step
  as a REML estimate on an input that raised before the backoff existed.
  That branch now raises the candidate error instead, keyed on a
  per-iteration `rescue_alpha` marker and an identity check on the line
  search's return — which hands back the current mode itself in exactly the
  two no-endorsement cases (every trial rejected, or a no-op proposal
  accepted without fitting anything; the latter found by Codex in round 2,
  where the zero lambda delta would otherwise satisfy strict convergence).
  Stalls and convergence from accepted-progress states keep their existing
  semantics.
- **`lambda_history` records fitted vectors only** (revised in review —
  PR #183). At the candidate site the last history entry is by construction
  the vector that just failed; on rescue it is replaced with the damped
  vector the iteration actually adopted, so consumers reading deltas off
  the history (`_lambda_max_delta`, the governance pack's "REML path
  history") never see a vector no fit used. A rescue is also observable
  outside the trace: `verbose` prints the damping, and the level-2 debug
  payload carries `candidate_backoff_alpha` (null on ordinary iterations).

## Deliberately not doing

**Routing the bootstrap step through `_backtrack_scop_efs_candidate`.** The
structurally elegant endpoint — it would delete the special case, making
iteration 1 a real line search from the bootstrap. Rejected for this fix
because it changes the success path: the line search imposes a
downhill-vs-current acceptance gate, and iteration 1's contract today is to
adopt the EFS step unconditionally. Fits that currently pass could have their
trajectories changed or be newly rejected ("line_search_stalled" at the
bootstrap). That is a behaviour change on every SCOP REML fit and needs its
own validation campaign; worth considering later, not here.

**The graceful stop.** Rejected in the cold-retry design and the reasons
stand: on iteration 1 the only fallback to *return* is the near-unpenalized
`lambda=1e-4` bootstrap; most callers never check `converged`; it
reintroduces the silent degradation #178 removed. The backoff is different in
kind: it returns nothing degraded — it hands the loop a certified mode to
continue from, under the acceptance guard above: a rescue is never published
without acceptance-gated progress after it, and once such progress exists the
result carries the loop's ordinary semantics (including, e.g., an eventual
`max_reml_iter` exhaustion from an accepted state).

**Symmetrizing the other way** (make the line search fatal): strictly fewer
recoveries; the line search's `continue` is load-bearing — it is why NumPy
2.4.2 passes.

**Mutating `efs_alpha` on rescue.** The next proposal is recomputed from the
rescued mode and everything after it is line-search-protected; teaching the
step controller about the failure adds coupling with no case that needs it.

**Reflection in the backoff.** The line search reflects because an EFS
direction can be uphill in the exact objective. The backoff addresses
certifiability, not descent, and by continuity damping toward the certified
origin converges to a certifiable point; a reflected step moves away from the
EFS direction with no such argument. Standard backtracking does not reflect.

## The "unexercised path" objection

The graceful stop was rejected partly because nothing in the corpus would
exercise it. Post-#181 the backoff also has no naturally occurring trigger —
the cold rung rescues the one known real case. The objection does not carry
over, for three reasons:

- The graceful stop changed *result semantics* on its unexercised path
  (callers must start checking `converged`). The backoff preserves them: a
  certified mode mid-loop, or today's exception.
- The backoff's numerics — damped geometric lambda trials warm-started from a
  certified mode — are exercised constantly by the line search on real data.
  Only the trigger is rare, and the trigger is deterministic to inject
  (force certification to reject, the pattern
  `test_a_failed_certification_gets_a_cold_final_attempt` established).
- If the backoff itself fails, the behaviour is exactly today's: the same
  raise. There is no new silent path to go stale.

The asymmetry is worth closing despite #181 because the failure boundary is
one NumPy patch release wide, the certification bar is fixed at
`sqrt(rank·eps)`, and a structural residual the cold rung cannot rescue kills
the fit today. This is defense in depth along the exact seam the analysis
identified as recoverable.

## Testing

- **Rescue (the TDD red).** Bootstrap certification passes; the iteration-1
  candidate's entire four-rung ladder is forced to reject
  (`_scop_mode_newton_relative` monkeypatch, counting checks); subsequent
  checks are real. Before the change this raises the candidate error; after,
  the fit must succeed, the fitted curve must respect the monotone
  constraint, and the certification spy must show attempts beyond the
  exhausted ladder.
- **Exhaustion still raises.** Reject every certification after the
  bootstrap's: the fit must raise the exact candidate message — the backoff
  is bounded and the loud-failure posture survives.
- **The principled boundary.** Reject every certification from the start: the
  bootstrap message must be raised (no certified predecessor → no rescue).
  Fixed-lambda likewise if not already covered.
- **Rescue-then-stall.** Force the ladder to reject, let the rescue certify,
  then reject every line-search certification: the candidate error must be
  raised — a rescue the line search cannot move from is not a publishable
  fit.
- **Corpus inertness.** Across `tests/test_scop_efs.py` +
  `tests/test_scop_irls_state.py`, instrument the ladder on current master
  and again on the branch: the backoff must fire **zero** times, and every
  count — total certification checks, tolerance-rung rescues (29 at last
  measurement), cold-rung rescues (8) — must be identical before and after.
  The change must be invisible everywhere except injected failure.
- **Both sides of the numerics boundary.** Full suite on the default NumPy;
  SCOP suites on 2.4.1 (`test_stored_objective_reproduction_multi_scop`
  stays green — it is rescued by the cold rung before the backoff exists).

## Provenance

Derived from this repository's own line-search machinery and published
literature only: Armijo backtracking and trust-region step rejection (Nocedal
& Wright, *Numerical Optimization*, 2nd ed., §3.1, ch. 4) and the generalized
Fellner–Schall method (Wood & Fasiolo 2017, *Biometrics*). No external
implementation was consulted.

## Release

`release:none`, on the same reading as #178/#180/#181: not independent of the
unpublished `0.17.0` candidate it folds into
(`docs/development/releases.md:15`). Recovery-only: no fit that previously
succeeded can change, because the backoff runs strictly after the point where
the previous code raised. **This does not authorize a tag.**
