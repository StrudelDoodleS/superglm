# Retiring the SCOP deviance-stagnation acceptance gate

**Roadmap item:** `docs/superpowers/plans/2026-07-30-shape-constraint-roadmap.md`, Item 2c
**Status:** approved 2026-07-31; implemented on branch
`refactor/retire-scop-stagnation-gate`, commits `9d0d67c..e6868e3`
(`f3c3ac9` spec, `51d6036` plan, `00ff8a3` gate removal, `b5d1c74` dead logger,
`17d1439` `stagnation_log` channel, `e6868e3` roadmap status)
**Release:** folds into the unpublished `0.17.0`; no version bump

---

## Why now

Item 2 replaced a symptom-level treatment with the published fix. The
`exp(gamma) -> 0` boundary makes a SCOP coefficient unidentifiable, and under
`convergence="coefficients"` the criterion measures a quantity that no longer
tracks progress: the coefficient keeps drifting while the fit stops changing.
Our answer was to accept such a fit when its deviance stopped moving. The
published answer is to drop the unidentifiable direction, and PR #176 shipped
exactly that — rank truncation on the factor, with the resolved range carried
through mode certification, `H_penalized`, and both determinant routes.

Item 2c asks the follow-on question: with the cause fixed, is the workaround
still needed? This spec answers it with measurement, then removes it.

## Evidence

The gate is reached from exactly one site, and only when
`require_converged and not result.converged`. Instrumenting
`_scop_deviance_stagnated` and running the full suite therefore counts every
non-converged SCOP inner fit that reaches it.

Full suite, 5133 passed / 84 skipped, `-p no:randomly`:

| | count |
|---|---|
| gate calls | 33 |
| acceptances | 7 |
| acceptances from a real fit (`_fit_scop_reml_mode` on the stack) | **1** |
| that one test | `TestSCOPBoundaryStagnationAcceptance::test_gate_admits_a_stagnant_candidate` |

Every call originates in `tests/test_scop_efs.py`. Six acceptances are direct
unit calls on synthetic results; the seventh is a deliberately constructed
stagnant stub (`n_iter=256` against `max_iter=200`) inside the gate's own test
class. The declines are the same class plus outer-loop tests running caps of 1
and 10.

**No real fit anywhere in the corpus is accepted by the gate.** Reaching it and
being accepted by it are different bars — any non-converged inner fit reaches
it, and only a stagnant deviance is admitted, so the declines above already
include real outer-loop fits, as does this branch's own
`test_a_genuinely_non_converging_fit_still_raises`: a real fit that reaches the
gate's site and would be declined there. The only things that can still drive
the gate to accept are the tests that exist to test it. The boundary fixture that
motivated it went from 34 iterations plus a 35-iteration `max_iter` run to 9, 1,
1 — all converged — under #176.

The gate tests are stub-driven without exception. That is itself part of the
finding: there is no longer a real fit that can drive them.

## Decision

Remove the gate and the private diagnostic channel that exists to feed it. A
SCOP fit that exhausts `max_iter` on a stagnant deviance fails loudly, as any
other exhausted budget does.

Two considerations settle the safety question. Roadmap Item 3 already commits to
the posture that a loud refusal beats a silent accept. And the gate was never
the stationarity test: every mode it admitted still had to pass the penalized-
score certification downstream, which is untouched here.

### 2b is ambiguous, and this change does not resolve it

The roadmap sequences 2b — "strengthen the acceptance gate to the published
gradient-norm test on the penalized deviance" — before 2c, and in the same
breath says to **keep deviance stagnation as the primary criterion; the
reference method does**. That second clause has two readings, and
`docs/superpowers/specs/2026-07-30-shape-constraint-strategy.md` §4.2 (127-135)
— the note 2b was written from — names both objects explicitly: the reference
method's *primary convergence test* (relative penalized-deviance stagnation)
and its *acceptance gate* (the gradient norm).

**Gate reading.** "Deviance stagnation" is the acceptance rule this change
removes. Then 2b's premise is gone: there is no acceptance gate left to
strengthen, and the thing it asked to preserve is the thing being retired.

**Criterion reading.** It is the *inner PIRLS convergence test* — which is what
the reference method actually runs it as, that method having no acceptance
override at all. Then 2c removes neither of 2b's objects. What it removes is a
third thing, invented here: a post-hoc acceptance override applied after a
fit under `convergence="coefficients"` had already failed. Both of 2b's asks —
adopt the gradient-norm acceptance test, and make deviance stagnation the inner
convergence criterion — survive intact and remain implementable.

Neither reading is obviously right. The following clause, "and abandons the
coefficient-step test exactly as we did", cuts toward the gate reading, because
the coefficient-step test is what the gate stood in for. But under the criterion
reading 2b is *more* attractive after this change, not moot: making deviance
stagnation the inner criterion is arguably the principled replacement for what
is removed here. A fit whose penalized deviance had stopped moving would then
converge legitimately on its own criterion, rather than being overridden after
the fact — which is exactly the residual risk named under **Behaviour change**
below (data that still stagnates after rank truncation, of which we have no
example). This spec therefore records 2b as **unresolved**: not closed, not
superseded, not done. Deciding it requires deciding what it meant.

What certifies a mode afterwards is `_scop_mode_newton_relative`
(`scop_efs.py:698`), gated unconditionally at `scop_efs.py:931-962` against
`_scop_mode_tolerance` with up to two tightening retries, and after this change
the *sole* certification. It is not the sole acceptance precondition:
`require_converged and not result.converged: return None` (`scop_efs.py:857`)
still rejects a mode before certification is ever computed.

It is score-based, but a different object — a Newton-scaled *relative*
correction restricted to the estimable range — and asymmetric against a raw
gradient norm on two independent axes, not uniformly stronger on either.

- **Resolved vs truncated.** On a truncated direction it is deliberately
  silent: the profiled score is projected onto the solver's resolved range
  **before** the pseudoinverse (`scop_efs.py:740-747`), so a score a
  gradient-norm test would flag is discarded outright rather than required to
  vanish.
- **Flat vs stiff.** On the resolved range the two are not orderable. In an
  eigendirection of curvature `h` the criterion is `|g| / (h * max(1, |beta|))`
  against a fixed floor (`scop_efs.py:753-763`, `766-775`), against `|g|`
  against a gradient threshold — relatively more demanding as `h` falls, less
  as `h` rises. `H⁺g` is a step and `g` is a slope, so no threshold pair makes
  the Newton form uniformly more demanding across the resolved range. It is
  more demanding exactly where curvature is small — the flat-boundary regime 2b
  was aimed at — and less demanding where curvature is large.

Both asymmetries point the same way: what shipped is not a superset of 2b, so
2b cannot be marked done on the strength of it under either reading. And the
certification is vacuous on non-SCOP modes, where an empty `scop_states` zeroes
the mode score outright (`scop_efs.py:894-916`), leaving nothing to measure.
Anyone who wants 2b's literal gradient-norm test should treat it as
unimplemented and argue it on its merits.

## What is removed

Roughly **210 production lines** across three files, and **~480 test lines** —
about 690 in total. The roadmap's own estimate of "~90 lines" counted only the
gate's justification comment and predicate; the channel behind it is most of the
remainder.

Line anchors are current-tree and will drift during implementation.

### `src/superglm/reml/scop_efs.py` (~160 lines)

- `781-825` — the 42-line empirical justification comment and the three tuned
  constants (`_STAGNANT_DEVIANCE_TOLERANCE`, `_STAGNANT_DEVIANCE_WINDOW_MAX`,
  `_STAGNANT_DEVIANCE_WINDOW_MIN`), derived from 1017 corpus inner fits.
- `828-841` — `_scop_stagnation_window`.
- `844-900` — `_scop_deviance_stagnated`, including the window-contiguity
  `RuntimeError`.
- `973-981` — the `_record_stagnation=True` request and its comment.
- `991-1014` — the acceptance branch: the predicate call, the `logger.info`
  disclosure, and `result.converged = True`.
- `1280-1293` — the publish-time scrub and its comment.

### `src/superglm/solvers/pirls.py` (~27 lines)

- `StagnationRecord` (from `@dataclass(frozen=True)` at `225` through the class).
- The `stagnation_log` field at `296` and its 16-line note at `280-295`. That
  note documents a hazard — `fit_pirls` builds no `stagnation_log`, so routing a
  new caller through it would silently downgrade to raising — which ceases to
  exist along with the gate.

### `src/superglm/solvers/irls_direct.py` (~22 lines)

- The `StagnationRecord` import at `71`.
- `_record_stagnation` on both signatures (`391`, `488`), the forward that
  threads one to the other (`436`), and its docstring entry (`564-567`).
- The `stagnation_log` accumulator at `1264`, the guarded append at `1988-1996`,
  and the field pass-through at `2557`.

### What remains

`_fit_scop_reml_mode` keeps `if require_converged and not result.converged:
return None`. The mode certification at `1089` is untouched.
`iteration_log`/`record_diagnostics` remains as the general per-iteration
diagnostic — it is a separate, wider channel with its own consumers.

Each SCOP inner fit also stops paying for the per-iteration append.

## Behaviour change

A SCOP inner fit that exhausts `max_iter` with a deviance stagnant to roundoff
is no longer flipped to `converged=True`. It is rejected like any other
non-converged fit, and the caller raises `SCOP REML candidate did not converge
to a coefficient mode`.

Blast radius: no fit in the corpus takes this path. The change is real but
latent — it can only surface on data that still stagnates after rank
truncation, which we have no example of.

`stagnation_log` and `_record_stagnation` are private, and the field is
explicitly scrubbed before publication "so it never becomes a solver-dependent
public surface," so their removal has no public-contract impact.

This also retires the Python-3.10-only `converged is True` /
`termination_reason == "max_iter"` disclosure the roadmap flags at Item 2.

## Testing

**Delete** — they test machinery that will not exist:

- `TestSCOPBoundaryStagnationAcceptance` (`tests/test_scop_efs.py:2363-2691`, 329 lines)
- `TestStagnationChannelMatchesTheDiagnosticsRecorder` (`2692-2772`, 81 lines)

**Invert and keep one.** Replace `test_gate_admits_a_stagnant_candidate` with
its opposite. Concretely: drive `_fit_scop_reml_mode` with the same
long-stagnant-run stub the deleted test used, and assert that it now yields the
non-convergence — the stub never reaches geometry assembly, and `converged`
stays `False` with `termination_reason == "max_iter"` rather than being flipped.
The behaviour change should be pinned by a test rather than disappear with the
code that motivated it.

**Guard the regression.**
`TestMultiSCOPIntegration::test_stored_objective_reproduction_multi_scop`
(`3878`) is the genuine log-space-boundary fit that leaves the identified
Hessian near-singular. It must pass unchanged. This is the load-bearing
assertion of the whole change: it demonstrates #176 carries what the gate used
to carry.

**Edit, don't delete.**

- `TestSCOPREMLDoesNotPublishDiagnostics` (`2839-2885`) — drop the
  `stagnation_log` assertions, keep the `iteration_log` ones.
- `tests/test_scop_irls_state.py:341-388` —
  `test_stagnation_channel_matches_the_recorder_on_step_quality` exists to
  compare the two channels, so its subject evaporates with one of them. Its
  value does not: it is the only place that drives `step_halvings` and
  `step_rejected` to non-trivial values instead of hoping a fixture reaches
  them. Re-point it at `iteration_log` and rename accordingly.
- `tests/test_fit_transactions.py:368-397` —
  `test_published_stagnation_records_keep_their_fields` is about
  `_freeze_result_arrays` not rebuilding a dataclass as a bare tuple, and
  borrowed `StagnationRecord` only as a convenient example. **Delete it
  outright rather than re-pointing:** the property it guarded is already
  covered on the surviving dataclass — though *not* by
  `test_published_result_deeply_freezes_diagnostics_and_rank_metadata` at
  `348-365`, despite the surface resemblance. That test asserts
  `result.iteration_log[0].deviance = -1.0` raises `AttributeError`, which pins
  immutability, not field retention: if `_freeze_result_arrays` rebuilt the
  entry as a bare tuple — the exact failure the deleted test guarded — the
  assignment would raise `AttributeError` for the wrong reason and the
  assertion would still pass. What does cover it is `iteration_diagnostics()`
  (`src/superglm/model/api.py:773-808`), which reads 30 *named* fields off a
  published result's `iteration_log`, exercised by
  `TestIterationDiagnosticsSmallSample::test_opt_in_diagnostics_survive_small_samples`
  in `tests/test_scop_efs.py`. A bare-tuple rebuild fails loudly there.
- `TestIterationDiagnosticsSmallSample` (`2773-2838`) — one incidental mention.

**Verify.** Re-run the instrumented probe after removal to confirm no gate path
survives, and confirm the suite's pass count moves only by the count of deleted
tests.

## Release classification

`docs/development/releases.md` states that an existing unpublished release
candidate "blocks automation and requires a human decision." Decided
2026-07-31: **fold into `0.17.0`, no version bump.**

Rationale: this change is the direct consequence of #176, which produced
`0.17.0`, and one release then tells one coherent story — rank truncation on the
factor, plus the workaround it retires. It does not advance past the unpublished
candidate, so the one-release-bearing-PR-at-a-time rule is unaffected. The
behaviour change is declared in the PR body rather than hidden behind a
`release:none` label.

`0.16.2` remains skipped and must never be published.

## Out of scope

- Publishing `0.17.0` — deferred by explicit decision; publication requires its
  own instruction naming the tag.
- Roadmap Items 1, 3, 4 and the rest of the shape-constraint roadmap.
- Any change to `bs`/`cr` QP-path behaviour.
- The `iteration_log`/`record_diagnostics` channel.
