# Retiring the SCOP deviance-stagnation acceptance gate

**Roadmap item:** `docs/superpowers/plans/2026-07-30-shape-constraint-roadmap.md`, Item 2c
**Status:** approved 2026-07-31, not yet implemented
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

**No real fit anywhere in the corpus reaches the gate.** Its only remaining
consumers are the tests that exist to test it. The boundary fixture that
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

### 2b is already satisfied

The roadmap sequences 2b — "strengthen the acceptance gate to the published
gradient-norm test on the penalized deviance" — before 2c. That sequencing is
moot once the gate is gone, and the criterion it asks for is already in place:
`_scop_mode_newton_relative` (`scop_efs.py:701`) forms a relative Newton
correction from the profiled penalized score, and `scop_efs.py:1089` rejects any
mode exceeding `_scop_mode_tolerance`, with up to two tightening retries. That
is a score-based stationarity certification on the penalized objective, and
after this change it is the *sole* acceptance path. Removing the gate leaves a
stronger posture than the 2a->2b->2c sequence anticipated, not a weaker one.

Implementing a second gradient-norm test would be redundant. 2b closes with 2c.

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
- `tests/test_scop_irls_state.py:374-388` — drop the channel assertions.
- `tests/test_fit_transactions.py:373-395` — this test is about
  `_freeze_result_arrays` not rebuilding a dataclass, and borrowed
  `StagnationRecord` only as a convenient mutable dataclass reachable from a
  result. Re-point it at `IterationDiagnostics`, which is the analogous
  surviving per-iteration dataclass on `PIRLSResult` and preserves the test's
  actual subject.
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
