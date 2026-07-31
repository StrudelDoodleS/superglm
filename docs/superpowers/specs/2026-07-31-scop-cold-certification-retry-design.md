# A cold rung for the SCOP certification retry

**Issue:** #179
**Status:** approved 2026-07-31, not yet implemented
**Release:** folds into the unpublished `0.17.0`; no version bump

---

## The symptom

`fit_reml` on a monotone-constrained (SCOP) model raises

```
RuntimeError: SCOP REML candidate did not converge to a coefficient mode
```

and returns no model at all, on data it should handle. Whether a fit succeeds or
raises depends on numerical luck: the same code and the same data pass on NumPy
2.4.2 and fail on 2.4.1.

## What is actually wrong

While fitting, every candidate coefficient mode must pass a *certification* —
`_scop_mode_newton_relative` against `_scop_mode_tolerance` — confirming it is a
stationary point of the penalized likelihood. When certification fails,
`_fit_scop_reml_mode` retries up to twice before giving up.

**Both retries vary the one thing that cannot help, and never vary the thing that
can.** Each rung tightens `pirls_tol` and warm-starts from the mode that just
failed. Measured across `tests/test_scop_efs.py` + `tests/test_scop_irls_state.py`:

| | |
|---|---|
| certification checks | 609 |
| rejections | 51 |
| retries (depth > 0) | 43 |
| retries that warm-started | **43 of 43** |
| rescues | 29 |

The tolerance rungs are not useless — they rescue 29 modes whose residual was a
convergence artifact. But when the inner fit has already converged far tighter
than the bar, tightening it changes nothing, and the retry reproduces the failed
mode bit-identically. On the reported failure the achieved value is exactly
`1.3792e-06` at `pirls_tol` of 1e-6, 1e-10 and 1e-11 — three attempts, one
number, against a fixed bar of `7.1463e-08`.

A cold start is therefore never tried, and the warm start is a plausible cause:
the candidate is warm-started from a bootstrap fitted at `lambda=1e-4`, a long
way from the candidate's own lambdas.

## The change

Add a **final cold rung** to the certification retry in `_fit_scop_reml_mode`.

Precisely: the guard is `if _certification_retry < 2`, so today there are three
evaluations at depths 0, 1 and 2 — the original plus two tolerance retries.
Extend the guard to `< 3` and make the transition **from depth 2 to depth 3** the
cold one: `beta_init=None`, `intercept_init=None`, `scop_state_init=None`.
Depths 0→1 and 1→2 keep their current warm, tolerance-tightening behaviour
unchanged. The cold rung inherits the tightest `pirls_tol` the ladder reached, so
it differs from its predecessor in exactly one respect: the starting point.

Measured on NumPy 2.4.1, the platform that fails today: one candidate returned
`None` warm, the cold refit rescued it, and
`test_stored_objective_reproduction_multi_scop` passes.

### What does not change

- **The bar.** The cold mode must pass the same certification against the same
  `_scop_mode_tolerance`. Nothing about what the solver accepts moves.
- **The two tolerance rungs.** They rescue 29 of 43 retries; they stay.
- **Error handling.** If the cold rung also fails, `_fit_scop_reml_mode` returns
  `None` and the caller raises exactly as today. This is strictly more
  recoveries, never a new silent path.
- **The four fatal call sites.** Bootstrap, fixed-lambda, candidate and final fit
  keep raising on `None`.

## Deliberately not doing: the graceful stop

The obvious alternative is to make a candidate certification failure stop the
REML loop cleanly — return the last certified mode with
`termination_reason="certification_failed"` and `converged=False` — rather than
raising. Rejected, for three reasons.

**It would not fix the reported case.** The failure occurs on loop iteration 1,
where the only fallback is the `lambda=1e-4` bootstrap: a nearly unpenalized fit.
Returning that silently is a worse outcome than the error.

**It reintroduces silent degradation.** The deviance-stagnation acceptance rule
was retired on 2026-07-31 precisely because it silently accepted a fit that had
not converged. A graceful stop is the same shape: most callers do not check
`converged`. Adding one back immediately after removing one is the wrong
direction.

**Nothing would exercise it.** No case anywhere in the corpus has a candidate
failing certification on iteration 2 or later. It would ship untested, against a
scenario constructed to trigger it — which is how the current asymmetry survived
this long.

A loud refusal is recoverable: the user adjusts `k`, lambda, or the data. A
quiet early stop is not, because they never learn it happened. Roadmap Item 3
already commits to that posture.

## Known limitations

**The asymmetry remains.** A certification rejection is still fatal at four call
sites and survivable at the fifth (the line search, which `continue`s to the next
trial). This change makes reaching a fatal site much rarer; it does not remove
the inconsistency. #179 stays open for that.

**One data point.** The cold rescue is measured on exactly one failing case — the
reported one. The mechanism is principled, but this does not establish that cold
starts help generally.

**Raising the NumPy floor is not an alternative.** The failure boundary is a
single patch release (2.4.1 fails, 2.4.2 passes), so the fit sits exactly on the
threshold and any numerics change can flip it. Worse, numba requires
`numpy <= 2.4`, so the usable window is `>=2.4.2, <2.5` — one release wide, with
a ceiling imposed by a different dependency. Pinning is not available.

## Testing

- **Regression:** `test_stored_objective_reproduction_multi_scop` currently fails
  on NumPy <= 2.4.1 and must pass after the change. Verify the full suite on the
  default NumPy *and* on 2.4.1, i.e. both sides of the boundary.
- **New:** a test pinning that a candidate which fails certification while
  warm-started gets a cold attempt. Assert through the observable outcome — the
  fit succeeds where it previously raised — not by counting calls into the
  retry, so the test survives refactoring of the rung structure.
- **No regression in rescues:** re-run the retry instrumentation and confirm the
  29 tolerance-rung rescues still occur. The cold rung is additive; if the count
  drops, the rungs have been reordered wrongly.

## Release

`release:none`, on the same reading used for #178 and #180: not because there is
no runtime impact — a fit that previously raised may now succeed, which is
runtime impact on any reading — but because it is not *independent* of the
unpublished `0.17.0` candidate it folds into. `docs/development/releases.md:15`
turns on that word.

Version stays `0.17.0`; no bump. The behaviour change is recovery-only: no fit
that previously succeeded can now fail, because the cold rung is only reached
after the existing path would have returned `None`. **This does not authorize a
tag.**
