# Converged means determined on the directions that matter

Task #17. Follow-up C from the reml_tol determination work, plus two items
from the 2026-08-07 adversarial review and two more discovered during the
PR #243 review rounds.

## Evidence

Measured 2026-08-07 (`scratchpad/criterion_endgame.py`, `criterion_probe2.py`),
three fixtures x reml_tol {1e-6, 1e-9, 1e-11, 1e-13}, with refit-reproducibility
probes from a 30x-different start:

- **flat_12k** (healthy Newton): accepted log-lambda steps contract
  superlinearly (5.0 -> 0.02 -> 3e-4 -> 1e-6 across the ladder), refit gap
  0.0 at every tolerance. Determination IS step contraction.
- **benign_3k**: x1 determined (0.0809, static); x2 is a NULL direction
  marching geometrically to the 1e10 cap forever (~x e per iteration).
  Tighter tolerance just marches longer (9 -> 16 -> 24 iterations); max SE
  identical at every rung. A naive step criterion would never stop.
- **tensor_600** (shared-block margins, collinear multi-order penalties):
  informative lambdas frozen to 4 significant figures; margin_x2 capped at
  1e10; x2 marching x2.4 per iteration. `converged=True` at 1e-6 AND 1e-9
  with refit lambda gaps 0.18/0.053 -- entirely in the marching directions;
  worst SE gap 0.001%, edf identical. At 1e-11 the march exhausts the line
  search: `line_search_failed`, refit gap 1.19, still inferentially void.
  This fixture's 1e-9 endgame flips between BLAS kernel sets (18 clean
  iterations on one platform, a stalled line search on the CI runner) --
  observed live in CI on 2026-08-07.
- **Warm-start path-dependence** (found building a CI-wall regression):
  candidate-grade REML evaluations (search bar 1e-6) are path-dependent --
  resequencing probe order moves nll by ~1e-3, which is LR-scale noise at
  n=3000 and made a real-fixture CI wall test flaky by construction.

Unified diagnosis: every pathological endgame observed is a
lambda->infinity null-direction march the exact path never freezes
(discrete.py:772-780 already freezes flat directions; direct.py has no such
clause). The compound bar stops at an arbitrary point along the march
(platform- and start-dependent), `line_search_failed` is the march hitting
numerical exhaustion, and none of it moves published inference.

## Design

1. **Freeze diverging null directions on the exact path** (port the
   discrete clause): a lambda that marches monotonically upward across k
   consecutive accepted iterations while its objective contribution stays
   below floor is frozen at its current value, removed from the step and
   score criteria, and recorded per-lambda in the profile/diagnostics as
   `frozen_null_directions`. Symmetric clause for a lambda->0 march if
   evidence ever shows one.

2. **Determination criterion over the ACTIVE set, in log-lambda space**:
   converged when the score criterion is met AND the max accepted step over
   unfrozen lambdas has contracted below tau_step. Log-lambda steps are
   scale-free, so one default is portable across data scales -- the
   docstring's determination language becomes literally true. Expected
   effects: benign/tensor fits get FASTER (active lambdas static from
   ~iteration 9-11; no more paying 8-15 iterations of march), and the
   platform sensitivity leaves the stopping decision entirely.

3. **Endgame classification**: with freezing in place,
   `line_search_failed` is reachable only for genuinely stuck ACTIVE
   directions and stays `converged=False`. March-exhaustion cases become
   clean converged stops with frozen bookkeeping. `reml_diagnostics()`
   exposes the frozen set and the final active step.

4. **SCOP plateau exit**: separate engine, measure-then-decide during
   implementation on the 400-row monotone PSpline case: either derive the
   plateau thresholds from reml_tol or classify `objective_plateau` as its
   own honest termination reason surfaced in diagnostics.

5. **Publication budget**: plumb `max_reml_iter` from `estimate_p` through
   to the publication refit; with item 1 the fixed 20 stops binding in
   practice (the long runs were march iterations).

## Re-baseline expectation

Iteration counts DROP on flat/null designs; frozen-direction lambdas
publish at freeze-point instead of march-point (inferentially inert, values
change); `converged` flips False->True on march-exhaustion
line_search_failed cases. Every flip gets root-caused individually; py310
lane check at merge-base. Branch: `reml-converged-criterion`, stacked on
PR #243's head.

## What shipped (2026-08-08)

The implementation deviated from the sketch in three places, each toward a
smaller mechanism the endgame measurements justified:

1. **Freeze floor instead of march detection.** Both Newton engines already
   had the flat-direction freeze (the spec was wrong that direct.py lacked
   it); the defect was its bar, `0.1 * reml_tol`, which the 1e-9 default
   dragged to 1e-10 -- un-freezing what the historical 1e-6 default froze.
   Design item 1's k-consecutive-iteration march detector became a
   tolerance-independent floor: `freeze_tol = max(0.1 * reml_tol, 1e-7)`
   (`FLAT_DIRECTION_FREEZE_FLOOR` in objective.py, calibration in its
   comment). No symmetric lambda->0 clause: no evidence appeared.

2. **Active-set stop by masking, not a new step criterion.** Design item 2's
   tau_step criterion was unnecessary: masking the previous iteration's
   frozen directions out of the gradient fed to the existing compound
   criterion (both Newton engines) stops the loop as soon as the active set
   is determined. A frozen direction's persistent sub-floor gradient
   otherwise spins the loop to max_reml_iter doing nothing -- measured 37
   wasted iterations on the discrete benign fixture.

3. **Endgame classification measures the CURRENT active set only.** A dead
   feasible line search classifies `converged_at_precision` when every
   currently-active gradient sits under the floor. The drafted
   objective-change condition was wrong in principle: the last accepted
   step belongs to a since-frozen flat direction whenever one just crossed
   the freeze bar (measured 7.9e-5 on tensor_600 at 1e-11 -- march
   history, not active-set evidence), and the dead search itself already
   proves no further objective progress exists.

4. **SCOP plateau gated on contraction stall** (design item 4's
   measure-then-decide, decided): the plateau exit may fire only after two
   consecutive iterations fail to contract the max log-lambda step below
   0.9x the previous one. Measured on the 4000-row monotone fixture, the
   ungated plateau granted converged=True at iteration 5 with lambda still
   walking ~1%/iteration and the strict 1e-9 road reachable at iteration
   15; the noise-floor stall it exists to classify measures ratio ~1.05
   (400-row variant). The 0.5 ratio first tried was too blunt -- a steady
   0.6-ratio EFS tail is still buying precision.

5. **Publication budget**: `estimate_p(max_reml_iter=...)` reaches the REML
   publication refit only; candidate fits keep their budget; a pure-ML
   publication refuses the parameter rather than hosting an inert knob.

Not shipped here: `lambda2_init` is silently discarded by the SCOP EFS
bootstrap (measured: identical 16-digit trajectories under None and 1e4) --
an inert-parameter finding of the same family as the SCOP `convergence`
parameter, left for its own decision.
