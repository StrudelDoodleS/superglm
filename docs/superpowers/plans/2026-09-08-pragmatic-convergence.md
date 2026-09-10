# Pragmatic distributional convergence follow-through

The user asked to revisit overly harsh smoothing completion, accept useful
effectively infinite smoothing, and defer cross-predictor penalties. Continue
from the completed C3/C1 implementation at `1a8952a4`; preserve its raw evidence
and the original strategy documents in `.worktrees/roadmap-dossier`.

## Findings and intended behavior

The 610,212-policy NB2 benchmark used strict `practical_reml=False`; its final
penalties were moderate. Repeating with practical stopping enabled reproduces
the same Newton rejection. Fresh differentiation fails after an improving
accepted Newton fit. The recovery then tries an opposite half-step, rejects
its worse objective, and publishes derivatives belonging to an earlier fit.
This is not evidence that the coefficient fit is unusable or that infinity
penalties are invalid.

1. Preserve the accepted fit and resume bounded EFS when fresh Newton gradient
   evaluation becomes unavailable. Disable further Newton handoffs for that
   solve and clear unavailable terminal derivatives. Keep existing full-profile
   objective acceptance. Do not silently bring released beyond-cap parameters
   back inside their original box.
2. Permit practical completion when sustained objective and fitted-parameter
   changes are small even though genuinely outward smoothing steps stay large.
   Require fresh outward pressure and real accepted movement; duplicate
   cap-clipped trials cannot establish insensitivity. Keep separate evidence
   for a practical finite fit and an exact penalty-face result.
3. Retain inward-pressure refusal: very large penalties and tiny fitted changes
   can also occur in an over-smoothed model. Gaussian ridge gives a simple
   analytic counterexample without depending on roundoff.

## Stages and ownership

- [x] Audit the rejected NB2 trajectory, existing practical/endpoint controls,
  and primary references. Reproduce practical-enabled Newton behavior.
- [x] Finish the unmodified plain-EFS insurance comparison (24-iteration practical plateau).
- [x] Worker: Newton recovery and truthful derivative provenance in
  `smoothing/newton.py`, `smoothing/loop.py`, and focused endgame regressions.
  Write and demonstrate failing tests before changing production code.
- [x] Worker with max mathematical review: outward practical stopping and replay validation, with analytic
  inward/outward tests and distributional prediction/uncertainty regressions.
- [x] Refit the exact insurance workload, compare numerical outputs and actual
work/dispatch, and retain raw artifacts. Clock claims require a separately
  recorded run because external processes may contend with numerical work.
- [x] Investigate and, if supported by independent numerical evidence, repair
  the finite-NB2 numerical guard that blocks coefficient movement on a
  low-mean policy. Preserve the separate exact-Poisson limit behavior.
- [x] Independent review, focused and ordinary checks, and updated public
  guidance, roadmap and evidence report.

## New findings during validation

- Recovery retains the improving NB2 fit and stops practically in 11 iterations.
  Held-out means barely move, while changed smoothing penalties alter conditional
  coefficient covariance by 12.4%. This is a different smoothing solution, not
  representation parity.
- The terminal coefficient solve uses `objective_and_step` after 28 backtracks.
  Its full local Newton correction remains nontrivial. An independent
  fixed-lambda directional check confirms an improving step is blocked by the
  kernel's binary64 near-Poisson guard on one low-mean row; the likelihood and
  derivatives agree. Stable finite-NB2 arithmetic was validated against high-precision derivatives;
  the bounded range extension now permits the improving coefficient step.
- The outward gate passed mathematical review after a saved-pressure coverage
  regression was fixed. Fixed zero and cap penalties remain supported.
- The original GPD cap fit still changes its shape parameter materially over the
  last window. The wider-cap negative control still fails sustained parameter/coordinate
  stability. Existing parameter tolerances and strict Newton controls remain.

## Reference and reporting contract

[mgcv controls](https://stat.ethz.ch/R-manual/R-devel/library/mgcv/html/gam.control.html)
document practical EFS tolerances, a log-penalty cap, and bounded optimizer
fallbacks. [Wood and Fasiolo](https://arxiv.org/html/1606.04802) motivate bounded
updates and full objective acceptance. These are design evidence, not imported
implementation code.

Practical convergence is a useful accepted fit with observed insensitivity under
the recorded continuation. It does not need to claim exact stationarity, exact
infinite-penalty inference, or a rigorous derivative-error enclosure. Existing
`practical_plateau` already distinguishes this from strict certification.

Memory follow-through is not part of this stopping change. The earlier limits
refer to per-observation derivative/history arrays and dense coefficient-space
tables, respectively; neither is a reason to reject a practically settled fit.

## Implementation checkpoints

- `8b66828d`: retain accepted fits and resume EFS when Newton gradients are unavailable.
- `5816abd1`: practical outward-window policy and replay evidence.
- `5f994c8f`: bounded finite-NB2 low-mean extension, with 168 focused checks passing.
- Final source `5f994c8f` is frozen in `.worktrees/c3-pragmatic-final`.
  The same 610,212-policy NB2 workload reaches configured stationarity in nine
  outer iterations, with a settled inner mode and independently checked covariance.
- The final complete suite passed 11,546 tests with 174 skips, mandatory real-data
  availability and mpmath. Ruff, formatting, lock/dependency checks and the smoke
  script passed. Production source remained unchanged afterward.
- Six fresh serial timing workers compared released dense and final discrete
  severity fits. Independent review supports local observed reductions of 18.0%
  in median complete-fit time and 32.4% in process high-water RSS, with
  background activity recorded. This is not quiet-machine
  approval, a pure backend comparison or a universal speed guarantee.
- Public guidance, roadmap, numerical receipts and the
  [follow-through report](../../research/2026-09-pragmatic-convergence.md)
  record final results and remaining limits. Original strategy files are preserved.
