# Splitting `_fit_irls_direct_once`

> **Updated 2026-10-25 for #527 and #530** (master 5d295301; S0, branch
> `claude/irls-direct-split`). What changed underneath this design:
>
> - **#527 deleted block-coordinate SCOP.** The `_scop_joint` switch, the block
>   Gauss-Seidel step and its theta-only line search are gone, so SCOP has one form:
>   one joint Newton step per iteration (`SCOPJointNewton.step`, Pya & Wood 2015).
>   The "SCOP block-coordinate" harness mode and the "`_scop_joint` is a per-fit
>   parameter" bullet below are struck, and the anatomy's correction that the split
>   deletes that path is already done.
> - **SCOP now runs on observed-Newton rows** wherever the family and link support
>   them. `_observed_newton_active` no longer excludes SCOP, though
>   `_stagnation_newton_available` and `_can_reuse_weighted_gram` still do, so the
>   `CurvaturePolicy` collaborator covers SCOP too. The SCOP line search reads the
>   observed-row finiteness predicate (`observed_rows_invalid`), but it still has no
>   mean-space or link-domain guard.
> - **The SCOP step and search gained state the split must carry:**
>   - a damped trial re-solves the ordinary block at its own latent coordinates
>     (`_SCOPOrdinarySolve`, variable projection);
>   - a latent step is first shortened to `_LATENT_MAX_STEP` (`_latent_step_depth`);
>   - a damped SCOP step never stops the fit, under the coefficient or objective rule;
>   - NB2's unit-deviance rounding enters the SCOP merit allowance;
>   - trial predictors are centred and carry the `(level, centre)` pair
>     (`_SCOPTrialState`, `_level_about`).
> - **SCOP finalisation is cheaper.** It no longer runs a whole Newton step for the
>   retained Hessian: `SCOPJointNewton.block_curvature` takes no step. The anatomy's
>   F05 risk is retired.
> - **Inference.** A SCOP direction held at its log-space boundary gets no
>   covariance or EDF. That lives in `reml/scop_geometry.py`, outside this function.
> - **#530** puts the rounding constants in `superglm._roundoff` (`merit_gamma` is
>   `_gamma(...)`). `superglm._lambda_box` does not touch this function.
> - **Size at 5d295301:** 3,999 lines, 47 parameters and 360 locals. Comments and
>   docstrings are 19%. Measured counts and the block map are in
>   `2026-10-25-irls-direct-state-map.md`, which supersedes the line numbers here
>   and in the anatomy.
> - **S0 is done:** the harness (`benchmarks/irls_direct_characterisation.py`), the
>   solver-entry refusal and the state map.

Status: draft for the owner, 2026-10-23. Part of the 0.42 audit (no new features).

## Why

`superglm.solvers.irls_direct._fit_irls_direct_once` is the penalised IRLS fit behind every
direct-solver `fit` and `fit_reml`. It is 4,013 lines in one scope, measured at v0.41.0
(7b1732b7):

- 48 parameters, about 460 local names, 265 `if`s, nesting depth 8, 26 nested functions (1,035
  lines; the largest is the 366-line `true_mode_residual`).
- Setup takes 1,553 lines, the iteration loop 1,427, and finalisation 758.
- 35 PRs and 68 commits touched it; 1,655 of its lines were written in the last 30 days. PR #437
  (Binomial/log) alone added 650.
- It is not sloppy. 18% of it is comments, only 4% never runs in the suite, and only about 5% is
  copy-paste. It is long because four solvers share one loop and one set of variables:

| Inside `if` blocks on | Lines |
|---|---|
| `_has_scop` (shape-constrained Newton) | 1,336 |
| `_use_structured` (random-effect / factor-smooth engine) | 609 |
| `_use_qr` | 548 |
| `has_constraints` (linear-constraint QP) | 312 |

So every change has to reason about every mode at once, and a fix for one mode lands next to the
others' code. The owner: "Splitting that is a target." On #437: "I would and should have rejected
that approach. I just wanted it fixed NOW."

## Goals

1. The same fit, written so that each mode's code lives in its own unit, with a small explicit
   interface between the shared loop and the mode.
2. A change to one mode can be read and reviewed without reading the others.
3. A place to put the standard method for Binomial/log in place of #437's special cases (see the
   last section).

## Non-goals

- No new features, options or behaviour in the split itself.
- The REML optimisers (`optimize_direct_reml` 1,864 lines, `optimize_discrete_reml_cached_w`
  1,649, `optimize_scop_efs_reml` 1,012) have the same shape, but they are a later design, not this
  one.
- No change to `fit_irls_direct`'s signature or its callers in this design's first PR.

## Anti-goals (a failure even if it "works")

- Any loss of fit time or performance. The owner: "NO loss of time to fit or performance."
- Any change to any fitted number in the split PR. Coefficients, deviance, EDF, iteration counts
  and trace records must all stay bit-identical.
- Extracted functions with 30 to 40 arguments and tuple returns. The measured block boundaries are
  26 to 41 reads and up to 32 writes each, so a plain extract-function would do exactly that.
- A pile of small PRs. The owner dislikes PR spam: this is one PR for the split and one for the
  Binomial/log replacement.

## Decided approach

**Replace Function with Command** (Fowler, *Refactoring*, 2nd ed.; also called Method Object). The
fit becomes an object whose variables are fields, so its blocks can become methods without
argument lists. On top of that:

- **One step strategy per fit, chosen once at setup.** The modes are mutually exclusive in a
  single fit:
  - `_use_qr` requires no constraints and no SCOP.
  - The structured selector (`_structured/selection.py`) rejects any group with constraints or a
    SCOP reparameterisation, so SCOP's precedence over structured in the step never decides
    anything.
  - SCOP and linear-constraint (QP) groups never meet: every fit entry point refuses the mix
    (`tests/test_monotone_fit.py::test_scop_plus_qp_raises`). S0 adds the same refusal at the
    solver's entry, so the split can rely on it rather than on its callers.
  - ~~`_scop_joint` is a per-fit parameter, so joint and block-coordinate SCOP are two settings of
    one strategy.~~ (#527 deleted block-coordinate SCOP: SCOP has one form.)

  So a fit has exactly one of `QRStep`, `SCOPStep`, `StructuredStep` or `DenseStep`. The
  linear-constraint QP belongs to `DenseStep`. Each strategy owns the code that now sits inside
  its `if` blocks in setup, step, line search and finalisation.
- **#437's mean-space Newton is an overlay, not a mode.** It is available only without QR, SCOP
  or `_return_working_system`, it switches on mid-fit when Fisher scoring stalls
  (`_mean_space_newton`), and from then on it changes the Gram step and the line search of the
  dense and structured strategies. It becomes its own collaborator, with an `engaged` state the
  loop switches on, which those two strategies consult. Keeping it separate is also what lets the
  second PR replace it whole.
- **The feasibility contract is shared.** The committed-state check, the trial check and the
  finalisation refusal for linear constraints run whatever the step. They stay in the shared loop
  and finalisation, calling `DenseStep`'s constraint data, so moving the step cannot drop them.
- **The shared loop stays one short method**: step, then trial or line search, then convergence,
  then record. It calls the strategy through a few methods that the builder names from the state
  map (S0).
- **Instrumentation becomes a recorder.** Trace, extrema capture, per-iteration diagnostics and
  debug rows read 41 names and write 1, so they are the cleanest cut.
- **Convergence becomes its own unit.** It covers the deviance, coefficient and mode-score
  criteria and the score centre.
- **Module layout:** `superglm/solvers/irls_direct/` becomes a package (`__init__` re-exports
  today's names). The builder picks the split, but roughly: the fit object, one module per step
  strategy, the line search, convergence, the recorder, and finalisation.
- **Hot-path rule.** A Python attribute read costs more than a local one. Inside the iteration
  loop and the line search, a method binds the fields it reads repeatedly to locals first. No new
  allocation per iteration, no extra copies of arrays, and no change to the order of any
  floating-point operation.

## Corrections from the anatomy (read `anatomy/report.md`, which overrides this section where they differ)

The read-only Opus anatomy (2026-10-23, at f76a7265) found these. The builder follows §5 of the
report for the fit object's fields, collaborators, the strategy interface and the order of moves.

- **A `CurvaturePolicy` collaborator** owns Fisher versus observed curvature. It covers the
  Tweedie/log observed Newton from the start, the Gamma/log Fisher-to-Newton switch with its
  REML latch, and the rows L02 forms. Convergence switches curvature mid-fit, so the switch calls
  the policy (and `disable_weighted_gram_reuse()` on the builder). It must not rebind a name that
  other code captures.
- **The mean-space overlay has two halves.** The Newton half acts on dense, QP and structured. The
  true-score certificate half acts on every route, QR and SCOP included, through
  `certificate_inputs()`: held rank, active rows and SCOP groups.
- **`CenteredSystemBuilder` is shared** by QR, dense, QP and the centred finaliser. So there are
  two finalisers (centred, used by QR, dense, QP and SCOP; and structured), not four.
- **`deviance_relative_change` stays in the loop.** F02's separation backstop reads it, so it is
  not the recorder's.
- **Keep the convergence precedence exactly.** The L16 criterion, then the step-rejected veto,
  then the mean-space judge, then the QP feasibility/KKT veto, then the reason chain. After the
  loop: F03, then F04, then the truncated relabel.
- **S1 hazards:**
  - `retained_rank_info` and `certify_centered_factor` must not become bound methods. A deferred
    bound method would keep the whole fit object alive, which #492 deliberately avoided.
  - Hot-path local binding must never cache `_score_centre`, `_observed_newton_active`,
    `_can_reuse_weighted_gram` or `_state_center` past the points where they are rebound.
  - State objects must keep their identity, because `scop_merit_errors` is keyed by `id()`.
- ~~**Block-coordinate SCOP (`_scop_joint=False`) has no production caller;** only three tests use
  it. The split deletes that path and its tests, so SCOP has one form.~~ Done by #527.
- **The Binomial/log replacement has two gaps the design missed.**
  - The constraint QP exists only on the dense route: the structured selector refuses constrained
    groups, and QR has no QP. Binomial/log random-effect fits are structured.
  - The QP works on centred slopes with the intercept profiled out, so an event-row bound
    `η_i ≤ 0` has to be rewritten through `α = mean_z − mean_x'β` and moves with W.

  The second PR's design must solve both before anything is built.

## Acceptance gates (every stage)

1. **Bit-identical on the same machine and threads.** The S0 characterisation harness (below)
   compares, against master and bitwise:
   - `beta`, the intercept (and its compensated remainder), `eta`, `mu` and the final weights;
   - deviance, EDF (total and per group) and scale;
   - iteration, line-search and halving counts, the convergence reason, and the active set of the
     constraint QP;
   - every trace and diagnostic record field except wall-clock fields (`*_s`, timestamps) and
     logging text.

   Bitwise is the right test here: a refactor on one machine must not reorder arithmetic. This is
   not a cross-machine load check.
2. **Release differential:** 0 of 1,174 flagged. Run `benchmarks/release_differential.py run`
   in sharded form on master and on the branch, then `compare`, as the 0.41 release did.
3. **Full suite green**, including the slow and real-data suites with `SUPERGLM_REQUIRE_DATA=1`.
4. **Fit time.**
   - Work counts must be identical on the `books.py` standard and family sets.
   - Wall time is checked in alternating master and branch runs, never all of A then all of B:
     the machine drifts by ±20%.
   - No case may be slower beyond its alternation noise. Pinned per-iteration timings come from
     the IRLS profile phases (`irls_*_s`).

## Stages (one branch, one PR for S0 to S5)

- **S0, characterisation harness and state map.** Read-only on the solver.
  - A harness that fits a fixed matrix and pins its outputs bitwise against master. The matrix
    covers every mode (QR, SCOP joint ~~, SCOP block-coordinate~~, structured RE and fs/sz, dense
    with and without the constraint QP) crossed with the families and links the differential covers.
    It includes Binomial/log fits with offsets that engage the mean-space Newton mid-fit, and one
    that does not, so the Fisher-to-Newton transition is pinned.
  - The solver-entry refusal of mixed SCOP and QP groups.
  - A state map: for each block, which fields it reads and writes, and what every closure
    captures. The interfaces in S2 to S4 come from this map, not from guessing.
- **S1, the method object.** Move the function into a class: variables become fields, and nested
  functions become methods. No logic moves. Gates 1 to 4.
- **S2, the recorder out.**
- **S3, step strategies:** one at a time, QR first (smallest), then structured, SCOP and dense.
- **S4, line search and convergence.**
- **S5, finalisation per strategy.** Then delete the original function body, and review the whole
  diff as one change.
  - Done (2026-10-27): two finalisers, each step naming its own. The centred one
    (`_finalise._finalise_centred`) serves QR, dense, QP and SCOP; the structured step's is
    its own. Both return `_IRLSDirectFinalGeometry`, which publication reads, so 13 fields
    left the fit. `_finish` keeps F01 to F04 and the truncated relabel in order, with no
    route flag. What remained of the original body that nothing read is deleted, and eleven
    bitwise cleanups landed, the stop rule's redundant feasibility veto among them, with a
    test of the invariant it rested on. The state map's §6.7, points 38 to 45, is the
    as-built record.
  - Next: coverage tests for the mutants only a probe catches, a review of the whole diff,
    then a rebase onto master and the ports.

## The Binomial/log replacement (second PR, after the split)

A literature sweep (note: `audit042/research/logbinomial_note.md`) names the problem:
log-binomial maximum likelihood, a concave maximisation over the polyhedron `Xβ + o ≤ 0` whose
maximum can lie on the boundary.

- The standard method puts the bound into each Newton step as linear constraints, a sequential QP,
  and certifies a boundary optimum by its KKT multipliers.
- Halving back into the feasible region, R `glm`'s approach and what #437 extended, is
  documented to fail at boundary maxima, at rates up to 59% in simulation.
- Only event rows (y > 0) can be active, and Fisher's working response on an event row always
  points outside the region.

The replacement stacks the event rows' `η_i ≤ 0` into the existing constrained QP solver, runs an
Armijo search on the unclipped penalised log-likelihood, and stops on KKT. It retires #437's
special cases: the clip band for this link, halve-until-feasible, the score-on-unclipped-η
certificate and its weak-slope logic, and `mean_space_boundary`, which becomes a certified result.

- This PR changes results on boundary cases, so the differential will flag them. Each flagged
  case needs a better or equal certified optimum, or the owner's sign-off.
- Open problems:
  - The constraint rows may be many, which clashes with "never materialise matrices".
  - `solve_constrained_qp` refuses a singular Hessian bounded only by constraints, and a level
    made only of event rows is exactly that.
  - #454's 0.99 fraction-to-the-boundary cap caused refusals.
  - SCOP groups are nonlinear in η.
  - REML at a boundary optimum has no literature, so that part is ours.

  This PR gets its own short design once the split has landed.

## Left to the builder

- The exact strategy interface (method names and signatures), from the S0 state map.
- The module boundaries inside the package.
- Which nested functions become methods of the fit object and which move into a strategy.

## Calls made in this design (not yet confirmed by the owner)

- One PR for the split (S0 to S5) and a second for the Binomial/log replacement. The
  replacement goes after the split, so it lands in a contained overlay.
- `fit_irls_direct`'s signature and private switches stay as they are in the split PR.
- The REML optimisers are a separate, later design.

## Risks

- The 26 closures capture variables that later code rebinds. Moving them into methods must keep
  any late-binding behaviour they rely on. The S0 state map lists every capture.
- `fit_irls_direct`'s private switches (`_return_working_system`, `_compute_fit_statistics`,
  `_defer_rank_info` and others) cut across modes. They stay parameters in this PR.
- Bitwise identity can break on harmless-looking moves, such as an `np.sum` over a reordered
  concatenation. Gate 1 catches it at every stage, so stages stay small.
