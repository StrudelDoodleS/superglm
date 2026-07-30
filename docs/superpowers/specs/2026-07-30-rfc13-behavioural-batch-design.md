# RFC-13 Behavioural Batch — Design

**Status: IMPLEMENTED** on `fix/rfc13-behavioural-batch` (PR #174). Approved as
a design first; the sections below have since been reconciled against the code
that actually shipped, and record the superseded decisions they replaced so the
reasoning is not lost. Read the code as the authority; read this for the *why*.
**Source:** audit `docs/audit/2026-07-28/architecture-audit.md` §E row 13, §J.4
item 6, Tranche 1 item 4; subsystem findings S3, S13, S16 in
`docs/audit/2026-07-28/subsystems/solvers.md`.
**Base:** `master` at `e8e31f4` (= v0.16.1). Every `file.py:NNN` reference below
is a line number *at that base commit*, not on the branch tip.
**Release impact: patch → 0.16.2.**

Four independent correctness-hygiene items, each reproduced through the public
API by the audit. They share no code except the merit-delta helper's new home,
so they are implemented as four commits in one PR.

---

## Item 1 — `max_iter=0` raises `ValueError`, not `UnboundLocalError`

### Current behaviour

Both outer loops reference loop-body locals unconditionally after the loop:
`_fit_pirls_inner` (`pirls.py:843`, reading `retained`/`dev`/`outer`/
`termination_reason` at 1218 and 1376-1391) and `_fit_irls_direct_once`
(`irls_direct.py:1285`, reading `it`/`retained`/`dev`/`step_rejected`/
`termination_reason` at 2118-2131 and 2483-2499). Reproduced on this tree
through all three public entry paths — `fit`, `fit_reml`, and `fit` with
`selection_penalty` — each raising
`UnboundLocalError: cannot access local variable 'it'`.

### Design

Guard the two public solver entry points:

- `fit_irls_direct` (`irls_direct.py:383`): `max_iter < 1` → `ValueError`
- `fit_pirls` (`pirls.py:1435`): `max_iter_outer < 1` → `ValueError`;
  `max_iter_inner < 1` → `ValueError`

The private `_fit_pirls_inner` / `_fit_irls_direct_once` helpers stay unguarded;
they are only ever reached through the validated entry points.

The condition is `< 1`, not `== 0`: `range(-5)` is empty too and fails
identically. Message form follows the existing `max_halving` precedent at
`irls_state.py:185`: `"max_iter must be at least 1, got 0"`.

`max_iter=1` remains legal — the discrete POI loop depends on it
(`reml/discrete.py:551-578`).

### Tests

Public-API assertions that each of `fit`, `fit_reml`, and the
`selection_penalty` path raises `ValueError` (not `UnboundLocalError`) for
`max_iter=0`, plus direct solver-level tests for negative values and for
`max_iter_inner`.

---

## Item 2 — Port the polarization merit delta to pirls

### Current behaviour

`_irls_trial_is_unsafe` (`irls_state.py:141`) already accepts an injectable
`merit_delta`. `irls_direct` supplies `_stable_penalized_deviance_delta`
(`irls_direct.py:117`, wired at 1815); pirls (`pirls.py:1043`) supplies
nothing and falls through to the raw comparison
`candidate_merit > committed_merit + roundoff` at `irls_state.py:171`.

In an ill-conditioned smooth basis the two penalty quadratics are each
evaluated accurately while their difference loses enough digits to reverse
sign. The audit demonstrated a −1.5e-7 improvement read as +1.7e-5, i.e. a
safe terminal step rejected. This is audit finding S3's named divergence
risk between the two IRLS orchestrations.

### Design

Move `_stable_penalized_deviance_delta` from `irls_direct.py` to
`irls_state.py`, next to the `MeritDelta` alias it satisfies
(`irls_state.py:119`). pirls already imports from `irls_state`; importing from
`irls_direct` would invert the dependency direction. Two importers to update:
`irls_direct.py:1815` and `tests/test_irls_state.py:14`. The symbol is private,
so no compatibility shim.

Generalize it so both penalty contributions are optional:

```python
def _stable_penalized_deviance_delta(
    candidate, committed,
    penalty_matvec=None,      # quadratic S: matrix or matvec; None → term skipped
    nonsmooth_penalty=None,   # callable beta -> float, pre-scaled by the caller
) -> float:
    terms = [float(candidate.deviance), -float(committed.deviance)]
    if penalty_matvec is not None:
        terms.append(<polarized Δβ' S (β_c + β_m), unchanged>)
    if nonsmooth_penalty is not None:
        terms.append(nonsmooth_penalty(candidate.beta))
        terms.append(-nonsmooth_penalty(committed.beta))
    return float(math.fsum(terms))
```

`irls_direct`'s call is unchanged in behaviour. pirls passes:

- `penalty_matvec = S if (has_smooth_penalty and S is not None) else None`
- `nonsmooth_penalty = lambda b: 2.0 * penalty.eval(b, groups)`

The `2.0` lives at the call site because it is pirls's merit convention, not
the helper's.

**Stability scope (decided).** The quadratic gets the polarization identity;
the group-lasso term is entered as two pre-scaled values inside the same
`math.fsum`. Per-group differencing of `penalty.eval(beta, [g])` was
considered and rejected: it costs `2·|groups|` Python calls per trial, relies
on `eval` being group-separable, and the demonstrated failure mode is the
quadratic. The residual exposure — `P(β_c) − P(β_m)` differenced at roughly
`eps·|P|` absolute — is recorded here as a known limit, not a defect.

**Consistency constraint.** The delta must be the difference of exactly what
`_state_merit` returns, because `_irls_trial_is_unsafe` compares
`delta > roundoff` where `roundoff` is scaled from `_state_merit` magnitudes.
pirls's `penalized_deviance` is `deviance + βʹSβ + 2·penalty.eval(β, groups)`
(`pirls.py:756`); the three term groups reproduce it exactly.

### Behaviour change

This is the user-visible change in the batch. pirls will accept terminal steps
it previously rejected in ill-conditioned smooth bases, so fitted coefficients
can move for those fits.

### Tests

- Unit: a state pair where the naive difference reverses sign and the polarized
  delta does not.
- Contract: for well-conditioned states the polarized delta agrees with the
  naive difference to ~1e-12 relative.
- Contract: `nonsmooth_penalty=None` and `penalty_matvec=None` each reduce to
  the expected subset of terms.
- End-to-end: a pirls fit on an ill-conditioned smooth basis emits no spurious
  `step_rejected`. This answers the open "Verify:" note on audit finding S3.

---

## Item 3 — Pure-H QP solves through the rank policy; `converged` surfaced

### Current behaviour

`solve_constrained_qp` (`constrained_qp.py:57`) issues three raw
`np.linalg.solve(H, g)` calls (lines 96, 100, 117) — audit S16 records these as
the only unguarded dense solves in the subsystem, bypassing the shared rank
policy that every other consumer routes through (`rank.py`, external callers
listed in `subsystems/solvers.md:54`). Singular `H` therefore raises
`LinAlgError` rather than being rank-truncated.

`QPResult.converged` (`constrained_qp.py:35`) already exists and is already set
to `False` on `max_iter` exhaustion (line 199) — but all three call sites
discard it: `irls_direct.py:1625`, `scop.py:171`, `scop.py:286`.

`_project_feasible` (`constrained_qp.py:38`) caps at 100 sweeps and returns a
possibly-still-infeasible point with no signal.

### Design

**Rank policy.** Symmetrize once at entry, decompose that, and solve once:

```python
H_sym = 0.5 * (H + H.T)
try:
    decomposition = decompose_gram(H_sym)
except ValueError as exc:
    raise ValueError(f"solve_constrained_qp requires a usable PSD H: {exc}") from exc
beta_unc = decomposition.solve(g)
```

`H` is `XtWX + S` at `irls_direct.py:1622` (PSD) and `XʹX + λP + 1e-8·I` at
`scop.py:162-163` (PD), so `decompose_gram` with its default
`allow_indefinite=False` is correct; it falls back to the spectral path when
Cholesky fails. The indefinite KKT solve at line 136 keeps its existing `lstsq`
fallback — the audit scopes this item to pure-H solves.

Verified: `RankDecomposition.solve` divides by `column_scale` on both the RHS
and the solution (`rank.py:168,173`), so it returns in original coordinates.
The `column_scale` trap recorded in the RFC-12b disposition note does not apply.

**One decomposition, one solve.** The three raw solves at base (lines 96, 100,
117) are all `H⁻¹g`, so they collapse to a single `decomposition.solve(g)`
above the loop; the in-loop empty-active-set branch becomes
`step = beta_unc - beta`, reusing the *solution vector*, not merely the
factorization. That removes a redundant O(p³) per active-set iteration.

**Symmetrization is consistent, not local to the decomposition.**
`decompose_gram` internally works on the symmetric part, so passing a raw
asymmetric `H` would have left the decomposition minimizing `0.5(H + Hʹ)` while
the KKT block, the stationarity residual `H @ beta - g` and the multiplier test
still used the asymmetric `H` — two different quadratics on the two paths.
`H_sym` is therefore materialized once and used for *all* of them. For an
exactly symmetric `H` — which every in-tree caller builds by construction —
`0.5 * (H + H.T)` is bitwise identity, so this costs nothing in behaviour.

**Rank truncation is not licence to answer an unanswerable question.**
`decomposition.solve` is a pseudo-inverse, so it returns `H⁺g` even when the
normal equations are inconsistent. When `rank < width` and `g` has a component
outside `range(H)`, the objective is unbounded below along a null direction of
`H` and `H⁺g` is a projection rather than a stationary point. Neither entry
path can follow that descent: the unconstrained solve returns `H⁺g` directly,
and the active-set loop's empty-active-set step is `beta_unc - beta`, which
also lies in `range(H)` — measured to be exactly zero in the reproducer, so the
loop stalls on its own stationarity test and returns the same wrong point.
Returning that as `converged` would be a silent wrong answer, so a `ValueError`
is raised before either early return. The check is skipped at full rank, where
`range(H)` is everything.

*The claim is about the entry paths, not about every direction the method can
ever form.* Once the active set is **non-empty** the KKT block `[[H, -A_eqᵀ],
[A_eq, 0]]` can be nonsingular even with singular `H`, and it then does produce
a null direction. Measured: on `H = diag(1,0)`, `g = (1,1)`, `x2 <= 1`,
`x1 >= 3` (true optimum `[3,1]`, objective 0.5), an empty initial active set
returns `[3,0]` at objective 1.5, while `active_set_init=[0]` — the constraint
whose row covers the null direction — reaches `[3,1]` exactly. A warm-started
solve can therefore reach the finite optimum that a cold one cannot; see the
follow-up item below.

*Detection: project `g` onto `null(H)`, do not test the solve residual.* The
first implementation compared the relative normal-equation residual against
`SHARED_RANK_POLICY.factor_rcond` (`√eps ≈ 1.5e-8`). That threshold does not
match the decomposition's: `decompose_gram` truncates at `gram_rcond = eps`, so
it legitimately retains blocks conditioned up to ~`4.5e15`, while a residual is
amplified by exactly that retained condition number. Measured false-positive
rate on **genuinely consistent** systems (`g` built inside the retained span):
**109/120** refused at retained condition `1e9`, **119/120** at `1e10`,
**119/119** at `1e12`. No residual threshold fixes this, because at those
conditions a correct solution's own residual (`5.8e-8`, `5.8e-7`, `5.5e-5`
respectively) already exceeds any bound that would still catch inconsistency.
Consistency is instead `g ⊥ null(H)`, which the decomposition's existing
`null_basis()` measures directly and without amplification. The threshold is
the roundoff floor of that basis, `_NULL_BASIS_ACCURACY_SLACK * eps * retained
condition`, which drives the false-positive rate to **0/120 at all three
conditions** while still catching **120/120** inconsistent systems. Above a
retained condition of ~`1/(slack·eps)` the floor saturates at 1 and the check
fails open — a real resolution limit, and the safe direction to fail.
(The slack factor is a constant of this module, deliberately *not*
`SHARED_RANK_POLICY.certification_band`: that band governs factor
certification and merely holds the same value, so retuning it for its own
purpose must not move this gate.)

*The floor applies to the spectral half of `null(H)` only.* `rank._null_basis`
stacks two things with disjoint row supports: **exact unit vectors** for
structurally zero columns, and **computed spectral directions** `D⁻¹·V_discarded`
whose accuracy decays as `eps · κ_retained`. Keying one floor off the retained
condition and applying it to both let an ill-conditioned retained block
desensitize the exact half. Measured on `blkdiag(K, 0)` where `K` carries a
spectral truncation: at retained conditions `1e10 / 1e11 / 1e13` a structural
inconsistency of `1e-6 / 1e-5 / 1e-3` of `‖g‖` was **accepted as converged**
while the objective fell without bound along that exact unit direction — travel
back toward the silent wrong answer the gate exists to prevent, on input master
raised `LinAlgError` for. `_null_space_mass` therefore returns the two masses
separately; the structural one is tested against a flat `slack · eps` and only
the spectral one against `_consistency_floor`. The split is orthogonal, so the
two are independent.

**Projection failure — `converged` describes the returned point.** This is the
other half of S16's "caps at 100 sweeps without reporting failure", and it is
what makes the flag mean what its name says. `_project_feasible` still returns
only `beta`; the flag is computed at the two in-loop returns from
`_is_feasible(A, beta, b, tol)` on the point being returned.

*Rejected first implementation.* `_project_feasible` returned
`(beta, feasible)` and the QP latched that starting-point verdict into
`QPResult.converged`. Measured **27/100** spurious `converged = False` at
`p ∈ {105, 120, 150, 180}` with `A = np.eye(q)` — exactly the shape SCOP's
solver-space `qp_initialize` builds — where the active-set loop had in fact
reached a feasible KKT point with `n_iter < max_iter`. Each sweep of the
projection repairs only the single worst violation, so more than 100 violated
non-negativity constraints exhausts the budget for a perfectly feasible
problem; "mutually infeasible" and "merely more constraints than sweeps" are
indistinguishable there, and the loop routinely recovers from the second. Both
SCOP call sites then warned misleadingly. Pinned by
`TestConvergenceFlag::test_projection_budget_overrun_that_the_loop_repairs_reports_convergence`
in `tests/test_constrained_qp.py`, which asserts the precondition (the
projection really does overrun) before asserting the flag.

*Why the projection's history carries no information about the answer.* The
two in-loop returns fire only once stationarity and dual feasibility already
hold (zero step, all multipliers ≥ `-tol`), which leaves primal feasibility of
the *returned* point as the sole outstanding KKT condition. Testing it there is
therefore both necessary and sufficient, and nothing about the starting point
is relevant.

*Iteration exhaustion stays unconditionally `converged = False`.* The loop
never reached its own stationarity/multiplier test, so no certificate exists to
complete, and consulting feasibility there would report success for a search cut
off mid-flight — every interior point is feasible.

*Feasibility is tested with a relative tolerance.* A step that lands *on* a
constraint reproduces `b_i` only to about `eps · |Aᵢ @ beta|`, so an unscaled
absolute bound made a genuine KKT point with `|Aᵢ @ beta| ≳ 1.5e3` report
`converged = False`. `_feasibility_slack` divides the raw slack by
`max(1, |b_i|, |Aᵢ @ beta|)`.

**This is inert for every in-tree caller, and did not fix the in-tree
symptom.** All three pass `b = 0` (`scop.py:172`, `scop.py:290`, and
`irls_direct.py` via `_spline_constraints.py`), and at `b_i = 0` the relative
test is algebraically identical to the absolute one for any `tol ∈ (0,1)`.
Measured across 7 monotone/convex/SCOP fits spanning Gaussian, Binomial and
Poisson: **0 nonzero `b` entries in 189 constraint rows, max `|Aᵢ @ beta|` =
0.72, max per-row scale exactly 1.0.** What removed the SCOP spurious warnings
was the returned-point change above, not the rescaling. The rescaling is
retained because it is correct for external callers with nonzero `b`.

*The loop body uses the same measure as the boundaries — for the gate only.*
`_is_feasible` governs the projection and both `converged` flags, but the
loop's own full-step test and its blocking gate were left absolute. Wherever
the scaling is observable the two then disagree by construction: `_is_feasible`
accepts a row the loop calls violated, so the loop blocks on it with a negative
slack and `alpha_min < 0` — a backward step. Both gates now go through
`_feasibility_scale`.

**The deferred negative-`alpha` population therefore *shrank*, not grew.** An
earlier draft of this note said the reverse. Rows already violated by more than
1.0 with a tiny negative directional derivative used to pass the absolute gate
and step backward; they are now skipped, and where every violated row is
skipped the loop accepts `beta_new` and terminates `converged=False` rather
than stepping backward. Both behaviours stay inside the
`_project_feasible`-overrun population the follow-up already scopes, so the
population *moved* rather than needing new scoping.

*The ratio stays raw.* Scaling numerator and denominator by the same row factor
is algebraically neutral and **not** neutral in floating point: it rounds twice
where the raw quotient rounds once, and on a row with slack above 1 the scaled
numerator collapses to exactly `1.0`. An earlier draft scaled both and claimed
neutrality. Measured, that shifted `alpha` by an ulp — and in an active-set
search an ulp flips which row is "blocking" and reroutes the whole path, which
is how a fixture that took 13 iterations appeared to take 6 and then exhausted
`max_iter` on CI's BLAS. With the raw ratio restored, routing is **bitwise
inert on the nonzero-`b` population (0/120)**; the only cases that differ are
`b = 0` problems with coefficients above 1, all six of which are
`max_iter`-exhausted solves reaching a *lower* objective. In-tree the scale is
exactly 1, so the scaled quantities are bitwise the raw ones and fitted values
are unchanged. Clamping `alpha ≥ 0` — the pre-existing half — stays deferred.

*The routing is pinned by a decision trace, not by a numeric outcome.*
Routing has no BLAS-robust end-to-end signature: with the raw ratio it is
bitwise inert on every nonzero-`b` probe, and the only differing population is
`b = 0` exhausted solves whose paths are chaotic. `solve_constrained_qp`
therefore takes `_trace_run: TraceRun | None = None` — the house `_fit_trace`
seam, reusing the existing `step_decision` event kind — and emits one event per
blocking decision on the `constrained_qp_blocking` channel. Underscore-prefixed
and keyword-only because the function is re-exported from `superglm.solvers`
and this is not public API, and no field is added to `QPResult`.

The hook **re-derives** everything it records — the per-row scale, the scaled
derivative, the considered set, and whether the convergence test accepts the
full step — instead of reusing the loop's arrays or echoing its booleans. A
hook fed the loop's own gating arithmetic would agree with it by construction
and could never witness the gate being wrong. That is what makes the three
tests bite: each fails under a targeted revert and passes otherwise.

| Mutation | Test that fails |
|---|---|
| full-step gate reverted to absolute | `..._no_step_the_convergence_test_accepts_reaches_the_blocking_search` |
| blocking gate reverted to absolute | `..._the_blocked_row_always_passes_the_scaled_gate` |
| both (= pre-routing) | both of the above |
| ratio doubly rounded | `..._recorded_alpha_is_the_raw_quotient_for_the_blocking_row` |

Default off and bitwise inert: `tracing` is resolved once before the loop, the
payload is built through `emit_lazy` so it is never constructed when disabled,
and fitted values plus 60 direct solves are byte-identical to the parent.

**Surfacing.** Each of the three call sites checks `result.converged` and emits
`logger.warning` naming its context. `scop.py` has no module logger and gains
one. No public API change.

*Correction.* This paragraph originally justified the warning as "matching the
existing precedent at `irls_direct.py:1648` and `pirls.py:1225`". **Both halves
of that were false against the tree**, and the error propagated into the
implementation: the first is now `irls_direct.py:1623` and is an `== 3`
*equality latch* that fires exactly once, not on every qualifying iteration;
the second is now `pirls.py:1240` and is `logger.info`, not
`logger.warning`. The convention the file
actually follows is fire-once, which is what the latch below implements.

Two refinements the first draft did not have:

- **The `irls_direct` warning is latched to one per fit** (`_warned_qp_nonconvergence`,
  set alongside the existing SVD-fallback latch). QP non-convergence normally
  persists for the remainder of the fit, so the unlatched version emitted one
  identical line per IRLS iteration. The message names the iteration at which
  the condition was first seen and states that later iterations are not
  reported.
- **The two SCOP sites name which of them fired** — "SCOP raw-space QP
  initialization" (`SCOPReparameterization.qp_initialize`) versus "SCOP
  solver-space QP initialization" (`SCOPSolverReparam.qp_initialize`). They
  previously shared one message, so a warning could not be traced back to a
  call site.

### Behaviour change

- A *consistent* singular `H` moves from raising `LinAlgError` to returning a
  rank-truncated solve, consistent with the rest of the solver stack.
- Three inputs now raise `ValueError` rather than `LinAlgError` or a plausible
  wrong answer: a materially indefinite `H`; a rank-deficient `H` whose `g` has
  a component outside `range(H)`; and an `H` the rank policy cannot
  equilibrate. The middle case is the one that changed during review — it
  previously returned `H⁺g` with `converged=True`.
- An asymmetric `H` is now solved consistently as its symmetric part on every
  path, instead of as two different quadratics.
- `QPResult.converged` becomes meaningful, so the three call sites can emit
  diagnostics that a currently degraded-but-working fit did not previously get.

### Follow-ups (not this PR)

- **Support the constrained-but-bounded inconsistent case.** When `H` is
  rank-deficient, `g ∉ range(H)`, but the constraints block every null
  direction, the problem has a genuine finite optimum that this item now
  refuses rather than computes. The measurement above shows the machinery is
  closer than it looks: a **non-empty** active set whose rows cover the null
  direction already produces the right answer through the existing KKT block
  (`active_set_init=[0]` reaches `[3,1]` exactly). What is missing is a way to
  reach that active set from a cold start, i.e. seeding the active set from the
  constraints that bound `null(H)` rather than relying on the ratio test to
  discover them. That is a feature, not a bug fix, and not audit S16.
- **The stationarity test is absolute.** `np.linalg.norm(step) < tol` uses a
  fixed `tol` on a quantity whose natural scale is `‖beta‖`. Everything else in
  the function went scale-aware; this did not. It is the reason a fixture with
  `‖beta‖ ≈ 8e3` terminated in 13 iterations on one BLAS and exhausted
  `max_iter` on another with a *stable* active set `[3, 2, 0]` — not cycling,
  just a step whose norm sits either side of `1e-12` depending on rounding.
  Pre-existing; not fixed here.
- **Active-set cycling has no anti-cycling rule.** Not observed (the trace
  above shows monotone growth to a stable set, 4 distinct sets over the run),
  but there is nothing preventing it. Filed for completeness.
- **The negative-`alpha` ratio test.** `alpha` can go negative when the current
  iterate already violates constraint `i` (`slack < 0` with `a_step < -tol`),
  stepping backwards. Reachable only from an infeasible start. Confirmed
  pre-existing and byte-identical to `e8e31f4`, originating in `234adee`.
- **The outer convergence rule** at `pirls.py:1086` / `irls_direct.py:1857`
  still differences raw penalized deviances. Pre-existing, shared with
  `irls_direct`, not a regression.

### Tests

`tests/test_constrained_qp.py`:

- `TestRankDeficientHessian` — singular `H` returns a finite solution in the
  retained subspace instead of raising; a binding constraint still reaches the
  optimum; the indefinite-`H` error names this function; a well-conditioned
  solution is unchanged by the rank policy.
- `TestConvergenceFlag` — a `max_iter`-starved QP reports `converged=False`
  *even though its truncated point happens to be feasible*; a mutually
  infeasible constraint system reports `converged=False` with
  `n_iter < max_iter`; a feasible solve still reports `converged=True`; and a
  projection-budget overrun that the loop repairs reports `converged=True`
  (the regression that the rejected starting-point design produced).
- `TestCallSiteWarnings` — `caplog` assertions at each of the three call sites,
  plus one that the `irls_direct` warning is latched to one per fit.
- `TestInconsistentNormalEquations` — unconstrained and constrained
  inconsistent systems raise; the error names the rank and the null mass; a
  *consistent* rank-deficient system still solves; ridge regularization is a
  workable escape. Two parametrized tests pin the threshold boundary at
  retained conditions `1e9 / 1e10 / 1e12`: consistent systems must all solve
  (this fails under either a residual gate or a fixed `factor_rcond`
  threshold), and inconsistent ones must all still raise (this fails if the
  floor is loosened to 1).
- `TestSymmetrization` — an asymmetric `H` minimizes its symmetric part; a
  symmetric input is untouched.
- `TestFeasibilityToleranceScaling` — badly scaled constraints do not report
  spurious non-convergence, the scaling does not mask a real violation, and the
  projection and the convergence test share one feasibility rule.
- `TestStructuralAliasConsistency` — at retained conditions `1e10 / 1e11 /
  1e13`, a structural inconsistency below the spectral floor still raises (each
  case asserts the spectral floor *would* have missed it, so a shared floor
  cannot pass the test), and a consistent system with a roundoff-scale
  structural component still solves.
- `TestLoopFeasibilityRouting` — the loop no longer blocks on rows the
  convergence test considers satisfied (same optimum in 6 iterations where the
  absolute test took 13), and the `b = 0` shape every in-tree caller uses is
  pinned as the exactly-inert case.

---

## Item 4 — Warn instead of silently dropping the W(ρ) correction

### Current behaviour

`compute_dW_deta` (`w_derivatives.py:50`) returns `None` when the link lacks
`deriv2_inverse` or the distribution lacks `variance_derivative` (line 72).
`reml_w_correction` then returns `None` at line 267 with the comment "Custom
link/distribution w/o 2nd-order" — the REML gradient and Hessian silently lose
the weight-derivative term.

All eleven built-in links implement `deriv2_inverse`
(`subsystems/families-profiling.md:20`), so only user-supplied custom links
reach this. The observed path already fails loudly for the same capability gap
via `validate_observed_derivative_capability`
(`observed_geometry.py:452-489`); the Fisher path has no equivalent.

### Design

A private helper in `w_derivatives.py` emits a `UserWarning` naming the
module-qualified link/distribution pair, the missing method(s), and the
consequence.

**Call site: `reml_w_correction` at the `dW_deta is None` branch (line 267) —
deliberately not inside `compute_dW_deta`.** `compute_dW_deta` has a second
public entry point, `model_compute_dW_deta` (`model/reml_ops.py:15`, surfaced
as `Model._compute_dW_deta` and re-exported from `reml/__init__.py`). That is a
bare derivative query making no REML claim, so a warning about skipped
smoothing-parameter gradients does not belong on it. (The finite-difference
fallback is *not* an exposure: `_compute_d2W_deta2_fd` is reached only from the
`w_correction_order >= 2` branch of `reml_w_correction`, which sits after the
`dW_deta is None` gate, so its three internal `compute_dW_deta` calls can never
hit the capability gate.)

The structural-zero branch at line 270 (`not np.any(dW_deta)` — Gamma/log,
where the correction is genuinely zero rather than unavailable) must stay
silent. The two branches are already distinct.

**Per-iteration spam** is handled by the stdlib default warning filter, which
dedups on `(message, category, module, lineno)`. Because `stacklevel=3` keys
the registry on the *caller's* frame, and `reml_w_correction` has two call
sites (`reml/direct.py:599` and `model/reml_ops.py:30`), including the link and
distribution class names in the message bounds this at **at most two** warnings
per unique class pair per process — one per call site. Measured: five REML
iterations from a single call site emit one warning. `pytest.warns` still
observes it because pytest resets filters inside its context.
`pyproject.toml:127-129` filters only the `bs` `FutureWarning`, so nothing
interferes.

The prefix that carries this must be **an unconditional pair** of
**module-qualified** names — `f"{cls.__module__}.{cls.__qualname__}"` for each
side, via the `_qualified_class_name` helper. Both properties are load-bearing
and each was a real defect before review caught it:

- **Unconditional pair** (round 2). The message was assembled only from
  whichever class was missing a method, so when just the link lacked
  `deriv2_inverse` the distribution was never named. The same custom link used
  with Poisson and then with Gamma produced byte-identical text and the filter
  suppressed the second.
- **Module-qualified, not bare `__name__`** (round 3). Two distinct classes both
  called `MyLink`, in different modules, rendered identically and collided the
  same way. `__qualname__` rather than `__name__` additionally keeps nested and
  locally defined classes distinct from a module-level class of the same name.

In both cases the result was one warning where the design promises two, with
the second degraded pair going unreported. The qualified pair is also the more
useful report: the user needs to know which *combination* was degraded, and for
a custom class the module is exactly what they need to locate it. The
missing-method clause stays unqualified — repeating the full path a few words
later buys no distinctness and no information.

Raising, as the observed path does, was considered and rejected: it would
remove a currently-working if degraded path for custom links.

### Tests

- A custom link without `deriv2_inverse` warns under `pytest.warns(UserWarning)`
  and the message names `deriv2_inverse`.
- A custom distribution without `variance_derivative` warns and names *that*
  method.
- Gamma/log does **not** warn (structural zero, line 270).
- A built-in link does not warn.
- Dedup granularity, on all three axes, with every variant raised from a
  *single* call site so the filter key differs only in message text:
  - Two different missing methods → two warnings, not one
    (`test_warns_once_per_class_pair_not_once_per_iteration`), and repeats of
    each collapse to one, standing in for per-iteration spam.
  - One custom link, two counterpart distributions → two warnings, not one
    (`test_one_custom_link_still_warns_for_each_counterpart`). The case round-2
    review found broken; it fails with `assert 1 == 2` against a message that
    names only the missing method.
  - Two same-named classes from different modules → two warnings, not one
    (`test_same_class_name_in_different_modules_warns_twice`). The case round-3
    review found broken; it fails with `assert 1 == 2` against a message that
    names classes by bare `__name__`. The two types are built by a helper that
    sets `__module__` and `__qualname__` explicitly — two `class` statements in
    one test file share `__module__` and so could not express the collision —
    and the test asserts the collision holds (same `__name__`, same
    `__qualname__`, different classes) before relying on it.

---

## Packaging

One PR, four commits (one per item) plus a version-bump commit.

**release:patch → 0.16.2**, declared in the PR body per AGENTS.md, with the
exact next version bumped in the same PR via `scripts/bump_version.py` +
`uv lock`. Rationale for patch rather than none: item 2 can move fitted
coefficients, item 3 changes singular-H QP from raising to solving, and item 4
adds a warning. No API surface changes.

Review follows the dual-bot flow: push, open PR, comment `@claude please review`
and `@codex please review`, verify comment URLs, expect 3-4 rounds, and verify
every finding against the code before implementing it.

**Gitignore trap:** `docs/superpowers/` is ignored but tracked by convention.
This file needs `git add -f` and a `git log --stat` check.
