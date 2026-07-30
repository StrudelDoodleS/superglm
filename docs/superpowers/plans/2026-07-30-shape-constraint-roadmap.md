# Shape constraints — the efficient path

**Design note:** `docs/superpowers/specs/2026-07-30-shape-constraint-strategy.md`
**Status:** DRAFT — not approved, nothing started

Ordered by *defect severity ÷ effort*, not by architectural appeal. Each item is
independently shippable and independently valuable; stopping after any one leaves
the library better than it is now.

---

## First: merge PR #174

It is green, complete, and its findings are fixed or disclosed. Nothing below
depends on further work there. Keeping it open costs review rounds that have
stopped producing findings — round 9's headline defect provably did not exist.

---

## Item 1 — Disclose or fix QP-path smoothing-parameter selection

**Why first:** it is the only item that changes what a user's *fitted model* is, it
is the gap the literature explicitly names, and the disclosure half costs almost
nothing.

Under `fit_reml()` with automatic λ, a `bs`/`cr` monotone fit strips its constraints
(`model/fit_ops.py:1257-1270`, `reml_setup.py:154-163`), selects λ from an
**unconstrained** fit, then refits constrained. Pya & Wood §4.1 name this exact
procedure "the ad hoc method used with QP".

**1a — Disclose (small).** Document it in `docs/guide/monotone.md` and in
`model.summary()` where the engine is already printed
(`inference/summary.py:467`). A user choosing `bs` over `ps` for a monotone fit is
today making a smoothing-parameter-quality decision without being told.

**1b — Measure (small).** Quantify the gap: fit the same monotone problems under
`ps`/SCOP and `bs`/QP, compare selected λ and predictive error. Pya & Wood's own
simulations suggest the effect is real but modest. If it is negligible in our
corpus, 1c is not worth doing and that is a result.

**1c — Fix (large, only if 1b justifies it).** Constrained λ selection for the QP
path. The literature says gradient-based multi-λ selection is what the active set
breaks, so realistic options are a derivative-free outer search over λ, or steering
users to `ps`. **Not** a small change.

---

## Item 2 — Replace the SCOP stagnation workaround with the published fix

**Why:** we shipped a symptom-level treatment for a named, analysed failure mode
with a published remedy, and the remedy uses machinery we already own.

The `exp(γ) → 0` boundary makes a coefficient unidentifiable — "β is simply very
negative, but the data contain no information on how negative". Our answer was to
accept the fit when the deviance stops moving. Pya & Wood's answer (§3.2) is to
**drop the unidentifiable direction**: SVD the R factor when rank-deficient, apply a
pseudo-inverse with a threshold relative to the largest singular value times
`sqrt(eps)`.

**2a** — Implement rank truncation in the SCOP Newton step, reusing
`solvers/rank.py` (which already has exactly this policy, at exactly this threshold).
**2b** — Strengthen the acceptance gate from "hit `max_iter` with no rejections" to
the published gradient-norm test on the penalized deviance — certifying a stationary
point rather than inferring one. Keep deviance stagnation as the primary criterion;
the reference method does, and abandons the coefficient-step test exactly as we did.
**2c** — Then reassess whether the stagnation acceptance rule is still needed at all.

Expect this to *simplify* `reml/scop_efs.py`. It also removes the Python-3.10-only
`converged is True` / `termination_reason == "max_iter"` disclosure currently in the
PR body.

---

## Item 3 — Require full column rank at the QP boundary

**Why:** it reverts the premise that generated review rounds 6–9, and it converts
silent infeasibility into a loud refusal.

Established constrained-least-squares practice requires the design to be of full
column rank, at least when projected into the null space of any equality
constraints, and refuses rank-deficient input by design. Audit item 3 read our
equivalent `LinAlgError` as a robustness gap and removed it; four downstream
components then met inputs they were never written for.

**3a** — Validate rank at `solve_constrained_qp`'s entry and raise with a message
naming the likely cause (collinear columns, a near-empty categorical level).
**3b** — Retire the machinery that existed only to tolerate rank deficiency, guided
by what 3a makes unreachable.
**3c** — Re-run the adversarial ensemble; the disclosed 165 infeasible answers
should become refusals.

**Blocked on a decision:** this is a behaviour change for anyone currently getting an
answer where they would now get an error. Defensible — a loud refusal beats a silent
constraint violation — but it is a minor release, not a patch.

---

## Item 4 — State the REML-for-SCOP position

**Why:** cheap, and it is a governance exposure rather than a technical one.

The reference method has no REML; it optimizes GCV/UBRE, and the 2024 EFS extension deliberately
targets GCV/UBRE too. Our SCOP-under-REML has no published basis and no reference
implementation to check against. Record that in
`docs/governance/model_risk_pack.md` as an explicit position — that the method is an
extension beyond Pya & Wood, with whatever validation we have standing behind it.

Not a defect. But "we follow Pya & Wood" is currently doing work it cannot support.

---

## Item 5 — Softplus for the SCOP transform (optional)

The reference method offers an opt-in softplus in place of `exp`. It addresses overflow, not
identifiability — a coefficient wanting to be zero still runs to `−∞`. Worth doing
only after Item 2, and only if overflow shows up in practice.

**If adopted, mind the delta-method Jacobian:** with softplus active the correct
Jacobian is the logistic function, not `exp(β)`.

---

## Explicitly not doing

- **Retiring the QP.** Pya & Wood specify one. Our two SCOP
  initialization sites mirror the reference implementation exactly.
- **Migrating `bs`/`cr` to SCOP.** Measured: `bs` collapses bitwise onto `ps`
  because SCOP discards the integrated-derivative penalty that defines it, and `cr`
  loses its natural boundary conditions because SCOP bypasses `_apply_constraints`.
- **Implementing Boland (1997).** The right algorithm for rank-deficient QP, but a
  from-scratch implementation of a 29-year-old paper with no usable Python reference
  (`quadprog` is positive-definite only). Item 3 removes the need.
- **Further numerical patching of `constrained_qp.py`.** Every fix in rounds 6–9 was
  correct; none addressed why the failures kept arriving.

---

## Item 6 — Constrained-term coverage: reference note, not a backlog

Surveyed 2026-07-31 from published package documentation. The reference
implementation exposes roughly forty constrained bases. **We are not chasing
them, and the count overstates the gap**: they are combinatorial rather than
forty algorithms — near enough `{increasing, decreasing} × {convex, concave} ×
{univariate, tensor marginals} × {± numeric by}` over one reparameterization.
Each combination is exposed there as a separate named basis; superglm factors
it as `Constraint.fit/postfit × {increasing, decreasing, convex, concave}`,
which is the better factoring and should stay.

The four kinds already cover the shapes a pricing model normally asks for. One
extension is worth keeping in view **only if a real pricing need appears**:

**Constrained tensor terms.** `TensorInteraction` passes `constraint=None` to
its marginals, so a monotone marginal inside an interaction is unavailable.
This is the only gap that would need new machinery rather than a composition,
and Pya Arnqvist (2024, arXiv:2403.09438) describes the method. Do not start it
speculatively.

Deliberately not pursuing: combined monotonicity-and-curvature (`micx`, `micv`,
`mdcx`, `mdcv` — a conjunction we could compose cheaply if ever asked), numeric
`by`-variable constraints (`*By`), and the specialty tail (`mifo`/`miso`
finish/start at zero, `po`/`ipo`/`dpo`/`cpop` positivity and cyclic,
`lmpi`/`lipl` locally monotone with plateau).

**No published implementation offers a shape-constrained factor-smooth basis**, so
"a monotone curve per factor level" is not a gap against the reference
implementation — nobody offers it.

### The RE/FS/SZ guard is not part of this gap

`fit_ops.py`'s `_reject_structured_fit_constraints` refuses a *fit-time* shape
constraint alongside `RandomEffect` or `FactorSmooth`. It arrived with those
features in `f082e9bd` (#165, 0.15.0) as a deliberate scope boundary, and
`2026-07-27-pr165-release-review-remediation-design.md` §4 states the reason:

> "Fit-time constrained REML is not currently defined for identity, repeated,
> or sum-to-zero penalty components."

RE/FS/SZ carry compact structured penalties; defining constrained REML over
them would mean expanding to dense `Kk × Kk` penalties and losing exactly the
compactness that makes those terms affordable. **Post-fit constraints are
supported** with RE/FS/SZ via `penalty_component_quadratic()`, which is what
the error message points at, so this is narrower than "cannot be combined".

The published reference is more permissive — its documentation states that
unconstrained smooth terms may be added alongside constrained ones — so the
restriction is superglm's own, not inherited. Lifting it is a separate piece of
work from Item 6 and would need constrained REML defined over compact penalty
components first. Note also that relaxing it makes the `O(p³)` projection in
`restrict_to_scop_resolved_range` live on wide models; it is currently unreachable
because a constrained model cannot carry a wide factor-smooth term.

Sources: published package documentation for the reference implementations;
Pya & Wood (2015); Pya Arnqvist (2024), arXiv:2403.09438.

---

## Suggested sequence

1. Merge #174.
2. **Item 1b** — measure the λ gap. One session. Decides Item 1c.
3. **Item 4** — write the governance position. One session.
4. **Item 2** — the SCOP rank-truncation fix. The highest-value engineering item.
5. **Item 3** — needs a release-scope decision before it can start.
Item 6 is a reference note, not scheduled work.

Items 1b and 4 are cheap and independent; either can go first.

**Status 2026-07-31:** Item 2 is implemented on `fix/scop-rank-truncation`
(rank truncation moved to the factor, resolved range carried through the
determinant and curvature, factor built only when the Gram cannot resolve the
step). Item 2c is now answerable with evidence — the boundary fit converges in
single figures rather than exhausting its budget — but the stagnation gate has
deliberately not been removed; that is a separate change.
