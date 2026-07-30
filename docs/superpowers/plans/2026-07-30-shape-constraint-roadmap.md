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
`scam`'s gradient-norm test on the penalized deviance — certifying a stationary point
rather than inferring one. Keep deviance stagnation as the primary criterion; `scam`
does, and abandoned the coefficient-step test exactly as we did.
**2c** — Then reassess whether the stagnation acceptance rule is still needed at all.

Expect this to *simplify* `reml/scop_efs.py`. It also removes the Python-3.10-only
`converged is True` / `termination_reason == "max_iter"` disclosure currently in the
PR body.

---

## Item 3 — Require full column rank at the QP boundary

**Why:** it reverts the premise that generated review rounds 6–9, and it converts
silent infeasibility into a loud refusal.

mgcv's `pcls` states "X must be of full column rank, at least when projected into
the null space of any equality constraints" — it refuses rank-deficient input by
design. Audit item 3 read our equivalent `LinAlgError` as a robustness gap and
removed it; four downstream components then met inputs they were never written for.

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

`scam` has no REML; it optimizes GCV/UBRE, and the 2024 EFS extension deliberately
targets GCV/UBRE too. Our SCOP-under-REML has no published basis and no reference
implementation to check against. Record that in
`docs/governance/model_risk_pack.md` as an explicit position — that the method is an
extension beyond Pya & Wood, with whatever validation we have standing behind it.

Not a defect. But "we follow Pya & Wood" is currently doing work it cannot support.

---

## Item 5 — Softplus for the SCOP transform (optional)

`scam` ≥ 1.2-17 offers opt-in softplus in place of `exp`. It addresses overflow, not
identifiability — a coefficient wanting to be zero still runs to `−∞`. Worth doing
only after Item 2, and only if overflow shows up in practice.

**If adopted, do not copy scam's covariance bug:** their delta-method Jacobian uses
`exp(β)` unconditionally (`R/scam.r:1519`) even when softplus is active, where the
correct Jacobian is the logistic function.

---

## Explicitly not doing

- **Retiring the QP.** scam ships one; Pya & Wood specify one. Our two SCOP
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

## Suggested sequence

1. Merge #174.
2. **Item 1b** — measure the λ gap. One session. Decides Item 1c.
3. **Item 4** — write the governance position. One session.
4. **Item 2** — the SCOP rank-truncation fix. The highest-value engineering item.
5. **Item 3** — needs a release-scope decision before it can start.

Items 1b and 4 are cheap and independent; either can go first.
