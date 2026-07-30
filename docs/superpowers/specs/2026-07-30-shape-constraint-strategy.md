# Shape-constrained smoothing: strategy after the RFC-13 QP episode

**Status:** DESIGN NOTE — not yet approved
**Date:** 2026-07-30
**Context:** Nine bot-review rounds on PR #174 hardened `solvers/constrained_qp.py`
without anyone asking whether the solver should exist. This note answers that,
from the primary literature and from the code.

---

## 1. The recommendation this note overturns

Earlier in this investigation I recommended **retiring the QP** in favour of the
SCOP reparameterization, on the reasoning that the reference method avoids
constrained QP entirely and superglm already implements that approach.

**That recommendation was wrong on its central factual claim.** Pya & Wood (2015)
§3.2, step 1, specifies a QP as the *first step of the SCOP algorithm*:

> "To obtain an initial estimate of β, minimize ‖g(y) − Xβ̃‖² + β̃ᵀS_λβ̃ w.r.t. β̃,
> subject to linear inequality constraints ensuring that β̃_j > 0 whenever
> β̃_j = exp(β_j). **This is a standard quadratic programming (QP) problem.**"

The published algorithm therefore requires an active-set QP as a component.
superglm already mirrors this at
`solvers/scop.py:187` and `:302`.

**So the reference implementation of "the reparameterization approach" ships a
constrained active-set QP as a mandatory component.** Retiring the QP is not what
Pya & Wood did and not what we should do.

Two further premises of mine were also false, both corrected by direct measurement:

- **"SCOP requires `discrete=True`."** False. `dm_builder.py:740` sits inside the
  discrete branch and is the *less-travelled* path; the default path has its own
  SCOP branch at `features/_spline_build.py:79-93` with no discrete gate. Engine
  selection tracks the **spline class**, never discretization. Binning appears
  nowhere in the published method.
- **"Grafting SCOP onto `bs`/`cr` is plumbing."** False, and measurably so.
  `BSplineSmooth` + SCOP produces a **bitwise-identical fit to `PSpline`**
  (max difference `0.000e+00` over a 300-point grid) because SCOP substitutes its
  own first-difference penalty and discards the integrated-derivative penalty that
  *defines* `bs`. `CubicRegressionSpline` + SCOP **silently loses its natural
  boundary conditions**, because the SCOP branch early-returns before
  `_apply_constraints` applies the `f''=0` projection.

---

## 2. What the two engines actually are

| spline | user `kind=` | engine | why |
|---|---|---|---|
| `PSpline` | `ps` — **the default** | SCOP | has `_build_scop_reparameterization` |
| `BSplineSmooth` | `bs` | **QP only** | integrated-derivative penalty; no reparameterization |
| `CubicRegressionSpline` | `cr` | **QP only** | value-at-knot basis; natural boundary projection |
| `NaturalSpline` | `ns` | rejected at construction | — |

The capability sets are **disjoint** — no spec has both methods. A user who simply
asks for a monotone spline already gets SCOP. The QP serves `bs` and `cr`, which is
exactly the population its introducing commit named (`234adee`, "QP monotone engine
for bs/cr splines (Phase 2)").

**Both engines support all four shape kinds** (increasing, decreasing, convex,
concave). Neither has a kind the other lacks. The QP's unique capability is not a
*kind* but a *composition*: its constraints are linear inequalities on raw
coefficients, composed through whatever projections the spec already applies — which
is why it can sit on top of CRS's natural-boundary projection. SCOP cannot, because
it *replaces* the parameterization.

---

## 3. The real defect, and it is not numerical

The nine rounds fixed numerical faults in the QP. The substantive quality gap is
elsewhere, and both the code and the literature name it.

**In our code:** under `fit_reml()` with automatic λ, QP constraints are **stripped
entirely** — `model/fit_ops.py:1257-1270` and `reml_setup.py:154-163` set
`monotone_engine = None; constraints = None`, run unconstrained REML, then refit
constrained. **So on the QP path the smoothing parameter is chosen without the
constraint in view.** SCOP gets exact joint constrained REML.

**In the literature:** this is precisely the practice Pya & Wood built SCOP-splines
to replace. Paper §4.1, on their QP comparator:

> "the selection of the smoothing parameter for SCAM is well founded, **in contrast
> to the ad hoc method used with QP of choosing λ from an unconstrained fit, and
> then refitting subject to constraint**."

And §1, on why:

> "The use of linear inequality constraints makes it difficult to optimize standard
> smoothness selection criteria… **The difficulty arises because the derivatives of
> these criteria change discontinuously as constraints enter or leave the set of
> active constraints.**"

**This is the finding that should drive the roadmap.** A `bs`/`cr` monotone fit under
automatic λ is choosing its smoothing parameter by a method the literature calls ad
hoc, and our audit already flagged the surrounding state management as "scattered
across three modules".

Worth keeping in proportion: the authors' own simulations found only "a slight
advantage" for SCAM over QP on fit quality, and attributed even that to λ selection
rather than to the estimator. The QP is not producing bad fits. It is producing fits
whose smoothness was chosen without reference to the constraint.

---

## 4. Two published upgrades we are not using

**4.1 — SVD rank-truncation is the actual remedy for the `exp(β) → 0` boundary.**
Our SCOP convergence saga (a coefficient drifting to `exp(γ) → 0` while the deviance
sits flat to 13 significant figures) is a named, analysed failure mode. Paper §3.2:

> "the non-linear constraints mean that parameters can be poorly identified on flat
> sections of a fitted curve, where β is simply very negative, but the data contain
> no information on how negative… a simpler strategy **substitutes a singular value
> decomposition for the R factor at step 8 if it is rank deficient**… Too small is
> judged relative to the largest singular value multiplied by some power (in the
> range .5 to 1) of the machine precision."

We do not do this. Our deviance-stagnation acceptance rule treats the *symptom*
(the fit will not terminate) rather than the *cause* (the parameter is
unidentifiable and should be dropped). The published fix is rank truncation at a
`sqrt(eps)`-relative threshold — machinery we already own in `solvers/rank.py`.

**4.2 — Our stagnation gate has precedent, but the published one is stronger.**
The reference method uses
relative penalized-deviance stagnation as its *primary* convergence test
as its primary convergence test, and the coefficient-step criterion is likewise
abandoned there, exactly as we found. But acceptance is gated on the
**gradient norm** of the penalized deviance, where we gate on `termination_reason ==
"max_iter"` plus absence of step rejections. A gradient-norm gate *certifies* a
stationary point; ours *infers* one from lack of movement. Theirs is the more
defensible test, which matters for governance.

**4.3 — Softplus in place of `exp` is available but addresses less than it appears.**
The reference method offers an opt-in softplus (`(1/b)·log(1+exp(bx))`, linear
above a threshold). It fixes *overflow*, not the identifiability of the flat region —
a coefficient that wants to be zero still has to go to `−∞`. Lower priority than 4.1.

---

## 5. An unpublished dependency worth flagging

**The reference method has no REML.** It optimizes GCV or UBRE, and the 2024 EFS extension
deliberately targets GCV/UBRE rather than REML. superglm runs SCOP under REML
(`reml/scop_efs.py`, `scop_geometry.py`). There is no published REML or
Laplace-marginal-likelihood scheme for SCOP-splines and no reference implementation
to validate against.

That is not a defect — it may well be correct — but it is **beyond the literature**,
which matters for a library whose output has to be defensible. It should be stated
as such rather than assumed to rest on Pya & Wood.

---

## 6. What a QP is still needed for

Beyond SCOP initialization, an inequality-constrained QP expresses things the
reparameterization structurally cannot:

- **two-sided bounds** (`a ≤ f(x) ≤ b`) — bounding a cumulative sum from above is not
  a coordinate-wise positivity condition, and the reference method has no upper bound
- **cross-term constraints** (`f₁(x) + f₂(z) ≥ 0`, orderings between smooths) — the
  reparameterization is per-term and coefficient-local
- **point and derivative constraints** at arbitrary locations
- **necessary rather than merely sufficient** shape certification — SCOP's condition
  is sufficient-only for cubic and higher (their §6)
- **bases without a first-difference derivative structure**

If any of those are on the roadmap, the QP stays regardless.

---

## 7. Decisions proposed

1. **Do not retire the QP.** Keep it, scoped to what it is for: SCOP initialization,
   and enforcement for `bs`/`cr`.
2. **Do not migrate `bs`/`cr` to SCOP.** It destroys what those classes are —
   measured, not argued.
3. **Adopt mgcv's discipline at the QP boundary**: require full column rank and
   refuse otherwise, as established constrained-least-squares practice does.
   This reverts
   the premise of audit item 3, which read a load-bearing guard as a robustness gap.
4. **Fix QP-path λ selection**, or document it prominently as the ad hoc method the
   literature says it is. This is the largest real quality gap.
5. **Replace the SCOP stagnation workaround with the published fix** — SVD rank
   truncation of the R factor — and strengthen the acceptance gate to a gradient-norm
   test.
6. **State the REML-for-SCOP position explicitly** in the governance documentation.

---

## 8. Sources

Pya & Wood (2015), *Shape constrained additive models*, Statistics and Computing
25:543–559, [DOI 10.1007/s11222-013-9448-7](https://doi.org/10.1007/s11222-013-9448-7),
plus its supplementary material S.1–S.7 ·
Pya Arnqvist (2024), *On some extensions of shape-constrained GAM in R*,
[arXiv:2403.09438](https://arxiv.org/html/2403.09438v1) ·
published package documentation ·
published package documentation ·
Wood (1994), SIAM J. Sci. Comput. 15(5):1126–1133 (abstract only) ·
Liao & Meyer, *cgam*, JSS 89(5), [arXiv:1812.07696](https://arxiv.org/pdf/1812.07696)
