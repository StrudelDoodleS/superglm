# Research track 3: sparse factorization & selected inverse — findings

## ⭐ AI-REML is strictly better than EFS for the mgcv-parity constraint

**AI-REML approximates only the HESSIAN; the gradient stays exact, so the fixed point is the exact REML
optimum. It is an approximate Newton *step*, not an approximate *estimator*.** EFS/SOP by contrast make the PQL
simplification — mgcv's own `?gam` concedes EFS is *"eventually **approximately** maximizing the marginal
likelihood."*

**This re-ranks the EFS discussion: AI-REML should be primary, EFS/SOP secondary.**

Concrete form for superglm's criterion (agent's derivation, flagged as inferred — verify before building):
keep the quadratic form `β̂'SᵢH⁻¹Sⱼβ̂` and the cheap penalty term `−½tr(S_λ⁻SᵢS_λ⁻Sⱼ)`; **drop**
`+½tr(H⁻¹SᵢH⁻¹Sⱼ)`. Compute `gⱼ = Sⱼβ̂`, `hⱼ = H⁻¹gⱼ` — **exactly q solves total** — then `AIᵢⱼ = gᵢ'hⱼ`.

- **No published instance of AI-REML applied to GAM smoothing parameters was found — a genuine gap.**
- AIREMLF90 converges in **5-15 rounds vs REMLF90's 50-300**.
- Documented failure modes with standard fixes: non-PD updates handled by blending AI with an EM update and
  increasing the EM weight until the update lands in the parameter space; boundary cases inflate the
  convergence criterion.

## The production architecture (animal-breeding standard) — Takahashi and AI are COMPLEMENTS

Meyer's AI-REML paper p.5, verbatim: *"Johnson and Thompson… and Gilmour et al. (1995)… derive expressions for
∂logL/∂θᵢ… **which require selected elements of C⁻¹**. Their scheme is computationally feasible due to the
**sparse matrix inversion method of Takahashi et al. (1973)**."*

**sparse Cholesky → Takahashi selected inversion for the first-order traces → AI for the curvature.**
That is ASReml, BLUPF90, WOMBAT and DMU. **The first-order trace does NOT cancel under AI** — AI kills only the
second-order traces. This is exactly the split the audit's §H.4 was missing.

## ⭐ Smith (1995) reverse-mode Cholesky AD — possibly cheaper than Takahashi for the gradient

Backward-differentiating the Cholesky yields a matrix F at *"about twice as much work as one likelihood
evaluation"*, after which **every** gradient component is `tr(F ∂M/∂θᵢ)` — **all q traces at O(1)
factorizations**. Better constant than selected inversion (which costs ~2× per pass) — but selected inversion
also hands you `diag(H⁻¹)`, which superglm needs for EDF/SEs.

**LMMsolver (Boer 2023, Statistical Modelling 23(5-6):465-479)** uses exactly this for sparse P-spline mixed
models and reports **mgcv 600 s → 1 s** and **mgcv 38 min → 30 s**. **Closest published prior art to what
superglm wants — read before committing to Takahashi.**

Recommended split: **Smith-1995 Cholesky AD for the gradient traces, Takahashi for `diag(H⁻¹)`.**

## Correction to the track's own earlier cost model

`O(qp² + p³)` for mgcv's Newton is only right **when W is frozen**. Wood (2011) states `O(M·n·q²)` for the
log-determinant second derivatives, noting *"this step dominates the method's computational cost."* The extra
`T_k` terms exist because **W depends on λ through β̂** — they vanish for Gaussian-identity and for any
fREML/bam working model (confirmed: `fast-REML.r` contains no `T_k` terms) but dominate otherwise.

## lme4's escape route is closed to superglm — and the reason matters

lme4 §3.5 verbatim: the log-det gradient *"is easy to express but **difficult to evaluate for the general
model**… **Although we do not use these results in lme4**"*; §4 requires user optimizers to *"not require a
gradient function."* It profiles β and σ out, leaving <10 parameters for BOBYQA/Nelder–Mead, needing only
`log|L_θ|`.

**But lme4 never reports a penalised EDF or per-coefficient posterior SE — that is WHY it never needs H⁻¹.**
superglm, targeting mgcv parity, does need `diag(H⁻¹)`. So profiling-plus-derivative-free is **not available**.

Symbolic reuse confirmed: *"The symbolic phase… **does not depend on the value of θ**."* ⇒ symbolic analysis
once per fit, numeric refactorization per iteration.

## HDFE refutation strengthened to near-proof

**Kline, Saggio & Sølvsten (2020, Econometrica 88(5)), Appendix B.3, verbatim:** with hundreds of thousands of
effects, *"**making it infeasible to compute `S_xx⁻¹` directly**. To circumvent this obstacle, we instead
compute… `S_xx⁻¹xᵢ` **separately for each i = 1,..,n**"* — n separate PCG solves, **8 hours on 32 cores** for a
million effects. They then replace it with a Johnson–Lindenstrauss randomised estimator needing only
p ≈ 250-2500 solves: **8 hours → under 5 minutes, error ~1e-4**, with explicit bias correction.

That is what "the inverse is inaccessible to alternating projections" looks like when economists actually
need it.

- `lfe` provides FE standard errors *"by **bootstrapping**"* only; `reghdfe` not at all.
- `lfe` manual: connected components are for *"**when there are only two factors**"*; beyond two
  *"their interpretation is in general not well understood"*; `exactDOF` *"may fail if there are too many
  levels."*
- **Somaini–Wolak cannot generalise past two dimensions** — the construction depends on `D'D` and `H'H` both
  being diagonal; with three FE blocks the Schur complements stop being diagonal-plus-low-rank.
- **Their `Pᵢᵢ` is `diag(XH⁻¹X')` — leverage in *observation* space, NOT `diag(H⁻¹)` in coefficient space.
  superglm needs BOTH.** (Note: `diag(XH⁻¹X')` is exactly the `h` vector from track 2's Rank-1 leverage
  reformulation — so the two tracks need the same object for different purposes.)

## ⚠ Two repo findings that change the implementation path

1. **EFS is already shipped, just unwired.** `reml/efs.py` and `reml/scop_efs.py` exist but are reachable only
   on the λ₁>0 and SCOP paths. **Wiring the existing EFS/AI machinery to the main REML path is a far smaller
   change than assumed.**
2. **The dense inverse is worse than the audit described.** `_safe_decompose_H`
   (`solvers/irls_direct.py:235-350`) does `cho_solve((L,True), np.eye(p))` — an **explicit dense p×p
   inverse — on essentially every PIRLS/Newton/line-search iteration**, and `inference/covariance.py` builds a
   **second** dense inverse by a different route. The design comment at `irls_direct.py:8` still reads
   *"p is ~50-80… making the p×p solve trivially fast."* **That stale assumption is the actual root of the
   large-p ceiling** — sharper than RFC-7's framing.

## Two practical warnings

- **scikit-sparse has NO selected inversion and won't soon** — the `spinv` PR (scikit-sparse#1) is still open,
  never merged. **Port Davis's `sparseinv` (~200 lines, BSD-3)** — which also resolves the audit's GPL
  contamination concern for a permissively-licensed library.
- **MSSM's warning, worth heeding literally:** they found sparse Cholesky **slower than dense** when H wasn't
  genuinely sparse, and note the Newton update's by-products are *"**inherently dense**. At best this would
  simply nullify any advantage gained by utilizing sparse matrices."*
  **Profile `nnz(L_H)` on a real model before investing.**

## Net recommendation

Adopt **sparse Cholesky + Takahashi selected inversion + AI-REML** — the verified animal-breeding architecture
— **rather than EFS**, because AI keeps the REML gradient exact and therefore preserves the mgcv-parity claim.
Evaluate **Smith-1995 Cholesky AD** as a possibly-cheaper substitute for selected inversion in the gradient,
keeping Takahashi for `diag(H⁻¹)`. Gate the whole sparse investment on a measured `nnz(L_H)` fill-in test.
