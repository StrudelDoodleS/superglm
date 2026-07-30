# Research track 2: fast smoothing-parameter estimation — findings

**Headline: cost CAN be made flat in q, exactly, with no criterion change.** The measured O(outer·q·n·p²)
blow-up is **not intrinsic to Wood (2011)** — it is an artifact of how the W(ρ) correction is assembled in
`reml/w_derivatives.py`. Verified algebraically, numerically (1.8e-15), and benchmarked at superglm's exact
profiled dimensions.

## ⭐ RANK 1 — Leverage-diagonal reformulation of the W(ρ) gradient. Exact. Flat in q.

Current (`w_derivatives.py:260-266`): `a_j = dW_deta * deta_j`; `C_j = X'diag(a_j)X` (p×p Gram, **O(np²), q
times**); `grad_corr[j] = 0.5·tr(H⁻¹C_j)`.

By cyclicity of trace: `tr(H⁻¹X'diag(a_j)X) = tr(diag(a_j)·XH⁻¹X') = Σₖ a_j[k]·hₖ` with
`hₖ ≡ (XH⁻¹X')ₖₖ`. So with `L = chol(H)`, `M = X L^{-T}`, `h = rowSums(M ⊙ M)`:

> **grad_correction = 0.5 · Aᵀh**, A = [a₁ … a_q] (n×q)

`h` computed **once** in O(np²); each of the q corrections then costs **O(n)**.
**O(q·n·p²) → O(np² + qn).**

**Measured** (identity check: max rel error 1.8e-15):
| n=20000, p=225 | current | reformulated | speedup |
|---|---|---|---|
| q=4 | 149 ms | 151 ms | 1.0× |
| q=25 | 970 ms | 156 ms | **6.2×** |
| q=40 | 1513 ms | 155 ms | **9.8×** |

**The reformulated column is FLAT in q.** That is the sublinearity the audit was looking for.

**Free by-product:** `h` is the hat-matrix diagonal ⇒ per-observation leverage and total EDF at zero marginal
cost.

**Memory caveat:** `M = X L^{-T}` is n×p dense — 36 MB at n=20k/p=225, but **1.2 GB at the MTPL2 n=678k**.
**Chunk over rows** (accumulate `h` blockwise); arithmetic unchanged, still one O(np²) pass.

## ⭐ RANK 2 — The Hessian also consumes C_j; dropping it is PARITY-SAFE

`gradient.py:156-158` reuses `C_j` for the Hessian, so Rank 1 only pays if the Hessian is handled too.

**The unlocking architectural fact** (`direct.py:334-346`): convergence is tested on
`‖proj_grad‖ < tol·(1+|obj|)`. **The fixed point is determined entirely by the gradient. Any change to the
Hessian is parity-safe by construction; only gradient changes move estimates.**

- **Option A (trivial):** pass `dH_extra=None`. `gradient.py:123` then switches to `use_compact_trace`
  (block-local `tr(H⁻¹dSᵢH⁻¹dSⱼ)`, no Grams). Newton direction becomes Gauss-Newton-ish; **fixed point
  unchanged**. This is what the reference implementation does with a cheaper outer optimizer.
- **Option B (exact, also q-independent):** cross-terms `tr(H⁻¹λᵢSᵢH⁻¹Cⱼ) = Σₖ aⱼ[k]·gᵢ[k]` with
  `gᵢ[k] = λᵢ‖Vᵢᵀ H⁻¹ xₖ‖²`, `Ωᵢ = VᵢVᵢᵀ` of rank rᵢ. All `gᵢ` cost O(n p Σᵢrᵢ); **because penalty blocks are
  disjoint, Σᵢrᵢ ≈ p ⇒ O(np²) total, not O(q np²)**. Only the pure second-order `tr(H⁻¹CᵢH⁻¹Cⱼ)` resists.
- **Never use `w_correction_order=2` at large q** — it is O(q²np²) (325 Grams/iteration at q=25). Hard-gate it.

**Combined Rank 1+2 on the 172 s / q=25 case: → ~32-40 s (4.3-5.4×). q=40 (currently >520 s timeout) → well
under a minute.** No criterion change.

> **This supersedes the audit's RFC-2 framing.** RFC-2 proposed batched whitened Grams: same O(T·q·n·p²) flops
> at a ~10× better BLAS constant. This is asymptotically better — **flat in q, not merely cheaper per gram**.
> The C2 verifier had already spotted that the gradient alone becomes O(R(np²+qn)) via the leverage identity
> but noted the Hessian retains O(R·q·n·p²); Rank 2 is the missing piece that removes that too.

## ⭐ RANK 3 — superglm's EFS implements the WRONG update (one-line fix, ~2× fewer iterations)

Wood & Fasiolo (2017), Biometrics 73(4):1071-1081, arXiv:1606.04802. Their eq. 3/5:
`λ*ⱼ = φ·[tr(S_λ⁻Sⱼ) − tr((X'WX+S_λ)⁻¹Sⱼ)]/(β̂ᵀSⱼβ̂)·λⱼ`, i.e. for non-overlapping penalties
**λ* = φ(rⱼ − λⱼgⱼ)/bⱼ**. Positivity guaranteed by their **Theorem 1**.

superglm (`efs.py:249-256`, identically `runner.py:229-236`) computes **λ* = r/(b+g)** — which is precisely the
*"accelerated EM"* update W&F describe in §2.1 and then prove (**their Theorem 2**) takes a **strictly shorter
step** than their method. Same fixed point ⇒ **estimates unaffected, only iteration count**.

**Measured:** Gaussian, n=3000, q=8, k=12, to max|Δlog λ|<1e-6 — superglm form **49 iterations** vs
Wood–Fasiolo form **25**. Poisson/log n=8000 q=6: **68 vs 42**. Consistent **~1.6-2×**.

Fix: `lam_new = (r_j - lambdas[pc.name] * trace_term) / max(inv_phi * quad, 1e-300)` (retain a clamp for the
observed-Hessian-indefinite case; W&F §3 requires the expected Hessian or a nearest-PD substitute).

**Doc bug:** `efs.py:11-12` and `:77-79` cite the paper's subtitle as *"…shape constrained regression"*;
correct is *"…Tweedie location, scale and shape models"*.

## ⚠ RANK 7 (correctness, not speed) — superglm's EFS is WRONG for overlapping penalties

`efs.py:251` and `runner.py:233` hardcode `rⱼ = rank(Ωⱼ)`, i.e. `tr(S_λ⁻Sⱼ) = rⱼ/λⱼ` — **valid only for
non-overlapping penalties.** Under **`select=True` double penalties** or **tensor `ti()` terms** (multiple
penalties on the same coefficient block) this is silently wrong. This is exactly the case W&F's general form
and SOP's `Λ_{k_l}` form were written to fix. **Correctness issue independent of speed, and it sits on
protected semantics.**

## RANK 4 — BFGS outer loop for large q

the reference implementation's own docs: `optimizer = c("outer","bfgs")` — *"For models with large numbers of smoothing parameters,
the bfgs method option can be faster than the default newton optimizer."* W&F's own §4 comparison is against
quasi-Newton, not Newton (motorcycle: EFS 39 steps vs QN 32; Cox colon: 15 vs 16).

**Why it fits superglm:** BFGS needs only the gradient, so with Rank 1 the whole outer iteration is
`O(pirls(np²+p³) + np² + qn)` — **fully q-independent, and exactly Wood (2011) LAML because the gradient is
exact. Zero parity risk.** Cleanest large-q path preserving reference parity by construction.

## RANK 5 — SAP / SOP

SAP: Rodríguez-Álvarez, Lee, Kneib, Durbán & Eilers (2015), Statist. Comput. 25:941-957.
SOP: Rodríguez-Álvarez, Durbán, Lee & Eilers (2019), Statist. Comput. 29:483-500, arXiv:1801.07278.
SOP generalises SAP (SAP = anisotropic-tensor special case). Mixed-model form with precision linear in inverse
variances: `G_k⁻¹ = Σ_l σ_{k_l}⁻² Λ_{k_l}`.

Update (their Thm 1, eqs 13-14): `σ̂²_{k_l} = (α̂ᵀΛ_{k_l}α̂)/ED_{k_l}` with the computational identity (§3.3)
`G_k Z'PZ G_k Λ_{k_l} = (G_k − C*_{kk})Λ_{k_l}` where **C\* is the inverse of the mixed-model coefficient
matrix already computed for β̂,α̂**. Reduces to `ED_{k_l} = tr((G_k − C*_{kk})Λ_{k_l}/σ²_{k_l})`, and *"where
Λ_{k_l} is diagonal, only the diagonal elements need be obtained"*.

**Cost:** O(np²+p³) for the one factorisation + q **block-local** traces of O(p_k²) (or O(p_k) diagonal).
**q-dependent part is lower-order ⇒ effectively constant in q.** Same class as EFS.

**Reported timings:** Doppler, 15 variance params: **SOP 1.0 s vs the reference implementation 45 s (45×)**, EDs 50.2 vs 50.0.
X-ray diffraction Poisson, 200 B-splines/80 adaptive: SOP <3 s, *"the reference implementation around 1000 times slower"*.
2D adaptive, **128 variance parameters**: SOP 22 s, **~30× faster than Wood (2011)**.

**Criterion:** solves REML score equations of the *PQL-linearised* REML log-likelihood ⇒ **same fixed point as
EFS, not exactly Wood (2011) for non-Gaussian.**

**SOP advantages over EFS specifically for superglm:**
1. **No Moore-Penrose pseudo-inverses** — they call out W&F's reliance on them as *"may present numerical
   instabilities"*.
2. **Partial EDFs fall out free**, with `Σ_l ED_{k_l} = ED_k = tr(H_k)` — directly useful for `summary()`.
3. **Handles overlapping penalties correctly** — fixes the Rank-7 bug above (`select=True`, tensor `ti()`).
4. Positivity as checkable rank conditions (their Thm 2) — though conditional on the current iterate, which
   they concede *"may not be an easy task"* to check in advance.

## §6 — Does any of this change estimates? (the reference-parity question) — MEASURED

W&F are explicit (§3): the general update follows PQL/POI, *"[which] both neglect the dependence of
∂²l/∂β∂βᵀ on λ"* — **exactly** superglm's W(ρ) correction. They state *"in practice the λ estimate no longer
exactly maximizes l_r"*. Koslik (2024, arXiv:2411.11498) names it honestly: **qREML** (*quasi* REML).

**Two superglm families get EXACT REML from EFS for free** (dW/dη ≡ 0):
- **Gaussian/identity** (W constant).
- **Gamma/log** — from superglm's own `compute_dW_deta`: `sw(g₁/V)(2g₂ − g₁²V'/V) = sw(1/μ)(2μ−2μ) = 0`.
  **For Gamma/log severity models, EFS *is* exact Wood (2011) REML.**
Non-zero for Poisson/log (dW/dη = W), Binomial/logit, Tweedie/log (vanishing only at p=2).

**Measured EFS fixed point vs derivative-free optimum of exact Wood-2011 LAML:**
| family | n | q | max\|Δlog λ\| | ΔEDF | max rel Δμ̂ |
|---|---|---|---|---|---|
| Poisson/log | 400 | 5 | 0.0151 | −0.022 | 2.2e-3 |
| Poisson/log | 2 000 | 5 | 0.0031 | −0.002 | 1.7e-4 |
| Poisson/log | 8 000 | 6 | 0.0011 | −0.000 | 3.4e-5 |
| Binomial/logit | 400 | 5 | 0.158 | −0.244 | 2.7e-2 |
| Binomial/logit | 2 000 | 5 | **1.554** ⚠ | −0.050 | 3.9e-3 |

**Conclusion: EFS/SOP are safe on fitted values and EDF, NOT safe on reported λ in flat regions.** The Binomial
n=2000 row: one λ drifted by e^1.55≈4.7 and EFS hit an 800-iteration cap, yet EDF moved 0.05 and μ̂ 0.4% — a
weakly-identified λ on a flat REML surface. **Newton detects the flatness and terminates; the fixed point
crawls.** Matters if `summary()` prints λ/`sp` that users diff against the reference implementation.

## RANK 6 — Line search: 8 full PIRLS fits per Newton iteration

`direct.py:419-470`, `max_ls=8`, **each a complete `fit_irls_direct` to convergence**. Remedies: (1) cheap
frozen-W surrogate objective (one back-solve, no data pass) for early rejections, full PIRLS only on the
survivor — Wood (2011) §6 and W&F both use step-halving with a *single* objective evaluation per trial;
(2) W&F's §4 examples converged **"without step length control"** at all — their Theorem 3 says the FS step
asymptotically lands *between* λ and λ̂ and does not overshoot. Orthogonal to Rank 1; stacks.

## RANK 8 — REJECT: AD / implicit differentiation as a replacement for the analytic correction

1. **superglm's analytic correction IS already the IFT solution** — `w_derivatives.py:253-254` computes
   `dβ/dρ_j = −H⁻¹Sⱼβ̂`, exactly the `−[∂²L/∂w∂wᵀ]⁻¹∂²L/∂w∂λᵀ` of Lorraine, Vicol & Duvenaud (2020,
   arXiv:1911.02590) Thm 1. **Nothing for AD to discover**; it would rediscover the same expression with tape
   overhead.
2. **The ML cost story does not transfer.** Lorraine et al. get hyperparameter-count independence because the
   hypergradient is a *single* vector-inverse-Hessian product. superglm's objective is a **log-determinant**
   whose ρ-gradient is a **trace** — one per j, not one vector solve total. Neumann/CG buys nothing a Cholesky
   of a 225×225 matrix doesn't give exactly.
3. Their own trade-offs (Neumann more stable than CG but worse in ℓ₂ per step) are unattractive when an exact
   answer exists. Recent bilevel work targets p ≫ 10⁶; superglm's H is 225×225.

**One useful AD-adjacent precedent:** Koslik's qREML uses AD for the *inner* penalised likelihood only, and
finite-differences the AD gradient for the Hessian because *"much faster than computing the Hessian using
AD"*. If superglm adds families with painful analytic `deriv3_inverse` (the `_compute_d2W_deta2_fd` fallback
anticipates this), AD is an **implementation aid, not a performance strategy** — which is exactly the gap W&F
cite as EFS's motivation (*"only necessary to compute with the same first and second derivatives … not the
third or fourth order derivatives required by alternative approaches"*).

## RANK 9-10 — parked

- Hutchinson sketch of the irreducible `tr(H⁻¹CᵢH⁻¹Cⱼ)`: with `M = XL^{-T}`, equals `⟨Nᵢ,Nⱼ⟩`,
  `Nᵢ = Mᵀdiag(aᵢ)M`; s probes give O(q·s·np) vs O(q·np²) — factor p/s (~11× at p=225,s=20). Grounded in Dong
  et al. (2017, arXiv:1711.03481), SLQ. **Only matters if you keep an exact Hessian, which Rank 2/4 says you
  shouldn't. Park.**
- **SQUAREM** (Varadhan & Roland; arXiv:1810.11163) for the fixed point — off-the-shelf, 2-10× on iteration
  count for slow monotone EM/MM fixed points; stacks with Rank 3. Note it cannot rescue a *non-converging*
  fixed point (the Binomial case).

## Honest failure modes

- **EFS linear convergence is not a bound**: measured 25-70 iterations typical but **800+ and non-convergent**
  for a weakly-identified Binomial λ. W&F's Thm 3 corollary: *"iteration of update (3) will generally converge
  more slowly than Newton's method, when close to the optimum, and certainly no faster."* **Any EFS-default
  policy needs an iteration cap plus Newton fallback, never an unconditional switch.**
- EFS only guaranteed to increase l_r when ∂²l/∂β∂βᵀ is λ-free; `efs.py:285-334`'s "stale-basis uphill guard"
  documents this honestly — it is a heuristic, not a monotonicity proof, and cannot be made one under PQL.
- SOP positivity is conditional on the current iterate (their Thm 2).
