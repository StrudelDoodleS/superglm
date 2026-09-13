# Distribution-estimation correctness audit — 2026-08-20

Scope decision document for **release 0.29.0**. Audit target: `master` @ `96701ac8`
(= published v0.28.0), worktree `.worktrees/release-0.28.0`, tree clean. Everything
here is **analysis**: no source changes accompany this document.

Subject: every quantity superglm estimates that is not a regression coefficient —
the dispersion φ, the NB2 shape θ, the Tweedie index p, effective degrees of
freedom, and the smoothing parameters λ chosen by REML. Two findings were seeded
(A: the Tweedie scale profile, issue #339; B: NB θ estimated outside the REML
loop); the systematic sweep (C) covered the rest of the class.

**Method.** Code claims carry `file:line` against `96701ac8`. Magnitude claims
carry a probe. Probes are equivalence controls (shipped criterion vs an exact
criterion reconstructed from the repo's own density machinery) validated two
ways: against the shipped optimizer (`fit_reml`'s λ̂ reproduced to 3 significant
digits by the probe's grid argmin of the shipped criterion, twice) and against
mgcv 1.9.3 run as a **black-box oracle** (GPL: executed only, outputs pinned
below, no source consulted; methods cited from the authors' published papers).
MASS 7.3.65 was not needed as an oracle. statsmodels 0.14.6 (BSD-3) was read and
run freely. No timing claims are made anywhere in this document, so the
benchmark-lock/thread-pinning protocol was not needed (probes still ran with
BLAS pools pinned to 1).

Probe scripts live in the session scratchpad
(`/tmp/claude-1000/-home-max-projects-superglm/a434966d-111f-4688-bcd5-ae368cdb67f4/scratchpad/audit/`):
`probe_a0_consistency.py`, `probe_a1_tweedie_sweep.py`, `probe_common.py`,
`probe_a2_re2.py`, `probe_b1_nb.py`, `probe_b2_nb_edge.py`,
`probe_b3_theta_root.py`, `probe_c1_efs.py`, `probe_c3_phi_p.py`,
`oracle_mgcv_re.R`, `oracle_mgcv_nb.R`/`_nb2.R`, `oracle_mgcv_tw.R`. They are
scratch-only by design; copy them out before the session directory is reclaimed
if the reproductions are wanted long-term. All are synthetic-data only.

---

## 1. Verdict table, ranked by user-visible impact

| # | Finding | Verdict | Measured magnitude | 0.29.0? |
|---|---|---|---|---|
| 1 | **B2** — NB θ Newton takes the wrong root; the silent clamp to [0.1, 50] publishes it with `converged=True` | **Defect, confirmed** | θ̂ = 50.0 where the truth is 0.05 and mgcv finds 0.0507 — **3 orders of magnitude**, silent, via the public `NegativeBinomial("auto")` + `fit_reml` path | **Yes — fix** |
| 2 | **A** — Tweedie estimated-scale REML substitutes a Gaussian-style scale profile that charges zero rows a `log φ` the exact saturated likelihood does not | **Defect, confirmed & quantified** | Graduated by criterion curvature: identified fits λ̂ off ≤ 0.13 decades, edf ≤ 0.4, μ̂ ≤ 2 % RMS, τ² ≤ 2 %; **flat** RandomEffect cells: τ² off **13–22×** with the RE run to the boundary while the exact optimum (= mgcv's) is interior | **Yes — fix** |
| 3 | **B1** — NB θ is frozen before REML, at a fit with global λ₂ = 0.1 | **Defect, confirmed** | θ̂ biased **−0.2 % to −45 %** (worst measured: 0.554 vs joint 1.007, mgcv 0.996); λ̂ and μ̂ almost unmoved (≤ 1.3 %, ≤ 1.3 % RMS) — the damage is to θ̂ itself and everything priced off V(μ)=μ+μ²/θ | **Yes — fix** |
| 4 | **C2** — AIC/BIC count no estimated family parameter (φ, θ, p) and carry no smoothing-uncertainty correction | **Split-convention gap + documented approximation** | Constant −2·(#family params) tilt in cross-family comparisons (Poisson vs NB-auto; fixed-p vs `estimate_p`); conventions verified: R counts σ (pinned run: `logLik df=3` for a 2-coef gaussian glm), statsmodels does not, mgcv docs describe corrected-edf conditional AIC | Document; optional +param count. WPS correction **excluded** |
| 5 | **C3** — Three published-φ conventions for Tweedie, chosen by entry point | **Inconsistency, confirmed** | Same data: `fit_reml` publishes Pearson 2.3044; `estimate_p` publishes MLE 2.4128 (mgcv `tw()` scale 2.4191); the QP-passthrough branch computes a third (deviance-based) | Document in 0.29.0; unify later |
| 6 | **C1** — `optimize_efs_reml`'s φ uses penalty **rank** where the criterion uses **nullity**, `len(y)` where the contract is `sum(w)`, and a deviance estimate where Gamma's profile is a digamma root | **Defect in dead code** | φ off **3.2×** on a frequency-weighted Gamma; would displace the EFS fixed point **1.87×** in λ — but `fit_reml` coerces λ₁ = 0 first, so the driver is unreachable from the public API | Fix-or-delete note only |
| 7 | **C6** — `profiling/nb.py` cites "MASS::glm.nb and MASS::theta.ml **source code**" | **Provenance hazard** (MIT project citing GPL-2\|GPL-3 source as a reference) | 6 occurrences in `nb.py`, echoed in `docs/audit/2026-07-28/subsystems/families-profiling.md:59,61` | **Yes — reword** |
| — | Everything else swept (§5) | **Verified clean** | Density↔deviance identities ≤ 5e-13 rel.; Gaussian/Gamma profilers exact vs brute force; `estimate_p` matches mgcv `tw()` to 0.26 % on φ and 3 digits on p | — |

---

## 2. Finding A — the Tweedie scale profile

### 2.1 The code

`src/superglm/reml/objective.py:263` gates on `scale_known`. Estimated-scale
families dispatch at `objective.py:292-314`:

- `Gaussian` → `profile_gaussian_reml_scale` (`reml/scale.py:74-113`)
- `Gamma` → `profile_gamma_reml_scale` (`reml/scale.py:220-302`)
- everything else → `scale_term = 0.5 * max(n - M_p, 1.0) * np.log(d_plus_pq)`
  (`objective.py:310-314`), with `n = len(y)` (`objective.py:265`)

`scale_known` is `False` for exactly Gaussian (`distributions.py:97-99`), Gamma
(`:132-134`) and Tweedie (`:283-285`); Poisson (`:62-63`), NegativeBinomial
(`:181-183`) and Binomial (`:235-237`) report `True` and never enter the branch.
So Tweedie is the only **shipped** family that lands in the fallback — plus any
user-supplied custom distribution with `scale_known=False`, which inherits the
same substitution silently.

The in-loop gradient/Newton consumes the matching implicit dispersion
φ = D_p/(n − M_p) (`reml/direct.py:611-656`, `reml/discrete.py:646-690`), so the
optimizer is internally consistent: it stops at the interior minimum of the
criterion it is handed, as issue #339's comment states. The defect is the
criterion, not the search.

### 2.2 Why the substituted form is wrong (literature position)

Wood's Eq. (4) criterion — restated as the extended-GAM LAML in Wood, Pya &
Säfken (2016), §3.3 — is

> V(θ, φ) = −D(β̂, θ)/(2φ) + l̃(θ, φ) − ½·[log|XᵀWX + S_λ| − log|S_λ|₊] + (M_p/2)·log(2πφ)

where **l̃ is the family's saturated log-likelihood**, evaluated exactly. For
Gaussian, l̃(φ) = −(n/2)log(2πφ) + const, and profiling φ out yields exactly the
`0.5·(n − M_p)·(1 + log 2π + log φ̂)` form — which is what the substituted branch
generalizes from. For Tweedie with 1 < p < 2 the response is compound
Poisson–gamma (Jørgensen 1997): the distribution has an atom at zero,
P(Y=0) = exp(−w·μ^{2−p}/(φ(2−p))), and the density at y > 0 is
f(y) = a(y, φ, p)·exp{−w·d(y, μ)/(2φ)} with a(·) the Dunn & Smyth (2005)
summed-from-the-middle series (WPS 2016, Supplementary Appendix J gives the
stable evaluation superglm's `_tweedie_series.py` machinery follows).

Consequently l̃ is **not** −(n/2)log φ-shaped:

- a **zero row's** saturated contribution is exactly 0 for every φ — an atom
  probability has no `1/√φ` Jacobian (verified from the repo's own density:
  probe A0 shows `tweedie_logpdf(0, μ, φ)` equals the atom formula to machine
  precision at φ ∈ {0.5, 1, 2, 4}, and the l̃ contribution is +0.00 at all four);
- a **positive row** contributes ≈ −½·log φ near the saddlepoint regime and
  more at large φ (probe A0: ∂l̃ᵢ/∂log φ measured −0.53 → −0.87 across
  φ = 0.25 → 4).

So the exact profile's "residual likelihood size" is governed by the positive
rows, not n; the substituted branch overweights the deviance arm in proportion
to the zero fraction, exactly as #339 states. The established treatment — WPS
2016 §3.3, implemented as mgcv's `tw()`/`Tweedie()` families — evaluates l̃
exactly inside the criterion. Its assumptions (1 < p < 2; stable series
evaluation) are precisely superglm's supported regime (`Tweedie.__init__`
hard-rejects p outside (1, 2), `distributions.py:278-281`), so **we are inside
the established method's assumptions and outside its implementation**.

### 2.3 The probe construction, and its two validations

The exact criterion was reconstructed per fixed λ from repo machinery only:
V_ex(λ) = ½(log|H| − log|S|₊) + min_φ [D_p/(2φ) − l̃(φ) − (M_p/2)log(2πφ)],
with l̃(φ) = `distribution.log_likelihood(y, μ̂, w, φ)` + D/(2φ). The identity
behind that (l + D/(2φ) is μ-free) was verified first, for **all six families**
(probe A0: max relative discrepancy 5e-13, at Tweedie p=1.8; ≤ 1e-15 elsewhere).

Validation 1 — against the shipped optimizer: the probe's grid argmin of the
**shipped** criterion reproduces `fit_reml`'s λ̂:

| cell | probe grid argmin | shipped `fit_reml` |
|---|---|---|
| smooth, p=1.5, φ=2 | log₁₀λ = −1.916 | −1.913 (edf 7.14 vs 7.13) |
| RE `ident_p15` | λ = 43.71 | 43.83 |
| RE `flat_p15_lowzero` | λ = 111.4 | 109.5 |

Validation 2 — against mgcv (`gam(..., family=Tweedie(p, link="log"),
method="REML")`, black-box, outputs pinned in §2.5): the **exact**-criterion
optimum matches mgcv's REML answer where mgcv's is well-determined.

### 2.4 Measured consequences — smooth terms (probe A1)

One CRS(10) smooth, n = 1200, CPG data, 31+17-point λ grids, quadratic
refinement. `d` = log₁₀λ̂(shipped) − log₁₀λ̂(exact); positive = shipped
over-smooths.

| design | p | φ | zero frac | d (decades) | edf shipped→exact | rel. μ̂ RMS |
|---|---|---|---|---|---|---|
| sharp | 1.2 | 0.5 | 0.08 | +0.052 | 8.67→8.83 | 9.2e-4 |
| sharp | 1.2 | 2.0 | 0.45 | +0.028 | 6.56→6.64 | 1.1e-3 |
| sharp | 1.2 | 6.0 | 0.74 | **−0.153** | 6.68→6.26 | 1.3e-2 |
| sharp | 1.5 | 0.5 | 0.02 | +0.050 | 8.49→8.65 | 1.2e-3 |
| sharp | 1.5 | 2.0 | 0.32 | +0.060 | 7.14→7.32 | 2.8e-3 |
| sharp | 1.5 | 6.0 | 0.66 | **−0.197** | 7.18→6.60 | 1.7e-2 |
| sharp | 1.8 | 0.5 | 0.00 | +0.041 | 8.57→8.70 | 1.1e-3 |
| sharp | 1.8 | 2.0 | 0.07 | +0.127 | 6.62→7.00 | 6.1e-3 |
| sharp | 1.8 | 6.0 | 0.42 | +0.045 | 4.86→4.96 | 5.3e-3 |
| weak | 1.5 | 0.5 | 0.01 | +0.070 | 4.70→4.85 | 2.0e-3 |
| weak | 1.5 | 6.0 | 0.67 | **−0.224** | 4.58→4.16 | 2.0e-2 |
| weak | 1.8 | 0.5 | 0.00 | +0.121 | 4.02→4.24 | 3.8e-3 |
| weak (6 cells) | — | — | — | 0.000 | both at λ ceiling | 0 |

Readings: (i) for **smooth terms the error is small everywhere** — worst
|Δedf| = 0.58, worst prediction shift 2 % RMS; (ii) the **sign flips with the
zero fraction** — low-zero cells over-smooth slightly (the positive rows' l̃ has
saddlepoint φ-curvature the Gaussian form lacks), zero-heavy cells (≥ 0.42)
**under-smooth**, matching #339's direction; (iii) when both criteria agree a
term should vanish (6 weak cells at the λ ceiling), they agree exactly —
concordant outcomes stay concordant.

### 2.5 Measured consequences — RandomEffect variance components (probe A2 + mgcv)

Balanced unpenalized Categorical (6 × 500) beside a 40-level `RandomEffect`,
n = 3000. τ² = φ/λ (per criterion's own φ). No engineered separation (a first
attempt with engineered claim-free levels was discarded: separated fixed-effect
levels make the fixed-λ fits non-MLE in *both* arms, so nothing is measured).

| cell | zero frac | true τ² | τ²(shipped) | τ²(exact) | τ²(mgcv, pinned) | shipped/exact | RE edf shipped→exact (mgcv) |
|---|---|---|---|---|---|---|---|
| `ident_p15` (τ=0.3, φ=3, p=1.5) | 0.54 | 0.090 | 0.0667 | 0.0655 | **0.06582** (sd 0.25655) | 1.02 | 23.7→23.3 (**23.4**) |
| `flat_p15_lowzero` (τ=0.08, φ=0.5) | 0.03 | 0.0064 | 0.00517 | 0.00652 | **0.00662** (sd 0.08139) | **0.79** | 14.4→18.1 (**18.2**) |
| `flat_p18` (τ=0.08, φ=3, p=1.8) | 0.22 | 0.0064 | 4.0e-5 (λ at grid ceiling) | 7.3e-4 | **8.99e-4** (sd 0.02998) | **0.055 — 18–22× off** | 0.03→0.64 (**0.79**) |
| `flat_p15` (τ=0.08, φ=3) | 0.52 | 0.0064 | ~3e-5 | ~3e-5 | 1.76e-4 (CI spans 19 orders of magnitude) | — (all in one flat basin) | ≈0 all three |

Readings: (i) the exact-criterion optimum **is** mgcv's answer — τ² within
0.5–2 %, RE edf within 0.15 where identified — which validates the whole
construction end-to-end; (ii) where the RE is **identified**, the shipped
criterion's error is ~2 % in τ² — negligible; (iii) where the criterion is
**flat**, the shipped criterion lands on the wrong side of the
boundary/interior divide: in `flat_p18` it runs the RE to zero (τ² 4e-5, edf
0.03) while the exact criterion and mgcv keep a small interior component (τ²
9e-4, edf 0.6–0.8) — **18–22× in τ²**. This is the #339 failure class at #339's
magnitude (17× there), with the sign depending on the configuration: the field
case under-shrank; these synthetic flat cells over-shrink; the zero-heavy A1
cells under-smooth. The confirmed prediction is *graduated by curvature*, not a
fixed direction.

**Falsification status of the seeded prediction:** confirmed. Sharply
identified terms move ≤ 2 % in every measured quantity; flat directions move by
factors, up to a boundary-vs-interior qualitative change.

### 2.6 What a fix looks like (established method, already in-repo)

Replace `objective.py:310-314` with an exact Tweedie scale profile in
`reml/scale.py`: minimize D_p/(2φ) − l̃(φ) − (M_p/2)log(2πφ) over log φ, with
l̃(φ) evaluated through `tweedie_logpdf` (or, cheaper and fit-invariant per
evaluation: precompute the zero/positive split once; only Σ log a(yᵢ, φ) over
positive rows varies with φ — the deviance term is already available). The
`ProfiledScaleTerm` contract (`reml/scale.py:16-22`) needs
`d_inverse_phi_d_penalized_deviance`, which follows from implicit
differentiation of the profile score exactly as the Gamma profiler does
(`reml/scale.py:177-217`). The probe's inner minimization converged with a
bounded scalar solve at xatol 1e-9 on every one of ~4,000 evaluations across
probes A1/A2. This changes λ̂, coefficients, edf, φ̂ and information criteria for
**every Tweedie `fit_reml`** — release notes must say so, with the measured
sizes above (most fits: sub-percent; flat variance components: potentially
large, and *more correct*).

---

## 3. Finding B — NB2 θ

### 3.1 The code

`_maybe_estimate_nb_theta` (`model/fit_ops.py:976-985`) runs before design/fit
at both `fit_ops.py:1126` (`fit`) and `fit_ops.py:1501` (`fit_reml`).
`estimate_nb_theta` (`profiling/nb.py:344-538`) alternates a GLM fit at the
model's **configured** smoothing (`configured_lambda2`, i.e. the `spline_penalty`
default 0.1 — REML has not run yet) with `_theta_ml` Newton steps
(`profiling/nb.py:273-323`), converging on |Δθ| < `xatol` = 1e-2
(`nb.py:519`). θ is then frozen into `NegativeBinomial(theta_hat)`
(`fit_ops.py:983`) and REML never revisits it. `_theta_ml` starts at θ = 1.0
always (`nb.py:450`), takes **unsafeguarded** Newton steps, and clips every
iterate into `bounds=(0.1, 50.0)` (`nb.py:279`, `:318`); the auto path passes no
overrides, so those defaults always bind (`fit_ops.py:982`).

The score and information formulas themselves are the correct NB2 profile
derivatives (checked analytically against ∂ℓ/∂θ = ψ(y+θ) − ψ(θ) + log θ + 1 −
log(θ+μ) − (y+θ)/(μ+θ); standard NB2 ML theory, Lawless 1987), and probe A0
confirms the NB2 `log_likelihood`/`deviance_unit` pair is exactly consistent.

### 3.2 The literature position

The alternation itself is the Venables & Ripley (2002, ch. 7.4) `glm.nb`
scheme and converges to the joint (β, θ) MLE **for a fixed model** — its flaw
here is not the alternation but (a) running it at a smoothing that REML then
abandons, and (b) the inner Newton's missing safeguards. The established method
for *penalized* models is WPS 2016: θ enters the LAML criterion and is
optimized **alongside log λ** by Newton — mgcv's `nb()` documents exactly that
("theta is estimated alongside the smoothing parameters by ML or REML", mgcv
1.9.3 `negbin` help). mgcv's older `negbin()` performance-iteration is
documented as a Pearson-moment scheme and is not the model here.

### 3.3 B1 — the freeze, measured (probes B1/B2 + mgcv oracle)

Joint reference: alternate wide-bounds `_theta_ml` at the REML fit's μ̂ with a
full `fit_reml` at the updated θ, to |Δθ| < 1e-6 (≤ 15 alternations). CRS
smooth, n = 2000–3000.

| cell | θ true | θ̂ shipped | θ̂ joint | mgcv `nb()` (pinned) | shipped error | Δλ̂ | Δμ̂ RMS |
|---|---|---|---|---|---|---|---|
| sharp sin(2πx), θ=0.6 | 0.6 | 0.5970 | 0.6142 | 0.60976 | −2.9 % | −0.18 % | 6.9e-4 |
| sharp, θ=5 | 5.0 | 4.7636 | 4.8879 | 4.80199 | −2.6 % | −0.06 % | 1.7e-4 |
| weak, θ=5 | 5.0 | 4.5392 | 4.5636 | — | −0.5 % | −0.09 % | 3.9e-5 |
| weak, θ=0.6 | 0.6 | 0.6112 | 0.6125 | — | −0.2 % | −0.13 % | 7.2e-5 |
| **hi-freq sin(6πx), θ=1** | 1.0 | **0.5541** | **1.0066** | **0.99574** | **−45 %** | +1.3 % | 1.3e-2 |

Decomposition (probe B1): re-running wide-bounds θ-ML at the *calibration*
fit's μ̂ reproduces the shipped θ̂ to 4e-4 — so the entire bias is the μ̂ the
freeze point sees, i.e. **lack-of-fit at λ₂ = 0.1 absorbed into overdispersion,
biasing θ̂ down**, graduated by how far the true function is from what λ₂ = 0.1
can follow. λ̂ is almost θ-insensitive (≤ 1.3 % everywhere measured), so the
smoothing choice survives; the casualty is θ̂ itself, hence V(μ) = μ + μ²/θ
(45 % low θ ⇒ up to ~80 % overstated μ² variance share), every Wald SE built on
it, and any Poisson-vs-NB comparison.

**xatol = 1e-2 is a non-issue**: tightening to 1e-8 moved θ̂ by ≤ 1.2e-4
(probe B1, all four cells) — four orders below the freeze bias. Defensible;
document, don't change.

### 3.4 B2 — the wrong root and the silent clamp (probes B2/B3 + oracle)

Data: n = 4000, smooth signal, θ_true = 0.05 (heavy but realistic
overdispersion: ȳ = 3.8, s² = 325). The profile NLL at the fitted μ̂ (probe
B3) has its minimum at θ ≈ 0.051 and **rises monotonically** through θ = 50
(NLL 1.178 → 8.01) — yet:

- `_theta_ml` from the fixed start 1.0, default bounds → **50.0** (the clip
  turns an uphill Newton run into the upper bound; successive clipped iterates
  are equal, so |Δθ| < xatol reports `converged=True`);
- from 1.0 with wide bounds → 6.97e8 (the Newton crosses into a region where
  the profile information is negative and takes ascent steps);
- from 0.05 with wide bounds → 0.0509 (correct — the formulas are fine, the
  globalization is absent);
- public path end-to-end (`NegativeBinomial("auto")` + `fit_reml`, probe B2):
  **publishes θ̂ = 50.0000**; mgcv `nb()` on the same CSV: **0.05068**.

The same clamp also binds benignly at the other end: θ_true = 150 (near-Poisson
data) publishes θ̂ = 50 where the free optimum is ≈ 340 — V(μ̄) overstated 6.5 %,
and the NB fit is stopped ~9 AIC short of the Poisson fit it should approach.

This is the audit's top-ranked item because it is **silent, categorical, and
reachable with default settings**: heavy-overdispersion count data is the
population NB2 exists for. Fix shape (standard 1-D practice, no novelty
claimed): a data-driven start (moment estimator μ̄²/(s² − μ̄) or equivalent),
score-sign safeguarding / bisection fallback so steps never ascend the NLL,
bounds either widened by orders of magnitude or made a loud warning when they
bind, and `converged=False` whenever a bound is active.

### 3.5 Recommended B fix for 0.29.0

1. **B2 first** (small, self-contained, catastrophic when hit).
2. **B1**: after REML converges, re-estimate θ at the REML μ̂ and alternate
   (θ-update ↔ `fit_reml` warm-started) to a joint fixed point. The probe's
   loop converged in ≤ 15 alternations (typically ~4–8) with λ̂ moving ≤ 1.3 %,
   so warm-started refits are cheap; this reuses only existing pieces. The full
   WPS-2016 treatment (θ as a coordinate of the LAML Newton) is the eventual
   destination but is **not** required to remove the measured bias, and is
   excluded from 0.29.0 (§6).

---

## 4. Finding C1 — the EFS φ (dead code on the public path)

`reml/efs.py:122-131` and `:227-233` estimate
φ = (deviance + βᵀSβ)/(n − M_p) with `M_p = compute_total_penalty_rank(...)` —
the total penalty **rank** — and `n = len(y)` (`efs.py:91`), for **all**
estimated-scale families. Three separate deviations from the criterion the rest
of the codebase optimizes:

1. rank vs **nullity** (`objective.py:274-290` uses `compute_penalty_nullity`,
   penalty_algebra.py:1171 — Wood's M_p is the null-space dimension);
2. `len(y)` vs the frequency-weight contract's `sum(w)`
   (`solvers/dispersion.py:12-36`, `reml/scale.py:229-230`);
3. a deviance-form estimate where Gamma's Eq.-(4) profile is a digamma root.

Measured (probe C1, Gamma, frequency weights 1–5, n = 900, Σw = 2691, CRS(12),
rank(S) = 12, M_p = 2): φ_efs = 1.2051 vs exact profile φ = 0.3748 — **3.2×**,
dominated by len(y)-vs-sum(w). Isolation is clean: the EFS fixed-point map
evaluated at the direct-REML optimum with the *correct* φ reproduces λ̂ to 0.1 %;
with φ_efs it lands at **1.87× λ̂**. (Reference: Wood & Fasiolo 2017 give the
update with σ̂² = ‖y − Xβ̂‖²/[n − tr{(XᵀX+S)⁻¹XᵀX}] — an n−edf form that
coincides with the D_p/(n − M_p) profile at the joint optimum; superglm's
rank-based divisor coincides with neither.)

**Reachability:** none from the public API. `fit_reml` validates and coerces
λ₁ = 0 (`model/base.py:1136-1145`, called at `fit_ops.py:1492`), so
`use_direct` (`fit_ops.py:1629`) is always true and the EFS branches in
`model/reml_execute.py:354,398` never run. The SCOP EFS loop, which **does**
run inside `fit_reml`, resolves φ correctly through the objective's profiled
scale (`reml/scop_efs.py:620-637`). Verdict: latent defect; **fix or delete the
dead driver** — do not leave a 1.87×-λ landmine for whoever revives λ₁+REML.

---

## 5. The systematic sweep — everything else examined

### 5.1 Verified correct (evidence attached)

| item | check | result |
|---|---|---|
| `log_likelihood` ↔ `deviance_unit` mutual consistency, all 6 families | probe A0: l + D/(2φ) must be μ-free | ≤ 5e-13 relative (Tweedie p=1.8 series floor), ≤ 1e-15 elsewhere; includes y=0 rows for Poisson/NB/Tweedie |
| Tweedie unit deviance at y = 0 | code path `profiling/tweedie.py:454-530` (y=0 rides the `rounded_to_minus_one` branch) | evaluates exactly 2μ^{2−p}/(2−p); the zero-atom identity in §2.2 holds to machine precision |
| `profile_gaussian_reml_scale` "exact" claim | brute-force min over log φ of the Wood scale part | φ to 5e-9, criterion to <1e-10 (probe A0) |
| `profile_gamma_reml_scale` "exact" claim | same, with the exact Gamma l̃ | φ to 5e-9, criterion to <1e-10 |
| `scale_known` per family | read, per-family (§2.1 line refs) | correct: exactly {Gaussian, Gamma, Tweedie} estimated; NB2 deliberately φ=1 (θ carries overdispersion), Poisson/Binomial φ=1 |
| edf | `solvers/irls_direct.py:2733-2735`: 1 + tr((XᵀWX+S)⁺·XᵀWX) on centered coordinates | the standard tr(F) definition |
| `estimate_p` (Tweedie index) | end-to-end on CPG data (probe C3) vs mgcv `tw()` (pinned) | p̂ 1.4990 vs true 1.5 (mgcv reports p=1.5); published φ 2.4128 vs mgcv scale 2.4191 (0.26 %) |
| Pearson-φ residual d.f. contract | `solvers/dispersion.py:12-36` | `sum(w)` for frequency-weighted families, `len(y)` for Tweedie EDM prior weights — matches the documented weight semantics; BIC's `_likelihood_size` (`inference/metrics.py:486-494`) uses the same helper, consistently |
| Poisson nll = deviance/2 shortcut in the objective | `objective.py:324-326` | valid up to λ-independent constants (Poisson l̃ is φ-free) |
| SCOP EFS φ | `reml/scop_efs.py:620-637` | uses the objective's profiled scale; not the efs.py form |
| NB θ profile CI | `profiling/nb.py:541-599` | fixed-μ̂ profile (documented as such); inherits B1/B2 through θ̂ and μ̂ but is internally sound |

### 5.2 Gaps and inconsistencies (beyond A/B/C1)

**C2 — information criteria.** `aic = −2·ll(μ̂, φ̂) + 2·edf`
(`inference/metrics.py:481-483`); no term for estimated φ, θ, or p, and edf is
plain tr(F). Convention check (all pinned/verified this audit): R counts the
Gaussian σ (glm logLik df = mean-params + 1); statsmodels does not
(`res.aic ≡ −2llf + 2k` verified by direct run); mgcv's conditional AIC uses
the WPS-corrected edf and its docs describe the scale entering the likelihood,
not the df. So superglm's *within-family* AIC is inside established practice.
The genuine exposure is **cross-family and cross-entry-point comparisons**:
Poisson vs `NegativeBinomial("auto")` differs by an estimated θ that costs 0;
a fixed-p Tweedie vs `estimate_p` differs by an estimated p that costs 0. The
WPS 2016 correction (their motivating "well known problem with AIC" for models
with estimated smoothing) is additionally absent — a documented approximation,
not a 0.29.0 item (§6).

**C3 — three published φ's for Tweedie.** After `fit_reml`, φ = Pearson with
n−edf (`model/reml_finalize.py:718-719` → `solvers/irls_direct.py:2782-2790`);
after `estimate_p`, φ = the profile MLE re-profiled at the public mean
(`model/profile_ops.py:492-560`); under QP passthrough, φ = D_p/(len(y) − M_p)
(`reml_finalize.py:722-733`). Measured on one dataset (37 % zeros): 2.3044 vs
2.4128 (4.7 %), mgcv's default (a Fletcher-2012-improved Pearson per its
`gam.scale` docs) 2.4191. All three superglm variants are defensible estimators
individually; publishing different ones by entry point is not. Document in
0.29.0; unify (recommendation: the exact MLE, which the repo already computes)
in a later release.

**C5 — absences (state, don't fix).** No quasi-Poisson/quasi-binomial path
exists (`grep quasi` over `src/` finds only separation diagnostics); Poisson
and Binomial pin φ = 1 with no overdispersion escape hatch other than NB2 —
the guide should say "use NB2 / Tweedie for overdispersed counts" explicitly.
`Binomial` is Bernoulli-only by contract (`distributions.py:228-233`,
`validate_response` rejects non-{0,1}); grouped-binomial users must expand
rows. Custom estimated-scale distributions silently inherit the reduced scale
branch (§2.1) — after the A fix, that branch should either raise or warn.

**C7 — clamps and epsilons.** The sweep looked for guards that bias estimates
rather than merely protect divisions. Found and measured: the NB θ clamp (§3.4,
the worst offender in the codebase); the EFS λ clip [1e-6, 1e10]
(`efs.py:269`, dead path); the direct-path working-infinity λ bounds (standard
practice; Wood's "working infinity" treatment). Not found: any
scale-proportional jitter inside the REML determinants (the class of the prior
logdet-tilt incident) — `d_plus_pq = max(Dp, 1e-300)` and the φ floors
(1e-10) sit ~290 and ~8 orders below realized values in every probe run.
`clip_mu`/`stabilize_eta`/`_VARIANCE_FLOOR` bind only in separation regimes,
which are already warned on (`fit_ops.py:1399-1410`).

**C6 — the MASS attribution.** `profiling/nb.py:15` lists
"MASS::glm.nb and MASS::theta.ml source code" as a reference (further mentions
at `nb.py:3,285,357,362,450`, echoed in
`docs/audit/2026-07-28/subsystems/families-profiling.md:59,61`). MASS is
GPL-2\|GPL-3; superglm is MIT (`LICENSE`, `pyproject.toml:7`). A written claim
of derivation from GPL source is exactly what a clean-room provenance record
must not contain — independent of whether any source was in fact read.
Recommended wording: cite **Venables & Ripley (2002), ch. 7.4** for the
alternating scheme (correct, keep) and **Lawless (1987)** for the NB2 profile
score/information; describe the algorithm ("alternating GLM fit and safeguarded
Newton on the closed-form profile score") without naming any implementation's
source; drop the word "source code" everywhere, including the 2026-07-28 audit
doc echo. "MASS-style" in `docs/guide/families.md:124,132` (naming an algorithm
by its best-known implementation) is acceptable but simplest to sweep in the
same pass.

---

## 6. Recommended 0.29.0 scope

**In (ranked):**

1. **B2 — safeguarded, data-started NB θ estimation** with honest convergence
   and loud bounds (§3.4). Regression fixture: the θ_true = 0.05 dataset above,
   asserting θ̂ ≈ 0.05 (mgcv-pinned 0.05068), plus a θ_true = 150 fixture for
   the upper end. These must be **new fixtures that fail on 0.28.0** — per this
   repo's history, a test that would pass against unfixed code is not evidence.
2. **A — exact Tweedie scale profile** in `reml/scale.py` (§2.6), removing the
   `objective.py:310-314` fallback for Tweedie; the fallback should then
   warn-or-raise for unknown custom estimated-scale families. Verification by
   equivalence, not accuracy: the probe-A2 designs with mgcv-pinned τ²/edf as
   fixtures (`ident_p15`: τ² 0.0658, RE edf 23.4; `flat_p18`: τ² 8.99e-4, RE
   edf 0.79; smooth p=1.5 φ=2: edf ≈ 7.3), tolerances a few percent.
   Release-notes text: every Tweedie `fit_reml` result moves; identified terms
   by ≲2 %, weakly identified variance components potentially by orders of
   magnitude (toward the mgcv-agreeing answer); #339 closes against this.
3. **B1 — post-REML θ re-estimation to the joint fixed point** (§3.5).
   Fixture: the hi-freq design, θ̂ within a few percent of 1.0 (mgcv-pinned
   0.99574).
4. **C6 — attribution rewording** (§5.2). One small text PR, no behavior.
5. **Documentation**: the three-φ inventory (C3) and the AIC counting position
   (C2) stated in the families/metrics guide; "no quasi families" stated.

**Deliberately out, and why:**

- **WPS-2016 joint Newton for θ/p/φ inside the LAML** — the correct end state,
  but B1's alternation removes the measured bias at a fraction of the surface
  area; the joint Newton wants the W(ρ)-derivative machinery audit as a
  prerequisite.
- **The WPS AIC smoothing-uncertainty correction and family-parameter df** —
  behavior change to every reported AIC; needs its own design decision
  (which convention, given R/statsmodels/mgcv split three ways). Documenting
  the current convention is 0.29.0's job.
- **Unifying the published Tweedie φ** — entangled with downstream consumers
  of `result.phi` (screening T-scores, `model/screening_ops.py:839`; summaries);
  document now, unify with its own measured before/after.
- **C1 (EFS φ)** — dead code; fix-or-delete belongs to whichever release
  revives λ₁ + REML, with the probe-C1 numbers as its spec.
- **Fletcher (2012) scale option** — nice-to-have parity with mgcv's default,
  not a correctness item.

---

## 7. Literature position (consolidated)

| object | established method | assumptions | are we inside them? |
|---|---|---|---|
| φ-profiled LAML for penalized GLMs | Wood (2011), JRSS-B 73(1):3–36, Eq. (4); restated as V(θ,φ) with exact l̃ in Wood, Pya & Säfken (2016), JASA 111(516):1548–1563, §3.3 | Fisher-regular likelihood; exact saturated likelihood available | Yes. superglm implements it exactly for Gaussian/Gamma (verified §5.1) and substitutes a Gaussian-shaped l̃ for Tweedie (Finding A) |
| Tweedie density/deviance, 1<p<2 | Dunn & Smyth (2005), Statistics & Computing 15:267–280 (series); Jørgensen (1997), *The Theory of Dispersion Models* (EDM structure, zero atom); WPS 2016 SA J (stable derivatives) | series/saddlepoint evaluation stable in the supported range | Yes — superglm's own `tweedie_logpdf` passes the A0 identities; the exact criterion needs no new machinery |
| NB θ in unpenalized GLMs | Venables & Ripley (2002), ch. 7.4 — alternate GLM fit with profile-score Newton; NB2 ML theory incl. score/information: Lawless (1987), Canad. J. Statist. 15(3):209–225 | fixed design/model between alternations; a *globalized* 1-D solve | Alternation: yes. Globalization: **no** — fixed start 1.0 + unsafeguarded Newton + silent clamp (B2) |
| NB θ in penalized models | WPS 2016; mgcv `nb()` docs: "θ estimated alongside the smoothing parameters by ML or REML" | θ inside the LAML Newton | No — θ frozen pre-REML at λ₂=0.1 (B1); measured cost §3.3 |
| Fellner–Schall updates | Wood & Fasiolo (2017), Biometrics 73(4):1071–1081; σ̂² = RSS/(n − edf) | φ from an n−edf (or equivalently the Eq.-4 profile) estimate | Update formula: yes (fixed point verified to 0.1 %). φ: no — rank-based divisor + len(y) (C1, dead path) |
| Scale estimators for reporting | Pearson/deviance/Fletcher (Fletcher 2012, Biometrika 99(1):230–237), per mgcv `gam.scale` docs (default fletcher) | — | Pearson (fit_reml) and exact MLE (estimate_p) both inside practice; the inconsistency is superglm-local (C3) |
| AIC for GAMs | WPS 2016 corrected conditional AIC; convention on counting dispersion split across R (counts) / statsmodels (doesn't) / mgcv (corrected edf) | — | Within-family: inside practice. Cross-family with estimated θ/p: outside all three (nobody else offers that comparison uncorrected) |

Nothing in this audit is claimed as new territory: every fix recommended above
is an application of the cited methods. The searches run: WPS 2016 and
Wood & Fasiolo 2017 full texts (alphaXiv); mgcv 1.9.3 `negbin`, `logLik.gam`,
`gam.scale` documentation (public docs); statsmodels 0.14.6 source/run (BSD-3);
R 4.5.0 / mgcv 1.9.3 / MASS 7.3.65 black-box runs pinned in §2.5/§3.3/§3.4/§5.1.

---

## 8. What this audit could not settle

1. **The certification gate on flat Tweedie cells.** `fit_reml` on the
   `flat_p18` design raises `ObservedModeNotCertifiedError`
   (`reml/direct.py:575`) — the mode-certification threshold trips exactly in
   the flat-criterion regime where Finding A bites hardest, so some of the
   worst-affected fits currently *refuse* rather than mis-publish. Whether the
   A fix changes certification behavior (the exact criterion moves the λ path
   the certifier walks) was not tested. The A fix's fixture set must include
   this design to find out.
2. **Direction of the #339 field case.** The synthetic cells reproduce the
   magnitude class (13–22×) but in the over-shrink direction; #339's field fit
   under-shrank (τ 17× high) on a weighted, offset, partially separated design.
   The mechanism (flat criterion + wrong scale profile) is confirmed as the
   common cause via the mgcv agreement, but this audit did not build a synthetic
   cell reproducing the under-shrink sign on a RandomEffect specifically; only
   the A1 zero-heavy smooth cells show it. Closing #339 should re-run the
   original reproduction against the fix.
3. **Cost of the exact profile in the REML inner loop.** Each criterion
   evaluation gains a 1-D φ solve whose per-iteration cost is a positive-rows
   series-density pass. No timing was performed (it would need the
   benchmark-lock protocol); if it proves material, the fit-invariant reduction
   (zero/positive split precomputed once, saddlepoint warm starts) is the
   designed fallback, not a re-derived criterion.
4. **`estimate_p` under the A fix.** The p-search's REML candidates currently
   embed the reduced criterion, so p̂ could move slightly once A is fixed. The
   one measured cell (p̂ 1.4990 vs mgcv 1.5) suggests the effect is inside the
   search's own resolution, but it was not swept across zero fractions.
5. **How far B1's bias can go.** −45 % was the worst *constructed* cell
   (hi-frequency truth vs the λ₂ = 0.1 calibration). Nothing bounds it in
   principle — any data whose structure λ₂ = 0.1 cannot follow converts misfit
   into spurious overdispersion at the freeze point. The joint fixed point
   removes the question rather than bounding it.
