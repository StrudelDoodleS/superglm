# SuperGLM Roadmap

Status: Living document
Last strategic review: 2026-09-07

This document records current direction, not commitments.

Priority order may change as benchmarks, implementation experience,
published research, or user needs change. A roadmap item is not an
implementation specification and should not be started without a scoped
issue/design.

Legend:
- ACTIVE — currently being worked on
- READY — sufficiently understood to scope
- RESEARCH — requires investigation before implementation
- NEXT — likely near-term candidate
- DEFERRED — intentionally not being pursued now
- KILLED — investigated and rejected unless new evidence emerges
- SHIPPED — completed

# SuperGLM Feature Roadmap: A Prioritized, Sourced Dossier

## TL;DR
- The single largest, highest-leverage build is a **discrete/marginal cross-product assembly layer for multiple linear predictors** (gap #1). mgcv already ships the exact low-level primitive (`XWXd` with `lt`/`rt` term-subset arguments and the `"lpip"` attribute — verbatim: *"The 'lpip' attribute of X is a list of the coefficient indices for each term. Required if subsetting via lt and rt"*) but does NOT wire it to multi-predictor LSS families, so this is a clean-room engineering-plus-integration build, not open research and not a pure port.
- **STEP 1's headline hypothesis is REFUTED**: `mssm` (Joshua Krause, Borst & van Rij; PyPI v1.2.5, 18 Aug 2026, GPL-3.0) does distributional regression with smooths, automatic EFS smoothing selection, and likelihood-based joint inference in Python today. SuperLSS's real differentiators are its MIT license, its actuarial family set (Tweedie-LSS, GPD, generalized Gamma, two-piece), and its scoring/portfolio/refusal machinery — not being "the only one."
- The transcript's "killer combination" (adaptive conditional-exponential-family GAM + distributional PCA + functional compiler) is **tier-(c) research/moonshot at best**: the per-row normalizing-constant quadrature `A(η(x))` reintroduces an `O(n·Q·R)` cost per Newton step that the discrete method cannot remove, defeating the very speed argument used to justify it. Build the functional compiler (f) alone; defer (a)+(b).

## Key Findings

### Landscape audit (STEP 1), verified
- **The closest Python competitor is `mssm`** (`JoKra1/mssm`, PyPI v1.2.5 released 18 Aug 2026, **GPL-3.0**). Per arXiv 2506.13132 (Krause, Borst, van Rij, "The Mixed-Sparse-Smooth-Model Toolbox (MSSM)") it implements `GAMM`, `GAMMLSS`, and general smooth models (`GSMM`); multiple distribution parameters each with their own additive predictor; EFS (Wood & Fasiolo 2017) and the new L-qEFS quasi-Newton update (arXiv 2606.26804 "Structured Secant Methods…", v2 dated 20 Aug 2026, which names *"the open-source mssm Python toolbox (v ≥ 1.2.5)"*); and Bayesian/likelihood joint inference (β|y ≈ N(β̂, H⁻¹)). **This directly does what SuperLSS does.** Three crucial gaps remain in `mssm`: (1) it is **sparse-Cholesky/QR based, NOT discretized-covariate** — the Wood-Li-Shaddick-Augustin 2017 method is only its mgcv *comparison* baseline, not implemented; (2) its shipped GAMMLSS families are only three — verbatim from arXiv 2506.13132 §3.3.1: *"The GAMMLSS class in MSSM … currently supports Gaussian (mean and standard deviation), Multinomial, and Gamma (mean and scale) distributions"* — **no Tweedie-LSS or GPD built in** (Tweedie-LSS appears only as a qEFS *example* in 2606.26804 §5); (3) GPL license blocks commercial actuarial adoption.
- **Discretized "gigadata" methods** (Wood, Li, Shaddick, Augustin 2017 *JASA* 112:1199; Li & Wood 2020 *Stat. Comput.* 30:19) exist in Python **only** in SuperGLM's scalar path — confirmed against `mssm`, `pygam`, `statsmodels` GLMGam, `glum`, `tabmat`, `interpret`. None discretize covariates for marginal cross-products.
- **CRITICAL SUB-QUESTION answered.** mgcv's `XWXd(X,w,k,ks,ts,dt,v,qc,...,lt=NULL,rt=NULL)` (source `R/misc.r`) computes `XᵀWX` from marginal discretized matrices; its `lt`/`rt` arguments "use only columns of X corresponding to these model matrix terms" and require the `"lpip"` attribute. **This is precisely the mechanism to build cross-predictor blocks XⱼᵀWⱼₖXₖ with a per-pair weight vector.** The method's own design intent (Li & Wood 2020 abstract, verbatim): *"we do not rely on discretizing covariates jointly, which would typically require the use of very coarse discretization. The most expensive computation in model estimation is the formation of the matrix cross product XᵀWX where X is a model matrix and W a diagonal or tri-diagonal matrix."* However: the multi-linear-predictor family `twlss` (Tweedie LSS) is documented "Can only be fitted using EFS method," and per `family.mgcv` the LSS/general families are "most of which are currently only usable with `gam`, although some can also be used with `bam`." So mgcv **has the primitive but does not expose a discrete fitting path for multi-predictor LSS**. Gap #1 is therefore: port the published `XWXd`/`lt`/`rt` primitive (clean-room, since mgcv is GPL) and do the genuine integration research to assemble the *coupled* penalized Hessian and the EFS traces from it. Not pure research; not a pure port.
- No INLA-style nested Laplace, no TMB/RTMB AD-tape+sparse-Laplace, no mlt/tram most-likely-transformation, no VGAM reduced-rank VGLM, no GJRM copula additive, no gss SS-ANOVA conditional density, no generalized linear array models, no qgam non-crossing quantile GAM, no SCAM shape-constrained additive models — **all confirmed absent from Python** (checked `statsmodels`, `pygam`, `glum`, `interpret`, `ngboost`, `lightgbmlss`, `xgboostlss`, `skpro`, `mapie`, `scoringrules`, `chainladder`, `formulaic`, `scikit-misc`, `mssm`). Python GLMM support is genuinely weak: `statsmodels` offers only `BinomialBayesMixedGLM`/`PoissonBayesMixedGLM` with **independent random effects only** and Laplace/VB posteriors; there is no arbitrary-random-effect AD+Laplace equivalent to `TMB`/`glmmTMB` in Python.
- Genuinely strong Python competition to respect: `glum`+`tabmat` (fast sparse/categorical GLM, already a SuperGLM dependency), `interpret` (EBM/GA2M with FAST interaction detection, MIT), `ngboost`/`lightgbmlss`/`xgboostlss` (distributional GBMs — the predictive bar), `scoringrules` (CRPS/energy/variogram; the `scores` package ships twCRPS per Allen, Ginsbourger & Ziegel 2023, *"Evaluating forecasts for high-impact events using transformed kernel scores,"* SIAM/ASA J. Uncertainty Quantification 11(3):906–940, doi:10.1137/22M1532184), `chainladder` (reserving).

### Repository grounding (verified against the code)
SuperGLM is MIT-licensed (LICENSE: "Copyright (c) 2026 Max Hicks"), `pyproject.toml` version 0.30.0, deps `numpy/scipy/pandas/narwhals/numba/scikit-learn/joblib/tabmat/threadpoolctl`. The distributional engine (`src/superglm/distributional/__init__.py`) is confirmed to export: family adapters (`gamma`, `gaussian`, `generalized_gamma`, `generalized_pareto`, `log_normal`, `negative_binomial`, `tweedie`, `two_piece`); `Predictor`; the `family.py` contract stack (`ConfigurableDistributionalFamily`, `ParameterSupport`, `AtomFamily`, `ExpectedShortfallFamily`, `VarianceFamily`, `PriorWeighted*` variants); `weights` (prior vs frequency semantics); `posterior` (draws, covariance, simultaneous critical value); `residuals`; `checks` (`qq`, `worm`, `pit`, `binned`, `calibration`, `scores` including `crps_closed_form`, `crps_numeric`, `threshold_weighted_crps`, `murphy_diagram`); `terms` (`term_test`, `summary_table`); `surfaces` (`portfolio`, `risk_curves`, `density_fan`). README confirms "SuperLSS currently refuses `discrete=True` until its multi-parameter route is complete." This is the architecture every dossier below maps onto.

## Details — Candidate Dossiers

Rubric axes, each 1–5: **(1) Novelty in Python** (5 = nothing usable exists); **(2) Statistical value**; **(3) Predictive upside vs GBMs**; **(4) Speed/scale upside** (asymptotic, not constant-factor); **(5) Feasibility** (5 = few person-months, low risk); **(6) Risk of subtle wrongness** (5 = high risk). All complexity in n rows, K linear predictors, per-predictor coefficient counts qₖ, marginal grid sizes mₖ.

---

### C1. Discrete marginal cross-product assembly for multiple linear predictors — *the flagship*
**Pitch.** Give SuperLSS a `discrete=True` path so a K-predictor Tweedie/GPD/gaulss fit assembles its coupled penalized Hessian from marginal discretized bases instead of dense `n×q` matrices, removing the `n` factor from the dominant cost.

**What it unlocks.** Portfolio-scale distributional pricing (10⁷–10⁸ rows) with full likelihood inference — impossible in Python today; `mssm` is sparse but not discretized, and mgcv does not expose discrete multi-predictor LSS.

**Core mathematics.** For general smooth models (Wood, Pya, Säfken 2016 *JASA* 111:1548), the penalized Hessian couples predictors through blocks `Hⱼₖ = Xⱼᵀ Wⱼₖ Xₖ`, where `Wⱼₖ = diag(∂²ℓ/∂ηⱼ∂ηₖ)` is the per-observation cross-weight from the family's `l2` array (mgcv's `gamlss.gH` assembles exactly these from per-observation derivative arrays l1–l4). Under **marginal** discretization, each covariate is discretized once into a unique-value matrix `X̄` plus an index vector `k`; the cross-product `Xⱼᵀ Wⱼₖ Xₖ` is computed directly on `X̄` via the `XWXd` recurrence with `lt=j, rt=k` (the `"lpip"` attribute supplies the per-term coefficient indices that make `lt`/`rt` subsetting possible). When predictors j,k share a covariate's grid, the block reuses one `X̄`; when grids differ, the two index columns `ks[j], ks[k]` drive a general two-index summation. **Complexity:** dense assembly is `O(n·(Σqₖ)²)` (the current 9.7s, 192-coef benchmark). Marginal-discrete assembly is `O(Σⱼₖ (n·cⱼₖ + Πmargin-basis))` where the `n` term carries only a small per-unique-value constant `cⱼₖ` (a few basis columns); Li & Wood (2020) report a dataset-specific 30× cross-product speedup (abstract, verbatim: *"a 30 fold reduction in cross product computation time for the Black Smoke model dataset"* — that figure is for the UK Black Smoke data, not a universal constant). Memory drops from `O(n·Σqₖ)` to `O(Σn·(#margins) + Σmₖ·basis)`. The EFS/LAML objective needs `tr(S_λ⁻ Vβ)`-type leverages, not the full inverse — obtainable from the sparse Cholesky of the assembled `H+S_λ` and `diagXVXd` (mgcv exposes this exact routine), so the coupled blocks are formed once per outer step, not per trace.

**Discretization error.** Marginal (not joint) discretization keeps error `O(mₖ⁻²)` per margin (Wood et al. 2017 show it propagates negligibly into λ and Vβ at fine grids); SuperGLM should bound it by refitting at doubled `n_bins` and asserting coefficient/λ movement below a dimension-scaled tolerance — the project's existing "byte-identical refit receipt" discipline extends directly.

**References.** Wood, Li, Shaddick, Augustin (2017) *JASA* 112:1199, doi:10.1080/01621459.2016.1195744; Li & Wood (2020) *Stat. Comput.* 30:19, doi:10.1007/s11222-019-09864-2; Wood, Pya, Säfken (2016) *JASA* 111:1548. Currie, Durban, Eilers (2006) generalized linear array models *JRSSB* 68:259 for the Kronecker/array arithmetic.

**Existing implementations.** mgcv `XWXd`/`XWyd`/`Xbd`/`diagXVXd` (GPL-2/3, `R/misc.r`, `discrete.c`) — **clean-room reimplementation required** from the two papers. No permissive-licensed version anywhere.

**Implementation sketch.** New module `distributional/discrete/`: `MarginalDesign` (per-covariate unique-value matrix + index), `CrossBlockAssembler` (the `lt`/`rt` recurrence, Numba/C++), wired into the existing EFS objective in `distributional/smoothing/`. Reuse the scalar `discrete=True` assembly already shipped; the new work is the *cross-predictor* weight vectors `Wⱼₖ` fed from each family adapter's existing `l2` Hessian arrays. **Hardest sub-problem:** the constraint (sum-to-zero) reparameterization `qc`/Householder `v` must be applied consistently to a basis that appears in two predictors — get this wrong and Vβ is silently biased.

**Effort.** 6–9 person-months; needs numerical-linear-algebra + P-spline expertise. **Fails loudly:** refit-at-finer-grid receipt must match within tolerance or raise; Cholesky of indefinite `H+S_λ` must raise (perturbed-Cholesky certificate), never silently pivot.

**Rubric.** Novelty 5 | StatValue 5 | Predictive 3 | Speed 5 | Feasibility 2 | Risk 4. — Uniquely absent in Python, the only *asymptotic* win here, but the coupled-constraint reparameterization is the classic place to get Vβ subtly wrong.

---

### C2. Sparse AD-tape + Laplace core (a permissive TMB/RTMB-equivalent)
**Pitch.** A reverse-mode AD tape over a sparse-Cholesky Laplace marginal likelihood, so any new family or random-effect structure needs only a log-likelihood, not hand-derived l1–l4 arrays.

**What it unlocks.** Arbitrary Gaussian random effects (Python has none — `statsmodels` does independent REs only), and it collapses the marginal cost of *every* future family (C6, C8, C13, C14). This is the classic "one layer unlocks four things."

**Core mathematics.** Marginal likelihood `L(θ)=∫ p(y|β,θ)p(β|θ)dβ ≈ exp(ℓ(β̂,θ))·|H/2π|^{-1/2}` with `H=−∂²ℓ/∂β∂βᵀ` sparse; the Laplace gradient needs `∂/∂θ log|H|=tr(H⁻¹ ∂H/∂θ)` computed via sparse Cholesky and selected inversion (Takahashi). Complexity is set by the Cholesky fill of `H`, `O(n·bandwidth²)` for banded P-splines.

**References.** Kristensen, Nielsen, Berg, Skaug, Bell (2016) "TMB: Automatic Differentiation and Laplace Approximation" *J. Stat. Soft.* 70:1, arXiv:1509.00660; Wood, Pya, Säfken (2016).

**Existing implementations.** TMB/RTMB (GPL), `glmmTMB` (AGPL/GPL) — clean-room from paper. Permissive AD building blocks exist (JAX/PyTorch are GPU-first and violate the CPU-structure constraint; `sympy`/`numba` hand-tape is the CPU-only route). **Speculation (labelled):** a Numba-backed reverse tape restricted to the family log-density (small, per-observation) plus analytic sparse-Cholesky for the β-block is feasible without a general AD framework.

**Implementation sketch.** `core/adtape/` producing l1/l2 arrays consumed by the *existing* `gamlss.gH`-style assembler; the Laplace `log|H|` derivative reuses the C1 sparse Cholesky. **Hardest sub-problem:** selected inverse / `tr(H⁻¹∂H)` at scale without forming `H⁻¹`.

**Effort.** 9–12 pm; AD + sparse-linear-algebra expert. **Fails loudly:** finite-difference gradient check against the tape at every family-registration; refuse if Hessian indefinite at the mode.

**Rubric.** Novelty 5 | StatValue 5 | Predictive 3 | Speed 3 | Feasibility 2 | Risk 4.

---

### C3. Robust indefinite-Hessian EFS/Newton with certified stationarity (fixes the two named failures)
**Pitch.** Trust-region + perturbed-Cholesky step control and a stable multi-penalty reparameterization, plus a real stationarity certificate replacing `practical_plateau`.

**Diagnosis of the two failures (labelled analysis).**
- *Correlated Tweedie interaction fit:* three competing hypotheses — (a) EFS oscillation (Wood & Fasiolo 2017 *Biometrics* 73:1071 document EFS convergence conditions and that λ-updates can fail to improve), (b) a flat power-direction (the Tweedie profile likelihood in p is famously flat near the data-implied p), (c) ill-conditioning from correlated tensor terms. **Distinguishing diagnostic:** monitor the EFS λ-trajectory (oscillation → (a)), the curvature of the profile in p at the plateau (flat → (b)), and the condition number of each `Hⱼⱼ` block after reparameterization (high → (c)). Literature fix for (a): damped/half-stepped EFS (mgcv's `fast.REML.fit` step-fail logic); for (b): profile p on a grid and fix it, or reparameterize to `log((p−1)/(2−p))`; for (c): Wood (2011) *JRSSB* 73:3 stable similarity-transform reparameterization for penalties on wildly different scales.
- *GPD smoothing-cap reference fit:* the cap is almost certainly **masking an unbounded likelihood direction** — GPD log-likelihood is irregular as shape ξ→ near/above 0.5 and can be unbounded. **Standard remedy (proven):** penalized MLE of **Coles & Dixon (1999)** "Likelihood-Based Inference for Extreme Value Models," *Extremes* 2(1):5–23, doi:10.1023/A:1009905222644, which restricts ξ<1 and shrinks toward smaller ξ via an exponential penalty on the shape; equivalently a PC-prior on ξ (Opitz et al. 2018) shrinking toward the ξ=0 exponential-tail model. Replace the cap with this penalty and the direction becomes bounded.

**`practical_plateau` verdict.** Defensible as a *stopping heuristic* but not as a certificate. A certified stationarity test costs one extra factorization per candidate optimum: check `‖∇_λ V‖ < tol·scale` and that the Laplace-REML Hessian in log-λ is positive definite. Roughly +5–10% wall-clock; worth it for the model-risk pack.

**References.** Wood (2011); Wood & Fasiolo (2017); Wood, Pya, Säfken (2016); Coles & Dixon (1999).

**Existing implementations.** mgcv `newton`/`fast.REML.fit`/`gam.reparam` (GPL) — clean-room. **Effort.** 3–5 pm. **Fails loudly:** already the culture; add "refuse to certify" as a distinct return state from "converged."

**Rubric.** Novelty 4 | StatValue 4 | Predictive 2 | Speed 2 | Feasibility 4 | Risk 3.

---

### C4. Cross-predictor shared smooths and penalties (an `id=` mechanism across predictors)
**Pitch.** Let one smooth (or its λ) be shared between, e.g., the mean and dispersion predictors, and support double-penalty null-space shrinkage per term across the multi-predictor model.

**What it unlocks.** Scientifically valuable models currently unreachable: a common shape function shared by frequency and severity; equal smoothness across parameters; term-selection (a smooth shrunk to zero) inside a distributional fit.

**Core mathematics.** Sharing a basis across predictors adds off-diagonal coupling to the penalty `S_λ` and changes the EFS update (the shared λ's Fellner-Schall ratio sums traces from both blocks). Centering must be imposed **once** on the shared basis or the two appearances become collinear. Double penalty (Marra & Wood 2011 *CSDA* 55:2372): add a null-space penalty `S⁰` so a term can be shrunk to exactly zero.

**References.** Wood (2017) *GAMs: An Introduction with R* (2e), `id=` mechanism; Marra & Wood (2011). **Existing:** mgcv `id=` and `select=TRUE` (GPL) — clean-room. Note mgcv's own caveat: under `discrete=TRUE` "the discrete method cannot force the linked bases to be identical" — a real interaction with C1 to design around.

**Implementation sketch.** Extend `Predictor` to accept a shared-smooth handle; the EFS objective in `distributional/smoothing/` sums cross-block traces for shared λ. **Hardest sub-problem:** identifiability/centering of a basis living in two predictors (this is also C1's hardest sub-problem — build them together).

**Effort.** 3–4 pm. **Fails loudly:** rank-check the combined penalty null space; raise on collinear shared bases.

**Rubric.** Novelty 5 | StatValue 4 | Predictive 2 | Speed 1 | Feasibility 3 | Risk 4.

---

### C5. Shape constraints under a distributional likelihood (SCAM/SCOP-style)
**Pitch.** Monotone/convex constraints on any predictor — including the dispersion or shape predictor — via re-parameterized (exponentiated-coefficient) P-splines compatible with EFS.

**What it unlocks.** Business-mandated monotone relativities inside a distributional model (SuperGLM has this for scalar GLMs via `Constraint.fit.increasing` SCOP/QP, but not for LSS terms).

**Core mathematics.** Pya & Wood (2015) *Stat. Comput.* 25:543: re-parameterize coefficients as `β = Σ exp(β̃)` so monotonicity holds by construction; the penalty acts on `β̃`; EFS/REML carries over but the mapping is nonlinear so Vβ needs the Jacobian. **Constrained dispersion/shape is well-posed** (the link keeps the parameter in-support) and is genuinely useful in pricing — e.g., monotone-in-exposure dispersion. Standard errors under an *active* constraint are one-sided; report constrained intervals, never symmetric Wald.

**References.** Pya & Wood (2015); Pya (2024) `scam` v1.2-21 (GPL, based on mgcv); mgcv 1.9-4 now also ships `scasm` (shape-constrained additive smooth models). cgam (arXiv:1812.07696) uses cone projection — faster but GPL and less EFS-friendly. Clean-room required.

**Implementation sketch.** New constrained basis in `features/spline.py` mirroring the existing SCOP path, exposed to `Predictor`. **Hardest sub-problem:** the exp-reparameterization makes the Hessian indefinite far from the optimum — needs C3's step control.

**Effort.** 4–6 pm. **Fails loudly:** assert monotonicity of fitted values post-fit; refuse symmetric CIs on active constraints.

**Rubric.** Novelty 5 | StatValue 4 | Predictive 3 | Speed 1 | Feasibility 3 | Risk 3.

---

### C6. Conditional / most-likely transformation models (mlt/tram-equivalent)
**Pitch.** `F(y|x)=F_Z(h(y,x))` with a monotone transformation `h` learned by maximum likelihood — a distribution-free alternative to picking a parametric family.

**What it unlocks.** Escapes family choice entirely (relevant when no named family fits a severity tail); handles censoring/truncation via the exact likelihood on `F`. Absent in Python.

**Core mathematics.** `h(y,x)=Σ (a(y)⊗b(x))ᵀϑ` with `dh/dy>0`; likelihood uses only `F` and `F'`, cheap to evaluate. Shift-scale (Siegfried et al. 2023) and multivariate (Klein et al. 2022) extensions exist.

**References.** Hothorn, Kneib, Bühlmann (2014) *JRSSB*; Hothorn, Möst, Bühlmann (2018) *Scand. J. Stat.* 45:110, doi:10.1111/sjos.12291; Hothorn (2020) "Most Likely Transformations: The mlt Package" *JSS* 92(1). **Existing:** `mlt`/`tram`/`basefun` (GPL). Clean-room; heavy synergy with C5 (the monotonicity is the same machinery).

**Implementation sketch.** A `TransformationFamily` adapter implementing the family contract on `(F_Z, h, ∂h)`; reuses posterior/scoring/`Predictor` unchanged. **Hardest sub-problem:** enforcing `dh/dy>0` while keeping the log-likelihood concave.

**Effort.** 5–7 pm. **Fails loudly:** reject non-monotone `h`; refuse tail functionals when `h` is extrapolated beyond observed y.

**Rubric.** Novelty 5 | StatValue 5 | Predictive 4 | Speed 2 | Feasibility 3 | Risk 3.

---

### C7. Non-crossing additive quantile GAM (qgam-equivalent)
**Pitch.** Calibrated Bayesian additive quantile regression with a smooth pinball loss and a non-crossing guarantee.

**What it unlocks.** Direct quantile pricing with honest credible intervals; Python has nothing equivalent (`statsmodels` quantile reg is linear, no smooths, no calibration).

**Core mathematics.** Fasiolo et al. (2021) replace the pinball loss with the ELF smooth loss and calibrate the learning rate 1/σ so credible intervals attain nominal coverage; a location-scale learning-rate lets σ vary with x. Non-crossing is imposed either by monotone-in-u constraints (transcript (d)'s `dQ/du>0`, which is C5 machinery) or by rearrangement (Chernozhukov, Fernández-Val, Galichon 2010).

**References.** Fasiolo, Wood, Zaffran, Nedellec, Goude (2021) *JASA* "Fast calibrated additive quantile regression," arXiv:1707.03307; `qgam` (GPL). **Transcript (d) verdict:** the "transport/quantile GAM" is **largely a transformation model in disguise** — a coherent monotone `Q(u|x)` and a monotone `h(y|x)` are inverse constructions. Recommend building C6 first and exposing quantiles as its inverse, rather than a separate estimator.

**Effort.** 4–6 pm on top of C5/C6. **Fails loudly:** assert non-crossing on the prediction grid; report calibration diagnostic with every fit.

**Rubric.** Novelty 5 | StatValue 4 | Predictive 4 | Speed 2 | Feasibility 3 | Risk 3.

---

### C8. Copula additive frequency–severity (and low-dim multivariate) models
**Pitch.** Joint model of two+ responses (e.g., frequency and severity, or AD/BI/TPPD claim types) with covariate-dependent dependence via a copula, each margin a full LSS model.

**What it unlocks.** Correct frequency–severity dependence and sample-selection/endogeneity corrections — GJRM's domain, absent in Python.

**Core mathematics.** `F₁₂(y₁,y₂|ϑ)=C(F₁(y₁),F₂(y₂);ρ(x))` with `ρ(x)` itself a GAM; penalized joint likelihood with mgcv-style smoothing (GJRM literally uses mgcv's penalty setup + trust-region). Transcript (e)'s Cholesky-parameterized `Σ(x)=L(x)L(x)ᵀ` is the truly-multivariate route (Kock & Klein 2025 *JCGS* 34:1189, arXiv:2306.02711) — but that paper is **Bayesian with shrinkage**; a fast frequentist REML/EFS version for d>3 is genuinely novel and hard.

**References.** Marra & Radice (2017) *CSDA* 112:99; Marra & Radice (2025) *Copula Additive Distributional Regression Using R* (GJRM, GPL); Kock & Klein (2025) *JCGS* 34(4):1189. Clean-room.

**Implementation sketch.** A `CopulaFamily` taking two registered `Predictor` margins + a dependence predictor; reuses C2's AD tape for the copula score. **Hardest sub-problem:** identifiability of `ρ(x)` when margins are also flexible.

**Effort.** 8–10 pm (bivariate); d>3 is research. **Fails loudly:** refuse tail dependence functionals when the copula family cannot represent them.

**Rubric.** Novelty 5 | StatValue 5 | Predictive 3 | Speed 1 | Feasibility 2 | Risk 4.

---

### C9. Automatic model builder: componentwise boosting → penalized-LSS refit
**Pitch.** A structure-discovery stage (which feature acts on which parameter, linear vs smooth, which pairwise interactions) via componentwise/gamboostLSS-style boosting, followed by a refit as a penalized LSS model for inference.

**What it unlocks.** Closes much of the GBM predictive gap while keeping GAM interpretability; per-predictor selection prevents the search spending its whole budget on the mean.

**Core mathematics.** gamboostLSS does simultaneous variable + effect-type + parameter selection by cyclically boosting each predictor's base-learners. FAST (Lou et al., the `interpret` EBM/GA2M line) ranks pairwise interactions cheaply on residuals. `gamsel` (Chouldechova & Hastie) does zero/linear/nonlinear per feature. **Two-stage validity caveat (proven concern):** naive refit-after-selection invalidates p-values/intervals — must use sample-splitting or selective inference (Lee et al. 2016). Recommend: boost on split A for structure, refit + infer on split B, and report intervals as post-selection-honest only under that protocol. This is the transcript's item (g) with strong heredity — recommend enforcing `f_{jk}≠0 ⟹ f_j,f_k≠0` in the candidate generation.

**Search objective.** NCV is already road-mapped (excluded); use out-of-sample deviance/CRPS with incremental base-learner updates so each candidate step is cheap even though a full EFS fit is not.

**References.** Mayr, Fenske, Hofner, Kneib, Schmid (2012) gamboostLSS *JRSS-C*; Thomas et al. gamboostLSS non-cyclical; Lou, Caruana, Gehrke, Hooker (2013) GA2M/FAST *KDD*. `mboost`/`gamboostLSS` (GPL), `interpret` (MIT). Clean-room for the LSS boosting.

**Effort.** 6–8 pm. **Fails loudly:** label any interval computed on the same data used for selection as "selection-inflated, invalid."

**Rubric.** Novelty 4 | StatValue 4 | Predictive 5 | Speed 2 | Feasibility 3 | Risk 4.

---

### C10. Conformal predictive distributions wrapper
**Pitch.** A finite-sample-valid calibration layer (CQR, CV+/jackknife+, Mondrian by segment, distributional conformal) wrapping any fitted SuperLSS model.

**What it unlocks.** Coverage guarantees that the likelihood model alone cannot promise under misspecification — the missing complement to GAM inference.

**Design answer to the brief's headline question.** A coherent split is achievable: **likelihood-based inference reports on *effects* (Wood 2013 p-values, Marra–Wood unconditional Bayesian intervals), conformal reports on *predictions* (marginal coverage).** They do not contradict because they answer different questions; the danger is presenting a conformal predictive interval as if it were a credible interval on a parameter. Rule: never overlay them on the same axis without distinct labels.

**References.** Romano, Patterson, Candès (2019) CQR *NeurIPS*; Barber, Candès, Ramdas, Tibshirani (2021) jackknife+/CV+ *Ann. Stat.* 49:486; Dimitriadis, Gneiting, Jordan (2021) CORP reliability *PNAS* 118:e2016191118. `mapie` (BSD) exists but does not wrap distributional GAMs. Mostly permissive → light clean-room.

**Effort.** 3–4 pm. **Fails loudly:** raise if calibration set too small for requested coverage; refuse Mondrian bins with insufficient counts.

**Rubric.** Novelty 4 | StatValue 4 | Predictive 3 | Speed 4 | Feasibility 4 | Risk 2.

---

### C11. Portfolio aggregation via FFT / Panjer / saddlepoint (no Monte Carlo)
**Pitch.** Aggregate a heterogeneous portfolio's predictive distribution analytically instead of by simulation, with honest refusal when tail moments don't exist.

**What it unlocks.** Fast, exact-to-numerical-tolerance VaR/TVaR at portfolio scale on CPU — the actuarial payoff of the whole stack. Extends the existing `portfolio()`/`actual_expected()`.

**Core mathematics.** Panjer recursion for compound (a,b,0) frequency classes; FFT convolution of discretized severity for the aggregate `S=ΣXᵢ`; saddlepoint approximation `f_S(s)≈(2πK''(ŝ))^{-1/2}exp(K(ŝ)−ŝs)` when the CGF `K` exists. FFT is `O(M log M)` in the discretization size M — genuinely fast on CPU; Monte Carlo is `O(N_sim)` with `√N` error. **Refusal rule (aligns with existing culture):** TVaR/expected-shortfall must raise when the severity's second (or relevant) moment diverges — this is exactly the "finite Monte Carlo variance when the second moment diverges" case the project already refuses. **Threshold-weighted CRPS** (Gneiting & Ranjan 2011; Allen, Ginsbourger & Ziegel 2023, SIAM/ASA JUQ 11(3):906, doi:10.1137/22M1532184) already lives in the codebase (`threshold_weighted_crps`) and is the correct large-loss-focused score to pair with this.

**References.** Panjer (1981) *ASTIN Bull.*; Embrechts & Frei (2009) saddlepoint; Klugman, Panjer, Willmot *Loss Models*. Permissive — clean build.

**Effort.** 4–6 pm. **Fails loudly:** aliasing check on FFT tail; refuse moment-based functionals outside the support of existence.

**Rubric.** Novelty 4 | StatValue 4 | Predictive 2 | Speed 5 | Feasibility 4 | Risk 3.

---

### C12. Inference compiler for arbitrary functionals (transcript (f))
**Pitch.** `model.functional(lambda dist: dist.stop_loss(25000), X, se=True)` returns estimate + delta-method SE + CI for any `Ψ(β,x)=∫ψ(y)p_β(y|x)dy`.

**What it unlocks.** Uniform, honest uncertainty for stop-loss layers `E[min((Y−a)₊,L)]`, VaR/TVaR, entropy, risk differences, feature derivatives — generalizing SuperLSS's existing tail functionals.

**Core mathematics.** `∇_β Ψ` by AD (C2) or hand-derived family gradients; `Var(Ψ)=∇Ψᵀ Vβ ∇Ψ`. **When the delta method is invalid (must fail loudly):** (i) non-existent moments (stop-loss/TVaR on a heavy tail — refuse, per C11); (ii) active boundary/shape constraints (C5) — one-sided; (iii) non-smooth functionals such as **VaR at an atom** (a Tweedie mass at zero) where the quantile is set-valued — refuse or return the interval of subgradients; (iv) functionals whose gradient differentiates through a quadrature or root-find — need the implicit-function theorem, not naive AD.

**References.** standard delta method; Wood (2013) *Biometrika* smooth-term inference; Marra & Wood (2012) *Scand. J. Stat.* 39:53 unconditional intervals. Permissive.

**Effort.** 4–6 pm (much less if C2 exists). **Fails loudly:** the four cases above each raise a typed numerical-domain error.

**Rubric.** Novelty 5 | StatValue 5 | Predictive 2 | Speed 3 | Feasibility 3 | Risk 4. — **This is the transcript's genuinely valuable idea; build it, decoupled from (a)/(b).**

---

### C13. GPD tail-splicing with estimated threshold and penalized shape
**Pitch.** Splice a GPD tail onto a body distribution at a smooth, estimated threshold, with the shape penalized per Coles & Dixon so the fit is bounded.

**What it unlocks.** Principled large-loss modelling — the freMTPL2/actuarial core; directly resolves the GPD failure in C3.

**Core mathematics.** Body `F_B` for `y<u`, GPD `(1−F_B(u))·G_ξ,σ(y−u)` above; continuity at u; threshold u itself a smooth function of x. Penalize ξ toward 0 (Coles & Dixon 1999 exponential penalty on the shape; or PC-prior, Opitz et al. 2018).

**References.** Coles & Dixon (1999) *Extremes* 2:5, doi:10.1023/A:1009905222644; Scarrott & MacDonald (2012) threshold review. Permissive.

**Effort.** 4–6 pm. **Fails loudly:** refuse finite TVaR when ξ̂≥0.5 unless penalty binds; certify threshold identifiability.

**Rubric.** Novelty 5 | StatValue 4 | Predictive 3 | Speed 2 | Feasibility 3 | Risk 4.

---

### C14. Adaptive distributional GAM via conditional exponential family (transcript (a)+(b)) — research/moonshot
**Pitch.** Replace named parameters with `log p(y|x)=log p₀(y)+Σ_r T_r(y)η_r(x)−A(η(x))`, each `η_r` a structured additive predictor, so the model discovers *where in the response* each feature acts.

**Honest quantitative verdict (this is the crux the brief asks for).** The idea is real and has precedent (gss `sscden`/`ssden` penalized-likelihood conditional density, Gu 2014 *JSS* 58(5); the Expxorcist, Suggala, Kolar, Ravikumar NeurIPS 2017 30:4449; Bayesian structured additive density regression). But the **per-row normalizing constant `A(η(x))` requires a quadrature over y for every row, every Newton step, every λ update** — `O(n·Q·R)` with Q nodes and R response-basis terms — and **this n·Q factor cannot be discretized away**, because A depends on the full per-row `η(x)`. So C14 *defeats* the C1 speed argument that the transcript itself invoked to justify it: the transcript's "cross-block count grows like R²" understates the cost by the `n·Q` quadrature factor. The log-partition Hessian is `Cov_{p(y|x)}[T(y)]` (an R×R per row), which is dense and per-row, making the coupled K=R-predictor Hessian expensive, not tractable. **Moment-existence trap:** a finite polynomial T-basis forces a light (sub-exponential) tail — so on heavy-tailed severity it will *silently* impose a thin tail, violating the project's moment-existence rule. That alone disqualifies it as a default actuarial model.

**Transcript (b) distributional-PCA** reduced-rank `η(y,x)=B_y(y)ᵀLz(x)` overlaps VGAM `rrvglm` (Yee & Hastie 2003) and suffers rotation non-identifiability `Lz=(LQ)(Q⁻¹z)` that breaks effect inference unless Fisher-orthogonalized — and Fisher-orthogonalization is ill-defined when `I(θ)` varies with x.

**Verdict.** Tier (c) moonshot, and **only** on bounded/light-tailed responses with a spline (not polynomial) T-basis to keep the tail honest. Do **not** ship as a pricing default. The functional compiler (C12) delivers most of the transcript's promised "where does the feature act" value at a fraction of the risk.

**Rubric.** Novelty 5 | StatValue 4 | Predictive 4 | Speed 1 | Feasibility 1 | Risk 5.

---

## Comparison table

| # | Capability | Nov | Stat | Pred | Speed | Feas | Risk | Tier |
|---|---|---|---|---|---|---|---|---|
| C1 | Discrete multi-predictor assembly | 5 | 5 | 3 | 5 | 2 | 4 | b |
| C2 | Sparse AD-tape + Laplace core | 5 | 5 | 3 | 3 | 2 | 4 | b |
| C3 | Robust EFS/Newton + certification | 4 | 4 | 2 | 2 | 4 | 3 | a |
| C4 | Cross-predictor shared smooths | 5 | 4 | 2 | 1 | 3 | 4 | b |
| C5 | Shape constraints under LSS | 5 | 4 | 3 | 1 | 3 | 3 | a/b |
| C6 | Conditional transformation models | 5 | 5 | 4 | 2 | 3 | 3 | b |
| C7 | Non-crossing quantile GAM | 5 | 4 | 4 | 2 | 3 | 3 | b |
| C8 | Copula additive freq–severity | 5 | 5 | 3 | 1 | 2 | 4 | b/c |
| C9 | Boosting → LSS-refit builder | 4 | 4 | 5 | 2 | 3 | 4 | b |
| C10 | Conformal predictive wrapper | 4 | 4 | 3 | 4 | 4 | 2 | a |
| C11 | FFT/Panjer/saddlepoint aggregation | 4 | 4 | 2 | 5 | 4 | 3 | a |
| C12 | Functional inference compiler | 5 | 5 | 2 | 3 | 3 | 4 | a/b |
| C13 | GPD tail-splicing + penalized shape | 5 | 4 | 3 | 2 | 3 | 4 | b |
| C14 | Adaptive conditional-exp-family GAM | 5 | 4 | 4 | 1 | 1 | 5 | c |

## Dependency graph (which layers unlock which)

```
                 ┌─────────────────────────────┐
                 │ C3 Robust EFS/Newton + certs │  (foundational; fixes 2 failures)
                 └───────────┬─────────────────┘
                             │ underpins every iterative fit
        ┌────────────────────┼───────────────────────────┐
        ▼                    ▼                             ▼
┌───────────────┐   ┌──────────────────┐        ┌────────────────────┐
│ C1 Discrete    │   │ C2 AD-tape +     │        │ C5 Shape           │
│ multi-pred     │   │ sparse Laplace   │        │ constraints (LSS)  │
│ assembly       │   │ core             │        └─────────┬──────────┘
└───┬───────┬────┘   └───┬─────┬────┬───┘                  │
    │       │            │     │    │                       ▼
 shares  scales      C6 trans- C8  C13 GPD        C6/C7 (monotonicity reuse)
 hardest C4/C8/C13   formation copula splice
 sub-prob            models    additive
    │                │     │
    ▼                ▼     ▼
 C4 shared        C7 quantile (= inverse of C6)
 smooths          C12 functional compiler (cheap once C2 exists)

 Independent of the two flagships:
   C11 aggregation · C10 conformal · C9 builder (wants C1 for cheap incremental fits)
   C14 (needs C1+C2+C12; still moonshot)
```

Two layers unlock four-plus things each: **C1** (discrete assembly) shares its hardest sub-problem — cross-predictor constraint/centering — with C4, C8, C13 and makes C9's incremental search affordable; **C2** (AD tape) collapses the marginal cost of C6, C8, C12, C13 and is the only route to arbitrary random effects. **C3** underpins everything and is cheap. This is why C3 comes first and C1/C2 are the twin flagships.

## Sequenced roadmap

**Tier (a) — high-confidence ports of known-good methods (build now, ~6–9 months):**
- **C3** robust EFS/Newton + certified stationarity (fixes both named failures; foundational).
- **C11** FFT/Panjer/saddlepoint aggregation (pure win, actuarial payoff, low risk).
- **C10** conformal predictive wrapper (mostly permissive libraries, coverage guarantees).
- **C12** functional inference compiler (delivers the transcript's real value; even richer once C2 lands).
- **C5** shape constraints under LSS (extends existing scalar SCOP path).

**Tier (b) — genuine research-and-build (the differentiators, ~12–24 months):**
- **C1** discrete multi-predictor assembly *(the flagship — start immediately in parallel with C3)*.
- **C2** sparse AD-tape + Laplace core.
- **C4** cross-predictor shared smooths (co-develop with C1).
- **C6** conditional transformation models → **C7** quantiles as its inverse.
- **C13** GPD tail-splicing + penalized shape.
- **C9** boosting→refit builder.
- **C8** bivariate copula freq–severity (multivariate d>3 slips toward tier c).

**Tier (c) — speculative moonshots (with stated failure modes):**
- **C14** adaptive conditional-exp-family GAM — *failure mode: per-row quadrature `O(n·Q·R)` kills the speed case; finite T-basis silently imposes a light tail, violating moment-existence.* Only attempt on bounded/light-tailed responses with a spline T-basis.
- **Truly-multivariate frequentist REML copula (d>3)** — *failure mode: `Σ(x)=L(x)L(x)ᵀ` identifiability and EFS conditioning at high d are unproven outside the Bayesian setting of Kock & Klein 2025.*

## KILL LIST

- **Transcript (b) standalone "distributional PCA from likelihood geometry."** Reduced-rank `η(y,x)=B_y(y)ᵀLz(x)` is VGAM `rrvglm` territory (Yee & Hastie 2003) and carries the rotation ambiguity `Lz=(LQ)(Q⁻¹z)`, which destroys effect interpretability unless Fisher-orthogonalized — and Fisher-orthogonalization is ill-defined when `I(θ)` varies with x. Non-convex alternating estimation with no inference guarantees. **Kill as a standalone feature**; the low-rank idea can live *inside* C8's dependence structure where identifiability is handled by the copula.
- **Transcript (h) "GAM on the manifold of probability distributions"** (Fisher–Rao exponential map). The transcript itself flags it as a not-to-start moonshot; it has no tractable estimator, no software precedent, and the Exp-map requires the same per-row partition-function quadrature as C14 plus geodesic solves. **Kill.** Failure mode: no finite-time algorithm and no loud-failure story.
- **Full INLA/SPDE nested-Laplace port.** Enormous surface area (R-INLA is GPL and vast), and SuperGLM's domain (heavy-tailed pricing, not spatial latent Gaussian fields) does not need it. The useful 20% (sparse Laplace for random effects) is exactly C2. **Kill the full port; keep C2.**
- **Isotonic distributional regression as a core model.** IDR (Henzi, Ziegel, Gneiting 2021 *JRSSB* 83:963; Python `isodisreg`) is tuning-free and calibrated, but it is a nonparametric benchmark with a partial-order restriction, not a likelihood-inference model — it gives no effect inference and no smooth relativities. **Kill as a core feature; adopt only as a built-in benchmark** in `cross_validate` (a few days' work) to keep SuperLSS honest against a parameter-free competitor.
- **Naive dense scaling to R=5–8 learned predictors** (the transcript's "worth measuring"). Quantitatively: dense assembly is `O(n·(Σqₖ)²)`; going from K=3 to R=8 at qₖ≈64 multiplies the `(Σqₖ)²` factor ~7× *and*, under C14's density formulation, adds the `O(n·Q·R)` quadrature — so the honest projection from the 9.7s benchmark is tens of seconds to minutes, not "worth measuring." **Kill dense R=8**; it is only viable after C1.
- **GPU/JAX/PyTorch GAM backends.** Violate the CPU-only, structure-not-hardware constraint; also GPU-first frameworks give no asymptotic win on the sparse banded problems here. **Kill.**

## The single highest-leverage thing to build next

**C1 — the discrete marginal cross-product assembly layer for multiple linear predictors — with C3's step-control as its immediate prerequisite.**

The argument: (1) It is the one self-identified gap (#1) that is simultaneously *unique in Python* (mssm is sparse but not discretized; mgcv has the primitive but does not expose it for LSS families) and an *asymptotic* win (removes the `n` factor from the dominant `XⱼᵀWⱼₖXₖ` cost, the 9.7s→sub-second path). (2) Its hardest sub-problem — cross-predictor constraint/centering reparameterization — is *shared* with C4, C8, and C13, so solving it once pays down four capabilities. (3) It is the enabler for C9's incremental search to be affordable at portfolio scale. (4) The published primitive (`XWXd` with `lt`/`rt` + `lpip`, Li & Wood 2020) means the algorithm is known and validated; the risk is implementation correctness, which the project's byte-identical refit-receipt culture is uniquely equipped to police. Everything else either rests on it, benefits from it, or is a lower-ceiling tier-(a) port.

## Recommendations
1. **Immediately:** start C3 (target the two failure fixes first) and C1 in parallel; land the GPD penalty (Coles & Dixon 1999) as the first C3 deliverable to unblock the reserving/GPD-cap benchmark.
2. **Next quarter:** ship tier (a) — C11, C12, C10, C5 — each independently valuable and low-risk; they make the package visibly ambitious while C1/C2 mature.
3. **Then:** C2, then C6→C7 and C13 on top of it; C4 co-developed with C1.
4. **Benchmarks/thresholds that change the plan:** if the C1 finer-grid refit receipt cannot match dense Vβ within a dimension-scaled tolerance, *stop and treat the coupled-constraint reparameterization as research* rather than shipping a biased covariance. If C9's split-refit intervals fail coverage on freMTPL2, drop the two-stage inference claim and market it as prediction-only. If C14 is ever attempted, gate it behind a moment-existence assertion on the T-basis and refuse heavy-tailed responses. Before claiming the GBM gap is closed, reproduce Chevalier & Côté (arXiv:2412.14916, "From point to probabilistic gradient boosting for claim frequency and severity prediction," PMC12575580) — which finds "LightGBM stands out as the most computationally efficient with little or no loss in predictive performance" across five claim datasets — as the external yardstick.
5. **License hygiene:** every recommendation above requires clean-room reimplementation from papers because mgcv, gamlss, scam, mlt/tram, GJRM, gss, qgam, TMB/glmmTMB, mboost/gamboostLSS, and mssm are all GPL/AGPL. Do not read their source while implementing; cite the papers. `interpret` (MIT), `mapie` (BSD), `scoringrules`/`scores` are safe to depend on or reference.

## Caveats
- The mgcv "discrete + multi-predictor LSS" status is asserted from the released documentation (`family.mgcv`: LSS families "currently only usable with `gam`, ... some ... with `bam`"; `twlss` "Can only be fitted using EFS method") plus the `XWXd` `lt`/`rt`/`lpip` primitive; I did not execute mgcv 1.9-4 to confirm a hard error, so treat "not exposed for discrete LSS" as strongly-evidenced-but-not-executed.
- Person-month estimates are engineering judgment, not measured; the tier-(b) research items (C8 d>3, C14) could each expand indefinitely.
- `mssm`'s exact feature set is from arXiv 2506.13132 / 2606.26804 and PyPI/GitHub metadata as of Aug 2026; a point release could add a Tweedie-LSS built-in family and narrow SuperLSS's family-set advantage. Its three built-in GAMMLSS families (Gaussian/Multinomial/Gamma) and GPL license are the load-bearing facts for the "SuperLSS is still differentiated" conclusion.
- Predictive-upside scores are directional: no head-to-head SuperLSS-vs-lightgbmlss study on freMTPL2 exists yet; the Chevalier & Côté (2025) actuarial GBM comparison is the closest external benchmark and should be reproduced before any GBM-parity claim.
- The GPD penalty form (Coles & Dixon 1999 exponential penalty restricting ξ<1 and favouring smaller ξ) is confirmed from the paper and secondary sources (e.g., Extremal Random Forests, arXiv:2201.12865); the exact tuning-constant recommendation should be re-derived against SuperGLM's oracle before shipping.
