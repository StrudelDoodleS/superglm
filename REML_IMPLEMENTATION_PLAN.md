# SuperGLM REML Implementation Plan

## Wood-style REML Smoothing Parameter Estimation: Audit, Gap Analysis, and Architecture

**Date:** 2026-03-08
**Scope:** Comprehensive comparison of the current SuperGLM REML implementation against Wood's methodology (Wood 2011, Wood et al. 2016, mgcv source), with a step-by-step refactoring plan.

**Design constraint:** SuperGLM is not a pure GAM library. It combines smooth spline terms with group lasso, group elastic net, sparse group lasso, and ridge penalties for simultaneous feature selection and smoothing. The REML implementation must coexist with these L1-based selection penalties. REML controls within-group smoothness (per-term λ_j); group lasso / elastic net controls between-group selection (λ₁). They are orthogonal.

---

## Critical Implementation Warnings

Three traps identified during senior review that must be addressed before writing code:

### 1. The Cholesky Trap (Step 2)

H = X'WX + S is **frequently rank-deficient** in practice (collinear categoricals, extreme IRLS weights in Binomial/Poisson tails). `np.linalg.cholesky` will hard-crash. **Never remove the eigh code path.** Step 2 implements an armored `_safe_decompose_H` that tries Cholesky first and falls back seamlessly to thresholded eigendecomposition. See Step 2 for the complete implementation.

### 2. The L1 Non-Differentiability Trap (Step 6)

The reason to use Fellner-Schall (not Newton) for the BCD path (λ₁ > 0) is **not** that H⁻¹ is unavailable — `pirls.py` already computes the full p×p inverse for edf. The real reason is that **L1 penalties (group lasso / elastic net) make the REML surface non-differentiable** at active-set boundaries. Newton's exact Hessian assumes a smooth L2 penalty space; it can overreact when groups are being zeroed/activated between REML iterations. Fellner-Schall is a monotone fixed-point iteration that is stable under active-set changes.

### 3. The Estimated Scale Trap (Step 7)

The current REML objective is **mathematically wrong** for Gamma, NB2, and Tweedie families. It omits the `((n-p_eff)/2) · log(φ̂)` term and the `1/φ̂` scaling. This causes estimated λ to diverge 2–5× from mgcv for typical claim severity data. **Step 7 is a critical correctness fix**, not an optimization. It has been elevated to priority 3 in the implementation order.

---

## Table of Contents

1. [Post-PR #2 Priorities](#post-pr-2-priorities)
2. [Phase 1: Assessment of the Existing Codebase](#phase-1-assessment-of-the-existing-codebase)
3. [Phase 2: Gap Analysis](#phase-2-gap-analysis)
4. [Phase 3: Implementation Plan](#phase-3-implementation-plan)
5. [Appendix: mgcv Optimizer Reference](#appendix-mgcv-optimizer-reference)

---

## Post-PR #2 Priorities

With the direct `lambda1=0` REML path in much better shape after PR #2, the next major items should shift from Newton cleanup to the remaining gaps versus `mgcv`.

**Scope note:** This section is a roadmap, not a single-PR scope. These items should be split into several focused follow-up PRs. The recommended next PR is a benchmarking / profiling PR, not a combined implementation of everything below.

### 1. `lambda1 > 0` REML / fREML path

Implement EFS / generalized Fellner-Schall for the BCD active-set path. Do not keep forcing exact Newton through the non-smooth group-lasso path; the right analogue there is a stable monotone update under active-set changes.

### 2. Real large-n fREML / `bam`-style speed path

Build the speed story around `discrete=True`, reuse of sufficient statistics, and performance iteration / simultaneous beta-lambda updates. This is the highest-value practical gap if fit time matters.

### 3. Broaden `mgcv` parity

Add committed parity cases for:
- `NB2`
- `Tweedie`
- `select=True`
- tensor/interactions
- split null/range smooths

Also add direct prediction parity, not just deviance / EDF / scale parity.

### 4. Benchmarking and profiling harness

Maintain a reproducible benchmark comparing:
- `SuperGLM fit_reml(discrete=False)`
- `SuperGLM fit_reml(discrete=True)`
- `mgcv::gam(method="REML")`
- `mgcv::bam(method="fREML", discrete=TRUE)`

Report at least wall time, outer iterations, inner PIRLS iterations (if available), EDF, deviance, and scale on shared datasets with fixed thread counts.

### Suggested PR sequencing

1. PR #3: benchmarking / profiling harness only
2. PR #4: `lambda1 > 0` REML / EFS path
3. PR #5: large-`n` fREML / `bam`-style speed path
4. PR #6: broaden `mgcv` parity
5. PR #7: numerical / inference polish
6. PR #8: refactor `model.py` after behavior stabilizes

### 5. Numerical and inference polish

Add edge correction / near-boundary lambda handling and improve smoothing-parameter uncertainty support in inference, including a clearer conditional vs unconditional covariance story.

### 6. Refactor after optimization work stabilizes

Once the current REML / fREML behavior is settled, split the large `model.py` file by responsibility:
- REML optimizer internals
- inference / covariance / summary helpers
- plotting and reporting helpers

### Non-goals for now

Do not prioritize:
- more quasi-Newton variants
- raw lambda matching as the primary success metric

Prefer deviance, EDF, scale, prediction parity, convergence behavior, and benchmarked runtime.

## Phase 1: Assessment of the Existing Codebase

### 1.1 Model Matrix (X) and Penalty Matrices (S_j) — Construction & Storage

The model matrix is never materialized as a single dense (n, p) array. Instead, `DesignMatrix` (`group_matrix.py:292`) holds a list of per-group `GroupMatrix` wrappers, each providing `matvec`, `rmatvec`, and `gram(W)` operations. Four concrete types:

| Type | Storage | `gram(W)` cost |
|---|---|---|
| `DenseGroupMatrix` | Dense (n, p_g) | O(n · p_g²) |
| `SparseGroupMatrix` | CSR (n, p_g) | O(nnz · p_g) |
| `SparseSSPGroupMatrix` | CSR **B** (n, K) + dense **R⁻¹** (K, p_g) | O(nnz_B · K) via numba |
| `DiscretizedSSPGroupMatrix` | Dense **B_unique** (n_bins, K) + int index (n,) + dense **R⁻¹** (K, p_g) | O(n + n_bins · K²) |

The full **X**ᵀ**WX** is assembled block-by-block via `_block_xtwx` (`group_matrix.py:265`), which computes diagonal blocks via `gm.gram(W)` and off-diagonal blocks via `_cross_gram`. This is the primary Gram computation path for both the direct solver and the REML loop.

**Penalty matrices.** Each SSP group stores:
- `gm.omega`: the (K, K) B-spline-space penalty matrix **Ω** (second-difference D₂ᵀD₂ for P-splines, integrated ∫B_i''B_j''dx for CR splines)
- `gm.R_inv`: the (K, p_g) SSP reparametrization matrix
- `gm.projection`: optional (K, n_sub) subspace projection (for natural/CR constraints, or linear-split subgroups)

The penalty in the reparametrized space is **Ω_ssp** = (**R⁻¹**)ᵀ **Ω** **R⁻¹**. The full block-diagonal penalty **S**(λ) is assembled by `_build_penalty_matrix` (`irls_direct.py:42`):

```
S(λ) = ⊕_j  λ_j · (R_j⁻¹)ᵀ Ω_j R_j⁻¹
```

This matches Wood's formulation structurally. Non-SSP groups (categoricals, numerics) and unpenalized groups contribute zero blocks.

### 1.2 The PIRLS / Direct IRLS Inner Loop

Two solvers exist, auto-dispatched by `fit()` and `fit_reml()`:

**`fit_pirls` (`pirls.py:377`)** — PIRLS with proximal Newton BCD. Used when λ₁ > 0 (group lasso / elastic net / SGL active). Each outer PIRLS iteration:
1. Computes working weights W_i = w_i · (dμ_i/dη_i)² / V(μ_i) and working response z_i
2. Computes per-group Hessians H_g = X_g' diag(W) X_g via `gm.gram(W)`
3. Runs BCD inner cycles: Newton step d_g = H_g⁻¹ ∇_g then proximal operator (group soft-thresholding)
4. Checks deviance convergence

**`fit_irls_direct` (`irls_direct.py:90`)** — Direct penalized IRLS. Used when λ₁ = 0 (no group selection). Each iteration:
1. Computes the same W, z
2. Forms the augmented (p+1) × (p+1) system via gram-based operations
3. Solves via eigendecomposition of M_aug
4. Checks deviance convergence

**Key strength:** The direct solver forms X'WX via `_block_xtwx` (gram-based, never materializes (n, p)) and solves via `np.linalg.eigh` of the augmented system. The (p+1) × (p+1) eigendecomposition is O(p³) per iteration, fast for typical p ~ 50–100.

### 1.3 Current Smoothing Parameter Selection

Smoothing parameters are selected in `fit_reml()` (`model.py:1628`). There are two paths:

**Path A: Direct REML (λ₁=0).** `_optimize_direct_reml` (`model.py:1252`) uses `scipy.optimize.minimize` with L-BFGS-B on log λ. The objective (`_reml_laml_objective`, `model.py:1180`) and gradient (`_reml_direct_gradient`, `model.py:1226`) are the Laplace-approximate REML criterion and its first derivative.

Each evaluation of the objective requires a full `fit_irls_direct` (inner loop convergence), followed by eigendecomposition of both S and X'WX + S for the log-determinants.

**Path B: Fixed-point REML (λ₁>0).** `_run_reml_once` (`model.py:1384`) implements the Wood (2011) fixed-point iteration around PIRLS:

```
λ_j_new = r_j / (β̂_j' Ω_ssp_j β̂_j + tr[(X'WX + S)⁻¹[j,j] Ω_ssp_j])
```

This uses Anderson(1) acceleration on log λ and "cheap iterations" (skip solver/DM rebuild when Δλ < 1%).

---

## Phase 2: Gap Analysis

### 2.1 Mathematical Formulation — Wood's REML Score vs. Current

**Wood's REML criterion** (Wood 2011, eq. 3; Wood 2017, §6.2.2) for a GAM with fixed dispersion parameter φ:

```
V_R(ρ) = -ℓ(y | β̂_ρ) + ½ β̂_ρ' S_ρ β̂_ρ + ½ log|X'ŴX + S_ρ| - ½ log|S_ρ|₊ + const
```

where ρ = log λ, β̂_ρ is the PIRLS solution for that λ, and Ŵ is the working weight matrix at convergence.

**Current objective** (`_reml_laml_objective`, `model.py:1180`) matches this exactly. The four terms correspond one-to-one. The only notational difference is that φ-scaling is omitted for known-scale families — correct since for Poisson/Binomial, φ = 1 and cancels.

**Wood's gradient** (Wood 2011, eq. 6):

```
∂V_R/∂ρ_j = ½ tr[(X'ŴX + S)⁻¹ S_j] - ½ tr[S⁺ S_j] + ½ β̂' S_j β̂
```

where S_j = ∂S/∂ρ_j = λ_j Ω_ssp_j.

**Current gradient** (`_reml_direct_gradient`, `model.py:1226`):
```python
grad[i] = 0.5 * (lam * (quad + trace_term) - penalty_ranks[g.name])
```

This matches Wood's gradient exactly when the j-th penalty is block-diagonal (which it is).

### 2.2 Gap Analysis — Detailed

#### Gap 1: No Hessian — L-BFGS-B instead of Newton

**This is the single largest algorithmic gap.**

Wood's outer iteration uses a **custom damped Newton method** with the exact REML Hessian:

```
∂²V_R / ∂ρ_j ∂ρ_k = ½ tr[H⁻¹ S_j H⁻¹ S_k]  +  δ_{jk} · [½ β̂' S_j β̂ + ½ tr(H⁻¹ S_j) - ½ r_j]
```

where H = X'WX + S. Wood's Newton converges in 3–8 outer iterations. Your code uses `scipy.optimize.minimize(method='L-BFGS-B')` which approximates the Hessian from gradient history and typically needs 10–30 function evaluations, each requiring a full IRLS solve.

**Critically, mgcv does NOT use L-BFGS-B as its default optimizer.** (See [Appendix](#appendix-mgcv-optimizer-reference) for full details.) mgcv's `gam()` default is `optimizer=c("outer","newton")` — Wood's custom Newton method with:
- Exact analytic gradient and Hessian (computed via implicit differentiation through the P-IRLS solution)
- Step-halving line search with steepest-descent fallback
- Step size capped at 5 in log(λ) space

L-BFGS-B appears in mgcv only as:
- `gam(..., optimizer=c("outer","optim"))` — a non-default, **explicitly discouraged** option using R's `optim(method="L-BFGS-B")` with **finite-difference** gradients (not analytic)
- `gamm()` — the mixed-model representation pathway (completely different code path)

Our current use of L-BFGS-B is therefore closer to the discouraged `gam(..., optimizer="optim")` pathway than to mgcv's default. The key improvement path is implementing the exact Newton method with analytic Hessian.

| | L-BFGS-B (current) | Newton (Wood's default) |
|---|---|---|
| Gradient evaluations | Many (each requires IRLS convergence) | Few (typically 3–8 outer steps) |
| Hessian | Approximate (secant updates) | Exact analytic |
| Step control | Wolfe line search | Damped Newton + step-halving + steepest descent fallback |
| Cost per step | 1 IRLS + 1 eigh | 1 IRLS + trace computations (recycled from decomposition) |

#### Gap 2: Redundant decompositions (no QR/Cholesky reuse)

Wood's implementation recycles a single orthogonal decomposition for:
1. The PIRLS solve (β̂ = R⁻¹ Qᵀ [√W z; 0])
2. log|H| = 2 Σ log|R_jj| (trivially stable from the triangular factor)
3. H⁻¹ = R⁻¹ R⁻ᵀ (for gradient trace terms and Hessian)

Our current code performs **three separate decompositions**:
- `eigh(M_aug)` in `fit_irls_direct` for the PIRLS solve (line 216)
- `eigvalsh(M)` in `_reml_laml_objective` for log|H| (line 1218)
- `eigvalsh(S)` in `_reml_laml_objective` for log|S|₊ (line 1213)

#### Gap 3: Penalty log-determinant not cached

log|S|₊ depends only on the penalty structure and λ. For block-diagonal penalties:

```
log|S|₊ = Σ_j  r_j · log(λ_j) + Σ_j  log|Ω_j|₊
```

where log|Ω_j|₊ is constant across the entire REML optimization and can be computed once. Our code recomputes the full eigendecomposition of S on every objective evaluation.

#### Gap 4: Numerical stability of log-determinant

Our current log|H| computation (`_reml_laml_objective`, lines 1213–1221):
```python
eigvals_m = np.linalg.eigvalsh(M)
logdet_m = float(np.sum(np.log(pos_m)))
```

Problems:
1. **Cancellation:** log|H| - log|S|₊ subtracts two large numbers, losing precision
2. **Rank threshold fragility:** The threshold `1e-10 * max(eigvals.max(), 1e-12)` is ad hoc
3. **No recycling:** The decomposition is separate from the solve

Wood's approach: Cholesky or QR of the augmented system gives log|H| = 2 Σ log|diag(L)| — trivially stable.

#### Gap 5: No Fellner-Schall update (critical for the BCD path)

Wood et al. (2016) / Wood & Fasiolo (2017) introduced the Extended Fellner-Schall (EFS) update used by `bam()`:

```
λ_j_new = (λ_j² β̂' Ω_j β̂ + tr(F S_j)) / r_j
```

where F = (X'WX+S)⁻¹ X'WX is the hat matrix analogue. The EFS update avoids division by small quantities and has better contraction properties. This is what `bam(optimizer="efs")` uses and is the most robust monotone update for large-data problems.

Our fixed-point update is the standard Wood (2011) formula, which is closely related but differs in how the trace term is computed. The EFS formulation is more numerically robust.

**Why EFS matters specifically for SuperGLM:** For the direct path (λ₁=0), Newton with exact Hessian is appropriate since the REML surface is smooth (purely L2 penalties). But for the BCD path (λ₁>0), the L1 penalties (group lasso, elastic net, SGL) make the REML surface **non-differentiable** at active-set boundaries. The EFS monotone update is inherently more stable than Newton when the active set of coefficients is changing between REML iterations.

#### Gap 6: No estimated-scale REML

For families with unknown φ (Gamma, NB2, Tweedie), the REML criterion should include φ:

```
V_R = (1/φ)[-ℓ + ½β̂'Sβ̂] + ½log|X'WX/φ + S/φ|₊ - ½log|S/φ|₊ + ((n-M_p)/2)log(φ)
```

Our code does not distinguish known vs. estimated scale in the REML objective. For Poisson (φ=1) this is correct, but for Gamma/NB2/Tweedie the criterion is incomplete.

#### Gap 7: R_inv coupling in REML loop

When λ_j changes, the SSP reparametrization R⁻¹_j (which depends on λ_j via Cholesky of G + λ_j Ω) changes, requiring coefficient remapping via `_map_beta_between_bases`. This is handled correctly but adds O(K³) per group per REML iteration.

Wood avoids this by working in the original (un-reparametrized) basis during the REML optimization. However, our BCD solver depends on the SSP conditioning for the proximal Newton inner loop, so this coupling is necessary for the BCD path (λ₁ > 0).

#### Gap 8: No Hessian PD projection for step control

Without the Hessian, there is no natural way to detect saddle points or control step size. L-BFGS-B handles this via line search. Wood's approach perturbs the Hessian to be positive-definite (adding a ridge to negative eigenvalues) and falls back to steepest descent when the Newton step is too large.

### 2.3 What Currently Works Well

These aspects of the implementation align well with or exceed Wood's approach:

1. **Gram-based X'WX assembly** via `_block_xtwx` — avoids materializing the (n, p) matrix entirely. Wood's implementation does form √W·X for QR, which is O(n·p) memory.

2. **Discretized SSP matrices** — the `DiscretizedSSPGroupMatrix` with bin-aggregated gram/matvec is directly analogous to mgcv's `bam()` discretization and provides the same O(n_bins) speedup.

3. **Anderson acceleration on log-lambda** — a sound acceleration that mgcv does not use (mgcv uses Newton instead, which is faster but more complex).

4. **Cheap iterations** — skipping PIRLS when Δλ < 1% matches the spirit of Wood's `gam.fit3` which avoids re-convergence of PIRLS when λ barely changed.

5. **Dual solver dispatch** — automatic switching between BCD (λ₁>0) and direct IRLS (λ₁=0) is a clean design that mgcv doesn't need (mgcv has no L1 penalties).

6. **Group penalty integration** — REML coexisting with group lasso/elastic net is a genuine extension beyond mgcv's capabilities. The fixed-point REML path around BCD is the correct architecture for this.

---

## Phase 3: Implementation Plan

### 3.1 New Data Structures

#### PenaltyCache — per-group, computed once at REML entry

```python
@dataclass
class PenaltyCache:
    """Pre-computed per-group penalty eigenstructure."""
    omega_ssp: NDArray         # R_inv.T @ omega @ R_inv, (p_g, p_g)
    log_det_omega_plus: float  # log|Ω|₊ (constant across lambda iterations)
    rank: float                # rank(Ω) = r_j
    eigvals_omega: NDArray     # eigenvalues of Ω_ssp (for S log-det shortcut)
```

Store as `dict[str, PenaltyCache]` keyed by group name. Compute once in `fit_reml()` before the outer loop.

#### PIRLSDecomposition — cached from the converged solve

```python
@dataclass
class PIRLSDecomposition:
    """Cached decomposition from the converged PIRLS step.

    The decomposition is obtained via Cholesky when H is positive definite,
    falling back to thresholded eigendecomposition when H is rank-deficient
    (see _safe_decompose_H in Step 2). Both paths produce the same H_inv
    and log_det_H interface — the REML loop does not need to know which
    was used.
    """
    log_det_H: float           # log|H| (Cholesky) or log|H|₊ (eigh fallback)
    H_inv: NDArray             # (X'WX + S)⁻¹ or pseudo-inverse
    XtWX: NDArray              # X'WX (cached for trace terms and edf)
    F_hat: NDArray             # H_inv @ XtWX (hat matrix analogue)
    cholesky_ok: bool          # True if Cholesky path was used
```

This replaces the pattern of computing `eigh(M_aug)` in the solver and then recomputing `eigh(XtWX + S)` separately for REML.

### 3.2 Algorithm Blueprint — Inner Loop (PIRLS for fixed λ)

```
function PIRLS_with_decomposition(X, y, w, family, link, groups, S(λ), offset):
    """Penalized IRLS returning cached decomposition (Cholesky or eigh fallback)."""

    β ← warm_start or zeros(p)
    α ← link(mean(y))

    for iter = 1, ..., max_iter:
        η ← X @ β + α + offset
        μ ← link⁻¹(η)
        W ← w · (dμ/dη)² / V(μ)
        z ← η + (y - μ) / (dμ/dη)

        # Gram-based X'WX (existing _block_xtwx — no change)
        XtWX ← _block_xtwx(gms, groups, W)

        # Armored decomposition: try Cholesky, fall back to eigh
        H ← XtWX + S
        H_inv, log_det_H, cholesky_ok ← _safe_decompose_H(H)

        # Solve for beta + intercept
        XtWz ← X' (W · (z - offset))
        β ← H_inv @ (XtWz - XtW1 · (Σ(W·z) / Σ(W)))
        α ← (Σ(W · z) - XtW1' β) / Σ(W)

        # Deviance convergence check
        if converged: break

    # After convergence: cache for REML (H_inv already computed)
    F_hat ← H_inv @ XtWX                   # for edf and EFS

    return β, α, dev, PIRLSDecomposition(log_det_H, H_inv, XtWX, F_hat, cholesky_ok)
```

**Key difference from current:** A single `_safe_decompose_H` call serves triple duty — solving the linear system, computing log|H|, and providing H⁻¹ for trace terms. Currently these are three separate decompositions. The Cholesky path is faster and gives a numerically ideal log-determinant; the eigh fallback preserves robustness for rank-deficient H.

**Performance note:** For p ~ 100 and n ~ 10⁶, the dominant cost is forming X'WX via `_block_xtwx`, which is O(n). The O(p³) decomposition/solve is negligible. The switch does not change asymptotic cost — it improves numerical stability and eliminates redundant decompositions.

### 3.3 Algorithm Blueprint — Outer Loop (Newton on REML Score)

```
function REML_Newton(X, y, w, family, link, groups, penalties):
    """Newton optimization of the REML criterion over log-lambdas."""

    # Pre-compute penalty eigenstructure (ONCE)
    for each group j:
        Ω_ssp_j ← R_inv_j' Ω_j R_inv_j
        eigvals_j ← eigh(Ω_ssp_j)
        r_j ← sum(eigvals_j > ε · max(eigvals_j))
        log_det_Ω_j ← sum(log(eigvals_j[> threshold]))
        cache_j ← PenaltyCache(Ω_ssp_j, log_det_Ω_j, r_j, eigvals_j)

    # Initialize log-lambdas
    ρ ← [log(λ₁⁰), ..., log(λ_m⁰)]

    for outer = 1, ..., max_outer:
        λ ← exp(ρ)
        S ← Σ_j λ_j · Ω_ssp_j  (block-diagonal)

        # === Inner loop: converge PIRLS for this λ ===
        β, α, dev, decomp ← PIRLS_with_decomposition(X, y, w, ..., S)

        # === REML objective (cached log-dets) ===
        nll ← -log_likelihood(y, μ, w, φ)
        penalty_quad ← β' S β
        log_det_S_plus ← Σ_j (r_j · log(λ_j) + cache_j.log_det_Ω_plus)  # CACHED
        V_R ← nll + 0.5 · (penalty_quad + decomp.log_det_H - log_det_S_plus)

        # === REML gradient (m-vector) ===
        for j = 1, ..., m:
            H_inv_jj ← decomp.H_inv[g_j, g_j]
            grad_j ← 0.5 · (
                λ_j · (β[g_j]' Ω_ssp_j β[g_j] + tr(H_inv_jj Ω_ssp_j))
                - r_j
            )

        # === REML Hessian (m × m matrix) ===
        for j, k = 1, ..., m:
            H_inv_Sj ← decomp.H_inv[:, g_j] @ (λ_j · Ω_ssp_j)
            H_inv_Sk ← decomp.H_inv[:, g_k] @ (λ_k · Ω_ssp_k)
            hess_jk ← 0.5 · tr(H_inv_Sj' H_inv_Sk)
            if j == k:
                hess_jk += grad_j

        # === Perturb Hessian to positive definite ===
        eigvals_H, eigvecs_H ← eigh(Hessian)
        eigvals_H ← max(eigvals_H, ε · max(eigvals_H))
        Hessian_PD ← eigvecs_H · diag(eigvals_H) · eigvecs_H'

        # === Newton step ===
        Δρ ← -solve(Hessian_PD, gradient)
        Δρ ← clip(Δρ, -5, 5)  # cap step size (Wood's maxNstep=5)

        # === Steepest descent fallback (Wood's approach) ===
        if norm(Δρ) too large or Hessian indefinite:
            Δρ ← -gradient / norm(gradient) · min_step

        # === Step-halving line search ===
        step ← 1.0
        for ls = 1, ..., max_ls:
            ρ_trial ← ρ + step · Δρ
            ρ_trial ← clip(ρ_trial, log(1e-6), log(1e6))
            V_trial ← evaluate_REML(ρ_trial)  # requires PIRLS convergence
            if V_trial < V_R:
                ρ ← ρ_trial; break
            step ← step / 2

        # === Convergence check ===
        if max(|gradient|) < tol:
            break

    return λ, β, α, decomp.H_inv
```

**Key features:**
1. Inner PIRLS is fully converged before any derivatives are computed (matching Wood 2011)
2. log|S|₊ uses cached penalty eigenstructure — no recomputation
3. log|H| comes from the Cholesky factor — no separate eigendecomposition
4. Exact analytic Hessian from H⁻¹ (already available from PIRLS solve)
5. Hessian projected to PD, with steepest descent fallback (matching mgcv's `newton` option)
6. Step size capped at 5 in log(λ) space (matching `gam.control(newton=list(maxNstep=5))`)

### 3.4 Refactoring Steps — Prioritized

#### Step 1: Cache penalty eigenstructure [Small effort]

**Files:** `model.py` (in `fit_reml`), `reml.py` (new `PenaltyCache` dataclass)

Before the outer REML loop, compute and cache per-group:
- Ω_ssp_j, its eigenvalues, rank r_j, and log|Ω_j|₊

This replaces the per-evaluation `np.linalg.eigvalsh(S)` in `_reml_laml_objective` (line 1213). The log|S|₊ computation becomes:
```python
logdet_s = sum(
    cache.rank * np.log(lambdas[g.name]) + cache.log_det_omega_plus
    for g_name, cache in penalty_caches.items()
)
```

#### Step 2: Switch IRLS solve from eigh to Cholesky-with-fallback [Medium effort]

**Files:** `solvers/irls_direct.py`

> **WARNING — The Cholesky Trap.** In applied GAM fitting, H = X'WX + S is
> frequently **not** strictly positive definite. Unpenalized categorical
> variables with collinearity, or IRLS working weights (W) pushed near zero
> in Binomial/Poisson tails, make H numerically rank-deficient. Standard
> `np.linalg.cholesky` will hard-crash with `LinAlgError`. Wood handles this
> in mgcv by using pivoted QR or symmetric eigendecompositions to safely drop
> zero-rank dimensions.
>
> **Rule: never remove the eigh code path.** Cholesky is attempted first for
> speed and log-determinant stability; on failure, we fall back seamlessly to
> the existing thresholded eigendecomposition.

Replace lines 215–224 in `fit_irls_direct` with the following armored decomposition:

```python
import scipy.linalg

def _safe_decompose_H(H: NDArray) -> tuple[NDArray, NDArray, float]:
    """Decompose H = X'WX + S, returning (H_inv, beta_solve_func_result, log_det_H).

    Attempts Cholesky first (fast, numerically ideal for log-det).
    Falls back to thresholded eigendecomposition for rank-deficient H.

    Returns
    -------
    H_inv : (p, p) inverse (or pseudo-inverse) of H.
    log_det_H : log|H| (from Cholesky diagonal) or log|H|₊ (from positive eigenvalues).
    cholesky_ok : bool — True if Cholesky succeeded.
    """
    p = H.shape[0]

    # === Primary path: Cholesky ===
    try:
        L = scipy.linalg.cholesky(H, lower=True, check_finite=False)
        log_det_H = 2.0 * float(np.sum(np.log(np.diag(L))))
        # Solve H⁻¹ via forward/back substitution (numerically superior to inv)
        H_inv = scipy.linalg.cho_solve((L, True), np.eye(p))
        return H_inv, log_det_H, True
    except np.linalg.LinAlgError:
        pass

    # === Fallback: thresholded eigendecomposition ===
    # This is the existing logic from irls_direct.py:216-224, preserved exactly.
    eigvals, eigvecs = np.linalg.eigh(H)
    threshold = 1e-6 * max(eigvals.max(), 1e-12)
    with np.errstate(divide="ignore"):
        inv_eigvals = np.where(eigvals > threshold, 1.0 / eigvals, 0.0)
    H_inv = (eigvecs * inv_eigvals[None, :]) @ eigvecs.T

    # log|H|₊ from positive eigenvalues only
    pos_eigvals = eigvals[eigvals > threshold]
    log_det_H = float(np.sum(np.log(pos_eigvals))) if pos_eigvals.size > 0 else 0.0

    return H_inv, log_det_H, False
```

**Usage in the IRLS loop (per iteration):**
```python
# Form H = XtWX + S (existing _block_xtwx + _build_penalty_matrix)
H = XtWX + S

# Armored decomposition: Cholesky when possible, eigh fallback
H_inv, log_det_H, cholesky_ok = _safe_decompose_H(H)

# Solve for beta (intercept handled separately via closed-form)
XtWz_adjusted = dm.rmatvec(W * z_off) - XtW1 * (np.sum(W * z_off) / sum_W)
beta = H_inv @ XtWz_adjusted  # or cho_solve if Cholesky succeeded
intercept = (np.sum(W * z_off) - XtW1 @ beta) / sum_W
```

**After convergence (once):** The final `H_inv` and `log_det_H` are returned in a `PIRLSDecomposition` for the REML loop to reuse. The separate `_invert_xtwx_plus_penalty` (line 73) and post-solve eigendecomposition (line 253) become unnecessary.

**When does Cholesky fail in practice?**
- Unpenalized groups (categoricals with rare levels, numeric features near-collinear with spline null space)
- Very early IRLS iterations where W has extreme range (before μ stabilizes)
- Models with many unpenalized interaction terms

The fallback preserves exact numerical behavior with the current codebase — no regression risk.

#### Step 3: Implement analytic REML Hessian [Medium effort]

**Files:** `model.py` (new method `_reml_direct_hessian`)

```python
def _reml_direct_hessian(
    self,
    H_inv: NDArray,
    lambdas: dict[str, float],
    reml_groups: list[tuple[int, GroupSlice]],
    penalty_caches: dict[str, PenaltyCache],
    gradient: NDArray,
) -> NDArray:
    m = len(reml_groups)
    hess = np.zeros((m, m))

    # Pre-compute H⁻¹ S_j blocks
    H_inv_Sj = {}
    for i, (idx, g) in enumerate(reml_groups):
        omega_ssp = penalty_caches[g.name].omega_ssp
        lam = lambdas[g.name]
        H_inv_Sj[i] = H_inv[:, g.sl] @ (lam * omega_ssp)

    for i in range(m):
        for j in range(i, m):
            h = 0.5 * float(np.sum(H_inv_Sj[i] * H_inv_Sj[j]))
            hess[i, j] = h
            hess[j, i] = h
        hess[i, i] += gradient[i]  # δ_{ij} term

    return hess
```

**Cost:** O(p² · Σ p_j) for the trace products. For p = 100 and m = 6 groups with p_j ~ 12, this is ~720k flops — negligible.

#### Step 4: Replace L-BFGS-B with damped Newton [Medium effort]

**Files:** `model.py` (rewrite `_optimize_direct_reml`)

Replace `scipy.optimize.minimize(method='L-BFGS-B')` with the hand-written Newton loop from §3.3:

```python
for outer in range(max_reml_iter):
    # Converge PIRLS (warm-started)
    result, decomp = fit_irls_direct_cholesky(...)

    # Evaluate REML objective (with cached log-dets)
    obj = self._reml_laml_objective_cached(...)

    # Exact gradient and Hessian
    grad = self._reml_direct_gradient(result, decomp.H_inv, ...)
    hess = self._reml_direct_hessian(decomp.H_inv, ...)

    # PD-projected Newton step with step-halving
    delta = _newton_step(grad, hess, max_step=5.0)
    rho = _line_search(rho, delta, grad, evaluate_reml)

    if max(|grad|) < reml_tol:
        break
```

The step-halving line search requires PIRLS convergence at each trial point. Warm-starting from the previous solution makes each trial cheap (2–3 IRLS iterations).

#### Step 5: Add steepest descent fallback [Small effort]

**Files:** `model.py` (within the Newton loop from Step 4)

When the Hessian has negative eigenvalues or the Newton step is too large, fall back to a steepest descent step:

```python
eigvals_h, eigvecs_h = np.linalg.eigh(hess)
if eigvals_h.min() < 1e-3 * eigvals_h.max():
    # Steepest descent with limited step
    delta = -grad * min_step / max(np.linalg.norm(grad), 1e-8)
else:
    eigvals_h = np.maximum(eigvals_h, 1e-3 * eigvals_h.max())
    hess_pd = (eigvecs_h * eigvals_h) @ eigvecs_h.T
    delta = -np.linalg.solve(hess_pd, grad)
    delta = np.clip(delta, -5, 5)
```

This matches mgcv's `gam.fit3` behavior.

#### Step 6: Implement Fellner-Schall update for the BCD path [Small effort]

**Files:** `model.py` (new method or option in `_run_reml_once`)

> **Why Fellner-Schall for the BCD path, not Newton.**
>
> One might think Newton is impractical for the BCD path because BCD only
> computes per-block Hessians, not the full H⁻¹. But in fact, `pirls.py`
> (lines 306–339) already computes the **exact full dense p×p** (X'WX+S)⁻¹
> at the end of each outer PIRLS iteration to calculate effective degrees of
> freedom. So H⁻¹ *is* available, and the Newton Hessian cross-traces
> tr(H⁻¹S_j H⁻¹S_k) would be trivially cheap at O(p²).
>
> **The real reason** to use Extended Fellner-Schall here is that the **L1
> penalties (group lasso / elastic net / SGL) make the REML surface
> non-differentiable** with respect to the active set. Wood's exact Newton
> Hessian assumes a smooth, purely L2 penalty space. When the BCD solver
> zeros out a group entirely, the coefficient vector sits at a kink in the
> L1 penalty. A Newton step computed from the smooth Hessian can overreact
> to active-set changes — the gradient and Hessian are only valid for the
> current active set, but the Newton step may cross into a different active
> set where they are wrong. The EFS method is a **monotone, fixed-point
> iteration** that is vastly more stable when the active set of coefficients
> is changing dynamically between REML iterations.

The EFS update is a one-line per-group formula:

```python
for idx, g in reml_groups:
    omega_ssp = cache.omega_ssp
    S_j = lambdas[g.name] * omega_ssp
    quad = float(beta[g.sl] @ S_j @ beta[g.sl])  # λ² β'Ωβ
    trace_FS = float(np.trace(F_hat[g.sl, g.sl] @ S_j))
    lambdas_new[g.name] = (quad + trace_FS) / penalty_ranks[g.name]
```

where F_hat = H⁻¹ X'WX is already computed for edf. This provides the robust
REML update for the BCD path (λ₁ > 0), and could also serve as a fallback for
the direct path when Newton struggles with a non-convex REML landscape.

#### Step 7: Handle estimated scale (φ) in REML [Medium effort — CRITICAL CORRECTNESS]

**Files:** `model.py:_reml_laml_objective`, `_reml_direct_gradient`, `_reml_direct_hessian`

> **This step is a critical correctness fix, not an optimization.**
>
> The current `_reml_laml_objective` evaluates the Laplace-approximate REML
> criterion without explicitly profiling out the dispersion parameter φ.
> For Poisson and Binomial (known scale, φ=1) this is correct. But for
> **Gamma, Negative Binomial, and Tweedie** — all of which SuperGLM supports
> and which are heavily used in insurance applications — the optimal λ values
> will **diverge significantly from mgcv** without proper φ treatment.

**The full φ-profiled REML criterion** (Wood 2017, §6.2.2):

```
V_R(ρ) = (1/φ̂)[-ℓ(y|β̂) + ½ β̂'S β̂]
        + ½ log|X'WX/φ̂ + S/φ̂|₊
        - ½ log|S/φ̂|₊
        + ((n - M_p) / 2) · log(φ̂)
```

where M_p = Σ_j rank(Ω_j) and φ̂ is the profiled scale estimator:

```python
M_p = sum(cache.rank for cache in penalty_caches.values())
phi_hat = (dev + float(beta @ S @ beta)) / max(n - M_p, 1.0)
```

**The log-determinant terms simplify under φ-scaling.** Since S/φ = Σ_j (λ_j/φ) Ω_j, the
log-determinants factor as:

```
log|X'WX/φ + S/φ|₊ = log|H/φ|₊ = log|H|₊ - p_eff · log(φ)
log|S/φ|₊ = log|S|₊ - M_p · log(φ)
```

So the full criterion becomes:

```
V_R = (1/φ̂)[-ℓ + ½ β̂'Sβ̂] + ½[log|H| - log|S|₊] + ((n - p_eff)/2) · log(φ̂)
```

which differs from the known-scale version only by the `((n-p_eff)/2) · log(φ̂)` term and
the `1/φ̂` scaling of the penalized deviance.

**Gradient correction.** φ̂ depends on ρ through β̂ and S:

```
∂φ̂/∂ρ_j = (1/(n-M_p)) · [∂dev/∂ρ_j + β̂' S_j β̂ + β̂' S (∂β̂/∂ρ_j)]
```

In the Laplace approximation with β̂ at the penalized MLE, ∂dev/∂ρ_j + β̂' S_j β̂ ≈ 0
(the score equation), so the dominant contribution is the direct β̂'S_j β̂ term. The
practical implementation profiles φ̂ at each Newton step and treats it as fixed for the
gradient/Hessian computation at that step — this is Wood's approach.

```python
# In _reml_laml_objective:
if family.scale_known:
    phi = 1.0
    scale_term = 0.0
else:
    M_p = sum(cache.rank for cache in penalty_caches.values())
    phi = (dev + float(beta @ S @ beta)) / max(n - M_p, 1.0)
    scale_term = 0.5 * (n - p_eff) * np.log(phi)

nll_scaled = nll / phi
penalty_scaled = 0.5 * penalty_quad / phi
V_R = nll_scaled + penalty_scaled + 0.5 * (log_det_H - logdet_s) + scale_term
```

**Testing:** On a Gamma regression with `family="gamma"`, compare the estimated λ against
`mgcv::gam(y ~ s(x), family=Gamma(link="log"), method="REML")`. Without Step 7, expect
the SuperGLM λ to be wrong by 2–5× for typical insurance claim severity data.

#### Step 8 (Optional): Eliminate R_inv coupling in direct REML [Large effort]

For the direct REML path (λ₁ = 0), work in the original B-spline basis during the Newton loop. The penalty is simply λ_j Ω_j (no R⁻¹ needed). Only apply the SSP reparametrization for the final coefficient extraction.

**Deferral rationale:** This is a large refactor that requires a second code path for the direct solver. The current R_inv coupling adds only O(K³) per group per REML iteration, which is negligible compared to the O(n) gram computation. Defer unless profiling shows this is a bottleneck.

### 3.5 Summary Table

| Priority | Step | Gap Addressed | Impact | Effort |
|---|---|---|---|---|
| 1 | Cache penalty eigenstructure | Redundant S eigendecomp | Correctness + speed | Small |
| 2 | Cholesky-with-fallback for IRLS | Redundant decompositions | Numerical stability | Medium |
| 3 | Analytic REML Hessian | No Hessian available | Enables Newton | Medium |
| 4 | Newton optimizer | L-BFGS-B → Newton | 2–3× fewer IRLS solves | Medium |
| 5 | Steepest descent fallback | No step control | Robustness | Small |
| 6 | Fellner-Schall for BCD path | L1 non-differentiability | BCD path robustness | Small |
| **7** | **Estimated scale REML** | **Wrong criterion for Gamma/NB2/Tweedie** | **CRITICAL correctness** | Medium |
| 8 | R_inv decoupling (optional) | REML loop overhead | Minor speedup | Large |

**Recommended order:** 1 → 2 → 7 → 3 → 4 → 5 → 6. Steps 1–2 are prerequisites for steps 3–4 (the Newton loop needs cached log-determinants and Cholesky-based H⁻¹). **Step 7 is elevated to immediately after Step 2** because it is a correctness fix — without it, REML produces wrong λ for all estimated-scale families (Gamma, NB2, Tweedie), which are the primary insurance use case. Step 6 is independent and can be done in parallel.

### 3.6 Testing Strategy

#### Unit-level validation against mgcv

**Test 1: REML objective parity.** For a small problem (n = 500, 2 splines, Poisson), fit in mgcv with `gam(y ~ s(x1) + s(x2), family=poisson, method="REML")` and extract `gam$sp` (optimal λ) and `gam$gcv.ubre` (REML criterion value). Compare against `_reml_laml_objective` evaluated at the same λ and β̂.

**Test 2: Gradient parity.** Use mgcv internals to extract the REML gradient at a specific (λ, β̂) pair. Compare against `_reml_direct_gradient`.

**Test 3: Lambda convergence.** On synthetic data with known smooth functions, verify estimated λ̂ matches mgcv's `gam$sp` to within 5–10%.

**Test 4: Effective degrees of freedom.** Compare `sum(gam$edf)` against `effective_df` at the converged solution. Should match within ±0.5 for well-identified models.

#### Numerical stability tests

**Test 5: Condition number sweep.** Vary λ from 10⁻⁶ to 10⁶ and verify log|H|, log|S|₊, and V_R are finite and vary smoothly.

**Test 6: Near-singular penalty.** For CR spline with K = 15 (Ω has rank K-3), verify log-determinant computation is stable.

#### Integration tests

**Test 7: MTPL2 benchmark.** On the 678k French MTPL2 dataset:
- `fit_reml()` with Newton should converge in ≤ 8 outer iterations
- Total time should be ≤ 15s (comparable to current ~11s)
- Estimated λ should match current fixed-point results to within 10%

**Test 8: Select=True with Newton.** Verify double-penalty decomposition works with Newton REML — both linear and spline subgroup λs should be estimated.

**Test 9: Group lasso + REML.** Verify the BCD path (λ₁ > 0) with Fellner-Schall update produces the same results as the current fixed-point iteration.

#### R comparison script

```r
library(mgcv)
data <- read.csv("test_data.csv")
m <- gam(y ~ s(x1, k=15) + s(x2, k=15),
         family=poisson, data=data, method="REML")
cat("sp:", m$sp, "\n")
cat("reml:", m$gcv.ubre, "\n")
cat("edf:", sum(m$edf), "\n")
```

Run the same data through SuperGLM and compare all three outputs.

---

## Appendix: mgcv Optimizer Reference

### What optimizer does mgcv actually use?

mgcv's `gam()` default is `optimizer=c("outer","newton")` — a **custom damped Newton method** with:

- **Exact analytic gradient and Hessian** of the REML/ML/GCV score, computed via implicit differentiation through the P-IRLS solution (Wood 2011, §3)
- **Step-halving** line search with steepest descent fallback
- **Step size capped at 5** in log(λ) space (`gam.control(newton=list(maxNstep=5))`)
- Convergence in **3–8 outer Newton iterations** for typical problems

The following table documents all optimizer options:

| Function | Optimizer | Derivatives | Notes |
|---|---|---|---|
| `gam()` default | Custom Newton (exact 1st+2nd order) | Analytic (implicit diff) | Wood (2011). Default and recommended. |
| `gam(..., optimizer=c("outer","bfgs"))` | Custom BFGS | Analytic 1st order, BFGS Hessian approx | For many smoothing parameters. |
| `gam(..., optimizer=c("outer","optim"))` | R's `optim(method="L-BFGS-B")` | **Finite difference** | **Explicitly discouraged.** Finite-difference gradients are unreliable when PIRLS convergence tolerance ≈ FD step. |
| `gam(..., optimizer=c("outer","nlm"))` | R's `nlm()` | Finite difference | Also discouraged. |
| `bam()` default | Performance iteration + fREML | Analytic REML derivatives | Wood, Li & Fasiolo (2017). Simultaneous β + λ convergence. |
| `bam(..., optimizer="efs")` | Extended Fellner-Schall | 1st+2nd order log-lik only | Wood & Fasiolo (2017). Monotone REML-increasing update. |
| `gamm()` | R's `optim(method="L-BFGS-B")` | Via nlme mixed model | Completely different code path. Wood (2004). |

### Key insight for SuperGLM

Our current direct REML path uses `scipy.optimize.minimize(method='L-BFGS-B')`, which is closest to the **discouraged** `gam(..., optimizer="optim")` option. However, our version is better than mgcv's L-BFGS-B option because we supply **analytic gradients** (not finite differences). Still, the path to mgcv parity is implementing the exact Newton method with analytic Hessian (Steps 3–5 above).

Our fixed-point REML path (for λ₁ > 0, BCD) is closest in spirit to the **Fellner-Schall** approach — a simple iterative update using only first/second derivatives of the log-likelihood. The proposed EFS update (Step 6) would bring this closer to `bam(optimizer="efs")`.

### References

- Wood, S.N. (2011). Fast stable restricted maximum likelihood and marginal likelihood estimation of semiparametric generalized linear models. *JRSS-B*, 73(1), 3–36.
- Wood, S.N. (2017). *Generalized Additive Models: An Introduction with R*, 2nd ed. Chapman & Hall/CRC.
- Wood, S.N., Pya, N. & Säfken, B. (2016). Smoothing parameter and model selection for general smooth models. *JASA*, 111(516), 1548–1563.
- Wood, S.N. & Fasiolo, M. (2017). A generalized Fellner-Schall method for smoothing parameter optimization with application to Tweedie location, scale and shape models. *Biometrics*, 73(4), 1071–1081.
- Wood, S.N., Li, Z., Shaddick, G. & Augustin, N.H. (2017). Generalized additive models for gigadata: modeling the U.K. black smoke network daily data. *JASA*, 112(519), 1199–1210.
- Li, Z. & Wood, S.N. (2020). Faster model matrix crossproducts for large generalized linear models with discretized covariates. *Statistics and Computing*, 30, 19–25.
