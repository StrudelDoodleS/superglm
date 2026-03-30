# REML / fREML Paper Checklist

This checklist is the working spec for pushing SuperGLM toward paper-complete
implementations of:

- Wood (2011): direct REML / ML for penalized GLMs
- Wood, Goude, Shaw (2015): large-data / discrete fREML

The rule for this document is simple:

- If the paper gives a mathematical or algorithmic requirement, capture it here.
- If the repo already implements it, mark it `done`.
- If it is partial, mark it `partial` and point at the code.
- If the paper is silent and we are making an engineering choice, say so explicitly.

## Status Legend

- `todo`
- `partial`
- `done`
- `n/a`
- `engineering` — not in the paper; our choice, documented

---

## Wood 2011: Direct REML

### Objective (Section 2, Equations 4-5)

- `done` Laplace-approximate REML/ML objective.
  `reml_laml_objective()` reml_optimizer.py:167-226.
  Both known-scale and profiled-scale paths match paper.
- `done` Profiled-scale handling for estimated-φ families.
  `0.5*(n-M_p)*log(D_p + Σλ_j β'S_j β)` at lines 213-223.
- `todo` Write equation references into the objective implementation comments.
- `done` Every determinant term matches paper notation and sign convention.
  `log|X'WX+S| - log|S|₊` structure confirmed.

### Gradient (Section 3.4)

- `done` Fixed-W gradient.
  `reml_direct_gradient()` reml_optimizer.py:234-255.
  `∂V/∂ρ_j = 0.5(λ_j(β'Ω_j β/φ + tr(H⁻¹Ω_j)) - r_j)` matches paper.
- `todo` Link each implemented term to the paper equation / appendix source.
- `done` Chain rule on log-lambda scale is correct (implicit: ∂/∂ρ = λ ∂/∂λ).

### Hessian (Section 3.5, Appendix B)

- `done` Total gradient in diagonal correction (Eq 6.2).
  Line 622 passes `grad` (partial + W correction) to `reml_direct_hessian()`.
  Regression test: `test_hessian_diagonal_uses_total_gradient`.
- `done` Outer Hessian via implicit-differentiation Jacobian.
  `hess[i,j] = -0.5 * sum(F_i * F_j.T)` at line 307.
  Equivalent to paper's Appendix B K/T matrix formulation — same result,
  different route (outer product of H⁻¹ dH_j vs explicit trace products).
- `done` Penalty quadratic Hessian.
  `S_beta.T @ H_inv @ S_beta` at lines 313-317.
- `done` Profiled-scale Hessian correction.
  `hess -= 0.5*(n-M_p)*outer(q,q)/(D+pq)²` at lines 319-325.
- `todo` Write down exactly which Hessian terms come from direct vs implicit
  differentiation, per paper appendix.
- `todo` Add a checklist item per Hessian term from the paper appendices.

### W(rho) / Implicit Differentiation (Appendix C-D)

- `done` dW/dη computation.
  `compute_dW_deta()` reml_optimizer.py:54-80.
  `dW_i/dη = w_i(2(d²μ/dη²)(dμ/dη)⁻¹ - V'(μ)/V(μ))` matches Appendix D.
- `done` First-order IFT chain: dβ̂/dρ → dη/dρ → dW → C_j → gradient correction.
  `reml_w_correction()` reml_optimizer.py:88-159.
- `done` **Second-order W(rho) terms: benchmarked and closed.**
  First-order approximation error measured at <0.06% across all families
  (Poisson, Gamma, Tweedie, NB2). Second-order terms not worth implementing.
  Benchmark: scratch/benchmark_w_correction_error.py.
- `done` dH_extra incorporated into Hessian off-diagonals (line 290-291).
- `todo` Add FD tests isolating W(rho) contributions by family/link
  (Poisson/log, Binomial/logit, Tweedie/log, NB2/log).

### Newton / Optimization Loop (Section 3, Section 6.2)

- `done` Compound convergence criterion: grad + obj change.
  `score_scale = max(1+|obj|, 1)` at line 590. Both paths.
- `done` Modified Newton via eigendecomposition + eigenvalue flooring.
  Lines 656-662. Floor at eps^0.7.
- `done` Armijo line search with step halving.
  Lines 674-737. c=1e-4, max 8 halvings.
- `done` Steepest descent fallback.
  Lines 726-736.
- `engineering` Active-set freezing of converged components.
  Lines 631-640. Not in Wood (2011); practical stabilization.
  Tolerance `freeze_tol = 0.1 * _tol` is heuristic.
- `engineering` Proportional step cap (max_newton_step=5.0).
  Lines 669-671. Not in paper; prevents wild log-lambda jumps.
- `done` Bootstrap initialization with minimal penalty.
  Lines 421-439.

### Reparameterization / Stability (Section 3.1, Appendix B)

- `done` SSP reparameterization absorbs Ω into design matrix.
  `compute_R_inv()` dm_builder.py:53-67.
  `R_inv = inv(chol(G + λΩ + εI)')` with ε=1e-8.
- `done` Cached log|S|₊ computation.
  `cached_logdet_s_plus()` reml.py:79-93.
  `log|S|₊ = Σ_j(r_j log λ_j + log|Ω_j|₊)` — stable form from Section 3.1.
- `done` Penalty caches precomputed once per fit.
  `build_penalty_caches()` reml.py:46-76.
- `todo` **Full Appendix B similarity transform not implemented.**
  Code uses simplified eigenvalue-based log-det (threshold + sum logs).
  Appendix B gives a recursive Frobenius-norm column separation algorithm
  for general multi-penalty log-det computation.
  Currently sufficient for SSP (single penalty per group after R_inv) but
  missing full robustness for general multi-penalty cases.
  **Decision**: implement the full algorithm (Slice 3).
- `engineering` R_inv regularization ε=1e-8 is fixed, not adaptive.
  Paper is silent on the exact regularization constant.

### Select Penalties (Section 5.3)

- `done` Null-space penalty reparameterization for select=True.
  dm_builder.py:495-516.
- `engineering` `select_snap` heuristic snaps degenerate groups to upper λ bound.
  reml_optimizer.py:466-472. Not from paper.

### Validation

- `done` FD gradient test (fixed-W partial gradient).
  test_reml_fd.py: `test_gradient_matches_fd`.
- `done` FD Hessian test (outer Hessian, ~5% tolerance).
  test_reml_fd.py: `test_hessian_matches_fd`.
- `done` Total gradient vs outer FD test.
  test_reml_fd.py: `test_total_gradient_matches_outer_fd`.
- `done` Total Hessian (with dH_extra) vs FD test.
  test_reml_fd.py: `test_total_hessian_matches_fd`.
- `done` Hessian diagonal uses total gradient (regression test).
  test_reml_fd.py: `test_hessian_diagonal_uses_total_gradient`.
- `done` W correction zero for Gamma/log, nonzero for Poisson/log.
  test_reml_fd.py: `test_w_correction_zero_for_gamma_log`,
  `test_w_correction_nonzero_for_poisson_log`.
- `done` Expand FD coverage to NB2, Tweedie (23 tests, up from 11).
- `done` FD test of dW/dη for Poisson/log and Gamma/log.
- `todo` Expand FD coverage to Binomial.
- `todo` Add stress tests for rank deficiency, extreme lambdas, near-collinearity.

---

## Wood, Goude, Shaw 2015: fREML / Discrete Large-Data Path

### POI / Working-Model Outer Loop

- `partial` Audit
  `optimize_discrete_reml_cached_w()` reml_optimizer.py:806-1220
  against the paper's POI/discrete iteration.
- `todo` Write down the exact points where the repo switches from paper logic to local
  engineering choices.
- `todo` Confirm the current line search / acceptance logic is consistent with the paper
  or explicitly documented as an adaptation.

### Cached Augmented Solves

- `partial` Audit
  `_solve_cached_augmented()` reml_optimizer.py:768.
- `todo` Confirm the cached solve path preserves the paper's intended objective
  approximation rather than introducing extra drift.

### Discretized Matrix Machinery

- `partial` Audit
  `DiscretizedSSPGroupMatrix` group_matrix.py:359.
- `partial` Audit
  `_block_xtwx()` group_matrix.py:755
  and related `tabmat`-backed paths.
- `todo` Confirm that the discrete path does not accidentally densify or violate the
  paper's intended large-data cost structure.

### Accuracy Envelope

- `todo` Define exact-vs-discrete drift metrics:
  objective, lambda, EDF, deviance, fitted values.
- `todo` Add exact-vs-discrete regression tests with explicit tolerances.
- `todo` Add a policy for when to warn or fall back to exact REML.
- `todo` Document what `discrete=True` means mathematically in SuperGLM.

### Benchmarks

- `todo` Benchmark MTPL2 exact vs discrete.
- `todo` Benchmark synthetic ill-conditioned cases.
- `todo` Benchmark estimated-scale families.
- `todo` Benchmark highly smooth and highly rough regimes.

---

## Shared Numerical Work

- `done` Pivoted Cholesky before SVD fallback in irls_direct.py.
- `done` SVD fallback uses masked division (no RuntimeWarning).
- `engineering` Eigenvalue floor eps^0.7 (Nocedal & Wright standard: eps^0.5).
  Retained as benchmarked heuristic. Not paper-mandated.
- `engineering` SVD threshold 1e-10 * s_max. Higham suggests eps*p*s_max.
  Retained as benchmarked heuristic.
- `engineering` EFS uphill-step guard is stale-basis heuristic, not monotonicity.
  reml_optimizer.py:1458-1497. Documented limitation.
- `todo` Benchmark-backed rationale for retained heuristic thresholds.

---

## Documentation Deliverables

- `done` Keep this checklist current as items move from `todo` to `done`.
- `todo` Add a short provenance note in the eventual PR description: paper-driven
  implementation audit, not source-driven parity work.
