# Phase 5a Progress

## Completed

### Task 1: Return SCOP state from fit_irls_direct
- **Status**: DONE
- Added `H_penalized: NDArray | None = None` field to `SCOPNewtonResult` dataclass
- Added `return_scop_state: bool = False` parameter to `fit_irls_direct`
- Stores `H_scop_penalized` from Newton result in SCOP state dict
- Collects converged SCOP state per group: beta_eff, H_scop_penalized, S_scop, B_scop, reparam, bin_idx, group_sl, group_name
- Appends scop_converged dict to return tuple when requested
- Default `return_scop_state=False` preserves backward compatibility
- All 9 new tests pass, all existing tests pass

### Task 2: Build SCOP PenaltyComponent objects
- **Status**: DONE
- Created `src/superglm/reml/scop_efs.py` with `build_scop_penalty_components()`
- SCOP terms use S_scop directly as omega_ssp (no SSP/R_inv transform)
- Rank computed via eigendecomposition matching `_rank_and_logdet` pattern
- 9 new pure unit tests (no @pytest.mark.slow): all pass

### Task 4: Joint Hessian assembly
- **Status**: DONE
- Added `assemble_joint_hessian()` to `src/superglm/reml/scop_efs.py`
- Replaces SCOP blocks in XtWX+S with full Newton Hessians (H_scop_penalized)
- Returns mapping dict of group_name -> slice for SCOP groups
- 7 new pure unit tests (no @pytest.mark.slow): all pass
  - Empty SCOP passthrough, block replacement, linear block preservation
  - Mapping correctness, block-diagonal log-det additivity
  - Inverse validity, input immutability

### Task 3: SCOP-aware penalty quadratic
- **Status**: DONE
- Added `compute_scop_aware_penalty_quad()` to `src/superglm/reml/scop_efs.py`
- Subtracts wrong gamma-space SCOP contribution, adds correct beta_eff-space contribution
- Falls back to standard `beta @ S @ beta` when no SCOP terms present
- 4 new pure unit tests all pass:
  - test_scop_only_model: verifies beta_eff space quad, shows it differs from naive gamma-space
  - test_mixed_ssp_and_scop: mixed SSP + SCOP blocks computed correctly
  - test_no_scop_terms_fallback: empty scop_states gives standard quad
  - test_zero_lambda_scop_contributes_zero: lambda=0 contributes zero

### Task 5: SCOP-aware EFS lambda update
- **Status**: DONE
- Added `_is_scop_component()` helper to `src/superglm/reml/scop_efs.py`
- Added `scop_efs_lambda_update()` to `src/superglm/reml/scop_efs.py`
- SSP components: quad uses gamma-space beta from result.beta
- SCOP components: quad uses beta_eff from SCOP converged state
- Trace term always uses H_joint_inv sliced at pc.group_sl
- Uphill-step guard clips log-step to [-5, 5]
- Near-zero beta guard returns lam_old unchanged
- 5 new pure unit tests (no @pytest.mark.slow): all pass
  - test_ssp_component_uses_gamma_space
  - test_scop_component_uses_beta_eff (verifies different from gamma_eff, matches manual calc)
  - test_uphill_guard_clips_log_step
  - test_near_zero_beta_returns_old_lambda
  - test_returns_positive (20-trial fuzz)

### Task 6: SCOP-aware REML objective
- DONE
- Added `scop_states: dict[int, dict] | None = None` parameter to `reml_laml_objective`
- Penalty quad: dispatches to `compute_scop_aware_penalty_quad` when scop_states present
- Log-det: dispatches to `assemble_joint_hessian` to replace SCOP blocks before eigendecomposition
- Default `scop_states=None` preserves backward compatibility (all 161 REML tests unchanged)
- 2 new slow integration tests: `test_objective_accepts_scop_state`, `test_objective_without_scop_state_unchanged`

### Task 7: Full SCOP EFS outer loop
- DONE
- Added `optimize_scop_efs_reml()` to `src/superglm/reml/scop_efs.py`
- Follows same structure as `optimize_efs_reml` but uses `fit_irls_direct` with SCOP Newton solver
- Bootstrap phase: one IRLS with minimal penalty -> one EFS step for data-informed initial lambdas
- Main loop: inner fit -> joint Hessian -> SCOP PCs -> phi estimate -> lambda updates -> step damping -> convergence check
- Anderson(1) acceleration on log-lambda scale for faster convergence
- Uphill-step guard via REML objective comparison with half-step damping
- Exported from `superglm.reml.__init__`
- 4 new slow integration tests:
  - test_converges: returns REMLResult, converges within 20 iters
  - test_lambda_responds_to_noise: higher noise -> higher lambda
  - test_predictions_are_monotone: fitted values respect monotone constraint
  - test_returns_reml_result_with_history: lambda_history has multiple entries with correct keys

## Outstanding
- Task 8: Wire fit_reml to SCOP EFS optimizer
- Task 9: Regression and edge-case tests

### Task 8: Wire fit_reml to SCOP EFS optimizer
- DONE
- Replaced blanket `NotImplementedError` with engine-specific routing:
  - QP monotone (BSplineSmooth) still raises (no Newton Hessian)
  - SCOP monotone with unfixed lambda -> `optimize_scop_efs_reml`
  - SCOP monotone with all fixed lambdas -> Phase 4 single-fit path
  - No monotone -> existing direct/EFS paths unchanged
- Moved `offset_arr` computation before SCOP routing (was after guard)
- Post-fit housekeeping: _result, _reml_lambdas, _reml_penalties, _reml_result, _fit_stats, _reml_profile
- Changed `test_fit_reml_without_fixed_lambdas_raises_scop` to `test_fit_reml_without_fixed_lambdas_works_scop`
- Added `TestSCOPFitRemlIntegration` class with 4 tests
- All 77 tests pass across `test_scop_efs.py` + `test_monotone_fit.py`

## Test Results
- `tests/test_scop_efs.py`: 44/44 pass (40 existing + 4 new integration)
- `tests/test_monotone_fit.py`: 33/33 pass (1 modified: raises -> works)
- `tests/test_ssp_audit.py`: 2/2 pass
- `tests/test_multi_penalty.py` (non-slow): 41/41 pass
