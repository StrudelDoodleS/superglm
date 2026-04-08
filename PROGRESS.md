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

## Outstanding
- Task 6: SCOP-aware REML objective
- Task 7: Full SCOP EFS outer loop
- Task 8: Wire fit_reml to SCOP EFS optimizer
- Task 9: Regression and edge-case tests

## Test Results
- `tests/test_scop_efs.py`: 30/30 pass (non-slow: 9 Task 2 + 7 Task 4 + 4 Task 3 + 5 Task 5; slow: 9 Task 1)
- `tests/test_monotone_fit.py` (non-slow): 7/7 pass
- `tests/test_ssp_audit.py`: 2/2 pass
- `tests/test_multi_penalty.py` (non-slow): 41/41 pass
