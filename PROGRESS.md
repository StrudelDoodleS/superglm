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

## Outstanding

- Task 3: SCOP-aware penalty quadratic
- Task 5: SCOP-aware EFS lambda update
- Task 6: SCOP-aware REML objective
- Task 7: Full SCOP EFS outer loop
- Task 8: Wire fit_reml to SCOP EFS optimizer
- Task 9: Regression and edge-case tests

## Test Results
- `tests/test_scop_efs.py`: 25/25 pass (9 Task 1 + 9 Task 2 + 7 Task 4)
- `tests/test_monotone_fit.py` (non-slow): 7/7 pass
- `tests/test_ssp_audit.py`: 2/2 pass
- `tests/test_multi_penalty.py` (non-slow): 41/41 pass
