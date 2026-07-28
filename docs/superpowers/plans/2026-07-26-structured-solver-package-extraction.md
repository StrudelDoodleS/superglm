# Structured Solver Package Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the 5,920-line structured solver into focused internal modules
behind a behavior-compatible `superglm.solvers.structured` facade before
merging PR #165.

**Architecture:** Extract one responsibility at a time while the facade remains
importable after every commit. Move numerical definitions byte-for-byte,
change only imports and test patch targets, and finish with an explicit
re-export facade. Keep scalar, block, SZ, REML, Tabmat, kernel, and array-layout
behavior unchanged.

**Tech Stack:** Python 3.10+, NumPy, SciPy, Numba, Tabmat, pytest, Ruff, GitHub
Actions, the structured-credibility profiling harness.

---

## File Map

- Create `src/superglm/solvers/_structured/__init__.py`: internal package
  marker with no re-exports.
- Create `src/superglm/solvers/_structured/selection.py`: backend eligibility,
  cost policy, and fallback decisions.
- Create `src/superglm/solvers/_structured/layout.py`: scalar/block coefficient
  layouts, cached layout construction, and structured design products.
- Create `src/superglm/solvers/_structured/moments.py`: scalar/block/SZ
  unpenalized systems and sufficient-statistic assembly.
- Create `src/superglm/solvers/_structured/operators.py`: compact operator
  classes, low-rank representations, traces, products, diagonals, and
  materialization.
- Create `src/superglm/solvers/_structured/geometry.py`: centered
  estimability, null-space, Ritz, and public SZ geometry.
- Create `src/superglm/solvers/_structured/factors.py`: scalar/block Schur and
  profiled factor classes.
- Create `src/superglm/solvers/_structured/assembly.py`: penalized operators,
  augmented factors, cached result records, and cached solves.
- Create `src/superglm/solvers/_structured/state.py`: retained support and
  structured-fit state records.
- Create `tests/test_structured_module_boundaries.py`: red/green ownership and
  facade architecture checks.
- Modify `src/superglm/solvers/structured.py`: progressively import moved
  definitions, then become an explicit compatibility facade.
- Modify `src/superglm/solvers/sum_to_zero.py`: import common operators and BDLR
  algebra from their true owner.
- Modify `tests/test_structured_factor.py`: patch the geometry owner.
- Modify `tests/test_sum_to_zero_structured_factor.py`: patch the geometry
  owner.
- Modify `tests/test_structured_allocations.py`: patch the moments owner.
- Read
  `docs/superpowers/specs/2026-07-26-structured-solver-package-extraction-design.md`
  as the governing scope.

## Frozen Invariants

- Numerical reference revision: `b3cf321dd4aee8f6a60213ae35f6c1f33d71fbca`.
- Existing imports from `superglm.solvers.structured` continue to return the
  moved objects.
- `structured.py` owns no function or class implementation at completion.
- Internal modules and `sum_to_zero.py` do not import implementation symbols
  from the facade.
- No numerical expression, tolerance, error, dtype, array shape, writeability
  rule, lazy import, backend decision, or REML behavior changes.
- LSS remains untouched.

### Task 1: Freeze behavior and performance evidence

**Files:**

- Read: `benchmarks/profile_structured_credibility.py`
- Output: `/tmp/superglm-structured-split-baseline-fs/`
- Output: `/tmp/superglm-structured-split-baseline-sz/`

- [ ] **Step 1: Confirm the branch and approved design commit**

Run:

```bash
rtk git status --short
rtk git branch --show-current
rtk git rev-parse HEAD
rtk git rev-parse b3cf321
```

Expected: clean worktree on `feature/structured-credibility`; `HEAD` includes
the design commit `54fa407`; the numerical reference resolves to `b3cf321`.

- [ ] **Step 2: Run the focused pre-move suite**

Run:

```bash
rtk proxy env \
  PYTHONPATH=/home/mhick/python_projects/superglm/.worktrees/structured-credibility/src \
  /home/mhick/python_projects/superglm/.worktrees/structured-credibility/.venv/bin/python \
  -m pytest \
  tests/test_random_effect*.py \
  tests/test_factor_smooth*.py \
  tests/test_structured*.py \
  tests/test_sum_to_zero_structured_factor.py \
  tests/test_block_schur_factor.py \
  tests/test_reml_newton_fixes.py \
  tests/test_cached_w_validation.py \
  tests/test_fit_state_retention.py -q
```

Expected: all collected tests pass. Record the count in the work log.

- [ ] **Step 3: Capture the million-row FS baseline**

Run:

```bash
rtk proxy env \
  PYTHONPATH=/home/mhick/python_projects/superglm/.worktrees/structured-credibility/src \
  /home/mhick/python_projects/superglm/.worktrees/structured-credibility/.venv/bin/python \
  benchmarks/profile_structured_credibility.py \
  --n 1000000 --levels 300 --family poisson --discrete \
  --random-effects 0 --small-width 4 --structured-term factor_smooth \
  --block-size 10 --factor-basis fs --global-spline --weights nonuniform \
  --backend structured --repetitions 5 --warmups 1 --max-reml-iter 20 \
  --reml-tol 1e-7 --seed 20260724 --sample-interval-ms 250 \
  --no-cprofile --no-tracemalloc --no-dense-parity \
  --output-dir /tmp/superglm-structured-split-baseline-fs
```

Expected: convergence with `score_objective_tolerance`, seven REML iterations,
and a prediction checksum near `1045154.11521279`.

- [ ] **Step 4: Capture the million-row SZ baseline**

Run:

```bash
rtk proxy env \
  PYTHONPATH=/home/mhick/python_projects/superglm/.worktrees/structured-credibility/src \
  /home/mhick/python_projects/superglm/.worktrees/structured-credibility/.venv/bin/python \
  benchmarks/profile_structured_credibility.py \
  --n 1000000 --levels 300 --family poisson --discrete \
  --random-effects 0 --small-width 4 --structured-term factor_smooth \
  --block-size 10 --factor-basis sz --global-spline --weights nonuniform \
  --backend structured --repetitions 5 --warmups 1 --max-reml-iter 20 \
  --reml-tol 1e-7 --seed 20260724 --sample-interval-ms 250 \
  --no-cprofile --no-tracemalloc --no-dense-parity \
  --output-dir /tmp/superglm-structured-split-baseline-sz
```

Expected: convergence with `score_objective_tolerance`, five REML iterations,
and a prediction checksum near `1045155.20722107`.

### Task 2: Extract compact operators and low-rank algebra

**Files:**

- Create: `src/superglm/solvers/_structured/__init__.py`
- Create: `src/superglm/solvers/_structured/operators.py`
- Create: `tests/test_structured_module_boundaries.py`
- Modify: `src/superglm/solvers/structured.py`
- Modify: `src/superglm/solvers/sum_to_zero.py`

- [ ] **Step 1: Write the failing operator-ownership test**

Create `tests/test_structured_module_boundaries.py` with:

```python
"""Architecture checks for the structured-solver compatibility facade."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

import superglm.solvers.structured as structured

_SOLVER_DIR = Path(structured.__file__).resolve().parent


def _assert_owned(module_name: str, symbols: tuple[str, ...]) -> None:
    path = _SOLVER_DIR / "_structured" / f"{module_name}.py"
    assert path.is_file(), f"missing structured owner: {path}"
    owner = import_module(f"superglm.solvers._structured.{module_name}")
    for symbol in symbols:
        assert getattr(structured, symbol) is getattr(owner, symbol)


def test_compact_operators_have_internal_owner() -> None:
    _assert_owned(
        "operators",
        (
            "SymmetricBlockOperator",
            "BlockSymmetricOperator",
            "SumToZeroBlockOperator",
            "CenteredBlockOperator",
            "LowRankSymmetricOperator",
            "SumBlockOperator",
            "CompactSymmetricOperator",
            "_BlockDiagonalLowRank",
            "_operator_bdlr",
            "_trace_symmetric_bdlr",
            "materialize_compact_operator",
            "compact_operator_diagonal",
        ),
    )
```

- [ ] **Step 2: Run the test and verify the expected red failure**

Run:

```bash
rtk pytest tests/test_structured_module_boundaries.py -q
```

Expected: FAIL because `_structured/operators.py` does not exist.

- [ ] **Step 3: Move the operator layer without editing bodies**

Create an empty `src/superglm/solvers/_structured/__init__.py`. Move these
complete definitions and assignments from `structured.py` into
`_structured/operators.py`:

```text
SymmetricBlockOperator
BlockSymmetricOperator
SumToZeroBlockOperator
CenteredBlockOperator
LowRankSymmetricOperator
SumBlockOperator
CompactSymmetricOperator
_DiagonalLowRank
_GeneralDiagonalLowRank
_block_operator_dlr
_merge_dlr
_operator_dlr
_trace_symmetric_dlr
_multiply_symmetric_dlr
_general_dlr_diagonal
_general_dlr_square_diagonal
_trace_general_product
_BlockDiagonalLowRank
_GeneralBlockDiagonalLowRank
_apply_local_blocks
_block_operator_bdlr
_sum_to_zero_operator_bdlr
_empty_block_part
_merge_bdlr
_operator_bdlr
_trace_symmetric_bdlr
_multiply_symmetric_bdlr
_multiply_symmetric_bdlr_coalesced
_general_bdlr_diagonal
_general_bdlr_square_diagonal
_trace_general_bdlr_product
materialize_compact_operator
compact_operator_diagonal
```

Use this import header:

```python
"""Compact structured operators and low-rank algebra."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.linalg
from numpy.typing import NDArray

from superglm.factor_smooth_geometry import (
    adjoint_sum_to_zero_blocks,
    expand_sum_to_zero_blocks,
)
```

Import the moved names explicitly near the top of `structured.py`. In
`sum_to_zero.py`, replace its import from `superglm.solvers.structured` with:

```python
from superglm.solvers._structured.operators import (
    BlockSymmetricOperator,
    CompactSymmetricOperator,
    SumToZeroBlockOperator,
    _BlockDiagonalLowRank,
    _general_bdlr_diagonal,
    _general_bdlr_square_diagonal,
    _multiply_symmetric_bdlr_coalesced,
    _operator_bdlr,
    _trace_general_bdlr_product,
    _trace_symmetric_bdlr,
)
```

- [ ] **Step 4: Run focused operator and SZ tests**

Run:

```bash
rtk pytest \
  tests/test_structured_module_boundaries.py \
  tests/test_structured_factor.py \
  tests/test_block_schur_factor.py \
  tests/test_sum_to_zero_structured_factor.py -q
rtk ruff check \
  src/superglm/solvers/_structured/operators.py \
  src/superglm/solvers/structured.py \
  src/superglm/solvers/sum_to_zero.py \
  tests/test_structured_module_boundaries.py
```

Expected: all tests and Ruff pass.

- [ ] **Step 5: Commit the operator extraction**

Run:

```bash
rtk git add src/superglm/solvers tests/test_structured_module_boundaries.py
rtk git commit -m "Extract structured operator algebra"
```

### Task 3: Extract centered and sum-to-zero geometry

**Files:**

- Create: `src/superglm/solvers/_structured/geometry.py`
- Modify: `src/superglm/solvers/structured.py`
- Modify: `tests/test_structured_module_boundaries.py`
- Modify: `tests/test_structured_factor.py`
- Modify: `tests/test_sum_to_zero_structured_factor.py`

- [ ] **Step 1: Append the failing geometry-ownership test**

Append:

```python
def test_estimability_geometry_has_internal_owner() -> None:
    _assert_owned(
        "geometry",
        (
            "_bounded_centered_estimability",
            "_orthonormal_column_span",
            "_sum_to_zero_public_null_geometry",
            "_certified_ritz_discarded",
            "centered_operator_coefficient_estimable",
        ),
    )
```

- [ ] **Step 2: Verify red**

Run:

```bash
rtk pytest tests/test_structured_module_boundaries.py::test_estimability_geometry_has_internal_owner -q
```

Expected: FAIL because `_structured/geometry.py` does not exist.

- [ ] **Step 3: Move the complete geometry block**

Move `_MAX_DENSE_CENTERED_ESTIMABILITY_WIDTH` and every complete definition
from `_bounded_centered_estimability` through
`centered_operator_coefficient_estimable`, plus
`_coefficient_estimable_from_null_basis`, into `geometry.py`. Do not move
`compact_operator_diagonal`, which Task 2 already assigned to `operators.py`.

Use this import header:

```python
"""Rank-aware estimability geometry for compact structured operators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
from numpy.typing import NDArray

from superglm.solvers._structured.operators import (
    BlockSymmetricOperator,
    CenteredBlockOperator,
    SumToZeroBlockOperator,
    SymmetricBlockOperator,
    compact_operator_diagonal,
    materialize_compact_operator,
)
from superglm.solvers.rank import (
    SHARED_RANK_POLICY,
    RankDecomposition,
    decompose_gram,
    needs_factor_certification,
)
```

Import every moved name required by the remaining monolith explicitly from
`geometry.py`.

- [ ] **Step 4: Retarget implementation-detail monkeypatches**

In both geometry test files, replace:

```python
"superglm.solvers.structured._bounded_centered_estimability"
```

with:

```python
"superglm.solvers._structured.geometry._bounded_centered_estimability"
```

Do not add a facade forwarding shim.

- [ ] **Step 5: Run the geometry suite**

Run:

```bash
rtk pytest \
  tests/test_structured_module_boundaries.py \
  tests/test_structured_factor.py \
  tests/test_sum_to_zero_structured_factor.py -q
rtk ruff check \
  src/superglm/solvers/_structured/geometry.py \
  src/superglm/solvers/structured.py \
  tests/test_structured_module_boundaries.py \
  tests/test_structured_factor.py \
  tests/test_sum_to_zero_structured_factor.py
```

Expected: all tests and Ruff pass.

- [ ] **Step 6: Commit the geometry extraction**

Run:

```bash
rtk git add src/superglm/solvers tests/test_structured_module_boundaries.py \
  tests/test_structured_factor.py tests/test_sum_to_zero_structured_factor.py
rtk git commit -m "Extract structured estimability geometry"
```

### Task 4: Extract scalar and block Schur factors

**Files:**

- Create: `src/superglm/solvers/_structured/factors.py`
- Modify: `src/superglm/solvers/structured.py`
- Modify: `tests/test_structured_module_boundaries.py`

- [ ] **Step 1: Append the failing factor-ownership test**

Append:

```python
def test_schur_factors_have_internal_owner() -> None:
    _assert_owned(
        "factors",
        (
            "ScalarSchurFactor",
            "BlockSchurFactor",
            "ProfiledBlockSchurFactor",
            "ProfiledScalarSchurFactor",
        ),
    )
```

- [ ] **Step 2: Verify red**

Run:

```bash
rtk pytest tests/test_structured_module_boundaries.py::test_schur_factors_have_internal_owner -q
```

Expected: FAIL because `_structured/factors.py` does not exist.

- [ ] **Step 3: Move the four factor classes unchanged**

Move the complete definitions of `ScalarSchurFactor`, `BlockSchurFactor`,
`ProfiledBlockSchurFactor`, and `ProfiledScalarSchurFactor` into `factors.py`.
Use:

```python
"""Scalar and block Schur factorizations for structured systems."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import scipy.linalg
from numpy.typing import NDArray

from superglm.solvers._structured.geometry import (
    _coefficient_estimable_from_null_basis,
)
from superglm.solvers._structured.operators import (
    BlockSymmetricOperator,
    CenteredBlockOperator,
    CompactSymmetricOperator,
    SymmetricBlockOperator,
    _BlockDiagonalLowRank,
    _DiagonalLowRank,
    _general_bdlr_diagonal,
    _general_bdlr_square_diagonal,
    _general_dlr_diagonal,
    _general_dlr_square_diagonal,
    _multiply_symmetric_bdlr,
    _multiply_symmetric_dlr,
    _operator_bdlr,
    _operator_dlr,
    _trace_general_bdlr_product,
    _trace_general_product,
    _trace_symmetric_bdlr,
    _trace_symmetric_dlr,
)
from superglm.solvers.hessian_factor import _component_indices, _component_omega
from superglm.types import PenaltyComponent
```

Import the four moved classes explicitly in `structured.py`.

- [ ] **Step 4: Run factor tests**

Run:

```bash
rtk pytest \
  tests/test_structured_module_boundaries.py \
  tests/test_structured_factor.py \
  tests/test_block_schur_factor.py \
  tests/test_sum_to_zero_structured_factor.py -q
rtk ruff check \
  src/superglm/solvers/_structured/factors.py \
  src/superglm/solvers/structured.py \
  tests/test_structured_module_boundaries.py
```

Expected: all tests and Ruff pass.

- [ ] **Step 5: Commit the factor extraction**

Run:

```bash
rtk git add src/superglm/solvers tests/test_structured_module_boundaries.py
rtk git commit -m "Extract structured Schur factors"
```

### Task 5: Extract backend selection

**Files:**

- Create: `src/superglm/solvers/_structured/selection.py`
- Modify: `src/superglm/solvers/structured.py`
- Modify: `tests/test_structured_module_boundaries.py`

- [ ] **Step 1: Append the failing selection-ownership test**

Append:

```python
def test_backend_selection_has_internal_owner() -> None:
    _assert_owned(
        "selection",
        (
            "StructuredGroupSelection",
            "StructuredBackendDecision",
            "select_structured_group",
            "resolve_structured_backend",
        ),
    )
```

- [ ] **Step 2: Verify red**

Run:

```bash
rtk pytest tests/test_structured_module_boundaries.py::test_backend_selection_has_internal_owner -q
```

Expected: FAIL because `_structured/selection.py` does not exist.

- [ ] **Step 3: Move selection definitions and their two policy constants**

Move:

```text
StructuredGroupSelection
StructuredBackendDecision
_AUTO_MIN_COEFFICIENT_WIDTH
_AUTO_MAX_STRUCTURED_COST_RATIO
_structured_auto_is_beneficial
_block_structured_auto_is_beneficial
_sum_to_zero_structured_auto_is_beneficial
_selection_failure
select_structured_group
_factor_smooth_singular_local_level
resolve_structured_backend
```

Use:

```python
"""Eligibility and cost policy for the structured direct backend."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from superglm.group_matrix import (
    FactorSmoothGroupMatrix,
    GroupMatrix,
    RandomEffectGroupMatrix,
)
from superglm.types import GroupSlice, PenaltyComponent
```

Import the moved public and repository-private names explicitly in
`structured.py`.

- [ ] **Step 4: Run backend selection and IRLS tests**

Run:

```bash
rtk pytest \
  tests/test_structured_module_boundaries.py \
  tests/test_structured_allocations.py \
  tests/test_structured_irls.py \
  tests/test_random_effect_discrete.py \
  tests/test_factor_smooth_discrete.py -q
rtk ruff check \
  src/superglm/solvers/_structured/selection.py \
  src/superglm/solvers/structured.py \
  tests/test_structured_module_boundaries.py
```

Expected: all tests and Ruff pass.

- [ ] **Step 5: Commit the selection extraction**

Run:

```bash
rtk git add src/superglm/solvers tests/test_structured_module_boundaries.py
rtk git commit -m "Extract structured backend selection"
```

### Task 6: Extract layouts and sufficient-statistic assembly

**Files:**

- Create: `src/superglm/solvers/_structured/layout.py`
- Create: `src/superglm/solvers/_structured/moments.py`
- Modify: `src/superglm/solvers/structured.py`
- Modify: `tests/test_structured_module_boundaries.py`
- Modify: `tests/test_structured_allocations.py`

- [ ] **Step 1: Append failing ownership tests**

Append:

```python
def test_structured_layouts_have_internal_owner() -> None:
    _assert_owned(
        "layout",
        (
            "ScalarStructuredLayout",
            "BlockStructuredLayout",
            "get_structured_layout",
            "structured_design_matvec",
            "structured_design_rmatvec",
        ),
    )


def test_structured_moments_have_internal_owner() -> None:
    _assert_owned(
        "moments",
        (
            "ScalarStructuredSystem",
            "BlockStructuredSystem",
            "SumToZeroBlockStructuredSystem",
            "build_scalar_structured_system",
            "build_block_structured_system",
            "build_structured_system",
        ),
    )
```

- [ ] **Step 2: Verify red**

Run:

```bash
rtk pytest \
  tests/test_structured_module_boundaries.py::test_structured_layouts_have_internal_owner \
  tests/test_structured_module_boundaries.py::test_structured_moments_have_internal_owner -q
```

Expected: two failures because the owner modules do not exist.

- [ ] **Step 3: Move layout records and operations**

Move:

```text
ScalarStructuredLayout
BlockStructuredLayout
_MAX_FUSED_DENSE_SMALL_WIDTH
_validate_structured_inputs
build_scalar_structured_layout
get_scalar_structured_layout
build_block_structured_layout
get_block_structured_layout
get_structured_layout
structured_design_matvec
structured_design_rmatvec
```

Use:

```python
"""Coefficient layouts and design products for structured systems."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from superglm._group_matrix._group_matrix_execution import MatrixExecutionPlan
from superglm.group_matrix import (
    DenseGroupMatrix,
    DesignMatrix,
    FactorSmoothGroupMatrix,
    GroupMatrix,
    RandomEffectGroupMatrix,
)
from superglm.types import GroupSlice
```

- [ ] **Step 4: Move system records and moment construction**

Move:

```text
ScalarStructuredSystem
BlockStructuredSystem
SumToZeroBlockStructuredSystem
build_scalar_structured_system
_optimized_discrete_factor_smooth_cross
build_block_structured_system
build_structured_system
```

Use:

```python
"""Compact sufficient-statistic assembly for structured systems."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from superglm._group_matrix._group_matrix_algebra import (
    _cross_gram,
    _random_effect_cross_gram,
)
from superglm._group_matrix._group_matrix_kernels import (
    _dense_small_weighted_moments,
    _random_effect_sufficient_stats,
)
from superglm.factor_smooth_geometry import adjoint_sum_to_zero_blocks
from superglm.group_matrix import (
    DenseGroupMatrix,
    FactorSmoothGroupMatrix,
    GroupMatrix,
)
from superglm.solvers._structured.layout import (
    BlockStructuredLayout,
    ScalarStructuredLayout,
    build_block_structured_layout,
    build_scalar_structured_layout,
)
from superglm.solvers._structured.operators import (
    BlockSymmetricOperator,
    SumToZeroBlockOperator,
    SymmetricBlockOperator,
)
from superglm.types import GroupSlice
```

Import all moved repository-visible names explicitly in `structured.py`.

- [ ] **Step 5: Retarget the allocation patch**

In `tests/test_structured_allocations.py`, replace:

```python
import superglm.solvers.structured as structured_module
```

with:

```python
import superglm.solvers._structured.moments as structured_moments
```

and replace:

```python
monkeypatch.setattr(structured_module.np, "zeros", guarded_zeros)
```

with:

```python
monkeypatch.setattr(structured_moments.np, "zeros", guarded_zeros)
```

- [ ] **Step 6: Run layout, allocation, and moment tests**

Run:

```bash
rtk pytest \
  tests/test_structured_module_boundaries.py \
  tests/test_structured_allocations.py \
  tests/test_random_effect_discrete.py \
  tests/test_factor_smooth_structured_system.py \
  tests/test_factor_smooth_discrete.py \
  tests/test_factor_smooth_sz_reml.py -q
rtk ruff check \
  src/superglm/solvers/_structured/layout.py \
  src/superglm/solvers/_structured/moments.py \
  src/superglm/solvers/structured.py \
  tests/test_structured_module_boundaries.py \
  tests/test_structured_allocations.py
```

Expected: all tests and Ruff pass.

- [ ] **Step 7: Commit layouts and moments**

Run:

```bash
rtk git add src/superglm/solvers tests/test_structured_module_boundaries.py \
  tests/test_structured_allocations.py
rtk git commit -m "Extract structured layouts and moments"
```

### Task 7: Extract assembly and retained state; finish the facade

**Files:**

- Create: `src/superglm/solvers/_structured/assembly.py`
- Create: `src/superglm/solvers/_structured/state.py`
- Modify: `src/superglm/solvers/structured.py`
- Modify: `tests/test_structured_module_boundaries.py`

- [ ] **Step 1: Append the final failing ownership and facade tests**

Append:

```python
def test_penalized_assembly_has_internal_owner() -> None:
    _assert_owned(
        "assembly",
        (
            "CachedScalarStructuredSolution",
            "CachedBlockStructuredSolution",
            "CachedSumToZeroStructuredSolution",
            "build_penalized_structured_operator",
            "build_augmented_structured_factor",
            "solve_cached_structured",
        ),
    )


def test_retained_structured_state_has_internal_owner() -> None:
    _assert_owned(
        "state",
        (
            "StructuredLevelSupport",
            "FactorSmoothLevelSupport",
            "StructuredLinearSystemState",
        ),
    )


def test_structured_module_is_implementation_free_facade() -> None:
    import ast

    tree = ast.parse(Path(structured.__file__).read_text())
    implementations = [
        node.name
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    assert implementations == []
    assert structured.__all__
```

- [ ] **Step 2: Verify red**

Run:

```bash
rtk pytest \
  tests/test_structured_module_boundaries.py::test_penalized_assembly_has_internal_owner \
  tests/test_structured_module_boundaries.py::test_retained_structured_state_has_internal_owner \
  tests/test_structured_module_boundaries.py::test_structured_module_is_implementation_free_facade -q
```

Expected: three failures because assembly/state do not exist and the facade
still contains implementations.

- [ ] **Step 3: Move assembly records and operations**

Move:

```text
CachedScalarStructuredSolution
CachedBlockStructuredSolution
CachedSumToZeroStructuredSolution
_lambda_for_component
_dense_component_omega
build_penalized_scalar_operator
build_penalized_block_operator
build_penalized_sum_to_zero_operator
build_penalized_structured_operator
build_augmented_scalar_factor
build_augmented_block_factor
build_augmented_sum_to_zero_factor
build_augmented_structured_factor
solve_cached_scalar_structured
solve_cached_block_structured
solve_cached_sum_to_zero_structured
solve_cached_structured
```

Use:

```python
"""Penalty assembly and cached solves for structured systems."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from superglm.group_matrix import (
    FactorSmoothGroupMatrix,
    GroupMatrix,
    RandomEffectGroupMatrix,
)
from superglm.solvers._structured.factors import (
    BlockSchurFactor,
    ProfiledBlockSchurFactor,
    ProfiledScalarSchurFactor,
    ScalarSchurFactor,
)
from superglm.solvers._structured.moments import (
    BlockStructuredSystem,
    ScalarStructuredSystem,
    SumToZeroBlockStructuredSystem,
)
from superglm.solvers._structured.operators import (
    BlockSymmetricOperator,
    SumToZeroBlockOperator,
    SymmetricBlockOperator,
)
from superglm.solvers.hessian_factor import _component_indices
from superglm.types import GroupSlice, PenaltyComponent

if TYPE_CHECKING:
    from superglm.solvers.sum_to_zero import ProfiledSumToZeroBlockFactor
```

Keep the imports of `SumToZeroBlockFactor` and
`ProfiledSumToZeroBlockFactor` local to their existing functions.

- [ ] **Step 4: Move retained support and fit-state records**

Move:

```text
StructuredLevelSupport
FactorSmoothLevelSupport
StructuredLinearSystemState
```

Use:

```python
"""Retained support and linear-system state for structured fits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from superglm.solvers._structured.factors import (
    BlockSchurFactor,
    ProfiledBlockSchurFactor,
    ProfiledScalarSchurFactor,
    ScalarSchurFactor,
)
from superglm.solvers._structured.moments import (
    BlockStructuredSystem,
    ScalarStructuredSystem,
    SumToZeroBlockStructuredSystem,
)
from superglm.solvers._structured.operators import (
    BlockSymmetricOperator,
    CenteredBlockOperator,
    SumToZeroBlockOperator,
    SymmetricBlockOperator,
)

if TYPE_CHECKING:
    from superglm.solvers.sum_to_zero import (
        ProfiledSumToZeroBlockFactor,
        SumToZeroBlockFactor,
    )
```

- [ ] **Step 5: Replace `structured.py` with an explicit facade**

The file contains only its module docstring, future annotations import,
explicit imports from all eight owners, and `__all__`. Re-export all 116
top-level definitions, aliases, and policy constants from the numerical
baseline. Generate the exact compatibility inventory before replacing the
file with:

```bash
rtk proxy python - <<'PY'
import ast
import subprocess

source = subprocess.run(
    ["git", "show", "b3cf321:src/superglm/solvers/structured.py"],
    check=True,
    capture_output=True,
    text=True,
).stdout
tree = ast.parse(source)
names = []
for node in tree.body:
    if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
        names.append(node.name)
    elif isinstance(node, ast.Assign):
        names.extend(
            target.id for target in node.targets if isinstance(target, ast.Name)
        )
for name in sorted(names):
    print(name)
assert len(names) == 116
PY
```

Use that exact sorted inventory for explicit owner imports and `__all__`.
This includes private compatibility names even though wildcard imports
normally omit them.

- [ ] **Step 6: Run the full focused structured suite**

Run:

```bash
rtk proxy env \
  PYTHONPATH=/home/mhick/python_projects/superglm/.worktrees/structured-credibility/src \
  /home/mhick/python_projects/superglm/.worktrees/structured-credibility/.venv/bin/python \
  -m pytest \
  tests/test_structured_module_boundaries.py \
  tests/test_random_effect*.py \
  tests/test_factor_smooth*.py \
  tests/test_structured*.py \
  tests/test_sum_to_zero_structured_factor.py \
  tests/test_block_schur_factor.py \
  tests/test_reml_newton_fixes.py \
  tests/test_cached_w_validation.py \
  tests/test_fit_state_retention.py -q
rtk ruff check src/superglm/solvers tests/test_structured_module_boundaries.py
```

Expected: all tests and Ruff pass.

- [ ] **Step 7: Commit the final facade**

Run:

```bash
rtk git add src/superglm/solvers tests/test_structured_module_boundaries.py
rtk git commit -m "Finish structured solver compatibility facade"
```

### Task 8: Prove the move was mechanical and behavior-compatible

**Files:**

- Compare: `b3cf321:src/superglm/solvers/structured.py`
- Read: `src/superglm/solvers/_structured/*.py`
- Read: `src/superglm/solvers/structured.py`

- [ ] **Step 1: Compare every moved AST definition with the numerical baseline**

Run:

```bash
rtk proxy python - <<'PY'
import ast
import subprocess
from collections import defaultdict
from pathlib import Path

old_source = subprocess.run(
    [
        "git",
        "show",
        "b3cf321:src/superglm/solvers/structured.py",
    ],
    check=True,
    capture_output=True,
    text=True,
).stdout
old_tree = ast.parse(old_source)
old = {
    node.name: ast.dump(node, include_attributes=False)
    for node in old_tree.body
    if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
}

new = defaultdict(list)
for path in sorted(Path("src/superglm/solvers/_structured").glob("*.py")):
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            new[node.name].append((path, ast.dump(node, include_attributes=False)))

assert old.keys() == new.keys(), (sorted(old.keys() - new.keys()), sorted(new.keys() - old.keys()))
duplicates = {name: values for name, values in new.items() if len(values) != 1}
assert not duplicates, duplicates
changed = [
    name
    for name, old_dump in old.items()
    if new[name][0][1] != old_dump
]
assert not changed, changed
print(f"AST-identical definitions: {len(old)}")
PY
```

Expected: `AST-identical definitions: 111`.

- [ ] **Step 2: Prove facade imports and internal direction**

Run:

```bash
rtk pytest tests/test_structured_module_boundaries.py -q
rtk rg -n \"from superglm\\.solvers\\.structured import\" \
  src/superglm/solvers/_structured src/superglm/solvers/sum_to_zero.py \
  --glob '*.py'
rtk wc -l src/superglm/solvers/structured.py \
  src/superglm/solvers/_structured/*.py
```

Expected: boundary tests pass; `rg` returns no matches; the facade is small and
no implementation owner approaches the original 5,920 lines.

- [ ] **Step 3: Run import and serialization checks**

Run:

```bash
rtk pytest \
  tests/test_factor_smooth_inference.py::test_factor_smooth_prediction_survives_released_state_and_pickle \
  tests/test_fit_state_retention.py \
  tests/test_core.py::TestClassIdentity::test_pickle_round_trip -q
```

Expected: all serialization checks pass.

### Task 9: Run complete quality, numerical, and performance gates

**Files:**

- Output: `/tmp/superglm-structured-split-final-fs/`
- Output: `/tmp/superglm-structured-split-final-sz/`

- [ ] **Step 1: Run static and complete repository checks**

Run:

```bash
rtk ruff check src/ tests/
rtk proxy uv run ruff format --check src/ tests/
rtk git diff --check
rtk proxy uv lock --check
rtk proxy uv pip check
rtk pytest -m "not slow" tests/ -q
rtk pytest tests/ -q
```

Expected: all commands pass.

- [ ] **Step 2: Repeat the FS benchmark**

Run:

```bash
rtk proxy env \
  PYTHONPATH=/home/mhick/python_projects/superglm/.worktrees/structured-credibility/src \
  /home/mhick/python_projects/superglm/.worktrees/structured-credibility/.venv/bin/python \
  benchmarks/profile_structured_credibility.py \
  --n 1000000 --levels 300 --family poisson --discrete \
  --random-effects 0 --small-width 4 --structured-term factor_smooth \
  --block-size 10 --factor-basis fs --global-spline --weights nonuniform \
  --backend structured --repetitions 5 --warmups 1 --max-reml-iter 20 \
  --reml-tol 1e-7 --seed 20260724 --sample-interval-ms 250 \
  --no-cprofile --no-tracemalloc --no-dense-parity \
  --output-dir /tmp/superglm-structured-split-final-fs
```

- [ ] **Step 3: Repeat the SZ benchmark**

Run:

```bash
rtk proxy env \
  PYTHONPATH=/home/mhick/python_projects/superglm/.worktrees/structured-credibility/src \
  /home/mhick/python_projects/superglm/.worktrees/structured-credibility/.venv/bin/python \
  benchmarks/profile_structured_credibility.py \
  --n 1000000 --levels 300 --family poisson --discrete \
  --random-effects 0 --small-width 4 --structured-term factor_smooth \
  --block-size 10 --factor-basis sz --global-spline --weights nonuniform \
  --backend structured --repetitions 5 --warmups 1 --max-reml-iter 20 \
  --reml-tol 1e-7 --seed 20260724 --sample-interval-ms 250 \
  --no-cprofile --no-tracemalloc --no-dense-parity \
  --output-dir /tmp/superglm-structured-split-final-sz
```

- [ ] **Step 4: Assert numerical and 5% performance parity**

Run:

```bash
rtk proxy \
  /home/mhick/python_projects/superglm/.worktrees/structured-credibility/.venv/bin/python \
  - <<'PY'
import json
import statistics
from pathlib import Path

pairs = {
    "fs": (
        Path("/tmp/superglm-structured-split-baseline-fs/summary.json"),
        Path("/tmp/superglm-structured-split-final-fs/summary.json"),
    ),
    "sz": (
        Path("/tmp/superglm-structured-split-baseline-sz/summary.json"),
        Path("/tmp/superglm-structured-split-final-sz/summary.json"),
    ),
}
for basis, (before_path, after_path) in pairs.items():
    before = json.loads(before_path.read_text())["backends"]["structured"]
    after = json.loads(after_path.read_text())["backends"]["structured"]
    old = before["model"]
    new = after["model"]
    assert old["termination_reason"] == new["termination_reason"]
    assert old["reml_iterations"] == new["reml_iterations"]
    assert abs(old["objective"] - new["objective"]) < 1e-6
    assert abs(old["deviance"] - new["deviance"]) < 1e-6
    assert abs(old["effective_df"] - new["effective_df"]) < 1e-6
    assert abs(old["prediction_checksum"] - new["prediction_checksum"]) < 1e-5
    assert old["lambdas"].keys() == new["lambdas"].keys()
    assert max(
        abs(old["lambdas"][key] / new["lambdas"][key] - 1.0)
        for key in old["lambdas"]
    ) < 1e-9
    old_mean = statistics.mean(before["wall_times_s"])
    new_mean = statistics.mean(after["wall_times_s"])
    assert new_mean <= old_mean * 1.05, (basis, old_mean, new_mean)
    print(basis, old_mean, new_mean, 100.0 * (new_mean / old_mean - 1.0))
PY
```

Expected: exact numerical parity within the declared tolerances and no more
than 5% pooled mean wall-time regression for FS or SZ.

- [ ] **Step 5: Re-run release packaging gates**

Run:

```bash
rtk pytest \
  tests/test_version_metadata.py \
  tests/test_release_management.py \
  tests/test_release_packaging.py \
  tests/test_supply_chain_governance.py -q
rtk proxy uv build
rtk proxy uv run twine check dist/*
rtk proxy uv run check-wheel-contents dist/*.whl
rtk proxy uv run python scripts/verify_release_artifacts.py dist/*
```

Expected: version `0.15.0` metadata, tests, and both artifacts pass.

### Task 10: Push and complete the review/CI loop

**Files:**

- PR: `https://github.com/StrudelDoodleS/superglm/pull/165`

- [ ] **Step 1: Audit the final diff and branch**

Run:

```bash
rtk git status --short
rtk git diff --check
rtk git diff --stat b3cf321..HEAD
rtk git diff --name-only origin/master..HEAD
```

Expected: clean worktree; only structured-solver extraction, its tests/docs,
and the already approved 0.15.0 changes differ. No LSS paths appear.

- [ ] **Step 2: Push the branch**

Run:

```bash
rtk git push origin feature/structured-credibility
```

Expected: pre-push hooks and push pass.

- [ ] **Step 3: Request a fresh Codex review**

Post a new PR comment that tags `@codex review` and asks specifically for:

```text
Please review the behavior-preserving structured-solver package extraction.
Check facade import compatibility, internal import direction and cycles,
operator/geometry ownership, SZ shared-algebra imports, serialization,
allocation guards, and any numerical or performance regression.
```

- [ ] **Step 4: Wait at least 15 minutes and inspect review plus CI**

Use GitHub PR checks, review comments, and GraphQL thread state. Do not infer
cleanliness from a top-level summary when unresolved inline threads remain.

- [ ] **Step 5: Resolve all actionable feedback**

For each actionable comment: reproduce or verify it, add a failing regression
test when behavior is implicated, implement the smallest in-scope fix, run the
relevant focused gate, reply in the inline thread, and resolve the thread.
Push fixes and request another fresh Codex review.

- [ ] **Step 6: Finish only when the PR is clean**

Acceptance evidence:

- all GitHub checks green;
- Codex reports no major issue on the final head;
- zero unresolved actionable review threads;
- PR remains open and draft;
- branch worktree clean;
- master unchanged;
- LSS untouched.

Do not merge, mark ready, tag, or publish without a separate explicit user
instruction.
