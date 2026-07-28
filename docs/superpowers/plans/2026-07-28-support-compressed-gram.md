# Support-Compressed Gram Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make exact-path Gram formation exploit repeated design rows losslessly, so a tensor interaction stops costing 7.2× a baseline fit.

**Architecture:** Exact-path spline and tensor blocks are `SparseSSPGroupMatrix` — stored at full `n` rows even when the underlying covariate has few distinct values. `DiscretizedSSPGroupMatrix` already implements the compressed algebra we want (`B_unique`, `bin_idx`, bincount-aggregated gram) and already has fast dispatch paths in `_cross_gram`. We add a **lossless** sibling that reuses that machinery with exact unique-row indices instead of bins, and construct it whenever a group's distinct-row count is small enough to pay. Nothing about the fitted basis changes — only its storage — so results must be numerically identical.

**Tech Stack:** numpy, scipy.sparse, numba (existing kernels only — this plan adds no new kernels), pytest.

## Global Constraints

- `discrete=True` is the **lossy binned** fREML path and must not silently drift from exact. This plan's compression is **lossless deduplication**, never binning. The two must remain distinguishable in code, telemetry and `design_summary()`.
- `select=True` (mgcv double penalty) and `selection_penalty > 0` (sparse group penalty) are different tools; this plan touches neither.
- Spline `k` matches mgcv's basis dimension; built columns = k−1. Storage changes must not alter column counts.
- `sample_weight=` is exposure weight. `exposure=` must never be reintroduced.
- Public API surface is unchanged by this plan. No new user-facing keyword arguments.
- All performance claims verified with cProfile (project convention).
- Baseline to beat, recorded at `f082e9b`, n=100k real freMTPL2 (`docs/audit/2026-07-28/measured-tensor-cost.md`): baseline exact 8.79 s, +1 tensor exact 63.33 s.

---

## Verified Evidence Behind This Plan

Measured on the real fitted design (`docs/audit/2026-07-28/measured-tensor-cost.md` and the verification run):

| group | class | shape | stored nnz | density | **distinct rows** |
|---|---|---|---|---|---|
| DrivAge | SparseSSPGroupMatrix | (100000, 9) | 400,000 | 0.444 | **82** |
| VehAge | SparseSSPGroupMatrix | (100000, 9) | 400,000 | 0.444 | **55** |
| BonusMalus | SparseSSPGroupMatrix | (100000, 9) | 400,000 | 0.444 | **98** |
| VehPower | SparseSSPGroupMatrix | (100000, 9) | 400,000 | 0.444 | **12** |
| DrivAge:BonusMalus | SparseSSPGroupMatrix | (100000, 81) | **8,100,000** | **1.000** | **2,664** |

Two independent wastes on the tensor: it is stored **fully dense inside a CSR container** (density 1.000 — the row-Kronecker of two locally-supported marginals lost their sparsity), and it stores 100,000 rows when only 2,664 are distinct.

Direct measurement of the tensor block's Gram, same `W`:

```
superglm gm.gram(W)                561.1 ms
support-compressed equivalent        2.9 ms   rel_err 8.28e-15   speedup 193.5x
```

**193× exact.** No factorization, no approximation — purely exploiting that rows repeat.

**Why the win is available at all:** insurance rating variables are integer-valued. DrivAge has 82 distinct values, BonusMalus 98. Compression here is *lossless*, unlike `discrete=True` binning.

**Why this is low-risk:** `DiscretizedSSPGroupMatrix` (`_group_matrix_discretized.py:37-80`) already implements exactly this algebra, and `_cross_gram` (`_group_matrix_algebra.py:~700-760`) already routes discretized groups to fast paths. `SparseSSPGroupMatrix` currently falls into the `support_space_types` branch → `_cross_gram_by_columns`, the column-at-a-time loop the p-scaling profile measured at ~1.8M sparse matvecs.

**Deliberately NOT in this plan** (measured, negative or unverified):
- tabmat for cell aggregation — measured `CategoricalMatrix.sandwich` at **0.80×** vs `np.bincount` (n=1e6, L=3000). It is not the win here. tabmat remains interesting for categorical×dense cross blocks, which is separate work.
- The row-tensor / G-operator factorization (L2). Its timing looked strong (0.4 ms) but correctness was **not** verified — the per-cell sign gauge did not recover cleanly. L1 support compression captures 193× already; L2 is a follow-up only if profiling still shows tensor Gram on top.
- Interaction *discovery* (score-statistic screening). Independent subsystem; gets its own plan.

---

## File Structure

- **Create** `src/superglm/_group_matrix/_group_matrix_support.py` — support detection: given a CSR basis, decide whether compression pays and return `(B_unique, row_index)`. Pure function, no I/O, one responsibility.
- **Create** `tests/test_support_compression.py` — detection, class equivalence, dispatch, and parity tests.
- **Modify** `src/superglm/_group_matrix/_group_matrix_discretized.py` — add `SupportCompressedSSPGroupMatrix` subclass (identity marker only; inherits all algebra).
- **Modify** `src/superglm/dm_builder.py` — construct the compressed variant when detection says it pays.
- **Modify** `src/superglm/model/design_summary.py` — report compression honestly, distinct from `discrete=True`.

---

### Task 1: Support detection

**Files:**
- Create: `src/superglm/_group_matrix/_group_matrix_support.py`
- Test: `tests/test_support_compression.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `detect_row_support(B_csr: sp.spmatrix, max_ratio: float = 0.5) -> tuple[NDArray, NDArray] | None` returning `(B_unique_dense, row_index)` where `B_unique_dense` is `(n_support, p_b)` float64 and `row_index` is `(n,)` intp, or `None` when compression does not pay.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_support_compression.py
import numpy as np
import scipy.sparse as sp

from superglm._group_matrix._group_matrix_support import detect_row_support


def test_detect_row_support_compresses_repeated_rows():
    base = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 4.0]])
    rows = base[np.array([0, 1, 2, 0, 1, 0])]
    result = detect_row_support(sp.csr_matrix(rows))
    assert result is not None
    b_unique, row_index = result
    assert b_unique.shape == (3, 2)
    assert row_index.shape == (6,)
    np.testing.assert_allclose(b_unique[row_index], rows)


def test_detect_row_support_declines_when_rows_are_distinct():
    rows = np.arange(20.0).reshape(10, 2)
    assert detect_row_support(sp.csr_matrix(rows)) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_support_compression.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'superglm._group_matrix._group_matrix_support'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/superglm/_group_matrix/_group_matrix_support.py
"""Lossless row-support detection for factored SSP group matrices.

Exact-path spline and tensor bases repeat rows whenever the underlying
covariate is integer-valued or otherwise low-cardinality.  Storing one copy
per distinct row turns an O(n) weighted gram into an O(n) bincount plus an
O(n_support) dense gram, with no change to the basis itself.

This is deduplication, not binning: it introduces no discretization error and
is unrelated to ``discrete=True``.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

DEFAULT_MAX_SUPPORT_RATIO = 0.5


def detect_row_support(
    B_csr: sp.spmatrix, max_ratio: float = DEFAULT_MAX_SUPPORT_RATIO
) -> tuple[NDArray, NDArray] | None:
    """Return ``(B_unique, row_index)`` when row compression pays, else None.

    ``max_ratio`` is the largest distinct-row fraction worth compressing;
    above it the bookkeeping costs more than the saved arithmetic.
    """
    dense = np.asarray(B_csr.toarray(), dtype=np.float64)
    n_rows = dense.shape[0]
    if n_rows == 0:
        return None
    b_unique, row_index = np.unique(dense, axis=0, return_inverse=True)
    if b_unique.shape[0] > max_ratio * n_rows:
        return None
    return b_unique, np.asarray(row_index, dtype=np.intp).ravel()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_support_compression.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/superglm/_group_matrix/_group_matrix_support.py tests/test_support_compression.py
git commit -m "feat: add lossless row-support detection for SSP bases"
```

---

### Task 2: Support-compressed group matrix class

**Files:**
- Modify: `src/superglm/_group_matrix/_group_matrix_discretized.py`
- Test: `tests/test_support_compression.py`

**Interfaces:**
- Consumes: `detect_row_support` from Task 1.
- Produces: `SupportCompressedSSPGroupMatrix(B_unique, R_inv, row_index)` — subclass of `DiscretizedSSPGroupMatrix`, adds only the class identity and the property `is_lossless_support -> bool` (always `True`). All algebra (`gram`, `matvec`, `rmatvec`, `gram_rmatvec`, `toarray`, `row_subset`) is inherited unchanged.

- [ ] **Step 1: Write the failing test**

```python
def test_support_compressed_gram_matches_sparse_ssp():
    import scipy.sparse as sp
    from superglm._group_matrix._group_matrix_core import SparseSSPGroupMatrix
    from superglm._group_matrix._group_matrix_discretized import (
        SupportCompressedSSPGroupMatrix,
    )
    from superglm._group_matrix._group_matrix_support import detect_row_support

    rng = np.random.default_rng(0)
    base = rng.normal(size=(40, 6))
    idx = rng.integers(0, 40, 5000)
    B = sp.csr_matrix(base[idx])
    R_inv = rng.normal(size=(6, 4))
    W = np.abs(rng.normal(1.0, 0.2, 5000))

    reference = SparseSSPGroupMatrix(B, R_inv)
    b_unique, row_index = detect_row_support(B)
    compressed = SupportCompressedSSPGroupMatrix(b_unique, R_inv, row_index)

    assert compressed.is_lossless_support is True
    assert compressed.shape == reference.shape
    np.testing.assert_allclose(compressed.gram(W), reference.gram(W), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(compressed.toarray(), reference.toarray(), rtol=1e-12, atol=1e-12)
    v = rng.normal(size=4)
    np.testing.assert_allclose(compressed.matvec(v), reference.matvec(v), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(compressed.rmatvec(W), reference.rmatvec(W), rtol=1e-12, atol=1e-12)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_support_compression.py::test_support_compressed_gram_matches_sparse_ssp -v`
Expected: FAIL with `ImportError: cannot import name 'SupportCompressedSSPGroupMatrix'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/superglm/_group_matrix/_group_matrix_discretized.py`:

```python
class SupportCompressedSSPGroupMatrix(DiscretizedSSPGroupMatrix):
    """Exact SSP basis stored one row per distinct row.

    Numerically identical to ``SparseSSPGroupMatrix`` over the same basis;
    only the storage differs.  Distinguished from its binned parent because
    ``discrete=True`` is a lossy fREML path and this is not: no binning
    occurs and no discretization error is introduced.
    """

    __slots__ = ()

    @property
    def is_lossless_support(self) -> bool:
        return True
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_support_compression.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/superglm/_group_matrix/_group_matrix_discretized.py tests/test_support_compression.py
git commit -m "feat: add SupportCompressedSSPGroupMatrix reusing discretized algebra"
```

---

### Task 3: Cross-gram dispatch inherits the fast paths

**Files:**
- Test: `tests/test_support_compression.py`
- Modify (only if the test fails): `src/superglm/_group_matrix/_group_matrix_algebra.py`

**Interfaces:**
- Consumes: `SupportCompressedSSPGroupMatrix` from Task 2.
- Produces: no new symbols. Establishes that `_cross_gram` routes compressed groups through `_disc_disc_2d_hist` / `_agg_by_bin` rather than `_cross_gram_by_columns`.

Because `SupportCompressedSSPGroupMatrix` subclasses `DiscretizedSSPGroupMatrix`, the existing `isinstance` dispatch in `_cross_gram` should already select the fast disc×disc and disc×other branches. This task **verifies that** and fixes the dispatch only if it does not hold — the `support_space_types` branch is checked before the discretized branches in some orderings, which would shadow the fast path.

- [ ] **Step 1: Write the failing test**

```python
def test_cross_gram_uses_fast_path_for_compressed_groups():
    import scipy.sparse as sp
    from superglm._group_matrix._group_matrix_algebra import _cross_gram
    from superglm._group_matrix._group_matrix_core import SparseSSPGroupMatrix
    from superglm._group_matrix._group_matrix_discretized import (
        SupportCompressedSSPGroupMatrix,
    )
    from superglm._group_matrix._group_matrix_support import detect_row_support

    rng = np.random.default_rng(1)
    n = 4000
    out = {}

    def make(n_support, p_b, p_g, seed):
        gen = np.random.default_rng(seed)
        base = gen.normal(size=(n_support, p_b))
        idx = gen.integers(0, n_support, n)
        B = sp.csr_matrix(base[idx])
        R_inv = gen.normal(size=(p_b, p_g))
        b_unique, row_index = detect_row_support(B)
        return (
            SparseSSPGroupMatrix(B, R_inv),
            SupportCompressedSSPGroupMatrix(b_unique, R_inv, row_index),
        )

    ref_i, comp_i = make(30, 5, 3, 10)
    ref_j, comp_j = make(25, 4, 2, 11)
    W = np.abs(rng.normal(1.0, 0.2, n))

    expected = _cross_gram(ref_i, ref_j, W)
    actual = _cross_gram(comp_i, comp_j, W, profile=out)

    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)
    assert "block_cross_fallback_s" not in out, (
        f"compressed groups took the column-at-a-time fallback; profile={out}"
    )
```

- [ ] **Step 2: Run test to verify current behaviour**

Run: `uv run pytest tests/test_support_compression.py::test_cross_gram_uses_fast_path_for_compressed_groups -v`
Expected: either PASS (dispatch already correct — skip Step 3) or FAIL on the `block_cross_fallback_s` assertion.

- [ ] **Step 3: Fix dispatch ordering only if Step 2 failed**

In `_group_matrix_algebra.py`, the `support_space_types` branch must not shadow the discretized branches. Narrow it so compressed groups fall through:

```python
    support_space_types = (_SparseSSPGroupMatrix, FactorSmoothGroupMatrix, *SplineCatTypes)
    if (
        isinstance(gm_i, support_space_types) or isinstance(gm_j, support_space_types)
    ) and not (
        isinstance(gm_i, DiscretizedSSPGroupMatrix)
        and isinstance(gm_j, DiscretizedSSPGroupMatrix)
    ):
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_support_compression.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add tests/test_support_compression.py src/superglm/_group_matrix/_group_matrix_algebra.py
git commit -m "test: pin fast cross-gram dispatch for support-compressed groups"
```

---

### Task 4: Construct compressed groups during design-matrix build

**Files:**
- Modify: `src/superglm/dm_builder.py`
- Test: `tests/test_support_compression.py`

**Interfaces:**
- Consumes: `detect_row_support` (Task 1), `SupportCompressedSSPGroupMatrix` (Task 2).
- Produces: no new public symbols. After this task, `SuperGLM(...).​_build_design_matrix(...)` yields `SupportCompressedSSPGroupMatrix` for low-cardinality spline and tensor groups on the exact path.

- [ ] **Step 1: Write the failing test**

```python
def test_exact_path_builds_compressed_groups_for_integer_covariates():
    import pandas as pd
    from superglm import SuperGLM
    from superglm._group_matrix._group_matrix_discretized import (
        SupportCompressedSSPGroupMatrix,
    )
    from superglm.features.spline import Spline

    rng = np.random.default_rng(3)
    n = 5000
    X = pd.DataFrame(
        {
            "age": rng.integers(18, 90, n).astype(float),
            "bm": rng.integers(50, 130, n).astype(float),
        }
    )
    w = np.full(n, 0.5)
    y = rng.poisson(0.1, n) / w

    model = SuperGLM(
        family="poisson",
        selection_penalty=None,
        discrete=False,
        features={"age": Spline(kind="ps", k=10), "bm": Spline(kind="ps", k=10)},
    )
    model._add_interaction("age", "bm")
    model._build_design_matrix(X, y, w, None)

    kinds = {g.name: type(gm).__name__ for g, gm in zip(model._groups, model._dm.group_matrices)}
    assert kinds["age"] == "SupportCompressedSSPGroupMatrix", kinds
    assert kinds["age:bm"] == "SupportCompressedSSPGroupMatrix", kinds
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_support_compression.py::test_exact_path_builds_compressed_groups_for_integer_covariates -v`
Expected: FAIL with `AssertionError` showing `SparseSSPGroupMatrix`

- [ ] **Step 3: Write minimal implementation**

In `dm_builder.py`, wherever a `SparseSSPGroupMatrix` is constructed for the exact path, route through a helper. Add near the other construction helpers:

```python
def _build_ssp_group(B_csr, R_inv):
    """Build the cheapest exact representation of a factored SSP block.

    Compression is lossless deduplication of repeated rows; it never bins and
    is independent of ``discrete=True``.
    """
    from superglm._group_matrix._group_matrix_discretized import (
        SupportCompressedSSPGroupMatrix,
    )
    from superglm._group_matrix._group_matrix_support import detect_row_support

    detected = detect_row_support(B_csr)
    if detected is None:
        return SparseSSPGroupMatrix(B_csr, R_inv)
    b_unique, row_index = detected
    return SupportCompressedSSPGroupMatrix(b_unique, R_inv, row_index)
```

Then replace each exact-path `SparseSSPGroupMatrix(B, R_inv)` construction site with `_build_ssp_group(B, R_inv)`. Locate them with:

```bash
grep -n "SparseSSPGroupMatrix(" src/superglm/dm_builder.py
```

Carry over any attributes the call sites set afterwards (`omega`, `projection`, `omega_components`, `component_types`, `lambda_policies`) unchanged — both classes expose the same names.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_support_compression.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/superglm/dm_builder.py tests/test_support_compression.py
git commit -m "feat: build support-compressed SSP groups on the exact path"
```

---

### Task 5: Numerical parity gate

**Files:**
- Test: `tests/test_support_compression.py`

**Interfaces:**
- Consumes: everything from Tasks 1-4.
- Produces: no symbols. Establishes that compression changes no fitted quantity.

This is the task that protects the release. Compression must be invisible in results.

- [ ] **Step 1: Write the failing test**

```python
def test_compression_does_not_change_fitted_results(monkeypatch):
    import pandas as pd
    from superglm import SuperGLM
    from superglm._group_matrix import _group_matrix_support
    from superglm.features.spline import Spline

    rng = np.random.default_rng(7)
    n = 4000
    X = pd.DataFrame(
        {
            "age": rng.integers(18, 90, n).astype(float),
            "bm": rng.integers(50, 130, n).astype(float),
        }
    )
    w = rng.uniform(0.2, 1.0, n)
    y = rng.poisson(0.2, n) / w

    def fit():
        model = SuperGLM(
            family="poisson",
            selection_penalty=None,
            discrete=False,
            features={"age": Spline(kind="ps", k=10), "bm": Spline(kind="ps", k=10)},
        )
        model._add_interaction("age", "bm")
        return model.fit_reml(X, y, sample_weight=w)

    compressed = fit()
    monkeypatch.setattr(
        _group_matrix_support, "detect_row_support", lambda *a, **k: None
    )
    uncompressed = fit()

    np.testing.assert_allclose(
        compressed.result.beta, uncompressed.result.beta, rtol=1e-8, atol=1e-8
    )
    np.testing.assert_allclose(
        compressed.metrics().edf, uncompressed.metrics().edf, rtol=1e-6
    )
    np.testing.assert_allclose(
        compressed.result.deviance, uncompressed.result.deviance, rtol=1e-9
    )
```

- [ ] **Step 2: Run test to verify it fails or passes**

Run: `uv run pytest tests/test_support_compression.py::test_compression_does_not_change_fitted_results -v`
Expected: PASS. If it FAILS, compression has changed numerics — stop and diagnose with superpowers:systematic-debugging before proceeding. Do not loosen the tolerances to make it pass.

Note: `monkeypatch.setattr` on the module attribute only takes effect if `dm_builder._build_ssp_group` imports the function at call time (as written in Task 3, Step 3). Keep that import inside the function.

- [ ] **Step 3: Run the existing suites that cover this ground**

Run: `uv run pytest tests/test_theory_invariants.py tests/test_interactions.py tests/test_discretize_fit.py tests/test_reml.py -q`
Expected: PASS, same counts as on `origin/master`. Record the before/after counts in the commit message.

- [ ] **Step 4: Commit**

```bash
git add tests/test_support_compression.py
git commit -m "test: pin fitted-result parity for support-compressed groups"
```

---

### Task 6: Report compression honestly, and measure the win

**Files:**
- Modify: `src/superglm/model/design_summary.py`
- Test: `tests/test_support_compression.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `design_summary()` gains a `lossless_support` boolean column, distinct from any existing discrete/binned reporting.

- [ ] **Step 1: Write the failing test**

```python
def test_design_summary_distinguishes_lossless_support_from_discrete():
    import pandas as pd
    from superglm import SuperGLM
    from superglm.features.spline import Spline

    rng = np.random.default_rng(11)
    n = 3000
    X = pd.DataFrame({"age": rng.integers(18, 90, n).astype(float)})
    w = np.full(n, 1.0)
    y = rng.poisson(0.2, n)

    model = SuperGLM(
        family="poisson",
        selection_penalty=None,
        discrete=False,
        features={"age": Spline(kind="ps", k=10)},
    ).fit(X, y, sample_weight=w)

    summary = model.design_summary()
    assert "lossless_support" in summary.columns
    assert bool(summary.loc[summary["term"] == "age", "lossless_support"].iloc[0]) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_support_compression.py::test_design_summary_distinguishes_lossless_support_from_discrete -v`
Expected: FAIL with `AssertionError: 'lossless_support' not in columns` (adjust the `term` column name if `design_summary()` uses a different one — inspect with `model.design_summary().columns` first)

- [ ] **Step 3: Write minimal implementation**

In `design_summary.py`, where per-group rows are assembled, add:

```python
        "lossless_support": bool(getattr(group_matrix, "is_lossless_support", False)),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_support_compression.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Measure against the tracked baseline**

Run: `uv run python benchmarks/benchmark_tensor_cost.py --n 100000`

Expected, from `docs/audit/2026-07-28/measured-tensor-cost.md`: `tensor_cost_ti_exact` was **63.33 s** at `f082e9b`. The tensor block's Gram alone measured 561 ms → 2.9 ms in isolation, and Gram work is ~65% of the exact tensor fit, so expect a large drop — but **record what actually happens rather than asserting a target**. If the improvement is much smaller than the isolated measurement suggests, the Gram is no longer the bottleneck; re-profile with `--profile` and follow the numbers.

- [ ] **Step 6: Commit**

```bash
git add src/superglm/model/design_summary.py tests/test_support_compression.py
git commit -m "feat: report lossless support compression in design_summary"
```

---

## Follow-up work, deliberately out of scope

1. **Interaction discovery** — screening candidate pairs by an exact penalised score statistic computed from the same weighted cell aggregates. Independent subsystem, own plan, no dependency on this one.
2. **Row-tensor (G-operator) factorization** — only if profiling after Task 6 still shows tensor Gram on top. Requires verifying the factorization correctness that this plan's evidence did **not** establish.
3. **RFC-1 (`docs/audit/2026-07-28/architecture-audit.md`)** — the packed/mixed bin-space centering cliff. Note that Task 4 may partially unlock it for free: the packed path requires every group to be Discretized/Tensor/Categorical, and exact-path splines now present as a `DiscretizedSSPGroupMatrix` subclass. Worth re-checking the ladder dispatch counters after this lands before doing RFC-1 work.
