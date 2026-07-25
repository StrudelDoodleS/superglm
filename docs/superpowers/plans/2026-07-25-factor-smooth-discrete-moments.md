# FactorSmooth Discrete Moments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Experimentally replace repeated million-row discrete FactorSmooth
moment scans with compact cell aggregation, then retain bounded Numba
parallelism only if it produces an additional measured improvement.

**Architecture:** The dominant `FactorSmoothGroupMatrix` aggregates changing
PIRLS quantities by `(factor level, spline support bin)` in one compiled pass.
It derives raw local moments and optimized crosses for dense ordinary blocks
and a matching discretized global spline from those cells, while the
structured solver retains Tabmat for the ordinary-small system and keeps the
existing compact fallback for unsupported groups. A second, separately gated
Numba `prange` implementation uses private contiguous-chunk accumulators and a
fixed-order reduction without changing global thread settings.

**Tech Stack:** Python 3.10+, NumPy, SciPy, Numba, Tabmat, pytest, cProfile,
tracemalloc, Ruff, MkDocs

**Design:** `docs/superpowers/specs/2026-07-25-factor-smooth-discrete-moments-design.md`

**Worktree:** `/home/mhick/python_projects/superglm/.worktrees/structured-credibility`

---

## File Map

- Modify `src/superglm/_group_matrix/_group_matrix_kernels.py`
  - compiled serial and optional parallel cell reductions
  - compact cell-to-basis contractions
- Modify `src/superglm/_group_matrix/_group_matrix_core.py`
  - FactorSmooth cell-moment and optimized-cross methods
  - internal dispatch and natural-map contraction
- Modify `src/superglm/solvers/structured.py`
  - consume cell moments before constructing dominant/small crosses
  - preserve Tabmat ordinary moments and compact fallbacks
- Modify `tests/test_factor_smooth_discrete.py`
  - randomized raw cell algebra and parallel parity
- Modify `tests/test_factor_smooth_structured_system.py`
  - full structured-system parity and scan-count regression
- Modify `tests/test_structured_allocations.py`
  - forbid observation-by-small-width compatibility materialization
- Modify `docs/guide/fitting.md`
  - record only retained benchmark/profile evidence

## Experimental Discipline

Each production change begins with a failing test and is committed separately.
Task 4 is a keep/revert gate for serial cell aggregation. Task 6 is a separate
keep/revert gate for parallel reduction. Do not continue optimizing a path
that misses its gate.

Do not use Cython, Rust, C, or C++ extensions. Numba and Tabmat are explicitly
allowed. Do not modify any LSS path.

### Task 1: Reproduce and Archive the Baseline

**Files:**

- Verify: `/tmp/superglm-sz-1m-clean-rerun/summary.json`
- Verify: `/tmp/superglm-sz-1m-cprofile/cprofile_structured_top.txt`

- [ ] **Step 1: Verify the clean baseline**

```bash
rtk proxy jq \
  '{wall: .backends.structured.wall_times_s,
    converged: .backends.structured.model.converged,
    reml_iterations: .backends.structured.model.reml_iterations,
    checksum: .backends.structured.model.prediction_checksum}' \
  /tmp/superglm-sz-1m-clean-rerun/summary.json
```

Expected:

```text
wall = [7.694315768079832, 7.654054712038487]
converged = true
reml_iterations = 5
checksum = 1045155.2072170011
```

- [ ] **Step 2: Verify the profiled repeated-scan stack**

```bash
rtk grep -n \
  "build_block_structured_system\\|factor_smooth_dense_cross_gram\\|factor_smooth_sufficient_stats" \
  /tmp/superglm-sz-1m-cprofile/cprofile_structured_top.txt
```

Expected cumulative evidence:

```text
build_block_structured_system        3.910 s, 14 calls
factor_smooth_dense_cross_gram       2.418 s, 182 calls
factor_smooth_sufficient_stats       0.780 s, 14 calls
```

- [ ] **Step 3: Confirm the branch is clean before experimentation**

```bash
rtk git status --short
rtk git log -3 --oneline
```

Expected: clean worktree with the design/tool-boundary commits at the tip.

### Task 2: Add Serial Cell Sufficient Statistics

**Files:**

- Modify: `src/superglm/_group_matrix/_group_matrix_kernels.py`
- Modify: `src/superglm/_group_matrix/_group_matrix_core.py`
- Modify: `tests/test_factor_smooth_discrete.py`

- [ ] **Step 1: Add failing randomized cell-moment tests**

Add this local helper to `tests/test_factor_smooth_discrete.py`:

```python
def _cell_test_matrix(
    *,
    n: int,
    n_levels: int,
    block_size: int,
    factor_basis: str,
) -> FactorSmoothGroupMatrix:
    rng = np.random.default_rng(1907)
    support = np.linspace(-1.0, 1.0, 17)
    basis = np.column_stack(
        [support**power for power in range(block_size)]
    )
    natural_map = np.eye(block_size)
    natural_map[0, 1] = 0.15
    return FactorSmoothGroupMatrix(
        basis,
        bin_idx=rng.integers(0, len(support), size=n, dtype=np.intp),
        codes=rng.integers(0, n_levels, size=n, dtype=np.intp),
        n_levels=n_levels,
        natural_map=natural_map,
        levels=tuple(f"level-{index}" for index in range(n_levels)),
        repeated_penalty_components=(
            ("wiggle", np.eye(block_size, dtype=np.float64)),
        ),
        factor_basis=factor_basis,
    )
```

Add a parametrized test over `factor_basis in ("fs", "sz")` and
`signed in (False, True)`:

```python
@pytest.mark.parametrize("factor_basis", ["fs", "sz"])
@pytest.mark.parametrize("signed", [False, True])
def test_discrete_cell_sufficient_stats_match_dense_reference(
    factor_basis: str,
    signed: bool,
) -> None:
    rng = np.random.default_rng(8351)
    gm = _cell_test_matrix(
        n=513,
        n_levels=11,
        block_size=6,
        factor_basis=factor_basis,
    )
    W = rng.uniform(0.1, 1.7, size=gm.shape[0])
    if signed:
        W[::9] *= -0.4
    Wz = rng.normal(size=gm.shape[0])

    cell_weights, D, xtw, xtwz = gm.factor_smooth_discrete_cell_moments(W, Wz)
    raw_design = gm.B_unique[gm.bin_idx]
    for level in range(gm.n_levels):
        rows = gm.codes == level
        expected_D = raw_design[rows].T @ (W[rows, None] * raw_design[rows])
        expected_xtw = raw_design[rows].T @ W[rows]
        expected_xtwz = raw_design[rows].T @ Wz[rows]
        np.testing.assert_allclose(
            D[level],
            gm.natural_map.T @ expected_D @ gm.natural_map,
            rtol=2.0e-12,
            atol=2.0e-12,
        )
        np.testing.assert_allclose(
            xtw[level],
            expected_xtw @ gm.natural_map,
            rtol=2.0e-12,
            atol=2.0e-12,
        )
        np.testing.assert_allclose(
            xtwz[level],
            expected_xtwz @ gm.natural_map,
            rtol=2.0e-12,
            atol=2.0e-12,
        )
    assert cell_weights.shape == (gm.n_levels, gm.B_unique.shape[0])
```

- [ ] **Step 2: Run and verify RED**

```bash
rtk uv run pytest \
  tests/test_factor_smooth_discrete.py::test_discrete_cell_sufficient_stats_match_dense_reference \
  -q
```

Expected: FAIL with missing
`FactorSmoothGroupMatrix.factor_smooth_discrete_cell_moments`.

- [ ] **Step 3: Implement the serial compiled cell kernel**

In `_group_matrix_kernels.py`, add:

```python
@njit(cache=True)
def _factor_smooth_support_cell_sufficient_stats(
    basis,
    bin_idx,
    codes,
    weights,
    rhs,
    n_levels,
):
    n_bins = basis.shape[0]
    width = basis.shape[1]
    cell_weights = np.zeros((n_levels, n_bins))
    cell_rhs = np.zeros((n_levels, n_bins))
    for row in range(len(codes)):
        level = codes[row]
        support = bin_idx[row]
        cell_weights[level, support] += weights[row]
        cell_rhs[level, support] += rhs[row]

    gram = np.zeros((n_levels, width, width))
    xtw = np.zeros((n_levels, width))
    xt_rhs = np.zeros((n_levels, width))
    for level in range(n_levels):
        for support in range(n_bins):
            weight = cell_weights[level, support]
            rhs_value = cell_rhs[level, support]
            if weight == 0.0 and rhs_value == 0.0:
                continue
            for left in range(width):
                left_value = basis[support, left]
                xtw[level, left] += left_value * weight
                xt_rhs[level, left] += left_value * rhs_value
                weighted_left = left_value * weight
                for right in range(left, width):
                    product = weighted_left * basis[support, right]
                    gram[level, left, right] += product
                    if left != right:
                        gram[level, right, left] += product
    return cell_weights, gram, xtw, xt_rhs
```

- [ ] **Step 4: Expose transformed cell moments**

Import the kernel in `_group_matrix_core.py` and add:

```python
def factor_smooth_discrete_cell_moments(
    self,
    W: NDArray,
    rhs: NDArray,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Return cell weights and natural-basis moments for a discrete block."""
    if not self.is_discrete:
        raise ValueError("cell moments require a discrete FactorSmooth matrix")
    cell_weights, raw_gram, raw_xtw, raw_rhs = (
        _factor_smooth_support_cell_sufficient_stats(
            self.B_unique,
            self.bin_idx,
            self.codes,
            np.asarray(W, dtype=np.float64),
            np.asarray(rhs, dtype=np.float64),
            self.n_levels,
        )
    )
    local_gram = np.einsum(
        "ai,kab,bj->kij",
        self.natural_map,
        raw_gram,
        self.natural_map,
        optimize=True,
    )
    local_gram = 0.5 * (local_gram + local_gram.transpose(0, 2, 1))
    return (
        cell_weights,
        local_gram,
        raw_xtw @ self.natural_map,
        raw_rhs @ self.natural_map,
    )
```

Make the discrete branch of `factor_smooth_sufficient_stats` delegate to this
method and discard only `cell_weights`. Keep the exact CSR branch unchanged.

- [ ] **Step 5: Run focused tests and verify GREEN**

```bash
rtk uv run pytest \
  tests/test_factor_smooth_discrete.py \
  tests/test_factor_smooth_matrix.py \
  tests/test_factor_smooth_sz_matrix.py \
  -q
rtk uv run ruff check \
  src/superglm/_group_matrix/_group_matrix_kernels.py \
  src/superglm/_group_matrix/_group_matrix_core.py \
  tests/test_factor_smooth_discrete.py
```

Expected: all tests and Ruff pass.

- [ ] **Step 6: Commit**

```bash
rtk git add \
  src/superglm/_group_matrix/_group_matrix_kernels.py \
  src/superglm/_group_matrix/_group_matrix_core.py \
  tests/test_factor_smooth_discrete.py
rtk git commit -m "Aggregate discrete FactorSmooth cell moments"
```

### Task 3: Batch Dense and Matching-Spline Crosses

**Files:**

- Modify: `src/superglm/_group_matrix/_group_matrix_kernels.py`
- Modify: `src/superglm/_group_matrix/_group_matrix_core.py`
- Modify: `src/superglm/solvers/structured.py`
- Modify: `tests/test_factor_smooth_structured_system.py`
- Modify: `tests/test_structured_allocations.py`

- [ ] **Step 1: Add failing dense-cell cross parity**

Add:

```python
@pytest.mark.parametrize("small_width", [1, 4, 13])
def test_discrete_factor_smooth_dense_cell_cross_matches_reference(
    small_width: int,
) -> None:
    rng = np.random.default_rng(778)
    dominant = _dominant(discrete=True, n=511, factor_basis="sz")
    W = rng.uniform(0.2, 1.5, size=dominant.shape[0])
    small = rng.normal(size=(dominant.shape[0], small_width))

    actual = dominant.factor_smooth_discrete_dense_cell_cross_gram(W, small)
    raw_expected = np.empty(
        (dominant.n_levels, dominant.block_size, small_width),
        dtype=np.float64,
    )
    effective_basis = dominant.B_unique @ dominant.natural_map
    for level in range(dominant.n_levels):
        rows = dominant.codes == level
        raw_expected[level] = (
            effective_basis[dominant.bin_idx[rows]].T
            @ (W[rows, None] * small[rows])
        )
    np.testing.assert_allclose(actual, raw_expected, rtol=2e-12, atol=2e-12)
```

The raw `K`-level cross is intentional; SZ contrast conversion remains in the
structured operator.

- [ ] **Step 2: Run and verify RED**

```bash
rtk uv run pytest \
  tests/test_factor_smooth_structured_system.py::test_discrete_factor_smooth_dense_cell_cross_matches_reference \
  -q
```

Expected: FAIL with missing dense-cell cross method.

- [ ] **Step 3: Add the batched dense-cell kernel**

In `_group_matrix_kernels.py`, add:

```python
@njit(cache=True)
def _factor_smooth_support_dense_cell_cross(
    basis,
    bin_idx,
    codes,
    weights,
    dense_small,
    n_levels,
):
    n_bins = basis.shape[0]
    width = basis.shape[1]
    small_width = dense_small.shape[1]
    cell_cross = np.zeros((n_levels, n_bins, small_width))
    for row in range(len(codes)):
        level = codes[row]
        support = bin_idx[row]
        weight = weights[row]
        for column in range(small_width):
            cell_cross[level, support, column] += (
                weight * dense_small[row, column]
            )

    raw = np.zeros((n_levels, width, small_width))
    for level in range(n_levels):
        for support in range(n_bins):
            for left in range(width):
                basis_value = basis[support, left]
                for column in range(small_width):
                    raw[level, left, column] += (
                        basis_value * cell_cross[level, support, column]
                    )
    return raw
```

- [ ] **Step 4: Add compact cross dispatch methods**

In `FactorSmoothGroupMatrix`, add:

```python
def factor_smooth_discrete_dense_cell_cross_gram(
    self,
    W: NDArray,
    dense_small: NDArray,
) -> NDArray:
    """Return raw-level crosses after one dense-block cell aggregation."""
    if not self.is_discrete:
        raise ValueError("dense cell crosses require a discrete FactorSmooth")
    raw = _factor_smooth_support_dense_cell_cross(
        self.B_unique,
        self.bin_idx,
        self.codes,
        np.asarray(W, dtype=np.float64),
        np.asarray(dense_small, dtype=np.float64),
        self.n_levels,
    )
    return np.einsum(
        "ai,kaq->kiq",
        self.natural_map,
        raw,
        optimize=True,
    )

def factor_smooth_discrete_shared_bin_cross_gram(
    self,
    cell_weights: NDArray,
    other,
) -> NDArray | None:
    """Return a cross for a discretized SSP sharing this observation bin map."""
    from ._group_matrix_discretized import DiscretizedSSPGroupMatrix

    if (
        not self.is_discrete
        or not isinstance(other, DiscretizedSSPGroupMatrix)
        or not np.array_equal(self.bin_idx, other.bin_idx)
    ):
        return None
    other_support = other.B_unique @ other.R_inv
    raw = np.einsum(
        "gb,ba,bq->gaq",
        cell_weights,
        self.B_unique,
        other_support,
        optimize=True,
    )
    return np.einsum(
        "ai,kaq->kiq",
        self.natural_map,
        raw,
        optimize=True,
    )
```

- [ ] **Step 5: Add the failing scan-count and allocation regressions**

Add `DiscretizedSSPGroupMatrix` to the imports in
`tests/test_factor_smooth_structured_system.py`, then add:

```python
def test_discrete_sz_batches_supported_small_crosses_without_column_scans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(20260725)
    dominant = _dominant(discrete=True, n=240, factor_basis="sz")
    n = dominant.shape[0]
    dense = DenseGroupMatrix(rng.normal(size=(n, 4)))
    support = np.linspace(-1.0, 1.0, dominant.B_unique.shape[0])
    global_spline = DiscretizedSSPGroupMatrix(
        B_unique=np.column_stack([np.ones_like(support), support, support**2]),
        R_inv=np.eye(3),
        bin_idx=dominant.bin_idx.copy(),
    )
    matrices = [dense, global_spline, dominant]
    groups = [
        GroupSlice(name="dense", start=0, end=4),
        GroupSlice(name="global", start=4, end=7),
        GroupSlice(name="factor", start=7, end=7 + dominant.shape[1]),
    ]
    W = rng.uniform(0.2, 1.8, size=n)
    Wz = rng.normal(size=n)
    legacy_calls = 0
    original = FactorSmoothGroupMatrix.factor_smooth_dense_cross_gram

    def counted(self, weights, values):
        nonlocal legacy_calls
        legacy_calls += 1
        return original(self, weights, values)

    monkeypatch.setattr(
        FactorSmoothGroupMatrix,
        "factor_smooth_dense_cross_gram",
        counted,
    )
    system = build_block_structured_system(
        matrices,
        groups,
        W,
        Wz,
        dominant_group_index=2,
    )

    reference = np.column_stack([matrix.toarray() for matrix in matrices])
    expected = reference.T @ (W[:, None] * reference)
    np.testing.assert_allclose(
        materialize_compact_operator(system.operator),
        expected,
        rtol=2.0e-12,
        atol=2.0e-11,
    )
    assert legacy_calls == 0
```

Run this test before integration. Expected: FAIL because the current builder
calls `factor_smooth_dense_cross_gram` seven times.

In `tests/test_structured_allocations.py`, construct the same supported
discrete geometry, monkeypatch
`DiscretizedSSPGroupMatrix.toarray` to raise
`AssertionError("optimized discrete cross materialized observation rows")`,
call `build_block_structured_system`, and assert
`system.operator.C.shape == (dominant.n_levels, dominant.block_size, 7)`.

- [ ] **Step 6: Integrate cell moments before small crosses**

In `build_block_structured_system`, compute the dominant moments before
constructing `C`:

```python
cell_weights = None
if dominant.is_discrete:
    (
        cell_weights,
        D,
        raw_xtw_structured,
        raw_xtwz_structured,
    ) = dominant.factor_smooth_discrete_cell_moments(
        weights,
        weighted_rhs,
    )
else:
    D, raw_xtw_structured, raw_xtwz_structured = (
        dominant.factor_smooth_sufficient_stats(
            weights,
            weighted_rhs,
        )
    )
```

When iterating `layout.small_matrices`, dispatch in this order:

```python
if dominant.is_discrete and type(matrix) is DenseGroupMatrix:
    cross = dominant.factor_smooth_discrete_dense_cell_cross_gram(
        weights,
        matrix.M,
    )
elif dominant.is_discrete and cell_weights is not None:
    cross = dominant.factor_smooth_discrete_shared_bin_cross_gram(
        cell_weights,
        matrix,
    )
else:
    cross = None

if cross is not None:
    cross_blocks.append(cross)
    continue
```

Leave the existing per-column SZ and `_cross_gram` FS fallbacks immediately
after this dispatch. Remove the later duplicate call to
`factor_smooth_sufficient_stats`.

In the `layout.dense_small_matrix is not None` branch, use the same batched
cell cross for discrete FactorSmooths:

```python
if dominant.is_discrete:
    C = dominant.factor_smooth_discrete_dense_cell_cross_gram(
        weights,
        layout.dense_small_matrix,
    )
else:
    C = dominant.factor_smooth_dense_cross_gram(
        weights,
        layout.dense_small_matrix,
    )
```

- [ ] **Step 7: Verify the original RED tests become GREEN**

```bash
rtk uv run pytest \
  tests/test_factor_smooth_structured_system.py \
  tests/test_structured_allocations.py \
  -q
```

Expected: all pass, including `legacy_calls == 0` for the supported geometry.

- [ ] **Step 8: Verify full FS/SZ structured parity**

```bash
rtk uv run pytest \
  tests/test_factor_smooth_structured_parity.py \
  tests/test_factor_smooth_structured_system.py \
  tests/test_factor_smooth_reml.py \
  tests/test_factor_smooth_sz_reml.py \
  tests/test_factor_smooth_mgcv_parity.py \
  tests/test_factor_smooth_sz_mgcv_parity.py \
  -q
rtk uv run ruff check src/superglm/ tests/test_factor_smooth_* tests/test_structured_allocations.py
```

Expected: all tests and Ruff pass.

- [ ] **Step 9: Commit**

```bash
rtk git add \
  src/superglm/_group_matrix/_group_matrix_kernels.py \
  src/superglm/_group_matrix/_group_matrix_core.py \
  src/superglm/solvers/structured.py \
  tests/test_factor_smooth_structured_system.py \
  tests/test_structured_allocations.py
rtk git commit -m "Batch discrete FactorSmooth cross moments"
```

### Task 4: Serial Cell Go/No-Go Benchmark

**Files:**

- Verify: `benchmarks/profile_structured_credibility.py`
- Verify: `/tmp/superglm-sz-cell-{20k,100k,250k,1m}/summary.json`

- [ ] **Step 1: Run the serial benchmark matrix**

For each row count, run three clean repetitions and one warm-up:

```bash
rtk uv run python benchmarks/profile_structured_credibility.py \
  --n 20000 --levels 300 --family poisson --discrete \
  --random-effects 0 --small-width 4 \
  --structured-term factor_smooth --block-size 10 \
  --factor-basis sz --global-spline --weights nonuniform \
  --backend structured --repetitions 3 --warmups 1 \
  --max-reml-iter 20 --reml-tol 1e-7 \
  --no-cprofile --no-tracemalloc --no-dense-parity \
  --output-dir /tmp/superglm-sz-cell-20k

rtk uv run python benchmarks/profile_structured_credibility.py \
  --n 100000 --levels 300 --family poisson --discrete \
  --random-effects 0 --small-width 4 \
  --structured-term factor_smooth --block-size 10 \
  --factor-basis sz --global-spline --weights nonuniform \
  --backend structured --repetitions 3 --warmups 1 \
  --max-reml-iter 20 --reml-tol 1e-7 \
  --no-cprofile --no-tracemalloc --no-dense-parity \
  --output-dir /tmp/superglm-sz-cell-100k

rtk uv run python benchmarks/profile_structured_credibility.py \
  --n 250000 --levels 300 --family poisson --discrete \
  --random-effects 0 --small-width 4 \
  --structured-term factor_smooth --block-size 10 \
  --factor-basis sz --global-spline --weights nonuniform \
  --backend structured --repetitions 3 --warmups 1 \
  --max-reml-iter 20 --reml-tol 1e-7 \
  --no-cprofile --no-tracemalloc --no-dense-parity \
  --output-dir /tmp/superglm-sz-cell-250k

rtk uv run python benchmarks/profile_structured_credibility.py \
  --n 1000000 --levels 300 --family poisson --discrete \
  --random-effects 0 --small-width 4 \
  --structured-term factor_smooth --block-size 10 \
  --factor-basis sz --global-spline --weights nonuniform \
  --backend structured --repetitions 3 --warmups 1 \
  --max-reml-iter 20 --reml-tol 1e-7 \
  --no-cprofile --no-tracemalloc --no-dense-parity \
  --output-dir /tmp/superglm-sz-cell-1m
```

- [ ] **Step 2: Apply the serial keep/revert gate**

Compare against the committed baseline:

- 20,000 rows: 7.69 seconds from the original converged reference, noting its
  twelve REML iterations;
- 1,000,000 rows: 7.67 seconds median, five REML iterations.

Also compare phase timings and equal iteration counts rather than wall time
alone. Keep the serial cell commits only if the million-row same-iteration
moment stacks improve materially and the 20,000-row case regresses by at most
5%.

If the gate fails, revert Tasks 2 and 3. The baseline/profile evidence remains
in the design document and `/tmp` artifacts:

```bash
rtk git revert --no-edit HEAD
rtk git revert --no-edit HEAD~2
```

Before running those commands, confirm `HEAD` is the Task 3 commit and
`HEAD~1` is the Task 2 commit with `rtk git log -4 --oneline`. The first
revert adds a commit, making the original Task 2 commit `HEAD~2` for the
second command. If history differs, stop and resolve the exact hashes before
any revert.

### Task 5: Add Bounded Parallel Cell Reduction

Proceed only if Task 4 retains the serial cell path and its profile still shows
cell aggregation as a material stack.

**Files:**

- Modify: `src/superglm/_group_matrix/_group_matrix_kernels.py`
- Modify: `src/superglm/_group_matrix/_group_matrix_core.py`
- Modify: `tests/test_factor_smooth_discrete.py`

- [ ] **Step 1: Add failing dispatch and determinism tests**

Add tests for a private decision helper:

```python
@pytest.mark.parametrize(
    ("n", "n_levels", "n_bins", "small_width", "expected"),
    [
        (20_000, 300, 256, 4, 1),
        (1_000_000, 300, 256, 4, 8),
        (1_000_000, 3_000, 512, 20, 1),
    ],
)
def test_factor_smooth_parallel_chunk_policy(
    n: int,
    n_levels: int,
    n_bins: int,
    small_width: int,
    expected: int,
) -> None:
    assert _factor_smooth_cell_chunk_count(
        n=n,
        n_levels=n_levels,
        n_bins=n_bins,
        small_width=small_width,
        available_threads=8,
    ) == expected
```

The final case falls back to serial because its private accumulation estimate
exceeds 64 MiB.

Add a repeated-run test asserting:

```python
first = gm.factor_smooth_discrete_cell_moments(W, Wz, chunks=4)
second = gm.factor_smooth_discrete_cell_moments(W, Wz, chunks=4)
for left, right in zip(first, second, strict=True):
    np.testing.assert_array_equal(left, right)
```

Run and verify RED because the helper and `chunks` argument do not exist.

- [ ] **Step 2: Implement the pure chunk decision**

In `_group_matrix_core.py`, add `import numba` and:

```python
_FACTOR_SMOOTH_PARALLEL_MIN_ROWS = 100_000
_FACTOR_SMOOTH_PARALLEL_MAX_CHUNKS = 8
_FACTOR_SMOOTH_PARALLEL_MAX_BYTES = 64 * 1024 * 1024


def _factor_smooth_cell_chunk_count(
    *,
    n: int,
    n_levels: int,
    n_bins: int,
    small_width: int,
    available_threads: int,
) -> int:
    if n < _FACTOR_SMOOTH_PARALLEL_MIN_ROWS or available_threads < 2:
        return 1
    chunks = min(_FACTOR_SMOOTH_PARALLEL_MAX_CHUNKS, available_threads)
    bytes_required = (
        chunks
        * n_levels
        * n_bins
        * (2 + small_width)
        * np.dtype(np.float64).itemsize
    )
    return 1 if bytes_required > _FACTOR_SMOOTH_PARALLEL_MAX_BYTES else chunks
```

- [ ] **Step 3: Implement deterministic private-chunk aggregation**

Import `prange` from Numba and add:

```python
@njit(cache=True, parallel=True)
def _factor_smooth_support_cell_aggregates_parallel(
    bin_idx,
    codes,
    weights,
    rhs,
    n_levels,
    n_bins,
    n_chunks,
):
    chunk_weights = np.zeros((n_chunks, n_levels, n_bins))
    chunk_rhs = np.zeros((n_chunks, n_levels, n_bins))
    n = len(codes)
    for chunk in prange(n_chunks):
        start = chunk * n // n_chunks
        stop = (chunk + 1) * n // n_chunks
        for row in range(start, stop):
            level = codes[row]
            support = bin_idx[row]
            chunk_weights[chunk, level, support] += weights[row]
            chunk_rhs[chunk, level, support] += rhs[row]

    cell_weights = np.zeros((n_levels, n_bins))
    cell_rhs = np.zeros((n_levels, n_bins))
    for chunk in range(n_chunks):
        for level in range(n_levels):
            for support in range(n_bins):
                cell_weights[level, support] += (
                    chunk_weights[chunk, level, support]
                )
                cell_rhs[level, support] += chunk_rhs[chunk, level, support]
    return cell_weights, cell_rhs
```

Extract the cell-to-basis contraction from the serial kernel into a compiled
helper shared by serial and parallel aggregation. Do not duplicate the
`k x k` algebra.

- [ ] **Step 4: Dispatch without changing global thread settings**

Use `numba.get_num_threads()` only to query availability. Compute `chunks` in
the Python method and call the serial kernel for `chunks == 1`; otherwise call
the parallel aggregate followed by the common contraction.

Change the private method signature to:

```python
def factor_smooth_discrete_cell_moments(
    self,
    W: NDArray,
    rhs: NDArray,
    *,
    chunks: int | None = None,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
```

When `chunks is None`, call `_factor_smooth_cell_chunk_count` with
`available_threads=numba.get_num_threads()`. An explicit positive `chunks`
value is a test/profiling hook capped at eight; reject zero or negative values
with `ValueError("chunks must be positive")`.

Do not call `numba.set_num_threads`.

- [ ] **Step 5: Verify parity, determinism, and small serial dispatch**

```bash
rtk uv run pytest \
  tests/test_factor_smooth_discrete.py \
  tests/test_factor_smooth_structured_system.py \
  -q
rtk uv run ruff check \
  src/superglm/_group_matrix/_group_matrix_kernels.py \
  src/superglm/_group_matrix/_group_matrix_core.py \
  tests/test_factor_smooth_discrete.py
```

- [ ] **Step 6: Commit**

```bash
rtk git add \
  src/superglm/_group_matrix/_group_matrix_kernels.py \
  src/superglm/_group_matrix/_group_matrix_core.py \
  tests/test_factor_smooth_discrete.py
rtk git commit -m "Parallelize large FactorSmooth cell reductions"
```

### Task 6: Parallel Go/No-Go and Whole-Fit Call-Stack Analysis

**Files:**

- Modify: `docs/guide/fitting.md`

- [ ] **Step 1: Benchmark configured Numba thread counts**

Run the million-row case with `NUMBA_NUM_THREADS` set before Python starts:

```bash
rtk proxy env NUMBA_NUM_THREADS=1 uv run python \
  benchmarks/profile_structured_credibility.py \
  --n 1000000 --levels 300 --family poisson --discrete \
  --random-effects 0 --small-width 4 \
  --structured-term factor_smooth --block-size 10 \
  --factor-basis sz --global-spline --weights nonuniform \
  --backend structured --repetitions 3 --warmups 1 \
  --max-reml-iter 20 --reml-tol 1e-7 \
  --no-cprofile --no-tracemalloc --no-dense-parity \
  --output-dir /tmp/superglm-sz-cell-threads-1

rtk proxy env NUMBA_NUM_THREADS=2 uv run python \
  benchmarks/profile_structured_credibility.py \
  --n 1000000 --levels 300 --family poisson --discrete \
  --random-effects 0 --small-width 4 \
  --structured-term factor_smooth --block-size 10 \
  --factor-basis sz --global-spline --weights nonuniform \
  --backend structured --repetitions 3 --warmups 1 \
  --max-reml-iter 20 --reml-tol 1e-7 \
  --no-cprofile --no-tracemalloc --no-dense-parity \
  --output-dir /tmp/superglm-sz-cell-threads-2

rtk proxy env NUMBA_NUM_THREADS=4 uv run python \
  benchmarks/profile_structured_credibility.py \
  --n 1000000 --levels 300 --family poisson --discrete \
  --random-effects 0 --small-width 4 \
  --structured-term factor_smooth --block-size 10 \
  --factor-basis sz --global-spline --weights nonuniform \
  --backend structured --repetitions 3 --warmups 1 \
  --max-reml-iter 20 --reml-tol 1e-7 \
  --no-cprofile --no-tracemalloc --no-dense-parity \
  --output-dir /tmp/superglm-sz-cell-threads-4

rtk proxy env NUMBA_NUM_THREADS=8 uv run python \
  benchmarks/profile_structured_credibility.py \
  --n 1000000 --levels 300 --family poisson --discrete \
  --random-effects 0 --small-width 4 \
  --structured-term factor_smooth --block-size 10 \
  --factor-basis sz --global-spline --weights nonuniform \
  --backend structured --repetitions 3 --warmups 1 \
  --max-reml-iter 20 --reml-tol 1e-7 \
  --no-cprofile --no-tracemalloc --no-dense-parity \
  --output-dir /tmp/superglm-sz-cell-threads-8
```

`rtk proxy env` applies the thread setting only to that benchmark process; it
does not mutate global settings.

- [ ] **Step 2: Apply the parallel keep/revert gate**

Keep the parallel commit only if at least one 2/4/8-thread configuration
improves repeatably over the serial cell path and the 20,000-row dispatch
remains serial with at most 5% regression.

If it fails:

```bash
rtk git revert --no-edit HEAD
```

Confirm with `rtk git log -2 --oneline` that `HEAD` is exactly
`Parallelize large FactorSmooth cell reductions` before reverting.

- [ ] **Step 3: Run the final million-row whole-fit cProfile**

Use the retained fastest configuration, or omit the environment assignment if
parallelism was reverted:

```bash
rtk proxy env NUMBA_NUM_THREADS=8 uv run python \
  benchmarks/profile_structured_credibility.py \
  --n 1000000 --levels 300 --family poisson --discrete \
  --random-effects 0 --small-width 4 \
  --structured-term factor_smooth --block-size 10 \
  --factor-basis sz --global-spline --weights nonuniform \
  --backend structured --repetitions 1 --warmups 0 \
  --max-reml-iter 20 --reml-tol 1e-7 \
  --cprofile --no-tracemalloc --no-dense-parity \
  --output-dir /tmp/superglm-sz-cell-final-profile
```

Verify that:

- `factor_smooth_dense_cross_gram` no longer has 182 calls;
- the total structured-system stack is lower than 3.91 seconds;
- clean and profiled convergence, objective, lambdas, EDF, and prediction
  checksum agree within existing tolerances.

- [ ] **Step 4: Measure final allocation separately**

```bash
rtk uv run python benchmarks/profile_structured_credibility.py \
  --n 1000000 --levels 300 --family poisson --discrete \
  --random-effects 0 --small-width 4 \
  --structured-term factor_smooth --block-size 10 \
  --factor-basis sz --global-spline --weights nonuniform \
  --backend structured --repetitions 1 --warmups 0 \
  --max-reml-iter 3 --reml-tol 1e-7 \
  --no-cprofile --tracemalloc --no-dense-parity \
  --output-dir /tmp/superglm-sz-cell-final-allocation
```

- [ ] **Step 5: Document only retained evidence**

Update the measured SZ section in `docs/guide/fitting.md` with:

- baseline and final clean medians;
- row count, groups, `k`, convergence iterations, and thread configuration;
- serial/parallel crossover;
- peak allocation;
- final cumulative call stack;
- an explicit statement that unsupported small groups retain compact fallback.

Do not publish timings from failed or contended runs as performance claims.

- [ ] **Step 6: Commit**

```bash
rtk git add docs/guide/fitting.md
rtk git commit -m "Document large-n FactorSmooth moment performance"
```

### Task 7: Final Regression and Review Gates

**Files:**

- Verify all modified source, tests, benchmark, and documentation files

- [ ] **Step 1: Run focused FactorSmooth and structured suites**

```bash
rtk uv run pytest \
  tests/test_factor_smooth_discrete.py \
  tests/test_factor_smooth_feature.py \
  tests/test_factor_smooth_inference.py \
  tests/test_factor_smooth_matrix.py \
  tests/test_factor_smooth_mgcv_parity.py \
  tests/test_factor_smooth_penalties.py \
  tests/test_factor_smooth_reml.py \
  tests/test_factor_smooth_structured_parity.py \
  tests/test_factor_smooth_structured_system.py \
  tests/test_factor_smooth_sz_feature.py \
  tests/test_factor_smooth_sz_inference.py \
  tests/test_factor_smooth_sz_matrix.py \
  tests/test_factor_smooth_sz_mgcv_parity.py \
  tests/test_factor_smooth_sz_penalties.py \
  tests/test_factor_smooth_sz_reml.py \
  tests/test_structured_allocations.py \
  tests/test_structured_credibility_benchmark.py \
  -q
```

- [ ] **Step 2: Run legacy RE/FS performance sentinels**

Use the same fixed seed, thread configuration, repetitions, and geometry as
the committed pre-feature baseline. Require no material RE regression and no
more than 5% FS median regression.

- [ ] **Step 3: Run repository gates**

```bash
rtk uv run pytest tests/ -q
rtk uv run ruff check src/ tests/ benchmarks/
rtk uv run ruff format --check src/ tests/ benchmarks/
rtk uv run mkdocs build --strict
rtk uv run python run_test.py
rtk git diff --check origin/master...HEAD
```

If the known origin/master formatting issue in
`benchmarks/benchmark_multi_scop_discrete_convergence.py` remains the only
format failure, verify it independently against origin/master and do not edit
that unrelated file.

- [ ] **Step 4: Audit prohibited scope**

```bash
rtk proxy git diff --name-only origin/master...HEAD
rtk grep -n -i "lss" \
  src/superglm/_group_matrix \
  src/superglm/solvers/structured.py \
  tests/test_factor_smooth_discrete.py \
  tests/test_factor_smooth_structured_system.py \
  docs/guide/fitting.md
rtk grep -n -i "cython\\|rust\\|c++\\|cpp extension\\|native c" \
  src tests benchmarks docs/guide/fitting.md
```

Confirm no LSS path changed and no prohibited native extension was added.

- [ ] **Step 5: Request independent code review**

Review the complete range from `59ed4ff` through `HEAD` for:

- correctness of cell sufficient statistics;
- FS/SZ raw-versus-public geometry;
- deterministic parallel reduction;
- compact allocation guarantees;
- thread oversubscription and fallback behavior;
- benchmark validity and keep/revert decisions;
- no LSS changes.

Fix every Critical or Important finding with a new failing regression test,
then repeat affected gates.

- [ ] **Step 6: Commit any review fixes and leave a clean branch**

```bash
rtk git status --short
rtk git log -8 --oneline
```

Expected: clean worktree with benchmark-proven optimization commits only.
