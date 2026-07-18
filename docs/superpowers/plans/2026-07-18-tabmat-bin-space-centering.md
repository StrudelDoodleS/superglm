# Tabmat Bin-Space Centering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the mixed discretized-centering execution paths with one cached, certified,
public-Tabmat bin-space plan that removes first-use Numba overhead and spline row materialization.

**Architecture:** A private immutable plan stores a `SplitMatrix` containing ordinary predictor
blocks and categorical bin-code blocks. It transforms augmented raw moments to solver coordinates
blockwise, while `DesignMatrix` owns the pickle-reset cache and the centered-system dispatcher
retains the existing safety certificate and stable fallback.

**Tech Stack:** Python 3.10+, NumPy, Tabmat 4.2.1 public API, pytest, Ruff.

---

### Task 1: Specify dispatch, parity, and no-JIT behavior

**Files:**
- Modify: `tests/test_rank_policy.py`

- [ ] Add a real mixed-design test that patches `_weighted_bincount_2d`,
  `_fused_bincount_2`, `_disc_disc_2d_hist`, and `_cat_weighted_bincount` to raise, patches group
  `toarray`/`row_subset` to raise, and asserts centered Gram/RHS/means against a dense reference.
- [ ] Assert one public `SplitMatrix.sandwich` and two public `transpose_matvec` calls per accepted
  system build.
- [ ] Parameterize low/high-cardinality categorical, fragmented numeric, one/four compressed
  spline, aliased numeric, and zero-weight cases.
- [ ] Add exact-type rejection tests for tensor and unsupported support-space groups, asserting the
  fit-local tri-state remains `None`.
- [ ] Run `rtk pytest tests/test_rank_policy.py -k 'bin_space or mixed_discrete'` and verify failure
  because the current route invokes Numba or lacks the plan/cache API.

### Task 2: Specify cache lifecycle

**Files:**
- Modify: `tests/test_rank_policy.py`

- [ ] Add a test that accesses `dm.mixed_bin_space_centering_plan` twice and asserts object identity.
- [ ] Round-trip the `DesignMatrix` through `pickle.dumps`/`pickle.loads` and assert the restored
  private cache is `None`, followed by successful lazy reconstruction.
- [ ] Assert a native low-cardinality categorical component is a `tabmat.CategoricalMatrix`, and
  patch `CategoricalGroupMatrix.toarray` to prove no dense one-hot duplicate is created.
- [ ] Run the cache tests and verify the expected missing-property failure.

### Task 3: Implement the immutable bin-space plan

**Files:**
- Create: `src/superglm/_group_matrix/_group_matrix_bin_space.py`
- Modify: `src/superglm/_group_matrix/_group_matrix_tabmat.py`

- [ ] Add a shared native-categorical constructor that remaps SuperGLM's base sentinel to Tabmat's
  dropped first category using `CategoricalMatrix(..., categories=..., drop_first=True)`.
- [ ] Implement exact topology validation: exact `DiscretizedSSPGroupMatrix`, `DenseGroupMatrix`,
  and at most one `CategoricalGroupMatrix`; reject subclasses and every other group type.
- [ ] Build one logical-order `SplitMatrix` using DenseMatrix, native CategoricalMatrix, and bin-code
  CategoricalMatrix public constructors.
- [ ] Store read-only ordinary augmented/global indices and compressed block descriptors.
- [ ] Implement `preflight(W, sum_w)` using virtual standardization plus bounded support summaries.
- [ ] Implement `moments(W, weighted_z)` with one sandwich, two transpose products, and blockwise
  transforms; return `WeightedMoments` in solver order.
- [ ] Run the Task 1 tests and verify the plan unit behavior passes before dispatcher integration.

### Task 4: Cache and dispatch through one route

**Files:**
- Modify: `src/superglm/group_matrix.py`
- Modify: `src/superglm/_group_matrix/_group_matrix_centered.py`
- Modify: `src/superglm/solvers/centered_system.py`
- Modify: `tests/test_rank_policy.py`

- [ ] Add `_mixed_bin_space_centering_plan = None` initialization and pickle reset to `DesignMatrix`.
- [ ] Add a lazy `mixed_bin_space_centering_plan` property that validates the resulting plan shape.
- [ ] Replace current specialized/partial mixed moment selection with the bin-space plan while
  retaining the scaled `100_000 * compressed_group_count` floor.
- [ ] Keep unsupported layouts unattempted; keep preflight/certificate rejection attempted and
  fit-locally locked to stable chunks.
- [ ] Remove obsolete `mixed_centering_execution_plan` cache/property and tests so there is one
  accepted mixed route.
- [ ] Run all new rank-policy tests and verify green.

### Task 5: Adversarial and regression verification

**Files:**
- Modify: `tests/test_rank_policy.py` only if a newly reproduced bug needs a regression test first.

- [ ] Run `rtk pytest tests/test_rank_policy.py tests/test_matrix_execution_plan.py tests/test_irls_direct.py -q`.
- [ ] Run targeted alias, zero-weight, unsafe-offset, tensor-rejection, low-cardinality, and pickle
  tests independently and inspect route counters.
- [ ] Run `rtk ruff check src/superglm/_group_matrix src/superglm/group_matrix.py src/superglm/solvers/centered_system.py tests/test_rank_policy.py`.
- [ ] Run `rtk git diff --check`.

### Task 6: Benchmark and final review

**Files:**
- Do not modify production files during measurement.

- [ ] Run fresh-process, fixed-affinity, one-thread candidate/stable/frozen comparisons for 10k and
  60k canonical mixed layouts, fragmented dense groups, one/four splines, and low/high-cardinality
  categoricals.
- [ ] Record cold and warm wall/process time, RSS delta, Python traced peak, parity, convergence,
  iteration count, Tabmat kernel counts, and Numba kernel counts.
- [ ] Reject or narrow any topology with a repeatable full-fit regression above 3%.
- [ ] Inspect the complete diff for accidental API changes, private Tabmat dependencies, expanded
  spline rows, dense categorical duplicates, or unrelated user changes.
- [ ] Report the exact evidence and remaining caveats to the parent agent.
