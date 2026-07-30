# Discrete REML Terminal Coherence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ensure discrete REML can only publish a smoothing-parameter state
whose objective and convergence were evaluated together, then retain the
faster algebraically equivalent batched FactorSmooth cell contraction.

**Architecture:** The cached-W POI optimizer will test the current candidate
before constructing or applying another Newton step. A converged candidate
therefore flows unchanged into the authoritative full PIRLS refit. Discrete
FactorSmooths will keep one compiled row pass for compact `(level, bin)`
aggregates, then use batched NumPy/BLAS contractions in natural coordinates.

**Tech Stack:** Python 3.10+, NumPy, Numba, Tabmat, pytest, cProfile, Ruff

**Design:**
`docs/superpowers/specs/2026-07-25-discrete-reml-terminal-coherence-design.md`

**Worktree:**
`/home/mhick/python_projects/superglm/.worktrees/structured-credibility`

**Execution:** Inline in the primary session, matching the user's explicit
preference against subagent-led development.

---

## File Map

- Modify `tests/test_factor_smooth_sz_the reference implementation_parity.py`
  - reproduce the terminal lambda jump with algebraically equivalent moments
  - assert terminal/evaluated state coherence
- Modify `src/superglm/reml/discrete.py`
  - move convergence to the evaluated-candidate boundary
  - avoid terminal Hessian and line-search work
- Modify `tests/test_factor_smooth_discrete.py`
  - require compact batched contraction without the raw-Gram `einsum` route
  - retain FS/SZ signed-weight row-oracle coverage
- Modify `src/superglm/_group_matrix/_group_matrix_kernels.py`
  - reduce the cell kernel to the single observation pass
- Modify `src/superglm/_group_matrix/_group_matrix_core.py`
  - contract compact cells with batched matrix multiplication
- Verify `benchmarks/profile_structured_credibility.py`
  - compare convergence work and million-row time to fit

No LSS file and no C, C++, Cython, or Rust source is in scope.

### Task 1: Pin the Unevaluated Terminal-Step Regression

**Files:**

- Modify: `tests/test_factor_smooth_sz_the reference implementation_parity.py`
- Test: `tests/test_factor_smooth_sz_the reference implementation_parity.py`

- [ ] **Step 1: Import the compact FactorSmooth matrix**

Add this import beside the existing SuperGLM imports:

```python
from superglm.group_matrix import FactorSmoothGroupMatrix
```

- [ ] **Step 2: Write the failing integration regression**

Append this test after
`test_sz_superglm_exact_and_discrete_predictions_match`:

```python
def test_discrete_sz_terminal_lambdas_are_the_evaluated_candidate(
    the reference implementation_sz_fixture: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = the reference implementation_sz_fixture["poisson_discrete"]
    data = case["data"]
    X = pd.DataFrame({"x": data["x"], "f": data["f"]})
    y = np.asarray(data["y"], dtype=np.float64)
    offset = np.log(np.asarray(data["exposure"], dtype=np.float64))
    original = FactorSmoothGroupMatrix.factor_smooth_discrete_cell_moments

    def equivalent_batched_moments(self, W, rhs):
        cell_weights, _gram, _xtw, _xt_rhs = original(self, W, rhs)
        cell_rhs = np.zeros_like(cell_weights)
        np.add.at(
            cell_rhs,
            (self.codes, self.bin_idx),
            np.asarray(rhs, dtype=np.float64),
        )
        effective_basis = np.ascontiguousarray(
            self.B_unique @ self.natural_map,
            dtype=np.float64,
        )
        weighted_basis = cell_weights[:, :, None] * effective_basis[None, :, :]
        local_gram = (
            effective_basis.T[None, :, :] @ weighted_basis
        )
        local_gram = 0.5 * (
            local_gram + local_gram.transpose(0, 2, 1)
        )
        return (
            np.ascontiguousarray(cell_weights),
            np.ascontiguousarray(local_gram),
            np.ascontiguousarray(cell_weights @ effective_basis),
            np.ascontiguousarray(cell_rhs @ effective_basis),
        )

    monkeypatch.setattr(
        FactorSmoothGroupMatrix,
        "factor_smooth_discrete_cell_moments",
        equivalent_batched_moments,
    )
    model = SuperGLM(
        family="poisson",
        features={"x": Spline(kind="ps", k=7, m=2)},
        interactions=[
            FactorSmooth(
                "x",
                group="f",
                basis="sz",
                kind="ps",
                k=6,
                m=2,
            )
        ],
        selection_penalty=0.0,
        direct_solve="structured",
        discrete=True,
        n_bins=512,
        tol=1e-10,
        max_iter=200,
    )
    model.fit_reml(
        X,
        y,
        offset=offset,
        max_reml_iter=50,
        reml_tol=1e-9,
        pirls_tol=1e-10,
        max_pirls_iter=200,
        runtime_validation="skip",
    )

    result = model._reml_result
    assert result.converged
    assert result.lambda_history[-1] == result.lambda_history[-2]
    assert model._reml_lambdas["x:f:sz:wiggle"] == pytest.approx(
        case["reference"]["unscaled_lambdas"]["sz_wiggle"],
        rel=5.0e-2,
    )
```

This uses the real optimizer and fixture. The monkeypatch changes only
floating-point summation order; it does not stub the optimizer.

- [ ] **Step 3: Run the test and verify RED**

```bash
rtk pytest \
  tests/test_factor_smooth_sz_the reference implementation_parity.py::test_discrete_sz_terminal_lambdas_are_the_evaluated_candidate \
  -q
```

Expected: FAIL because the last evaluated wiggle lambda is approximately
`6.424373509`, while the terminal refit uses approximately `2.363394936`.

- [ ] **Step 4: Commit the failing regression**

```bash
rtk git add tests/test_factor_smooth_sz_the reference implementation_parity.py
rtk git commit -m "Test discrete REML terminal lambda coherence"
```

### Task 2: Stop Before Applying an Unevaluated Newton Step

**Files:**

- Modify: `src/superglm/reml/discrete.py`
- Test: `tests/test_factor_smooth_sz_the reference implementation_parity.py`
- Test: `tests/test_reml_newton_fixes.py`

- [ ] **Step 1: Separate current-candidate gradient work from Hessian work**

In `optimize_discrete_reml_cached_w`, retain the current
`reml_direct_gradient(...)` call, but move the
`reml_direct_hessian(...)` call below the convergence guard introduced in the
next step.

- [ ] **Step 2: Check convergence at the evaluated candidate**

Immediately after projecting `grad` into `proj_grad_d`, add:

```python
        proj_grad_norm = float(np.max(np.abs(proj_grad_d)))
        obj_change = abs(obj - prev_obj) if poi_iter > 0 else np.inf

        if verbose:
            lam_str = ", ".join(
                f"{name}={cand_lambdas[name]:.4g}"
                for name in group_names
            )
            print(
                f"  POI iter {poi_iter + 1}  obj={obj:.4f}  "
                f"|grad|={proj_grad_norm:.6f}  "
                f"delta_obj={obj_change:.6g}  [{lam_str}]"
            )

        if poi_iter >= 1:
            grad_converged_d = proj_grad_norm < _tol * score_scale_d
            obj_converged_d = obj_change < _tol * score_scale_d
            if grad_converged_d and obj_converged_d:
                rho = rho_clipped
                prev_obj = obj
                converged = True
                _t_newton += _time.perf_counter() - _t0
                break
        prev_obj = obj
```

The assignment `rho = rho_clipped` makes the final full PIRLS refit derive
exactly the already evaluated `cand_lambdas`.

- [ ] **Step 3: Build the Hessian only for a nonterminal candidate**

After the new guard, retain:

```python
        hess = reml_direct_hessian(
            dm.group_matrices,
            distribution,
            XtWX_S_inv,
            cand_lambdas,
            gradient=grad,
            penalty_caches=penalty_caches,
            pirls_result=pirls_result,
            n_obs=len(y),
            phi_hat=phi_hat,
            inverse_phi=inverse_phi,
            d_inverse_phi_d_penalized_deviance=inverse_phi_derivative,
            penalty_nullity=penalty_nullity if not scale_known else None,
            reml_penalties=penalties,
            tensor_pair_evaluations=cand_tensor_pair_evals,
        )
```

Keep active-set construction, modified Newton, step capping, and line search
unchanged.

- [ ] **Step 4: Remove the stale post-line-search convergence block**

Delete the later duplicate calculation beginning with:

```python
        # Convergence check -- compound criterion with score_scale
```

Retain tensor diagnostic recording, but reuse the `proj_grad_norm` computed at
the candidate boundary. Delete the later verbose block and the later
`prev_obj` assignment as well.

- [ ] **Step 5: Run the regression and verify GREEN**

```bash
rtk pytest \
  tests/test_factor_smooth_sz_the reference implementation_parity.py::test_discrete_sz_terminal_lambdas_are_the_evaluated_candidate \
  -q
```

Expected: PASS with the terminal wiggle lambda near `6.42437`.

- [ ] **Step 6: Run discrete optimizer regressions**

```bash
rtk pytest \
  tests/test_reml_newton_fixes.py \
  tests/test_cached_w_validation.py \
  tests/test_factor_smooth_sz_the reference implementation_parity.py \
  -q
```

Expected: all tests pass. The terminal iteration may perform one fewer Hessian
and line-search calculation, but accepted nonterminal histories remain
unchanged.

- [ ] **Step 7: Commit the optimizer fix**

```bash
rtk git add src/superglm/reml/discrete.py
rtk git commit -m "Fix discrete REML terminal lambda coherence"
```

### Task 3: Replace Compact Gram Loops with Batched Cell Contractions

**Files:**

- Modify: `tests/test_factor_smooth_discrete.py`
- Modify: `src/superglm/_group_matrix/_group_matrix_kernels.py`
- Modify: `src/superglm/_group_matrix/_group_matrix_core.py`

- [ ] **Step 1: Write a failing routing test**

Add this test after
`test_discrete_cell_moments_match_explicit_row_level_reference`:

```python
def test_discrete_cell_moments_do_not_use_raw_gram_einsum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gm = _cell_test_matrix(factor_basis="fs")
    weights = np.linspace(-0.4, 1.7, gm.shape[0])
    rhs = np.linspace(-1.2, 0.9, gm.shape[0])

    def forbidden_einsum(*_args, **_kwargs):
        raise AssertionError("discrete cell moments used raw-Gram einsum")

    monkeypatch.setattr(np, "einsum", forbidden_einsum)
    cell_weights, local_gram, xtw_nat, rhs_nat = (
        gm.factor_smooth_discrete_cell_moments(weights, rhs)
    )

    assert cell_weights.shape == (gm.n_levels, gm.B_unique.shape[0])
    assert local_gram.shape == (
        gm.n_levels,
        gm.block_size,
        gm.block_size,
    )
    assert xtw_nat.shape == rhs_nat.shape == (
        gm.n_levels,
        gm.block_size,
    )
```

- [ ] **Step 2: Run the routing test and verify RED**

```bash
rtk pytest \
  tests/test_factor_smooth_discrete.py::test_discrete_cell_moments_do_not_use_raw_gram_einsum \
  -q
```

Expected: FAIL with
`AssertionError: discrete cell moments used raw-Gram einsum`.

- [ ] **Step 3: Reduce the compiled kernel to cell aggregation**

In `_group_matrix_kernels.py`, replace
`_factor_smooth_support_cell_sufficient_stats` with:

```python
@njit(cache=True)
def _factor_smooth_support_cell_aggregates(
    bin_idx,
    codes,
    weights,
    rhs,
    n_levels,
    n_bins,
):
    """Aggregate changing FactorSmooth values by level/support cell."""
    cell_weights = np.zeros((n_levels, n_bins))
    cell_rhs = np.zeros((n_levels, n_bins))
    for row in range(len(codes)):
        level = codes[row]
        support = bin_idx[row]
        cell_weights[level, support] += weights[row]
        cell_rhs[level, support] += rhs[row]
    return cell_weights, cell_rhs
```

This preserves the only row-dependent work and removes all
`level/support/left/right` Gram loops.

- [ ] **Step 4: Import the aggregate-only kernel**

In `_group_matrix_core.py`, replace the old kernel import with:

```python
    _factor_smooth_support_cell_aggregates,
```

- [ ] **Step 5: Contract cells directly in natural coordinates**

Replace the body of the discrete moment contraction after validation with:

```python
        basis = self.B_unique
        support_index = self.bin_idx
        if basis is None or support_index is None:
            raise RuntimeError(
                "discrete FactorSmooth support is unavailable"
            )

        cell_weights, cell_rhs = _factor_smooth_support_cell_aggregates(
            support_index,
            self.codes,
            weights,
            rhs_values,
            self.n_levels,
            basis.shape[0],
        )
        effective_basis = np.ascontiguousarray(
            basis @ self.natural_map,
            dtype=np.float64,
        )
        weighted_basis = (
            cell_weights[:, :, None] * effective_basis[None, :, :]
        )
        local_gram = (
            effective_basis.T[None, :, :] @ weighted_basis
        )
        local_gram = 0.5 * (
            local_gram + local_gram.transpose(0, 2, 1)
        )
        return (
            np.ascontiguousarray(cell_weights),
            np.ascontiguousarray(local_gram),
            np.ascontiguousarray(cell_weights @ effective_basis),
            np.ascontiguousarray(cell_rhs @ effective_basis),
        )
```

- [ ] **Step 6: Run cell algebra and allocation tests**

```bash
rtk pytest \
  tests/test_factor_smooth_discrete.py \
  tests/test_factor_smooth_structured_system.py \
  tests/test_structured_allocations.py \
  -q
```

Expected: all tests pass for FS, SZ, rectangular natural maps, signed weights,
empty levels, and no observation-sized FactorSmooth materialization.

- [ ] **Step 7: Run the SZ terminal and reference parity tests**

```bash
rtk pytest tests/test_factor_smooth_sz_the reference implementation_parity.py -q
```

Expected: all tests pass. The discrete terminal lambda remains near `6.42437`
and exact/discrete prediction parity remains within the pinned tolerance.

- [ ] **Step 8: Commit the batched contraction**

```bash
rtk git add \
  src/superglm/_group_matrix/_group_matrix_kernels.py \
  src/superglm/_group_matrix/_group_matrix_core.py \
  tests/test_factor_smooth_discrete.py
rtk git commit -m "Batch discrete FactorSmooth Gram moments"
```

### Task 4: Profile Convergence Work and Million-Row Time to Fit

**Files:**

- Verify: `benchmarks/profile_structured_credibility.py`
- Produce: `/tmp/superglm-sz-1m-terminal-batched/summary.json`
- Produce: `/tmp/superglm-sz-1m-terminal-cprofile/summary.json`

- [ ] **Step 1: Run the five-repeat million-row benchmark**

```bash
rtk proxy uv run python \
  benchmarks/profile_structured_credibility.py \
  --n 1000000 \
  --levels 300 \
  --family poisson \
  --discrete \
  --random-effects 0 \
  --small-width 4 \
  --structured-term factor_smooth \
  --block-size 10 \
  --factor-basis sz \
  --global-spline \
  --weights nonuniform \
  --backend structured \
  --repetitions 5 \
  --warmups 1 \
  --max-reml-iter 20 \
  --reml-tol 1e-7 \
  --no-cprofile \
  --no-tracemalloc \
  --output-dir /tmp/superglm-sz-1m-terminal-batched
```

Expected:

- convergence is true;
- REML iterations remain `5`;
- prediction checksum remains within `1e-8` relative of
  `1045155.2072170012`;
- median time is no more than 5% above the retained `4.4742056 s` median.

- [ ] **Step 2: Inspect the compact benchmark result**

```bash
rtk json /tmp/superglm-sz-1m-terminal-batched/summary.json
```

Record the five wall times, median, REML iteration count, line-search solve
count, lambda values, deviance, and checksum in the final handoff.

- [ ] **Step 3: Run one cProfile fit**

```bash
rtk proxy uv run python \
  benchmarks/profile_structured_credibility.py \
  --n 1000000 \
  --levels 300 \
  --family poisson \
  --discrete \
  --random-effects 0 \
  --small-width 4 \
  --structured-term factor_smooth \
  --block-size 10 \
  --factor-basis sz \
  --global-spline \
  --weights nonuniform \
  --backend structured \
  --repetitions 1 \
  --warmups 1 \
  --max-reml-iter 20 \
  --reml-tol 1e-7 \
  --cprofile \
  --no-tracemalloc \
  --output-dir /tmp/superglm-sz-1m-terminal-cprofile
```

Expected: the removed
`_factor_smooth_support_cell_sufficient_stats` Gram contraction is absent from
the call stack, and the final converged outer iteration performs no
lambda line search.

- [ ] **Step 4: Inspect the profile and call stack**

```bash
rtk grep -n \
  "factor_smooth\\|build_block_structured_system\\|line_search\\|solve_cached_structured" \
  /tmp/superglm-sz-1m-terminal-cprofile/cprofile_structured_top.txt
```

Expected: no row-dependent quadruple Gram loop and one fewer terminal
Hessian/line-search cycle than the pre-fix control flow.

### Task 5: Complete Regression Verification

**Files:**

- Verify: `src/superglm/`
- Verify: `tests/`

- [ ] **Step 1: Run the focused solver suite**

```bash
rtk pytest \
  tests/test_factor_smooth_discrete.py \
  tests/test_factor_smooth_structured_system.py \
  tests/test_factor_smooth_sz_the reference implementation_parity.py \
  tests/test_reml_newton_fixes.py \
  tests/test_cached_w_validation.py \
  tests/test_random_effect_reml.py \
  -q
```

Expected: all focused tests pass.

- [ ] **Step 2: Run Ruff**

```bash
rtk ruff check src/ tests/
```

Expected: no lint errors.

- [ ] **Step 3: Run the complete test suite**

```bash
rtk pytest tests/ -q
```

Expected: all tests pass, with only established skips.

- [ ] **Step 4: Check patch integrity**

```bash
rtk git diff --check
rtk git status --short
```

Expected: no whitespace errors and no unrelated files.

- [ ] **Step 5: Commit any test-only formatting adjustment**

Only if Ruff or the test suite required a formatting-only adjustment:

```bash
rtk git add src/superglm/reml/discrete.py tests/
rtk git commit -m "Polish discrete REML convergence tests"
```

If no adjustment was needed, do not create an empty commit.

## Follow-on Milestone

After this plan is verified, continue the active lambda-convergence goal with
a separate evidence-gathering design covering:

- exact versus discrete convergence traces across Gaussian, Poisson, Gamma,
  Tweedie, tensor, RE, FS, and SZ models;
- false convergence, boundary lambdas, flat-curvature steps, repeated
  step-halving, and max-iteration exhaustion;
- objective/PIRLS/gradient/Hessian/line-search timing attribution;
- reference parity where a clean-room fixture exists; and
- only those additional safeguards that improve the representative corpus
  without increasing time to fit materially.

That milestone is deliberately separate: terminal-state coherence is a
confirmed correctness defect, while broader convergence-policy changes need
their own measured design.
