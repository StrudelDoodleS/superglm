# Factor-Smooth Feasibility Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make structured FS/SZ backend eligibility use the same weights, lambda-scaled penalties, symmetry, and cost policy as the eventual factorization.

**Architecture:** Keep eligibility in `solvers/_structured/selection.py` and canonical override extraction in `solvers/_structured/overrides.py`. Compute the existing automatic cost decision once, use it before any necessary factor-smooth row scan, and make both selection and assembly consume the same symmetrized local blocks.

**Tech Stack:** Python 3.10+, NumPy, SciPy sparse matrices, pytest, Ruff, existing compact `FactorSmoothGroupMatrix` kernels.

---

## File Map

- `tests/test_factor_smooth_structured_parity.py`: focused dispatch, cache, numerical-rank,
  and Gram-parity regressions.
- `src/superglm/solvers/_structured/selection.py`: automatic cost decision, exact weighted
  feasibility moments, lambda-scaled local penalties, and numeric cache identity.
- `src/superglm/solvers/_structured/overrides.py`: symmetric canonical local-block extraction.
- `docs/superpowers/specs/2026-07-27-factor-smooth-feasibility-review-fixes-design.md`:
  approved behavior contract; no further edits expected.

### Task 1: Add failing dispatch and numerical-feasibility regressions

**Files:**
- Modify: `tests/test_factor_smooth_structured_parity.py`

- [ ] **Step 1: Import the public compatibility-facade resolver**

Add `resolve_structured_backend` to the existing import from
`superglm.solvers.structured`.

- [ ] **Step 2: Add a compact direct-selection fixture**

Add this helper beside `_factor_smooth_override()`:

```python
def _selection_factor_smooth_matrix(
    *,
    factor_basis: str,
    n_levels: int,
    local_basis: np.ndarray,
    repeated_penalty_components: tuple[tuple[str, np.ndarray], ...],
) -> tuple[FactorSmoothGroupMatrix, GroupSlice]:
    rows_per_level = local_basis.shape[0] // n_levels
    codes = np.repeat(np.arange(n_levels, dtype=np.intp), rows_per_level)
    block_size = local_basis.shape[1]
    matrix = FactorSmoothGroupMatrix(
        sp.csr_matrix(local_basis),
        codes,
        n_levels,
        natural_map=np.eye(block_size),
        levels=tuple(f"level-{level}" for level in range(n_levels)),
        repeated_penalty_components=repeated_penalty_components,
        factor_basis=factor_basis,
    )
    return matrix, GroupSlice(
        name=f"x:group:{factor_basis}",
        start=0,
        end=matrix.shape[1],
        penalized=True,
    )
```

- [ ] **Step 3: Prove an automatic below-crossover SZ fit skips row scanning**

Add a test that constructs a four-level, two-column SZ matrix, a zero override, and a
zero-weight final level. Patch the class method because `FactorSmoothGroupMatrix` uses
slots:

```python
def test_auto_sz_cost_fallback_precedes_override_feasibility_scan(monkeypatch) -> None:
    n_levels = 4
    x = np.tile(np.linspace(-1.0, 1.0, 6), n_levels)
    matrix, group = _selection_factor_smooth_matrix(
        factor_basis="sz",
        n_levels=n_levels,
        local_basis=np.column_stack((np.ones_like(x), x)),
        repeated_penalty_components=(("wiggle", np.eye(2)),),
    )
    weights = np.ones(matrix.shape[0])
    weights[matrix.codes == n_levels - 1] = 0.0
    override = np.zeros((matrix.shape[1], matrix.shape[1]))
    original = FactorSmoothGroupMatrix.factor_smooth_sufficient_stats
    calls = 0

    def counted(self, W, rhs):
        nonlocal calls
        calls += 1
        return original(self, W, rhs)

    monkeypatch.setattr(
        FactorSmoothGroupMatrix,
        "factor_smooth_sufficient_stats",
        counted,
    )
    automatic = resolve_structured_backend(
        [matrix],
        [group],
        direct_solve="auto",
        coefficient_width=matrix.shape[1],
        row_weights=weights,
        lambda2={f"{group.name}:wiggle": 1.0},
        S_override=override,
    )
    assert not automatic.use_structured
    assert "crossover" in automatic.fallback_reason
    assert calls == 0

    with pytest.raises(ValueError, match="authoritative S_override"):
        resolve_structured_backend(
            [matrix],
            [group],
            direct_solve="structured",
            coefficient_width=matrix.shape[1],
            row_weights=weights,
            lambda2={f"{group.name}:wiggle": 1.0},
            S_override=override,
        )
    assert calls == 1
```

- [ ] **Step 4: Prove tiny nonzero weights retain their numerical meaning**

Use `_factor_smooth_problem(..., factor_basis="sz")`, set all weights for the final
level to `1e-20`, and use an authoritative zero local penalty. Fit under `auto` and
explicit Gram. Require automatic fallback with an authoritative-override reason,
coefficient parity with Gram, and early rejection under forced structured mode.

```python
def test_authoritative_sz_override_preserves_tiny_weight_rank() -> None:
    dm, groups, penalties, y, weights, offset, lambdas = _factor_smooth_problem(
        _gaussian_response,
        factor_basis="sz",
    )
    matrix = dm.group_matrices[3]
    assert isinstance(matrix, FactorSmoothGroupMatrix)
    weights = np.array(weights, copy=True)
    weights[matrix.codes == matrix.n_levels - 1] = 1.0e-20
    override = _factor_smooth_override(
        dm,
        groups,
        penalties,
        lambdas,
        local_penalty=None,
    )

    automatic, _ = irls_direct.fit_irls_direct(
        X=dm,
        y=y,
        weights=weights,
        family=Gaussian(),
        link=IdentityLink(),
        groups=groups,
        lambda2=lambdas,
        offset=offset,
        direct_solve="auto",
        S_override=override,
        tol=1.0e-10,
    )
    gram, _ = irls_direct.fit_irls_direct(
        X=dm,
        y=y,
        weights=weights,
        family=Gaussian(),
        link=IdentityLink(),
        groups=groups,
        lambda2=lambdas,
        offset=offset,
        direct_solve="gram",
        S_override=override,
        tol=1.0e-10,
    )
    assert automatic.direct_backend == "gram"
    assert "authoritative S_override" in automatic.direct_fallback_reason
    np.testing.assert_allclose(automatic.beta, gram.beta, atol=3.0e-8)

    with pytest.raises(ValueError, match="authoritative S_override"):
        irls_direct.fit_irls_direct(
            X=dm,
            y=y,
            weights=weights,
            family=Gaussian(),
            link=IdentityLink(),
            groups=groups,
            lambda2=lambdas,
            offset=offset,
            direct_solve="structured",
            S_override=override,
            tol=1.0e-10,
        )
```

- [ ] **Step 5: Prove lambda magnitudes invalidate feasibility cache entries**

Construct a 20-level FS matrix whose local basis supplies only the second coordinate
and whose sole penalty is `diag([1, 0])`. Resolve `lambda=1.0` and then `lambda=1e-20`
on the same matrix:

```python
def test_factor_smooth_feasibility_cache_includes_lambda_scales() -> None:
    n_levels = 20
    local_basis = np.tile(
        np.array([[0.0, 1.0], [0.0, 2.0]]),
        (n_levels, 1),
    )
    matrix, group = _selection_factor_smooth_matrix(
        factor_basis="fs",
        n_levels=n_levels,
        local_basis=local_basis,
        repeated_penalty_components=(("wiggle", np.diag([1.0, 0.0])),),
    )
    weights = np.ones(matrix.shape[0])
    moderate = resolve_structured_backend(
        [matrix],
        [group],
        direct_solve="auto",
        coefficient_width=matrix.shape[1],
        row_weights=weights,
        lambda2={f"{group.name}:wiggle": 1.0},
    )
    tiny = resolve_structured_backend(
        [matrix],
        [group],
        direct_solve="auto",
        coefficient_width=matrix.shape[1],
        row_weights=weights,
        lambda2={f"{group.name}:wiggle": 1.0e-20},
    )
    assert moderate.use_structured
    assert not tiny.use_structured
    assert "singular local block" in tiny.fallback_reason
```

- [ ] **Step 6: Run the three tests and verify RED**

Run:

```bash
rtk test .venv/bin/python -m pytest \
  tests/test_factor_smooth_structured_parity.py::test_auto_sz_cost_fallback_precedes_override_feasibility_scan \
  tests/test_factor_smooth_structured_parity.py::test_authoritative_sz_override_preserves_tiny_weight_rank \
  tests/test_factor_smooth_structured_parity.py::test_factor_smooth_feasibility_cache_includes_lambda_scales -q
```

Expected failures:

- automatic cost fallback calls the sentinel scan;
- tiny positive weights select structured execution or raise later;
- the tiny-lambda decision incorrectly reuses the moderate-lambda eligibility result.

### Task 2: Align backend selection with factorization

**Files:**
- Modify: `src/superglm/solvers/_structured/selection.py`
- Test: `tests/test_factor_smooth_structured_parity.py`

- [ ] **Step 1: Resolve scaled local penalties once**

Replace component-name-only accumulation with a helper returning the symmetric local
penalty and exact ordered numeric identity:

```python
def _factor_smooth_local_penalty(
    matrix: FactorSmoothGroupMatrix,
    group_name: str,
    lambda2: float | dict[str, float],
) -> tuple[NDArray, tuple[tuple[str, float], ...]]:
    local_penalty = np.zeros((matrix.block_size, matrix.block_size), dtype=np.float64)
    resolved = []
    for suffix, omega in matrix.repeated_penalty_components:
        lam = _factor_smooth_component_lambda(group_name, suffix, lambda2)
        resolved.append((suffix, lam))
        values = np.asarray(omega, dtype=np.float64)
        local_penalty += lam * (0.5 * (values + values.T))
    return local_penalty, tuple(resolved)
```

The small `_factor_smooth_component_lambda()` helper must preserve the existing mapping
fallback: component name first, then group name, then zero.

- [ ] **Step 2: Hash exact weights and penalty scales**

Change `_factor_smooth_singular_local_level()` to accept `local_penalty` and
`penalty_identity`. Build the cache key from the identity and a BLAKE2 digest of exact
contiguous float64 weights:

```python
weight_bytes = np.ascontiguousarray(weights).view(np.uint8)
weight_digest = hashlib.blake2b(weight_bytes, digest_size=16).digest()
cache_key = (penalty_identity, weight_digest)
```

Pass `weights`, not `(weights > 0).astype(float)`, to
`factor_smooth_sufficient_stats()`.

- [ ] **Step 3: Make override feasibility use exact weights**

In `_factor_smooth_override_singular_local_level()`, pass the validated float64
`weights` directly to `factor_smooth_sufficient_stats()`.

- [ ] **Step 4: Match the factorization rank policy**

In `_first_singular_factor_smooth_block()`, symmetrize all blocks before
`np.linalg.eigvalsh()`:

```python
symmetric = 0.5 * (local_blocks + local_blocks.transpose(0, 2, 1))
eigenvalues = np.linalg.eigvalsh(symmetric)
```

Retain the existing scale floor and `eps * block_size * scale * 10` cutoff.

- [ ] **Step 5: Compute and reuse the automatic cost decision**

Extract the existing final crossover calculation into
`_structured_auto_cost_decision()`, returning the complete
`StructuredBackendDecision`. Compute it once for automatic mode after dominant geometry
is known.

```python
def _structured_auto_cost_decision(
    dominant_matrix: GroupMatrix,
    selection: StructuredGroupSelection,
    coefficient_width: int,
    small_size: int,
) -> StructuredBackendDecision:
    if isinstance(dominant_matrix, FactorSmoothGroupMatrix):
        if dominant_matrix.factor_basis == "sz":
            use_structured, cost_ratio = _sum_to_zero_structured_auto_is_beneficial(
                dominant_matrix.n_levels,
                dominant_matrix.block_size,
                small_size,
            )
        else:
            use_structured, cost_ratio = _block_structured_auto_is_beneficial(
                dominant_matrix.n_levels,
                dominant_matrix.block_size,
                small_size,
            )
        dimensions = (
            f"K={dominant_matrix.n_levels}, "
            f"k={dominant_matrix.block_size}, q={small_size}"
        )
        geometry_name = "FactorSmooth"
    else:
        use_structured, cost_ratio = _structured_auto_is_beneficial(
            dominant_matrix.shape[1],
            small_size,
        )
        dimensions = f"K={dominant_matrix.shape[1]}, q={small_size}"
        geometry_name = "RandomEffect"
    fallback_reason = None
    if not use_structured:
        fallback_reason = (
            f"{geometry_name} geometry is below the measured structured crossover "
            f"(p={coefficient_width}, {dimensions}, "
            f"estimated_cost_ratio={cost_ratio:.3f}; "
            f"require p >= {_AUTO_MIN_COEFFICIENT_WIDTH} and ratio <= "
            f"{_AUTO_MAX_STRUCTURED_COST_RATIO:.2f})"
        )
    return StructuredBackendDecision(
        use_structured=use_structured,
        group_index=selection.group_index,
        group_name=selection.group_name,
        fallback_reason=fallback_reason,
    )
```

When a factor-smooth local penalty is singular and a row scan would be required, return
that decision immediately if it selects Gram. Forced mode still scans. At the end of
automatic resolution, return the same precomputed decision.

- [ ] **Step 6: Run focused tests and verify GREEN**

Run:

```bash
rtk test .venv/bin/python -m pytest \
  tests/test_factor_smooth_structured_parity.py::test_auto_sz_cost_fallback_precedes_override_feasibility_scan \
  tests/test_factor_smooth_structured_parity.py::test_authoritative_sz_override_preserves_tiny_weight_rank \
  tests/test_factor_smooth_structured_parity.py::test_factor_smooth_feasibility_cache_includes_lambda_scales -q
```

Expected: all three pass.

- [ ] **Step 7: Commit the dispatch/rank fix**

```bash
rtk git add src/superglm/solvers/_structured/selection.py \
  tests/test_factor_smooth_structured_parity.py
rtk git commit -m "Align factor-smooth feasibility with factorization"
```

### Task 3: Canonicalize tolerated override asymmetry

**Files:**
- Modify: `tests/test_factor_smooth_structured_parity.py`
- Modify: `src/superglm/solvers/_structured/overrides.py`

- [ ] **Step 1: Add the failing FS/SZ Gram-parity regression**

Parameterize `factor_basis` over `fs` and `sz`. Build an authoritative local penalty
from `2 * I`, set `[0, 1] = 0.2` and `[1, 0] = 0.2 + 1e-10`, then fit automatic and
Gram systems through `_factor_smooth_problem()`:

```python
@pytest.mark.parametrize("factor_basis", ["fs", "sz"])
def test_authoritative_factor_smooth_override_roundoff_asymmetry_matches_gram(
    factor_basis: str,
) -> None:
    dm, groups, penalties, y, weights, offset, lambdas = _factor_smooth_problem(
        _gaussian_response,
        factor_basis=factor_basis,
    )
    matrix = dm.group_matrices[3]
    assert isinstance(matrix, FactorSmoothGroupMatrix)
    local_penalty = 2.0 * np.eye(matrix.block_size)
    local_penalty[0, 1] = 0.2
    local_penalty[1, 0] = 0.2 + 1.0e-10
    override = _factor_smooth_override(
        dm,
        groups,
        penalties,
        lambdas,
        local_penalty=local_penalty,
    )
    automatic, _ = irls_direct.fit_irls_direct(
        X=dm,
        y=y,
        weights=weights,
        family=Gaussian(),
        link=IdentityLink(),
        groups=groups,
        lambda2=lambdas,
        offset=offset,
        direct_solve="auto",
        S_override=override,
        tol=1.0e-10,
    )
    gram, _ = irls_direct.fit_irls_direct(
        X=dm,
        y=y,
        weights=weights,
        family=Gaussian(),
        link=IdentityLink(),
        groups=groups,
        lambda2=lambdas,
        offset=offset,
        direct_solve="gram",
        S_override=override,
        tol=1.0e-10,
    )
    assert automatic.direct_backend == "structured"
    assert automatic.direct_fallback_reason is None
    np.testing.assert_allclose(automatic.beta, gram.beta, atol=3.0e-8)
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
rtk test .venv/bin/python -m pytest \
  tests/test_factor_smooth_structured_parity.py::test_authoritative_factor_smooth_override_roundoff_asymmetry_matches_gram -q
```

Expected: both parameter cases fail at the structured operator's strict symmetry check.

- [ ] **Step 3: Symmetrize shared extracted local blocks**

In `_factor_smooth_override_local_blocks()`, first construct `local_blocks` for FS or
SZ, then return:

```python
return 0.5 * (local_blocks + local_blocks.transpose(0, 2, 1))
```

Selection and assembly already call this helper, so no second production edit is needed.

- [ ] **Step 4: Run the test and verify GREEN**

Run the exact Task 3 Step 2 command. Expected: two parameter cases pass.

- [ ] **Step 5: Commit canonical extraction**

```bash
rtk git add src/superglm/solvers/_structured/overrides.py \
  tests/test_factor_smooth_structured_parity.py
rtk git commit -m "Canonicalize factor-smooth override blocks"
```

### Task 4: Validate and close the review round

**Files:**
- Verify: `src/superglm/solvers/_structured/selection.py`
- Verify: `src/superglm/solvers/_structured/overrides.py`
- Verify: `src/superglm/solvers/_structured/assembly.py`
- Verify: `tests/test_factor_smooth_structured_parity.py`

- [ ] **Step 1: Run the complete factor-smooth parity file**

```bash
rtk test .venv/bin/python -m pytest tests/test_factor_smooth_structured_parity.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Run the broader structured numerical gate**

```bash
rtk test .venv/bin/python -m pytest \
  tests/test_structured_irls.py \
  tests/test_block_schur_factor.py \
  tests/test_sum_to_zero_structured_factor.py \
  tests/test_factor_smooth_structured_parity.py -q
```

Expected: all tests pass.

- [ ] **Step 3: Run static and diff checks**

```bash
rtk ruff check src/superglm/solvers/_structured/selection.py \
  src/superglm/solvers/_structured/overrides.py \
  tests/test_factor_smooth_structured_parity.py
rtk git diff --check
rtk git status --short
```

Expected: Ruff and diff checks succeed; only intentional commits remain.

- [ ] **Step 4: Push through the repository pre-push gate**

```bash
rtk git push origin feature/structured-credibility
```

Expected: the configured non-slow suite passes and the remote branch reaches the exact
local head.

- [ ] **Step 5: Reply in each inline thread and resolve it**

Use the numeric inline comment IDs for replies and the four thread IDs for GraphQL
resolution:

- `PRRT_kwDORfJEl86UK6sD`
- `PRRT_kwDORfJEl86UK6sG`
- `PRRT_kwDORfJEl86UK6sL`
- `PRRT_kwDORfJEl86UK6sN`

Each reply must cite the exact fixing commit and focused test evidence.

- [ ] **Step 6: Verify zero unresolved threads**

```bash
rtk proxy python \
  /home/mhick/.codex/plugins/cache/openai-curated-remote/github/0.1.8-2841cf9749ae/skills/gh-address-comments/scripts/fetch_comments.py \
  | rtk proxy jq '[.review_threads[] | select(.isResolved == false)] | length'
```

Expected: `0`.

- [ ] **Step 7: Request and monitor a fresh exact-head Codex review and CI**

Post a new `@codex review` comment naming the full exact SHA. Continue the established
review loop until Codex reports no actionable findings and all exact-head checks are
green. Leave PR #165 draft and do not merge or publish it.
