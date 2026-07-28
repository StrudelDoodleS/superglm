# Unified FactorSmooth FS/SZ Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement, document, profile, and verify a unified
`FactorSmooth(basis="fs"|"sz", kind=...)` API with clean-room mgcv-compatible
SZ deviations, compact exact/discrete execution, and no LSS changes.

**Architecture:** Keep the existing FS natural parameterization and compiled
raw grouped-spline kernels unchanged. Add a shared factor-contrast geometry
module, expose `(K - 1)k` SZ public coordinates through the compact group
matrix, and solve SZ through a symmetric equality-constrained block factor
whose dense border is only `q + k + R`. Extend the compact penalty/operator
protocols so dense oracles, REML derivatives, retained covariance, and
reporting all consume the same public geometry.

**Tech Stack:** Python 3.10+, NumPy, SciPy, pandas, Numba, tabmat, pytest,
R 4.5.3 with mgcv 1.9-4, cProfile, tracemalloc, MkDocs, Ruff, mypy.

---

## File Map

### New files

- `src/superglm/factor_smooth_geometry.py` — pure contrast, expansion,
  adjoint, and penalty helpers shared by feature, matrix, penalty, and solver
  code.
- `src/superglm/solvers/sum_to_zero.py` — rank-aware symmetric constrained
  factor and profiled adapter implementing the full `HessianFactor` protocol.
- `tests/test_factor_smooth_sz_feature.py` — constructor, compatibility,
  marginal, prediction, reconstruction, and warning contracts.
- `tests/test_factor_smooth_sz_matrix.py` — exact/discrete compact matrix
  algebra and allocation guards.
- `tests/test_factor_smooth_sz_penalties.py` — contrast penalty, rank,
  pseudo-determinant, quadratic, and matvec tests.
- `tests/test_sum_to_zero_structured_factor.py` — randomized dense-oracle
  factor, covariance, determinant, trace, singular-local, and failure tests.
- `tests/test_factor_smooth_sz_reml.py` — dense/structured, exact/discrete,
  Gaussian/Poisson, finite-difference, and retained-state integration.
- `tests/test_factor_smooth_sz_inference.py` — SZ-specific reporting and curve
  covariance tests.
- `tests/test_factor_smooth_sz_mgcv_parity.py` — pinned mgcv fit and
  construction parity.
- `tests/fixtures/factor_smooth_sz_mgcv_reference.R` — clean-room fixture
  generator that executes mgcv without embedding its source.
- `tests/fixtures/factor_smooth_sz_mgcv_reference.json` — generated,
  version-pinned reference values.

### Modified files

- `src/superglm/features/factor_smooth.py` — unified public API and FS/SZ
  marginal/prediction behavior.
- `src/superglm/types.py` — compact factor basis metadata and SZ penalty kind.
- `src/superglm/dm_builder.py` — pass factor geometry into the group matrix.
- `src/superglm/_group_matrix/_group_matrix_core.py` — contrast-aware compact
  matrix façade while retaining existing raw kernels.
- `src/superglm/solvers/hessian_factor.py` — dense materialization for the SZ
  compact penalty.
- `src/superglm/solvers/structured.py` — SZ compact operator/system,
  structured dispatch, centering, and factor construction hooks.
- `src/superglm/solvers/irls_direct.py` — accept the SZ system/factor in final
  PIRLS and retained REML geometry.
- `src/superglm/reml/penalty_algebra.py` — SZ component spectrum and compact
  penalty operations.
- `src/superglm/reml/observed_geometry.py` — observed-Hessian SZ dispatch and
  inertia certificate.
- `src/superglm/model/api.py` — `splines=` deprecation warning and reporting
  type annotations.
- `src/superglm/model/base.py` — resolved factor-smooth compatibility checks.
- `src/superglm/model/reml_finalize.py` — retained SZ factor/operator/support
  state.
- `src/superglm/inference/covariance.py` — accept the profiled SZ covariance
  factor.
- `src/superglm/inference/factor_smooths.py` — basis-aware FS versus SZ
  reporting.
- `benchmarks/profile_structured_credibility.py` — FS/SZ benchmark dimension
  and model selection.
- `tests/test_structured_credibility_benchmark.py` — benchmark harness smoke
  coverage.
- `tests/test_api.py`, `tests/test_cross_validate.py`,
  `tests/test_factor_smooth_matrix.py`, `tests/test_factor_smooth_discrete.py`,
  `tests/test_factor_smooth_penalties.py`,
  `tests/test_factor_smooth_inference.py`,
  `tests/test_factor_smooth_structured_system.py`,
  `tests/test_factor_smooth_structured_parity.py`, and
  `tests/test_structured_allocations.py` — existing regression suites extended
  with basis-aware contracts.
- `src/superglm/__init__.py`, `docs/getting-started/quickstart.md`,
  `docs/guide/interactions.md`, `docs/guide/credibility.md`,
  `docs/guide/fitting.md`, `docs/api/features.md`, and
  `docs/api/inference.md` — canonical explicit-feature API, FS/SZ guidance,
  prediction semantics, and measured profile evidence.

No file whose path contains `lss` is in this plan.

---

### Task 0: Reconfirm the remote-master baseline

**Files:**

- Inspect only: branch ancestry, worktree status, and existing FS/RE tests

- [ ] **Step 1: Refresh and compare the base**

```bash
rtk git fetch origin master
rtk git rev-list --left-right --count HEAD...origin/master
rtk git status
```

Expected: the right-hand count is zero and the worktree is clean before
implementation. If `origin/master` has moved, merge it into this isolated
feature branch, resolve without editing any LSS path, and repeat these checks;
do not build on a stale base.

- [ ] **Step 2: Record the exact base and run baseline regressions**

```bash
rtk git rev-parse origin/master
rtk uv run pytest tests/test_random_effect_reml.py tests/test_factor_smooth_feature.py tests/test_factor_smooth_reml.py tests/test_factor_smooth_structured_parity.py -q
```

Expected: the exact base SHA is recorded in the eventual review handoff and
the existing RE/FS baseline passes before SZ changes.

- [ ] **Step 3: Capture same-hardware RE/FS performance sentinels**

```bash
rtk uv run python benchmarks/profile_structured_credibility.py --structured-term random_effect --family poisson --n 2000 --levels 50 --backend structured --repetitions 2 --warmups 1 --max-reml-iter 3 --output-dir /tmp/superglm-re-pre-sz
rtk uv run python benchmarks/profile_structured_credibility.py --structured-term factor_smooth --global-spline --family poisson --n 2000 --levels 20 --block-size 5 --backend structured --repetitions 2 --warmups 1 --max-reml-iter 3 --output-dir /tmp/superglm-fs-pre-sz
rtk json /tmp/superglm-re-pre-sz/summary.json
rtk json /tmp/superglm-fs-pre-sz/summary.json
```

Retain the summaries for the Task 11 same-command comparison. If temporary
state is lost, rerun these two commands from the Task 0 base commit in a
separate read-only comparison worktree before making a regression claim.

---

### Task 1: Public API, shorthand warning, and model compatibility

**Files:**

- Modify: `src/superglm/features/factor_smooth.py`
- Modify: `src/superglm/model/api.py`
- Modify: `src/superglm/model/base.py`
- Create: `tests/test_factor_smooth_sz_feature.py`
- Modify: `tests/test_api.py`
- Modify: `tests/test_cross_validate.py`

- [ ] **Step 1: Write failing constructor and compatibility tests**

Add these focused contracts:

```python
from superglm import Categorical, FactorSmooth, RandomEffect, Spline, SuperGLM


def test_factor_smooth_basis_defaults_and_names() -> None:
    fs = FactorSmooth("age", group="region")
    sz = FactorSmooth("age", group="region", basis="sz")
    assert (fs.basis, fs.name) == ("fs", "age:region:fs")
    assert (sz.basis, sz.name) == ("sz", "age:region:sz")
    assert fs.kind == sz.kind == "ps"


def test_sz_requires_explicit_global_spline() -> None:
    with pytest.raises(ValueError, match=r"basis='sz'.*features=.*Spline"):
        SuperGLM(interactions=[FactorSmooth("age", group="region", basis="sz")])

    model = SuperGLM(
        features={"age": Spline(kind="ps", k=7, m=2)},
        interactions=[FactorSmooth("age", group="region", basis="sz")],
    )
    assert model._interaction_order == ["age:region:sz"]


@pytest.mark.parametrize("group_spec", [Categorical(), RandomEffect()])
@pytest.mark.parametrize("basis", ["fs", "sz"])
def test_factor_smooth_rejects_duplicate_group_geometry(group_spec, basis) -> None:
    features = {"age": Spline(), "region": group_spec}
    with pytest.raises(ValueError, match=r"region.*duplicates"):
        SuperGLM(
            features=features,
            interactions=[FactorSmooth("age", group="region", basis=basis)],
        )


def test_factor_smooth_rejects_duplicate_pair_despite_custom_names() -> None:
    with pytest.raises(ValueError, match=r"\('age', 'region'\).*more than once"):
        SuperGLM(
            features={"age": Spline()},
            interactions=[
                FactorSmooth("age", group="region", name="first"),
                FactorSmooth("age", group="region", basis="sz", name="second"),
            ],
        )


def test_fs_remains_valid_without_main_effect() -> None:
    model = SuperGLM(interactions=[FactorSmooth("age", group="region")])
    assert model._interaction_order == ["age:region:fs"]


def test_fs_may_coexist_with_global_spline() -> None:
    model = SuperGLM(
        features={"age": Spline()},
        interactions=[FactorSmooth("age", group="region")],
    )
    assert model._interaction_order == ["age:region:fs"]
```

Also test invalid `basis`, unsupported `kind`, `variable == group`, and that
an SZ lambda-policy mapping rejects FS-only keys such as `null_0`. Deep-copy,
model configuration capture, and `clone_unfitted()` must preserve `basis`,
`kind`, `k`, `m`, `unseen`, and the wiggle policy without sharing mutable
state.

Add deprecation tests to `tests/test_api.py`:

```python
def test_splines_shorthand_emits_one_future_warning() -> None:
    with pytest.warns(FutureWarning, match=r"splines=.*features=.*Spline") as caught:
        SuperGLM(splines=[])
    assert len(caught) == 1


def test_explicit_features_do_not_emit_spline_shorthand_warning() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        SuperGLM(features={"age": Spline()})
```

Parameterize the warning test over `[]` and `["age"]`, and add a silent
`splines=None` case. In `tests/test_cross_validate.py`, capture warnings around
an auto-detect model's cross-validation, base/subclass `clone_unfitted()`, and
pickle round trip. Assert the caller's original construction emits exactly
one matching warning and no internal reconstruction or deserialization emits
another.

- [ ] **Step 2: Run the tests and verify the new contracts fail**

Run:

```bash
rtk uv run pytest tests/test_factor_smooth_sz_feature.py tests/test_api.py -q
```

Expected: failures for the missing `basis` attribute/argument, missing SZ
compatibility validation, duplicate categorical geometry, pair deduplication,
and missing `FutureWarning`.

- [ ] **Step 3: Add `basis` and basis-specific lambda validation**

Change the constructor signature and state:

```python
def __init__(
    self,
    variable: str,
    *,
    group: str,
    basis: Literal["fs", "sz"] = "fs",
    kind: str = "ps",
    k: int = 6,
    m: int = 2,
    unseen: Literal["population", "error"] = "population",
    missing: Literal["error"] = "error",
    lambda_policy: LambdaPolicy | dict[str, LambdaPolicy] | None = None,
    name: str | None = None,
):
    if basis not in ("fs", "sz"):
        raise ValueError(f"basis must be 'fs' or 'sz', got {basis!r}")
    if kind != "ps":
        raise NotImplementedError("FactorSmooth currently supports only kind='ps'.")
    valid_components = (
        {"wiggle", *(f"null_{index}" for index in range(m))}
        if basis == "fs"
        else {"wiggle"}
    )
    self.basis = basis
    self.kind = kind
    self.name = name or f"{variable}:{group}:{basis}"
```

Keep every existing FS default, component name, and error intact.
Update the class docstring to distinguish fully penalized FS from centered SZ
without describing either one as a generic tensor-product interaction.

- [ ] **Step 4: Implement one resolved-configuration validator**

Add to `model/base.py`:

```python
def validate_factor_smooth_configuration(model, *, features_resolved: bool) -> None:
    from superglm.features.categorical import Categorical
    from superglm.features.factor_smooth import FactorSmooth
    from superglm.features.random_effect import RandomEffect
    from superglm.features.spline import _SplineBase

    terms = [
        spec
        for name in model._interaction_order
        if isinstance((spec := model._interaction_specs[name]), FactorSmooth)
    ]
    seen: dict[tuple[str, str], str] = {}
    for term in terms:
        pair = (term.variable, term.group)
        if pair in seen:
            raise ValueError(
                f"FactorSmooth pair {pair!r} is configured more than once "
                f"({seen[pair]!r} and {term.name!r})."
            )
        seen[pair] = term.name

    if not features_resolved:
        return

    for term in terms:
        group_spec = model._specs.get(term.group)
        if isinstance(group_spec, Categorical | RandomEffect):
            raise ValueError(
                f"FactorSmooth {term.name!r} group {term.group!r} duplicates "
                f"the group-intercept geometry of {type(group_spec).__name__} "
                "on the same column; remove that group main effect and use an "
                "explicit features map containing only the intended main effects."
            )
        if term.basis == "sz" and not isinstance(
            model._specs.get(term.variable), _SplineBase
        ):
            raise ValueError(
                f"FactorSmooth {term.name!r} with basis='sz' requires a global "
                f"Spline for {term.variable!r}; use "
                f"features={{{term.variable!r}: Spline(...)}}."
            )
```

Call it at the end of `init_model` with
`features_resolved=model._splines is None`, and at the end of
`model.base.auto_detect()` with `features_resolved=True`. The latter covers
both fitting through `_auto_detect_specs_if_needed()` and the public
`model.auto_detect()` entry point. Remove the older RandomEffect-only special
case so one function owns the rules.

- [ ] **Step 5: Emit the public deprecation warning without internal repeats**

In `SuperGLM.__init__`, before `base.init_model(...)`:

```python
if splines is not None:
    import warnings

    warnings.warn(
        "`splines=` auto-detection is deprecated and will be removed in a "
        "future release; use explicit features such as "
        "`features={'age': Spline(...)}`.",
        FutureWarning,
        stacklevel=2,
    )
```

The base-class clone uses `ModelConfig.materialize()` and therefore does not
re-enter the constructor. Extend the subclass clone warning filter to suppress
only this exact library-generated `FutureWarning` during internal cloning.
Mark the `splines` parameter as deprecated in the constructor docstring,
point to explicit `features={name: Spline(...)}`, and retain the companion
legacy controls without announcing a removal version.

- [ ] **Step 6: Run focused tests and commit**

Run:

```bash
rtk uv run pytest tests/test_factor_smooth_sz_feature.py tests/test_factor_smooth_feature.py tests/test_api.py tests/test_cross_validate.py -q
rtk uv run ruff check src/superglm/features/factor_smooth.py src/superglm/model/api.py src/superglm/model/base.py tests/test_factor_smooth_sz_feature.py tests/test_api.py tests/test_cross_validate.py
rtk git add src/superglm/features/factor_smooth.py src/superglm/model/api.py src/superglm/model/base.py tests/test_factor_smooth_sz_feature.py tests/test_api.py tests/test_cross_validate.py
rtk git commit -m "Add unified FactorSmooth basis API"
```

Expected: all focused tests pass; the commit contains no LSS path.

---

### Task 2: Shared SZ contrast geometry and feature behavior

**Files:**

- Create: `src/superglm/factor_smooth_geometry.py`
- Modify: `src/superglm/features/factor_smooth.py`
- Modify: `src/superglm/types.py`
- Modify: `tests/test_factor_smooth_sz_feature.py`

- [ ] **Step 1: Write failing geometry, build, prediction, and reconstruction tests**

```python
def test_sz_build_has_k_minus_one_blocks_and_one_wiggle_component() -> None:
    x = np.tile(np.linspace(-1.0, 1.0, 12), 3)
    group = np.repeat(["a", "b", "c"], 12)
    spec = FactorSmooth("x", group="g", basis="sz", k=6, m=2)
    info = spec.build(x, group, {})
    assert info.n_cols == 12
    assert info.factor_smooth_factor_basis == "sz"
    assert [name for name, _ in info.repeated_penalty_components] == ["wiggle"]
    assert info.repeated_penalty_components[0][1].shape == (6, 6)


def test_sz_transform_and_score_sum_to_zero_over_levels() -> None:
    x = np.tile(np.linspace(-0.8, 0.8, 9), 3)
    group = np.repeat(["a", "b", "c"], 9)
    spec = FactorSmooth("x", group="g", basis="sz", k=6)
    spec.build(x, group, {})
    beta = np.linspace(-0.4, 0.7, 12)
    grid = np.linspace(-0.7, 0.7, 11)
    frames = [
        spec.score(grid, np.repeat(level, len(grid)), beta)
        for level in spec._levels
    ]
    np.testing.assert_allclose(np.sum(frames, axis=0), 0.0, atol=1e-13)
    assert spec.transform(np.array([0.2]), np.array(["c"])).shape == (1, 12)


def test_sz_reconstruct_returns_all_levels_with_exact_zero_sum() -> None:
    x = np.tile(np.linspace(-1.0, 1.0, 12), 3)
    group = np.repeat(["a", "b", "c"], 12)
    spec = FactorSmooth("x", group="g", basis="sz", k=6, m=2)
    spec.build(x, group, {})
    beta = np.linspace(-0.4, 0.7, 12)
    reconstructed = spec.reconstruct(beta)
    blocks = np.stack(list(reconstructed["coefficients"].values()))
    assert reconstructed["basis"] == "sz"
    np.testing.assert_allclose(blocks.sum(axis=0), 0.0, atol=0.0)


def test_sz_requires_two_fitted_levels() -> None:
    with pytest.raises(ValueError, match="at least two"):
        FactorSmooth("x", group="g", basis="sz").build(
            np.linspace(0.0, 1.0, 20),
            np.repeat("only", 20),
            {},
        )


@pytest.mark.parametrize("basis", ["fs", "sz"])
def test_factor_smooth_reports_clear_marginal_rank_error(basis) -> None:
    with pytest.raises(ValueError, match=r"smaller k.*non-smooth"):
        FactorSmooth("x", group="g", basis=basis, k=6).build(
            np.resize([0.0, 0.5, 1.0], 30),
            np.repeat(["a", "b"], 15),
            {},
        )
```

- [ ] **Step 2: Run the feature tests and verify failure**

Run:

```bash
rtk uv run pytest tests/test_factor_smooth_sz_feature.py -q
```

Expected: failures for missing contrast helpers, incorrect width, FS natural
components on SZ, and missing reconstructed final level.

- [ ] **Step 3: Implement the clean shared geometry helpers**

Create `factor_smooth_geometry.py`:

```python
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def sum_to_zero_contrast(n_levels: int) -> NDArray[np.float64]:
    if n_levels < 2:
        raise ValueError("sum-to-zero factor geometry requires at least two levels")
    return np.vstack(
        (
            np.eye(n_levels - 1, dtype=np.float64),
            -np.ones((1, n_levels - 1), dtype=np.float64),
        )
    )


def expand_sum_to_zero_blocks(free: NDArray) -> NDArray[np.float64]:
    values = np.asarray(free, dtype=np.float64)
    if values.ndim not in (2, 3):
        raise ValueError("free blocks must have shape (K-1, k[, m])")
    return np.concatenate((values, -np.sum(values, axis=0, keepdims=True)), axis=0)


def adjoint_sum_to_zero_blocks(raw: NDArray) -> NDArray[np.float64]:
    values = np.asarray(raw, dtype=np.float64)
    if values.ndim not in (2, 3) or values.shape[0] < 2:
        raise ValueError("raw blocks must have shape (K, k[, m])")
    return values[:-1] - values[-1:]


def sum_to_zero_penalty(local: NDArray, n_levels: int) -> NDArray[np.float64]:
    contrast = sum_to_zero_contrast(n_levels)
    return np.kron(contrast.T @ contrast, np.asarray(local, dtype=np.float64))
```

- [ ] **Step 4: Split FS and SZ marginal construction without changing FS**

Retain the current `_natural_parameterization(...)` call only under
`self.basis == "fs"`. For SZ:

```python
if self.basis == "fs":
    natural_map, components = _natural_parameterization(
        raw_dense,
        penalty,
        rank=self.k - self.m,
    )
else:
    if np.linalg.matrix_rank(raw_dense) < self.k:
        raise ValueError(
            "FactorSmooth marginal basis is rank deficient; use more distinct "
            "numeric values, a smaller k, or a suitable non-smooth feature."
        )
    natural_map = np.eye(self.k, dtype=np.float64)
    components = (("wiggle", np.asarray(penalty, dtype=np.float64)),)
self._natural_map = natural_map
self._base_penalty_components = components
```

Set `GroupInfo.n_cols` from:

```python
coefficient_levels = n_levels if self.basis == "fs" else n_levels - 1
n_cols = coefficient_levels * self.k
```

Store `factor_smooth_factor_basis=self.basis`. Require two fitted levels for
SZ after sorted factorization.

- [ ] **Step 5: Implement basis-aware scoring and reconstruction**

Add a private coefficient-block validator:

```python
def _level_blocks(self, beta: NDArray) -> NDArray[np.float64]:
    coefficient_levels = len(self._levels) if self.basis == "fs" else len(self._levels) - 1
    expected = coefficient_levels * self.k
    values = np.asarray(beta, dtype=np.float64)
    if values.shape != (expected,):
        raise ValueError(f"beta must have shape ({expected},).")
    free = values.reshape(coefficient_levels, self.k)
    return free if self.basis == "fs" else expand_sum_to_zero_blocks(free)
```

Use all reconstructed blocks in `score()`. In `transform()`, retain current FS
columns; for SZ, write the first `K - 1` levels into their own blocks and write
the final level as `-basis` into every free block. Return zero rows for unseen
population levels. Include `"basis": self.basis` in `reconstruct()`.

- [ ] **Step 6: Extend `GroupInfo` metadata and invariant**

Add:

```python
factor_smooth_factor_basis: Literal["fs", "sz"] = "fs"
```

Validate:

```python
coefficient_levels = n_levels if self.factor_smooth_factor_basis == "fs" else n_levels - 1
if self.n_cols != coefficient_levels * block_size:
    raise ValueError(
        "factor_smooth n_cols does not match its factor basis, level count, and block size"
    )
```

- [ ] **Step 7: Run tests and commit**

```bash
rtk uv run pytest tests/test_factor_smooth_sz_feature.py tests/test_factor_smooth_feature.py -q
rtk uv run ruff check src/superglm/factor_smooth_geometry.py src/superglm/features/factor_smooth.py src/superglm/types.py tests/test_factor_smooth_sz_feature.py
rtk git add src/superglm/factor_smooth_geometry.py src/superglm/features/factor_smooth.py src/superglm/types.py tests/test_factor_smooth_sz_feature.py
rtk git commit -m "Add sum-to-zero factor smooth geometry"
```

Expected: SZ feature tests pass and existing FS tests remain unchanged.

---

### Task 3: Compact exact/discrete SZ matrix algebra

**Files:**

- Modify: `src/superglm/dm_builder.py`
- Modify: `src/superglm/_group_matrix/_group_matrix_core.py`
- Create: `tests/test_factor_smooth_sz_matrix.py`
- Modify: `tests/test_factor_smooth_matrix.py`
- Modify: `tests/test_factor_smooth_discrete.py`

- [ ] **Step 1: Write dense-oracle tests for exact and discrete matrices**

Construct `C = sum_to_zero_contrast(K)`, marginal `B`, and
`X_ref[row] = kron(C[group[row]], B[row])`. Assert for both exact and discrete
objects:

```python
np.testing.assert_allclose(gm.toarray(), X_ref, atol=1e-13)
np.testing.assert_allclose(gm.matvec(beta), X_ref @ beta, atol=1e-13)
np.testing.assert_allclose(gm.rmatvec(rows), X_ref.T @ rows, atol=1e-13)
np.testing.assert_allclose(gm.gram(W), X_ref.T @ (W[:, None] * X_ref), atol=1e-12)
```

Assert raw sufficient statistics retain shape `(K, k, k)`, while
`gram_rmatvec()` returns public shapes `((K - 1)k, (K - 1)k)` and
`((K - 1)k,)`. For a narrow ordinary matrix, assert
`factor_smooth_dense_cross_gram()` retains raw shape `(K, k, q)` and that
applying the contrast adjoint matches `X_ref.T @ W @ X_small`; also assert the
generic `_cross_gram()` returns that public oracle in both argument orders.
Add a monkeypatch guard proving these fused operations and
`factor_smooth_sufficient_stats()` do not call `toarray()`.

- [ ] **Step 2: Run matrix tests and verify failure**

```bash
rtk uv run pytest tests/test_factor_smooth_sz_matrix.py -q
```

Expected: constructor/shape and matvec failures because the matrix still
assumes `Kk` independent blocks.

- [ ] **Step 3: Pass factor basis into `FactorSmoothGroupMatrix`**

In `dm_builder.py`, pass:

```python
factor_basis=info.factor_smooth_factor_basis,
```

Add the keyword-only constructor parameter and slot:

```python
factor_basis: Literal["fs", "sz"] = "fs"
```

Validate it and set:

```python
self.factor_basis = factor_basis
coefficient_levels = n_levels if factor_basis == "fs" else n_levels - 1
self.shape = (n_rows, coefficient_levels * self.block_size)
```

Preserve it in both `row_subset()` branches.

- [ ] **Step 4: Apply contrast algebra around the existing compiled kernels**

Do not modify the CSR/support Numba kernels. Before their `matvec`, expand SZ
free blocks:

```python
natural_coefficients = coefficients.reshape(self.coefficient_levels, self.block_size)
if self.factor_basis == "sz":
    natural_coefficients = expand_sum_to_zero_blocks(natural_coefficients)
raw_coefficients = natural_coefficients @ self.natural_map.T
```

After raw `rmatvec` and natural transformation:

```python
natural = np.asarray(raw @ self.natural_map, dtype=np.float64)
if self.factor_basis == "sz":
    natural = adjoint_sum_to_zero_blocks(natural)
return natural.ravel()
```

Keep `factor_smooth_sufficient_stats()` raw-level because the constrained
solver needs all symmetric levels.

- [ ] **Step 5: Implement small-model public Gram/oracle conversion**

For SZ:

```python
def _sum_to_zero_public_gram(local: NDArray) -> NDArray:
    n_levels, block_size, _ = local.shape
    free_levels = n_levels - 1
    result = np.zeros((free_levels * block_size, free_levels * block_size))
    last = local[-1]
    for left in range(free_levels):
        left_sl = slice(left * block_size, (left + 1) * block_size)
        result[left_sl, left_sl] += local[left]
        for right in range(free_levels):
            right_sl = slice(right * block_size, (right + 1) * block_size)
            result[left_sl, right_sl] += last
    return result
```

Use `adjoint_sum_to_zero_blocks()` for public transpose products. Build
`toarray()` directly from the marginal basis and contrast row; never create a
one-hot matrix.

- [ ] **Step 6: Run matrix suites and commit**

```bash
rtk uv run pytest tests/test_factor_smooth_sz_matrix.py tests/test_factor_smooth_matrix.py tests/test_factor_smooth_discrete.py -q
rtk uv run ruff check src/superglm/dm_builder.py src/superglm/_group_matrix/_group_matrix_core.py tests/test_factor_smooth_sz_matrix.py
rtk git add src/superglm/dm_builder.py src/superglm/_group_matrix/_group_matrix_core.py tests/test_factor_smooth_sz_matrix.py tests/test_factor_smooth_matrix.py tests/test_factor_smooth_discrete.py
rtk git commit -m "Add compact SZ matrix operations"
```

Expected: dense oracles pass for exact/discrete and existing FS kernel tests
remain green.

---

### Task 4: Compact SZ penalty algebra

**Files:**

- Modify: `src/superglm/types.py`
- Modify: `src/superglm/solvers/hessian_factor.py`
- Modify: `src/superglm/reml/penalty_algebra.py`
- Create: `tests/test_factor_smooth_sz_penalties.py`
- Modify: `tests/test_factor_smooth_penalties.py`

- [ ] **Step 1: Write failing spectral and operator tests**

For `K=4`, `k=6`, `m=2`, assert:

```python
component = next(pc for pc in penalties if pc.name == "x:g:sz:wiggle")
Omega = penalty_component_dense_matrix(component, gm)
expected = np.kron(C.T @ C, S)
np.testing.assert_allclose(Omega, expected, atol=1e-13)
assert component.penalty_kind == "sum_to_zero"
assert component.rank == 12
positive = np.linalg.eigvalsh(S)
positive = positive[positive > np.finfo(np.float64).eps * positive.max() * 100]
assert component.log_det_omega_plus == pytest.approx(
    3 * np.log(positive).sum() + 4 * np.log(4),
)
np.testing.assert_allclose(
    penalty_component_matvec(component, beta, gm),
    expected @ beta,
)
assert penalty_component_quadratic(component, beta, gm) == pytest.approx(
    beta @ expected @ beta
)
```

Also compare that value to the positive eigenvalues of the fully materialized
`expected` matrix, so the closed-form identity and dense oracle must agree.
Build a model with ordinary spline penalties before and after the SZ term,
permute the returned `PenaltyComponent` list in a unit-level assembly test,
and assert the SZ component is resolved by its stable group/name/kind rather
than a positional cursor.

- [ ] **Step 2: Run the penalty tests and verify failure**

```bash
rtk uv run pytest tests/test_factor_smooth_sz_penalties.py -q
```

Expected: the current repeated-penalty invariant rejects `(K - 1)k` public
width and computes FS rank/logdet.

- [ ] **Step 3: Add the compact penalty kind**

Extend `PenaltyComponent.penalty_kind`:

```python
penalty_kind: Literal["dense", "identity", "repeated", "sum_to_zero"] = "dense"
```

In `build_penalty_components`, branch by `gm.factor_basis`:

```python
if gm.factor_basis == "sz":
    if len(gm.repeated_penalty_components) != 1:
        raise ValueError("SZ requires exactly one marginal wiggle penalty")
    suffix, omega_j = gm.repeated_penalty_components[0]
    local_rank, local_log_det, local_eigvals, omega_ssp_j = _rank_and_logdet(
        omega_j, omega_j
    )
    K = gm.n_levels
    eigvals = np.concatenate(
        (
            np.tile(local_eigvals, max(K - 2, 0)),
            K * local_eigvals,
        )
    )
    group_components.append(
        PenaltyComponent(
            name=f"{g.name}:{suffix}",
            group_name=g.name,
            group_index=idx,
            group_sl=g.sl,
            omega_raw=omega_j,
            omega_ssp=omega_ssp_j,
            rank=float((K - 1) * local_rank),
            log_det_omega_plus=float(
                (K - 1) * local_log_det + local_rank * np.log(K)
            ),
            eigvals_omega=np.sort(eigvals)[::-1],
            component_type="wiggle",
            lambda_policy=lp_map.get(suffix),
            penalty_kind="sum_to_zero",
            repeat_count=K,
            block_width=gm.block_size,
        )
    )
```

Leave the existing FS repeated branch byte-for-byte equivalent.

- [ ] **Step 4: Implement dense, quadratic, and matvec helpers**

Validate `repeat_count=K`, `block_width=k`, and public width `(K - 1)k`.
Materialize with `sum_to_zero_penalty(omega, K)`.

For compact operations:

```python
blocks = beta.reshape(K - 1, k)
raw = expand_sum_to_zero_blocks(blocks)
quadratic = np.einsum("ki,ij,kj->", raw, omega, raw, optimize=True)
raw_product = raw @ omega.T
product = adjoint_sum_to_zero_blocks(raw_product).ravel()
```

Teach `build_penalty_matrix()` and
`solvers.hessian_factor._component_omega()` to use the same compact
materializer only on explicit dense-reference paths.

- [ ] **Step 5: Run penalty and existing REML algebra tests, then commit**

```bash
rtk uv run pytest tests/test_factor_smooth_sz_penalties.py tests/test_factor_smooth_penalties.py tests/test_multi_penalty.py tests/test_penalties.py -q
rtk uv run ruff check src/superglm/types.py src/superglm/solvers/hessian_factor.py src/superglm/reml/penalty_algebra.py tests/test_factor_smooth_sz_penalties.py
rtk git add src/superglm/types.py src/superglm/solvers/hessian_factor.py src/superglm/reml/penalty_algebra.py tests/test_factor_smooth_sz_penalties.py tests/test_factor_smooth_penalties.py
rtk git commit -m "Add compact SZ penalty algebra"
```

Expected: SZ spectrum matches dense eigendecomposition and all existing
penalty kinds remain green.

---

### Task 5: Public SZ compact operator

**Files:**

- Modify: `src/superglm/solvers/structured.py`
- Modify: `tests/test_factor_smooth_sz_matrix.py`
- Create: `tests/test_sum_to_zero_structured_factor.py`

- [ ] **Step 1: Write a failing compact-operator dense oracle**

Build random symmetric raw `D_j`, raw cross `C_j`, and ordinary `A`; construct
the public dense matrix with `T = diag(I_q, C_factor kron I_k)`. Assert:

```python
operator = SumToZeroBlockOperator(
    A=A,
    C=C_raw,
    D=D_raw,
    small_indices=small_indices,
    structured_indices=structured_indices,
)
np.testing.assert_allclose(
    operator.matvec(rhs),
    (T.T @ H_raw @ T) @ rhs,
    atol=1e-12,
)
np.testing.assert_allclose(
    compact_operator_diagonal(operator),
    np.diag(T.T @ H_raw @ T),
    atol=1e-12,
)
```

- [ ] **Step 2: Run the test and verify missing operator failure**

```bash
rtk uv run pytest tests/test_sum_to_zero_structured_factor.py::test_sum_to_zero_operator_matches_dense_free_coordinates -q
```

Expected: import failure for `SumToZeroBlockOperator`.

- [ ] **Step 3: Implement `SumToZeroBlockOperator`**

Add beside `BlockSymmetricOperator`:

```python
@dataclass(frozen=True)
class SumToZeroBlockOperator:
    A: NDArray
    C: NDArray
    D: NDArray
    small_indices: NDArray
    structured_indices: NDArray
    shape: tuple[int, int] = field(init=False)

    def __post_init__(self) -> None:
        for name, dtype in (
            ("A", np.float64),
            ("C", np.float64),
            ("D", np.float64),
            ("small_indices", np.intp),
            ("structured_indices", np.intp),
        ):
            values = np.array(getattr(self, name), dtype=dtype, copy=True)
            values.setflags(write=False)
            object.__setattr__(self, name, values)
        if self.C.ndim != 3:
            raise ValueError("C must have shape (K, k, q)")
        K, k, q = self.C.shape
        if K < 2 or self.D.shape != (K, k, k):
            raise ValueError("SZ raw blocks must have shapes (K,k,q) and (K,k,k)")
        if self.A.shape != (q, q):
            raise ValueError("SZ ordinary block has the wrong shape")
        if self.small_indices.shape != (q,):
            raise ValueError("SZ small_indices width does not match A")
        if self.structured_indices.shape != (K - 1, k):
            raise ValueError("SZ public indices must have shape (K-1,k)")
        if not all(np.all(np.isfinite(values)) for values in (self.A, self.C, self.D)):
            raise ValueError("SZ operator blocks must be finite")
        if not np.allclose(self.A, self.A.T, rtol=0.0, atol=1e-13):
            raise ValueError("SZ ordinary block must be symmetric")
        if not np.allclose(self.D, self.D.transpose(0, 2, 1), rtol=0.0, atol=1e-13):
            raise ValueError("Every SZ local block must be symmetric")
        all_indices = np.concatenate((self.small_indices, self.structured_indices.ravel()))
        if len(np.unique(all_indices)) != len(all_indices):
            raise ValueError("SZ index partitions must be disjoint")
        if not np.array_equal(np.sort(all_indices), np.arange(len(all_indices))):
            raise ValueError("SZ index partitions must cover every coefficient once")
        object.__setattr__(self, "shape", (len(all_indices), len(all_indices)))

    @property
    def n_levels(self) -> int:
        return int(self.C.shape[0])

    @property
    def block_size(self) -> int:
        return int(self.C.shape[1])

    def matvec(self, rhs: NDArray) -> NDArray:
        values = np.asarray(rhs, dtype=np.float64)
        vector = values.ndim == 1
        if vector:
            values = values[:, None]
        if values.ndim != 2 or values.shape[0] != self.shape[0]:
            raise ValueError(
                f"rhs must have shape ({self.shape[0]},) or ({self.shape[0]}, m)"
            )
        small = values[self.small_indices]
        free = values[self.structured_indices]
        raw = expand_sum_to_zero_blocks(free)
        raw_result = (
            np.einsum("kiq,qm->kim", self.C, small, optimize=True)
            + np.einsum("kij,kjm->kim", self.D, raw, optimize=True)
        )
        result = np.empty_like(values)
        result[self.small_indices] = self.A @ small + np.einsum(
            "kiq,kim->qm", self.C, raw, optimize=True
        )
        result[self.structured_indices] = adjoint_sum_to_zero_blocks(raw_result)
        return result[:, 0] if vector else result
```

Include the same finite, symmetry, shape, uniqueness, and complete-partition
checks as `BlockSymmetricOperator`.

- [ ] **Step 4: Extend compact operator unions and BDLR conversion**

Add SZ to `CenteredBlockOperator`, `SumBlockOperator`, and
`CompactSymmetricOperator` unions. Convert it to public BDLR as:

```python
base = BlockSymmetricOperator(
    A=operator.A,
    C=operator.C[:-1] - operator.C[-1:],
    D=operator.D[:-1],
    small_indices=operator.small_indices,
    structured_indices=operator.structured_indices,
)
last_basis = np.zeros((operator.shape[0], operator.block_size))
for indices in operator.structured_indices:
    last_basis[indices] = np.eye(operator.block_size)
last = _BlockDiagonalLowRank(
    blocks=np.zeros_like(operator.D[:-1]),
    structured_indices=operator.structured_indices,
    basis=last_basis,
    core=operator.D[-1],
    shape=operator.shape,
)
return _merge_bdlr((_block_operator_bdlr(base), last))
```

The SZ diagonal is `diag(D_j + D_K)` for each public level block. Centering
continues to apply through the existing rank-two update.

- [ ] **Step 5: Run the operator test and commit**

```bash
rtk uv run pytest tests/test_sum_to_zero_structured_factor.py::test_sum_to_zero_operator_matches_dense_free_coordinates -q
rtk uv run ruff check src/superglm/solvers/structured.py tests/test_sum_to_zero_structured_factor.py
rtk git add src/superglm/solvers/structured.py tests/test_sum_to_zero_structured_factor.py
rtk git commit -m "Add compact SZ structured operator"
```

Expected: operator matvec and diagonal agree with dense free-coordinate
algebra.

---

### Task 6: Rank-aware constrained SZ factor

**Files:**

- Create: `src/superglm/solvers/sum_to_zero.py`
- Modify: `src/superglm/solvers/structured.py`
- Modify: `tests/test_sum_to_zero_structured_factor.py`

- [ ] **Step 1: Write failing randomized factor protocol tests**

For random `K`, `k`, and `q` (including `q=0`), construct an identifiable raw
system, public dense `H_free`, and `DenseHessianFactor`. Compare:

```python
factor = SumToZeroBlockFactor(
    A=A,
    C=C_raw,
    D=D_raw,
    small_indices=small_indices,
    structured_indices=structured_indices,
    term_name="x:g:sz",
    level_labels=tuple(f"level_{index}" for index in range(K)),
)
np.testing.assert_allclose(factor.solve(rhs), np.linalg.solve(H_free, rhs), rtol=2e-11)
assert factor.logdet() == pytest.approx(np.linalg.slogdet(H_free)[1], abs=2e-10)
np.testing.assert_allclose(
    factor.selected_inverse_block(selected),
    np.linalg.inv(H_free)[np.ix_(selected, selected)],
    rtol=2e-10,
)
np.testing.assert_allclose(
    factor.selected_inverse_diagonal(np.arange(H_free.shape[0])),
    np.diag(np.linalg.inv(H_free)),
    rtol=2e-10,
)
```

Compare every `HessianFactor` trace/operator method against
`DenseHessianFactor`. Add:

- an identifiable example with singular local blocks;
- an unidentifiable example whose error names the term and deficient levels;
- permutation tests proving no raw level is numerically privileged;
- `raw_level_inverse_block(level)` for every level, including the reconstructed
  final level;
- a determinant test that the absolute KKT determinant equals the free Hessian
  determinant.

- [ ] **Step 2: Run the factor test and verify missing implementation**

```bash
rtk uv run pytest tests/test_sum_to_zero_structured_factor.py -q
```

Expected: import/factor failures after the operator-only test.

- [ ] **Step 3: Implement local PSD decomposition**

Create a private immutable record containing `pinv`, `null`, positive
eigenvalues, and rank. Use:

```python
eigenvalues, eigenvectors = scipy.linalg.eigh(
    0.5 * (block + block.T),
    driver="evr",
    check_finite=False,
)
scale = max(float(np.max(np.abs(eigenvalues), initial=0.0)), 1.0)
threshold = max(
    np.finfo(np.float64).eps ** (2 / 3) * scale,
    np.finfo(np.float64).eps * block.shape[0] * scale * 10.0,
)
if eigenvalues[0] < -threshold:
    raise np.linalg.LinAlgError(
        f"Structured term {term_name!r} level {level!r} has negative local curvature"
    )
positive = eigenvalues > threshold
U_pos = eigenvectors[:, positive]
pinv = (U_pos / eigenvalues[positive]) @ U_pos.T
null = eigenvectors[:, ~positive]
```

Accept `level_labels: tuple[Any, ...] | None`, validate its length against
`K`, and default to integer indices only for low-level callers. Record
deficient fitted labels and require the final bordered system, not each local
block, to establish identifiability.

Keep import ownership acyclic: `sum_to_zero.py` may import the private BDLR
records/helpers from `structured.py`, while `structured.py` imports
`SumToZeroBlockFactor` and `ProfiledSumToZeroBlockFactor` only inside the
factor-construction functions.

- [ ] **Step 4: Build and factor the symmetric dense border**

For `P_j = D_j+`, `N_j = null(D_j)`, assemble:

```text
Q = A - sum(C_j' P_j C_j)
R = sum(P_j C_j)
M = sum(P_j)
E = [C_1' N_1, ..., C_K' N_K]
N = [N_1, ..., N_K]

border = [[ Q,  E, -R' ],
          [ E', 0,  N' ],
          [-R,  N,  -M ]]
```

Use `scipy.linalg.ldl` as the primary factor. Solve the permuted unit-lower,
block-diagonal, and upper systems once per RHS. Compute log-absolute-determinant
and inertia from the LDL `D`. Fall back to SVD only when LDL factorization or
its residual check fails.

Require inertia `(public_width - local_positive_rank, k, 0)` for the border;
combined with the eliminated positive directions this proves the public
Hessian is positive definite. A zero count raises:

```python
np.linalg.LinAlgError(
    f"Structured SZ term {term_name!r} is globally unidentifiable after "
    f"enforcing sum-to-zero; deficient fitted levels={deficient_levels!r}. "
    "Use basis='fs', reduce k, or provide more numeric support."
)
```

- [ ] **Step 5: Recover coefficients and construct inverse BDLR**

The border unknown order is `[ordinary, promoted_null, multiplier]`. Build
`U_raw` so:

```text
a       = I * a_border
theta_j = -P_j C_j a_border + N_j gamma_j - P_j multiplier
```

Then:

```text
Cov_raw = blockdiag(0_q, P_1, ..., P_K) + U_raw border^-1 U_raw'
```

The public coefficients are ordinary coefficients plus raw levels
`0 .. K-2`, so drop only the final raw level from this BDLR. Implement
`solve`, selected covariance, and raw-level covariance from those stored
blocks/bases. Set:

```python
self._logdet = sum(np.log(local.positive_eigenvalues).sum() for local in locals)
self._logdet += border_factor.logabsdet
```

- [ ] **Step 6: Implement the complete `HessianFactor` protocol**

Use existing compact BDLR helpers explicitly:

```python
def trace_inverse_operator(self, operator):
    return _trace_symmetric_bdlr(
        self._inverse_bdlr(),
        _operator_bdlr(operator, self.structured_indices),
    )

def inverse_operator_diagonal(self, operator):
    return _general_bdlr_diagonal(
        _multiply_symmetric_bdlr(
            self._inverse_bdlr(),
            _operator_bdlr(operator, self.structured_indices),
        )
    )

def inverse_operator_square_diagonal(self, operator):
    return _general_bdlr_square_diagonal(
        _multiply_symmetric_bdlr(
            self._inverse_bdlr(),
            _operator_bdlr(operator, self.structured_indices),
        )
    )
```

Implement `operator_cross_trace` and `penalty_operator_cross_trace` with the
matching existing helpers. The SZ penalty operator has zero `A/C` and the same
`scale * omega_ssp` in every one of its `K` raw `D` blocks.

- [ ] **Step 7: Add `ProfiledSumToZeroBlockFactor`**

Mirror the existing intercept-profiling contract without materializing an
inverse:

```python
self.shape = (len(xtw), len(xtw))
self.small_indices = augmented_factor.small_indices[1:] - 1
self.structured_indices = augmented_factor.structured_indices - 1

def solve(self, rhs):
    augmented_rhs = np.zeros((self.shape[0] + 1, rhs_columns))
    augmented_rhs[1:] = rhs_2d
    return self.augmented_factor.solve(augmented_rhs)[1:]

def logdet(self) -> float:
    return float(self.augmented_factor.logdet() - np.log(self.sum_w))
```

Drop the intercept row from the augmented inverse BDLR for compact slope
traces. Shift penalty/index requests by one. Delegate
`raw_level_inverse_block()` to the augmented factor with slope indexing.

- [ ] **Step 8: Run the full factor suite and commit**

```bash
rtk uv run pytest tests/test_sum_to_zero_structured_factor.py tests/test_structured_factor.py -q
rtk uv run ruff check src/superglm/solvers/sum_to_zero.py src/superglm/solvers/structured.py tests/test_sum_to_zero_structured_factor.py
rtk git add src/superglm/solvers/sum_to_zero.py src/superglm/solvers/structured.py tests/test_sum_to_zero_structured_factor.py
rtk git commit -m "Add rank-aware constrained SZ factor"
```

Expected: all dense-oracle protocol comparisons pass, identifiable singular
locals succeed, and globally deficient systems fail specifically.

---

### Task 7: Structured PIRLS and REML integration

**Files:**

- Modify: `src/superglm/solvers/structured.py`
- Modify: `src/superglm/solvers/irls_direct.py`
- Modify: `src/superglm/reml/observed_geometry.py`
- Modify: `src/superglm/model/reml_finalize.py`
- Create: `tests/test_factor_smooth_sz_reml.py`
- Modify: `tests/test_factor_smooth_structured_system.py`
- Modify: `tests/test_factor_smooth_structured_parity.py`
- Modify: `tests/test_structured_allocations.py`

- [ ] **Step 1: Write failing public-system and end-to-end parity tests**

Add a construction test showing `build_structured_system()` returns public
moments equal to `X_sz.T @ W` while retaining raw `(K,k,k)` blocks. Add
Gaussian and Poisson models:

```python
features = {"x": Spline(kind="ps", k=7, m=2)}
interactions = [FactorSmooth("x", group="g", basis="sz", k=6, m=2)]
```

Fit otherwise identical `"gram"` and `"structured"` models for exact and
`discrete=True`. Compare beta, intercept, deviance, EDF, fitted lambdas,
predictions, REML objective, gradient, Hessian, and direct backend.
Cover both an estimated wiggle lambda and
`lambda_policy={"wiggle": LambdaPolicy(mode="fixed", value=...)}`. Assert
there is exactly one SZ smoothing parameter, and check the estimated-lambda
REML gradient and Hessian against central finite differences in log-lambda
coordinates.

Monkeypatch these SZ methods to raise on the structured path:

```python
FactorSmoothGroupMatrix.toarray
FactorSmoothGroupMatrix.gram
penalty_component_dense_matrix
```

The structured fit must still pass.

Add an ineligible dense-small configuration: `direct_solve="structured"`
raises the existing explicit eligibility error without materializing SZ,
while `"auto"` records a non-empty fallback reason and produces the dense
reference result on a size kept deliberately small.

- [ ] **Step 2: Run integration tests and verify dispatch/shape failures**

```bash
rtk uv run pytest tests/test_factor_smooth_sz_reml.py tests/test_factor_smooth_structured_system.py -q
```

Expected: raw/public moment shape mismatch and missing constrained-factor
dispatch.

- [ ] **Step 3: Add `SumToZeroBlockStructuredSystem`**

Store:

```python
operator: SumToZeroBlockOperator
xtw_small: NDArray
xtw_structured: NDArray          # public C' raw X'W
xtwz_small: NDArray
xtwz_structured: NDArray         # public C' raw X'Wz
raw_xtw_structured: NDArray      # (K, k)
raw_xtwz_structured: NDArray     # (K, k)
sum_w: float
sum_wz: float
dominant_group_index: int
dominant_group_name: str
```

In `build_block_structured_system`, branch on
`dominant.factor_basis == "sz"`. Use the same fused raw sufficient-statistic
and dense-cross kernels, construct `SumToZeroBlockOperator`, and obtain public
transpose products with `adjoint_sum_to_zero_blocks()`.

- [ ] **Step 4: Make layout and auto-crossover geometry-aware**

For SZ, reshape public structured indices to `(K - 1, k)`. The auto cost uses:

```python
coefficient_width = (K - 1) * k + q
border_width = q + k + 1  # includes intercept during PIRLS
structured_cost = (
    K * k**3
    + K * k**2 * border_width
    + border_width**3
)
```

FS retains its existing estimate.

- [ ] **Step 5: Assemble raw penalties and augmented constrained factors**

When the operator is SZ:

- add the shared marginal `lambda * S` to all `K` raw `D` blocks;
- accept only `penalty_kind == "sum_to_zero"` for the dominant group;
- build augmented raw `C` by prepending `raw_xtw_structured`;
- build raw RHS by prepending the intercept RHS separately and retaining all
  `K` raw level right-hand sides;
- instantiate `SumToZeroBlockFactor`;
- pass the fitted `FactorSmoothGroupMatrix.levels` so numerical errors name
  real group labels;
- wrap it in `ProfiledSumToZeroBlockFactor`.

Extend `solve_cached_structured()` and cached result unions accordingly.

- [ ] **Step 6: Extend IRLS/finalization without geometry-specific dense arrays**

Use `system.xtw_structured` for public `XtW1`, while augmented constrained
factor construction consumes `raw_xtw_structured`. Accept the new operator and
factor in final type checks. Build retained `CenteredBlockOperator` over the
public SZ operator and store raw per-level information from `operator.D`.

In observed geometry, use `augmented_factor.public_positive_definite` for SZ
instead of rejecting the expected negative constraint inertia. Preserve the
existing Schur eigenvalue check for FS/RE.

- [ ] **Step 7: Run structured, REML, derivative, and allocation tests**

```bash
rtk uv run pytest tests/test_factor_smooth_sz_reml.py tests/test_factor_smooth_structured_system.py tests/test_factor_smooth_structured_parity.py tests/test_structured_allocations.py tests/test_reml_fd.py tests/test_reml_observed_geometry.py -q
```

Expected: exact/discrete structured results match dense reference tolerances;
allocation guards prove no expanded SZ design, penalty, or Hessian.

- [ ] **Step 8: Commit**

```bash
rtk uv run ruff check src/superglm/solvers/structured.py src/superglm/solvers/irls_direct.py src/superglm/reml/observed_geometry.py src/superglm/model/reml_finalize.py tests/test_factor_smooth_sz_reml.py
rtk git add src/superglm/solvers/structured.py src/superglm/solvers/irls_direct.py src/superglm/reml/observed_geometry.py src/superglm/model/reml_finalize.py tests/test_factor_smooth_sz_reml.py tests/test_factor_smooth_structured_system.py tests/test_factor_smooth_structured_parity.py tests/test_structured_allocations.py
rtk git commit -m "Integrate structured SZ REML fitting"
```

---

### Task 8: Basis-aware inference, prediction edge cases, and reporting

**Files:**

- Modify: `src/superglm/inference/covariance.py`
- Modify: `src/superglm/inference/factor_smooths.py`
- Modify: `src/superglm/model/api.py`
- Create: `tests/test_factor_smooth_sz_inference.py`
- Modify: `tests/test_factor_smooth_inference.py`

- [ ] **Step 1: Write failing SZ reporting and prediction tests**

Assert:

```python
report = model.factor_smooth("x:g:sz", grid=grid)
assert report.basis == "sz"
assert report.collapsed is None
assert set(report.lambdas) == {"wiggle"}
assert "credibility" not in report.table
assert "shrinkage" not in report.table
assert "collapsed" not in report.table
assert report.diagnostics["max_abs_level_effect_sum"] < 1e-11
```

Also assert:

- all fitted levels, including the final sorted level, have curves and finite
  intervals;
- all fitted levels have finite `effective_df`, its sum equals the term EDF,
  and relabeling/permuting the fitted levels only permutes those values;
- curve effects sum to zero at every grid point;
- one-row predictions retain shape `(1,)`;
- `random_effects="population"` removes only the SZ deviation;
- unseen groups under `"population"` equal the global curve;
- unseen groups under `"error"` raise with the label;
- numeric group keys `0.0` and `-0.0` predict identically;
- `lambda=1e10` leaves finite mean-zero polynomial deviations instead of
  claiming collapse;
- `retain_fit_state=False` still reports through retained compact state;
- a co-fitted second global spline remains present in `summary()` and
  `smooth()` output when SZ levels are sparse or uneven.

- [ ] **Step 2: Run inference tests and verify FS assumptions fail**

```bash
rtk uv run pytest tests/test_factor_smooth_sz_inference.py -q
```

Expected: repeated-penalty lookup, `Kk` coefficient reshape, credibility, and
final-level covariance failures.

- [ ] **Step 3: Generalize result metadata and penalty lookup**

Change:

```python
class FactorSmoothResult:
    basis: Literal["fs", "sz"]
    collapsed: bool | None
```

Accept `"repeated"` for FS and `"sum_to_zero"` for SZ, and require exactly the
component geometry expected by `spec.basis`.

- [ ] **Step 4: Preserve the FS reporting branch exactly**

Move the existing local credibility, shrinkage, collapse warning, table, and
curve code into an FS-specific helper with no numerical changes. Existing
`tests/test_factor_smooth_inference.py` must pass unchanged.

- [ ] **Step 5: Implement SZ reporting**

Expand fitted public coefficients through `spec._level_blocks()`. Build a
table containing:

```python
{
    "level": spec._levels,
    "count": support.count,
    "fit_weight": support.fit_weight,
    "information_trace": np.trace(support.information, axis1=1, axis2=2),
    "information_rank": information_rank,
    "effective_df": level_edf,
    "coefficient_norm": np.linalg.norm(raw_coefficients, axis=1),
    "has_information": has_information,
    "sufficient_support": sufficient_support,
}
```

For public levels, use the selected `k x k` covariance. For the reconstructed
final level, call `raw_level_inverse_block()` on the retained structured factor;
on the dense backend, apply the `[-I, ..., -I]` contrast to the already-dense
group covariance. Compute curve variance as `B Cov_level B'`.

Do not reshape the `(K - 1)k` public coefficient EDF into levels. Attribute
the term EDF symmetrically in raw constrained coordinates:

```python
base_level_df = (K - 1) * spec.k / K
scaled_penalties = [
    (
        fitted_lambdas[component.name],
        np.asarray(component.omega_ssp, dtype=np.float64),
    )
    for component in penalties
]
level_edf = np.array(
    [
        base_level_df
        - sum(
            smoothing_lambda * np.trace(raw_inverse_blocks[level] @ omega)
            for smoothing_lambda, omega in scaled_penalties
        )
        for level in range(K)
    ]
)
```

Use unscaled inverse-Hessian blocks here and multiply by `phi` only for curve
posterior covariance. Assert
`level_edf.sum() == group_width - trace(H^-1 Omega_total)` and equals the
existing group EDF within tolerance. This gives every raw level an equal share
of the `k` dimensions removed by the sum-to-zero constraint and is invariant
to which sorted level is stored last.

Set:

```python
collapsed = None
diagnostics["max_abs_level_effect_sum"] = float(
    np.max(
        np.abs(
            curves.pivot(
                index=spec.variable,
                columns="level",
                values="effect",
            ).sum(axis=1)
        )
    )
)
```

Do not emit a collapse warning or label this as credibility.

- [ ] **Step 6: Run reporting/inference suites and commit**

```bash
rtk uv run pytest tests/test_factor_smooth_sz_inference.py tests/test_factor_smooth_inference.py tests/test_factor_smooth_mgcv_parity.py -q
rtk uv run ruff check src/superglm/inference/covariance.py src/superglm/inference/factor_smooths.py src/superglm/model/api.py tests/test_factor_smooth_sz_inference.py
rtk git add src/superglm/inference/covariance.py src/superglm/inference/factor_smooths.py src/superglm/model/api.py tests/test_factor_smooth_sz_inference.py tests/test_factor_smooth_inference.py
rtk git commit -m "Add SZ prediction and reporting"
```

Expected: FS reports remain unchanged and SZ reports never claim full
credibility shrinkage.

---

### Task 9: Pin clean-room mgcv 1.9-4 parity

**Files:**

- Create: `tests/fixtures/factor_smooth_sz_mgcv_reference.R`
- Create: `tests/fixtures/factor_smooth_sz_mgcv_reference.json`
- Create: `tests/test_factor_smooth_sz_mgcv_parity.py`

- [ ] **Step 1: Write the fixture generator**

Reuse only the repository-owned JSON serializer structure from the FS fixture.
Generate deterministic Gaussian `gam`, Poisson `gam`, and Poisson
`bam(discrete=TRUE)` fits with:

```r
y ~ s(x, bs = "ps", k = 7, m = 2) +
  s(x, f, bs = "sz", k = 6, xt = list(bs = "ps"), m = 2, id = 1)
```

Record:

- `R.version.string`, `packageVersion("mgcv")`, seed, and exact formula;
- data and prediction frames;
- response predictions, global-only predictions, deviation term, deviance,
  scale, total/SZ/global EDF;
- raw `sp`, `S.scale`, and `sp / S.scale`;
- construction fixture from `smoothCon(..., absorb.cons=TRUE)`: `ncol(X)`,
  penalty count/rank/nullity, flattened design, flattened penalty, and
  one-row `PredictMat`;
- the corresponding no-`id` smoothing-parameter count.

No mgcv source text or implementation code enters the fixture.

- [ ] **Step 2: Generate and inspect the pinned JSON**

```bash
rtk Rscript tests/fixtures/factor_smooth_sz_mgcv_reference.R tests/fixtures/factor_smooth_sz_mgcv_reference.json
rtk json tests/fixtures/factor_smooth_sz_mgcv_reference.json
```

Expected metadata: R 4.5.3, mgcv 1.9.4, construction width 18, one shared
penalty, rank 12, nullity 6, and finite `1 x 18` prediction design.

- [ ] **Step 3: Write parity tests**

Fit matching SuperGLM models with explicit `Spline` plus `basis="sz"`.
Define the column mapping explicitly as the sorted-level contrast
`C = [I; -1]` Kronecker the shared marginal P-spline coordinates; if mgcv's
marginal coordinates differ, estimate one deterministic square change of
basis from the construction fixture and apply its congruence to the penalty.
Reject the fixture if that mapping is rank deficient or fails on the separate
one-row `PredictMat` fixture. Compare:

- construction design/penalty after the documented column mapping;
- exact zero-sum over all levels;
- unscaled smoothing lambda;
- predictions and deviation curves;
- deviance and EDF;
- exact/discrete SuperGLM parity;
- one-row prediction;
- unseen SuperGLM population prediction against mgcv prediction with the SZ
  term excluded.

Start with construction `atol=2e-10`; Gaussian prediction/deviance/EDF
`rtol=1e-3`, `rtol=5e-6`, `atol=5e-3`; Poisson
`rtol=1e-2`, `rtol=5e-4`, `atol=1e-1`; and discrete Poisson
`rtol=1.2e-2`, `rtol=8e-4`, `atol=1.2e-1`. Require the unscaled shared lambda
within 5%. If any bound fails, diagnose basis scaling, convergence, or fixture
mapping; do not widen it without recording a specific numerical cause beside
the test.

- [ ] **Step 4: Run parity tests and commit**

```bash
rtk uv run pytest tests/test_factor_smooth_sz_mgcv_parity.py tests/test_factor_smooth_mgcv_parity.py -q
rtk git add tests/fixtures/factor_smooth_sz_mgcv_reference.R tests/fixtures/factor_smooth_sz_mgcv_reference.json tests/test_factor_smooth_sz_mgcv_parity.py
rtk git commit -m "Pin SZ parity against mgcv"
```

Expected: pinned construction and fit parity pass without calling R during
pytest.

---

### Task 10: Canonical documentation and shorthand migration

**Files:**

- Modify: `src/superglm/__init__.py`
- Modify: `docs/getting-started/quickstart.md`
- Modify: `docs/guide/interactions.md`
- Modify: `docs/guide/credibility.md`
- Modify: `docs/guide/fitting.md`
- Modify: `docs/api/features.md`
- Modify: `docs/api/inference.md`
- Modify: `tests/test_factor_smooth_sz_feature.py`

- [ ] **Step 1: Write documentation contract checks**

Add a test that scans executable Python examples in public docs and the module
docstring:

```python
import ast
import re
from pathlib import Path

module_doc = ast.get_docstring(ast.parse(Path("src/superglm/__init__.py").read_text()))
python_fence = re.compile(r"```(?:python|py)\n(.*?)```", re.DOTALL)
violations = [
    str(path)
    for path in Path("docs").rglob("*.md")
    for example in python_fence.findall(path.read_text())
    if re.search(r"\bsplines\s*=", example)
]
if re.search(r"\bsplines\s*=", module_doc or ""):
    violations.append("src/superglm/__init__.py")
assert violations == []
```

Assert interactions documentation contains all of:
`basis="fs"`, `basis="sz"`, `group=`, `by=`, `SplineCategorical`,
`random_effects="population"`, and `unseen`.

- [ ] **Step 2: Run the doc contract and verify current shorthand/examples fail**

```bash
rtk uv run pytest tests/test_factor_smooth_sz_feature.py -q
```

Expected: quickstart/module examples and missing SZ guidance fail.

- [ ] **Step 3: Replace public shorthand examples**

Use:

```python
model = SuperGLM(
    family="poisson",
    features={
        "DrivAge": Spline(),
        "VehAge": Spline(),
        "BonusMalus": Spline(),
    },
)
```

Document the legacy `splines` keyword compatibility window and
`FutureWarning` in prose without using an executable shorthand example or
promising a removal version. Keep `n_knots`, `degree`, and `categorical_base`
documented as compatibility controls for that legacy path.

- [ ] **Step 4: Add one explicit FS/SZ choice table**

Document:

| Need | API | Main spline | Sparse-level behavior |
|---|---|---|---|
| Reference-coded fixed interaction | `SplineCategorical` | model-dependent | no pooling |
| Fully penalized random curves | `FactorSmooth(..., basis="fs")` | optional | full wiggle/null shrinkage |
| Centered deviation curves | `FactorSmooth(..., basis="sz")` | required | wiggle shrinks, polynomial null space remains |

Explain that `group=` corresponds to mgcv's factor argument, not generic
`by=`; tuple spline-spline interactions remain `ti()`-style; `te`/`tp` are
not added. Spell out that SZ means “sum-to-zero” and that the result is a
continuous rating curve per level, not a finite two-way lookup table.

Add a hierarchy note: make/model/trim applications can use explicit composite
nested IDs with FS terms for sparse levels, but this release fits their
penalties independently rather than introducing a correlated nested
random-effect covariance. A single SZ term may use one chosen hierarchy
level; stacking make/model/trim SZ terms is not advertised as a hierarchical
decomposition because the initial constraint is global, not sum-to-zero
within each parent. Recommend FS for sparse nested levels and caution against
adding duplicate categorical/random-intercept main effects.

- [ ] **Step 5: Document prediction and compatibility rules**

Use the canonical SZ example:

```python
model = SuperGLM(
    family="poisson",
    features={"age": Spline(kind="ps", k=7, m=2)},
    interactions=[
        FactorSmooth("age", group="region", basis="sz", kind="ps", k=6, m=2)
    ],
).fit_reml(X, y)
```

State known, unseen-population, unseen-error, and
`random_effects="population"` behavior. State that group
`Categorical`/`RandomEffect` duplicates are errors. Clarify that FS null
penalties are REML smoothing/variance components, not selection penalties,
and that SZ intentionally has no null-space selection penalty.

Explain the execution split explicitly: tabmat still handles ordinary dense,
sparse, and categorical blocks and the dense-small side of the solver; the
factor smooth keeps its more compact `codes + shared basis` representation and
existing compiled raw-moment kernels, with SZ contrast algebra wrapped around
them. State why converting it to a generic expanded tabmat sparse block would
increase memory and sandwich-product cost.

- [ ] **Step 6: Build docs, run contracts, and commit**

```bash
rtk uv run pytest tests/test_factor_smooth_sz_feature.py -q
rtk uv run mkdocs build --strict
rtk git add src/superglm/__init__.py docs/getting-started/quickstart.md docs/guide/interactions.md docs/guide/credibility.md docs/guide/fitting.md docs/api/features.md docs/api/inference.md tests/test_factor_smooth_sz_feature.py
rtk git commit -m "Document unified FS and SZ smooths"
```

Expected: strict docs build and public shorthand scan pass.

---

### Task 11: Exact/discrete performance and cProfile evidence

**Files:**

- Modify: `benchmarks/profile_structured_credibility.py`
- Modify: `tests/test_structured_credibility_benchmark.py`
- Modify: `docs/guide/fitting.md`

- [ ] **Step 1: Write failing benchmark-harness tests**

Add `factor_basis: str = "fs"` to `CaseConfig` expectations and test:

```python
config = CaseConfig(
    n=600,
    levels=20,
    family="poisson",
    discrete=True,
    random_effects=0,
    secondary_levels=None,
    small_width=2,
    weights="nonuniform",
    seed=919,
    structured_term="factor_smooth",
    block_size=6,
    global_spline=True,
    factor_basis="sz",
)
prepared = prepare_case(config)
model = _new_model(prepared, "structured")
term = model._interaction_specs["curve_x:curve_group:sz"]
assert term.basis == "sz"
assert config.dominant_width == 114
assert "sz_k6" in config.slug
```

- [ ] **Step 2: Run the harness smoke test and verify failure**

```bash
rtk uv run pytest tests/test_structured_credibility_benchmark.py -q
```

Expected: missing config field and FS-only model construction.

- [ ] **Step 3: Extend the harness**

Validate `factor_basis in ("fs", "sz")`; compute width as `Kk` for FS and
`(K - 1)k` for SZ; require `global_spline=True` for SZ; pass
`basis=config.factor_basis` into `FactorSmooth`; include `--factor-basis` in
CLI and slug. Reject `factor_basis != "fs"` when the structured term is a
plain random effect, where the option has no meaning. When
`--matrix factor-smooth` is selected, derive each matrix case with
`dataclasses.replace(..., factor_basis=args.factor_basis)` and force
`global_spline=True` for SZ; the default `factor_basis="fs"` therefore
preserves the existing FS matrix. Add one more large matrix case, giving exact
and discrete SZ cases at multiple `K` values: at least three sizes where both
backends run, so the measured crossover is bracketed, and two larger
structured-only cases that exercise scaling without a dense allocation.

- [ ] **Step 4: Run clean wall, allocation, and cProfile measurements**

Use a temporary results directory:

```bash
rtk uv run python benchmarks/profile_structured_credibility.py --matrix factor-smooth --factor-basis sz --global-spline --backend both --repetitions 3 --warmups 1 --output-dir /tmp/superglm-sz-profile-matrix
rtk uv run python benchmarks/profile_structured_credibility.py --structured-term factor_smooth --factor-basis sz --global-spline --family poisson --n 6000 --levels 50 --block-size 6 --backend both --repetitions 3 --warmups 1 --cprofile --tracemalloc --output-dir /tmp/superglm-sz-profile-exact
rtk uv run python benchmarks/profile_structured_credibility.py --structured-term factor_smooth --factor-basis sz --global-spline --family poisson --n 20000 --levels 300 --block-size 10 --discrete --backend structured --repetitions 3 --warmups 1 --cprofile --tracemalloc --output-dir /tmp/superglm-sz-profile-discrete
rtk uv run python benchmarks/profile_structured_credibility.py --structured-term random_effect --family poisson --n 2000 --levels 50 --backend structured --repetitions 2 --warmups 1 --max-reml-iter 3 --output-dir /tmp/superglm-re-post-sz
rtk uv run python benchmarks/profile_structured_credibility.py --structured-term factor_smooth --factor-basis fs --global-spline --family poisson --n 2000 --levels 20 --block-size 5 --backend structured --repetitions 2 --warmups 1 --max-reml-iter 3 --output-dir /tmp/superglm-fs-post-sz
rtk json /tmp/superglm-sz-profile-matrix/matrix_summary.json
rtk json /tmp/superglm-sz-profile-exact/summary.json
rtk log /tmp/superglm-sz-profile-exact/cprofile_structured_top.txt
rtk json /tmp/superglm-sz-profile-discrete/summary.json
rtk log /tmp/superglm-sz-profile-discrete/cprofile_structured_top.txt
rtk json /tmp/superglm-re-post-sz/summary.json
rtk json /tmp/superglm-fs-post-sz/summary.json
```

Expected:

- backend is `structured`;
- parity deltas meet test tolerances where dense runs;
- no memory growth proportional to `(Kk)^2`;
- top cumulative stack names the raw sufficient-statistic kernels,
  constrained border factor, REML objective, and PIRLS calls;
- the both-backend matrix contains a reproducible size after which structured
  wall time beats dense. If it does not, treat that as a performance failure:
  inspect the call stack, optimize, and rerun rather than documenting a
  hoped-for crossover;
- same-command RE and FS median wall/RSS remain within 15% of the Task 0
  sentinels. Rerun noisy cases; if a repeatable regression remains, profile
  and fix it before completion.

- [ ] **Step 5: Record measured evidence in fitting docs**

Add a dated table containing the exact command parameters, median clean wall
times, peak Python allocation, sampled peak RSS, selected backend, parity
delta, and top five cumulative cProfile functions from the two
`summary.json`/profile reports. State the observed crossover only for the
measured hardware/runtime and keep `direct_solve="auto"` as the user
recommendation.

- [ ] **Step 6: Run smoke tests and commit**

```bash
rtk uv run pytest tests/test_structured_credibility_benchmark.py tests/test_factor_smooth_sz_reml.py -q
rtk uv run ruff check benchmarks/profile_structured_credibility.py tests/test_structured_credibility_benchmark.py
rtk git add benchmarks/profile_structured_credibility.py tests/test_structured_credibility_benchmark.py docs/guide/fitting.md
rtk git commit -m "Profile structured SZ smooths"
```

---

### Task 12: Full verification and completion audit

**Files:**

- Inspect: all changed paths relative to `origin/master`
- Modify only if a verification failure exposes a defect

- [ ] **Step 1: Run all focused factor-smooth and structured suites**

```bash
rtk uv run pytest tests/test_factor_smooth_feature.py tests/test_factor_smooth_matrix.py tests/test_factor_smooth_discrete.py tests/test_factor_smooth_penalties.py tests/test_factor_smooth_reml.py tests/test_factor_smooth_inference.py tests/test_factor_smooth_mgcv_parity.py tests/test_factor_smooth_structured_system.py tests/test_factor_smooth_structured_parity.py tests/test_factor_smooth_sz_feature.py tests/test_factor_smooth_sz_matrix.py tests/test_factor_smooth_sz_penalties.py tests/test_sum_to_zero_structured_factor.py tests/test_factor_smooth_sz_reml.py tests/test_factor_smooth_sz_inference.py tests/test_factor_smooth_sz_mgcv_parity.py tests/test_random_effect_reml.py tests/test_structured_factor.py tests/test_structured_irls.py tests/test_structured_allocations.py -q
```

Expected: all pass.

- [ ] **Step 2: Run full repository gates**

```bash
rtk uv run pytest tests/ -q
rtk uv run ruff check src/ tests/ benchmarks/
rtk uv run ruff format --check src/ tests/ benchmarks/
rtk uv run mypy src/
rtk uv run mkdocs build --strict
rtk uv run python run_test.py
```

Expected: all commands exit zero. If mypy has a documented pre-existing
baseline, record the exact unchanged diagnostics and prove no new diagnostic
touches changed files.

- [ ] **Step 3: Audit every objective requirement against evidence**

Check:

```bash
rtk git diff --name-only origin/master...HEAD
rtk git diff --stat origin/master...HEAD
rtk grep -n 'basis=\"sz\"' src tests docs benchmarks
rtk grep -n 'FutureWarning' src/superglm/model tests/test_api.py
rtk grep -n 'toarray\\|kron' tests/test_factor_smooth_sz_reml.py tests/test_structured_allocations.py
```

Map each requirement to its passing test, fixture metadata, profile artifact,
or documentation section. Confirm:

- FS default and existing RE/FS parity;
- SZ global spline rule and duplicate geometry errors;
- exact pointwise sum-to-zero;
- one shared wiggle lambda;
- exact/discrete compact structured solving;
- dense/structured and mgcv parity;
- prediction/reporting semantics;
- shorthand warning and canonical explicit docs;
- profile/crossover evidence.

- [ ] **Step 4: Prove no LSS change**

```bash
rtk git diff --name-only origin/master...HEAD | rtk grep -i 'lss' || true
rtk git diff origin/master...HEAD | rtk grep -i 'lss' || true
```

Expected: no output. Inspect any match manually and remove the accidental LSS
change before claiming completion.

- [ ] **Step 5: Run final diff hygiene and commit any verification-only repair**

```bash
rtk proxy git diff --check origin/master...HEAD
rtk git status
```

Expected: no whitespace errors and a clean worktree. If verification required
a repair, rerun the affected focused and full gates before committing it with
an imperative subsystem-scoped subject.

- [ ] **Step 6: Request final code review**

Invoke `superpowers:requesting-code-review`, provide the design spec, this
plan, the exact base SHA recorded in Task 0, final SHA, test/profile evidence,
and the explicit no-LSS constraint. Address only technically validated
findings, rerun relevant gates, and leave the branch clean.
