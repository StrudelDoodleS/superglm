# Tweedie Compound Poisson–Gamma Generator Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `generate_tweedie_cpg()` reject invalid or numerically unsafe simulations, formally support exact vector `phi`, preserve ordinary seeded output exactly, and never turn a positive-claim Gamma underflow into a false Tweedie structural zero.

**Architecture:** Keep the public function and CPG formulas in `profiling/tweedie.py`, with small private normalizers and sampler-output validators immediately above it. Add one focused generator test module rather than enlarging the 3,500-line profile-likelihood suite. Preserve the existing Poisson/Gamma call signatures and direct arithmetic order on valid ordinary inputs; validation surrounds those operations without changing their RNG stream.

**Tech Stack:** Python 3.10+, NumPy `Generator`, NumPy typing, pytest, Ruff, mypy, uv.

---

## File map

- Create `tests/test_tweedie_generator.py`: focused public-contract, RNG, numerical-boundary, and weighted-vector characterization tests.
- Modify `src/superglm/profiling/tweedie.py`: input normalization, exact CPG parameter safety, draw validation, error context, and corrected public documentation.
- Do not modify public exports, notebooks, diagnostics, profile likelihood, or the existing broad test layout. Existing callers already satisfy the approved contract.

## Task 1: Lock the public input and reproducibility contract

**Files:**
- Create: `tests/test_tweedie_generator.py`
- Modify: `src/superglm/profiling/tweedie.py:23-96`

- [ ] **Step 1: Create the recording RNG and frozen legacy oracle**

Start `tests/test_tweedie_generator.py` with reusable test support. The frozen helper must retain the pre-hardening arithmetic and calls verbatim so the test detects both changed samples and changed RNG consumption.

```python
"""Focused tests for compound Poisson-Gamma Tweedie simulation."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

import superglm.profiling.tweedie as tweedie_module
from superglm import generate_tweedie_cpg


class _RecordingRNG:
    def __init__(
        self,
        counts,
        gamma_values=(),
        *,
        poisson_error: Exception | None = None,
        gamma_error: Exception | None = None,
    ):
        self.counts = counts
        self.gamma_values = gamma_values
        self.poisson_error = poisson_error
        self.gamma_error = gamma_error
        self.poisson_calls: list[np.ndarray] = []
        self.gamma_calls: list[tuple[np.ndarray, np.ndarray]] = []

    def poisson(self, lam):
        self.poisson_calls.append(np.array(lam, copy=True))
        if self.poisson_error is not None:
            raise self.poisson_error
        return np.array(self.counts, copy=True)

    def gamma(self, shape, scale):
        self.gamma_calls.append(
            (np.array(shape, copy=True), np.array(scale, copy=True))
        )
        if self.gamma_error is not None:
            raise self.gamma_error
        return np.array(self.gamma_values, copy=True)


def _legacy_generate_tweedie_cpg(n, mu, phi, p, rng):
    mu = np.broadcast_to(np.asarray(mu, dtype=np.float64), (n,)).copy()
    lam = np.power(mu, 2 - p) / ((2 - p) * phi)
    alpha = (2 - p) / (p - 1)
    beta = phi * (p - 1) * np.power(mu, p - 1)
    counts = rng.poisson(lam)
    y = np.zeros(n, dtype=np.float64)
    positive = counts > 0
    if np.any(positive):
        y[positive] = rng.gamma(alpha * counts[positive], scale=beta[positive])
    return y
```

- [ ] **Step 2: Add valid-call, exact-formula, and zero-length tests**

```python
def test_scalar_call_is_bitwise_legacy_compatible_and_consumes_same_rng_stream():
    legacy_rng = np.random.default_rng(271828)
    hardened_rng = np.random.default_rng(271828)

    expected = _legacy_generate_tweedie_cpg(
        128, mu=2.75, phi=0.8, p=1.45, rng=legacy_rng
    )
    actual = generate_tweedie_cpg(
        128, mu=2.75, phi=0.8, p=1.45, rng=hardened_rng
    )

    np.testing.assert_array_equal(actual, expected)
    assert pickle.dumps(hardened_rng.bit_generator.state) == pickle.dumps(
        legacy_rng.bit_generator.state
    )


def test_exact_vector_mu_and_phi_use_per_observation_cpg_parameters():
    rng = _RecordingRNG(counts=[0, 2, 1], gamma_values=[7.0, 11.0])

    y = generate_tweedie_cpg(
        3,
        mu=np.array([1.0, 4.0, 9.0]),
        phi=np.array([0.5, 1.0, 2.0]),
        p=1.5,
        rng=rng,
    )

    np.testing.assert_array_equal(rng.poisson_calls[0], [4.0, 4.0, 3.0])
    shape, scale = rng.gamma_calls[0]
    np.testing.assert_array_equal(shape, [2.0, 1.0])
    np.testing.assert_array_equal(scale, [1.0, 3.0])
    np.testing.assert_array_equal(y, [0.0, 7.0, 11.0])
    assert y.dtype == np.float64
    assert y.shape == (3,)
    assert not np.shares_memory(y, np.asarray(rng.gamma_values))


@pytest.mark.parametrize(
    ("mu", "phi"),
    [
        (2.0, 0.8),
        (np.array(2.0), np.array(0.8)),
        (np.empty(0), np.empty(0)),
    ],
)
def test_zero_length_returns_fresh_float64_array_without_draws(mu, phi):
    rng = _RecordingRNG(counts=[])

    first = generate_tweedie_cpg(0, mu=mu, phi=phi, p=1.5, rng=rng)
    second = generate_tweedie_cpg(0, mu=mu, phi=phi, p=1.5, rng=rng)

    assert first.shape == second.shape == (0,)
    assert first.dtype == second.dtype == np.float64
    assert first is not second
    assert rng.poisson_calls == []
    assert rng.gamma_calls == []
```

- [ ] **Step 3: Add pre-draw validation tests**

```python
@pytest.mark.parametrize(
    ("n", "error"),
    [(True, TypeError), (np.bool_(False), TypeError), (2.0, TypeError), (-1, ValueError)],
)
def test_invalid_n_is_rejected_before_rng_use(n, error):
    rng = _RecordingRNG(counts=[])
    with pytest.raises(error, match="n"):
        generate_tweedie_cpg(n, mu=1.0, phi=1.0, p=1.5, rng=rng)
    assert rng.poisson_calls == []


def test_numpy_integer_n_is_accepted():
    rng = _RecordingRNG(counts=[0, 0])
    y = generate_tweedie_cpg(np.int64(2), mu=1.0, phi=1.0, p=1.5, rng=rng)
    np.testing.assert_array_equal(y, [0.0, 0.0])


@pytest.mark.parametrize(
    "p",
    [True, 1.0, 2.0, np.nan, np.inf, 1.5 + 0j, "1.5", np.array([1.5])],
)
def test_invalid_p_is_rejected_before_rng_use(p):
    rng = _RecordingRNG(counts=[])
    with pytest.raises(ValueError, match="p"):
        generate_tweedie_cpg(2, mu=1.0, phi=1.0, p=p, rng=rng)
    assert rng.poisson_calls == []


@pytest.mark.parametrize("name", ["mu", "phi"])
@pytest.mark.parametrize(
    "value",
    [
        True,
        0.0,
        -1.0,
        np.nan,
        np.inf,
        1.0 + 0j,
        "1.0",
        np.array([1.0]),
        np.ones((2, 1)),
        np.ones((1, 2)),
    ],
)
def test_invalid_mu_or_phi_is_rejected_before_rng_use(name, value):
    rng = _RecordingRNG(counts=[])
    arguments = {"mu": np.ones(2), "phi": np.ones(2)}
    arguments[name] = value
    with pytest.raises(ValueError, match=name):
        generate_tweedie_cpg(2, p=1.5, rng=rng, **arguments)
    assert rng.poisson_calls == []


@pytest.mark.parametrize("n", [0, 1])
def test_rng_requires_callable_poisson_and_gamma_methods_even_without_draws(n):
    with pytest.raises(TypeError, match="rng"):
        generate_tweedie_cpg(n, mu=1.0, phi=1.0, p=1.5, rng=object())


def test_zero_length_still_validates_scalar_domains_before_returning():
    rng = _RecordingRNG(counts=[])
    with pytest.raises(ValueError, match="mu"):
        generate_tweedie_cpg(0, mu=0.0, phi=1.0, p=1.5, rng=rng)
    assert rng.poisson_calls == []


class _WrongSignatureRNG:
    def poisson(self):
        return np.array([0])

    def gamma(self, shape, scale):
        return np.array([1.0])


def test_incompatible_rng_method_signature_raises_type_error():
    with pytest.raises(TypeError, match="poisson"):
        generate_tweedie_cpg(
            1, mu=1.0, phi=1.0, p=1.5, rng=_WrongSignatureRNG()
        )
```

- [ ] **Step 4: Run the contract tests and confirm RED**

Run:

```bash
rtk uv run pytest tests/test_tweedie_generator.py -q
```

Expected: the exact-vector formula test already passes through incidental broadcasting,
while zero-length calls record a Poisson draw and invalid inputs either pass through or
raise inconsistent NumPy errors. The passing vector test protects behavior while the
contract is formalized.

- [ ] **Step 5: Implement strict public normalization without changing ordinary arithmetic**

Add `import operator` with the standard-library imports in `profiling/tweedie.py`, then add these helpers above the public generator:

```python
def _normalize_cpg_size(n: int) -> int:
    if isinstance(n, (bool, np.bool_)):
        raise TypeError("n must be a non-negative integer, not a boolean")
    try:
        size = operator.index(n)
    except TypeError as exc:
        raise TypeError("n must be a non-negative integer") from exc
    if size < 0:
        raise ValueError("n must be a non-negative integer")
    return size


def _normalize_cpg_power(p: float) -> float:
    raw = np.asarray(p)
    if (
        raw.ndim != 0
        or np.issubdtype(raw.dtype, np.bool_)
        or not np.issubdtype(raw.dtype, np.number)
        or np.iscomplexobj(raw)
    ):
        raise ValueError("p must be a finite real scalar in the open interval (1, 2)")
    power = float(raw)
    if not np.isfinite(power) or not 1.0 < power < 2.0:
        raise ValueError("p must be a finite real scalar in the open interval (1, 2)")
    return power


def _normalize_cpg_parameter(name: str, value: float | NDArray, n: int) -> NDArray:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be a finite, strictly positive real scalar or array "
            f"with shape ({n},)"
        ) from exc
    if (
        np.issubdtype(raw.dtype, np.bool_)
        or not np.issubdtype(raw.dtype, np.number)
        or np.iscomplexobj(raw)
    ):
        raise ValueError(
            f"{name} must be a finite, strictly positive real scalar or array "
            f"with shape ({n},)"
        )
    if raw.ndim == 0:
        normalized = np.full(n, float(raw), dtype=np.float64)
    elif raw.ndim == 1 and raw.shape == (n,):
        normalized = np.array(raw, dtype=np.float64, copy=True)
    else:
        raise ValueError(f"{name} must be a scalar or an array with shape ({n},)")
    if not np.all(np.isfinite(normalized)) or np.any(normalized <= 0.0):
        raise ValueError(f"{name} must contain only finite, strictly positive values")
    return normalized


def _resolve_cpg_rng(rng: np.random.Generator | None):
    resolved = np.random.default_rng() if rng is None else rng
    if not callable(getattr(resolved, "poisson", None)) or not callable(
        getattr(resolved, "gamma", None)
    ):
        raise TypeError("rng must provide callable poisson and gamma methods")
    return resolved
```

Change the public signature to `phi: float | NDArray`, document scalar or exact `(n,)` `mu` and `phi`, document all parameter domains, and replace the function body with this first green version:

```python
size = _normalize_cpg_size(n)
power = _normalize_cpg_power(p)
mu_array = _normalize_cpg_parameter("mu", mu, size)
phi_array = _normalize_cpg_parameter("phi", phi, size)
resolved_rng = _resolve_cpg_rng(rng)

with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
    lam = np.power(mu_array, 2.0 - power) / ((2.0 - power) * phi_array)
    alpha = (2.0 - power) / (power - 1.0)
    beta = phi_array * (power - 1.0) * np.power(mu_array, power - 1.0)

if size == 0:
    return np.empty(0, dtype=np.float64)

counts = resolved_rng.poisson(lam)
y = np.zeros(size, dtype=np.float64)
positive = counts > 0
if np.any(positive):
    y[positive] = resolved_rng.gamma(
        alpha * counts[positive], scale=beta[positive]
    )
return y
```

- [ ] **Step 6: Run focused tests, format, and confirm GREEN**

```bash
rtk uv run ruff format src/superglm/profiling/tweedie.py tests/test_tweedie_generator.py
rtk uv run pytest tests/test_tweedie_generator.py -q
rtk uv run pytest tests/test_tweedie_profile.py -q -k "TestGenerateTweedieCPG"
```

Expected: all new contract tests and both existing generator tests pass. The frozen oracle confirms bitwise values and identical RNG state.

- [ ] **Step 7: Self-review and commit Task 1**

Check that invalid inputs and `n == 0` make no sampler calls, boolean/numeric-string/complex values are rejected before float conversion, and exact shape `(n,)` is the only vector form.

```bash
rtk git diff --check
rtk git add src/superglm/profiling/tweedie.py tests/test_tweedie_generator.py
rtk git commit -m "Validate Tweedie generator inputs"
```

## Task 2: Reject unsafe parameters and malformed or underflowed draws

**Files:**
- Modify: `src/superglm/profiling/tweedie.py`
- Modify: `tests/test_tweedie_generator.py`

- [ ] **Step 1: Add derived-parameter and exact Poisson-boundary tests**

```python
@pytest.mark.parametrize(
    ("mu", "phi", "message"),
    [
        (np.finfo(np.float64).tiny, np.finfo(np.float64).max, "Poisson rate"),
        (np.finfo(np.float64).max, np.finfo(np.float64).max, "Gamma scale"),
    ],
)
def test_unrepresentable_derived_parameters_fail_before_draw(mu, phi, message):
    rng = _RecordingRNG(counts=[])
    with pytest.raises(ValueError, match=message):
        generate_tweedie_cpg(1, mu=mu, phi=phi, p=1.5, rng=rng)
    assert rng.poisson_calls == []


def test_exact_numpy_poisson_limit_is_accepted_but_next_rate_is_rejected():
    limit = float(np.iinfo(np.int64).max) - 10.0 * np.sqrt(
        float(np.iinfo(np.int64).max)
    )
    endpoint_phi = 2.0 / limit
    accepted_rng = _RecordingRNG(counts=[0])
    generate_tweedie_cpg(1, mu=1.0, phi=endpoint_phi, p=1.5, rng=accepted_rng)
    assert accepted_rng.poisson_calls[0][0] == limit

    rejected_rng = _RecordingRNG(counts=[0])
    with pytest.raises(ValueError, match="Poisson rate"):
        generate_tweedie_cpg(
            1,
            mu=1.0,
            phi=np.nextafter(endpoint_phi, 0.0),
            p=1.5,
            rng=rejected_rng,
        )
    assert rejected_rng.poisson_calls == []


def test_unrepresentable_realized_gamma_shape_stops_before_gamma(monkeypatch):
    monkeypatch.setattr(
        tweedie_module,
        "_prepare_cpg_parameters",
        lambda mu, phi, p: (
            np.ones_like(mu),
            np.finfo(np.float64).max,
            np.ones_like(phi),
        ),
    )
    rng = _RecordingRNG(counts=[2], gamma_values=[1.0])
    with pytest.raises(ValueError, match="Gamma shape"):
        generate_tweedie_cpg(1, mu=1.0, phi=1.0, p=1.5, rng=rng)
    assert rng.gamma_calls == []
```

- [ ] **Step 2: Add sampler exception and malformed-output tests**

```python
def test_poisson_sampler_value_error_is_wrapped_with_original_cause():
    cause = ValueError("sampler rejected lambda")
    rng = _RecordingRNG(counts=[], poisson_error=cause)
    with pytest.raises(ValueError, match="Poisson sampler") as error:
        generate_tweedie_cpg(1, mu=1.0, phi=1.0, p=1.5, rng=rng)
    assert error.value.__cause__ is cause
    assert rng.gamma_calls == []


def test_gamma_sampler_value_error_is_wrapped_with_original_cause():
    cause = ValueError("sampler rejected shape")
    rng = _RecordingRNG(counts=[1], gamma_error=cause)
    with pytest.raises(ValueError, match="Gamma sampler") as error:
        generate_tweedie_cpg(1, mu=1.0, phi=1.0, p=1.5, rng=rng)
    assert error.value.__cause__ is cause


class _WrongGammaSignatureRNG:
    def poisson(self, lam):
        return np.ones(np.shape(lam), dtype=np.int64)

    def gamma(self):
        return np.array([1.0])


def test_incompatible_gamma_signature_raises_type_error():
    with pytest.raises(TypeError, match="gamma"):
        generate_tweedie_cpg(
            1, mu=1.0, phi=1.0, p=1.5, rng=_WrongGammaSignatureRNG()
        )


@pytest.mark.parametrize(
    ("counts", "message"),
    [
        (1, "shape"),
        ([[1]], "shape"),
        ([1.0], "integer"),
        ([-1], "non-negative"),
        ([True], "integer"),
        (np.array([np.iinfo(np.uint64).max], dtype=np.uint64), "int64"),
    ],
)
def test_malformed_poisson_output_is_rejected_before_gamma(counts, message):
    rng = _RecordingRNG(counts=counts, gamma_values=[1.0])
    with pytest.raises(RuntimeError, match=message):
        generate_tweedie_cpg(1, mu=1.0, phi=1.0, p=1.5, rng=rng)
    assert rng.gamma_calls == []


@pytest.mark.parametrize(
    ("gamma_values", "error", "message"),
    [
        (1.0, RuntimeError, "shape"),
        ([[1.0]], RuntimeError, "shape"),
        ([1.0 + 0j], RuntimeError, "real"),
        (["1.0"], RuntimeError, "real"),
        ([True], RuntimeError, "real"),
        ([-1.0], RuntimeError, "negative"),
        ([0.0], ValueError, "underflow"),
        ([np.nan], ValueError, "finite"),
        ([np.inf], ValueError, "finite"),
    ],
)
def test_invalid_gamma_output_is_rejected(gamma_values, error, message):
    rng = _RecordingRNG(counts=[1], gamma_values=gamma_values)
    with pytest.raises(error, match=message):
        generate_tweedie_cpg(1, mu=1.0, phi=1.0, p=1.5, rng=rng)
```

- [ ] **Step 3: Add genuine near-boundary underflow and overflow regressions**

These are not fake-RNG tests. With `lambda=1` and the closest float below `p=2`,
NumPy 1.24 through 2.4 returns exact zero for essentially every positive-count Gamma
draw. The old generator silently reports a zero fraction near 1 instead of `exp(-1)`.
At the opposite boundary, finite parameters can still produce an infinite realization
because Gamma support is unbounded; the generator must report that without clipping or
resampling.

```python
def test_near_two_gamma_underflow_cannot_become_structural_zero():
    power = np.nextafter(2.0, 1.0)
    phi = 1.0 / (2.0 - power)
    rng = np.random.default_rng(20260717)

    with pytest.raises(ValueError, match="underflow"):
        generate_tweedie_cpg(10_000, mu=1.0, phi=phi, p=power, rng=rng)


def test_near_one_realized_gamma_overflow_is_reported_without_clipping():
    power = np.nextafter(1.0, 2.0)
    mu = 0.75 * np.finfo(np.float64).max
    phi = mu ** (2.0 - power) / (2.0 - power)
    rng = np.random.default_rng(20260717)

    with pytest.raises(ValueError, match="finite"):
        generate_tweedie_cpg(10_000, mu=mu, phi=phi, p=power, rng=rng)
```

- [ ] **Step 4: Run the numerical/output tests and confirm RED**

```bash
rtk uv run pytest tests/test_tweedie_generator.py -q -k "unrepresentable or poisson_limit or sampler or malformed or gamma_output or near_two or near_one"
```

Expected: current code emits floating warnings, delegates the boundary to NumPy, broadcasts malformed outputs, and accepts Gamma zeros as structural zeros.

- [ ] **Step 5: Implement exact parameter and sampler validation**

Add the constant and helpers above `generate_tweedie_cpg()`. Keep the direct `beta = phi * (p - 1) * mu**(p - 1)` order: under the joint CPG/Poisson constraints, a directly underflowed `beta` cannot be rescued into a valid NumPy-safe simulation, while a log-domain recomputation can falsely turn true overflow into a finite value.

```python
_POISSON_LAM_MAX = float(np.iinfo(np.int64).max) - 10.0 * np.sqrt(
    float(np.iinfo(np.int64).max)
)


def _prepare_cpg_parameters(
    mu: NDArray, phi: NDArray, p: float
) -> tuple[NDArray, float, NDArray]:
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        lam = np.power(mu, 2.0 - p) / ((2.0 - p) * phi)
        alpha = (2.0 - p) / (p - 1.0)
        beta = phi * (p - 1.0) * np.power(mu, p - 1.0)
    if not np.all(np.isfinite(lam)) or np.any(lam <= 0.0):
        raise ValueError("derived Poisson rate must be finite and strictly positive")
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("derived Gamma shape must be finite and strictly positive")
    if not np.all(np.isfinite(beta)) or np.any(beta <= 0.0):
        raise ValueError("derived Gamma scale must be finite and strictly positive")
    if np.any(lam > _POISSON_LAM_MAX):
        raise ValueError("derived Poisson rate exceeds NumPy's safe int64 limit")
    return lam, alpha, beta


def _draw_cpg_counts(rng, lam: NDArray) -> NDArray:
    try:
        raw = rng.poisson(lam)
    except TypeError as exc:
        raise TypeError("rng poisson method has an incompatible signature") from exc
    except (ValueError, OverflowError, FloatingPointError) as exc:
        raise ValueError("Poisson sampler rejected the validated CPG rate") from exc
    try:
        counts = np.asarray(raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Poisson sampler returned an invalid result") from exc
    if counts.shape != lam.shape:
        raise RuntimeError(f"Poisson sampler result must have shape {lam.shape}")
    if np.issubdtype(counts.dtype, np.bool_) or not np.issubdtype(
        counts.dtype, np.integer
    ):
        raise RuntimeError("Poisson sampler result must have an integer dtype")
    if np.any(counts < 0):
        raise RuntimeError("Poisson sampler result must be non-negative")
    if np.any(counts > np.iinfo(np.int64).max):
        raise RuntimeError("Poisson sampler result exceeds the signed int64 limit")
    return np.array(counts, dtype=np.int64, copy=True)


def _draw_cpg_positive_values(
    rng,
    counts: NDArray,
    positive: NDArray,
    alpha: float,
    beta: NDArray,
) -> NDArray:
    with np.errstate(over="ignore", invalid="ignore"):
        shapes = alpha * counts[positive]
    if not np.all(np.isfinite(shapes)) or np.any(shapes <= 0.0):
        raise ValueError("derived positive-event Gamma shape is not representable")
    try:
        raw = rng.gamma(shapes, scale=beta[positive])
    except TypeError as exc:
        raise TypeError("rng gamma method has an incompatible signature") from exc
    except (ValueError, OverflowError, FloatingPointError) as exc:
        raise ValueError("Gamma sampler rejected the validated CPG parameters") from exc
    try:
        values = np.asarray(raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Gamma sampler returned an invalid result") from exc
    expected_shape = (int(np.count_nonzero(positive)),)
    if values.shape != expected_shape:
        raise RuntimeError(f"Gamma sampler result must have shape {expected_shape}")
    if (
        np.issubdtype(values.dtype, np.bool_)
        or not np.issubdtype(values.dtype, np.number)
        or np.iscomplexobj(values)
    ):
        raise RuntimeError("Gamma sampler result must contain real numeric values")
    with np.errstate(over="ignore", invalid="ignore"):
        values = np.array(values, dtype=np.float64, copy=True)
    if np.any(values < 0.0):
        raise RuntimeError("Gamma sampler result cannot contain negative values")
    if not np.all(np.isfinite(values)):
        raise ValueError("Gamma sampler result must be finite; output overflowed")
    if np.any(values == 0.0):
        raise ValueError(
            "positive-count Gamma draw underflowed to zero; structural-zero mass "
            "would be incorrect"
        )
    return values
```

Replace the first green function's sampling section with:

```python
lam, alpha, beta = _prepare_cpg_parameters(mu_array, phi_array, power)
if size == 0:
    return np.empty(0, dtype=np.float64)

counts = _draw_cpg_counts(resolved_rng, lam)
y = np.zeros(size, dtype=np.float64)
positive = counts > 0
if np.any(positive):
    y[positive] = _draw_cpg_positive_values(
        resolved_rng, counts, positive, alpha, beta
    )
if y.shape != (size,) or y.dtype != np.float64:
    raise RuntimeError("internal CPG output shape or dtype invariant failed")
if not np.all(np.isfinite(y)) or np.any(y < 0.0):
    raise ValueError("generated Tweedie values must be finite and non-negative")
return y
```

- [ ] **Step 6: Run focused and inherited boundary tests and confirm GREEN**

```bash
rtk uv run ruff format src/superglm/profiling/tweedie.py tests/test_tweedie_generator.py
rtk uv run pytest tests/test_tweedie_generator.py -q
rtk uv run pytest tests/test_tweedie_profile.py tests/test_tweedie_profile_performance.py -q -k "GenerateTweedieCPG or generator_moments or generator_boundary"
```

Expected: all pass; the genuine near-2 call raises instead of returning a catastrophically false all-zero sample.

- [ ] **Step 7: Self-review and commit Task 2**

Confirm the Poisson endpoint itself remains accepted, no Gamma draw follows invalid counts, raw Gamma values are validated before assignment, and zero is allowed only where `N == 0`.

```bash
rtk git diff --check
rtk git add src/superglm/profiling/tweedie.py tests/test_tweedie_generator.py
rtk git commit -m "Harden Tweedie generator sampling"
```

## Task 3: Characterize prior-weight vector dispersion without bloating the suite

**Files:**
- Modify: `tests/test_tweedie_generator.py`

- [ ] **Step 1: Add one vectorized `phi / weight` distribution characterization**

Use four groups in one vectorized call. This covers the official-notebook behavior without duplicating the existing scalar `p=1.05`, `p=1.95`, and insurance examples.

```python
def test_vector_phi_over_prior_weights_has_expected_group_distributions():
    n_per_group = 20_000
    weights = np.repeat(np.array([0.5, 1.0, 2.0, 4.0]), n_per_group)
    mu = 3.0
    base_phi = 1.2
    power = 1.5
    phi = base_phi / weights

    y = generate_tweedie_cpg(
        len(weights),
        mu=mu,
        phi=phi,
        p=power,
        rng=np.random.default_rng(8675309),
    )

    for weight in np.unique(weights):
        group = y[weights == weight]
        effective_phi = base_phi / weight
        expected_variance = effective_phi * mu**power
        lam = mu ** (2.0 - power) / ((2.0 - power) * effective_phi)
        np.testing.assert_allclose(np.mean(group), mu, rtol=0.04)
        np.testing.assert_allclose(np.var(group), expected_variance, rtol=0.12)
        np.testing.assert_allclose(np.mean(group == 0.0), np.exp(-lam), atol=0.015)
```

- [ ] **Step 2: Run the statistical characterization repeatedly**

```bash
rtk uv run pytest tests/test_tweedie_generator.py::test_vector_phi_over_prior_weights_has_expected_group_distributions -q
rtk uv run pytest tests/test_tweedie_generator.py::test_vector_phi_over_prior_weights_has_expected_group_distributions -q
rtk uv run pytest tests/test_tweedie_generator.py::test_vector_phi_over_prior_weights_has_expected_group_distributions -q
```

Expected: three deterministic passes with the fixed seed; runtime remains small enough that no `slow` marker is needed.

- [ ] **Step 3: Run all direct generator characterizations**

```bash
rtk uv run pytest tests/test_tweedie_generator.py tests/test_tweedie_profile.py tests/test_tweedie_profile_performance.py -q -k "tweedie_cpg or GenerateTweedieCPG or generator_moments or generator_boundary"
```

- [ ] **Step 4: Commit the vector-dispersion characterization**

```bash
rtk git diff --check
rtk git add tests/test_tweedie_generator.py
rtk git commit -m "Characterize vector Tweedie dispersion"
```

## Task 4: Independent review and branch-wide verification

**Files:**
- Review: `src/superglm/profiling/tweedie.py`
- Review: `tests/test_tweedie_generator.py`
- Review: all branch changes relative to `origin/master`

- [ ] **Step 1: Run focused static checks**

```bash
rtk uv run ruff check src/superglm/profiling/tweedie.py tests/test_tweedie_generator.py
rtk uv run ruff format --check src/superglm/profiling/tweedie.py tests/test_tweedie_generator.py
rtk uv run mypy --no-incremental --follow-imports=skip src/superglm/profiling/tweedie.py
```

Expected: Ruff passes. Record any mypy diagnostics and compare them with the same command on `origin/master`; this branch must introduce none.

- [ ] **Step 2: Dispatch independent spec and quality reviews**

Give one fresh reviewer the approved design, commits from Tasks 1–3, and the exact test commands. Require review of:

- structural zeros versus positive-claim underflow;
- exact Poisson threshold behavior;
- direct-formula/RNG compatibility;
- error taxonomy and pre-draw state safety;
- vector `phi / weight` convention;
- excessive or redundant tests.

Fix every confirmed finding with a failing regression first, rerun focused tests, and commit the fix separately.

- [ ] **Step 3: Run all non-slow tests**

```bash
rtk uv run pytest tests/ -q -m "not slow"
```

Expected: zero failures.

- [ ] **Step 4: Run the complete suite on the final HEAD**

```bash
rtk uv run pytest tests/ -q
```

Expected: zero failures. Record collected, passed, and skipped counts plus elapsed time.

- [ ] **Step 5: Audit final provenance and cleanliness**

```bash
rtk git merge-base HEAD origin/master
rtk git rev-parse origin/master
rtk git status --short --untracked-files=all
rtk git log --oneline origin/master..HEAD
rtk git diff --check origin/master...HEAD
rtk git diff --stat origin/master...HEAD
```

Expected: merge base equals the recorded `origin/master` base, worktree is clean, every intended commit is present, and no generated/debug artifact is tracked.

- [ ] **Step 6: Prepare publication handoff**

Summarize the corrected generator contract, the near-`p=2` false-zero diagnosis, exact compatibility evidence, focused/full-suite results, remaining inherited mypy baseline, and the already measured profile-MLE performance facts. Do not push or open a PR until the user chooses the publication option.
