# Tweedie Numerical Kernel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace cancellation-prone and approximation-selecting Tweedie mathematics with independently derived, certified exact numerical primitives and neutral reference tests.

**Architecture:** Add two private top-level modules so distributions and profiling share one implementation without circular imports. `_tweedie_numerics.py` owns validation, unit deviance, and Pearson dispersion; `_tweedie_density.py` owns the positive compound-Poisson/Gamma log series, score, certification diagnostics, and typed failures. Existing public functions become compatibility wrappers around this kernel.

**Tech Stack:** Python 3.10+, NumPy, SciPy (`gammaln`, `logsumexp`), pytest, JSON reference fixtures.

---

## File map

- Create `src/superglm/_tweedie_numerics.py`: shared scalar/array validation, stable unit deviance, Pearson dispersion, and constants.
- Create `src/superglm/_tweedie_density.py`: exact series evaluator, term-ratio tail certification, score, diagnostics, and explicit approximate evaluator.
- Modify `src/superglm/distributions.py`: strict `Tweedie.p` construction and shared deviance/log-likelihood calls.
- Modify `src/superglm/profiling/tweedie.py`: generator validation reuse, compatibility wrappers, dispersion profiler integration, and removal of silent approximation selection.
- Create `tests/test_tweedie_numerics.py`: mathematical and validation unit tests.
- Create `tests/test_tweedie_density.py`: exact series, tail, score, and error-path tests.
- Create `tests/test_tweedie_reference.py`: neutral fixture and deterministic profile comparisons.
- Create `tests/fixtures/tweedie_reference_values.json`: black-box input/output values only.
- Create `scripts/check_tweedie_reference.py`: version-neutral fixture comparison command; it never invokes or names an external package.

### Task 1: Introduce shared scalar validation and stable unit deviance

**Files:**
- Create: `src/superglm/_tweedie_numerics.py`
- Create: `tests/test_tweedie_numerics.py`
- Modify: `src/superglm/profiling/tweedie.py:75-151`
- Modify: `tests/test_tweedie_generator.py`

- [ ] **Step 1: Write failing scalar and deviance regressions**

```python
import numpy as np
import pytest

from superglm._tweedie_numerics import (
    compound_poisson_gamma_parameters,
    normalize_tweedie_power,
    tweedie_unit_deviance,
)


@pytest.mark.parametrize(
    ("value", "exception"),
    [
        (True, TypeError),
        (np.bool_(False), TypeError),
        (np.array([1.5]), TypeError),
        (object(), TypeError),
        (np.nan, ValueError),
    ],
)
def test_normalize_tweedie_power_rejects_non_real_scalar(value, exception):
    with pytest.raises(exception):
        normalize_tweedie_power(value)


@pytest.mark.parametrize("p", [1.000001, 1.01, 1.1, 1.5, 1.99, 1.999999])
def test_unit_deviance_is_exactly_zero_at_equal_values(p):
    values = np.array([1e-20, 1.0, 1e12])
    result = tweedie_unit_deviance(values, values.copy(), p)
    np.testing.assert_array_equal(result, np.zeros_like(values))


def test_unit_deviance_matches_stable_closed_form_at_p_one_half():
    mu = np.array([1e-12, 1.0, 1e12])
    y = mu * np.array([1.0 + 1e-8, 4.0, 1e-8])
    delta = (y - mu) / mu
    root_difference = delta / (np.sqrt(1.0 + delta) + 1.0)
    expected = 4.0 * np.sqrt(mu) * root_difference**2
    np.testing.assert_allclose(
        tweedie_unit_deviance(y, mu, 1.5), expected, rtol=2e-12, atol=0.0
    )


def test_compound_parameters_recover_weighted_tweedie_moments():
    parameters = compound_poisson_gamma_parameters(
        np.array([4.0]), 0.5, 1.5, weights=np.array([2.0])
    )
    expected_variance = 0.5 * 4.0**1.5 / 2.0
    np.testing.assert_allclose(
        parameters.rate * parameters.shape * parameters.scale,
        [4.0],
    )
    np.testing.assert_allclose(
        parameters.rate
        * parameters.shape
        * (1.0 + parameters.shape)
        * parameters.scale**2,
        [expected_variance],
    )
```

- [ ] **Step 2: Run the tests and verify the new module is missing**

Run: `rtk pytest tests/test_tweedie_numerics.py`

Expected: collection fails with `ModuleNotFoundError: No module named 'superglm._tweedie_numerics'`.

- [ ] **Step 3: Implement scalar validation and the stable ratio formula**

Create these interfaces and use the displayed formulas exactly:

```python
from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

import numpy as np
from numpy.typing import NDArray

PHI_LOWER_BOUND = 1e-12


class TweedieNumericalError(RuntimeError):
    """Exact Tweedie arithmetic could not be represented or certified."""


@dataclass(frozen=True)
class CompoundPoissonGammaParameters:
    rate: NDArray[np.float64]
    shape: float
    scale: NDArray[np.float64]


def normalize_real_scalar(name: str, value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be one finite real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def normalize_tweedie_power(value: object) -> float:
    p = normalize_real_scalar("p", value)
    if not 1.0 < p < 2.0:
        raise ValueError(f"Tweedie p must be in (1, 2), got {p}")
    return p


def normalize_positive_scalar(name: str, value: object) -> float:
    result = normalize_real_scalar(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be strictly positive")
    return result


def compound_poisson_gamma_parameters(mu, phi, p, *, weights=None):
    power = normalize_tweedie_power(p)
    dispersion = normalize_positive_scalar("phi", phi)
    mu_arr = np.asarray(mu, dtype=np.float64)
    weight_arr = np.ones_like(mu_arr) if weights is None else np.asarray(weights, dtype=np.float64)
    if mu_arr.ndim != 1 or weight_arr.shape != mu_arr.shape:
        raise ValueError("mu and weights must be matching one-dimensional arrays")
    if np.any(~np.isfinite(mu_arr)) or np.any(mu_arr <= 0.0):
        raise ValueError("mu must be finite and strictly positive")
    if np.any(~np.isfinite(weight_arr)) or np.any(weight_arr <= 0.0):
        raise ValueError("weights must be finite and strictly positive")
    effective_phi = dispersion / weight_arr
    shape = (2.0 - power) / (power - 1.0)
    rate = mu_arr ** (2.0 - power) / (effective_phi * (2.0 - power))
    scale = effective_phi * (power - 1.0) * mu_arr ** (power - 1.0)
    return CompoundPoissonGammaParameters(rate=rate, shape=shape, scale=scale)


def _deviance_remainder(log_ratio: NDArray[np.float64], q: float) -> NDArray[np.float64]:
    out = np.expm1(q * log_ratio) - q * np.expm1(log_ratio)
    close = np.abs(log_ratio) <= 1e-3
    if np.any(close):
        z = log_ratio[close]
        term = z * z
        series = (q * q - q) * term / 2.0
        factorial = 2.0
        q_power = q * q
        for order in range(3, 13):
            factorial *= order
            term *= z
            q_power *= q
            series += (q_power - q) * term / factorial
        out[close] = series
    return out


def tweedie_unit_deviance(y: object, mu: object, p: object) -> NDArray[np.float64]:
    power = normalize_tweedie_power(p)
    y_arr, mu_arr = np.broadcast_arrays(
        np.asarray(y, dtype=np.float64), np.asarray(mu, dtype=np.float64)
    )
    if np.any(~np.isfinite(y_arr)) or np.any(y_arr < 0.0):
        raise ValueError("y must be finite and nonnegative")
    if np.any(~np.isfinite(mu_arr)) or np.any(mu_arr <= 0.0):
        raise ValueError("mu must be finite and strictly positive")

    q = 2.0 - power
    result = np.empty(y_arr.shape, dtype=np.float64)
    equal = y_arr == mu_arr
    zero = (y_arr == 0.0) & ~equal
    result[equal] = 0.0
    result[zero] = 2.0 * np.exp(q * np.log(mu_arr[zero])) / q

    positive = ~(equal | zero)
    if np.any(positive):
        log_ratio = np.log(y_arr[positive]) - np.log(mu_arr[positive])
        remainder = _deviance_remainder(log_ratio, q)
        scale = np.exp(q * np.log(mu_arr[positive]))
        values = 2.0 * scale * remainder / ((1.0 - power) * q)
        tolerance = 64.0 * np.finfo(np.float64).eps * np.maximum(scale, 1.0)
        if np.any(values < -tolerance):
            raise TweedieNumericalError("unit deviance became materially negative")
        result[positive] = np.maximum(values, 0.0)
    return result
```

- [ ] **Step 4: Run the numerical tests**

Run: `rtk pytest tests/test_tweedie_numerics.py tests/test_tweedie_generator.py`

Expected: scalar, deviance, compound-moment, and unchanged generator RNG-order tests pass. Route
`_prepare_cpg_parameters` through `compound_poisson_gamma_parameters` without changing the existing
single Poisson call followed by at most one Gamma call.

- [ ] **Step 5: Commit the new primitive**

```bash
rtk git add src/superglm/_tweedie_numerics.py src/superglm/profiling/tweedie.py tests/test_tweedie_numerics.py tests/test_tweedie_generator.py
rtk git commit -m "Stabilize Tweedie unit deviance"
```

### Task 2: Add stable Pearson dispersion and wire public distribution behavior

**Files:**
- Modify: `src/superglm/_tweedie_numerics.py`
- Modify: `src/superglm/distributions.py:268-316`
- Modify: `src/superglm/profiling/tweedie.py:75-89,815-849,1088-1102`
- Modify: `tests/test_tweedie_numerics.py`
- Modify: `tests/test_core.py:31-37`

- [ ] **Step 1: Add failing dispersion and constructor tests**

```python
@pytest.mark.parametrize("df_resid", [0, -1, np.nan, np.inf, True, "2", [2]])
def test_pearson_dispersion_rejects_invalid_residual_df(df_resid):
    from superglm.profiling.tweedie import estimate_phi

    exception = TypeError if isinstance(df_resid, (bool, str, list)) else ValueError
    with pytest.raises(exception, match="df_resid"):
        estimate_phi(np.array([1.0]), np.array([1.0]), 1.5, df_resid=df_resid)


def test_pearson_dispersion_preserves_valid_tiny_means():
    from superglm.profiling.tweedie import estimate_phi

    y = np.array([1e-12, 2e-12])
    mu = np.array([1e-20, 2e-20])
    expected = np.sum((y - mu) ** 2 / mu**1.5) / 2.0
    assert estimate_phi(y, mu, 1.5, df_resid=2.0) == pytest.approx(expected, rel=2e-13)


@pytest.mark.parametrize("p", [np.array([1.5]), np.array(1.5, dtype=object), True, "1.5"])
def test_tweedie_constructor_rejects_unsupported_power_objects(p):
    from superglm import Tweedie

    with pytest.raises(TypeError):
        Tweedie(p)
```

- [ ] **Step 2: Confirm the regressions fail against the current implementation**

Run: `rtk pytest tests/test_tweedie_numerics.py tests/test_core.py -k 'pearson_dispersion or constructor_rejects'`

Expected: invalid residual DF is accepted or fails inconsistently, tiny means are floored, and array/object powers are accepted.

- [ ] **Step 3: Implement log-scaled Pearson terms**

Add `pearson_dispersion` to `_tweedie_numerics.py`. Use `scipy.special.logsumexp` over nonzero residual terms, return `PHI_LOWER_BOUND` for an exact zero numerator, and raise `TweedieNumericalError` if the true result exceeds `float64` range.

```python
def pearson_dispersion(y, mu, p, weights, df_resid):
    from scipy.special import logsumexp

    power = normalize_tweedie_power(p)
    denominator = normalize_positive_scalar("df_resid", df_resid)
    y_arr = np.asarray(y, dtype=np.float64)
    mu_arr = np.asarray(mu, dtype=np.float64)
    w_arr = np.ones_like(y_arr) if weights is None else np.asarray(weights, dtype=np.float64)
    if y_arr.ndim != 1 or mu_arr.shape != y_arr.shape or w_arr.shape != y_arr.shape:
        raise ValueError("y, mu, and weights must be matching one-dimensional arrays")
    if np.any(~np.isfinite(y_arr)) or np.any(y_arr < 0.0):
        raise ValueError("y must be finite and nonnegative")
    if np.any(~np.isfinite(mu_arr)) or np.any(mu_arr <= 0.0):
        raise ValueError("mu must be finite and strictly positive")
    if np.any(~np.isfinite(w_arr)) or np.any(w_arr <= 0.0):
        raise ValueError("weights must be finite and strictly positive")

    residual = np.abs(y_arr - mu_arr)
    nonzero = residual > 0.0
    if not np.any(nonzero):
        return PHI_LOWER_BOUND
    log_terms = (
        np.log(w_arr[nonzero])
        + 2.0 * np.log(residual[nonzero])
        - power * np.log(mu_arr[nonzero])
    )
    log_phi = float(logsumexp(log_terms) - math.log(denominator))
    if log_phi > math.log(np.finfo(np.float64).max):
        raise TweedieNumericalError("Pearson dispersion exceeds float64 range")
    return max(float(math.exp(log_phi)), PHI_LOWER_BOUND)
```

- [ ] **Step 4: Replace duplicated public arithmetic**

Make `Tweedie.__init__` call `normalize_tweedie_power`, make `Tweedie.deviance_unit` call
`tweedie_unit_deviance`, and make both `estimate_phi` and `_pearson_phi_from_prepared` call
`pearson_dispersion`. Remove their `np.maximum(mu, 1e-10)` expressions and move
`_PHI_LOWER_BOUND` to the shared constant.

- [ ] **Step 5: Run affected tests and commit**

Run: `rtk pytest tests/test_tweedie_numerics.py tests/test_core.py tests/test_tweedie_profile.py -k 'deviance or estimate_phi or pearson or power'`

Expected: all selected tests pass.

```bash
rtk git add src/superglm/_tweedie_numerics.py src/superglm/distributions.py src/superglm/profiling/tweedie.py tests/test_tweedie_numerics.py tests/test_core.py
rtk git commit -m "Validate Tweedie dispersion inputs"
```

### Task 3: Implement the certified compound-Poisson density series

**Files:**
- Create: `src/superglm/_tweedie_density.py`
- Create: `tests/test_tweedie_density.py`

- [ ] **Step 1: Write failing exact-series and score tests**

```python
import numpy as np
import pytest

from superglm._tweedie_density import evaluate_tweedie_density


def test_positive_density_matches_hand_summed_shape_one_case():
    result = evaluate_tweedie_density(
        np.array([1.0]), np.array([1.0]), 2.0, 1.5
    )
    assert result.logpdf[0] == pytest.approx(-1.5358655264538403, abs=2e-15)
    assert result.diagnostics.certified


def test_exact_series_matches_independent_single_observation_reference():
    result = evaluate_tweedie_density(
        np.array([0.04564326798684731]),
        np.array([2.859891821890267]),
        0.10602153698295053,
        1.05,
    )
    assert result.logpdf[0] == pytest.approx(-25.2177010089, abs=2e-10)
    assert result.diagnostics.certified
    assert result.diagnostics.n_approximate == 0


@pytest.mark.parametrize("p", [1.0001, 1.05, 1.5, 1.95, 1.9999])
def test_log_phi_score_matches_centered_finite_difference(p):
    y = np.array([0.02, 0.5, 3.0])
    mu = np.array([0.3, 1.2, 2.7])
    phi = 0.7
    step = 2e-6
    center = evaluate_tweedie_density(y, mu, phi, p)
    plus = evaluate_tweedie_density(y, mu, phi * np.exp(step), p)
    minus = evaluate_tweedie_density(y, mu, phi * np.exp(-step), p)
    numerical = (plus.logpdf - minus.logpdf) / (2.0 * step)
    np.testing.assert_allclose(center.log_phi_score, numerical, rtol=3e-7, atol=3e-8)
```

- [ ] **Step 2: Verify the module does not yet exist**

Run: `rtk pytest tests/test_tweedie_density.py`

Expected: collection fails with `ModuleNotFoundError`.

- [ ] **Step 3: Add immutable evaluation types and exact term formulas**

```python
@dataclass(frozen=True)
class TweedieDensityDiagnostics:
    n_positive: int
    n_exact: int
    n_approximate: int
    max_terms: int
    exact: bool
    certified: bool
    requested_rtol: float
    max_relative_tail_error: float
    method: str = "compound_poisson_series"


@dataclass(frozen=True)
class TweedieDensityEvaluation:
    logpdf: NDArray[np.float64]
    log_phi_score: NDArray[np.float64]
    diagnostics: TweedieDensityDiagnostics


class TweedieDensityError(RuntimeError):
    """An exact Tweedie density evaluation could not certify its tails."""


def _readonly(values):
    result = np.array(values, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _validate_density_arrays(y, mu, weights):
    y_arr = np.asarray(y, dtype=np.float64)
    mu_arr = np.asarray(mu, dtype=np.float64)
    weight_arr = (
        np.ones_like(y_arr) if weights is None else np.asarray(weights, dtype=np.float64)
    )
    if y_arr.ndim != 1 or mu_arr.shape != y_arr.shape or weight_arr.shape != y_arr.shape:
        raise ValueError("y, mu, and weights must be matching one-dimensional arrays")
    if np.any(~np.isfinite(y_arr)) or np.any(y_arr < 0.0):
        raise ValueError("y must be finite and nonnegative")
    if np.any(~np.isfinite(mu_arr)) or np.any(mu_arr <= 0.0):
        raise ValueError("mu must be finite and strictly positive")
    if np.any(~np.isfinite(weight_arr)) or np.any(weight_arr <= 0.0):
        raise ValueError("weights must be finite and strictly positive")
    return y_arr, mu_arr, weight_arr


def approximate_tweedie_logpdf(y, mu, phi, p, *, weights=None):
    """Return a permanently labelled saddlepoint diagnostic evaluation."""
    y_arr, mu_arr, weight_arr = _validate_density_arrays(y, mu, weights)
    power = normalize_tweedie_power(p)
    dispersion = normalize_positive_scalar("phi", phi)
    effective_phi = dispersion / weight_arr
    logpdf = np.empty_like(y_arr)
    score = np.full_like(y_arr, np.nan)
    positive = y_arr > 0.0
    rate = mu_arr ** (2.0 - power) / (effective_phi * (2.0 - power))
    logpdf[~positive] = -rate[~positive]
    score[~positive] = rate[~positive]
    if np.any(positive):
        deviance = tweedie_unit_deviance(y_arr[positive], mu_arr[positive], power)
        logpdf[positive] = -0.5 * (
            np.log(2.0 * np.pi * effective_phi[positive] * y_arr[positive] ** power)
            + deviance / effective_phi[positive]
        )
    return TweedieDensityEvaluation(
        logpdf=_readonly(logpdf),
        log_phi_score=_readonly(score),
        diagnostics=TweedieDensityDiagnostics(
            n_positive=int(np.count_nonzero(positive)),
            n_exact=int(np.count_nonzero(~positive)),
            n_approximate=int(np.count_nonzero(positive)),
            max_terms=0,
            exact=not np.any(positive),
            certified=not np.any(positive),
            requested_rtol=0.0,
            max_relative_tail_error=0.0 if not np.any(positive) else np.inf,
            method="saddlepoint",
        ),
    )


def _log_term(j: int, lam: float, alpha: float, scaled_y: float, y: float) -> float:
    return (
        -lam
        - scaled_y
        - math.log(y)
        + j * (math.log(lam) + alpha * math.log(scaled_y))
        - float(gammaln(j + 1.0))
        - float(gammaln(j * alpha))
    )


def _log_forward_ratio(j: int, lam: float, alpha: float, scaled_y: float) -> float:
    return (
        math.log(lam)
        - math.log(j + 1.0)
        + alpha * math.log(scaled_y)
        + float(gammaln(j * alpha))
        - float(gammaln((j + 1.0) * alpha))
    )


def _term_log_phi_score(j: int, lam: float, alpha: float, scaled_y: float) -> float:
    return lam + scaled_y - j * (1.0 + alpha)
```

Implement `_find_mode_index` by bracketing and binary-searching the first integer `j` whose forward
log ratio is nonpositive. Implement `_certified_series` by expanding upward and downward from that
mode, maintaining log terms and score terms. Once each outward ratio is below one and monotone,
bound its unsummed tail geometrically. Stop only when both log tail bounds are at least
`-log(1 / rtol)` below the accumulated `logsumexp`. Apply the same bound to the first moment
`sum(j * term_j)` before returning the score; certifying density mass alone is insufficient.
Otherwise raise `TweedieDensityError` after `max_terms=1_000_000` total terms. The exception retains
only observation index, power, dispersion, term count, tolerance, and a neutral reason string.

- [ ] **Step 4: Implement vector preparation and zero mass**

For each observation use prior-weight effective dispersion `phi_i = phi / weight_i` and:

```python
alpha = (2.0 - p) / (p - 1.0)
lam = mu ** (2.0 - p) / (phi_i * (2.0 - p))
scale = phi_i * (p - 1.0) * mu ** (p - 1.0)
scaled_y = y / scale
```

For `y == 0`, set `logpdf = -lam` and `log_phi_score = lam`. For `y > 0`, call the certified series.
Return read-only owned arrays and diagnostics with `n_approximate=0`, `exact=True`, and
`certified=True`.

- [ ] **Step 5: Run density tests and commit**

Run: `rtk pytest tests/test_tweedie_density.py`

Expected: all exact density, certification, zero-mass, weight, and score tests pass.

```bash
rtk git add src/superglm/_tweedie_density.py tests/test_tweedie_density.py
rtk git commit -m "Add certified Tweedie density series"
```

### Task 4: Route public likelihood and dispersion profiling through the exact kernel

**Files:**
- Modify: `src/superglm/profiling/tweedie.py:307-803,873-2338`
- Modify: `src/superglm/distributions.py:311-316`
- Modify: `tests/test_tweedie_profile.py`
- Modify: `tests/test_tweedie_density.py`

- [ ] **Step 1: Add failing no-silent-approximation regressions**

```python
def test_public_logpdf_uses_certified_exact_density_for_hard_low_power_case():
    from superglm.profiling.tweedie import tweedie_logpdf

    value = tweedie_logpdf(
        np.array([0.04564326798684731]),
        np.array([2.859891821890267]),
        0.10602153698295053,
        1.05,
    )
    assert value[0] == pytest.approx(-25.2177010089, abs=2e-10)


def test_public_logpdf_propagates_exact_certification_failure(monkeypatch):
    import superglm._tweedie_density as density
    from superglm.profiling.tweedie import tweedie_logpdf

    def fail(*args, **kwargs):
        raise density.TweedieDensityError("tail not certified")

    monkeypatch.setattr(density, "evaluate_tweedie_density", fail)
    with pytest.raises(density.TweedieDensityError, match="tail not certified"):
        tweedie_logpdf(np.array([1.0]), np.array([1.0]), 1.0, 1.5)
```

- [ ] **Step 2: Verify the hard case currently selects the approximation**

Run: `rtk pytest tests/test_tweedie_density.py -k 'public_logpdf'`

Expected: the hard-case value is approximately `-22.7108285355`, so the exact-value assertion fails.

- [ ] **Step 3: Replace internal density branching with compatibility adapters**

Make `_evaluate_tweedie_density` call `superglm._tweedie_density.evaluate_tweedie_density` and adapt
its diagnostics to the existing private result while downstream code is migrated. Delete the
`t_arg_limit`-driven exact/saddlepoint selection. `tweedie_logpdf` and `Tweedie.log_likelihood` use
exact mode unconditionally. Expose `approximate_tweedie_logpdf` from the private density module as
the only explicit saddlepoint diagnostic entry point; it returns `exact=False`, `certified=False`,
and `method="saddlepoint"`. No likelihood/profile caller invokes it by default.

Update `_PhiEvaluationCache` so both NLL and analytic score use the new evaluation returned from a
single call; negate `log_phi_score` exactly once when the optimizer requests the NLL derivative.
Remove branch-threshold calibration, branch-edge optimization, and fallback segments
whose only purpose was compensating for exact/approximate switching.

- [ ] **Step 4: Run the full low-level density/dispersion suite**

Run: `rtk pytest tests/test_tweedie_density.py tests/test_tweedie_profile.py -k 'logpdf or phi or density or saddlepoint'`

Expected: exact-value and score tests pass; legacy tests that required silent fallback are rewritten
to assert explicit approximate diagnostics instead.

- [ ] **Step 5: Commit the integration**

```bash
rtk git add src/superglm/_tweedie_density.py src/superglm/profiling/tweedie.py src/superglm/distributions.py tests/test_tweedie_density.py tests/test_tweedie_profile.py
rtk git commit -m "Use exact Tweedie likelihood profiling"
```

### Task 5: Add neutral reference fixtures and a clean-room comparison harness

**Files:**
- Create: `tests/fixtures/tweedie_reference_values.json`
- Create: `tests/test_tweedie_reference.py`
- Create: `scripts/check_tweedie_reference.py`

- [ ] **Step 1: Define and validate the neutral fixture schema**

Use this committed top-level structure:

```json
{
  "format": "superglm.tweedie.reference.v1",
  "tolerances": {"logpdf_abs": 1e-8, "p_abs": 0.0002, "phi_rel": 0.0005},
  "density_cases": [
    {
      "y": 0.04564326798684731,
      "mu": 2.859891821890267,
      "phi": 0.10602153698295053,
      "p": 1.05,
      "weight": 1.0,
      "logpdf": -25.2177010089
    }
  ],
  "profile_cases": [
    {
      "case": "seed_101_low_power",
      "seed": 101,
      "n": 800,
      "true_p": 1.2,
      "true_phi": 0.8,
      "reference_p": 1.196897
    }
  ]
}
```

Expand `density_cases` with independently produced black-box values at distances
`1e-6, 1e-4, 1e-2, 0.25, 0.5` from the open boundaries and across zero/positive response, mean,
dispersion, and weight scales. Fixture generation happens outside the repository; only numeric
input/output records enter this file. `profile_cases` records the full deterministic data recipe and
the black-box fitted power; the test regenerates the response with the package generator and never
loads serialized external objects.

- [ ] **Step 2: Write the fixture comparison test**

```python
def test_density_matches_neutral_reference_fixture():
    payload = json.loads(FIXTURE.read_text())
    tolerance = payload["tolerances"]["logpdf_abs"]
    for case in payload["density_cases"]:
        result = tweedie_logpdf(
            np.array([case["y"]]),
            np.array([case["mu"]]),
            case["phi"],
            case["p"],
            weights=np.array([case["weight"]]),
        )
        assert result[0] == pytest.approx(case["logpdf"], abs=tolerance)
```

- [ ] **Step 3: Implement a version-neutral JSON-lines checker**

`scripts/check_tweedie_reference.py` accepts `--command` and sends each fixture input as one JSON
line to that command's stdin. It parses one JSON output line containing `logpdf`, compares against
the local exact kernel and committed value, prints the maximum error, and exits nonzero when the
fixture tolerance is exceeded. The script records no external identifiers or source information.

Run: `rtk proxy uv run python scripts/check_tweedie_reference.py --fixture tests/fixtures/tweedie_reference_values.json --self-check`

Expected: prints a maximum absolute error no larger than `1e-8` and exits zero. `--self-check` uses
the local kernel to validate protocol/schema without an external executable.

- [ ] **Step 4: Run and commit the reference suite**

Run: `rtk pytest tests/test_tweedie_reference.py tests/test_tweedie_density.py`

Expected: all reference and exact-density cases pass.

```bash
rtk git add tests/fixtures/tweedie_reference_values.json tests/test_tweedie_reference.py scripts/check_tweedie_reference.py
rtk git commit -m "Add neutral Tweedie reference suite"
```

### Task 6: Prove end-to-end low-power correctness and numerical performance

**Files:**
- Modify: `tests/test_tweedie_reference.py`
- Modify: `tests/test_tweedie_profile.py`

- [ ] **Step 1: Add the deterministic seed-101 regression**

```python
def test_low_power_profile_selects_certified_interior_solution():
    rng = np.random.default_rng(101)
    n = 800
    x = rng.normal(size=n)
    mu = np.exp(0.3 + 0.45 * x)
    y = generate_tweedie_cpg(n, mu=mu, phi=0.8, p=1.2, rng=rng)
    model = SuperGLM(
        family=Tweedie(1.5), selection_penalty=0.0, features={"x": Numeric()}
    )
    result = estimate_tweedie_p(model, pd.DataFrame({"x": x}), y)
    assert result.p_hat == pytest.approx(1.196897, abs=2e-4)
    assert result.phi_hat > 0.0
    assert result.density_exact is True
    assert result.converged is True
```

- [ ] **Step 2: Demonstrate the regression fails on the original density path**

Run: `rtk pytest tests/test_tweedie_reference.py -k low_power_profile -x`

Expected before integration: `p_hat == 1.05` and `converged is False`.

- [ ] **Step 3: Add a bounded-work assertion**

Mark a test `@pytest.mark.slow` that evaluates 200 representative positive observations at
`p=1.05`, `1.5`, and `1.95`; assert certification and that `max_terms` remains below 100,000 for
these ordinary cases. Keep wall-clock measurements as developer diagnostics rather than a flaky CI
threshold; the one-million-term limit remains a hard safety failure.

- [ ] **Step 4: Run the numerical workstream gates**

Run: `rtk pytest tests/test_tweedie_numerics.py tests/test_tweedie_density.py tests/test_tweedie_reference.py tests/test_tweedie_profile.py -k 'tweedie or phi or density or deviance'`

Run: `rtk ruff check src/superglm/_tweedie_numerics.py src/superglm/_tweedie_density.py src/superglm/distributions.py src/superglm/profiling/tweedie.py tests/test_tweedie_numerics.py tests/test_tweedie_density.py tests/test_tweedie_reference.py`

Expected: selected tests and Ruff pass.

- [ ] **Step 5: Commit the end-to-end regression**

```bash
rtk git add tests/test_tweedie_reference.py tests/test_tweedie_profile.py
rtk git commit -m "Cover low-power Tweedie profiling"
```
