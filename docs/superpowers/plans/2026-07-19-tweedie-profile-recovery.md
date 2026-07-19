# Tweedie Profile Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Tweedie density, dispersion, profile-power, fit statistics, and fixed-power REML results agree with independent references while preserving the performance of PR #156.

**Architecture:** Keep the remediation branch's profile search and fit workspace. Repair only reproduced numerical defects: use the existing stable deviance kernel, retain valid tiny means in Pearson scale, publish the terminal Tweedie PIRLS scale, add one compact vectorized compound-Poisson log-series for rows SciPy cannot evaluate, and reuse the shared density normalizer for fitted/null likelihoods. Default likelihood evaluation is exact or raises clearly; the existing explicit forced-saddlepoint compatibility path remains approximate.

**Tech Stack:** Python 3.13, NumPy, SciPy (`gammaln`, `wright_bessel`), pytest, Ruff, mypy, Git worktrees.

---

## Fixed context and file map

The branch is `codex/tweedie-profile-recovery`, based on PR #156 commit `3656b50`. The frozen comparison worktree is `/home/mhick/python_projects/superglm/.worktrees/fremtpl-benchmark-remediation`. Do not copy production code from the abandoned `codex/tweedie-correctness` branch.

- Create `src/superglm/_tweedie_series.py`: one vectorized positive-term log-series and first-moment evaluator; no `Decimal`, scalar-per-row loop, certification objects, or alternate object model.
- Modify `src/superglm/distributions.py`: delegate Tweedie unit deviance to the stable kernel already used by profiling.
- Modify `src/superglm/profiling/tweedie.py`: remove the tiny-mean Pearson floor, route failed/large Wright rows to the vectorized exact series, and expose one private fitted/null likelihood-pair helper.
- Modify `src/superglm/model/reml_finalize.py`: retain the terminal Tweedie PIRLS Pearson scale.
- Modify `src/superglm/model/fit_ops.py`: calculate Tweedie fitted/null likelihoods with one shared density evaluation.
- Create `tests/test_tweedie_numerics.py`: compact regression tests for deviance, Pearson scale, density references, exact routing, score, and fitted/null reuse.
- Create `tests/test_tweedie_reml_reference.py`: one deterministic black-box R comparison.
- Create `tests/test_tweedie_profile_reference.py`: one deterministic joint profile reference, with its generator and response digest in the test.
- Modify `tests/test_tweedie_profile_performance.py`: structural batch-routing and density-pass characterizations only; wall-clock gates remain alternating verification.

Explicitly excluded: metaclasses, descriptors, reducers, slots, copy hooks, recursive object-graph validation, serialization, concurrency, editor/reporting/plotting work, universal float certification, and generic solver rewrites.

### Task 1: Stable unit deviance and Pearson dispersion

**Files:**
- Create: `tests/test_tweedie_numerics.py`
- Modify: `src/superglm/distributions.py:302-310`
- Modify: `src/superglm/profiling/tweedie.py:816-850`

- [ ] **Step 1: Add the two reproduced failing tests**

```python
from __future__ import annotations

import numpy as np
import pytest

from superglm.distributions import Tweedie
from superglm.profiling.tweedie import estimate_phi


@pytest.mark.parametrize("p", [1.000001, 1.01, 1.5, 1.99, 1.999999])
def test_unit_deviance_is_exactly_zero_when_response_equals_extreme_mean(p: float) -> None:
    values = np.array([1.0e-20, 1.0, 1.0e12])
    actual = Tweedie(p).deviance_unit(values, values)
    np.testing.assert_array_equal(actual, np.zeros_like(values))


def test_pearson_phi_preserves_valid_tiny_means() -> None:
    y = np.array([1.0e-12, 2.0e-12])
    mu = np.array([1.0e-20, 2.0e-20])
    p = 1.5
    expected = float(np.mean((y - mu) ** 2 / mu**p))
    assert estimate_phi(y, mu, p) == pytest.approx(expected, rel=2.0e-15)
```

- [ ] **Step 2: Run the tests and confirm the reproduced failures**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src rtk .venv/bin/pytest tests/test_tweedie_numerics.py -q -p no:cacheprovider
```

Expected: both tests fail. Current deviance includes `-20.02246`; current Pearson scale is about `2.5e-9` instead of `1.207106757e6`.

- [ ] **Step 3: Reuse the stable deviance kernel and remove only the artificial Pearson floor**

Replace `Tweedie.deviance_unit` with:

```python
    def deviance_unit(self, y: NDArray, mu: NDArray) -> NDArray:
        """Tweedie unit deviance evaluated without close-mean cancellation."""
        from superglm.profiling.tweedie import _tweedie_positive_unit_deviance

        return _tweedie_positive_unit_deviance(y, mu, self.p)
```

Replace the Pearson calculation in `estimate_phi` with:

```python
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        pearson = (y - mu) ** 2 / np.power(mu, p)
    denom = float(df_resid if df_resid is not None else len(y))
    numer = float(np.sum(weights * pearson)) if weights is not None else float(np.sum(pearson))
    phi_hat = numer / denom
    if not np.isfinite(phi_hat):
        raise FloatingPointError("Tweedie Pearson dispersion is non-finite for these inputs")
    return phi_hat
```

- [ ] **Step 4: Run focused and existing numerical tests**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src rtk .venv/bin/pytest tests/test_tweedie_numerics.py tests/test_tweedie_profile.py -q -k "deviance or estimate_phi or weighted_phi" -p no:cacheprovider
```

Expected: PASS.

- [ ] **Step 5: Benchmark the 10,000-row kernel and commit**

```bash
PYTHONPATH=src rtk .venv/bin/python - <<'PY'
import time
import numpy as np
from superglm.distributions import Tweedie
y = np.geomspace(1e-8, 1e8, 10_000)
mu = y * np.exp(np.linspace(-1e-4, 1e-4, len(y)))
family = Tweedie(1.5)
started = time.perf_counter()
for _ in range(100):
    family.deviance_unit(y, mu)
elapsed = (time.perf_counter() - started) / 100
assert elapsed < 0.05, elapsed
print({"seconds_per_call": elapsed})
PY
rtk git add src/superglm/distributions.py src/superglm/profiling/tweedie.py tests/test_tweedie_numerics.py
rtk git commit -m "fix: stabilize Tweedie deviance and Pearson scale"
```

### Task 2: Publish the terminal fixed-power REML scale

**Files:**
- Create: `tests/test_tweedie_reml_reference.py`
- Modify: `src/superglm/model/reml_finalize.py:9-12,315-342`

- [ ] **Step 1: Add the deterministic R black-box regression**

```python
"""Black-box Tweedie REML comparison generated with R 4.5.3/package 1.9-4."""

import numpy as np
import pandas as pd
import pytest

from superglm import SuperGLM, Tweedie
from superglm.features.spline import Spline


def test_fixed_power_terminal_scale_matches_black_box_reference() -> None:
    n = 300
    x = np.linspace(0.0, 1.0, n)
    row = np.arange(1, n + 1)
    mean = np.exp(0.4 + np.sin(2.0 * np.pi * x))
    y = mean * (0.65 + 0.7 * ((row % 11) / 10.0))
    y[row % 5 == 0] = 0.0
    frame = pd.DataFrame({"x": x})
    model = SuperGLM(
        features={"x": Spline(n_knots=6)},
        family=Tweedie(p=1.5),
        selection_penalty=0,
    )
    model.fit_reml(frame, y, max_reml_iter=30)
    assert model._reml_result.converged
    assert model.result.phi == pytest.approx(0.3741648, rel=0.02)
    assert model.result.deviance == pytest.approx(309.7028, rel=0.001)
    assert model.result.effective_df == pytest.approx(5.87704, abs=0.5)
    assert float(np.mean(model.predict(frame))) == pytest.approx(1.496676, rel=0.002)
```

- [ ] **Step 2: Confirm only the published scale fails**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src rtk .venv/bin/pytest tests/test_tweedie_reml_reference.py -q -p no:cacheprovider
```

Expected: FAIL because the model publishes about `1.0517`; terminal PIRLS already contains about `0.3699`.

- [ ] **Step 3: Give terminal Tweedie PIRLS scale precedence**

Import `Tweedie` and begin the final scale selection with:

```python
from superglm.distributions import Gamma, Gaussian, Tweedie, clip_mu

# ...
    if isinstance(model._distribution, Tweedie):
        phi_fixed = float(final_pirls.phi)
    elif terminal_evaluation is not None and terminal_evaluation.profiled_scale is not None:
        phi_fixed = terminal_evaluation.profiled_scale.phi
```

Keep the remaining `elif` and `else` branches unchanged.

- [ ] **Step 4: Run the R reference and REML regression slice, then commit**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src rtk .venv/bin/pytest tests/test_tweedie_reml_reference.py tests/test_reml.py tests/test_tweedie_profile.py -q -k "terminal_scale or reml" --maxfail=1 -p no:cacheprovider
rtk git add src/superglm/model/reml_finalize.py tests/test_tweedie_reml_reference.py
rtk git commit -m "fix: retain terminal Tweedie REML dispersion"
```

Expected: PASS.

### Task 3: Replace silent default saddlepoint fallback with one vectorized exact series

**Files:**
- Create: `src/superglm/_tweedie_series.py`
- Extend: `tests/test_tweedie_numerics.py`
- Create: `tests/test_tweedie_profile_reference.py`
- Modify: `src/superglm/profiling/tweedie.py:323-810`

- [ ] **Step 1: Add compact neutral density and routing references**

Append to `tests/test_tweedie_numerics.py`:

```python
from superglm.profiling.tweedie import (
    _evaluate_tweedie_density,
    _prepare_tweedie_density,
    _tweedie_logpdf_impl,
    tweedie_logpdf,
)


@pytest.mark.parametrize(
    ("y", "mu", "phi", "p", "weight", "expected"),
    [
        (0.017, 0.02, 0.004, 1.0001, 0.4, -242.08168865838033),
        (9000.0, 10000.0, 200.0, 1.01, 0.5, -8.636743836168788),
        (0.22, 0.15, 0.125, 1.25, 4.0, 1.0417074672233964),
        (0.03, 4.0, 0.7, 1.5, 0.2, -2.2657661799346407),
        (80.0, 50.0, 5.3, 1.75, 4.0, -5.206967741958142),
        (0.0002, 0.001, 1.3, 1.99, 0.1, 5.575914834713504),
        (0.04564326798684731, 2.859891821890267, 0.10602153698295053, 1.05, 1.0, -25.217701008861372),
    ],
)
def test_public_density_matches_neutral_high_precision_reference(
    y: float, mu: float, phi: float, p: float, weight: float, expected: float
) -> None:
    actual = tweedie_logpdf(
        np.array([y]), np.array([mu]), phi, p, weights=np.array([weight])
    )
    assert actual[0] == pytest.approx(expected, rel=0.0, abs=2.5e-9)


def test_default_density_uses_exact_series_instead_of_saddlepoint() -> None:
    actual, diagnostics = _tweedie_logpdf_impl(
        np.array([0.04564326798684731]),
        np.array([2.859891821890267]),
        0.10602153698295053,
        1.05,
    )
    assert actual[0] == pytest.approx(-25.217701008861372, abs=2.5e-9)
    assert diagnostics.n_series == 1
    assert diagnostics.n_saddlepoint == 0


def test_exact_series_log_phi_score_matches_finite_difference() -> None:
    y = np.array([0.04564326798684731, 9000.0])
    mu = np.array([2.859891821890267, 10000.0])
    weights = np.array([1.0, 0.5])
    p = 1.05
    phi = 0.10602153698295053
    prepared = _prepare_tweedie_density(y, mu, p, weights=weights)
    evaluated = _evaluate_tweedie_density(prepared, phi, compute_score=True)
    step = 1.0e-5
    upper = _evaluate_tweedie_density(prepared, phi * np.exp(step)).logpdf
    lower = _evaluate_tweedie_density(prepared, phi * np.exp(-step)).logpdf
    finite_difference = -float(np.mean(upper - lower)) / (2.0 * step)
    assert evaluated.score_valid
    assert evaluated.log_phi_score is not None
    assert float(np.mean(evaluated.log_phi_score)) == pytest.approx(
        finite_difference, rel=2.0e-6, abs=2.0e-7
    )
```

- [ ] **Step 2: Add one real joint-profile reference without a checker framework**

Create `tests/test_tweedie_profile_reference.py`:

```python
"""Neutral Tweedie profile reference generated independently at high precision."""

import hashlib

import numpy as np
import pandas as pd
import pytest

from superglm import SuperGLM, Tweedie
from superglm.features.numeric import Numeric
from superglm.profiling.tweedie import estimate_tweedie_p, generate_tweedie_cpg


@pytest.mark.slow
def test_joint_profile_matches_neutral_reference() -> None:
    rng = np.random.default_rng(101)
    x = rng.standard_normal(800)
    mu = np.exp(0.3 + 0.45 * x)
    y = generate_tweedie_cpg(800, mu=mu, phi=0.8, p=1.2, rng=rng)
    digest = hashlib.sha256(np.ascontiguousarray(y, dtype="<f8").tobytes()).hexdigest()
    assert digest == "7d2c5cf30a0d8f3c1a7fb281adb2c864900f1ec16e59fdfff536d197f3186477"
    model = SuperGLM(features={"x": Numeric()}, family=Tweedie(p=1.5))
    result = estimate_tweedie_p(
        model,
        pd.DataFrame({"x": x}),
        y,
        p_bounds=(1.05, 1.95),
        xatol=1.0e-4,
        maxiter=30,
        phi_method="mle",
        method="brent",
    )
    assert result.p_hat == pytest.approx(1.1968971098776182, abs=2.0e-4)
    assert result.phi_hat == pytest.approx(0.8068142191615686, rel=5.0e-4)
    assert result.converged
    assert result.density_exact
    assert result.n_saddlepoint == 0
```

- [ ] **Step 3: Confirm failures are caused by default approximation**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src rtk .venv/bin/pytest tests/test_tweedie_numerics.py -q -k "neutral or exact_series or saddlepoint" -p no:cacheprovider
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src rtk .venv/bin/pytest tests/test_tweedie_profile_reference.py -q -m slow -p no:cacheprovider
```

Expected: hard density differs by about `2.5069`; profile returns about `p=1.05`, `phi=0.713`, and `converged=False`.

- [ ] **Step 4: Implement the single vectorized positive-term series**

Create `src/superglm/_tweedie_series.py`:

```python
"""Vectorized compound-Poisson normalizer used by Tweedie exact density."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import gammaln

_SERIES_RTOL = 5.0e-15
_SERIES_MAX_TERMS = 100_000


def tweedie_log_series(
    log_t: NDArray,
    a: float,
    *,
    rtol: float = _SERIES_RTOL,
    max_terms: int = _SERIES_MAX_TERMS,
) -> tuple[NDArray, NDArray]:
    """Return log(sum terms) and E[J] for t**j/(j! Gamma(a*j)), j >= 1."""
    values = np.asarray(log_t, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise FloatingPointError("Tweedie exact series requires finite one-dimensional log(t)")
    if not np.isfinite(a) or a <= 0.0:
        raise FloatingPointError("Tweedie exact series requires finite a > 0")
    log_sum = np.full(values.shape, -np.inf, dtype=np.float64)
    log_first_moment = np.full(values.shape, -np.inf, dtype=np.float64)
    active = np.ones(values.shape, dtype=np.bool_)

    for j in range(1, max_terms + 1):
        indices = np.flatnonzero(active)
        if indices.size == 0:
            break
        active_log_t = values[indices]
        log_term = j * active_log_t - gammaln(j + 1.0) - gammaln(a * j)
        log_sum[indices] = np.logaddexp(log_sum[indices], log_term)
        log_first_moment[indices] = np.logaddexp(
            log_first_moment[indices], np.log(float(j)) + log_term
        )
        log_ratio = (
            active_log_t
            - np.log(j + 1.0)
            - gammaln(a * (j + 1.0))
            + gammaln(a * j)
        )
        declining = log_ratio < 0.0
        if not np.any(declining):
            continue
        declining_indices = indices[declining]
        declining_log_ratio = log_ratio[declining]
        ratio = np.exp(declining_log_ratio)
        log_one_minus_ratio = np.log1p(-ratio)
        log_next = log_term[declining] + declining_log_ratio
        log_mass_tail = log_next - log_one_minus_ratio
        log_moment_factor = np.logaddexp(
            np.log(j + 1.0) - log_one_minus_ratio,
            declining_log_ratio - 2.0 * log_one_minus_ratio,
        )
        log_moment_tail = log_next + log_moment_factor
        done = (
            (log_mass_tail <= log_sum[declining_indices] + np.log(rtol))
            & (
                log_moment_tail
                <= log_first_moment[declining_indices] + np.log(rtol)
            )
        )
        active[declining_indices[done]] = False

    if np.any(active):
        raise FloatingPointError(
            "Tweedie exact series did not converge for "
            f"{int(np.count_nonzero(active))} row(s) within {max_terms} terms"
        )
    expected_j = np.exp(log_first_moment - log_sum)
    if not np.all(np.isfinite(log_sum)) or not np.all(np.isfinite(expected_j)):
        raise FloatingPointError("Tweedie exact series produced a non-finite result")
    return log_sum, expected_j
```

- [ ] **Step 5: Route only failed/large Wright rows through the series**

Import `tweedie_log_series`, add `n_series: int = 0` to `_TweedieLogpdfDiagnostics`, and replace the fallback mask with:

```python
        use_series = ~exact & (prepared.t_arg_limit > 0.0)
        series_expected_j = np.full(len(log_t), np.nan, dtype=np.float64)
        if np.any(use_series):
            series_log_sum, expected_j = tweedie_log_series(log_t[use_series], prepared.a)
            positive_logpdf[use_series] = (
                series_log_sum
                - prepared.positive_log_y[use_series]
                + prepared.positive_canonical_c[use_series]
                * inverse_phi_positive[use_series]
            )
            series_expected_j[use_series] = expected_j
        saddlepoint = ~(exact | use_series)
```

Fill series scores before the Wright score block:

```python
            if np.any(use_series):
                positive_score[use_series] = (
                    series_expected_j[use_series] / (prepared.p - 1.0)
                    + prepared.positive_canonical_c[use_series]
                    * inverse_phi_positive[use_series]
                )
```

Construct diagnostics with `n_series=int(np.count_nonzero(use_series))` and current positive/saddlepoint counts. Update the public docstring to state that positive default limits use exact Wright-or-series evaluation. Preserve `t_arg_limit <= 0` solely as the existing explicit forced-saddlepoint compatibility path; never silently approximate for a positive limit.

- [ ] **Step 6: Run focused and real-profile tests, then commit**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src rtk .venv/bin/pytest tests/test_tweedie_numerics.py tests/test_tweedie_profile.py -q -k "logpdf or density or score or branch or saddlepoint" --maxfail=1 -p no:cacheprovider
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src rtk .venv/bin/pytest tests/test_tweedie_profile_reference.py -q -m slow -p no:cacheprovider
rtk git add src/superglm/_tweedie_series.py src/superglm/profiling/tweedie.py tests/test_tweedie_numerics.py tests/test_tweedie_profile_reference.py
rtk git commit -m "fix: vectorize exact Tweedie density fallback"
```

Expected: PASS; neutral `p`/`phi` agree, convergence is true, and no default saddlepoint rows remain.

### Task 4: Reuse the fitted/null normalizer in fit statistics

**Files:**
- Extend: `tests/test_tweedie_numerics.py`
- Extend: `tests/test_tweedie_profile_performance.py`
- Modify: `src/superglm/profiling/tweedie.py:733-810`
- Modify: `src/superglm/model/fit_ops.py:12-22,205-238`

- [ ] **Step 1: Add correctness and one-pass routing tests**

Append to `tests/test_tweedie_numerics.py`:

```python
import superglm.profiling.tweedie as tweedie_module
from superglm.links import LogLink
from superglm.model.fit_ops import _compute_fit_stats


def test_tweedie_fit_stats_reuses_one_density_normalizer(monkeypatch) -> None:
    y = np.array([0.0, 0.3, 1.2, 4.5])
    mu = np.array([0.2, 0.5, 1.5, 3.7])
    null_mu = np.full_like(y, 1.1)
    weights = np.array([0.4, 0.8, 1.2, 1.8])
    family = Tweedie(1.55)
    expected_ll = float(np.sum(tweedie_logpdf(y, mu, 0.8, 1.55, weights=weights)))
    expected_null_ll = float(
        np.sum(tweedie_logpdf(y, null_mu, 0.8, 1.55, weights=weights))
    )
    real_evaluate = tweedie_module._evaluate_tweedie_density
    calls = 0

    def counted(prepared, phi, *, compute_score=False):
        nonlocal calls
        calls += 1
        return real_evaluate(prepared, phi, compute_score=compute_score)

    monkeypatch.setattr(tweedie_module, "_evaluate_tweedie_density", counted)
    stats = _compute_fit_stats(
        y, mu, weights, None, family, LogLink(), 0.8, null_mu=null_mu
    )
    assert calls == 1
    assert stats.log_likelihood == pytest.approx(expected_ll, abs=1.0e-11)
    assert stats.null_log_likelihood == pytest.approx(expected_null_ll, abs=1.0e-11)
```

Append to `tests/test_tweedie_profile_performance.py`:

```python
def test_ten_thousand_row_likelihood_pair_is_vectorized(monkeypatch) -> None:
    n = 10_000
    y = np.geomspace(0.01, 100.0, n)
    mu = y * np.exp(np.linspace(-0.2, 0.2, n))
    null_mu = np.full(n, 1.0)
    weights = np.geomspace(0.5, 2.0, n)
    real_series = tweedie_module.tweedie_log_series
    batch_sizes: list[int] = []

    def counted_series(log_t, a, **kwargs):
        batch_sizes.append(len(log_t))
        return real_series(log_t, a, **kwargs)

    monkeypatch.setattr(tweedie_module, "tweedie_log_series", counted_series)
    fitted, null = tweedie_module._tweedie_logpdf_pair(
        y, mu, null_mu, 0.8, 1.5, weights=weights
    )
    assert fitted.shape == null.shape == (n,)
    assert len(batch_sizes) <= 1
    assert all(size > 1 for size in batch_sizes)
```

- [ ] **Step 2: Confirm the pair helper is absent and fit statistics use two evaluations**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src rtk .venv/bin/pytest tests/test_tweedie_numerics.py::test_tweedie_fit_stats_reuses_one_density_normalizer tests/test_tweedie_profile_performance.py::test_ten_thousand_row_likelihood_pair_is_vectorized -q -p no:cacheprovider
```

Expected: FAIL because `_tweedie_logpdf_pair` does not exist and fit statistics call density twice.

- [ ] **Step 3: Add the private likelihood-pair helper**

Add after `tweedie_logpdf`:

```python
def _tweedie_logpdf_pair(
    y: NDArray,
    mu: NDArray,
    null_mu: NDArray,
    phi: float,
    p: float,
    *,
    weights: NDArray | None = None,
) -> tuple[NDArray, NDArray]:
    """Return fitted/null exact log densities with one shared normalizer pass."""
    prepared = _prepare_tweedie_density(y, mu, p, weights=weights)
    evaluation = _evaluate_tweedie_density(prepared, phi)
    if evaluation.diagnostics.n_saddlepoint:
        raise FloatingPointError("Tweedie fitted/null reuse requires exact density evaluation")
    null_array = np.asarray(null_mu, dtype=np.float64)
    if (
        null_array.shape != prepared.mu.shape
        or not np.all(np.isfinite(null_array))
        or np.any(null_array <= 0.0)
    ):
        raise ValueError("null_mu must match mu and be finite and strictly positive")
    inverse_phi = prepared.weights / _validate_tweedie_phi(phi)
    canonical_fit = np.empty_like(prepared.y)
    canonical_fit[prepared.zero_mask] = -prepared.zero_rate_numerator[prepared.zero_mask]
    canonical_fit[prepared.positive_mask] = prepared.positive_canonical_c
    with np.errstate(all="ignore"):
        canonical_null = (
            prepared.y * np.power(null_array, 1.0 - prepared.p) / (1.0 - prepared.p)
            - np.power(null_array, 2.0 - prepared.p) / (2.0 - prepared.p)
        )
        shared_normalizer = evaluation.logpdf - canonical_fit * inverse_phi
        null_logpdf = shared_normalizer + canonical_null * inverse_phi
    if not np.all(np.isfinite(null_logpdf)):
        raise FloatingPointError("Tweedie null log likelihood is non-finite")
    return evaluation.logpdf.copy(), null_logpdf
```

- [ ] **Step 4: Use the helper only for Tweedie fit statistics**

Import `Tweedie` in `model/fit_ops.py`, then begin `_compute_fit_stats` with:

```python
    if null_mu is None:
        null_mu = _compute_null_mu(y, weights, offset, distribution, link)

    if isinstance(distribution, Tweedie):
        from superglm.profiling.tweedie import _tweedie_logpdf_pair

        fitted_logpdf, null_logpdf = _tweedie_logpdf_pair(
            y, mu, null_mu, phi, distribution.p, weights=weights
        )
        ll = float(np.sum(fitted_logpdf))
        null_ll = float(np.sum(null_logpdf))
    else:
        ll = distribution.log_likelihood(y, mu, weights, phi)
        null_ll = distribution.log_likelihood(y, null_mu, weights, phi)
```

Remove the prior unconditional fitted and null calls; leave deviance, Pearson, and `FitStats` construction unchanged.

- [ ] **Step 5: Run focused tests and the 10,000-row timing check, then commit**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src rtk .venv/bin/pytest tests/test_tweedie_numerics.py tests/test_tweedie_profile_performance.py -q -k "normalizer or ten_thousand" -p no:cacheprovider
PYTHONPATH=src rtk .venv/bin/python - <<'PY'
import time
import numpy as np
from superglm.profiling.tweedie import _tweedie_logpdf_pair
n = 10_000
y = np.geomspace(0.01, 100.0, n)
mu = y * np.exp(np.linspace(-0.2, 0.2, n))
null_mu = np.full(n, 1.0)
w = np.geomspace(0.5, 2.0, n)
started = time.perf_counter()
for _ in range(20):
    _tweedie_logpdf_pair(y, mu, null_mu, 0.8, 1.5, weights=w)
elapsed = (time.perf_counter() - started) / 20
assert elapsed < 0.05, elapsed
print({"seconds_per_pair": elapsed})
PY
rtk git add src/superglm/profiling/tweedie.py src/superglm/model/fit_ops.py tests/test_tweedie_numerics.py tests/test_tweedie_profile_performance.py
rtk git commit -m "perf: reuse Tweedie fit-stat normalizers"
```

Expected: PASS; the density evaluator is called once and no scalar-per-row path exists.

### Task 5: Audit profile state and enforce branch-level gates

**Files:**
- Verify: `tests/test_tweedie_profile.py`
- Verify: `tests/test_tweedie_profile_performance.py`
- Verify: all touched production and test files

- [ ] **Step 1: Run existing state characterizations without rewriting them**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src rtk .venv/bin/pytest tests/test_tweedie_profile.py -q -k "atomically_synchronizes or final_profile_refit_failure or finalizes_cached_winner or brent_evaluates_and_can_select_endpoint or aggregate_convergence or failed_profile_optimizer or winning_phi_boundary or invalid_cached_record or offset_aware" --maxfail=1 -p no:cacheprovider
```

Expected: PASS. These already establish winning-record publication, rollback, endpoints, failure propagation, and offset-aware REML means. Add no machinery for behavior already covered.

- [ ] **Step 2: Run exact-MLE performance characterizations with counters**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src rtk .venv/bin/pytest tests/test_tweedie_profile_performance.py -q -m slow -s --maxfail=1 -p no:cacheprovider
```

Expected: PASS with bounded density-pass and candidate-fit counts.

- [ ] **Step 3: Alternate remediation and recovery wall-clock runs**

Run identical deterministic workloads in this worktree and `/home/mhick/python_projects/superglm/.worktrees/fremtpl-benchmark-remediation`, alternating `baseline, recovery, baseline, recovery` under `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`. Record medians for fixed-power `fit()`, fixed-power `fit_reml()`, and `estimate_tweedie_p(phi_method="mle")`.

Use these exact gates:

```python
fit_limit = baseline_fit + max(0.10 * baseline_fit, 0.050)
reml_limit = baseline_reml + max(0.10 * baseline_reml, 0.050)
profile_limit = baseline_profile + max(0.25 * baseline_profile, 0.250)
assert recovery_fit <= fit_limit
assert recovery_reml <= reml_limit
assert recovery_profile <= profile_limit
```

If a gate fails, profile that exact workload before changing code; do not expand into unrelated fit systems.

- [ ] **Step 4: Run formatting, lint, and targeted type checks**

```bash
rtk .venv/bin/ruff format --check src/superglm/_tweedie_series.py src/superglm/distributions.py src/superglm/profiling/tweedie.py src/superglm/model/reml_finalize.py src/superglm/model/fit_ops.py tests/test_tweedie_numerics.py tests/test_tweedie_reml_reference.py tests/test_tweedie_profile_reference.py tests/test_tweedie_profile_performance.py
rtk .venv/bin/ruff check src/superglm/_tweedie_series.py src/superglm/distributions.py src/superglm/profiling/tweedie.py src/superglm/model/reml_finalize.py src/superglm/model/fit_ops.py tests/test_tweedie_numerics.py tests/test_tweedie_reml_reference.py tests/test_tweedie_profile_reference.py tests/test_tweedie_profile_performance.py
rtk .venv/bin/mypy src/superglm/_tweedie_series.py src/superglm/distributions.py src/superglm/profiling/tweedie.py src/superglm/model/reml_finalize.py src/superglm/model/fit_ops.py
```

Expected: all commands exit zero.

- [ ] **Step 5: Run focused, developer, relevant slow, and full suites**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src rtk .venv/bin/pytest tests/test_tweedie_numerics.py tests/test_tweedie_reml_reference.py tests/test_tweedie_profile_reference.py tests/test_tweedie_profile.py tests/test_tweedie_profile_performance.py -q --maxfail=1 -p no:cacheprovider
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src rtk .venv/bin/pytest tests/ -q -m "not slow and not browser" --maxfail=1 -p no:cacheprovider
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src rtk .venv/bin/pytest tests/test_tweedie_profile_reference.py tests/test_tweedie_profile_performance.py -q -m slow --maxfail=1 -p no:cacheprovider
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src rtk .venv/bin/pytest tests/ -q --maxfail=1 -p no:cacheprovider
```

Expected: all suites pass.

- [ ] **Step 6: Review scope, forbidden patterns, and diff health**

```bash
rtk git diff 3656b50 --stat
rtk git diff 3656b50 -- src tests
rtk grep -n "Decimal\|metaclass\|__reduce__\|__slots__\|descriptor\|recursive.*graph" src/superglm/_tweedie_series.py src/superglm/profiling/tweedie.py src/superglm/model/fit_ops.py src/superglm/model/reml_finalize.py src/superglm/distributions.py
rtk git diff --check
rtk git status --short
```

Expected: only focused files changed; forbidden-pattern search has no new matches; diff check is clean. Do not push or open a PR until the user requests publication.
