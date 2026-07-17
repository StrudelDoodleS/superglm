"""Focused contract tests for compound Poisson-Gamma generation."""

import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from superglm.profiling.tweedie import generate_tweedie_cpg


class _RecordingRNG:
    """Small sampler double that preserves and records complete array calls."""

    def __init__(
        self,
        *,
        counts=(),
        gamma_values=(),
        poisson_exception: BaseException | None = None,
        gamma_exception: BaseException | None = None,
    ):
        self._counts = np.array(counts, copy=True)
        self._gamma_values = np.array(gamma_values, copy=True)
        self._poisson_exception = poisson_exception
        self._gamma_exception = gamma_exception
        self.poisson_calls = []
        self.gamma_calls = []
        self.poisson_returns = []
        self.gamma_returns = []

    def poisson(self, lam):
        self.poisson_calls.append(np.array(lam, copy=True))
        if self._poisson_exception is not None:
            raise self._poisson_exception
        result = self._counts.copy()
        self.poisson_returns.append(result)
        return result

    def gamma(self, shape, *, scale):
        self.gamma_calls.append((np.array(shape, copy=True), np.array(scale, copy=True)))
        if self._gamma_exception is not None:
            raise self._gamma_exception
        result = self._gamma_values.copy()
        self.gamma_returns.append(result)
        return result


def _legacy_generate_tweedie_cpg(n, mu, phi, p, rng=None):
    """Frozen pre-hardening implementation used as a compatibility oracle."""
    if rng is None:
        rng = np.random.default_rng()

    mu = np.broadcast_to(np.asarray(mu, dtype=np.float64), (n,)).copy()

    # CPG parameters (Jørgensen 1997)
    lam = np.power(mu, 2 - p) / ((2 - p) * phi)  # Poisson rate
    alpha = (2 - p) / (p - 1)  # Gamma shape per claim
    beta = phi * (p - 1) * np.power(mu, p - 1)  # Gamma scale per claim

    # Vectorised: draw N ~ Poisson(lam), then Y|N ~ Gamma(alpha*N, beta)
    N = rng.poisson(lam)
    y = np.zeros(n, dtype=np.float64)
    pos = N > 0
    if np.any(pos):
        # Gamma additive property: sum of N iid Gamma(alpha, beta) = Gamma(N*alpha, beta)
        y[pos] = rng.gamma(alpha * N[pos], scale=beta[pos])

    return y


def _rng_that_must_not_be_used():
    unexpected_call = AssertionError("sampler called during input validation")
    return _RecordingRNG(
        poisson_exception=unexpected_call,
        gamma_exception=unexpected_call,
    )


def _assert_no_sampler_calls(rng):
    assert rng.poisson_calls == []
    assert rng.gamma_calls == []


def test_seeded_generation_matches_legacy_output_and_rng_consumption():
    hardened_rng = np.random.default_rng(271828)
    legacy_rng = np.random.default_rng(271828)

    actual = generate_tweedie_cpg(
        128,
        mu=2.75,
        phi=0.8,
        p=1.45,
        rng=hardened_rng,
    )
    expected = _legacy_generate_tweedie_cpg(
        128,
        mu=2.75,
        phi=0.8,
        p=1.45,
        rng=legacy_rng,
    )

    np.testing.assert_array_equal(actual, expected)
    assert pickle.dumps(hardened_rng.bit_generator.state, protocol=5) == pickle.dumps(
        legacy_rng.bit_generator.state,
        protocol=5,
    )


def test_vector_parameters_preserve_exact_sampler_calls_and_owned_output():
    mu = np.array([1.0, 4.0, 9.0])
    phi = np.array([0.5, 1.0, 2.0])
    counts = np.array([0, 2, 1])
    gamma_values = np.array([7.0, 11.0])
    rng = _RecordingRNG(counts=counts, gamma_values=gamma_values)

    y = generate_tweedie_cpg(3, mu=mu, phi=phi, p=1.5, rng=rng)

    assert len(rng.poisson_calls) == 1
    np.testing.assert_array_equal(rng.poisson_calls[0], np.array([4.0, 4.0, 3.0]))
    assert len(rng.gamma_calls) == 1
    shape, scale = rng.gamma_calls[0]
    np.testing.assert_array_equal(shape, np.array([2.0, 1.0]))
    np.testing.assert_array_equal(scale, np.array([1.0, 3.0]))
    np.testing.assert_array_equal(y, np.array([0.0, 7.0, 11.0]))
    assert y.shape == (3,)
    assert y.dtype == np.dtype(np.float64)
    assert y.flags.owndata
    assert not np.shares_memory(y, mu)
    assert not np.shares_memory(y, phi)
    assert not np.shares_memory(y, counts)
    assert not np.shares_memory(y, gamma_values)
    assert not np.shares_memory(y, rng.poisson_returns[0])
    assert not np.shares_memory(y, rng.gamma_returns[0])


@pytest.mark.parametrize(
    ("mu", "phi", "p"),
    [
        pytest.param(2.5, 0.8, 1.5, id="python-scalars"),
        pytest.param(
            np.array(2.5),
            np.array(0.8),
            np.array(1.5),
            id="zero-dimensional-arrays",
        ),
        pytest.param(
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            1.5,
            id="empty-exact-vectors",
        ),
    ],
)
def test_zero_samples_accept_valid_parameter_forms_without_sampler_calls(mu, phi, p):
    rng = _rng_that_must_not_be_used()

    first = generate_tweedie_cpg(0, mu=mu, phi=phi, p=p, rng=rng)
    second = generate_tweedie_cpg(0, mu=mu, phi=phi, p=p, rng=rng)

    assert first.shape == (0,)
    assert first.dtype == np.dtype(np.float64)
    assert first.flags.owndata
    assert second.shape == (0,)
    assert second.dtype == np.dtype(np.float64)
    assert second.flags.owndata
    assert first is not second
    assert not np.shares_memory(first, second)
    _assert_no_sampler_calls(rng)


def test_numpy_integer_sample_count_is_accepted():
    rng = _RecordingRNG(counts=np.array([0]))

    y = generate_tweedie_cpg(np.int64(1), mu=2.0, phi=1.0, p=1.5, rng=rng)

    assert y.shape == (1,)
    assert y.dtype == np.dtype(np.float64)
    assert len(rng.poisson_calls) == 1
    assert rng.gamma_calls == []


@pytest.mark.parametrize(
    ("n", "error_type"),
    [
        pytest.param(True, TypeError, id="python-bool"),
        pytest.param(np.bool_(False), TypeError, id="numpy-bool"),
        pytest.param(2.0, TypeError, id="float"),
        pytest.param(-1, ValueError, id="negative"),
    ],
)
def test_invalid_sample_count_is_rejected_before_sampler_use(n, error_type):
    rng = _rng_that_must_not_be_used()

    with pytest.raises(error_type, match=r"\bn\b"):
        generate_tweedie_cpg(n, mu=2.0, phi=1.0, p=1.5, rng=rng)

    _assert_no_sampler_calls(rng)


@pytest.mark.parametrize(
    "p",
    [
        pytest.param(True, id="python-bool"),
        pytest.param(np.bool_(False), id="numpy-bool"),
        pytest.param(1, id="lower-bound"),
        pytest.param(2, id="upper-bound"),
        pytest.param(np.nan, id="nan"),
        pytest.param(np.inf, id="infinity"),
        pytest.param(1.5 + 0.0j, id="complex-zero-imaginary"),
        pytest.param("1.5", id="numeric-string"),
        pytest.param(np.array(1.5, dtype=object), id="numeric-object-scalar"),
        pytest.param(np.array([1.5]), id="shape-one-array"),
    ],
)
def test_invalid_power_is_rejected_before_sampler_use(p):
    rng = _rng_that_must_not_be_used()

    with pytest.raises(ValueError, match=r"\bp\b"):
        generate_tweedie_cpg(1, mu=2.0, phi=1.0, p=p, rng=rng)

    _assert_no_sampler_calls(rng)


_INVALID_POSITIVE_PARAMETERS = [
    pytest.param(True, id="python-bool"),
    pytest.param(np.bool_(False), id="numpy-bool"),
    pytest.param(0, id="zero"),
    pytest.param(-1, id="negative"),
    pytest.param(np.nan, id="nan"),
    pytest.param(np.inf, id="infinity"),
    pytest.param(1.0 + 0.0j, id="complex-zero-imaginary"),
    pytest.param("1.0", id="numeric-string"),
    pytest.param(np.array(1.0, dtype=object), id="numeric-object-scalar"),
    pytest.param(np.array([1.0, 2.0], dtype=object), id="numeric-object-vector"),
    pytest.param(np.ones(1), id="wrong-length"),
    pytest.param(np.ones((2, 1)), id="column-vector"),
    pytest.param(np.ones((1, 2)), id="row-vector"),
]


@pytest.mark.parametrize("value", _INVALID_POSITIVE_PARAMETERS)
@pytest.mark.parametrize("name", ["mu", "phi"])
def test_invalid_positive_parameter_is_rejected_before_sampler_use(name, value):
    rng = _rng_that_must_not_be_used()
    arguments = {"mu": 1.0, "phi": 1.0}
    arguments[name] = value

    with pytest.raises(ValueError, match=rf"\b{name}\b"):
        generate_tweedie_cpg(2, p=1.5, rng=rng, **arguments)

    _assert_no_sampler_calls(rng)


@pytest.mark.parametrize(
    "rng",
    [
        pytest.param(object(), id="no-sampler-methods"),
        pytest.param(
            SimpleNamespace(poisson=lambda lam: None, gamma=None),
            id="gamma-not-callable",
        ),
        pytest.param(
            SimpleNamespace(poisson=None, gamma=lambda shape, *, scale: None),
            id="poisson-not-callable",
        ),
    ],
)
def test_rng_requires_callable_poisson_and_gamma_even_for_zero_samples(rng):
    with pytest.raises(TypeError, match=r"\brng\b"):
        generate_tweedie_cpg(0, mu=1.0, phi=1.0, p=1.5, rng=rng)


def test_zero_samples_still_validate_parameter_domains_before_sampler_use():
    rng = _rng_that_must_not_be_used()

    with pytest.raises(ValueError, match=r"\bmu\b"):
        generate_tweedie_cpg(0, mu=0.0, phi=1.0, p=1.5, rng=rng)

    _assert_no_sampler_calls(rng)


class _WrongPoissonSignatureRNG:
    def __call__(self):
        return self

    def poisson(self):
        return np.array([0])

    def gamma(self, shape, *, scale):
        return np.zeros_like(shape, dtype=np.float64)


def test_wrong_poisson_signature_raises_a_useful_type_error():
    rng = _WrongPoissonSignatureRNG()
    assert callable(rng)

    with pytest.raises(TypeError, match=r"\bpoisson\b"):
        generate_tweedie_cpg(1, mu=1.0, phi=1.0, p=1.5, rng=rng)
