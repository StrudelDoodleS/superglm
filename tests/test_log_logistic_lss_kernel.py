from importlib import import_module, util
from itertools import product

import mpmath as mp
import numpy as np
import pytest
from scipy.stats import fisk

from tests._declared_scalar_oracles import assert_scaled_jets, jets, log_density, log_window
from tests._oracle_store import OracleStore


def _density_reference(y, scale, shape):
    """The 100-digit ``mp.diff`` jets of the log density in ``(scale, shape)`` to order 4."""
    return jets(lambda x, b: log_density("log_logistic", mp.mpf(y), x, b), (scale, shape), 4)


def _window_reference(lower, upper, scale, shape):
    """The 100-digit ``mp.diff`` jets of the log window probability to order 4."""
    return jets(
        lambda s, a: log_window("log_logistic", mp.mpf(lower), mp.mpf(upper), s, a),
        (scale, shape),
        4,
    )


# The mpmath jets' answers are stored (``tests/_oracle_store.py``): their
# ``mp.diff`` stencils were nearly all of this module's time.
DENSITY_REFERENCE = OracleStore(
    "log_logistic_density_jets",
    _density_reference,
    depends=(jets, log_density),
    module="tests/test_log_logistic_lss_kernel.py",
)
WINDOW_REFERENCE = OracleStore(
    "log_logistic_window_jets",
    _window_reference,
    depends=(jets, log_window),
    module="tests/test_log_logistic_lss_kernel.py",
)


def test_the_stored_oracle_answers_are_current():
    DENSITY_REFERENCE.verify()
    WINDOW_REFERENCE.verify()


@pytest.fixture(scope="module")
def kernel():
    name = "superglm.distributional.kernels.log_logistic"
    assert util.find_spec(name) is not None, "production log-logistic numerical module is missing"
    return import_module(name)


def test_log_logistic_law_and_all_mixed_jets(kernel):
    for s, a in product((1e-6, 1.0, 1e6), (0.6, 1.0, 1.5, 2.0, 4.0, 12.0)):
        law = fisk(c=a, scale=s)
        # Independent inverse in log space; scipy.ppf cancels at 1e-100.  Each point is
        # the quantile rounded once from 50 digits, the same float on every platform
        # (and so the stored oracle's key), where float64 libm calls may differ.
        ps = (0.001, 0.1, 0.5, 0.9, 0.999, 1e-12, 1e-30, 1e-100)
        with mp.workdps(50):
            points = [float(s * mp.exp((mp.log(p) - mp.log1p(-p)) / a)) for p in ps]
            points += [
                float(s * mp.exp((mp.log1p(-p) - mp.log(p)) / a)) for p in (1e-12, 1e-30, 1e-100)
            ]
        for index, y in enumerate(points):
            actual = kernel.density_jets(np.array([y]), np.array([[s, a]]), np.ones(1), order=4)
            assert actual.valid.all()
            ref = DENSITY_REFERENCE(y, s, a)
            assert_scaled_jets(
                (actual.jets[0] + actual.carrier, *actual.jets[1:]), ref, (s, a), factor=2048
            )
            if index < 5:
                np.testing.assert_allclose(
                    actual.jets[0] + actual.carrier,
                    law.logpdf(y),
                    rtol=1024 * 2.0**-53,
                    atol=1024 * 2.0**-53,
                )


def test_log_logistic_windows_and_moment_boundaries(kernel):
    for lo, hi in (
        (0.0, np.inf),
        (0.0, 1e-100),
        (0.0, 0.1),
        (0.2, 3.0),
        (1e50, 2e50),
        (1.0, np.nextafter(1.0, np.inf)),
    ):
        t = (2.0, 0.6)
        actual, valid = kernel.window_jets(np.array([lo]), np.array([hi]), np.array([t]), order=4)
        assert valid.all()
        reference = WINDOW_REFERENCE(float(lo), float(hi), *t)
        assert_scaled_jets(actual, reference, t, factor=8192)
    for a in (
        0.6,
        1.0,
        np.nextafter(1.0, np.inf),
        1.5,
        2.0,
        np.nextafter(2.0, np.inf),
        4.0,
        12.0,
        1e8,
        1e100,
    ):
        theta = np.array([[3.0, a]])
        mean, variance = kernel.mean(theta)[0], kernel.variance(theta)[0]
        assert np.isinf(mean) == (a <= 1)
        assert np.isinf(variance) == (a <= 2)
        # 230 digits resolve variance O(a^-2) for the independent moment subtraction.
        with mp.workdps(230):
            x = mp.pi / mp.mpf(a)
            if a > 1:
                np.testing.assert_allclose(mean, float(3 * x / mp.sin(x)), rtol=4096 * 2.0**-53)
            if a > 2:
                reference = float(9 * (2 * x / mp.sin(2 * x) - (x / mp.sin(x)) ** 2))
                np.testing.assert_allclose(variance, reference, rtol=8192 * 2.0**-53, atol=0)
        assert kernel.quantile(0.5, theta)[0] == 3.0
        for p in (0.001, 0.9, 1e-100):
            q = kernel.quantile(p, theta)
            if a < 1e8:
                np.testing.assert_allclose(kernel.cdf(q, theta), p, rtol=8192 * 2.0**-53, atol=0)
    assert not kernel.density_jets(np.zeros(1), np.ones((1, 2)), np.ones(1), order=2).valid[0]


def test_log_logistic_lazy_orders(kernel, monkeypatch):
    from superglm.distributional.kernels._declared import log_logistic as generated

    density = generated.density

    def guarded(y, s, a, m, order, *levels):
        assert all(len(level) == 0 for level in levels[order + 1 :])
        return density(y, s, a, m, order, *levels)

    monkeypatch.setattr(generated, "density", guarded)
    for order in range(5):
        assert (
            len(kernel.density_jets(np.ones(1), np.ones((1, 2)), np.ones(1), order=order).jets)
            == order + 1
        )


def test_concentrated_cdf_retains_near_median_resolution(kernel):
    scale = 1e6
    y = np.array([np.nextafter(scale, 0.0), scale, np.nextafter(scale, np.inf)])
    a = 1e16
    theta = np.tile([scale, a], (3, 1))
    with mp.workdps(80):
        expected = [float(1 / (1 + (mp.mpf(scale) / mp.mpf(float(x))) ** mp.mpf(a))) for x in y]
    np.testing.assert_allclose(kernel.cdf(y, theta), expected, rtol=128 * 2.0**-53)


@pytest.mark.parametrize("scale", [1e-6, 1.0, 1e6])
@pytest.mark.parametrize("shape", [1e13, 1e16])
def test_concentrated_quantiles_preserve_order_and_scale(kernel, scale, shape):
    p = np.array([0.01, 0.1, np.nextafter(0.5, 0.0), 0.5, np.nextafter(0.5, 1.0), 0.9, 0.99])
    theta = np.tile([scale, shape], (len(p), 1))
    actual = kernel.quantile(p, theta)
    with mp.workdps(100):
        expected = np.array(
            [
                float(
                    mp.mpf(scale)
                    * mp.exp(mp.log(mp.mpf(float(q)) / (1 - mp.mpf(float(q)))) / mp.mpf(shape))
                )
                for q in p
            ]
        )
    assert np.all(np.diff(actual) >= 0)
    assert np.all(actual[p < 0.5] <= scale)
    assert np.all(actual[p > 0.5] >= scale)
    # Each exact high-precision quantile is rounded once to binary64.
    # There is no demand for unattainable CDF inversion between adjacent floats.
    np.testing.assert_array_equal(actual, expected)
