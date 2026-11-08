from importlib import import_module, util
from itertools import product

import mpmath as mp
import numpy as np
import pytest
from scipy.stats import fisk

from tests._declared_scalar_oracles import assert_scaled_jets, jets, log_density, log_window


@pytest.fixture(scope="module")
def kernel():
    name = "superglm.distributional.kernels.log_logistic"
    assert util.find_spec(name) is not None, "production log-logistic numerical module is missing"
    return import_module(name)


def test_log_logistic_law_and_all_mixed_jets(kernel):
    for s, a in product((1e-6, 1.0, 1e6), (0.6, 1.0, 1.5, 2.0, 4.0, 12.0)):
        law = fisk(c=a, scale=s)
        # Independent inverse in log space; scipy.ppf cancels at 1e-100.
        ps = (0.001, 0.1, 0.5, 0.9, 0.999, 1e-12, 1e-30, 1e-100)
        points = [s * np.exp((np.log(p) - np.log1p(-p)) / a) for p in ps]
        points += [s * np.exp((np.log1p(-p) - np.log(p)) / a) for p in (1e-12, 1e-30, 1e-100)]
        for index, y in enumerate(points):
            actual = kernel.density_jets(np.array([y]), np.array([[s, a]]), np.ones(1), order=4)
            assert actual.valid.all()
            ref = jets(lambda x, b: log_density("log_logistic", mp.mpf(y), x, b), (s, a), 4)
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
        reference = jets(
            lambda s, a: log_window("log_logistic", mp.mpf(lo), mp.mpf(hi), s, a), t, 4
        )
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
