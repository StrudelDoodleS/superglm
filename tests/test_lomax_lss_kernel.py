from importlib import import_module, util
from itertools import product
from math import factorial

import mpmath as mp
import numpy as np
import pytest
from scipy.stats import lomax

from tests._declared_scalar_oracles import assert_scaled_jets, jets, log_density


@pytest.fixture(scope="module")
def kernel():
    name = "superglm.distributional.kernels.lomax"
    assert util.find_spec(name) is not None, "production Lomax numerical boundary is missing"
    return import_module(name)


def test_lomax_law_and_all_mixed_jets(kernel):
    for s, a in product((1e-6, 1.0, 1e6), (0.25, 0.75, 1.0, 1.25, 2.0, 5.0, 25.0)):
        law = lomax(c=a, scale=s)
        points = [(law.ppf(p), True) for p in (0.001, 0.1, 0.5, 0.9, 0.999)]
        points += [(law.ppf(p), False) for p in (1e-12, 1e-30, 1e-100)]
        with np.errstate(over="ignore"):
            points += [(law.isf(p), False) for p in (1e-12, 1e-30, 1e-100)]
        for y, body in points:
            if not np.isfinite(y):
                # An unrepresentable quantile is a labelled range case, not an oracle point.
                assert a == 0.25 and y == np.inf
                continue
            out = kernel.density_jets(np.array([y]), np.array([[s, a]]), np.ones(1), order=4)
            assert out.valid.all(), (s, a, y)
            with mp.workdps(100):
                reference = jets(lambda x, b: log_density("lomax", mp.mpf(y), x, b), (s, a), 4)
            actual = (out.jets[0] + out.carrier, *out.jets[1:])
            assert_scaled_jets(actual, reference, (s, a))
            if body:
                np.testing.assert_allclose(
                    actual[0], law.logpdf(y), rtol=512 * 2.0**-53, atol=512 * 2.0**-53
                )
            natural_h = out.jets[2][0]
            predictor_h = natural_h * np.array([s * s, s * a, a * a])
            predictor_h[[0, 2]] += out.jets[1][0] * [s, a]
            with mp.workdps(100):
                ref_eta = jets(
                    lambda u, v: log_density("lomax", mp.mpf(y), mp.exp(u), mp.exp(v)),
                    (np.log(s), np.log(a)),
                    2,
                )[2]
            np.testing.assert_allclose(
                predictor_h, ref_eta, rtol=1024 * 2.0**-53, atol=1024 * 2.0**-53 * max(1, a)
            )


def test_lomax_zero_limit(kernel):
    for s, a in product((1e-6, 1.0, 1e6), (0.25, 1.0, 2.0, 25.0)):
        out = kernel.density_jets(np.zeros(1), np.array([[s, a]]), np.array([5.0]), order=4)
        assert out.valid.all()
        expected = [np.array([5 * (np.log(a) - np.log(s))])]
        for d in range(1, 5):
            row = np.zeros(d + 1)
            row[0] = 5 * (-1) ** d * factorial(d - 1) / s**d
            row[-1] = 5 * (-1) ** (d - 1) * factorial(d - 1) / a**d
            expected.append(row)
        assert_scaled_jets(out.jets, expected, (s, a))
        assert not out.carrier.any()


def test_lomax_lazy_orders(kernel, monkeypatch):
    from superglm.distributional.kernels._declared import lomax as generated

    original = generated.density
    seen = []

    def guarded(y, scale, shape, multiplier, order, *levels):
        assert all(len(level) == 0 for level in levels[order + 1 :])
        seen.append(order)
        return original(y, scale, shape, multiplier, order, *levels)

    monkeypatch.setattr(generated, "density", guarded)
    y, theta = np.array([0.0, 2.0]), np.array([[1.0, 0.5], [2.0, 3.0]])
    for order in range(5):
        out = kernel.density_jets(y, theta, np.ones(2), order=order)
        assert len(out.jets) == order + 1
        if order == 0:
            value, carrier = out.jets[0], out.carrier
        np.testing.assert_array_equal(out.jets[0], value)
        np.testing.assert_array_equal(out.carrier, carrier)
    assert seen == list(range(5))


def test_oracle_catches_packed_coordinate_mutations(kernel):
    theta = (3.0, 1.25)
    out = kernel.density_jets(np.array([2.0]), np.array([theta]), np.ones(1), order=2)
    ref = jets(lambda s, a: log_density("lomax", mp.mpf(2), s, a), theta, 2)
    for mutation in (np.array([[0.0, 1.0, 0.0]]), out.jets[2] * np.array([9.0, 3.75, 1.5625])):
        with pytest.raises(AssertionError):
            assert_scaled_jets((*out.jets[:2], mutation), ref, theta)


def test_metadata_and_numerical_range_admission(kernel, monkeypatch):
    from superglm.distributional.kernels._declared import lomax as generated

    assert generated.PARAMETERS == ("scale", "shape")
    assert generated.COORDINATES == "natural"
    assert generated.LAWS == ("density", "logsf", "logcdf") and generated.MAX_ORDER == 4
    # A finite zero after an intermediate power overflow is not a valid jet.
    for scale in (1e-100, 1e100):
        out = kernel.density_jets(np.array([scale]), np.array([[scale, 2.0]]), np.ones(1), order=4)
        assert not out.valid[0]
    monkeypatch.setattr(generated, "COORDINATES", "log")
    with pytest.raises(ValueError, match="natural-coordinate"):
        kernel.density_jets(np.ones(1), np.ones((1, 2)), np.ones(1), order=2)


def test_lomax_window_extremes(kernel):
    from tests._declared_scalar_oracles import log_window

    cases = [
        (0.0, np.inf),
        (0.0, 1e-100),
        (0.0, 1e-12),
        (0.0, 0.1),
        (0.2, 2.0),
        (1e50, 1e50 * (1 + 1e-8)),
        (1e50, np.inf),
        (1.0, np.nextafter(1.0, np.inf)),
    ]
    for lo, hi in cases:
        for s, a in ((1.0, 0.25), (1e6, 25.0), (1e-6, 1.25)):
            actual, valid = kernel.window_jets(
                np.array([lo]), np.array([hi]), np.array([[s, a]]), order=4
            )
            assert valid.all(), (lo, hi, s, a)
            reference = jets(
                lambda x, b: log_window("lomax", mp.mpf(lo), mp.mpf(hi), x, b), (s, a), 4
            )
            assert_scaled_jets(actual, reference, (s, a), factor=8192)
    empty, valid = kernel.window_jets(np.array([2.0]), np.array([2.0]), np.ones((1, 2)), order=2)
    assert empty[0][0] == -np.inf and not valid[0]
    # The naive CDF difference loses a representable finite window entirely.
    law = lomax(c=2.0, scale=1.0)
    assert law.cdf(2e50) - law.cdf(1e50) == 0
    actual, valid = kernel.window_jets(
        np.array([1e50]), np.array([2e50]), np.array([[1.0, 2.0]]), order=2
    )
    assert valid.all() and np.isfinite(actual[0][0])


def test_lomax_deep_left_fourth_is_not_false_analytic_evidence(kernel):
    from superglm.distributional.kernels._declared import law_jets
    from superglm.distributional.kernels._declared import lomax as generated
    from tests._declared_scalar_oracles import log_window

    y, theta = np.array([1e-100]), np.array([[1.0, 3.0]])
    raw, _ = law_jets(generated, "logcdf", 4, y, theta)
    reference = jets(lambda s, a: log_window("lomax", mp.mpf(0), mp.mpf(1e-100), s, a), theta[0], 4)
    with pytest.raises(AssertionError):
        assert_scaled_jets(raw, reference, theta[0])
    actual, valid = kernel.window_jets(np.zeros(1), y, theta, order=4)
    assert valid[0]
    assert_scaled_jets(actual, reference, theta[0], factor=8192)


def test_lomax_predictions_moments_and_units(kernel):
    alphas = np.array(
        [
            0.25,
            0.75,
            np.nextafter(1.0, 0.0),
            1.0,
            np.nextafter(1.0, np.inf),
            np.nextafter(2.0, 0.0),
            2.0,
            np.nextafter(2.0, np.inf),
            5.0,
            25.0,
        ]
    )
    theta = np.column_stack((np.full(len(alphas), 3.0), alphas))
    for name, boundary in (("mean", 1), ("variance", 2), ("expected_shortfall", 1)):
        actual = (
            kernel.expected_shortfall(0.9, theta)
            if name == "expected_shortfall"
            else getattr(kernel, name)(theta)
        )
        np.testing.assert_array_equal(np.isposinf(actual), alphas <= boundary)
        for i in np.flatnonzero(alphas > boundary):
            with mp.workdps(100):
                a, s = mp.mpf(float(alphas[i])), mp.mpf(3)
                if name == "variance":
                    expected = s * s * a / ((a - 1) ** 2 * (a - 2))
                elif name == "mean":
                    expected = s / (a - 1)
                else:
                    q = s * mp.expm1(-mp.log1p(-mp.mpf(0.9)) / a)
                    expected = q + (s + q) / (a - 1)
                np.testing.assert_allclose(actual[i], float(expected), rtol=4096 * 2.0**-53)
    for p in (0.001, 0.5, 0.9, 0.999, 1e-100):
        q = kernel.quantile(p, theta)
        np.testing.assert_allclose(kernel.cdf(q, theta), p, rtol=8192 * 2.0**-53, atol=0)
        scaled = theta.copy()
        scaled[:, 0] *= 1000
        np.testing.assert_allclose(kernel.quantile(p, scaled), 1000 * q, rtol=4096 * 2.0**-53)
    np.testing.assert_array_equal(kernel.cdf(-1.0, theta), np.zeros(len(theta)))
    np.testing.assert_array_equal(kernel.cdf(np.inf, theta), np.ones(len(theta)))
    for p in (0, 1, -0.1, np.nan):
        with pytest.raises(ValueError, match="probabilit"):
            kernel.quantile(p, theta)
    with pytest.raises(ValueError, match="NaN"):
        kernel.cdf(np.nan, theta)
    with pytest.raises(ValueError, match="finite.*numerical range"):
        kernel.mean(np.array([[1e308, np.nextafter(1.0, np.inf)]]))
    with pytest.raises(ValueError, match="finite.*numerical range"):
        kernel.quantile(0.9, np.array([[1.0, 0.001]]))
    # The intermediate exp(t) overflows, but scale*expm1(t) fits float64.
    actual = kernel.quantile(0.9, np.array([[1e-200, 0.003]]))[0]
    with mp.workdps(100):
        expected = float(mp.mpf(1e-200) * mp.expm1(-mp.log1p(-mp.mpf(0.9)) / mp.mpf(0.003)))
    np.testing.assert_allclose(actual, expected, rtol=8192 * 2.0**-53)


def test_lomax_fisher_matches_integrated_score_outer_product(kernel):
    for s, a in ((0.3, 0.75), (2.0, 1.0), (7.0, 3.0)):
        with mp.workdps(60):
            scale, shape = mp.mpf(s), mp.mpf(a)
            # t=log(1+y/s) is exponential(rate=a). Independent score outer product.
            score = (
                lambda t: (-1 + (shape + 1) * (1 - mp.exp(-t))) / scale,
                lambda t: 1 / shape - t,
            )
            expected = [
                float(
                    mp.quad(
                        lambda t: score[i](t) * score[j](t) * shape * mp.exp(-shape * t),
                        [0, mp.inf],
                    )
                )
                for i, j in ((0, 0), (0, 1), (1, 1))
            ]
        actual = kernel.expected_information(np.array([[s, a]]), np.array([5.0]))[0]
        np.testing.assert_allclose(actual, 5 * np.array(expected), rtol=1024 * 2.0**-53, atol=0)
        assert actual[0] * actual[2] - actual[1] ** 2 > 0


def test_lomax_expected_shortfall_independent_survival_integral(kernel):
    for a in (1.01, 1.5, 5.0):
        for p in (0.1, 0.9, 0.999):
            with mp.workdps(70):
                s, shape, prob = mp.mpf(2), mp.mpf(a), mp.mpf(p)
                q = s * mp.expm1(-mp.log1p(-prob) / shape)
                # y=s*(exp(t)-1): integrate S(y) dy, without using ES identity.
                start = mp.log1p(q / s)
                integrated = mp.quad(lambda t: s * mp.exp((1 - shape) * t), [start, mp.inf])
                expected = float(q + integrated / (1 - prob))
            np.testing.assert_allclose(
                kernel.expected_shortfall(p, np.array([[2.0, a]])), expected, rtol=4096 * 2.0**-53
            )


def test_lomax_quantile_avoids_premature_probability_scale_underflow(kernel):
    # -log(1-p)/a underflows, but rescaling by the monetary scale produces a
    # representable quantile. Log-space evaluation has O(log(range)*u) error.
    theta, p = np.array([[1e300, 1e200]]), 1e-200
    with mp.workdps(100):
        expected = float(mp.mpf(1e300) * mp.expm1(-mp.log1p(-mp.mpf(p)) / mp.mpf(1e200)))
    q = kernel.quantile(p, theta)
    np.testing.assert_allclose(q, expected, rtol=8192 * 2.0**-53, atol=0)
    np.testing.assert_allclose(kernel.cdf(q, theta), p, rtol=8192 * 2.0**-53, atol=0)
    assert -np.log1p(-p) / theta[0, 1] == 0.0
