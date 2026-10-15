"""The probit link's normal cdf and pdf are scipy.stats.norm's float64 values, without scipy.stats."""

import numpy as np
import pytest
import scipy.stats

from superglm.links import ProbitLink


def _grid():
    """Ordinary values, the +/-20 eta band edges, the mean clips' etas, signed zeros and tails."""
    band = 20.0
    edges = [band, np.nextafter(band, 0.0), np.nextafter(band, np.inf)]
    clips = [scipy.stats.norm.ppf(c) for c in (1e-7, 1 - 1e-7, 1e-15, 1 - 1e-15)]
    special = np.array(
        edges + [-e for e in edges] + clips + [0.0, -0.0, 5e-324, 38.5, -38.5, 1e300]
    )
    rng = np.random.default_rng(20261015)
    return np.concatenate(
        [
            np.linspace(-40.0, 40.0, 20_001),
            special,
            rng.normal(scale=8.0, size=5_000),
            [np.inf, -np.inf],
        ]
    )


def _bits(values):
    return np.asarray(values, dtype=np.float64).view(np.uint64)


@pytest.mark.parametrize(
    ("method", "reference"),
    [
        ("inverse", lambda eta: scipy.stats.norm.cdf(eta)),
        ("deriv_inverse", lambda eta: scipy.stats.norm.pdf(eta)),
        ("deriv2_inverse", lambda eta: -eta * scipy.stats.norm.pdf(eta)),
        ("deriv3_inverse", lambda eta: (eta**2 - 1) * scipy.stats.norm.pdf(eta)),
    ],
)
def test_probit_eta_kernels_match_scipy_stats_bit_for_bit(method, reference):
    eta = _grid()
    with np.errstate(invalid="ignore", over="ignore"):
        actual = getattr(ProbitLink(), method)(eta)
        expected = reference(eta)
    finite = np.isfinite(expected)
    np.testing.assert_array_equal(_bits(actual[finite]), _bits(expected[finite]))
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_array_equal(
        actual[~finite & ~np.isnan(expected)], expected[~finite & ~np.isnan(expected)]
    )
    scalar = getattr(ProbitLink(), method)(-1.25)
    assert type(scalar) is type(reference(-1.25))
    assert _bits(scalar) == _bits(reference(-1.25))


def test_probit_deriv_matches_scipy_stats_bit_for_bit():
    mu = np.r_[np.linspace(0.0, 1.0, 10_001), 1e-7, 1 - 1e-7, 1e-15, 1 - 1e-15, 1e-300]
    norm = scipy.stats.norm
    expected = 1.0 / norm.pdf(norm.ppf(np.clip(mu, 1e-15, 1 - 1e-15)))
    np.testing.assert_array_equal(_bits(ProbitLink().deriv(mu)), _bits(expected))


def test_probit_eta_kernels_do_not_route_through_scipy_stats(monkeypatch):
    """Mutation check: master's ``norm.cdf``/``norm.pdf`` calls fail this test."""

    def refused(*args, **kwargs):
        raise AssertionError("probit eta kernel called scipy.stats.norm")

    monkeypatch.setattr(scipy.stats.norm, "cdf", refused)
    monkeypatch.setattr(scipy.stats.norm, "pdf", refused)
    link, eta = ProbitLink(), np.linspace(-6.0, 6.0, 13)
    for method in ("inverse", "deriv_inverse", "deriv2_inverse", "deriv3_inverse"):
        getattr(link, method)(eta)
