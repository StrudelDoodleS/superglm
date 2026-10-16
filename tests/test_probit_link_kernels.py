"""The probit link's normal cdf and pdf: binary64, accurate to a derived bound, and scipy-identical."""

import math
from decimal import Decimal, localcontext

import numpy as np
import pytest
import scipy
import scipy.stats

from superglm.links import ProbitLink, _normal_pdf

U = 2.0**-53
#: scipy releases whose ``_norm_pdf`` is ``np.exp(-x**2/2.0) / np.sqrt(2*np.pi)``
#: and whose ``_norm_cdf`` is ``ndtr`` (read in scipy/stats/_continuous_distns.py
#: at v1.15.0, v1.16.0, v1.17.0, v1.18.0). Another release needs that re-read.
_SCIPY_PARITY = {(1, 15), (1, 16), (1, 17), (1, 18)}
_PI_60 = Decimal("3.14159265358979323846264338327950288419716939937510582097494")


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


def test_the_density_is_the_in_repo_expression_bit_for_bit():
    x = _grid()
    with np.errstate(over="ignore"):
        expected = np.exp(-(x**2) / 2.0) / np.sqrt(2 * np.pi)
        actual = _normal_pdf(x)
    np.testing.assert_array_equal(_bits(actual), _bits(expected))


def test_the_density_is_within_its_rounding_bound_of_the_exact_value():
    """``|phi^ - phi| <= ((x**2 / 2) + 5.5) u |phi|`` against a 60-digit oracle, while phi is normal.

    ``fl(x*x)`` carries ``|d1| <= u`` and halving is exact, so the exponent
    is ``-(x**2 / 2)(1 + d1)`` and ``exp`` turns ``d1`` into a relative
    factor ``exp(x**2 d1 / 2)``, at most ``(x**2 / 2) u``. NumPy's accuracy
    validation set holds float64 ``exp`` within 1 ulp of the stored,
    correctly rounded value, so up to 1.5 ulp (``3u`` relative) of the exact
    one, on its sampled inputs. ``sqrt(fl(2 pi))`` carries ``u/2`` from pi
    and ``u`` from the correctly rounded sqrt, and the division ``u``. First
    order: ``(x**2/2 + 5.5) u``; the second-order terms are below ``1e-10``
    of it on this grid. phi is normal while ``x**2 <= 2 (1022 ln 2 - ln
    sqrt(2 pi))``, ``|x| <= 37.6``; past that it is subnormal and the bound
    is absolute instead, so the grid stays within ``|x| <= 37``.
    """
    rng = np.random.default_rng(37)
    x = np.concatenate([np.linspace(-37.0, 37.0, 1_001), rng.uniform(-37.0, 37.0, 500), [0.0]])
    computed = _normal_pdf(x)
    with localcontext() as context:
        context.prec = 60
        root = (2 * _PI_60).sqrt()
        for value, observed in zip(x.tolist(), computed.tolist(), strict=True):
            exact = (-(Decimal(value) * Decimal(value)) / 2).exp() / root
            bound = (value * value / 2.0 + 5.5) * U * (1.0 + 1e-9)
            assert abs(Decimal(observed) - exact) <= Decimal(bound) * exact, value


@pytest.mark.parametrize("method", ["inverse", "deriv_inverse", "deriv2_inverse", "deriv3_inverse"])
def test_an_extended_argument_is_evaluated_in_binary64(method):
    """``np.longdouble`` is cast to float64, never carried through (``AGENTS.md``).

    Mutation check: promoting with ``promote_types`` (as ``scipy.stats.norm``
    did on master) keeps ``longdouble`` where it is wider than binary64.
    """
    eta = np.linspace(-6.0, 6.0, 25)
    link = ProbitLink()
    with np.errstate(all="ignore"):
        extended = getattr(link, method)(eta.astype(np.longdouble))
        binary64 = getattr(link, method)(eta)
    assert np.asarray(extended).dtype == np.float64
    np.testing.assert_array_equal(_bits(extended), _bits(binary64))


@pytest.mark.parametrize(
    ("method", "reference"),
    [
        ("inverse", lambda eta: scipy.stats.norm.cdf(eta)),
        ("deriv_inverse", lambda eta: scipy.stats.norm.pdf(eta)),
        ("deriv2_inverse", lambda eta: -eta * scipy.stats.norm.pdf(eta)),
        ("deriv3_inverse", lambda eta: (eta**2 - 1) * scipy.stats.norm.pdf(eta)),
    ],
)
def test_probit_eta_kernels_keep_parity_with_the_scipy_stats_master_used(method, reference):
    """Master called ``scipy.stats.norm``; for the verified scipy releases the values are identical."""
    release = tuple(int(part) for part in scipy.__version__.split(".")[:2])
    if release not in _SCIPY_PARITY:
        pytest.skip(
            f"scipy {scipy.__version__}: re-read _norm_pdf/_norm_cdf before claiming parity"
        )
    eta = _grid()
    with np.errstate(invalid="ignore", over="ignore"):
        actual = getattr(ProbitLink(), method)(eta)
        expected = reference(eta)
    finite = np.isfinite(expected)
    np.testing.assert_array_equal(_bits(actual[finite]), _bits(expected[finite]))
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    tails = ~finite & ~np.isnan(expected)
    np.testing.assert_array_equal(actual[tails], expected[tails])
    scalar = getattr(ProbitLink(), method)(-1.25)
    assert type(scalar) is type(reference(-1.25))
    assert _bits(scalar) == _bits(reference(-1.25))


def test_probit_deriv_keeps_parity_with_scipy_stats():
    release = tuple(int(part) for part in scipy.__version__.split(".")[:2])
    if release not in _SCIPY_PARITY:
        pytest.skip(f"scipy {scipy.__version__}: re-read _norm_pdf before claiming parity")
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
    assert math.isfinite(float(link.inverse(0.0)))
