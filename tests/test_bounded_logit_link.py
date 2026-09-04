"""Contract tests for the reusable two-wall logit link."""

from __future__ import annotations

import math

import numpy as np
import pytest

from superglm.distributional.families._links import BoundedLogitLink
from superglm.links import Link

_WALLS = [(0.0, 1.0), (0.2, 0.3), (-0.9, 0.9), (0.0, 0.5), (0.5, 1.0)]


def _rel(a, b):
    return abs(a - b) / (1.0 + abs(b))


def test_the_link_satisfies_the_link_protocol_and_is_frozen():
    link = BoundedLogitLink(0.0, 1.0)
    assert isinstance(link, Link)
    assert all(hasattr(link, name) for name in ("deriv2_inverse", "deriv3_inverse"))
    with pytest.raises(Exception):
        link.lower = 0.5  # type: ignore[misc]
    assert BoundedLogitLink(0.0, 1.0) == BoundedLogitLink(0.0, 1.0)
    assert BoundedLogitLink(0.0, 1.0) != BoundedLogitLink(0.0, 0.5)


@pytest.mark.parametrize(("lower", "upper"), _WALLS)
def test_the_inverse_stays_strictly_inside_the_walls_at_the_support_probe(lower, upper):
    link = BoundedLogitLink(lower, upper)
    eta = np.array([-20.0, -2.0, 0.0, 2.0, 20.0])
    values = link.inverse(eta)
    assert values.shape == eta.shape
    assert np.all(values > lower) and np.all(values < upper)
    assert np.all(np.diff(values) > 0.0)


def test_the_default_walls_reproduce_the_measured_edge_values():
    values = BoundedLogitLink(0.0, 1.0).inverse(np.array([-20.0, 20.0]))
    assert float(values[0]) == 2.0611536181902037e-09
    assert float(values[1]) == 0.99999999793884631


@pytest.mark.parametrize(("lower", "upper"), _WALLS)
@pytest.mark.parametrize("eta", [-8.0, -3.0, -0.7, 0.0, 1.3, 4.0, 9.0])
def test_inverse_derivatives_match_mpmath(lower, upper, eta):
    mp = pytest.importorskip("mpmath")
    link = BoundedLogitLink(lower, upper)
    with mp.workdps(50):
        span = mp.mpf(upper) - mp.mpf(lower)

        def inverse(value):
            return mp.mpf(lower) + span / (1 + mp.exp(-value))

        references = [float(mp.diff(inverse, mp.mpf(eta), order)) for order in (1, 2, 3)]
    got = (
        float(link.deriv_inverse(np.array([eta]))[0]),
        float(link.deriv2_inverse(np.array([eta]))[0]),
        float(link.deriv3_inverse(np.array([eta]))[0]),
    )
    for value, reference in zip(got, references, strict=True):
        assert _rel(value, reference) <= 5.0e-16


@pytest.mark.parametrize(("lower", "upper"), _WALLS)
def test_link_inverts_the_inverse_and_deriv_inverts_deriv_inverse(lower, upper):
    link = BoundedLogitLink(lower, upper)
    eta = np.array([-7.0, -1.0, 0.0, 0.4, 6.0])
    values = link.inverse(eta)
    assert np.allclose(link.link(values), eta, rtol=0, atol=1e-12)
    assert np.allclose(link.deriv(values) * link.deriv_inverse(eta), 1.0, rtol=0, atol=1e-12)


def test_the_link_refuses_inputs_outside_the_open_interval():
    link = BoundedLogitLink(0.0, 1.0)
    for outside in (0.0, 1.0, -0.5, 1.5, math.nan, math.inf):
        with pytest.raises(ValueError, match="between"):
            link.link(np.array([outside]))
        with pytest.raises(ValueError, match="between"):
            link.deriv(np.array([outside]))


def test_walls_must_be_finite_ordered_and_wide_enough_to_stay_open():
    for bad in ((1.0, 0.0), (0.5, 0.5), (math.nan, 1.0), (0.0, math.inf)):
        with pytest.raises(ValueError, match="wall"):
            BoundedLogitLink(*bad)
    with pytest.raises(ValueError, match="wall"):
        BoundedLogitLink(0.5, float(np.nextafter(0.5, 1.0)))
    with pytest.raises(ValueError, match="wall"):
        BoundedLogitLink(True, 1.0)  # type: ignore[arg-type]


def test_the_link_is_exported_from_the_families_package():
    from superglm.distributional import families

    assert families.BoundedLogitLink is BoundedLogitLink
    assert "BoundedLogitLink" in families.__all__
