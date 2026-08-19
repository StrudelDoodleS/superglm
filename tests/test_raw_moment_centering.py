"""General raw-moment centering rung (RFC-1).

The centering ladder's fast rungs all require particular group-matrix types.
Designs outside those sets fall to a chunked-dense pass that re-materialises the
(n, p) design in blocks -- measured at ~47% of a plain spline REML fit. The
per-block moment dispatch already computes the same raw quantities far more
cheaply, so the general rung feeds those through the shared scaling certificate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from superglm import SuperGLM
from superglm.features.categorical import Categorical
from superglm.features.spline import Spline
from superglm.solvers.centered_system import build_centered_system


def _fitted_design(n=20_000, seed=0, with_numeric=False):
    rng = np.random.default_rng(seed)
    # Continuous, not integer-valued: an axis on few distinct values is
    # losslessly row-compressed, which makes the whole design eligible for the
    # packed rung above -- and the packed rung would then answer before this
    # one ever ran.  The general raw-moment rung exists for the ordinary
    # spline design that no support-indexed rung can serve, so the fixture has
    # to be one.
    frame = pd.DataFrame(
        {
            "a": rng.uniform(18.0, 90.0, n),
            "b": rng.uniform(50.0, 130.0, n),
            "cat": rng.choice(list("XYZ"), n),
        }
    )
    features = {
        "a": Spline(kind="ps", k=8),
        "b": Spline(kind="ps", k=8),
        "cat": Categorical(base="first"),
    }
    if with_numeric:
        # Deliberately ill-located: mean far exceeds the centred RMS, which is
        # what the scaling certificate exists to reject.
        frame["power"] = rng.normal(9.0, 3.0, n)
        from superglm.features.numeric import Numeric

        features["power"] = Numeric()
    weights = rng.uniform(0.4, 1.0, n)
    response = rng.poisson(0.2, n) / weights
    model = SuperGLM(family="poisson", selection_penalty=None, discrete=False, features=features)
    model._build_design_matrix(frame, response, weights, None)
    return model, weights


def _systems(model, weights, seed=1):
    """Build the centered system with the rung enabled and with it disabled."""
    rng = np.random.default_rng(seed)
    n, p = model._dm.shape
    W = np.abs(rng.normal(1.0, 0.2, n)) * weights
    z_off = rng.normal(size=n)
    penalty = np.zeros((p, p))
    return dict(dm=model._dm, W=W, z_off=z_off, penalty=penalty)


def test_raw_moment_rung_matches_chunked_centering():
    model, weights = _fitted_design()
    kwargs = _systems(model, weights)

    profile: dict = {}
    fast = build_centered_system(**kwargs, profile=profile)
    slow = build_centered_system(**kwargs, profile={}, _force_chunked=True)

    np.testing.assert_allclose(fast.data_gram, slow.data_gram, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(fast.rhs, slow.rhs, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(fast.mean_x, slow.mean_x, rtol=1e-9, atol=1e-9)
    assert profile.get("centered_raw_moment_hits", 0) >= 1, (
        f"general raw-moment rung did not fire; profile={profile}"
    )


def test_raw_moment_rung_declines_for_ill_located_columns():
    """The certificate must still reject what it was built to reject."""
    model, weights = _fitted_design(with_numeric=True)
    kwargs = _systems(model, weights)

    profile: dict = {}
    fast = build_centered_system(**kwargs, profile=profile)
    slow = build_centered_system(**kwargs, profile={}, _force_chunked=True)

    np.testing.assert_allclose(fast.data_gram, slow.data_gram, rtol=1e-7, atol=1e-7)


def test_a_compressible_spline_design_is_served_by_the_packed_rung_instead():
    """The counterpart to the fixture note above.

    When every axis lands on few distinct values the blocks are losslessly
    row-compressed, the packed rung accepts the design, and the raw-moment
    rung is never consulted.  That is the intended ordering -- packed is the
    cheaper build and needs no location certificate -- so it is pinned here
    rather than left as an accident of fixture data.
    """
    rng = np.random.default_rng(0)
    n = 20_000
    frame = pd.DataFrame(
        {
            "a": rng.integers(18, 90, n).astype(float),
            "b": rng.integers(50, 130, n).astype(float),
            "cat": rng.choice(list("XYZ"), n),
        }
    )
    weights = rng.uniform(0.4, 1.0, n)
    response = rng.poisson(0.2, n) / weights
    model = SuperGLM(
        family="poisson",
        selection_penalty=None,
        discrete=False,
        features={
            "a": Spline(kind="ps", k=8),
            "b": Spline(kind="ps", k=8),
            "cat": Categorical(base="first"),
        },
    )
    model._build_design_matrix(frame, response, weights, None)
    kwargs = _systems(model, weights)

    profile: dict = {}
    fast = build_centered_system(**kwargs, profile=profile)
    slow = build_centered_system(**kwargs, profile={}, _force_chunked=True)

    np.testing.assert_allclose(fast.data_gram, slow.data_gram, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(fast.rhs, slow.rhs, rtol=1e-9, atol=1e-9)
    assert profile.get("centered_raw_moment_hits", 0) == 0
