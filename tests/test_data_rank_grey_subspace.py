"""The published data rank without a TSQR of every row: the Gram, then its grey subspace.

The data rank (``RankInfo.data``: rank, representatives, estimability flags)
is decided at the factor route's resolution.  The Gram decides it when it can
certify itself; otherwise ``rank.decompose_gram_deflated`` factors only the
rows projected onto the Gram's grey subspace (``n x k``) and certifies the
rest from the Gram, and only a failed certificate pays for the TSQR of every
row, the route every such fit took before.  The fixtures reach the
Gram's uncertifiable band the ways the freMTPL2 books do: a level whose curve
has fewer distinct points than basis functions, tensor cells with three
points, and a near-collinear pair inside the band ``eps < sigma^2 /
sigma_max^2 < n eps``, where the Gram's own tolerance answers differently.
"""

import dataclasses
import pickle

import numpy as np
import pandas as pd
import pytest

import superglm.solvers.centered_system as centered_system
import superglm.solvers.rank as rank
from superglm import SuperGLM
from superglm.features.categorical import Categorical
from superglm.features.numeric import Numeric
from superglm.features.spline import Spline


def _assert_same_decisions(left, right):
    """Rank, representatives, structural aliases and every estimability flag equal."""
    assert left.rank == right.rank
    np.testing.assert_array_equal(left.active_columns, right.active_columns)
    np.testing.assert_array_equal(left.structural_aliases, right.structural_aliases)
    np.testing.assert_array_equal(left.coefficient_estimable(), right.coefficient_estimable())


def _assert_same_bits(left, right):
    """Every field equal bit for bit (array memory order aside, which pickling records)."""
    for field in dataclasses.fields(left):
        a, b = getattr(left, field.name), getattr(right, field.name)
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            assert a.dtype == b.dtype and a.shape == b.shape, field.name
            assert a.tobytes() == b.tobytes(), field.name
        else:
            assert pickle.dumps(a) == pickle.dumps(b), field.name


@pytest.fixture
def row_factor_passes(monkeypatch):
    """Count the TSQRs over the design's rows, keeping each factor."""
    factors = []
    original = centered_system._data_factor

    def counted(*args):
        out = original(*args)
        factors.append(out[0])
        return out

    monkeypatch.setattr(centered_system, "_data_factor", counted)
    return factors


def _thin_level_curve(seed=0, n=1500):
    """A spline-by-level pair whose last level holds two distinct points: its curve aliases."""
    rng = np.random.default_rng(seed)
    band = rng.integers(0, 4, n)
    z = rng.uniform(0.0, 10.0, n)
    z[band == 3] = np.where(rng.random(int(np.sum(band == 3))) < 0.5, 2.0, 7.0)
    frame = pd.DataFrame({"band": [f"L{j}" for j in band], "z": z})
    mu = np.exp(-1.0 + 0.1 * band + 0.3 * np.sin(z))
    features = {"band": Categorical(), "z": Spline(kind="cr", n_knots=6)}
    return frame, rng.poisson(mu).astype(float), features, [("z", "band")]


def _sparse_tensor_cells(seed=1, n=1500):
    """A spline-by-spline pair whose high corner holds three distinct points."""
    rng = np.random.default_rng(seed)
    x1 = rng.integers(0, 40, n).astype(float)
    x2 = rng.integers(0, 40, n).astype(float)
    corner = (x1 > 25) & (x2 > 25)
    keep = ~corner
    keep[np.flatnonzero(corner)[:30]] = True
    x1, x2 = x1[keep], x2[keep]
    pick = (x1 > 25) & (x2 > 25)
    x1[pick] = np.array([30.0, 35.0, 39.0])[np.arange(pick.sum()) % 3]
    x2[pick] = np.array([31.0, 36.0, 38.0])[np.arange(pick.sum()) % 3]
    frame = pd.DataFrame({"x1": x1, "x2": x2})
    mu = np.exp(-1.0 + 0.02 * x1 - 0.01 * x2)
    features = {"x1": Spline(kind="cr", n_knots=8), "x2": Spline(kind="cr", n_knots=8)}
    return frame, rng.poisson(mu).astype(float), features, [("x1", "x2")]


def _near_collinear_pair(seed=2, n=1500, separation=6e-8):
    """``v = u + separation * e`` with ``e`` orthogonal to ``u``: one direction in the band."""
    rng = np.random.default_rng(seed)
    u = rng.normal(size=n)
    e = rng.normal(size=n)
    e -= u * (u @ e) / (u @ u)
    e /= np.linalg.norm(e) / np.linalg.norm(u)
    z = rng.uniform(0.0, 10.0, n)
    frame = pd.DataFrame({"u": u, "v": u + separation * e, "z": z})
    mu = np.exp(-1.0 + 0.2 * u + 0.3 * np.sin(z))
    features = {"u": Numeric(), "v": Numeric(), "z": Spline(kind="cr", n_knots=6)}
    return frame, rng.poisson(mu).astype(float), features, None


FIXTURES = {
    "thin_level_curve": _thin_level_curve,
    "sparse_tensor_cells": _sparse_tensor_cells,
    "near_collinear_pair": _near_collinear_pair,
}


def _fit(kind, full_factor):
    """Fit ``kind``; ``full_factor`` fails the grey-subspace certificate, so the full factor decides."""
    frame, y, features, interactions = FIXTURES[kind]()
    model = SuperGLM(family="poisson", features=features, interactions=interactions)
    if not full_factor:
        model.fit_reml(frame, y)
        return model
    original = rank.decompose_gram_deflated
    rank.decompose_gram_deflated = lambda *args, **kwargs: (None, {})
    try:
        model.fit_reml(frame, y)
    finally:
        rank.decompose_gram_deflated = original
    return model


# The near-collinear pair is unpenalized, so its Hessian (the standard
# errors' decomposition) is certified on the row factor under either setting;
# these two kinds isolate the data rank.
PENALIZED_NULLS = ("sparse_tensor_cells", "thin_level_curve")


@pytest.mark.parametrize("kind", PENALIZED_NULLS)
def test_default_decides_the_data_rank_without_a_row_factor(kind, row_factor_passes):
    """No TSQR of every row by default, and today's decision; the base branch ran one per fit."""
    model = _fit(kind, full_factor=False)
    data = model.result.rank_info.data
    # Precondition: the Gram cannot certify itself here.
    assert data.grey_subspace is not None and data.grey_subspace["certified"] == 1.0
    assert 0 < data.grey_subspace["k"] < model._dm.p
    assert row_factor_passes == []
    assert model._reml_profile["data_rank_route"] == "grey_subspace"
    reference = _fit(kind, full_factor=True).result.rank_info.data
    _assert_same_decisions(data, reference)


@pytest.mark.parametrize("kind", sorted(FIXTURES))
def test_the_route_moves_no_fitted_value(kind):
    """The data rank is decided after the solve: coefficients, EDF and Hessian are the same bits."""
    grey, full = _fit(kind, full_factor=False), _fit(kind, full_factor=True)
    np.testing.assert_array_equal(grey.result.beta, full.result.beta)
    assert grey.result.intercept == full.result.intercept
    assert grey.result.effective_df == full.result.effective_df
    np.testing.assert_array_equal(
        grey.result.rank_info.feature_edf, full.result.rank_info.feature_edf
    )
    _assert_same_bits(grey.result.rank_info.augmented, full.result.rank_info.augmented)


def test_grey_subspace_and_gram_tolerance_side_by_side(monkeypatch):
    """On each grey-zone kind: the deflated route decides as the full factor; the Gram's tolerance may not.

    Thin levels and tensor cells are exact nulls (equilibrated eigenvalues of
    round-off, ~1e-17 of the largest), far below either cut.  The
    near-collinear pair has ``sigma^2 / sigma_max^2`` near ``1e-15``: above
    the factor's boundary ``eps`` and below the Gram's tolerance ``n eps``.
    The full factor keeps it and so does the deflated route, which factors
    those rows; ``decompose_gram`` alone drops it and reports both
    coefficients not estimable.
    """
    grams = []
    decide = rank.decide_data_rank

    def recorded(gram, *args, **kwargs):
        grams.append(np.array(gram))
        return decide(gram, *args, **kwargs)

    import superglm.solvers.irls_direct as irls_direct

    monkeypatch.setattr(irls_direct, "decide_data_rank", recorded)
    for kind in sorted(FIXTURES):
        grams.clear()
        deflated = _fit(kind, full_factor=False).result.rank_info.data
        full = _fit(kind, full_factor=True).result.rank_info.data
        _assert_same_decisions(deflated, full)
        tolerance = rank.decompose_gram(grams[0])
        if kind == "near_collinear_pair":
            assert deflated.grey_subspace is not None
            assert full.rank == full.width
            assert tolerance.rank == full.rank - 1
            assert int(np.sum(~tolerance.coefficient_estimable())) == 2
        else:
            assert tolerance.rank == full.rank


def test_a_failed_certificate_keeps_the_full_factor(monkeypatch, row_factor_passes):
    """No split for the grey subspace: the full factor decides, bit for bit as it did before."""
    monkeypatch.setattr(rank, "_GREY_FLOOR", float("inf"))
    model = _fit("thin_level_curve", full_factor=False)
    data = model.result.rank_info.data
    assert data.grey_subspace is None and data.method == "qr_svd"
    assert len(row_factor_passes) == 1
    _assert_same_bits(data, rank.decompose_factor(row_factor_passes[0]))


@pytest.mark.parametrize("discrete", [False, True], ids=["exact", "discrete"])
@pytest.mark.parametrize("kind", sorted(FIXTURES))
def test_grey_projection_is_the_rows_product(kind, discrete):
    """``grouped_projected_factor`` is ``R`` of ``A V`` and ``A'A V``, against the dense design.

    ``A = sqrt(W) (X - c)``.  Each entry of ``A'(AV)`` is within
    ``gamma_m (|X| + |c|)' W (|X| + |c|) |V|`` of the dense product, ``m``
    the routine's rounding count plus the dense reference's own ``n + p``;
    ``R'R`` adds the TSQR's backward error.  Exact and binned designs
    reach the gather, one-hot, spline-by-level and dense-rows routes.
    """
    frame, y, features, interactions = FIXTURES[kind]()
    model = SuperGLM(
        family="poisson", features=features, interactions=interactions, discrete=discrete
    )
    model.fit_reml(frame, y)
    dm = model._dm
    rng = np.random.default_rng(5)
    W = rng.uniform(0.1, 2.0, dm.n)
    X = dm.toarray()
    centre = (W @ X) / W.sum()
    basis = rng.normal(size=(dm.p, 3))
    factor, cross, roundings, _ = centered_system.grouped_projected_factor(
        dm, W, basis, center=centre
    )
    A = np.sqrt(W)[:, None] * (X - centre)
    product = A @ basis
    magnitude = (np.abs(X) + np.abs(centre)) * np.sqrt(W)[:, None]
    scale = magnitude.T @ (magnitude @ np.abs(basis))
    gamma = (roundings + dm.n + dm.p) * np.finfo(float).eps / 2
    gamma /= 1.0 - gamma
    assert np.all(np.abs(cross - A.T @ product) <= 2.0 * gamma * scale)
    # Householder QR's columnwise backward error, gamma~_(m k) (Higham 2002,
    # Thm 19.4, a small constant c; c = 4 here), on top of B's own.
    norms = np.sqrt(np.diag(np.abs(basis).T @ scale))
    qr_gamma = 4 * (dm.n * basis.shape[1] + roundings) * np.finfo(float).eps / 2
    tolerance = 4.0 * (gamma + qr_gamma) * np.outer(norms, norms)
    assert np.all(np.abs(factor.T @ factor - product.T @ product) <= tolerance)
