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
    assert row_factor_passes == []
    data = model.result.rank_info.data
    # Precondition: the Gram cannot certify itself here.
    assert data.grey_subspace is not None and data.grey_subspace["certified"] == 1.0
    assert 0 < data.grey_subspace["k"] < model._dm.p
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
    """On each grey-zone kind: the deflated route decides as the full factor; the Gram cannot.

    Thin levels and tensor cells are exact nulls (equilibrated eigenvalues of
    round-off, ~1e-17 of the largest), far below either cut.  The
    near-collinear pair has ``sigma^2 / sigma_max^2`` near ``1e-15``: above
    the factor's boundary ``eps`` and below the Gram's resolution ``n eps``.
    The full factor keeps it and so does the deflated route, which factors
    those rows; the Gram cannot certify that direction, so it is not
    authoritative there.  Which side of its own cut ``eigh`` puts an
    eigenvalue below its resolution is round-off, so that is not asserted.
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
            assert rank.decompose_gram_if_authoritative(grams[0], psd_by_construction=True) is None
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


def _coupled_gram(grey_ratio=None, coupling=3e-12, n=40, seed=0):
    """Rows ``A`` with one exact alias, columns 4 and 5, and a Gram coupled across the split.

    ``A`` has squared singular values ``1, 0.8, 0.6, 0.5`` and ``1e-6`` (the
    split, ``w``), or ``1e-6`` and a grey value placed at ``grey_ratio`` of
    the largest after Jacobi scaling, and ``0`` on ``y = (e4 - e5) /
    sqrt(2)``.  The Gram is ``A'A + c (w y' + y w')``: a formation error
    within its bound (``rows`` below), which turns the Gram's null vector
    towards ``w`` by about ``c / 1e-6``, on columns 0 to 3.  ``project`` is
    the exact rows' product.
    """
    rng = np.random.default_rng(seed)
    y = np.zeros(6)
    y[4], y[5] = np.sqrt(0.5), -np.sqrt(0.5)
    vectors = np.linalg.qr(np.column_stack((y, rng.normal(size=(6, 5)))))[0]
    vectors[:, 0] = y
    left = np.linalg.qr(rng.normal(size=(n, 6)))[0]
    squared = np.array([0.0, 1e-6, 0.5, 0.6, 0.8, 1.0])
    if grey_ratio is not None:
        squared = np.array([0.0, grey_ratio, 1e-6, 0.6, 0.8, 1.0])
    for _ in range(4 if grey_ratio is not None else 1):
        A = left @ np.diag(np.sqrt(squared)) @ vectors.T
        if grey_ratio is not None:
            # The column scaling moves the grey ratio; it barely depends on it.
            values = np.linalg.svd(A / np.linalg.norm(A, axis=0), compute_uv=False) ** 2
            squared[1] *= grey_ratio / (values[-2] / values[0])
    w = vectors[:, 2 if grey_ratio is not None else 1]
    gram = A.T @ A + coupling * (np.outer(w, y) + np.outer(y, w))

    def project(basis):
        B = A @ basis
        return np.linalg.qr(B, mode="r"), A.T @ B, n, np.linalg.norm(A, axis=0)

    return A, gram, project


def test_the_published_null_basis_is_the_rows_not_the_grams():
    """The surrogate publishes the full factor's estimability flags under a coupled Gram.

    The Gram's null vector carries ``3e-6`` of ``w`` on the estimable
    columns 0 to 3; a surrogate whose null space is the Gram's grey
    eigenvectors (005481cd) flags every coefficient not estimable, though
    its rank certificate passes.  The first-order correction ``V2 n - V1
    Lambda1^-1 E n`` takes that component out with the coupling ``E`` the
    projection already formed.
    """
    A, gram, project = _coupled_gram()
    deflated, info = rank.decompose_gram_deflated(gram, project, rows=10**6)
    full = rank.decompose_factor(A)
    assert deflated is not None
    # Precondition: the uncorrected rotation is far above the flag tolerance.
    assert info["rotation"] > 100 * rank.SHARED_RANK_POLICY.factor_rcond
    _assert_same_decisions(deflated, full)
    np.testing.assert_array_equal(
        deflated.coefficient_estimable(), [True, True, True, True, False, False]
    )


def test_the_certificate_refuses_a_grey_value_within_its_shift():
    """A grey singular value whose square is within the coupling's shift of the cut: the full factor decides.

    The grey direction sits at ``(1 + 1e-3)`` times the cut, ``2e-19`` of
    the largest from it, against a shift of ``||E||^2 / gap`` near
    ``1e-17``.  Mutation check: certifying regardless (the margin test
    replaced by ``False``) fails this.
    """
    cut = rank.SHARED_RANK_POLICY.factor_rcond**2
    _, gram, project = _coupled_gram(grey_ratio=cut * (1.0 + 1e-3))
    deflated, info = rank.decompose_gram_deflated(gram, project, rows=10**6)
    assert deflated is None
    assert info["k"] == 2.0 and info["margin"] <= info["shift"]


def test_rows_louder_than_the_grams_split_leave_no_gap():
    """Grey rows whose singular value reaches the retained block's give no gap: the full factor decides.

    ``project`` returns the grey block of rows that disagree with the Gram,
    ``1e4`` times louder, so ``sigma_grey^2`` passes the smallest retained
    eigenvalue and the gap between the blocks is negative.  Mutation check:
    without the gap refusal the quadratic shift divides by zero.
    """
    _, gram, project = _coupled_gram(grey_ratio=1e-12, coupling=0.0)

    def louder(basis):
        R, cross, n, norms = project(basis)
        return 1e4 * R, cross, n, norms

    deflated, info = rank.decompose_gram_deflated(gram, louder, rows=10**6)
    assert deflated is None
    assert info["gap"] <= 0.0


@pytest.mark.parametrize("case", ["certified", "residual_only", "too_loose"])
def test_the_certificate_carries_the_structural_residual(case):
    """A structural null whose bound reaches the cut sends the decision to the full factor.

    The alias ``y`` passes as a structural null with bound ``||A y|| <=
    bound``.  At bound 0 the grey subspace (a ``1e-12`` direction) decides
    as the full factor; at bound 1, ``bound^2`` is above ``cut_low`` and the
    certificate refuses.  At 4 ``factor_rcond`` the grey margin still clears
    the shift, so only the residual clause refuses.  Mutation checks:
    certifying regardless fails the last two; deleting only the residual
    clause fails ``residual_only``.
    """
    bound = {"certified": 0.0, "too_loose": 1.0}.get(case, 4 * rank.SHARED_RANK_POLICY.factor_rcond)
    A, gram, project = _coupled_gram(grey_ratio=1e-12, coupling=0.0)
    structural = [(slice(4, 6), np.array([[np.sqrt(0.5)], [-np.sqrt(0.5)]]), bound)]
    deflated, info = rank.decompose_gram_deflated(gram, project, rows=10**6, structural=structural)
    assert info["structural"] == 1.0
    if case == "certified":
        assert deflated is not None
        _assert_same_decisions(deflated, rank.decompose_factor(A))
    else:
        assert deflated is None
        assert info["structural_residual"] >= rank.SHARED_RANK_POLICY.factor_rcond
    if case == "residual_only":
        assert info["margin"] > info["shift"]


@pytest.mark.parametrize("blocks", ["one_block", "many_blocks"])
@pytest.mark.parametrize("discrete", [False, True], ids=["exact", "discrete"])
@pytest.mark.parametrize("kind", sorted(FIXTURES))
def test_grey_projection_is_the_rows_product(kind, discrete, blocks, monkeypatch):
    """``grouped_projected_factor`` is ``R`` of ``A V`` and ``A'A V``, against the dense design.

    ``A = sqrt(W) (X - c)``.  Each entry of ``A'(AV)`` is within
    ``gamma_m (|X| + |c|)' W (|X| + |c|) |V|`` of the dense product, ``m``
    the routine's rounding count plus the dense reference's own ``n + p``;
    ``R'R`` adds the TSQR's backward error.  Exact and binned designs
    reach the gather, one-hot, spline-by-level and dense-rows routes.
    ``many_blocks`` cuts the rows into 64-row blocks and 256-row spans
    (about 24 and 6 here), so the per-block row offsets and the span flush
    run.  Mutation check: dropping ``- offset`` in ``_gather_table_rows_at``
    fails it.
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
    if blocks == "many_blocks":
        monkeypatch.setattr(centered_system, "_GREY_LEAF_ROWS", 64)
        monkeypatch.setattr(centered_system, "_GREY_SPAN_ROWS", 256)
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


@pytest.mark.parametrize("discrete", [False, True], ids=["exact", "discrete"])
def test_structural_nulls_are_nulls_of_the_rows(discrete):
    """A group's occupied support rows give ``A N_g = 0``; each bound covers ``||A N_g||_F``.

    The thin level's curve sits on two points, so its block holds such nulls
    (the tensor fixture's one null runs across groups and is the grey
    subspace's), and the dense design checks every bound.  Mutation check:
    reading all support rows, occupied or not, finds none here.
    """
    frame, y, features, interactions = FIXTURES["thin_level_curve"]()
    model = SuperGLM(
        family="poisson", features=features, interactions=interactions, discrete=discrete
    )
    model.fit_reml(frame, y)
    dm = model._dm
    W = np.random.default_rng(3).uniform(0.1, 2.0, dm.n)
    X = dm.toarray()
    centre = (W @ X) / W.sum()
    found = centered_system.grouped_structural_nulls(dm, W, center=centre)
    assert sum(null.shape[1] for _, null, _ in found) > 0
    A = np.sqrt(W)[:, None] * (X - centre)
    for columns, null, bound in found:
        np.testing.assert_allclose(null.T @ null, np.eye(null.shape[1]), atol=1e-12)
        assert np.linalg.norm(A[:, columns] @ null) <= bound
