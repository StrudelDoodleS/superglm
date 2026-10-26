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

    import superglm.solvers.irls_direct._fit as irls_direct

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
    eigenvectors flags every coefficient not estimable, though
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


@pytest.mark.parametrize("bound", [0.0, 1.0])
def test_every_uncertain_direction_structural_reads_no_row(bound):
    """No grey direction once the structural nulls are deflated: decided with no pass over the rows.

    The alias ``y`` is the only direction below the floor, and it is
    structural; the next eigenvalue, ``1e-6``, clears the floor and the cut.
    So rank ``n_active - s`` is certified from the Gram and the residual
    bound, and the projection is never called (``k = 0``).  A residual bound
    above the cut still refuses.  Mutation: on the base the empty grey
    subspace returned ``None``, so the full factor ran.
    """
    A, gram, project = _coupled_gram(coupling=0.0)
    calls = []

    def counted(basis):
        calls.append(basis.shape[1])
        return project(basis)

    structural = [(slice(4, 6), np.array([[np.sqrt(0.5)], [-np.sqrt(0.5)]]), bound)]
    deflated, info = rank.decompose_gram_deflated(gram, counted, rows=10**6, structural=structural)
    assert calls == []
    assert info["k"] == 0.0 and info["structural"] == 1.0
    if bound:
        assert deflated is None
        return
    assert deflated is not None and deflated.grey_subspace["k"] == 0.0
    _assert_same_decisions(deflated, rank.decompose_factor(A))
    assert deflated.rank == 5


def test_the_grams_eigenpairs_are_read_not_recomputed(monkeypatch):
    """With no structural null, the grey check reads the eigenpairs the authority check computed.

    One ``eigh`` of the equilibrated Gram serves both, and the decision is
    the full factor's.  Mutation: the base computed it twice.
    """
    A, gram, project = _coupled_gram(grey_ratio=1e-12, coupling=0.0)
    calls = []
    eigh = np.linalg.eigh

    def counted(matrix, *args, **kwargs):
        calls.append(matrix.shape)
        return eigh(matrix, *args, **kwargs)

    monkeypatch.setattr(rank.np.linalg, "eigh", counted)
    decided = rank.decide_data_rank(
        gram, lambda: pytest.fail("ran the full factor"), project=project, rows=10**6
    )
    assert calls == [(6, 6)]
    assert decided.grey_subspace is not None
    _assert_same_decisions(decided, rank.decompose_factor(A))


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


def test_chunked_projection_is_the_rows_product(monkeypatch):
    """``chunked_projected_factor`` is ``R`` of ``A V`` and ``A'A V``, against the dense design.

    ``A = sqrt(W) ((X - c) - c_lo)`` with the exact centre pair of two
    columns at 1e16 (``c_lo`` nonzero) and a third that is not, in 25 blocks
    of 32 rows, so the per-block centring, both ``sqrt(w)`` scalings and the
    block-order sum all run.  The aliased columns leave ``A V`` of order one,
    so ``A'(A V)`` is not a rounding.  The rows are centred first, so each
    entry of ``A'(AV)`` is within ``gamma_m |A|' |A| |V|`` of the dense
    product, ``m`` the routine's rounding count plus the reference's own
    ``n + p``; ``R'R`` adds the TSQR's backward error.  Mutation checks (each
    fails it): dropping one ``sqrt(w)`` factor from ``cross``, and centring
    without ``center_lo``.
    """
    from superglm.inference import _metrics_design
    from superglm.inference import metrics as metrics_module

    n = 800
    z = np.tile([-2.0, 0.0, 2.0, 4.0], n // 4)
    s = np.tile([0.1, 0.7, 0.3, 0.9, 0.5], n // 5)
    X = np.column_stack((1e16 + z, 1e16 + z + 2.0, s))
    W = np.random.default_rng(7).uniform(0.1, 2.0, n)
    center, center_lo = metrics_module._centre_pair(X, W)
    assert center_lo is not None and np.any(center_lo != 0.0)
    monkeypatch.setattr(_metrics_design, "_bounded_chunk_rows", lambda **_: 32)
    basis = np.random.default_rng(5).normal(size=(3, 2))
    factor, cross, roundings, _ = centered_system.chunked_projected_factor(
        lambda: _metrics_design.iter_dense_chunks(X),
        n,
        W,
        basis,
        center=center,
        center_lo=center_lo,
    )
    A = np.sqrt(W)[:, None] * ((X - center) - center_lo)
    product = A @ basis
    scale = np.abs(A).T @ (np.abs(A) @ np.abs(basis))
    gamma = (roundings + n + 3) * np.finfo(float).eps / 2
    gamma /= 1.0 - gamma
    assert np.linalg.norm(A.T @ product) > 1.0
    assert np.all(np.abs(cross - A.T @ product) <= 2.0 * gamma * scale)
    norms = np.sqrt(np.diag(np.abs(basis).T @ scale))
    qr_gamma = 4 * (n * basis.shape[1] + roundings) * np.finfo(float).eps / 2
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


def test_a_near_structural_null_stays_with_its_coupling():
    """A near alias in one support turns the null basis through another's weak direction.

    Supports ``[u, u + 2e-14 v]`` and ``[z, z + 1e-7 v, w]`` (centred
    orthonormal columns, unit weights): the rows' one null is ``(-1, 1,
    2e-7, -2e-7, 0)``, so the full factor flags columns 2 and 3 not
    estimable, 9 tolerances clear.  The first block's ``2e-14`` null passed
    the support's ``n_s eps`` cut; deflated, it left the surrogate's null
    exactly on columns 0 and 1 and flagged 2 and 3 estimable.  Its residual
    is about 18 times its own rounding, so it stays in the Gram and the grey
    block resolves the pair.  Mutation check: deflating every candidate
    that passes the support's ``n_s eps`` cut flags ``[F, F, T, T, T]``.
    """
    from superglm._group_matrix._group_matrix_discretized import DiscretizedSSPGroupMatrix
    from superglm.group_matrix import DesignMatrix

    n = 100
    rng = np.random.default_rng(0)
    u, v, z, w = np.linalg.qr(np.column_stack((np.ones(n), rng.normal(size=(n, 4)))))[0][:, 1:].T
    rows = np.arange(n)
    supports = (np.column_stack((u, u + 2e-14 * v)), np.column_stack((z, z + 1e-7 * v, w)))
    groups = [DiscretizedSSPGroupMatrix(S, np.eye(S.shape[1]), rows) for S in supports]
    dm = DesignMatrix(groups, n, 5)
    W = np.ones(n)
    system = centered_system.build_centered_system(
        dm=dm, W=W, z_off=np.zeros(n), penalty=np.zeros((5, 5))
    )
    centre, lo = system.centre_pair()
    full = rank.decompose_factor(
        centered_system.grouped_weighted_factor(dm, W, center=centre, center_lo=lo)
    )
    decided = rank.decide_data_rank(
        system.data_gram,
        lambda: full,
        project=lambda basis: centered_system.grouped_projected_factor(
            dm, W, basis, center=centre, center_lo=lo
        ),
        structural=lambda: centered_system.grouped_structural_nulls(
            dm, W, center=centre, center_lo=lo
        ),
        rows=n,
    )
    np.testing.assert_array_equal(full.coefficient_estimable(), [False] * 4 + [True])
    assert decided.grey_subspace is not None
    _assert_same_decisions(decided, full)
    assert decided.grey_subspace["structural"] == 0.0 and decided.grey_subspace["k"] == 2.0


def _forced_full_factor(monkeypatch):
    """Fail every grey-subspace certificate from here on: the full factor decides."""
    monkeypatch.setattr(rank, "decompose_gram_deflated", lambda *args, **kwargs: (None, {}))


@pytest.mark.parametrize("rows", ["new_rows", "new_weights"])
def test_metrics_decides_the_data_rank_as_the_fit_does(rows, monkeypatch):
    """``metrics()`` off the fit's geometry decides its data rank by the fit's route, no row factor.

    New rows read an evaluation design (dense blocks, ``chunked_projected_factor``);
    the fit's rows under other weights read the grouped design.  Each makes
    no TSQR of every row, and its decisions are the full factor's.
    Mutation: on the base both ran ``streamed_weighted_factor`` over every row.
    """
    import superglm.inference.metrics as metrics_module

    model = _fit("thin_level_curve", full_factor=False)
    frame, y, _, _ = _thin_level_curve()
    weights = None
    if rows == "new_rows":
        frame, y = frame.iloc[::2].reset_index(drop=True), y[::2]
    else:
        weights = np.linspace(0.7, 1.3, len(y))
    decided, passes = [], []
    decide, factor = metrics_module.decide_data_rank, metrics_module.streamed_weighted_factor
    monkeypatch.setattr(
        metrics_module,
        "decide_data_rank",
        lambda *args, **kwargs: decided.append(decide(*args, **kwargs)) or decided[-1],
    )
    monkeypatch.setattr(
        metrics_module,
        "streamed_weighted_factor",
        lambda *args, **kwargs: passes.append(1) or factor(*args, **kwargs),
    )
    model.metrics(frame, y, sample_weight=weights).coefficient_se
    assert passes == [] and decided[0].grey_subspace is not None
    _forced_full_factor(monkeypatch)
    model.metrics(frame, y, sample_weight=weights).coefficient_se
    assert len(passes) == 1
    _assert_same_decisions(decided[0], decided[1])


def test_rebuilt_inference_state_decides_the_data_rank_as_the_fit_does(monkeypatch):
    """A fit without rank metadata (``state_ops._legacy_active_state``) takes the fit's route.

    It is what a shape repair rebuilds; its data rank is the fit's, from the
    grey subspace.  Mutation: on the base it factored every row.
    """
    from superglm.model import state_ops

    model = _fit("thin_level_curve", full_factor=False)
    solver = model._solver_pirls_result()
    W = state_ops._solver_space_working_weights(model)
    data = state_ops._legacy_active_state(model, solver, W)[-1]
    assert data.grey_subspace is not None
    _forced_full_factor(monkeypatch)
    reference = state_ops._legacy_active_state(model, solver, W)[-1]
    assert reference.grey_subspace is None and reference.method == "qr_svd"
    _assert_same_decisions(data, reference)
    _assert_same_decisions(data, model.result.rank_info.data)


def test_unpenalized_fits_keep_the_data_rank_on_the_full_factor(monkeypatch):
    """Unpenalized, ``metrics()`` and the rebuilt state certify the data rank on the full factor.

    There the data rank is pseudo-inverted for the published standard errors,
    so the grey surrogate must not stand in for it.  The near-collinear pair
    alone (``selection_penalty=0``) is the regime: its Gram cannot certify
    itself, yet the grey-subspace certificate would pass, so ``certify=``
    is the only thing that keeps the full factor.
    Mutation: dropping ``certify=`` from ``metrics._certified_data_rank`` or
    from ``state_ops._legacy_active_state`` each leaves a grey subspace.
    """
    import superglm.inference.metrics as metrics_module
    from superglm.model import state_ops

    frame, y, _, _ = _near_collinear_pair()
    frame = frame[["u", "v"]]
    model = SuperGLM(
        family="poisson",
        features={"u": Numeric(), "v": Numeric()},
        selection_penalty=0,
    )
    model.fit(frame, y)
    calls = []

    def spied(module):
        decide = module.decide_data_rank

        def wrapper(gram, *args, **kwargs):
            decided = decide(gram, *args, **kwargs)
            calls.append((np.array(gram), kwargs, decided))
            return decided

        monkeypatch.setattr(module, "decide_data_rank", wrapper)

    spied(metrics_module)
    spied(state_ops)
    half, y_half = frame.iloc[::2].reset_index(drop=True), y[::2]
    model.metrics(half, y_half).coefficient_se
    solver = model._solver_pirls_result()
    W = state_ops._solver_space_working_weights(model)
    state_ops._legacy_active_state(model, solver, W)
    assert len(calls) == 2
    # the precondition: the Gram is not authoritative and the grey certificate would pass
    for gram, kwargs, _ in calls:
        assert rank.decompose_gram_if_authoritative(gram, psd_by_construction=True) is None
        deflated, _ = rank.decompose_gram_deflated(
            gram, kwargs["project"], rows=kwargs["rows"], structural=kwargs["structural"]()
        )
        assert deflated is not None
    # the routes: both keep the data rank on the full factor
    for _, _, decided in calls:
        assert decided.grey_subspace is None
        assert decided.rank == 2


@pytest.mark.parametrize("discrete", [False, True], ids=["exact", "discrete"])
def test_scop_terminal_data_rank_takes_the_fits_route(discrete, monkeypatch):
    """SCOP REML's published data rank comes from the grey subspace, as a direct fit's does.

    A monotone P-spline beside the thin level's pair: the terminal data Gram
    cannot certify itself.  No TSQR of every row for it, and the decisions
    are the full factor's.  Mutation: on the base ``_hydrate_scop_terminal_rank_info``
    factored every row.
    """
    import superglm.reml.scop_efs as scop_efs
    from superglm import Constraint, PSpline

    frame, y, features, interactions = _thin_level_curve()
    frame = frame.assign(x=np.random.default_rng(3).uniform(0.0, 1.0, len(y)))
    features = {**features, "x": PSpline(n_knots=8, constraint=Constraint.fit.increasing)}
    passes = []
    factor = scop_efs.grouped_weighted_factor
    monkeypatch.setattr(
        scop_efs,
        "grouped_weighted_factor",
        lambda *args, **kwargs: passes.append(1) or factor(*args, **kwargs),
    )

    def fit():
        model = SuperGLM(
            family="poisson", features=features, interactions=interactions, discrete=discrete
        )
        model.fit_reml(frame, y)
        assert getattr(model.result, "scop_inference", None) is not None
        return model.result.rank_info.data

    data = fit()
    assert passes == [] and data.grey_subspace is not None
    _forced_full_factor(monkeypatch)
    reference = fit()
    assert len(passes) == 1
    _assert_same_decisions(data, reference)
