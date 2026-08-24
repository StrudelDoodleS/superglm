"""The dense screening path's design factors: exactness pins against its Grams.

Issue #257.  ``penalized_score_statistic_ladder`` used to be handed the pair's
assembled moments, so the spectrum it worked in was the design's SQUARED.  It
is handed a triangular factor of the same design instead, and these are the
pins that the factor carries EXACTLY the information the moments carried:
every Gram the old route assembled is recovered from ``PairFactor.joint`` to
round-off, and the tensor block comes back in the same C-order the tensor
penalty is assembled in.

The four moment producers are still here as the arbiter — see the disposition
note in :mod:`superglm.screening._pair_moments`.  They are the reference this
file grades the factor against; they are no longer the production route.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg

from superglm.screening._numeric_margin import (
    numeric_numeric_moments,
    numeric_pair_moments,
)
from superglm.screening._overlap import pair_overlap_moments, tensor_penalty
from superglm.screening._pair_factor import (
    PairFactor,
    numeric_cat_factor,
    numeric_numeric_factor,
    pair_design_factor,
)
from superglm.screening._pair_moments import pair_cell_moments, pair_score_curvature


def _gram_blocks(factor: PairFactor):
    """``(M, C, V, u_m, U)`` read back off the joint factor's own Gram.

    The overlap's internal column order is the FACTOR's, not
    ``pair_overlap_moments``', so only span-invariant statements may be made
    about ``M`` and ``C`` directly; the caller reorders when it compares.
    """
    q, k = factor.overlap_width, factor.tensor_width
    G = factor.joint.T @ factor.joint
    return (
        G[:q, :q],
        G[:q, q : q + k],
        G[q : q + k, q : q + k],
        G[:q, -1],
        G[q : q + k, -1],
    )


def _profiled_from_grams(U, V, C, M, u_m):
    """``(V_eff, U_eff)`` the way the moment route formed them."""
    MinvC = scipy.linalg.solve(M, C, assume_a="pos")
    V_eff = V - C.T @ MinvC
    return 0.5 * (V_eff + V_eff.T), U - MinvC.T @ u_m


def _profiled_from_factor(factor: PairFactor):
    """``(V_eff, U_eff)`` off the factor, with no Gram difference anywhere."""
    q, k = factor.overlap_width, factor.tensor_width
    R_eff = factor.joint[q : q + k, q : q + k]
    z_t = factor.joint[q : q + k, -1]
    return R_eff.T @ R_eff, R_eff.T @ z_t


def _gridded_case(seed, n=4000, n_a=17, n_b=13, k_a=4, k_b=3):
    rng = np.random.default_rng(seed)
    codes_a = rng.integers(0, n_a, n)
    codes_b = rng.integers(0, n_b, n)
    B_a = rng.normal(size=(n_a, k_a))
    B_b = rng.normal(size=(n_b, k_b))
    score = rng.normal(size=n)
    weights = rng.uniform(0.2, 2.0, n)
    S_cell, W_cell = pair_cell_moments(codes_a, codes_b, n_a, n_b, score, weights)
    return B_a, B_b, S_cell, W_cell


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_factor_reproduces_every_moment_the_dense_route_assembled(seed):
    """One triangular factor carries ``U``, ``V``, ``C``, ``M`` and ``u_m``.

    The whole change rests on this: the ladder is handed strictly less
    arithmetic and strictly more information, so nothing the moment route
    could answer may become unanswerable.  Asserted against the producers
    themselves rather than against a re-derivation of them.
    """
    B_a, B_b, S_cell, W_cell = _gridded_case(seed)
    U, V = pair_score_curvature(B_a, B_b, S_cell, W_cell)
    M, C, u_m = pair_overlap_moments(B_a, B_b, S_cell, W_cell)

    factor = pair_design_factor(B_a, B_b, S_cell, W_cell)
    k_a, k_b = B_a.shape[1], B_b.shape[1]
    assert factor.tensor_width == k_a * k_b
    assert factor.overlap_width == 1 + k_a + k_b
    assert factor.joint.shape == (factor.overlap_width + factor.tensor_width + 1,) * 2

    M_f, C_f, V_f, um_f, U_f = _gram_blocks(factor)
    # The overlap is a SPAN, and the factor orders it ``[1 | B | A]`` where
    # ``pair_overlap_moments`` orders it ``[1 | A | B]``.  Reorder before
    # comparing rather than asserting the factor's own order is the old one.
    order = [0] + [1 + k_b + p for p in range(k_a)] + [1 + q for q in range(k_b)]
    np.testing.assert_allclose(M_f[np.ix_(order, order)], M, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(C_f[order], C, rtol=1e-12, atol=1e-11 * np.abs(C).max())
    np.testing.assert_allclose(um_f[order], u_m, rtol=1e-11, atol=1e-11 * np.abs(u_m).max())
    np.testing.assert_allclose(V_f, V, rtol=1e-12, atol=1e-11 * np.abs(V).max())
    np.testing.assert_allclose(U_f, U, rtol=1e-11, atol=1e-11 * np.abs(U).max())


@pytest.mark.parametrize(("k_a", "k_b"), [(4, 3), (3, 4), (1, 5), (5, 1), (2, 2)])
def test_the_tensor_block_stays_in_the_order_the_penalty_is_assembled_in(k_a, k_b):
    """C-order ``p * k_b + q``, whichever margin the builder expands inside.

    ``tensor_penalty(S_a, S_b)`` is ``kron(S_a, I) + kron(I, S_b)`` and no
    permutation stands between it and the block it penalizes.  The builder
    chooses which margin to reduce inside by width, so the orientation it
    picks must not be visible here: a transposed tensor order would leave
    every ``edf0`` wrong and nothing else would notice.
    """
    B_a, B_b, S_cell, W_cell = _gridded_case(7, n_a=11, n_b=9, k_a=k_a, k_b=k_b)
    _, V = pair_score_curvature(B_a, B_b, S_cell, W_cell)
    _, _, V_f, _, _ = _gram_blocks(pair_design_factor(B_a, B_b, S_cell, W_cell))
    np.testing.assert_allclose(V_f, V, rtol=1e-11, atol=1e-11 * np.abs(V).max())

    # And the penalty really is aligned: a diagonal S_a with distinct entries
    # puts a distinguishable value on every tensor column, so a transposed
    # order would move mass onto the wrong ones.
    S_a = np.diag(np.arange(1.0, k_a + 1.0))
    S_b = np.diag(np.arange(1.0, k_b + 1.0) * 10.0)
    expected = np.add.outer(np.arange(1.0, k_a + 1.0), np.arange(1.0, k_b + 1.0) * 10.0)
    np.testing.assert_allclose(np.diag(tensor_penalty(S_a, S_b)), expected.reshape(-1))


def test_a_zero_weight_cell_contributes_nothing_and_raises_nothing():
    """``W == 0`` forces ``S_cell == 0``, so the working response there is 0.

    The response column is ``S_cell / sqrt(W_cell)``, which is 0/0 on an empty
    cell.  It is not a convention: the working weights are non-negative, so a
    zero cell weight means every row in it had zero weight, and
    ``working_score`` carries the same ``weights * dmu_deta`` factor -- the
    cell's score is then exactly zero and the ratio's numerator vanishes with
    its denominator.
    """
    codes_a = np.array([0, 0, 0, 0])
    codes_b = np.array([2, 2, 0, 0])
    B_a = np.array([[1.5, -0.5]])
    B_b = np.array([[1.0], [2.0], [4.0]])
    score = np.array([1.0, 2.0, 3.0, 4.0])
    weights = np.array([0.5, 0.5, 1.0, 1.0])
    S_cell, W_cell = pair_cell_moments(codes_a, codes_b, 1, 3, score, weights)
    assert W_cell[0, 1] == 0.0

    U, V = pair_score_curvature(B_a, B_b, S_cell, W_cell)
    _, _, V_f, _, U_f = _gram_blocks(pair_design_factor(B_a, B_b, S_cell, W_cell))
    np.testing.assert_allclose(V_f, V, rtol=1e-13, atol=1e-13 * np.abs(V).max())
    np.testing.assert_allclose(U_f, U, rtol=1e-13, atol=1e-13 * np.abs(U).max())


def test_the_factor_profiles_the_overlap_without_ever_differencing_two_grams():
    """``V_eff = R_eff' R_eff`` and ``U_eff = R_eff' z_t``, by Frisch-Waugh.

    This is the identity #257 turns on.  The moment route reaches ``V_eff``
    as ``V - C' M^-1 C``, a difference of two Grams; the factor route reads
    it off a trailing diagonal block of one triangular factor, where nothing
    has been subtracted and the spectrum is the design's rather than its
    square.
    """
    B_a, B_b, S_cell, W_cell = _gridded_case(3)
    U, V = pair_score_curvature(B_a, B_b, S_cell, W_cell)
    M, C, u_m = pair_overlap_moments(B_a, B_b, S_cell, W_cell)
    V_gram, U_gram = _profiled_from_grams(U, V, C, M, u_m)
    V_fac, U_fac = _profiled_from_factor(pair_design_factor(B_a, B_b, S_cell, W_cell))

    np.testing.assert_allclose(V_fac, V_gram, rtol=0, atol=1e-11 * np.abs(V_gram).max())
    np.testing.assert_allclose(U_fac, U_gram, rtol=0, atol=1e-11 * np.abs(U_gram).max())


def _numeric_cat_case(seed, L=7, n=3000, thin=None):
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, L, n)
    if thin is not None:
        codes[: n - thin] = rng.integers(1, L, n - thin)
        codes[n - thin :] = 0
    menu = np.eye(L)[:, 1:]
    z = rng.normal(size=n)
    score = rng.normal(size=n)
    weights = rng.uniform(0.2, 2.0, n)
    return codes, L, menu, z, score, weights


@pytest.mark.parametrize("seed", [0, 5])
def test_the_numeric_cat_factor_reproduces_its_own_moments(seed):
    codes, L, menu, z, score, weights = _numeric_cat_case(seed)
    U, V, C, M, u_m = numeric_pair_moments(codes, L, menu, z, score, weights)
    factor = numeric_cat_factor(codes, L, menu, z, score, weights)
    k = menu.shape[1]
    assert (factor.tensor_width, factor.overlap_width) == (k, 1 + k + 1)

    M_f, C_f, V_f, um_f, U_f = _gram_blocks(factor)
    # The factor orders the overlap ``[1 | z | menu]``; numeric_pair_moments
    # orders it ``[1 | menu | z]``.
    order = [0] + [2 + p for p in range(k)] + [1]
    np.testing.assert_allclose(M_f[np.ix_(order, order)], M, rtol=1e-11, atol=1e-10)
    np.testing.assert_allclose(C_f[order], C, rtol=1e-11, atol=1e-10 * np.abs(C).max())
    np.testing.assert_allclose(um_f[order], u_m, rtol=1e-10, atol=1e-10 * np.abs(u_m).max())
    np.testing.assert_allclose(V_f, V, rtol=1e-11, atol=1e-10 * np.abs(V).max())
    np.testing.assert_allclose(U_f, U, rtol=1e-10, atol=1e-10 * np.abs(U).max())


def test_the_numeric_numeric_factor_reproduces_its_own_moments():
    rng = np.random.default_rng(4)
    n = 5000
    z1, z2 = rng.normal(size=n), rng.normal(scale=3.0, size=n)
    score, weights = rng.normal(size=n), rng.uniform(0.2, 2.0, n)
    U, V, C, M, u_m = numeric_numeric_moments(z1, z2, score, weights)
    factor = numeric_numeric_factor(z1, z2, score, weights)
    assert (factor.tensor_width, factor.overlap_width) == (1, 3)

    M_f, C_f, V_f, um_f, U_f = _gram_blocks(factor)
    # The factor orders the overlap ``[1 | z2 | z1]``; the moments order it
    # ``[1 | z1 | z2]``.
    order = [0, 2, 1]
    np.testing.assert_allclose(M_f[np.ix_(order, order)], M, rtol=1e-11, atol=1e-10)
    np.testing.assert_allclose(C_f[order], C, rtol=1e-11, atol=1e-10 * np.abs(C).max())
    np.testing.assert_allclose(um_f[order], u_m, rtol=1e-10, atol=1e-10 * np.abs(u_m).max())
    np.testing.assert_allclose(V_f, V, rtol=1e-11, atol=1e-10 * np.abs(V).max())
    np.testing.assert_allclose(U_f, U, rtol=1e-10, atol=1e-10 * np.abs(U).max())


def _row_space_numeric_cat_factor(codes, n_g, menu, z, score, weights):
    """The same factor, taken from ROWS instead of from per-cell moments.

    The reference for :func:`numeric_cat_factor`'s one surviving Gram.  It
    sorts the rows by level and reduces each level's own ``(rows, 3)`` block
    by QR, so the ``[1, z, yhat]`` geometry inside a cell is never squared.
    Same emission, same reduction, one different step -- which is what makes
    it an arbiter rather than a reimplementation.
    """
    root = np.sqrt(weights)
    yhat = np.divide(score, root, out=np.zeros_like(score, dtype=float), where=root > 0.0)
    k = menu.shape[1]
    w = (1 + k) * 2 + 1
    blocks = []
    for g in range(n_g):
        rows = np.flatnonzero(codes == g)
        if rows.size == 0:
            continue
        local = np.stack([root[rows], root[rows] * z[rows], yhat[rows]], axis=1)
        R = np.linalg.qr(local, mode="r")
        a_g = np.concatenate(([1.0], menu[g]))
        block = np.empty((R.shape[0], w))
        block[:, : w - 1] = np.kron(a_g[None, :], R[:, :2])
        block[:, -1] = R[:, -1]
        blocks.append(block)
    joint = np.linalg.qr(np.concatenate(blocks, axis=0), mode="r")
    padded = np.zeros((w, w))
    padded[: joint.shape[0]] = joint
    # Kronecker order is ``[1, z, m_1, m_1 z, ...]``; regroup to
    # ``[overlap | tensor | yhat]``.
    perm = (
        [0, 1] + [2 * p for p in range(1, k + 1)] + [2 * p + 1 for p in range(1, k + 1)] + [w - 1]
    )
    return PairFactor(
        joint=np.linalg.qr(padded[:, perm], mode="r"),
        overlap_width=2 + k,
        tensor_width=k,
    )


@pytest.mark.parametrize("thin", [400, 3, 1])
def test_the_numeric_cat_cell_gram_is_not_what_limits_the_pair(thin):
    """T6: the one Gram this builder keeps, graded against a row-space QR.

    :func:`numeric_cat_factor` roots each level's ``[1, z, yhat]`` moment
    block from the bincount pass that already runs, which squares a
    TWO-dimensional geometry inside the cell.  The alternative is to sort the
    rows and factor each level's rows directly, which is the reference here.
    The compromise is only defensible if the two agree far inside what the
    pair's own answer is determined to, including on a level starved down to
    a single row -- the geometry #257 is about.
    """
    codes, L, menu, z, score, weights = _numeric_cat_case(11, thin=thin)
    cell = numeric_cat_factor(codes, L, menu, z, score, weights)
    rows = _row_space_numeric_cat_factor(codes, L, menu, z, score, weights)

    # THE WHOLE GRAM, RESPONSE DIAGONAL INCLUDED.  Comparing only ``V_eff``
    # and ``U_eff`` leaves the ``sum s**2 / w`` channel unpinned -- and that
    # channel is what makes the per-level block exactly PSD, so the clip in
    # the root is inert rather than silently deleting a direction.  Measured:
    # with that entry taken at half its value the profiled blocks still agree
    # to 1e-10 and only this comparison moves.
    G_cell = cell.joint.T @ cell.joint
    G_rows = rows.joint.T @ rows.joint
    assert np.abs(G_cell - G_rows).max() / max(np.abs(G_rows).max(), 1e-300) < 1e-12, thin

    V_cell, U_cell = _profiled_from_factor(cell)
    V_rows, U_rows = _profiled_from_factor(rows)
    scale = max(np.abs(V_rows).max(), 1e-300)
    assert np.abs(V_cell - V_rows).max() / scale < 1e-10, thin
    assert np.abs(U_cell - U_rows).max() / max(np.abs(U_rows).max(), 1e-300) < 1e-10, thin
