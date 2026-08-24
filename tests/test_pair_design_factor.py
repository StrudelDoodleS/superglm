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

**WHERE EVERY MAGNITUDE CONSTANT IN THIS FILE COMES FROM.**  Each assertion
here compares two backward-stable routes to the SAME quantity — a Gram entry
read off the factor against the producer that assembles it — so the achieved
difference is round-off and nothing else, and the honest thing is to record it
rather than to leave the constants unexplained.  Measured at one thread on
Python 3.14.6, as ``max|a - b| / max|b|`` over each test's own comparisons::

    test_the_factor_reproduces_every_moment_...        9.60e-16   asserted 1e-11
    test_the_tensor_block_stays_in_the_order_...       7.37e-16   asserted 1e-11
    test_a_zero_weight_cell_contributes_nothing_...    1.25e-16   asserted 1e-13
    test_the_factor_profiles_the_overlap_...           9.09e-16   asserted 1e-11
    test_a_rank_deficient_overlap_profiles_...         1.83e-15   asserted 1e-11
    test_the_numeric_cat_factor_reproduces_...         3.11e-15   asserted 1e-10
    test_the_numeric_numeric_factor_reproduces_...     1.04e-14   asserted 1e-10
    test_the_numeric_cat_cell_gram_... (whole Gram)    9.30e-16   asserted 1e-12
    test_the_numeric_cat_cell_gram_... (profiled)      1.17e-14   asserted 1e-10

So every constant sits between 3 and 5 orders above what it has to cover.  That
band is deliberate and it is PLAN-257 section 9's rule applied rather than
dodged: the issue thread records an 859x swing on ``ns(6)`` between Python 3.12
and 3.14 with everything else held, and #324 records a bound set from one
pinned run failing on SANDYBRIDGE at eight threads, so a constant this file
sets from one machine needs three orders of allowance before it is a bound
rather than a fingerprint.  A tighter constant here would be measuring this
interpreter.  A looser one is only defensible if something about the geometry
justifies it, and none of these geometries does — the rank-deficient case was
two orders looser than its siblings for no stated reason and is now at theirs.
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
    _profiled_factor,
    numeric_cat_factor,
    numeric_numeric_factor,
    pair_design_factor,
)
from superglm.screening._pair_moments import pair_cell_moments, pair_score_curvature
from superglm.screening._score_stat import penalized_score_statistic_ladder


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
    R_eff, z_t = _profiled_factor(factor)
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


def test_a_rank_deficient_overlap_profiles_onto_its_own_range_and_no_further():
    """A menu wider than its support, which an ``OrderedCategorical`` reaches.

    The joint reduction is unpivoted, so a column of the overlap its
    predecessors already span leaves a zero pivot and the direction ``Q`` puts
    there is an arbitrary unit vector orthogonal to what came before -- NOT in
    ``range(X_o)``.  Residualizing on the whole leading block then removes
    curvature the overlap never explained.

    The arbiter is the moment route's own profiling with a PSEUDO-inverse,
    which projects onto ``range(M)`` and nothing wider.  Measured on the mixed
    suite's ``band x power`` pair -- a 7-column inner-spline menu on 5 levels,
    overlap rank 14 of 17 -- the bare slice reported a statistic of 13.28
    against an exact-design 22.31, and moved the published ``z`` from 10.07 to
    5.67.  This is the same defect on a fixture small enough to state.
    """
    rng = np.random.default_rng(21)
    n_a, n_b, k_b = 5, 9, 3
    # SEVEN columns on FIVE support points: two of them are exactly dependent,
    # so the overlap is rank deficient before the pair is formed.
    base = rng.normal(size=(n_a, 5))
    B_a = np.concatenate([base, base[:, :2] @ rng.normal(size=(2, 2))], axis=1)
    B_b = rng.normal(size=(n_b, k_b))
    codes_a = rng.integers(0, n_a, 4000)
    codes_b = rng.integers(0, n_b, 4000)
    S_cell, W_cell = pair_cell_moments(
        codes_a, codes_b, n_a, n_b, rng.normal(size=4000), rng.uniform(0.2, 2.0, 4000)
    )
    U, V = pair_score_curvature(B_a, B_b, S_cell, W_cell)
    M, C, u_m = pair_overlap_moments(B_a, B_b, S_cell, W_cell)
    assert np.linalg.matrix_rank(M) < M.shape[0], "the fixture's premise"

    MinvC = np.linalg.pinv(M, hermitian=True) @ C
    V_gram = 0.5 * ((V - C.T @ MinvC) + (V - C.T @ MinvC).T)
    U_gram = U - MinvC.T @ u_m
    V_fac, U_fac = _profiled_from_factor(pair_design_factor(B_a, B_b, S_cell, W_cell))

    # THE SIBLING'S CONSTANT, NOT A LOOSER ONE FOR THE ILL-POSED GEOMETRY.
    # This was ``1e-9``, two orders past every other bound in the file, with no
    # sentence saying why -- and the deficiency here is EXACT rather than
    # marginal, so it costs no accuracy: ``M``'s singular values run
    # 2.93e-13 .. 4.71e+04 with rank 8 of 11, and both routes' cuts fall in the
    # fifteen-order gap between the dropped directions and the kept ones.  What
    # is left to bound is ordinary round-off, measured at 3.99e-16 on ``V_eff``
    # and 1.83e-15 on ``U_eff``, which is 5.5e+03x inside ``1e-11``.
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


# The producers that assembled the dense route's Grams.  They are KEPT -- every
# test above grades the factor against them, and
# ``test_the_dense_path_s_ceiling_is_its_gram_and_not_its_arithmetic`` cannot
# make its point without them -- but they must not regain a production caller,
# which is what this list is for.
_RETIRED_GRAM_PRODUCERS = frozenset(
    {
        "pair_score_curvature",
        "pair_overlap_moments",
        "numeric_pair_moments",
        "numeric_numeric_moments",
        "tensor_penalty",
    }
)

# The three of them ``screening/__init__.py`` names in ``__all__``, which IS the
# arbiter's public surface.  It is neither all five nor a pattern: ``_overlap``
# is never imported there, so ``pair_overlap_moments`` and ``tensor_penalty``
# are re-exported by nothing and re-exporting either would be a new decision.
_RE_EXPORTED_PRODUCERS = frozenset(
    {
        "numeric_numeric_moments",
        "numeric_pair_moments",
        "pair_score_curvature",
    }
)


def test_no_production_module_imports_a_retired_gram_producer():
    """The fingerprint of the rule issue #257 replaced, enforced by import.

    #322's precedent is to delete what nothing calls.  These five are kept
    instead, and the reason is that they still have real callers -- as this
    suite's arbiter, which is a use rather than dead code, and the same
    standing ``_reference_edf`` has in the structured suite.  What must not
    happen is one of them quietly becoming the production route again for a
    pair some future branch adds, because the moment route's accuracy ceiling
    is architectural and a new caller would inherit it without being told.

    An AST scan rather than a grep: a name reached through
    ``superglm.screening.pair_score_curvature`` is the same defect as an
    ``import``, and both are import statements.

    **THE SCAN COVERS ALL OF ``src/superglm/``, NOT JUST ``model/``.**  It used
    to stop at ``src/superglm/model/`` because that is where every screening
    CALLER lives -- but a producer re-adopted from inside
    ``src/superglm/screening/`` would inherit the same accuracy ceiling and
    slip past, and that is the plausible route rather than an exotic one:
    ``_structured.py`` is a production module that already imports from its
    siblings.  Widening costs one exemption, ``screening/__init__.py``.

    **THE EXEMPTION IS THREE NAMES, NOT THE FILE, AND IT USED TO SAY FIVE.**
    ``__init__.py`` imports ``_numeric_margin`` and ``_pair_moments`` and
    re-exports ``numeric_numeric_moments``, ``numeric_pair_moments`` and
    ``pair_score_curvature``; it never imports ``_overlap``, so
    ``pair_overlap_moments`` and ``tensor_penalty`` are re-exported by nothing
    -- consistent with ``_overlap``'s own header calling them not-live, and not
    with the sentence that used to stand here.  Exempting the FILE would also
    have hidden a re-adoption inside it, which is a production edit like any
    other.  Exempting the three NAMES leaves the other two guarded there and
    makes re-exporting a sixth producer a deliberate edit to
    ``_RE_EXPORTED_PRODUCERS`` rather than a silent one.

    **A RELATIVE IMPORT IS AN IMPORT.**  Matching only modules that start with
    ``superglm.screening`` misses ``from ._overlap import tensor_penalty``,
    where ``node.module`` is ``"_overlap"`` and ``node.level`` is 1 -- and a
    sibling is exactly where the re-adoption this scan was widened for would
    appear.  ``node.level > 0`` closes it with no module test needed: these
    five names live nowhere but ``superglm.screening``, so a relative import
    that binds one of them resolved into that package whatever its dots say.
    The ``ast.Attribute`` fallback does not cover the case, because ``from X
    import name`` binds a bare ``Name``.

    Shown to bite on both widenings rather than only on the original scope.
    Adding ``from superglm.screening._overlap import tensor_penalty`` to
    ``_structured.py`` -- a screening module, so invisible to the old
    ``model/``-only scan -- reds this test naming the file and line; so does
    the relative form ``from ._overlap import tensor_penalty`` in the same
    file, which the module test alone let through.  And nothing else under
    ``src/superglm/`` reaches any of the five, so both arrive green.
    """
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1] / "src" / "superglm"
    re_export = root / "screening" / "__init__.py"
    offenders = []
    for path in sorted(root.rglob("*.py")):
        watched = _RETIRED_GRAM_PRODUCERS
        if path == re_export:
            watched = watched - _RE_EXPORTED_PRODUCERS
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (
                (node.module or "").startswith("superglm.screening") or node.level > 0
            ):
                for alias in node.names:
                    if alias.name in watched:
                        offenders.append(f"{path.name}:{node.lineno} {alias.name}")
            elif isinstance(node, ast.Attribute) and node.attr in watched:
                offenders.append(f"{path.name}:{node.lineno} .{node.attr}")
    assert offenders == [], (
        "the dense screening path reads design factors since issue #257; these "
        f"reach for the moments it replaced: {offenders}"
    )


def _zero_weight_boundary_case(outlier_z, *, position):
    """One ``numeric_cat`` pair, optionally carrying one zero-weight row.

    Level 0 holds a real per-level slope -- its ``z`` takes two distinct
    values against differing scores -- so erasing that level's spread is
    visible in the published statistic rather than only in the factor.
    ``position`` places the zero-weight row first, last, or nowhere.
    """
    codes = [0, 0, 0, 0, 1, 1, 1, 1]
    z = [1.0, 2.0, 1.0, 2.0, 0.5, 1.5, 2.5, 3.5]
    w = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    s = [0.3, 0.9, 0.4, 1.1, -0.2, 0.4, 0.1, 0.8]
    if position is not None:
        # A zero-weight row's score is exactly zero by construction: both cell
        # tables carry the same ``weights * dmu_deta`` factor, so a row with no
        # weight contributes no score either.
        at = len(codes) if position == "last" else 0
        codes.insert(at, 0)
        z.insert(at, outlier_z)
        w.insert(at, 0.0)
        s.insert(at, 0.0)
    return (
        np.asarray(codes, dtype=np.intp),
        2,
        np.array([[0.0], [1.0]]),
        np.asarray(z, dtype=np.float64),
        np.asarray(s, dtype=np.float64),
        np.asarray(w, dtype=np.float64),
    )


def _published(args):
    """``(statistic, edf0)`` for a ``numeric_cat`` pair, which is unpenalized."""
    rung = penalized_score_statistic_ladder(numeric_cat_factor(*args), None, budgets=(4.0,))[0]
    return float(rung.statistic), float(rung.edf0)


@pytest.mark.parametrize("outlier_z", [1e20, 1e308])
@pytest.mark.parametrize("position", ["first", "last"])
def test_a_zero_weight_row_cannot_move_a_numeric_cat_pair(outlier_z, position):
    """A row with no weight is not in the fit and may not reach the screen.

    The level reference is a scatter with duplicate indices, so whichever row
    of the level comes LAST supplies the value every other row is shifted
    against.  A zero-weight row is a row, and its ``z`` is unconstrained by
    anything -- it is not in the fit -- so before this was fixed the reference
    could be arbitrarily far from the level's own values and the shift stopped
    being exact.  Two regimes, both from one appended row at unit weights:

    * ``z = 1e20`` erased the level outright.  ``1 - 1e20`` and ``2 - 1e20``
      round to the SAME double, so ``m2`` came back exactly zero, the level's
      real slope went with it, and the pair published ``statistic 0.0,
      edf0 0.0`` where it reads 0.1203333333 and 1.0.
    * ``z = 1e308`` overflowed.  The level's four shifted values summed past
      the largest double, ``offset`` came back infinite, and the reduction
      raised ``ValueError: array must not contain infs or NaNs`` from inside
      ``_combine_row_factors`` -- a screen that crashes on a row that is not
      in the fit.

    Both are decided by the row's POSITION, which is why the control arm here
    places the same row first: that arm passed throughout.
    """
    baseline = _published(_zero_weight_boundary_case(outlier_z, position=None))
    assert baseline == pytest.approx((0.12033333333333333, 1.0), rel=1e-12, abs=0.0)

    with_row = _published(_zero_weight_boundary_case(outlier_z, position=position))
    assert with_row == pytest.approx(baseline, rel=1e-12, abs=0.0)


@pytest.mark.parametrize("outlier_z", [1e20, 1e308])
def test_a_zero_weight_row_leaves_the_emitted_factor_bit_identical(outlier_z):
    """Stronger than the published invariance, and the reason it holds.

    The reference is selected from the level's positively-weighted rows, so a
    zero-weight row changes no input to any accumulation: its weight is zero
    in every ``bincount`` channel and its score is exactly zero.  The factor
    is therefore the same BITS, not merely the same answer to a tolerance.
    """
    base = numeric_cat_factor(*_zero_weight_boundary_case(outlier_z, position=None)).joint
    for position in ("first", "last"):
        got = numeric_cat_factor(*_zero_weight_boundary_case(outlier_z, position=position)).joint
        assert np.array_equal(got, base), f"zero-weight row placed {position} moved the factor"


def test_a_level_whose_every_row_has_zero_weight_emits_an_exact_zero_block():
    """The case the reference selection above cannot fill, stated rather than left.

    Choosing the reference from ``codes_g[w > 0]`` leaves a level with NO
    positively-weighted row unwritten, so its reference stays 0.0.  That is
    inert rather than arbitrary: the level's ``w0`` and ``s0`` are both exactly
    zero, so ``positive`` and ``spread`` are both False, every divide is
    guarded to zero, and the level emits an exact zero block whatever its
    ``z`` holds -- including a value whose square would overflow.
    """
    codes = np.array([0, 0, 1, 1, 1], dtype=np.intp)
    z = np.array([1e308, -1e308, 0.5, 1.5, 2.5])
    w = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
    s = np.array([0.0, 0.0, -0.2, 0.4, 0.1])
    factor = numeric_cat_factor(codes, 2, np.array([[0.0], [1.0]]), z, s, w)
    assert np.all(np.isfinite(factor.joint))

    # The dead level contributes nothing: dropping its rows entirely leaves
    # the same factor, bit for bit.
    alive = np.array([2, 3, 4])
    same = numeric_cat_factor(
        codes[alive], 2, np.array([[0.0], [1.0]]), z[alive], s[alive], w[alive]
    )
    assert np.array_equal(factor.joint, same.joint)
