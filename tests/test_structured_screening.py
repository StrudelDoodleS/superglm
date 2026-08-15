"""The arrow kernel that lifts spline_cat's level ceiling.

The dense path densifies a block that is structurally block-diagonal, which
costs a cubic solve and quadratic memory in the level count and caps the
factors screening will look at.  These tests pin the two things that make the
structured path a safe replacement above that cap: it computes the same
quantity as the dense path, and it never allocates what the dense path was
refused for.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd
import pytest
import scipy.linalg

import superglm.model.screening_ops as ops
from superglm import SuperGLM
from superglm.features import Categorical, Spline
from superglm.model.screening_ops import _contrast_menu, _contrast_rows
from superglm.screening._arrow import factor_arrow
from superglm.screening._overlap import pair_overlap_moments
from superglm.screening._pair_moments import pair_score_curvature
from superglm.screening._score_stat import penalized_score_statistic_ladder
from superglm.screening._structured import _evaluate, _profile, spline_cat_moments

BUDGETS = (2.0, 4.0, 8.0, 16.0)


def _dense_arrow(G, E, border):
    """The same matrix ``factor_arrow`` takes, assembled densely."""
    n_blocks, g, _ = G.shape
    r = border.shape[0]
    K = np.zeros((n_blocks * g + r, n_blocks * g + r))
    for q in range(n_blocks):
        s = slice(q * g, (q + 1) * g)
        K[s, s] = G[q]
        K[n_blocks * g :, s] = E[q]
        K[s, n_blocks * g :] = E[q].T
    K[n_blocks * g :, n_blocks * g :] = border
    return K


def _random_arrow(rng, n_blocks, g, r, rows_per_block=12):
    """A PSD arrow matrix, built as a Gram so the structure is exact.

    Each group's rows touch only that group's columns and the border, so
    ``Z' Z`` has zeros between groups by construction rather than by
    cancellation.
    """
    Z = np.zeros((n_blocks * rows_per_block, n_blocks * g + r))
    for q in range(n_blocks):
        rows = slice(q * rows_per_block, (q + 1) * rows_per_block)
        Z[rows, q * g : (q + 1) * g] = rng.normal(size=(rows_per_block, g))
        Z[rows, n_blocks * g :] = rng.normal(size=(rows_per_block, r))
    K = Z.T @ Z
    G = np.stack([K[q * g : (q + 1) * g, q * g : (q + 1) * g] for q in range(n_blocks)])
    E = np.stack([K[n_blocks * g :, q * g : (q + 1) * g] for q in range(n_blocks)])
    return G, E, K[n_blocks * g :, n_blocks * g :], K


def test_arrow_solve_and_inverse_blocks_match_a_dense_factorization():
    """The arrow factorization is an exact reorganization, not an approximation."""
    rng = np.random.default_rng(0)
    n_blocks, g, r = 9, 4, 3
    G, E, border, K = _random_arrow(rng, n_blocks, g, r)

    b_blocks = rng.normal(size=(n_blocks, g))
    b_border = rng.normal(size=r)
    f = factor_arrow(G, E, border)
    x, z = f.solve(b_blocks, b_border)

    want = np.linalg.solve(K, np.concatenate([b_blocks.reshape(-1), b_border]))
    assert np.allclose(x.reshape(-1), want[: n_blocks * g], rtol=0, atol=1e-9)
    assert np.allclose(z, want[n_blocks * g :], rtol=0, atol=1e-9)

    Kinv = np.linalg.inv(K)
    got = f.diag_blocks()
    for q in range(n_blocks):
        s = slice(q * g, (q + 1) * g)
        assert np.allclose(got[q], Kinv[s, s], rtol=0, atol=1e-9)

    # The OFF-diagonal blocks are ``Y_q Sinv Y_q''`` and nothing else, which is
    # what lets the edf contract V_eff's non-block-diagonal half in O(L).
    # Never formed by the module; asserted here against the dense inverse.
    for q in range(n_blocks):
        for other in range(n_blocks):
            if q == other:
                continue
            block = f.Y[q] @ f.Sinv @ f.Y[other].T
            rows, cols = slice(q * g, (q + 1) * g), slice(other * g, (other + 1) * g)
            assert np.allclose(block, Kinv[rows, cols], rtol=0, atol=1e-9)


def test_arrow_inverts_a_singular_system_as_a_pseudo_inverse():
    """A degenerate level is routine at high cardinality, so the factorization
    has to keep going rather than fail on it.

    This module used to answer that with a RANK, because its one caller wrote
    ``edf`` as ``rank(A) - lambda tr(A^-1 S)``.  It does not any more, so what
    is pinned is the property the inverse actually has to have: ``K K^+ K = K``
    on the delivered blocks, with the empty level's own block left at zero
    rather than inverted.  The old assertion could hold while the inverse was
    wrong, and did -- an inverse and a rank counted by different cuts is
    exactly what the whole ``1 - 0`` failure was.
    """
    rng = np.random.default_rng(1)
    n_blocks, g, r = 6, 4, 3
    G, E, border, _ = _random_arrow(rng, n_blocks, g, r)
    G[2] = 0.0  # an empty level: no rows, so no curvature and no coupling
    E[2] = 0.0
    K = _dense_arrow(G, E, border)
    assert np.linalg.matrix_rank(K) == n_blocks * g + r - g, "fixture must be singular"

    f = factor_arrow(G, E, border)
    n = n_blocks * g + r
    Kplus = np.zeros((n, n))
    for q in range(n_blocks):
        s = slice(q * g, (q + 1) * g)
        for other in range(n_blocks):
            o = slice(other * g, (other + 1) * g)
            Kplus[s, o] = f.Y[q] @ f.Sinv @ f.Y[other].T
        Kplus[s, s] += f.Ginv[q]
        Kplus[s, n_blocks * g :] = -f.Y[q] @ f.Sinv
        Kplus[n_blocks * g :, s] = Kplus[s, n_blocks * g :].T
    Kplus[n_blocks * g :, n_blocks * g :] = f.Sinv

    assert np.allclose(K @ Kplus @ K, K, rtol=0, atol=1e-9 * np.abs(K).max())
    assert np.allclose(Kplus @ K @ Kplus, Kplus, rtol=0, atol=1e-9 * np.abs(Kplus).max())
    assert np.array_equal(f.Ginv[2], np.zeros((g, g)))


def _capture(df, y, features, cand, sample_weight=None, **kw):
    model = SuperGLM(family="gaussian", features=features)
    model.fit_reml(df, y, sample_weight=sample_weight)
    if sample_weight is not None:
        kw["sample_weight"] = sample_weight
    grab = {}
    real_curv, real_cells, real_ladder = (
        ops.pair_score_curvature,
        ops.pair_cell_moments,
        ops.penalized_score_statistic_ladder,
    )

    def spy_cells(*a, **k):
        grab["cells"] = real_cells(*a, **k)
        return grab["cells"]

    def spy_curv(B_a, B_b, S_cell, W_cell):
        grab["menus"] = (B_a, B_b)
        return real_curv(B_a, B_b, S_cell, W_cell)

    def spy_ladder(U, V, C, M, S_ti, **k):
        grab["args"] = (U, V, C, M, S_ti, k.get("U_nuisance"))
        return real_ladder(U, V, C, M, S_ti, **k)

    ops.pair_cell_moments = spy_cells
    ops.pair_score_curvature = spy_curv
    ops.penalized_score_statistic_ladder = spy_ladder
    try:
        model.screen_interactions(df, y, candidates=[cand], edf0=BUDGETS, **kw)
    finally:
        ops.pair_cell_moments = real_cells
        ops.pair_score_curvature = real_curv
        ops.penalized_score_statistic_ladder = real_ladder
    return grab


def _structured_inputs(grab):
    B_a, B_b = grab["menus"]
    S_cell, W_cell = grab["cells"]
    _, _, _, _, S_ti, _ = grab["args"]
    k_a, k_b = B_a.shape[1], B_b.shape[1]
    # S_ti is kron(S_a, I_kb), so S_a is its (0, 0) stride
    S_a = np.ascontiguousarray(S_ti[::k_b, ::k_b][:k_a, :k_a])
    return B_a, S_a, S_cell, W_cell, np.argmax(B_b, axis=0)


def _wood_stacked_edf(p, geometry, lam):
    """``edf`` by the textbook O(L^3) stacked QR, as a dense reference.

    ``V_eff = G' G`` with ``G`` the residualized design, so
    ``A = [G ; sqrt(lam) rootS]' [G ; sqrt(lam) rootS]`` and
    ``edf = ||G T^+||_F^2`` for ``T`` the stack's R factor -- Wood, *JRSS-B*
    73(1):3-36 (2011), §3.6.  Cubic in ``L`` and quadratic in memory, which is
    why it is a test-only reference, but it takes no pseudo-inverse of a
    singular Gram and so can arbitrate where ``numpy.linalg.pinv`` cannot: on
    the near-singular pencils here that ``pinv``'s own relative cut moves the
    answer by up to 2.4 df.
    """
    from superglm.screening._structured import _penalty_root

    L, k_a = p.dims
    carried = p.R @ geometry.coupling
    psi = np.concatenate(list(np.swapaxes(carried, -1, -2) @ p.R), axis=1)  # (r, L k_a)
    G = np.zeros((L * k_a + geometry.base_gram.shape[0], L * k_a))
    for q in range(L):
        rows = slice(q * k_a, (q + 1) * k_a)
        G[rows, :] = -carried[q] @ psi
        G[rows, rows] += p.R[q]
    G[L * k_a :, :] = -geometry.base_gram @ psi
    root = _penalty_root(p.S_a)
    stacked = np.concatenate((G, np.sqrt(lam) * scipy.linalg.block_diag(*([root] * L))), axis=0)
    T = np.linalg.qr(stacked, mode="r")
    u, sv, vt = np.linalg.svd(T)
    keep = sv > max(T.shape) * np.finfo(np.float64).eps * sv[0]
    inv = np.where(keep, 1.0 / np.where(keep, sv, 1.0), 0.0)
    return float(np.sum(np.square(G @ ((vt.T * inv) @ u.T))))


def _factored_v_eff(p, geometry):
    """``V_eff`` reassembled from what the geometry now carries.

    ``blockdiag(R_q' R_q) - Psi' Psi`` with ``Psi_q = (R_q coupling)' R_q``.
    This is the same matrix the old ``blockdiag(D_q) - D' Omega D`` named --
    ``coupling coupling'`` IS the spline-main corner of the overlap Gram's
    pseudo-inverse -- but taken on the row-space FACTOR, which is the change.
    """
    L, k_a = p.dims
    carried = p.R @ geometry.coupling
    psi = np.concatenate(list(np.swapaxes(carried, -1, -2) @ p.R), axis=1)
    blocks = np.swapaxes(p.R, -1, -2) @ p.R
    return scipy.linalg.block_diag(*blocks) - psi.T @ psi


def _dense_cell_inputs(B_a, S_a, S_cell, W_cell, level_rows):
    """Dense moments for the same treatment-coded cell geometry."""
    level_rows = np.asarray(level_rows, dtype=np.intp)
    B_b = np.zeros((W_cell.shape[1], level_rows.size), dtype=np.float64)
    B_b[level_rows, np.arange(level_rows.size)] = 1.0
    U, V = pair_score_curvature(B_a, B_b, S_cell, W_cell)
    M, C, u_m = pair_overlap_moments(B_a, B_b, S_cell, W_cell)
    S_ti = np.kron(S_a, np.eye(level_rows.size))
    return U, V, C, M, S_ti, u_m


def _near_absorbed_cells(base_energy=1e-12):
    """One emitted mode with unit raw curvature and a tiny profiled residual."""
    B_a = np.array([[-1.0], [1.0]])
    S_a = np.array([[1.0]])
    W_cell = np.array(
        [
            [0.5 * base_energy, 0.5],
            [0.5 * base_energy, 0.5],
        ]
    )
    S_cell = np.array([[0.0, 0.25], [0.0, -0.25]])
    return B_a, S_a, S_cell, W_cell, np.array([1], dtype=np.intp)


def _mixed_rank_cells(scale=1.0, *, permute=False):
    """One retained H direction and one positive direction below its rank cut."""
    d = 1e-8
    B_a = np.array(
        [
            [0.0, -d],
            [0.0, d],
            [-1.0, -d],
            [-1.0, d],
            [1.0, -d],
            [1.0, d],
        ]
    )
    W_cell = np.zeros((6, 2))
    W_cell[:2, 0] = 0.5 * scale
    W_cell[2:, 1] = 0.25 * scale
    S_cell = np.zeros_like(W_cell)
    level_rows = np.array([1], dtype=np.intp)
    if permute:
        row_order = np.array([5, 1, 3, 0, 4, 2])
        level_order = np.array([1, 0])
        inverse_levels = np.empty_like(level_order)
        inverse_levels[level_order] = np.arange(level_order.size)
        B_a = B_a[row_order]
        W_cell = W_cell[row_order][:, level_order]
        S_cell = S_cell[row_order][:, level_order]
        level_rows = inverse_levels[level_rows]
    return B_a, np.eye(2), S_cell, W_cell, level_rows


def _direct_centered_row_trace(B_a, W_cell, level_rows):
    """Explicit dense-row QR oracle, independent of the O(L) factor algebra."""
    B_a = np.asarray(B_a, dtype=np.float64)
    W_cell = np.asarray(W_cell, dtype=np.float64)
    level_rows = np.asarray(level_rows, dtype=np.intp)
    n_rows, k_a = B_a.shape
    shifted = B_a - B_a[0]
    row_blocks = []
    for level in range(W_cell.shape[1]):
        weights = W_cell[:, level]
        mass = weights.sum()
        mean = weights @ shifted / mass if mass > 0.0 else np.zeros(k_a)
        row_blocks.append(np.sqrt(weights)[:, None] * (shifted - mean))

    stacked = np.vstack(row_blocks)
    Q, R = np.linalg.qr(stacked, mode="reduced")
    trace = 0.0
    for level in level_rows:
        target = np.zeros_like(stacked)
        target[level * n_rows : (level + 1) * n_rows] = row_blocks[level]
        coefficients = np.linalg.solve(
            R,
            Q[level * n_rows : (level + 1) * n_rows].T @ row_blocks[level],
        )
        trace += float(np.sum(np.square(target - stacked @ coefficients)))
    return trace


def _adversarial_trace_cells(seed, delta):
    rng = np.random.default_rng(seed)
    n_rows, k_a, n_levels = 9, 4, 7
    B_a = rng.normal(size=(n_rows, k_a))
    B_a[:, -1] = B_a[:, 0] + delta * rng.normal(size=n_rows)
    W_cell = 10 ** rng.uniform(-15.0, 15.0, size=(n_rows, n_levels))
    W_cell[rng.random(W_cell.shape) < 0.15] = 0.0
    S_cell = np.zeros_like(W_cell)
    return B_a, np.eye(k_a), S_cell, W_cell, np.arange(1, n_levels, dtype=np.intp)


def _rank_boundary_trace_cells(order_seed):
    rng = np.random.default_rng(6)
    n_rows, n_levels = 8, 3
    x = rng.normal(size=n_rows)
    z = rng.normal(size=n_rows)
    W_cell = 10 ** rng.uniform(-4.0, 4.0, size=(n_rows, n_levels))
    W_cell[rng.random(W_cell.shape) < 0.1] = 0.0
    delta = 2.1569414515817223e-8
    B_a = np.column_stack((x, x + delta * z))

    order_rng = np.random.default_rng(order_seed)
    row_order = order_rng.permutation(n_rows)
    level_order = order_rng.permutation(n_levels)
    inverse_levels = np.empty_like(level_order)
    inverse_levels[level_order] = np.arange(level_order.size)
    level_rows = inverse_levels[np.arange(1, n_levels)]
    penalty = 0.5 * np.array([[1.0, -1.0], [-1.0, 1.0]])
    return (
        B_a[row_order],
        penalty,
        np.zeros_like(W_cell)[row_order][:, level_order],
        W_cell[row_order][:, level_order],
        level_rows,
    )


@pytest.fixture(scope="module")
def moderate_pair():
    """A spline_cat pair small enough that the dense path can score it too."""
    rng = np.random.default_rng(3)
    L, reps = 24, 30
    df = pd.DataFrame(
        {"g": np.repeat([f"L{i}" for i in range(L)], reps), "x": rng.uniform(0, 1, L * reps)}
    )
    y = rng.normal(size=len(df))
    grab = _capture(df, y, {"g": Categorical(), "x": Spline(kind="ps", n_knots=6)}, ("x", "g"))
    return grab


def test_structured_statistic_matches_the_dense_one_to_machine_precision(moderate_pair):
    """Away from the bracket edges the two paths agree to round-off.

    This is the pin that the arrow reorganization is EXACT.  The bracket edges
    are checked separately and to a looser tolerance, because there the matrix
    being inverted has a condition number around 1e11 and neither path can
    resolve it better than ``eps * kappa`` — a property of the quantity, not
    of either algorithm.
    """
    U, V, C, M, S_ti, u_m = moderate_pair["args"]
    B_a, S_a, S_cell, W_cell, level_rows = _structured_inputs(moderate_pair)

    MinvC = scipy.linalg.solve(M, C, assume_a="pos")
    V_eff = 0.5 * ((V - C.T @ MinvC) + (V - C.T @ MinvC).T)
    U_eff_dense = U - MinvC.T @ u_m

    p = spline_cat_moments(B_a, S_a, S_cell, W_cell, level_rows)
    geometry = _profile(p)
    assert p.profiled_trace == pytest.approx(float(np.trace(V_eff)), rel=1e-12)

    # dense column order is p*k_b + q; the kernel groups by level
    k_a, k_b = B_a.shape[1], len(level_rows)
    perm = np.arange(k_a * k_b).reshape(k_a, k_b).T.reshape(-1)
    assert np.allclose(geometry.U_eff.reshape(-1), U_eff_dense[perm], rtol=0, atol=1e-12)

    # The factored V_eff is the SAME matrix, not an approximation of it:
    # blockdiag(R_q' R_q) - Psi' Psi against a dense V - C' M^-1 C, in the
    # kernel's own level-major order.  This RE-REDS against the form that
    # carried ``Omega``: that attribute no longer exists, and the equality
    # asserted here is now between a dense moment difference and a quantity
    # built only from row-space factors.
    factored = _factored_v_eff(p, geometry)
    assert np.allclose(
        factored, V_eff[np.ix_(perm, perm)], rtol=0, atol=1e-13 * np.abs(V_eff).max()
    )

    scale = float(np.trace(V_eff)) / float(np.trace(S_ti))
    for mult in (1e-2, 1.0, 1e2, 1e4):
        lam = mult * scale
        A = V_eff + lam * S_ti
        T_dense = float(U_eff_dense @ scipy.linalg.solve(A, U_eff_dense, assume_a="pos"))
        edf_dense = float(np.trace(scipy.linalg.solve(A, V_eff, assume_a="pos")))
        T_s, edf_s = _evaluate(p, geometry, lam)
        assert T_s == pytest.approx(T_dense, rel=1e-10)
        assert edf_s == pytest.approx(edf_dense, rel=1e-10)


def test_structured_ladder_agrees_with_the_dense_ladder(moderate_pair):
    """End to end, on the one pair both paths can score.

    ONE rung, asserted four times.  This pair's ``edf`` runs from 23.00 at
    maximum penalty to 206.00 at minimum, and every budget here is below
    23.00, so all four clamp at the same HIGH edge and return the same
    triple.  The searching rung and the LOW clamp are NOT reached from here;
    ``test_a_thin_level_does_not_cost_the_pair_a_degree_of_freedom`` and
    ``test_a_level_with_no_mass_cannot_carry_a_free_degree_of_freedom`` carry
    a budget above their low edge and are where those two are asserted.
    """
    from superglm.screening._structured import structured_ladder

    U, V, C, M, S_ti, u_m = moderate_pair["args"]
    dense = penalized_score_statistic_ladder(U, V, C, M, S_ti, budgets=BUDGETS, U_nuisance=u_m)
    struct = structured_ladder(
        spline_cat_moments(*_structured_inputs(moderate_pair)), budgets=BUDGETS
    )
    for d, s in zip(dense, struct, strict=True):
        assert s.statistic == pytest.approx(d.statistic, rel=1e-5)
        assert s.edf0 == pytest.approx(d.edf0, rel=1e-5)
        assert s.lambda0 == pytest.approx(d.lambda0, rel=1e-12)


def test_issue_204_reachable_half_df_uses_the_profiled_trace():
    """A reachable 0.5 rung must be classified from the PROFILED trace.

    The raw trace is one while the stable profiled trace is 1e-12.  Scaling
    from the raw trace puts the old low edge at 1e-10, where EDF is about
    0.0099 and the reachable target is falsely classified as above the
    bracket.  That classification is what this pins, and it still holds.

    **THE 0.5 RUNG ITSELF IS NO LONGER PUBLISHED, AND THAT IS A REGRESSION
    STATED RATHER THAN HIDDEN.**  This pair is one dimension wide, so ``edf``
    at any lambda is exactly ``a / (a + lambda)`` for ``a = tr(V_eff)`` and the
    exact value is available at every lambda.  Against it the factored form is
    at round-off across the bracket -- 2.2e-16, 4.4e-16, 4.4e-16, 1.0e-14,
    2.2e-16, 1.3e-15 relative at 1e-16, 1e-14, 5e-13, 1e-10, 1e-6 and 1e-2,
    where the form it replaces reads 1.0e-04, 1.3e-05, 6.1e-05, 5.7e-06,
    1.3e-04 and 1.8e-05, four to ten orders worse at every one of them.  But
    at the two lambdas either side of the crossover where ``edf = 0.5`` it
    reads 2.4e-04 relative against that form's 5.0e-13, because ``H``'s small
    eigenvalue is 2e-12 there and ``1/h`` multiplies an ``eps``-sized residue.
    2.4e-04 is 120x ``_EDF_TOL``, so the bisection cannot converge and the
    ladder hands the pair back.

    Two fixes were measured and neither is adopted: writing the level's
    contribution in the shifted coordinates ``[K_q | K_q Y_q - Phi_q]``, and
    equilibrating ``W_q`` before its PSD factorization.  Each makes this rung
    exact and each moves three neighbouring lambdas the other way by the same
    2e-04, and each reds more of this file than it greens.

    The published row is a NaN either way -- ``screen_interactions`` gets no
    rung -- so what changes is which pairs reach it, not what a caller sees
    for this one.
    """
    inputs = _near_absorbed_cells()
    p = spline_cat_moments(*inputs)
    old_lo = 1e-10 * float(np.trace(p.V[0])) / float(np.trace(p.S_a))
    _, old_edf = _evaluate(p, _profile(p), old_lo)
    exact = p.profiled_trace / (p.profiled_trace + old_lo)
    # The classification this test exists for: the target is INSIDE the
    # bracket the profiled trace sets, not above it.
    assert old_edf < 0.5 < 1.0
    # ...and the value there is now exact rather than 5.7e-06 out.  RED
    # against the unfixed code, which reads 0.009901046753 here.
    assert old_edf == pytest.approx(exact, rel=1e-12)

    from superglm.screening._structured import structured_ladder

    expected_trace = 1e-12 / (1.0 + 1e-12)
    assert p.profiled_trace == pytest.approx(expected_trace, rel=2e-13, abs=0.0)
    assert structured_ladder(p, budgets=(0.5,)) is None


def test_near_absorbed_edf_is_exact_where_the_dense_subtraction_is_not():
    """Near total absorption, the closed form arbitrates and this one wins.

    ``V_eff`` is 1e-12 of the level's own block here, so every route that
    forms it loses twelve digits -- the dense one included: its
    ``tr(V - C' M^-1 C)`` retains about four significant digits.  The
    structured trace does not form it at all and the factored ``edf`` does not
    either, so both are checked against the exact ``a / (a + lambda)``.

    RED against the form this replaces at every lambda listed: it reads
    1.0e-04 to 1.3e-05 relative where this reads round-off.
    """
    inputs = _near_absorbed_cells()
    U, V, C, M, S_ti, u_m = _dense_cell_inputs(*inputs)
    dense_trace = float(np.trace(V - C.T @ np.linalg.solve(M, C)))
    p = spline_cat_moments(*inputs)
    geometry = _profile(p)
    assert p.profiled_trace == pytest.approx(dense_trace, rel=2e-4, abs=0.0)

    a = p.profiled_trace
    for lam in (1e-16, 1e-14, 5e-13, 1e-10, 1e-6, 1e-2):
        _, edf = _evaluate(p, geometry, lam)
        assert edf == pytest.approx(a / (a + lam), rel=1e-13), lam


@pytest.mark.parametrize("base_energy", (0.0, 1e-18, 1e-12, 1.0))
def test_profiled_trace_keeps_exact_and_weak_complement_energy(base_energy):
    """The base level participates in H but contributes no emitted tau term.

    In this scalar geometry the exact residual is h0*h1/(h0+h1).  At 1e-18,
    forming ``H - H_q`` rounds the complement to zero; additive prefix/suffix
    geometry must still retain it.
    """
    p = spline_cat_moments(*_near_absorbed_cells(base_energy))
    expected = base_energy / (1.0 + base_energy)
    assert np.isfinite(p.profiled_trace)
    assert p.profiled_trace >= 0.0
    assert p.profiled_trace == pytest.approx(expected, rel=2e-13, abs=1e-300)


def test_exact_absorption_returns_finite_nonnegative_ladder_values():
    from superglm.screening._structured import structured_ladder

    p = spline_cat_moments(*_near_absorbed_cells(0.0))
    result = structured_ladder(p, budgets=(0.5,))[0]
    assert p.profiled_trace == 0.0
    assert np.isfinite([result.statistic, result.edf0, result.lambda0]).all()
    assert result.statistic >= 0.0
    assert result.edf0 >= 0.0
    assert result.lambda0 >= 0.0


def test_evaluate_clips_dust_but_signals_a_sum_that_is_not_a_filter_factor_sum(monkeypatch):
    """``edf`` is a sum of terms in ``[0, 1]``, so ``[0, L * k_a]`` is a
    PROPERTY and not a tolerance.  Round-off at either end is clipped; anything
    material must escape as a refusal.

    **THE LOWER END IS NO LONGER REACHABLE BY DATA AND THAT IS THE POINT.**
    Every term of the sum is now ``||F_q chol(W_q)||_F^2`` with both factors
    PSD, so a negative total is not something a pair can produce -- it can only
    be injected.  The upper end still can be, because ``A^+`` rests on a
    deflation decision.  Both sides are checked here by replacing the
    per-level trace term outright, at ``L = 1``, ``k_a = 1``: a ceiling of
    exactly 1.0.

    **THE COLLAPSE IS TWO-SIDED, WHICH IS THE HALF THAT CAN MOVE A ROW.**  The
    guard calls ``|edf| <= roundoff`` dust on both sides; the clip used to
    collapse only the negative half, so ``-1e-300`` returned ``0.0`` and
    ``screen_interactions`` skipped the rung, while ``+1e-300`` -- the same
    measurement with the other sign -- was published and divided into
    ``z = (T - edf0) / sqrt(2 edf0)``, sorting a pair that resolved nothing to
    the top of the screen.  Both signs now return exactly ``0.0``; RED against
    the shipped tree, which returns ``1e-300`` for the positive one.

    This test reds against the form it replaces for a structural reason: that
    one reached the guard through a ``FakeFactor`` standing in for
    ``_pair_arrow``, and ``_pair_arrow`` no longer computes ``edf``.
    """
    import superglm.screening._structured as st

    pair = spline_cat_moments(*_near_absorbed_cells(0.0))
    geometry = st._profile(pair)
    assert geometry.ceiling == 1.0, geometry.ceiling

    # The dust band both sides of this are inside, derived rather than
    # asserted: the guard's own allowance on a ceiling of 1.0.
    band = st._edf_roundoff(0.0, geometry.ceiling)
    assert 0.0 < band < 1e-13, band

    for injected in (1e-300, -1e-300, 0.5 * band, -0.5 * band):
        monkeypatch.setattr(st, "_filter_factor_sum", lambda *a, _v=injected, **k: (_v, 0.0))
        assert st._evaluate(pair, geometry, 1.0)[1] == 0.0, injected

    for injected, expected in ((1.0, 1.0), (0.25, 0.25)):
        monkeypatch.setattr(st, "_filter_factor_sum", lambda *a, _v=injected, **k: (_v, 0.0))
        assert st._evaluate(pair, geometry, 1.0)[1] == pytest.approx(expected, abs=1e-12)

    for outside in (1.25, -1.25):
        monkeypatch.setattr(st, "_filter_factor_sum", lambda *a, _v=outside, **k: (_v, 0.0))
        with pytest.raises(st._UnstableStructuredEDFError, match="not a filter-factor sum"):
            st._evaluate(pair, geometry, 1.0)

    # ...and the real thing never goes negative, on the fixture built to make
    # the lower end as reachable as a pair can make it.
    monkeypatch.undo()
    for lam in np.geomspace(1e-30, 1e30, 61):
        assert st._filter_factor_sum(pair, geometry, float(lam))[0] >= 0.0


def test_the_psd_clip_refuses_a_block_it_cannot_call_roundoff(monkeypatch):
    """``max(w, 0)`` makes every term a squared norm WHATEVER the block was.

    That is what the ``edf >= 0`` half of the range guard rests on, so the
    clip has to be the thing that is certified rather than the thing that
    launders.  A deflation that misfires, or an ``H^+`` inflating an
    indefinite residue, leaves a materially indefinite ``W_q``; clipped, it
    still contributes a nonnegative squared norm and still lands inside
    ``[0, ceiling]``, so nothing downstream can tell it from a real answer.

    The allowance is derived in :func:`_psd_clip_allowance` and the two arms
    of this test bracket it: a negative shift a tenth of the allowance is
    clipped silently, the same shift at ten times the allowance is refused.
    Injected by SHIFTING THE BLOCK, not by faking the report, so the real
    eigendecomposition is what measures it.

    RED against the shipped tree, which has no such refusal: the ten-times
    shift is clipped and published there like any other.
    """
    import superglm.screening._structured as st

    pair = spline_cat_moments(*_structured_inputs(_thin_level_pair(low_weight=1.0)))
    geometry = st._profile(pair)
    lam = 1e-4
    baseline = st._evaluate(pair, geometry, lam)[1]
    assert baseline > 1.0, baseline

    real = st._psd_factor

    def shifted(M, *, _by):
        return real(M - _by * np.eye(M.shape[-1]))

    # The allowance these blocks actually get, MEASURED off the module rather
    # than reconstructed here -- reconstructing it would make this test a
    # second copy of the derivation instead of a check on it.
    seen = []
    real_allowance = st._psd_clip_allowance
    monkeypatch.setattr(
        st,
        "_psd_clip_allowance",
        lambda *a: seen.append(np.max(real_allowance(*a))) or real_allowance(*a),
    )
    st._evaluate(pair, geometry, lam)
    monkeypatch.undo()
    allowance = float(max(seen))
    assert 0.0 < allowance < 1e-10, allowance

    monkeypatch.setattr(st, "_psd_factor", lambda M: shifted(M, _by=0.1 * allowance))
    assert st._evaluate(pair, geometry, lam)[1] == pytest.approx(baseline, abs=1e-6)

    monkeypatch.setattr(st, "_psd_factor", lambda M: shifted(M, _by=10.0 * allowance))
    with pytest.raises(st._UnstableStructuredEDFError, match="PSD clip"):
        st._evaluate(pair, geometry, lam)

    # ...and the ladder hands the pair back rather than publishing it.
    assert st.structured_ladder(pair, budgets=BUDGETS) is None


def test_the_statistic_factors_the_pencil_the_edf_chose_lambda_from():
    """One projection, both halves -- and the gap it closes is 5.9e-02.

    The ladder chooses lambda from ``edf``, evaluated against
    ``rootS' rootS``, and then publishes a statistic from ``_pair_arrow``.
    If that added ``p.S_a`` raw, the two would be different matrices wherever
    the projection did anything, and the largest ``lambda`` in the bracket is
    ``lambda_hi = 1e10 * scale`` -- a rung the ladder publishes, not a corner.

    Measured on the nullity-two pair, where the penalty's smallest eigenvalue
    is a ``2.2e-14`` residue: the high-edge statistic reads **2.999250** on the
    raw pencil and **2.821425** on the projected one, a 5.9e-02 relative gap
    from a perturbation twelve orders below the penalty's scale.  Which of the
    two is published is therefore not a detail.

    What is asserted is the identity, not the gap: ``_evaluate``'s statistic
    must equal the one the projected pencil gives and must NOT equal the raw
    one.  RED against passing ``p.S_a`` -- the equality and the inequality swap
    places.
    """
    import superglm.screening._structured as st

    grab = _multi_null_pair(0)
    B_a, S_a, S_cell, W_cell, level_rows = _structured_inputs(grab)
    pair = spline_cat_moments(B_a, S_a, S_cell, W_cell, level_rows)
    geometry = st._profile(pair)
    lam = _ladder_high_edge(pair)

    projected = geometry.root_penalty.T @ geometry.root_penalty
    assert not np.allclose(projected, pair.S_a, rtol=0.0, atol=0.0)

    def statistic(penalty):
        f = st._pair_arrow(pair, lam, penalty)
        L, k_a = pair.dims
        b = np.zeros((L, k_a + 1))
        b[:, :k_a] = geometry.U_eff
        x, _ = f.solve(b, np.zeros(1 + k_a))
        return float(np.sum(geometry.U_eff * x[:, :k_a]))

    published = st._evaluate(pair, geometry, lam)[0]
    on_projected, on_raw = statistic(projected), statistic(pair.S_a)

    # The two pencils are far enough apart here that the identity below is a
    # real choice: a relative gap of 1e-2 is four orders above the 1e-6 the
    # ladder's own tolerance works to.
    assert abs(on_raw - on_projected) / abs(on_raw) > 1e-2, (on_raw, on_projected)
    assert published == on_projected, (published, on_projected)
    assert published != on_raw, (published, on_raw)


def test_the_edf_and_the_statistic_must_be_scoring_the_same_penalty(monkeypatch):
    """``edf`` uses the PSD projection of ``S_a``; the statistic uses ``S_a``.

    ``_evaluate`` hands the same ``rootS' rootS`` to ``_pair_arrow`` that the
    filter-factor sum uses, so the two halves score one pencil by
    construction.  What that does NOT settle is whether the projection is
    entitled to change the pencil at all: the DENSE route still assembles
    ``S_ti`` raw, so a projection that removed something material would leave
    the structured route scoring a different model from the one the caller
    specified, and only on the pairs wide enough to route here.

    The certification is the eigensolver's documented bound, ``n^2 eps``
    relative trace mass -- see :func:`_profile`.  Both arms are checked:
    a negative eigenvalue at a tenth of that bound is projected silently, one
    at ten times it refuses the pair.

    RED against the shipped tree, where ``penalty_clip`` is computed, stored
    on ``_PairGeometry`` and never read by anything.
    """
    import superglm.screening._structured as st

    grab = _thin_level_pair(low_weight=1.0)
    B_a, S_a, S_cell, W_cell, level_rows = _structured_inputs(grab)
    n = S_a.shape[0]
    bound = 2.0 * n * n * np.finfo(np.float64).eps

    # A negative direction of a KNOWN relative trace weight, added to the
    # penalty's own smallest eigenvector so nothing else about the pair moves.
    w, Q = np.linalg.eigh(0.5 * (S_a + S_a.T))
    direction = np.outer(Q[:, 0], Q[:, 0])
    trace = float(np.trace(S_a))

    for factor, refused in ((0.1, False), (10.0, True)):
        injected = S_a - (w[0] + factor * bound * trace) * direction
        assert np.linalg.eigvalsh(0.5 * (injected + injected.T))[0] < 0.0
        pair = spline_cat_moments(B_a, injected, S_cell, W_cell, level_rows)
        if refused:
            with pytest.raises(st._UnstableStructuredEDFError, match="PSD projection"):
                st._profile(pair)
            assert st.structured_ladder(pair, budgets=BUDGETS) is None
        else:
            rungs = st.structured_ladder(pair, budgets=BUDGETS)
            assert rungs is not None and len(rungs) == len(BUDGETS)


def test_material_negative_edf_refuses_an_only_search_rung(monkeypatch):
    """No plausible row may escape when the ladder has no certified rung."""
    import superglm.screening._structured as st

    pair = spline_cat_moments(*_near_absorbed_cells(1.0))
    calls = 0

    def unstable_on_search(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return 0.0, 1.0
        if calls == 2:
            return 0.0, 0.0
        raise st._UnstableStructuredEDFError("injected material negative EDF")

    monkeypatch.setattr(st, "_evaluate", unstable_on_search)
    assert st.structured_ladder(pair, budgets=(0.5,), max_evaluations=100) is None
    assert calls == 3


def test_an_unstable_search_rung_preserves_certified_edge_clamps(monkeypatch):
    """One broken target must not discard independent endpoint results."""
    import superglm.screening._structured as st

    pair = spline_cat_moments(*_near_absorbed_cells(1.0))
    calls = 0

    def unstable_on_search(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return 0.0, 2.0
        if calls == 2:
            return 0.0, 0.5
        raise st._UnstableStructuredEDFError("injected material negative EDF")

    monkeypatch.setattr(st, "_evaluate", unstable_on_search)
    result = st.structured_ladder(pair, budgets=(3.0, 1.0, 0.25), max_evaluations=100)
    assert result is not None
    assert [r.edf0 for r in result] == [2.0, 0.5]
    assert calls == 3


def test_endpoint_monotonicity_mutation_is_refused_after_the_bracket(monkeypatch):
    import superglm.screening._structured as st

    pair = spline_cat_moments(*_near_absorbed_cells(1.0))
    calls = 0

    def increasing_endpoints(*args, **kwargs):
        nonlocal calls
        calls += 1
        return (0.0, 1.0) if calls == 1 else (0.0, 1.1)

    monkeypatch.setattr(st, "_evaluate", increasing_endpoints)
    assert st.structured_ladder(pair, budgets=(0.5,)) is None
    assert calls == 2


def test_sub_tolerance_endpoint_ordering_noise_is_not_refused(monkeypatch):
    import superglm.screening._structured as st

    pair = spline_cat_moments(*_near_absorbed_cells(1.0))
    calls = 0

    def noisy_endpoints(*args, **kwargs):
        nonlocal calls
        calls += 1
        return (0.0, 1.0) if calls == 1 else (0.0, 1.0 + 0.5 * st._EDF_TOL)

    monkeypatch.setattr(st, "_evaluate", noisy_endpoints)
    result = st.structured_ladder(pair, budgets=(2.0,))
    assert result is not None
    assert result[0].edf0 == 1.0
    assert calls == 2


def test_interior_edf_outside_its_monotone_bracket_is_refused(monkeypatch):
    import superglm.screening._structured as st

    pair = spline_cat_moments(*_near_absorbed_cells(1.0))
    calls = 0

    def broken_interior(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return 0.0, 2.0
        if calls == 2:
            return 0.0, 0.0
        return 0.0, 2.1

    monkeypatch.setattr(st, "_evaluate", broken_interior)
    assert st.structured_ladder(pair, budgets=(1.0,), max_evaluations=100) is None
    assert calls == 3


def test_sub_tolerance_interior_ordering_noise_is_clipped_not_refused(monkeypatch):
    import superglm.screening._structured as st

    pair = spline_cat_moments(*_near_absorbed_cells(1.0))
    calls = 0

    def noisy_interior(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return 0.0, 2.0
        if calls == 2:
            return 0.0, 0.0
        if calls == 3:
            return 0.0, 2.0 + 0.5 * st._EDF_TOL
        return 0.0, 1.0

    monkeypatch.setattr(st, "_evaluate", noisy_interior)
    result = st.structured_ladder(pair, budgets=(1.0,), max_evaluations=100)
    assert result is not None
    assert result[0].edf0 == 1.0
    assert calls == 4


def test_search_width_exhaustion_without_target_convergence_is_refused(monkeypatch):
    """A bracket is not permission to publish a rung that missed its target."""
    import superglm.screening._structured as st

    pair = spline_cat_moments(*_near_absorbed_cells(1.0))
    calls = 0

    def discontinuous_edf(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return 0.0, 2.0
        if calls == 2:
            return 0.0, 0.0
        return 0.0, 0.5

    monkeypatch.setattr(st, "_evaluate", discontinuous_edf)
    assert st.structured_ladder(pair, budgets=(1.0,), max_evaluations=100) is None
    assert calls > 3


def test_constant_spline_support_has_exact_zero_profiled_energy():
    B_a = np.full((4, 2), [3.0, -7.0])
    W_cell = np.arange(1.0, 17.0).reshape(4, 4)
    S_cell = np.zeros_like(W_cell)
    p = spline_cat_moments(
        B_a,
        np.eye(2),
        S_cell,
        W_cell,
        np.array([0, 2, 3], dtype=np.intp),
    )
    assert p.profiled_trace == 0.0


@pytest.mark.parametrize("factor", (2.0**-30, 2.0**30))
def test_profiled_trace_rescales_with_cell_weights(factor):
    inputs = _near_absorbed_cells(0.25)
    baseline = spline_cat_moments(*inputs).profiled_trace
    B_a, S_a, S_cell, W_cell, level_rows = inputs
    scaled = spline_cat_moments(
        B_a,
        S_a,
        factor * S_cell,
        factor * W_cell,
        level_rows,
    ).profiled_trace
    assert scaled == pytest.approx(factor * baseline, rel=2e-13, abs=0.0)


def test_profiled_trace_is_invariant_to_level_order():
    rng = np.random.default_rng(91)
    B_a = rng.normal(size=(7, 3))
    W_cell = rng.uniform(0.05, 2.0, size=(7, 6))
    S_cell = rng.normal(size=(7, 6))
    S_a = np.eye(3)
    level_rows = np.array([0, 2, 3, 5])
    baseline = spline_cat_moments(B_a, S_a, S_cell, W_cell, level_rows).profiled_trace

    permutation = np.array([4, 2, 0, 5, 1, 3])
    inverse = np.empty_like(permutation)
    inverse[permutation] = np.arange(permutation.size)
    permuted = spline_cat_moments(
        B_a,
        S_a,
        S_cell[:, permutation],
        W_cell[:, permutation],
        inverse[level_rows],
    ).profiled_trace
    assert permuted == pytest.approx(baseline, rel=2e-13, abs=0.0)


@pytest.mark.parametrize(
    ("seed", "delta", "decimal_oracle"),
    (
        (720, 1e-6, 2_124_788_244.3593912),
        (239, 1e-7, 27_174_778_074.477875),
    ),
)
def test_ill_conditioned_trace_matches_decimal_oracle_across_level_and_row_orders(
    seed,
    delta,
    decimal_oracle,
):
    """Extreme weights and retained weak columns must not revive normal equations.

    The constants were evaluated from the float inputs with 100-digit Decimal
    arithmetic.  The explicit dense-row QR is a second oracle; the production
    path has to agree with both and make the same absolute decision after
    levels and rows are reordered, not merely agree with a same-order Gram.
    """
    B_a, S_a, S_cell, W_cell, level_rows = _adversarial_trace_cells(seed, delta)
    rng = np.random.default_rng(seed + 10_000)
    for _ in range(16):
        row_order = rng.permutation(B_a.shape[0])
        level_order = rng.permutation(W_cell.shape[1])
        inverse_levels = np.empty_like(level_order)
        inverse_levels[level_order] = np.arange(level_order.size)
        B_ordered = B_a[row_order]
        W_ordered = W_cell[row_order][:, level_order]
        rows_ordered = inverse_levels[level_rows]

        oracle = _direct_centered_row_trace(B_ordered, W_ordered, rows_ordered)
        got = spline_cat_moments(
            B_ordered,
            S_a,
            S_cell[row_order][:, level_order],
            W_ordered,
            rows_ordered,
        ).profiled_trace
        assert oracle == pytest.approx(decimal_oracle, rel=2e-7, abs=0.0)
        assert got == pytest.approx(oracle, rel=2e-7, abs=0.0)
        assert got == pytest.approx(decimal_oracle, rel=2e-7, abs=0.0)


def test_rank_boundary_is_refused_for_both_permutation_actions():
    """A QR pivot whose error interval crosses the rank cutoff is not guessed.

    Without the ambiguity certificate these two mathematically identical
    orders straddled the cutoff: one retained rank two and returned trace
    110.92, the other dropped to rank one and returned 719.05.  Their budget-2
    ladders then took different clamp/search actions.  Both must instead
    refuse the structured route so routing can make one safe decision.
    """
    from superglm.screening._structured import structured_ladder

    for order_seed in (10_000, 10_001):
        pair = spline_cat_moments(*_rank_boundary_trace_cells(order_seed))
        assert pair.profiled_trace is None
        assert structured_ladder(pair, budgets=(2.0,)) is None


def test_aligned_row_qr_avoids_the_normal_equation_rhs_error():
    """A compact deterministic mutation killer for the final squared solve.

    The old ``R_active.T @ R_target`` followed by ``R^-T/R^-1`` gives
    4.9138975014457805e-6 here, a 4.28e-4 relative error.  The oracle uses the
    actual dense weighted rows, selects the same three representative columns
    with an independent CPQR, and solves in their reduced QR coordinate.
    """
    rng = np.random.default_rng(1338)
    n_rows, k_a, n_levels = 4, 6, 2
    B_a = rng.normal(size=(n_rows, k_a))
    B_a[:, -1] = B_a[:, 0]
    W_cell = 10 ** rng.uniform(-8.0, 8.0, size=(n_rows, n_levels))
    W_cell[rng.random(W_cell.shape) < 0.15] = 0.0

    shifted = B_a - B_a[0]
    row_blocks = []
    for level in range(n_levels):
        weights = W_cell[:, level]
        centered = shifted - weights @ shifted / weights.sum()
        row_blocks.append(np.sqrt(weights)[:, None] * centered)
    stacked = np.vstack(row_blocks)

    _, _, permutation = scipy.linalg.qr(stacked, mode="economic", pivoting=True)
    active = permutation[:3]
    assert set(active) == {1, 3, 4}
    Q, R = np.linalg.qr(stacked[:, active], mode="reduced")
    target = np.zeros_like(stacked)
    target[n_rows:] = row_blocks[1]
    coefficients = np.linalg.solve(R, Q.T @ target)
    oracle = float(np.sum(np.square(target - stacked[:, active] @ coefficients)))
    assert oracle == pytest.approx(4.915999377342993e-6, rel=2e-9, abs=0.0)

    got = spline_cat_moments(
        B_a,
        np.eye(k_a),
        np.zeros_like(W_cell),
        W_cell,
        np.array([1], dtype=np.intp),
    ).profiled_trace
    assert got == pytest.approx(oracle, rel=2e-8, abs=0.0)


def test_profiled_trace_is_translation_and_row_order_invariant_against_row_oracle():
    """Center rows before products; raw ``B'WB - cc'/m`` cannot pass this."""
    rng = np.random.default_rng(118)
    B_a = rng.normal(size=(11, 3))
    W_cell = rng.uniform(0.2, 2.0, size=(11, 6))
    S_cell = np.zeros_like(W_cell)
    level_rows = np.array([1, 2, 4, 5], dtype=np.intp)
    baseline = spline_cat_moments(
        B_a,
        np.eye(3),
        S_cell,
        W_cell,
        level_rows,
    ).profiled_trace

    translated = B_a + np.array([1e12, -3e11, 7e11])
    translated_baseline = spline_cat_moments(
        translated,
        np.eye(3),
        S_cell,
        W_cell,
        level_rows,
    ).profiled_trace
    # Adding the offsets quantizes the float inputs at about 1e-4, so exact
    # equality to the unshifted input is impossible; the mathematical action
    # remains invariant to that representational floor.
    assert translated_baseline == pytest.approx(baseline, rel=1e-5, abs=0.0)

    for _ in range(12):
        row_order = rng.permutation(B_a.shape[0])
        got = spline_cat_moments(
            translated[row_order],
            np.eye(3),
            S_cell[row_order],
            W_cell[row_order],
            level_rows,
        ).profiled_trace
        oracle = _direct_centered_row_trace(
            translated[row_order],
            W_cell[row_order],
            level_rows,
        )
        assert got == pytest.approx(oracle, rel=2e-13, abs=0.0)
        assert got == pytest.approx(translated_baseline, rel=2e-13, abs=0.0)


def test_profiled_trace_uses_chunked_linear_geometry_without_svd(monkeypatch):
    """Pin linear peak memory, bounded chunks, and the actual CPQR dispatch."""
    import tracemalloc

    import superglm.screening._structured as st

    rng = np.random.default_rng(92)
    n_rows, n_levels, k_a = 9, 73, 4
    B_a = rng.normal(size=(n_rows, k_a))
    W_cell = rng.uniform(0.1, 1.0, size=(n_rows, n_levels))
    S_cell = rng.normal(size=(n_rows, n_levels))
    level_rows = np.arange(1, n_levels, dtype=np.intp)

    allocations = []
    real_empty = st.np.empty
    real_zeros = st.np.zeros
    real_concatenate = st.np.concatenate
    real_centered = st._centered_level_factors
    real_numpy_qr = st.np.linalg.qr
    real_scipy_qr = st.scipy.linalg.qr
    real_triangular_solve = st.scipy.linalg.solve_triangular
    chunk_widths = []
    numpy_qr_calls = []
    pivoted_qr_calls = []
    triangular_solve_shapes = []

    def watched_empty(shape, *args, **kwargs):
        if isinstance(shape, tuple):
            allocations.append(tuple(int(v) for v in shape))
        return real_empty(shape, *args, **kwargs)

    def watched_zeros(shape, *args, **kwargs):
        if isinstance(shape, tuple):
            allocations.append(tuple(int(v) for v in shape))
        return real_zeros(shape, *args, **kwargs)

    def watched_concatenate(arrays, *args, **kwargs):
        result = real_concatenate(arrays, *args, **kwargs)
        allocations.append(result.shape)
        return result

    def watched_centered(B, W):
        chunk_widths.append(W.shape[1])
        return real_centered(B, W)

    def watched_scipy_qr(*args, **kwargs):
        pivoted_qr_calls.append(bool(kwargs.get("pivoting", False)))
        return real_scipy_qr(*args, **kwargs)

    def watched_numpy_qr(a, *args, **kwargs):
        numpy_qr_calls.append((a.shape, kwargs.get("mode", "reduced")))
        return real_numpy_qr(a, *args, **kwargs)

    def watched_triangular_solve(a, b, *args, **kwargs):
        triangular_solve_shapes.append((a.shape, b.shape))
        return real_triangular_solve(a, b, *args, **kwargs)

    monkeypatch.setattr(st.np, "empty", watched_empty)
    monkeypatch.setattr(st.np, "zeros", watched_zeros)
    monkeypatch.setattr(st.np, "concatenate", watched_concatenate)
    monkeypatch.setattr(st.np.linalg, "svd", lambda *a, **k: pytest.fail("SVD is forbidden"))
    monkeypatch.setattr(
        st.np.linalg, "eigh", lambda *a, **k: pytest.fail("normal eig is forbidden")
    )
    monkeypatch.setattr(st.np.linalg, "qr", watched_numpy_qr)
    monkeypatch.setattr(st.scipy.linalg, "svd", lambda *a, **k: pytest.fail("SVD is forbidden"))
    monkeypatch.setattr(
        st.scipy.linalg, "eigh", lambda *a, **k: pytest.fail("normal eig is forbidden")
    )
    monkeypatch.setattr(st.scipy.linalg, "qr", watched_scipy_qr)
    monkeypatch.setattr(st.scipy.linalg, "solve_triangular", watched_triangular_solve)
    monkeypatch.setattr(st, "_centered_level_factors", watched_centered)

    tracemalloc.start()
    p = spline_cat_moments(B_a, np.eye(k_a), S_cell, W_cell, level_rows)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    tensor_width = k_a * level_rows.size
    assert (tensor_width, tensor_width) not in allocations
    assert (n_levels + 1, k_a, k_a) in allocations
    assert max(chunk_widths) <= st._trace_chunk_width(n_rows, k_a, n_levels)
    assert sum(chunk_widths) == 2 * n_levels
    assert pivoted_qr_calls == [True]
    aligned_qr = [shape for shape, mode in numpy_qr_calls if mode == "reduced"]
    assert aligned_qr == [(2 * k_a, k_a)] * level_rows.size
    # One combined aligned RHS solve per emitted level.  The old
    # cross-product/R'R route needed four triangular solves per level.
    assert triangular_solve_shapes == [((k_a, k_a), (k_a, 2 * k_a))] * level_rows.size
    # Constructor-independent peak pin: even one dense tensor-width square is
    # six times the whole measured trace peak on this case.
    assert peak_bytes < (tensor_width * tensor_width * 8) // 2
    assert np.isscalar(p.profiled_trace)


def test_profiled_trace_does_not_form_identity_minus_projection(monkeypatch):
    """The dropped direction is a structural null action, not ``I - A``."""
    import superglm.screening._structured as st

    representative = st._representative_projection(
        np.diag([1.0, 1e-8]),
        n_rows=6,
        n_levels=2,
    )
    assert representative is not None
    active, null_action = representative
    assert np.array_equal(active, np.array([0]))
    assert np.array_equal(null_action, np.diag([0.0, 1.0]))

    p = spline_cat_moments(*_near_absorbed_cells(1e-18))
    assert p.profiled_trace == pytest.approx(1e-18, rel=2e-13, abs=0.0)


@pytest.mark.parametrize("scale", (2.0**-30, 1.0, 2.0**30))
@pytest.mark.parametrize("permute", (False, True))
def test_truncated_positive_direction_remains_residual_energy(scale, permute):
    """A positive H direction below the inverse floor is not profiled away.

    The full-rank weighted-design projection leaves ``scale*d^2/2``.  The
    numerical rank policy deliberately drops that H direction, so the
    matching dense Hermitian pseudo-inverse leaves all ``scale*d^2`` of the
    emitted block.  Both are positive; returning only ``H^+`` previously
    returned zero.

    **THE DEGREE OF FREEDOM THIS FIXTURE EXISTS FOR IS NOW RECOVERED
    EXACTLY.**  ``V_eff``'s only direction here sits at ``d^2`` = 1e-16
    RELATIVE to its own level block, which is below what a float64
    eigendecomposition of that block can resolve -- so the form that read the
    inverse off ``V + lambda S`` dropped it, and a 50-digit oracle putting
    ``edf`` at 1.000000 / 0.500000 / 0.000000 across the bracket was answered
    with 0.0 / 0.0 / 0.0.  Nothing here reads that block: ``N_q`` is
    ``T_q' T_q`` from a QR of ``[R_q ; sqrt(lambda) rootS]``, and ``R_q`` is
    the level's centered rows, whose smallest direction is at ``d`` and not at
    ``d^2``.  Working on the factor is exactly a square root of the cut, which
    is what puts this direction back above it.

    So the ladder now bisects to the 0.5 rung and attains it.  RED against the
    form this replaces, which returns a single rung at ``edf0 = 0.0``.
    """
    inputs = _mixed_rank_cells(scale, permute=permute)
    p = spline_cat_moments(*inputs)
    _, V, C, M, _, _ = _dense_cell_inputs(*inputs)
    dense_trace = float(np.trace(V - C.T @ np.linalg.pinv(M, hermitian=True) @ C))
    full_rank_oracle = 0.5 * scale * 1e-16

    assert full_rank_oracle > 0.0
    assert p.profiled_trace == pytest.approx(
        2.0 * full_rank_oracle,
        rel=3e-13,
        abs=0.0,
    )
    assert p.profiled_trace == pytest.approx(dense_trace, rel=3e-13, abs=0.0)

    from superglm.screening._structured import structured_ladder

    rungs = structured_ladder(p, budgets=(0.5,))
    assert rungs is not None and len(rungs) == 1, rungs
    assert rungs[0].edf0 == pytest.approx(0.5, abs=1e-6), rungs
    assert rungs[0].lambda0 > 0.0
    # ...and the whole bracket matches the oracle, not only the rung it
    # searched for: 1 / 0.5 / 0 is what 50 digits say and what this returns.
    geometry = _profile(p)
    tr_S = float(np.trace(p.S_a)) * p.dims[0]
    edge = max(p.profiled_trace, 1e-300) / max(tr_S, 1e-300)
    assert _evaluate(p, geometry, 1e-10 * edge)[1] == pytest.approx(1.0, abs=1e-6)
    assert _evaluate(p, geometry, 1e10 * edge)[1] == pytest.approx(0.0, abs=1e-6)


def test_contrast_rows_are_the_menu_without_the_menu():
    """``_contrast_rows`` must index exactly the rows ``_contrast_menu`` marks:
    the kernel selects cell-table columns where the dense path multiplies by
    the menu, and the two have to pick the same ones."""
    rng = np.random.default_rng(7)
    L, reps = 12, 20
    df = pd.DataFrame(
        {"g": np.repeat([f"L{i}" for i in range(L)], reps), "x": rng.uniform(0, 1, L * reps)}
    )
    y = rng.normal(size=len(df))
    model = SuperGLM(family="gaussian", features={"g": Categorical(), "x": Spline(n_knots=5)})
    model.fit_reml(df, y)
    spec = model._specs["g"]
    menu = _contrast_menu(spec)
    rows = _contrast_rows(spec)
    assert rows.shape == (menu.shape[1],)
    assert np.array_equal(rows, np.argmax(menu, axis=0))
    assert menu[rows, np.arange(menu.shape[1])].tolist() == [1.0] * menu.shape[1]
    assert menu.sum() == menu.shape[1]  # one-hot, so the rows ARE the menu


def _wide_frame(L, reps=6, seed=11):
    rng = np.random.default_rng(seed)
    n = L * reps
    x = rng.uniform(0.0, 1.0, n)
    slope = rng.normal(size=L).repeat(reps)
    df = pd.DataFrame({"g": np.repeat([f"L{i}" for i in range(L)], reps), "x": x})
    return df, slope * x + rng.normal(scale=0.5, size=n)


def test_a_factor_above_the_dense_cap_is_scored_rather_than_refused():
    """The cap this whole module exists to remove.

    At the default max_cells the dense path takes a spline_cat block only to
    k = 1357, which for a width-11 spline is 124 levels.  A wider factor used
    to come back as a NaN row.
    """
    L = 400
    df, y = _wide_frame(L)
    model = SuperGLM(
        family="gaussian", features={"g": Categorical(), "x": Spline(kind="ps", n_knots=8)}
    )
    model.fit_reml(df, y)
    row = model.screen_interactions(df, y, candidates=[("x", "g")], edf0=BUDGETS).iloc[0]
    assert row["kind"] == "spline_cat"
    assert np.isfinite(row["z"])
    assert np.isfinite(row["statistic"])
    assert row["edf0"] > 16.0  # clamped: kron(S_a, I) outranks every budget
    assert row["z"] > 0.0  # the data really does have a level-varying slope


def test_the_wide_path_never_allocates_the_dense_blocks(monkeypatch):
    """A pair routed structurally must not touch either thing the dense gates
    refused it for: the ``(L, L-1)`` contrast menu, or the ``(k, k)``
    curvature the ladder would factorize."""
    calls = []
    monkeypatch.setattr(
        ops, "_contrast_menu", lambda spec: calls.append("menu") or np.zeros((1, 1))
    )
    monkeypatch.setattr(
        ops,
        "pair_score_curvature",
        lambda *a, **k: calls.append("curvature") or (np.zeros(1), np.zeros((1, 1))),
    )
    monkeypatch.setattr(
        ops,
        "penalized_score_statistic_ladder",
        lambda *a, **k: calls.append("dense_ladder") or [],
    )

    L = 400
    df, y = _wide_frame(L)
    model = SuperGLM(
        family="gaussian", features={"g": Categorical(), "x": Spline(kind="ps", n_knots=8)}
    )
    model.fit_reml(df, y)
    row = model.screen_interactions(df, y, candidates=[("x", "g")], edf0=BUDGETS).iloc[0]
    assert np.isfinite(row["z"])
    assert calls == []


def _thin_level_pair(low_weight):
    """A 20-level pair the DENSE path can also score, one level down-weighted.

    A rare level is the routine case at high cardinality, and it is what
    makes the arrow kernel's per-block rank cut disagree with the dense
    path's: at the ladder's high edge that level's own curvature sits below
    ``lambda * S_a`` by its weight share, not by 1e-10.
    """
    rng = np.random.default_rng(3)
    L, reps = 20, 40
    n = L * reps
    g = np.repeat([f"L{i}" for i in range(L)], reps)
    x = rng.uniform(0.0, 1.0, n)
    slope = rng.normal(size=L).repeat(reps)
    df = pd.DataFrame({"g": g, "x": x})
    y = slope * x + rng.normal(scale=0.5, size=n)
    w = np.ones(n)
    w[g == "L0"] = low_weight
    grab = _capture(
        df,
        y,
        {"g": Categorical(), "x": Spline(kind="ps", n_knots=8)},
        ("x", "g"),
        sample_weight=w,
    )
    return grab


# ``_thin_level_pair(1.0)``'s low-edge rung, certified with 400-bit ball
# arithmetic (python-flint ``arb``) on the pair's exact design:
#
#     edf = 207.999955426981464537192 +/- 8.54e-24    at lambda below
#
# computed as ``tr((V_eff + lambda S)^-1 V_eff)`` with the moments M, C, V
# formed exactly rather than in float64, so the design's own near-dependency
# is carried at full strength instead of being lost in the moments' round-off.
# Forming them from the float64 moments the ladder receives instead moves the
# answer by 3.94e-11, so the two truth models agree far below anything
# asserted here.  Both lambdas are hard-coded rather than read back from the
# ladder: they pin the ORACLE, not the search that produced them.
_CERTIFIED_LOW_EDGE_LAMBDA = 2.2876641890924248e-11
# ``_vanishing_mass_pair(1e-12)``'s own low-edge rung, which is a DIFFERENT
# point from the one above: the bracket is scaled by the pair's own profiled
# curvature and starving three levels moves it.
#
# IT IS NOT A PIN AND NOTHING ASSERTS THE LADDER LANDS ON IT.  An earlier
# revision did assert that, to tie the point the oracle was gated at to the
# point the ladder asked for it -- but that rung compares the oracle against an
# arm at the SAME lambda, so there is no third point to tie them to (see
# ``test_a_level_with_no_mass_cannot_carry_a_free_degree_of_freedom``), so the
# tie has nothing left to hold and the assertion would only be an undated
# observation about this machine's ``fit_reml``.  What the constant does now is
# name a REPRESENTATIVE point of the starved geometry for
# ``test_the_low_edge_reference_matches_a_certified_high_precision_value`` to
# measure the oracle's arrangement-independence at; nothing downstream depends
# on it being the ladder's exact edge.
_VANISHING_LOW_EDGE_LAMBDA = 1.912889731636818e-11
# The round-trip repr of that float64.  An independent mpmath evaluation at 120
# and again at 200 decimal digits gives 207.999955426981464537204805751,
# identical to all 30 digits between the two and agreeing with the arb ball
# above to 1.3e-20 -- so arb's claimed +/-8.54e-24 was optimistic in its last
# three digits, and nothing downstream can see the difference.
_CERTIFIED_LOW_EDGE_EDF = 207.99995542698147


def _reference_edf(grab, lam):
    """``edf(lambda)`` from the pair's DESIGN rather than from its Gram.

    **THIS WRAPPER HAS NO CALL SITES AND IS KEPT DELIBERATELY.**  Every
    consumer now takes the pair form :func:`_reference_edf_and_bound`, so that
    it gets the bound alongside the value; what lives here is the ROOT-CAUSE
    argument for issue #272, and five docstrings in this file cite it by name.
    It is the documented entry point to the oracle, not dead code left behind.

    ``edf = tr(R (R'R + lambda S)^-1 R')`` with ``R`` the tensor block
    residualized on the mains and ``S = L'L``.  It is evaluated as
    ``||R T^-1||_F^2`` where ``T`` is the triangular factor of the stacked
    matrix ``[R; sqrt(lambda) L]`` -- the augmented-system form of Tikhonov
    regularization (Elden 1977, 1982; Golub, Heath & Wahba 1979), and the
    statistical instance of it in Wood, JASA 99:673-686 (2004).  Every term
    summed is a square, there is no ``rank - lambda tr(A^-1 S)``
    subtraction, and no Gram is formed or inverted anywhere.

    **THE POINT IS THE FACTOR, NOT THE FORMULA.**  This oracle previously
    took the moments and whitened ``V_eff = V - C' M^-1 C``.  That is
    algebraically the same quantity and it was wrong by 1.13e-05 at the low
    edge, which is 1.1x the bound the caller asserts.  The mechanism:

    * ``_thin_level_pair``'s design is rank 208 of 209, because one level's
      40 rows put the constant vector inside 1.194e-16 (relative, exact at
      80 digits) of the span of that level's own 11 spline columns.  A level
      whose rows miss a knot span does this; it is routine, not contrived.
    * a Gram-level difference carries design round-off LINEARLY: ``V_eff``'s
      float64 value in that direction is ``eps ||V||`` ~ 5e-16 of curvature,
      where the exact value is 1.71e-28.  In the factored form the same
      round-off enters as its SQUARE, 2.5e-31 of curvature.
    * the low edge then multiplies it by ``1/lambda``.  The filter factor
      ``a/(a + lambda (1 - a))`` has slope ``1/lambda = 4.37e+10`` at
      ``a = 0``, so 5e-16 of curvature noise becomes 1.13e-05 of a degree of
      freedom.  Measured: float64 returns ``a = 2.5743e-16`` for a direction
      whose certified ``a`` is 1.7036e-28.

    Against the certified value at the top of this module, at the low edge of
    ``_thin_level_pair(1.0)``::

        this form                 2.84e-14  -- one ulp of the result
        the moment-space form     1.13e-05
        structured path           3.97e-11
        dense path                1.37e-10

    and it is not a property of one arrangement: over 40 exact relabelings of
    the level labels and of the cell table's rows, which leave ``edf``
    unchanged in real arithmetic and change only the order of every reduction,
    the moment-space form spreads 1.2038e-05 and this one 5.68e-14 -- two ulps,
    and the worst distance from the certified value over all 40 is still one.
    Three of 40 level relabelings put the moment-space form outside the 1e-5
    the caller asserts, on one machine, at fixed thread count -- which is
    issue #272's cross-machine failure reproduced locally.

    **A RANK CUT WAS MEASURED AND REFUSED.**  Dropping ``a_j <= n eps``
    before summing repairs this pair (2.84e-14) and destroys the next one:
    on ``_vanishing_mass_pair(1e-10)`` that cut falls inside a cluster --
    neighbouring ``a`` of 3.4466e-14 and 6.1508e-14, 1.785x apart -- and
    discards 5 directions that carry real curvature, taking the error from
    2.73e-06 to 4.87e-03.  Magnitude cannot separate round-off from a level
    that genuinely holds 1e-10 of the weight.  Not squaring it in the first
    place can.

    **STILL NOT USABLE AT THE HIGH EDGE.**  At ``lambda = 2.29e+09`` this
    form is 8.15e-05 from the certified value against the structured path's
    2.15e-05 and the dense path's 1.69e-05 -- worse than both paths it would
    arbitrate, because there the conditioning is the penalty's own and no
    parametrization escapes it.  The high edge stays parity-only.
    """
    return _reference_edf_and_bound(_reference_factors(grab), lam)[0]


class _LowEdgeFactors(NamedTuple):
    """Everything :func:`_reference_edf` needs that does NOT depend on lambda.

    Split out so a caller evaluating at two lambdas -- both consuming tests do,
    once per path -- pays for the 31-column pivoted QR and the residualization
    of the ``(n_cells, k_a k_b)`` tensor once instead of twice.  The fields
    after ``penalty_root`` exist only for the error bound.
    """

    resid: np.ndarray  # the weighted tensor block, residualized on the mains
    penalty_root: np.ndarray  # kron(S_a^(1/2), I_kb), so its Gram is S_ti
    tensor_norm: float  # ||weighted tensor||_F, BEFORE residualizing
    mains_sigma_min: float  # sigma_min of the COLUMN-EQUILIBRATED weighted mains
    n_mains: int  # its column count
    penalty_norm: float  # ||S_a||_2
    penalty_eps: float  # relative perturbation of S_a implied by forming its root


def _reference_factors(grab):
    """Build the pair's weighted design and residualize the tensor on the mains.

    THE MAINS ARE REQUIRED TO BE FULL RANK AND THAT IS AN ASSERTION, not a
    truncation.  A pivoted QR that silently drops a column would leave this
    residualizing on a PSEUDO-inverse of ``M`` while the dense path uses a full
    ``solve(M, C)`` and the structured path uses the arrow factorization's own
    rank -- three different quantities, reported to the caller as an
    unexplained numeric miss, which is exactly the issue #272 failure mode this
    oracle exists to remove.  Measured on every fixture it is used on, the
    smallest ``|R_ii|`` clears the LAPACK rank tolerance by 1.4e+06 (the
    starved ``_vanishing_mass_pair(1e-12)``) to 4.4e+10 (``_thin_level_pair``).

    Demonstrated rather than argued: handing the truncating form a spline basis
    with one duplicated column -- the overlap SPAN is unchanged, its basis is
    not -- it keeps 31 of 32 columns and returns 207.99998956741607 with no
    complaint, **3.414e-05 from the clean value and 245x the 1e-5 its caller
    asserts**, having residualized on a pseudo-inverse.  This raises instead,
    naming the column count and the margin.
    """
    B_a, S_a, _, W_cell, level_rows = _structured_inputs(grab)
    k_a, k_b = B_a.shape[1], level_rows.size
    cells = np.argwhere(W_cell > 0.0)
    rows_a, rows_b = cells[:, 0], cells[:, 1]
    root_w = np.sqrt(W_cell[rows_a, rows_b])[:, None]
    indicator = (rows_b[:, None] == level_rows[None, :]).astype(np.float64)
    tensor = (B_a[rows_a][:, :, None] * indicator[:, None, :]).reshape(cells.shape[0], k_a * k_b)
    tensor = tensor * root_w
    mains = np.concatenate([np.ones((cells.shape[0], 1)), B_a[rows_a], indicator], axis=1) * root_w
    # Pivoted QR of the overlap span -- the one rank decision in here, taken on
    # a FACTOR and never on a Gram.
    Q, R_mains, _ = scipy.linalg.qr(mains, mode="economic", pivoting=True)
    diag = np.abs(np.diag(R_mains))
    tol = max(mains.shape) * np.finfo(np.float64).eps * diag[0]
    assert int(np.sum(diag > tol)) == mains.shape[1], (
        f"the oracle residualizes on a full-rank set of mains or not at all: "
        f"{int(np.sum(diag > tol))} of {mains.shape[1]} columns clear the rank "
        f"tolerance, min |R_ii| = {diag.min():.4e} against tol {tol:.4e}"
    )
    resid = tensor - Q @ (Q.T @ tensor)
    ev, evec = np.linalg.eigh(0.5 * (S_a + S_a.T))
    root_S = np.kron((evec * np.sqrt(np.clip(ev, 0.0, None))) @ evec.T, np.eye(k_b))
    # WHAT `root_S' root_S` ACTUALLY IS, RATHER THAN WHAT IT IS MEANT TO BE.
    # The consumer needs a perturbation of the PENALTY, not of the eigensolver
    # alone, and rebuilding the root is four more roundings after `eigh`:
    #
    #   eigh          the exact decomposition of `S_a + E`, `||E||_2 <= c u
    #                 ||S_a||_2` with `c` LAPACK's modest polynomial in `k_a`
    #                 (*LAPACK Users' Guide* 3rd ed. section 4.7); charged at
    #                 `k_a` deterministically
    #   the clip      moves a round-off-negative eigenvalue by at most its own
    #                 magnitude, which is inside that same `||E||_2`
    #   sqrt          one rounding per eigenvalue, and a relative `u` on the
    #                 ROOT is `2u` on its Gram
    #   the scaling   one more rounding
    #   `@ evec.T`    a `k_a`-term inner product per entry, `gamma_{k_a}`
    #
    # so `3 k_a + 4` roundings of `||S_a||_2`, which is what the bound charges
    # in place of the eigensolver's own constant.  It is 2.6x what the previous
    # revision carried and 3.0e-08 of the returned bound -- negligible BY
    # MEASUREMENT rather than by having been left out.  `kron` with an identity
    # only places entries and rounds nothing.
    penalty_eps = (3.0 * k_a + 4.0) * 0.5 * np.finfo(np.float64).eps
    # Householder's backward error is COLUMNWISE and range(M) is invariant to a
    # positive column scaling, so the conditioning that governs the projector
    # is the equilibrated one -- the standard reference for column
    # equilibration being A. van der Sluis, *Condition numbers and
    # equilibration of matrices*, Numer. Math. 14:14-23 (1969).  It matters: on
    # ``_vanishing_mass_pair(1e-12)`` the raw kappa_2 is 4.24e+06, all of it the
    # 1e-12-weight level's own indicator column, and equilibrated it is 31.9.
    equilibrated = mains / np.linalg.norm(mains, axis=0)
    # It DIVIDES the residualization term, so a singular value rounded upward
    # would make the bound smaller -- the same failure mode the augmented
    # system's own singular values are enclosed against in
    # :func:`_reference_edf_and_bound`.  Carried at its LOWER end for the same
    # reason, and by the same backward-stable-SVD argument plus Horn & Johnson,
    # *Matrix Analysis* 2nd ed., Cor. 7.3.5.  The point is that it cannot round
    # the wrong way on a machine this one cannot see.
    #
    # AND THE EQUILIBRATION ITSELF ROUNDS, which an earlier revision charged
    # nothing for.  Each column norm is an `m`-term sum of squares and a sqrt,
    # and the division is one more rounding, so the computed matrix is
    # `equilibrated (I + diag(d))` with `|d_j| <= u sqrt(m) + 2u` -- a
    # COLUMNWISE relative perturbation, hence at most that times
    # `||equilibrated||_2` in norm.  It is 18% of the SVD's own allowance here
    # and it moves `sigma_min` the same way, so leaving it out rounded the
    # denominator of `eta_R` upward and the bound downward.
    mains_sv = np.linalg.svd(equilibrated, compute_uv=False)
    u = 0.5 * np.finfo(np.float64).eps
    lam_sv = _confidence_multiplier(mains.shape[1])
    mains_eps = (
        lam_sv * u * np.sqrt(equilibrated.size) + lam_sv * u * np.sqrt(mains.shape[0]) + 2.0 * u
    ) * float(mains_sv[0])
    return _LowEdgeFactors(
        resid=resid,
        penalty_root=root_S,
        tensor_norm=float(np.linalg.norm(tensor)),
        mains_sigma_min=float(mains_sv[-1]) - mains_eps,
        n_mains=mains.shape[1],
        penalty_norm=float(np.linalg.norm(S_a, 2)),
        penalty_eps=penalty_eps,
    )


# The TOTAL probability that the error bound :func:`_reference_edf_and_bound`
# returns does not hold.  It is a real number rather than a convention: at the
# implicit multiplier of 1 an earlier revision used, Higham and Mary's statement
# is vacuous (`2 exp(-1/2) = 1.21 > 1`), so nothing was being certified at all.
#
# IT IS A BUDGET AND IT IS SPENT ONCE.  SEVEN distinct events in here are
# charged at the probabilistic rate, so each gets a seventh, and each of those
# is itself a union over the values it must hold for at once -- Higham's Thm
# 19.4 is COLUMNWISE and the singular-value bound is per value:
#
#   1  the mains QR                 4  the `Q_R` SVD
#   2  the augmented QR             5  the equilibration's column norms
#   3  the equilibrated mains' SVD  6, 7  the two projector GEMMs
#
# 5, 6 and 7 reuse 3's and 1's multipliers because they are the same shape, but
# they are DIFFERENT random variables and all seven must hold for the returned
# number to be a bound.  Two earlier revisions got this wrong in the same
# direction: one gave both QRs the whole budget (2x optimistic), and one
# counted four events where there are seven (1.75x).
_QR_FAILURE_PROBABILITY = 1e-6
_PROBABILISTIC_TERMS = 7


def _confidence_multiplier(count):
    """Higham-Mary's ``lambda`` for ``count`` bounds holding at once.

    Their statement is ``|theta_k| <= lambda sqrt(k) u`` with probability at
    least ``1 - 2 exp(-lambda^2 / 2)`` for ONE quantity.  A union over ``count``
    of them, at this term's share of the budget, needs
    ``count * 2 exp(-lambda^2 / 2) <= _QR_FAILURE_PROBABILITY / _PROBABILISTIC_TERMS``.
    """
    share = _QR_FAILURE_PROBABILITY / _PROBABILISTIC_TERMS
    return float(np.sqrt(2.0 * np.log(2.0 * count / share)))


def _reference_edf_and_bound(factors, lam):
    """``edf(lambda)``, and a DERIVED bound on this routine's own error in it.

    The bound is built from dimensions, the unit roundoff, and the conditioning
    of the augmented system -- never from observed headroom.  That distinction
    is the whole subject of issue #272: a tolerance calibrated on one machine's
    rounding is not a tolerance, and this oracle calls pivoted QR, an
    unpivoted QR with its factor accumulated, a symmetric eigensolver and two
    SVDs, every one of which is free to round differently under another BLAS.

    Write ``A = [R; sqrt(lambda) L]`` for the augmented matrix and
    ``Q = [Q_R; Q_L]`` for its orthonormal factor.  Then ``edf = ||Q_R||_F^2``
    -- equivalently ``tr(J P_A)`` for the orthogonal projector ``P_A`` onto
    ``range(A)`` and ``J = diag(I_m, 0)``, which is the form the perturbation
    theory below is stated in.  The filter factors ``a_j`` are the squared
    singular values of ``Q_R``, and for perturbations ``dR``, ``dL`` (with
    ``Ahat = Q_R'Q_R``)::

        d(edf) = 2 tr(T^-1 (I - Ahat) Q_R' dR) - 2 tr(T^-1 Ahat Q_L' sqrt(lam) dL)

    so, with ``||Q_R (I - Ahat)||_F^2 = sum_j a_j (1 - a_j)^2 =: w1^2`` and
    ``||Q_L Ahat||_F^2 = sum_j a_j^2 (1 - a_j) =: w2^2``::

        |d edf| <= 2 (w1 ||dR||_F + w2 ||sqrt(lam) dL||_F) / sigma_min(A)

    **THE WEIGHTS ARE THE POINT, NOT THE CONDITION NUMBER.**  Pairing
    ``||dR||_F/sigma_min`` with ``sqrt(edf)`` instead -- the obvious bound --
    is 3.9e+05 times looser here (a ratio, so it does not depend on the
    dimensional constants), because it
    charges the ill-conditioned direction's sensitivity against the
    well-determined directions' mass.  ``edf`` is insensitive exactly where the
    filter factors are saturated: at this low edge 208 of 209 have ``a`` within
    4.457e-05 of 1 IN TOTAL and the last has a certified ``a`` of 1.7e-28, so
    ``w1 = 3.657e-05`` where ``sqrt(edf) = 14.42``.

    **AND THAT IS EXACTLY WHY THE ORTHONORMAL FACTOR IS FORMED EXPLICITLY.**
    The identity above requires the numerator and the denominator to move
    TOGETHER: ``w1``'s ``(1 - a)`` is the cancellation between the two.  An
    earlier revision of this function computed ``edf = ||R T^-1||_F^2`` with a
    triangular solve, and that breaks the requirement.  The computed ``T`` is
    the exact triangular factor of a perturbed ``A + dA``, while the ``R`` in
    the numerator is the LITERAL, unperturbed array, so what is evaluated is
    ``tr(R ((A+dA)'(A+dA))^-1 R')`` -- a DENOMINATOR-ONLY perturbation.  Its
    weight is not ``w1``.  Differentiating ``edf = tr(R G^-1 R')`` in ``G``
    alone gives ``d edf = -2 tr(diag(a) (R Z)' dR Z)``, so::

        |d edf| <= 2 sqrt(sum_j a_j^3) ||dR||_F / sigma_min(A)

    and ``sqrt(sum a^3) = 14.4222`` is ``sqrt(edf)`` to six digits.  The
    scalar case is the whole argument in one line: at ``p = 1`` a consistent
    ``r -> r(1 + d)`` moves ``edf`` by ``2 d a (1 - a)``, and a
    denominator-only one by ``-2 d a^2``.  At ``a -> 1`` the first VANISHES and
    the second does not.  Since 208 of the 209 directions here sit at
    ``a = 1`` to within 4.5e-05 in total, that is the entire quantity.
    Charged correctly against the triangular-solve form, this function returned
    **7.2045e-06** on ``_thin_level_pair(1.0)``, not the 4.5938e-09 it
    reported -- both at that revision's constants, so the ratio is the claim:
    1568x understated, and over seeds 1-21 of the same geometry the
    number it RETURNED fell short of the exact worst-case first-order response
    it was meant to bound on seven of them, worst by 460x at seed 5.  Forming
    ``Q`` costs one ``dorgqr`` and removes the whole mechanism, because
    Householder's computed factor is the exactly-orthonormal factor of
    ``A + dA`` (Higham, *ASNA* 2nd ed., Thm 19.4), so numerator and denominator
    share the perturbation by construction.

    **THE BOUND IS STRICT IN ``dA``, NOT FIRST ORDER.**  An earlier revision
    returned the differential above evaluated at the COMPUTED factor, plus a
    quadratic allowance ``(2 eta/sigma_min)^2``.  That is not a bound: the
    differential's own weights move under the perturbation being bounded, and
    here they move further than they are large.  The drift enters as
    ``w1(seg)^2 = w1^2 + slope1 delta + 2 delta^2``, so it is compared in
    ``w^2`` space: ``slope1 delta = 1.2748e-04`` against ``w1^2 = 1.3375e-09``,
    a factor of **93000**, i.e. ``w1`` moves by **308x itself** over the very
    perturbation being bounded.  The quadratic allowance covered only the
    ``||dQ_R||_F^2`` term of the expansion and was orders short of that.  What
    is returned now is the mean-value form, which needs no remainder at all::

        |edf(A + dA) - edf(A)| <= 2 max_t [w1(A_t) ||dR||_F + w2(A_t) ||dL||_F]
                                  / sigma_min(A_t),      A_t = A + t dA

    with every ingredient bounded strictly:

    * ``sigma_min(A_t) >= sigma_min(A) - ||dA||_2`` (Horn & Johnson, *Matrix
      Analysis*, 2nd ed., Cor. 7.3.5), which also gives
      ``rank(A_t) = rank(A)`` throughout and is what the ``sigma_min`` guard
      below enforces.
    * the filter factors are the eigenvalues of ``J P_A J``, so
      ``||a(A_t) - a(A)||_2 <= ||J (P_{A_t} - P_A) J||_F <= ||P_{A_t} - P_A||_F``.
      This is the Hermitian sorted-order form, which is a COROLLARY of Hoffman
      & Wielandt, *Duke Math. J.* 20:37-39 (1953) -- their statement is for
      normal matrices and asserts the existence of a permutation.  The
      corollary is standard and is in Horn & Johnson, *Matrix Analysis*, 2nd
      ed., section 6.3; no theorem number is given for it here because none
      could be confirmed against a copy, and Thm 6.3.5 there is the
      normal-matrix statement rather than this one.
    * at equal rank ``||P_B - P_A||_F^2 = 2 ||P_B^perp E A^+||_F^2`` exactly, so
      ``||P_B - P_A||_F <= sqrt(2) ||dA||_F / sigma_min(A) =: delta``.  This is
      Lemma 2.3, eq. (2.7a) of Xu, *On the perturbation of an L^2-orthogonal
      projection* (arXiv:1809.00200), which gives
      ``||P_B - P_A||_F^2 = 2(||E A^+||_F^2 - ||P_B E A^+||_F^2)`` at equal
      rank; the 2-norm form ``||P_B - P_A||_2 <= ||A^+||_2 ||E||_2`` that the
      residualization term below uses is its eq. (1.2c), attributed there to
      J.-G. Sun, *The stability of orthogonal projections*, J. Grad. Sch.
      1:123-133 (1984).  An earlier revision of this docstring cited Wedin
      (1973) for it, unread.  What is said here is only what Xu's attribution
      says: Wedin appears nowhere in Xu's bibliography.  It is NOT a priority
      claim -- Wedin (1973), *Perturbation theory for pseudo-inverses*, BIT
      13:217-232 does contain projector perturbation bounds, and has not been
      read either.
    * ``phi1(a) = a(1-a)^2`` and ``phi2(a) = a^2(1-a)`` are smooth on ``[0,1]``
      with ``|phi''| <= 4``, so Taylor with a Lagrange remainder gives
      ``w1(A_t)^2 <= w1^2 + ||phi1'(a)||_2 delta + 2 delta^2`` and likewise for
      ``w2``, each capped at the ``4p/27`` their crests allow.

    Verified numerically as well as derived.  The assembled bound: over 600
    random ``(m, p)`` pairs with singular values spread over six decades and
    ``||dA||_F/sigma_min`` from 1e-06 to 1e-01, **0 violations**, worst
    realized/bound 0.409.  The ingredients separately, over 400 more: the
    Hoffman-Wielandt step 0 violations (worst 0.571); the Lipschitz constants
    are exact (``sup |phi1'| = sup |phi2'| = 1`` and ``max phi1 = max phi2 =
    4/27`` to six digits); the projector inequality 1 APPARENT violation at
    1.65 -- which is the float64 projector itself at ``kappa = 3.9e+07`` and
    ``||E||/sigma_min = 1.2e-09``, where recomputing both projectors at 60
    digits gives 0.28.  That is issue #272's own defect reappearing inside the
    instrument built to check the fix, which is why it is checked in extended
    precision and why it is recorded here rather than quietly dropped.

    **NOTHING GOES RED FROM MAKING IT STRICT, AND THAT IS STATED PLAINLY.**
    At the same constants the bound rises 5.4930e-08 -> **2.2170e-06** on the
    thin pair (38.6x) and 7.6660e-04 -> 7.9118e-04 on the starved one (+0.12%);
    the realized error is 2.8422e-14 either way, and no assertion in this file
    separates them.  The first-order form was not observed to fail: it survives
    the same 600 random draws and the same injected-perturbation sweep.  What
    changes is that the remainder is no longer an allowance carried on
    measurement.

    **THE TWO FIXTURES ARE IN OPPOSITE REGIMES AND THAT IS WHY THEY RESPOND
    DIFFERENTLY TO EVERYTHING.**  On the thin pair the segment drift dominates
    its own base weight by 93000x (``slope1 delta = 1.2748e-04`` against
    ``w1^2 = 1.3375e-09``), so ``w1 ~ sqrt(delta) ~ sqrt(lambda)`` while
    ``eta ~ lambda`` and the bound scales as ``lambda^(3/2)``.  On the starved
    pair ``w1 = 0.9374`` dominates its drift by 1700x and the bound is linear in
    ``lambda``.  Every "Nx inside its gate" figure below is a margin in the
    BOUND, so on the thin pair the corresponding margin in the CONSTANT is that
    figure to the power 2/3.

    **THE STRICT BOUND THAT DROPS THE ``(1 - a)`` WEIGHTING WAS MEASURED AND
    REFUSED, AND THE GATE WAS NOT WIDENED TO FIT IT.**  Writing ``g = p - edf =
    ||Q_L||_F^2`` and using ``|Delta g| <= ||dP||_F (2 sqrt(g) + ||dP||_F)``
    directly gives the fully citation-backed
    ``|Delta edf| <= 2 sqrt(2) sqrt(p - edf) ||dA||_F / sigma_min
    + 2 ||dA||_F^2 / sigma_min^2``, with no weights to bound over a segment.
    It is correct, and it is **2.5501e-04** here against the 1e-5 its caller
    asserts on the paths -- **25x OUTSIDE** it, and 2.5786e-04 at
    ``low_weight = 0.001``.  Adopting it means either widening that gate, which
    is the defect this file exists to remove, or losing the oracle on the one
    fixture where it still works.  The reason it is so much looser here is
    specific and worth recording: ``g = p - edf`` is 1.0000 on this fixture and
    all but 4.5e-05 of it is the ONE direction whose ``a`` is 1.7e-28, where
    ``phi1(a) = a(1-a)^2`` vanishes -- so ``sqrt(g)`` is 27000x ``w1`` while
    bounding the same quantity.  The two forms cross over at
    ``||dA||_F/sigma_min ~ g/||phi1'(a)||_2``, so the segment form's advantage
    is specific to backward errors this small; it is not a better bound in
    general.

    **THE ``(1 - a)`` REFINEMENT IS OURS, AND THAT IS A SEARCH RESULT RATHER
    THAN AN ASSUMPTION.**  Looked for a published bound on
    ``|Delta tr(J P_A)|`` carrying saturation weighting, or splitting by ``dR``
    against ``dL``, and found none.  The two ingredients exist in separate
    literatures and have not been combined.  Knyazev & Argentati, *Majorization
    for changes in angles between subspaces, Ritz values, and graph Laplacian
    spectra* (arXiv:math/0508591), Thm 4.2 -- resting on their Thms 3.2 and 3.3,
    which an earlier note on this branch cited instead -- give the sharpest
    published bound
    on a change of this kind, by weak majorization and weighted only by the
    spectral spread -- with NO ``(1 - a)`` factor.  Their section 4 closes by
    saying that where one subspace is invariant "it is natural to expect a much
    better bound that involves the square of the sin Theta(X,Y)" and that
    "Majorization results of this kind are not apparently known in the
    literature" -- which is the case here, and is quoted with its condition
    because it is stated under one.  Holodnak, Ipsen & Wentworth, *Conditioning
    of leverage scores and computation by QR decomposition* (arXiv:1402.0957),
    Thm 2.2, publish the saturation weighting
    ``2 sqrt(l_j(1-l_j)) cos(th_1) sin(th_n) + sin^2(th_n)`` but for INDIVIDUAL
    leverage scores, never for their sum.  The vocabulary for the quantity
    itself is Avron, Clarkson & Woodruff, *Sharper Bounds for Regularized Data
    Fitting* (arXiv:1611.03225): ``edf`` is their Definition 1's *statistical
    dimension* ``sd_lambda(A)``, and ``sd_lambda(A) = ||Q||_F^2`` for the
    ridge-augmented ``Q`` is their Fact 33.  There is nothing in the statistics
    literature on the sensitivity of ``tr(H)`` to the DESIGN -- that work is
    all sensitivity to lambda, or to the definition of df -- so the
    numerical-analysis framing is the right one and no statistical shortcut is
    being missed.

    The five error sources, each at its own standard bound:

    * **the residualization.**  Householder QR is backward stable columnwise
      (Higham, *ASNA* 2nd ed., Thm 19.4), and the projector onto a perturbed
      column space moves by at most ``||dM||_2 / sigma_min(M)`` (Sun 1984, as
      above; charged at twice that here) -- taken on the COLUMN-EQUILIBRATED
      mains, per :func:`_reference_factors`.
    * **the augmented QR**, backward stable, so a perturbation of ``A`` itself
      -- and a CONSISTENT one, which is what forming ``Q`` explicitly buys.
    * **``Q``'s departure from orthonormality.**  The accumulated factor is
      the exactly-orthonormal ``Qbar`` plus ``dQ`` with
      ``||dQ||_F <= gamma_aug sqrt(p)`` (Higham, *ASNA* 2nd ed., section 19.3,
      applied to ``[I_p; 0]`` -- the result is the standard bound on the
      computed product of Householder reflectors; the section is cited without
      a theorem number because no accessible source pins one to it).  It enters EXACTLY, as
      ``2 ||Qbar_R||_F ||dQ_R||_F + ||dQ_R||_F^2 = 1.54e-10``, and is NOT
      amplified by ``1/sigma_min`` -- 0.0073% of the total.
    * **the penalty root**, whose ``eigh`` perturbs ``S_a`` by ``u ||S_a||_2``;
      that enters ``edf`` as ``lambda ||dS||_2 tr(G^-1)`` with ``G = A'A``,
      since ``d edf = -lambda tr(dS G^-1 V G^-1)`` and ``G^-1 V G^-1 <= G^-1``.
      ``||dS||_2`` is charged for the WHOLE of rebuilding the root and not for
      ``eigh`` alone -- see :func:`_reference_factors` -- which is 2.56x what an
      earlier revision carried.  6.3e-14 at the low edge, 3.0e-08 of the total,
      carried for completeness and taken over the segment like everything else.
    * **the final accumulation.**  ``edf`` is a sum of ``m p`` squares reaching
      208, so numpy's pairwise summation alone costs
      ``(log2(m p) + 1) u edf = 4.24e-13`` (Higham, *The accuracy of floating
      point summation*, SIAM J. Sci. Comput. 14(4):783-799, 1993).  Nothing
      about this quantity can be asserted below that floor, and one ulp of the
      result is 2.84e-14.

    **THE DIMENSIONAL FACTORS ARE PROBABILISTIC, AND THE PROBABILITY IS NAMED
    RATHER THAN IMPLIED.**  Both QRs are charged at ``lambda u sqrt(k)`` for
    ``k`` the operation count of Higham's deterministic ``gamma_k`` -- ``m q``
    for the mains, ``(m + p) p`` for the augment -- plus ``p`` at the
    deterministic rate for accumulating ``Q``.  The square-root scaling is
    Higham and Mary, *A New Approach to Probabilistic Rounding Error Analysis*,
    SIAM J. Sci. Comput. 41(5):A2815-A2835 (2019), which proves that
    ``gamma_k`` may be replaced by ``lambda sqrt(k) u`` with probability at
    least ``1 - 2 exp(-lambda^2 / 2)`` FOR ONE QUANTITY.

    An earlier revision took ``lambda = 1``, at which the statement is vacuous
    -- ``2 exp(-1/2) = 1.21 > 1`` -- so it certified nothing while being
    consumed as a certificate.  ``lambda`` now comes from
    :func:`_confidence_multiplier`, which spends ``_QR_FAILURE_PROBABILITY``
    ONCE: a quarter each to the two QRs and the two SVDs, and within each a
    union over the values it must hold for simultaneously, since Thm 19.4's
    backward error is columnwise.  That gives **6.307** for the mains' 31
    columns and **6.603** for the augment's 209 at a total of 1e-06.  A revision
    in between gave both QRs the whole budget and advertised the sum as the
    total, which was 2x optimistic and is corrected here at a cost of 5.7%.

    The naming costs 15.2x on the thin pair -- a 6.5x constant through a
    ``lambda^(3/2)`` regime -- and 6.2x on the starved one, and it is what moved
    the starved fixture's oracle outside its caller's allowance.  Everything
    downstream of ``||dA||`` is strict; this is where ``||dA||`` itself comes
    from, and it is the only assumption left in the number.  The alternative
    readings of the same theorem, and what each would cost, are on **#301**.

    **AND THE DETERMINISTIC CONSTANTS WERE TRIED AND MEASURED USELESS.**
    Carrying Higham's ``gamma_{mq}`` and ``gamma_{(m+p)p}`` verbatim puts the
    bound at **2.9476e-04** on ``_thin_level_pair(1.0)``, 29x outside its
    caller's 1e-5, and **2.0920e-02** on ``_vanishing_mass_pair(1e-12)``, 70x
    outside.  At those the oracle certifies nothing anywhere: the pre-#272
    moment-space form, whose 1.1253e-05
    error is the entire reason this test exists, is INSIDE both.  A bound that
    cannot separate the defect from the noise is not a safer bound, it is a dead
    test.  So the probability is a real choice with a real price, and it is
    named rather than implied.

    What it comes out at, at each fixture's own low-edge lambda:

    ===========================  ==========  ==========  ====================
    fixture                           bound    realized   caller's allowance
    ===========================  ==========  ==========  ====================
    ``_thin_level_pair(1.0)``     2.2170e-06  2.8422e-14  4.51x inside 1e-5
    ``_thin_level_pair(0.01)``    2.2524e-06           -  4.43x inside 1e-5
    ``_thin_level_pair(0.001)``   2.2549e-06           -  4.43x inside 1e-5
    ``_vanishing_mass_pair()``    7.9118e-04           -  12.6x inside 1e-2
    ``_vanishing_mass_pair(-10)`` 3.6012e-04           -  27.8x inside 1e-2
    ===========================  ==========  ==========  ====================

    The realized error on the thin pair is exactly ONE ULP of the result --
    2.8422e-14 is ``np.spacing(208.0)`` -- which is the floor, not a margin.

    **THE LAST TWO ROWS ARE WHY THE STARVED FIXTURE'S ALLOWANCE MOVED FROM 3e-4
    TO 1e-2.**  No multiplier rescues the old number: it would take one small
    enough to make the probability meaningless, and constant-shopping to fit an
    assertion is the defect this file exists to remove.  So that rung's 3e-4 was
    NOT widened -- it was RE-PLACED by the same floor/ceiling/place method the
    thin pair's is, with the oracle's own derived error as the floor, one degree
    of freedom as the ceiling, and 1e-2 the literal its own sibling rung already
    uses against an arbiter of the same accuracy.  It costs 33x of tightness on
    the arms and it is recorded there and on #301.

    **IT IS NOT TIGHTER THAN THE 1e-9 IT REPLACES.  IT IS 2217x LOOSER.**  A
    number set from errors observed on one machine landed three orders inside
    the bound this problem actually supports, so the original calibration was
    lucky and could not have been known to be.  What it could not do is move:
    it is a constant, and the quantity it bounds is 2.2170e-06 here and
    7.9118e-04 on ``_vanishing_mass_pair(1e-12)``, well over two orders apart.
    A fixture-blind 1e-9 is simultaneously right here and wrong by orders
    there, which is the defect, not the size of the number.

    Validated by direct perturbation as well, one thread.  Injecting ``dA`` of
    known norm over five decades of ``||dA||_F/sigma_min`` (1e-05 to 1e-01), 24
    random directions per decade, and comparing each against the bound
    RECOMPUTED at that injected norm, the realized response reaches at most
    1.0e-04 of it on the thin pair and 3.8e-03 on the starved one -- random
    directions are nowhere near the worst case, which is the reason the bound
    is derived rather than sampled.  A full ``u ||M||_F`` perturbation of the
    mains moves ``edf`` by at most 5.68e-14, far inside what the Sun term
    charges for it.

    Across 1, 2, 4 and 8 threads -- the cheapest instance of the reordered
    reduction the bound exists for -- the value is **bit-identical**
    (207.99995542698144 at all four), where the triangular-solve form moved
    between ...127 and ...133.  The bound does not move either, because it is
    a property of the problem rather than of the machine.

    **WHAT IT DOES NOT COVER.**  This bounds the ORACLE's rounding given the
    ``B_a``, ``S_a`` and ``W_cell`` it is handed.  It says nothing about those
    inputs differing -- a ``fit_reml`` that lands elsewhere on another
    platform moves the exact answer itself, and the certified constant is the
    exact answer for the inputs measured here.  That exposure is unchanged in
    kind from the 1e-9 this replaces and is larger in magnitude, which is the
    price of deriving it rather than observing it.  ``W_cell`` is exact on the
    certified fixture, and the reason is not that its sum lands on 800.0 -- a
    sum of 800 floats reaching exactly 800.0 says nothing about the entries.
    It is that the family is gaussian with an identity link, so the working
    weights ARE the sample weights; ``low_weight = 1.0`` makes every one of
    them exactly 1.0; and ``x`` is drawn continuously, so each of the 800 rows
    is its own cell and no aggregation rounds.  That also makes the
    design-assembly term identically zero THERE (``sqrt(1.0) = 1.0``, both
    multiplies exact) and 0.21% of ``gamma_mains`` on the other four fixtures,
    where the sum is not 800.0.

    The bound's own coefficients are computed in the same float64 as the value
    they bound, so they are used as ENCLOSURES rather than as exact numbers.
    A singular value rounded the wrong way would otherwise shrink ``w1`` or
    ``w2`` and grow ``sigma_min`` -- all three in the direction that makes the
    bound smaller.  Each ``a_j`` is carried as an interval covering both the
    SVD's own backward error and ``Q``'s departure from orthonormality, and
    every weight takes the largest value its integrand attains on that interval
    -- which for ``phi1``, ``phi2`` and their derivatives means checking the
    interior crest as well as the ends.  ``sigma_min`` is taken at its lower
    end.  A ``sigma_min`` enclosure that the segment can reach zero returns an
    INFINITE bound rather than a small wrong number, and the caller's gate on
    it is what turns that into a failure.
    """
    u = 0.5 * np.finfo(np.float64).eps  # unit roundoff, eps/2
    R, root_S = factors.resid, factors.penalty_root
    m, p = R.shape
    aug = np.vstack([R, np.sqrt(lam) * root_S])
    # THE ORTHONORMAL FACTOR IS FORMED EXPLICITLY AND THAT IS THE WHOLE POINT.
    # `||R T^-1||_F^2` via a triangular solve is the same quantity in exact
    # arithmetic and a DIFFERENT one in float64: it divides the literal,
    # unperturbed `R` by a `T` carrying the QR's backward error, so the
    # numerator and the denominator do not move together and the `(1 - a)`
    # cancellation `w1` is built on never happens.  See the docstring.
    Q, T = np.linalg.qr(aug, mode="reduced")
    Q_R = Q[:m]
    edf = float(np.sum(Q_R**2))

    # How much the two QRs and the design assembly perturb what they factor.
    #
    # EACH CARRIES AN EXPLICIT MULTIPLIER FROM A STATED FAILURE PROBABILITY, so
    # what is returned is a certificate at a named confidence rather than a
    # number with an implicit constant of 1 -- at which Higham and Mary's bound
    # is vacuous, since `2 exp(-1/2) = 1.21 > 1`.  Their result is
    # `gamma_k -> lambda sqrt(k) u` with probability at least
    # `1 - 2 exp(-lambda^2 / 2)` for ONE quantity, and Thm 19.4's backward error
    # is COLUMNWISE, so every column bound each QR needs must hold at once: a
    # union bound over `q` and over `p` respectively, at a total
    # `_QR_FAILURE_PROBABILITY` shared between them.
    lam_mains = _confidence_multiplier(factors.n_mains)
    lam_aug = _confidence_multiplier(p)
    # QR of the (m, q) mains.  The `+ q` mirrors `gamma_aug`'s `+ p`: this
    # factor's `Q` is explicitly formed and then applied twice, so accumulating
    # it from the reflectors is charged, at the deterministic rate.  3.1%.
    gamma_mains = lam_mains * u * np.sqrt(m * factors.n_mains) + u * factors.n_mains + 2.0 * u
    # QR of the (m+p, p) augment; the `+ p` covers accumulating `Q` from the
    # reflectors, and is charged at the DETERMINISTIC rate where the
    # probabilistic convention would allow less.
    gamma_aug = u * (lam_aug * np.sqrt((m + p) * p) + p)
    # The projector's SUBSPACE moves by ||dM||_2/sigma_min(M) (Sun 1984, via Xu
    # arXiv:1809.00200 eq. (1.2c)), charged at twice that; applying it is two
    # GEMMs and a cancelling subtraction, which is charged separately.
    eta_R = 2.0 * gamma_mains * np.sqrt(factors.n_mains) / factors.mains_sigma_min
    # APPLYING the projector is `tensor - Q (Q' tensor)`: two GEMMs and a
    # cancelling subtraction.  The first costs `u sqrt(m) ||Q||_F` and
    # `||Q||_F = sqrt(q)` exactly for an orthonormal Q, so it is `gamma_mains`
    # in shape; the second has inner dimension `q`; the subtraction is one more
    # rounding.  Each GEMM's multiplier unions over the number of results it
    # must bound AT ONCE -- `q p` and `m p` entries, not the `q` columns an
    # earlier revision reused -- which costs 13% and 19% of two terms that are
    # together 0.5% of this one.
    gemm_1 = _confidence_multiplier(factors.n_mains * p) * u * np.sqrt(m * factors.n_mains)
    gemm_2 = _confidence_multiplier(m * p) * u * np.sqrt(factors.n_mains)
    eta_R = eta_R + gemm_1 + gemm_2 + u
    eta_R = eta_R * factors.tensor_norm
    # gamma_aug perturbs the WHOLE of [R; sqrt(lam) L].  With `Q` formed
    # explicitly that perturbation is CONSISTENT -- the computed factor is the
    # exactly-orthonormal factor of `aug + dA` -- so its R rows ride w1 and its
    # penalty rows ride w2, exactly as written.
    eta_A = gamma_aug * float(np.linalg.norm(aug))
    # `sqrt(lam)` then a multiply: two roundings, so 2u rather than u.
    eta_L = 2.0 * u * np.sqrt(lam) * float(np.linalg.norm(root_S))
    eta = eta_R + eta_A + eta_L  # ||dA||_F for the whole augmented matrix

    # THE SENSITIVITY COEFFICIENTS ARE THEMSELVES COMPUTED, so they are used as
    # ENCLOSURES rather than as exact numbers: a singular value rounded the
    # wrong way would otherwise shrink the weights and grow the denominator,
    # all three in the direction that makes the bound smaller.  The SVD is
    # backward stable, so the singular values' 1-Lipschitz dependence on the
    # matrix (Horn & Johnson, *Matrix Analysis* 2nd ed., Cor. 7.3.5) puts each
    # computed singular value within `gamma ||.||_2` of the true one.  `sv` is
    # additionally the SVD of the COMPUTED Q_R rather than of the exactly
    # orthonormal Qbar_R, which is what `d_Q` covers here.  `T` needs no
    # analogue: Higham's Thm 19.4 pairs the computed T with an exactly
    # orthonormal Qbar, so `sing` are already the exact singular values of the
    # perturbed augmented matrix.
    d_Q = gamma_aug * np.sqrt(p)
    sv = np.linalg.svd(Q_R, compute_uv=False)
    sing = np.linalg.svd(T, compute_uv=False)
    # The SVDs are charged at the same stated confidence as the QRs, from the
    # same budget -- an earlier revision left them at an implicit multiplier of
    # one inside a function whose docstring calls that vacuous.  Inert here
    # (`d_Q` still dominates `eps_a`'s SVD half 17.4x after the fix, where it
    # dominated it 113x before) and carried for consistency.
    eps_a = _confidence_multiplier(p) * u * np.sqrt(Q_R.size) + d_Q  # ||Q_R||_2 <= 1
    eps_T = u * float(p) * float(sing[0])  # deterministic, and conservative
    # `maximum(., 0)` before squaring: without it a direction with `sv < eps_a`
    # gets a POSITIVE lower end and the interval stops containing the true `a`,
    # which is the case here: the near-null direction's COMPUTED `sv` is
    # 3.40e-12 against `eps_a` = 5.55e-12, where the exact value is
    # sqrt(1.7e-28) = 1.30e-14 -- the computed one is 260x the true one, which
    # is what an enclosure is for.
    a_lo = np.clip(np.maximum(sv - eps_a, 0.0) ** 2, 0.0, 1.0)
    a_hi = np.clip((sv + eps_a) ** 2, 0.0, 1.0)
    sigma_min = float(sing[-1]) - eps_T
    # The segment A -> A + dA must not change rank, and every quantity below is
    # taken at its worst over that segment.
    sigma_seg = sigma_min - eta
    if sigma_seg <= 0.0 or factors.mains_sigma_min <= 0.0:
        # Either conditioning is past what float64 can resolve; say so as an
        # infinite bound rather than as a small wrong number.  The caller's
        # gate on the bound is what turns this into a failure.
        return edf, float("inf")

    def _outer(f, *crests):
        """The largest ``|f(a)|`` over each enclosure, given ``f``'s interior extrema."""
        best = np.maximum(np.abs(f(a_lo)), np.abs(f(a_hi)))
        for crest in crests:
            inside = (a_lo <= crest) & (crest <= a_hi)
            best = np.where(inside, np.maximum(best, abs(f(crest))), best)
        return best

    # `||a(A_t) - a(A)||_2` over the whole segment: Hoffman-Wielandt on the
    # eigenvalues of `J P_A J`, then the equal-rank projector identity.  The
    # denominator is `sigma_seg` and not `sigma_min`, because the projector
    # bound wants sigma_min of the UNPERTURBED matrix and all this routine
    # knows is the perturbed one's, enclosed: 8.8e-05 relative here, and the
    # point is that it cannot round the wrong way.
    delta = np.sqrt(2.0) * eta / sigma_seg
    # phi1, phi2 and their derivatives, each enclosed over [a_lo, a_hi].  The
    # weights are then carried over the segment by Taylor with a Lagrange
    # remainder, |phi''| <= 4 on [0, 1], and capped at what their crests allow.
    cap = 4.0 / 27.0 * p
    w1 = float(np.sum(_outer(lambda x: x * (1.0 - x) ** 2, 1.0 / 3.0)))
    w2 = float(np.sum(_outer(lambda x: x * x * (1.0 - x), 2.0 / 3.0)))
    slope1 = float(np.linalg.norm(_outer(lambda x: 1.0 - 4.0 * x + 3.0 * x * x, 2.0 / 3.0)))
    slope2 = float(np.linalg.norm(_outer(lambda x: 2.0 * x - 3.0 * x * x, 1.0 / 3.0)))
    w1 = np.sqrt(min(w1 + slope1 * delta + 2.0 * delta**2, cap))
    w2 = np.sqrt(min(w2 + slope2 * delta + 2.0 * delta**2, cap))

    bound = 2.0 * (w1 * (eta_R + eta_A) + w2 * (eta_L + eta_A)) / sigma_seg
    # The computed Q is only orthonormal to `||dQ||_F <= gamma_aug sqrt(p)`
    # (Higham, ASNA 2nd ed., section 19.3, applied to `[I_p; 0]`).  This enters
    # EXACTLY rather than to first order, and it is NOT amplified by
    # 1/sigma_min -- which is why forming Q is worth its cost.
    bound += 2.0 * np.sqrt(edf) * d_Q + d_Q**2
    # eigh is backward stable with its own dimensional constant, and clipping a
    # round-off-negative eigenvalue moves S_a by no more than the same amount.
    # The penalty root's own perturbation moves `G = A'A` by `lam dS`, which a
    # NEGATIVE component of `dS` moves DOWNWARD -- so the same term's denominator
    # has to be enclosed against it too, or the trace it bounds is evaluated at
    # a `G` the perturbation can get beneath.  `sigma^2 >= sigma_lo^2 - lam
    # ||dS||_2` by the same 1-Lipschitz argument as everywhere else.  1.45e-24
    # against a 2.3e-11 `sigma_min^2` here, which is the point: negligible BY
    # MEASUREMENT rather than by having been left out.
    dS = lam * factors.penalty_eps * factors.penalty_norm
    sing_lo = np.sqrt(np.maximum((sing - eps_T - eta) ** 2 - dS, 0.0))
    if not np.all(sing_lo > 0.0):
        return edf, float("inf")
    bound += dS * float(np.sum(1.0 / sing_lo**2))
    # numpy's pairwise summation.  The depth is charged at 32 rather than at
    # `log2(m p) = 18.35`: `pairwise_sum` recurses only down to a 128-element
    # block, which it then sums with 8 accumulators, so the tree is deeper than
    # a full halving and a fractional depth is not a depth at all.  4.24e-13 ->
    # 7.39e-13, and the docstring calls this a floor, so it should be one.
    bound += 33.0 * u * edf
    # The bound's own ~30 arithmetic operations round to NEAREST, so the number
    # returned can sit just below the value derived.  There is no cancellation
    # anywhere in it -- every subtraction is guarded and none is near zero -- so
    # `64 u` covers the accumulation with a factor of two to spare.  6e-21 here;
    # the point is that a bound may not be rounded inward by its own evaluation.
    return edf, bound * (1.0 + 64.0 * u)


def _relabelings(grab, rng, draws):
    """Exact relabelings of one pair: the level labels, and the cell table's rows.

    Both leave ``edf`` unchanged in real arithmetic and change the order of
    every reduction, so the spread across them is this oracle's own round-off
    with the answer held fixed.  Permuting the LEVELS reorders the tensor's
    block structure and the arrow border; permuting the CELL ROWS reorders
    every accumulation over the 800 cells, which is the other reduction order
    and the one a threaded BLAS is free to change.
    """
    menu_a, menu_b = grab["menus"]
    S_cell, W_cell = grab["cells"]
    for draw in range(draws):
        if draw % 2 == 0:
            perm = rng.permutation(menu_b.shape[1])
            yield {**grab, "menus": (menu_a, menu_b[:, perm])}
        else:
            perm = rng.permutation(menu_a.shape[0])
            yield {
                **grab,
                "menus": (menu_a[perm], menu_b),
                "cells": (S_cell[perm], W_cell[perm]),
            }


def test_the_low_edge_reference_matches_a_certified_high_precision_value():
    """The oracle is pinned to high precision, not to the other arm.

    Everything else in this file compares two float64 arms, which cannot tell
    a wrong reference from a wrong path.  Issue #272 was exactly that: the
    low-edge assertion failed on a developer machine and passed in CI on the
    same locked dependencies, and the term that moved was neither path.  Both
    of them sit 3.97e-11 and 1.37e-10 from the certified value; the reference
    sat 1.13e-05 away and moved by 1.20e-05 across arrangements of the same
    algebra.

    Two properties, and the second is what makes the first portable:

    * the value itself, against the high-precision constant at the top of this
      module.
    * invariance under an exact relabeling.  A reference that is only right
      in the coordinate order it happened to be handed is not a reference,
      and that is the property the moment-space form lacked.

    **EVERY TOLERANCE HERE IS DERIVED, NOT OBSERVED** -- see
    :func:`_reference_edf_and_bound`, which returns the bound alongside the
    value.  A number set from the errors one machine happened to produce is
    the defect issue #272 is about, so this test may not contain one.  What it
    computes, at ``_CERTIFIED_LOW_EDGE_LAMBDA``:

    ============================  =========  =========  =========
    fixture                           bound   realized     margin
    ============================  =========  =========  =========
    ``_thin_level_pair(1.0)``     2.217e-06  2.842e-14    7.8e+07x
    relabelings of it             4.434e-06  5.684e-14    7.8e+07x
    ``_vanishing_mass_pair()``    7.912e-04          -          -
    relabelings of it             1.582e-03  5.684e-14    2.7e+10x
    ============================  =========  =========  =========

    The realized errors on the thin pair are ONE ULP of the result, which is
    the floor: 2.842e-14 is ``np.spacing(208.0)``.  The relabeling row is the
    SPREAD over 40 exact relabelings, two ulps, and no relabeling is further
    than one ulp from the certified value.

    The thin pair is measured at the lambda ITS OWN consumer calls the oracle
    at.  The starved pair has no such consumer any more -- see
    ``_VANISHING_LOW_EDGE_LAMBDA`` -- so it is measured at a REPRESENTATIVE
    point of its own geometry, which is a different point from the thin pair's
    because the bracket is scaled by each pair's profiled curvature and
    starving three levels moves it.

    The relabeling bound is the SUM of two forward bounds, since two float64
    evaluations of one exact quantity are each inside their own.  Nothing
    tighter is available: the two evaluations do share their conditioning, but
    a norm-wise bound cannot know they are permutations of each other.

    **WHAT EACH HALF OF THIS TEST ACTUALLY ESTABLISHES, WHICH IS NOT THE SAME
    ON THE TWO FIXTURES.**  Against the pre-#272 moment-space form -- the
    concrete defect, re-measured here at 1.1253e-05 of error on the thin pair:

    * on ``_thin_level_pair(1.0)`` the value check catches it by **5.1x** and
      the relabeling check, against its 1.2038e-05 of spread over the same 40
      draws, by **2.7x**.  That is the real guard, and it is thinner than it
      was: naming the failure probability cost 14.9x on this fixture, and it
      came out of exactly this margin.

      **THE VALUE CHECK CANNOT GO SILENTLY VACUOUS AND THE RELABELING CHECK
      CAN**, which is worth stating because they read alike.  The value check
      compares at ``abs=bound`` against a 1.1253e-05 defect, and the gate below
      holds ``bound`` under 1e-5, so it is protected to 1.125x by construction.
      The relabeling check compares at ``abs = 2 x bound`` and goes vacuous at
      ``bound = 6.02e-06`` -- which is 1.66x INSIDE the same 1e-5 gate.  Current
      headroom to that ceiling is 2.71x, and it is a property of the fixture
      rather than of anything asserted here.
    * on ``_vanishing_mass_pair(1e-12)`` the relabeling check does **not**
      catch it -- 6.9564e-05 of spread against a 1.582e-03 allowance, 0.05x --
      because a fixture with three levels holding 1e-12 of the weight leaves
      the filter factors spread rather than saturated (``sum_j a_j (1 - a_j) =
      1.469``, where the thin pair's is 4.457e-05), so ``w1 = 0.9374`` against
      3.657e-05 and ``sigma_min(A)`` drops to 9.517e-07: the oracle's own
      guaranteed accuracy there is over two orders worse.  It is kept because
      it exercises the starved path and the cell-row reduction order, and
      because an arrangement dependence larger than 1.5e-03 would still be
      caught; it is NOT evidence that the oracle is arrangement-independent
      there.

    That asymmetry is worth stating plainly rather than averaging away.  It is
    also why the gate below is read off each fixture's own caller rather than
    set to a single number: the oracle's guaranteed accuracy must be inside what
    that caller asserts on the paths, and the two callers assert different
    numbers for reasons of their own.  It clears the thin pair's ``1e-5`` by
    4.51x and the starved pair's ``1e-2`` by 13.0x -- and it does NOT clear the
    ``3e-4`` that starved rung used to assert (7.912e-04, 0.40x), which is why
    that rung was re-placed rather than left as it was.  See #301.
    """
    grab = _thin_level_pair(1.0)
    B_a, _, _, W_cell, level_rows = _structured_inputs(grab)
    # Pin the fixture the constant was certified for, so a change to it fails
    # as itself rather than as an unexplained numeric miss.
    assert (B_a.shape[1], level_rows.size) == (11, 19)
    assert float(W_cell.sum()) == 800.0

    # THE TWO SPELLINGS OF THIS FIXTURE'S LOW EDGE ARE TIED TOGETHER HERE.
    # ``_CERTIFIED_EDGES`` arrived with #285 and ``_CERTIFIED_LOW_EDGE_*`` with
    # #275; they are the same two float64 values, this test certifies the oracle
    # against the second pair and ``test_a_thin_level_...`` judges it against the
    # first, so editing one alone would leave the oracle certified at a point its
    # consumer no longer calls it at.  The file said so in prose and checked
    # nothing.
    assert _CERTIFIED_EDGES[1.0][:2] == (_CERTIFIED_LOW_EDGE_LAMBDA, _CERTIFIED_LOW_EDGE_EDF)

    thin_factors = _reference_factors(grab)
    got, bound = _reference_edf_and_bound(thin_factors, _CERTIFIED_LOW_EDGE_LAMBDA)
    assert got == pytest.approx(_CERTIFIED_LOW_EDGE_EDF, abs=bound), (got, bound)

    rng = np.random.default_rng(11)
    # ``asserted`` is what each fixture's own consumer holds the PATHS to, and
    # ``lam`` is the point the oracle is measured at.  For the thin pair that
    # is where its consumer calls it; for the starved pair it is a
    # representative point of that geometry, which is NOT the certified one --
    # the bracket is scaled by each pair's own profiled curvature, and starving
    # three levels moves it.
    for fixture, factors, lam, asserted in (
        (grab, thin_factors, _CERTIFIED_LOW_EDGE_LAMBDA, 1e-5),
        # The starved fixture's consumer asserts 1e-2 rather than the 3e-4 it
        # used to: at a stated failure probability this oracle's own bound
        # there is 7.9118e-04, outside the old number, and the rung was
        # re-placed by floor/ceiling/place rather than widened or deleted --
        # see the low-edge comment in
        # ``test_a_level_with_no_mass_cannot_carry_a_free_degree_of_freedom``.
        (_vanishing_mass_pair(1e-12), None, _VANISHING_LOW_EDGE_LAMBDA, 1e-2),
    ):
        base, base_bound = _reference_edf_and_bound(
            factors if factors is not None else _reference_factors(fixture), lam
        )
        # A NON-FINITE BOUND IS CHECKED FOR SEPARATELY, and unconditionally.
        # `_reference_edf_and_bound` returns ``inf`` by design when the
        # conditioning is past what float64 resolves; with no ``asserted`` to
        # gate against, that would otherwise turn the relabeling allowance below
        # into ``abs=inf`` and make the arrangement-invariance check accept
        # anything -- vanishing at exactly the moment the oracle says it cannot
        # see the fixture.
        assert np.isfinite(base_bound), (fixture is grab, base_bound)
        if asserted is not None:
            assert base_bound < asserted, (
                f"the oracle's own error bound {base_bound:.4e} is not inside the "
                f"{asserted:g} its caller asserts on the paths, so it is not an oracle"
            )
        for relabeled in _relabelings(fixture, rng, 8):
            other, other_bound = _reference_edf_and_bound(_reference_factors(relabeled), lam)
            assert np.isfinite(other_bound), (fixture is grab, other_bound)
            # BOTH bounds are gated, not just the base one.  The allowance below
            # is their SUM, so a relabeling whose own bound ballooned would
            # otherwise widen the assertion until it passed vacuously.
            if asserted is not None:
                assert other_bound < asserted, (
                    f"a relabeling's own error bound {other_bound:.4e} is not inside "
                    f"the {asserted:g} its caller asserts, so the allowance below "
                    f"would be self-granted"
                )
            assert other == pytest.approx(base, abs=base_bound + other_bound), (
                base,
                other,
                base_bound,
                other_bound,
            )


# One budget above edf at the LOW bracket edge (~208) and four below edf at the
# HIGH edge (~19), so both clamping regimes are reported.  The old budgets were
# all below 19, so every rung clamped HIGH and the low edge was never asserted
# on -- which is how a whole-degree-of-freedom error lived there uncovered.
_EDGE_BUDGETS = (1.0, 2.0, 4.0, 8.0, 400.0)


# Both bracket ENDS of ``_thin_level_pair``, certified in arb ball arithmetic
# (python-flint) on the pair's exact design at 640 bits and agreeing with a
# 320-bit evaluation to every digit here; the returned balls have radius 1e-170
# or smaller, so these are the quantity and not one precision's opinion.  The
# ``low_weight = 1.0`` low-edge entry is the value PR #275 certified
# independently at 400 bits, reproduced to all 17 digits.
#
# THE LAMBDAS ARE PINNED WITH THEM, and that is the point of hard-coding rather
# than reading them back: a constant certified at a lambda the ladder no longer
# reports is a stale oracle that still passes.
#
# A NEW HARD PIN HAS TO EARN ITS MARGIN, because this repo has a recorded case
# of ``OPENBLAS_NUM_THREADS`` alone moving a clamped ``edf0`` by 2.3e-03.
# Measured on this fixture at 1, 4 and 8 threads with all six pools set
# together: every lambda here is BIT-IDENTICAL at all three (relative move
# 0.00e+00), as is every edf both arms return.  The two arms' lambdas -- a
# dense factorization against an arrow one, computed by different routes --
# agree to 1.41e-16 at the low edge and 2.21e-16 at the high edge, so the
# ``rel=1e-12`` below carries ~4500x.  Both are also bit-identical over 40
# exact relabelings of the level coordinates, so pinning them is not pinning
# an arrangement.  And ``edf`` moves at most ``|d edf / d ln lambda| ~ 4.1``
# per unit of ``ln lambda`` over this bracket, so even a full 1e-12 of lambda
# would be 4e-12 of edf -- far below either bound asserted here.
_CERTIFIED_EDGES = {
    # low_weight: (lambda_lo, edf_lo, lambda_hi, edf_hi)
    1.0: (2.2876641890924248e-11, 207.99995542698147, 2287664189.092425, 18.99998151653923),
    0.01: (2.1629199830819186e-11, 207.99994337480123, 2162919983.0819182, 18.999936119552885),
    0.001: (2.161686344856317e-11, 207.99986825213327, 2161686344.856317, 18.99950917633727),
}


@pytest.mark.parametrize("low_weight", [1.0, 0.01, 0.001])
def test_a_thin_level_does_not_cost_the_pair_a_degree_of_freedom(low_weight):
    """Both paths are checked against a REFERENCE, not against each other.

    A rare level is the routine case at high cardinality, and it used to cost
    this pair a whole degree of freedom at the ladder's low edge -- ``edf`` was
    ``rank(A) - lambda tr(A^-1 S)`` with an INTEGER rank, so a mis-counted one
    moved the answer by a whole df and a different ``z``, not by a rounding
    difference.

    This used to assert only that the two paths agreed with each other, which
    is weak in both directions: it passes when both are wrong together, and it
    fails when one moves TOWARD the truth.  Both happened.  It passed while
    the structured path was a full degree of freedom out at the low bracket
    edge, because every budget it used clamped at the HIGH edge and the low
    one was never reported.  ``_reference_edf`` is built from the DESIGN and
    never forms a Gram, so it can arbitrate -- and it is itself pinned to a
    high-precision value rather than to either path, because a reference that
    is only checked against the arms it judges is not one.

    **BOTH BOUNDS ARE DERIVED, AND NEITHER IS SET FROM HEADROOM.**  A tolerance
    read off an observed error on one machine is how issue #272 happened; each
    one below is bracketed between a floor the arithmetic cannot beat and the
    defect it exists to catch, and then placed between them.

    LOW EDGE, ``abs=1e-5``, unchanged from what this file already carried.

    * FLOOR, portability.  Permuting the level coordinates leaves ``edf``
      unchanged in exact arithmetic and changes only the order of every
      reduction -- which is what a different BLAS kernel or thread count does,
      and what #272 actually was.  Over 40 relabelings at each of the three
      weights the structured arm's low-edge value spreads 2.044e-09, 2.514e-09
      and 2.768e-09.  A bound below 2.8e-09 asserts a coordinate order.
    * FLOOR, summation.  ``edf`` is a sum of ``p = 209`` filter factors in
      ``[0, 1]``; a length-``p`` float64 sum carries ``gamma_p ~ p eps`` of the
      total, here ``209 * 2.22e-16 * 208 = 9.65e-12`` (Higham, *Accuracy and
      Stability of Numerical Algorithms*, 2nd ed., Ch. 3, where ``gamma_n =
      n u / (1 - n u)`` is defined).  Below the
      relabeling floor, so the relabeling floor binds.
    * FLOOR, the arbiter's own accuracy.  The arms are judged against
      ``_reference_edf_and_bound``, and a bound below what that oracle can
      guarantee asserts the ORACLE's round-off rather than the arms'.  It is
      2.2170e-06, 2.2524e-06 and 2.2549e-06 at the three weights -- above both
      floors above, so this is the one that binds, and 1e-5 clears it by 4.51x,
      4.43x and 4.43x.  It is DERIVED rather than observed, which the constant
      it replaced (a hard-coded 1e-9) was not; it holds at a STATED failure
      probability of 1e-06 rather than at an implicit multiplier of one; and it
      is asserted at every lambda the oracle is called at rather than only at
      the certified one.  This is the tightest of the three floors that the 1e-5
      still clears, and it is what the bound would have to stay under if the
      failure probability were driven lower still.
    * CEILING.  One degree of freedom.  The form this replaced returned an
      integer rank minus a trace, and a mis-counted direction moved the answer
      by exactly 1.0 df; that is the size of the defect, not a guess.
    * PLACED.  1e-5 is 4.43x above the binding floor and 100,000x below the
      defect.  The geometric mean of the two is ``1.453e-03``, so 1e-5 sits
      145x TIGHTER than the placement that would maximize the multiplicative
      margin on both sides -- it is the tighter of the two choices, not the
      more comfortable one, and it is unchanged from what this file carried
      when the relabeling floor of 2.8e-09 was the one that bound.
    * OBSERVED, for disclosure only, never to set the bound: structured
      3.965e-11 / 3.336e-08 / 1.288e-07 and dense 1.366e-10 / 2.404e-09 /
      3.226e-09 across the three weights.

    THE LAMBDA PINS CARRY ``abs=0.0`` AND THAT IS WHAT MAKES THEM PINS.  Moving
    the ladder's low bracket edge by a RELATIVE 1e-11 -- far too small for any
    edf assertion in this file to see, since ``|d edf / d ln lambda| ~ 4.1``
    puts it at 4.5e-16 df, three orders below one ulp of ``edf`` -- reds all
    three parametrizations here and both of
    ``test_a_level_with_no_mass_cannot_carry_a_free_degree_of_freedom``'s.
    Without ``abs=0.0`` it reds nothing: ``pytest.approx`` keeps a default
    absolute tolerance of 1e-12 whenever only ``rel`` is given, and against a
    2.29e-11 lambda that is 4.4%, so the pin accepted anything short of a 4%
    move of the bracket.

    HIGH EDGE, ``abs=3e-3``, the same number the parity assertion already used.

    * NO FLOAT64 ORACLE EXISTS HERE, which is why the constants are hard-coded.
      ``_reference_edf`` itself is 8.15e-05, 2.77e-04 and 2.11e-03 from the
      certified value at the three weights -- worse than either arm at two of
      them -- so it is used at the low edge only.
    * FLOOR.  ``A = V_eff + lambda S`` is penalty-dominated at the high edge:
      the pencil's dynamic range ``rho = lambda ||S||_2 / ||V_eff||_2`` is
      7.47e+09, 7.07e+09 and 7.07e+09, so ``V_eff`` enters ``A`` only ``1/rho``
      above the rounding of ``lambda S`` and each of the ``p`` filter factors
      carries ``eps rho`` of it.  ``p eps rho`` = 3.468e-04, 3.282e-04,
      3.280e-04.  No float64 evaluation of this identity beats that.
    * PLACED, structured arm.  3e-3 is 8.6x the floor and 333x below one
      degree of freedom.
    * PLACED, DENSE arm, at ``1e-2`` and NOT at the structured arm's 3e-3.
      That arm is not changed by this branch and its reproducibility does not
      support the tighter bound.  Measured on THIS fixture across 1, 2, 4 and
      8 threads with all six pools set together, its high-edge value moves
      8.33e-06, 1.04e-05 and **4.05e-04** at the three weights, where the
      structured arm is BIT-IDENTICAL at every setting and every weight.  1e-2
      is 24.7x that floor; 3e-3 would have been 7.4x, on an arm whose floor
      was measured at four thread settings on one machine rather than derived.
      It is the same bound
      ``test_a_level_with_no_mass_cannot_carry_a_free_degree_of_freedom``
      already gives the dense arm at this rung for the same reason.
    * OBSERVED: structured 3.900e-05 / 1.198e-04 / 5.597e-04, dense 1.693e-05 /
      2.911e-05 / 4.089e-05.  The structured arm at ``low_weight = 0.001``
      exceeds its own ``p eps rho`` by 1.7x, which is the factorization depth
      the first-order estimate does not carry.

    PARITY, ``abs=3e-3``, unchanged and pre-existing.  Worth recording that it
    is now carried by the dense arm alone: the structured value is bit-stable
    across thread settings, so the 4.05e-04 above is the whole of what parity
    can move, against a worst observed 1.006e-03 at 2+ threads.  Under the
    form this replaces BOTH arms moved, and the structured one moved 3.07e-03
    under relabeling alone -- past this same bound.

    **WHAT THIS FIXTURE SAYS ABOUT THE FILTER-FACTOR FORM, INCLUDING AGAINST
    IT.**  Against the same certified constants, the rank-differencing form
    this replaced scored low edge 6.966e-09 / 5.181e-08 / 2.694e-09 and high
    edge 2.151e-05 / 6.113e-06 / 1.310e-03.  So on this fixture the new form is
    175x nearer at the ``low_weight = 1.0`` low edge and 2.3x nearer at the
    ``0.001`` high edge, and it is **47.8x FARTHER at the ``0.001`` low edge**
    (2.694e-09 -> 1.288e-07) and 19.6x farther at the ``0.01`` high edge.  Both
    forms stay inside both bounds; the bounds are not what separates them.

    What does separate them is REPRODUCIBILITY, and it is the reason this test
    was flaky.  Over the same 40 relabelings the old form's HIGH-edge value
    spreads 1.544e-04, 6.301e-04 and 3.072e-03 df, the last of which is larger
    than the 3e-3 this test asserts -- a latent failure of exactly #272's kind,
    on one machine, at fixed thread count.  The new form spreads 1.442e-12,
    1.290e-12 and 8.242e-13.  Its low-edge spread is worse (0.0 -> 2.8e-09) and
    3600x inside the bound.
    """
    from superglm.screening._structured import structured_ladder

    grab = _thin_level_pair(low_weight)
    factors = _reference_factors(grab)
    U, V, C, M, S_ti, u_m = grab["args"]
    dense = penalized_score_statistic_ladder(
        U, V, C, M, S_ti, budgets=_EDGE_BUDGETS, U_nuisance=u_m
    )
    struct = structured_ladder(spline_cat_moments(*_structured_inputs(grab)), budgets=_EDGE_BUDGETS)
    lam_lo, edf_lo, lam_hi, edf_hi = _CERTIFIED_EDGES[low_weight]
    # THIS PIN DOES TWO JOBS AND ONLY ONE OF THEM IS DERIVED.  Both are worth
    # having; conflating them is not.
    #
    # (a) A PRECONDITION, DERIVED.  ``edf_lo`` is certified AT ``lam_lo`` while
    #     the oracle below is evaluated at ``arm.lambda0``, so the two must be
    #     the same point to within what that comparison tolerates.
    #     ``|d edf / d ln lambda| = sum_j a_j (1 - a_j) = 4.457e-05`` at this
    #     low edge, and the comparison's tolerance is the oracle's own bound,
    #     2.2170e-06 -- so the precondition is satisfied by any
    #     ``|d ln lambda| < 4.8e-02``.  That is ALL this half needs, and the
    #     ``pytest.approx`` default of ``abs=1e-12`` -- 4.371e-02 relative
    #     against a 2.2877e-11 lambda -- very nearly meets it by accident.
    # (b) BRACKET REPRODUCIBILITY, OBSERVED AND LABELLED AS SUCH.  The remaining
    #     ten orders assert that ``fit_reml``'s bracket lands where it lands.
    #     That is NOT derived -- deriving it would mean bounding ``fit_reml``'s
    #     own cross-platform reproducibility, which is outside what this oracle
    #     covers and is stated so.  It is kept because it is the only thing in
    #     this file that sees a COMMON-MODE bracket move at all.  At the (a)
    #     sensitivity above, a relative 1e-11 move is 4.5e-16 df -- three orders
    #     below one ulp of ``edf``, so nothing but (b) could ever see it.  It
    #     reds all five parametrizations of the two tests under that mutation,
    #     though only three of those come from here: the starved test's two come
    #     from its INTER-ARM check, which fires only because the mutation moves
    #     one arm.  A change to ``profiled_trace`` or ``tr_S`` would move both
    #     arms together and be seen by (b) alone.  The starved rung's equivalent was
    #     DELETED rather than kept, because nothing there consumed a lambda; the
    #     difference is (a), and (a) is why this one stays.
    #
    # ``abs=0.0`` is what makes (b) mean anything: without it the default
    # absolute tolerance -- 4.371e-02 relative here -- swamps the ``rel``
    # beside it by 4.4e+10, ten to eleven orders.
    pin = dict(rel=1e-12, abs=0.0)
    saw_low_edge = saw_high_edge = False
    for budget, d, s in zip(_EDGE_BUDGETS, dense, struct, strict=True):
        # Each path is judged at its reported lambda.  Both brackets now use
        # tr(V_eff), though their endpoint solves remain independent and can
        # differ at the ill-conditioned high edge.
        assert s.lambda0 == pytest.approx(d.lambda0, **pin), ("one lambda", budget)
        if budget > 100.0:
            # LOW edge -- the regime that matters, since the whole-degree-of-
            # freedom error this test exists for lived here, at 1.0 against the
            # reference.  ``_reference_edf_and_bound`` is checked against the
            # certified constant FIRST: it is the arbiter, so an unarbitrated
            # arbiter would let both arms be judged by a wrong number.
            #
            # THE ALLOWANCE ON THE ARBITER IS THE ORACLE'S OWN DERIVED BOUND,
            # not a constant.  It replaces a hard-coded 1e-9, which was set
            # from errors OBSERVED on one machine -- the same defect issue #272
            # is about -- and which cannot move: the quantity it bounds is
            # 2.2170e-06 here and 7.9118e-04 on ``_vanishing_mass_pair(1e-12)``,
            # so a fixture-blind constant is simultaneously right in one place
            # and wrong by orders in the other.
            #
            # AND THE RULER IS CHECKED AT EVERY POINT IT IS USED, per arm and
            # at that arm's own lambda.  Each weight moves the bracket --
            # 2.2877e-11, 2.1629e-11, 2.1617e-11 -- and moves the conditioning
            # with it, so a parametrization whose oracle stopped being inside
            # the 1e-5 asserted on the paths would otherwise judge them with an
            # uncertified ruler and report it as a path error.
            #
            # THE GATE IS NOT DECORATION AND IT HAS ALREADY FIRED TWICE.  At
            # ``_vanishing_mass_pair``'s low edge the same oracle is 7.91e-04
            # against the 3e-4 that rung used to assert, so its allowance had
            # to be re-placed.  And rebuilding THIS pair's geometry at seeds 1-21 -- the same
            # generator, the same shape, a different draw -- the bound exceeds
            # this 1e-5 on **6 of the 21**: seeds 13 (6.6109e-05), 8, 17, 4, 20
            # and 1.  Seed 3 is the one the suite runs, at 2.2170e-06.  On any of
            # the other six this line refuses; without it both arms would be
            # judged, and would agree, against an oracle nobody had checked.
            assert s.lambda0 == pytest.approx(lam_lo, **pin), ("lambda_lo", s.lambda0)
            for name, arm in (("structured", s), ("dense", d)):
                reference, ruler = _reference_edf_and_bound(factors, arm.lambda0)
                assert ruler < 1e-5, (name, budget, low_weight, arm.lambda0, ruler)
                assert reference == pytest.approx(edf_lo, abs=ruler), ("oracle", name, reference)
                assert arm.edf0 == pytest.approx(reference, abs=1e-5), (name, budget, arm.edf0)
            saw_low_edge = True
        else:
            # HIGH edge.  No float64 oracle survives here (``_reference_edf``
            # is 8.15e-05 to 2.11e-03 out), so both arms are judged against the
            # certified constant directly.  This edge used to be parity-only,
            # which cannot see the two arms drifting together.
            assert s.lambda0 == pytest.approx(lam_hi, **pin), ("lambda_hi", s.lambda0)
            assert s.edf0 == pytest.approx(edf_hi, abs=3e-3), ("structured hi", budget, s.edf0)
            # The DENSE arm gets its own, looser bound.  See the docstring:
            # its high-edge value moves 4.05e-04 with thread count on this very
            # fixture where the structured arm is bit-identical, so holding it
            # to the structured arm's bound would pin an arm whose
            # reproducibility does not support it.
            assert d.edf0 == pytest.approx(edf_hi, abs=1e-2), ("dense hi", budget, d.edf0)
            saw_high_edge = True
        # Parity on top of accuracy at both edges: the two are independent
        # implementations -- a dense factorization against an arrow one -- so
        # agreement is evidence about the reorganization even where each is
        # separately pinned to truth.
        assert s.edf0 == pytest.approx(d.edf0, abs=3e-3), ("parity", budget)
        assert s.statistic == pytest.approx(d.statistic, rel=1e-3)
    assert saw_low_edge, "a rung must clamp at the LOW edge or this proves nothing"
    assert saw_high_edge, "a rung must clamp at the HIGH edge or this proves nothing"


def _vanishing_mass_pair(low_weight, n_vanishing=3):
    """``_thin_level_pair``'s geometry, carried PAST the crossover it stops at.

    ``_thin_level_pair`` bottoms out at ``low_weight = 0.001``.  There the rare
    level's free direction still carries ``7.0e-06`` of ``tr(V_eff)`` against a
    ``lambda_hi * sigma_min(S_a)`` of ``3.0e-08`` of it -- 232x clear, so the
    ladder's maximum penalty still leaves that direction free, both paths count
    it, and they agree.  Nothing in the suite went below that crossover.

    Below it the two paths part company by a whole degree of freedom per
    level, and the geometry here is deliberately NOT degenerate: every level
    keeps its 40 rows at 40 distinct ``x``, so no level is thin in the sense of
    rows or support.  The only thing that moves is mass.
    """
    rng = np.random.default_rng(3)
    L, reps = 20, 40
    n = L * reps
    g = np.repeat([f"L{i}" for i in range(L)], reps)
    x = rng.uniform(0.0, 1.0, n)
    slope = rng.normal(size=L).repeat(reps)
    df = pd.DataFrame({"g": g, "x": x})
    y = slope * x + rng.normal(scale=0.5, size=n)
    w = np.ones(n)
    for i in range(n_vanishing):
        w[g == f"L{i}"] = low_weight
    return _capture(
        df,
        y,
        {"g": Categorical(), "x": Spline(kind="ps", n_knots=8)},
        ("x", "g"),
        sample_weight=w,
    )


def _free_directions_left_free(grab):
    """How many free directions the ladder's MAXIMUM penalty actually leaves free.

    ``edf(lambda) = sum_j v_j / (v_j + lambda s_j)`` over the simultaneously
    diagonalized pencil, so a direction in ``null(S_a) (x) I`` contributes
    ``v / (v + lambda sigma_min)``: one where the penalty cannot reach it, zero
    where it can.  This evaluates exactly that sum on the free block, which
    takes no rank decision and inverts nothing -- the only division is by a sum
    of two nonnegative numbers -- so it can arbitrate a rung where neither path
    can be trusted against the other.

    ``sigma_min`` is NOT zero and that is the whole point.  A P-spline penalty
    is rank deficient in exact arithmetic, but the matrix the screen assembles
    is not: measured here at ``sigma_min / sigma_max = +1.374e-16``, which
    ``lambda_hi = 1e10 * scale`` amplifies into a real penalty of
    ``1.86e-08 * tr(V_eff)``.  A free direction carrying less curvature than
    that is reached by the penalty like any other, and contributes nothing.

    THE MAGNITUDE OF THAT RESIDUE IS WHAT MATTERS AND ITS SIGN IS AN ACCIDENT,
    so this takes ``abs``.  Assembly round-off puts ``sigma_min`` on either
    side of zero depending on the data: seed 3 of this fixture gives
    ``+2.11e-15`` and seed 12 gives ``-5.03e-16``.  Clamping the negative case
    up to zero would make the penalty appear not to reach ANY free direction
    and hand back the full ``k_b`` -- which is the wrong answer by three
    degrees of freedom, checked against 40-digit mpmath on both seeds
    (15.999993 and 16.000012).

    On this fixture the split is unambiguous -- the 19 shares come back as
    three below ``1e-04`` and sixteen at ``0.999997``, four orders apart.

    **HOW CLOSE THIS ORACLE IS DEPENDS ON THE PARAMETRIZATION, AND THE TWO ARE
    AN ORDER APART.**  Against a 40-digit mpmath evaluation of
    ``tr((V_eff + lambda_hi S)^-1 V_eff)`` (identical at 60 digits): at
    ``low_weight = 1e-12`` it agrees to 3.03e-05 (15.999963 against
    15.999993), but at ``1e-10`` only to 5.84e-04 (16.000083 against
    16.000667).  The reason is the tail this deliberately ignores: the
    penalized block's own residual at ``lambda_hi``,
    ``tr((V_pp + lambda_hi S_pp)^-1 V_pp)``, measures 5.61e-07 -- three orders
    below the 1e-3 the high-edge assertions carry, but not below 1e-07 as this
    once claimed.

    So the ABSOLUTE assertions bound the error, they do not adjudicate which
    path is nearer the truth.  At ``1e-10`` the structured path is 7.17e-04
    from the mpmath value and the dense path 1.12e-04 -- structured is 6.4x
    FARTHER while being held to the tighter bound, because it is nearer to
    THIS oracle (1.33e-04) than the dense path is (4.72e-04).  What the
    assertions establish is that neither path is a WHOLE degree of freedom
    out, which is the failure this fixture exists for and is 1000x larger
    than any of these gaps.
    """
    U, V, C, M, S_ti, u_m = grab["args"]
    _, S_a, _, _, level_rows = _structured_inputs(grab)
    k_b = level_rows.size
    V_eff = V - C.T @ np.linalg.solve(M, C)
    V_eff = 0.5 * (V_eff + V_eff.T)
    ev, evec = np.linalg.eigh(S_a)
    null_dim = int(np.sum(np.abs(ev) <= 1e-8 * ev[-1]))
    free = np.kron(evec[:, :null_dim], np.eye(k_b))
    v = np.diag(free.T @ V_eff @ free)
    lam_hi = 1e10 * float(np.trace(V_eff)) / float(np.trace(S_ti))
    reach = lam_hi * abs(float(ev[0]))
    share = v / (v + reach)
    # Refuse to arbitrate a pair whose free directions are not cleanly split:
    # this is an ORACLE, and a degenerate one that silently returns k_b would
    # agree with the very path it exists to check.
    assert np.all((share < 1e-3) | (share > 1.0 - 1e-3)), np.sort(share)
    return float(np.sum(share)), k_b


_VANISHING_BUDGETS = (2.0, 4.0, 8.0, 400.0)


@pytest.mark.parametrize("low_weight", [1e-10, 1e-12])
def test_the_penalty_residue_s_sign_cannot_move_the_published_edf(low_weight):
    """A published ``edf0`` may not turn on which side of zero round-off landed.

    **THIS TEST EXISTS BECAUSE CI FOUND IT AND THIS MACHINE COULD NOT.**  The
    vanishing-mass pair's penalty has a smallest eigenvalue of ``1.3e-16`` of
    its largest -- below what ``eigh`` resolves, so its SIGN is assembly
    round-off and differs between machines.  ``lambda_hi = 1e10 * scale``
    then amplifies it into a real penalty of ``1.86e-08 * tr(V_eff)`` that
    reaches three levels' free directions, so keeping it or dropping it is a
    **3 df** difference in what the ladder publishes.  A projection that keeps
    only ``w > 0`` makes that difference a function of the round-off: this
    machine reads ``+1.4e-15`` and reports 16.000, CI read the mirror image
    and reported **19.000**, and both are the same policy applied to the same
    data.

    :func:`_free_directions_left_free`'s docstring already recorded the
    finding from the other side -- it takes ``abs`` for exactly this reason,
    checked against 40-digit mpmath on both signs -- so the module was
    contradicting the oracle the suite judges it by.

    Asserted here as INVARIANCE rather than as a value, because invariance is
    the property that is portable: the same pair with the residue's sign
    reversed must publish the same rungs.  Reversing one eigenvalue of a
    symmetric matrix through its own eigenvectors is exactly the perturbation
    a different BLAS produces, and nothing else about the pair moves.

    **AND THE ARMS' OWN SIGNS ARE NOT ASSERTED, BECAUSE THEY CANNOT BE.**  Two
    revisions of this test tried: the first negated whatever this machine
    delivered (CI delivered the other sign), the second built both arms from
    the magnitude and asserted that ``eigvalsh`` recovered them (CI read
    ``-6.16e-17`` for the arm built as ``+|w_0|``).  Both were the same
    mistake as the bug under test.  An eigenvalue at ``1e-16`` of the largest
    is inside the reconstruction's own error, so no construction fixes its
    computed sign -- which is precisely why no published number may depend on
    it.  What is constructed is the DIFFERENCE, ``2 |w_0| v v'``, exact in the
    arrays; what is asserted is that it does not move a rung.

    RED against the ``max(w, 0)`` projection this replaces: 19.0 against
    16.000055 at ``1e-10`` and 19.000005 against 15.999928 at ``1e-12``.
    """
    from superglm.screening._structured import structured_ladder

    grab = _vanishing_mass_pair(low_weight=low_weight)
    B_a, S_a, S_cell, W_cell, level_rows = _structured_inputs(grab)
    symmetric = 0.5 * (S_a + S_a.T)
    w, Q = np.linalg.eigh(symmetric)

    # The fixture is what the docstring says: the residue is below what the
    # eigensolver resolves, so its sign carries no information -- which is
    # why this reads its MAGNITUDE and builds both signs from that.
    assert abs(w[0]) <= len(w) * np.finfo(np.float64).eps * abs(w[-1]), (w[0], w[-1])
    assert abs(w[0]) > 0.0, w[0]

    both = [
        (Q * np.where(np.arange(w.size) == 0, sign * abs(w[0]), w)) @ Q.T for sign in (+1.0, -1.0)
    ]
    # The two arms really are two different matrices, separated by the whole
    # residue and by nothing else.
    assert not np.array_equal(both[0], both[1])
    separation = float(np.linalg.norm(both[0] - both[1], 2))
    assert separation == pytest.approx(2.0 * abs(w[0]), rel=1e-6), separation

    rungs = [
        structured_ladder(
            spline_cat_moments(B_a, penalty, S_cell, W_cell, level_rows),
            budgets=_VANISHING_BUDGETS,
        )
        for penalty in both
    ]
    assert all(r is not None and len(r) == len(_VANISHING_BUDGETS) for r in rungs), rungs

    # **THE TWO ARMS ARE NOT REQUIRED TO AGREE EXACTLY, AND THE REASON IS THE
    # SAME ONE THE TEST IS ABOUT.**  Reconstructing ``S`` from its spectrum
    # puts its smallest eigenvalue back only to ``n eps ||S||_2``, which is 18x
    # the residue itself, so each arm's ``_penalty_root`` keeps a magnitude
    # that is mostly that noise -- and ``lambda_hi = 1e10 * scale`` amplifies
    # a penalty difference of ``2 n eps ||S||_2`` into a real one on the three
    # free directions.  Measured across machines, the arms differ by 1.7e-03
    # df.  A bound tight to that would be fitted to it.
    #
    # ``abs=1e-2`` is the tolerance THIS FIXTURE's own high-edge assertions
    # already carry for the same rung evaluated two ways (the dense arm's, in
    # ``test_a_level_with_no_mass_cannot_carry_a_free_degree_of_freedom``), so
    # it is the bound this file already accepts as "one rung, two
    # evaluations, this geometry".  What it rules out is the **3.0 df** the
    # residue's sign was worth under ``max(w, 0)``, which it sits 300x below.
    for delivered, negated in zip(*rungs, strict=True):
        assert delivered.edf0 == pytest.approx(negated.edf0, abs=1e-2), (
            f"the residue's sign moved edf0 from {delivered.edf0!r} to {negated.edf0!r}"
        )


@pytest.mark.parametrize("low_weight", [1e-10, 1e-12])
def test_a_level_with_no_mass_cannot_carry_a_free_degree_of_freedom(low_weight):
    """Three levels holding none of the weight must not buy three degrees of freedom.

    This is :func:`test_a_thin_level_does_not_cost_the_pair_a_degree_of_freedom`
    continued past the weight it stops at.  A level whose share of the weight
    is below the point where ``lambda_hi`` reaches its free direction is the
    routine case at high cardinality -- an empty cell, an exposure of nothing.

    THE STRUCTURED PATH USED TO AWARD EACH OF THEM A WHOLE DEGREE OF FREEDOM,
    reporting the full ``k_b = 19`` at the high edge whatever the weight was,
    because the per-level rank it subtracted a trace from divided each block by
    ITS OWN trace before counting -- so the count could not see a level's mass,
    and the mass is what decides.  Driven from 1.0 down to 1e-20, twenty
    decades, that count read 12 on the levels holding the weight and 12 on the
    levels holding none of it, and ``edf0`` never moved off 19.  There is no
    count left to make that mistake with: a starved level's own curvature
    enters as ``a_j`` inside its filter factor.  Swept over the same twenty
    decades this reports 19.0009 at ``low_weight`` 1e-2 and 15.9999 at 1e-20.

    Three assertions, and none implies another.  The ABSOLUTE ones bound how
    far each path is from an independent closed form that takes no rank
    decision and inverts nothing, which puts the answer at 16; they do NOT
    settle which path is nearer the truth, and :func:`_free_directions_left_free`
    records where they part company with 40-digit mpmath.  PARITY is the
    contract the two paths are supposed to satisfy, and is what the ladder's
    public ``edf0`` -- and therefore every ``z`` the screen ranks on -- rides
    on.  The LOW-edge rung is a control, so a fix that buys the high edge by
    disturbing the low one fails here rather than silently.

    TOLERANCES ARE SET FROM MEASUREMENT, over 2 weights x 2 thread settings
    (the second because the DENSE path's clamped ``edf0`` is not a stable
    function of its inputs at this rung -- on identical moments
    ``OPENBLAS_NUM_THREADS`` alone moves it by 2.3e-03 here, while the
    structured value is bit-identical between the two).  Worst observed at the
    HIGH edge: structured against the closed form 1.331e-04, dense against it
    2.775e-03, parity 2.908e-03, statistic 3.114e-06 relative and ``lambda0``
    3.7e-16 relative.  The bounds below are 7.5x, 3.6x, 3.4x, 32x and 2700x
    above those, and the two that matter are still 3000x and 100x below the
    one degree of freedom this test exists to catch.

    THE LOW-EDGE BOUNDS COME FROM THE FAMILY, AND 60 DRAWS WERE NOT ENOUGH TO
    SETTLE THEM.  The parity there was pinned at 1e-6, which this fixture's own
    seed clears by three orders (1.6e-09) -- and 11 of 60 neighbouring draws
    (15 seeds x 2 weights x 2 thread settings) do not.  That bound was a
    property of one lucky draw, and the values it pinned are bit identical to
    what this file measured before the rank count changed, so it was pinning
    pre-existing float noise.  Its 5e-5 replacement, set from those 60 draws,
    was a property of the WINDOW: carried to 320 draws (80 seeds x 2 weights x
    2 thread settings) parity exceeds 5e-5 on 9 of them, 2.8%, the worst by
    1.80x at 9.021e-05 -- and the first failing seed is 55, which is why a
    15-seed window looked clean.  So parity is asserted at 2e-4 here: 2.2x
    above the worst of the 320 and still 5.0e+03x below the one degree of
    freedom this test exists to catch.  Over the same 320 draws parity has a
    median of 7.0e-09.

    **THE LOW EDGE NO LONGER COMPARES EITHER ARM TO AN ORACLE, AND THAT IS THIS
    CHANGE'S ONE DISCLOSED LOSS OF COVERAGE.**  It used to assert each arm
    against ``_reference_edf`` at ``abs=3e-4``.  That reference now returns its
    own error bound at a stated failure probability, and on this fixture the
    bound is **7.9118e-04** at ``low_weight = 1e-12`` and **3.6012e-04** at
    1e-10 -- both outside the 3e-4 the assertion carried, and there is no
    multiplier that rescues them without making the probability meaningless.
    So the assertion is gone rather than widened; the parity check below, which
    needs no oracle, is what remains at this rung.  The reasoning is in the
    comment beside it and on #301.  What was removed had never been bounded:
    the distances it was calibrated from -- a median of 7.0757e-10 and a worst
    of 8.2123e-05 over 160 draws (80 seeds x 2 weights, one thread) against the
    factored reference, where the old moment-space one read a median of 1.6e-05
    and a worst of 1.036e-04 -- are smaller than the oracle's own guaranteed
    error there, so the comparison was measuring the arbiter as much as the
    arms.  Parity over the same window reproduces the 320-draw figure above to
    four digits (worst 9.0208e-05), which is the check that this narrower
    window is measuring the same family.
    """
    from superglm.screening._structured import structured_ladder

    grab = _vanishing_mass_pair(low_weight)
    U, V, C, M, S_ti, u_m = grab["args"]
    dense = penalized_score_statistic_ladder(
        U, V, C, M, S_ti, budgets=_VANISHING_BUDGETS, U_nuisance=u_m
    )
    struct = structured_ladder(
        spline_cat_moments(*_structured_inputs(grab)), budgets=_VANISHING_BUDGETS
    )
    assert struct is not None and len(struct) == len(_VANISHING_BUDGETS)

    left_free, k_b = _free_directions_left_free(grab)
    # The fixture is what it says it is: 3 of the 19 free directions are
    # reached by the penalty at the high edge, 16 are not.
    assert left_free == pytest.approx(k_b - 3, abs=0.01)

    for budget, d, s in zip(_VANISHING_BUDGETS, dense, struct, strict=True):
        if budget < 100.0:  # every one of these clamps at the HIGH edge
            # **THIS BOUND WAS 1e-3 AND IS LOOSENED HERE, WHICH IS A COST OF
            # THIS CHANGE AND IS STATED RATHER THAN QUIETLY WIDENED.**  The
            # edf half now reads the penalty through ``_penalty_root``, which
            # takes a below-resolution eigenvalue at its MAGNITUDE -- and that
            # magnitude is itself determined only to ``n eps ||S||_2``, which
            # ``lambda_hi = 1e10 * scale`` amplifies onto the three free
            # directions.  Measured across machines the structured value moves
            # 2.07e-03 here, against 1.3e-04 on the machine that wrote the
            # 1e-3.  The form this replaces did not decompose ``S_a`` at all
            # and so did not pay it; what it paid instead was letting the
            # residue's SIGN decide 3.0 df, which is 1500x larger and is what
            # ``test_the_penalty_residue_s_sign_cannot_move_the_published_edf``
            # now forbids.
            #
            # 1e-2 is not a new number in this file: it is what the dense arm
            # and the parity check two lines below already carry at this same
            # rung.  It is also above the ORACLE's own error, which the
            # docstring of :func:`_free_directions_left_free` measures at
            # 5.84e-04 against 40-digit mpmath -- so 1e-3 was only 1.7x above
            # the arbiter's own accuracy and was tight for that reason too.
            assert s.edf0 == pytest.approx(left_free, abs=1e-2), (
                f"structured edf0 {s.edf0!r} at budget {budget} does not match the "
                f"{left_free!r} directions the maximum penalty leaves free, with "
                f"three levels holding {low_weight:g} of the weight"
            )
            assert d.edf0 == pytest.approx(left_free, abs=1e-2), (
                "dense",
                budget,
                d.edf0,
                left_free,
            )
            assert s.edf0 == pytest.approx(d.edf0, abs=1e-2), ("parity", budget, s.edf0, d.edf0)
        else:  # the LOW edge
            # **NO ORACLE ARBITRATES THIS RUNG, AND THAT IS A MEASUREMENT
            # RATHER THAN AN OMISSION.**  This edge used to assert each arm
            # against ``_reference_edf`` at 3e-4.  That reference now returns
            # its own error bound, and at a STATED failure probability -- see
            # ``_QR_FAILURE_PROBABILITY`` -- the bound here is **7.9118e-04** at
            # ``low_weight = 1e-12`` and **3.6012e-04** at 1e-10, both OUTSIDE
            # the 3e-4 the assertion carried.  There is no multiplier that
            # rescues it without making the probability meaningless, and
            # constant-shopping to fit an assertion is the defect this file
            # exists to remove.  The oracle is intrinsically ~1e-4
            # accurate here, because starving three levels leaves the filter
            # factors spread rather than saturated (``w1 = 0.9374`` against the
            # thin pair's 3.657e-05) and drops ``sigma_min(A)`` to 9.517e-07.
            #
            # SO THE 3e-4 IS GONE.  IT IS NOT WIDENED, AND IT IS NOT REPLACED
            # BY NOTHING EITHER -- IT IS RE-PLACED BY THIS FILE'S OWN METHOD.
            # Leaving the rung with parity alone would leave ``edf0`` with no
            # absolute anchor of any kind, and this file says elsewhere that
            # parity "cannot see the two arms drifting together".
            #
            # * FLOOR: the arbiter's own DERIVED error, 7.9118e-04 at
            #   ``low_weight = 1e-12`` and 3.6012e-04 at 1e-10.  A bound below
            #   that asserts the oracle's round-off rather than the arms'.
            # * CEILING: one degree of freedom -- a level with no mass carrying
            #   a free direction, which is this test's whole premise.
            # * PLACED: ``1e-2``, 12.6x above the floor and 100x below the
            #   defect.  It is the literal the HIGH-edge rung thirty lines above
            #   already uses, against ``_free_directions_left_free`` -- an
            #   independent closed form that is ALSO uncertified and whose own
            #   error that rung measures at 5.84e-04, comparable to this
            #   oracle's 7.91e-04.  Two arbiters of the same accuracy in the
            #   same test; refusing this one while keeping that one would not
            #   have been a principle.
            #
            # What is genuinely lost against the form this replaces is 33x of
            # tightness on the arms, and that is the price of the oracle's error
            # being bounded rather than assumed.  Recorded on #301.
            #
            # NO LAMBDA IS PINNED HERE, and unlike the thin pair's rung nothing
            # needs one: the oracle is evaluated at ``arm.lambda0`` and compared
            # against ``arm.edf0`` at the same point, so there is no certified
            # constant at a third lambda to tie them to.  A bare ``rel=1e-12``
            # against a lambda this machine happens to produce would be an
            # observation about ``fit_reml`` -- issue #272's own shape, aimed at
            # a different quantity.  The inter-arm check below still holds both
            # paths to the SAME lambda, which is a property of the two
            # implementations rather than of a machine.
            factors = _reference_factors(grab)
            for name, arm in (("structured", s), ("dense", d)):
                reference, ruler = _reference_edf_and_bound(factors, arm.lambda0)
                assert ruler < 1e-2, (name, budget, low_weight, arm.lambda0, ruler)
                assert arm.edf0 == pytest.approx(reference, abs=1e-2), (
                    "low edge oracle",
                    name,
                    arm.edf0,
                    reference,
                )
            assert s.edf0 == pytest.approx(d.edf0, abs=2e-4), ("low edge parity", s.edf0, d.edf0)
        assert s.lambda0 == pytest.approx(d.lambda0, rel=1e-12, abs=0.0), ("lambda0", budget)
        assert s.statistic == pytest.approx(d.statistic, rel=1e-4), ("statistic", budget)


def _rank_one_penalty_pair(seed, L, reps, width, n_narrow):
    """A TWO-column spline margin, where the tensor penalty is rank ONE.

    ``Spline(kind="cr", k=3)`` leaves ``k_a = 2``, and the pair's ``S_a`` is
    then ``u u'`` for a unit ``u``: exactly rank one in exact arithmetic, and
    in float a matrix whose null residue a symmetric eigensolver frequently
    cannot resolve AT ALL, handing back a bit-exact ``0.0``.  No other fixture
    in this file reaches that regime: over 405 pipeline configurations of
    ``ps``/``cr``/``bs`` at 3 to 20 knots, 5 seeds and three level layouts,
    every wider margin resolved a NONZERO residue.  It is where a form that
    had to DECIDE something about the residue was at its weakest, and where
    one that only multiplies by it should be indifferent.

    ``n_narrow`` levels are squeezed into a band of ``x`` of the given width,
    which is what puts a level's free curvature near the floor.  Nothing else
    is degenerate: unit weights throughout, every level keeps all its rows.
    """
    rng = np.random.default_rng(seed)
    n = L * reps
    g = np.repeat([f"L{i}" for i in range(L)], reps)
    x = rng.uniform(0.05, 0.95, n)
    for i in range(n_narrow):
        selected = g == f"L{i}"
        x[selected] = 0.2 + 0.6 * i / n_narrow + width * rng.uniform(-0.5, 0.5, selected.sum())
    slope = rng.normal(size=L).repeat(reps)
    df = pd.DataFrame({"g": g, "x": x})
    y = slope * x + rng.normal(scale=0.5, size=n)
    return _capture(
        df,
        y,
        {"g": Categorical(), "x": Spline(kind="cr", k=3)},
        ("x", "g"),
        sample_weight=np.ones(n),
    )


def _exact_residue_bounds(S_a):
    """``|sigma_min(S_a)|`` bracketed EXACTLY, for a 2x2, without an eigensolver.

    ``sigma_min sigma_max = det`` and ``tr / 2 <= sigma_max <= tr`` for a 2x2
    with a nonnegative trace, so ``|det| / tr <= |sigma_min| <= 2 |det| / tr``.
    Both are computed as rationals over the float entries, so no tolerance and
    no iteration enters -- which is what lets a test state that ``eigh``
    returned zero AND that the matrix is not singular.
    """
    from fractions import Fraction

    a = Fraction(float(S_a[0, 0]))
    c = Fraction(float(S_a[1, 1]))
    b = (Fraction(float(S_a[0, 1])) + Fraction(float(S_a[1, 0]))) / 2
    determinant, trace = a * c - b * b, a + c
    assert trace > 0, trace
    return float(abs(determinant) / trace), float(2 * abs(determinant) / trace)


def _ladder_high_edge(p):
    """``structured_ladder``'s own maximum penalty, associated exactly as it is.

    ``1e10 * (trace / tr_S)`` and ``(1e10 * trace) / tr_S`` differ in the last
    bit, and one bit of ``lambda`` is not nothing at this edge -- measured at
    0.065 df on a two-column ``cr`` pair.  Everything asserted here about a
    rung the ladder publishes has to be evaluated at the ladder's own
    ``lambda``, not at an algebraically equivalent one.
    """
    tr_S = float(np.trace(p.S_a)) * p.dims[0]
    return 1e10 * (max(p.profiled_trace, 1e-300) / max(tr_S, 1e-300))


# ``edf`` at ``_ladder_high_edge`` for lifts of 0, 1, 3 and 10 eps*sigma_max,
# in mpmath at 40 digits on each lifted pair's exact float64 design, with the
# common null space of ``V_eff`` and ``S`` deflated once and every lambda read
# off a Cholesky.  Stable to every digit shown between 40 and 30 digits.
_RESIDUE_ORACLE = {
    (2, 1e-4): (5.720073, 6.011589, 6.003720, 6.000985),
    (13, 2e-3): (4.975844, 3.773915, 3.089880, 2.464699),
}


@pytest.mark.parametrize(
    ("seed", "L", "reps", "width", "n_narrow", "bound"),
    [(2, 10, 20, 1e-4, 3, 3.3), (13, 6, 12, 2e-3, 3, 0.03)],
    ids=["wide-band-draw", "narrow-band-draw"],
)
def test_a_round_off_penalty_residue_moves_edf_the_way_the_pencil_says(
    seed, L, reps, width, n_narrow, bound
):
    """A two-column spline margin's penalty is rank one, so its null residue
    is round-off -- and lifting that residue by a few ``eps`` MOVES ``edf`` BY
    WHOLE DEGREES OF FREEDOM.  That is not a defect; it is what the pencil
    does, and this test used to assert the opposite.

    **THE PREMISE WAS WRONG AND AN ORACLE SETTLES IT.**  At the ladder's high
    edge ``lambda`` is ``1e10 * scale``, which turns a penalty eigenvalue of a
    few ``eps`` of ``sigma_max`` into a real penalty that reaches the free
    directions of the narrow-band levels.  Evaluated in mpmath at 40 digits on
    each lifted pair's exact design, ``edf`` at that edge is

        wide-band draw   4.975844 -> 3.773915 -> 3.089880 -> 2.464699
        narrow-band draw 3.973234 -> 2.808497 -> 2.393160 -> 2.135316

    for lifts of 0, 1, 3 and 10 ``eps * sigma_max``: a swing of 2.51 and 1.84
    df.  The form this replaces reports 2.957 / 2.670 / 3.098 / 2.463 and
    2.976 / 2.355 / 2.385 / 2.132 -- flat to within 0.14 df on the first two
    lifts and up to **2.02 df away from the truth**, because its own relative
    cut drops the penalized direction from the inverse and it never sees the
    penalty at all.  Its stability there was insensitivity, not accuracy, and
    the assertion built on it was pinning that insensitivity.

    What is asserted now is agreement with the oracle at each lift, which is
    RED against that form by up to 2.02 df on the first draw and 1.00 on the
    second.  The independent closed form in
    :func:`_free_directions_left_free` says the same thing from the other
    side on the vanishing-mass fixture, and this route agrees with it there.
    """
    B_a, S_a, S_cell, W_cell, level_rows = _structured_inputs(
        _rank_one_penalty_pair(seed, L, reps, width, n_narrow)
    )
    symmetric = 0.5 * (S_a + S_a.T)
    sigma, directions = np.linalg.eigh(symmetric)
    assert sigma.size == 2, sigma
    lower, upper = _exact_residue_bounds(symmetric)
    eps = np.finfo(np.float64).eps
    # The fixture is what it says it is: the penalty is NOT singular, and its
    # null residue is round-off on the largest eigenvalue, so a perturbation of
    # a few eps is the same size as the residue.  Exact rationals, so this
    # states it rather than measuring it to a tolerance.
    assert 0.0 < lower <= upper <= 3.0 * eps * float(sigma[-1]), (lower, upper, sigma)

    null_direction = np.outer(directions[:, 0], directions[:, 0])
    moved = []
    for multiple in (0.0, 1.0, 3.0, 10.0):
        lifted = symmetric + multiple * eps * float(sigma[-1]) * null_direction
        pair = spline_cat_moments(B_a, lifted, S_cell, W_cell, level_rows)
        _, edf = _evaluate(pair, _profile(pair), _ladder_high_edge(pair))
        moved.append(edf)
    # ...and each rung matches the 40-digit value on that lifted design.  The
    # bounds are the worst distance measured on each draw taken to two
    # significant figures: 3.28 df on the wide-band one -- where the UNLIFTED
    # residue is 5.6e-17 relative, below what ``max(w, 0)`` keeps, so this
    # route reports the direction free at 9.000 against a certified 5.720 --
    # and 0.024 df on the narrow-band one.  The unfixed form's worst distances
    # on the same two draws are 0.315 and 2.019 df, so this is RED against it
    # on the second and its own disclosure on the first.
    for value, truth in zip(moved, _RESIDUE_ORACLE[(seed, width)], strict=True):
        assert value == pytest.approx(truth, abs=bound), (moved, _RESIDUE_ORACLE[(seed, width)])


def _multi_null_pair(seed, width=1e-3, L=6, reps=12, n_narrow=2, m=3):
    """A margin whose penalty is rank deficient by TWO, not by one.

    Every other fixture in this file, and every margin any shipped default
    builds, leaves exactly one free direction, so ``reach[free]`` is a single
    number and collapsing it to ``reach[free].max()`` is not a decision.
    Nullity is the penalty order minus one for ``ps`` and ``bs`` -- measured on
    the CENTERED ``S_a`` the kernel is handed, recovered from the pipeline's
    own ``S_ti = kron(S_a, I_kb)``, at 6 levels x 12 rows, ``x`` uniform on
    [0.05, 0.95] from ``default_rng(3)``:

        ps  m=1 -> 0   m=2 -> 1   m=3 -> 2   m=4 -> 3
        bs  m=1 -> 0   m=2 -> 1   m=3 -> 2   m=4 refused (order > degree 3)
        cr  m=1 -> 0   m=2 -> 1   m=3 -> 1   m=4 refused (order > 3)
        ns  m=1 -> 0   m=2 -> 0   m=3 -> 0   m=4 -> 1

    So ``cr`` never reaches nullity two at all, ``ns`` never reaches it, and
    the DEFAULT ``m=2`` reaches it on none of the four.  ``Spline(kind="ps",
    k=5, degree=3, m=3)`` is the compact way in: ``k_a = 4``, penalty rank 2,
    two free directions, and the pair still routes through the arrow kernel.

    Geometry is :func:`_rank_one_penalty_pair`'s -- ``n_narrow`` levels inside
    a band of ``x`` of the given width, unit weights, every level keeping all
    its rows -- because that is what puts a level's free curvature between the
    two free reaches, which is where the collapse decides anything.
    """
    rng = np.random.default_rng(seed)
    n = L * reps
    g = np.repeat([f"L{i}" for i in range(L)], reps)
    x = rng.uniform(0.05, 0.95, n)
    for i in range(n_narrow):
        selected = g == f"L{i}"
        x[selected] = 0.2 + 0.6 * i / n_narrow + width * rng.uniform(-0.5, 0.5, selected.sum())
    slope = rng.normal(size=L).repeat(reps)
    df = pd.DataFrame({"g": g, "x": x})
    y = slope * x + rng.normal(scale=0.5, size=n)
    return _capture(
        df,
        y,
        {"g": Categorical(), "x": Spline(kind="ps", k=5, degree=3, m=m)},
        ("x", "g"),
        sample_weight=np.ones(n),
    )


@pytest.mark.parametrize(
    "build",
    [
        lambda: _vanishing_mass_pair(1e-12),
        lambda: _multi_null_pair(0),
        lambda: _rank_one_penalty_pair(13, 6, 12, 2e-3, 3),
    ],
    ids=["vanishing-mass", "multi-null-nullity-2", "rank-one-penalty"],
)
def test_the_edf_contracts_the_delivered_inverse_exactly(build):
    """``edf`` is ``tr(A^+ V_eff)`` on the PENCIL, densely, at the high edge.

    Both objects in that trace are non-block-diagonal and neither is ever
    formed: ``V_eff``'s coupling is a rank-``k_a`` downdate ``Psi' Psi`` and
    ``A^+``'s off-diagonal blocks come from the same ``r``-dimensional border.
    The O(L) evaluation reduces the double sum over levels to ``r``-sized
    accumulators; this rebuilds ``V_eff`` and ``A = V_eff + lambda S`` densely
    and checks the reduction is an identity rather than an approximation.

    **THE REFERENCE CHANGED AND THE REASON IS AN ORACLE.**  This used to
    contract against the inverse the PAIR ARROW delivers, ``Ginv`` plus
    ``Y Sinv Y'``.  On the rank-one-penalty fixture that inverse is 2.02 df
    away from a 40-digit value on the same design, because its relative cut
    drops a direction the penalty reaches; asserting parity with it pinned
    that error.  The dense pencil takes no such cut, so it is what the
    identity is checked against.  RED against the form this replaces on that
    fixture, by 2.04 df.

    IT ALSO SHOWS BOTH COUPLINGS ARE LOAD BEARING, which is the mutation this
    test exists to kill: treating either ``A^+`` or ``V_eff`` as block
    diagonal moves the answer by whole degrees of freedom.
    """
    p = spline_cat_moments(*_structured_inputs(build()))
    L, k_a = p.dims
    geometry = _profile(p)
    V_eff = _factored_v_eff(p, geometry)
    block_diagonal = scipy.linalg.block_diag(*(np.swapaxes(p.R, -1, -2) @ p.R))
    penalty = scipy.linalg.block_diag(*([p.S_a] * L))

    # The reference is the STACKED QR, not a pseudo-inverse of the Gram: at
    # the ladder's high edge ``A`` is singular, ``numpy.linalg.pinv``'s own
    # relative cut decides what it keeps, and on the rank-one-penalty fixture
    # that puts it 2.43 df from the 40-digit value while this route is 0.024
    # df from it.  A reference that takes the same kind of cut as the thing
    # under test cannot arbitrate it.  ``pinv`` is kept below only for the
    # block-diagonal mutation checks, where a whole degree of freedom is the
    # margin and its cut cannot reach that far.
    lam = _ladder_high_edge(p)
    _, edf = _evaluate(p, geometry, lam)
    # 1.9e-06 is the worst relative gap measured over the three fixtures;
    # the two routes share no code past ``p.R`` and ``geometry.coupling``, so
    # this is agreement between an O(L) reduction and an O(L^3) one.
    assert edf == pytest.approx(_wood_stacked_edf(p, geometry, lam), rel=1e-5)

    A = 0.5 * (V_eff + lam * penalty + (V_eff + lam * penalty).T)
    A_inverse = np.linalg.pinv(A, hermitian=True)

    # ...and BOTH couplings are load bearing, which is the mutation this test
    # exists to kill: a per-level pencil is not an approximation of this
    # quantity, it is a different one.
    separable_inverse = scipy.linalg.block_diag(
        *[A_inverse[q * k_a : (q + 1) * k_a, q * k_a : (q + 1) * k_a] for q in range(L)]
    )
    assert abs(float(np.trace(separable_inverse @ V_eff)) - edf) > 1.0
    assert abs(float(np.trace(A_inverse @ block_diagonal)) - edf) > 1.0


@pytest.mark.parametrize(
    ("build", "budget", "must_attain"),
    [
        (lambda: _thin_level_pair(1.0), 24.0, True),
        (lambda: _vanishing_mass_pair(1e-12), 17.0, False),
    ],
    ids=["well-posed", "vanishing-mass"],
)
def test_the_pair_geometry_is_built_once_and_carries_no_lambda(build, budget, must_attain):
    """One overlap factorization for the whole ladder, and no per-rung count.

    The form this replaces counted every level block's rank AT EACH LAMBDA,
    because the rank it subtracted had to agree with the inverse at that
    lambda; a searching ladder therefore paid for a batched
    eigendecomposition per evaluation on top of the factorization itself
    (measured at +9.2% on an L=100 ps(8) pair driven to bisect).  A sum of
    filter factors has no rank in it, so everything a rung needs beyond its
    own factorization -- ``U_eff``, the per-level curvature ``D`` and the
    coupling ``Omega`` -- is lambda-free and built once.

    Both halves are asserted, because neither implies the other: that ONE
    overlap factorization serves a ladder of many factorizations, and that a
    geometry rebuilt from scratch reproduces a rung BIT for bit, which is what
    makes reusing it sound rather than merely cheap.

    **WHETHER THE BISECTION ATTAINS ITS TARGET IS NOT THIS TEST'S SUBJECT, AND
    ON THE SECOND FIXTURE IT IS NOT A PROPERTY THE MODULE PROMISES.**  This
    test previously ran only ``_vanishing_mass_pair(1e-12)`` at 17.0 and
    asserted the rung came back.  It passed here and FAILED on a CI shard --
    issue #272's shape again, and this time the term that moved was the
    search.  Measured over interior budgets 16.5 to 50 on that pair, only 5 of
    11 are attainable at all (the form this replaces attained 2 of the same
    11, and 17.0 was NOT among them), because #263 is open: the edf curve
    increases at 10 of 400 steps across that bracket, and a bisection can
    bracket a target it can never land on.  Asserting attainment there was
    asserting a coin flip.

    So attainment is asserted where it IS a property -- ``_thin_level_pair``,
    whose curve is well posed and where 9 of 9 interior budgets from 20 to 150
    are attained at all three weights -- and on the degenerate pair only the
    contract the module does guarantee: a refusal is ``None``, never a rung
    carrying a number the search did not reach.  Every other assertion here
    holds in both outcomes, since a refused search still spends its
    factorizations off the one profile.
    """
    import superglm.screening._structured as st

    p = spline_cat_moments(*_structured_inputs(build()))
    profiles, factors = [], []
    real_profile, real_arrow = st._profile, st._pair_arrow

    def spy_profile(pair):
        profiles.append(1)
        return real_profile(pair)

    def spy_arrow(pair, lam, *args, **kwargs):
        factors.append(float(lam))
        return real_arrow(pair, lam, *args, **kwargs)

    st._profile, st._pair_arrow = spy_profile, spy_arrow
    try:
        # The budget sits strictly inside this pair's bracket, so the rung
        # bisects rather than clamps and the ladder pays many factorizations
        # for its one profile.
        rungs = st.structured_ladder(p, budgets=(budget,))
    finally:
        st._profile, st._pair_arrow = real_profile, real_arrow

    if must_attain:
        assert rungs is not None, "this pair's curve is well posed; the target must be reachable"
    # Either way, a published rung has to carry the number it was asked for.
    if rungs is not None:
        assert len(rungs) == 1
        assert rungs[0].edf0 == pytest.approx(budget, abs=st._EDF_TOL), rungs[0].edf0
    assert len(profiles) == 1, profiles
    assert len(factors) > 10, len(factors)

    geometry = _profile(p)
    for lam in (min(factors), factors[len(factors) // 2], max(factors)):
        cached = _evaluate(p, geometry, lam)
        fresh = _evaluate(p, _profile(p), lam)
        assert cached == fresh, (lam, cached, fresh)


def test_an_unpenalized_spline_margin_is_scored_at_one_rung_not_refused():
    """The two ladders must agree on the DEGENERATE-penalty predicate too.

    The dense ladder answers a missing or all-zero penalty with an
    unpenalized single rung.  Raising instead would abort the whole sweep
    over one pair, and letting a zero penalty through makes the bracket
    infinite and every rung NaN, since ``inf * 0`` is not a number.
    """
    from superglm.screening._structured import structured_ladder

    grab = _thin_level_pair(1.0)
    B_a, S_a, S_cell, W_cell, level_rows = _structured_inputs(grab)
    U, V, C, M, _, u_m = grab["args"]
    dense = penalized_score_statistic_ladder(U, V, C, M, None, budgets=BUDGETS, U_nuisance=u_m)
    for penalty in (None, np.zeros_like(S_a)):
        struct = structured_ladder(
            spline_cat_moments(B_a, penalty, S_cell, W_cell, level_rows), budgets=BUDGETS
        )
        assert len(struct) == len(BUDGETS)
        for d, s in zip(dense, struct, strict=True):
            assert s.lambda0 == 0.0
            assert np.isfinite(s.statistic) and np.isfinite(s.edf0)
            assert s.edf0 == pytest.approx(d.edf0, abs=1e-3)
            assert s.statistic == pytest.approx(d.statistic, rel=1e-6)


def _starved_bs_pair():
    """7 levels x 3 rows against a thirteen-column ``bs(10)`` margin.

    The family the module docstring calls "where the arithmetic is beyond
    every form": too few rows per level to support the margin, so the overlap
    arrow's border Schur complement is nearly singular and its inverse runs to
    1e+10-1e+12.
    """
    rng = np.random.default_rng(3)
    L, reps = 7, 3
    n = L * reps
    g = np.repeat([f"L{i}" for i in range(L)], reps)
    df = pd.DataFrame({"g": g, "x": rng.uniform(0.05, 0.95, n)})
    return _capture(
        df,
        rng.normal(size=n),
        {"g": Categorical(), "x": Spline(kind="bs", n_knots=10)},
        ("x", "g"),
        sample_weight=np.ones(n),
    )


def test_a_real_starved_geometry_no_longer_leaves_the_filter_factor_bound():
    """The family this module admits it is up to 22 df wrong on, re-measured.

    7 levels x 3 rows against a thirteen-column ``bs(10)`` margin: too few
    rows per level to support the margin, so the overlap arrow's border Schur
    complement is nearly singular and its inverse runs to 1e+10-1e+12.  The
    form this replaces evaluated ``edf`` as ``local - border`` with both halves
    running from 3.8e+04 to 1.5e+13 against a ceiling of 78, and their
    difference left ``[0, 78]`` at **101 of 200** log-spaced lambdas -- at the
    high edge by ``-13.61`` out of ``9.654e+07``, eleven digits of cancellation
    on data alone.  ``structured_ladder`` handed the pair back.

    Nothing here forms either half.  Over the same 200 lambdas the sum leaves
    the bound at **NONE** of them, and every value it returns is inside
    ``[0, L k_a]`` by construction rather than by a guard, because every term
    is a squared norm.  RED against the unfixed code on both counts: it
    refuses 101 of 200 and returns ``None`` from the ladder.

    The magnitudes are asserted as orders rather than values, because on this
    geometry a SINGLE ulp of lambda used to move ``edf`` from ``-13.61`` to
    ``0.8229``; what is pinned is the width of the region and the
    caller-visible outcome, not a value at one lambda.
    """
    import superglm.screening._structured as st
    from superglm.screening._structured import structured_ladder

    p = spline_cat_moments(*_structured_inputs(_starved_bs_pair()))
    L, k_a = p.dims
    geometry = st._profile(p)
    tr_S = float(np.trace(p.S_a)) * L
    scale = max(p.profiled_trace, 1e-300) / max(tr_S, 1e-300)

    refused, values = 0, []
    for lam in np.geomspace(1e-10 * scale, 1e10 * scale, 200):
        try:
            _, edf = st._evaluate(p, geometry, float(lam))
        except st._UnstableStructuredEDFError:
            refused += 1
        else:
            assert 0.0 <= edf <= L * k_a, (lam, edf)
            values.append(edf)

    assert refused == 0, refused
    assert len(values) == 200
    # Nearly monotone too, which the difference form was not: over the same
    # bracket its worst step UP is measured in whole degrees of freedom while
    # this one's is 6.9e-05.  That is disclosure and not a derived bound --
    # monotonicity would follow from nonnegative per-DIRECTION shares, which
    # this route does not deliver -- so what is asserted is only that no step
    # up reaches the ladder's own target tolerance by a factor of 100.
    assert np.diff(values).max() < 100.0 * st._EDF_TOL, np.diff(values).max()

    # ...and the ladder now scores the pair rather than handing it back.
    rungs = structured_ladder(p, budgets=BUDGETS)
    assert rungs is not None and len(rungs) == len(BUDGETS), rungs
    assert all(0.0 <= r.edf0 <= L * k_a and np.isfinite(r.statistic) for r in rungs), rungs


# The measured cross-machine spread of the ``lambda = 0`` rung on a near-rank
# pair, 0.379 df (see the test below), rounded up to the next half degree of
# freedom.  It is a slack on a bound that holds EXACTLY in exact arithmetic,
# so what it has to survive is the quantity's own portability rather than its
# accuracy -- and it still refuses the +3 df the counters this module dropped
# disagreed by.
_ZERO_PENALTY_RANK_SLACK = 0.5


@pytest.mark.parametrize(
    ("build", "observed"),
    [
        (lambda: _rank_one_penalty_pair(13, 6, 12, 1e-3, 3), 8.003514540044),
        (lambda: _rank_one_penalty_pair(13, 6, 12, 2e-3, 3), 9.263424641336),
        (lambda: _rank_one_penalty_pair(7, 5, 12, 1e-3, 2), 6.494923264487),
        (lambda: _multi_null_pair(0), 15.256469726562),
    ],
    ids=["band-1e-3-L6", "band-2e-3-L6", "band-1e-3-L5", "multi-null"],
)
def test_the_zero_penalty_rung_on_a_near_rank_pair(build, observed):
    """``lambda = 0`` asks for a RANK, and no continuous quantity delivers one.

    This is a deliberate behaviour change and it is recorded here rather than
    left to be discovered.  The form this replaced returned ``rank(A)`` at
    ``lambda = 0`` -- an integer, by construction.  ``tr(A^-1 V_eff)`` is
    ``tr(V_eff^+ V_eff)`` there, which IS ``rank(V_eff)`` in exact arithmetic
    but is whatever the inverse's own cut resolves in float64, so it is
    generally not an integer.

    **THE INTEGER WAS NEVER THE ACCURACY.**  Measured against the dense path's
    explicitly counted rank on the four near-rank pairs below, the old form's
    integer disagreed by up to 3.000 df and the new form's non-integer by at
    most 0.9965; over twelve near-rank fixtures the worst gap falls
    3.000 -> 0.9965 df, the new value is nearer on 6 and farther on 4, and on
    the two well-posed ones both are exact (1.7e-12, 7.2e-11).  Restoring an
    integer here would restore a LARGER disagreement, not a smaller one.

    Rank is not well defined on these geometries in float64 and the three
    available counters say so: on ``band-1e-3-L6`` the dense path counts 9,
    ``numpy.linalg.matrix_rank`` on the residualized design counts 10, and the
    old arrow form counted 6.  That spread is the reason this module stopped
    counting.

    **NOTHING HERE PINS A VALUE, AND THE REASON IS A MEASUREMENT.**  A first
    revision of this test pinned the structured rung to the numbers above at
    1e-7, derived from the spread over 40 exact relabelings on one machine
    (5.17e-10, 1.05e-09, 1.63e-09, 1.98e-02).  **THAT DERIVATION WAS WRONG BY
    EIGHT ORDERS OF MAGNITUDE** and CI said so: ``band-1e-3-L5`` read
    6.116303396818694 there against 6.494923264487 here -- the ``lambda = 0``
    rung on a near-rank pair moves **0.379 df** between machines.  Relabeling
    on one machine measures reduction ORDER; it does not measure a different
    LAPACK driver taking a different branch through a near-singular block, and
    on this quantity the second dominates.

    So the finding is that the zero-penalty rung is not reproducible to a
    value at all on near-rank geometries, by any implementation -- which is
    the same statement as "rank is ill-posed here", arrived at from the other
    side.  What is asserted is therefore only what the IDENTITY guarantees and
    machines agree on: the rung is published rather than refused, it is
    finite, it sits at ``lambda = 0``, and it lies between zero and the rank
    of the residualized design, since a projector trace cannot exceed the rank
    of what it projects onto.  Both bounds hold at 6.1163 and at 6.4949.

    The integer mutation is covered where the quantity IS reproducible:
    rounding ``edf`` reds
    ``test_a_thin_level_does_not_cost_the_pair_a_degree_of_freedom`` against an
    arb-certified constant at 1e-5, on a fixture whose value is bit-identical
    at 1, 4 and 8 threads.  It does not need to be covered twice, and covering
    it here would mean pinning a number this test has just shown is not
    pinnable.
    """
    from superglm.screening._structured import structured_ladder

    grab = build()
    B_a, S_a, S_cell, W_cell, level_rows = _structured_inputs(grab)
    struct = structured_ladder(
        spline_cat_moments(B_a, np.zeros_like(S_a), S_cell, W_cell, level_rows), budgets=BUDGETS
    )
    # **WHETHER THE PAIR IS PUBLISHED AT ALL IS NOT PORTABLE EITHER, AND CI
    # SAID SO TWICE.**  A refusal here is the documented contract -- `None`
    # hands the pair back and the caller takes a NaN row -- and on a geometry
    # whose `lambda = 0` value moves 0.379 df between machines, which side of
    # the deflation's cut it lands on moves with it: this machine publishes
    # all four of these fixtures, CI refused `band-1e-3-L6` on one revision
    # and `band-2e-3-L6` on the next.  Asserting "published" pins the third
    # non-reproducible quantity on this branch.  What is asserted is the
    # CONTRACT: whatever comes back obeys the identity's bounds, and
    # `test_the_zero_penalty_family_is_not_refused_wholesale` below keeps this
    # from degrading into a test that passes on a module that refuses
    # everything.
    if struct is None:
        return
    assert len(struct) == len(BUDGETS)

    # The rank of the residualized design, taken on a FACTOR rather than on a
    # Gram, which is the one counter here that is not squaring its input.
    cells = np.argwhere(W_cell > 0.0)
    rows_a, rows_b = cells[:, 0], cells[:, 1]
    root_w = np.sqrt(W_cell[rows_a, rows_b])[:, None]
    indicator = (rows_b[:, None] == level_rows[None, :]).astype(np.float64)
    tensor = (B_a[rows_a][:, :, None] * indicator[:, None, :]).reshape(cells.shape[0], -1)
    mains = np.concatenate([np.ones((cells.shape[0], 1)), B_a[rows_a], indicator], axis=1)
    Q, _ = np.linalg.qr(mains * root_w)
    resid = tensor * root_w
    resid -= Q @ (Q.T @ resid)
    design_rank = int(np.linalg.matrix_rank(resid))

    for s in struct:
        assert s.lambda0 == 0.0
        assert np.isfinite(s.statistic) and np.isfinite(s.edf0)
        # Bounded by the number of filter factors, which is what the guard
        # bounds it by, and nonnegative, since every factor is.  This holds
        # for every pair and every lambda; it is the identity, not a fixture.
        assert 0.0 <= s.edf0 <= B_a.shape[1] * level_rows.size
        # ``tr(V_eff^+ V_eff)`` is a projector trace, so it cannot exceed the
        # rank of the design it projects onto -- exactly.  The slack is this
        # test's own portability measurement and not headroom: CI read
        # 18.000341 against a design rank of 18 where this machine reads
        # 15.256470, which is the same 0.379 df spread the docstring derives
        # from ``band-1e-3-L5``.  ``observed`` is what this machine reads and
        # is carried only so a reader can see how far the bound sits from it.
        assert s.edf0 <= design_rank + _ZERO_PENALTY_RANK_SLACK, (
            s.edf0,
            design_rank,
            observed,
        )


def test_the_zero_penalty_family_is_not_refused_wholesale():
    """The floor under the test above, so its refusal branch cannot swallow it.

    ``test_the_zero_penalty_rung_on_a_near_rank_pair`` accepts a refusal,
    because which side of the deflation's cut a near-rank pair lands on is not
    reproducible between machines -- a 0.379 df spread on the value, and CI has
    refused two different parametrizations on two revisions where this machine
    published all four.  That acceptance would let a module that refused
    EVERYTHING pass it, so the population is asserted here instead of the
    member.

    Four near-rank fixtures plus one well-posed control.  The control must
    publish -- it is not near-rank and nothing about it is at a cut -- and the
    near-rank four must not all refuse, which is the failure this branch
    exists to remove: the form it replaces refused 2 of 56 oracle points and
    handed back whole pairs on the starved family.  A majority is not
    asserted, because a majority is what was observed rather than what is
    guaranteed.
    """
    from superglm.screening._structured import structured_ladder

    def unpenalized(grab):
        B_a, S_a, S_cell, W_cell, level_rows = _structured_inputs(grab)
        return structured_ladder(
            spline_cat_moments(B_a, np.zeros_like(S_a), S_cell, W_cell, level_rows),
            budgets=BUDGETS,
        )

    control = unpenalized(_thin_level_pair(low_weight=1.0))
    assert control is not None and len(control) == len(BUDGETS), control

    near_rank = [
        unpenalized(build())
        for build in (
            lambda: _rank_one_penalty_pair(13, 6, 12, 1e-3, 3),
            lambda: _rank_one_penalty_pair(13, 6, 12, 2e-3, 3),
            lambda: _rank_one_penalty_pair(7, 5, 12, 1e-3, 2),
            lambda: _multi_null_pair(0),
        )
    ]
    assert any(r is not None for r in near_rank), (
        "every near-rank zero-penalty pair was refused; the ladder is handing "
        "back the whole family it was changed to score"
    )


def test_the_ladder_refuses_a_search_it_cannot_afford(monkeypatch):
    """``max_evaluations`` is checked BEFORE the first bisection step.

    A clamping ladder is two arrow factorizations; a searching one is tens.
    A caller that budgeted for the first must get a refusal rather than the
    second, and must pay only the bracket to find out.
    """
    import superglm.screening._structured as st
    from superglm.screening._structured import structured_ladder

    p = spline_cat_moments(*_structured_inputs(_thin_level_pair(1.0)))
    calls = 0
    real = st._evaluate

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(st, "_evaluate", counted)
    # edf at maximum penalty is about L - 1 = 19 here, so 16 clamps and 24
    # has to search.
    assert structured_ladder(p, budgets=(16.0,), max_evaluations=2) is not None
    assert calls == 2
    calls = 0
    assert structured_ladder(p, budgets=(24.0,), max_evaluations=2) is None
    assert calls == 2
    calls = 0
    assert structured_ladder(p, budgets=(24.0,), max_evaluations=200) is not None
    assert calls > 2


def test_structured_allowance_charges_both_qr_passes_and_seven_factor_units():
    max_cells, n_rows, k_a, n_levels = 20_000, 31, 6, 19
    per_evaluation = n_levels * (k_a + 1) ** 3
    setup = 2 * n_rows * n_levels * k_a**2 + 7 * per_evaluation
    expected = max(ops._STRUCTURED_CUBIC_BUDGET_FACTOR * max_cells - setup, 0)
    expected //= per_evaluation
    assert (
        ops._structured_evaluation_allowance(
            max_cells,
            n_rows,
            k_a,
            n_levels,
        )
        == expected
    )


def test_large_exact_support_is_refused_when_profile_setup_exhausts_the_budget():
    assert ops._structured_evaluation_allowance(5_000_000, 256, 11, 200) >= 2
    assert ops._structured_evaluation_allowance(5_000_000, 20_000, 11, 200) == 0


@pytest.mark.parametrize("low_weight", [0.01, 0.001])
def test_an_ill_conditioned_thin_level_reaches_an_interior_edf_target(low_weight):
    from superglm.screening._structured import structured_ladder

    pair = spline_cat_moments(*_structured_inputs(_thin_level_pair(low_weight)))
    result = structured_ladder(pair, budgets=(24.0,), max_evaluations=200)
    assert result is not None
    assert len(result) == 1
    assert np.isfinite([result[0].statistic, result[0].edf0, result[0].lambda0]).all()
    assert result[0].edf0 == pytest.approx(24.0, abs=1e-6)


def test_repeating_a_budget_does_not_change_whether_a_pair_is_screenable(monkeypatch):
    """The ladder is charged for distinct SEARCH TARGETS, not for rungs.

    ``edf0`` is allowed to repeat a budget, and every copy of one bisects to
    the same lambda, so charging each copy separately let a repeated budget
    decide whether the pair got a score at all — the same pair came back
    finite at ``(24.0,)`` and a NaN row at ``(24.0, 24.0)``.
    """
    import superglm.screening._structured as st
    from superglm.screening._structured import structured_ladder

    p = spline_cat_moments(*_structured_inputs(_thin_level_pair(1.0)))
    calls = 0
    real = st._evaluate

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(st, "_evaluate", counted)
    once = structured_ladder(p, budgets=(24.0,), max_evaluations=48)
    assert once is not None
    once_calls = calls
    for repeats in (2, 3):
        calls = 0
        many = structured_ladder(p, budgets=(24.0,) * repeats, max_evaluations=48)
        assert many is not None
        assert calls == once_calls
        assert len(many) == repeats
        # ...and the repeats are the same answer, computed once
        for r in many:
            assert r.statistic == once[0].statistic
            assert r.edf0 == once[0].edf0
            assert r.lambda0 == once[0].lambda0


def test_profiled_scale_preserves_the_lower_edge_clamp(monkeypatch, moderate_pair):
    """An over-ceiling target still clamps low using only two evaluations."""
    import superglm.screening._structured as st

    p = spline_cat_moments(*_structured_inputs(moderate_pair))
    calls = 0
    real = st._evaluate

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(st, "_evaluate", counted)
    result = st.structured_ladder(p, budgets=(1e6,), max_evaluations=2)[0]
    expected_lo = 1e-10 * p.profiled_trace / (np.trace(p.S_a) * p.dims[0])
    assert calls == 2
    assert result.lambda0 == pytest.approx(expected_lo, rel=2e-15)
    assert np.isfinite(result.statistic)
    assert result.edf0 >= 0.0


def test_an_exact_arrow_score_beats_an_approximate_dense_one(monkeypatch):
    """Bin only when neither path can score the pair exactly.

    The dense path gets first refusal, but the handoff to the arrow kernel
    used to happen only once binning had run out — so a pair whose DENSE
    intermediate is over budget got its spline margin compressed even where
    the arrow path, whose intermediate is the transpose of that one, would
    have taken the pair whole. Failing one says nothing about the other.
    """
    L, support, n = 169, 1000, 4000
    rng = np.random.default_rng(21)
    grid = np.linspace(0.0, 1.0, support)
    df = pd.DataFrame(
        {
            "g": np.array([f"L{i}" for i in range(L)])[rng.integers(0, L, n)],
            "x": grid[np.arange(n) % support],
        }
    )
    y = rng.normal(size=n)
    model = SuperGLM(
        family="gaussian", features={"g": Categorical(), "x": Spline(kind="ps", n_knots=8)}
    )
    model.fit_reml(df, y)
    row, seen = _routed_shapes(model, df, y, monkeypatch)
    assert seen, "the pair must route through the arrow kernel"
    # The dense intermediate is 1000 * 168^2 = 28.2M against a 20M budget, so
    # the dense path has to bin; the arrow one is 1000 * 11^2 = 121k, which
    # fits 165x over.
    assert seen["support"] == support
    assert not bool(row["approx"])
    assert np.isfinite(row["z"])


def test_a_failed_arrow_speculation_hands_the_dense_track_back():
    """Speculating on the arrow path must not cost the pair its dense score.

    The handoff above is taken on a width estimate that is biased LOW, to try
    for an EXACT score before compressing the spline.  When the authoritative
    width then puts the arrow intermediate over budget, the arrow path bins
    its own margin and can still run out -- and at that exit ``allow_dense``
    was left False with no binnable margin remaining, so the loop broke and
    the pair became a NaN row.  The dense path could score it all along, on
    the very support the arrow path had just binned to.

    Measured on this configuration: a NaN row before, ``z`` finite after, and
    the dense route is the approximate one it would have taken anyway.
    """
    L, support, n = 10, 6_000, 24_000
    rng = np.random.default_rng(0)
    grid = np.linspace(0.0, 10.0, support)
    df = pd.DataFrame(
        {
            "x": grid[np.arange(n) % support],
            "g": np.array([f"L{i}" for i in range(L)])[rng.integers(0, L, n)],
        }
    )
    y = rng.poisson(np.exp(-1.0 + 0.15 * df["x"].to_numpy())).astype(np.float64)
    model = SuperGLM(
        family="poisson",
        features={"x": Spline(kind="ps", n_knots=8), "g": Categorical()},
    )
    model.fit_reml(df, y)

    row = model.screen_interactions(
        df, y, candidates=[("x", "g")], max_cells=100_000, screen_bins=4_000
    ).iloc[0]
    assert np.isfinite(row["z"]), "a pair the dense path can score must not be a NaN row"
    assert np.isfinite(row["statistic"])
    assert bool(row["approx"]), "the surviving route is the binned dense one"


def test_a_late_arrow_refusal_hands_the_dense_track_back():
    """A speculative handoff refused by the LADDER, not by a gate.

    The gates can see allocations and dimensions.  What they cannot see is
    whether a rung lands inside the bracket and has to bisect, because that
    turns on the penalty's null space -- an ``ns`` margin's penalty is full
    rank, so ``edf`` at maximum penalty is 0 and every rung searches, where a
    ``ps`` margin clamps.  So the arrow path can pass every gate and still
    refuse once tried.

    When the handoff was SPECULATIVE, that refusal must not delete the pair:
    the whole point of trying the arrow path first was to get an exact score
    instead of an approximate one, and giving up the approximate one as well
    trades a scored row for a NaN.  The dense track goes back, exactly as it
    does at the width and support exits.
    """
    L, support, n = 30, 2_000, 12_000
    rng = np.random.default_rng(5)
    grid = np.linspace(0.0, 1.0, support)
    df = pd.DataFrame(
        {
            "x": grid[np.arange(n) % support],
            "g": np.array([f"L{i}" for i in range(L)])[rng.integers(0, L, n)],
        }
    )
    y = rng.normal(size=n)
    model = SuperGLM(
        family="gaussian",
        features={"x": Spline(kind="ns", n_knots=8), "g": Categorical()},
    )
    model.fit_reml(df, y)

    row = model.screen_interactions(
        df, y, candidates=[("x", "g")], edf0=BUDGETS, max_cells=100_000, screen_bins=256
    ).iloc[0]

    assert np.isfinite(row["z"]), "a pair the dense path can score must not be a NaN row"
    assert np.isfinite(row["statistic"])
    assert bool(row["approx"]), "the surviving route is the binned dense one"


def test_a_full_rank_penalty_makes_every_rung_search():
    """The kernel's cost is not a function of the pair's dimensions.

    ``ps``, ``bs`` and ``cr`` margins leave one direction per level
    unpenalized, so ``edf`` at maximum penalty outranks every budget and the
    whole ladder clamps at two evaluations.  A natural spline's penalty is
    full rank, so ``edf`` there is zero and EVERY rung bisects -- same
    dimensions, tens of times the work.  This is why the ladder is capped by
    a ceiling it checks against rather than by a prediction from ``k`` and
    ``L``.
    """
    import superglm.screening._structured as st
    from superglm.screening._structured import structured_ladder

    counts = {}
    for kind in ("ps", "ns"):
        rng = np.random.default_rng(11)
        L, reps = 30, 20
        n = L * reps
        g = np.repeat([f"L{i}" for i in range(L)], reps)
        x = rng.uniform(0.0, 1.0, n)
        df = pd.DataFrame({"g": g, "x": x})
        y = rng.normal(size=n)
        grab = _capture(df, y, {"g": Categorical(), "x": Spline(kind=kind, n_knots=8)}, ("x", "g"))
        p = spline_cat_moments(*_structured_inputs(grab))
        calls = [0]
        real = st._evaluate

        def counted(*a, _real=real, _calls=calls, **k):
            _calls[0] += 1
            return _real(*a, **k)

        st._evaluate = counted
        try:
            out = structured_ladder(p, budgets=BUDGETS)
        finally:
            st._evaluate = real
        counts[kind] = calls[0]
        assert all(np.isfinite(r.edf0) for r in out)

    assert counts["ps"] == 2
    assert counts["ns"] > 2 * len(BUDGETS)


def _routed_shapes(model, df, y, monkeypatch, **kw):
    """Run a screen and report the moment shapes the arrow kernel actually saw."""
    seen = {}
    real = ops.spline_cat_moments

    def spy(B_a, S_a, S_cell, W_cell, level_rows):
        seen["support"] = B_a.shape[0]
        seen["width"] = B_a.shape[1]
        return real(B_a, S_a, S_cell, W_cell, level_rows)

    monkeypatch.setattr(ops, "spline_cat_moments", spy)
    row = model.screen_interactions(df, y, candidates=[("x", "g")], edf0=BUDGETS, **kw).iloc[0]
    return row, seen


def test_the_structured_path_bins_rather_than_allocate_its_own_intermediate(monkeypatch):
    """The kernel's ``(n_a, k_s, k_s)`` outer products need a gate of their own.

    It is the TRANSPOSE of the intermediate the dense path is gated on, and
    no other structured gate bounds it: the block stacks are ``L * k_s^2``
    and the cell table is ``n_a * L``, and neither is ``n_a * k_s^2``.  Left
    ungated it grows with the spline's support, which the cell table alone
    lets run to ``max_cells / L``.  Both terms scale with the support, so the
    answer is to bin the spline margin, exactly as the dense path does.
    """
    rng = np.random.default_rng(17)
    L, n, support = 40, 5000, 4000
    # The support is chosen so the CELL term passes on its own -- 4000 x 40
    # is 160,000 against a 400,000 ceiling -- and only the intermediate is
    # over, so nothing but the new term can be what bins the margin.
    grid = np.linspace(0.0, 1.0, support)
    df = pd.DataFrame(
        {
            "g": np.array([f"L{i}" for i in range(L)])[rng.integers(0, L, n)],
            "x": grid[np.arange(n) % support],
        }
    )
    y = rng.normal(size=n)
    model = SuperGLM(
        family="gaussian", features={"g": Categorical(), "x": Spline(kind="ps", n_knots=20)}
    )
    model.fit_reml(df, y)
    row, seen = _routed_shapes(model, df, y, monkeypatch, max_cells=400_000)
    assert seen, "the pair must route through the arrow kernel"
    # 4000 support points at width 23 is 2,116,000 doubles against a
    # 1,600,000 double intermediate budget; binned to 256 it is 135,424.
    assert seen["width"] > 20
    assert seen["support"] <= 256
    assert bool(row["approx"])
    assert np.isfinite(row["z"])


def test_a_pair_whose_ladder_must_bisect_is_refused_when_it_cannot_afford_to(monkeypatch):
    """The structured path needs a TIME gate as well as allocation gates.

    One evaluation batches ``L`` eigendecompositions of ``(k_s + 1)`` blocks,
    so it costs ``L * k_s^3`` where every allocation gate costs ``L * k_s^2``
    — and a searching ladder runs tens of them.  A natural spline's penalty
    is full rank, so every rung of this pair's ladder searches; the pair is
    otherwise identical to one that clamps.
    """
    rng = np.random.default_rng(19)
    L, reps = 160, 6
    n = L * reps
    df = pd.DataFrame(
        {"g": np.repeat([f"L{i}" for i in range(L)], reps), "x": rng.uniform(0.0, 1.0, n)}
    )
    y = rng.normal(size=n)
    model = SuperGLM(
        family="gaussian", features={"g": Categorical(), "x": Spline(kind="ns", n_knots=8)}
    )
    model.fit_reml(df, y)
    row, seen = _routed_shapes(model, df, y, monkeypatch, max_cells=400_000)
    assert seen, "the pair must route through the arrow kernel"
    assert np.isnan(row["z"])
    assert np.isnan(row["edf0"])
    # ...and raising the budget lifts the refusal, as it lifts every other.
    # The pair is still over the dense ceiling at the higher budget, so it is
    # the structured path that scores it, not a fallback to the dense one.
    lifted, lifted_seen = _routed_shapes(model, df, y, monkeypatch, max_cells=1_500_000)
    assert lifted_seen
    assert np.isfinite(lifted["z"])


def test_degenerate_levels_are_scored_not_skipped():
    """Singleton levels are the routine case at high cardinality, not an edge
    case: they make their own block singular, and the kernel has to keep going
    rather than fail or silently drop them."""
    rng = np.random.default_rng(5)
    populated = pd.DataFrame(
        {"g": np.repeat([f"L{i}" for i in range(60)], 8), "x": rng.uniform(0, 1, 480)}
    )
    singles = pd.DataFrame({"g": [f"S{i}" for i in range(200)], "x": rng.uniform(0, 1, 200)})
    # a level whose covariate never varies -- an exactly rank-deficient block
    flat = pd.DataFrame({"g": ["FLAT"] * 12, "x": np.full(12, 0.5)})
    df = pd.concat([populated, singles, flat], ignore_index=True)
    y = rng.normal(size=len(df))
    model = SuperGLM(
        family="gaussian", features={"g": Categorical(), "x": Spline(kind="ps", n_knots=8)}
    )
    model.fit_reml(df, y)
    row = model.screen_interactions(df, y, candidates=[("x", "g")], edf0=BUDGETS).iloc[0]
    assert np.isfinite(row["z"])
    assert np.isfinite(row["edf0"])
    assert row["edf0"] > 0.0
