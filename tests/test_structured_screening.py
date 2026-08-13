"""The arrow kernel that lifts spline_cat's level ceiling.

The dense path densifies a block that is structurally block-diagonal, which
costs a cubic solve and quadratic memory in the level count and caps the
factors screening will look at.  These tests pin the two things that make the
structured path a safe replacement above that cap: it computes the same
quantity as the dense path, and it never allocates what the dense path was
refused for.
"""

from __future__ import annotations

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

    This test reds against the form it replaces for a structural reason: that
    one reached the guard through a ``FakeFactor`` standing in for
    ``_pair_arrow``, and ``_pair_arrow`` no longer computes ``edf``.
    """
    import superglm.screening._structured as st

    pair = spline_cat_moments(*_near_absorbed_cells(0.0))
    geometry = st._profile(pair)
    assert geometry.ceiling == 1.0, geometry.ceiling

    for injected, expected in ((1e-300, 1e-300), (-1e-300, 0.0), (1.0, 1.0)):
        monkeypatch.setattr(st, "_filter_factor_sum", lambda *a, _v=injected, **k: _v)
        assert st._evaluate(pair, geometry, 1.0)[1] == pytest.approx(expected, abs=1e-12)

    for outside in (1.25, -1.25):
        monkeypatch.setattr(st, "_filter_factor_sum", lambda *a, _v=outside, **k: _v)
        with pytest.raises(st._UnstableStructuredEDFError, match="not a filter-factor sum"):
            st._evaluate(pair, geometry, 1.0)

    # ...and the real thing never goes negative, on the fixture built to make
    # the lower end as reachable as a pair can make it.
    monkeypatch.undo()
    for lam in np.geomspace(1e-30, 1e30, 61):
        assert st._filter_factor_sum(pair, geometry, float(lam)) >= 0.0


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
_CERTIFIED_LOW_EDGE_EDF = 207.99995542698146


def _reference_edf(grab, lam):
    """``edf(lambda)`` from the pair's DESIGN rather than from its Gram.

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

        this form                 1.99e-13
        the moment-space form     1.13e-05
        structured path           6.97e-09
        dense path                1.37e-10

    and it is not a property of one arrangement: over 60 symmetric
    relabelings of the tensor and overlap coordinates, which leave ``edf``
    exactly unchanged in real arithmetic and change only the order of every
    reduction, the moment-space form spreads 1.4721e-05 and this one 7.39e-13.
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
    B_a, S_a, _, W_cell, level_rows = _structured_inputs(grab)
    k_a, k_b = B_a.shape[1], level_rows.size
    cells = np.argwhere(W_cell > 0.0)
    rows_a, rows_b = cells[:, 0], cells[:, 1]
    root_w = np.sqrt(W_cell[rows_a, rows_b])[:, None]
    indicator = (rows_b[:, None] == level_rows[None, :]).astype(np.float64)
    tensor = (B_a[rows_a][:, :, None] * indicator[:, None, :]).reshape(cells.shape[0], k_a * k_b)
    mains = np.concatenate([np.ones((cells.shape[0], 1)), B_a[rows_a], indicator], axis=1)
    # Pivoted QR of the overlap span, so a rank-deficient set of mains
    # residualizes rather than raises -- the one rank decision in here, taken
    # on a FACTOR and never on a Gram.
    Q, R_mains, _ = scipy.linalg.qr(mains * root_w, mode="economic", pivoting=True)
    tol = max(mains.shape) * np.finfo(np.float64).eps * abs(R_mains[0, 0])
    Q = Q[:, : int(np.sum(np.abs(np.diag(R_mains)) > tol))]
    resid = tensor * root_w
    resid -= Q @ (Q.T @ resid)
    ev, evec = np.linalg.eigh(0.5 * (S_a + S_a.T))
    root_S = np.kron((evec * np.sqrt(np.clip(ev, 0.0, None))) @ evec.T, np.eye(k_b))
    T = np.linalg.qr(np.vstack([resid, np.sqrt(lam) * root_S]), mode="r")
    return float(np.sum(scipy.linalg.solve_triangular(T, resid.T, trans="T", lower=False) ** 2))


def test_the_low_edge_reference_matches_a_certified_high_precision_value():
    """The oracle is pinned to arbitrary precision, not to the other arm.

    Everything else in this file compares two float64 arms, which cannot tell
    a wrong reference from a wrong path.  Issue #272 was exactly that: the
    low-edge assertion failed on a developer machine and passed in CI on the
    same locked dependencies, and the term that moved was neither path.  Both
    of them sit 6.97e-09 and 1.37e-10 from the certified value; the reference
    sat 1.13e-05 away and moved by 1.47e-05 across arrangements of the same
    algebra.

    Two properties, and the second is what makes the first portable:

    * the value itself, against 400-bit ball arithmetic.  1e-9 is 500x above
      the worst error this form was measured at over five fixtures
      (1.93e-12), 1350x above its spread over exact relabelings (7.39e-13),
      and 7000x above what a one-ulp perturbation of the basis moves it
      (1.42e-13, worst of 20 draws; ``S_a`` 5.68e-14; ``lambda`` 0.00e+00).
    * invariance under an exact relabeling.  A reference that is only right
      in the coordinate order it happened to be handed is not a reference,
      and that is the property the moment-space form lacked.
    """
    grab = _thin_level_pair(1.0)
    B_a, _, _, W_cell, level_rows = _structured_inputs(grab)
    # Pin the fixture the constant was certified for, so a change to it fails
    # as itself rather than as an unexplained numeric miss.
    assert (B_a.shape[1], level_rows.size) == (11, 19)
    assert float(W_cell.sum()) == 800.0

    got = _reference_edf(grab, _CERTIFIED_LOW_EDGE_LAMBDA)
    assert got == pytest.approx(_CERTIFIED_LOW_EDGE_EDF, abs=1e-9), got

    menu_a, menu_b = grab["menus"]
    rng = np.random.default_rng(11)
    for _ in range(8):
        relabeled = {**grab, "menus": (menu_a, menu_b[:, rng.permutation(level_rows.size)])}
        assert _reference_edf(relabeled, _CERTIFIED_LOW_EDGE_LAMBDA) == pytest.approx(got, abs=1e-9)


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
      Stability of Numerical Algorithms*, 2nd ed., Thm 2.5).  Below the
      relabeling floor, so the relabeling floor binds.
    * CEILING.  One degree of freedom.  The form this replaced returned an
      integer rank minus a trace, and a mis-counted direction moved the answer
      by exactly 1.0 df; that is the size of the defect, not a guess.
    * PLACED.  1e-5 is 3613x above the floor and 100,000x below the defect --
      5.26x below the geometric mean of the two, ``5.261e-05``, which is the
      placement that maximizes the multiplicative margin on both sides.  The
      bound is the tighter of the two choices, not the more comfortable one.
    * OBSERVED, for disclosure only, never to set the bound: structured
      3.965e-11 / 3.336e-08 / 1.288e-07 and dense 1.366e-10 / 2.404e-09 /
      3.226e-09 across the three weights.

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
    U, V, C, M, S_ti, u_m = grab["args"]
    dense = penalized_score_statistic_ladder(
        U, V, C, M, S_ti, budgets=_EDGE_BUDGETS, U_nuisance=u_m
    )
    struct = structured_ladder(spline_cat_moments(*_structured_inputs(grab)), budgets=_EDGE_BUDGETS)
    lam_lo, edf_lo, lam_hi, edf_hi = _CERTIFIED_EDGES[low_weight]
    saw_low_edge = saw_high_edge = False
    for budget, d, s in zip(_EDGE_BUDGETS, dense, struct, strict=True):
        # Each path is judged at its reported lambda.  Both brackets now use
        # tr(V_eff), though their endpoint solves remain independent and can
        # differ at the ill-conditioned high edge.
        assert s.lambda0 == pytest.approx(d.lambda0, rel=1e-12), ("one lambda", budget)
        if budget > 100.0:
            # LOW edge -- the regime that matters, since the whole-degree-of-
            # freedom error this test exists for lived here, at 1.0 against the
            # reference.  ``_reference_edf`` is checked against the certified
            # constant FIRST: it is the arbiter, so an unarbitrated arbiter
            # would let both arms be judged by a wrong number.  1e-9 is PR
            # #275's bound, extended from the one weight it certified to all
            # three; worst observed here 1.99e-13, 5000x inside it.
            assert s.lambda0 == pytest.approx(lam_lo, rel=1e-12), ("lambda_lo", s.lambda0)
            reference = _reference_edf(grab, s.lambda0)
            assert reference == pytest.approx(edf_lo, abs=1e-9), ("oracle", reference)
            assert s.edf0 == pytest.approx(reference, abs=1e-5), ("structured", budget, s.edf0)
            assert d.edf0 == pytest.approx(reference, abs=1e-5), ("dense", budget, d.edf0)
            saw_low_edge = True
        else:
            # HIGH edge.  No float64 oracle survives here (``_reference_edf``
            # is 8.15e-05 to 2.11e-03 out), so both arms are judged against the
            # certified constant directly.  This edge used to be parity-only,
            # which cannot see the two arms drifting together.
            assert s.lambda0 == pytest.approx(lam_hi, rel=1e-12), ("lambda_hi", s.lambda0)
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

    THE DISTANCE FROM ``_reference_edf`` WAS RE-MEASURED WHEN THAT REFERENCE
    WAS REPLACED (issue #272).  It used to read a median of 1.6e-05 and a
    worst of 1.036e-04; most of that median was the old moment-space
    reference's own round-off, which this fixture's near-zero-mass levels
    make worse -- on ``low_weight = 1e-12`` it was 4.20e-05 out against the
    certified value, where the structured path was 3.04e-09 out.  Against the
    factored reference, over 160 draws (80 seeds x 2 weights, one thread),
    the structured value's distance has a median of 7.0757e-10 and a worst of
    8.2123e-05, inside the 3e-4 asserted below on every one of them with 3.7x
    to spare.  Parity over the same window reproduces the 320-draw figure
    above to four digits (worst 9.0208e-05), which is the check that this
    narrower window is measuring the same family.
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
            assert s.edf0 == pytest.approx(left_free, abs=1e-3), (
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
        else:  # the LOW edge, where the reference oracle is sound
            reference = _reference_edf(grab, s.lambda0)
            assert s.edf0 == pytest.approx(reference, abs=3e-4), (
                "low edge oracle",
                s.edf0,
                reference,
            )
            assert s.edf0 == pytest.approx(d.edf0, abs=2e-4), ("low edge parity", s.edf0, d.edf0)
        assert s.lambda0 == pytest.approx(d.lambda0, rel=1e-12), ("lambda0", budget)
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

    def spy_arrow(pair, lam):
        factors.append(float(lam))
        return real_arrow(pair, lam)

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


@pytest.mark.parametrize(
    ("build", "expected", "tol"),
    [
        (lambda: _rank_one_penalty_pair(13, 6, 12, 1e-3, 3), 8.003514540044, 1e-9),
        (lambda: _rank_one_penalty_pair(13, 6, 12, 2e-3, 3), 9.263424641336, 1e-9),
        (lambda: _rank_one_penalty_pair(7, 5, 12, 1e-3, 2), 6.494923264487, 1e-9),
        (lambda: _multi_null_pair(0), 15.256469726562, 1e-9),
    ],
    ids=["band-1e-3-L6", "band-2e-3-L6", "band-1e-3-L5", "multi-null"],
)
def test_the_zero_penalty_rung_on_a_near_rank_pair(build, expected, tol):
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
    assert struct is not None and len(struct) == len(BUDGETS)

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
        # rank of the design it projects onto.  ``expected`` is the value
        # measured here and is carried only so a reader can see how far the
        # bound sits from it; it is not asserted.
        assert s.edf0 <= design_rank + tol, (s.edf0, design_rank, expected)


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
