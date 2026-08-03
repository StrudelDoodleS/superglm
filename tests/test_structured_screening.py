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
    assert f.rank == n_blocks * g + r

    Kinv = np.linalg.inv(K)
    got = f.diag_blocks()
    for q in range(n_blocks):
        s = slice(q * g, (q + 1) * g)
        assert np.allclose(got[q], Kinv[s, s], rtol=0, atol=1e-9)


def test_arrow_reports_the_rank_of_a_singular_system():
    """A degenerate level is routine at high cardinality, so its rank has to be
    counted rather than assumed: edf is read off ``rank(A) - lambda tr(A^-1 S)``
    and a mis-counted rank moves edf by a whole degree of freedom."""
    rng = np.random.default_rng(1)
    n_blocks, g, r = 6, 4, 3
    G, E, border, K = _random_arrow(rng, n_blocks, g, r)
    G[2] = 0.0  # an empty level: no rows, so no curvature and no coupling
    E[2] = 0.0
    K = _dense_arrow(G, E, border)
    f = factor_arrow(G, E, border)
    assert f.rank == np.linalg.matrix_rank(K)


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
    U_eff, rank_m = _profile(p)
    assert p.profiled_trace == pytest.approx(float(np.trace(V_eff)), rel=1e-12)

    # dense column order is p*k_b + q; the kernel groups by level
    k_a, k_b = B_a.shape[1], len(level_rows)
    perm = np.arange(k_a * k_b).reshape(k_a, k_b).T.reshape(-1)
    assert np.allclose(U_eff.reshape(-1), U_eff_dense[perm], rtol=0, atol=1e-12)

    scale = float(np.trace(V_eff)) / float(np.trace(S_ti))
    for mult in (1e-2, 1.0, 1e2, 1e4):
        lam = mult * scale
        A = V_eff + lam * S_ti
        T_dense = float(U_eff_dense @ scipy.linalg.solve(A, U_eff_dense, assume_a="pos"))
        edf_dense = float(np.trace(scipy.linalg.solve(A, V_eff, assume_a="pos")))
        T_s, edf_s = _evaluate(p, U_eff, rank_m, lam)
        assert T_s == pytest.approx(T_dense, rel=1e-10)
        assert edf_s == pytest.approx(edf_dense, rel=1e-10)


def test_structured_ladder_agrees_with_the_dense_ladder(moderate_pair):
    """End to end, every rung, including the clamped ones at the bracket edge."""
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
    """A reachable 0.5 rung must search rather than clamp at the old low edge.

    The raw trace is one while the stable profiled trace is 1e-12.  Scaling
    from the raw trace puts the old low edge at 1e-10, where EDF is about
    0.0099 and the reachable target is falsely classified as above the
    bracket.  This assertion therefore kills the exact regression mutation,
    not merely a change in a private trace value.
    """
    inputs = _near_absorbed_cells()
    p = spline_cat_moments(*inputs)
    U_eff, rank_m = _profile(p)
    old_lo = 1e-10 * float(np.trace(p.V[0])) / float(np.trace(p.S_a))
    _, old_edf = _evaluate(p, U_eff, rank_m, old_lo)
    assert old_edf == pytest.approx(0.00990194, rel=2e-5)

    from superglm.screening._structured import structured_ladder

    result = structured_ladder(p, budgets=(0.5,))[0]
    expected_trace = 1e-12 / (1.0 + 1e-12)
    assert p.profiled_trace == pytest.approx(expected_trace, rel=2e-13, abs=0.0)
    assert result.edf0 == pytest.approx(0.5, abs=2e-6)
    assert result.lambda0 < old_lo / 10.0


def test_near_absorbed_trace_and_ladder_match_the_dense_path():
    """Dense and structured paths take the same SEARCH action near absorption."""
    inputs = _near_absorbed_cells()
    B_a, S_a, S_cell, W_cell, level_rows = inputs
    U, V, C, M, S_ti, u_m = _dense_cell_inputs(*inputs)
    dense_trace = float(np.trace(V - C.T @ np.linalg.solve(M, C)))
    p = spline_cat_moments(*inputs)

    dense = penalized_score_statistic_ladder(
        U,
        V,
        C,
        M,
        S_ti,
        budgets=(0.5,),
        U_nuisance=u_m,
    )[0]
    from superglm.screening._structured import structured_ladder

    structured = structured_ladder(p, budgets=(0.5,))[0]
    # The dense subtraction retains about four significant digits here; the
    # structured residual construction retains the full weak energy.
    assert p.profiled_trace == pytest.approx(dense_trace, rel=2e-4, abs=0.0)
    assert structured.lambda0 == pytest.approx(dense.lambda0, rel=3e-6)
    assert structured.edf0 == pytest.approx(dense.edf0, abs=2e-6)
    assert structured.statistic == pytest.approx(dense.statistic, rel=1e-12)


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


def test_evaluate_clips_absorption_dust_but_signals_material_negative(monkeypatch):
    """Only round-off may become zero; a rank/inverse mismatch must escape."""
    import superglm.screening._structured as st

    pair = spline_cat_moments(*_near_absorbed_cells(0.0))
    U_eff = np.zeros_like(pair.U)

    class FakeFactor:
        def __init__(self, rank, inverse_block):
            self.rank = rank
            self.inverse_block = inverse_block

        def solve(self, block_rhs, border_rhs):
            return np.zeros_like(block_rhs), np.zeros_like(border_rhs)

        def diag_blocks(self):
            blocks = np.zeros((1, 2, 2))
            blocks[0, 0, 0] = self.inverse_block
            return blocks

    monkeypatch.setattr(st, "_pair_arrow", lambda *a, **k: FakeFactor(0, 1e-300))
    _, dust_edf = st._evaluate(pair, U_eff, 0, 1.0, np.array([0]))
    assert dust_edf == 0.0

    monkeypatch.setattr(st, "_pair_arrow", lambda *a, **k: FakeFactor(0, -1e-300))
    _, upper_dust_edf = st._evaluate(pair, U_eff, 0, 1.0, np.array([0]))
    assert upper_dust_edf == 0.0

    monkeypatch.setattr(st, "_pair_arrow", lambda *a, **k: FakeFactor(1, 1.25))
    with pytest.raises(st._UnstableStructuredEDFError, match="numerically inconsistent"):
        st._evaluate(pair, U_eff, 0, 1.0, np.array([0]))


def test_material_negative_edf_refuses_the_whole_ladder(monkeypatch):
    """No plausible row may escape after an unstable intermediate EDF."""
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


def test_negative_penalty_trace_refuses_the_nonmonotone_edf_curve(monkeypatch):
    """PSD inverse action requires both ``penalty >= 0`` and ``edf <= rank``."""
    import superglm.screening._structured as st

    rng = np.random.default_rng(14)
    n_rows, k_a, n_levels = 7, 4, 6
    B_a = rng.normal(size=(n_rows, k_a))
    delta = 10 ** rng.uniform(-9.0, -4.0)
    B_a[:, -1] = B_a[:, 0] + delta * rng.normal(size=n_rows)
    W_cell = 10 ** rng.uniform(-6.0, 6.0, size=(n_rows, n_levels))
    W_cell[rng.random(W_cell.shape) < 0.15] = 0.0
    penalty_vector = rng.normal(size=k_a)
    S_a = np.outer(penalty_vector, penalty_vector)
    level_rows = np.arange(1, n_levels, dtype=np.intp)
    pair = spline_cat_moments(
        B_a,
        S_a,
        np.zeros_like(W_cell),
        W_cell,
        level_rows,
    )
    U_eff, rank_m = st._profile(pair)
    ranks = st.block_ranks(pair)
    scale = pair.profiled_trace / (np.trace(S_a) * level_rows.size)
    lam = 1e8 * scale
    factor = st._pair_arrow(pair, lam, ranks)
    blocks = factor.diag_blocks()[:, :k_a, :k_a]
    penalty_term = lam * float(np.einsum("lpr,rp->", blocks, S_a, optimize=True))
    assert pair.profiled_trace == pytest.approx(239_102.68907761583, rel=2e-13)
    assert factor.rank - rank_m == 19
    assert penalty_term == pytest.approx(-58.45079964351394, rel=2e-12)

    with pytest.raises(st._UnstableStructuredEDFError, match="penalty trace"):
        st._evaluate(pair, U_eff, rank_m, lam, ranks)

    calls = 0
    real_evaluate = st._evaluate

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_evaluate(*args, **kwargs)

    monkeypatch.setattr(st, "_evaluate", counted)
    assert st.structured_ladder(pair, budgets=(13.0,)) is None
    assert calls > 2


def test_increasing_endpoint_edf_is_refused_even_when_each_value_is_bounded():
    """More penalty cannot increase EDF; bounded endpoints are not sufficient."""
    import superglm.screening._structured as st

    rng = np.random.default_rng(2)
    n_rows, k_a, n_levels = 8, 4, 6
    B_a = rng.normal(size=(n_rows, k_a))
    delta = 10 ** rng.uniform(-9.0, -4.0)
    B_a[:, -1] = B_a[:, 0] + delta * rng.normal(size=n_rows)
    W_cell = 10 ** rng.uniform(-6.0, 6.0, size=(n_rows, n_levels))
    W_cell[rng.random(W_cell.shape) < 0.15] = 0.0
    penalty_vector = rng.normal(size=k_a)
    S_a = np.outer(penalty_vector, penalty_vector)
    level_rows = np.arange(1, n_levels, dtype=np.intp)
    pair = spline_cat_moments(
        B_a,
        S_a,
        np.zeros_like(W_cell),
        W_cell,
        level_rows,
    )
    U_eff, rank_m = st._profile(pair)
    ranks = st.block_ranks(pair)
    scale = pair.profiled_trace / (np.trace(S_a) * level_rows.size)
    _, edf_lo = st._evaluate(pair, U_eff, rank_m, 1e-10 * scale, ranks)
    _, edf_hi = st._evaluate(pair, U_eff, rank_m, 1e10 * scale, ranks)

    assert pair.profiled_trace == pytest.approx(28_223.33962068437, rel=2e-13)
    assert edf_lo == pytest.approx(13.999660397988439, rel=2e-13)
    assert edf_hi == pytest.approx(14.030428796673123, rel=2e-13)
    assert edf_hi > edf_lo
    assert st.structured_ladder(pair, budgets=(14.0,)) is None


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
    returned zero.  The constructed arrow EDF jumps across 0.5 at that rank
    boundary, so the ladder must now refuse the unattainable rung rather than
    publish the nearest side of the jump.
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

    assert structured_ladder(p, budgets=(0.5,)) is None


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


def _reference_edf(U, V, C, M, S_ti, lam):
    """``edf(lambda)`` from a form that cannot cancel, for use as an ORACLE.

    ``sum_j a_j / (a_j + lambda (1 - a_j))`` over the simultaneously
    diagonalized dense pencil.  There is no ``rank - lambda tr(A^-1 S)``
    subtraction here, which is what makes it independent of the two paths it
    arbitrates.

    **It is not usable at the ladder's HIGH edge, and that is a property of
    float64 rather than of this implementation.**  ``1 - a`` cancels whenever
    ``a`` approaches 1, which is exactly what the penalty's null space
    produces: on the pair below, 19 of 209 directions carry ``a`` within 1e-12
    of 1, so ``1 - a`` is pure round-off there -- and the high edge multiplies
    it by ``lambda = 2.2e+09``, turning 1e-13 of noise into 1e-04 per
    direction.

    Three algebraically EQUIVALENT ways of writing the same oracle then
    disagree, on one machine in one process, at ``lambda = 2.16e+09``::

        via a                     18.999976474337
        via (v, s), balanced      18.997645230554
        both terms diagonalized   18.999417915384

    -- a spread of 2.3e-03.  At the low edge, ``lambda = 2.16e-11``, all three
    agree to 3.4e-09.  The oracle is therefore sound at one end of the bracket
    and beyond float64 at the other, and is used only where it is sound.
    """
    eps = np.finfo(np.float64).eps
    MinvC = np.linalg.solve(M, C)
    V_eff = V - C.T @ MinvC
    V_eff = 0.5 * (V_eff + V_eff.T)
    S = 0.5 * (S_ti + S_ti.T)
    G = 0.5 * ((V_eff + S) + (V_eff + S).T)
    w, Q = np.linalg.eigh(G)
    keep = w > w.size * eps * w.max()
    whiten = Q[:, keep] / np.sqrt(w[keep])
    Vt = whiten.T @ V_eff @ whiten
    a = np.clip(np.linalg.eigvalsh(0.5 * (Vt + Vt.T)), 0.0, 1.0)
    den = a + lam * (1.0 - a)
    ok = den > 0.0
    return float(np.sum(a[ok] / den[ok]))


# One budget above edf at the LOW bracket edge (~208) and four below edf at the
# HIGH edge (~19), so both clamping regimes are reported.  The old budgets were
# all below 19, so every rung clamped HIGH and the low edge was never asserted
# on -- which is how a whole-degree-of-freedom error lived there uncovered.
_EDGE_BUDGETS = (1.0, 2.0, 4.0, 8.0, 400.0)


@pytest.mark.parametrize("low_weight", [1.0, 0.01, 0.001])
def test_a_thin_level_does_not_cost_the_pair_a_degree_of_freedom(low_weight):
    """Both paths are checked against a REFERENCE, not against each other.

    ``edf`` is ``rank(A) - lambda tr(A^-1 S)`` and the rank is an INTEGER, so
    a mis-counted one moves the answer by a whole degree of freedom -- a
    different ``z``, not a rounding difference.

    This used to assert only that the two paths agreed with each other, which
    is weak in both directions: it passes when both are wrong together, and it
    fails when one moves TOWARD the truth.  Both happened.  It passed while
    the structured path was a full degree of freedom out at the low bracket
    edge, because every budget it used clamped at the HIGH edge and the low
    one was never reported.  ``_reference_edf`` is a form that cannot cancel,
    so it can arbitrate.

    Neither path dominates, which is the other reason agreement proves
    nothing: measured against the reference, at ``low_weight`` 0.01 the
    structured path is closer (4.20e-04 against dense 5.26e-04) and at 0.001
    the dense one is (2.07e-05 against structured 1.10e-03).

    Tolerances are set from measurement.  At the LOW edge the oracle is exact
    and the assertion is 1e-5 against a worst observed 1.991e-06 over the three
    weights -- three hundred times tighter than this test carried before, at
    the edge where the degree-of-freedom error lived.  At the HIGH edge the oracle is not usable
    at all (see ``_reference_edf``), so only path parity is asserted there, at
    3e-3 against a worst observed structured-vs-dense of 1.0780e-03 and a
    statistic gap of 1.5641e-06 relative.  That 3e-3 is unchanged, and is still
    333x tighter than the one degree of freedom this test exists to catch.

    The structured path is knowingly the less accurate at the high edge: it
    resolves directions the penalty has flattened, which amplifies ``1/w``
    inside ``tr(A^-1 S)``.  That is the deliberate price of resolving them at
    all -- see ``_solve_floor`` in :mod:`superglm.screening._arrow` -- and it
    buys the low edge, where the same change takes the error from 1.0 to
    8.1e-09.
    """
    from superglm.screening._structured import structured_ladder

    grab = _thin_level_pair(low_weight)
    U, V, C, M, S_ti, u_m = grab["args"]
    dense = penalized_score_statistic_ladder(
        U, V, C, M, S_ti, budgets=_EDGE_BUDGETS, U_nuisance=u_m
    )
    struct = structured_ladder(spline_cat_moments(*_structured_inputs(grab)), budgets=_EDGE_BUDGETS)
    saw_low_edge = False
    for budget, d, s in zip(_EDGE_BUDGETS, dense, struct, strict=True):
        # Each path is judged at its reported lambda.  Both brackets now use
        # tr(V_eff), though their endpoint solves remain independent and can
        # differ at the ill-conditioned high edge.
        if budget > 100.0:
            # LOW edge.  The oracle is sound here -- three equivalent forms of
            # it agree to 3.4e-09 -- so this is asserted tightly, and it is the
            # regime that matters: the whole-degree-of-freedom error this test
            # exists for lived at this edge, at 1.0 against the reference.
            # 1e-5 is set from the worst observed over the three weights,
            # 1.991e-06, and is 100,000x below the error it guards against.
            assert s.edf0 == pytest.approx(_reference_edf(U, V, C, M, S_ti, s.lambda0), abs=1e-5), (
                "structured",
                budget,
                s.edf0,
            )
            assert d.edf0 == pytest.approx(_reference_edf(U, V, C, M, S_ti, d.lambda0), abs=1e-5), (
                "dense",
                budget,
                d.edf0,
            )
            saw_low_edge = True
        # Parity holds at BOTH edges, and at the high edge it is all there is:
        # the oracle is beyond float64 there (see _reference_edf), so the two
        # paths arbitrate each other.  They are independent implementations --
        # a dense factorization against an arrow one -- so agreement is
        # evidence even if it is not proof.
        assert s.edf0 == pytest.approx(d.edf0, abs=3e-3), ("parity", budget)
        assert s.statistic == pytest.approx(d.statistic, rel=1e-3)
    assert saw_low_edge, "a rung must clamp at the LOW edge or this proves nothing"


def test_the_block_rank_is_the_one_a_dense_rank_call_reports():
    """The balanced count is not merely self-consistent, it is right.

    ``rank(P + lambda T)`` is the same for every positive ``lambda``, so an
    independent dense rank of the same blocks has to agree -- and the counts
    read off the two bracket edges are what must not be trusted for it.
    """
    from superglm.screening._structured import _unpenalized_blocks, block_ranks

    p = spline_cat_moments(*_structured_inputs(_thin_level_pair(0.01)))
    L, k_a = p.dims
    P = _unpenalized_blocks(p)
    T = np.zeros((k_a + 1, k_a + 1))
    T[:k_a, :k_a] = p.S_a
    want = sum(np.linalg.matrix_rank(P[q] / np.trace(P[q]) + T / np.trace(T)) for q in range(L))
    assert int(block_ranks(p).sum()) == want


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
