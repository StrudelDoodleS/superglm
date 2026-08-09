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

    monkeypatch.setattr(st, "_pair_arrow", lambda *a, **k: FakeFactor(1, -1.25))
    with pytest.raises(st._UnstableStructuredEDFError, match="penalty trace"):
        st._evaluate(pair, U_eff, 0, 1.0, np.array([0]))


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
    because :func:`~superglm.screening._structured.block_ranks` divided each
    level block by ITS OWN trace before counting it -- so the count could not
    see a level's mass, and the mass is what decides.  Driven from 1.0 down to
    1e-20, twenty decades, that count read 12 on the levels holding the weight
    and 12 on the levels holding none of it, and ``edf0`` never moved off 19.

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
    median of 7.0e-09, and the structured value's distance from
    ``_reference_edf`` -- which IS sound at this edge, and is what makes the
    rung a control on accuracy rather than on agreement -- has a median of
    1.6e-05 and a worst of 1.036e-04, inside the 3e-4 asserted below on every
    one of the 320 with 2.9x to spare.
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
            reference = _reference_edf(U, V, C, M, S_ti, s.lambda0)
            assert s.edf0 == pytest.approx(reference, abs=3e-4), (
                "low edge oracle",
                s.edf0,
                reference,
            )
            assert s.edf0 == pytest.approx(d.edf0, abs=2e-4), ("low edge parity", s.edf0, d.edf0)
        assert s.lambda0 == pytest.approx(d.lambda0, rel=1e-12), ("lambda0", budget)
        assert s.statistic == pytest.approx(d.statistic, rel=1e-4), ("statistic", budget)


def test_the_block_rank_matches_a_dense_rank_only_where_the_terms_separate():
    """A dense rank call arbitrates this count in one of the two regimes.

    Where no level's free curvature comes near the penalty the high edge
    still applies, ``rank(P + lambda T)`` is the ordinary lambda-independent
    rank, an independent dense call has to agree, and it has to agree at
    EVERY lambda.  ``_thin_level_pair(0.01)``'s rarest level is that case: its
    free direction carries 1.7e-02 of curvature against a reach of 7.4e-06,
    2317x clear, so nothing there depends on where in the bracket it is
    counted.

    Where a level's curvature falls UNDER that reach the dense call stops
    being an oracle in either direction, and that is the whole reason this
    count exists.  On ``_vanishing_mass_pair`` the balanced call still reads
    228 -- the rank of the exact family, which is correct and useless, since
    three of the directions it counts are ones the inverse at the high edge
    cannot resolve and the penalty has flattened.  A dense rank of the FLOAT
    block the arrow kernel actually inverts reads 222 instead: it also loses
    each starved level's own contrast, which cancels against ``rank(M)`` and
    must be kept.  225 is neither of them, and it is the count that makes
    ``rank - lambda tr(A^-1 S)`` come out at the sixteen degrees of freedom
    the penalty genuinely leaves free.
    """
    from superglm.screening._structured import _unpenalized_blocks, block_ranks

    def penalty_block(p):
        _, k_a = p.dims
        T = np.zeros((k_a + 1, k_a + 1))
        T[:k_a, :k_a] = p.S_a
        return T

    def balanced_dense_rank(p):
        P, T = _unpenalized_blocks(p), penalty_block(p)
        return sum(
            np.linalg.matrix_rank(P[q] / np.trace(P[q]) + T / np.trace(T)) for q in range(p.dims[0])
        )

    def bracket(p):
        tr_S = float(np.trace(p.S_a)) * p.dims[0]
        scale = max(p.profiled_trace, 1e-300) / max(tr_S, 1e-300)
        return 1e-10 * scale, scale, 1e10 * scale

    p = spline_cat_moments(*_structured_inputs(_thin_level_pair(0.01)))
    want = balanced_dense_rank(p)
    assert want == 228
    for lam in bracket(p):
        assert int(block_ranks(p, lam).sum()) == want

    starved = spline_cat_moments(*_structured_inputs(_vanishing_mass_pair(1e-12)))
    lo, mid, hi = bracket(starved)
    assert balanced_dense_rank(starved) == 228
    assert int(block_ranks(starved, lo).sum()) == 228
    assert int(block_ranks(starved, mid).sum()) == 228
    assert int(block_ranks(starved, hi).sum()) == 225
    P, T = _unpenalized_blocks(starved), penalty_block(starved)
    assert sum(np.linalg.matrix_rank(P[q] + hi * T) for q in range(starved.dims[0])) == 222


def _rank_one_penalty_pair(seed, L, reps, width, n_narrow):
    """A TWO-column spline margin, where the tensor penalty is rank ONE.

    ``Spline(kind="cr", k=3)`` leaves ``k_a = 2``, and the pair's ``S_a`` is
    then ``u u'`` for a unit ``u``: exactly rank one in exact arithmetic, and
    in float a matrix whose null residue a symmetric eigensolver frequently
    cannot resolve AT ALL, handing back a bit-exact ``0.0``.  That is the one
    regime :data:`~superglm.screening._structured._PENALTY_RESIDUE_FLOOR`
    governs, and no other fixture in this file reaches it: over 405 pipeline
    configurations of ``ps``/``cr``/``bs`` at 3 to 20 knots, 5 seeds and three
    level layouts, every wider margin resolved a NONZERO residue and the floor
    was never consulted.

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
    bit, and one bit of ``lambda`` is not nothing at this edge: on
    ``_resolved_residue_pair(3)`` the two spellings move the reported edf by
    0.065 df.  Everything asserted here about a rung the ladder publishes has
    to be evaluated at the ladder's own ``lambda``, not at an equivalent one.
    """
    tr_S = float(np.trace(p.S_a)) * p.dims[0]
    return 1e10 * (max(p.profiled_trace, 1e-300) / max(tr_S, 1e-300))


def _free_direction_curvature(p, lam):
    """The free direction's numbers: ``(v per level, reach, lam sigma_max, cut, dust)``.

    ``v`` is each level's Schur complement on that direction, which is what
    :func:`~superglm.screening._structured.block_ranks` compares.  ``reach``
    is ``lam |sigma_min|`` -- the eigensolver's OWN residue, at its magnitude
    because the sign of a null residue is round-off -- and ``cut`` is what
    :data:`~superglm.screening._structured._PENALTY_RESIDUE_FLOOR` would
    substitute for it.  ``dust`` is the other floor, ``_RCOND`` on the level's
    raw moments, returned so a test can show which of the two decides.

    Both families below carry a penalty that is rank deficient by exactly one,
    which the assertion states rather than assumes.
    """
    from superglm.screening._arrow import _RCOND, _solve_floor
    from superglm.screening._structured import _PENALTY_RESIDUE_FLOOR

    _, k_a = p.dims
    sigma, directions = np.linalg.eigh(0.5 * (p.S_a + p.S_a.T))
    reach = float(lam) * np.abs(sigma)
    top = float(reach.max())
    free = reach <= _solve_floor(k_a + 1) * top
    assert int(free.sum()) == 1, (sigma, free)
    mass = np.where(p.m > 0.0, p.m, 1.0)
    direction = directions[:, free]
    projected = (p.c @ direction).ravel()
    curvature = np.einsum("ip,lij,jq->lpq", direction, p.V, direction, optimize=True).reshape(-1)
    curvature = curvature - projected**2 / mass
    dust = _RCOND * (np.einsum("lpp->l", p.V) + p.m)
    return curvature, float(reach[free][0]), top, _PENALTY_RESIDUE_FLOOR * top, dust


def _floor_band_directions(curvature, floor, dust, direction):
    """The free directions that moving the floor the parametrized way would move.

    ``direction > 0`` -- DELETING the constant -- keeps every direction between
    the level's own dust floor and it; ``direction < 0`` -- TRIPLING it -- drops
    every direction in the ``(1, 3] eps`` band.  Whether a draw carries any such
    direction is a property of that draw rather than of the constant: 4 of the 9
    draws that reach the regime for the tripling family below carry none at all,
    their two nearest sitting between 3.78 and 4.98 times the floor rather than
    inside the band (3.99 and 4.10 on seed 6).  Which is why the scan's
    predicate and the assertions below have to read this from one place instead
    of two copies of it.
    """
    if direction > 0:
        return (curvature > dust) & (curvature <= floor)
    return (curvature > floor) & (curvature <= 3.0 * floor)


def _unpenalized_level_ranks(p):
    """What every level contributes before any free direction is judged.

    Its own contrast where it holds mass, plus the ``k_a - 1`` directions the
    penalty genuinely reaches.  Written here so a test can state the whole
    count :func:`~superglm.screening._structured.block_ranks` should return
    without borrowing that function's arithmetic.
    """
    _, k_a = p.dims
    return np.where(p.m > 0.0, 1, 0) + (k_a - 1)


_FLOOR_SCAN_SEEDS = 40


def _first_exact_zero_draw(build, start, exhibits, scan=_FLOOR_SCAN_SEEDS):
    """The first draw that is EXACTLY singular AND shows what the caller asserts.

    Deterministic and bounded: ``start`` first, then ``0 .. scan - 1``.  A draw
    qualifies on two counts, not one -- ``eigh`` reports its ``S_a`` as exactly
    singular, so the floor is read at all, and ``exhibits`` finds on it the
    direction the caller's assertions are about.  The second is not implied by
    the first: the floor is read on every draw that reaches the regime but only
    moves the count on some of them, so a scan that stopped at the regime would
    hand a caller a draw its assertions do not describe.

    Both are facts about the platform's LAPACK rather than about the fixture,
    and they are DIFFERENT facts, so they are counted apart: returns
    ``(seed, pair, reached)``, or ``(None, None, reached)`` where ``reached`` is
    how many of the scanned draws landed in the regime -- which is what lets the
    caller say WHICH of the two it is skipping on.
    """
    reached = 0
    for seed in (start, *range(scan)):
        p = spline_cat_moments(*_structured_inputs(build(seed)))
        if p.dims[1] != 2 or np.linalg.eigvalsh(0.5 * (p.S_a + p.S_a.T))[0] != 0.0:
            continue
        reached += 1
        if exhibits(p):
            return seed, p, reached
    return None, None, reached


@pytest.mark.parametrize(
    ("seed", "L", "reps", "width", "n_narrow", "multiple", "direction"),
    [
        (2, 10, 20, 1e-4, 3, 0.0, +1),
        (13, 6, 12, 2e-3, 3, 3.0, -1),
    ],
    ids=["deleting-the-floor-adds-df", "tripling-it-removes-df"],
)
def test_the_penalty_residue_floor_is_pinned_from_both_sides(
    seed, L, reps, width, n_narrow, multiple, direction, monkeypatch
):
    """The floor's OWN regime, where the eigensolver reports an exact zero.

    Everywhere else :func:`~superglm.screening._structured.block_ranks` uses
    the residue the eigensolver returns and the constant is never read, so the
    rest of this file cannot constrain it: mutating it to 0, 1e-4, 1e-2, 1, 3,
    10 and 100 times ``eps`` left every other test here green.  A future
    maintainer could delete it outright.

    Here the margin has two columns, its penalty is rank one, and ``eigh``
    hands back a bit-exact ``0.0`` -- while :func:`_exact_residue_bounds`,
    which uses no eigensolver at all, shows the same float matrix is NOT
    singular.  So the floor is a guess at something real, and both directions
    of getting it wrong cost whole degrees of freedom:

    * REMOVED, the count keeps directions the penalty has flattened.  On the
      first case two directions carry ``4.15e-03`` and ``4.94e-03`` of one
      round-off unit of the largest reach, against an exact residue of
      ``0.045`` to ``0.090`` of it, so their exact shares are 0.084 and 0.099
      -- and the ladder reports 8.00 df where it should report 6.00.  This is
      the same failure as on four wide pairs of the shape the structured
      kernel exists for, where removing the floor puts the count 5 df above a
      high-precision oracle.
    * INFLATED, it discards directions the data almost wholly carries.  On the
      second case two directions sit inside the ``(1, 3] eps`` band at
      ``1.59`` and ``2.00`` round-off units, with exact shares of 0.9942 and
      0.9954, and a 3x floor takes the ladder from 4.98 df to 2.98.

    The assertions below therefore bound the constant on BOTH sides from
    exact arithmetic rather than from either path's output, and each case
    fails if the constant is moved the way its parametrization names.

    **WHETHER A DRAW REACHES THE REGIME AT ALL IS A PROPERTY OF THE PLATFORM'S
    LAPACK, SO THE FIXTURE IS SCANNED RATHER THAN ASSUMED.**  This used to skip
    whenever the shipped seed resolved a nonzero residue -- and of the 41 draws
    the scan below walks, only 14 land in the regime for the first family here
    and only 10 for the second, so the other 27 and 31 resolve one.
    On a machine where the shipped seeds are among them BOTH parametrizations
    would have skipped, leaving the constant unpinned, which is precisely the
    hole this test exists to close, and a skip does not show up in a summary
    line.  :func:`_first_exact_zero_draw` therefore walks the shipped seed and
    then seeds 0 to 39, and takes the first draw that lands in the regime AND
    carries a direction this parametrization moves.

    **REACHING THE REGIME IS NOT ENOUGH, WHICH IS WHY THE SCAN TAKES A
    PREDICATE.**  The floor is read on every draw that reaches it but moves the
    count on only some.  Counted by seed rather than by draw -- the walk
    repeats the shipped seed, which is why 41 draws cover 40 seeds and the 14
    and 10 above are 13 and 9 -- 4 of the 9 seeds that reach the regime for the
    tripling family have NO direction in the ``(1, 3] eps`` band, and 1 of the
    13 for the deleting family is a pair ``structured_ladder`` declines
    outright.  A scan that stopped at the regime would hand this test one of
    those on any platform whose LAPACK moved it off the shipped seed, and it
    would then be RED about the draw rather than about the constant.  Selecting
    the fixture is not weakening the test: nothing below is relaxed to fit the
    predicate, which decides at the constant's SHIPPED value so that moving or
    deleting the constant still fails these assertions rather than emptying the
    band and skipping, and 12 of the 40 seeds qualify for the deleting family
    and 5 for the tripling one, on every one of which the count assertions hold
    as written.  Only if no draw qualifies is anything skipped, and the skip
    distinguishes the two platform facts -- none reached the regime, or some
    did and none showed the phenomenon.  The share bounds quoted above are
    measured on the SHIPPED draw and are asserted only when the scan returns
    it; the count assertions, which are what pin the constant, hold for
    whichever draw the scan lands on.
    """
    import superglm.screening._structured as structured
    from superglm.screening._structured import block_ranks, structured_ladder

    eps = float(np.finfo(np.float64).eps)

    def exhibits(p):
        """Does THIS draw carry what the assertions below state about it?

        The direction the parametrized move would move, and a pair BOTH ladders
        will score -- the mutated one included, since the degree of freedom
        asserted at the end reads an edf from each and a declined pair has
        neither.

        Decided with the constant pinned to the value this test defends, and
        deliberately NOT to the installed one.  What varies between platforms
        is which draw reaches the regime, never what the constant is; a
        predicate that read the installed value would let a maintainer who
        deleted the constant empty the band on every draw, run the scan out and
        SKIP -- reopening from the other side exactly the fail-open this scan
        was written to close -- instead of failing the assertions below.
        """
        with monkeypatch.context() as shipped:
            shipped.setattr(structured, "_PENALTY_RESIDUE_FLOOR", eps)
            curvature, _reach, _top, floor, dust = _free_direction_curvature(
                p, _ladder_high_edge(p)
            )
            if not _floor_band_directions(curvature, floor, dust, direction).any():
                return False
            if structured_ladder(p, budgets=(2.0,)) is None:
                return False
        with monkeypatch.context() as moved_floor:
            moved_floor.setattr(structured, "_PENALTY_RESIDUE_FLOOR", multiple * eps)
            return structured_ladder(p, budgets=(2.0,)) is not None

    drawn, p, reached = _first_exact_zero_draw(
        lambda s: _rank_one_penalty_pair(s, L, reps, width, n_narrow), seed, exhibits
    )
    if p is None:
        pytest.skip(
            (
                f"none of the {1 + _FLOOR_SCAN_SEEDS} draws scanned reached the floor's regime: "
                "this platform's eigensolver resolved a nonzero residue on every one of them, so "
                "the floor is never read here and nothing on this machine can constrain it"
            )
            if reached == 0
            else (
                f"{reached} of the {1 + _FLOOR_SCAN_SEEDS} draws scanned reached the floor's "
                "regime, so it IS read on this platform, but none of them carries a direction "
                "this parametrization moves: nothing here can show what moving it costs"
            )
        )
    assert p.dims[1] == 2, p.dims
    S_a = 0.5 * (p.S_a + p.S_a.T)
    sigma = np.linalg.eigvalsh(S_a)
    assert sigma[0] == 0.0, sigma

    # The matrix is NOT singular, and the floor is above the residue it stands
    # in for -- which is the whole reason a direction can be wrongly dropped.
    residue_lo, residue_hi = _exact_residue_bounds(S_a)
    assert 0.0 < residue_lo <= residue_hi < eps * float(sigma[-1])

    lam_hi = _ladder_high_edge(p)
    curvature, _reach, top, floor, dust = _free_direction_curvature(p, lam_hi)
    # The constant is what decides here, not the other floor: that one is four
    # orders below it, so neither comparison below is really about the dust.
    assert dust.max() < floor, (dust.max(), floor)
    # The exact share every free direction carries, at its most generous: the
    # SMALLER residue makes the penalty reach least and the share largest.
    share = curvature / (curvature + lam_hi * residue_lo)
    # Deleting the constant leaves the level's own dust floor standing, so a
    # direction under THAT is not one deleting it would recover.  Read through
    # the same helper the scan's predicate used, so the draw it selected and
    # the assertions it was selected for cannot drift apart.
    moved = _floor_band_directions(curvature, floor, dust, direction)
    if direction > 0:  # removing the floor keeps these, and they are flattened
        df_moved = int(moved.sum())
        assert df_moved >= 1, (curvature / floor, share)
        if drawn == seed:
            assert df_moved == 2, (curvature / floor, share)
            assert share[moved].max() < 0.11, share[moved]
    else:  # tripling the floor drops these, and the data carries them
        df_moved = -int(moved.sum())
        assert df_moved <= -1, (curvature / floor, share)
        if drawn == seed:
            assert df_moved == -2, (curvature / floor, share)
            assert share[moved].min() > 0.98, share[moved]

    shipped_rank = int(block_ranks(p, lam_hi).sum())
    shipped_edf = structured_ladder(p, budgets=(2.0,))[0].edf0
    monkeypatch.setattr(structured, "_PENALTY_RESIDUE_FLOOR", multiple * eps)
    mutated_rank = int(block_ranks(p, lam_hi).sum())
    mutated_edf = structured_ladder(p, budgets=(2.0,))[0].edf0

    assert mutated_rank - shipped_rank == df_moved, (drawn, shipped_rank, mutated_rank)
    assert mutated_edf - shipped_edf == pytest.approx(float(df_moved), abs=1e-6), (
        drawn,
        shipped_edf,
        mutated_edf,
    )


def _resolved_residue_pair(seed, width=2e-3, L=6, reps=12, n_narrow=2):
    """A margin whose null residue the eigensolver RESOLVES, below one ``eps``.

    ``Spline(kind="cr", n_knots=3)`` leaves ``k_a = 4`` and a penalty of rank
    three, and ``eigh`` returns a nonzero ``sigma_min`` on every seed scanned
    -- signed, and between 0.007 and 0.23 round-off units of ``sigma_max``
    over twelve of them.  Resolving the residue is not by itself enough to
    separate the rules; a level's curvature has to land BETWEEN the residue
    and the floor.  No other fixture in this file arranges that.
    :func:`_rank_one_penalty_pair` is the exact-zero regime, where every
    candidate rule reads the same floor by construction, and on
    ``_thin_level_pair`` and ``_vanishing_mass_pair`` -- ``ps(8)``, residue
    resolved at 0.42 round-off units -- the free curvature is 2.0e-07 to
    1.7e+05 times the floor and never in between, so those agree too.

    ``n_narrow`` levels are squeezed into a band of ``x`` of the given width,
    which is what puts one level's free curvature between the resolved residue
    and the floor.  Nothing else is degenerate: unit weights throughout, every
    level keeps all its rows.
    """
    rng = np.random.default_rng(seed)
    n = L * reps
    g = np.repeat([f"L{i}" for i in range(L)], reps)
    x = rng.uniform(0.05, 0.95, n)
    for i in range(n_narrow):
        selected = g == f"L{i}"
        x[selected] = 0.2 + 0.5 * i / n_narrow + width * rng.uniform(-0.5, 0.5, selected.sum())
    slope = rng.normal(size=L).repeat(reps)
    df = pd.DataFrame({"g": g, "x": x})
    y = slope * x + rng.normal(scale=0.5, size=n)
    return _capture(
        df,
        y,
        {"g": Categorical(), "x": Spline(kind="cr", n_knots=3)},
        ("x", "g"),
        sample_weight=np.ones(n),
    )


@pytest.mark.parametrize(
    ("seed", "residue_sign"),
    [(3, +1.0), (4, -1.0), (7, -1.0)],
    ids=["positive-residue", "negative-residue", "negative-residue-again"],
)
def test_a_resolved_penalty_residue_is_used_rather_than_the_floor(seed, residue_sign):
    """A residue the eigensolver DID resolve is the reach, floor or no floor.

    :func:`test_the_penalty_residue_floor_is_pinned_from_both_sides` pins the
    CONSTANT.  It cannot pin the RULE, because it only ever runs where the
    residue is an exact zero and every candidate rule reads the floor there.
    Both of the forms this one replaced --
    ``max(residue, _PENALTY_RESIDUE_FLOOR * top)`` and an unconditional
    ``_PENALTY_RESIDUE_FLOOR * top`` -- leave that test, and every other test
    in this file, green while costing a whole degree of freedom here:
    reverting the line to either one fails exactly the three cases below and
    nothing else in the three screening suites.

    The family is the one the regression was found on: 6 levels, 12 rows in
    every level, unit weights, two levels inside a 2e-3 band of ``x``, a
    ``cr(n_knots=3)`` margin.  ``eigh`` resolves ``sigma_min`` on all twelve
    seeds scanned, at 0.007 to 0.23 round-off units of ``sigma_max`` -- under
    the floor every time, so the floor and the measurement disagree -- and one
    level's free curvature lands between the two on six of them.  There the
    reported edf is exactly 1.000000 df apart between the rules, and it is the
    reach-aware value that is right: an exact-rational oracle puts seeds 3, 4
    and 7 at 5.006935, 5.399635 and 5.139469, against 5.004628 / 5.284769 /
    5.001895 counted from the residue and 4.004628 / 4.284769 / 4.001895
    counted from the floor.

    The sign is asserted rather than assumed because it is what ``np.abs``
    earns its place for.  A null residue is round-off and comes out either
    way; on the two negative seeds here, taking ``np.maximum(sigma, 0.0)``
    instead reports NO residue, hands the floor its case, and costs the same
    degree of freedom.  On the positive seed that mutation is invisible, which
    is why more than one sign is parametrized.

    Nothing here needs a new oracle: the assertion is that the count KEEPS the
    direction the measured reach leaves free, and both the reach and the
    curvature come from the pair's own moments.
    """
    from superglm.screening._structured import (
        _evaluate,
        _profile,
        block_ranks,
        structured_ladder,
    )

    p = spline_cat_moments(*_structured_inputs(_resolved_residue_pair(seed)))
    assert p.dims[1] == 4, p.dims
    sigma = np.linalg.eigvalsh(0.5 * (p.S_a + p.S_a.T))
    eps = float(np.finfo(np.float64).eps)
    # RESOLVED, signed, and under the floor: the one regime where substituting
    # the constant for the measurement is observable.
    assert sigma[0] != 0.0, sigma
    assert np.sign(sigma[0]) == residue_sign, sigma[0]
    assert abs(sigma[0]) < eps * float(abs(sigma[-1])), (sigma[0], sigma[-1])

    lam_hi = _ladder_high_edge(p)
    curvature, reach, _top, floor, dust = _free_direction_curvature(p, lam_hi)
    # Neither comparison is decided by the OTHER floor, the one on the level's
    # raw moments: it is four orders below the smaller of the two.
    assert dust.max() < reach < floor, (dust.max(), reach, floor)

    kept_by_reach = curvature > reach
    kept_by_floor = curvature > floor
    window = int(kept_by_reach.sum()) - int(kept_by_floor.sum())
    assert window == 1, (curvature / reach, curvature / floor)

    # The count keeps it.  ``base`` is written out here rather than taken from
    # block_ranks, so this is a statement about the answer and not a copy of
    # the arithmetic that produces it.
    base = _unpenalized_level_ranks(p)
    assert int(block_ranks(p, lam_hi).sum()) == int(base.sum()) + int(kept_by_reach.sum()), (
        int(block_ranks(p, lam_hi).sum()),
        int(base.sum()),
        curvature / reach,
    )

    # And a whole degree of freedom rides on it, through the same ``_evaluate``
    # the ladder uses -- handed the floor's count explicitly for the contrast.
    U_eff, rank_m = _profile(p)
    _, edf = _evaluate(p, U_eff, rank_m, lam_hi)
    _, floored_edf = _evaluate(p, U_eff, rank_m, lam_hi, base + kept_by_floor)
    assert edf - floored_edf == pytest.approx(float(window), abs=1e-6), (edf, floored_edf)

    # It is the ladder's published rung, not an internal number: every budget
    # below the high edge clamps there.
    assert structured_ladder(p, budgets=(2.0,))[0].edf0 == edf


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
