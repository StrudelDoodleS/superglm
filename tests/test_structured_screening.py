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


@pytest.mark.parametrize("low_weight", [1.0, 0.01, 0.001])
def test_a_thin_level_does_not_cost_the_pair_a_degree_of_freedom(low_weight):
    """Cross-path agreement on a system the rank cut has to resolve.

    ``edf`` is ``rank(A) - lambda tr(A^-1 S)`` and the rank is an INTEGER, so
    a mis-counted one moves the answer by a whole degree of freedom -- which
    is a different ``z``, not a rounding difference.  Counting it at the
    bracket edge did exactly that: measured 17.99995 against the dense path's
    18.99991 with one level of twenty carried at 1/100th the weight.
    """
    from superglm.screening._structured import structured_ladder

    grab = _thin_level_pair(low_weight)
    U, V, C, M, S_ti, u_m = grab["args"]
    dense = penalized_score_statistic_ladder(U, V, C, M, S_ti, budgets=BUDGETS, U_nuisance=u_m)
    struct = structured_ladder(spline_cat_moments(*_structured_inputs(grab)), budgets=BUDGETS)
    for d, s in zip(dense, struct, strict=True):
        assert s.edf0 == pytest.approx(d.edf0, abs=1e-3)
        assert s.statistic == pytest.approx(d.statistic, rel=1e-3)


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
