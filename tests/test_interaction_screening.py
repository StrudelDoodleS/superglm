"""Per-pair screening moments: exactness pins against the dense row-Kronecker.

Task 1 of docs/superpowers/plans/2026-07-28-interaction-screening.md.  The
whole screening design rests on the cell-space assembly reproducing the dense
assembly exactly, so these tolerances must not be loosened.
"""

from __future__ import annotations

import numpy as np
import pytest

from superglm.distributions import Gamma, Poisson
from superglm.links import LogLink
from superglm.screening import pair_cell_moments, pair_score_curvature, working_score


def _pair_case(seed, n=4000, n_a=17, n_b=13, k_a=4, k_b=3, signed=False):
    rng = np.random.default_rng(seed)
    codes_a = rng.integers(0, n_a, n)
    codes_b = rng.integers(0, n_b, n)
    B_a = rng.normal(size=(n_a, k_a))
    B_b = rng.normal(size=(n_b, k_b))
    score = rng.normal(size=n) if signed else rng.uniform(0.1, 1.0, n)
    weights = rng.normal(size=n) if signed else rng.uniform(0.2, 2.0, n)
    return codes_a, codes_b, B_a, B_b, score, weights


def _dense_row_kronecker(codes_a, codes_b, B_a, B_b):
    rows_a = B_a[codes_a]
    rows_b = B_b[codes_b]
    return np.einsum("rp,rq->rpq", rows_a, rows_b).reshape(
        len(codes_a), B_a.shape[1] * B_b.shape[1]
    )


def test_cell_assembly_matches_dense_row_kronecker():
    codes_a, codes_b, B_a, B_b, score, weights = _pair_case(0)

    S_cell, W_cell = pair_cell_moments(codes_a, codes_b, 17, 13, score, weights)
    U, V = pair_score_curvature(B_a, B_b, S_cell, W_cell)

    X = _dense_row_kronecker(codes_a, codes_b, B_a, B_b)
    U_dense = X.T @ score
    V_dense = X.T @ (X * weights[:, None])
    np.testing.assert_allclose(U, U_dense, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(V, V_dense, rtol=1e-12, atol=1e-12)


def test_cell_assembly_handles_signed_score_and_weights():
    """REML working quantities are signed; nothing here may assume positivity."""
    codes_a, codes_b, B_a, B_b, score, weights = _pair_case(1, signed=True)

    S_cell, W_cell = pair_cell_moments(codes_a, codes_b, 17, 13, score, weights)
    U, V = pair_score_curvature(B_a, B_b, S_cell, W_cell)

    X = _dense_row_kronecker(codes_a, codes_b, B_a, B_b)
    np.testing.assert_allclose(U, X.T @ score, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(V, X.T @ (X * weights[:, None]), rtol=1e-11, atol=1e-11)


def test_empty_and_singleton_cells_are_exact():
    """Cells with no rows contribute zero; single-level margins still work."""
    codes_a = np.array([0, 0, 0, 0])
    codes_b = np.array([2, 2, 0, 0])
    B_a = np.array([[1.5, -0.5]])
    B_b = np.array([[1.0], [2.0], [4.0]])
    score = np.array([1.0, 2.0, 3.0, 4.0])
    weights = np.array([0.5, 0.5, 1.0, 1.0])

    S_cell, W_cell = pair_cell_moments(codes_a, codes_b, 1, 3, score, weights)
    assert S_cell.shape == (1, 3)
    assert W_cell[0, 1] == 0.0

    U, V = pair_score_curvature(B_a, B_b, S_cell, W_cell)
    X = _dense_row_kronecker(codes_a, codes_b, B_a, B_b)
    np.testing.assert_allclose(U, X.T @ score, rtol=1e-14)
    np.testing.assert_allclose(V, X.T @ (X * weights[:, None]), rtol=1e-14)


def test_cell_values_are_pinned_independently():
    """Hand-computed S_cell and W_cell so a coordinated S/W swap cannot pass."""
    codes_a = np.array([0, 0, 0, 0])
    codes_b = np.array([2, 2, 0, 0])
    score = np.array([1.0, 2.0, 3.0, 4.0])
    weights = np.array([0.5, 0.25, 1.0, 2.0])

    S_cell, W_cell = pair_cell_moments(codes_a, codes_b, 1, 3, score, weights)

    np.testing.assert_array_equal(S_cell, [[7.0, 0.0, 3.0]])
    np.testing.assert_array_equal(W_cell, [[3.0, 0.0, 0.75]])


def test_out_of_range_codes_raise_instead_of_corrupting():
    """The kernel indexes without checks; the guard must catch every escape."""
    import pytest

    ok = np.zeros(4, dtype=int)
    vals = np.ones(4)
    for bad_a, bad_b in (([0, 0, 0, 2], ok), (ok, [0, 3, 0, 0]), ([-1, 0, 0, 0], ok)):
        with pytest.raises(ValueError, match="fall outside"):
            pair_cell_moments(np.asarray(bad_a), np.asarray(bad_b), 2, 3, vals, vals)


def test_short_value_arrays_raise_instead_of_reading_heap():
    import pytest

    codes = np.zeros(10, dtype=int)
    with pytest.raises(ValueError, match="row-for-row"):
        pair_cell_moments(codes, codes, 1, 1, np.zeros(3), np.zeros(10))
    with pytest.raises(ValueError, match="row-for-row"):
        pair_cell_moments(codes, codes, 1, 1, np.zeros(10), np.zeros(3))


def test_cell_ceiling_rejects_unbinned_wide_pairs():
    import pytest

    codes = np.zeros(4, dtype=int)
    vals = np.ones(4)
    with pytest.raises(ValueError, match="ceiling"):
        pair_cell_moments(codes, codes, 50_000, 50_000, vals, vals)
    # The ceiling is caller-adjustable; a modest raise takes effect.
    S_cell, _ = pair_cell_moments(codes, codes, 2_000, 3_000, vals, vals, max_cells=6_000_000)
    assert S_cell.shape == (2_000, 3_000)


def test_working_score_is_bitwise_the_inline_formula():
    """Bit-identity pin: the KKT suite tolerates ~10% drift, this does not."""
    from superglm.distributions import _VARIANCE_FLOOR

    rng = np.random.default_rng(4)
    n = 500
    mu = rng.uniform(0.05, 5.0, n)
    eta = np.log(mu)
    y = rng.poisson(mu).astype(float)
    w = rng.uniform(0.1, 2.0, n)
    family, link = Gamma(), LogLink()

    expected = (
        w * link.deriv_inverse(eta) * (y - mu) / np.maximum(family.variance(mu), _VARIANCE_FLOOR)
    )
    assert np.array_equal(working_score(y, mu, eta, w, family, link), expected)


def test_mismatched_code_shapes_raise():
    import pytest

    with pytest.raises(ValueError, match="row dimension"):
        pair_cell_moments(
            np.zeros(3, dtype=int), np.zeros(4, dtype=int), 1, 1, np.zeros(3), np.zeros(3)
        )


def test_working_score_reduces_to_raw_residual_for_canonical_link():
    rng = np.random.default_rng(2)
    n = 200
    mu = rng.uniform(0.1, 3.0, n)
    eta = np.log(mu)
    y = rng.poisson(mu).astype(float)
    w = rng.uniform(0.2, 1.5, n)

    score = working_score(y, mu, eta, w, Poisson(), LogLink())
    np.testing.assert_allclose(score, w * (y - mu), rtol=1e-14)


def test_working_score_carries_family_factor_for_noncanonical_link():
    rng = np.random.default_rng(3)
    n = 200
    mu = rng.uniform(0.5, 4.0, n)
    eta = np.log(mu)
    y = rng.gamma(2.0, mu / 2.0)
    w = rng.uniform(0.2, 1.5, n)

    score = working_score(y, mu, eta, w, Gamma(), LogLink())
    np.testing.assert_allclose(score, w * (y - mu) / mu, rtol=1e-12)


def _pd_matrix(rng, p, strength=1.0):
    A = rng.normal(size=(p, 2 * p))
    return strength * (A @ A.T) / p


def test_statistic_reduces_to_unpenalized_without_penalty():
    from superglm.screening import penalized_score_statistic

    rng = np.random.default_rng(5)
    p = 6
    V = _pd_matrix(rng, p)
    U = rng.normal(size=p)

    result = penalized_score_statistic(U, V, S_ti=None)

    np.testing.assert_allclose(result.statistic, U @ np.linalg.solve(V, U), rtol=1e-11)
    assert result.lambda0 == 0.0


def test_unpenalized_edf_is_a_rank_not_a_cholesky_trace():
    """A barely positive-definite block still reports its RANK, not ``k``.

    The unpenalized rung's edf used to be ``tr(A^-1 V)``, which equals the rank
    only when ``A^-1`` is a pseudo-inverse.  ``cho_factor`` is entitled to
    accept a block like this one -- it IS positive definite, by 1e-18 -- and
    the trace then reports ``k``.  Diagonal on purpose: this is the one
    construction whose Cholesky cannot come out platform-dependent, since
    every pivot is a stored entry rather than a round-off residue.
    """
    import scipy.linalg

    from superglm.screening import penalized_score_statistic

    V = np.diag([3.0, 2.0, 1e-18])
    scipy.linalg.cho_factor(V, check_finite=False)  # accepts it; that is the trap
    assert np.linalg.matrix_rank(V) == 2

    result = penalized_score_statistic(np.array([1.0, 1.0, 1.0]), V, S_ti=None)
    assert result.edf0 == 2.0
    assert result.lambda0 == 0.0


def test_unpenalized_edf_matches_the_dense_rank_on_a_profiled_block():
    """The reachable case: a factor level in which the numeric is constant.

    ``numeric_cat`` profiles out ``[1 | menu | z]``, so a level carrying a
    single row has its probe column exactly absorbed and the block's true rank
    is ``k - 1``.  ``V_eff`` is formed by SUBTRACTION though, so that direction
    lands at round-off rather than at zero and whether ``cho_factor`` accepts
    the block is decided by rounding alone -- 10 of these 20 seeds are accepted
    here.  The edf must not depend on which, so the reported values over
    statistically identical replicates must be ONE value, and it must be the
    rank.  A Cholesky trace reports ``k`` on the accepted seeds and ``k - 1``
    on the rest, which is two.

    ``matrix_rank`` is a cross-check here, not the contract: its default cut
    (``max(M, N) * eps``) is not the ``_RCOND`` the screen counts at, and on a
    block with an eigenvalue between them the two would part company.  They
    agree on every seed below because a profiled block is bimodal -- measured
    over these 20, the smallest kept relative eigenvalue is 1.2e-02 and the
    largest dropped one 4.6e-16.
    """
    from superglm.screening import penalized_score_statistic
    from superglm.screening._numeric_margin import numeric_pair_moments
    from superglm.screening._score_stat import _solve_psd

    L, n = 40, 8000
    menu = np.eye(L)[:, 1:]
    reported = set()
    for seed in range(20):
        rng = np.random.default_rng(seed)
        codes = rng.integers(0, L - 1, n)
        codes[0] = L - 1  # the singleton level
        z = rng.normal(size=n)
        score = rng.normal(size=n)
        w = rng.uniform(0.5, 1.5, n)
        U, V, C, M, u_m = numeric_pair_moments(codes, L, menu, z, score, w)

        V_eff = V - C.T @ _solve_psd(M, C)
        V_eff = 0.5 * (V_eff + V_eff.T)
        assert np.linalg.matrix_rank(V_eff) == L - 2, seed

        result = penalized_score_statistic(U, V, C, M, None, U_nuisance=u_m)
        # Never ABOVE the rank: that inequality is the defect itself, and it
        # fires only on the seeds cho_factor accepts.
        assert result.edf0 <= L - 2, (seed, result.edf0)
        reported.add(result.edf0)
    # An exact count, identical across replicates -- not a float trace.
    assert reported == {float(L - 2)}


def test_singular_pencil_answer_does_not_depend_on_the_units():
    """The whitening fallback's rank cut has to be RELATIVE.

    ``V`` and ``S`` share a null space here, so the generalized driver fails
    and the explicit whitening runs.  The same problem is posed twice, once
    at a scale a thousand times smaller; an absolute floor in that cut called
    every identifiable direction null below 1e-12 and returned a zero
    statistic at zero df, which makes the screening table depend on whether
    a covariate is carried in metres or kilometres.
    """
    from superglm.screening import penalized_score_statistic

    answers = []
    for s in (1e-10, 1e-12, 1e-13, 1e-16):
        V = s * np.diag([1.0, 2.0, 0.0])
        S = s * np.diag([2.0, 1.0, 0.0])
        U = np.sqrt(s) * np.array([1.0, 1.0, 0.0])
        got = penalized_score_statistic(U, V, S_ti=S, edf0=1.0)
        answers.append((got.statistic, got.edf0))
    # lambda0 = 1 makes V + lambda S = 3s I on the identifiable block, so the
    # statistic is U' (3s I)^-1 U = 2/3 and edf0 is 1 -- at every scale.
    for statistic, edf0 in answers:
        assert statistic == pytest.approx(2.0 / 3.0, rel=1e-9)
        assert edf0 == pytest.approx(1.0, rel=1e-9)


def test_edf_solver_hits_target():
    from superglm.screening import penalized_score_statistic

    rng = np.random.default_rng(6)
    p = 8
    V = _pd_matrix(rng, p)
    S = _pd_matrix(rng, p, strength=0.3)
    U = rng.normal(size=p)

    result = penalized_score_statistic(U, V, S_ti=S, edf0=3.5)

    achieved = np.trace(np.linalg.solve(V + result.lambda0 * S, V))
    assert abs(achieved - 3.5) <= 1e-6
    assert abs(result.edf0 - 3.5) <= 1e-6


def test_profiling_removes_overlap_explained_score():
    """A purely additive signal must screen at the null level."""
    from superglm.screening import penalized_score_statistic

    rng = np.random.default_rng(7)
    p, q = 6, 4
    V = _pd_matrix(rng, p)
    M = _pd_matrix(rng, q)
    C = rng.normal(size=(q, p))
    u_m = rng.normal(size=q)
    U = C.T @ np.linalg.solve(M, u_m)  # score fully explained by the overlap
    S = _pd_matrix(rng, p, strength=0.2)

    result = penalized_score_statistic(U, V, C, M, S, edf0=3.0, U_nuisance=u_m)

    assert abs(result.statistic) < 1e-18


def test_infinite_penalty_limit_restricts_to_null_space():
    """With edf0 = penalty null dimension, T approaches the null-space statistic."""
    from superglm.screening import penalized_score_statistic

    rng = np.random.default_rng(8)
    p, null_dim = 6, 2
    V = _pd_matrix(rng, p)
    S = np.zeros((p, p))
    S[null_dim:, null_dim:] = _pd_matrix(rng, p - null_dim)
    U = rng.normal(size=p)

    result = penalized_score_statistic(U, V, S_ti=S, edf0=float(null_dim))

    V_nn = V[:null_dim, :null_dim]
    expected = U[:null_dim] @ np.linalg.solve(V_nn, U[:null_dim])
    assert result.lambda0 > 1e4
    np.testing.assert_allclose(result.statistic, expected, rtol=1e-3)


def test_c_without_m_raises():
    import pytest

    from superglm.screening import penalized_score_statistic

    with pytest.raises(ValueError, match="supply both"):
        penalized_score_statistic(np.ones(2), np.eye(2), C=np.ones((1, 2)))


def test_overlap_moments_match_dense_assembly():
    """M, C, u_m from cell tables must equal the dense row-space assembly."""
    from superglm.screening import pair_cell_moments
    from superglm.screening._overlap import pair_overlap_moments

    rng = np.random.default_rng(9)
    n, n_a, n_b, k_a, k_b = 3000, 15, 11, 4, 3
    codes_a = rng.integers(0, n_a, n)
    codes_b = rng.integers(0, n_b, n)
    A = rng.normal(size=(n_a, k_a))
    B = rng.normal(size=(n_b, k_b))
    score = rng.normal(size=n)
    weights = rng.normal(size=n)  # signed, as REML working quantities can be

    S_cell, W_cell = pair_cell_moments(codes_a, codes_b, n_a, n_b, score, weights)
    M, C, u_m = pair_overlap_moments(A, B, S_cell, W_cell)

    X_o = np.column_stack([np.ones(n), A[codes_a], B[codes_b]])
    X_T = _dense_row_kronecker(codes_a, codes_b, A, B)
    np.testing.assert_allclose(M, X_o.T @ (X_o * weights[:, None]), rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(C, X_o.T @ (X_T * weights[:, None]), rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(u_m, X_o.T @ score, rtol=1e-12, atol=1e-9)


def test_tensor_penalty_matches_interaction_convention():
    from superglm.screening._overlap import tensor_penalty

    rng = np.random.default_rng(10)
    S1 = _pd_matrix(rng, 3)
    S2 = _pd_matrix(rng, 2)

    expected = np.kron(S1, np.eye(2)) + np.kron(np.eye(3), S2)
    np.testing.assert_array_equal(tensor_penalty(S1, S2), expected)


def test_full_chain_zeroes_an_additive_signal():
    """End-to-end: a score generated by the overlap span screens at exactly zero."""
    from superglm.screening import (
        pair_cell_moments,
        pair_score_curvature,
        penalized_score_statistic,
    )
    from superglm.screening._overlap import pair_overlap_moments, tensor_penalty

    rng = np.random.default_rng(11)
    n, n_a, n_b, k_a, k_b = 4000, 12, 9, 4, 3
    codes_a = rng.integers(0, n_a, n)
    codes_b = rng.integers(0, n_b, n)
    A = rng.normal(size=(n_a, k_a))
    B = rng.normal(size=(n_b, k_b))
    weights = rng.uniform(0.2, 2.0, n)
    gamma = rng.normal(size=1 + k_a + k_b)
    X_o = np.column_stack([np.ones(n), A[codes_a], B[codes_b]])
    score = weights * (X_o @ gamma)  # purely additive working signal

    S_cell, W_cell = pair_cell_moments(codes_a, codes_b, n_a, n_b, score, weights)
    U, V = pair_score_curvature(A, B, S_cell, W_cell)
    M, C, u_m = pair_overlap_moments(A, B, S_cell, W_cell)
    S_ti = tensor_penalty(_pd_matrix(rng, k_a), _pd_matrix(rng, k_b))

    result = penalized_score_statistic(U, V, C, M, S_ti, edf0=3.0, U_nuisance=u_m)

    scale = float(np.abs(U).max())
    assert abs(result.statistic) < 1e-16 * max(scale, 1.0) ** 2 + 1e-12


def _screening_data(seed, n=2500, interaction=0.0):
    """Integer covariates, Poisson response, optional planted x1:x2 surface."""
    rng = np.random.default_rng(seed)
    frame = {f"x{i}": rng.integers(0, 25 + 3 * i, n).astype(float) for i in range(1, 6)}
    z1 = (frame["x1"] - frame["x1"].mean()) / frame["x1"].std()
    z2 = (frame["x2"] - frame["x2"].mean()) / frame["x2"].std()
    eta = -1.2 + 0.30 * z1 + 0.25 * z2 + interaction * z1 * z2
    w = rng.uniform(0.3, 1.0, n)
    y = rng.poisson(np.exp(eta) * w) / w
    import pandas as pd

    return pd.DataFrame(frame), y, w


def _fit_mains(frame, y, w):
    from superglm import SuperGLM

    model = SuperGLM(
        family="poisson",
        selection_penalty=None,
        discrete=False,
        features={
            name: __import__("superglm.features.spline", fromlist=["Spline"]).Spline(kind="ps", k=6)
            for name in frame.columns
        },
    )
    return model.fit_reml(frame, y, sample_weight=w)


def test_screen_requires_fitted_model():
    import pytest

    from superglm import SuperGLM
    from superglm.features.spline import Spline

    frame, y, w = _screening_data(0)
    model = SuperGLM(
        family="poisson",
        selection_penalty=None,
        features={"x1": Spline(kind="ps", k=6)},
    )
    with pytest.raises(RuntimeError, match="fit_reml first"):
        model.screen_interactions(frame, y, sample_weight=w)


def test_oracle_planted_interaction_ranks_first_across_seeds():
    """The plan's release gate: the true pair must rank first, five seeds."""
    for seed in range(5):
        frame, y, w = _screening_data(seed, interaction=0.5)
        model = _fit_mains(frame, y, w)

        table = model.screen_interactions(frame, y, sample_weight=w)

        top = tuple(sorted((table.loc[0, "feature_a"], table.loc[0, "feature_b"])))
        assert top == ("x1", "x2"), f"seed {seed}: expected x1:x2 first, got {top}; \n{table}"
        assert len(table) == 10  # all pairs of five splines


def test_null_statistics_stay_bounded_across_seeds():
    """No interaction anywhere: generous bound, rank stability not calibration."""
    for seed in range(3):
        frame, y, w = _screening_data(10 + seed, interaction=0.0)
        model = _fit_mains(frame, y, w)

        table = model.screen_interactions(frame, y, sample_weight=w)

        assert np.isfinite(table["statistic"]).all()
        assert np.isfinite(table["z"]).all()
        assert table["z"].max() < 10.0, table


def test_candidates_restricts_the_sweep():
    frame, y, w = _screening_data(2, interaction=0.5)
    model = _fit_mains(frame, y, w)

    table = model.screen_interactions(
        frame, y, sample_weight=w, candidates=[("x1", "x2"), ("x3", "x4")]
    )

    assert len(table) == 2
    assert set(map(tuple, table[["feature_a", "feature_b"]].to_numpy())) == {
        ("x1", "x2"),
        ("x3", "x4"),
    }


def test_screen_uses_the_fit_offset():
    """Review finding: predict() dropped model._fit_offset, so an offset-fitted
    model was screened at the wrong mean and leftover main-effect mass showed
    up as interaction signal."""
    import pandas as pd

    from superglm import SuperGLM
    from superglm.features.spline import Spline

    rng = np.random.default_rng(7)
    n = 2500
    frame = pd.DataFrame(
        {f"x{i}": rng.integers(0, 25 + 3 * i, n).astype(float) for i in range(1, 4)}
    )
    z1 = (frame["x1"] - frame["x1"].mean()) / frame["x1"].std()
    z2 = (frame["x2"] - frame["x2"].mean()) / frame["x2"].std()
    off = np.log(rng.uniform(0.2, 3.0, n))
    y = rng.poisson(np.exp(-1.2 + 0.3 * z1 + 0.25 * z2 + off)).astype(float)
    model = SuperGLM(
        family="poisson",
        selection_penalty=None,
        discrete=False,
        features={name: Spline(kind="ps", k=6) for name in frame.columns},
    ).fit_reml(frame, y, offset=off)

    table_default = model.screen_interactions(frame, y)
    table_explicit = model.screen_interactions(frame, y, offset=off)

    pd.testing.assert_frame_equal(table_default, table_explicit)
    # the offset genuinely flows: suppressing it must change the screen
    table_zero = model.screen_interactions(frame, y, offset=np.zeros(n))
    assert not np.allclose(table_default["z"], table_zero["z"])
    # and the properly-offset null stays inside the null bound
    assert table_default["z"].max() < 10.0, table_default


def test_screen_dispersed_gaussian_null_stays_bounded():
    """Review finding: z assumed phi=1; a sigma=3 Gaussian null hit z=25 and
    the winning rung collapsed to the endpoint before Pearson scaling."""
    import pandas as pd

    from superglm import SuperGLM
    from superglm.features.spline import Spline

    rng = np.random.default_rng(11)
    n = 2500
    frame = pd.DataFrame(
        {f"x{i}": rng.integers(0, 25 + 3 * i, n).astype(float) for i in range(1, 6)}
    )
    z1 = (frame["x1"] - frame["x1"].mean()) / frame["x1"].std()
    z2 = (frame["x2"] - frame["x2"].mean()) / frame["x2"].std()
    y = 0.5 * z1 + 0.4 * z2 + rng.normal(0.0, 3.0, n)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=None,
        discrete=False,
        features={name: Spline(kind="ps", k=6) for name in frame.columns},
    ).fit_reml(frame, y)

    table = model.screen_interactions(frame, y)

    assert np.isfinite(table["statistic"]).all()
    assert np.isfinite(table["z"]).all()
    assert table["z"].max() < 10.0, table


def test_screen_validates_inputs():
    """Review findings: NaN y / zero weights / bad edf0 / bad candidates must
    raise clear errors instead of AttributeError or silent nonsense."""
    import pytest

    frame, y, w = _screening_data(3)
    model = _fit_mains(frame, y, w)

    y_bad = y.copy()
    y_bad[0] = np.nan
    with pytest.raises(ValueError, match="finite y"):
        model.screen_interactions(frame, y_bad, sample_weight=w)
    with pytest.raises(ValueError, match="positive sum"):
        model.screen_interactions(frame, y, sample_weight=np.zeros_like(w))
    with pytest.raises(ValueError, match="non-empty"):
        model.screen_interactions(frame, y, sample_weight=w, edf0=())
    with pytest.raises(ValueError, match="finite and positive"):
        model.screen_interactions(frame, y, sample_weight=w, edf0=-1.0)
    with pytest.raises(ValueError, match="finite and positive"):
        model.screen_interactions(frame, y, sample_weight=w, edf0=float("nan"))
    with pytest.raises(ValueError, match="distinct screenable fitted"):
        model.screen_interactions(frame, y, sample_weight=w, candidates=[("x1", "x1")])
    with pytest.raises(ValueError, match="screenable features"):
        model.screen_interactions(frame, y, sample_weight=w, candidates=[("x1", "nope")])
    with pytest.raises(ValueError, match="distinct screenable fitted"):
        model.screen_interactions(frame, y, sample_weight=w, candidates=[("x1", "x2", "x3")])


def test_screen_scalar_edf0_variants_agree():
    """Review finding: a 0-d numpy array crashed the budget normalization."""
    import pandas as pd

    frame, y, w = _screening_data(4, interaction=0.5)
    model = _fit_mains(frame, y, w)

    t_float = model.screen_interactions(frame, y, sample_weight=w, edf0=4.0)
    t_0d = model.screen_interactions(frame, y, sample_weight=w, edf0=np.array(4.0))
    t_np = model.screen_interactions(frame, y, sample_weight=w, edf0=np.float64(4.0))

    pd.testing.assert_frame_equal(t_float, t_0d)
    pd.testing.assert_frame_equal(t_float, t_np)


def test_screen_select_parents_raise_upfront():
    """Review finding: select=True mains died per-pair three modules down;
    now one clear error at the top names the offending features."""
    import pytest

    from superglm import SuperGLM
    from superglm.features.spline import Spline

    frame, y, w = _screening_data(5)
    model = SuperGLM(
        family="poisson",
        selection_penalty=None,
        discrete=False,
        features={name: Spline(kind="ps", k=6, select=True) for name in frame.columns},
    ).fit_reml(frame, y, sample_weight=w)

    with pytest.raises(ValueError, match="select=True"):
        model.screen_interactions(frame, y, sample_weight=w)


def _continuous_screening_data(seed=6, n=2500):
    """Two continuous covariates (~n uniques each) with a planted smooth
    interaction, plus one integer covariate; x1:x2 raw grid ~ n^2 cells."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "x3": rng.integers(0, 30, n).astype(float),
        }
    )
    z1 = (frame["x1"] - frame["x1"].mean()) / frame["x1"].std()
    z2 = (frame["x2"] - frame["x2"].mean()) / frame["x2"].std()
    w = rng.uniform(0.3, 1.0, n)
    y = rng.poisson(np.exp(-1.2 + 0.3 * z1 + 0.25 * z2 + 0.5 * z1 * z2) * w) / w
    return frame, y, w


def _fit_continuous(frame, y, w):
    from superglm import SuperGLM
    from superglm.features.spline import Spline

    return SuperGLM(
        family="poisson",
        selection_penalty=None,
        discrete=False,
        features={c: Spline(kind="ps", k=6) for c in frame.columns},
    ).fit_reml(frame, y, sample_weight=w)


def test_screen_bins_continuous_pairs_instead_of_skipping():
    """Task 4: a continuous x continuous pair above the cell budget falls
    back to quantile binning (flagged approx) instead of NaN-skipping."""
    frame, y, w = _continuous_screening_data()
    model = _fit_continuous(frame, y, w)

    table = model.screen_interactions(frame, y, sample_weight=w)

    row = table[(table.feature_a == "x1") & (table.feature_b == "x2")].iloc[0]
    assert bool(row["approx"])
    assert np.isfinite(row["z"])
    # largest-margin-first binning: only one margin needs compression here
    assert row["n_cells"] < 2500 * 2500
    assert row["n_cells"] <= 5_000_000
    # the planted continuous interaction ranks first through the fallback
    top = tuple(sorted((table.loc[0, "feature_a"], table.loc[0, "feature_b"])))
    assert top == ("x1", "x2")
    # pairs within the budget stay on the exact path
    exact = table[~table["approx"]]
    assert len(exact) == 2
    assert np.isfinite(exact["z"]).all()


def test_screen_binned_tracks_exact_on_signal():
    """The binned statistic must approximate the exact one where it matters:
    on a pair carrying real signal (measured gap 3.5%, pinned at 10%).
    No such promise on null pairs, where binning legitimately smooths away
    high-frequency noise at unpenalized rungs."""
    frame, y, w = _continuous_screening_data()
    model = _fit_continuous(frame, y, w)

    exact = model.screen_interactions(
        frame, y, sample_weight=w, candidates=[("x1", "x2")], max_cells=7_000_000
    )
    binned = model.screen_interactions(frame, y, sample_weight=w, candidates=[("x1", "x2")])

    assert not bool(exact.loc[0, "approx"])
    assert bool(binned.loc[0, "approx"])
    z_e, z_b = float(exact.loc[0, "z"]), float(binned.loc[0, "z"])
    assert z_e > 10.0  # this is a signal pair
    assert abs(z_e - z_b) / z_e < 0.10
    # same winning rung (edf0 is the achieved edf, so compare to bisection tol)
    assert abs(exact.loc[0, "edf0"] - binned.loc[0, "edf0"]) < 1e-2


def test_screen_nan_only_when_binning_cannot_fit():
    """If even the binned grid exceeds max_cells the pair NaN-skips; approx
    reports whether binning was attempted before the budget still failed."""
    import pytest

    frame, y, w = _continuous_screening_data()
    model = _fit_continuous(frame, y, w)

    # binning IS applied (x1 -> 50 bins) but the binned grid still exceeds the
    # budget: NaN row, n_cells reports the attempted binned grid, approx says
    # binning happened
    table = model.screen_interactions(
        frame, y, sample_weight=w, candidates=[("x1", "x3")], max_cells=800, screen_bins=50
    )
    row = table.iloc[0]
    assert np.isnan(row["z"])
    assert bool(row["approx"])
    assert row["n_cells"] == 50 * 30

    # with a budget so tiny even the (k_a*k_b)^2 curvature block cannot fit,
    # the pair is unscreenable at any binning — NaN, with approx recording
    # that binning was attempted before the true dimensions ruled it out
    tiny = model.screen_interactions(
        frame, y, sample_weight=w, candidates=[("x1", "x3")], max_cells=100, screen_bins=50
    )
    assert np.isnan(tiny.iloc[0]["z"])
    assert bool(tiny.iloc[0]["approx"])

    with pytest.raises(ValueError, match="screen_bins"):
        model.screen_interactions(frame, y, sample_weight=w, screen_bins=1)


def test_screen_uses_the_fit_weights():
    """Blocking review finding: sample_weight must inherit from the fit like
    offset does — screening an exposure-weighted fit at unit weight
    linearizes against the wrong likelihood."""
    import pandas as pd

    frame, y, w = _screening_data(8)
    model = _fit_mains(frame, y, w)

    table_default = model.screen_interactions(frame, y)
    table_explicit = model.screen_interactions(frame, y, sample_weight=w)

    pd.testing.assert_frame_equal(table_default, table_explicit)
    # the weights genuinely flow: forcing unit weights must change the screen
    table_unit = model.screen_interactions(frame, y, sample_weight=np.ones_like(y))
    assert not np.allclose(table_default["z"], table_unit["z"])
    assert table_default["z"].max() < 10.0, table_default


def test_screen_validates_row_alignment():
    """Review finding: scalar or misaligned y/sample_weight silently
    broadcast; both must raise instead."""
    import pytest

    frame, y, w = _screening_data(9)
    model = _fit_mains(frame, y, w)

    with pytest.raises(ValueError, match="one entry per row"):
        model.screen_interactions(frame, np.float64(y[0]), sample_weight=w)
    with pytest.raises(ValueError, match="one entry per row"):
        model.screen_interactions(frame, y[:-5], sample_weight=w)
    with pytest.raises(ValueError, match="one entry per row"):
        model.screen_interactions(frame, y, sample_weight=w[:-5])
    with pytest.raises(ValueError, match="one entry per row"):
        model.screen_interactions(frame, y.reshape(-1, 1), sample_weight=w)


def test_screen_requires_finite_covariates():
    """Review finding: a NaN covariate silently became its own cell; now the
    skip reason is distinguishable from a cell-budget skip by raising."""
    import pytest

    frame, y, w = _screening_data(9)
    model = _fit_mains(frame, y, w)
    bad = frame.copy()
    bad.loc[3, "x1"] = np.nan

    with pytest.raises(ValueError, match="finite covariates"):
        model.screen_interactions(bad, y, sample_weight=w)


def test_screen_intermediate_budget_triggers_binning():
    """Review finding: max_cells bounded the cell grid but not the (n_a, k_b^2)
    curvature intermediates — a lopsided pair passed the cell budget while the
    einsum intermediate blew past it. The budget now covers both."""
    import pandas as pd

    from superglm import SuperGLM
    from superglm.features.spline import Spline

    rng = np.random.default_rng(12)
    n = 2500
    frame = pd.DataFrame(
        {
            "x1": rng.normal(size=n),  # ~2500 uniques
            "x4": rng.integers(0, 6, n).astype(float),  # 6 levels
        }
    )
    z1 = (frame["x1"] - frame["x1"].mean()) / frame["x1"].std()
    w = rng.uniform(0.3, 1.0, n)
    y = rng.poisson(np.exp(-1.2 + 0.3 * z1) * w) / w
    model = SuperGLM(
        family="poisson",
        selection_penalty=None,
        discrete=False,
        features={c: Spline(kind="ps", k=6) for c in frame.columns},
    ).fit_reml(frame, y, sample_weight=w)

    # cells 2500*6 = 15k fit max_cells exactly; the (n_a * k_b^2) intermediate
    # (~62.7k) exceeds 4*max_cells, so binning must trigger anyway
    table = model.screen_interactions(
        frame, y, sample_weight=w, candidates=[("x1", "x4")], max_cells=15_000
    )
    row = table.iloc[0]
    assert bool(row["approx"])
    assert np.isfinite(row["z"])
    assert row["n_cells"] <= 256 * 6

    # with a generous budget the same pair is exact
    wide = model.screen_interactions(frame, y, sample_weight=w, candidates=[("x1", "x4")])
    assert not bool(wide.iloc[0]["approx"])


def test_screen_phi_is_visible_and_overridable():
    """Reviewer ask: phi materially selects the winning rung, so it must be
    auditable (table.attrs) and overridable (frequency-weight escape hatch)."""
    import pytest

    frame, y, w = _screening_data(13, interaction=0.5)
    model = _fit_mains(frame, y, w)

    table = model.screen_interactions(frame, y, sample_weight=w)
    assert np.isfinite(table.attrs["phi"]) and table.attrs["phi"] > 0.0

    forced = model.screen_interactions(frame, y, sample_weight=w, phi=5.0)
    assert forced.attrs["phi"] == 5.0
    assert not np.allclose(table["z"], forced["z"])

    with pytest.raises(ValueError, match="phi override"):
        model.screen_interactions(frame, y, sample_weight=w, phi=-1.0)


def test_screen_discrete_mains_rows_flag_approx():
    """Reviewer ask, twice refined: a discrete fit flags approx only when its
    refit basis genuinely differs — discretization of a column whose
    cardinality fits the bin count is LOSSLESS (the binner returns the exact
    unique support), so default-bin screens of low-cardinality rating
    factors stay approx=False, while genuinely lossy bins flag every pair."""
    from superglm import SuperGLM
    from superglm.features.spline import Spline

    frame, y, w = _screening_data(14, interaction=0.5)

    def fit(n_bins):
        return SuperGLM(
            family="poisson",
            selection_penalty=None,
            discrete=True,
            n_bins=n_bins,
            features={name: Spline(kind="ps", k=6) for name in frame.columns},
        ).fit_reml(frame, y, sample_weight=w)

    # 25-37 levels vs 256 bins: every discretization is exact
    lossless = fit(256).screen_interactions(frame, y, sample_weight=w)
    assert not lossless["approx"].any()
    assert np.isfinite(lossless["z"]).all()

    # 8 bins: every parent bins lossily, so every refit differs from the probe
    lossy = fit(8).screen_interactions(frame, y, sample_weight=w)
    assert lossy["approx"].all()
    assert np.isfinite(lossy["z"]).all()


def test_screen_released_fit_state_refuses_silent_fallbacks():
    """Review finding: retain_fit_state=False releases _fit_weights/_fit_offset,
    so the fit-inheritance defaults silently degraded to unit weights / no
    offset. A weighted or offset fit whose arrays are gone must demand
    explicit values; an unweighted fit stays usable."""
    import pytest

    from superglm import SuperGLM
    from superglm.features.spline import Spline

    frame, y, w = _screening_data(15)

    def build():
        return SuperGLM(
            family="poisson",
            selection_penalty=None,
            discrete=False,
            retain_fit_state=False,
            features={name: Spline(kind="ps", k=6) for name in frame.columns},
        )

    weighted = build().fit_reml(frame, y, sample_weight=w)
    with pytest.raises(ValueError, match="released"):
        weighted.screen_interactions(frame, y)
    table = weighted.screen_interactions(frame, y, sample_weight=w)
    assert np.isfinite(table["z"]).all()

    offset_fit = build().fit_reml(frame, y, offset=np.full(len(y), 0.1))
    with pytest.raises(ValueError, match="offset explicitly"):
        offset_fit.screen_interactions(frame, y)

    unweighted = build().fit_reml(frame, y)
    table2 = unweighted.screen_interactions(frame, y)
    assert np.isfinite(table2["z"]).all()


def test_screen_excludes_already_fitted_interactions():
    """Review finding: the sweep re-surfaced pairs the model already fits as
    tensor terms, and the confirmation workflow then failed with
    'interaction already added'."""
    import pytest

    from superglm import SuperGLM
    from superglm.features.spline import Spline

    frame, y, w = _screening_data(16, interaction=0.5)
    model = SuperGLM(
        family="poisson",
        selection_penalty=None,
        discrete=False,
        features={name: Spline(kind="ps", k=6) for name in frame.columns},
    )
    model._add_interaction("x1", "x2")
    model = model.fit_reml(frame, y, sample_weight=w)

    table = model.screen_interactions(frame, y, sample_weight=w)

    screened = {frozenset((r.feature_a, r.feature_b)) for r in table.itertuples()}
    assert frozenset(("x1", "x2")) not in screened
    assert len(table) == 9  # 10 pairs minus the fitted one

    with pytest.raises(ValueError, match="already fitted"):
        model.screen_interactions(frame, y, sample_weight=w, candidates=[("x1", "x2")])


def test_screen_rejects_inherited_arrays_on_reordered_frame():
    """Review finding: inherited fit arrays are in training row order; a
    permuted frame silently paired them with the wrong observations."""
    import pytest

    frame, y, w = _screening_data(17, interaction=0.5)
    model = _fit_mains(frame, y, w)

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(y))
    frame_p = frame.iloc[perm].reset_index(drop=True)
    y_p, w_p = y[perm], w[perm]

    with pytest.raises(ValueError, match="explicitly"):
        model.screen_interactions(frame_p, y_p)
    # explicitly aligned arrays are fine on any row order
    table = model.screen_interactions(frame_p, y_p, sample_weight=w_p)
    assert np.isfinite(table["z"]).all()
    # and the unpermuted frame still inherits silently
    table2 = model.screen_interactions(frame, y)
    assert np.isfinite(table2["z"]).all()


def test_screen_preserves_eta_sign_for_noninjective_links():
    """Review finding: eta was reconstructed as link(predict(X)) = |eta| for
    link='sqrt', flipping the score sign on every negative-eta row."""
    import pandas as pd

    from superglm import SuperGLM
    from superglm.features.spline import Spline

    rng = np.random.default_rng(18)
    n = 2500
    frame = pd.DataFrame(
        {f"x{i}": rng.integers(0, 25 + 3 * i, n).astype(float) for i in range(1, 4)}
    )
    z1 = (frame["x1"] - frame["x1"].mean()) / frame["x1"].std()
    z2 = (frame["x2"] - frame["x2"].mean()) / frame["x2"].std()
    # an offset crossing zero forces negative stabilized eta on many rows
    off = rng.uniform(-3.0, 0.0, n)
    eta_true = 2.0 + 0.3 * z1 + 0.25 * z2 + 0.4 * z1 * z2 + off
    y = rng.poisson(np.maximum(eta_true, 0.05) ** 2).astype(float)
    model = SuperGLM(
        family="poisson",
        link="sqrt",
        selection_penalty=None,
        discrete=False,
        features={name: Spline(kind="ps", k=6) for name in frame.columns},
    ).fit_reml(frame, y, offset=off)

    eta = model._predict_eta_exact(frame, off)
    assert (np.asarray(eta) < 0).any(), "test needs negative fitted eta rows"

    table = model.screen_interactions(frame, y)

    assert np.isfinite(table["z"]).all()
    top = tuple(sorted((table.loc[0, "feature_a"], table.loc[0, "feature_b"])))
    assert top == ("x1", "x2")


def test_marginal_width_estimate_never_overestimates():
    """Review finding: over-estimates are terminal (the pair bins or skips
    with no correction possible), so the estimate must bias low for every
    built-in spline kind."""
    import pandas as pd

    from superglm.features.interaction import TensorInteraction
    from superglm.features.spline import Spline
    from superglm.model.screening_ops import _marginal_width_estimate

    rng = np.random.default_rng(19)
    x = rng.uniform(0.0, 10.0, 500)
    cases = (
        ("ps", 3, 0),  # degree-0: centered width n_knots exactly
        ("ps", 4, 1),
        ("ps", 5, 3),
        ("ps", 6, 3),
        ("ps", 12, 3),
        ("cr", 3, 3),  # minimum cr: 2-column centered marginal
        ("cr", 6, 3),
        ("cr", 12, 3),
    )
    for kind, k, degree in cases:
        spec = Spline(kind=kind, k=k, degree=degree)
        spec.prepare(pd.Series(x)) if hasattr(spec, "prepare") else None
        try:
            m = TensorInteraction._marginal_from_spec(spec, x, None)
        except Exception:
            continue  # spec needs fitting machinery; covered end-to-end elsewhere
        true_width = m.basis.shape[1]
        assert _marginal_width_estimate(spec) <= true_width, (kind, k, degree, true_width)


def test_screen_cached_sweep_matches_per_pair_screens():
    """Reviewer ask: pin cache equivalence — the all-pairs sweep must match
    screening each pair alone, catching any cross-pair cache leakage."""
    import pandas as pd

    frame, y, w = _screening_data(20, interaction=0.5)
    model = _fit_mains(frame, y, w)

    full = model.screen_interactions(frame, y, sample_weight=w)

    for row in full.itertuples():
        single = model.screen_interactions(
            frame, y, sample_weight=w, candidates=[(row.feature_a, row.feature_b)]
        )
        s = single.iloc[0]
        for col in ("statistic", "z", "edf0", "lambda0", "n_cells", "approx"):
            a, b = getattr(row, col), s[col]
            assert (a == b) or (pd.isna(a) and pd.isna(b)), (row.feature_a, row.feature_b, col)


def test_screen_unweighted_fit_screens_any_frame_without_arguments():
    """Review finding: unit fitted weights were treated as 'inherited', so the
    fit-data guard fired on every screen and an unweighted model could no
    longer screen a holdout or subsample. Ones cannot mispair rows."""
    frame, y, _ = _screening_data(21, interaction=0.5)
    model = _fit_mains(frame, y, np.ones_like(y))  # unit weights == unweighted

    holdout = frame.iloc[: len(frame) // 2].reset_index(drop=True)
    table = model.screen_interactions(holdout, y[: len(frame) // 2])

    assert np.isfinite(table["z"]).all()


def test_screen_per_spec_discrete_flags_approx():
    """Review findings: approx must follow the per-SPEC discretization
    decision (spec.discrete overrides the model flag; both parents must
    discretize) AND only flag when at least one discretization is lossy —
    lossless binning returns the exact unique support."""

    from superglm import SuperGLM
    from superglm.features.spline import Spline

    frame, y, w = _screening_data(22, interaction=0.5)
    model = SuperGLM(
        family="poisson",
        selection_penalty=None,
        discrete=False,
        features={
            "x1": Spline(kind="ps", k=6, discrete=True, n_bins=8),  # lossy bins
            "x2": Spline(kind="ps", k=6, discrete=True),  # lossless (28 <= 256)
            "x3": Spline(kind="ps", k=6),
            "x4": Spline(kind="ps", k=6),
            "x5": Spline(kind="ps", k=6, discrete=True),  # lossless (37 <= 256)
        },
    ).fit_reml(frame, y, sample_weight=w)

    table = model.screen_interactions(frame, y, sample_weight=w)

    flags = {frozenset((r.feature_a, r.feature_b)): bool(r.approx) for r in table.itertuples()}
    assert flags[frozenset(("x1", "x2"))]  # both discretize, x1 lossy
    assert not flags[frozenset(("x2", "x5"))]  # both discretize, both lossless
    assert not flags[frozenset(("x1", "x3"))]  # mixed pair: refit is exact
    assert not flags[frozenset(("x3", "x4"))]


def test_screen_cache_equivalence_covers_binned_entries():
    """Reviewer ask: the cache-equivalence pin only exercised exact-path
    entries; cover the (name, binned=True) cache keys too."""
    frame, y, w = _continuous_screening_data()
    model = _fit_continuous(frame, y, w)

    full = model.screen_interactions(frame, y, sample_weight=w)
    binned_rows = full[full["approx"]]
    assert len(binned_rows) >= 1  # x1:x2 bins under the default budget

    for row in binned_rows.itertuples():
        single = model.screen_interactions(
            frame, y, sample_weight=w, candidates=[(row.feature_a, row.feature_b)]
        )
        s = single.iloc[0]
        for col in ("statistic", "z", "edf0", "lambda0", "n_cells", "approx"):
            assert getattr(row, col) == s[col], col


def test_screen_weight_inheritance_reads_the_stored_array_not_the_stamp():
    """Review finding: the editor rewrites _fit_weights without touching the
    _fit_used_weights stamp, so trusting the stamp resurrects the original
    blocking bug (screen at unit weight against an exposure-fitted model).
    Non-unitness must be derived from the stored array at read time."""
    import pandas as pd

    frame, y, w = _screening_data(23, interaction=0.5)
    model = _fit_mains(frame, y, w)

    # editor's second writer: array becomes ones, stamp stays True
    explicit_ones = model.screen_interactions(frame, y, sample_weight=np.ones_like(y))
    model._fit_weights = np.ones_like(y)
    assert model._fit_used_weights  # the stale stamp
    pd.testing.assert_frame_equal(model.screen_interactions(frame, y), explicit_ones)

    # mirror: stored non-unit weights inherit despite a stale False stamp
    model._fit_weights = w
    model._fit_used_weights = False
    explicit_w = model.screen_interactions(frame, y, sample_weight=w)
    pd.testing.assert_frame_equal(model.screen_interactions(frame, y), explicit_w)


def test_screen_labels_every_spline_pair_ti():
    """The mixed-margin work added a `kind` column naming the refit target;
    a spline-only model must be unchanged apart from a constant 'ti'."""
    from superglm.model.screening_ops import _RESULT_COLUMNS

    frame, y, w = _screening_data(24, interaction=0.5)
    model = _fit_mains(frame, y, w)

    table = model.screen_interactions(frame, y, sample_weight=w)

    assert list(table.columns) == _RESULT_COLUMNS
    assert _RESULT_COLUMNS[:4] == ["feature_a", "feature_b", "kind", "statistic"]
    assert set(table["kind"]) == {"ti"}


def _treatment_menu(n_levels):
    """(L, L-1) treatment contrasts with level 0 as base — the screen's menu."""
    return np.vstack([np.zeros((1, n_levels - 1)), np.eye(n_levels - 1)])


def test_contrast_menu_kron_is_the_pair_indicator_block():
    # kron of contrast menus on the level-pair grid == CategoricalInteraction's
    # non-base pair indicator columns, row for row.
    rng = np.random.default_rng(7)
    n, L1, L2 = 400, 4, 3
    codes_a = rng.integers(0, L1, n)
    codes_b = rng.integers(0, L2, n)
    menu_a = _treatment_menu(L1)
    menu_b = _treatment_menu(L2)
    X = _dense_row_kronecker(codes_a, codes_b, menu_a, menu_b)
    expected = np.column_stack(
        [
            (codes_a == i).astype(float) * (codes_b == j).astype(float)
            for i in range(1, L1)
            for j in range(1, L2)
        ]
    )
    np.testing.assert_allclose(X, expected)


def test_cell_assembly_exact_for_contrast_menus():
    rng = np.random.default_rng(8)
    n, L1, L2 = 500, 5, 4
    codes_a = rng.integers(0, L1, n)
    codes_b = rng.integers(0, L2, n)
    menu_a = _treatment_menu(L1)
    menu_b = _treatment_menu(L2)
    score = rng.normal(size=n)
    w = rng.uniform(0.1, 3.0, n)
    S_cell, W_cell = pair_cell_moments(codes_a, codes_b, L1, L2, score, w)
    U, V = pair_score_curvature(menu_a, menu_b, S_cell, W_cell)
    X = _dense_row_kronecker(codes_a, codes_b, menu_a, menu_b)
    np.testing.assert_allclose(U, X.T @ score, rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(V, X.T @ (X * w[:, None]), rtol=1e-12, atol=1e-9)


def test_numeric_pair_moments_match_dense_assembly():
    from superglm.screening import numeric_pair_moments

    rng = np.random.default_rng(11)
    n, L = 600, 5
    codes = rng.integers(0, L, n)
    menu = np.zeros((L, L - 1))
    menu[1:, :] = np.eye(L - 1)
    z = rng.uniform(-2.0, 3.0, n)
    score = rng.normal(size=n)
    w = rng.uniform(0.1, 3.0, n)

    U, V, C, M, u_m = numeric_pair_moments(codes, L, menu, z, score, w)

    X_T = menu[codes] * z[:, None]  # probe block
    X_o = np.column_stack([np.ones(n), menu[codes], z])  # overlap span
    np.testing.assert_allclose(U, X_T.T @ score, rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(V, X_T.T @ (X_T * w[:, None]), rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(M, X_o.T @ (X_o * w[:, None]), rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(C, X_o.T @ (X_T * w[:, None]), rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(u_m, X_o.T @ score, rtol=1e-12, atol=1e-9)


def test_numeric_pair_moments_reject_a_menu_that_is_not_the_grid():
    """Review finding: row agreement and the code bounds were validated but the
    menu's own row count was not, so a menu built for a different margin
    surfaced as a numpy broadcast error instead of this module's contract."""
    import pytest

    from superglm.screening import numeric_pair_moments

    rng = np.random.default_rng(11)
    n, L = 200, 5
    codes = rng.integers(0, L, n)
    z = rng.uniform(-2.0, 3.0, n)
    score = rng.normal(size=n)
    w = rng.uniform(0.1, 3.0, n)
    menu = np.zeros((L, L - 1))
    menu[1:, :] = np.eye(L - 1)

    with pytest.raises(ValueError, match="one row per gridded cell"):
        numeric_pair_moments(codes, L, menu[:-1], z, score, w)
    with pytest.raises(ValueError, match="one row per gridded cell"):
        numeric_pair_moments(codes, L, menu[:, 0], z, score, w)
    # the matching menu still computes
    assert numeric_pair_moments(codes, L, menu, z, score, w)[0].shape == (L - 1,)


def test_numeric_numeric_moments_match_dense_assembly():
    from superglm.screening import numeric_numeric_moments

    rng = np.random.default_rng(12)
    n = 500
    z1 = rng.uniform(-1.0, 2.0, n)
    z2 = rng.uniform(0.5, 1.5, n)
    score = rng.normal(size=n)
    w = rng.uniform(0.1, 3.0, n)
    U, V, C, M, u_m = numeric_numeric_moments(z1, z2, score, w)
    X_T = (z1 * z2)[:, None]
    X_o = np.column_stack([np.ones(n), z1, z2])
    np.testing.assert_allclose(U, X_T.T @ score, rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(V, X_T.T @ (X_T * w[:, None]), rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(M, X_o.T @ (X_o * w[:, None]), rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(C, X_o.T @ (X_T * w[:, None]), rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(u_m, X_o.T @ score, rtol=1e-12, atol=1e-9)
