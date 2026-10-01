"""The ``fs`` leaf factor (one-engine design §3.4, §3.6): rows, not moments.

Dense references are float64 on well-conditioned rows; the pivot tests at
penalty-pinned directions use exact rational references (``fractions``) and
the square-root form's first-order bound.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest
import scipy.interpolate

from superglm.solvers._structured.block_leaves import (
    FactorSmoothLeafFactor,
    FactorSmoothPenalizedOperator,
    ProfiledFactorSmoothLeafFactor,
)
from superglm.solvers.structured import (
    BlockSymmetricOperator,
    CenteredBlockOperator,
    LowRankSymmetricOperator,
    _block_operator_bdlr,
    build_penalized_block_operator,
    materialize_compact_operator,
)
from superglm.types import PenaltyComponent
from tests._leaf_systems import leaf_system_from_rows

_U = 2.0**-53


def _gamma(count: float) -> float:
    return count * _U / (1.0 - count * _U)


def _design(levels, Z, X):
    """``[1 X Z]`` with each row's ``Z`` in its level's block, the factor's global order."""
    n, k = Z.shape
    q = X.shape[1]
    design = np.zeros((n, 1 + q + (int(levels.max()) + 1) * k))
    design[:, 0] = 1.0
    design[:, 1 : 1 + q] = X
    for row, level in enumerate(levels):
        design[row, 1 + q + level * k : 1 + q + (level + 1) * k] = Z[row]
    return design


def _agreement(dense, levels, Z, X, w, local, small) -> tuple[float, float]:
    """``(relative, logdet)``: what two backward-stable factorizations of ``H`` agree to.

    Each factors ``H + dH`` with ``||dH||_2 <= eta ||H||_2``: a Cholesky-class
    factor of the rows has ``|dH| <= gamma_{n+p+1} (|D|'|W||D| + |S|)``
    entrywise (Higham 2002, Theorems 10.3 and 19.4, the row accumulations
    included) and ``||.||_2 <= p ||.||_max``.  A solve, an inverse entry and
    ``log det`` then move by at most ``kappa eta / (1 - kappa eta)`` relative
    to ``||x||_2``, ``||H^-1||_2`` (Higham 2002, Theorem 7.2 and section
    14.1) and ``p kappa eta`` absolute (first order of ``tr(H^-1 dH)``); two
    factorizations by twice that.  ``|W|`` keeps signed rows' cancellation in
    the bound.
    """
    p = dense.shape[0]
    design = np.abs(_design(levels, Z, X))
    magnitude = design.T @ (np.abs(w)[:, None] * design)
    q = X.shape[1]
    magnitude[1 : 1 + q, 1 : 1 + q] += np.abs(small)
    k = Z.shape[1]
    for level in range(local.shape[0]):
        start = 1 + q + level * k
        magnitude[start : start + k, start : start + k] += np.abs(local[level])
    eta = p * _gamma(len(w) + p + 1) * np.max(magnitude) / np.linalg.norm(dense, 2)
    kappa_eta = float(np.linalg.cond(dense) * eta)
    assert kappa_eta < 0.5
    return 2.0 * kappa_eta / (1.0 - kappa_eta), 2.0 * p * kappa_eta / (1.0 - kappa_eta)


def _rows(rng, *, n_levels=5, block_size=3, border=4, per_level=9, signed=False):
    levels = np.repeat(np.arange(n_levels), per_level)
    n = len(levels)
    Z = rng.normal(size=(n, block_size))
    X = rng.normal(size=(n, border))
    w = rng.uniform(0.4, 2.0, size=n)
    if signed:
        w = np.where(rng.uniform(size=n) < 0.15, -0.3 * w, w)
    return levels, Z, X, w, rng.normal(size=n)


def _factor(rng, *, signed=False, n_levels=5, block_size=3, border=4, penalty_scale=1.0, **kw):
    levels, Z, X, w, wz = _rows(
        rng, n_levels=n_levels, block_size=block_size, border=border, signed=signed
    )
    system = leaf_system_from_rows(
        Z,
        X,
        levels,
        w,
        wz,
        n_levels=n_levels,
        small_indices=np.arange(border),
        structured_indices=np.arange(border, border + n_levels * block_size).reshape(
            n_levels, block_size
        ),
        signed=signed,
    )
    roots = rng.normal(size=(n_levels, block_size, block_size))
    local = penalty_scale * (np.einsum("kji,kjl->kil", roots, roots) + 1.5 * np.eye(block_size))
    small = 0.4 * np.eye(border)
    penalized = FactorSmoothPenalizedOperator.with_penalties(system.operator, small, local)
    factor = FactorSmoothLeafFactor(system, penalized, **kw)
    dense = _dense(levels, Z, X, w, local, small)
    return factor, system, dense, (levels, Z, X, w, wz, local, small)


def _dense(levels, Z, X, w, local, small):
    """The augmented Hessian ``[1 X Z]' W [1 X Z] + S`` in the factor's global order."""
    n, k = Z.shape
    q = X.shape[1]
    K = local.shape[0]
    design = np.zeros((n, 1 + q + K * k))
    design[:, 0] = 1.0
    design[:, 1 : 1 + q] = X
    for row, level in enumerate(levels):
        start = 1 + q + level * k
        design[row, start : start + k] = Z[row]
    H = design.T @ (w[:, None] * design)
    H[1 : 1 + q, 1 : 1 + q] += small
    for level in range(K):
        start = 1 + q + level * k
        H[start : start + k, start : start + k] += local[level]
    return 0.5 * (H + H.T)


@pytest.mark.parametrize("signed", [False, True])
def test_leaf_factor_solve_logdet_and_selected_inverse_match_dense(signed) -> None:
    """Against LAPACK on the dense ``H``, within what two factorizations agree to (``_agreement``)."""
    rng = np.random.default_rng(731)
    factor, _, dense, (levels, Z, X, w, _, local, small) = _factor(rng, signed=signed)
    relative, logdet = _agreement(dense, levels, Z, X, w, local, small)
    inverse = np.linalg.inv(dense)
    rhs = rng.normal(size=dense.shape[0])
    multi = rng.normal(size=(dense.shape[0], 4))
    expected = inverse @ rhs
    np.testing.assert_allclose(
        factor.solve(rhs), expected, rtol=0.0, atol=relative * np.linalg.norm(expected)
    )
    expected = inverse @ multi
    error = np.abs(factor.solve(multi) - expected)
    assert np.all(error <= relative * np.linalg.norm(expected, axis=0)[None, :])
    assert abs(factor.logdet() - np.linalg.slogdet(dense)[1]) <= logdet
    assert factor.rank == dense.shape[0]
    selected = np.array([0, 2, 4, 7, 13], dtype=np.intp)
    entry = relative * np.linalg.norm(inverse, 2)
    np.testing.assert_allclose(
        factor.selected_inverse_block(selected),
        inverse[np.ix_(selected, selected)],
        rtol=0.0,
        atol=entry,
    )
    np.testing.assert_allclose(
        factor.selected_inverse_diagonal(selected), np.diag(inverse)[selected], rtol=0.0, atol=entry
    )


def test_leaf_factor_data_solve_is_the_normal_equations_solution() -> None:
    """``solve_data`` reads the right-hand side that travelled inside the leaf QR.

    Within ``_agreement``'s solve bound plus the right-hand side's own: the
    reference forms ``D'Wz`` in ``n``-term sums, within ``gamma_n |D|'|Wz|``
    (Higham 2002, eq. 3.5), and ``H^-1`` carries it at ``||H^-1||_2``; the
    factor's travels with its rows, likewise.
    """
    rng = np.random.default_rng(732)
    factor, _, dense, (levels, Z, X, w, wz, local, small) = _factor(rng)
    relative, _ = _agreement(dense, levels, Z, X, w, local, small)
    design = _design(levels, Z, X)
    expected = np.linalg.solve(dense, design.T @ wz)
    data = _gamma(len(w)) * np.linalg.norm(np.abs(design).T @ np.abs(wz))
    tolerance = (
        relative * np.linalg.norm(expected) + 2.0 * np.linalg.norm(np.linalg.inv(dense), 2) * data
    )
    np.testing.assert_allclose(factor.solve_data(), expected, rtol=0.0, atol=tolerance)
    centred = factor.solve_data(centred=True)
    np.testing.assert_allclose(centred[1:], expected[1:], rtol=0.0, atol=tolerance)


def _components(factor) -> tuple[PenaltyComponent, PenaltyComponent]:
    start = int(factor.structured_indices.min())
    stop = int(factor.structured_indices.max()) + 1
    omega = np.diag(np.linspace(1.4, 0.0, factor.block_size))
    repeated = PenaltyComponent(
        name="fs:wiggle",
        group_name="fs",
        group_index=1,
        group_sl=slice(start, stop),
        omega_raw=omega,
        omega_ssp=omega,
        rank=float(factor.n_levels * (factor.block_size - 1)),
        penalty_kind="repeated",
        repeat_count=factor.n_levels,
        block_width=factor.block_size,
    )
    first = 1 if len(factor.small_indices) and factor.small_indices[0] == 0 else 0
    width = len(factor.small_indices) - first
    dense_small = PenaltyComponent(
        name="small",
        group_name="small",
        group_index=0,
        group_sl=slice(first, first + width),
        omega_raw=np.diag(np.linspace(0.4, 1.0, width)),
        omega_ssp=np.diag(np.linspace(0.4, 1.0, width)),
        rank=float(width),
    )
    return repeated, dense_small


def _expanded(component: PenaltyComponent, width: int) -> np.ndarray:
    result = np.zeros((width, width))
    if component.penalty_kind == "repeated":
        block = np.kron(np.eye(component.repeat_count), component.omega_ssp)
        result[component.group_sl, component.group_sl] = block
    else:
        result[component.group_sl, component.group_sl] = component.omega_ssp
    return result


def test_leaf_factor_penalty_traces_match_dense() -> None:
    rng = np.random.default_rng(733)
    factor, _, dense, _ = _factor(rng)
    inverse = np.linalg.inv(dense)
    repeated, small = _components(factor)
    for component in (repeated, small):
        assert factor.trace_inverse_penalty(component) == pytest.approx(
            np.trace(inverse @ _expanded(component, dense.shape[0])), abs=3e-11
        )
    for left in (repeated, small):
        for right in (repeated, small):
            expected = (
                1.2
                * 0.7
                * np.trace(
                    inverse
                    @ _expanded(left, dense.shape[0])
                    @ inverse
                    @ _expanded(right, dense.shape[0])
                )
            )
            assert factor.penalty_cross_trace(left, right, 1.2, 0.7) == pytest.approx(
                expected, abs=3e-11
            )


def test_leaf_factor_operator_protocol_matches_dense() -> None:
    rng = np.random.default_rng(734)
    factor, system, dense, _ = _factor(rng)
    inverse = np.linalg.inv(dense)
    p = dense.shape[0]
    raw = BlockSymmetricOperator(
        A=np.pad(system.operator.A, ((1, 0), (1, 0))),
        C=np.pad(system.operator.C, ((0, 0), (0, 0), (1, 0))),
        D=system.operator.D,
        small_indices=factor.small_indices,
        structured_indices=factor.structured_indices,
    )
    core = rng.normal(size=(2, 2))
    low_rank = LowRankSymmetricOperator(basis=rng.normal(size=(p, 2)), core=0.5 * (core + core.T))
    centered = CenteredBlockOperator(
        raw=raw, cross=rng.normal(size=p), total=1.7, center=rng.normal(size=p)
    )
    repeated, _ = _components(factor)
    for compact in (raw, low_rank, centered):
        product = inverse @ materialize_compact_operator(compact)
        assert factor.trace_inverse_operator(compact) == pytest.approx(np.trace(product), abs=3e-11)
        np.testing.assert_allclose(
            factor.inverse_operator_diagonal(compact), np.diag(product), atol=3e-11
        )
        np.testing.assert_allclose(
            factor.inverse_operator_square_diagonal(compact), np.diag(product @ product), atol=5e-10
        )
    expected = np.trace(
        inverse
        @ materialize_compact_operator(centered)
        @ inverse
        @ materialize_compact_operator(low_rank)
    )
    assert factor.operator_cross_trace(centered, low_rank) == pytest.approx(expected, abs=5e-10)
    expected = np.trace(
        inverse @ (1.3 * _expanded(repeated, p)) @ inverse @ materialize_compact_operator(centered)
    )
    assert factor.penalty_operator_cross_trace(repeated, 1.3, centered) == pytest.approx(
        expected, abs=5e-10
    )


def test_profiled_leaf_factor_matches_the_slope_block_and_its_edf_identity() -> None:
    """``M_ss`` from the centred factor, and ``edf`` through ``diag(H^+ (H - S))``."""
    rng = np.random.default_rng(735)
    factor, system, dense, _ = _factor(rng, n_levels=4, block_size=3, border=3)
    xtw = np.empty(system.operator.shape[0])
    xtw[system.operator.small_indices] = system.xtw_small
    xtw[system.operator.structured_indices] = system.xtw_structured
    profiled = ProfiledFactorSmoothLeafFactor(augmented_factor=factor, sum_w=system.sum_w, xtw=xtw)
    expected = np.linalg.inv(dense)[1:, 1:]
    rhs = rng.normal(size=profiled.shape[0])
    np.testing.assert_allclose(profiled.solve(rhs), expected @ rhs, atol=3e-12)
    selected = np.array([0, 3, 8], dtype=np.intp)
    np.testing.assert_allclose(
        profiled.selected_inverse_block(selected), expected[np.ix_(selected, selected)], atol=3e-12
    )
    np.testing.assert_allclose(
        profiled.selected_inverse_diagonal(selected), np.diag(expected)[selected], atol=3e-12
    )
    assert profiled.logdet() == pytest.approx(factor.logdet() - np.log(system.sum_w), abs=2e-12)
    repeated, small = _components(profiled)
    for component in (repeated, small):
        assert profiled.trace_inverse_penalty(component) == pytest.approx(
            np.trace(expected @ _expanded(component, profiled.shape[0])), abs=3e-11
        )
    # the factor's own centred data operator: the identity route
    mean = xtw / system.sum_w
    own = CenteredBlockOperator(raw=system.operator, cross=xtw, total=system.sum_w, center=mean)
    product = expected @ materialize_compact_operator(own)
    assert profiled.trace_inverse_operator(own) == pytest.approx(np.trace(product), abs=3e-11)
    np.testing.assert_allclose(
        profiled.inverse_operator_diagonal(own), np.diag(product), atol=3e-11
    )
    np.testing.assert_allclose(
        profiled.inverse_operator_square_diagonal(own), np.diag(product @ product), atol=5e-10
    )


def _window_level(rng, *, n=40, k=6, lam=1e-9, weight=1e4, signed=False):
    """One level's rows in a 1e-3 window of a cubic B-spline basis: only lambda pins
    most of its directions (the design's §4.10 probe shape)."""
    knots = np.concatenate(([0.0] * 4, np.linspace(0.0, 1.0, k - 2)[1:-1], [1.0] * 4))
    x = 0.5 + 1e-3 * rng.uniform(size=n)
    Z = scipy.interpolate.BSpline.design_matrix(x, knots, 3).toarray()
    w = weight * rng.uniform(0.5, 1.5, size=n)
    if signed:
        w = np.where(rng.uniform(size=n) < 0.13, -0.2 * w, w)
    P = lam * np.diag(np.linspace(1.0, 2.0, k))
    return Z, w, P


def _exact_logdet(Z, w, P) -> float:
    """``log det(Z' W Z + P)`` over the float64 rows as exact rationals; ``nan`` unless positive."""
    k = Z.shape[1]
    D = [[Fraction(P[i, j]) for j in range(k)] for i in range(k)]
    for row, weight in zip(Z, w, strict=True):
        f = Fraction(weight)
        values = [Fraction(v) for v in row]
        for i in range(k):
            if values[i]:
                for j in range(k):
                    D[i][j] += f * values[i] * values[j]
    total = 0.0
    for c in range(k):
        pivot = D[c][c]
        if pivot <= 0:
            return float("nan")
        total += math.log(pivot.numerator) - math.log(pivot.denominator)
        for r in range(c + 1, k):
            factor = D[r][c] / pivot
            for j in range(c, k):
                D[r][j] -= factor * D[c][j]
    return total


def _pivot_bound(Z, w, P, *, signed: bool) -> float:
    """First-order bound on ``log|D|`` from the square-root route (module docstring of block_leaves).

    Row-level errors ``||dA_j|| <= g ||A_j||`` on ``A = [sqrt|w| Z; P^1/2]``
    give ``|tr(D^-1 A'J dA)| <= g sum_j sqrt(G_jj (D^-1 G D^-1)_jj)`` with the
    unsigned ``G = A'A``; signed rows add the middle factor's ``||dM|| ||A
    D^-1 A'||_*``, at most ``g sqrt(sum G_jj sum (D^-1 G D^-1)_jj)`` times its
    order ``k``.  ``g`` sums the level QR, the per-lambda QR, the pseudo-rows
    and the eigendecomposition, each ``2 gamma_{2 m p}`` or smaller (Higham
    2002, Theorem 19.4 with its constant taken as 2): ``10 gamma_{2 m p}``.
    """
    n, k = Z.shape
    G = Z.T @ (np.abs(w)[:, None] * Z) + P
    D = Z.T @ (w[:, None] * Z) + P
    inverse = np.linalg.inv(D)
    sandwich = np.diag(inverse @ G @ inverse)
    g = 10.0 * _gamma(2.0 * (n + 2 * k) * (k + 2))
    bound = g * float(np.sum(np.sqrt(np.diag(G) * np.abs(sandwich))))
    if signed:
        bound += g * k * math.sqrt(float(np.sum(np.diag(G))) * float(np.sum(np.abs(sandwich))))
    return 2.0 * bound


@pytest.mark.parametrize("signed", [False, True])
def test_level_pivots_come_from_the_rows_not_the_moments(signed) -> None:
    """Penalty-pinned directions: each level's ``log|D_l|`` against the exact rational one.

    A Cholesky of the moments ``Z'WZ + P`` (or, for signed rows, of ``R'MR +
    P``) loses ``eps kappa(D_l)`` there, about 1e-1 at lambda 1e-9 beside
    weights 1e4 (design §4.10); the leaf route stays inside the square-root
    bound, which scales with ``sqrt(kappa)``.
    """
    rng = np.random.default_rng(4100 + int(signed))
    K, k = 3, 6
    blocks = [_window_level(rng, k=k, signed=signed) for _ in range(K)]
    exact = [_exact_logdet(Z, w, P) for Z, w, P in blocks]
    assert all(np.isfinite(exact))
    Z = np.vstack([Z for Z, _, _ in blocks])
    w = np.concatenate([w for _, w, _ in blocks])
    levels = np.repeat(np.arange(K), [len(b[1]) for b in blocks])
    X = rng.normal(size=(len(w), 2))
    system = leaf_system_from_rows(
        Z,
        X,
        levels,
        w,
        np.zeros(len(w)),
        n_levels=K,
        small_indices=np.arange(2),
        structured_indices=np.arange(2, 2 + K * k).reshape(K, k),
        signed=signed,
    )
    local = np.stack([P for _, _, P in blocks])
    penalized = FactorSmoothPenalizedOperator.with_penalties(system.operator, np.eye(2), local)
    factor = FactorSmoothLeafFactor(system, penalized)
    for level, (Zl, wl, Pl) in enumerate(blocks):
        error = abs(factor.level_logdets[level] - exact[level])
        assert error <= _pivot_bound(Zl, wl, Pl, signed=signed), (level, error)


def test_non_finite_rows_are_refused_as_curvature() -> None:
    rng = np.random.default_rng(736)
    levels, Z, X, w, wz = _rows(rng)
    X[3, 1] = np.inf
    with pytest.raises(np.linalg.LinAlgError, match="non-finite"):
        leaf_system_from_rows(
            Z,
            X,
            levels,
            w,
            wz,
            n_levels=5,
            small_indices=np.arange(4),
            structured_indices=np.arange(4, 19).reshape(5, 3),
        )


def test_fisher_rows_refuse_a_negative_weight_by_contract() -> None:
    rng = np.random.default_rng(737)
    levels, Z, X, w, wz = _rows(rng)
    w[0] = -1.0
    with pytest.raises(ValueError, match="non-negative"):
        leaf_system_from_rows(
            Z,
            X,
            levels,
            w,
            wz,
            n_levels=5,
            small_indices=np.arange(4),
            structured_indices=np.arange(4, 19).reshape(5, 3),
        )


def test_an_exact_duplicate_border_column_is_truncated_and_disclosed_not_refused() -> None:
    """The retired moment factor refused a coupled null; the §3.6 border certifies it."""
    rng = np.random.default_rng(738)
    levels, Z, X, w, wz = _rows(rng)
    X[:, 3] = X[:, 2]
    system = leaf_system_from_rows(
        Z,
        X,
        levels,
        w,
        wz,
        n_levels=5,
        small_indices=np.arange(4),
        structured_indices=np.arange(4, 19).reshape(5, 3),
    )
    local = np.broadcast_to(np.eye(3), (5, 3, 3)).copy()
    penalized = FactorSmoothPenalizedOperator.with_penalties(
        system.operator, np.zeros((4, 4)), local
    )
    factor = FactorSmoothLeafFactor(system, penalized)
    assert factor.rank == factor.shape[0] - 1
    assert factor.rank_truncated
    assert factor.border_certificate.rank == 3


def test_leaf_factor_refuses_large_structured_inverse_materialization() -> None:
    rng = np.random.default_rng(739)
    factor, _, _, _ = _factor(
        rng, n_levels=20, block_size=4, border=2, max_structured_inverse_block=16
    )
    with pytest.raises(ValueError, match="Refusing to materialize"):
        factor.selected_inverse_block(factor.structured_indices.ravel())


def test_block_operator_bdlr_drops_structural_zero_low_rank_parts() -> None:
    rng = np.random.default_rng(740)
    _, system, _, _ = _factor(rng)
    operator = system.operator
    pure_blocks = BlockSymmetricOperator(
        A=np.zeros_like(operator.A),
        C=np.zeros_like(operator.C),
        D=operator.D,
        small_indices=operator.small_indices,
        structured_indices=operator.structured_indices,
    )
    small_only = BlockSymmetricOperator(
        A=operator.A,
        C=np.zeros_like(operator.C),
        D=np.zeros_like(operator.D),
        small_indices=operator.small_indices,
        structured_indices=operator.structured_indices,
    )
    block_repr = _block_operator_bdlr(pure_blocks)
    small_repr = _block_operator_bdlr(small_only)
    assert block_repr.basis.shape[1] == 0
    assert block_repr.core.shape == (0, 0)
    assert small_repr.basis.shape[1] == len(operator.small_indices)
    assert small_repr.core.shape == operator.A.shape


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_a_non_finite_fs_operator_block_is_refused_as_curvature(bad) -> None:
    rng = np.random.default_rng(741)
    _, system, _, _ = _factor(rng)
    operator = system.operator
    D = np.array(operator.D, dtype=np.float64)
    D[1, 0, 0] = bad
    with pytest.raises(np.linalg.LinAlgError, match="must be finite"):
        BlockSymmetricOperator(
            A=operator.A,
            C=operator.C,
            D=D,
            small_indices=operator.small_indices,
            structured_indices=operator.structured_indices,
        )


def test_block_override_cross_validation_uses_participating_coordinate_scale() -> None:
    rng = np.random.default_rng(742)
    _, system, _, _ = _factor(rng)
    operator = system.operator
    penalty = np.zeros(operator.shape)
    flat_structured = operator.structured_indices.ravel()
    penalty[operator.small_indices, operator.small_indices] = 1.0
    penalty[operator.small_indices[0], operator.small_indices[0]] = 1.0e12
    penalty[flat_structured, flat_structured] = 1.0
    penalty[flat_structured[0], operator.small_indices[1]] = 1.0e-3
    penalty[operator.small_indices[1], flat_structured[0]] = 1.0e-3
    with pytest.raises(ValueError, match="couples the dominant and dense-small blocks"):
        build_penalized_block_operator(system, [], [], 0.0, S_override=penalty)


def test_a_weakly_identified_border_slope_is_left_out_of_the_fs_laplace_term() -> None:
    """§3.9 on the fs leaf factor: ``reml.identified`` rebuilds it with the column excluded.

    ``xt`` is 5 except on two rows of prior weight 1e-15, so its information is
    at noise level.  Before stage 2 the fs block factor was a family the
    identified Laplace could not restrict (``reml_laplace_exclusion_unsupported``
    counted every evaluation); the leaf factor is rebuilt with ``xt`` out.
    """
    import warnings

    import pandas as pd

    from superglm import FactorSmooth, Numeric, SuperGLM
    from superglm.reml.identified import WeakIdentificationWarning

    rng = np.random.default_rng(4242)
    n, K = 600, 5
    frame = pd.DataFrame(
        {
            "x": rng.uniform(size=n),
            "x1": rng.normal(size=n),
            "g": [f"g{c}" for c in rng.integers(0, K, n)],
            "xt": 5.0,
        }
    )
    frame.loc[[0, 1], "xt"] = [6.0, 7.0]
    weight = np.ones(n)
    weight[[0, 1]] = 1e-15
    y = rng.poisson(np.exp(0.3 * frame["x1"].to_numpy() + np.sin(3 * frame["x"].to_numpy())))
    model = SuperGLM(
        family="poisson",
        features={"x1": Numeric(), "xt": Numeric()},
        interactions=[FactorSmooth("x", group="g", basis="fs", k=5)],
        selection_penalty=0,
        direct_solve="structured",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit_reml(frame, y.astype(float), sample_weight=weight)
    profile = model._reml_profile
    assert profile["direct_backend"] == "structured"
    assert "xt" in " ".join(profile["reml_laplace_excluded_labels"])
    assert profile["reml_laplace_exclusion_unsupported"] == 0
    assert any(issubclass(item.category, WeakIdentificationWarning) for item in caught)


@pytest.mark.parametrize("discrete", [False, True])
def test_the_leaf_and_factor_memos_reuse_only_bitwise_equal_inputs(discrete) -> None:
    """The one-slot memos (perf F1) key on the exact rows and the exact penalty parts.

    The leaf memo is one slot of the lineage cache (perf F15): a lambda rebuild's
    layout on the same term and border objects shares it, one whose border
    matrix was re-created misses and replaces it.  Mutations: the leaf memo
    keyed on ``W`` alone (a new right-hand side would reuse a stale triangle),
    the factor memo on the border penalty alone, the slot matched without its
    sources (a re-created border would reuse a stale triangle), or a slot per
    source set (it grows with every rebuild).
    """
    from superglm.group_matrix import DenseGroupMatrix
    from superglm.solvers._structured.assembly import build_augmented_block_factor
    from superglm.solvers._structured.block_leaves import build_factor_smooth_leaf_system
    from superglm.solvers._structured.layout import (
        build_factor_smooth_leaf_layout,
        get_structured_layout,
    )
    from tests.test_factor_smooth_structured_system import _design

    rng, dm, groups, index = _design(discrete=discrete)
    layout = get_structured_layout(dm, groups, dominant_group_index=index)
    W = rng.uniform(0.3, 1.6, size=dm.n)
    Wz = rng.normal(size=dm.n)
    first = build_factor_smooth_leaf_system(layout, W, Wz)
    assert build_factor_smooth_leaf_system(layout, W.copy(), Wz.copy()) is first
    moved = Wz.copy()
    moved[0] += 1.0
    second = build_factor_smooth_leaf_system(layout, W, moved)
    assert second is not first
    assert not np.array_equal(second.leaf.top, first.leaf.top)
    assert build_factor_smooth_leaf_system(layout, W, moved, signed=True) is not second

    K, k = second.operator.n_levels, second.operator.block_size
    q = len(second.operator.small_indices)
    local = np.broadcast_to(np.eye(k), (K, k, k)).copy()
    penalized = FactorSmoothPenalizedOperator.with_penalties(second.operator, np.eye(q), local)
    factor, _ = build_augmented_block_factor(second, penalized)
    same = FactorSmoothPenalizedOperator.with_penalties(second.operator, np.eye(q), local.copy())
    assert build_augmented_block_factor(second, same)[0] is factor
    scaled = FactorSmoothPenalizedOperator.with_penalties(second.operator, np.eye(q), 2.0 * local)
    other, _ = build_augmented_block_factor(second, scaled)
    assert other is not factor
    assert other.logdet() != factor.logdet()

    shared: dict = {}
    matrices = list(dm.group_matrices)

    def rebuilt(group_matrices):
        return build_factor_smooth_leaf_layout(
            group_matrices, groups, dominant_group_index=index, nesting_cache=shared
        )

    one, two = rebuilt(matrices), rebuilt(list(matrices))
    assert two is not one
    held = build_factor_smooth_leaf_system(one, W, Wz)
    assert build_factor_smooth_leaf_system(two, W, Wz) is held
    fresh = list(matrices)
    fresh[0] = DenseGroupMatrix(2.0 * np.asarray(matrices[0].M))
    moved_border = build_factor_smooth_leaf_system(rebuilt(fresh), W, Wz)
    assert moved_border is not held
    assert not np.array_equal(moved_border.leaf.top, held.leaf.top)
    assert len(shared[("fs_slot", "leaf_memo")]) == 1
    assert len(shared[("fs_slot", "prior")]) <= 2


@pytest.mark.parametrize("discrete", [False, True])
def test_the_weight_derivative_moments_scale_exactly_and_stay_symmetric(discrete) -> None:
    """The multi-weight moment pass (perf F5/F16) is homogeneous in the weights.

    A power-of-two weight scale multiplies every product and sum exactly, so
    the operators at ``2^33 a`` are exactly ``2^33`` times those at ``a``, and
    every local block is exactly symmetric.  Mutation: the exact route's blocks
    taken from the natural map's matmul as they come, whose two triangles
    round apart by ``u`` of the entries: at ``2^33`` that exceeds the
    operator's symmetry check and the REML weight derivative is refused.
    """
    from superglm.solvers._structured.block_leaves import factor_smooth_moment_operators
    from superglm.solvers._structured.layout import get_structured_layout
    from tests.test_factor_smooth_structured_system import _design

    rng, dm, groups, index = _design(discrete=discrete)
    layout = get_structured_layout(dm, groups, dominant_group_index=index)
    a = rng.uniform(-1.0, 1.0, size=dm.n)
    scale = 2.0**33
    (unit, unit_cross, unit_total), (big, big_cross, big_total) = factor_smooth_moment_operators(
        layout, [a, scale * a]
    )
    for block in (unit.D, big.D):
        np.testing.assert_array_equal(block, block.transpose(0, 2, 1))
    np.testing.assert_array_equal(big.D, scale * unit.D)
    np.testing.assert_array_equal(big.C, scale * unit.C)
    np.testing.assert_array_equal(big.A, scale * unit.A)
    np.testing.assert_array_equal(big_cross, scale * unit_cross)
    assert big_total == scale * unit_total


def test_the_border_bound_keeps_a_level_constant_column_at_tiny_lambda() -> None:
    """The residual-relative bound (block_leaves module docstring) is informative where it must be.

    A border column constant within each level lies in every level's basis
    span (the basis is a partition of unity), so after the levels are
    eliminated only lambda pins it: its Schur pivot is lambda-sized.  A
    Gram-level majorant ``eps a a'`` charges it its whole mass (``u_s`` up to
    1.6e5 at the stage-2 entry gate) and the certificate would truncate a
    direction the exact matrix keeps; the residual-relative bound keeps it.
    Mutation: the bound replaced by the Gram-level majorant ``g v~^2``.
    """
    rng = np.random.default_rng(4300)
    K, k, per_level = 6, 6, 40
    knots = np.concatenate(([0.0] * 4, np.linspace(0.0, 1.0, k - 2)[1:-1], [1.0] * 4))
    levels = np.repeat(np.arange(K), per_level)
    x = rng.uniform(size=len(levels))
    Z = scipy.interpolate.BSpline.design_matrix(x, knots, 3).toarray()
    attribute = rng.normal(size=K)[levels]
    X = np.column_stack([rng.normal(size=len(levels)), attribute])
    w = 1e4 * rng.uniform(0.5, 1.5, size=len(levels))
    system = leaf_system_from_rows(
        Z,
        X,
        levels,
        w,
        np.zeros(len(levels)),
        n_levels=K,
        small_indices=np.arange(2),
        structured_indices=np.arange(2, 2 + K * k).reshape(K, k),
    )
    local = np.broadcast_to(1e-9 * np.eye(k), (K, k, k)).copy()
    penalized = FactorSmoothPenalizedOperator.with_penalties(
        system.operator, np.zeros((2, 2)), local
    )
    factor = FactorSmoothLeafFactor(system, penalized)
    assert factor.border_certificate.rank == 2
    assert not factor.rank_truncated
    assert factor.border_certificate.u_s < factor.scaled_schur_eigenvalues()[0]


def test_a_levenberg_shift_on_signed_fs_rows_adds_exactly_its_diagonal() -> None:
    """``irls_direct``'s Levenberg step on an observed fs iterate (design §3.11): ``H + E``.

    The shift goes into the penalty parts the factor takes as square roots, so
    the shifted factor is the factor of ``H + diag(E)`` on the slopes and the
    levels (the intercept is never shifted).
    """
    from superglm.solvers.irls_direct import _levenberg_shifted_leaf_operator

    rng = np.random.default_rng(744)
    factor, system, dense, _ = _factor(rng, signed=True)
    shifted, diagonal = _levenberg_shifted_leaf_operator(factor.penalized, 1e-3, system)
    shifted_factor = FactorSmoothLeafFactor(system, shifted)
    expected = dense.copy()
    expected[np.arange(1, dense.shape[0]), np.arange(1, dense.shape[0])] += diagonal
    assert np.all(diagonal > 0.0)
    assert shifted_factor.logdet() == pytest.approx(np.linalg.slogdet(expected)[1], abs=1e-11)
    rhs = rng.normal(size=dense.shape[0])
    np.testing.assert_allclose(
        shifted_factor.solve(rhs), np.linalg.solve(expected, rhs), atol=5e-12
    )


# -- the stage-2 verifier's findings (large offsets, the border bound, leverage, memo) ----
def _exact_matrix(values) -> list[list[Fraction]]:
    return [[Fraction(float(v)) for v in row] for row in np.atleast_2d(values)]


def _exact_centred_gram(X, a, weights) -> list[list[Fraction]]:
    """``sum_r a_r (x_r - m)(x_r - m)'`` over exact rationals, ``m`` the ``weights``-weighted mean."""
    rows = _exact_matrix(X)
    fa = [Fraction(float(v)) for v in a]
    fw = [Fraction(float(v)) for v in weights]
    p = len(rows[0])
    total = sum(fw)
    mean = [sum(w * row[j] for w, row in zip(fw, rows, strict=True)) / total for j in range(p)]
    gram = [[Fraction(0)] * p for _ in range(p)]
    for weight, row in zip(fa, rows, strict=True):
        centred = [row[j] - mean[j] for j in range(p)]
        for i in range(p):
            if centred[i]:
                scaled = weight * centred[i]
                for j in range(p):
                    gram[i][j] += scaled * centred[j]
    return gram


def _offset_design(discrete: bool):
    """The structured-system tests' fs term beside a dense border with a column at ``1e8``."""
    from superglm.group_matrix import DenseGroupMatrix, DesignMatrix
    from superglm.types import GroupSlice
    from tests.test_factor_smooth_structured_system import _dominant

    rng = np.random.default_rng(81)
    dominant = _dominant(discrete=discrete)
    n = dominant.shape[0]
    dense = DenseGroupMatrix(np.column_stack([1e8 + rng.normal(size=n), rng.normal(size=n)]))
    groups = [GroupSlice(name="dense", start=0, end=2), GroupSlice(name="fs", start=2, end=26)]
    return rng, DesignMatrix([dense, dominant], n, 26), groups


@pytest.mark.parametrize("discrete", [False, True])
def test_the_centred_operators_are_formed_on_the_c0_shifted_rows(discrete) -> None:
    """The stage-2 verifier's SAFE findings 1-2: a column at ``1e8`` (an epoch time, an ID).

    The centred data operator (edf, estimability, the mode score's scale) and a
    REML weight-derivative operator are centred on the working-weighted mean.
    Centring is shift-invariant, so formed on the rows shifted by ``c0`` they
    round at the columns' spread about ``c0``: each entry is within ``8
    gamma_{4np}`` of the shifted rows' Cauchy-Schwarz scale of the exact
    rational one (Higham 2002, Theorem 19.4 for the leaf QR; every centring
    term is bounded by the same scale).  Mutations: the operator of the raw
    moments centred on ``X'w / sum w`` (the stage-2 route; its ``1e8`` entry
    is off by 100%), or the derivative moments formed on the unshifted rows.
    """
    from superglm.solvers._structured.block_leaves import (
        build_factor_smooth_leaf_system,
        factor_smooth_moment_operators,
    )
    from superglm.solvers.structured import centred_data_operator, get_structured_layout

    rng, dm, groups = _offset_design(discrete)
    n, p = dm.n, dm.p
    layout = get_structured_layout(dm, groups, dominant_group_index=1)
    W = rng.uniform(0.3, 1.6, size=n)
    system = build_factor_smooth_leaf_system(
        layout, W, rng.normal(size=n), prior_weights=np.ones(n)
    )
    X = np.asarray(dm.toarray(), dtype=np.float64)
    shift = np.zeros(p)
    shift[layout.small_indices] = system.leaf.center
    rows = X - shift
    g = 8.0 * _gamma(4.0 * n * p)

    operator = centred_data_operator(system)
    computed = materialize_compact_operator(operator)
    exact = np.array(_exact_centred_gram(X, W, W), dtype=np.float64)
    scale = np.sqrt(np.sum(W[:, None] * rows**2, axis=0))
    np.testing.assert_array_less(np.abs(computed - exact), g * np.outer(scale, scale))

    a = rng.uniform(-1.0, 1.0, size=n)
    raw, cross, total = factor_smooth_moment_operators(layout, [a], center=system.leaf.center)[0]
    derivative = materialize_compact_operator(
        CenteredBlockOperator(raw=raw, cross=cross, total=total, center=operator.center)
    )
    exact_a = np.array(_exact_centred_gram(X, a, W), dtype=np.float64)
    spread = np.sqrt(np.sum(np.abs(a)[:, None] * rows**2, axis=0))
    mass = float(np.sum(np.abs(a)))
    d = np.abs(operator.center)
    envelope = (
        np.outer(spread, spread)
        + mass * np.outer(d, d)
        + math.sqrt(mass) * (np.outer(spread, d) + np.outer(d, spread))
    )
    np.testing.assert_array_less(np.abs(derivative - exact_a), g * envelope)


def _exact_border_logdet(levels, Z, X, w, local, small) -> float:
    """``log det`` of the intercept-profiled border ``Q'' = Q_rest + S`` over exact rationals.

    ``Q_x = [1 X]' W [1 X] - sum_l C_l' D_l^-1 C_l`` with ``D_l = Z_l' W_l Z_l +
    P_l`` and ``C_l = Z_l' W_l [1 X_l]`` (the level blocks eliminated), then the
    super-root: ``Q_rest = Q_x[1:, 1:] - q q' / Q_x[0, 0]``.
    """
    A = _exact_matrix(np.column_stack([np.ones(len(w)), X]))
    Zf, wf = _exact_matrix(Z), [Fraction(float(v)) for v in w]
    m, k = len(A[0]), Z.shape[1]
    Q = [[Fraction(0)] * m for _ in range(m)]
    for weight, a in zip(wf, A, strict=True):
        for i in range(m):
            for j in range(m):
                Q[i][j] += weight * a[i] * a[j]
    for level in range(local.shape[0]):
        rows = np.flatnonzero(levels == level)
        D = _exact_matrix(local[level])
        C = [[Fraction(0)] * m for _ in range(k)]
        for r in rows:
            for i in range(k):
                if Zf[r][i]:
                    for j in range(k):
                        D[i][j] += wf[r] * Zf[r][i] * Zf[r][j]
                    for j in range(m):
                        C[i][j] += wf[r] * Zf[r][i] * A[r][j]
        Y = [row[:] for row in C]  # D^-1 C by elimination
        for c in range(k):
            for r in range(c + 1, k):
                factor = D[r][c] / D[c][c]
                for j in range(c, k):
                    D[r][j] -= factor * D[c][j]
                for j in range(m):
                    Y[r][j] -= factor * Y[c][j]
        for c in range(k - 1, -1, -1):
            for j in range(m):
                Y[c][j] = (Y[c][j] - sum(D[c][t] * Y[t][j] for t in range(c + 1, k))) / D[c][c]
        for i in range(m):
            for j in range(m):
                Q[i][j] -= sum(C[t][i] * Y[t][j] for t in range(k))
    S = _exact_matrix(small)
    rest = [
        [Q[i][j] - Q[i][0] * Q[0][j] / Q[0][0] + S[i - 1][j - 1] for j in range(1, m)]
        for i in range(1, m)
    ]
    total = 0.0
    for c in range(len(rest)):
        pivot = rest[c][c]
        total += math.log(pivot.numerator) - math.log(pivot.denominator)
        for r in range(c + 1, len(rest)):
            factor = rest[r][c] / pivot
            for j in range(c, len(rest)):
                rest[r][j] -= factor * rest[c][j]
    return total


@pytest.mark.parametrize("signed", [False, True])
def test_the_border_bound_covers_the_border_log_determinant_error(signed) -> None:
    """The border certificate's ``logdet_bound`` covers the measured error (stage-2 verifier, V1).

    The level-constant column at lambda 1e-9 beside weights 1e4 (the border
    bound test above) against the exact rational ``log det`` of the profiled
    border: the certified first-order bound ``tau tr(Q_s,ret^-1)`` must cover
    it.  Mutation: the border bound ``U`` set to zero (only the profiling's
    own ``gamma_3`` term is left): ``u_s`` and the bound fall to about 1e-15,
    below the measured error of about 1e-13.
    """
    rng = np.random.default_rng(4300)
    K, k, per_level = 6, 6, 40
    knots = np.concatenate(([0.0] * 4, np.linspace(0.0, 1.0, k - 2)[1:-1], [1.0] * 4))
    levels = np.repeat(np.arange(K), per_level)
    x = rng.uniform(size=len(levels))
    Z = scipy.interpolate.BSpline.design_matrix(x, knots, 3).toarray()
    X = np.column_stack([rng.normal(size=len(levels)), rng.normal(size=K)[levels]])
    w = 1e4 * rng.uniform(0.5, 1.5, size=len(levels))
    if signed:
        w = np.where(rng.uniform(size=len(w)) < 0.1, -0.2 * w, w)
    system = leaf_system_from_rows(
        Z,
        X,
        levels,
        w,
        np.zeros(len(levels)),
        n_levels=K,
        small_indices=np.arange(2),
        structured_indices=np.arange(2, 2 + K * k).reshape(K, k),
        signed=signed,
    )
    local = np.broadcast_to(1e-9 * np.eye(k), (K, k, k)).copy()
    small = np.zeros((2, 2))
    factor = FactorSmoothLeafFactor(
        system, FactorSmoothPenalizedOperator.with_penalties(system.operator, small, local)
    )
    exact = _exact_border_logdet(levels, Z, X, w, local, small)
    certificate = factor.border_certificate
    assert certificate.rank == 2
    assert abs(factor._border.logdet - exact) <= certificate.logdet_bound


@pytest.mark.parametrize("signed", [False, True])
def test_row_quadratic_forms_are_the_augmented_inverse_diagonal(signed) -> None:
    """``a_i' H^+ a_i`` from the level factors and the border inverse (design §3.10, A).

    The factor works on the border rows shifted by ``c0``; the rows it is
    handed are raw ``[1, x_i]``.  Well-conditioned rows (border offset 25)
    against the dense inverse of the raw augmented Hessian, one row per level
    and rows spanning two levels, dense and sparse input; ``w_i`` times them
    sums to ``tr(H^-1 A'WA)``, the augmented edf.  Mutation: the rows'
    border half left unshifted (no ``R'`` map), or no row forms at all.
    """
    import scipy.sparse

    rng = np.random.default_rng(761)
    K, k, q = 5, 3, 4
    levels, Z, X, w, wz = _rows(rng, n_levels=K, block_size=k, border=q, signed=signed)
    X = X + 25.0
    system = leaf_system_from_rows(
        Z,
        X,
        levels,
        w,
        wz,
        n_levels=K,
        small_indices=np.arange(q),
        structured_indices=np.arange(q, q + K * k).reshape(K, k),
        center=X.mean(axis=0),
        signed=signed,
    )
    roots = rng.normal(size=(K, k, k))
    local = np.einsum("kji,kjl->kil", roots, roots) + 1.5 * np.eye(k)
    small = 0.4 * np.eye(q)
    factor = FactorSmoothLeafFactor(
        system, FactorSmoothPenalizedOperator.with_penalties(system.operator, small, local)
    )
    dense = _dense(levels, Z, X, w, local, small)
    n = len(w)
    design = np.zeros((n, dense.shape[0]))
    design[:, 0] = 1.0
    design[:, 1 : 1 + q] = X
    for row, level in enumerate(levels):
        design[row, 1 + q + level * k : 1 + q + (level + 1) * k] = Z[row]
    inverse = np.linalg.inv(dense)
    expected = np.einsum("ij,jk,ik->i", design, inverse, design)
    tolerance = dense.shape[0] * _gamma(4 * dense.shape[0]) * np.linalg.cond(dense)
    np.testing.assert_allclose(factor.row_quadratic_forms(design), expected, atol=tolerance)
    sparse = scipy.sparse.csr_array(design)
    np.testing.assert_allclose(factor.row_quadratic_forms(sparse), expected, atol=tolerance)
    spanning = design[:2].copy()
    spanning[:, 1 + q : 1 + q + 2 * k] = rng.normal(size=(2, 2 * k))
    np.testing.assert_allclose(
        factor.row_quadratic_forms(spanning),
        np.einsum("ij,jk,ik->i", spanning, inverse, spanning),
        atol=tolerance,
    )
    edf = float(np.trace(inverse @ (design.T @ (w[:, None] * design))))
    assert float(np.sum(w * factor.row_quadratic_forms(design))) == pytest.approx(
        edf, abs=n * tolerance
    )


def test_a_system_and_its_factor_are_freed_without_the_cyclic_collector() -> None:
    """The factor memo holds its factor weakly (perf F17; stage-2 verifier, T7).

    The factor holds its system, so a strong memo slot on the system made a
    reference cycle that kept every iterate's system and factor (about 1.4 MB
    each at K500) alive until the cyclic collector ran: 1.31x master's peak
    RSS on Poisson K500.  With the collector off, both must go when their last
    reference is dropped, and the memo must still reuse a live factor.
    Mutation: the memo holding the factor itself.
    """
    import gc
    import weakref

    from superglm.solvers._structured.assembly import build_augmented_block_factor

    rng = np.random.default_rng(907)
    _, system, _, (_, _, _, _, _, local, small) = _factor(rng)
    penalized = FactorSmoothPenalizedOperator.with_penalties(system.operator, small, local)
    factor, _ = build_augmented_block_factor(system, penalized)
    assert build_augmented_block_factor(system, penalized)[0] is factor
    held_system, held_factor = weakref.ref(system), weakref.ref(factor)
    # A kernel's first use in the process (numba compiling or loading it) can
    # leave garbage cycles through frames that hold these objects; collect them
    # before the collector goes off, so that only a cycle of the system and its
    # factor themselves, which is still reachable here, could keep them alive.
    gc.collect()
    enabled = gc.isenabled()
    gc.disable()
    try:
        del factor, system, penalized
        assert held_factor() is None
        assert held_system() is None
    finally:
        if enabled:
            gc.enable()
