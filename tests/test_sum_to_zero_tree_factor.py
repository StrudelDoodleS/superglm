"""The ``sz`` balance tree against dense references (one-engine design §3.5).

Factor-level fixtures are explicit rows (``leaf_system_from_rows``): level
basis rows ``z``, border rows ``x``, prior weights, and the level penalty
``lambda * Omega`` with ``Omega`` zero on the unpenalized polynomial
coordinates, as sz's natural parameterization makes it.  The dense
reference is the public Hessian ``[1 X_pub]' W [1 X_pub] + S`` with ``X_pub``
the ``K - 1`` sum-to-zero contrast columns.  Tolerances are
``c p eps kappa_s(H)`` (Higham 2002, Theorems 10.3-10.7 on the Jacobi-scaled
matrix), exact nulls are checked against ``H n = 0`` and pseudo-determinants
against ``det(H + N N') / det(N'N)``.
"""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pandas as pd
import pytest

from superglm import FactorSmooth, LambdaPolicy, Numeric, Spline, SuperGLM
from superglm.solvers._structured.balance_tree import (
    ProfiledSumToZeroTreeFactor,
    SumToZeroPenalizedOperator,
    SumToZeroTreeFactor,
    balance_tree,
)
from superglm.solvers._structured.border import BorderGenerators
from superglm.solvers._structured.operators import (
    CenteredBlockOperator,
    LowRankSymmetricOperator,
    SumBlockOperator,
    SumToZeroBlockOperator,
)
from superglm.types import PenaltyComponent
from tests._leaf_systems import leaf_system_from_rows

EPS = np.finfo(float).eps
_U = EPS / 2.0


def _case(
    *,
    K=7,
    k=4,
    q=3,
    n=300,
    seed=0,
    lam=0.7,
    signed=False,
    null=2,
    rows=None,
    weights=None,
    border=None,
    relabel=None,
):
    rng = np.random.default_rng(seed)
    levels = rng.integers(0, K, n)
    levels[:K] = np.arange(K)
    if relabel is not None:
        levels = relabel(levels)
    x = rng.uniform(size=n)
    Z = np.column_stack([x**j for j in range(k)]) + 0.1 * rng.normal(size=(n, k))
    if rows is not None:
        Z = rows(Z, levels)
    X = rng.normal(size=(n, q)) if border is None else border(rng, n, levels)
    W = rng.uniform(0.5, 2.0, n)
    if signed:
        W[rng.uniform(size=n) < 0.15] *= -0.3
    if weights is not None:
        W = weights(W, levels)
    z = rng.normal(size=n)
    system = leaf_system_from_rows(
        Z,
        X,
        levels,
        W,
        W * z,
        n_levels=K,
        small_indices=np.arange(q),
        structured_indices=np.arange(q, q + (K - 1) * k).reshape(K - 1, k),
        center=X.mean(axis=0),
        signed=signed,
        basis="sz",
    )
    omega = np.diag(np.r_[np.ones(k - null), np.zeros(null)])
    S_b = np.diag(np.linspace(0.2, 0.9, q))
    penalized = SumToZeroPenalizedOperator.with_penalties(
        system.operator, S_b, np.broadcast_to(lam * omega, (K, k, k)).copy()
    )
    return dict(
        K=K,
        k=k,
        q=q,
        levels=levels,
        Z=Z,
        X=X,
        W=W,
        z=z,
        system=system,
        penalized=penalized,
        omega=omega,
        lam=lam,
        S_b=S_b,
    )


def _dense(case):
    K, k, q = case["K"], case["k"], case["q"]
    levels, Z, W = case["levels"], case["Z"], case["W"]
    n = len(W)
    p = q + (K - 1) * k
    X = np.zeros((n, 1 + p))
    X[:, 0] = 1.0
    X[:, 1 : 1 + q] = case["X"]
    for r in range(n):
        level = levels[r]
        if level < K - 1:
            X[r, 1 + q + level * k : 1 + q + (level + 1) * k] = Z[r]
        else:
            for other in range(K - 1):
                X[r, 1 + q + other * k : 1 + q + (other + 1) * k] = -Z[r]
    C = np.vstack((np.eye(K - 1), -np.ones((1, K - 1))))
    Omega = np.zeros((1 + p, 1 + p))
    Omega[1 + q :, 1 + q :] = np.kron(C.T @ C, case["omega"])
    S = case["lam"] * Omega
    S[1 : 1 + q, 1 : 1 + q] = case["S_b"]
    return X.T @ (W[:, None] * X) + S, X, Omega


def _component(case, *, profiled: bool) -> PenaltyComponent:
    K, k, q = case["K"], case["k"], case["q"]
    start = q + (0 if profiled else 1)
    return PenaltyComponent(
        name="sz",
        group_name="sz",
        group_index=1,
        group_sl=slice(start, start + (K - 1) * k),
        omega_raw=None,
        omega_ssp=case["omega"],
        rank=float(k - 2),
        penalty_kind="sum_to_zero",
        repeat_count=K,
        block_width=k,
    )


def _kappa(H: np.ndarray) -> float:
    scale = 1.0 / np.sqrt(np.abs(np.diag(H)))
    return float(np.linalg.cond(scale[:, None] * H * scale[None, :]))


def _pdet(H: np.ndarray, N: np.ndarray) -> float:
    """``log pdet(H) = log det(H + N N') - log det(N'N)`` for ``N`` spanning the null space."""
    return float(np.linalg.slogdet(H + N @ N.T)[1] - np.linalg.slogdet(N.T @ N)[1])


def _gamma(count: float) -> float:
    return count * _U / (1.0 - count * _U)


def _magnitude(case) -> np.ndarray:
    """``|X|'|W||X| + |S|`` over ``_dense``'s design and penalty: what ``H``'s entries round against."""
    _, X, Omega = _dense(case)
    S = case["lam"] * Omega
    S[1 : 1 + case["q"], 1 : 1 + case["q"]] = case["S_b"]
    return np.abs(X).T @ (np.abs(case["W"])[:, None] * np.abs(X)) + np.abs(S)


def _cholesky_logdet(B: np.ndarray) -> float:
    return 2.0 * float(np.sum(np.log(np.diag(np.linalg.cholesky(B)))))


def _logdet_agreement(B: np.ndarray, magnitude: np.ndarray, count: int) -> float:
    """What two backward-stable factorizations agree on ``log det B`` to, ``B`` positive definite.

    Each factors ``B + dB`` with ``|dB| <= gamma_count magnitude`` entrywise:
    the rows' accumulation (``fl(X'WX + S)``, Higham 2002, section 3.5) and a
    Cholesky-class factor, of the formed matrix (Theorem 10.3; ``|R'||R|`` is
    below ``sqrt(B_ii B_jj) <= sqrt(magnitude_ii magnitude_jj)``) or of the
    rows (Theorem 19.4).  On the Jacobi-scaled ``B``, whose diagonal is one,
    ``||dB||_2 <= eta = p gamma_count max_ij magnitude_ij / sqrt(B_ii B_jj)``
    and ``|log det(I + E)| <= -p log(1 - ||E||_2) <= p kappa eta / (1 - kappa
    eta)`` (``||B_s^-1||_2 <= kappa``); two factorizations by twice that
    (``_agreement`` in ``test_factor_smooth_leaf_factor.py``).
    """
    p = B.shape[0]
    scale = 1.0 / np.sqrt(np.diag(B))
    eta = p * _gamma(count) * float(np.max(scale[:, None] * magnitude * scale[None, :]))
    kappa_eta = float(np.linalg.cond(scale[:, None] * B * scale[None, :])) * eta
    assert kappa_eta < 0.5
    return 2.0 * p * kappa_eta / (1.0 - kappa_eta)


def test_the_balance_basis_is_orthonormal_and_sums_to_zero() -> None:
    """``N = [h_v]`` over the levels: orthonormal columns, each summing to zero (§3.5)."""
    for K in (2, 3, 7, 16, 33):
        tree = balance_tree(K)
        N = np.zeros((K, tree.n_internal))
        for node in range(tree.n_internal):
            for side in range(2):
                child = int(tree.child[node, side])
                lo, hi = (-child - 1, -child) if child < 0 else (tree.lo[child], tree.hi[child])
                N[lo:hi, node] = tree.h[node, side]
        np.testing.assert_allclose(N.T @ N, np.eye(K - 1), atol=4 * K * EPS)
        np.testing.assert_allclose(N.sum(axis=0), 0.0, atol=4 * K * EPS)
        assert tree.depth == int(np.ceil(np.log2(K)))


@pytest.mark.parametrize("signed", [False, True])
def test_the_tree_factor_matches_the_dense_hessian(signed: bool) -> None:
    """Every product the fit reads, against the dense public Hessian.

    Mutation: the public diagonal without the last level's cross block (the
    sum-to-zero constraint's own term) fails the diagonal checks.
    """
    case = _case(signed=signed, seed=11)
    factor = SumToZeroTreeFactor(case["system"], case["penalized"])
    H, X, Omega = _dense(case)
    p = H.shape[0]
    tolerance = 50 * p * EPS * _kappa(H)
    Hinv = np.linalg.inv(H)
    assert factor.logdet() == pytest.approx(np.linalg.slogdet(H)[1], abs=tolerance)
    rhs = np.random.default_rng(1).normal(size=(p, 3))
    np.testing.assert_allclose(
        factor.solve(rhs), Hinv @ rhs, atol=tolerance * np.max(np.abs(Hinv @ rhs))
    )
    data = X.T @ (case["W"] * case["z"])
    np.testing.assert_allclose(
        factor.solve_data(), Hinv @ data, atol=tolerance * np.max(np.abs(Hinv @ data))
    )
    np.testing.assert_allclose(
        factor.selected_inverse_diagonal(np.arange(p)), np.diag(Hinv), rtol=tolerance
    )
    selected = np.array([0, 1, 5, 9, 12, p - 1])
    np.testing.assert_allclose(
        factor.selected_inverse_block(selected),
        Hinv[np.ix_(selected, selected)],
        atol=tolerance * np.max(np.abs(Hinv)),
    )
    # the last level's covariance: minus the sum of the public levels
    K, k, q = case["K"], case["k"], case["q"]
    public = 1 + q + np.arange((K - 1) * k)
    lift = np.kron(-np.ones((1, K - 1)), np.eye(k))
    last = lift @ Hinv[np.ix_(public, public)] @ lift.T
    np.testing.assert_allclose(
        factor.raw_level_inverse_block(K - 1), last, atol=tolerance * np.max(np.abs(last))
    )
    component = _component(case, profiled=False)
    assert factor.trace_inverse_penalty(component) == pytest.approx(
        np.trace(Hinv @ Omega), rel=tolerance
    )
    assert factor.penalty_cross_trace(component, component, 2.0, 3.0) == pytest.approx(
        np.trace(Hinv @ (2 * Omega) @ Hinv @ (3 * Omega)), rel=tolerance
    )
    rng = np.random.default_rng(5)
    A = rng.normal(size=(q + 1, q + 1))
    D = rng.normal(size=(K, k, k))
    operator = SumToZeroBlockOperator(
        A=A + A.T,
        C=rng.normal(size=(K, k, q + 1)),
        D=D + D.transpose(0, 2, 1),
        small_indices=factor.small_indices,
        structured_indices=factor.structured_indices,
    )
    both = SumBlockOperator(
        (
            operator,
            LowRankSymmetricOperator(
                basis=rng.normal(size=(p, 2)), core=np.array([[1.0, 0.3], [0.3, -0.5]])
            ),
        )
    )
    O1 = operator.matvec(np.eye(p))
    O2 = both.matvec(np.eye(p))
    M = Hinv @ O2
    scale = np.max(np.abs(Hinv)) * np.max(np.abs(O2))
    assert factor.trace_inverse_operator(both) == pytest.approx(
        np.trace(M), abs=tolerance * p * scale
    )
    assert factor.operator_cross_trace(operator, both) == pytest.approx(
        np.trace(Hinv @ O1 @ M), abs=tolerance * p * p * scale**2
    )
    np.testing.assert_allclose(
        factor.inverse_operator_diagonal(both), np.diag(M), atol=tolerance * p * scale
    )
    np.testing.assert_allclose(
        factor.inverse_operator_square_diagonal(both),
        np.diag(M @ M),
        atol=tolerance * p**2 * scale**2,
    )


def test_the_profiled_tree_factor_is_the_slope_block_and_the_identity_routes() -> None:
    """``M_ss`` and the own data operator's identity routes against ``H_c = H / H_00``."""
    case = _case(seed=12)
    factor = SumToZeroTreeFactor(case["system"], case["penalized"])
    system = case["system"]
    xtw = np.empty(system.operator.shape[0])
    xtw[system.operator.small_indices] = system.xtw_small
    xtw[system.operator.structured_indices] = system.xtw_structured
    profiled = ProfiledSumToZeroTreeFactor(augmented_factor=factor, sum_w=system.sum_w, xtw=xtw)
    H, _, Omega = _dense(case)
    Hc = H[1:, 1:] - np.outer(H[1:, 0], H[0, 1:]) / H[0, 0]
    Mss = np.linalg.inv(Hc)
    tolerance = 50 * Hc.shape[0] * EPS * _kappa(Hc)
    assert profiled.logdet() == pytest.approx(np.linalg.slogdet(Hc)[1], abs=tolerance)
    own = CenteredBlockOperator(
        raw=system.operator, cross=xtw, total=system.sum_w, center=xtw / system.sum_w
    )
    own_dense = own.matvec(np.eye(Hc.shape[0]))
    MO = Mss @ own_dense
    np.testing.assert_allclose(profiled.inverse_operator_diagonal(own), np.diag(MO), atol=tolerance)
    np.testing.assert_allclose(
        profiled.inverse_operator_square_diagonal(own), np.diag(MO @ MO), atol=tolerance
    )
    component = _component(case, profiled=True)
    assert profiled.trace_inverse_penalty(component) == pytest.approx(
        np.trace(Mss @ Omega[1:, 1:]), rel=tolerance
    )


def _two_thin_siblings(Z, levels):
    """Levels 0 and 1 (siblings in the tree) with rows zero on the unpenalized coordinates."""
    Z = Z.copy()
    Z[np.isin(levels, (0, 1)), 2:] = 0.0
    return Z


# 3.7 is not a power of two, so fl(3.7 z) leaves a rounding-level remainder
# after the Householder step on z: the pivot is tiny but not an exact zero.
_COLLINEAR = 3.7


def _collinear_siblings(Z, levels):
    """Levels 0 and 1 with their last unpenalized column ``fl(3.7 x)`` its neighbour's."""
    Z = Z.copy()
    rows = np.isin(levels, (0, 1))
    Z[rows, 3] = _COLLINEAR * Z[rows, 2]
    return Z


@pytest.mark.parametrize(
    ("rows", "directions"),
    [
        (_two_thin_siblings, ((0, 0, 1, 0), (0, 0, 0, 1))),
        (_collinear_siblings, ((0, 0, _COLLINEAR, -1),)),
    ],
    ids=["zero_rows", "collinear_rows"],
)
def test_a_subtree_null_is_deferred_and_its_pseudo_determinant_is_exact(rows, directions) -> None:
    """Decision 3: an eps-level tree pivot is carried to the border, not eliminated.

    Levels 0 and 1 are free in the same unpenalized directions (zero rows, or
    collinear rows whose pivot is left at the rounding level), so ``H`` has a
    null supported on their subtree: the balance node that merges them
    defers those coordinates, and the border truncates them.  The published
    ``log|H|`` is gram's ``log sum w + log pdet(H_c)``.  Mutation: no deferral
    (``_DEFERRAL_FACTOR = 0``) eliminates the rounding-level pivot of the
    collinear rows and the determinant is off by tens (an exactly zero pivot
    is deferred either way).
    """
    case = _case(K=8, k=4, q=2, n=400, seed=3, lam=0.3, rows=rows)
    factor = SumToZeroTreeFactor(case["system"], case["penalized"])
    H, X, _ = _dense(case)
    k, q = case["k"], case["q"]
    N = np.zeros((H.shape[0], len(directions)))
    for column, direction in enumerate(directions):
        N[1 + q : 1 + q + k, column] = direction
        N[1 + q + k : 1 + q + 2 * k, column] = -np.asarray(direction, dtype=float)
    # a null of the formed H up to its rounding (Higham 2002, section 3.5)
    assert np.max(np.abs(H @ N)) <= 4 * H.shape[0] * EPS * np.max(np.abs(H) @ np.abs(N))
    Hc = H[1:, 1:] - np.outer(H[1:, 0], H[0, 1:]) / H[0, 0]
    reference = np.log(H[0, 0]) + _pdet(Hc, N[1:])
    assert factor.deferred == len(directions)
    assert factor.rank == H.shape[0] - len(directions)
    tolerance = 50 * H.shape[0] * EPS * _kappa(H + N @ N.T)
    assert factor.logdet() == pytest.approx(reference, abs=tolerance)
    np.testing.assert_allclose(
        X @ factor.solve_data(),
        X @ (np.linalg.pinv(H) @ (X.T @ (case["W"] * case["z"]))),
        atol=tolerance * np.max(np.abs(case["z"])),
    )


def _level_attribute(rng, n, levels):
    attribute = np.array([-3.0, 1.0, 2.0, 0.0, 4.0, -1.0, 2.0, -2.0])
    return np.column_stack((rng.normal(size=n), attribute[levels]))


def _constant_last(Z, levels):
    Z = Z.copy()
    Z[:, -1] = 1.0
    return Z


def test_a_coupled_null_takes_gram_s_pseudo_determinant() -> None:
    """Decision 2: a coupled null (tree part nonzero) keeps ``log sum w + log pdet(H_c)``.

    A border column equal to a level attribute is the intercept plus the
    levels' unpenalized constants: an exact null of ``H`` whose tree part is
    not zero, so ``det(T) pdet(Q)`` misses ``det(n'n)`` of its public null
    vectors.  Mutation: without that term the determinant is 3.8 off.
    """
    case = _case(
        K=8, k=4, q=2, n=400, seed=4, lam=0.3, null=1, rows=_constant_last, border=_level_attribute
    )
    case["S_b"] = np.diag([0.5, 0.0])
    system = case["system"]
    case["penalized"] = SumToZeroPenalizedOperator.with_penalties(
        system.operator, case["S_b"], np.asarray(case["penalized"].penalty_local)
    )
    factor = SumToZeroTreeFactor(system, case["penalized"])
    H, _, _ = _dense(case)
    K, k, q = case["K"], case["k"], case["q"]
    attribute = np.array([-3.0, 1.0, 2.0, 0.0, 4.0, -1.0, 2.0, -2.0])
    N = np.zeros(H.shape[0])
    N[0] = attribute.mean()
    N[2] = -1.0
    for level in range(K - 1):
        N[1 + q + level * k + k - 1] = attribute[level] - attribute.mean()
    assert np.max(np.abs(H @ N)) <= 64 * EPS * np.max(np.abs(H)) * np.max(np.abs(N))
    Hc = H[1:, 1:] - np.outer(H[1:, 0], H[0, 1:]) / H[0, 0]
    reference = np.log(H[0, 0]) + _pdet(Hc, N[1:, None])
    tolerance = 50 * H.shape[0] * EPS * _kappa(H + np.outer(N, N))
    assert factor.rank == H.shape[0] - 1
    assert factor.logdet() == pytest.approx(reference, abs=tolerance)
    assert factor.weakly_identified_coefficients == ()


def _weightless_level(W, levels):
    W = W.copy()
    W[levels == 2] = 0.0
    return W


def test_a_weightless_level_defers_the_super_root_and_is_not_refused() -> None:
    """A level with no weight lets the others' constants reproduce the intercept.

    The super-root pivot is then zero to rounding: it is deferred into the
    border's certified rank decision (decision 3), never refused, and the
    published determinant is still gram's convention.  The profiled edf's
    total is ``rank(H_c) - tr(G S)``.  Mutation: the super-root refusal
    restored raises ``LinAlgError`` here.
    """
    case = _case(
        K=8,
        k=4,
        q=2,
        n=400,
        seed=6,
        lam=0.3,
        null=1,
        rows=_constant_last,
        weights=_weightless_level,
    )
    factor = SumToZeroTreeFactor(case["system"], case["penalized"])
    assert factor._super_deferred
    H, X, Omega = _dense(case)
    K, k, q = case["K"], case["k"], case["q"]
    N = np.zeros(H.shape[0])
    N[0] = 1.0
    for level in range(K - 1):
        N[1 + q + level * k + k - 1] = (K - 1.0) if level == 2 else -1.0
    assert np.max(np.abs(H @ N)) <= 64 * EPS * np.max(np.abs(H)) * K
    Hc = H[1:, 1:] - np.outer(H[1:, 0], H[0, 1:]) / H[0, 0]
    tolerance = 50 * H.shape[0] * EPS * _kappa(H + np.outer(N, N))
    assert factor.logdet() == pytest.approx(np.log(H[0, 0]) + _pdet(Hc, N[1:, None]), abs=tolerance)
    # the fitted rows of positive weight are unique: (H + N N') theta = X'W z
    # solves the normal equations (N'X'W z = 0), whatever the null component
    live = case["W"] > 0.0
    fitted = X @ factor.solve_data()
    reference = X @ np.linalg.solve(H + np.outer(N, N), X.T @ (case["W"] * case["z"]))
    np.testing.assert_allclose(
        fitted[live], reference[live], atol=tolerance * np.max(np.abs(reference))
    )
    system = case["system"]
    xtw = np.empty(system.operator.shape[0])
    xtw[system.operator.small_indices] = system.xtw_small
    xtw[system.operator.structured_indices] = system.xtw_structured
    profiled = ProfiledSumToZeroTreeFactor(augmented_factor=factor, sum_w=system.sum_w, xtw=xtw)
    own = CenteredBlockOperator(
        raw=system.operator, cross=xtw, total=system.sum_w, center=xtw / system.sum_w
    )
    trace_penalty = np.trace(
        np.linalg.pinv(Hc) @ (H[1:, 1:] - (X.T @ (case["W"][:, None] * X))[1:, 1:])
    )
    edf = float(np.sum(profiled.inverse_operator_diagonal(own)))
    assert edf == pytest.approx(
        np.linalg.matrix_rank(Hc) - trace_penalty, abs=tolerance * H.shape[0]
    )


def test_a_signed_level_shift_is_the_level_space_penalty() -> None:
    """The Levenberg shift of an observed sz iterate: ``E`` diagonal in level space (§3.11).

    Its public form ``C' E_lev C`` is not diagonal, so ``irls_direct`` applies
    it to the coefficients instead of scaling them.  Mutation: applying the
    fs diagonal to the public coefficients misses the last level's block.
    """
    from superglm.solvers.irls_direct import _levenberg_shifted_leaf_operator

    case = _case(signed=True, seed=21)
    factor = SumToZeroTreeFactor(case["system"], case["penalized"])
    shifted, apply = _levenberg_shifted_leaf_operator(factor.penalized, 1e-3, case["system"])
    shifted_factor = SumToZeroTreeFactor(case["system"], shifted)
    H, _, _ = _dense(case)
    p = H.shape[0] - 1
    E = np.column_stack([apply(np.eye(p)[:, j]) for j in range(p)])
    expected = H.copy()
    expected[1:, 1:] += E
    tolerance = 50 * H.shape[0] * EPS * _kappa(expected)
    assert np.max(np.abs(E - np.diag(np.diag(E)))) > 0.0
    assert shifted_factor.logdet() == pytest.approx(np.linalg.slogdet(expected)[1], abs=tolerance)


def _thin_model(direct_solve: str, *, discrete: bool = False):
    rng = np.random.default_rng(3)
    n, levels = 3000, 25
    g = rng.integers(0, levels, n)
    for level in (0, 1):
        rows = np.flatnonzero(g == level)
        g[rows[1:]] = levels - 1
    x = rng.uniform(size=n)
    frame = pd.DataFrame({"x": x, "x1": rng.normal(size=n), "g": [f"g{code:02d}" for code in g]})
    y = 0.3 * frame["x1"].to_numpy() + 0.3 * np.sin(2 * np.pi * x) + rng.normal(0, 0.5, n)
    model = SuperGLM(
        family="gaussian",
        features={"x1": Numeric(), "x": Spline(n_knots=6, lambda_policy=LambdaPolicy.fixed(1.0))},
        interactions=[
            FactorSmooth("x", group="g", basis="sz", k=6, lambda_policy=LambdaPolicy.fixed(2.0))
        ],
        selection_penalty=0,
        direct_solve=direct_solve,
        discrete=discrete,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit_reml(frame, y)
    return model, frame, caught


@pytest.mark.parametrize("discrete", [False, True])
def test_thin_levels_are_named_and_kept(discrete: bool) -> None:
    """Decision 4: a level with one row makes an exact alias; flagged and kept, never refused.

    Beside the required global Spline, the main effect's unpenalized linear
    trend and a one-row level's deviation trade freely.  auto fits on the
    balance tree, truncates the alias, names both levels in one warning and
    in the profile, and fits the rows as gram does.
    """
    model, frame, caught = _thin_model("auto", discrete=discrete)
    assert model.result.direct_backend == "structured"
    assert model.result.direct_fallback_reason is None
    assert model._reml_profile["structured_thin_levels"] == ("g00", "g01")
    messages = [
        str(item.message) for item in caught if "fewer distinct x values" in str(item.message)
    ]
    assert len(messages) == 1 and "g00, g01" in messages[0]
    # The alias is exact only in exact arithmetic (a B-spline row's rounding
    # leaves it at the rounding level), so the fitted rows are defined by the
    # penalized objective, which the tree meets at least as well as gram:
    # within the objective's own rounding, n eps |objective| (Higham §3.1).
    gram, _, _ = _thin_model("gram", discrete=discrete)

    def penalized(fitted) -> float:
        from superglm.reml.penalty_algebra import build_penalty_matrix

        dm = fitted._dm
        S = build_penalty_matrix(
            dm.group_matrices, fitted._groups, fitted._reml_lambdas, dm.p, fitted._reml_penalties
        )
        beta = fitted.result.beta
        return float(fitted.result.deviance + beta @ S @ beta)

    tree_objective, gram_objective = penalized(model), penalized(gram)
    assert tree_objective <= gram_objective + len(frame) * EPS * abs(gram_objective)


def _one_row(count):
    """Levels ``0 .. count - 1`` keep one row each; their other rows move to the last level."""

    def relabel(levels):
        levels = levels.copy()
        last = int(levels.max())
        for level in range(count):
            rows = np.flatnonzero(levels == level)
            levels[rows[1:]] = last
        return levels

    return relabel


def _weightless_pair(W, levels):
    """Levels 2 and 5, in different subtrees of the balance tree, carry no weight."""
    W = W.copy()
    W[np.isin(levels, (2, 5))] = 0.0
    return W


@pytest.mark.parametrize(
    ("relabel", "weights", "nullity"),
    [(_one_row(4), None, 2), (_one_row(6), None, 4), (None, _weightless_pair, 2)],
    ids=["four_one_row_levels", "six_one_row_levels", "two_weightless_levels"],
)
def test_an_exhausted_subtree_is_deferred_against_its_unreduced_columns(
    relabel, weights, nullity
) -> None:
    """Decision 3 as the stage-3 verifier's finding 1 needs it: the yardstick is the unreduced column.

    Levels 0-3 with one row each: the balance nodes over (0, 1) and (2, 3)
    use up their two rows on their own pivots, so the unpenalized coordinates
    they export are rounding noise, and so is the parent's stacked column.
    Judged against that noise itself, a remainder of 1e-16 was never small:
    it was divided by, the border bound came out infinite and every border
    column was declared null (rank 28 of 32, ``log|H|`` 174 off).  Judged
    against the subtree's unreduced column (leaf rows and penalty roots, the
    scale of Householder QR's columnwise rounding, Higham 2002 Theorem 19.4),
    the noise is deferred to the border, which truncates exactly the
    nullity; ``log|H|`` is gram's ``log sum w + log pdet(H_c)``.  Two
    weightless levels in different subtrees exhaust theirs the same way.
    Mutation: deferral against the stacked matrix's own column norm.
    """
    case = _case(K=8, k=4, q=3, n=400, seed=3, lam=0.7, relabel=relabel, weights=weights)
    factor = SumToZeroTreeFactor(case["system"], case["penalized"])
    H, X, _ = _dense(case)
    values, vectors = np.linalg.eigh(H)
    # the structural nullity, separated from the rest by many decades
    assert values[nullity - 1] <= 4 * H.shape[0] * EPS * values[-1] < values[nullity] * 1e-6
    N = vectors[:, :nullity]
    Hc = H[1:, 1:] - np.outer(H[1:, 0], H[0, 1:]) / H[0, 0]
    reference = np.log(H[0, 0]) + _pdet(Hc, N[1:])
    tolerance = 50 * H.shape[0] * EPS * _kappa(H + N @ N.T)
    assert factor.rank == H.shape[0] - nullity
    assert factor.border_certificate.rank > 0
    assert factor.logdet() == pytest.approx(reference, abs=tolerance)
    live = case["W"] > 0.0
    fitted = X @ factor.solve_data()
    expected = X @ (np.linalg.pinv(H, rcond=1e-9) @ (X.T @ (case["W"] * case["z"])))
    np.testing.assert_allclose(
        fitted[live], expected[live], atol=tolerance * np.max(np.abs(expected))
    )


def _alias_beside_a_random_effect(S_b: np.ndarray):
    """A one-row level beside a random effect, built by hand: ``(factor, case)``.

    ``K = 8``, ``k = 4``, ``omega = diag(1, 1, 0, 0)``.  Level 0's single row
    has unpenalized coordinates ``(1, 1)``, so its free direction is ``(1,
    -1)``.  The border holds ``z_2`` and ``z_3`` (the global Spline's share
    of the null-space functions, which represents the alias) and a complete
    three-level one-hot block, whose sum is the generator.  ``thin_counts``
    is set as ``layout.thin_level_counts`` would set it.
    """
    K, k, n, lam = 8, 4, 400, 0.3
    rng = np.random.default_rng(7)
    levels = rng.integers(1, K, n)
    levels[0] = 0
    x = rng.uniform(size=n)
    Z = np.column_stack([x**j for j in range(k)]) + 0.1 * rng.normal(size=(n, k))
    Z[0, 2:] = 1.0
    effect = rng.integers(0, 3, n)
    effect[:3] = np.arange(3)
    X = np.column_stack((Z[:, 2:], np.eye(3)[effect]))
    q = X.shape[1]
    W = rng.uniform(0.5, 2.0, n)
    z = rng.normal(size=n)
    system = leaf_system_from_rows(
        Z,
        X,
        levels,
        W,
        W * z,
        n_levels=K,
        small_indices=np.arange(q),
        structured_indices=np.arange(q, q + (K - 1) * k).reshape(K - 1, k),
        generators=BorderGenerators(matrix=np.array([[0.0, 0.0, 1.0, 1.0, 1.0]]).T, references=[2]),
        basis="sz",
    )
    system = dataclasses.replace(system, thin_counts=(np.array([0]), np.array([1])))
    omega = np.diag([1.0, 1.0, 0.0, 0.0])
    penalized = SumToZeroPenalizedOperator.with_penalties(
        system.operator, S_b, np.broadcast_to(lam * omega, (K, k, k)).copy()
    )
    case = dict(K=K, k=k, q=q, levels=levels, Z=Z, X=X, W=W, z=z, omega=omega, lam=lam, S_b=S_b)
    return SumToZeroTreeFactor(system, penalized), case


def test_a_penalty_coupling_a_generator_and_a_thin_alias_keeps_the_alias_out() -> None:
    """The alias deflation's coupled-penalty fallback (``_penalized_aliases``), driven directly.

    A thin level's penalized alias ``A`` is deflated with the border's
    structural generators ``G`` as one block, ``a_NN = [G A]' S [G A]``, and
    certified on ``A'SA`` alone: exact when ``a_NN = diag(G'SG, A'SA)``, as
    ``A`` is zero on ``G``'s columns and a penalty is block diagonal by term.
    A penalty with ``G'SA != 0`` voids that, and the aliases stay with the
    pivoted factorization.  No public model builds such a penalty (a random
    effect's ridge is its own diagonal block), so the leaf is built by hand
    (``_alias_beside_a_random_effect``).  Block diagonal ``S_b``: the alias is
    deflated beside the generator, ``H`` has full rank and ``log|H| = log
    det H``.  ``S_b = 5 I - w w'``, ``w = (1, -1, 1, 1, 1)`` the border part
    of the generator plus eight times the alias (exact entries):
    ``G'SG = A'SA = 6`` but ``a_NN`` is singular, and the generator plus the
    alias is an exact null of ``H``.  The alias stays out, the generator alone
    is deflated, and the factor has ``H``'s rank and pseudo-determinant.
    Mutation: the guard removed: the alias joins the deflation and, its
    ``a_NN`` unresolved, ``_deflate`` declines every generator (none
    deflated); on c9ac978b, before that check, the Cholesky of ``a_NN``
    raised ``LinAlgError`` (Claude review of #425, Nit).
    """
    factor, case = _alias_beside_a_random_effect(np.diag([0.5, 0.7, 0.3, 0.4, 0.6]))
    H, _, _ = _dense(case)
    assert factor._alias_x is not None and factor._alias_x.shape[1] == 1
    assert factor.border_certificate.deflated == 2
    assert factor.rank == H.shape[0]
    # n + 2 roundings forming H, p + 1 in its Cholesky (``_logdet_agreement``)
    n, p = len(case["W"]), H.shape[0]
    tolerance = _logdet_agreement(H, _magnitude(case), n + p + 3)
    assert abs(factor.logdet() - _cholesky_logdet(H)) <= tolerance

    w = np.array([1.0, -1.0, 1.0, 1.0, 1.0])
    factor, case = _alias_beside_a_random_effect(5.0 * np.eye(5) - np.outer(w, w))
    H, _, _ = _dense(case)
    K, k, q = case["K"], case["k"], case["q"]
    N = np.zeros(H.shape[0])
    N[: 1 + q] = (-1.0, 1.0, -1.0, 1.0, 1.0, 1.0)  # intercept, z_2 and z_3, the effect's levels
    for level in range(K - 1):
        share = K - 1.0 if level == 0 else -1.0
        N[1 + q + level * k + 2 : 1 + q + (level + 1) * k] = (share, -share)
    # H N = 0 exactly; fl(H) is within gamma_{n+2} |X|'|W||X| + |S| (the scaled
    # rows, the sum, the added penalty) and the product adds gamma_p (Higham
    # 2002, section 3.5 and Lemma 3.3)
    magnitude = _magnitude(case)
    assert np.all(np.abs(H @ N) <= _gamma(n + p + 2) * (magnitude @ np.abs(N)))
    # log H_00 + log pdet(H_c) = log det(H + t t') - log(t't), t = (0, N_1):
    # the Schur complement on the intercept, N_1 spanning H_c's null space;
    # one more rounding where t t' (exact) is added
    tail = np.r_[0.0, N[1:]]
    B = H + np.outer(tail, tail)
    tolerance = _logdet_agreement(B, magnitude + np.outer(np.abs(tail), np.abs(tail)), n + p + 4)
    assert factor._alias_x is None
    assert factor.border_certificate.deflated == 1
    assert factor.rank == H.shape[0] - 1
    assert abs(factor.logdet() - (_cholesky_logdet(B) - np.log(tail @ tail))) <= tolerance


def _pinv_known_nullity(H: np.ndarray, nullity: int) -> np.ndarray:
    values, vectors = np.linalg.eigh(H)
    kept = vectors[:, nullity:]
    return (kept / values[nullity:]) @ kept.T


@pytest.mark.parametrize(
    ("kwargs", "nullity"),
    [
        (dict(seed=0), 0),
        (dict(K=2, n=120, seed=1), 0),
        (dict(K=9, n=500, seed=2, lam=1e-7), 0),
        (dict(signed=True, seed=21), 0),
        (dict(K=8, q=3, n=400, seed=3, relabel=_one_row(4)), 2),
        (
            dict(
                K=8,
                q=2,
                n=400,
                seed=6,
                lam=0.3,
                null=1,
                rows=_constant_last,
                weights=_weightless_level,
            ),
            1,
        ),
    ],
    ids=["plain", "two_levels", "tiny_lambda", "signed", "one_row_levels", "weightless"],
)
def test_row_quadratic_forms_are_the_augmented_inverse_diagonal(kwargs, nullity) -> None:
    """``a_i' H^+ a_i`` up each row's path and through the border (design §3.10, definition A).

    The stage-3 verifier's finding 3: sz leverage materialized the slope
    inverse (about 1e-8 accurate at tiny lambda) and refused any term wider
    than 256 coefficients.  Every public row: a level's rows hold ``z`` in
    its block, the last level's ``-z`` in every block; combinations of rows
    take the whole-tree pass.  Against the dense pseudo-inverse on rows of
    positive weight, where ``a' H^+ a`` is the same for every generalized
    inverse (``a`` is in the range of ``H``).  Tolerance ``p gamma_{4p}
    kappa`` (Higham 2002, Theorems 10.3 and 19.4).  ``w_i`` times the forms
    sums to the augmented edf.  Mutation: the path's transfer ``u <- A' u``
    left out, or the last level's rows read with the wrong sign.
    """
    case = _case(**kwargs)
    factor = SumToZeroTreeFactor(case["system"], case["penalized"])
    H, X, _ = _dense(case)
    p = H.shape[0]
    inverse = _pinv_known_nullity(H, nullity)
    values = np.abs(np.linalg.eigvalsh(H))
    kappa = values[-1] / values[nullity]
    tolerance = p * 4 * p * EPS * kappa
    live = case["W"] != 0.0
    expected = np.einsum("ij,jk,ik->i", X, inverse, X)
    forms = factor.row_quadratic_forms(X)
    scale = np.max(np.abs(expected[live]))
    np.testing.assert_allclose(forms[live], expected[live], atol=tolerance * scale)
    last = case["levels"] == case["K"] - 1
    assert np.any(last & live)
    combined = np.random.default_rng(3).normal(size=(4, int(np.sum(live)))) @ X[live]
    np.testing.assert_allclose(
        factor.row_quadratic_forms(combined),
        np.einsum("ij,jk,ik->i", combined, inverse, combined),
        atol=tolerance * float(np.max(np.abs(combined))) ** 2 * float(np.max(np.abs(inverse))),
    )
    W = case["W"]
    edf = float(np.trace(inverse @ (X.T @ (W[:, None] * X))))
    assert float(np.sum(W * forms)) == pytest.approx(edf, abs=len(W) * tolerance * scale)


def _pairs(levels, level=3):
    rows = np.flatnonzero(levels == level)
    return rows[0 : len(rows) - 1 : 2], rows[1 : len(rows) : 2]


def _paired_rows(Z, levels):
    """Level 3's rows in identical pairs (with ``_paired_border``, ``_opposite_pairs``)."""
    Z = Z.copy()
    first, second = _pairs(levels)
    Z[second] = Z[first]
    return Z


def _paired_border(rng, n, levels):
    X = rng.normal(size=(n, 3))
    first, second = _pairs(levels)
    X[second] = X[first]
    return X


def _opposite_pairs(W, levels):
    """Each identical pair weighted ``w`` and ``-w``: the signed Gram of level 3 cancels."""
    W = W.copy()
    first, second = _pairs(levels)
    W[first] = np.abs(W[first])
    W[second] = -W[first]
    return W


def _negative_level(W, levels):
    """Level 3 on negative curvature only: its signed Gram is negative definite."""
    W = W.copy()
    W[levels == 3] = -np.abs(W[levels == 3])
    return W


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(signed=True, seed=21),
        dict(signed=True, seed=21, rows=_two_thin_siblings),
        dict(signed=True, seed=21, null=0, lam=1000.0, weights=_negative_level),
        dict(
            signed=True,
            seed=21,
            lam=3.0,
            rows=_paired_rows,
            border=_paired_border,
            weights=_opposite_pairs,
        ),
        dict(K=8, q=3, n=400, seed=3, relabel=_one_row(4)),
        dict(
            K=8,
            q=2,
            n=400,
            seed=6,
            lam=0.3,
            null=1,
            rows=_constant_last,
            weights=_weightless_level,
        ),
    ],
    ids=[
        "signed",
        "signed_thin_siblings",
        "signed_negative_level",
        "signed_cancelling",
        "one_row_levels",
        "weightless",
    ],
)
def test_coefficient_estimability_is_the_data_s(kwargs) -> None:
    """Estimable when no null vector of ``[1, X]`` over the rows with ``w != 0`` moves it.

    The package's rule (the dense solver and the exact reference read the
    data alone), on the balance tree of the data: the same leaf triangles
    (``sqrt|w|`` rows, never the signed middle) without the penalty, its
    certified nulls confirmed on the rows.  Reference: the
    column-equilibrated SVD of ``sqrt|w| [1, X]``, null below ``max(n, p)
    eps`` of the largest singular value (``dgesdd``'s backward error),
    touching a coordinate above ``sqrt(eps)``.  Mutations: the fit's own
    (penalized) aliases deciding; the signed rows' system taken as the data
    tree's (a level of negative curvature has no unpenalized signed factor).
    """
    case = _case(**kwargs)
    factor = SumToZeroTreeFactor(case["system"], case["penalized"])
    _, X, _ = _dense(case)
    A = np.sqrt(np.abs(case["W"]))[:, None] * X
    A = A / np.maximum(np.linalg.norm(A, axis=0), np.finfo(np.float64).tiny)
    _, singular, rows = np.linalg.svd(A, full_matrices=False)
    null = rows[singular <= max(A.shape) * EPS * singular[0]]
    expected = ~np.any(np.abs(null) > np.sqrt(EPS), axis=0)
    np.testing.assert_array_equal(factor.coefficient_estimable(), expected)
