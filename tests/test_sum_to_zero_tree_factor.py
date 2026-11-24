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
import scipy.linalg

from superglm import FactorSmooth, LambdaPolicy, Numeric, Spline, SuperGLM
from superglm.solvers import centered_system
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
    centred=True,
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
        center=X.mean(axis=0) if centred else None,
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


def _thin_model(direct_solve: str, *, discrete: bool = False, main: Spline | None = None):
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
        features={
            "x1": Numeric(),
            "x": main or Spline(n_knots=6, lambda_policy=LambdaPolicy.fixed(1.0)),
        },
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


def test_a_stiff_main_effect_leaves_its_thin_aliases_exact_nulls() -> None:
    """A thin level beside a main effect at the top of the REML box fits as gram fits it.

    ``_thin_model`` with the main effect in its raw basis (``penalty="none"``)
    at lambda ``1e10``: each one-row level's alias is linear, in the null
    space of the main effect's penalty, so the tree truncates it as gram
    does.  Converged, with gram's fitted values and EDF within ``n eps`` of
    their scale times the border's conditioning allowance.  Mutation:
    without the veto the alias's representation error is certified as
    curvature and divided by: the fit stops unconverged with an EDF 0.8 off.
    """
    main = Spline("cr", n_knots=6, penalty="none", lambda_policy=LambdaPolicy.fixed(1e10))
    model, frame, _ = _thin_model("structured", main=main)
    gram, _, _ = _thin_model("gram", main=main)
    assert model.result.direct_backend == "structured" and model.result.converged
    fitted, expected = model.predict(frame), gram.predict(frame)
    scale = float(np.max(np.abs(expected)))
    assert np.max(np.abs(fitted - expected)) <= 1e3 * len(frame) * EPS * scale
    edf, edf_gram = np.sum(model.result.effective_df), np.sum(gram.result.effective_df)
    assert abs(edf - edf_gram) <= 1e3 * len(frame) * EPS * edf_gram


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


def _zero_on_level_0(rng, n, levels):
    """Three border columns, the last one zero on level 0's rows."""
    X = rng.normal(size=(n, 3))
    X[levels == 0, -1] = 0.0
    return X


@pytest.mark.parametrize("residue", [2.0**-1030, 2.0**-1074])
def test_a_thin_signed_leaf_s_residue_below_the_underflow_floor_is_reflected(residue) -> None:
    """A signed leaf's exhausted column is reflected whatever its residue, down to the smallest subnormal.

    A one-row level's signed leaf is a rank-deficient stack triangularized
    over every border column: each exhausted column holds the rounding of the
    one before, which decays into the subnormal range under a wide border.
    There ``1 / (alpha - beta)`` overflows, so the reflector is formed on the
    column lifted by a power of two (LAPACK's ``dlarfg``).  Which column the
    rounding leaves a residue in differs by BLAS kernel, so it is planted:
    level 0's single row is zero on the last border column, which stays
    exactly zero in its leaf, and a zero pseudo-row below it gets ``residue``
    there (``residue^2`` is zero: the same ``H``, observed weights all
    positive).  The factor is the dense ``H``'s: rank, ``log|H|`` and fitted
    values.  Mutation: without the lift the leaf's export is not finite and
    the pivot certificate refuses a positive definite ``H``.
    """
    case = _case(
        K=6,
        k=4,
        n=240,
        seed=5,
        signed=True,
        relabel=_one_row(1),
        weights=lambda W, levels: np.abs(W),
        border=_zero_on_level_0,
        centred=False,
    )
    system = case["system"]
    rows = system.leaf.pseudo_rows.copy()
    column = rows.shape[2] - 2
    row = int(np.flatnonzero(~rows[0].any(axis=1))[-1])
    assert row > column and not rows[0, :, column].any()
    rows[0, row, column] = residue
    leaf = dataclasses.replace(system.leaf, pseudo_rows=rows)
    factor = SumToZeroTreeFactor(dataclasses.replace(system, leaf=leaf), case["penalized"])
    _matches_dense(factor, case)


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


def _alias_beside_a_random_effect(
    S_b: np.ndarray,
    *,
    thin_row=(1.0, 1.0),
    border_roots=None,
    split: float = 0.0,
    head=None,
    null: int = 2,
):
    """A one-row level beside a random effect, built by hand: ``(factor, case)``.

    ``K = 8``, ``k = 4``, ``omega = diag(1, 1, 0, 0)``.  Level 0's single row
    has unpenalized coordinates ``thin_row``, so its free direction is the
    one orthogonal to it (``(1, -1)`` by default).  The border holds ``z_2``
    and ``z_3`` (the global Spline's share of the null-space functions, which
    represents the alias) and a complete three-level one-hot block, whose sum
    is the generator.  ``thin_counts`` is set as ``layout.thin_level_counts``
    would set it, and ``border_roots`` as the assembly would, or left out as
    a hand-built operator leaves them.  A nonzero ``split`` holds ``z_3`` as
    two columns ``z_3 / 2 +/- split a`` (``a`` standard normal), whose
    difference the data barely see; ``head(Z)`` gives the border other
    leading columns than ``z_2`` and ``z_3``.  ``null = 3`` leaves ``z_1``
    unpenalized and in the border too, so the level has two free directions.
    """
    K, k, n, lam = 8, 4, 400, 0.3
    rng = np.random.default_rng(7)
    levels = rng.integers(1, K, n)
    levels[0] = 0
    x = rng.uniform(size=n)
    Z = np.column_stack([x**j for j in range(k)]) + 0.1 * rng.normal(size=(n, k))
    Z[0, k - null :] = thin_row
    effect = rng.integers(0, 3, n)
    effect[:3] = np.arange(3)
    X = np.column_stack((Z[:, k - null :] if head is None else head(Z), np.eye(3)[effect]))
    q = X.shape[1]
    W = rng.uniform(0.5, 2.0, n)
    z = rng.normal(size=n)
    if split:
        a = split * rng.normal(size=n)
        X = np.column_stack((Z[:, 2], Z[:, 3] / 2 + a, Z[:, 3] / 2 - a, X[:, 2:]))
        q = X.shape[1]
    system = leaf_system_from_rows(
        Z,
        X,
        levels,
        W,
        W * z,
        n_levels=K,
        small_indices=np.arange(q),
        structured_indices=np.arange(q, q + (K - 1) * k).reshape(K - 1, k),
        generators=BorderGenerators(
            matrix=np.r_[np.zeros(q - 3), np.ones(3)][:, None], references=[q - 3]
        ),
        basis="sz",
    )
    system = dataclasses.replace(system, thin_counts=(np.array([0]), np.array([1])))
    omega = np.diag(np.r_[np.ones(k - null), np.zeros(null)])
    penalized = SumToZeroPenalizedOperator.with_penalties(
        system.operator, S_b, np.broadcast_to(lam * omega, (K, k, k)).copy(), border_roots
    )
    case = dict(K=K, k=k, q=q, levels=levels, Z=Z, X=X, W=W, z=z, omega=omega, lam=lam, S_b=S_b)
    return SumToZeroTreeFactor(system, penalized), case


def _misrepresenting(monkeypatch, rows, delta) -> None:
    """``scipy.linalg.lstsq`` with ``delta`` added to its solution's ``rows``: a forced representation error.

    The thin-alias least squares solves for ``[intercept | border]`` in its
    Jacobi-scaled coordinates, one column per null-space function.
    """
    solve = scipy.linalg.lstsq

    def misrepresented(*args, **kwargs):
        solution, residues, rank, singular = solve(*args, **kwargs)
        solution = solution.copy()
        solution[rows] += delta
        return solution, residues, rank, singular

    monkeypatch.setattr(scipy.linalg, "lstsq", misrepresented)


def _showing(monkeypatch, shown: float) -> None:
    """``np.linalg.eigh`` with the split border's ``[1 | border]`` Gram's eigenvalues over ``shown``.

    The correction a residual shows divides by them, so each one then removes
    ``shown`` of the representation's error.
    """
    eigh = np.linalg.eigh

    def partly(a, *args, **kwargs):
        values, vectors = eigh(a, *args, **kwargs)
        return (values / shown if np.shape(a) == (7, 7) else values), vectors

    monkeypatch.setattr(np.linalg, "eigh", partly)


def _matches_dense(factor, case, null=None) -> None:
    """``factor``'s rank, ``log|H|`` and fitted values against the dense ``H``, ``null`` its exact null.

    ``log H_00 + log pdet(H_c) = log det(H + t t') - log(t't)``, ``t = (0,
    null_1)``; the fitted values are those of ``(H + null null') beta = X'Wz``
    (``X null = 0``), within ``8 p u kappa_s`` (Higham 2002, Theorem 10.6).
    """
    H, X, _ = _dense(case)
    n, p = len(case["W"]), H.shape[0]
    magnitude = _magnitude(case)
    B, G, shift = H, H, 0.0
    if null is not None:
        assert np.all(np.abs(H @ null) <= _gamma(n + p + 2) * (magnitude @ np.abs(null)))
        tail = np.r_[0.0, null[1:]]
        B, G = H + np.outer(tail, tail), H + np.outer(null, null)
        magnitude = magnitude + np.outer(np.abs(tail), np.abs(tail))
        shift = float(np.log(tail @ tail))
    assert factor.rank == p - (null is not None)
    tolerance = _logdet_agreement(B, magnitude, n + p + 4)
    assert abs(factor.logdet() - (_cholesky_logdet(B) - shift)) <= tolerance
    expected = X @ np.linalg.solve(G, X.T @ (case["W"] * case["z"]))
    reach = 8 * p * _U * _kappa(G) * float(np.max(np.abs(expected)))
    assert np.max(np.abs(X @ factor.solve_data() - expected)) <= reach


def _alias_null(case, f, border=None) -> np.ndarray:
    """``K`` times level 0's alias in ``_alias_beside_a_random_effect``, ``f`` its free direction.

    Border ``f`` on ``z_2, z_3`` (or ``border``, its representation on a split
    border), level 0 ``(K - 1) f`` and every other public level ``-f`` on the
    unpenalized coordinates (the last level's ``-f`` is implied): exact
    entries, a null of the data and of the level penalty.
    """
    K, k, q = case["K"], case["k"], case["q"]
    N = np.zeros(1 + q + (K - 1) * k)
    N[1 : 1 + q - 3] = f if border is None else border
    for level in range(K - 1):
        share = K - 1.0 if level == 0 else -1.0
        N[1 + q + level * k + 2 : 1 + q + (level + 1) * k] = share * np.asarray(f)
    return N


_EFFECT = (np.array([2, 3, 4]), None, None, 0.5)  # the random effect's ridge: no raw root


@pytest.mark.parametrize("width", [2, 3], ids=["full_rank", "beside_a_null"])
def test_graded_raw_penalty_keeps_a_thin_alias_in_the_log_determinant(width) -> None:
    """The raw block curves both mapped directions equally, including the thin alias."""
    s, ridge = 2.0**-50, 2.0**-34
    strong = np.array([1.0, -1.0, 1.0])[:width]
    weak = np.array([1.0, 1.0, 0.0])[:width]
    omega = ridge * (np.outer(strong, strong) + s * np.outer(weak, weak))
    C = np.array([[2.0**24, 0.5], [2.0**24, -0.5], [0.0, 0.0]])[:width]
    block = ridge * np.eye(2)
    np.testing.assert_array_equal(C.T @ omega @ C, block)
    factor, case = _alias_beside_a_random_effect(
        scipy.linalg.block_diag(block, 0.5 * np.eye(3)),
        thin_row=(1.0, 1.0),
        border_roots=((np.arange(2), omega, C, 1.0), _EFFECT),
    )
    # Replace the first border coordinate by the exact data alias, whose
    # border entries are (1, -1). This unit-determinant change of basis gives
    # it pivot 2 * ridge and just one coupling, -ridge to the second column.
    H, _, _ = _dense(case)
    keep = np.arange(len(H)) != 1
    rest = H[np.ix_(keep, keep)]
    rest[1, 1] -= ridge / 2.0
    expected = np.log(2.0 * ridge) + _cholesky_logdet(rest)
    tolerance = _logdet_agreement(
        rest, _magnitude(case)[np.ix_(keep, keep)], len(case["W"]) + len(H) + 4
    )
    assert (factor.rank, factor.logdet()) == (len(H), pytest.approx(expected, abs=tolerance))


@pytest.mark.parametrize("with_null", [False, True], ids=["full_rank", "beside_a_null"])
def test_graded_raw_penalty_keeps_unit_curvature_on_a_data_null_column(with_null) -> None:
    """SSP scaling maps the weak and strong directions of the raw penalty to unit curvature."""
    s = 2.0**-50
    strong, weak = np.array([1.0, -1.0]), np.ones(2)
    C = np.array([[2.0**24, 0.5], [2.0**24, -0.5]])
    if with_null:
        strong, weak = np.append(strong, 1.0), np.append(weak, 0.0)
        C = np.vstack((C, [0.0, 0.0]))
    omega = np.outer(strong, strong) + s * np.outer(weak, weak)
    np.testing.assert_array_equal(C.T @ omega @ C, np.eye(2))
    S_b = scipy.linalg.block_diag(np.eye(2), 0.5 * np.eye(3))
    factor, case = _alias_beside_a_random_effect(
        S_b, thin_row=(0.0, 1.0), border_roots=((np.arange(2), omega, C, 1.0), _EFFECT)
    )
    _matches_dense(factor, case)


@pytest.mark.parametrize("shown", [1.0, 0.6])
def test_a_representation_error_the_data_barely_see_does_not_certify_a_null_alias(
    monkeypatch, shown
) -> None:
    """A representation error above the decision's resolution is read from the residual, not as curvature.

    Level 0's single row ``(1, 0)`` puts the alias on ``z_3``, which the
    border holds split in two, ``z_3 / 2 +/- 1e-4 a``, under a difference
    penalty ``[[1, -1], [-1, 1]]``: the alias ``(0, 1, 1)`` is in its null.
    The least squares' solution is given ``+/-1e-4`` on the two halves (its
    scaled coordinates), along the difference the data barely see, so the
    misfit stays inside its ``sqrt(eps)`` acceptance while the border
    part's raw image is about ``1e-5`` of its scale, well above the
    decision's ``sqrt(eps)`` resolution.  The formed rule certifies the
    alias (asserted, without roots).  Each correction the residual shows
    removes ``shown`` of the error left (the Gram's eigenvalues it divides
    by, forced larger): all of it at once, or 0.6 of it, which after the
    five refinement steps leaves an image seven times the resolution that
    the correction still due, twice over, covers.  So with the roots the
    alias is left to the pivoted factorization: rank, pseudo-determinant and
    fitted values are the dense ``H``'s.  Mutations: with the refined slack
    on the unrefined representation, or at ``shown = 0.6`` with the
    correction counted once, the alias is deflated on the forced error.
    """
    omega = np.array([[1.0, -1.0], [-1.0, 1.0]])
    S_b = scipy.linalg.block_diag(0.0, omega, 0.5 * np.eye(3))
    roots = ((np.array([1, 2]), omega, np.eye(2), 1.0), (np.array([3, 4, 5]), None, None, 0.5))
    _showing(monkeypatch, shown)
    _misrepresenting(monkeypatch, [2, 3], np.array([[1e-4], [-1e-4]]))
    formed, _ = _alias_beside_a_random_effect(S_b, thin_row=(1.0, 0.0), split=1e-4)
    factor, case = _alias_beside_a_random_effect(
        S_b, thin_row=(1.0, 0.0), split=1e-4, border_roots=roots
    )
    monkeypatch.undo()
    assert formed._alias_x is not None and formed.border_certificate.deflated == 2
    assert factor._alias_x is None and factor.border_certificate.deflated == 1
    _matches_dense(factor, case, _alias_null(case, (0.0, 1.0), border=(0.0, 1.0, 1.0)))


@pytest.mark.parametrize("shown", [1.0, 0.7])
def test_a_representation_error_the_residual_shows_does_not_veto_a_curved_alias(
    monkeypatch, shown
) -> None:
    """A curved alias is judged on its refined representation, not vetoed on the least squares' error.

    The border of the test above, with the halves under ``[[1 + s, -1 + s],
    [-1 + s, 1 + s]]``, ``s = 1e-7``: the alias ``(0, 1, 1)`` is curved, at
    ``sqrt(s)`` of full curvature.  The least squares' solution is given
    ``+/-8e-3`` on the two halves, an error whose image is 5.8 times the
    alias's own and which the residual shows (the normal equations' error on
    an ill-conditioned border, forced).  Refined on its residual, each step
    removing ``shown`` of what is left, the representation shows the alias's
    own image above the correction still due, so the alias stays certified
    and is deflated: rank, determinant and fitted values are the dense
    ``H``'s.  Mutations: judged unrefined, with the first correction counted
    twice against an image that is mostly that error, the alias is vetoed and
    the pivoted factorization drops a direction ``H`` curves, rank 34 for 35;
    so is it at ``shown = 0.7`` after one refinement step.
    """
    s, lam = 1e-7, 1e4
    omega = np.array([[1.0 + s, -1.0 + s], [-1.0 + s, 1.0 + s]])
    S_b = scipy.linalg.block_diag(0.0, lam * omega, 0.5 * np.eye(3))
    roots = ((np.array([1, 2]), omega, np.eye(2), lam), (np.array([3, 4, 5]), None, None, 0.5))
    _showing(monkeypatch, shown)
    _misrepresenting(monkeypatch, [2, 3], np.array([[8e-3], [-8e-3]]))
    factor, case = _alias_beside_a_random_effect(
        S_b, thin_row=(1.0, 0.0), split=1e-6, border_roots=roots
    )
    monkeypatch.undo()
    assert factor._alias_x is not None and factor.border_certificate.deflated == 2
    _matches_dense(factor, case)


def test_a_representation_error_inside_the_residual_s_rounding_does_not_certify_a_null_alias(
    monkeypatch,
) -> None:
    """An error the residual cannot show is bounded by the residual's own rounding, not read as curvature.

    Every row is a unit row of the level basis, so each level's triangle is
    its rows exactly (every Householder step has ``tau = 0``), and level 0's
    single row ``e_2`` makes the border's ``z_3`` its alias.  Two border
    columns ``w_1, w_2`` agree but on two ``z_3`` rows, where they hold ``1``
    and ``1 + 2**-20`` swapped: their difference, which their raw penalty
    ``[[1, -1], [-1, 1]]`` curves at full strength, is one the data see at
    ``1e-6`` of their norm.  The least squares returns its exact
    representation of ``z_3`` (rounded to it) plus ``+/-2**-56`` on the two
    columns: far below the rows' unit roundoff, so ``R_B U`` rounds to
    ``R_Z`` and the correction the residual shows is exactly zero (asserted).
    The formed rule certifies the alias on that error (asserted, without
    roots); the residual's rounding, lifted through the Gram the least
    squares kept, covers it, so with the roots the alias is left to the
    pivoted factorization: rank, pseudo-determinant, fitted values and EDF
    are the dense ``H``'s.  Mutation: without the residual's rounding the
    alias is deflated on the error's penalty, ``3e-37``, and EDF is 36,876
    for 15.26.
    """
    K, k, n = 5, 4, 17
    levels = np.repeat(np.arange(K), [1] + [k] * (K - 1))
    Z = np.vstack((np.eye(k)[2], np.tile(np.eye(k), (K - 1, 1))))
    rng = np.random.default_rng(3)
    w_1 = 2.0 ** rng.integers(0, 3, n)  # powers of two: the error's products are exact
    rows = np.array([4, 8])  # the z_3 rows of levels 1 and 2
    w_1[rows] = 1.0
    w_2 = w_1.copy()
    w_1[rows[1]] = w_2[rows[0]] = 1.0 + 2.0**-20
    X = np.column_stack((Z[:, 2:], w_1, w_2))
    q, W, z = X.shape[1], np.ones(n), rng.normal(size=n)
    system = leaf_system_from_rows(
        Z,
        X,
        levels,
        W,
        W * z,
        n_levels=K,
        small_indices=np.arange(q),
        structured_indices=np.arange(q, q + (K - 1) * k).reshape(K - 1, k),
        basis="sz",
    )
    system = dataclasses.replace(system, thin_counts=(np.array([0]), np.array([1])))
    omega, lam, pair = np.diag([1.0, 1.0, 0.0, 0.0]), 0.3, np.array([[1.0, -1.0], [-1.0, 1.0]])
    S_b = scipy.linalg.block_diag(np.zeros((2, 2)), pair)
    penalized = SumToZeroPenalizedOperator.with_penalties(
        system.operator,
        S_b,
        np.broadcast_to(lam * omega, (K, k, k)).copy(),
        ((np.array([2, 3]), pair, np.eye(2), 1.0),),
    )
    from superglm.solvers._structured.balance_tree import _root_images

    solve, images, shown = scipy.linalg.lstsq, _root_images, []

    def exact_plus_error(*args, **kwargs):
        solution, *rest = solve(*args, **kwargs)
        solution = solution.copy()  # [1 | border] x (z_2, z_3), Jacobi-scaled
        solution[:, 1] = np.round(solution[:, 1])  # z_3's representation, 2 e_(z_3)
        solution[3:, 1] += 2.0**-56 * np.array([1.0, -1.0])
        return (solution, *rest)

    def spy(border, spread, roots, error=None):
        if error is not None:
            shown.append(error[0])
        return images(border, spread, roots, error)

    monkeypatch.setattr(scipy.linalg, "lstsq", exact_plus_error)
    monkeypatch.setattr("superglm.solvers._structured.balance_tree._root_images", spy)
    formed = SumToZeroTreeFactor(system, dataclasses.replace(penalized, border_roots=None))
    factor = SumToZeroTreeFactor(system, penalized)
    monkeypatch.undo()
    assert formed._alias_x is not None and formed.border_certificate.deflated == 1
    assert shown and not np.any(shown[0])
    assert factor._alias_x is None and factor.border_certificate.deflated == 0
    case = dict(K=K, k=k, q=q, levels=levels, Z=Z, X=X, W=W, z=z, omega=omega, lam=lam, S_b=S_b)
    null = np.zeros(1 + q + (K - 1) * k)
    null[2] = 1.0
    null[1 + q + 3 :: k] = -1.0
    null[1 + q + 3] = K - 1.0
    _matches_dense(factor, case, null)
    H, design, _ = _dense(case)
    values = np.abs(np.linalg.eigvalsh(H))
    tolerance = 4 * H.shape[0] ** 2 * EPS * values[-1] / values[1]
    edf = float(np.trace(_pinv_known_nullity(H, 1) @ (design.T @ design)))
    data = LowRankSymmetricOperator(basis=design.T, core=np.eye(n))  # X'WX, the fit's EDF trace
    assert factor.trace_inverse_operator(data) == pytest.approx(edf, abs=n * tolerance)


@pytest.mark.parametrize("grade", [1.0, 2.0**20, 2.0**-20])
def test_a_penalty_root_shows_a_null_within_its_tilt_and_a_vector_within_its_metric(grade) -> None:
    """``penalty_factor_tilt``'s two bounds on a graded block, against an eigensolve at its backward error.

    ``S = L G'G L``, ``G = [(1, -1, 0), (1, 1, -2)]`` and ``L = diag(1, 1,
    grade)``, has the exact null ``x = L^-1 (1, 1, 1)``.  The root of ``S +
    M F M`` (``M`` the block's metric, ``F`` symmetric, coupling the null to
    the weakest kept pair at ``||F||_2 = 3 eps ||B||_2``, LAPACK's bound)
    shows ``x`` within ``||t * x||`` and above a hundredth of it, in any
    grading, so the bound is the size of the effect and sits in the metric
    the eigensolve ran in.  ``||R v|| = ||s * v||`` on the leading pair.
    """
    L = np.diag([1.0, 1.0, grade])
    G = np.array([[1.0, -1.0, 0.0], [1.0, 1.0, -2.0]]) @ L
    S, x = G.T @ G, np.array([1.0, 1.0, 1.0 / grade])
    assert not np.any(S @ x)
    root, full, tilt = centered_system.penalty_factor_tilt(S)
    M = np.sqrt(np.diag(S))
    values, vectors = np.linalg.eigh(S / np.outer(M, M))
    y, weakest = M * x / np.linalg.norm(M * x), vectors[:, 1]
    F = 3 * EPS * values[-1] * (np.outer(y, weakest) + np.outer(weakest, y))
    shown = np.linalg.norm(centered_system.penalty_factor(S + np.outer(M, M) * F) @ x)
    assert 0.01 * np.linalg.norm(tilt * x) <= shown <= np.linalg.norm(tilt * x)
    leading = vectors[:, -1] / M
    assert np.linalg.norm(root @ leading) == pytest.approx(
        np.linalg.norm(full * leading), rel=1e-12
    )


@pytest.mark.parametrize("formed", [3e-11, 5e-11, 1e-8])
@pytest.mark.parametrize("off", ["the root", "the map"])
def test_a_null_border_column_s_formed_penalty_is_cleared(monkeypatch, off, formed) -> None:
    """A border column that is a data null and, to the decision's resolution, a null of its raw penalty.

    Level 0's single row ``(1, 0)`` makes the border's ``z_3`` column itself
    the alias.  Its raw coefficients ``(1, 1, 1)`` are an exact null of the
    raw penalty ``G'G``, ``G = [(1, -1, 0), (1, 1, -2)]``, and the formed
    block, the Gram of the root's image, still keeps ``formed`` on the
    column, lambda times a small square, in one of two ways.  The map leaves
    the column ``1e-10`` off the null (as a whitened basis does at a large
    lambda): an image 10,000 times the products' rounding and the root's
    tilt, inside the resolution.  Or the root is given a null tilt bound of
    eight times the resolution (a kept eigenvalue at the eigensolver's cut)
    and a tilt of a quarter of it (``penalty_factor_tilt``, forced): twice
    the resolution, inside the tilt.  The veto leaves the alias out and the
    column's formed penalty is cleared, so the factor is that of the block
    without it: rank, pseudo-determinant and fitted values.  Mutations:
    without the clearing, with the resolution left out of it (the map), or
    with the root's tilt left out of what a null shows (the root), the
    column stays live at a diagonal just above its data-side bound, which
    makes the rank cut's tolerance vacuous (3e-11 and 5e-11).
    """
    G = np.array([[1.0, -1.0, 0.0], [1.0, 1.0, -2.0]])
    omega, null = G.T @ G, np.ones(3)
    C = np.column_stack(([1.0, 0.0, 0.0], null))
    assert not np.any(omega @ null)
    root, full, tilt = centered_system.penalty_factor_tilt(omega)
    if off == "the map":
        C[0, 1] += 1e-10
    else:
        tilt = 8.0 * np.sqrt(EPS) * full
        root[0] += 0.25 * np.linalg.norm(tilt * null) * null / 3.0
        monkeypatch.setattr(centered_system, "penalty_factor_tilt", lambda _: (root, full, tilt))
    image = root @ C
    gram = image.T @ image
    lam = formed / float(gram[1, 1])
    S_b = scipy.linalg.block_diag(lam * 0.5 * (gram + gram.T), 0.5 * np.eye(3))
    roots = ((np.array([0, 1]), omega, C, lam), _EFFECT)
    factor, case = _alias_beside_a_random_effect(S_b, thin_row=(1.0, 0.0), border_roots=roots)
    assert factor._alias_x is None and factor.border_certificate.deflated == 1
    S_b[1, :] = S_b[:, 1] = 0.0
    _matches_dense(factor, case, _alias_null(case, (0.0, 1.0)))


@pytest.mark.parametrize("formed", [3e-11, 1e-8])
def test_a_null_column_s_formed_penalty_above_its_image_s_gram_by_rounding_is_cleared(
    formed,
) -> None:
    """The clearing accounts for the products' rounding: a diagonal that far above the images' Gram is cleared.

    The map variant of the test above (``z_3``'s raw coefficients ``1e-10``
    off the exact null of ``G'G``, lambda chosen so the Gram of the root's
    image is ``formed`` on the column), with that one diagonal raised to
    ``lambda (image + rounding)^2``, ``image`` and ``rounding`` the
    component's own (``_root_images``): ``5.6e-5`` of it above the Gram,
    and inside the ``lambda (image + 2 rounding)^2`` the roots account for
    (both asserted), as a formed matrix rounded or cut apart from its root
    can hold.  The veto leaves the alias out and the column's formed penalty
    is cleared, so rank, pseudo-determinant and fitted values are those of
    the block without it.  Mutation: with the rounding left out
    of the accounting the column is kept, at ``3e-11`` the rank is 32 for 33,
    and at ``1e-8`` ``log|H|`` is ``4e-5`` off, beyond its bound.
    """
    G = np.array([[1.0, -1.0, 0.0], [1.0, 1.0, -2.0]])
    omega = G.T @ G
    C = np.column_stack(([1.0, 0.0, 0.0], np.ones(3)))
    C[0, 1] += 1e-10
    image = centered_system.penalty_factor(omega) @ C
    gram = image.T @ image
    lam = formed / float(gram[1, 1])
    from superglm.solvers._structured.balance_tree import _root_images

    root = (np.array([0, 1]), omega, C, lam)
    unit = np.array([[0.0], [1.0]])
    ((_, shown, rounding, _),) = _root_images(unit, unit, (root,))
    S_b = scipy.linalg.block_diag(lam * 0.5 * (gram + gram.T), 0.5 * np.eye(3))
    S_b[1, 1] = lam * float(shown[0] + rounding[0]) ** 2
    assert lam * gram[1, 1] < S_b[1, 1] <= lam * float(shown[0] + 2.0 * rounding[0]) ** 2
    factor, case = _alias_beside_a_random_effect(
        S_b, thin_row=(1.0, 0.0), border_roots=(root, _EFFECT)
    )
    assert factor._alias_x is None and factor.border_certificate.deflated == 1
    S_b[1, :] = S_b[:, 1] = 0.0
    _matches_dense(factor, case, _alias_null(case, (0.0, 1.0)))


def _weakly_curved(name: str):
    """``(S_b, roots)``: the border's ``z_3`` column under a penalty far below its raw penalty's largest scale."""
    position = np.array([0, 1])
    if name == "two components":  # combined diag(2e12, 2e-6): each curves z_3 by 1e-9 of its norm
        lam = 1e12
        omegas = [np.outer((1.0, s * 1e-9), (1.0, s * 1e-9)) for s in (1.0, -1.0)]
        return lam * sum(omegas), tuple((position, omega, np.eye(2), lam) for omega in omegas)
    if name == "graded diagonal":  # formed 1e-4 on z_3, 1e-8 of the root's norm
        lam = 1e12
        omega = np.diag([1.0, 1e-16])
        return lam * omega, ((position, omega, np.eye(2), lam),)
    # a formed penalty its root does not show (a column whose image is rounding)
    return np.diag([1.0, 0.171]), ((position, np.diag([1.0, 0.0]), np.eye(2), 1.0),)


@pytest.mark.parametrize("name", ["two components", "graded diagonal", "unaccounted"])
def test_a_weak_penalty_on_a_data_null_border_column_is_kept(name) -> None:
    """A null column's formed penalty is not cleared for being small beside its raw penalty's norm.

    Level 0's single row ``(1, 0)`` makes the border's ``z_3`` column the
    alias, a data null within its bound, and each case gives it a formed
    penalty that is real: two components that each curve it by ``1e-9`` of
    their norm, a graded diagonal that curves it by ``1e-8`` of it, and a
    diagonal the roots do not account for.  In its penalty block's own
    metric the column is at full curvature in the first two, so the alias is
    deflated on it; in the third the diagonal is above the Gram of the
    images, so it is kept.  ``H`` keeps its full rank: rank, determinant and
    fitted values are the dense ``H``'s.  Mutation: judged against the stacked
    root's norm, with no account of the diagonal, the column is cleared, the
    factor loses a rank and ``log|H|`` moves by 13 to 22.
    """
    block, roots = _weakly_curved(name)
    S_b = scipy.linalg.block_diag(block, 0.5 * np.eye(3))
    factor, case = _alias_beside_a_random_effect(
        S_b, thin_row=(1.0, 0.0), border_roots=(*roots, _EFFECT)
    )
    assert (factor._alias_x is None) == (name == "unaccounted")
    assert factor.border_certificate.exact_null == 0
    _matches_dense(factor, case)


def test_a_thin_alias_is_judged_the_same_in_any_border_column_units() -> None:
    """The alias decision reads no column's units: the same model with two border columns scaled.

    Level 0's alias ``(1, -1)`` on the border's ``z_2 - 1`` and ``z_3``
    (the intercept carries the constant) under the raw penalty ``diag(0.5,
    0.7)`` at lambda ``1e-10``, a curvature the pivoted factorization does
    not resolve, so the alias is identified only by its deflation.  With the
    two columns held as ``M (z_2 - 1)`` and ``M z_3``, ``M = 1e9``, the
    coefficients are ``1 / M`` of the intercept's and the formed penalty
    ``M^2`` larger: the alias is deflated as at ``M = 1``, with the same rank
    and fitted values, and ``log|H|`` moves by ``4 log M``.  Mutation: with
    the border share measured on the coefficients, the alias at ``M = 1e9``
    is left out, a rank is lost and ``log|H|`` is 27.5 off.
    """
    raw, lam = np.diag([0.5, 0.7]), 1e-10
    factors = []
    for M in (1.0, 1e9):
        S_b = scipy.linalg.block_diag(lam * M * M * raw, np.diag([0.3, 0.4, 0.6]))
        roots = ((np.array([0, 1]), raw, M * np.eye(2), lam), _EFFECT)
        head = lambda Z, M=M: M * np.column_stack((Z[:, 2] - 1.0, Z[:, 3]))  # noqa: E731
        factor, case = _alias_beside_a_random_effect(S_b, border_roots=roots, head=head)
        assert factor._alias_x is not None and factor.border_certificate.deflated == 2
        H, X, _ = _dense(case)
        factors.append((factor, X @ factor.solve_data(), H, case))
    (unit, fitted, H, case), (scaled, moved, _, _) = factors
    p = H.shape[0]
    assert unit.rank == scaled.rank == p
    tolerance = p * p * EPS * abs(scaled.logdet())  # p pivots' logarithms, each to p eps
    assert abs(scaled.logdet() - unit.logdet() - 4.0 * np.log(1e9)) <= tolerance
    # the fitted values do not move along the alias, so their conditioning is H's without it
    alias = np.r_[1.0, _alias_null(case, (1.0, -1.0))[1:]]
    reach = 8 * p * _U * _kappa(H + np.outer(alias, alias)) * float(np.max(np.abs(fitted)))
    assert np.max(np.abs(moved - fitted)) <= reach


def test_a_tilted_level_basis_does_not_certify_a_null_alias(monkeypatch) -> None:
    """The resolution: a border part the level basis's own tilt makes is not curvature.

    The border also holds ``z_1``, a function the level penalty curves, and
    its raw penalty, the all-ones block on raw coefficients ``z_1 (1, 0, 0) +
    z_2 (1, -1, 0) + z_3 (0, 1, -1)`` at lambda ``1e10``, curves ``z_1``
    alone; level 0's alias ``(1, -1)`` on ``z_2, z_3`` is in its null.  The
    level penalty's null basis is
    tilted by ``1e-10`` toward ``z_1`` (its eigenvectors, forced), and the
    border reproduces the tilted function exactly, so the residual is silent
    and the alias carries ``1e-10`` on ``z_1``: a raw image 2,000 times its
    representation error and 150 times under ``sqrt(eps)`` of its scale.
    The alias is left to the pivoted factorization, which truncates the
    exact null: rank, pseudo-determinant and fitted values are the dense
    ``H``'s.  Mutation: without the resolution the alias is deflated on
    ``1e10`` times the squared tilt (two deflated, rank ``p``).
    """
    eigh, tilted = np.linalg.eigh, []

    def tilting(a, *args, **kwargs):
        values, vectors = eigh(a, *args, **kwargs)
        if np.shape(a) == (4, 4):  # the balanced level penalty, diag(c, c, 0, 0)
            tilted.append(True)
            vectors = vectors.copy()
            vectors[1, np.argmax(np.abs(vectors[2]))] = 1e-10
        return values, vectors

    monkeypatch.setattr(np.linalg, "eigh", tilting)
    S_b = np.diag([1e10, 0.0, 0.0, 0.5, 0.5, 0.5])
    C = np.array([[1.0, 1.0, 0.0], [0.0, -1.0, 1.0], [0.0, 0.0, -1.0]])
    roots = (
        (np.array([0, 1, 2]), np.ones((3, 3)), C, 1e10),
        (np.array([3, 4, 5]), None, None, 0.5),
    )
    factor, case = _alias_beside_a_random_effect(S_b, border_roots=roots, head=lambda Z: Z[:, 1:])
    monkeypatch.undo()
    assert tilted and factor._alias_x is None and factor.border_certificate.deflated == 1
    _matches_dense(factor, case, _alias_null(case, (1.0, -1.0), border=(0.0, 1.0, -1.0)))


def _two_thin_levels_beside_a_stiff_term(t: float = 0.5, split: float = 0.0):
    """Two one-row levels at ``x = t`` and ``-t`` beside a stiff term: ``(system, penalized, case, null)``.

    Unpenalized coordinates ``(1, x)``; the border holds ``x`` (a unit
    ridge) and a stiff term (an identity at ``1e6`` on two columns).  Each
    level's alias is linear, ``x -/+ t``; their difference is the constant,
    which the intercept carries, and ``null`` is ``K`` times it: an exact
    null of ``H``.  A nonzero ``split`` holds ``x`` as two columns ``x / 2
    +/- split a`` (``a`` standard normal) under ``diag(1, 4)``, whose
    difference the data barely see.
    """
    K, k, n, lam = 8, 4, 400, 0.3
    rng = np.random.default_rng(5)
    levels = rng.integers(2, K, n)
    levels[:2] = (0, 1)
    x = rng.uniform(-1.0, 1.0, n)
    x[:2] = (t, -t)
    Z = np.column_stack([np.cos(3 * x), np.sin(2 * x), np.ones(n), x])
    Z[:, :2] += 0.1 * rng.normal(size=(n, 2))
    v = rng.uniform(-1.0, 1.0, n)
    X = np.column_stack((x, v, v**2 - 1.0 / 3.0))
    ridge = np.ones(1)
    if split:
        a = split * rng.normal(size=n)
        X = np.column_stack((x / 2 + a, x / 2 - a, X[:, 1:]))
        ridge = np.array([1.0, 4.0])
    q, held = X.shape[1], len(ridge)
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
        basis="sz",
    )
    system = dataclasses.replace(system, thin_counts=(np.array([0, 1]), np.array([1, 1])))
    omega = np.diag([1.0, 1.0, 0.0, 0.0])
    S_b = np.diag(np.r_[ridge, 1e6, 1e6])
    roots = (
        (np.arange(held), np.diag(ridge), np.eye(held), 1.0),
        (np.arange(held, q), None, None, 1e6),
    )
    penalized = SumToZeroPenalizedOperator.with_penalties(
        system.operator, S_b, np.broadcast_to(lam * omega, (K, k, k)).copy(), roots
    )
    case = dict(K=K, k=k, q=q, levels=levels, Z=Z, X=X, W=W, z=z, omega=omega, lam=lam, S_b=S_b)
    null = np.zeros(1 + q + (K - 1) * k)
    for level, f in ((0, (-t, 1.0)), (1, (t, 1.0))):
        sign = 1.0 if level == 0 else -1.0
        null[0] += sign * f[0]  # intercept and x: the main effect's share, K f / K
        null[1 : 1 + held] += sign * f[1]
        for public in range(K - 1):
            share = K - 1.0 if public == level else -1.0
            null[1 + q + public * k + 2 : 1 + q + (public + 1) * k] += sign * share * np.asarray(f)
    return system, penalized, case, null


def test_a_curved_alias_mixed_into_the_intercept_s_combination_is_not_deflated(monkeypatch) -> None:
    """The share test: a combination the intercept holds whole is not certified by a mixed-in curved part.

    ``_two_thin_levels_beside_a_stiff_term`` at ``x = 40`` and ``-40``: the
    difference of the two aliases is the constant, ``80 / K`` on the
    intercept, and the sum is ``2 / K`` on ``x``.  The eigenvectors of the
    aliases' formed curvature are turned by ``1.5e-7`` (forced), so the
    intercept's combination carries ``1.5e-7`` of the sum: a border part
    that is exact, with no representation error, curved by ``x``'s root and
    certified by the formed bar (about 18 times over: a turn under ``3.5e-8``
    is not), and ``7.9e-9`` of the combination's representation in the least
    squares' Jacobi coordinates, half the form's ``sqrt(eps)`` resolution
    (``3.8e-9`` of its coefficients).  The sum alone is deflated and the pivoted
    factorization truncates the constant's exact null.  Mutation: without
    the share test the mixed combination is certified beside the sum (two
    columns), and their penalty Gram is singular, so the deflation declines
    and the sum loses its structural deflation (none deflated).
    """
    system, penalized, case, null = _two_thin_levels_beside_a_stiff_term(40.0)
    eigh, turned = np.linalg.eigh, []
    c, s = np.cos(1.5e-7), np.sin(1.5e-7)

    def mixing(a, *args, **kwargs):
        values, vectors = eigh(a, *args, **kwargs)
        if np.shape(a) == (2, 2):  # the two aliases' formed curvature, the only 2 x 2 here
            turned.append(True)
            vectors = vectors @ np.array([[c, -s], [s, c]])
            values = np.einsum("ij,ik,kj->j", vectors, np.asarray(a), vectors)
        return values, vectors

    monkeypatch.setattr(np.linalg, "eigh", mixing)
    formed = SumToZeroTreeFactor(system, dataclasses.replace(penalized, border_roots=None))
    factor = SumToZeroTreeFactor(system, penalized)
    monkeypatch.undo()
    assert turned and formed._alias_x is not None and formed._alias_x.shape[1] == 2
    alias = factor._alias_x
    assert alias is not None and alias.shape[1] == 1
    assert np.linalg.norm(alias[1:]) > 0.5 * np.linalg.norm(alias)  # the sum, on x
    assert factor.border_certificate.deflated == 1
    _matches_dense(factor, case, null)


def test_a_null_combination_a_representation_error_turns_is_judged_re_formed(monkeypatch) -> None:
    """The combinations are re-formed on the refined border, where a null one carries no curved share.

    ``_two_thin_levels_beside_a_stiff_term`` with ``x`` held as two halves:
    the two aliases' sum is curved, on the halves, and their difference is
    the constant, which the intercept carries whole.  The least squares'
    representation of the constant is given ``+/-1e-4`` on the halves (its
    scaled coordinates), along the difference the data barely see: an error
    whose image under ``diag(1, 4)`` is neither along the sum's nor across
    it.  The eigenvectors of the formed curvature therefore turn the
    constant's combination by the share of the sum that cancels the error's
    image along it, and the formed rule certifies both (asserted, without
    roots).  Refined, the turned combination's border part is that share of
    the sum and nothing else: curved at full strength, and far above the
    share test's resolution.  Re-formed on the refined border it is the
    constant again, so the sum alone is deflated and the pivoted
    factorization truncates the constant's exact null: rank,
    pseudo-determinant and fitted values are the dense ``H``'s.  Mutation:
    judged in the unrefined border's eigenvectors, both are certified.
    """
    system, penalized, case, null = _two_thin_levels_beside_a_stiff_term(split=1e-6)
    _misrepresenting(monkeypatch, [1, 2], np.array([[1e-4, 0.0], [-1e-4, 0.0]]))
    formed = SumToZeroTreeFactor(system, dataclasses.replace(penalized, border_roots=None))
    factor = SumToZeroTreeFactor(system, penalized)
    monkeypatch.undo()
    assert formed._alias_x is not None and formed._alias_x.shape[1] == 2
    alias = factor._alias_x
    assert alias is not None and alias.shape[1] == 1
    assert factor.border_certificate.deflated == 1
    _matches_dense(factor, case, null)


def test_a_certified_set_the_veto_keeps_whole_is_the_formed_rule_s_own(monkeypatch) -> None:
    """Where the veto rejects nothing, the aliases are the formed rule's, bit for bit.

    A one-row level with three unpenalized coordinates has two free
    directions, on ``z_2`` and ``z_3``, each curved by its own ridge: the
    formed rule certifies both and the veto keeps both.  The combinations
    the veto judges are re-formed, here turned by 0.3 (forced: on real fits
    by whatever the representation's error mixed), and span the same two
    aliases, so the factor is given the formed rule's own columns and a
    fit the veto does not touch keeps master's numbers.  Mutation: handed
    the re-formed combinations, the aliases differ from the formed rule's.
    """
    ridge = np.array([1.0, 2.0, 3.0])
    S_b = scipy.linalg.block_diag(np.diag(ridge), 0.5 * np.eye(3))
    roots = ((np.arange(3), np.diag(ridge), np.eye(3), 1.0), (np.arange(3, 6), None, None, 0.5))
    formed, _ = _alias_beside_a_random_effect(S_b, thin_row=(1.0, 0.0, 0.0), null=3)
    eigh, seen = np.linalg.eigh, []
    c, s = np.cos(0.3), np.sin(0.3)

    def turning(a, *args, **kwargs):
        values, vectors = eigh(a, *args, **kwargs)
        if np.shape(a) == (2, 2):  # the two aliases' curvature: formed, then re-formed
            seen.append(True)
            if len(seen) == 2:
                vectors = vectors @ np.array([[c, -s], [s, c]])
        return values, vectors

    monkeypatch.setattr(np.linalg, "eigh", turning)
    factor, case = _alias_beside_a_random_effect(
        S_b, thin_row=(1.0, 0.0, 0.0), null=3, border_roots=roots
    )
    monkeypatch.undo()
    assert len(seen) == 2 and formed._alias_x is not None and formed._alias_x.shape[1] == 2
    assert np.array_equal(factor._alias_x, formed._alias_x)
    _matches_dense(factor, case)


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
