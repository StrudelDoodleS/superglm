"""Stage 0 of the one-engine design: the root causes the speed verifier found, fixed by type.

Each test pins one piece and fails under the mutation named in its docstring
(design section 14, T3), which the change record demonstrates by running it:

- the border centre is the shifted prior-weighted mean of every column that is
  not one-hot, chosen by type (section 3.2);
- a complete one-hot border block is deflated structurally, the retained
  block is Rump-verified, and the truncated subspace is disclosed (3.6);
- an alias that touches the intercept is published in the centred convention,
  ``log sum_w + log pdet(H_c)``, with no refusal and no coupling term (3.6);
- leverage is the influence diagonal with the intercept (3.10);
- PIRLS stops on the certificate's own score in centred coordinates (3.8);
- a weakly identified coefficient is flagged and kept, not refused (3.9).

Bounds derive from dimensions, eps and the certificate's own quantities; the
exact references are rational (``fractions``) from the float64 rows.
"""

from __future__ import annotations

import math
import warnings
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest
import scipy.linalg

from superglm import (
    Categorical,
    FactorSmooth,
    LambdaPolicy,
    Numeric,
    RandomEffect,
    Spline,
    SuperGLM,
    Tweedie,
)
from superglm.group_matrix import (
    DenseGroupMatrix,
    DesignMatrix,
    RandomEffectGroupMatrix,
)
from superglm.inference.covariance import _active_penalty_matrix
from superglm.model.fit_state import fitted_lambda2
from superglm.solvers._structured.border import factor_border
from superglm.solvers.mode_score import mode_certification_bar
from superglm.solvers.structured import (
    ProfiledNestedSchurFactor,
    build_augmented_nested_factor,
    build_nested_structured_system,
    build_penalized_nested_operator,
    get_structured_layout,
    nested_prior_statistics,
)
from superglm.types import GroupSlice, PenaltyComponent

EPS = float(np.finfo(np.float64).eps)
# the certificate's bar at the tolerance these fits use (reml_tol 1e-9)
BAR = mode_certification_bar(1e-9)


# ------------------------------------------------------------------ fixtures
def _adversarial_base(seed: int = 5, n: int = 1500, K: int = 40):
    """The speed verifier's adversarial base: a 40-level RE with one-row and zero-weight levels."""
    rng = np.random.default_rng(seed)
    big = K - 5
    popularity = rng.gamma(1.5, size=big)
    g = rng.choice(big, size=n - 5, p=popularity / popularity.sum())
    g = np.concatenate([g, np.arange(big, K)])
    g = g[rng.permutation(n)]
    frame = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "u": rng.uniform(size=n),
            "cat": np.array([f"c{c}" for c in rng.integers(0, 8, n)], dtype=object),
            "g": np.array([f"g{c:02d}" for c in g], dtype=object),
        }
    )
    eta = 0.1 + 0.3 * frame["x1"].to_numpy() + 0.4 * np.sin(4 * frame["u"].to_numpy())
    eta += rng.normal(0, 0.3, K)[g]
    weight = np.ones(n)
    present = np.unique(g[g < big])
    weight[np.isin(g, present[[2, 9, 17]])] = 0.0
    sole = np.flatnonzero(g == present[5])
    weight[sole[1:]] = 0.0
    return frame, eta, weight, rng, [f"g{c:02d}" for c in range(K)]


def _response(family: str, eta: np.ndarray, rng) -> np.ndarray:
    mu = np.exp(np.clip(eta, -5, 5))
    if family == "poisson":
        return rng.poisson(mu).astype(float)
    if family == "gamma":
        return rng.gamma(2.0, mu / 2.0)
    counts = rng.poisson(mu / 0.5)
    return np.array([rng.gamma(2.0, 0.25, c).sum() for c in counts])


def _fit(
    frame, y, weight, family, numerics, levels, *, lam_g=None, lam_u=None, solve="auto", extra=()
):
    spec = {name: Numeric() for name in numerics}
    spec["u"] = Spline(
        kind="ps", k=8, lambda_policy=None if lam_u is None else LambdaPolicy.fixed(lam_u)
    )
    spec["cat"] = Categorical()
    spec["g"] = RandomEffect(
        levels=levels, lambda_policy=None if lam_g is None else LambdaPolicy.fixed(lam_g)
    )
    for name in extra:
        spec[name] = RandomEffect()
    model = SuperGLM(
        family=Tweedie(p=1.5) if family == "tweedie" else family,
        features=spec,
        selection_penalty=0,
        direct_solve=solve,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_reml(frame, y, sample_weight=weight, pirls_tol=1e-10, reml_tol=1e-9)
    return model


def _nan_se(model, frame, y, weight) -> int:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        se = model.metrics(frame, y, sample_weight=weight).coefficient_se
    return int(sum(np.count_nonzero(np.isnan(values)) for values in se.values()))


def _border_layout(columns, codes, weights, re_levels: int, tree_levels: int, tree_codes):
    """A one-level chain (``tree_codes``) beside dense ``columns`` and one complete RE block."""
    n = len(weights)
    matrices = [
        DenseGroupMatrix(np.column_stack(columns)),
        RandomEffectGroupMatrix(codes, re_levels),
        RandomEffectGroupMatrix(tree_codes, tree_levels),
    ]
    groups, start = [], 0
    for name, matrix in zip(("dense", "crossed", "tree"), matrices, strict=True):
        groups.append(
            GroupSlice(
                name=name, start=start, end=start + matrix.shape[1], penalized=name != "dense"
            )
        )
        start += matrix.shape[1]
    dm = DesignMatrix(matrices, n=n, p=start)
    layout = get_structured_layout(dm, groups, dominant_group_index=2, chain_group_indices=(2,))
    return dm, groups, layout


def _identity(groups, index) -> PenaltyComponent:
    group = groups[index]
    return PenaltyComponent(
        name=group.name,
        group_name=group.name,
        group_index=index,
        group_sl=group.sl,
        omega_raw=None,
        penalty_kind="identity",
    )


def _exact_centred_hessian(X: np.ndarray, w: np.ndarray, S: np.ndarray):
    """``sum_w`` and ``H_c = X'WX - (X'w)(X'w)'/sum_w + S`` in exact rationals."""
    rows = [[Fraction(float(v)) for v in row] for row in X]
    weights = [Fraction(float(v)) for v in w]
    p = X.shape[1]
    sum_w = sum(weights)
    cross = [sum(weights[r] * rows[r][j] for r in range(len(rows))) for j in range(p)]
    H = [
        [
            sum(weights[r] * rows[r][i] * rows[r][j] for r in range(len(rows)))
            - cross[i] * cross[j] / sum_w
            + Fraction(float(S[i, j]))
            for j in range(p)
        ]
        for i in range(p)
    ]
    return sum_w, H


def _exact_inverse_diagonal(H, columns) -> list[Fraction]:
    """``(H^-1)_jj`` for ``j`` in ``columns``, by Gauss-Jordan elimination in rationals."""
    p = len(H)
    A = [list(row) + [Fraction(int(i == j)) for j in range(p)] for i, row in enumerate(H)]
    for c in range(p):
        pivot = next(r for r in range(c, p) if A[r][c] != 0)
        A[c], A[pivot] = A[pivot], A[c]
        A[c] = [value / A[c][c] for value in A[c]]
        for r in range(p):
            if r != c and A[r][c] != 0:
                factor = A[r][c]
                A[r] = [a - factor * b for a, b in zip(A[r], A[c], strict=True)]
    return [A[j][p + j] for j in columns]


def _exact_published_logdet(X: np.ndarray, w: np.ndarray, S: np.ndarray, null=None) -> float:
    """``log sum_w + log pdet(H_c)``, ``H_c = X'WX - (X'w)(X'w)'/sum_w + S``, in exact rationals.

    ``null`` is the exact null vector of ``H_c`` (rationals) when it is singular.
    """
    p = X.shape[1]
    sum_w, H = _exact_centred_hessian(X, w, S)
    norm = Fraction(1)
    if null is not None:
        assert all(sum(H[i][j] * null[j] for j in range(p)) == 0 for i in range(p))
        norm = sum(v * v for v in null)
        H = [[H[i][j] + null[i] * null[j] for j in range(p)] for i in range(p)]
    # Gaussian elimination in rationals: the determinant
    det = Fraction(1)
    for c in range(p):
        pivot = next(r for r in range(c, p) if H[r][c] != 0)
        if pivot != c:
            H[c], H[pivot] = H[pivot], H[c]
            det = -det
        det *= H[c][c]
        for r in range(c + 1, p):
            if H[r][c] != 0:
                factor = H[r][c] / H[c][c]
                H[r] = [a - factor * b for a, b in zip(H[r], H[c], strict=True)]
    assert det > 0
    log = lambda value: math.log(value.numerator) - math.log(value.denominator)  # noqa: E731
    return log(sum_w) + log(det) - log(norm)


# ------------------------------------------------------ 3.2: the border centre
def test_the_border_centre_is_the_shifted_prior_weighted_mean_chosen_by_type():
    """Section 3.2, T3 rows: spread-rule, unweighted and unshifted centring all fail here.

    A column constant (7.25) on the rows of positive prior weight, whatever it
    is on zero-weight rows, centres to exactly that constant, so its centred
    rows are exact zeros where the likelihood reads them (an unshifted
    weighted mean leaves a residue, an unweighted one the zero-weight rows'
    pull).  A shift of a column moves its centre by the shift to one rounding,
    whether or not the offset exceeds the spread (the spread rule centres one
    and not the other).  A one-hot block keeps 0.
    """
    rng = np.random.default_rng(3)
    n = 400
    tree = rng.integers(0, 12, n)
    weights = np.exp(rng.normal(size=n))
    weights[:40] = 0.0
    constant = np.full(n, 7.25)
    constant[:40] = rng.normal(size=40) * 1e3
    base = 0.1 * rng.normal(size=n)  # |mean| far below its spread
    columns = [constant, base, base + 1e8]
    dm, groups, layout = _border_layout(columns, rng.integers(0, 5, n), weights, 5, 12, tree)
    center, generators = nested_prior_statistics(layout, weights)
    assert center[0] == 7.25
    shift = center[2] - center[1]
    assert abs(shift - 1e8) <= 4 * EPS * 1e8
    reference = base[40] + np.sum(weights * (base - base[40])) / np.sum(weights)
    assert abs(center[1] - reference) <= 4 * n * EPS * np.sum(np.abs(base)) / n
    assert np.all(center[3:] == 0.0)
    # the complete RE block is the structural null generator, with its most
    # prior-exposed level as the reference
    exposure = np.bincount(dm.group_matrices[1].codes, weights=weights, minlength=5)
    assert generators is not None and generators.count == 1
    assert generators.references[0] == 3 + int(np.argmax(exposure))
    # the prior weights, not the working weights, key the centre
    assert nested_prior_statistics(layout, weights)[0] is center
    assert not np.array_equal(nested_prior_statistics(layout, None)[0], center)


# ------------------------------------------- 3.6: deflation, dpstrf, Rump, disclosure
def _complete_block(lam: float):
    """A 9-level chain beside ``[x, x^2]`` and a complete 6-level RE block at ``lambda``."""
    rng = np.random.default_rng(11)
    n, K, T = 240, 6, 9
    tree = rng.integers(0, T, n)
    crossed = rng.integers(0, K, n)
    weights = np.exp(rng.normal(size=n)) * 1e4
    x = rng.normal(size=n)
    dm, groups, layout = _border_layout([x, x * x], crossed, weights, K, T, tree)
    matrices = list(dm.group_matrices)
    components = [_identity(groups, 1), _identity(groups, 2)]
    lambdas = {"crossed": lam, "tree": 50.0}
    system = build_nested_structured_system(
        matrices, groups, weights, 0.3 * weights, layout=layout, prior_weights=weights
    )
    penalized = build_penalized_nested_operator(
        system, matrices, groups, lambdas, reml_penalties=components
    )
    factor, _ = build_augmented_nested_factor(system, penalized)
    S = np.zeros((dm.p, dm.p))
    S[groups[1].sl, groups[1].sl] = lam * np.eye(K)
    S[groups[2].sl, groups[2].sl] = 50.0 * np.eye(T)
    tree_floor = 10 * EPS * (np.bincount(tree).max() + 2) * dm.p
    return dm, groups, weights, system, factor, S, tree_floor


@pytest.mark.parametrize("lam", [1e-7, 1e-9])
def test_a_complete_border_block_is_deflated_to_the_exact_rank_and_logdet(lam):
    """Section 3.6 step 1, T3 row "structural deflation removed" and "fixed cutoff".

    A complete one-hot border block sums to the intercept, so after the
    super-root its data curvature along ``1_g`` is exactly zero and only
    ``K lambda`` carries it: far below the rounding ``u_s`` of the border for
    a tiny ``lambda``.  Deflated by type, the factor keeps the direction and
    its ``log|H|`` matches the exact rational ``log sum_w + log |H_c|`` within
    the certificate's first-order bound plus the tree's pivot floors.
    Without the deflation the rank drops by one and ``log|H|`` moves by
    ``log(K lambda / u_s)``-sized amounts.
    """
    dm, _, weights, _, factor, S, tree_floor = _complete_block(lam)
    assert factor.rank == dm.p + 1 and not factor.rank_truncated
    certificate = factor.border_certificate
    assert certificate.deflated == 1 and certificate.decrements == 0
    exact = _exact_published_logdet(dm.toarray(), weights, S)
    assert abs(factor.logdet() - exact) <= certificate.logdet_bound + tree_floor


@pytest.mark.parametrize("lam", [1e-7, 1e-9])
def test_tree_variances_beside_a_deflated_block_read_the_data_side_inverse(lam):
    """Section 3.6 step 1: ``diag(H^-1)`` on the tree is ``Z_uu + F_u Q^+ F_u'``.

    ``F = T^-1 C`` is data, so ``F v = 0`` on the deflated structural null and
    the product reads the data-side inverse (``BorderFactor.inverse_data``).
    The full ``Q^+`` carries ``1 / (K lambda)`` along that null, and its
    explicit product with ``F`` cancels those entries in floating point: at
    ``lambda = 1e-9`` the tree's trace moves by far more than the bound below
    (the ``crossed_re`` grid measured 1.9 of edf).  The exact reference is
    ``sum_{j in tree} (H_c^-1)_jj`` in rationals.  The bound: the border's
    backward perturbation moves ``tr(H^-1 lambda_t Omega)`` by at most the
    certificate's first-order ``logdet_bound`` (``|tr(H^-1 E H^-1 S)| <=
    ||H^-1/2 E H^-1/2||_* ||H^-1/2 S H^-1/2||`` and the second factor is at
    most 1), the tree pivots by ``tree_floor`` of the trace, and the explicit
    products ``F_u Q_data F_u'`` by ``gamma_{2q+2} sum_u |F_u| |Q_data| |F_u|'``
    (Higham 2002, section 3.5).
    """
    dm, groups, weights, system, factor, S, tree_floor = _complete_block(lam)
    data = system.operator
    xtw = np.empty(data.shape[0])
    xtw[data.small_indices], xtw[data.structured_indices] = system.xtw_small, system.xtw_structured
    profiled = ProfiledNestedSchurFactor(
        augmented_factor=factor, sum_w=system.sum_w, xtw=xtw, data_operator=data
    )
    tree = groups[2].sl
    _, H = _exact_centred_hessian(dm.toarray(), weights, S)
    exact = float(sum(_exact_inverse_diagonal(H, range(tree.start, tree.stop))))
    actual = profiled.trace_inverse_penalty(_identity(groups, 2))
    q = len(factor.small_indices)
    gamma = (2 * q + 2) * EPS / (1.0 - (2 * q + 2) * EPS)
    Q_data = np.abs(factor._Q_inverse_data)
    products = sum(float(np.sum((np.abs(F) @ Q_data) * np.abs(F))) for F in factor._F)
    bound = (
        (factor.border_certificate.logdet_bound + tree_floor) / 50.0
        + gamma * products
        + tree_floor * exact
    )
    assert abs(actual - exact) <= bound


@pytest.mark.parametrize("lam", [1e-7, 1e-9])
def test_a_newton_right_hand_side_reads_the_data_side_inverse(lam):
    """Section 3.6 step 1: PIRLS's ``solve_data`` of a data-derived ``A'Wz``.

    ``A'Wz`` (``A = [1, X]``) is exactly orthogonal to the deflated structural
    null, but its float64 rounding is not; the full ``Q^+`` amplifies that
    rounding by ``1 / (K lambda)`` along the null (1.5e-4 and 1.7e-2 of the
    solution at these lambdas), which stalled PIRLS on ``crossed_re``.  The
    data side drops it, so the solve is as accurate as the deflated system is
    conditioned.  The reference solves ``H_aug x = A'Wz`` in exact rationals.
    Bound: the verified factorization is backward stable to ``tau`` in the
    Jacobi-scaled deflated system, so the scaled solution moves by at most
    ``tau ||Q_s^-1|| <= tau tr(Q_s^-1) = logdet_bound`` of itself (first
    order); the tree's pivots add ``tree_floor``, the rounding of the
    right-hand side ``(p + 3) eps``, and the scaling's spread converts to raw
    coordinates.
    """
    dm, _, weights, _, factor, S, tree_floor = _complete_block(lam)
    A = np.hstack((np.ones((dm.n, 1)), dm.toarray()))
    m = A.shape[1]
    z = np.random.default_rng(3).normal(size=dm.n)
    rows = [[Fraction(float(v)) for v in row] for row in A]
    w = [Fraction(float(v)) for v in weights]
    wz = [wr * Fraction(float(v)) for wr, v in zip(w, z, strict=True)]
    rhs = [sum(rows[r][j] * wz[r] for r in range(dm.n)) for j in range(m)]
    penalty = np.zeros((m, m))
    penalty[1:, 1:] = S
    system = [
        [
            sum(w[r] * rows[r][i] * rows[r][j] for r in range(dm.n))
            + Fraction(float(penalty[i, j]))
            for j in range(m)
        ]
        + [rhs[i]]
        for i in range(m)
    ]
    for c in range(m):
        pivot = next(r for r in range(c, m) if system[r][c] != 0)
        system[c], system[pivot] = system[pivot], system[c]
        system[c] = [value / system[c][c] for value in system[c]]
        for r in range(m):
            if r != c and system[r][c] != 0:
                factor_rc = system[r][c]
                system[r] = [a - factor_rc * b for a, b in zip(system[r], system[c], strict=True)]
    exact = np.array([float(system[i][m]) for i in range(m)])
    actual = factor.solve_data(np.array([float(value) for value in rhs]))
    diagonal = np.diag(A.T @ (weights[:, None] * A) + penalty)
    spread = float(np.sqrt(diagonal.max() / diagonal.min()))
    bound = (factor.border_certificate.logdet_bound + tree_floor + (m + 2) * EPS) * spread
    assert np.max(np.abs(actual - exact)) <= bound * np.max(np.abs(exact))


def test_rump_verification_rejects_a_pivot_that_overestimates_its_eigenvalue():
    """Section 3.6 step 3, T3 row "Rump verification removed (trust dpstrf's rank)".

    For ``[[1, 1 - d], [1 - d, 1]]`` complete pivoting's last pivot is ``2d -
    d^2`` while ``lambda_min = d``: with ``tau`` between them ``dpstrf`` keeps
    rank 2, and only the verification of the retained block (Rump 2006,
    Corollary 2.4: the Cholesky of ``A - (c_R + tau) I`` must complete) finds
    that no matrix in the error ball is certified positive definite there.
    """
    d = 7e-13
    Q = np.array([[1.0, 1.0 - d], [1.0 - d, 1.0]])
    u_s = 1e-12  # the bound U is set so that tau sits between d and 2d
    border = factor_border(Q, np.zeros((2, 2)), np.full(2, u_s / 2), None, term_name="probe")
    certificate = border.certificate
    assert d < certificate.tau < 2 * d - d * d
    assert certificate.rank == 1 and certificate.decrements == 1
    assert len(certificate.directions) == 1
    # the residual bound on the truncated cluster bounds its eigenvalue d from
    # above, or is inf where the residual exceeds the certified gap
    assert certificate.trailing_bound >= d


def test_the_pivot_tolerance_is_derived_so_a_certified_direction_is_kept():
    """Section 3.6 step 2, T3 row "fixed 1e-10 cutoff".

    ``[[1, 1 - d], [1 - d, 1]]`` with ``d = 2^-40`` (exact in float64) has
    ``lambda_min = d``, about 9e-13, with entries known to ``eps``.  The derived
    ``tau = c_R(2) + u_s`` is about 1e-15, so the direction is certified and
    kept, and the log-determinant is ``log(2d - d^2)`` within the certificate's
    first-order bound; a fixed relative cutoff of 1e-10 truncates it and loses a
    rank that the data identify.
    """
    d = 2.0**-40
    Q = np.array([[1.0, 1.0 - d], [1.0 - d, 1.0]])
    border = factor_border(Q, np.zeros((2, 2)), np.full(2, EPS), None, term_name="probe")
    certificate = border.certificate
    assert certificate.tau < d
    assert certificate.rank == 2 and certificate.decrements == 0
    assert abs(border.logdet - math.log(2.0 * d - d * d)) <= certificate.logdet_bound


def test_a_border_refusal_names_the_term_kind_and_no_internal_column():
    """The border factorization is shared by the nested chain and the fs and sz
    factors: a refusal names the term as its caller describes it, and not by a
    1-based index into the deflated rest coordinates, which is no model column."""
    Q = np.array([[1.0, 0.0], [0.0, -5.0]])
    with pytest.raises(
        np.linalg.LinAlgError,
        match=r"^FactorSmooth term 'x:g:sz' has materially negative Schur curvature -5 on its border",
    ) as refusal:
        factor_border(
            Q,
            np.zeros((2, 2)),
            np.full(2, EPS),
            None,
            term_name="x:g:sz",
            term_kind="FactorSmooth term",
        )
    assert "border column" not in str(refusal.value)


def test_a_truncated_border_inverts_its_compression_onto_the_retained_subspace():
    """Section 3.6, the generalized inverse (review P1: "remove truncated curvature").

    ``[[1, 0.8], [0.8, 1]]`` with bound ``0.3`` per column has ``tau`` about 0.6
    above its smaller eigenvalue 0.2, so one direction is truncated: complete
    pivoting keeps column 0 and ``Z`` spans ``[-0.8, 1]``, along which ``Q`` keeps
    curvature (``Q Z != 0``).  The inverse, the solve and ``log pdet`` must all
    describe the compression ``A = P Q P``, ``P = I - ZZ'``: ``A^+ = v v' / (v'Qv)``
    with ``v = [1, 0.8] / |.|``, ``v'Qv = 2.92 / 1.64``, so ``A^+ = [[1, 0.8],
    [0.8, 0.64]] / 2.92``.  Fails with the uncompressed ``(Q + ZZ')^-1 - ZZ'``,
    which has the eigenvalue -0.177 and ``log det(Q + ZZ') = 0.761``.  Every
    quantity is a 2 x 2 solve with ``kappa(A + ZZ') < 2``: ``8 eps`` covers it.
    """
    Q = np.array([[1.0, 0.8], [0.8, 1.0]])
    border = factor_border(Q, np.zeros((2, 2)), np.full(2, 0.3), None, term_name="probe")
    assert border.certificate.rank == 1 and border.null.shape[1] == 1
    expected = np.array([[1.0, 0.8], [0.8, 0.64]]) / 2.92
    bound = 8 * EPS
    assert np.max(np.abs(border.inverse - expected)) <= bound
    assert np.linalg.eigvalsh(border.inverse)[0] >= -bound
    assert abs(border.logdet - math.log(2.92 / 1.64)) <= bound
    rhs = np.array([[1.0, 0.0], [0.0, 1.0], [0.3, -2.0]]).T
    np.testing.assert_allclose(border.apply_data(rhs), expected @ rhs, rtol=0.0, atol=bound * 3)


def test_a_fit_whose_border_truncates_weak_columns_publishes_a_semidefinite_covariance():
    """Section 3.6 (review P1), as a complete fit: an ``fs`` term (eight levels of
    30 evenly spaced rows, ``k = 5``) beside two numeric columns constant within
    levels, every smoothing parameter fixed at 1e-22.  Both border columns
    then lie in the level functions' span to within the penalty, and the
    border truncates them with curvature of the penalty's size left on them.

    The published ``H^+`` must be positive semidefinite to its rounding (its
    eigenvalues are sums of the congruences of positive semidefinite blocks,
    so within ``gamma_p ||H^+||``), and ``edf = tr(H^+ X'WX) <= tr(H^+ H)``,
    which is the factor's rank because ``H^+`` annihilates the dropped Ritz
    block and residual (the fixture's data rank, 40, leaves a unit for
    rounding).  The uncompressed inverse gives variances near -3.6e21, an
    eigenvalue near -9.8e21 and edf 41.08 above the rank 41.
    """
    rng = np.random.default_rng(3)
    K, rows = 8, 30
    g = np.repeat(np.arange(K), rows)
    x = np.tile(np.linspace(0.0, 1.0, rows), K)
    a, b = rng.normal(size=K)[g], rng.normal(size=K)[g]
    y = (
        1.0
        + 0.5 * a
        - 0.3 * b
        + np.sin(3.0 * x) * (1.0 + 0.1 * g)
        + 0.1 * rng.normal(size=K * rows)
    )
    frame = pd.DataFrame(
        {"x": x, "a": a, "b": b, "g": np.array([f"g{c}" for c in g], dtype=object)}
    )
    policy = {c: LambdaPolicy.fixed(1e-22) for c in ("wiggle", "null_0", "null_1")}
    model = SuperGLM(
        family="gaussian",
        features={"a": Numeric(), "b": Numeric()},
        interactions=[FactorSmooth("x", group="g", basis="fs", k=5, lambda_policy=policy)],
        selection_penalty=0,
        direct_solve="structured",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_reml(frame, y)
    factor = model._linear_system_state.augmented_factor
    assert factor.border_certificate.rank < factor.border_certificate.width
    p = factor.shape[0]
    inverse = factor.selected_inverse_block(np.arange(p))
    eigenvalues = np.linalg.eigvalsh(0.5 * (inverse + inverse.T))
    gamma = p * EPS / 2 / (1.0 - p * EPS / 2)
    assert eigenvalues[0] >= -gamma * eigenvalues[-1]
    assert np.min(np.diag(inverse)) >= -gamma * np.max(np.diag(inverse))
    assert model._result.effective_df <= factor.rank


# ------------------------------------------- 3.6: the published convention
def test_an_alias_through_the_intercept_is_published_in_the_centred_convention():
    """Section 3.6 (critic F8), T3 rows "coupled-null refusal kept" and "log det(I + (FZ)'FZ) added".

    ``x = a[cat] + 1000`` lies in the span of the intercept and the
    categorical dummies, so ``H`` has an exact null whose intercept component
    is not zero.  The factor refuses nothing and publishes ``det(T)
    pdet(Q)``, which is exactly the dense backend's ``log sum_w + log
    pdet(H_c)``; a coupling term or a refusal of the coupled null breaks it.
    """
    rng = np.random.default_rng(4)
    n, T, C = 300, 8, 4
    tree = rng.integers(0, T, n)
    cat = rng.integers(0, C, n)
    attribute = rng.normal(0, 3, C)
    weights = np.exp(rng.normal(size=n))
    x = attribute[cat] + 1000.0
    dummies = np.eye(C)[cat][:, 1:]  # base level 0 dropped
    dm, groups, layout = _border_layout(
        [np.column_stack([x[:, None], dummies])[:, j] for j in range(C)],
        rng.integers(0, 3, n),
        weights,
        3,
        T,
        tree,
    )
    matrices = list(dm.group_matrices)
    components = [_identity(groups, 1), _identity(groups, 2)]
    lambdas = {"crossed": 0.8, "tree": 3.0}
    system = build_nested_structured_system(
        matrices, groups, weights, 0.3 * weights, layout=layout, prior_weights=weights
    )
    penalized = build_penalized_nested_operator(
        system, matrices, groups, lambdas, reml_penalties=components
    )
    factor, _ = build_augmented_nested_factor(system, penalized)
    p = dm.p
    assert factor.rank == p  # p + 1 augmented coordinates, nullity one
    X = dm.toarray()
    S = np.zeros((p, p))
    S[groups[1].sl, groups[1].sl] = 0.8 * np.eye(3)
    S[groups[2].sl, groups[2].sl] = 3.0 * np.eye(T)
    null = [Fraction(0)] * p
    null[0] = Fraction(1)
    x0 = Fraction(float(x[cat == 0][0]))
    for k in range(1, C):
        null[k] = -(Fraction(float(x[cat == k][0])) - x0)
    exact = _exact_published_logdet(X, weights, S, null=null)
    certificate = factor.border_certificate
    tree_floor = 10 * EPS * (np.bincount(tree).max() + 2) * p
    # the rational reference's logarithm is formed in float64 within 4 ulp
    assert abs(factor.logdet() - exact) <= certificate.logdet_bound + tree_floor + 4 * EPS * abs(
        exact
    )
    # disclosed: one truncated direction on the alias's own columns, unpenalized
    assert len(certificate.directions) == 1 and not certificate.weak.any()


# ------------------------------------------------------------- full fits
def _raw_offset_frame(offset: float):
    frame, eta, weight, rng, levels = _adversarial_base()
    n = len(frame)
    frame["xr"] = offset + rng.normal(size=n)
    frame["vr"] = 1e3 * (offset + rng.normal(size=n))
    eta = eta + 0.1 * (frame["xr"].to_numpy() - offset)
    return frame, eta, weight, rng, levels


def _influence_reference(model, weight):
    """``w_i / sum w + w_i x~_i' H_c^-1 x~_i`` by a dense Cholesky of the centred system, and
    the per-row distance it and any other backward-stable evaluation keep from the exact
    value of the same design.

    ``x~ = x - c`` about the shifted weighted mean ``c``.  Rounding in ``c``, in each
    difference and in the design's own entries (the raw-offset columns round their draw at
    ``u |x|``, twice for ``1e3 (offset + z)``) perturbs a row by ``e_i`` with ``|e_ij| <=
    gamma_{n+3} sum_r w_r |x_rj - x_ref,j| / sum w + u |c_j| + u |x~_ij| + 2 u |x_ij|``
    (Higham 2002, Lemma 3.1), which moves ``x~' H_c^-1 x~`` by at most ``2 sqrt(s t) + s``
    with ``s = ||D e||^2 / lambda_min(H_s)``, ``H_s = D H_c D`` the Jacobi scaling and ``t``
    the form itself.  The solves rest on backward-stable symmetric eliminations,
    ``2 gamma_{3p+1} p / ((1 - gamma_{p+1}) lambda_min(H_s))`` of the form for two of them
    (Higham 2002, Thm 10.4); ``w_i / sum w`` carries ``gamma_{n+1}``.
    """
    X = model._dm.toarray()
    p, n = X.shape[1], len(weight)
    S = _active_penalty_matrix(
        model._dm.group_matrices,
        model._groups,
        model._groups,
        fitted_lambda2(model),
        reml_penalties=model._reml_penalties,
    )
    W = np.asarray(weight, dtype=np.float64)
    total = float(np.sum(W))
    anchor = X[np.flatnonzero(W)[0]]
    center = anchor + (X - anchor).T @ W / total
    centred = X - center
    H = centred.T @ (W[:, None] * centred) + S
    H = 0.5 * (H + H.T)
    forms = np.sum(
        centred * scipy.linalg.cho_solve(scipy.linalg.cho_factor(H), centred.T).T, axis=1
    )
    scale = 1.0 / np.sqrt(np.diag(H))
    smallest = float(np.linalg.eigvalsh(scale[:, None] * H * scale[None, :])[0])
    u = EPS / 2.0

    def gamma(k):
        return k * u / (1.0 - k * u)

    error = gamma(n + 3) * (np.abs(X - anchor).T @ W) / total + u * np.abs(center)
    rows = (error + u * np.abs(centred) + 2.0 * u * np.abs(X)) * scale
    shift = np.sum(rows**2, axis=1) / smallest
    solve = 2.0 * gamma(3 * p + 1) * p / ((1.0 - gamma(p + 1)) * smallest)
    reference = W / total + W * forms
    gap = W * (solve * forms + 2.0 * np.sqrt(shift * forms) + shift) + gamma(n + 1) * W / total
    return reference, gap, p * solve


def test_leverage_is_the_influence_diagonal_with_the_intercept():
    """Section 3.10, decision (A); T3 row "raw coefficient factor or derived view feeding leverage".

    ``h_i = w_i a_i' H_aug^+ a_i`` with ``a_i = [1, x_i]``: shift-invariant, and its sum
    is the model's edf.  Gaussian/identity at fixed penalties makes the working weights
    the prior weights, so two designs equal up to a column offset (``1e2`` and ``1e8``)
    have the same exact leverage up to the rounding of their own entries.  Each fit's rows
    are within ``_influence_reference``'s distance (doubled: the fit and the reference are
    each that far from the exact value) of the dense centred reference, the sum within
    those distances and the edf's own ``p`` trace terms, and the two offsets within the sum
    of both.  The conditional ``w_i x_i'(X'WX + S)^-1 x_i`` of the raw columns moves with
    the offset by O(1) and sums to about ``edf - 1``.
    """
    rows, gaps = {}, {}
    for offset in (1e2, 1e8):
        frame, eta, weight, _, levels = _raw_offset_frame(offset)
        y = eta + np.random.default_rng(99).normal(size=len(eta))
        model = _fit(frame, y, weight, "gaussian", ["x1", "xr", "vr"], levels, lam_g=0.5, lam_u=2.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            leverage = np.asarray(model.metrics(frame, y, sample_weight=weight).leverage)
        reference, gap, trace = _influence_reference(model, weight)
        np.testing.assert_array_less(np.abs(leverage - reference), 2.0 * gap + np.finfo(float).tiny)
        edf = float(model.result.effective_df)
        assert abs(float(np.sum(leverage)) - edf) <= 2.0 * float(np.sum(gap)) + trace * edf
        rows[offset], gaps[offset] = leverage, gap
    np.testing.assert_array_less(
        np.abs(rows[1e2] - rows[1e8]), 2.0 * (gaps[1e2] + gaps[1e8]) + np.finfo(float).tiny
    )


@pytest.mark.slow
def test_pirls_stops_on_the_certificate_score_at_a_raw_offset():
    """Section 3.8; T3 row "raw-intercept step rule".

    Gamma with a column at ``|mean| / sd = 1e8``: PIRLS on the step length of
    the raw intercept stopped a few iterations short of the certificate (the
    speed verifier measured 1.9e-9 against the old fixed 1e-9 bar), or was held
    by the objective's rounding in raw coordinates.  Stopping on the
    certificate's own score from the centred state ``(alpha, beta)`` certifies
    the published mode, and the geometry's own reading of the score agrees.
    Fails with the step rule (``convergence="coefficients"``) for the REML
    PIRLS and the terminal refit: the terminal mode is then not certified.
    """
    frame, eta, weight, rng, levels = _raw_offset_frame(1e8)
    y = _response("gamma", eta, rng)
    model = _fit(frame, y, weight, "gamma", ["x1", "xr", "vr"], levels)
    profile = model._reml_profile
    assert profile["reml_terminal_mode_certified"] is True
    assert profile["reml_terminal_observed_mode_residual"] <= BAR
    assert not profile["reml_terminal_mode_floor_binding"]
    assert model.result.centred_intercept is not None


@pytest.mark.slow
def test_a_weakly_identified_coefficient_is_flagged_and_kept():
    """Section 3.9; T3 row "weakly identified coefficient inside the certificate".

    ``xt`` is 5 everywhere except two rows of prior weight 1e-15: its curvature
    lives at the rounding of the largest weight, the likelihood is flat along
    it to float64, and no Newton step places it.  The fit certifies the
    identified coefficients, keeps ``xt`` and names it; gating on it refuses.
    """
    frame, eta, weight, rng, levels = _adversarial_base()
    rows = np.flatnonzero(weight > 0)[:2]
    frame["xt"] = 5.0
    frame.loc[rows, "xt"] = [6.0, 7.0]
    weight = weight.copy()
    weight[rows] = 1e-15
    weight = np.where(weight == 0.0, 0.01, weight)
    y = _response("tweedie", eta, rng)
    model = _fit(frame, y, weight, "tweedie", ["x1", "xt"], levels)
    profile = model._reml_profile
    xt = next(group for group in model._groups if group.name == "xt").start
    assert xt in profile["reml_terminal_weakly_identified"]
    assert profile["reml_terminal_observed_mode_residual"] <= BAR


@pytest.mark.slow
def test_a_rare_offset_column_keeps_the_dense_rank_and_its_standard_errors():
    """Section 3.2 and 4.2, the verifier's ``offset_rare`` as a complete fit.

    The builder's rare column shifted by 5: before stage 0 (spread-rule centre,
    eigenvalue cutoff) the chain lost a rank against gram with NaN standard
    errors on estimable coefficients; the unfixed implementation fails this
    test.  The prior-weighted centre is exactly 5 on the weighted rows.  With
    the deflation and the verified pivoted factorization this fit no longer
    turns on the centre alone, so the spread-rule mutation is caught by the
    centre's unit tests (the first test here and the plumbing module).
    """
    frame, eta, weight, rng, levels = _adversarial_base()
    rows = np.flatnonzero(weight > 0)[:2]
    frame["xt"] = 5.0
    frame.loc[rows, "xt"] = [6.0, 7.0]
    weight = weight.copy()
    weight[rows] = 1e-15
    y = _response("poisson", eta, rng)
    model = _fit(frame, y, weight, "poisson", ["x1", "xt"], levels)
    gram = _fit(frame, y, weight, "poisson", ["x1", "xt"], levels, solve="gram")
    assert model.result.reml_hessian_rank == gram.result.reml_hessian_rank
    assert _nan_se(model, frame, y, weight) == _nan_se(gram, frame, y, weight)


# ------------------------------------------- the centred state after the fit (3.8)
def _offset_fit(family: str, *, numeric: bool = True):
    """The Sol review's fixture: an integer-step column at ``1e16`` beside a fixed RE."""
    rng = np.random.default_rng(10)
    n = 240
    g = np.tile(np.arange(40), 6)
    z = 2.0 * rng.integers(-4, 5, n)
    if family == "poisson":
        y = rng.poisson(np.exp(0.5 + 0.1 * z + 0.05 * np.sin(g))).astype(float)
    else:
        y = 3.0 + 0.2 * z + 0.05 * np.sin(g) + 0.01 * rng.normal(size=n)
    frame = pd.DataFrame({"x": 1e16 + z, "g": g})
    features = {"g": RandomEffect(lambda_policy=LambdaPolicy.fixed(1.0))}
    if numeric:
        features = {"x": Numeric(), **features}
    model = SuperGLM(family=family, features=features, selection_penalty=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_reml(frame, y)
    return model, frame, y


@pytest.mark.parametrize("family", ["gaussian", "poisson"])
def test_every_reading_of_a_fit_uses_its_centred_predictor(family):
    """Section 3.8: prediction and the post-fit consumers read ``alpha + (X - 1 c') beta``.

    With ``x = 1e16 + z`` the raw intercept is about ``-1e16 beta_x``, so ``X beta +
    intercept`` cancels to ``u 1e16 |beta_x|``: the review measured predictor errors of
    0.15 and a training-row deviance of 1.63 against the fit's 0.028.  Prediction on new
    rows centres the dense column before its product (``x - c`` is exact here, Sterbenz),
    so the two evaluations of the same three terms differ by ``2 gamma_3`` of their
    absolute sum.  The metrics' deviance is the fit's to the ``gamma_n`` of a sum of
    non-negative terms (the same ``mu``), and the working weights are the Fisher
    weights at the fit's own ``eta``.  Fails with ``_predict_eta``,
    ``_working_eta_mu`` or ``_solver_space_working_weights`` reading ``X beta +
    intercept``.
    """
    from superglm.model.state_ops import _solver_space_working_weights
    from superglm.solvers.mode_score import linear_predictor

    model, frame, y = _offset_fit(family)
    solver = model._solver_pirls_result()
    fitted = linear_predictor(model._dm, solver, None)
    u = EPS / 2

    def gamma(k: int) -> float:
        return k * u / (1.0 - k * u)

    alpha, centre = model.result.centred_intercept, model.result.state_center
    assert alpha is not None and centre is not None
    x = next(group for group in model._groups if group.name == "x").start
    beta = model.result.beta
    scale = (
        abs(alpha) + np.abs(frame["x"].to_numpy() - centre[x]) * abs(beta[x]) + np.max(np.abs(beta))
    )
    predicted = np.asarray(model._predict_eta_raw_exact(frame))
    np.testing.assert_array_less(np.abs(predicted - fitted), 2.0 * gamma(3) * scale)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        deviance = float(model.metrics(frame, y).deviance)
    assert abs(deviance - solver.deviance) <= 2.0 * gamma(len(y)) * solver.deviance
    expected = np.exp(fitted) if family == "poisson" else np.ones_like(fitted)
    np.testing.assert_allclose(_solver_space_working_weights(model), expected, rtol=4 * u, atol=0)


def test_a_fit_without_a_dense_column_keeps_its_raw_predictor():
    """Only a dense column needs the centre: without one the public result carries no
    centred state and predictions are the raw ``intercept + sum_t score_t`` as before."""
    model, frame, _ = _offset_fit("gaussian", numeric=False)
    assert model.result.centred_intercept is None and model.result.state_center is None
    g = next(group for group in model._groups if group.name == "g")
    expected = model.result.intercept + model.result.beta[g.sl][frame["g"].to_numpy()]
    np.testing.assert_array_equal(model._predict_eta_raw_exact(frame), expected)


def test_a_coefficient_revision_returns_the_predictor_to_the_raw_state():
    """The centred state is the fitted mode's: a revision it does not cover clears it.

    The editor's revision writes raw coordinates (``_patch_beta_block``); keeping
    ``(alpha, c)`` left ``linear_predictor`` at the old intercept, off by ``c' dbeta``
    (1e4 here).  After ``invalidate_revised_coefficient_mode`` both results read
    ``X beta + intercept`` and prediction scores the revised coefficients.  (A null
    revision keeps the state, and predictions bit for bit: ``test_piecewise_editor``.)
    """
    from superglm.editor.apply import _copy_model_for_editor_edits, _patch_beta_block
    from superglm.model.fit_state import (
        FittedStateRevision,
        invalidate_revised_coefficient_mode,
    )
    from superglm.solvers.mode_score import linear_predictor

    rng = np.random.default_rng(0)
    n = 400
    frame = pd.DataFrame({"x": rng.uniform(size=n), "t": 1e6 + rng.normal(size=n)})
    y = np.sin(3 * frame["x"]) + 0.5 * (frame["t"] - 1e6) + 0.1 * rng.normal(size=n)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"x": Spline(kind="ps", k=8), "t": Numeric()},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_reml(frame, y.to_numpy())
    assert model._solver_pirls_result().centred_intercept is not None
    copied = _copy_model_for_editor_edits(model, share_transient_state=True)
    revised = FittedStateRevision.start(copied, increment=True, freeze_auxiliary_arrays=True).model
    t = next(group for group in revised._groups if group.name == "t")
    _patch_beta_block(revised, [t], revised.result.beta[t.sl] + 0.01)
    invalidate_revised_coefficient_mode(revised)
    for result in (revised._result, revised._solver_result):
        assert result.centred_intercept is None and result.state_center is None
    solver = revised._solver_result
    raw = revised._dm.matvec(solver.beta) + solver.intercept
    np.testing.assert_array_equal(linear_predictor(revised._dm, solver, None), raw)
    magnitude = abs(solver.intercept) + np.abs(revised._dm.toarray()) @ np.abs(solver.beta)
    bound = 2.0 * (revised._dm.p + 2) * EPS * magnitude
    np.testing.assert_array_less(np.abs(revised._predict_eta_raw_exact(frame) - raw), bound)
