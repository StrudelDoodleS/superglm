"""Signed working rows on the nested chain (one-engine design §3.3 and §3.11).

The chain's construction for any sign pattern of the rows (the signed-rows
research note, §4): leaf and node centres on the absolute weights, the signed
deviation carried up the tree, the carried terms of ``Q`` and the ``F``/``e``
recursion, the running error bound and the tree pivot certificate.  Every
factor here is built by the production row pass from the same float64 rows as
an exact rational reference: ``H = [1 X]' W [1 X] + blockdiag(0, S)`` and its
``LDL'`` without pivoting, which certifies positive definiteness (Sylvester)
and gives ``log det H`` and ``diag(H^-1)`` exactly.

Bounds.  The chain and the dense reference both form ``H`` from rows, a
computation backward stable to ``gamma = (n + p + 10) eps`` of the absolute
companion ``A = |X|' |W| |X| + S`` entrywise (Higham 2002, Lemma 3.1; the
companion is what signed rows are accurate against, note §1).  In the
Jacobi-scaled metric ``H_s = D H D``, ``D = diag(H)^-1/2``, Weyl's inequality
then bounds ``|d log det H| <= p gamma ||A_s||_2 / lambda_min(H_s)`` and the
relative error of every ``(H^-1)_ii`` by ``gamma ||A_s||_2 / lambda_min(H_s)``
(first order: ``|e_i' H^-1 dH H^-1 e_i| <= ||dH_s|| (H_s^-1)_ii /
lambda_min``), twice that for the second order.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

from superglm.solvers.irls_direct import _build_iterate_factor
from superglm.solvers.structured import (
    build_augmented_nested_factor,
    build_nested_structured_system,
    build_penalized_structured_operator,
    solve_augmented_normal_equations,
)
from tests.test_nested_structured_plumbing import (
    CHAIN,
    VARIANT,
    _components,
    _dense_penalty,
    _lambdas,
    _layout,
    _nested_case,
    _with_dense,
)

EPS = np.finfo(np.float64).eps


# ── the exact reference ───────────────────────────────────────────────────


def _dyadic(values: np.ndarray) -> tuple[np.ndarray, int]:
    exponent = 0
    for value in np.unique(values):
        _, denominator = float(value).as_integer_ratio()
        exponent = max(exponent, denominator.bit_length() - 1)
    integers = np.empty(values.shape, dtype=object)
    for index, value in np.ndenumerate(values):
        numerator, denominator = float(value).as_integer_ratio()
        integers[index] = numerator * ((1 << exponent) // denominator)
    return integers, exponent


def _exact_hessian(X: np.ndarray, w: np.ndarray, S: np.ndarray) -> list[list[Fraction]]:
    """``[1 X]' W [1 X] + blockdiag(0, S)`` exactly from the float64 inputs."""
    design = np.hstack([np.ones((len(w), 1)), X])
    (rows, row_exp), (weights, weight_exp), (penalty, penalty_exp) = (
        _dyadic(design),
        _dyadic(w),
        _dyadic(S),
    )
    size = design.shape[1]
    H = [[Fraction(0)] * size for _ in range(size)]
    scale = Fraction(1, 1 << (2 * row_exp + weight_exp))
    live = [r for r in range(len(w)) if weights[r]]
    for i in range(size):
        for j in range(i, size):
            total = sum(weights[r] * rows[r, i] * rows[r, j] for r in live)
            H[i][j] = H[j][i] = Fraction(total) * scale
    for i in range(size - 1):
        for j in range(size - 1):
            if penalty[i, j]:
                H[i + 1][j + 1] += Fraction(int(penalty[i, j]), 1 << penalty_exp)
    return H


def _log(value: int) -> float:
    shift = max(value.bit_length() - 900, 0)
    return math.log(value >> shift) + shift * math.log(2.0)


def _exact_reference(H: list[list[Fraction]]) -> tuple[float, np.ndarray] | None:
    """``(log det H, diag(H^-1))`` when ``H`` is positive definite, else ``None``."""
    size = len(H)
    A = [row[:] for row in H]
    logdet = 0.0
    for c in range(size):
        pivot = A[c][c]
        if pivot <= 0:
            return None
        logdet += _log(pivot.numerator) - _log(pivot.denominator)
        for r in range(c + 1, size):
            if A[r][c]:
                ratio = A[r][c] / pivot
                A[r] = [a - ratio * b for a, b in zip(A[r], A[c], strict=True)]
    B = [row[:] + [Fraction(int(i == j)) for j in range(size)] for i, row in enumerate(H)]
    for c in range(size):
        inverse = 1 / B[c][c]
        B[c] = [value * inverse for value in B[c]]
        for r in range(size):
            if r != c and B[r][c]:
                ratio = B[r][c]
                B[r] = [a - ratio * b for a, b in zip(B[r], B[c], strict=True)]
    return logdet, np.array([float(B[i][size + i]) for i in range(size)])


def _companion_bound(X: np.ndarray, w: np.ndarray, S: np.ndarray, error: np.ndarray) -> float:
    """``gamma ||A_s||_2 / lambda_min(H_s)``, the relative first-order bound of the module docstring."""
    design = np.hstack([np.ones((len(w), 1)), X])
    penalty = np.zeros((design.shape[1],) * 2)
    penalty[1:, 1:] = S
    H = design.T @ (w[:, None] * design) + penalty
    A = np.abs(design).T @ (error[:, None] * np.abs(design)) + np.abs(penalty)
    scale = 1.0 / np.sqrt(np.abs(np.diag(H)))
    lambda_min = np.linalg.eigvalsh(scale[:, None] * H * scale[None, :])[0]
    gamma = (len(w) + design.shape[1] + 10) * EPS
    return gamma * np.linalg.norm(scale[:, None] * A * scale[None, :], 2) / lambda_min


# ── fixtures ──────────────────────────────────────────────────────────────


def _signed_case(kind: str, seed: int, lam_scale: float, weight_scale: float, offset: float):
    """The plumbing chain beside every border kind, with rows of three sign patterns.

    ``signed``: a fifth of the rows at -1/4 of their weight.  ``cancel``: one
    leaf's rows alternate +1 and -1 (its weights sum to exactly zero), beside
    a tenth of the other rows at -1/5.  ``fisher``: the non-negative rows.  The
    fixture's column of ones would alias the intercept and is replaced; the
    mean-10 column is offset by ``offset``.
    """
    case = _nested_case(seed=seed, n=300, weight_scale=weight_scale)
    dense = np.array(case.dm.group_matrices[0].M)
    dense[:, 0] = np.random.default_rng(seed + 100).normal(size=len(dense))
    dense[:, 2] += offset
    case = _with_dense(case, dense)
    rng = np.random.default_rng(seed)
    w = np.array(case.weights)
    if kind == "signed":
        w = np.where((rng.uniform(size=len(w)) < 0.2) & (w > 0.0), -0.25 * w, w)
    elif kind == "cancel":
        leaf = case.codes[2]
        rows = np.flatnonzero(leaf == 5)
        w[rows] = 1.0
        w[rows[1::2]] = -1.0
        if len(rows) % 2:
            w[rows[-1]] = 0.0
        w = np.where((rng.uniform(size=len(w)) < 0.1) & (w > 0.0) & (leaf != 5), -0.2 * w, w)
    elif kind == "negative-root":
        # a chain of one whose leaf 5 has weight -0.4 lambda: a positive pivot
        # 0.6 lambda under a negative shrunk weight s = -2/3 lambda, so the
        # super-root's |s|-centre and its dense x one-hot correction are live
        leaf = case.codes[2]
        rows = np.flatnonzero(leaf == 5)
        w[rows] = -0.4 * lam_scale * _lambdas()["variant"] / len(rows)
        w = np.where((rng.uniform(size=len(w)) < 0.1) & (w > 0.0) & (leaf != 5), -0.2 * w, w)
    lambdas = {name: value * lam_scale for name, value in _lambdas().items()}
    return case, w, lambdas


def _chain(kind: str) -> tuple[int, ...]:
    return (VARIANT,) if kind == "negative-root" else CHAIN


def _chain_factor(case, w, lambdas, error=None, chain=CHAIN):
    system = build_nested_structured_system(
        case.matrices,
        case.groups,
        w,
        np.zeros_like(w),
        layout=_layout(case, chain),
        prior_weights=np.abs(case.weights),
        error=error,
    )
    penalized = build_penalized_structured_operator(
        system, case.matrices, case.groups, lambdas, reml_penalties=_components(case)
    )
    return system, penalized


CASES = [
    pytest.param(kind, seed, lam, scale, offset, id=f"{kind}-{label}")
    for kind in ("fisher", "signed", "cancel", "negative-root")
    for seed, lam, scale, offset, label in (
        (1, 1.0, 1.0, 0.0, "plain"),
        (2, 1e-7, 1e4, 0.0, "tiny-lambda-large-weights"),
        (3, 1.0, 1.0, 1e6, "offset-1e6"),
    )
]


@pytest.mark.parametrize(("kind", "seed", "lam_scale", "weight_scale", "offset"), CASES)
def test_signed_rows_give_the_exact_logdet_and_inverse_diagonal(
    kind, seed, lam_scale, weight_scale, offset
) -> None:
    """The chain accepts exactly the positive definite ``H`` and matches the exact reference.

    Where the exact ``H`` is indefinite (the cancelling leaf at a tiny penalty
    under large weights: ``lambda_l`` cannot hold its ``-delta_l delta_l' /
    D_l``) the factor refuses; everywhere else it is within the derived
    companion bound.  Centring a cancelling leaf on its signed mean dropped its
    cross moment (``log|H|`` off by up to 14 on the note's fixtures); dropping
    the carried deviation fails these fixtures by 1e-2 and more.
    """
    case, w, lambdas = _signed_case(kind, seed, lam_scale, weight_scale, offset)
    system, penalized = _chain_factor(case, w, lambdas, error=np.abs(w), chain=_chain(kind))
    X = np.hstack([matrix.toarray() for matrix in case.matrices])
    S = _dense_penalty(case, lambdas)
    reference = _exact_reference(_exact_hessian(X, w, S))
    if reference is None:
        with pytest.raises(np.linalg.LinAlgError):
            build_augmented_nested_factor(system, penalized)
    else:
        factor, _ = build_augmented_nested_factor(system, penalized)
        exact_logdet, exact_diagonal = reference
        bound = _companion_bound(X, w, S, np.abs(w))
        assert factor.rank == factor.shape[0]
        assert abs(factor.logdet() - exact_logdet) <= X.shape[1] * bound
        diagonal = factor.selected_inverse_diagonal(np.arange(factor.shape[0]))
        assert np.max(np.abs(diagonal / exact_diagonal - 1.0)) <= 2.0 * bound
    # a leaf without a negative row carries an exact-zero (absent) deviation
    assert (system.operator.leaf.deviation is None) == (kind == "fisher")


def test_the_cancelling_fixture_at_a_tiny_penalty_is_indefinite() -> None:
    """The refusal branch above is reached: the cancelling leaf at lambda 1e-7
    under weights 1e4 makes the exact ``H`` indefinite, and the factor refuses."""
    case, w, lambdas = _signed_case("cancel", 2, 1e-7, 1e4, 0.0)
    X = np.hstack([matrix.toarray() for matrix in case.matrices])
    assert _exact_reference(_exact_hessian(X, w, _dense_penalty(case, lambdas))) is None


def test_a_tree_pivot_inside_its_uncertainty_refuses_the_iterate() -> None:
    """Note §4.4, item 1: ``D_u > E_u + 4 eps (|omega_u| + lambda_u)``.

    A leaf of two rows ``1e4`` and ``-(1e4 + lambda (1 - 1e-12))`` has a pivot
    of about ``1e-12 lambda`` (a few ulps of the rows either way), far inside
    ``E_u``, which charges the rows' error mass ``sum e = 2e4 + lambda``:
    refused, where a floor on ``|omega| + lambda`` alone (``4 eps (|omega| +
    lambda)``, about 5e-15) would accept a positive one.
    """
    case, w, lambdas = _signed_case("fisher", 1, 1.0, 1.0, 0.0)
    leaf = case.codes[2]
    rows = np.flatnonzero(leaf == 5)[:2]
    others = np.flatnonzero(leaf == 5)[2:]
    w[others] = 0.0
    lam = lambdas["variant"]
    w[rows] = [1e4, -(1e4 + lam * (1.0 - 1e-12))]
    system, penalized = _chain_factor(case, w, lambdas)
    with pytest.raises(np.linalg.LinAlgError, match="'variant' has a tree pivot .* within its"):
        build_augmented_nested_factor(system, penalized)


# ── Levenberg shifts (design §3.11) ───────────────────────────────────────


def test_a_refused_observed_iterate_takes_a_levenberg_shift_and_its_step() -> None:
    """An observed iterate whose ``H`` is not positive definite is shifted, never switched.

    ``_build_iterate_factor`` refuses the Fisher-rows call (propagated), and on
    observed rows takes the smallest shift of the fixed sequence whose factor
    certifies: ``H + E`` with ``E`` the shift times each node's error mass plus
    penalty and each border column's ``|H_jj|``.  The step solves ``(H + E)
    b_new = X'Wz + E b`` (Wood, Pya and Saefken 2016, §3.1.2): it is checked
    against the dense solve of the same shifted system, to the backward-stable
    bound of a Jacobi-scaled Cholesky, ``(p + 10) eps kappa_s(H + E)``.
    Without the ``E b`` term the step is the shifted Hessian's minimizer of a
    different problem and misses the reference by ``||(H + E)^-1 E b||``.
    """
    case, w, lambdas = _signed_case("cancel", 2, 1e-7, 1e4, 0.0)
    rng = np.random.default_rng(11)
    z = rng.normal(size=len(w))
    system = build_nested_structured_system(
        case.matrices,
        case.groups,
        w,
        w * z,
        layout=_layout(case),
        prior_weights=np.abs(case.weights),
    )
    penalized = build_penalized_structured_operator(
        system, case.matrices, case.groups, lambdas, reml_penalties=_components(case)
    )
    with pytest.raises(np.linalg.LinAlgError):
        _build_iterate_factor(system, penalized, observed=False)
    factor, rhs, shift, diagonal = _build_iterate_factor(system, penalized, observed=True)
    assert shift > 0.0 and diagonal is not None
    beta = rng.normal(size=case.dm.p)
    extra = np.concatenate(([0.0], diagonal * beta))
    step = solve_augmented_normal_equations(system, factor, rhs, extra=extra)

    X = np.hstack([np.ones((len(w), 1)), *(matrix.toarray() for matrix in case.matrices)])
    H = X.T @ (w[:, None] * X)
    H[1:, 1:] += _dense_penalty(case, lambdas)
    H[np.arange(1, H.shape[0]), np.arange(1, H.shape[0])] += diagonal
    reference = np.linalg.solve(H, X.T @ (w * z) + extra)
    scale = 1.0 / np.sqrt(np.abs(np.diag(H)))
    kappa = np.linalg.cond(scale[:, None] * H * scale[None, :])
    tolerance = (H.shape[0] + 10) * EPS * kappa
    residual = np.abs(step - reference) / np.maximum(np.abs(reference), 1.0)
    assert np.max(residual) <= tolerance
    unshifted = solve_augmented_normal_equations(system, factor, rhs)
    assert np.max(np.abs(unshifted - reference) / np.maximum(np.abs(reference), 1.0)) > tolerance


def _centred_leaf_case(basis: str):
    """An fs or sz leaf system on signed rows whose border carries a centre ``c0 != 0``."""
    from superglm.solvers.structured import FactorSmoothPenalizedOperator
    from tests._leaf_systems import leaf_system_from_rows
    from tests.test_factor_smooth_leaf_factor import _rows
    from tests.test_sum_to_zero_tree_factor import _case

    if basis == "sz":
        case = _case(signed=True, seed=21, border=lambda rng, n, _: 50.0 + rng.normal(size=(n, 3)))
        return case["system"], case["penalized"], case["X"].mean(axis=0)
    rng = np.random.default_rng(744)
    n_levels, block_size, border = 5, 3, 4
    centre = np.array([5.0, -3.0, 2.0, 1.0])
    levels, Z, X, w, wz = _rows(rng, n_levels=n_levels, block_size=block_size, signed=True)
    system = leaf_system_from_rows(
        Z,
        X + centre,
        levels,
        w,
        wz,
        n_levels=n_levels,
        small_indices=np.arange(border),
        structured_indices=np.arange(border, border + n_levels * block_size).reshape(
            n_levels, block_size
        ),
        center=centre,
        signed=True,
    )
    roots = rng.normal(size=(n_levels, block_size, block_size))
    local = np.einsum("kji,kjl->kil", roots, roots) + 1.5 * np.eye(block_size)
    penalized = FactorSmoothPenalizedOperator.with_penalties(
        system.operator, 0.4 * np.eye(border), local
    )
    return system, penalized, centre


@pytest.mark.parametrize("basis", ["fs", "sz"])
def test_a_shifted_leaf_step_keeps_its_intercept_in_the_centred_coordinate(basis) -> None:
    """A Levenberg-shifted fs or sz step reads entry 0 as ``alpha`` (design §3.8, §3.11).

    ``irls_direct`` solves ``(H + E) [b_0; b] = [1 X]' W z + E beta`` with
    ``centred=True`` and reads entry 0 as the centred intercept ``alpha = b_0 +
    c0' b``.  The data-side solve is centred there, so the ``E beta`` solve
    must be too: in raw coordinates its entry 0 is ``b_0`` and the step's
    intercept is off by ``-c0' b_extra`` (the review measured -3.55 on fs and
    +37.4 on sz).  The slopes agree; the two intercepts differ by a few
    roundings of terms no larger than ``|alpha|`` and ``|c0|' |b|``, so
    ``gamma_{q+4} (|alpha| + 2 |c0|' |b|)`` bounds them.
    """
    from superglm.solvers.irls_direct import _levenberg_shifted_leaf_operator
    from superglm.solvers.structured import build_augmented_structured_factor

    system, penalized, centre = _centred_leaf_case(basis)
    shifted, shift_of = _levenberg_shifted_leaf_operator(penalized, 1e-2, system)
    factor, rhs = build_augmented_structured_factor(system, shifted)
    p = factor.shape[0] - 1
    beta = np.random.default_rng(1).normal(size=p)
    apply = shift_of if callable(shift_of) else (lambda values: shift_of * values)
    extra = np.concatenate(([0.0], apply(beta)))
    raw = solve_augmented_normal_equations(system, factor, rhs, extra=extra)
    centred = solve_augmented_normal_equations(system, factor, rhs, centred=True, extra=extra)
    border = system.operator.small_indices
    q = border.size
    gamma = (q + 4) * EPS / 2 / (1.0 - (q + 4) * EPS / 2)
    scale = abs(centred[0]) + 2.0 * float(np.abs(centre) @ np.abs(raw[1:][border]))
    np.testing.assert_allclose(centred[1:], raw[1:], rtol=0.0, atol=gamma * scale)
    alpha = raw[0] + math.fsum(centre * raw[1:][border])
    assert abs(centred[0] - alpha) <= gamma * scale, (centred[0], alpha, gamma * scale)


def test_an_iterate_no_shift_certifies_raises_its_own_refusal(monkeypatch) -> None:
    """The unshifted refusal names the iterate's cause; the shifted ones do not.

    When no Levenberg shift certifies an observed iterate, the error raised is
    the factor's refusal of ``H`` itself, with the largest shift's refusal
    attached as a note, never the refusal of ``H + 1e8 diag(scale)`` alone.
    """
    import superglm.solvers.irls_direct as irls_direct

    case, w, lambdas = _signed_case("cancel", 2, 1e-7, 1e4, 0.0)
    system, penalized = _chain_factor(case, w, lambdas)
    builds = []

    def refuse(system_, operator):
        builds.append(operator)
        cause = "of the iterate" if operator is penalized else f"shifted, build {len(builds)}"
        raise np.linalg.LinAlgError(f"refusal {cause}")

    monkeypatch.setattr(irls_direct, "build_augmented_structured_factor", refuse)
    with pytest.raises(np.linalg.LinAlgError, match="refusal of the iterate") as raised:
        _build_iterate_factor(system, penalized, observed=True)
    assert len(builds) == 1 + len(irls_direct._LEVENBERG_SHIFTS)
    notes = getattr(raised.value, "__notes__", [])
    assert any(f"shifted, build {len(builds)}" in note for note in notes)
