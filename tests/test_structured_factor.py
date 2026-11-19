"""Structured operator, estimability and factor-protocol algebra tests."""

import importlib.util
import re
import warnings
from collections import Counter
from dataclasses import FrozenInstanceError
from importlib import import_module

import numpy as np
import pandas as pd
import pytest

from superglm.distributions import Poisson
from superglm.reml.gradient import reml_direct_hessian
from superglm.solvers._structured import block_leaves as leaves_module
from superglm.solvers._structured.assembly import (
    build_augmented_structured_factor,
    build_penalized_structured_operator,
    solve_cached_structured,
)
from superglm.solvers._structured.block_leaves import (
    FactorSmoothLeafFactor,
    FactorSmoothPenalizedOperator,
    ProfiledFactorSmoothLeafFactor,
)
from superglm.solvers._structured.operators import (
    BlockSymmetricOperator,
    CenteredBlockOperator,
    LowRankSymmetricOperator,
    SumBlockOperator,
    SymmetricBlockOperator,
    materialize_compact_operator,
)
from superglm.solvers.hessian_factor import (
    DenseHessianFactor,
    DerivativeCrossTraceFactor,
    HessianFactor,
    _component_omega,
)
from superglm.types import PenaltyComponent
from tests._leaf_systems import leaf_system_from_rows


def _spd_blocks():
    rng = np.random.default_rng(519)
    small_indices = np.array([0, 2, 6], dtype=np.intp)
    structured_indices = np.array([1, 3, 4, 5], dtype=np.intp)
    C = rng.normal(scale=0.3, size=(4, 3))
    d = rng.uniform(1.2, 2.0, size=4)
    root = rng.normal(size=(3, 3))
    Q = root.T @ root + np.eye(3)
    A = Q + C.T @ (C / d[:, None])
    H = np.zeros((7, 7))
    H[np.ix_(small_indices, small_indices)] = A
    H[np.ix_(structured_indices, small_indices)] = C
    H[np.ix_(small_indices, structured_indices)] = C.T
    H[structured_indices, structured_indices] = d
    return A, C, d, small_indices, structured_indices, H


@pytest.mark.parametrize("part", ["basis", "core"])
@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_a_non_finite_low_rank_part_is_refused_as_curvature(part, bad) -> None:
    """Both parts and both values, because the two values failed differently.

    ``reml_w_correction`` builds the basis as ``column_stack(dmean_i, dmean_j)``
    with no check of any kind, and the core as ``[[0, -sum_w], [-sum_w, 0]]``.
    Before the guard a NaN core failed ``allclose`` against its own transpose
    and was refused as *asymmetric* -- the right refusal under the wrong name --
    while an inf core passed it (matching infs compare equal) and was accepted
    outright, carrying a non-finite low-rank update into ``_operator_bdlr``. The
    NaN arm alone would pin only the ordering, leaving the silent-acceptance
    case free to come back.
    """
    rng = np.random.default_rng(4)
    basis = rng.normal(size=(7, 2))
    core = np.array([[0.0, -1.5], [-1.5, 0.0]])
    if part == "basis":
        basis[0, 0] = bad
    else:
        core = np.array([[0.0, bad], [bad, 0.0]])

    with pytest.raises(np.linalg.LinAlgError, match="must be finite"):
        LowRankSymmetricOperator(basis=basis, core=core)


def test_hessian_factor_module_exists():
    assert importlib.util.find_spec("superglm.solvers.hessian_factor") is not None
    hessian_factor = import_module("superglm.solvers.hessian_factor")
    inverse = np.eye(3)
    factor = hessian_factor.as_hessian_factor(inverse)
    assert isinstance(factor, hessian_factor.DenseHessianFactor)
    assert isinstance(factor, hessian_factor.HessianFactor)
    assert factor.shape == (3, 3)


def test_dense_hessian_factor_and_protocol_are_available():
    factors = import_module("superglm.solvers.hessian_factor")

    assert hasattr(factors, "HessianFactor")
    assert hasattr(factors, "DenseHessianFactor")


def test_symmetric_block_operator_is_frozen_and_owns_read_only_arrays():
    A, C, d, small_indices, structured_indices, _ = _spd_blocks()
    operator = SymmetricBlockOperator(
        A=A,
        C=C,
        d=d,
        small_indices=small_indices,
        structured_indices=structured_indices,
    )

    with pytest.raises(FrozenInstanceError):
        operator.A = np.eye(3)
    with pytest.raises(ValueError, match="read-only"):
        operator.d[0] = 0.0


def test_dense_hessian_factor_wraps_existing_inverse_contract():
    _, _, _, _, _, H = _spd_blocks()
    inverse = np.linalg.inv(H)
    logdet = np.linalg.slogdet(H)[1]
    factor = DenseHessianFactor(inverse=inverse, log_det=logdet)
    rhs = np.arange(1.0, 8.0)
    selected = np.array([5, 0, 3], dtype=np.intp)

    assert isinstance(factor, HessianFactor)
    np.testing.assert_allclose(factor.solve(rhs), np.linalg.solve(H, rhs))
    np.testing.assert_allclose(
        factor.selected_inverse_block(selected),
        inverse[np.ix_(selected, selected)],
    )
    np.testing.assert_allclose(
        factor.selected_inverse_diagonal(selected),
        np.diag(inverse)[selected],
    )
    np.testing.assert_allclose(factor.logdet(), logdet)
    assert factor.backend == "dense"


def test_dense_penalty_traces_match_materialized_formulas():
    """The dense factor's penalty traces; each structured factor pins its own
    against a dense reference in its module's tests."""
    *_, H = _spd_blocks()
    inverse = np.linalg.inv(H)
    dense_factor = DenseHessianFactor(inverse=inverse, log_det=np.linalg.slogdet(H)[1])
    identity = PenaltyComponent(
        name="broker",
        group_name="broker",
        group_index=1,
        group_sl=slice(3, 7),
        omega_raw=None,
        penalty_kind="identity",
    )
    omega = np.array([[1.5, 0.2], [0.2, 0.8]])
    dense_penalty = PenaltyComponent(
        name="spline",
        group_name="spline",
        group_index=0,
        group_sl=slice(0, 2),
        omega_raw=omega,
        omega_ssp=omega,
    )

    expected_identity_trace = np.trace(inverse[3:7, 3:7])
    expected_dense_trace = np.trace(inverse[0:2, 0:2] @ omega)
    identity_matrix = np.zeros_like(H)
    identity_matrix[3:7, 3:7] = np.eye(4)
    dense_matrix = np.zeros_like(H)
    dense_matrix[0:2, 0:2] = omega
    expected_identity_self = np.trace(inverse @ identity_matrix @ inverse @ identity_matrix)
    expected_cross = np.trace(inverse @ identity_matrix @ inverse @ dense_matrix)

    for factor in (dense_factor,):
        np.testing.assert_allclose(
            factor.trace_inverse_penalty(identity),
            expected_identity_trace,
        )
        np.testing.assert_allclose(
            factor.trace_inverse_penalty(dense_penalty),
            expected_dense_trace,
        )
        np.testing.assert_allclose(
            factor.penalty_cross_trace(identity, identity, 2.0, 3.0),
            6.0 * expected_identity_self,
        )
        np.testing.assert_allclose(
            factor.penalty_cross_trace(identity, dense_penalty, 2.0, 3.0),
            6.0 * expected_cross,
        )


def test_compact_centered_and_low_rank_operator_products_match_dense():
    """Compact operators applied through the dense factor match their materialized forms."""
    *_, H = _spd_blocks()
    inverse = np.linalg.inv(H)
    dense_factor = DenseHessianFactor(
        inverse=inverse,
        log_det=np.linalg.slogdet(H)[1],
    )
    rng = np.random.default_rng(818)
    base = SymmetricBlockOperator(
        A=rng.normal(size=(3, 3)),
        C=rng.normal(size=(4, 3)),
        d=rng.normal(size=4),
        small_indices=np.arange(3, dtype=np.intp),
        structured_indices=np.arange(3, 7, dtype=np.intp),
    )
    base = SymmetricBlockOperator(
        A=0.5 * (base.A + base.A.T),
        C=base.C,
        d=base.d,
        small_indices=base.small_indices,
        structured_indices=base.structured_indices,
    )
    centered = CenteredBlockOperator(
        raw=base,
        cross=rng.normal(size=7),
        total=-0.35,
        center=rng.normal(size=7),
    )
    low_rank = LowRankSymmetricOperator(
        basis=rng.normal(size=(7, 2)),
        core=np.array([[0.3, -0.2], [-0.2, 0.5]]),
    )
    combined = SumBlockOperator((centered, low_rank))
    other = SymmetricBlockOperator(
        A=np.diag([0.4, -0.1, 0.2]),
        C=rng.normal(scale=0.2, size=(4, 3)),
        d=rng.normal(scale=0.3, size=4),
        small_indices=np.arange(3, dtype=np.intp),
        structured_indices=np.arange(3, 7, dtype=np.intp),
    )
    combined_dense = materialize_compact_operator(combined)
    other_dense = materialize_compact_operator(other)
    identity = PenaltyComponent(
        name="broker",
        group_name="broker",
        group_index=1,
        group_sl=slice(3, 7),
        omega_raw=None,
        penalty_kind="identity",
    )
    identity_matrix = np.diag([0.0, 0.0, 0.0, 1.7, 1.7, 1.7, 1.7])

    for factor in (dense_factor,):
        np.testing.assert_allclose(
            factor.trace_inverse_operator(combined),
            np.trace(inverse @ combined_dense),
            atol=2e-12,
        )
        inverse_product = inverse @ combined_dense
        np.testing.assert_allclose(
            factor.inverse_operator_diagonal(combined),
            np.diag(inverse_product),
            atol=2e-12,
        )
        np.testing.assert_allclose(
            factor.inverse_operator_square_diagonal(combined),
            np.diag(inverse_product @ inverse_product),
            atol=2e-12,
        )
        np.testing.assert_allclose(
            factor.operator_cross_trace(combined, other),
            np.trace(inverse @ combined_dense @ inverse @ other_dense),
            atol=2e-12,
        )
        np.testing.assert_allclose(
            factor.penalty_operator_cross_trace(identity, 1.7, combined),
            np.trace(inverse @ identity_matrix @ inverse @ combined_dense),
            atol=2e-12,
        )


def _penalty(name, group_sl, omega=None, **kind):
    return PenaltyComponent(
        name=name,
        group_name=name,
        group_index=0,
        group_sl=group_sl,
        omega_raw=omega,
        omega_ssp=omega,
        **kind,
    )


def _symmetric(rng, size):
    values = rng.normal(size=(size, size))
    return values + values.T


def _leaf_factor(rng, n_levels: int, block_size: int, q: int, profiled: bool):
    """An fs leaf factor from random rows (augmented, or its intercept-profiled view)."""
    per_level = 7
    levels = np.repeat(np.arange(n_levels), per_level)
    n = len(levels)
    system = leaf_system_from_rows(
        rng.normal(size=(n, block_size)),
        rng.normal(size=(n, q)),
        levels,
        rng.uniform(0.5, 2.0, size=n),
        rng.normal(size=n),
        n_levels=n_levels,
        small_indices=np.arange(q),
        structured_indices=np.arange(q, q + n_levels * block_size).reshape(n_levels, block_size),
    )
    local = rng.normal(size=(n_levels, block_size, block_size))
    penalized = FactorSmoothPenalizedOperator.with_penalties(
        system.operator,
        0.3 * np.eye(q),
        np.einsum("kji,kjl->kil", local, local) + 0.5 * np.eye(block_size),
    )
    factor = FactorSmoothLeafFactor(system, penalized)
    if not profiled:
        return factor
    xtw = np.empty(system.operator.shape[0])
    xtw[system.operator.small_indices] = system.xtw_small
    xtw[system.operator.structured_indices] = system.xtw_structured
    return ProfiledFactorSmoothLeafFactor(augmented_factor=factor, sum_w=system.sum_w, xtw=xtw)


def _block_directions(profiled: bool, operator_type=BlockSymmetricOperator):
    """An fs leaf factor, augmented or intercept-profiled, and three REML directions."""
    rng = np.random.default_rng(2610)
    n_levels, block_size, q = 4, 2, 3
    factor = _leaf_factor(rng, n_levels, block_size, q, profiled)
    small = len(factor.small_indices)
    first = int(np.min(factor.structured_indices))

    def operator():
        local = rng.normal(size=(n_levels, block_size, block_size))
        return operator_type(
            A=_symmetric(rng, small),
            C=rng.normal(size=(n_levels, block_size, small)),
            D=local + local.transpose(0, 2, 1),
            small_indices=factor.small_indices,
            structured_indices=factor.structured_indices,
        )

    width = factor.shape[0]
    repeated = _penalty(
        "fs",
        slice(first, width),
        np.diag([1.4, 0.0]),
        penalty_kind="repeated",
        repeat_count=n_levels,
        block_width=block_size,
    )
    centered = CenteredBlockOperator(
        raw=operator(), cross=rng.normal(size=width), total=1.1, center=rng.normal(size=width)
    )
    low_rank = LowRankSymmetricOperator(basis=rng.normal(size=(width, 2)), core=np.eye(2))
    return factor, [
        (repeated, 1.3, centered),
        (
            _penalty("small", slice(0, q), np.diag([0.4, 0.7, 1.0])),
            0.6,
            SumBlockOperator((operator(), low_rank)),
        ),
        (_penalty("ridge", slice(0, 2), penalty_kind="identity"), 2.2, None),
    ]


def _bare_block_directions():
    """An fs leaf factor with no border but the intercept, and a repeated penalty."""
    rng = np.random.default_rng(2611)
    n_levels, block_size = 4, 2
    factor = _leaf_factor(rng, n_levels, block_size, 0, profiled=False)
    width = factor.shape[0]
    local = rng.normal(size=(n_levels, block_size, block_size))
    operator = BlockSymmetricOperator(
        A=np.zeros((1, 1)),
        C=np.zeros((n_levels, block_size, 1)),
        D=local + local.transpose(0, 2, 1),
        small_indices=factor.small_indices,
        structured_indices=factor.structured_indices,
    )
    repeated = _penalty(
        "fs",
        slice(1, width),
        np.diag([1.4, 0.0]),
        penalty_kind="repeated",
        repeat_count=n_levels,
        block_width=block_size,
    )
    return factor, [
        (repeated, 1.3, operator),
        (_penalty("ridge", slice(1, width), penalty_kind="identity"), 0.4, None),
    ]


def _dense_direction(component, scale, operator, width):
    matrix = np.zeros((width, width))
    indices = np.arange(width)[component.group_sl]
    if component.penalty_kind == "identity":
        matrix[indices, indices] = scale
    else:
        matrix[np.ix_(indices, indices)] = scale * _component_omega(component, width)
    return matrix if operator is None else matrix + materialize_compact_operator(operator)


def _cross_trace_tolerance(factor, dense_directions):
    """First-order float64 bound on tr(Z O_i Z O_j) evaluated two ways.

    Every evaluation splits ``Z = Z_local + U R U'`` and accumulates inner
    products no longer than ``p + q``, so each is within
    ``gamma_{p+q} F_i F_j`` of the exact value, where
    ``F = (||Z_local|| + ||U||^2 ||R||) ||O||_F`` bounds the Frobenius norms
    of the split pieces (Higham 2002, eq. 3.13). The dense reference also
    forms ``Z`` and two products: five gammas in all.
    """
    inverse = factor._inverse_bdlr()
    local = np.max(np.linalg.norm(inverse.blocks, 2, axis=(1, 2)))
    low_rank = (
        np.linalg.norm(inverse.basis, 2) ** 2 * np.linalg.norm(inverse.core, 2)
        if inverse.core.size
        else 0.0
    )
    split = local + low_rank
    norms = split * np.array([np.linalg.norm(matrix) for matrix in dense_directions])
    n = factor.shape[0] + inverse.core.shape[0]
    gamma = n * np.finfo(np.float64).eps / (1.0 - n * np.finfo(np.float64).eps)
    return 5.0 * gamma * np.outer(norms, norms)


@pytest.mark.parametrize(
    "build",
    [
        lambda: _block_directions(profiled=False),
        lambda: _block_directions(profiled=True),
        _bare_block_directions,
    ],
    ids=["block", "profiled-block", "block-without-small"],
)
def test_derivative_cross_traces_match_dense_reference(build):
    factor, directions = build()
    width = factor.shape[0]
    inverse = factor.solve(np.eye(width))
    dense = [_dense_direction(*direction, width) for direction in directions]
    expected = np.array(
        [[np.trace(inverse @ left @ inverse @ right) for right in dense] for left in dense]
    )

    traces = factor.derivative_cross_traces(directions)

    assert np.all(np.abs(traces - expected) <= _cross_trace_tolerance(factor, dense))


@pytest.mark.parametrize(
    ("build", "operator_type", "multiply", "with_operators"),
    [
        (_block_directions, BlockSymmetricOperator, "_multiply_symmetric_bdlr", True),
        (_block_directions, BlockSymmetricOperator, "_multiply_symmetric_bdlr", False),
    ],
    ids=["block", "block-penalty-only"],
)
def test_structured_reml_hessian_forms_each_inverse_product_once(
    monkeypatch, build, operator_type, multiply, with_operators
):
    """One ``H^-1 (lambda Omega + dH)`` product per direction per Hessian evaluation.

    A product is counted where it is formed: a direction applying its
    operator to the inverse basis, or a pairwise multiplication by the
    inverse's compact form. The pairwise path formed up to six such products
    for every pair of directions, and on block factors penalty-only pairs
    formed two.
    """
    created, formed = [], []

    class CountedOperator(operator_type):
        def __post_init__(self):
            super().__post_init__()
            created.append(id(self))

        def matvec(self, rhs):
            formed.append(id(self))
            return super().matvec(rhs)

    module = leaves_module
    pairwise_multiply = getattr(module, multiply)

    def counted_multiply(left, right):
        formed.append("pairwise")
        return pairwise_multiply(left, right)

    monkeypatch.setattr(module, multiply, counted_multiply)
    factor, directions = build(profiled=True, operator_type=CountedOperator)
    if not with_operators:
        directions = [(component, scale, None) for component, scale, _ in directions]
        created.clear()
    penalties = [component for component, _, _ in directions]
    lambdas = {component.name: scale for component, scale, _ in directions}
    operators = {i: op for i, (_, _, op) in enumerate(directions) if op is not None}
    common = {"gradient": np.zeros(len(penalties)), "reml_penalties": penalties}

    hessian = reml_direct_hessian(
        [], Poisson(), factor, lambdas, dH_extra=operators or None, **common
    )

    assert Counter(formed) == Counter(created)
    width = factor.shape[0]
    dense_operators = {i: materialize_compact_operator(op) for i, op in operators.items()}
    dense_hessian = reml_direct_hessian(
        [],
        Poisson(),
        factor.solve(np.eye(width)),
        lambdas,
        dH_extra=dense_operators or None,
        **common,
    )
    dense_directions = [_dense_direction(*direction, width) for direction in directions]
    tolerance = 0.5 * _cross_trace_tolerance(factor, dense_directions)
    assert np.all(np.abs(hessian - dense_hessian) <= tolerance)


def _counted(name):
    def method(self, *args):
        self.calls[name] += 1
        return getattr(self._dense, name)(*args)

    return method


class _DelegatingFactor:
    """A factor that is neither dense nor a derivative-cross-trace owner.

    Every operation is a counted call on a dense inverse, so a Hessian built
    on it takes the pairwise compact-trace loop, whatever ``dH_extra`` holds.
    """

    backend = "delegating"

    def __init__(self, dense: DenseHessianFactor):
        self._dense = dense
        self.shape = dense.shape
        self.calls: Counter = Counter()

    solve = _counted("solve")
    logdet = _counted("logdet")
    selected_inverse_block = _counted("selected_inverse_block")
    selected_inverse_diagonal = _counted("selected_inverse_diagonal")
    trace_inverse_penalty = _counted("trace_inverse_penalty")
    penalty_cross_trace = _counted("penalty_cross_trace")
    trace_inverse_operator = _counted("trace_inverse_operator")
    inverse_operator_diagonal = _counted("inverse_operator_diagonal")
    inverse_operator_square_diagonal = _counted("inverse_operator_square_diagonal")
    operator_cross_trace = _counted("operator_cross_trace")
    penalty_operator_cross_trace = _counted("penalty_operator_cross_trace")


@pytest.mark.parametrize(
    "carriers",
    [(0,), (3,), (0, 2), (0, 1, 2, 3)],
    ids=["first", "last", "alternate", "all"],
)
def test_the_pairwise_hessian_traces_the_w_rho_operators_of_a_non_dense_factor(carriers):
    """A factor with neither a dense inverse nor ``derivative_cross_traces`` takes the
    pairwise compact loop, which adds ``tr(Z S_i Z E_j)`` for an operator ``E_j`` on a
    pair's right, ``tr(Z S_j Z E_i)`` for one on its left and ``tr(Z E_i Z E_j)`` for
    both.  The dense factor forms ``H^-1 dH`` instead and the structured factors own
    their cross traces, so a stub delegating to a dense inverse forces the loop.
    Mutation check: dropping the right operator, the left operator or the
    operator-operator term fails the patterns that carry it.

    The dense-inverse route forms the same ``tr(Z O_i Z O_j)``, ``O = S + E``.  Each
    side is a chain of products of inner dimension ``p`` and a sum of ``p**2`` terms
    over the pieces ``S`` and ``E``, within ``8 gamma_{p^2} ||Z||_F^2 (||S_i||_F +
    ||E_i||_F) (||S_j||_F + ||E_j||_F)`` (Higham 2002, eq. 3.13); the Hessian's
    factor -1/2 makes the coefficient 4.
    """
    rng = np.random.default_rng(5521)
    width, m = 8, 4
    root = rng.normal(size=(width, width))
    inverse = np.linalg.inv(root @ root.T + width * np.eye(width))

    def omega(size):
        factor = rng.normal(size=(size, 2))
        return factor @ factor.T

    penalties = [
        _penalty("a", slice(0, 3), omega(3)),
        _penalty("a2", slice(0, 3), omega(3)),
        _penalty("b", slice(3, 6), omega(3)),
        _penalty("ridge", slice(5, 8), penalty_kind="identity"),
    ]
    lambdas = {"a": 1.3, "a2": 0.7, "b": 2.1, "ridge": 0.4}
    operators = {
        i: LowRankSymmetricOperator(basis=rng.normal(size=(width, 2)), core=_symmetric(rng, 2))
        for i in carriers
    }
    common = {"gradient": rng.normal(size=m), "reml_penalties": penalties}

    def stub():
        return _DelegatingFactor(DenseHessianFactor(inverse=inverse, log_det=0.0))

    factor = stub()
    assert isinstance(factor, HessianFactor)
    assert not isinstance(factor, (DenseHessianFactor, DerivativeCrossTraceFactor))
    hessian = reml_direct_hessian([], Poisson(), factor, lambdas, dH_extra=operators, **common)

    pairs = [(i, j) for i in range(m) for j in range(i, m)]
    assert factor.calls["penalty_cross_trace"] == len(pairs)
    assert factor.calls["penalty_operator_cross_trace"] == sum(
        (i in operators) + (j in operators) for i, j in pairs
    )
    assert factor.calls["operator_cross_trace"] == sum(
        i in operators and j in operators for i, j in pairs
    )

    dense_operators = {i: materialize_compact_operator(op) for i, op in operators.items()}
    dense_hessian = reml_direct_hessian(
        [], Poisson(), inverse, lambdas, dH_extra=dense_operators, **common
    )
    norms = np.array(
        [
            np.linalg.norm(_dense_direction(pc, lambdas[pc.name], None, width))
            + (np.linalg.norm(dense_operators[i]) if i in operators else 0.0)
            for i, pc in enumerate(penalties)
        ]
    )
    n = width * width
    gamma = n * np.finfo(np.float64).eps / (1.0 - n * np.finfo(np.float64).eps)
    tolerance = 4.0 * gamma * np.linalg.norm(inverse) ** 2 * np.outer(norms, norms)
    assert np.all(np.abs(hessian - dense_hessian) <= tolerance)

    # The operators move the Hessian far beyond that bound.
    without = reml_direct_hessian([], Poisson(), stub(), lambdas, dH_extra=None, **common)
    assert np.max(np.abs(hessian - without)) > 1e3 * np.max(tolerance)


# ------------------------------------------------ structured geometries under the penalised rule
_U = 2.0**-53


def _structured_geometry(name: str):
    """A Gaussian structured fit whose penalised Hessian has a null space known by construction.

    The ``sz`` term leaves each level's line unpenalised, so a numeric that is a function of the
    level alone (``a[g]``) is an exact alias of the levels' constants, lifted through the
    sum-to-zero contrast onto every level; a numeric pair with one a multiple of the other is an
    alias in the border alone.  Returns the model, the frame, the response, the null dimension
    ``k``, the terms the null space reaches, the coordinates the published gauge holds blank
    and the ``sz`` levels whose unpenalised part it holds.
    """
    from superglm import FactorSmooth, LambdaPolicy, Numeric, RandomEffect, Spline, SuperGLM
    from superglm.terms import interaction, term

    K, per, distinct = {"wide": (129, 12, 3), "multiple": (129, 12, None)}.get(name, (12, 60, None))
    rows = np.random.default_rng(len(name))
    g = np.repeat(np.arange(K), per)
    x = rows.uniform(size=g.size)
    if distinct is not None:
        # every level's rows sit at three values of x: its five basis columns have data rank 3
        x = rows.uniform(size=(K, distinct))[g, rows.integers(0, distinct, g.size)]
    h, a = rows.normal(size=g.size), rows.normal(size=(3, K))
    columns = {"w": rows.normal(size=g.size)}
    blank, held = (), ()
    if name == "wide":
        columns["z"] = a[0][g]
        reached, held = {"z", "x:g:sz"}, (K - 2,)
    elif name in ("schur", "separated", "multiple"):
        scales = {"schur": (1.0, 1.0, 1.0), "separated": (1.0, 1e-5, 1e-10)}
        for i, scale in enumerate(scales.get(name, np.geomspace(8e-6, 2e5, 3))):
            columns[f"z{i}"] = scale * a[i][g]
        reached, held = {"z0", "z1", "z2", "x:g:sz"}, (K - 4, K - 3, K - 2)
    elif name == "offset":
        columns["z"] = 1e9 + h
        columns["z2"] = 2.0 * (columns["z"] - 1e9)  # exact: Sterbenz
        reached, blank = {"z", "z2"}, ("z2",)
    elif name == "mixed":
        columns.update(big=1e8 * rows.normal(size=g.size), z1=h, z2=1e8 * h)
        reached, blank = {"z1", "z2"}, ("z2",)
    elif name == "lifted":
        # ``z - zw = 1e-10 a[g]``: the alias's sz part is 1e-10 of it in raw coordinates and
        # 1e-4 of it in the Jacobi-scaled ones, where the share floor reads it
        columns.update(z=1e-6 * h + 1e-10 * a[0][g], zw=1e-6 * h)
        reached, held = {"z", "zw", "x:g:sz"}, (K - 2,)
    y = (
        np.sin(3 * x)
        + 0.3 * np.sin(5 * x + g % 7)
        + 0.1 * columns["w"]
        + rows.normal(0, 0.3, g.size)
    )
    frame = pd.DataFrame({"x": x, **columns, "g": [f"g{v:03d}" for v in g]})
    if name == "offset":
        model = SuperGLM(
            family="gaussian",
            features={**{c: Numeric() for c in columns}, "g": RandomEffect()},
            selection_penalty=0,
            direct_solve="structured",
        )
    else:
        smooth = FactorSmooth(
            "x", group="g", basis="sz", k=5, lambda_policy={"wiggle": LambdaPolicy.fixed(0.7)}
        )
        spline = Spline(kind="ps", n_knots=6, lambda_policy=LambdaPolicy.fixed(1.0))
        model = SuperGLM(
            family="gaussian",
            terms=[term("x", spline), *(term(c, Numeric()) for c in columns), interaction(smooth)],
            selection_penalty=0,
            direct_solve="structured",
        )
    return model, frame, y, len(blank) + len(held), reached, blank, held


@pytest.mark.parametrize(
    "name", ["wide", "offset", "schur", "separated", "mixed", "multiple", "lifted"]
)
def test_a_structured_fit_publishes_the_dense_oracles_gauge_and_standard_errors(
    name, monkeypatch
) -> None:
    """Each geometry's structured fit against the dense rows' penalised Hessian ``H``.

    ``wide``: 129 levels whose data leave two of each level's five columns free, with one
    alias across all of them.  ``offset``: a numeric at ``1e9`` and a multiple of its centred
    values beside a random effect.  ``schur``: three numerics in the span of the levels'
    constants.  ``separated``: the same at scales ``1``, ``1e-5`` and ``1e-10``.  ``mixed``: an
    alias of a ``1e8``-scale column with an ordinary one, beside an identified ``1e8``-scale
    column.  ``multiple``: three such aliases at scales ``8e-6`` to ``2e5`` past 512 columns.
    ``lifted``: an alias whose ``sz`` part is ``1e-10`` of it in raw coordinates.

    The oracle reads ``H = X_c'X_c + S`` (unit Gaussian weights) from the dense design: its
    Jacobi-scaled eigenvalues have exactly ``k`` under ``n p u`` (the Gram's formation bound),
    and its null space reaches the terms the construction says.  The fit publishes ``k``
    constraints, each on the latest construction column the null space reaches
    (``identifiability.Gauge``): the later numeric's coordinate, blank, or the last ``sz``
    levels' unpenalised part, named in the footnote; ``G'N`` stays within the gauge's
    amplification limit.  Every other standard error is ``sqrt(diag Z (Z'HZ)^-1 Z')``,
    ``Z = null(G')``, to the Cholesky accuracy of the Jacobi-scaled ``Z'HZ`` (Demmel, LAWN 14).
    Deciding again from the retained factor reads the ``k`` truncated directions from the
    rows, never all the columns, and forms no dense design.

    Mutations: every coordinate handed over as undecided (``X_F`` read whole) or the retired
    data-only rule fails every case; the hand-off keeping one truncated direction fails
    ``schur``, ``separated`` and ``multiple``; ``_orthonormal`` without each direction's
    normalisation fails ``separated`` and ``multiple``; reach read in raw coordinates fails
    ``separated``, ``mixed`` and ``lifted``; the row pass centred without the mean's low part
    finds no alias in ``offset``.
    """
    from superglm.model import state_ops
    from superglm.model.fit_state import fitted_lambda2
    from superglm.reml.penalty_algebra import build_penalty_matrix
    from superglm.solvers.identifiability import _AMPLIFICATION_LIMIT, _SHARE_FLOOR
    from tests.test_penalised_identifiability import _redecide, _row_passes

    model, frame, y, k, reached, blank, held = _structured_geometry(name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_reml(frame, y)
        metrics = model.metrics(frame, y)
        se = np.concatenate(
            [np.atleast_1d(np.asarray(metrics.coefficient_se_raw[g.name])) for g in model._groups]
        )
    assert model._reml_profile["direct_backend"] == "structured"
    assert model._fit_inference_info.get("structured_covariance")
    groups = {g.name: g for g in model._groups}

    n, p = model._dm.n, model._dm.p
    X = np.asarray(model._dm.toarray(), dtype=np.float64)
    X = X - X[0]  # exact for a column at 1e9 (Sterbenz), so the mean is taken at the spread
    X = X - X.mean(axis=0)
    S = build_penalty_matrix(
        list(model._dm.group_matrices),
        list(model._groups),
        fitted_lambda2(model),
        p,
        reml_penalties=getattr(model, "_reml_penalties", None),
    )
    H = X.T @ X + S
    jacobi = 1.0 / np.sqrt(np.diag(H))
    values, vectors = np.linalg.eigh(jacobi[:, None] * H * jacobi[None, :])
    cut = n * p * _U
    assert np.all(values[:k] <= cut) and values[k] > cut
    share = np.einsum("ij,ij->i", vectors[:, :k], vectors[:, :k])
    assert {g.name for g in model._groups if np.max(share[g.sl]) > _SHARE_FLOOR**2} == reached
    null = jacobi[:, None] * vectors[:, :k]

    record = model._fit_inference_info["identifiability"]
    assert record.gauge is not None
    G = record.gauge.functionals
    assert G.shape[1] == k and len(record.aliases) == k
    assert np.linalg.cond(G.T @ null) <= _AMPLIFICATION_LIMIT
    units = np.zeros((p, len(blank)))
    units[[groups[term].start for term in blank], np.arange(len(blank))] = 1.0
    np.testing.assert_array_equal(G[:, : len(blank)], units)
    for column, level in zip(G.T[len(blank) :], held, strict=True):
        smooth = groups["x:g:sz"]
        block = np.arange(smooth.start + 5 * level, smooth.start + 5 * level + 5)
        assert set(np.flatnonzero(column)) <= set(block)
    named = [alias.term for alias in record.aliases]
    assert named == [*blank, *(f"x:g:sz[g{level:03d}]'s unpenalised part 1" for level in held)]

    Z = np.linalg.svd(G.T, full_matrices=True)[2][k:].T
    reduced = Z.T @ H @ Z
    covariance = Z @ np.linalg.inv(reduced) @ Z.T
    oracle = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    shown = ~np.any(units, axis=1)
    np.testing.assert_array_equal(~np.isnan(se), shown)
    assert np.all(oracle[~shown] <= np.sqrt(p * _U) * np.max(oracle))
    scale = 1.0 / np.sqrt(np.diag(reduced))
    kappa = np.linalg.cond(scale[:, None] * reduced * scale[None, :])
    np.testing.assert_allclose(se[shown], oracle[shown], rtol=4.0 * np.sqrt(p) * kappa * _U)

    import superglm.group_matrix as group_matrix

    def dense(_self):
        raise AssertionError("the owner reads the truncated directions, never the dense design")

    monkeypatch.setattr(group_matrix.DesignMatrix, "toarray", dense)
    widths = _row_passes(monkeypatch)
    redecided = _redecide(model)
    truncated = state_ops.retained_linear_state(model).profiled_factor.null_directions()
    assert truncated.shape[1] == k and redecided.row_directions == k
    assert widths == [k, k]
    assert [alias.term for alias in redecided.aliases] == named


def _geometry_pairs() -> dict:
    """A real ``(system, penalized operator)`` pair of each structured geometry."""
    from tests.test_signed_rows import _centred_leaf_case, _chain_factor, _signed_case

    case, weights, lambdas = _signed_case("fisher", 1, 1.0, 1.0, 0.0)
    return {
        "nested": _chain_factor(case, weights, lambdas),
        "sz": _centred_leaf_case("sz")[:2],
        "fs": _centred_leaf_case("fs")[:2],
    }


class _NotAStructuredSystem:
    """Matches no geometry of the structured-kind table."""


@pytest.mark.parametrize("dispatch", ["penalize", "factor", "solve", "profile"])
def test_an_unknown_structured_system_is_refused_by_every_dispatch(dispatch) -> None:
    from superglm.solvers._structured.assembly import profile_structured_factor

    refuse = {
        "penalize": lambda system: build_penalized_structured_operator(system, [], [], 1.0),
        "factor": lambda system: build_augmented_structured_factor(system, None),
        "solve": lambda system: solve_cached_structured(system, [], [], 1.0),
        "profile": lambda system: profile_structured_factor(system, None, 1.0, np.ones(1)),
    }[dispatch]
    with pytest.raises(TypeError, match=r"^Unsupported structured system _NotAStructuredSystem\.$"):
        refuse(_NotAStructuredSystem())


@pytest.mark.parametrize(
    ("geometry", "other", "message"),
    [
        ("nested", "fs", "Nested structured systems require a nested penalized operator."),
        ("sz", "nested", "sz leaf systems require the sz penalized operator."),
        ("fs", "sz", "fs leaf systems require the fs penalized operator."),
    ],
)
def test_a_penalized_operator_of_another_geometry_is_refused_by_name(
    geometry, other, message
) -> None:
    pairs = _geometry_pairs()
    with pytest.raises(TypeError, match=f"^{re.escape(message)}$"):
        build_augmented_structured_factor(pairs[geometry][0], pairs[other][1])
