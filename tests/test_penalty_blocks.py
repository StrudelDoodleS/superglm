"""Block-wise log|S_lambda|_+ against dense and exact spectral references.

Wood, Pya and Saefken (2016), sec. 3.1.1: single-penalty blocks add
``rank log(lambda)``; multi-lambda blocks use the Wood (2011) Appendix B
similarity transform; Kronecker-sum tensors have eigenvalues
``lambda_1 a_i + lambda_2 b_j``.  Tolerances are ``rank * width * eps`` times
the conditioning the reference itself resolves.
"""

import copy
import math
import pickle
from decimal import Decimal, localcontext
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp

from superglm.group_matrix import FactorSmoothGroupMatrix
from superglm.model.reml_setup import collect_reml_groups
from superglm.reml import penalty_algebra as algebra
from superglm.reml import penalty_blocks as blocks
from superglm.reml import penalty_support as support_module
from superglm.reml.penalty_algebra import (
    build_penalty_components,
    penalty_component_dense_matrix,
    penalty_logdet,
)
from superglm.types import GroupSlice

EPS = np.finfo(np.float64).eps


def _dense_reference(components, lambdas):
    """log pdet, gradient and Hessian of the assembled penalty from its eigendecomposition."""
    width = components[0].group_sl.stop - components[0].group_sl.start
    matrices = [penalty_component_dense_matrix(item) for item in components]
    total = sum(lambdas[item.name] * matrix for item, matrix in zip(components, matrices))
    values, vectors = np.linalg.eigh(0.5 * (total + total.T))
    keep = values > width * EPS * values[-1] * 1e3
    pinv = (vectors[:, keep] / values[keep]) @ vectors[:, keep].T
    scaled = [pinv @ (lambdas[item.name] * matrix) for item, matrix in zip(components, matrices)]
    gradient = np.array([np.trace(item) for item in scaled])
    hessian = -np.array([[np.sum(a * b.T) for b in scaled] for a in scaled])
    hessian += np.diag(gradient)
    condition = values[keep][-1] / values[keep][0]
    return int(keep.sum()), math.fsum(np.log(values[keep])), gradient, hessian, condition


def _assert_matches(components, lambdas, reference):
    rank, logdet, gradient, hessian, condition = reference
    evaluation = penalty_logdet(lambdas, components)
    tolerance = 64 * rank * len(gradient) * EPS * condition
    assert evaluation.rank == rank
    assert abs(evaluation.logdet - logdet) <= tolerance * max(1.0, abs(logdet))
    names = [item.name for item in components]
    computed = np.array([evaluation.gradient[name] for name in names])
    computed_hessian = np.array([[evaluation.hessian[a, b] for b in names] for a in names])
    np.testing.assert_allclose(computed, gradient, rtol=0, atol=tolerance * rank)
    np.testing.assert_allclose(computed_hessian, hessian, rtol=0, atol=tolerance * rank)
    return evaluation


def _rotated(values, seed):
    width = len(values)
    basis = np.linalg.qr(np.random.default_rng(seed).normal(size=(width, width)))[0]
    matrix = (basis * values) @ basis.T
    return 0.5 * (matrix + matrix.T), basis


def _multi_context(raw, coordinate_map, source=None):
    width = coordinate_map.shape[1]
    group = SimpleNamespace(name="g", sl=slice(0, width), size=width)
    matrix = SimpleNamespace(
        R_inv=coordinate_map,
        omega=sum(raw),
        omega_components=[(f"c{index}", item) for index, item in enumerate(raw)],
    )
    kwargs = {} if source is None else {"_reuse_raw_from": source}
    return build_penalty_components([matrix], [(0, group)], **kwargs)


def _single_context(omega, coordinate_map, source=None):
    width = coordinate_map.shape[1]
    group = SimpleNamespace(name="s", sl=slice(0, width), size=width)
    matrix = SimpleNamespace(R_inv=coordinate_map, omega=omega)
    kwargs = {} if source is None else {"_reuse_raw_from": source}
    return build_penalty_components([matrix], [(0, group)], **kwargs)


def _second_differences(width):
    D = np.diff(np.eye(width), n=2, axis=0)
    return D.T @ D


def _well_conditioned_map(width, seed):
    rng = np.random.default_rng(seed)
    return np.linalg.qr(rng.normal(size=(width, width)))[0] @ np.diag(rng.uniform(0.5, 2.0, width))


@pytest.mark.parametrize("weight", [1e-3, 1.0, 40.0])
def test_single_dense_block_matches_the_dense_reference(weight):
    omega = _second_differences(9)
    components = _single_context(omega, _well_conditioned_map(9, 3))
    assert algebra._context_geometry(components).coordinate_map is not None
    lambdas = {"s": weight}
    evaluation = _assert_matches(components, lambdas, _dense_reference(components, lambdas))
    assert evaluation.gradient == {"s": 7.0}
    assert evaluation.hessian == {("s", "s"): 0.0}


@pytest.mark.parametrize("weights", [(1.0, 1.0), (30.0, 0.2), (1e-3, 5.0)])
def test_multi_penalty_block_matches_the_dense_reference(weights):
    first, _ = _rotated(np.r_[np.geomspace(1, 50, 6), np.zeros(3)], 1)
    second, _ = _rotated(np.r_[np.zeros(2), np.geomspace(0.5, 4, 7)], 2)
    components = _multi_context([first, second], _well_conditioned_map(9, 4))
    lambdas = {"g:c0": weights[0], "g:c1": weights[1]}
    _assert_matches(components, lambdas, _dense_reference(components, lambdas))


def _kronecker_pair(p1=5, p2=4):
    """A te()-style pair: second differences on one margin, first differences on the other."""
    left = _second_differences(p1)
    first_differences = np.diff(np.eye(p2), axis=0)
    right = first_differences.T @ first_differences
    return np.kron(left, np.eye(p2)), np.kron(np.eye(p1), right), left, right


@pytest.mark.parametrize("weights", [(1.0, 1.0), (25.0, 0.04), (0.01, 3.0)])
def test_tensor_closed_form_matches_the_dense_reference(weights):
    first, second, _, _ = _kronecker_pair()
    components = _multi_context([first, second], _well_conditioned_map(20, 5))
    lambdas = {"g:c0": weights[0], "g:c1": weights[1]}
    evaluation = _assert_matches(components, lambdas, _dense_reference(components, lambdas))
    geometry = algebra._context_geometry(components)
    assert geometry.plain.kronecker is not None
    # The closed form and the Appendix B route evaluate the same block.
    transformed = algebra._PlainBlock(
        geometry.plain.rank, roots=geometry.plain.roots, volume=geometry.plain.volume
    )
    rank, logdet, gradient, hessian = algebra._plain_block_evaluate(
        transformed, geometry, geometry.matrices, np.array(weights)
    )
    tolerance = 64 * rank * 20 * EPS * _dense_reference(components, lambdas)[4]
    assert rank == evaluation.rank
    assert abs(logdet - evaluation.logdet) <= tolerance * abs(logdet)
    np.testing.assert_allclose(gradient, list(evaluation.gradient.values()), atol=tolerance)
    assert abs(hessian[0, 1] - evaluation.hessian["g:c0", "g:c1"]) <= tolerance


def _commuting_pair(width=12, seed=7):
    """S1 = V diag(a) V', S2 = V diag(b) V': S_lambda has eigenvalues lambda_1 a + lambda_2 b."""
    a = np.r_[np.geomspace(1.0, 1e3, 7), np.zeros(width - 7)]
    b = np.r_[np.zeros(2), np.geomspace(0.5, 4.0, width - 2)]
    first, basis = _rotated(a, seed)
    second = (basis * b) @ basis.T
    return first, 0.5 * (second + second.T), a, b


@pytest.mark.parametrize("ratio", [1e-12, 1e-6, 1e6, 1e12])
def test_lambda_ratio_does_not_leak_the_dominant_rounding(ratio):
    """Appendix B keeps lambda_1 S_1's rounding out of the eigenvalues lambda_2 S_2 sets.

    The reference is the exact spectrum ``lambda_1 a_i + lambda_2 b_i``.  The
    transform's error is set by each component's own conditioning, whatever
    the ratio; an eigendecomposition of the assembled sum is not, and the
    fixture is chosen so that it misses by far more than the tolerance.
    """
    first, second, a, b = _commuting_pair()
    support = support_module._penalty_support([first, second])
    roots = algebra._support_roots(support)
    weights = np.array([ratio, 1.0])
    cells = weights[0] * a + weights[1] * b
    expected = math.fsum(np.log(cells))
    fraction = weights[0] * a / cells
    expected_gradient = np.array([fraction.sum(), (1 - fraction).sum()])
    expected_cross = float(np.sum(fraction * (1 - fraction)))
    logdet, gradient, hessian = blocks.multi_penalty_logdet(roots, weights)
    condition = max(a.max() / a[a > 0].min(), b.max() / b[b > 0].min())
    tolerance = 8 * len(a) * EPS * condition
    assert abs(logdet - expected) <= tolerance
    np.testing.assert_allclose(gradient, expected_gradient, rtol=0, atol=tolerance)
    np.testing.assert_allclose(
        hessian, expected_cross * np.array([[1, -1], [-1, 1]]), rtol=0, atol=tolerance
    )
    if abs(math.log10(ratio)) >= 12:
        assembled = sum(w * root.T @ root for w, root in zip(weights, roots))
        leaked = math.fsum(np.log(np.linalg.eigvalsh(assembled)))
        assert abs(leaked - expected) > 1e4 * tolerance


@pytest.mark.parametrize("ratio", [1e-12, 1e12])
def test_tensor_closed_form_holds_at_extreme_weight_ratios(ratio):
    first, second, left, right = _kronecker_pair()
    coordinate_map = _well_conditioned_map(20, 6)
    components = _multi_context([first, second], coordinate_map)
    geometry = algebra._context_geometry(components)
    a, u = np.linalg.eigh(left)
    b, v = np.linalg.eigh(right)
    a[np.abs(a) < 1e-12], b[np.abs(b) < 1e-12] = 0.0, 0.0
    cells = ratio * a[:, None] + b[None, :]
    active = cells > 0
    range_basis = np.kron(u, v)[:, active.ravel()]
    mapped = coordinate_map.T @ range_basis
    expected = math.fsum(np.log(cells[active])) + np.linalg.slogdet(mapped.T @ mapped)[1]
    evaluation = penalty_logdet({"g:c0": ratio, "g:c1": 1.0}, components)
    assert geometry.plain.kronecker is not None
    assert evaluation.rank == int(active.sum()) == geometry.support.rank
    assert abs(evaluation.logdet - expected) <= 64 * 20**2 * EPS * 1e3 * abs(expected)


def test_a_kronecker_split_is_structural_not_approximate():
    first, second, _, _ = _kronecker_pair()
    assert blocks.kronecker_split(first, second)[:2] == (5, 4)
    assert blocks.kronecker_split(second, first)[4] is True
    perturbed = first.copy()
    perturbed[0, 5] += 1e-15
    perturbed[5, 0] += 1e-15
    assert blocks.kronecker_split(perturbed, second) is None


def _factor_smooth(factor_basis, components):
    n_levels, block_size = 4, 6
    basis = sp.csr_matrix(np.eye(block_size)[np.arange(24) % block_size])
    codes = np.arange(24, dtype=np.intp) % n_levels
    gm = FactorSmoothGroupMatrix(
        basis,
        codes,
        n_levels,
        natural_map=np.eye(block_size),
        levels=("a", "b", "c", "d"),
        repeated_penalty_components=components,
        factor_basis=factor_basis,
    )
    width = (n_levels - (factor_basis == "sz")) * block_size
    group = GroupSlice(name=f"x:group:{factor_basis}", start=0, end=width)
    return build_penalty_components([gm], collect_reml_groups([group], [gm]))


@pytest.mark.parametrize("weight", [0.05, 7.0])
def test_sum_to_zero_block_matches_the_dense_reference(weight):
    wiggle = np.diag([2.0, 0.8, 0.3, 0.1, 0.0, 0.0])
    components = _factor_smooth("sz", (("wiggle", wiggle),))
    lambdas = {components[0].name: weight}
    _assert_matches(components, lambdas, _dense_reference(components, lambdas))


@pytest.mark.parametrize("weights", [(1.0, 1.0), (40.0, 0.01)])
def test_repeated_block_matches_the_dense_reference(weights):
    wiggle = np.diag([2.0, 0.8, 0.3, 0.1, 0.0, 0.0])
    null = np.diag([0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
    components = _factor_smooth("fs", (("wiggle", wiggle), ("null", null)))
    assert {item.penalty_kind for item in components} == {"repeated"}
    lambdas = dict(zip([item.name for item in components], weights))
    evaluation = _assert_matches(components, lambdas, _dense_reference(components, lambdas))
    # Complementary ranges: exact row counts and no curvature.
    assert sorted(evaluation.gradient.values()) == [8.0, 16.0]
    assert set(evaluation.hessian.values()) == {0.0}


def test_zero_weight_face_uses_the_active_support():
    first, _ = _rotated(np.r_[np.geomspace(1, 50, 6), np.zeros(3)], 1)
    second, _ = _rotated(np.r_[np.zeros(2), np.geomspace(0.5, 4, 7)], 2)
    components = _multi_context([first, second], _well_conditioned_map(9, 4))
    lambdas = {"g:c0": 2.0, "g:c1": 0.0}
    evaluation = penalty_logdet(lambdas, components)
    reference = _dense_reference(components[:1], lambdas)
    assert evaluation.rank == reference[0] == 6
    assert abs(evaluation.logdet - reference[1]) <= 64 * 6 * 9 * EPS * reference[4]
    assert evaluation.gradient == {"g:c0": 6.0, "g:c1": 0.0}


def test_stored_penalty_without_the_raw_rank_keeps_the_solver_space_support():
    # C' C = [[1, 1], [1, 1 + 1e-16]] rounds to rank one in binary64.
    components = _single_context(np.eye(2), np.array([[1.0, 1.0], [0.0, 1e-8]]))
    geometry = algebra._context_geometry(components)
    assert geometry.coordinate_map is None and geometry.raw_refused
    solver = support_module._penalty_support([components[0].omega_ssp])
    assert penalty_logdet({"s": 1.0}, components).rank == solver.rank


def test_admitted_rounded_map_stays_within_weyl_of_the_stored_penalty():
    """S = [[1, 1], [1, fl(1 + 1e-6)]]: the raw route reports ln det(C' C) = ln 1e-6.

    Weyl puts ln det S within ``r x / (1 - x)``, ``x = ||S - C'C||_F / lambda_min(C'C)``,
    of it, about 8e-11 here; exact rational arithmetic gives both sides.
    """
    coordinate_map = np.array([[1.0, 1.0], [0.0, 1e-3]])
    components = _single_context(np.eye(2), coordinate_map)
    assert algebra._context_geometry(components).coordinate_map is not None
    evaluation = penalty_logdet({"s": 1.0}, components)
    assert evaluation.rank == 2
    with localcontext() as context:
        context.prec = 60
        c = [[Fraction(float(value)) for value in row] for row in coordinate_map]
        exact = [[sum(c[k][i] * c[k][j] for k in range(2)) for j in range(2)] for i in range(2)]
        stored = [[Fraction(float(value)) for value in row] for row in components[0].omega_ssp]
        difference = sum((stored[i][j] - exact[i][j]) ** 2 for i in range(2) for j in range(2))
        half_trace = _decimal(exact[0][0] + exact[1][1]) / 2
        smallest = half_trace - (half_trace**2 - _decimal(_det(exact))).sqrt()
        x = _decimal(difference).sqrt() / smallest
        bridge = 2 * x / (1 - x)
        truth = _decimal(_det(stored)).ln()
    rounding = 16 * EPS * np.linalg.cond(coordinate_map)
    assert abs(Decimal(evaluation.logdet) - truth) <= bridge + Decimal(rounding)
    # The raw value is ln det(C' C) itself, not the stored penalty's determinant.
    assert abs(evaluation.logdet - math.log(1e-6)) <= rounding


def _det(matrix):
    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]


def _decimal(value):
    return Decimal(value.numerator) / Decimal(value.denominator)


def test_later_contexts_reuse_the_raw_support_and_a_refusal(monkeypatch):
    counts = {"support": 0}
    original = support_module._penalty_support

    def counted(*args, **kwargs):
        counts["support"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(support_module, "_penalty_support", counted)
    omega = _second_differences(7)
    coordinate_map = np.random.default_rng(5).normal(size=(7, 6))
    first = _single_context(omega, coordinate_map)
    later = _single_context(omega, 2 * coordinate_map, source=first)
    assert counts["support"] == 1
    assert algebra._context_geometry(later).support is algebra._context_geometry(first).support
    expected = penalty_logdet({"s": 3.0}, first).logdet + 2 * 5 * math.log(2.0)
    assert abs(penalty_logdet({"s": 3.0}, later).logdet - expected) <= 64 * 5 * EPS * 1e3
    refused = _single_context(np.eye(2), np.array([[1.0, 1.0], [0.0, 1e-8]]))
    counts["support"] = 0
    carried = _single_context(np.eye(2), np.array([[1.0, 1.0], [0.0, 1e-3]]), source=refused)
    assert counts["support"] == 0
    assert algebra._context_geometry(carried).raw_refused


@pytest.mark.parametrize("method", ["pickle protocol 4", "pickle protocol 5", "deepcopy"])
def test_a_reloaded_plain_context_evaluates_the_same(method):
    first, second, _, _ = _kronecker_pair()
    contexts = [
        _single_context(_second_differences(7), np.random.default_rng(5).normal(size=(7, 6))),
        _multi_context([first, second], _well_conditioned_map(20, 5)),
    ]
    for components in contexts:
        lambdas = {item.name: 1.7 for item in components}
        expected = penalty_logdet(lambdas, components)
        if method == "deepcopy":
            copied = copy.deepcopy(components)
        else:
            copied = pickle.loads(pickle.dumps(components, protocol=int(method[-1])))
        geometry = algebra._context_geometry(copied)
        assert geometry is not None and geometry.plain is None
        assert penalty_logdet(lambdas, copied) == expected


def test_a_certified_context_evaluates_under_the_plain_evaluator():
    """A model saved before the split carries certified contexts; the plain evaluator reads them."""
    first, second, _, _ = _kronecker_pair()
    coordinate_map = _well_conditioned_map(20, 5)
    group = SimpleNamespace(name="g", sl=slice(0, 20), size=20)
    matrix = SimpleNamespace(
        R_inv=coordinate_map,
        omega=first + second,
        omega_components=[("c0", first), ("c1", second)],
    )
    certified = build_penalty_components([matrix], [(0, group)], certified=True)
    plain = build_penalty_components([matrix], [(0, group)])
    assert algebra._context_geometry(certified).certified
    lambdas = {"g:c0": 3.0, "g:c1": 0.2}
    old, new = penalty_logdet(lambdas, certified), penalty_logdet(lambdas, plain)
    assert old.rank == new.rank
    assert abs(old.logdet - new.logdet) <= 64 * 20**2 * EPS * 1e3 * abs(new.logdet)
    reference = algebra._compute_penalty_logdet_evaluation(lambdas, certified)
    assert abs(reference.logdet - new.logdet) <= reference.logdet_error + 64 * 20**2 * EPS * 1e3
