"""Block-wise log|S_lambda|_+ against dense and exact spectral references.

Wood, Pya and Saefken (2016), sec. 3.1.1: single-penalty blocks add
``rank log(lambda)``; multi-lambda blocks use the Wood (2011) Appendix B
similarity transform; Kronecker-sum tensors have eigenvalues
``lambda_1 a_i + lambda_2 b_j``.  A dense eigendecomposition moves every
eigenvalue by at most about ``width * eps * lambda_max`` (Weyl), so it resolves
``log|S|_+`` and its derivatives to ``rank * width * eps * condition``; the
tolerances below are twice that, for the reference and the evaluator.  The
exact references carry 60 decimal digits.
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

from superglm.features._spline_penalties import build_integrated_derivative_penalty
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
    # Computed null eigenvalues lie within width * eps * lambda_max of zero;
    # every fixture's positive spectrum is above sqrt(width * eps) of the largest.
    keep = values > math.sqrt(width * EPS) * values[-1]
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
    tolerance = _dense_tolerance(components, rank, condition)
    assert evaluation.rank == rank
    assert abs(evaluation.logdet - logdet) <= tolerance
    names = [item.name for item in components]
    computed = np.array([evaluation.gradient[name] for name in names])
    computed_hessian = np.array([[evaluation.hessian[a, b] for b in names] for a in names])
    np.testing.assert_allclose(computed, gradient, rtol=0, atol=tolerance)
    # Each Hessian entry is a trace of two products with S^+.
    np.testing.assert_allclose(computed_hessian, hessian, rtol=0, atol=2 * tolerance)
    return evaluation


def _dense_tolerance(components, rank, condition):
    width = components[0].group_sl.stop - components[0].group_sl.start
    return 2 * rank * width * EPS * condition


def _exact_reference(roots, weights, digits=60):
    """``log det``, gradient and Hessian of ``sum_j w_j R_j' R_j`` from the given roots.

    Decimal conversion of a binary64 value is exact, and ``digits``-digit
    arithmetic rounds each operation at ``10**-digits`` relative, so for
    condition numbers below ``10**(digits - 25)`` this is the exact value for
    these roots, far below binary64's resolution.  Also returns ``kappa = 2 sum_j
    ||W_j||_F ||W_j S^-1||_F`` for ``W_j = sqrt(w_j) R_j``: ``d log det = 2 sum_j
    tr(S^-1 W_j' dW_j)``, so a perturbation of each root by ``delta ||W_j||_F``
    moves ``log det`` by at most ``delta kappa`` to first order.
    """
    with localcontext() as context:
        context.prec = digits
        width = roots[0].shape[1]
        terms = []
        for root, weight in zip(roots, weights, strict=True):
            rows = [[Decimal(float(value)) for value in row] for row in np.asarray(root)]
            scale = Decimal(float(weight))
            terms.append(
                [
                    [
                        scale * sum((row[i] * row[j] for row in rows), Decimal(0))
                        for j in range(width)
                    ]
                    for i in range(width)
                ]
            )
        total = [[sum(term[i][j] for term in terms) for j in range(width)] for i in range(width)]
        work = [
            row[:] + [Decimal(int(i == j)) for j in range(width)] for i, row in enumerate(total)
        ]
        logdet = Decimal(0)
        for column in range(width):
            pivot = work[column][column]
            assert pivot > 0
            logdet += pivot.ln()
            work[column] = [value / pivot for value in work[column]]
            for row in range(width):
                if row != column and work[row][column]:
                    factor = work[row][column]
                    work[row] = [
                        a - factor * b for a, b in zip(work[row], work[column], strict=True)
                    ]
        inverse = [row[width:] for row in work]
        products = [
            [
                [sum(inverse[i][k] * term[k][j] for k in range(width)) for j in range(width)]
                for i in range(width)
            ]
            for term in terms
        ]
        gradient = [sum(product[i][i] for i in range(width)) for product in products]
        hessian = [
            [-sum(a[i][k] * b[k][i] for i in range(width) for k in range(width)) for b in products]
            for a in products
        ]
        for index, value in enumerate(gradient):
            hessian[index][index] += value
        kappa = Decimal(0)
        for root, weight in zip(roots, weights, strict=True):
            rows = [[Decimal(float(value)) for value in row] for row in np.asarray(root)]
            scale = Decimal(float(weight)).sqrt()
            acted = [
                [sum(row[k] * inverse[k][j] for k in range(width)) for j in range(width)]
                for row in rows
            ]
            norm = sum(value * value for row in rows for value in row).sqrt()
            kappa += 2 * scale * norm * scale * sum(v * v for row in acted for v in row).sqrt()
        return (
            float(logdet),
            np.array(gradient, dtype=float),
            np.array(hessian, dtype=float),
            float(kappa),
        )


def _assert_exact(computed, reference, width):
    """Within ``width eps (kappa + |log det|)``: a normwise backward-stable evaluation's error.

    The transform and the QR act on the stacked roots by orthogonal steps, so
    the computed value is exact for roots moved by about ``width eps`` of
    their norms (Higham 2002, Theorems 19.4 and 19.10), which ``kappa``
    converts to ``log det``; the derivatives are traces of the same factors.
    The sum of ``width`` rounded logarithms adds ``width eps |log det|``.
    """
    logdet, gradient, hessian = computed
    tolerance = width * EPS * reference[3]
    assert abs(logdet - reference[0]) <= tolerance + width * EPS * abs(reference[0])
    np.testing.assert_allclose(gradient, reference[1], rtol=0, atol=tolerance)
    np.testing.assert_allclose(hessian, reference[2], rtol=0, atol=2 * tolerance)


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


@pytest.mark.parametrize("swapped", [False, True])
@pytest.mark.parametrize("weights", [(1.0, 1.0), (25.0, 0.04), (0.01, 3.0)])
def test_tensor_closed_form_matches_the_dense_reference(weights, swapped):
    first, second, _, _ = _kronecker_pair()
    pair = [second, first] if swapped else [first, second]
    components = _multi_context(pair, _well_conditioned_map(20, 5))
    lambdas = {"g:c0": weights[0], "g:c1": weights[1]}
    evaluation = _assert_matches(components, lambdas, _dense_reference(components, lambdas))
    geometry = algebra._context_geometry(components)
    assert geometry.plain.kronecker is not None
    assert geometry.plain.kronecker[2] is swapped
    # The closed form and the Appendix B route evaluate the same block.
    transformed = algebra._PlainBlock(
        geometry.plain.rank, roots=geometry.plain.roots, volume=geometry.plain.volume
    )
    rank, logdet, gradient, hessian = algebra._plain_block_evaluate(
        transformed, geometry, geometry.matrices, np.array(weights)
    )
    tolerance = _dense_tolerance(components, rank, _dense_reference(components, lambdas)[4])
    assert rank == evaluation.rank
    assert abs(logdet - evaluation.logdet) <= tolerance
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


_COLUMN_SCALED = {
    "dominant spectrum below the root cut": ([np.diag([1.0, 1e-16]), np.eye(2)], (1e10, 1e-6)),
    "below the cut after the support's scaling": ([np.diag([1.0, 5e-17]), np.eye(2)], (1e10, 5e-7)),
    "rank-deficient dominant": ([np.diag([1.0, 1e-16, 0.0]), np.eye(3)], (1e12, 1.0)),
    "direction only the dominant spans": (
        [np.diag([1.0, 1e-20, 0.0]), np.diag([1.0, 0.0, 1.0])],
        (1e12, 1.0),
    ),
    "tiny trailing column": ([np.diag([1.0, 1.0, 1e-20]), np.eye(3)], (1.0, 1e-20)),
    "trailing column only the dominant spans": (
        [np.diag([1.0, 1.0, 1e-20]), np.diag([1.0, 0.0, 0.0])],
        (1.0, 1e-7),
    ),
}


@pytest.mark.parametrize("case", list(_COLUMN_SCALED))
def test_the_transform_keeps_what_column_scaled_components_set(case):
    """The support's column scaling keeps directions a fixed relative cut of one root drops.

    A dominant penalty's eigenvalues far below eps of its largest are exact
    data here, and its weight lifts them to the smaller terms' level, so
    discarding them changes ``log|S|`` by O(1) or loses a support direction.
    """
    matrices, weights = _COLUMN_SCALED[case]
    support = support_module._penalty_support(matrices)
    assert support.rank == len(matrices[0])
    roots = algebra._support_roots(support)
    reference = _exact_reference(roots, weights)
    _assert_exact(blocks.multi_penalty_logdet(roots, np.array(weights)), reference, support.rank)


def test_column_scaled_components_through_the_context():
    """``diag(1, 1e-16)`` and ``I`` at ``(1e10, 1e-6)``: ``log(2e4)``, gradient ``(3/2, 1/2)``."""
    components = _multi_context([np.diag([1.0, 1e-16]), np.eye(2)], np.eye(2))
    weights = (1e10, 1e-6)
    evaluation = penalty_logdet(dict(zip(("g:c0", "g:c1"), weights)), components)
    block = algebra._context_geometry(components).plain
    reference = _exact_reference(block.roots, weights)
    exact = math.log(1e10 + 1e-6) + math.log(1e-6 + 1e-6)
    gradient = np.array(list(evaluation.gradient.values()))
    hessian = np.array(list(evaluation.hessian.values())).reshape(2, 2)
    assert evaluation.rank == 2
    _assert_exact((evaluation.logdet - block.volume, gradient, hessian), reference, 2)
    tolerance = 2 * EPS * reference[3]
    assert abs(evaluation.logdet - exact) <= tolerance + 4 * EPS * abs(exact)
    np.testing.assert_allclose(gradient, [1.5, 0.5], rtol=0, atol=tolerance)


@pytest.mark.parametrize("ratio", [1e-12, 1e12])
def test_non_commuting_penalties_at_extreme_weight_ratios(ratio):
    """A column-scaled dominant penalty against a rotated one, at lambda ratios 1e+-12.

    At ``1e12`` the dominant term's ``1e-16`` direction contributes ``1e-4``
    of the rotated term's eigenvalues there, so it is part of the answer.
    """
    first = np.diag([1.0, 1.0, 1e-16, 0.0, 0.0])
    second, _ = _rotated(np.r_[0.0, np.geomspace(0.5, 4.0, 4)], 3)
    support = support_module._penalty_support([first, second])
    assert support.rank == 5
    roots = algebra._support_roots(support)
    weights = (ratio, 1.0)
    reference = _exact_reference(roots, weights)
    _assert_exact(blocks.multi_penalty_logdet(roots, np.array(weights)), reference, 5)


def test_the_dominance_order_survives_grams_beyond_binary64():
    """``1e155 I`` against a rank-two ``1e163`` root in three dimensions: both Grams overflow.

    Householder QR without row sorting or pivoting is not row-wise stable when
    row norms vary widely (Powell and Reid 1969; Cox and Higham 1998, Pitman
    Res. Notes Math. 380), and here the large term is stacked last.  Ranked by
    overflowed norms the two terms compare equal and stay unrotated, so the
    QR's rounding of ``u 1e163`` reaches the large term's null direction, where
    the small one sets ``1e155``: an error of order ``u 1e8``.  Ranked by their
    logarithms the large term dominates and is rotated out.
    """
    rows = np.linalg.qr(np.random.default_rng(8).normal(size=(3, 3)))[0][:2]
    roots = [1e150 * np.eye(3), 1e150 * rows]
    weights = (1e10, 1e26)
    reference = _exact_reference(roots, weights)
    _assert_exact(blocks.multi_penalty_logdet(roots, np.array(weights)), reference, 3)


@pytest.mark.parametrize("small_first", [True, False])
def test_the_stacked_root_factor_does_not_depend_on_the_component_order(small_first):
    """``1e-64 I`` and the rank-two ``K``, so ``log det(K + 1e-64 I) = log 6 - 64 log 10``.

    Stacked in component order, the small term's ``1e-32`` rows came first and
    the QR took one as a pivot row, losing it in ``K``'s rows: ``-145.428`` and
    gradient ``0.865`` against ``-145.574`` and ``1``.  The condition number is
    about ``1e64``, so the reference carries 160 digits.
    """
    small = 1e-64 * np.eye(3)
    rank_two = np.array([[2.0, 0.0, 1.0], [0.0, 2.0, -1.0], [1.0, -1.0, 1.0]])
    raw = [small, rank_two] if small_first else [rank_two, small]
    block = algebra._context_geometry(_multi_context(raw, np.eye(3))).plain
    weights = (1.0, 1.0)
    reference = _exact_reference(block.roots, weights, digits=160)
    computed = blocks.multi_penalty_logdet(block.roots, np.array(weights))
    _assert_exact(computed, reference, block.rank)


def _skewed_tensor(seed=3):
    """A cubic B-spline curvature margin on lognormal(0, 3) quantile knots, by differences.

    The integrated penalty scales with the knot intervals, so on these knots
    its support root's singular values span about 4e16: the support keeps
    directions that a fixed relative cut on that root drops.
    """
    sample = np.random.default_rng(seed).lognormal(0.0, 3.0, 2000)
    inner = np.quantile(sample, np.linspace(0.0, 1.0, 4))
    knots = np.r_[[inner[0]] * 3, inner, [inner[-1]] * 3]
    margin = build_integrated_derivative_penalty(knots, 3, 2)
    differences = np.diff(np.eye(3), axis=0)
    return np.kron(margin, np.eye(3)), np.kron(np.eye(len(margin)), differences.T @ differences)


@pytest.mark.parametrize("weights", [(1.0, 1.0), (1e6, 1e-6), (1e12, 1.0), (1e-12, 1.0)])
def test_skewed_integrated_penalty_margins_through_the_transform(weights):
    support = support_module._penalty_support(list(_skewed_tensor()))
    roots = algebra._support_roots(support)
    reference = _exact_reference(roots, weights)
    _assert_exact(blocks.multi_penalty_logdet(roots, np.array(weights)), reference, support.rank)


@pytest.mark.parametrize("weights", [(1.0, 1.0), (1e6, 1e-6)])
def test_a_margin_eigenvalue_below_the_eigensolver_floor_leaves_the_block_to_its_roots(
    monkeypatch, weights
):
    """On seed 14 the support keeps four of the margin's six directions, and
    ``eigvalsh`` puts the third and fourth near 2, below its floor ``p eps
    ||A||_2`` of about 16: their values are rounding.  The margins' eigenvalues
    are forced nonnegative here, so the closed form is on offer whatever signs
    ``eigvalsh`` gives that tail.  The block is evaluated from its support
    roots, exactly; the unguarded closed form gives 207.178 against 206.325 at
    ``(1, 1)``.
    """
    eigvalsh, margins = np.linalg.eigvalsh, algebra._kronecker_margins

    def nonnegative_margins(support, raw):
        with monkeypatch.context() as patch:
            patch.setattr(np.linalg, "eigvalsh", lambda a: np.sort(np.abs(eigvalsh(a))))
            return margins(support, raw)

    monkeypatch.setattr(algebra, "_kronecker_margins", nonnegative_margins)
    components = _multi_context(list(_skewed_tensor(14)), np.eye(18))
    block = algebra._context_geometry(components).plain
    assert block.kronecker is None
    lambdas = {"g:c0": weights[0], "g:c1": weights[1]}
    evaluation = penalty_logdet(lambdas, components)
    computed = (
        evaluation.logdet - block.volume,
        np.array([evaluation.gradient[name] for name in lambdas]),
        np.array([[evaluation.hessian[a, b] for b in lambdas] for a in lambdas]),
    )
    _assert_exact(computed, _exact_reference(block.roots, weights), block.rank)


def test_a_finite_solver_penalty_keeps_its_dominant_term_when_the_raw_gram_overflows():
    """Raw ``1e300 I`` and ``I`` under the map ``1e-150 I``: stored as ``I`` and ``1e-300 I``.

    The dominant weighted raw root is ``1e155 I``, whose Gram norm is beyond
    binary64; at ``(1e10, 1)`` the answer is ``2 log(1e10 + 1e-300)``.
    """
    components = _multi_context([1e300 * np.eye(2), np.eye(2)], 1e-150 * np.eye(2))
    np.testing.assert_array_equal(components[0].omega_ssp, np.eye(2))
    evaluation = penalty_logdet({"g:c0": 1e10, "g:c1": 1.0}, components)
    exact = 2 * math.log(1e10)
    assert evaluation.rank == 2
    assert abs(evaluation.logdet - exact) <= 8 * EPS * abs(exact)
    assert abs(evaluation.gradient["g:c0"] - 2.0) <= 8 * EPS
    assert 0.0 <= evaluation.gradient["g:c1"] <= 1e-300


def _failing_eigvalsh(matrix):
    raise np.linalg.LinAlgError("Eigenvalues did not converge")


@pytest.mark.parametrize("eigensolver_fails", [False, True])
def test_a_kronecker_margin_near_the_top_of_binary64(monkeypatch, eigensolver_fails):
    """``A = 1e308 (I + 0.02 J)`` and ``I_2``, as ``A (x) I_2`` and ``I_6``, under ``1e-154 I``.

    ``A + A'`` overflows, so the margin is halved before the sum and the
    closed form holds; a margin the eigensolver fails on leaves the block to
    its support roots.  The cells are ``1e-308 (a_i + 1)`` for ``A``'s
    eigenvalues ``d + 2o`` and ``d - o`` (twice), each repeated twice.
    """
    if eigensolver_fails:
        margins = algebra._kronecker_margins

        def failing_margins(support, raw):
            with monkeypatch.context() as patch:
                patch.setattr(np.linalg, "eigvalsh", _failing_eigvalsh)
                return margins(support, raw)

        monkeypatch.setattr(algebra, "_kronecker_margins", failing_margins)
    margin = 1e308 * (np.eye(3) + 0.02 * np.ones((3, 3)))
    components = _multi_context([np.kron(margin, np.eye(2)), np.eye(6)], 1e-154 * np.eye(6))
    block = algebra._context_geometry(components).plain
    assert (block.kronecker is None) == eigensolver_fails
    evaluation = penalty_logdet({"g:c0": 1.0, "g:c1": 1.0}, components)
    diagonal, off = margin[0, 0], margin[0, 1]
    terms = [2 * math.log(diagonal + 2 * off), 4 * math.log(diagonal - off), 12 * math.log(1e-154)]
    assert evaluation.rank == 6
    assert abs(evaluation.logdet - math.fsum(terms)) <= 2 * 6 * EPS * math.fsum(map(abs, terms))
    assert abs(evaluation.gradient["g:c0"] - 6.0) <= 12 * EPS
    assert 0.0 <= evaluation.gradient["g:c1"] <= 1e-300


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
    # Each margin's eigenvalues move by width * eps of its largest, and the
    # reference's normal-equation volume carries cond(mapped)**2 rounding.
    condition = max(a.max() / a[a > 0].min(), b.max() / b[b > 0].min(), np.linalg.cond(mapped) ** 2)
    assert abs(evaluation.logdet - expected) <= 2 * evaluation.rank * 20 * EPS * condition


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
    assert abs(evaluation.logdet - reference[1]) <= _dense_tolerance(components, 6, reference[4])
    assert evaluation.gradient == {"g:c0": 6.0, "g:c1": 0.0}


def test_a_face_whose_stored_penalty_loses_a_raw_direction_is_refused():
    """``diag(1, 1, 0)`` and ``diag(0, 0, 1)`` under ``C = [[1, 1, 0], [0, 1e-8, 0], [0, 0, 1]]``.

    With the second weight zero the raw face has rank 2, but the stored first
    penalty ``C' diag(1, 1, 0) C`` rounds to rank 1, which the nullity counts.
    The face is refused, as the certified context refuses it, rather than
    pairing a rank-2 determinant with that nullity.
    """
    coordinate_map = np.array([[1.0, 1.0, 0.0], [0.0, 1e-8, 0.0], [0.0, 0.0, 1.0]])
    raw = [np.diag([1.0, 1.0, 0.0]), np.diag([0.0, 0.0, 1.0])]
    components = _multi_context(raw, coordinate_map)
    np.testing.assert_array_equal(components[0].omega_ssp, [[1, 1, 0], [1, 1, 0], [0, 0, 0]])
    with pytest.raises(support_module.PenaltyNumericalError, match="preserve the raw rank"):
        penalty_logdet({"g:c0": 1.0, "g:c1": 0.0}, components)


def test_stored_penalty_without_the_raw_rank_keeps_the_solver_space_support():
    # C' C = [[1, 1], [1, 1 + 1e-16]] rounds to rank one in binary64.  The
    # agreement bars refuse this map; the next map rechecks them.
    components = _single_context(np.eye(2), np.array([[1.0, 1.0], [0.0, 1e-8]]))
    geometry = algebra._context_geometry(components)
    assert geometry.coordinate_map is None and not geometry.raw_refused
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
    # Doubling the map doubles the transported root exactly, so the constants
    # differ by 2 r log 2 up to the rounding of the r logarithms each sums.
    expected = penalty_logdet({"s": 3.0}, first).logdet + 2 * 5 * math.log(2.0)
    assert abs(penalty_logdet({"s": 3.0}, later).logdet - expected) <= 4 * 5 * EPS * abs(expected)
    # A selection the raw penalty itself refuses is carried, not reselected.
    asymmetric = omega.copy()
    asymmetric[0, 1] += 1e-9
    refused = _single_context(asymmetric, coordinate_map)
    counts["support"] = 0
    carried = _single_context(asymmetric, 2 * coordinate_map, source=refused)
    assert counts["support"] == 0
    assert algebra._context_geometry(carried).raw_refused


def test_a_map_the_agreement_bars_refuse_leaves_the_next_map_to_its_own_check(monkeypatch):
    """Discrete REML and EFS rebuild ``C`` as lambda moves; only map-free refusals persist.

    ``C = [[1, 1], [0, d]]`` on ``Omega = I`` stores ``C' C``, whose smallest
    eigenvalue ``d**2 / 2`` the rounding of ``1 + d**2`` resolves to ``eps / d**2``
    relative: the bars refuse ``d = 1e-5`` (``x`` about ``1800 sqrt(eps)``) and admit
    ``d = 1e-3`` (``x`` about ``0.18 sqrt(eps)``).
    """
    counts = {"support": 0}
    original = support_module._penalty_support

    def counted(*args, **kwargs):
        counts["support"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(support_module, "_penalty_support", counted)
    refused = _single_context(np.eye(2), np.array([[1.0, 1.0], [0.0, 1e-5]]))
    first = algebra._context_geometry(refused)
    assert first.coordinate_map is None and not first.raw_refused
    assert first.raw_support is not None and counts["support"] == 1
    admitted = _single_context(np.eye(2), np.array([[1.0, 1.0], [0.0, 1e-3]]), source=refused)
    later = algebra._context_geometry(admitted)
    assert counts["support"] == 1
    assert later.coordinate_map is not None and later.support is first.raw_support
    expected = 2 * math.log(1e-3)
    assert abs(penalty_logdet({"s": 1.0}, admitted).logdet - expected) <= 4 * 2 * EPS * abs(
        expected
    )


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
        assert geometry.raw_inputs is None and geometry.raw_support is None
        assert penalty_logdet(lambdas, copied) == expected


def test_a_saved_plain_tensor_context_carries_the_enclosed_ssp_roots():
    """A release before the plain evaluator evaluates a zero-weight face from enclosed SSP roots.

    The certified evaluator stays in this module, so it stands in for that
    release: the reloaded context's face agrees with the plain one within its
    certified bound plus the plain evaluation's rounding.
    """
    first, second, _, _ = _kronecker_pair()
    components = _multi_context([first, second], _well_conditioned_map(20, 5))
    assert algebra._context_geometry(components).ssp_roots == ()
    loaded = pickle.loads(pickle.dumps(components, protocol=5))
    geometry = algebra._context_geometry(loaded)
    assert [root.shape for root in geometry.ssp_roots] == [
        error.shape for error in geometry.ssp_root_errors
    ]
    summary, _, bound = geometry.evaluate(np.array([2.0, 0.0]))
    plain = penalty_logdet({"g:c0": 2.0, "g:c1": 0.0}, components)
    assert summary.rank == plain.rank
    rounding = 2 * plain.rank * 20 * EPS * max(1.0, abs(plain.logdet))
    assert abs(summary.logdet_s_plus - plain.logdet) <= bound + rounding


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
    tolerance = _dense_tolerance(plain, new.rank, _dense_reference(plain, lambdas)[4])
    assert abs(old.logdet - new.logdet) <= tolerance
    reference = algebra._compute_penalty_logdet_evaluation(lambdas, certified)
    assert abs(reference.logdet - new.logdet) <= reference.logdet_error + tolerance
