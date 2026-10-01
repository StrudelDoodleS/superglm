"""A single dense penalty is selected once per fit, on its raw basis.

Discrete REML and EFS rebuild each smooth's SSP map ``C = R_inv`` whenever
lambda moves. Because log pdet(C.T Omega C) = log pdet(Omega) +
log det(U.T C C.T U) for an orthonormal range basis U of Omega, the raw support
and its unit-weight summary are lambda-free and transfer between REML steps;
the certified coordinate volume and the Weyl agreement with the stored solver
penalty follow each map.
"""

import copy
import pickle
from dataclasses import replace
from decimal import Decimal, localcontext
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from superglm import Categorical, Spline, SuperGLM
from superglm.reml import multi_penalty as kernel
from superglm.reml import penalty_algebra as algebra
from superglm.reml import penalty_support as support_module
from superglm.reml.objective import reml_laml_objective
from superglm.solvers.dispersion import model_weight_semantics
from superglm.types import GroupInfo


def _data():
    rng = np.random.default_rng(11)
    n = 3000
    x1, x2, x3 = rng.uniform(size=(3, n))
    level = rng.choice(np.array(["a", "b", "c"]), size=n)
    y = rng.poisson(np.exp(-0.5 + np.sin(2 * np.pi * x1) + 0.5 * x2**2 + 0.3 * (level == "b")))
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "level": level}), y


def _fit(discrete):
    frame, y = _data()
    features = {
        "x1": Spline(kind="ps", k=10),
        "x2": Spline(kind="ps", k=10),
        "x3": Spline(kind="ps", k=8),
        "level": Categorical(),
    }
    return SuperGLM(family="poisson", discrete=discrete, features=features).fit_reml(frame, y)


def _solver_space(component):
    """The same component on its solver-space support, as before the raw path."""
    copied = replace(component)
    algebra._attach_context_geometry([copied])
    return copied


def _context(omega, coordinate_map, source=None):
    width = coordinate_map.shape[1]
    group = SimpleNamespace(name="s", sl=slice(0, width), size=width)
    matrix = SimpleNamespace(R_inv=coordinate_map, omega=omega)
    kwargs = {} if source is None else {"_reuse_raw_from": source}
    return algebra.build_penalty_context([matrix], [(0, group)], **kwargs)[0]


def _second_differences(width):
    """Omega = D.T D for integer D, exact in binary64 and in Decimal."""
    D = np.diff(np.eye(width), n=2, axis=0)
    return D, D.T @ D


def _decimal(matrix):
    return [[Decimal.from_float(float(value)) for value in row] for row in np.asarray(matrix)]


def _product(left, right):
    return [
        [sum((a * b for a, b in zip(row, column)), Decimal(0)) for column in zip(*right)]
        for row in left
    ]


def _transpose(matrix):
    return [list(column) for column in zip(*matrix)]


def _ln_det(matrix):
    """ln det of a symmetric positive definite Decimal matrix, by elimination."""
    work, total = [row[:] for row in matrix], Decimal(0)
    for column in range(len(work)):
        pivot = work[column][column]
        assert pivot > 0
        total += pivot.ln()
        for row in range(column + 1, len(work)):
            factor = work[row][column] / pivot
            for j in range(column + 1, len(work)):
                work[row][j] -= factor * work[column][j]
    return total


def _inverse(matrix):
    size = len(matrix)
    work = [row[:] + [Decimal(int(i == j)) for j in range(size)] for i, row in enumerate(matrix)]
    for column in range(size):
        pivot = work[column][column]
        work[column] = [value / pivot for value in work[column]]
        for row in range(size):
            if row != column:
                factor = work[row][column]
                work[row] = [a - factor * b for a, b in zip(work[row], work[column], strict=True)]
    return [row[size:] for row in work]


def _frobenius(matrix):
    return sum((value * value for row in matrix for value in row), Decimal(0)).sqrt()


def _weyl_bridge(stored, root):
    """Bound |log pdet_r(stored) - log pdet(root.T root)| for an exact rank-r root.

    Weyl: every eigenvalue moves by at most ||stored - root.T root||_2 <= the
    Frobenius norm e. With lambda_r(root.T root) = lambda_min(root root.T) >=
    1 / ||(root root.T)^-1||_F and x = e / lambda_r < 1/2, the retained
    log-eigenvalues move by at most r x / (1 - x) in total.
    """
    ideal = _product(_transpose(root), root)
    difference = [[a - b for a, b in zip(u, v, strict=True)] for u, v in zip(stored, ideal)]
    smallest = 1 / _frobenius(_inverse(_product(root, _transpose(root))))
    x = _frobenius(difference) / smallest
    assert x < Decimal("0.5")
    return len(root) * x / (1 - x)


@pytest.mark.parametrize("discrete", [True, False])
def test_fit_builds_each_penalty_support_once(monkeypatch, discrete):
    counts = {"support": 0, "unit": 0}
    build_support = support_module._penalty_support
    summarize = kernel._evaluate_penalty_summary

    def counted_support(*args, **kwargs):
        counts["support"] += 1
        return build_support(*args, **kwargs)

    def counted_summary(support, values, *args, **kwargs):
        counts["unit"] += bool(np.all(np.asarray(values) == 1.0))
        return summarize(support, values, *args, **kwargs)

    monkeypatch.setattr(support_module, "_penalty_support", counted_support)
    monkeypatch.setattr(kernel, "_evaluate_penalty_summary", counted_summary)
    model = _fit(discrete)
    groups = {component.group_name for component in model._reml_penalties}
    assert len(groups) == 3
    # Every REML step rebuilds the SSP maps; the raw selection must not follow.
    assert counts["support"] <= len(groups)
    assert counts["unit"] <= len(groups)
    geometries = [algebra._context_geometry([item]) for item in model._reml_penalties]
    assert all(geometry.coordinate_map is not None for geometry in geometries)
    # The fitted model's terminal context hands nothing on, so it keeps no receipts.
    assert all(
        geometry.raw_family is None and geometry.raw_summary is None for geometry in geometries
    )


def test_raw_support_has_the_solver_space_rank_and_log_lambda_derivatives():
    # Log-determinant values are checked against exact targets below; the two
    # supports' values certify different representatives, so they are not
    # compared with each other here.
    model = _fit(discrete=True)
    for component in model._reml_penalties:
        weights = {component.name: model._reml_lambdas[component.name]}
        geometry = algebra._context_geometry([component])
        assert geometry.coordinate_map is not None and geometry.volume[0] != 0.0
        raw = algebra._compute_penalty_logdet_evaluation(weights, [component])
        reference = _solver_space(component)
        assert algebra._context_geometry([reference]).coordinate_map is None
        solver = algebra._compute_penalty_logdet_evaluation(weights, [reference])
        assert raw.rank == solver.rank == component.rank
        assert raw.gradient == solver.gradient
        assert raw.hessian == solver.hessian


def test_projected_map_logdet_is_certified_against_exact_arithmetic():
    # A centred spline's map drops one raw coordinate: C is 7 x 6, not square.
    D, omega = _second_differences(7)
    coordinate_map = np.random.default_rng(5).normal(size=(7, 6))
    components = _context(omega, coordinate_map)
    geometry = algebra._context_geometry(components)
    assert geometry.coordinate_map.shape == (7, 6) and geometry.support.rank == 5
    evaluation = algebra._compute_penalty_logdet_evaluation({"s": 3.0}, components)
    assert evaluation.rank == 5
    with localcontext() as context:
        context.prec = 100
        # Independent of the selected support: Omega C = D.T (D C) exactly, so
        # log pdet(C.T Omega C) = ln det(Y Y.T) for Y = D C. The certificate
        # covers the stored solver penalty; Weyl bridges it to C.T Omega C.
        root = _product(_decimal(D), _decimal(coordinate_map))
        expected = _ln_det(_product(root, _transpose(root))) + 5 * Decimal(3).ln()
        bridge = _weyl_bridge(_decimal(components[0].omega_ssp), root)
        error = abs(Decimal.from_float(evaluation.logdet) - expected)
        assert error <= Decimal.from_float(float(evaluation.logdet_error)) + bridge


def test_full_rank_logdet_is_certified_against_the_stored_solver_penalty():
    _, omega = _second_differences(6)
    omega = omega + 2 * np.eye(6)
    coordinate_map = np.random.default_rng(2).normal(size=(6, 6))
    components = _context(omega, coordinate_map)
    assert algebra._context_geometry(components).coordinate_map is not None
    evaluation = algebra._compute_penalty_logdet_evaluation({"s": 3.0}, components)
    assert evaluation.rank == 6
    with localcontext() as context:
        context.prec = 100
        expected = _ln_det(_decimal(components[0].omega_ssp)) + 6 * Decimal(3).ln()
        error = abs(Decimal.from_float(evaluation.logdet) - expected)
        assert error <= Decimal.from_float(float(evaluation.logdet_error))


@pytest.mark.parametrize("small", [1e-6, 1e-8])
def test_stored_penalty_rounding_keeps_its_rank_and_log_determinant(small):
    # C.T C = [[1, 1], [1, 1 + small**2]]. Binary64 stores 1 + 1e-12 with a
    # 1e-4 relative error in the smallest eigenvalue, and rounds 1 + 1e-16 to
    # 1, so the stored solver penalty loses its second direction.
    components = _context(np.eye(2), np.array([[1.0, 1.0], [0.0, small]]))
    reference = [_solver_space(components[0])]
    evaluation = algebra._compute_penalty_logdet_evaluation({"s": 1.0}, components)
    assert algebra._context_geometry(components).coordinate_map is None
    assert evaluation == algebra._compute_penalty_logdet_evaluation({"s": 1.0}, reference)
    nullity = [
        algebra.compute_penalty_nullity(
            hessian_rank=3, penalties=penalties, lambdas={"s": 1.0}, coefficient_width=2
        )
        for penalties in (components, reference)
    ]
    assert nullity[0] == nullity[1]
    # The stored penalty's exact determinant decides its rank.
    (a, b), (c, d) = _decimal(components[0].omega_ssp)
    assert evaluation.rank == (2 if a * d - b * c > 0 else 1)


class _ScaledContrast:
    """Two sign columns, the second a 1e8-scaled contrast, with an identity penalty."""

    def __init__(self, scale):
        self.scale, self.R_inv = scale, None

    def raw(self, x):
        z1, z2 = np.where(x < 2, 1.0, -1.0), np.where(x % 2 == 0, 1.0, -1.0)
        return np.column_stack([z1, self.scale * (z2 - z1)])

    def build(self, x, sample_weight=None):
        basis = sp.csr_matrix(self.raw(x))
        return GroupInfo(basis, 2, penalty_matrix=np.eye(2), reparametrize=True)

    def set_reparametrisation(self, R_inv):
        self.R_inv = R_inv

    def transform(self, x):
        return self.raw(x) @ self.R_inv

    def reconstruct(self, beta):
        return {"beta": self.R_inv @ beta}


def test_scaled_contrast_fit_selects_the_solver_space_smoothing_parameter(monkeypatch):
    x = np.tile(np.arange(4), 75)
    z1, z2 = np.where(x < 2, 1.0, -1.0), np.where(x % 2 == 0, 1.0, -1.0)
    y = np.random.default_rng(4).poisson(np.exp(0.3 + 0.5 * z1 + 0.2 * z2))
    frame = pd.DataFrame({"x": x})

    def fit():
        return SuperGLM(family="poisson", features={"x": _ScaledContrast(1e8)}).fit_reml(frame, y)

    model = fit()
    # The solver-space support everywhere is the behaviour before the raw path.
    monkeypatch.setattr(algebra, "_single_penalty_raw_family", lambda *args: None)
    reference = fit()
    assert model._reml_lambdas == reference._reml_lambdas
    assert model._reml_result.objective == reference._reml_result.objective
    ranks = [
        algebra._compute_penalty_logdet_evaluation(
            item._reml_lambdas, list(item._reml_penalties)
        ).rank
        for item in (model, reference)
    ]
    assert ranks[0] == ranks[1]


def _counted(monkeypatch, counts):
    for module, name in [
        (support_module, "_penalty_support"),
        (algebra, "_support_coordinate_volume"),
    ]:
        original = getattr(module, name)

        def counted(*args, _original=original, _name=name, **kwargs):
            counts[_name] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(module, name, counted)


def test_raw_rank_that_differs_from_the_declared_rank_keeps_the_solver_space_support(
    monkeypatch,
):
    # Fourth differences on 80 coefficients: the eps**(2/3) declared rank
    # drops one direction that the raw support keeps.
    D = np.diff(np.eye(80), n=4, axis=0)
    omega = D.T @ D
    coordinate_map = np.linalg.inv(np.linalg.cholesky(np.eye(80) + omega).T)
    counts = dict.fromkeys(["_penalty_support", "_support_coordinate_volume"], 0)
    _counted(monkeypatch, counts)
    components = _context(omega, coordinate_map)
    assert support_module._penalty_support([omega]).rank != components[0].rank
    geometry = algebra._context_geometry(components)
    assert geometry.coordinate_map is None and geometry.raw_refusal is not None
    evaluation = algebra._compute_penalty_logdet_evaluation({"s": 2.0}, components)
    reference = [_solver_space(components[0])]
    assert evaluation == algebra._compute_penalty_logdet_evaluation({"s": 2.0}, reference)
    assert evaluation.rank == components[0].rank
    # The rank check refuses before any map is certified, and a later context
    # of the same fit carries the refusal instead of reselecting the support.
    assert counts["_support_coordinate_volume"] == 0
    counts["_penalty_support"] = 0
    later = _context(omega, 2 * coordinate_map, source=components)
    assert counts == {"_penalty_support": 0, "_support_coordinate_volume": 0}
    assert algebra._context_geometry(later).raw_refusal is not None


def _refused(case, monkeypatch):
    from superglm.reml.penalty_support import PenaltyNumericalError

    _, omega = _second_differences(7)
    coordinate_map = np.random.default_rng(5).normal(size=(7, 6))
    if case == "float32 map":
        coordinate_map = coordinate_map.astype(np.float32)
    elif case == "identity map":
        coordinate_map = np.eye(7)
    elif case == "truncating map":
        coordinate_map = np.eye(7, 6)
    elif case == "asymmetric raw penalty":
        omega[0, 1] += 1e-9
    else:

        def refuse(*args, **kwargs):
            raise PenaltyNumericalError(case)

        name = "_support_coordinate_volume" if case == "volume" else "_solver_penalty_agreement"
        monkeypatch.setattr(algebra, name, refuse)
    return omega, coordinate_map


@pytest.mark.parametrize(
    "case",
    [
        "float32 map",
        "identity map",
        "truncating map",
        "asymmetric raw penalty",
        "volume",
        "agreement",
    ],
)
def test_refused_raw_support_keeps_the_solver_space_rank_and_logdet(monkeypatch, case):
    omega, coordinate_map = _refused(case, monkeypatch)
    components = _context(omega, coordinate_map)
    geometry = algebra._context_geometry(components)
    assert geometry.coordinate_map is None
    # Declined maps are cheap to recheck; refusals after raw work are carried.
    declined = case in {"float32 map", "identity map", "truncating map"}
    assert (geometry.raw_refusal is None) is declined
    evaluation = algebra._compute_penalty_logdet_evaluation({"s": 3.0}, components)
    reference = [_solver_space(components[0])]
    assert evaluation == algebra._compute_penalty_logdet_evaluation({"s": 3.0}, reference)
    assert evaluation.rank == components[0].rank


def test_unusable_maps_are_declined_before_any_raw_work():
    _, omega = _second_differences(7)
    coordinate_map = np.random.default_rng(5).normal(size=(7, 6))
    components = _context(omega, coordinate_map)
    for unusable in (np.where(np.eye(7, 6) > 0, np.nan, coordinate_map), coordinate_map[:, :5]):
        matrix = SimpleNamespace(R_inv=unusable, omega=omega)
        assert algebra._single_penalty_raw_family(matrix, components, 5.0, None) is None


@pytest.mark.parametrize("method", ["pickle protocol 4", "pickle protocol 5", "deepcopy"])
def test_reloaded_context_keeps_the_raw_support(method):
    _, omega = _second_differences(7)
    components = _context(omega, np.random.default_rng(5).normal(size=(7, 6)))
    expected = algebra._compute_penalty_logdet_evaluation({"s": 3.0}, components)
    if method == "deepcopy":
        copied = copy.deepcopy(components)
    else:
        copied = pickle.loads(pickle.dumps(components, protocol=int(method[-1])))
    geometry = algebra._context_geometry(copied)
    assert geometry is not None and geometry.coordinate_map is not None
    assert algebra._compute_penalty_logdet_evaluation({"s": 3.0}, copied) == expected


def test_reml_objective_moves_only_through_the_logdet():
    """At one fitted state, the two supports change the objective only through log|S|+.

    Ranks, nullity and log-lambda derivatives are equal, so a REML step can
    differ only where it compares objective values, by half the log-determinant
    difference plus the rounding of the objective's last operations.
    """
    model = _fit(discrete=False)
    _, y = _data()
    penalties = list(model._reml_penalties)
    assert all(algebra._context_geometry([item]).coordinate_map is not None for item in penalties)
    solver_space = [_solver_space(component) for component in penalties]
    lambdas = model._reml_lambdas
    evaluations, logdets = [], []
    for reml_penalties in (penalties, solver_space):
        evaluations.append(
            reml_laml_objective(
                model._dm,
                model._distribution,
                model._link,
                model._groups,
                y,
                model.result,
                lambdas,
                np.ones(len(y)),
                np.zeros(len(y)),
                log_det_H=model.result.log_det_H,
                hessian_rank=model.result.reml_hessian_rank,
                reml_penalties=reml_penalties,
                return_evaluation=True,
                weight_semantics=model_weight_semantics(model),
            )
        )
        logdets.append(algebra._compute_penalty_logdet_evaluation(lambdas, reml_penalties))
    raw, solver = evaluations
    assert raw.penalty_quad == solver.penalty_quad
    # Poisson has a known scale, so the objective reports no nullity; compute
    # Wood's M_p directly from each list of penalties.
    nullity = [
        algebra.compute_penalty_nullity(
            hessian_rank=model.result.reml_hessian_rank,
            penalties=items,
            lambdas=lambdas,
            coefficient_width=len(model.result.beta),
        )
        for items in (penalties, solver_space)
    ]
    assert nullity[0] == nullity[1]
    assert logdets[0].rank == logdets[1].rank
    assert logdets[0].gradient == logdets[1].gradient
    assert logdets[0].hessian == logdets[1].hessian
    # value_k = fl(nll + 0.5 fl(a - s_k)) with a = fl(pq + log|H|) shared and
    # s_k each support's log|S|+. With |delta| <= u per rounding (Higham 2002,
    # section 2.2) and exact halving,
    #   |v_0 - v_1 - (s_1 - s_0)/2| <= u/(1-u) (|a-s_0|/2 + |a-s_1|/2 + |v_0| + |v_1|),
    # and |a - s_k| <= (1+u)(|pq| + |log|H||) + |s_k|. Evaluated in exact
    # rational arithmetic.
    unit = Fraction(1, 2**53)
    shared = (1 + unit) * (abs(Fraction(raw.penalty_quad)) + abs(Fraction(model.result.log_det_H)))
    values = [Fraction(raw.value), Fraction(solver.value)]
    logs = [Fraction(item.logdet) for item in logdets]
    differences = [shared + abs(item) for item in logs]
    bound = unit / (1 - unit) * (sum(differences) / 2 + abs(values[0]) + abs(values[1]))
    assert abs(values[0] - values[1] - (logs[1] - logs[0]) / 2) <= bound
