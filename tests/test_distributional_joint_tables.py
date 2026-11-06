"""Joint tables and compiled rows read the same numbers as the group routes and the natural route."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import superglm.distributional.kernels.tweedie as tweedie_kernels
import superglm.distributional.solver.chunks as chunking
from superglm import SuperLSS, cat, s
from superglm._frame import as_eager_frame
from superglm._group_matrix._group_matrix_algebra import PassFactors
from superglm._group_matrix._group_matrix_core import CategoricalGroupMatrix
from superglm._parallel import parallel_config
from superglm.distributional.families.gamma import GammaLS
from superglm.distributional.families.gaussian import GaussianLS
from superglm.distributional.families.tweedie import TweedieLSS
from superglm.distributional.family import COMPLETE_OBSERVATION
from superglm.distributional.kernels._common import _NumericalEvaluationError
from superglm.distributional.layout import build_stacked_layout
from superglm.distributional.predictor import Predictor, compile_predictors
from superglm.distributional.solver import _joint_tables
from superglm.distributional.solver import solver as solver_api
from superglm.distributional.solver._joint_tables import joint_table_plan, stack_in_range
from superglm.distributional.solver.derivatives import transform_natural_derivatives
from superglm.features import Categorical, Numeric, Spline

from ._distributional_weights import resolved_prior

N = 6000
# Rounding of a sum over the rows: u times n, against the entry's scale.
ROUND = 2.0**-53 * N * 8


def _frame(n=N, seed=20261106):
    rng = np.random.default_rng(seed)
    return as_eager_frame(
        pd.DataFrame(
            {
                "x": rng.choice(np.linspace(-1, 1, 25), n),
                "z": rng.choice(np.linspace(0, 3, 31), n),
                "g": rng.choice(["a", "b", "c", "d"], n),
                "h": rng.choice(["u", "v"], n),
            }
        )
    ), rng


def _problem(family, offsets=False):
    frame, rng = _frame()
    weights = resolved_prior(rng.choice(np.linspace(0.1, 1.0, 10), N))
    names = [spec.name for spec in family.parameters]
    features = [
        {
            "x": Spline(kind="cr", n_knots=8, discrete=True),
            "z": Spline(kind="cr", n_knots=6, discrete=True),
            "g": Categorical(),
            "h": Categorical(),
        },
        # The same covariates as the first predictor, so their channels share row passes.
        {"x": Spline(kind="cr", n_knots=5, discrete=True), "h": Categorical()},
    ]
    predictors = tuple(
        Predictor(name, features[index] if index < 2 else {}) for index, name in enumerate(names)
    )
    layout = build_stacked_layout(
        compile_predictors(
            frame,
            weights,
            family.parameters,
            predictors,
            offsets={names[0]: 0.05 * rng.normal(size=N)} if offsets else None,
            model_discrete=True,
            n_bins_config=20,
        )
    )
    if isinstance(family, TweedieLSS):
        positive = rng.random(N) < 0.15
        response = np.where(positive, rng.gamma(2.0, 1.5, N), 0.0)
    else:
        response = rng.gamma(2.0, 1.5, N) + 0.1
    plan = family.bind_likelihood(response, weights, COMPLETE_OBSERVATION)
    coefficients = 0.02 * rng.normal(size=layout.n_coefficients)
    return family, layout, response, plan, coefficients


def _geometry(
    problem, *, joint=True, compiled=True, factors=None, monkeypatch=None, sums=None, chunk="auto"
):
    family, layout, response, plan, coefficients = problem
    if not joint:
        monkeypatch.setattr(chunking, "_joint_plan", lambda *_args: None)
    try:
        return chunking.assemble_chunked_geometry(
            family,
            layout,
            response,
            plan,
            coefficients,
            penalty=np.eye(layout.n_coefficients) * 0.1,
            chunk_size=chunk,
            curvature_source="observed",
            pass_factors=PassFactors() if factors is None else factors,
            _likelihood_sums=sums,
            _compiled=compiled,
        )
    finally:
        if not joint:
            monkeypatch.undo()


def _assert_same_geometry(left, right):
    scale = np.sqrt(np.abs(np.diag(left.data_curvature)))
    entry = np.outer(scale, scale)
    assert np.max(np.abs(left.data_curvature - right.data_curvature) / entry) <= ROUND
    assert np.max(np.abs(left.score_data - right.score_data) / (scale * np.sqrt(N))) <= ROUND
    np.testing.assert_array_equal(right.data_curvature, right.data_curvature.T)


FAMILIES = (
    lambda: GaussianLS(scale_floor=0.02),
    lambda: GammaLS(),
    lambda: TweedieLSS(),
)


@pytest.mark.parametrize("make", FAMILIES, ids=("gaussian", "gamma", "tweedie"))
@pytest.mark.parametrize("offsets", (False, True), ids=("plain", "offsets"))
def test_joint_tables_read_what_the_group_routes_form(monkeypatch, make, offsets):
    """Mutation checks: a flipped pair read the wrong way round, a sink level kept in a
    contraction and a table read at the wrong slot all fail here."""
    problem = _problem(make(), offsets=offsets)
    plan = joint_table_plan(problem[1], 1 << 26)
    assert plan is not None
    # A row pass per distinct pair: the second predictor's blocks share the first's columns.
    assert {(table.left, table.right) for table in plan.tables if table.right is not None} == {
        (a, b) for a in range(4) for b in range(a + 1, 4)
    }
    groups = _geometry(problem, joint=False, monkeypatch=monkeypatch)
    tables = _geometry(problem, compiled=False)
    _assert_same_geometry(groups, tables)
    major = tables.formed().majorant
    np.testing.assert_allclose(major, groups.formed().majorant, rtol=ROUND, atol=0.0)


def test_the_compiled_rows_read_what_the_natural_route_forms(monkeypatch):
    problem = _problem(TweedieLSS(), offsets=True)
    sums = [0.0, 0.0]
    compiled = _geometry(problem, sums=sums)
    natural_sums = [0.0, 0.0]
    natural = _geometry(problem, compiled=False, sums=natural_sums)
    _assert_same_geometry(natural, compiled)
    assert abs(sums[0] - natural_sums[0]) <= ROUND * abs(natural_sums[0])
    np.testing.assert_allclose(compiled.formed().majorant, natural.formed().majorant, rtol=ROUND)
    assert compiled.formed().rows == natural.formed().rows


@pytest.mark.parametrize("semantics", ("prior", "frequency"))
@pytest.mark.parametrize("shared_power", (True, False), ids=("one_power", "row_powers"))
def test_tweedie_predictor_rows_are_the_chain_rule_of_the_natural_rows(semantics, shared_power):
    """Mutation checks: a sign in any of the nine channels, or a power read from the wrong row."""
    rng = np.random.default_rng(7)
    n = 500
    y = np.where(rng.random(n) < 0.3, rng.gamma(2.0, 2.0, n), 0.0)
    weight = (
        rng.integers(1, 6, n).astype(float) if semantics == "frequency" else rng.random(n) + 0.1
    )
    eta = np.column_stack(
        (rng.normal(0.2, 1.0, n), rng.normal(0.1, 0.5, n), rng.normal(0.0, 0.8, n))
    )
    if shared_power:
        eta[:, 2] = 0.3
    family = TweedieLSS()
    links = tuple(spec.default_link for spec in family.parameters)
    links = tuple(_resolved_link(link) for link in links)
    power_link = links[2]
    rows = tweedie_kernels.tweedie_predictor_rows(y, weight, semantics, power_link, shared_power)
    stack = np.empty((9, n))
    stack[:3] = eta.T
    completed, value = rows.evaluate(0, 0, n, stack, 2)
    assert completed
    theta = np.column_stack((np.exp(eta[:, 0]), np.exp(eta[:, 1]), power_link.inverse(eta[:, 2])))
    natural = tweedie_kernels.evaluate_tweedie_rows(
        y, theta[:, 0], theta[:, 1], theta[:, 2], weight, semantics, derivative_order=2
    )
    evaluation = _natural(natural)
    transformed = transform_natural_derivatives(evaluation, eta, links)
    np.testing.assert_allclose(stack[:6].T, transformed.curvature_packed, rtol=1e-11, atol=1e-13)
    np.testing.assert_allclose(stack[6:9].T, transformed.score_eta, rtol=1e-11, atol=1e-13)
    assert value == pytest.approx(float(np.sum(natural.log_likelihood)), rel=1e-13)


def _resolved_link(link):
    from superglm.links import resolve_link

    return resolve_link(link) if isinstance(link, str) else link


def _natural(evaluated):
    from superglm.distributional.family import NaturalLikelihoodEvaluation

    return NaturalLikelihoodEvaluation(
        optimizing_log_likelihood=evaluated.log_likelihood,
        parameter_independent_carrier=np.zeros_like(evaluated.log_likelihood),
        score=evaluated.score,
        hessian_packed=evaluated.hessian_packed,
        valid=evaluated.valid,
    )


def test_a_zero_row_outside_the_window_takes_the_natural_route(monkeypatch):
    """The rows decline, and the pass is the natural route's. Mutation check: a window on
    the mean alone keeps the row and skips the natural route's reading of it."""
    family, layout, response, plan, coefficients = _problem(TweedieLSS())
    state = layout.predictors[0]
    coefficients = coefficients.copy()
    coefficients[state.intercept_index] = 101.0
    asked = []
    natural = TweedieLSS.evaluate_natural

    def counted(self, *args, **kwargs):
        asked.append(kwargs.get("derivative_order"))
        return natural(self, *args, **kwargs)

    monkeypatch.setattr(TweedieLSS, "evaluate_natural", counted)
    problem = (family, layout, response, plan, coefficients)
    factors = PassFactors()
    try:
        compiled = _geometry(problem, factors=factors)
    except ValueError:
        compiled = None
    assert asked, "the pass must have reached the natural route"
    try:
        natural_route = _geometry(problem, compiled=False)
    except ValueError:
        natural_route = None
    assert (compiled is None) == (natural_route is None)
    if compiled is not None:
        _assert_same_geometry(natural_route, compiled)


def test_channels_past_the_plain_range_decline_the_compiled_rows(monkeypatch):
    """A dispersion predictor near the window's edge sets each zero row's channels past
    2**128, so the pass declines to the natural route and the group routes read it.

    Mutation checks: a compiled pass that tests no channel, or none of the rows it
    wrote, keeps channels the tables cannot hold in range.
    """
    family, layout, _response, plan, coefficients = _problem(TweedieLSS())
    response = np.zeros(N)
    plan = family.bind_likelihood(response, plan.weights, COMPLETE_OBSERVATION)
    asked = []
    natural = TweedieLSS.evaluate_natural

    def counted(self, *args, **kwargs):
        asked.append(kwargs.get("derivative_order"))
        return natural(self, *args, **kwargs)

    monkeypatch.setattr(TweedieLSS, "evaluate_natural", counted)
    _geometry((family, layout, response, plan, coefficients))
    assert not asked
    coefficients = coefficients.copy()
    coefficients[layout.predictors[1].intercept_index] = -95.0
    problem = (family, layout, response, plan, coefficients)
    declined = _geometry(problem)
    assert asked
    _assert_same_geometry(_geometry(problem, joint=False, monkeypatch=monkeypatch), declined)


def test_a_row_the_natural_route_refuses_is_refused_the_same_way():
    """A fused trial needs the refusal to be a derivative failure, which a value retry follows."""
    family, layout, response, plan, coefficients = _problem(TweedieLSS())
    coefficients = coefficients.copy()
    coefficients[layout.predictors[0].intercept_index] = 720.0
    problem = (family, layout, response, plan, coefficients)
    with pytest.raises(ValueError) as compiled:
        _geometry(problem)
    with pytest.raises(ValueError) as natural:
        _geometry(problem, compiled=False)
    assert type(compiled.value) is type(natural.value)
    assert str(compiled.value) == str(natural.value)


@pytest.mark.parametrize("eta_power", (40.0, -40.0), ids=("upper", "lower"))
def test_a_trial_with_its_power_on_a_wall_is_refused_as_the_natural_route_refuses_it(eta_power):
    """The family's power is strictly inside its walls, so a trial that reaches one is rejected and
    the search stays inside. Mutation check: drop the wall test from the row kernel, and the
    compiled rows give this trial a value and a geometry."""
    family, layout, response, plan, coefficients = _problem(TweedieLSS())
    coefficients = coefficients.copy()
    coefficients[layout.predictors[2].intercept_index] = eta_power
    with pytest.raises(ValueError, match="power") as compiled:
        _geometry((family, layout, response, plan, coefficients))
    with pytest.raises(ValueError, match="power") as natural:
        _geometry((family, layout, response, plan, coefficients), compiled=False)
    assert str(compiled.value) == str(natural.value)
    with pytest.raises(ValueError, match="power"):
        chunking.evaluate_chunked_log_likelihood(
            family, layout, response, plan, coefficients, chunk_size="auto",
            pass_factors=PassFactors(),
        )  # fmt: skip


def test_a_tweedie_refusal_is_a_numerical_evaluation_error(monkeypatch):
    family, layout, response, plan, coefficients = _problem(TweedieLSS())

    def refuse(*_args, **_kwargs):
        raise tweedie_kernels.TweedieNumericalRefusal("forced")

    monkeypatch.setattr(
        "superglm.distributional.families.tweedie.evaluate_tweedie_rows", refuse, raising=True
    )
    theta = np.ones((N, 3))
    theta[:, 2] = 1.5
    with pytest.raises(_NumericalEvaluationError, match="forced"):
        family.evaluate_natural(response, theta, plan, derivative_order=2)


def test_the_first_trial_of_a_tweedie_fit_owns_its_geometry():
    family, layout, response, plan, coefficients = _problem(TweedieLSS())
    context = solver_api._validated_context(
        family,
        layout,
        response,
        plan,
        np.eye(layout.n_coefficients) * 0.1,
        coefficient_curvature="observed",
        chunk_size="auto",
        coefficient_face=None,
    )
    assert solver_api._fused_first_trial_eligible(context)
    fused = solver_api._evaluate_fused_trial(context, coefficients, "observed", None)
    assert fused is not None
    state, geometry = fused
    separate = solver_api._evaluate_state(context, coefficients, derivative_order=0)
    assert state.penalized_optimizing_log_likelihood == pytest.approx(
        separate.penalized_optimizing_log_likelihood, rel=1e-13
    )
    _assert_same_geometry(solver_api._geometry(context, state, "observed"), geometry)


def test_a_refused_first_trial_falls_back_to_a_value_trial(monkeypatch):
    """Mutation check: leave Tweedie's refusal a plain ``ValueError`` and the trial raises."""
    family, layout, response, plan, coefficients = _problem(TweedieLSS())
    context = solver_api._validated_context(
        family,
        layout,
        response,
        plan,
        np.eye(layout.n_coefficients) * 0.1,
        coefficient_curvature="observed",
        chunk_size="auto",
        coefficient_face=None,
    )
    monkeypatch.setattr(chunking, "_compiled_rows", lambda *_args: None)

    def refuse(*_args, **_kwargs):
        raise tweedie_kernels.TweedieNumericalRefusal("forced")

    monkeypatch.setattr("superglm.distributional.families.tweedie.evaluate_tweedie_rows", refuse)
    assert solver_api._evaluate_fused_trial(context, coefficients, "observed", None) is None


@pytest.mark.parametrize("make", FAMILIES, ids=("gaussian", "gamma", "tweedie"))
@pytest.mark.parametrize("compiled", (True, False), ids=("compiled", "stacked"))
def test_row_shards_read_what_the_group_routes_form(monkeypatch, make, compiled):
    """Stack blocks that split shards, and shards that hold several stack blocks.

    Mutation checks: a shard left out of the merge or of the totals, rows read at
    their stack column and not their row, a block's rows evaluated from row 0, and a
    stack block's columns counted from the first row and not the block's.
    """
    problem = _problem(make())
    groups = _geometry(problem, joint=False, monkeypatch=monkeypatch)
    monkeypatch.setattr(_joint_tables, "_SHARD_ROWS", 257)
    monkeypatch.setattr(chunking, "STACK_BLOCK_ROWS", 31)
    monkeypatch.setattr(chunking, "contraction_block_rows", lambda _n, _bytes, chunk: chunk)
    assert joint_table_plan(problem[1], 1 << 26).n_shards > 1
    tables = _geometry(problem, compiled=compiled, chunk=1000)
    _assert_same_geometry(groups, tables)
    major = tables.formed().majorant
    np.testing.assert_allclose(major, groups.formed().majorant, rtol=ROUND, atol=0.0)


@pytest.mark.parametrize("make", FAMILIES, ids=("gaussian", "gamma", "tweedie"))
def test_the_bits_do_not_depend_on_the_workers(monkeypatch, make):
    """Each shard sums its rows in row order, the shards merge in order, and the partition
    reads the rows and tables alone. Mutation check: shards sized by the worker count."""
    monkeypatch.setattr(_joint_tables, "_SHARD_ROWS", 257)
    problem = _problem(make())
    with parallel_config(n_jobs=1):
        one = _geometry(problem)
    with parallel_config(n_jobs=4):
        four = _geometry(problem)
    np.testing.assert_array_equal(one.data_curvature, four.data_curvature)
    np.testing.assert_array_equal(one.score_data, four.score_data)


def test_the_shards_tables_take_at_most_half_the_allowance(monkeypatch):
    monkeypatch.setattr(_joint_tables, "_SHARD_ROWS", 257)
    layout = _problem(GaussianLS(scale_floor=0.02))[1]
    roomy = joint_table_plan(layout, 1 << 26)
    assert roomy.n_shards == np.ceil(N / 257)
    tight = joint_table_plan(layout, 4 * roomy.table_bytes)
    assert (tight.n_shards, tight.shard_rows) == (2, N // 2)
    assert joint_table_plan(layout, roomy.table_bytes).n_shards == 1


def test_a_row_block_holds_what_the_shards_tables_leave(monkeypatch):
    """Mutation check: blocks that fill the allowance whatever the shards hold."""
    problem = _problem(GaussianLS(scale_floor=0.02))
    monkeypatch.setattr(_joint_tables, "_SHARD_ROWS", 257)
    plan = joint_table_plan(problem[1], 1 << 26)
    chunk = 500
    allowance = plan.shard_bytes + 8 * plan.stack_rows * 2 * chunk
    monkeypatch.setattr(chunking, "AUTO_CONTRACTION_MEMORY_BYTES", allowance)
    # The plan reads its own budget here, so a smaller allowance keeps the same shards.
    monkeypatch.setattr(
        chunking,
        "joint_table_plan",
        lambda layout, _budget, make=joint_table_plan: make(layout, 1 << 26),
    )
    add_rows = _joint_tables.JointTables.add_rows
    blocks = []

    def recorded(self, start, stop, stack):
        blocks.append(stop - start)
        add_rows(self, start, stop, stack)

    monkeypatch.setattr(_joint_tables.JointTables, "add_rows", recorded)
    _geometry(problem, compiled=False, chunk=chunk)
    assert set(blocks) == {2 * chunk}


def test_channels_past_float64s_plain_range_keep_the_group_routes(monkeypatch):
    """A forced block and a real one. Mutation check: a gate that reads no channel sums
    weights the tables cannot hold in range."""
    problem = _problem(GaussianLS(scale_floor=0.02))
    groups = _geometry(problem, joint=False, monkeypatch=monkeypatch)
    monkeypatch.setattr(chunking, "stack_in_range", lambda *_args: False)
    _assert_same_geometry(groups, _geometry(problem, compiled=False))
    monkeypatch.undo()
    assert stack_in_range(np.ones((2, 5)), 5)
    assert stack_in_range(np.zeros((2, 5)), 5)
    for bad in (2.0**129, 2.0**-129, np.inf, np.nan):
        stack = np.ones((2, 5))
        stack[1, 3] = bad
        assert not stack_in_range(stack, 5)
        assert stack_in_range(stack, 3)


def test_a_design_without_one_index_per_group_keeps_its_group_routes():
    frame, rng = _frame(400)
    family = GaussianLS(scale_floor=0.02)
    weights = resolved_prior(np.ones(400))
    layout = build_stacked_layout(
        compile_predictors(
            frame,
            weights,
            family.parameters,
            (Predictor("location", {"x": Numeric(), "g": Categorical()}), Predictor("scale", {})),
            model_discrete=True,
            n_bins_config=20,
        )
    )
    assert joint_table_plan(layout, 1 << 26) is None


def test_tables_over_budget_keep_the_group_routes():
    layout = _problem(GaussianLS(scale_floor=0.02))[1]
    assert joint_table_plan(layout, 1 << 26) is not None
    assert joint_table_plan(layout, 16) is None


@pytest.mark.parametrize("zero_weights", (False, True), ids=("positive_weights", "zero_weights"))
def test_a_fit_reads_the_same_numbers_through_every_route(monkeypatch, zero_weights):
    """What a user reads: objective, EDF, prediction and standard errors, against the group
    and natural routes. Mutation check: drop a channel from one row kernel and this moves."""
    rng = np.random.default_rng(11)
    frame = pd.DataFrame(
        {
            "x": rng.choice(np.linspace(-1, 1, 25), N),
            "z": rng.choice(np.linspace(0, 3, 31), N),
            "g": rng.choice(["a", "b", "c", "d"], N),
            "h": rng.choice(["u", "v"], N),
        }
    )
    weights = rng.choice(np.linspace(0.1, 1.0, 10), N)
    if zero_weights:
        weights[::7] = 0.0
    positive = rng.random(N) < 0.15
    y = np.where(positive, rng.gamma(2.0, 1.5, N), 0.0)

    def fit():
        family = TweedieLSS()
        model = SuperLSS(
            family,
            family.mu(s("x", kind="cr", k=8), s("z", kind="cr", k=6), cat("g"), cat("h")),
            family.phi(s("x", kind="cr", k=5), cat("h")),
            family.p(),
            discrete=True,
        )
        model.fit_reml(frame, y, sample_weight=weights)
        return model

    routed = fit()
    monkeypatch.setattr(chunking, "_joint_plan", lambda *_args: None)
    reference = fit()
    assert routed.smoothing_convergence_reason_ == reference.smoothing_convergence_reason_
    assert routed.result_.total_effective_df == pytest.approx(
        reference.result_.total_effective_df, abs=1e-3
    )
    rows = frame.iloc[::37]
    np.testing.assert_allclose(
        np.asarray(routed.predict_link(rows)), np.asarray(reference.predict_link(rows)), atol=1e-4
    )
    np.testing.assert_allclose(routed.covariance_, reference.covariance_, rtol=5e-3, atol=1e-8)


def test_a_subclass_with_its_own_likelihood_reads_its_own_numbers():
    """The compiled rows evaluate ``TweedieLSS``'s law, which a subclass may have changed.
    Mutation check: serve any family that has the hook, and the score keeps its parent's."""

    class Halved(TweedieLSS):
        def evaluate_natural(self, y, theta, plan, *, derivative_order=2):
            evaluated = super().evaluate_natural(y, theta, plan, derivative_order=derivative_order)
            if derivative_order == 0:
                return replace(
                    evaluated,
                    optimizing_log_likelihood=evaluated.optimizing_log_likelihood / 2.0,
                )
            return replace(
                evaluated,
                optimizing_log_likelihood=evaluated.optimizing_log_likelihood / 2.0,
                score=evaluated.score / 2.0,
                hessian_packed=evaluated.hessian_packed / 2.0,
            )

    problem = _problem(TweedieLSS())
    family, layout, response, plan, coefficients = problem
    subclass = Halved()
    assert subclass.predictor_rows is not None
    whole_sums = [0.0, 0.0]
    whole = _geometry(problem, sums=whole_sums)
    halved_sums = [0.0, 0.0]
    halved = _geometry((subclass, layout, response, plan, coefficients), sums=halved_sums)
    assert halved_sums[0] == pytest.approx(whole_sums[0] / 2.0, rel=ROUND)
    np.testing.assert_allclose(halved.score_data, whole.score_data / 2.0, rtol=ROUND, atol=ROUND)
    value = chunking.evaluate_chunked_log_likelihood(
        subclass, layout, response, plan, coefficients, chunk_size="auto",
        pass_factors=PassFactors(),
    )  # fmt: skip
    assert value.optimizing_log_likelihood == pytest.approx(whole_sums[0] / 2.0, rel=ROUND)


def test_a_level_past_a_groups_table_keeps_the_group_routes():
    """The row passes index their tables without bounds checks. Mutation check: drop the
    index test, and this plan forms and reads outside its table."""
    family, layout, response, plan, coefficients = _problem(GaussianLS())
    assert joint_table_plan(layout, chunking.AUTO_CONTRACTION_MEMORY_BYTES) is not None
    group = next(
        group
        for group in layout.predictors[0].design.group_matrices
        if isinstance(group, CategoricalGroupMatrix)
    )
    held = group.codes
    group.codes = held.copy()
    group.codes[3] = group.n_levels + 1
    try:
        assert joint_table_plan(layout, chunking.AUTO_CONTRACTION_MEMORY_BYTES) is None
    finally:
        group.codes = held


def test_predictor_passes_read_what_the_chunks_read():
    """The compiled predictors, against the row-range route. Mutation check: an offset read for
    the step, whose change has none, or one group's table left out."""
    family, layout, response, plan, coefficients = _problem(TweedieLSS(), offsets=True)
    factors = PassFactors()
    change = chunking.maximum_chunked_predictor_change(
        layout, coefficients, chunk_size="auto", pass_factors=factors
    )
    assert change == pytest.approx(
        chunking.maximum_chunked_predictor_change(layout, coefficients, chunk_size="auto"),
        rel=1e-13,
    )
    eta, theta = chunking.materialize_terminal_predictions(
        layout, coefficients, chunk_size="auto", pass_factors=factors
    )
    reference_eta, reference_theta = chunking.materialize_terminal_predictions(
        layout, coefficients, chunk_size="auto"
    )
    np.testing.assert_allclose(eta, reference_eta, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(theta, reference_theta, rtol=1e-12)
    assert chunking.predictions_are(
        layout, coefficients, eta, theta, chunk_size="auto", pass_factors=factors
    )
    moved = eta.copy()
    moved[-1, 0] = np.nextafter(moved[-1, 0], np.inf)
    assert not chunking.predictions_are(
        layout, coefficients, moved, theta, chunk_size="auto", pass_factors=factors
    )
