"""Joint tables and compiled rows read the same numbers as the group routes and the natural route."""

from __future__ import annotations

import math
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
from superglm.distributional.families.tweedie import BoundedPowerLink, TweedieLSS
from superglm.distributional.family import COMPLETE_OBSERVATION
from superglm.distributional.kernels._common import _NumericalEvaluationError
from superglm.distributional.layout import build_stacked_layout
from superglm.distributional.predictor import Predictor, compile_predictors
from superglm.distributional.solver import _joint_tables
from superglm.distributional.solver import solver as solver_api
from superglm.distributional.solver._joint_tables import joint_table_plan, stack_in_range
from superglm.distributional.solver.derivatives import transform_natural_derivatives
from superglm.features import Categorical, Numeric, Spline
from superglm.links import LogLink

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


def _problem(family, offsets=False, power=None):
    """``power`` gives the last predictor rows that differ: an ``"offset"`` or a ``"term"``."""
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
    if power == "term":
        features.append({"x": Spline(kind="cr", n_knots=5, discrete=True)})
    predictors = tuple(
        Predictor(name, features[index] if index < len(features) else {})
        for index, name in enumerate(names)
    )
    offset_values = {names[0]: 0.05 * rng.normal(size=N)} if offsets else {}
    if power == "offset":
        offset_values[names[-1]] = 0.5 * rng.normal(size=N)
    layout = build_stacked_layout(
        compile_predictors(
            frame,
            weights,
            family.parameters,
            predictors,
            offsets=offset_values or None,
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


class _CountedLink:
    """The power link, counting each of its three evaluations."""

    def __init__(self, link):
        self._link = link
        self.lower, self.upper = link.lower, link.upper
        self.calls = 0

    def _counted(self, method, eta):
        self.calls += 1
        return getattr(self._link, method)(eta)

    def inverse(self, eta):
        return self._counted("inverse", eta)

    def deriv_inverse(self, eta):
        return self._counted("deriv_inverse", eta)

    def deriv2_inverse(self, eta):
        return self._counted("deriv2_inverse", eta)


@pytest.mark.parametrize("shared_power", (True, False), ids=("one_power", "row_powers"))
def test_the_shared_power_reaches_the_link_once_per_value(shared_power):
    """Four passes of four blocks, powers a, a, b, a: the link runs once per change of the shared
    power, and once per block where each row has its own. The rows equal a fresh evaluation.
    Mutation checks: never refresh the remembered power, or remember a power that rows differ in."""
    rng = np.random.default_rng(3)
    n, block = 400, 100
    y = np.where(rng.random(n) < 0.3, rng.gamma(2.0, 2.0, n), 0.0)
    weight = rng.random(n) + 0.1
    eta_mu, eta_phi = rng.normal(0.2, 1.0, n), rng.normal(0.1, 0.5, n)
    link = _resolved_link(TweedieLSS().parameters[2].default_link)
    counted = _CountedLink(link)
    rows = tweedie_kernels.tweedie_predictor_rows(y, weight, "prior", counted, shared_power)
    shared = {"a": 0.3, "b": -0.6}
    for pass_power in "aaba":
        eta_p = np.full(n, shared[pass_power]) if shared_power else rng.normal(0.0, 0.8, n)
        eta_p[::block] = eta_p[0]  # every block opens on one value, and its rows then part
        stack = np.empty((9, n))
        stack[:3] = eta_mu, eta_phi, eta_p
        fresh_stack = stack.copy()
        for first in range(0, n, block):
            fresh = tweedie_kernels.tweedie_predictor_rows(y, weight, "prior", link, shared_power)
            expected = fresh.evaluate(0, first, first + block, fresh_stack, 2)
            assert rows.evaluate(0, first, first + block, stack, 2) == expected
        np.testing.assert_array_equal(stack, fresh_stack)
    expected_evaluations = 3 if shared_power else 16
    assert counted.calls == 3 * expected_evaluations


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

    Mutation checks: a shard left out of the merge or of the totals, a shard's total that
    keeps only its last stack block, rows read at their stack column and not their row, a
    block's rows evaluated from row 0, and a stack block's columns counted from the first
    row and not the block's.
    """
    problem = _problem(make())
    group_sums = [0.0, 0.0]
    groups = _geometry(problem, joint=False, monkeypatch=monkeypatch, sums=group_sums)
    monkeypatch.setattr(_joint_tables, "_SHARD_ROWS", 257)
    monkeypatch.setattr(chunking, "STACK_BLOCK_ROWS", 31)
    monkeypatch.setattr(chunking, "contraction_block_rows", lambda _n, _bytes, chunk: chunk)
    assert joint_table_plan(problem[1], 1 << 26).n_shards > 1
    sums = [0.0, 0.0]
    tables = _geometry(problem, compiled=compiled, chunk=1000, sums=sums)
    _assert_same_geometry(groups, tables)
    assert abs(sums[0] - group_sums[0]) <= ROUND * abs(group_sums[0])
    major = tables.formed().majorant
    np.testing.assert_allclose(major, groups.formed().majorant, rtol=ROUND, atol=0.0)


@pytest.mark.parametrize("make", FAMILIES, ids=("gaussian", "gamma", "tweedie"))
def test_the_bits_do_not_depend_on_the_workers(monkeypatch, make):
    """Each shard sums its rows in row order, the shards merge in order, and the partition
    reads the rows and tables alone. Mutation check: shards sized by the worker count."""
    monkeypatch.setattr(_joint_tables, "_SHARD_ROWS", 257)
    problem = _problem(make())
    sums_one, sums_four = [0.0, 0.0], [0.0, 0.0]
    with parallel_config(n_jobs=1):
        one = _geometry(problem, sums=sums_one)
    with parallel_config(n_jobs=4):
        four = _geometry(problem, sums=sums_four)
    np.testing.assert_array_equal(one.data_curvature, four.data_curvature)
    np.testing.assert_array_equal(one.score_data, four.score_data)
    assert sums_one == sums_four


def test_the_shards_tables_take_at_most_half_the_allowance(monkeypatch):
    """Mutation check: no floor on the shards, and a pool of workers shares two of them."""
    monkeypatch.setattr(_joint_tables, "_SHARD_ROWS", 257)
    layout = _problem(GaussianLS(scale_floor=0.02))[1]
    roomy = joint_table_plan(layout, 1 << 26)
    assert roomy.n_shards == np.ceil(N / 257)
    assert not roomy.one_table_set
    exact = joint_table_plan(layout, 16 * roomy.table_bytes)
    assert not exact.one_table_set
    assert (exact.n_shards, exact.shard_rows) == (8, np.ceil(N / 8))
    for budget in (14 * roomy.table_bytes, roomy.table_bytes):
        tight = joint_table_plan(layout, budget)
        assert tight.one_table_set
        assert (tight.n_shards, tight.shard_rows) == (1, N)


def test_a_one_set_pass_buffers_what_its_tables_leave(monkeypatch):
    """Mutation check: a buffer that fills the allowance whatever the tables hold."""
    layout = _problem(GaussianLS(scale_floor=0.02))[1]
    roomy = joint_table_plan(layout, 1 << 26)
    assert roomy.block_rows == _joint_tables.TABLE_BLOCK_ROWS
    bytes_per_row = 8 * roomy.stack_rows
    left = _joint_tables.MORSEL_ROWS + 1000
    tight = joint_table_plan(layout, roomy.table_bytes + left * bytes_per_row)
    assert tight.block_rows == left
    assert joint_table_plan(layout, roomy.table_bytes).block_rows == _joint_tables.MORSEL_ROWS


@pytest.mark.parametrize("make", FAMILIES, ids=("gaussian", "gamma", "tweedie"))
@pytest.mark.parametrize("compiled", (True, False), ids=("compiled", "stacked"))
def test_one_set_of_tables_reads_what_the_group_routes_form(monkeypatch, make, compiled):
    """Morsels that split a buffer and buffers that split a chunk, on and off the pool.

    Mutation checks: a table's task left out, the totals left out, a buffer's rows read from
    row 0, a morsel's rows read from its first column, and a morsel left without its weights.
    """
    problem = _problem(make())
    groups = _geometry(problem, joint=False, monkeypatch=monkeypatch)
    monkeypatch.setattr(chunking, "MORSEL_ROWS", 31)
    monkeypatch.setattr(_joint_tables, "MORSEL_ROWS", 31)
    monkeypatch.setattr(_joint_tables, "TABLE_BLOCK_ROWS", 257)
    assert joint_table_plan(problem[1], 1 << 26).one_table_set
    for n_jobs in (1, 4):
        with parallel_config(n_jobs=n_jobs):
            tables = _geometry(problem, compiled=compiled, chunk=1000)
        _assert_same_geometry(groups, tables)
        major = tables.formed().majorant
        np.testing.assert_allclose(major, groups.formed().majorant, rtol=ROUND, atol=0.0)


@pytest.mark.parametrize("make", FAMILIES, ids=("gaussian", "gamma", "tweedie"))
def test_the_bits_of_one_set_of_tables_do_not_depend_on_the_workers(monkeypatch, make):
    """Each table sums its rows in row order, whichever worker has its task, and the likelihood
    adds its morsels in order. Mutation check: a morsel size read from the worker count."""
    monkeypatch.setattr(chunking, "MORSEL_ROWS", 31)
    monkeypatch.setattr(_joint_tables, "MORSEL_ROWS", 31)
    monkeypatch.setattr(_joint_tables, "TABLE_BLOCK_ROWS", 257)
    problem = _problem(make())
    sums_one, sums_four = [0.0, 0.0], [0.0, 0.0]
    with parallel_config(n_jobs=1):
        one = _geometry(problem, sums=sums_one)
    with parallel_config(n_jobs=4):
        four = _geometry(problem, sums=sums_four)
    np.testing.assert_array_equal(one.data_curvature, four.data_curvature)
    np.testing.assert_array_equal(one.score_data, four.score_data)
    np.testing.assert_array_equal(one.formed().majorant, four.formed().majorant)
    assert sums_one == sums_four


def test_a_one_set_pass_hands_the_pool_its_rows_and_its_tables(monkeypatch):
    """A compiled pass whose tables leave too few shards: work, not wall time. Every morsel and every
    table is its own pool item, so the row pass is not one task on one thread.
    Mutation check: a row pass of one item per buffer, or one task for all the tables."""
    monkeypatch.setattr(chunking, "MORSEL_ROWS", 512)
    monkeypatch.setattr(_joint_tables, "MORSEL_ROWS", 512)
    monkeypatch.setattr(_joint_tables, "TABLE_BLOCK_ROWS", 2048)
    problem = _problem(TweedieLSS())
    plan = joint_table_plan(problem[1], 1 << 26)
    assert plan.one_table_set
    sections = {"morsels": [], "tables": []}
    pooled_row_blocks, pooled_for = chunking.pooled_row_blocks, _joint_tables.pooled_for

    def blocks(fn, n_rows, block_rows):
        sections["morsels"].append(int(np.ceil(n_rows / block_rows)))
        return pooled_row_blocks(fn, n_rows, block_rows)

    def tasks(fn, items, item_bytes, **kwargs):
        sections["tables"].append(len(items))
        return pooled_for(fn, items, item_bytes, **kwargs)

    monkeypatch.setattr(chunking, "pooled_row_blocks", blocks)
    monkeypatch.setattr(_joint_tables, "pooled_for", tasks)
    _geometry(problem)
    assert sections["morsels"] == [4, 4, 4]
    assert sections["tables"] == [len(plan.tables) + 1] * 3


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


class _NaturalTweedie(TweedieLSS):
    """The same law through the natural route: compiled rows serve only the exact family."""


@pytest.mark.parametrize(
    ("family_type", "zero_weights", "shard_rows"),
    (
        (TweedieLSS, False, None),
        (TweedieLSS, False, 257),
        (TweedieLSS, True, None),
        (_NaturalTweedie, False, None),
    ),
    ids=("compiled", "compiled_shards", "zero_weights", "natural_into_tables"),
)
def test_a_fit_reads_the_same_numbers_through_every_route(
    monkeypatch, family_type, zero_weights, shard_rows
):
    """What a user reads, against the group and natural routes, through the solver, the fused first
    trial and the covariance passes. The search reaches the same EDF and predictions, whatever name
    rounding gives its stop on a flat direction; at one set of smoothing parameters, which leaves
    no such direction to drift along, EDF, predictions and covariance agree to a generous multiple
    of rounding on the book's scale.

    Mutation checks: every joint block scaled by 1.5, one row kernel's channel dropped, a block
    off by 1e-7, and a route that stops being taken, which the dispatch counts catch.
    """
    rng = np.random.default_rng(11)
    frame = pd.DataFrame(
        {
            "x": rng.uniform(-1, 1, N),
            "z": rng.uniform(0, 3, N),
            "g": rng.choice(["a", "b", "c", "d"], N),
            "h": rng.choice(["u", "v"], N),
        }
    )
    weights = rng.choice(np.linspace(0.1, 1.0, 10), N)
    if zero_weights:
        weights[::7] = 0.0
    positive = rng.random(N) < 0.15
    y = np.where(positive, rng.gamma(2.0, 1.5, N), 0.0)

    def build():
        family = family_type()
        return SuperLSS(
            family,
            family.mu(s("x", kind="cr", k=8), s("z", kind="cr", k=6), cat("g"), cat("h")),
            family.phi(s("x", kind="cr", k=5), cat("h")),
            family.p(),
            discrete=True,
        )

    counts = {"compiled": 0, "tables": 0, "shards": 0}
    compiled_geometry = chunking._compiled_geometry
    add_rows = _joint_tables.JointTables.add_rows
    add_block = _joint_tables.JointTables.add_block

    def counted_compiled(*args):
        counts["compiled"] += 1
        return compiled_geometry(*args)

    def counted_rows(self, *args):
        counts["tables"] += 1
        return add_rows(self, *args)

    def counted_block(self, *args):
        counts["shards"] += 1
        return add_block(self, *args)

    monkeypatch.setattr(chunking, "_compiled_geometry", counted_compiled)
    monkeypatch.setattr(_joint_tables.JointTables, "add_rows", counted_rows)
    monkeypatch.setattr(_joint_tables.JointTables, "add_block", counted_block)
    if shard_rows is not None:
        monkeypatch.setattr(_joint_tables, "_SHARD_ROWS", shard_rows)

    routed = build().fit_reml(frame, y, sample_weight=weights)
    lambdas = dict(routed._model.fit_state.smoothing.lambdas)
    fixed = build().fit(frame, y, sample_weight=weights, lambdas=lambdas)
    if family_type is TweedieLSS:
        # A row of zero weight has left the fit before these routes.
        assert counts["compiled"] > 0
    else:
        assert counts["compiled"] == 0 and counts["tables"] > 0
    assert (counts["shards"] > 0) == (shard_rows is not None)
    monkeypatch.undo()
    monkeypatch.setattr(chunking, "_joint_plan", lambda *_args: None)
    reference = build().fit_reml(frame, y, sample_weight=weights)
    fixed_reference = build().fit(frame, y, sample_weight=weights, lambdas=lambdas)
    # Two searches stop within the search's own tolerance of each other, on any platform's rounding.
    assert routed.result_.total_effective_df == pytest.approx(
        reference.result_.total_effective_df, abs=1e-4
    )
    rows = frame.iloc[::37]
    np.testing.assert_allclose(
        np.asarray(routed.predict_link(rows)), np.asarray(reference.predict_link(rows)), atol=1e-4
    )
    assert fixed.result_.total_effective_df == pytest.approx(
        fixed_reference.result_.total_effective_df, abs=1e-9
    )
    np.testing.assert_allclose(
        np.asarray(fixed.predict_link(rows)),
        np.asarray(fixed_reference.predict_link(rows)),
        rtol=1e-9,
        atol=1e-9,
    )
    variance = np.diag(fixed_reference.covariance_)
    scale = np.sqrt(np.outer(variance, variance))
    assert np.max(np.abs(fixed.covariance_ - fixed_reference.covariance_) / scale) <= 1e-9


@pytest.mark.parametrize("power", ("offset", "term"))
def test_rows_that_differ_in_their_power_read_their_own(power):
    """An offset or a term on the power, against the natural route. Mutation checks: a plan that
    calls the power shared with the offset clause dropped, or with no term test, reads every
    row of a block at its first row's power."""
    problem = _problem(TweedieLSS(), power=power)
    family, layout, response, plan, coefficients = problem
    factors = PassFactors()
    assert not chunking._joint_plan(layout, factors).shares_last_predictor
    sums = [0.0, 0.0]
    compiled = _geometry(problem, factors=factors, sums=sums)
    assert chunking._compiled_rows(
        family, layout, response, plan, chunking._joint_plan(layout, factors), factors
    )
    natural_sums = [0.0, 0.0]
    natural = _geometry(problem, compiled=False, sums=natural_sums)
    _assert_same_geometry(natural, compiled)
    assert abs(sums[0] - natural_sums[0]) <= ROUND * abs(natural_sums[0])
    value = chunking.evaluate_chunked_log_likelihood(
        family, layout, response, plan, coefficients, chunk_size="auto", pass_factors=factors
    )
    assert value.optimizing_log_likelihood == pytest.approx(natural_sums[0], rel=ROUND)


def test_one_session_keeps_a_layouts_own_plan_and_rows():
    """A second layout on the same designs, with an offset on the power, shares no plan or rows
    with the first. Mutation checks: a plan key without the offsets, a rows key without the plan."""
    problem = _problem(TweedieLSS())
    family, layout, response, plan, coefficients = problem
    factors = PassFactors()
    _geometry(problem, factors=factors)
    rng = np.random.default_rng(5)
    moved = replace(
        layout,
        predictors=(
            *layout.predictors[:2],
            replace(layout.predictors[2], offset=0.5 * rng.normal(size=N)),
        ),
    )
    shifted = (family, moved, response, plan, coefficients)
    assert not chunking._joint_plan(moved, factors).shares_last_predictor
    assert chunking._joint_plan(layout, factors).shares_last_predictor
    _assert_same_geometry(_geometry(shifted, compiled=False), _geometry(shifted, factors=factors))


def test_a_fit_keeps_one_plan_and_one_response_however_many_inner_fits_it_takes(monkeypatch):
    """Counted over a search of several coefficient fits: each session holds one joint plan, one
    set of compiled rows and one response. Mutation check: contexts that keep their own copy of
    the response, so every coefficient fit adds a response and rows to the session."""
    rng = np.random.default_rng(11)
    frame = pd.DataFrame({"x": rng.uniform(-1, 1, N), "h": rng.choice(["u", "v"], N)})
    y = np.where(rng.random(N) < 0.15, rng.gamma(2.0, 1.5, N), 0.0)
    sessions: list[PassFactors] = []
    contexts = [0]
    built = [0]
    init = PassFactors.__init__
    validated = solver_api._validated_context
    tweedie_rows = tweedie_kernels.tweedie_predictor_rows

    def recorded_init(self):
        init(self)
        sessions.append(self)

    def counted_context(*args, **kwargs):
        contexts[0] += 1
        return validated(*args, **kwargs)

    def counted_rows(*args):
        built[0] += 1
        return tweedie_rows(*args)

    monkeypatch.setattr(PassFactors, "__init__", recorded_init)
    monkeypatch.setattr(solver_api, "_validated_context", counted_context)
    monkeypatch.setattr(
        "superglm.distributional.families.tweedie.tweedie_predictor_rows", counted_rows
    )
    family = TweedieLSS()
    SuperLSS(
        family,
        family.mu(s("x", kind="cr", k=8), cat("h")),
        family.phi(s("x", kind="cr", k=5)),
        family.p(),
        discrete=True,
    ).fit_reml(frame, y)
    serving = [
        session
        for session in sessions
        if any(key and key[0] == "rows" for key in session.joint_plans)
    ]
    assert contexts[0] > len(serving) >= 1
    assert built[0] == len(serving)
    for session in serving:
        rows_entries = [e for key, e in session.joint_plans.items() if key and key[0] == "rows"]
        assert len(rows_entries) == 1
        assert len(session.joint_plans) == 2


def test_a_declined_pass_holds_no_tables_through_its_fallback(monkeypatch):
    """Two sets of shard tables are never alive together. Mutation check: a decline that keeps the
    declined pass's tables while the natural pass allocates its own."""
    problem = _problem(TweedieLSS())
    family, layout, response, plan, coefficients = problem
    coefficients = coefficients.copy()
    coefficients[layout.predictors[0].intercept_index] = 101.0
    created: list = []
    live_at_creation: list[int] = []
    init = _joint_tables.JointTables.__init__

    def recorded(self, joint):
        live_at_creation.append(sum(tables._shards.size for tables in created))
        init(self, joint)
        created.append(self)

    monkeypatch.setattr(_joint_tables.JointTables, "__init__", recorded)
    try:
        _geometry((family, layout, response, plan, coefficients))
    except ValueError:
        pass
    assert len(created) == 2
    assert live_at_creation == [0, 0]


def _intercepts(y, weights, family=None, power_link=None, *, kind="prior"):
    family = TweedieLSS() if family is None else family
    n = len(y)
    frame = as_eager_frame(pd.DataFrame({"unused": np.zeros(n)}))
    resolved = resolved_prior(weights)
    parameters = family.parameters
    if power_link is not None:
        parameters = (*parameters[:2], replace(parameters[2], default_link=power_link))
    layout = build_stacked_layout(
        compile_predictors(
            frame,
            resolved,
            parameters,
            tuple(Predictor(spec.name, {}) for spec in parameters),
            model_discrete=True,
            n_bins_config=20,
        )
    )
    return family, layout, y, family.bind_likelihood(y, resolved, COMPLETE_OBSERVATION)


def _outcome(call):
    """What a caller sees: the error's type and message, or the numbers."""
    try:
        return "value", call()
    except ValueError as error:
        return type(error), str(error)


@pytest.mark.parametrize(
    "coefficients",
    [
        (0.0, -746.0, -10.0),
        (0.0, -740.0, -10.0),
        (0.0, 710.0, -10.0),
        (-746.0, 0.0, -10.0),
        (-740.0, 0.0, -10.0),
        (710.0, 0.0, -10.0),
    ],
    ids=(
        "phi_zero",
        "phi_subnormal",
        "phi_infinite",
        "mean_zero",
        "mean_subnormal",
        "mean_infinite",
    ),
)
def test_a_positive_row_past_the_normal_range_gets_the_natural_routes_decision(coefficients):
    """A mean or dispersion that leaves float64's normal range, at both derivative orders: the same
    refusal, or the same numbers, as the natural route, and never a ``ZeroDivisionError`` a trial's
    backtracking would not catch. Mutation check: drop the test from the row kernel."""
    y = np.full(3, 1e-300)
    family, layout, response, plan = _intercepts(y, np.full(3, math.exp(-100.0)))
    values = np.array(coefficients)
    problem = (family, layout, response, plan, values)

    def value(factors):
        return chunking.evaluate_chunked_log_likelihood(
            family, layout, response, plan, values, chunk_size="auto", pass_factors=factors
        ).optimizing_log_likelihood

    with np.errstate(all="ignore"):
        assert _outcome(lambda: value(PassFactors())) == _outcome(lambda: value(None))
        kind, compiled = _outcome(lambda: _geometry(problem))
        reference_kind, natural = _outcome(lambda: _geometry(problem, compiled=False))
    assert kind == reference_kind
    if kind == "value":
        np.testing.assert_array_equal(compiled.data_curvature, natural.data_curvature)
    else:
        assert compiled == natural


def test_an_underflowed_dispersion_is_refused_on_the_fit_as_on_the_natural_route(monkeypatch):
    """From an initial state a trial could reach: a clean ``ValueError``, not a ``ZeroDivisionError``."""
    y = np.full(3, 1e-300)
    family, layout, response, plan = _intercepts(y, np.full(3, math.exp(-100.0)))
    initial = np.array([0.0, -746.0, -10.0])

    def fit():
        return solver_api.fit_dense_fixed_lambda(
            family, layout, response, plan, np.zeros((3, 3)), initial=initial, chunk_size="auto"
        )

    with np.errstate(all="ignore"), pytest.raises(ValueError) as compiled:
        fit()
    monkeypatch.setattr(chunking, "_joint_plan", lambda *_args: None)
    with np.errstate(all="ignore"), pytest.raises(ValueError) as natural:
        fit()
    assert str(compiled.value) == str(natural.value)


def _cancelling_problem():
    g = np.array(list("aabbaabb"))
    h = np.array(list("uvuvuvuv"))
    frame = as_eager_frame(pd.DataFrame({"g": g, "h": h}))
    family = GaussianLS(scale_floor=0.0)
    weights = resolved_prior(np.ones(8))
    layout = build_stacked_layout(
        compile_predictors(
            frame,
            weights,
            family.parameters,
            (
                Predictor("location", {"g": Categorical(), "h": Categorical()}),
                Predictor("scale", {}),
            ),
            offsets={"location": np.where(g == "a", -(2.0**54), 0.0)},
            model_discrete=True,
            n_bins_config=20,
        )
    )
    y = np.array([-1.0, 1.0] * 4)
    plan = family.bind_likelihood(y, weights, COMPLETE_OBSERVATION)
    return family, layout, y, plan, np.array([2.0**54, -(2.0**54), 1.0, 0.0])


def test_predictors_that_cancel_sum_as_the_row_route_and_prediction_do():
    """The groups first and then the intercept: a term of 1 next to 2**54 rounds away unless the
    intercept cancels first, so the order decides the published mean. Mutation check: start each
    predictor from its intercept, and the last two rows publish a mean of 1."""
    family, layout, y, plan, coefficients = _cancelling_problem()
    eta, theta = chunking.materialize_terminal_predictions(
        layout, coefficients, chunk_size="auto", pass_factors=PassFactors()
    )
    reference_eta, reference_theta = chunking.materialize_terminal_predictions(
        layout, coefficients, chunk_size="auto"
    )
    np.testing.assert_array_equal(eta, reference_eta)
    np.testing.assert_array_equal(theta, reference_theta)
    fitted = solver_api.fit_dense_fixed_lambda(
        family, layout, y, plan, np.zeros((4, 4)), initial=coefficients, chunk_size="auto"
    )
    assert np.asarray(fitted.log_likelihood) == pytest.approx(
        float(
            np.sum(
                -0.5 * np.log(2 * np.pi)
                - np.log(fitted.theta[:, 1])
                - 0.5 * ((y - fitted.theta[:, 0]) / fitted.theta[:, 1]) ** 2
            )
        ),
        rel=1e-14,
    )


def test_a_spline_predictor_multiplies_its_factors_as_the_row_route_and_prediction_do():
    """``B @ (R_inv @ beta)`` and ``(B @ R_inv) @ beta`` differ when large slopes cancel against an
    offset, so the association decides the published mean and whether the objective read it.
    Mutation check: read ``B @ R_inv`` here, and the means reach 32 where the row route's are 0."""
    x = np.repeat(np.random.default_rng(109).uniform(-1, 1, 80), 2)
    frame = as_eager_frame(pd.DataFrame({"x": x}))
    family = GaussianLS(scale_floor=0.0)
    weights = resolved_prior(np.ones(x.size))

    def layout_with(offsets):
        return build_stacked_layout(
            compile_predictors(
                frame,
                weights,
                family.parameters,
                (
                    Predictor("location", {"x": Spline(kind="cr", n_knots=4, discrete=True)}),
                    Predictor("scale", {}),
                ),
                offsets=offsets,
                model_discrete=True,
                n_bins_config=20,
            )
        )

    design = layout_with(None).predictors[0].design
    slopes = 2.0**54 * np.arange(1.0, design.p + 1.0)
    layout = layout_with({"location": -design.matvec(slopes)})
    coefficients = np.concatenate(([0.0], slopes, [0.0]))
    eta, theta = chunking.materialize_terminal_predictions(
        layout, coefficients, chunk_size="auto", pass_factors=PassFactors()
    )
    reference_eta, reference_theta = chunking.materialize_terminal_predictions(
        layout, coefficients, chunk_size="auto"
    )
    np.testing.assert_array_equal(eta, reference_eta)
    np.testing.assert_array_equal(theta, reference_theta)


@pytest.mark.parametrize(
    "read",
    [
        lambda layout, values, factors: chunking.materialize_terminal_predictions(
            layout, values, chunk_size="auto", pass_factors=factors
        ),
        lambda layout, values, factors: chunking.predictions_are(
            layout,
            values,
            np.zeros((N, len(layout.predictors))),
            np.zeros((N, len(layout.predictors))),
            chunk_size="auto",
            pass_factors=factors,
        ),
        lambda layout, values, factors: chunking.maximum_chunked_predictor_change(
            layout, values, chunk_size="auto", pass_factors=factors
        ),
    ],
    ids=("terminal", "same", "change"),
)
@pytest.mark.parametrize("predictor", (0, 2), ids=("mean", "power"))
def test_a_non_finite_predictor_is_refused_on_both_routes(read, predictor):
    """Finite coefficients whose sum overflows, where a bounded link would turn the infinite power
    predictor into a finite wall value that a user would read. A group's table overflows on the
    mean, and the power's intercept and offset do on the power. Mutation check: drop the test
    from the joint branch."""
    _family, layout, _response, _plan, coefficients = _problem(TweedieLSS(), power="term")
    coefficients = coefficients.copy()
    coefficients[layout.predictors[predictor].coefficient_slice] = 1.7e308
    for factors in (PassFactors(), None):
        with pytest.raises(ValueError, match="non-finite"):
            read(layout, coefficients, factors)


def test_a_link_with_walls_past_the_familys_keeps_the_natural_route():
    """The compiled rows refuse a power at the link's walls and the family at its own, so a trial
    between them would get a number from one route and a refusal from the other. A narrower link
    is safe: its wall is inside the family's, and the rows decline there.
    Mutation check: drop the walls' test from the hook."""
    family = TweedieLSS(power_lower=1.3, power_upper=1.6)
    y = np.array([0.0, 0.0, 2.0, 1.0])
    weights = np.ones(4)
    plan = family.bind_likelihood(y, resolved_prior(weights), COMPLETE_OBSERVATION)

    def rows(lower, upper):
        links = (LogLink(), LogLink(), BoundedPowerLink(lower, upper))
        return family.predictor_rows(y, plan, links, True)

    assert rows(1.3, 1.6) is not None
    assert rows(1.35, 1.55) is not None
    assert rows(1.3 - 1e-10, 1.6) is None
    assert rows(1.3, 1.6 + 1e-10) is None


def test_the_first_trial_owns_its_geometry_only_where_compiled_rows_serve():
    """An unbinned Tweedie fit, and one whose weights the rows do not take, keep their value pass.
    Mutation check: read whether the hook exists, not whether its rows serve this fit."""

    def eligible(problem):
        family, layout, response, plan, _coefficients = problem
        context = solver_api._validated_context(
            family, layout, response, plan, np.eye(layout.n_coefficients) * 0.1,
            coefficient_curvature="observed", chunk_size="auto", coefficient_face=None,
        )  # fmt: skip
        return solver_api._fused_first_trial_eligible(context)

    problem = _problem(TweedieLSS())
    assert eligible(problem)
    family, layout, response, _plan, coefficients = problem
    weights = np.ones(N)
    weights[::7] = 1e-50
    past = family.bind_likelihood(response, resolved_prior(weights), COMPLETE_OBSERVATION)
    assert not eligible((family, layout, response, past, coefficients))
    frame, rng = _frame()
    unbinned = build_stacked_layout(
        compile_predictors(
            frame,
            resolved_prior(np.ones(N)),
            family.parameters,
            tuple(
                Predictor(spec.name, {"x": Numeric()} if index == 0 else {})
                for index, spec in enumerate(family.parameters)
            ),
            model_discrete=False,
        )
    )
    plain = family.bind_likelihood(response, resolved_prior(np.ones(N)), COMPLETE_OBSERVATION)
    assert not eligible((family, unbinned, response, plain, np.zeros(unbinned.n_coefficients)))


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


def test_a_predictor_change_with_a_nan_in_any_block_is_refused(monkeypatch):
    """Mutation check: take the largest block by Python's ``max``, which keeps its first
    operand against a NaN, and the NaN of the later block goes unreported."""
    _family, layout, _response, _plan, coefficients = _problem(GaussianLS())
    monkeypatch.setattr(chunking, "_joint_predictor_rows", lambda *_a, **_k: [1.0, float("nan")])
    with pytest.raises(ValueError, match="non-finite"):
        chunking.maximum_chunked_predictor_change(
            layout, coefficients, chunk_size="auto", pass_factors=PassFactors()
        )


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
    np.testing.assert_array_equal(eta, reference_eta)
    np.testing.assert_array_equal(theta, reference_theta)
    assert chunking.predictions_are(
        layout, coefficients, eta, theta, chunk_size="auto", pass_factors=factors
    )
    moved = eta.copy()
    moved[-1, 0] = np.nextafter(moved[-1, 0], np.inf)
    assert not chunking.predictions_are(
        layout, coefficients, moved, theta, chunk_size="auto", pass_factors=factors
    )
