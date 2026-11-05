"""Value-only chunk work and numerical correctness are separate contracts."""

from __future__ import annotations

import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import superglm.distributional.solver.chunks as chunking
from superglm import (
    GammaLS,
    GeneralizedGammaLSS,
    GeneralizedParetoLSS,
    LogNormalLS,
    NegativeBinomialLS,
    TweedieLSS,
    TwoPieceLogNormalLSS,
    TwoPieceNormalLSS,
    _parallel,
)
from superglm._frame import as_eager_frame
from superglm._parallel import parallel_config, resolve_n_jobs
from superglm.distributional.families.gaussian import GaussianLS
from superglm.distributional.family import COMPLETE_OBSERVATION
from superglm.distributional.layout import build_stacked_layout
from superglm.distributional.predictor import Predictor, compile_predictors
from superglm.features import Numeric, Spline
from superglm.group_matrix import DesignMatrix

from ._distributional_weights import resolved_prior


def _problem(*, discrete=True, scale_intercept=True):
    n = 23
    x = np.linspace(-0.9, 1.1, n)
    frame = as_eager_frame(pd.DataFrame({"x": x, "z": np.sin(2.3 * x)}))
    family = GaussianLS(scale_floor=0.02)
    weights = resolved_prior(np.linspace(0.6, 1.8, n))
    layout = build_stacked_layout(
        compile_predictors(
            frame,
            weights,
            family.parameters,
            (
                Predictor(
                    "location",
                    {"x": Numeric(), "z": Spline(kind="cr", n_knots=4, discrete=discrete)},
                ),
                Predictor("scale", {"x": Numeric()}, intercept=scale_intercept),
            ),
            offsets={"location": 0.03 * x**2, "scale": -0.1 + 0.02 * x},
            model_discrete=discrete,
            n_bins_config=7,
        )
    )
    coefficients = np.linspace(-0.2, 0.3, layout.n_coefficients)
    response = 0.4 + np.cos(x)
    plan = family.bind_likelihood(response, weights, COMPLETE_OBSERVATION)
    return family, layout, response, plan, coefficients


def _evaluate(operation, problem, chunk_size):
    family, layout, response, plan, coefficients = problem
    if operation == "likelihood":
        return chunking.evaluate_chunked_log_likelihood(
            family, layout, response, plan, coefficients, chunk_size=chunk_size
        )
    if operation == "change":
        return chunking.maximum_chunked_predictor_change(
            layout, coefficients, chunk_size=chunk_size
        )
    return chunking.materialize_terminal_predictions(layout, coefficients, chunk_size=chunk_size)


@pytest.mark.parametrize("operation", ["likelihood", "change", "terminal"])
def test_value_only_passes_do_not_prepare_geometry(monkeypatch, operation):
    """Reintroducing chunk DesignMatrix/geometry construction breaks this witness."""
    problem = _problem()
    design_subsets = 0
    execution_plans = 0
    original_subset = DesignMatrix.row_subset
    original_plan = chunking.PredictorExecutionPlan

    def counted_subset(self, indices):
        nonlocal design_subsets
        design_subsets += 1
        return original_subset(self, indices)

    def counted_plan(*args, **kwargs):
        nonlocal execution_plans
        execution_plans += 1
        return original_plan(*args, **kwargs)

    monkeypatch.setattr(DesignMatrix, "row_subset", counted_subset)
    monkeypatch.setattr(chunking, "PredictorExecutionPlan", counted_plan)
    _evaluate(operation, problem, 7)
    assert (design_subsets, execution_plans) == (0, 0)


@pytest.mark.parametrize("operation", ["likelihood", "change", "terminal"])
def test_value_only_group_subsets_are_bounded_and_not_retained(monkeypatch, operation):
    # Refused range evaluation must retain the bounded generic path.
    monkeypatch.setattr(chunking, "group_range_matvec", lambda *args: None, raising=False)
    problem = _problem()
    references = []
    largest_subset = 0
    most_live = 0
    group_types = {
        type(group) for state in problem[1].predictors for group in state.design.group_matrices
    }

    def tracked_subset(original):
        def subset(self, indices):
            nonlocal largest_subset, most_live
            result = original(self, indices)
            largest_subset = max(largest_subset, result.shape[0])
            storage = result.bin_idx if hasattr(result, "bin_idx") else result.M
            references.append(weakref.ref(storage))
            most_live = max(most_live, sum(ref() is not None for ref in references))
            return result

        return subset

    for group_type in group_types:
        monkeypatch.setattr(group_type, "row_subset", tracked_subset(group_type.row_subset))
    _evaluate(operation, problem, 7)
    assert largest_subset == 7
    assert most_live <= 2
    assert all(ref() is None for ref in references)


@pytest.mark.parametrize("operation", ["likelihood", "change", "terminal"])
def test_value_only_ordinary_ranges_do_not_construct_group_subsets(monkeypatch, operation):
    problem = _problem()

    def forbidden_subset(*args):
        pytest.fail("ordinary contiguous values rebuilt a group subset")

    for group_type in {
        type(group) for state in problem[1].predictors for group in state.design.group_matrices
    }:
        monkeypatch.setattr(group_type, "row_subset", forbidden_subset)
    _evaluate(operation, problem, 7)


@pytest.mark.parametrize("include_offsets", [False, True])
def test_geometry_ranges_preserve_stored_values_without_generic_subsets(
    monkeypatch, include_offsets
):
    _, layout, _, _, coefficients = _problem()
    rows = next(chunking.iter_row_chunks(23, 7))
    expected, _ = chunking._predictor_chunk(
        layout, coefficients, rows, include_offsets=include_offsets
    )

    def forbidden_subset(*args):
        pytest.fail("ordinary contiguous geometry used generic row indexing")

    monkeypatch.setattr(DesignMatrix, "row_subset", forbidden_subset)
    for group_type in {
        type(group) for state in layout.predictors for group in state.design.group_matrices
    }:
        monkeypatch.setattr(group_type, "row_subset", forbidden_subset)
    actual, plans = chunking._predictor_chunk(
        layout, coefficients, rows, include_offsets=include_offsets
    )
    tolerance = 16 * layout.n_coefficients * np.finfo(float).eps
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
    assert all(plan.design.n == 7 for plan in plans)


@pytest.mark.parametrize("chunk_size", [1, 7, 23, 31])
@pytest.mark.parametrize("discrete", [False, True])
@pytest.mark.parametrize("scale_intercept", [False, True])
def test_chunk_values_match_stored_design_with_offsets(chunk_size, discrete, scale_intercept):
    problem = _problem(discrete=discrete, scale_intercept=scale_intercept)
    family, layout, response, plan, coefficients = problem
    expected_change = np.empty((len(response), len(layout.predictors)))
    expected_eta = np.empty_like(expected_change)
    expected_theta = np.empty_like(expected_change)
    for state in layout.predictors:
        local = coefficients[state.coefficient_slice]
        intercept = state.intercept_index is not None
        # The stored design includes the actual discretised training basis.
        values = state.design.toarray() @ local[int(intercept) :]
        if intercept:
            values += local[0]
        expected_change[:, state.parameter_index] = values
        expected_eta[:, state.parameter_index] = values + state.offset
        expected_theta[:, state.parameter_index] = state.link.inverse(values + state.offset)
    eta, theta = _evaluate("terminal", problem, chunk_size)
    # Dot products and chunk reductions have dimension-scaled roundoff bounds.
    tolerance = 16 * max(len(response), layout.n_coefficients) * np.finfo(float).eps
    np.testing.assert_allclose(eta, expected_eta, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(theta, expected_theta, rtol=tolerance, atol=tolerance)
    assert not eta.flags.writeable
    assert not theta.flags.writeable
    np.testing.assert_allclose(
        _evaluate("change", problem, chunk_size),
        np.max(np.abs(expected_change)),
        rtol=tolerance,
        atol=tolerance,
    )
    expected = family.evaluate_natural(response, expected_theta, plan, derivative_order=0)
    actual = _evaluate("likelihood", problem, chunk_size)
    np.testing.assert_allclose(
        actual.optimizing_log_likelihood,
        np.sum(expected.optimizing_log_likelihood),
        rtol=tolerance,
        atol=tolerance,
    )
    np.testing.assert_allclose(
        actual.parameter_independent_carrier,
        np.sum(expected.parameter_independent_carrier),
        rtol=tolerance,
        atol=tolerance,
    )


_ARRAY_FAMILIES = (
    GaussianLS,
    GammaLS,
    LogNormalLS,
    GeneralizedParetoLSS,
    NegativeBinomialLS,
    TwoPieceNormalLSS,
    TwoPieceLogNormalLSS,
)


def _family_problem(family_type):
    n = 400
    x = np.linspace(-0.9, 1.1, n)
    frame = as_eager_frame(pd.DataFrame({"x": x}))
    family = family_type()
    weights = resolved_prior(np.ones(n))
    layout = build_stacked_layout(
        compile_predictors(
            frame,
            weights,
            family.parameters,
            tuple(Predictor(spec.name, {"x": Numeric()}) for spec in family.parameters),
        )
    )
    response = 0.5 + np.exp(np.sin(3.0 * x))
    if family_type is NegativeBinomialLS:
        response = np.round(response)
    plan = family.bind_likelihood(response, weights, COMPLETE_OBSERVATION)
    return family, layout, response, plan, np.full(layout.n_coefficients, 0.05)


@pytest.mark.parametrize("family_type", _ARRAY_FAMILIES, ids=lambda family: family.__name__)
def test_row_chunks_on_the_pool_keep_their_bits_and_their_order(monkeypatch, family_type):
    """Two chunks must be inside the kernels at once. Mutation check: a stream that never pools
    breaks the barrier."""
    monkeypatch.setattr(chunking, "_MIN_POOLED_CHUNK_ROWS", 1)
    problem = _family_problem(family_type)
    meetings = []
    evaluate_natural = family_type.evaluate_natural

    def recorded(self, *args, **kwargs):
        if meetings:
            meetings.pop().wait(timeout=30)
        return evaluate_natural(self, *args, **kwargs)

    monkeypatch.setattr(family_type, "evaluate_natural", recorded)

    def stream():
        chunks = list(
            chunking.iter_likelihood_chunks(*problem, chunk_size=32, curvature_source="observed")
        )
        likelihood = chunking.evaluate_chunked_log_likelihood(*problem, chunk_size=32)
        return chunks, likelihood

    with parallel_config(n_jobs=1):
        serial, serial_likelihood = stream()
    barrier = threading.Barrier(2)
    meetings.extend((barrier, barrier))
    with parallel_config(n_jobs=4):
        pooled, pooled_likelihood = stream()
    assert not meetings
    assert [chunk.rows.start for chunk in pooled] == [chunk.rows.start for chunk in serial]
    for left, right in zip(serial, pooled, strict=True):
        np.testing.assert_array_equal(left.score_eta, right.score_eta)
        np.testing.assert_array_equal(left.curvature_packed, right.curvature_packed)
        np.testing.assert_array_equal(
            left.optimizing_log_likelihood, right.optimizing_log_likelihood
        )
    assert pooled_likelihood == serial_likelihood


def test_a_failure_on_the_pool_names_the_first_failing_chunk(monkeypatch):
    """The search recovers from a refused trial, so which chunk refused must not depend on timing."""
    monkeypatch.setattr(chunking, "_MIN_POOLED_CHUNK_ROWS", 1)
    problem = _family_problem(GammaLS)
    evaluate_natural = GammaLS.evaluate_natural

    def refuses(self, y, theta, plan, **kwargs):
        if plan.weights.input_positions[0] >= 64:
            raise FloatingPointError("overflow in a later chunk")
        return evaluate_natural(self, y, theta, plan, **kwargs)

    monkeypatch.setattr(GammaLS, "evaluate_natural", refuses)
    with parallel_config(n_jobs=4), pytest.raises(chunking._TrialDerivativeError) as refused:
        list(
            chunking.iter_likelihood_chunks(
                *problem,
                chunk_size=32,
                curvature_source="observed",
                _recover_derivative_failure=True,
            )
        )
    assert refused.value.rows.start == 64


class _FamilyFromOutsideThePackage(GammaLS):
    """A subclass can change ``evaluate_natural``, which never promised to be thread-safe."""


@pytest.mark.parametrize(
    ("family", "chunk_rows", "pooled"),
    [
        (TweedieLSS(), 8192, False),
        (GammaLS(), 8192, True),
        (GammaLS(), 16, False),
        (_FamilyFromOutsideThePackage(), 8192, False),
    ],
    ids=["family-pools-its-rows", "array-kernels", "short-chunks", "family-from-outside"],
)
def test_the_stream_pools_only_where_it_repays(monkeypatch, family, chunk_rows, pooled):
    """Results come back in row order, a window at a time, on at most the array kernels' workers."""
    monkeypatch.setattr(chunking, "AUTO_CONTRACTION_MEMORY_BYTES", 16 * 8 * chunk_rows)
    sections = []
    pooled_for = chunking.pooled_for

    def spy(fn, items, item_bytes, **kwargs):
        sections.append((len(items), kwargs["max_workers"]))
        return pooled_for(fn, items, item_bytes, **kwargs)

    monkeypatch.setattr(chunking, "pooled_for", spy)
    with parallel_config(n_jobs=64):
        starts = list(
            chunking._evaluated_chunks(
                family, lambda rows: rows.start, 10 * chunk_rows, chunk_rows, held_floats=8
            )
        )
    assert starts == list(range(0, 10 * chunk_rows, chunk_rows))
    if pooled:
        assert sections == [(2, chunking._ARRAY_CHUNK_WORKERS)] * 5
    else:
        assert not sections


def _helpers_started(monkeypatch):
    started = []
    grow = _parallel._Crew.grow

    def recording(self, count):
        started.append(count)
        grow(self, count)

    monkeypatch.setattr(_parallel._Crew, "grow", recording)
    return started


def test_the_stream_keeps_to_numbas_thread_count(monkeypatch):
    """SuperLSS has no ``n_jobs``, so Numba's count is its thread setting. Mutation check: a stream
    that sets its own ``n_jobs`` skips the cap and starts four helpers here."""
    import numba

    monkeypatch.setenv("SUPERGLM_N_JOBS", "16")
    monkeypatch.setattr(numba, "get_num_threads", lambda: 2)
    started = _helpers_started(monkeypatch)
    list(
        chunking._evaluated_chunks(
            GammaLS(), lambda rows: rows.start, 8 * 8192, 8192, held_floats=8
        )
    )
    assert max(started) == 1


def test_a_stream_window_narrows_itself_and_not_the_fits_crew(monkeypatch):
    """A window can be the fit's first section, and the negative binomial's finish kernels use
    every core later. Mutation checks: a stream that sets its own ``n_jobs`` builds the crew at 8
    threads, and one with no limit starts 15 helpers here."""
    crews = []
    started = _helpers_started(monkeypatch)
    build = _parallel._Crew.__init__

    def recording(self, size, started=0):
        crews.append(size)
        build(self, size, started)

    monkeypatch.setattr(_parallel._Crew, "__init__", recording)
    with parallel_config(n_jobs=16), _parallel.fit_pool():
        list(
            chunking._evaluated_chunks(
                GammaLS(), lambda rows: rows.start, 32 * 8192, 8192, held_floats=8
            )
        )
    assert crews == [16]
    assert max(started) == chunking._ARRAY_CHUNK_WORKERS - 1


@pytest.mark.parametrize("range_geometry", [False, True], ids=["copied-designs", "row-ranges"])
def test_a_window_counts_the_designs_its_chunks_hold(monkeypatch, range_geometry):
    """Chunks that copy their designs keep them until the consumer reads them. Mutation check:
    counting only the channels holds a longer window of copies."""
    family, layout, response, plan, coefficients = _family_problem(GammaLS)
    chunk_rows = 16
    channels = 3 * 2 + 3 + 2
    designs = chunking._chunk_design_floats(layout)
    assert designs > 0
    held = channels if range_geometry else channels + designs
    allowance = 8 * chunk_rows * channels * (channels + designs)
    monkeypatch.setattr(chunking, "AUTO_CONTRACTION_MEMORY_BYTES", allowance)
    monkeypatch.setattr(chunking, "_MIN_POOLED_CHUNK_ROWS", 1)
    windows = []
    pooled_for = chunking.pooled_for

    def spy(fn, items, item_bytes, **kwargs):
        windows.append(len(items))
        return pooled_for(fn, items, item_bytes, **kwargs)

    monkeypatch.setattr(chunking, "pooled_for", spy)
    with parallel_config(n_jobs=4):
        chunks = list(
            chunking.iter_likelihood_chunks(
                family,
                layout,
                response,
                plan,
                coefficients,
                chunk_size=chunk_rows,
                curvature_source="observed",
                _range_geometry=range_geometry,
            )
        )
    assert windows[0] == allowance // (8 * held * chunk_rows)
    assert sum(windows) == len(chunks)


def test_a_stream_worker_keeps_a_nested_row_pool_on_its_own_thread():
    """Mutation check: a kernel's own pool would start threads for every chunk."""
    widths = []

    def record_width(rows):
        widths.append(resolve_n_jobs())
        return rows.start

    with parallel_config(n_jobs=64):
        list(chunking._evaluated_chunks(GammaLS(), record_width, 8 * 8192, 8192, held_floats=8))
    assert widths == [1] * 8


def test_only_the_compiled_row_loops_pool_their_own_rows():
    assert TweedieLSS.pools_rows and GeneralizedGammaLSS.pools_rows
    assert not any(getattr(family, "pools_rows", False) for family in _ARRAY_FAMILIES)


def test_a_missed_support_product_is_formed_once_across_threads():
    """Mutation check: without the lock each thread that missed forms its own."""
    formed = []

    class Counted(np.ndarray):
        def __matmul__(self, other):
            formed.append(1)
            time.sleep(0.01)
            return np.asarray(self) @ other

    group = SimpleNamespace(B_unique=np.eye(3), R_inv=np.eye(3).view(Counted))
    predictions = chunking._SupportPredictions()
    beta = np.array([1.0, 2.0, 3.0])
    with ThreadPoolExecutor(6) as pool:
        values = list(pool.map(lambda _: predictions(group, beta), range(6)))
    assert len(formed) == 1
    for value in values:
        np.testing.assert_array_equal(value, beta)
