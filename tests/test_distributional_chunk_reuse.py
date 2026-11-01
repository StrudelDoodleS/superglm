"""Chunked endpoint reuse retains aggregate geometry and certifies fixed data."""

from __future__ import annotations

import gc
import weakref
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import superglm.distributional.solver.solver as solver
from superglm._frame import as_eager_frame
from superglm.distributional.families.gaussian import GaussianLS
from superglm.distributional.families.tweedie import TweedieLSS
from superglm.distributional.family import COMPLETE_OBSERVATION
from superglm.distributional.layout import build_stacked_layout
from superglm.distributional.predictor import Predictor, compile_predictors
from superglm.distributional.solver import DenseSolverConfig, fit_dense_fixed_lambda
from superglm.features import Categorical, Numeric, RandomEffect
from superglm.group_matrix import (
    CategoricalGroupMatrix,
    DesignMatrix,
    DiscretizedSCOPGroupMatrix,
    DiscretizedSSPGroupMatrix,
    DiscretizedTensorGroupMatrix,
    FactorSmoothGroupMatrix,
    RandomEffectGroupMatrix,
    SparseGroupMatrix,
    SupportCompressedSSPGroupMatrix,
)

from ._distributional_weights import resolved_prior


def _problem(n=96):
    x = np.linspace(-1.0, 1.0, n)
    family = GaussianLS()
    weights = resolved_prior(np.linspace(0.8, 1.2, n))
    layout = build_stacked_layout(
        compile_predictors(
            as_eager_frame(pd.DataFrame({"x": x, "z": -x})),
            weights,
            family.parameters,
            (Predictor("location", {"x": Numeric()}), Predictor("scale", {"z": Numeric()})),
            offsets={"location": np.zeros(n), "scale": np.zeros(n)},
        )
    )
    for state in layout.predictors:
        state.design.group_matrices[0].M = state.design.group_matrices[0].M.copy()
    y = 0.3 + 0.4 * x + np.random.default_rng(412).normal(size=n)
    plan = family.bind_likelihood(y, weights, COMPLETE_OBSERVATION)
    penalty = np.diag([0.0, 0.4, 0.0, 0.4])
    config = DenseSolverConfig(coefficient_curvature="observed", tolerance=1e-9)
    return family, layout, y, plan, penalty, config


def _fit(problem, *, session=None, source=None, **kwargs):
    family, layout, y, plan, penalty, config = problem
    return fit_dense_fixed_lambda(
        family,
        layout,
        y,
        plan,
        penalty,
        config=config,
        chunk_size=kwargs.pop("chunk_size", 17),
        initial=kwargs.pop("initial", None if source is None else source.coefficients),
        _reuse_session=session,
        _reuse_source=source,
        **kwargs,
    )


def test_chunked_reuse_skips_initial_likelihood_and_geometry_refresh(monkeypatch):
    problem = _problem()
    session = solver._DenseObservedReuseSession()
    first = _fit(problem, session=session)
    assert first.converged
    assert first.terminal_curvature.actual_source == "observed"
    assert first.terminal_curvature.fallback_count == 0

    def forbid_refresh(*args, **kwargs):
        raise AssertionError("reused chunk endpoint refreshed likelihood derivatives")

    monkeypatch.setattr(GaussianLS, "evaluate_natural", forbid_refresh)
    second = _fit(problem, session=session, source=first)

    assert second.converged
    assert second.iterations == 0
    np.testing.assert_array_equal(second.terminal_score, first.terminal_score)
    np.testing.assert_array_equal(second.terminal_data_curvature, first.terminal_data_curvature)


def test_chunked_reuse_matches_fresh_fit_after_changing_penalty():
    problem = _problem()
    session = solver._DenseObservedReuseSession()
    first = _fit(problem, session=session)
    next_problem = (*problem[:4], 3.0 * problem[4], problem[5])
    reused = _fit(next_problem, session=session, source=first)
    fresh = _fit(next_problem, initial=first.coefficients)
    dense = _fit(next_problem, initial=first.coefficients, chunk_size=None)

    for result in (fresh, dense):
        np.testing.assert_allclose(reused.coefficients, result.coefficients, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(
            reused.terminal_data_curvature, result.terminal_data_curvature, rtol=1e-8, atol=1e-10
        )
        np.testing.assert_allclose(reused.terminal_score, result.terminal_score, atol=1e-7)
        assert reused.log_likelihood == pytest.approx(result.log_likelihood, rel=1e-12)
    # The cross-predictor block is retained, including its signed entries.
    cross = reused.terminal_data_curvature[:2, 2:]
    assert np.any(cross < -0.01)
    assert np.any(cross > 0.01)


@pytest.mark.parametrize(
    "change", ["point", "response", "weights", "family", "layout", "config", "chunk", "copy"]
)
def test_chunked_reuse_refuses_changed_provenance(change, monkeypatch):
    problem = _problem()
    session = solver._DenseObservedReuseSession()
    first = _fit(problem, session=session)
    family, layout, y, plan, penalty, config = problem
    kwargs = {}
    source = first
    if change == "point":
        kwargs["initial"] = first.coefficients + 0.01
    elif change == "response":
        y = y.copy()
    elif change == "weights":
        plan = family.bind_likelihood(y, resolved_prior(np.ones(len(y))), COMPLETE_OBSERVATION)
    elif change == "family":
        family = GaussianLS()
    elif change == "layout":
        layout = replace(layout)
    elif change == "config":
        config = replace(config, tolerance=2e-9)
    elif change == "chunk":
        kwargs["chunk_size"] = 19
    elif change == "copy":
        source = replace(first)
    evaluations = 0
    original = GaussianLS.evaluate_natural

    def counted(*args, **kwargs):
        nonlocal evaluations
        evaluations += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(GaussianLS, "evaluate_natural", counted)
    _fit((family, layout, y, plan, penalty, config), session=session, source=source, **kwargs)
    assert evaluations > 0


@pytest.mark.parametrize("change", ["offset"])
def test_chunked_reuse_refuses_in_place_data_changes_at_fixed_point(change, monkeypatch):
    family, layout, y, plan, _, config = _problem()
    config = replace(config, tolerance=1e20)
    problem = (family, layout, y, plan, np.eye(layout.n_coefficients) * 100.0, config)
    session = solver._DenseObservedReuseSession()
    first = _fit(problem, session=session, initial=np.zeros(layout.n_coefficients))
    assert first.converged
    if change == "response":
        y[0] += 0.5
    elif change == "design":
        # Slope coefficients are zero, so this changes curvature without changing eta.
        layout.predictors[0].design.group_matrices[0].M[0] += 0.5
    else:
        layout.predictors[0].offset.setflags(write=True)
        layout.predictors[0].offset[0] += 0.5
    evaluations = 0
    original = GaussianLS.evaluate_natural

    def counted(*args, **kwargs):
        nonlocal evaluations
        evaluations += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(GaussianLS, "evaluate_natural", counted)
    _fit(problem, session=session, source=first)
    assert evaluations > 0


def test_chunked_reuse_retains_raw_score_before_penalty_cancellation():
    family, layout, y, plan, _, config = _problem()
    config = replace(config, tolerance=1e20)
    initial = np.array([1.0, 0.0, 0.0, 0.0])
    problem = (family, layout, y, plan, np.eye(4) * 1e20, config)
    session = solver._DenseObservedReuseSession()
    first = _fit(problem, session=session, initial=initial)
    next_problem = (family, layout, y, plan, np.eye(4) * 100.0, config)
    reused = _fit(next_problem, session=session, source=first)
    fresh = _fit(next_problem, initial=initial)
    recovered = first.terminal_score + first.penalty @ first.coefficients
    actual = fresh.terminal_score + fresh.penalty @ fresh.coefficients
    assert abs(recovered[0] - actual[0]) > 1.0
    np.testing.assert_array_equal(reused.terminal_score, fresh.terminal_score)


def test_chunked_reuse_does_not_keep_terminal_row_buffers_alive():
    problem = _problem()
    session = solver._DenseObservedReuseSession()
    source = _fit(problem, session=session)
    record = session._chunk_results[id(source)]
    p = problem[1].n_coefficients
    assert record.coefficients.nbytes + record.score_data.nbytes + record.data_curvature.nbytes == (
        8 * (2 * p + p * p)
    )
    assert not record.coefficients.flags.writeable
    assert not record.score_data.flags.writeable
    assert not record.data_curvature.flags.writeable
    references = (weakref.ref(source), weakref.ref(source.eta), weakref.ref(source.theta))
    del source
    gc.collect()
    assert all(reference() is None for reference in references)
    assert not session._chunk_results


def _factor_smooth_problem(discrete, factor_basis="fs"):
    n = 108
    bins = np.arange(n, dtype=np.intp) % 9
    codes = (np.arange(n, dtype=np.intp) // 9) % 3
    x = np.linspace(-1.0, 1.0, 9)
    support = np.column_stack((x, x * x - np.mean(x * x)))
    width = 6 if factor_basis == "fs" else 4
    frame = pd.DataFrame({f"x{i}": np.roll(support[bins, 0], i) for i in range(width)})
    family = GaussianLS()
    weights = resolved_prior(np.ones(n))
    layout = build_stacked_layout(
        compile_predictors(
            as_eager_frame(frame),
            weights,
            family.parameters,
            (
                Predictor("location", {name: Numeric() for name in frame.columns}),
                Predictor("scale", {}),
            ),
            offsets={"location": np.zeros(n), "scale": np.zeros(n)},
        )
    )
    group = FactorSmoothGroupMatrix(
        support if discrete else sp.csr_matrix(support[bins]),
        codes,
        3,
        natural_map=np.array([[1.0, 0.2], [0.0, 0.8]]),
        levels=("a", "b", "c"),
        repeated_penalty_components=(),
        factor_basis=factor_basis,
        bin_idx=bins if discrete else None,
    )
    location = replace(
        layout.predictors[0],
        design=DesignMatrix([group], n=n, p=width),
        groups=(replace(layout.predictors[0].groups[0], start=0, end=width),),
    )
    layout = replace(layout, predictors=(location, layout.predictors[1]))
    y = 0.2 + 0.3 * support[bins, 0] + 0.2 * codes
    y += np.random.default_rng(518).normal(size=n)
    plan = family.bind_likelihood(y, weights, COMPLETE_OBSERVATION)
    penalty = np.eye(layout.n_coefficients) * 0.6
    for state in layout.predictors:
        penalty[state.intercept_index, state.intercept_index] = 0.0
    config = DenseSolverConfig(coefficient_curvature="observed", tolerance=1e-9)
    return family, layout, y, plan, penalty, config


@pytest.mark.parametrize("discrete", [False, True])
@pytest.mark.parametrize("factor_basis", ["fs", "sz"])
def test_factor_smooth_reuse_skips_refresh_and_matches_new_penalty(
    discrete, factor_basis, monkeypatch
):
    problem = _factor_smooth_problem(discrete, factor_basis)

    def forbid_expansion(*args, **kwargs):
        raise AssertionError("factor-smooth reuse expanded the observation design")

    monkeypatch.setattr(FactorSmoothGroupMatrix, "toarray", forbid_expansion)
    session = solver._DenseObservedReuseSession()
    source = _fit(problem, session=session)
    assert source.converged

    record = session._chunk_results[id(source)]
    p = problem[1].n_coefficients
    assert record.coefficients.nbytes + record.score_data.nbytes + record.data_curvature.nbytes == (
        8 * (2 * p + p * p)
    )

    def forbid_refresh(*args, **kwargs):
        raise AssertionError("factor-smooth endpoint refreshed likelihood derivatives")

    with monkeypatch.context() as patch:
        patch.setattr(GaussianLS, "evaluate_natural", forbid_refresh)
        same = _fit(problem, session=session, source=source)
    assert same.iterations == 0
    next_problem = (*problem[:4], problem[4] * 1.7, problem[5])
    reused = _fit(next_problem, session=session, source=source)
    fresh = _fit(next_problem, initial=source.coefficients)
    assert reused.converged and fresh.converged
    np.testing.assert_array_equal(reused.coefficients, fresh.coefficients)
    np.testing.assert_array_equal(reused.terminal_data_curvature, fresh.terminal_data_curvature)
    np.testing.assert_array_equal(reused.terminal_score, fresh.terminal_score)
    assert reused.log_likelihood == fresh.log_likelihood


def _compressed_problem(kind):
    family, layout, y, plan, penalty, config = _problem()
    n = len(y)
    states = []
    for state in layout.predictors:
        bins = np.arange(n, dtype=np.intp) % 8
        support = np.linspace(-1, 1, 8)[:, None]
        if kind is DiscretizedSCOPGroupMatrix:
            group = kind(support, bins)
        elif kind is DiscretizedTensorGroupMatrix:
            left = np.linspace(-1, 1, 4)[:, None]
            right = np.linspace(0.5, 1.5, 2)[:, None]
            group = kind(left, right, bins // 2, bins % 2, np.kron(left, right), np.eye(1), bins, 1)
        else:
            group = kind(support, np.eye(1), bins)
        states.append(replace(state, design=DesignMatrix([group], n=n, p=1)))
    return family, replace(layout, predictors=tuple(states)), y, plan, penalty, config


@pytest.mark.parametrize(
    "kind",
    [
        DiscretizedSSPGroupMatrix,
        SupportCompressedSSPGroupMatrix,
        DiscretizedSCOPGroupMatrix,
        DiscretizedTensorGroupMatrix,
    ],
)
def test_supported_compressed_reuse_does_not_expand_slopes(kind, monkeypatch):
    family, layout, y, plan, _, config = _compressed_problem(kind)
    problem = (family, layout, y, plan, np.eye(4) * 100.0, replace(config, tolerance=1e20))
    session = solver._DenseObservedReuseSession()
    first = _fit(problem, session=session, initial=np.zeros(4))
    assert first.converged
    assert id(first) in session._chunk_results

    def forbidden(*args, **kwargs):
        raise AssertionError("compressed reuse expanded slopes or refreshed likelihood")

    monkeypatch.setattr(kind, "toarray", forbidden)
    monkeypatch.setattr(GaussianLS, "evaluate_natural", forbidden)
    second = _fit(problem, session=session, source=first)
    np.testing.assert_array_equal(second.terminal_data_curvature, first.terminal_data_curvature)


def _builtin_context(family, y=None):
    y = np.arange(1.0, 13.0) if y is None else np.asarray(y, dtype=np.float64)
    weights = resolved_prior(np.ones(len(y)))
    plan = family.bind_likelihood(y, weights, COMPLETE_OBSERVATION)
    layout = build_stacked_layout(
        compile_predictors(
            as_eager_frame(pd.DataFrame({"row": y})),
            weights,
            family.parameters,
            tuple(Predictor(parameter.name, {}) for parameter in family.parameters),
            offsets={parameter.name: np.zeros(len(y)) for parameter in family.parameters},
        )
    )
    p = layout.n_coefficients
    return solver._validated_context(
        family,
        layout,
        y,
        plan,
        np.zeros((p, p)),
        coefficient_curvature="observed",
        chunk_size=5,
        coefficient_face=None,
    )


def test_three_parameter_tweedie_reuse_matches_fresh_changing_penalty():
    # Independent compound-Poisson/gamma sampling at power 1.5.
    rng = np.random.default_rng(732)
    counts = rng.poisson(np.sqrt(1.5) / 0.4, size=48)
    y = np.zeros(len(counts))
    positive = counts > 0
    y[positive] = rng.gamma(counts[positive], 0.4 * np.sqrt(1.5))
    context = _builtin_context(TweedieLSS(), y)
    config = DenseSolverConfig(coefficient_curvature="observed", tolerance=1e-8)
    penalty = np.diag([0.5, 0.5, 5.0])
    problem = (context.family, context.layout, y, context.likelihood_plan, penalty, config)
    session = solver._DenseObservedReuseSession()
    source = _fit(problem, session=session, initial=np.array([0.3, -0.1, 0.0]))
    assert source.converged
    assert id(source) in session._chunk_results
    next_problem = (*problem[:4], penalty * 1.7, config)
    reused = _fit(next_problem, session=session, source=source)
    fresh = _fit(next_problem, initial=source.coefficients)
    assert reused.converged and fresh.converged
    np.testing.assert_array_equal(reused.coefficients, fresh.coefficients)
    np.testing.assert_array_equal(reused.terminal_score, fresh.terminal_score)
    np.testing.assert_array_equal(reused.terminal_data_curvature, fresh.terminal_data_curvature)
    assert reused.optimizing_log_likelihood == fresh.optimizing_log_likelihood
    assert np.linalg.cond(reused.terminal_penalized_curvature) < 100.0


def test_three_parameter_tweedie_reuse_skips_likelihood_refresh(monkeypatch):
    context = _builtin_context(TweedieLSS())
    config = DenseSolverConfig(coefficient_curvature="observed", tolerance=1e20)
    problem = (
        context.family,
        context.layout,
        context.response,
        context.likelihood_plan,
        np.eye(3) * 1000.0,
        config,
    )
    session = solver._DenseObservedReuseSession()
    source = _fit(problem, session=session, initial=np.zeros(3), chunk_size=5)
    assert source.converged
    assert id(source) in session._chunk_results

    def forbidden(*args, **kwargs):
        raise AssertionError("three-parameter reuse refreshed row likelihood")

    monkeypatch.setattr(TweedieLSS, "evaluate_natural", forbidden)
    reused = _fit(problem, session=session, source=source, chunk_size=5)
    assert reused.converged
    assert reused.iterations == 0


def _categorical_problem(kind):
    n = 96
    x = np.linspace(-1.0, 1.0, n)
    categories = np.resize(np.array(["a", "b", "c"]), n)
    family = GaussianLS()
    weights = resolved_prior(np.ones(n))
    category = RandomEffect() if kind is RandomEffectGroupMatrix else Categorical(base="a")
    layout = build_stacked_layout(
        compile_predictors(
            as_eager_frame(pd.DataFrame({"category": categories, "x": x})),
            weights,
            family.parameters,
            (
                Predictor("location", {"category": category, "x": Numeric()}),
                Predictor("scale", {"x": Numeric()}),
            ),
            offsets={"location": np.zeros(n), "scale": np.zeros(n)},
        )
    )
    if kind is SparseGroupMatrix:
        state = layout.predictors[0]
        groups = list(state.design.group_matrices)
        groups[0] = SparseGroupMatrix(sp.csr_matrix(groups[0].toarray()))
        layout = replace(
            layout,
            predictors=(
                replace(state, design=DesignMatrix(groups, n, state.design.p)),
                layout.predictors[1],
            ),
        )
    y = 0.2 + 0.3 * x + 0.4 * (categories == "b") + np.random.default_rng(121).normal(size=n)
    plan = family.bind_likelihood(y, weights, COMPLETE_OBSERVATION)
    penalty = 0.4 * np.eye(layout.n_coefficients)
    for state in layout.predictors:
        penalty[state.intercept_index, state.intercept_index] = 0.0
    config = DenseSolverConfig(coefficient_curvature="observed", tolerance=1e-9)
    return family, layout, y, plan, penalty, config


@pytest.mark.parametrize(
    "kind", [CategoricalGroupMatrix, RandomEffectGroupMatrix, SparseGroupMatrix]
)
def test_category_and_csr_reuse_does_not_expand_or_refresh(kind, monkeypatch):
    family, layout, y, plan, _, config = _categorical_problem(kind)
    p = layout.n_coefficients
    problem = (family, layout, y, plan, np.eye(p) * 1000.0, replace(config, tolerance=1e20))
    session = solver._DenseObservedReuseSession()
    source = _fit(problem, session=session, initial=np.zeros(p))
    assert source.converged
    assert id(source) in session._chunk_results

    def forbidden(*args, **kwargs):
        raise AssertionError("category/CSR reuse expanded design or refreshed likelihood")

    monkeypatch.setattr(kind, "toarray", forbidden)
    monkeypatch.setattr(GaussianLS, "evaluate_natural", forbidden)
    reused = _fit(problem, session=session, source=source)
    assert reused.iterations == 0
    np.testing.assert_array_equal(reused.terminal_data_curvature, source.terminal_data_curvature)


@pytest.mark.parametrize(
    "kind", [CategoricalGroupMatrix, RandomEffectGroupMatrix, SparseGroupMatrix]
)
def test_category_and_csr_changing_penalty_reuse_matches_fresh(kind):
    problem = _categorical_problem(kind)
    session = solver._DenseObservedReuseSession()
    source = _fit(problem, session=session)
    assert source.converged
    assert id(source) in session._chunk_results
    next_problem = (*problem[:4], problem[4] * 1.8, problem[5])
    reused = _fit(next_problem, session=session, source=source)
    fresh = _fit(next_problem, initial=source.coefficients)
    np.testing.assert_array_equal(reused.coefficients, fresh.coefficients)
    np.testing.assert_array_equal(reused.terminal_score, fresh.terminal_score)
    np.testing.assert_array_equal(reused.terminal_data_curvature, fresh.terminal_data_curvature)
    assert reused.log_likelihood == fresh.log_likelihood
