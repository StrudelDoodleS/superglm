"""Chunked endpoint reuse retains aggregate geometry and certifies fixed data."""

from __future__ import annotations

import gc
import weakref
from dataclasses import fields, replace

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import superglm.distributional.solver.solver as solver
from superglm._frame import as_eager_frame
from superglm.distributional.families.gamma import GammaLS
from superglm.distributional.families.gaussian import GaussianLS
from superglm.distributional.families.generalized_gamma import GeneralizedGammaLSS
from superglm.distributional.families.generalized_pareto import GeneralizedParetoLSS
from superglm.distributional.families.log_normal import LogNormalLS
from superglm.distributional.families.negative_binomial import NegativeBinomialLS
from superglm.distributional.families.tweedie import TweedieLSS
from superglm.distributional.families.two_piece import TwoPieceLogNormalLSS, TwoPieceNormalLSS
from superglm.distributional.family import COMPLETE_OBSERVATION
from superglm.distributional.layout import build_stacked_layout
from superglm.distributional.predictor import Predictor, compile_predictors
from superglm.distributional.solver import DenseSolverConfig, fit_dense_fixed_lambda
from superglm.distributional.weights import ResolvedLikelihoodWeights
from superglm.features import Categorical, Numeric, RandomEffect
from superglm.group_matrix import (
    CategoricalGroupMatrix,
    DenseGroupMatrix,
    DesignMatrix,
    DiscretizedSCOPGroupMatrix,
    DiscretizedSSPGroupMatrix,
    DiscretizedTensorGroupMatrix,
    RandomEffectGroupMatrix,
    SparseGroupMatrix,
    SupportCompressedSSPGroupMatrix,
)

from ._distributional_weights import resolved_prior

_BUILTIN_FAMILIES = (
    GaussianLS,
    GammaLS,
    NegativeBinomialLS,
    LogNormalLS,
    GeneralizedGammaLSS,
    GeneralizedParetoLSS,
    TwoPieceLogNormalLSS,
    TwoPieceNormalLSS,
    TweedieLSS,
)


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


@pytest.mark.parametrize("change", ["response", "design", "offset"])
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


@pytest.mark.parametrize("field", ["B_unique", "R_inv", "bin_idx"])
def test_compressed_certificate_refuses_changed_transform_or_row_map(field, monkeypatch):
    family, layout, y, plan, _, config = _compressed_problem(DiscretizedSSPGroupMatrix)
    problem = (family, layout, y, plan, np.eye(4) * 100.0, replace(config, tolerance=1e20))
    session = solver._DenseObservedReuseSession()
    source = _fit(problem, session=session, initial=np.zeros(4))
    assert id(source) in session._chunk_results
    values = getattr(layout.predictors[0].design.group_matrices[0], field)
    values.flat[0] += 1
    calls = 0
    original = GaussianLS.evaluate_natural

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(GaussianLS, "evaluate_natural", counted)
    _fit(problem, session=session, source=source)
    assert calls > 0


def test_unfamiliar_group_semantics_refuse_chunk_reuse():
    class CustomDenseGroup(DenseGroupMatrix):
        pass

    family, layout, y, plan, penalty, config = _problem()
    state = layout.predictors[0]
    custom = CustomDenseGroup(state.design.group_matrices[0].M)
    layout = replace(
        layout,
        predictors=(replace(state, design=DesignMatrix([custom], len(y), 1)), layout.predictors[1]),
    )
    session = solver._DenseObservedReuseSession()
    source = _fit((family, layout, y, plan, penalty, config), session=session)
    assert source.converged
    assert not session._chunk_results


def test_certificate_hashes_noncontiguous_designs_in_bounded_buffers(monkeypatch):
    problem = _problem(n=24_000)
    for state in problem[1].predictors:
        original = state.design.group_matrices[0].M.reshape(-1)
        backing = np.column_stack((original, original))
        state.design.group_matrices[0].M = backing[:, :1]
        assert not state.design.group_matrices[0].M.flags.c_contiguous
    context = solver._validated_context(
        *problem[:5], coefficient_curvature="observed", chunk_size=17, coefficient_face=None
    )
    original_sha256 = solver.hashlib.sha256
    sizes = []

    class BoundedHash:
        def __init__(self, *args, **kwargs):
            self.digest = original_sha256(*args, **kwargs)

        def update(self, value):
            sizes.append(len(value))
            self.digest.update(value)

        def hexdigest(self):
            return self.digest.hexdigest()

    monkeypatch.setattr(solver.hashlib, "sha256", BoundedHash)
    assert solver._chunk_reuse_data_certificate(context) is not None
    assert max(sizes) <= 8192 * 8


@pytest.mark.parametrize(
    "family,field",
    [
        (GaussianLS(), "parameter_independent_carrier"),
        (GammaLS(), "parameter_independent_carrier"),
        (GammaLS(), "exact_response"),
        (NegativeBinomialLS(), "parameter_independent_carrier"),
        (NegativeBinomialLS(), "exact_response"),
        (NegativeBinomialLS(), "exact_count"),
        (LogNormalLS(), "exact_response"),
        (LogNormalLS(), "parameter_independent_carrier"),
        (GeneralizedGammaLSS(), "exact_response"),
        (GeneralizedGammaLSS(), "parameter_independent_carrier"),
        (GeneralizedParetoLSS(), "exact_response"),
        (GeneralizedParetoLSS(), "parameter_independent_carrier"),
        (TwoPieceLogNormalLSS(), "exact_response"),
        (TwoPieceLogNormalLSS(), "parameter_independent_carrier"),
        (TwoPieceNormalLSS(), "exact_response"),
        (TwoPieceNormalLSS(), "parameter_independent_carrier"),
    ],
)
def test_certificate_covers_actual_prepared_family_arrays(family, field):
    context = _builtin_context(family)
    plan = context.likelihood_plan
    before = solver._chunk_reuse_data_certificate(context)
    assert before is not None
    identifier = plan.plan_identifier
    mutable = getattr(plan, field).copy()
    object.__setattr__(plan, field, mutable)
    mutable[0] += 1.0
    # Stored digests can remain unchanged while a manually assembled plan's
    # numerical inputs change. The certificate must inspect the actual arrays.
    assert plan.plan_identifier == identifier
    assert solver._chunk_reuse_data_certificate(context) != before


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


@pytest.mark.parametrize("family_type", _BUILTIN_FAMILIES)
def test_every_builtin_family_has_positive_certificate_eligibility(family_type):
    context = _builtin_context(family_type())
    assert solver._chunk_reuse_data_certificate(context) is not None


@pytest.mark.parametrize("family_type", _BUILTIN_FAMILIES)
def test_certificate_covers_live_weight_semantics_despite_stored_digests(family_type):
    context = _builtin_context(family_type())
    before = solver._chunk_reuse_data_certificate(context)
    assert before is not None
    plan = context.likelihood_plan
    identifier = plan.plan_identifier
    object.__setattr__(plan.weights.provenance.contract, "semantics", "frequency")
    assert plan.plan_identifier == identifier
    assert solver._chunk_reuse_data_certificate(context) != before


def test_custom_weight_slicing_semantics_refuse_certificate():
    class CustomWeights(ResolvedLikelihoodWeights):
        def take(self, indices):
            return super().take(indices[::-1])

    context = _builtin_context(GaussianLS())
    plan = context.likelihood_plan
    object.__setattr__(plan, "weights", CustomWeights(**vars(plan.weights)))
    assert solver._chunk_reuse_data_certificate(context) is None


@pytest.mark.parametrize(
    "family_type,field,value",
    [
        (GaussianLS, "scale_floor", 0.02),
        (LogNormalLS, "scale_floor", 0.02),
        (GeneralizedGammaLSS, "scale_floor", 0.02),
        (GeneralizedParetoLSS, "shape_upper", 0.9),
        (TwoPieceLogNormalLSS, "skew_bound", 0.8),
        (TwoPieceNormalLSS, "skew_bound", 0.8),
        (TweedieLSS, "power_upper", 1.9),
    ],
)
def test_certificate_covers_live_family_configuration(family_type, field, value):
    context = _builtin_context(family_type())
    before = solver._chunk_reuse_data_certificate(context)
    assert before is not None
    object.__setattr__(context.family, field, value)
    assert solver._chunk_reuse_data_certificate(context) != before


@pytest.mark.parametrize("family_type", [GeneralizedParetoLSS, TwoPieceNormalLSS, TweedieLSS])
def test_certificate_covers_bounded_link_configuration(family_type):
    context = _builtin_context(family_type())
    before = solver._chunk_reuse_data_certificate(context)
    assert before is not None
    link = context.layout.predictors[-1].link
    object.__setattr__(link, "upper", link.upper - 0.01)
    assert solver._chunk_reuse_data_certificate(context) != before


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


@pytest.mark.parametrize("subclass_target", ["family", "plan"])
def test_contract_registration_does_not_grant_inherited_eligibility(subclass_target):
    context = _builtin_context(GaussianLS())
    if subclass_target == "family":

        class CustomFamily(GaussianLS):
            pass

        context = replace(context, family=CustomFamily())
    else:
        plan = context.likelihood_plan

        class CustomPlan(type(plan)):
            pass

        copied = CustomPlan(
            **{item.name: getattr(plan, item.name) for item in fields(plan) if item.init}
        )
        context = replace(context, likelihood_plan=copied)
    assert solver._chunk_reuse_data_certificate(context) is None


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


@pytest.mark.parametrize(
    "kind,field",
    [
        (CategoricalGroupMatrix, "codes"),
        (RandomEffectGroupMatrix, "codes"),
        (SparseGroupMatrix, "data"),
        (SparseGroupMatrix, "indices"),
        (SparseGroupMatrix, "indptr"),
    ],
)
def test_category_and_csr_changes_refuse_reuse_even_when_eta_is_fixed(kind, field, monkeypatch):
    family, layout, y, plan, _, config = _categorical_problem(kind)
    p = layout.n_coefficients
    problem = (family, layout, y, plan, np.eye(p) * 1000.0, replace(config, tolerance=1e20))
    session = solver._DenseObservedReuseSession()
    source = _fit(problem, session=session, initial=np.zeros(p))
    assert id(source) in session._chunk_results
    group = layout.predictors[0].design.group_matrices[0]
    if field == "codes":
        group.codes[0] = (group.codes[0] + 1) % group.n_levels
    elif field == "data":
        group.M.data[0] += 0.25
    elif field == "indices":
        group.M.indices[0] = 1 - group.M.indices[0]
    else:
        # Move the second row's entry into the originally empty first row.
        group.M.indptr[1] = group.M.indptr[2]
    calls = 0
    original = GaussianLS.evaluate_natural

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(GaussianLS, "evaluate_natural", counted)
    _fit(problem, session=session, source=source)
    assert calls > 0


@pytest.mark.parametrize("change", ["shape", "data_dtype", "indices_dtype", "indptr_dtype"])
def test_csr_certificate_covers_storage_shape_and_dtypes(change):
    problem = _categorical_problem(SparseGroupMatrix)
    context = solver._validated_context(
        *problem[:5], coefficient_curvature="observed", chunk_size=17, coefficient_face=None
    )
    before = solver._chunk_reuse_data_certificate(context)
    assert before is not None
    matrix = context.layout.predictors[0].design.group_matrices[0].M
    if change == "shape":
        matrix._shape = (matrix.shape[0], matrix.shape[1] + 1)
    elif change == "data_dtype":
        matrix.data = matrix.data.astype(np.float32)
    else:
        name = change.removesuffix("_dtype")
        value = getattr(matrix, name)
        dtype = np.int64 if value.dtype != np.dtype(np.int64) else np.int32
        setattr(matrix, name, value.astype(dtype))
    assert solver._chunk_reuse_data_certificate(context) != before


@pytest.mark.parametrize("replacement", ["csc", "csr_array", "subclass"])
def test_csr_certificate_refuses_unrecognized_matrix_semantics(replacement):
    problem = _categorical_problem(SparseGroupMatrix)
    context = solver._validated_context(
        *problem[:5], coefficient_curvature="observed", chunk_size=17, coefficient_face=None
    )
    group = context.layout.predictors[0].design.group_matrices[0]
    if replacement == "csc":
        group.M = group.M.tocsc()
    elif replacement == "csr_array":
        group.M = sp.csr_array(group.M)
    else:

        class CustomCSR(sp.csr_matrix):
            pass

        group.M = CustomCSR(group.M)
    assert solver._chunk_reuse_data_certificate(context) is None


@pytest.mark.parametrize("flag", ["_has_sorted_indices", "_has_canonical_format"])
def test_csr_certificate_covers_independent_cached_flag_changes(flag):
    problem = _categorical_problem(SparseGroupMatrix)
    context = solver._validated_context(
        *problem[:5], coefficient_curvature="observed", chunk_size=17, coefficient_face=None
    )
    before = solver._chunk_reuse_data_certificate(context)
    assert before is not None
    matrix = context.layout.predictors[0].design.group_matrices[0].M
    setattr(matrix, flag, not getattr(matrix, flag, None))
    assert solver._chunk_reuse_data_certificate(context) != before


def test_csr_certificate_does_not_populate_cached_flags():
    problem = _categorical_problem(SparseGroupMatrix)
    context = solver._validated_context(
        *problem[:5], coefficient_curvature="observed", chunk_size=17, coefficient_face=None
    )
    matrix = context.layout.predictors[0].design.group_matrices[0].M
    flags = ("_has_sorted_indices", "_has_canonical_format")
    for flag in flags:
        vars(matrix).pop(flag, None)
    assert solver._chunk_reuse_data_certificate(context) is not None
    assert not any(flag in vars(matrix) for flag in flags)
