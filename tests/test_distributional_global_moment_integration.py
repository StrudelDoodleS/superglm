"""Streamed global geometry admission, cleanup and complete fallback replay."""

from __future__ import annotations

import weakref
from dataclasses import fields, replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from superglm._frame import as_eager_frame
from superglm.distributional import _global_moment_policy as policy
from superglm.distributional.families.gamma import GammaLS
from superglm.distributional.families.gaussian import GaussianLS, LowerBoundedLogLink
from superglm.distributional.family import COMPLETE_OBSERVATION
from superglm.distributional.layout import build_stacked_layout
from superglm.distributional.predictor import compile_predictors
from superglm.distributional.solver import chunks
from superglm.distributional.solver._global_moments import GlobalMomentRefusalError
from superglm.distributional.weights import ResolvedLikelihoodWeights
from superglm.group_matrix import DiscretizedSplineCategoricalGroupMatrix, DiscretizedSSPGroupMatrix
from superglm.links import IdentityLink, LogLink

from ._distributional_weights import resolved_prior
from .test_distributional_automatic_panels import _literal_design, _predictor, _problem
from .test_distributional_small_group_panel_integration import _assert_geometry, _stream


def _row_stream(layout, score, curvature):
    start = 0
    for chunk in _stream(layout, score, curvature):
        stop = start + len(chunk.score_eta)
        chunk.rows = chunks.RowChunk(start, stop, np.arange(start, stop, dtype=np.intp))
        yield chunk
        start = stop


def _assemble(layout, **kwargs):
    p = layout.n_coefficients
    return chunks.assemble_chunked_geometry(
        None,
        layout,
        None,
        None,
        np.zeros(p),
        penalty=np.zeros((p, p)),
        chunk_size=8,
        curvature_source="observed",
        **kwargs,
    )


def test_global_success_builds_before_stream_and_never_allocates_fallback(monkeypatch):
    layout, score, curvature = _problem()
    events = []
    result = object()

    class Plan:
        def reset(self, **kwargs):
            events.append("reset")

        def add_chunk(self, plans, score_eta, curvature_packed):
            events.append("add")

        def finish(self):
            events.append("finish")
            return result

        def close(self):
            events.append("close")

    def build(*args, **kwargs):
        assert events == []
        events.append("build")
        return SimpleNamespace(plan=Plan())

    def stream(*args, **kwargs):
        events.append("stream")
        try:
            yield from _row_stream(layout, score, curvature)
        finally:
            events.append("stream-close")

    def forbidden(*args, **kwargs):
        raise AssertionError("global success allocated the fallback accumulator")

    monkeypatch.setattr(
        chunks, "automatic_global_moment_budget", lambda *args: 64 << 20, raising=False
    )
    monkeypatch.setattr(chunks, "build_global_moment_plan", build, raising=False)
    monkeypatch.setattr(chunks, "iter_likelihood_chunks", stream)
    monkeypatch.setattr(chunks, "GroupedGeometryAccumulator", forbidden)
    assert _assemble(layout) is result
    assert events[:3] == ["build", "reset", "stream"]
    assert events.count("add") == 5
    assert events[-3:] == ["stream-close", "finish", "close"]


@pytest.mark.parametrize("failure", ["add", "finish"])
def test_recoverable_partial_geometry_is_released_before_complete_replay(monkeypatch, failure):
    layout, score, curvature = _problem()
    events = []
    borrowed = []
    passes = []

    class Plan:
        def __init__(self):
            self.buffer = np.zeros((7, 7))
            borrowed.append(weakref.ref(self.buffer))
            self.count = 0

        def reset(self, **kwargs):
            pass

        def add_chunk(self, plans, score_eta, curvature_packed):
            self.count += 1
            events.append("add")
            if failure == "add" and self.count == 3:
                raise GlobalMomentRefusalError("numerical-domain", recoverable=True)

        def finish(self):
            raise GlobalMomentRefusalError("nonfinite-final-output", recoverable=True)

        def close(self):
            self.buffer = None
            events.append("plan-close")

    def stream(*args, **kwargs):
        if passes:
            assert events[-1] == "plan-close"
            assert all(reference() is None for reference in borrowed)
        passes.append([])
        try:
            for chunk in _row_stream(layout, score, curvature):
                passes[-1].append(chunk.rows.start)
                if len(passes) == 1:
                    borrowed.append(weakref.ref(chunk.score_eta))
                    borrowed.append(weakref.ref(chunk.plans[0]))
                yield chunk
        finally:
            events.append("stream-close")

    monkeypatch.setattr(chunks, "automatic_global_moment_budget", lambda *args: 64 << 20)
    monkeypatch.setattr(
        chunks, "build_global_moment_plan", lambda *args, **kwargs: SimpleNamespace(plan=Plan())
    )
    monkeypatch.setattr(chunks, "iter_likelihood_chunks", stream)
    result = _assemble(layout)
    assert len(passes) == 2
    assert passes[0] == ([0, 8, 16] if failure == "add" else [0, 8, 16, 24, 32])
    assert passes[1] == [0, 8, 16, 24, 32]
    assert events.count("plan-close") == 1
    assert events.count("stream-close") == 2
    p = layout.n_coefficients
    _assert_geometry(
        result, layout, _literal_design(layout), score, curvature, np.zeros((p, p)), np.zeros(p)
    )


@pytest.mark.parametrize("failure", ["authority", "shape", "unrelated", "stream", "rows"])
def test_contract_failures_propagate_after_cleanup_without_replay(monkeypatch, failure):
    layout, score, curvature = _problem()
    events = []
    error = (
        GlobalMomentRefusalError(failure)
        if failure in ("authority", "shape")
        else RuntimeError("original stream failure")
    )

    class Plan:
        def reset(self, **kwargs):
            pass

        def add_chunk(self, *args):
            if failure not in ("stream", "rows"):
                raise error

        def close(self):
            events.append("plan-close")

    def stream(*args, **kwargs):
        events.append("stream")
        try:
            for chunk in _row_stream(layout, score, curvature):
                if failure == "stream":
                    raise error
                if failure == "rows":
                    chunk.rows = chunks.RowChunk(1, 9, np.arange(1, 9, dtype=np.intp))
                yield chunk
        finally:
            events.append("stream-close")

    monkeypatch.setattr(chunks, "automatic_global_moment_budget", lambda *args: 64 << 20)
    monkeypatch.setattr(
        chunks, "build_global_moment_plan", lambda *args, **kwargs: SimpleNamespace(plan=Plan())
    )
    monkeypatch.setattr(chunks, "iter_likelihood_chunks", stream)
    with pytest.raises(ValueError if failure == "rows" else type(error)) as caught:
        _assemble(layout)
    if failure != "rows":
        assert caught.value is error
    assert events == ["stream", "stream-close", "plan-close"]


@pytest.mark.parametrize("budget", [None, 1, 64 << 20])
def test_explicit_panel_options_never_attempt_global(monkeypatch, budget):
    layout, score, curvature = _problem()

    def forbidden(*args, **kwargs):
        raise AssertionError("explicit panel option reached global admission")

    monkeypatch.setattr(chunks, "automatic_global_moment_budget", forbidden)
    monkeypatch.setattr(chunks, "build_global_moment_plan", forbidden)
    monkeypatch.setattr(
        chunks,
        "iter_likelihood_chunks",
        lambda *args, **kwargs: _row_stream(layout, score, curvature),
    )
    result = _assemble(layout, small_group_panel_byte_budget=budget)
    p = layout.n_coefficients
    _assert_geometry(
        result, layout, _literal_design(layout), score, curvature, np.zeros((p, p)), np.zeros(p)
    )


@pytest.mark.parametrize("reason", ["unsupported-layout", "budget"])
def test_constructor_refusal_opens_only_baseline_stream(monkeypatch, reason):
    layout, score, curvature = _problem()
    calls = []

    def build(*args, **kwargs):
        assert not calls
        calls.append("build")
        assert kwargs["chunk_size"] == 8
        return SimpleNamespace(plan=None, reason=reason)

    def stream(*args, **kwargs):
        calls.append("stream")
        yield from _row_stream(layout, score, curvature)

    monkeypatch.setattr(chunks, "automatic_global_moment_budget", lambda *args: 64 << 20)
    monkeypatch.setattr(chunks, "build_global_moment_plan", build)
    monkeypatch.setattr(chunks, "iter_likelihood_chunks", stream)
    result = _assemble(layout)
    assert calls == ["build", "stream"]
    p = layout.n_coefficients
    _assert_geometry(
        result, layout, _literal_design(layout), score, curvature, np.zeros((p, p)), np.zeros(p)
    )


@pytest.mark.parametrize("companion", ["mixed", "intercept"])
def test_real_streamed_signed_rectangular_geometry_matches_stored_row_oracle(
    monkeypatch, companion
):
    layout, score, curvature = _problem(("mixed", companion))
    curvature[::4] = 0
    monkeypatch.setattr(chunks, "automatic_global_moment_budget", lambda *args: 64 << 20)
    monkeypatch.setattr(
        chunks,
        "iter_likelihood_chunks",
        lambda *args, **kwargs: _row_stream(layout, score, curvature),
    )
    builds = []
    original = chunks.build_global_moment_plan

    def build(*args, **kwargs):
        built = original(*args, **kwargs)
        assert built.plan is not None, built.reason
        builds.append(built.plan)
        return built

    monkeypatch.setattr(chunks, "build_global_moment_plan", build)
    result = _assemble(layout)
    assert len(builds) == 1
    p = layout.n_coefficients
    _assert_geometry(
        result, layout, _literal_design(layout), score, curvature, np.zeros((p, p)), np.zeros(p)
    )


def _bound(family):
    return family.bind_likelihood(np.ones(37), resolved_prior(np.ones(37)), COMPLETE_OBSERVATION)


def _metadata_rows(layout, rows, bins=256):
    """Change dimensions only; any attempted access to these arrays must fail."""

    class ShapeOnly:
        def __init__(self, shape):
            self.shape = shape

        def __array__(self, *args, **kwargs):
            raise AssertionError("automatic selector scanned source storage")

        def __getitem__(self, item):
            raise AssertionError("automatic selector indexed source storage")

    for state in layout.predictors:
        state.design.n = rows
        for group in state.design.group_matrices:
            group.shape = (rows, group.shape[1])
            if type(group) in (DiscretizedSSPGroupMatrix, DiscretizedSplineCategoricalGroupMatrix):
                group.B_unique = ShapeOnly((bins, group.B_unique.shape[1]))
                group.R_inv = ShapeOnly(group.R_inv.shape)


@pytest.mark.parametrize(
    "rows,bins,admitted",
    [(262143, 256, False), (262144, 256, True), (262144, 257, False), (524288, 256, True)],
)
@pytest.mark.parametrize("family_type", [GaussianLS, GammaLS])
def test_metadata_scope_boundaries_do_not_scan_arrays(rows, bins, admitted, family_type):
    layout, _, _ = _problem()
    _metadata_rows(layout, rows, bins)
    family = family_type()
    layout = replace(
        layout,
        predictors=tuple(
            replace(state, link=spec.default_link)
            for state, spec in zip(layout.predictors, family.parameters)
        ),
    )
    assert policy.automatic_global_moment_budget(family, _bound(family), layout) == (
        (64 << 20) if admitted else None
    )


@pytest.mark.parametrize(
    "mode", ["mixed", "intercept", "numeric", "category", "support", "numeric_spline"]
)
def test_every_slope_predictor_must_have_the_existing_mixed_envelope(mode):
    layout, _, _ = _problem(("mixed", mode))
    _metadata_rows(layout, 262144)
    family = GaussianLS()
    assert policy.automatic_global_moment_budget(family, _bound(family), layout) == (
        (64 << 20) if mode in ("mixed", "intercept") else None
    )


def test_exact_family_and_bound_plan_replay_capability_is_required():
    layout, _, _ = _problem()
    _metadata_rows(layout, 262144)
    family = GaussianLS()
    plan = _bound(family)

    class CustomGaussian(GaussianLS):
        pass

    class CustomPlan(type(plan)):
        pass

    for candidate, bound in (
        (CustomGaussian(), plan),
        (family, object.__new__(CustomPlan)),
        (family, _bound(GammaLS())),
        (object(), plan),
    ):
        assert policy.automatic_global_moment_budget(candidate, bound, layout) is None


@pytest.mark.parametrize("link_type", [IdentityLink, LogLink, LowerBoundedLogLink])
@pytest.mark.parametrize("predictor_index", [0, 1])
def test_replay_link_admission_requires_exact_audited_types(link_type, predictor_index):
    layout, _, _ = _problem(("mixed", "intercept"))
    _metadata_rows(layout, 262144)
    family = GaussianLS()
    plan = _bound(family)

    class StatefulLink(link_type):
        def inverse(self, eta):
            raise AssertionError("metadata policy evaluated a custom link")

    for concrete_type, expected in ((link_type, 64 << 20), (StatefulLink, None)):
        link = concrete_type(1e-6) if link_type is LowerBoundedLogLink else concrete_type()
        states = list(layout.predictors)
        states[predictor_index] = replace(states[predictor_index], link=link)
        candidate = replace(layout, predictors=tuple(states))
        assert policy.automatic_global_moment_budget(family, plan, candidate) == expected


@pytest.mark.parametrize("predictor_index,base", [(0, IdentityLink), (1, LogLink)])
def test_custom_stateful_links_use_one_actual_baseline_stream(monkeypatch, predictor_index, base):
    layout, _, _ = _problem(("mixed", "intercept"))
    # The cost gate remains real; only the measured row-count scope is reduced
    # for this small dispatch witness. Repeated source rows preserve its design.
    indices = np.resize(np.arange(37, dtype=np.intp), 512)
    states = [
        replace(state, design=state.design.row_subset(indices), offset=state.offset[indices])
        for state in layout.predictors
    ]

    class StatefulLink(base):
        def __init__(self):
            self.calls = 0

        def inverse(self, eta):
            self.calls += 1
            return super().inverse(eta) + self.calls * 1e-3

    link = StatefulLink()
    states[predictor_index] = replace(states[predictor_index], link=link)
    layout = replace(layout, predictors=tuple(states))
    family = GaussianLS()
    response = np.ones(512)
    plan = family.bind_likelihood(response, resolved_prior(np.ones(512)), COMPLETE_OBSERVATION)
    monkeypatch.setattr(policy, "AUTO_GLOBAL_MOMENT_MIN_ROWS", 1)

    def forbidden(*args, **kwargs):
        raise AssertionError("custom link entered replayable global execution")

    monkeypatch.setattr(chunks, "build_global_moment_plan", forbidden)
    original = chunks.iter_likelihood_chunks
    ranges = []

    def stream(*args, **kwargs):
        for chunk in original(*args, **kwargs):
            ranges.append((chunk.rows.start, chunk.rows.stop))
            yield chunk

    monkeypatch.setattr(chunks, "iter_likelihood_chunks", stream)
    p = layout.n_coefficients
    result = chunks.assemble_chunked_geometry(
        family,
        layout,
        response,
        plan,
        np.zeros(p),
        penalty=np.zeros((p, p)),
        chunk_size=64,
        curvature_source="observed",
    )
    assert ranges == [(start, start + 64) for start in range(0, 512, 64)]
    assert link.calls == len(ranges)
    assert np.all(np.isfinite(result.data_curvature))


def _stateful_weights(normal, calls):
    class StatefulWeights(ResolvedLikelihoodWeights):
        def take(self, indices):
            calls.append(tuple(indices.tolist()))
            return super().take(indices)

    return StatefulWeights(
        **{field.name: getattr(normal, field.name) for field in fields(ResolvedLikelihoodWeights)}
    )


@pytest.mark.parametrize("family_type", [GaussianLS, GammaLS])
def test_replay_requires_exact_resolved_weights_without_calling_take(family_type):
    layout, _, _ = _problem(("mixed", "intercept"))
    _metadata_rows(layout, 262144)
    family = family_type()
    layout = replace(
        layout,
        predictors=tuple(
            replace(state, link=spec.default_link)
            for state, spec in zip(layout.predictors, family.parameters)
        ),
    )
    normal = resolved_prior(np.ones(37))
    calls = []
    for weights, expected in ((normal, 64 << 20), (_stateful_weights(normal, calls), None)):
        plan = family.bind_likelihood(np.ones(37), weights, COMPLETE_OBSERVATION)
        assert plan.weights is weights
        assert policy.automatic_global_moment_budget(family, plan, layout) == expected
        assert calls == []


@pytest.mark.parametrize("family_type", [GaussianLS, GammaLS])
def test_stateful_weights_use_one_actual_baseline_stream(monkeypatch, family_type):
    layout, _, _ = _problem(("mixed", "intercept"))
    family = family_type()
    indices = np.resize(np.arange(37, dtype=np.intp), 512)
    layout = replace(
        layout,
        predictors=tuple(
            replace(
                state,
                design=state.design.row_subset(indices),
                offset=state.offset[indices],
                link=spec.default_link,
            )
            for state, spec in zip(layout.predictors, family.parameters)
        ),
    )
    calls = []
    weights = _stateful_weights(resolved_prior(np.ones(512)), calls)
    response = np.ones(512)
    plan = family.bind_likelihood(response, weights, COMPLETE_OBSERVATION)
    assert plan.weights is weights
    monkeypatch.setattr(policy, "AUTO_GLOBAL_MOMENT_MIN_ROWS", 1)

    def forbidden(*args, **kwargs):
        raise AssertionError("custom weights entered replayable global execution")

    monkeypatch.setattr(chunks, "build_global_moment_plan", forbidden)
    original = chunks.iter_likelihood_chunks
    ranges = []

    def stream(*args, **kwargs):
        for chunk in original(*args, **kwargs):
            ranges.append((chunk.rows.start, chunk.rows.stop))
            yield chunk

    monkeypatch.setattr(chunks, "iter_likelihood_chunks", stream)
    p = layout.n_coefficients
    result = chunks.assemble_chunked_geometry(
        family,
        layout,
        response,
        plan,
        np.zeros(p),
        penalty=np.zeros((p, p)),
        chunk_size=64,
        curvature_source="observed",
    )
    expected_ranges = [(start, start + 64) for start in range(0, 512, 64)]
    assert ranges == expected_ranges
    assert calls == [tuple(range(start, stop)) for start, stop in expected_ranges]
    assert np.all(np.isfinite(result.data_curvature))


def test_existing_reuse_registration_does_not_implicitly_authorize_replay():
    from superglm.distributional.family import _register_likelihood_reuse_contract

    class CustomFamily:
        pass

    class CustomPlan:
        pass

    _register_likelihood_reuse_contract(CustomFamily, CustomPlan, prepared_array_fields=())
    layout, _, _ = _problem()
    _metadata_rows(layout, 262144)
    assert policy.automatic_global_moment_budget(CustomFamily(), CustomPlan(), layout) is None
    with pytest.raises(ValueError, match="different likelihood reuse contract"):
        _register_likelihood_reuse_contract(
            CustomFamily, CustomPlan, prepared_array_fields=(), deterministic_chunk_replay=True
        )


@pytest.mark.parametrize("mutation", ["wide", "custom", "lossless"])
def test_global_scope_keeps_unsupported_source_layouts_on_existing_routes(mutation):
    from superglm.group_matrix import DenseGroupMatrix, SupportCompressedSSPGroupMatrix

    layout, _, _ = _problem()
    _metadata_rows(layout, 262144)
    groups = layout.predictors[0].design.group_matrices
    if mutation == "wide":
        groups[0].shape = (262144, 33)
    else:

        class CustomGroup(DenseGroupMatrix):
            pass

        index = next(
            i
            for i, group in enumerate(groups)
            if type(group)
            is (DenseGroupMatrix if mutation == "custom" else DiscretizedSSPGroupMatrix)
        )
        group = groups[index]
        replacement = object.__new__(
            CustomGroup if mutation == "custom" else SupportCompressedSSPGroupMatrix
        )
        replacement.shape = group.shape
        layout.predictors[0].design.group_matrices = (
            groups[:index] + (replacement,) + groups[index + 1 :]
        )
    family = GaussianLS()
    assert policy.automatic_global_moment_budget(family, _bound(family), layout) is None


@pytest.mark.parametrize("family_type", [GaussianLS, GammaLS])
@pytest.mark.parametrize("curvature_source", ["observed", "fisher"])
def test_real_bound_likelihood_keeps_chunks_and_penalized_endpoint(
    monkeypatch, family_type, curvature_source
):
    family = family_type()
    n = 37
    frame = as_eager_frame(
        pd.DataFrame(
            {
                "x": np.linspace(-1, 1, n),
                "s": np.cos(np.arange(n) * 0.7),
                "g": np.resize(list("abc"), n),
            }
        )
    )
    weights = resolved_prior(np.linspace(0.7, 1.4, n))
    layout = build_stacked_layout(
        compile_predictors(
            frame,
            weights,
            family.parameters,
            tuple(_predictor(spec.name, "mixed") for spec in family.parameters),
            model_discrete=True,
            n_bins_config=9,
        )
    )
    response = 1.2 + 0.6 * np.sin(np.arange(n) * 0.6)
    bound = family.bind_likelihood(response, weights, COMPLETE_OBSERVATION)
    p = layout.n_coefficients
    coefficients = np.linspace(-0.08, 0.13, p)
    penalty = np.diag(np.linspace(0, 0.2, p))
    options = dict(penalty=penalty, chunk_size=7, curvature_source=curvature_source)
    baseline = chunks.assemble_chunked_geometry(
        family,
        layout,
        response,
        bound,
        coefficients,
        small_group_panel_byte_budget=None,
        **options,
    )
    monkeypatch.setattr(chunks, "automatic_global_moment_budget", lambda *args: 64 << 20)
    original = chunks.iter_likelihood_chunks
    scores, curvature, row_sizes = [], [], []

    def stream(*args, **kwargs):
        for chunk in original(*args, **kwargs):
            row_sizes.append(len(chunk.score_eta))
            scores.append(chunk.score_eta.copy())
            curvature.append(chunk.curvature_packed.copy())
            yield chunk

    def forbidden(*args, **kwargs):
        raise AssertionError("admitted real likelihood fell back to grouped geometry")

    monkeypatch.setattr(chunks, "iter_likelihood_chunks", stream)
    monkeypatch.setattr(chunks, "GroupedGeometryAccumulator", forbidden)
    result = chunks.assemble_chunked_geometry(
        family, layout, response, bound, coefficients, **options
    )
    assert row_sizes == [7] * 5 + [2]
    _assert_geometry(
        result,
        layout,
        _literal_design(layout),
        np.vstack(scores),
        np.vstack(curvature),
        penalty,
        coefficients,
    )
    np.testing.assert_array_equal(result.penalty, penalty)
    for field in (
        "score_data",
        "score_penalized",
        "data_curvature",
        "penalized_curvature",
        "penalty",
    ):
        expected = getattr(baseline, field)
        tolerance = 64 * n * p * np.finfo(float).eps * max(1.0, np.linalg.norm(expected))
        np.testing.assert_allclose(getattr(result, field), expected, rtol=0, atol=tolerance)
