"""Automatic panels preserve bounded chunks and stored-design geometry."""

from __future__ import annotations

import hashlib
import weakref
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from superglm._frame import as_eager_frame
from superglm.distributional import _panel_policy as policy
from superglm.distributional.families.gaussian import GaussianLS
from superglm.distributional.family import COMPLETE_OBSERVATION
from superglm.distributional.layout import build_stacked_layout
from superglm.distributional.predictor import Predictor, PredictorExecutionPlan, compile_predictors
from superglm.distributional.solver import chunks
from superglm.features import Categorical, Numeric, Spline
from superglm.group_matrix import (
    CategoricalGroupMatrix,
    DenseGroupMatrix,
    DesignMatrix,
    DiscretizedSCOPGroupMatrix,
    DiscretizedSplineCategoricalGroupMatrix,
    DiscretizedSSPGroupMatrix,
    DiscretizedTensorGroupMatrix,
    FactorSmoothGroupMatrix,
    SparseGroupMatrix,
    SupportCompressedSplineCategoricalGroupMatrix,
    SupportCompressedSSPGroupMatrix,
)

from ._distributional_weights import resolved_prior
from ._gaussian_lss_oracles import gaussian_row_oracle
from .test_distributional_small_group_panel_integration import _assert_geometry, _stream


def _problem(modes=("mixed", "mixed")):
    n = 37
    x = np.linspace(-1, 1, n)
    z = np.cos(np.arange(n) * 0.7)
    frame = as_eager_frame(pd.DataFrame({"x": x, "s": z, "g": np.resize(list("abc"), n)}))
    family = GaussianLS()
    layout = build_stacked_layout(
        compile_predictors(
            frame,
            resolved_prior(np.ones(n)),
            family.parameters,
            tuple(_predictor(name, mode) for name, mode in zip(("location", "scale"), modes)),
            model_discrete=True,
            n_bins_config=9,
        )
    )
    score = np.column_stack((np.sin(np.arange(n)), np.linspace(-0.3, 0.8, n)))
    curvature = np.column_stack((x, np.cos(np.arange(n)), -x))
    return layout, score, curvature


def _predictor(name, mode):
    features = {}
    if mode in ("mixed", "numeric", "numeric_spline"):
        features["x"] = Numeric()
    if mode in ("mixed", "support", "support_category", "numeric_spline"):
        features["s"] = Spline(kind="cr", k=4)
    if mode in ("mixed", "category", "support_category"):
        features["g"] = Categorical()
    # Different intercept choices give rectangular curvature blocks without
    # relying on tiny, roundoff-sensitive raw spline values for dispatch.
    return Predictor(
        name,
        features,
        interactions=(("s", "g"),) if mode == "mixed" else (),
        intercept=not (name == "scale" and mode == "mixed"),
    )


def _assemble(monkeypatch, problem, *, before_advance=None, **kwargs):
    layout, score, curvature = problem
    monkeypatch.setattr(
        chunks,
        "iter_likelihood_chunks",
        lambda *args, **options: _stream(layout, score, curvature, before_advance=before_advance),
    )
    return chunks.assemble_chunked_geometry(
        None,
        layout,
        None,
        None,
        np.zeros(layout.n_coefficients),
        penalty=np.zeros((layout.n_coefficients, layout.n_coefficients)),
        chunk_size=8,
        curvature_source="observed",
        **kwargs,
    )


def test_default_dispatch_builds_bounded_panels_for_mixed_ordinary_layout(monkeypatch):
    problem = _problem()
    original = chunks.build_small_group_panels
    builds = []

    def record(plans, rows, *, byte_budget):
        result = original(plans, rows, byte_budget=byte_budget)
        builds.append((plans[0].design.n, byte_budget, result.workspace is not None))
        return result

    monkeypatch.setattr(chunks, "build_small_group_panels", record)
    _assemble(monkeypatch, problem)
    assert builds == [(8, 64 * 1024**2, True)] * 4 + [(5, 64 * 1024**2, True)]


@pytest.mark.parametrize(
    "budget",
    [True, False, -1, 2.5, "unknown", np.array("auto"), np.str_("auto"), np.array([1, 2])],
)
def test_invalid_explicit_budget_is_not_converted_to_automatic(monkeypatch, budget):
    with pytest.raises(ValueError, match="byte_budget must be a nonnegative integer"):
        _assemble(monkeypatch, _problem(), small_group_panel_byte_budget=budget)


@pytest.mark.parametrize(
    "mode", ["numeric", "category", "support", "support_category", "numeric_spline", "intercept"]
)
def test_unadmitted_layouts_do_not_attempt_automatic_panels(monkeypatch, mode):
    problem = _problem((mode, mode))

    def forbidden(*args, **kwargs):
        raise AssertionError("unadmitted layout attempted automatic panels")

    monkeypatch.setattr(chunks, "build_small_group_panels", forbidden)
    assert policy.automatic_small_group_panel_budget(problem[0]) is None
    _assemble(monkeypatch, problem)


@pytest.mark.parametrize("mode", ["numeric", "category", "support", "support_category"])
def test_every_predictor_with_slopes_requires_the_mixed_layout(mode):
    layout, _, _ = _problem(("mixed", mode))
    assert policy.automatic_small_group_panel_budget(layout) is None


def test_intercept_only_predictor_can_accompany_mixed_predictor(monkeypatch):
    problem = _problem(("mixed", "intercept"))
    original = chunks.build_small_group_panels
    accepted = []

    def record(*args, **kwargs):
        built = original(*args, **kwargs)
        accepted.append(built.workspace is not None)
        return built

    monkeypatch.setattr(chunks, "build_small_group_panels", record)
    _assemble(monkeypatch, problem)
    assert accepted == [True] * 5


def _with_replaced_group(layout, index, group):
    state = layout.predictors[0]
    groups = list(state.design.group_matrices)
    groups[index] = group
    design = DesignMatrix(groups, n=state.design.n, p=sum(g.shape[1] for g in groups))
    return replace(layout, predictors=(replace(state, design=design), *layout.predictors[1:]))


@pytest.mark.parametrize("width,admitted", [(32, True), (33, False)])
def test_group_width_is_a_conservative_scope_limit(width, admitted):
    layout, _, _ = _problem()
    group = DenseGroupMatrix(np.ones((layout.predictors[0].design.n, width)))
    changed = _with_replaced_group(layout, 0, group)
    assert (policy.automatic_small_group_panel_budget(changed) is not None) is admitted


@pytest.mark.parametrize(
    "kind", ["tensor", "scop", "sparse", "factor", "discrete_factor", "custom"]
)
def test_specialized_and_extension_groups_keep_their_routes(kind):
    layout, _, _ = _problem()
    n = layout.predictors[0].design.n
    basis = np.ones((3, 1))
    bins = np.arange(n, dtype=np.intp) % 3
    if kind == "tensor":
        group = DiscretizedTensorGroupMatrix(
            basis, basis, bins, bins, basis, np.ones((1, 1)), bins, tensor_id=1
        )
    elif kind == "scop":
        group = DiscretizedSCOPGroupMatrix(basis, bins)
    elif kind == "sparse":
        group = SparseGroupMatrix(sp.csr_matrix(np.ones((n, 1))))
    elif kind in ("factor", "discrete_factor"):
        discrete = kind == "discrete_factor"
        group = FactorSmoothGroupMatrix(
            basis if discrete else sp.csr_matrix(np.ones((n, 1))),
            np.zeros(n, dtype=np.intp),
            1,
            natural_map=np.ones((1, 1)),
            levels=("one",),
            repeated_penalty_components=(),
            bin_idx=bins if discrete else None,
        )
    else:

        class CustomDense(DenseGroupMatrix):
            pass

        group = CustomDense(np.ones((n, 1)))
    assert policy.automatic_small_group_panel_budget(_with_replaced_group(layout, 0, group)) is None


def test_custom_design_is_not_admitted():
    class CustomDesign(DesignMatrix):
        pass

    layout, _, _ = _problem()
    state = layout.predictors[0]
    design = CustomDesign(state.design.group_matrices, n=state.design.n, p=state.design.p)
    layout = replace(layout, predictors=(replace(state, design=design), *layout.predictors[1:]))
    assert policy.automatic_small_group_panel_budget(layout) is None


def _fingerprints(layout):
    result = []
    fields = ("M", "codes", "B_unique", "R_inv", "bin_idx", "row_idx", "bin_idx_level")
    for state in layout.predictors:
        for group in state.design.group_matrices:
            for name in fields:
                array = getattr(group, name, None)
                if isinstance(array, np.ndarray):
                    result.append(
                        (type(group), name, id(array), hashlib.sha256(array.tobytes()).digest())
                    )
    return result


def test_selector_reads_only_metadata_and_preserves_stored_arrays_and_caches(monkeypatch):
    layout, _, _ = _problem()
    before = _fingerprints(layout)
    caches = [dict(state.design.__dict__) for state in layout.predictors]

    def forbidden(*args, **kwargs):
        raise AssertionError("automatic selector inspected row data or constructed a plan")

    with monkeypatch.context() as patch:
        for name in ("array", "asarray", "all", "any", "isfinite"):
            patch.setattr(np, name, forbidden)
        for name in ("toarray", "row_subset"):
            patch.setattr(DesignMatrix, name, forbidden)
        patch.setattr(PredictorExecutionPlan, "__init__", forbidden)
        assert policy.automatic_small_group_panel_budget(layout) == 64 * 1024**2
    assert before == _fingerprints(layout)
    for state, snapshot in zip(layout.predictors, caches, strict=True):
        assert snapshot.keys() == state.design.__dict__.keys()
        assert all(value is state.design.__dict__[key] for key, value in snapshot.items())


@pytest.mark.parametrize("budget", [None, 1, 2**20])
def test_explicit_panel_overrides_bypass_automatic_admission(monkeypatch, budget):
    problem = _problem(("numeric", "numeric"))
    original = chunks.build_small_group_panels
    builds = []

    def forbidden(*args, **kwargs):
        raise AssertionError("explicit panel override consulted automatic eligibility")

    def record(*args, **kwargs):
        built = original(*args, **kwargs)
        builds.append((kwargs["byte_budget"], built.workspace is not None))
        return built

    monkeypatch.setattr(chunks, "automatic_small_group_panel_budget", forbidden)
    monkeypatch.setattr(chunks, "build_small_group_panels", record)
    _assemble(monkeypatch, problem, small_group_panel_byte_budget=budget)
    assert builds == ([] if budget is None else [(budget, budget == 2**20)] * 5)


def _literal_design(layout):
    matrices = []
    for state in layout.predictors:
        pieces = [np.ones((state.design.n, 1))] if state.intercept_index is not None else []
        for group in state.design.group_matrices:
            if type(group) is DenseGroupMatrix:
                values = group.M.copy()
            elif type(group) is CategoricalGroupMatrix:
                values = np.zeros(group.shape)
                for row, code in enumerate(group.codes):
                    if code < group.n_levels:
                        values[row, code] = 1
            elif type(group) in (DiscretizedSSPGroupMatrix, SupportCompressedSSPGroupMatrix):
                values = group.B_unique[group.bin_idx] @ group.R_inv
            else:
                assert type(group) in (
                    DiscretizedSplineCategoricalGroupMatrix,
                    SupportCompressedSplineCategoricalGroupMatrix,
                )
                values = np.zeros(group.shape)
                values[group.row_idx] = group.B_unique[group.bin_idx_level] @ group.R_inv
            pieces.append(values)
        matrices.append(np.column_stack(pieces))
    return tuple(matrices)


def _assert_signed_geometry(result, problem, literal):
    layout, score, curvature = problem
    _assert_geometry(
        result,
        layout,
        literal,
        score,
        curvature,
        np.zeros((layout.n_coefficients, layout.n_coefficients)),
        np.zeros(layout.n_coefficients),
    )


@pytest.mark.parametrize("failure", [None, "byte-budget", "numerical-domain"])
def test_automatic_panels_and_refusals_preserve_signed_stored_geometry(monkeypatch, failure):
    problem = _problem()
    if failure == "byte-budget":
        monkeypatch.setattr(policy, "AUTO_SMALL_GROUP_PANEL_BYTES", 1)
    elif failure == "numerical-domain":
        # An otherwise valid tiny numeric column is outside the panel writer's
        # reassociation envelope. Grouped assembly can still evaluate it.
        for state in problem[0].predictors:
            group = state.design.group_matrices[0]
            group.M = group.M * 2.0**-200
    literal = _literal_design(problem[0])
    assert literal[0].shape[1] != literal[1].shape[1]
    before = _fingerprints(problem[0])
    _assert_signed_geometry(_assemble(monkeypatch, problem), problem, literal)
    assert _fingerprints(problem[0]) == before


def test_automatic_budget_resolves_once_and_builder_refusals_remain_explained(monkeypatch):
    problem = _problem()
    original_policy = chunks.automatic_small_group_panel_budget
    original_builder = chunks.build_small_group_panels
    resolutions = []
    refusals = []
    monkeypatch.setattr(policy, "AUTO_SMALL_GROUP_PANEL_BYTES", 1)

    def resolve(layout):
        resolutions.append(id(layout))
        return original_policy(layout)

    def build(*args, **kwargs):
        built = original_builder(*args, **kwargs)
        refusals.append(built.reason)
        return built

    monkeypatch.setattr(chunks, "automatic_small_group_panel_budget", resolve)
    monkeypatch.setattr(chunks, "build_small_group_panels", build)
    _assemble(monkeypatch, problem)
    assert resolutions == [id(problem[0])]
    assert refusals == ["byte-budget"] * 5


def test_actual_group_domain_refusal_keeps_grouped_dispatch(monkeypatch):
    problem = _problem()
    for state in problem[0].predictors:
        group = state.design.group_matrices[0]
        group.M = group.M * 2.0**-200
    original = chunks.build_small_group_panels
    refusals = []

    def record(*args, **kwargs):
        built = original(*args, **kwargs)
        assert built.workspace is None
        refusals.append(built.reason)
        return built

    monkeypatch.setattr(chunks, "build_small_group_panels", record)
    _assemble(monkeypatch, problem)
    assert refusals == ["numerical-domain"] * 5


def test_late_channel_domain_refusal_uses_one_grouped_contribution(monkeypatch):
    problem = _problem()
    problem[2][:, 1] *= 2.0**-200
    literal = _literal_design(problem[0])
    original = PredictorExecutionPlan.cross_moment
    calls = []

    def record(self, other, weights):
        calls.append(len(weights))
        return original(self, other, weights)

    monkeypatch.setattr(PredictorExecutionPlan, "cross_moment", record)
    result = _assemble(monkeypatch, problem)
    assert calls == [8, 8, 8, 8, 5]
    left, right = literal
    weights = problem[2][:, 1]
    expected = left.T @ (weights[:, None] * right)
    scale = np.max(np.abs(left).T @ (np.abs(weights[:, None]) * np.abs(right)))
    tolerance = 32 * len(weights) * np.finfo(float).eps * scale
    np.testing.assert_allclose(
        result.data_curvature[
            problem[0].predictors[0].coefficient_slice,
            problem[0].predictors[1].coefficient_slice,
        ],
        expected,
        rtol=0,
        atol=tolerance,
    )


@pytest.mark.parametrize("failure", [None, "score", "curvature"])
def test_automatic_workspace_lifetime_ends_before_next_chunk_or_exception(monkeypatch, failure):
    problem = _problem()
    original = chunks.build_small_group_panels
    references = []
    workspaces = []

    def raises(*args, **kwargs):
        raise RuntimeError("injected assembly error")

    def build(*args, **kwargs):
        result = original(*args, **kwargs)
        workspace = result.workspace
        assert workspace is not None
        references.extend(weakref.ref(panel) for panel in workspace.panels)
        workspaces.append(workspace)
        if failure == "score":
            monkeypatch.setattr(chunks.GroupedGeometryAccumulator, "add_score", raises)
        elif failure == "curvature":
            workspace.cross_moment = raises
        return result

    def released():
        assert all(reference() is None for reference in references)
        assert all(workspace.retained_bytes == 0 for workspace in workspaces)

    monkeypatch.setattr(chunks, "build_small_group_panels", build)
    if failure:
        with pytest.raises(RuntimeError, match="injected assembly error"):
            _assemble(monkeypatch, problem, before_advance=released)
    else:
        _assemble(monkeypatch, problem, before_advance=released)
    assert references
    released()


@pytest.mark.parametrize("curvature_source", ["observed", "fisher"])
def test_real_likelihood_uses_explicit_row_bound_and_independent_gaussian_law(
    monkeypatch, curvature_source
):
    layout, _, _ = _problem()
    literal = _literal_design(layout)
    n = literal[0].shape[0]
    coefficients = np.linspace(-0.08, 0.13, layout.n_coefficients)
    eta = np.column_stack(
        [
            matrix @ coefficients[state.coefficient_slice]
            for matrix, state in zip(literal, layout.predictors)
        ]
    )
    family = GaussianLS()
    response = 0.2 + np.sin(np.arange(n) * 0.6)
    weights = np.linspace(0.7, 1.4, n)
    plan = family.bind_likelihood(response, resolved_prior(weights), COMPLETE_OBSERVATION)
    oracle = gaussian_row_oracle(
        response,
        eta[:, 0],
        family.scale_floor + np.exp(eta[:, 1]),
        weights,
        semantics="prior",
        scale_floor=family.scale_floor,
    )
    original = chunks.build_small_group_panels
    sizes = []

    def record(plans, rows, **kwargs):
        sizes.append(plans[0].design.n)
        return original(plans, rows, **kwargs)

    monkeypatch.setattr(chunks, "build_small_group_panels", record)
    penalty = np.zeros((layout.n_coefficients, layout.n_coefficients))
    result = chunks.assemble_chunked_geometry(
        family,
        layout,
        response,
        plan,
        coefficients,
        penalty=penalty,
        chunk_size=7,
        curvature_source=curvature_source,
    )
    curvature = (
        oracle.observed_link_curvature_packed
        if curvature_source == "observed"
        else oracle.fisher_link_curvature_packed
    )
    _assert_geometry(result, layout, literal, oracle.link_score, curvature, penalty, coefficients)
    assert sizes == [7] * 5 + [2]
    assert chunks.AUTO_CHUNK_SELECTOR == "distributional-auto-v1"
    assert chunks.AUTO_CHUNK_MEMORY_BYTES == 8 * 1024**2
