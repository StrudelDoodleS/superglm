"""Independent stored-design checks for optional bounded group panels."""

from __future__ import annotations

import gc
import tracemalloc
import weakref

import numpy as np
import pytest
import scipy.sparse as sp

from superglm.distributional.predictor import PredictorExecutionPlan
from superglm.group_matrix import (
    CategoricalGroupMatrix,
    DenseGroupMatrix,
    DesignMatrix,
    DiscretizedSCOPGroupMatrix,
    DiscretizedSplineCategoricalGroupMatrix,
    DiscretizedSSPGroupMatrix,
    DiscretizedTensorGroupMatrix,
    FactorSmoothGroupMatrix,
    SparseSSPGroupMatrix,
    SplineCategoricalGroupMatrix,
    SupportCompressedSplineCategoricalGroupMatrix,
    SupportCompressedSSPGroupMatrix,
)


def _plan(groups, *, intercept=True):
    n = groups[0].shape[0]
    return PredictorExecutionPlan(
        DesignMatrix(groups, n=n, p=sum(group.shape[1] for group in groups)), intercept
    )


def _problem(n=43, *, factor_basis="fs", discrete_factor=True, lossless_support=False):
    rng = np.random.default_rng(914)
    dense = rng.normal(size=(n, 2))
    codes = rng.integers(-1, 3, size=n)
    category = np.eye(4)[np.where(codes == -1, 3, codes), :3]
    support = rng.normal(size=(7, 4)) / 3
    bins = rng.integers(7, size=n)
    transform = rng.normal(size=(4, 2)) / 3
    sparse_raw = rng.normal(size=(n, 4)) / 3
    sparse_raw[np.abs(sparse_raw) < 0.2] = 0
    level_rows = np.arange(n, dtype=np.intp)[::3]
    grouped = np.zeros((n, 2))
    grouped[level_rows] = sparse_raw[level_rows] @ transform
    discrete_grouped = np.zeros((n, 2))
    discrete_grouped[level_rows] = support[bins[level_rows]] @ transform
    factor_codes = rng.integers(4, size=n)
    factor_map = rng.normal(size=(4, 2)) / 3
    factor_raw = support[bins] if discrete_factor else sparse_raw
    natural = factor_raw @ factor_map
    coefficient_levels = 4 if factor_basis == "fs" else 3
    factor = np.zeros((n, coefficient_levels * 2))
    for row, code in enumerate(factor_codes):
        if code < coefficient_levels:
            factor[row, code * 2 : code * 2 + 2] = natural[row]
        else:
            factor[row] = np.tile(-natural[row], coefficient_levels)
    factor_group = FactorSmoothGroupMatrix(
        support if discrete_factor else sp.csr_matrix(sparse_raw),
        factor_codes,
        4,
        natural_map=factor_map,
        levels=tuple("abcd"),
        repeated_penalty_components=(),
        factor_basis=factor_basis,
        bin_idx=bins if discrete_factor else None,
    )
    support_type = (
        SupportCompressedSSPGroupMatrix if lossless_support else DiscretizedSSPGroupMatrix
    )
    category_type = (
        SupportCompressedSplineCategoricalGroupMatrix
        if lossless_support
        else DiscretizedSplineCategoricalGroupMatrix
    )
    left = [
        DenseGroupMatrix(dense),
        CategoricalGroupMatrix(codes, 3),
        support_type(support, transform, bins),
        factor_group,
    ]
    right = [
        SparseSSPGroupMatrix(sp.csr_matrix(sparse_raw), transform),
        SplineCategoricalGroupMatrix(sp.csr_matrix(sparse_raw), transform, level_rows),
        category_type(support, transform, bins, level_rows, n_rows=n),
        DenseGroupMatrix(dense[:, :1]),
    ]
    return (
        (_plan(left), _plan(right, intercept=False)),
        (
            np.column_stack([np.ones(n), dense, category, support[bins] @ transform, factor]),
            np.column_stack([sparse_raw @ transform, grouped, discrete_grouped, dense[:, :1]]),
        ),
    )


def _build(*args, **kwargs):
    from superglm.distributional.solver._small_group_panels import build_small_group_panels

    return build_small_group_panels(*args, **kwargs)


def _warm_panel_range_kernel():
    from superglm.distributional.solver._small_group_panels import (
        _in_range,
        _warmup_small_group_panels,
    )

    # Exclude one-time predicate/writer compilation and cache loading from
    # workspace tracing. All panel construction, optional support tables,
    # renderer buffers and contraction allocations remain inside the trace.
    assert _in_range(np.ones((2, 2), dtype=np.float64))
    _warmup_small_group_panels()


def _assert_cross(actual, left, right, weights):
    expected = left.T @ (weights[:, None] * right)
    scale = np.max(np.abs(left).T @ (np.abs(weights)[:, None] * np.abs(right)), initial=0)
    gamma = 32 * np.finfo(float).eps * max(len(weights), left.shape[1], right.shape[1], 1)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=gamma * scale)


def _assert_reconstruction(actual, expected):
    # Fixtures use well-scaled four-column basis maps. This dimension/norm
    # backward-error allowance covers their dense and sparse accumulation.
    scale = np.linalg.norm(expected, ord=np.inf) if expected.size else 0
    gamma = 8 * np.finfo(expected.dtype).eps * max(*expected.shape, 4)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=gamma * scale)


@pytest.mark.parametrize("factor_basis", ["fs", "sz"])
@pytest.mark.parametrize("discrete_factor", [False, True])
@pytest.mark.parametrize("lossless_support", [False, True])
def test_panels_match_signed_rectangular_stored_design_without_toarray(
    monkeypatch, factor_basis, discrete_factor, lossless_support
):
    plans, literal = _problem(
        factor_basis=factor_basis,
        discrete_factor=discrete_factor,
        lossless_support=lossless_support,
    )
    rows = np.array([42, 0, 7, 7, 4, 18, 2], dtype=np.intp)

    def forbidden(*args, **kwargs):
        raise AssertionError("compressed toarray must not render a panel")

    for cls in (
        DiscretizedSSPGroupMatrix,
        SparseSSPGroupMatrix,
        SplineCategoricalGroupMatrix,
        DiscretizedSplineCategoricalGroupMatrix,
        FactorSmoothGroupMatrix,
        SupportCompressedSSPGroupMatrix,
        SupportCompressedSplineCategoricalGroupMatrix,
    ):
        monkeypatch.setattr(cls, "toarray", forbidden)
    result = _build(plans, rows, byte_budget=2**20)
    assert result.workspace is not None, result.reason
    with result.workspace as workspace:
        for actual, expected in zip(workspace.panels, literal, strict=True):
            _assert_reconstruction(actual, expected[rows])
            assert not actual.flags.writeable
        np.testing.assert_array_equal(workspace.panels[0][:, 3:6], literal[0][rows, 3:6])
        weights = np.array([-0.7, 0, 1.4, -0.2, 0.4, 0.9, -1.1])
        _assert_cross(
            workspace.cross_moment(0, 1, weights), literal[0][rows], literal[1][rows], weights
        )
        _assert_cross(
            workspace.cross_moment(1, 0, weights), literal[1][rows], literal[0][rows], weights
        )


def test_panels_reuse_rows_across_channels_and_cover_nondivisor_tiles():
    plans, literal = _problem()
    sums = [np.zeros((left.shape[1], right.shape[1])) for left, right in [literal, literal]]
    channels = [np.linspace(-0.8, 1.1, 43), np.cos(np.arange(43))]
    for start in range(0, 43, 11):
        stop = min(start + 11, 43)
        result = _build(plans, slice(start, stop), byte_budget=2**20)
        assert result.workspace is not None
        with result.workspace as workspace:
            identities = tuple(map(id, workspace.panels))
            for index, weights in enumerate(channels):
                sums[index] += workspace.cross_moment(0, 1, weights[start:stop])
                assert tuple(map(id, workspace.panels)) == identities
    for actual, weights in zip(sums, channels, strict=True):
        _assert_cross(actual, *literal, weights)


def test_explicit_group_selection_preserves_global_columns():
    plans, literal = _problem()
    result = _build(plans, slice(0, 9), byte_budget=2**20, group_indices=((0, 2), (1, 3)))
    assert result.workspace is not None
    with result.workspace as workspace:
        np.testing.assert_array_equal(workspace.column_indices[0], [0, 1, 2, 6, 7])
        np.testing.assert_array_equal(workspace.column_indices[1], [2, 3, 6])
        for panel, expected, columns in zip(
            workspace.panels, literal, workspace.column_indices, strict=True
        ):
            _assert_reconstruction(panel, expected[:9, columns])


def test_workspace_refuses_budget_before_rendering_and_releases_arrays(monkeypatch):
    import superglm.distributional.solver._small_group_panels as panels

    plans, _ = _problem()
    result = _build(plans, slice(0, 43), byte_budget=2**20)
    assert result.workspace is not None
    workspace = result.workspace
    references = [weakref.ref(panel) for panel in workspace.panels]
    workspace.close()
    gc.collect()
    assert all(reference() is None for reference in references)
    with pytest.raises(RuntimeError, match="closed"):
        workspace.cross_moment(0, 1, np.ones(43))

    def forbidden(*args, **kwargs):
        raise AssertionError("budget refusal must precede row rendering")

    monkeypatch.setattr(panels, "_render_group", forbidden)
    refused = _build(plans, slice(0, 43), byte_budget=128)
    assert refused.workspace is None
    assert refused.reason == "byte-budget"
    assert refused.estimated_peak_bytes > 128


def test_peak_and_retained_workspace_follow_tile_size_not_source_rows():
    _warm_panel_range_kernel()
    retained = []
    for n in (500, 50_000):
        x = np.linspace(-0.8, 0.9, n)
        plans = (_plan([DenseGroupMatrix(np.column_stack([x, x * x]))]),)
        tracemalloc.start()
        try:
            result = _build(plans, slice(0, 127), byte_budget=2**20)
            assert result.workspace is not None
            with result.workspace as workspace:
                block = workspace.cross_moment(0, 0, np.linspace(-0.5, 1.1, 127))
                retained.append(workspace.retained_bytes)
                _, peak = tracemalloc.get_traced_memory()
                assert block.shape == (3, 3)
                assert peak <= result.estimated_peak_bytes
        finally:
            tracemalloc.stop()
    assert retained[0] == retained[1]


@pytest.mark.parametrize("kind", ["tensor", "scop", "unknown", "histogram", "lossless-histogram"])
def test_specialized_and_unknown_groups_refuse_panel_dispatch(kind):
    bins = np.array([0, 1, 0, 1], dtype=np.intp)
    if kind == "tensor":
        group = DiscretizedTensorGroupMatrix(
            np.ones((2, 1)),
            np.ones((2, 1)),
            bins,
            bins,
            np.ones((4, 1)),
            np.ones((1, 1)),
            bins * 3,
            tensor_id=4,
        )
    elif kind == "scop":
        group = DiscretizedSCOPGroupMatrix(np.ones((2, 1)), bins)
    elif kind in ("histogram", "lossless-histogram"):
        support_type = (
            DiscretizedSSPGroupMatrix if kind == "histogram" else SupportCompressedSSPGroupMatrix
        )
        group = support_type(np.ones((2, 1)), np.ones((1, 1)), bins)
    else:

        class CustomDense(DenseGroupMatrix):
            pass

        group = CustomDense(np.ones((4, 1)))
    result = _build((_plan([group]),), slice(0, 4), byte_budget=2**20)
    assert result.workspace is None
    assert result.reason in {
        "specialized-group",
        "unsupported-group",
        "specialized-histogram-layout",
    }


def test_panel_numerical_domain_refuses_unsafe_reassociation():
    plans, _ = _problem()
    result = _build(plans, slice(0, 4), byte_budget=2**20)
    assert result.workspace is not None
    with result.workspace as workspace:
        assert workspace.cross_moment(0, 1, np.full(4, 1e200)) is None
        assert workspace.cross_moment(0, 1, np.ones(4, dtype=np.float32)) is None
        with pytest.raises(ValueError, match="shape"):
            workspace.cross_moment(0, 1, np.ones(3))
    extreme = _plan([DenseGroupMatrix(np.full((4, 1), 1e200))])
    refused = _build((extreme,), slice(0, 4), byte_budget=2**20)
    assert refused.workspace is None
    assert refused.reason == "numerical-domain"


@pytest.mark.parametrize("raw, mapping", [(1e200, 1e-200), (1e-200, 1e200)])
def test_raw_factor_range_refuses_even_when_rendered_design_is_moderate(raw, mapping):
    # A guard on only the rendered design misses these unsafe reassociations.
    groups = [
        DenseGroupMatrix(np.ones((4, 1))),
        DiscretizedSSPGroupMatrix(
            np.full((2, 1), raw), np.full((1, 1), mapping), np.array([0, 1, 0, 1], dtype=np.intp)
        ),
    ]
    result = _build((_plan(groups),), slice(0, 4), byte_budget=2**20)
    assert result.workspace is None
    assert result.reason == "numerical-domain"


def test_signed_oracle_detects_absolute_weight_mutation(monkeypatch):
    import superglm.distributional.solver._small_group_panels as panels

    original = panels.SmallGroupPanelWorkspace.cross_moment

    def mutated(self, left, right, weights):
        return original(self, left, right, np.abs(weights))

    monkeypatch.setattr(panels.SmallGroupPanelWorkspace, "cross_moment", mutated)
    plans, literal = _problem()
    rows = np.array([7, 7, 18, 2], dtype=np.intp)
    result = _build(plans, rows, byte_budget=2**20)
    with result.workspace as workspace:
        weights = np.array([-1.0, 0.2, -0.8, 0.4])
        with pytest.raises(AssertionError):
            _assert_cross(
                workspace.cross_moment(0, 1, weights), literal[0][rows], literal[1][rows], weights
            )


def test_numerical_checks_only_visit_selected_source_rows():
    dense = np.full((50_000, 2), np.nan)
    dense[3:9] = 0.5
    basis = np.full((50_000, 3), np.nan)
    basis[5:8] = 0.25
    bins = np.full(50_000, 6, dtype=np.intp)
    plans = (_plan([DenseGroupMatrix(dense), DiscretizedSSPGroupMatrix(basis, np.eye(3), bins)]),)
    result = _build(plans, slice(3, 9), byte_budget=2**20)
    assert result.workspace is not None
    result.workspace.close()


@pytest.mark.parametrize("kind", ["exact", "discrete"])
def test_unsorted_spline_category_refuses_without_sorting_source(kind):
    basis = np.arange(12, dtype=float).reshape(4, 3) / 12
    if kind == "exact":
        group = SplineCategoricalGroupMatrix(
            sp.csr_matrix(basis), np.eye(3), np.array([3, 0], dtype=np.intp)
        )
    else:
        group = DiscretizedSplineCategoricalGroupMatrix(
            basis, np.eye(3), np.arange(4, dtype=np.intp), np.array([3, 0], dtype=np.intp), n_rows=4
        )
    result = _build((_plan([group]),), slice(0, 4), byte_budget=2**20)
    assert result.workspace is None
    assert result.reason == "unsupported-group"
    assert group._sorted_rows is None


def test_empty_rows_and_exception_exit_release_workspace():
    plans, _ = _problem()
    result = _build(plans, slice(0, 0), byte_budget=2**20)
    workspace = result.workspace
    with pytest.raises(LookupError):
        with workspace:
            cross = workspace.cross_moment(0, 1, np.empty(0))
            np.testing.assert_array_equal(cross, np.zeros_like(cross))
            raise LookupError("caller error")
    assert workspace.retained_bytes == 0
    assert workspace.panels == ()


def test_csr_rendering_does_not_populate_borrowed_lazy_flags():
    group = SparseSSPGroupMatrix(sp.eye(8, format="csr"), np.eye(8))
    group.B.__dict__.pop("_has_canonical_format", None)
    group.B.__dict__.pop("_has_sorted_indices", None)
    before = dict(group.B.__dict__)
    result = _build((_plan([group]),), slice(0, 4), byte_budget=2**20)
    assert result.workspace is not None
    result.workspace.close()
    assert group.B.__dict__.keys() == before.keys()
    assert all(group.B.__dict__[key] is value for key, value in before.items())


def test_selected_ordinary_groups_can_exclude_specialized_groups():
    bins = np.array([0, 1, 0, 1], dtype=np.intp)
    plans = (
        _plan(
            [
                DenseGroupMatrix(np.arange(4, dtype=float)[:, None]),
                DiscretizedSCOPGroupMatrix(np.ones((2, 1)), bins),
                DenseGroupMatrix(np.full((4, 1), 0.5)),
            ]
        ),
    )
    result = _build(plans, slice(0, 4), byte_budget=2**20, group_indices=((0, 2),))
    assert result.workspace is not None
    with result.workspace as workspace:
        np.testing.assert_array_equal(workspace.column_indices[0], [0, 1, 3])
        np.testing.assert_array_equal(
            workspace.panels[0], np.column_stack([np.ones(4), np.arange(4), np.full(4, 0.5)])
        )


@pytest.mark.parametrize("factor_basis", ["fs", "sz"])
def test_mixed_renderer_peak_is_covered_for_small_and_large_sources(factor_basis):
    _warm_panel_range_kernel()
    retained = []
    for source_rows in (500, 50_000):
        plans, _ = _problem(source_rows, factor_basis=factor_basis, discrete_factor=False)
        tracemalloc.start()
        try:
            result = _build(plans, slice(1, 128), byte_budget=2**20)
            assert result.workspace is not None
            with result.workspace as workspace:
                moment = workspace.cross_moment(0, 1, np.linspace(-0.5, 1.1, 127))
                retained.append(workspace.retained_bytes)
                _, peak = tracemalloc.get_traced_memory()
                assert moment.shape == tuple(panel.shape[1] for panel in workspace.panels)
                assert peak <= result.estimated_peak_bytes
        finally:
            tracemalloc.stop()
    assert retained[0] == retained[1]


def test_owned_csr_selection_refuses_duplicate_storage():
    # Duplicate storage can invalidate both raw-width memory bounds and the
    # sparse Gram's triangular accumulation assumptions. Do not normalize it.
    basis = sp.csr_matrix(
        (np.array([0.2, 0.3]), np.array([0, 0]), np.array([0, 2, 2])), shape=(2, 1)
    )
    group = SparseSSPGroupMatrix(basis, np.ones((1, 1)))
    result = _build((_plan([group]),), slice(0, 2), byte_budget=2**20)
    assert result.workspace is None
    assert result.reason == "unsupported-group"
    np.testing.assert_array_equal(basis.data, [0.2, 0.3])


def test_selected_csr_cache_incoherence_refuses_but_unselected_rows_are_not_scanned():
    group = SparseSSPGroupMatrix(sp.csr_matrix(np.ones((12, 2))), np.eye(2))
    plans = (_plan([group]),)
    group.B.data[-2:] *= 2
    selected = _build(plans, slice(0, 4), byte_budget=2**20)
    assert selected.workspace is not None
    selected.workspace.close()
    refused = _build(plans, slice(8, 12), byte_budget=2**20)
    assert refused.workspace is None
    assert refused.reason == "unsupported-group"


def test_materialized_spline_category_gram_cache_conservatively_refuses():
    group = SplineCategoricalGroupMatrix(
        sp.csr_matrix(np.ones((4, 2))), np.eye(2), np.arange(4, dtype=np.intp)
    )
    group._dense_level = np.ones((4, 2))
    result = _build((_plan([group]),), slice(0, 4), byte_budget=2**20)
    assert result.workspace is None
    assert result.reason == "unsupported-group"


def test_nested_csr_array_subclass_refuses_before_custom_hooks():
    class CustomArray(np.ndarray):
        def astype(self, *args, **kwargs):
            raise AssertionError("custom CSR buffer hooks must not run")

    group = SparseSSPGroupMatrix(sp.eye(4, format="csr"), np.eye(4))
    group.B.indices = group.B.indices.view(CustomArray)
    result = _build((_plan([group]),), slice(0, 4), byte_budget=2**20)
    assert result.workspace is None
    assert result.reason == "unsupported-group"


@pytest.mark.parametrize("source", ["dense", "support", "bins", "codes", "level-rows"])
def test_borrowed_array_subclass_refuses_before_gather(source):
    class CustomArray(np.ndarray):
        def __getitem__(self, key):
            raise AssertionError("custom source indexing must not run")

    plans, _ = _problem()
    if source == "dense":
        group, attribute = plans[0].design.group_matrices[0], "M"
    elif source == "support":
        group, attribute = plans[0].design.group_matrices[2], "B_unique"
    elif source == "bins":
        group, attribute = plans[0].design.group_matrices[2], "bin_idx"
    elif source == "codes":
        group, attribute = plans[0].design.group_matrices[1], "codes"
    else:
        group, attribute = plans[1].design.group_matrices[1], "row_idx"
    setattr(group, attribute, getattr(group, attribute).view(CustomArray))
    result = _build(plans, slice(0, 4), byte_budget=2**20)
    assert result.workspace is None
    assert result.reason == "unsupported-group"


@pytest.mark.parametrize(
    "index_dtype, pointer_dtype",
    [
        (np.dtype(np.int32), np.dtype(np.int64)),
        (np.dtype(np.int64), np.dtype(np.int32)),
        (np.dtype(np.int16), np.dtype(np.int16)),
        (np.dtype(np.int32).newbyteorder("S"), np.dtype(np.int32).newbyteorder("S")),
    ],
)
def test_csr_index_conversion_refuses_before_source_slicing(
    monkeypatch, index_dtype, pointer_dtype
):
    group = SparseSSPGroupMatrix(sp.eye(8, format="csr"), np.eye(8))
    group.B.indices = group.B.indices.astype(index_dtype)
    group.B.indptr = group.B.indptr.astype(pointer_dtype)
    group._indices = group.B.indices
    group._indptr = group.B.indptr

    def forbidden(*args, **kwargs):
        raise AssertionError("refusal must precede source-sized index conversion")

    monkeypatch.setattr(sp.csr_matrix, "__getitem__", forbidden)
    result = _build((_plan([group]),), slice(0, 2), byte_budget=2**20)
    assert result.workspace is None
    assert result.reason == "unsupported-group"


@pytest.mark.parametrize("layout", ["vector", "vector-strided", "vector-reversed", "C", "F", "A"])
@pytest.mark.parametrize("readonly", [False, True])
def test_in_range_matches_predicate_on_edges_and_strided_arrays(layout, readonly):
    from superglm.distributional.solver._small_group_panels import _in_range

    edges = [
        0.0,
        -0.0,
        2.0**-128,
        -(2.0**-128),
        2.0**128,
        -(2.0**128),
        np.nextafter(2.0**-128, 0.0),
        np.nextafter(2.0**128, np.inf),
        np.nextafter(2.0**-128, np.inf),
        np.nextafter(2.0**128, 0.0),
        np.nextafter(0.0, 1.0),
        np.nan,
        np.inf,
        -np.inf,
    ]
    for edge in edges:
        base = np.full((6, 8), 0.5)
        base[2, 4] = edge
        values = {
            "vector": base.ravel(),
            "vector-strided": base.ravel()[::2],
            "vector-reversed": base.ravel()[::-1],
            "C": base,
            "F": np.asfortranarray(base),
            "A": base[::2, ::2],
        }[layout]
        if readonly:
            values.flags.writeable = False
        magnitude = np.abs(values)
        expected = bool(
            np.all((magnitude == 0) | ((magnitude >= 2.0**-128) & (magnitude <= 2.0**128)))
        )
        assert _in_range(values) == expected


def test_in_range_dispatches_rank_one_and_two_without_copying(monkeypatch):
    import superglm.distributional.solver._small_group_panels as panels

    captured = []

    def compiled(values):
        captured.append(values)
        return True

    monkeypatch.setattr(panels, "_tensor_operand_in_reassociation_range", compiled)
    for values in (np.ones(8)[::2], np.ones((4, 6))[:, ::2]):
        values.flags.writeable = False
        assert panels._in_range(values)
        assert captured[-1].ndim == 2
        assert np.shares_memory(captured[-1], values)
        assert not captured[-1].flags.writeable
    assert len(captured) == 2


def test_in_range_refuses_types_before_views_or_compiled_dispatch(monkeypatch):
    import superglm.distributional.solver._small_group_panels as panels

    class CustomArray(np.ndarray):
        def __getitem__(self, key):
            raise AssertionError("subclass view must not be constructed")

    def forbidden(values):
        raise AssertionError("ineligible dtype/type reached compiled predicate")

    monkeypatch.setattr(panels, "_tensor_operand_in_reassociation_range", forbidden)
    for values in (
        [1.0],
        np.ones(2, dtype=np.float32),
        np.ones(2, dtype=np.int64),
        np.ones(2, dtype=np.dtype(float).newbyteorder("S")),
        np.ones(2).view(CustomArray),
    ):
        assert panels._in_range(values) is False


@pytest.mark.parametrize("shape", [(0,), (0, 3), (3, 0), (), (2, 3, 4)])
def test_in_range_empty_and_generic_rank_semantics(shape):
    from superglm.distributional.solver._small_group_panels import _in_range

    values = np.zeros(shape)
    assert _in_range(values)
    if values.size:
        values[...] = np.inf
        assert not _in_range(values)


def test_small_support_transform_dispatch_is_bounded_and_optional(monkeypatch):
    import superglm.distributional.solver._small_group_panels as panels

    plans, literal = _problem(n=43)
    seen = []
    original = panels._support_transform

    def observed(basis, transform):
        seen.append(basis.shape[0])
        return original(basis, transform)

    monkeypatch.setattr(panels, "_support_transform", observed)
    cap = panels._SUPPORT_TABLE_MAX_CELLS
    monkeypatch.setattr(panels, "_SUPPORT_TABLE_MAX_CELLS", 0)
    fallback = _build(plans, slice(0, 43), byte_budget=2**20)
    assert fallback.workspace is not None
    baseline_estimate = fallback.estimated_peak_bytes
    with fallback.workspace as workspace:
        for actual, expected in zip(workspace.panels, literal, strict=True):
            _assert_reconstruction(actual, expected)
    assert 43 in seen
    seen.clear()
    monkeypatch.setattr(panels, "_SUPPORT_TABLE_MAX_CELLS", cap)
    optimized = _build(plans, slice(0, 43), byte_budget=2**20)
    assert optimized.workspace is not None
    assert optimized.estimated_peak_bytes > baseline_estimate
    with optimized.workspace as workspace:
        for actual, expected in zip(workspace.panels, literal, strict=True):
            _assert_reconstruction(actual, expected)
    assert seen == [7, 7, 7]
    seen.clear()
    tight = _build(plans, slice(0, 43), byte_budget=baseline_estimate)
    assert tight.workspace is not None
    assert tight.estimated_peak_bytes == baseline_estimate
    tight.workspace.close()
    assert 43 in seen


def test_unsafe_unused_support_falls_back_without_refusing_selected_rows(monkeypatch):
    import superglm.distributional.solver._small_group_panels as panels

    basis = np.array([[0.25, 0.5], [np.nan, 1e200]])
    transform = np.eye(2)
    bins = np.zeros(11, dtype=np.intp)
    seen = []
    original = panels._support_transform

    def observed(values, mapping):
        seen.append(values.shape[0])
        return original(values, mapping)

    monkeypatch.setattr(panels, "_support_transform", observed)
    plans = (
        _plan(
            [DenseGroupMatrix(np.ones((11, 1))), DiscretizedSSPGroupMatrix(basis, transform, bins)]
        ),
    )
    result = _build(plans, slice(0, 11), byte_budget=2**20)
    assert result.workspace is not None
    result.workspace.close()
    assert seen == [11]


def test_unused_transformed_support_does_not_broaden_output_domain_refusal():
    basis = np.array([[0.25], [2.0**128]])
    plans = (
        _plan(
            [
                DenseGroupMatrix(np.ones((11, 1))),
                DiscretizedSSPGroupMatrix(basis, np.array([[2.0]]), np.zeros(11, dtype=np.intp)),
            ]
        ),
    )
    result = _build(plans, slice(0, 11), byte_budget=2**20)
    assert result.workspace is not None
    with result.workspace as workspace:
        np.testing.assert_array_equal(workspace.panels[0][:, -1], np.full(11, 0.5))


def test_every_renderer_certifies_output_without_a_full_panel_postpass(monkeypatch):
    import superglm.distributional.solver._small_group_panels as panels

    plans, literal = _problem(n=43, factor_basis="sz", discrete_factor=False)
    full_shapes = {matrix.shape for matrix in literal}
    original = panels._require_range

    def no_postpass(values):
        assert values.shape not in full_shapes, "redundant full-panel range pass"
        return original(values)

    monkeypatch.setattr(panels, "_require_range", no_postpass)
    result = _build(plans, slice(0, 43), byte_budget=2**20)
    assert result.workspace is not None
    with result.workspace as workspace:
        for actual, expected in zip(workspace.panels, literal, strict=True):
            _assert_reconstruction(actual, expected)


@pytest.mark.parametrize("writer", ["copy", "gather", "scatter", "factor"])
@pytest.mark.parametrize("layout", ["C", "F", "A"])
@pytest.mark.parametrize("readonly", [False, True])
def test_checked_writers_preserve_range_predicate_and_strided_outputs(writer, layout, readonly):
    import superglm.distributional.solver._small_group_panels as panels

    edges = [
        0.0,
        -0.0,
        2.0**-128,
        -(2.0**-128),
        2.0**128,
        -(2.0**128),
        np.nextafter(2.0**-128, 0.0),
        np.nextafter(2.0**128, np.inf),
        np.nextafter(0.0, 1.0),
        np.nan,
        np.inf,
        -np.inf,
    ]
    for edge in edges:
        if layout == "A":
            values = np.full((4, 6), 0.25)[:, ::2]
        else:
            values = np.full((4, 3), 0.25, order=layout)
        values[2, 1] = edge
        values.flags.writeable = not readonly
        out = np.full((6, 12), np.nan)[:, ::2]
        expected_valid = edge == 0 or 2.0**-128 <= abs(edge) <= 2.0**128
        if writer == "copy":
            target = out[:4, :3]
            valid = panels._checked_copy(values, target)
            expected = values
        elif writer == "gather":
            target = out[:4, :3]
            rows = np.array([2, 0, 2, 3], dtype=np.intp)
            valid = panels._checked_gather(values, rows, target)
            expected = values[rows]
        elif writer == "scatter":
            target = out[:, :3]
            target.fill(0)
            rows = np.array([5, 1, 3, 0], dtype=np.intp)
            valid = panels._checked_scatter(values, rows, target)
            expected = np.zeros((6, 3))
            expected[rows] = values
        else:
            target = out[:4]
            codes = np.array([0, 1, 2, 1], dtype=np.intp)
            valid = panels._checked_factor_scatter(values, codes, target, 3, 2, True)
            expected = np.zeros((4, 6))
            for row, code in enumerate(codes):
                if code < 2:
                    expected[row, code * 3 : code * 3 + 3] = values[row]
                else:
                    expected[row] = np.tile(-values[row], 2)
        assert bool(valid) == bool(expected_valid)
        if valid:
            np.testing.assert_array_equal(target, expected)
            np.testing.assert_array_equal(np.signbit(target), np.signbit(expected))


def test_failed_checked_write_does_not_publish_or_retain_partial_panel(monkeypatch):
    import superglm.distributional.solver._small_group_panels as panels

    references = []

    def failed(values, rows, out):
        out[0, 0] = 0.25
        references.append(weakref.ref(out.base))
        return False

    monkeypatch.setattr(panels, "_checked_gather", failed)
    plans, _ = _problem()
    result = _build(plans, slice(0, 7), byte_budget=2**20)
    assert result.workspace is None
    assert result.reason == "numerical-domain"
    gc.collect()
    assert all(reference() is None for reference in references)


@pytest.mark.parametrize("factor_basis", ["fs", "sz"])
def test_all_panel_cells_are_initialized_before_publication(monkeypatch, factor_basis):
    plans, literal = _problem(factor_basis=factor_basis)
    original = np.empty

    def poisoned(*args, **kwargs):
        result = original(*args, **kwargs)
        if result.dtype == np.float64:
            result.fill(np.nan)
        return result

    monkeypatch.setattr(np, "empty", poisoned)
    result = _build(plans, slice(0, 43), byte_budget=2**20)
    assert result.workspace is not None
    with result.workspace as workspace:
        for actual, expected in zip(workspace.panels, literal, strict=True):
            _assert_reconstruction(actual, expected)


def test_support_tables_are_released_before_the_next_group_render(monkeypatch):
    import superglm.distributional.solver._small_group_panels as panels

    references = []
    original = panels._support_transform

    def observed(basis, transform):
        assert all(reference() is None for reference in references)
        result = original(basis, transform)
        references.append(weakref.ref(result))
        return result

    monkeypatch.setattr(panels, "_support_transform", observed)
    plans, _ = _problem()
    result = _build(plans, slice(0, 43), byte_budget=2**20)
    assert result.workspace is not None
    assert len(references) == 3
    assert all(reference() is None for reference in references)
    result.workspace.close()


@pytest.mark.parametrize("table_enabled", [False, True])
def test_support_indices_preserve_numpy_negative_index_semantics(monkeypatch, table_enabled):
    import superglm.distributional.solver._small_group_panels as panels

    if not table_enabled:
        monkeypatch.setattr(panels, "_SUPPORT_TABLE_MAX_CELLS", 0)
    basis = np.arange(8, dtype=float).reshape(4, 2) / 8
    bins = np.array([-1, -4, 2, -1, 0, 3, -2], dtype=np.intp)
    plans = (
        _plan(
            [DenseGroupMatrix(np.ones((7, 1))), DiscretizedSSPGroupMatrix(basis, np.eye(2), bins)]
        ),
    )
    result = _build(plans, slice(0, 7), byte_budget=2**20)
    assert result.workspace is not None
    with result.workspace as workspace:
        np.testing.assert_array_equal(workspace.panels[0][:, -2:], basis[bins])


@pytest.mark.parametrize("index", [-5, 4, np.iinfo(np.intp).max])
def test_invalid_support_index_is_rejected_before_unchecked_writer(monkeypatch, index):
    import superglm.distributional.solver._small_group_panels as panels

    def forbidden(*args):
        raise AssertionError("invalid support index reached unchecked writer")

    monkeypatch.setattr(panels, "_checked_gather", forbidden)
    with pytest.raises(IndexError, match="outside"):
        panels._support_product(
            np.ones((4, 2)),
            np.array([index], dtype=np.intp),
            np.eye(2),
            table_byte_allowance=2**20,
            out=np.empty((1, 2)),
        )


def test_large_support_is_not_scanned_when_the_table_size_gate_fails(monkeypatch):
    import superglm.distributional.solver._small_group_panels as panels

    basis = np.ones((5000, 2))
    original = panels._in_range

    def bounded(values):
        assert values is not basis, "large support was scanned before the size gate"
        return original(values)

    monkeypatch.setattr(panels, "_in_range", bounded)
    plans = (
        _plan(
            [
                DenseGroupMatrix(np.ones((17, 1))),
                DiscretizedSSPGroupMatrix(basis, np.eye(2), np.zeros(17, dtype=np.intp)),
            ]
        ),
    )
    result = _build(plans, slice(0, 17), byte_budget=2**20)
    assert result.workspace is not None
    result.workspace.close()


def test_table_reassociation_has_a_raw_product_backward_error_bound():
    rng = np.random.default_rng(894)
    basis = 1 + rng.normal(scale=1e-12, size=(7, 8))
    transform = np.tile(np.array([1.0, -1.0])[:, None], (4, 3)) * 1e4
    bins = rng.integers(7, size=103)
    plans = (
        _plan(
            [DenseGroupMatrix(np.ones((103, 1))), DiscretizedSSPGroupMatrix(basis, transform, bins)]
        ),
    )
    expected = basis[bins].astype(np.longdouble) @ transform.astype(np.longdouble)
    scale = np.abs(basis[bins]) @ np.abs(transform)
    bound = 16 * np.finfo(float).eps * basis.shape[1] * np.max(scale)
    result = _build(plans, slice(0, 103), byte_budget=2**20)
    assert result.workspace is not None
    with result.workspace as workspace:
        np.testing.assert_allclose(workspace.panels[0][:, -3:], expected, rtol=0, atol=bound)


def test_failed_support_table_write_releases_its_owned_table(monkeypatch):
    import superglm.distributional.solver._small_group_panels as panels

    references = []
    original = panels._checked_gather

    def failed_table(values, rows, out):
        if values.shape[0] == 7:
            references.append(weakref.ref(values))
            out[0, 0] = 0.5
            return False
        return original(values, rows, out)

    monkeypatch.setattr(panels, "_checked_gather", failed_table)
    plans, _ = _problem()
    result = _build(plans, slice(0, 43), byte_budget=2**20)
    assert result.workspace is None
    assert result.reason == "numerical-domain"
    assert references
    gc.collect()
    assert all(reference() is None for reference in references)


def test_support_table_rebuilds_after_source_and_assignment_changes():
    basis = np.array([[0.25, 0.5], [0.5, 0.75]])
    transform = np.eye(2)
    bins = np.zeros(11, dtype=np.intp)
    group = DiscretizedSSPGroupMatrix(basis, transform, bins)
    plans = (_plan([DenseGroupMatrix(np.ones((11, 1))), group]),)
    first = _build(plans, slice(0, 11), byte_budget=2**20)
    assert first.workspace is not None
    first.workspace.close()
    basis *= 2
    transform *= 0.5
    bins[:] = 1
    second = _build(plans, slice(0, 11), byte_budget=2**20)
    assert second.workspace is not None
    with second.workspace as workspace:
        np.testing.assert_array_equal(workspace.panels[0][:, -2:], basis[bins] @ transform)


@pytest.mark.parametrize("value, transform", [(2.0**128, 2.0), (2.0**-128, 0.5)])
@pytest.mark.parametrize("table_enabled", [False, True])
def test_checked_support_output_keeps_the_original_domain(
    monkeypatch, value, transform, table_enabled
):
    import superglm.distributional.solver._small_group_panels as panels

    if not table_enabled:
        monkeypatch.setattr(panels, "_SUPPORT_TABLE_MAX_CELLS", 0)
    plans = (
        _plan(
            [
                DenseGroupMatrix(np.ones((11, 1))),
                DiscretizedSSPGroupMatrix(
                    np.array([[value]]), np.array([[transform]]), np.zeros(11, dtype=np.intp)
                ),
            ]
        ),
    )
    result = _build(plans, slice(0, 11), byte_budget=2**20)
    assert result.workspace is None
    assert result.reason == "numerical-domain"


def test_zero_and_one_only_columns_have_complete_coverage():
    n = 5
    groups = [
        CategoricalGroupMatrix(np.full(n, -1), 2),
        SplineCategoricalGroupMatrix(
            sp.csr_matrix(np.ones((n, 2))), np.eye(2), np.empty(0, dtype=np.intp)
        ),
    ]
    plans = (_plan(groups), PredictorExecutionPlan(DesignMatrix([], n=n, p=0), True))
    for rows in (slice(0, n), slice(0, 0)):
        result = _build(plans, rows, byte_budget=2**20)
        assert result.workspace is not None
        with result.workspace as workspace:
            expected_rows = n if rows.stop else 0
            np.testing.assert_array_equal(
                workspace.panels[0],
                np.column_stack([np.ones(expected_rows), np.zeros((expected_rows, 4))]),
            )
            np.testing.assert_array_equal(workspace.panels[1], np.ones((expected_rows, 1)))


def test_poison_oracle_detects_missing_factor_writes(monkeypatch):
    import superglm.distributional.solver._small_group_panels as panels

    plans, literal = _problem()
    original = np.empty

    def poisoned(*args, **kwargs):
        result = original(*args, **kwargs)
        if result.dtype == np.float64:
            result.fill(np.nan)
        return result

    monkeypatch.setattr(np, "empty", poisoned)
    monkeypatch.setattr(panels, "_checked_factor_scatter", lambda *args: True)
    result = _build(plans, slice(0, 43), byte_budget=2**20)
    assert result.workspace is not None
    with result.workspace as workspace:
        with pytest.raises(AssertionError):
            _assert_reconstruction(workspace.panels[0], literal[0])
