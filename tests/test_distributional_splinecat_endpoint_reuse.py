"""Live spline-category representations certify bounded endpoint reuse."""

from __future__ import annotations

import gc
import pickle
import tracemalloc
import weakref
from dataclasses import fields, replace

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import superglm.distributional.solver.solver as solver
from superglm._frame import as_eager_frame
from superglm.distributional.families.gaussian import GaussianLS
from superglm.distributional.family import COMPLETE_OBSERVATION
from superglm.distributional.layout import build_stacked_layout
from superglm.distributional.predictor import Predictor, compile_predictors
from superglm.distributional.solver import DenseSolverConfig, fit_dense_fixed_lambda
from superglm.features import Numeric
from superglm.group_matrix import (
    DesignMatrix,
    DiscretizedSplineCategoricalGroupMatrix,
    SplineCategoricalGroupMatrix,
    SupportCompressedSplineCategoricalGroupMatrix,
)

from ._distributional_weights import resolved_prior

KINDS = (
    SplineCategoricalGroupMatrix,
    DiscretizedSplineCategoricalGroupMatrix,
    SupportCompressedSplineCategoricalGroupMatrix,
)


def _problem(kind, n=120):
    bins = np.arange(n, dtype=np.intp) % 11
    x = np.linspace(-1.0, 1.0, 11)
    support = np.column_stack((x, x * x - np.mean(x * x)))
    family = GaussianLS()
    weights = resolved_prior(np.ones(n))
    layout = build_stacked_layout(
        compile_predictors(
            as_eager_frame(pd.DataFrame({"x": support[bins, 0], "xx": support[bins, 1]})),
            weights,
            family.parameters,
            (Predictor("location", {"x": Numeric(), "xx": Numeric()}), Predictor("scale", {})),
            offsets={"location": np.zeros(n), "scale": np.zeros(n)},
        )
    )
    # Unsorted membership makes the discrete row-order cache computationally live.
    rows = np.flatnonzero(np.arange(n) % 3 != 0)[::-1]
    transform = np.array([[1.0, 0.2], [0.0, 0.8]])
    group = (
        kind(sp.csr_matrix(support[bins]), transform, rows)
        if kind is SplineCategoricalGroupMatrix
        else kind(support, transform, bins, rows)
    )
    group.spline_cat_feature = "category"
    location = replace(
        layout.predictors[0],
        design=DesignMatrix([group], n=n, p=2),
        groups=(replace(layout.predictors[0].groups[0], start=0, end=2),),
    )
    layout = replace(layout, predictors=(location, layout.predictors[1]))
    y = 0.2 + group.matvec(np.array([0.3, -0.2]))
    y += np.random.default_rng(672).normal(size=n)
    plan = family.bind_likelihood(y, weights, COMPLETE_OBSERVATION)
    penalty = np.diag([0.0, 0.6, 0.6, 0.0])
    config = DenseSolverConfig(coefficient_curvature="observed", tolerance=1e-9)
    return family, layout, y, plan, penalty, config


def _fit(problem, *, session=None, source=None, initial=None):
    return fit_dense_fixed_lambda(
        *problem[:5],
        config=problem[5],
        chunk_size=19,
        initial=initial if source is None else source.coefficients,
        _reuse_session=session,
        _reuse_source=source,
    )


def _context(problem):
    return solver._validated_context(
        *problem[:5], coefficient_curvature="observed", chunk_size=19, coefficient_face=None
    )


def _group(problem):
    return problem[1].predictors[0].design.group_matrices[0]


def _with_group(problem, group):
    state = problem[1].predictors[0]
    state = replace(state, design=DesignMatrix([group], n=group.n_rows, p=2))
    layout = replace(problem[1], predictors=(state, problem[1].predictors[1]))
    return (problem[0], layout, *problem[2:])


@pytest.mark.parametrize("kind", KINDS)
def test_splinecat_endpoint_reuses_and_matches_changed_penalty(kind, monkeypatch):
    problem = _problem(kind)
    session = solver._DenseObservedReuseSession()
    source = _fit(problem, session=session)
    assert source.converged
    assert id(source) in session._chunk_results

    def forbid_refresh(*args, **kwargs):
        raise AssertionError("certified splinecat endpoint refreshed likelihood")

    with monkeypatch.context() as patch:
        patch.setattr(GaussianLS, "evaluate_natural", forbid_refresh)
        same = _fit(problem, session=session, source=source)
    assert same.iterations == 0
    np.testing.assert_array_equal(same.terminal_data_curvature, source.terminal_data_curvature)

    changed = (*problem[:4], 1.7 * problem[4], problem[5])
    reused = _fit(changed, session=session, source=source)
    fresh = _fit(changed, initial=source.coefficients)
    assert reused.converged and fresh.converged
    for name in ("eta", "terminal_score", "terminal_data_curvature"):
        expected = getattr(fresh, name)
        tolerance = 512 * max(expected.shape) * np.finfo(float).eps
        tolerance *= max(1.0, float(np.linalg.norm(expected)))
        np.testing.assert_allclose(getattr(reused, name), expected, rtol=0.0, atol=tolerance)


_EXACT_ARRAYS = (
    "B.data",
    "B.indices",
    "B.indptr",
    "B_level.data",
    "B_level.indices",
    "B_level.indptr",
    "_data",
    "_indices",
    "_indptr",
    "_dense_level",
    "R_inv",
    "row_idx",
    "_sorted_rows",
)
_DISCRETE_ARRAYS = ("B_unique", "R_inv", "bin_idx_level", "row_idx", "_row_order", "_sorted_rows")
_ARRAY_CASES = [
    (kind, name)
    for kind in KINDS
    for name in (_EXACT_ARRAYS if kind is SplineCategoricalGroupMatrix else _DISCRETE_ARRAYS)
]


def _warm(group):
    group.row_subset(np.arange(group.n_rows))
    if type(group) is SplineCategoricalGroupMatrix:
        group.gram(np.ones(group.n_rows))
        assert type(group._dense_level) is np.ndarray


def _owner(group, path):
    if "." in path:
        owner, name = path.split(".")
        return getattr(group, owner), name
    return group, path


def _mutate(group, path):
    owner, name = _owner(group, path)
    values = getattr(owner, name).copy()
    if name in ("indices", "_indices", "bin_idx_level"):
        values.flat[0] = (
            1 - values.flat[0] if "indices" in name else (values.flat[0] + 1) % group.n_bins
        )
    elif name in ("indptr", "_indptr"):
        values[1] = values[2]
    elif name in ("row_idx", "_sorted_rows", "_row_order"):
        values[:2] = values[1::-1]
    else:
        values.flat[0] += 0.125
    setattr(owner, name, values)


@pytest.mark.parametrize("kind,path", _ARRAY_CASES)
def test_each_live_array_mutation_refuses_endpoint_reuse(kind, path, monkeypatch):
    problem = _problem(kind)
    problem = (*problem[:5], replace(problem[5], tolerance=1e20))
    group = _group(problem)
    _warm(group)
    session = solver._DenseObservedReuseSession()
    source = _fit(problem, session=session, initial=np.zeros(problem[1].n_coefficients))
    assert id(source) in session._chunk_results
    context = _context(problem)
    before = solver._chunk_reuse_data_certificate(context)
    _mutate(group, path)
    assert solver._chunk_reuse_data_certificate(context) != before
    # A zero predictor cannot certify the changed design or its curvature.
    np.testing.assert_array_equal(group.matvec(np.zeros(2)), np.zeros(group.n_rows))

    def fresh_evaluation(*args, **kwargs):
        raise RuntimeError("fresh likelihood required")

    monkeypatch.setattr(GaussianLS, "evaluate_natural", fresh_evaluation)
    with pytest.raises(RuntimeError, match="fresh likelihood required"):
        _fit(problem, session=session, source=source)


@pytest.mark.parametrize("kind,path", _ARRAY_CASES)
def test_identical_bytes_custom_array_semantics_are_refused(kind, path):
    problem = _problem(kind)
    group = _group(problem)
    _warm(group)
    owner, name = _owner(group, path)

    class CustomArray(np.ndarray):
        def __matmul__(self, other):
            return super().__matmul__(other) + 1.0

    original = getattr(owner, name)
    custom = original.view(CustomArray)
    np.testing.assert_array_equal(custom, original)
    setattr(owner, name, custom)
    assert solver._chunk_reuse_data_certificate(_context(problem)) is None


@pytest.mark.parametrize("matrix_name", ["B", "B_level"])
@pytest.mark.parametrize("replacement", ["csc", "csr_array", "subclass"])
def test_unknown_nested_sparse_semantics_are_refused(matrix_name, replacement):
    problem = _problem(SplineCategoricalGroupMatrix)
    group = _group(problem)
    matrix = getattr(group, matrix_name)
    if replacement == "csc":
        matrix = matrix.tocsc()
    elif replacement == "csr_array":
        matrix = sp.csr_array(matrix)
    else:

        class CustomCSR(sp.csr_matrix):
            pass

        matrix = CustomCSR(matrix)
    setattr(group, matrix_name, matrix)
    assert solver._chunk_reuse_data_certificate(_context(problem)) is None


@pytest.mark.parametrize("matrix_name", ["B", "B_level"])
@pytest.mark.parametrize("flag", ["_has_sorted_indices", "_has_canonical_format"])
def test_csr_flags_are_live_and_not_populated(matrix_name, flag):
    problem = _problem(SplineCategoricalGroupMatrix)
    context = _context(problem)
    matrix = getattr(_group(problem), matrix_name)
    matrix.__dict__.pop(flag, None)
    before = solver._chunk_reuse_data_certificate(context)
    assert before is not None
    assert flag not in matrix.__dict__
    setattr(matrix, flag, True)
    assert solver._chunk_reuse_data_certificate(context) != before
    setattr(matrix, flag, "custom truth value")
    assert solver._chunk_reuse_data_certificate(context) is None


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("name", ["n_rows", "_p_b", "shape", "spline_cat_feature", "n_bins"])
def test_structural_state_changes_invalidate_certificate(kind, name):
    if kind is SplineCategoricalGroupMatrix and name == "n_bins":
        return
    problem = _problem(kind)
    context = _context(problem)
    before = solver._chunk_reuse_data_certificate(context)
    assert before is not None
    group = _group(problem)
    value = getattr(group, name)
    changed = (
        (value[0], value[1] + 1)
        if name == "shape"
        else value + ("changed" if isinstance(value, str) else 1)
    )
    setattr(group, name, changed)
    assert solver._chunk_reuse_data_certificate(context) != before


@pytest.mark.parametrize("kind", KINDS)
def test_unknown_group_subclasses_are_refused(kind):
    class CustomGroup(kind):
        pass

    problem = _problem(kind)
    original = _group(problem)
    replacement = CustomGroup.__new__(CustomGroup)
    for name in (
        SplineCategoricalGroupMatrix.__slots__
        if kind is SplineCategoricalGroupMatrix
        else DiscretizedSplineCategoricalGroupMatrix.__slots__
    ):
        setattr(replacement, name, getattr(original, name))
    problem = _with_group(problem, replacement)
    assert solver._chunk_reuse_data_certificate(_context(problem)) is None


@pytest.mark.parametrize("kind", KINDS)
def test_certificate_leaves_lazy_caches_unpopulated_and_tracks_transitions(kind):
    problem = _problem(kind)
    group = _group(problem)
    context = _context(problem)
    before = solver._chunk_reuse_data_certificate(context)
    assert before is not None
    assert group._sorted_rows is None
    if kind is SplineCategoricalGroupMatrix:
        assert group._dense_level is False
        group._dense_level = None
        sparse = solver._chunk_reuse_data_certificate(context)
        assert sparse is not None and sparse != before
        group._dense_level = False
    else:
        assert group._row_order is None
    _warm(group)
    warm = solver._chunk_reuse_data_certificate(context)
    assert warm is not None and warm != before
    assert solver._chunk_reuse_data_certificate(context) == warm
    restored = pickle.loads(pickle.dumps(group))
    assert restored._sorted_rows is None
    restored_problem = _with_group(problem, restored)
    assert solver._chunk_reuse_data_certificate(_context(restored_problem)) is not None


@pytest.mark.parametrize("kind", KINDS)
def test_reuse_record_retains_only_coefficient_aggregates_and_expires(kind):
    problem = _problem(kind)
    session = solver._DenseObservedReuseSession()
    source = _fit(problem, session=session)
    record = session._chunk_results[id(source)]
    arrays = [
        getattr(record, field.name)
        for field in fields(record)
        if isinstance(getattr(record, field.name), np.ndarray)
    ]
    p = problem[1].n_coefficients
    assert sum(array.nbytes for array in arrays) == 8 * (2 * p + p * p)
    assert not session._dense
    source_ref = weakref.ref(source)
    del source
    gc.collect()
    assert source_ref() is None
    assert not session._chunk_results


@pytest.mark.parametrize("kind", KINDS)
def test_certificate_hashes_bounded_buffers_without_expansion_or_cache_creation(kind, monkeypatch):
    problem = _problem(kind, n=100_000)
    group = _group(problem)
    if kind is not SplineCategoricalGroupMatrix:
        # Exercise bounded iteration over a genuinely noncontiguous live basis.
        group.B_unique = group.B_unique[::-1, ::-1]
        assert not group.B_unique.flags.c_contiguous
    context = _context(problem)
    real_sha256 = solver.hashlib.sha256
    updates = []

    class BoundedDigest:
        def __init__(self, data=b""):
            self.digest = real_sha256()
            self.update(data)

        def update(self, data):
            assert len(data) <= 64 * 1024
            updates.append(len(data))
            self.digest.update(data)

        def hexdigest(self):
            return self.digest.hexdigest()

    def forbid_expansion(*args, **kwargs):
        raise AssertionError("certificate materialized or subsetted the design")

    monkeypatch.setattr(solver.hashlib, "sha256", BoundedDigest)
    monkeypatch.setattr(kind, "toarray", forbid_expansion)
    monkeypatch.setattr(kind, "row_subset", forbid_expansion)
    if kind is SplineCategoricalGroupMatrix:
        monkeypatch.setattr(sp.csr_matrix, "toarray", forbid_expansion)
    tracemalloc.start()
    try:
        before = solver._chunk_reuse_data_certificate(context)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert before is not None
    assert updates and max(updates) == 64 * 1024
    # Four digest buffers allow iterator/contiguity workspace and Python
    # metadata; one observation-length float64 copy alone exceeds this bound.
    assert peak < 4 * 64 * 1024
    assert group._sorted_rows is None
    assert solver._chunk_reuse_data_certificate(context) == before
