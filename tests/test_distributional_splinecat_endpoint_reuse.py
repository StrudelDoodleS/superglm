"""Live spline-category representations certify bounded endpoint reuse."""

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
