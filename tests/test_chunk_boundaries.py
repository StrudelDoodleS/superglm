"""Chunked loops give the same numbers when the data spans several chunks.

Production loops read rows in chunks of 8,192 rows or more, and almost every
fixture is smaller, so at production sizes a test meets one chunk and cannot
see a loop that drops the rows after its first chunk or counts a row on a chunk
edge twice. Each test here shrinks the chunk knobs to a few dozen rows
(``small_chunks``), so its data spans several chunks with a ragged last one,
and compares what a user reads with the same quantity at the production sizes,
where the data is one chunk, or a reference formed here.

``BOUNDARY_TESTS`` names, for every function in ``src/superglm`` with a
``range(0, n, step)`` loop, the test that crosses its boundary (here, in
``test_chunk_boundaries_design.py``, ``test_chunk_boundaries_distributional.py`` or
beside the code it tests), and
``test_every_chunked_loop_names_a_test_across_its_boundary`` keeps it complete. A loop
chunked another way (a ``range`` from another start, or a ``while`` loop) is not in it.
"""

from __future__ import annotations

import ast
import functools
import importlib
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from superglm import (
    Categorical,
    FactorSmooth,
    NegativeBinomial,
    Numeric,
    Piecewise,
    RandomEffect,
    Spline,
    SuperGLM,
)
from superglm.features import factor_smooth as factor_smooth_module
from superglm.model import fit_ops

ROWS = 61  # a chunk: 600 rows are nine full chunks and a ragged tenth
U = 2.0**-53

# Every row-chunk size a GLM fit, its prediction and its metrics read: (module, name, rows).
_ROW_KNOBS = (
    ("superglm.solvers.centered_system", "_FACTOR_CHUNK_ROWS", ROWS),
    # one dense column of a 600-row design per column tile of the dense-split cross
    ("superglm.solvers.centered_system", "_FACTOR_CHUNK_BYTES", 8 * 600),
    ("superglm.solvers.mode_score", "_CHUNK", ROWS),
    ("superglm.reml.identified", "_CHUNK", ROWS),
    ("superglm.features._spline_runtime", "_SPLINE_SCORE_CHUNK_SIZE", ROWS),
    ("superglm.features.interaction", "_TENSOR_SCORE_CHUNK_SIZE", ROWS),
    ("superglm.features.factor_smooth", "_MARGINAL_QR_CHUNK_ROWS", ROWS),
    ("superglm.features.piecewise", "_GRAM_CHUNK_ROWS", ROWS),
    ("superglm.model.runtime_canonicalize", "_RUNTIME_MEAN_CHUNK_SIZE", ROWS),
    ("superglm.inference._metrics_design", "_MAX_DESIGN_CHUNK_ROWS", ROWS),
    ("superglm._group_matrix._lookup_tables", "_BLOCK_ROWS", ROWS),
    ("superglm._nb2_deviance", "_CHUNK", ROWS),
)
# Functions whose chunk is a keyword default rather than a module constant.
_DEFAULT_KNOBS = (
    ("superglm.solvers._structured.block_leaves", "factor_smooth_prior_statistics"),
    ("superglm.solvers._structured.block_leaves", "_leaf_pass"),
    ("superglm.solvers._structured.block_leaves", "factor_smooth_moment_operators"),
    ("superglm.solvers._structured.moments", "nested_prior_statistics"),
    ("superglm.solvers._structured.moments", "_nested_pass"),
    ("superglm._group_matrix._group_matrix_centered", "centered_gram_rhs"),
    ("superglm._group_matrix._group_matrix_centered", "centered_signed_grams"),
    ("superglm._group_matrix._group_matrix_centered", "_compact_centered_rmatvec"),
    ("superglm._group_matrix._group_matrix_centered", "centered_rhs"),
)


def _shrink_default(monkeypatch, function, name: str, value: int) -> None:
    """Set the default of ``function``'s parameter ``name`` for the test."""
    code = function.__code__
    keyword_only = code.co_varnames[code.co_argcount : code.co_argcount + code.co_kwonlyargcount]
    if name in keyword_only:
        monkeypatch.setitem(function.__kwdefaults__, name, value)
        return
    positional = code.co_varnames[: code.co_argcount]
    defaults = list(function.__defaults__)
    defaults[positional.index(name) - (len(positional) - len(defaults))] = value
    monkeypatch.setattr(function, "__defaults__", tuple(defaults))


def shrink_chunks(monkeypatch) -> None:
    """Shrink every chunk a GLM fit reads to ``ROWS`` rows."""
    for module, name, rows in _ROW_KNOBS:
        monkeypatch.setattr(importlib.import_module(module), name, rows)
    for module, name in _DEFAULT_KNOBS:
        _shrink_default(
            monkeypatch, getattr(importlib.import_module(module), name), "chunk_size", ROWS
        )


def _book(n: int = 600, seed: int = 20261119):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "x1": rng.uniform(0.0, 1.0, n),
            "x2": rng.uniform(-1.0, 1.0, n),
            "x3": rng.normal(size=n),
            "c1": rng.choice(["a", "b", "c", "d", "e"], n),
            "g": rng.choice([f"g{i}" for i in range(8)], n),
        }
    )
    exposure = rng.uniform(0.2, 1.0, n)
    eta = (
        0.3
        + np.sin(2.0 * np.pi * frame["x1"])
        + 0.5 * frame["x2"] ** 2
        + 0.2 * frame["x3"]
        + frame["c1"].map(dict(zip("abcde", [0.0, 0.2, -0.1, 0.3, -0.2])))
    )
    y = rng.poisson(exposure * np.exp(eta)).astype(float)
    weights = rng.uniform(0.5, 2.0, n)
    frame["x4"] = rng.normal(size=n)
    return frame, y, np.log(exposure), weights


def _outputs(model, frame, y, offset, weights) -> dict:
    """What a user reads from a fit: EDF, deviance, predictions and metrics."""
    metrics = model.metrics(frame, y, sample_weight=weights, offset=offset)
    return {
        "edf": model.result.effective_df,
        "deviance": model.result.deviance,
        "predictions": model.predict(frame, offset=offset),
        "metrics": np.array([metrics.deviance, metrics.aic]),
    }


# family, features, interactions, direct_solve: the exact and binned GLM routes, the
# structured factor-smooth and nested random-effect routes, a non-canonical family's
# observed geometry and the NB2 deviance.
_MODELS = {
    "poisson": (
        "poisson",
        lambda: {
            "x1": Spline(kind="ps", k=8),
            "x2": Spline(kind="cr", k=6),
            "x3": Numeric(),
            "x4": Numeric(),
            "c1": Categorical(),
            "g": RandomEffect(),
        },
        [("x1", "x2")],
        "auto",
    ),
    "gamma": (
        "gamma",
        lambda: {"x2": Spline(kind="ps", k=6), "x3": Piecewise([-0.5, 0.5]), "c1": Categorical()},
        [FactorSmooth("x1", group="g", k=5)],
        "auto",
    ),
    "negative_binomial": (
        NegativeBinomial(theta=3.0),
        lambda: {"x1": Spline(kind="cr", k=7), "c1": Categorical()},
        [],
        "auto",
    ),
    "factor_smooth": (
        "poisson",
        lambda: {"x2": Spline(kind="ps", k=6), "x3": Numeric()},
        [FactorSmooth("x1", group="g", k=5)],
        "structured",
    ),
    "nested": (
        "poisson",
        lambda: {"x3": Numeric(), "c1": Categorical(), "g": RandomEffect()},
        [],
        "structured",
    ),
}


@pytest.mark.parametrize("discrete", [False, True])
@pytest.mark.parametrize("case", sorted(_MODELS))
def test_a_reml_fit_reads_the_same_across_chunk_boundaries(monkeypatch, case, discrete):
    """At the same smoothing parameters, a fit whose rows span ten chunks matches the one-chunk fit.

    REML searches once; both fits then hold every smoothing parameter at what it
    found, so each is one REML evaluation there and no stop decision lies between
    them (a searched pair can stop an iteration apart when rounding moves the stop,
    and then differ by 3e-8). They differ only in the order rows are summed, so what
    a user reads agrees to rounding: at most 2.3e-14 relative on data seeds 0-9,
    native and under the Haswell kernel. The bar, 1e-9, sits far below the 1e-3 a row
    counted twice on each chunk edge, or the rows after the first chunk dropped,
    moves these numbers. The loops only the search runs (the REML derivatives) are
    checked by the tests ``BOUNDARY_TESTS`` names for them.
    """
    family, features, interactions, solve = _MODELS[case]
    frame, y, offset, weights = _book()
    if case == "gamma":
        y = np.random.default_rng(7).gamma(2.0, np.exp(0.2 * frame["x2"]) / 2.0)

    def model():
        return SuperGLM(
            family=family,
            features=features(),
            interactions=interactions,
            discrete=discrete,
            selection_penalty=0,
            direct_solve=solve,
        )

    searched = model().fit_reml(frame, y, offset=offset, sample_weight=weights)
    lambdas = dict(searched.reml_diagnostics()["lambdas"])
    monkeypatch.setattr(
        fit_ops,
        "initialize_component_lambdas",
        lambda penalties, default: ({pc.name: lambdas[pc.name] for pc in penalties}, set()),
    )

    def fit():
        fixed = model().fit_reml(frame, y, offset=offset, sample_weight=weights)
        assert fixed.reml_diagnostics()["lambdas"] == lambdas
        return _outputs(fixed, frame, y, offset, weights)

    production, marginal_rows = fit(), factor_smooth_module._MARGINAL_QR_CHUNK_ROWS
    with monkeypatch.context() as patch:
        shrink_chunks(patch)
        # A factor smooth's two null coordinates come out of its marginal QR in an order its
        # rounding sets, which the lambdas' names do not follow, so the marginal is built as
        # in production; its loop's boundary test is beside it (BOUNDARY_TESTS).
        patch.setattr(factor_smooth_module, "_MARGINAL_QR_CHUNK_ROWS", marginal_rows)
        small = fit()
    for name, expected in production.items():
        np.testing.assert_allclose(small[name], expected, rtol=1e-9, atol=0.0, err_msg=name)


def _gamma(k: int) -> float:
    return k * U / (1.0 - k * U)


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Each column's weighted mean, every sum exactly rounded (``math.fsum``)."""
    total = math.fsum(weights)
    return np.array([math.fsum(weights * column) / total for column in values.T])


def _dense_design(n: int = 600, seed: int = 41):
    """Two dense columns 1e3 from zero whose rows after the first chunk sit apart from it."""
    from superglm.group_matrix import DenseGroupMatrix, DesignMatrix

    rng = np.random.default_rng(seed)
    values = 1.0e3 + rng.normal(size=(n, 2))
    values[ROWS:] += [3.0, -2.0]
    weights = rng.uniform(0.5, 2.0, n)
    return DesignMatrix([DenseGroupMatrix(values)], n=n, p=2), values, weights


def test_the_dense_centres_and_centring_read_every_chunk(monkeypatch):
    """The prior-weighted centre, the offset mean and the centred diagonal against their definitions.

    Each mean is ``x_ref + sum w (x - x_ref) / sum w`` summed in fixed chunks,
    within ``gamma_(n+4) max |x - x_ref|`` of the exact mean plus a rounding of
    the result (Higham 2002, Lemma 3.1), whatever the grouping; the diagonal
    within ``gamma_(4n)`` of itself (the corrected two-pass bound). The rows
    after the first chunk sit apart from it, so a dropped chunk moves each by
    about one unit and a row counted twice at a chunk edge by about ``1e-3``:
    either is far outside the bound.
    """
    from superglm.solvers import mode_score

    monkeypatch.setattr(mode_score, "_CHUNK", ROWS)
    dm, values, weights = _dense_design()
    n = dm.n
    mean = _weighted_mean(values, weights)
    bound = _gamma(n + 4) * np.abs(values - values[0]).max(axis=0) + 2.0 * U * np.abs(mean)

    centre = mode_score.prior_weighted_centre(dm, weights)
    assert np.all(np.abs(centre - mean) <= bound)

    state_centre = values[7].copy()
    offset_mean = mode_score.centre_offset_mean(dm, weights, weights.sum(), state_centre, mean)
    assert np.all(np.abs(offset_mean - (mean - state_centre)) <= bound)

    # the diagonal's pass reads at least 256 rows a chunk: three chunks here
    mean_x, sum_w, diagonal = mode_score.weighted_column_centring(dm, weights, weights > 0.0)
    squares = np.array([math.fsum(weights * (c - m) ** 2) for c, m in zip(values.T, mean)])
    assert np.all(np.abs(mean_x - mean) <= bound)
    assert sum_w == pytest.approx(math.fsum(weights), rel=_gamma(n))
    assert np.all(np.abs(diagonal - squares) <= _gamma(4 * n) * squares)


def test_the_weak_identification_tests_read_every_chunk(monkeypatch):
    """A column is weak by what the rows after the first chunk carry, and both weak tests say so.

    Column 0 is ordinary. Columns 1 and 2 are 5 on every row of full weight
    and move only on rows of weight ``1e-15``, far below ``gamma_n`` times the
    largest weight, so both are weakly identified (``reml/identified.py``).
    Row 0 is such a row: column 1 reads 7 there, so its shifted centre needs
    every chunk's rows to land on 5 (from the first chunk alone it lands near
    6.8 and the full-weight rows then carry curvature); column 2 is 5 on row 0
    and moves only on rows after the first chunk, so its centred mass is there.
    """
    from superglm.group_matrix import DenseGroupMatrix, DesignMatrix
    from superglm.reml.identified import laplace_excluded_coefficients
    from superglm.solvers import mode_score

    n = 600
    rng = np.random.default_rng(5)
    weights = np.ones(n)
    tiny = np.r_[0, np.arange(3 * ROWS, n, 37)]
    weights[tiny] = 1.0e-15
    values = np.column_stack([rng.normal(size=n), np.full(n, 5.0), np.full(n, 5.0)])
    values[tiny, 1] = 7.0
    values[tiny[1:], 2] = 6.0 + rng.uniform(size=tiny.size - 1)
    dm = DesignMatrix([DenseGroupMatrix(values)], n=n, p=3)

    def weak_tests():
        mean_x, _, _ = mode_score.weighted_column_centring(dm, weights, weights > 0.0)
        mask = mode_score.weakly_identified_mask(
            dm=dm,
            fisher_weights=weights,
            positive_prior=weights > 0.0,
            mean_x=mean_x,
            penalty_diagonal=np.zeros(3),
            unpenalized=np.ones(3, dtype=bool),
        )
        return laplace_excluded_coefficients(dm, weights, None).tolist(), mask.tolist()

    assert weak_tests() == ([1, 2], [False, True, True])
    for module in (mode_score, importlib.import_module("superglm.reml.identified")):
        monkeypatch.setattr(module, "_CHUNK", ROWS)
    assert weak_tests() == ([1, 2], [False, True, True])


def test_the_signed_offset_pair_reads_both_chunks():
    """The observed geometry's signed weighted mean, an exact pair, across its two 8,192-row chunks.

    Its chunk is fixed at 8,192 rows, so the design takes 8,192 + 1,000; the
    last 1,000 rows sit 3 units apart and some weights are negative. The pair's
    sum is within ``gamma_(n+4)`` of the exactly rounded mean relative to
    ``sum |w| |x - m| / |sum w|`` (the corrected two-pass bound), and a row
    counted twice at the edge moves it by about ``1e-4``.
    """
    from superglm.group_matrix import DenseGroupMatrix, DesignMatrix
    from superglm.reml.observed_geometry import _stable_signed_offset

    n = 8192 + 1000
    rng = np.random.default_rng(8192)
    values = 50.0 + rng.normal(size=(n, 2))
    values[8192:] += 3.0
    weights = rng.uniform(0.5, 1.5, n)
    weights[rng.choice(n, 40, replace=False)] *= -0.1
    dm = DesignMatrix([DenseGroupMatrix(values)], n=n, p=2)
    anchor, lo = _stable_signed_offset(dm, weights, float(np.sum(weights)))
    mean = _weighted_mean(values, weights)
    spread = np.array([math.fsum(np.abs(weights) * np.abs(c - m)) for c, m in zip(values.T, mean)])
    bound = _gamma(n + 4) * spread / abs(math.fsum(weights)) + 2.0 * U * np.abs(mean)
    assert np.all(np.abs((anchor + lo) - mean) <= bound)


def test_the_structured_layouts_prior_centres_read_every_chunk():
    """The factor-smooth and nested layouts' border centres, chunked and in one pass.

    Each centre is a shifted prior-weighted mean summed in leaf order, so the
    two passes differ only in how the sum is grouped: within twice
    ``gamma_(n+4)`` relative to ``max |x|`` of the border rows, a bound a
    dropped chunk or a row counted twice at a chunk edge exceeds by orders of
    magnitude. Each layout caches its centre per prior weights, so the cache
    is emptied before each pass.
    """
    from superglm.solvers._structured.block_leaves import (
        _lineage_slot,
        factor_smooth_prior_statistics,
    )
    from superglm.solvers._structured.layout import get_structured_layout
    from superglm.solvers._structured.moments import nested_prior_statistics
    from tests.test_factor_smooth_structured_system import _design
    from tests.test_nested_structured_plumbing import _layout, _nested_case

    _, dm, groups, dominant = _design(discrete=False)
    layout = get_structured_layout(dm, groups, dominant_group_index=dominant)
    weights = np.random.default_rng(3).uniform(0.5, 2.0, dm.n)
    centres = []
    for chunk in (dm.n, 7):
        _lineage_slot(layout, "prior").clear()
        centres.append(factor_smooth_prior_statistics(layout, weights, chunk_size=chunk)[0])
    bound = 2.0 * _gamma(dm.n + 4) * np.max(np.abs(dm.toarray()))
    assert np.max(np.abs(centres[0])) > 0.0
    assert np.all(np.abs(centres[1] - centres[0]) <= bound)

    case = _nested_case()
    layout = _layout(case)
    centres = []
    for chunk in (case.dm.n, ROWS):
        layout.prior_cache.clear()
        centres.append(nested_prior_statistics(layout, case.weights, chunk_size=chunk)[0])
    bound = 2.0 * _gamma(case.dm.n + 4) * np.max(np.abs(case.dm.toarray()))
    assert np.max(np.abs(centres[0])) > 0.0
    assert np.all(np.abs(centres[1] - centres[0]) <= bound)


@pytest.mark.parametrize("discrete", [False, True])
def test_the_factor_smooth_moment_operators_read_every_chunk(discrete):
    """A factor smooth's REML weight-derivative moments, chunked and in one pass.

    Each moment sums weighted row products in leaf order, so the two passes differ
    only in how the sums are grouped, far below 1e-10 of the largest entry; a dropped
    chunk or a row counted twice at a chunk edge moves an entry by a row's share. The
    binned and exact layouts take the table and the raw-row routes through the loop.
    """
    from superglm.solvers._structured.block_leaves import factor_smooth_moment_operators
    from superglm.solvers._structured.layout import get_structured_layout
    from tests.test_factor_smooth_structured_system import _design

    rng, dm, groups, index = _design(discrete=discrete)
    layout = get_structured_layout(dm, groups, dominant_group_index=index)
    signed = [rng.uniform(-1.0, 1.0, size=dm.n), rng.uniform(0.5, 2.0, size=dm.n)]
    whole, chunked = (
        factor_smooth_moment_operators(layout, signed, chunk_size=rows) for rows in (dm.n, 7)
    )
    for (operator, cross, total), (blocked, blocked_cross, blocked_total) in zip(
        whole, chunked, strict=True
    ):
        pairs = [(operator.D, blocked.D), (operator.C, blocked.C), (operator.A, blocked.A)]
        for exact, piecewise in [*pairs, (cross, blocked_cross), (total, blocked_total)]:
            scale = np.max(np.abs(exact))
            assert scale > 0.0
            np.testing.assert_allclose(piecewise, exact, rtol=0.0, atol=1e-10 * scale)


# Every function in src/superglm with a ``range(0, n, step)`` loop (not every chunked loop:
# one from another start, or a ``while`` loop, is not seen), and the test whose data spans
# two or more of its chunks (paths under src/superglm and tests/).
# Each named test fails when the loop drops the rows after its first chunk (checked by
# mutation); a new chunked loop fails the test below until its boundary test is named.
BOUNDARY_TESTS: dict[str, str | tuple[str, ...]] = {
    "_frame.py::_update_polars_text": (
        "test_chunk_boundaries_design.py::test_a_text_columns_length_segment_covers_every_slice",
        "test_polars_guard_versions.py::test_a_text_column_digests_the_same_in_any_chunk_size",
    ),
    "_group_matrix/_column_local_centering.py::_anchor_centred_columns": (
        "test_chunk_boundaries_design.py::test_anchor_centring_sums_and_writes_every_support_chunk"
    ),
    "_group_matrix/_column_local_centering.py::recentred_gram_rows": (
        "test_column_local_centering.py::test_a_projection_the_repair_forms_serves_every_chunk_of_failing_columns"
    ),
    "_group_matrix/_group_matrix_algebra.py::_chunked_support_bincount_2d": (
        "test_support_compression.py::test_agg_by_bin_expands_the_spline_cat_level_in_chunks"
    ),
    "_group_matrix/_group_matrix_algebra.py::_cross_gram_sparse_ssp": (
        "test_chunk_boundaries_design.py::test_cancellation_sensitive_sparse_grams_sum_every_projected_row_block"
    ),
    "_group_matrix/_group_matrix_algebra.py::_cross_gram_tensor_spline_categorical": (
        "test_chunk_boundaries_design.py::test_tensor_by_spline_cat_channel_tiles_sum_every_row_chunk",
        "test_support_compression.py::test_tensor_by_spline_cat_fallback_expands_rows_in_chunks",
        "test_support_compression.py::test_tensor_fallback_bounds_its_channel_accumulator",
    ),
    "_group_matrix/_group_matrix_algebra.py::_support_support_raw_cross": (
        "test_cross_matrix_histogram_dispatch.py::test_short_discrete_cross_expands_only_bounded_support_panels"
    ),
    "_group_matrix/_group_matrix_algebra.py::_support_support_raw_cross_expanded": (
        "test_chunk_boundaries_design.py::test_expanded_support_cross_sums_every_row_chunk"
    ),
    "_group_matrix/_group_matrix_centered.py::_compact_centered_rmatvec": (
        "test_chunk_boundaries_design.py::test_centred_right_hand_sides_sum_every_row_chunk"
    ),
    "_group_matrix/_group_matrix_centered.py::centered_gram_rhs": (
        "test_chunk_boundaries.py::test_a_reml_fit_reads_the_same_across_chunk_boundaries"
    ),
    "_group_matrix/_group_matrix_centered.py::centered_rhs": (
        "test_chunk_boundaries_design.py::test_centred_right_hand_sides_sum_every_row_chunk"
    ),
    "_group_matrix/_group_matrix_centered.py::centered_signed_grams": (
        "test_reml_batched_w_derivatives.py::test_batched_signed_grams_match_centered_product_oracle"
    ),
    "_group_matrix/_group_matrix_core.py::_saturated_ssp_gram": (
        "test_chunk_boundaries_design.py::test_a_saturated_gram_reads_rows_as_a_dense_view_only_after_checking_each_one",
        "test_ssp_gram_memory.py::test_blocked_gram_matches_exact_factor_target",
    ),
    "_group_matrix/_group_matrix_core.py::_solver_space_gram": (
        "test_chunk_boundaries_design.py::test_cancellation_sensitive_sparse_grams_sum_every_projected_row_block"
    ),
    "_group_matrix/_group_matrix_support.py::_row_hashes": (
        "test_support_compression.py::test_compiled_row_scan_is_the_dense_scan_bit_for_bit"
    ),
    "_group_matrix/_group_matrix_support.py::_row_index_chunked": (
        "test_support_compression.py::test_hashed_grouping_matches_byte_keyed_grouping"
    ),
    "_group_matrix/_group_matrix_support.py::_verified_representatives": (
        "test_chunk_boundaries_design.py::test_a_wrong_grouping_in_the_last_row_chunk_fails_verification"
    ),
    "_group_matrix/_lookup_tables.py::_Sweeps.__init__": (
        "test_lookup_table_gram.py::test_sweep_ranges_are_capped_whatever_the_row_count"
    ),
    "_group_matrix/_lookup_tables.py::gather_channels": (
        "test_distributional_binned_rows.py::test_leverages_equal_the_dense_row_products"
    ),
    "_group_matrix/_lookup_tables.py::table_leverages": (
        "test_reml_leverage_tables.py::test_a_random_effect_is_a_lookup_on_both_leverage_routes"
    ),
    "_group_matrix/_lookup_tables.py::table_rmatvec": (
        "test_reml_leverage_tables.py::test_hand_built_dense_columns_match_the_materialised_design"
    ),
    "_nb2_deviance.py::unit_deviance": (
        "test_chunk_boundaries.py::test_a_reml_fit_reads_the_same_across_chunk_boundaries"
    ),
    "_parallel.py::pooled_row_blocks": (
        "test_chunk_boundaries_design.py::test_pooled_row_blocks_partition_the_rows"
    ),
    "distributional/checks/compare.py::_murphy_diagram": (
        "test_chunk_boundaries_distributional.py::test_murphy_curves_are_the_same_in_threshold_blocks"
    ),
    "distributional/checks/qq.py::_simulated_envelope": (
        "test_chunk_boundaries_distributional.py::test_qq_envelope_is_the_same_in_simulation_blocks"
    ),
    "distributional/families/gaussian.py::GaussianLikelihoodPlan.carrier_rounding": (
        "test_chunk_boundaries_distributional.py::test_gaussian_carrier_formed_in_weight_blocks_is_the_eager_carrier"
    ),
    "distributional/families/gaussian.py::_derived_carrier_digest": (
        "test_chunk_boundaries_distributional.py::test_gaussian_carrier_formed_in_weight_blocks_is_the_eager_carrier"
    ),
    "distributional/families/tweedie.py::_tail_log_survival": (
        "test_tweedie_lss_distribution_functions.py::test_tweedie_quantile_inverts_the_atom_edge_as_the_bisection_does"
    ),
    "distributional/kernels/_curvature_direction.py::third_from_hessian_direction": (
        "test_chunk_boundaries_distributional.py::test_generalized_pareto_third_derivatives_are_the_same_in_row_blocks"
    ),
    "distributional/kernels/_negative_binomial_numba.py::_recurrence_row": (
        "test_chunk_boundaries_distributional.py::test_negative_binomial_recurrence_sums_every_term_past_its_block"
    ),
    "distributional/kernels/_tweedie_windows.py::tweedie_window_jets": (
        "test_tweedie_response_bounds.py::test_rare_row_admission_is_independent_of_other_rows_and_tiles"
    ),
    "distributional/kernels/negative_binomial.py::_iter_recurrence_tiles": (
        "test_chunk_boundaries_distributional.py::test_negative_binomial_recurrence_sums_every_term_past_its_block"
    ),
    "distributional/model.py::_predict_eta_pooled": (
        "test_lss_prediction_se.py::test_a_book_is_scored_without_keeping_every_terms_contribution"
    ),
    "distributional/posterior.py::_theta_blocks": (
        "test_lss_posterior.py::test_prior_weights_are_chunked_in_step_with_their_rows"
    ),
    "distributional/posterior.py::posterior_parameters": (
        "test_lss_posterior.py::test_chunking_never_changes_the_pushforward"
    ),
    "distributional/smoothing/derivatives.py::_absolute_contraction": (
        "test_distributional_laml_derivatives.py::test_binned_layout_derivatives_equal_the_dense_row_chunks"
    ),
    "distributional/smoothing/derivatives.py::_analytic_first_derivative": (
        "test_distributional_laml_derivatives.py::test_row_blocks_of_the_family_derivatives_change_no_bit"
    ),
    "distributional/smoothing/derivatives.py::_analytic_second_derivatives": (
        "test_distributional_laml_derivatives.py::test_generalized_pareto_natural_thirds_are_row_blocked"
    ),
    "distributional/smoothing/derivatives.py::_cross_all": (
        "test_distributional_laml_derivatives.py::test_cross_all_matches_the_per_component_assembly"
    ),
    "distributional/smoothing/derivatives.py::_gradient_pass": (
        "test_distributional_bounded_derivatives.py::test_bounded_derivatives_match_identical_compiled_design"
    ),
    "distributional/smoothing/derivatives.py::_hessian_pass": (
        "test_distributional_bounded_derivatives.py::test_bounded_derivatives_match_identical_compiled_design",
        "test_distributional_laml_derivatives.py::test_binned_layout_derivatives_equal_the_dense_row_chunks",
    ),
    "distributional/smoothing/derivatives.py::_leverage_blocks": (
        "test_distributional_bounded_derivatives.py::test_bounded_derivatives_match_identical_compiled_design"
    ),
    "distributional/smoothing/endpoint_direction.py::_curvature_packed": (
        "test_distributional_gaussian_preparation.py::test_derived_curvature_dispatch_owns_bounded_children_and_refreshes"
    ),
    "distributional/smoothing/endpoint_laml.py::_row_certificate_error": (
        "test_chunk_boundaries_distributional.py::test_endpoint_row_certificate_sums_every_row_chunk"
    ),
    "distributional/solver/_joint_tables.py::JointTables.add_rows": (
        "test_distributional_joint_tables.py::test_one_set_of_tables_reads_what_the_group_routes_form"
    ),
    "distributional/solver/_joint_tables.py::_split_tables": (
        "test_chunk_boundaries_distributional.py::test_joint_tables_hold_each_channel_of_a_column_pair_once"
    ),
    "distributional/solver/_likelihood_cache.py::_is_range": (
        "test_chunk_boundaries_distributional.py::test_likelihood_cache_compares_every_index_block_before_reusing_a_child"
    ),
    "distributional/solver/assembly.py::_gram_channel_tasks": (
        "test_distributional_assembly.py::test_the_dense_gram_keeps_a_running_total_rather_than_every_block"
    ),
    "distributional/solver/certificate.py::dense_rows.rows": (
        "test_lss_terminal_rank_certificate.py::test_near_pair_keeps_rank_with_exact_variance"
    ),
    "distributional/solver/chunks.py::_compiled_one_table_set": (
        "test_distributional_joint_tables.py::test_a_one_set_pass_hands_the_pool_its_rows_and_its_tables"
    ),
    "distributional/solver/chunks.py::iter_row_chunks": (
        "test_declared_scalar_vertical_slice.py::test_fixed_penalty_independent_objective_chunk_replay_and_offsets"
    ),
    "features/_spline_runtime.py::score": (
        "test_chunk_boundaries.py::test_a_reml_fit_reads_the_same_across_chunk_boundaries"
    ),
    "features/factor_smooth.py::FactorSmooth._build_marginal": (
        "test_factor_smooth_feature.py::test_streamed_tsqr_keeps_the_legacy_wiggle_and_puts_null_j_on_degree_j"
    ),
    "features/interaction.py::TensorInteraction.score": (
        "test_chunk_boundaries.py::test_a_reml_fit_reads_the_same_across_chunk_boundaries",
        "test_discretize_fit.py::TestDiscretizedTensorInteraction::test_exact_tensor_prediction_batches_large_observed_support",
    ),
    "features/interaction.py::TensorInteraction.transform": (
        "test_chunk_boundaries_design.py::test_tensor_design_rows_fill_every_chunk"
    ),
    "features/interaction.py::_rows_repeat": (
        "test_support_compression.py::test_rows_repeat_compares_every_block"
    ),
    "features/piecewise.py::Piecewise._weighted_gram": (
        "test_chunk_boundaries_design.py::test_piecewise_weighted_gram_sums_every_row_chunk"
    ),
    "inference/_metrics_design.py::EvaluationDesign.iter_dense_chunks": (
        "test_chunk_boundaries_design.py::test_evaluation_design_moments_sum_every_frame_chunk"
    ),
    "inference/_metrics_design.py::iter_dense_chunks": (
        "test_chunk_boundaries_design.py::test_projected_design_moments_sum_every_design_chunk",
        "test_data_rank_grey_subspace.py::test_chunked_projection_is_the_rows_product",
    ),
    "model/runtime_canonicalize.py::_runtime_training_feature_column_means": (
        "test_discretize_fit.py::test_binned_fit_predicts_what_it_fitted_on_a_skewed_column"
    ),
    "reml/_leverage_rows.py::_support_block.pieces": (
        "test_reml_leverage_hessian.py::test_a_pooled_block_forms_its_solved_rows_a_piece_at_a_time"
    ),
    "reml/_leverage_rows.py::solved_row_blocks": (
        "test_reml_leverage_hessian.py::test_a_pooled_block_forms_its_solved_rows_a_piece_at_a_time"
    ),
    "reml/identified.py::laplace_excluded_coefficients": (
        "test_chunk_boundaries.py::test_the_weak_identification_tests_read_every_chunk"
    ),
    "reml/multi_penalty.py::_refine_product": (
        "test_compensated_dot.py::test_product_refinement_batches_are_bounded_by_entries"
    ),
    "reml/observed_geometry.py::_stable_signed_offset.chunks": (
        "test_chunk_boundaries.py::test_the_signed_offset_pair_reads_both_chunks"
    ),
    "screening/_pair_factor.py::numeric_cat_factor": (
        "test_pair_design_factor.py::test_the_numeric_cat_factor_reproduces_its_own_moments"
    ),
    "screening/_pair_factor.py::numeric_numeric_factor": (
        "test_chunk_boundaries_design.py::test_the_numeric_numeric_factor_reduces_every_row_chunk"
    ),
    "screening/_pair_factor.py::pair_design_factor": (
        "test_structured_screening.py::test_a_level_with_no_mass_cannot_carry_a_free_degree_of_freedom"
    ),
    "screening/_structured.py::_filter_factor_sum": (
        "test_screening_reference_variance.py::test_structured_variance_matches_the_observation_space_quadratic"
    ),
    "screening/_structured.py::_profile": (
        "test_screening_reference_variance.py::test_structured_variance_matches_the_observation_space_quadratic"
    ),
    "screening/_structured.py::_profiled_curvature_trace": (
        "test_chunk_boundaries_design.py::test_structured_trace_and_base_rows_merge_every_level_chunk"
    ),
    "screening/_structured.py::_unemitted_overlap_rows": (
        "test_chunk_boundaries_design.py::test_structured_trace_and_base_rows_merge_every_level_chunk"
    ),
    "solvers/_structured/block_leaves.py::_leaf_pass": (
        "test_chunk_boundaries.py::test_a_reml_fit_reads_the_same_across_chunk_boundaries"
    ),
    "solvers/_structured/block_leaves.py::factor_smooth_moment_operators": (
        "test_chunk_boundaries.py::test_the_factor_smooth_moment_operators_read_every_chunk"
    ),
    "solvers/_structured/block_leaves.py::factor_smooth_prior_statistics": (
        "test_chunk_boundaries.py::test_the_structured_layouts_prior_centres_read_every_chunk"
    ),
    "solvers/_structured/moments.py::_centered_leaf_pass": (
        "test_chunk_boundaries.py::test_a_reml_fit_reads_the_same_across_chunk_boundaries"
    ),
    "solvers/_structured/moments.py::_prior_center": (
        "test_chunk_boundaries.py::test_the_structured_layouts_prior_centres_read_every_chunk"
    ),
    "solvers/_structured/nested.py::_square_diagonal_by_columns": (
        "test_chunk_boundaries_design.py::test_square_diagonal_fills_every_column_block"
    ),
    "solvers/centered_system.py::_dense_chunks.chunks": (
        "test_centred_state_offsets.py::test_the_dense_block_takes_the_corrected_two_pass_at_every_offset"
    ),
    "solvers/centered_system.py::anchored_dense_moments": (
        "test_chunk_boundaries.py::test_a_reml_fit_reads_the_same_across_chunk_boundaries"
    ),
    "solvers/centered_system.py::dense_split_cross": (
        "test_chunk_boundaries.py::test_a_reml_fit_reads_the_same_across_chunk_boundaries"
    ),
    "solvers/centered_system.py::grouped_augmented_row_quadratic_forms": (
        "test_reml_leverage_hessian.py::test_the_nested_factor_takes_the_leverage_route_within_rounding"
    ),
    "solvers/centered_system.py::grouped_projected_factor.blocks": (
        "test_data_rank_grey_subspace.py::test_grey_projection_is_the_rows_product"
    ),
    "solvers/centered_system.py::iter_grouped_design_chunks": (
        "test_reml_leverage_hessian.py::test_leverage_terms_match_derivative_grams_within_rounding"
    ),
    "solvers/centered_system.py::iter_grouped_design_leaves": (
        "test_tsqr_certificate.py::test_data_factor_reuse_keys_every_input"
    ),
    "solvers/mode_score.py::CompensatedSum._fallback": (
        "test_coefficient_revisions.py::test_a_compensated_predictor_reads_the_same_in_any_row_chunks"
    ),
    "solvers/mode_score.py::CompensatedSum._join": (
        "test_coefficient_revisions.py::test_a_compensated_predictor_reads_the_same_in_any_row_chunks"
    ),
    "solvers/mode_score.py::CompensatedSum.value": (
        "test_coefficient_revisions.py::test_a_compensated_predictor_reads_the_same_in_any_row_chunks"
    ),
    "solvers/mode_score.py::_centred_block_gram": (
        "test_binomial_log_mean_space.py::test_the_weak_block_gram_is_whole_across_its_row_chunks"
    ),
    "solvers/mode_score.py::centre_offset_mean": (
        "test_chunk_boundaries.py::test_the_dense_centres_and_centring_read_every_chunk"
    ),
    "solvers/mode_score.py::centred_data_score": (
        "test_binomial_log_mean_space.py::test_an_exact_alias_on_a_large_design_is_not_refused"
    ),
    "solvers/mode_score.py::centred_matvec": (
        "test_chunk_boundaries.py::test_a_reml_fit_reads_the_same_across_chunk_boundaries"
    ),
    "solvers/mode_score.py::dense_centred_matvec": (
        "test_centred_state_offsets.py::test_the_proximal_rows_serve_the_block_products_of_their_outer_iteration"
    ),
    "solvers/mode_score.py::dense_centred_rmatvec": (
        "test_centred_state_offsets.py::test_the_proximal_rows_serve_the_block_products_of_their_outer_iteration"
    ),
    "solvers/mode_score.py::prior_weighted_centre": (
        "test_chunk_boundaries.py::test_the_dense_centres_and_centring_read_every_chunk"
    ),
    "solvers/mode_score.py::weakly_identified_mask": (
        "test_chunk_boundaries.py::test_the_weak_identification_tests_read_every_chunk"
    ),
    "solvers/mode_score.py::weighted_column_centring": (
        "test_chunk_boundaries.py::test_the_dense_centres_and_centring_read_every_chunk"
    ),
}
_SRC = Path(importlib.import_module("superglm").__file__).parent
_TESTS = Path(__file__).parent


def _chunked_functions() -> set[str]:
    """``path::qualname`` of every function holding a ``range(0, n, step)``, step not 1, 2 or -1."""
    found = set()
    for path in sorted(_SRC.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if not re.search(r"\brange\(\s*0\s*,", source):
            continue  # no range(0, ...) to parse for
        stack: list[str] = []

        def visit(node, path=path, stack=stack):
            named = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            if named:
                stack.append(node.name)
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "range"
                and len(node.args) == 3
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == 0
                and not (
                    isinstance(node.args[2], ast.Constant) and node.args[2].value in (1, 2, -1)
                )
            ):
                found.add(f"{path.relative_to(_SRC).as_posix()}::{'.'.join(stack)}")
            for child in ast.iter_child_nodes(node):
                visit(child)
            if named:
                stack.pop()

        visit(ast.parse(source))
    return found


@functools.cache
def _module_body(file: str) -> list[ast.stmt]:
    return ast.parse((_TESTS / file).read_text(encoding="utf-8")).body


def _defines(node_id: str) -> bool:
    """Whether the test module of ``node_id`` defines its (class and) test function."""
    file, *names = node_id.split("::")
    scope = _module_body(file)
    for name in names:
        match = [n for n in scope if getattr(n, "name", None) == name]
        if not match:
            return False
        scope = getattr(match[0], "body", [])
    return True


def test_every_chunked_loop_names_a_test_across_its_boundary():
    found = _chunked_functions()
    assert sorted(found - BOUNDARY_TESTS.keys()) == [], "name a boundary test in BOUNDARY_TESTS"
    assert sorted(BOUNDARY_TESTS.keys() - found) == [], "these hold no chunked loop any more"
    named = [t for v in BOUNDARY_TESTS.values() for t in ((v,) if isinstance(v, str) else v)]
    assert [t for t in named if not _defines(t)] == []
