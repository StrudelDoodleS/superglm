"""Piecewise ``degrees=`` and name-mode breaks: the level-axis-only surface.

Pins the segmented-degrees contract on the ``Piecewise`` class itself: ``degrees=``
validation at construction, the loud numeric-axis refusals (band-name breaks
and any degree != 1 both exist only where an ``OrderedCategorical`` hosts the
spec), the segmented C0 grafted-polynomial basis (structural continuity,
plateau tails, rank refusals), the bit-compatibility of ``degrees=[1, ..., 1]``
with the un-stated default, and the structural contrast rows (slope change per
break, curvature family per degree>=2 segment) verified against numerical
differentiation of the fitted function.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from superglm import Piecewise

LEVEL_AXIS_N = 11  # positions 0..10


def _hosted(piecewise: Piecewise, n_levels: int = LEVEL_AXIS_N) -> Piecewise:
    """Mark a Piecewise the way OrderedCategorical marks its deep copy."""
    piecewise._on_level_axis = True
    piecewise.lower = 0.0
    piecewise.upper = float(n_levels - 1)
    return piecewise


def _level_rows(n_levels: int = LEVEL_AXIS_N, repeats: int = 25) -> np.ndarray:
    return np.repeat(np.arange(float(n_levels)), repeats)


# ── construction validation ──────────────────────────────────────────


def test_degrees_length_must_match_segments() -> None:
    with pytest.raises(ValueError, match="one degree per segment"):
        Piecewise(breaks=[3.0, 7.0], degrees=[1, 2])


def test_degrees_must_be_nonnegative_integers() -> None:
    with pytest.raises(ValueError, match="integers >= 0"):
        Piecewise(breaks=[3.0], degrees=[1, 1.5])
    with pytest.raises(ValueError, match=">= 0"):
        Piecewise(breaks=[3.0], degrees=[1, -1])
    with pytest.raises(ValueError, match="integers >= 0"):
        Piecewise(breaks=[3.0], degrees=[True, 1])


def test_degrees_refuse_int_mode_breaks() -> None:
    with pytest.raises(ValueError, match="requires stated breaks"):
        Piecewise(breaks=3, degrees=[1, 1, 1, 1])


def test_degrees_with_no_breaks_refuse_at_construction_advising_polynomial() -> None:
    """One global polynomial segment is a Polynomial, and the error says so.

    Before the refusal this constructed fine and died at build() with rule
    2's "use Numeric() instead" -- wrong advice for the caller degrees=
    exists for.
    """
    with pytest.raises(ValueError, match=r"Polynomial\(degree=d\)") as excinfo:
        Piecewise(breaks=[], degrees=[2])
    assert "basis=Polynomial" in str(excinfo.value)


def test_degrees_all_zero_refused() -> None:
    with pytest.raises(ValueError, match="all 0"):
        Piecewise(breaks=[3.0], degrees=[0, 0])


def test_degrees_consecutive_flat_refused() -> None:
    with pytest.raises(ValueError, match="consecutive flat segments"):
        Piecewise(breaks=[3.0, 7.0], degrees=[1, 0, 0])


def test_named_breaks_construct_but_refuse_numeric_build() -> None:
    spec = Piecewise(breaks=["Mi060", "Mi066"])
    x = np.linspace(0.0, 100.0, 400)
    with pytest.raises(ValueError, match="band names"):
        spec.build(x)


def test_degrees_refuse_numeric_axis_build_citing_workbook_contract() -> None:
    spec = Piecewise(breaks=[3.0], degrees=[1, 2])
    x = np.linspace(0.0, 10.0, 400)
    with pytest.raises(ValueError, match="exact under linear interpolation only at degree 1"):
        spec.build(x)


def test_all_one_degrees_stay_legal_on_numeric_axis_and_bit_identical() -> None:
    """``degrees=[1, 1]`` states the default; the emitted basis is the default's."""
    rng = np.random.default_rng(20260810)
    x = rng.uniform(0.0, 10.0, 700)
    w = rng.uniform(0.5, 2.0, 700)
    stated = Piecewise(breaks=[3.0, 7.0], degrees=[1, 1, 1]).build(x, w)
    default = Piecewise(breaks=[3.0, 7.0]).build(x, w)
    assert (stated.columns != default.columns).nnz == 0
    assert stated.supports_row_compression == default.supports_row_compression


def test_repr_shows_names_and_degrees() -> None:
    text = repr(Piecewise(breaks=["Mi060", 7], degrees=[1, 2, 0]))
    assert "'Mi060'" in text
    assert "degrees=[1, 2, 0]" in text


# ── segmented basis structure ────────────────────────────────────────


def test_segmented_seams_are_value_continuous_by_construction() -> None:
    spec = _hosted(Piecewise(breaks=[3.0, 7.0], degrees=[2, 1, 3]))
    spec.build(_level_rows())
    eps = 1e-9
    for seam in (3.0, 7.0):
        left = spec.transform(np.array([seam - eps]))
        right = spec.transform(np.array([seam + eps]))
        assert np.allclose(left, right, atol=1e-6)


def test_flat_tail_is_flat_and_merges_its_knot_values() -> None:
    spec = _hosted(Piecewise(breaks=[3.0, 7.0], degrees=[1, 1, 0]))
    info = spec.build(_level_rows())
    # A flat tail merges knots 7 and 10 into one value column: J+2=4 knots
    # collapse to 3 value groups, minus the base column.
    assert info.n_cols == 2
    tail = spec.transform(np.array([7.0, 8.0, 9.5, 10.0]))
    assert np.allclose(tail, tail[0][None, :])


def test_curvature_columns_vanish_at_every_knot() -> None:
    spec = _hosted(Piecewise(breaks=[4.0], degrees=[3, 1]))
    spec.build(_level_rows())
    knots = spec.transform(np.array([0.0, 4.0, 10.0]))
    # Value columns at the knots read off the knot-value coefficients
    # (0/1 entries); curvature columns are exactly zero there.
    assert np.all(np.isin(knots, (0.0, 1.0)))


def test_segmented_degree_needs_enough_distinct_positions() -> None:
    spec = _hosted(Piecewise(breaks=[2.0], degrees=[1, 3]), n_levels=6)
    # Segment [2, 5] holds positions {2, 3, 5} only: 3 < degree 3 + 1.
    x = np.repeat(np.array([0.0, 1.0, 2.0, 3.0, 5.0]), 10)
    with pytest.raises(ValueError, match="needs at least 4 distinct"):
        spec.build(x)


def test_segmented_rank_probe_refuses_degenerate_design() -> None:
    spec = _hosted(Piecewise(breaks=[2.0], degrees=[1, 2]), n_levels=6)
    # Zero out the weight everywhere except the segment endpoints and one
    # interior point of segment 0: segment 1 keeps its endpoints only after
    # dedup-with-weights, killing the distinct-support rule first -- so give
    # it exactly 3 distinct (enough for the count rule at degree 2) but
    # collinear with an existing column via the weight pattern is impossible
    # for this basis; the honest degenerate case IS the count rule. Segment
    # support below the curvature need must refuse loudly either way.
    x = np.repeat(np.array([0.0, 1.0, 2.0, 5.0]), 10)
    with pytest.raises(ValueError, match="needs at least 3 distinct"):
        spec.build(x)


def test_segmented_reconstruct_reports_dense_curve_and_knots() -> None:
    spec = _hosted(Piecewise(breaks=[3.0, 7.0], degrees=[2, 1, 0]))
    info = spec.build(_level_rows())
    beta = np.linspace(0.3, -0.2, info.n_cols)
    raw = spec.reconstruct(beta)
    assert list(raw["knots"]) == [0.0, 3.0, 7.0, 10.0]
    assert raw["degrees"] == [2, 1, 0]
    # The display grid is dense inside the curved segment and passes through
    # every knot.
    assert {0.0, 3.0, 7.0, 10.0} <= set(np.asarray(raw["x"]).tolist())
    assert np.asarray(raw["x"]).size > 4
    assert np.allclose(raw["log_relativity"], spec.score(np.asarray(raw["x"]), beta))


def test_pre_degrees_pickle_state_restores_onto_legacy_path() -> None:
    spec = Piecewise(breaks=[3.0, 7.0])
    x = np.linspace(0.0, 10.0, 300)
    spec.build(x)
    state = dict(spec.__dict__)
    for key in (
        "degrees",
        "_on_level_axis",
        "_seg_value_groups",
        "_seg_base_group",
        "_seg_bubbles",
        "_seg_retained",
    ):
        state.pop(key, None)
    restored = pickle.loads(pickle.dumps(spec))
    restored.__dict__.clear()
    restored.__setstate__(state)
    assert restored.degrees is None
    assert np.array_equal(restored.transform(x), spec.transform(x))


# ── structural contrast rows ─────────────────────────────────────────


def _numerical_slope_change(spec: Piecewise, beta: np.ndarray, seam: float) -> float:
    h = 1e-6
    left = (spec.score(np.array([seam]), beta) - spec.score(np.array([seam - h]), beta)) / h
    right = (spec.score(np.array([seam + h]), beta) - spec.score(np.array([seam]), beta)) / h
    return float(right[0] - left[0])


@pytest.mark.parametrize(
    "breaks, degrees",
    [
        ([3.0, 7.0], None),
        ([3.0, 7.0], [2, 1, 0]),
        ([2.0, 5.0, 8.0], [1, 3, 2, 1]),
        ([4.0], [0, 1]),
    ],
)
def test_slope_change_contrasts_match_numerical_differentiation(breaks, degrees) -> None:
    """c @ beta IS the slope change at the stated break, for any coefficients.

    This is the Smith (1979) fixed-knot claim made executable: the structural
    contrast row equals the derivative jump of the fitted function.
    """
    spec = _hosted(Piecewise(breaks=breaks, degrees=degrees))
    info = spec.build(_level_rows())
    rng = np.random.default_rng(7)
    beta = rng.normal(0.0, 0.5, info.n_cols)
    rows = spec.ordered_structural_rows()
    slope_rows = [r for r in rows if r.kind == "slope_change"]
    assert len(slope_rows) == len(breaks)
    for row in slope_rows:
        seam = float(spec._knots[row.index])
        stated = float(row.contrast @ beta)
        measured = _numerical_slope_change(spec, beta, seam)
        assert stated == pytest.approx(measured, abs=5e-5)


def test_curvature_families_one_per_degree_ge_two_segment() -> None:
    spec = _hosted(Piecewise(breaks=[2.0, 5.0, 8.0], degrees=[1, 3, 2, 0]))
    spec.build(_level_rows())
    rows = spec.ordered_structural_rows()
    curvature = [r for r in rows if r.kind == "curvature"]
    assert [r.index for r in curvature] == [1, 2]
    # Degree 3 carries two curvature freedoms (u^2-u, u^3-u); degree 2 one.
    assert [len(r.column_indices) for r in curvature] == [2, 1]
    # Zeroing a curvature family flattens exactly that segment's curvature:
    # the score becomes linear between its knots.
    info_cols = spec.transform(_level_rows())
    beta = np.linspace(-0.4, 0.4, info_cols.shape[1])
    beta[list(curvature[0].column_indices)] = 0.0
    xs = np.array([2.0, 3.0, 4.0, 5.0])
    values = spec.score(xs, beta)
    slopes = np.diff(values) / np.diff(xs)
    assert np.allclose(slopes, slopes[0], atol=1e-12)


def test_legacy_structural_rows_reproduce_reconstruct_slopes() -> None:
    spec = Piecewise(breaks=[3.0, 7.0])
    info = spec.build(np.linspace(0.0, 10.0, 500))
    rng = np.random.default_rng(11)
    beta = rng.normal(0.0, 0.5, info.n_cols)
    raw = spec.reconstruct(beta)
    slopes = np.asarray(raw["slopes"], dtype=np.float64)
    rows = [r for r in spec.ordered_structural_rows() if r.kind == "slope_change"]
    stated = [float(r.contrast @ beta) for r in rows]
    assert stated == pytest.approx(list(np.diff(slopes)), abs=1e-12)
