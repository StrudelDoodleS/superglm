"""Guards for the worth-gate benchmark's two derived readings.

The benchmark itself is a measurement and is not run in CI.  What is guarded
here is the arithmetic underneath it: the Cp identity that turns `T/phi >
2*edf0` into a threshold on `z`, and the participation ratio that reads the
shape of a score the total cannot see.  Both are cheap and deterministic; the
tests that fit real models are marked `slow`.
"""

from __future__ import annotations

import numpy as np
import pytest
from benchmarks import screening_worth_gate as gate
from benchmarks.screening_worth_gate import (
    HOT_CELLS,
    MATCHED,
    SHRINKAGE_ARMS,
    _build_parser,
    _make,
    _run_concentration,
    _run_gate_ladder,
    _run_shrinkage,
    _shrinkage_spec,
    cell_contributions,
    concentration,
    participation_ratio,
    worth_threshold,
)


@pytest.mark.parametrize("edf0", [1.0, 49.0, 225.0, 576.0, 1599.0])
def test_threshold_is_exactly_where_the_cp_criterion_switches(edf0: float) -> None:
    """`z > sqrt(edf0/2)` must be the same statement as `T/phi > 2*edf0`.

    The screen reports z, not T, so the gate is only usable if the two agree
    exactly rather than approximately.
    """
    at_criterion = 2.0 * edf0  # T/phi sitting exactly on Mallows' Cp
    z_at = (at_criterion - edf0) / np.sqrt(2.0 * edf0)
    assert z_at == pytest.approx(worth_threshold(edf0), rel=1e-12)

    for scale, expected_above in ((1.01, True), (0.99, False)):
        z = (scale * at_criterion - edf0) / np.sqrt(2.0 * edf0)
        assert bool(z > worth_threshold(edf0)) is expected_above


def test_threshold_grows_with_block_width_so_no_constant_can_replace_it() -> None:
    """The point of the gate: 8x8 and 41x41 cannot share a cutoff."""
    narrow, wide = worth_threshold(7**2), worth_threshold(40**2)
    assert narrow == pytest.approx(4.95, abs=0.01)
    assert wide == pytest.approx(28.28, abs=0.01)
    # a z that is decisive at 8x8 is below the bar at 41x41
    assert narrow < 12.0 < wide


def test_participation_ratio_counts_effective_contributors() -> None:
    assert participation_ratio(np.ones(40)) == pytest.approx(40.0)
    assert participation_ratio(np.array([1.0, 0.0, 0.0, 0.0])) == pytest.approx(1.0)
    # scale-free: doubling every contribution changes nothing
    t = np.array([5.0, 1.0, 1.0, 0.5])
    assert participation_ratio(2.0 * t) == pytest.approx(participation_ratio(t))


def test_cell_contributions_ignore_empty_cells() -> None:
    """Empty cells contribute nothing and must not inflate the null."""
    joint = np.array([0, 0, 3, 3, 3])
    resid = np.array([1.0, 1.0, 2.0, 2.0, 2.0])
    t, occupied = cell_contributions(resid, joint, n_cells=8, phi=1.0)
    assert occupied == 2
    assert t[1] == 0.0 and t[2] == 0.0
    # cell 0: n=2, mean=1 -> 2;  cell 3: n=3, mean=2 -> 12
    assert t[0] == pytest.approx(2.0)
    assert t[3] == pytest.approx(12.0)


def test_concentration_null_sits_at_one_for_chi_square_contributions() -> None:
    """k independent chi^2_1 contributions give P = k/3, so the ratio is 1.

    This is what makes the reading comparable across block widths -- the same
    property that makes PSST's z comparable across them.
    """
    rng = np.random.default_rng(4242)
    k = 1600
    ratios = [concentration(rng.chisquare(df=1, size=k), k) for _ in range(24)]
    assert np.mean(ratios) == pytest.approx(1.0, abs=0.08)


def test_concentration_separates_spiky_from_diffuse_at_equal_total() -> None:
    """The claim the benchmark rests on, isolated from any model fitting.

    Both vectors carry the same total score.  Every chi^2-family metric reads
    only that total, so only a shape statistic can tell them apart.
    """
    rng = np.random.default_rng(99)
    k = 1600
    diffuse = rng.chisquare(df=1, size=k)
    spiky = rng.chisquare(df=1, size=k)
    spiky[:HOT_CELLS] += diffuse.sum() / HOT_CELLS  # same total, 5 cells carry it
    spiky *= diffuse.sum() / spiky.sum()

    assert spiky.sum() == pytest.approx(diffuse.sum())
    assert concentration(diffuse, k) == pytest.approx(1.0, abs=0.15)
    assert concentration(spiky, k) < 0.25


def test_concentration_is_nan_rather_than_a_crash_on_a_dead_block() -> None:
    assert np.isnan(concentration(np.zeros(10), 10))
    assert np.isnan(concentration(np.ones(3), 0))


def test_parser_defaults_match_the_documented_run() -> None:
    args = _build_parser().parse_args([])
    assert (args.reps, args.n, args.wide_levels) == (3, 12_000, 41)


def test_generator_plants_exactly_the_advertised_number_of_live_cells() -> None:
    """A spiky truth with the wrong support would invalidate table 3."""
    data = _make("spike", 6.0, n_levels=9, n=2_000, seed=5)
    assert data.frame.shape == (2_000, 2)
    assert data.joint.max() < 81
    noise = _make("none", 0.0, n_levels=9, n=2_000, seed=5)
    # same seed, same parents: the only difference is the planted cells
    assert (data.joint == noise.joint).all()
    assert not np.allclose(data.y, noise.y)


def test_every_documented_shrinkage_arm_is_actually_built() -> None:
    """The guide quotes a holdout figure per arm, so each must be reproducible.

    The `pooled` arm is the one that went missing once already -- the guide
    cited its holdout gain while the benchmark had no such arm, leaving a
    documented number with no way to check it.
    """
    assert SHRINKAGE_ARMS == ("mains", "fixed", "pooled")

    kwargs, cols = _shrinkage_spec("mains")
    assert "interactions" not in kwargs and cols == ["g", "h"]

    kwargs, cols = _shrinkage_spec("fixed")
    assert kwargs["interactions"] == [("g", "h")]
    assert "gh" not in kwargs["features"]

    kwargs, cols = _shrinkage_spec("pooled")
    assert "interactions" not in kwargs
    assert cols == ["g", "h", "gh"]
    # the cell must be a RandomEffect, not another fixed factor -- a plain
    # Categorical here would silently make `pooled` a second `fixed`
    assert type(kwargs["features"]["gh"]).__name__ == "RandomEffect"


def test_unknown_shrinkage_arm_is_rejected_rather_than_silently_skipped() -> None:
    with pytest.raises(ValueError, match="unknown shrinkage arm"):
        _shrinkage_spec("credibility")


@pytest.mark.slow
def test_gate_ladder_thresholds_on_the_edf0_the_screen_reported(monkeypatch) -> None:
    """The Cp identity needs the SAME edf0 on both sides.

    For an unpenalized `cat_cat` the screen normalizes `z` by the block's
    ACHIEVED rank, which drops below `(n_levels - 1)**2` as soon as a joint
    cell is empty in the training split.  Re-deriving the nominal rank for the
    threshold compares a z on one scale against a bar on another.  The width
    here is deliberately sparse so the two values differ.
    """
    monkeypatch.setattr(gate, "LADDER", {10: (0.5,)})
    rows = _run_gate_ladder(reps=1, n=600)

    assert len(rows) == 1
    row = rows[0]
    assert row["nominal_edf0"] == 81.0
    # the split leaves cells empty, so the achieved rank is strictly lower
    assert row["edf0"] < row["nominal_edf0"]
    # and the published threshold is the one that goes with the reported z
    assert row["threshold"] == pytest.approx(worth_threshold(row["edf0"]))
    assert row["threshold"] != pytest.approx(worth_threshold(row["nominal_edf0"]))


@pytest.mark.slow
def test_shrinkage_table_reproduces_its_arms_and_prefers_pooling_over_fixed() -> None:
    """Small-width end-to-end: the arms run and report a full row each.

    Width is kept small so this stays a harness check. The ordering the guide
    reports (pooled beats fixed on holdout) is a property of the WIDE case and
    is not asserted here -- table 4 at the default width is where that lives.
    """
    rows = _run_shrinkage(reps=1, n=2_000, n_levels=6)
    assert [row["model"] for row in rows] == list(SHRINKAGE_ARMS)
    for row in rows:
        assert np.isfinite(row["holdout"]) and row["holdout"] > 0
        assert row["params"] >= 1 and row["edf"] > 0
    fixed = next(r for r in rows if r["model"] == "fixed")
    pooled = next(r for r in rows if r["model"] == "pooled")
    # whatever the holdout ordering at this width, pooling must spend strictly
    # fewer effective df than the unshrunk interaction -- that is what makes it
    # a different model class rather than a relabelling
    assert pooled["edf"] < fixed["edf"]


@pytest.mark.slow
def test_concentration_table_runs_end_to_end_and_ranks_spiky_below_diffuse() -> None:
    rows = _run_concentration(reps=1, n=2_400, n_levels=7)
    assert len(rows) == len(MATCHED)
    by_label = {row["label"]: row for row in rows}
    assert all(np.isfinite(row["concentration"]) for row in rows)

    # Only the ORDERING is asserted here. P/(k/3) is calibrated to 1 under the
    # null, but its variance is large at the small k this test uses to stay
    # fast, so pinning the null value here would be flaky -- that calibration
    # is covered at k=1600 by the pure-arithmetic test above.
    spiky = by_label["spiky 5 cells @ 8.0"]["concentration"]
    diffuse = by_label["diffuse sd=0.41"]["concentration"]
    assert spiky < diffuse
