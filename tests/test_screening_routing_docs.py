"""Documentation contracts for the screening routing paragraph.

``screen_interactions``' budget paragraph drifted away from the constants it
describes and stayed wrong for weeks: it quoted a penalized multiplier of 16
against a live ``_PENALIZED_LADDER_COST`` of 2, a ceiling of 678 against a real
one of 1357, per-pair times from a superseded implementation, and it said a
too-wide ``spline_cat`` pair is "refused with a NaN row" when the module
retries it through the arrow kernel.  A reader could not tell from it which
kernel scores their pair.

These tests re-derive every figure in that paragraph from the live constants
and check the routing claims against the router, so the paragraph cannot drift
from the code again without failing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import superglm.model.screening_ops as ops
from superglm import SuperGLM
from superglm.features import Categorical, Spline
from superglm.model.screening_ops import (
    _CUBIC_BUDGET_FACTOR,
    _INTERMEDIATE_BUDGET_FACTOR,
    _PENALIZED_LADDER_COST,
)

BUDGETS = (2.0, 4.0, 8.0, 16.0)
# Line wrapping is not part of the contract, so match against one flat line.
DOC = " ".join((SuperGLM.screen_interactions.__doc__ or "").split())


def _cubic_ceiling(max_cells: int, penalized: bool) -> int:
    """The widest block ``_within_cubic_budget`` admits, by search not formula."""
    cost = _PENALIZED_LADDER_COST if penalized else 1
    k = 1
    while cost * (k + 1) ** 3 <= _CUBIC_BUDGET_FACTOR * max_cells:
        k += 1
    return k


def _quadratic_ceiling(max_cells: int) -> int:
    k = 1
    while (k + 1) ** 2 <= _INTERMEDIATE_BUDGET_FACTOR * max_cells:
        k += 1
    return k


def _route(pair, model, df, y, **kw):
    """Which kernel scored ``pair``: the dense moments or the arrow kernel."""
    seen: dict[str, object] = {}
    real_curv, real_struct = ops.pair_score_curvature, ops.spline_cat_moments

    def spy_struct(menu_l, S_l, S_cell, W_cell, level_rows):
        seen.update(route="arrow", k_s=int(menu_l.shape[1]), levels=int(level_rows.size))
        return real_struct(menu_l, S_l, S_cell, W_cell, level_rows)

    def spy_curv(B_a, B_b, S_cell, W_cell):
        seen.update(route="dense", k_s=int(B_a.shape[1]), levels=int(B_b.shape[1]))
        return real_curv(B_a, B_b, S_cell, W_cell)

    ops.spline_cat_moments, ops.pair_score_curvature = spy_struct, spy_curv
    try:
        row = model.screen_interactions(df, y, candidates=[pair], edf0=BUDGETS, **kw).iloc[0]
    finally:
        ops.spline_cat_moments, ops.pair_score_curvature = real_struct, real_curv
    return seen.get("route", "refused"), seen, row


def _spline_cat_frame(levels, support, rows, seed=0):
    rng = np.random.default_rng(seed)
    grid = np.linspace(0.0, 1.0, support)
    return pd.DataFrame(
        {
            "x": grid[np.arange(rows) % support],
            "g": np.array([f"L{i}" for i in range(levels)])[rng.integers(0, levels, rows)],
        }
    ), rng.normal(size=rows)


def test_the_documented_ceilings_are_the_ones_the_constants_produce():
    """The two ceilings, and the way they scale, come from the live constants.

    ``_PENALIZED_LADDER_COST`` fell from 16 to 2 when the ladder started
    sharing one decomposition; the paragraph kept quoting 16 and the 678 that
    followed from it.
    """
    assert _cubic_ceiling(5_000_000, penalized=False) == 1709
    assert _cubic_ceiling(5_000_000, penalized=True) == 1357
    assert "1709" in DOC and "1357" in DOC
    assert "678" not in DOC, "678 followed from a penalized multiplier of 16, retired"
    assert "16x the work" not in DOC

    # The cap is a function of max_cells, not a constant, and the docstring
    # has to say so with the figures the constants actually give.
    assert "(1000 * max_cells)^(1/3)" in DOC
    assert "(500 * max_cells)^(1/3)" in DOC
    for max_cells, ceiling in (
        (5_000_000, 1357),
        (1_000_000, 793),
        (100_000, 368),
        (50_000, 292),
        (10_000, 170),
    ):
        assert _cubic_ceiling(max_cells, penalized=True) == ceiling
        assert f"{ceiling} at " in DOC  # "170" alone is a substring of "1709"

    # Below the crossover the allocation gate is the binding one, so the cubic
    # formula alone would overstate the cap.
    assert _quadratic_ceiling(2_000) < _cubic_ceiling(2_000, penalized=True)
    assert _quadratic_ceiling(10_000) > _cubic_ceiling(10_000, penalized=True)
    assert "2 * sqrt(max_cells)" in DOC and "3906" in DOC
    # ...and the unpenalized crossover is four times higher, which the
    # paragraph now says rather than leaving the cat_cat reader to infer it.
    assert _quadratic_ceiling(15_625) == _cubic_ceiling(15_625, penalized=False)
    assert "~15625" in DOC


def test_the_paragraph_carries_no_per_pair_timings():
    """The quoted times were measured against a superseded implementation.

    Timings belong with the constants they calibrate, in the module that owns
    them; repeating stale ones in the public docstring is how this drifted.
    """
    opening, closing = "``max_cells`` bounds allocation AND time", "A categorical margin never bins"
    assert opening in DOC and closing in DOC, "the paragraph was reworded; re-anchor this test"
    paragraph = DOC.split(opening, 1)[1].split(closing, 1)[0]
    for stale in ("1.3 s", "1.9 s", "0.81 s", "0.67 s"):
        assert stale not in paragraph
    assert " s per pair" not in paragraph
    assert "times are deliberately not quoted here" in paragraph


@pytest.mark.slow
def test_a_too_wide_spline_cat_pair_is_retried_not_refused():
    """The block-dimension exit hands the pair to the arrow kernel.

    The paragraph said such a pair "is refused with a NaN row", which is what
    happens to every OTHER kind.  A reader who believed it could not tell that
    the arrow kernel had scored their pair at all.
    """
    assert "refused with a NaN row by the same" not in DOC
    assert "RETRIED through the arrow kernel" in DOC
    # ...and the replacement is pinned positively, or deleting it outright
    # leaves this module green while the reader loses the diagnostic.
    assert "A NaN row is not by itself a routing signal" in DOC
    assert "on EITHER path" in DOC
    # the same counting correction, one paragraph down, for numeric_cat
    assert "1710 UNPINNED levels" in DOC

    # A library-default Spline() margin is 13 columns wide in the probe, so it
    # crosses the 1357 cap at 105 contrasts: 13 * 104 = 1352, 13 * 105 = 1365.
    assert "``L = 106`` UNPINNED levels" in DOC
    for levels, expected in ((105, "dense"), (106, "arrow")):
        df, y = _spline_cat_frame(levels, support=400, rows=8_000)
        model = SuperGLM(family="gaussian", features={"x": Spline(), "g": Categorical()})
        model.fit_reml(df, y)
        route, seen, row = _route(("x", "g"), model, df, y)
        assert seen["k_s"] == 13
        assert seen["levels"] == levels - 1, "contrasts, one per non-base level"
        assert route == expected, f"{levels} levels, block {13 * (levels - 1)}"
        assert np.isfinite(row["z"]), "the wide pair is scored, not refused"


@pytest.mark.slow
def test_a_pinned_level_widens_no_block_so_the_threshold_is_the_unpinned_count():
    """The threshold counts CONTRAST columns, not the declared level universe.

    ``Categorical(levels=...)`` may declare levels that no training row
    reaches; those are pinned to base and carry no column, so the probe block
    is ``k_spline * len(non-base)``.  Reading the declared universe against
    the 106-level threshold misroutes the pair on paper.
    """
    assert "declared ``levels=`` universe can be far larger" in DOC
    assert "pinned to base and contributes none" in DOC

    declared, populated = 140, 105
    levels = [f"L{i}" for i in range(declared)]
    rng = np.random.default_rng(0)
    grid = np.linspace(0.0, 1.0, 400)
    df = pd.DataFrame(
        {
            "x": grid[np.arange(8_000) % 400],
            "g": pd.Categorical(
                np.array(levels[:populated])[rng.integers(0, populated, 8_000)],
                categories=levels,
            ),
        }
    )
    y = rng.normal(size=8_000)
    model = SuperGLM(
        family="gaussian",
        features={"x": Spline(), "g": Categorical(levels=levels)},
    )
    with pytest.warns(UserWarning, match="pinned to base"):
        model.fit_reml(df, y)

    route, seen, row = _route(("x", "g"), model, df, y)
    assert len(model._specs["g"]._non_base) == populated - 1
    assert seen["levels"] == populated - 1, "contrasts, not the declared universe"
    assert seen["k_s"] * seen["levels"] == 1352 <= _cubic_ceiling(5_000_000, penalized=True)
    assert route == "dense", "34 more declared levels than the threshold, still dense"
    assert np.isfinite(row["z"])


@pytest.mark.slow
def test_the_support_exit_routes_a_block_far_below_the_cap():
    """The block-dimension gate is not the only door into the arrow kernel.

    A pair whose DENSE support intermediate ``n_x*(L-1)^2 + L*k_s^2`` blows
    ``4 * max_cells`` is handed over while the arrow kernel's TRANSPOSED
    intermediate ``n_x*k_s^2`` still fits — with a block dimension nowhere
    near the cap.  Believing the cap was the only door is what made "the arrow
    kernel only ever sees factors above a hundred levels" look true.
    """
    assert "far BELOW the cap" in DOC
    assert "``k = 231``" in DOC
    assert "46,119" in DOC

    levels, support, k_s = 22, 46_119, 11
    df, y = _spline_cat_frame(levels, support=support, rows=2 * support)
    model = SuperGLM(
        family="gaussian",
        features={"x": Spline(kind="ps", n_knots=8), "g": Categorical()},
    )
    model.fit_reml(df, y)
    route, seen, row = _route(("x", "g"), model, df, y)

    assert seen["k_s"] == k_s
    assert seen["levels"] == levels - 1
    block = k_s * (levels - 1)
    assert block == 231 < _cubic_ceiling(5_000_000, penalized=True)
    # support = 46,119 is a BOUNDARY fixture, not an arbitrary size: the
    # window between these two inequalities is ~1.7% wide, and the arrow
    # kernel's evaluation allowance is the gate that closes it from the other
    # side (setup charges 245.8M of the 250M ceiling here).  Assert all three
    # so a failure names the gate that moved instead of blaming the route.
    assert support * (levels - 1) ** 2 + levels * k_s**2 > _INTERMEDIATE_BUDGET_FACTOR * 5_000_000
    assert support * k_s**2 <= _INTERMEDIATE_BUDGET_FACTOR * 5_000_000
    assert ops._structured_evaluation_allowance(5_000_000, support, k_s, levels) >= 2
    assert route == "arrow", "the support exit, not the block-dimension one"
    assert not bool(row["approx"]), "the handoff is taken to keep the score EXACT"
