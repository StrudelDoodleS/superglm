"""Categorical interaction cells are read by their pair of levels, never by a joined label.

``CategoricalInteraction.reconstruct`` labels its cells ``"level1:level2"``, which two cells
share once a level holds a colon: ``("x", "y:z")`` and ``("x:y", "z")`` are both
``"x:y:z"``, and the base cell ``("a", "b:c")`` reads like the fitted cell ``("a:b", "c")``.
Every reader a user sees (the rating table and its workbook, the plot data, both heatmaps and
``relativities()``) must still give each cell its own factor. The reference is
``model.predict``: a cell's relativity is the double ratio
``mu(l1, l2) mu(b1, b2) / (mu(l1, b2) mu(b1, l2))``, which cancels the intercept and both main
effects.
"""

from __future__ import annotations

import itertools
import warnings
from io import BytesIO

import numpy as np
import pandas as pd
import pytest
from openpyxl import load_workbook

from superglm import Categorical, InteractionCellLabelWarning, SuperGLM
from superglm.export.excel import write_rating_table_workbook
from superglm.export.rating_tables import build_rating_table_payload
from superglm.export.summary import build_summary_export_payload
from superglm.features.interaction import interaction_cells
from superglm.plotting import plot_interaction
from superglm.plotting.data import build_interaction_plot_data

# Sorted, so ``base="first"`` makes "a" and "b:c" the bases.
COLON_P = ["a", "a:b", "x", "x:y"]
COLON_Q = ["b:c", "c", "y:z", "z"]


def _fit(p_levels, q_levels, seed=0):
    rng = np.random.default_rng(seed)
    n = 4000
    X = pd.DataFrame({"p": rng.choice(p_levels, n), "q": rng.choice(q_levels, n)})
    cell = {pair: rng.normal(0.0, 0.4) for pair in itertools.product(p_levels, q_levels)}
    eta = np.array([cell[pair] for pair in zip(X["p"], X["q"])]) - 1.0
    y = rng.poisson(np.exp(eta)).astype(float)
    model = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        features={"p": Categorical(base="first"), "q": Categorical(base="first")},
        interactions=[("p", "q")],
    )
    model.fit(X, y)
    return model, X, y


def _reference(model, p_levels, q_levels) -> dict[tuple[str, str], float]:
    grid = pd.DataFrame(list(itertools.product(p_levels, q_levels)), columns=["p", "q"])
    mu = dict(zip(map(tuple, grid.to_numpy()), model.predict(grid)))
    b1, b2 = p_levels[0], q_levels[0]
    return {(a, b): mu[(a, b)] * mu[(b1, b2)] / (mu[(a, b2)] * mu[(b1, b)]) for a, b in mu}


@pytest.fixture(scope="module")
def colon_fit():
    model, X, y = _fit(COLON_P, COLON_Q)
    return model, X, y, _reference(model, COLON_P, COLON_Q)


def _grid(reference, p_levels, q_levels) -> np.ndarray:
    return np.array([[reference[(a, b)] for b in q_levels] for a in p_levels])


def test_the_fixture_has_colliding_labels(colon_fit):
    model, *_ = colon_fit
    labels = [f"{a}:{b}" for a, b in itertools.product(COLON_P, COLON_Q)]
    assert len(set(labels)) < len(labels)
    assert set(_pairs(model)) == set(itertools.product(COLON_P[1:], COLON_Q[1:]))


def test_rating_table_gives_each_colon_cell_its_own_factor(colon_fit):
    model, X, y, reference = colon_fit
    payload = build_rating_table_payload(model, X, y, impact_bins=())
    (block,) = payload.interactions
    assert list(block.table.columns) == ["p", *COLON_Q]
    assert list(block.table["p"]) == COLON_P
    np.testing.assert_allclose(
        block.table.iloc[:, 1:].to_numpy(dtype=float),
        _grid(reference, COLON_P, COLON_Q),
        rtol=1e-9,
    )

    buffer = BytesIO()
    write_rating_table_workbook(
        payload,
        buffer,
        sheet_name="Rating Tables",
        summary_sheet_name="Model Summary",
        impact_sheet_name="Discretization Impact",
    )
    sheet = load_workbook(BytesIO(buffer.getvalue()), data_only=True)["Rating Tables"]
    title_row = next(
        cell.row for (cell,) in sheet.iter_rows(min_col=1, max_col=1) if cell.value == "p:q"
    )
    rows = list(
        sheet.iter_rows(
            min_row=title_row + 2,
            max_row=title_row + 2 + len(COLON_P),
            max_col=1 + len(COLON_Q),
            values_only=True,
        )
    )
    assert list(rows[0]) == ["p", *COLON_Q]
    assert [row[0] for row in rows[1:]] == COLON_P
    np.testing.assert_allclose(
        np.array([row[1:] for row in rows[1:]], dtype=float),
        _grid(reference, COLON_P, COLON_Q),
        rtol=1e-9,
    )


def test_plot_data_and_heatmap_give_each_colon_cell_its_own_factor(colon_fit):
    model, X, _, reference = colon_fit
    effect = build_interaction_plot_data(model, "p:q", X=X)["effect"]
    got = {(row.p, row.q): row.relativity for row in effect.itertuples()}
    assert got.keys() == reference.keys()
    np.testing.assert_allclose(
        [got[pair] for pair in reference], list(reference.values()), rtol=1e-9
    )

    log_grid = np.log(_grid(reference, COLON_P, COLON_Q))
    figure = plot_interaction(model, "p:q")
    np.testing.assert_allclose(figure.axes[0].get_images()[0].get_array(), log_grid, atol=1e-9)


def test_plotly_heatmap_gives_each_colon_cell_its_own_factor(colon_fit):
    pytest.importorskip("plotly")
    model, _, _, reference = colon_fit
    figure = plot_interaction(model, "p:q", engine="plotly")
    log_grid = np.log(_grid(reference, COLON_P, COLON_Q))
    np.testing.assert_allclose(np.asarray(figure.data[0].z), log_grid, atol=1e-9)


def test_relativities_give_each_colon_cell_its_own_factor(colon_fit):
    model, _, _, reference = colon_fit
    table = model.relativities()["p:q"]
    pairs = _pairs(model)
    # The labels stay "level1:level2" in pair order, so two rows share "x:y:z"; the trailing
    # level1 and level2 columns say which cell each row is.
    assert list(table.columns) == ["level", "relativity", "log_relativity", "level1", "level2"]
    assert list(table["level"]) == [f"{a}:{b}" for a, b in pairs]
    assert list(zip(table["level1"], table["level2"])) == pairs
    by_cell = table.set_index(["level1", "level2"])["relativity"]
    np.testing.assert_allclose(
        [by_cell[pair] for pair in pairs], [reference[pair] for pair in pairs], rtol=1e-9
    )
    np.testing.assert_allclose(np.exp(table["log_relativity"]), table["relativity"], rtol=1e-12)


def _pairs(model) -> list[tuple[str, str]]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InteractionCellLabelWarning)
        return [tuple(pair) for pair in model.reconstruct_feature("p:q")["pairs"]]


def test_term_inference_reads_each_colon_cell_by_its_pair_and_warns_of_merged_labels(colon_fit):
    model, _, _, reference = colon_fit
    with pytest.warns(InteractionCellLabelWarning) as record:
        inference = model.term_inference("p:q")
    warning = record.pop(InteractionCellLabelWarning)
    message = str(warning.message)
    assert "'x:y:z' is ('x', 'y:z') and ('x:y', 'z')" in message
    assert "'a:b:c' is ('a', 'b:c') and ('a:b', 'c')" in message
    assert "pair_relativity" in message
    assert warning.filename == __file__  # at the caller's line, not inside superglm

    pairs = _pairs(model)
    assert list(inference.pair_relativity) == list(inference.pair_log_relativity) == pairs
    np.testing.assert_allclose(
        [inference.pair_relativity[pair] for pair in pairs],
        [reference[pair] for pair in pairs],
        rtol=1e-9,
    )
    np.testing.assert_allclose(
        np.exp(list(inference.pair_log_relativity.values())),
        list(inference.pair_relativity.values()),
        rtol=1e-12,
    )
    # The label-keyed dict is unchanged: one entry per label, so it merges two cells.
    assert len(inference.relativity) == len(pairs) - 1


def test_reconstruct_feature_warns_of_merged_labels_and_names_its_per_pair_lists(colon_fit):
    model, _, _, reference = colon_fit
    with pytest.warns(InteractionCellLabelWarning, match="pair_relativities") as record:
        raw = model.reconstruct_feature("p:q")
    assert record.pop(InteractionCellLabelWarning).filename == __file__
    cells = interaction_cells(raw)
    np.testing.assert_allclose(list(cells.values()), [reference[pair] for pair in cells], rtol=1e-9)


def test_summary_names_the_cells_of_rows_that_share_a_name(colon_fit):
    model, _, _, reference = colon_fit
    summary = model.summary()
    note = "In row order, p:q[x:y:z] is (p=x, q=y:z) then (p=x:y, q=z)."
    assert note in " ".join(str(summary).split())
    assert note in summary._repr_html_()
    payload = build_summary_export_payload(model)
    assert any(note in text for text in payload.notes)
    # The note's order is the rows' order: each row's estimate is its cell's log relativity.
    shared = [row.estimate for row in payload.terms if row.term == "p:q[x:y:z]"]
    np.testing.assert_allclose(
        shared, np.log([reference[("x", "y:z")], reference[("x:y", "z")]]), rtol=1e-9
    )


def test_labels_that_do_not_collide_keep_their_dicts_and_raise_no_warning():
    p_levels, q_levels = ["A", "B"], ["base", "p", "z"]
    model, _, _ = _fit(p_levels, q_levels, seed=1)
    with warnings.catch_warnings():
        warnings.simplefilter("error", InteractionCellLabelWarning)
        inference = model.term_inference("p:q")
        raw = model.reconstruct_feature("p:q")
        text = str(model.summary())
    assert "In row order" not in text
    assert inference.relativity == raw["relativities"]
    assert inference.relativity == {
        f"{a}:{b}": v for (a, b), v in inference.pair_relativity.items()
    }


def test_rating_table_keeps_its_row_keys_when_a_level_is_named_like_the_row_column():
    p_levels, q_levels = ["A", "B"], ["base", "p", "z"]
    model, X, y = _fit(p_levels, q_levels, seed=1)
    reference = _reference(model, p_levels, q_levels)
    (block,) = build_rating_table_payload(model, X, y, impact_bins=()).interactions
    # The spline-by-categorical block's rule: the key column is suffixed, the levels kept.
    assert list(block.table.columns) == ["p (axis 1)", *q_levels]
    assert list(block.table["p (axis 1)"]) == p_levels
    np.testing.assert_allclose(
        block.table.iloc[:, 1:].to_numpy(dtype=float),
        _grid(reference, p_levels, q_levels),
        rtol=1e-9,
    )
