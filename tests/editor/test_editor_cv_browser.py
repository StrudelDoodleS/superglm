from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold

from superglm import Categorical, Spline, SuperGLM, cross_validate
from superglm.editor import EditorSession

pytest.importorskip("playwright.sync_api")
pytestmark = pytest.mark.browser


def _model() -> SuperGLM:
    return SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        features={"age": Spline(n_knots=6), "region": Categorical(base="first")},
    )


@pytest.fixture
def cv_widget():
    rng = np.random.default_rng(20261003)
    n = 400
    X = pd.DataFrame({"age": rng.uniform(18.0, 80.0, n), "region": rng.choice(["C", "A", "B"], n)})
    eta = -0.5 + 0.2 * np.sin(X["age"].to_numpy() / 12.0) + 0.2 * (X["region"] == "B")
    y = rng.poisson(np.exp(eta)).astype(np.float64)
    supplied = cross_validate(
        _model(),
        X.iloc[:300],
        y[:300],
        cv=KFold(3, shuffle=True, random_state=0),
        scoring=("deviance", "gini", "nll"),
        return_estimators=True,
    )
    session = EditorSession.from_model(
        _model().fit(X.iloc[:300], y[:300]),
        train_data=(X.iloc[:300], y[:300]),
        validation_data=(X.iloc[300:], y[300:]),
        cv=supplied,
    )
    widget = session.widget()
    try:
        yield widget
    finally:
        widget.close()


def _open_cv_tab(chromium_browser, widget, width: int = 1280):
    page = chromium_browser.new_page(viewport={"width": width, "height": 900})
    page.goto(widget.app_url, wait_until="domcontentloaded")
    page.locator("#chart path.edited").first.wait_for()
    page.locator("#cvTab").click()
    page.locator("#reportFrame .cv-card").first.wait_for()
    return page


def test_cv_tab_renders_a_supplied_result(cv_widget, chromium_browser):
    page = _open_cv_tab(chromium_browser, cv_widget)
    try:
        assert page.locator("#reportTitle").text_content() == "Cross-validation"
        assert page.locator("#reportStatus").text_content() == (
            "3 folds · KFold · 300 rows · supplied with edit(model, cv=result)"
        )
        assert page.locator("#reportFrame .cv-card").count() == 3
        assert page.locator("#reportFrame .cv-fold-table tbody tr").count() == 4
        assert sorted(page.locator("#reportFrame .cv-term-name").all_text_contents()) == [
            "age",
            "region",
        ]
        page.locator('#reportFrame [data-cv-term="region"]').click()
        assert page.locator("#reportFrame .cv-level title").all_text_contents() == ["A", "B", "C"]
        page.locator("#cvTermSearch").fill("reg")
        assert page.locator("#reportFrame .cv-term").count() == 1
        page.locator("#cvTermSearch").press("Escape")
        assert page.locator("#reportFrame .cv-term").count() == 2
    finally:
        page.close()


def test_run_cv_from_the_tab_adds_the_current_model(cv_widget, chromium_browser):
    page = _open_cv_tab(chromium_browser, cv_widget)
    try:
        page.locator('#reportFrame [data-cv-start="cv"]').click()
        page.locator('#reportFrame .cv-card-row[data-origin="run"]').first.wait_for()
        assert page.locator('#reportFrame .cv-card-row[data-origin="run"]').count() == 3
        assert page.locator('#reportFrame .cv-job[data-cv-job="cv"]').text_content() == (
            "Run CV finished. Its folds are shown beside the supplied ones."
        )
    finally:
        page.close()


_RUN_CV_DISABLED = """() => document.querySelector('#reportFrame [data-cv-start="cv"]').disabled"""


def test_undo_and_redo_of_a_waiting_change_update_the_open_tab(cv_widget, chromium_browser):
    """Undo and Redo of a waiting change keep the model revision; the tab follows them anyway.

    Its waiting chip, Run CV's state and its reason come from the report,
    which a change of revision alone used to refresh.
    """
    cv_widget.session.stage_structural("collapse", "region", {"levels": ["A", "B"]})
    page = _open_cv_tab(chromium_browser, cv_widget)
    chips = page.locator("#reportFrame .cv-chip")
    try:
        assert page.evaluate(_RUN_CV_DISABLED) is True
        assert chips.all_text_contents() == ["1 change waiting for refit"]

        page.locator("#undoAction").click()
        page.wait_for_function(f"() => !({_RUN_CV_DISABLED})()")
        assert chips.count() == 0 and cv_widget.session.pending == []

        page.locator("#redoAction").click()
        page.wait_for_function(_RUN_CV_DISABLED)
        assert chips.all_text_contents() == ["1 change waiting for refit"]
        assert len(cv_widget.session.pending) == 1
    finally:
        page.close()


_FOLD_OPACITY = """(selector) => [0, 1, 2].map((fold) => [
  ...document.querySelectorAll(`#reportFrame [data-cv-chart] ${selector}[data-fold="${fold}"]`)
].map((node) => Number(getComputedStyle(node).opacity)))"""


def _lit(page, selector: str) -> list[str]:
    """Each fold's marks on the chart, 'full' or 'dimmed'; all of one fold's marks agree."""
    states = []
    for opacities in page.evaluate(_FOLD_OPACITY, selector):
        assert opacities, "a fold has no marks"
        assert len(set(opacities)) == 1, opacities
        states.append(opacities[0])
    plain = max(states)
    return ["dimmed" if value < 0.5 * plain else "full" for value in states]


def test_a_legend_entry_picks_out_its_fold_on_hover_and_focus(cv_widget, chromium_browser):
    page = _open_cv_tab(chromium_browser, cv_widget)
    try:
        page.locator('#reportFrame [data-cv-term="region"]').click()
        keys = page.locator("#reportFrame [data-cv-chart] .cv-fold-key")
        assert [text.strip() for text in keys.all_text_contents()] == ["Fold 1", "Fold 2", "Fold 3"]
        # Fold 1 leftmost: within each level the folds sit left to right in order.
        for level in range(3):
            xs = [
                page.locator(
                    f'#reportFrame .cv-fold-dot[data-fold="{fold}"][data-level="{level}"]'
                ).bounding_box()["x"]
                for fold in range(3)
            ]
            assert xs == sorted(xs) and len(set(xs)) == 3, (level, xs)

        assert _lit(page, ".cv-fold-dot") == ["full", "full", "full"]
        keys.nth(1).hover()
        assert _lit(page, ".cv-fold-dot") == ["dimmed", "full", "dimmed"]
        page.mouse.move(5, 5)
        assert _lit(page, ".cv-fold-dot") == ["full", "full", "full"]

        keys.nth(2).focus()
        assert _lit(page, ".cv-fold-dot") == ["dimmed", "dimmed", "full"]
        keys.nth(2).blur()
        assert _lit(page, ".cv-fold-dot") == ["full", "full", "full"]

        page.locator('#reportFrame [data-cv-term="age"]').click()
        page.locator("#reportFrame [data-cv-chart] .cv-fold-key").first.hover()
        assert _lit(page, ".cv-fold-line") == ["full", "dimmed", "dimmed"]
    finally:
        page.close()


def test_final_fit_from_the_tab_shows_on_final_fit_and_in_export(cv_widget, chromium_browser):
    page = _open_cv_tab(chromium_browser, cv_widget)
    try:
        page.locator('#reportFrame [data-cv-start="final_fit"]').click()
        done = page.locator('#reportFrame .cv-job[data-cv-job="final_fit"][data-status="done"]')
        done.wait_for()
        assert done.text_content() == (
            "Final fit finished on 400 rows. Export offers it as Final fit model."
        )
        page.locator("#finalTab").click()
        section = page.locator("#reportFrame .final-fit")
        section.locator("table").wait_for()
        assert "Refitted on 400 train and validation rows" in section.text_content()
        page.locator("#exportAction").click()
        assert page.locator("#exportFinalFit").is_enabled()
    finally:
        page.close()


def test_the_tab_fills_the_report_panel_and_its_header_scrolls_with_it(cv_widget, chromium_browser):
    page = _open_cv_tab(chromium_browser, cv_widget, width=1600)
    try:
        panel = page.locator("#reportPanel")
        inner = panel.evaluate(
            "(node) => node.clientWidth - parseFloat(getComputedStyle(node).paddingLeft)"
            " - parseFloat(getComputedStyle(node).paddingRight)"
        )
        for selector in ("#reportPanel .report-header", "#reportFrame", "#reportFrame .cv-cards"):
            assert page.locator(selector).bounding_box()["width"] >= inner - 1, selector
        # The header scrolls away with the cards, so it never sits over them.
        header = page.locator("#reportPanel .report-header")
        card = page.locator("#reportFrame .cv-card").first
        gap = card.bounding_box()["y"] - header.bounding_box()["y"]
        card.hover()
        page.mouse.wheel(0, 200)
        page.wait_for_function(
            "() => document.querySelector('#reportPanel .report-header')"
            ".getBoundingClientRect().top < 0",
            timeout=5000,
        )
        assert abs(card.bounding_box()["y"] - header.bounding_box()["y"] - gap) <= 0.5
    finally:
        page.close()


def test_a_hand_edited_term_reads_held_after_run_cv_and_sorts_last(cv_widget, chromium_browser):
    with cv_widget._lock:
        cv_widget.session.select_levels("region", ["B"])
        cv_widget.session.shift("region", 0.1)
    page = _open_cv_tab(chromium_browser, cv_widget, width=1600)
    try:
        page.locator('#reportFrame [data-cv-start="cv"]').click()
        page.locator('#reportFrame .cv-card-row[data-origin="run"]').first.wait_for()
        rows = page.locator("#reportFrame .cv-term")
        assert rows.locator(".cv-term-name").all_text_contents() == ["age", "region"]
        held = rows.nth(1).locator(".cv-term-held")
        assert held.text_content() == "held"
        assert not any(char.isdigit() for char in rows.nth(1).text_content())
        held.hover()
        popover = page.locator("#uiPopover")
        popover.wait_for(state="visible")
        assert popover.locator("[data-popover-heading]").inner_text() == "Hand-edited"
        assert popover.locator("[data-popover-description]").inner_text() == (
            "The same curve on every fold."
        )
    finally:
        page.close()


def _card_rows(page) -> list[int]:
    """How many export format cards sit on each row, top to bottom."""
    tops = [
        round(page.locator(".export-format-card").nth(index).bounding_box()["y"])
        for index in range(page.locator(".export-format-card").count())
    ]
    return [tops.count(top) for top in sorted(set(tops))]


def test_the_export_format_cards_never_leave_one_alone_on_a_row(cv_widget, chromium_browser):
    page = _open_cv_tab(chromium_browser, cv_widget, width=1600)
    try:
        page.locator("#exportAction").click()
        page.locator("#exportDialog .export-format-card").first.wait_for()
        assert _card_rows(page) in ([2, 2], [4])
        page.set_viewport_size({"width": 560, "height": 900})
        assert _card_rows(page) in ([2, 2], [4])
    finally:
        page.close()
