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


def _open_cv_tab(chromium_browser, widget):
    page = chromium_browser.new_page(viewport={"width": 1280, "height": 900})
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
