"""The Diagnose tab in a browser: the verdict of every fit, and the fit that did not converge, on the first screen."""

from __future__ import annotations

import dataclasses
import warnings

import pytest
from sklearn.model_selection import KFold
from tests._lss_editor_fixtures import lognormal_case

from superglm import LogNormalLS, SuperLSS, cat, cross_validate, s

pytest.importorskip("playwright.sync_api")
pytestmark = pytest.mark.browser

# A notebook cell's frame, and a 16:9 window.
_FRAMES = [{"width": 1100, "height": 720}, {"width": 1600, "height": 900}]


def _cv_with_a_fold_that_did_not_converge():
    """Three folds: the second is a fit stopped at its second round, which cannot have converged.

    The others are the fixture's own converged fit, so no fold depends on where a search lands.
    """
    case = lognormal_case()
    cv = cross_validate(
        case.model,
        case.X,
        case.y,
        cv=KFold(3, shuffle=True, random_state=0),
        fit_mode="fit_reml",
        return_estimators=True,
        error_score="raise",
    )
    family = LogNormalLS()
    stopped = SuperLSS(
        family,
        family.mean(s("age", k=8), cat("region"), s("power", k=6)),
        family.scale(s("age", k=6), cat("region")),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stopped.fit_reml(case.X, case.y, max_reml_iter=2, outer="efs")
    assert not stopped.result_.converged
    return dataclasses.replace(cv, estimators=[case.model, stopped, case.model])


def test_diagnose_opens_on_the_fit_that_did_not_converge(open_lss_editor_page):
    with open_lss_editor_page(cv=_cv_with_a_fold_that_did_not_converge()) as (page, _session):
        page.locator("#diagnoseTab").click()
        frame = page.locator("#reportFrame")
        frame.locator(".diagnose-table").first.wait_for()
        assert frame.locator('[data-diagnose-fit][aria-pressed="true"]').inner_text() == "Fold 2"
        rows = frame.locator(".diagnose-fits tbody tr")
        assert rows.count() == 4
        # The word and the cross, not the colour, tell it apart.
        failed = frame.locator('.diagnose-fits tr[data-status="not_converged"]')
        assert failed.count() == 1
        assert "Did not converge." in failed.inner_text()
        assert "✕" in failed.inner_text()
        assert "Stopped because" in failed.inner_text()
        assert all("Did not converge." not in text for text in _others(rows, failed))
        # One line across the fits names it, and its name selects it.
        summary = frame.locator(".diagnose-head").first.inner_text()
        assert "1 of 4 fits did not converge" in summary
        frame.locator('.diagnose-fits [data-diagnose-fit="fold-3"]').click()
        assert frame.locator('[data-diagnose-fit][aria-pressed="true"]').inner_text() == "Fold 3"
        frame.locator(".diagnose-head [data-diagnose-fit]").click()
        assert frame.locator('[data-diagnose-fit][aria-pressed="true"]').inner_text() == "Fold 2"
        assert "max_reml_iter" in frame.locator(".diagnose-explain").inner_text()


def _others(rows, failed) -> list[str]:
    texts = rows.all_inner_texts()
    return [text for text in texts if text != failed.inner_text()]


@pytest.mark.parametrize(
    "viewport", _FRAMES, ids=lambda frame: f"{frame['width']}x{frame['height']}"
)
def test_every_fit_s_verdict_and_the_search_path_are_on_the_first_screen(
    open_lss_editor_page, viewport
):
    with open_lss_editor_page(cv=_cv_with_a_fold_that_did_not_converge(), viewport=viewport) as (
        page,
        _session,
    ):
        page.locator("#diagnoseTab").click()
        frame = page.locator("#reportFrame")
        frame.locator(".diagnose-chart").wait_for()
        seen = page.evaluate(
            """() => ({
                height: window.innerHeight,
                verdicts: [...document.querySelectorAll('.diagnose-fits td.diagnose-result')]
                    .map((cell) => cell.getBoundingClientRect().bottom),
                chart: document.querySelector('.diagnose-chart').getBoundingClientRect().bottom,
                scrolled: document.querySelector('#reportPanel').scrollTop,
            })"""
        )
        assert len(seen["verdicts"]) == 4
        assert max(seen["verdicts"]) <= seen["height"]
        assert seen["chart"] <= seen["height"]
        assert seen["scrolled"] == 0


def test_a_folded_section_stays_as_it_was_left_when_another_fit_opens(open_lss_editor_page):
    with open_lss_editor_page(cv=_cv_with_a_fold_that_did_not_converge()) as (page, _session):
        page.locator("#diagnoseTab").click()
        frame = page.locator("#reportFrame")
        frame.locator(".diagnose-table").first.wait_for()
        fold = frame.locator('[data-diagnose-fold="search"]')
        assert fold.get_attribute("aria-expanded") == "false"
        fold.click()
        assert (
            frame.locator('[data-diagnose-fold="search"]').get_attribute("aria-expanded") == "true"
        )
        frame.locator('.diagnose-fits [data-diagnose-fit="model"]').click()
        frame.locator('.diagnose-fits [data-diagnose-fit="fold-2"]').click()
        assert (
            frame.locator('[data-diagnose-fold="search"]').get_attribute("aria-expanded") == "true"
        )
        frame.locator('[data-diagnose-fold="search"]').click()
        assert (
            frame.locator('[data-diagnose-fold="search"]').get_attribute("aria-expanded") == "false"
        )
