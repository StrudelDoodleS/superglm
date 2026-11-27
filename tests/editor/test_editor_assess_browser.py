"""The editor's Assess tab: its list, its overview, and the reports it holds, each with its own evidence."""

from __future__ import annotations

from contextlib import contextmanager
from urllib.parse import urlsplit

import pytest
from tests.editor.assess_table import (
    assert_report_open,
    bare_glm_session,
    binomial_cv_session,
    open_report,
    pareto_lss_session,
    poisson_cv_session,
    wait_for_report,
)

pytest.importorskip("playwright.sync_api")
pytestmark = pytest.mark.browser

_SESSIONS = {
    "poisson": poisson_cv_session,
    "binomial": binomial_cv_session,
    "pareto": pareto_lss_session,
    "bare": bare_glm_session,
}


@pytest.fixture
def assess_page(chromium_browser):
    """Open a book's editor page: ``with assess_page(case, viewport) as (page, session)``."""

    @contextmanager
    def open_page(case: str = "poisson", viewport: tuple[int, int] = (1278, 860)):
        session = _SESSIONS[case]()
        widget = session.widget()
        try:
            width, height = viewport
            page = chromium_browser.new_page(viewport={"width": width, "height": height})
            try:
                page.goto(widget.app_url + "&test=1", wait_until="domcontentloaded")
                page.locator("#chart path.edited").first.wait_for()
                yield page, session
            finally:
                page.close()
        finally:
            widget.close()

    return open_page


def test_a_report_already_drawn_is_drawn_again_from_the_page(assess_page):
    """Going back to a report the page holds for this model asks Python for nothing."""
    with assess_page() as (page, _session):
        asked = []

        def record(request):
            if request.method == "POST" and urlsplit(request.url).path == "/report":
                asked.append(request.post_data_json["report"])

        page.on("request", record)
        for name in ("validation", "cv", "validation", "cv", "diagnose", "cv"):
            open_report(page, name)
            wait_for_report(page, name)
        assert sorted(asked) == ["cv", "diagnose", "validation"]


def _is_report_request(request, name: str) -> bool:
    return (
        request.method == "POST"
        and urlsplit(request.url).path == "/report"
        and request.post_data_json["report"] == name
    )


def test_a_final_fit_that_finishes_shows_on_final_fit(assess_page):
    """Final fit, held from before the job ran, is asked again once the job finishes.

    The open Cross-validation report is asked once after the job: an answer to a request
    sent after the job settled is kept, not dropped and asked for again.
    """
    with assess_page() as (page, _session):
        open_report(page, "final")
        wait_for_report(page, "final")
        assert "Run Final fit on all rows" in page.locator("#reportFrame .final-fit").inner_text()
        open_report(page, "cv")
        wait_for_report(page, "cv")
        asked = []
        page.on("request", lambda request: _is_report_request(request, "cv") and asked.append(1))
        # The job's end asks for the open report at least once.
        with page.expect_request(lambda request: _is_report_request(request, "cv")):
            page.locator('#reportFrame [data-cv-start="final_fit"]').click()
            page.locator(
                '#reportFrame .cv-job[data-cv-job="final_fit"][data-status="done"]'
            ).wait_for()
        # Current only once an answer is kept; a dropped one is asked again before that.
        page.locator('#reportFrame[aria-busy="false"][data-freshness="current"]').wait_for(
            state="attached"
        )
        assert len(asked) == 1
        open_report(page, "final")
        page.locator("#reportFrame .final-fit table").wait_for()
        assert "Refitted on" in page.locator("#reportFrame .final-fit").inner_text()


def test_the_top_bar_is_editor_and_assess_and_assess_opens_on_its_overview(assess_page):
    with assess_page() as (page, _session):
        tabs = page.get_by_role("tablist", name="Editor views").get_by_role("tab")
        assert tabs.all_inner_texts() == ["Editor", "Assess"]
        page.locator("#assessTab").click()
        assert page.locator("#assessOverviewTitle").inner_text() == "Is this model any good?"
        assert page.locator("#assessList [data-assess-entry]").all_inner_texts() == [
            "Overview",
            "Validation",
            "Cross-validation",
            "Final fit",
            "Smoothing search",
        ]
        assert page.locator("#assessList .assess-group-title").all_inner_texts() == ["Reports"]
        assert page.locator("#assessCards .assess-card-title").all_inner_texts() == [
            "Validation",
            "Cross-validation",
            "Final fit",
            "Smoothing search",
        ]
        page.locator('#assessCards [data-card="diagnose"]').click()
        wait_for_report(page, "diagnose")
        assert page.locator("#reportTitle").inner_text() == "Smoothing search"
        assert_report_open(page, "diagnose")
        assert page.evaluate("document.activeElement?.dataset.assessEntry") == "diagnose"
        page.locator("#editorTab").click()
        page.locator("#assessTab").click()
        assert page.locator("#assessOverview").is_visible()


def test_a_superlss_session_has_explain_after_assess(assess_page):
    with assess_page(case="pareto") as (page, _session):
        tabs = page.get_by_role("tablist", name="Editor views").get_by_role("tab")
        assert tabs.all_inner_texts() == ["Editor", "Assess", "Explain"]


def test_the_list_steps_with_the_keyboard(assess_page):
    with assess_page() as (page, _session):
        page.locator("#assessTab").click()
        page.locator('[data-assess-entry="assess"]').focus()
        page.keyboard.press("ArrowDown")
        page.keyboard.press("End")
        page.keyboard.press("Enter")
        wait_for_report(page, "diagnose")
        assert page.locator('#assessList [tabindex="0"]').count() == 1
