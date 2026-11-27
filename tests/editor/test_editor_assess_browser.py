"""The editor's reports, as the Assess tab will hold them: each keeps its own evidence."""

from __future__ import annotations

from contextlib import contextmanager
from urllib.parse import urlsplit

import pytest
from tests.editor.assess_table import (
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


def test_a_final_fit_that_finishes_shows_on_final_fit(assess_page):
    """Final fit, held from before the job ran, is asked again once the job finishes."""
    with assess_page() as (page, _session):
        open_report(page, "final")
        wait_for_report(page, "final")
        assert "Run Final fit on all rows" in page.locator("#reportFrame .final-fit").inner_text()
        open_report(page, "cv")
        page.locator('#reportFrame [data-cv-start="final_fit"]').click()
        page.locator('#reportFrame .cv-job[data-cv-job="final_fit"][data-status="done"]').wait_for()
        open_report(page, "final")
        page.locator("#reportFrame .final-fit table").wait_for()
        assert "Refitted on" in page.locator("#reportFrame .final-fit").inner_text()
