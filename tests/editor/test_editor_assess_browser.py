"""The editor's Assess tab: its list, its overview, and the reports it holds, each with its own evidence."""

from __future__ import annotations

from contextlib import contextmanager
from urllib.parse import urlsplit

import pytest
from tests.editor._editor_waits import hold_requests, wait_for_editor_idle
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


def test_a_card_click_moves_focus_to_the_list_without_a_popover(assess_page):
    """Focus a click moved shows no tooltip; keyboard focus on an entry still shows its own."""
    with assess_page() as (page, _session):
        popover = page.locator("#uiPopover")
        page.locator("#assessTab").click()
        page.locator('#assessCards [data-card="cv"]').click()
        wait_for_report(page, "cv")
        assert page.evaluate("document.activeElement?.dataset.assessEntry") == "cv"
        assert popover.is_hidden()
        page.keyboard.press("ArrowDown")
        assert page.evaluate("document.activeElement?.dataset.assessEntry") == "final"
        popover.wait_for(state="visible")
        assert popover.locator("[data-popover-heading]").inner_text() == "Final fit"


def test_each_card_shows_the_number_its_report_shows(assess_page):
    with assess_page() as (page, _session):
        page.locator("#assessTab").click()
        page.locator('#assessCards [data-card="diagnose"][data-freshness="current"]').wait_for()
        cards = {
            c.get_attribute("data-card"): c.locator(".assess-card-value").inner_text()
            for c in page.locator("#assessCards [data-card]").all()
        }
        open_report(page, "validation")
        wait_for_report(page, "validation")
        row = page.get_by_role("table", name="Split metrics").locator("tr", has_text="Validation")
        assert (
            cards["validation"]
            == "Deviance " + row.locator("td").nth(1).inner_text().split(" Δ")[0].strip()
        )
        open_report(page, "cv")
        wait_for_report(page, "cv")
        first = page.locator("#reportFrame .cv-card").first
        assert cards["cv"].startswith(
            "Mean deviance " + first.locator(".cv-card-value strong").first.inner_text()
        )
        assert cards["final"] == "Not run"
        assert cards["diagnose"].endswith("fits converged") or cards["diagnose"].endswith(
            "did not converge"
        )


def test_a_card_keeps_keyboard_focus_when_a_report_lands(assess_page):
    """The cards are drawn again as each report lands; the card that had focus keeps it."""
    with assess_page() as (page, _session):
        held = hold_requests(page, "**/report")
        page.locator("#assessTab").click()
        held.wait(1)
        page.locator('#assessCards [data-card="cv"]').evaluate(
            "card => card.focus({focusVisible: true})"
        )
        held[0].continue_()
        page.locator('#assessCards [data-card="validation"][data-freshness="current"]').wait_for()
        assert page.evaluate("document.activeElement?.dataset.card") == "cv"
        assert page.evaluate("document.activeElement.matches(':focus-visible')")
        for count in (2, 3):
            held.wait(count)
            held[count - 1].continue_()
        page.locator('#assessCards [data-card="diagnose"][data-freshness="current"]').wait_for()
        assert page.evaluate("document.activeElement?.dataset.card") == "cv"


def test_a_card_pressed_while_a_report_lands_still_opens_its_report(assess_page):
    """A report landing between pointer-down and pointer-up leaves the pressed card in place."""
    with assess_page() as (page, _session):
        held = hold_requests(page, "**/report")
        page.locator("#assessTab").click()
        held.wait(1)
        page.locator('#assessCards [data-card="final"]').hover()
        page.mouse.down()
        held[0].continue_()
        page.locator('#assessCards [data-card="validation"][data-freshness="current"]').wait_for()
        page.mouse.up()
        page.locator('#reportPanel[data-report="final"]').wait_for(state="visible", timeout=5000)
        page.unroute("**/report")
        for route in held[1:]:
            route.continue_()
        wait_for_report(page, "final")


def test_the_list_keeps_its_names_at_1180_px(assess_page):
    with assess_page(viewport=(1180, 720)) as (page, _session):
        page.locator("#assessTab").click()
        label = page.locator('[data-assess-entry="cv"] .assess-row-label')
        assert label.is_visible() and label.bounding_box()["width"] > 40
        assert page.evaluate(
            "document.querySelector('#assessView').scrollWidth <= document.querySelector('#assessView').clientWidth"
        )


def test_below_1086_px_the_list_is_icons_with_names_in_popovers(assess_page):
    with assess_page(viewport=(1000, 720)) as (page, _session):
        page.locator("#assessTab").click()
        nav = page.get_by_role("navigation", name="Assess")
        assert nav.bounding_box()["width"] <= 40
        assert nav.get_by_role("button", name="Smoothing search").is_visible()
        nav.get_by_role("button", name="Smoothing search").hover()
        page.locator("#uiPopover [data-popover-heading]", has_text="Smoothing search").wait_for()
        assert page.evaluate(
            "document.querySelector('#assessView').scrollWidth <= document.querySelector('#assessView').clientWidth"
        )
        open_report(page, "cv")
        wait_for_report(page, "cv")
        assert page.evaluate(
            "document.documentElement.scrollWidth <= document.documentElement.clientWidth"
        )


def _is_state_read(request) -> bool:
    return urlsplit(request.url).path == "/state"


def test_open_in_new_window_shows_the_view_and_follows_edits(assess_page, choose_feature):
    """A second window opens on the report shown and follows an edit made in the first.

    The second window's hello, once it has loaded, makes the first re-read once. Moving
    between entries tells the second window nothing. After an edit the second window
    re-reads the session, and the first, already agreeing, re-reads nothing: the
    windows agree after one round.
    """
    with assess_page() as (page, _session):
        open_report(page, "validation")
        wait_for_report(page, "validation")
        with page.expect_request(_is_state_read), page.expect_popup() as opened:
            page.locator("#reportPanel [data-assess-window]").click()
        popup = opened.value
        assert "assess=validation" in popup.url and "token=" in popup.url
        wait_for_report(popup, "validation")
        assert_report_open(popup, "validation")
        assert popup.title() == "Assess · SuperGLM Editor"
        assert popup.locator("#editorTab").is_hidden() and popup.locator("#undoAction").is_hidden()
        assert popup.locator("[data-assess-window]:visible").count() == 0
        # The only tab is Assess, so an arrow key stays on the report open.
        popup.locator("#assessTab").focus()
        popup.keyboard.press("ArrowLeft")
        assert_report_open(popup, "validation")
        reads, main_reads = [], []
        popup.on("request", lambda request: _is_state_read(request) and reads.append(request))
        page.on("request", lambda request: _is_state_read(request) and main_reads.append(request))
        for name in ("cv", "validation", "cv"):  # switching entries tells nobody anything
            open_report(page, name)
            wait_for_report(page, name)
        assert reads == []
        page.locator("#editorTab").click()
        choose_feature(page, "age")
        page.locator('button[data-op="select_all"]').click()
        page.get_by_role("button", name="Increase selection").click()
        wait_for_editor_idle(page)
        popup.get_by_role("table", name="Split metrics").get_by_text("Δ").first.wait_for()
        wait_for_report(popup, "validation")
        assert 1 <= len(reads) <= 2
        assert main_reads == []


def _with_a_running_cv_job(route) -> None:
    """Answer the cv report as Python would while a Run CV runs."""
    if route.request.post_data_json.get("report") != "cv":
        route.fallback()
        return
    response = route.fetch()
    payload = response.json()
    running = {"job_id": "cv-held", "kind": "cv", "status": "running", "progress": []}

    def mark(node) -> bool:
        if isinstance(node, dict):
            if isinstance(node.get("jobs"), dict) and "cv" in node["jobs"]:
                node["jobs"]["cv"] = running
                return True
            return any(mark(value) for value in node.values())
        return isinstance(node, list) and any(mark(value) for value in node)

    assert mark(payload)
    route.fulfill(response=response, json=payload)


def test_the_second_window_follows_a_job_and_starts_or_cancels_none(assess_page):
    """Jobs start in the main window only; the second shows their progress without controls.

    A click that reaches the hidden controls, as a script's would, sends nothing either.
    """
    with assess_page() as (page, _session):
        open_report(page, "validation")
        wait_for_report(page, "validation")
        with page.expect_popup() as opened:
            page.locator("#reportPanel [data-assess-window]").click()
        popup = opened.value
        wait_for_report(popup, "validation")
        hold_requests(popup, "**/job_status")
        popup.route("**/report", _with_a_running_cv_job)
        sent = []
        popup.on(
            "request",
            lambda request: (
                urlsplit(request.url).path in ("/job_start", "/job_cancel")
                and sent.append(urlsplit(request.url).path)
            ),
        )
        open_report(popup, "cv")
        frame = popup.locator("#reportFrame")
        frame.locator('.cv-job[data-cv-job="cv"][data-status="running"]').wait_for()
        assert frame.locator("[data-cv-start], [data-cv-cancel]").count() == 3
        for name in ("Run CV on current model", "Final fit on all rows", "Cancel"):
            assert frame.get_by_role("button", name=name).count() == 0
        frame.locator('[data-cv-start="final_fit"]').dispatch_event("click")
        frame.locator('[data-cv-cancel="cv"]').dispatch_event("click")
        # A request after the clicks, answered by the interception probe's route.
        popup.evaluate("() => fetch('/__intercepting__').then((reply) => reply.status)")
        assert sent == []
        # The held polls' route callbacks must not outlive the page.
        popup.unroute_all(behavior="ignoreErrors")


def test_the_second_window_hides_the_editor_before_its_script_runs(assess_page, chromium_browser):
    with assess_page() as (page, _session):
        early = chromium_browser.new_page()
        held = []
        early.route("**/assets/main.js", lambda route: held.append(route))
        try:
            early.goto(page.url + "&assess=cv", wait_until="commit")
            early.wait_for_function(
                """() => document.getElementById("editorView")
                  && document.querySelector('link[href="/assets/styles/assess.css"]')?.sheet"""
            )
            assert early.title() == "SuperGLM Editor"  # main.js has not run
            assert early.locator("#editorTab").is_hidden()
            assert early.locator("#editorView").is_hidden()
        finally:
            early.close()
