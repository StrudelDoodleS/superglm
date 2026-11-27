"""The four SuperLSS checks on the Assess overview and in its list: thumbnail, number and flag."""

from __future__ import annotations

import itertools
from contextlib import contextmanager
from urllib.parse import urlsplit

import pytest
from tests.editor._editor_waits import hold_requests, wait_for_editor_idle
from tests.editor.assess_table import bare_glm_session
from tests.test_editor_checks import _session, _wrong_case

pytest.importorskip("playwright.sync_api")
pytestmark = pytest.mark.browser

CHECKS = ("check_qq", "check_pit", "check_ae", "check_calibration")
LABELS = {
    "check_qq": "Q-Q plot",
    "check_pit": "PIT histogram",
    "check_ae": "Actual over expected",
    "check_calibration": "Quantile calibration",
}

# The page's own /report answer for a check, and the face assess_check.js reads from it.
_FACE_FROM_REPORT = """async (report) => {
  const token = new URLSearchParams(location.search).get("token") || "";
  const reply = await fetch("/report", {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-SuperGLM-Editor-Token": token },
    body: JSON.stringify({ report }),
  });
  const { checkFace } = await import("/assets/views/assess_check.js");
  return checkFace(report, await reply.json(), "current");
}"""
# A colour token as the page computes it.
_TOKEN_COLOUR = """(token) => {
  const probe = document.createElement("span");
  probe.style.color = `var(${token})`;
  document.body.append(probe);
  const colour = getComputedStyle(probe).color;
  probe.remove();
  return colour;
}"""


def _wrong_session():
    """Log-normal amounts under a Gamma fit, read on 1,000 validation rows: three checks flag."""
    return _session(_wrong_case())


def _edited_session():
    """The same fit with ``mean:x`` flattened, so each check has the fit beside the edit."""
    session = _wrong_session()
    session.select_indices("mean:x", list(range(session.terms["mean:x"].size)))
    session.weighted_average("mean:x")
    return session


@pytest.fixture
def checks_page(chromium_browser):
    """``with checks_page(build) as page``: the editor of ``build()``'s session, 1278 px wide."""

    @contextmanager
    def open_page(build=_wrong_session):
        widget = build().widget()
        try:
            page = chromium_browser.new_page(viewport={"width": 1278, "height": 860})
            try:
                page.goto(widget.app_url + "&test=1", wait_until="domcontentloaded")
                # A flattened curve draws a line with no height, which has no box to be visible.
                page.locator("#chart path.edited").first.wait_for(state="attached")
                yield page
            finally:
                page.close()
        finally:
            widget.close()

    return open_page


def _open_overview(page) -> None:
    page.locator("#assessTab").click()
    page.locator("#assessOverview").wait_for(state="visible")


def _wait_for_checks(page) -> None:
    for name in CHECKS:
        page.locator(f'#assessCards [data-card="{name}"][data-freshness="current"]').wait_for()


def _report_requests(page) -> list[str]:
    asked: list[str] = []

    def record(request):
        if request.method == "POST" and urlsplit(request.url).path == "/report":
            asked.append(request.post_data_json["report"])

    page.on("request", record)
    return asked


def test_the_overview_draws_a_thumbnail_and_the_payloads_number_for_each_check(checks_page):
    with checks_page() as page:
        _open_overview(page)
        _wait_for_checks(page)
        titles = page.locator("#assessCards .assess-section-title").all_inner_texts()
        assert titles == ["Reports", "Distribution", "Calibration"]
        for name in CHECKS:
            card = page.locator(f'#assessCards [data-card="{name}"]')
            assert card.locator(".assess-card-title").inner_text() == LABELS[name]
            plot = card.locator(".assess-card-plot svg")
            assert plot.locator("path").count() >= 1
            box = plot.bounding_box()
            assert (round(box["width"]), round(box["height"])) == (160, 62)
            face = page.evaluate(_FACE_FROM_REPORT, name)
            assert card.locator(".assess-card-value").inner_text() == face["value"]
            assert card.locator(".assess-card-note").inner_text() == face["note"]


def test_a_flagged_check_is_orange_on_its_card_and_in_the_list(checks_page):
    with checks_page() as page:
        _open_overview(page)
        _wait_for_checks(page)
        orange = page.evaluate(_TOKEN_COLOUR, "--orange")
        orange_text = page.evaluate(_TOKEN_COLOUR, "--orange-text")
        flagged = {name for name in CHECKS if page.evaluate(_FACE_FROM_REPORT, name)["flagged"]}
        # The wrong family leaves the Q-Q, PIT and quantile bands; A/E stays inside.
        assert flagged == {"check_qq", "check_pit", "check_calibration"}
        for name in CHECKS:
            card = page.locator(f'#assessCards [data-card="{name}"]')
            row = page.locator(f'#assessList [data-assess-entry="{name}"]')
            on = name in flagged
            assert card.get_attribute("data-flagged") == str(on).lower()
            assert card.locator(".assess-flag").is_visible() is on
            assert row.locator(".assess-flag").is_visible() is on
            value_colour = card.locator(".assess-card-value").evaluate(
                "node => getComputedStyle(node).color"
            )
            assert (value_colour == orange_text) is on
            assert row.get_attribute("aria-label") == (
                f"{LABELS[name]}, flagged" if on else LABELS[name]
            )
            if on:
                dot = card.locator(".assess-flag").evaluate(
                    "node => getComputedStyle(node).backgroundColor"
                )
                assert dot == orange


def test_a_flagged_check_stays_orange_while_its_earlier_result_is_kept(checks_page, choose_feature):
    """An edit keeps each check's earlier number on its card while it is asked again, and
    after the update fails; a flagged one keeps its flag and its orange through both."""
    with checks_page() as page:
        _open_overview(page)
        _wait_for_checks(page)
        orange_text = page.evaluate(_TOKEN_COLOUR, "--orange-text")
        flagged = ("check_qq", "check_pit", "check_calibration")
        held = hold_requests(page, "**/report")
        page.locator("#editorTab").click()
        choose_feature(page, "mean:x")
        page.locator('button[data-op="select_all"]').click()
        page.get_by_role("button", name="Increase selection").click()
        wait_for_editor_idle(page)
        _open_overview(page)
        # The reports are asked one at a time: let them through up to the first flagged check.
        for asked in itertools.count(1):
            held.wait(asked)
            asking = held[-1].request.post_data_json["report"]
            if asking in flagged:
                break
            held[-1].continue_()

        def assert_kept_and_orange(name: str, shown: str) -> None:
            card = page.locator(f'#assessCards [data-card="{name}"]')
            page.locator(f'#assessCards [data-card="{name}"]{shown}').wait_for()
            assert card.get_attribute("data-flagged") == "true"
            assert card.locator(".assess-flag").is_visible()
            colour = card.locator(".assess-card-value").evaluate(
                "node => getComputedStyle(node).color"
            )
            assert colour == orange_text

        # The check asked again is updating; the two behind it wait, their numbers out of date.
        for name in flagged:
            freshness = "updating" if name == asking else "stale"
            assert_kept_and_orange(name, f'[data-freshness="{freshness}"]')
        page.unroute("**/report")
        page.route("**/report", lambda route: route.abort())
        held[-1].abort()
        for name in flagged:
            assert_kept_and_orange(name, ':has(.assess-card-note:text-matches("^Update failed"))')


def test_a_superglm_session_shows_no_check_groups(checks_page):
    with checks_page(bare_glm_session) as page:
        _open_overview(page)
        page.locator('#assessCards [data-card="diagnose"]').wait_for()
        assert page.locator("#assessCards .assess-section-title").all_inner_texts() == ["Reports"]
        assert page.locator('#assessCards [data-card^="check_"]').count() == 0
        for name in CHECKS:
            assert page.locator(f'#assessList [data-assess-entry="{name}"]').is_hidden()
        for group in ("distribution", "calibration"):
            assert page.locator(f'#assessList [data-assess-group="{group}"]').is_hidden()
        assert page.locator("#overviewCompare").is_hidden()
        # The keyboard steps over the hidden checks: End is the last report.
        page.locator('[data-assess-entry="assess"]').focus()
        page.keyboard.press("End")
        assert page.evaluate("document.activeElement?.dataset.assessEntry") == "diagnose"


def test_the_report_cards_keep_their_place_and_focus_while_the_checks_land(checks_page):
    with checks_page() as page:
        held = hold_requests(page, "**/report")
        _open_overview(page)
        held.wait(1)
        card = page.locator('#assessCards [data-card="validation"]')
        card.evaluate("node => { node.focus({focusVisible: true}); node.dataset.probe = 'held'; }")
        page.unroute("**/report")
        for route in held:
            route.continue_()
        _wait_for_checks(page)
        assert page.evaluate("document.activeElement?.dataset.card") == "validation"
        assert page.evaluate("document.activeElement?.dataset.probe") == "held"
        cards = page.locator("#assessCards [data-card]").evaluate_all(
            "nodes => nodes.map((node) => node.dataset.card)"
        )
        assert cards == ["validation", "cv", "final", "diagnose", *CHECKS]


def test_compare_is_off_and_says_why_until_there_is_an_edit(checks_page):
    with checks_page() as page:
        _open_overview(page)
        _wait_for_checks(page)
        switch = page.locator("#overviewCompare")
        assert switch.get_attribute("role") == "switch"
        assert switch.inner_text().strip() == "Compare with the fit"
        assert switch.get_attribute("aria-disabled") == "true"
        assert switch.get_attribute("aria-checked") == "false"
        # Playwright will not click an element marked disabled; a click still reaches it.
        switch.dispatch_event("click")
        assert switch.get_attribute("aria-checked") == "false"
        switch.hover()
        popover = page.locator("#uiPopover")
        popover.wait_for(state="visible")
        assert (
            popover.locator("[data-popover-description]").inner_text()
            == "The model is as fitted; there is no edit to compare."
        )


def test_compare_adds_the_fit_to_each_thumbnail_without_asking_python(checks_page):
    with checks_page(_edited_session) as page:
        _open_overview(page)
        _wait_for_checks(page)
        switch = page.locator("#overviewCompare")
        assert switch.get_attribute("aria-disabled") == "false"
        assert page.locator("#assessCards .assess-card-plot path.check-original").count() == 0
        asked = _report_requests(page)
        switch.click()
        assert switch.get_attribute("aria-checked") == "true"
        for name in CHECKS:
            plot = page.locator(f'#assessCards [data-card="{name}"] .assess-card-plot')
            assert plot.locator("path.check-original").count() == 1
        # The fit is a grey ring painted under the edit, its stroke wider than the edit's.
        widths = page.locator('#assessCards [data-card="check_qq"] .assess-card-plot').evaluate(
            """plot => ["check-original", "check-edited"].map((role) =>
                parseFloat(getComputedStyle(plot.querySelector(`path.${role}`)).strokeWidth))"""
        )
        assert widths[0] > widths[1]
        switch.click()
        assert page.locator("#assessCards .assess-card-plot path.check-original").count() == 0
        assert asked == []
