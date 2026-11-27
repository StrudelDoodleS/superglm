"""The four SuperLSS checks on Assess: thumbnail, number and flag, and each check drawn large."""

from __future__ import annotations

import itertools
from contextlib import contextmanager
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
import pytest
from tests.editor._editor_waits import hold_requests, wait_for_editor_idle
from tests.editor.assess_table import bare_glm_session, open_report, wait_for_report
from tests.test_editor_checks import GROUPS, LAMBDA, SEED, _calm_case, _session, _wrong_case

from superglm import GammaLS, SuperLSS, cat, s
from superglm.editor import EditorSession

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


# The check open large, drawn again in the page from its own /report answer at the frame's
# size: the points the readout may name, as the view's scene places them.
_MARKS_FROM_REPORT = """async ([report, compare]) => {
  const token = new URLSearchParams(location.search).get("token") || "";
  const reply = await fetch("/report", {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-SuperGLM-Editor-Token": token },
    body: JSON.stringify({ report }),
  });
  const { checkScene } = await import("/assets/chart/check_plots.js");
  const frame = document.getElementById("checkFrame");
  const box = { width: frame.clientWidth, height: frame.clientHeight };
  return checkScene(report, await reply.json(), box, { compare }).marks;
}"""
# Record each mutation under #checkFrame: whether its target sits inside the readout.
_WATCH_FRAME = """() => {
  window.__frameMutations = [];
  new MutationObserver((records) => {
    for (const record of records) {
      const node = record.target.nodeType === Node.ELEMENT_NODE ? record.target : record.target.parentElement;
      window.__frameMutations.push(Boolean(node?.closest(".check-readout")));
    }
  }).observe(document.getElementById("checkFrame"),
    { subtree: true, attributes: true, childList: true, characterData: true });
}"""
_NO_EDIT = "The model is as fitted; there is no edit to compare."


def _calm_session():
    """A Gamma fit of Gamma amounts, read on 200 validation rows: no check flags."""
    return _session(_calm_case())


def _many_inputs_session():
    """A Gamma fit on five inputs, so Actual over expected draws six panels in three rows."""
    rng = np.random.default_rng(SEED)
    n = 800
    X = pd.DataFrame(
        {
            "x": rng.uniform(0.0, 1.0, n),
            "g": rng.choice(GROUPS, n),
            "u": rng.uniform(0.0, 1.0, n),
            "w": rng.uniform(0.0, 1.0, n),
            "h": rng.choice(np.array(["p", "q"]), n),
        }
    )
    mean = np.exp(1.0 + 0.5 * np.sin(2.0 * np.pi * X["x"].to_numpy()))
    y = rng.gamma(4.0, mean / 4.0)
    family = GammaLS()
    spec = SuperLSS(
        family,
        family.mean(s("x", k=6), cat("g"), s("u", k=4), s("w", k=4), cat("h")),
        family.scale(cat("g")),
    )
    smooths = {f"mean:{name}#wiggle": LAMBDA for name in ("x", "u", "w")}
    lss = spec.fit(X.iloc[:600], y[:600], lambdas=smooths)
    return EditorSession.from_model(
        lss,
        train_data=(X.iloc[:600], y[:600]),
        validation_data=(X.iloc[600:].reset_index(drop=True), y[600:]),
    )


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


def _open_check(page, name: str) -> None:
    """Open check ``name`` from the Assess list and wait until it is drawn current."""
    if page.locator("#assessView").is_hidden():
        page.locator("#assessTab").click()
    page.locator(f'#assessList [data-assess-entry="{name}"]').click()
    page.locator(f'#checkPanel[data-report="{name}"]').wait_for(state="visible")
    page.locator('#checkFrame[data-freshness="current"] svg.check-plot').wait_for(state="attached")


def _average_mean_x(page, choose_feature) -> None:
    """Average every value of ``mean:x``, as an analyst does through the selection menu."""
    page.locator("#editorTab").click()
    choose_feature(page, "mean:x")
    page.locator('button[data-op="select_all"]').click()
    page.get_by_role("button", name="Level selected values").hover()
    page.locator('button[data-op="average"]').click()
    wait_for_editor_idle(page)


def _nearest(marks: list[dict], x: float, y: float) -> dict:
    return min(marks, key=lambda mark: (mark["x"] - x) ** 2 + (mark["y"] - y) ** 2)


def _readout(page):
    return page.locator("#checkFrame .check-readout")


def test_opening_a_check_draws_its_title_split_and_points(checks_page):
    with checks_page(_calm_session) as page:
        _open_check(page, "check_qq")
        assert page.locator("#reportPanel").is_hidden()
        assert page.locator("#checkTitle").inner_text() == "Q-Q plot"
        assert page.locator("#checkSplit").inner_text() == "Validation, 200 rows"
        assert page.locator('[data-assess-entry="check_qq"][aria-current="true"]').count() == 1
        frame = page.locator("#checkFrame")
        assert frame.get_attribute("role") == "img"
        assert frame.get_attribute("tabindex") == "0"
        assert frame.get_attribute("aria-label") == "Q-Q plot"
        points = page.locator("#checkFrame path:is(.check-edited, .check-outside)").evaluate_all(
            "paths => paths.reduce((sum, p) => sum + (p.getAttribute('d').match(/M/g) || []).length, 0)"
        )
        payload_points = page.evaluate(
            """async () => {
              const token = new URLSearchParams(location.search).get("token") || "";
              const reply = await fetch("/report", { method: "POST",
                headers: { "Content-Type": "application/json", "X-SuperGLM-Editor-Token": token },
                body: JSON.stringify({ report: "check_qq" }) });
              return (await reply.json()).models.edited.index.length;
            }"""
        )
        assert points == payload_points == 200
        # The plot fills the frame's width, and its height when it is not taller.
        size = frame.evaluate(
            """frame => { const svg = frame.querySelector("svg");
              return [svg.getAttribute("width"), svg.getAttribute("height"),
                frame.clientWidth, frame.clientHeight].map(Number); }"""
        )
        assert size[:2] == size[2:]
        assert page.locator("#checkLegend").is_hidden()
        assert page.locator("#checkCompare").get_attribute("aria-disabled") == "true"


def test_compare_draws_the_fit_without_asking_python(checks_page, choose_feature):
    with checks_page() as page:
        _average_mean_x(page, choose_feature)
        _open_check(page, "check_ae")
        switch = page.locator("#checkCompare")
        page.locator('#checkCompare[aria-disabled="false"]').wait_for()
        assert page.locator("#checkFrame path.check-original").count() == 0
        asked = _report_requests(page)
        switch.click()
        assert switch.get_attribute("aria-checked") == "true"
        assert page.locator("#checkFrame path.check-original").count() == 1
        assert page.locator("#checkLegend").is_visible()
        assert page.locator("#checkLegend").inner_text().split() == ["As", "edited", "As", "fitted"]
        # The fit's intervals are as thin as the edit's; only its rings are wider.
        widths = page.locator("#checkFrame svg.check-plot").evaluate(
            """svg => Object.fromEntries(["original", "original-interval", "interval"].map((role) =>
                [role, parseFloat(getComputedStyle(svg.querySelector(`path.check-${role}`)).strokeWidth)]))"""
        )
        assert widths["original-interval"] == widths["interval"] < widths["original"]
        # One flag: the overview's switch reads it too.
        page.locator('[data-assess-entry="assess"]').click()
        assert page.locator("#overviewCompare").get_attribute("aria-checked") == "true"
        _open_check(page, "check_ae")
        assert page.locator("#checkFrame path.check-original").count() == 1
        switch.click()
        assert page.locator("#checkFrame path.check-original").count() == 0
        assert page.locator("#checkLegend").is_hidden()
        assert asked == []


def test_compare_is_off_and_disabled_after_undo(checks_page, choose_feature):
    """Undo turns Compare off at once, while the check is still asked about the fit, and a
    failed answer keeps it off: the payload kept from the edit draws no fit."""
    with checks_page() as page:
        _average_mean_x(page, choose_feature)
        _open_check(page, "check_ae")
        switch = page.locator("#checkCompare")
        page.locator('#checkCompare[aria-disabled="false"]').wait_for()
        switch.click()
        page.locator("#checkFrame path.check-original").wait_for(state="attached")
        # Live, its popover is its own Help entry, without the checks' shared ending.
        switch.hover()
        popover = page.locator("#uiPopover")
        popover.wait_for(state="visible")
        assert popover.locator("[data-popover-description]").inner_text() == (
            "Draws the model as fitted in grey beside the model as edited in blue, on every check. "
            "It changes only what is drawn."
        )
        page.mouse.move(2, 2)
        held = hold_requests(page, "**/report")
        page.keyboard.press("Control+z")
        # The check is asked again once Undo has committed; its answer is held.
        held.wait(1)
        assert held[-1].request.post_data_json["report"] == "check_ae"

        def assert_off() -> None:
            assert switch.get_attribute("aria-disabled") == "true"
            assert switch.get_attribute("aria-checked") == "false"
            assert (
                page.locator(
                    "#checkFrame path:is(.check-original, .check-original-interval)"
                ).count()
                == 0
            )
            assert page.locator("#checkLegend").is_hidden()

        page.locator('#checkFrame[data-freshness="updating"]').wait_for()
        assert_off()
        # Off, its popover says why, over its Help entry.
        switch.hover()
        page.wait_for_function(
            "text => document.querySelector('#uiPopover [data-popover-description]')?.textContent === text",
            arg=_NO_EDIT,
        )
        assert popover.is_visible()
        page.mouse.move(2, 2)
        # The answer fails: the edit's payload stays on screen, and Compare stays off.
        page.unroute("**/report")
        page.route("**/report", lambda route: route.abort())
        held[-1].abort()
        page.locator('#checkFrame[data-freshness="stale"]').wait_for()
        assert page.locator("#checkRetry").is_visible()
        assert_off()


def test_hovering_a_check_changes_nothing_outside_its_readout(checks_page):
    with checks_page() as page:
        _open_check(page, "check_qq")
        marks = page.evaluate(_MARKS_FROM_REPORT, ["check_qq", False])
        box = page.locator("#checkFrame svg.check-plot").bounding_box()
        page.evaluate(_WATCH_FRAME)
        moves = [
            (box["width"] * (0.1 + 0.8 * i / 39), box["height"] * (0.9 - 0.8 * i / 39))
            for i in range(40)
        ]
        for x, y in moves:
            page.mouse.move(box["x"] + x, box["y"] + y)
        last = _nearest(marks, *moves[-1])
        page.wait_for_function(
            "text => document.querySelector('#checkFrame .check-readout text')?.textContent === text",
            arg=last["text"],
        )
        mutations = page.evaluate("window.__frameMutations")
        assert mutations and all(mutations)
        assert _readout(page).is_visible()
        assert page.locator("#uiPopover").is_hidden()
        page.mouse.move(2, 2)
        page.locator("#checkFrame .check-readout[hidden]").wait_for(state="attached")


def test_a_tall_check_scrolls_and_hover_reads_through_the_scroll(chromium_browser):
    """At 640 px high (above 620 px the page keeps its height), six A/E panels need 556 px
    against a 505 px frame, which scrolls them."""
    widget = _many_inputs_session().widget()
    try:
        page = chromium_browser.new_page(viewport={"width": 1278, "height": 640})
        try:
            page.goto(widget.app_url + "&test=1", wait_until="domcontentloaded")
            page.locator("#chart path.edited").first.wait_for(state="attached")
            _open_check(page, "check_ae")
            frame = page.locator("#checkFrame")
            assert frame.evaluate("frame => frame.scrollHeight > frame.clientHeight")
            marks = page.evaluate(_MARKS_FROM_REPORT, ["check_ae", False])
            frame.evaluate("frame => { frame.scrollTop = frame.scrollHeight; }")
            assert frame.evaluate("frame => frame.scrollTop") > 0
            lowest = max(marks, key=lambda mark: mark["y"])
            box = page.locator("#checkFrame svg.check-plot").bounding_box()
            page.mouse.move(box["x"] + lowest["x"], box["y"] + lowest["y"])
            page.wait_for_function(
                "text => document.querySelector('#checkFrame .check-readout text')?.textContent === text",
                arg=_nearest(marks, lowest["x"], lowest["y"])["text"],
            )
        finally:
            page.close()
    finally:
        widget.close()


def test_a_resize_redraws_the_check_without_a_request(checks_page):
    with checks_page() as page:
        _open_check(page, "check_pit")
        svg_width = "() => Number(document.querySelector('#checkFrame svg.check-plot').getAttribute('width'))"
        before = page.evaluate(svg_width)
        asked = _report_requests(page)
        page.set_viewport_size({"width": 1000, "height": 860})
        page.wait_for_function(f"({svg_width})() < {before}")
        after = page.evaluate(svg_width)
        assert after == page.locator("#checkFrame").evaluate("frame => frame.clientWidth")
        assert asked == []


@pytest.mark.pins_behaviour
def test_validation_still_opens_in_the_report_panel(checks_page):
    """Pins that a report view keeps its own panel and renderReport: opening Validation shows the
    report panel with its report, and the check panel stays hidden until a check is opened."""
    with checks_page() as page:
        page.locator("#assessTab").click()
        open_report(page, "validation")
        wait_for_report(page, "validation")
        assert page.locator("#checkPanel").is_hidden()
        assert page.locator("#reportTitle").inner_text() == "Validation"
        assert page.locator("#reportFrame").inner_text().strip() != ""
        _open_check(page, "check_pit")
        assert page.locator("#reportPanel").is_hidden()
        open_report(page, "validation")
        wait_for_report(page, "validation")
        assert page.locator("#checkPanel").is_hidden()
