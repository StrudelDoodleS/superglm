"""A SuperLSS model, driven from the browser."""

from __future__ import annotations

import io
from urllib.parse import urlsplit

import joblib
import pytest
from tests.editor._chart_layout import wait_for_chart_to_fit

from superglm import SuperLSS
from superglm.editor import widget as widget_module

pytest.importorskip("playwright.sync_api")
pytestmark = pytest.mark.browser

_LOG_NORMAL_TERMS = ["mean:age", "mean:region", "mean:power", "scale:age", "scale:region"]


def _posted(path: str):
    return lambda response: (
        response.request.method == "POST" and urlsplit(response.url).path == path
    )


def test_feature_list_groups_by_parameter(open_lss_editor_page):
    with open_lss_editor_page() as (page, session):
        assert list(session.terms) == _LOG_NORMAL_TERMS
        headers = page.locator("#featureList").get_by_text("Average claim", exact=True)
        assert headers.count() == 1
        assert page.locator("#featureList").get_by_text("Spread of claim sizes").count() == 1

        rows = page.locator("#featureList [data-term]")
        assert rows.evaluate_all("els => els.map(e => e.dataset.term)") == _LOG_NORMAL_TERMS
        assert rows.count() == len(_LOG_NORMAL_TERMS)
        # Every row carries its own mini curve.
        assert page.locator("#featureList [data-term] svg.mini-curve").count() == rows.count()
        for term in _LOG_NORMAL_TERMS:
            assert page.locator(f'#featureList [data-term="{term}"] svg.mini-curve').count() == 1


def test_spread_curve_axis_and_drag(open_lss_editor_page):
    with open_lss_editor_page(selected_term="scale:age") as (page, session):
        axis_titles = page.locator("#chart text.label").evaluate_all(
            "els => els.map(e => e.textContent)"
        )
        assert "coefficient of variation" in axis_titles

        term = session.terms["scale:age"]
        before = term.edited_log_effect.copy()
        page.get_by_role("radiogroup", name="Chart tools").get_by_role(
            "radio", name="Handles", exact=True
        ).click()
        wait_for_chart_to_fit(page)
        handle = page.locator("#chart .control-handle").first
        handle.wait_for()
        box = handle.bounding_box()
        assert box is not None
        x = box["x"] + box["width"] / 2
        y = box["y"] + box["height"] / 2
        with page.expect_response(_posted("/control")) as control:
            page.mouse.move(x, y)
            page.mouse.down()
            page.mouse.move(x, y - 40, steps=5)
            page.mouse.up()
        assert control.value.status == 200
        page.wait_for_function(
            "() => window.__superglmTest.mutationStatus() !== 'running'",
        )

        assert (session.terms["scale:age"].edited_log_effect != before).any()
        # Only the dragged curve moved.
        assert (
            session.terms["mean:age"].edited_log_effect
            == session.terms["mean:age"].original_log_effect
        ).all()

        page.locator("#historyTab").click()
        first_row = page.locator("#historyFrame .history-list li").first
        first_row.wait_for()
        chip = first_row.locator(".parameter-chip")
        assert chip.inner_text() == "Spread"
        assert chip.get_attribute("data-role") == "spread"


def test_cv_tab_hidden_and_final_fit_offered(open_lss_editor_page):
    with open_lss_editor_page() as (page, session):
        assert page.locator("#cvTab").is_hidden()
        assert page.locator("#validationTab").is_visible()
        assert page.locator("#finalTab").is_visible()

        # Arrow keys step Validation -> Final Fit, past the tab that is not offered.
        page.locator("#validationTab").focus()
        page.keyboard.press("ArrowRight")
        assert page.evaluate("document.activeElement?.id") == "finalTab"
        page.keyboard.press("ArrowRight")
        assert page.evaluate("document.activeElement?.id") == "editorTab"
        page.keyboard.press("ArrowLeft")
        page.keyboard.press("ArrowLeft")
        assert page.evaluate("document.activeElement?.id") == "validationTab"

        page.locator("#finalTab").click()
        run = page.locator("#finalFitRun")
        run.wait_for(state="visible")
        # The model was opened with its rows, so the plan offers the fit.
        assert widget_module.final_fit_reason(session) is None
        assert run.is_enabled()
        assert page.locator("#finalFitRunNote").is_hidden()


def test_final_fit_button_follows_the_plan_reason(open_lss_editor_page, monkeypatch):
    reason = "Final fit needs a validation split. Give the editor one, then open it again."
    monkeypatch.setattr(widget_module, "final_fit_reason", lambda session: reason)
    with open_lss_editor_page() as (page, _session):
        page.locator("#finalTab").click()
        run = page.locator("#finalFitRun")
        run.wait_for(state="visible")
        assert run.is_disabled()
        assert run.get_attribute("data-popover-body") == reason


def test_joblib_export_keeps_history(open_lss_editor_page):
    with open_lss_editor_page(selected_term="scale:age") as (page, session):
        session.select_indices("scale:age", [20, 21, 22, 23])
        session.shift("scale:age", 0.3)
        page.get_by_role("button", name="Export model or workbook").click()
        dialog = page.get_by_role("dialog", name="Export")
        dialog.wait_for(state="visible")
        dialog.get_by_role("radio", name="Python model").check()
        # The page hands the blob to the browser's save dialog, so read the
        # bytes the server answers with.
        served: list[bytes] = []

        def capture(route) -> None:
            response = route.fetch()
            served.append(response.body())
            route.fulfill(response=response)

        page.route("**/download_export*", capture)
        with page.expect_response(
            lambda response: (
                urlsplit(response.url).path.endswith("/download_export")
                and "format=joblib" in response.url
            )
        ) as download:
            dialog.get_by_role("button", name="Download", exact=True).click()
        assert download.value.status == 200
        loaded = joblib.load(io.BytesIO(served[0]))

    assert type(loaded) is SuperLSS
    assert [record["predictor"] for record in loaded._editor_history] == ["scale"]
    assert loaded._editor_history[0]["term"] == "scale:age"
