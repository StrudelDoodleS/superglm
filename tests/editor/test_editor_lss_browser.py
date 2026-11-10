"""A SuperLSS model, driven from the browser."""

from __future__ import annotations

import dataclasses
import io
import re
from urllib.parse import urlsplit

import joblib
import pytest
from sklearn.model_selection import KFold
from tests._lss_editor_fixtures import lognormal_case
from tests.editor._chart_layout import wait_for_chart_to_fit
from tests.editor.test_editor_workspace_browser import select_chart_tool

from superglm import SuperLSS, cross_validate
from superglm.editor.lss.availability import LSS_UNAVAILABLE

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
        titles = page.locator("#featureList .feature-group-title")
        assert titles.get_by_text("Mean", exact=True).count() == 1
        assert titles.get_by_text("CV", exact=True).count() == 1

        rows = page.locator("#featureList [data-term]")
        assert rows.evaluate_all("els => els.map(e => e.dataset.term)") == _LOG_NORMAL_TERMS
        assert rows.count() == len(_LOG_NORMAL_TERMS)
        # Every row carries its own mini curve.
        assert page.locator("#featureList [data-term] svg.mini-curve").count() == rows.count()
        for term in _LOG_NORMAL_TERMS:
            assert page.locator(f'#featureList [data-term="{term}"] svg.mini-curve').count() == 1


def test_cv_curve_axis_and_drag(open_lss_editor_page):
    with open_lss_editor_page(selected_term="scale:age") as (page, session):
        axis_titles = page.locator("#chart text.label").evaluate_all(
            "els => els.map(e => e.textContent)"
        )
        assert "relativity of σ" in axis_titles

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
        assert chip.inner_text() == "CV"
        assert chip.get_attribute("data-role") == "spread"


def test_cv_tab_draws_folds_and_runs_cv_and_final_fit(open_lss_editor_page):
    case = lognormal_case()
    cv = cross_validate(
        case.model,
        case.X,
        case.y,
        cv=KFold(3, shuffle=True, random_state=4),
        fit_mode="fit_reml",
        return_estimators=True,
        error_score="raise",
    )
    with open_lss_editor_page(cv=cv) as (page, _session):
        page.locator("#cvTab").click()
        frame = page.locator("#reportFrame")
        frame.locator(".cv-card").first.wait_for()
        titles = frame.locator(".cv-card-title").all_inner_texts()
        assert [title.split("\n")[0].split(" ·")[0] for title in titles] == [
            "Log score",
            "CRPS",
            "Gini",
            "Actual / expected",
        ]
        assert "closer to 1 is better" in titles[3]
        assert frame.locator(".cv-fold-table tbody tr").count() == 4
        assert "A/E" in frame.locator(".cv-fold-table thead").inner_text()
        terms = frame.locator(".cv-term-list")
        assert terms.locator(".feature-group-title").all_inner_texts() == ["Mean", "CV"]
        listed = terms.locator(".cv-term").evaluate_all("els => els.map(e => e.dataset.cvTerm)")
        assert [key.split(":")[0] for key in listed] == ["mean"] * 3 + ["scale"] * 2
        assert (
            "age" in frame.locator('.cv-term[data-cv-term="scale:age"] .cv-term-name').inner_text()
        )
        assert page.locator("#finalFitRun").count() == 0

        # A chart's heading names its parameter once and says what the curve is a relativity of;
        # the log-normal's second parameter is a relativity of σ, not of the CV shown elsewhere.
        for term, heading in (
            ("scale:age", "age · CV · relativity of σ"),
            ("mean:age", "age · Mean · relativity"),
        ):
            frame.locator(f'.cv-term[data-cv-term="{term}"]').click()
            assert frame.locator(".cv-axis-title").first.text_content() == heading
            assert "fold range" in frame.locator(".cv-legend").inner_text()
        assert "supplied with edit(" not in page.locator("#reportStatus").inner_text()

        # The table draws the folds: each holds out its own piece of the book.
        assert frame.locator(".cv-held").count() == 3
        assert frame.locator(".cv-held .is-out").count() == 3
        assert "Rounds" in frame.locator(".cv-fold-table thead").inner_text()

        pill = page.locator("#jobPill")
        assert pill.is_hidden()
        run = frame.get_by_role("button", name="Run CV on current model")
        assert run.is_enabled()
        # Leaving the tab while it runs: the pill carries on and says how it ended.
        run.click()
        page.locator("#editorTab").click()
        pill.wait_for(state="visible", timeout=30000)
        page.wait_for_function(
            "() => /Run CV finished: Log score/.test(document.querySelector('#jobPill').innerText)",
            timeout=60000,
        )
        pill.click()
        assert page.locator("#cvTab").get_attribute("aria-selected") == "true"
        frame.locator('[data-origin="run"]').first.wait_for(timeout=60000)
        assert "Current model" in frame.locator(".cv-card").first.inner_text()
        pill.wait_for(state="hidden", timeout=30000)

        frame.get_by_role("button", name="Final fit on all rows").click()
        page.wait_for_function(
            "() => /Final fit finished/.test(document.querySelector('#reportFrame').innerText)",
            timeout=60000,
        )
        page.locator("#finalTab").click()
        frame.get_by_text("Refitted on", exact=False).first.wait_for()
        final = frame.locator(".final-fit").inner_text()
        # The session has no test split, so none is said to stay held out; the refit's status
        # is the Diagnose tab's sentence.
        assert "test split" not in final
        assert "Converged" in final
        # Nothing was edited, so there is no change to compare with; A/E reads to four decimals.
        split_table = frame.locator('table[aria-label="Split metrics"]').inner_text()
        assert "Δ" not in split_table
        last_cell = frame.locator('table[aria-label="Split metrics"] tbody tr td').last
        assert re.fullmatch(r"\d\.\d{4}", last_cell.inner_text().strip())


def test_diagnose_tab_lists_the_fits_and_switches_between_them(open_lss_editor_page):
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
    with open_lss_editor_page(cv=cv) as (page, _session):
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.locator("#diagnoseTab").click()
        frame = page.locator("#reportFrame")
        frame.locator(".diagnose-table").first.wait_for()
        assert page.locator("#diagnoseTab").get_attribute("aria-selected") == "true"
        fits = frame.locator(".diagnose-fits [data-diagnose-fit]")
        assert fits.all_inner_texts() == ["All rows", "Fold 1", "Fold 2", "Fold 3"]
        # What the converged fits share is said once, not on every row, and no code shows.
        assert frame.inner_text().count("passed its own check") == 1
        assert "outer=" not in frame.inner_text()
        assert frame.locator(".diagnose-matrix .diagnose-group th").all_inner_texts() == [
            "Mean",
            "CV",
        ]
        assert frame.locator(".diagnose-line").count() == 4
        # The timings are folded until asked for.
        assert frame.locator(".diagnose-phase").count() == 0
        frame.locator('[data-diagnose-fold="phases"]').click()
        assert frame.locator(".diagnose-phase").count() > 0

        fits.nth(2).click()
        pressed = frame.locator('[data-diagnose-fit][aria-pressed="true"]')
        assert pressed.inner_text() == "Fold 2"
        assert "Fold 2's held-out rows" in frame.inner_text()
        assert frame.locator('table[aria-label="Rows that cost the most"] tbody tr').count() == 10
        # A poll or a return to the tab keeps the fit in view.
        page.locator("#cvTab").click()
        page.locator("#reportFrame .cv-card").first.wait_for()
        page.locator("#diagnoseTab").click()
        frame.locator(".diagnose-table").first.wait_for()
        assert frame.locator('[data-diagnose-fit][aria-pressed="true"]').inner_text() == "Fold 2"
        assert errors == []


def test_a_slow_fold_s_mark_opens_it_in_diagnose(open_lss_editor_page):
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
    slow = dataclasses.replace(cv, fold_scores=cv.fold_scores.assign(n_reml_iter=[10, 40, 10]))
    with open_lss_editor_page(cv=slow) as (page, _session):
        page.locator("#cvTab").click()
        frame = page.locator("#reportFrame")
        frame.locator(".cv-card").first.wait_for()
        marks = frame.locator("[data-diagnose-open]")
        assert marks.count() == 1
        marks.first.click()
        frame.locator(".diagnose-table").first.wait_for()
        assert page.locator("#diagnoseTab").get_attribute("aria-selected") == "true"
        assert frame.locator('[data-diagnose-fit][aria-pressed="true"]').inner_text() == "Fold 2"


def test_toolbar_tooltips_stay_in_one_place_along_the_row(open_lss_editor_page):
    with open_lss_editor_page() as (page, _session):
        popover = page.locator("#uiPopover")
        edges = {}
        for button in ("#revertAction", "#refreshAction", "#revertAction"):
            page.locator(button).hover()
            popover.wait_for(state="visible")
            box = popover.bounding_box()
            title = popover.locator("[data-popover-heading]").inner_text()
            edges.setdefault(button, []).append((title, box["x"] + box["width"], box["y"]))
        assert edges["#revertAction"][0][0] == "Revert to original model"
        assert edges["#refreshAction"][0][0] == "Refresh from Python"
        assert edges["#revertAction"][0][1:] == edges["#refreshAction"][0][1:]
        assert edges["#revertAction"][1][1:] == edges["#refreshAction"][0][1:]

        # In the gap between two buttons it stays open; away from the row it closes.
        revert = page.locator("#revertAction").bounding_box()
        refresh = page.locator("#refreshAction").bounding_box()
        gap_x = (revert["x"] + revert["width"] + refresh["x"]) / 2
        assert revert["x"] + revert["width"] < gap_x < refresh["x"]
        page.mouse.move(gap_x, revert["y"] + revert["height"] / 2)
        assert popover.is_visible()
        page.mouse.move(4, 400)
        popover.wait_for(state="hidden")


def test_a_tooltip_does_not_outlive_a_button_re_rendered_away(open_lss_editor_page):
    with open_lss_editor_page() as (page, _session):
        popover = page.locator("#uiPopover")
        page.locator("#refreshAction").hover()
        popover.wait_for(state="visible")
        page.evaluate("""() => {
          const button = document.getElementById('refreshAction');
          button.replaceWith(button.cloneNode(true));
        }""")
        page.mouse.move(4, 400)
        popover.wait_for(state="hidden")


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


def test_tools_the_session_does_not_offer_are_neither_requested_nor_offered(open_lss_editor_page):
    requested: list[str] = []

    def record(page):
        page.on("request", lambda request: requested.append(urlsplit(request.url).path))

    with open_lss_editor_page(selected_term="scale:region", prepare=record) as (page, _session):
        state = LSS_UNAVAILABLE
        metrics, summary = page.locator("#metricGrid"), page.locator("#summaryFrame")
        page.wait_for_function(
            "() => ['metricGrid', 'summaryFrame'].every("
            "id => document.getElementById(id)?.dataset.freshness === 'unavailable')"
        )
        assert metrics.inner_text() == state["metrics"]
        assert summary.inner_text() == state["summary"]
        assert metrics.get_attribute("aria-busy") == summary.get_attribute("aria-busy") == "false"

        # An edit schedules the evidence the page shows; none of it is these routes.
        select_chart_tool(page, "Select")
        with page.expect_response(_posted("/select")):
            page.locator("#chart .point").nth(2).click()
        page.locator("#selectionMenu").wait_for(state="visible")
        with page.expect_response(_posted("/op")):
            page.get_by_role("button", name="Increase selection", exact=True).click()
        page.locator("#historyTab").click()
        page.locator("#summaryTab").click()

        assert page.locator(".evidence-retry:visible").count() == 0
        refit_offset = page.locator("#refitOffset")
        assert refit_offset.is_disabled()
        assert refit_offset.get_attribute("title") == state["refit_offset"]
        assert not {"/summary", "/metrics"} & set(requested)
        assert "/op" in requested
