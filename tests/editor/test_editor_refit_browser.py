from __future__ import annotations

import pytest

pytest.importorskip("playwright.sync_api")
pytestmark = pytest.mark.browser


def _path(url: str) -> str:
    from urllib.parse import urlsplit

    return urlsplit(url).path


def _reload_editor(page, term: str) -> None:
    page.reload(wait_until="domcontentloaded")
    page.locator("#chart path.edited").first.wait_for()
    page.wait_for_function(
        "term => document.querySelector('#status')?.dataset.term === term",
        arg=term,
    )
    page.wait_for_function(
        """() => {
            const metrics = document.querySelector('#metricGrid');
            const summary = document.querySelector('#summaryFrame');
            return metrics?.getAttribute('aria-busy') === 'false'
                && summary?.getAttribute('aria-busy') === 'false';
        }"""
    )


def _track_dialog_and_requests(page) -> list[object]:
    requests: list[object] = []

    def record_request(request) -> None:
        requests.append(request)

    page.on("request", record_request)
    page.evaluate(
        """() => {
            const dialog = document.querySelector('#structuralConfirmDialog');
            const showModal = dialog.showModal.bind(dialog);
            window.__structuralDialogOpenCount = 0;
            dialog.showModal = () => {
                window.__structuralDialogOpenCount += 1;
                showModal();
            };
        }"""
    )
    return requests


def test_structural_confirmation_is_bypassed_when_manual_history_is_empty(open_editor_page):
    labels = [
        "MyReallyLongCategoryNameThatWouldNeverFit",
        "Family👨‍👩‍👧‍👦DriverCaféCategory",
    ]
    with open_editor_page(selected_term="long_category") as (page, session):
        session.select_levels("long_category", labels)
        _reload_editor(page, "long_category")
        requests = _track_dialog_and_requests(page)
        collapse = page.get_by_role("button", name="Collapse and refit", exact=True)

        with page.expect_response(
            lambda response: (
                response.request.method == "POST" and _path(response.url) == "/collapse_levels"
            )
        ) as response_info:
            collapse.click()

        response = response_info.value
        page.locator("#appBusyOverlay").wait_for(state="hidden")
        assert response.status == 200
        assert response.request.post_data_json == {
            "term": "long_category",
            "method": "auto",
        }
        assert page.evaluate("window.__structuralDialogOpenCount") == 0
        assert page.locator("#structuralConfirmDialog").get_attribute("open") is None
        assert [_path(request.url) for request in requests].count("/collapse_levels") == 1


def test_structural_confirmation_cancel_escape_and_continue_are_atomic(open_editor_page):
    labels = [
        "MyReallyLongCategoryNameThatWouldNeverFit",
        "Family👨‍👩‍👧‍👦DriverCaféCategory",
    ]
    with open_editor_page(
        selected_term="long_category", viewport={"width": 360, "height": 560}
    ) as (page, session):
        session.select_levels("long_category", labels)
        session.shift("long_category", 0.05)
        session.shift("long_category", -0.01)
        _reload_editor(page, "long_category")

        before_revision = session.model_revision
        before_model = session.model
        before_history = list(session.history)
        before_selection = session.selection("long_category").tolist()
        before_values = session.terms["long_category"].edited_log_effect.tolist()
        before_timing = page.locator("#advancedTiming").text_content()
        page.locator("#summarySource").evaluate("node => { node.value = 'refit'; }")
        requests = _track_dialog_and_requests(page)

        dialog = page.locator("#structuralConfirmDialog")
        collapse = page.get_by_role("button", name="Collapse and refit", exact=True)
        page.evaluate("window.__superglmTest.setAppBusy(true, 'Testing busy guard', 'Waiting')")
        collapse.evaluate("node => node.click()")
        page.evaluate("() => new Promise(resolve => requestAnimationFrame(() => resolve()))")
        assert requests == []
        assert dialog.is_hidden()
        assert page.evaluate("window.__structuralDialogOpenCount") == 0
        page.evaluate("window.__superglmTest.setAppBusy(false)")
        before_busy = page.locator(".app-shell").get_attribute("aria-busy")

        collapse.click()
        dialog.wait_for(state="visible")

        assert dialog.get_by_role("heading").text_content() == "Collapse levels"
        assert dialog.locator("#structuralConfirmMessage").text_content() == (
            "Collapse levels MyReallyLongCategoryNameThatWouldNeverFit, "
            "Family👨‍👩‍👧‍👦DriverCaféCategory in long_category? "
            "This refit clears 2 manual edit history entries."
        )
        dialog_box = dialog.bounding_box()
        assert dialog_box is not None
        assert dialog_box["x"] >= 0
        assert dialog_box["x"] + dialog_box["width"] <= page.evaluate("window.innerWidth")
        assert dialog.locator("#structuralConfirmMessage").evaluate(
            "node => node.scrollWidth <= node.clientWidth"
        )
        dialog.get_by_role("button", name="Cancel", exact=True).click()
        dialog.wait_for(state="hidden")
        page.evaluate("() => new Promise(resolve => requestAnimationFrame(() => resolve()))")

        assert requests == []
        assert page.locator(".app-shell").get_attribute("aria-busy") == before_busy
        assert page.locator("#appBusyOverlay").is_hidden()
        assert page.locator("#summarySource").input_value() == "refit"
        assert page.locator("#advancedTiming").text_content() == before_timing
        assert page.locator("#chart .point.selected[data-index]").count() == 2
        assert collapse.evaluate("node => document.activeElement === node")
        assert session.model_revision == before_revision
        assert session.model is before_model
        assert session.selection("long_category").tolist() == before_selection
        assert session.terms["long_category"].edited_log_effect.tolist() == before_values
        assert len(session.history) == 2
        assert all(actual is expected for actual, expected in zip(session.history, before_history))

        collapse.click()
        dialog.wait_for(state="visible")
        page.keyboard.press("Escape")
        dialog.wait_for(state="hidden")
        page.evaluate("() => new Promise(resolve => requestAnimationFrame(() => resolve()))")
        assert requests == []
        assert collapse.evaluate("node => document.activeElement === node")
        assert page.locator("#summarySource").input_value() == "refit"

        collapse.click()
        dialog.wait_for(state="visible")
        with page.expect_response(
            lambda response: (
                response.request.method == "POST" and _path(response.url) == "/collapse_levels"
            )
        ) as response_info:
            dialog.get_by_role("button", name="Continue and refit", exact=True).click()

        response = response_info.value
        page.locator("#appBusyOverlay").wait_for(state="hidden")
        assert response.status == 200
        assert response.request.post_data_json == {
            "term": "long_category",
            "method": "auto",
        }
        assert [_path(request.url) for request in requests].count("/collapse_levels") == 1
        assert page.evaluate("window.__structuralDialogOpenCount") == 3
        assert collapse.evaluate("node => document.activeElement === node")
