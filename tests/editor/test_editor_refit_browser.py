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
    _wait_for_editor_idle(page)


def _wait_for_editor_idle(page) -> None:
    page.wait_for_function(
        """() => {
            const metrics = document.querySelector('#metricGrid');
            const summary = document.querySelector('#summaryFrame');
            return metrics?.getAttribute('aria-busy') === 'false'
                && summary?.getAttribute('aria-busy') === 'false'
                && document.querySelector('#appBusyOverlay')?.hidden;
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


def _capture_editor_state(page, session, term: str) -> dict[str, object]:
    return {
        "revision": session.model_revision,
        "model": session.model,
        "history_list": session.history,
        "history": tuple(session.history),
        "redo_list": session.redo_stack,
        "redo": tuple(session.redo_stack),
        "selection": session.selection(term).tolist(),
        "values": session.terms[term].edited_log_effect.tolist(),
        "busy": page.locator(".app-shell").get_attribute("aria-busy"),
        "timing": page.locator("#advancedTiming").text_content(),
        "source": page.locator("#summarySource").input_value(),
        "selected_points": page.locator("#chart .point.selected[data-index]").count(),
    }


def _assert_dismissal_unchanged(
    page, session, term: str, before: dict[str, object], launcher, requests: list[object]
) -> None:
    assert requests == []
    assert page.locator(".app-shell").get_attribute("aria-busy") == before["busy"]
    assert page.locator("#appBusyOverlay").is_hidden()
    assert page.locator("#summarySource").input_value() == before["source"]
    assert page.locator("#advancedTiming").text_content() == before["timing"]
    assert page.locator("#chart .point.selected[data-index]").count() == before["selected_points"]
    assert launcher.evaluate("node => document.activeElement === node")
    assert session.model_revision == before["revision"]
    assert session.model is before["model"]
    assert session.history is before["history_list"]
    assert session.redo_stack is before["redo_list"]
    assert session.selection(term).tolist() == before["selection"]
    assert session.terms[term].edited_log_effect.tolist() == before["values"]
    expected_history = before["history"]
    expected_redo = before["redo"]
    assert len(session.history) == len(expected_history)
    assert len(session.redo_stack) == len(expected_redo)
    assert all(actual is expected for actual, expected in zip(session.history, expected_history))
    assert all(actual is expected for actual, expected in zip(session.redo_stack, expected_redo))


def _abort_structural_requests(page) -> list[object]:
    unexpected: list[object] = []

    def abort(route) -> None:
        unexpected.append(route.request)
        route.abort()

    page.route("**/collapse_levels", abort)
    return unexpected


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
        before = _capture_editor_state(page, session, "long_category")

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

        _assert_dismissal_unchanged(page, session, "long_category", before, collapse, requests)

        collapse.click()
        dialog.wait_for(state="visible")
        page.keyboard.press("Escape")
        dialog.wait_for(state="hidden")
        page.evaluate("() => new Promise(resolve => requestAnimationFrame(() => resolve()))")
        _assert_dismissal_unchanged(page, session, "long_category", before, collapse, requests)

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


def test_redo_only_history_requires_confirmation_and_dismissal_is_atomic(open_editor_page):
    term = "long_category"
    labels = [
        "MyReallyLongCategoryNameThatWouldNeverFit",
        "Family👨‍👩‍👧‍👦DriverCaféCategory",
    ]
    with open_editor_page(selected_term=term) as (page, session):
        session.select_levels(term, labels)
        session.shift(term, 0.05)
        session.undo()
        session.select_levels(term, labels)
        assert session.history == []
        assert len(session.redo_stack) == 1
        _reload_editor(page, term)
        page.locator("#summarySource").evaluate("node => { node.value = 'refit'; }")
        requests = _track_dialog_and_requests(page)
        unexpected_structural = _abort_structural_requests(page)
        dialog = page.locator("#structuralConfirmDialog")
        collapse = page.get_by_role("button", name="Collapse and refit", exact=True)
        before = _capture_editor_state(page, session, term)

        collapse.click()
        dialog.wait_for(state="visible", timeout=1000)
        assert dialog.get_by_role("heading").text_content() == "Collapse levels"
        assert dialog.locator("#structuralConfirmMessage").text_content() == (
            "Collapse levels MyReallyLongCategoryNameThatWouldNeverFit, "
            "Family👨‍👩‍👧‍👦DriverCaféCategory in long_category? "
            "This refit clears 1 manual edit history entry."
        )
        dialog.get_by_role("button", name="Cancel", exact=True).click()
        dialog.wait_for(state="hidden")
        page.evaluate("() => new Promise(resolve => requestAnimationFrame(() => resolve()))")
        assert unexpected_structural == []
        _assert_dismissal_unchanged(page, session, term, before, collapse, requests)

        collapse.click()
        dialog.wait_for(state="visible")
        page.keyboard.press("Escape")
        dialog.wait_for(state="hidden")
        page.evaluate("() => new Promise(resolve => requestAnimationFrame(() => resolve()))")
        assert unexpected_structural == []
        _assert_dismissal_unchanged(page, session, term, before, collapse, requests)


def test_changed_snapshot_requires_fresh_confirmation_before_structural_refit(open_editor_page):
    term = "long_category"
    labels = [
        "MyReallyLongCategoryNameThatWouldNeverFit",
        "Family👨‍👩‍👧‍👦DriverCaféCategory",
    ]
    with open_editor_page(selected_term=term) as (page, session):
        session.select_levels(term, labels)
        session.shift(term, 0.05)
        session.shift(term, -0.01)
        _reload_editor(page, term)
        requests = _track_dialog_and_requests(page)
        unexpected_structural = _abort_structural_requests(page)
        dialog = page.locator("#structuralConfirmDialog")
        collapse = page.get_by_role("button", name="Collapse and refit", exact=True)

        collapse.click()
        dialog.wait_for(state="visible")
        assert dialog.locator("#structuralConfirmMessage").text_content() == (
            "Collapse levels MyReallyLongCategoryNameThatWouldNeverFit, "
            "Family👨‍👩‍👧‍👦DriverCaféCategory in long_category? "
            "This refit clears 2 manual edit history entries."
        )
        page.keyboard.press("Control+z")
        page.evaluate("() => new Promise(resolve => requestAnimationFrame(() => resolve()))")
        assert requests == []
        assert len(session.history) == 2
        assert session.redo_stack == []

        with page.expect_response(
            lambda response: response.request.method == "POST" and _path(response.url) == "/op"
        ):
            page.locator("#undoAction").evaluate("node => node.click()")
        _wait_for_editor_idle(page)
        page.wait_for_function("() => !document.querySelector('#redoAction').disabled")
        assert len(session.history) == 1
        assert len(session.redo_stack) == 1

        with page.expect_response(
            lambda response: (
                response.request.method == "POST"
                and _path(response.url) == "/op"
                and response.request.post_data_json == {"operation": "select_all"}
            )
        ):
            page.locator('button[data-op="select_all"]').evaluate("node => node.click()")
        selected_labels = list(session.terms[term].levels)
        page.wait_for_function(
            "count => document.querySelectorAll('#chart .point.selected[data-index]').length === count",
            arg=len(selected_labels),
        )
        requests.clear()
        before = _capture_editor_state(page, session, term)

        dialog.get_by_role("button", name="Continue and refit", exact=True).click()
        page.wait_for_function("() => window.__structuralDialogOpenCount === 2", timeout=1000)
        assert dialog.is_visible()
        assert unexpected_structural == []
        assert requests == []
        assert dialog.locator("#structuralConfirmMessage").text_content() == (
            f"Collapse levels {', '.join(selected_labels)} in long_category? "
            "This refit clears 2 manual edit history entries."
        )

        page.keyboard.press("Escape")
        dialog.wait_for(state="hidden")
        page.evaluate("() => new Promise(resolve => requestAnimationFrame(() => resolve()))")
        _assert_dismissal_unchanged(page, session, term, before, collapse, requests)
