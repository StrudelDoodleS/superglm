from __future__ import annotations

import threading

import numpy as np
import pandas as pd
import pytest
from playwright.sync_api import sync_playwright

from superglm import Categorical, Spline, SuperGLM
from superglm.editor import EditorSession


@pytest.fixture
def browser_editor_widget():
    rng = np.random.default_rng(20260711)
    n = 120
    X = pd.DataFrame(
        {
            "age": rng.uniform(18.0, 80.0, n),
            "region": rng.choice(["A", "B", "C"], n),
        }
    )
    y = 0.3 + 0.01 * X["age"].to_numpy() + 0.2 * (X["region"] == "B")
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"age": Spline(n_knots=7), "region": Categorical(base="first")},
    )
    model.fit(X, y)
    widget = EditorSession.from_model(model, terms=["age", "region"]).widget()
    try:
        yield widget
    finally:
        widget.close()


@pytest.mark.browser
def test_editor_browser_loads_authoritative_state(browser_editor_widget):
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1180, "height": 720})
        page.goto(browser_editor_widget.app_url)
        page.locator("#chart .edited").first.wait_for()
        assert page.locator("#term").input_value() == "age"
        assert page.locator("#status").get_attribute("style") in (None, "")
        browser.close()


@pytest.mark.browser
def test_editor_browser_suppresses_duplicate_action_while_pending(
    browser_editor_widget, monkeypatch
):
    original_operate = browser_editor_widget._operate
    first_shift_started = threading.Event()
    second_shift_started = threading.Event()
    release_shifts = threading.Event()
    shift_requests = 0
    request_lock = threading.Lock()

    def delayed_operate(operation, term=None):
        nonlocal shift_requests
        if operation == "shift_up":
            with request_lock:
                shift_requests += 1
                request_number = shift_requests
            first_shift_started.set()
            if request_number == 2:
                second_shift_started.set()
            if not release_shifts.wait(timeout=5):
                raise RuntimeError("Timed out waiting to release shift_up")
        return original_operate(operation, term)

    monkeypatch.setattr(browser_editor_widget, "_operate", delayed_operate)

    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True)
        try:
            page = browser.new_page(viewport={"width": 1180, "height": 720})
            page.goto(browser_editor_widget.app_url)
            edited_path = page.locator("#chart path.edited").first
            edited_path.wait_for()
            page.locator("#mode").select_option("select")
            page.locator('button[data-op="select_all"]').click()
            page.locator("#selectionMenu").wait_for(state="visible")
            original_path = edited_path.get_attribute("d")

            page.locator('button[data-op="shift_up"]').dblclick()
            assert first_shift_started.wait(timeout=2)
            second_shift_started.wait(timeout=0.5)
            release_shifts.set()

            page.wait_for_function(
                """originalPath =>
                    document.querySelector('#chart path.edited')?.getAttribute('d') !== originalPath
                """,
                arg=original_path,
            )
            assert shift_requests == 1
            assert browser_editor_widget.session.model_revision == 1
        finally:
            release_shifts.set()
            browser.close()


@pytest.mark.browser
def test_editor_browser_failed_term_switch_keeps_authoritative_term(
    browser_editor_widget, monkeypatch
):
    original_set_term = browser_editor_widget._set_term
    original_operate = browser_editor_widget._operate
    operation_terms = []

    def reject_region(term):
        if term == "region":
            raise ValueError("region unavailable")
        return original_set_term(term)

    def record_operation_term(operation, term=None):
        operation_terms.append(browser_editor_widget.selected_term)
        return original_operate(operation, term)

    monkeypatch.setattr(browser_editor_widget, "_set_term", reject_region)
    monkeypatch.setattr(browser_editor_widget, "_operate", record_operation_term)

    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True)
        try:
            page = browser.new_page(viewport={"width": 1180, "height": 720})
            page.goto(browser_editor_widget.app_url)
            page.locator("#chart .edited").first.wait_for()

            page.locator("#term").select_option("region")
            page.wait_for_function(
                "document.querySelector('#status')?.textContent.includes('region unavailable')"
            )

            assert page.locator("#term").input_value() == "age"
            assert browser_editor_widget.selected_term == "age"

            age_points = browser_editor_widget.terms["age"]["n_points"]
            page.locator('button[data-op="select_all"]').click()
            page.wait_for_function(
                "expected => document.querySelector('#status')?.textContent.includes(expected)",
                arg=f"{age_points} of {age_points} selected",
            )
            assert operation_terms == ["age"]
        finally:
            browser.close()


@pytest.mark.browser
def test_editor_browser_lost_term_response_uses_recovered_authoritative_term(
    browser_editor_widget, monkeypatch
):
    original_set_term = browser_editor_widget._set_term
    original_operate = browser_editor_widget._operate
    operation_terms = []

    def apply_region_then_lose_response(term):
        payload = original_set_term(term)
        if term == "region":
            raise ValueError("response lost")
        return payload

    def record_operation_term(operation, term=None):
        operation_terms.append(browser_editor_widget.selected_term)
        return original_operate(operation, term)

    monkeypatch.setattr(browser_editor_widget, "_set_term", apply_region_then_lose_response)
    monkeypatch.setattr(browser_editor_widget, "_operate", record_operation_term)

    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True)
        try:
            page = browser.new_page(viewport={"width": 1180, "height": 720})
            page.goto(browser_editor_widget.app_url)
            page.locator("#chart .edited").first.wait_for()

            page.locator("#term").select_option("region")
            page.wait_for_function(
                "document.querySelector('#status')?.textContent.includes('response lost')"
            )

            assert page.locator("#term").input_value() == "region"
            assert browser_editor_widget.selected_term == "region"

            region_points = browser_editor_widget.terms["region"]["n_points"]
            page.locator('button[data-op="select_all"]').click()
            page.wait_for_function(
                "expected => document.querySelector('#status')?.textContent.includes(expected)",
                arg=f"{region_points} of {region_points} selected",
            )
            assert operation_terms == ["region"]
        finally:
            browser.close()


@pytest.mark.browser
def test_editor_browser_zoom_preserves_handle_visuals(browser_editor_widget):
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True)
        try:
            page = browser.new_page(viewport={"width": 1180, "height": 720})
            page.goto(browser_editor_widget.app_url)
            handles = page.locator("#chart .control-handle")
            handles.first.wait_for()
            initial_handle_count = handles.count()
            initial_contribution_count = page.locator("#chart .basis-contribution").count()
            assert page.locator("#mode").input_value() == "handles"

            page.locator("#mode").select_option("zoom")

            assert page.locator("#mode").input_value() == "zoom"
            assert handles.count() == initial_handle_count
            if initial_contribution_count:
                assert (
                    page.locator("#chart .basis-contribution").count() == initial_contribution_count
                )
        finally:
            browser.close()


@pytest.mark.browser
def test_editor_browser_report_error_clears_mismatched_report(browser_editor_widget, monkeypatch):
    original_report = browser_editor_widget._report

    def fail_final_report(report="validation"):
        if report == "final":
            raise ValueError("final unavailable")
        return original_report(report)

    monkeypatch.setattr(browser_editor_widget, "_report", fail_final_report)

    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True)
        try:
            page = browser.new_page(viewport={"width": 1180, "height": 720})
            page.goto(browser_editor_widget.app_url)
            page.locator("#chart .edited").first.wait_for()

            page.locator('.app-tab[data-view="validation"]').click()
            page.locator("#reportFrame .cv-report").wait_for()
            validation_html = page.locator("#reportFrame").inner_html()
            assert "CV Report" in validation_html

            page.locator('.app-tab[data-view="final"]').click()
            page.wait_for_function(
                "document.querySelector('#reportStatus')?.textContent.includes('final unavailable')"
            )

            final_html = page.locator("#reportFrame").inner_html()
            assert "CV Report" not in final_html
            assert final_html == ""
            assert page.locator("#reportTitle").text_content() == "Final Fit Report"
        finally:
            browser.close()
