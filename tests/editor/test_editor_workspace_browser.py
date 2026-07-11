from __future__ import annotations

import pytest

pytest.importorskip("playwright.sync_api")
pytestmark = pytest.mark.browser


def test_real_editor_boots_and_draws_svg(open_editor_page):
    with open_editor_page() as (page, _session):
        edited_path = page.locator("#chart path.edited").first

        assert page.title() == "SuperGLM Editor"
        assert page.locator("#chart").get_attribute("role") == "img"
        assert page.locator("#term").input_value() == "curve"
        geometry = edited_path.evaluate(
            """path => ({
                namespace: path.namespaceURI,
                command: path.getAttribute('d')?.slice(0, 1),
                length: path.getTotalLength(),
            })"""
        )
        assert geometry["namespace"] == "http://www.w3.org/2000/svg"
        assert geometry["command"] == "M"
        assert geometry["length"] > 0


def test_selection_popovers_keep_focus_and_explain_parent_icons(open_editor_page):
    with open_editor_page() as (page, _session):
        page.select_option("#mode", "select")
        page.locator('button[data-op="select_all"]').click()
        menu = page.locator("#selectionMenu")
        menu.wait_for(state="visible")

        trigger = menu.locator('button[data-op="shift_up"]')
        popover = page.locator("#uiPopover")
        trigger.hover()
        popover.wait_for(state="visible")
        trigger.focus()
        page.mouse.move(4, 4)

        assert popover.is_visible()
        assert trigger.get_attribute("aria-describedby") == "uiPopover"

        page.locator("#term").focus()
        popover.wait_for(state="hidden")
        assert trigger.get_attribute("aria-describedby") is None

        trigger.focus()
        popover.wait_for(state="visible")
        page.keyboard.press("Escape")
        popover.wait_for(state="hidden")
        assert trigger.get_attribute("aria-describedby") is None

        for control, heading in [
            ("level", "Level selected values"),
            ("snap", "Snap selected values"),
        ]:
            parent = menu.locator(f'button[data-help-control="{control}"]')
            parent.hover()
            popover.wait_for(state="visible")
            assert popover.locator("[data-popover-heading]").inner_text() == heading
            page.mouse.move(4, 4)
            popover.wait_for(state="hidden")


def test_application_bar_exposes_views_undo_redo_and_save(open_editor_page):
    with open_editor_page() as (page, _session):
        tabs = page.get_by_role("tablist", name="Editor views")
        assert tabs.get_by_role("tab").all_inner_texts() == [
            "Editor",
            "Validation",
            "Final Fit",
        ]
        assert page.get_by_role("button", name="Undo edit").is_disabled()
        assert page.get_by_role("button", name="Redo edit").is_disabled()
        assert page.get_by_role("button", name="Save edited model").is_visible()

        page.select_option("#mode", "select")
        page.locator('button[data-op="select_all"]').click()
        page.locator("#selectionMenu").wait_for(state="visible")
        page.get_by_role("button", name="Increase selection").click()
        undo = page.get_by_role("button", name="Undo edit")
        page.wait_for_function("() => !document.querySelector('#undoAction').disabled")
        undo.click()
        redo = page.get_by_role("button", name="Redo edit")
        page.wait_for_function("() => !document.querySelector('#redoAction').disabled")
        redo.click()
        page.wait_for_function("() => !document.querySelector('#undoAction').disabled")


def test_context_bar_reports_term_kind_and_edf(open_editor_page):
    with open_editor_page(selected_term="curve") as (page, _session):
        context = page.get_by_role("region", name="Term context")
        assert context.get_by_label("Term").input_value() == "curve"
        assert "spline" in context.locator("#termKind").inner_text().lower()
        assert "EDF" in context.locator("#termEdf").inner_text()
