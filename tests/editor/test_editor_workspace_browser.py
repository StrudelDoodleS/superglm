from __future__ import annotations

import pytest

pytest.importorskip("playwright.sync_api")
pytestmark = pytest.mark.browser


def select_chart_tool(page, name: str) -> None:
    page.get_by_role("radiogroup", name="Chart tools").get_by_role(
        "radio", name=name, exact=True
    ).click()


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
        select_chart_tool(page, "Select")
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

        select_chart_tool(page, "Select")
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
        inspector_toggle = context.get_by_role("button", name="Inspector")
        controlled_id = inspector_toggle.get_attribute("aria-controls")
        assert controlled_id
        assert page.locator(f"#{controlled_id}").count() == 1


def test_tool_rail_selects_one_mode_and_supports_roving_shortcuts(open_editor_page):
    with open_editor_page() as (page, _session):
        rail = page.get_by_role("radiogroup", name="Chart tools")
        select = rail.get_by_role("radio", name="Select", exact=True)
        move = rail.get_by_role("radio", name="Move", exact=True)
        zoom = rail.get_by_role("radio", name="Zoom", exact=True)
        handles = rail.get_by_role("radio", name="Handles", exact=True)

        assert select.get_attribute("aria-checked") == "true"
        assert select.get_attribute("tabindex") == "0"
        assert move.get_attribute("aria-checked") == "false"

        move.click()
        assert move.get_attribute("aria-checked") == "true"
        assert move.get_attribute("tabindex") == "0"
        assert select.get_attribute("aria-checked") == "false"
        assert select.get_attribute("tabindex") == "-1"

        move.focus()
        page.keyboard.press("ArrowDown")
        assert zoom.get_attribute("aria-checked") == "true"
        assert zoom.evaluate("node => document.activeElement === node")

        page.keyboard.press("End")
        assert handles.get_attribute("aria-checked") == "true"
        assert handles.evaluate("node => document.activeElement === node")

        page.locator("#chart").focus()
        page.keyboard.press("v")
        assert select.get_attribute("aria-checked") == "true"
        page.keyboard.press("?")
        assert page.get_by_role("button", name="Help", exact=True).is_visible()


def test_handles_tool_is_disabled_only_when_the_term_has_no_controls(open_editor_page):
    with open_editor_page(selected_term="territory") as (page, _session):
        handles = page.get_by_role("radiogroup", name="Chart tools").get_by_role(
            "radio", name="Handles", exact=True
        )
        assert handles.is_disabled()

        with page.expect_response(
            lambda response: (
                response.request.method == "POST"
                and response.url.split("?", maxsplit=1)[0].endswith("/term")
            )
        ):
            page.select_option("#term", "curve")
        page.wait_for_function(
            "term => document.querySelector('#status')?.dataset.term === term", arg="curve"
        )
        assert handles.is_enabled()
        assert (
            page.get_by_role("radiogroup", name="Chart tools")
            .get_by_role("radio", name="Select", exact=True)
            .get_attribute("aria-checked")
            == "true"
        )


def test_existing_svg_selection_operation_posts_linearise_unchanged(open_editor_page):
    with open_editor_page() as (page, session):
        select_chart_tool(page, "Select")
        with page.expect_response(
            lambda response: (
                response.request.method == "POST"
                and response.url.split("?", maxsplit=1)[0].endswith("/op")
            )
        ):
            page.locator('button[data-op="select_all"]').click()
        page.locator("#selectionMenu").wait_for(state="visible")
        straighten = page.get_by_role("button", name="Straighten selection", exact=True)

        with page.expect_response(
            lambda response: (
                response.request.method == "POST"
                and response.url.split("?", maxsplit=1)[0].endswith("/op")
            )
        ) as operation_response:
            straighten.click()

        response = operation_response.value
        assert response.status == 200
        assert response.request.post_data_json == {"operation": "linearise"}
        assert session.history[-1].operation == "linear_interpolate"
