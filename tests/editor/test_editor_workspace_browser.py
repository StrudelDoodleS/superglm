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

        trigger = menu.locator('button[data-op="linearise"]')
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
        assert popover.locator("[data-popover-heading]").inner_text() == "Straighten selection"
        assert popover.locator("[data-popover-description]").inner_text() == (
            "Interpolate the selected relativities between their first and last points."
        )
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


def test_busy_state_makes_all_editor_regions_inert_and_cleans_up(open_editor_page):
    with open_editor_page() as (page, _session):
        reference_ci = page.locator("#ciToggle")
        reference_ci.focus()
        assert reference_ci.evaluate("node => document.activeElement === node")

        page.evaluate("window.__superglmTest.setAppBusy(true, 'Testing busy state', 'Waiting')")

        for selector in ["#appBar", ".context-bar", "#editorView", "#reportPanel"]:
            assert page.locator(selector).get_attribute("inert") == ""
        assert page.locator(".app-shell").get_attribute("aria-busy") == "true"
        overlay = page.locator("#appBusyOverlay")
        assert overlay.get_attribute("aria-live") is None
        assert overlay.is_visible()
        announcement = page.locator("#appBusyAnnouncement")
        assert announcement.get_attribute("role") == "status"
        assert announcement.get_attribute("aria-live") == "polite"
        assert announcement.get_attribute("tabindex") == "-1"
        assert announcement.evaluate("node => document.activeElement === node")
        assert page.locator("#appBusyTitle").inner_text() == "Testing busy state"
        assert page.locator("#appBusyMessage").inner_text() == "Waiting"
        detail = page.locator("#appBusyDetail")
        assert detail.get_attribute("aria-hidden") == "true"

        page.evaluate(
            "window.__superglmTest.setAppBusy(true, 'Ignored repeat title', 'Ignored repeat detail')"
        )
        assert page.locator("#appBusyTitle").inner_text() == "Testing busy state"
        assert page.locator("#appBusyMessage").inner_text() == "Waiting"
        first_elapsed = detail.inner_text()

        page.evaluate(
            """() => {
                const target = document.querySelector('#appBusyAnnouncement');
                window.__busyLiveMutationCount = 0;
                window.__busyLiveObserver = new MutationObserver(records => {
                    window.__busyLiveMutationCount += records.length;
                });
                window.__busyLiveObserver.observe(target, {
                    subtree: true,
                    childList: true,
                    characterData: true,
                });
            }"""
        )
        page.wait_for_timeout(650)
        observed = page.evaluate(
            """() => {
                window.__busyLiveObserver.disconnect();
                return {
                    mutations: window.__busyLiveMutationCount,
                    elapsed: document.querySelector('#appBusyDetail').textContent,
                };
            }"""
        )
        assert observed["mutations"] == 0
        assert observed["elapsed"] != first_elapsed
        assert page.locator("#appAlert").get_attribute("role") == "alert"
        assert page.locator("#appAlert").get_attribute("aria-live") == "assertive"

        page.evaluate("window.__superglmTest.setAppBusy(false)")

        for selector in ["#appBar", ".context-bar", "#editorView", "#reportPanel"]:
            assert page.locator(selector).get_attribute("inert") is None
        assert page.locator(".app-shell").get_attribute("aria-busy") == "false"
        assert overlay.is_hidden()
        assert reference_ci.evaluate("node => document.activeElement === node")

        page.evaluate("window.__superglmTest.setAppBusy(true, 'Testing fallback', 'Waiting')")
        assert announcement.evaluate("node => document.activeElement === node")
        reference_ci.evaluate("node => { node.hidden = true; }")
        page.evaluate("window.__superglmTest.setAppBusy(false)")

        assert page.evaluate("document.activeElement?.id") in {"term", "inspectorToggle"}
        for selector in ["#appBar", ".context-bar", "#editorView", "#reportPanel"]:
            assert page.locator(selector).get_attribute("inert") is None


def test_text_selection_is_scoped_and_reduced_motion_stops_spinner(open_editor_page):
    with open_editor_page() as (page, _session):
        page.get_by_role("button", name="Help", exact=True).click()
        help_panel = page.get_by_role("tabpanel", name="Help")
        assert help_panel.is_visible()

        for selector in ["#helpPane", "#status", "#reportFrame", "#appAlert"]:
            user_select = page.locator(selector).evaluate(
                "node => getComputedStyle(node).userSelect"
            )
            assert user_select in {"auto", "text"}
        for selector in ["#chart", "#toolRail", "#selectionMenu"]:
            assert (
                page.locator(selector).evaluate("node => getComputedStyle(node).userSelect")
                == "none"
            )

        page.emulate_media(reduced_motion="reduce")
        motion = page.locator(".busy-spinner").evaluate(
            """node => {
                const style = getComputedStyle(node);
                const value = style.animationDuration.split(',')[0].trim();
                const seconds = value.endsWith('ms')
                    ? Number.parseFloat(value) / 1000
                    : Number.parseFloat(value);
                return { name: style.animationName, seconds };
            }"""
        )
        assert motion["name"] == "none" or motion["seconds"] <= 0.000001


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


def test_analyst_can_discover_edit_undo_redo_help_and_save(open_editor_page):
    with open_editor_page() as (page, _session):
        select_chart_tool(page, "Select")
        with page.expect_response(
            lambda response: (
                response.request.method == "POST"
                and response.url.split("?", maxsplit=1)[0].endswith("/select")
            )
        ):
            page.locator("#chart .point").nth(2).click()
        page.locator("#selectionMenu").wait_for(state="visible")

        increase = page.get_by_role("button", name="Increase selection", exact=True)
        increase.focus()
        tooltip = page.get_by_role("tooltip")
        tooltip.wait_for(state="visible")
        assert tooltip.locator("[data-popover-heading]").inner_text() == "Increase selection"
        assert tooltip.locator("[data-popover-description]").inner_text() == (
            "Increase selected relativities by 5%."
        )

        with page.expect_response(
            lambda response: (
                response.request.method == "POST"
                and response.url.split("?", maxsplit=1)[0].endswith("/op")
            )
        ) as edit_response:
            increase.click()
        assert edit_response.value.request.post_data_json == {"operation": "shift_up"}

        undo = page.get_by_role("button", name="Undo edit")
        page.wait_for_function("() => !document.querySelector('#undoAction').disabled")
        assert undo.is_enabled()
        with page.expect_response(
            lambda response: (
                response.request.method == "POST"
                and response.url.split("?", maxsplit=1)[0].endswith("/op")
            )
        ) as undo_response:
            undo.click()
        assert undo_response.value.request.post_data_json == {"operation": "undo"}

        redo = page.get_by_role("button", name="Redo edit")
        page.wait_for_function("() => !document.querySelector('#redoAction').disabled")
        assert redo.is_enabled()
        with page.expect_response(
            lambda response: (
                response.request.method == "POST"
                and response.url.split("?", maxsplit=1)[0].endswith("/op")
            )
        ) as redo_response:
            redo.click()
        assert redo_response.value.request.post_data_json == {"operation": "redo"}

        page.get_by_role("button", name="Help", exact=True).click()
        assert page.get_by_role("tabpanel", name="Help").is_visible()

        save = page.get_by_role("button", name="Save edited model")
        assert save.is_visible()
        with page.expect_response(
            lambda response: (
                response.request.method == "POST"
                and response.url.split("?", maxsplit=1)[0].endswith("/save_directory")
            )
        ):
            save.click()
        page.get_by_role("dialog", name="Save Edited Model").wait_for(state="visible")


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


def test_same_term_control_loss_falls_back_to_select_before_drawing(open_editor_page):
    with open_editor_page() as (page, _session):
        rail = page.get_by_role("radiogroup", name="Chart tools")
        select = rail.get_by_role("radio", name="Select", exact=True)
        handles = rail.get_by_role("radio", name="Handles", exact=True)
        select_chart_tool(page, "Handles")
        assert handles.get_attribute("aria-checked") == "true"

        def remove_controls(route):
            response = route.fetch()
            payload = response.json()
            payload["terms"]["curve"]["controls"] = None
            route.fulfill(response=response, json=payload)

        page.route("**/op", remove_controls)
        with page.expect_response(
            lambda response: (
                response.request.method == "POST"
                and response.url.split("?", maxsplit=1)[0].endswith("/op")
            )
        ):
            page.locator('button[data-op="reset"]').click()

        actual = {
            "select_checked": select.get_attribute("aria-checked"),
            "select_tabindex": select.get_attribute("tabindex"),
            "handles_disabled": handles.is_disabled(),
            "handles_checked": handles.get_attribute("aria-checked"),
            "handles_tabindex": handles.get_attribute("tabindex"),
            "control_count": page.locator("#chart .control-handle").count(),
        }
        assert actual == {
            "select_checked": "true",
            "select_tabindex": "0",
            "handles_disabled": True,
            "handles_checked": "false",
            "handles_tabindex": "-1",
            "control_count": 0,
        }


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


def test_inspector_uses_one_slot_for_summary_history_advanced_and_help(open_editor_page):
    with open_editor_page() as (page, _session):
        inspector = page.get_by_role("complementary", name="Model inspector")

        assert inspector.count() == 1
        assert inspector.get_by_role("tab").all_inner_texts() == [
            "Summary",
            "History",
            "Advanced",
            "Help",
        ]

        inspector.get_by_role("tab", name="Advanced").click()
        advanced = inspector.get_by_role("tabpanel", name="Advanced")
        assert advanced.is_visible()
        assert advanced.get_by_label("Build animation duration").is_visible()
        assert page.locator("#buildDurationWrap").count() == 1

        page.get_by_role("button", name="Help", exact=True).click()
        help_panel = inspector.get_by_role("tabpanel", name="Help")
        assert help_panel.is_visible()
        assert "Straighten selection" in help_panel.inner_text()


def test_inspector_tabs_use_roving_focus_and_arrow_key_selection(open_editor_page):
    with open_editor_page() as (page, _session):
        inspector = page.get_by_role("complementary", name="Model inspector")
        summary = inspector.get_by_role("tab", name="Summary")
        history = inspector.get_by_role("tab", name="History")
        help_tab = inspector.get_by_role("tab", name="Help")

        summary.focus()
        page.keyboard.press("ArrowRight")
        assert history.get_attribute("aria-selected") == "true"
        assert history.get_attribute("tabindex") == "0"
        assert history.evaluate("node => document.activeElement === node")
        assert summary.get_attribute("tabindex") == "-1"

        page.keyboard.press("End")
        assert help_tab.get_attribute("aria-selected") == "true"
        assert help_tab.evaluate("node => document.activeElement === node")

        page.keyboard.press("Home")
        assert summary.get_attribute("aria-selected") == "true"
        page.keyboard.press("ArrowLeft")
        assert help_tab.get_attribute("aria-selected") == "true"
        assert help_tab.evaluate("node => document.activeElement === node")


def test_inspector_toggle_close_scrim_and_escape_restore_the_opener(open_editor_page):
    with open_editor_page(viewport={"width": 900, "height": 720}) as (page, _session):
        inspector = page.locator("#inspector")
        toggle = page.get_by_role("button", name="Inspector", exact=True)
        close = inspector.get_by_role("button", name="Close inspector")
        help_action = page.get_by_role("button", name="Help", exact=True)
        scrim = page.locator("#inspectorScrim")

        assert inspector.get_attribute("data-open") == "false"
        assert toggle.get_attribute("aria-expanded") == "false"

        toggle.click()
        assert inspector.get_attribute("data-open") == "true"
        close.click()
        assert inspector.get_attribute("data-open") == "false"
        assert toggle.evaluate("node => document.activeElement === node")

        toggle.click()
        scrim.click(position={"x": 1, "y": 1})
        assert inspector.get_attribute("data-open") == "false"
        assert toggle.evaluate("node => document.activeElement === node")

        help_action.click()
        assert inspector.get_attribute("data-open") == "true"
        assert not scrim.is_hidden()
        scrim.click(position={"x": 1, "y": 1})
        assert inspector.get_attribute("data-open") == "false"
        assert help_action.evaluate("node => document.activeElement === node")

        help_action.click()
        page.keyboard.press("Escape")
        assert inspector.get_attribute("data-open") == "false"
        assert help_action.evaluate("node => document.activeElement === node")


def test_keyboard_help_restores_the_control_that_owned_focus(open_editor_page):
    with open_editor_page(viewport={"width": 900, "height": 720}) as (page, _session):
        inspector = page.locator("#inspector")
        reference_ci = page.get_by_role("button", name="Reference CI")

        reference_ci.focus()
        page.keyboard.press("?")
        assert inspector.get_by_role("tabpanel", name="Help").is_visible()

        page.keyboard.press("Escape")
        assert inspector.get_attribute("data-open") == "false"
        assert reference_ci.evaluate("node => document.activeElement === node")


def boxes_overlap(first: dict, second: dict) -> bool:
    return not (
        first["x"] + first["width"] <= second["x"]
        or second["x"] + second["width"] <= first["x"]
        or first["y"] + first["height"] <= second["y"]
        or second["y"] + second["height"] <= first["y"]
    )


def test_notebook_view_keeps_chart_and_inspector_side_by_side(open_editor_page):
    with open_editor_page(viewport={"width": 1180, "height": 720}) as (page, _session):
        chart = page.locator("#chart").bounding_box()
        metrics = page.locator(".metrics-strip").bounding_box()
        inspector = page.locator("#inspector").bounding_box()

        assert chart is not None
        assert metrics is not None
        assert inspector is not None
        assert chart["width"] >= 600
        assert metrics["y"] >= chart["y"] + chart["height"]
        assert inspector["x"] > chart["x"] + chart["width"]
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        assert page.evaluate("document.documentElement.scrollHeight <= window.innerHeight")


def test_narrow_view_syncs_the_dismissible_inspector_drawer(open_editor_page):
    with open_editor_page(viewport={"width": 900, "height": 720}) as (page, _session):
        inspector = page.locator("#inspector")
        chart = page.locator("#chart").bounding_box()

        assert chart is not None
        assert inspector.get_attribute("data-open") == "false"
        assert chart["width"] >= 700
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        page.get_by_role("button", name="Help", exact=True).click()
        assert inspector.get_attribute("data-open") == "true"
        page.keyboard.press("Escape")
        assert inspector.get_attribute("data-open") == "false"

        page.set_viewport_size({"width": 1100, "height": 720})
        page.wait_for_function(
            "() => document.querySelector('#inspector')?.dataset.open === 'true'"
        )
        page.set_viewport_size({"width": 900, "height": 720})
        page.wait_for_function(
            "() => document.querySelector('#inspector')?.dataset.open === 'false'"
        )


def test_open_narrow_drawer_clears_its_scrim_when_resized_wide(open_editor_page):
    with open_editor_page(viewport={"width": 900, "height": 720}) as (page, _session):
        inspector = page.locator("#inspector")
        scrim = page.locator("#inspectorScrim")
        reference_ci = page.get_by_role("button", name="Reference CI")

        page.get_by_role("button", name="Help", exact=True).click()
        assert inspector.get_attribute("data-open") == "true"
        assert not scrim.is_hidden()

        page.set_viewport_size({"width": 1100, "height": 720})
        page.wait_for_function("() => !matchMedia('(max-width: 1047px)').matches")
        page.wait_for_function("() => document.querySelector('#inspectorScrim')?.hidden")
        assert scrim.is_hidden()
        scrim.evaluate("node => { node.hidden = false; }")
        assert scrim.evaluate("node => getComputedStyle(node).display === 'none'")
        reference_ci.click()
        assert reference_ci.get_attribute("aria-pressed") == "true"


def test_resize_to_narrow_restores_focus_from_inspector_to_toggle(open_editor_page):
    with open_editor_page(viewport={"width": 1100, "height": 720}) as (page, _session):
        inspector = page.locator("#inspector")
        summary_tab = inspector.get_by_role("tab", name="Summary")
        toggle = page.get_by_role("button", name="Inspector", exact=True)

        summary_tab.focus()
        assert summary_tab.evaluate("node => document.activeElement === node")
        page.set_viewport_size({"width": 900, "height": 720})
        page.wait_for_function(
            "() => document.querySelector('#inspector')?.dataset.open === 'false'"
        )

        assert toggle.evaluate("node => document.activeElement === node")


@pytest.mark.parametrize(
    ("width", "expected_open"),
    [(1047, "false"), (1048, "true")],
)
def test_workspace_breakpoint_preserves_chart_width_without_overflow(
    open_editor_page, width, expected_open
):
    with open_editor_page(viewport={"width": width, "height": 720}) as (page, _session):
        chart = page.locator("#chart").bounding_box()

        assert chart is not None
        assert page.locator("#inspector").get_attribute("data-open") == expected_open
        assert chart["width"] >= 600
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")


def test_short_window_scrolls_without_chart_metric_overlap(open_editor_page):
    with open_editor_page(viewport={"width": 1180, "height": 540}) as (page, _session):
        page.wait_for_function(
            """() => {
                const frame = document.querySelector('#summaryFrame');
                return frame?.getAttribute('aria-busy') === 'false'
                    && frame.textContent.trim().length > 0;
            }"""
        )
        chart = page.locator("#chart").bounding_box()
        metrics = page.locator(".metrics-strip").bounding_box()
        workspace = page.locator(".editor-workspace").bounding_box()

        assert chart is not None
        assert metrics is not None
        assert workspace is not None
        assert chart["height"] >= 360
        assert metrics["height"] < 250
        assert workspace["height"] < 650
        assert metrics["y"] >= chart["y"] + chart["height"]
        assert not boxes_overlap(chart, metrics)
        scroll_height = page.evaluate("document.documentElement.scrollHeight")
        assert scroll_height > page.evaluate("window.innerHeight")
        assert scroll_height < 900
        assert metrics["y"] + metrics["height"] <= scroll_height

        page.locator(".metrics-strip").scroll_into_view_if_needed()
        visible_metrics = page.locator(".metrics-strip").bounding_box()
        assert visible_metrics is not None
        assert visible_metrics["y"] >= 0
        assert visible_metrics["y"] + visible_metrics["height"] <= (
            page.evaluate("window.innerHeight") + 1
        )
