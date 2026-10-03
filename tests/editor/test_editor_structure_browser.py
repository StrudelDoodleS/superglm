from __future__ import annotations

from urllib.parse import urlsplit

import numpy as np
import pytest
from tests.test_editor_structure import EPS, _line_residual, _pinning_tolerance

from superglm.editor.payloads import session_payload, timeline_payload
from superglm.editor.shapes import _numeric_edges

pytest.importorskip("playwright.sync_api")
pytestmark = pytest.mark.browser


def _posted(path: str):
    return lambda response: (
        response.request.method == "POST" and urlsplit(response.url).path == path
    )


def _reload_editor(page, term: str) -> None:
    page.reload(wait_until="domcontentloaded")
    page.locator("#chart path.edited").first.wait_for()
    page.wait_for_function(
        "term => document.querySelector('#status')?.dataset.term === term",
        arg=term,
    )


def _plot_point(page, display_index: int) -> dict[str, float]:
    """The client position of a display point's x, halfway down the plot."""
    return page.evaluate(
        """index => {
            const svg = document.querySelector('#chart');
            const scale = svg._scale;
            const point = svg.createSVGPoint();
            point.x = scale.sx(scale.x[index]);
            point.y = scale.margin.top + scale.innerH / 2;
            const client = point.matrixTransform(svg.getScreenCTM());
            return { x: client.x, y: client.y };
        }""",
        display_index,
    )


def _settled_after_refit(page) -> None:
    """Wait until a structural refit has handed the page back to the analyst."""
    page.wait_for_function(
        "() => document.querySelector('#appBusyOverlay')?.hidden"
        " && !document.querySelector('#editorView')?.hasAttribute('inert')"
    )


def _stage_and_refit(page, icon) -> None:
    """Click a structural icon, which stages the change, then Refit it."""
    with page.expect_response(_posted("/stage")) as staged:
        icon.click()
    assert staged.value.status == 200
    page.wait_for_function("() => !document.querySelector('#refitPendingAction').disabled")
    with page.expect_response(_posted("/refit_pending")) as refitted:
        page.locator("#refitPendingAction").click()
    assert refitted.value.status == 200
    _settled_after_refit(page)


def _box_select_x(page, lo: float, hi: float) -> None:
    """Drag a Select box over the whole plot height between two x values."""
    corners = page.evaluate(
        """([lo, hi]) => {
            const svg = document.querySelector('#chart');
            const scale = svg._scale;
            const client = (x, y) => {
                const point = svg.createSVGPoint();
                point.x = x;
                point.y = y;
                const mapped = point.matrixTransform(svg.getScreenCTM());
                return { x: mapped.x, y: mapped.y };
            };
            return [
                client(scale.sx(lo), scale.margin.top + 1),
                client(scale.sx(hi), scale.margin.top + scale.innerH - 1),
            ];
        }""",
        [lo, hi],
    )
    page.mouse.move(corners[0]["x"], corners[0]["y"])
    page.mouse.down()
    page.mouse.move(corners[1]["x"], corners[1]["y"], steps=4)
    with page.expect_response(_posted("/select")):
        page.mouse.up()


def _drawn_y(page) -> list[float]:
    return page.evaluate("() => document.querySelector('#chart')._scale.y")


def _history_sections(page) -> dict[str, list[str]]:
    """The History pane's rows by section, top to bottom."""
    return page.evaluate(
        """() => Object.fromEntries(['waiting', 'applied', 'undone'].map(kind => [
            kind,
            Array.from(
                document.querySelectorAll(`#historyFrame .history-section.${kind} .history-label`),
                node => node.textContent,
            ),
        ]))"""
    )


def _timeline_sections(session) -> dict[str, list[str]]:
    """The sections the session's timeline asks for: newest first, undone in Redo's order."""
    timeline = timeline_payload(session)
    marker = next(i for i, entry in enumerate(timeline) if entry["kind"] == "marker")
    done = timeline[:marker][::-1]
    return {
        "waiting": [entry["label"] for entry in done if entry.get("status") == "waiting"],
        "applied": [entry["label"] for entry in done if entry.get("status") != "waiting"],
        "undone": [entry["label"] for entry in timeline[marker + 1 :]],
    }


def test_line_icon_pins_a_run_of_points_and_undo_and_redo_step_across_it(open_editor_page):
    with open_editor_page() as (page, session):
        before = session.model
        _box_select_x(page, 3.0, 5.0)
        selected = session.selection("curve")
        # Precondition: the box took one contiguous run of the drawn points.
        assert selected.size > 2
        np.testing.assert_array_equal(np.diff(selected), 1)
        unshaped = _drawn_y(page)

        line = page.locator("#shapeLine")
        line.wait_for(state="visible")
        assert line.get_attribute("aria-disabled") == "false"
        _stage_and_refit(page, line)

        spec = session.model._specs["curve"]
        [pinned] = spec.polynomial_ranges
        term = session.terms["curve"]
        assert pinned.degree == 1
        assert pinned.lo <= term.x[selected[0]] and term.x[selected[-1]] <= pinned.hi

        band = page.locator("#chart .shape-range")
        assert band.count() == 1
        assert band.locator(".shape-range-label").text_content() == "Line"
        assert band.get_attribute("data-popover-title") == "Line"
        assert band.get_attribute("data-popover-body") == (
            f"Pinned to a straight line from {pinned.lo:g} to {pinned.hi:g}."
        )

        # The curve the page draws on the range is the pinned line: log of the
        # plotted relativity against x, to the fit's round-off plus one exp and
        # one log per value.
        drawn = page.evaluate(
            "() => { const s = document.querySelector('#chart')._scale;"
            " return { x: s.x, y: s.y }; }"
        )
        x = np.asarray(drawn["x"])
        inside = (x >= pinned.lo) & (x <= pinned.hi)
        effect = np.log(np.asarray(drawn["y"])[inside])
        bound = _pinning_tolerance(session.model, "curve", spec, effect) + 4 * EPS * (
            1.0 + np.max(np.abs(effect))
        )
        assert _line_residual(x[inside], effect) <= bound

        shaped = session.model
        undo = page.locator("#undoAction")
        label = session.structure_history[-1].label
        assert undo.get_attribute("data-popover-body") == f"Undo: {label}"
        with page.expect_response(_posted("/op")):
            undo.click()
        page.wait_for_function("() => !document.querySelector('#chart .shape-range')")
        assert session.model is before
        # A JSON float64 round-trips exactly, so the curve is the very one drawn before.
        assert _drawn_y(page) == unshaped

        redo = page.locator("#redoAction")
        assert redo.get_attribute("data-popover-body") == f"Redo: {label}"
        with page.expect_response(_posted("/op")):
            redo.click()
        page.locator("#chart .shape-range").wait_for()
        assert session.model is shaped
        assert _drawn_y(page) == drawn["y"]


def test_a_numeric_run_holding_too_few_values_disables_the_higher_shapes(open_editor_page):
    with open_editor_page() as (page, session):
        support = session_payload(session)["curve"]["shape"]["support"]
        held = np.array(support["through"][1:]) - np.array(support["below"][:-1])
        # Precondition: some two-point run holds two or three training values.
        [candidates] = np.nonzero((held >= 2) & (held < 4))
        assert candidates.size
        k = int(candidates[0])
        session.select_indices("curve", [k, k + 1])
        _reload_editor(page, "curve")
        page.locator("#selectionMenu").wait_for(state="visible")

        cubic = page.locator("#shapeCubic")
        assert cubic.get_attribute("aria-disabled") == "true"
        assert cubic.get_attribute("data-popover-body") == (
            "Select at least 4 distinct values for a Cubic."
        )
        assert page.locator("#shapeLine").get_attribute("aria-disabled") == "false"


def test_quadratic_on_bands_spans_whole_bands_and_cubic_says_why_not(open_editor_page):
    with open_editor_page(selected_term="age_band") as (page, session):
        session.select_levels("age_band", ["25-34", "35-44", "45-54"])
        _reload_editor(page, "age_band")
        page.locator("#selectionMenu").wait_for(state="visible")

        cubic = page.locator("#shapeCubic")
        assert cubic.get_attribute("aria-disabled") == "true"
        assert cubic.get_attribute("data-popover-body") == "Select at least 4 bands for a Cubic."
        _stage_and_refit(page, page.locator("#shapeQuadratic"))

        declared = session.model._specs["age_band"]._spline_obj.polynomial_ranges
        assert [(r.lo, r.hi, r.degree) for r in declared] == [("25-34", "45-54", 2)]
        # The shaded range runs between the edge bands' positions, where the
        # pinned piece ends, so ranges sharing an edge band meet there.
        extent = page.evaluate(
            """() => {
                const svg = document.querySelector('#chart');
                const { sx, x } = svg._scale;
                const rect = svg.querySelector('.shape-range rect');
                const left = Number(rect.getAttribute('x'));
                return {
                    left,
                    right: left + Number(rect.getAttribute('width')),
                    expected: [sx(x[1]), sx(x[3])],
                };
            }"""
        )
        assert [extent["left"], extent["right"]] == pytest.approx(extent["expected"], abs=1e-9)
        label = page.locator("#chart .shape-range-label")
        assert label.text_content() == "Quadratic"


def test_back_to_back_runs_give_ranges_that_meet(open_editor_page):
    with open_editor_page() as (page, session):
        for run, icon in (((60, 100), "#shapeLine"), ((101, 140), "#shapeFlat")):
            session.select_indices("curve", list(range(run[0], run[1] + 1)))
            _reload_editor(page, "curve")
            page.locator("#selectionMenu").wait_for(state="visible")
            _stage_and_refit(page, page.locator(icon))

        spec = session.model._specs["curve"]
        first, second = spec.polynomial_ranges
        assert (first.degree, second.degree) == (1, 0)
        assert first.hi == second.lo
        # Snapped outward on its own, the second run would start past the first
        # range and leave a free sliver, with a kink at each end, between them.
        grid = session.terms["curve"].x
        assert _numeric_edges(spec, grid[101], grid[140])[0] > first.hi


def test_feature_search_filters_the_list_and_opens_the_first_match(open_editor_page):
    with open_editor_page() as (page, _session):
        feature_list = page.get_by_role("navigation", name="Features")
        # At 1180px the list opens collapsed to a strip; the analyst opens it.
        assert feature_list.get_attribute("data-open") == "false"
        page.locator("#featureListToggle").click()
        assert feature_list.get_attribute("data-open") == "true"
        search = page.get_by_role("searchbox", name="Search features")
        rows = feature_list.locator("[data-term]")
        assert rows.count() == 4

        search.fill("TERR")
        assert [row.get_attribute("data-term") for row in rows.all()] == ["territory"]
        with page.expect_response(_posted("/term")):
            search.press("Enter")
        page.wait_for_function(
            "() => document.querySelector('#status')?.dataset.term === 'territory'"
        )
        current = feature_list.locator('[aria-current="true"]')
        assert current.get_attribute("data-term") == "territory"
        # The shape icons stay hidden on an unordered categorical.
        assert page.locator("#shapeLine").is_hidden()

        search.press("Escape")
        assert search.input_value() == ""
        assert rows.count() == 4


def test_set_reference_icon_needs_exactly_one_level(open_editor_page):
    with open_editor_page(selected_term="territory") as (page, session):
        set_reference = page.locator("#setReference")
        session.select_levels("territory", ["T03", "T04"])
        _reload_editor(page, "territory")
        page.locator("#selectionMenu").wait_for(state="visible")
        assert set_reference.is_hidden()

        session.select_levels("territory", ["T03"])
        _reload_editor(page, "territory")
        set_reference.wait_for(state="visible")
        _stage_and_refit(page, set_reference)
        page.wait_for_function(
            "() => document.querySelector('#termReference')?.textContent"
            " === 'reference T03 · pinned'"
        )


def test_refresh_pulls_a_notebook_side_structural_change(open_editor_page):
    with open_editor_page(selected_term="territory") as (page, session):
        undo = page.locator("#undoAction")
        refresh = page.locator("#refreshAction")
        reference = page.locator("#termReference")
        refresh.wait_for(state="visible")
        page.wait_for_function("() => !document.querySelector('#refreshAction').disabled")
        session.select_levels("territory", ["T01", "T02"])
        session.replace_with_collapsed_levels("territory", method="fit")
        assert undo.is_disabled()
        assert page.locator("#chart .level-group-marker").count() == 0
        assert reference.text_content() == "reference T01 · first"

        refresh.click()
        page.wait_for_function("() => !document.querySelector('#undoAction').disabled")
        assert undo.get_attribute("data-popover-body") == "Undo: collapse T01 + T02 in territory"
        page.locator("#chart .level-group-marker").first.wait_for()
        assert page.locator("#chart .level-group-marker").count() == 2
        # The reference T01 now sits inside the new group, which keeps it.
        assert reference.text_content() == "reference T01+T02 · kept"


def test_revert_is_one_step_that_undo_takes_back(open_editor_page):
    with open_editor_page(
        selected_term="territory", collapsed_levels=("territory", ("T01", "T02"))
    ) as (page, session):
        collapsed = session.model
        revert = page.locator("#revertAction")
        undo = page.locator("#undoAction")
        markers = page.locator("#chart .level-group-marker")
        page.wait_for_function("() => !document.querySelector('#revertAction').disabled")
        assert markers.count() == 2

        with page.expect_response(_posted("/revert_to_original")) as response_info:
            revert.click()
        assert response_info.value.status == 200
        page.wait_for_function("() => document.querySelector('#revertAction').disabled")
        # Nothing is lost, so nothing asked first.
        assert page.locator("dialog[open]").count() == 0
        assert markers.count() == 0
        assert undo.get_attribute("data-popover-body") == "Undo: revert to original model"

        with page.expect_response(_posted("/op")):
            undo.click()
        page.wait_for_function("() => !document.querySelector('#revertAction').disabled")
        assert markers.count() == 2
        assert session.model is collapsed


def test_history_lists_waiting_changes_above_applied_ones_with_ids_and_notes(open_editor_page):
    with open_editor_page() as (page, session):
        _box_select_x(page, 3.0, 5.0)
        with page.expect_response(_posted("/op")):
            page.get_by_role("button", name="Increase selection").click()
        session.stage_structural(
            "shape", "curve", {"lo": 6.0, "hi": 8.0, "degree": 1, "join": "tangent"}
        )
        [step] = session.pending
        _reload_editor(page, "curve")

        page.locator("#historyTab").click()
        page.locator("#historyFrame .history-section.waiting").wait_for()
        assert _history_sections(page) == _timeline_sections(session)
        assert _history_sections(page)["waiting"] == [step.label]
        waiting = page.locator(f'#historyFrame [data-step-id="{step.step_id}"]')
        assert waiting.locator(".history-id").text_content() == step.step_id
        # Undo takes the newest step, which is the waiting one.
        assert page.locator("#historyFrame .history-undo-chip").count() == 1
        assert waiting.locator(".history-undo-chip").count() == 1

        # A note is written in place and saved on Enter.
        note = "Young-driver tail is noise"
        waiting.get_by_role("button", name="Add a note").click()
        field = page.get_by_role("textbox", name="Note for this step")
        field.fill(note)
        with page.expect_request(
            lambda request: request.method == "POST" and urlsplit(request.url).path == "/note"
        ) as note_info:
            field.press("Enter")
        assert note_info.value.post_data_json == {"id": step.step_id, "note": note}
        page.wait_for_function(
            'id => document.querySelector(`[data-step-id="${id}"] .history-note`)',
            arg=step.step_id,
        )
        assert waiting.locator(".history-note").text_content() == note
        assert session.step_notes[step.step_id] == note

        # Escape keeps the note as it was and sends nothing.
        notes: list[object] = []
        page.on(
            "request",
            lambda request: urlsplit(request.url).path == "/note" and notes.append(request),
        )
        waiting.get_by_role("button", name="Edit note").click()
        field.fill("changed my mind")
        field.press("Escape")
        assert notes == []
        assert waiting.locator(".history-note").text_content() == note

        # Undo moves the waiting step under Undone, note and all; Redo puts it back.
        with page.expect_response(_posted("/op")):
            page.keyboard.press("Control+z")
        page.locator("#historyFrame .history-section.undone").wait_for()
        assert _history_sections(page) == _timeline_sections(session)
        undone = page.locator("#historyFrame .history-section.undone .history-note")
        assert undone.text_content() == note
        with page.expect_response(_posted("/op")):
            page.keyboard.press("Control+Shift+z")
        page.locator("#historyFrame .history-section.waiting").wait_for()
        assert _history_sections(page) == _timeline_sections(session)

        # After Refit the step is applied, and its note stays with it.
        with page.expect_response(_posted("/refit_pending")):
            page.keyboard.press("r")
        _settled_after_refit(page)
        page.wait_for_function(
            "() => !document.querySelector('#historyFrame .history-section.waiting')"
        )
        assert _history_sections(page) == _timeline_sections(session)
        assert note in page.locator("#historyFrame .history-section.applied").text_content()


def test_waiting_changes_show_in_the_feature_list_status_line_and_export(open_editor_page):
    with open_editor_page(selected_term="territory") as (page, session):
        session.stage_structural(
            "collapse", "territory", {"levels": ["T02", "T03"], "group_label": None}
        )
        _reload_editor(page, "territory")
        status = page.locator("#status")
        assert status.text_content() == (
            "1 change waiting for refit · the curve and metrics are from the last refit"
        )
        assert page.locator("#status .status-waiting").text_content() == (
            "1 change waiting for refit"
        )
        assert page.locator("#featureList .feature-row-waiting").count() == 1
        assert (
            page.locator('#featureList [data-term="territory"] .feature-row-waiting').count() == 1
        )

        page.locator("#exportAction").click()
        note = page.locator("#exportPendingNote")
        note.wait_for(state="visible")
        assert note.text_content() == (
            "1 waiting change is not included. The export is the last refit."
        )
        page.locator("#exportDialogClose").click()

        # With a selection, the waiting count still leads the line.
        session.select_levels("territory", ["T05"])
        _reload_editor(page, "territory")
        assert status.text_content().startswith("1 change waiting for refit · 1 of ")


def test_a_change_staged_by_its_icon_shows_at_once_in_the_status_line_and_feature_list(
    open_editor_page,
):
    # A stage keeps the model revision, so nothing redraws the page for it
    # unless the waiting change itself is part of what the views key on.
    with open_editor_page(selected_term="territory") as (page, session):
        session.select_levels("territory", ["T02", "T03"])
        _reload_editor(page, "territory")
        status = page.locator("#status")
        dot = page.locator('#featureList [data-term="territory"] .feature-row-waiting')
        assert status.text_content().startswith("2 of 10 selected · ")
        assert dot.count() == 0
        revision = session.model_revision

        with page.expect_response(_posted("/stage")) as staged:
            page.get_by_role("button", name="Collapse", exact=True).click()
        assert staged.value.status == 200
        page.wait_for_function(
            "() => document.querySelector('#refitPendingCount')?.textContent === '1'"
        )
        assert session.model_revision == revision
        assert status.text_content().startswith("1 change waiting for refit · 2 of 10 selected · ")
        assert dot.count() == 1


def test_a_waiting_collapse_is_drawn_dashed_with_a_bracket_under_the_axis(open_editor_page):
    with open_editor_page(selected_term="territory") as (page, session):
        session.stage_structural(
            "collapse", "territory", {"levels": ["T02", "T03"], "group_label": None}
        )
        _reload_editor(page, "territory")
        bracket = page.locator("#chart .pending-group-bracket")
        assert bracket.count() == 1
        assert bracket.locator(".pending-group-label").text_content() == "T02 + T03 · waiting"
        assert page.locator("#chart rect.exposure.waiting").count() == 2
        assert page.locator("#chart .pending-group-ring").count() == 2
        # The curve is still the last refit's: nothing is grouped in force yet.
        assert page.locator("#chart .level-group-marker").count() == 0
        # The bracket has its own row between the level labels and the axis title.
        rows = page.evaluate(
            """() => {
                const svg = document.querySelector('#chart');
                const label = svg.querySelector('.pending-group-label').getBBox();
                const title = svg.querySelector('.x-axis-title').getBBox();
                const ticks = Array.from(svg.querySelectorAll('.x-tick-label'), n => n.getBBox());
                return {
                    ticksBottom: Math.max(...ticks.map(box => box.y + box.height)),
                    labelTop: label.y,
                    labelBottom: label.y + label.height,
                    titleTop: title.y,
                };
            }"""
        )
        assert rows["ticksBottom"] <= rows["labelTop"]
        assert rows["labelBottom"] <= rows["titleTop"]


def test_a_waiting_ungroup_marks_the_levels_that_leave_their_group(open_editor_page):
    with open_editor_page(
        selected_term="territory", collapsed_levels=("territory", ("T02", "T03"))
    ) as (page, session):
        session.select_levels("territory", ["T02", "T03"])
        _reload_editor(page, "territory")
        page.locator("#selectionMenu").wait_for(state="visible")
        with page.expect_response(_posted("/stage")) as staged:
            page.get_by_role("button", name="Ungroup", exact=True).click()
        assert staged.value.status == 200
        page.wait_for_function(
            "() => document.querySelector('#refitPendingCount')?.textContent === '1'"
        )
        # The whole group breaks up, so the draft has no group left to draw.
        assert session_payload(session)["territory"]["pending"]["groups"] == {}
        bracket = page.locator("#chart .pending-group-bracket.ungroup")
        assert bracket.count() == 1
        assert (
            bracket.locator(".pending-group-label").text_content() == "T02, T03 ungrouped · waiting"
        )
        assert bracket.get_attribute("data-popover-body") == (
            "T02, T03 leave the group T02+T03 at the next Refit."
        )
        assert page.locator("#chart rect.exposure.waiting").count() == 2
        # Until Refit the fit still groups them: its markers stay, and no ring
        # announces a new group.
        assert page.locator("#chart .level-group-marker").count() == 2
        assert page.locator("#chart .pending-group-ring").count() == 0

        with page.expect_response(_posted("/refit_pending")):
            page.locator("#refitPendingAction").click()
        _settled_after_refit(page)
        assert page.locator("#chart .pending-group-bracket").count() == 0
        assert page.locator("#chart .level-group-marker").count() == 0


def test_a_waiting_range_is_a_dashed_box_until_refit_pins_it(open_editor_page):
    with open_editor_page() as (page, session):
        session.stage_structural(
            "shape", "curve", {"lo": 3.0, "hi": 5.0, "degree": 1, "join": "tangent"}
        )
        _reload_editor(page, "curve")
        waiting = page.locator("#chart .pending-range")
        assert waiting.count() == 1
        assert (
            waiting.locator(".pending-range-label").text_content().endswith(" · waiting for refit")
        )
        assert page.locator("#chart .shape-range").count() == 0
        [staged] = session_payload(session)["curve"]["pending"]["ranges"]
        extent = page.evaluate(
            """([lo, hi]) => {
                const svg = document.querySelector('#chart');
                const rect = svg.querySelector('.pending-range-box');
                const left = Number(rect.getAttribute('x'));
                return {
                    left,
                    right: left + Number(rect.getAttribute('width')),
                    expected: [svg._scale.sx(lo), svg._scale.sx(hi)],
                };
            }""",
            [staged["lo"], staged["hi"]],
        )
        assert [extent["left"], extent["right"]] == pytest.approx(extent["expected"], abs=1e-9)

        with page.expect_response(_posted("/refit_pending")):
            page.keyboard.press("r")
        _settled_after_refit(page)
        assert waiting.count() == 0
        assert page.locator("#chart .shape-range").count() == 1
