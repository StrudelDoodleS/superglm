"""The Knots tool, driven from the browser against the real routes."""

from __future__ import annotations

import pytest

pytest.importorskip("playwright.sync_api")
pytestmark = pytest.mark.browser

_CHIP = "#termKnots"


def _in_force_count(session) -> int:
    return int(session.model._specs["curve"].fitted_base_knots.size)


def _wait_for_chip(page, text: str, *, waiting: bool) -> None:
    page.wait_for_function(
        """([selector, text, waiting]) => {
            const chip = document.querySelector(selector);
            return chip && !chip.hidden && chip.textContent === text
                && chip.dataset.waiting === String(waiting);
        }""",
        arg=[_CHIP, text, waiting],
    )


def test_one_knot_more_waits_for_refit_and_undo_takes_it_back(open_editor_page):
    with open_editor_page() as (page, session):
        # The curve term is Spline(n_knots=7): seven evenly spaced knots, ticked under the axis.
        _wait_for_chip(page, "7 knots · even spacing", waiting=False)
        assert page.locator("#chart .knot-tick").count() == 7
        assert _in_force_count(session) == 7

        page.get_by_role("radiogroup", name="Chart tools").get_by_role(
            "radio", name="Knots", exact=True
        ).click()
        page.locator("#chart .knot-handle").first.wait_for()
        assert page.locator("#chart .knot-handle").count() == 7
        assert page.locator("#knotCount").inner_text() == "7"

        page.get_by_role("button", name="One knot more").click()
        _wait_for_chip(page, "8 knots · even spacing", waiting=True)
        refit = page.locator("#refitPendingAction")
        page.wait_for_function("() => !document.querySelector('#refitPendingAction').disabled")
        assert refit.get_attribute("aria-label") == "Refit, 1 change waiting"
        # The Refit count and the chip show the change waiting; the status line keeps to the
        # gestures, so they fit it.
        assert page.locator("#status").inner_text().startswith("Knots. Drag to move")
        # The waiting draft's knots are drawn; the fit in force is untouched until Refit.
        assert page.locator("#chart .knot-handle").count() == 8
        assert _in_force_count(session) == 7

        refit.click()
        _wait_for_chip(page, "8 knots · even spacing", waiting=False)
        assert _in_force_count(session) == 8

        # Undo after a Refit brings the change back as waiting, the fit before it in force.
        page.get_by_role("button", name="Undo edit").click()
        _wait_for_chip(page, "8 knots · even spacing", waiting=True)
        assert _in_force_count(session) == 7
        page.get_by_role("button", name="Undo edit").click()
        _wait_for_chip(page, "7 knots · even spacing", waiting=False)
        assert page.locator("#chart .knot-handle").count() == 7
        assert refit.is_disabled()


def test_a_dropped_knot_stays_where_it_was_dropped_while_its_change_is_staged(open_editor_page):
    """The handle must not flash back to its old place before the staged change answers."""
    with open_editor_page() as (page, session):
        page.get_by_role("radiogroup", name="Chart tools").get_by_role(
            "radio", name="Knots", exact=True
        ).click()
        handles = page.locator("#chart .knot-handle")
        handles.first.wait_for()
        before = [float(x) for x in handles.evaluate_all("els => els.map(e => e.dataset.knotX)")]
        released = []

        def held(route):
            # Read the knots while the stage request is still in flight.
            released.extend(
                float(x) for x in handles.evaluate_all("els => els.map(e => e.dataset.knotX)")
            )
            route.continue_()

        page.route("**/stage*", held)
        box = handles.nth(3).bounding_box()
        y = box["y"] + box["height"] / 2
        page.mouse.move(box["x"] + box["width"] / 2, y)
        page.mouse.down()
        page.mouse.move(box["x"] + box["width"] / 2 + 60, y, steps=8)
        page.mouse.up()
        _wait_for_chip(page, "7 knots · placed by hand", waiting=True)
        assert released, "the knot change was not staged"
        assert before[3] not in released
        assert len(released) == len(before)


def test_arrow_keys_pressed_while_a_knot_change_is_staged_move_the_same_knot(open_editor_page):
    """Each press acts on the knots as the presses left them; the latest follows the first."""
    with open_editor_page() as (page, session):
        page.get_by_role("radiogroup", name="Chart tools").get_by_role(
            "radio", name="Knots", exact=True
        ).click()
        handles = page.locator("#chart .knot-handle")
        handles.first.wait_for()
        before = [float(x) for x in handles.evaluate_all("els => els.map(e => e.dataset.knotX)")]
        box = handles.nth(3).bounding_box()
        page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        held = []

        def hold(route):
            held.append(route)

        page.route("**/stage*", hold)
        page.keyboard.press("ArrowRight")
        for _ in range(100):
            if held:
                break
            page.wait_for_timeout(20)
        assert len(held) == 1
        page.keyboard.press("ArrowRight")
        page.keyboard.press("ArrowRight")
        # A control that would send a change of its own says why it waits.
        page.get_by_role("button", name="One knot more").click()
        page.wait_for_function(
            "text => document.querySelector('#status')?.textContent.includes(text)",
            arg="Another change is still running; try again once it has finished.",
        )
        # Unrouting sends the held request on.
        page.unroute("**/stage*")
        page.wait_for_function(
            "() => document.querySelector('#refitPendingAction')?.getAttribute('aria-label')"
            " === 'Refit, 2 changes waiting'"
        )
        moved = session.pending[-1].params["positions"]
        assert [x for i, x in enumerate(moved) if i != 3] == [
            x for i, x in enumerate(before) if i != 3
        ]
        assert 0.25 < moved[3] - before[3] < 0.35


def _in_force_kind(session) -> str:
    return type(session.model._specs["curve"]).__name__


def _wait_for_kind(page, kind: str, *, waiting: bool) -> None:
    page.wait_for_function(
        """([kind, waiting]) => {
            const select = document.querySelector("#knotKind");
            return select && select.value === kind && select.dataset.waiting === String(waiting);
        }""",
        arg=[kind, waiting],
    )


def test_a_kind_change_waits_for_refit_and_undo_takes_it_back(open_editor_page):
    with open_editor_page() as (page, session):
        # The curve term is Spline(n_knots=7): a P-spline, without shrinkage.
        page.get_by_role("radiogroup", name="Chart tools").get_by_role(
            "radio", name="Knots", exact=True
        ).click()
        kind = page.get_by_role("combobox", name="Kind")
        _wait_for_kind(page, "ps", waiting=False)
        assert page.get_by_role("button", name="Shrink").get_attribute("aria-pressed") == "false"

        kind.select_option("cr")
        _wait_for_kind(page, "cr", waiting=True)
        refit = page.locator("#refitPendingAction")
        page.wait_for_function("() => !document.querySelector('#refitPendingAction').disabled")
        assert refit.get_attribute("aria-label") == "Refit, 1 change waiting"
        undo = page.get_by_role("button", name="Undo edit")
        assert undo.get_attribute("data-popover-body") == "Undo: kind cr in curve"
        assert _in_force_kind(session) == "PSpline"

        refit.click()
        _wait_for_kind(page, "cr", waiting=False)
        assert _in_force_kind(session) == "CubicRegressionSpline"
        # The knots stay where they were.
        assert page.locator("#chart .knot-handle").count() == 7

        # Undo after a Refit brings the change back as waiting, the fit before it in force.
        undo.click()
        _wait_for_kind(page, "cr", waiting=True)
        assert _in_force_kind(session) == "PSpline"
        undo.click()
        _wait_for_kind(page, "ps", waiting=False)
        assert refit.is_disabled()
