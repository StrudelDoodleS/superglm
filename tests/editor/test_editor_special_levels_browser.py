"""Free levels, Make special and Back on the curve, driven from the browser."""

from __future__ import annotations

import pytest

pytest.importorskip("playwright.sync_api")
pytestmark = pytest.mark.browser

_NO_RING = "() => !document.querySelector('#chart .pending-special-ring')"
_REFIT_READY = "() => !document.querySelector('#refitPendingAction').disabled"


def _refit(page) -> None:
    page.wait_for_function(_REFIT_READY)
    page.locator("#refitPendingAction").click()
    page.wait_for_function(_NO_RING)


def test_free_levels_then_make_special_and_back_on_the_curve(open_editor_page):
    with open_editor_page(selected_term="age_band") as (page, session):
        toggle = page.locator("#freeLevelsToggle")
        toggle.click()
        page.locator("#chart .free-levels .free-level").first.wait_for(state="attached")
        # Every level is on the curve, so every level is compared.
        assert page.locator("#chart .free-levels .free-level").count() == 6
        # The intervals show with Reference CI, and go with it.
        assert page.locator("#chart .free-levels .free-whisker").count() == 0
        page.locator("#ciToggle").click()
        page.locator("#chart .free-levels .free-whisker").first.wait_for(state="attached")
        assert page.locator("#chart .free-levels .free-whisker").count() == 6
        # Two intervals at one level step to either side of it: the free fit's
        # right of its diamond, the curve's left of its point.
        diamond_x = _diamond_centres(page)[2][0]
        free_x = sorted(
            float(x)
            for x in page.locator("#chart .free-levels .free-whisker").evaluate_all(
                "els => els.map(e => e.getAttribute('x1'))"
            )
        )[2]
        curve_x = sorted(
            {
                float(x)
                for x in page.locator("#chart .ci-whisker").evaluate_all(
                    "els => els.filter(e => e.getAttribute('x1') === e.getAttribute('x2'))"
                    ".map(e => e.getAttribute('x1'))"
                )
            }
        )[2]
        assert free_x - diamond_x == pytest.approx(5.0, abs=0.01)
        assert diamond_x - curve_x == pytest.approx(5.0, abs=0.01)
        # The marks keep to the plot, as its points do, when it is zoomed.
        for mark in ("free-level", "free-whisker"):
            node = page.locator(f"#chart .free-levels .{mark}").first
            assert node.get_attribute("clip-path") == "url(#plotClip)"
        assert toggle.get_attribute("aria-pressed") == "true"

        page.locator('#chart .point[data-index="3"]').click()
        make = page.locator("#makeSpecial")
        make.wait_for(state="visible")
        assert page.locator("#returnToCurve").is_hidden()
        make.click()
        page.locator('#chart .pending-special-ring[data-level="45-54"]').wait_for(state="attached")
        assert list(session.model._specs["age_band"]._special_display) == []

        _refit(page)
        assert list(session.model._specs["age_band"]._special_display) == ["45-54"]
        # The comparison belongs to the model it was fitted beside.
        page.wait_for_function("() => !document.querySelector('#chart .free-levels .free-level')")
        assert toggle.get_attribute("aria-pressed") == "false"

        # The special level stays selected, so it can go straight back.
        back = page.locator("#returnToCurve")
        back.wait_for(state="visible")
        assert back.get_attribute("aria-disabled") == "false"
        back.click()
        page.locator('#chart .pending-special-ring[data-level="45-54"]').wait_for(state="attached")
        _refit(page)
        assert list(session.model._specs["age_band"]._special_display) == []
        assert list(session.terms["age_band"].levels) == [
            "18-24",
            "25-34",
            "35-44",
            "45-54",
            "55-64",
            "65+",
        ]


def _diamond_centres(page) -> list[tuple[float, float]]:
    paths = page.locator("#chart .free-levels .free-level").evaluate_all(
        "els => els.map(e => e.getAttribute('d'))"
    )
    # "M cx top L right cy L ...": the centre is the first x and the second point's y.
    return sorted((float(d.split()[1]), float(d.split()[5])) for d in paths)


def test_free_levels_is_one_button_for_the_diamonds_and_their_line_and_comes_back_with_no_fit(
    open_editor_page,
):
    with open_editor_page(selected_term="age_band") as (page, _session):
        fits = []
        page.on(
            "request",
            lambda request: fits.append(request.url) if "/free_levels" in request.url else None,
        )
        # On an ordered term the line is Free levels' own: no Unsmoothed button.
        assert page.locator("#unsmoothedToggle").is_hidden()
        toggle = page.locator("#freeLevelsToggle")
        toggle.click()
        page.locator("#chart .free-levels .free-level").first.wait_for(state="attached")
        diamonds = _diamond_centres(page)
        assert len(diamonds) == 6 and len(fits) == 1
        # One line joins the six diamonds, through each centre.
        line = page.locator("#chart .free-levels path.free-line")
        assert line.count() == 1
        assert line.get_attribute("clip-path") == "url(#plotClip)"
        corners = [
            tuple(map(float, step.split()[1:3]))
            for step in line.get_attribute("d").replace("L", "|L").replace("M", "|M").split("|")
            if step.strip()
        ]
        assert sorted(corners) == [(round(x, 2), round(y, 2)) for x, y in diamonds]
        assert page.locator("#chart .legend-layer line.free-line").count() == 1

        toggle.click()
        page.wait_for_function("() => !document.querySelector('#chart .free-levels')")
        assert toggle.get_attribute("aria-pressed") == "false"
        # On again for the same term and fit: the kept comparison, drawn with no fit.
        toggle.click()
        page.locator("#chart .free-levels path.free-line").wait_for(state="attached")
        assert toggle.get_attribute("aria-pressed") == "true"
        assert _diamond_centres(page) == diamonds
        assert len(fits) == 1
