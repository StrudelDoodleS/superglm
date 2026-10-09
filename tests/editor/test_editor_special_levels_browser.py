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
        # Each whisker carries a tick at the fitted curve its flag is judged against,
        # and the marks keep to the plot, as its points do, when it is zoomed.
        assert page.locator("#chart .free-levels .free-curve-tick").count() == 6
        for mark in ("free-level", "free-curve-tick"):
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


def test_free_levels_off_and_on_again_takes_no_fit_and_unsmoothed_runs_through_its_diamonds(
    open_editor_page,
):
    with open_editor_page(selected_term="age_band") as (page, _session):
        fits = []
        page.on(
            "request",
            lambda request: fits.append(request.url) if "/free_levels" in request.url else None,
        )
        toggle = page.locator("#freeLevelsToggle")
        toggle.click()
        page.locator("#chart .free-levels .free-level").first.wait_for(state="attached")
        diamonds = _diamond_centres(page)
        assert len(diamonds) == 6 and len(fits) == 1

        toggle.click()
        page.wait_for_function("() => !document.querySelector('#chart .free-levels .free-level')")
        assert toggle.get_attribute("aria-pressed") == "false"
        # On again for the same term and fit: the kept comparison, drawn with no fit.
        toggle.click()
        page.locator("#chart .free-levels .free-level").first.wait_for(state="attached")
        assert toggle.get_attribute("aria-pressed") == "true"
        assert _diamond_centres(page) == diamonds
        assert len(fits) == 1

        # The Unsmoothed line is the same fit: a dot on each level, on its diamond.
        page.locator("#unsmoothedToggle").click()
        dots = page.locator("#chart .unsmoothed-layer .unsmoothed-dot")
        dots.first.wait_for(state="attached")
        centres = dots.evaluate_all(
            "els => els.map(e => [Number(e.getAttribute('cx')), Number(e.getAttribute('cy'))])"
        )
        assert len(centres) == 6
        assert page.locator("#chart .unsmoothed-layer path.unsmoothed").count() == 1
        # Measured again: the line may widen the y-axis.
        diamonds = _diamond_centres(page)
        for (dx, dy), (fx, fy) in zip(sorted(map(tuple, centres)), diamonds, strict=True):
            assert abs(dx - fx) <= 0.01 and abs(dy - fy) <= 0.01
