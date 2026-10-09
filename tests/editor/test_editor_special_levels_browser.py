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
