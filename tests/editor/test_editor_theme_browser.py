from __future__ import annotations

import pytest

pytest.importorskip("playwright.sync_api")
pytestmark = pytest.mark.browser

THEME = "() => document.documentElement.dataset.theme"
GROUND = "() => getComputedStyle(document.body).backgroundColor"
CURVE = "() => getComputedStyle(document.querySelector('#chart path.edited')).stroke"


def _channels(colour: str) -> tuple[int, ...]:
    return tuple(int(part) for part in colour.strip("rgba()").split(",")[:3])


def _await_theme(page, theme: str) -> None:
    """The browser delivers a colour-scheme change to the page asynchronously."""
    page.wait_for_function("theme => document.documentElement.dataset.theme === theme", arg=theme)


def test_dark_choice_restyles_the_page_and_survives_a_reload(open_editor_page):
    with open_editor_page() as (page, _session):
        page.emulate_media(color_scheme="light")
        _await_theme(page, "light")
        light_ground, light_curve = page.evaluate(GROUND), page.evaluate(CURVE)
        assert min(_channels(light_ground)) > 240

        page.get_by_role("button", name="Theme: Auto").click()
        assert page.evaluate(THEME) == "dark"
        dark_ground, dark_curve = page.evaluate(GROUND), page.evaluate(CURVE)
        assert max(_channels(dark_ground)) < 48
        assert dark_curve != light_curve
        assert page.get_by_role("button", name="Theme: Dark").is_visible()
        assert page.evaluate("() => localStorage.getItem('superglm.editor.theme')") == "dark"

        page.reload(wait_until="domcontentloaded")
        # The first-paint script restores the choice before the app loads.
        assert page.evaluate(THEME) == "dark"
        page.locator("#chart path.edited").first.wait_for()
        assert page.evaluate(GROUND) == dark_ground
        assert page.evaluate(CURVE) == dark_curve

        # The explicit choice wins over the browser's setting; Auto follows it.
        page.emulate_media(color_scheme="dark")
        page.wait_for_function("() => matchMedia('(prefers-color-scheme: dark)').matches")
        assert page.evaluate(THEME) == "dark"
        page.get_by_role("button", name="Theme: Dark").click()
        assert page.get_by_role("button", name="Theme: Auto").is_visible()
        assert page.evaluate(THEME) == "dark"
        page.emulate_media(color_scheme="light")
        _await_theme(page, "light")
        assert page.evaluate(GROUND) == light_ground
