from __future__ import annotations

import json

import pytest

pytest.importorskip("playwright.sync_api")
pytestmark = pytest.mark.browser

THEME = "() => document.documentElement.dataset.theme"
GROUND = "() => getComputedStyle(document.body).backgroundColor"
CURVE = "() => getComputedStyle(document.querySelector('#chart path.edited')).stroke"
STORED = "key => localStorage.getItem(key)"
# styles/dark.css: gruvbox's dark0_hard ground and the editor's own edit blue.
DARK_GROUND = "rgb(29, 32, 33)"
DARK_EDIT = "rgb(131, 168, 232)"
FOLLOW = "Follow the browser's light or dark setting"
BLOCK_STORAGE = """
Object.defineProperty(window, 'localStorage', {
  configurable: true,
  get() { throw new DOMException('The operation is insecure.', 'SecurityError'); },
});
"""


def _channels(colour: str) -> tuple[int, ...]:
    return tuple(int(part) for part in colour.strip("rgba()").split(",")[:3])


def _await_theme(page, theme: str) -> None:
    """The browser delivers a colour-scheme change to the page asynchronously."""
    page.wait_for_function("theme => document.documentElement.dataset.theme === theme", arg=theme)


def test_switch_flips_to_the_warm_dark_and_survives_a_reload(open_editor_page):
    with open_editor_page() as (page, _session):
        page.emulate_media(color_scheme="light")
        _await_theme(page, "light")
        light_ground, light_curve = page.evaluate(GROUND), page.evaluate(CURVE)
        assert min(_channels(light_ground)) > 240
        switch = page.get_by_role("switch", name="Dark theme")
        assert not switch.is_checked()

        switch.click()
        assert page.evaluate(THEME) == "dark"
        assert switch.is_checked()
        assert page.evaluate(GROUND) == DARK_GROUND
        assert page.evaluate(CURVE) == DARK_EDIT
        assert page.evaluate(STORED, "superglm.editor.theme") == "dark"
        settings = json.loads(page.evaluate(STORED, "superglm.editor.settings"))
        assert settings["followBrowserTheme"] is False

        page.reload(wait_until="domcontentloaded")
        # The first-paint script restores the choice before the app loads.
        assert page.evaluate(THEME) == "dark"
        page.locator("#chart path.edited").first.wait_for()
        assert page.get_by_role("switch", name="Dark theme").is_checked()
        assert page.evaluate(GROUND) == DARK_GROUND

        # The browser still says light; the flipped switch wins until flipped back.
        page.get_by_role("switch", name="Dark theme").click()
        assert page.evaluate(THEME) == "light"
        assert page.evaluate(GROUND) == light_ground
        assert page.evaluate(CURVE) == light_curve


def test_following_the_browser_again_hands_it_the_switch(open_editor_page):
    with open_editor_page() as (page, _session):
        page.emulate_media(color_scheme="light")
        _await_theme(page, "light")
        switch = page.get_by_role("switch", name="Dark theme")
        switch.click()
        page.locator("#settingsTab").click()
        follow = page.get_by_label(FOLLOW)
        assert not follow.is_checked()

        follow.click()
        _await_theme(page, "light")
        assert follow.is_checked()
        assert not switch.is_checked()
        assert page.evaluate(STORED, "superglm.editor.theme") is None

        page.emulate_media(color_scheme="dark")
        _await_theme(page, "dark")
        assert switch.is_checked()
        assert page.evaluate(GROUND) == DARK_GROUND


def test_blocked_storage_follows_the_browser_and_still_flips(open_editor_page):
    with open_editor_page() as (page, _session):
        errors: list[str] = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.emulate_media(color_scheme="dark")
        page.add_init_script(BLOCK_STORAGE)
        page.reload(wait_until="domcontentloaded")
        page.locator("#chart path.edited").first.wait_for()
        assert page.evaluate("() => { try { localStorage; return false; } catch { return true; } }")
        assert page.evaluate(THEME) == "dark"
        switch = page.get_by_role("switch", name="Dark theme")
        assert switch.is_checked()

        switch.click()
        assert page.evaluate(THEME) == "light"
        assert not switch.is_checked()
        assert errors == []
