from __future__ import annotations

import json

import pytest

pytest.importorskip("playwright.sync_api")
pytestmark = pytest.mark.browser

THEME = "() => document.documentElement.dataset.theme"
GROUND = "() => getComputedStyle(document.body).backgroundColor"
CURVE = "() => getComputedStyle(document.querySelector('#chart path.edited')).stroke"
STORED = "key => localStorage.getItem(key)"
KNOB = "() => getComputedStyle(document.querySelector('#themeSwitch .theme-switch-knob')).animationName"
FADING = "() => document.documentElement.classList.contains('theme-fading')"
# The flip's keyframes in play on the switch: [name, duration, delay] in seconds.
SWITCH_ANIMATIONS = """() => [...document.querySelectorAll('#themeSwitch, #themeSwitch *')]
  .map((node) => getComputedStyle(node))
  .filter((style) => style.animationName !== 'none')
  .map((style) => [style.animationName, parseFloat(style.animationDuration), parseFloat(style.animationDelay)])"""
# The fade's colour transitions as they run: [element, property, start
# relative to the knob's keyframes (None while pending), duration in ms].
FADE_TRANSITIONS = """() => {
  const knob = document.getAnimations().find((a) => a.animationName?.startsWith('theme-knob-'));
  const name = (node) => `${node.tagName.toLowerCase()}#${node.id}.${node.getAttribute('class') ?? ''}`;
  return document.getAnimations()
    .filter((a) => a instanceof CSSTransition && /color|shadow/.test(a.transitionProperty))
    .map((a) => [name(a.effect.target), a.transitionProperty,
      a.startTime === null || !knob ? null : a.startTime - knob.startTime, a.effect.getTiming().duration]);
}"""
TWO_FRAMES = "() => new Promise((done) => requestAnimationFrame(() => requestAnimationFrame(done)))"
SCHEME = "() => getComputedStyle(document.documentElement).colorScheme"
# Body text, inherited text in the inspector, and a muted app-bar tab.
TEXT = """() => ['body', '.inspector', '#validationTab']
  .map((selector) => getComputedStyle(document.querySelector(selector)).color)"""
# styles/dark.css: gruvbox's dark0_hard ground and the editor's own edit blue.
DARK_GROUND = "rgb(29, 32, 33)"
DARK_EDIT = "rgb(131, 168, 232)"
DARK_TEXT = ["rgb(235, 219, 178)", "rgb(235, 219, 178)", "rgb(168, 153, 132)"]
FOLLOW = "Follow the browser's light or dark setting"
THEME_KEY = "superglm.editor.theme"
APP_MODULE = "**/assets/main.js*"
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


def _await_landing(page) -> None:
    """The grounds cross-fade until the knob lands and theme.js ends the fade."""
    page.wait_for_function("() => !document.documentElement.classList.contains('theme-fading')")


def test_switch_flips_to_the_warm_dark_and_survives_a_reload(open_editor_page):
    with open_editor_page() as (page, _session):
        page.emulate_media(color_scheme="light")
        _await_theme(page, "light")
        light_ground, light_curve = page.evaluate(GROUND), page.evaluate(CURVE)
        assert min(_channels(light_ground)) > 240
        switch = page.get_by_role("switch", name="Dark theme")
        assert not switch.is_checked()
        # Nothing plays on first paint.
        assert page.evaluate(SWITCH_ANIMATIONS) == []

        switch.click()
        assert page.evaluate(THEME) == "dark"
        assert switch.is_checked()
        assert page.evaluate(KNOB) == "theme-knob-to-night"
        assert page.evaluate(FADING)
        _await_landing(page)
        assert page.evaluate(GROUND) == DARK_GROUND
        assert page.evaluate(CURVE) == DARK_EDIT
        assert page.evaluate(STORED, "superglm.editor.theme") == "dark"
        settings = json.loads(page.evaluate(STORED, "superglm.editor.settings"))
        assert settings["followBrowserTheme"] is False

        page.reload(wait_until="domcontentloaded")
        assert page.evaluate(THEME) == "dark"
        page.locator("#chart path.edited").first.wait_for()
        assert page.get_by_role("switch", name="Dark theme").is_checked()
        # A restored switch does not play.
        assert page.evaluate(SWITCH_ANIMATIONS) == []
        assert page.evaluate(GROUND) == DARK_GROUND

        # The browser still says light; the flipped switch wins until flipped back.
        page.get_by_role("switch", name="Dark theme").click()
        assert page.evaluate(THEME) == "light"
        assert page.evaluate(KNOB) == "theme-knob-to-day"
        _await_landing(page)
        assert page.evaluate(GROUND) == light_ground
        assert page.evaluate(CURVE) == light_curve
        # Each flip restarts the keyframes under the theme it reaches.
        page.get_by_role("switch", name="Dark theme").click()
        assert page.evaluate(KNOB) == "theme-knob-to-night"


def test_the_first_paint_script_alone_restores_the_stored_theme(open_editor_page):
    """index.html sets the theme before the app's module runs, so a reload
    never paints the browser's theme first when a choice is stored."""
    with open_editor_page() as (page, _session):
        for scheme, stored in (("light", "dark"), ("dark", "light")):
            page.emulate_media(color_scheme=scheme)
            page.evaluate("([key, value]) => localStorage.setItem(key, value)", [THEME_KEY, stored])
            # Hold the app's module back, so only the first-paint script runs.
            page.route(APP_MODULE, lambda route: route.abort())
            page.reload(wait_until="domcontentloaded")
            assert page.locator("#chart path.edited").count() == 0
            assert page.evaluate(THEME) == stored
            page.unroute(APP_MODULE)


def test_following_the_browser_again_hands_it_the_switch(open_editor_page):
    with open_editor_page() as (page, _session):
        page.emulate_media(color_scheme="light")
        _await_theme(page, "light")
        switch = page.get_by_role("switch", name="Dark theme")
        switch.click()
        _await_landing(page)
        page.locator("#settingsTab").click()
        follow = page.get_by_label(FOLLOW)
        assert not follow.is_checked()

        follow.click()
        _await_theme(page, "light")
        assert follow.is_checked()
        assert not switch.is_checked()
        assert page.evaluate(STORED, "superglm.editor.theme") is None
        # Handed back, the switch moves without playing.
        assert page.evaluate(SWITCH_ANIMATIONS) == []

        page.emulate_media(color_scheme="dark")
        _await_theme(page, "dark")
        assert switch.is_checked()
        assert page.evaluate(SWITCH_ANIMATIONS) == []
        assert not page.evaluate(FADING)
        assert page.evaluate(GROUND) == DARK_GROUND


def _restarted(fade: list) -> list:
    """Transitions that did not start with the click's keyframes."""
    return [row for row in fade if row[2] is None or abs(row[2]) > 1]


def test_a_flip_is_one_cross_fade(open_editor_page):
    """Every colour fades once, from the click, and is done when the knob lands."""
    with open_editor_page() as (page, _session):
        page.emulate_media(color_scheme="light")
        _await_theme(page, "light")
        switch = page.get_by_role("switch", name="Dark theme")

        switch.click()
        page.evaluate(TWO_FRAMES)
        fade = page.evaluate(FADE_TRANSITIONS)
        assert {"color", "background-color"} <= {prop for _node, prop, _start, _ms in fade}
        assert _restarted(fade) == []
        assert {ms for _node, _prop, _start, ms in fade} == {600}
        # The page keeps its scheme until the knob lands.
        assert page.evaluate(SCHEME) == "light"
        _await_landing(page)
        assert page.evaluate(FADE_TRANSITIONS) == []
        assert page.evaluate(TEXT) == DARK_TEXT
        assert page.evaluate(SCHEME) == "dark"

        # Flipped back mid-flip, each colour turns round from where it is.
        switch.click()
        page.evaluate(TWO_FRAMES)
        switch.click()
        page.evaluate(TWO_FRAMES)
        assert _restarted(page.evaluate(FADE_TRANSITIONS)) == []
        assert page.evaluate(SCHEME) == "dark"
        _await_landing(page)
        assert page.evaluate(FADE_TRANSITIONS) == []
        assert page.evaluate(TEXT) == DARK_TEXT


def test_reduced_motion_lands_the_flip_at_once(open_editor_page):
    with open_editor_page() as (page, _session):
        page.emulate_media(color_scheme="light", reduced_motion="reduce")
        _await_theme(page, "light")
        page.get_by_role("switch", name="Dark theme").click()
        assert page.evaluate(THEME) == "dark"
        played = page.evaluate(SWITCH_ANIMATIONS)
        # The flip still selects its keyframes; none of them takes any time.
        assert "theme-knob-to-night" in [name for name, _duration, _delay in played]
        assert all(duration < 0.001 and delay == 0 for _name, duration, delay in played)
        _await_landing(page)
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
        _await_landing(page)
        assert errors == []
