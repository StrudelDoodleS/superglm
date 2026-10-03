from __future__ import annotations

import json
from urllib.parse import urlsplit

import pytest

pytest.importorskip("playwright.sync_api")
pytestmark = pytest.mark.browser

SETTINGS_KEY = "superglm.editor.settings"
SWITCHES = (
    "Refit after every structural change",
    "Keep the reference level when collapsing",
    "Follow the browser's light or dark setting",
    "Request timings",
)
# What a private window or a blocked origin does: every storage call throws.
BLOCK_STORAGE = """(() => {
    const refuse = () => { throw new DOMException('The operation is insecure.', 'SecurityError'); };
    Storage.prototype.getItem = refuse;
    Storage.prototype.setItem = refuse;
    Storage.prototype.removeItem = refuse;
})()"""


def _reload(page, term: str) -> None:
    page.reload(wait_until="domcontentloaded")
    page.locator("#chart path.edited").first.wait_for()
    page.wait_for_function(
        "term => document.querySelector('#status')?.dataset.term === term", arg=term
    )


def _settings_pane(page):
    inspector = page.get_by_role("complementary", name="Model inspector")
    inspector.get_by_role("tab", name="Settings").click()
    pane = inspector.get_by_role("tabpanel", name="Settings")
    pane.wait_for(state="visible")
    return pane


def _checked(pane) -> list[str | None]:
    return [
        pane.get_by_role("switch", name=name).get_attribute("aria-checked") for name in SWITCHES
    ]


def _posted(path: str):
    return lambda response: (
        response.request.method == "POST" and urlsplit(response.url).path == path
    )


def test_settings_are_kept_under_one_key_and_take_effect(open_editor_page):
    with open_editor_page(
        selected_term="territory", collapsed_levels=("territory", ("T02", "T03"))
    ) as (page, _session):
        assert page.locator("#groupDisplayMode").input_value() == "expanded"
        pane = _settings_pane(page)
        assert _checked(pane) == ["false", "true", "true", "false"]
        assert page.locator("#settingsTiming").get_attribute("hidden") is not None

        pane.get_by_role("switch", name="Refit after every structural change").click()
        pane.get_by_role("switch", name="Request timings").click()
        pane.get_by_text("Collapsed", exact=True).click()
        page.locator("#buildDuration").evaluate(
            """node => {
                node.value = '6000';
                node.dispatchEvent(new Event('input', { bubbles: true }));
                node.dispatchEvent(new Event('change', { bubbles: true }));
            }"""
        )

        assert _checked(pane) == ["true", "true", "true", "true"]
        assert page.locator("#settingsTiming").get_attribute("hidden") is None
        assert page.locator("#buildDurationValue").text_content() == "6 s"
        stored = json.loads(page.evaluate(f"() => localStorage.getItem('{SETTINGS_KEY}')"))
        assert stored == {
            "refitEveryChange": True,
            "keepReference": True,
            "followBrowserTheme": True,
            "groupsDefault": "collapsed",
            "buildDurationMs": 6000,
            "showTimings": True,
        }
        # The other preferences keep their own keys.
        assert page.evaluate("() => localStorage.getItem('superglm.editor.theme')") is None

        _reload(page, "territory")
        # A grouped term now opens collapsed, and every choice reads back.
        assert page.locator("#groupDisplayMode").input_value() == "collapsed"
        assert page.evaluate("() => document.querySelector('#chart')._scale.displayIsCollapsed")
        pane = _settings_pane(page)
        assert _checked(pane) == ["true", "true", "true", "true"]
        assert pane.get_by_role("radio", name="Collapsed").is_checked()
        assert page.locator("#buildDuration").input_value() == "6000"


def test_keep_reference_off_goes_with_each_structural_request(open_editor_page):
    # A setting that changes what Python builds travels with every request
    # that builds: the operation's own route, which refits at once, and
    # /stage, which waits for Refit (D8). Territory declares base="first".
    with open_editor_page(selected_term="territory") as (page, session):
        pane = _settings_pane(page)
        pane.get_by_role("switch", name="Keep the reference level when collapsing").click()
        pane.get_by_role("switch", name="Refit after every structural change").click()
        assert _checked(pane)[:2] == ["true", "false"]
        reference = page.locator("#termReference")
        collapse = page.get_by_role("button", name="Collapse", exact=True)

        session.select_levels("territory", ["T04", "T05"])
        _reload(page, "territory")
        page.locator("#selectionMenu").wait_for(state="visible")
        with page.expect_response(_posted("/collapse_levels")) as refitted:
            collapse.click()
        assert refitted.value.status == 200
        assert refitted.value.request.post_data_json["keep_reference"] is False
        page.locator("#chart .level-group-marker").first.wait_for()
        page.locator("#appBusyOverlay").wait_for(state="hidden")
        # The refit chose the reference by its declared rule, not kept it.
        assert reference.text_content() == "reference T01 · first"

        _settings_pane(page).get_by_role(
            "switch", name="Refit after every structural change"
        ).click()
        session.select_levels("territory", ["T01", "T02"])
        _reload(page, "territory")
        page.locator("#selectionMenu").wait_for(state="visible")
        with page.expect_response(_posted("/stage")) as staged:
            collapse.click()
        assert staged.value.status == 200
        assert staged.value.request.post_data_json["keep_reference"] is False
        page.wait_for_function(
            "() => document.querySelector('#refitPendingCount')?.textContent === '1'"
        )
        # Kept, the reference would wait as the group T01+T02; it does not.
        assert reference.get_attribute("data-waiting") == "false"
        assert reference.text_content() == "reference T01 · first"


def test_follow_the_browser_is_the_theme_choice(open_editor_page):
    with open_editor_page() as (page, _session):
        page.emulate_media(color_scheme="dark")
        page.wait_for_function("() => document.documentElement.dataset.theme === 'dark'")
        pane = _settings_pane(page)
        follow = pane.get_by_role("switch", name="Follow the browser's light or dark setting")
        assert follow.get_attribute("aria-checked") == "true"

        # Off keeps the theme on screen, as a choice that outlives the page.
        follow.click()
        assert follow.get_attribute("aria-checked") == "false"
        assert page.evaluate("() => localStorage.getItem('superglm.editor.theme')") == "dark"
        page.emulate_media(color_scheme="light")
        page.wait_for_function("() => !matchMedia('(prefers-color-scheme: dark)').matches")
        assert page.evaluate("() => document.documentElement.dataset.theme") == "dark"

        # On is Auto again: the theme follows the browser and nothing is stored.
        follow.click()
        page.wait_for_function("() => document.documentElement.dataset.theme === 'light'")
        assert page.evaluate("() => localStorage.getItem('superglm.editor.theme')") is None

        # Choosing a theme in the top bar turns the switch off.
        page.get_by_role("button", name="Theme: Auto").click()
        assert follow.get_attribute("aria-checked") == "false"


def test_settings_render_their_defaults_and_still_work_with_storage_blocked(open_editor_page):
    with open_editor_page() as (page, _session):
        errors: list[str] = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.context.add_init_script(BLOCK_STORAGE)
        page.emulate_media(color_scheme="dark")
        _reload(page, "curve")

        # With nothing remembered, the theme follows the browser.
        assert page.evaluate("() => document.documentElement.dataset.theme") == "dark"
        pane = _settings_pane(page)
        assert _checked(pane) == ["false", "true", "true", "false"]
        assert pane.get_by_role("radio", name="Expanded").is_checked()
        assert page.locator("#buildDurationValue").text_content() == "10 s"

        refit_every = pane.get_by_role("switch", name="Refit after every structural change")
        refit_every.click()
        # The change holds for the page, though nothing could be stored.
        assert refit_every.get_attribute("aria-checked") == "true"
        assert errors == []
