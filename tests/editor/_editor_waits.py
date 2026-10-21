"""Waits on the editor page's own state, shared by the browser suites.

Playwright reports a request before it runs the route handler that holds it,
and the page starts a change's mutation before it answers it, so a test that
reads either a moment after the event, or waits a fixed time for it, is
racing the page.  These wait on what the page itself records.

The first ``page.route`` of a page turns request interception on in the
browser (``Fetch.enable``), and a request the page sends before the renderer has
picked that up is not intercepted: it reaches the server, the route never
runs, and the test waits for a request it can no longer hold.  On a slow runner
that window is wide enough to lose.  ``hold_requests`` therefore waits until the
page's own requests are being intercepted before it returns.
"""

from __future__ import annotations

import itertools

_EDITOR_IDLE = "() => window.__superglmTest?.mutationStatus?.() !== 'running'"
_HOLDS = itertools.count()
_SENTINEL = "**/__intercepting__"
# Fire one request at the sentinel route at a time until the route has seen one.
_PROBE_UNTIL_INTERCEPTED = """() => {
    const probe = window.__interceptionProbe;
    if (probe.hits > 0) return true;
    if (!probe.inFlight) {
        probe.inFlight = true;
        fetch("/__intercepting__").catch(() => {}).finally(() => { probe.inFlight = false; });
    }
    return false;
}"""


def wait_for_editor_idle(page) -> None:
    """Wait until no change is running.

    The page drops a click on Undo, or on a control that would send a change of
    its own, while one runs, and it keeps one running a little past the state
    that shows its result: it settles the paint first.
    """
    page.wait_for_function(_EDITOR_IDLE)


class HeldRoutes(list):
    """The routes held for a request pattern, as the handler receives them."""

    def __init__(self, page, key: str) -> None:
        super().__init__()
        self._page = page
        self._key = key

    def wait(self, count: int = 1) -> None:
        """Wait until the handler has run ``count`` times."""
        self._page.wait_for_function(
            "([key, count]) => window[key] >= count", arg=[self._key, count]
        )


def hold_requests(page, pattern: str) -> HeldRoutes:
    """Hold every request matching ``pattern`` until its route is continued.

    Each held route is recorded in the page as well, so ``wait`` can wait for
    the handler on page state instead of a fixed time.
    """
    wait_for_interception(page)
    key = f"__heldRoutes{next(_HOLDS)}"
    page.evaluate("key => { window[key] = 0; }", key)
    held = HeldRoutes(page, key)

    def hold(route) -> None:
        held.append(route)
        page.evaluate("key => { window[key] += 1; }", key)

    page.route(pattern, hold)
    return held


def wait_for_interception(page) -> None:
    """Return once a request the page sends is run through a route.

    The sentinel route stays: removing the last route would switch interception
    off again, and the route the caller adds next would switch it on afresh.
    """
    page.evaluate("() => { window.__interceptionProbe = { hits: 0, inFlight: false }; }")

    def answer(route) -> None:
        route.fulfill(status=204)
        page.evaluate("() => { window.__interceptionProbe.hits += 1; }")

    page.route(_SENTINEL, answer)
    page.wait_for_function(_PROBE_UNTIL_INTERCEPTED)
