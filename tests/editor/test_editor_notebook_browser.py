"""The editor inside a notebook cell, driven in a real browser.

A harness page plays the notebook: it loads the anywidget host module and
hands it a widget model whose messages travel to Python through Playwright,
where the editor's notebook transport answers them as it does in a kernel.
"""

from __future__ import annotations

import base64
import io
import json
import queue
from contextlib import contextmanager

import joblib
import numpy as np
import pandas as pd
import pytest
from PIL import Image
from sklearn.model_selection import KFold

import superglm.editor.notebook as notebook
import superglm.editor.widget as widget_module
from superglm import Categorical, Spline, SuperGLM, cross_validate
from superglm.editor import EditorSession
from superglm.editor.assets import read_app_asset

pytest.importorskip("playwright.sync_api")
pytest.importorskip("anywidget")
pytestmark = pytest.mark.browser

_ORIGIN = "http://notebook.test"
_HARNESS = """<!doctype html>
<html><body><div id="cell"></div><div id="cell2"></div>
<script type="module">
import host from "/host.js";
const listeners = new Set();
const model = {
  get: (name) => window.viewState[name],
  send: (content) => window.pySend(JSON.stringify(content)),
  on: (event, callback) => { if (event === "msg:custom") listeners.add(callback); },
  off: (event, callback) => listeners.delete(callback),
};
async function poll() {
  for (const { content, buffer } of await window.pyPoll()) {
    const bytes = Uint8Array.from(atob(buffer), (character) => character.charCodeAt(0));
    for (const callback of listeners) callback(content, [new DataView(bytes.buffer)]);
  }
  setTimeout(poll, 10);
}
window.viewState = await (await fetch("/view.json")).json();
host.render({ model, el: document.getElementById("cell") });
if (window.viewState.views === 2) host.render({ model, el: document.getElementById("cell2") });
poll();
</script></body></html>"""


class _QueueView:
    """A widget view that queues Python's messages for the harness to poll."""

    model_id = "harness"

    def __init__(self):
        self.outbox: queue.Queue = queue.Queue()
        self.handler = None
        self.most_parts = 0
        self.requests: list[str] = []

    def on_msg(self, handler):
        self.handler = handler

    def send(self, content, buffers=None):
        self.most_parts = max(self.most_parts, content.get("parts", 0))
        self.outbox.put((content, bytes(buffers[0]) if buffers else b""))

    def close(self):
        pass


@contextmanager
def _notebook_editor(
    chromium_browser,
    session,
    monkeypatch,
    views=1,
    dark=False,
    host_filter="",
    host_inverts_dark=False,
):
    view = _QueueView()
    transport = notebook.NotebookTransport
    monkeypatch.setattr(
        widget_module,
        "NotebookTransport",
        lambda app, token: transport(app, token, view=view),
    )
    widget = session.widget(mode="notebook")
    page = chromium_browser.new_page(viewport={"width": 1280, "height": 900})
    try:

        def poll():
            replies = []
            while True:
                try:
                    content, payload = view.outbox.get_nowait()
                except queue.Empty:
                    return replies
                replies.append({"content": content, "buffer": base64.b64encode(payload).decode()})

        page.expose_function("pyPoll", poll)

        def send(text):
            content = json.loads(text)
            view.requests.append(f"{content['method']} {content['url']}")
            view.handler(view, content, [])

        page.expose_function("pySend", send)
        state = {
            "bundle": notebook.app_bundle(),
            "height": 860,
            "views": views,
            "host_inverts_dark": host_inverts_dark,
        }
        assets = {
            "/": ("text/html", _HARNESS),
            "/host.js": ("text/javascript", read_app_asset(notebook.HOST_MODULE).decode()),
            "/view.json": ("application/json", json.dumps(state)),
        }
        page.route(
            f"{_ORIGIN}/**",
            lambda route: route.fulfill(
                content_type=assets[route.request.url.removeprefix(_ORIGIN)][0],
                body=assets[route.request.url.removeprefix(_ORIGIN)][1],
            ),
        )
        if dark:
            page.emulate_media(color_scheme="dark")
        if host_filter:
            page.add_init_script(
                f"document.addEventListener('DOMContentLoaded', () => "
                f"document.getElementById('cell').style.filter = '{host_filter}')"
            )
        page.goto(f"{_ORIGIN}/")
        frame = page.frame_locator("#cell iframe")
        frame.locator("#chart path.edited").first.wait_for()
        if views == 2:
            page.frame_locator("#cell2 iframe").locator("#chart path.edited").first.wait_for()
        yield page, frame, widget, view
    finally:
        page.close()
        widget.close()


@pytest.fixture
def curve_session(editor_browser_model):
    return EditorSession.from_model(
        editor_browser_model, terms=["curve", "territory", "age_band", "long_category"]
    )


def test_the_editor_edits_the_model_from_inside_a_notebook_cell(
    chromium_browser, curve_session, monkeypatch
):
    with _notebook_editor(chromium_browser, curve_session, monkeypatch) as (page, frame, _w, _v):
        # The page lives in its own frame: none of its elements or styles
        # reach the notebook around it.
        assert page.locator("#chart").count() == 0
        assert page.evaluate("getComputedStyle(document.body).backgroundColor") == (
            "rgba(0, 0, 0, 0)"
        )
        assert frame.locator("html").get_attribute("data-theme") in {"light", "dark"}
        # Markup injected into the page cannot run script: it shares the
        # notebook's origin, so it admits only its own modules.
        # The refusal itself is the evidence: a handler that never fired
        # would leave window.injected unset just the same.
        frame.locator("body").evaluate(
            """(body) => {
                window.refused = [];
                document.addEventListener("securitypolicyviolation",
                    (event) => window.refused.push(event.effectiveDirective));
                body.insertAdjacentHTML(
                    "beforeend", '<img src="nope:" onerror="window.injected = 1">'
                );
            }"""
        )
        page.wait_for_function(
            "() => document.querySelector('#cell iframe').contentWindow.refused.length > 0"
        )
        assert frame.locator("body").evaluate("() => window.refused") == ["script-src-attr"]
        assert frame.locator("body").evaluate("() => window.injected") is None

        frame.get_by_role("radiogroup", name="Chart tools").get_by_role(
            "radio", name="Select", exact=True
        ).click()
        frame.locator('button[data-op="select_all"]').click()
        frame.locator("#selectionMenu").wait_for(state="visible")
        frame.get_by_role("button", name="Increase selection").click()
        undo = frame.get_by_role("button", name="Undo edit")
        undo.and_(frame.locator(":enabled")).wait_for()
        assert [record.operation for record in curve_session.history] == ["shift"]
        # The metrics for the edited model follow, over the same messages.
        frame.locator('#metricGrid[data-freshness="current"]').wait_for()

        undo.click()
        frame.get_by_role("button", name="Redo edit").and_(frame.locator(":enabled")).wait_for()
        assert curve_session.history == []


def test_export_arrives_in_parts_and_saves_to_a_kernel_path(
    chromium_browser, curve_session, monkeypatch, tmp_path
):
    monkeypatch.setattr(notebook, "MESSAGE_PART_BYTES", 4096)
    with _notebook_editor(chromium_browser, curve_session, monkeypatch) as (page, frame, w, view):
        frame.locator("#exportAction").click()
        # The kernel's file manager would open on the kernel's machine.
        assert frame.locator("#exportOpenDirectory").is_hidden()

        with page.expect_download() as download:
            frame.locator("#exportDownload").click()
        downloaded = joblib.load(download.value.path())
        assert view.most_parts > 1
        direct = joblib.load(io.BytesIO(w._export_bytes("joblib", None).data))
        np.testing.assert_array_equal(downloaded.result.beta, direct.result.beta)

        frame.locator("#exportDirectory").fill(str(tmp_path))
        frame.locator("#exportSave").click()
        frame.locator("#exportStatus", has_text="Saved").wait_for()
        assert (tmp_path / "superglm_edited_model.joblib").is_file()


def test_a_reattached_cell_rebuilds_the_editor_from_python(
    chromium_browser, curve_session, monkeypatch
):
    with _notebook_editor(chromium_browser, curve_session, monkeypatch) as (page, frame, _w, _v):
        curve_session.select_indices("curve", [0, 1, 2])
        curve_session.shift("curve", 0.1)
        # A notebook detaches and re-attaches a cell's output, as a scrolled
        # or moved cell does; the frame comes back with a blank window.
        page.evaluate(
            """() => {
                const cell = document.getElementById("cell");
                const parent = cell.parentNode;
                parent.removeChild(cell);
                parent.appendChild(cell);
            }"""
        )
        frame.locator("#chart path.edited").first.wait_for()
        frame.get_by_role("button", name="Undo edit").and_(frame.locator(":enabled")).wait_for()


def test_an_edit_in_one_output_refreshes_the_others_once(
    chromium_browser, curve_session, monkeypatch
):
    """Two outputs of one widget share its session; an edit in one shows in both."""
    with _notebook_editor(chromium_browser, curve_session, monkeypatch, views=2) as (
        page,
        first,
        _w,
        view,
    ):
        second = page.frame_locator("#cell2 iframe")
        assert second.get_by_role("button", name="Undo edit").is_disabled()
        first.get_by_role("radiogroup", name="Chart tools").get_by_role(
            "radio", name="Select", exact=True
        ).click()
        first.locator('button[data-op="select_all"]').click()
        first.locator("#selectionMenu").wait_for(state="visible")
        first.get_by_role("button", name="Increase selection").click()
        # The second output re-reads the session it did not change.
        second.get_by_role("button", name="Undo edit").and_(second.locator(":enabled")).wait_for()
        assert [record.operation for record in curve_session.history] == ["shift"]

        # Re-reading sends no notice of its own: the outputs fall quiet.
        page.wait_for_timeout(1000)
        settled = len(view.requests)
        page.wait_for_timeout(1000)
        assert view.requests[settled:] == []


def test_an_operation_edits_the_term_its_own_output_shows(
    chromium_browser, curve_session, monkeypatch, choose_feature
):
    """Python's selected term is shared; an output that shows another term edits its own."""
    with _notebook_editor(chromium_browser, curve_session, monkeypatch, views=2) as (
        page,
        first,
        _w,
        view,
    ):
        second = page.frame_locator("#cell2 iframe")
        for frame in (second, first):
            frame.get_by_role("radiogroup", name="Chart tools").get_by_role(
                "radio", name="Select", exact=True
            ).click()
        second.locator('button[data-op="select_all"]').click()
        second.locator("#selectionMenu").wait_for(state="visible")

        # The first output moves Python's selected term to territory.
        seen = len(view.requests)
        choose_feature(first, "territory")
        first.locator('button[data-op="select_all"]').click()
        first.locator("#selectionMenu").wait_for(state="visible")
        for _ in range(100):  # the second output's re-read of the session
            if "GET /state" in view.requests[seen:]:
                break
            page.wait_for_timeout(50)
        assert "GET /state" in view.requests[seen:]
        assert len(curve_session.selection("territory")) == 10

        # The second output still shows curve, and its operation edits curve.
        second.locator("#selectionMenu").get_by_role("button", name="Increase selection").click()
        second.get_by_role("button", name="Undo edit").and_(second.locator(":enabled")).wait_for()
        assert [record.term for record in curve_session.history] == ["curve"]


def _app_bar_colour(page):
    """The median colour of a patch of the editor's app bar, as drawn on screen."""
    box = page.locator("#cell iframe").bounding_box()
    patch = Image.open(
        io.BytesIO(
            page.screenshot(
                clip={"x": box["x"] + box["width"] / 2, "y": box["y"] + 6, "width": 40, "height": 6}
            )
        )
    ).convert("RGB")
    return np.median(np.asarray(patch, dtype=float).reshape(-1, 3), axis=0)


def test_databricks_dark_theme_inversion_is_undone(chromium_browser, curve_session, monkeypatch):
    """Databricks' dark theme inverts each output; the editor shows its own colours anyway."""
    databricks_dark = "invert(1) saturate(0.5)"
    with _notebook_editor(chromium_browser, curve_session, monkeypatch, dark=True) as (
        page,
        *_rest,
    ):
        own = _app_bar_colour(page)
    with _notebook_editor(
        chromium_browser,
        curve_session,
        monkeypatch,
        dark=True,
        host_filter=databricks_dark,
        host_inverts_dark=True,
    ) as (page, *_rest):
        assert page.locator("#cell iframe").evaluate("f => f.style.filter") == (
            "invert(1) saturate(2)"
        )
        undone = _app_bar_colour(page)
    with _notebook_editor(
        chromium_browser, curve_session, monkeypatch, dark=True, host_filter=databricks_dark
    ) as (page, *_rest):
        inverted = _app_bar_colour(page)
    # The night theme's dark bar comes back; left alone, the host turns it pale.
    assert np.abs(undone - own).max() <= 6, (own, undone)
    assert np.abs(inverted - own).max() > 100, (own, inverted)


def test_run_cv_reports_its_job_over_widget_messages(chromium_browser, monkeypatch):
    rng = np.random.default_rng(20261007)
    n = 400
    X = pd.DataFrame({"age": rng.uniform(18.0, 80.0, n), "region": rng.choice(["C", "A", "B"], n)})
    eta = -0.5 + 0.2 * np.sin(X["age"].to_numpy() / 12.0) + 0.2 * (X["region"] == "B")
    y = rng.poisson(np.exp(eta)).astype(np.float64)

    def model():
        return SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            features={"age": Spline(n_knots=6), "region": Categorical(base="first")},
        )

    supplied = cross_validate(
        model(),
        X.iloc[:300],
        y[:300],
        cv=KFold(3, shuffle=True, random_state=0),
        scoring=("deviance", "gini", "nll"),
        return_estimators=True,
    )
    session = EditorSession.from_model(
        model().fit(X.iloc[:300], y[:300]),
        train_data=(X.iloc[:300], y[:300]),
        validation_data=(X.iloc[300:], y[300:]),
        cv=supplied,
    )
    with _notebook_editor(chromium_browser, session, monkeypatch) as (_page, frame, _w, _v):
        frame.locator("#cvTab").click()
        frame.locator('#reportFrame [data-cv-start="cv"]').click()
        frame.locator('#reportFrame .cv-card-row[data-origin="run"]').first.wait_for()
        assert frame.locator('#reportFrame .cv-card-row[data-origin="run"]').count() == 3
