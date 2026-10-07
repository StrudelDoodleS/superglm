"""The editor's notebook mode: its page bundle and its message transport."""

from __future__ import annotations

import io
import json
import queue
import re
import threading

import joblib
import numpy as np
import pandas as pd
import pytest

import superglm.editor.notebook as notebook
import superglm.editor.widget as widget_module
from superglm import Categorical, Spline, SuperGLM
from superglm.editor import EditorSession
from superglm.editor.errors import EditorValueError
from superglm.editor.notebook import MODULE_PREFIX, app_bundle, pack_modules, page_html
from superglm.editor.server import create_editor_app


@pytest.fixture(scope="module")
def session_model():
    rng = np.random.default_rng(20261007)
    X = pd.DataFrame({"x": rng.uniform(0.0, 10.0, 300), "c": rng.choice(list("abcd"), 300)})
    y = rng.poisson(np.exp(0.2 + 0.1 * np.sin(X["x"].to_numpy()))).astype(np.float64)
    features = {"x": Spline(n_knots=6), "c": Categorical(base="first")}
    return SuperGLM(family="poisson", selection_penalty=0.0, features=features).fit(X, y)


class _QueueView:
    model_id = "test-view"

    def __init__(self):
        self.outbox: queue.Queue = queue.Queue()
        # Change notices can overtake a reply (a job may publish before its
        # start request is answered), so they queue apart, as a page reads
        # them apart.
        self.notices: queue.Queue = queue.Queue()
        self.handler = None
        self.closed = False

    def on_msg(self, handler):
        self.handler = handler

    def send(self, content, buffers=None):
        if content.get("type") == notebook.CHANGED:
            self.notices.put(content)
            return
        self.outbox.put((content, [bytes(buffer) for buffer in buffers or []]))

    def close(self):
        self.closed = True


@pytest.fixture
def notebook_widget(session_model, monkeypatch):
    pytest.importorskip("anywidget")
    view = _QueueView()
    transport = widget_module.NotebookTransport
    monkeypatch.setattr(
        widget_module, "NotebookTransport", lambda app, token: transport(app, token, view=view)
    )
    widget = EditorSession.from_model(session_model).widget(mode="notebook")
    try:
        yield widget, view
    finally:
        widget.close()


def _request(view, request_id, method, url, payload=None, headers=None):
    view.handler(
        view,
        {
            "type": notebook.REQUEST,
            "id": request_id,
            "method": method,
            "url": url,
            "headers": headers or [["content-type", "application/json"]],
            "body": None if payload is None else json.dumps(payload),
        },
        [],
    )


def _reply(view, timeout=30.0):
    """One whole reply: its parts in order, as the page assembles them."""
    content, buffers = view.outbox.get(timeout=timeout)
    parts = {content["part"]: buffers[0]}
    while len(parts) < content["parts"]:
        more, buffers = view.outbox.get(timeout=timeout)
        assert more["id"] == content["id"]
        parts[more["part"]] = buffers[0]
    return content, b"".join(parts[index] for index in range(content["parts"]))


# -- The page bundle ---------------------------------------------------------


def test_the_bundle_lists_every_module_after_the_modules_it_imports():
    bundle = app_bundle()
    paths = [module["path"] for module in bundle["modules"]]
    assert paths[-1] == bundle["entry"] == "main.js"
    assert len(paths) == len(set(paths))
    for index, module in enumerate(bundle["modules"]):
        # Any relative module string left in code, whatever form imports it,
        # cannot resolve against a blob URL; JSDoc type imports are comments.
        code = re.sub(r"/\*.*?\*/", "", module["source"], flags=re.S)
        code = re.sub(r"(?m)^\s*//.*$", "", code)
        assert not re.search(r"""["']\.\.?/[^"'\s]*\.js["']""", code), module["path"]
        for target in re.findall(rf"""["']{MODULE_PREFIX}([^"']+)["']""", module["source"]):
            assert target in paths[:index], (module["path"], target)


def test_the_page_carries_its_styles_inline_and_runs_no_script_of_its_own():
    html = app_bundle()["html"]
    assert "<script" not in html.lower()
    assert html.count(notebook.PAGE_POLICY) == 1
    assert "/assets/" not in html
    index = notebook.read_app_asset("index.html").decode()
    for sheet in re.findall(r'href="/assets/([^"]+)"', index):
        assert f'<style data-asset="{sheet}">' in html
    # The bundle travels in the widget's state, one message under Databricks'
    # 5 MB limit for a widget message.
    assert len(json.dumps(app_bundle())) < 2 * 2**20


def test_the_module_marker_never_occurs_in_the_app_itself():
    for module in app_bundle()["modules"]:
        assert MODULE_PREFIX not in notebook.read_app_asset(module["path"]).decode()


def test_packing_rewrites_every_relative_import_form():
    modules = pack_modules(
        {
            "main.js": (
                'import { a } from "./a.js";\n'
                "import './b/side.js';\n"
                'export * from "./b/c.js";\n'
                "// import('./not-a-module.js') in a type comment stays\n"
            ),
            "a.js": 'export { c } from "./b/c.js";\nexport const a = 1;\n',
            "b/side.js": "globalThis.side = true;\n",
            "b/c.js": "export const c = 2;\n",
        }
    )
    paths = [module["path"] for module in modules]
    assert paths.index("b/c.js") < paths.index("a.js") < paths.index("main.js")
    main = modules[-1]["source"]
    assert f'from "{MODULE_PREFIX}a.js"' in main
    assert f"import '{MODULE_PREFIX}b/side.js'" in main
    assert f'export * from "{MODULE_PREFIX}b/c.js"' in main
    assert "import('./not-a-module.js')" in main


def test_packing_refuses_an_import_cycle_by_naming_it():
    with pytest.raises(ValueError, match=r"cycle: main\.js -> a\.js -> main\.js"):
        pack_modules({"main.js": 'import "./a.js";', "a.js": 'import "./main.js";'})


def test_packing_names_the_importer_of_a_missing_module():
    with pytest.raises(ValueError, match=r"main\.js imports '\./gone\.js'"):
        pack_modules({"main.js": 'import "./gone.js";'})


def test_a_page_without_one_head_is_refused_rather_than_left_without_its_policy():
    with pytest.raises(ValueError, match="one <head>"):
        page_html("<body></body>", lambda path: "")


def test_page_html_inlines_each_stylesheet_in_place():
    html = page_html(
        '<head><script>theme()</script><link rel="stylesheet" href="/assets/a.css"></head>'
        '<body><script type="module" src="/assets/main.js"></script>'
        "<SCRIPT>upper()</SCRIPT ><script>spaced()</script\n></body>",
        lambda path: f"/* {path} */",
    )
    assert html == (
        f'<head>{notebook.PAGE_POLICY}<style data-asset="a.css">\n/* a.css */</style></head>'
        "<body></body>"
    )


# -- The transport -----------------------------------------------------------


def test_a_request_gets_the_local_servers_answer(notebook_widget):
    widget, view = notebook_widget
    _request(view, 1, "GET", "/state")
    content, body = _reply(view)
    assert content["id"] == 1 and content["status"] == 200
    state = json.loads(body)
    assert state["selected_term"] == widget.selected_term
    assert set(state["terms"]) == set(widget.session.terms)

    _request(view, 2, "POST", "/op", {"operation": "nope"})
    content, body = _reply(view)
    assert content["status"] == 400
    assert json.loads(body) == {"error": "Unknown editor operation: 'nope'"}

    _request(view, 3, "GET", "/nope")
    content, body = _reply(view)
    assert (content["status"], json.loads(body)) == (404, {"error": "not found"})


def test_the_page_never_supplies_the_token(notebook_widget):
    """Python adds the widget's token itself, replacing any the page sends."""
    _widget, view = notebook_widget
    _request(view, 4, "GET", "/state", headers=[["X-SuperGLM-Editor-Token", "forged"]])
    content, _body = _reply(view)
    assert content["status"] == 200


def test_a_large_reply_arrives_in_parts_that_rebuild_it_exactly(notebook_widget, monkeypatch):
    widget, view = notebook_widget
    monkeypatch.setattr(notebook, "MESSAGE_PART_BYTES", 1000)
    _request(view, 5, "GET", "/download_export?format=joblib&filename=m.joblib")
    content, body = _reply(view)
    assert content["status"] == 200 and content["parts"] == -(-len(body) // 1000) > 1
    assert dict(content["headers"])["content-disposition"].startswith("attachment;")
    rebuilt = joblib.load(io.BytesIO(body))
    direct = joblib.load(io.BytesIO(widget._export_bytes("joblib", "m.joblib").data))
    np.testing.assert_array_equal(rebuilt.result.beta, direct.result.beta)


def test_an_empty_reply_is_one_empty_part(notebook_widget):
    _widget, view = notebook_widget
    _request(view, 6, "GET", "/favicon.ico")
    content, body = _reply(view)
    assert (content["status"], content["parts"], body) == (204, 1, b"")


def test_a_waiting_request_never_holds_the_others(notebook_widget, monkeypatch):
    """A job-status wait blocks only its own request, as on the local server."""
    widget, view = notebook_widget
    release = threading.Event()

    def waiting_status(job_id, wait=False):
        release.wait(timeout=30.0)
        return {"job_id": job_id, "status": "done"}

    monkeypatch.setattr(widget, "_job_status", waiting_status)
    _request(view, 7, "POST", "/job_status", {"job_id": "j", "wait": True})
    _request(view, 8, "GET", "/state")
    first, _body = _reply(view)
    release.set()
    second, _body = _reply(view)
    assert (first["id"], second["id"]) == (8, 7)


def test_a_change_tells_every_view_and_a_read_does_not(notebook_widget):
    _widget, view = notebook_widget
    _request(view, 20, "POST", "/op", {"operation": "select_all", "term": "x"})
    content, _body = _reply(view)
    assert content["status"] == 200
    assert view.notices.get(timeout=30.0) == {"type": notebook.CHANGED, "origin": 20}

    _request(view, 21, "POST", "/metrics", {"metric": "deviance"})
    _request(view, 22, "GET", "/state")
    _request(view, 23, "POST", "/op", {"operation": "nope"})
    replies = [_reply(view)[0] for _ in range(3)]
    assert sorted(reply["id"] for reply in replies) == [21, 22, 23]
    with pytest.raises(queue.Empty):
        view.notices.get(timeout=0.5)


def test_a_finished_profile_tells_every_view_and_a_failed_one_does_not(
    notebook_widget, monkeypatch
):
    """A profile job replaces the in-force model; its start request is only a read."""
    widget, view = notebook_widget
    monkeypatch.setattr(widget, "_profile_distribution", lambda *a, **k: {"profile_trace": []})
    _request(view, 30, "POST", "/profile_distribution/start", {"parameter": "tweedie_p"})
    assert _reply(view)[0]["status"] == 200
    assert view.notices.get(timeout=60.0) == {"type": notebook.CHANGED, "origin": None}

    def refuse(*_args, **_kwargs):
        raise EditorValueError("no profile for this family")

    monkeypatch.setattr(widget, "_profile_distribution", refuse)
    _request(view, 31, "POST", "/profile_distribution/start", {"parameter": "tweedie_p"})
    assert _reply(view)[0]["status"] == 200
    _request(view, 32, "GET", "/profile_distribution/status/2?wait=true")
    content, body = _reply(view)
    assert json.loads(body)["status"] == "error"
    with pytest.raises(queue.Empty):
        view.notices.get(timeout=0.5)


def test_published_cv_and_final_fit_tell_every_view(monkeypatch):
    pytest.importorskip("anywidget")
    from sklearn.model_selection import KFold

    from superglm import cross_validate

    rng = np.random.default_rng(20261008)
    X = pd.DataFrame({"x": rng.uniform(0.0, 10.0, 240), "c": rng.choice(list("abc"), 240)})
    y = rng.poisson(np.exp(0.1 * np.sin(X["x"].to_numpy()))).astype(np.float64)

    def model():
        features = {"x": Spline(n_knots=5), "c": Categorical(base="first")}
        return SuperGLM(family="poisson", selection_penalty=0.0, features=features)

    supplied = cross_validate(
        model(), X.iloc[:180], y[:180], cv=KFold(2, shuffle=True, random_state=0)
    )
    session = EditorSession.from_model(
        model().fit(X.iloc[:180], y[:180]),
        train_data=(X.iloc[:180], y[:180]),
        validation_data=(X.iloc[180:], y[180:]),
        cv=supplied,
    )
    view = _QueueView()
    transport = widget_module.NotebookTransport
    monkeypatch.setattr(
        widget_module, "NotebookTransport", lambda app, token: transport(app, token, view=view)
    )
    widget = session.widget(mode="notebook")
    try:
        for request_id, kind in ((40, "cv"), (41, "final_fit")):
            _request(view, request_id, "POST", "/job_start", {"kind": kind})
            assert _reply(view)[0]["status"] == 200
            assert view.notices.get(timeout=120.0) == {"type": notebook.CHANGED, "origin": None}
        assert widget._cv_run is not None and widget._final_fit is not None
    finally:
        widget.close()


def test_every_post_route_is_a_change_or_a_read(notebook_widget):
    """A new route must be placed: a read listed as a change would make views refresh each other."""
    widget, _view = notebook_widget
    posts = {
        route.path
        for route in create_editor_app(widget).routes
        if "POST" in getattr(route, "methods", set())
    }
    assert not notebook.CHANGE_ROUTES & notebook.READ_ROUTES
    assert posts == notebook.CHANGE_ROUTES | notebook.READ_ROUTES


def test_messages_that_are_not_requests_are_ignored(notebook_widget):
    _widget, view = notebook_widget
    view.handler(view, {"type": "something else", "id": 9}, [])
    view.handler(view, ["not", "a", "request"], [])
    _request(view, 10, "GET", "/health")
    content, _body = _reply(view)
    assert content["id"] == 10 and view.outbox.empty()


def test_closing_the_widget_closes_its_view_and_stops_answering(session_model, monkeypatch):
    pytest.importorskip("anywidget")
    view = _QueueView()
    transport = widget_module.NotebookTransport
    monkeypatch.setattr(
        widget_module, "NotebookTransport", lambda app, token: transport(app, token, view=view)
    )
    widget = EditorSession.from_model(session_model).widget(mode="notebook")
    widget.close()
    assert view.closed
    _request(view, 11, "GET", "/health")
    with pytest.raises(queue.Empty):
        view.outbox.get(timeout=0.5)


def test_closing_mid_request_finishes_it_and_releases_its_threads(notebook_widget, monkeypatch):
    """A request still running at close ends with the loop, not after it.

    Stopping the loop at once left the request unfinished and its AnyIO
    worker thread waiting for a loop that had closed.
    """
    widget, view = notebook_widget
    transport = widget._notebook
    entered, release = threading.Event(), threading.Event()

    def waiting_status(job_id, wait=False):
        entered.set()
        release.wait(timeout=30.0)
        return {"job_id": job_id, "status": "done"}

    monkeypatch.setattr(widget, "_job_status", waiting_status)
    before = set(threading.enumerate())
    _request(view, 12, "POST", "/job_status", {"job_id": "j", "wait": True})
    assert entered.wait(timeout=30.0)
    started = set(threading.enumerate()) - before
    workers = [thread for thread in started if "AnyIO worker" in thread.name]
    assert workers

    widget.close()  # returns while the handler still waits
    release.set()
    transport._thread.join(timeout=30.0)
    assert not transport._thread.is_alive() and transport._loop.is_closed()
    for worker in workers:
        worker.join(timeout=30.0)
        assert not worker.is_alive(), worker.name


# -- Display and mode ----------------------------------------------------------


def test_the_real_view_answers_through_anywidget_and_displays_as_a_widget(session_model):
    pytest.importorskip("anywidget")
    formatters = pytest.importorskip("IPython.core.formatters")
    widget = EditorSession.from_model(session_model).widget(mode="notebook")
    try:
        view = widget._notebook.view
        sent = queue.Queue()
        view.send = lambda content, buffers=None: sent.put((content, buffers))
        # anywidget's own message dispatch, as a kernel delivers a page's message.
        view._handle_custom_msg(
            {"type": notebook.REQUEST, "id": 1, "method": "GET", "url": "/health", "headers": []},
            [],
        )
        content, buffers = sent.get(timeout=30.0)
        assert (content["status"], json.loads(bytes(buffers[0]))) == (200, {"ok": True})
        assert view.bundle["entry"] == "main.js"

        data, _metadata = formatters.DisplayFormatter().format(widget)
        assert data["application/vnd.jupyter.widget-view+json"]["model_id"] == view.model_id
        assert "text/html" not in data
    finally:
        widget.close()


def test_a_closed_notebook_editor_displays_as_closed_text(session_model):
    pytest.importorskip("anywidget")
    formatters = pytest.importorskip("IPython.core.formatters")
    widget = EditorSession.from_model(session_model).widget(mode="notebook")
    widget.close()
    # The closed anywidget view has no comm, so its model id is gone.
    assert widget._repr_mimebundle_() == {"text/plain": "SuperGLM editor (closed)"}
    data, _metadata = formatters.DisplayFormatter().format(widget)
    assert data == {"text/plain": "SuperGLM editor (closed)"}


def test_server_mode_still_displays_its_local_page(session_model):
    formatters = pytest.importorskip("IPython.core.formatters")
    widget = EditorSession.from_model(session_model).widget(mode="server")
    try:
        data, _metadata = formatters.DisplayFormatter().format(widget)
        assert widget.app_url in data["text/html"]
        assert "application/vnd.jupyter.widget-view+json" not in data
    finally:
        widget.close()


def test_the_mode_defaults_to_notebook_on_databricks_only(monkeypatch):
    monkeypatch.delenv("DATABRICKS_RUNTIME_VERSION", raising=False)
    assert widget_module._display_mode(None) == "server"
    monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "15.4")
    pytest.importorskip("anywidget")
    assert widget_module._display_mode(None) == "notebook"
    assert widget_module._display_mode("server") == "server"
    with pytest.raises(EditorValueError, match="mode must be"):
        widget_module._display_mode("iframe")


def test_notebook_mode_without_anywidget_says_how_to_install_it(session_model, monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "anywidget", None)
    notebook.notebook_view_class.cache_clear()
    before = set(threading.enumerate())
    try:
        with pytest.raises(ImportError, match=r"pip install 'superglm\[notebook\]'"):
            EditorSession.from_model(session_model).widget(mode="notebook")
        assert not set(threading.enumerate()) - before
    finally:
        notebook.notebook_view_class.cache_clear()


def test_databricks_without_anywidget_keeps_the_local_server_and_says_why(
    session_model, monkeypatch
):
    monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "15.4")
    monkeypatch.setitem(__import__("sys").modules, "anywidget", None)
    notebook.notebook_view_class.cache_clear()
    try:
        with pytest.warns(UserWarning, match=r"%pip install 'superglm\[notebook\]'"):
            widget = EditorSession.from_model(session_model).widget()
        try:
            assert widget.mode == "server" and widget.app_url is not None
        finally:
            widget.close()
    finally:
        notebook.notebook_view_class.cache_clear()


def test_without_its_own_bundle_method_the_view_displays_by_model_id():
    """ipywidgets 7 has no _repr_mimebundle_; its view MIME type carries the model id."""

    class SevenView(_QueueView):
        model_id = "seven"

    transport = notebook.NotebookTransport(object(), "token", view=SevenView())
    try:
        bundle = transport.mimebundle()
    finally:
        transport.close()
    assert bundle["application/vnd.jupyter.widget-view+json"] == {
        "version_major": 2,
        "version_minor": 0,
        "model_id": "seven",
    }
