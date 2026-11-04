"""State replies carry only the terms a client lacks, and a page's view needs no reply."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import numpy as np
import pandas as pd
import pytest

from superglm import Spline, SuperGLM
from superglm.editor import EditorSession
from superglm.editor.term_delta import KNOWN_STATE_HEADER, TermDeltas


@pytest.fixture(scope="module")
def two_term_model():
    rng = np.random.default_rng(20261104)
    frame = pd.DataFrame({"x": rng.uniform(-1, 1, 200), "z": rng.uniform(-1, 1, 200)})
    y = 1.0 + 0.4 * frame["x"] + 0.2 * np.sin(3 * frame["z"]) + rng.normal(0, 0.05, 200)
    features = {"x": Spline(n_knots=6), "z": Spline(n_knots=6)}
    return SuperGLM(family="gaussian", selection_penalty=0.0, features=features).fit(frame, y)


@pytest.fixture
def widget(two_term_model):
    widget = EditorSession.from_model(two_term_model, terms=["x", "z"]).widget()
    try:
        yield widget
    finally:
        widget.close()


def call(widget, path, payload=None, known=None):
    headers = {"X-SuperGLM-Editor-Token": widget._token, "Content-Type": "application/json"}
    if known is not None:
        headers[KNOWN_STATE_HEADER] = str(known)
    request = urllib.request.Request(
        f"{widget.url}{path}",
        data=None if payload is None else json.dumps(payload).encode(),
        headers=headers,
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def test_a_reply_holds_only_the_terms_changed_since_the_one_its_client_holds(widget):
    whole = call(widget, "/select", {"term": "x", "indices": [1, 2]}, known="none")
    assert set(whole["terms"]) == {"x", "z"}

    reply = call(
        widget,
        "/op",
        {"operation": "shift_up", "term": "x"},
        known=whole["state_generation"],
    )

    assert set(reply["terms"]) == {"x"}
    assert list(reply["term_versions"]) == ["x", "z"]
    assert reply["term_versions"]["z"] == whole["term_versions"]["z"]
    assert reply["term_versions"]["x"] != whole["term_versions"]["x"]


def test_a_reply_merged_with_the_terms_its_client_holds_is_the_whole_state(widget):
    held = call(widget, "/select", {"term": "x", "indices": [1, 2]}, known="none")

    reply = call(
        widget,
        "/op",
        {"operation": "shift_up", "term": "x"},
        known=held["state_generation"],
    )

    plain = call(widget, "/state")
    assert {**held["terms"], **reply["terms"]} == plain["terms"]


def test_a_term_that_changed_because_another_did_is_sent_too():
    deltas = TermDeltas()
    first = {"state_generation": 1, "terms": {"edited": {"y": [1.0]}, "anchored": {"y": [2.0]}}}
    deltas.reply(first, "none")

    second = {"state_generation": 2, "terms": {"edited": {"y": [1.5]}, "anchored": {"y": [2.5]}}}
    assert set(deltas.reply(second, "1")["terms"]) == {"edited", "anchored"}


def test_a_state_inside_a_reply_is_cut_and_the_rest_of_the_reply_kept():
    deltas = TermDeltas()
    deltas.reply({"state_generation": 1, "terms": {"a": {"y": [1.0]}}}, "none")

    reply = deltas.reply(
        {
            "ok": True,
            "summary": "kept",
            "state": {"state_generation": 2, "terms": {"a": {"y": [1.0]}}},
        },
        "1",
    )

    assert reply["summary"] == "kept"
    assert reply["state"]["terms"] == {}
    assert list(reply["state"]["term_versions"]) == ["a"]


def test_a_reply_with_no_state_passes_through():
    deltas = TermDeltas()
    payload = {"ok": True}
    assert deltas.reply(payload, "1") is payload


@pytest.mark.parametrize("known", ["none", "999999", "not-a-number"])
def test_a_client_whose_reply_is_unknown_gets_every_term(widget, known):
    assert set(call(widget, "/state", known=known)["terms"]) == {"x", "z"}


def test_a_forgotten_reply_gets_every_term():
    deltas = TermDeltas()
    terms = {"a": {"y": [1.0]}}
    for generation in range(1, 70):
        deltas.reply({"state_generation": generation, "terms": terms}, "none")

    assert deltas.reply({"state_generation": 70, "terms": terms}, "69")["terms"] == {}
    assert set(deltas.reply({"state_generation": 71, "terms": terms}, "1")["terms"]) == {"a"}


def test_a_client_that_sends_no_header_gets_the_plain_state_after_one_that_did(widget):
    call(widget, "/state", known="none")

    plain = call(widget, "/state")

    assert set(plain["terms"]) == {"x", "z"}
    assert "term_versions" not in plain


def test_a_quiet_select_records_the_view_and_answers_with_no_state(widget):
    before = call(widget, "/state")["state_generation"]

    reply = call(widget, "/select", {"term": "z", "indices": [3, 1], "quiet": True})

    assert reply == {"ok": True}
    after = call(widget, "/state")
    assert after["state_generation"] == before + 1
    assert after["selected_term"] == "z"
    assert after["selection"]["z"] == [1, 3]


def test_an_edit_applies_to_the_view_a_quiet_select_recorded(widget):
    call(widget, "/select", {"term": "z", "indices": [2], "quiet": True})
    before = call(widget, "/state")["terms"]["z"]["y"]

    after = call(widget, "/op", {"operation": "shift_up"})["terms"]["z"]["y"]

    changed = [
        index for index, (old, new) in enumerate(zip(before, after, strict=True)) if old != new
    ]
    assert changed == [2]


def test_a_quiet_select_is_refused_as_a_plain_one_is(widget):
    with pytest.raises(urllib.error.HTTPError) as refused:
        call(widget, "/select", {"term": "x", "indices": [-1], "quiet": True})
    assert refused.value.code == 400
