"""The Table view and the Excel workbook of a SuperLSS session show the library's rating tables.

The factors a block holds are checked through the model: two rows that differ in one column,
priced by ``predict_parameters``, must differ by the ratio of that column's factors, which is
independent of the payload builder.
"""

from __future__ import annotations

import functools
import io
import json
import urllib.request

import numpy as np
import pandas as pd
import pytest
from openpyxl import load_workbook

from superglm import GammaLS, SuperLSS, cat, s, term
from superglm.editor.errors import EditorValueError
from superglm.export import lss_rating_tables
from superglm.features.piecewise import Piecewise
from tests._lss_editor_fixtures import (
    FitCase,
    gamma_case,
    lognormal_case,
    nb_case,
    open_case,
)

KNOTS = {"count": 5, "strategy": "uniform"}


@functools.cache
def refusing_case(both: bool = False) -> FitCase:
    """The gamma case with a piecewise term of ``power`` in its scale (and its mean if ``both``).

    The library has no exact table for a ``Piecewise`` term, so it leaves out every parameter
    holding one and tabulates the others.
    """
    base = gamma_case()
    family = GammaLS()
    piecewise = term("power", Piecewise(breaks=[8.0], base=8.0))
    mean = (s("age", k=6), piecewise) if both else (s("age", k=6), cat("region"))
    model = SuperLSS(
        family,
        family.mean(*mean),
        family.scale(s("age", k=5), cat("region"), piecewise),
    ).fit_reml(base.X, base.y, outer="efs+newton")
    return FitCase(model, base.X, base.y, None, None)


def _factor(block, value) -> float:
    """The factor a block gives a level, or a covariate value read from its piecewise cubic."""
    rows = block["rows"]
    if block["columns"] == [block["columns"][0], "Relativity", "Weight"]:
        return next(row[1] for row in rows if row[0] == value)
    for key, relativity, _weight, a, b, c, d in rows:
        lower, upper = (float(end) for end in key.strip("[])").split(", "))
        if not np.isfinite(lower) or not np.isfinite(upper):
            if lower <= value < upper:
                return relativity
            continue
        if lower <= value < upper:
            u = (value - lower) / (upper - lower)
            return float(np.exp(a + u * (b + u * (c + u * d))))
    raise AssertionError(f"no row of the block holds {value!r}")


def _priced(case, column, values, parameter, offsets=None) -> np.ndarray:
    """``parameter`` on the first training row with ``column`` set to each of ``values``."""
    rows = pd.concat([case.X.iloc[[0]]] * len(values), ignore_index=True)
    rows[column] = list(values)
    return case.model.predict_parameters(rows, offsets=offsets)[parameter].to_numpy()


@pytest.mark.parametrize(
    ("key", "column", "values"),
    [
        ("mean:region", "region", ("R0", "R3")),
        ("mean:age", "age", (30.0, 70.0)),
        ("scale:region", "region", ("R0", "R4")),
        ("scale:age", "age", (25.0, 65.0)),
    ],
)
def test_the_factors_of_a_gamma_term_are_the_models_ratios(key, column, values):
    case = gamma_case()
    session = open_case(case)
    widget = session.widget()
    try:
        block = widget._rating_table(key)
    finally:
        widget.close()

    parameter = key.split(":")[0]
    assert block["available"] is True
    priced = _priced(case, column, values, parameter)
    first, second = (_factor(block, value) for value in values)
    np.testing.assert_allclose(priced[1] / priced[0], second / first, rtol=1e-10)
    assert block["note"].startswith("Mean = " if parameter == "mean" else "CV = ")


def test_a_log_normal_sigma_table_multiplies_sigma_above_its_floor():
    case = lognormal_case()
    floor = case.model._require_fitted().layout.predictor("scale").link.floor
    session = open_case(case)
    widget = session.widget()
    try:
        block = widget._rating_table("scale:region")
    finally:
        widget.close()

    sigma = _priced(case, "region", ("R0", "R5"), "scale")
    ratio = (sigma[1] - floor) / (sigma[0] - floor)
    np.testing.assert_allclose(ratio, _factor(block, "R5") / _factor(block, "R0"), rtol=1e-10)
    assert block["note"].startswith("Sigma = floor + base x product of the relativities")
    assert f"the floor is {floor:g}" in block["note"]


def test_a_negative_binomial_table_multiplies_overdispersion():
    case = nb_case()
    session = open_case(case)
    widget = session.widget()
    try:
        block = widget._rating_table("theta:region")
    finally:
        widget.close()

    theta = _priced(case, "region", ("R0", "R2"), "theta", offsets={"mean": np.zeros(2)})
    overdispersion = 1.0 / theta
    ratio = overdispersion[1] / overdispersion[0]
    np.testing.assert_allclose(ratio, _factor(block, "R2") / _factor(block, "R0"), rtol=1e-10)
    assert block["note"].startswith("Overdispersion = ")


def test_the_route_answers_in_the_contract_shape():
    session = open_case(gamma_case())
    widget = session.widget()
    try:
        request = urllib.request.Request(
            f"{widget.url}/rating_table",
            data=json.dumps({"term": "scale:age"}).encode("utf-8"),
            method="POST",
            headers={
                "Content-Type": "application/json",
                "X-SuperGLM-Editor-Token": widget._token,
            },
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
    finally:
        widget.close()

    assert set(payload) == {
        "term",
        "available",
        "reason",
        "columns",
        "rows",
        "formats",
        "note",
        "model_revision",
    }
    assert payload["term"] == "scale:age" and payload["available"] is True
    assert payload["columns"][:3] == ["age", "Relativity", "Weight"]


def test_the_tools_are_no_longer_listed_as_off():
    session = open_case(gamma_case())
    widget = session.widget()
    try:
        assert {"rating_table", "export_xlsx"}.isdisjoint(widget._state()["unavailable"])
    finally:
        widget.close()


def test_the_payload_is_built_once_per_revision_and_rebuilt_after_an_edit(monkeypatch):
    # Mutation checks: building on every call fails the first count; reusing a payload across an
    # edit fails the second count and the changed factor.
    calls = []
    build = lss_rating_tables.build_lss_rating_table_payload

    def counting(*args, **kwargs):
        calls.append(1)
        return build(*args, **kwargs)

    monkeypatch.setattr(lss_rating_tables, "build_lss_rating_table_payload", counting)
    session = open_case(gamma_case())
    widget = session.widget()
    try:
        before = widget._rating_table("mean:region")
        for key in ("mean:age", "scale:age", "scale:region", "mean:region"):
            widget._rating_table(key)
        assert len(calls) == 1

        session.select_levels("mean:region", ["R1"])
        session.shift("mean:region", 0.2)
        after = widget._rating_table("mean:region")
        assert len(calls) == 2
    finally:
        widget.close()

    assert after["model_revision"] == session.model_revision != before["model_revision"]
    factors = {row[0]: row[1] for row in before["rows"]}
    moved = {row[0]: row[1] for row in after["rows"]}
    np.testing.assert_allclose(moved["R1"], factors["R1"] * np.exp(0.2), rtol=1e-9)
    assert {k: v for k, v in moved.items() if k != "R1"} == pytest.approx(
        {k: v for k, v in factors.items() if k != "R1"}, rel=1e-9
    )


def test_changes_waiting_for_refit_are_not_in_the_table_until_refit():
    session = open_case(gamma_case())
    widget = session.widget()
    try:
        before = widget._rating_table("mean:age")
        session.stage_structural("knots", "mean:age", KNOTS)
        assert len(session.pending) == 1
        assert widget._rating_table("mean:age") == before
        session.refit_pending()
        after = widget._rating_table("mean:age")
    finally:
        widget.close()

    assert after["model_revision"] != before["model_revision"]
    assert len(after["rows"]) != len(before["rows"])


def test_a_parameter_the_library_leaves_out_gives_its_sentence_for_each_of_its_terms():
    case = refusing_case()
    payload = case.model.rating_table_payload(case.X)
    sentence = payload.omitted["scale"]
    session = open_case(case)
    widget = session.widget()
    try:
        blocks = {key: widget._rating_table(key) for key in session.terms}
    finally:
        widget.close()

    assert "Piecewise" in sentence
    for key, block in blocks.items():
        if key.startswith("scale:"):
            assert (block["available"], block["reason"], block["rows"]) == (False, sentence, [])
        else:
            assert block["available"] is True and block["rows"]


@pytest.mark.parametrize(
    ("case", "sheets"),
    [(gamma_case, ["Mean", "CV", "Summary"]), (refusing_case, ["Mean", "Summary"])],
)
def test_the_workbook_has_a_sheet_for_each_parameter_with_a_table(case, sheets):
    session = open_case(case())
    widget = session.widget()
    try:
        data = widget._export_bytes("xlsx").data
    finally:
        widget.close()

    assert load_workbook(io.BytesIO(data)).sheetnames == sheets


def test_a_model_with_no_table_at_all_says_why_in_the_librarys_sentences():
    case = refusing_case(both=True)
    session = open_case(case)
    widget = session.widget()
    try:
        preview = widget._rating_table("mean:age")
        with pytest.raises(EditorValueError) as refused:
            widget._export_bytes("xlsx")
    finally:
        widget.close()

    for text in (preview["reason"], refused.value.public_message):
        assert "Parameter 'mean' was left out" in text
        assert "Parameter 'scale' was left out" in text
    assert preview["available"] is False and preview["rows"] == []
