"""Structure files: a fitted model's structural decisions, read back and applied (spec 2026-10-03, phase 7c)."""

from __future__ import annotations

import copy
import json
import re

import numpy as np
import pandas as pd
import pytest

from superglm import (
    Categorical,
    OrderedCategorical,
    PolynomialRange,
    Spline,
    Structure,
    SuperGLM,
    collapse_levels,
    read_structure,
)
from superglm.structure import FORMAT, FeatureStructure, StructureError

BRANDS = ["B1", "B2", "B10", "B11", "B12", "B13", "B14"]
BANDS = ["0", "1", "2", "3", "4", "5", "6", "7"]


def _frame(seed: int = 20261003, n: int = 800, brands=BRANDS):
    rng = np.random.default_rng(seed)
    brand = rng.choice(brands, n)
    area = rng.choice(["A", "B", "C", "D"], n)
    age = rng.uniform(18.0, 80.0, n)
    band = rng.choice(BANDS, n)
    effects = dict(zip(BRANDS, [0.0, 0.1, 0.25, 0.22, -0.1, 0.05, 0.07], strict=True))
    y = (
        0.5
        + np.array([effects.get(b, 0.06) for b in brand])
        + 0.1 * (area == "C")
        + 0.2 * np.sin(age / 15.0)
        + 0.03 * band.astype(int)
        + rng.normal(0.0, 0.05, n)
    )
    return pd.DataFrame({"brand": brand, "area": area, "age": age, "band": band}), y


def _declared(features) -> SuperGLM:
    return SuperGLM(family="gaussian", selection_penalty=0.0, spline_penalty=0.1, features=features)


def _book():
    """A fitted book whose features carry a grouping, a reference, ranges and an unseen group."""
    X, y = _frame()
    grouping = collapse_levels(X["brand"], groups={"Other": ["B13", "B14"]})
    model = _declared(
        {
            "brand": Categorical(base="B2", grouping=grouping, unseen="Other"),
            "area": Categorical(base="first"),
            "age": Spline(
                kind="bs",
                n_knots=6,
                polynomial_ranges=[
                    PolynomialRange(30.0, 45.0, 1),
                    PolynomialRange(60.0, 70.0, 0, "kink"),
                ],
            ),
            "band": OrderedCategorical(
                order=BANDS,
                basis=Spline(
                    kind="bs", n_knots=4, polynomial_ranges=[PolynomialRange("5", "7", 0, "kink")]
                ),
            ),
        }
    )
    model.fit(X, y)
    return model, X, y


def _payload(**features):
    return {"format": FORMAT, "superglm_version": "0", "features": features}


def _categorical(**overrides):
    entry = {
        "kind": "categorical",
        "levels": ["A", "B", "C", "D"],
        "groups": {"CD": ["C", "D"]},
        "reference": "A",
        "unseen": "CD",
    }
    entry.update(overrides)
    return entry


# -- The file -----------------------------------------------------------------


def test_the_file_holds_every_structural_decision_and_no_coefficients():
    model, _, _ = _book()
    payload = json.loads(Structure.from_model(model).to_json())
    assert sorted(payload) == ["features", "format", "superglm_version"]
    assert payload["format"] == "superglm.structure.v1"
    assert payload["features"] == {
        "area": {
            "groups": {},
            "kind": "categorical",
            "levels": ["A", "B", "C", "D"],
            "reference": "A",
            "unseen": "error",
        },
        "age": {
            "kind": "spline",
            "ranges": [
                {"degree": 1, "hi": 45.0, "join": "tangent", "lo": 30.0},
                {"degree": 0, "hi": 70.0, "join": "kink", "lo": 60.0},
            ],
        },
        "band": {
            "groups": {},
            "kind": "ordered",
            "levels": BANDS,
            "ranges": [{"degree": 0, "hi": "7", "join": "kink", "lo": "5"}],
            "reference": model._specs["band"]._base_level,
            "unseen": "error",
        },
        "brand": {
            "groups": {"Other": ["B13", "B14"]},
            "kind": "categorical",
            "levels": ["B1", "B10", "B11", "B12", "B13", "B14", "B2"],
            "reference": "B2",
            "unseen": "Other",
        },
    }


def test_export_is_byte_stable_and_reads_back_to_the_same_bytes(tmp_path):
    model, _, _ = _book()
    first, second = tmp_path / "one.json", tmp_path / "two.json"
    text = Structure.from_model(model).to_json(first)
    Structure.from_model(model).to_json(second)
    assert first.read_bytes() == second.read_bytes() == text.encode("utf-8")
    # Sorted keys, two-space indent, one trailing newline: the file diffs cleanly.
    assert text == json.dumps(json.loads(text), sort_keys=True, indent=2) + "\n"
    assert read_structure(first).to_json() == text
    assert Structure.from_json(text).to_json() == text
    assert read_structure(json.loads(text)).to_json() == text


def test_native_integer_levels_survive_the_round_trip():
    rng = np.random.default_rng(20261006)
    code = rng.choice([1, 2, 3, 10], 400)
    power = rng.choice([4, 5, 6, 7], 400)
    rank = rng.choice([1, 2, 3, 4, 5], 400)
    y = 0.5 + 0.1 * (code == 2) + 0.05 * power + 0.02 * rank + rng.normal(0.0, 0.05, 400)
    X = pd.DataFrame({"code": code, "power": power, "rank": rank})
    model = _declared(
        {
            "code": Categorical(base=3),
            "power": Categorical(
                base="first",
                levels=[4, 5, 6, 7],
                grouping=collapse_levels(power, groups={"6+": [6, 7]}),
            ),
            "rank": OrderedCategorical(order=[1, 2, 3, 4, 5], base=2),
        }
    )
    model.fit(X, y)
    text = Structure.from_model(model).to_json()
    assert '"levels": [\n        1,\n        2,\n        3,\n        10\n      ]' in text
    features = read_structure(json.loads(text)).features
    assert features["code"].levels == [1, 2, 3, 10]
    assert type(features["code"].reference) is int and features["code"].reference == 3
    assert features["power"].levels == [4, 5, 6, 7]
    assert features["power"].groups == {"6+": [6, 7]}
    assert features["rank"].levels == [1, 2, 3, 4, 5]
    assert type(features["rank"].reference) is int
    assert all(type(level) is int for entry in features.values() for level in entry.levels)


def test_a_grouping_without_declared_levels_takes_native_types_from_x():
    rng = np.random.default_rng(20261007)
    power = rng.choice([4, 5, 6, 7], 300)
    y = 0.5 + 0.05 * power + rng.normal(0.0, 0.05, 300)
    X = pd.DataFrame({"power": power})
    model = _declared(
        {"power": Categorical(base="first", grouping=collapse_levels(power, groups={"6+": [6, 7]}))}
    )
    model.fit(X, y)
    # A grouping matches levels as text, so the fitted model alone keeps only text.
    assert Structure.from_model(model).features["power"].levels == ["4", "5", "6", "7"]
    entry = Structure.from_model(model, X=X).features["power"]
    assert entry.levels == [4, 5, 6, 7]
    assert entry.groups == {"6+": [6, 7]}
    assert entry.reference == 4 and type(entry.reference) is int


def test_an_unfitted_model_has_no_structure_to_export():
    model = _declared({"area": Categorical(base="first")})
    with pytest.raises(
        StructureError,
        match=re.escape(
            "Structure.from_model needs a fitted model: 'area' has no fitted levels yet; "
            "fit the model first."
        ),
    ):
        Structure.from_model(model)


def test_a_structure_built_in_python_is_checked_like_a_file():
    with pytest.raises(StructureError, match="not one of its levels"):
        Structure(
            features={
                "area": FeatureStructure(
                    kind="categorical", levels=["A", "B"], groups={"AZ": ["A", "Z"]}, reference="AZ"
                )
            }
        )


# -- Read refusals (S4): one fixed sentence each -------------------------------

READ_REFUSALS = {
    "unknown format": (
        {"format": "superglm.structure.v2", "superglm_version": "0", "features": {}},
        "Unknown structure format 'superglm.structure.v2': this version of superglm reads "
        "'superglm.structure.v1' files; export the structure again with it.",
    ),
    "no format": (
        {"features": {}},
        "Unknown structure format None: this version of superglm reads "
        "'superglm.structure.v1' files; export the structure again with it.",
    ),
    "member outside the levels": (
        _payload(area=_categorical(groups={"CD": ["C", "D", "Z"]})),
        "Group 'CD' of 'area' holds 'Z', which is not one of its levels; add it to the "
        "levels or take it out of the group.",
    ),
    "member in two groups": (
        _payload(area=_categorical(groups={"CD": ["C", "D"], "BC": ["B", "C"]}, unseen="CD")),
        "Level 'C' of 'area' is in more than one group; keep it in one.",
    ),
    "group named like a level outside it": (
        _payload(area=_categorical(groups={"A": ["C", "D"]}, reference="B", unseen="error")),
        "Group 'A' of 'area' has the name of a level outside it; rename the group.",
    ),
    "reference not a level": (
        _payload(area=_categorical(reference="Z")),
        "The reference 'Z' of 'area' is not a level or group of the term; choose an "
        "ungrouped level or a group.",
    ),
    "reference inside a group": (
        _payload(area=_categorical(reference="C")),
        "The reference 'C' of 'area' is not a level or group of the term; choose an "
        "ungrouped level or a group.",
    ),
    "unseen group missing": (
        _payload(area=_categorical(unseen="Rest")),
        "New levels of 'area' go to 'Rest', which is not a group of the term; name one "
        "of its groups, or use 'error' or 'base'.",
    ),
    "unseen group without groups": (
        _payload(area=_categorical(groups={}, unseen="A")),
        "New levels of 'area' go to 'A', which is not a group of the term; name one "
        "of its groups, or use 'error' or 'base'.",
    ),
    "ordered unseen": (
        _payload(band=_categorical(kind="ordered", groups={}, unseen="base", ranges=[])),
        "'band' is an ordered term, which refuses new levels; set its unseen to 'error'.",
    ),
    "range degree": (
        _payload(
            age={"kind": "spline", "ranges": [{"lo": 1.0, "hi": 2.0, "degree": 5, "join": "kink"}]}
        ),
        "The spline of 'age' refuses the range 1–2; change or remove that range.",
    ),
    "range backwards": (
        _payload(
            age={"kind": "spline", "ranges": [{"lo": 3.0, "hi": 2.0, "degree": 1, "join": "kink"}]}
        ),
        "The spline of 'age' refuses the Line range 3–2; change or remove that range.",
    ),
    "malformed kind": (
        _payload(age={"kind": "piecewise", "ranges": []}),
        "The structure entry for 'age' has a malformed 'kind'; export the structure again.",
    ),
    "malformed levels": (
        _payload(area=_categorical(levels="ABCD")),
        "The structure entry for 'area' has a malformed 'levels'; export the structure again.",
    ),
    "duplicate level text": (
        _payload(code=_categorical(levels=[1, "1", 2], groups={}, reference=2, unseen="error")),
        "The structure entry for 'code' has a malformed 'levels'; export the structure again.",
    ),
    "unknown key": (
        _payload(area={**_categorical(), "group": {}}),
        "The structure entry for 'area' has a malformed 'group'; export the structure again.",
    ),
    "ranges on a categorical": (
        _payload(
            area={**_categorical(), "ranges": [{"lo": 1, "hi": 2, "degree": 0, "join": "kink"}]}
        ),
        "The structure entry for 'area' has a malformed 'ranges'; export the structure again.",
    ),
    "features not a mapping": (
        {"format": FORMAT, "superglm_version": "0", "features": []},
        "The structure's 'features' field is malformed; export the structure again.",
    ),
}


@pytest.mark.parametrize(
    ("payload", "sentence"), list(READ_REFUSALS.values()), ids=list(READ_REFUSALS)
)
def test_a_bad_file_is_refused_with_its_fixed_sentence(payload, sentence, tmp_path):
    with pytest.raises(StructureError) as refused:
        read_structure(payload)
    assert str(refused.value) == sentence
    path = tmp_path / "structure.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(StructureError, match=re.escape(sentence)):
        read_structure(path)


def test_text_that_is_not_json_is_refused_and_points_at_read_structure(tmp_path):
    sentence = "The structure is not valid JSON; to read a file, pass its path to read_structure."
    for text in ("structure.json", '{"format": NaN}', b"\xff\xfe"):
        with pytest.raises(StructureError, match=re.escape(sentence)):
            Structure.from_json(text)
    path = tmp_path / "broken.json"
    path.write_text("{", encoding="utf-8")
    with pytest.raises(StructureError, match=re.escape(sentence)):
        read_structure(path)


def test_a_refusal_is_a_value_error():
    assert issubclass(StructureError, ValueError)
    payload = copy.deepcopy(READ_REFUSALS["unknown format"][0])
    with pytest.raises(ValueError):
        read_structure(payload)
