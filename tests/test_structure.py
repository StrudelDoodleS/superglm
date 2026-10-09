"""Structure files: a fitted model's structural decisions, read back and applied (spec 2026-10-03, phase 7c)."""

from __future__ import annotations

import copy
import json
import re
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

from superglm import (
    BSplineSmooth,
    Categorical,
    OrderedCategorical,
    Piecewise,
    PolynomialRange,
    PSpline,
    Spline,
    Structure,
    SuperGLM,
    collapse_levels,
    read_structure,
)
from superglm import structure as structure_module
from superglm.editor import EditorSession
from superglm.editor.errors import EditorValueError
from superglm.features._spline_ranges import RangeError
from superglm.structure import FORMAT, FeatureStructure, StructureError

U = 2.0**-53

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
            # Model order: each level where its group sits among the fitted levels.
            "levels": ["B1", "B10", "B11", "B12", "B2", "B13", "B14"],
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


def _lss_model():
    from superglm import GaussianLS, SuperLSS, cat, s

    family = GaussianLS()
    return SuperLSS(
        family, family.location(s("age", kind="cr", k=8), cat("region")), family.scale(s("age"))
    )


@pytest.mark.parametrize(
    ("model", "kind"),
    [
        (lambda: None, "NoneType"),
        (lambda: object(), "object"),
        (_lss_model, "SuperLSS"),
    ],
    ids=["None", "object", "SuperLSS"],
)
def test_only_a_superglm_has_a_structure_to_export_or_take(model, kind):
    for method, call in (
        ("from_model", lambda: Structure.from_model(model())),
        ("apply", lambda: Structure(features={}).apply(model())),
        ("apply", lambda: _brand_structure().apply(model())),
    ):
        with pytest.raises(StructureError) as refused:
            call()
        assert str(refused.value) == (
            f"Structure.{method} takes a SuperGLM model, not a {kind}; structure files do not "
            "cover other models yet."
        )


@pytest.mark.parametrize(
    "features", [{}, {"age": Spline(n_knots=6)}], ids=["no features", "spline only"]
)
def test_an_unfitted_model_without_levels_has_no_structure_to_export(features):
    with pytest.raises(StructureError) as refused:
        Structure.from_model(_declared(features))
    assert str(refused.value) == "Structure.from_model needs a fitted model; fit the model first."


def test_levels_a_file_cannot_hold_are_refused_by_name_on_export():
    """A fitted term whose levels are not plain JSON scalars, or share a text, says why.

    The refusal names the level and what to do, not "export the structure
    again", which is the export that failed.
    """
    rng = np.random.default_rng(0)
    n = 300
    stamps = pd.to_datetime(["2020-01-01", "2021-01-01", "2022-01-01"])
    dated = _declared({"d": Categorical(base="first")}).fit(
        pd.DataFrame({"d": rng.choice(stamps, n)}), rng.normal(size=n)
    )
    with pytest.raises(StructureError) as refused:
        Structure.from_model(dated)
    first = dated._specs["d"]._levels[0]
    assert str(refused.value) == (
        f"{first!r} in 'd' cannot be written to a structure file, which holds text, numbers "
        "and booleans; give the term plain labels."
    )

    mixed = _declared({"c": Categorical(base="first")}).fit(
        pd.DataFrame({"c": np.array([1, "1", 2] * 100, dtype=object)}), rng.normal(size=n)
    )
    with pytest.raises(StructureError) as refused:
        Structure.from_model(mixed)
    assert str(refused.value) == (
        "Levels 1 and '1' of 'c' read as the same text, which is how a structure file tells "
        "levels apart; give the term distinct labels."
    )


def test_a_number_a_file_cannot_hold_as_itself_is_refused():
    """A Decimal, a Fraction or an extended float is refused, never a TypeError from json.

    json cannot write a Decimal, as a database NUMERIC column arrives. A
    Fraction or an extended float would be written as a float64 that need not
    equal it and reads as other text, so the file could not find the level
    again. Every way into a file is checked: from_model names the level, and an
    entry built in Python or read from a mapping is refused in its fixed sentence.
    """
    from decimal import Decimal
    from fractions import Fraction

    rng = np.random.default_rng(0)
    n = 300
    for name, column in (
        ("power", [Decimal("4"), Decimal("5.5"), Decimal("6")] * 100),
        ("share", [Fraction(1, 3), Fraction(3, 2)] * 150),
    ):
        model = _declared({name: Categorical(base="first")})
        model.fit(pd.DataFrame({name: np.array(column, dtype=object)}), rng.normal(size=n))
        with pytest.raises(StructureError) as refused:
            Structure.from_model(model)
        first = model._specs[name]._levels[0]
        assert str(refused.value) == (
            f"{first!r} in {name!r} cannot be written to a structure file, which holds text, "
            "numbers and booleans; give the term plain labels."
        )

    malformed = "The structure entry for 'c' has a malformed {!r}; export the structure again."
    payload = json.loads(
        Structure(features={"c": FeatureStructure("categorical", [1, 2], {}, 1)}).to_json()
    )
    for number in (Decimal("1.5"), Fraction(3, 2), np.longdouble(1.5)):
        for field, entry in (
            ("levels", FeatureStructure("categorical", [number, 2], {}, 2)),
            ("reference", FeatureStructure("categorical", [1.5, 2], {}, number)),
        ):
            with pytest.raises(StructureError) as refused:
                Structure(features={"c": entry})
            assert str(refused.value) == malformed.format(field)
        payload["features"]["c"]["levels"] = [number, 2]
        with pytest.raises(StructureError) as refused:
            Structure.from_json(payload)
        assert str(refused.value) == malformed.format("levels")
        spline = FeatureStructure("spline", ranges=[PolynomialRange(number, 2.0, 1)])
        with pytest.raises(StructureError) as refused:
            Structure(features={"c": spline})
        assert str(refused.value) == (
            f"The spline of 'c' refuses the Line range {number}–2; change or remove that range."
        )


def test_importing_superglm_loads_no_pydantic():
    """The structure file is checked by plain Python, so importing superglm stays light."""
    script = (
        "import sys, superglm, superglm.structure\n"
        "loaded = sorted(m for m in sys.modules if m.split('.')[0] in ('pydantic', 'pydantic_core'))\n"
        "assert not loaded, loaded\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], check=False, capture_output=True, text=True, timeout=120
    )
    assert completed.returncode == 0, completed.stderr


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

_HUGE = 10**400

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
    # from_json takes any Mapping, and a key that is not text cannot be sorted beside text.
    "range with a key that is not text": (
        _payload(
            age={
                "kind": "spline",
                "ranges": [{"lo": 1.0, "hi": 2.0, "degree": 1, "join": "kink", 0: "extra"}],
            }
        ),
        "The structure entry for 'age' has a malformed 'ranges'; export the structure again.",
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
    # JSON reads an integer of any length; one past float64's range is no level or edge.
    "integer past float range as a level": (
        _payload(area=_categorical(levels=["A", "B", "C", "D", _HUGE])),
        "The structure entry for 'area' has a malformed 'levels'; export the structure again.",
    ),
    "integer past float range as the reference": (
        _payload(area=_categorical(reference=_HUGE)),
        "The structure entry for 'area' has a malformed 'reference'; export the structure again.",
    ),
    "integer past float range as a member": (
        _payload(area=_categorical(groups={"CD": ["C", "D", _HUGE]})),
        f"Group 'CD' of 'area' holds {_HUGE!r}, which is not one of its levels; add it to the "
        "levels or take it out of the group.",
    ),
    "integer past float range as a spline edge": (
        _payload(
            age={
                "kind": "spline",
                "ranges": [{"lo": _HUGE, "hi": 2.0, "degree": 1, "join": "kink"}],
            }
        ),
        "The spline of 'age' refuses the Line range inf–2; change or remove that range.",
    ),
    "integer past float range as an ordered edge": (
        _payload(
            band=_categorical(
                kind="ordered",
                levels=["0", "1"],
                groups={},
                reference="0",
                unseen="error",
                ranges=[{"lo": -_HUGE, "hi": "1", "degree": 0, "join": "kink"}],
            )
        ),
        "The spline of 'band' refuses the Flat range -inf–1; change or remove that range.",
    ),
}


def test_to_json_refuses_a_structure_changed_after_it_was_built():
    """Structure's features and entries are mutable; the checks ran only when it was built.

    to_json then wrote a reference no level or group holds (a file
    read_structure refuses), or failed on an entry that is not one with a
    raw error. It now checks the structure as it is, in the same sentences.
    """
    structure = read_structure(_payload(area=_categorical()))
    structure.features["area"].reference = "Z"
    with pytest.raises(StructureError) as moved:
        structure.to_json()
    with pytest.raises(StructureError) as read:
        Structure.from_json(json.dumps(_payload(area=_categorical(reference="Z"))))
    assert str(moved.value) == str(read.value)

    structure.features["area"] = "not an entry"
    with pytest.raises(StructureError) as replaced:
        structure.to_json()
    assert (
        str(replaced.value)
        == "The structure's 'features' field is malformed; export the structure again."
    )


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


# -- Apply (S3) ------------------------------------------------------------------


def _plain(**overrides) -> SuperGLM:
    """The book declared with no structural decisions, as a new year's model starts."""
    features = {
        "brand": Categorical(base="first"),
        "area": Categorical(base="first"),
        "age": Spline(n_knots=6),
        "band": OrderedCategorical(order=BANDS, basis=Spline(kind="bs", n_knots=4)),
    }
    features.update(overrides)
    return _declared(features)


def _brand_structure(unseen: str = "Other") -> Structure:
    return Structure(
        features={
            "brand": FeatureStructure(
                kind="categorical",
                levels=sorted(BRANDS),
                groups={"Other": ["B13", "B14"]},
                reference="B1",
                unseen=unseen,
            )
        }
    )


def _linear_predictor_bound(X, *models) -> float:
    """How far two evaluations of one fitted linear predictor can round apart.

    The same fit path on the same data gives the same estimates, so the
    predictions (identity link: mu = eta) differ only in how each model
    evaluates eta = b0 + sum_j x_j beta_j. Every product passes through at
    most p + 1 roundings, so each evaluation errs by at most
    gamma_(p+1) * max|x_j| * (|b0| + ||beta||_1) (Higham 2002, section 3.1).
    """
    bound = 0.0
    for model in models:
        largest = 1.0  # the intercept's column
        for name, spec in model._specs.items():
            design = np.asarray(spec.transform(X[name].to_numpy()), dtype=np.float64)
            largest = max(largest, float(np.max(np.abs(design))))
        terms = len(model.result.beta) + 1
        gamma = terms * U / (1.0 - terms * U)
        size = abs(model.result.intercept) + float(np.abs(model.result.beta).sum())
        bound += gamma * largest * size
    return bound


def test_round_trip_through_the_editor_rebuilds_the_in_force_model(tmp_path):
    X, y = _frame()
    grouping = collapse_levels(X["brand"], groups={"Other": ["B13", "B14"]})
    model = _plain(brand=Categorical(base="first", grouping=grouping))
    model.fit(X, y)
    session = EditorSession.from_model(model, terms=["brand", "age"])
    session.stage_structural("collapse", "brand", {"levels": ["B10", "B11"]})
    session.stage_structural("set_reference", "brand", {"level": "B2"})
    session.stage_structural("shape", "age", {"lo": 30.0, "hi": 45.0, "degree": 1})
    session.stage_structural("shape", "age", {"lo": 60.0, "hi": 70.0, "degree": 0, "join": "kink"})
    session.refit_pending(method="fit")
    session.set_unseen("brand", "Other")
    in_force = session.model
    path = tmp_path / "structure.json"
    session.export_structure(path)

    fresh = _plain()
    applied = read_structure(path).apply(fresh)
    assert applied is not fresh and applied._result is None and fresh._result is None
    applied.fit(X, y)

    ours, theirs = applied._specs["brand"], in_force._specs["brand"]
    assert ours._grouping.group_to_originals == theirs._grouping.group_to_originals
    assert ours._grouping.group_to_originals["B10+B11"] == ["B10", "B11"]
    # The same design columns in the same order, so the same fit.
    assert ours._levels == theirs._levels
    assert ours._base_level == theirs._base_level == "B2"
    assert ours.unseen == theirs.unseen == "Other"
    assert applied._specs["age"].polynomial_ranges == in_force._specs["age"].polynomial_ranges
    assert [(r.lo, r.hi, r.degree, r.join) for r in applied._specs["age"].polynomial_ranges] == [
        (30.0, 45.0, 1, "tangent"),
        (60.0, 70.0, 0, "kink"),
    ]
    gap = np.max(np.abs(applied.predict(X) - in_force.predict(X)))
    assert gap <= _linear_predictor_bound(X, applied, in_force)
    # Exporting the rebuilt model gives the same file.
    assert Structure.from_model(applied).to_json() == path.read_text(encoding="utf-8")


def test_a_categorical_column_round_trips_in_its_fitted_level_order():
    # A categorical dtype orders a grouped term's levels by its categories, not
    # by sorted group label, so the file has to carry the model's order.
    X, y = _frame()
    X = X.assign(brand=X["brand"].astype("category"))
    grouping = collapse_levels(X["brand"], groups={"Other": ["B13", "B14"]})
    model = _plain(brand=Categorical(base="B2", grouping=grouping, unseen="Other")).fit(X, y)
    assert model._specs["brand"]._levels == ["B1", "B10", "B11", "B12", "Other", "B2"]
    applied = Structure.from_model(model).apply(_plain()).fit(X, y)
    assert applied._specs["brand"]._levels == model._specs["brand"]._levels
    gap = np.max(np.abs(applied.predict(X) - model.predict(X)))
    assert gap <= _linear_predictor_bound(X, applied, model)


def test_next_years_new_level_takes_the_other_group_with_one_warning():
    X, y = _frame()
    model = _brand_structure().apply(_plain())
    model.fit(X, y)
    next_year, _ = _frame(seed=2027, n=60, brands=[*BRANDS, "B99"])
    new = (next_year["brand"] == "B99").to_numpy()
    assert new.any()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mu = model.predict(next_year)
    routed = [str(w.message) for w in caught if "unseen at fit" in str(w.message)]
    assert routed == [
        "Routing rows with categorical levels unseen at fit to the group 'Other' "
        f"(unseen='Other'): ['B99'] over {int(new.sum())} row(s). They take that group's effect."
    ]
    as_member = next_year.assign(brand=np.where(new, "B13", next_year["brand"]))
    assert np.array_equal(mu, model.predict(as_member))


def test_apply_places_new_levels_in_x_where_the_structure_says_new_levels_go():
    next_year, next_y = _frame(seed=2027, brands=[*BRANDS, "B99"])
    count = int((next_year["brand"] == "B99").sum())
    with pytest.warns(UserWarning) as placed:
        model = _brand_structure().apply(_plain(), X=next_year)
    assert [str(w.message) for w in placed] == [
        "Levels of 'brand' the structure does not list go to the group 'Other' "
        f"(unseen='Other'): ['B99'] over {count} row(s)."
    ]
    assert model._specs["brand"]._grouping.group_to_originals["Other"] == ["B13", "B14", "B99"]
    model.fit(next_year, next_y)
    # Without X the grouping covers only the structure's levels, and the fit
    # says so, as a fit, with the remedy.
    with pytest.raises(ValueError) as refused:
        _brand_structure().apply(_plain()).fit(next_year, next_y)
    assert str(refused.value) == (
        "Feature 'brand': Training data contains levels the grouping does not cover: ['B99']. "
        "Covered: ['B1', 'B10', 'B11', 'B12', 'B13', 'B14', 'B2']. Build the grouping from the "
        "full column, or pass the data to Structure.apply(model, X=data), which places them "
        "where the structure sends new levels."
    )


def test_a_one_level_group_new_levels_go_to_round_trips():
    # "Other" groups only itself, so it says nothing a level does not, except
    # as the group new levels go to; the file keeps it for that.
    X = pd.DataFrame({"cat": np.tile(["A", "B", "Other"], 20)})
    y = np.tile([1.0, 2.0, 4.0], 20)
    grouping = collapse_levels(X["cat"], groups={"Other": ["Other"]})
    model = _declared({"cat": Categorical(base="A", grouping=grouping, unseen="Other")}).fit(X, y)

    structure = Structure.from_model(model)
    applied = read_structure(json.loads(structure.to_json())).apply(model).fit(X, y)

    assert structure.features["cat"].groups == {"Other": ["Other"]}
    probe = pd.DataFrame({"cat": ["new", "Other", "A"]})
    with pytest.warns(UserWarning, match="to the group 'Other'"):
        expected = model.predict(probe)
    with pytest.warns(UserWarning, match="to the group 'Other'"):
        np.testing.assert_array_equal(applied.predict(probe), expected)
    assert expected[0] == expected[1]


def test_apply_fits_new_levels_in_x_as_their_own_without_an_unseen_group():
    next_year, next_y = _frame(seed=2027, brands=[*BRANDS, "B99"])
    count = int((next_year["brand"] == "B99").sum())
    with pytest.warns(UserWarning) as placed:
        model = _brand_structure(unseen="base").apply(_plain(), X=next_year)
    assert [str(w.message) for w in placed] == [
        "Levels of 'brand' the structure does not list are fitted as levels of their own "
        f"(unseen='base'): ['B99'] over {count} row(s)."
    ]
    model.fit(next_year, next_y)
    assert "B99" in model._specs["brand"]._levels


def test_a_group_whose_levels_have_no_rows_next_year_is_pinned_and_still_takes_new_levels():
    # The rare levels a book groups into "Other" are the ones most likely to vanish.
    X, y = _frame(brands=["B1", "B2", "B10", "B11", "B12"])
    with pytest.warns(UserWarning, match=r"pinned to base .*\['Other'\]"):
        model = _brand_structure().apply(_plain()).fit(X, y)
    assert model._specs["brand"]._pinned_levels == ["Other"]
    new = (X["brand"] == "B12").to_numpy()
    with pytest.warns(UserWarning, match=r"to the group 'Other' \(unseen='Other'\): \['B99'\]"):
        mu = model.predict(X.assign(brand=np.where(new, "B99", X["brand"])))
    # "Other" had no rows, so it rates at the reference B1.
    assert np.array_equal(mu, model.predict(X.assign(brand=np.where(new, "B1", X["brand"]))))


def test_a_reference_group_with_no_rows_next_year_falls_back_like_a_declared_level():
    X, y = _frame(brands=["B1", "B2", "B10", "B11", "B12"])
    structure = Structure(
        features={
            "brand": FeatureStructure(
                kind="categorical",
                levels=sorted(BRANDS),
                groups={"Other": ["B13", "B14"]},
                reference="Other",
            )
        }
    )
    with pytest.warns(UserWarning, match="base level 'Other' has no effective training rows"):
        model = structure.apply(_plain()).fit(X, y)
    assert model._specs["brand"]._base_fallback[0] == "Other"


def test_apply_never_fits_and_leaves_the_other_features_alone():
    fresh = _plain()
    before = copy.deepcopy(fresh._specs)
    applied = _brand_structure().apply(fresh)
    assert applied._result is None
    for name in ("area", "age", "band"):
        assert applied._specs[name] is not fresh._specs[name]
        assert type(applied._specs[name]) is type(before[name])
        assert vars(applied._specs[name]).keys() == vars(before[name]).keys()
    assert applied._specs["area"].base == "first"
    assert fresh._specs["brand"]._grouping is None and fresh._specs["brand"].unseen == "error"
    assert applied._specs["brand"]._grouping.group_to_originals["Other"] == ["B13", "B14"]


def test_apply_keeps_the_models_separation_group_pricing_and_bound_levels():
    """The copy is the same model: its model-level rules come with it.

    ``bind_levels`` bound area's universe from a frame holding D; next
    year's rows lack D, so only the binding keeps D a known level that
    predict rates (pinned to the base) instead of refusing.
    """
    X, y = _frame()
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        spline_penalty=0.1,
        separation="error",
        group_pricing="spanned",
        features={"brand": Categorical(base="first"), "area": Categorical(base="first")},
    ).bind_levels(X)

    applied = _brand_structure().apply(model)

    assert (applied._separation, applied._group_pricing) == ("error", "spanned")
    # brand's rebuilt term declares the structure's universe and reference, so
    # its old binding goes; area's, which the structure leaves alone, stays.
    bound = dict(model._config.level_bindings)
    assert applied._config.level_bindings == (("area", bound["area"]),)
    without_d = (X["area"] != "D").to_numpy()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*pinned.*", category=UserWarning)
        applied.fit(X[without_d], y[without_d])
    probe = X[~without_d].head(3)
    assert np.isfinite(applied.predict(probe)).all()
    # The copy's own clone keeps them too.
    again = applied.clone_unfitted()
    assert (again._separation, again._group_pricing) == ("error", "spanned")


def test_apply_keeps_the_bound_universe_of_a_term_it_rebuilds_without_one():
    """An ungrouped structure declares no universe, so the binding's universe stays.

    The binding's base does not: the structure's reference replaces it.
    """
    X, y = _frame()
    model = _declared({"brand": Categorical(base="most_exposed")}).bind_levels(X)
    structure = Structure(
        features={
            "brand": FeatureStructure(kind="categorical", levels=sorted(BRANDS), reference="B10")
        }
    )

    applied = structure.apply(model)
    without_b14 = (X["brand"] != "B14").to_numpy()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*pinned.*", category=UserWarning)
        applied.fit(X[without_b14], y[without_b14])

    assert applied._specs["brand"]._base_level == "B10"
    b14, b10 = applied.predict(pd.DataFrame({"brand": ["B14", "B10"]}))
    assert b14 == b10


@pytest.mark.parametrize("estimated", ["auto_selection", "reml_smoothing", "nb2_theta"])
def test_apply_to_a_fitted_model_takes_the_penalties_and_family_it_was_declared_with(estimated):
    """Penalties and a theta a fit estimated belong to that fit, like its coefficients.

    A calibrated ``selection_penalty="auto"``, REML smoothing and an NB2
    ``theta="auto"`` are not carried into the copy, so applying a structure
    to the fitted model and to its declaration give the same model and the
    same next fit.
    """
    from superglm.distributions import NegativeBinomial

    def counted(frame, y, seed):
        if estimated != "nb2_theta":
            return frame, y
        mu = np.exp(y)
        return frame, np.random.default_rng(seed).negative_binomial(2.0, 2.0 / (2.0 + mu))

    X, y = counted(*_frame(), seed=0)

    def declared():
        return SuperGLM(
            family=NegativeBinomial(theta="auto") if estimated == "nb2_theta" else "gaussian",
            selection_penalty="auto" if estimated == "auto_selection" else 0.0,
            spline_penalty=0.1,
            features={"brand": Categorical(base="first"), "age": Spline(kind="bs", n_knots=8)},
        )

    fitted = declared()
    if estimated == "reml_smoothing":
        fitted.fit_reml(X, y)
    else:
        fitted.fit(X, y)
    structure = _brand_structure()

    from_fit, from_declaration = structure.apply(fitted), structure.apply(declared())

    assert from_fit._penalty_config.lambda1 == from_declaration._penalty_config.lambda1
    assert from_fit.lambda2 == from_declaration.lambda2 == 0.1
    assert getattr(from_fit.family, "theta", None) == getattr(
        from_declaration.family, "theta", None
    )
    next_year, y_next = counted(*_frame(seed=2027), seed=1)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*pinned.*", category=UserWarning)
        from_fit.fit(next_year, y_next)
        from_declaration.fit(next_year, y_next)
    np.testing.assert_array_equal(from_fit.predict(next_year), from_declaration.predict(next_year))


def test_ranges_on_a_ps_spline_rebuild_it_as_bs():
    structure = Structure(
        features={"age": FeatureStructure(kind="spline", ranges=[PolynomialRange(30.0, 45.0, 1)])}
    )
    fresh = _plain()
    assert isinstance(fresh._specs["age"], PSpline)
    applied = structure.apply(fresh)
    spline = applied._specs["age"]
    assert isinstance(spline, BSplineSmooth)
    assert (spline.n_knots, spline.degree) == (fresh._specs["age"].n_knots, 3)
    assert spline.polynomial_ranges == (PolynomialRange(30.0, 45.0, 1),)


def test_an_ordered_term_takes_its_groups_reference_and_band_ranges():
    X, y = _frame()
    structure = Structure(
        features={
            "band": FeatureStructure(
                kind="ordered",
                levels=list(BANDS),
                groups={"0-1": ["0", "1"]},
                reference="3",
                ranges=[PolynomialRange("5", "7", 0, "kink")],
            )
        }
    )
    applied = structure.apply(_plain())
    applied.fit(X, y)
    band = applied._specs["band"]
    assert band._grouping.group_to_originals["0-1"] == ["0", "1"]
    assert band._base_level == "3"
    assert band._spline_obj.polynomial_ranges == (PolynomialRange("5", "7", 0, "kink"),)
    assert Structure.from_model(applied).features["band"] == structure.features["band"]


def test_an_ordered_special_keeps_its_domain_spelling_through_export_and_apply():
    """order=[1.0, ..., 6.0, 9.0] with specials=[9] reports the special as 9.0, beside 1.0.

    The rebuild named it by its raw label 9 only: the editor found no rows
    for it, and the next export, listing 9, no longer applied to the
    declaration. JSON writes 9 and 9.0 apart, so the files show the spelling.
    """
    order = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 9.0]
    band = np.repeat(order, 20)
    X = pd.DataFrame({"band": band})
    y = 1.0 + 0.1 * band + 0.5 * (band == 9.0) + np.random.default_rng(3).normal(0.0, 0.05, 140)

    def declared():
        basis = Spline(kind="bs", n_knots=4)
        return _declared({"band": OrderedCategorical(order=order, specials=[9], basis=basis)})

    first = Structure.from_model(declared().fit(X, y))
    rebuilt = first.apply(declared()).fit(X, y)
    second = Structure.from_model(rebuilt)
    again = Structure.from_model(second.apply(declared()).fit(X, y))

    assert first.to_json() == second.to_json() == again.to_json()
    assert json.loads(first.to_json())["features"]["band"]["levels"][-1] == 9.0
    weights = EditorSession.from_model(rebuilt, train_data=(X, y)).terms["band"].weights
    np.testing.assert_array_equal(weights, np.full(7, 20.0))


def test_a_grouped_ordered_term_fits_a_special_its_domain_spells_differently():
    """A grouped term fits the special that order= spells 9.0 and specials= spells 9.

    The grouping names the special as specials= does, 9, but the rows were
    first matched to the domain's 9.0 and so fell outside it: a grouping=
    built from the column, a grouped Structure.apply and an editor collapse
    all refused the special's own rows as levels the grouping does not cover.
    """
    order = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 9.0]
    band = np.repeat(order, 20)
    X = pd.DataFrame({"band": band})
    y = 1.0 + 0.1 * band + 0.5 * (band == 9.0) + np.random.default_rng(3).normal(0.0, 0.05, 140)

    def declared(grouping=None):
        basis = Spline(kind="bs", n_knots=4)
        term = OrderedCategorical(order=order, specials=[9], grouping=grouping, basis=basis)
        return _declared({"band": term})

    direct = declared(collapse_levels(X["band"], groups={"5-6": ["5.0", "6.0"]})).fit(X, y)
    groups = FeatureStructure(
        kind="ordered", levels=order, groups={"5-6": [5.0, 6.0]}, reference=1.0
    )
    applied = Structure(features={"band": groups}).apply(declared(), X=X).fit(X, y)
    session = EditorSession.from_model(declared().fit(X, y), train_data=(X, y))
    session.stage_structural("collapse", "band", {"levels": ["5.0", "6.0"]})
    session.refit_pending(method="fit")

    for model in (direct, applied, session.model):
        # The special's free column is its own rows' indicator.
        design = model._specs["band"].transform(band)
        np.testing.assert_array_equal(design[:, -1], band == 9.0)
        gap = np.max(np.abs(model.predict(X) - direct.predict(X)))
        assert gap <= _linear_predictor_bound(X, model, direct)
    np.testing.assert_array_equal(session.terms["band"].weights, np.full(7, 20.0))


def test_a_group_of_ordered_levels_named_as_a_special_is_refused():
    """5.0 and 6.0 grouped as "9" beside specials=[9], which the term reports as 9.0.

    The special's indicator matches both spellings, so it claimed the group's
    rows too, and 5, 6 and 9 predicted as one level: renaming the group "5-6"
    to "9" moved the fit by up to 0.57.
    """
    order = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 9.0]
    band = np.repeat(order, 20)
    X = pd.DataFrame({"band": band})
    y = 1.0 + 0.1 * band + 0.5 * (band == 9.0) + np.random.default_rng(3).normal(0.0, 0.05, 140)
    refusal = (
        "OrderedCategorical grouping names a group of other levels '9', a spelling of the free "
        "level 9.0, so that level's indicator would claim the group's rows; give the group "
        "another name."
    )

    def declared(grouping=None):
        basis = Spline(kind="bs", n_knots=4)
        term = OrderedCategorical(order=order, specials=[9], grouping=grouping, basis=basis)
        return _declared({"band": term})

    with pytest.raises(ValueError, match=re.escape(refusal)):
        declared(collapse_levels(X["band"], groups={"9": ["5.0", "6.0"]}))
    feature = FeatureStructure(
        kind="ordered", levels=order, groups={"9": [5.0, 6.0]}, reference=1.0
    )
    with pytest.raises(StructureError) as applied:
        Structure(features={"band": feature}).apply(declared(), X=X)
    # Structure.apply refuses in its own sentence and reports the library's as the cause.
    assert str(applied.value.__cause__) == refusal
    session = EditorSession.from_model(declared().fit(X, y), train_data=(X, y))
    # The editor refuses the name when it is staged, in a sentence of its own
    # (an EditorClientError, so the browser shows it, not an internal error).
    with pytest.raises(EditorValueError) as staged:
        session.stage_structural("collapse", "band", {"levels": ["5.0", "6.0"], "group_label": "9"})
    assert str(staged.value) == (
        "That group name is how a free level of this term is spelled, so the free level "
        "would claim the group's rows. Give the group another name."
    )
    assert session.pending == []


def test_a_reference_named_like_a_base_policy_is_that_level_after_apply():
    """The level named "first" weighs most, so the fit makes it the reference.

    The file holds the level, but apply passed it on as base="first", which
    reads as the policy and chose A: under selection_penalty=10 that moved
    the predictions by up to 0.65.
    """
    X = pd.DataFrame({"x": np.tile(["A", "first", "C", "D"], 30)})
    y, w = np.tile([1.0, 2.0, 4.0, 8.0], 30), np.tile([1.0, 4.0, 1.0, 1.0], 30)

    def declared():
        features = {"x": Categorical(base="most_exposed")}
        return SuperGLM(family="gaussian", selection_penalty=10.0, features=features)

    model = declared().fit(X, y, sample_weight=w)
    applied = Structure.from_model(model).apply(declared()).fit(X, y, sample_weight=w)

    assert model._specs["x"]._base_level == applied._specs["x"]._base_level == "first"
    bound = _linear_predictor_bound(X, model, applied)
    assert np.max(np.abs(applied.predict(X) - model.predict(X))) <= bound


def test_an_ordered_reference_band_named_first_is_that_band_after_apply():
    """The band named "first" weighs most, so the fit makes it the reference.

    Apply passed it on as base="first", which read as the policy and chose
    A, the first band: every reported relativity was rebased to A.
    """
    name = "first"
    bands = ["A", name, "C", "D", "E"]
    X = pd.DataFrame({"band": np.tile(bands, 30)})
    y, w = np.tile([1.0, 2.0, 4.0, 8.0, 9.0], 30), np.tile([1.0, 4.0, 1.0, 1.0, 1.0], 30)

    def declared():
        basis = Spline(kind="bs", n_knots=3, degree=2)
        return _declared({"band": OrderedCategorical(order=bands, basis=basis)})

    model = declared().fit(X, y, sample_weight=w)
    applied = Structure.from_model(model).apply(declared()).fit(X, y, sample_weight=w)

    assert model._specs["band"]._base_level == applied._specs["band"]._base_level == name


def _narrower_next_year():
    """Next year's rows, whose ages stop a year short of this year's oldest."""
    X, _ = _frame()
    top = float(X["age"].max())
    next_year, next_y = _frame(seed=2027)
    keep = (next_year["age"] < top - 1.0).to_numpy()
    return top, next_year[keep].reset_index(drop=True), next_y[keep]


@pytest.mark.parametrize("declared", [False, True], ids=["new", "declares the range"])
def test_a_range_to_this_years_end_fits_next_years_narrower_data_as_written(declared):
    # The editor writes a range dragged to the end as the data's maximum. A
    # model that already declares the range is placed on the data as well.
    top, next_year, next_y = _narrower_next_year()
    structure = _age(PolynomialRange(70.0, top, 0, "kink"))
    ranged = Spline(kind="bs", n_knots=6, polynomial_ranges=[PolynomialRange(70.0, top, 0, "kink")])
    target = _plain(age=ranged) if declared else _plain()
    with pytest.warns(UserWarning) as placed:
        model = structure.apply(target, X=next_year)
    assert [str(w.message) for w in placed] == [
        f"The spline of 'age' is fitted out past this data to hold the Flat range 70–{top:g} "
        "as written."
    ]
    model.fit(next_year, next_y)
    spline = model._specs["age"]
    assert spline.polynomial_ranges == (PolynomialRange(70.0, top, 0, "kink"),)
    assert spline.fitted_boundary == (float(next_year["age"].min()), top)


@pytest.mark.parametrize("declared", [False, True], ids=["new", "declares the range"])
def test_an_ordered_band_range_to_a_band_next_year_lacks_fits_as_written(declared):
    X, y = _frame()
    keep = (X["band"] != "7").to_numpy()
    next_year, next_y = X[keep].reset_index(drop=True), y[keep]
    ranged = Spline(kind="bs", n_knots=4, polynomial_ranges=[PolynomialRange("5", "7", 0, "kink")])
    target = _plain(band=OrderedCategorical(order=BANDS, basis=ranged)) if declared else _plain()
    structure = Structure(
        features={
            "band": FeatureStructure(
                kind="ordered",
                levels=list(BANDS),
                reference="3",
                ranges=[PolynomialRange("5", "7", 0, "kink")],
            )
        }
    )
    with pytest.warns(UserWarning) as placed:
        model = structure.apply(target, X=next_year)
    assert [str(w.message) for w in placed] == [
        "The spline of 'band' is fitted out past this data to hold the Flat range 5–7 as written."
    ]
    model.fit(next_year, next_y)
    band = model._specs["band"]
    assert band._spline_obj.polynomial_ranges == (PolynomialRange("5", "7", 0, "kink"),)
    assert band._basis_spline.fitted_boundary[1] == band._range_edge_value("7")
    assert np.isfinite(model.predict(next_year.iloc[:1].assign(band="7"))).all()


def _no_old_bands():
    X, _ = _frame()
    return X[~X["band"].isin(["6", "7"])].reset_index(drop=True)


@pytest.mark.parametrize(
    ("structure", "data", "sentence"),
    [
        (
            lambda: _age(PolynomialRange(100.0, 120.0, 1, "kink")),
            lambda: _frame()[0],
            "The spline of 'age' refuses the Line range 100–120; change or remove that range.",
        ),
        (
            lambda: _age(PolynomialRange(30.0, 45.0, 1), PolynomialRange(100.0, 120.0, 0, "kink")),
            lambda: _frame()[0],
            "The spline of 'age' refuses the Flat range 100–120; change or remove that range.",
        ),
        (
            lambda: Structure(
                features={
                    "band": FeatureStructure(
                        kind="ordered",
                        levels=list(BANDS),
                        reference="3",
                        ranges=[PolynomialRange("6", "7", 0, "kink")],
                    )
                }
            ),
            _no_old_bands,
            "The spline of 'band' refuses the Flat range 6–7; change or remove that range.",
        ),
    ],
    ids=["a range past the data", "beside a range it holds", "bands the data lacks"],
)
def test_apply_with_x_refuses_a_range_the_data_cannot_hold(structure, data, sentence):
    with pytest.raises(StructureError) as refused:
        structure().apply(_plain(), X=data())
    assert str(refused.value) == sentence
    assert isinstance(refused.value.__cause__, RangeError)


# -- Apply refusals (S4) ---------------------------------------------------------


def _age(*ranges) -> Structure:
    return Structure(features={"age": FeatureStructure(kind="spline", ranges=list(ranges))})


APPLY_REFUSALS = {
    "feature not in the model": (
        lambda: Structure(
            features={
                "region": FeatureStructure(kind="categorical", levels=["N", "S"], reference="N")
            }
        ),
        _plain,
        "The structure names 'region', which is not a feature of this model; remove it from "
        "the structure or apply it to a model that has it.",
    ),
    "kind does not match": (
        lambda: Structure(
            features={"age": FeatureStructure(kind="categorical", levels=[1, 2], reference=1)}
        ),
        _plain,
        "The structure has 'age' as a categorical term, but the model does not; apply it to "
        "a model that declares 'age' as a categorical term.",
    ),
    "degree above the spline's": (
        lambda: _age(PolynomialRange(30.0, 45.0, 2)),
        lambda: _plain(age=Spline(kind="bs", n_knots=6, degree=1, m=1)),
        "The spline of 'age' refuses the Quadratic range 30–45; change or remove that range.",
    ),
    "tangent on a linear spline": (
        lambda: _age(PolynomialRange(30.0, 45.0, 1, "tangent")),
        lambda: _plain(age=Spline(kind="bs", n_knots=6, degree=1, m=1)),
        "The spline of 'age' refuses the Line range 30–45; change or remove that range.",
    ),
    "overlapping ranges": (
        lambda: _age(PolynomialRange(30.0, 45.0, 1), PolynomialRange(40.0, 50.0, 0, "kink")),
        _plain,
        "The spline of 'age' refuses the Flat range 40–50; change or remove that range.",
    ),
    "outside a declared boundary": (
        lambda: _age(PolynomialRange(85.0, 90.0, 1)),
        lambda: _plain(age=Spline(kind="bs", n_knots=6, boundary=(18.0, 80.0))),
        "The spline of 'age' refuses the Line range 85–90; change or remove that range.",
    ),
    "a spline that takes no shapes": (
        lambda: _age(PolynomialRange(30.0, 45.0, 1)),
        lambda: _plain(age=Spline(kind="bs", n_knots=6, select=True)),
        "The spline of 'age' refuses the Line range 30–45; change or remove that range.",
    ),
    "a band range inside a group": (
        lambda: Structure(
            features={
                "band": FeatureStructure(
                    kind="ordered",
                    levels=list(BANDS),
                    groups={"6-7": ["6", "7"]},
                    reference="3",
                    ranges=[PolynomialRange("4", "7", 0, "kink")],
                )
            }
        ),
        _plain,
        "The spline of 'band' refuses the Flat range 4–7; change or remove that range.",
    ),
    "ordered levels that are not the model's": (
        lambda: Structure(
            features={"band": FeatureStructure(kind="ordered", levels=BANDS[:-1], reference="3")}
        ),
        _plain,
        "The levels of 'band' in the structure are not the levels the model declares for it; "
        "apply the structure to a model declared with the same levels.",
    ),
    "ordered levels in another order": (
        lambda: Structure(
            features={"band": FeatureStructure(kind="ordered", levels=BANDS[::-1], reference="3")}
        ),
        _plain,
        "The levels of 'band' in the structure are in another order than the model declares "
        "them; apply the structure to a model that declares them in the same order.",
    ),
    "declared levels the grouping misses": (
        _brand_structure,
        lambda: _plain(brand=Categorical(base="first", levels=[*BRANDS, "B15"])),
        "The levels of 'brand' in the structure are not the levels the model declares for it; "
        "apply the structure to a model declared with the same levels.",
    ),
    "ungrouped levels that are not the declared ones": (
        lambda: Structure(
            features={
                "area": FeatureStructure(kind="categorical", levels=["A", "C"], reference="A")
            }
        ),
        lambda: _plain(area=Categorical(base="first", levels=["A", "B"])),
        "The levels of 'area' in the structure are not the levels the model declares for it; "
        "apply the structure to a model declared with the same levels.",
    ),
    "a group member the declared levels leave out": (
        _brand_structure,
        lambda: _plain(brand=Categorical(base="first", levels=BRANDS[:-1])),
        "The levels of 'brand' in the structure are not the levels the model declares for it; "
        "apply the structure to a model declared with the same levels.",
    ),
}


@pytest.mark.parametrize(
    ("structure", "model", "sentence"), list(APPLY_REFUSALS.values()), ids=list(APPLY_REFUSALS)
)
def test_apply_refuses_with_its_fixed_sentence(structure, model, sentence):
    with pytest.raises(StructureError) as refused:
        structure().apply(model())
    assert str(refused.value) == sentence


def test_apply_refuses_levels_in_x_that_the_declared_levels_leave_out():
    # Placing them in a group would widen the universe the model declares.
    next_year, _ = _frame(seed=2027, brands=[*BRANDS, "B99"])
    declared = _plain(brand=Categorical(base="first", levels=BRANDS))
    with pytest.raises(StructureError) as refused:
        _brand_structure().apply(declared, X=next_year)
    assert str(refused.value) == (
        "The data holds levels of 'brand' that the model's levels= leaves out: ['B99']; "
        "add them to its levels= or leave those rows out."
    )


def test_a_term_declared_before_the_grouping_check_round_trips():
    """A model from superglm 0.36.1 declares levels=A, B, C while its grouping also maps D.

    Construction refuses that declaration now; a model pickled before then
    keeps it and scores D as its group Other. Its own structure applies back
    to it and refits the same model.
    """
    rng = np.random.default_rng(2)
    n = 600
    X = pd.DataFrame({"t": rng.choice(list("ABC"), n), "age": rng.uniform(18.0, 80.0, n)})
    y = 0.5 + 0.2 * (X["t"] == "B") + 0.1 * np.sin(X["age"] / 15.0) + rng.normal(0.0, 0.05, n)
    legacy = Categorical(base="A", levels=list("ABC"))
    legacy._grouping = collapse_levels(list("ABCD"), groups={"Other": ["C", "D"]})
    model = _declared({"t": legacy, "age": Spline(kind="bs", n_knots=5)}).fit(X, y)

    structure = Structure.from_model(model)
    assert structure.features["t"].levels == ["A", "B", "C", "D"]
    applied = structure.apply(model)
    applied.fit(X, y)

    assert applied._specs["t"]._levels == model._specs["t"]._levels == ["A", "B", "Other"]
    probe = pd.DataFrame({"t": list("ABCD"), "age": [40.0] * 4})
    gap = np.max(np.abs(applied.predict(probe) - model.predict(probe)))
    assert gap <= _linear_predictor_bound(probe, applied, model)
    d, c = applied.predict(probe)[[3, 2]]
    assert d == c


def test_an_unexpected_library_error_becomes_the_features_refusal(monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("deep inside")

    monkeypatch.setattr(structure_module, "rebuilt_categorical", broken)
    with pytest.raises(StructureError) as refused:
        _brand_structure().apply(_plain())
    assert str(refused.value) == (
        "The structure could not be applied to 'brand': the model's declaration of it does "
        "not accept these decisions."
    )
    assert isinstance(refused.value.__cause__, RuntimeError)


# -- Layering ---------------------------------------------------------------------

_WITHOUT_THE_EDITOR = r"""
import json
import sys

import numpy as np
import pandas as pd

from superglm import (
    Categorical, OrderedCategorical, PolynomialRange, Spline, Structure, SuperGLM,
    collapse_levels, read_structure,
)

rng = np.random.default_rng(7)
n = 400
X = pd.DataFrame({
    "brand": rng.choice(["A", "B", "C", "D"], n),
    "band": rng.choice(["0", "1", "2", "3", "4", "5"], n),
    "age": rng.uniform(18.0, 80.0, n),
})
y = 0.5 + 0.1 * (X["brand"] == "C") + 0.1 * np.sin(X["age"] / 15.0) + rng.normal(0.0, 0.05, n)


def declared(brand, band, age):
    return SuperGLM(
        family="gaussian", selection_penalty=0.0, spline_penalty=0.1,
        features={"brand": brand, "band": band, "age": age},
    )


bands = ["0", "1", "2", "3", "4", "5"]
model = declared(
    Categorical(base="A", grouping=collapse_levels(X["brand"], groups={"CD": ["C", "D"]}), unseen="CD"),
    OrderedCategorical(order=bands, basis=Spline(
        kind="bs", n_knots=3, polynomial_ranges=[PolynomialRange("3", "5", 0, "kink")]
    )),
    Spline(kind="bs", n_knots=6, polynomial_ranges=[PolynomialRange(30.0, 45.0, 1)]),
)
model.fit(X, y)
plain = declared(
    Categorical(base="first"),
    OrderedCategorical(order=bands, basis=Spline(kind="bs", n_knots=3)),
    Spline(kind="bs", n_knots=6),
)
read_structure(json.loads(Structure.from_model(model, X).to_json())).apply(plain, X=X)
editor = sorted(name for name in sys.modules if name.split(".")[:2] == ["superglm", "editor"])
assert not editor, editor
"""


def test_reading_and_applying_a_structure_never_imports_the_editor():
    # A fresh process: this one has imported the editor for the tests above. The
    # script's model carries a grouping, a reference, an unseen group and ranges
    # on a spline and an ordered term, and applying its structure to the plain
    # declaration rebuilds every one of them.
    completed = subprocess.run(
        [sys.executable, "-c", _WITHOUT_THE_EDITOR],
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stderr


def _band_model(specials=None):
    order = BANDS if specials is None else [*BANDS, *specials]
    basis = Spline(kind="bs", n_knots=4)
    return _declared({"band": OrderedCategorical(order=order, specials=specials, basis=basis)})


def test_a_structure_takes_levels_off_the_curve_and_puts_them_back():
    """specials= in a file frees a band of the declaration; a file without it puts the band back."""
    X, y = _frame()
    plain = _band_model().fit(X, y)
    entry = FeatureStructure(kind="ordered", levels=BANDS, reference="0", specials=["3"])
    special = Structure(features={"band": entry}).apply(_band_model()).fit(X, y)
    assert list(special._specs["band"]._special_display) == ["3"]
    exported = Structure.from_model(special)
    assert json.loads(exported.to_json())["features"]["band"]["specials"] == ["3"]
    # The level keeps its place: the export lists the bands in order.
    assert exported.features["band"].levels == BANDS
    again = Structure.from_model(exported.apply(_band_model()).fit(X, y))
    assert again.to_json() == exported.to_json()

    back = FeatureStructure(kind="ordered", levels=BANDS, reference="0", specials=[])
    # An explicit empty list, unlike none at all, survives the file.
    assert (
        read_structure(json.loads(Structure(features={"band": back}).to_json()))
        .features["band"]
        .specials
        == []
    )
    returned = Structure(features={"band": back}).apply(special).fit(X, y)
    assert list(returned._specs["band"]._special_display) == []
    gap = np.max(np.abs(returned.predict(X) - plain.predict(X)))
    assert gap <= _linear_predictor_bound(X, returned, plain)
    # A file that names no specials keeps only the declared ones: the export of
    # the returned model puts "3" back on the special model's curve too.
    again = Structure.from_model(returned).apply(special)
    assert list(again._specs["band"]._special_display) == []


def test_a_structure_refuses_to_free_a_level_of_a_term_with_positional_breaks():
    """Freeing a band would move every break stated by a position after it."""
    declared = _declared({"band": OrderedCategorical(order=BANDS, basis=Piecewise(breaks=[3, 5]))})
    entry = FeatureStructure(kind="ordered", levels=BANDS, reference="0", specials=["2"])
    with pytest.raises(StructureError) as refused:
        Structure(features={"band": entry}).apply(declared)
    assert str(refused.value) == (
        "'band' states its Piecewise breaks by position, which a level the structure makes "
        "special would move; state them by band name."
    )


def test_a_structure_without_specials_keeps_the_declared_ones_and_cannot_put_them_on_the_curve():
    X, y = _frame()
    X.loc[X.index[:80], "band"] = "MISSING"
    declared = _band_model(specials=["MISSING"]).fit(X, y)
    legacy = json.loads(Structure.from_model(declared).to_json())
    assert legacy["features"]["band"]["specials"] == ["MISSING"]
    del legacy["features"]["band"]["specials"]
    kept = read_structure(legacy).apply(_band_model(specials=["MISSING"]))
    assert list(kept._specs["band"]._special_display) == ["MISSING"]

    legacy["features"]["band"]["specials"] = []
    with pytest.raises(StructureError) as refused:
        read_structure(legacy).apply(_band_model(specials=["MISSING"]))
    assert str(refused.value) == (
        "'MISSING' is declared special in the model's 'band', so it has no place on the curve for "
        "the structure to put it in; list it among the structure's specials, or declare it on the "
        "curve."
    )


@pytest.mark.parametrize(
    ("change", "sentence"),
    [
        (
            {"specials": ["9"]},
            "'9' of 'band' is listed as a special level but is not one of its levels; add it to "
            "the levels or take it out of the specials.",
        ),
        (
            {"specials": ["3"], "groups": {"3-4": ["3", "4"]}},
            "Special level '3' of 'band' is in group '3-4'; a special level stands alone, so take "
            "it out of the group.",
        ),
        (
            {"specials": ["0"]},
            "The reference '0' of 'band' is a special level, and the reference must lie on the "
            "curve; choose a level on the curve.",
        ),
        (
            {"specials": "3"},
            "The structure entry for 'band' has a malformed 'specials'; export the structure again.",
        ),
    ],
)
def test_structure_specials_are_checked_in_fixed_sentences(change, sentence):
    entry = {"kind": "ordered", "levels": BANDS, "reference": "0", **change}
    with pytest.raises(StructureError) as refused:
        Structure(features={"band": FeatureStructure(**entry)})
    assert str(refused.value) == sentence


def test_only_an_ordered_entry_may_name_specials():
    entry = FeatureStructure(kind="categorical", levels=BRANDS, reference="B1", specials=["B2"])
    with pytest.raises(StructureError) as refused:
        Structure(features={"brand": entry})
    assert str(refused.value) == (
        "The structure entry for 'brand' has a malformed 'specials'; export the structure again."
    )
