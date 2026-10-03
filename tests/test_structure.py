"""Structure files: a fitted model's structural decisions, read back and applied (spec 2026-10-03, phase 7c)."""

from __future__ import annotations

import copy
import json
import re
import warnings

import numpy as np
import pandas as pd
import pytest

from superglm import (
    BSplineSmooth,
    Categorical,
    OrderedCategorical,
    PolynomialRange,
    PSpline,
    Spline,
    Structure,
    SuperGLM,
    collapse_levels,
    read_structure,
)
from superglm.editor import EditorSession
from superglm.editor import collapse as collapse_module
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
    "declared levels the grouping misses": (
        _brand_structure,
        lambda: _plain(brand=Categorical(base="first", levels=[*BRANDS, "B15"])),
        "The levels of 'brand' in the structure are not the levels the model declares for it; "
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


def test_an_unexpected_library_error_becomes_the_features_refusal(monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("deep inside")

    monkeypatch.setattr(collapse_module, "_rebuilt_categorical", broken)
    with pytest.raises(StructureError) as refused:
        _brand_structure().apply(_plain())
    assert str(refused.value) == (
        "The structure could not be applied to 'brand': the model's declaration of it does "
        "not accept these decisions."
    )
    assert isinstance(refused.value.__cause__, RuntimeError)
