"""Grouped categorical parents keep one raw-label contract in interactions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from superglm import Categorical, SuperGLM, collapse_levels
from superglm.features.interaction import (
    CategoricalInteraction,
    NumericCategorical,
    PolynomialCategorical,
    SplineCategorical,
)
from superglm.features.numeric import Numeric
from superglm.features.polynomial import Polynomial
from superglm.features.spline import Spline

_KINDS = ("spline", "polynomial", "numeric", "categorical")
_RAW = np.tile(np.array(["a", "b", "c", "d"], dtype=object), 6)
_OTHER = np.repeat(np.array(["u", "v", "w"], dtype=object), 8)
_X = np.linspace(-0.9, 1.1, _RAW.size)


def _grouping():
    return collapse_levels(_RAW, groups={"AB": ["a", "b"]})


def _mapped(values, grouping):
    return np.asarray([grouping.original_to_group[str(value)] for value in values], dtype=object)


def _materialize(result) -> np.ndarray:
    infos = result if isinstance(result, list) else [result]
    blocks = []
    for info in infos:
        if info.columns is not None:
            block = info.columns.toarray() if sp.issparse(info.columns) else info.columns
        elif info.spline_cat_basis is not None:
            block = info.spline_cat_basis.toarray()
            block = block * np.asarray(info.spline_cat_mask)[:, None]
        else:
            block = np.asarray(info.spline_cat_basis_unique)[info.spline_cat_bin_idx]
            block = block * np.asarray(info.spline_cat_mask)[:, None]
        if info.projection is not None:
            block = block @ info.projection
        blocks.append(np.asarray(block, dtype=np.float64))
    return np.hstack(blocks)


def _width(result) -> int:
    infos = result if isinstance(result, list) else [result]
    return sum(info.n_cols for info in infos)


def _new_interaction(kind: str):
    if kind == "spline":
        return SplineCategorical("x", "g")
    if kind == "polynomial":
        return PolynomialCategorical("x", "g")
    if kind == "numeric":
        return NumericCategorical("x", "g")
    return CategoricalInteraction("g", "h")


def _interaction_case(kind: str, *, grouped: bool = True):
    grouping = _grouping() if grouped else None
    fitted_labels = _mapped(_RAW, grouping) if grouping is not None else _RAW.copy()

    actual_cat = Categorical(base="c", grouping=grouping)
    reference_cat = Categorical(base="c")
    actual_cat.build(_RAW)
    reference_cat.build(fitted_labels)

    if kind == "spline":
        actual_parent = Spline(n_knots=5)
        reference_parent = Spline(n_knots=5)
        actual_parent.build(_X)
        reference_parent.build(_X)
        actual = _new_interaction(kind)
        reference = _new_interaction(kind)
    elif kind == "polynomial":
        actual_parent = Polynomial(degree=2)
        reference_parent = Polynomial(degree=2)
        actual_parent.build(_X)
        reference_parent.build(_X)
        actual = _new_interaction(kind)
        reference = _new_interaction(kind)
    elif kind == "numeric":
        actual_parent = Numeric()
        reference_parent = Numeric()
        actual_parent.build(_X)
        reference_parent.build(_X)
        actual = _new_interaction(kind)
        reference = _new_interaction(kind)
    else:
        actual_parent = Categorical(base="u")
        reference_parent = Categorical(base="u")
        actual_parent.build(_OTHER)
        reference_parent.build(_OTHER)
        actual = _new_interaction(kind)
        reference = _new_interaction(kind)

    actual_parents = {"x": actual_parent, "g": actual_cat}
    reference_parents = {"x": reference_parent, "g": reference_cat}
    if kind == "categorical":
        actual_parents = {"g": actual_cat, "h": actual_parent}
        reference_parents = {"g": reference_cat, "h": reference_parent}
        actual_args = (_RAW, _OTHER)
        reference_args = (fitted_labels, _OTHER)
    else:
        actual_args = (_X, _RAW)
        reference_args = (_X, fitted_labels)

    actual_build = actual.build(*actual_args, actual_parents)
    reference_build = reference.build(*reference_args, reference_parents)
    return (
        actual,
        reference,
        actual_build,
        reference_build,
        actual_args,
        reference_args,
        actual_parents,
        reference_parents,
    )


@pytest.mark.parametrize("kind", _KINDS)
def test_grouped_raw_build_matches_manually_pregrouped_build(kind):
    _, _, actual, reference, *_ = _interaction_case(kind)
    np.testing.assert_allclose(_materialize(actual), _materialize(reference))
    assert np.count_nonzero(_materialize(actual)) > 0


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("route", ("transform", "score"))
def test_grouped_raw_runtime_routes_match_manually_pregrouped_routes(kind, route):
    actual, reference, actual_build, _, actual_args, reference_args, *_ = _interaction_case(kind)
    if route == "transform":
        got = actual.transform(*actual_args)
        expected = reference.transform(*reference_args)
    else:
        beta = np.linspace(0.2, 1.0, _width(actual_build))
        got = actual.score(*actual_args, beta)
        expected = reference.score(*reference_args, beta)
    np.testing.assert_allclose(got, expected)


@pytest.mark.parametrize("kind", _KINDS)
def test_originals_in_one_group_receive_identical_columns_and_scores(kind):
    actual, _, actual_build, _, _, _, *_ = _interaction_case(kind)
    if kind == "categorical":
        args = (
            np.array(["a", "b"], dtype=object),
            np.array(["v", "v"], dtype=object),
        )
    else:
        args = (
            np.array([0.25, 0.25]),
            np.array(["a", "b"], dtype=object),
        )
    transformed = actual.transform(*args)
    scored = actual.score(*args, np.linspace(0.2, 1.0, _width(actual_build)))
    np.testing.assert_allclose(transformed[0], transformed[1])
    np.testing.assert_allclose(scored[0], scored[1])


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("route", ("build", "transform", "score"))
@pytest.mark.parametrize(
    ("bad_label", "message"),
    ((None, "missing"), ("not-fitted", "unseen")),
)
def test_grouped_runtime_routes_reject_missing_and_unseen_original_labels(
    kind,
    route,
    bad_label,
    message,
):
    actual, _, actual_build, _, actual_args, _, actual_parents, _ = _interaction_case(kind)
    args = list(actual_args)
    categorical_index = 0 if kind == "categorical" else 1
    labels = np.asarray(args[categorical_index], dtype=object).copy()
    labels[0] = bad_label
    args[categorical_index] = labels
    with pytest.raises(ValueError, match=message):
        if route == "build":
            _new_interaction(kind).build(*args, actual_parents)
        elif route == "transform":
            actual.transform(*args)
        else:
            actual.score(*args, np.ones(_width(actual_build)))


@pytest.mark.parametrize("route", ("build", "transform", "score"))
@pytest.mark.parametrize(
    ("bad_label", "message"),
    ((None, "missing"), ("not-fitted", "unseen")),
)
def test_categorical_interaction_second_margin_rejects_invalid_labels(
    route,
    bad_label,
    message,
):
    actual, actual_build, actual_parents = _two_grouped_categorical_case(siblings_observed=True)
    args = [
        np.tile(np.array(["a", "b", "d"], dtype=object), 4),
        np.tile(np.array(["u", "v", "z"], dtype=object), 4),
    ]
    labels = np.asarray(args[1], dtype=object).copy()
    labels[0] = bad_label
    args[1] = labels
    with pytest.raises(ValueError, match=message):
        if route == "build":
            CategoricalInteraction("g", "h").build(*args, actual_parents)
        elif route == "transform":
            actual.transform(*args)
        else:
            actual.score(*args, np.ones(_width(actual_build)))


def _partial_group_case(kind: str, *, sibling_observed: bool):
    domain = np.array(["a", "b", "c", "d"], dtype=object)
    grouping = collapse_levels(domain, groups={"BC": ["b", "c"]})
    fitted = (
        np.tile(np.array(["a", "b", "d"], dtype=object), 4)
        if sibling_observed
        else np.tile(np.array(["a", "d"], dtype=object), 6)
    )
    x = np.linspace(-0.8, 0.9, fitted.size)
    cat = Categorical(base="a", grouping=grouping)
    cat.build(fitted)

    if kind == "spline":
        parent = Spline(n_knots=5)
        parent.build(x)
    elif kind == "polynomial":
        parent = Polynomial(degree=2)
        parent.build(x)
    elif kind == "numeric":
        parent = Numeric()
        parent.build(x)
    else:
        other = np.tile(np.array(["u", "v"], dtype=object), 6)
        parent = Categorical(base="u")
        parent.build(other)

    term = _new_interaction(kind)
    if kind == "categorical":
        args = (fitted, other)
        parents = {"g": cat, "h": parent}
    else:
        args = (x, fitted)
        parents = {"x": parent, "g": cat}
    result = term.build(*args, parents)
    return term, result, parents


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("route", ("build", "transform", "score"))
def test_valid_original_mapping_to_group_absent_from_fit_raises(kind, route):
    term, result, parents = _partial_group_case(kind, sibling_observed=False)
    args = (
        (np.array(["b"], dtype=object), np.array(["v"], dtype=object))
        if kind == "categorical"
        else (np.array([0.2]), np.array(["b"], dtype=object))
    )
    with pytest.raises(ValueError, match=r"\[g\].*unseen.*BC"):
        if route == "build":
            _new_interaction(kind).build(*args, parents)
        elif route == "transform":
            term.transform(*args)
        else:
            term.score(*args, np.ones(_width(result)))


@pytest.mark.parametrize("kind", _KINDS)
def test_unobserved_original_is_accepted_when_grouped_sibling_was_fitted(kind):
    term, result, parents = _partial_group_case(kind, sibling_observed=True)
    if kind == "categorical":
        sibling_args = (
            np.array(["b"], dtype=object),
            np.array(["v"], dtype=object),
        )
        unseen_member_args = (
            np.array(["c"], dtype=object),
            np.array(["v"], dtype=object),
        )
    else:
        sibling_args = (np.array([0.2]), np.array(["b"], dtype=object))
        unseen_member_args = (np.array([0.2]), np.array(["c"], dtype=object))

    sibling_build = _new_interaction(kind).build(*sibling_args, parents)
    unseen_member_build = _new_interaction(kind).build(*unseen_member_args, parents)
    np.testing.assert_allclose(
        _materialize(sibling_build),
        _materialize(unseen_member_build),
    )
    np.testing.assert_allclose(
        term.transform(*sibling_args),
        term.transform(*unseen_member_args),
    )
    beta = np.linspace(0.2, 1.0, _width(result))
    np.testing.assert_allclose(
        term.score(*sibling_args, beta),
        term.score(*unseen_member_args, beta),
    )


def test_main_effect_rejects_group_absent_from_fit_but_accepts_fitted_sibling():
    domain = np.array(["a", "b", "c", "d"], dtype=object)
    grouping = collapse_levels(domain, groups={"BC": ["b", "c"]})
    absent = Categorical(base="a", grouping=grouping)
    absent.build(np.array(["a", "d", "a", "d"], dtype=object))
    for route in ("transform", "score"):
        with pytest.raises(ValueError, match="unseen.*BC"):
            if route == "transform":
                absent.transform(np.array(["b"], dtype=object))
            else:
                absent.score(np.array(["b"], dtype=object), np.array([0.3]))

    sibling = Categorical(base="a", grouping=grouping)
    sibling.build(np.array(["a", "b", "d", "a", "b", "d"], dtype=object))
    np.testing.assert_array_equal(
        sibling.transform(np.array(["b", "c"], dtype=object))[0],
        sibling.transform(np.array(["b", "c"], dtype=object))[1],
    )
    np.testing.assert_allclose(
        sibling.score(np.array(["b", "c"], dtype=object), np.array([0.4, 0.7])),
        [0.4, 0.4],
    )


def _two_grouped_categorical_case(*, siblings_observed: bool):
    domain1 = np.array(["a", "b", "c", "d"], dtype=object)
    domain2 = np.array(["u", "v", "w", "z"], dtype=object)
    grouping1 = collapse_levels(domain1, groups={"BC": ["b", "c"]})
    grouping2 = collapse_levels(domain2, groups={"VW": ["v", "w"]})
    if siblings_observed:
        fitted1 = np.tile(np.array(["a", "b", "d"], dtype=object), 4)
        fitted2 = np.tile(np.array(["u", "v", "z"], dtype=object), 4)
    else:
        fitted1 = np.tile(np.array(["a", "d"], dtype=object), 6)
        fitted2 = np.tile(np.array(["u", "z"], dtype=object), 6)
    cat1 = Categorical(base="a", grouping=grouping1)
    cat2 = Categorical(base="u", grouping=grouping2)
    cat1.build(fitted1)
    cat2.build(fitted2)
    parents = {"g": cat1, "h": cat2}
    term = CategoricalInteraction("g", "h")
    result = term.build(fitted1, fitted2, parents)
    return term, result, parents


@pytest.mark.parametrize(
    ("margin", "bad_label", "group_label"),
    ((0, "b", "BC"), (1, "v", "VW")),
)
@pytest.mark.parametrize("route", ("build", "transform", "score"))
def test_categorical_interaction_rejects_absent_group_on_either_margin(
    margin,
    bad_label,
    group_label,
    route,
):
    term, result, parents = _two_grouped_categorical_case(siblings_observed=False)
    args = [
        np.array(["d"], dtype=object),
        np.array(["z"], dtype=object),
    ]
    args[margin] = np.array([bad_label], dtype=object)
    with pytest.raises(ValueError, match=rf"unseen.*{group_label}"):
        if route == "build":
            CategoricalInteraction("g", "h").build(*args, parents)
        elif route == "transform":
            term.transform(*args)
        else:
            term.score(*args, np.ones(_width(result)))


@pytest.mark.parametrize(
    ("margin", "sibling", "unobserved"),
    ((0, "b", "c"), (1, "v", "w")),
)
def test_categorical_interaction_accepts_grouped_sibling_on_either_margin(
    margin,
    sibling,
    unobserved,
):
    term, result, parents = _two_grouped_categorical_case(siblings_observed=True)
    sibling_args = [
        np.array(["d"], dtype=object),
        np.array(["z"], dtype=object),
    ]
    unseen_args = [arg.copy() for arg in sibling_args]
    sibling_args[margin] = np.array([sibling], dtype=object)
    unseen_args[margin] = np.array([unobserved], dtype=object)

    np.testing.assert_array_equal(
        _materialize(CategoricalInteraction("g", "h").build(*sibling_args, parents)),
        _materialize(CategoricalInteraction("g", "h").build(*unseen_args, parents)),
    )
    np.testing.assert_array_equal(
        term.transform(*sibling_args),
        term.transform(*unseen_args),
    )
    beta = np.linspace(0.2, 1.0, _width(result))
    np.testing.assert_allclose(
        term.score(*sibling_args, beta),
        term.score(*unseen_args, beta),
    )


def test_group_label_collision_is_mapped_exactly_once_on_every_route():
    raw = np.tile(np.array(["A", "G", "C"], dtype=object), 3)
    grouping = collapse_levels(raw, groups={"G": ["A"], "H": ["G"]})
    mapped = _mapped(raw, grouping)
    x = np.ones(raw.size)

    actual_cat = Categorical(base="C", grouping=grouping)
    reference_cat = Categorical(base="C")
    actual_cat.build(raw)
    reference_cat.build(mapped)
    actual = NumericCategorical("x", "g")
    reference = NumericCategorical("x", "g")
    actual_info = actual.build(x, raw, {"x": Numeric(), "g": actual_cat})
    reference_info = reference.build(x, mapped, {"x": Numeric(), "g": reference_cat})

    expected_first_cycle = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    np.testing.assert_array_equal(actual_info.columns[:3], expected_first_cycle)
    np.testing.assert_array_equal(actual_info.columns, reference_info.columns)
    np.testing.assert_array_equal(actual.transform(x, raw)[:3], expected_first_cycle)
    np.testing.assert_array_equal(actual.score(x, raw, np.array([2.0, 5.0]))[:3], [2.0, 5.0, 0.0])


def test_integer_originals_and_string_groups_match_public_reference_routes():
    raw = np.tile(np.array([1, 2, 3, 4], dtype=np.int64), 6)
    grouping = collapse_levels(raw, groups={"low": ["1", "2"]})
    mapped = _mapped(raw, grouping)
    actual_cat = Categorical(base="3", grouping=grouping)
    reference_cat = Categorical(base="3")
    actual_info = actual_cat.build(raw)
    reference_info = reference_cat.build(mapped)
    np.testing.assert_array_equal(actual_info.cat_codes, reference_info.cat_codes)
    np.testing.assert_array_equal(actual_cat.transform(raw), reference_cat.transform(mapped))
    beta = np.array([0.3, 0.7])
    np.testing.assert_allclose(actual_cat.score(raw, beta), reference_cat.score(mapped, beta))

    x = np.linspace(-0.5, 0.8, raw.size)
    actual = NumericCategorical("x", "g")
    reference = NumericCategorical("x", "g")
    actual_build = actual.build(x, raw, {"x": Numeric(), "g": actual_cat})
    reference_build = reference.build(x, mapped, {"x": Numeric(), "g": reference_cat})
    np.testing.assert_allclose(actual_build.columns, reference_build.columns)
    np.testing.assert_allclose(actual.transform(x, raw), reference.transform(x, mapped))
    np.testing.assert_allclose(actual.score(x, raw, beta), reference.score(x, mapped, beta))


def test_grouped_spline_discrete_build_matches_manually_pregrouped_build():
    grouping = _grouping()
    mapped = _mapped(_RAW, grouping)
    actual_cat = Categorical(base="c", grouping=grouping)
    reference_cat = Categorical(base="c")
    actual_cat.build(_RAW)
    reference_cat.build(mapped)
    actual_spline = Spline(n_knots=5)
    reference_spline = Spline(n_knots=5)
    actual_spline.build(_X)
    reference_spline.build(_X)
    actual = SplineCategorical("x", "g")
    reference = SplineCategorical("x", "g")

    got = actual.build_discrete(
        _X,
        _RAW,
        {"x": actual_spline, "g": actual_cat},
        n_bins=7,
    )
    expected = reference.build_discrete(
        _X,
        mapped,
        {"x": reference_spline, "g": reference_cat},
        n_bins=7,
    )
    np.testing.assert_allclose(_materialize(got), _materialize(expected))
    assert np.count_nonzero(_materialize(got)) > 0


@pytest.mark.parametrize(
    ("group_first", "group_second"),
    ((True, False), (False, True), (True, True)),
)
def test_categorical_interaction_maps_either_or_both_grouped_margins(
    group_first,
    group_second,
):
    grouping1 = _grouping() if group_first else None
    grouping2 = collapse_levels(_OTHER, groups={"UV": ["u", "v"]}) if group_second else None
    mapped1 = _mapped(_RAW, grouping1) if grouping1 is not None else _RAW.copy()
    mapped2 = _mapped(_OTHER, grouping2) if grouping2 is not None else _OTHER.copy()
    base1 = "c"
    base2 = "w" if group_second else "u"

    actual1 = Categorical(base=base1, grouping=grouping1)
    actual2 = Categorical(base=base2, grouping=grouping2)
    reference1 = Categorical(base=base1)
    reference2 = Categorical(base=base2)
    actual1.build(_RAW)
    actual2.build(_OTHER)
    reference1.build(mapped1)
    reference2.build(mapped2)
    actual = CategoricalInteraction("g", "h")
    reference = CategoricalInteraction("g", "h")
    got = actual.build(_RAW, _OTHER, {"g": actual1, "h": actual2})
    expected = reference.build(mapped1, mapped2, {"g": reference1, "h": reference2})

    np.testing.assert_array_equal(_materialize(got), _materialize(expected))
    np.testing.assert_array_equal(
        actual.transform(_RAW, _OTHER),
        reference.transform(mapped1, mapped2),
    )
    beta = np.linspace(0.2, 1.0, got.n_cols)
    np.testing.assert_allclose(
        actual.score(_RAW, _OTHER, beta),
        reference.score(mapped1, mapped2, beta),
    )


def test_spline_categorical_reconstruct_uses_fitted_group_labels_internally():
    actual, _, actual_build, _, *_ = _interaction_case("spline")
    result = actual.reconstruct(np.linspace(0.05, 0.25, _width(actual_build)), n_points=31)
    assert "AB" in result["levels"]
    assert "AB" not in _grouping().all_original_levels
    assert np.isfinite(result["per_level"]["AB"]["log_relativity"]).all()


@pytest.mark.parametrize("kind", _KINDS)
def test_ungrouped_interaction_runtime_contract_is_unchanged(kind):
    actual, _, actual_build, _, actual_args, _, *_ = _interaction_case(kind, grouped=False)
    transformed = actual.transform(*actual_args)
    np.testing.assert_allclose(transformed, _materialize(actual_build))
    beta = np.linspace(0.2, 1.0, _width(actual_build))
    np.testing.assert_allclose(actual.score(*actual_args, beta), transformed @ beta)


@pytest.mark.parametrize("kind", (*_KINDS, "spline_discrete"))
def test_ungrouped_interaction_build_skips_categorical_resolver(kind, monkeypatch):
    import superglm.features.interaction as interaction_module

    case_kind = "spline" if kind == "spline_discrete" else kind
    _, _, _, _, actual_args, _, actual_parents, _ = _interaction_case(
        case_kind,
        grouped=False,
    )

    def reject_resolution(*args, **kwargs):
        raise AssertionError("ungrouped build must not rescan fitted categorical labels")

    monkeypatch.setattr(
        interaction_module,
        "_resolve_categorical_labels",
        reject_resolution,
    )
    term = _new_interaction(case_kind)
    if kind == "spline_discrete":
        term.build_discrete(*actual_args, actual_parents, n_bins=7)
    else:
        term.build(*actual_args, actual_parents)


@pytest.mark.parametrize("kind", (*_KINDS, "spline_discrete"))
def test_grouped_interaction_build_still_dispatches_one_raw_resolution(kind, monkeypatch):
    import superglm.features.interaction as interaction_module

    case_kind = "spline" if kind == "spline_discrete" else kind
    _, _, _, _, actual_args, _, actual_parents, _ = _interaction_case(case_kind)
    original = interaction_module._resolve_categorical_labels
    calls = []

    def record_resolution(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(
        interaction_module,
        "_resolve_categorical_labels",
        record_resolution,
    )
    term = _new_interaction(case_kind)
    if kind == "spline_discrete":
        term.build_discrete(*actual_args, actual_parents, n_bins=7)
    else:
        term.build(*actual_args, actual_parents)
    assert len(calls) == 1


def test_both_grouped_categorical_build_margins_each_resolve_once(monkeypatch):
    import superglm.features.interaction as interaction_module

    _, _, parents = _two_grouped_categorical_case(siblings_observed=True)
    left = np.tile(np.array(["a", "b", "d"], dtype=object), 4)
    right = np.tile(np.array(["u", "v", "z"], dtype=object), 4)
    original = interaction_module._resolve_categorical_labels
    calls = []

    def record_resolution(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(
        interaction_module,
        "_resolve_categorical_labels",
        record_resolution,
    )
    CategoricalInteraction("g", "h").build(left, right, parents)
    assert len(calls) == 2


def test_public_grouped_interaction_fit_predict_and_canonicalization():
    raw = np.tile(np.array(["a", "b", "c", "d"], dtype=object), 30)
    x = np.linspace(-1.0, 1.0, raw.size)
    grouping = collapse_levels(raw, groups={"AB": ["a", "b"]})
    y = 1.0 + 0.2 * x + 0.3 * np.isin(raw, ["a", "b"]) * x
    frame = pd.DataFrame({"x": x, "g": raw})
    model = SuperGLM(
        family="gaussian",
        selection_penalty=None,
        features={
            "x": Spline(n_knots=5),
            "g": Categorical(base="c", grouping=grouping),
        },
        interactions=[("x", "g")],
    ).fit(frame, y)

    prediction = model.predict(frame)
    assert prediction.shape == (raw.size,)
    assert np.isfinite(prediction).all()
    canonical = model._runtime_canonical_state
    assert canonical["terms"]
    assert canonical["solver_to_public_complete"] is True
    assert canonical["diagnostics"]["max_abs_eta_delta"] < 1e-12
    assert canonical["diagnostics"]["max_abs_mu_delta"] < 1e-12
    same_group = pd.DataFrame({"x": [0.25, 0.25], "g": ["a", "b"]})
    np.testing.assert_allclose(model.predict(same_group)[0], model.predict(same_group)[1])


def test_default_and_explicit_screening_include_grouped_factor_and_match_reference():
    rng = np.random.default_rng(7)
    raw = np.tile(np.array(["a", "b", "c", "d"], dtype=object), 45)
    rng.shuffle(raw)
    x = rng.normal(size=raw.size)
    grouping = collapse_levels(raw, groups={"AB": ["a", "b"]})
    mapped = _mapped(raw, grouping)
    y = 0.2 * x + 0.5 * np.isin(raw, ["a", "b"]) + rng.normal(scale=0.5, size=raw.size)

    raw_frame = pd.DataFrame({"x": x, "g": raw})
    grouped_model = SuperGLM(
        family="gaussian",
        selection_penalty=None,
        features={
            "x": Numeric(),
            "g": Categorical(base="c", grouping=grouping),
        },
    ).fit_reml(raw_frame, y)
    default = grouped_model.screen_interactions(raw_frame, y)
    explicit = grouped_model.screen_interactions(
        raw_frame,
        y,
        candidates=[("x", "g")],
    )

    reference_frame = pd.DataFrame({"x": x, "g": mapped})
    reference_model = SuperGLM(
        family="gaussian",
        selection_penalty=None,
        features={"x": Numeric(), "g": Categorical(base="c")},
    ).fit_reml(reference_frame, y)
    reference = reference_model.screen_interactions(
        reference_frame,
        y,
        candidates=[("x", "g")],
    )

    assert len(default) == 1
    assert default.iloc[0]["kind"] == "numeric_cat"
    pd.testing.assert_frame_equal(default, explicit)
    pd.testing.assert_frame_equal(default, reference, rtol=1e-12, atol=1e-12)
