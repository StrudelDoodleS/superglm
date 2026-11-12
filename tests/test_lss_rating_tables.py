"""A SuperLSS rating table has to reproduce the model it was exported from, parameter by parameter.

``floor + base x product of relativities`` (``rule``) is checked row by row against
``predict_parameters`` with a consumer written here that reads only what the payload and the
workbook carry: level labels, knots, polynomial coefficients, per-unit relativities, interaction
cells and the offsets it is handed.  Tolerances are a multiple of ``u = 2**-53`` times the number
of factors in the product.  The mutation checks show that the floor, an interaction block and the
negative-binomial reciprocal each carry a number a user would read.
"""

from __future__ import annotations

import dataclasses
import warnings
from functools import cache

import numpy as np
import pandas as pd
import pytest

from superglm import (
    GammaLS,
    GaussianLS,
    LogNormalLS,
    NegativeBinomialLS,
    Numeric,
    OrderedCategorical,
    Polynomial,
    Spline,
    SuperLSS,
    TweedieLSS,
    cat,
    re,
    s,
    term,
    ti,
)
from superglm.features.interaction import CategoricalInteraction, NumericInteraction
from superglm.features.piecewise import Piecewise
from superglm.terms import interaction

U = 2.0**-53


def _book(n: int = 500, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "age": rng.uniform(18, 80, n),
            "dens": rng.uniform(0, 1, n),
            "region": rng.choice(["A", "B", "C"], n),
            "seg": rng.choice(["p", "q"], n),
            "expo": rng.uniform(0.3, 1.0, n),
        }
    )


def _fit(family, first_terms, other, y, X, **offsets):
    first = "location" if isinstance(family, GaussianLS) else "mean"
    second = "theta" if isinstance(family, NegativeBinomialLS) else "scale"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SuperLSS(
            family,
            getattr(family, first)(
                *first_terms, **({"offset": offsets["mean"]} if offsets else {})
            ),
            getattr(family, second)(*other),
        ).fit_reml(X, y)


@cache
def lognormal():
    """Mean: spline, two categoricals, their interaction, a numeric term and a log-exposure offset."""
    X, rng = _book(), np.random.default_rng(1)
    mu = np.exp(
        7
        + 0.4 * np.sin(X.age / 12)
        + 0.3 * (X.region == "B")
        + 0.5 * ((X.region == "C") & (X.seg == "q"))
    )
    sigma = np.exp(-0.7 + 0.5 * ((X.age - 50) / 30) ** 2 + 0.2 * (X.region == "C"))
    y = np.exp(np.log(mu * X.expo) - 0.5 * sigma**2 + sigma * rng.standard_normal(len(X)))
    offsets = {"mean": np.log(X.expo.to_numpy())}
    fam = LogNormalLS()
    mean_terms = (
        s("age", k=6),
        cat("region"),
        cat("seg"),
        interaction(CategoricalInteraction("region", "seg")),
        term("dens", Numeric()),
    )
    return _fit(fam, mean_terms, (s("age", k=5), cat("region")), y, X, **offsets), X, offsets


@cache
def gamma():
    """Mean on categorical, numeric and spline terms; scale holds a ``ti`` tensor."""
    X, rng = _book(seed=5), np.random.default_rng(2)
    mu = np.exp(1 + 0.3 * np.sin(X.age / 12) + 0.3 * (X.region == "B") + 0.4 * X.dens)
    cv = np.exp(-0.6 + 0.3 * (X.seg == "q"))
    y = rng.gamma(1 / cv**2, mu * cv**2)
    fam = GammaLS()
    mean_terms = (s("age", k=6), cat("region"), term("dens", Numeric()))
    other = (s("age", k=5), s("dens", k=4), ti("age", "dens", n_knots=(3, 3)), cat("seg"))
    return _fit(fam, mean_terms, other, y, X), X


@cache
def negative_binomial():
    """Mean with an exposure offset; theta on a spline and a categorical."""
    X, rng = _book(seed=9), np.random.default_rng(3)
    mu = X.expo * np.exp(1.0 + 0.3 * np.sin(X.age / 12) + 0.4 * (X.region == "C"))
    theta = np.exp(0.3 - 0.5 * (X.seg == "q"))
    y = rng.negative_binomial(theta, theta / (theta + mu)).astype(float)
    offsets = {"mean": np.log(X.expo.to_numpy())}
    fam = NegativeBinomialLS()
    return (
        _fit(fam, (s("age", k=6), cat("region")), (s("age", k=5), cat("seg")), y, X, **offsets),
        X,
        offsets,
    )


@cache
def gaussian():
    """Gaussian: an identity-link location beside a floored-log scale."""
    X, rng = _book(seed=4), np.random.default_rng(4)
    y = (
        2
        + 0.03 * X.age
        + 0.5 * (X.region == "B")
        + np.exp(-0.5 + 0.4 * (X.seg == "q")) * rng.standard_normal(len(X))
    )
    fam = GaussianLS()
    return _fit(fam, (s("age", k=5), cat("region")), (cat("seg"),), y, X), X


def _ppform(table: pd.DataFrame, x: np.ndarray) -> np.ndarray:
    """exp of the cubic on the interval ``x`` falls in; a tail row is read, not evaluated."""
    lo = np.array([float(k.strip("[)]").split(", ")[0]) for k in table.iloc[:, 0]])
    hi = np.array([float(k.strip("[)]").split(", ")[1]) for k in table.iloc[:, 0]])
    i = np.clip(np.searchsorted(lo, x, "right") - 1, 0, len(lo) - 1)
    bounded = np.isfinite(hi - lo)[i]
    u = np.where(bounded, (x - lo[i]) / np.where(bounded, (hi - lo)[i], 1.0), 0.0)
    a, b, c, d = (table[k].to_numpy(float)[i] for k in "abcd")
    return np.where(
        bounded, np.exp(a + u * (b + u * (c + u * d))), table.Relativity.to_numpy(float)[i]
    )


def consume(table, X, offset=None, *, skip=(), floor=True, reciprocal=True):
    """The parameter at every row of ``X``, from ``table``'s blocks alone."""
    product = np.full(len(X), table.base_relativity)
    sign = -1.0 if table.rule == "reciprocal" else 1.0
    factors = 1
    for block in table.main_effects:
        t, name = block.table, block.name
        if block.kind == "categorical":
            lookup = dict(zip(t[name].astype(str), t.Relativity))
            product *= X[name].astype(str).map(lookup).to_numpy(float)
        elif block.kind == "numeric":
            product *= t.Relativity.iloc[0] ** X[name].to_numpy(float)
        elif block.kind == "continuous_ppform":
            product *= _ppform(t, X[name].to_numpy(float))
        elif block.kind == "offset_per_unit":
            product *= np.exp(sign * offset) * t.Relativity.iloc[0]
        else:
            raise AssertionError(block.kind)
        factors += 1
    for block in table.interactions:
        if block.name in skip:
            continue
        cells = block.table.set_index(block.table.columns[0])
        first, second = block.name.split(":")
        product *= [cells.loc[a, b] for a, b in zip(X[first], X[second])]
        factors += 1
    if table.rule == "reciprocal":
        return (1.0 / product if reciprocal else product), factors
    return (
        table.floor * floor + product if table.rule == "floor_plus_product" else product
    ), factors


def worst_error(table, X, truth, offset=None, **mutations):
    value, factors = consume(table, X, offset, **mutations)
    return float(np.max(np.abs(value / truth - 1.0))), factors


def check(table, X, offsets, model):
    truth = model.predict_parameters(X, offsets=offsets)[table.parameter].to_numpy()
    error, factors = worst_error(
        table, X, truth, None if offsets is None else offsets.get(table.parameter)
    )
    assert error <= 16 * (factors + 2) * U, (table.parameter, error)
    return error


def _special_named_interaction_case(*, categorical):
    rng = np.random.default_rng(2)
    if categorical:
        n = 1200
        X = pd.DataFrame(
            {"x": rng.choice(list("abc"), n), "special": rng.choice(["p", "q"], n, p=[0.3, 0.7])}
        )
        eta = (
            0.5
            + 0.3 * (X.x == "b")
            - 0.2 * (X.x == "c")
            + 0.25 * (X.special == "q")
            + 0.4 * ((X.x == "b") & (X.special == "q"))
        )
        mains = (cat("x", base="a"), cat("special", base="p"))
        ispec = CategoricalInteraction("x", "special")
    else:
        n = 400
        X = pd.DataFrame({"x": rng.uniform(-1, 1, n), "special": rng.normal(size=n)})
        eta = 0.5 + 0.4 * X.x - 0.2 * X.special + 0.3 * X.x * X.special
        mains = (term("x", Numeric()), term("special", Numeric()))
        ispec = NumericInteraction("x", "special")
    y = rng.gamma(4, np.exp(eta) / 4)
    family = GammaLS()
    model = SuperLSS(family, family.mean(*mains, interaction(ispec)), family.scale()).fit(X, y)
    return model, X


def test_numeric_special_named_interaction_keeps_the_export_omission_reason():
    model, X = _special_named_interaction_case(categorical=False)
    payload = model.rating_table_payload(X)
    reason = payload.omitted["mean"]
    assert "its interaction 'x:special'" in reason
    assert "has no exact rating table yet" in reason


def test_categorical_special_named_interaction_tables_reproduce_predictions():
    model, X = _special_named_interaction_case(categorical=True)
    payload = model.rating_table_payload(X)
    assert payload.omitted == {}
    for table in payload.parameters:
        check(table, X, None, model)


def test_lognormal_mean_and_sigma_reproduce_predict_parameters():
    model, X, offsets = lognormal()
    payload = model.rating_table_payload(X, offsets=offsets)
    mean, scale = payload.parameters
    assert (mean.quantity, mean.rule, scale.quantity, scale.rule) == (
        "Mean",
        "identity",
        "Sigma",
        "floor_plus_product",
    )
    assert scale.floor == LogNormalLS().scale_floor
    assert [b.kind for b in mean.main_effects] == [
        "continuous_ppform",
        "categorical",
        "categorical",
        "numeric",
        "offset_per_unit",
    ]
    assert [b.name for b in mean.interactions] == ["region:seg"]
    assert set(payload.blocks()) >= {"mean:age", "mean:region", "mean:region:seg", "scale:age"}
    for table in payload.parameters:
        check(table, X, offsets, model)


def test_gamma_cv_reproduces_predict_parameters_and_the_tensor_parameter_is_named():
    model, X = gamma()
    payload = model.rating_table_payload(X)
    assert [t.parameter for t in payload.parameters] == ["mean"]
    sentence = payload.omitted["scale"]
    assert "'scale'" in sentence and "age:dens" in sentence and "TensorInteraction" in sentence
    check(payload.parameters[0], X, None, model)


def test_negative_binomial_table_is_the_overdispersion():
    model, X, offsets = negative_binomial()
    payload = model.rating_table_payload(X, offsets=offsets)
    theta = payload.parameters[1]
    assert (theta.quantity, theta.rule) == ("Overdispersion", "reciprocal")
    check(payload.parameters[0], X, offsets, model)
    check(theta, X, offsets, model)
    truth = model.predict_parameters(X, offsets=offsets)["theta"].to_numpy()
    overdispersion, _ = consume(dataclasses.replace(theta, rule="identity"), X)
    assert np.max(np.abs(overdispersion * truth - 1.0)) <= 16 * 8 * U


def test_a_non_log_parameter_is_named_and_left_out_while_the_others_export():
    model, X = gaussian()
    payload = model.rating_table_payload(X)
    assert [t.parameter for t in payload.parameters] == ["scale"]
    assert payload.omitted["location"].startswith(
        "Parameter 'location' was left out: it is on an identity link"
    )
    check(payload.parameters[0], X, None, model)


def test_a_tweedie_model_tabulates_mean_and_dispersion_and_states_its_constant_power():
    X, rng = _book(seed=6), np.random.default_rng(6)
    mu = np.exp(3 + 0.3 * np.sin(X.age / 12) + 0.4 * (X.region == "B"))
    claims = rng.poisson(0.8, len(X))
    y = np.array([rng.gamma(2.0, m / 1.6, k).sum() for m, k in zip(mu, claims, strict=True)])
    fam = TweedieLSS()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SuperLSS(
            fam, fam.mu(s("age", k=5), cat("region")), fam.phi(cat("seg")), fam.p()
        ).fit_reml(X, y)
    payload = model.rating_table_payload(X)

    assert [(t.parameter, t.quantity, t.rule) for t in payload.parameters] == [
        ("mean", "Mean", "identity"),
        ("dispersion", "Dispersion", "identity"),
    ]
    for table in payload.parameters:
        check(table, X, None, model)
    # The power has no terms, so it is one number, and a reader of the workbook is told it.
    power = float(model.predict_parameters(X.iloc[:1])["power"].iloc[0])
    assert payload.omitted == {
        "power": f"Parameter 'power' has no terms: it is {power:.6g} on every row and needs no table."
    }


def test_categorical_factors_are_the_coefficients_read_directly():
    model, X, offsets = lognormal()
    fitted = model._require_fitted()
    spec = next(p for p in fitted.compiled_predictors if p.name == "mean").compiled.specs["seg"]
    free = [level for level in spec._levels if level != spec._base_level]
    beta = fitted.prediction_coefficients[fitted.layout.term_slices["mean:seg"]]
    table = model.rating_table_payload(X, offsets=offsets).blocks()["mean:seg"].table
    got = dict(zip(table.seg, table.Relativity))
    assert got[spec._base_level] == 1.0
    for level, coefficient in zip(free, beta):
        assert got[level] == pytest.approx(np.exp(coefficient), rel=4 * U)


def test_rows_differing_in_one_column_differ_by_that_blocks_ratio():
    model, X = gamma()
    table = model.rating_table_payload(X).parameters[0]
    rows = X.iloc[[0, 0]].reset_index(drop=True)
    rows.loc[1, "region"] = "B" if rows.loc[0, "region"] != "B" else "A"
    rows.loc[1, "dens"] = rows.loc[0, "dens"] + 0.25
    ratio = model.predict_parameters(rows)["mean"].pipe(lambda m: m[1] / m[0])
    region = table.main_effects[1].table.set_index("region").Relativity
    per_unit = table.main_effects[2].table.Relativity.iloc[0]
    expected = region[rows.loc[1, "region"]] / region[rows.loc[0, "region"]] * per_unit**0.25
    assert ratio == pytest.approx(expected, rel=16 * U)


def test_mutations_each_break_the_invariant():
    model, X, offsets = lognormal()
    payload = model.rating_table_payload(X, offsets=offsets)
    mean, scale = payload.parameters
    truth = model.predict_parameters(X, offsets=offsets)
    assert worst_error(scale, X, truth["scale"].to_numpy(), floor=False)[0] > 1e-3
    assert (
        worst_error(mean, X, truth["mean"].to_numpy(), offsets["mean"], skip=("region:seg",))[0]
        > 1e-2
    )
    assert worst_error(mean, X, truth["mean"].to_numpy(), offsets["mean"])[0] < 1e-13
    model, X, offsets = negative_binomial()
    theta = model.rating_table_payload(X, offsets=offsets).parameters[1]
    truth = model.predict_parameters(X, offsets=offsets)["theta"].to_numpy()
    assert worst_error(theta, X, truth, None, reciprocal=False)[0] > 0.1


def test_a_hand_edited_model_is_tabulated_as_it_predicts():
    model, X, offsets = lognormal()
    fitted = model._require_fitted()
    assert np.array_equal(fitted.coefficients, fitted.prediction_coefficients)
    edited = fitted.coefficients.copy()
    columns = fitted.layout.term_slices["mean:region"]
    edited[columns] += 0.3
    revised = model._with_revised_coefficients(edited, edited_terms=["mean:region"], X_train=X)
    reread = revised._require_fitted()
    assert np.array_equal(reread.coefficients, reread.prediction_coefficients)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        payload = revised.rating_table_payload(X, offsets=offsets)
    before = model.rating_table_payload(X, offsets=offsets).blocks()["mean:region"].table
    after = payload.blocks()["mean:region"].table
    assert not np.allclose(before.Relativity, after.Relativity)
    for table in payload.parameters:
        check(table, X, offsets, revised)


def test_the_workbook_rebuilds_the_product_from_its_cells(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    model, X, offsets = lognormal()
    path = model.export_rating_tables(tmp_path / "lss.xlsx", X, offsets=offsets)
    book = openpyxl.load_workbook(path)
    assert book.sheetnames == ["Mean", "Sigma", "Summary"]
    payload = model.rating_table_payload(X, offsets=offsets)
    for table in payload.parameters:
        ws = book[table.quantity]
        assert ws["A2"].value == "Base" and ws["C2"].value == table.base_relativity
        assert (ws["A3"].value, ws["C3"].value) == (
            ("Floor", table.floor) if table.rule == "floor_plus_product" else (None, None)
        )
        read = _read_sheet(ws, table)
        truth = model.predict_parameters(X, offsets=offsets)[table.parameter].to_numpy()
        error, _ = worst_error(read, X, truth, offsets.get(table.parameter))
        assert error <= 1e-13  # cells are written as float64; this is the whole path


def _read_sheet(ws, table):
    """The table's blocks as a consumer finds them on the sheet: titles on row 5, headers on row 7."""

    def frame(row, col, width=None):
        header = []
        while ws.cell(row=row, column=col + len(header)).value is not None and (
            width is None or len(header) < width
        ):
            header.append(ws.cell(row=row, column=col + len(header)).value)
        rows = []
        while ws.cell(row=row + 1 + len(rows), column=col).value is not None:
            rows.append(
                [ws.cell(row=row + 1 + len(rows), column=col + k).value for k in range(len(header))]
            )
        return pd.DataFrame(rows, columns=header)

    mains = []
    starts = [c for c in range(1, ws.max_column + 1) if ws.cell(row=5, column=c).value is not None]
    for col, following in zip(starts, [*starts[1:], ws.max_column + 1], strict=True):
        name = ws.cell(row=5, column=col).value
        t = frame(7, col, following - col)
        kind = (
            "continuous_ppform"
            if "a" in t.columns
            else "offset_per_unit"
            if name == "Offset Multiplier"
            else "numeric"
            if t.iloc[0, 0] == "per_unit"
            else "categorical"
        )
        mains.append(type("B", (), {"name": name, "kind": kind, "table": t})())
    cells = []
    for block in table.interactions:
        row = next(
            r for r in range(8, ws.max_row + 1) if ws.cell(row=r, column=1).value == block.name
        )
        cells.append(type("B", (), {"name": block.name, "table": frame(row + 2, 1)})())
    return type("T", (), {**table.__dict__, "main_effects": mains, "interactions": cells})()


def _gamma_with(mean_terms):
    X, rng = _book(400, 21), np.random.default_rng(8)
    X["band"] = rng.choice([f"B{i}" for i in range(5)], len(X))
    y = np.exp(0.5 + 0.1 * X.band.str[1].astype(int) + 0.004 * X.age) * rng.gamma(5, 0.2, len(X))
    fam = GammaLS()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SuperLSS(fam, fam.mean(*mean_terms), fam.scale(cat("region"))).fit_reml(X, y)
    return model, X


@pytest.mark.parametrize(
    "mean_terms",
    [
        (
            term(
                "band",
                OrderedCategorical(order=[f"B{i}" for i in range(5)], basis=Spline(kind="ps", k=5)),
            ),
        ),
        (s("age", k=6, select=True), cat("region")),
        (s("age", k=6, kind="cr", discrete=True), re("band")),
    ],
    ids=["ordered", "select", "discrete-and-random-effect"],
)
def test_other_exact_term_kinds_reproduce_predict_parameters(mean_terms):
    model, X = _gamma_with(mean_terms)
    for table in model.rating_table_payload(X).parameters:
        check(table, X, None, model)


@pytest.mark.parametrize(
    ("spec", "kind"),
    [
        (Polynomial(degree=3), "Polynomial"),
        (Piecewise(breaks=[40.0, 60.0], base=40.0), "Piecewise"),
    ],
)
def test_a_term_without_an_exact_table_is_named_for_its_parameter_only(spec, kind):
    model, X = _gamma_with((term("age", spec), cat("region")))
    payload = model.rating_table_payload(X)
    assert [t.parameter for t in payload.parameters] == ["scale"]
    assert f"its term 'age' is a {kind}" in payload.omitted["mean"]


def _summary_sheet(path):
    """The Summary sheet as (header row, one dict per parameter row, every other text cell)."""
    openpyxl = pytest.importorskip("openpyxl")
    ws = openpyxl.load_workbook(path)["Summary"]
    header = [cell.value for cell in ws[1] if cell.value is not None]
    rows = []
    for row in range(2, ws.max_row + 1):
        values = [ws.cell(row=row, column=c).value for c in range(1, len(header) + 1)]
        if values[1] is None:  # a note, not a parameter row
            break
        rows.append(dict(zip(header, values, strict=True)))
    text = [
        ws.cell(row=r, column=1).value
        for r in range(2 + len(rows), ws.max_row + 1)
        if ws.cell(row=r, column=1).value is not None
    ]
    return header, rows, text


def test_the_summary_names_each_parameter_as_the_editor_does_with_its_code_name_beside(tmp_path):
    from superglm.editor.display import parameter_display
    from superglm.export.lss_rating_tables import parameter_label

    model, X, offsets = negative_binomial()
    path = model.export_rating_tables(tmp_path / "nb.xlsx", X, offsets=offsets)
    _, rows, _ = _summary_sheet(path)
    assert [(r["Parameter"], r["Name in code"], r["Table"]) for r in rows] == [
        ("Mean", "mean", "Mean"),
        ("Overdispersion", "theta", "Overdispersion"),
    ]
    # The names are the editor's own, so the workbook and the editor read alike.
    for family in (LogNormalLS(), GammaLS(), NegativeBinomialLS(), TweedieLSS()):
        for spec in family.parameters:
            assert parameter_label(spec.name) == parameter_display(spec, average=False).label


def test_the_summary_has_a_floor_column_only_when_a_parameter_has_a_floor(tmp_path):
    model, X, offsets = lognormal()
    floored, rows, _ = _summary_sheet(
        model.export_rating_tables(tmp_path / "ln.xlsx", X, offsets=offsets)
    )
    assert "Floor" in floored
    floors = {
        t.quantity: t.floor for t in model.rating_table_payload(X, offsets=offsets).parameters
    }
    assert {r["Table"]: r["Floor"] for r in rows} == floors
    assert floors["Sigma"] > 0.0
    nb, X, offsets = negative_binomial()
    header, _, _ = _summary_sheet(nb.export_rating_tables(tmp_path / "nb.xlsx", X, offsets=offsets))
    assert "Floor" not in header


def test_the_notes_define_weight_and_a_tweedie_gives_the_chance_of_no_claim(tmp_path):
    model, X, offsets = lognormal()
    _, _, notes = _summary_sheet(
        model.export_rating_tables(tmp_path / "ln.xlsx", X, offsets=offsets)
    )
    assert any(
        "Weight is the sum of the sample weights" in text and "number of rows" in text
        for text in notes
    )
    assert not any("exp(-lambda)" in text for text in notes)

    X, rng = _book(seed=6), np.random.default_rng(6)
    mu = np.exp(3 + 0.3 * np.sin(X.age / 12) + 0.4 * (X.region == "B"))
    claims = rng.poisson(0.8, len(X))
    y = np.array([rng.gamma(2.0, m / 1.6, k).sum() for m, k in zip(mu, claims, strict=True)])
    fam = TweedieLSS()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tweedie = SuperLSS(
            fam, fam.mu(s("age", k=5), cat("region")), fam.phi(cat("seg")), fam.p()
        ).fit_reml(X, y)
    _, _, notes = _summary_sheet(tweedie.export_rating_tables(tmp_path / "tw.xlsx", X))
    sentence = next(text for text in notes if "exp(-lambda)" in text)
    assert "lambda = w * mean^(2 - p) / (dispersion * (2 - p))" in sentence
    # The sentence is the Tweedie's point mass: check it against the model's own P(Y = 0), for a
    # row of weight 1 and for weighted rows, whose weight enters as the dispersion over w.
    row = X.iloc[:5]
    par = tweedie.predict_parameters(row)
    p = par["power"].to_numpy()
    theta = par[["mean", "dispersion", "power"]].to_numpy()
    for w in (np.ones(len(row)), np.array([0.5, 2.0, 1.0, 0.25, 3.0])):
        lam = w * par["mean"].to_numpy() ** (2 - p) / (par["dispersion"].to_numpy() * (2 - p))
        atom = tweedie.family.cdf_prior_weighted(np.zeros(len(row)), theta, w)
        np.testing.assert_allclose(np.exp(-lam), atom, rtol=1e-9)


def test_an_interaction_whose_level_names_collide_on_a_colon_gives_each_cell_its_own_factor():
    """("a", "b:c") and ("a:b", "c") share the label "a:b:c"; each still gets its own factor."""
    X, rng = _book(seed=3), np.random.default_rng(4)
    X["region"] = X["region"].map({"A": "a", "B": "a:b", "C": "z"})
    X["seg"] = X["seg"].map({"p": "b:c", "q": "c"})
    eta = 1 + 0.3 * (X.region == "a:b") + 0.4 * (X.seg == "c")
    y = rng.gamma(4.0, np.exp(eta + 0.5 * ((X.region == "a:b") & (X.seg == "c"))) / 4.0)
    model = _fit(
        GammaLS(),
        (cat("region"), cat("seg"), interaction(CategoricalInteraction("region", "seg"))),
        (s("age", k=5),),
        y,
        X,
    )
    payload = model.rating_table_payload(X)
    assert payload.omitted == {}
    mean = next(table for table in payload.parameters if table.parameter == "mean")
    (block,) = mean.interactions
    assert block.table.columns[0] == "region"
    assert sorted(block.table.columns[1:]) == ["b:c", "c"]
    assert sorted(block.table["region"]) == ["a", "a:b", "z"]
    for table in payload.parameters:
        check(table, X, None, model)


def test_an_interaction_level_named_as_its_row_column_keeps_its_row_keys():
    """A "seg" level called "region" takes the row-key header, so that header is suffixed."""
    X, rng = _book(seed=8), np.random.default_rng(9)
    X["seg"] = X["seg"].map({"p": "region", "q": "other"})
    eta = 1 + 0.3 * (X.region == "B") + 0.4 * (X.seg == "other")
    y = rng.gamma(4.0, np.exp(eta + 0.5 * ((X.region == "C") & (X.seg == "other"))) / 4.0)
    model = _fit(
        GammaLS(),
        (cat("region"), cat("seg"), interaction(CategoricalInteraction("region", "seg"))),
        (s("age", k=5),),
        y,
        X,
    )
    payload = model.rating_table_payload(X)
    assert payload.omitted == {}
    mean = next(table for table in payload.parameters if table.parameter == "mean")
    (block,) = mean.interactions
    assert block.table.columns[0] == "region (axis 1)"
    assert sorted(block.table.columns[1:]) == ["other", "region"]
    assert sorted(block.table["region (axis 1)"]) == ["A", "B", "C"]
    for table in payload.parameters:
        check(table, X, None, model)


def test_a_frequency_weighted_tweedie_gives_the_chance_of_no_claim_without_the_weight(tmp_path):
    """A frequency weight counts repeated rows, so each row keeps the plain law's point mass."""
    X, rng = _book(seed=12), np.random.default_rng(12)
    mu = np.exp(3 + 0.3 * np.sin(X.age / 12) + 0.4 * (X.region == "B"))
    claims = rng.poisson(0.8, len(X))
    y = np.array([rng.gamma(2.0, m / 1.6, k).sum() for m, k in zip(mu, claims, strict=True)])
    w = rng.integers(1, 4, len(X)).astype(float)
    fam = TweedieLSS()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tweedie = SuperLSS(
            fam,
            fam.mu(s("age", k=5), cat("region")),
            fam.phi(cat("seg")),
            fam.p(),
            weight_semantics="frequency",
        ).fit_reml(X, y, sample_weight=w)
    _, _, notes = _summary_sheet(tweedie.export_rating_tables(tmp_path / "tw.xlsx", X, w))
    sentence = next(text for text in notes if "exp(-lambda)" in text)
    assert "lambda = mean^(2 - p) / (dispersion * (2 - p))" in sentence and "w *" not in sentence
    row = X.iloc[:5]
    par = tweedie.predict_parameters(row)
    p = par["power"].to_numpy()
    lam = par["mean"].to_numpy() ** (2 - p) / (par["dispersion"].to_numpy() * (2 - p))
    atom = tweedie.family.cdf(np.zeros(len(row)), par[["mean", "dispersion", "power"]].to_numpy())
    np.testing.assert_allclose(np.exp(-lam), atom, rtol=1e-9)
