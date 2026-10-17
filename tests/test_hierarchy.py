"""Nested random effects declared as one ``Hierarchy`` (make, model, year).

A hierarchy is lme4's ``(1 | make/model/year)``: one REML variance per level,
each level keyed by the tuple of its parts, so a model name reused under two
makes is two models and a separator inside a name never merges two paths.
Fixtures are synthetic.
"""

from __future__ import annotations

import pickle
import warnings

import numpy as np
import pandas as pd
import pytest

from superglm import Hierarchy, Numeric, RandomEffect, SuperGLM
from superglm.features.random_effect import validate_declared_nesting

MAKES = ("ford", "kia", "mazda", "volvo")
MODELS = ("alpha", "beta", "gamma")  # every make reuses the same model names
YEARS = ("2018", "2019", "2020")


def _book(seed: int = 0, n: int = 1500) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    make = np.array(MAKES)[rng.integers(0, len(MAKES), n)]
    model = np.array(MODELS)[rng.integers(0, len(MODELS), n)]
    year = np.array(YEARS)[rng.integers(0, len(YEARS), n)]
    engine = np.array(["e1", "e2", "e3"])[rng.integers(0, 3, n)]
    x = rng.normal(size=n)
    make_effect = dict(zip(MAKES, rng.normal(0.0, 0.4, len(MAKES)), strict=True))
    model_effect = {(a, b): rng.normal(0.0, 0.3) for a in MAKES for b in MODELS}
    eta = 0.2 * x + np.array(
        [make_effect[a] + model_effect[a, b] for a, b in zip(make, model, strict=True)]
    )
    frame = pd.DataFrame({"x": x, "make": make, "model": model, "year": year, "engine": engine})
    frame["vehicle"] = frame["make"] + "-" + frame["model"] + "-" + frame["year"]
    return frame, rng.poisson(np.exp(eta)).astype(float)


def _fit(spec, frame, y, **features) -> SuperGLM:
    model = SuperGLM(
        family="poisson",
        features={"x": Numeric(), "vehicle": spec, **features},
        selection_penalty=0,
    )
    return model.fit_reml(frame, y)


def _joined() -> Hierarchy:
    return Hierarchy(names=("make", "model", "year"), sep="-")


def test_one_joined_column_and_separate_columns_fit_identically() -> None:
    """Both declarations build the same path keys, so the fits agree bit for bit.

    Mutation: keying a level by its own part alone (implicit nesting) merges
    ``("ford", "alpha")`` with ``("kia", "alpha")`` and the fits differ.
    """
    frame, y = _book()
    joined = _fit(_joined(), frame, y)
    separate = _fit(Hierarchy(columns=("make", "model", "year")), frame.drop(columns="vehicle"), y)
    assert [g.name for g in joined._groups] == [
        "x",
        "vehicle:make",
        "vehicle:model",
        "vehicle:year",
    ]
    assert [g.name for g in separate._groups] == [g.name for g in joined._groups]
    np.testing.assert_array_equal(joined.result.beta, separate.result.beta)
    np.testing.assert_array_equal(
        joined.predict(frame), separate.predict(frame.drop(columns="vehicle"))
    )
    table = joined.random_effects("vehicle:model").table
    assert len(table) == len(MAKES) * len(MODELS)
    assert ("kia", "alpha") in set(table["level"])


def test_a_separator_inside_a_name_is_refused_under_sep_and_kept_under_columns() -> None:
    """``sep=`` never guesses a split; ``columns=`` needs none.

    Mutation: an ``rsplit`` heuristic for over-long values fits the frame.
    """
    frame, y = _book()
    frame.loc[frame["make"] == "volvo", "make"] = "mercedes-benz"
    frame["vehicle"] = frame["make"] + "-" + frame["model"] + "-" + frame["year"]
    with pytest.raises(ValueError) as refused:
        _fit(_joined(), frame, y)
    message = str(refused.value)
    assert "'mercedes-benz-alpha-2018' (4 parts)" in message
    assert "Hierarchy(columns=" in message
    model = _fit(Hierarchy(columns=("make", "model", "year")), frame.drop(columns="vehicle"), y)
    assert ("mercedes-benz",) in set(model.random_effects("vehicle:make").table["level"])


def test_paths_that_join_to_the_same_text_stay_two_levels() -> None:
    """Make "A-B" with model "C" and make "A" with model "B-C" are two models.

    Mutation: keying a path by ``"-".join(parts)`` merges the two into one
    level, and the two rows predict alike.
    """
    rng = np.random.default_rng(4)
    n = 900
    cell = rng.integers(0, 6, n)
    # Three makes with two models each; "A-B"/"C" and "A"/"B-C" both join to "A-B-C".
    makes = np.array(["A-B", "A-B", "A", "A", "D", "D"])
    models = np.array(["C", "X", "B-C", "X", "C", "X"])
    frame = pd.DataFrame({"x": rng.normal(size=n), "make": makes[cell], "model": models[cell]})
    log_mean = np.array([0.9, 0.0, -0.9, 0.0, 0.1, -0.1])[cell]
    y = rng.poisson(np.exp(log_mean)).astype(float)
    model = _fit(Hierarchy(columns=("make", "model")), frame, y)
    levels = set(model.random_effects("vehicle:model").table["level"])
    assert {("A-B", "C"), ("A", "B-C")} <= levels and len(levels) == 6
    rows = frame.loc[[int(np.flatnonzero(cell == 0)[0]), int(np.flatnonzero(cell == 2)[0])]]
    predicted = model.predict(rows.assign(x=0.0))
    assert predicted[0] > 1.5 * predicted[1]


def test_normalize_merges_spellings_reports_them_and_applies_at_predict() -> None:
    """Case, width and spacing variants are one label at fit and at prediction.

    Mutation: folding the labels only at fit leaves ``"ＦＯＲＤ"`` unseen at
    prediction, so it predicts at the population value.
    """
    frame, y = _book()
    frame = frame.drop(columns="vehicle")
    spelled = frame["make"].to_numpy(dtype=object)
    ford = np.flatnonzero(spelled == "ford")
    spelled[ford[::3]] = "Ford"
    spelled[ford[1::3]] = "  FORD "
    frame["make"] = spelled
    with pytest.warns(
        UserWarning, match=r"merged 1 group.*make 'Ford' <- '  FORD ', 'Ford', 'ford'"
    ):
        model = _fit(Hierarchy(columns=("make", "model", "year"), normalize=True), frame, y)
    merged = model.reconstruct_feature("vehicle")["merged_labels"]
    # "Ford" holds the most rows (or ties and is seen first), so it labels the make.
    assert merged == {"make": {"Ford": ("  FORD ", "Ford", "ford")}}
    assert ("Ford",) in set(model.random_effects("vehicle:make").table["level"])
    row = frame.iloc[[ford[2]]].assign(x=0.0)
    folded = row.assign(make="\uff26\uff2f\uff32\uff24\u00a0")  # full-width FORD, no-break space
    np.testing.assert_array_equal(model.predict(folded), model.predict(row))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        raw = _fit(Hierarchy(columns=("make", "model", "year")), frame, y)
    assert {("ford",), ("Ford",), ("  FORD ",)} <= set(
        raw.random_effects("vehicle:make").table["level"]
    )


def test_normalize_reports_the_spelling_most_rows_use() -> None:
    """Labels keep the data's spelling; only the comparison is folded.

    Mutation: keying the levels by the folded form reports ``("ford",)`` for a
    column that only ever says ``"Ford"``, and ``("ford",)`` again once a few
    rows say ``"FORD "``.
    """
    frame, y = _book()
    frame = frame.drop(columns="vehicle").assign(
        make=lambda f: f["make"].str.title(), model=lambda f: f["model"].str.upper()
    )
    spec = Hierarchy(columns=("make", "model", "year"), normalize=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)  # nothing merges, so nothing warns
        clean = _fit(spec, frame, y)
    labels = set(clean.random_effects("vehicle:model").table["level"])
    assert ("Ford", "ALPHA") in labels and ("ford", "alpha") not in labels
    assert clean.reconstruct_feature("vehicle")["merged_labels"] == {}

    spelled = frame["make"].to_numpy(dtype=object)
    spelled[np.flatnonzero(spelled == "Ford")[::5]] = "FORD "  # a fifth, so "Ford" leads
    with pytest.warns(UserWarning, match=r"make 'Ford' <- 'FORD ', 'Ford'"):
        merged = _fit(
            Hierarchy(columns=("make", "model", "year"), normalize=True),
            frame.assign(make=spelled),
            y,
        )
    assert ("Ford",) in set(merged.random_effects("vehicle:make").table["level"])
    row = frame.head(1).assign(x=0.0, make="Ford")
    np.testing.assert_array_equal(merged.predict(row.assign(make="ｆｏｒｄ")), merged.predict(row))


def test_an_unseen_path_takes_the_deepest_level_it_shares() -> None:
    """A new model of a known make gets the make; a new make gets the population.

    Mutation: zeroing a row's whole contribution when its full path is unseen
    gives the new model the population value instead of its make's.
    """
    frame, y = _book()
    model = _fit(_joined(), frame, y)
    new = pd.DataFrame({"x": [0.0, 0.0], "vehicle": ["kia-omega-2019", "tesla-alpha-2019"]})
    conditional = np.log(model.predict(new))
    population = np.log(model.predict(new, random_effects="population"))
    make_effect = model.reconstruct_feature("vehicle")["level_effects"]["make"][("kia",)]
    # Each side is a few rounded sums of O(1) terms, an exp and a log.
    scale = 1.0 + np.abs(conditional).max() + np.abs(population).max()
    np.testing.assert_allclose(
        conditional - population, [make_effect, 0.0], rtol=0, atol=_gamma(16) * scale
    )
    assert abs(make_effect) > 1e-3
    strict = _fit(Hierarchy(names=("make", "model", "year"), sep="-", unseen="error"), frame, y)
    with pytest.raises(ValueError, match="unseen Hierarchy paths"):
        strict.predict(new)


def test_weighted_unseen_paths_take_the_weighted_mean_under_their_deepest_known_part() -> None:
    """``unseen="weighted"``: a new level is priced as its group's average exposure.

    A new model of a known make takes, at the model and year levels, the
    weighted mean of that make's effects there; a new year of a known model
    the mean of that model's years; a new make the book's means, which add
    up to the weighted mean of every training path's total.  Mutations:
    equal shares (the average level) or the book-wide mean for a known make
    both miss the expected values.
    """
    frame, y = _book()
    weight = np.random.default_rng(3).uniform(0.2, 2.0, len(frame))
    model = SuperGLM(
        family="poisson",
        features={
            "x": Numeric(),
            "vehicle": Hierarchy(names=("make", "model", "year"), sep="-", unseen="weighted"),
        },
        selection_penalty=0,
    ).fit_reml(frame, y, sample_weight=weight)
    levels = model.reconstruct_feature("vehicle")["level_effects"]
    paths = list(zip(frame["make"], frame["model"], frame["year"], strict=True))
    by_row = {
        "model": np.array([levels["model"][path[:2]] for path in paths]),
        "year": np.array([levels["year"][path] for path in paths]),
    }

    def mean(level: str, rows: np.ndarray) -> float:
        return float(np.sum(weight[rows] * by_row[level][rows]) / np.sum(weight[rows]))

    kia = frame["make"].to_numpy() == "kia"
    kia_beta = kia & (frame["model"].to_numpy() == "beta")
    everyone = np.ones(len(frame), dtype=bool)
    new = pd.DataFrame(
        {"x": 0.0, "vehicle": ["kia-omega-2019", "kia-beta-2031", "tesla-alpha-2019"]}
    )
    parts = model.term_contributions(new)
    make_mean = float(
        np.sum(weight * np.array([levels["make"][path[:1]] for path in paths])) / np.sum(weight)
    )
    expected = np.array(
        [
            [levels["make"][("kia",)], mean("model", kia), mean("year", kia)],
            [levels["make"][("kia",)], levels["model"][("kia", "beta")], mean("year", kia_beta)],
            [make_mean, mean("model", everyone), mean("year", everyone)],
        ]
    )
    got = parts[["vehicle:make", "vehicle:model", "vehicle:year"]].to_numpy()
    # A weighted mean of O(1) effects over n rows, against the same in n roundings.
    np.testing.assert_allclose(got, expected, rtol=0, atol=_gamma(2 * len(frame)))
    totals = model.reconstruct_feature("vehicle")["effects"]
    book = float(np.sum(weight * np.array([totals[path] for path in paths])) / np.sum(weight))
    assert abs(got[2].sum() - book) <= _gamma(4 * len(frame))
    assert abs(mean("model", kia) - mean("model", everyone)) > 1e-3  # the prefix matters here
    report = model.random_effects("vehicle:model")
    assert report.unseen_effect == pytest.approx(mean("model", everyone), abs=_gamma(len(frame)))
    population = _fit(_joined(), frame, y)
    assert population.random_effects("vehicle:model").unseen_effect == 0.0
    assert population.random_effects("vehicle:model").unseen_se == 0.0


def test_term_contributions_split_the_predictor_by_term_and_by_hierarchy_level() -> None:
    """One column per term and per hierarchy level, summing with the intercept to the predictor.

    Each column is its term's own log relativity: the hierarchy levels' fitted
    effects and the ``relativities()`` tables of the categorical and of the
    ordered band, whose base level reads zero there and its effect sits in
    the intercept.  Mutations: one combined ``"vehicle"`` column, the term's
    ``score``, still sums to the predictor but fails the per-level columns;
    the band's raw score without its base shift fails its table.
    """
    from superglm import Categorical, OrderedCategorical, Spline

    frame, y = _book()
    bands = [f"p{i}" for i in range(6)]
    frame["power"] = np.array(bands)[np.random.default_rng(8).integers(0, 6, len(frame))]
    band = OrderedCategorical(
        values={b: float(i) for i, b in enumerate(bands)}, basis=Spline(kind="cr", k=4)
    )
    model = _fit(_joined(), frame, y, engine=Categorical(), power=band)
    rows = frame.iloc[::50].copy()
    rows.index = rows.index + 1000  # a pandas index is carried through
    parts = model.term_contributions(rows)
    assert list(parts.columns) == [
        "Intercept",
        "x",
        "vehicle:make",
        "vehicle:model",
        "vehicle:year",
        "engine",
        "power",
    ]
    assert parts.index.equals(rows.index)
    eta = np.log(model.predict(rows))
    # Six O(1) terms summed twice in different orders, an exp and a log.
    scale = np.abs(parts.to_numpy()).sum(axis=1) + np.abs(eta)
    np.testing.assert_allclose(parts.sum(axis=1), eta, rtol=0, atol=float(_gamma(16) * scale.max()))
    levels = model.reconstruct_feature("vehicle")["level_effects"]
    for (_, row), make, model_name, year in zip(
        parts.iterrows(), rows["make"], rows["model"], rows["year"], strict=True
    ):
        assert row["vehicle:make"] == levels["make"][(make,)]
        assert row["vehicle:model"] == levels["model"][(make, model_name)]
        assert row["vehicle:year"] == levels["year"][(make, model_name, year)]
    tables = model.relativities()
    engine = tables["engine"].set_index("level")["log_relativity"]
    np.testing.assert_array_equal(parts["engine"], engine[rows["engine"]].to_numpy())
    power = tables["power"].set_index("level")["log_relativity"]
    expected = power[rows["power"]].to_numpy()
    # The same curve value less the same base value, in one and two roundings.
    atol = _gamma(4) * (np.abs(expected).max() + np.abs(power).max())
    np.testing.assert_allclose(parts["power"], expected, rtol=0, atol=atol)
    assert np.abs(power).max() > 1e-3 and np.min(np.abs(parts["power"])) < atol  # base reads 0
    population = model.term_contributions(rows, random_effects="population")
    vehicle = ["vehicle:make", "vehicle:model", "vehicle:year"]
    assert not population[vehicle].to_numpy().any()
    pd.testing.assert_frame_equal(population.drop(columns=vehicle), parts.drop(columns=vehicle))


def test_a_crossed_effect_shares_its_parent_level_instead_of_refitting_it() -> None:
    """``(1|make/model/year) + (1|make:engine)``: make is fitted once.

    Mutation: without ``nested_in`` sharing, ``make_engine`` fits its own make
    level, which the duplicate check refuses.
    """
    frame, y = _book()
    frame = frame.drop(columns="vehicle")
    shared = Hierarchy(columns=("make", "engine"), nested_in="vehicle:make")
    model = _fit(Hierarchy(columns=("make", "model", "year")), frame, y, make_engine=shared)
    names = [g.name for g in model._groups]
    assert names[-1] == "make_engine:engine" and "make_engine:make" not in names
    assert len(model.random_effects("make_engine:engine").table) == len(MAKES) * 3
    with pytest.raises(
        ValueError, match="group the training rows identically.*nested_in='vehicle:make'"
    ):
        _fit(
            Hierarchy(columns=("make", "model", "year")),
            frame,
            y,
            make_engine=Hierarchy(columns=("make", "engine")),
        )
    with pytest.raises(ValueError, match="must group the training rows exactly as 'vehicle:make'"):
        _fit(
            Hierarchy(columns=("make", "model", "year")),
            frame,
            y,
            make_engine=Hierarchy(columns=("engine", "make"), nested_in="vehicle:make"),
        )


def test_a_random_effect_nested_in_a_hierarchy_level_is_checked_and_used() -> None:
    """A declared hierarchy level is validated on the rows and leads the chain.

    The leaf (make x engine, 20 levels) nests in ``vehicle:make`` (4 levels)
    and in ``engine_family`` (8 levels, crossed with make); Rule B alone takes
    the heavier ``engine_family``.  Mutation: resolving the declaration by
    feature name misses ``"vehicle:make"`` and keeps ``engine_family``.
    """
    frame, y = _book()
    frame = frame.drop(columns="vehicle")
    engine = np.random.default_rng(5).integers(0, 5, len(frame))
    frame["make_engine"] = frame["make"] + "|" + pd.Series(engine).astype(str)
    make_index = frame["make"].map({make: i for i, make in enumerate(MAKES)})
    frame["engine_family"] = "f" + ((make_index + engine) % 8).astype(str)
    chains = {}
    for declared in (None, "vehicle:make"):
        model = SuperGLM(
            family="poisson",
            features={
                "x": Numeric(),
                "vehicle": Hierarchy(columns=("make", "model")),
                "engine_family": RandomEffect(),
                "make_engine": RandomEffect(nested_in=declared),
            },
            selection_penalty=0,
            direct_solve="structured",
        ).fit_reml(frame, y)
        chains[declared] = model._reml_profile["structured_chain"]
    assert chains == {
        None: ("engine_family", "make_engine"),
        "vehicle:make": ("vehicle:make", "make_engine"),
    }
    broken = frame.copy()
    broken.loc[5, "make_engine"] = broken.loc[
        broken["make"] != broken.loc[5, "make"], "make_engine"
    ].iloc[0]
    with pytest.raises(ValueError, match=r"nested_in='vehicle:make'.*\(row 5\)"):
        validate_declared_nesting(
            {
                "vehicle": Hierarchy(columns=("make", "model")),
                "make_engine": RandomEffect(nested_in="vehicle:make"),
            },
            lambda name: broken[name].to_numpy(),
        )


def _gamma(k: int) -> float:
    """Higham's ``gamma_k = k u / (1 - k u)`` for float64."""
    u = np.finfo(np.float64).eps / 2
    return k * u / (1 - k * u)


def test_relativities_combine_every_level_and_carry_the_path_error() -> None:
    """A path's relativity is the product of its levels'; its error is ``sqrt(m' V m)``.

    Fitted on the structured backend, so the error comes from solves with the
    retained factor, as on a large book, and is checked against the formed
    covariance.  Mutation: reporting a path's error from the leaf
    coefficient's variance alone drops the make and model variances and
    covariances.
    """
    frame, y = _book()
    model = SuperGLM(
        family="poisson",
        features={"x": Numeric(), "vehicle": _joined()},
        selection_penalty=0,
        direct_solve="structured",
    ).fit_reml(frame, y)
    report = model.reconstruct_feature("vehicle")
    levels = report["level_effects"]
    path = ("kia", "beta", "2019")
    parts = [levels["make"][path[:1]], levels["model"][path[:2]], levels["year"][path]]
    # The same three numbers, summed in at most two roundings either way.
    assert abs(report["effects"][path] - sum(parts)) <= 2 * _gamma(2) * sum(map(abs, parts))
    table = model.relativities(with_se=True)["vehicle"]
    row = table.loc[table["level"] == path].iloc[0]
    assert row["log_relativity"] == report["effects"][path]
    covariance, active = model._coef_covariance
    assert hasattr(covariance, "solve")
    groups = {g.name: g for g in active}
    contrast = np.zeros(covariance.shape[0])
    for depth, level in enumerate(("make", "model", "year")):
        index = model._specs["vehicle"]._path_codes[depth][path[: depth + 1]]
        contrast[groups[f"vehicle:{level}"].start + index] = 1.0
    dense = np.asarray(covariance, dtype=np.float64)
    expected = np.sqrt(contrast @ dense @ contrast)
    # A solve with the factor and the block formed from it agree to a
    # backward-stable solve's forward error, p u times the condition number.
    tolerance = _gamma(dense.shape[0]) * np.linalg.cond(dense)
    assert row["se_log_relativity"] == pytest.approx(expected, rel=tolerance)
    assert tolerance < 1e-6  # the bound is informative on this fixture


def test_a_fitted_hierarchy_survives_pickle_and_its_configuration_refits() -> None:
    """Pickling keeps the paths and folding; the captured configuration refits the same model.

    Mutation: pickling that drops ``normalize`` (deep copies kept exact) leaves
    the title-case spelling unseen after loading.
    """
    frame, y = _book()
    frame = frame.drop(columns="vehicle")
    frame.loc[::5, "make"] = frame.loc[::5, "make"].str.upper()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model = _fit(Hierarchy(columns=("make", "model", "year"), normalize=True), frame, y)
        loaded = pickle.loads(pickle.dumps(model))
        refit = SuperGLM(**model._config.constructor_kwargs()).fit_reml(frame, y)
    probe = frame.head(20).assign(make=lambda f: f["make"].str.title())
    np.testing.assert_array_equal(loaded.predict(probe), model.predict(probe))
    np.testing.assert_array_equal(refit.predict(probe), model.predict(probe))
    assert (
        loaded.random_effects("vehicle:year").tau_squared
        == model.random_effects("vehicle:year").tau_squared
    )


def test_declarations_that_cannot_be_read_unambiguously_are_refused() -> None:
    """Exactly one of ``sep``/``columns``; names that would alias; levels to fit."""
    with pytest.raises(TypeError, match="levels"):
        Hierarchy(levels=("make", "model"), sep="-")  # type: ignore[call-arg]  # names= now
    with pytest.raises(ValueError, match="unseen must be"):
        Hierarchy(columns=("make", "model"), unseen="mean")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="exactly one of sep="):
        Hierarchy(names=("make", "model"), sep="-", columns=("make", "model"))
    with pytest.raises(ValueError, match="needs names="):
        Hierarchy(sep="-")
    with pytest.raises(ValueError, match="RandomEffect"):
        Hierarchy(columns=("make",))
    with pytest.raises(ValueError, match="may not contain ':'"):
        Hierarchy(names=("make", "a:b"), sep="-")
    frame, y = _book()
    one_model = frame.assign(model=frame["make"] + "-only")
    with pytest.warns(UserWarning, match="'vehicle:make' and 'vehicle:model' group the training"):
        _fit(Hierarchy(columns=("make", "model", "year")), one_model.drop(columns="vehicle"), y)
    with pytest.raises(ValueError, match="no deeper level to fit"):
        _fit(
            _joined(),
            frame,
            y,
            other=Hierarchy(columns=("make", "model"), nested_in="vehicle:model"),
        )
    with pytest.raises(ValueError, match="random-effect level this model fits"):
        _fit(_joined(), frame, y, other=Hierarchy(columns=("make", "engine"), nested_in="vehicle"))
    with pytest.raises(NotImplementedError, match="Hierarchy feature.*fit_reml"):
        SuperGLM(family="poisson", features={"vehicle": _joined()}).fit(frame, y)
