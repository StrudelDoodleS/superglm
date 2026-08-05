"""A specials OrderedCategorical is deferred by screening, and says so."""

import warnings

import numpy as np
import pandas as pd
import pytest

from superglm import Categorical, SuperGLM
from superglm.features.ordered_categorical import OrderedCategorical
from superglm.features.polynomial import Polynomial
from superglm.features.spline import Spline

BANDS = [str(i) for i in range(1, 11)]


def _specials_frame(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    band = rng.choice(BANDS, n)
    band = np.where(rng.random(n) < 0.18, "MISSING", band)
    df = pd.DataFrame(
        {
            "band": band,
            "region": rng.choice(list("ABCD"), n),
            "age": rng.uniform(18.0, 80.0, n),
        }
    )
    y = rng.poisson(np.exp(-1.5 + 0.004 * df["age"])).astype(np.float64)
    return df, y


def _specials_oc():
    return OrderedCategorical(
        order=BANDS,
        specials=["MISSING"],
        basis=Spline(kind="ps", n_knots=6),
    )


def _fit_with_specials(df, y):
    model = SuperGLM(
        family="poisson",
        features={
            "band": _specials_oc(),
            "region": Categorical(),
            "age": Spline(kind="ps", n_knots=6),
        },
    )
    model.fit_reml(df, y)
    return model


def test_a_specials_term_is_excluded_without_aborting_the_sweep():
    # FALSE TODAY: _margin_kind reads a specials OC as "spline", so the eager
    # pre-read resolves it to level scores, MISSING maps to NaN (it is a known
    # level with no entry in _level_to_value) and the WHOLE sweep dies with
    # "screen_interactions requires finite covariates; 'band' maps to
    # non-finite scores" before one statistic is computed.
    df, y = _specials_frame()
    model = _fit_with_specials(df, y)

    table = model.screen_interactions(df, y)

    pairs = {frozenset((a, b)) for a, b in zip(table["feature_a"], table["feature_b"])}
    assert pairs == {frozenset(("region", "age"))}
    assert "band" not in set(table["feature_a"]) | set(table["feature_b"])
    assert np.isfinite(table["z"]).all()


def test_the_deferred_term_and_its_reason_are_reported_on_the_table():
    # FALSE TODAY: attrs carries "phi" alone, so a term that was never screened
    # is indistinguishable from one that screened badly.
    df, y = _specials_frame()
    model = _fit_with_specials(df, y)

    table = model.screen_interactions(df, y)

    deferred = table.attrs["deferred_features"]
    assert set(deferred) == {"band"}
    assert "specials" in deferred["band"]
    assert "deferred" in deferred["band"]


def test_naming_a_specials_term_in_candidates_raises_with_the_reason():
    # FALSE TODAY: the pair is accepted and dies in the pre-read with the
    # generic "maps to non-finite scores" -- also a ValueError, so this test
    # only passes once the deferral is what refuses it.
    df, y = _specials_frame()
    model = _fit_with_specials(df, y)

    with pytest.raises(ValueError, match="no screenable margin") as excinfo:
        model.screen_interactions(df, y, candidates=[("band", "age")])
    assert "specials" in str(excinfo.value)


def test_polynomial_and_step_mode_oc_are_reported_deferred_too():
    # FALSE TODAY: both are dropped silently -- the sweep returns exactly one
    # row (region x age) and no record that two fitted mains were skipped.
    df, y = _specials_frame()
    df = df.assign(dens=np.linspace(0.0, 1.0, len(df)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        step = OrderedCategorical(order=BANDS + ["MISSING"], basis="step")
        model = SuperGLM(
            family="poisson",
            features={
                "band": step,
                "region": Categorical(),
                "age": Spline(kind="ps", n_knots=6),
                "dens": Polynomial(degree=2),
            },
        )
        model.fit_reml(df, y)

    table = model.screen_interactions(df, y)

    deferred = table.attrs["deferred_features"]
    assert set(deferred) == {"band", "dens"}
    assert "step" in deferred["band"]
    assert "Polynomial" in deferred["dens"]
    assert "age" not in deferred and "region" not in deferred


def test_a_fully_screenable_model_reports_an_empty_mapping():
    # FALSE TODAY: KeyError -- the key does not exist at all.
    df, y = _specials_frame(n=1500)
    model = SuperGLM(
        family="poisson",
        features={"region": Categorical(), "age": Spline(kind="ps", n_knots=6)},
    )
    model.fit_reml(df, y)

    table = model.screen_interactions(df, y)

    assert table.attrs["deferred_features"] == {}
