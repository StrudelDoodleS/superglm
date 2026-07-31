"""Mixed-type PSST: eligibility, pair kinds, and per-kind screening."""

import numpy as np
import pandas as pd
import pytest

from superglm import Categorical, SuperGLM
from superglm.features.numeric import Numeric
from superglm.features.ordered_categorical import OrderedCategorical
from superglm.features.polynomial import Polynomial
from superglm.features.spline import Spline

BANDS = ["18-25", "26-35", "36-45", "46-55", "56+"]


def _mixed_frame(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "age": rng.uniform(18.0, 80.0, n),
            "power": rng.uniform(20.0, 200.0, n),
            "region": rng.choice(list("ABCD"), n),
            "brand": rng.choice(["B1", "B2", "B3"], n),
            "bm": rng.uniform(0.5, 2.0, n),
            "band": rng.choice(BANDS, n),
        }
    )
    return df, rng


def _fit_mixed(df, y, **kwargs):
    model = SuperGLM(
        family="poisson",
        features={
            "age": Spline(kind="ps", n_knots=6),
            "power": Spline(kind="ps", n_knots=6),
            "region": Categorical(),
            "brand": Categorical(),
            "bm": Numeric(),
            "band": OrderedCategorical(order=BANDS, basis=Spline(kind="ps", n_knots=4)),
        },
        **kwargs,
    )
    model.fit_reml(df, y)
    return model


def _null_y(df, rng):
    return rng.poisson(np.exp(-1.5 + 0.004 * df["age"]), len(df)).astype(np.float64)


@pytest.mark.xfail(reason="kinds land in tasks 4-6", strict=True)
def test_default_sweep_covers_every_eligible_kind():
    df, rng = _mixed_frame()
    y = _null_y(df, rng)
    model = _fit_mixed(df, y)
    table = model.screen_interactions(df, y)
    got = {
        (frozenset((a, b)), kind)
        for a, b, kind in zip(table["feature_a"], table["feature_b"], table["kind"])
    }
    # spot-check one pair of every kind
    assert (frozenset(("age", "power")), "ti") in got
    assert (frozenset(("age", "band")), "ti") in got  # OC screens as a spline margin
    assert (frozenset(("age", "region")), "spline_cat") in got
    assert (frozenset(("bm", "region")), "numeric_cat") in got
    assert (frozenset(("region", "brand")), "cat_cat") in got
    # numeric_numeric needs two Numerics; single bm pairs with nothing numeric
    assert not any(k == "numeric_numeric" for _, k in got)
    # deferred: spline x numeric absent from the default sweep
    assert (frozenset(("age", "bm"))) not in {p for p, _ in got}


def test_candidates_rejects_deferred_and_ineligible_kinds():
    df, rng = _mixed_frame()
    y = _null_y(df, rng)
    model = _fit_mixed(df, y)
    with pytest.raises(ValueError, match="deferred"):
        model.screen_interactions(df, y, candidates=[("age", "bm")])
    df2 = df.assign(poly=np.linspace(0.0, 1.0, len(df)))
    model2 = SuperGLM(
        family="poisson",
        features={"age": Spline(kind="ps", n_knots=6), "poly": Polynomial(degree=2)},
    )
    model2.fit_reml(df2, y)
    # A fitted-but-unscreenable main is DEFERRED, and must say so: the generic
    # "screenable features" listing would send the caller hunting for a typo.
    with pytest.raises(ValueError, match="deferred"):
        model2.screen_interactions(df2, y, candidates=[("age", "poly")])
    # ... while a name the model never fitted still gets the generic message.
    with pytest.raises(ValueError, match="screenable features"):
        model2.screen_interactions(df2, y, candidates=[("age", "nope")])


# The default sweep over this mixed model still hits deferred kinds (tasks
# 4-5) and OrderedCategorical raw values (task 6); the exclusion itself is
# pinned unconditionally by the candidates= test below.
@pytest.mark.xfail(reason="kinds land in tasks 4-6", strict=True)
def test_fitted_pairs_of_every_class_are_excluded():
    df, rng = _mixed_frame()
    y = _null_y(df, rng)
    model = _fit_mixed(df, y, interactions=[("region", "brand"), ("age", "region")])
    table = model.screen_interactions(df, y)
    pairs = {frozenset((a, b)) for a, b in zip(table["feature_a"], table["feature_b"])}
    assert frozenset(("region", "brand")) not in pairs
    assert frozenset(("age", "region")) not in pairs
    with pytest.raises(ValueError, match="already fitted"):
        model.screen_interactions(df, y, candidates=[("region", "brand")])


def test_oc_margin_defers_by_name_without_breaking_its_siblings():
    """An OC main screens as a spline margin, but on its MAPPED level values,
    which lands in a later task.  Until then the pair must name that cause
    instead of failing a float cast on the label column -- and it must not
    poison the pure-spline pairs of the same model."""
    df, rng = _mixed_frame()
    y = _null_y(df, rng)
    model = SuperGLM(
        family="poisson",
        features={
            "age": Spline(kind="ps", n_knots=6),
            "power": Spline(kind="ps", n_knots=6),
            "band": OrderedCategorical(order=BANDS, basis=Spline(kind="ps", n_knots=4)),
        },
    )
    model.fit_reml(df, y)

    table = model.screen_interactions(df, y, candidates=[("age", "power")])
    assert list(table["kind"]) == ["ti"]
    assert np.isfinite(table["z"]).all()

    with pytest.raises(NotImplementedError, match="OrderedCategorical"):
        model.screen_interactions(df, y, candidates=[("age", "band")])


def test_fitted_pairs_of_every_class_are_rejected_as_candidates():
    """Exclusion now keys on parent_names, so it covers every interaction
    class rather than TensorInteraction alone.  The candidates= path proves
    the fitted-pair set without needing the deferred kinds to compute, so it
    stands in for the default-sweep half of the pin above until tasks 4-6."""
    df, rng = _mixed_frame()
    y = _null_y(df, rng)
    fitted = [("age", "power"), ("age", "region"), ("region", "brand"), ("bm", "region")]
    model = _fit_mixed(df, y, interactions=fitted)
    assert {type(spec).__name__ for spec in model._interaction_specs.values()} == {
        "TensorInteraction",
        "SplineCategorical",
        "CategoricalInteraction",
        "NumericCategorical",
    }
    for pair in fitted:
        with pytest.raises(ValueError, match="already fitted"):
            model.screen_interactions(df, y, candidates=[pair])


def test_factor_smooth_pair_is_excluded():
    # The group column cannot also be a Categorical main -- the model rejects
    # that as duplicated group-intercept geometry -- so the grouping margin is
    # a spline-mode OrderedCategorical, which _margin_kind screens as a spline.
    from superglm import FactorSmooth

    df, rng = _mixed_frame()
    y = _null_y(df, rng)
    model = SuperGLM(
        family="poisson",
        features={
            "age": Spline(kind="ps", n_knots=6, m=2),
            "band": OrderedCategorical(order=BANDS, basis=Spline(kind="ps", n_knots=4)),
        },
        interactions=[FactorSmooth("age", group="band", basis="fs", kind="ps", k=5)],
        selection_penalty=0.0,
    )
    model.fit_reml(df, y)
    table = model.screen_interactions(df, y)
    pairs = {frozenset((a, b)) for a, b in zip(table["feature_a"], table["feature_b"])}
    assert frozenset(("age", "band")) not in pairs
    # Positive proof: both margins ARE screenable, so the pair is dropped as
    # already fitted rather than as ineligible.
    with pytest.raises(ValueError, match="already fitted"):
        model.screen_interactions(df, y, candidates=[("age", "band")])
