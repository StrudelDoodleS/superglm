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


# _fit_mixed also fits a Numeric and an OrderedCategorical main, whose pairs
# still raise until tasks 5-6, so a default sweep over it cannot run yet.  Name
# the pairs whose kinds exist today; drop the candidates= argument once the
# whole sweep is implemented.
_IMPLEMENTED_PAIRS = [
    ("age", "power"),
    ("age", "region"),
    ("age", "brand"),
    ("power", "region"),
    ("power", "brand"),
    ("region", "brand"),
]


def test_cat_cat_planted_table_ranks_first_with_exact_df():
    df, rng = _mixed_frame(n=20000, seed=3)
    boost = ((df["region"] == "B") & (df["brand"] == "B2")).astype(float)
    y = rng.poisson(np.exp(-1.3 + 0.5 * boost)).astype(np.float64)
    model = _fit_mixed(df, y)
    table = model.screen_interactions(df, y, candidates=_IMPLEMENTED_PAIRS)
    top = table.iloc[0]
    assert {top["feature_a"], top["feature_b"]} == {"region", "brand"}
    assert top["kind"] == "cat_cat"
    # (L1-1)(L2-1) = 3*2; unpenalized rung reports the achieved rank as edf0
    assert top["edf0"] == pytest.approx(6.0, abs=0.26)
    assert top["lambda0"] == 0.0
    assert top["z"] > 10.0
    row = table[table["kind"] == "cat_cat"].iloc[0]
    assert row["n_cells"] == 4 * 3
    assert not row["approx"]


def _planted_bend(seed=4):
    """One region's age curve bends away from the shared main effect.

    The age main absorbs the population average of the bump, so only the
    across-level DEVIATION is screenable — the amplitude below leaves that
    deviation well clear of the noise floor at this n.
    """
    df, rng = _mixed_frame(n=20000, seed=seed)
    bend = np.where(df["region"] == "C", np.sin((df["age"] - 18.0) / 62.0 * np.pi) * 0.8, 0.0)
    return df, rng.poisson(np.exp(-1.3 + bend)).astype(np.float64)


def test_spline_cat_planted_deviation_curve_ranks_first():
    df, y = _planted_bend()
    model = _fit_mixed(df, y)
    table = model.screen_interactions(df, y, candidates=_IMPLEMENTED_PAIRS)
    top = table.iloc[0]
    assert {top["feature_a"], top["feature_b"]} == {"age", "region"}
    assert top["kind"] == "spline_cat"
    assert top["z"] > 8.0


def test_spline_cat_flags_approx_when_only_its_spline_margin_bins_lossily():
    """A SplineCategorical refit discretizes on a SINGLE-parent gate: its spline
    margin bins whenever that parent does, with no say from the factor.  So a
    spline_cat row must flag approx exactly when the spline margin's
    cardinality overruns its resolved bin count -- the same probe-vs-refit
    basis gap ti flags, which a both-parents rule would silently miss."""
    rng = np.random.default_rng(21)
    n = 4000
    df = pd.DataFrame(
        {
            # 40 distinct ages: binned exactly at 256 bins, lossily at 8
            "age": rng.integers(20, 60, n).astype(float),
            "power": rng.uniform(20.0, 200.0, n),
            "region": rng.choice(list("ABCD"), n),
            "brand": rng.choice(["B1", "B2", "B3"], n),
        }
    )
    y = rng.poisson(np.exp(-1.5 + 0.01 * df["age"])).astype(np.float64)

    def flags(**kwargs):
        model = SuperGLM(
            family="poisson",
            features={
                "age": Spline(kind="ps", n_knots=6),
                "power": Spline(kind="ps", n_knots=6),
                "region": Categorical(),
                "brand": Categorical(),
            },
            **kwargs,
        )
        model.fit_reml(df, y)
        table = model.screen_interactions(df, y)
        assert not table["approx"].isna().any()
        return {
            frozenset((row.feature_a, row.feature_b)): (row.kind, bool(row.approx))
            for row in table.itertuples()
        }

    age_region = frozenset(("age", "region"))
    power_region = frozenset(("power", "region"))
    two_factors = frozenset(("region", "brand"))

    # 8 bins: 40 distinct ages bin lossily, so the refit basis differs
    lossy = flags(discrete=True, n_bins=8)
    assert lossy[age_region] == ("spline_cat", True)
    assert lossy[power_region] == ("spline_cat", True)
    assert lossy[two_factors] == ("cat_cat", False)

    # 256 bins: age's 40 values bin exactly, power's 4000 do not
    partly = flags(discrete=True, n_bins=256)
    assert partly[age_region] == ("spline_cat", False)
    assert partly[power_region] == ("spline_cat", True)
    assert partly[two_factors] == ("cat_cat", False)

    # no fit-time discretization at all: no refit binning, nothing to flag
    off = flags(discrete=False)
    assert off[age_region] == ("spline_cat", False)
    assert off[power_region] == ("spline_cat", False)
    assert off[two_factors] == ("cat_cat", False)


def test_mixed_pair_order_does_not_leak_into_the_row():
    """A spline_cat pair is assembled with the categorical margin LAST, whichever
    order the caller names it in.  That swap is a column permutation the
    statistic is invariant to, and it must reach neither the reported columns
    nor any number in the row."""
    df, y = _planted_bend()
    model = _fit_mixed(df, y)
    fwd = model.screen_interactions(df, y, candidates=[("age", "region")]).iloc[0]
    rev = model.screen_interactions(df, y, candidates=[("region", "age")]).iloc[0]
    assert (fwd["feature_a"], fwd["feature_b"]) == ("age", "region")
    assert (rev["feature_a"], rev["feature_b"]) == ("region", "age")
    for column in ("kind", "statistic", "z", "edf0", "lambda0", "n_cells", "approx"):
        assert fwd[column] == rev[column], column


def test_two_level_factor_pairs_are_legal():
    df, rng = _mixed_frame(n=6000, seed=12)
    df = df.assign(fuel=rng.choice(["diesel", "petrol"], len(df)))
    y = _null_y(df, rng)
    model = SuperGLM(
        family="poisson",
        features={"fuel": Categorical(), "brand": Categorical()},
    )
    model.fit_reml(df, y)
    table = model.screen_interactions(df, y)
    row = table.iloc[0]
    assert row["kind"] == "cat_cat"
    assert row["edf0"] == pytest.approx(2.0, abs=0.26)  # (2-1)*(3-1)
    assert np.isfinite(row["z"])


def test_spline_cat_confirms_by_refit():
    """The pair the screen ranks first is real: the SplineCategorical refit
    the `spline_cat` kind names finds it in the likelihood too."""
    df, y = _planted_bend()
    base = _fit_mixed(df, y)
    dev0 = base._result.deviance
    confirm = _fit_mixed(df, y, interactions=[("age", "region")])
    assert dev0 - confirm._result.deviance > 50.0
