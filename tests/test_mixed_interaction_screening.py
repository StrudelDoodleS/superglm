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


def test_oc_margin_screens_beside_its_spline_siblings():
    """An OC main screens as a spline margin on its MAPPED level values, so it
    pairs with a plain spline as a `ti` -- reading the label column through the
    parent resolution rather than failing a float cast on it -- and computing it
    must leave the pure-spline pairs of the same model untouched."""
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

    alone = model.screen_interactions(df, y, candidates=[("age", "power")])
    assert list(alone["kind"]) == ["ti"]
    assert np.isfinite(alone["z"]).all()

    oc_pair = model.screen_interactions(df, y, candidates=[("age", "band")]).iloc[0]
    assert oc_pair["kind"] == "ti"
    assert np.isfinite(oc_pair["z"])
    # the band margin grids on its 5 mapped score points, age on its own support
    assert int(oc_pair["n_cells"]) == 5 * len(np.unique(df["age"]))

    swept = {
        frozenset((row.feature_a, row.feature_b)): row
        for row in model.screen_interactions(df, y).itertuples()
    }
    assert {pair: row.kind for pair, row in swept.items()} == {
        frozenset(("age", "power")): "ti",
        frozenset(("age", "band")): "ti",
        frozenset(("power", "band")): "ti",
    }
    # The OC margin shares the sweep's caches with its siblings without moving
    # them: the pure-spline pair scores exactly what it scores on its own.
    assert swept[frozenset(("age", "power"))].z == pytest.approx(alone["z"].iloc[0])
    assert swept[frozenset(("age", "band"))].z == pytest.approx(oc_pair["z"])


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


def test_cat_cat_planted_table_ranks_first_with_exact_df():
    df, rng = _mixed_frame(n=20000, seed=3)
    boost = ((df["region"] == "B") & (df["brand"] == "B2")).astype(float)
    y = rng.poisson(np.exp(-1.3 + 0.5 * boost)).astype(np.float64)
    model = _fit_mixed(df, y)
    table = model.screen_interactions(df, y)
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
    table = model.screen_interactions(df, y)
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
    """A spline_cat pair is assembled with the categorical margin LAST, and a
    numeric_cat pair resolves which margin carries the slope by KIND rather
    than by argument position -- whichever order the caller names them in.
    Either reordering is a column permutation the statistic is invariant to,
    and it must reach neither the reported columns nor any number in the row."""
    df, y = _planted_bend()
    model = _fit_mixed(df, y)
    for a, b in (("age", "region"), ("bm", "region")):
        fwd = model.screen_interactions(df, y, candidates=[(a, b)]).iloc[0]
        rev = model.screen_interactions(df, y, candidates=[(b, a)]).iloc[0]
        assert (fwd["feature_a"], fwd["feature_b"]) == (a, b)
        assert (rev["feature_a"], rev["feature_b"]) == (b, a)
        for column in ("kind", "statistic", "z", "edf0", "lambda0", "n_cells", "approx"):
            assert fwd[column] == rev[column], (a, b, column)


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


def test_numeric_cat_planted_slope_ranks_first_with_exact_df():
    df, rng = _mixed_frame(n=20000, seed=5)
    slope = np.where(df["region"] == "D", 0.35, 0.0)
    y = rng.poisson(np.exp(-1.6 + slope * (df["bm"] - 1.0))).astype(np.float64)
    model = _fit_mixed(df, y)
    table = model.screen_interactions(df, y)
    top = table.iloc[0]
    assert {top["feature_a"], top["feature_b"]} == {"bm", "region"}
    assert top["kind"] == "numeric_cat"
    assert top["edf0"] == pytest.approx(3.0, abs=0.26)  # L-1 with L=4
    assert top["n_cells"] == 4
    # A numeric margin has no basis to bin and none to discretize, so the
    # refit sees the probe's own columns: numeric kinds are never approximate.
    assert not top["approx"]
    assert top["z"] > 6.0
    # The pair's budget scales with the FACTOR's width, never with n: a
    # 4-level pair over 20000 rows computes unchanged at a cell budget that
    # would refuse any gridded pair outright.
    tiny = model.screen_interactions(df, y, candidates=[("bm", "region")], max_cells=100).iloc[0]
    assert tiny["z"] == pytest.approx(top["z"])
    assert tiny["n_cells"] == 4


def test_numeric_numeric_planted_product_ranks_first():
    df, rng = _mixed_frame(n=20000, seed=6)
    df = df.assign(dens=rng.uniform(0.0, 1.0, len(df)))
    y = rng.poisson(np.exp(-1.6 + 0.3 * (df["bm"] - 1.25) * (df["dens"] - 0.5))).astype(np.float64)
    model = SuperGLM(
        family="poisson",
        features={
            "age": Spline(kind="ps", n_knots=6),
            "region": Categorical(),
            "brand": Categorical(),
            "bm": Numeric(),
            "dens": Numeric(),
        },
    )
    model.fit_reml(df, y)
    table = model.screen_interactions(df, y)
    top = table.iloc[0]
    assert top["kind"] == "numeric_numeric"
    assert {top["feature_a"], top["feature_b"]} == {"bm", "dens"}
    assert top["edf0"] == pytest.approx(1.0, abs=0.01)
    assert top["n_cells"] == 1
    assert not top["approx"]
    assert top["z"] > 5.0


def test_numeric_cat_refuses_a_factor_too_wide_for_its_blocks():
    """A z-moment pair has no grid to bin, so it cannot approximate: an
    unaffordable numeric_cat pair is REFUSED, never degraded.  Every block it
    builds scales with the factor's width and the largest is the (L+1)-wide
    overlap curvature, so the gate is `(L+1)**2 <= max_cells` -- applied to
    the level count alone, before the dense (L, L-1) menu is ever built."""
    L, reps = 2300, 2
    rng = np.random.default_rng(31)
    df = pd.DataFrame(
        {
            "g": np.repeat([f"L{i}" for i in range(L)], reps),
            "bm": rng.uniform(0.5, 2.0, L * reps),
        }
    )
    # A Gaussian fit keeps this wide-factor model cheap to build; the gate
    # reads level counts, so the family it was fitted with is immaterial.
    y = rng.normal(size=len(df))
    model = SuperGLM(family="gaussian", features={"g": Categorical(), "bm": Numeric()})
    model.fit_reml(df, y)

    refused = model.screen_interactions(df, y).iloc[0]  # default max_cells
    assert refused["kind"] == "numeric_cat"
    assert np.isnan(refused["statistic"]) and np.isnan(refused["z"])
    assert refused["n_cells"] == L  # the grid it was refused for
    assert not refused["approx"]  # refusal is not approximation

    # One cell short of the block budget is still a refusal ...
    short = model.screen_interactions(df, y, max_cells=(L + 1) ** 2 - 1).iloc[0]
    assert np.isnan(short["z"])
    assert short["n_cells"] == L
    # ... and at the budget the same pair computes, exactly and unpenalized.
    lifted = model.screen_interactions(df, y, max_cells=(L + 1) ** 2).iloc[0]
    assert np.isfinite(lifted["z"])
    assert lifted["edf0"] == pytest.approx(L - 1, abs=0.26)  # achieved rank
    assert lifted["lambda0"] == 0.0
    assert lifted["n_cells"] == L
    assert not lifted["approx"]


def test_oc_margin_screens_as_spline_and_confirms():
    df, rng = _mixed_frame(n=20000, seed=8)
    band_idx = df["band"].map({b: i for i, b in enumerate(BANDS)}).to_numpy()
    ramp = (band_idx / 4.0) * (df["power"] - 110.0) / 90.0 * 0.35
    y = rng.poisson(np.exp(-1.5 + ramp)).astype(np.float64)
    model = _fit_mixed(df, y)
    table = model.screen_interactions(df, y)
    top = table.iloc[0]
    assert {top["feature_a"], top["feature_b"]} == {"band", "power"}
    assert top["kind"] == "ti"
    assert top["z"] > 6.0
    # The ti() refit the kind names finds the same ramp in the likelihood.
    # Measured 34.8 on 5 extra parameters at this amplitude; the OC-parented
    # refit's deviance is bit-identical to the same fit on the mapped scores,
    # so the margin under test is exact and the bound is just a floor.
    confirm = _fit_mixed(df, y, interactions=[("band", "power")])
    assert model._result.deviance - confirm._result.deviance > 30.0


def test_oc_cat_pair_is_spline_cat_kind():
    df, rng = _mixed_frame(n=8000, seed=9)
    y = _null_y(df, rng)
    model = _fit_mixed(df, y)
    table = model.screen_interactions(df, y, candidates=[("band", "region")])
    assert list(table["kind"]) == ["spline_cat"]
    assert np.isfinite(table["z"]).all()
    # 5 score points x 4 levels
    assert int(table["n_cells"].iloc[0]) == 5 * 4


def test_oc_cat_planted_deviation_confirms():
    df, rng = _mixed_frame(n=20000, seed=13)
    band_idx = df["band"].map({b: i for i, b in enumerate(BANDS)}).to_numpy()
    bend = np.where(df["region"] == "A", (band_idx / 4.0 - 0.5) * 0.5, 0.0)
    y = rng.poisson(np.exp(-1.4 + bend)).astype(np.float64)
    model = _fit_mixed(df, y)
    table = model.screen_interactions(df, y)
    top = table.iloc[0]
    assert {top["feature_a"], top["feature_b"]} == {"band", "region"}
    assert top["kind"] == "spline_cat"
    assert top["z"] > 6.0
    confirm = _fit_mixed(df, y, interactions=[("band", "region")])
    assert model._result.deviance - confirm._result.deviance > 30.0


def test_oc_pairs_never_flag_approx_under_discrete_mode():
    """An OC-parented refit refuses fit-time discretization outright, whatever
    the model flag says and whatever its inner spline would say alone, so its
    refit basis cannot drift from the probe's: the row stays exact at a bin
    count that flags every plain spline margin of the same model."""
    df, rng = _mixed_frame(n=6000, seed=14)
    y = _null_y(df, rng)
    # 4 bins < 5 score points: were the OC margin consulted as a bare spline,
    # its support would bin LOSSILY and the pair would be flagged approximate.
    model = _fit_mixed(df, y, discrete=True, n_bins=4)
    table = model.screen_interactions(df, y)
    flags = {
        frozenset((row.feature_a, row.feature_b)): (row.kind, bool(row.approx))
        for row in table.itertuples()
    }
    assert flags[frozenset(("band", "region"))] == ("spline_cat", False)
    assert flags[frozenset(("band", "brand"))] == ("spline_cat", False)
    assert flags[frozenset(("band", "power"))] == ("ti", False)
    assert flags[frozenset(("band", "age"))] == ("ti", False)
    # ... while the same model's plain-spline pairs DO flag, so it is the OC
    # parent and not the model flag that keeps those rows exact.
    assert flags[frozenset(("age", "region"))] == ("spline_cat", True)
    assert flags[frozenset(("age", "power"))] == ("ti", True)


def test_oc_select_inner_spline_raises_upfront():
    df, rng = _mixed_frame(n=4000, seed=10)
    y = _null_y(df, rng)
    model = SuperGLM(
        family="poisson",
        features={
            "band": OrderedCategorical(
                order=BANDS, basis=Spline(kind="ps", n_knots=4, select=True)
            ),
            "power": Spline(kind="ps", n_knots=5),
        },
    )
    model.fit_reml(df, y)
    with pytest.raises(ValueError, match="select"):
        model.screen_interactions(df, y, candidates=[("band", "power")])
