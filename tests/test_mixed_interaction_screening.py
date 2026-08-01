"""Mixed-type PSST: eligibility, pair kinds, and per-kind screening."""

import numpy as np
import pandas as pd
import pytest

from superglm import Categorical, SuperGLM
from superglm.features.numeric import Numeric
from superglm.features.ordered_categorical import OrderedCategorical
from superglm.features.polynomial import Polynomial
from superglm.features.spline import Spline

from . import _datasets

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
    # A second Numeric and a 2-level factor, so a caller that declares them as
    # features gets numeric_numeric, a 1-df numeric_cat and a 2-df cat_cat.
    # Drawn from their OWN stream: the columns above, and every draw a caller
    # takes from `rng` afterwards, are what they were before these existed.
    extra = np.random.default_rng(9000 + seed)
    df["dens"] = extra.uniform(0.0, 1.0, n)
    df["fuel"] = extra.choice(["diesel", "petrol"], n)
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
    class rather than TensorInteraction alone.  The class names are pinned
    below: a rename or a re-dispatch that quietly stopped building one of
    these four would otherwise leave the exclusion untested for it."""
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
    the level count alone, before the dense (L, L-1) menu is ever built.  The
    cubic gate applies to the same width for the same reason (the blocks are
    also FACTORIZED at that width), and at the default budget it is the
    binding one: `(L-1)**3 <= 1000 * max_cells` admits 1710 levels where the
    allocation gate admits 2235."""
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

    # One cell short of the allocation budget is still a refusal ...
    short = model.screen_interactions(df, y, max_cells=(L + 1) ** 2 - 1).iloc[0]
    assert np.isnan(short["z"])
    assert short["n_cells"] == L
    # ... and so is the allocation budget on its own: the (L-1)^3 solve the
    # blocks feed needs more than twice that, and BOTH gates must pass.
    allocation_only = model.screen_interactions(df, y, max_cells=(L + 1) ** 2).iloc[0]
    assert np.isnan(allocation_only["z"])

    budget = max((L + 1) ** 2, -(-((L - 1) ** 3) // 1000))
    one_short = model.screen_interactions(df, y, max_cells=budget - 1).iloc[0]
    assert np.isnan(one_short["z"])
    # At the budget that clears both, the same pair computes, exactly and
    # unpenalized.
    lifted = model.screen_interactions(df, y, max_cells=budget).iloc[0]
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
    # refit's deviance is identical to floating-point tolerance to the same fit
    # on the mapped scores, so the margin under test is exact and the bound is
    # just a floor.
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


# ── release gates: a pure null must not float any pair to the top ──────
#
# These are GATES, not floor measurements.  The measured per-kind noise
# maxima live in benchmarks/screening_null_floors.py, which sweeps four
# families over a wider battery; the bound of 10 below is deliberately
# generous against those maxima so a routine seed change never reds the
# suite, while a kind whose moments went wrong still trips it loudly.
# `z` is a ranking score, never a p-value: "bounded" here means the null
# sweep produced nothing a reader would mistake for signal.


@pytest.mark.parametrize("seed", range(4))
def test_mixed_null_z_stays_bounded_poisson(seed):
    df, rng = _mixed_frame(n=8000, seed=100 + seed)
    y = _null_y(df, rng)
    model = _fit_mixed(df, y)
    table = model.screen_interactions(df, y)
    # every kind of the full mixed sweep is under the gate, not just the
    # kinds that happen to survive a NaN row
    assert set(table["kind"]) == {"ti", "spline_cat", "numeric_cat", "cat_cat"}
    finite = table["z"][np.isfinite(table["z"])]
    assert len(finite) == len(table)  # no degenerate margin in this frame
    assert (finite < 10.0).all()


@pytest.mark.parametrize("seed", range(4))
def test_mixed_null_z_stays_bounded_dispersed_gaussian(seed):
    df, rng = _mixed_frame(n=8000, seed=200 + seed)
    y = rng.normal(loc=1.0 + 0.01 * df["age"], scale=10.0, size=len(df))
    model = SuperGLM(
        family="gaussian",
        features={
            "age": Spline(kind="ps", n_knots=6),
            "region": Categorical(),
            "brand": Categorical(),
            "bm": Numeric(),
            # the second Numeric and the 2-level factor put the battery's
            # heaviest-tailed configurations under a gate: numeric_numeric,
            # numeric_cat at df 1, and cat_cat at df 2
            "dens": Numeric(),
            "fuel": Categorical(),
        },
    )
    model.fit_reml(df, y)
    table = model.screen_interactions(df, y)
    # every kind of this sweep is under the gate, not just the kinds that
    # happen to survive a NaN row (no ti: one spline margin, so no pair of them)
    assert set(table["kind"]) == {"spline_cat", "numeric_cat", "cat_cat", "numeric_numeric"}
    finite = table["z"][np.isfinite(table["z"])]
    assert len(finite) == len(table)
    assert (finite < 10.0).all()


# ── real-book end-to-end sanity (skips when the parquet is absent) ─────


def _freq_available():
    return _datasets.find("freMTPL2freq.parquet") is not None


FREQ_SKIP = pytest.mark.skipif(
    not _freq_available(),
    reason="data/freMTPL2freq.parquet not found (gitignored)",
)


def _fremtpl_features():
    # This mirrors the worked example in docs/guide/screening.md, and the
    # specification is deliberate.  BonusMalus is strongly curved on this book
    # (splined it reports edf 7.5 of rank 11), so specifying it as a Numeric
    # would both mis-fit the margin and demote every BonusMalus pair to the
    # deferred spline x numeric kind -- the sweep would then screen fewer
    # kinds than it can.  LogDensity is the honest linear margin: give it a
    # spline and the smooth collapses to edf 1.4 of rank 11.
    return {
        "DrivAge": Spline(kind="ps", n_knots=8),
        "VehAge": Spline(kind="ps", n_knots=12),
        "BonusMalus": Spline(
            kind="ps",
            n_knots=12,
            knot_strategy="quantile_tempered",
            knot_alpha=0.2,
        ),
        "LogDensity": Numeric(),
        "VehBrand": Categorical(),
        "Region": Categorical(),
    }


def _fremtpl_frame(n_rows=80_000):
    df = _datasets.load_freq().sample(n_rows, random_state=0).reset_index(drop=True)
    # The weight contract is Var(y) = phi * V(mu) / w, so an exposure-weighted
    # response is the claim RATE, not the count -- counts weighted by exposure
    # under-estimate phi and inflate every z in the sweep.  Clip the exposure
    # first, exactly as tests/test_realdata_parity.py::_prep_freq does, so a
    # near-zero denominator cannot manufacture a several-hundred-claim rate.
    df["Exposure"] = df["Exposure"].clip(lower=0.01)
    # log1p is the house transform for this column (see the credibility demo).
    df["LogDensity"] = np.log1p(df["Density"].to_numpy(dtype=np.float64))
    exposure = df["Exposure"].to_numpy(dtype=np.float64)
    y = df["ClaimNb"].to_numpy(dtype=np.float64) / exposure
    return df, y, exposure


@FREQ_SKIP
def test_fremtpl_mixed_sweep_end_to_end():
    df, y, exposure = _fremtpl_frame()
    model = SuperGLM(family="poisson", features=_fremtpl_features())
    model.fit_reml(df, y, sample_weight=exposure)
    table = model.screen_interactions(df, y, sample_weight=exposure)
    # every v1 kind this feature set can produce shows up and computes
    assert {"ti", "spline_cat", "numeric_cat", "cat_cat"} <= set(table["kind"])
    assert np.isfinite(table["z"]).any()
    # Six features make fifteen pairs; the three LogDensity x spline pairs are
    # deferred, so the sweep reports twelve.  This pins the deferral as a
    # silent drop from the default sweep rather than a NaN row.
    assert len(table) == 12
    # the queue is workable on a real book: the top pair refits and improves
    top = table.iloc[0]
    confirm = SuperGLM(
        family="poisson",
        features=_fremtpl_features(),
        interactions=[(top["feature_a"], top["feature_b"])],
    )
    confirm.fit_reml(df, y, sample_weight=exposure)
    assert confirm._result.deviance < model._result.deviance


@FREQ_SKIP
def test_fremtpl_example_specification_is_not_mis_specified():
    """The guide's worked example is an exemplar, so its spec must hold up.

    Two claims in docs/guide/screening.md are load-bearing and both are
    measurable: BonusMalus is curved (so specifying it as a Numeric would be
    wrong, and would demote its pairs to the deferred spline x numeric kind),
    and LogDensity is not (so it is an honest Numeric rather than one chosen
    to dodge a deferral).
    """
    df, y, exposure = _fremtpl_frame()
    base = _fremtpl_features()

    def deviance(overrides):
        model = SuperGLM(family="poisson", features={**base, **overrides})
        model.fit_reml(df, y, sample_weight=exposure)
        return model._result.deviance

    splined = deviance({})
    # Linearising BonusMalus costs real deviance -- it is genuinely curved.
    assert deviance({"BonusMalus": Numeric()}) - splined > 50.0
    # Splining LogDensity buys almost nothing -- it is genuinely linear.
    smoothed = deviance({"LogDensity": Spline(kind="ps", n_knots=8)})
    assert splined - smoothed < 10.0


def test_grouped_categorical_margins_are_excluded_until_refits_support_them():
    """A grouped factor is excluded: its confirmatory refit cannot be built.

    `Categorical(grouping=...)` fits fine as a main effect, and the screen's own
    `_categorical_codes` collapses it correctly — but no interaction builder maps
    the raw column through the grouping, so a confirmatory refit validates
    original labels against grouped levels and raises.  Screening such a pair
    would hand the caller a refit that cannot run, so the margin is excluded.
    The `pytest.raises` below pins the underlying defect: when it starts passing,
    the exclusion in `_margin_kind` can go.
    """
    from superglm.features.grouping import collapse_levels

    rng = np.random.default_rng(4321)
    n = 4000
    levels = ["R1", "R2", "R3", "R4", "R5"]
    df = pd.DataFrame({"age": rng.uniform(18.0, 80.0, n), "region": rng.choice(levels, n)})
    y = rng.poisson(0.3, n).astype(np.float64)
    grouping = collapse_levels(df["region"], groups={"R45": ["R4", "R5"]})

    def feats():
        return {
            "age": Spline(kind="ps", n_knots=5),
            "region": Categorical(grouping=grouping),
        }

    model = SuperGLM(family="poisson", features=feats())
    model.fit_reml(df, y)

    table = model.screen_interactions(df, y)
    assert "region" not in set(table["feature_a"]) | set(table["feature_b"])

    with pytest.raises(ValueError, match="deferred|screenable"):
        model.screen_interactions(df, y, candidates=[("age", "region")])

    # The defect the exclusion protects against: the refit itself cannot be built.
    with pytest.raises(ValueError, match="unseen categorical levels"):
        SuperGLM(family="poisson", features=feats(), interactions=[("age", "region")]).fit_reml(
            df, y
        )


def test_clamped_ladder_costs_no_decomposition_and_agrees_with_every_rung(monkeypatch):
    """For a spline_cat whose factor is wide, kron(S, I) has a null space above
    every budget, so all four rungs clamp to the same achieved edf.

    Two things must hold.  The whole ladder costs ZERO decompositions -- the
    ladder brackets first with two ordinary solves, and a rung that clamps is
    answered from that bracket, so the pencil is never built.  This is the
    case that matters most for wide factors, where decomposing would be
    strictly more expensive than the solves it would replace.  And running any
    single budget on its own must reproduce the ladder's row exactly, which is
    the property the old clamped-rung skip relied on for its correctness.
    """
    import superglm.screening._score_stat as score_stat

    L, reps = 40, 25
    rng = np.random.default_rng(77)
    n = L * reps
    df = pd.DataFrame(
        {
            "g": np.repeat([f"L{i}" for i in range(L)], reps),
            "x": rng.uniform(0.0, 1.0, n),
        }
    )
    y = rng.normal(size=n)
    model = SuperGLM(
        family="gaussian",
        features={"g": Categorical(), "x": Spline(kind="ps", n_knots=6)},
    )
    model.fit_reml(df, y)

    builds = []
    real = score_stat._build_pencil

    def counted(*args, **kwargs):
        builds.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(score_stat, "_build_pencil", counted)

    ladder = model.screen_interactions(df, y, candidates=[("x", "g")]).iloc[0]
    # four rungs, no decomposition at all -- every one of them clamps
    assert builds == []
    assert ladder["kind"] == "spline_cat"
    assert ladder["edf0"] > 16.0  # the achieved clamp sits above every budget

    # ... and each rung on its own returns exactly this row
    builds.clear()
    for budget in (4.0, 8.0, 16.0):
        rung = model.screen_interactions(df, y, candidates=[("x", "g")], edf0=budget).iloc[0]
        assert rung["statistic"] == ladder["statistic"]
        assert rung["edf0"] == ladder["edf0"]
        assert rung["lambda0"] == ladder["lambda0"]
        assert rung["z"] == ladder["z"]


def test_bisecting_ladder_resolves_every_rung_from_one_decomposition(monkeypatch):
    """A ti pair whose budgets land inside the bracket resolves a DIFFERENT
    lambda0 per rung, so every rung genuinely has to be evaluated -- and all
    four come out of a single decomposition, since the pencil is independent
    of both lambda and edf0."""
    import superglm.screening._score_stat as score_stat

    rng = np.random.default_rng(5)
    n = 20000
    x1 = rng.integers(0, 200, n) / 200.0
    x2 = rng.integers(0, 200, n) / 200.0
    df = pd.DataFrame({"x1": x1, "x2": x2})
    y = rng.normal(0.0, 1.0, n) + 1.5 * x1 * x2
    model = SuperGLM(
        family="gaussian",
        features={
            "x1": Spline(kind="ps", n_knots=8),
            "x2": Spline(kind="ps", n_knots=8),
        },
    )
    model.fit_reml(df, y)

    builds = []
    real_build = score_stat._build_pencil
    lambdas = []
    real_ladder = score_stat.penalized_score_statistic_ladder

    def counted_build(*args, **kwargs):
        builds.append(1)
        return real_build(*args, **kwargs)

    def recording_ladder(*args, **kwargs):
        out = real_ladder(*args, **kwargs)
        lambdas.append([r.lambda0 for r in out])
        return out

    monkeypatch.setattr(score_stat, "_build_pencil", counted_build)
    monkeypatch.setattr(
        "superglm.model.screening_ops.penalized_score_statistic_ladder", recording_ladder
    )
    row = model.screen_interactions(df, y, candidates=[("x1", "x2")]).iloc[0]

    assert builds == [1], "one decomposition serves the whole ladder"
    assert len(lambdas) == 1 and len(lambdas[0]) == 4, "all four rungs reported"
    # The rungs are genuinely distinct -- this is the case the old
    # clamped-rung skip had to be fenced away from.
    assert len(set(lambdas[0])) == 4, f"expected four distinct lambda0, got {lambdas[0]}"
    assert row["edf0"] == pytest.approx(2.0, abs=1e-3)


def _balanced_factorial(la, lb, reps=2, seed=19):
    """A fully populated La x Lb design: every cell occupied, no singletons.

    Keeps the pair's curvature full rank so the lifted screen takes the
    Cholesky branch -- the point under test is the block DIMENSION gate, and a
    rank-deficient block would spend the test's time in the pseudo-inverse.
    """
    a = np.repeat(np.arange(la), lb * reps)
    b = np.tile(np.repeat(np.arange(lb), reps), la)
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "a": np.array([f"A{i}" for i in range(la)])[a],
            "b": np.array([f"B{i}" for i in range(lb)])[b],
            "z": rng.uniform(0.5, 2.0, la * lb * reps),
        }
    )
    return df, rng.normal(size=len(df))


def test_cat_cat_refuses_a_block_too_wide_to_solve():
    """Review finding: the cell and intermediate budgets bound ALLOCATION,
    which grows as k^2, while the per-rung factorization grows as k^3 -- so
    they admitted cat_cat blocks up to k = 4472, measured at 24s and 1.3GB per
    pair.  The cubic gate refuses on the block dimension, with the same NaN-row
    semantics as every other budget, and `max_cells` lifts it."""
    la, lb = 44, 41
    k = (la - 1) * (lb - 1)  # 1720, one rung above the default ceiling of 1709
    df, y = _balanced_factorial(la, lb)
    model = SuperGLM(
        family="gaussian",
        features={"a": Categorical(), "b": Categorical(), "z": Numeric()},
    )
    model.fit_reml(df, y)

    table = model.screen_interactions(df, y)
    refused = table[table["kind"] == "cat_cat"].iloc[0]
    assert np.isnan(refused["statistic"]) and np.isnan(refused["z"])
    assert refused["n_cells"] == la * lb  # the grid it was refused for
    assert not refused["approx"]  # refusal is not approximation
    assert table.index[table["kind"] == "cat_cat"][0] == len(table) - 1  # sorts last
    # the other pairs in the same sweep are unaffected
    assert np.isfinite(table[table["kind"] != "cat_cat"]["z"]).all()

    # One unit short of the cubic budget is still a refusal ...
    budget = -(-(k**3) // 1000)
    short = model.screen_interactions(df, y, candidates=[("a", "b")], max_cells=budget - 1).iloc[0]
    assert np.isnan(short["z"])
    assert short["n_cells"] == la * lb
    # ... and at the budget the same pair computes, exactly and unpenalized.
    lifted = model.screen_interactions(df, y, candidates=[("a", "b")], max_cells=budget).iloc[0]
    assert np.isfinite(lifted["z"])
    assert lifted["edf0"] == pytest.approx(k, abs=0.5)  # achieved rank
    assert lifted["lambda0"] == 0.0
    assert lifted["n_cells"] == la * lb
    assert not lifted["approx"]


def test_cubic_gate_leaves_ordinary_pairs_alone():
    """The gate is a tail guard: a pair whose block is small enough to solve
    quickly must screen identically to before, at the default budget and at
    any budget above its own cubic threshold."""
    la, lb = 20, 20
    k = (la - 1) * (lb - 1)  # 361 -- the cost here is 0.04s, not seconds
    df, y = _balanced_factorial(la, lb, reps=3)
    model = SuperGLM(family="gaussian", features={"a": Categorical(), "b": Categorical()})
    model.fit_reml(df, y)

    default = model.screen_interactions(df, y).iloc[0]
    assert np.isfinite(default["z"])
    assert default["n_cells"] == la * lb

    budget = -(-(k**3) // 1000)
    at_threshold = model.screen_interactions(df, y, max_cells=budget).iloc[0]
    assert at_threshold["z"] == default["z"]
    assert at_threshold["statistic"] == default["statistic"]
    # below it, the same pair is refused -- the gate scales with max_cells the
    # way every other budget here does
    below = model.screen_interactions(df, y, max_cells=budget - 1).iloc[0]
    assert np.isnan(below["z"])


def test_penalized_blocks_are_charged_the_ladder_they_run():
    """A penalized block runs a whole ladder where an unpenalized one returns a
    single rung, so the same dimension buys less time, and the gate charges it
    _PENALIZED_LADDER_COST times the work.

    All three pairs below sit at exactly k = 144, which is deliberately
    between the two ceilings the charge creates.  What separates them is what
    each one has to solve:

      * ``cat_cat`` is unpenalized, so k = 144 is inside its budget;
      * ``ti`` is penalized and has no structure to exploit -- refused;
      * ``spline_cat`` is penalized too, but its bordered system is an arrow
        matrix, so being over the DENSE budget routes it to the structured
        kernel rather than refusing it.

    That last one is the whole point of the arrow path: the charge still
    applies, it just no longer terminates the pair.
    """
    # k = 12 * 12 = 144 for every pair.  max_cells is chosen so 144 lands
    # between the unpenalized ceiling (5.5e6^(1/3) = 176) and the penalized
    # one (2.75e6^(1/3) = 140), and still clears the (k^2 <= 4*max_cells)
    # intermediate gate at 20736 <= 22000.  The spline margins are carried on
    # a 40-point grid so that every pair also clears the SUPPORT-scaled
    # intermediate budgets (dense and structured alike); the subject here is
    # the cubic charge, and an allocation refusal would mask it.
    max_cells = 5_500
    n_levels, reps = 13, 14
    rng = np.random.default_rng(23)
    n = n_levels * n_levels * reps
    a = np.repeat(np.arange(n_levels), n_levels * reps)
    b = np.tile(np.repeat(np.arange(n_levels), reps), n_levels)
    grid = np.linspace(0.0, 1.0, 40)
    df = pd.DataFrame(
        {
            "g": np.array([f"G{i}" for i in range(n_levels)])[a],
            "h": np.array([f"H{i}" for i in range(n_levels)])[b],
            "x1": grid[rng.integers(0, grid.size, n)],
            "x2": grid[rng.integers(0, grid.size, n)],
        }
    )
    y = rng.normal(size=n)
    model = SuperGLM(
        family="gaussian",
        features={
            "g": Categorical(),
            "h": Categorical(),
            "x1": Spline(kind="ps", n_knots=9),  # centered width 12
            "x2": Spline(kind="ps", n_knots=9),
        },
    )
    model.fit_reml(df, y)
    # edf at maximum penalty is about L - 1 = 12 for the spline_cat pair, so
    # the default ladder's top rung would bisect and be charged for it.  The
    # subject here is the CUBIC charge; capping the ladder at 8 keeps every
    # rung clamped so the structured evaluation budget stays out of the way.
    table = model.screen_interactions(df, y, max_cells=max_cells, edf0=(2.0, 4.0, 8.0)).set_index(
        ["feature_a", "feature_b"]
    )
    assert table.loc[("g", "h"), "kind"] == "cat_cat"
    assert table.loc[("x1", "x2"), "kind"] == "ti"
    assert table.loc[("g", "x1"), "kind"] == "spline_cat"

    assert np.isfinite(table.loc[("g", "h"), "z"])  # unpenalized: inside budget
    assert np.isnan(table.loc[("x1", "x2"), "z"])  # penalized, dense only
    assert np.isfinite(table.loc[("g", "x1"), "z"])  # penalized, but structured

    # raising max_cells lifts the ti refusal, as it lifts every other budget
    lifted = model.screen_interactions(df, y, candidates=[("x1", "x2")], max_cells=20_000_000)
    assert np.isfinite(lifted.iloc[0]["z"])


def _span_cosine(probe, refit):
    """Minimum principal cosine between two column spans of equal rank.

    Both sides are identifiable deviation spaces -- the probe centered by
    ``tensor_marginal_ingredients()``, the refit by the build of the spec it
    resolved -- so they must have the same rank AND the same span.  Equal rank
    is asserted here rather than left to the cosine because containment alone
    is too weak: a refit that skipped centering carries one extra direction and
    still reproduces every probe direction.
    """

    def orth(m):
        m = np.asarray(m, dtype=np.float64)
        q, r = np.linalg.qr(m)
        diag = np.abs(np.diag(r))
        return q[:, diag > 1e-10 * max(1.0, diag.max())]

    qi, qo = orth(probe), orth(refit)
    assert qi.shape[1] == qo.shape[1], f"probe rank {qi.shape[1]} != refit rank {qo.shape[1]}"
    return float(np.linalg.svd(qo.T @ qi, compute_uv=False).min())


@pytest.mark.parametrize("spline_kind", ["ps", "cr"])
def test_spline_cat_probe_spans_the_basis_its_refit_builds(spline_kind):
    """The screen's probe and the term it names as its refit must agree.

    This is the invariant the whole confirm-by-refit contract rests on, so it
    is pinned as an identity rather than as a value.  It held for ``ps`` and
    failed for ``cr``, because cubic regression splines carry two bases: a
    main effect uses the projected-B-spline ``CubicRegressionSpline`` while an
    interaction marginal uses ``CardinalCRSpline``.  ``TensorInteraction`` had
    that routing and ``SplineCategorical`` did not, so on the margin below the
    ``cr`` probe and its own refit spanned different spaces -- minimum
    principal cosine 0.9028, a 25.5 degree rotation.
    """
    import scipy.sparse as sp

    from superglm.features.interaction import TensorInteraction

    rng = np.random.default_rng(0)
    n = 8000
    # Skewed, so uniform and quantile knot placement genuinely disagree.
    x = np.round(np.concatenate([rng.uniform(0, 1, 6400), rng.uniform(0, 10, 1600)]), 2)
    df = pd.DataFrame({"x": x, "f": pd.Categorical(rng.integers(0, 4, n).astype(str))})
    y = rng.poisson(np.exp(0.2 + 0.15 * np.log1p(x))).astype(np.float64)
    features = {"x": Spline(kind=spline_kind, n_knots=10), "f": Categorical()}
    support = np.unique(x)
    counts = np.unique(x, return_counts=True)[1]

    # The probe, exactly as screen_interactions assembles it from the mains fit.
    mains = SuperGLM(family="poisson", features=features)
    mains.fit_reml(df, y)
    info = TensorInteraction._marginal_from_spec(
        mains._specs["x"], x, None, support=support, counts=counts
    )
    basis = info.basis
    probe = np.asarray(basis.todense() if sp.issparse(basis) else basis, dtype=np.float64)

    # The refit, read back from the spec the fitted term actually stored --
    # not from the parent spec, which is the thing that used to differ.
    confirmed = SuperGLM(family="poisson", features=features, interactions=[("x", "f")])
    confirmed.fit_reml(df, y)
    ispec = confirmed._interaction_specs[next(iter(confirmed._interaction_specs))]
    stored = ispec._spline_spec
    raw = np.asarray(sp.csr_matrix(stored._raw_basis_matrix(support)).todense(), dtype=np.float64)
    projection = getattr(stored, "_interaction_projection", None)
    refit = raw if projection is None else raw @ np.asarray(projection, dtype=np.float64)

    assert _span_cosine(probe, refit) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("discrete", [False, True])
@pytest.mark.parametrize("spline_kind", ["ps", "cr"])
def test_spline_cat_design_keeps_full_rank_beside_its_mains(spline_kind, discrete):
    """A masked interaction block must not rebuild its own level indicator.

    An interaction marginal resolved but never built carries no identifiability
    projection, and the cardinal cr basis reproduces the constant function
    exactly -- its columns sum to 1 at every x.  Masking that uncentered block
    to a level then duplicates the categorical main effect's indicator column
    for that level: one lost rank per non-base level, in a design the solver
    still has to invert.
    """
    rng = np.random.default_rng(3)
    n = 4000
    x = rng.uniform(0.0, 10.0, n)
    df = pd.DataFrame({"x": x, "f": pd.Categorical(rng.integers(0, 4, n).astype(str))})
    y = rng.poisson(2.0, n).astype(np.float64)

    model = SuperGLM(
        family="poisson",
        features={"x": Spline(kind=spline_kind, n_knots=8), "f": Categorical()},
        interactions=[("x", "f")],
        discrete=discrete,
        n_bins=64 if discrete else None,
    )
    model._build_design_matrix(df, y, np.ones(n), None)
    design = model._dm.toarray()
    singular = np.linalg.svd(design, compute_uv=False)
    rank = int(np.sum(singular > singular.max() * 1e-10))
    assert rank == design.shape[1], f"{design.shape[1] - rank} deficient column(s)"


@pytest.mark.parametrize("spline_kind", ["ps", "cr"])
def test_spline_cat_predicts_the_design_it_fitted(spline_kind):
    """Storing a resolved parent spec is only safe if predict reads it back.

    ``transform``/``score`` rebuild the basis from ``self._spline_spec``, so
    substituting the interaction basis at build time without storing it would
    silently score a different linear predictor than the fitted design.
    """
    rng = np.random.default_rng(1)
    n = 8000
    x = np.round(np.concatenate([rng.uniform(0, 1, 6400), rng.uniform(0, 10, 1600)]), 2)
    df = pd.DataFrame({"x": x, "f": pd.Categorical(rng.integers(0, 4, n).astype(str))})
    y = rng.poisson(np.exp(0.2 + 0.15 * np.log1p(x))).astype(np.float64)

    model = SuperGLM(
        family="poisson",
        features={"x": Spline(kind=spline_kind, n_knots=10), "f": Categorical()},
        interactions=[("x", "f")],
    )
    model.fit_reml(df, y)
    mu = model.predict(df)
    term = np.where(y > 0, y * np.log(np.maximum(y, 1e-300) / mu), 0.0)
    assert 2.0 * float(np.sum(term - (y - mu))) == pytest.approx(
        model._result.deviance, rel=1e-9, abs=1e-9
    )


def test_select_cr_parent_keeps_its_own_basis_in_an_interaction():
    """The interaction cr routing must not touch a select=True parent.

    The cardinal spec an interaction marginal resolves to is built with
    ``select=False`` and one penalty order, so it cannot carry a select
    parent's double-penalty geometry -- substituting drops the identifiability
    projection that shrinks the block and leaves the design rank-deficient.
    ``screen_interactions`` refuses select parents outright, so no probe goes
    unmatched by leaving them alone.
    """
    from superglm.features.interaction import interaction_spline_spec

    rng = np.random.default_rng(0)
    n = 4000
    x = rng.uniform(0.0, 10.0, n)
    spec = Spline(kind="cr", n_knots=8, select=True)
    plain = Spline(kind="cr", n_knots=8)

    df = pd.DataFrame({"x": x, "f": pd.Categorical(rng.integers(0, 4, n).astype(str))})
    y = rng.poisson(1.5, n).astype(np.float64)
    fitted_select = SuperGLM(family="poisson", features={"x": spec, "f": Categorical()})
    fitted_select.fit_reml(df, y)
    fitted_plain = SuperGLM(family="poisson", features={"x": plain, "f": Categorical()})
    fitted_plain.fit_reml(df, y)

    sel = fitted_select._specs["x"]
    assert interaction_spline_spec(sel, x) is sel, "select parent must pass through"
    # The control: an ordinary cr parent IS resolved, so the guard is specific
    # rather than the routing being dead.
    ordinary = fitted_plain._specs["x"]
    assert interaction_spline_spec(ordinary, x) is not ordinary

    # And screening refuses the select parent, so nothing is left unscreened
    # that a probe would have needed to match.
    with pytest.raises(ValueError, match="select=True"):
        fitted_select.screen_interactions(df, y)


@pytest.mark.parametrize("m_order", [1, 2, 3])
def test_cr_interaction_keeps_the_penalty_order_it_was_asked_for(m_order):
    """A cr parent's penalty order must survive the interaction routing.

    ``CardinalCRSpline`` implements one penalty -- the integrated squared
    second derivative -- and rejects ``m=3`` outright.  Routing every cr parent
    through it would therefore raise for ``m=3`` and silently answer ``m=1``
    with a second-derivative penalty.  The parent's own quadrature penalty
    supports all three, so anything but ``m=2`` keeps it, in the varying
    coefficient block and the tensor marginal alike.
    """
    from superglm.features.spline import CardinalCRSpline, CubicRegressionSpline

    rng = np.random.default_rng(4)
    n = 4000
    x = rng.uniform(0.0, 10.0, n)
    df = pd.DataFrame(
        {
            "x": x,
            "z": rng.uniform(0.0, 5.0, n),
            "f": pd.Categorical(rng.integers(0, 4, n).astype(str)),
        }
    )
    y = rng.poisson(2.0, n).astype(np.float64)
    parent = Spline(kind="cr", n_knots=8, m=m_order)

    varying = SuperGLM(
        family="poisson",
        features={"x": parent, "f": Categorical()},
        interactions=[("x", "f")],
    )
    varying.fit_reml(df, y)
    stored = varying._interaction_specs[next(iter(varying._interaction_specs))]._spline_spec
    assert stored._m_orders == (m_order,)

    if m_order == 2:
        # The one order the cardinal basis implements, so the routing is live.
        assert isinstance(stored, CardinalCRSpline)
    else:
        # Everything else keeps the parent, whose quadrature penalty is the
        # only implementation that has the requested order at all.  The second
        # assertion is the control: these orders build genuinely different
        # penalties, so "kept the parent" is not vacuously true.
        assert isinstance(stored, CubicRegressionSpline)
        charged = stored._build_penalty()
        assert np.allclose(charged, stored._build_penalty_for_order(m_order))
        assert not np.allclose(charged, stored._build_penalty_for_order(2))

    # The tensor path takes the same routing, so it must survive m=3 too.
    tensor = SuperGLM(
        family="poisson",
        features={"x": parent, "z": Spline(kind="cr", n_knots=8)},
        interactions=[("x", "z")],
    )
    tensor.fit_reml(df, y)
    assert np.isfinite(tensor._result.deviance)
