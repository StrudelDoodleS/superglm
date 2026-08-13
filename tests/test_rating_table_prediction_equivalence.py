"""The exported rating table has to reproduce the model it was exported from.

The workbook's contract is multiplicative: ``base_relativity`` times one
relativity per main-effect block, times one per interaction block, reproduces
``model.predict`` row by row.  That is what a filed tariff *is*, so it is pinned
here directly -- the consumer below reads only what a workbook carries (level
labels, knots, log relativities, bin interval strings, a per-unit relativity,
interaction cells) and never consults the fitted spec.

Three claims are separated on purpose, because only the first can be exact:

* **Exactly tabulable blocks** -- ``Categorical``, ``OrderedCategorical``,
  ``Numeric``, ``Piecewise``, and the categorical-by-categorical interaction --
  reproduce ``model.predict`` to round-off.
* **Binned blocks** -- ``Spline`` and ``Polynomial`` -- are exported through the
  discretisation path, so they carry the binning error the impact sheet exists
  to report.  Two things are asserted of them.  Centering INVARIANCE: the
  reconstructed prediction may not equal ``model.predict``, but it must not
  depend on which reporting centering was asked for.  And ABSENCE OF BIAS: the
  binning error is spread, not scale, so the exposure-weighted mean log ratio
  must stay inside a bound derived from the bin geometry.  The second exists
  because the first cannot see a constant dropped from the binned path -- both
  centerings drop it, so the comparison cancels.
* **The base relativity** is neither: it is the one number with no tolerable
  approximation, so an unrepresentable one has to stop the export rather than
  ship as ``inf`` or ``0.0``.

``centering="mean"`` is swept beside the default everywhere, because that is
the mode that was wrong: ``_recenter_term`` subtracted a per-term constant that
the exported base relativity did not absorb, scaling every reconstructed
prediction by a uniform factor that no ratio-based spot check can see.
"""

from __future__ import annotations

import re
from functools import cache

import numpy as np
import pandas as pd
import pytest

from superglm import (
    Categorical,
    Numeric,
    OrderedCategorical,
    Polynomial,
    Spline,
    SuperGLM,
)
from superglm.export import rating_tables
from superglm.export.rating_tables import (
    RatingTableBaseNotRepresentableError,
    build_rating_table_payload,
)
from superglm.features.grouping import collapse_levels
from superglm.features.piecewise import Piecewise

_EPS = float(np.finfo(np.float64).eps)

# Roundings along the two paths this module compares, on the exactly tabulable
# model.  Reconstruction: the base exp (1), a two-point interpolation and its
# exp for the piecewise block (6), a power for the numeric block (3), and four
# products (4).  Model: a hat dot product (2), five additive term contributions
# (5) and an exp (1).  Twenty-two, plus the comparison's own two, rounded up to
# 32 for headroom on a differently ordered BLAS -- the same count and the same
# rounding the piecewise workbook test derives, because it is the same two
# paths with two more exactly tabulable terms in the product.
#
# The interaction fixture adds one factor to each side -- a cell lookup in the
# reconstruction, an additive contribution in the model -- so its count is 24
# rather than 22 and the same 32 still covers it.  Measured there: 4.36e-16
# (native) and 6.17e-16 (mean), against this bound of 7.11e-15.
_RECONSTRUCTION_RTOL = 32 * _EPS

_CENTERINGS = ("native", "mean")
_INTERVAL = re.compile(r"^\[([^,]+), ([^)]+)\)$")

_BAND_LEVELS = ["b1", "b2", "b3", "b4", "b5"]
_TERRITORY_LEVELS = ["t1", "t2", "t3", "t4", "t5", "t6"]


def _frame(n: int = 900, seed: int = 20260811) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """One frame that carries a column for every shipped main-effect term type."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "age": rng.uniform(18.0, 80.0, n),
            "score": rng.normal(0.0, 1.0, n),
            "region": rng.choice(["A", "B", "C"], n),
            "territory": rng.choice(_TERRITORY_LEVELS, n),
            "band": rng.choice(_BAND_LEVELS, n),
            "density": rng.uniform(0.0, 1.0, n),
            "x": rng.uniform(0.0, 10.0, n),
        }
    )
    eta = (
        -1.2
        + 0.15 * np.sin(X["age"].to_numpy() / 8.0)
        + 0.10 * X["score"].to_numpy()
        + 0.25 * (X["region"].to_numpy() == "B")
        + 0.12 * np.array([_TERRITORY_LEVELS.index(v) for v in X["territory"]])
        + 0.08 * np.array([_BAND_LEVELS.index(v) for v in X["band"]])
        + 0.20 * X["density"].to_numpy()
        + 0.03 * X["x"].to_numpy()
    )
    sample_weight = rng.uniform(0.5, 2.0, n)
    y = rng.poisson(np.exp(eta) * sample_weight).astype(np.float64)
    return X, y, sample_weight


def _territory_grouping():
    """t2+t3 and t4+t5 collapse; the grouped mean differs from the level mean."""
    return collapse_levels(
        _TERRITORY_LEVELS,
        groups={"t2+t3": ["t2", "t3"], "t4+t5": ["t4", "t5"]},
        order=_TERRITORY_LEVELS,
    )


def _exactly_tabulable_features() -> dict:
    return {
        "region": Categorical(base="first"),
        "territory": Categorical(base="first", grouping=_territory_grouping()),
        "band": OrderedCategorical(order=_BAND_LEVELS, basis=Spline(kind="ps", n_knots=4)),
        "density": Numeric(),
        "x": Piecewise(breaks=[2.5, 5.0, 7.5], base=5.0),
    }


def _every_term_type_features() -> dict:
    features = {"age": Spline(n_knots=8), "score": Polynomial(degree=3)}
    features.update(_exactly_tabulable_features())
    return features


_SEGMENT_LEVELS = ["s1", "s2", "s3"]


def _interaction_frame(
    n: int = 1200, seed: int = 20260813
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """A frame for the interaction fixture, kept separate from ``_frame``.

    Its own columns and its own response, so that adding an interaction cannot
    move the fitted values the exactness and shift-accounting tests above pin.
    Both interacting columns are plain ``Categorical``: that is the pair the
    export tabulates as a full cell table, and the grouped ``territory`` of the
    main fixture is deliberately not used, because the exported interaction
    table is keyed on GROUPED levels while its main-effect block is keyed on the
    original ones, and the grouping map is not in the workbook.
    """
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "region": rng.choice(["A", "B", "C"], n),
            "segment": rng.choice(_SEGMENT_LEVELS, n),
            "density": rng.uniform(0.0, 1.0, n),
            "x": rng.uniform(0.0, 10.0, n),
        }
    )
    region = X["region"].to_numpy()
    segment = X["segment"].to_numpy()
    eta = (
        -1.2
        + 0.25 * (region == "B")
        + 0.45 * (region == "C")
        + 0.30 * (segment == "s2")
        + 0.55 * (segment == "s3")
        # The interaction proper: a cell that is not the sum of its margins.
        + 0.40 * ((region == "C") & (segment == "s3"))
        + 0.20 * X["density"].to_numpy()
        + 0.03 * X["x"].to_numpy()
    )
    sample_weight = rng.uniform(0.5, 2.0, n)
    y = rng.poisson(np.exp(eta) * sample_weight).astype(np.float64)
    return X, y, sample_weight


def _interaction_features() -> dict:
    """Centered main effects beside the interaction, so both paths are live."""
    return {
        "region": Categorical(base="first"),
        "segment": Categorical(base="first"),
        "density": Numeric(),
        "x": Piecewise(breaks=[2.5, 5.0, 7.5], base=5.0),
    }


@cache
def _fit(kind: str) -> tuple[SuperGLM, pd.DataFrame, np.ndarray, np.ndarray]:
    """Fit once per model shape; every test here reads the fit, none mutates it.

    ``"exact"`` carries only exactly tabulable terms, ``"all"`` adds the binned
    ``Spline`` and ``Polynomial``, and ``"interaction"`` puts a
    categorical-by-categorical interaction beside centered main effects -- the
    one interaction kind the export tabulates as a full cell table, and so the
    only one for which the payload's product claim can be exact.
    """
    if kind == "interaction":
        X, y, sample_weight = _interaction_frame()
        features: dict = _interaction_features()
        interactions: list[tuple[str, str]] | None = [("region", "segment")]
    else:
        X, y, sample_weight = _frame()
        features = _exactly_tabulable_features() if kind == "exact" else _every_term_type_features()
        interactions = None
    model = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        features=features,
        interactions=interactions,
    )
    model.fit(X, y, sample_weight=sample_weight)
    return model, X, y, sample_weight


def _payload(model, X, y, sample_weight, centering: str):
    return build_rating_table_payload(
        model,
        X,
        y,
        sample_weight=sample_weight,
        n_bins=150,
        impact_bins=(20,),
        centering=centering,
    )


def _block_multiplier(block, X: pd.DataFrame) -> np.ndarray:
    """The per-row factor one exported block implies, read out of the block alone."""
    name = block.name
    table = block.table
    if block.kind == "categorical":
        lookup = {str(k): float(v) for k, v in zip(table[name], table["Relativity"])}
        return np.array([lookup[str(value)] for value in X[name]], dtype=np.float64)
    if block.kind == "numeric":
        assert list(table[name]) == ["per_unit"]
        return float(table["Relativity"].iloc[0]) ** X[name].to_numpy(dtype=np.float64)
    if block.kind == "piecewise":
        knots = table[name].to_numpy(dtype=np.float64)
        log_relativity = table["Log relativity"].to_numpy(dtype=np.float64)
        return np.exp(np.interp(X[name].to_numpy(dtype=np.float64), knots, log_relativity))
    if block.kind == "continuous":
        bounds = [_INTERVAL.match(str(label)) for label in table[name]]
        assert all(bound is not None for bound in bounds)
        edges = np.array(
            [float(bound.group(1)) for bound in bounds] + [float(bounds[-1].group(2))],
            dtype=np.float64,
        )
        relativity = table["Relativity"].to_numpy(dtype=np.float64)
        index = np.digitize(X[name].to_numpy(dtype=np.float64), edges, right=False)
        return relativity[np.clip(index, 1, len(relativity)) - 1]
    if block.kind == "offset":
        # The ``offset_source=`` form, which is the one a consumer can actually
        # apply: the block is keyed on a raw column of the frame, exactly like
        # a categorical lookup.  The nameless "Offset Multiplier" block is keyed
        # on the multiplier itself and is an exposure summary of an input the
        # consumer already holds, so it is not part of this reconstruction.
        lookup = {str(k): float(v) for k, v in zip(table[name], table["Relativity"])}
        return np.array([lookup[str(value)] for value in X[name]], dtype=np.float64)
    raise AssertionError(f"unhandled exported block kind {block.kind!r}")


def _interaction_multiplier(block, X: pd.DataFrame) -> np.ndarray:
    """The per-row factor one exported interaction cell table implies.

    Read the same way a consumer reads it off the sheet: the first column
    carries the row key and is named for the first parent, every remaining
    column header is a level of the second parent, and the cell is the factor.
    The second parent's name comes from the block name, which is the only place
    the sheet records it.
    """
    table = block.table
    parent1 = str(table.columns[0])
    parent2 = block.name.split(":")[1]
    assert parent2 in X.columns, f"{block.name!r} names a column the frame does not have"
    levels2 = [str(column) for column in table.columns[1:]]
    cells = {
        (str(row[parent1]), level2): float(row[level2])
        for _, row in table.iterrows()
        for level2 in levels2
    }
    return np.array(
        [cells[(str(a), str(b))] for a, b in zip(X[parent1], X[parent2])],
        dtype=np.float64,
    )


def _predict_from_payload(
    payload, X: pd.DataFrame, *, with_interactions: bool = True
) -> np.ndarray:
    """The premium the exported workbook implies, read out of the workbook alone.

    ``with_interactions=False`` exists only to measure how far the
    reconstruction moves without the interaction factor, so that the exactness
    claim beside it is a discriminating measurement.  No consumer would drop it.
    """
    reconstructed = np.full(len(X), float(payload.base_relativity), dtype=np.float64)
    for block in payload.main_effects:
        reconstructed = reconstructed * _block_multiplier(block, X)
    if with_interactions:
        for block in payload.interactions:
            reconstructed = reconstructed * _interaction_multiplier(block, X)
    return reconstructed


@pytest.mark.parametrize("centering", _CENTERINGS)
def test_the_exactly_tabulable_workbook_reproduces_the_predictions(centering):
    """Every term here is a lookup or an interpolation, so the product is exact."""
    assert _RECONSTRUCTION_RTOL <= 1e-12

    model, X, y, sample_weight = _fit("exact")
    payload = _payload(model, X, y, sample_weight, centering)

    np.testing.assert_allclose(
        _predict_from_payload(payload, X),
        model.predict(X),
        rtol=_RECONSTRUCTION_RTOL,
        atol=0.0,
    )


def test_the_reconstruction_does_not_depend_on_the_reporting_centering():
    """Binned terms cannot be exact, but they must not move with the centering.

    ``centering=`` is a presentation choice.  A workbook that rates a different
    premium depending on it is wrong whatever the binning error is, and this is
    the only form of the claim that a spline or polynomial block can carry.
    """
    model, X, y, sample_weight = _fit("all")
    assert {
        block.kind for block in _payload(model, X, y, sample_weight, "native").main_effects
    } == {
        "categorical",
        "continuous",
        "numeric",
        "piecewise",
    }

    reconstructions = [
        _predict_from_payload(_payload(model, X, y, sample_weight, centering), X)
        for centering in _CENTERINGS
    ]
    np.testing.assert_allclose(
        reconstructions[1], reconstructions[0], rtol=_RECONSTRUCTION_RTOL, atol=0.0
    )


def test_the_base_relativity_moves_by_exactly_the_shift_the_blocks_applied():
    """The mechanism, stated in the units the sheet is written in.

    The exported base is the only place the removed constants can go, so the
    log of the ratio between the two exports' base relativities has to be the
    total the blocks subtracted -- no more, and in particular not less.  The
    second assertion is the reason this quantity has to be READ BACK: the
    obvious re-derivation, ``mean(log_relativity)`` over the reported values,
    is a measurably different number on this fixture, because an
    ``OrderedCategorical`` is never recentered (its level mean was never
    removed) and a grouped ``Categorical`` is recentered on its GROUPED levels
    before expansion (its expanded level mean is not what was removed).  What
    rejects that re-derivation is the exactness sweep above; what this test
    fixes is that the two candidate constants really do differ here, so that
    sweep is a discriminating measurement rather than a coincidence.
    """
    model, X, y, sample_weight = _fit("all")
    native = _payload(model, X, y, sample_weight, "native")
    mean = _payload(model, X, y, sample_weight, "mean")

    applied = 0.0
    rederived = 0.0
    for block_native, block_mean in zip(native.main_effects, mean.main_effects):
        log_native = np.log(block_native.table["Relativity"].to_numpy(dtype=np.float64))
        log_mean = np.log(block_mean.table["Relativity"].to_numpy(dtype=np.float64))
        applied += float(np.mean(log_native - log_mean))
        if log_mean.size > 1:
            rederived += float(np.mean(log_native))

    transferred = np.log(mean.base_relativity) - np.log(native.base_relativity)
    assert transferred == pytest.approx(applied, abs=1e-12)
    assert applied > 0.5
    assert abs(rederived - applied) > 0.05


def test_an_ordered_categorical_block_carries_no_centering_shift():
    """``_recenter_term`` never touches an OC, so it must transfer nothing."""
    model, X, y, sample_weight = _fit("exact")
    native = _payload(model, X, y, sample_weight, "native")
    mean = _payload(model, X, y, sample_weight, "mean")

    band_native = next(b for b in native.main_effects if b.name == "band")
    band_mean = next(b for b in mean.main_effects if b.name == "band")
    np.testing.assert_array_equal(
        band_mean.table["Relativity"].to_numpy(), band_native.table["Relativity"].to_numpy()
    )
    # ...and the level mean it would have contributed is not zero, so the
    # assertion above is a real constraint rather than a vacuous one.
    assert abs(float(np.mean(np.log(band_native.table["Relativity"].to_numpy())))) > 0.01


@pytest.mark.parametrize("centering", _CENTERINGS)
def test_an_offset_model_reconstructs_from_the_offset_lookup_and_the_blocks(centering):
    """The offset block is a rating factor like any other and must multiply in.

    It carries no centering shift of its own -- it is not a fitted term's
    relativities -- so it is the block most likely to be swept into a total
    that assumes every exported block was centered.  Exported through
    ``offset_source=``, which is the form keyed on a raw column, so the
    consumer applies it by lookup rather than by holding the link-scale offset.
    """
    rng = np.random.default_rng(4242)
    n = 400
    term = np.resize(np.array([12.0, 24.0, 36.0]), n)
    X = pd.DataFrame({"region": rng.choice(["A", "B", "C"], n), "term": term})
    offset = np.log(term / 12.0)
    sample_weight = rng.uniform(0.5, 2.0, n)
    eta = -1.4 + 0.3 * (X["region"].to_numpy() == "B") + offset
    y = rng.poisson(np.exp(eta) * sample_weight).astype(np.float64)

    model = SuperGLM(
        family="poisson", selection_penalty=0.0, features={"region": Categorical(base="first")}
    )
    model.fit(X, y, sample_weight=sample_weight, offset=offset)

    payload = build_rating_table_payload(
        model,
        X,
        y,
        sample_weight=sample_weight,
        offset_source="term",
        centering=centering,
    )
    offset_block = next(block for block in payload.main_effects if block.kind == "offset")
    assert offset_block.name == "term"
    assert offset_block.centering_shift == 0.0

    np.testing.assert_allclose(
        _predict_from_payload(payload, X),
        model.predict(X, offset=offset),
        rtol=_RECONSTRUCTION_RTOL,
        atol=0.0,
    )


# ── The interaction half of the payload's product claim ────────────────────
#
# ``build_rating_table_payload``'s docstring states a contract over the
# interaction blocks as well as the main-effect ones.  Nothing pinned it: the
# reconstruction above multiplied main effects only, so an interaction block
# could have been empty, transposed, or keyed on the wrong parent and every
# equivalence test would still have passed.


@pytest.mark.parametrize("centering", _CENTERINGS)
def test_the_workbook_with_an_interaction_reproduces_the_predictions(centering):
    """A categorical-by-categorical interaction is a full cell table, so exact."""
    assert _RECONSTRUCTION_RTOL <= 1e-12

    model, X, y, sample_weight = _fit("interaction")
    payload = _payload(model, X, y, sample_weight, centering)
    assert [block.name for block in payload.interactions] == ["region:segment"]

    np.testing.assert_allclose(
        _predict_from_payload(payload, X),
        model.predict(X),
        rtol=_RECONSTRUCTION_RTOL,
        atol=0.0,
    )


def test_the_interaction_factor_is_load_bearing_not_decorative():
    """The exactness above has to be earned by the interaction block.

    An interaction whose cells were all 1.0 -- or one dropped from the product
    -- would leave the test above asserting only that the main effects are
    right.  This fixes how far the reconstruction moves without it, so the
    exactness claim is a discriminating measurement.

    Measured on this fixture: 5.59e-01 maximum relative error without the
    block, against 6.2e-16 with it.  The thresholds are floors on "the
    interaction does something", set an order of magnitude below that so a
    refit which happens to shrink the effect does not turn this into a flake;
    they are not tuned to the observed value.
    """
    model, X, y, sample_weight = _fit("interaction")
    payload = _payload(model, X, y, sample_weight, "mean")

    without = _predict_from_payload(payload, X, with_interactions=False)
    relative = np.abs(without - model.predict(X)) / model.predict(X)
    assert relative.max() > 0.05

    cells = np.concatenate(
        [
            block.table.iloc[:, 1:].to_numpy(dtype=np.float64).ravel()
            for block in payload.interactions
        ]
    )
    assert np.abs(np.log(cells)).max() > 0.05


def test_an_interaction_block_carries_no_centering_shift_of_its_own():
    """Interactions are reconstructed from beta and never recentered.

    ``_total_centering_shift`` sums over the MAIN-EFFECT blocks only.  That is
    correct exactly because the interaction path removes no constant, and the
    two exports' interaction cells being bit-identical is what says so -- if an
    interaction were ever centered, its constant would have to join the total
    and the base would be wrong by it.
    """
    model, X, y, sample_weight = _fit("interaction")
    native = _payload(model, X, y, sample_weight, "native")
    mean = _payload(model, X, y, sample_weight, "mean")

    for block_native, block_mean in zip(native.interactions, mean.interactions):
        np.testing.assert_array_equal(
            block_mean.table.iloc[:, 1:].to_numpy(dtype=np.float64),
            block_native.table.iloc[:, 1:].to_numpy(dtype=np.float64),
        )
    # ...while the main effects beside them genuinely were shifted, so the
    # equality above is a constraint rather than an observation about a model
    # in which nothing moved at all.
    transferred = np.log(mean.base_relativity) - np.log(native.base_relativity)
    assert abs(transferred) > 0.01


# ── The binned path's ABSOLUTE accuracy ────────────────────────────────────


def _rebinning_bias_bound(payload, X: pd.DataFrame, sample_weight: np.ndarray) -> float:
    """A worst-case bound on the exposure-weighted log bias of the binned blocks.

    Derived, not observed.  ``discretize`` sets each bin's log relativity to the
    exposure-weighted MEAN of the smooth log relativity over the rows in that
    bin, so within a bin the weighted residual sums to zero identically, and
    therefore so does the weighted mean residual over the whole column.  The
    binning error is spread, not bias: it cannot move the exposure-weighted
    geometric mean of the reconstruction at all.

    One thing breaks that identity, and only one: a row that the workbook bins
    differently from ``discretize``.  The block prints its edges through
    ``_format_interval`` at ``.10g``, so a printed edge is within 5e-10
    relative of the exact one and any row inside that band can land one bin to
    either side.  A displaced row contributes at most its own weight share
    times the step between neighbouring bins, so summing

        (weight share of rows within the printing band of an edge)
        x (largest step between neighbouring log relativities)

    over the binned blocks bounds the bias from above.  Every input is read
    from the payload, the frame and the weights -- nothing is fitted to what
    the reconstruction happens to produce.
    """
    total = 0.0
    for block in payload.main_effects:
        if block.kind != "continuous":
            continue
        bounds = [_INTERVAL.match(str(label)) for label in block.table[block.name]]
        assert all(bound is not None for bound in bounds)
        edges = np.array(
            [float(bound.group(1)) for bound in bounds] + [float(bounds[-1].group(2))],
            dtype=np.float64,
        )
        log_relativity = np.log(block.table["Relativity"].to_numpy(dtype=np.float64))
        step = float(np.max(np.abs(np.diff(log_relativity))))

        values = X[block.name].to_numpy(dtype=np.float64)
        # ``.10g`` keeps ten significant digits, so the printed edge is within
        # half a unit in the tenth: 5e-10 relative, floored at absolute for
        # edges near zero.
        band = 5e-10 * np.maximum(np.abs(edges), 1.0)
        displaceable = np.zeros(len(values), dtype=bool)
        for edge, half_width in zip(edges, band):
            displaceable |= np.abs(values - edge) <= half_width
        share = float(sample_weight[displaceable].sum() / sample_weight.sum())
        total += share * step
    return total


def test_the_binned_reconstruction_carries_no_uniform_scale_error():
    """Binned blocks cannot be exact, but they must not be BIASED.

    The centering-invariance test beside this one compares the two centerings
    against each other, so a constant dropped from the binned path -- the exact
    shape of issue #253, on the one path this fix does not route through
    ``centering_shift`` -- cancels and passes.  The exact fixture never reaches
    that path at all.  This is the assertion that closes the gap: a dropped or
    doubled constant multiplies every reconstructed prediction by ``exp(c)``
    and moves the exposure-weighted mean log ratio by exactly ``c``, while the
    binning error itself cannot move it at all.

    The tolerance is the derived re-binning bound, not the observed residual.
    On this fixture it is 8.1e-02 against a measured bias of 5.9e-04, and the
    smallest constant this export actually transfers is the 0.5 the
    shift-accounting test below fixes -- six times the bound.  So the check
    bites on a real regression and is not sized to the headroom it happens to
    have.  It does NOT catch a scale error smaller than the bound; that is what
    the exactly tabulable sweep is for, on the paths where exactness is
    available.
    """
    model, X, y, sample_weight = _fit("all")
    predicted = model.predict(X)

    for centering in _CENTERINGS:
        payload = _payload(model, X, y, sample_weight, centering)
        reconstructed = _predict_from_payload(payload, X)
        bias = float(np.average(np.log(reconstructed) - np.log(predicted), weights=sample_weight))
        bound = _rebinning_bias_bound(payload, X, sample_weight)
        assert bound < 0.1, f"{centering}: bound {bound} is too loose to discriminate"
        assert abs(bias) <= bound, f"{centering}: bias {bias} exceeds derived bound {bound}"


# ── The other mean centering ───────────────────────────────────────────────


def test_the_two_mean_centerings_disagree_on_an_ordered_categorical():
    """``relativities(centering="mean")`` is a SECOND implementation, and differs.

    ``term_inference`` centers through ``_recenter_term``, which never reaches
    an ``OrderedCategorical``; ``relativities`` centers through
    ``_term_model_ops._center_df``, which does, and records no shift while
    doing it.  Nothing in the export consumes the second one today, which is
    why this PR could fix the first alone.  It is pinned rather than
    reconciled because reconciling it moves values other tests fix, and the
    reader who next wires ``centering_shift`` into the plot-data path needs
    the divergence measured rather than discovered.
    """
    model, X, y, sample_weight = _fit("exact")
    native = model.relativities(centering="native")
    centered = model.relativities(centering="mean")

    def _frame_shift(name: str) -> float:
        removed = np.asarray(native[name]["log_relativity"], dtype=np.float64) - np.asarray(
            centered[name]["log_relativity"], dtype=np.float64
        )
        assert np.ptp(removed) < 1e-12, f"{name}: relativities() shift is not a constant"
        return float(removed[0])

    # The OrderedCategorical: one path shifts it, the other does not.
    assert model.term_inference("band", centering="mean").centering_shift == 0.0
    assert abs(_frame_shift("band")) > 0.01

    # Every other term type agrees exactly, so the disagreement above is
    # specific to the OrderedCategorical branch and not a general drift
    # between two independently written centerings.
    for name in ("region", "territory", "x"):
        assert _frame_shift(name) == pytest.approx(
            model.term_inference(name, centering="mean").centering_shift, abs=1e-12
        )


# ── The base relativity is the one number with no tolerable approximation ──


def test_an_unrepresentable_base_relativity_stops_the_export(monkeypatch):
    """A base that leaves float64 range must raise, not ship as ``inf``.

    Every relativity the centering constant came OUT of goes through
    ``_safe_exp``, which clips at +/- 500.  The base is the sum of the
    ordered-reference intercept and everything the blocks gave back, so it
    reaches further than any single term, and clipping it would emit a
    complete-looking workbook that rates every risk by a factor of
    ``exp(clip)`` off the model -- the same silent uniform error as issue #253,
    reintroduced at the one cell that multiplies the whole tariff.

    The intercept is substituted rather than fitted: reaching 709 by fitting
    would need a response near ``e**709``, so the only honest way to exercise
    the guard is to hand it the argument directly.
    """
    model, X, y, sample_weight = _fit("exact")

    for intercept in (800.0, -800.0):
        monkeypatch.setattr(
            rating_tables,
            "ordered_reference_intercept",
            lambda *args, intercept=intercept, **kwargs: intercept,
        )
        # Not merely non-finite: the negative tail flushes to exactly 0.0, which
        # is just as unusable and just as quiet, because every RATIO in the
        # workbook still reads correctly.
        with np.errstate(over="ignore", under="ignore"):
            assert np.exp(intercept) in (np.inf, 0.0)
        with pytest.raises(RatingTableBaseNotRepresentableError, match="base relativity"):
            _payload(model, X, y, sample_weight, "mean")


def test_a_representable_base_is_left_exactly_alone():
    """The guard rejects; it never repairs, rounds or clips a usable base."""
    model, X, y, sample_weight = _fit("exact")
    payload = _payload(model, X, y, sample_weight, "mean")
    expected = float(
        np.exp(
            rating_tables.ordered_reference_intercept(
                model.result.intercept,
                model.result.beta,
                model._feature_order,
                model._specs,
                model._groups,
            )
            + rating_tables._total_centering_shift(payload.main_effects)
        )
    )
    assert payload.base_relativity == expected


# ── ``centering=`` is validated where it is accepted ───────────────────────


def test_an_unknown_centering_is_rejected_even_with_no_centerable_term():
    """The check has to live at the export boundary, not in a term builder.

    ``_main_effect_inference`` validates ``centering``, but a model whose every
    term is a ``Spline`` or ``Polynomial`` never calls it -- those blocks come
    from the discretisation path -- so such a model used to accept
    ``centering="Mean"`` and quietly export native values under a name the
    caller believed meant something else.
    """
    X, y, sample_weight = _frame()
    model = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        features={"age": Spline(n_knots=8), "score": Polynomial(degree=3)},
    )
    model.fit(X, y, sample_weight=sample_weight)
    assert all(
        block.kind == "continuous"
        for block in _payload(model, X, y, sample_weight, "native").main_effects
    )

    with pytest.raises(ValueError, match="centering must be one of"):
        _payload(model, X, y, sample_weight, "Mean")
