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
  binning error is spread, not scale, so the mean log ratio must stay inside a
  bound derived from the bin geometry -- measured in the geometry weighting
  ``discretize`` bins in, which is the exposure weights for a frequency family
  and unit mass per physical row for ``Tweedie``.  The second exists because
  the first cannot see a constant dropped from the binned path -- both
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

import dataclasses
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
    families,
)
from superglm.diagnostics.discretize import _validated_discretization_weights
from superglm.export import rating_tables
from superglm.export.rating_tables import (
    RatingTableBaseNotRepresentableError,
    build_rating_table_payload,
)
from superglm.features.grouping import collapse_levels
from superglm.features.piecewise import Piecewise
from superglm.links import LogLink, stabilize_eta
from superglm.model.base import predict_eta_raw_exact

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


def _fit_binned_family(family) -> tuple[SuperGLM, pd.DataFrame, np.ndarray, np.ndarray]:
    """A binned fit under a chosen family, for the geometry-measure comparison.

    Its own frame rather than ``_frame``'s: the point of the comparison is the
    weights, so both arms need a response the family can actually carry -- a
    Poisson count and a Tweedie's point mass at zero with a continuous positive
    part -- fitted on identical covariates.

    Not ``@cache``d, unlike ``_fit``.  The Tweedie arm is called as
    ``families.tweedie(p=1.5)``, a fresh object with identity hashing at each
    call site, so that entry could never be hit again and only pinned a fitted
    model for the process lifetime.  One caller, two fits.
    """
    rng = np.random.default_rng(20260813)
    n = 900
    X = pd.DataFrame({"age": rng.uniform(18.0, 80.0, n), "density": rng.uniform(0.0, 1.0, n)})
    mu = np.exp(-1.2 + 0.15 * np.sin(X["age"].to_numpy() / 8.0) + 0.20 * X["density"].to_numpy())
    weight = rng.uniform(0.5, 2.0, n)
    y = (
        rng.poisson(mu * weight).astype(np.float64)
        if family == "poisson"
        else rng.gamma(shape=2.0, scale=mu / 2.0) * (rng.random(n) > 0.3)
    )
    model = SuperGLM(
        family=family,
        selection_penalty=0.0,
        features={"age": Spline(n_knots=8), "density": Numeric()},
    )
    model.fit(X, y, sample_weight=weight)
    return model, X, y, weight


_SEGMENT_LEVELS = ["s1", "s2", "s3"]


def _interaction_frame(
    n: int = 1200, seed: int = 20260813, *, grouped_parent: bool = False
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """A frame for the interaction fixture, kept separate from ``_frame``.

    Its own columns and its own response, so that adding an interaction cannot
    move the fitted values ``test_the_exactly_tabulable_workbook_reproduces_
    the_predictions`` and ``test_the_base_relativity_moves_by_exactly_the_
    shift_the_blocks_applied`` pin.

    The default pair is two plain ``Categorical`` columns: that is the pair the
    export tabulates as a full cell table and the one the payload's exactness
    claim covers.  ``grouped_parent=True`` swaps the first for a ``territory``
    that will carry a ``grouping=``, which is the configuration OUTSIDE that
    claim -- the interaction table is keyed on grouped levels while the
    main-effect block is keyed on the original ones and the workbook carries no
    map between them.  Both arms exist so the exactness claim and its exception
    are each fitted rather than one being an unexplained absence.
    """
    rng = np.random.default_rng(seed)
    first = "territory" if grouped_parent else "region"
    first_levels = _TERRITORY_LEVELS if grouped_parent else ["A", "B", "C"]
    X = pd.DataFrame(
        {
            first: rng.choice(first_levels, n),
            "segment": rng.choice(_SEGMENT_LEVELS, n),
            "density": rng.uniform(0.0, 1.0, n),
            "x": rng.uniform(0.0, 10.0, n),
        }
    )
    parent = X[first].to_numpy()
    segment = X["segment"].to_numpy()
    eta = (
        -1.2
        + 0.25 * (parent == first_levels[1])
        + 0.45 * (parent == first_levels[-1])
        + 0.30 * (segment == "s2")
        + 0.55 * (segment == "s3")
        # The interaction proper: a cell that is not the sum of its margins.
        + 0.40 * ((parent == first_levels[-1]) & (segment == "s3"))
        + 0.20 * X["density"].to_numpy()
        + 0.03 * X["x"].to_numpy()
    )
    sample_weight = rng.uniform(0.5, 2.0, n)
    y = rng.poisson(np.exp(eta) * sample_weight).astype(np.float64)
    if grouped_parent:
        # ``x`` would add a Piecewise block that this fixture has no use for.
        X = X.drop(columns=["x"])
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
    if kind == "grouped_interaction":
        X, y, sample_weight = _interaction_frame(grouped_parent=True)
        features: dict = {
            "territory": Categorical(base="first", grouping=_territory_grouping()),
            "segment": Categorical(base="first"),
            "density": Numeric(),
        }
        interactions: list[tuple[str, str]] | None = [("territory", "segment")]
    elif kind == "interaction":
        X, y, sample_weight = _interaction_frame()
        features = _interaction_features()
        interactions = [("region", "segment")]
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


def _nearest(grid: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Index of the closest printed axis value, ties to the lower index."""
    right = np.clip(np.searchsorted(grid, x), 1, len(grid) - 1)
    return np.where((x - grid[right - 1]) <= (grid[right] - x), right - 1, right)


def _interaction_multiplier(block, X: pd.DataFrame) -> np.ndarray:
    """The per-row factor one exported interaction block implies.

    Read the same way a consumer reads it off the sheet: the first column
    carries the row key and is named for the first parent, every remaining
    column header is a key of the second parent, and the cell is the factor.
    The second parent's name comes from the block name, which is the only place
    the sheet records it.

    Two keying disciplines, and the consumer picks between them from the ONE
    thing they hold that the sheet does not: the dtype of their own column.  A
    categorical parent gives an exact label, so the block is a full cell table
    and the lookup is exact.  A continuous parent gives a number that is
    almost never one of the printed axis values, because those are a uniform
    ``n_bins``-point sample of the fitted range -- so the only lookup available
    is the nearest one on each axis, and the factor found is the surface at a
    grid node rather than at the risk.  That gap is the interaction's share of
    the discretisation error, and it is what issue #287 was about.
    """
    table = block.table
    parent1 = str(table.columns[0])
    parent2 = block.name.split(":")[1]
    assert parent2 in X.columns, f"{block.name!r} names a column the frame does not have"

    gridded = all(pd.api.types.is_numeric_dtype(X[parent]) for parent in (parent1, parent2))
    if gridded:
        axis1 = table[parent1].to_numpy(dtype=np.float64)
        axis2 = np.array([float(column) for column in table.columns[1:]], dtype=np.float64)
        cells = table.iloc[:, 1:].to_numpy(dtype=np.float64)
        return cells[
            _nearest(axis1, X[parent1].to_numpy(dtype=np.float64)),
            _nearest(axis2, X[parent2].to_numpy(dtype=np.float64)),
        ]

    levels2 = [str(column) for column in table.columns[1:]]
    cells_by_pair = {
        (str(row[parent1]), level2): float(row[level2])
        for _, row in table.iterrows()
        for level2 in levels2
    }
    return np.array(
        [cells_by_pair[(str(a), str(b))] for a, b in zip(X[parent1], X[parent2])],
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


def test_a_grouped_interaction_parent_keys_its_two_blocks_differently():
    """The exactness claim above stops at a grouped parent, and this says where.

    The interaction fixture uses two ungrouped ``Categorical`` parents, and its
    own docstring says the grouped ``territory`` is avoided deliberately.  That
    is an absence, and an absence pins nothing: the payload docstring claims a
    product contract over "the categorical-by-categorical interaction" with no
    qualification a reader could act on, and nothing here would have noticed.

    So the excluded configuration is fitted and its shape fixed.
    ``_categorical_block`` expands the main effect back to the six original
    levels while ``_interaction_blocks`` keys cells on the four grouped ones,
    and no block carries the map, so a consumer starting from a raw row has
    nothing to look up.  This is a characterisation of a pre-existing gap
    (issue #286) rather than a fix: it fails the moment the export starts
    emitting the mapping or the original-level cells, which is the point --
    whoever does that has to come here and correct the contract beside it.
    """
    model, X, y, sample_weight = _fit("grouped_interaction")
    payload = _payload(model, X, y, sample_weight, "native")

    main = next(block for block in payload.main_effects if block.name == "territory")
    interaction = next(block for block in payload.interactions)
    assert interaction.name == "territory:segment"

    main_keys = [str(level) for level in main.table["territory"]]
    interaction_keys = [str(key) for key in interaction.table[interaction.table.columns[0]]]
    raw_levels = {str(value) for value in X["territory"]}

    # The main effect is keyed on what the frame holds; the interaction is not.
    assert main_keys == _TERRITORY_LEVELS
    assert interaction_keys == ["t1", "t2+t3", "t4+t5", "t6"]
    assert raw_levels - set(interaction_keys) == {"t2", "t3", "t4", "t5"}

    # No block anywhere in the payload carries the original-to-group map, so
    # the gap cannot be closed by a consumer reading the workbook.  Scanned over
    # every CELL and every column HEADER of every main-effect and interaction
    # block -- not the first column of the main effects alone, which would leave
    # a map green if it arrived as an extra column, an interaction-block column
    # or a sheet of its own, and so would promise a tripwire for two of #286's
    # three remedies without having one.
    grouped_label = "t2+t3"
    carriers = sorted(
        block.name
        for block in (*payload.main_effects, *payload.interactions)
        if grouped_label
        in {str(value) for column in block.table.columns for value in block.table[column]}
        | {str(column) for column in block.table.columns}
    )
    assert carriers == ["territory:segment"], (
        f"{grouped_label!r} should appear only where the interaction keys it; found {carriers}"
    )

    # And the map cannot arrive as a new payload FIELD without coming through
    # here either, which is #286's third shape.
    assert {field.name for field in dataclasses.fields(payload)} == {
        "base_relativity",
        "selected_n_bins",
        "main_effects",
        "interactions",
        "discretization_impact",
        "summary",
    }

    # And the consequence: the documented reconstruction cannot be performed.
    with pytest.raises(KeyError):
        _interaction_multiplier(interaction, X)

    # The share it costs, so the entry is a measurement and not an anecdote.
    absent = np.array(
        [str(value) not in set(interaction_keys) for value in X["territory"]], dtype=bool
    )
    assert float(sample_weight[absent].sum() / sample_weight.sum()) > 0.6


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


def _geometry_weight(model, sample_weight: np.ndarray) -> np.ndarray:
    """The measure ``discretize`` bins and averages in, read the way it reads it.

    Not the same thing as ``sample_weight``, and the difference is a family
    branch rather than a detail: non-Tweedie weights are frequency mass, so the
    geometry measure is the weighting itself, while Tweedie weights are prior
    precision and every physical row is given unit geometry mass instead.  The
    zero-bias identity below is a statement about THIS measure, so it is read
    from the same function ``discretize`` reads it from rather than assumed to
    be the caller's weights.
    """
    return _validated_discretization_weights(model, sample_weight, len(sample_weight))[1]


def _binned_bias_roundoff_bound(payload, X: pd.DataFrame, predicted: np.ndarray) -> float:
    """A worst-case bound on the geometry-weighted log bias of the binned blocks.

    Derived, not observed.  ``discretize`` sets each bin's log relativity to the
    geometry-weighted MEAN of the smooth log relativity over the rows in that
    bin, so within a bin the weighted residual sums to zero identically, and
    therefore so does the weighted mean residual over the whole column.  The
    binning error is spread, not bias: it cannot move the geometry-weighted
    geometric mean of the reconstruction at all.

    The measure is ``_geometry_weight``, not the caller's ``sample_weight``.
    They coincide for every non-Tweedie family, which is why this read the
    weights directly and was right by accident; on a Tweedie fit they differ
    and the identity holds only in the former.

    One thing used to break that identity: a row the workbook binned
    differently from ``discretize``, because the block printed its edges at
    ``.10g`` and a ten-digit edge is a different float64 from the exact one.
    That is issue #278, and it is fixed -- the edges are printed at round-trip
    precision, so the workbook's partition IS ``discretize``'s partition and no
    row is displaced at all.  What is left is round-off, and it is what this
    now budgets, in the two places it enters:

    * Each per-row log ratio.  A relative error ``d`` on either path becomes an
      ABSOLUTE error ``d`` on its log, and the weighted mean of bounded errors
      is bounded by the same quantity, so this contributes at most
      ``_RECONSTRUCTION_RTOL`` -- the same 32-rounding accounting, over a
      product with two more (exact) lookups in it.
    * The weighted average itself.  ``|fl(sum a_i) - sum a_i| <= n eps sum|a_i|``
      for any summation order, so dividing through by ``sum w_i`` leaves
      ``n eps max_i |r_i|``, and ``|r_i|`` is bounded by the sum of the logs
      that go into it: the base, one factor per block, and the prediction.

    Every input is read from the payload, the frame and ``model.predict`` --
    the quantities a consumer holds -- and never from the residual, so this is
    a bound and not a fit to the headroom.  Measured on this fixture: 8.0e-13
    (native) and 6.1e-13 (mean), against the 8.1e-02 the printing band used to
    force.
    """
    magnitude = np.abs(np.log(float(payload.base_relativity))) + np.abs(np.log(predicted))
    for block in payload.main_effects:
        magnitude = magnitude + np.abs(np.log(_block_multiplier(block, X)))
    return _RECONSTRUCTION_RTOL + len(X) * _EPS * float(magnitude.max())


def test_the_binned_reconstruction_carries_no_uniform_scale_error():
    """Binned blocks cannot be exact, but they must not be BIASED.

    ``test_the_reconstruction_does_not_depend_on_the_reporting_centering``
    compares the two centerings
    against each other, so a constant dropped from the binned path -- the exact
    shape of issue #253, on the one path this fix does not route through
    ``centering_shift`` -- cancels and passes.  The exact fixture never reaches
    that path at all.  This is the assertion that closes the gap: a dropped or
    doubled constant multiplies every reconstructed prediction by ``exp(c)``
    and moves the exposure-weighted mean log ratio by exactly ``c``, while the
    binning error itself cannot move it at all.

    The tolerance is the derived round-off budget, not the observed residual.
    On this fixture it is 8.0e-13 against a measured bias of 4.1e-17, and the
    smallest constant this export actually transfers is the 0.5 that
    ``test_the_base_relativity_moves_by_exactly_the_shift_the_blocks_applied``
    fixes -- 6.3e+11 times the bound.  So the check
    bites on a real regression and is not sized to the headroom it happens to
    have.  It does NOT catch a scale error smaller than the bound; that is what
    the exactly tabulable sweep is for, on the paths where exactness is
    available.

    The bound used to be 8.1e-02, dominated by the rows the ``.10g`` interval
    strings re-binned (issue #278).  With the edges printed at round-trip
    precision no row is displaced, the bias collapses from 5.9e-04 to round-off,
    and what is left to budget is arithmetic -- eleven orders tighter, and the
    pointwise binning error is untouched at 4.9e-01, because that error was
    never the biased part.

    ``bound < 0.1`` is a guard against the bound going vacuous, and it is the
    same guard as before: it says the budget stays far enough below the 0.5
    constant that a dropped or doubled centering shift cannot hide inside it.
    """
    model, X, y, sample_weight = _fit("all")
    predicted = model.predict(X)
    geometry_weight = _geometry_weight(model, sample_weight)

    for centering in _CENTERINGS:
        payload = _payload(model, X, y, sample_weight, centering)
        reconstructed = _predict_from_payload(payload, X)
        bias = float(np.average(np.log(reconstructed) - np.log(predicted), weights=geometry_weight))
        bound = _binned_bias_roundoff_bound(payload, X, predicted)
        assert bound < 0.1, f"{centering}: bound {bound} is too loose to discriminate"
        assert abs(bias) <= bound, f"{centering}: bias {bias} exceeds derived bound {bound}"


def _printed_edges(block) -> np.ndarray:
    """The bin boundaries a consumer recovers from the interval strings alone."""
    bounds = [_INTERVAL.match(str(label)) for label in block.table[block.table.columns[0]]]
    assert all(bound is not None for bound in bounds), "every row is keyed on an interval"
    return np.array(
        [float(bound.group(1)) for bound in bounds] + [float(bounds[-1].group(2))],
        dtype=np.float64,
    )


def test_the_printed_bin_edges_are_the_edges_the_model_binned_on():
    """A consumer applying the printed table must land every risk in the model's bin.

    This is the assertion the workbook's own numbers could never make.  The
    binned blocks are checked elsewhere for being unbiased in the geometry
    measure and for agreeing between centerings, and both of those compare the
    sheet against itself; the impact sheet quantifies the binning loss, and it
    bins on the exact ``edges`` array.  So every existing check was made on the
    inside of an edge representation that the consumer never sees.

    What the consumer sees is a string.  The block used to print its boundaries
    at ``.10g``, and ten significant digits do not identify a binary64 -- it
    takes seventeen -- so the printed edge was a DIFFERENT number from the one
    the model binned on.  For a reported quantity that would be a rounding.
    Here the map from value to bin is discontinuous at the edge, so an edge
    moved by 5e-10 relative moves every row inside that band a whole bin over,
    and under the default ``exposure_quantile`` the edges ARE data values, so a
    row sits exactly on one by construction.  Measured against the unfixed
    export on this fixture: 302 of 302 printed edges differed from exact, by up
    to 4.99e-09; 133 of 900 rows (14.8%) took a different factor; and the
    reconstruction moved by 2.29e-01 relative.

    Asserted three ways, on the three things a consumer actually does, and each
    is exact rather than toleranced -- ``repr`` round-trips by construction
    (Steele and White 1990's first free-format property), so there is no
    round-off here to allow for.
    """
    from superglm.diagnostics.discretize import _compute_edges

    model, X, y, sample_weight = _fit("all")
    geometry_weight = _geometry_weight(model, sample_weight)
    payload = _payload(model, X, y, sample_weight, "native")

    binned = [block for block in payload.main_effects if block.kind == "continuous"]
    assert [block.name for block in binned] == ["age", "score"], "both binned blocks are here"

    exact_edges = {}
    for block in binned:
        values = X[block.name].to_numpy(dtype=np.float64)
        exact = _compute_edges(
            values, geometry_weight, payload.selected_n_bins, "exposure_quantile"
        )
        printed = _printed_edges(block)
        exact_edges[block.name] = exact

        # 1. The string parses back to the model's own edge, bit for bit.
        assert len(printed) == len(exact)
        np.testing.assert_array_equal(printed, exact)

        # 2. So the printed table induces the model's partition, row for row.
        printed_bin = np.clip(np.digitize(values, printed, right=False), 1, len(printed) - 1)
        exact_bin = np.clip(np.digitize(values, exact, right=False), 1, len(exact) - 1)
        np.testing.assert_array_equal(printed_bin, exact_bin)

    # 3. And the premium a consumer computes off the sheet is the premium the
    #    model's own edges give -- identically, not to a tolerance.
    def with_exact_edges(frame: pd.DataFrame) -> np.ndarray:
        out = np.full(len(frame), float(payload.base_relativity), dtype=np.float64)
        for block in payload.main_effects:
            if block.kind != "continuous":
                out = out * _block_multiplier(block, frame)
                continue
            relativity = block.table["Relativity"].to_numpy(dtype=np.float64)
            edges = exact_edges[block.name]
            index = np.digitize(frame[block.name].to_numpy(dtype=np.float64), edges, right=False)
            out = out * relativity[np.clip(index, 1, len(relativity)) - 1]
        return out

    np.testing.assert_array_equal(_predict_from_payload(payload, X), with_exact_edges(X))


def test_the_binning_measure_is_physical_rows_for_tweedie_and_the_weights_otherwise():
    """The zero-bias identity above is in the GEOMETRY measure, which is family-scoped.

    ``build_rating_table_payload``'s docstring states the binned blocks are
    bias-free in the mean.  In which mean is not a detail: ``discretize`` bins
    and averages in ``geometry_weight``, which is ``sample_weight`` for the
    frequency families and unit mass per physical row for ``Tweedie``, whose
    weights are prior precision rather than case counts.  Stated
    unconditionally the claim is false -- measured on a Tweedie(p=1.5) fit with
    weights on [0.5, 20], the residual mean is 1.2e-18 per physical row and
    -8.7e-04 under the prior weights, and per bin 1.9e-17 against 6.4e-03.

    Asserted through the observable consequence rather than by re-deriving the
    average: if a family bins on physical rows then its binned relativities
    cannot depend on the weights at all, so handing the same fit two very
    different weight vectors has to leave the table bit-identical.  For a
    frequency family the same comparison has to MOVE it, or the test would pass
    on an export that ignored weights everywhere.
    """

    def binned(model, X, y, weight):
        return model.discretization_impact(
            X, y, sample_weight=weight, n_bins=20, features=["age"]
        ).tables["age"]

    for family, expect_invariant in (("poisson", False), (families.tweedie(p=1.5), True)):
        model, X, y, fitted_weight = _fit_binned_family(family)
        flat = np.ones(len(X), dtype=np.float64)
        spread = np.linspace(0.5, 20.0, len(X))

        assert np.allclose(_geometry_weight(model, flat), 1.0)
        assert np.allclose(_geometry_weight(model, spread), 1.0 if expect_invariant else spread)

        under_flat = binned(model, X, y, flat)
        under_spread = binned(model, X, y, spread)

        # The exported Weight column is the EVALUATION weight, so it moves for
        # both families -- which is what makes the relativity comparison a real
        # one rather than a comparison of two identical calls.
        assert not np.array_equal(
            under_flat["sample_weight"].to_numpy(), under_spread["sample_weight"].to_numpy()
        )

        moved = float(
            np.max(
                np.abs(
                    np.log(under_flat["relativity"].to_numpy(dtype=np.float64))
                    - np.log(under_spread["relativity"].to_numpy(dtype=np.float64))
                )
            )
        )
        if expect_invariant:
            np.testing.assert_array_equal(
                under_flat["bin_from"].to_numpy(), under_spread["bin_from"].to_numpy()
            )
            assert moved == 0.0, f"tweedie binned on the weights: max |dlog rel| {moved}"
        else:
            # Measured 2.1e-02; the floor is an order of magnitude below, so a
            # refit that happens to shrink the effect does not flake it.
            assert moved > 2e-03, f"poisson ignored the weights: max |dlog rel| {moved}"


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


@pytest.mark.parametrize("intercept", [-709.0, -720.0, -740.0, -744.0, -745.0])
def test_a_subnormal_base_relativity_stops_the_export(monkeypatch, intercept):
    """Between about -708.4 and -745.1, ``exp`` returns a SUBNORMAL, not zero.

    ``isfinite`` is true and the value is strictly positive, so the guard used
    to pass it -- but a subnormal has no implicit leading mantissa bit, so it
    carries fewer than the full 53 significant bits, and the deficit grows as
    the value shrinks.  The base is therefore not the number the export says it
    is.  Measured on the round trip ``log(exp(x))``, which is the relative
    error of the exported multiplier:

        exp(-720.0) -> 2.03e-313, off by 3.0e-12
        exp(-740.0) -> 4.2e-322,  off by 2.6e-03
        exp(-744.0) -> 1e-323,    off by 2.5e-01
        exp(-745.0) -> 5e-324,    off by 5.6e-01, one significant bit left

    That last one is a workbook whose every premium is out by a factor of
    ``exp(0.56)`` = 1.75 -- precisely the silent uniform mis-rating this guard
    exists to refuse, arriving through the branch that was meant to catch it.

    The cutoff is derived twice over and fitted to nothing.  ``tiny`` is the
    smallest float64 with a full mantissa, so it is exactly where the round
    trip stops degrading; and it is also, to the five figures Microsoft
    publishes, Excel's "smallest allowed positive number" 2.2251E-308, because
    Excel declines to implement IEEE 754's denormals for this very reason
    ("denormalized numbers by their very nature have a variable number of
    significant digits", *Floating-point arithmetic may give inaccurate result
    in Excel*) and flushes anything smaller to zero on load.
    """
    model, X, y, sample_weight = _fit("exact")
    monkeypatch.setattr(
        rating_tables,
        "ordered_reference_intercept",
        lambda *args, **kwargs: intercept,
    )

    # The premise: this really is the subnormal window, not the flush-to-zero
    # one the finite/positive check already covered.
    with np.errstate(under="ignore"):
        base = float(np.exp(intercept))
    assert 0.0 < base < np.finfo(np.float64).tiny

    with pytest.raises(RatingTableBaseNotRepresentableError, match="base relativity"):
        _payload(model, X, y, sample_weight, "mean")


def test_the_smallest_base_the_export_accepts_is_the_smallest_exact_one():
    """The guard rejects subnormals and nothing above them -- both sides pinned.

    A cutoff that only ever rejects is untestable in the direction that
    matters: ``base < 1.0`` would satisfy the test above and refuse most real
    Poisson tariffs.  So the accepted side is fixed here too.

    Both sides are addressed through ``log``, and ``exp(log(v))`` is NOT the
    identity down here.  Half an ulp of a number near 708.4 is 5.7e-14
    ABSOLUTE, and an absolute error in an exponent is a RELATIVE error in its
    ``exp``, so the round trip can carry ~256 ulps of ``tiny``; measured on this
    build it lands 124 ulps high, a relative error of 2.75e-14.  The accepted
    and rejected sides are therefore separated by a FACTOR OF TWO, 1.8e13 times
    that smear, so no libm whose ``log`` is within an ulp can make them swap --
    and nothing finer than that may be asserted about the value handed back.

    The previous form asserted ``approx(tiny, rel=1e-15)``, finer by eleven
    orders of magnitude than the round trip it measured, and passed anyway --
    see the tolerance note below, which is the reason and is pinned so it
    cannot come back.
    """
    model, X, y, sample_weight = _fit("exact")
    tiny = float(np.finfo(np.float64).tiny)

    # The cutoff constant itself, which no round trip is involved in.
    assert rating_tables._SMALLEST_EXACT_BASE == tiny

    # ``abs=0.0`` is load-bearing, not decoration.  ``pytest.approx`` accepts a
    # value within EITHER tolerance and supplies an absolute default when only
    # ``rel=`` is given, so down here that default decides the comparison
    # outright and admits anything at all -- including ``0.0``, the one value
    # this guard exists to reject.  Read back from the object rather than
    # asserted as a literal, so this documents pytest's MECHANISM and does not
    # break, misleadingly and inside a test about a float64 cutoff, if pytest
    # ever retunes the constant.
    default_abs = pytest.approx(tiny, rel=1e-15).tolerance
    assert default_abs > tiny, (
        f"pytest.approx's absolute default ({default_abs:g}) no longer dwarfs {tiny:g}; "
        "the abs=0.0 below may have stopped being necessary"
    )
    assert 0.0 == pytest.approx(tiny, rel=1e-15)
    assert 0.0 != pytest.approx(tiny, rel=1e-15, abs=0.0)

    # Just inside, one exponent step above the cutoff.  Tolerance derived: half
    # an ulp of |log(2*tiny)| is 5.7e-14 relative, a full ulp of drift on
    # another libm is 1.1e-13, plus exp's own <= 1 ulp -- so 1e-12 is 5.5x the
    # worst case and still 5.5e11 times tighter than the factor of two.
    accepted = rating_tables._base_relativity(float(np.log(2.0 * tiny)))
    assert accepted >= tiny
    assert accepted == pytest.approx(2.0 * tiny, rel=1e-12, abs=0.0)

    # Just outside: one exponent step below, in the subnormals.
    with pytest.raises(RatingTableBaseNotRepresentableError, match="base relativity"):
        rating_tables._base_relativity(float(np.log(tiny / 2.0)))

    # And an ordinary fit is nowhere near either edge, so the guard is not
    # quietly rejecting the models this export exists for.
    payload = _payload(model, X, y, sample_weight, "mean")
    assert payload.base_relativity > 1e-6


# ── The product contract is a statement about the LINK ─────────────────────


@pytest.mark.parametrize(
    ("family", "link", "response"),
    [
        ("gaussian", "identity", "continuous"),
        ("binomial", "logit", "bernoulli"),
    ],
)
def test_a_non_log_link_model_cannot_be_exported_at_all(family, link, response):
    """Every block is an ``exp``, so the product is the prediction only under log.

    Under any other link the same arithmetic runs to completion and emits a
    workbook of the wrong quantity, with nothing raised and every relativity
    ratio internally consistent -- the shape of error this module keeps having
    to refuse.  Measured before the guard, on a three-factor fit:

        gaussian/identity: the product is ``exp(linear_predictor)``, 9.19
            maximum relative error; row 0 reconstructs 25.5680 against a
            prediction of 3.2413, and log(25.5680) = 3.2413.
        binomial/logit:    the product is the ODDS; row 0 reconstructs 2.0201
            against a predicted probability of 0.6689, and
            0.6689 / (1 - 0.6689) = 2.0201.

    Refused rather than repaired, because there is nothing to repair it into:
    applying the inverse link to the product would stop the table being a table
    of factors, and a logit model has no multiplicative tariff to export.  The
    offset path has required this since before issue #253
    (``test_fit_offset_export_rejects_non_log_link_model``); what was missing is
    that a model WITHOUT an offset never reached a link check.
    """
    rng = np.random.default_rng(4)
    n = 600
    X = pd.DataFrame(
        {"region": rng.choice(["A", "B", "C"], n), "density": rng.uniform(0.0, 1.0, n)}
    )
    region = X["region"].to_numpy()
    eta = 2.0 + 0.5 * (region == "B") + 0.9 * (region == "C") + 0.7 * X["density"].to_numpy()
    y = (
        eta + rng.normal(0.0, 0.1, n)
        if response == "continuous"
        else (rng.random(n) < 1.0 / (1.0 + np.exp(-(eta - 2.5)))).astype(np.float64)
    )
    model = SuperGLM(
        family=family,
        link=link,
        selection_penalty=0.0,
        features={"region": Categorical(base="first"), "density": Numeric()},
    )
    model.fit(X, y)

    with pytest.raises(ValueError, match="log-link models"):
        build_rating_table_payload(model, X, y, n_bins=10, impact_bins=(10,))


def test_the_log_link_families_this_export_exists_for_are_not_refused():
    """The guard rejects a link, not a family -- otherwise it would be untestable.

    ``"gaussian"`` above is refused for its identity link, so without this the
    check could have been ``family == "poisson"`` and passed.  Gamma and
    Tweedie are the other two ratemaking families and both carry a log link
    here; each must still export and still reproduce its predictions.
    """
    rng = np.random.default_rng(11)
    n = 400
    X = pd.DataFrame(
        {"region": rng.choice(["A", "B", "C"], n), "density": rng.uniform(0.0, 1.0, n)}
    )
    region = X["region"].to_numpy()
    mu = np.exp(-0.4 + 0.3 * (region == "B") + 0.6 * (region == "C"))

    for family, y in (
        ("poisson", rng.poisson(mu).astype(np.float64)),
        ("gamma", rng.gamma(shape=4.0, scale=mu / 4.0)),
        (families.tweedie(p=1.5), rng.gamma(shape=2.0, scale=mu / 2.0) * (rng.random(n) > 0.3)),
    ):
        model = SuperGLM(
            family=family,
            selection_penalty=0.0,
            features={"region": Categorical(base="first"), "density": Numeric()},
        )
        model.fit(X, y)
        assert isinstance(model._link, LogLink), f"{family} is not log-linked here"

        payload = build_rating_table_payload(model, X, y, n_bins=10, impact_bins=(10,))
        np.testing.assert_allclose(
            _predict_from_payload(payload, X),
            model.predict(X),
            rtol=_RECONSTRUCTION_RTOL,
            atol=0.0,
        )


# ── The product contract is also a statement about the RANGE ───────────────


@cache
def _saturating_fit():
    """A converged log-link Poisson fit whose own frame stays inside the clip.

    Nothing here is degenerate: ``sum_insured`` runs over [0, 10], the fitted
    predictor reaches 19.0, and the exported base relativity is an ordinary
    number, so no existing guard has anything to say.  What reaches the
    stabilization boundary is applying the exported table to LARGER risks --
    which is what a filed tariff is for -- because a ``Numeric`` block is one
    per-unit relativity raised to whatever the consumer holds.
    """
    rng = np.random.default_rng(11)
    n = 800
    x = rng.uniform(0.0, 10.0, n)
    X = pd.DataFrame({"sum_insured": x})
    y = rng.poisson(np.minimum(np.exp(-1.0 + 2.0 * x), 1e12)).astype(np.float64)
    model = SuperGLM(family="poisson", selection_penalty=0.0, features={"sum_insured": Numeric()})
    model.fit(X, y)
    return model, X, y


def test_the_exported_table_keeps_going_where_the_model_saturates():
    """Inside the stabilization range the contract holds; outside it is exact too.

    ``model.predict`` clips a log-link predictor to [-80, 80] before the inverse
    link.  The workbook has no such bound, because a clamp is not a factor and
    no block could carry one.  So the two agree to round-off below the clip and
    diverge above it by exactly ``exp(eta - 80)`` -- not by "some amount", which
    is the difference between a measurement and a hedge, and is what lets the
    payload contract state the boundary instead of shrugging at it.

    Measured here: 1.78e+08 at eta 99.0, first breaching the round-off claim at
    eta 80.41.  The export refuses a frame that saturates
    (``test_a_saturated_export_frame_stops_the_export``); this is the case that
    guard cannot reach, because the payload is frame-independent and these rows
    are rated after it is written.
    """
    model, X, y = _saturating_fit()
    payload = build_rating_table_payload(model, X, y, n_bins=10, impact_bins=(10,))
    assert payload.base_relativity > 1e-6, "an ordinary base; no existing guard fires"

    rated = pd.DataFrame({"sum_insured": np.linspace(0.0, 50.0, 200)})
    eta = predict_eta_raw_exact(model, rated)
    stabilized = stabilize_eta(eta, model._link)
    inside = eta == stabilized
    assert inside.any() and not inside.all(), "the sweep must straddle the clip"

    reconstructed = _predict_from_payload(payload, rated)
    predicted = model.predict(rated)

    # This fixture does NOT get the 32 eps the exactly-tabulable one derives,
    # and the reason is CONDITIONING rather than operation count.  Every
    # quantity compared here is an ``exp``, and ``exp`` has condition number
    # ``|z|``: an absolute error in the exponent comes out as ``|z|`` times
    # itself, relative, in the value.  Two terms follow, and the second is the
    # one that dominates:
    #
    #   * the numeric block's multiplier is ``relativity ** x``.  One rounding
    #     of ``exp`` inside ``relativity`` is an absolute error in
    #     ``log(relativity)``, which the power multiplies by ``x`` -- so ``x``
    #     units of eps, with ``x`` running to 50 here rather than staying O(1).
    #   * the exponent itself is ``eta``, formed by a dot product and an
    #     intercept add, and reaching |eta| = 99 on this sweep.  Those roundings
    #     are absolute in ``eta``, so each contributes ``|eta|`` units of eps to
    #     the value -- and both sides of each comparison pay it, the
    #     reconstruction through ``exp(x*log rel)`` and the model through
    #     ``exp(eta)`` (or ``exp(eta - stabilized)``, where the subtraction is
    #     itself exact by Sterbenz, 80 <= 99 <= 160, but carries eta's error in).
    #
    # So the budget is ``x + 2|eta| + O(1)``, not ``x + O(1)``: the earlier form
    # counted the model side as three units of eps when they are three units of
    # ``|eta|``eps, omitting a term four times larger than the one it kept.  It
    # passed here and went red on the 3.12 version floor at 1.53e-14.
    eta_max = float(np.max(np.abs(eta)))
    x_max = float(rated["sum_insured"].max())
    power_rtol = (x_max + 2.0 * eta_max + 8.0) * _EPS

    np.testing.assert_allclose(reconstructed[inside], predicted[inside], rtol=power_rtol, atol=0.0)
    # Outside, the ratio is the clip and nothing else.  Compared against
    # ``stabilize_eta``'s own output rather than a literal 80, so the assertion
    # follows the rule ``predict`` applies instead of a copy of it.
    ratio = reconstructed[~inside] / predicted[~inside]
    np.testing.assert_allclose(
        ratio, np.exp(eta[~inside] - stabilized[~inside]), rtol=power_rtol, atol=0.0
    )
    assert float(ratio.max()) > 1e8, "and it is a mis-rating, not a rounding difference"


def test_a_saturated_export_frame_stops_the_export():
    """The rows the export CAN see are refused rather than shipped.

    Same fitted model as above, exported on the frame that saturates it.  The
    workbook would disagree with ``model.predict`` on the very data it was built
    from, which is the shape of failure this module exists to refuse -- a
    complete-looking sheet whose numbers are not the model's.
    """
    model, _, _ = _saturating_fit()
    rated = pd.DataFrame({"sum_insured": np.linspace(0.0, 50.0, 200)})
    eta = predict_eta_raw_exact(model, rated)
    # 38 is derived, not observed: the fit recovers eta = -1 + 2x, which crosses
    # 80 at x = 40.5, and 38 of the 200 points of linspace(0, 50) sit above that.
    # The adjacent points are 0.5 apart in eta, so the count only moves if beta
    # moves by 1.2e-03 relative -- ten orders of magnitude more than a reordered
    # BLAS can do. Stated so a future refit's red is legible rather than cryptic.
    assert (eta != stabilize_eta(eta, model._link)).sum() == 38

    with pytest.raises(ValueError, match="saturates on this frame"):
        build_rating_table_payload(model, rated, model.predict(rated), n_bins=10, impact_bins=(10,))


def test_a_binomial_model_cannot_be_exported_whatever_its_frame_looks_like():
    """``clip_mu`` bounds a binomial mean, so the family is refused, not the frame.

    ``clip_mu`` clamps a ``Binomial`` mean into [1e-7, 1 - 1e-7].  For the
    positive families the band is [1e-50, 1e50], which the eta clip already sits
    strictly inside -- ``exp(+/-80)`` is [1.8e-35, 5.5e34] -- and ``Gaussian`` is
    not clamped, so ``Binomial`` is the only family a log link can reach it with.

    A level with a 100% event rate is all it takes to make it bite in-frame: the
    MLE puts it at ``mu = 1`` and the fit lands slightly above, at eta +0.3647,
    so ``exp`` returns a "probability" of 1.4401 where ``predict`` returns
    0.9999999 -- 974 of 3000 rows rewritten, 4.40e-01 maximum relative error
    against a documented round-off claim of 7.1e-15.

    But the refusal is by FAMILY, and the second fixture below is why.  A
    binomial whose every fitted mean sits inside the clamp exports with nothing
    to complain about, and then breaks on a row rated later: the usable band is
    ``-16.118 <= eta <= -1.0e-7``, a mere 20.1% of the ``[-80, 0]`` a log link
    permits, and ``mu > 1`` out of sample is the characteristic hazard of
    log-binomial regression rather than an exotic corner.  A frame-scoped gate
    would have passed that model and shipped the workbook.

    The lower clamp, by contrast, is out of reach: a zero-event level converges
    to eta -15.03, a mean of 2.98e-07, above the 1e-7 floor -- 0 rows clamped at
    n=8000.  Recorded so the next reader does not go looking for it.
    """
    rng = np.random.default_rng(3)
    n = 3000
    region = rng.choice(["A", "B", "C"], n)
    X = pd.DataFrame({"region": region})
    p = np.where(region == "A", 0.30, np.where(region == "B", 0.10, 1.0))
    y = (rng.random(n) < p).astype(np.float64)

    model = SuperGLM(
        family="binomial",
        link="log",
        selection_penalty=0.0,
        features={"region": Categorical(base="first")},
    )
    model.fit(X, y)
    assert isinstance(model._link, LogLink), "the link gate must not be what refuses this"

    # The clamp is what bites, not the eta clip: the predictor is nowhere near 80.
    eta = predict_eta_raw_exact(model, X)
    assert float(np.abs(eta).max()) < 80.0
    assert float(np.exp(eta).max()) > 1.0, "a mean above one is what the clamp catches"

    with pytest.raises(ValueError, match="not supported for Binomial"):
        build_rating_table_payload(model, X, y, n_bins=10, impact_bins=(10,))

    # And the case a frame-scoped gate would have let through: every fitted mean
    # inside the clamp, so nothing is visible at export time.
    rng2 = np.random.default_rng(31)
    n2 = 1200
    X2 = pd.DataFrame({"dose": rng2.uniform(0.0, 1.0, n2)})
    mu2 = np.exp(-2.0 + 0.5 * X2["dose"].to_numpy())
    y2 = (rng2.random(n2) < mu2).astype(np.float64)
    benign = SuperGLM(
        family="binomial", link="log", selection_penalty=0.0, features={"dose": Numeric()}
    )
    benign.fit(X2, y2)
    eta2 = predict_eta_raw_exact(benign, X2)
    mu_pre = benign._link.inverse(stabilize_eta(eta2, benign._link))
    assert np.array_equal(mu_pre, benign.predict(X2)), "nothing is clamped on this frame"

    # The block is still one per-unit relativity raised to whatever a consumer
    # holds, so a larger dose walks the "probability" straight past one.
    beta = float(np.asarray(benign.result.beta, dtype=np.float64).ravel()[0])
    dose_at_one = -float(benign.result.intercept) / beta
    assert dose_at_one > 1.0, "the fitted frame stays under mu = 1, as intended"
    far = pd.DataFrame({"dose": [dose_at_one * 2.0]})
    assert float(np.exp(predict_eta_raw_exact(benign, far))[0]) > 1.0
    assert float(benign.predict(far)[0]) <= 1.0 - 1e-7

    with pytest.raises(ValueError, match="not supported for Binomial"):
        build_rating_table_payload(benign, X2, y2, n_bins=10, impact_bins=(10,))


def test_a_continuous_offset_exports_a_binned_block_that_is_not_a_row_exact_factor():
    """The offset multiplier block is binned above 20 distinct values, not looked up.

    ``_offset_multiplier_block`` emits one exact row per distinct multiplier only
    while there are fewer than 20 of them.  At 20 or more -- the normal case for
    a continuous exposure -- it bins them like a continuous block: rows keyed on
    interval STRINGS, each carrying the exposure-weighted average multiplier of
    its bin.  So a consumer cannot look its own multiplier up at all, and the
    factor it does find is an average.

    This is a characterisation of a pre-existing shape, not a fix.  What is new
    is that the payload contract claimed row-exact equivalence over every
    main-effect block without excepting this one, and the equivalence tests
    reconstruct through the ``offset_source=`` form, which IS an exact lookup,
    so nothing here would have noticed.
    """
    rng = np.random.default_rng(19)
    n = 800
    X = pd.DataFrame({"region": rng.choice(["A", "B", "C"], n)})
    exposure = rng.uniform(0.1, 2.0, n)
    offset = np.log(exposure)
    region = X["region"].to_numpy()
    y = rng.poisson(np.exp(-1.0 + 0.4 * (region == "B") + 0.8 * (region == "C")) * exposure).astype(
        np.float64
    )

    model = SuperGLM(
        family="poisson", selection_penalty=0.0, features={"region": Categorical(base="first")}
    )
    model.fit(X, y, offset=offset)
    payload = build_rating_table_payload(model, X, y, offset=offset, n_bins=150, impact_bins=(20,))

    block = next(b for b in payload.main_effects if b.kind == "offset")
    multiplier = np.exp(offset)
    assert len(np.unique(np.round(multiplier, 12))) == n >= 20, "well past the exact-row limit"
    assert len(block.table) == 150 < n, "binned, so it cannot be one row per multiplier"

    # Keyed on interval strings, which is what makes an exact lookup impossible.
    bounds = [_INTERVAL.match(str(label)) for label in block.table.iloc[:, 0]]
    assert all(bound is not None for bound in bounds)

    edges = np.array(
        [float(b.group(1)) for b in bounds] + [float(bounds[-1].group(2))], dtype=np.float64
    )
    relativity = block.table["Relativity"].to_numpy(dtype=np.float64)
    applied = relativity[
        np.clip(np.digitize(multiplier, edges, right=False), 1, len(relativity)) - 1
    ]

    # Every row gets a factor that is not its own multiplier.
    err = np.abs(applied - multiplier) / multiplier
    assert int(np.count_nonzero(err > 1e-12)) == n
    assert float(err.max()) == pytest.approx(8.861e-02, rel=1e-3)

    # And so the documented reconstruction misses by a binning error, not round-off.
    reconstructed = np.full(n, float(payload.base_relativity), dtype=np.float64)
    for other in payload.main_effects:
        if other.kind == "categorical":
            lookup = {
                str(k): float(v) for k, v in zip(other.table[other.name], other.table["Relativity"])
            }
            reconstructed = reconstructed * np.array(
                [lookup[str(value)] for value in X[other.name]], dtype=np.float64
            )
    reconstructed = reconstructed * applied
    relative = np.abs(reconstructed - model.predict(X, offset=offset)) / model.predict(
        X, offset=offset
    )
    assert float(relative.max()) > 1e-2 > _RECONSTRUCTION_RTOL


def test_an_empty_offset_bin_ships_a_factor_a_risk_can_be_rated_on():
    """An empty bin still ships a row, so it still ships a factor.

    ``_offset_multiplier_block`` bins the exposure multiplier above 20 distinct
    values.  Under the default ``bin_strategy="exposure_quantile"`` no bin can
    be empty -- every edge is a data value, so each bin holds at least the row
    on its own left edge.  ``bin_strategy`` is a public argument, and
    ``"uniform"`` is ``linspace(min, max, n_bins + 1)``: on a skewed exposure
    with a gap in it, empty bins are the normal outcome rather than a corner.

    Those bins used to report ``Relativity = 0.0`` (issue #291).  Measured
    against the unfixed export on this fixture: 123 of 150 bins, and a risk
    whose multiplier fell in one priced at exactly zero -- while every other
    number on the sheet, including every relativity ratio, stayed correct.  That
    is the failure shape of issue #253 one level down, and it needs no extreme
    coefficient at all.

    Asserted on what a consumer computes, not on the fill rule: for a multiplier
    swept across every bin the block prints, the factor it finds must be a
    multiplier the bin could actually contain, and the premium must be positive.
    A rule that filled with ``1.0`` would pass "positive" and fail "inside the
    interval", which is why both are here.
    """
    rng = np.random.default_rng(2026)
    n = 800
    X = pd.DataFrame({"region": rng.choice(["A", "B", "C"], n)})
    # A skewed exposure with a real gap: nearly all the mass near 0.1, a tail at 2.
    exposure = np.concatenate([rng.uniform(0.05, 0.25, n - 20), rng.uniform(1.8, 2.0, 20)])
    rng.shuffle(exposure)
    offset = np.log(exposure)
    region = X["region"].to_numpy()
    y = rng.poisson(np.exp(-1.0 + 0.4 * (region == "B") + 0.8 * (region == "C")) * exposure).astype(
        np.float64
    )

    model = SuperGLM(
        family="poisson", selection_penalty=0.0, features={"region": Categorical(base="first")}
    )
    model.fit(X, y, offset=offset)
    payload = build_rating_table_payload(
        model, X, y, offset=offset, n_bins=150, impact_bins=(20,), bin_strategy="uniform"
    )

    block = next(b for b in payload.main_effects if b.kind == "offset")
    relativity = block.table["Relativity"].to_numpy(dtype=np.float64)
    weight = block.table["Weight"].to_numpy(dtype=np.float64)
    edges = _printed_edges(block)

    # The fixture reaches the branch at all, which is the half a fill rule
    # cannot fake: without empty bins there is nothing here to get wrong.
    assert int(np.count_nonzero(weight == 0.0)) > 0, "the uniform strategy leaves bins empty"

    # A risk is rated by keying its own multiplier into the printed intervals.
    probe = 0.5 * (edges[:-1] + edges[1:])
    factor = relativity[np.clip(np.digitize(probe, edges, right=False), 1, len(relativity)) - 1]
    assert np.all(factor > 0.0), "no bin prices a risk at zero"
    # A weighted average of values inside an interval is inside it, so the only
    # slack this needs is the average's own round-off: ``np.average`` over at
    # most n values, i.e. n eps relative.  Not fitted -- it is eleven orders
    # below the gap a fill of 1.0 would leave against a bin spanning [0.05, 2.0].
    slack = len(X) * _EPS * np.abs(edges)
    assert np.all((factor >= edges[:-1] - slack[:-1]) & (factor <= edges[1:] + slack[1:])), (
        "each bin's factor is a multiplier that bin could hold"
    )

    # And the offset block's boundaries round-trip too -- it shares the printer.
    # Binned on ``exp(offset)``, which is what the block bins and what a consumer
    # holds, rather than on ``exposure``: ``exp(log(x))`` is not ``x`` in general,
    # and pinning the wrong array would make this pass or fail on an ulp.
    from superglm.diagnostics.discretize import _compute_edges

    np.testing.assert_array_equal(edges, _compute_edges(np.exp(offset), np.ones(n), 150, "uniform"))


def test_a_bin_holding_only_zero_weight_rows_ships_a_factor_too():
    """ "Empty" and "carries no weight" are different conditions, and both reach here.

    ``sample_weight`` is only validated non-negative, and ``_compute_edges``
    builds ``"uniform"`` edges from ``x[sample_weight > 0.0]`` -- so the edge
    range is set by the positive-weight rows while ``np.digitize`` bins all of
    them. A zero-weight row sitting in a gap between two positive-weight
    clusters therefore lands alone in a bin that is NOT empty and carries no
    weight at all, and the weighted-average branch raised
    ``ZeroDivisionError: Weights sum to zero, can't be normalized`` from inside
    NumPy.

    Loud rather than silent, and pre-existing, so it is not the defect this
    module is about -- but it is the same bin under a different predicate, and
    a bin with no weight has no weighted mean to report whether or not it has
    rows. Both take the midpoint.
    """
    rng = np.random.default_rng(4)
    n = 400
    X = pd.DataFrame({"region": rng.choice(["A", "B"], n)})
    exposure = np.concatenate(
        [rng.uniform(0.05, 0.25, n - 13), rng.uniform(1.8, 2.0, 10), [0.9, 1.0, 1.1]]
    )
    sample_weight = np.ones(n)
    # The three rows in the gap are the ones that carry nothing.
    sample_weight[-3:] = 0.0
    offset = np.log(exposure)
    y = rng.poisson(np.exp(-1.0 + 0.4 * (X["region"].to_numpy() == "B")) * exposure).astype(
        np.float64
    )

    model = SuperGLM(
        family="poisson", selection_penalty=0.0, features={"region": Categorical(base="first")}
    )
    model.fit(X, y, sample_weight=sample_weight, offset=offset)
    payload = build_rating_table_payload(
        model,
        X,
        y,
        sample_weight=sample_weight,
        offset=offset,
        n_bins=150,
        impact_bins=(20,),
        bin_strategy="uniform",
    )

    block = next(b for b in payload.main_effects if b.kind == "offset")
    edges = _printed_edges(block)
    relativity = block.table["Relativity"].to_numpy(dtype=np.float64)

    # The configuration really is the one described: a bin with rows in it and
    # no weight behind them.
    bin_index = np.clip(np.digitize(exposure, edges, right=False), 1, len(relativity)) - 1
    weightless = [
        b
        for b in range(len(relativity))
        if np.any(bin_index == b) and float(sample_weight[bin_index == b].sum()) == 0.0
    ]
    assert weightless, "a non-empty bin whose rows carry no weight"

    probe = 0.5 * (edges[:-1] + edges[1:])
    factor = relativity[np.clip(np.digitize(probe, edges, right=False), 1, len(relativity)) - 1]
    assert np.all(factor > 0.0), "no bin prices a risk at zero"
    slack = len(X) * _EPS * np.abs(edges)
    assert np.all((factor >= edges[:-1] - slack[:-1]) & (factor <= edges[1:] + slack[1:]))


def test_a_relativity_a_consumer_cannot_multiply_by_stops_the_export():
    """Two different mechanisms produce an unusable factor, and both are refused.

    ``_safe_exp`` clips its argument to +/-500 so a quasi-separated CONFIDENCE
    BOUND comes back finite.  Right for a bound, wrong for a factor: the clipped
    value is representable -- 1.4e+217 and 7.1e-218 -- so a check for ``inf`` or
    ``0.0`` never fires on it and only a comparison against the endpoints
    catches it.  That is the ``Piecewise`` path in both centerings, and
    ``Categorical`` under ``centering="mean"``.

    Every other path exponentiates with a plain ``np.exp``, where the failure is
    the opposite: ``inf``, a subnormal, or exactly ``0.0``.  So the guard has to
    carry both arms, which is why it is stated over the emitted value rather
    than over which routine produced it.

    ``0.0`` is refused, and that used to be carved out (issue #291).  It is the
    single worst value a multiplicative tariff can carry: it prices every risk
    it covers at zero while every relativity RATIO on the sheet still reads
    correctly -- the same silent shape ``_base_relativity`` already refuses an
    infinite or zero base for.

    The reason it needs its own guard, rather than riding on the base's, is
    CANCELLATION.  Term contributions of +800 and -700 have an ordinary product,
    ``exp(100)``; clipped they become ``exp(500) * exp(-500) = 1``, so the
    workbook rates every such risk 2.7e+43 low while the predictor -- and
    therefore the base guard and the saturation gate, which both look at the
    SUM -- stay entirely healthy.  The sum is well behaved exactly when the
    parts are not.

    The blocks here are CONSTRUCTED, and that is stated rather than dressed up:
    no fitted main effect reaches either endpoint.  A fitted reproduction does
    exist one level out, on the interaction cells, and it has its own test
    below.
    """
    from superglm.inference._term_types import _MAX_LOG_REL

    ceiling = float(np.exp(_MAX_LOG_REL))
    floor = float(np.exp(-_MAX_LOG_REL))
    assert np.isfinite(ceiling) and floor > 0.0, "both clip endpoints are representable"
    # The mechanism the clip check exists for: a clipped factor is an ordinary
    # finite positive number, so no range check on inf/0 would ever see it.
    from superglm.inference._term_types import _safe_exp

    assert float(_safe_exp(800.0)) == ceiling and float(_safe_exp(-800.0)) == floor
    with np.errstate(over="ignore"):
        assert np.isinf(np.exp(800.0)) and np.exp(-800.0) == 0.0

    def block(*values: float) -> rating_tables.RatingTableBlock:
        return rating_tables.RatingTableBlock(
            name="region",
            kind="categorical",
            table=pd.DataFrame(
                {
                    "region": ["a"] * len(values),
                    "Relativity": list(values),
                    "Weight": [1.0] * len(values),
                }
            ),
        )

    # Accepted: one ulp inside each endpoint, so the guard rejects the clip and
    # not merely "a large number" -- without this it could have been `> 1e10`.
    rating_tables._require_usable_relativities_export(
        [block(float(np.nextafter(ceiling, 0.0)), float(np.nextafter(floor, np.inf)), 1.0)], []
    )

    # Both sides of 0.0, which the guard used to cover on neither.
    for bad in (ceiling, floor, 0.0, -1.0, 5e-324, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="cannot multiply by"):
            rating_tables._require_usable_relativities_export([block(1.0, bad)], [])

    # A block whose factor column is named something else is caught, not exempt.
    with pytest.raises(ValueError, match="no 'Relativity' column"):
        rating_tables._require_usable_relativities_export(
            [
                rating_tables.RatingTableBlock(
                    name="region",
                    kind="categorical",
                    table=pd.DataFrame({"region": ["a"], "Factor": [0.0]}),
                )
            ],
            [],
        )

    # And an ordinary tariff is nowhere near either endpoint.
    model, X, y, sample_weight = _fit("exact")
    payload = _payload(model, X, y, sample_weight, "mean")
    for emitted in payload.main_effects:
        values = np.asarray(emitted.table["Relativity"], dtype=np.float64)
        assert np.all((values > floor) & (values < ceiling))


def _diagonal_band_interaction_fit():
    """Two continuous parents whose data lies along ``a == b``, fitted honestly.

    Nothing here is extreme.  The response is an ordinary Poisson count, the
    coefficients come out of the fit untouched, and every row's linear predictor
    stays inside +/-3.14 against a saturation bound of 80.  What is extreme is
    the region the EXPORT samples: ``reconstruct`` lays the interaction on
    ``linspace(lo1, hi1) x linspace(lo2, hi2)``, the parents' bounding BOX, and
    a diagonal band leaves the two off-diagonal corners with no exposure at all.
    The corner cells are therefore a tensor surface extrapolated well outside
    its data, which is where a degree-4 pair goes to 1.8e+155 and to 0.0.
    """
    rng = np.random.default_rng(11)
    n = 800
    u = rng.uniform(0.0, 10.0, n)
    X = pd.DataFrame({"a": u + rng.normal(0.0, 0.25, n), "b": u + rng.normal(0.0, 0.25, n)})
    mu = np.exp(-1.0 + 0.05 * X["a"] + 0.05 * X["b"] + 0.02 * X["a"] * X["b"])
    y = rng.poisson(mu).astype(np.float64)
    model = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        features={"a": Polynomial(degree=4), "b": Polynomial(degree=4)},
        interactions=[("a", "b")],
    )
    model.fit(X, y)
    return model, X, y


def test_an_interaction_cell_that_is_not_a_usable_factor_stops_the_export():
    """Interaction cells are emitted relativities, so they are checked too.

    They were not (issue #289): the guard ran on ``main_effects`` while
    ``_interaction_blocks`` was built afterwards, inside the constructor call.
    Interaction cells are also the only exported factors that never touch
    ``_safe_exp`` under any centering, so they were the one place all three of
    the disciplines this module applies to the base -- reject ``inf``, reject
    ``0.0``, reject subnormal -- were absent together.

    Fitted, not constructed, and it needs no cancellation: the export samples
    the interaction on the parents' bounding box while the data lies along a
    diagonal band, so the corner cells are pure extrapolation with no exposure
    behind them.  Measured on the fixture below, against the UNFIXED export,
    which succeeded and shipped the workbook:

        3 of 400 interaction cells were exactly 0.0 and the smallest non-zero
        one was 1e-262, while every row's |eta| stayed at 3.14 against a bound
        of 80, the base relativity was 9.66e-24 and representable, and every
        main-effect relativity sat inside the clip.

    So the saturation gate, the base guard and the per-block guard were all
    silent -- correctly, each on its own terms -- and a risk keyed into a corner
    cell priced at exactly zero.

    It is the FLOOR arm that catches this fixture and not the ceiling: the
    largest cell, 1.8e+155, is well inside ``exp(+500)`` = 1.4e+217.  A guard
    written only against the overflow side would still ship this workbook.  That
    is pinned below on constructed cells rather than on these, because where a
    degree-4 tensor lands 90 units outside its data is a property of the LAPACK
    that fitted it.
    """
    model, X, y = _diagonal_band_interaction_fit()

    # The reachability claim, stated as the state of the fit rather than assumed:
    # every other gate really is silent here, so this is not a fit that any
    # existing check would have refused anyway.
    raw = predict_eta_raw_exact(model, X)
    assert np.all(raw == stabilize_eta(raw, model._link)), "no row saturates"

    with pytest.raises(ValueError, match="cannot multiply by"):
        build_rating_table_payload(model, X, y, n_bins=20, impact_bins=(20,))

    # What was shipping: the cells themselves, read the way _interaction_blocks
    # builds them, with the main-effect half of the payload entirely healthy.
    # Asserted as "some cell leaves the usable band" rather than as the exact
    # value: the fixture reaches 1e-262 and 0.0 here, but it is a degree-4
    # tensor extrapolated far outside its data, so the digits it lands on are a
    # property of this LAPACK.  The band is the invariant, and it is cleared by
    # more than 40 orders of magnitude either way.
    cells = rating_tables._interaction_blocks(model, 20)[0].table.iloc[:, 1:].to_numpy(np.float64)
    unusable = ~np.isfinite(cells) | (cells >= np.exp(500.0)) | (cells <= np.exp(-500.0))
    assert np.any(unusable), "at least one exported cell is not a factor at all"

    # The two arms, on CONSTRUCTED cells so that the mechanism is pinned without
    # depending on where a fitted extrapolation happens to land.  Each is refused
    # only because the guard now SEES interactions: the ``inf`` arm has been in
    # the predicate all along, and the ``0.0`` arm is issue #291's half.
    def interaction(cell: float) -> rating_tables.InteractionTableBlock:
        return rating_tables.InteractionTableBlock(
            name="a:b",
            table=pd.DataFrame(
                {"a": ["0.0", "1.0"], "1.0": [1.0, cell], "2.0": [1.0, 1.0]},
            ),
        )

    for cell in (float("inf"), 0.0, float(np.exp(500.0)), float(np.exp(-500.0))):
        with pytest.raises(ValueError, match="cannot multiply by"):
            rating_tables._require_usable_relativities_export([], [interaction(cell)])
    rating_tables._require_usable_relativities_export([], [interaction(1.0)])

    # An ordinary interaction is untouched, and its first column -- level labels,
    # which may look numeric -- is a key rather than a factor.
    ordinary, X2, y2, w2 = _fit("interaction")
    payload = _payload(ordinary, X2, y2, w2, "native")
    assert payload.interactions, "the fixture really does export an interaction"
    rating_tables._require_usable_relativities_export(payload.main_effects, payload.interactions)


def _grid_interaction_frame(n: int = 600, seed: int = 5):
    """A continuous-by-continuous interaction, which the export ships as a grid."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"age": rng.uniform(18.0, 80.0, n), "density": rng.uniform(0.0, 10.0, n)})
    mu = np.exp(-1.0 + 0.02 * X["age"] + 0.05 * X["density"] + 0.004 * X["age"] * X["density"])
    y = rng.poisson(mu).astype(np.float64)
    model = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        features={"age": Spline(n_knots=6), "density": Spline(n_knots=6)},
        interactions=[("age", "density")],
    )
    model.fit(X, y)
    return model, X, y


def test_the_impact_sheet_covers_the_binned_continuous_interaction():
    """Every block the workbook approximates has to be named on the sheet.

    The sheet exists so a reader can see WHERE the exported table stops being
    the model.  A continuous-by-continuous interaction is sampled onto the same
    lossy ``n_bins`` grid as a binned main effect, so leaving it off the sheet
    tells a reader the workbook approximates two terms when it approximates
    three (issue #287).

    This replaces ``test_the_impact_sheet_does_not_cover_a_continuous_
    interaction``, which characterised the gap and was written to fail here.
    """
    model, X, y = _grid_interaction_frame()

    assert list(model._interaction_order) == ["age:density"]
    assert rating_tables._continuous_features(model) == ["age", "density"]

    payload = build_rating_table_payload(model, X, y, n_bins=20, impact_bins=(10,))

    # The interaction really is exported, and really is on the lossy grid.
    interaction = next(block for block in payload.interactions)
    assert interaction.name == "age:density"
    assert interaction.table.shape == (20, 21)

    sheet = payload.discretization_impact
    assert set(sheet["feature"]) == {"age", "density", "age:density"}

    # And it carries the same information as the main-effect rows, rather than a
    # name beside a row of blanks.  Swept at 10 and again at the exported 20,
    # which the sweep folds in so a row describes the table in hand.
    rows = sheet[(sheet["feature"] == "age:density") & (sheet["n_bins"] == 10)]
    assert len(rows) == 1
    assert not rows.isna().to_numpy().any()
    # ``actual_bins`` counts the block's own table rows, which for a grid is its
    # cells: 10 nodes per axis.
    assert int(rows["actual_bins"].iloc[0]) == 100
    for column in ("deviance_change_pct", "mean_abs_prediction_change_pct"):
        assert float(rows[column].iloc[0]) == pytest.approx(
            float(sheet[(sheet["feature"] == "age") & (sheet["n_bins"] == 10)][column].iloc[0])
        )


# A grid cell is a dot product of two marginal bases against the coefficient
# block, and the exported value and the oracle below reach it by different
# associations -- ``reconstruct`` through a matrix chain, ``score`` through an
# einsum.  Charging each the standard p-term inner-product bound at p = 49
# coefficients, twice, and one ulp for the ``exp``, gives ~100 u; 128 u is that
# rounded up.  Derived from the operation count, not from the observed gap.
_GRID_CELL_RTOL = 128 * _EPS


def test_the_exported_grids_orientation_is_load_bearing():
    """Which axis is which cannot be checked from the grid's shape.

    Both axes carry ``n_points`` nodes, so the surface is square and a
    transposition changes no shape.  Worse, the sweep and the workbook
    reconstruction read it through the SAME transpose, so a joint flip cancels
    in ``test_the_sheets_prediction_change_is_the_whole_workbooks_error`` and
    that test passes with both sides wrong.

    Pinned here against the model instead: the cell under row key ``x1`` and
    column header ``x2`` must be the fitted interaction's own factor at
    ``(x1, x2)``.  The fixture's axes are asymmetric on purpose -- ``age``
    18-80 against ``density`` 0-10 -- so a swapped pair is not merely a
    different number, it is off the other axis's domain entirely, and the
    corner cells below are the strongest form of that.

    The spec is consulted here, unlike everywhere else in this module, because
    it is the ORACLE: the claim is that the sheet agrees with the model, and
    only the model can say what it should have been.
    """
    model, X, y = _grid_interaction_frame()
    payload = build_rating_table_payload(model, X, y, n_bins=12, impact_bins=(12,))
    block = next(b for b in payload.interactions if b.name == "age:density")
    assert block.kind == "grid"

    table = block.table
    axis_age = table["age"].to_numpy(dtype=np.float64)
    axis_density = np.array([float(c) for c in table.columns[1:]], dtype=np.float64)
    assert axis_age[0] != pytest.approx(axis_density[0])

    ispec = model._interaction_specs["age:density"]
    beta = rating_tables._interaction_beta(model, "age:density")
    last = len(axis_age) - 1
    for i, j in ((0, last), (last, 0), (0, 0), (last, last), (3, 8)):
        expected = float(
            np.exp(ispec.score(np.array([axis_age[i]]), np.array([axis_density[j]]), beta))[0]
        )
        assert float(table.iloc[i, j + 1]) == pytest.approx(expected, rel=_GRID_CELL_RTOL), (
            f"cell ({i}, {j}) is not the model's factor at "
            f"(age={axis_age[i]}, density={axis_density[j]})"
        )


def test_the_sheet_describes_the_workbook_that_was_actually_exported():
    """A reader holds ONE table, and the sheet has to have a row about it.

    The defaults are ``n_bins=150`` against ``impact_bins=(20, 50, 100, 200,
    250)``, so before this the sheet described five resolutions and not the one
    shipped -- and since the error falls with resolution, the 200 and 250 rows
    reported LESS movement than the exported table carries.  A reader taking
    the smallest number on the sheet as their bound got one below their own
    error, which is the understatement this whole change is about.
    """
    model, X, y = _grid_interaction_frame()
    payload = build_rating_table_payload(model, X, y, n_bins=37, impact_bins=(20, 50))
    sheet = payload.discretization_impact

    exported = sheet[sheet["exported"]]
    assert set(exported["n_bins"]) == {37}
    assert set(sheet[~sheet["exported"]]["n_bins"]) == {20, 50}
    assert set(exported["feature"]) == {"age", "density", "age:density"}

    # And the marked row is the one that describes the blocks in hand.
    row = exported.iloc[0]
    predicted = model.predict(X)
    workbook = _predict_from_payload(payload, X)
    moved_pct = np.abs(workbook - predicted) / predicted * 100.0
    assert float(row["max_abs_prediction_change_pct"]) == pytest.approx(
        float(moved_pct.max()), rel=_RECONSTRUCTION_RTOL
    )


def test_the_sheets_prediction_change_is_the_whole_workbooks_error():
    """The number on the sheet is what the reader's own workbook is off by.

    ``max_abs_prediction_change_pct`` is the only quantity on the sheet that
    reads as a promise: "apply this table and no premium moves further than
    this".  With the interaction outside the sweep it was not one -- the sheet
    reported the two main effects' binning while the workbook also carried a
    gridded interaction, so the reconstruction sat FURTHER from ``model.predict``
    than the sheet's own maximum.  Measured on this fit at ``n_bins=20``: the
    sheet claimed 29.75% and the workbook was 32.81% off, and on the mean
    4.92% against 5.51%.

    Read out of the workbook alone, exactly as ``_predict_from_payload`` does
    everywhere else in this module, so what is compared is what a consumer
    holds.
    """
    model, X, y = _grid_interaction_frame()

    for n_bins in (20, 50):
        payload = build_rating_table_payload(model, X, y, n_bins=n_bins, impact_bins=(n_bins,))
        sheet = payload.discretization_impact
        row = sheet[sheet["n_bins"] == n_bins].iloc[0]

        predicted = model.predict(X)
        workbook = _predict_from_payload(payload, X)
        moved_pct = np.abs(workbook - predicted) / predicted * 100.0

        assert float(row["max_abs_prediction_change_pct"]) == pytest.approx(
            float(moved_pct.max()), rel=_RECONSTRUCTION_RTOL
        )
        assert float(row["mean_abs_prediction_change_pct"]) == pytest.approx(
            float(moved_pct.mean()), rel=_RECONSTRUCTION_RTOL
        )


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
