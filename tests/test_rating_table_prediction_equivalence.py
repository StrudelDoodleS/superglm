"""The exported rating table has to reproduce the model it was exported from.

The workbook's contract is multiplicative: ``base_relativity`` times one
relativity per main-effect block reproduces ``model.predict`` row by row.  That
is what a filed tariff *is*, so it is pinned here directly -- the consumer below
reads only what a workbook carries (level labels, knots, log relativities, bin
interval strings, a per-unit relativity) and never consults the fitted spec.

Two claims are separated on purpose, because only one of them can be exact:

* **Exactly tabulable terms** -- ``Categorical``, ``OrderedCategorical``,
  ``Numeric``, ``Piecewise`` -- reproduce ``model.predict`` to round-off.
* **Binned terms** -- ``Spline`` and ``Polynomial`` -- are exported through the
  discretisation path, so they carry the binning error the impact sheet exists
  to report.  For them the exactness claim is centering INVARIANCE: the
  reconstructed prediction may not equal ``model.predict``, but it must not
  depend on which reporting centering was asked for.

``centering="mean"`` is swept beside the default everywhere, because that is
the mode that was wrong: ``_recenter_term`` subtracted a per-term constant that
the exported base relativity did not absorb, scaling every reconstructed
prediction by a uniform factor that no ratio-based spot check can see.
"""

from __future__ import annotations

import re

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
from superglm.export.rating_tables import build_rating_table_payload
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


_FITTED: dict[str, SuperGLM] = {}


def _fit(kind: str) -> tuple[SuperGLM, pd.DataFrame, np.ndarray, np.ndarray]:
    X, y, sample_weight = _frame()
    if kind not in _FITTED:
        features = _exactly_tabulable_features() if kind == "exact" else _every_term_type_features()
        model = SuperGLM(family="poisson", selection_penalty=0.0, features=features)
        model.fit(X, y, sample_weight=sample_weight)
        _FITTED[kind] = model
    return _FITTED[kind], X, y, sample_weight


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


def _predict_from_payload(payload, X: pd.DataFrame) -> np.ndarray:
    reconstructed = np.full(len(X), float(payload.base_relativity), dtype=np.float64)
    for block in payload.main_effects:
        reconstructed = reconstructed * _block_multiplier(block, X)
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
