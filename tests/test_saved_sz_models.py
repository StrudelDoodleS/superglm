"""sz models saved by superglm v0.35.0 load, predict and rebuild their inference (design §3.12, T8).

The fixtures in ``fixtures/saved_v0_35_0`` were written by the v0.35.0 release
(commit ea837196) with ``scripts/make_saved_sz_fixtures.py`` and
``direct_solve="structured"``: their retained linear system names the
retired range-space family (``SumToZeroBlockFactor``,
``ProfiledSumToZeroBlockFactor``, ``SumToZeroBlockStructuredSystem`` and the
factor's ``_LocalPSD`` and ``_SymmetricBorderFactor``).  Each record holds the
model, its training rows, its predictions and, for models that retained
their fit state, the standard errors and every level's curve standard error
(the implied last level's included) v0.35.0 reported.  The saved models keep
their raw B-spline coordinates: the balance tree takes their penalty's square
root as it is.
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import numpy as np
import pytest

from superglm.solvers._structured.balance_tree import SumToZeroTreeFactor

from .test_saved_fs_models import (
    _gamma,
    _se_tolerance,
    assert_predicts_as_saved,
    saved_se_scale,
)

FIXTURES = Path(__file__).parent / "fixtures" / "saved_v0_35_0"
NOTICE = "rebuilt with the current solver"


def _load(name: str) -> dict:
    with open(FIXTURES / f"{name}.pkl", "rb") as handle:
        return pickle.load(handle)


def _curve_rounding(model, curves) -> np.ndarray:
    """The bound two float64 evaluations of the term's curves are held to, per row.

    A level's curve is ``B (M beta)``: the grid basis ``B`` (``k`` columns)
    times that level's block of the term's coefficients ``M beta`` (the
    implied last level's included).  Chained dot products of ``w`` and ``k``
    terms are within ``gamma_{k+w} |B| |M| |beta|`` of the exact product
    (Higham 2002, eq. 3.5 twice), so two evaluations differ by twice that.
    ``|M| |beta|`` is formed a coefficient at a time from the map itself.
    """
    from superglm.inference.factor_smooths import _resolve_factor_smooth

    group, spec = _resolve_factor_smooth(model, "x:g:sz")
    beta = np.asarray(model.result.beta[group.sl], dtype=np.float64)
    blocks = np.zeros_like(spec._level_blocks(beta))
    for j in np.flatnonzero(beta):
        unit = np.zeros(beta.size)
        unit[j] = 1.0
        blocks += np.abs(spec._level_blocks(unit)) * abs(beta[j])
    levels = {str(level): index for index, level in enumerate(spec._levels)}
    basis = np.abs(spec.marginal_basis(curves[spec.variable].to_numpy()))
    rows = np.asarray([levels[str(level)] for level in curves["level"]])
    magnitude = np.einsum("ij,ij->i", basis, blocks[rows])
    return 2.0 * _gamma(beta.size + spec.k) * magnitude


@pytest.mark.parametrize("name", ["sz_gaussian_exact", "sz_poisson_discrete"])
def test_an_sz_model_saved_by_v0_35_0_loads_predicts_and_rebuilds_its_inference(name) -> None:
    """T8 row "retired-class shim removed": without the module ``__getattr__`` of
    ``sum_to_zero`` and ``moments`` the pickle does not load at all.

    Predictions never read the solver state: they are v0.35.0's within two
    evaluations' rounding (``assert_predicts_as_saved``).  The first
    inference call rebuilds it with the balance tree at the saved
    coefficients and smoothing parameters, once, with the notice; its
    standard errors, and every level's curve standard errors, agree with the
    ones v0.35.0 reported.
    """
    record = _load(name)
    assert record["version"] == "0.35.0"
    assert record["state_types"]["augmented_factor"] == "SumToZeroBlockFactor"
    model = record["model"]
    frame, y = record["frame"], record["y"]

    assert_predicts_as_saved(model, frame, record["prediction"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        se = model.metrics(frame, y).coefficient_se
    assert sum(NOTICE in str(item.message) for item in caught) == 1
    assert isinstance(model._linear_system_state.augmented_factor, SumToZeroTreeFactor)

    tolerance = _se_tolerance(model)
    scale = saved_se_scale(model, frame, y)
    for term, saved in record["se"].items():
        saved = scale * np.asarray(saved, dtype=float)
        rebuilt = np.asarray(se[term], dtype=float)
        np.testing.assert_array_equal(np.isfinite(rebuilt), np.isfinite(saved))
        finite = np.isfinite(saved)
        relative = np.abs(rebuilt[finite] - saved[finite]) / np.abs(saved[finite])
        assert np.all(relative <= tolerance), (term, float(np.max(relative)), tolerance)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        curves = model.factor_smooth("x:g:sz", grid=7).curves
    excess = np.abs(curves["effect"].to_numpy() - record["curve_effect"]) / _curve_rounding(
        model, curves
    )
    assert np.all(excess <= 1.0), float(np.max(excess))
    saved_curve_se = record["curve_se"]
    relative = np.abs(curves["posterior_se"].to_numpy() - saved_curve_se) / saved_curve_se
    assert np.all(relative <= tolerance), (float(np.max(relative)), tolerance)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.metrics(frame, y)
    assert not any(NOTICE in str(item.message) for item in caught)


def test_a_released_sz_model_saved_by_v0_35_0_predicts_and_asks_for_a_refit() -> None:
    """A model saved with ``retain_fit_state=False`` predicts; ``metrics``, ``summary`` and ``factor_smooth``
    each ask for a refit (``summary`` and the term reports raised an unrelated
    ``AttributeError`` before the retained state was resolved first).
    """
    record = _load("sz_gaussian_released")
    assert record["version"] == "0.35.0"
    model = record["model"]
    frame, y = record["frame"], record["y"]
    assert_predicts_as_saved(model, frame, record["prediction"])
    with pytest.raises(RuntimeError, match="refit with retain_fit_state=True"):
        model.metrics(frame, y).coefficient_se
    for call in (model.summary, lambda: model.factor_smooth("x:g:sz")):
        with pytest.raises(RuntimeError, match="refit with retain_fit_state=True"):
            call()


@pytest.mark.parametrize(
    "name", ["sz_poisson_separated", "sz_gaussian_all_thin", "sz_gaussian_one_row"]
)
def test_an_sz_model_saved_by_v0_36_0_predicts_as_saved(name) -> None:
    """A v0.36.0 model whose levels its data identify only in part predicts as it did (#444).

    v0.36.0 recorded at fit the thin levels and the levels whose line
    separates, and predicted them by convention
    (``scripts/make_saved_sz_v0_36_0_fixtures.py``).  A term penalizes its
    lines only when it selects them (``select=True``), which a model saved by
    v0.36.0 cannot: it keeps its convention.  Its conditional
    predictor on the training rows and on a grid over every level, and its
    population predictor on the grid, are v0.36.0's within two evaluations'
    rounding: ``gamma`` over the predictor's products and the convention's
    rule, times every magnitude they touch (``_eta_magnitude``, Higham 2002,
    sections 3.1 and 3.5).
    """
    from superglm.model import base

    from .test_factor_smooth_sz_thin_and_influence import _eta_magnitude

    with open(FIXTURES.parent / "saved_v0_36_0" / f"{name}.pkl", "rb") as handle:
        record = pickle.load(handle)
    assert record["version"] == "0.36.0"
    model = record["model"]
    spec = model._interaction_specs["x:g:sz"]
    assert spec._has_population_offset
    assert not spec._selects_lines
    assert [name for name, _ in spec._base_penalty_components] == ["wiggle"]
    count = len(model.result.beta) + len(spec._levels) * spec.k + 4 * spec.k + 2
    cases = (
        (record["frame"], "conditional", record["eta"]),
        (record["grid"], "conditional", record["eta_grid"]),
        (record["grid"], "population", record["eta_population"]),
    )
    for frame, effects, saved in cases:
        eta = base.predict_eta_exact(model, frame, random_effects=effects, warn=False)
        bound = 2.0 * _gamma(count) * _eta_magnitude(model, frame)
        assert np.all(np.abs(eta - saved) <= bound), (effects, float(np.max(np.abs(eta - saved))))


def _grid_over_levels(spec):
    """Every level of the term at 21 values of ``x``: the levels' own rows and everything between."""
    import pandas as pd

    levels = [str(level) for level in spec._levels]
    return pd.DataFrame(
        {
            "x": np.tile(np.linspace(0.0, 1.0, 21), len(levels)),
            "g": np.repeat(np.asarray(levels, dtype=object), 21),
        }
    )


def _drop_the_0_41_state(spec, *, record: bool) -> None:
    """The attributes a pickle written before 0.41 lacks; ``record`` also those before 0.36."""
    for name in (
        "_separated_directions",
        "_free_complement",
        "_group_left_out",
        "_group_held",
        "_folds_group",
    ):
        delattr(spec, name)
    if record:
        for name in (
            "_unidentified_levels",
            "_free_directions",
            "_weightless_levels",
            "_separated_levels",
            "_population_null_space",
        ):
            delattr(spec, name)


def test_an_sz_model_saved_before_0_41_does_not_read_its_group_term_into_the_rule() -> None:
    """A 0.36 to 0.40 model with a group term predicts its terms as they were summed (PR 508 round 1).

    0.40 accepted an ``OrderedCategorical`` on the sz group (with a warning)
    and predicted the pair as ``a_l + b(x)' identified(raw)``: the group
    term's value read nowhere in the sz term's rule.  The 0.41 rule folds it
    into each level's line, a convention of the same release as the
    lowest-order projector (``_free_complement``), and a model saved before
    keeps the rule it was saved with: unseen levels and thin levels off
    their own rows moved by the group term's mean otherwise.  A state without
    ``_free_complement`` or ``_separated_directions`` (what 0.36 to 0.40 pickle)
    moves a prediction, over every level and the x between its rows, by the
    group term's own value alone when that term's coefficients change.
    Mutation: the fold read whatever the state.
    """
    import copy
    import dataclasses

    from superglm import (
        FactorSmooth,
        LambdaPolicy,
        OrderedCategorical,
        Polynomial,
        Spline,
        SuperGLM,
    )
    from superglm._frame import as_eager_frame
    from superglm.model import base

    from .test_factor_smooth_sz_thin_and_influence import (
        _eta_magnitude,
        _gamma,
        _signed_aliased_frame,
    )

    frame, y, weight = _signed_aliased_frame("one_row", response="fisher")
    order = sorted(set(frame["g"]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the redundant group term's warning
        model = SuperGLM(
            family="gaussian",
            features={
                "x": Spline(kind="ps", n_knots=6, lambda_policy=LambdaPolicy.fixed(1.0)),
                "g": OrderedCategorical(order=order, basis=Polynomial(degree=2)),
            },
            interactions=[
                FactorSmooth(
                    "x",
                    group="g",
                    basis="sz",
                    k=6,
                    lambda_policy={"wiggle": LambdaPolicy.fixed(0.7)},
                )
            ],
            selection_penalty=0,
        )
        model.fit_reml(frame, y, sample_weight=weight)
    spec = model._interaction_specs["x:g:sz"]
    assert spec._unidentified_levels and spec._free_complement == "lowest_order"
    # the aliased group term is solved at ~0 here; give it the common level it
    # may take, as any solver's split of the redundant columns could
    beta = np.array(model.result.beta)
    ordered = next(g for g in model._groups if g.feature_name == "g")
    beta[ordered.sl] = np.random.default_rng(7).normal(size=ordered.size)
    model._result = dataclasses.replace(model.result, beta=beta)
    _drop_the_0_41_state(spec, record=False)
    loaded = pickle.loads(pickle.dumps(model))
    grid = _grid_over_levels(loaded._interaction_specs["x:g:sz"])
    without = copy.deepcopy(loaded)
    unset = np.array(loaded.result.beta)
    unset[ordered.sl] = 0.0
    without._result = dataclasses.replace(loaded.result, beta=unset)
    plan = base._prediction_plan(loaded)
    term = next(term for term in plan["features"] if term["name"] == "g")
    own = base._score_prediction_term_exact(
        term, as_eager_frame(grid), np.asarray(loaded.result.beta)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eta = base.predict_eta_exact(loaded, grid, warn=False)
        eta_without = base.predict_eta_exact(without, grid, warn=False)
    count = len(loaded.result.beta) + 30 * 6 + 4 * 6 + 2
    bound = 2.0 * _gamma(count) * _eta_magnitude(loaded, grid)
    assert np.all(np.abs((eta - eta_without) - own) <= bound)


def test_an_sz_model_saved_before_0_36_is_rebuilt_with_the_orthogonal_projector() -> None:
    """A pickle without the level record predicts, off its rows, as 0.36 to 0.40 rebuilt it (PR 508 round 1).

    Those releases rebuilt the record at the first prediction and predicted a
    thin level by the orthogonal projector (``_free_complement`` absent).  The
    0.41 rebuild recorded the lowest-order one, which moves a thin level's
    curve off its own rows.  The rebuilt model now predicts, on a grid over
    every level and for the population, bit for bit what the same fit does
    in the state those releases rebuilt.  Mutation: the rebuild's default
    complement.
    """
    from superglm.model import base

    from .test_factor_smooth_sz_thin_and_influence import (
        _fit,
        _model,
        _signed_aliased_frame,
    )

    frame, y, weight = _signed_aliased_frame("one_row", response="fisher")
    model = _fit(_model("gaussian", "auto", lam=None, numerics=("x1", "x10")), frame, y, weight)
    spec = model._interaction_specs["x:g:sz"]
    grid = _grid_over_levels(spec)
    grid = grid.assign(**{name: frame[name].iloc[0] for name in frame.columns if name not in grid})
    spec._free_complement = "orthogonal"
    expected = {
        effects: base.predict_eta_exact(model, grid, random_effects=effects, warn=False)
        for effects in ("conditional", "population")
    }
    spec._free_complement = "lowest_order"
    assert not np.array_equal(
        expected["conditional"],
        base.predict_eta_exact(model, grid, random_effects="conditional", warn=False),
    )
    _drop_the_0_41_state(spec, record=True)
    loaded = pickle.loads(pickle.dumps(model))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for effects, saved in expected.items():
            eta = base.predict_eta_exact(loaded, grid, random_effects=effects, warn=False)
            assert np.array_equal(eta, saved), effects
    assert loaded._interaction_specs["x:g:sz"]._free_complement == "orthogonal"
