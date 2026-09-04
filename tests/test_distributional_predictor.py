from __future__ import annotations

import pickle
from dataclasses import FrozenInstanceError
from typing import Any

import numpy as np
import pandas as pd
import pytest

from superglm import Constraint, SuperLSS
from superglm._frame import as_eager_frame
from superglm.distributional import GaussianLS
from superglm.distributional.family import ParameterSpec, ParameterSupport
from superglm.distributional.layout import build_stacked_layout
from superglm.distributional.prediction_design import build_joint_prediction_design
from superglm.distributional.predictor import Predictor, compile_predictors
from superglm.distributional.weights import (
    ResolvedLikelihoodWeights,
    UnsupportedLikelihoodContractError,
    WeightContract,
    WeightSemantics,
    resolve_likelihood_weights,
)
from superglm.features import (
    Categorical,
    FactorSmooth,
    Numeric,
    OrderedCategorical,
    Spline,
    SplineCategorical,
)
from superglm.features.interaction import (
    PolynomialCategorical,
    PolynomialInteraction,
    TensorInteraction,
)
from superglm.features.polynomial import Polynomial
from superglm.features.spline import PSpline
from superglm.group_matrix import SupportCompressedSSPGroupMatrix
from superglm.links import IdentityLink, LogLink


def _parameter(name: str, default_link: str = "identity") -> ParameterSpec:
    return ParameterSpec(
        name=name,
        default_link=default_link,
        role=name,
        support=ParameterSupport(),
    )


def _frame(n: int = 18):
    return as_eager_frame(
        pd.DataFrame(
            {
                "x": np.linspace(-1.0, 1.0, n),
                "z": np.linspace(0.0, 2.0, n),
            }
        )
    )


def _resolved(
    values: np.ndarray,
    *,
    semantics: WeightSemantics,
) -> ResolvedLikelihoodWeights:
    return resolve_likelihood_weights(
        values,
        n_observations=len(values),
        contract=WeightContract(semantics=semantics),
    )


@pytest.mark.parametrize(
    ("predictors", "message"),
    [
        ((Predictor("location", {"x": Numeric()}),), "missing.*scale"),
        (
            (
                Predictor("location", {"x": Numeric()}),
                Predictor("location", {"z": Numeric()}),
            ),
            "duplicate.*location",
        ),
        (
            (
                Predictor("location", {"x": Numeric()}),
                Predictor("shape", {"z": Numeric()}),
            ),
            "unknown.*shape",
        ),
        (
            (
                Predictor("scale", {"z": Numeric()}),
                Predictor("location", {"x": Numeric()}),
            ),
            "order.*location.*scale",
        ),
    ],
)
def test_predictors_must_match_family_names_and_order(
    predictors: tuple[Predictor, ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        compile_predictors(
            _frame(),
            _resolved(np.ones(18), semantics="prior"),
            (_parameter("location"), _parameter("scale", "log")),
            predictors,
        )


def test_predictor_configuration_and_offsets_are_immutable_owned_state() -> None:
    location_offset = np.linspace(-0.2, 0.3, 18)
    predictors = (
        Predictor("location", {"x": Numeric()}, intercept=True),
        Predictor("scale", {"z": Numeric()}, link=LogLink(), intercept=False),
    )

    compiled = compile_predictors(
        _frame(),
        _resolved(np.ones(18), semantics="prior"),
        (_parameter("location"), _parameter("scale", "log")),
        predictors,
        offsets={"location": location_offset},
    )
    location_offset[:] = 99.0

    assert isinstance(compiled[0].link, IdentityLink)
    assert isinstance(compiled[1].link, LogLink)
    assert compiled[0].intercept is True
    assert compiled[1].intercept is False
    np.testing.assert_allclose(compiled[0].offset, np.linspace(-0.2, 0.3, 18))
    np.testing.assert_array_equal(compiled[1].offset, np.zeros(18))
    assert not compiled[0].offset.flags.writeable

    with pytest.raises(FrozenInstanceError):
        predictors[0].name = "other"  # ty: ignore[invalid-assignment]
    with pytest.raises(TypeError):
        predictors[0].features["z"] = Numeric()  # ty: ignore[invalid-assignment]


def test_offsets_reject_unknown_names_bad_shapes_and_nonfinite_values() -> None:
    predictors = (
        Predictor("location", {}),
        Predictor("scale", {}),
    )
    parameters = (_parameter("location"), _parameter("scale", "log"))

    with pytest.raises(ValueError, match="unknown offset.*shape"):
        compile_predictors(
            _frame(),
            _resolved(np.ones(18), semantics="prior"),
            parameters,
            predictors,
            offsets={"shape": np.zeros(18)},
        )
    with pytest.raises(ValueError, match="offset.*location.*length 18"):
        compile_predictors(
            _frame(),
            _resolved(np.ones(18), semantics="prior"),
            parameters,
            predictors,
            offsets={"location": np.zeros(17)},
        )
    with pytest.raises(ValueError, match="offset.*scale.*finite"):
        bad = np.zeros(18)
        bad[4] = np.nan
        compile_predictors(
            _frame(),
            _resolved(np.ones(18), semantics="prior"),
            parameters,
            predictors,
            offsets={"scale": bad},
        )


def test_reused_feature_objects_compile_to_independent_predictor_state() -> None:
    shared = Spline(n_knots=5, degree=2, penalty="ssp", select=True)
    caller_before = pickle.dumps(shared)
    predictors = (
        Predictor("location", {"x": shared}),
        Predictor("scale", {"x": shared}, link="log"),
    )

    compiled = compile_predictors(
        _frame(),
        _resolved(np.linspace(0.7, 1.4, 18), semantics="prior"),
        (_parameter("location"), _parameter("scale", "log")),
        predictors,
    )

    location_spec = compiled[0].compiled.specs["x"]
    scale_spec = compiled[1].compiled.specs["x"]
    assert location_spec is not shared
    assert scale_spec is not shared
    assert location_spec is not scale_spec
    assert pickle.dumps(shared) == caller_before


def test_sample_weights_must_match_predictor_rows() -> None:
    with pytest.raises(ValueError, match="resolved likelihood weights.*18"):
        compile_predictors(
            _frame(),
            _resolved(np.ones(17), semantics="prior"),
            (_parameter("location"),),
            (Predictor("location", {"x": Numeric()}),),
        )


def test_predictor_compilation_refuses_an_unresolved_weight_array() -> None:
    with pytest.raises(TypeError, match="ResolvedLikelihoodWeights"):
        compile_predictors(
            _frame(),
            np.ones(18),  # type: ignore[arg-type]
            (_parameter("location"),),
            (Predictor("location", {"x": Numeric()}),),
        )


def _select_fixture(values: np.ndarray, n: int) -> tuple[pd.DataFrame, np.ndarray]:
    """One ordered driver plus a plain scale covariate, and a response on both."""
    rng = np.random.default_rng(20260818)
    codes = pd.Categorical(values).codes.astype(np.float64)
    z = rng.normal(size=n)
    y = 0.4 + 0.35 * codes + rng.normal(scale=0.3, size=n)
    return pd.DataFrame({"g": values, "z": z}), y


def _fit_select_location(feature, frame: pd.DataFrame, y: np.ndarray) -> SuperLSS:
    return SuperLSS(
        family=GaussianLS(scale_floor=0.01),
        predictors=(
            Predictor("location", {"g": feature}),
            Predictor("scale", {"z": Numeric()}),
        ),
    ).fit(frame, y, lambdas={"location:g#null": 1.0, "location:g#wiggle": 1.0})


def test_hosted_select_smooth_predicts_the_surface_it_was_fitted_on() -> None:
    """``select=True`` inside an ``OrderedCategorical`` must centre at both ends.

    Fit time centres the hosted basis and stamps the constant it removed; the
    predict-time half subtracts it inside the spline runtime.  The constant has
    to reach the spec that actually evaluates the basis -- the inner spline, not
    the wrapper, which delegates every numeric path to it -- or prediction
    silently returns the uncentred surface, shifted by the whole weighted mean
    level of the smooth.
    """
    n = 400
    rng = np.random.default_rng(7)
    levels = [f"L{index}" for index in range(8)]
    labels = np.asarray([levels[i] for i in rng.integers(0, len(levels), n)], dtype=object)
    frame, y = _select_fixture(labels, n)

    model = _fit_select_location(
        OrderedCategorical(
            order=levels,
            basis=Spline(kind="ps", n_knots=4, degree=3, penalty="ssp", select=True),
        ),
        frame,
        y,
    )

    coefficients = model.coef_by_predictor_["location"]
    eta = model.predict_link(frame)["location"].to_numpy()
    # The centred term contributes zero mean over the fitted rows, so the mean
    # linear predictor at those rows is exactly the fitted intercept.
    assert eta.mean() == pytest.approx(float(coefficients[0]), abs=1e-10)


def test_support_compressed_select_smooth_centres_instead_of_refusing() -> None:
    """A heaped numeric axis compresses to a lossless support block.

    ``SupportCompressedSSPGroupMatrix`` subclasses the binned container, so an
    exact-type test against the parent refuses it outright -- an ordinary
    ``select=True`` spline over repeated values could not be fitted at all.
    """
    n = 5000
    rng = np.random.default_rng(11)
    frame, y = _select_fixture(rng.integers(0, 20, n).astype(np.float64), n)

    model = _fit_select_location(
        Spline(kind="ps", n_knots=5, degree=3, penalty="ssp", select=True), frame, y
    )

    fitted = model._model
    assert fitted is not None
    matrix = fitted.compiled_predictors[0].compiled.design.group_matrices[0]
    assert isinstance(matrix, SupportCompressedSSPGroupMatrix)

    coefficients = model.coef_by_predictor_["location"]
    eta = model.predict_link(frame)["location"].to_numpy()
    assert eta.mean() == pytest.approx(float(coefficients[0]), abs=1e-10)


def test_select_centering_refuses_a_hosted_basis_with_an_active_special() -> None:
    """An active special's rows are zero in the smooth block and must not be shifted.

    Named for the branch it covers.  This shape compiles to two groups, so the
    group-count check refuses it too -- on a message that never says the word
    ``specials``.  Matching on that word is therefore what makes this a test of
    the specials refusal rather than of the group-count one.
    """
    n = 300
    rng = np.random.default_rng(5)
    levels = [f"L{index}" for index in range(6)]
    labels = np.asarray([levels[i] for i in rng.integers(0, len(levels), n)], dtype=object)
    labels[:40] = "UNKNOWN"
    frame, y = _select_fixture(labels, n)

    feature = OrderedCategorical(
        order=levels,
        specials=["UNKNOWN"],
        basis=Spline(kind="ps", n_knots=3, degree=3, penalty="ssp", select=True),
    )
    with pytest.raises(ValueError, match="specials"):
        _fit_select_location(feature, frame, y)


def test_select_centering_accepts_a_hosted_basis_whose_specials_are_all_pinned() -> None:
    """A special absent from the training column is pinned, and must still fit.

    The other branch of the same guard, and the one no other check covers: with
    every declared special pinned the term emits no indicator block and
    compiles to a single group, so the group-count refusal never fires.

    Pinned is the only shape a special can take here without rows -- ``fit``
    requires strictly positive weights, so a level cannot be pinned by carrying
    zero-weight rows.  With no special rows in the design there is no
    structural zero for centring to shift, and the fitted block is the one the
    same spec without ``specials=`` builds: this asserts that equality
    coefficient for coefficient rather than merely that nothing raised.  The
    level stays known, and scores with no contribution from the term, which
    after centring is the weighted mean level of the smooth.
    """
    n = 300
    rng = np.random.default_rng(5)
    levels = [f"L{index}" for index in range(6)]
    labels = np.asarray([levels[i] for i in rng.integers(0, len(levels), n)], dtype=object)
    frame, y = _select_fixture(labels, n)

    def basis() -> Any:
        return Spline(kind="ps", n_knots=3, degree=3, penalty="ssp", select=True)

    with pytest.warns(UserWarning, match="no effective training rows"):
        pinned = _fit_select_location(
            OrderedCategorical(
                order=[*levels, "UNSEEN"],
                specials=["UNSEEN"],
                basis=basis(),
            ),
            frame,
            y,
        )
    plain = _fit_select_location(OrderedCategorical(order=levels, basis=basis()), frame, y)

    fitted = pinned._model
    assert fitted is not None
    groups = fitted.compiled_predictors[0].compiled.groups
    assert [group.feature_name for group in groups] == ["g"]

    pinned_coefficients = np.asarray(pinned.coef_by_predictor_["location"], dtype=float)
    plain_coefficients = np.asarray(plain.coef_by_predictor_["location"], dtype=float)
    assert pinned_coefficients == pytest.approx(plain_coefficients, abs=1e-12)

    eta = pinned.predict_link(frame)["location"].to_numpy()
    assert eta == pytest.approx(plain.predict_link(frame)["location"].to_numpy(), abs=1e-12)
    assert eta.mean() == pytest.approx(float(pinned_coefficients[0]), abs=1e-10)

    with_pinned_level = frame.copy()
    with_pinned_level.loc[with_pinned_level.index[:5], "g"] = "UNSEEN"
    scored = pinned.predict_link(with_pinned_level)["location"].to_numpy()[:5]
    assert scored == pytest.approx(float(pinned_coefficients[0]), abs=1e-10)


def _weighted_knot_fixture() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """A frame whose weight mass sits entirely on one half of the x range."""
    rng = np.random.default_rng(3)
    n = 800
    x = rng.uniform(0.0, 1.0, n)
    z = rng.uniform(0.0, 1.0, n)
    y = rng.normal(2.0 + 1.5 * np.sin(2.0 * np.pi * x), 0.25 + 0.7 * z**2)
    return pd.DataFrame({"x": x, "z": z}), y, np.where(x > 0.5, 20.0, 1.0)


def _knots_under(weight_semantics: str, sample_weight: np.ndarray) -> np.ndarray:
    """Return the location smooth's knots under one declared weight contract."""
    frame, _, _ = _weighted_knot_fixture()
    compiled = compile_predictors(
        as_eager_frame(frame),
        _resolved(sample_weight, semantics=weight_semantics),
        GaussianLS().parameters,
        (
            Predictor(
                name="location",
                features={"x": Spline("cr", k=8, knot_strategy="quantile_rows")},
            ),
            Predictor(name="scale", features={"z": Spline("cr", k=6)}),
        ),
    )
    return np.asarray(compiled[0].compiled.specs["x"]._knots, dtype=np.float64)


def test_the_declared_weight_contract_decides_learned_knot_geometry() -> None:
    """Prior weights place knots on physical rows; frequency weights on mass.

    A strategy that reads the weight is required to see this at all: plain
    ``quantile`` takes unweighted percentiles of the retained rows, and
    ``uniform`` -- the default -- ignores the weight by construction, so under
    either of those the two contracts agree and this parameter looks inert
    whether or not it is wired up.  ``quantile_rows`` is one that reads it.
    """
    _, _, weights = _weighted_knot_fixture()
    ones = np.ones_like(weights)

    prior = _knots_under("prior", weights)
    frequency = _knots_under("frequency", weights)

    # Live: the declaration alone moves the geometry, on the same weights.
    assert not np.allclose(prior, frequency)

    # And it moves it to the right place. Prior weights say how precisely each
    # row was measured, not how many rows there are, so the geometry must be
    # the one unweighted rows would have produced -- not merely "different".
    np.testing.assert_allclose(prior, _knots_under("frequency", ones), rtol=0, atol=0)
    assert not np.allclose(frequency, _knots_under("frequency", ones))


def _selected_smooth_compilation(
    frame: pd.DataFrame,
    sample_weight: np.ndarray,
    *,
    semantics: str,
    discrete: bool,
):
    resolved = _resolved(sample_weight, semantics=semantics)
    compiled = compile_predictors(
        as_eager_frame(frame),
        resolved,
        GaussianLS().parameters,
        (
            Predictor(
                "location",
                {
                    "x": Spline(
                        kind="ps",
                        n_knots=5,
                        knot_strategy="quantile_rows",
                        select=True,
                        discrete=discrete,
                    )
                },
            ),
            Predictor("scale", {}),
        ),
        model_discrete=discrete,
        n_bins_config=6,
    )
    return resolved, compiled[0]


@pytest.mark.parametrize("discrete", [False, True], ids=["dense", "discrete"])
def test_prior_selected_smooth_geometry_and_centering_follow_physical_rows(discrete: bool):
    x = np.array([-3.0, -2.2, -1.8, -1.0, -0.4, 0.1, 0.5, 1.2, 2.0, 3.5])
    frame = pd.DataFrame({"x": x})
    unequal = np.array([0.1, 12.0, 0.2, 8.0, 0.4, 6.0, 0.3, 4.0, 0.5, 2.0])

    resolved, weighted = _selected_smooth_compilation(
        frame,
        unequal,
        semantics="prior",
        discrete=discrete,
    )
    _, physical = _selected_smooth_compilation(
        frame,
        np.ones(len(frame)),
        semantics="prior",
        discrete=discrete,
    )

    weighted_spec = weighted.compiled.specs["x"]
    physical_spec = physical.compiled.specs["x"]
    np.testing.assert_allclose(weighted_spec._knots, physical_spec._knots, rtol=0.0, atol=0.0)
    weighted_matrix = weighted.compiled.design.group_matrices[0]
    np.testing.assert_allclose(
        weighted_matrix.rmatvec(resolved.geometry_values),
        0.0,
        rtol=0.0,
        atol=2.0e-12,
    )
    if discrete:
        physical_matrix = physical.compiled.design.group_matrices[0]
        np.testing.assert_array_equal(weighted_matrix.bin_idx, physical_matrix.bin_idx)


@pytest.mark.parametrize("discrete", [False, True], ids=["dense", "discrete"])
def test_frequency_selected_smooth_geometry_and_centering_match_literal_expansion(
    discrete: bool,
):
    x = np.array([-3.0, -2.2, -1.8, -1.0, -0.4, 0.1, 0.5, 1.2, 2.0, 3.5])
    counts = np.array([1, 4, 2, 6, 1, 3, 5, 1, 4, 2], dtype=np.float64)
    repeated = np.repeat(np.arange(len(x)), counts.astype(np.intp))

    resolved, compact = _selected_smooth_compilation(
        pd.DataFrame({"x": x}),
        counts,
        semantics="frequency",
        discrete=discrete,
    )
    expanded_resolved, expanded = _selected_smooth_compilation(
        pd.DataFrame({"x": x[repeated]}),
        np.ones(len(repeated)),
        semantics="prior",
        discrete=discrete,
    )

    compact_spec = compact.compiled.specs["x"]
    expanded_spec = expanded.compiled.specs["x"]
    np.testing.assert_allclose(compact_spec._knots, expanded_spec._knots, rtol=0.0, atol=0.0)
    compact_matrix = compact.compiled.design.group_matrices[0]
    expanded_matrix = expanded.compiled.design.group_matrices[0]
    np.testing.assert_allclose(
        compact_matrix.rmatvec(resolved.geometry_values),
        0.0,
        rtol=0.0,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        expanded_matrix.rmatvec(expanded_resolved.geometry_values),
        0.0,
        rtol=0.0,
        atol=2.0e-12,
    )
    if discrete:
        np.testing.assert_allclose(
            compact_matrix.B_unique,
            expanded_matrix.B_unique,
            rtol=0.0,
            atol=2.0e-12,
        )


def _ordinary_geometry_compilation(
    frame: pd.DataFrame,
    sample_weight: np.ndarray,
    *,
    semantics: WeightSemantics,
    discrete: bool,
):
    resolved = _resolved(sample_weight, semantics=semantics)
    compiled = compile_predictors(
        as_eager_frame(frame),
        resolved,
        GaussianLS().parameters,
        (
            Predictor(
                "location",
                {
                    "x": Spline(
                        kind="ps",
                        n_knots=4,
                        knot_strategy="quantile_rows",
                        discrete=discrete,
                    )
                },
            ),
            Predictor("scale", {}),
        ),
        model_discrete=discrete,
        n_bins_config=20,
    )
    return resolved, compiled


def _tensor_geometry_compilation(
    frame: pd.DataFrame,
    sample_weight: np.ndarray,
    *,
    semantics: WeightSemantics,
    discrete: bool,
):
    resolved = _resolved(sample_weight, semantics=semantics)
    compiled = compile_predictors(
        as_eager_frame(frame),
        resolved,
        GaussianLS().parameters,
        (
            Predictor(
                "location",
                {
                    "x1": Spline(
                        kind="ps",
                        n_knots=5,
                        knot_strategy="quantile_rows",
                        discrete=discrete,
                    ),
                    "x2": Spline(
                        kind="ps",
                        n_knots=5,
                        knot_strategy="quantile_rows",
                        discrete=discrete,
                    ),
                },
                interaction_specs={"x1:x2": TensorInteraction("x1", "x2", n_knots=(4, 4))},
            ),
            Predictor("scale", {}),
        ),
        model_discrete=discrete,
        n_bins_config=20,
    )
    return resolved, compiled


class _LegacyTensorPSpline(PSpline):
    marginal_calls = 0

    def tensor_marginal_ingredients(self, x: np.ndarray):
        type(self).marginal_calls += 1
        return super().tensor_marginal_ingredients(x)


class _WeightAwareTensorPSpline(PSpline):
    marginal_calls = 0

    def tensor_marginal_ingredients(
        self,
        x: np.ndarray,
        *,
        support: np.ndarray | None = None,
        counts: np.ndarray | None = None,
    ):
        type(self).marginal_calls += 1
        return super().tensor_marginal_ingredients(
            x,
            support=support,
            counts=counts,
        )


def _custom_tensor_geometry_compilation(
    spline_type: type[PSpline],
    sample_weight: np.ndarray,
    *,
    semantics: WeightSemantics,
    discrete: bool,
):
    frame, _ = _geometry_fixture()
    resolved = _resolved(sample_weight, semantics=semantics)
    return compile_predictors(
        as_eager_frame(frame),
        resolved,
        GaussianLS().parameters,
        (
            Predictor(
                "location",
                {
                    "x1": spline_type(
                        n_knots=5,
                        knot_strategy="quantile_rows",
                        discrete=discrete,
                    ),
                    "x2": PSpline(
                        n_knots=5,
                        knot_strategy="quantile_rows",
                        discrete=discrete,
                    ),
                },
                interaction_specs={"x1:x2": TensorInteraction("x1", "x2")},
            ),
            Predictor("scale", {}),
        ),
        model_discrete=discrete,
        n_bins_config=20,
    )


@pytest.mark.parametrize("discrete", [False, True])
def test_legacy_tensor_marginal_accepts_unit_physical_geometry(discrete: bool) -> None:
    frame, _ = _geometry_fixture()
    _LegacyTensorPSpline.marginal_calls = 0

    compiled = _custom_tensor_geometry_compilation(
        _LegacyTensorPSpline,
        np.geomspace(0.2, 8.0, len(frame)),
        semantics="prior",
        discrete=discrete,
    )

    assert any(group.feature_name == "x1:x2" for group in compiled[0].compiled.groups)
    assert _LegacyTensorPSpline.marginal_calls == 1


@pytest.mark.parametrize("discrete", [False, True])
def test_legacy_tensor_marginal_refuses_nonunit_frequency_geometry(discrete: bool) -> None:
    _, counts = _geometry_fixture()
    _LegacyTensorPSpline.marginal_calls = 0

    with pytest.raises(NotImplementedError, match="cannot certify non-unit geometry"):
        _custom_tensor_geometry_compilation(
            _LegacyTensorPSpline,
            counts,
            semantics="frequency",
            discrete=discrete,
        )

    assert _LegacyTensorPSpline.marginal_calls == 0


@pytest.mark.parametrize("discrete", [False, True])
def test_weight_aware_tensor_marginal_accepts_frequency_geometry(discrete: bool) -> None:
    _, counts = _geometry_fixture()
    _WeightAwareTensorPSpline.marginal_calls = 0

    compiled = _custom_tensor_geometry_compilation(
        _WeightAwareTensorPSpline,
        counts,
        semantics="frequency",
        discrete=discrete,
    )

    assert any(group.feature_name == "x1:x2" for group in compiled[0].compiled.groups)
    assert _WeightAwareTensorPSpline.marginal_calls == 1


def _geometry_fixture() -> tuple[pd.DataFrame, np.ndarray]:
    return (
        pd.DataFrame(
            {
                "x": [-3.0, -2.2, -1.8, -1.0, -0.4, 0.1, 0.5, 1.2, 2.0, 3.5],
                "x1": [-3.0, -2.2, -1.8, -1.0, -0.4, 0.1, 0.5, 1.2, 2.0, 3.5],
                "x2": [0.4, -1.1, 1.3, -0.7, 0.2, 1.8, -1.5, 0.9, -0.1, 2.4],
            }
        ),
        np.array([1, 4, 2, 6, 1, 3, 5, 1, 4, 2], dtype=np.float64),
    )


def _projection_projector(projection: np.ndarray) -> np.ndarray:
    values = np.asarray(projection, dtype=np.float64)
    return values @ values.T


def _roundoff_bound(values: np.ndarray, *, mass: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    dimension = max(array.shape, default=1)
    return float(
        512.0
        * np.finfo(np.float64).eps
        * dimension
        * max(1.0, mass)
        * max(1.0, np.linalg.norm(array, ord=np.inf))
    )


def _prediction_matrix(frame: pd.DataFrame, compiled) -> np.ndarray:
    layout = build_stacked_layout(compiled)
    return build_joint_prediction_design(frame, compiled, layout).local["location"]


_ORDERED_LEVELS = tuple(f"L{index}" for index in range(10))


def _ordered_geometry_fixture() -> tuple[pd.DataFrame, np.ndarray]:
    return (
        pd.DataFrame({"band": _ORDERED_LEVELS}),
        np.array([1, 2, 1, 2, 1, 2, 1, 2, 1, 2], dtype=np.float64),
    )


def _ordered_geometry_feature(route: str, *, model_discrete: bool) -> OrderedCategorical:
    common = dict(
        n_knots=6,
        knot_strategy="quantile_rows",
        discrete=model_discrete,
    )
    if route in {"ordinary", "special"}:
        basis = Spline(kind="cr", **common)
    elif route == "select":
        basis = Spline(kind="ps", select=True, **common)
    elif route == "scop":
        basis = Spline(
            kind="ps",
            constraint=Constraint.fit.increasing,
            **common,
        )
    else:  # pragma: no cover - test parameter owns the route set
        raise AssertionError(f"unknown ordered geometry route {route!r}")
    return OrderedCategorical(
        order=_ORDERED_LEVELS,
        basis=basis,
        specials=["L5"] if route == "special" else None,
    )


def _ordered_geometry_compilation(
    frame: pd.DataFrame,
    sample_weight: np.ndarray,
    *,
    semantics: WeightSemantics,
    route: str,
    model_discrete: bool,
):
    resolved = _resolved(sample_weight, semantics=semantics)
    compiled = compile_predictors(
        as_eager_frame(frame),
        resolved,
        GaussianLS().parameters,
        (
            Predictor(
                "location",
                {"band": _ordered_geometry_feature(route, model_discrete=model_discrete)},
            ),
            Predictor("scale", {}),
        ),
        model_discrete=model_discrete,
        n_bins_config=20,
    )
    return resolved, compiled


def _term_geometry_invariants(
    compiled,
    scoring_frame: pd.DataFrame,
    fit_frame: pd.DataFrame,
    geometry_weight: np.ndarray,
    *,
    term_name: str,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, float]:
    layout = build_stacked_layout(compiled)
    predictor = compiled[0]
    global_slices = tuple(
        layout.term_slices[f"location:{group.name}"]
        for group in predictor.compiled.groups
        if group.feature_name == term_name
    )
    global_indices = np.concatenate(
        tuple(np.arange(group_slice.start, group_slice.stop) for group_slice in global_slices)
    )
    predictor_state = layout.predictor("location")
    local_slices = tuple(
        slice(
            group_slice.start - predictor_state.coefficient_slice.start,
            group_slice.stop - predictor_state.coefficient_slice.start,
        )
        for group_slice in global_slices
    )
    local_indices = np.concatenate(
        tuple(np.arange(local_slice.start, local_slice.stop) for local_slice in local_slices)
    )
    scoring_design = build_joint_prediction_design(
        scoring_frame,
        compiled,
        layout,
    ).local["location"][:, local_indices]
    fit_design = build_joint_prediction_design(
        fit_frame,
        compiled,
        layout,
    ).local["location"][:, local_indices]

    left, singular_values, _ = np.linalg.svd(scoring_design, full_matrices=False)
    rank_tolerance = (
        np.finfo(np.float64).eps * max(scoring_design.shape) * max(1.0, float(singular_values[0]))
    )
    rank = int(np.sum(singular_values > rank_tolerance))
    projector = left[:, :rank] @ left[:, :rank].T
    target = np.sin(np.linspace(-1.2, 1.4, len(scoring_frame)))
    fitted_function = projector @ target

    lambdas = {name: 1.0 for name in layout.penalty_names}
    penalty = layout.penalty_matrix(lambdas)[np.ix_(global_indices, global_indices)]
    inverse_design = np.linalg.pinv(
        scoring_design,
        rcond=rank_tolerance / singular_values[0],
    )
    function_penalty = inverse_design.T @ penalty @ inverse_design
    gram = fit_design.T @ (fit_design * geometry_weight[:, None])
    edf = float(np.trace(np.linalg.pinv(gram + penalty) @ gram))
    return rank, projector, fitted_function, function_penalty, edf


_ORDERED_GEOMETRY_ROUTES = (
    pytest.param("ordinary", False, id="ordinary-dense"),
    pytest.param("ordinary", True, id="ordinary-model-discrete"),
    pytest.param("special", False, id="masked-special"),
    pytest.param("select", False, id="select-dense"),
    pytest.param("scop", False, id="scop-dense"),
    pytest.param("scop", True, id="scop-model-discrete"),
)


@pytest.mark.parametrize(("route", "model_discrete"), _ORDERED_GEOMETRY_ROUTES)
def test_frequency_ordered_inner_geometry_matches_literal_expansion(
    route: str,
    model_discrete: bool,
) -> None:
    frame, counts = _ordered_geometry_fixture()
    repeated = np.repeat(np.arange(len(frame)), counts.astype(np.intp))
    compact_weights, compact = _ordered_geometry_compilation(
        frame,
        counts,
        semantics="frequency",
        route=route,
        model_discrete=model_discrete,
    )
    expanded_frame = frame.iloc[repeated].reset_index(drop=True)
    expanded_weights, expanded = _ordered_geometry_compilation(
        expanded_frame,
        np.ones(len(repeated)),
        semantics="prior",
        route=route,
        model_discrete=model_discrete,
    )

    compact_inner = compact[0].compiled.specs["band"]._basis_spline
    expanded_inner = expanded[0].compiled.specs["band"]._basis_spline
    np.testing.assert_allclose(compact_inner._knots, expanded_inner._knots, rtol=0.0, atol=0.0)
    compact_invariants = _term_geometry_invariants(
        compact,
        frame,
        frame,
        compact_weights.geometry_values,
        term_name="band",
    )
    expanded_invariants = _term_geometry_invariants(
        expanded,
        frame,
        expanded_frame,
        expanded_weights.geometry_values,
        term_name="band",
    )
    assert compact_invariants[0] == expanded_invariants[0]
    for compact_value, expanded_value in zip(
        compact_invariants[1:],
        expanded_invariants[1:],
        strict=True,
    ):
        np.testing.assert_allclose(
            compact_value,
            expanded_value,
            rtol=0.0,
            atol=_roundoff_bound(np.atleast_1d(expanded_value), mass=float(len(repeated))),
        )


@pytest.mark.parametrize(("route", "model_discrete"), _ORDERED_GEOMETRY_ROUTES)
def test_prior_ordered_inner_geometry_follows_physical_rows(
    route: str,
    model_discrete: bool,
) -> None:
    frame, _ = _ordered_geometry_fixture()
    unequal = np.geomspace(0.1, 12.0, len(frame))
    prior_weights, weighted = _ordered_geometry_compilation(
        frame,
        unequal,
        semantics="prior",
        route=route,
        model_discrete=model_discrete,
    )
    unit_weights, physical = _ordered_geometry_compilation(
        frame,
        np.ones(len(frame)),
        semantics="prior",
        route=route,
        model_discrete=model_discrete,
    )

    weighted_inner = weighted[0].compiled.specs["band"]._basis_spline
    physical_inner = physical[0].compiled.specs["band"]._basis_spline
    np.testing.assert_allclose(weighted_inner._knots, physical_inner._knots, rtol=0.0, atol=0.0)
    weighted_invariants = _term_geometry_invariants(
        weighted,
        frame,
        frame,
        prior_weights.geometry_values,
        term_name="band",
    )
    physical_invariants = _term_geometry_invariants(
        physical,
        frame,
        frame,
        unit_weights.geometry_values,
        term_name="band",
    )
    assert weighted_invariants[0] == physical_invariants[0]
    for weighted_value, physical_value in zip(
        weighted_invariants[1:],
        physical_invariants[1:],
        strict=True,
    ):
        np.testing.assert_allclose(
            weighted_value,
            physical_value,
            rtol=0.0,
            atol=_roundoff_bound(np.atleast_1d(physical_value), mass=float(len(frame))),
        )


def _polynomial_geometry_fixture() -> tuple[pd.DataFrame, np.ndarray]:
    return (
        pd.DataFrame(
            {
                "x": np.array([-1.5, -1.0, -0.7, -0.2, 0.1, 0.4, 0.8, 1.1, 1.4, 1.9]),
                "x2": np.array([0.3, -1.2, 1.4, -0.5, 0.8, 1.7, -1.0, 0.1, 1.1, -0.2]),
                "g": np.array(["a", "b", "c", "b", "c", "a", "b", "c", "a", "b"]),
                "band": _ORDERED_LEVELS,
            }
        ),
        np.array([1, 2, 1, 2, 1, 2, 1, 2, 1, 2], dtype=np.float64),
    )


def _polynomial_geometry_compilation(
    frame: pd.DataFrame,
    sample_weight: np.ndarray,
    *,
    semantics: WeightSemantics,
    route: str,
):
    if route == "standalone":
        features = {"x": Polynomial(powers=[1, 2, 3])}
        interactions = {}
    elif route == "categorical":
        features = {
            "x": Polynomial(powers=[1, 2, 3]),
            "g": Categorical(base="a"),
        }
        interactions = {"x:g": PolynomialCategorical("x", "g")}
    elif route == "interaction":
        features = {
            "x": Polynomial(powers=[1, 2, 3]),
            "x2": Polynomial(powers=[1, 2]),
        }
        interactions = {"x:x2": PolynomialInteraction("x", "x2")}
    elif route == "ordered-hosted":
        features = {
            "band": OrderedCategorical(
                order=_ORDERED_LEVELS,
                basis=Polynomial(powers=[1, 2, 3]),
            )
        }
        interactions = {}
    else:  # pragma: no cover - test parameter owns the route set
        raise AssertionError(f"unknown polynomial geometry route {route!r}")

    resolved = _resolved(sample_weight, semantics=semantics)
    compiled = compile_predictors(
        as_eager_frame(frame),
        resolved,
        GaussianLS().parameters,
        (
            Predictor("location", features, interaction_specs=interactions),
            Predictor("scale", {}),
        ),
    )
    return resolved, compiled


def _polynomial_term_name(route: str) -> str:
    return {
        "standalone": "x",
        "categorical": "x:g",
        "interaction": "x:x2",
        "ordered-hosted": "band",
    }[route]


def _standalone_polynomial_observables(
    compiled,
    scoring_frame: pd.DataFrame,
    fit_frame: pd.DataFrame,
    geometry_weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    spec = compiled[0].compiled.specs["x"]
    fit_basis = spec.transform(fit_frame["x"].to_numpy())
    moment = geometry_weight @ fit_basis
    scoring_basis = spec.transform(scoring_frame["x"].to_numpy())
    target = scoring_frame["x"].to_numpy() ** 3 - 0.3 * scoring_frame["x"].to_numpy()
    coefficients = np.linalg.lstsq(scoring_basis, target, rcond=None)[0]
    reconstruction = np.asarray(
        spec.reconstruct(coefficients, n_points=41)["log_relativity"],
        dtype=np.float64,
    )
    return moment, reconstruction


_POLYNOMIAL_GEOMETRY_ROUTES = ("standalone", "categorical", "interaction", "ordered-hosted")


def test_prior_standalone_polynomial_centers_on_physical_rows() -> None:
    frame, _ = _polynomial_geometry_fixture()
    resolved, compiled = _polynomial_geometry_compilation(
        frame,
        np.geomspace(0.1, 20.0, len(frame)),
        semantics="prior",
        route="standalone",
    )
    spec = compiled[0].compiled.specs["x"]
    moment = resolved.geometry_values @ spec.transform(frame["x"].to_numpy())

    np.testing.assert_allclose(
        moment,
        0.0,
        rtol=0.0,
        atol=_roundoff_bound(spec.transform(frame["x"].to_numpy()), mass=float(len(frame))),
    )


@pytest.mark.parametrize("route", _POLYNOMIAL_GEOMETRY_ROUTES)
def test_frequency_polynomial_geometry_matches_literal_expansion(route: str) -> None:
    frame, counts = _polynomial_geometry_fixture()
    repeated = np.repeat(np.arange(len(frame)), counts.astype(np.intp))
    compact_weights, compact = _polynomial_geometry_compilation(
        frame,
        counts,
        semantics="frequency",
        route=route,
    )
    expanded_frame = frame.iloc[repeated].reset_index(drop=True)
    expanded_weights, expanded = _polynomial_geometry_compilation(
        expanded_frame,
        np.ones(len(repeated)),
        semantics="prior",
        route=route,
    )

    compact_invariants = _term_geometry_invariants(
        compact,
        frame,
        frame,
        compact_weights.geometry_values,
        term_name=_polynomial_term_name(route),
    )
    expanded_invariants = _term_geometry_invariants(
        expanded,
        frame,
        expanded_frame,
        expanded_weights.geometry_values,
        term_name=_polynomial_term_name(route),
    )
    assert compact_invariants[0] == expanded_invariants[0]
    for compact_value, expanded_value in zip(
        compact_invariants[1:], expanded_invariants[1:], strict=True
    ):
        np.testing.assert_allclose(
            compact_value,
            expanded_value,
            rtol=0.0,
            atol=_roundoff_bound(np.atleast_1d(expanded_value), mass=float(len(repeated))),
        )
    if route == "standalone":
        compact_observables = _standalone_polynomial_observables(
            compact,
            frame,
            frame,
            compact_weights.geometry_values,
        )
        expanded_observables = _standalone_polynomial_observables(
            expanded,
            frame,
            expanded_frame,
            expanded_weights.geometry_values,
        )
        for compact_value, expanded_value in zip(
            compact_observables, expanded_observables, strict=True
        ):
            np.testing.assert_allclose(
                compact_value,
                expanded_value,
                rtol=0.0,
                atol=_roundoff_bound(expanded_value, mass=float(len(repeated))),
            )


@pytest.mark.parametrize("route", _POLYNOMIAL_GEOMETRY_ROUTES)
def test_prior_polynomial_geometry_follows_physical_rows(route: str) -> None:
    frame, _ = _polynomial_geometry_fixture()
    unequal = np.geomspace(0.1, 20.0, len(frame))
    prior_weights, weighted = _polynomial_geometry_compilation(
        frame,
        unequal,
        semantics="prior",
        route=route,
    )
    unit_weights, physical = _polynomial_geometry_compilation(
        frame,
        np.ones(len(frame)),
        semantics="prior",
        route=route,
    )

    weighted_invariants = _term_geometry_invariants(
        weighted,
        frame,
        frame,
        prior_weights.geometry_values,
        term_name=_polynomial_term_name(route),
    )
    physical_invariants = _term_geometry_invariants(
        physical,
        frame,
        frame,
        unit_weights.geometry_values,
        term_name=_polynomial_term_name(route),
    )
    assert weighted_invariants[0] == physical_invariants[0]
    for weighted_value, physical_value in zip(
        weighted_invariants[1:], physical_invariants[1:], strict=True
    ):
        np.testing.assert_allclose(
            weighted_value,
            physical_value,
            rtol=0.0,
            atol=_roundoff_bound(np.atleast_1d(physical_value), mass=float(len(frame))),
        )
    if route == "standalone":
        weighted_observables = _standalone_polynomial_observables(
            weighted,
            frame,
            frame,
            prior_weights.geometry_values,
        )
        physical_observables = _standalone_polynomial_observables(
            physical,
            frame,
            frame,
            unit_weights.geometry_values,
        )
        for weighted_value, physical_value in zip(
            weighted_observables, physical_observables, strict=True
        ):
            np.testing.assert_allclose(
                weighted_value,
                physical_value,
                rtol=0.0,
                atol=_roundoff_bound(physical_value, mass=float(len(frame))),
            )


@pytest.mark.parametrize("discrete", [False, True], ids=["dense", "discrete"])
def test_frequency_ordinary_spline_identifiable_geometry_matches_literal_expansion(
    discrete: bool,
) -> None:
    frame, counts = _geometry_fixture()
    repeated = np.repeat(np.arange(len(frame)), counts.astype(np.intp))
    compact_weights, compact = _ordinary_geometry_compilation(
        frame[["x"]],
        counts,
        semantics="frequency",
        discrete=discrete,
    )
    expanded_weights, expanded = _ordinary_geometry_compilation(
        frame[["x"]].iloc[repeated].reset_index(drop=True),
        np.ones(len(repeated)),
        semantics="prior",
        discrete=discrete,
    )

    compact_matrix = compact[0].compiled.design.group_matrices[0]
    expanded_matrix = expanded[0].compiled.design.group_matrices[0]
    compact_dense = compact_matrix.toarray()
    expanded_dense = expanded_matrix.toarray()
    tolerance = _roundoff_bound(expanded_dense, mass=float(len(repeated)))
    np.testing.assert_allclose(
        compact_dense,
        expanded_dense[np.cumsum(np.r_[0.0, counts[:-1]]).astype(np.intp)],
        rtol=0.0,
        atol=tolerance,
    )
    np.testing.assert_allclose(
        compact_matrix.rmatvec(compact_weights.geometry_values),
        0.0,
        rtol=0.0,
        atol=tolerance,
    )
    np.testing.assert_allclose(
        expanded_matrix.rmatvec(expanded_weights.geometry_values),
        0.0,
        rtol=0.0,
        atol=tolerance,
    )
    np.testing.assert_allclose(
        _prediction_matrix(frame[["x"]], compact),
        _prediction_matrix(frame[["x"]], expanded),
        rtol=0.0,
        atol=tolerance,
    )


@pytest.mark.parametrize("discrete", [False, True], ids=["dense", "discrete"])
def test_prior_ordinary_spline_identifiable_geometry_follows_physical_rows(
    discrete: bool,
) -> None:
    frame, _ = _geometry_fixture()
    unequal = np.array([0.1, 12.0, 0.2, 8.0, 0.4, 6.0, 0.3, 4.0, 0.5, 2.0])
    resolved, weighted = _ordinary_geometry_compilation(
        frame[["x"]], unequal, semantics="prior", discrete=discrete
    )
    _, physical = _ordinary_geometry_compilation(
        frame[["x"]], np.ones(len(frame)), semantics="prior", discrete=discrete
    )

    weighted_spec = weighted[0].compiled.specs["x"]
    physical_spec = physical[0].compiled.specs["x"]
    np.testing.assert_allclose(weighted_spec._knots, physical_spec._knots, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        _projection_projector(weighted_spec._interaction_projection),
        _projection_projector(physical_spec._interaction_projection),
        rtol=0.0,
        atol=_roundoff_bound(physical_spec._interaction_projection, mass=float(len(frame))),
    )
    matrix = weighted[0].compiled.design.group_matrices[0]
    np.testing.assert_allclose(
        matrix.rmatvec(resolved.geometry_values),
        0.0,
        rtol=0.0,
        atol=_roundoff_bound(matrix.toarray(), mass=float(len(frame))),
    )


def _tensor_centering_moments(compiled, geometry: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    predictor = compiled[0]
    interaction = predictor.compiled.interaction_specs["x1:x2"]
    group_index = next(
        index
        for index, group in enumerate(predictor.compiled.groups)
        if group.feature_name == "x1:x2"
    )
    matrix = predictor.compiled.design.group_matrices[group_index]
    marginal1 = np.asarray(interaction._marginal1.basis, dtype=np.float64)
    marginal2 = np.asarray(interaction._marginal2.basis, dtype=np.float64)
    if hasattr(matrix, "idx1"):
        mass1 = np.bincount(matrix.idx1, weights=geometry, minlength=len(marginal1))
        mass2 = np.bincount(matrix.idx2, weights=geometry, minlength=len(marginal2))
        return mass1 @ marginal1, mass2 @ marginal2
    return geometry @ marginal1, geometry @ marginal2


@pytest.mark.parametrize("discrete", [False, True], ids=["dense", "discrete"])
def test_frequency_tensor_identifiable_geometry_matches_literal_expansion(discrete: bool) -> None:
    frame, counts = _geometry_fixture()
    tensor_frame = frame[["x1", "x2"]]
    repeated = np.repeat(np.arange(len(frame)), counts.astype(np.intp))
    compact_weights, compact = _tensor_geometry_compilation(
        tensor_frame,
        counts,
        semantics="frequency",
        discrete=discrete,
    )
    expanded_weights, expanded = _tensor_geometry_compilation(
        tensor_frame.iloc[repeated].reset_index(drop=True),
        np.ones(len(repeated)),
        semantics="prior",
        discrete=discrete,
    )

    compact_interaction = compact[0].compiled.interaction_specs["x1:x2"]
    expanded_interaction = expanded[0].compiled.interaction_specs["x1:x2"]
    for compact_marginal, expanded_marginal in zip(
        (compact_interaction._marginal1, compact_interaction._marginal2),
        (expanded_interaction._marginal1, expanded_interaction._marginal2),
        strict=True,
    ):
        tolerance = _roundoff_bound(
            expanded_marginal.projection,
            mass=float(len(repeated)),
        )
        np.testing.assert_allclose(
            _projection_projector(compact_marginal.projection),
            _projection_projector(expanded_marginal.projection),
            rtol=0.0,
            atol=tolerance,
        )
    for moment in (
        *_tensor_centering_moments(compact, compact_weights.geometry_values),
        *_tensor_centering_moments(expanded, expanded_weights.geometry_values),
    ):
        np.testing.assert_allclose(
            moment,
            0.0,
            rtol=0.0,
            atol=_roundoff_bound(moment, mass=float(len(repeated))),
        )
    np.testing.assert_allclose(
        _prediction_matrix(tensor_frame, compact),
        _prediction_matrix(tensor_frame, expanded),
        rtol=0.0,
        atol=_roundoff_bound(
            _prediction_matrix(tensor_frame, expanded),
            mass=float(len(repeated)),
        ),
    )


@pytest.mark.parametrize("discrete", [False, True], ids=["dense", "discrete"])
def test_prior_tensor_identifiable_geometry_follows_physical_rows(discrete: bool) -> None:
    frame, _ = _geometry_fixture()
    tensor_frame = frame[["x1", "x2"]]
    unequal = np.array([0.1, 12.0, 0.2, 8.0, 0.4, 6.0, 0.3, 4.0, 0.5, 2.0])
    resolved, weighted = _tensor_geometry_compilation(
        tensor_frame,
        unequal,
        semantics="prior",
        discrete=discrete,
    )
    _, physical = _tensor_geometry_compilation(
        tensor_frame,
        np.ones(len(frame)),
        semantics="prior",
        discrete=discrete,
    )

    weighted_interaction = weighted[0].compiled.interaction_specs["x1:x2"]
    physical_interaction = physical[0].compiled.interaction_specs["x1:x2"]
    for weighted_marginal, physical_marginal in zip(
        (weighted_interaction._marginal1, weighted_interaction._marginal2),
        (physical_interaction._marginal1, physical_interaction._marginal2),
        strict=True,
    ):
        np.testing.assert_allclose(
            _projection_projector(weighted_marginal.projection),
            _projection_projector(physical_marginal.projection),
            rtol=0.0,
            atol=_roundoff_bound(
                physical_marginal.projection,
                mass=float(len(frame)),
            ),
        )
    for moment in _tensor_centering_moments(weighted, resolved.geometry_values):
        np.testing.assert_allclose(
            moment,
            0.0,
            rtol=0.0,
            atol=_roundoff_bound(moment, mass=float(len(frame))),
        )


def _spline_categorical_geometry_compilation(
    frame: pd.DataFrame,
    sample_weight: np.ndarray,
    *,
    semantics: WeightSemantics,
    discrete: bool,
    select: bool = False,
):
    resolved = _resolved(sample_weight, semantics=semantics)
    compiled = compile_predictors(
        as_eager_frame(frame),
        resolved,
        GaussianLS().parameters,
        (
            Predictor(
                "location",
                {
                    "x": Spline(
                        kind="cr",
                        n_knots=5,
                        knot_strategy="quantile_rows",
                        discrete=discrete,
                        select=select,
                    ),
                    "g": Categorical(base="a"),
                },
                interaction_specs={"x:g": SplineCategorical("x", "g")},
            ),
            Predictor("scale", {}),
        ),
        model_discrete=discrete,
        n_bins_config=20,
    )
    return resolved, compiled


def _spline_categorical_centering_moment(
    compiled,
    x: np.ndarray,
    geometry: np.ndarray,
) -> np.ndarray:
    interaction = compiled[0].compiled.interaction_specs["x:g"]
    raw_basis = interaction._spline_spec._raw_basis_matrix(np.asarray(x, dtype=np.float64))
    return geometry @ (raw_basis @ interaction._projection)


@pytest.mark.parametrize("discrete", [False, True], ids=["dense", "discrete"])
def test_frequency_selected_spline_categorical_geometry_matches_literal_expansion(
    discrete: bool,
) -> None:
    frame, counts = _geometry_fixture()
    spline_cat_frame = frame[["x"]].assign(
        g=np.array(["a", "b", "c", "b", "c", "a", "b", "c", "a", "b"])
    )
    repeated = np.repeat(np.arange(len(frame)), counts.astype(np.intp))
    compact_weights, compact = _spline_categorical_geometry_compilation(
        spline_cat_frame,
        counts,
        semantics="frequency",
        discrete=discrete,
        select=True,
    )
    expanded_frame = spline_cat_frame.iloc[repeated].reset_index(drop=True)
    expanded_weights, expanded = _spline_categorical_geometry_compilation(
        expanded_frame,
        np.ones(len(repeated)),
        semantics="prior",
        discrete=discrete,
        select=True,
    )

    compact_parent = compact[0].compiled.specs["x"]
    expanded_parent = expanded[0].compiled.specs["x"]
    compact_interaction = compact[0].compiled.interaction_specs["x:g"]
    expanded_interaction = expanded[0].compiled.interaction_specs["x:g"]
    for compact_projection, expanded_projection in (
        (compact_parent._interaction_projection, expanded_parent._interaction_projection),
        (compact_interaction._projection, expanded_interaction._projection),
    ):
        np.testing.assert_allclose(
            _projection_projector(compact_projection),
            _projection_projector(expanded_projection),
            rtol=0.0,
            atol=_roundoff_bound(expanded_projection, mass=float(len(repeated))),
        )
    compact_invariants = _term_geometry_invariants(
        compact,
        spline_cat_frame,
        spline_cat_frame,
        compact_weights.geometry_values,
        term_name="x:g",
    )
    expanded_invariants = _term_geometry_invariants(
        expanded,
        spline_cat_frame,
        expanded_frame,
        expanded_weights.geometry_values,
        term_name="x:g",
    )
    assert compact_invariants[0] == expanded_invariants[0]
    for compact_value, expanded_value in zip(
        compact_invariants[1:], expanded_invariants[1:], strict=True
    ):
        np.testing.assert_allclose(
            compact_value,
            expanded_value,
            rtol=0.0,
            atol=_roundoff_bound(np.atleast_1d(expanded_value), mass=float(len(repeated))),
        )


@pytest.mark.parametrize("discrete", [False, True], ids=["dense", "discrete"])
def test_prior_selected_spline_categorical_geometry_follows_physical_rows(
    discrete: bool,
) -> None:
    frame, _ = _geometry_fixture()
    spline_cat_frame = frame[["x"]].assign(
        g=np.array(["a", "b", "c", "b", "c", "a", "b", "c", "a", "b"])
    )
    unequal = np.array([0.1, 12.0, 0.2, 8.0, 0.4, 6.0, 0.3, 4.0, 0.5, 2.0])
    prior_weights, weighted = _spline_categorical_geometry_compilation(
        spline_cat_frame,
        unequal,
        semantics="prior",
        discrete=discrete,
        select=True,
    )
    unit_weights, physical = _spline_categorical_geometry_compilation(
        spline_cat_frame,
        np.ones(len(frame)),
        semantics="prior",
        discrete=discrete,
        select=True,
    )

    weighted_parent = weighted[0].compiled.specs["x"]
    physical_parent = physical[0].compiled.specs["x"]
    weighted_interaction = weighted[0].compiled.interaction_specs["x:g"]
    physical_interaction = physical[0].compiled.interaction_specs["x:g"]
    for weighted_projection, physical_projection in (
        (weighted_parent._interaction_projection, physical_parent._interaction_projection),
        (weighted_interaction._projection, physical_interaction._projection),
    ):
        np.testing.assert_allclose(
            _projection_projector(weighted_projection),
            _projection_projector(physical_projection),
            rtol=0.0,
            atol=_roundoff_bound(physical_projection, mass=float(len(frame))),
        )
    weighted_invariants = _term_geometry_invariants(
        weighted,
        spline_cat_frame,
        spline_cat_frame,
        prior_weights.geometry_values,
        term_name="x:g",
    )
    physical_invariants = _term_geometry_invariants(
        physical,
        spline_cat_frame,
        spline_cat_frame,
        unit_weights.geometry_values,
        term_name="x:g",
    )
    assert weighted_invariants[0] == physical_invariants[0]
    for weighted_value, physical_value in zip(
        weighted_invariants[1:], physical_invariants[1:], strict=True
    ):
        np.testing.assert_allclose(
            weighted_value,
            physical_value,
            rtol=0.0,
            atol=_roundoff_bound(np.atleast_1d(physical_value), mass=float(len(frame))),
        )


@pytest.mark.parametrize("discrete", [False, True], ids=["dense", "discrete"])
def test_frequency_spline_categorical_geometry_matches_literal_expansion(discrete: bool) -> None:
    frame, counts = _geometry_fixture()
    spline_cat_frame = frame[["x"]].assign(
        g=np.array(["a", "b", "c", "b", "c", "a", "b", "c", "a", "b"])
    )
    repeated = np.repeat(np.arange(len(frame)), counts.astype(np.intp))
    compact_weights, compact = _spline_categorical_geometry_compilation(
        spline_cat_frame,
        counts,
        semantics="frequency",
        discrete=discrete,
    )
    expanded_weights, expanded = _spline_categorical_geometry_compilation(
        spline_cat_frame.iloc[repeated].reset_index(drop=True),
        np.ones(len(repeated)),
        semantics="prior",
        discrete=discrete,
    )

    compact_interaction = compact[0].compiled.interaction_specs["x:g"]
    expanded_interaction = expanded[0].compiled.interaction_specs["x:g"]
    np.testing.assert_allclose(
        compact_interaction._knots,
        expanded_interaction._knots,
        rtol=0.0,
        atol=0.0,
    )
    tolerance = _roundoff_bound(
        expanded_interaction._projection,
        mass=float(len(repeated)),
    )
    np.testing.assert_allclose(
        _projection_projector(compact_interaction._projection),
        _projection_projector(expanded_interaction._projection),
        rtol=0.0,
        atol=tolerance,
    )
    np.testing.assert_allclose(
        _spline_categorical_centering_moment(
            compact,
            spline_cat_frame["x"].to_numpy(),
            compact_weights.geometry_values,
        ),
        0.0,
        rtol=0.0,
        atol=tolerance,
    )
    np.testing.assert_allclose(
        _spline_categorical_centering_moment(
            expanded,
            spline_cat_frame["x"].to_numpy()[repeated],
            expanded_weights.geometry_values,
        ),
        0.0,
        rtol=0.0,
        atol=tolerance,
    )
    np.testing.assert_allclose(
        _prediction_matrix(spline_cat_frame, compact),
        _prediction_matrix(spline_cat_frame, expanded),
        rtol=0.0,
        atol=_roundoff_bound(
            _prediction_matrix(spline_cat_frame, expanded),
            mass=float(len(repeated)),
        ),
    )


@pytest.mark.parametrize("discrete", [False, True], ids=["dense", "discrete"])
def test_prior_spline_categorical_geometry_follows_physical_rows(discrete: bool) -> None:
    frame, _ = _geometry_fixture()
    spline_cat_frame = frame[["x"]].assign(
        g=np.array(["a", "b", "c", "b", "c", "a", "b", "c", "a", "b"])
    )
    unequal = np.array([0.1, 12.0, 0.2, 8.0, 0.4, 6.0, 0.3, 4.0, 0.5, 2.0])
    resolved, weighted = _spline_categorical_geometry_compilation(
        spline_cat_frame,
        unequal,
        semantics="prior",
        discrete=discrete,
    )
    _, physical = _spline_categorical_geometry_compilation(
        spline_cat_frame,
        np.ones(len(frame)),
        semantics="prior",
        discrete=discrete,
    )

    weighted_interaction = weighted[0].compiled.interaction_specs["x:g"]
    physical_interaction = physical[0].compiled.interaction_specs["x:g"]
    np.testing.assert_allclose(
        weighted_interaction._knots,
        physical_interaction._knots,
        rtol=0.0,
        atol=0.0,
    )
    tolerance = _roundoff_bound(
        physical_interaction._projection,
        mass=float(len(frame)),
    )
    np.testing.assert_allclose(
        _projection_projector(weighted_interaction._projection),
        _projection_projector(physical_interaction._projection),
        rtol=0.0,
        atol=tolerance,
    )
    np.testing.assert_allclose(
        _spline_categorical_centering_moment(
            weighted,
            spline_cat_frame["x"].to_numpy(),
            resolved.geometry_values,
        ),
        0.0,
        rtol=0.0,
        atol=tolerance,
    )


def _factor_smooth_geometry_fixture() -> tuple[pd.DataFrame, np.ndarray]:
    n_observations = 30
    return (
        pd.DataFrame(
            {
                "x": np.linspace(-3.0, 3.0, n_observations)
                + 0.08 * np.sin(np.arange(n_observations)),
                "g": np.resize(np.array(["a", "b", "c"]), n_observations),
            }
        ),
        np.resize(np.array([1, 4, 2, 6, 1, 3, 5], dtype=np.float64), n_observations),
    )


def _factor_smooth_geometry_compilation(
    frame: pd.DataFrame,
    sample_weight: np.ndarray,
    *,
    semantics: WeightSemantics,
    discrete: bool,
):
    term_name = "x:g:fs"
    resolved = _resolved(sample_weight, semantics=semantics)
    compiled = compile_predictors(
        as_eager_frame(frame),
        resolved,
        GaussianLS().parameters,
        (
            Predictor(
                "location",
                {"x": Numeric(), "g": Categorical(base="a")},
                interaction_specs={
                    term_name: FactorSmooth("x", group="g", k=6),
                },
            ),
            Predictor("scale", {}),
        ),
        model_discrete=discrete,
        n_bins_config=40,
    )
    return resolved, compiled


def _factor_smooth_invariants(
    compiled,
    scoring_frame: pd.DataFrame,
    fit_frame: pd.DataFrame,
    fit_weight: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, float]:
    term_name = "x:g:fs"
    layout = build_stacked_layout(compiled)
    group = next(group for group in compiled[0].compiled.groups if group.feature_name == term_name)
    intercept_width = int(layout.predictor("location").intercept_index is not None)
    term_slice = slice(group.start + intercept_width, group.end + intercept_width)
    scoring_design = build_joint_prediction_design(
        scoring_frame,
        compiled,
        layout,
    ).local["location"][:, term_slice]
    left, singular_values, _ = np.linalg.svd(scoring_design, full_matrices=False)
    rank_tolerance = np.finfo(np.float64).eps * max(scoring_design.shape) * singular_values[0]
    rank = int(np.sum(singular_values > rank_tolerance))
    projector = left[:, :rank] @ left[:, :rank].T
    target = np.sin(scoring_frame["x"].to_numpy()) + 0.2 * (scoring_frame["g"].to_numpy() == "c")
    fitted_function = projector @ target

    lambdas = {name: 1.0 for name in layout.penalty_names}
    penalty = layout.penalty_matrix(lambdas)[term_slice, term_slice]
    inverse_design = np.linalg.pinv(scoring_design, rcond=rank_tolerance / singular_values[0])
    function_penalty = inverse_design.T @ penalty @ inverse_design

    fit_design = build_joint_prediction_design(
        fit_frame,
        compiled,
        layout,
    ).local["location"][:, term_slice]
    gram = fit_design.T @ (fit_design * fit_weight[:, None])
    edf = float(np.trace(np.linalg.solve(gram + penalty, gram)))
    return rank, projector, fitted_function, function_penalty, edf


@pytest.mark.parametrize("discrete", [False, True], ids=["dense", "discrete"])
def test_frequency_factor_smooth_invariants_match_literal_expansion(discrete: bool) -> None:
    frame, counts = _factor_smooth_geometry_fixture()
    repeated = np.repeat(np.arange(len(frame)), counts.astype(np.intp))
    _, compact = _factor_smooth_geometry_compilation(
        frame,
        counts,
        semantics="frequency",
        discrete=discrete,
    )
    expanded_frame = frame.iloc[repeated].reset_index(drop=True)
    _, expanded = _factor_smooth_geometry_compilation(
        expanded_frame,
        np.ones(len(repeated)),
        semantics="prior",
        discrete=discrete,
    )

    compact_invariants = _factor_smooth_invariants(
        compact,
        frame,
        frame,
        counts,
    )
    expanded_invariants = _factor_smooth_invariants(
        expanded,
        frame,
        expanded_frame,
        np.ones(len(repeated)),
    )
    compact_rank, compact_projector, compact_fit, compact_penalty, compact_edf = compact_invariants
    expanded_rank, expanded_projector, expanded_fit, expanded_penalty, expanded_edf = (
        expanded_invariants
    )
    tolerance = _roundoff_bound(expanded_penalty, mass=float(len(repeated)))
    assert compact_rank == expanded_rank
    np.testing.assert_allclose(
        compact_projector,
        expanded_projector,
        rtol=0.0,
        atol=tolerance,
    )
    np.testing.assert_allclose(
        compact_fit,
        expanded_fit,
        rtol=0.0,
        atol=tolerance,
    )
    np.testing.assert_allclose(
        compact_penalty,
        expanded_penalty,
        rtol=0.0,
        atol=tolerance,
    )
    np.testing.assert_allclose(
        compact_edf,
        expanded_edf,
        rtol=0.0,
        atol=tolerance,
    )


@pytest.mark.parametrize("discrete", [False, True], ids=["dense", "discrete"])
def test_prior_factor_smooth_geometry_follows_physical_rows(discrete: bool) -> None:
    frame, _ = _factor_smooth_geometry_fixture()
    unequal = np.linspace(0.1, 5.0, len(frame))
    _, weighted = _factor_smooth_geometry_compilation(
        frame,
        unequal,
        semantics="prior",
        discrete=discrete,
    )
    _, physical = _factor_smooth_geometry_compilation(
        frame,
        np.ones(len(frame)),
        semantics="prior",
        discrete=discrete,
    )
    weighted_rank, weighted_projector, weighted_fit, weighted_penalty, _ = (
        _factor_smooth_invariants(weighted, frame, frame, unequal)
    )
    physical_rank, physical_projector, physical_fit, physical_penalty, _ = (
        _factor_smooth_invariants(
            physical,
            frame,
            frame,
            np.ones(len(frame)),
        )
    )
    tolerance = _roundoff_bound(physical_penalty, mass=float(len(frame)))
    assert weighted_rank == physical_rank
    np.testing.assert_allclose(
        weighted_projector,
        physical_projector,
        rtol=0.0,
        atol=tolerance,
    )
    np.testing.assert_allclose(
        weighted_fit,
        physical_fit,
        rtol=0.0,
        atol=tolerance,
    )
    np.testing.assert_allclose(
        weighted_penalty,
        physical_penalty,
        rtol=0.0,
        atol=tolerance,
    )


def test_superlss_declares_the_prior_contract_by_default() -> None:
    frame, y, weights = _weighted_knot_fixture()
    predictors = [
        Predictor(name="location", features={"x": Spline("cr", k=8)}),
        Predictor(name="scale", features={"z": Spline("cr", k=6)}),
    ]

    default = SuperLSS(family=GaussianLS(), predictors=predictors)
    declared = SuperLSS(family=GaussianLS(), predictors=predictors, weight_semantics="frequency")

    assert default._weight_contract == WeightContract(semantics="prior")
    assert declared._weight_contract == WeightContract(semantics="frequency")
    assert default.weight_semantics == "prior"
    assert declared.weight_semantics == "frequency"

    default.fit(
        frame,
        y,
        sample_weight=weights,
        lambdas={"location:x#wiggle": 1.0, "scale:z#wiggle": 1.0},
    )
    assert default._require_fitted().null_model.weight_semantics == "prior"

    with pytest.raises(UnsupportedLikelihoodContractError, match="semantics"):
        SuperLSS(family=GaussianLS(), predictors=predictors, weight_semantics="frequency_case")
