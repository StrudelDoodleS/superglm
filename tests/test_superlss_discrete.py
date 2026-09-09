"""Public grouped fitting on the identical, unquantized repeated-support design."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import superglm.distributional.solver.chunks as chunking
from superglm import SuperLSS
from superglm._group_matrix._cross_matrix_execution import CrossMatrixExecutionPlan
from superglm._group_matrix._group_matrix_execution import MatrixExecutionPlan
from superglm.distributional import GaussianLS, Predictor, TweedieLSS
from superglm.distributional.families.gamma import GammaLS
from superglm.distributional.families.generalized_gamma import GeneralizedGammaLSS
from superglm.distributional.families.generalized_pareto import GeneralizedParetoLSS
from superglm.distributional.families.log_normal import LogNormalLS
from superglm.distributional.families.negative_binomial import NegativeBinomialLS
from superglm.distributional.families.two_piece import TwoPieceLogNormalLSS, TwoPieceNormalLSS
from superglm.distributional.solver.assembly import dense_predictor_matrices
from superglm.features import Categorical, Spline, SplineCategorical
from superglm.features.interaction import TensorInteraction
from superglm.types import LambdaPolicy


def _fixture(kind: str, semantics: str):
    rng = np.random.default_rng(202609082)
    n = 384
    x = rng.choice(np.linspace(-1, 1, 9), n)
    z = rng.choice(np.linspace(-1, 1, 7), n)
    frame = pd.DataFrame({"x": x, "z": z, "g": rng.choice(["a", "b", "c"], n)})
    weights = rng.integers(1, 4, n).astype(float)
    if semantics == "prior":
        weights /= 2
    weights[::29] = 0
    if kind == "gaussian":
        family = GaussianLS(scale_floor=0.02)
        names = ("location", "scale")
        y = 0.3 + 0.5 * np.sin(np.pi * x) + rng.normal(0, 0.45, n)
    elif kind == "nb2":
        family = NegativeBinomialLS()
        names = ("mean", "theta")
        mean = np.exp(0.6 + 0.3 * np.sin(np.pi * x))
        theta = 2.0
        exposure = weights if semantics == "prior" else np.ones(n)
        exposure = np.maximum(exposure, 0.5)
        y = rng.negative_binomial(exposure * theta, theta / (theta + mean)) / exposure
    else:
        family = TweedieLSS(power_lower=1.08, power_upper=1.92)
        names = ("mean", "dispersion", "power")
        mean = np.exp(0.3 + 0.2 * np.sin(np.pi * x))
        exposure = weights if semantics == "prior" else np.ones(n)
        exposure = np.maximum(exposure, 0.5)
        counts = rng.poisson(2 * np.sqrt(mean) * exposure)
        y = np.zeros(n)
        positive = counts > 0
        y[positive] = rng.gamma(
            counts[positive], np.sqrt(mean[positive]) / (2 * exposure[positive])
        )
    offsets = {name: 0.015 * np.sin(np.arange(n) + i) for i, name in enumerate(names)}
    predictors = tuple(
        Predictor(
            name,
            {"x": Spline(kind="cr", n_knots=4, lambda_policy=LambdaPolicy.estimate())}
            if i == 0
            else {},
        )
        for i, name in enumerate(names)
    )
    return frame, np.asarray(y, dtype=float), weights, offsets, family, predictors


def _assert_parity(dense: SuperLSS, grouped: SuperLSS, frame, offsets):
    # A solve-amplified roundoff envelope for these well-conditioned fits.
    n, p = len(frame), len(dense.coef_)
    tolerance = np.sqrt(np.finfo(float).eps) * (n + p)
    left, right = dense._require_fitted(), grouped._require_fitted()
    for a, b in zip(dense_predictor_matrices(left.layout), dense_predictor_matrices(right.layout)):
        np.testing.assert_allclose(a, b, rtol=0, atol=128 * p * np.finfo(float).eps)
    np.testing.assert_allclose(
        list(dense.coef_.values()), list(grouped.coef_.values()), rtol=tolerance, atol=tolerance
    )
    np.testing.assert_allclose(
        dense.covariance_, grouped.covariance_, rtol=tolerance, atol=tolerance
    )
    np.testing.assert_allclose(
        dense.predict_parameters(frame, offsets=offsets),
        grouped.predict_parameters(frame, offsets=offsets),
        rtol=tolerance,
        atol=tolerance,
    )
    np.testing.assert_allclose(
        left.result.log_likelihood, right.result.log_likelihood, rtol=tolerance
    )
    np.testing.assert_allclose(
        left.result.terminal_score, right.result.terminal_score, rtol=tolerance, atol=tolerance
    )
    np.testing.assert_allclose(
        left.result.terminal_data_curvature,
        right.result.terminal_data_curvature,
        rtol=tolerance,
        atol=tolerance,
    )
    assert dense.result_.coefficient_converged and grouped.result_.coefficient_converged
    assert right.result.execution_backend_identifier == "distributional-chunked-v1"
    assert right.result.resolved_chunk_size is not None
    assert grouped.result_.curvature_telemetry.actual_source == "observed"


@pytest.mark.parametrize("kind", ["gaussian", "nb2", "tweedie"])
@pytest.mark.parametrize("semantics", ["prior", "frequency"])
def test_public_discrete_observed_fixed_fit_parity(kind: str, semantics: str, monkeypatch) -> None:
    monkeypatch.setattr(chunking, "AUTO_CHUNK_MEMORY_BYTES", 8192)
    frame, y, weights, offsets, family, predictors = _fixture(kind, semantics)
    models = [
        SuperLSS(
            family=family,
            predictors=predictors,
            weight_semantics=semantics,
            discrete=discrete,
            n_bins=32,
        ).fit(
            frame,
            y,
            sample_weight=weights,
            offsets=offsets,
            lambdas={f"{predictors[0].name}:x#wiggle": 1.0},
        )
        for discrete in (False, True)
    ]
    _assert_parity(*models, frame, offsets)


@pytest.mark.parametrize("kind", ["gaussian", "nb2", "tweedie"])
@pytest.mark.parametrize("semantics", ["prior", "frequency"])
def test_public_discrete_complete_smoothing_and_roundtrip(
    kind: str, semantics: str, monkeypatch
) -> None:
    monkeypatch.setattr(chunking, "AUTO_CHUNK_MEMORY_BYTES", 8192)
    frame, y, weights, offsets, family, predictors = _fixture(kind, semantics)
    models = [
        SuperLSS(
            family=family,
            predictors=predictors,
            discrete=discrete,
            n_bins=32,
            weight_semantics=semantics,
        ).fit_reml(
            frame,
            y,
            sample_weight=weights,
            offsets=offsets,
            outer="efs+newton",
        )
        for discrete in (False, True)
    ]
    dense, grouped = models
    _assert_parity(dense, grouped, frame, offsets)
    assert dense.smoothing_convergence_reason_ == grouped.smoothing_convergence_reason_
    assert dense.smoothing_certified_ and grouped.smoothing_certified_
    np.testing.assert_allclose(
        list(dense.smoothing_parameters_.values()),
        list(grouped.smoothing_parameters_.values()),
        rtol=np.sqrt(np.finfo(float).eps) * len(frame),
    )
    restored = SuperLSS.from_bytes(grouped.to_bytes())
    assert restored.discrete and restored.n_bins == 32
    assert restored.training_telemetry() == grouped.training_telemetry()
    np.testing.assert_array_equal(restored.covariance_, grouped.covariance_)
    np.testing.assert_array_equal(
        restored.predict_parameters(frame, offsets=offsets),
        grouped.predict_parameters(frame, offsets=offsets),
    )
    # Predictor templates returned to callers are independent and can seed a fresh model.
    clone = SuperLSS(
        family=family,
        predictors=restored.predictors,
        discrete=True,
        n_bins=32,
        weight_semantics=semantics,
    )
    fixed = dict(grouped.smoothing_parameters_)
    clone.fit(frame, y, sample_weight=weights, offsets=offsets, lambdas=fixed)
    restored.fit(frame, y, sample_weight=weights, offsets=offsets, lambdas=fixed)
    assert restored.training_telemetry().execution_backend_identifier == "distributional-chunked-v1"
    assert restored.training_telemetry().resolved_chunk_size is not None
    np.testing.assert_array_equal(
        clone.predict_parameters(frame), restored.predict_parameters(frame)
    )


@pytest.mark.parametrize("kind", ["nb2", "tweedie"])
def test_public_discrete_explicit_fisher_still_requires_capability(kind: str) -> None:
    _, _, _, _, family, predictors = _fixture(kind, "prior")
    with pytest.raises(ValueError, match="expected information"):
        SuperLSS(
            family=family, predictors=predictors, discrete=True, coefficient_curvature="fisher"
        )


@pytest.mark.parametrize("discrete", [False, True])
def test_public_serialization_size_ignores_primed_category_lookups(discrete) -> None:
    from superglm._group_matrix._group_matrix_core import SplineCategoricalGroupMatrix
    from superglm._group_matrix._group_matrix_discretized import (
        DiscretizedSplineCategoricalGroupMatrix,
    )

    frame, y, _, _, family, _ = _fixture("gaussian", "frequency")
    model = SuperLSS(
        family=family,
        predictors=(
            Predictor(
                "location",
                {"x": Spline(kind="cr", n_knots=4), "g": Categorical(base="a")},
                interaction_specs={"x:g": SplineCategorical("x", "g")},
            ),
            Predictor("scale", {}),
        ),
        discrete=discrete,
        n_bins=32,
    ).fit(
        frame,
        y,
        lambdas=dict.fromkeys(
            ("location:x#wiggle", "location:x:g[b]#wiggle", "location:x:g[c]#wiggle"),
            1.0,
        ),
    )
    model = SuperLSS.from_bytes(model.to_bytes())
    cold_size = len(model.to_bytes())
    groups = [
        group
        for state in model._require_fitted().layout.predictors
        for group in state.design.group_matrices
        if isinstance(
            group, (SplineCategoricalGroupMatrix, DiscretizedSplineCategoricalGroupMatrix)
        )
    ]
    assert groups
    for group in groups:
        assert group._sorted_rows is None
        group.row_subset(np.arange(17))
        assert group._sorted_rows is not None
    payload = model.to_bytes()
    assert len(payload) == cold_size
    restored = SuperLSS.from_bytes(payload)
    np.testing.assert_array_equal(
        restored.predict_parameters(frame), model.predict_parameters(frame)
    )


def test_public_discrete_tensor_and_categorical_interaction_parity() -> None:
    frame, y, weights, offsets, family, _ = _fixture("gaussian", "frequency")
    predictors = (
        Predictor(
            "location",
            {
                "x": Spline(kind="cr", n_knots=4),
                "z": Spline(kind="cr", n_knots=4),
                "g": Categorical(base="a"),
            },
            interaction_specs={"x:z": TensorInteraction("x", "z", n_knots=(4, 4))},
        ),
        Predictor(
            "scale",
            {"x": Spline(kind="cr", n_knots=4), "g": Categorical(base="a")},
            interaction_specs={"x:g": SplineCategorical("x", "g")},
        ),
    )
    models = [
        SuperLSS(
            family=family,
            predictors=predictors,
            discrete=discrete,
            n_bins=32,
            weight_semantics="frequency",
        ).fit(
            frame,
            y,
            sample_weight=weights,
            offsets=offsets,
            lambdas=dict.fromkeys(
                (
                    "location:x#wiggle",
                    "location:z#wiggle",
                    "location:x:z#margin_x",
                    "location:x:z#margin_z",
                    "scale:x#wiggle",
                    "scale:x:g[b]#wiggle",
                    "scale:x:g[c]#wiggle",
                ),
                1.0,
            ),
        )
        for discrete in (False, True)
    ]
    _assert_parity(*models, frame, offsets)


@pytest.mark.parametrize(
    "family",
    [
        GammaLS(),
        LogNormalLS(),
        GeneralizedGammaLSS(),
        GeneralizedParetoLSS(),
        TwoPieceNormalLSS(),
        TwoPieceLogNormalLSS(),
    ],
)
def test_public_discrete_remaining_builtin_families(family) -> None:
    frame, _, weights, _, _, _ = _fixture("gaussian", "frequency")
    rng = np.random.default_rng(202609085)
    n = len(frame)
    location = 0.3 + 0.15 * np.sin(np.pi * frame["x"].to_numpy())
    if isinstance(family, GammaLS):
        y = rng.gamma(3.0, np.exp(location) / 3.0)
    elif isinstance(family, GeneralizedParetoLSS):
        y = np.exp(location) * np.expm1(-0.2 * np.log(rng.uniform(size=n))) / 0.2
    elif isinstance(family, GeneralizedGammaLSS):
        q, sigma = 0.7, 0.4
        y = np.exp(location + sigma / q * np.log(rng.gamma(1 / q**2, q**2, n)))
    elif isinstance(family, (TwoPieceNormalLSS, TwoPieceLogNormalLSS)):
        negative = rng.uniform(size=n) < 0.4
        noise = np.abs(rng.normal(size=n)) * np.where(negative, -0.8, 1.2)
        y = location + 0.4 * noise
        if isinstance(family, TwoPieceLogNormalLSS):
            y = np.exp(y)
    else:
        y = np.exp(location + rng.normal(0, 0.4, n))
    names = tuple(parameter.name for parameter in family.parameters)
    predictors = tuple(
        Predictor(name, {"x": Spline(kind="cr", n_knots=4)} if i == 0 else {})
        for i, name in enumerate(names)
    )
    offsets = {name: 0.005 * np.sin(np.arange(n) + i) for i, name in enumerate(names)}
    models = [
        SuperLSS(
            family=family,
            predictors=predictors,
            discrete=discrete,
            n_bins=32,
            weight_semantics="frequency",
        ).fit(
            frame, y, sample_weight=weights, offsets=offsets, lambdas={f"{names[0]}:x#wiggle": 1.0}
        )
        for discrete in (False, True)
    ]
    _assert_parity(*models, frame, offsets)


def test_public_discrete_dispatches_compressed_signed_chunks(monkeypatch) -> None:
    frame, y, weights, offsets, family, predictors = _fixture("gaussian", "prior")
    monkeypatch.setattr(chunking, "AUTO_CHUNK_MEMORY_BYTES", 8192)
    original_moments = MatrixExecutionPlan.moments
    original_cross = CrossMatrixExecutionPlan.cross_moment
    original_geometry = chunking.assemble_chunked_geometry
    compressed_rows = []
    cross_signs = []
    assembling = False

    def geometry(*args, **kwargs):
        nonlocal assembling
        assembling = True
        try:
            return original_geometry(*args, **kwargs)
        finally:
            assembling = False

    def moments(self, *args, **kwargs):
        if assembling and any(
            type(group).__name__.startswith("Discretized") for group in self.group_matrices
        ):
            compressed_rows.append(self.n)
        return original_moments(self, *args, **kwargs)

    def cross(self, weights, *args, **kwargs):
        if assembling:
            cross_signs.extend(np.sign(weights).tolist())
        return original_cross(self, weights, *args, **kwargs)

    monkeypatch.setattr(MatrixExecutionPlan, "moments", moments)
    monkeypatch.setattr(CrossMatrixExecutionPlan, "cross_moment", cross)
    monkeypatch.setattr(chunking, "assemble_chunked_geometry", geometry)
    model = SuperLSS(family=family, predictors=predictors, discrete=True, n_bins=32).fit_reml(
        frame,
        y,
        sample_weight=weights,
        offsets=offsets,
        outer="efs+newton",
    )
    telemetry = model.training_telemetry()
    assert telemetry.execution_backend_identifier == "distributional-chunked-v1"
    assert 0 < telemetry.resolved_chunk_size < np.count_nonzero(weights)
    assert compressed_rows and max(compressed_rows) <= telemetry.resolved_chunk_size
    assert {-1.0, 1.0} <= set(cross_signs)


def test_per_term_discrete_preserves_explicit_dense_execution_policy() -> None:
    frame, y, weights, offsets, family, _ = _fixture("gaussian", "prior")
    model = SuperLSS(
        family=family,
        predictors=(
            Predictor("location", {"x": Spline(kind="cr", n_knots=4, discrete=True)}),
            Predictor("scale", {}),
        ),
        discrete=False,
    ).fit(frame, y, sample_weight=weights, offsets=offsets, lambdas={"location:x#wiggle": 1.0})
    telemetry = model.training_telemetry()
    assert not telemetry.discrete
    assert telemetry.execution_backend_identifier == "distributional-dense-v1"
    assert telemetry.resolved_chunk_size is None
