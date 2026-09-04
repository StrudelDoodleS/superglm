from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import superglm.distributional as distributional
import superglm.distributional.families as distributional_families
import superglm.distributional.fit_state as fit_state_module
from superglm import Spline, SuperLSS
from superglm.distributional import GammaLS, Predictor
from superglm.distributional.families.negative_binomial import NegativeBinomialLS
from superglm.distributional.result import DistributionalEFSConfig
from superglm.features import RandomEffect


def test_negative_binomial_lss_is_exported_from_both_public_family_namespaces() -> None:
    """Kills omitting either supported public import path for the NB2 family."""
    assert distributional.NegativeBinomialLS is NegativeBinomialLS
    assert distributional_families.NegativeBinomialLS is NegativeBinomialLS


def test_public_gamma_reml_exposes_an_exact_face_at_the_default_lambda_cap(
    monkeypatch,
) -> None:
    """Kills accepting an exact face only inside private EFS state."""
    x_unique = np.linspace(-1.0, 1.0, 24)
    groups = np.array(["a", "b", "c"])
    residual_factors = np.array([0.62, 0.84, 1.16, 1.38])
    x = np.repeat(x_unique, len(groups) * len(residual_factors))
    group = np.tile(np.repeat(groups, len(residual_factors)), len(x_unique))
    factors = np.tile(np.tile(residual_factors, len(groups)), len(x_unique))
    mean = np.exp(0.45 + 0.48 * np.sin(np.pi * x) + 0.18 * x)
    response = mean * factors
    weights = 0.65 + 0.7 * (x + 1.0) / 2.0
    frame = pd.DataFrame({"x": x, "group": group})
    default_cap = DistributionalEFSConfig().maximum_lambda
    assert default_cap == 1.0e10
    null_configs = []
    real_null_fit = fit_state_module.fit_joint_null_model

    def capture_null_config(*args, **kwargs):
        null_configs.append(kwargs.get("config"))
        return real_null_fit(*args, **kwargs)

    monkeypatch.setattr(fit_state_module, "fit_joint_null_model", capture_null_config)

    model = SuperLSS(
        family=GammaLS(),
        predictors=(
            Predictor("mean", {"x": Spline(kind="cr", n_knots=5)}),
            Predictor("scale", {"group": RandomEffect()}),
        ),
    ).fit_reml(
        frame,
        response,
        sample_weight=weights,
        lambdas={"mean:x#wiggle": 0.5, "scale:group#wiggle": default_cap},
        max_reml_iter=120,
        reml_tol=1.0e-3,
        max_inner_iter=150,
        inner_tol=1.0e-9,
    )

    fitted = model._require_fitted()
    assert fitted.smoothing is not None
    assert fitted.fit_state.requested_solver_config.coefficient_curvature == "observed"
    assert fitted.fit_state.requested_solver_config.max_iterations == 150
    assert fitted.fit_state.requested_solver_config.tolerance == 1.0e-9
    assert fitted.result.config.coefficient_curvature == "observed"
    assert len(null_configs) == 1
    assert null_configs[0].coefficient_curvature == "observed"
    assert model.training_telemetry().curvature_policy == "observed"
    assert fitted.smoothing.config.maximum_lambda == default_cap
    assert fitted.smoothing.converged is True
    assert fitted.smoothing.matched_certified is False
    with pytest.raises(
        RuntimeError,
        match="exact coefficient face is numerically supported but not certified",
    ):
        fitted.smoothing.assert_matched_certified()
    evidence = fitted.smoothing.terminal_endpoint_directions["scale:group#wiggle"]
    assert evidence.authority_identifier == "analytic-observed-curvature-direction/v1"
    assert evidence.decision == "endpoint"
    assert evidence.lower_bound > 0.0
    face_events = tuple(
        item
        for item in fitted.smoothing.history
        if item.activated_face_components or item.revalidated_face_components
    )
    assert len(face_events) == 2
    assert face_events[0].activated_face_components == ("scale:group#wiggle",)
    assert face_events[1].revalidated_face_components == ("scale:group#wiggle",)
    assert fitted.smoothing.history[-1] is face_events[1]
    assert all(len(item.coefficient_fit_indices) == 2 for item in face_events)
    assert len(fitted.smoothing.coefficient_fits) == 1 + sum(
        len(item.coefficient_fit_indices) for item in fitted.smoothing.history
    )
    activation_position = fitted.smoothing.history.index(face_events[0])
    for item in fitted.smoothing.history[activation_position + 1 :]:
        for fit_index in item.coefficient_fit_indices:
            config = fitted.smoothing.coefficient_fits[fit_index].config
            assert config.coefficient_curvature == "observed"
            assert config.tolerance == 1.0e-12
            assert config.newton_decrement_tolerance is None
    terminal_recheck = face_events[1]
    assert terminal_recheck.accepted_fit_index is not None
    source = fitted.smoothing.coefficient_fits[terminal_recheck.source_fit_index]
    endpoint = fitted.smoothing.coefficient_fits[terminal_recheck.accepted_fit_index]
    assert endpoint.score_relative <= endpoint.config.tolerance
    np.testing.assert_array_equal(endpoint.coefficients, source.coefficients)
    endpoint_face = endpoint.coefficient_face
    assert endpoint_face is not None
    moved_coefficients = np.array(endpoint.coefficients, copy=True)
    moved_coefficients += 1.0e-6 * endpoint_face.null_basis[:, 0]
    moved_fits = list(fitted.smoothing.coefficient_fits)
    moved_fits[terminal_recheck.accepted_fit_index] = replace(
        endpoint,
        coefficients=moved_coefficients,
    )
    with pytest.raises(ValueError, match="canonical endpoint state"):
        replace(fitted.smoothing, coefficient_fits=tuple(moved_fits))
    assert model.exact_face_components_ == ("scale:group#wiggle",)
    assert model.result_.exact_face_components == model.exact_face_components_
    assert fitted.fit_state.exact_face_components == model.exact_face_components_
    assert model.smoothing_parameters_["scale:group#wiggle"] == default_cap

    constrained = fitted.layout.term_slices["scale:group"]
    covariance = model.covariance_
    assert np.all(np.isfinite(covariance))
    np.testing.assert_array_equal(
        covariance[constrained, :],
        np.zeros((constrained.stop - constrained.start, covariance.shape[1])),
    )
    np.testing.assert_array_equal(
        covariance[:, constrained],
        np.zeros((covariance.shape[0], constrained.stop - constrained.start)),
    )
    assert model.result_.term_edf["scale:group"] == 0.0
    parameters = model.predict_parameters(frame).to_numpy()
    assert np.all(np.isfinite(parameters))
    assert np.all(parameters > 0.0)
