from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import superglm.distributional as distributional
import superglm.distributional.families as distributional_families
import superglm.distributional.fit_state as fit_state_module
from superglm import Spline
from superglm.distributional import GammaLS, Predictor
from superglm.distributional.families.negative_binomial import NegativeBinomialLS
from superglm.distributional.result import DistributionalEFSConfig
from superglm.features import RandomEffect
from tests.bound_predictor_fixtures import model_from_templates


@pytest.mark.parametrize(
    "name",
    [
        "GaussianLS",
        "GammaLS",
        "GeneralizedGammaLSS",
        "GeneralizedParetoLSS",
        "LogNormalLS",
        "NegativeBinomialLS",
        "TweedieLSS",
        "TwoPieceLogNormalLSS",
        "TwoPieceNormalLSS",
        "Predictor",
    ],
)
def test_review_supported_constructors_import_from_root(name):
    import superglm

    assert getattr(superglm, name) is getattr(distributional, name)
    assert name in superglm.__all__


def test_negative_binomial_lss_is_exported_from_both_public_family_namespaces() -> None:
    """Kills omitting either supported public import path for the NB2 family."""
    assert distributional.NegativeBinomialLS is NegativeBinomialLS
    assert distributional_families.NegativeBinomialLS is NegativeBinomialLS


def test_a_non_converged_fit_warns_and_its_summary_says_so(monkeypatch) -> None:
    """Returned, not refused, and never silent (the GPD repro of register A3).

    A strict fit stopped at its iteration cap is a non-converged fit: it warns,
    and every summary row says the numbers are an iterate's. A negative term
    EDF -- what the non-converged GPD shape smooth reported, -2.196 with an
    empty note -- is called out on its row. Mutation check: master warned
    nothing and left every note empty.
    """
    from superglm import ConvergenceWarning, GaussianLS, SuperLSS, s
    from superglm.distributional import terms as terms_module

    rng = np.random.default_rng(2)
    n = 600
    frame = pd.DataFrame({"x": rng.uniform(-1.0, 1.0, n)})
    response = np.sin(2.0 * frame["x"].to_numpy()) + rng.normal(0.0, 0.3, n)
    family = GaussianLS()
    model = SuperLSS(family, family.location(s("x", kind="cr", k=6)), family.scale())
    with pytest.warns(ConvergenceWarning, match="did not converge"):
        model.fit_reml(frame, response, max_reml_iter=1, practical_reml=False)
    assert model.result_.converged is False
    notes = model.summary()["note"]
    assert all("fit not converged" in note for note in notes)

    real_outcome = terms_module._term_test_from_covariance

    def negative_edf(prepared, matrix):
        return replace(real_outcome(prepared, matrix), edf=-2.196)

    monkeypatch.setattr(terms_module, "_term_test_from_covariance", negative_edf)
    table = model.summary()
    term_notes = table.loc[table["term"] != "(intercept)", "note"]
    assert all("negative EDF (-2.2) is not interpretable" in note for note in term_notes)


def test_a_fixed_lambda_fit_discloses_non_convergence_without_a_warning() -> None:
    """``ConvergenceWarning`` is ``fit_reml``'s contract; ``fit`` holds the
    smoothing fixed and discloses a coefficient loop stopped at
    ``max_inner_iter`` through ``result_.converged`` and every summary row, as
    before this warning existed. Mutation check: on af53c8d4 the warning was
    emitted from the code ``fit`` and ``fit_reml`` share, so ``fit`` warned too.
    """
    import warnings

    from superglm import ConvergenceWarning, GaussianLS, SuperLSS, s

    rng = np.random.default_rng(2)
    n = 600
    frame = pd.DataFrame({"x": rng.uniform(-1.0, 1.0, n)})
    response = np.sin(2.0 * frame["x"].to_numpy()) + rng.normal(0.0, 0.3, n)
    family = GaussianLS()
    model = SuperLSS(family, family.location(s("x", kind="cr", k=6)), family.scale())
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(frame, response, lambdas={"location:x#wiggle": 1.0}, max_inner_iter=1)
    assert model.result_.converged is False
    assert all("fit not converged" in note for note in model.summary()["note"])


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

    model = model_from_templates(
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
