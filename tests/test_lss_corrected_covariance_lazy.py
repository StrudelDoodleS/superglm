"""Authenticated on-demand smoothing curvature for corrected LSS covariance."""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Callable

import matplotlib
import numpy as np
import pandas as pd
import pytest

from tests.bound_predictor_fixtures import model_from_templates

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from superglm import Spline, SuperLSS
from superglm.diagnostics.convergence import ConvergenceWarning
from superglm.distributional import Predictor
from superglm.distributional import api as api_module
from superglm.distributional import posterior as posterior_module
from superglm.distributional import terms as terms_module
from superglm.distributional.families.gaussian import GaussianLS
from superglm.distributional.family import COMPLETE_OBSERVATION
from superglm.distributional.model import DenseDistributionalModel
from superglm.distributional.posterior import posterior_covariance
from superglm.distributional.serialization import distributional_manifest
from superglm.distributional.smoothing.derivatives import (
    LamlDerivativeError,
    LamlDerivatives,
)
from superglm.distributional.smoothing.derivatives import (
    laml_derivatives as real_laml_derivatives,
)
from superglm.distributional.solver.assembly import dense_predictor_matrices
from superglm.distributional.terms import summary_table, term_effect, term_test


@pytest.fixture(scope="module")
def stationary_case() -> tuple[DenseDistributionalModel, pd.DataFrame]:
    """A retained stationary fit whose Newton handoff stops without a Hessian pass."""
    rng = np.random.default_rng(7351)
    x = rng.uniform(-1.0, 1.0, 320)
    location = 0.6 * np.sin(2.4 * x)
    scale = np.exp(-1.0 + 0.5 * np.cos(1.8 * x))
    frame = pd.DataFrame({"x": x})
    response = location + scale * rng.standard_normal(len(x))
    fitted = (
        model_from_templates(
            family=GaussianLS(),
            predictors=[
                Predictor("location", {"x": Spline("cr", k=7)}),
                Predictor("scale", {"x": Spline("cr", k=5)}),
            ],
        )
        .fit_reml(frame, response, outer="efs+newton")
        ._require_fitted()
    )
    assert fitted.smoothing is not None
    assert fitted.smoothing.convergence_reason == "stationary"
    assert fitted.smoothing.smoothing_hessian is None
    assert fitted.fit_state.retained_rows is not None
    return fitted, frame


def _terminal_derivative_arguments(fitted: DenseDistributionalModel) -> tuple:
    rows = fitted.fit_state.retained_rows
    assert rows is not None and fitted.smoothing is not None
    plan = fitted.family.bind_likelihood(
        rows.response,
        rows.likelihood_weights,
        COMPLETE_OBSERVATION,
    )
    return (fitted.family, fitted.layout, rows.response, plan), {
        "lambdas": fitted.lambdas,
        "fit": fitted.result,
        "dense_matrices": dense_predictor_matrices(fitted.layout),
        "step": fitted.smoothing.config.derivative_step,
    }


def _terminal_derivatives(fitted: DenseDistributionalModel) -> LamlDerivatives:
    arguments, options = _terminal_derivative_arguments(fitted)
    return real_laml_derivatives(*arguments, **options, want_hessian=True)


def _published_model(
    fitted: DenseDistributionalModel,
    derivatives: LamlDerivatives,
    *,
    hessian: np.ndarray | None = None,
    certificate: np.ndarray | None = None,
) -> DenseDistributionalModel:
    assert fitted.smoothing is not None
    resolved_hessian = derivatives.hessian if hessian is None else hessian
    resolved_certificate = derivatives.hessian_certificate if certificate is None else certificate
    assert resolved_hessian is not None and resolved_certificate is not None
    smoothing = dataclasses.replace(
        fitted.smoothing,
        smoothing_hessian=resolved_hessian,
        smoothing_hessian_certificate=resolved_certificate,
    )
    state = dataclasses.replace(fitted.fit_state, smoothing=smoothing)
    return DenseDistributionalModel(family=fitted.family, _fit_state=state)


@pytest.fixture(scope="module")
def published_case(stationary_case) -> tuple[DenseDistributionalModel, pd.DataFrame]:
    fitted, frame = stationary_case
    return _published_model(fitted, _terminal_derivatives(fitted)), frame


def _with_smoothing_stop(
    fitted: DenseDistributionalModel,
    *,
    reason: str,
    converged: bool,
) -> DenseDistributionalModel:
    assert fitted.smoothing is not None
    smoothing = dataclasses.replace(fitted.smoothing)
    object.__setattr__(smoothing, "convergence_reason", reason)
    object.__setattr__(smoothing, "converged", converged)
    state = dataclasses.replace(fitted.fit_state, smoothing=smoothing)
    return DenseDistributionalModel(family=fitted.family, _fit_state=state)


def _with_terminal_result(fitted: DenseDistributionalModel, terminal) -> DenseDistributionalModel:
    """Build an adversarial state while keeping its terminal-result identities coherent."""
    assert fitted.smoothing is not None
    smoothing = dataclasses.replace(fitted.smoothing)
    fits = list(smoothing.coefficient_fits)
    fits[smoothing.terminal_fit_index] = terminal
    object.__setattr__(smoothing, "coefficient_fits", tuple(fits))
    state = dataclasses.replace(fitted.fit_state)
    object.__setattr__(state, "solver_result", terminal)
    object.__setattr__(state, "smoothing", smoothing)
    return DenseDistributionalModel(family=fitted.family, _fit_state=state)


def _public_model(
    fitted: DenseDistributionalModel,
    frame: pd.DataFrame,
) -> SuperLSS:
    model = model_from_templates(
        family=fitted.family,
        predictors=fitted.fit_state.predictor_templates,
    )
    model._model = fitted
    model._training_frame = frame
    return model


def _install_derivative_transform(
    monkeypatch: pytest.MonkeyPatch,
    transform: Callable[[LamlDerivatives], LamlDerivatives],
) -> None:
    def replay(*args, **kwargs) -> LamlDerivatives:
        return transform(real_laml_derivatives(*args, **kwargs))

    monkeypatch.setattr(posterior_module, "laml_derivatives", replay, raising=False)


def _install_derivative_spy(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    calls: list[bool] = []

    def spy(*args, **kwargs) -> LamlDerivatives:
        calls.append(bool(kwargs.get("want_hessian", True)))
        return real_laml_derivatives(*args, **kwargs)

    monkeypatch.setattr(posterior_module, "laml_derivatives", spy, raising=False)
    return calls


def _install_covariance_spy(monkeypatch: pytest.MonkeyPatch, module) -> list[str]:
    calls: list[str] = []

    def spy(fitted, *, kind="fixed") -> np.ndarray:
        calls.append(kind)
        return posterior_covariance(fitted, kind=kind)

    monkeypatch.setattr(module, "posterior_covariance", spy)
    return calls


def test_retained_stationary_fit_replays_exact_hessian_without_mutation(stationary_case) -> None:
    fitted, _ = stationary_case
    assert fitted.smoothing is not None
    derivatives = _terminal_derivatives(fitted)
    published = _published_model(fitted, derivatives)
    fit_state = fitted.fit_state
    smoothing = fitted.smoothing
    coefficients = np.array(fitted.coefficients, copy=True)
    manifest = distributional_manifest(fitted)

    expected = posterior_covariance(published, kind="corrected")
    actual = posterior_covariance(fitted, kind="corrected")

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(fitted.coefficients, coefficients)
    assert fitted.fit_state is fit_state
    assert fitted.smoothing is smoothing
    assert fitted.smoothing.smoothing_hessian is None
    assert distributional_manifest(fitted) == manifest


def test_compact_fit_refuses_lazy_corrected_covariance(stationary_case) -> None:
    fitted, _ = stationary_case
    compact_state = dataclasses.replace(fitted.fit_state, retained_rows=None)
    compact = DenseDistributionalModel(family=fitted.family, _fit_state=compact_state)

    with pytest.raises(RuntimeError, match="retained training rows"):
        posterior_covariance(compact, kind="corrected")


@pytest.mark.parametrize(
    ("reason", "converged"),
    [("practical_plateau", True), ("gradient_unresolved", False)],
)
def test_nonstationary_or_failed_stop_refuses_replay(
    stationary_case,
    reason: str,
    converged: bool,
) -> None:
    fitted, _ = stationary_case
    altered = _with_smoothing_stop(fitted, reason=reason, converged=converged)

    with pytest.raises(RuntimeError, match="stationary smoothing result"):
        posterior_covariance(altered, kind="corrected")


def test_failed_stop_without_terminal_gradient_refuses_before_row_replay(stationary_case) -> None:
    fitted, _ = stationary_case
    altered = _with_smoothing_stop(fitted, reason="efs_not_converged", converged=False)
    assert altered.smoothing is not None
    object.__setattr__(altered.smoothing, "terminal_gradient", None)
    object.__setattr__(altered.smoothing, "terminal_gradient_certificate", None)

    with pytest.raises(RuntimeError, match="stationary smoothing result"):
        posterior_covariance(altered, kind="corrected")


@pytest.mark.parametrize("source", ["fit_state", "terminal_result"])
def test_plan_identifier_must_match_fit_state_and_terminal_result(
    stationary_case,
    source: str,
) -> None:
    fitted, _ = stationary_case
    if source == "fit_state":
        state = dataclasses.replace(fitted.fit_state)
        object.__setattr__(state, "family_likelihood_plan_identifier", "forged-plan")
        altered = DenseDistributionalModel(family=fitted.family, _fit_state=state)
    else:
        terminal = dataclasses.replace(
            fitted.result,
            family_likelihood_plan_identifier="forged-plan",
        )
        altered = _with_terminal_result(fitted, terminal)

    with pytest.raises(RuntimeError, match="likelihood plan identifier"):
        posterior_covariance(altered, kind="corrected")


def test_rebound_plan_identifier_must_match_both_stored_authorities(stationary_case) -> None:
    fitted, _ = stationary_case
    terminal = dataclasses.replace(
        fitted.result,
        family_likelihood_plan_identifier="forged-plan",
    )
    altered = _with_terminal_result(fitted, terminal)
    object.__setattr__(
        altered.fit_state,
        "family_likelihood_plan_identifier",
        "forged-plan",
    )

    with pytest.raises(RuntimeError, match="replay produced a different likelihood plan"):
        posterior_covariance(altered, kind="corrected")


def test_retained_rows_must_preserve_root_weight_provenance(stationary_case) -> None:
    fitted, _ = stationary_case
    state = dataclasses.replace(fitted.fit_state)
    forged = dataclasses.replace(state.weight_provenance, root_digest="forged-root")
    object.__setattr__(state, "weight_provenance", forged)
    altered = DenseDistributionalModel(family=fitted.family, _fit_state=state)

    with pytest.raises(RuntimeError, match="root weight provenance"):
        posterior_covariance(altered, kind="corrected")


def test_terminal_result_identity_must_match_the_smoothing_authority(stationary_case) -> None:
    fitted, _ = stationary_case
    state = dataclasses.replace(fitted.fit_state)
    object.__setattr__(state, "solver_result", dataclasses.replace(fitted.result))
    altered = DenseDistributionalModel(family=fitted.family, _fit_state=state)

    with pytest.raises(RuntimeError, match="terminal-result provenance"):
        posterior_covariance(altered, kind="corrected")


def test_terminal_lambda_mapping_must_match_the_smoothing_authority(stationary_case) -> None:
    fitted, _ = stationary_case
    state = dataclasses.replace(fitted.fit_state)
    lambdas = dict(state.lambdas)
    name = next(iter(lambdas))
    lambdas[name] *= 1.01
    object.__setattr__(state, "lambdas", lambdas)
    altered = DenseDistributionalModel(family=fitted.family, _fit_state=state)

    with pytest.raises(RuntimeError, match="smoothing provenance"):
        posterior_covariance(altered, kind="corrected")


def test_terminal_face_must_match_the_fit_state_authority(stationary_case) -> None:
    fitted, _ = stationary_case
    assert fitted.result.coefficient_face is None
    state = dataclasses.replace(fitted.fit_state)
    object.__setattr__(state, "exact_face_components", (fitted.layout.penalty_names[0],))
    altered = DenseDistributionalModel(family=fitted.family, _fit_state=state)

    with pytest.raises(RuntimeError, match="active-face provenance"):
        posterior_covariance(altered, kind="corrected")


@pytest.mark.parametrize(
    ("source", "disclosure"),
    [
        ("fisher", "expected \\(Fisher\\) information"),
        ("nearest_semidefinite", "nearest positive semidefinite approximation"),
    ],
)
def test_fallback_terminal_curvature_replays_with_its_substitute(
    stationary_case,
    monkeypatch: pytest.MonkeyPatch,
    source: str,
    disclosure: str,
) -> None:
    """A substitute terminal curvature is replayed with its own rows and disclosed, not refused."""
    fitted, _ = stationary_case
    curvature = dataclasses.replace(
        fitted.result.terminal_curvature,
        actual_source=source,
        reason="material_indefiniteness_after_retry",
        fallback_count=1,
    )
    terminal = dataclasses.replace(fitted.result, terminal_curvature=curvature)
    altered = _with_terminal_result(fitted, terminal)
    sources: list[str] = []

    def spy(*args, **kwargs) -> LamlDerivatives:
        sources.append(kwargs["curvature"])
        return real_laml_derivatives(*args, **kwargs)

    monkeypatch.setattr(posterior_module, "laml_derivatives", spy, raising=False)
    with pytest.warns(ConvergenceWarning, match=disclosure):
        corrected = posterior_covariance(altered, kind="corrected")

    assert sources == ["terminal"]
    assert np.all(np.isfinite(corrected))
    # The smoothing search's own derivative call still refuses the substitute.
    arguments, options = _terminal_derivative_arguments(altered)
    with pytest.raises(LamlDerivativeError, match="observed terminal curvature"):
        real_laml_derivatives(*arguments, **options, want_hessian=True)


@pytest.mark.parametrize("mutation", ["solver-policy", "requested-source", "fallback-count"])
def test_observed_terminal_curvature_needs_no_substitute(stationary_case, mutation: str) -> None:
    """The terminal policy always requests observed curvature; only the accepted source matters.

    A Fisher-stepped fit whose observed curvature was accepted gets the same
    corrected covariance, without a disclosure.
    """
    fitted, _ = stationary_case
    if mutation == "solver-policy":
        config = dataclasses.replace(fitted.result.config, coefficient_curvature="fisher")
        terminal = dataclasses.replace(fitted.result, config=config)
    else:
        changes = (
            {"requested_source": "fisher"}
            if mutation == "requested-source"
            else {"reason": "adversarial fallback", "fallback_count": 1}
        )
        curvature = dataclasses.replace(fitted.result.terminal_curvature, **changes)
        terminal = dataclasses.replace(fitted.result, terminal_curvature=curvature)
    altered = _with_terminal_result(fitted, terminal)

    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        actual = posterior_covariance(altered, kind="corrected")
    np.testing.assert_array_equal(actual, posterior_covariance(fitted, kind="corrected"))


@pytest.mark.parametrize(
    ("reason", "converged"),
    [
        ("practical_plateau", True),
        ("stationary", False),
        ("gradient_unresolved", False),
        ("objective_rejected", False),
    ],
    ids=["reason", "converged", "gradient-unresolved", "objective-rejected"],
)
def test_published_hessian_requires_a_converged_stationary_stop(
    published_case,
    reason: str,
    converged: bool,
) -> None:
    published, _ = published_case
    altered = _with_smoothing_stop(published, reason=reason, converged=converged)
    assert altered.smoothing is not None
    assert altered.smoothing.smoothing_hessian is not None

    with pytest.raises(RuntimeError, match="stationary smoothing result"):
        posterior_covariance(altered, kind="corrected")


def _published_terminal_identity_mismatch(
    fitted: DenseDistributionalModel,
) -> DenseDistributionalModel:
    state = dataclasses.replace(fitted.fit_state)
    object.__setattr__(state, "solver_result", dataclasses.replace(fitted.result))
    return DenseDistributionalModel(family=fitted.family, _fit_state=state)


def _published_plan_mismatch(fitted: DenseDistributionalModel) -> DenseDistributionalModel:
    state = dataclasses.replace(fitted.fit_state)
    object.__setattr__(state, "family_likelihood_plan_identifier", "forged-plan")
    return DenseDistributionalModel(family=fitted.family, _fit_state=state)


def _published_lambda_mismatch(fitted: DenseDistributionalModel) -> DenseDistributionalModel:
    state = dataclasses.replace(fitted.fit_state)
    lambdas = dict(state.lambdas)
    name = next(iter(lambdas))
    lambdas[name] *= 1.01
    object.__setattr__(state, "lambdas", lambdas)
    return DenseDistributionalModel(family=fitted.family, _fit_state=state)


def _published_face_mismatch(fitted: DenseDistributionalModel) -> DenseDistributionalModel:
    assert fitted.result.coefficient_face is None
    state = dataclasses.replace(fitted.fit_state)
    object.__setattr__(state, "exact_face_components", (fitted.layout.penalty_names[0],))
    return DenseDistributionalModel(family=fitted.family, _fit_state=state)


def _published_solver_policy_mismatch(
    fitted: DenseDistributionalModel,
) -> DenseDistributionalModel:
    config = dataclasses.replace(fitted.result.config, coefficient_curvature="fisher")
    return _with_terminal_result(fitted, dataclasses.replace(fitted.result, config=config))


def _published_requested_source_mismatch(
    fitted: DenseDistributionalModel,
) -> DenseDistributionalModel:
    curvature = dataclasses.replace(
        fitted.result.terminal_curvature,
        requested_source="fisher",
    )
    return _with_terminal_result(
        fitted,
        dataclasses.replace(fitted.result, terminal_curvature=curvature),
    )


def _published_actual_source_mismatch(
    fitted: DenseDistributionalModel,
) -> DenseDistributionalModel:
    curvature = dataclasses.replace(
        fitted.result.terminal_curvature,
        actual_source="fisher",
    )
    return _with_terminal_result(
        fitted,
        dataclasses.replace(fitted.result, terminal_curvature=curvature),
    )


def _published_fallback_mismatch(fitted: DenseDistributionalModel) -> DenseDistributionalModel:
    curvature = dataclasses.replace(
        fitted.result.terminal_curvature,
        reason="adversarial fallback",
        fallback_count=1,
    )
    return _with_terminal_result(
        fitted,
        dataclasses.replace(fitted.result, terminal_curvature=curvature),
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (_published_terminal_identity_mismatch, "terminal-result provenance"),
        (_published_plan_mismatch, "likelihood plan identifier"),
        (_published_lambda_mismatch, "smoothing provenance"),
        (_published_face_mismatch, "active-face provenance"),
    ],
    ids=[
        "accepted-result-identity",
        "plan",
        "lambda",
        "face",
    ],
)
def test_published_hessian_authenticates_every_terminal_authority(
    published_case,
    mutation: Callable[[DenseDistributionalModel], DenseDistributionalModel],
    message: str,
) -> None:
    published, _ = published_case
    altered = mutation(published)
    assert altered.smoothing is not None
    assert altered.smoothing.smoothing_hessian is not None

    with pytest.raises(RuntimeError, match=message):
        posterior_covariance(altered, kind="corrected")


@pytest.mark.parametrize(
    "mutation",
    [
        _published_solver_policy_mismatch,
        _published_requested_source_mismatch,
        _published_fallback_mismatch,
    ],
    ids=["solver-policy", "requested-source", "fallback"],
)
def test_published_hessian_serves_any_observed_terminal_curvature(
    published_case,
    mutation: Callable[[DenseDistributionalModel], DenseDistributionalModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    published, _ = published_case
    calls = _install_derivative_spy(monkeypatch)

    actual = posterior_covariance(mutation(published), kind="corrected")

    assert calls == []
    np.testing.assert_array_equal(actual, posterior_covariance(published, kind="corrected"))


def test_published_hessian_is_not_used_for_a_substitute_curvature(
    published_case,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The published Hessian was formed with observed curvature, so a substitute replays."""
    published, _ = published_case
    calls = _install_derivative_spy(monkeypatch)

    with pytest.warns(ConvergenceWarning, match="first-order correction"):
        posterior_covariance(_published_actual_source_mismatch(published), kind="corrected")

    assert calls == [True]


def _forged_names(derivatives: LamlDerivatives) -> LamlDerivatives:
    return dataclasses.replace(
        derivatives,
        names=tuple(f"forged:{name}" for name in derivatives.names),
    )


def _forged_gradient(derivatives: LamlDerivatives) -> LamlDerivatives:
    gradient = np.array(derivatives.gradient, copy=True)
    gradient[0] += 1.0
    return dataclasses.replace(derivatives, gradient=gradient)


@pytest.mark.parametrize(
    ("transform", "message"),
    [
        (_forged_names, "ordered smoothing names"),
        (_forged_gradient, "terminal gradient"),
    ],
)
def test_replayed_derivatives_must_authenticate_terminal_evidence(
    stationary_case,
    monkeypatch: pytest.MonkeyPatch,
    transform: Callable[[LamlDerivatives], LamlDerivatives],
    message: str,
) -> None:
    fitted, _ = stationary_case
    _install_derivative_transform(monkeypatch, transform)

    with pytest.raises(RuntimeError, match=message):
        posterior_covariance(fitted, kind="corrected")


@pytest.mark.parametrize("coordinate", ["rank", "source", "face"])
def test_replayed_derivative_provenance_coordinates_are_authenticated_independently(
    stationary_case,
    monkeypatch: pytest.MonkeyPatch,
    coordinate: str,
) -> None:
    fitted, _ = stationary_case

    def alter_one_coordinate(derivatives: LamlDerivatives) -> LamlDerivatives:
        provenance = list(derivatives.provenance)
        if coordinate == "rank":
            provenance[1] = int(provenance[1]) + 1
        elif coordinate == "source":
            provenance[2] = "fisher"
        else:
            provenance[3] = ("forged-face",)
        return dataclasses.replace(derivatives, provenance=tuple(provenance))

    _install_derivative_transform(monkeypatch, alter_one_coordinate)

    with pytest.raises(RuntimeError, match="derivative provenance"):
        posterior_covariance(fitted, kind="corrected")


def test_replayed_gradient_agrees_within_the_sum_of_certificates(
    stationary_case,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fitted, _ = stationary_case

    def within_certificate(derivatives: LamlDerivatives) -> LamlDerivatives:
        gradient = np.array(derivatives.gradient, copy=True)
        certificate = np.array(derivatives.gradient_certificate, copy=True)
        gradient[0] += 5.0e-5
        certificate[0] += 1.0e-4
        return dataclasses.replace(
            derivatives,
            gradient=gradient,
            gradient_certificate=certificate,
        )

    _install_derivative_transform(monkeypatch, within_certificate)
    corrected = posterior_covariance(fitted, kind="corrected")
    assert np.all(np.isfinite(corrected))


def _missing_hessian(derivatives: LamlDerivatives) -> LamlDerivatives:
    return dataclasses.replace(derivatives, hessian=None, hessian_certificate=None)


def _indefinite_hessian(derivatives: LamlDerivatives) -> LamlDerivatives:
    assert derivatives.hessian is not None
    hessian = np.array(derivatives.hessian, copy=True)
    hessian[0, 0] = -max(1.0, abs(float(hessian[0, 0])))
    return dataclasses.replace(derivatives, hessian=hessian)


def _untrusted_certificate(derivatives: LamlDerivatives) -> LamlDerivatives:
    assert derivatives.hessian is not None and derivatives.hessian_certificate is not None
    certificate = np.full_like(
        derivatives.hessian_certificate,
        np.max(np.abs(derivatives.hessian)) + 1.0,
    )
    return dataclasses.replace(derivatives, hessian_certificate=certificate)


@pytest.mark.parametrize(
    ("transform", "message"),
    [
        (_missing_hessian, "replay returned no smoothing Hessian"),
        (_indefinite_hessian, "positive definite"),
        (_untrusted_certificate, "Hessian certificate"),
    ],
)
def test_replayed_hessian_and_certificate_must_be_trusted(
    stationary_case,
    monkeypatch: pytest.MonkeyPatch,
    transform: Callable[[LamlDerivatives], LamlDerivatives],
    message: str,
) -> None:
    fitted, _ = stationary_case
    _install_derivative_transform(monkeypatch, transform)

    with pytest.raises(RuntimeError, match=message):
        posterior_covariance(fitted, kind="corrected")


def test_derivative_replay_failure_is_a_clear_refusal(
    stationary_case,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fitted, _ = stationary_case

    def fail(*args, **kwargs):
        raise LamlDerivativeError("adversarial derivative failure")

    monkeypatch.setattr(posterior_module, "laml_derivatives", fail, raising=False)
    with pytest.raises(RuntimeError, match="smoothing derivative replay failed"):
        posterior_covariance(fitted, kind="corrected")


@pytest.mark.parametrize("operation", ["effect", "test"])
def test_unknown_term_is_refused_before_resolving_lazy_covariance(
    stationary_case,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    fitted, frame = stationary_case
    calls = _install_covariance_spy(monkeypatch, terms_module)

    with pytest.raises(KeyError, match="unknown term 'location:nope'"):
        if operation == "effect":
            term_effect(fitted, frame, "location", "nope", covariance="corrected")
        else:
            term_test(fitted, frame, "location", "nope", covariance="corrected")

    assert calls == []


@pytest.mark.parametrize("operation", ["effect", "test"])
def test_compact_unknown_term_reports_the_term_error_before_retained_rows(
    stationary_case,
    operation: str,
) -> None:
    fitted, frame = stationary_case
    compact_state = dataclasses.replace(fitted.fit_state, retained_rows=None)
    compact = DenseDistributionalModel(family=fitted.family, _fit_state=compact_state)

    with pytest.raises(KeyError, match="unknown term 'location:nope'"):
        if operation == "effect":
            term_effect(compact, frame, "location", "nope", covariance="corrected")
        else:
            term_test(compact, frame, "location", "nope", covariance="corrected")


@pytest.mark.parametrize("operation", ["effect", "test"])
def test_invalid_term_grid_count_precedes_lazy_covariance(
    stationary_case,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    fitted, frame = stationary_case
    calls = _install_covariance_spy(monkeypatch, terms_module)

    with pytest.raises(ValueError, match="n_points"):
        if operation == "effect":
            term_effect(
                fitted,
                frame,
                "location",
                "x",
                covariance="corrected",
                n_points=1,
            )
        else:
            term_test(
                fitted,
                frame,
                "location",
                "x",
                covariance="corrected",
                n_points=1,
            )

    assert calls == []


def test_invalid_effect_alpha_precedes_lazy_covariance(
    stationary_case,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fitted, frame = stationary_case
    calls = _install_covariance_spy(monkeypatch, terms_module)

    with pytest.raises(ValueError, match="alpha"):
        term_effect(
            fitted,
            frame,
            "location",
            "x",
            covariance="corrected",
            alpha=1.5,
        )

    assert calls == []


@pytest.mark.parametrize("operation", ["effect", "test"])
def test_invalid_training_frame_precedes_lazy_covariance(
    stationary_case,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    fitted, _ = stationary_case
    calls = _install_covariance_spy(monkeypatch, terms_module)

    with pytest.raises(ValueError, match="pandas or eager Polars"):
        if operation == "effect":
            term_effect(fitted, object(), "location", "x", covariance="corrected")
        else:
            term_test(fitted, object(), "location", "x", covariance="corrected")

    assert calls == []


@pytest.mark.parametrize("operation", ["effect", "test"])
def test_failed_term_sweep_precedes_lazy_covariance(
    stationary_case,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    fitted, frame = stationary_case
    missing_x = pd.DataFrame({"not_x": np.zeros(len(frame))})
    calls = _install_covariance_spy(monkeypatch, terms_module)

    with pytest.raises(KeyError, match="x"):
        if operation == "effect":
            term_effect(fitted, missing_x, "location", "x", covariance="corrected")
        else:
            term_test(fitted, missing_x, "location", "x", covariance="corrected")

    assert calls == []


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [({"n_points": 1}, "n_points"), ({"alpha": 1.5}, "alpha")],
    ids=["n-points", "alpha"],
)
def test_public_plot_validates_term_arguments_before_lazy_covariance(
    stationary_case,
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict[str, float],
    message: str,
) -> None:
    fitted, frame = stationary_case
    model = _public_model(fitted, frame)
    calls = _install_covariance_spy(monkeypatch, api_module)

    with pytest.raises(ValueError, match=message):
        model.plot(
            parameter=None,
            terms="x",
            covariance="corrected",
            n_sim=8,
            **kwargs,
        )

    assert calls == []


@pytest.mark.parametrize(
    ("frame", "error", "message"),
    [
        (object(), ValueError, "pandas or eager Polars"),
        (pd.DataFrame({"not_x": [0.0]}), KeyError, "x"),
    ],
    ids=["frame", "sweep"],
)
def test_public_plot_validates_frame_and_sweep_before_lazy_covariance(
    stationary_case,
    monkeypatch: pytest.MonkeyPatch,
    frame,
    error: type[Exception],
    message: str,
) -> None:
    fitted, _ = stationary_case
    model = _public_model(fitted, frame)
    calls = _install_covariance_spy(monkeypatch, api_module)

    with pytest.raises(error, match=message):
        model.plot(
            parameter=None,
            terms="x",
            covariance="corrected",
            n_points=8,
            n_sim=8,
        )

    assert calls == []


def test_public_plot_prepares_a_later_failing_sweep_before_lazy_covariance(
    stationary_case,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fitted, frame = stationary_case
    model = _public_model(fitted, frame)
    calls = _install_covariance_spy(monkeypatch, api_module)
    real_sweep = terms_module._sweep
    sweeps = 0

    def fail_second_sweep(*args, **kwargs):
        nonlocal sweeps
        sweeps += 1
        if sweeps == 2:
            raise KeyError("forced later sweep failure")
        return real_sweep(*args, **kwargs)

    monkeypatch.setattr(terms_module, "_sweep", fail_second_sweep)
    with pytest.raises(KeyError, match="forced later sweep failure"):
        model.plot(
            parameter=None,
            terms="x",
            covariance="corrected",
            n_points=8,
            n_sim=8,
        )

    assert sweeps == 2
    assert calls == []


@pytest.mark.parametrize("operation", ["effect", "plot"])
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [({"n_sim": 1}, "n_draws"), ({"seed": "not-an-integer"}, "invalid literal")],
    ids=["draw-count", "seed"],
)
def test_simultaneous_draw_arguments_precede_lazy_covariance(
    stationary_case,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    kwargs: dict[str, object],
    message: str,
) -> None:
    fitted, frame = stationary_case
    module = terms_module if operation == "effect" else api_module
    calls = _install_covariance_spy(monkeypatch, module)

    with pytest.raises(ValueError, match=message):
        if operation == "effect":
            term_effect(
                fitted,
                frame,
                "location",
                "x",
                covariance="corrected",
                n_points=8,
                **kwargs,
            )
        else:
            _public_model(fitted, frame).plot(
                parameter=None,
                terms="x",
                covariance="corrected",
                n_points=8,
                **kwargs,
            )

    assert calls == []


@pytest.mark.parametrize("operation", ["effect", "summary"])
def test_one_top_level_term_operation_replays_at_most_once(
    stationary_case,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    fitted, frame = stationary_case
    calls = _install_derivative_spy(monkeypatch)
    if operation == "effect":
        term_effect(
            fitted,
            frame,
            "location",
            "x",
            covariance="corrected",
            n_points=8,
            n_sim=8,
        )
    else:
        summary_table(fitted, frame, covariance="corrected")
    assert calls == [True]


@pytest.mark.parametrize("operation", ["effect", "summary"])
def test_published_hessian_term_path_replays_zero_times(
    stationary_case,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    fitted, frame = stationary_case
    published = _published_model(fitted, _terminal_derivatives(fitted))
    calls = _install_derivative_spy(monkeypatch)
    if operation == "effect":
        term_effect(
            published,
            frame,
            "location",
            "x",
            covariance="corrected",
            n_points=8,
            n_sim=8,
        )
    else:
        summary_table(published, frame, covariance="corrected")
    assert calls == []


def test_public_two_parameter_plot_replays_lazy_hessian_once(
    stationary_case,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fitted, frame = stationary_case
    model = _public_model(fitted, frame)
    calls = _install_derivative_spy(monkeypatch)
    figures = model.plot(
        parameter=None,
        terms="x",
        covariance="corrected",
        n_points=8,
        n_sim=8,
    )
    try:
        assert set(figures) == {"location", "scale"}
        assert calls == [True]
    finally:
        for figure in figures.values():
            plt.close(figure)


def test_public_two_parameter_plot_with_published_hessian_replays_zero_times(
    stationary_case,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fitted, frame = stationary_case
    published = _published_model(fitted, _terminal_derivatives(fitted))
    model = _public_model(published, frame)
    calls = _install_derivative_spy(monkeypatch)
    figures = model.plot(
        parameter=None,
        terms="x",
        covariance="corrected",
        n_points=8,
        n_sim=8,
    )
    try:
        assert set(figures) == {"location", "scale"}
        assert calls == []
    finally:
        for figure in figures.values():
            plt.close(figure)


def test_compact_fit_with_published_hessian_uses_the_fast_path(
    stationary_case,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fitted, _ = stationary_case
    published = _published_model(fitted, _terminal_derivatives(fitted))
    compact_state = dataclasses.replace(published.fit_state, retained_rows=None)
    compact = DenseDistributionalModel(family=published.family, _fit_state=compact_state)
    calls = _install_derivative_spy(monkeypatch)
    corrected = posterior_covariance(compact, kind="corrected")

    assert np.all(np.isfinite(corrected))
    assert calls == []


@pytest.mark.parametrize("failure", ["indefinite", "certificate"])
def test_published_hessian_still_passes_the_trust_gate(
    stationary_case,
    failure: str,
) -> None:
    fitted, _ = stationary_case
    derivatives = _terminal_derivatives(fitted)
    assert derivatives.hessian is not None and derivatives.hessian_certificate is not None
    hessian = np.array(derivatives.hessian, copy=True)
    certificate = np.array(derivatives.hessian_certificate, copy=True)
    if failure == "indefinite":
        hessian[0, 0] = -max(1.0, abs(float(hessian[0, 0])))
        message = "positive definite"
    else:
        certificate[:] = np.max(np.abs(hessian)) + 1.0
        message = "Hessian certificate"
    published = _published_model(
        fitted,
        derivatives,
        hessian=hessian,
        certificate=certificate,
    )

    with pytest.raises(RuntimeError, match=message):
        posterior_covariance(published, kind="corrected")


@dataclasses.dataclass(frozen=True)
class _IndefiniteObservedGaussianLS:
    """GaussianLS whose observed scale-scale curvature is negated at every state.

    Fisher-stepped fits converge on the true mode, but the terminal observed
    curvature is materially indefinite, so the fit accepts the expected
    information instead.
    """

    inner: GaussianLS = GaussianLS()

    @property
    def parameters(self):
        return self.inner.parameters

    @property
    def capabilities(self):
        return self.inner.capabilities

    @property
    def default_prediction_name(self):
        return self.inner.default_prediction_name

    def to_config(self):
        return {"type": "IndefiniteObservedGaussianLS"}

    def bind_likelihood(self, y, weights, observation):
        return self.inner.bind_likelihood(y, weights, observation)

    def initialize(self, y, plan):
        return self.inner.initialize(y, plan)

    def evaluate_natural(self, y, theta, plan, *, derivative_order=2):
        evaluation = self.inner.evaluate_natural(y, theta, plan, derivative_order=derivative_order)
        if evaluation.hessian_packed is None:
            return evaluation
        hessian = np.array(evaluation.hessian_packed, dtype=np.float64)
        hessian[:, 2] = -hessian[:, 2]
        return dataclasses.replace(evaluation, hessian_packed=hessian)

    def expected_information_natural(self, theta, plan):
        return self.inner.expected_information_natural(theta, plan)

    def default_prediction(self, theta):
        return self.inner.default_prediction(theta)


@pytest.fixture(scope="module")
def fisher_fallback_case() -> tuple[DenseDistributionalModel, pd.DataFrame]:
    """A stationary smoothing result whose terminal curvature fell back to Fisher.

    The Newton endgame differentiates the curvature the fit accepted, so a fit
    that falls back can certify a stationary stop on its own. The parent's
    endgame differentiated observed curvature only, refused this fit, and it
    never stopped stationary.
    """
    rng = np.random.default_rng(7351)
    x = rng.uniform(-1.0, 1.0, 320)
    frame = pd.DataFrame({"x": x})
    response = 0.6 * np.sin(2.4 * x) + np.exp(-1.0 + 0.5 * np.cos(1.8 * x)) * rng.standard_normal(
        len(x)
    )
    fitted = (
        model_from_templates(
            family=_IndefiniteObservedGaussianLS(),
            predictors=[
                Predictor("location", {"x": Spline("cr", k=7)}),
                Predictor("scale", {"x": Spline("cr", k=5)}),
            ],
            coefficient_curvature="fisher",
        )
        .fit_reml(frame, response, outer="efs+newton")
        ._require_fitted()
    )
    assert fitted.smoothing is not None
    assert fitted.smoothing.convergence_reason == "stationary"
    assert fitted.result.terminal_curvature.actual_source == "fisher"
    return fitted, frame


def test_fisher_fallback_corrected_covariance_is_computed_and_disclosed(
    fisher_fallback_case,
) -> None:
    """Kills the refusal: the parent commit raised "requires observed terminal curvature".

    V and J already use the accepted Fisher curvature; V_rho is replayed with
    it, so V + J V_rho J' is the corrected covariance of that substitute, and
    the correction J V_rho J' is positive semidefinite and not zero.
    """
    fitted, frame = fisher_fallback_case
    fixed = posterior_covariance(fitted, kind="fixed")
    with pytest.warns(ConvergenceWarning, match="first-order correction") as record:
        corrected = posterior_covariance(fitted, kind="corrected")
    assert len(record) == 1
    assert "expected (Fisher) information" in str(record[0].message)
    correction = corrected - fixed
    np.testing.assert_array_equal(corrected, corrected.T)
    eigenvalues = np.linalg.eigvalsh(correction)
    width = len(correction)
    unit_roundoff = np.finfo(np.float64).eps / 2
    assert eigenvalues[-1] > 0.0
    # Forming V + R R' and subtracting V again rounds each entry at most twice.
    assert eigenvalues[0] >= -4 * width * unit_roundoff * np.max(np.abs(corrected))

    # Every public surface that resolves the corrected covariance discloses once per call.
    with pytest.warns(ConvergenceWarning, match="first-order correction") as record:
        term_effect(fitted, frame, "location", "x", covariance="corrected", n_points=5)
    assert len(record) == 1
