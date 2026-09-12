"""Refresh an old negative-curvature verdict once BFGS approaches a stop."""

from dataclasses import replace

import numpy as np
import pytest
from benchmarks._c3_c1_fixtures import lss_model, marked_book
from benchmarks.c3_c1_complete_fit import gaussian_fragmented_fixture

from superglm.distributional import GeneralizedParetoLSS
from superglm.distributional.smoothing import newton
from superglm.distributional.smoothing.derivatives import LamlDerivativeError


def _case(kind):
    if kind == "gaussian":
        model, frame, y, *_ = gaussian_fragmented_fixture(4096, 4)
        return model, frame, y
    book = marked_book(10000, tail=True)
    keep = (book["policy"] < 10000) & (book["losses"] > 1000)
    frame = book["frame"].iloc[book["policy"][keep]].reset_index(drop=True)
    return lss_model(GeneralizedParetoLSS(), 3), frame, book["losses"][keep] - 1000


@pytest.mark.parametrize("kind", ["gaussian", "gpd"])
def test_newton_refreshes_indefinite_curvature_after_accepted_bfgs_steps(monkeypatch, kind):
    """The old Hessian must not veto completion at every later accepted point.

    On the unfixed solver, these real fits exhaust the Newton budget despite
    satisfying the gradient, objective-change and remaining-gain checks.
    They request one Hessian and retain its negative-curvature flag throughout.
    """
    real_derivatives = newton.laml_derivatives
    real_freeze = newton.freeze_flat_directions
    hessians = []
    gradient_fits = []
    gradient_workspaces = []
    gradients = []
    active_masks = []

    def record_freeze(*args, **kwargs):
        result = real_freeze(*args, **kwargs)
        gradient = gradients[-1]
        active_masks.append(
            ~np.asarray(result.frozen) & (gradient.gradient_certificate < np.abs(gradient.gradient))
        )
        return result

    def record_derivatives(*args, **kwargs):
        if kwargs["want_hessian"]:
            assert kwargs["fit"] is gradient_fits[-1]
            assert kwargs["reuse"] is gradient_workspaces[-1]
        result = real_derivatives(*args, **kwargs)
        if kwargs["want_hessian"]:
            hessians.append((kwargs["fit"], dict(kwargs["lambdas"]), result, active_masks[-1]))
        else:
            gradient_fits.append(kwargs["fit"])
            gradient_workspaces.append(kwargs["reuse"])
            gradients.append(result)
        return result

    monkeypatch.setattr(newton, "laml_derivatives", record_derivatives)
    monkeypatch.setattr(newton, "freeze_flat_directions", record_freeze)
    model, frame, y = _case(kind)
    model.fit_reml(frame, y, outer="efs+newton", practical_reml=False)
    smoothing = model._require_fitted().smoothing

    first_fit, first_lambdas, first, active = hessians[0]
    # The initial negative eigenvalue is separated from both the derivative
    # certificate and a conservative relative perturbation of the matrix.
    block = first.hessian[np.ix_(active, active)]
    error = np.linalg.norm(first.hessian_certificate[np.ix_(active, active)], 2)
    error += np.sqrt(np.finfo(float).eps) * np.linalg.norm(block, 2)
    assert np.linalg.eigvalsh(block)[0] < -error
    assert len(hessians) > 1, "an old indefinite Hessian needs a fresh evaluation near a stop"
    assert hessians[-1][0] is not first_fit
    assert hessians[-1][1] != first_lambdas
    assert any(item.accepted and item.step_source == "bfgs" for item in smoothing.history)
    assert smoothing.converged and smoothing.matched_certified
    assert smoothing.convergence_reason == "stationary"
    assert smoothing.newton_iterations < smoothing.config.max_newton_iterations
    assert gradient_fits[-1] is smoothing.terminal_fit
    assert smoothing.terminal_evidence_fresh
    assert smoothing.terminal_projected_gradient_norm <= smoothing.stationarity_bar
    assert all(
        value <= smoothing.stationarity_bar
        for value in smoothing.terminal_gradient_certificate.values()
    )


@pytest.mark.parametrize("failure", ["unavailable", "untrusted", "indefinite"])
def test_failed_curvature_refresh_keeps_bounded_bfgs_fallback(monkeypatch, failure):
    """A refresh failure keeps the accepted state and the existing inverse memory."""
    real_derivatives = newton.laml_derivatives
    hessian_calls = []
    gradient_fits = []
    gradients = []

    def fail_refresh(*args, **kwargs):
        if kwargs["want_hessian"]:
            hessian_calls.append(dict(kwargs["lambdas"]))
            if len(hessian_calls) > 1 and failure == "unavailable":
                raise LamlDerivativeError("injected unavailable Hessian at curvature refresh")
        else:
            gradient_fits.append(kwargs["fit"])
        result = real_derivatives(*args, **kwargs)
        if not kwargs["want_hessian"]:
            gradients.append(result)
        if kwargs["want_hessian"] and len(hessian_calls) > 1 and failure == "untrusted":
            result = replace(
                result,
                hessian_certificate=np.full_like(
                    result.hessian, 1.0 + np.linalg.norm(result.hessian, 2)
                ),
            )
        if kwargs["want_hessian"] and len(hessian_calls) > 1 and failure == "indefinite":
            result = replace(
                result,
                hessian=-np.eye(len(result.names))
                * (1.0 + np.linalg.norm(result.hessian_certificate, 2)),
            )
        return result

    monkeypatch.setattr(newton, "laml_derivatives", fail_refresh)
    model, frame, y = _case("gaussian")
    model.fit_reml(frame, y, outer="efs+newton", practical_reml=False)
    smoothing = model._require_fitted().smoothing

    assert len(hessian_calls) > 1
    assert any(
        item.accepted
        and item.step_source == "bfgs"
        and dict(item.lambdas_before) == hessian_calls[1]
        for item in smoothing.history
    )
    assert smoothing.newton_iterations <= smoothing.config.max_newton_iterations
    assert gradient_fits[-1] is smoothing.terminal_fit
    assert smoothing.terminal_fit.converged
    assert dict(smoothing.terminal_gradient) == dict(
        zip(gradients[-1].names, gradients[-1].gradient, strict=True)
    )
    assert dict(smoothing.terminal_gradient_certificate) == dict(
        zip(gradients[-1].names, gradients[-1].gradient_certificate, strict=True)
    )
    assert smoothing.smoothing_hessian is None
    assert smoothing.smoothing_hessian_certificate is None
    if failure in {"unavailable", "indefinite"}:
        assert len(hessian_calls) > 2
        assert not smoothing.converged
        assert smoothing.convergence_reason == "max_iterations"
    if smoothing.converged:
        # The existing fallback contract certifies first-order stationarity;
        # it does not claim a positive-definite terminal Hessian.
        assert smoothing.convergence_reason == "stationary"
        assert smoothing.terminal_evidence_fresh
        assert smoothing.terminal_projected_gradient_norm <= smoothing.stationarity_bar
        assert all(
            value <= smoothing.stationarity_bar
            for value in smoothing.terminal_gradient_certificate.values()
        )
