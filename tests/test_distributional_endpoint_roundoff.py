from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import superglm.distributional.efs as efs_module
import superglm.distributional.smoothing.faces as smoothing_faces
import tests.test_distributional_endpoint_laml as endpoint_tests
from superglm import GaussianLS, Predictor, Spline
from superglm._frame import as_eager_frame
from superglm.distributional.layout import build_stacked_layout
from superglm.distributional.predictor import compile_predictors
from superglm.distributional.smoothing.penalty_face import build_penalty_face
from superglm.distributional.weights import WeightContract, resolve_likelihood_weights
from superglm.features import RandomEffect


def _spline_face():
    family = GaussianLS()
    weights = resolve_likelihood_weights(
        np.ones(20, dtype=np.float64),
        n_observations=20,
        contract=WeightContract(semantics="prior"),
    )
    frame = as_eager_frame(pd.DataFrame({"x": np.linspace(0.0, 1.0, 20)}))
    predictors = (
        Predictor(
            "location",
            {"x": Spline(kind="cr", n_knots=5)},
            intercept=True,
        ),
        Predictor(
            "scale",
            {"x": Spline(kind="cr", n_knots=5)},
            intercept=True,
        ),
    )
    layout = build_stacked_layout(compile_predictors(frame, weights, family.parameters, predictors))
    face = build_penalty_face(layout, (layout.penalty_names[0],))
    return face


def test_endpoint_revalidation_projection_roundoff_has_an_arithmetic_bound() -> None:
    face = _spline_face()
    canonical = face.lift_vector(np.linspace(1.0, 2.0, face.reduced_width))
    projected = face.project(canonical)
    movement = float(np.linalg.norm(projected - canonical, ord=2))

    bound = smoothing_faces._endpoint_revalidation_projection_bound(face, canonical)

    assert bound is not None
    assert movement <= bound


def test_endpoint_revalidation_projection_bound_rejects_a_material_scalar_move() -> None:
    family = GaussianLS()
    weights = resolve_likelihood_weights(
        np.ones(1, dtype=np.float64),
        n_observations=1,
        contract=WeightContract(semantics="prior"),
    )
    frame = as_eager_frame(pd.DataFrame({"effect": ["only"]}))
    layout = build_stacked_layout(
        compile_predictors(
            frame,
            weights,
            family.parameters,
            (
                Predictor("location", {"effect": RandomEffect()}),
                Predictor("scale", {"effect": RandomEffect()}),
            ),
        )
    )
    face = build_penalty_face(layout, (layout.penalty_names[0],))
    canonical = np.zeros(face.width, dtype=np.float64)
    altered = canonical + 1.0e-14 * face.constraint_basis[:, 0]

    bound = smoothing_faces._endpoint_revalidation_projection_bound(face, altered)
    projected = face.project(altered)
    movement = float(np.linalg.norm(projected - altered, ord=2))

    assert bound is not None
    assert movement > bound


def test_stationary_endpoint_revalidation_accepts_a_subnormal_projection_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    real_check = efs_module._check_face_direction

    def perturb_revalidation_start(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        if calls == 2:
            endpoint_initial = np.array(kwargs["endpoint_initial"], copy=True)
            endpoint_initial[0] = np.nextafter(0.0, np.inf)
            kwargs["endpoint_initial"] = endpoint_initial
        return real_check(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(smoothing_faces, "_check_face_direction", perturb_revalidation_start)
    monkeypatch.setattr(efs_module, "_check_face_direction", perturb_revalidation_start)
    component_name, smoothing = endpoint_tests._scalar_efs_fit(0.5)

    assert calls == 2
    assert smoothing.converged is True
    assert smoothing.terminal_fit.coefficient_face is not None
    assert smoothing.history[-1].revalidated_face_components == (component_name,)


def test_endpoint_revalidation_records_material_state_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    real_check = efs_module._check_face_direction
    real_fit = smoothing_faces._fit_endpoint_authority_stationary
    inject_result = False
    injected = False

    def inject_endpoint_result(*args: object, **kwargs: object):
        nonlocal injected
        fit = real_fit(*args, **kwargs)  # type: ignore[arg-type]
        if inject_result and not injected and kwargs.get("face") is not None:
            endpoint_face = kwargs["face"]
            assert endpoint_face is not None
            assert endpoint_face.null_basis.shape[1] > 0
            injected = True
            fit = replace(
                fit,
                coefficients=fit.coefficients + 1.0e-6 * endpoint_face.null_basis[:, 0],
            )
        return fit

    def perturb_revalidation_start(*args: object, **kwargs: object):
        nonlocal calls, inject_result
        calls += 1
        if calls == 2:
            np.testing.assert_array_equal(kwargs["endpoint_initial"], kwargs["initial"])
            inject_result = True
        try:
            return real_check(*args, **kwargs)  # type: ignore[arg-type]
        finally:
            inject_result = False

    monkeypatch.setattr(
        smoothing_faces, "_fit_endpoint_authority_stationary", inject_endpoint_result
    )
    monkeypatch.setattr(smoothing_faces, "_check_face_direction", perturb_revalidation_start)
    monkeypatch.setattr(efs_module, "_check_face_direction", perturb_revalidation_start)
    (
        _family,
        _layout,
        _response,
        _plan,
        _lambdas,
        component_name,
        _finite,
        _config,
        smoothing,
    ) = endpoint_tests._gamma_isolated_cap_face_problem()

    terminal_event = smoothing.history[-1]
    assert calls == 2
    assert smoothing.converged is False
    assert terminal_event.deactivated_face_components == (component_name,)
    assert terminal_event.endpoint_assessment_failure_reason == "endpoint_state_changed"


def test_result_rejects_a_within_bound_state_change_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    real_check = efs_module._check_face_direction
    real_fit = smoothing_faces._fit_endpoint_authority_stationary
    inject_result = False
    injected = False

    def inject_endpoint_result(*args: object, **kwargs: object):
        nonlocal injected
        fit = real_fit(*args, **kwargs)  # type: ignore[arg-type]
        if inject_result and not injected and kwargs.get("face") is not None:
            endpoint_face = kwargs["face"]
            assert endpoint_face is not None
            assert endpoint_face.null_basis.shape[1] > 0
            injected = True
            fit = replace(
                fit,
                coefficients=fit.coefficients + 1.0e-6 * endpoint_face.null_basis[:, 0],
            )
        return fit

    def material_refusal(*args: object, **kwargs: object):
        nonlocal calls, inject_result
        calls += 1
        if calls == 2:
            np.testing.assert_array_equal(kwargs["endpoint_initial"], kwargs["initial"])
            inject_result = True
        try:
            return real_check(*args, **kwargs)  # type: ignore[arg-type]
        finally:
            inject_result = False

    monkeypatch.setattr(
        smoothing_faces, "_fit_endpoint_authority_stationary", inject_endpoint_result
    )
    monkeypatch.setattr(smoothing_faces, "_check_face_direction", material_refusal)
    monkeypatch.setattr(efs_module, "_check_face_direction", material_refusal)
    _values = endpoint_tests._gamma_isolated_cap_face_problem()
    smoothing = _values[-1]
    retraction = next(item for item in smoothing.history if item.deactivated_face_components)
    endpoint_index = retraction.coefficient_fit_indices[1]
    endpoint_fit = smoothing.coefficient_fits[endpoint_index]
    source_fit = smoothing.coefficient_fits[retraction.source_fit_index]
    assert endpoint_fit.coefficient_face is not None
    within_bound = replace(
        endpoint_fit,
        coefficients=source_fit.coefficients
        + 1.0e-14 * endpoint_fit.coefficient_face.null_basis[:, 0],
    )
    fits = list(smoothing.coefficient_fits)
    fits[endpoint_index] = within_bound

    with pytest.raises(ValueError, match="within the refit envelope"):
        replace(smoothing, coefficient_fits=tuple(fits))

    assert endpoint_fit.terminal_reduced_rank is not None
    bad_rank = replace(
        endpoint_fit.terminal_reduced_rank,
        rank=endpoint_fit.terminal_reduced_rank.rank - 1,
    )
    bad_endpoint = replace(endpoint_fit, terminal_reduced_rank=bad_rank)
    fits[endpoint_index] = bad_endpoint
    with pytest.raises(ValueError, match="full observed rank"):
        replace(smoothing, coefficient_fits=tuple(fits))
