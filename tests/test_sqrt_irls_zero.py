"""Square-root-link IRLS regressions at the two-branch origin."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from superglm import SuperGLM
from superglm.distributions import Poisson
from superglm.features import Numeric
from superglm.links import SqrtLink
from superglm.penalties.group_lasso import GroupLasso
from superglm.solvers.irls_direct import fit_irls_direct
from superglm.solvers.pirls import fit_pirls
from superglm.types import GroupSlice


@pytest.mark.parametrize("response", [1.0e-12, 2.0, 100.0, 1.0e12])
def test_poisson_sqrt_exact_zero_start_is_response_scale_equivariant(
    response: float,
) -> None:
    n = 30
    natural_eta = float(np.sqrt(response))
    y = np.full(n, response)
    offset = np.full(n, -natural_eta)

    result, _ = fit_irls_direct(
        X=np.empty((n, 0)),
        y=y,
        weights=np.ones(n),
        family=Poisson(),
        link=SqrtLink(),
        groups=[],
        lambda2=0.0,
        offset=offset,
        max_iter=20,
        tol=1.0e-30,
        record_diagnostics=True,
    )

    fitted_eta = result.intercept + offset
    assert result.converged
    assert result.iteration_log is not None
    assert not result.iteration_log[0].step_rejected
    assert result.iteration_log[0].accepted_alpha == 1.0
    np.testing.assert_allclose(
        fitted_eta / natural_eta,
        np.ones(n),
        rtol=2.0e-14,
        atol=2.0e-14,
    )
    assert result.intercept / natural_eta == pytest.approx(2.0, rel=2.0e-14)


@pytest.mark.parametrize("response", [1.0e-12, 1.0, 1.0e12])
def test_poisson_sqrt_negative_branch_is_retained_across_response_scales(
    response: float,
) -> None:
    n = 30
    natural_eta = float(np.sqrt(response))

    result, _ = fit_irls_direct(
        X=np.empty((n, 0)),
        y=np.full(n, response),
        weights=np.ones(n),
        family=Poisson(),
        link=SqrtLink(),
        groups=[],
        lambda2=0.0,
        intercept_init=-0.5 * natural_eta,
        max_iter=20,
        tol=1.0e-30,
    )

    assert result.converged
    assert result.intercept / natural_eta == pytest.approx(-1.0, rel=2.0e-14)


@pytest.mark.parametrize("route", ["direct", "penalized"])
@pytest.mark.parametrize("sign", [-1.0, 1.0])
@pytest.mark.parametrize(
    "initial_magnitude",
    [
        pytest.param(np.nextafter(0.0, 1.0), id="min-subnormal"),
        pytest.param(1.0e-320, id="1e-320"),
        pytest.param(1.0e-310, id="1e-310"),
        pytest.param(1.0e-307, id="1e-307"),
        pytest.param(1.0e-300, id="1e-300"),
    ],
)
def test_poisson_sqrt_extreme_nonzero_start_uses_finite_branch_trust_response(
    route: str,
    sign: float,
    initial_magnitude: float,
) -> None:
    n = 8
    y = np.full(n, 100.0)
    initial_eta = sign * initial_magnitude

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if route == "direct":
            result, _ = fit_irls_direct(
                X=np.empty((n, 0)),
                y=y,
                weights=np.ones(n),
                family=Poisson(),
                link=SqrtLink(),
                groups=[],
                lambda2=0.0,
                intercept_init=initial_eta,
                max_iter=20,
                tol=1.0e-12,
            )
        else:
            result = fit_pirls(
                X=np.zeros((n, 1)),
                y=y,
                weights=np.ones(n),
                family=Poisson(),
                link=SqrtLink(),
                groups=[GroupSlice(name="x", start=0, end=1)],
                penalty=GroupLasso(lambda1=0.1),
                beta_init=np.zeros(1),
                intercept_init=initial_eta,
                max_iter_outer=20,
                max_iter_inner=20,
                tol=1.0e-12,
            )

    runtime_warnings = [
        warning for warning in caught if issubclass(warning.category, RuntimeWarning)
    ]
    assert runtime_warnings == []
    assert result.converged, (route, sign, initial_magnitude, result.termination_reason)
    assert np.signbit(result.intercept) == np.signbit(initial_eta)
    assert result.intercept**2 == pytest.approx(100.0, rel=5.0e-14)


def _fit_public_constant(
    *,
    response: float,
    initial_eta: float,
    selection_penalty: float,
) -> tuple[SuperGLM, pd.DataFrame, np.ndarray]:
    n = 30
    natural_eta = float(np.sqrt(response))
    X = pd.DataFrame({"x": np.zeros(n)})
    y = np.full(n, response)
    offset = np.full(n, initial_eta - natural_eta)
    model = SuperGLM(
        family="poisson",
        link="sqrt",
        features={"x": Numeric()},
        selection_penalty=selection_penalty,
        max_iter=200,
    ).fit(X, y, offset=offset)
    return model, X, offset


@pytest.mark.parametrize("response", [1.0e-12, 100.0, 1.0e12])
@pytest.mark.parametrize("selection_penalty", [0.0, 1.0e-12, 1.0])
def test_public_poisson_sqrt_crosses_old_origin_and_stabilization_boundaries(
    response: float,
    selection_penalty: float,
) -> None:
    natural_eta = float(np.sqrt(response))
    old_origin_tolerance = 64.0 * np.finfo(float).eps * natural_eta
    magnitudes = (
        0.0,
        0.99 * old_origin_tolerance,
        old_origin_tolerance,
        1.01 * old_origin_tolerance,
        2.0 * old_origin_tolerance,
        100.0 * old_origin_tolerance,
        0.99e-7,
        1.0e-7,
        1.01e-7,
        1.0e-6,
    )

    for magnitude in magnitudes:
        signs = (1.0,) if magnitude == 0.0 else (-1.0, 1.0)
        for sign in signs:
            initial_eta = sign * magnitude
            model, X, offset = _fit_public_constant(
                response=response,
                initial_eta=initial_eta,
                selection_penalty=selection_penalty,
            )
            prediction = model.predict(X, offset=offset)
            fitted_eta = float(model.result.intercept + offset[0])

            assert model.result.converged, (
                response,
                selection_penalty,
                initial_eta,
                model.result.termination_reason,
            )
            np.testing.assert_allclose(prediction, response, rtol=5.0e-6, atol=0.0)
            if initial_eta != 0.0:
                assert np.signbit(fitted_eta) == np.signbit(initial_eta)


@pytest.mark.parametrize("response", [1.0e-30, 1.0e-16, 1.0e-14, 1.0e-12])
@pytest.mark.parametrize("selection_penalty", [0.0, 0.1])
@pytest.mark.parametrize("exact_zero_start", [False, True])
def test_public_poisson_sqrt_preserves_genuinely_tiny_means(
    response: float,
    selection_penalty: float,
    exact_zero_start: bool,
) -> None:
    initial_eta = 0.0 if exact_zero_start else float(np.sqrt(response))
    model, X, offset = _fit_public_constant(
        response=response,
        initial_eta=initial_eta,
        selection_penalty=selection_penalty,
    )

    assert model.result.converged
    np.testing.assert_allclose(
        model.predict(X, offset=offset),
        response,
        rtol=5.0e-7,
        atol=0.0,
    )


@pytest.mark.parametrize("selection_penalty", [0.0, 0.1])
def test_public_poisson_sqrt_handles_heterogeneous_zero_and_weight_rows(
    selection_penalty: float,
) -> None:
    y = np.tile(np.array([0.0, 1.0, 4.0, 100.0, 25.0]), 8)
    weights = np.tile(np.array([1.0, 0.5, 2.0, 3.0, 0.0]), 8)
    expected_mean = float(np.average(y, weights=weights))
    natural_eta = float(np.sqrt(expected_mean))
    X = pd.DataFrame({"x": np.zeros(len(y))})
    offset = np.full(len(y), -natural_eta)

    model = SuperGLM(
        family="poisson",
        link="sqrt",
        features={"x": Numeric()},
        selection_penalty=selection_penalty,
        max_iter=200,
    ).fit(X, y, sample_weight=weights, offset=offset)

    assert model.result.converged
    np.testing.assert_allclose(
        model.predict(X, offset=offset),
        expected_mean,
        rtol=5.0e-6,
        atol=0.0,
    )


@pytest.mark.parametrize("selection_penalty", [0.0, 0.1])
def test_public_poisson_sqrt_all_zero_response_has_finite_terminal_geometry(
    selection_penalty: float,
) -> None:
    X = pd.DataFrame({"x": np.zeros(20)})
    y = np.zeros(len(X))
    model = SuperGLM(
        family="poisson",
        link="sqrt",
        features={"x": Numeric()},
        selection_penalty=selection_penalty,
    ).fit(X, y)
    prediction = model.predict(X)

    assert model.result.converged
    assert np.all(np.isfinite(prediction))
    assert np.all(prediction <= 1.0e-49)
