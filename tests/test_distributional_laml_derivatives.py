"""Second directional differences of packed row curvature and exact negative-LAML
derivatives in log lambda.

The Gaussian location-scale family with the identity location link and the
plain log scale link (``scale_floor=0.0``, so ``sigma = exp(s)`` exactly) has
closed-form packed curvature.  ``_curvature_packed`` packs the negated row
Hessian in the linear predictors in ``packed_pairs`` order, ``(0, 0), (0, 1),
(1, 1)`` for two parameters; with ``s = log sigma``:

    c_00 = exp(-2 s)
    c_01 = 2 (y - mu) exp(-2 s)
    c_11 = 2 (y - mu)^2 exp(-2 s)

so the second directional derivatives along the parameter axes are

    d2c/dmu2   = (0,            0,                   4 exp(-2 s))
    d2c/dmu ds = (0,            4 exp(-2 s),         8 (y - mu) exp(-2 s))
    d2c/ds2    = (4 exp(-2 s),  8 (y - mu) exp(-2 s), 8 (y - mu)^2 exp(-2 s))

The interior fixtures use n = 1,500 and two cubic regression splines per
predictor for Gaussian (location, scale) and Gamma (mean, scale) fits at the
strict EFS stop. ``SuperLSS.fit_reml`` steps both families with Fisher
curvature; the dense solver publishes the observed Hessian as the terminal
curvature under either step policy, so the fixtures run under both to show the
derivative of ``joint_laplace_objective`` does not depend on the inner geometry
that found the mode.
"""

from __future__ import annotations

import functools
import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from superglm.distributional import GammaLS, Predictor
from superglm.distributional.families.gaussian import GaussianLS
from superglm.distributional.family import COMPLETE_OBSERVATION
from superglm.distributional.model import fit_dense_distributional
from superglm.distributional.result import DenseSolverConfig, DistributionalEFSConfig
from superglm.distributional.smoothing.derivatives import (
    LamlDerivativeError,
    _cross_all,
    _first_derivatives,
    _second_derivatives,
    laml_derivatives,
)
from superglm.distributional.smoothing.endpoint_direction import (
    DEFAULT_STEP,
    _curvature_packed,
    finite_difference_curvature_direction,
    finite_difference_curvature_second_direction,
)
from superglm.distributional.solver.packing import packed_pairs
from superglm.distributional.solver.solver import _DenseObservedReuseSession
from superglm.distributional.weights import WeightContract, resolve_likelihood_weights
from superglm.features import Spline
from tests._laml_legacy import legacy_laml_derivatives
from tests._laml_oracle import oracle_gradient, probe_fit


def _gaussian_rows(n: int = 500, seed: int = 11):
    rng = np.random.default_rng(seed)
    mu = rng.normal(0.0, 1.5, n)
    s = rng.uniform(-1.0, 0.7, n)
    y = mu + np.exp(s) * rng.normal(size=n)
    eta = np.column_stack([mu, s])
    return y, eta


def _gaussian_plan(y):
    family = GaussianLS(scale_floor=0.0)
    weights = resolve_likelihood_weights(
        None, n_observations=len(y), contract=WeightContract("prior")
    )
    plan = family.bind_likelihood(y, weights, COMPLETE_OBSERVATION)
    links = tuple(parameter.default_link for parameter in family.parameters)
    return family, plan, links


def _closed_form_second(y, eta, q: int, r: int):
    mu, s = eta[:, 0], eta[:, 1]
    e = np.exp(-2.0 * s)
    d = y - mu
    if (q, r) == (0, 0):
        return np.column_stack([0 * e, 0 * e, 4 * e])
    if (q, r) in {(0, 1), (1, 0)}:
        return np.column_stack([0 * e, 4 * e, 8 * d * e])
    return np.column_stack([4 * e, 8 * d * e, 8 * d * d * e])


@pytest.mark.parametrize("pair", [(0, 0), (0, 1), (1, 1)])
def test_second_difference_matches_the_closed_form_gaussian_curvature(pair) -> None:
    y, eta = _gaussian_rows()
    family, plan, links = _gaussian_plan(y)
    unit = np.zeros((len(y), 2))
    a = unit.copy()
    a[:, pair[0]] = 1.0
    b = unit.copy()
    b[:, pair[1]] = 1.0
    got = finite_difference_curvature_second_direction(family, y, eta, a, b, links, plan)
    want = _closed_form_second(y, eta, *pair)
    scale = np.maximum(np.abs(want), 1.0)
    assert np.max(np.abs(got.values - want) / scale) < 1e-7
    assert np.all(np.abs(got.values - want) <= got.certificate + 64 * np.finfo(float).eps * scale)


def test_second_difference_is_bilinear_and_symmetric() -> None:
    y, eta = _gaussian_rows()
    family, plan, links = _gaussian_plan(y)
    rng = np.random.default_rng(3)
    a = rng.normal(size=(len(y), 2))
    b = rng.normal(size=(len(y), 2))
    ab = finite_difference_curvature_second_direction(family, y, eta, a, b, links, plan)
    ba = finite_difference_curvature_second_direction(family, y, eta, b, a, links, plan)
    a2b = finite_difference_curvature_second_direction(family, y, eta, 2.0 * a, b, links, plan)
    tol = ab.certificate + ba.certificate + 1e-9 * np.maximum(np.abs(ab.values), 1.0)
    assert np.all(np.abs(ab.values - ba.values) <= tol)
    assert np.all(np.abs(a2b.values - 2.0 * ab.values) <= 2.0 * tol + a2b.certificate)


def test_second_difference_reduces_to_first_difference_of_first_difference() -> None:
    """The outer central difference carries its own ``h^2/6 * c(4)/c(2)`` truncation.

    Along the scale axis every ``s``-derivative of the closed-form channels
    multiplies by ``-2``, so that relative term is ``4 h^2 / 6``: ``6.7e-5`` at
    ``h = 10 * DEFAULT_STEP`` (measured 6.67e-5, above the bound) and
    ``6.7e-7`` at ``h = DEFAULT_STEP``.  The bound is on the agreement of the
    two constructions, so the outer step is the one that keeps the outer
    truncation below it.
    """
    y, eta = _gaussian_rows(n=200)
    family, plan, links = _gaussian_plan(y)
    a = np.zeros((len(y), 2))
    a[:, 1] = 1.0
    h = DEFAULT_STEP
    plus = finite_difference_curvature_direction(family, y, eta + h * a, a, links, plan).values
    minus = finite_difference_curvature_direction(family, y, eta - h * a, a, links, plan).values
    outer = (plus - minus) / (2.0 * h)
    got = finite_difference_curvature_second_direction(family, y, eta, a, a, links, plan)
    assert np.max(np.abs(got.values - outer) / np.maximum(np.abs(outer), 1.0)) < 1e-5


def test_second_difference_fails_closed_on_bad_inputs() -> None:
    y, eta = _gaussian_rows(n=50)
    family, plan, links = _gaussian_plan(y)
    a = np.zeros((len(y), 2))
    a[:, 0] = 1.0
    with pytest.raises(ValueError, match="step"):
        finite_difference_curvature_second_direction(family, y, eta, a, a, links, plan, step=0.0)
    bad = a.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        finite_difference_curvature_second_direction(family, y, eta, bad, a, links, plan)
    with pytest.raises(ValueError, match="shape"):
        finite_difference_curvature_second_direction(family, y, eta, a[:, :1], a, links, plan)


# --- Exact gradient and Hessian of the negative LAML -------------------------

N_INTERIOR = 1500


def _interior_data(kind: str, seed: int = 17):
    """Build an interior fixture with n = 1,500 and two smooths per predictor."""
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0.0, 1.0, N_INTERIOR)
    x2 = rng.uniform(0.0, 1.0, N_INTERIOR)
    frame = pd.DataFrame({"x1": x1, "x2": x2})
    g1 = np.sin(2.0 * np.pi * x1)
    g2 = 4.0 * (x2 - 0.5) ** 2 - 0.33
    h1 = np.sin(np.pi * x1)
    h2 = np.cos(2.0 * np.pi * x2)
    if kind == "gaussian":
        mu = 2.0 * g1 + 1.5 * g2
        sigma = np.exp(-0.4 + 0.8 * h1 + 0.5 * h2)
        y = mu + sigma * rng.normal(0.0, 1.0, N_INTERIOR)
        family = GaussianLS(scale_floor=0.01)
        names = ("location", "scale")
    elif kind == "gamma":
        mean = np.exp(1.0 + 0.9 * g1 + 0.6 * g2)
        cv = np.exp(-0.8 + 0.5 * h1 + 0.4 * h2)
        shape = 1.0 / cv**2
        y = np.maximum(mean * rng.gamma(shape, 1.0, N_INTERIOR) / shape, 1.0e-10)
        family = GammaLS()
        names = ("mean", "scale")
    else:
        raise ValueError(kind)
    predictors = (
        Predictor(names[0], {"x1": Spline(kind="cr", k=10), "x2": Spline(kind="cr", k=8)}),
        Predictor(names[1], {"x1": Spline(kind="cr", k=8), "x2": Spline(kind="cr", k=6)}),
    )
    return frame, y, family, predictors


@functools.cache
def _interior_case(kind: str, curvature: str):
    """Return (family, layout, y, plan, lambdas, fit, solver_config, session) at the EFS stop."""
    frame, y, family, predictors = _interior_data(kind)
    model = fit_dense_distributional(
        frame,
        y,
        family=family,
        predictors=predictors,
        weight_contract=WeightContract("prior"),
        config=DenseSolverConfig(
            max_iterations=500, tolerance=1.0e-11, coefficient_curvature=curvature
        ),
        efs_config=DistributionalEFSConfig(
            max_iterations=250, practical_convergence=False, outer="efs"
        ),
        retain_rows=True,
    )
    smoothing = model.smoothing
    assert smoothing is not None
    fit = smoothing.terminal_fit
    assert fit.coefficient_face is None and not smoothing.unresolved_upper_bound
    # ``curvature`` is the solver's STEP policy; the dense solver publishes the
    # observed Hessian as the terminal curvature under both, and that is the H
    # inside the objective.
    assert fit.terminal_curvature.actual_source == "observed"
    rows = model.fit_state.retained_rows
    assert rows is not None
    response = np.asarray(rows.response, dtype=np.float64)
    plan = family.bind_likelihood(response, rows.likelihood_weights, COMPLETE_OBSERVATION)
    assert plan.plan_identifier == fit.family_likelihood_plan_identifier
    return (
        family,
        model.layout,
        response,
        plan,
        dict(smoothing.lambdas),
        fit,
        fit.config,
        _DenseObservedReuseSession(),
    )


def _gradient_at(family, layout, y, plan, lambdas, base_fit, config, session):
    fit = probe_fit(family, layout, y, plan, lambdas, base_fit, config)
    return laml_derivatives(
        family,
        layout,
        y,
        plan,
        lambdas=lambdas,
        fit=fit,
        dense_matrices=session.dense_matrices(layout),
        want_hessian=False,
    ).gradient


@pytest.mark.parametrize("curvature", ["observed", "fisher"])
@pytest.mark.parametrize("kind", ["gaussian", "gamma"])
def test_laml_gradient_matches_richardson_difference(kind, curvature) -> None:
    family, layout, y, plan, lambdas, fit, config, session = _interior_case(kind, curvature)
    derivatives = laml_derivatives(
        family,
        layout,
        y,
        plan,
        lambdas=lambdas,
        fit=fit,
        dense_matrices=session.dense_matrices(layout),
        want_hessian=False,
    )
    for index, name in enumerate(derivatives.names):
        oracle = oracle_gradient(family, layout, y, plan, lambdas, fit, config, name)
        tolerance = 3.0 * oracle.resolution + derivatives.gradient_certificate[index]
        assert abs(derivatives.gradient[index] - oracle.value) <= tolerance, (
            name,
            derivatives.gradient[index],
            oracle,
        )
        # The omitted term separates the exact gradient from the FS terms; at
        # this strict EFS stop it remains materially nonzero.
        assert abs(derivatives.fs_gradient[index] - derivatives.gradient[index]) > 1e-4


@pytest.mark.parametrize("curvature", ["observed", "fisher"])
def test_laml_hessian_matches_difference_of_gradient(curvature) -> None:
    family, layout, y, plan, lambdas, fit, config, session = _interior_case("gaussian", curvature)
    derivatives = laml_derivatives(
        family,
        layout,
        y,
        plan,
        lambdas=lambdas,
        fit=fit,
        dense_matrices=session.dense_matrices(layout),
    )
    assert np.allclose(derivatives.hessian, derivatives.hessian.T, atol=1e-12, rtol=0.0)
    h = 1e-2
    for k, name in enumerate(derivatives.names):
        up = dict(lambdas)
        up[name] = lambdas[name] * math.exp(h)
        down = dict(lambdas)
        down[name] = lambdas[name] * math.exp(-h)
        g_up = _gradient_at(family, layout, y, plan, up, fit, config, session)
        g_down = _gradient_at(family, layout, y, plan, down, fit, config, session)
        column = (g_up - g_down) / (2.0 * h)
        tolerance = (
            3e-3 * np.maximum(np.abs(column), np.max(np.abs(derivatives.hessian)))
            + derivatives.hessian_certificate[:, k]
        )
        assert np.all(np.abs(derivatives.hessian[:, k] - column) <= tolerance), name


def test_gamma_analytic_and_finite_difference_third_derivatives_agree() -> None:
    family, layout, y, plan, lambdas, fit, config, session = _interior_case("gamma", "observed")
    analytic = laml_derivatives(
        family,
        layout,
        y,
        plan,
        lambdas=lambdas,
        fit=fit,
        dense_matrices=session.dense_matrices(layout),
        want_hessian=False,
    )

    class _NoDirection(type(family)):
        predictor_curvature_directional_derivative = None  # type: ignore[assignment]

    numeric = laml_derivatives(
        _NoDirection(),
        layout,
        y,
        plan,
        lambdas=lambdas,
        fit=fit,
        dense_matrices=session.dense_matrices(layout),
        want_hessian=False,
    )
    assert analytic.third_derivative_authority != numeric.third_derivative_authority
    assert np.all(
        np.abs(analytic.gradient - numeric.gradient)
        <= numeric.gradient_certificate + analytic.gradient_certificate + 1e-10
    )


def test_derivatives_refuse_a_non_converged_fit() -> None:
    family, layout, y, plan, lambdas, fit, config, session = _interior_case("gaussian", "fisher")
    with pytest.raises(LamlDerivativeError, match="converged"):
        laml_derivatives(
            family,
            layout,
            y,
            plan,
            lambdas=lambdas,
            fit=replace(fit, converged=False, convergence_reason="max_iterations"),
            dense_matrices=session.dense_matrices(layout),
        )


def test_config_derivative_step_default_is_the_finite_difference_step() -> None:
    """``results`` cannot import ``smoothing.endpoint_direction`` (import cycle); pin the value."""
    assert DistributionalEFSConfig().derivative_step == DEFAULT_STEP


# --- Derivative-pass cost: the pre-perf assembly is the oracle ---------------


class _GammaNoDirection(GammaLS):
    """GammaLS on the finite-difference path (the retained Tweedie cell's authority)."""

    predictor_curvature_directional_derivative = None  # type: ignore[assignment]


def _legacy_case(kind: str, authority: str):
    family, layout, y, plan, lambdas, fit, config, session = _interior_case(kind, "observed")
    if authority == "finite-difference":
        family = _GammaNoDirection()
    return family, layout, y, plan, lambdas, fit, session.dense_matrices(layout)


@pytest.mark.parametrize(
    ("kind", "authority"),
    [("gaussian", "analytic"), ("gamma", "analytic"), ("gamma", "finite-difference")],
)
def test_derivatives_match_the_legacy_assembly(kind, authority) -> None:
    """The frozen pre-optimization assembly is the oracle for every value.

    Gradient and Hessian within the legacy certificate plus 1e-10 of the matrix scale, on
    the analytic-first fixtures and on the finite-difference-first path the Tweedie cell runs.
    """
    family, layout, y, plan, lambdas, fit, dense = _legacy_case(kind, authority)
    new = laml_derivatives(family, layout, y, plan, lambdas=lambdas, fit=fit, dense_matrices=dense)
    old = legacy_laml_derivatives(
        family, layout, y, plan, lambdas=lambdas, fit=fit, dense_matrices=dense
    )
    assert new.names == old.names
    gradient_scale = np.max(np.abs(old.gradient))
    assert np.all(
        np.abs(new.gradient - old.gradient) <= old.gradient_certificate + 1e-10 * gradient_scale
    )
    assert np.all(np.abs(new.fs_gradient - old.fs_gradient) <= 1e-12 * gradient_scale)
    hessian_scale = np.max(np.abs(old.hessian))
    assert np.all(
        np.abs(new.hessian - old.hessian) <= old.hessian_certificate + 1e-10 * hessian_scale
    )


# --- Derivative-pass cost: row-chunked assembly of every X' V_k X ------------


def _random_blocks(seed: int = 5, n: int = 5000, widths=(40, 31, 22), m: int = 7):
    rng = np.random.default_rng(seed)
    matrices = tuple(rng.normal(size=(n, w)) for w in widths)
    starts = np.cumsum((0, *widths))
    slices = tuple(slice(int(a), int(b)) for a, b in zip(starts[:-1], starts[1:], strict=True))
    channels = len(widths) * (len(widths) + 1) // 2
    packed = rng.normal(size=(n, m, channels))
    return matrices, slices, packed, int(starts[-1])


def _reference_cross(matrices, slices, packed_k, width, *, absolute):
    """``X' diag-blocks(packed_k) X`` in the global layout, one channel at a time (pure numpy)."""
    result = np.zeros((width, width))
    for channel, (left, right) in enumerate(packed_pairs(len(matrices))):
        a = np.abs(matrices[left]) if absolute else matrices[left]
        b = np.abs(matrices[right]) if absolute else matrices[right]
        block = a.T @ (packed_k[:, channel, None] * b)
        result[slices[left], slices[right]] += block
        if left != right:
            result[slices[right], slices[left]] += block.T
    return 0.5 * (result + result.T)


def test_cross_all_matches_the_per_component_assembly() -> None:
    """K = 3 blocks of widths (40, 31, 22), n = 5,000, m = 7 packed weight arrays: the
    row-chunked all-at-once assembly equals the per-component one to 1e-12 relative.
    (The absolute-valued form the plan also named went with the entrywise certificate.)"""
    matrices, slices, packed, width = _random_blocks()
    got = _cross_all(matrices, slices, packed, width, chunk=1024)
    assert got.shape == (packed.shape[1], width, width)
    for k in range(packed.shape[1]):
        want = _reference_cross(matrices, slices, packed[:, k, :], width, absolute=False)
        assert np.max(np.abs(got[k] - want)) <= 1e-12 * np.max(np.abs(want))
        assert np.array_equal(got[k], got[k].T)


# --- Derivative-pass cost: single-direction polarisation, upper-triangle Hessian --


def test_single_direction_polarisation_matches_the_two_direction_mixed_difference() -> None:
    """``D4[e_q, e_r] = (D4[e_q+e_r, e_q+e_r] - D4[e_q, e_q] - D4[e_r, e_r]) / 2``.

    On the Gaussian closed-form fixture the single-direction mixed second difference
    agrees with the two-direction polarisation of
    ``finite_difference_curvature_second_direction`` within the sum of the two
    certificates, and with the closed form within its own; the Hessian stencil costs
    ``1 + 4K + 4K(K-1)/2`` evaluations (13 at K = 2) instead of ``1 + 4K + 8K(K-1)/2``.
    """
    y, eta = _gaussian_rows()
    family, plan, links = _gaussian_plan(y)

    def rows(values):
        return _curvature_packed(family, y, values, links, plan)

    first, _certificates, stencils, evaluations = _first_derivatives(
        rows, eta, DEFAULT_STEP, analytic_first=None
    )
    second, second_certificate, more = _second_derivatives(rows, eta, DEFAULT_STEP, stencils)
    assert evaluations + more == 1 + 4 * 2 + 4 * 1
    a = np.zeros((len(y), 2))
    a[:, 0] = 1.0
    b = np.zeros((len(y), 2))
    b[:, 1] = 1.0
    two = finite_difference_curvature_second_direction(family, y, eta, a, b, links, plan)
    floor = 64 * np.finfo(float).eps * np.maximum(np.abs(two.values), 1.0)
    assert np.all(
        np.abs(second[(0, 1)] - two.values) <= second_certificate[(0, 1)] + two.certificate + floor
    )
    want = _closed_form_second(y, eta, 0, 1)
    scale = np.maximum(np.abs(want), 1.0)
    assert np.max(np.abs(second[(0, 1)] - want) / scale) < 1e-7
    assert np.all(
        np.abs(second[(0, 1)] - want)
        <= second_certificate[(0, 1)] + 64 * np.finfo(float).eps * scale
    )


def test_hessian_is_formed_on_the_upper_triangle_and_mirrored() -> None:
    """The Hessian and its certificate are mirrored from ``k <= ell`` (exactly symmetric,
    not symmetrised) at ``1 + 4K + 4K(K-1)/2`` evaluations, against the legacy full square
    at ``1 + 4K + 8K(K-1)/2`` within the two certificates on the finite-difference path."""
    family, layout, y, plan, lambdas, fit, dense = _legacy_case("gamma", "finite-difference")
    new = laml_derivatives(family, layout, y, plan, lambdas=lambdas, fit=fit, dense_matrices=dense)
    assert np.array_equal(new.hessian, new.hessian.T)
    assert np.array_equal(new.hessian_certificate, new.hessian_certificate.T)
    k = fit.eta.shape[1]
    assert new.evaluations == 1 + 4 * k + 4 * k * (k - 1) // 2
    old = legacy_laml_derivatives(
        family, layout, y, plan, lambdas=lambdas, fit=fit, dense_matrices=dense
    )
    assert old.evaluations == 1 + 4 * k + 8 * k * (k - 1) // 2
    scale = np.max(np.abs(old.hessian))
    assert np.all(
        np.abs(new.hessian - old.hessian)
        <= new.hessian_certificate + old.hessian_certificate + 1e-10 * scale
    )


# --- Derivative-pass cost: a scalar Hessian certificate ----------------------


def test_hessian_certificate_bounds_the_error_against_the_legacy_reference() -> None:
    """On the Gamma finite-difference path the certificate at steps 1e-3 and 1e-9 covers
    the deviation from the legacy Hessian at step 1e-4 entrywise; at the default step the
    Hessian stays trusted (certificate under a tenth of the smallest diagonal) and at
    1e-9 it does not, so the endgame keeps failing over to BFGS there."""
    family, layout, y, plan, lambdas, fit, dense = _legacy_case("gamma", "finite-difference")
    reference = legacy_laml_derivatives(
        family, layout, y, plan, lambdas=lambdas, fit=fit, dense_matrices=dense, step=1e-4
    )
    fraction = DistributionalEFSConfig().hessian_certificate_fraction
    for step, trusted in ((1e-3, True), (1e-9, False)):
        new = laml_derivatives(
            family, layout, y, plan, lambdas=lambdas, fit=fit, dense_matrices=dense, step=step
        )
        assert np.all(np.abs(new.hessian - reference.hessian) <= new.hessian_certificate), step
        bar = fraction * np.min(np.diag(new.hessian))
        assert bool(np.max(new.hessian_certificate) < bar) is trusted, step


def test_hessian_certificate_costs_no_absolute_assembly(monkeypatch) -> None:
    """The certificate is a norm bound from scalars: one row-chunked assembly per Hessian
    pass, for the values, and none of the design by absolute value."""
    from superglm.distributional.smoothing import derivatives as derivatives_module

    calls: list[bool] = []
    real = derivatives_module._cross_all

    def spy(*args, **kwargs):
        calls.append(bool(kwargs.get("absolute", False)))
        return real(*args, **kwargs)

    monkeypatch.setattr(derivatives_module, "_cross_all", spy)
    family, layout, y, plan, lambdas, fit, dense = _legacy_case("gamma", "finite-difference")
    laml_derivatives(family, layout, y, plan, lambdas=lambdas, fit=fit, dense_matrices=dense)
    assert calls == [False]
