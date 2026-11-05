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
import warnings
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from superglm.distributional import GammaLS, Predictor
from superglm.distributional.families.gaussian import GaussianLS
from superglm.distributional.family import COMPLETE_OBSERVATION
from superglm.distributional.model import fit_dense_distributional
from superglm.distributional.results.iteration import DistributionalEFSConfig
from superglm.distributional.results.solver import DenseSolverConfig
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
        # The stop the oracle was calibrated at: the default start of 0.1 over
        # the covariate, 0.1 / hbar**3 in the knot intervals the penalties use.
        lambdas={
            f"{predictor.name}:{feature}#wiggle": 0.1
            * float((np.ptp(frame[feature]) / (spec.n_knots + 1)) ** -3)
            for predictor in predictors
            for feature, spec in predictor.features.items()
        },
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


def test_derivatives_refuse_a_compact_historical_fit() -> None:
    family, layout, y, plan, lambdas, fit, _config, session = _interior_case("gaussian", "fisher")
    with pytest.raises(LamlDerivativeError, match="retained predictor rows"):
        laml_derivatives(
            family,
            layout,
            y,
            plan,
            lambdas=lambdas,
            fit=replace(fit, eta=None, theta=None),
            dense_matrices=session.dense_matrices(layout),
        )


def test_config_derivative_step_default_is_the_finite_difference_step() -> None:
    """``results`` cannot import ``smoothing.endpoint_direction`` (import cycle); pin the value."""
    assert DistributionalEFSConfig().derivative_step == DEFAULT_STEP


# --- Derivative-pass cost: the pre-perf assembly is the oracle ---------------


class _GammaNoDirection(GammaLS):
    """GammaLS on the finite-difference path (the retained Tweedie cell's authority)."""

    predictor_curvature_directional_derivative = None  # type: ignore[assignment]
    predictor_curvature_second_directional_derivative = None  # type: ignore[assignment]


@pytest.mark.parametrize("kind", ["gaussian", "gamma"])
def test_analytic_fourth_derivatives_replace_the_second_difference(kind) -> None:
    """The Hessian's fourth-derivative term from the family's second curvature direction.

    One stacked call of ``predictor_curvature_second_directional_derivative``
    replaces the ``1 + 4K + 4K(K-1)/2`` packed-curvature evaluations of the
    second difference, so a Hessian pass after its gradient pass evaluates
    none; the Hessian agrees with the differenced one within the two
    certificates.
    """
    from superglm.distributional.smoothing.derivatives import LamlDerivativeWorkspace

    family, layout, y, plan, lambdas, fit, config, session = _interior_case(kind, "observed")
    dense = session.dense_matrices(layout)
    workspace = LamlDerivativeWorkspace()
    common = dict(lambdas=lambdas, fit=fit, dense_matrices=dense)
    laml_derivatives(family, layout, y, plan, want_hessian=False, reuse=workspace, **common)
    analytic = laml_derivatives(family, layout, y, plan, reuse=workspace, **common)
    assert analytic.evaluations == 0

    class _Differenced(type(family)):
        predictor_curvature_second_directional_derivative = None  # type: ignore[assignment]

    differenced_family = (
        _Differenced(scale_floor=family.scale_floor) if kind == "gaussian" else _Differenced()
    )
    numeric = laml_derivatives(differenced_family, layout, y, plan, **common)
    k = fit.eta.shape[1]
    assert numeric.evaluations == 1 + 4 * k + 4 * k * (k - 1) // 2
    scale = np.max(np.abs(numeric.hessian))
    assert np.all(
        np.abs(analytic.hessian - numeric.hessian)
        <= numeric.hessian_certificate + analytic.hessian_certificate + 1e-10 * scale
    )
    assert np.max(analytic.hessian_certificate) < np.max(numeric.hessian_certificate)


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


# --- Derivative-pass cost: the family's state once per pass, not once per axis --

_U = 2.0**-53


@pytest.mark.parametrize("blocked", [False, True], ids=["one block", "row blocks"])
@pytest.mark.parametrize("kind", ["gaussian", "gamma"])
def test_stacked_axis_directions_reproduce_the_per_axis_pass(kind, blocked, monkeypatch) -> None:
    """A family with ``stacked_curvature_directions`` takes the K axes in one call per row block.

    The same family without the flag takes one call per axis over all rows.
    Every row and direction runs the same arithmetic either way, so the
    gradient, Hessian and both certificates agree within ``8 u`` of their
    largest entry.
    """
    from superglm.distributional.kernels import _curvature_direction as curvature_kernels

    family, layout, y, plan, lambdas, fit, _config, session = _interior_case(kind, "observed")
    n, k = fit.eta.shape
    # Three blocks, the last one shorter, or the whole case in one block.
    chunk = n // 3 + 1 if blocked else curvature_kernels.STACKED_ROW_CHUNK
    assert blocked or n <= chunk
    monkeypatch.setattr(curvature_kernels, "STACKED_ROW_CHUNK", chunk)
    shapes: list[tuple[int, ...]] = []

    class _Counted(type(family)):
        def predictor_curvature_directional_derivative(self, y, eta, direction, links, plan):
            shapes.append(np.shape(direction))
            return super().predictor_curvature_directional_derivative(
                y, eta, direction, links, plan
            )

    class _PerAxis(_Counted):
        stacked_curvature_directions = False

    def make(cls):
        return cls(scale_floor=family.scale_floor) if kind == "gaussian" else cls()

    common = dict(lambdas=lambdas, fit=fit, dense_matrices=session.dense_matrices(layout))
    stacked = laml_derivatives(make(_Counted), layout, y, plan, **common)
    assert shapes == [(k, min(chunk, n - start), k) for start in range(0, n, chunk)]
    shapes.clear()
    per_axis = laml_derivatives(make(_PerAxis), layout, y, plan, **common)
    assert shapes == [(n, k)] * k
    for name in ("gradient", "gradient_certificate", "hessian", "hessian_certificate"):
        got, want = getattr(stacked, name), getattr(per_axis, name)
        assert np.all(np.abs(got - want) <= 8 * _U * np.max(np.abs(want))), name


def _stacked_family_cases():
    from superglm.distributional.families import (
        GeneralizedGammaLSS,
        GeneralizedParetoLSS,
        LogNormalLS,
        NegativeBinomialLS,
        TweedieLSS,
        TwoPieceLogNormalLSS,
        TwoPieceNormalLSS,
    )

    return {
        "gaussian": GaussianLS,
        "gamma": GammaLS,
        "log-normal (mean)": LogNormalLS,
        "log-normal (location)": lambda: LogNormalLS(parametrisation="location"),
        "negative binomial": NegativeBinomialLS,
        "tweedie": TweedieLSS,
        "generalized gamma (mean)": GeneralizedGammaLSS,
        "generalized gamma (location)": lambda: GeneralizedGammaLSS(parametrisation="location"),
        "generalized pareto": GeneralizedParetoLSS,
        "two-piece log-normal (mean)": TwoPieceLogNormalLSS,
        "two-piece log-normal (location)": lambda: TwoPieceLogNormalLSS(parametrisation="location"),
        "two-piece normal": TwoPieceNormalLSS,
    }


def _stacked_rows(family, n: int, rng):
    """A response and a predictor state inside the family's domain."""
    kind = type(family).__name__
    if kind in ("GaussianLS", "TwoPieceNormalLSS"):
        y = rng.normal(1.0, 1.0, n)
    elif kind == "NegativeBinomialLS":
        y = rng.poisson(2.0, n).astype(float)
    elif kind == "TweedieLSS":
        y = rng.gamma(2.0, 1.0, n) * (rng.uniform(size=n) < 0.6)
    else:
        y = np.exp(rng.normal(0.5, 0.6, n))
    eta = rng.normal(0.0, 0.3, (n, len(family.parameters)))
    if kind in ("GaussianLS", "TwoPieceNormalLSS"):
        eta[:, 0] += 1.0
    if kind == "GeneralizedGammaLSS":
        eta[:, 2] = rng.uniform(-0.1, 0.8, n)
    return y, eta


#: Families whose likelihood takes truncated and censored rows.
_BOUNDED_CASES = (
    "gamma",
    "gaussian",
    "generalized gamma (location)",
    "generalized gamma (mean)",
    "generalized pareto",
    "log-normal (location)",
    "log-normal (mean)",
)


def _bounded_observation(y, rng):
    """A fifth of the rows right-censored, some truncated below or above."""
    from superglm.distributional.family import bounded_observation
    from superglm.distributional.observation import ResponseBounds

    n = len(y)
    side = np.where(rng.uniform(size=n) < 0.2, 1, 0)
    lower = np.where(rng.uniform(size=n) < 0.3, y - 0.5 * np.abs(y) - 0.1, -np.inf)
    if np.all(y > 0.0):
        lower = np.where(np.isfinite(lower), np.maximum(lower, 0.0), lower)
    upper = np.where((rng.uniform(size=n) < 0.3) & (side == 0), y + np.abs(y) + 0.1, np.inf)
    return bounded_observation(ResponseBounds.from_arrays(lower, upper, side))


@pytest.mark.parametrize(
    ("name", "bounded"),
    [(name, False) for name in sorted(_stacked_family_cases())]
    + [(name, True) for name in _BOUNDED_CASES],
)
def test_stacked_curvature_directions_match_one_direction_at_a_time(name, bounded) -> None:
    """``(m, n, K)`` directions on one state evaluation equal ``m`` single calls.

    The K parameter axes the LAML pass asks for and three random directions,
    per family, on complete rows and, where the family takes them, on
    truncated and censored rows: within ``4 u`` of each direction's largest
    entry (values and, for NB2, its certificate).
    """
    from superglm.distributional.smoothing.endpoint_direction import (
        analytic_curvature_direction,
    )

    family = _stacked_family_cases()[name]()
    assert family.stacked_curvature_directions is True
    rng = np.random.default_rng(23)
    n = 400
    y, eta = _stacked_rows(family, n, rng)
    links = tuple(spec.default_link for spec in family.parameters)
    k = len(links)
    weights = resolve_likelihood_weights(None, n_observations=n, contract=WeightContract("prior"))
    observation = _bounded_observation(y, rng) if bounded else COMPLETE_OBSERVATION
    plan = family.bind_likelihood(y, weights, observation)
    directions = np.concatenate(
        [np.repeat(np.eye(k)[:, None, :], n, axis=1), rng.normal(size=(3, n, k))]
    )
    stacked, stacked_certificate = analytic_curvature_direction(
        family, y, eta, directions, links, plan
    )
    assert stacked.shape == (len(directions), n, k * (k + 1) // 2)
    for index, direction in enumerate(directions):
        single, certificate = analytic_curvature_direction(family, y, eta, direction, links, plan)
        assert np.all(np.abs(stacked[index] - single) <= 4 * _U * np.max(np.abs(single))), index
        if certificate is not None:
            assert np.all(
                np.abs(stacked_certificate[index] - certificate) <= 4 * _U * np.max(certificate)
            ), index


@pytest.mark.parametrize("bounded", [False, True], ids=["complete", "bounded"])
def test_generalized_pareto_natural_thirds_are_row_blocked(bounded, monkeypatch) -> None:
    """The Hessian pass's stacked natural thirds over row blocks equal one block over all rows.

    Each row's thirds use only that row, so blocks of 7 rows reproduce the
    single block within ``4 u`` of each pair's largest entry, values and
    certificate, on complete and on truncated and censored rows.
    """
    from superglm.distributional.families import GeneralizedParetoLSS
    from superglm.distributional.families import generalized_pareto as family_module
    from superglm.distributional.kernels import _curvature_direction as curvature_kernels
    from superglm.distributional.smoothing.derivatives import _analytic_second_derivatives
    from superglm.distributional.smoothing.endpoint_direction import DEFAULT_STEP

    family = GeneralizedParetoLSS()
    rng = np.random.default_rng(29)
    n = 40
    y, eta = _stacked_rows(family, n, rng)
    links = tuple(spec.default_link for spec in family.parameters)
    weights = resolve_likelihood_weights(None, n_observations=n, contract=WeightContract("prior"))
    observation = _bounded_observation(y, rng) if bounded else COMPLETE_OBSERVATION
    plan = family.bind_likelihood(y, weights, observation)
    shapes: list[tuple[int, ...]] = []
    original = family_module.scale_hessian_direction

    def counted(response, scale, shape, direction, multiplier):
        shapes.append(np.shape(direction))
        return original(response, scale, shape, direction, multiplier)

    monkeypatch.setattr(family_module, "scale_hessian_direction", counted)
    results = []
    for chunk in (n, 7):
        monkeypatch.setattr(curvature_kernels, "STACKED_ROW_CHUNK", chunk)
        shapes.clear()
        results.append(_analytic_second_derivatives(family, y, eta, links, plan, step=DEFAULT_STEP))
        assert shapes == [(2, min(chunk, n - start), 2) for start in range(0, n, chunk)]
    (whole_values, whole_certificate), (blocked_values, blocked_certificate) = results
    for pair, want in whole_values.items():
        assert np.all(np.abs(blocked_values[pair] - want) <= 4 * _U * np.max(np.abs(want))), pair
        bar = 4 * _U * np.max(whole_certificate[pair])
        assert np.all(np.abs(blocked_certificate[pair] - whole_certificate[pair]) <= bar), pair


# --- Binned layouts: every caller reads the rows from the layout's bin tables --

_LAML_RESULTS = ("gradient", "gradient_certificate", "hessian", "hessian_certificate")
# The group kinds each binned fixture holds.
_BINNED_KINDS = {
    "K2": {
        "DiscretizedTensorGroupMatrix",
        "DiscretizedSplineCategoricalGroupMatrix",
        "CategoricalGroupMatrix",
        "RandomEffectGroupMatrix",
        "DenseGroupMatrix",
    },
    "compressed": {
        "SupportCompressedSSPGroupMatrix",
        "SupportCompressedSplineCategoricalGroupMatrix",
        "CategoricalGroupMatrix",
        "DenseGroupMatrix",
    },
}
_BINNED_KINDS["K3"] = _BINNED_KINDS["K2"]
# Without a dense term the leverage reads its term-pair tables.
_BINNED_KINDS["K2-lookup"] = _BINNED_KINDS["K2"] - {"DenseGroupMatrix"}
_BINNED_KINDS["K3-lookup"] = _BINNED_KINDS["K2-lookup"]
_BINNED_KINDS["compressed-lookup"] = _BINNED_KINDS["compressed"] - {"DenseGroupMatrix"}


def _compressed_case(dense: bool = True):
    """An exact-mode fit on integer covariates, which compresses each spline to its supports."""
    from superglm.features import Categorical, Numeric
    from superglm.features.interaction import SplineCategorical

    n = 1500
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(
        {
            "a": rng.integers(18, 60, n).astype(float),
            "b": rng.integers(0, 12, n).astype(float),
            "d": rng.normal(size=n),
            "g": rng.choice(list("abc"), n),
            "h": rng.choice(list("pqrs"), n),
        }
    )
    y = (
        np.sin(frame["a"] / 9)
        + 0.2 * (frame["g"] == "b")
        + rng.normal(size=n) * np.exp(-0.5 + 0.05 * frame["b"])
    )
    predictors = [
        Predictor(
            "location",
            {
                "a": Spline(kind="cr", n_knots=6),
                "g": Categorical(base="a"),
                "h": Categorical(base="p"),
                **({"d": Numeric()} if dense else {}),
            },
            interaction_specs={"a:h": SplineCategorical("a", "h")},
        ),
        Predictor("scale", {"b": Spline(kind="cr", n_knots=5), "h": Categorical(base="p")}),
    ]
    return GaussianLS(), frame, y, predictors, False


@functools.cache
def _binned_case(case: str):
    """A binned fit: discretised, with a tensor, a categorical, a random effect, a level and a
    dense term (``K2``, ``K3``; ``-lookup``: no dense term), or exact mode on integer covariates
    (``compressed``)."""
    from superglm.distributional.families import GeneralizedGammaLSS
    from superglm.features import Categorical, Numeric, RandomEffect
    from superglm.features.interaction import SplineCategorical, TensorInteraction

    if case.startswith("compressed"):
        family, frame, y, predictors, discrete = _compressed_case(dense=case == "compressed")
        return _binned_fit(family, frame, y, predictors, discrete)
    k_parameters = int(case[1])
    n = 900
    rng = np.random.default_rng(61 + k_parameters)
    frame = pd.DataFrame(
        {
            "x": rng.uniform(-1, 1, n),
            "z": rng.uniform(-1, 1, n),
            "d": rng.normal(size=n),
            "g": rng.choice(list("abc"), n),
            "h": rng.choice(list("pq"), n),
            "r": rng.choice([f"l{i}" for i in range(12)], n),
        }
    )
    signal = np.sin(2 * frame["x"]) + 0.5 * frame["z"] * frame["x"] + 0.3 * (frame["g"] == "b")
    if k_parameters == 2:
        family = GaussianLS()
        y = signal + np.exp(-0.5 + 0.3 * frame["z"]) * rng.normal(size=n)
    else:
        family = GeneralizedGammaLSS()
        y = np.exp(0.4 * signal + 0.3 * rng.normal(size=n))
    names = [parameter.name for parameter in family.parameters]
    predictors = [
        Predictor(
            names[0],
            {
                "x": Spline(kind="cr", n_knots=6),
                "z": Spline(kind="cr", n_knots=5),
                "g": Categorical(base="a"),
                **({} if case.endswith("lookup") else {"d": Numeric()}),
                "r": RandomEffect(),
            },
            interaction_specs={"x:z": TensorInteraction("x", "z", n_knots=(4, 4))},
        ),
        Predictor(
            names[1],
            {"x": Spline(kind="cr", n_knots=5), "h": Categorical(base="p")},
            interaction_specs={"x:h": SplineCategorical("x", "h")},
        ),
    ]
    if k_parameters == 3:
        predictors.append(Predictor(names[2], {"z": Spline(kind="cr", n_knots=4)}))
    return _binned_fit(family, frame, y, predictors, True)


def _binned_fit(family, frame, y, predictors, discrete: bool):
    model = fit_dense_distributional(
        frame,
        np.asarray(y, dtype=np.float64),
        family=family,
        predictors=predictors,
        weight_contract=WeightContract("prior"),
        efs_config=DistributionalEFSConfig(max_iterations=4, outer="efs"),
        discrete=discrete,
        n_bins=24,
    )
    rows = model.fit_state.retained_rows
    plan = family.bind_likelihood(rows.response, rows.likelihood_weights, COMPLETE_OBSERVATION)
    options = dict(lambdas=dict(model.smoothing.lambdas), fit=model.smoothing.terminal_fit)
    return family, model.layout, rows.response, plan, options


def test_the_corrected_covariance_replay_builds_no_dense_design_on_a_binned_layout(
    monkeypatch,
) -> None:
    """The bin-table route reads its own rows, so the replay does not form the n x p design for it.

    An unchunked exact-mode fit on integer covariates is the route that formed it. The
    correction equals the one the dense matrices give, bit for bit. Mutation check:
    choosing the matrices by the chunk size alone builds the design once.
    """
    from superglm.distributional import posterior as posterior_module
    from superglm.distributional.solver.binned_rows import binned_layout

    family, frame, y, predictors, discrete = _compressed_case()
    model = fit_dense_distributional(
        frame,
        np.asarray(y, dtype=np.float64),
        family=family,
        predictors=predictors,
        weight_contract=WeightContract("prior"),
        efs_config=DistributionalEFSConfig(max_iterations=4, outer="efs"),
        discrete=discrete,
        n_bins=24,
    )
    assert binned_layout(model.layout)
    assert model.smoothing.terminal_fit.resolved_chunk_size is None
    dense_design = posterior_module.dense_predictor_matrices
    replay = posterior_module.laml_derivatives
    builds: list[bool] = []
    replays: list[bool] = []

    def counted_design(layout):
        builds.append(True)
        return dense_design(layout)

    def counted_replay(*args, **kwargs):
        replays.append(True)
        return replay(*args, **kwargs)

    monkeypatch.setattr(posterior_module, "dense_predictor_matrices", counted_design)
    monkeypatch.setattr(posterior_module, "laml_derivatives", counted_replay)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corrected = posterior_module.posterior_covariance(model, kind="corrected")
    assert replays == [True] and not builds

    monkeypatch.setattr(posterior_module, "binned_layout", lambda layout: False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from_dense = posterior_module.posterior_covariance(model, kind="corrected")
    assert replays == [True, True] and builds == [True]
    assert not np.array_equal(corrected, model.covariance)
    np.testing.assert_array_equal(corrected, from_dense)


def _binned_family(family, authority: str):
    if authority == "analytic":
        return family
    differenced = type(
        "Differenced",
        (type(family),),
        {
            "predictor_curvature_directional_derivative": None,
            "predictor_curvature_second_directional_derivative": None,
        },
    )
    return differenced()


@pytest.mark.parametrize("authority", ["analytic", "finite-difference"])
@pytest.mark.parametrize(
    "case", ["K2", "K3", "K2-lookup", "K3-lookup", "compressed", "compressed-lookup"]
)
def test_binned_layout_derivatives_equal_the_dense_row_chunks(case, authority, monkeypatch) -> None:
    """The passes form no rows, so they sum the dense row chunks' terms in another order: the
    gradient, Hessian and both certificates agree within rounding of the problem's scale. The
    fixtures hold every binned group kind, and the differenced pass nonzero certificates. The
    passes run over several row chunks, so a chunk read from the wrong rows shows."""
    from superglm.distributional._row_design import bounded_predictor_matrices
    from superglm.distributional.smoothing import derivatives as derivatives_module
    from superglm.distributional.solver.binned_rows import binned_layout

    monkeypatch.setattr(derivatives_module, "_ROW_CHUNK", 97)
    family, layout, y, plan, options = _binned_case(case)
    family = _binned_family(family, authority)
    assert binned_layout(layout)
    kinds = {type(g).__name__ for state in layout.predictors for g in state.design.group_matrices}
    assert kinds >= _BINNED_KINDS[case]
    matrices = bounded_predictor_matrices(layout)
    binned = laml_derivatives(family, layout, y, plan, dense_matrices=matrices, **options)
    monkeypatch.setattr(derivatives_module, "binned_rows", lambda layout: None)
    dense = laml_derivatives(family, layout, y, plan, dense_matrices=matrices, **options)
    assert binned.third_derivative_authority == dense.third_derivative_authority
    assert bool(np.max(binned.hessian_certificate) > 1e-9) is (authority == "finite-difference")
    for name in (*_LAML_RESULTS, "fs_gradient", "penalty_hessian"):
        reference = getattr(dense, name)
        tolerance = 2.0**-53 * 64 * layout.n_coefficients * np.max(np.abs(reference), initial=1.0)
        np.testing.assert_allclose(
            getattr(binned, name), reference, rtol=0.0, atol=tolerance, err_msg=name
        )


@pytest.mark.parametrize("case", ["K2-lookup", "K3-lookup"])
def test_row_blocks_of_the_family_derivatives_change_no_bit(case, monkeypatch) -> None:
    """A stacked family's third and fourth derivatives run in row blocks, which bound their
    scratch; its rows are independent, so the blocks leave the results as one call gives them."""
    from superglm.distributional._row_design import bounded_predictor_matrices
    from superglm.distributional.kernels import _curvature_direction as curvature_kernels

    family, layout, y, plan, options = _binned_case(case)
    assert family.stacked_curvature_directions is True
    matrices = bounded_predictor_matrices(layout)
    whole = laml_derivatives(family, layout, y, plan, dense_matrices=matrices, **options)
    monkeypatch.setattr(curvature_kernels, "STACKED_ROW_CHUNK", 97)
    blocks = laml_derivatives(family, layout, y, plan, dense_matrices=matrices, **options)
    for name in _LAML_RESULTS:
        np.testing.assert_array_equal(getattr(blocks, name), getattr(whole, name), name)


@pytest.mark.parametrize(
    "case", ["K2", "K3", "K2-lookup", "K3-lookup", "compressed", "compressed-lookup"]
)
def test_binned_layout_derivatives_never_read_the_callers_rows(case, monkeypatch) -> None:
    """The layout decides the rows: materialised matrices and bounded chunks of 97 rows (whose
    dense route rounds differently) give the same bits, and no bounded chunk is formed."""
    from superglm.distributional._row_design import (
        BoundedPredictorMatrix,
        bounded_predictor_matrices,
    )
    from superglm.distributional.solver.assembly import dense_predictor_matrices

    family, layout, y, plan, options = _binned_case(case)
    family = _binned_family(family, "finite-difference")
    materialised = laml_derivatives(
        family, layout, y, plan, dense_matrices=dense_predictor_matrices(layout), **options
    )

    def forbidden(self, rows):
        pytest.fail("a binned layout's derivative pass read the caller's bounded rows")

    monkeypatch.setattr(BoundedPredictorMatrix, "__getitem__", forbidden)
    bounded = bounded_predictor_matrices(layout, chunk_size=97)
    chunked = laml_derivatives(family, layout, y, plan, dense_matrices=bounded, **options)
    for name in _LAML_RESULTS:
        np.testing.assert_array_equal(getattr(chunked, name), getattr(materialised, name), name)
