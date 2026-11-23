"""Coefficient stationarity under changes of origin and units (#576)."""

import warnings
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from superglm import GaussianLS, SuperLSS
from superglm._frame import as_eager_frame
from superglm.diagnostics.convergence import ConvergenceWarning
from superglm.distributional.family import COMPLETE_OBSERVATION
from superglm.distributional.layout import build_stacked_layout
from superglm.distributional.predictor import Predictor, compile_predictors
from superglm.distributional.results.solver import (
    RetainedCoordinates,
    _assessment_retained_kkt_ratio,
)
from superglm.distributional.smoothing.penalty_face import build_penalty_face
from superglm.distributional.solver import solver
from superglm.distributional.solver.shape import _face_of
from superglm.features import Numeric, RandomEffect, Spline
from superglm.group_matrix import DesignMatrix, SparseGroupMatrix
from superglm.solvers.rank import RankDecomposition

from ._distributional_weights import resolved_prior
from .test_distributional_endpoint_laml import _UnitGaussian


@pytest.mark.parametrize("force_fallback", [False, True], ids=["observed", "fisher"])
def test_a_material_gain_is_measured_with_the_accepted_factor(monkeypatch, force_fallback):
    """A weakly weighted Gaussian can still gain 0.5. On fallback give the
    rejected observed matrix a misleading eigenvalue.
    """
    family = _UnitGaussian()
    weights = resolved_prior(np.array([1e-16]))
    y = np.array([1e8])
    layout = build_stacked_layout(
        compile_predictors(
            as_eager_frame(pd.DataFrame({"x": [0.0]})),
            weights,
            family.parameters,
            (Predictor("mean", {}),),
        )
    )
    resolve = solver.resolve_curvature

    def fallback(source, matrix, **kwargs):
        decision = resolve(source, -matrix, **kwargs)
        return replace(decision, telemetry=replace(decision.telemetry, minimum_eigenvalue=1.0))

    if force_fallback:
        monkeypatch.setattr(solver, "resolve_curvature", fallback)
    fit = solver.fit_dense_fixed_lambda(
        family,
        layout,
        y,
        family.bind_likelihood(y, weights, COMPLETE_OBSERVATION),
        np.zeros((1, 1)),
        initial=np.zeros(1),
        config=solver.DenseSolverConfig(tolerance=1e-7, coefficient_curvature="fisher"),
    )
    assert fit.terminal_curvature.actual_source == ("fisher" if force_fallback else "observed")
    assert not fit.converged
    assert fit.convergence_reason == "material_likelihood_gain"


@pytest.mark.parametrize("chunk_size", [None, 11], ids=["dense", "chunked"])
@pytest.mark.pins_behaviour
def test_a_whitened_spline_resolves_a_contrast_on_its_rare_rows(chunk_size):
    """Pin the Gaussian least-squares predictions for a score above tolerance
    in the compiler's unit SSP coordinates. A rare large row must not hide it.
    """
    x = np.r_[np.linspace(0, 0.2, 256), np.linspace(0.2, 1, 24)]
    weights = resolved_prior(np.ones(len(x)))
    family = _UnitGaussian()
    layout = build_stacked_layout(
        compile_predictors(
            as_eager_frame(pd.DataFrame({"x": x})),
            weights,
            family.parameters,
            (Predictor("mean", {"x": Spline(kind="cr", k=6)}),),
        )
    )
    (design,) = solver.dense_predictor_matrices(layout)
    radius = np.ptp(design, axis=0) / 2
    j = np.argmax(radius)
    assert radius[j] > 2
    # Choose y in the column space with just this slope's initial score.
    tol = 1e-5
    score = np.zeros(design.shape[1])
    score[j] = 2 * tol
    y = np.linalg.lstsq(design.T, score, rcond=None)[0]
    fit = solver.fit_dense_fixed_lambda(
        family,
        layout,
        y,
        family.bind_likelihood(y, weights, COMPLETE_OBSERVATION),
        np.zeros((len(score), len(score))),
        initial=np.zeros_like(score),
        config=solver.DenseSolverConfig(tolerance=tol),
        chunk_size=chunk_size,
    )
    assert fit.converged
    np.testing.assert_allclose(fit.eta[:, 0], y, rtol=0, atol=1e-7 * np.max(np.abs(y)))


@pytest.mark.pins_behaviour
def test_well_conditioned_terminal_score_needs_no_extra_solve(monkeypatch):
    """Pin the solve count after the rank decision on this Gaussian with unit design spread."""
    z = np.tile([-1.0, -1.0, 1.0, 1.0], 16)
    y = 0.25 * z + np.tile([-1.0, 1.0], 32)
    certified = solver._certified_terminal_curvature
    solve = RankDecomposition.solve
    factors = []
    extra_solves = []

    def terminal(*args):
        decision = certified(*args)
        factors.append(decision.decomposition)
        return decision

    def counted(rank, rhs):
        if any(rank is factor for factor in factors):
            extra_solves.append(rank)
        return solve(rank, rhs)

    monkeypatch.setattr(solver, "_certified_terminal_curvature", terminal)
    monkeypatch.setattr(RankDecomposition, "solve", counted)
    X = as_eager_frame(pd.DataFrame({"x": z}))
    family = GaussianLS(scale_floor=0)
    weights = resolved_prior(np.ones(len(z)))
    plan = family.bind_likelihood(y, weights, COMPLETE_OBSERVATION)
    predictors = (Predictor("location", {"x": Numeric()}), Predictor("scale", {}))
    layout = build_stacked_layout(compile_predictors(X, weights, family.parameters, predictors))
    # The analytic MLE with a sub-tolerance, nonzero perturbation of the slope.
    fit = solver.fit_dense_fixed_lambda(
        family, layout, y, plan, np.zeros((3, 3)), initial=np.array([0.0, 0.25 + 1e-9, 0.0])
    )
    assert fit.converged
    np.testing.assert_allclose(fit.eta[:, 0], 0.25 * z, atol=1e-7)
    assert factors and not extra_solves


def test_a_small_resolvable_correction_is_taken_in_either_unit(monkeypatch):
    """A small gain is resolvable: the score is 1e-4 in predictor units, with no penalty floor."""
    z = np.tile([-1.0, -1.0, 1.0, 1.0], 16)
    y = 1e-4 * z + np.tile([-1.0, 1.0], 32)
    monkeypatch.setattr(
        solver, "_initial_coefficients", lambda ctx: np.zeros(ctx.layout.n_coefficients)
    )
    for scale in (1.0, 1e-9):
        X = pd.DataFrame({"x": scale * z})
        family = GaussianLS(scale_floor=0)
        model = SuperLSS(family, family.location("x"), family.scale()).fit(X, y)
        assert model.result_.converged
        np.testing.assert_allclose(model.predict(X), 1e-4 * z, rtol=0, atol=1e-7)


def test_a_newton_step_has_the_same_size_in_shifted_coordinates(monkeypatch):
    z = np.tile([-1.0, -1.0, 1.0, 1.0], 16)
    y = 0.25 * z + np.tile([-1.0, 1.0], 32)
    monkeypatch.setattr(
        solver, "_initial_coefficients", lambda ctx: np.zeros(ctx.layout.n_coefficients)
    )
    steps = []
    for x in (z, 8.0 - 0.125 * z):
        family = GaussianLS(scale_floor=0)
        model = SuperLSS(
            family, family.location("x"), family.scale(), coefficient_curvature="fisher"
        )
        model.fit(pd.DataFrame({"x": x}), y)
        steps.append(model._require_fitted().fit_state.solver_result.history[0].step_relative)
    assert steps[0] == pytest.approx(0.25 / 1.25)
    assert steps[1] == pytest.approx(steps[0])


@pytest.mark.parametrize("discrete", [False, True], ids=["dense", "chunked"])
def test_a_reduced_fit_must_still_fit_the_variation_in_a_dropped_column(monkeypatch, discrete):
    """Force the refit of a shifted slope; its multiplier must reject the intercept-only MLE."""
    rng = np.random.default_rng(11)
    z = rng.normal(size=3000)
    y = 1.0 + 0.7 * z + rng.normal(0.0, 0.5, len(z))
    origin, unit = 3e7, 1e-7
    design = np.column_stack((np.ones(len(z)), z))
    level, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    log_scale = np.log(np.std(y - design @ [level, slope]))
    start = solver._initial_coefficients

    def at_optimum(context):
        if context.layout.n_coefficients == 3:
            return np.array([level - origin * slope, slope / unit, log_scale])
        return start(context)

    monkeypatch.setattr(solver, "_initial_coefficients", at_optimum)
    monkeypatch.setattr(solver, "_drops_design_aliases", lambda *_: True)
    family = GaussianLS(scale_floor=0)
    X = pd.DataFrame({"x": unit * (origin + z)})
    model = SuperLSS(family, family.location("x"), family.scale(), discrete=discrete).fit(X, y)
    np.testing.assert_allclose(model.predict(X), level + slope * z, rtol=0, atol=1e-6)


@pytest.mark.parametrize("discrete", [False, True], ids=["dense", "chunked"])
@pytest.mark.parametrize(
    "shift,scale,response_unit", [(3e7, 1.1e-7, 1), (0.0, 1e-10, 1), (0, 1, 1e10)]
)
def test_gaussian_fit_reaches_the_centred_mle_or_discloses_the_stop(
    discrete, shift, scale, response_unit
):
    rng = np.random.default_rng(4)
    z = rng.normal(size=3000)
    y = response_unit * (1 + 0.7 * z + rng.normal(0, 0.5, len(z)))
    x = scale * (shift + z)
    X = pd.DataFrame({"x": x})
    # Compute the Gaussian MLE on the actual float64 inputs in centred units.
    design = np.column_stack((np.ones(len(x)), (x - x.mean()) / x.std()))
    mu = design @ np.linalg.lstsq(design, y, rcond=None)[0]
    sigma = np.sqrt(np.mean((y - mu) ** 2))
    log_likelihood = -len(y) * (0.5 * (1.0 + np.log(2 * np.pi)) + np.log(sigma))
    family = GaussianLS()
    model = SuperLSS(family, family.location("x"), family.scale(), discrete=discrete)
    tol = 1e-7
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(X, y, inner_tol=tol)
    if shift == 0 and response_unit == 1:
        # The centred Gaussian problem is well conditioned after equilibration.
        assert model.result_.converged
    if not model.result_.converged:
        assert any(issubclass(item.category, ConvergenceWarning) for item in caught)
        return
    assert abs(model.result_.log_likelihood - log_likelihood) <= tol * (1 + abs(log_likelihood))
    np.testing.assert_allclose(model.predict(X), mu, atol=tol * (1 + np.max(abs(mu))), rtol=0)


@pytest.mark.parametrize("space", ["full", "face", "retained"])
@pytest.mark.parametrize("intercept", [True, False])
@pytest.mark.parametrize("chunked", [False, True])
def test_same_direction_has_the_same_score_in_different_units(
    space, intercept, chunked, monkeypatch
):
    """A nonzero intercept score must be removed from the shifted slope's score."""
    # Both extrema lie in the last partial seven-row chunk.
    z = np.zeros(32)
    z[-2:] = [-1.0, 1.0]
    y = np.arange(len(z), dtype=float)
    weights = resolved_prior(np.ones(len(z)))
    family = GaussianLS()
    plan = family.bind_likelihood(y, weights, COMPLETE_OBSERVATION)
    session = solver._DenseObservedReuseSession()
    matvec = SparseGroupMatrix.matvec

    def bounded_product(group, coefficients):
        assert group.shape[0] <= 7
        return matvec(group, coefficients)

    monkeypatch.setattr(SparseGroupMatrix, "matvec", bounded_product)

    def reading(a, b):
        X = as_eager_frame(pd.DataFrame({"x": a * z + b, "level": np.arange(len(z)) % 3}))
        predictors = (
            Predictor("location", {"x": Numeric(), "level": RandomEffect()}, intercept=intercept),
            Predictor("scale", {}),
        )
        layout = build_stacked_layout(compile_predictors(X, weights, family.parameters, predictors))
        if chunked:
            location, scale = layout.predictors
            numeric, effect = location.design.group_matrices
            numeric = SparseGroupMatrix(sparse.csr_matrix(numeric.toarray()))
            design = DesignMatrix([numeric, effect], len(z), location.design.p)
            layout = replace(layout, predictors=(replace(location, design=design), scale))
        face = build_penalty_face(layout, layout.penalty_names) if space == "face" else None
        context = solver._validated_context(
            family,
            layout,
            y,
            plan,
            layout.penalty_matrix({layout.penalty_names[0]: 1.0}),
            coefficient_curvature="observed",
            chunk_size=7 if chunked else None,
            coefficient_face=face,
            _reuse_session=session,
        )
        score = np.zeros(layout.n_coefficients)
        if intercept:
            score[0] = 3.0
        score[int(intercept)] = a * 7.0 + b * 3.0
        drop = np.zeros(len(score), dtype=bool)
        drop[int(intercept) + 1 : -1] = True
        if space == "full":
            indicator_score = np.zeros_like(score)
            indicator_score[int(intercept) + 1] = 1.0
            assert solver._optimization_score_relative(context, indicator_score, 0.0) == (
                2.0 if intercept else 1.0
            )
        if space != "full":
            # A component normal to the feasible directions cannot decide this stop.
            score[drop] = 1e5
        if space == "retained":
            context = replace(context, retained=RetainedCoordinates(drop))
        return solver._optimization_score_relative(context, score, -6.0)

    assert reading(1.0, 0.0) == pytest.approx(1.0)
    assert reading(-0.125, 8.0 if intercept else 0.0) == pytest.approx(1.0)


def test_endpoint_rechecks_the_tangential_score_in_predictor_units():
    """Two slopes tied in predictor units have projected score (3, 3) in either encoding."""
    z = np.tile([-1.0, 1.0], 8)
    v = np.tile([-1.0, -1.0, 1.0, 1.0], 4)
    y = np.repeat([-1.0, 1.0], 8)
    family = GaussianLS(scale_floor=0)
    weights = resolved_prior(np.ones(len(z)))
    plan = family.bind_likelihood(y, weights, COMPLETE_OBSERVATION)
    for a, b in [(1.0, 0.0), (-0.125, 8.0)]:
        X = as_eager_frame(pd.DataFrame({"x": a * z + b, "v": v}))
        predictors = (
            Predictor("location", {"x": Numeric(), "v": Numeric()}),
            Predictor("scale", {}),
        )
        layout = build_stacked_layout(compile_predictors(X, weights, family.parameters, predictors))
        face = _face_of(
            np.array([[0.0, a, -1.0, 0.0]]), np.zeros(4), None, layout.coefficient_names
        )
        fit = solver.fit_dense_fixed_lambda(
            family, layout, y, plan, np.zeros((4, 4)), initial=np.zeros(4), coefficient_face=face
        )
        # Endpoint authority must recompute the ratio when the supplied terminal score changes.
        fit = replace(fit, terminal_score=np.array([1.0, a * 7.0 + b, -1.0, 0.0]))
        assert _assessment_retained_kkt_ratio(fit) == pytest.approx(
            3.0 / (1.0 + abs(fit.penalized_optimizing_log_likelihood))
        )
