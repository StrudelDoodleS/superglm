"""The one-pass leverage route for the REML weight-derivative terms.

``reml_w_correction(leverage_hessian=True)`` forms the W(rho) gradient
correction and the outer Hessian's first-order cross terms from one pass of
solved rows ``H^-1 x_r`` (``reml/_leverage_rows.py``), in place of one
``p x p`` derivative Gram per smoothing parameter.  These tests pin its
values against the Gram route within a bound derived in the unit roundoff, on
both row producers, and its cost: one solved-row pass per REML iteration,
whatever the number of smoothing parameters.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

import superglm.reml.direct as direct
import superglm.reml.w_derivatives as w_derivatives
from superglm import Categorical, FactorSmooth, Numeric, Spline, SuperGLM
from superglm.distributions import Poisson
from superglm.group_matrix import CategoricalGroupMatrix, DenseGroupMatrix, DesignMatrix
from superglm.links import LogLink
from superglm.reml.penalty_algebra import (
    penalty_component_dense_matrix,
    penalty_component_matvec,
)
from superglm.solvers.pirls import PIRLSResult, REMLGeometrySummary
from superglm.types import GroupSlice, PenaltyComponent

_U = np.finfo(np.float64).eps / 2.0


def _gamma(k: int) -> float:
    return k * _U / (1.0 - k * _U)


def _poisson_frame(n: int, offset: float, seed: int = 31):
    rng = np.random.default_rng(seed)
    # Coarse covariates: every spline term has a small indexed support, so the
    # support route serves the design (one row of its tables per support row).
    frame = pd.DataFrame(
        {
            "a": np.round(rng.uniform(0.0, 1.0, n), 1),
            "b": np.round(rng.uniform(-2.0, 2.0, n), 1),
            "e": np.round(rng.uniform(0.0, 3.0, n), 1),
            "c": rng.integers(0, 5, n).astype(str),
            "d": rng.normal(offset, 4.0, n),
        }
    )
    eta = (
        -0.8
        + np.sin(3.0 * frame.a)
        + 0.25 * frame.b**2
        + 0.15 * (frame.c.astype(int) % 3)
        + 0.02 * (frame.d - offset)
        + 0.2 * np.cos(frame.e)
    )
    weights = rng.uniform(0.3, 1.0, n)
    y = rng.poisson(weights * np.exp(eta)) / weights
    return frame, y, weights


def _poisson_model(offset: float) -> SuperGLM:
    frame, y, weights = _poisson_frame(1500, offset)
    return SuperGLM(
        family="poisson",
        features={
            "a": Spline(n_knots=6),
            "b": Spline(n_knots=6),
            "e": Spline(n_knots=5),
            "c": Categorical(),
            "d": Numeric(),
        },
        interactions=[("a", "b"), ("e", "c")],
        weight_semantics="frequency",
        discrete=False,
    ).fit_reml(frame, y, sample_weight=weights, runtime_validation="skip")


def _correction_inputs(model: SuperGLM) -> tuple[tuple, dict]:
    inverse = np.asarray(model._fit_inference_info["XtWX_inv"], dtype=np.float64)
    args = (model._dm, model._link, model._groups, model.result, inverse, model._reml_lambdas)
    kwargs = {
        "sample_weight": np.asarray(model._fit_weights, dtype=np.float64),
        "offset_arr": np.zeros(model._dm.n),
        "distribution": model._distribution,
        "reml_penalties": model._reml_penalties,
    }
    return args, kwargs


def _rounding_bounds(model: SuperGLM) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reference terms from the Gram route and the bounds both routes keep, in ``u``.

    Each term is a sum over rows of products whose magnitudes the test forms
    from absolute values: with ``M_r = |x_r| + |c_raw|`` (the centred row, plus
    the centre on the indexed groups the support route reads raw),
    ``|a|_jr = |w'_r| M_r' |H^-1| |s_j|`` (the weight change and the scale of
    its rounding) and ``G_i = |H^-1| |S_i| |H^-1|``, the gradient term ``j`` is
    within ``gamma_K E_j`` of its exact value on either route, ``E_j = sum_r
    |a|_jr (M_r' |H^-1| M_r + 1/sum_w) / 2``, and the cross term ``P_ij`` within
    ``gamma_K F_ij``, ``F_ij = sum_r (M_r' G_i M_r) |a|_jr``: products, solves
    and sums of at most ``K = n + 4p + 16`` roundings along each chain (Higham
    2002, secs. 3.1, 3.5).  Two routes differ by at most twice that.
    """
    args, kwargs = _correction_inputs(model)
    gradient, extra = w_derivatives.reml_w_correction(*args, **kwargs)
    dm, link, inverse = model._dm, model._link, args[4]
    penalties, lambdas = model._reml_penalties, model._reml_lambdas
    n, p, m = dm.n, dm.p, len(penalties)
    summary = model.result.reml_geometry
    dense = w_derivatives.dense_columns(dm)
    design = dm.toarray()
    centred = design - np.where(dense, design.mean(axis=0), np.asarray(summary.mean_x))
    magnitude = np.abs(centred) + np.where(dense, 0.0, np.abs(summary.mean_x))
    eta = design @ model.result.beta + model.result.intercept
    mu = link.inverse(eta)
    weight_derivative = w_derivatives.compute_dW_deta(
        link, model._distribution, mu, eta, kwargs["sample_weight"]
    )
    smoothing = np.zeros((p, p, m))
    penalty_rhs = np.zeros((p, m))
    for i, pc in enumerate(penalties):
        block = lambdas[pc.name] * penalty_component_dense_matrix(
            pc, dm.group_matrices[pc.group_index]
        )
        smoothing[pc.group_sl, pc.group_sl, i] = block
        penalty_rhs[pc.group_sl, i] = lambdas[pc.name] * penalty_component_matvec(
            pc, model.result.beta[pc.group_sl], dm.group_matrices[pc.group_index]
        )
    absolute_inverse = np.abs(inverse)
    weight_scale = np.abs(weight_derivative)[:, None] * (
        magnitude @ absolute_inverse @ np.abs(penalty_rhs)
    )
    leverage_scale = np.einsum("ij,jk,ik->i", magnitude, absolute_inverse, magnitude)
    gradient_scale = 0.5 * weight_scale.T @ (leverage_scale + 1.0 / summary.sum_w)
    quadratic_scale = np.stack(
        [
            np.einsum(
                "ij,jk,ik->i",
                magnitude,
                absolute_inverse @ np.abs(smoothing[:, :, i]) @ absolute_inverse,
                magnitude,
            )
            for i in range(m)
        ],
        axis=1,
    )
    cross_scale = quadratic_scale.T @ weight_scale
    k = n + 4 * p + 16
    hessian_terms = []
    for i in range(m):
        left = inverse @ smoothing[:, :, i]
        hessian_terms.append([float(np.sum(left * (inverse @ extra[j]).T)) for j in range(m)])
    cross = np.asarray(hessian_terms)
    reference_cross = -0.5 * (cross + cross.T)
    gradient_bound = 2.0 * _gamma(k) * gradient_scale
    cross_bound = _gamma(k) * (cross_scale + cross_scale.T)
    return gradient, reference_cross, gradient_bound, cross_bound


@pytest.mark.parametrize("offset", [1.0e3, 1.0e9])
@pytest.mark.parametrize("producer", ["support", "chunked"])
def test_leverage_terms_match_derivative_grams_within_rounding(monkeypatch, producer, offset):
    """Both row producers reproduce the Gram route's gradient and cross terms.

    The bound is in ``u`` (``_rounding_bounds``) and sharp: it is below a
    millionth of the terms, so a dropped or mis-weighted term fails it.
    """
    import superglm.reml._leverage_rows as leverage_rows

    model = _poisson_model(offset)
    gradient, reference_cross, gradient_bound, cross_bound = _rounding_bounds(model)
    planned = []
    original_plan = leverage_rows._support_plan

    def plan(*args, **kwargs):
        result = original_plan(*args, **kwargs) if producer == "support" else None
        planned.append(result is not None)
        return result

    monkeypatch.setattr(leverage_rows, "_support_plan", plan)
    args, kwargs = _correction_inputs(model)
    leverage = w_derivatives.reml_w_correction(*args, leverage_hessian=True, **kwargs)

    assert planned == [producer == "support"]
    assert leverage is not None and leverage[1] is None
    assert np.all(np.abs(leverage[0] - gradient) <= gradient_bound)
    assert np.all(np.abs(leverage[2] - reference_cross) <= cross_bound)
    assert np.max(gradient_bound) < 1e-6 * np.max(np.abs(gradient))
    assert np.max(cross_bound) < 1e-6 * np.max(np.abs(reference_cross))


def _outlier_case() -> dict:
    """A dense column with a ``1e200`` row that carries no weight, beside a categorical."""
    rng = np.random.default_rng(9411)
    n = 120
    columns = rng.normal(size=(n, 3))
    columns[0, 0] = 1.0e200
    sample_weight = np.ones(n)
    sample_weight[0] = 0.0
    codes = rng.integers(0, 4, n)
    level = CategoricalGroupMatrix(np.where(codes == 3, -1, codes), 3)
    beta = np.array([0.1, -0.07, 0.05, 0.2, -0.1, 0.05])
    link = LogLink()
    one_hot = level.toarray()
    eta = columns[:, :3] @ beta[:3] + one_hot @ beta[3:] - 0.4
    eta[0] = -0.4
    mu = link.inverse(eta)
    active = sample_weight > 0.0
    weights = sample_weight * mu
    sum_w = float(np.sum(weights[active]))
    design = np.column_stack((columns, one_hot))
    mean_x = np.sum(weights[active, None] * design[active], axis=0) / sum_w
    centred = design[active] - mean_x
    lambdas = {"x0": 1.0, "x1": 2.0, "x2": 3.0, "lvl": 0.5}
    penalty = np.diag([1.0, 2.0, 3.0, 0.5, 0.5, 0.5])
    inverse = np.linalg.inv(centred.T @ (weights[active, None] * centred) + penalty)
    groups = [GroupSlice(name=f"x{i}", start=i, end=i + 1) for i in range(3)]
    groups.append(GroupSlice(name="lvl", start=3, end=6))
    penalties = [
        PenaltyComponent(
            name=group.name,
            group_name=group.name,
            group_index=i,
            group_sl=group.sl,
            omega_raw=None,
            penalty_kind="identity",
        )
        for i, group in enumerate(groups)
    ]
    dm = DesignMatrix(
        [DenseGroupMatrix(columns[:, i : i + 1]) for i in range(3)] + [level], n=n, p=6
    )
    result = PIRLSResult(
        beta=beta,
        intercept=-0.4,
        n_iter=1,
        deviance=0.0,
        converged=True,
        phi=1.0,
        effective_df=6.0,
        reml_geometry=REMLGeometrySummary(
            mean_x=mean_x,
            sum_w=sum_w,
            column_scale=np.ones(6),
        ),
    )
    return {
        "dm": dm,
        "link": link,
        "groups": groups,
        "pirls_result": result,
        "XtWX_S_inv": inverse,
        "lambdas": lambdas,
        "sample_weight": sample_weight,
        "offset_arr": np.zeros(n),
        "distribution": Poisson(),
        "reml_penalties": penalties,
    }


@pytest.mark.parametrize("producer", ["support", "chunked"])
def test_leverage_route_never_forms_a_row_without_weight(monkeypatch, producer):
    """A ``1e200`` row with zero weight derivative must not overflow into the terms."""
    import superglm.reml._leverage_rows as leverage_rows

    if producer == "chunked":
        monkeypatch.setattr(leverage_rows, "_support_plan", lambda *args, **kwargs: None)
    case = _outlier_case()

    leverage = w_derivatives.reml_w_correction(**case, leverage_hessian=True)

    assert leverage is not None
    assert np.all(np.isfinite(leverage[0])) and np.all(np.isfinite(leverage[2]))


def _count_row_passes(monkeypatch) -> dict:
    """Count W-correction calls, derivative Grams and solved-row passes during a fit."""
    counts = {"calls": 0, "grams": 0, "passes": 0}
    correction = direct.reml_w_correction

    def counted_correction(*args, **kwargs):
        counts["calls"] += 1
        return correction(*args, **kwargs)

    gram = w_derivatives.centered_gram_rhs
    grams = w_derivatives.centered_signed_grams

    def counted_gram(*args, **kwargs):
        counts["grams"] += 1
        return gram(*args, **kwargs)

    def counted_grams(*args, **kwargs):
        counts["grams"] += len(kwargs["weights"])
        return grams(*args, **kwargs)

    passes = getattr(w_derivatives, "solved_row_blocks", None)

    def counted_passes(*args, **kwargs):
        counts["passes"] += 1
        return passes(*args, **kwargs)

    monkeypatch.setattr(direct, "reml_w_correction", counted_correction)
    monkeypatch.setattr(w_derivatives, "centered_gram_rhs", counted_gram)
    monkeypatch.setattr(w_derivatives, "centered_signed_grams", counted_grams)
    monkeypatch.setattr(w_derivatives, "solved_row_blocks", counted_passes, raising=False)
    return counts


@pytest.mark.parametrize("pairs", [1, 2])
def test_weight_correction_makes_one_row_pass_per_reml_iteration(monkeypatch, pairs):
    """The cost model: one solved-row pass per REML iteration, no derivative Gram.

    The Gram route forms one ``p x p`` Gram per smoothing parameter per
    iteration (``m`` per call), so this fails there; here the count is the
    number of iterations whatever ``m`` is.
    """
    # A small fixture: admit it below the work bar that keeps cheap Grams.
    monkeypatch.setattr(direct, "_LEVERAGE_HESSIAN_MIN_GRAM_WORK", 0.0, raising=False)
    counts = _count_row_passes(monkeypatch)
    frame, y, weights = _poisson_frame(1500, 1.0e3)
    interactions = [("a", "b"), ("e", "c")][:pairs]
    model = SuperGLM(
        family="poisson",
        features={
            "a": Spline(n_knots=6),
            "b": Spline(n_knots=6),
            "e": Spline(n_knots=5),
            "c": Categorical(),
            "d": Numeric(),
        },
        interactions=interactions,
        weight_semantics="frequency",
        discrete=False,
    ).fit_reml(frame, y, sample_weight=weights, runtime_validation="skip")

    assert len(model._reml_penalties) >= 4
    assert counts["calls"] >= 2
    assert counts["grams"] == 0
    assert counts["passes"] == counts["calls"]
    assert model._reml_profile["reml_w_correction_mode"] == "leverage_hessian"


def test_a_fit_whose_grams_are_cheap_keeps_them():
    """Below the work bar the exact first-order Hessian and its path are kept."""
    frame, y, weights = _poisson_frame(1500, 1.0e3)
    model = SuperGLM(
        family="poisson",
        features={
            "a": Spline(n_knots=6),
            "b": Spline(n_knots=6),
            "e": Spline(n_knots=5),
            "c": Categorical(),
            "d": Numeric(),
        },
        interactions=[("a", "b")],
        weight_semantics="frequency",
        discrete=False,
    ).fit_reml(frame, y, sample_weight=weights, runtime_validation="skip")

    m, n, p = len(model._reml_penalties), model._dm.n, model._dm.p
    assert m >= 4 and m * n * p * p < direct._LEVERAGE_HESSIAN_MIN_GRAM_WORK
    assert model._reml_profile["reml_w_correction_mode"] == "derivative_grams"


def test_a_model_the_structured_solver_can_fit_keeps_the_derivative_grams(monkeypatch):
    """Both backends must build one outer Hessian, so the dense one keeps the Grams."""
    monkeypatch.setattr(direct, "_LEVERAGE_HESSIAN_MIN_GRAM_WORK", 0.0, raising=False)
    rng = np.random.default_rng(947)
    n = 400
    frame = pd.DataFrame(
        {
            "x": rng.uniform(-1.0, 1.0, n),
            "z": rng.normal(size=n),
            "segment": [f"s{code}" for code in rng.integers(0, 6, n)],
        }
    )
    eta = -0.2 + 0.3 * np.sin(2.0 * frame.x) + 0.1 * frame.z
    y = rng.poisson(np.exp(eta)).astype(np.float64)
    model = SuperGLM(
        family="poisson",
        features={"x": Spline(n_knots=5), "z": Numeric()},
        interactions=[FactorSmooth("x", group="segment", k=6)],
        selection_penalty=0.0,
        direct_solve="gram",
    ).fit_reml(frame, y, runtime_validation="skip")

    assert len(model._reml_penalties) >= 4
    assert model._reml_profile["reml_w_correction_mode"] == "derivative_grams"


_REML_TOL = 1e-9  # NEWTON_REML_TOL_DEFAULT, passed explicitly so the bound below is its own


def test_each_route_stops_where_the_other_route_is_stationary(monkeypatch):
    """Both routes stop at one stationary point, judged by the stopping rule itself.

    The exact REML engine has two certified exits, each a bar on the
    projected gradient with the directions it froze as flat zeroed:

    - ``score_objective_tolerance`` (``convergence.evaluate_reml_candidate``):
      below ``reml_tol * (1 + |V|)``, frozen directions masked by
      ``mask_frozen_stop_gradient``;
    - ``converged_at_precision`` (``convergence.classify_dead_feasible_exit``,
      a line search that rejected every trial): below ``max(1e-7, reml_tol) *
      (1 + |V|)``, or ``DECREMENT_GRADIENT_CEILING`` times that when the Newton
      decrement left is below the objective's resolution.

    Which exit a fit takes can turn on rounding in its last line search (it
    differs between OpenBLAS kernels), so the test reads the exit the fit took
    and applies that exit's bar.  The objective is no measure here: along a
    frozen direction it may still fall by up to the freeze bar, ``1e-7 * (1 +
    |V|)``, per unit of log-lambda.

    The leverage route leaves only a Hessian term out, so its gradient is the
    Gram route's.  Each fit's terminal candidate is therefore re-scored with
    the *other* route's weight correction, on the same PIRLS state, inverse
    and frozen set, and must pass the same bar.  Rounding between the routes
    is ``O(u)`` of the gradient terms (``test_leverage_terms_match_*``),
    orders below either bar.  A gradient that is wrong on either route moves
    that route's stationary point, and the other route's gradient there then
    misses the bar: halving the leverage route's gradient correction fails
    this test (the mutation run is recorded on PR #487).
    """
    from superglm.reml.convergence import (
        DECREMENT_GRADIENT_CEILING,
        FLAT_DIRECTION_FREEZE_FLOOR,
        _score_scale,
        mask_frozen_stop_gradient,
    )

    monkeypatch.setattr(direct, "_LEVERAGE_HESSIAN_MIN_GRAM_WORK", 0.0, raising=False)
    frame, y, weights = _poisson_frame(1500, 1.0e3)
    correction = direct.reml_w_correction
    partial = direct.reml_direct_gradient

    def fit(leverage: bool) -> tuple[SuperGLM, dict]:
        last: dict = {}

        def recorded_correction(*args, **kwargs):
            kwargs["leverage_hessian"] = leverage and kwargs.get("leverage_hessian", False)
            last["correction"] = (args, dict(kwargs), correction(*args, **kwargs))
            return last["correction"][2]

        def recorded_partial(*args, **kwargs):
            last["partial"] = partial(*args, **kwargs)
            return last["partial"]

        monkeypatch.setattr(direct, "reml_w_correction", recorded_correction)
        monkeypatch.setattr(direct, "reml_direct_gradient", recorded_partial)
        model = SuperGLM(
            family="poisson",
            features={
                "a": Spline(n_knots=6),
                "b": Spline(n_knots=6),
                "e": Spline(n_knots=5),
                "c": Categorical(),
                "d": Numeric(),
            },
            interactions=[("a", "b"), ("e", "c")],
            weight_semantics="frequency",
            discrete=False,
        ).fit_reml(frame, y, sample_weight=weights, reml_tol=_REML_TOL, runtime_validation="skip")
        return model, last

    for leverage in (True, False):
        model, last = fit(leverage)
        result = model._reml_result
        assert result.converged
        assert model._reml_profile["reml_w_correction_mode"] == (
            "leverage_hessian" if leverage else "derivative_grams"
        )
        args, kwargs, own = last["correction"]
        swapped: dict[str, Any] = dict(kwargs)
        swapped["leverage_hessian"] = not leverage
        other = correction(*args, **swapped)
        assert own is not None and other is not None
        objective = float(result.objective_history[-1])
        scale = _score_scale(objective)
        frozen = np.asarray(model._reml_profile["reml_freeze_decision"]["frozen"])
        # Interior smoothing parameters, clear of both bounds REML clips log
        # lambda to (1e-6 and 1e10, ``reml/direct.py``): there the bound
        # projection is the identity, so the projected score is the score.
        rho = np.log(np.array([model._reml_lambdas[pc.name] for pc in model._reml_penalties]))
        assert np.all((rho > np.log(1e-6) + 0.01) & (rho < np.log(1e10) - 0.01))

        def stop_norm(
            weight_gradient: np.ndarray, *, exit: str = result.termination_reason
        ) -> float:
            gradient = last["partial"] + weight_gradient
            if exit == "score_objective_tolerance":
                stop = mask_frozen_stop_gradient(
                    gradient, frozen, objective=objective, tolerance=_REML_TOL
                )
            else:
                stop = np.where(frozen, 0.0, gradient)
            return float(np.max(np.abs(stop)))

        if result.termination_reason == "score_objective_tolerance":
            bar = _REML_TOL * scale
        else:
            assert result.termination_reason == "converged_at_precision"
            bar = max(FLAT_DIRECTION_FREEZE_FLOOR, _REML_TOL) * scale
            if stop_norm(own[0]) >= bar:  # the Newton-decrement arm
                bar *= DECREMENT_GRADIENT_CEILING
        assert stop_norm(own[0]) < bar
        assert stop_norm(other[0]) < bar
