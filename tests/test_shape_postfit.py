import copy
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.interpolate import BSpline, PPoly

from superglm import (
    Constraint,
    FactorSmooth,
    LambdaPolicy,
    Numeric,
    PSpline,
    RandomEffect,
    SuperGLM,
)
from superglm._roundoff import gamma
from superglm.constraints import (
    _ROW_STEPS,
    _SLACK_STEPS,
    MonotoneRepairResult,
    shape_constraint_certificate,
    shape_constraint_is_roundoff_feasible,
    shape_derivative_matrix,
    shape_slack_tolerance,
)
from superglm.distributions import Gamma, Gaussian, Poisson
from superglm.links import IdentityLink, LogLink
from superglm.model import shape_ops
from superglm.model.fit_state import fitted_lambda2
from tests._shape_violation import curvature_violation, monotonicity_violation


def _minimum_signed_derivative(spec, beta, kind: str) -> float:
    """Independent span-wise certificate for a fitted B-spline term."""
    order = 1 if kind in {"increasing", "decreasing"} else 2
    sign = -1.0 if kind in {"decreasing", "concave"} else 1.0
    raw_beta = spec._R_inv @ beta if spec._R_inv is not None else beta
    polynomial = PPoly.from_spline(BSpline(spec._knots, raw_beta, spec.degree, extrapolate=False))
    signed_derivative = polynomial.derivative(order)
    stationary = signed_derivative.derivative().roots(
        discontinuity=False,
        extrapolate=False,
    )
    breakpoints = polynomial.x[(polynomial.x >= spec._lo) & (polynomial.x <= spec._hi)]
    stationary = stationary[
        np.isfinite(stationary) & (stationary >= spec._lo) & (stationary <= spec._hi)
    ]
    candidates = np.unique(np.concatenate(([spec._lo, spec._hi], breakpoints, stationary)))
    return float(np.min(sign * signed_derivative(candidates)))


def test_apply_shape_postfit_repairs_convex_term():
    rng = np.random.default_rng(0)
    x = np.linspace(0.0, 1.0, 200)
    y = -((x - 0.35) ** 2) + 0.05 * rng.normal(size=len(x))
    df = pd.DataFrame({"x": x})

    model = SuperGLM(
        family="gaussian",
        features={"x": PSpline(n_knots=10, constraint=Constraint.postfit.convex)},
    ).fit(df, y)

    model.apply_shape_postfit(df)
    repair = model._shape_repairs["x"]

    assert repair.kind == "convex"
    assert repair.max_violation_after <= 1e-8
    assert repair.max_violation_after <= repair.max_violation_before


def test_apply_shape_postfit_installs_one_synchronized_revision_without_row_copies():
    rng = np.random.default_rng(20260718)
    x = np.linspace(0.0, 1.0, 240)
    y = -((x - 0.4) ** 2) + 0.03 * rng.normal(size=x.size)
    X = pd.DataFrame({"x": x})
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"x": PSpline(n_knots=10, constraint=Constraint.postfit.convex)},
    ).fit(X, y)
    model.summary()
    old_revision = model._fit_revision
    old_beta = model.result.beta
    row_state = {
        name: getattr(model, name)
        for name in (
            "_dm",
            "_fit_X_ref",
            "_fit_y_ref",
            "_fit_sample_weight_ref",
            "_fit_offset_ref",
            "_fit_weights",
            "_fit_offset",
        )
    }

    returned = model.apply_shape_postfit(X)

    assert returned is model
    assert model._fit_revision == old_revision + 1
    assert model._fit_state.revision == model._fit_revision
    assert model._fit_state.repair_revision == 1
    for name, projected in model._fit_state.projections.items():
        assert getattr(model, name) is projected, name
    assert all(getattr(model, name) is value for name, value in row_state.items())
    assert not np.shares_memory(model.result.beta, old_beta)
    np.testing.assert_array_equal(model.result.beta, model._solver_result.beta)
    assert model._summary_cache is None
    assert model._fit_metrics_cache is None
    # The retained fitted-design refresh avoids an n-by-k public spline allocation;
    # solver/public canonical predictors may differ by final-operation roundoff only.
    np.testing.assert_allclose(model._fit_mu, model.predict(X), rtol=0.0, atol=5e-15)
    expected_deviance = float(
        np.sum(
            model._fit_weights * model._distribution.deviance_unit(model._fit_y_ref, model._fit_mu)
        )
    )
    assert model.result.deviance == pytest.approx(expected_deviance)
    assert model._solver_result.deviance == pytest.approx(expected_deviance)
    assert not model.result.beta.flags.writeable
    with pytest.raises(AttributeError, match="published"):
        model.result.deviance = 0.0


def test_a_repair_forms_its_frame_digest_certificates_and_grid_basis_once(monkeypatch):
    # The retained-data guard, the scoring guard and the guard the repair stores each read
    # the frame's digest; the early exit and the repairer certify the fitted coefficients,
    # and the repairer and the publication rule the repaired ones; the projection and the
    # publication rule evaluate the fitted basis on the grid. Each is formed once a call.
    import superglm.constraints as constraints

    rng = np.random.default_rng(20260718)
    x = np.linspace(0.0, 1.0, 240)
    y = -((x - 0.4) ** 2) + 0.03 * rng.normal(size=x.size)
    X = pd.DataFrame({"x": x})
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"x": PSpline(n_knots=10, constraint=Constraint.postfit.convex)},
    ).fit(X, y)
    spec, n_grid = model._specs["x"], 500
    counts = {"digests": 0, "certificates": 0, "grid bases": 0}
    real_hash, real_certificate = pd.util.hash_pandas_object, constraints._certificate_candidates
    real_transform = spec.transform

    def counted_hash(frame, *args, **kwargs):
        counts["digests"] += isinstance(frame, pd.DataFrame)
        return real_hash(frame, *args, **kwargs)

    def counted_certificate(*args, **kwargs):
        counts["certificates"] += 1
        return real_certificate(*args, **kwargs)

    def counted_transform(points):
        counts["grid bases"] += np.size(points) == n_grid
        return real_transform(points)

    monkeypatch.setattr(pd.util, "hash_pandas_object", counted_hash)
    monkeypatch.setattr(constraints, "_certificate_candidates", counted_certificate)
    monkeypatch.setattr(spec, "transform", counted_transform)
    revision = model._fit_revision

    model.apply_shape_postfit(X, n_grid=n_grid)

    assert model._fit_revision == revision + 1
    assert counts == {"digests": 1, "certificates": 2, "grid bases": 1}


def test_gaussian_shape_repair_refreshes_dispersion_after_final_edf() -> None:
    """Coefficient repair must not leave covariance and likelihood on the old scale."""
    rng = np.random.default_rng(20260718)
    x = np.linspace(0.0, 1.0, 240)
    X = pd.DataFrame({"x": x})
    y = -((x - 0.4) ** 2) + 0.03 * rng.normal(size=x.size)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"x": PSpline(n_knots=10, constraint=Constraint.postfit.convex)},
    ).fit(X, y)
    old_phi = model.result.phi

    model.apply_shape_postfit(X)

    variance = model._distribution.variance(model._fit_mu)
    pearson = float(np.sum(model._fit_weights * (y - model._fit_mu) ** 2 / variance))
    expected_phi = pearson / max(float(x.size) - model.result.effective_df, 1.0)
    assert model.result.phi == pytest.approx(expected_phi, rel=2e-13, abs=2e-13)
    assert model._solver_result.phi == pytest.approx(expected_phi, rel=2e-13, abs=2e-13)
    assert model.result.phi != pytest.approx(old_phi)
    assert model._fit_stats.log_likelihood == pytest.approx(
        model._distribution.log_likelihood(
            y,
            model._fit_mu,
            model._fit_weights,
            phi=expected_phi,
        )
    )


def test_shape_revision_clears_solver_identity_and_stale_reml_mode(tmp_path, monkeypatch) -> None:
    """A different coefficient vector cannot retain an accepted-mode identity/objective."""
    from superglm._debug import get_debug_level, set_debug_level

    previous = get_debug_level()
    monkeypatch.setenv("SUPERGLM_DEBUG_DIR", str(tmp_path))
    set_debug_level(2)
    try:
        rng = np.random.default_rng(20260718)
        x = np.linspace(0.0, 1.0, 240)
        X = pd.DataFrame({"x": x})
        y = -((x - 0.4) ** 2) + 0.03 * rng.normal(size=x.size)
        model = SuperGLM(
            family="gaussian",
            selection_penalty=0.0,
            features={"x": PSpline(n_knots=9, constraint=Constraint.postfit.convex)},
        ).fit_reml(X, y, max_reml_iter=3)
        assert model._solver_result.state_id is not None
        assert model._reml_result.objective is not None

        model.apply_shape_postfit(X)

        assert model._solver_result.state_id is None
        assert model._solver_result.evaluation_id is None
        assert model._solver_result.log_det_H is None
        assert model._solver_result.reml_hessian_rank is None
        assert not model._solver_result.converged
        assert model._solver_result.termination_reason == "coefficients_revised"
        assert model._reml_result.objective is None
        assert not model._reml_result.converged
        assert model._reml_result.termination_reason == "coefficients_revised"
    finally:
        set_debug_level(previous)


def test_failed_second_shape_repair_rolls_back_first_repair(monkeypatch):
    rng = np.random.default_rng(20260719)
    x = np.linspace(0.0, 1.0, 220)
    z = np.linspace(1.0, 0.0, 220)
    X = pd.DataFrame({"x": x, "z": z})
    y = -((x - 0.35) ** 2) - 0.6 * ((z - 0.7) ** 2) + 0.02 * rng.normal(size=x.size)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={
            "x": PSpline(n_knots=9, constraint=Constraint.postfit.convex),
            "z": PSpline(n_knots=9, constraint=Constraint.postfit.convex),
        },
    ).fit(X, y)
    old_dict = model.__dict__
    old_state = model._fit_state
    old_result = model.result
    old_solver_result = model._solver_result
    old_beta = model.result.beta.copy()
    old_predictions = model.predict(X)
    old_runtime_canonical_state = copy.deepcopy(model._runtime_canonical_state)
    real_repairer = shape_ops._repairer
    repair_calls = 0

    def failing_repairer(kind):
        delegate = real_repairer(kind)

        class FailOnSecondRepair:
            def repair(self, *args, **kwargs):
                nonlocal repair_calls
                repair_calls += 1
                if repair_calls == 2:
                    raise RuntimeError("injected second repair failure")
                return delegate.repair(*args, **kwargs)

        return FailOnSecondRepair()

    monkeypatch.setattr(shape_ops, "_repairer", failing_repairer)

    with pytest.raises(RuntimeError, match="injected second repair failure"):
        model.apply_shape_postfit(X)

    assert repair_calls == 2
    assert model.__dict__ is old_dict
    assert model._fit_state is old_state
    assert model.result is old_result
    assert model._solver_result is old_solver_result
    assert not hasattr(model, "_shape_repairs")
    np.testing.assert_array_equal(model.result.beta, old_beta)
    np.testing.assert_array_equal(model.predict(X), old_predictions)
    np.testing.assert_equal(model._runtime_canonical_state, old_runtime_canonical_state)


def test_compact_shape_repair_is_rejected_before_mutating_fitted_state():
    x = np.linspace(0.0, 1.0, 180)
    X = pd.DataFrame({"x": x})
    y = -((x - 0.4) ** 2) + 0.002 * np.sin(31.0 * x)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        retain_fit_state=False,
        features={"x": PSpline(n_knots=8, constraint=Constraint.postfit.convex)},
    ).fit(X, y)
    original_dict = model.__dict__
    original_result = model.result
    original_summary = model.summary()

    with pytest.raises(RuntimeError, match="retain_fit_state=True"):
        model.apply_shape_postfit(X)

    assert model.__dict__ is original_dict
    assert model.result is original_result
    assert model.summary() is original_summary


def test_non_gaussian_shape_repair_rebuilds_working_geometry():
    from superglm.inference.covariance import _active_penalty_matrix
    from superglm.model import state_ops
    from superglm.solvers.rank import decompose_gram

    rng = np.random.default_rng(20260720)
    x = np.linspace(0.0, 1.0, 400)
    X = pd.DataFrame({"x": x})
    y = rng.poisson(np.exp(1.2 - (x - 0.48) ** 2)).astype(np.float64)
    selection_lambda = 0.05
    model = SuperGLM(
        family="poisson",
        selection_penalty=selection_lambda,
        spline_penalty=0.2,
        features={"x": PSpline(n_knots=10, constraint=Constraint.postfit.convex)},
    ).fit(X, y)
    old_working_weights = state_ops._solver_space_working_weights(model).copy()

    model.apply_shape_postfit(X)

    assert model.result.rank_info is None
    assert model._solver_result.rank_info is None
    inference = model._fit_inference_info
    working_weights = inference["W"]
    assert not np.allclose(working_weights, old_working_weights, rtol=1e-8, atol=1e-10)
    # For Poisson/log, profiling the intercept enforces the canonical score
    # equation sum(w * mu) = sum(w * y), even though the rowwise curvature
    # changes and therefore still requires a full inference refresh.
    assert float(np.sum(working_weights)) == pytest.approx(
        float(np.sum(model._fit_weights * y)),
        rel=2e-11,
        abs=2e-11,
    )

    active_names = {group.name for group in inference["active_groups"]}
    active_design = np.hstack(
        [
            matrix.toarray()
            for matrix, group in zip(
                model._dm.group_matrices,
                model._groups,
                strict=True,
            )
            if group.name in active_names
        ]
    )
    mean_x = (active_design.T @ working_weights) / np.sum(working_weights)
    centered = active_design - mean_x
    penalty = _active_penalty_matrix(
        model._dm.group_matrices,
        model._groups,
        inference["active_groups"],
        fitted_lambda2(model),
        reml_penalties=getattr(model, "_reml_penalties", None),
    )
    original_by_name = {group.name: group for group in model._groups}
    for active_group in inference["active_groups"]:
        original = original_by_name[active_group.name]
        beta_group = np.asarray(model._solver_result.beta[original.sl])
        norm_group = float(np.linalg.norm(beta_group))
        unit = beta_group / norm_group
        penalty[active_group.sl, active_group.sl] += (
            selection_lambda * original.weight / norm_group
        ) * (np.eye(original.size) - np.outer(unit, unit))
    expected_hessian = centered.T @ (working_weights[:, None] * centered) + penalty
    expected_inverse = decompose_gram(expected_hessian).pseudo_inverse()

    np.testing.assert_allclose(
        inference["XtWX_inv_aug"][1:, 1:],
        expected_inverse,
        rtol=2e-11,
        atol=2e-11,
    )
    assert model.result.effective_df == pytest.approx(1.0 + np.sum(inference["edf"]))
    assert model._solver_result.effective_df == pytest.approx(model.result.effective_df)


@pytest.mark.parametrize("binds", [True, False], ids=["rows-bind", "no-row-binds"])
def test_a_repair_beside_a_constrained_term_keeps_the_face_its_fit_recorded(binds):
    """A ``Constraint.fit`` term beside a ``Constraint.postfit`` one, Poisson.

    Repairing ``z`` moves the working weights, which dropped the fit's rank
    record: ``a``, held on five of its seven directions, had its EDF and
    covariance rebuilt off the face (EDF 4.80 against 1.01; curve SEs off by
    up to 0.087 against a largest SE of 0.076), and a record of rows that do
    not bind read as no record at all. The repair leaves ``a`` where its QP
    put it, so its rows still bind: the record keeps them and is formed again
    at the new weights. ``a``'s EDF is checked against ``Z (Z'HZ)^-1 Z'`` for
    an orthonormal null-space basis ``Z`` of the recorded rows, ``H`` formed
    from the repaired predictions; its curve variances against ``H^-1``, the
    covariance with no row held, which is what a fit whose rows bind reports
    (Wood 2026, section 5.1); and the record's weight total and mean row
    against those predictions. Mutation
    check: dropping the record, keeping it at the fit's weights, forming it
    again off the face, or leaving its intercept part at the fit's weights
    each fail.
    """
    import scipy.linalg

    from superglm import Spline
    from superglm.inference.covariance import _active_penalty_matrix

    rng = np.random.default_rng(7)
    n = 600
    a, z = rng.uniform(0.0, 1.0, n), rng.uniform(0.0, 1.0, n)
    effect = 0.8 * a - 0.9 * (a > 0.5) if binds else 2.5 * a
    y = rng.poisson(np.exp(1.0 + effect + 0.8 * z + 0.4 * np.sin(9.0 * z)))
    X = pd.DataFrame({"a": a, "z": z})
    model = SuperGLM(
        family="poisson",
        features={
            "a": Spline(kind="bs", n_knots=4, constraint=Constraint.fit.increasing),
            "z": Spline(kind="bs", n_knots=8, constraint=Constraint.postfit.increasing),
        },
        spline_penalty=1.0,
        tol=1e-12,
    ).fit(X, y)
    group = next(group for group in model._groups if group.feature_name == "a")
    fitted = model._solver_result.rank_info
    theta = model.result.beta[group.sl].copy()
    rows = fitted.face_rows
    if binds:  # held on part of the term
        assert 0 < group.size - np.linalg.matrix_rank(fitted.face[group.sl]) < group.size
    else:
        assert rows.shape == (0, len(fitted.mean_x)) and fitted.face is None

    model.apply_shape_postfit(X)

    assert set(model._shape_repairs) == {"z"}
    assert np.array_equal(model.result.beta[group.sl], theta)
    mu = model.predict(X)  # Poisson/log: the working weight is the mean
    design = np.hstack([matrix.toarray() for matrix in model._dm.group_matrices])
    centred = design - (design.T @ mu) / np.sum(mu)
    data = centred.T @ (mu[:, None] * centred)
    penalty = _active_penalty_matrix(
        model._dm.group_matrices, model._groups, model._groups, fitted_lambda2(model)
    )
    face = scipy.linalg.null_space(rows) if binds else np.eye(design.shape[1])
    reduced = face.T @ (data + penalty) @ face
    covariance = face @ np.linalg.solve(reduced, face.T)
    unheld = np.linalg.inv(data + penalty)
    term = model.term_inference("a", with_se=True, n_points=50)
    basis = model._specs["a"].transform(term.x)
    variance = model.result.phi * np.einsum("ij,jk,ik->i", basis, unheld[group.sl, group.sl], basis)
    # The two inverses part by their formation and solve roundings, a few u
    # per entry amplified by the reduced Hessian's condition (Higham 2002,
    # sec. 7.1); the dropped record moved the EDF by 3.8 and the variances by
    # more than their size.
    bound = 64 * design.shape[1] * np.linalg.cond(reduced) * np.finfo(float).eps
    expected_edf = float(np.trace(covariance[group.sl] @ data[:, group.sl]))
    assert abs(term.edf - expected_edf) <= bound * max(1.0, expected_edf)
    unheld_bound = 64 * design.shape[1] * np.linalg.cond(data + penalty) * np.finfo(float).eps
    np.testing.assert_allclose(
        term.se_log_relativity**2, variance, rtol=0, atol=unheld_bound * variance.max()
    )
    kept = model._solver_result.rank_info
    assert kept is not None and kept is model.result.rank_info
    assert np.array_equal(kept.face_rows, rows) and (kept.face is None) == (fitted.face is None)
    # The intercept's part of the record, the repaired weights' total and mean
    # row, to the sums' rounding, n u of their absolute totals.
    rounding = n * np.finfo(float).eps
    assert abs(kept.sum_w - np.sum(mu)) <= rounding * np.sum(mu)
    np.testing.assert_allclose(
        kept.mean_x, (design.T @ mu) / np.sum(mu), rtol=0, atol=rounding * np.abs(design).max()
    )


def test_shape_repair_preserves_geometry_when_working_weights_are_coefficient_invariant():
    rng = np.random.default_rng(20260721)
    x = np.linspace(0.0, 1.0, 300)
    X = pd.DataFrame({"x": x})
    mu = np.exp(0.5 - (x - 0.5) ** 2)
    y = rng.gamma(shape=8.0, scale=mu / 8.0)
    model = SuperGLM(
        family="gamma",
        selection_penalty=0.0,
        spline_penalty=0.5,
        features={"x": PSpline(n_knots=9, constraint=Constraint.postfit.convex)},
    ).fit(X, y)
    old_beta = model.result.beta.copy()
    old_rank = model._solver_result.rank_info

    model.apply_shape_postfit(X)

    assert not np.allclose(model.result.beta, old_beta)
    assert model._solver_result.rank_info is not None
    assert model._solver_result.rank_info.sum_w == pytest.approx(old_rank.sum_w)
    assert model.result.effective_df == pytest.approx(old_rank.total_edf)


@pytest.mark.parametrize("spline_penalty", [0.0, 0.2])
@pytest.mark.parametrize("penalty_name", ["group_lasso", "ridge"])
def test_shape_repair_rebuilds_exact_selection_inference_curvature(
    spline_penalty,
    penalty_name,
):
    from superglm.inference.covariance import _active_penalty_matrix
    from superglm.model import state_ops
    from superglm.solvers.rank import decompose_gram

    x = np.linspace(0.0, 1.0, 500)
    X = pd.DataFrame({"x": x})
    y = x + 0.25 * np.sin(4.0 * np.pi * x)
    lambda1 = 0.05
    model = SuperGLM(
        family="gaussian",
        penalty=penalty_name,
        selection_penalty=lambda1,
        spline_penalty=spline_penalty,
        features={"x": PSpline(n_knots=12, constraint=Constraint.postfit.increasing)},
    ).fit(X, y)
    old_beta = model.result.beta.copy()

    model.apply_shape_postfit(X)

    assert np.linalg.norm(model.result.beta - old_beta) > 1e-3
    inference = model._fit_inference_info
    working_weights = inference["W"]
    active_names = {group.name for group in inference["active_groups"]}
    active_design = np.hstack(
        [
            matrix.toarray()
            for matrix, group in zip(model._dm.group_matrices, model._groups, strict=True)
            if group.name in active_names
        ]
    )
    mean_x = (active_design.T @ working_weights) / np.sum(working_weights)
    centered = active_design - mean_x
    data_gram = centered.T @ (working_weights[:, None] * centered)
    curvature = _active_penalty_matrix(
        model._dm.group_matrices,
        model._groups,
        inference["active_groups"],
        fitted_lambda2(model),
        reml_penalties=getattr(model, "_reml_penalties", None),
    )
    original_by_name = {group.name: group for group in model._groups}
    for active_group in inference["active_groups"]:
        original = original_by_name[active_group.name]
        beta_group = np.asarray(model._solver_result.beta[original.sl])
        if penalty_name == "ridge":
            curvature[active_group.sl, active_group.sl] += lambda1 * np.eye(original.size)
        else:
            norm_group = float(np.linalg.norm(beta_group))
            unit = beta_group / norm_group
            curvature[active_group.sl, active_group.sl] += (
                lambda1 * original.weight / norm_group
            ) * (np.eye(original.size) - np.outer(unit, unit))
    expected_inverse = decompose_gram(data_gram + curvature).pseudo_inverse()
    expected_edf = np.diag(expected_inverse @ data_gram)

    np.testing.assert_allclose(
        inference["XtWX_inv_aug"][1:, 1:],
        expected_inverse,
        rtol=3e-10,
        atol=3e-10,
    )
    np.testing.assert_allclose(inference["edf"], expected_edf, rtol=3e-10, atol=3e-10)
    covariance, _ = state_ops.coef_covariance(model)
    np.testing.assert_allclose(
        covariance / model.result.phi,
        expected_inverse,
        rtol=3e-10,
        atol=3e-10,
    )
    assert model.result.effective_df == pytest.approx(1.0 + float(np.sum(expected_edf)))


def test_shape_repair_can_deselect_the_last_group_and_rebuild_empty_geometry():
    rng = np.random.default_rng(20260723)
    x = np.linspace(0.0, 1.0, 300)
    X = pd.DataFrame({"x": x})
    mu = np.exp(1.0 - 0.8 * x)
    y = rng.gamma(shape=8.0, scale=mu / 8.0)
    model = SuperGLM(
        family="gamma",
        selection_penalty=0.001,
        spline_penalty=0.0,
        features={"x": PSpline(n_knots=9, constraint=Constraint.postfit.increasing)},
    ).fit(X, y)
    assert model._solver_result.rank_info.selected_group_names == ("x",)

    model.apply_shape_postfit(X)

    assert np.linalg.norm(model.result.beta) <= 1e-12
    assert model.result.rank_info is None
    assert model.result.effective_df == pytest.approx(1.0)
    assert model._fit_inference_info["edf"].size == 0
    assert model._fit_inference_info["XtWX_inv"].shape == (0, 0)


@pytest.mark.parametrize("mutated_input", ["features", "response"])
def test_shape_repair_rejects_mutated_retained_fit_data_atomically(mutated_input):
    x = np.linspace(0.0, 1.0, 220)
    X = pd.DataFrame({"x": x})
    y = -((x - 0.35) ** 2) + 0.01 * np.sin(19.0 * x)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"x": PSpline(n_knots=9, constraint=Constraint.postfit.convex)},
    ).fit(X, y)
    original_dict = model.__dict__
    original_result = model.result
    original_beta = model.result.beta.copy()
    if mutated_input == "features":
        X.loc[X.index[0], "x"] += 0.25
    else:
        y[0] += 5.0

    with pytest.raises(RuntimeError, match="retained fit data.*mutated"):
        model.apply_shape_postfit(X)

    assert model.__dict__ is original_dict
    assert model.result is original_result
    assert not hasattr(model, "_shape_repairs")
    np.testing.assert_array_equal(model.result.beta, original_beta)


@pytest.mark.parametrize("changed_input", ["sample_weight", "offset"])
def test_shape_repair_rejects_scoring_geometry_that_differs_from_fit(changed_input):
    rng = np.random.default_rng(20260720)
    x = np.linspace(0.0, 1.0, 220)
    X = pd.DataFrame({"x": x})
    y = -((x - 0.35) ** 2) + 0.01 * rng.normal(size=x.size)
    sample_weight = 1.0 + x
    offset = 0.05 * np.sin(2.0 * np.pi * x)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"x": PSpline(n_knots=9, constraint=Constraint.postfit.convex)},
    ).fit(X, y, sample_weight=sample_weight, offset=offset)
    original_dict = model.__dict__
    original_result = model.result
    original_beta = model.result.beta.copy()
    repair_weight = sample_weight.copy()
    repair_offset = offset.copy()
    if changed_input == "sample_weight":
        repair_weight[0] += 1.0
    else:
        repair_offset[0] += 1.0

    with pytest.raises(RuntimeError, match="sample_weight and offset must match the fitted data"):
        model.apply_shape_postfit(
            X,
            sample_weight=repair_weight,
            offset=repair_offset,
        )

    assert model.__dict__ is original_dict
    assert model.result is original_result
    assert not hasattr(model, "_shape_repairs")
    np.testing.assert_array_equal(model.result.beta, original_beta)


def test_repeated_shape_repair_skips_fitted_state_revision(monkeypatch):
    x = np.linspace(0.0, 1.0, 200)
    X = pd.DataFrame({"x": x})
    y = -((x - 0.4) ** 2) + 0.005 * np.sin(23.0 * x)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"x": PSpline(n_knots=9, constraint=Constraint.postfit.convex)},
    ).fit(X, y)
    model.apply_shape_postfit(X)
    installed_result = model.result
    installed_revision = model._fit_revision

    def fail_if_started(*args, **kwargs):
        raise AssertionError("no-op repair constructed a fitted-state revision")

    monkeypatch.setattr(shape_ops.FittedStateRevision, "start", fail_if_started)

    assert model.apply_shape_postfit(X) is model
    assert model.result is installed_result
    assert model._fit_revision == installed_revision


def test_all_exactly_feasible_repairs_preserve_the_accepted_reml_state(monkeypatch):
    """Analytically shape-feasible terms must be a true fitted-state no-op."""
    rng = np.random.default_rng(42)
    n = 180
    X = pd.DataFrame({"x": rng.uniform(size=n), "z": rng.uniform(size=n)})
    y = 0.5 + 1.3 * X["x"].to_numpy() + 0.7 * X["z"].to_numpy()
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        spline_penalty=1.0,
        features={
            "x": PSpline(n_knots=8, constraint=Constraint.postfit.increasing),
            "z": PSpline(n_knots=8, constraint=Constraint.postfit.increasing),
        },
    ).fit_reml(X, y, max_reml_iter=3)
    accepted = {
        "dict": model.__dict__,
        "fit_state": model._fit_state,
        "result": model.result,
        "solver_result": model._solver_result,
        "reml_result": model._reml_result,
        "revision": model._fit_revision,
        "objective": model._reml_result.objective,
        "converged": model.result.converged,
    }

    for name in ("x", "z"):
        groups = [group for group in model._groups if group.feature_name == name]
        beta = np.concatenate([model.result.beta[group.sl] for group in groups])
        assert _minimum_signed_derivative(model._specs[name], beta, "increasing") > 0.5

    def fail_if_started(*args, **kwargs):
        raise AssertionError("an exact no-op must not construct a fitted-state revision")

    monkeypatch.setattr(shape_ops.FittedStateRevision, "start", fail_if_started)

    assert model.apply_shape_postfit(X) is model
    assert model.__dict__ is accepted["dict"]
    assert model._fit_state is accepted["fit_state"]
    assert model.result is accepted["result"]
    assert model._solver_result is accepted["solver_result"]
    assert model._reml_result is accepted["reml_result"]
    assert model._fit_revision == accepted["revision"]
    assert model._reml_result.objective == accepted["objective"]
    assert model.result.converged is accepted["converged"]
    assert not hasattr(model, "_shape_repairs")


@pytest.mark.parametrize("seed", [3, 5, 7, 20260718])
def test_convex_postfit_projection_is_finite_feasible_and_better_than_zero_term(seed):
    """A failed curvature QP must never publish an arbitrarily large coefficient vector."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, 120)
    X = pd.DataFrame({"x": x})
    y = -((x - 0.4) ** 2) + 0.05 * rng.normal(size=x.size)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"x": PSpline(n_knots=10, constraint=Constraint.postfit.convex)},
    ).fit(X, y)
    original_intercept = model.result.intercept
    zero_term_deviance = float(np.sum((y - original_intercept) ** 2))

    model.apply_shape_postfit(X)

    group_beta = np.concatenate(
        [model.result.beta[group.sl] for group in model._groups if group.feature_name == "x"]
    )
    spec = model._specs["x"]
    dense_reconstruction = spec.reconstruct(group_beta, n_points=2001)
    repaired_curve = dense_reconstruction["log_relativity"]
    training_term = spec.transform(x) @ group_beta
    predictions = model.predict(X)

    assert np.all(np.isfinite(model.result.beta))
    assert np.all(np.isfinite(repaired_curve))
    assert np.all(np.isfinite(predictions))
    assert curvature_violation(repaired_curve, "convex") <= 1e-10
    assert model.result.deviance <= zero_term_deviance * (1.0 + 1e-10)
    assert model.result.intercept == pytest.approx(original_intercept, rel=0.0, abs=0.0)
    assert np.average(training_term, weights=model._fit_weights) == pytest.approx(0.0, abs=2e-12)
    np.testing.assert_allclose(
        predictions,
        model.result.intercept + training_term,
        rtol=2e-13,
        atol=2e-13,
    )
    repair = model._shape_repairs["x"]
    np.testing.assert_allclose(
        repair.repaired_log_effect,
        spec.transform(repair.grid) @ group_beta,
        rtol=2e-13,
        atol=2e-13,
    )


def test_monotone_repair_certifies_between_the_repair_grid_points():
    """A 500-point value grid can miss a sizeable derivative sign reversal."""
    rng = np.random.default_rng(13)
    x = np.linspace(0.0, 1.0, 80)
    X = pd.DataFrame({"x": x})
    y = x + 0.3 * rng.normal(size=x.size)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={
            "x": PSpline(
                n_knots=12,
                constraint=Constraint.postfit.increasing,
            )
        },
    ).fit(X, y)

    model.apply_shape_postfit(X, n_grid=500)

    groups = [group for group in model._groups if group.feature_name == "x"]
    beta = np.concatenate([model.result.beta[group.sl] for group in groups])
    spec = model._specs["x"]
    repair_grid_curve = spec.reconstruct(beta, n_points=500)["log_relativity"]
    assert monotonicity_violation(repair_grid_curve, "increasing") <= 5e-15
    assert _minimum_signed_derivative(spec, beta, "increasing") >= -2e-9
    assert (
        shape_constraint_certificate(
            spec,
            beta,
            "increasing",
        ).minimum_scaled_slack
        >= -1e-11
    )


def test_monotone_projection_takes_few_qp_solves(monkeypatch):
    import superglm.solvers.constrained_qp as constrained_qp

    rng = np.random.default_rng(13)
    x = np.linspace(0.0, 1.0, 80)
    X = pd.DataFrame({"x": x})
    y = x + 0.3 * rng.normal(size=x.size)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"x": PSpline(n_knots=12, constraint=Constraint.postfit.increasing)},
    ).fit(X, y)
    real_solve = constrained_qp.solve_constrained_qp
    solve_calls = 0

    def counted_solve(*args, **kwargs):
        nonlocal solve_calls
        solve_calls += 1
        return real_solve(*args, **kwargs)

    monkeypatch.setattr(constrained_qp, "solve_constrained_qp", counted_solve)

    model.apply_shape_postfit(X, n_grid=500)

    assert 1 <= solve_calls <= QP_SOLVE_BUDGET


#: QP solves one repair may take: its exchange converges quadratically once the
#: binding spans' minima are found, so a handful of steps suffice.
QP_SOLVE_BUDGET = 12
#: Relative KKT residual of the closest monotone curve, as the oracle reads it at the
#: exact points where the derivative touches zero. Inside a span such a point is a
#: double root, and a minimum feasible to rounding locates it only to about sqrt(u)
#: of the span: the exchange may hold it with two cuts that far apart. So the oracle
#: reads up to about sqrt(u) there, and about 1e-16 where the repair touches at knots.
#: The same fixtures' repairs in the grid metric read 1e-3 to 1e-2 here.
CLOSEST_REPAIR_KKT = float(np.sqrt(np.finfo(np.float64).eps / 2.0))

_SWEEP_SHAPES = {
    "wiggle": (lambda x: 0.9 * x + 0.25 * np.sin(12 * x), "uniform"),
    "hook": (
        lambda x: 0.8 * np.log1p(6 * x) - 1.2 * np.maximum(x - 0.55, 0.0) ** 1.5,
        "beta",
    ),
}


def _x_groups(model):
    return [group for group in model._groups if group.feature_name == "x"]


def _sweep_fit(shape, direction, family, kind, size):
    """One fixture of the post-fit monotone sweep: a REML fit whose repair binds."""
    import zlib

    from superglm import Spline

    curve, design = _SWEEP_SHAPES[shape]
    rng = np.random.default_rng(zlib.crc32(repr((shape, direction, family, kind, size)).encode()))
    n = 4000
    x = rng.uniform(0.0, 1.0, n) if design == "uniform" else rng.beta(1.0, 3.0, n)
    eta = curve(x) if direction == "increasing" else curve(1.0 - x)
    if family == "poisson":
        y = rng.poisson(np.exp(0.5 + eta)).astype(np.float64)
    else:
        y = eta + 0.35 * rng.normal(size=n)
    constraint = getattr(Constraint.postfit, direction)
    spline = (
        Spline(kind="cr", k=size, constraint=constraint)
        if kind == "cr"
        else Spline(kind="bs", n_knots=size, constraint=constraint)
    )
    X = pd.DataFrame({"x": x})
    model = SuperGLM(family=family, selection_penalty=0.0, features={"x": spline}).fit_reml(X, y)
    return model, X


def _wiggly_monotone_fit(kind, size):
    return _sweep_fit("wiggle", "increasing", "gaussian", kind, size)


@pytest.mark.parametrize(
    ("shape", "direction", "family", "kind", "size"),
    [
        # The exchange used to stop after 14 span refinements, still infeasible.
        ("wiggle", "increasing", "gaussian", "cr", 8),
        # SLSQP used to stop 1.7e-6 of the zero-term loss short of the closest curve on
        # the grid.
        ("hook", "decreasing", "poisson", "bs", 8),
    ],
)
def test_monotone_repair_is_the_closest_monotone_curve(shape, direction, family, kind, size):
    from superglm.constraints import MonotoneRepairer
    from tests._shape_projection_oracle import grid_metric, projection_certificate

    model, X = _sweep_fit(shape, direction, family, kind, size)
    spec, groups = model._specs["x"], _x_groups(model)
    weights = shape_ops._grid_weights(
        spec, X["x"].to_numpy(), np.asarray(model._fit_weights, dtype=np.float64), 500
    )
    grid_repair = MonotoneRepairer(direction).repair(
        spec, model.result.beta, groups, weights=weights, n_grid=500
    )
    fitted = _term_beta(model.result.beta, groups)

    certificate = _certified_repair(model, X, direction)

    assert certificate.minimum_derivative >= -1e-10
    assert certificate.kkt_residual <= CLOSEST_REPAIR_KKT
    # Called without a metric, the public repairer projects the curve on the grid.
    H, g = grid_metric(spec, grid_repair.grid, grid_repair.original_log_effect, weights)
    grid_beta = _term_beta(grid_repair.repaired_beta_reparam, groups)
    grid_certificate = projection_certificate(spec, direction, grid_beta, H, g, fitted)
    assert grid_certificate.minimum_derivative >= -1e-10
    assert grid_certificate.kkt_residual <= CLOSEST_REPAIR_KKT


def _quartic_noise_fit():
    """A quartic P-spline on 200 Gaussian rows of noise, whose bracket's inner solves stall.

    Its first Bernstein inner solve exhausts every start on every OpenBLAS kernel;
    whether a later one certifies, or the outer relaxation's does, turns on the
    kernel's rounding.
    """
    rng = np.random.default_rng(4)
    x = np.sort(rng.uniform(size=200))
    y = rng.normal(size=200)
    X = pd.DataFrame({"x": x})
    feature = PSpline(n_knots=10, degree=4, m=2)
    model = SuperGLM(
        family="gaussian", selection_penalty=0.0, spline_penalty=0.2, features={"x": feature}
    ).fit(X, y)
    return model, X


def test_grid_repair_whose_bracket_inner_solve_never_certifies_is_published(monkeypatch):
    """A bracket whose inner solve stops off the inner cone publishes, with its gap disclosed.

    Forced: the exchange is skipped, the bracket takes one step, and every start of
    its inner solve stops uncertified at the unconstrained minimum, which breaks the
    inner rows. The bracket then held no feasible point and the repair raised. Its
    incumbent is feasible without a solve (the apex, or the best multiple of the
    solve's start), so the repair is increasing everywhere and says it is not the
    closest such curve.
    """
    import superglm.solvers.constrained_qp as constrained_qp
    from superglm import constraints
    from tests._shape_projection_oracle import grid_metric, projection_certificate

    model, _ = _quartic_noise_fit()
    spec, groups = model._specs["x"], _x_groups(model)
    monkeypatch.setattr(constraints, "_PROJECTION_STEPS", 0)
    monkeypatch.setattr(constraints, "_BRACKET_STEPS", 1)
    real_solve_cone_qp = constraints._solve_cone_qp
    inner_solves = []

    def off_the_rows(H, g, A, b, **kwargs):
        beta = np.linalg.lstsq(H, g, rcond=None)[0]
        assert np.min(A @ beta - b) < 0.0
        return constrained_qp.QPResult(beta=beta, n_iter=kwargs["max_iter"], converged=False)

    def inner_never_certifies(*args, feasible=False, **kwargs):
        if not feasible:
            return real_solve_cone_qp(*args, **kwargs)
        inner_solves.append(args[2].shape)
        with monkeypatch.context() as patch:
            patch.setattr(constrained_qp, "solve_constrained_qp", off_the_rows)
            return real_solve_cone_qp(*args, feasible=True, **kwargs)

    monkeypatch.setattr(constraints, "_solve_cone_qp", inner_never_certifies)

    with pytest.warns(RuntimeWarning, match="stopped before it certified the closest increasing"):
        repair = constraints.MonotoneRepairer().repair(spec, model.result.beta, groups)

    assert len(inner_solves) == 1
    H, g = grid_metric(spec, repair.grid, repair.original_log_effect, np.ones(repair.grid.size))
    repaired = _term_beta(repair.repaired_beta_reparam, groups)
    fitted = _term_beta(model.result.beta, groups)
    certificate = projection_certificate(spec, "increasing", repaired, H, g, fitted)
    assert certificate.minimum_derivative >= -1e-10


def _rise_fit(t, x, **spline):
    """A Gaussian fit of a wiggly rise on predictor values ``x``, sampled at positions ``t``."""
    X = pd.DataFrame({"x": x})
    y = 0.9 * t + 0.25 * np.sin(12 * t)
    feature = PSpline(constraint=Constraint.postfit.increasing, **spline)
    model = SuperGLM(family="gaussian", selection_penalty=0.0, features={"x": feature})
    return model.fit(X, y), X


def _term_beta(beta, groups):
    return np.concatenate([np.asarray(beta, dtype=np.float64)[group.sl] for group in groups])


def _posterior_metric(model):
    """The one-term fit's penalized Hessian, read off its own posterior covariance.

    The Bayesian covariance of the term's coefficients is ``phi (X'WX + S)^-1``
    with the intercept profiled (Wood, *Shape constrained additive smooth models*,
    Statistical Modelling 26(4), 2026, section 2), so ``phi`` times its inverse is
    the metric the repair projects in, formed by the inference code instead of the
    repair's. Returns it and the fitted coefficients it projects.
    """
    groups = _x_groups(model)
    covariance, active = model._coef_covariance
    assert [group.name for group in active] == [group.name for group in groups]
    H = model._solver_result.phi * np.linalg.inv(covariance)
    return 0.5 * (H + H.T), _term_beta(model.result.beta, groups)


def _certified_repair(model, X, kind, **repair):
    """Repair the fit, and certify the result as the projection in its posterior metric."""
    from tests._shape_projection_oracle import projection_certificate

    H, fitted = _posterior_metric(model)
    model.apply_shape_postfit(X, **repair)
    beta = _term_beta(model._shape_repairs["x"].repaired_beta_reparam, _x_groups(model))
    return projection_certificate(model._specs["x"], kind, beta, H, H @ fitted, fitted)


@pytest.mark.parametrize("offset", [1e8, 1e12])
def test_monotone_repair_far_from_the_origin_is_the_closest_monotone_curve(offset):
    """Span polynomials come from representable points, not rounded collocation nodes."""
    t = np.linspace(0.0, 1.0, 200)
    model, X = _rise_fit(t, offset + t, n_knots=9)

    certificate = _certified_repair(model, X, "increasing")

    assert certificate.minimum_derivative >= -1e-10
    assert certificate.kkt_residual <= CLOSEST_REPAIR_KKT


def test_monotone_repair_across_a_one_ulp_span_is_the_closest_monotone_curve():
    """A span one ULP wide keeps its own polynomial piece in the projection's rows."""
    rng = np.random.default_rng(3)
    x = np.sort(rng.uniform(0.0, 1.0, 600))
    X = pd.DataFrame({"x": x})
    knots = [0.4, float(np.nextafter(0.4, np.inf)), 0.8]
    feature = PSpline(knots=knots, degree=2, constraint=Constraint.postfit.increasing)
    y = 0.9 * x + 0.25 * np.sin(12 * x) + 0.1 * rng.normal(size=x.size)
    model = SuperGLM(family="gaussian", selection_penalty=0.0, features={"x": feature}).fit(X, y)

    certificate = _certified_repair(model, X, "increasing")

    assert certificate.minimum_derivative >= -1e-10
    assert certificate.kkt_residual <= CLOSEST_REPAIR_KKT


_REVIEW_SHAPES = {
    "dip": lambda x: x - 0.5 * np.exp(-(((x - 0.5) / 0.07) ** 2)),
    "rise": lambda x: np.sqrt(x) - 0.45 * x**3,
    "hook": lambda x: 0.8 * np.log1p(6 * x) - 1.2 * np.maximum(x - 0.55, 0.0) ** 1.5,
    "flat": lambda x: 0.0 * x,
    "bowl": lambda x: 1.5 * (x - 0.4) ** 2 + 0.08 * np.sin(14 * x),
}
_REVIEW_BASES = {
    "bs3": {"kind": "bs", "n_knots": 8},
    "bs4": {"kind": "bs", "n_knots": 8, "degree": 4},
    "bs5": {"kind": "bs", "n_knots": 6, "degree": 5},
    "ps3": {"kind": "ps", "n_knots": 10},
}


def _review_fit(shape, direction, family, tag):
    """One fixture of the review sweep: a REML fit, n = 4000, whose ``direction`` rule binds."""
    import zlib

    from superglm import Spline

    basis = _REVIEW_BASES[tag]
    key = repr((shape, direction, family, tag, sorted(basis.items())))
    rng = np.random.default_rng(zlib.crc32(key.encode()) + 7)
    n = 4000
    x = rng.uniform(0, 1, n) if zlib.crc32(shape.encode()) % 2 else rng.beta(1.2, 2.5, n)
    curve = _REVIEW_SHAPES[shape]
    eta = {"decreasing": curve(1.0 - x), "concave": -curve(x)}.get(direction, curve(x))
    if family == "poisson":
        y = rng.poisson(np.exp(0.3 + eta)).astype(np.float64)
    else:
        y = eta + 0.3 * rng.normal(size=n)
    X = pd.DataFrame({"x": x})
    spline = Spline(constraint=getattr(Constraint.postfit, direction), **basis)
    model = SuperGLM(family=family, selection_penalty=0.0, features={"x": spline})
    return model.fit_reml(X, y), X


@pytest.mark.parametrize(
    ("shape", "direction", "family", "tag"),
    [
        ("dip", "increasing", "poisson", "bs4"),
        ("rise", "decreasing", "poisson", "bs3"),
        ("rise", "decreasing", "poisson", "ps3"),
        ("rise", "increasing", "gaussian", "bs5"),
        ("hook", "increasing", "poisson", "bs5"),
        ("hook", "decreasing", "gaussian", "bs5"),
        ("flat", "convex", "gaussian", "bs4"),
        ("flat", "convex", "gaussian", "bs5"),
        ("flat", "convex", "poisson", "bs5"),
        ("bowl", "concave", "poisson", "bs4"),
    ],
)
def test_repair_along_directions_the_penalty_prices_is_published(shape, direction, family, tag):
    """Repairs master publishes, which the grid metric's exact projection refused.

    Each fitted basis has wiggly directions that a sparse region of the axis barely
    sees. The grid metric omits the smoothing penalty, so its projection travelled
    10 to 1,000 times the fit's size along them, and the publication rule refused
    it as worse than dropping the term. The penalized Hessian prices them as the
    rule does: the repair publishes, and it is the projection in that metric.
    """
    model, X = _review_fit(shape, direction, family, tag)

    certificate = _certified_repair(model, X, direction)

    assert certificate.minimum_derivative >= -1e-10
    assert certificate.kkt_residual <= CLOSEST_REPAIR_KKT


def _beta_rise(family, n=3000):
    """A wiggly rise on a skewed axis, n rows: an increasing rule binds where it wiggles."""
    rng = np.random.default_rng(11)
    x = rng.beta(1.2, 2.5, n)
    eta = 0.9 * x + 0.25 * np.sin(12 * x)
    if family == "poisson":
        return pd.DataFrame({"x": x}), rng.poisson(np.exp(0.3 + eta)).astype(np.float64)
    return pd.DataFrame({"x": x}), eta + 0.3 * rng.normal(size=n)


@pytest.mark.parametrize("family", ["gaussian", "poisson"])
def test_select_term_is_projected_in_both_of_its_penalties(family):
    """``select=True`` adds a null-space penalty with its own smoothing parameter.

    The posterior's precision carries both, so a metric missing either one fails
    the oracle's optimality check.
    """
    from superglm import Spline

    X, y = _beta_rise(family)
    feature = Spline(kind="bs", n_knots=8, select=True, constraint=Constraint.postfit.increasing)
    model = SuperGLM(family=family, selection_penalty=0.0, features={"x": feature}).fit_reml(X, y)
    assert len([penalty for penalty in model._reml_penalties if penalty.group_name == "x"]) == 2

    certificate = _certified_repair(model, X, "increasing")

    assert certificate.minimum_derivative >= -1e-10
    assert certificate.kkt_residual <= CLOSEST_REPAIR_KKT


@pytest.mark.parametrize("selection_penalty", [0.1, 10.0])
@pytest.mark.parametrize("family", ["gaussian", "poisson"])
def test_repair_under_a_selection_penalty_is_the_projection_in_the_smooth_metric(
    family, selection_penalty
):
    """The rule's merit carries the group lasso; the metric is the smooth's ``X'WX + S``.

    The posterior adds the lasso's curvature, ``lambda w / ||beta|| (I - u u')``
    (``pirls._add_selection_local_curvature``), so the reference takes it off again.
    The term stays selected, and its repair publishes.
    """
    from tests._shape_projection_oracle import projection_certificate

    X, y = _beta_rise(family)
    feature = PSpline(n_knots=10, constraint=Constraint.postfit.increasing)
    model = SuperGLM(family=family, selection_penalty=selection_penalty, features={"x": feature})
    model.fit(X, y)
    (group,) = _x_groups(model)
    posterior, fitted = _posterior_metric(model)
    norm = float(np.linalg.norm(fitted))
    unit = fitted / norm
    curvature = (
        selection_penalty * group.weight / norm * (np.eye(fitted.size) - np.outer(unit, unit))
    )
    H = posterior - curvature

    model.apply_shape_postfit(X)

    beta = _term_beta(model._shape_repairs["x"].repaired_beta_reparam, [group])
    certificate = projection_certificate(
        model._specs["x"], "increasing", beta, H, H @ fitted, fitted
    )
    assert np.any(beta != 0.0)
    assert certificate.minimum_derivative >= -1e-10
    assert certificate.kkt_residual <= CLOSEST_REPAIR_KKT


def test_repair_metric_reads_the_fits_kept_gram_as_it_would_form_it():
    """The fit keeps its centred data Gram at its weights; the metric reads it in place.

    Read, it equals the Gram formed over the rows to rounding, and it is still there
    for the fit's first inference, which takes it.
    """
    from superglm.model import state_ops

    model, _ = _sweep_fit("hook", "decreasing", "poisson", "bs", 8)
    groups = _x_groups(model)
    W = state_ops._solver_space_working_weights(model)
    record = model._solver_result.rank_info
    assert record.data_gram is not None

    kept = shape_ops._term_data_hessian(model, groups, W)

    assert record.data_gram is not None
    model._fit_inference_info  # noqa: B018 - the first inference takes the Gram
    assert record.data_gram is None
    formed = shape_ops._term_data_hessian(model, groups, W)
    (matrix,) = [
        gm for gm, g in zip(model._dm.group_matrices, model._groups, strict=True) if g in groups
    ]
    columns = np.column_stack([matrix.matvec(unit) for unit in np.eye(formed.shape[0])])
    magnitude = np.abs(columns).T @ (W[:, None] * np.abs(columns))
    np.testing.assert_array_less(
        np.abs(kept - formed), gamma(W.size + 2) * 4.0 * magnitude + 1e-300
    )


def test_two_group_term_metric_holds_its_cross_blocks():
    """A term on two column groups gets one ``X'WX``, cross blocks included, intercept profiled.

    The profiled Gram is the Gram of the columns centred at their weighted means.
    """
    from superglm import Spline

    rng = np.random.default_rng(5)
    n = 400
    X = pd.DataFrame({"a": rng.uniform(0.0, 1.0, n), "b": rng.uniform(0.0, 1.0, n)})
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"a": Spline(kind="cr", k=6), "b": Spline(kind="cr", k=6)},
    )
    model._add_interaction("a", "b", decompose=True)
    model._build_design_matrix(X, rng.normal(size=n), np.ones(n), None)
    pairs = [
        (matrix, group)
        for matrix, group in zip(model._dm.group_matrices, model._groups, strict=True)
        if group.subgroup_type in {"bilinear", "wiggly"}
    ]
    assert len(pairs) == 2
    W = rng.uniform(0.5, 2.0, n)
    columns = np.column_stack(
        [matrix.matvec(unit) for matrix, group in pairs for unit in np.eye(group.size)]
    )
    centred = columns - (W @ columns) / W.sum()
    expected = centred.T @ (W[:, None] * centred)

    metric = shape_ops._term_data_hessian(model, [group for _, group in pairs], W)

    magnitude = np.abs(columns).T @ (W[:, None] * np.abs(columns))
    np.testing.assert_array_less(np.abs(metric - expected), gamma(n + 2) * 4.0 * magnitude + 1e-300)


@pytest.mark.parametrize(("n_knots", "degree"), [(0, 1), (6, 3)])
def test_repair_after_summary_publishes_what_it_publishes_before(n_knots, degree):
    """``summary()`` takes the fit's kept Gram, so the repair forms its metric again.

    One row carries unit weight and the rest 1e-20, so the term's curvature, about 7e-19,
    is far below the rounding of its raw moments. Profiled as ``X'WX - (X'W1)(1'WX) / 1'W``
    it came out -6e-17, and the repair refused the metric as not positive definite. Formed
    from centred columns, as the fit forms it, the repair publishes the same curve.
    """
    x = np.linspace(0.0, 1.0, 100)
    X = pd.DataFrame({"x": x})
    weights = np.full(x.size, 1e-20)
    weights[-1] = 1.0
    feature = PSpline(
        n_knots=n_knots,
        degree=degree,
        m=1,
        penalty="none",
        constraint=Constraint.postfit.increasing,
    )
    model = SuperGLM(
        family="gaussian", selection_penalty=0.0, spline_penalty=0.0, features={"x": feature}
    ).fit(X, 1.0 - x, sample_weight=weights)
    summarised = copy.deepcopy(model)
    summarised.summary()
    assert summarised._solver_result.rank_info.data_gram is None

    model.apply_shape_postfit(X)
    summarised.apply_shape_postfit(X)

    assert "x" in model._shape_repairs and "x" in summarised._shape_repairs
    np.testing.assert_allclose(summarised.predict(X), model.predict(X), rtol=0.0, atol=1e-12)


def test_repair_under_a_large_smoothing_parameter_is_the_projection_in_the_fits_hessian():
    """The metric's penalty block is formed as the fit forms it, from a transported root.

    At lambda 1e14 the congruence ``R' Omega R`` left rounding on the penalty's null space
    that lambda turned into up to 2% of the data's curvature there, and the repair was the
    projection in another metric: a KKT residual of 7e-5 in the posterior's.
    """
    x = np.linspace(0.0, 1.0, 200)
    X = pd.DataFrame({"x": x})
    feature = PSpline(n_knots=9, m=3, constraint=Constraint.postfit.increasing)
    model = SuperGLM(
        family="gaussian", selection_penalty=0.0, spline_penalty=1e14, features={"x": feature}
    ).fit(X, (x - 0.4) ** 2)

    certificate = _certified_repair(model, X, "increasing")

    assert certificate.minimum_derivative >= -1e-10
    assert certificate.kkt_residual <= CLOSEST_REPAIR_KKT


@pytest.mark.parametrize("scale", [1e165, 1e-150])
def test_monotone_repair_in_extreme_predictor_units_matches_unit_units(scale):
    """Derivative rows that would under- or overflow are normalized range-safely."""
    t = np.linspace(0.0, 1.0, 200)
    spline = {"n_knots": 9, "penalty": "none", "m": 1}
    unit, X_unit = _rise_fit(t, t, **spline)
    scaled, X_scaled = _rise_fit(t, scale * t, **spline)

    unit.apply_shape_postfit(X_unit)
    scaled.apply_shape_postfit(X_scaled)

    expected = unit._shape_repairs["x"].repaired_log_effect
    repaired = scaled._shape_repairs["x"].repaired_log_effect
    np.testing.assert_allclose(repaired, expected, rtol=0.0, atol=1e-9 * np.max(np.abs(expected)))


def test_monotone_repair_on_a_grid_coarser_than_the_basis_is_published():
    """Five grid points cannot pin twelve coefficients: the grid metric is semidefinite.

    The repairer called without a metric projects on that grid; the full call
    projects in the penalized Hessian over the rows, which is definite.
    """
    from superglm.constraints import MonotoneRepairer

    t = np.linspace(0.0, 1.0, 200)
    model, X = _rise_fit(t, t, n_knots=7, degree=5, m=1, penalty="none")
    spec, groups = model._specs["x"], _x_groups(model)
    fit_weights = np.asarray(model._fit_weights, dtype=np.float64)
    weights = shape_ops._grid_weights(spec, X["x"].to_numpy(), fit_weights, 5)
    grid_repair = MonotoneRepairer("increasing").repair(
        spec, model.result.beta, groups, weights=weights, n_grid=5
    )

    model.apply_shape_postfit(X, n_grid=5)

    for repair in (grid_repair, model._shape_repairs["x"]):
        beta = _term_beta(repair.repaired_beta_reparam, groups)
        assert shape_constraint_is_roundoff_feasible(spec, beta, "increasing")


def test_monotone_repair_on_a_wide_piecewise_linear_basis_is_published():
    """A 440-row projection gets a budget that grows with its rows."""
    x = np.linspace(0.0, 1.0, 1000)
    X = pd.DataFrame({"x": x})
    feature = PSpline(
        n_knots=220, degree=1, m=1, penalty="none", constraint=Constraint.postfit.increasing
    )
    model = SuperGLM(family="gaussian", selection_penalty=0.0, features={"x": feature})
    model.fit(X, 1.0 - x)

    model.apply_shape_postfit(X)

    repair = model._shape_repairs["x"]
    assert repair.max_violation_after == 0.0


def test_distinct_unit_rows_memory_is_linear_in_the_rows():
    """Repeated rows are found without an all-pairs comparison."""
    import tracemalloc

    from superglm.constraints import _distinct_unit_rows

    rows = np.random.default_rng(0).normal(size=(400, 200))
    repeated = np.vstack([rows, 3.0 * rows])

    tracemalloc.start()
    try:
        unit, merged = _distinct_unit_rows(repeated)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    assert unit.shape == rows.shape
    assert merged == [[index, index + 400] for index in range(400)]
    assert peak <= 16 * repeated.nbytes


def test_distinct_unit_rows_memory_is_linear_when_the_rows_share_a_sum():
    """An evenly spaced degree-0 basis's jump rows share their entry sums.

    Every pair then falls in one sorted window; comparing them all at once held
    about 400 times the rows.
    """
    import tracemalloc

    from superglm.constraints import (
        _distinct_unit_rows,
        _shape_breakpoints,
        _shape_constraint_rows,
    )

    x = np.repeat(np.linspace(0.0, 1.0, 401), 3)
    feature = PSpline(
        n_knots=399, degree=0, m=1, penalty="none", constraint=Constraint.postfit.increasing
    )
    model = SuperGLM(family="gaussian", selection_penalty=0.0, features={"x": feature})
    model.fit(pd.DataFrame({"x": x}), np.sin(6.0 * x))
    spec = model._specs["x"]
    rows = _shape_constraint_rows(spec, _shape_breakpoints(spec), "increasing")

    tracemalloc.start()
    try:
        unit, merged = _distinct_unit_rows(rows)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    assert unit.shape == (399, rows.shape[1])
    assert merged == [[index] for index in range(399)]
    assert peak <= 16 * rows.nbytes


def test_curvature_repair_of_arbitrary_coefficients_is_published():
    """A solve whose own start stays outside the cone starts from a point inside it."""
    from superglm import Spline
    from superglm.constraints import CurvatureRepairer

    x = np.linspace(0.0, 1.0, 500)
    X = pd.DataFrame({"x": x})
    feature = Spline(kind="cr", k=20, constraint=Constraint.postfit.convex)
    model = SuperGLM(family="gaussian", selection_penalty=0.0, features={"x": feature})
    model.fit(X, np.sin(6.0 * x))
    spec, groups = model._specs["x"], _x_groups(model)
    beta = model.result.beta.copy()
    coefficients = np.random.default_rng(0).normal(size=sum(group.size for group in groups))
    offset = 0
    for group in groups:
        beta[group.sl] = coefficients[offset : offset + group.size]
        offset += group.size

    repair = CurvatureRepairer("convex").repair(spec, beta, groups, n_grid=500)

    repaired = np.concatenate([repair.repaired_beta_reparam[group.sl] for group in groups])
    assert shape_constraint_is_roundoff_feasible(spec, repaired, "convex")


@pytest.mark.slow
def test_monotone_repair_on_a_hundred_knot_basis_is_published():
    """Most of a large basis's rows bind; every solve still certifies."""
    from superglm import Spline

    rng = np.random.default_rng(0)
    x = rng.uniform(0.0, 1.0, 4000)
    X = pd.DataFrame({"x": x})
    y = 0.9 * x + 0.25 * np.sin(12 * x) + 0.3 * rng.normal(size=x.size)
    feature = Spline(kind="ps", n_knots=100, constraint=Constraint.postfit.increasing)
    model = SuperGLM(family="gaussian", selection_penalty=0.0, features={"x": feature})
    model.fit(X, y)

    certificate = _certified_repair(model, X, "increasing")

    assert certificate.minimum_derivative >= -1e-10


def _qp_iterations(monkeypatch):
    import superglm.solvers.constrained_qp as constrained_qp

    real_solve = constrained_qp.solve_constrained_qp
    count = {"solves": 0, "iterations": 0}

    def counted_solve(*args, **kwargs):
        result = real_solve(*args, **kwargs)
        count["solves"] += 1
        count["iterations"] += result.n_iter
        return result

    monkeypatch.setattr(constrained_qp, "solve_constrained_qp", counted_solve)
    return count


#: Active-set iterations of a curvature repair on a degree-4 basis, whose curvature
#: is quadratic on each span, so the exchange runs. Warm-started solves in the
#: penalized Hessian take it to about 300; from the QP's own start it took 2,409.
CURVATURE_EXCHANGE_ITERATIONS = 900


def test_curvature_exchange_on_a_degree_four_basis_takes_few_iterations(monkeypatch):
    """The review's slowest curvature repair: a flat concave fit on a quartic basis."""
    import zlib

    from superglm import Spline

    key = (
        "flat",
        "concave",
        "gaussian",
        "bs4",
        sorted(dict(kind="bs", n_knots=8, degree=4).items()),
    )
    rng = np.random.default_rng(zlib.crc32(repr(key).encode()) + 7)
    x = rng.uniform(0, 1, 4000) if zlib.crc32(b"flat") % 2 else rng.beta(1.2, 2.5, 4000)
    y = 0.3 * rng.normal(size=x.size)
    X = pd.DataFrame({"x": x})
    feature = Spline(kind="bs", n_knots=8, degree=4, constraint=Constraint.postfit.concave)
    model = SuperGLM(family="gaussian", selection_penalty=0.0, features={"x": feature})
    model.fit_reml(X, y)
    count = _qp_iterations(monkeypatch)

    model.apply_shape_postfit(X)

    assert "x" in model._shape_repairs
    assert count["iterations"] <= CURVATURE_EXCHANGE_ITERATIONS, count


#: Active-set iterations of a one-QP curvature repair: started inside the cone at
#: the rows the fitted curve breaks, the solve certifies in 7; from its own start
#: it took 16.
ONE_QP_ITERATIONS = 12


def test_one_qp_curvature_repair_takes_few_iterations(monkeypatch):
    """A cubic P-spline's curvature is linear on each span, so one warm-started QP is exact."""
    rng = np.random.default_rng(20261111)
    x = np.sort(rng.uniform(0.0, 1.0, 600))
    X = pd.DataFrame({"x": x})
    y = 1.5 * (x - 0.4) ** 2 + 0.08 * np.sin(14 * x) + 0.05 * rng.normal(size=x.size)
    feature = PSpline(n_knots=12, constraint=Constraint.postfit.convex)
    model = SuperGLM(family="gaussian", selection_penalty=0.0, features={"x": feature}).fit(X, y)
    count = _qp_iterations(monkeypatch)

    model.apply_shape_postfit(X)

    assert "x" in model._shape_repairs
    assert count["solves"] == 1
    assert count["iterations"] <= ONE_QP_ITERATIONS, count


# A grid repair at near-zero weights, below. Most spans of these projections come out
# flat, and a flat span's lowest point was its rounding: cuts there made the QPs
# degenerate, so starts ran their whole budgets, and its curvature took each exchange
# step past the rank cut, where the QP refused it. Iteration totals on this
# near-singular metric turn on the BLAS kernel's rounding (the 200-knot repair takes
# 279 to 1,134 single-threaded across this host's OpenBLAS kernels, and 1,585 on Linux
# ARM64), so the tests assert what the fix makes structural: the flat-span rule
# fires, every cone QP certifies, and at 200 knots the QP is given the floored,
# full-rank metric. The counts are a backstop well above every kernel's and below the
# unfixed code's: at 721150d40 the four 120-knot repairs took 249, 14,977, 2,974 and
# 468 iterations, and the 200-knot one 8,697.
NEAR_ZERO_GRID_ITERATIONS = 1_000
FLOORED_GRID_ITERATIONS = 4_000


def _repair_work(monkeypatch):
    """Spies on a repair's work: QP iterations and ranks, cone solves, flat spans skipped."""
    import superglm.solvers.constrained_qp as constrained_qp
    from superglm import constraints

    work = {"iterations": 0, "truncated": 0, "uncertified": 0, "flat": 0}
    real_qp = constrained_qp.solve_constrained_qp
    real_cone = constraints._solve_cone_qp
    real_minimum = constraints._piece_minimum

    def qp(*args, **kwargs):
        result = real_qp(*args, **kwargs)
        work["iterations"] += result.n_iter
        work["truncated"] += result.rank < result.width
        return result

    def cone(*args, **kwargs):
        result = real_cone(*args, **kwargs)
        work["uncertified"] += result is None
        return result

    def minimum(piece, p, beta, skip_flat=False):
        if not skip_flat:
            return real_minimum(piece, p, beta)
        found = real_minimum(piece, p, beta, skip_flat)
        work["flat"] += found is None and real_minimum(piece, p, beta) is not None
        return found

    monkeypatch.setattr(constrained_qp, "solve_constrained_qp", qp)
    monkeypatch.setattr(constraints, "_solve_cone_qp", cone)
    monkeypatch.setattr(constraints, "_piece_minimum", minimum)
    return work


def _near_zero_weight_repair(data, direction, knots, monkeypatch):
    """A ``ps`` repair on the grid at raw frequencies plus 1e-12: the grid metric and its work."""
    import warnings

    from superglm import Spline
    from superglm.constraints import MonotoneRepairer
    from tests._shape_projection_oracle import grid_metric, projection_certificate

    rng = np.random.default_rng(0)
    if data == "gapped":
        u = rng.uniform(0.0, 0.6, 3000)
        x = np.where(u < 0.3, u, u + 0.4)
    else:
        x = rng.beta(0.4, 3.0, 3000)
    t = x if direction == "increasing" else 1.0 - x
    y = 0.9 * t + 0.25 * np.sin(12 * t) + 0.3 * rng.normal(size=x.size)
    feature = Spline(kind="ps", n_knots=knots, constraint=getattr(Constraint.postfit, direction))
    model = SuperGLM(family="gaussian", selection_penalty=0.0, features={"x": feature})
    model.fit(pd.DataFrame({"x": x}), y)
    spec, groups = model._specs["x"], _x_groups(model)
    counts = np.histogram(x, bins=500)[0]
    weights = counts / counts.sum() + 1e-12
    work = _repair_work(monkeypatch)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        repair = MonotoneRepairer(direction).repair(
            spec, model.result.beta, groups, weights=weights
        )

    H, g = grid_metric(spec, repair.grid, repair.original_log_effect, weights)
    repaired = _term_beta(repair.repaired_beta_reparam, groups)
    fitted = _term_beta(model.result.beta, groups)
    return H, projection_certificate(spec, direction, repaired, H, g, fitted), work


@pytest.mark.slow
@pytest.mark.parametrize("direction", ["increasing", "decreasing"])
@pytest.mark.parametrize("data", ["gapped", "skewed"])
def test_grid_repair_with_near_zero_weights_on_a_large_basis_is_the_closest_curve(
    data, direction, monkeypatch
):
    """120-knot P-splines on data with a gap or a long tail, at the grid's raw frequencies."""
    _, certificate, work = _near_zero_weight_repair(data, direction, 120, monkeypatch)
    assert certificate.minimum_derivative >= -1e-10
    assert certificate.kkt_residual <= CLOSEST_REPAIR_KKT
    assert work["flat"] > 0, work
    # No exchange step's QP is left uncertified, nor a curvature step refused at the rank cut.
    assert work["uncertified"] == 0, work
    assert work["iterations"] <= NEAR_ZERO_GRID_ITERATIONS, work


@pytest.mark.slow
def test_grid_repair_on_a_metric_the_qp_would_truncate_is_the_closest_curve(monkeypatch):
    """A 200-knot P-spline whose grid metric loses a direction to the rank cut, floored."""
    from superglm.constraints import _jacobi_scale
    from superglm.solvers.rank import _symmetric_part, decompose_gram

    H, certificate, work = _near_zero_weight_repair("skewed", "increasing", 200, monkeypatch)
    scale = _jacobi_scale(H)
    decomposition = decompose_gram(_symmetric_part(H * scale[:, None] * scale[None, :]))
    assert decomposition.rank < decomposition.width
    # Measured in the caller's unfloored metric.
    assert certificate.minimum_derivative >= -1e-10
    assert certificate.kkt_residual <= CLOSEST_REPAIR_KKT
    assert work["flat"] > 0, work
    # The QP is given the floored metric, which it does not truncate.
    assert work["truncated"] == 0, work
    assert work["iterations"] <= FLOORED_GRID_ITERATIONS, work


# The flat Poisson fixture of the review sweep. Its grid weights fall to 1e-5 of
# the rest in the sparse tail, so the grid metric's cond(H) passes 1 / u, and an
# unscaled solve certified on some OpenBLAS kernels and cycled on others. The
# public repairer still projects in that metric when called without one; the full
# call projects in the penalized Hessian, whose Gram goes through BLAS as well.
_KERNEL_FIXTURE = """
import sys, warnings, zlib
import numpy as np, pandas as pd
from superglm import Constraint, Spline, SuperGLM
warnings.simplefilter("ignore")
key = ("flat", "decreasing", "poisson", "bs5", sorted(dict(kind="bs", n_knots=6, degree=5).items()))
rng = np.random.default_rng(zlib.crc32(repr(key).encode()) + 7)
x = rng.uniform(0, 1, 4000) if zlib.crc32(b"flat") % 2 else rng.beta(1.2, 2.5, 4000)
y = rng.poisson(np.exp(0.3), 4000).astype(float)
X = pd.DataFrame({"x": x})
spline = Spline(kind="bs", n_knots=6, degree=5, constraint=Constraint.postfit.decreasing)
model = SuperGLM(family="poisson", selection_penalty=0.0, features={"x": spline}).fit_reml(X, y)
from superglm.constraints import MonotoneRepairer
from superglm.model import shape_ops
spec, groups = model._specs["x"], [g for g in model._groups if g.feature_name == "x"]
weights = shape_ops._grid_weights(spec, x, np.asarray(model._fit_weights, dtype=float), 500)
try:
    grid = MonotoneRepairer("decreasing").repair(spec, model.result.beta, groups, weights=weights)
except RuntimeError as exc:
    print("refused", exc)
else:
    print("published", " ".join(repr(float(v)) for v in grid.repaired_log_effect))
try:
    model.apply_shape_postfit(X)
except RuntimeError as exc:
    print("refused", exc)
else:
    print("published", " ".join(repr(float(v)) for v in model._shape_repairs["x"].repaired_log_effect))
"""

#: How a process ends when it runs an instruction its CPU lacks: killed by SIGILL on
#: POSIX, ``STATUS_ILLEGAL_INSTRUCTION`` on Windows.
_ILLEGAL_INSTRUCTION_EXITS = {-4, 0xC000001D}


def _kernel_outcomes(script, kernels):
    """Each OpenBLAS kernel's output lines of ``script``, split into words.

    A kernel forced past the host CPU's instruction set (SkylakeX's AVX-512 on an AVX2
    runner) dies of an illegal instruction: it cannot run here, so it is reported
    and left out. Any other failure fails the test.
    """
    import os
    import subprocess
    import warnings

    outcomes, not_run = {}, {}
    for kernel in kernels:
        environment = {**os.environ, "OPENBLAS_CORETYPE": kernel, "OPENBLAS_NUM_THREADS": "1"}
        run = subprocess.run(
            [sys.executable, "-c", script],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=300,
        )
        if run.returncode in _ILLEGAL_INSTRUCTION_EXITS:
            not_run[kernel] = f"illegal instruction (exit {run.returncode}): the CPU lacks it"
            continue
        assert run.returncode == 0, (kernel, run.returncode, run.stderr[-4000:])
        outcomes[kernel] = [line.split() for line in run.stdout.splitlines()]

    if not_run:
        warnings.warn(f"OpenBLAS kernels this CPU cannot run, left out: {not_run}", stacklevel=2)
    if len(outcomes) < 2:
        pytest.skip(f"fewer than two OpenBLAS kernels run on this CPU; left out {not_run}")
    return outcomes


@pytest.mark.slow
def test_repair_decision_is_the_same_on_every_blas_kernel():
    """Publish or refuse, and the repaired curve, agree across OpenBLAS microkernels.

    Each kernel runs the grid-metric repairer and the full call, in that order.
    """
    outcomes = _kernel_outcomes(_KERNEL_FIXTURE, ("Haswell", "Zen", "SkylakeX", "Sandybridge"))
    for metric in (0, 1):
        decisions = {kernel: lines[metric][0] for kernel, lines in outcomes.items()}
        assert set(decisions.values()) == {"published"}, (metric, decisions)
        curves = [np.asarray([float(v) for v in lines[metric][1:]]) for lines in outcomes.values()]
        for curve in curves[1:]:
            np.testing.assert_allclose(curve, curves[0], rtol=0.0, atol=1e-9)


# The quartic P-spline of _quartic_noise_fit, through the public grid repairer. On an
# AVX-512 host's Prescott and Nehalem kernels its bracket's outer relaxation does not
# certify either, so the bracket stops at its incumbent and discloses the gap; on
# SkylakeX, Haswell and Sandybridge it closes. On macOS ARM64 the bracket used to end
# with no feasible point, and the repair raised.
_BRACKET_KERNEL_FIXTURE = """
import warnings
import numpy as np, pandas as pd
from superglm import PSpline, SuperGLM
from superglm.constraints import MonotoneRepairer
rng = np.random.default_rng(4)
x = np.sort(rng.uniform(size=200))
y = rng.normal(size=200)
feature = PSpline(n_knots=10, degree=4, m=2)
model = SuperGLM(
    family="gaussian", selection_penalty=0.0, spline_penalty=0.2, features={"x": feature}
).fit(pd.DataFrame({"x": x}), y)
spec, groups = model._specs["x"], [g for g in model._groups if g.feature_name == "x"]
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    try:
        repair = MonotoneRepairer().repair(spec, model.result.beta, groups)
    except RuntimeError as exc:
        print("refused", exc)
        raise SystemExit
gap = any("stopped before it certified" in str(w.message) for w in caught)
print("published", "disclosed" if gap else "certified")
print(" ".join(repr(float(v)) for v in repair.repaired_log_effect))
print(" ".join(repr(float(v)) for g in groups for v in repair.repaired_beta_reparam[g.sl]))
"""


@pytest.mark.slow
def test_bracket_repair_publishes_on_every_blas_kernel():
    """Every OpenBLAS kernel publishes the bracketed repair, increasing everywhere.

    A kernel whose bracket closes publishes a curve within the bracket's objective gap
    of the closest one, so those curves agree in the caller's metric: the distance from
    the fitted curve, in units of the zero term's. Their values can differ by more, along
    directions the near-singular grid metric barely weighs.
    One whose bracket does not close discloses that, and publishes its best feasible point.
    """
    import warnings

    from superglm.constraints import _OBJECTIVE_GAP, MonotoneRepairer
    from tests._shape_projection_oracle import grid_metric, projection_certificate

    model, _ = _quartic_noise_fit()
    spec, groups = model._specs["x"], _x_groups(model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        local = MonotoneRepairer().repair(spec, model.result.beta, groups)
    H, g = grid_metric(spec, local.grid, local.original_log_effect, np.ones(local.grid.size))
    fitted = _term_beta(model.result.beta, groups)
    outcomes = _kernel_outcomes(
        _BRACKET_KERNEL_FIXTURE, ("Prescott", "Nehalem", "Sandybridge", "Haswell", "SkylakeX")
    )
    decisions = {kernel: lines[0] for kernel, lines in outcomes.items()}
    assert all(decision[0] == "published" for decision in decisions.values()), decisions
    certified = []
    for kernel, ((_, status), curve, beta) in outcomes.items():
        coefficients = np.asarray([float(v) for v in beta])
        certificate = projection_certificate(spec, "increasing", coefficients, H, g, fitted)
        assert certificate.minimum_derivative >= -1e-10, (kernel, certificate)
        if status == "certified":
            certified.append(coefficients)
    # Two points each within _OBJECTIVE_GAP of the optimum in the scaled objective, where
    # the zero term's distance is one, lie within twice that of each other; the rest is
    # the rounding of forming the distances here.
    zero = 0.5 * float(fitted @ H @ fitted)
    gaps = [0.5 * float((b - fitted) @ H @ (b - fitted)) / zero for b in certified]
    assert len(gaps) >= 2, decisions
    assert max(gaps) - min(gaps) <= 2.0 * _OBJECTIVE_GAP + 2.0**-40, gaps


def test_select_convex_repair_certifies_each_polynomial_span():
    """Selection reparameterization must not hide between-grid negative curvature."""
    rng = np.random.default_rng(0)
    x = np.linspace(0.0, 1.0, 80)
    X = pd.DataFrame({"x": x})
    y = rng.normal(size=x.size)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        spline_penalty=0.2,
        features={
            "x": PSpline(
                n_knots=12,
                select=True,
                constraint=Constraint.postfit.convex,
            )
        },
    ).fit(X, y)

    model.apply_shape_postfit(X, n_grid=500)

    groups = [group for group in model._groups if group.feature_name == "x"]
    beta = np.concatenate([model.result.beta[group.sl] for group in groups])
    spec = model._specs["x"]
    repair_grid_curve = spec.reconstruct(beta, n_points=500)["log_relativity"]
    assert curvature_violation(repair_grid_curve, "convex") <= 5e-15
    assert _minimum_signed_derivative(spec, beta, "convex") >= -2e-8


def test_piecewise_linear_convex_repair_certifies_slope_jumps():
    x = np.linspace(0.0, 1.0, 100)
    X = pd.DataFrame({"x": x})
    y = -((x - 0.5) ** 2)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={
            "x": PSpline(
                n_knots=10,
                degree=1,
                m=1,
                constraint=Constraint.postfit.convex,
            )
        },
    ).fit(X, y)

    model.apply_shape_postfit(X)

    groups = [group for group in model._groups if group.feature_name == "x"]
    beta = np.concatenate([model.result.beta[group.sl] for group in groups])
    dense_curve = model._specs["x"].reconstruct(beta, n_points=4001)["log_relativity"]
    assert curvature_violation(dense_curve, "convex") <= 2e-12


def test_invalid_curvature_repair_is_rejected_before_publication(monkeypatch):
    """The fitted-state transaction must roll back a finite but catastrophic repair."""
    rng = np.random.default_rng(20260718)
    x = np.linspace(0.0, 1.0, 160)
    X = pd.DataFrame({"x": x})
    y = -((x - 0.4) ** 2) + 0.03 * rng.normal(size=x.size)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"x": PSpline(n_knots=9, constraint=Constraint.postfit.convex)},
    ).fit(X, y)
    original_dict = model.__dict__
    original_result = model.result
    original_beta = model.result.beta.copy()
    original_predictions = model.predict(X)

    class InvalidRepairer:
        def repair(self, spec, beta, groups, *, weights, n_grid, metric):
            del groups, weights, metric
            grid = np.linspace(spec._lo, spec._hi, n_grid)
            invalid_beta = np.full_like(beta, 1e100)
            invalid_curve = np.full(n_grid, 1e100)
            return MonotoneRepairResult(
                feature_name="",
                direction="convex",
                grid=grid,
                original_log_effect=spec.reconstruct(beta, n_points=n_grid)["log_relativity"],
                repaired_log_effect=invalid_curve,
                repaired_beta_reparam=invalid_beta,
                max_violation_before=1.0,
                max_violation_after=0.0,
                projection_residual=1e100,
            )

    monkeypatch.setattr(shape_ops, "_repairer", lambda _kind: InvalidRepairer())

    with pytest.raises(RuntimeError, match="shape repair.*publication"):
        model.apply_shape_postfit(X)

    assert model.__dict__ is original_dict
    assert model.result is original_result
    assert not hasattr(model, "_shape_repairs")
    np.testing.assert_array_equal(model.result.beta, original_beta)
    np.testing.assert_array_equal(model.predict(X), original_predictions)


def test_qp_that_never_certifies_a_curvature_projection_is_disclosed(monkeypatch):
    """No QP solve certifies a convex repair: published convex everywhere, with a warning."""
    rng = np.random.default_rng(3)
    x = np.linspace(0.0, 1.0, 120)
    X = pd.DataFrame({"x": x})
    y = -((x - 0.4) ** 2) + 0.05 * rng.normal(size=x.size)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"x": PSpline(n_knots=10, constraint=Constraint.postfit.convex)},
    ).fit(X, y)
    _never_certifying_qp(monkeypatch, nnls_stops=False)

    with pytest.warns(RuntimeWarning, match="stopped before it certified the closest convex"):
        model.apply_shape_postfit(X)

    beta = _term_beta(model._shape_repairs["x"].repaired_beta_reparam, _x_groups(model))
    assert _minimum_signed_derivative(model._specs["x"], beta, "convex") >= -1e-12 * float(
        np.linalg.norm(beta)
    )


def test_infeasible_monotone_repair_is_rejected_atomically(monkeypatch):
    x = np.linspace(0.0, 1.0, 180)
    X = pd.DataFrame({"x": x})
    y = 1.0 - 1.5 * x + 0.01 * np.sin(17.0 * x)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"x": PSpline(n_knots=9, constraint=Constraint.postfit.increasing)},
    ).fit(X, y)
    original_dict = model.__dict__
    original_result = model.result
    original_beta = model.result.beta.copy()
    original_predictions = model.predict(X)

    class FalselyCertifiedRepairer:
        def repair(self, spec, beta, groups, *, weights, n_grid, metric):
            del groups, weights, metric
            reconstruction = spec.reconstruct(beta, n_points=n_grid)
            return MonotoneRepairResult(
                feature_name="",
                direction="increasing",
                grid=reconstruction["x"],
                original_log_effect=reconstruction["log_relativity"].copy(),
                repaired_log_effect=reconstruction["log_relativity"].copy(),
                repaired_beta_reparam=beta.copy(),
                max_violation_before=1.0,
                max_violation_after=0.0,
                projection_residual=0.0,
            )

    monkeypatch.setattr(shape_ops, "_repairer", lambda _kind: FalselyCertifiedRepairer())

    with pytest.raises(RuntimeError, match="infeasible monotonicity"):
        model.apply_shape_postfit(X)

    assert model.__dict__ is original_dict
    assert model.result is original_result
    assert not hasattr(model, "_shape_repairs")
    np.testing.assert_array_equal(model.result.beta, original_beta)
    np.testing.assert_array_equal(model.predict(X), original_predictions)


def test_publication_rejects_sub_grid_scaled_derivative_violation(monkeypatch):
    x = np.linspace(0.0, 1.0, 180)
    X = pd.DataFrame({"x": x})
    y = 1.0 - 1.5 * x
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"x": PSpline(n_knots=9, constraint=Constraint.postfit.increasing)},
    ).fit(X, y)
    spec = model._specs["x"]
    group = next(group for group in model._groups if group.feature_name == "x")
    derivative_row = shape_derivative_matrix(spec, np.asarray([0.37]), 1)[0]
    term_beta = -5e-10 * derivative_row / np.linalg.norm(derivative_row)
    candidate_beta = np.zeros_like(model.result.beta)
    candidate_beta[group.sl] = term_beta
    certificate = shape_constraint_certificate(spec, term_beta, "increasing")
    assert -2e-9 < certificate.minimum_scaled_slack < -1e-10

    class SubGridInfeasibleRepairer:
        def repair(self, _spec, beta, groups, *, weights, n_grid, metric):
            del groups, weights, metric
            reconstruction = _spec.reconstruct(beta, n_points=n_grid)
            grid = reconstruction["x"]
            repaired = _spec.transform(grid) @ term_beta
            return MonotoneRepairResult(
                feature_name="",
                direction="increasing",
                grid=grid,
                original_log_effect=reconstruction["log_relativity"].copy(),
                repaired_log_effect=repaired,
                repaired_beta_reparam=candidate_beta.copy(),
                max_violation_before=1.0,
                max_violation_after=0.0,
                projection_residual=0.0,
            )

    monkeypatch.setattr(
        shape_ops,
        "_repairer",
        lambda _kind: SubGridInfeasibleRepairer(),
    )

    with pytest.raises(RuntimeError, match="infeasible monotonicity"):
        model.apply_shape_postfit(X)


def _off_rule_by(monkeypatch, model, allowance):
    """Repair ``model``, its projection moved off the rule by ``allowance(certified, held)``.

    The stub runs the real repairer and pushes its projection outward along the unit row
    where it touches zero, so the least slack becomes ``-allowance`` to rounding. A
    U-shaped curve on an exponential axis is about three times as large in the repairer's
    scale, the curve's weighted RMS, as in ``held``, the larger of the fitted and repaired
    coefficients' norms, at which publication judged the rule before.
    """
    real = shape_ops._repairer("increasing")
    spec = model._specs["x"]
    (group,) = [group for group in model._groups if group.feature_name == "x"]
    fitted_norm = float(np.linalg.norm(model.result.beta[group.sl]))
    stub = {}

    class OffRuleRepairer:
        def repair(self, spec_, beta, groups, *, weights, **options):
            result = real.repair(spec_, beta, groups, weights=weights, **options)
            term = result.repaired_beta_reparam[group.sl]
            target = result.original_log_effect
            certified = float(np.sqrt(weights @ target**2 / np.sum(weights)))
            held = max(fitted_norm, float(np.linalg.norm(term)))
            assert certified > 2.0 * held
            touch = shape_constraint_certificate(spec_, term, "increasing").worst_x
            row = shape_derivative_matrix(spec_, np.asarray([touch]), 1)[0]
            off = term - allowance(certified, held) * row / np.linalg.norm(row)
            result.repaired_beta_reparam = result.repaired_beta_reparam.copy()
            result.repaired_beta_reparam[group.sl] = off
            stub.update(term=off, certified=certified, held=held)
            return result

    monkeypatch.setattr(shape_ops, "_repairer", lambda _kind: OffRuleRepairer())
    return spec, group, stub


def _u_shape_on_a_skewed_axis():
    x = np.random.default_rng(1).exponential(1.0, 3000)
    X = pd.DataFrame({"x": x})
    y = (x - 1.0) ** 2 + 0.01 * np.random.default_rng(2).normal(size=x.size)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"x": PSpline(n_knots=12, constraint=Constraint.postfit.increasing)},
    ).fit(X, y)
    return model, X


def test_projection_off_its_rule_by_its_certified_rounding_is_published(monkeypatch):
    model, X = _u_shape_on_a_skewed_axis()
    unit = gamma(_SLACK_STEPS * model._specs["x"].transform(np.zeros(1)).shape[1])
    # Between the rounding at the fitted scale and the rounding the repairer certified.
    spec, group, stub = _off_rule_by(
        monkeypatch, model, lambda certified, held: unit * float(np.sqrt(certified * held))
    )

    model.apply_shape_postfit(X)

    slack = shape_constraint_certificate(spec, stub["term"], "increasing").minimum_scaled_slack
    # The repairer's own acceptance rule held, and the old reading at ``held`` would not.
    assert -shape_slack_tolerance(stub["term"], stub["certified"]) <= slack
    assert slack < -shape_slack_tolerance(stub["term"], stub["held"])
    published = model.result.beta[group.sl]
    assert shape_constraint_is_roundoff_feasible(spec, published, "increasing")
    grid = np.linspace(spec._lo, spec._hi, 400)
    moved = spec.transform(grid) @ (published - stub["term"])
    # The curve moves by a rounding over the inside ray's slack: below the fit's own ``tol``.
    assert np.max(np.abs(moved)) <= 1e-6 * stub["certified"]
    # Obeying its rule at its own scale, the published term needs no second repair.
    revised = model.result.beta.copy()
    model.apply_shape_postfit(X)
    np.testing.assert_array_equal(model.result.beta, revised)


def test_projection_off_its_rule_beyond_its_certified_rounding_is_refused(monkeypatch):
    model, X = _u_shape_on_a_skewed_axis()
    unit = gamma(_SLACK_STEPS * model._specs["x"].transform(np.zeros(1)).shape[1])
    _off_rule_by(monkeypatch, model, lambda certified, held: 4.0 * unit * certified)
    original_beta = model.result.beta.copy()

    with pytest.raises(RuntimeError, match="infeasible monotonicity"):
        model.apply_shape_postfit(X)

    np.testing.assert_array_equal(model.result.beta, original_beta)


def test_select_spline_curvature_repair_preserves_public_solver_intercept_relation():
    rng = np.random.default_rng(3)
    x = np.linspace(0.0, 1.0, 180)
    X = pd.DataFrame({"x": x})
    y = -((x - 0.4) ** 2) + 0.04 * rng.normal(size=x.size)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        spline_penalty=0.2,
        features={
            "x": PSpline(
                n_knots=10,
                select=True,
                constraint=Constraint.postfit.convex,
            )
        },
    ).fit(X, y)
    original_public_intercept = model.result.intercept

    model.apply_shape_postfit(X)

    groups = [group for group in model._groups if group.feature_name == "x"]
    group_beta = np.concatenate([model.result.beta[group.sl] for group in groups])
    training_term = model._specs["x"].transform(x) @ group_beta
    dense_curve = model._specs["x"].reconstruct(group_beta, n_points=2001)["log_relativity"]
    assert curvature_violation(dense_curve, "convex") <= 1e-10
    assert np.mean(training_term) == pytest.approx(0.0, abs=2e-12)
    assert model.result.intercept == pytest.approx(original_public_intercept, abs=0.0)
    assert model.result.intercept == pytest.approx(
        model._solver_result.intercept + model._runtime_canonical_state["intercept_shift"],
        rel=2e-13,
        abs=2e-13,
    )
    np.testing.assert_allclose(
        model.predict(X),
        model.result.intercept + training_term,
        rtol=2e-13,
        atol=2e-13,
    )


def test_compact_shape_penalty_quadratic_matches_dense_component_algebra():
    from superglm.reml.penalty_algebra import build_penalty_matrix

    rng = np.random.default_rng(20260724)
    x = np.linspace(0.0, 1.0, 140)
    X = pd.DataFrame({"x": x})
    y = np.sin(5.0 * x) + 0.03 * rng.normal(size=x.size)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        spline_penalty=0.7,
        features={"x": PSpline(n_knots=10, select=True, m=(1, 2))},
    ).fit_reml(X, y, max_reml_iter=3)
    beta = rng.normal(size=model._dm.p)
    dense_penalty = build_penalty_matrix(
        model._dm.group_matrices,
        model._groups,
        fitted_lambda2(model),
        model._dm.p,
        reml_penalties=model._reml_penalties,
    )

    terms = shape_ops._build_smooth_penalty_terms(model)
    compact_value = shape_ops._smooth_penalty_value(beta, terms)

    assert compact_value == pytest.approx(
        float(beta @ dense_penalty @ beta),
        rel=3e-13,
        abs=3e-13,
    )


def _patch_every_binding(monkeypatch, owner, name, replacement):
    """Replace ``owner.name`` in every ``superglm`` module that holds a reference to it.

    A function imported by name (``from m import f``) is a separate binding in the importing
    module, so patching it on ``owner`` alone is blind to every importer. The importers are found
    by identity over ``sys.modules`` rather than by a list that a new import would outdate.
    """
    original = getattr(owner, name)
    patched = []
    for module_name, module in list(sys.modules.items()):
        if not module_name.startswith("superglm"):
            continue
        for attribute, value in list(vars(module).items()):
            if value is original:
                monkeypatch.setattr(module, attribute, replacement)
                patched.append((module, attribute))
    assert (owner, name) in patched
    return patched


def _importers(owner, name):
    """The ``superglm`` modules (other than ``owner``) that bind ``owner.name`` under ``name``."""
    original = getattr(owner, name)
    return sorted(
        (
            module
            for module_name, module in list(sys.modules.items())
            if module_name.startswith("superglm")
            and module is not owner
            and vars(module).get(name) is original
        ),
        key=lambda module: module.__name__,
    )


def _structured_shape_model(structured_kind: str):
    """A REML fit with a post-fit increasing spline beside a structured (re, fs or sz) term."""
    rng = np.random.default_rng(20260727)
    n_levels = 36
    repeats = 8
    codes = np.repeat(np.arange(n_levels), repeats)
    x = np.tile(np.linspace(0.0, 1.0, repeats), n_levels)
    level_effects = rng.normal(scale=0.12, size=n_levels)
    y = 1.0 - 1.5 * x + level_effects[codes] + rng.normal(scale=0.02, size=len(x))
    X = pd.DataFrame(
        {
            "x": x,
            "group": np.array([f"g{code}" for code in codes], dtype=object),
        }
    )
    features = {
        "x": PSpline(
            n_knots=7,
            constraint=Constraint.postfit.increasing,
            lambda_policy=LambdaPolicy.fixed(0.8),
        )
    }
    interactions = []
    if structured_kind == "re":
        features["group"] = RandomEffect(
            lambda_policy=LambdaPolicy.fixed(1.1),
        )
    else:
        policies = {"wiggle": LambdaPolicy.fixed(1.2)}
        if structured_kind == "fs":
            policies.update(
                null_0=LambdaPolicy.fixed(0.9),
                null_1=LambdaPolicy.fixed(0.9),
            )
        interactions.append(
            FactorSmooth(
                "x",
                group="group",
                basis=structured_kind,
                k=5,
                lambda_policy=policies,
            )
        )
    model = SuperGLM(
        family="gaussian",
        features=features,
        interactions=interactions,
        selection_penalty=0.0,
        direct_solve="structured",
    ).fit_reml(
        X,
        y,
        max_reml_iter=2,
    )
    return model, X


@pytest.mark.parametrize("structured_kind", ["re", "fs", "sz"])
def test_structured_postfit_shape_repair_uses_compact_penalties(
    structured_kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from superglm.reml import penalty_algebra, penalty_support

    model, X = _structured_shape_model(structured_kind)
    x_groups = [group for group in model._groups if group.feature_name == "x"]
    beta_before = np.concatenate([model.result.beta[group.sl] for group in x_groups])
    revision_before = model._fit_revision

    materialize_penalty = penalty_algebra.penalty_component_dense_matrix
    make_support = penalty_support._penalty_support
    context_support = penalty_algebra._PenaltyGroupGeometry.get_support
    dense_penalty_shapes = set()
    support_dimensions = set()

    def reject_dense_component(component, *args, **kwargs):
        if component.penalty_kind != "dense":
            raise AssertionError("post-fit repair expanded a structured penalty component")
        matrix = materialize_penalty(component, *args, **kwargs)
        dense_penalty_shapes.add(matrix.shape)
        assert matrix is component.omega_ssp
        return matrix

    def checked_support(matrices):
        support = make_support(matrices)
        support_dimensions.add((tuple(matrix.shape for matrix in matrices), support.rank))
        return support

    def checked_context(context, *args, **kwargs):
        support = context_support(context, *args, **kwargs)
        support_dimensions.add((tuple(matrix.shape for matrix in context.matrices), support.rank))
        return support

    _patch_every_binding(
        monkeypatch,
        penalty_algebra,
        "penalty_component_dense_matrix",
        reject_dense_component,
    )
    monkeypatch.setattr(penalty_support, "_penalty_support", checked_support)
    monkeypatch.setattr(penalty_algebra._PenaltyGroupGeometry, "get_support", checked_context)

    model.apply_shape_postfit(X, n_grid=120)

    beta_after = np.concatenate([model.result.beta[group.sl] for group in x_groups])
    certificate = shape_constraint_certificate(
        model._specs["x"],
        beta_after,
        "increasing",
    )
    assert model._fit_revision == revision_before + 1
    assert not np.allclose(beta_after, beta_before)
    assert certificate.minimum_scaled_slack >= -2.0e-11
    assert np.all(np.isfinite(model.predict(X)))
    assert model.result.direct_backend == "structured"
    assert dense_penalty_shapes <= {(10, 10)}
    expected_support = {(((10, 10),), 9)}
    if structured_kind == "sz":
        expected_support.add((((5, 5),), 3))
    elif structured_kind == "fs":
        expected_support.add((((5, 5),) * 3, 5))
    assert support_dimensions == expected_support
    compact = next(
        component for component in model._reml_penalties if component.penalty_kind != "dense"
    )
    compact_width = {"re": 36, "fs": 180, "sz": 175}[structured_kind]
    assert compact.group_sl.stop - compact.group_sl.start == compact_width
    with pytest.raises(AssertionError, match="expanded a structured penalty"):
        penalty_algebra.penalty_component_dense_matrix(compact)


def test_structured_guard_covers_every_importer_of_the_materialiser() -> None:
    """The structured-penalty guard must see the modules that bind the materialiser by name."""
    from superglm.reml import gradient, penalty_algebra, w_derivatives

    importers = set(_importers(penalty_algebra, "penalty_component_dense_matrix"))
    assert {gradient, w_derivatives} <= importers


@pytest.mark.parametrize("structured_kind", ["re", "fs", "sz"])
@pytest.mark.parametrize("importer", ["gradient", "w_derivatives"])
def test_expanding_a_structured_penalty_inside_the_repair_reaches_the_guard(
    structured_kind: str,
    importer: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Positive control for the guard in the test above: a repair that did expand would fail.

    The repair's merit evaluation is replaced by one that expands a structured component through
    a module that imports the materialiser by name (the bindings a guard on the defining module
    alone does not reach); the refusal has to come out of ``apply_shape_postfit`` rather than be
    absorbed on the way.
    """
    import importlib

    from superglm.reml import penalty_algebra

    importing_module = importlib.import_module(f"superglm.reml.{importer}")
    model, X = _structured_shape_model(structured_kind)
    structured = next(c for c in model._reml_penalties if c.penalty_kind != "dense")

    def reject_dense_component(component, *args, **kwargs):
        if component.penalty_kind != "dense":
            raise AssertionError("post-fit repair expanded a structured penalty component")
        raise AssertionError("unexpected dense component")

    def merit_that_expands(beta, terms):
        return importing_module.penalty_component_dense_matrix(structured)

    _patch_every_binding(
        monkeypatch, penalty_algebra, "penalty_component_dense_matrix", reject_dense_component
    )
    monkeypatch.setattr(shape_ops, "_smooth_penalty_value", merit_that_expands)

    with pytest.raises(AssertionError, match="expanded a structured penalty"):
        model.apply_shape_postfit(X, n_grid=120)


def _wide_postfit_model(spline_penalty):
    rng = np.random.default_rng(20260725)
    n = 130
    x = np.linspace(0.0, 1.0, n)
    columns = {"x": x}
    columns.update({f"v{j}": rng.normal(size=n) for j in range(90)})
    X = pd.DataFrame(columns)
    y = 1.0 - 1.4 * x + 0.01 * rng.normal(size=n)
    features = {"x": PSpline(n_knots=6, constraint=Constraint.postfit.increasing)}
    features.update({f"v{j}": Numeric() for j in range(90)})
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        spline_penalty=spline_penalty,
        features=features,
    ).fit(X, y)
    assert model._dm.p > 90
    return model, X


@pytest.mark.parametrize("spline_penalty", [0.0, 0.5])
def test_wide_shape_repair_does_not_request_a_global_dense_penalty(
    monkeypatch,
    spline_penalty,
):
    """Publication merit must scale with penalty blocks, not total model width squared."""
    from superglm.reml import penalty_algebra

    model, X = _wide_postfit_model(spline_penalty)

    def reject_global_penalty(*args, **kwargs):
        raise AssertionError("shape repair requested a global dense p-by-p penalty")

    _patch_every_binding(
        monkeypatch, penalty_algebra, "build_penalty_matrix", reject_global_penalty
    )

    model.apply_shape_postfit(X, n_grid=80)

    assert np.all(np.isfinite(model.result.beta))


def test_global_penalty_guard_covers_every_importer_of_the_builder():
    """The guard must see the modules that bind ``build_penalty_matrix`` by name."""
    from superglm.model import reml_finalize
    from superglm.reml import direct, discrete, objective, penalty_algebra

    importers = set(_importers(penalty_algebra, "build_penalty_matrix"))
    assert {objective, direct, discrete, reml_finalize} <= importers


def test_a_guard_on_the_defining_module_alone_misses_the_importers(monkeypatch):
    """Why the guard patches every binding: patching ``penalty_algebra`` leaves importers live."""
    from superglm.reml import penalty_algebra

    model, _ = _wide_postfit_model(0.5)

    def reject_global_penalty(*args, **kwargs):
        raise AssertionError("shape repair requested a global dense p-by-p penalty")

    monkeypatch.setattr(penalty_algebra, "build_penalty_matrix", reject_global_penalty)
    for module in _importers(penalty_algebra, "build_penalty_matrix"):
        built = module.build_penalty_matrix(
            model._dm.group_matrices, model._groups, 1.0, model._dm.p
        )
        assert built.shape == (model._dm.p, model._dm.p)


@pytest.mark.parametrize(
    "importer",
    [
        "superglm.model.reml_finalize",
        "superglm.reml.direct",
        "superglm.reml.discrete",
        "superglm.reml.objective",
    ],
)
def test_a_global_penalty_request_inside_the_repair_reaches_the_guard(monkeypatch, importer):
    """Positive control: were the repair to ask for the global penalty, the guard would fire.

    The repair's merit evaluation is replaced by one that builds the global penalty through a
    module that imports the builder by name, the bindings a guard on ``penalty_algebra`` alone
    does not reach; the refusal must come out of ``apply_shape_postfit`` rather than be
    swallowed on the way (the fitted-state transaction catches and re-raises).
    """
    import importlib

    from superglm.reml import penalty_algebra

    importing_module = importlib.import_module(importer)
    model, X = _wide_postfit_model(0.5)

    def reject_global_penalty(*args, **kwargs):
        raise AssertionError("shape repair requested a global dense p-by-p penalty")

    def merit_that_builds_the_global_penalty(beta, terms):
        return importing_module.build_penalty_matrix(
            model._dm.group_matrices, model._groups, 1.0, model._dm.p
        )

    _patch_every_binding(
        monkeypatch, penalty_algebra, "build_penalty_matrix", reject_global_penalty
    )
    monkeypatch.setattr(shape_ops, "_smooth_penalty_value", merit_that_builds_the_global_penalty)

    with pytest.raises(AssertionError, match="global dense p-by-p penalty"):
        model.apply_shape_postfit(X, n_grid=80)


def test_a_production_global_penalty_request_reaches_the_guard(monkeypatch):
    """Positive control through production code: the REML objective builds the penalty itself.

    The merit evaluation calls ``reml_laml_objective`` without retained penalty geometry, the
    path that builds the dense penalty inside the objective module.
    """
    from superglm.model.shape_ops import model_weight_semantics
    from superglm.reml import penalty_algebra
    from superglm.reml.objective import reml_laml_objective

    model, X = _wide_postfit_model(0.5)

    def reject_global_penalty(*args, **kwargs):
        raise AssertionError("shape repair requested a global dense p-by-p penalty")

    def merit_that_evaluates_the_reml_objective(beta, terms):
        return reml_laml_objective(
            model._dm,
            model._distribution,
            model._link,
            model._groups,
            np.asarray(model._fit_y_ref, dtype=np.float64),
            model._solver_result,
            {group.name: 1.0 for group in model._groups},
            np.asarray(model._fit_weights, dtype=np.float64),
            np.zeros(len(model._fit_y_ref)),
            weight_semantics=model_weight_semantics(model),
        )

    _patch_every_binding(
        monkeypatch, penalty_algebra, "build_penalty_matrix", reject_global_penalty
    )
    monkeypatch.setattr(shape_ops, "_smooth_penalty_value", merit_that_evaluates_the_reml_objective)

    with pytest.raises(AssertionError, match="global dense p-by-p penalty"):
        model.apply_shape_postfit(X, n_grid=80)


def test_degree_zero_monotone_postfit_certifies_and_repairs_knot_jumps():
    """A step spline is monotone only when every one-sided knot jump has the right sign."""
    x = np.linspace(0.0, 1.0, 120)
    X = pd.DataFrame({"x": x})
    y = 1.0 - 2.0 * x
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={
            "x": PSpline(
                n_knots=7,
                degree=0,
                m=1,
                constraint=Constraint.postfit.increasing,
            )
        },
    ).fit(X, y)
    spec = model._specs["x"]
    groups = [group for group in model._groups if group.feature_name == "x"]
    beta_before = np.concatenate([model.result.beta[group.sl] for group in groups])
    breakpoints = np.unique(spec._knots[(spec._knots > spec._lo) & (spec._knots < spec._hi)])

    def knot_jumps(beta):
        left = np.nextafter(breakpoints, -np.inf)
        right = np.nextafter(breakpoints, np.inf)
        return spec.transform(right) @ beta - spec.transform(left) @ beta

    assert np.min(knot_jumps(beta_before)) < -1e-2
    assert (
        shape_constraint_certificate(spec, beta_before, "increasing").minimum_scaled_slack < -1e-2
    )

    old_revision = model._fit_revision
    model.apply_shape_postfit(X)
    beta_after = np.concatenate([model.result.beta[group.sl] for group in groups])

    assert model._fit_revision == old_revision + 1
    assert "x" in model._shape_repairs
    assert np.min(knot_jumps(beta_after)) >= -2e-11
    assert (
        shape_constraint_certificate(spec, beta_after, "increasing").minimum_scaled_slack >= -2e-11
    )


@pytest.mark.parametrize("kind", ["convex", "concave"])
def test_degree_zero_postfit_curvature_is_rejected_at_configuration(kind):
    """A discontinuous degree-zero spline has no classical convex/concave curve geometry."""
    constraint = getattr(Constraint.postfit, kind)
    with pytest.raises(ValueError, match=r"degree=0.*convex|degree=0.*concave"):
        PSpline(n_knots=7, degree=0, m=1, constraint=constraint)


def test_monotone_certificate_is_invariant_to_large_predictor_units():
    """Derivative rows measured in large x units must not be mistaken for zero geometry."""
    x = np.linspace(0.0, 1.0e16, 160)
    X = pd.DataFrame({"x": x})
    y = 1.0 - 2.0 * x / x[-1]
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"x": PSpline(n_knots=9, constraint=Constraint.postfit.increasing)},
    ).fit(X, y)
    spec = model._specs["x"]
    groups = [group for group in model._groups if group.feature_name == "x"]
    beta_before = np.concatenate([model.result.beta[group.sl] for group in groups])

    certificate_before = shape_constraint_certificate(spec, beta_before, "increasing")
    assert certificate_before.maximum_row_norm < 1e-12
    assert certificate_before.minimum_scaled_slack < -1e-3

    old_revision = model._fit_revision
    model.apply_shape_postfit(X)
    beta_after = np.concatenate([model.result.beta[group.sl] for group in groups])
    certificate_after = shape_constraint_certificate(spec, beta_after, "increasing")

    assert model._fit_revision == old_revision + 1
    assert certificate_after.minimum_scaled_slack >= -2e-11
    dense_curve = spec.reconstruct(beta_after, n_points=4001)["log_relativity"]
    assert np.min(np.diff(dense_curve)) >= -2e-11


def test_postfit_recertifies_after_editor_invalidates_an_installed_repair():
    from superglm.editor import EditorSession

    x = np.linspace(0.0, 1.0, 240)
    X = pd.DataFrame({"x": x})
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"x": PSpline(n_knots=9, constraint=Constraint.postfit.increasing)},
    ).fit(X, 1.0 - 2.0 * x)
    model.apply_shape_postfit(X)
    assert "x" in model._shape_repairs

    with pytest.warns(UserWarning, match="post-fit shape projection"):
        session = EditorSession.from_model(model, terms=["x"])
    session.terms["x"].edited_log_effect[:] = np.linspace(1.0, -1.0, session.terms["x"].size)
    edited = session.to_model()
    groups = [group for group in edited._groups if group.feature_name == "x"]
    beta_before = np.concatenate([edited.result.beta[group.sl] for group in groups])
    assert (
        shape_constraint_certificate(
            edited._specs["x"], beta_before, "increasing"
        ).minimum_scaled_slack
        < -1e-3
    )
    old_revision = edited._fit_revision

    edited.apply_shape_postfit(X)
    beta_after = np.concatenate([edited.result.beta[group.sl] for group in groups])

    assert edited._fit_revision == old_revision + 1
    assert (
        shape_constraint_certificate(
            edited._specs["x"], beta_after, "increasing"
        ).minimum_scaled_slack
        >= -2e-11
    )


def test_postfit_defaults_grid_projection_to_retained_fit_weights():
    x = np.linspace(0.0, 1.0, 500)
    X = pd.DataFrame({"x": x})
    y = x + 0.3 * np.sin(4.0 * np.pi * x)
    weights = np.where(x < 0.35, 50.0, 1.0)

    def fitted_model():
        return SuperGLM(
            family="gaussian",
            selection_penalty=0.0,
            features={"x": PSpline(n_knots=12, constraint=Constraint.postfit.increasing)},
        ).fit(X, y, sample_weight=weights)

    implicit = fitted_model()
    explicit = fitted_model()
    implicit.apply_shape_postfit(X)
    explicit.apply_shape_postfit(X, sample_weight=weights)

    np.testing.assert_allclose(implicit.result.beta, explicit.result.beta, rtol=0.0, atol=2e-12)
    assert implicit.result.deviance == pytest.approx(explicit.result.deviance, abs=2e-10)


def _fitted_explicit_knot_spec(*, degree, knots, constraint):
    x = np.linspace(0.0, 1.0, 101)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={
            "x": PSpline(
                knots=knots,
                boundary=(0.0, 1.0),
                degree=degree,
                m=1,
                penalty="none",
                constraint=constraint,
            )
        },
    ).fit(pd.DataFrame({"x": x}), np.sin(x))
    return model._specs["x"]


def test_certificate_keeps_constraints_from_unequal_knot_spans():
    """A tiny span must not erase derivative constraints on ordinary-width spans."""
    spec = _fitted_explicit_knot_spec(
        degree=1,
        knots=[0.4, 0.4 + 1e-15, 0.8],
        constraint=Constraint.postfit.increasing,
    )
    beta = np.array([-1.08544491, 0.0, 0.0, 0.0])
    points = np.unique(
        np.concatenate(
            (
                [spec._lo, spec._hi],
                spec._knots[(spec._knots >= spec._lo) & (spec._knots <= spec._hi)],
            )
        )
    )
    rows = shape_derivative_matrix(spec, points, 1)
    row_norms = np.linalg.norm(rows, axis=1)
    expected_slack = float(np.min((rows[row_norms > 0.0] @ beta) / row_norms[row_norms > 0.0]))

    assert np.max(row_norms) / np.min(row_norms[row_norms > 0.0]) > 1e12
    assert expected_slack < -0.05
    assert shape_constraint_certificate(
        spec, beta, "increasing"
    ).minimum_scaled_slack == pytest.approx(expected_slack, rel=2e-12, abs=2e-12)


def test_degree_zero_certificate_resolves_adjacent_float_knot_jumps():
    """One-sided jump rows must not skip a span only one floating-point ULP wide."""
    knot = 0.4
    spec = _fitted_explicit_knot_spec(
        degree=0,
        knots=[knot, np.nextafter(knot, np.inf), 0.8],
        constraint=Constraint.postfit.increasing,
    )
    beta = np.array([-100.0, 100.0, 99.62583426])
    raw_beta = spec._R_inv @ beta

    assert np.min(np.diff(raw_beta)) < -0.05
    assert shape_constraint_certificate(spec, beta, "increasing").minimum_scaled_slack < -0.05


def test_degree_one_curvature_resolves_adjacent_float_slope_jumps():
    """Piecewise-linear slope jumps must be formed from spans, not nextafter probes."""
    knot = 0.4
    spec = _fitted_explicit_knot_spec(
        degree=1,
        knots=[knot, np.nextafter(knot, np.inf), 0.8],
        constraint=Constraint.postfit.convex,
    )
    beta = np.array([0.64840049, -0.7620604, -0.31128911, 0.0])
    n_raw = len(spec._knots) - spec.degree - 1
    derivative_basis = BSpline(
        spec._knots,
        np.eye(n_raw),
        spec.degree,
        extrapolate=False,
    ).derivative(1)
    breakpoints = np.unique(spec._knots[(spec._knots >= spec._lo) & (spec._knots <= spec._hi)])
    exact_rows = []
    for point in breakpoints:
        previous = np.max(spec._knots[spec._knots < point])
        exact_rows.append((derivative_basis(point) - derivative_basis(previous)) @ spec._R_inv)
    exact_rows = np.asarray(exact_rows)
    row_norms = np.linalg.norm(exact_rows, axis=1)
    expected_slack = float(
        np.min((exact_rows[row_norms > 0.0] @ beta) / row_norms[row_norms > 0.0])
    )

    assert expected_slack < -0.5
    assert shape_constraint_certificate(spec, beta, "convex").minimum_scaled_slack == pytest.approx(
        expected_slack, rel=2e-8, abs=2e-8
    )


@pytest.mark.parametrize("kind", ["increasing", "convex"])
def test_summary_withholds_term_inference_after_a_shape_repair(kind):
    """A repair republishes coefficients from a CONSTRAINED estimator, and the
    reference distribution of that estimator is not the unconstrained fit's.

    Meyer & Woodroofe (*Ann. Statist.* 28(4):1083-1104, 2000) make the effective
    dimension of a shape-restricted fit depend on the number of active cone
    edges, which is a random variable; Meyer (*Ann. Appl. Statist.*
    2(3):1013-1033, 2008, Sec. 3) makes the null a mixture rather than the
    fixed-df chi-square the Wood test computes. A binding repair sits on the
    boundary, which is precisely where the constrained and unconstrained
    estimators stop agreeing. So the term test, ``ref_df`` and the curve SE band
    are withheld rather than printed against the wrong null -- issue #308.

    Both arms matter for a second reason: the engine records a curvature repair
    in ``_shape_repairs`` alone, and every reporting surface read
    ``_monotone_repairs``, so a ``convex`` repair was neither marked nor
    withheld anywhere.
    """
    rng = np.random.default_rng(0)
    x = np.linspace(0.0, 1.0, 400)
    truth = -((x - 0.35) ** 2) if kind == "convex" else np.where(x < 0.5, x, 1.2 - 1.4 * x)
    y = truth + 0.05 * rng.normal(size=len(x))
    df = pd.DataFrame({"x": x})

    constraint = getattr(Constraint.postfit, kind)
    model = SuperGLM(
        family="gaussian", features={"x": PSpline(n_knots=10, constraint=constraint)}
    ).fit(df, y)

    def _spline_row(m):
        (row,) = [r for r in m.summary()._coef_rows if r.group == "x" and r.is_spline]
        return row

    # Before the repair the term reports everything, or the assertions below
    # would pass against a term that never had a test to withhold.
    before = _spline_row(model)
    assert before.wald_p is not None and np.isfinite(before.wald_p)
    assert before.ref_df is not None
    assert before.curve_se_min is not None
    assert before.monotone_repaired is False

    model.apply_shape_postfit(df)
    assert model._shape_repairs["x"].max_violation_before > 0.0  # it bound

    after = _spline_row(model)
    assert after.monotone_repaired is True
    assert after.wald_chi2 is None
    assert after.wald_p is None
    assert after.ref_df is None
    assert after.curve_se_min is None
    assert after.curve_se_max is None
    # The point estimate is NOT withheld: projecting onto the cone is a weak
    # L_p improvement whatever produced the original estimate (Chernozhukov,
    # Fernandez-Val & Galichon, *Biometrika* 96(3):559-575, 2009, Prop. 1).
    assert after.edf is not None and np.isfinite(after.edf)

    # Rendered, in words, on both surfaces -- a row that merely lost its test
    # is indistinguishable from an inactive one otherwise.
    assert "repaired (inference withheld)" in str(model.summary())
    assert "repaired (inference withheld)" in model.summary()._repr_html_()

    # Coefficient-level uncertainty goes with it, or the same summary says
    # "inference withheld" and then prints the withheld quantities immediately
    # below: the level rows carry per-level SEs and intervals from the same
    # unconstrained covariance, and `detail="full"` prints a per-coefficient
    # SE, z, p and interval under the term line.
    for row in model.summary(detail="full")._coef_rows:
        if row.group != "x":
            continue
        assert row.se is None
        assert row.z is None
        assert row.p is None
        assert row.ci_low is None
        assert row.ci_high is None
    full = model.summary(detail="full")
    assert not [name for name in full._basis_detail if name.startswith("x")]

    # And on `metrics().summary()`, which built its rows without the repairs
    # dict at all and so printed a p-value the model itself had withheld.
    metrics_summary = model.metrics(df, y).summary(detail="full")
    (metrics_row,) = [r for r in metrics_summary._coef_rows if r.group == "x" and r.is_spline]
    assert metrics_row.monotone_repaired is True
    assert metrics_row.wald_p is None
    # ... including the per-coefficient disclosure, which `ModelMetrics.summary`
    # builds itself and which the HTML renderer emits unconditionally, so
    # missing it here would reach a notebook at the default detail level.
    assert not [name for name in metrics_summary._basis_detail if name.startswith("x")]
    assert "repaired (inference withheld)" in metrics_summary._repr_html_()


def test_an_already_feasible_term_beside_a_repaired_one_keeps_its_inference():
    """The early exit is a statement about the WHOLE model: once any one term
    binds, the loop runs the repairer for every pending term, and an
    already-feasible one comes back with its coefficients copied through
    unchanged.

    Recording that as a repair used to cost a cosmetic marker. With #308's
    withholding it would cost the term its entire published inference -- and the
    reason for withholding does not apply to it, because no cone projection
    happened, so there is no active edge and no boundary to sit on. The
    parametrized test above cannot see this case: it carries one term, so the
    early exit fires whenever that term is feasible.
    """
    rng = np.random.default_rng(3)
    n = 400
    x = np.linspace(0.0, 1.0, n)
    dips = np.where(x < 0.5, x, 1.2 - 1.4 * x)  # must be projected
    rises = 2.0 * x  # already increasing
    df = pd.DataFrame({"a": x, "b": rng.permutation(x)})
    y = dips + np.interp(df["b"], x, rises) + 0.05 * rng.normal(size=n)

    model = SuperGLM(
        family="gaussian",
        features={
            "a": PSpline(n_knots=10, constraint=Constraint.postfit.increasing),
            "b": PSpline(n_knots=10, constraint=Constraint.postfit.increasing),
        },
    ).fit(df, y)
    before = {r.group: r for r in model.summary()._coef_rows if r.is_spline}

    model.apply_shape_postfit(df)

    # The premise: exactly one term bound. Without it the test measures nothing
    # -- if `b` also needed projecting there is no feasible term to protect, and
    # if `a` did not the whole call is an early-exit no-op.
    assert set(model._shape_repairs) == {"a"}

    after = {r.group: r for r in model.summary()._coef_rows if r.is_spline}
    assert after["a"].wald_p is None
    assert after["a"].monotone_repaired is True
    # `b` was never projected, so it keeps its inference. Present and finite,
    # NOT bit-identical to `before`: the repair refreshes the model's
    # dispersion and effective df, and every standard error is scaled by
    # those, so an untouched term's SE legitimately moves. What must not
    # happen is that it disappears.
    assert after["b"].monotone_repaired is False
    for field in ("wald_chi2", "wald_p", "ref_df", "curve_se_min", "curve_se_max"):
        value = getattr(after["b"], field)
        assert value is not None, field
        assert np.all(np.isfinite(np.asarray(value, dtype=float))), field
    assert before["b"].wald_p is not None  # it had one to keep


def test_term_level_uncertainty_apis_qualify_a_repaired_term():
    """`summary()` is not the only surface that publishes a band.

    `term_inference()` is the public curve-plus-uncertainty API and the one
    `plot()`/`plot_data()` funnel through, and it hard-coded
    `monotone_repaired=False` while returning `se_log_relativity` and intervals
    from the unconstrained fit's covariance. `simultaneous_bands()` simulates
    from that same covariance around constrained coefficients, which is not a
    band around anything that was fitted. Both now qualify a repaired term the
    way the editor-stale path already qualifies an edited one.
    """
    rng = np.random.default_rng(0)
    x = np.linspace(0.0, 1.0, 400)
    y = np.where(x < 0.5, x, 1.2 - 1.4 * x) + 0.05 * rng.normal(size=len(x))
    df = pd.DataFrame({"x": x})
    model = SuperGLM(
        family="gaussian",
        features={"x": PSpline(n_knots=10, constraint=Constraint.postfit.increasing)},
    ).fit(df, y)

    # Before: the band is published and the flag is silent about the repair.
    baseline = model.term_inference("x", with_se=True)
    assert baseline.se_log_relativity is not None
    assert baseline.monotone_repaired is False
    model.simultaneous_bands("x")  # available

    model.apply_shape_postfit(df)
    assert model._shape_repairs["x"].max_violation_before > 0.0

    with pytest.warns(UserWarning, match="post-fit shape projection"):
        after = model.term_inference("x", with_se=True)
    assert after.se_log_relativity is None
    assert after.ci_lower is None
    assert after.monotone_repaired is True
    # The curve itself is kept -- the projection improves the point estimate.
    assert after.log_relativity is not None
    assert np.all(np.isfinite(np.asarray(after.log_relativity, dtype=float)))

    with pytest.raises(RuntimeError, match="post-fit shape repair"):
        model.simultaneous_bands("x")

    # `metrics().feature_se` is the same band by another door -- a per-feature
    # accessor, so it can refuse exactly rather than be caveated in prose.
    with pytest.raises(RuntimeError, match="post-fit shape repair"):
        model.metrics(df, y).feature_se("x")

    # An unrepaired term in the same model is untouched.
    assert model.term_inference("x", with_se=False).se_log_relativity is None


def test_relativities_drops_the_band_of_the_repaired_feature_only():
    """`relativities(with_se=True)` publishes `se_log_relativity` from the same
    covariance `term_inference` now withholds.

    Blanked per FEATURE, not per call: this API returns every term at once, and
    dropping the band on unrepaired terms because one term was projected would
    be exactly the over-reach the repair record itself avoids.
    """
    rng = np.random.default_rng(4)
    n = 400
    x = np.linspace(0.0, 1.0, n)
    df = pd.DataFrame({"a": x, "b": rng.permutation(x)})
    y = np.where(x < 0.5, x, 1.2 - 1.4 * x) + 2.0 * df["b"] + 0.05 * rng.normal(size=n)

    model = SuperGLM(
        family="gaussian",
        features={
            "a": PSpline(n_knots=10, constraint=Constraint.postfit.increasing),
            "b": PSpline(n_knots=10),
        },
    ).fit(df, y)
    before = model.relativities(with_se=True)
    assert "se_log_relativity" in before["a"].columns
    assert "se_log_relativity" in before["b"].columns

    model.apply_shape_postfit(df)
    assert set(model._shape_repairs) == {"a"}

    with pytest.warns(UserWarning, match="post-fit shape projection"):
        after = model.relativities(with_se=True)
    assert "se_log_relativity" not in after["a"].columns
    assert "se_log_relativity" in after["b"].columns
    # The curve survives in both.
    assert "log_relativity" in after["a"].columns
    assert np.all(np.isfinite(after["a"]["log_relativity"].to_numpy(dtype=float)))


def test_the_repaired_frame_is_found_under_a_non_string_feature_label():
    """`_shape_repaired_features` stringifies so one spelling answers for every
    surface; `relativities()` keys its output by the ORIGINAL feature label,
    which need not be a string.

    Intersecting the two key sets is therefore empty for a feature named `0`,
    and the guard goes silently inert -- the same failure mode as every defect
    this PR fixes, reintroduced one layer up. Asserted on a column label that
    is an int, since that is the reachable non-string case.
    """
    rng = np.random.default_rng(5)
    n = 400
    x = np.linspace(0.0, 1.0, n)
    df = pd.DataFrame({0: x})
    y = np.where(x < 0.5, x, 1.2 - 1.4 * x) + 0.05 * rng.normal(size=n)

    model = SuperGLM(
        family="gaussian",
        features={0: PSpline(n_knots=10, constraint=Constraint.postfit.increasing)},
    ).fit(df, y)
    assert "se_log_relativity" in model.relativities(with_se=True)[0].columns

    model.apply_shape_postfit(df)
    assert set(model._shape_repairs) == {0}

    with pytest.warns(UserWarning, match="post-fit shape projection"):
        after = model.relativities(with_se=True)
    assert "se_log_relativity" not in after[0].columns
    # The sibling surfaces resolve the same label too.
    with pytest.warns(UserWarning, match="post-fit shape projection"):
        assert model.term_inference(0, with_se=True).se_log_relativity is None
    (row,) = [r for r in model.summary()._coef_rows if r.group == "0" and r.is_spline]
    assert row.wald_p is None


# -- publication refusals: fault injection at the producer, nothing published --------------


def _fitted_for_publication_refusal(family="gaussian", scale=1.0):
    """A fitted model whose convex post-fit repair binds (the data are concave).

    ``scale`` is the unit of a Gaussian response.
    """
    rng = np.random.default_rng(20261024)
    x = np.linspace(0.0, 1.0, 240)
    X = pd.DataFrame({"x": x})
    if family == "poisson":
        y = rng.poisson(np.exp(1.2 - 2.0 * (x - 0.45) ** 2)).astype(np.float64)
    else:
        y = scale * (-((x - 0.4) ** 2) + 0.03 * rng.normal(size=x.size))
    model = SuperGLM(
        family=family,
        selection_penalty=0.0,
        features={"x": PSpline(n_knots=9, constraint=Constraint.postfit.convex)},
    ).fit(X, y)
    return model, X


def _publication_snapshot(model, X):
    return {
        "dict": model.__dict__,
        "result": model.result,
        "revision": model._fit_revision,
        "beta": model.result.beta.copy(),
        "predictions": model.predict(X),
    }


def _assert_nothing_published(model, X, snapshot):
    assert model.__dict__ is snapshot["dict"]
    assert model.result is snapshot["result"]
    assert model._fit_revision == snapshot["revision"]
    assert not getattr(model, "_shape_repairs", {})
    assert not getattr(model, "_monotone_repairs", {})
    np.testing.assert_array_equal(model.result.beta, snapshot["beta"])
    np.testing.assert_array_equal(model.predict(X), snapshot["predictions"])


def _repairer_with_fault(monkeypatch, mutate):
    """Route the real repair through ``mutate(result, group_slices)`` before validation."""
    real_repairer = shape_ops._repairer

    def faulty_repairer(kind):
        delegate = real_repairer(kind)

        class Faulty:
            def repair(self, spec, beta, groups, *, weights, n_grid, metric):
                result = delegate.repair(
                    spec, beta, groups, weights=weights, n_grid=n_grid, metric=metric
                )
                mutate(result, spec, groups)
                return result

        return Faulty()

    monkeypatch.setattr(shape_ops, "_repairer", faulty_repairer)


def _fault_non_finite_coefficients(result, spec, groups):
    result.repaired_beta_reparam = result.repaired_beta_reparam.copy()
    result.repaired_beta_reparam[groups[0].start] = np.nan


def _fault_wrong_coefficient_shape(result, spec, groups):
    result.repaired_beta_reparam = result.repaired_beta_reparam[:-1].copy()


def _fault_degenerate_grid(result, spec, groups):
    result.grid = result.grid[:2].copy()


def _fault_worse_than_zero_term(result, spec, groups):
    # Scaling stays inside the convex cone, so only the objective can object.
    scaled = result.repaired_beta_reparam.copy()
    for group in groups:
        scaled[group.sl] *= 1e3
    result.repaired_beta_reparam = scaled


@pytest.mark.parametrize(
    ("fault", "message"),
    [
        (_fault_non_finite_coefficients, "invalid coefficients"),
        (_fault_wrong_coefficient_shape, "invalid coefficients"),
        (_fault_degenerate_grid, "invalid repair grid"),
        (
            _fault_worse_than_zero_term,
            "candidate objective is worse than the feasible zero-term fallback",
        ),
    ],
)
def test_faulty_repair_result_is_refused_and_nothing_is_published(monkeypatch, fault, message):
    model, X = _fitted_for_publication_refusal()
    snapshot = _publication_snapshot(model, X)
    _repairer_with_fault(monkeypatch, fault)

    with pytest.raises(RuntimeError, match=f"Unsafe shape repair rejected.*{message}"):
        model.apply_shape_postfit(X)

    _assert_nothing_published(model, X, snapshot)


def test_continued_repair_is_judged_by_the_same_rule(monkeypatch):
    """The rule judges a continued repair as it judged the candidate it continues.

    The first projection, scaled by 1e3, stays in the cone and loses to the zero term,
    so the repair is continued; the continuation's result, scaled by 1e3 too, loses as
    well, is refused, and nothing is published.
    """
    model, X = _fitted_for_publication_refusal()
    snapshot = _publication_snapshot(model, X)
    projections, continued = [], []

    def first_scaled(result, spec, groups):
        if not projections:
            _fault_worse_than_zero_term(result, spec, groups)
        projections.append(None)

    _repairer_with_fault(monkeypatch, first_scaled)
    real_continued = shape_ops._continued_repair

    def scaled_continuation(*args, **kwargs):
        result = real_continued(*args, **kwargs)
        continued.append(result)
        _fault_worse_than_zero_term(result[0], kwargs["spec"], kwargs["groups"])
        return result

    monkeypatch.setattr(shape_ops, "_continued_repair", scaled_continuation)

    with pytest.raises(RuntimeError, match="Unsafe shape repair rejected.*worse than the feasible"):
        model.apply_shape_postfit(X)

    assert len(continued) == 1 and continued[0] is not None
    _assert_nothing_published(model, X, snapshot)


def _pushed_continuation(monkeypatch, push):
    """Route every continued projection through ``push(spec, grid, scales)``, the block it
    returns, where ``scales`` lists each projection's certified curve scale, the first's first."""
    import dataclasses

    real = shape_ops._repairer
    scales = []

    class Pushed:
        def __init__(self, kind):
            self.inner = real(kind)

        def repair(self, spec, beta, groups, **kwargs):
            result = self.inner.repair(spec, beta, groups, **kwargs)
            weights = np.asarray(kwargs["weights"], dtype=np.float64)
            scales.append(shape_ops._certified_curve_scale(result.original_log_effect, weights))
            if len(scales) == 1:
                return result
            block = push(spec, np.asarray(result.grid, dtype=np.float64), scales)
            full = np.array(result.repaired_beta_reparam, dtype=np.float64, copy=True)
            offset = 0
            for group in groups:
                full[group.sl] = block[offset : offset + group.size]
                offset += group.size
            # the repair's certificate is of the block it returned, not of this one
            return dataclasses.replace(result, repaired_beta_reparam=full, certificate=None)

    monkeypatch.setattr(shape_ops, "_repairer", Pushed)
    return scales


def test_a_continued_projection_is_judged_at_the_scale_it_was_certified_at(monkeypatch):
    """A continued repair is read at the scale its projection certified it at.

    The continuation projects the merit's Newton target, whose curve is 2.77 in weighted RMS
    against the fitted curve's 0.79, and reaches the zero curve, which the repairer certifies
    to the rounding at the larger scale. At 1984d54f3 publication read the result at the
    fitted curve's scale. Forced: each continued projection is replaced by the zero curve
    moved along one row of the rule, by more than the fitted term's rounding and less than
    the target's. That was refused as breaking the rule; it is zero to the rounding it was
    certified at, and is published as the zero curve.
    """
    model, X, _ = _sweep_case(25, True, "lognormal", "abs", "gamma", "decreasing")
    fitted = _term_beta(model.result.beta, _x_groups(model))
    rounding = float(gamma(_SLACK_STEPS * fitted.size))
    sent = []

    def push(spec, grid, scales):
        row = shape_derivative_matrix(spec, [np.median(grid)], 1)[0]
        size = rounding * np.sqrt(np.linalg.norm(fitted) * scales[-1])
        sent.append((size * row / np.linalg.norm(row), scales[0], scales[-1]))
        return sent[-1][0]  # rising there, so it breaks the decreasing rule

    _pushed_continuation(monkeypatch, push)
    model.apply_shape_postfit(X)

    block, first, last = sent[-1]
    certificate = shape_constraint_certificate(model._specs["x"], block, "decreasing")
    assert certificate.minimum_scaled_slack < -rounding * first
    assert rounding * np.linalg.norm(fitted) < np.linalg.norm(block) < rounding * last
    assert "x" in model._shape_repairs
    assert not np.any(_term_beta(model.result.beta, _x_groups(model)))


def _sweep_case(n_knots, select, axis, shape, family, kind):
    """One repair of the post-fit sweep: n = 2500, drawn in the sweep's order."""
    rng = np.random.default_rng(7)
    x = {
        "uniform": lambda: rng.uniform(0.0, 1.0, 2500),
        "exponential": lambda: rng.exponential(1.0, 2500),
        "lognormal": lambda: rng.lognormal(0.0, 0.8, 2500),
    }[axis]()
    if shape == "bump":
        f = 0.5 * np.exp(-(((x - np.quantile(x, 0.3)) / np.std(x)) ** 2))
    else:
        f = 0.5 * np.abs(x - np.median(x)) / np.std(x)
    rng.normal(size=x.size)
    counts = rng.poisson(np.exp(f)).astype(np.float64)
    y = counts if family == "poisson" else rng.gamma(2.0, np.exp(f) / 2.0)
    X = pd.DataFrame({"x": x})
    feature = PSpline(n_knots=n_knots, select=select, constraint=getattr(Constraint.postfit, kind))
    model = SuperGLM(family=family, selection_penalty=0.0, features={"x": feature})
    return model.fit(X, y), X, y


def _log_link_merit(model, X, y):
    """The rule's merit of a one-term log-link Poisson or Gamma fit, formed without its code.

    ``b'Sb`` takes ``S`` from the fit's posterior precision less its data part, and the
    deviance the intercept's closed form: ``log(sum y / sum e^eta)`` for Poisson and
    ``log(mean(y e^-eta))`` for Gamma. Returns ``b -> (merit, gradient, Hessian)``, the
    last two of half the merit in its Fisher form, all with the intercept profiled.
    """
    poisson = model._distribution.__class__.__name__ == "Poisson"
    basis = np.asarray(model._specs["x"].transform(X["x"].to_numpy()), dtype=np.float64)

    def data_part(weights):
        cross = basis.T @ weights
        return basis.T @ (basis * weights[:, None]) - np.outer(cross, cross) / weights.sum()

    precision, _ = _posterior_metric(model)
    fitted_mean = model.predict(X)
    S = precision - data_part(fitted_mean if poisson else np.ones_like(y))

    def at(b):
        eta = basis @ b
        if poisson:
            mu = np.exp(eta) * y.sum() / np.exp(eta).sum()
            units = 2.0 * (np.where(y > 0, y * np.log(np.where(y > 0, y, 1.0) / mu), 0.0) - y + mu)
            residual, weights = y - mu, mu
        else:
            mu = np.exp(eta) * np.mean(y * np.exp(-eta))
            units = 2.0 * (np.log(mu / y) + (y - mu) / mu)
            residual, weights = (y - mu) / mu, np.ones_like(y)
        return units.sum() + b @ S @ b, S @ b - basis.T @ residual, data_part(weights) + S

    return at


@pytest.mark.parametrize(
    ("n_knots", "select", "axis", "shape", "family", "kind"),
    [
        (12, False, "uniform", "bump", "poisson", "increasing"),
        (25, True, "uniform", "bump", "poisson", "increasing"),
        (25, True, "exponential", "abs", "gamma", "decreasing"),
    ],
)
def test_projection_the_zero_term_beats_is_continued_to_the_constrained_fit(
    n_knots, select, axis, shape, family, kind
):
    """Repairs master publishes, which the one-step projection left worse than the zero term.

    The projection in the penalized Hessian is exact for a quadratic merit only; far from
    a log-link fit it lost to the zero term in true merit, by 0.0085 to 0.064, and the
    rule refused it. Continued to the constrained fit, the repair publishes the true
    merit's minimiser over the cone, no worse than the zero term master published.
    """
    from tests._shape_projection_oracle import projection_certificate

    model, X, y = _sweep_case(n_knots, select, axis, shape, family, kind)
    merit = _log_link_merit(model, X, y)
    fitted = _term_beta(model.result.beta, _x_groups(model))
    zero_merit = merit(np.zeros_like(fitted))[0]

    model.apply_shape_postfit(X)

    beta = _term_beta(model.result.beta, _x_groups(model))
    published, gradient, hessian = merit(beta)
    assert published <= zero_merit * (1.0 + 1e-12)
    certificate = projection_certificate(
        model._specs["x"], kind, beta, hessian, hessian @ beta - gradient, fitted
    )
    assert certificate.minimum_derivative >= -1e-10
    assert certificate.kkt_residual <= CLOSEST_REPAIR_KKT


def test_non_finite_repair_curve_is_refused_and_nothing_is_published(monkeypatch):
    model, X = _fitted_for_publication_refusal()
    snapshot = _publication_snapshot(model, X)
    spec = model._specs["x"]
    real_transform = spec.transform
    short_grid = 7

    def shorten_grid(result, _spec, _groups):
        result.grid = np.linspace(_spec._lo, _spec._hi, short_grid)

    def transform_with_overflow(values, *args, **kwargs):
        basis = real_transform(values, *args, **kwargs)
        # Only the publication certificate evaluates the fitted basis on the short grid.
        # NaN stands in for a basis that overflowed.
        return np.full_like(basis, np.nan) if np.shape(values) == (short_grid,) else basis

    _repairer_with_fault(monkeypatch, shorten_grid)
    monkeypatch.setattr(spec, "transform", transform_with_overflow)

    with pytest.raises(RuntimeError, match="Unsafe shape repair rejected.*non-finite repair curve"):
        model.apply_shape_postfit(X)

    _assert_nothing_published(model, X, snapshot)


def test_repair_that_moves_the_fitted_mean_is_refused_and_nothing_is_published(monkeypatch):
    model, X = _fitted_for_publication_refusal()
    snapshot = _publication_snapshot(model, X)
    real_delta = shape_ops._shape_term_eta_delta
    calls = 0

    def delta_with_mean_shift(*args, **kwargs):
        nonlocal calls
        calls += 1
        delta = real_delta(*args, **kwargs)
        # The first evaluation is the candidate repair, the second the zero-term fallback.
        return delta + 1.0 if calls == 1 else delta

    monkeypatch.setattr(shape_ops, "_shape_term_eta_delta", delta_with_mean_shift)

    with pytest.raises(RuntimeError, match="Unsafe shape repair rejected.*centering changed"):
        model.apply_shape_postfit(X)

    assert calls >= 1
    _assert_nothing_published(model, X, snapshot)


def test_intercept_search_that_cannot_converge_is_refused_and_nothing_is_published(monkeypatch):
    model, X = _fitted_for_publication_refusal(family="poisson")
    snapshot = _publication_snapshot(model, X)

    def vast_information(**kwargs):
        # A Newton step of score/information ~ 1e-30 never reaches the score tolerance.
        return np.full(np.shape(kwargs["mu"]), 1e30)

    monkeypatch.setattr(shape_ops, "fisher_working_weights", vast_information)

    with pytest.raises(
        RuntimeError,
        match="Unsafe shape repair rejected.*conditional intercept optimization did not converge",
    ):
        model.apply_shape_postfit(X)

    _assert_nothing_published(model, X, snapshot)


def test_repair_without_retained_response_is_refused_and_nothing_is_published():
    model, X = _fitted_for_publication_refusal()
    snapshot = _publication_snapshot(model, X)
    model._fit_y_ref = None

    with pytest.raises(RuntimeError, match="requires retained fit features and response"):
        model.apply_shape_postfit(X)

    _assert_nothing_published(model, X, snapshot)


def _fault_unrepaired_coefficients(model):
    """A repair that hands back the fitted coefficients, which break the declared convexity."""

    def fault(result, spec, groups):
        result.repaired_beta_reparam = model.result.beta.copy()

    return fault


def test_repair_that_leaves_the_curvature_infeasible_is_refused_and_nothing_is_published(
    monkeypatch,
):
    model, X = _fitted_for_publication_refusal()
    snapshot = _publication_snapshot(model, X)
    _repairer_with_fault(monkeypatch, _fault_unrepaired_coefficients(model))

    with pytest.raises(RuntimeError, match="Unsafe shape repair rejected.*infeasible curvature"):
        model.apply_shape_postfit(X)

    _assert_nothing_published(model, X, snapshot)


def _corrupt_validation_input(monkeypatch, **corrupt):
    """Hand ``_validate_repair_for_publication`` an input the real producer would not."""
    real = shape_ops._validate_repair_for_publication

    def validate(*args, **kwargs):
        kwargs.update({name: change(kwargs[name]) for name, change in corrupt.items()})
        return real(*args, **kwargs)

    monkeypatch.setattr(shape_ops, "_validate_repair_for_publication", validate)


@pytest.mark.parametrize(
    ("corrupt", "message"),
    [
        ({"y": lambda y: y[:-1]}, "incoherent intercept geometry"),
        ({"scoring_weights": np.zeros_like}, "invalid scoring weights"),
        (
            {"scoring_weights": lambda w: np.where(np.arange(w.size) == 0, np.nan, w)},
            "invalid scoring weights",
        ),
        (
            {"current_eta": lambda eta: np.where(np.arange(eta.size) == 0, np.inf, eta)},
            "invalid profiled intercept",
        ),
    ],
    ids=[
        "response_of_the_wrong_length",
        "weights_that_sum_to_zero",
        "non_finite_weight",
        "predictor_that_overflows_the_gaussian_shift",
    ],
)
def test_incoherent_scoring_inputs_are_refused_and_nothing_is_published(
    monkeypatch, corrupt, message
):
    model, X = _fitted_for_publication_refusal()
    snapshot = _publication_snapshot(model, X)
    _corrupt_validation_input(monkeypatch, **corrupt)

    with pytest.raises(RuntimeError, match=f"Unsafe shape repair rejected.*{message}"):
        model.apply_shape_postfit(X)

    _assert_nothing_published(model, X, snapshot)


def _profiled_eta_with_fault(monkeypatch, mutate):
    """Route the real intercept profile through ``mutate(eta)`` on the way out."""
    real = shape_ops._profile_repaired_intercept

    def profiled(*args, **kwargs):
        eta, shift, rounding = real(*args, **kwargs)
        return mutate(eta), shift, rounding

    monkeypatch.setattr(shape_ops, "_profile_repaired_intercept", profiled)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda eta: eta[:-1], "invalid predictor"),
        (
            lambda eta: np.where(np.arange(eta.size) == 3, np.nan, eta),
            "non-finite linear predictor",
        ),
    ],
    ids=["predictor_of_the_wrong_length", "non_finite_predictor"],
)
def test_faulty_repaired_predictor_is_refused_and_nothing_is_published(
    monkeypatch, mutate, message
):
    model, X = _fitted_for_publication_refusal()
    snapshot = _publication_snapshot(model, X)
    _profiled_eta_with_fault(monkeypatch, mutate)

    with pytest.raises(RuntimeError, match=f"Unsafe shape repair rejected.*{message}"):
        model.apply_shape_postfit(X)

    _assert_nothing_published(model, X, snapshot)


def test_deviance_that_is_not_finite_on_the_link_scale_is_refused_and_nothing_is_published(
    monkeypatch,
):
    model, X = _fitted_for_publication_refusal()
    snapshot = _publication_snapshot(model, X)
    with monkeypatch.context() as patch:
        patch.setattr(
            model._distribution,
            "deviance_unit",
            lambda y, mu: np.full(np.shape(mu), np.nan),
        )
        with pytest.raises(
            RuntimeError, match="Unsafe shape repair rejected.*invalid link-domain predictions"
        ):
            model.apply_shape_postfit(X)

    _assert_nothing_published(model, X, snapshot)


def test_non_finite_fitted_objective_is_refused_and_nothing_is_published(monkeypatch):
    model, X = _fitted_for_publication_refusal()
    snapshot = _publication_snapshot(model, X)
    monkeypatch.setattr(shape_ops, "_smooth_penalty_value", lambda beta, terms: np.nan)

    with pytest.raises(
        RuntimeError, match="Unsafe shape repair rejected.*non-finite fitted objective"
    ):
        model.apply_shape_postfit(X)

    _assert_nothing_published(model, X, snapshot)


def test_non_finite_term_predictor_is_refused_and_nothing_is_published():
    model, X = _fitted_for_publication_refusal()
    snapshot = _publication_snapshot(model, X)
    index = next(i for i, group in enumerate(model._groups) if group.feature_name == "x")
    group_type = type(model._dm.group_matrices[index])
    real_matvec = group_type.matvec
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            group_type,
            "matvec",
            lambda self, beta: np.full_like(real_matvec(self, beta), np.nan),
        )
        with pytest.raises(
            RuntimeError, match="Unsafe shape repair rejected.*non-finite term predictor"
        ):
            model.apply_shape_postfit(X)

    _assert_nothing_published(model, X, snapshot)


def test_basis_that_does_not_match_the_repaired_coefficients_is_refused(monkeypatch):
    model, X = _fitted_for_publication_refusal()
    snapshot = _publication_snapshot(model, X)
    spec = model._specs["x"]
    real_transform = spec.transform
    short_grid = 7

    def shorten_grid(result, _spec, _groups):
        result.grid = np.linspace(_spec._lo, _spec._hi, short_grid)

    def transform_missing_a_column(values, *args, **kwargs):
        basis = real_transform(values, *args, **kwargs)
        return basis[:, :-1] if np.shape(values) == (short_grid,) else basis

    _repairer_with_fault(monkeypatch, shorten_grid)
    monkeypatch.setattr(spec, "transform", transform_missing_a_column)

    with pytest.raises(
        RuntimeError, match="Unsafe shape repair rejected.*incoherent fitted-basis geometry"
    ):
        model.apply_shape_postfit(X)

    _assert_nothing_published(model, X, snapshot)


def test_intercept_profile_without_a_finite_score_is_refused(monkeypatch):
    model, X = _fitted_for_publication_refusal(family="poisson")
    snapshot = _publication_snapshot(model, X)
    monkeypatch.setattr(
        shape_ops,
        "fisher_working_weights",
        lambda **kwargs: np.full(np.shape(kwargs["mu"]), np.nan),
    )

    with pytest.raises(
        RuntimeError, match="Unsafe shape repair rejected.*invalid profiled intercept"
    ):
        model.apply_shape_postfit(X)

    _assert_nothing_published(model, X, snapshot)


def _scaled_solution(x, spec):
    return 1e3 * x


def _strictly_concave_solution(x, spec):
    # Negative second derivative at the middle of the domain: outside the convex cone.
    row = shape_derivative_matrix(spec, np.asarray([0.5 * (spec._lo + spec._hi)]), 2)[0]
    return -row / np.linalg.norm(row)


def _non_finite_solution(x, spec):
    return np.full_like(x, np.nan)


@pytest.mark.parametrize(
    ("solution", "message"),
    [
        (_scaled_solution, "Curvature projection failed its feasible-zero objective certificate"),
        (_strictly_concave_solution, "Curvature projection returned an infeasible fitted state"),
        (_non_finite_solution, "Curvature projection returned non-finite fitted state"),
    ],
)
@pytest.mark.parametrize("scale", [1.0, 1e-12])
def test_faulty_projection_is_refused_and_nothing_is_published(
    monkeypatch, solution, message, scale
):
    """Each certificate of the cone projection refuses its own fault; the model is untouched."""
    from superglm import constraints

    model, X = _fitted_for_publication_refusal(scale=scale)
    snapshot = _publication_snapshot(model, X)
    real_projection = constraints._cone_projection

    def projection_with_fault(spec, H, g, kind, label, **kwargs):
        return solution(real_projection(spec, H, g, kind, label, **kwargs), spec)

    monkeypatch.setattr(constraints, "_cone_projection", projection_with_fault)

    with pytest.raises(RuntimeError, match=message):
        model.apply_shape_postfit(X)

    _assert_nothing_published(model, X, snapshot)


def _never_certifying_qp(monkeypatch, nnls_stops):
    """Every QP start stops uncertified at the apex; ``nnls_stops``: so does NNLS's KKT check."""
    import scipy.optimize

    import superglm.solvers.constrained_qp as constrained_qp

    monkeypatch.setattr(
        constrained_qp,
        "solve_constrained_qp",
        lambda H, *_args, **_kwargs: constrained_qp.QPResult(
            beta=np.zeros(H.shape[0]), n_iter=200, converged=False
        ),
    )
    if nnls_stops:

        def nnls_at_its_limit(*_args, **_kwargs):
            raise RuntimeError("Maximum number of iterations reached.")

        monkeypatch.setattr(scipy.optimize, "nnls", nnls_at_its_limit)


@pytest.mark.parametrize("nnls_stops", [False, True])
def test_qp_that_never_certifies_a_monotone_projection_is_disclosed(monkeypatch, nnls_stops):
    """No QP solve certifies: the repair is a feasible point no solve was needed for, disclosed.

    The bracket's incumbent is the best multiple of its inner solve's start, which
    the cone holds, so the term is published monotone everywhere with a warning that
    it is not the closest such curve. That holds when NNLS stops at its iteration
    limit inside the uncertified point's KKT check, too.
    """
    model, X = _wiggly_monotone_fit("cr", 8)
    _never_certifying_qp(monkeypatch, nnls_stops)

    with pytest.warns(RuntimeWarning, match="stopped before it certified the closest increasing"):
        model.apply_shape_postfit(X)

    repair = model._shape_repairs["x"]
    beta = _term_beta(repair.repaired_beta_reparam, _x_groups(model))
    assert np.linalg.norm(beta) > 0.0
    assert _minimum_signed_derivative(model._specs["x"], beta, "increasing") >= -1e-12 * float(
        np.linalg.norm(beta)
    )


def test_projection_that_runs_out_of_steps_publishes_the_inner_cone_repair(monkeypatch):
    """Forced out of steps, the repair is still monotone everywhere, and says so."""
    from superglm import constraints

    model, X = _wiggly_monotone_fit("cr", 8)
    monkeypatch.setattr(constraints, "_PROJECTION_STEPS", 0)
    monkeypatch.setattr(constraints, "_BRACKET_STEPS", 1)

    with pytest.warns(RuntimeWarning, match="increasing everywhere"):
        model.apply_shape_postfit(X)

    repair = model._shape_repairs["x"]
    spec = model._specs["x"]
    beta = np.concatenate([repair.repaired_beta_reparam[group.sl] for group in _x_groups(model)])
    assert _minimum_signed_derivative(spec, beta, "increasing") >= -1e-12 * np.linalg.norm(beta)


# -- the repair's tolerances follow the response's units ------------------------------------

RESPONSE_UNITS = [1e-9, 1.0, 1e6]


def _candidate_equal_to_the_fallback(result, _spec, groups):
    """A feasible repair that drops the term, so its objective IS the fallback's."""
    flat = result.repaired_beta_reparam.copy()
    for group in groups:
        flat[group.sl] = 0.0
    result.repaired_beta_reparam = flat


@pytest.mark.parametrize("scale", RESPONSE_UNITS)
@pytest.mark.parametrize("excess", [1e-6, 1e-10])
def test_candidate_worse_than_the_fallback_by_a_relative_excess_is_refused_in_any_units(
    monkeypatch, scale, excess
):
    model, X = _fitted_for_publication_refusal(scale=scale)
    snapshot = _publication_snapshot(model, X)
    _repairer_with_fault(monkeypatch, _candidate_equal_to_the_fallback)
    real_objective = shape_ops._shape_candidate_objective
    scored = []

    def candidate_scored_first_and_worse(*args, **kwargs):
        deviance, merit = real_objective(*args, **kwargs)
        scored.append(None)
        return (
            (deviance, merit)
            if len(scored) > 1
            else (deviance * (1 + excess), merit * (1 + excess))
        )

    monkeypatch.setattr(shape_ops, "_shape_candidate_objective", candidate_scored_first_and_worse)

    with pytest.raises(RuntimeError, match="Unsafe shape repair rejected.*worse than the feasible"):
        model.apply_shape_postfit(X)

    _assert_nothing_published(model, X, snapshot)


def test_candidate_within_the_rounding_of_its_sum_is_not_worse_than_the_fallback(monkeypatch):
    model, X = _fitted_for_publication_refusal(scale=1e-9)
    _repairer_with_fault(monkeypatch, _candidate_equal_to_the_fallback)
    real_objective = shape_ops._shape_candidate_objective
    scored = []

    def candidate_a_few_ulps_up(*args, **kwargs):
        deviance, merit = real_objective(*args, **kwargs)
        scored.append(None)
        return (
            (deviance, merit) if len(scored) > 1 else (deviance * (1 + 1e-15), merit * (1 + 1e-15))
        )

    monkeypatch.setattr(shape_ops, "_shape_candidate_objective", candidate_a_few_ulps_up)

    model.apply_shape_postfit(X)

    assert "x" in model._shape_repairs


@pytest.mark.parametrize("scale", RESPONSE_UNITS)
def test_repair_that_moves_the_mean_by_a_relative_amount_is_refused_in_any_units(
    monkeypatch, scale
):
    model, X = _fitted_for_publication_refusal(scale=scale)
    snapshot = _publication_snapshot(model, X)
    real_delta = shape_ops._shape_term_eta_delta
    calls = []

    def candidate_mean_moved(*args, **kwargs):
        delta = real_delta(*args, **kwargs)
        calls.append(None)
        return delta + 1e-6 * np.max(np.abs(delta)) if len(calls) == 1 else delta

    monkeypatch.setattr(shape_ops, "_shape_term_eta_delta", candidate_mean_moved)

    with pytest.raises(RuntimeError, match="Unsafe shape repair rejected.*centering changed"):
        model.apply_shape_postfit(X)

    _assert_nothing_published(model, X, snapshot)


@pytest.mark.parametrize("scale", [1e-12, 1.0])
def test_repair_that_leaves_a_relative_violation_is_refused_in_any_units(monkeypatch, scale):
    model, X = _fitted_for_publication_refusal(scale=scale)
    snapshot = _publication_snapshot(model, X)
    _repairer_with_fault(monkeypatch, _fault_unrepaired_coefficients(model))

    with pytest.raises(RuntimeError, match="Unsafe shape repair rejected.*infeasible curvature"):
        model.apply_shape_postfit(X)

    _assert_nothing_published(model, X, snapshot)


@pytest.mark.parametrize("scale", [1e-14, 1.0])
def test_a_relative_violation_is_not_roundoff_when_the_coefficients_are_tiny(scale):
    model, _ = _fitted_for_publication_refusal(scale=1.0)
    groups = [group for group in model._groups if group.feature_name == "x"]
    term_beta = np.concatenate([model.result.beta[group.sl] for group in groups])

    assert not shape_constraint_is_roundoff_feasible(model._specs["x"], scale * term_beta, "convex")


@pytest.mark.parametrize("kind", ["increasing", "convex"])
@pytest.mark.parametrize("scale", [1e-9, 1e6])
def test_post_fit_projection_is_equivariant_to_the_response_units(monkeypatch, kind, scale):
    from superglm import constraints

    real_projection = constraints._cone_projection
    problems = []

    def spy(spec, H, g, shape_kind, label, inside=None, **options):
        problems.append((H.copy(), g.copy(), inside.copy()))
        return real_projection(spec, H, g, shape_kind, label, inside=inside, **options)

    monkeypatch.setattr(constraints, "_cone_projection", spy)

    def repaired_predictions(unit):
        problems.clear()
        rng = np.random.default_rng(20261030)
        x = np.sort(rng.uniform(0.0, 1.0, 400))
        shape = (
            -((x - 0.4) ** 2) if kind == "convex" else np.where(x < 0.5, x, 1.2 - 1.4 * x)
        ) + 0.05 * rng.normal(size=x.size)
        model = SuperGLM(
            family="gaussian",
            selection_penalty=0.0,
            features={"x": PSpline(n_knots=9, constraint=getattr(Constraint.postfit, kind))},
        ).fit(pd.DataFrame({"x": x}), unit * shape)
        model.apply_shape_postfit(pd.DataFrame({"x": x}))
        return model.predict(pd.DataFrame({"x": x})) / unit, problems[0]

    reference, (reference_H, reference_g, reference_ray) = repaired_predictions(1.0)
    rescaled, (rescaled_H, rescaled_g, rescaled_ray) = repaired_predictions(scale)

    # The same scale-free problem solved twice agrees to the projection's own
    # tolerance, with headroom for a diverging path.
    np.testing.assert_allclose(rescaled, reference, rtol=0.0, atol=1e-9 * np.max(np.abs(reference)))
    # The cone projection is posed in units of the target's weighted RMS, so the
    # problem it solves, and with it every tolerance inside it, is the same in any units.
    np.testing.assert_allclose(rescaled_H, reference_H, rtol=1e-9, atol=0.0)
    np.testing.assert_allclose(rescaled_g, reference_g, rtol=1e-6, atol=1e-12)
    # The ray its solves start along depends on the predictor alone.
    np.testing.assert_allclose(rescaled_ray, reference_ray, rtol=1e-9, atol=0.0)


#: Power of two that lifts sample weights near 1e-310 to near 1e-9.
_WEIGHT_LIFT = 1000


def _rise_fit_at_weights(weights, n_knots, degree):
    """An unpenalized Gaussian fit of a wiggly rise at ``weights``, repaired increasing."""
    x = np.linspace(0.0, 1.0, weights.size)
    X = pd.DataFrame({"x": x})
    feature = PSpline(
        n_knots=n_knots,
        degree=degree,
        m=1,
        penalty="none",
        constraint=Constraint.postfit.increasing,
    )
    model = SuperGLM(
        family="gaussian", selection_penalty=0.0, spline_penalty=0.0, features={"x": feature}
    ).fit(X, 0.9 * x + 0.25 * np.sin(12 * x), sample_weight=weights)
    return model, X


@pytest.mark.parametrize(("n_knots", "degree"), [(6, 3), (10, 4)])
def test_repair_at_subnormal_sample_weights_is_the_repair_at_lifted_weights(n_knots, degree):
    """Weights of 1e-310 make the term's metric subnormal; the repair is still published.

    It was published wrong (cubic) or raised ``LinAlgError`` (quartic). The weights
    lifted by a power of two pose the same projection in normal numbers. The repair
    is a projection, nonexpansive in its metric, so it differs between the two by no
    more than the fits themselves do, which carry the fit's own subnormal rounding.
    """

    def fitted_and_repaired(model, X):
        fitted = _term_beta(model.result.beta, _x_groups(model))
        model.apply_shape_postfit(X)
        repaired = model._shape_repairs["x"].repaired_beta_reparam
        return fitted, _term_beta(repaired, _x_groups(model))

    weights = 1e-310 * np.random.default_rng(0).uniform(0.5, 1.5, 100)
    tiny_fit, tiny_repair = fitted_and_repaired(*_rise_fit_at_weights(weights, n_knots, degree))
    lifted, X = _rise_fit_at_weights(np.ldexp(weights, _WEIGHT_LIFT), n_knots, degree)
    # The lifted fit's metric, in normal numbers.
    H, _ = _posterior_metric(lifted)
    fit, repair = fitted_and_repaired(lifted, X)

    def norm(v):
        return float(np.sqrt(v @ H @ v))

    assert norm(tiny_repair - repair) <= norm(tiny_fit - fit) + 2.0**-40 * norm(fit)


def test_repair_is_unchanged_by_a_power_of_two_scale_of_its_metric_and_weights():
    """A metric and grid weights scaled into the subnormal range give the same repair, bit for bit."""
    from superglm.constraints import MonotoneRepairer

    model, X = _rise_fit_at_weights(np.random.default_rng(0).uniform(0.5, 1.5, 100), 6, 3)
    spec, groups = model._specs["x"], _x_groups(model)
    H, _ = _posterior_metric(model)
    weights = shape_ops._grid_weights(
        spec, X["x"].to_numpy(), np.asarray(model._fit_weights, dtype=np.float64), 500
    )
    # Exponents that put the largest entries near 2**-1030, at about 43 bits; each value
    # is first rounded to that grid, so the scale back up is exact.
    down_H = 1030 + int(np.frexp(np.max(np.abs(H)))[1])
    down_w = 1030 + int(np.frexp(np.max(weights))[1])
    tiny_H, tiny_w = np.ldexp(H, -down_H), np.ldexp(weights, -down_w)
    normal = MonotoneRepairer().repair(
        spec,
        model.result.beta,
        groups,
        weights=np.ldexp(tiny_w, down_w),
        metric=np.ldexp(tiny_H, down_H),
    )
    subnormal = MonotoneRepairer().repair(
        spec, model.result.beta, groups, weights=tiny_w, metric=tiny_H
    )

    np.testing.assert_array_equal(subnormal.repaired_beta_reparam, normal.repaired_beta_reparam)


def test_a_repair_that_collapses_a_term_is_feasible_at_the_scale_of_the_term(monkeypatch):
    model, X = _fitted_for_publication_refusal()

    def collapsed(result, _spec, groups):
        # A relative 1e-14 of the fitted term, shape-wise as violating as the fit itself.
        tiny = result.repaired_beta_reparam.copy()
        for group in groups:
            tiny[group.sl] = 1e-14 * model.result.beta[group.sl]
        result.repaired_beta_reparam = tiny

    _repairer_with_fault(monkeypatch, collapsed)

    model.apply_shape_postfit(X)

    assert "x" in model._shape_repairs

    def fail_if_started(*args, **kwargs):
        raise AssertionError("a repeated repair of a collapsed term started a revision")

    monkeypatch.setattr(shape_ops.FittedStateRevision, "start", fail_if_started)
    assert model.apply_shape_postfit(X) is model


def _log_link_model_and_data(scale):
    rng = np.random.default_rng(20261031)
    n = 240
    weights = rng.uniform(0.5, 2.0, n)
    y = scale * (1.0 + 0.1 * rng.normal(size=n))
    eta = np.log(scale) - 1.0 + 0.0 * y
    return SimpleNamespace(_distribution=Gaussian(), _link=LogLink()), eta, y, weights


def test_intercept_profile_takes_the_same_steps_in_any_units(monkeypatch):
    """Newton steps and their halvings follow the shape of the deviance, not its size."""
    path = {}
    real_weights = shape_ops.fisher_working_weights

    for scale in (1e-9, 1.0):
        model, eta, y, weights = _log_link_model_and_data(scale)
        evaluated = []

        def record(evaluated=evaluated, scale=scale, **kwargs):
            evaluated.append(float(kwargs["eta"][0]) - np.log(scale))
            return real_weights(**kwargs)

        monkeypatch.setattr(shape_ops, "fisher_working_weights", record)
        profiled_eta, _, _ = shape_ops._profile_repaired_intercept(model, eta, y=y, weights=weights)
        path[scale] = evaluated
        mu = np.exp(profiled_eta)
        score_rows = weights * (y - mu) * mu
        assert abs(np.sum(score_rows)) <= 1e-9 * np.sum(np.abs(score_rows))

    assert len(path[1.0]) > 2
    np.testing.assert_allclose(path[1e-9], path[1.0], rtol=0.0, atol=1e-9)


def test_intercept_profile_passes_over_the_rows_once_per_shift(monkeypatch):
    """Each shift's deviance rounding comes from the pass that evaluates its deviance.

    Zero-weight rows are left out of it, as ``_deviance_rounding`` leaves them.
    Mutation: ``_deviance_rounding`` called per shift doubles the inverse-link
    passes.
    """
    model, eta, y, weights = _log_link_model_and_data(1.0)
    weights[::7] = 0.0
    passes, shifts = [], []
    inverse = model._link.inverse
    information = shape_ops.fisher_working_weights
    monkeypatch.setattr(model._link, "inverse", lambda e: (passes.append(1), inverse(e))[1])
    monkeypatch.setattr(
        shape_ops,
        "fisher_working_weights",
        lambda **kwargs: (shifts.append(1), information(**kwargs))[1],
    )

    profiled_eta, _, rounding = shape_ops._profile_repaired_intercept(
        model, eta, y=y, weights=weights
    )

    assert len(shifts) > 2
    assert len(passes) == len(shifts)
    expected = shape_ops._deviance_rounding(model, profiled_eta, y=y, weights=weights)
    assert rounding == pytest.approx(expected, rel=gamma(len(y)), abs=0.0)


def test_repair_publication_reuses_the_profiled_deviance_rounding(monkeypatch):
    """Publication reads each intercept profile's deviance rounding; it makes no pass of its own.

    Mutation: recomputing it at the candidate and fallback predictors.
    """
    rng = np.random.default_rng(240)
    x = np.sort(rng.uniform(0.0, 1.0, 240))
    X = pd.DataFrame({"x": x})
    model = SuperGLM(
        family="gamma",
        selection_penalty=0.0,
        features={"x": PSpline(n_knots=8, constraint=Constraint.postfit.convex)},
    ).fit(X, np.exp(1.0 + 0.5 * x))
    recomputed = []
    real = shape_ops._deviance_rounding
    monkeypatch.setattr(
        shape_ops,
        "_deviance_rounding",
        lambda *args, **kwargs: (recomputed.append(1), real(*args, **kwargs))[1],
    )

    model.apply_shape_postfit(X)

    assert "x" in model._shape_repairs
    assert recomputed == []


def test_gaussian_intercept_is_profiled_when_the_residuals_are_tiny():
    rng = np.random.default_rng(20261032)
    n = 240
    weights = rng.uniform(0.5, 2.0, n)
    y = 1e-17 * (1.0 + 0.1 * rng.normal(size=n))
    model = SimpleNamespace(_distribution=Gaussian(), _link=IdentityLink())

    profiled_eta, _, _ = shape_ops._profile_repaired_intercept(
        model, np.zeros(n), y=y, weights=weights
    )

    residual_rows = weights * (y - profiled_eta)
    assert abs(np.sum(residual_rows)) <= 64 * np.finfo(np.float64).eps * np.sum(
        np.abs(residual_rows)
    )


@pytest.mark.parametrize("scale", [1e-6, 1.0, 1e6])
def test_intercept_profile_of_a_perfect_fit_stops_at_its_rounding(scale):
    """Residuals that are only rounding cannot cancel further: that is stationary."""
    rng = np.random.default_rng(20261033)
    n = 240
    weights = rng.uniform(0.5, 2.0, n)
    y = scale * np.exp(1.0 + 0.3 * rng.normal(size=n))
    eta = np.log(y)
    model = SimpleNamespace(_distribution=Gamma(), _link=LogLink())

    _, shift, _ = shape_ops._profile_repaired_intercept(model, eta, y=y, weights=weights)

    # The predictor of a perfect fit, to within the rounding of the predictor.
    assert abs(shift) <= gamma(_ROW_STEPS) * (2.0 + np.max(np.abs(eta)))


@pytest.mark.parametrize("scale", [1e-6, 1.0, 1e6])
def test_curvature_repair_of_a_near_constant_response_is_published(scale):
    """The repaired fit is close to perfect; its intercept profile still converges."""
    x = np.linspace(0.0, 1.0, 240)
    X = pd.DataFrame({"x": x})
    y = scale * 0.1 * np.exp(-1e-6 * (x - 0.4) ** 2)
    model = SuperGLM(
        family="gamma",
        selection_penalty=0.0,
        features={"x": PSpline(n_knots=9, constraint=Constraint.postfit.convex)},
    ).fit(X, y)

    model.apply_shape_postfit(X)

    assert "x" in model._shape_repairs


@pytest.mark.parametrize("family", ["gamma", "poisson"])
@pytest.mark.parametrize("offset", [1e-3, -1e-6])
def test_intercept_profile_reaches_a_perfect_fit_from_a_shifted_start(family, offset):
    """The last steps change the deviance by less than its rounding; they are not refused."""
    rng = np.random.default_rng(20261034)
    n = 240
    weights = rng.uniform(0.5, 2.0, n)
    y = np.exp(1.0 + 0.3 * rng.normal(size=n))
    eta = np.log(y)
    distribution = Gamma() if family == "gamma" else Poisson()
    model = SimpleNamespace(_distribution=distribution, _link=LogLink())

    _, shift, _ = shape_ops._profile_repaired_intercept(model, eta + offset, y=y, weights=weights)

    assert abs(shift + offset) <= gamma(_ROW_STEPS) * (2.0 + np.max(np.abs(eta)))


@pytest.mark.parametrize("family", ["poisson", "gamma"])
def test_intercept_profile_whose_first_step_passes_the_link_clamp_converges(family, monkeypatch):
    """A Fisher step far past the log link's clamp is bracketed, then bisected to the intercept.

    The first step from the fitted shift is about exp(29.5). Halving it ran out of
    halvings long before it returned inside the clamp, and where the clamp holds every
    row the Gamma deviance is flat, so unit steps were accepted that went nowhere: both
    refused the intercept, which is closed-form for these two families under the log link.
    """
    rng = np.random.default_rng(20261112)
    n = 240
    weights = rng.uniform(0.5, 2.0, n)
    eta = 0.5 + 0.1 * rng.normal(size=n)
    y = np.exp(30.0) * rng.gamma(2.0, 0.5, size=n)
    distribution = Gamma() if family == "gamma" else Poisson()
    model = SimpleNamespace(_distribution=distribution, _link=LogLink())
    evaluations = []
    inverse = model._link.inverse
    monkeypatch.setattr(model._link, "inverse", lambda e: (evaluations.append(1), inverse(e))[1])

    _, shift, _ = shape_ops._profile_repaired_intercept(model, eta, y=y, weights=weights)

    # The score is sum w (y - mu) for the Poisson and sum w (y / mu - 1) for the Gamma.
    if family == "poisson":
        expected = np.log(np.sum(weights * y)) - np.log(np.sum(weights * np.exp(eta)))
    else:
        expected = np.log(np.sum(weights * y * np.exp(-eta))) - np.log(np.sum(weights))
    assert shift == pytest.approx(expected, rel=1e-10, abs=0.0)
    # The first trial sits at the clamp's edge, not past it: bisecting back from the
    # overshoot instead takes over fifty evaluations.
    assert len(evaluations) <= 24


def test_intercept_profile_takes_no_trial_the_familys_clamp_moves(monkeypatch):
    """A trial whose mean ``clip_mu`` moves is outside the search's domain, and is not taken.

    The binomial clamp holds means within 1e-7 of 0 and 1, inside the logit's clamp of
    the predictor at 20, so the first step's trial at the link's edge has every mean
    clamped: its deviance is flat and its score is not the deviance's slope.
    """
    from superglm.distributions import Binomial, clip_mu
    from superglm.links import LogitLink

    rng = np.random.default_rng(20261113)
    n = 240
    weights = rng.uniform(0.5, 2.0, n)
    y = rng.uniform(0.2, 0.8, n)
    eta = np.full(n, -15.0)
    model = SimpleNamespace(_distribution=Binomial(), _link=LogitLink())
    taken = []
    information = shape_ops.fisher_working_weights

    def record(**kwargs):
        mean = model._link.inverse(kwargs["eta"])
        taken.append(bool(np.any(clip_mu(mean, model._distribution) != mean)))
        return information(**kwargs)

    monkeypatch.setattr(shape_ops, "fisher_working_weights", record)

    _, shift, _ = shape_ops._profile_repaired_intercept(model, eta, y=y, weights=weights)

    mean = np.sum(weights * y) / np.sum(weights)
    assert shift == pytest.approx(np.log(mean / (1.0 - mean)) + 15.0, rel=1e-10, abs=0.0)
    assert taken and not any(taken)


def test_curvature_repair_of_a_noise_free_log_linear_response_is_published():
    rng = np.random.default_rng(240)
    x = np.sort(rng.uniform(0.0, 1.0, 240))
    X = pd.DataFrame({"x": x})
    model = SuperGLM(
        family="gamma",
        selection_penalty=0.0,
        features={"x": PSpline(n_knots=8, constraint=Constraint.postfit.convex)},
    ).fit(X, np.exp(1.0 + 0.5 * x))

    model.apply_shape_postfit(X)

    assert "x" in model._shape_repairs


def test_a_repair_made_under_an_open_editor_session_marks_its_results_stale():
    # The repair changes the model in place, so the model object alone cannot tell a result
    # made before it from one made after.
    from superglm.editor import edit
    from superglm.editor.cv import InForce

    rng = np.random.default_rng(0)
    x = np.linspace(0.0, 1.0, 200)
    y = -((x - 0.35) ** 2) + 0.05 * rng.normal(size=len(x))
    df = pd.DataFrame({"x": x})
    model = SuperGLM(
        family="gaussian",
        features={"x": PSpline(n_knots=10, constraint=Constraint.postfit.convex)},
    ).fit(df, y)
    session = edit(model, train_data=(df, y))
    in_force = InForce.of(session)
    assert in_force.describes(session)
    model.apply_shape_postfit(df)
    assert not in_force.describes(session)
    assert InForce.of(session).describes(session)
    # the session's curve follows the repair
    repaired = model.term_inference(
        "x", with_se=False, n_points=session.n_points, centering=session.centering
    )
    np.testing.assert_allclose(
        session.terms["x"].original_log_effect, repaired.log_relativity, atol=1e-12
    )
    if hasattr(InForce, "token"):  # #623's page token changes wherever ``describes`` does
        assert in_force.token != InForce.of(session).token


def test_a_response_far_below_one_is_repaired_rather_than_refused():
    # The metric's quadratic forms and the target's scale are taken in units of a power of
    # two near their largest entries, so at 1e-165, where ``fitted @ H @ fitted`` and
    # ``target @ target`` underflow, the repair completes with a curve that obeys its rule.
    # (At that scale the deviances the publication rule compares are themselves below
    # float64's resolution, so which candidate it keeps is not asserted.)
    x = np.linspace(0.0, 1.0, 200)
    X = pd.DataFrame({"x": x})
    y = 1e-165 * (0.9 * x + 0.25 * np.sin(12.0 * x))
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"x": PSpline(n_knots=9, constraint=Constraint.postfit.increasing)},
    ).fit(X, y)

    model.apply_shape_postfit(X)

    curve = np.asarray(model._shape_repairs["x"].repaired_log_effect)
    assert np.all(np.isfinite(curve))
    assert np.all(np.diff(curve) >= -1e-12 * max(float(np.max(np.abs(curve))), 1e-300))
