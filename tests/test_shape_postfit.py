import copy

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
from superglm.constraints import (
    MonotoneRepairResult,
    curvature_violation,
    monotonicity_violation,
    shape_constraint_certificate,
    shape_derivative_matrix,
)
from superglm.model import shape_ops
from superglm.model.fit_state import fitted_lambda2


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


def test_postfit_repair_rejects_mixed_scop_model_before_state_revision() -> None:
    """A partial repair cannot discard the untouched term's nonlinear inference."""
    rng = np.random.default_rng(4)
    n = 320
    x = np.sort(rng.uniform(size=n))
    z = rng.uniform(size=n)
    X = pd.DataFrame({"x": x, "z": z})
    y = 0.3 + 1.4 * x - 0.8 * (z - 0.45) ** 2 + rng.normal(0.0, 0.08, size=n)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        spline_penalty=1.2,
        features={
            "x": PSpline(n_knots=7, constraint=Constraint.fit.increasing),
            "z": PSpline(n_knots=7, constraint=Constraint.postfit.convex),
        },
    ).fit(X, y)

    original_dict = model.__dict__
    original_result = model.result
    original_solver_result = model._solver_result
    original_revision = model._fit_revision
    original_edf = model.result.effective_df
    original_predictions = model.predict(X)
    original_covariance = model._coef_covariance[0].copy()
    assert model._solver_result.scop_inference is not None

    with pytest.raises(RuntimeError, match="joint SCOP inference"):
        model.apply_shape_postfit(X)

    assert model.__dict__ is original_dict
    assert model.result is original_result
    assert model._solver_result is original_solver_result
    assert model._fit_revision == original_revision
    assert model.result.effective_df == original_edf
    assert model._solver_result.scop_inference is not None
    np.testing.assert_array_equal(model.predict(X), original_predictions)
    np.testing.assert_array_equal(model._coef_covariance[0], original_covariance)


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


def test_mixed_scop_with_feasible_postfit_term_is_an_exact_noop():
    rng = np.random.default_rng(0)
    n = 500
    x = rng.uniform(size=n)
    z = rng.uniform(size=n)
    X = pd.DataFrame({"x": x, "z": z})
    y = 0.2 + 1.2 * x + 2.0 * z
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        spline_penalty=2.0,
        features={
            "x": PSpline(n_knots=8, constraint=Constraint.fit.increasing),
            "z": PSpline(n_knots=8, constraint=Constraint.postfit.increasing),
        },
    ).fit(X, y)
    groups = [group for group in model._groups if group.feature_name == "z"]
    beta = np.concatenate([model.result.beta[group.sl] for group in groups])
    assert (
        shape_constraint_certificate(model._specs["z"], beta, "increasing").minimum_scaled_slack
        > 0.0
    )
    old_revision = model._fit_revision
    old_result = model.result

    returned = model.apply_shape_postfit(X)

    assert returned is model
    assert model._fit_revision == old_revision
    assert model.result is old_result


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


def test_monotone_span_refinement_batches_violating_extrema(monkeypatch):
    import scipy.optimize

    rng = np.random.default_rng(13)
    x = np.linspace(0.0, 1.0, 80)
    X = pd.DataFrame({"x": x})
    y = x + 0.3 * rng.normal(size=x.size)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"x": PSpline(n_knots=12, constraint=Constraint.postfit.increasing)},
    ).fit(X, y)
    real_minimize = scipy.optimize.minimize
    solve_calls = 0

    def counted_minimize(*args, **kwargs):
        nonlocal solve_calls
        solve_calls += 1
        return real_minimize(*args, **kwargs)

    monkeypatch.setattr(scipy.optimize, "minimize", counted_minimize)

    model.apply_shape_postfit(X, n_grid=500)

    assert solve_calls <= 16


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
        def repair(self, spec, beta, groups, *, weights, n_grid):
            del groups, weights
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


def test_failed_curvature_optimizer_rolls_back_fitted_state(monkeypatch):
    from types import SimpleNamespace

    import scipy.optimize

    rng = np.random.default_rng(3)
    x = np.linspace(0.0, 1.0, 120)
    X = pd.DataFrame({"x": x})
    y = -((x - 0.4) ** 2) + 0.05 * rng.normal(size=x.size)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"x": PSpline(n_knots=10, constraint=Constraint.postfit.convex)},
    ).fit(X, y)
    original_dict = model.__dict__
    original_result = model.result
    original_beta = model.result.beta.copy()
    original_predictions = model.predict(X)

    monkeypatch.setattr(
        scipy.optimize,
        "minimize",
        lambda *_args, **_kwargs: SimpleNamespace(
            success=False,
            status=9,
            message="injected iteration limit",
        ),
    )

    with pytest.raises(RuntimeError, match="Curvature projection failed to converge"):
        model.apply_shape_postfit(X)

    assert model.__dict__ is original_dict
    assert model.result is original_result
    assert not hasattr(model, "_shape_repairs")
    np.testing.assert_array_equal(model.result.beta, original_beta)
    np.testing.assert_array_equal(model.predict(X), original_predictions)


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
        def repair(self, spec, beta, groups, *, weights, n_grid):
            del groups, weights
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
        def repair(self, _spec, beta, groups, *, weights, n_grid):
            del groups, weights
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


@pytest.mark.parametrize("structured_kind", ["re", "fs", "sz"])
def test_structured_postfit_shape_repair_uses_compact_penalties(
    structured_kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from superglm.reml import penalty_algebra

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
        runtime_validation="skip",
    )
    x_groups = [group for group in model._groups if group.feature_name == "x"]
    beta_before = np.concatenate([model.result.beta[group.sl] for group in x_groups])
    revision_before = model._fit_revision

    def reject_dense_component(*_args, **_kwargs):
        raise AssertionError("post-fit repair expanded a structured penalty component")

    monkeypatch.setattr(
        penalty_algebra,
        "penalty_component_dense_matrix",
        reject_dense_component,
    )

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


@pytest.mark.parametrize("spline_penalty", [0.0, 0.5])
def test_wide_shape_repair_does_not_request_a_global_dense_penalty(
    monkeypatch,
    spline_penalty,
):
    """Publication merit must scale with penalty blocks, not total model width squared."""
    from superglm.reml import penalty_algebra

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

    def reject_global_penalty(*args, **kwargs):
        raise AssertionError("shape repair requested a global dense p-by-p penalty")

    monkeypatch.setattr(penalty_algebra, "build_penalty_matrix", reject_global_penalty)

    model.apply_shape_postfit(X, n_grid=80)

    assert np.all(np.isfinite(model.result.beta))


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
