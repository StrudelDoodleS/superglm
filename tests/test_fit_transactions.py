"""Strong exception and atomic-install tests for ordinary fits."""

from __future__ import annotations

import copy
import pickle
from collections.abc import Mapping

import numpy as np
import pandas as pd
import pytest

from superglm import NegativeBinomial, Numeric, Spline, SuperGLM
from superglm.model import fit_ops, runtime_canonicalize
from superglm.model.fit_state import (
    FitCandidate,
    FitState,
    FittedStateRevision,
    _install_fit_state,
    fitted_lambda2,
)
from superglm.model.fit_workspace import FitWorkspace
from superglm.profiling.nb import NBProfileResult
from superglm.solvers.pirls import PIRLSResult, StagnationRecord

from ._fit_state_oracles import (
    InjectedFitFailure,
    assert_model_behavior_unchanged,
    snapshot_model_behavior,
)


@pytest.fixture
def two_fit_datasets():
    rng = np.random.default_rng(20260718)
    x = np.linspace(-1.0, 1.0, 160)
    z = rng.normal(size=len(x))
    weights = rng.uniform(0.4, 1.4, size=len(x))
    X_a = pd.DataFrame({"x": x, "z": z})
    X_b = pd.DataFrame({"x": x[::-1].copy(), "z": z * 1.15 + 0.2})
    y_a = rng.poisson(np.exp(0.1 + 0.3 * x - 0.15 * z)).astype(np.float64)
    y_b = rng.poisson(np.exp(-0.2 - 0.25 * x + 0.2 * z)).astype(np.float64)
    return (X_a, y_a, weights), (X_b, y_b, weights[::-1].copy())


def _model(*, penalized: bool = False, retain_fit_state: bool = True) -> SuperGLM:
    return SuperGLM(
        family="poisson",
        selection_penalty=0.02 if penalized else 0.0,
        retain_fit_state=retain_fit_state,
        features={"x": Numeric(), "z": Numeric()},
    )


def _inject_failure(monkeypatch, target: str) -> None:
    def fail(*args, **kwargs):
        raise InjectedFitFailure(target)

    if target == "canonicalize_fitted_model":
        monkeypatch.setattr(runtime_canonicalize, target, fail)
    else:
        monkeypatch.setattr(fit_ops, target, fail)


def _inject_path_failure(monkeypatch, target: str) -> None:
    def fail(*args, **kwargs):
        raise InjectedFitFailure(target)

    if target == "run_lambda_path":
        monkeypatch.setattr(fit_ops.path_ops, target, fail)
    elif target == "canonicalize_fitted_model":
        monkeypatch.setattr(runtime_canonicalize, target, fail)
    else:
        monkeypatch.setattr(fit_ops, target, fail)


def _inject_reml_failure(monkeypatch, target: str) -> None:
    def fail(*args, **kwargs):
        raise InjectedFitFailure(target)

    monkeypatch.setattr(fit_ops, target, fail)


def _reml_model(*, retain_fit_state: bool = True) -> SuperGLM:
    return SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        spline_penalty=0.3,
        retain_fit_state=retain_fit_state,
        features={"x": Spline(n_knots=6, penalty="ssp"), "z": Numeric()},
    )


@pytest.mark.parametrize(
    "failure_target",
    [
        "fit_irls_direct",
        "fit_pirls",
        "_compute_fit_stats",
        "canonicalize_fitted_model",
        "_maybe_release_fit_state",
        "capture_fit_state",
    ],
)
def test_failed_first_fit_remains_unfitted(two_fit_datasets, failure_target, monkeypatch):
    (X, y, weights), _ = two_fit_datasets
    model = _model(penalized=failure_target == "fit_pirls")
    original_dict = model.__dict__
    original_specs = model._specs
    original_penalty_config = model._penalty_config
    _inject_failure(monkeypatch, failure_target)

    with pytest.raises(InjectedFitFailure, match=failure_target):
        model.fit(X, y, sample_weight=weights)

    assert model.__dict__ is original_dict
    assert model._fit_state is None
    assert model._fit_revision == 0
    assert model._specs is original_specs
    assert model._penalty_config is original_penalty_config
    assert model._dm is None
    assert model._result is None
    with pytest.raises(RuntimeError, match="fitted|fitted|Not fitted"):
        _ = model.result


@pytest.mark.parametrize(
    "failure_target",
    [
        "fit_irls_direct",
        "fit_pirls",
        "_compute_fit_stats",
        "canonicalize_fitted_model",
        "_maybe_release_fit_state",
        "capture_fit_state",
    ],
)
def test_failed_refit_preserves_previous_revision(
    two_fit_datasets,
    failure_target,
    monkeypatch,
):
    first, second = two_fit_datasets
    X_a, y_a, weights_a = first
    X_b, y_b, weights_b = second
    model = _model(penalized=failure_target == "fit_pirls").fit(
        X_a,
        y_a,
        sample_weight=weights_a,
    )
    before = snapshot_model_behavior(model, X_a)
    _inject_failure(monkeypatch, failure_target)

    with pytest.raises(InjectedFitFailure, match=failure_target):
        model.fit(X_b, y_b, sample_weight=weights_b)

    assert_model_behavior_unchanged(model, X_a, before)


@pytest.mark.parametrize(
    "failure_phase",
    ["profile_design", "profile_solver", "final_solver", "public_copy"],
)
def test_failed_theta_profile_preserves_previous_revision(failure_phase, monkeypatch):
    """The entire public profile/refit operation has one transaction boundary."""
    X_old = pd.DataFrame({"c": pd.Categorical(["A", "B"] * 40)})
    y_old = np.asarray([1.0, 3.0] * 40)
    X_new = pd.DataFrame({"c": pd.Categorical(["B", "C"] * 40)})
    y_new = np.asarray([2.0, 4.0] * 40)
    model = SuperGLM(
        family=NegativeBinomial(theta=2.0),
        selection_penalty=0.0,
        splines=[],
    ).fit(X_old, y_old)
    model.family = NegativeBinomial(theta="auto")

    before = snapshot_model_behavior(model, X_old)
    config = model._config
    family_config = model._family_config
    penalty_config = model._penalty_config
    config_revision = model._config_revision

    def fail(*args, **kwargs):
        raise InjectedFitFailure(failure_phase)

    if failure_phase == "profile_design":
        monkeypatch.setattr(SuperGLM, "_build_design_matrix", fail)
    elif failure_phase == "profile_solver":
        monkeypatch.setattr("superglm.profiling.nb.fit_irls_direct", fail)
    else:
        profile_result = NBProfileResult(
            theta_hat=2.5,
            nll=1.2,
            n_evaluations=1,
            converged=True,
        )
        monkeypatch.setattr(
            "superglm.profiling.nb.estimate_nb_theta",
            lambda *args, **kwargs: profile_result,
        )
        if failure_phase == "final_solver":
            monkeypatch.setattr(fit_ops, "fit_irls_direct", fail)
        else:
            monkeypatch.setattr(NBProfileResult, "_detached_public_copy", fail)

    with pytest.raises(InjectedFitFailure, match=failure_phase):
        model.estimate_theta(X_new, y_new)

    assert_model_behavior_unchanged(model, X_old, before)
    assert model._config is config
    assert model._family_config is family_config
    assert model._penalty_config is penalty_config
    assert model._config_revision == config_revision
    assert model.family.theta == "auto"
    assert model.theta_ == pytest.approx(2.0)


def test_workspace_does_not_alias_previous_fitted_state(two_fit_datasets):
    (X, y, weights), _ = two_fit_datasets
    fitted = _model().fit(X, y, sample_weight=weights)
    previous = fitted._fit_state

    workspace = FitWorkspace.start(fitted, mode="fit", validated_inputs=None)

    assert workspace.model._specs is not fitted._specs
    assert workspace.model._groups is not fitted._groups
    assert workspace.model._groups == []
    assert workspace.model._dm is None
    assert workspace.previous_revision == fitted._fit_revision
    assert fitted._fit_state is previous


def test_workspace_start_does_not_copy_previous_row_scale_state(
    two_fit_datasets,
    monkeypatch,
):
    (X, y, weights), _ = two_fit_datasets
    fitted = _model().fit(X, y, sample_weight=weights)
    forbidden = {
        id(fitted._dm),
        id(fitted._fit_weights),
        id(fitted._fit_mu),
        id(fitted._fit_X_ref),
    }
    real_deepcopy = copy.deepcopy

    def guarded_deepcopy(value, memo=None):
        assert id(value) not in forbidden, "copied prior row-scale fit state"
        if memo is None:
            return real_deepcopy(value)
        return real_deepcopy(value, memo)

    monkeypatch.setattr("superglm.model.fit_state.copy.deepcopy", guarded_deepcopy)

    FitWorkspace.start(fitted, mode="fit", validated_inputs=None)


def test_fit_publication_releases_rebuildable_raw_spline_tabmat_plan(monkeypatch):
    import superglm._group_matrix._group_matrix_tabmat as tabmat_helpers
    import superglm.group_matrix as group_matrix_module

    monkeypatch.setattr(tabmat_helpers, "_MIN_RAW_SPLINE_TABMAT_ROWS", 100)
    n = 200
    x = np.linspace(0.0, 1.0, n)
    X = pd.DataFrame({"x": x, "z": x[::-1].copy()})
    y = np.ones(n, dtype=np.float64)
    real_builder = group_matrix_module._build_raw_spline_tabmat_plan
    built_plans = []

    def recording_builder(*args, **kwargs):
        plan = real_builder(*args, **kwargs)
        if plan is not None:
            built_plans.append(plan)
        return plan

    monkeypatch.setattr(group_matrix_module, "_build_raw_spline_tabmat_plan", recording_builder)
    model = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        features={"x": Spline(n_knots=8), "z": Spline(n_knots=8)},
    )

    model.fit(X, y, max_iter=2)

    assert len(built_plans) == 1
    assert model._dm.raw_spline_tabmat_plan_built is False
    assert model._dm._raw_spline_tabmat_holder.plan is None


def test_install_is_one_revision_dictionary_swap():
    model = _model()
    old_dict = model.__dict__
    prepared = dict(old_dict)
    state = FitState(
        revision=1,
        selection_penalty=0.0,
        distribution=None,
        projections={},
        retained=True,
    )
    prepared["_fit_state"] = state
    prepared["_fit_revision"] = 1
    candidate = FitCandidate(state=state, prepared_model_dict=prepared)

    _install_fit_state(model, candidate)

    assert model.__dict__ is prepared
    assert model.__dict__ is not old_dict
    assert model._fit_state is state
    assert model._fit_revision == 1


def test_fit_state_projections_and_result_arrays_are_immutable(two_fit_datasets):
    (X, y, weights), _ = two_fit_datasets
    model = _model().fit(X, y, sample_weight=weights)
    state = model._fit_state

    assert isinstance(state.projections, Mapping)
    with pytest.raises(TypeError):
        state.projections["_result"] = object()
    assert not state.projections["_result"].beta.flags.writeable
    assert not state.projections["_solver_result"].beta.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        model.result.beta[0] += 1.0


def test_published_result_rejects_scalar_and_beta_rebinding(two_fit_datasets):
    (X, y, weights), _ = two_fit_datasets
    model = _model().fit(X, y, sample_weight=weights)
    result = model.result
    beta = result.beta.copy()
    intercept = result.intercept

    with pytest.raises(AttributeError, match="published"):
        result.intercept = intercept + 1.0
    with pytest.raises(AttributeError, match="published"):
        result.beta = beta + 1.0
    with pytest.raises(ValueError):
        result.beta.setflags(write=True)
    assert result.rank_info is model._solver_result.rank_info
    with pytest.raises(ValueError):
        result.rank_info.mean_x.setflags(write=True)

    np.testing.assert_array_equal(result.beta, beta)
    assert result.intercept == intercept


def test_published_result_deeply_freezes_diagnostics_and_rank_metadata(two_fit_datasets):
    (X, y, weights), _ = two_fit_datasets
    model = _model().fit(
        X,
        y,
        sample_weight=weights,
        record_diagnostics=True,
    )
    result = model.result
    assert result.iteration_log

    assert isinstance(result.iteration_log, tuple)
    with pytest.raises(AttributeError):
        result.iteration_log.append(result.iteration_log[0])
    with pytest.raises(AttributeError):
        result.iteration_log[0].deviance = -1.0
    with pytest.raises(TypeError):
        result.rank_info.group_edf[model._groups[0].name] = -1.0


def test_published_stagnation_records_keep_their_fields():
    """Freezing must not strip the narrow channel's records to bare tuples.

    ``_freeze_result_arrays`` rebuilds any tuple it meets as a plain ``tuple``,
    so a record type that is one would lose its field names here and every
    later ``entry.deviance`` would raise. ``StagnationRecord`` is a dataclass
    for exactly this reason; this pins that it stays one.
    """
    result = PIRLSResult(
        beta=np.zeros(2),
        intercept=0.0,
        n_iter=1,
        deviance=1.0,
        converged=True,
        phi=1.0,
        effective_df=1.0,
        stagnation_log=[StagnationRecord(deviance=1.5, step_rejected=False, step_halvings=2)],
    )
    result._publish()

    entry = result.stagnation_log[0]
    assert isinstance(entry, StagnationRecord)
    assert entry.deviance == 1.5
    assert entry.step_rejected is False
    assert entry.step_halvings == 2
    with pytest.raises(AttributeError):
        entry.deviance = -1.0


@pytest.mark.parametrize(
    "round_trip",
    [
        pytest.param(copy.deepcopy, id="deepcopy"),
        pytest.param(lambda value: pickle.loads(pickle.dumps(value)), id="pickle"),
    ],
)
def test_result_publication_immutability_survives_round_trip(two_fit_datasets, round_trip):
    (X, y, weights), _ = two_fit_datasets
    model = _model().fit(X, y, sample_weight=weights)

    restored = round_trip(model)

    assert restored.result is restored._fit_state.projections["_result"]
    assert restored._solver_result is restored._fit_state.projections["_solver_result"]
    assert restored.result.rank_info is restored._solver_result.rank_info
    assert restored._fit_revision == restored._fit_state.revision
    assert not restored.result.beta.flags.writeable
    with pytest.raises(AttributeError, match="published"):
        restored.result.deviance = 0.0
    with pytest.raises(AttributeError, match="published"):
        restored.result.beta = restored.result.beta.copy()
    with pytest.raises(ValueError):
        restored.result.beta.setflags(write=True)
    with pytest.raises(ValueError):
        restored.result.rank_info.mean_x.setflags(write=True)


@pytest.mark.parametrize(
    "round_trip",
    [
        pytest.param(copy.deepcopy, id="deepcopy"),
        pytest.param(lambda value: pickle.loads(pickle.dumps(value)), id="pickle"),
    ],
)
def test_retained_fit_data_guard_reuses_metrics_cache_after_round_trip(
    two_fit_datasets,
    round_trip,
):
    X, y, weights = two_fit_datasets[0]
    restored = round_trip(_model().fit(X, y, sample_weight=weights))

    first = restored.metrics(
        restored._fit_X_ref,
        restored._fit_y_ref,
        sample_weight=restored._fit_sample_weight_ref,
        offset=restored._fit_offset_ref,
    )
    second = restored.metrics(
        restored._fit_X_ref,
        restored._fit_y_ref,
        sample_weight=restored._fit_sample_weight_ref,
        offset=restored._fit_offset_ref,
    )

    assert second is first


def test_refit_does_not_mutate_previous_state_result(two_fit_datasets):
    (X_a, y_a, weights_a), (X_b, y_b, weights_b) = two_fit_datasets
    model = _model().fit(X_a, y_a, sample_weight=weights_a)
    old_state = model._fit_state
    old_result = old_state.projections["_result"]
    old_beta = old_result.beta.copy()

    model.fit(X_b, y_b, sample_weight=weights_b)

    assert model._fit_state is not old_state
    assert model._fit_revision == old_state.revision + 1
    assert old_state.projections["_result"] is old_result
    np.testing.assert_array_equal(old_result.beta, old_beta)
    assert not old_result.beta.flags.writeable


def test_compact_fit_freezes_transferred_covariance(two_fit_datasets):
    (X, y, weights), _ = two_fit_datasets
    model = _model(retain_fit_state=False).fit(X, y, sample_weight=weights)

    covariance, _ = model.__dict__["_coef_covariance"]
    assert not covariance.flags.writeable
    assert not model.__dict__["_fit_inference_info"]["XtWX_inv_aug"].flags.writeable


def test_compact_fit_does_not_hash_rows_that_will_be_released(
    two_fit_datasets,
    monkeypatch,
):
    """Compact publication must not pay for a discarded retained-data guard."""
    from superglm.model.fit_data_guard import FitDataGuard

    (X, y, weights), _ = two_fit_datasets

    def fail_capture(*args, **kwargs):
        del args, kwargs
        raise AssertionError("compact fit attempted to capture a retained-data digest")

    monkeypatch.setattr(FitDataGuard, "capture", fail_capture)

    model = _model(retain_fit_state=False).fit(X, y, sample_weight=weights)

    assert model._fit_state.retained is False
    assert model._fit_data_guard is None


@pytest.mark.parametrize(
    "failure_target",
    [
        "run_lambda_path",
        "_compute_fit_stats",
        "canonicalize_fitted_model",
        "_maybe_release_fit_state",
    ],
)
def test_failed_first_path_fit_remains_unfitted(
    two_fit_datasets,
    failure_target,
    monkeypatch,
):
    (X, y, weights), _ = two_fit_datasets
    model = _model(penalized=True)
    original_dict = model.__dict__
    original_penalty_config = model._penalty_config
    _inject_path_failure(monkeypatch, failure_target)

    with pytest.raises(InjectedFitFailure, match=failure_target):
        model.fit_path(X, y, sample_weight=weights, n_lambda=3)

    assert model.__dict__ is original_dict
    assert model._fit_state is None
    assert model._fit_revision == 0
    assert model._penalty_config is original_penalty_config
    assert model._dm is None
    assert model._result is None


@pytest.mark.parametrize(
    "failure_target",
    [
        "run_lambda_path",
        "_compute_fit_stats",
        "canonicalize_fitted_model",
        "_maybe_release_fit_state",
    ],
)
def test_failed_path_refit_preserves_previous_revision(
    two_fit_datasets,
    failure_target,
    monkeypatch,
):
    (X_a, y_a, weights_a), (X_b, y_b, weights_b) = two_fit_datasets
    model = _model(penalized=True).fit(X_a, y_a, sample_weight=weights_a)
    before = snapshot_model_behavior(model, X_a)
    _inject_path_failure(monkeypatch, failure_target)

    with pytest.raises(InjectedFitFailure, match=failure_target):
        model.fit_path(X_b, y_b, sample_weight=weights_b, n_lambda=3)

    assert_model_behavior_unchanged(model, X_a, before)


def test_successful_path_fit_installs_final_path_state_without_rewriting_config(
    two_fit_datasets,
):
    (X, y, weights), _ = two_fit_datasets
    model = _model(penalized=True)
    model.lambda2 = 0.35
    configured_penalty = model._penalty_config
    configured_lambda2 = model._lambda2_config
    lambda_seq = np.array([0.2, 0.05, 0.01], dtype=np.float64)
    original_lambda_seq = lambda_seq.copy()

    path = model.fit_path(X, y, sample_weight=weights, lambda_seq=lambda_seq)

    assert model._penalty_config is configured_penalty
    assert model._lambda2_config is configured_lambda2
    assert model.penalty.lambda1 == pytest.approx(0.02)
    assert model.lambda2 == pytest.approx(0.35)
    assert model._fit_state.resolved_penalty.lambda1 == pytest.approx(lambda_seq[-1])
    assert model._fit_state.selection_penalty == pytest.approx(lambda_seq[-1])
    assert model._fit_state.resolved_lambda2 == pytest.approx(0.35)
    assert model._fit_state.revision == model._fit_revision == 1
    assert fitted_lambda2(model) == pytest.approx(0.35)
    np.testing.assert_array_equal(model.result.beta, path.coef_path[-1])
    assert model.result.intercept == path.intercept_path[-1]
    np.testing.assert_array_equal(lambda_seq, original_lambda_seq)
    assert lambda_seq.flags.writeable
    assert path.lambda_seq is not lambda_seq
    for values in (
        path.lambda_seq,
        path.coef_path,
        path.intercept_path,
        path.deviance_path,
        path.n_iter_path,
        path.converged_path,
        path.edf_path,
    ):
        assert values is not None
        assert not values.flags.writeable
        with pytest.raises(ValueError):
            values.setflags(write=True)
    with pytest.raises(AttributeError):
        path.lambda_seq = np.array([1.0])

    restored = pickle.loads(pickle.dumps(path))
    np.testing.assert_array_equal(restored.coef_path, path.coef_path)
    with pytest.raises(ValueError):
        restored.coef_path.setflags(write=True)
    with pytest.raises(AttributeError):
        restored.coef_path = np.empty((0, 0))


@pytest.mark.parametrize(
    ("path_kwargs", "message"),
    [
        ({"lambda_seq": []}, "non-empty"),
        ({"lambda_seq": [0.1, -0.01]}, "non-negative"),
        ({"lambda_seq": [0.01, 0.1]}, "non-increasing"),
        ({"lambda_seq": [0.1, np.nan]}, "finite"),
        ({"lambda_seq": [[0.1, 0.01]]}, "one-dimensional"),
        ({"n_lambda": 0}, "n_lambda"),
        ({"lambda_ratio": 0.0}, "lambda_ratio"),
        ({"lambda_ratio": 1.1}, "lambda_ratio"),
    ],
)
def test_invalid_path_controls_fail_without_publishing_state(
    two_fit_datasets,
    path_kwargs,
    message,
):
    (X, y, weights), _ = two_fit_datasets
    model = _model(penalized=True)
    original_dict = model.__dict__

    with pytest.raises(ValueError, match=message):
        model.fit_path(X, y, sample_weight=weights, **path_kwargs)

    assert model.__dict__ is original_dict
    assert model._fit_state is None
    assert model._fit_revision == 0


def test_path_accepts_unpenalized_zero_endpoint(two_fit_datasets):
    (X, y, weights), _ = two_fit_datasets
    model = _model(penalized=True)

    path = model.fit_path(
        X,
        y,
        sample_weight=weights,
        lambda_seq=np.array([0.1, 0.0]),
    )

    np.testing.assert_array_equal(path.lambda_seq, np.array([0.1, 0.0]))
    assert model._fit_state.selection_penalty == pytest.approx(0.0)


def test_fitted_model_clone_uses_installed_path_penalty(two_fit_datasets):
    (X, y, weights), _ = two_fit_datasets
    model = _model(penalized=True)
    model.fit_path(X, y, sample_weight=weights, lambda_seq=np.array([0.1, 0.005]))

    cloned = model._clone_without_features(set())

    assert cloned.selection_penalty == pytest.approx(0.005)
    assert model.selection_penalty == pytest.approx(0.02)


def test_fitted_model_clone_uses_installed_smoothing_not_staged_config(two_fit_datasets):
    (X, y, weights), _ = two_fit_datasets
    model = _model()
    model.lambda2 = 0.25
    model.fit(X, y, sample_weight=weights)
    model.lambda2 = 9.0

    cloned = model._clone_without_features(set())

    assert cloned.lambda2 == pytest.approx(0.25)
    assert model.lambda2 == pytest.approx(9.0)


def test_fitted_model_clone_uses_installed_family_and_link(two_fit_datasets):
    (X, y, weights), _ = two_fit_datasets
    model = _model().fit(X, y, sample_weight=weights)
    model.family = "gaussian"
    model.link = "identity"

    cloned = model._clone_without_features(set())

    assert type(cloned.family).__name__ == "Poisson"
    assert type(cloned.link).__name__ == "LogLink"
    assert model.family == "gaussian"
    assert model.link == "identity"


def test_fitted_model_clone_preserves_retention_policy(two_fit_datasets):
    (X, y, weights), _ = two_fit_datasets
    model = _model(retain_fit_state=False).fit(X, y, sample_weight=weights)

    cloned = model._clone_without_features(set())

    assert cloned._retain_fit_state is False


def test_installed_fit_state_is_authoritative_for_smoothing_parameters(two_fit_datasets):
    (X, y, weights), _ = two_fit_datasets
    model = _model()
    model.lambda2 = 0.25
    model.fit(X, y, sample_weight=weights)

    model._reml_lambdas = {"stale": 99.0}

    assert fitted_lambda2(model) == pytest.approx(0.25)


def test_fitted_state_revision_preserves_installed_smoothing_over_staged_config(
    two_fit_datasets,
):
    (X, y, weights), _ = two_fit_datasets
    model = _model()
    model.lambda2 = 0.25
    model.fit(X, y, sample_weight=weights)
    model.lambda2 = 9.0

    FittedStateRevision.start(model).commit()

    assert model.lambda2 == pytest.approx(9.0)
    assert fitted_lambda2(model) == pytest.approx(0.25)
    assert model._fit_state.resolved_lambda2 == pytest.approx(0.25)


def test_fitted_state_revision_rejects_public_solver_coefficient_divergence(
    two_fit_datasets,
):
    (X, y, weights), _ = two_fit_datasets
    model = _model().fit(X, y, sample_weight=weights)
    original_dict = model.__dict__
    revision = FittedStateRevision.start(model)
    revision.model._result.beta[0] += 1.0

    with pytest.raises(RuntimeError, match="public and solver coefficients"):
        revision.commit()

    assert model.__dict__ is original_dict


def test_fitted_state_revision_rejects_broken_canonical_intercept_relation(
    two_fit_datasets,
):
    (X, y, weights), _ = two_fit_datasets
    model = _model().fit(X, y, sample_weight=weights)
    original_dict = model.__dict__
    revision = FittedStateRevision.start(model)
    revision.model._result.intercept += 1.0

    with pytest.raises(RuntimeError, match="canonical intercept"):
        revision.commit()

    assert model.__dict__ is original_dict


@pytest.mark.parametrize("field", ["deviance", "phi", "effective_df"])
def test_fitted_state_revision_rejects_public_solver_scalar_divergence(
    two_fit_datasets,
    field,
):
    (X, y, weights), _ = two_fit_datasets
    model = _model().fit(X, y, sample_weight=weights)
    original_dict = model.__dict__
    revision = FittedStateRevision.start(model)
    setattr(revision.model._result, field, getattr(revision.model._result, field) + 1.0)

    with pytest.raises(RuntimeError, match="public and solver scalar"):
        revision.commit()

    assert model.__dict__ is original_dict


def test_fitted_state_revision_rejects_stale_nested_reml_result(two_fit_datasets):
    (X, y, weights), _ = two_fit_datasets
    model = _reml_model().fit_reml(
        X,
        y,
        sample_weight=weights,
        max_reml_iter=1,
    )
    original_dict = model.__dict__
    revision = FittedStateRevision.start(model)
    revision.model._reml_result.pirls_result = revision.model._result

    with pytest.raises(RuntimeError, match="REML result"):
        revision.commit()

    assert model.__dict__ is original_dict


@pytest.mark.parametrize(
    "failure_target",
    [
        "optimize_reml_best",
        "finalize_reml_fit",
        "_canonicalize_fitted_model",
        "_maybe_release_fit_state",
        "capture_fit_state",
    ],
)
def test_failed_first_reml_fit_remains_unfitted(
    two_fit_datasets,
    failure_target,
    monkeypatch,
):
    (X, y, weights), _ = two_fit_datasets
    model = _reml_model()
    original_dict = model.__dict__
    original_lambda2_config = model._lambda2_config
    _inject_reml_failure(monkeypatch, failure_target)

    with pytest.raises(InjectedFitFailure, match=failure_target):
        model.fit_reml(X, y, sample_weight=weights, max_reml_iter=1)

    assert model.__dict__ is original_dict
    assert model._fit_state is None
    assert model._fit_revision == 0
    assert model._lambda2_config is original_lambda2_config
    assert model._dm is None
    assert model._result is None


@pytest.mark.parametrize(
    "failure_target",
    [
        "optimize_reml_best",
        "finalize_reml_fit",
        "_canonicalize_fitted_model",
        "_maybe_release_fit_state",
        "capture_fit_state",
    ],
)
def test_failed_reml_refit_preserves_previous_revision(
    two_fit_datasets,
    failure_target,
    monkeypatch,
):
    (X_a, y_a, weights_a), (X_b, y_b, weights_b) = two_fit_datasets
    model = _reml_model().fit(X_a, y_a, sample_weight=weights_a)
    before = snapshot_model_behavior(model, X_a)
    _inject_reml_failure(monkeypatch, failure_target)

    with pytest.raises(InjectedFitFailure, match=failure_target):
        model.fit_reml(X_b, y_b, sample_weight=weights_b, max_reml_iter=1)

    assert_model_behavior_unchanged(model, X_a, before)


def test_successful_reml_fit_owns_optimized_lambdas_without_rewriting_config(
    two_fit_datasets,
):
    (X, y, weights), _ = two_fit_datasets
    model = _reml_model()
    configured_penalty = model._penalty_config
    configured_lambda2 = model._lambda2_config

    returned = model.fit_reml(X, y, sample_weight=weights, max_reml_iter=1)

    assert returned is model
    assert model._penalty_config is configured_penalty
    assert model._lambda2_config is configured_lambda2
    assert model.penalty.lambda1 == pytest.approx(0.0)
    assert model.lambda2 == pytest.approx(0.3)
    assert model._fit_state.resolved_penalty.lambda1 == pytest.approx(0.0)
    assert model._fit_state.resolved_lambda2 == model._reml_lambdas
    assert model._fit_state.resolved_lambda2 is not model._reml_lambdas
    assert model._fit_state.revision == model._fit_revision == 1
    assert set(model._fit_state.resolved_lambda2) == {"x"}
    assert fitted_lambda2(model) == model._reml_lambdas


def test_reml_fallback_keeps_smoothing_and_selection_fields_distinct(two_fit_datasets):
    (X, y, weights), _ = two_fit_datasets
    model = _model(penalized=False)
    model.lambda2 = 0.45

    model.fit_reml(X, y, sample_weight=weights, max_reml_iter=1)

    assert model._fit_state.resolved_penalty.lambda1 == pytest.approx(0.0)
    assert model._fit_state.resolved_lambda2 == pytest.approx(0.45)
    assert fitted_lambda2(model) == pytest.approx(0.45)
    assert not hasattr(model, "_reml_lambdas")


def test_fit_workspace_preserves_bounded_reml_runtime_hooks():
    model = _reml_model()
    model._max_analytical_per_w = 2
    model._select_snap = False

    workspace = FitWorkspace.start(model, mode="fit_reml", validated_inputs=None)

    assert workspace.model._max_analytical_per_w == 2
    assert workspace.model._select_snap is False
