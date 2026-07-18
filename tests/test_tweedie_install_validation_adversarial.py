"""Adversarial regressions for transactional Tweedie profile installation."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import superglm.profiling.tweedie as tweedie_module
from superglm import SuperGLM
from superglm.distributions import Tweedie
from superglm.features.numeric import Numeric
from superglm.model import fit_ops, profile_ops
from superglm.penalties.group_lasso import GroupLasso
from superglm.profiling.tweedie import (
    TweedieProfileResult,
    generate_tweedie_cpg,
)


def _small_tweedie_data(n: int = 40) -> tuple[pd.DataFrame, np.ndarray]:
    """Return a stable mixed zero/positive sample for final-refit checks."""
    rng = np.random.default_rng(20260718)
    x = np.linspace(-1.0, 1.0, n)
    mu = np.exp(0.7 + 0.25 * x)
    y = generate_tweedie_cpg(n, mu=mu, phi=0.8, p=1.5, rng=rng)
    return pd.DataFrame({"x": x}), y


def _new_model(*, retain_fit_state: bool = True, penalty=None) -> SuperGLM:
    kwargs = {"selection_penalty": 0.0} if penalty is None else {"penalty": penalty}
    return SuperGLM(
        family=Tweedie(p=1.5),
        features={"x": Numeric()},
        retain_fit_state=retain_fit_state,
        **kwargs,
    )


def _certified_result(
    *,
    p_hat: float = 1.5,
    phi_hat: float = 0.8,
    nll: float = 0.0,
    trace: pd.DataFrame | None = None,
) -> TweedieProfileResult:
    """Build the smallest token-stamped record accepted at the profile boundary."""
    if trace is None:
        trace = pd.DataFrame(
            {
                "p": [p_hat],
                "phi": [phi_hat],
                "nll": [nll],
                "edf": [1.0],
                "fit_converged": [True],
                "solver_converged": [True],
                "reml_converged": [None],
                "phi_converged": [True],
                "objective_finite": [True],
                "n_saddlepoint": [0],
                "density_method": ["exact"],
                "density_exact": [True],
            }
        )
    n_evaluations = len(trace)
    return TweedieProfileResult(
        p_hat=p_hat,
        phi_hat=phi_hat,
        nll=nll,
        n_evaluations=n_evaluations,
        converged=True,
        method="brent",
        phi_method="mle",
        search_trace=trace,
        density_method="exact",
        density_exact=True,
        _objective=lambda p, source="": float(nll),
        _ll_scale=1.0,
        _evaluation_count=lambda: n_evaluations,
        _evaluation_record=lambda p: None,
        _validation_token=tweedie_module._TWEEDIE_PROFILE_RESULT_TOKEN,
    )


def _snapshot_identity_state(model: SuperGLM) -> dict[str, object]:
    """Snapshot exact caller-owned state identities before a transaction."""
    return model.__dict__.copy()


def _assert_identity_state_unchanged(
    model: SuperGLM,
    snapshot: dict[str, object],
) -> None:
    assert model.__dict__.keys() == snapshot.keys()
    for name, value in snapshot.items():
        assert model.__dict__[name] is value, name


def _install_fake_profile(monkeypatch, result: TweedieProfileResult) -> None:
    monkeypatch.setattr(
        tweedie_module,
        "_estimate_tweedie_p_prepared",
        lambda *args, **kwargs: result,
    )


def test_token_stamped_nonminimum_trace_winner_is_rejected_before_refit(
    monkeypatch,
) -> None:
    X, y = _small_tweedie_data()
    model = _new_model()
    before = _snapshot_identity_state(model)
    trace = pd.DataFrame(
        {
            "p": [1.45, 1.55],
            "phi": [0.8, 0.8],
            "nll": [2.0, 1.0],
            "edf": [1.0, 1.0],
            "fit_converged": [True, True],
            "solver_converged": [True, True],
            "reml_converged": [None, None],
            "phi_converged": [True, True],
            "objective_finite": [True, True],
            "n_saddlepoint": [0, 0],
            "density_method": ["exact", "exact"],
            "density_exact": [True, True],
        }
    )
    result = _certified_result(p_hat=1.45, nll=2.0, trace=trace)
    _install_fake_profile(monkeypatch, result)
    fit_calls = []
    monkeypatch.setattr(
        SuperGLM,
        "fit",
        lambda *args, **kwargs: fit_calls.append((args, kwargs)),
    )

    with pytest.raises(RuntimeError, match="non-minimum winning trace record"):
        model.estimate_p(X, y)

    assert fit_calls == []
    _assert_identity_state_unchanged(model, before)


def test_finite_final_log_likelihood_mismatch_is_not_installed(monkeypatch) -> None:
    X, y = _small_tweedie_data(n=32)
    model = _new_model()
    before = _snapshot_identity_state(model)
    real_synchronize = profile_ops._synchronize_tweedie_profile_refit

    def corrupt_log_likelihood(candidate, sync_y, result):
        real_synchronize(candidate, sync_y, result)
        candidate._fit_stats = replace(
            candidate._fit_stats,
            log_likelihood=candidate._fit_stats.log_likelihood + 5.0,
        )

    monkeypatch.setattr(
        profile_ops,
        "_synchronize_tweedie_profile_refit",
        corrupt_log_likelihood,
    )

    with pytest.raises(RuntimeError, match="profile/final objective agreement"):
        model.estimate_p(X, y, method="grid", grid=np.array([1.5]))

    _assert_identity_state_unchanged(model, before)


def test_private_solver_beta_that_disagrees_with_fitted_means_is_not_installed(
    monkeypatch,
) -> None:
    X, y = _small_tweedie_data()
    model = _new_model()
    before = _snapshot_identity_state(model)
    result = _certified_result()
    _install_fake_profile(monkeypatch, result)
    real_synchronize = profile_ops._synchronize_tweedie_profile_refit

    def corrupt_solver_beta(candidate, sync_y, profile_result):
        real_synchronize(candidate, sync_y, profile_result)
        solver_result = candidate._solver_pirls_result()
        candidate._solver_result = replace(
            solver_result,
            beta=np.asarray(solver_result.beta) + 0.5,
        )

    monkeypatch.setattr(
        profile_ops,
        "_synchronize_tweedie_profile_refit",
        corrupt_solver_beta,
    )

    with pytest.raises(RuntimeError, match="solver fitted means"):
        model.estimate_p(X, y)

    _assert_identity_state_unchanged(model, before)


@pytest.mark.parametrize("storage_kind", ["masked", "complex", "bool"])
@pytest.mark.parametrize("corruption_target", ["public_result", "predictions"])
def test_nonplain_public_result_or_prediction_storage_is_not_installed(
    monkeypatch,
    corruption_target,
    storage_kind,
) -> None:
    X, y = _small_tweedie_data()
    model = _new_model()
    before = _snapshot_identity_state(model)
    result = _certified_result()
    _install_fake_profile(monkeypatch, result)
    real_synchronize = profile_ops._synchronize_tweedie_profile_refit

    def invalid_vector(values):
        values = np.asarray(values)
        if storage_kind == "masked":
            return np.ma.array(values, mask=np.zeros(values.shape, dtype=bool))
        if storage_kind == "complex":
            return values.astype(np.complex128)
        return np.ones(values.shape, dtype=bool)

    def corrupt_storage(candidate, sync_y, profile_result):
        real_synchronize(candidate, sync_y, profile_result)
        if corruption_target == "public_result":
            if storage_kind == "masked":
                candidate._result = replace(
                    candidate.result,
                    beta=invalid_vector(candidate.result.beta),
                )
            else:
                invalid_phi = 0.8 + 0.0j if storage_kind == "complex" else True
                candidate._result = replace(candidate.result, phi=invalid_phi)
        else:
            invalid_prediction = invalid_vector(np.ones(len(X), dtype=np.float64))
            candidate.predict = lambda *args, **kwargs: invalid_prediction

    monkeypatch.setattr(
        profile_ops,
        "_synchronize_tweedie_profile_refit",
        corrupt_storage,
    )

    with pytest.raises(RuntimeError, match="final fit is not installable"):
        model.estimate_p(X, y)

    _assert_identity_state_unchanged(model, before)


def test_negative_public_deviance_is_not_installed(monkeypatch) -> None:
    X, y = _small_tweedie_data()
    model = _new_model()
    before = _snapshot_identity_state(model)
    result = _certified_result()
    _install_fake_profile(monkeypatch, result)
    real_synchronize = profile_ops._synchronize_tweedie_profile_refit

    def corrupt_deviance(candidate, sync_y, profile_result):
        real_synchronize(candidate, sync_y, profile_result)
        candidate._result = replace(candidate.result, deviance=-1.0)

    monkeypatch.setattr(
        profile_ops,
        "_synchronize_tweedie_profile_refit",
        corrupt_deviance,
    )

    with pytest.raises(RuntimeError, match="public fitted result"):
        model.estimate_p(X, y)

    _assert_identity_state_unchanged(model, before)


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("missing_inference", "released inference state"),
        ("bad_covariance", "released covariance state"),
        ("bad_group_edf", "released group EDF state"),
    ],
)
def test_malformed_released_inference_cache_is_not_installed(
    monkeypatch,
    corruption,
    message,
) -> None:
    X, y = _small_tweedie_data()
    model = _new_model(retain_fit_state=False)
    before = _snapshot_identity_state(model)
    result = _certified_result()
    _install_fake_profile(monkeypatch, result)
    real_release = fit_ops._maybe_release_fit_state

    def corrupt_released_cache(candidate):
        released = real_release(candidate)
        if candidate._retain_fit_state:
            return released
        if corruption == "missing_inference":
            candidate.__dict__["_fit_inference_info"] = None
        elif corruption == "bad_covariance":
            covariance, groups = candidate.__dict__["_coef_covariance"]
            candidate.__dict__["_coef_covariance"] = (
                covariance + np.eye(len(covariance)),
                groups,
            )
        else:
            group_edf = dict(candidate.__dict__["_group_edf"])
            first_group = next(iter(group_edf))
            group_edf[first_group] += 1.0
            candidate.__dict__["_group_edf"] = group_edf
        return released

    monkeypatch.setattr(fit_ops, "_maybe_release_fit_state", corrupt_released_cache)

    with pytest.raises(RuntimeError, match=message):
        model.estimate_p(X, y)

    _assert_identity_state_unchanged(model, before)


def test_progress_callback_predict_injection_is_not_installed(monkeypatch) -> None:
    X, y = _small_tweedie_data()
    model = _new_model()
    result = _certified_result()
    _install_fake_profile(monkeypatch, result)
    events = []

    def injected_predict(*args, **kwargs):
        return np.full(len(X), -123.0)

    def inject_predict(event, payload):
        events.append(event)
        if event == "best_found":
            model.predict = injected_predict

    returned = model.estimate_p(X, y, progress_callback=inject_predict)

    assert returned is result
    assert events == ["best_found", "final_refit"]
    assert "predict" not in model.__dict__
    assert model.predict.__func__ is SuperGLM.predict
    assert np.all(model.predict(X) > 0.0)
    assert model._tweedie_profile_result is not result
    assert model._tweedie_profile_result.p_hat == result.p_hat
    assert model._tweedie_profile_result.phi_hat == result.phi_hat


def test_mutating_aliasing_penalty_deepcopy_is_rejected_before_profiling(
    monkeypatch,
) -> None:
    class MutatingAliasingGroupLasso(GroupLasso):
        def __init__(self):
            super().__init__(lambda1=0.0)
            self.deepcopy_calls = 0

        def __deepcopy__(self, memo):
            self.deepcopy_calls += 1
            self.lambda1 = 999.0
            return self

    X, y = _small_tweedie_data()
    penalty = MutatingAliasingGroupLasso()
    model = _new_model(penalty=penalty)
    # Isolate the profile entrypoint boundary from the constructor's generic
    # defensive-copy protocol.  The malicious hook aliases the configured
    # object, so resetting it here lets this regression prove that profiling
    # rejects the hook before starting another copy or touching caller state.
    penalty.deepcopy_calls = 0
    penalty.lambda1 = 0.0
    before = _snapshot_identity_state(model)
    profile_calls = []
    monkeypatch.setattr(
        tweedie_module,
        "_estimate_tweedie_p_prepared",
        lambda *args, **kwargs: profile_calls.append((args, kwargs)),
    )

    with pytest.raises(TypeError, match="custom copy hooks.*__deepcopy__"):
        model.estimate_p(X, y)

    assert profile_calls == []
    assert penalty.deepcopy_calls == 0
    assert penalty.lambda1 == 0.0
    _assert_identity_state_unchanged(model, before)
