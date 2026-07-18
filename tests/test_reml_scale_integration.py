"""Integration regressions for family-correct REML scale profiling."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from superglm.distributions import Gamma, Gaussian
from superglm.links import IdentityLink, LogLink
from superglm.model import reml_finalize
from superglm.reml.gradient import reml_direct_gradient, reml_direct_hessian
from superglm.reml.objective import REMLObjectiveEvaluation, reml_laml_objective
from superglm.reml.scale import (
    prepare_gamma_reml_scale_data,
    profile_gaussian_reml_scale,
)
from superglm.solvers.pirls import PIRLSResult
from superglm.types import GroupSlice, PenaltyComponent


class _CachedDesign:
    def __init__(self, group_matrices=None, p: int = 1):
        self.group_matrices = [] if group_matrices is None else group_matrices
        self.p = p

    def matvec(self, beta):
        del beta
        raise AssertionError("cached objective must not expand the design")


def _result(
    *,
    beta: np.ndarray | None = None,
    deviance: float = 4.1,
    hessian_rank: int = 2,
) -> PIRLSResult:
    if beta is None:
        beta = np.array([0.4])
    return PIRLSResult(
        beta=beta,
        intercept=0.2,
        n_iter=1,
        deviance=deviance,
        converged=True,
        phi=1.0,
        effective_df=0.0,
        log_det_H=float(np.log(7.0)),
        reml_hessian_rank=hessian_rank,
    )


def _cached_gaussian_objective(y: np.ndarray, weights: np.ndarray) -> REMLObjectiveEvaluation:
    evaluation = reml_laml_objective(
        _CachedDesign(),
        Gaussian(),
        IdentityLink(),
        [],
        y,
        _result(),
        {},
        weights,
        np.zeros_like(y),
        XtWX=np.array([[5.0]]),
        log_det_H=float(np.log(7.0)),
        S_override=np.array([[2.0]]),
        return_evaluation=True,
    )
    assert isinstance(evaluation, REMLObjectiveEvaluation)
    return evaluation


def test_gaussian_objective_uses_full_wood_scale_criterion_and_frequency_size() -> None:
    y = np.array([0.3, 0.9, 1.7])
    weights = np.array([2.0, 1.0, 3.0])

    evaluation = _cached_gaussian_objective(y, weights)

    penalty_quad = 0.4 * 2.0 * 0.4
    expected_scale = profile_gaussian_reml_scale(
        penalized_deviance=4.1 + penalty_quad,
        likelihood_size=float(np.sum(weights)),
        penalty_nullity=1.0,
    )
    expected = expected_scale.criterion + 0.5 * (np.log(7.0) - np.log(2.0))
    assert evaluation.value == pytest.approx(expected, rel=1e-13)
    assert evaluation.profiled_scale == expected_scale
    assert evaluation.penalty_nullity == 1.0


def test_gaussian_objective_frequency_weights_match_expanded_rows() -> None:
    y = np.array([0.3, 0.9, 1.7])
    weights = np.array([2, 1, 3], dtype=np.float64)
    repeated_y = np.repeat(y, weights.astype(int))

    weighted = _cached_gaussian_objective(y, weights)
    expanded = _cached_gaussian_objective(repeated_y, np.ones_like(repeated_y))

    assert weighted.value == pytest.approx(expanded.value, rel=1e-14)
    assert weighted.profiled_scale == expanded.profiled_scale


def _single_penalty_inputs():
    omega = np.array([[1.5]])
    group_matrix = SimpleNamespace(R_inv=np.eye(1), omega=omega)
    penalty = PenaltyComponent(
        name="smooth",
        group_name="smooth",
        group_index=0,
        group_sl=slice(0, 1),
        omega_raw=omega,
        omega_ssp=omega,
        rank=1.0,
    )
    return group_matrix, penalty


def test_gradient_and_hessian_use_profile_inverse_scale_and_its_fd_curvature() -> None:
    group_matrix, penalty = _single_penalty_inputs()
    result = _result(beta=np.array([0.7]), deviance=3.2)
    lambdas = {"smooth": 1.8}
    inverse = np.array([[0.31]])
    profile = profile_gaussian_reml_scale(4.5, 9.0, 1.0)

    gradient = reml_direct_gradient(
        [group_matrix],
        result,
        inverse,
        lambdas,
        phi_hat=123.0,
        inverse_phi=profile.inverse_phi,
        reml_penalties=[penalty],
    )
    q = lambdas["smooth"] * float(result.beta @ penalty.omega_ssp @ result.beta)
    trace = float(np.trace(inverse @ penalty.omega_ssp))
    expected_gradient = 0.5 * (profile.inverse_phi * q + lambdas["smooth"] * trace - 1.0)
    assert gradient[0] == pytest.approx(expected_gradient)

    baseline = reml_direct_hessian(
        [group_matrix],
        SimpleNamespace(scale_known=True),
        inverse,
        lambdas,
        gradient=gradient,
        pirls_result=result,
        inverse_phi=profile.inverse_phi,
        reml_penalties=[penalty],
    )
    corrected = reml_direct_hessian(
        [group_matrix],
        Gaussian(),
        inverse,
        lambdas,
        gradient=gradient,
        pirls_result=result,
        inverse_phi=profile.inverse_phi,
        d_inverse_phi_d_penalized_deviance=(profile.d_inverse_phi_d_penalized_deviance),
        reml_penalties=[penalty],
    )

    step = 1.0e-5
    inverse_lo = profile_gaussian_reml_scale(4.5 - step * q, 9.0, 1.0).inverse_phi
    inverse_hi = profile_gaussian_reml_scale(4.5 + step * q, 9.0, 1.0).inverse_phi
    fd_profile_curvature = 0.5 * q * (inverse_hi - inverse_lo) / (2.0 * step)
    assert (corrected - baseline)[0, 0] == pytest.approx(
        fd_profile_curvature,
        rel=2.0e-10,
        abs=2.0e-12,
    )


def test_finalize_profiles_phi_from_post_qp_state(monkeypatch: pytest.MonkeyPatch) -> None:
    initial = _result(beta=np.array([0.2]), deviance=8.0)
    post_qp = _result(beta=np.array([0.8]), deviance=2.5)
    best = SimpleNamespace(
        pirls_result=initial,
        lambdas={"smooth": 1.0},
        n_reml_iter=2,
        converged=True,
    )
    dm = SimpleNamespace(p=1, group_matrices=[], matvec=lambda beta: np.zeros(3))
    model = SimpleNamespace(
        _distribution=Gaussian(),
        _link=IdentityLink(),
        _dm=dm,
        _groups=[],
        _discrete=False,
        _direct_solve="auto",
    )
    captured = {}

    monkeypatch.setattr(
        reml_finalize,
        "build_penalty_context",
        lambda *args, **kwargs: ([], {}, {}),
    )
    monkeypatch.setattr(
        reml_finalize,
        "maybe_qp_passthrough_refit",
        lambda *args, **kwargs: post_qp,
    )

    def fake_profiled_phi(model, **kwargs):
        del model
        captured.update(kwargs)
        return 0.37

    monkeypatch.setattr(reml_finalize, "compute_profiled_phi", fake_profiled_phi)
    monkeypatch.setattr(reml_finalize, "update_reml_r_inv", lambda *args, **kwargs: None)
    monkeypatch.setattr(reml_finalize, "restore_qp_group_state", lambda *args, **kwargs: None)

    reml_finalize.finalize_reml_fit(
        model,
        best=best,
        use_direct=False,
        reml_groups=[],
        reml_penalties=[],
        y=np.array([0.5, 1.0, 1.5]),
        sample_weight=np.ones(3),
        offset=None,
        offset_arr=np.zeros(3),
        max_pirls_iter=2,
        pirls_tol=1.0e-6,
        qp_passthrough=True,
        qp_saved_state=[],
        profile={},
        total_start=0.0,
        compute_fit_stats=lambda *args: {},
    )

    assert captured["pirls_result"] is post_qp
    np.testing.assert_array_equal(captured["sample_weight"], np.ones(3))
    assert model._result.phi == pytest.approx(0.37)


def test_direct_gamma_prepares_rows_once_and_reuses_reduced_stats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import superglm.reml.direct as direct

    group_matrix, penalty = _single_penalty_inputs()
    dm = _CachedDesign([group_matrix])
    y = np.array([0.4, 0.8, 1.2, 2.0, 3.0])
    weights = np.ones_like(y)
    prepare_calls = 0
    forwarded_ids: list[int] = []
    real_prepare = prepare_gamma_reml_scale_data
    real_objective = reml_laml_objective

    def counted_prepare(actual_y, actual_weights):
        nonlocal prepare_calls
        prepare_calls += 1
        return real_prepare(actual_y, actual_weights)

    fit_calls = 0

    def fake_fit(*args, lambda2, **kwargs):
        nonlocal fit_calls
        del args, kwargs
        fit_calls += 1
        lam = float(lambda2["smooth"])
        result = _result(beta=np.array([0.3 + 0.01 * lam]), deviance=2.4)
        return result, np.eye(1), np.eye(1)

    def captured_objective(*args, **kwargs):
        forwarded_ids.append(id(kwargs["gamma_scale_data"]))
        return real_objective(*args, **kwargs)

    monkeypatch.setattr(direct, "prepare_gamma_reml_scale_data", counted_prepare)
    monkeypatch.setattr(direct, "fit_irls_direct", fake_fit)
    monkeypatch.setattr(direct, "reml_laml_objective", captured_objective)
    monkeypatch.setattr(
        direct,
        "build_penalty_matrix",
        lambda *args, **kwargs: np.array(
            [[float(args[2]["smooth"] if len(args) > 2 else kwargs["lambdas"]["smooth"])]]
        ),
    )
    monkeypatch.setattr(direct, "reml_direct_gradient", lambda *args, **kwargs: np.ones(1))
    monkeypatch.setattr(direct, "reml_direct_hessian", lambda *args, **kwargs: np.ones((1, 1)))
    monkeypatch.setattr(direct, "reml_w_correction", lambda *args, **kwargs: None)

    direct.optimize_direct_reml(
        dm,
        Gamma(),
        LogLink(),
        [GroupSlice(name="smooth", start=0, end=1)],
        False,
        y,
        weights,
        np.zeros_like(y),
        [(0, GroupSlice(name="smooth", start=0, end=1))],
        {"smooth": 1.0},
        {"smooth": 1.0},
        max_reml_iter=1,
        reml_tol=1.0e-8,
        verbose=False,
        reml_penalties=[penalty],
    )

    assert fit_calls >= 3
    assert prepare_calls == 1
    assert len(forwarded_ids) >= 2
    assert len(set(forwarded_ids)) == 1
