"""Runtime canonicalization regressions for spline-backed public terms."""

import numpy as np
import pandas as pd
import pytest

from superglm import SuperGLM
from superglm.distributions import clip_mu
from superglm.features.spline import PSpline, Spline
from superglm.links import stabilize_eta


def _feature_beta(model: SuperGLM, feature_name: str) -> np.ndarray:
    groups = model._feature_groups(feature_name)
    return np.concatenate([model.result.beta[g.sl] for g in groups])


def _runtime_canonicalization_diagnostics(model: SuperGLM) -> dict:
    state = getattr(model, "_runtime_canonical_state", None)
    assert state is not None
    diagnostics = state.get("diagnostics")
    assert diagnostics is not None
    return diagnostics


def _recompute_live_public_runtime_state(model: SuperGLM) -> dict[str, object]:
    X = model._fit_X_ref
    assert X is not None

    eta_after = np.full(len(X), model.result.intercept, dtype=np.float64)
    term_means_after: dict[str, float] = {}

    for feature_name in model._feature_order:
        spec = model._specs[feature_name]
        groups = model._feature_groups(feature_name)
        beta = np.concatenate([model.result.beta[g.sl] for g in groups])
        values = X[feature_name].to_numpy(dtype=np.float64)
        contribution = np.asarray(spec.score(values, beta), dtype=np.float64).ravel()
        eta_after += contribution
        term_means_after[feature_name] = float(np.mean(contribution))

    for interaction_name in model._interaction_order:
        spec = model._interaction_specs[interaction_name]
        groups = model._feature_groups(interaction_name)
        beta = np.concatenate([model.result.beta[g.sl] for g in groups])
        left_name, right_name = spec.parent_names
        left = X[left_name].to_numpy(dtype=np.float64)
        right = X[right_name].to_numpy(dtype=np.float64)
        if hasattr(spec, "score"):
            contribution = np.asarray(spec.score(left, right, beta), dtype=np.float64).ravel()
        else:
            contribution = np.asarray(spec.transform(left, right) @ beta, dtype=np.float64).ravel()
        eta_after += contribution
        term_means_after[interaction_name] = float(np.mean(contribution))

    solver = model._solver_pirls_result()
    eta_before = model._dm.matvec(solver.beta) + solver.intercept
    if model._fit_offset is not None:
        eta_before = eta_before + model._fit_offset
        eta_after = eta_after + model._fit_offset
    eta_before = stabilize_eta(eta_before, model._link)
    eta_after = stabilize_eta(eta_after, model._link)
    mu_before = clip_mu(model._link.inverse(eta_before), model._distribution)
    mu_after = clip_mu(model._link.inverse(eta_after), model._distribution)
    return {
        "term_means_after": term_means_after,
        "max_abs_eta_delta": float(np.max(np.abs(eta_after - eta_before))),
        "max_abs_mu_delta": float(np.max(np.abs(mu_after - mu_before))),
    }


def _assert_zero_mean_and_before_after_parity(
    model: SuperGLM,
    feature_name: str,
    x: np.ndarray,
) -> None:
    spec = model._specs[feature_name]
    beta = _feature_beta(model, feature_name)
    contribution = spec.score(x, beta)
    assert abs(np.mean(contribution)) < 1e-10

    diagnostics = _runtime_canonicalization_diagnostics(model)
    recomputed = _recompute_live_public_runtime_state(model)
    assert diagnostics["max_abs_eta_delta"] < 1e-10
    assert diagnostics["max_abs_mu_delta"] < 1e-10
    assert abs(diagnostics["max_abs_eta_delta"] - recomputed["max_abs_eta_delta"]) < 1e-12
    assert abs(diagnostics["max_abs_mu_delta"] - recomputed["max_abs_mu_delta"]) < 1e-12


def _manual_public_curve_se(
    model: SuperGLM,
    feature_name: str,
    x_eval: np.ndarray,
) -> np.ndarray:
    spec = model._specs[feature_name]
    groups = model._feature_groups(feature_name)
    Cov_active, active_groups = model._coef_covariance

    active_subs = [ag for ag in active_groups if ag.feature_name == feature_name]
    assert active_subs

    indices = np.concatenate([np.arange(ag.start, ag.end) for ag in active_subs])
    Cov_g = Cov_active[np.ix_(indices, indices)]
    M = spec.transform(x_eval)
    active_cols = np.concatenate(
        [
            np.arange(g.start, g.end) - groups[0].start
            for g in groups
            if any(ag.feature_name == feature_name and ag.name == g.name for ag in active_subs)
        ]
    )
    M = M[:, active_cols]
    Q = M @ Cov_g
    return np.sqrt(np.maximum(np.sum(Q * M, axis=1), 0.0))


def _make_monotone_poisson_data(
    n: int = 800,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0.0, 1.0, n))
    sample_weight = rng.uniform(0.3, 1.2, n)
    eta = -1.2 + 1.1 * x
    y = rng.poisson(sample_weight * np.exp(eta)).astype(float)
    X = pd.DataFrame({"x": x})
    return X, y, sample_weight


def _make_tensor_poisson_data(
    n: int = 500,
    seed: int = 123,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    z = rng.uniform(0.0, 1.0, n)
    sample_weight = rng.uniform(0.4, 1.4, n)
    eta = -0.7 + 0.8 * x - 0.5 * z + 0.9 * x * z
    y = rng.poisson(sample_weight * np.exp(eta)).astype(float)
    X = pd.DataFrame({"x": x, "z": z})
    return X, y, sample_weight


class TestRuntimeCanonicalization:
    def test_standalone_spline_public_term_is_zero_mean_and_prediction_parity_is_locked(self):
        X, y, sample_weight = _make_monotone_poisson_data(seed=0)
        model = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            features={"x": Spline(n_knots=10, penalty="ssp")},
        )
        model.fit_reml(X, y, sample_weight=sample_weight, max_reml_iter=10)

        assert model._runtime_canonical_state["solver_to_public"] is not None
        assert model._runtime_canonical_state["solver_to_public_complete"] is True
        _assert_zero_mean_and_before_after_parity(
            model,
            "x",
            X["x"].to_numpy(dtype=np.float64),
        )

    def test_monotone_fit_spline_public_term_is_zero_mean_and_prediction_parity_is_locked(self):
        X, y, sample_weight = _make_monotone_poisson_data(seed=1)
        model = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            features={
                "x": PSpline(
                    n_knots=10,
                    penalty="ssp",
                    monotone="increasing",
                    monotone_mode="fit",
                )
            },
        )
        model.fit(X, y, sample_weight=sample_weight)

        _assert_zero_mean_and_before_after_parity(
            model,
            "x",
            X["x"].to_numpy(dtype=np.float64),
        )

    def test_discrete_standalone_spline_fast_predict_respects_public_runtime_state(self):
        X, y, sample_weight = _make_monotone_poisson_data(seed=0)
        model = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            discrete=True,
            features={"x": Spline(n_knots=10, penalty="ssp")},
        )
        model.fit_reml(X, y, sample_weight=sample_weight, max_reml_iter=6)

        eta_exact = model._predict_eta_exact(X)
        eta_fast = model._predict_eta_fast_discrete(X)
        mu_exact = model.predict(X)
        mu_fast = model._predict_fast_discrete(X)

        assert np.max(np.abs(eta_exact - eta_fast)) < 3e-3
        assert np.max(np.abs(mu_exact - mu_fast)) < 3e-3

    @pytest.mark.parametrize("kind", ["bs", "ns", "cr"])
    def test_discrete_standalone_spline_public_term_is_zero_mean_in_live_runtime_space(
        self,
        kind: str,
    ):
        X, y, sample_weight = _make_monotone_poisson_data(seed=0)
        model = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            discrete=True,
            features={"x": Spline(kind=kind, n_knots=10, penalty="ssp")},
        )
        model.fit_reml(X, y, sample_weight=sample_weight, max_reml_iter=10)

        beta = _feature_beta(model, "x")
        contribution = np.asarray(
            model._specs["x"].score(X["x"].to_numpy(dtype=np.float64), beta),
            dtype=np.float64,
        )
        recomputed = _recompute_live_public_runtime_state(model)
        diagnostics = _runtime_canonicalization_diagnostics(model)

        assert abs(np.mean(contribution)) < 1e-10
        assert abs(recomputed["term_means_after"]["x"]) < 1e-10
        assert diagnostics["term_means_after"]["x"] == pytest.approx(
            recomputed["term_means_after"]["x"],
            abs=1e-12,
        )

    def test_qp_monotone_fit_spline_uses_affine_runtime_state(self):
        X, y, sample_weight = _make_monotone_poisson_data(seed=2)
        model = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            features={
                "x": Spline(
                    kind="cr",
                    n_knots=10,
                    penalty="ssp",
                    monotone="increasing",
                    monotone_mode="fit",
                )
            },
        )
        model.fit(X, y, sample_weight=sample_weight)

        state = model._runtime_canonical_state["terms"]["x"]
        assert state["mode"] == "affine"
        assert state["applied_to_public_model"] is True
        _assert_zero_mean_and_before_after_parity(
            model,
            "x",
            X["x"].to_numpy(dtype=np.float64),
        )

    def test_monotone_fit_summary_curve_se_uses_public_runtime_basis(self):
        X, y, sample_weight = _make_monotone_poisson_data(seed=3)
        model = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            features={
                "x": PSpline(
                    n_knots=10,
                    penalty="ssp",
                    monotone="increasing",
                    monotone_mode="fit",
                )
            },
        )
        model.fit(X, y, sample_weight=sample_weight)

        summary = model.summary()
        spline_row = next(row for row in summary._coef_rows if row.name == "x")
        x_grid = np.linspace(model._specs["x"]._lo, model._specs["x"]._hi, 200)
        expected = _manual_public_curve_se(model, "x", x_grid)

        assert spline_row.curve_se_min == pytest.approx(float(np.min(expected)), abs=1e-12)
        assert spline_row.curve_se_max == pytest.approx(float(np.max(expected)), abs=1e-12)

    def test_decomposed_tensor_interaction_term_is_compiled_blockwise_and_deferred(self):
        X, y, sample_weight = _make_tensor_poisson_data()
        model = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            features={
                "x": Spline(kind="cr", n_knots=4, penalty="ssp"),
                "z": Spline(kind="cr", n_knots=4, penalty="ssp"),
            },
        )
        model._add_interaction("x", "z", decompose=True)
        model.fit(X, y, sample_weight=sample_weight)

        interaction_name = model._interaction_order[0]
        state = model._runtime_canonical_state["terms"][interaction_name]
        assert state["mode"] == "blockwise"
        assert state["applied_to_public_model"] is False
        assert len(state["group_indices"]) == 2
        assert abs(state["term_mean_before"]) > 1e-6
        assert state["term_mean_after"] is None
        assert model._runtime_canonical_state["solver_to_public"] is None
        assert model._runtime_canonical_state["solver_to_public_complete"] is False

        diagnostics = _runtime_canonicalization_diagnostics(model)
        recomputed = _recompute_live_public_runtime_state(model)
        assert (
            diagnostics["term_means_after"][interaction_name]
            == recomputed["term_means_after"][interaction_name]
        )
        assert abs(diagnostics["term_means_after"][interaction_name]) > 1e-6
        assert abs(diagnostics["max_abs_eta_delta"] - recomputed["max_abs_eta_delta"]) < 1e-12
        assert abs(diagnostics["max_abs_mu_delta"] - recomputed["max_abs_mu_delta"]) < 1e-12
