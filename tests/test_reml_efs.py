"""Tests for the fit_reml() selection-penalty contract.

Historically this file covered the old EFS REML path that mixed sparse
selection_penalty handling into fit_reml(). The current contract is simpler:
fit_reml() is the smoothness-selection path and requires selection_penalty=0.
"""

import numpy as np
import pandas as pd
import pytest

from superglm import Numeric, SuperGLM
from superglm.features.spline import Spline
from superglm.model import base, fit_state
from superglm.model.reml_setup import collect_reml_groups, initialize_component_lambdas
from superglm.reml.efs import optimize_efs_reml
from superglm.reml.penalty_algebra import build_penalty_context


def _poisson_data(seed: int = 42, n: int = 800) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0, 10, n)
    x2 = rng.uniform(0, 10, n)
    mu = np.exp(0.5 + 0.3 * np.sin(x1) + 0.2 * np.cos(x2))
    y = rng.poisson(mu).astype(float)
    return pd.DataFrame({"x1": x1, "x2": x2}), y


def _gamma_data(seed: int = 123, n: int = 600) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0, 10, n)
    x2 = rng.uniform(0, 10, n)
    mu = np.exp(0.3 + 0.35 * np.sin(x1) + 0.15 * np.cos(x2))
    y = rng.gamma(shape=5.0, scale=mu / 5.0)
    y = np.maximum(y, 1e-4)
    return pd.DataFrame({"x1": x1, "x2": x2}), y


class TestREMLSelectionPenaltyContract:
    @pytest.mark.parametrize("selection_penalty", ["auto", 1e-8, 0.01])
    def test_rejects_selection_before_design_work(self, monkeypatch, selection_penalty):
        X = pd.DataFrame({"x": np.linspace(-1.0, 1.0, 24)})
        y = np.resize(np.array([1.0, 2.0, 3.0]), len(X))
        model = SuperGLM(
            family="poisson",
            selection_penalty=selection_penalty,
            features={"x": Numeric()},
        )
        build_called = False

        def unexpected_build(*args, **kwargs):
            nonlocal build_called
            build_called = True
            raise AssertionError("design work must not start")

        monkeypatch.setattr(base, "model_build_design_matrix", unexpected_build)

        with pytest.raises(ValueError, match="does not support selection penalties"):
            model.fit_reml(X, y)

        assert build_called is False

    @pytest.mark.parametrize("selection_penalty", [1e-8, 0.01])
    def test_poisson_rejects_positive_selection_penalty(self, selection_penalty):
        X, y = _poisson_data()
        model = SuperGLM(
            family="poisson",
            selection_penalty=selection_penalty,
            features={
                "x1": Spline(n_knots=8, penalty="ssp"),
                "x2": Spline(n_knots=8, penalty="ssp"),
            },
        )

        with pytest.raises(ValueError, match="does not support selection penalties"):
            model.fit_reml(X, y, max_reml_iter=20)

    def test_gamma_rejects_positive_selection_penalty(self):
        X, y = _gamma_data()
        model = SuperGLM(
            family="gamma",
            selection_penalty=0.01,
            features={
                "x1": Spline(n_knots=6, penalty="ssp"),
                "x2": Spline(n_knots=6, penalty="ssp"),
            },
        )

        with pytest.raises(ValueError, match="does not support selection penalties"):
            model.fit_reml(X, y, max_reml_iter=12, verbose=True)

    def test_select_true_still_rejects_if_selection_penalty_positive(self):
        X, y = _poisson_data(n=500)
        model = SuperGLM(
            family="poisson",
            selection_penalty=0.01,
            features={"x1": Spline(n_knots=8, penalty="ssp", select=True)},
        )

        with pytest.raises(ValueError, match="does not support selection penalties"):
            model.fit_reml(X[["x1"]], y, max_reml_iter=20)


def test_scalar_efs_seeded_history_and_terminal_fit_remain_exact() -> None:
    rng = np.random.default_rng(731)
    x = np.linspace(-1.0, 1.0, 80)
    response = rng.poisson(np.exp(0.25 + 0.4 * np.sin(np.pi * x))).astype(float)
    frame = pd.DataFrame({"x": x})
    weights = np.ones(len(frame))
    model = SuperGLM(
        family="poisson",
        selection_penalty=0.01,
        features={"x": Spline(kind="cr", n_knots=5, penalty="ssp")},
        spline_penalty=0.3,
    )
    y, sample_weight, _ = model._build_design_matrix(frame, response, weights, None)
    reml_groups = collect_reml_groups(model._groups, model._dm.group_matrices)
    penalties, caches, ranks = build_penalty_context(
        model._dm.group_matrices,
        reml_groups,
    )
    lambdas, estimated_names = initialize_component_lambdas(penalties, 0.3)

    result, _ = optimize_efs_reml(
        model._dm,
        model._distribution,
        model._link,
        model._groups,
        fit_state.configured_penalty(model),
        model._active_set,
        y,
        sample_weight,
        np.zeros(len(y)),
        reml_groups,
        ranks,
        lambdas,
        # The model is fitted with unit weights, so the two contracts agree
        # here; stating it keeps the pin on the geometry and not on whichever
        # reading the driver would otherwise have assumed.
        weight_semantics="frequency",
        max_reml_iter=5,
        reml_tol=1e-8,
        verbose=False,
        penalty_caches=caches,
        rebuild_dm=lambda values, weight: base.rebuild_dm_with_lambdas(
            model,
            values,
            weight,
        ),
        reml_penalties=penalties,
        estimated_names=estimated_names,
        pirls_tol=1e-8,
        max_pirls_iter=100,
    )

    # The shape of the seeded history -- its length, the single estimated
    # component, and the monotone climb from the small initial lambda -- is the
    # portable part of this contract and is asserted exactly.
    #
    # The values themselves are not bit-reproducible. Every step is the result
    # of a backward-stable dense factorisation, so it carries an absolute error
    # of O(eps * ||A||) and its last digits move with the BLAS microkernel.
    # Sweeping OPENBLAS_CORETYPE over PRESCOTT/NEHALEM/SANDYBRIDGE/HASWELL and
    # the machine default (thread counts 1..16 made no difference at this size)
    # spreads every pinned quantity below by at most 3.7e-14 relative, while the
    # smallest step the EFS iteration actually takes is 3.2e-2 relative. The
    # tolerance below sits ~2.7e4 above the measured noise floor and ~3e7 below
    # the smallest genuine behavioural step, so it still fails on any real
    # regression of the update rule, penalty rank or PIRLS weighting.
    history_tolerance = 1e-9
    history = result.lambda_history
    assert len(history) == 6
    assert all(set(step) == {"x"} for step in history)
    observed_history = [step["x"] for step in history]
    assert observed_history == sorted(observed_history)
    assert len(set(observed_history)) == 6
    np.testing.assert_allclose(
        observed_history,
        [
            0.01626419289939607,
            0.03495102648399008,
            0.07976654460262309,
            0.0895611385114593,
            0.10670080347116746,
            0.11008871046488522,
        ],
        rtol=history_tolerance,
        atol=0.0,
    )
    assert set(result.lambdas) == {"x"}
    assert result.lambdas["x"] == observed_history[-1]
    beta = result.pirls_result.beta
    assert beta.shape == (6,)
    np.testing.assert_allclose(
        beta,
        np.array(
            [
                -0.1279063044551653,
                0.36342378532888886,
                0.9588680928056398,
                1.030315559481865,
                0.10048361540442426,
                0.06796521432687205,
            ]
        ),
        rtol=history_tolerance,
        atol=0.0,
    )
    np.testing.assert_allclose(
        result.pirls_result.intercept,
        0.17010863862540995,
        rtol=history_tolerance,
        atol=0.0,
    )
    np.testing.assert_allclose(
        result.objective,
        46.435840822818534,
        rtol=history_tolerance,
        atol=0.0,
    )
    assert result.n_reml_iter == 5
    assert result.converged is False
