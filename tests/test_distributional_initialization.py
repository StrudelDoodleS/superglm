"""Starting penalties use information and penalty units, not a fixed scalar."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from superglm import GaussianLS, Predictor, Spline, SuperLSS
from superglm.types import LambdaPolicy
from tests.bound_predictor_fixtures import model_from_templates


def _fit_start(
    *,
    units: float = 1.0,
    model_kwargs=None,
    spline_kwargs=None,
    replication=1,
    family=None,
    **kwargs,
):
    rng = np.random.default_rng(81)
    x = np.linspace(0.0, 4.0, 240)
    y = np.sin(x) + rng.normal(scale=0.5, size=len(x))
    model = model_from_templates(
        family=GaussianLS() if family is None else family,
        predictors=[
            Predictor("location", {"x": Spline("cr", k=8, **(spline_kwargs or {}))}),
            Predictor("scale", {}),
        ],
        **(model_kwargs or {}),
    ).fit_reml(
        pd.DataFrame({"x": np.repeat(units * x, replication)}),
        np.repeat(y, replication),
        max_reml_iter=1,
        **kwargs,
    )
    return model._require_fitted().smoothing


def test_default_initial_penalty_tracks_cubic_feature_units():
    """Kills a raw common lambda: integral squared curvature scales as units^-3."""
    base = _fit_start()
    scaled = _fit_start(units=10.0)
    name = "location:x#wiggle"
    # The spline construction and eigensolve are well-conditioned here.
    tolerance = 4096 * 8 * np.finfo(float).eps
    assert scaled.initial_lambdas[name] == pytest.approx(
        1000.0 * base.initial_lambdas[name], rel=tolerance
    )


def test_none_requests_an_automatic_start():
    automatic = _fit_start(initial_lambda=None)
    assert automatic.initial_lambdas["location:x#wiggle"] > 0.0


def test_explicit_start_keeps_its_units_and_value():
    explicit = _fit_start(units=10.0, initial_lambda=3.0)
    assert explicit.initial_lambdas["location:x#wiggle"] == 3.0


def test_modal_edf_start_residualizes_nullspace_and_is_congruence_invariant():
    from superglm.distributional.smoothing.initialization import _information_scaled_lambda

    # Schur complement is diag(4, 9). Each mode must start at most half-active.
    factor = np.array([[2.0, 0.0, 0.0], [1.0, 2.0, 0.0], [2.0, 0.0, 3.0]])
    fisher = factor @ factor.T
    penalty = np.diag([0.0, 1.0, 1.0])
    transform = np.array([[1.0, 0.2, 0.3], [0.0, 2.0, 0.1], [0.1, 0.0, 0.8]])
    tolerance = 128 * 3 * np.finfo(float).eps * np.linalg.cond(transform) ** 2
    value = _information_scaled_lambda(fisher, penalty, 2)
    modal_edf = np.array([4.0, 9.0]) / (np.array([4.0, 9.0]) + value)
    assert np.all(modal_edf <= 0.5 + tolerance)
    assert np.max(modal_edf) == pytest.approx(0.5, rel=tolerance)
    assert value == pytest.approx(9.0, rel=tolerance)
    assert _information_scaled_lambda(
        transform.T @ fisher @ transform, transform.T @ penalty @ transform, 2
    ) == pytest.approx(9.0, rel=tolerance)
    assert _information_scaled_lambda(7 * fisher, 5 * penalty, 2) == pytest.approx(
        63 / 5, rel=tolerance
    )


def test_modal_edf_start_excludes_unsupported_modes_and_refuses_unresolved_penalty_rank():
    from superglm.distributional.smoothing.initialization import _information_scaled_lambda

    assert _information_scaled_lambda(
        np.diag([0.0, 4.0, 0.0]), np.diag([0.0, 1.0, 1.0]), 2
    ) == pytest.approx(4.0)
    assert _information_scaled_lambda(np.zeros((3, 3)), np.eye(3), 3) is None
    assert _information_scaled_lambda(np.eye(3), np.diag([0.0, 1e-18, 1.0]), 2) is None


def test_named_start_overrides_automatic_start():
    explicit = _fit_start(lambdas={"location:x#wiggle": 2.5})
    assert explicit.initial_lambdas["location:x#wiggle"] == 2.5


def test_frequency_weight_matches_row_replication():
    base = _fit_start(model_kwargs={"weight_semantics": "frequency"})
    weighted = _fit_start(
        model_kwargs={"weight_semantics": "frequency"}, sample_weight=np.full(240, 3.0)
    )
    repeated = _fit_start(model_kwargs={"weight_semantics": "frequency"}, replication=3)
    name = "location:x#wiggle"
    tolerance = 4096 * 8 * np.finfo(float).eps
    assert weighted.initial_lambdas[name] == pytest.approx(
        3 * base.initial_lambdas[name], rel=tolerance
    )
    assert repeated.initial_lambdas[name] == pytest.approx(
        weighted.initial_lambdas[name], rel=tolerance
    )


def test_fixed_policy_precedes_named_and_automatic_starts():
    result = _fit_start(
        spline_kwargs={"lambda_policy": {"wiggle": LambdaPolicy.fixed(0.75)}},
        lambdas={"location:x#wiggle": 2.5},
    )
    assert result.initial_lambdas["location:x#wiggle"] == 0.75


def test_discrete_start_preserves_backend_and_agrees_with_dense():
    dense = _fit_start()
    discrete = _fit_start(model_kwargs={"discrete": True, "n_bins": 256})
    name = "location:x#wiggle"
    tolerance = 4096 * 8 * np.finfo(float).eps
    assert discrete.initial_lambdas[name] == pytest.approx(
        dense.initial_lambdas[name], rel=tolerance
    )
    assert discrete.coefficient_fits[0].resolved_chunk_size is not None


def test_selection_penalties_calibrate_shared_group_separately():
    result = _fit_start(spline_kwargs={"select": True})
    assert set(result.initial_lambdas) == {"location:x#wiggle", "location:x#null"}
    assert all(value > 0 for value in result.initial_lambdas.values())
    assert result.initial_lambdas["location:x#wiggle"] != result.initial_lambdas["location:x#null"]


def test_automatic_start_is_clipped_to_upper_bound():
    result = _fit_start(units=100, max_lambda=0.01)
    assert result.initial_lambdas["location:x#wiggle"] == 0.01


class _ObservedOnlyGaussian:
    """Expose the actual Gaussian likelihood without its optional Fisher method."""

    def __init__(self):
        self.base = GaussianLS()

    def to_config(self):
        return self.base.to_config()

    @property
    def parameters(self):
        return self.base.parameters

    @property
    def default_prediction_name(self):
        return self.base.default_prediction_name

    def bind_likelihood(self, y, weights, observation):
        return self.base.bind_likelihood(y, weights, observation)

    def initialize(self, y, plan):
        return self.base.initialize(y, plan)

    def evaluate_natural(self, y, theta, plan, *, derivative_order=2):
        return self.base.evaluate_natural(y, theta, plan, derivative_order=derivative_order)

    def default_prediction(self, theta):
        return self.base.default_prediction(theta)


def test_missing_expected_information_uses_bounded_numeric_fallback():
    result = _fit_start(family=_ObservedOnlyGaussian())
    assert result.initial_lambdas["location:x#wiggle"] == 0.1
    bounded = _fit_start(family=_ObservedOnlyGaussian(), max_lambda=0.01)
    assert bounded.initial_lambdas["location:x#wiggle"] == 0.01


def test_unresolved_nonzero_nuisance_direction_refuses_calibration():
    from superglm.distributional.smoothing.initialization import _information_scaled_lambda

    # The single information direction lies entirely in the nuisance span.
    # Dropping its small nuisance coordinate invents one supported EDF.
    h = 1e-10
    fisher = np.array([[h * h, h], [h, 1.0]])
    assert _information_scaled_lambda(fisher, np.diag([0.0, 1.0]), 1) is None


@pytest.mark.parametrize("chunk_size", [None, 17])
def test_start_fisher_uses_backend_predictor_arithmetic_under_cancellation(chunk_size):
    from superglm._frame import as_eager_frame
    from superglm.distributional.family import COMPLETE_OBSERVATION
    from superglm.distributional.layout import build_stacked_layout
    from superglm.distributional.predictor import compile_predictors
    from superglm.distributional.result import DenseSolverConfig, DistributionalEFSConfig
    from superglm.distributional.smoothing.initialization import (
        prepare_distributional_initialization,
    )
    from superglm.distributional.solver.solver import _DenseObservedReuseSession
    from superglm.distributional.weights import WeightContract, resolve_likelihood_weights
    from superglm.features import Numeric

    n = 80
    x = np.linspace(0, 1, n)
    y = np.cos(x)
    family = GaussianLS()
    weights = resolve_likelihood_weights(
        np.ones(n), n_observations=n, contract=WeightContract("prior")
    )
    layout = build_stacked_layout(
        compile_predictors(
            as_eager_frame(pd.DataFrame({"x": x, "z": np.ones(n)})),
            weights,
            family.parameters,
            [Predictor("location", {"x": Spline("cr", k=8)}), Predictor("scale", {"z": Numeric()})],
            offsets={"scale": np.ones(n)},
        )
    )
    plan = family.bind_likelihood(y, weights, COMPLETE_OBSERVATION)
    ordinary = np.zeros(layout.n_coefficients)
    cancellation = ordinary.copy()
    scale = layout.predictor("scale")
    cancellation[scale.coefficient_slice] = [-float(2**54), float(2**54)]

    def prepare(initial):
        return prepare_distributional_initialization(
            family,
            layout,
            y,
            plan,
            supplied=None,
            config=DistributionalEFSConfig(initial_lambda=None),
            solver_config=DenseSolverConfig(),
            initial=initial,
            chunk_size=chunk_size,
            reuse_session=_DenseObservedReuseSession(),
            phase_recorder=None,
        )[0]

    # Both backend evaluators cancel intercept and slope before adding offset 1.
    # Changing this arithmetic order produces scale exp(0) instead of exp(1).
    assert prepare(cancellation) == prepare(ordinary)


def test_public_automatic_start_artifact_preserves_config_numeric_starts_and_predictions():
    x = np.linspace(0.0, 1.0, 80)
    frame = pd.DataFrame({"x": x})
    y = np.sin(4 * x) + np.random.default_rng(814).normal(scale=0.3, size=len(x))
    model = model_from_templates(
        family=GaussianLS(),
        predictors=[
            Predictor("location", {"x": Spline("cr", k=8)}),
            Predictor("scale", {}),
        ],
    ).fit_reml(frame, y, max_reml_iter=1)
    restored = SuperLSS.from_bytes(model.to_bytes())
    source = model._require_fitted().smoothing
    saved = restored._require_fitted().smoothing
    assert source.config.initial_lambda is None
    assert saved.config.initial_lambda is None
    assert saved.initial_lambdas == source.initial_lambdas
    assert all(isinstance(value, float) and value > 0 for value in saved.initial_lambdas.values())
    np.testing.assert_array_equal(restored.predict(frame), model.predict(frame))
    pd.testing.assert_frame_equal(
        restored.predict_parameters(frame), model.predict_parameters(frame), check_exact=True
    )
