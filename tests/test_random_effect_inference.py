"""Compact covariance and retained-state tests for random effects."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import superglm.model.state_ops as state_ops
from superglm import Numeric, RandomEffect, SuperGLM
from superglm.inference.covariance import StructuredCovarianceAccessor
from superglm.solvers.structured import StructuredLinearSystemState


def _fit_pair(
    *,
    retain_fit_state: bool = True,
    n_levels: int = 18,
    fit_dense: bool = True,
    max_reml_iter: int = 7,
) -> tuple[SuperGLM | None, SuperGLM, pd.DataFrame, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(862)
    repeats = 18
    codes = np.repeat(np.arange(n_levels), repeats)
    rng.shuffle(codes)
    x = rng.normal(size=len(codes))
    exposure = rng.uniform(0.5, 1.8, size=len(codes))
    effects = rng.normal(scale=0.3, size=n_levels)
    y = rng.poisson(exposure * np.exp(-0.3 + 0.2 * x + effects[codes])).astype(float)
    X = pd.DataFrame(
        {
            "x": x,
            "broker": np.array([f"b{i}" for i in codes], dtype=object),
        }
    )
    common = {
        "family": "poisson",
        "features": {"x": Numeric(), "broker": RandomEffect()},
        "selection_penalty": 0,
        "retain_fit_state": retain_fit_state,
    }
    dense = SuperGLM(**common, direct_solve="gram") if fit_dense else None
    structured = SuperGLM(**common, direct_solve="structured")
    if dense is not None:
        dense.fit_reml(X, y, offset=np.log(exposure), max_reml_iter=max_reml_iter)
    structured.fit_reml(
        X,
        y,
        offset=np.log(exposure),
        max_reml_iter=max_reml_iter,
    )
    return dense, structured, X, y, exposure


def test_structured_selected_covariance_matches_dense_augmented_inverse(monkeypatch):
    dense, structured, _, _, _ = _fit_pair()
    assert dense is not None

    def fail_dense_legacy(*_args, **_kwargs):
        raise AssertionError("structured inference rebuilt a dense coefficient system")

    monkeypatch.setattr(state_ops, "_legacy_active_state", fail_dense_legacy)

    dense_augmented = dense._fit_inference_info["XtWX_inv_aug"]
    compact_info = structured._fit_inference_info
    compact = compact_info["XtWX_inv_aug"]

    assert isinstance(structured._linear_system_state, StructuredLinearSystemState)
    assert isinstance(compact, StructuredCovarianceAccessor)

    # Intercept, the numeric slope, and a few random-effect levels exercise
    # every augmented covariance block without asking for the full K x K block.
    selected = np.array([0, 1, 2, 5, 9], dtype=np.intp)
    np.testing.assert_allclose(
        compact.selected_block(selected),
        dense_augmented[np.ix_(selected, selected)],
        rtol=3e-8,
        atol=3e-9,
    )
    np.testing.assert_allclose(
        compact.selected_diagonal(selected),
        np.diag(dense_augmented)[selected],
        rtol=3e-8,
        atol=3e-9,
    )

    slope_indices = np.array([0, 1, 4, 8], dtype=np.intp)
    np.testing.assert_allclose(
        compact.slope_selected_block(slope_indices),
        dense_augmented[1:, 1:][np.ix_(slope_indices, slope_indices)],
        rtol=3e-8,
        atol=3e-9,
    )
    np.testing.assert_allclose(
        compact.intercept_cross(slope_indices),
        dense_augmented[0, 1:][slope_indices],
        rtol=3e-8,
        atol=3e-9,
    )
    assert compact.intercept_variance() == pytest.approx(
        dense_augmented[0, 0],
        rel=3e-8,
        abs=3e-9,
    )
    assert compact_info["group_edf_map"]["broker"] == pytest.approx(
        dense._fit_inference_info["group_edf_map"]["broker"],
        rel=3e-8,
        abs=3e-9,
    )


def test_structured_summary_uses_selected_covariance_only(monkeypatch):
    _, structured, X, y, _ = _fit_pair()

    def fail_dense_legacy(*_args, **_kwargs):
        raise AssertionError("summary requested the legacy dense covariance path")

    monkeypatch.setattr(state_ops, "_legacy_active_state", fail_dense_legacy)

    summary = structured.summary()
    assert summary["fit"]["n_obs"] > 0
    assert np.isfinite(structured.metrics(X, y).coefficient_se["x"][0])


def test_released_structured_state_keeps_compact_factors_and_support():
    _, structured, X, _, exposure = _fit_pair(
        retain_fit_state=False,
        n_levels=48,
        fit_dense=False,
    )

    state = structured._linear_system_state
    assert isinstance(state, StructuredLinearSystemState)
    assert structured._dm is None
    assert structured._fit_weights is None
    assert structured._fit_X_ref is None
    assert isinstance(
        structured.__dict__["_fit_inference_info"]["XtWX_inv_aug"],
        StructuredCovarianceAccessor,
    )
    assert state.coefficient_factor.shape == (len(structured.result.beta),) * 2
    assert state.augmented_factor.shape == (len(structured.result.beta) + 1,) * 2
    assert state.backend == "structured"
    assert "broker" in state.support_totals
    support = state.support_totals["broker"]
    assert int(np.sum(support.count)) == len(X)
    assert support.information.shape == (48,)

    predictions = structured.predict(X.head(8), offset=np.log(exposure[:8]))
    assert np.all(np.isfinite(predictions))
    assert structured.summary()["fit"]["n_obs"] == len(X)


def test_structured_state_has_no_dominant_square_array():
    _, structured, _, _, _ = _fit_pair(
        n_levels=270,
        fit_dense=False,
        max_reml_iter=2,
    )
    state = structured._linear_system_state
    assert isinstance(state, StructuredLinearSystemState)
    dominant_size = len(state.system.operator.structured_indices)

    arrays = [
        value
        for owner in (
            state,
            state.system,
            state.system.operator,
            state.coefficient_factor,
            state.augmented_factor,
        )
        for value in vars(owner).values()
        if isinstance(value, np.ndarray)
    ]
    assert all(array.shape != (dominant_size, dominant_size) for array in arrays)
