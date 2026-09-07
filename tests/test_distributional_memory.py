from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from superglm._frame import as_eager_frame
from superglm.distributional.families.gaussian import GaussianLS
from superglm.distributional.family import COMPLETE_OBSERVATION
from superglm.distributional.layout import build_stacked_layout
from superglm.distributional.predictor import Predictor, compile_predictors
from superglm.distributional.result import DenseSolverConfig
from superglm.distributional.solver import fit_dense_fixed_lambda
from superglm.distributional.weights import WeightContract, resolve_likelihood_weights
from superglm.features import Numeric, Spline
from superglm.group_matrix import DesignMatrix

from ._distributional_weights import resolved_prior


def _discrete_problem(n: int = 137):
    rng = np.random.default_rng(7764)
    x = np.linspace(-1.0, 1.0, n)
    z = np.mod(0.13 + 1.7 * x, 1.0)
    frame = as_eager_frame(pd.DataFrame({"x": x, "z": z}))
    family = GaussianLS(scale_floor=0.015)
    weights = np.linspace(0.6, 1.4, n)
    response = (
        0.4 + 0.8 * x + rng.normal(scale=family.scale_floor + np.exp(-1.1 + 0.25 * z), size=n)
    )
    layout = build_stacked_layout(
        compile_predictors(
            frame,
            resolved_prior(weights),
            family.parameters,
            (
                Predictor(
                    "location",
                    {
                        "x": Numeric(),
                        "z": Spline(kind="cr", n_knots=7, discrete=True),
                    },
                ),
                Predictor(
                    "scale",
                    {"z": Spline(kind="cr", n_knots=6, discrete=True)},
                ),
            ),
            model_discrete=True,
            n_bins_config=23,
        )
    )
    penalty = layout.penalty_matrix(
        {name: 0.3 + 0.05 * index for index, name in enumerate(layout.penalty_names)}
    )
    return family, layout, response, weights, penalty


def _install_shape_sentinel(
    monkeypatch: pytest.MonkeyPatch,
    forbidden: set[tuple[int, ...]],
) -> list[tuple[int, ...]]:
    observed: list[tuple[int, ...]] = []
    for name in (
        "array",
        "asarray",
        "column_stack",
        "empty",
        "empty_like",
        "full",
        "ones",
        "zeros",
        "zeros_like",
    ):
        original: Callable = getattr(np, name)

        def checked(*args, _original=original, **kwargs):
            result = _original(*args, **kwargs)
            if isinstance(result, np.ndarray):
                observed.append(result.shape)
                if result.shape in forbidden:
                    raise AssertionError(f"forbidden observation-scale allocation {result.shape}")
            return result

        monkeypatch.setattr(np, name, checked)
    return observed


def test_discrete_chunked_fit_never_allocates_full_design_or_curvature_channels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    family, layout, response, weights, penalty = _discrete_problem()
    config = DenseSolverConfig(
        max_iterations=30,
        terminal_retry_iterations=5,
        tolerance=1.0e-7,
    )
    likelihood_plan = family.bind_likelihood(
        response,
        resolve_likelihood_weights(
            weights,
            n_observations=len(response),
            contract=WeightContract(semantics="prior"),
        ),
        COMPLETE_OBSERVATION,
    )

    # Numba resolves global NumPy functions when it first compiles each array
    # signature. Compile this tiny fit before the sentinel replaces those
    # functions so the sentinel observes runtime Python allocations only.
    fit_dense_fixed_lambda(
        family,
        layout,
        response,
        likelihood_plan,
        penalty,
        config=config,
        chunk_size=19,
    )
    n = len(response)
    n_channels = len(family.parameters) * (len(family.parameters) + 1) // 2
    assert layout.n_coefficients not in {len(family.parameters), n_channels}
    forbidden = {
        (n, layout.n_coefficients),
        (n, n_channels),
    }

    def forbid_toarray(*_args, **_kwargs):
        raise AssertionError("chunked discrete fitting materialized a full design")

    monkeypatch.setattr(DesignMatrix, "toarray", forbid_toarray)
    observed = _install_shape_sentinel(monkeypatch, forbidden)
    result = fit_dense_fixed_lambda(
        family,
        layout,
        response,
        likelihood_plan,
        penalty,
        config=config,
        chunk_size=19,
    )

    assert np.all(np.isfinite(result.coefficients))
    assert result.eta.shape == (n, len(family.parameters))
    assert result.theta.shape == result.eta.shape
    assert forbidden.isdisjoint(observed)
    assert any(shape[0] == 19 for shape in observed if shape)
