"""REML coordinates for splines that opt out of SSP."""

import numpy as np
import pandas as pd
import pytest

from superglm import Spline, SuperGLM


@pytest.fixture
def spline_without_ssp(discrete):
    x = np.linspace(0.0, 1.0, 400)
    y = np.sin(2 * np.pi * x) + np.random.default_rng(0).normal(0.0, 0.3, x.size)
    return SuperGLM(
        family="gaussian",
        selection_penalty=0,
        discrete=discrete,
        features={"x": Spline(kind="cr", n_knots=8, penalty="none")},
    ).fit_reml(pd.DataFrame({"x": x}), y)


@pytest.mark.parametrize("discrete", [False, True], ids=["exact", "discrete"])
def test_reml_preserves_spline_projection(spline_without_ssp, discrete):
    group = spline_without_ssp._dm.group_matrices[0]
    np.testing.assert_array_equal(group.R_inv, group.projection)


@pytest.mark.parametrize("discrete", [False, True], ids=["exact", "discrete"])
def test_reml_publishes_objective_in_original_coordinates(spline_without_ssp, discrete):
    diagnostics = spline_without_ssp.reml_diagnostics()
    assert diagnostics["objective"] == pytest.approx(
        diagnostics["objective_history"][-1], rel=0, abs=1e-12
    )


@pytest.mark.pins_behaviour
@pytest.mark.parametrize("discrete", [False, True], ids=["exact", "discrete"])
def test_a_row_subset_keeps_the_spline_out_of_ssp(spline_without_ssp, discrete):
    """Pin the opt-out on a row subset, which a later lambda update reads before whitening."""
    group = spline_without_ssp._dm.group_matrices[0]
    assert group.row_subset(np.arange(0, 400, 2)).reparametrize is False
