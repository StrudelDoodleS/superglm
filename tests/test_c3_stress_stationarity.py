"""Preserve certified existing-optimizer resolutions of the original C3 fixtures."""

from dataclasses import replace

import numpy as np
import pytest
from benchmarks._c3_c1_fixtures import fixture, lss_model, marked_book
from threadpoolctl import threadpool_limits

from superglm.distributional import GeneralizedParetoLSS


@pytest.mark.slow
@pytest.mark.parametrize("case", ["gpd", "tweedie"])
def test_original_stress_fixture_has_fresh_strict_newton_authority(case):
    if case == "tweedie":
        model, frame, y, weights = fixture(10000, mix=0.75, family="tweedie")
        expected_dimensions = (10000, 192)
    else:
        book = marked_book(10000, tail=True)
        keep = (book["policy"] < 10000) & (book["losses"] > 1000)
        frame = book["frame"].iloc[book["policy"][keep]].reset_index(drop=True)
        y = book["losses"][keep] - 1000
        weights = np.ones(len(y))
        model = lss_model(GeneralizedParetoLSS(), 3)
        expected_dimensions = (1401, 98)
    with threadpool_limits(limits=1):
        model.fit_reml(
            frame,
            y,
            sample_weight=weights,
            outer="efs+newton",
            practical_reml=False,
            initial_lambda=0.01,
            max_reml_iter=100,
            acceleration="multisecant",
        )
    smoothing = model._require_fitted().smoothing
    assert (len(y), len(model.result_.coefficients)) == expected_dimensions
    assert model.result_.converged and model.smoothing_certified_
    assert smoothing.convergence_reason == "stationary"
    assert smoothing.terminal_fit.converged
    assert smoothing.terminal_evidence_fresh
    assert not smoothing.unresolved_upper_bound
    # The criterion is the configured objective-scaled projected score, not a
    # fixed coefficient tolerance or a platform-specific roundoff magnitude.
    bar = smoothing.config.tolerance * (1 + abs(smoothing.objective))
    assert smoothing.terminal_projected_gradient_norm <= bar
    assert all(value <= bar for value in smoothing.terminal_gradient_certificate.values())
    # Mutating the published authority must be rejected on result replay.
    with pytest.raises(ValueError, match="stationary"):
        replace(smoothing, terminal_projected_gradient_norm=2 * bar)
