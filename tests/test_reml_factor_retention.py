"""RFC-12b Task 1: retained slope-system decomposition on PIRLSResult.

The cached-factor line search (design note
docs/superpowers/plans/2026-07-29-rfc12b-cached-linesearch.md) needs the
centered slope-system ``RankDecomposition`` that ``fit_irls_direct``
already computes for REML geometry, instead of only its pseudo-inverse.
Retention is opt-in so existing callers keep exactly their current
behavior and memory profile.
"""

import numpy as np
import pytest

from superglm.distributions import Gaussian
from superglm.group_matrix import DenseGroupMatrix, DesignMatrix
from superglm.links import IdentityLink
from superglm.solvers.irls_direct import fit_irls_direct
from superglm.types import GroupSlice


@pytest.fixture
def penalized_fit_kwargs():
    x = np.linspace(-1.3, 1.7, 31)
    X = np.column_stack((x + 7.0, x**2 - 4.0))
    y = 0.4 + 0.8 * x - 0.2 * x**2
    dm = DesignMatrix([DenseGroupMatrix(X)], n=len(y), p=2)
    groups = [GroupSlice(name="x", start=0, end=2)]
    return dict(
        X=dm,
        y=y,
        weights=np.ones_like(y),
        family=Gaussian(),
        link=IdentityLink(),
        groups=groups,
        lambda2=0.0,
        max_iter=5,
        return_xtwx=True,
        S_override=np.diag([0.7, 1.4]),
    )


class TestRemlDecompositionRetention:
    def test_default_call_retains_nothing(self, penalized_fit_kwargs):
        result, _inverse, _gram = fit_irls_direct(**penalized_fit_kwargs)
        assert result.reml_slope_decomposition is None

    def test_opt_in_retains_solving_decomposition(self, penalized_fit_kwargs):
        result, inverse, _gram = fit_irls_direct(
            **penalized_fit_kwargs, retain_reml_decomposition=True
        )
        decomposition = result.reml_slope_decomposition
        assert decomposition is not None

        rhs = np.array([0.3, -1.1])
        np.testing.assert_allclose(decomposition.solve(rhs), inverse @ rhs, rtol=1e-10, atol=1e-12)
        # log_det_H measure contract (solvers/pirls.py): at full rank the
        # intercept-profiled log|H_aug| = log(sum_w) + log|H_c|_+.
        assert result.reml_geometry is not None
        np.testing.assert_allclose(
            float(np.log(result.reml_geometry.sum_w)) + decomposition.log_pdet,
            result.log_det_H,
            rtol=1e-12,
        )

    def test_retention_without_reml_geometry_stays_none(self, penalized_fit_kwargs):
        kwargs = {**penalized_fit_kwargs, "return_xtwx": False}
        result, _inverse = fit_irls_direct(
            **kwargs,
            retain_reml_decomposition=True,
            _compute_reml_geometry=False,
            _compute_fit_statistics=False,
            compute_rank_info=False,
        )
        assert result.reml_slope_decomposition is None
