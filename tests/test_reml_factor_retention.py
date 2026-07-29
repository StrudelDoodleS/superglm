"""Retained slope-system decomposition on PIRLSResult (factor-L seam).

The seam retains the centered slope-system ``RankDecomposition`` that
``fit_irls_direct`` already computes for REML geometry, instead of only
its pseudo-inverse. Its live consumers are RFC-2/RFC-7 (route solves
through the factor rather than a materialized p x p pseudo-inverse);
RFC-12b, the retired original motivation, is dispositioned in
docs/superpowers/plans/2026-07-29-rfc12b-cached-linesearch.md. Retention
is opt-in (internal ``_retain_reml_decomposition`` kwarg) so existing
callers keep exactly their current behavior and memory profile.
"""

import numpy as np
import pytest

from superglm.distributions import Gaussian
from superglm.group_matrix import (
    DenseGroupMatrix,
    DesignMatrix,
    RandomEffectGroupMatrix,
)
from superglm.links import IdentityLink
from superglm.solvers.irls_direct import fit_irls_direct
from superglm.types import GroupSlice, PenaltyComponent


@pytest.fixture
def penalized_fit_inputs():
    x = np.linspace(-1.3, 1.7, 31)
    X_raw = np.column_stack((x + 7.0, x**2 - 4.0))
    y = 0.4 + 0.8 * x - 0.2 * x**2
    S = np.diag([0.7, 1.4])
    dm = DesignMatrix([DenseGroupMatrix(X_raw)], n=len(y), p=2)
    groups = [GroupSlice(name="x", start=0, end=2)]
    kwargs = dict(
        X=dm,
        y=y,
        weights=np.ones_like(y),
        family=Gaussian(),
        link=IdentityLink(),
        groups=groups,
        lambda2=0.0,
        max_iter=5,
        return_xtwx=True,
        S_override=S,
    )
    return kwargs, X_raw, S


class TestRemlDecompositionRetention:
    def test_default_call_retains_nothing(self, penalized_fit_inputs):
        kwargs, _X_raw, _S = penalized_fit_inputs
        result, _inverse, _gram = fit_irls_direct(**kwargs)
        assert result.reml_slope_decomposition is None

    def test_opt_in_retention_matches_independent_oracle(self, penalized_fit_inputs):
        kwargs, X_raw, S = penalized_fit_inputs
        result, inverse, _gram = fit_irls_direct(**kwargs, _retain_reml_decomposition=True)
        decomposition = result.reml_slope_decomposition
        assert decomposition is not None

        # Independent oracle: Gaussian/identity with unit weights means W = 1,
        # so the centered slope Hessian is (X - mean)' (X - mean) + S exactly.
        centered = X_raw - X_raw.mean(axis=0)
        H_c = centered.T @ centered + S
        rhs = np.array([0.3, -1.1])
        np.testing.assert_allclose(
            decomposition.solve(rhs), np.linalg.solve(H_c, rhs), rtol=1e-9, atol=1e-12
        )
        _sign, oracle_logdet = np.linalg.slogdet(H_c)
        np.testing.assert_allclose(decomposition.log_pdet, oracle_logdet, rtol=1e-10)

        # Consistency with the published geometry: same solve as the returned
        # inverse, and the log_det_H identified-coordinate measure
        # log(sum_w) + log|H_c|_+ (contract comment on PIRLSResult.log_det_H).
        np.testing.assert_allclose(decomposition.solve(rhs), inverse @ rhs, rtol=1e-10)
        assert result.reml_geometry is not None
        np.testing.assert_allclose(
            float(np.log(result.reml_geometry.sum_w)) + decomposition.log_pdet,
            result.log_det_H,
            rtol=1e-12,
        )

    def test_retention_through_certified_truncated_path(self):
        # A duplicated column with no penalty forces a rank-deficient slope
        # Hessian, taking the certification branch. The retained object must
        # be the certified decomposition, and the identified-coordinate
        # log_det_H measure must still hold under truncation — the branch
        # any factor-routing consumer must gate on (method/rank_truncated).
        x = np.linspace(-1.0, 1.0, 40)
        y = 1.0 + 0.5 * x
        dm = DesignMatrix([DenseGroupMatrix(np.column_stack((x, x)))], n=len(y), p=2)
        groups = [GroupSlice(name="x", start=0, end=2)]
        result, inverse, _gram = fit_irls_direct(
            X=dm,
            y=y,
            weights=np.ones_like(y),
            family=Gaussian(),
            link=IdentityLink(),
            groups=groups,
            lambda2=0.0,
            max_iter=5,
            return_xtwx=True,
            S_override=np.zeros((2, 2)),
            _retain_reml_decomposition=True,
        )
        decomposition = result.reml_slope_decomposition
        assert decomposition is not None
        # The certified decompose_factor result (method qr_svd), not the raw
        # pre-certification decompose_gram one (pivoted_cholesky) — every
        # other quantity below agrees between the two objects.
        assert decomposition.method == "qr_svd"
        assert decomposition.rank == 1
        assert decomposition.rank_truncated
        # A live cholesky_factor coexists with truncation (it factors the
        # representative submatrix): consumers gate on method/rank_truncated,
        # never on factor presence.
        assert decomposition.cholesky_factor is not None
        assert result.reml_hessian_rank == 1 + decomposition.rank

        rhs = np.array([0.4, -0.9])
        np.testing.assert_allclose(decomposition.solve(rhs), inverse @ rhs, rtol=1e-10)
        assert result.reml_geometry is not None
        np.testing.assert_allclose(
            float(np.log(result.reml_geometry.sum_w)) + decomposition.log_pdet,
            result.log_det_H,
            rtol=1e-12,
        )

    def test_retention_without_reml_geometry_stays_none(self, penalized_fit_inputs):
        kwargs, _X_raw, _S = penalized_fit_inputs
        kwargs = {**kwargs, "return_xtwx": False}
        result, _inverse = fit_irls_direct(
            **kwargs,
            _retain_reml_decomposition=True,
            _compute_reml_geometry=False,
            _compute_fit_statistics=False,
            compute_rank_info=False,
        )
        assert result.reml_slope_decomposition is None

    def test_structured_path_retains_nothing_by_design(self):
        # Structured Schur factors have their own retained-factor
        # protocol, so the structured backend must retain None even with
        # retention requested.
        rng = np.random.default_rng(701)
        n = 420
        n_levels = 24
        codes = rng.integers(0, n_levels, size=n, dtype=np.intp)
        numeric = rng.normal(size=(n, 2))
        y = -0.25 + numeric @ np.array([0.3, -0.18]) + rng.normal(scale=0.3, size=n)
        matrices = [DenseGroupMatrix(numeric), RandomEffectGroupMatrix(codes, n_levels)]
        groups = [
            GroupSlice(name="numeric", start=0, end=2, penalized=False),
            GroupSlice(name="policy", start=2, end=2 + n_levels, penalized=True),
        ]
        dm = DesignMatrix(matrices, n=n, p=2 + n_levels)
        penalties = [
            PenaltyComponent(
                name="policy",
                group_name="policy",
                group_index=1,
                group_sl=groups[1].sl,
                omega_raw=None,
                penalty_kind="identity",
            )
        ]
        result, _factor, _operator = fit_irls_direct(
            X=dm,
            y=y,
            weights=np.ones(n),
            family=Gaussian(),
            link=IdentityLink(),
            groups=groups,
            lambda2={"policy": 2.75},
            max_iter=50,
            tol=1e-9,
            return_xtwx=True,
            direct_solve="structured",
            reml_penalties=penalties,
            _retain_reml_decomposition=True,
        )
        assert result.direct_backend == "structured"
        assert result.reml_slope_decomposition is None
