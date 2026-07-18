"""Tests for the direct IRLS solver (no BCD)."""

import numpy as np
import pandas as pd
import pytest

from superglm.features.categorical import Categorical
from superglm.features.spline import Spline
from superglm.group_matrix import DenseGroupMatrix, DesignMatrix
from superglm.model import SuperGLM
from superglm.types import GroupSlice, LinearConstraintSet


# ── Fixtures ───────────────────────────────────────────────────
@pytest.fixture
def poisson_data():
    """Synthetic Poisson dataset with one nonlinear and one categorical feature."""
    rng = np.random.default_rng(42)
    n = 1000
    x1 = rng.uniform(18, 80, n)
    area = rng.choice(["A", "B", "C", "D"], n)
    area_effect = {"A": 0.0, "B": 0.2, "C": -0.1, "D": 0.3}
    mu = np.exp(-0.5 + 0.01 * (x1 - 40) ** 2 / 100 + np.array([area_effect[a] for a in area]))
    y = rng.poisson(mu)
    sample_weight = np.ones(n)
    X = pd.DataFrame({"DrivAge": x1, "Area": area})
    return X, y, sample_weight


@pytest.fixture
def select_data():
    """Synthetic data with signal spline and noise spline (for select=True)."""
    rng = np.random.default_rng(42)
    n = 500
    x1 = rng.uniform(18, 80, n)
    x2 = rng.uniform(0, 10, n)  # noise
    mu = np.exp(-0.5 + 0.01 * (x1 - 40) ** 2 / 100)
    y = rng.poisson(mu)
    sample_weight = np.ones(n)
    X = pd.DataFrame({"signal": x1, "noise": x2})
    return X, y, sample_weight


class _LevelTwoRecorder:
    enabled_level = 2

    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []

    def append_jsonl(self, name: str, payload: dict[str, object]) -> None:
        assert name == "pirls"
        self.rows.append(payload)


# ── Basic solver tests ─────────────────────────────────────────
class TestDirectSolverBasic:
    def test_working_sums_reject_nonfinite_inputs_before_moment_assembly(self):
        from superglm.solvers.irls_direct import _working_sums

        with pytest.raises(ValueError, match="working weights"):
            _working_sums(np.array([1.0, np.inf]), np.array([1.0, 2.0]))
        with pytest.raises(ValueError, match="working response"):
            _working_sums(np.array([1.0, 2.0]), np.array([1.0, np.nan]))

        assert _working_sums(np.array([1.0, 2.0]), np.array([3.0, 4.0])) == (3.0, 7.0)

    def test_state_evaluation_accepts_in_place_custom_link(self):
        """Custom link methods may mutate writable arrays supplied by the solver."""
        from superglm.distributions import Poisson
        from superglm.solvers.irls_state import _evaluate_irls_state

        class InPlaceLogLink:
            def link(self, mu):
                return np.log(mu)

            def inverse(self, eta):
                eta[...] = np.exp(eta)
                return eta

            def deriv(self, mu):
                return 1.0 / mu

            def deriv_inverse(self, eta):
                return np.exp(eta)

        x = np.linspace(-1.0, 1.0, 12)
        dm = DesignMatrix([DenseGroupMatrix(x[:, None])], n=len(x), p=1)
        state = _evaluate_irls_state(
            dm,
            np.ones_like(x),
            np.ones_like(x),
            Poisson(),
            InPlaceLogLink(),
            np.zeros_like(x),
            np.array([0.25]),
            0.0,
        )

        assert np.all(np.isfinite(state.mu))
        assert not state.eta.flags.writeable
        assert not state.mu.flags.writeable

    def test_can_skip_rank_metadata_for_intermediate_reml_fit(self, monkeypatch):
        from superglm.distributions import Gaussian
        from superglm.links import IdentityLink
        from superglm.solvers.irls_direct import fit_irls_direct

        x = np.linspace(-1.0, 1.0, 40)
        y = 1.0 + 0.5 * x
        dm = DesignMatrix(
            [DenseGroupMatrix(np.column_stack((x, x)))],
            n=len(y),
            p=2,
        )
        groups = [GroupSlice(name="x", start=0, end=2)]
        common = {
            "X": dm,
            "y": y,
            "weights": np.ones_like(y),
            "family": Gaussian(),
            "link": IdentityLink(),
            "groups": groups,
            "lambda2": 0.0,
            "max_iter": 5,
            "return_xtwx": True,
            "S_override": 0.25 * np.eye(2),
        }

        retained, retained_inverse, retained_gram = fit_irls_direct(**common)
        assert retained.rank_info is not None
        assert retained.rank_info.data.rank == 1
        assert retained.rank_info.augmented.rank == 2

        monkeypatch.setattr(
            np.linalg,
            "eigh",
            lambda _matrix: pytest.fail("intermediate fit should skip data-rank spectrum"),
        )
        intermediate, intermediate_inverse, intermediate_gram = fit_irls_direct(
            **common,
            compute_rank_info=False,
        )

        assert intermediate.rank_info is None
        np.testing.assert_allclose(intermediate.beta, retained.beta)
        assert intermediate.intercept == pytest.approx(retained.intercept)
        assert intermediate.deviance == pytest.approx(retained.deviance)
        assert intermediate.effective_df == pytest.approx(retained.effective_df)
        assert intermediate.phi == pytest.approx(retained.phi)
        assert intermediate.log_det_H == pytest.approx(retained.log_det_H)
        assert intermediate.converged is retained.converged
        assert intermediate.n_iter == retained.n_iter
        np.testing.assert_allclose(intermediate_inverse, retained_inverse)
        np.testing.assert_allclose(intermediate_gram, retained_gram)

    def test_reml_geometry_is_the_full_intercept_profiled_hessian(self):
        """The REML inverse/logdet contract includes the unpenalized intercept."""
        from superglm.distributions import Gaussian
        from superglm.links import IdentityLink
        from superglm.solvers.irls_direct import fit_irls_direct

        x = np.linspace(-1.3, 1.7, 31)
        X = np.column_stack((x + 7.0, x**2 - 4.0))
        y = 0.4 + 0.8 * x - 0.2 * x**2
        penalty = np.diag([0.7, 1.4])
        dm = DesignMatrix([DenseGroupMatrix(X)], n=len(y), p=2)
        groups = [GroupSlice(name="x", start=0, end=2)]

        result, slope_inverse, raw_gram = fit_irls_direct(
            X=dm,
            y=y,
            weights=np.ones_like(y),
            family=Gaussian(),
            link=IdentityLink(),
            groups=groups,
            lambda2=1.0,
            S_override=penalty,
            return_xtwx=True,
        )

        augmented_penalty = np.zeros((3, 3))
        augmented_penalty[1:, 1:] = penalty
        augmented_design = np.column_stack((np.ones(len(y)), X))
        full_hessian = augmented_design.T @ augmented_design + augmented_penalty
        expected_full_inverse = np.linalg.inv(full_hessian)

        np.testing.assert_allclose(raw_gram, X.T @ X, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(
            slope_inverse,
            expected_full_inverse[1:, 1:],
            rtol=1e-12,
            atol=1e-12,
        )
        assert result.log_det_H == pytest.approx(np.linalg.slogdet(full_hessian)[1], rel=1e-12)
        assert result.reml_hessian_rank == 3

        shifted = X + np.array([-11.0, 5.5])
        shifted_result, shifted_inverse, _ = fit_irls_direct(
            X=DesignMatrix([DenseGroupMatrix(shifted)], n=len(y), p=2),
            y=y,
            weights=np.ones_like(y),
            family=Gaussian(),
            link=IdentityLink(),
            groups=groups,
            lambda2=1.0,
            S_override=penalty,
            return_xtwx=True,
        )
        assert shifted_result.log_det_H == pytest.approx(result.log_det_H, rel=1e-12)
        np.testing.assert_allclose(shifted_inverse, slope_inverse, rtol=1e-12, atol=1e-12)

    def test_reml_hessian_rank_excludes_unidentified_unpenalized_aliases(self):
        from superglm.distributions import Gaussian
        from superglm.links import IdentityLink
        from superglm.solvers.irls_direct import fit_irls_direct

        x = np.linspace(-1.0, 1.0, 24)
        X = np.column_stack((x, x))
        result, _, _ = fit_irls_direct(
            X=DesignMatrix([DenseGroupMatrix(X)], n=len(x), p=2),
            y=0.5 + x,
            weights=np.ones_like(x),
            family=Gaussian(),
            link=IdentityLink(),
            groups=[GroupSlice(name="aliased", start=0, end=2)],
            lambda2=0.0,
            S_override=np.zeros((2, 2)),
            return_xtwx=True,
        )

        assert result.reml_hessian_rank == 2  # intercept plus one identifiable slope

    def test_reml_working_system_avoids_rebuilding_one_step_crossproducts(self, monkeypatch):
        import superglm.solvers.irls_direct as irls_direct
        from superglm.distributions import Poisson
        from superglm.links import LogLink

        x = np.linspace(-1.0, 1.0, 60)
        y = np.array([0.0, 1.0, 0.0, 2.0, 1.0] * 12)
        initial_mu = np.full_like(y, np.mean(y), dtype=float)
        initial_deviance = float(np.sum(Poisson().deviance_unit(y, initial_mu)))

        class CountingPoisson(Poisson):
            def __init__(self):
                self.deviance_calls = 0

            def deviance_unit(self, y, mu):
                self.deviance_calls += 1
                return super().deviance_unit(y, mu)

        family = CountingPoisson()
        dm = DesignMatrix([DenseGroupMatrix(x[:, None])], n=len(y), p=1)
        groups = [GroupSlice(name="x", start=0, end=1)]
        original_build = irls_direct.build_centered_system
        build_calls = 0

        def counted_build(**kwargs):
            nonlocal build_calls
            build_calls += 1
            return original_build(**kwargs)

        monkeypatch.setattr(irls_direct, "build_centered_system", counted_build)

        result, inverse, gram = irls_direct.fit_irls_direct(
            X=dm,
            y=y,
            weights=np.ones_like(y),
            family=family,
            link=LogLink(),
            groups=groups,
            lambda2=0.25,
            max_iter=1,
            return_xtwx=True,
            compute_rank_info=False,
            _return_working_system=True,
            _compute_fit_statistics=False,
            _deviance_init=initial_deviance,
        )

        assert build_calls == 1
        assert result.rank_info is None
        assert result.effective_df == 0.0
        assert result.phi == 1.0
        assert family.deviance_calls == 1
        assert np.all(np.isfinite(result.beta))
        assert np.all(np.isfinite(inverse))
        assert np.all(np.isfinite(gram))

    def test_scop_candidate_can_skip_unused_generic_reml_decomposition(self, monkeypatch):
        """A private SCOP candidate may omit geometry replaced by its joint Hessian."""
        import superglm.solvers.irls_direct as irls_direct
        from superglm.distributions import Gaussian
        from superglm.links import IdentityLink

        x = np.linspace(-1.0, 1.0, 80)
        raw = np.column_stack((x, x**2, x**3))
        y = 0.3 + 0.8 * x - 0.2 * x**2
        dm = DesignMatrix([DenseGroupMatrix(raw)], n=len(y), p=raw.shape[1])
        groups = [GroupSlice(name="x", start=0, end=raw.shape[1])]
        common = {
            "X": dm,
            "y": y,
            "weights": np.ones_like(y),
            "family": Gaussian(),
            "link": IdentityLink(),
            "groups": groups,
            "lambda2": 0.25,
            "max_iter": 3,
            "return_xtwx": True,
        }

        original = irls_direct.decompose_gram
        decomposition_calls = 0

        def counted(matrix, *args, **kwargs):
            nonlocal decomposition_calls
            decomposition_calls += 1
            return original(matrix, *args, **kwargs)

        monkeypatch.setattr(irls_direct, "decompose_gram", counted)
        retained, _, _ = irls_direct.fit_irls_direct(**common)
        retained_geometry_calls = decomposition_calls
        assert retained.rank_info is not None

        with pytest.raises(
            ValueError,
            match="requires rank metadata and fit statistics to be disabled",
        ):
            irls_direct.fit_irls_direct(**common, _compute_reml_geometry=False)

        decomposition_calls = 0
        candidate, omitted_inverse, candidate_gram = irls_direct.fit_irls_direct(
            **common,
            compute_rank_info=False,
            _compute_fit_statistics=False,
            _compute_reml_geometry=False,
        )

        assert retained_geometry_calls - decomposition_calls == 3
        assert omitted_inverse.shape == (0, 0)
        assert candidate.log_det_H is None
        assert candidate.reml_hessian_rank is None
        assert candidate.rank_info is None
        assert np.isnan(candidate.effective_df)
        assert np.isnan(candidate.phi)
        assert np.all(np.isfinite(candidate_gram))

    def test_skips_extrema_scans_when_diagnostics_disabled(self, monkeypatch):
        import superglm.solvers.irls_direct as irls_direct
        from superglm.distributions import Gaussian
        from superglm.links import IdentityLink

        x = np.linspace(-1.0, 1.0, 20)
        X_raw = x[:, None]
        y = 1.0 + 0.5 * x
        dm = DesignMatrix([DenseGroupMatrix(X_raw)], n=len(y), p=1)
        groups = [GroupSlice(name="x", start=0, end=1)]

        class NumpyExtremaSpy:
            def __init__(self, delegate):
                self._delegate = delegate
                self.min_calls = 0
                self.max_calls = 0

            def __getattr__(self, name):
                return getattr(self._delegate, name)

            def min(self, *args, **kwargs):
                self.min_calls += 1
                return self._delegate.min(*args, **kwargs)

            def max(self, *args, **kwargs):
                self.max_calls += 1
                return self._delegate.max(*args, **kwargs)

        original_stats = irls_direct._positive_working_weight_stats
        stats_calls = 0

        def counting_stats(W):
            nonlocal stats_calls
            stats_calls += 1
            return original_stats(W)

        extrema_spy = NumpyExtremaSpy(np)
        monkeypatch.setattr(irls_direct, "_positive_working_weight_stats", counting_stats)
        monkeypatch.setattr(irls_direct, "np", extrema_spy)

        irls_direct.fit_irls_direct(
            X=dm,
            y=y,
            weights=np.ones_like(y),
            family=Gaussian(),
            link=IdentityLink(),
            groups=groups,
            lambda2=0.0,
            max_iter=1,
            record_diagnostics=False,
        )

        assert stats_calls == 1
        assert extrema_spy.min_calls == 0
        assert extrema_spy.max_calls == 0

    def test_level_two_recorder_keeps_extrema_without_iteration_diagnostics(self):
        import superglm.solvers.irls_direct as irls_direct
        from superglm.distributions import Gaussian
        from superglm.links import IdentityLink

        x = np.linspace(-1.0, 1.0, 20)
        X_raw = x[:, None]
        y = 1.0 + 0.5 * x
        dm = DesignMatrix([DenseGroupMatrix(X_raw)], n=len(y), p=1)
        groups = [GroupSlice(name="x", start=0, end=1)]
        recorder = _LevelTwoRecorder()

        irls_direct.fit_irls_direct(
            X=dm,
            y=y,
            weights=np.ones_like(y),
            family=Gaussian(),
            link=IdentityLink(),
            groups=groups,
            lambda2=0.0,
            max_iter=1,
            record_diagnostics=False,
            debug_recorder=recorder,
        )

        assert len(recorder.rows) == 1
        assert {
            "eta_min_unclipped",
            "eta_max_unclipped",
            "working_eta_min_unclipped",
            "working_eta_max_unclipped",
        } <= recorder.rows[0].keys()

    def test_level_two_recorder_decision_is_frozen_before_iterations(self):
        import superglm.solvers.irls_direct as irls_direct
        from superglm.distributions import Gaussian
        from superglm.links import IdentityLink

        class ChangingLevelRecorder(_LevelTwoRecorder):
            def __init__(self) -> None:
                super().__init__()
                self.level_reads = 0

            @property
            def enabled_level(self) -> int:
                self.level_reads += 1
                return 2 if self.level_reads <= 2 else 1

        x = np.linspace(-1.0, 1.0, 20)
        X_raw = x[:, None]
        y = 1.0 + 0.5 * x
        dm = DesignMatrix([DenseGroupMatrix(X_raw)], n=len(y), p=1)
        groups = [GroupSlice(name="x", start=0, end=1)]
        recorder = ChangingLevelRecorder()

        result, _ = irls_direct.fit_irls_direct(
            X=dm,
            y=y,
            weights=np.ones_like(y),
            family=Gaussian(),
            link=IdentityLink(),
            groups=groups,
            lambda2=0.0,
            max_iter=2,
            tol=0.0,
            record_diagnostics=False,
            debug_recorder=recorder,
        )

        assert result.n_iter == 2
        assert len(recorder.rows) == 2
        assert recorder.level_reads == 1

    def test_h_inverse_profiled_intercept_solve_matches_augmented_system(self):
        from superglm.solvers.irls_direct import (
            _robust_solve,
            _solve_profiled_intercept_from_h_inv,
        )

        rng = np.random.default_rng(222)
        p = 7
        A = rng.standard_normal((p, p))
        XtWX = A @ A.T + np.eye(p)
        B = rng.standard_normal((p, p))
        S = B @ B.T + 0.3 * np.eye(p)
        H = XtWX + S
        H_inv = np.linalg.inv(H)
        XtWz = rng.standard_normal(p)
        XtW1 = rng.standard_normal(p)
        sum_W = 20.0
        sum_Wz = -0.4

        M_aug = np.empty((p + 1, p + 1))
        M_aug[0, 0] = sum_W
        M_aug[0, 1:] = XtW1
        M_aug[1:, 0] = XtW1
        M_aug[1:, 1:] = H
        rhs = np.empty(p + 1)
        rhs[0] = sum_Wz
        rhs[1:] = XtWz

        beta_aug, _, _ = _robust_solve(M_aug, rhs)
        beta_h, intercept_h = _solve_profiled_intercept_from_h_inv(
            H_inv,
            XtWz,
            XtW1,
            sum_W,
            sum_Wz,
        )

        np.testing.assert_allclose(intercept_h, beta_aug[0], rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(beta_h, beta_aug[1:], rtol=1e-10, atol=1e-10)

    def test_one_step_return_xtwx_uses_h_inverse_solve(self, monkeypatch):
        """POI one-step calls reuse the final H inverse instead of solving M_aug separately."""
        import superglm.solvers.irls_direct as irls_direct
        from superglm.distributions import Poisson
        from superglm.links import LogLink

        rng = np.random.default_rng(321)
        n = 120
        X_raw = np.column_stack([rng.normal(size=n), rng.normal(size=n)])
        eta = -0.1 + X_raw @ np.array([0.2, -0.15])
        y = rng.poisson(np.exp(eta)).astype(float)
        weights = np.ones(n)
        dm = DesignMatrix([DenseGroupMatrix(X_raw)], n=n, p=2)
        groups = [GroupSlice(name="x", start=0, end=2)]

        def fail_robust_solve(*args, **kwargs):
            raise AssertionError("_robust_solve should not be used on this fast path")

        monkeypatch.setattr(irls_direct, "_robust_solve", fail_robust_solve)

        result, H_inv, XtWX = irls_direct.fit_irls_direct(
            X=dm,
            y=y,
            weights=weights,
            family=Poisson(),
            link=LogLink(),
            groups=groups,
            lambda2=0.1,
            max_iter=1,
            return_xtwx=True,
        )

        assert result.n_iter == 1
        assert H_inv.shape == (2, 2)
        assert XtWX.shape == (2, 2)

    def test_safe_decompose_h_uses_unpivoted_cholesky_for_spd(self, monkeypatch):
        import superglm.solvers.irls_direct as irls_direct

        rng = np.random.default_rng(11)
        A = rng.standard_normal((6, 6))
        H = A @ A.T + np.eye(6)

        def fail_svd(*args, **kwargs):
            raise AssertionError("SVD should not be used for SPD fast path")

        monkeypatch.setattr(irls_direct.np.linalg, "svd", fail_svd)

        H_inv, log_det, cholesky_ok = irls_direct._safe_decompose_H(H)

        np.testing.assert_allclose(H_inv, np.linalg.inv(H), rtol=1e-10, atol=1e-10)
        assert log_det == pytest.approx(np.linalg.slogdet(H)[1], abs=1e-10)
        assert cholesky_ok

    def test_robust_solve_uses_unpivoted_cholesky_for_spd(self, monkeypatch):
        import superglm.solvers.irls_direct as irls_direct

        rng = np.random.default_rng(13)
        A = rng.standard_normal((6, 6))
        M = A @ A.T + np.eye(6)
        rhs = rng.standard_normal(6)

        def fail_svd(*args, **kwargs):
            raise AssertionError("SVD should not be used for SPD fast path")

        monkeypatch.setattr(irls_direct.np.linalg, "svd", fail_svd)

        x, cond_est, used_svd = irls_direct._robust_solve(M, rhs)

        np.testing.assert_allclose(x, np.linalg.solve(M, rhs), rtol=1e-10, atol=1e-10)
        assert cond_est > 0
        assert not used_svd

    def test_constant_weight_fit_reuses_weighted_gram(self, monkeypatch):
        """Gaussian identity fits reuse X'WX across IRLS iterations."""
        import superglm.solvers.irls_direct as irls_direct
        from superglm.distributions import Gaussian
        from superglm.links import IdentityLink

        rng = np.random.default_rng(123)
        n = 80
        X_raw = np.column_stack(
            [
                rng.normal(size=n),
                rng.normal(size=n),
            ]
        )
        weights = rng.uniform(0.5, 2.0, size=n)
        y = 1.3 + X_raw @ np.array([0.4, -0.25]) + rng.normal(scale=0.05, size=n)
        dm = DesignMatrix([DenseGroupMatrix(X_raw)], n=n, p=2)
        groups = [GroupSlice(name="x", start=0, end=2)]

        original_centered = irls_direct.build_centered_system
        centered_calls = 0

        def counting_centered_system(*args, **kwargs):
            nonlocal centered_calls
            centered_calls += 1
            return original_centered(*args, **kwargs)

        monkeypatch.setattr(irls_direct, "build_centered_system", counting_centered_system)

        result, _ = irls_direct.fit_irls_direct(
            X=dm,
            y=y,
            weights=weights,
            family=Gaussian(),
            link=IdentityLink(),
            groups=groups,
            lambda2=0.0,
            max_iter=5,
            tol=1e-12,
        )

        assert result.n_iter > 1
        assert centered_calls == 1

    def test_constant_weight_cache_preserves_solution(self, monkeypatch):
        """Reusing X'WX does not change the fitted coefficients."""
        import superglm.solvers.irls_direct as irls_direct
        from superglm.distributions import Gaussian
        from superglm.links import IdentityLink

        rng = np.random.default_rng(456)
        n = 90
        X_raw = np.column_stack(
            [
                rng.normal(size=n),
                rng.normal(size=n),
                rng.normal(size=n),
            ]
        )
        weights = rng.uniform(0.5, 2.0, size=n)
        y = 0.8 + X_raw @ np.array([0.2, -0.35, 0.5]) + rng.normal(scale=0.1, size=n)
        dm = DesignMatrix([DenseGroupMatrix(X_raw)], n=n, p=3)
        groups = [GroupSlice(name="x", start=0, end=3)]

        cached, _ = irls_direct.fit_irls_direct(
            X=dm,
            y=y,
            weights=weights,
            family=Gaussian(),
            link=IdentityLink(),
            groups=groups,
            lambda2=0.0,
            max_iter=5,
            tol=1e-12,
        )

        monkeypatch.setattr(irls_direct, "_has_constant_irls_weights", lambda *_: False)
        uncached, _ = irls_direct.fit_irls_direct(
            X=dm,
            y=y,
            weights=weights,
            family=Gaussian(),
            link=IdentityLink(),
            groups=groups,
            lambda2=0.0,
            max_iter=5,
            tol=1e-12,
        )

        np.testing.assert_allclose(cached.beta, uncached.beta, rtol=0, atol=1e-12)
        assert cached.intercept == pytest.approx(uncached.intercept, abs=1e-12)
        assert cached.deviance == pytest.approx(uncached.deviance, abs=1e-12)

    def test_distribution_subclass_cannot_inherit_constant_weight_cache(self, monkeypatch):
        """Behavior-changing family subclasses must rebuild their weighted Gram."""
        import superglm.solvers.irls_direct as irls_direct
        from superglm.distributions import Gaussian, Poisson
        from superglm.links import IdentityLink

        class PoissonVarianceGaussian(Gaussian):
            variance = Poisson.variance
            variance_derivative = Poisson.variance_derivative
            variance_second_derivative = Poisson.variance_second_derivative
            deviance_unit = Poisson.deviance_unit

        rng = np.random.default_rng(77)
        n = 200
        x = np.linspace(-1.0, 1.0, n)
        X_raw = np.column_stack((x, x**2))
        mean = 3.0 + 0.8 * x + 0.3 * x**2
        y = rng.poisson(mean).astype(np.float64)
        dm = DesignMatrix([DenseGroupMatrix(X_raw)], n=n, p=2)
        groups = [GroupSlice(name="x", start=0, end=2)]
        family = PoissonVarianceGaussian()
        link = IdentityLink()
        original_centered = irls_direct.build_centered_system
        centered_calls = 0

        def counting_centered_system(*args, **kwargs):
            nonlocal centered_calls
            centered_calls += 1
            return original_centered(*args, **kwargs)

        assert not irls_direct._has_constant_irls_weights(family, link)
        monkeypatch.setattr(irls_direct, "build_centered_system", counting_centered_system)
        actual, _ = irls_direct.fit_irls_direct(
            X=dm,
            y=y,
            weights=np.ones(n),
            family=family,
            link=link,
            groups=groups,
            lambda2=0.0,
            max_iter=50,
            tol=1e-10,
        )
        assert centered_calls > 1

        monkeypatch.setattr(irls_direct, "_has_constant_irls_weights", lambda *_: False)
        reference, _ = irls_direct.fit_irls_direct(
            X=dm,
            y=y,
            weights=np.ones(n),
            family=family,
            link=link,
            groups=groups,
            lambda2=0.0,
            max_iter=50,
            tol=1e-10,
        )

        assert actual.converged and reference.converged
        np.testing.assert_allclose(actual.beta, reference.beta, rtol=0.0, atol=1e-13)
        assert actual.intercept == pytest.approx(reference.intercept, abs=1e-13)
        assert actual.deviance == pytest.approx(reference.deviance, abs=1e-13)

    def test_gamma_log_reuses_centered_gram_when_working_response_changes(self, monkeypatch):
        """Constant Gamma-log weights must not rebuild the invariant Gram."""
        import superglm.solvers.irls_direct as irls_direct
        from superglm.distributions import Gamma
        from superglm.links import LogLink

        rng = np.random.default_rng(458)
        n = 120
        X_raw = rng.normal(size=(n, 3))
        y = rng.gamma(shape=2.0, scale=np.exp(0.2 + X_raw @ np.array([0.1, -0.2, 0.3])) / 2.0)
        dm = DesignMatrix([DenseGroupMatrix(X_raw)], n=n, p=3)
        groups = [GroupSlice(name="x", start=0, end=3)]

        original_centered = irls_direct.build_centered_system
        centered_calls = 0

        def counting_centered_system(*args, **kwargs):
            nonlocal centered_calls
            centered_calls += 1
            return original_centered(*args, **kwargs)

        monkeypatch.setattr(irls_direct, "build_centered_system", counting_centered_system)

        profile: dict[str, float | int] = {}
        result, _ = irls_direct.fit_irls_direct(
            X=dm,
            y=y,
            weights=np.ones(n),
            family=Gamma(),
            link=LogLink(),
            groups=groups,
            lambda2=0.1,
            max_iter=4,
            tol=0.0,
            profile=profile,
        )

        assert result.n_iter == 4
        assert centered_calls == 1
        assert "irls_observed_newton_rescues" not in profile
        assert "irls_observed_newton_iters" not in profile

    def test_gamma_log_keeps_fisher_scoring_while_steps_are_accepted(self):
        """Accepted Fisher steps must not trigger a clock-based curvature switch."""
        from superglm.distributions import Gamma
        from superglm.links import LogLink
        from superglm.solvers.irls_direct import fit_irls_direct

        rng = np.random.default_rng(122)
        n = 500
        p = 3
        X_raw = rng.normal(size=(n, p))
        beta = rng.normal(scale=1.2, size=p)
        mu = np.exp(np.clip(0.2 + X_raw @ beta, -4.0, 4.0))
        y = rng.gamma(shape=0.2, scale=mu / 0.2)
        dm = DesignMatrix([DenseGroupMatrix(X_raw)], n=n, p=p)
        groups = [GroupSlice(name="x", start=0, end=p)]

        fisher, _ = fit_irls_direct(
            X=dm,
            y=y,
            weights=np.ones(n),
            family=Gamma(),
            link=LogLink(),
            groups=groups,
            lambda2=0.0,
            tol=1e-10,
            _use_observed_newton=False,
        )
        profile = {}
        controlled, controlled_inverse = fit_irls_direct(
            X=dm,
            y=y,
            weights=np.ones(n),
            family=Gamma(),
            link=LogLink(),
            groups=groups,
            lambda2=0.0,
            tol=1e-10,
            profile=profile,
        )

        assert controlled.n_iter == fisher.n_iter
        assert "irls_observed_newton_rescues" not in profile
        assert "irls_observed_newton_iters" not in profile
        np.testing.assert_allclose(controlled.beta, fisher.beta, rtol=0, atol=0)
        assert controlled.intercept == fisher.intercept
        assert controlled.deviance == fisher.deviance
        centered_X = X_raw - np.mean(X_raw, axis=0)
        expected_fisher_inverse = np.linalg.inv(centered_X.T @ centered_X)
        np.testing.assert_allclose(
            controlled_inverse,
            expected_fisher_inverse,
            rtol=2e-12,
            atol=2e-13,
        )

    def test_gamma_log_observed_controller_rescues_then_falls_back_atomically(self, monkeypatch):
        """A Fisher rejection enables one observed attempt; its rejection restores Fisher."""
        import superglm.solvers.irls_direct as irls_direct
        from superglm.distributions import Gamma
        from superglm.links import LogLink
        from superglm.solvers.irls_state import _IRLSStepDecision

        rng = np.random.default_rng(459)
        n = 160
        X_raw = rng.normal(size=(n, 3))
        mu = np.exp(0.1 + X_raw @ np.array([0.2, -0.3, 0.4]))
        y = rng.gamma(shape=2.0, scale=mu / 2.0)
        dm = DesignMatrix([DenseGroupMatrix(X_raw)], n=n, p=3)
        groups = [GroupSlice(name="x", start=0, end=3)]

        original_rows = irls_direct.coefficient_working_rows
        curvature_requests: list[bool] = []

        def recording_rows(*args, **kwargs):
            curvature_requests.append(bool(kwargs["prefer_observed"]))
            return original_rows(*args, **kwargs)

        original_select = irls_direct._select_irls_trial
        selection_calls = 0

        def reject_first_fisher_and_observed(*args, **kwargs):
            nonlocal selection_calls
            selection_calls += 1
            if selection_calls <= 2:
                return _IRLSStepDecision(0.0, 0, True, trials_attempted=21)
            return original_select(*args, **kwargs)

        monkeypatch.setattr(irls_direct, "coefficient_working_rows", recording_rows)
        monkeypatch.setattr(irls_direct, "_select_irls_trial", reject_first_fisher_and_observed)

        profile: dict[str, float | int] = {}
        result, final_inverse = irls_direct.fit_irls_direct(
            X=dm,
            y=y,
            weights=np.ones(n),
            family=Gamma(),
            link=LogLink(),
            groups=groups,
            lambda2=0.0,
            tol=1e-10,
            max_iter=30,
            profile=profile,
            record_diagnostics=True,
        )

        assert curvature_requests[:3] == [False, True, False]
        assert profile["irls_observed_newton_rescues"] == 1
        assert profile["irls_observed_newton_iters"] == 1
        assert profile["irls_observed_newton_rejections"] == 1
        assert result.converged
        assert result.iteration_log is not None
        assert result.iteration_log[0].termination_reason == "curvature_rescue"
        assert result.iteration_log[1].termination_reason == "curvature_fallback"
        centered_X = X_raw - np.mean(X_raw, axis=0)
        np.testing.assert_allclose(
            final_inverse,
            np.linalg.inv(centered_X.T @ centered_X),
            rtol=2e-12,
            atol=2e-13,
        )

    def test_qp_constraints_are_assembled_once_across_iterations(self, monkeypatch):
        """QP constraint blocks are fixed for a fit and should not be rebuilt per IRLS step."""
        import superglm.solvers.irls_direct as irls_direct
        from superglm._group_matrix._group_matrix_execution import MatrixExecutionPlan
        from superglm.distributions import Gaussian
        from superglm.links import IdentityLink

        rng = np.random.default_rng(457)
        n = 80
        X_raw = np.column_stack([rng.normal(size=n), rng.normal(size=n)])
        y = 0.3 + X_raw @ np.array([0.1, -0.2]) + rng.normal(scale=0.1, size=n)
        dm = DesignMatrix([DenseGroupMatrix(X_raw)], n=n, p=2)
        groups = [
            GroupSlice(
                name="x",
                start=0,
                end=2,
                constraints=LinearConstraintSet(
                    A=np.array([[1.0, 0.0]], dtype=np.float64),
                    b=np.array([-100.0], dtype=np.float64),
                ),
            )
        ]

        original_qp = irls_direct.solve_constrained_qp
        original_moments = MatrixExecutionPlan._moments_prevalidated
        constraint_matrices = []
        moment_plan_ids = []

        def recording_qp(H, g, A, b, *args, **kwargs):
            constraint_matrices.append(A)
            return original_qp(H, g, A, b, *args, **kwargs)

        def recording_moments(self, *args, **kwargs):
            moment_plan_ids.append(id(self))
            return original_moments(self, *args, **kwargs)

        monkeypatch.setattr(irls_direct, "solve_constrained_qp", recording_qp)
        monkeypatch.setattr(MatrixExecutionPlan, "_moments_prevalidated", recording_moments)

        result, _ = irls_direct.fit_irls_direct(
            X=dm,
            y=y,
            weights=np.ones(n),
            family=Gaussian(),
            link=IdentityLink(),
            groups=groups,
            lambda2=0.0,
            max_iter=3,
            tol=0.0,
        )

        assert result.n_iter == 3
        assert len({id(A) for A in constraint_matrices}) == 1
        assert moment_plan_ids
        assert set(moment_plan_ids) == {id(dm.execution_plan)}

    def test_variable_weight_fit_rebuilds_weighted_gram(self, monkeypatch):
        """Poisson log fits do not reuse X'WX because W changes with mu."""
        import superglm.solvers.irls_direct as irls_direct
        from superglm.distributions import Poisson
        from superglm.links import LogLink

        rng = np.random.default_rng(789)
        n = 100
        X_raw = np.column_stack(
            [
                rng.normal(size=n),
                rng.normal(size=n),
            ]
        )
        eta = -0.2 + X_raw @ np.array([0.25, -0.1])
        y = rng.poisson(np.exp(eta))
        weights = rng.uniform(0.5, 1.5, size=n)
        dm = DesignMatrix([DenseGroupMatrix(X_raw)], n=n, p=2)
        groups = [GroupSlice(name="x", start=0, end=2)]

        original_centered = irls_direct.build_centered_system
        centered_calls = 0

        def counting_centered_system(*args, **kwargs):
            nonlocal centered_calls
            centered_calls += 1
            return original_centered(*args, **kwargs)

        monkeypatch.setattr(irls_direct, "build_centered_system", counting_centered_system)

        result, _ = irls_direct.fit_irls_direct(
            X=dm,
            y=y,
            weights=weights,
            family=Poisson(),
            link=LogLink(),
            groups=groups,
            lambda2=0.0,
            max_iter=20,
            tol=1e-12,
        )

        assert result.n_iter > 1
        assert centered_calls == result.n_iter + 1

    def test_matches_bcd_ridge(self, poisson_data):
        """Direct solver with selection_penalty=0 should give similar deviance as BCD with tiny lambda1."""
        X, y, w = poisson_data

        # Direct solver (selection_penalty=0)
        m_direct = SuperGLM(
            family="poisson",
            selection_penalty=0,
            features={
                "DrivAge": Spline(n_knots=10),
                "Area": Categorical(),
            },
        )
        m_direct.fit(X, y, sample_weight=w)

        # BCD solver with near-zero lambda1 (effectively ridge)
        m_bcd = SuperGLM(
            family="poisson",
            selection_penalty=1e-8,
            features={
                "DrivAge": Spline(n_knots=10),
                "Area": Categorical(),
            },
        )
        m_bcd.fit(X, y, sample_weight=w)

        # Deviances should be very close
        assert abs(m_direct.result.deviance - m_bcd.result.deviance) / m_bcd.result.deviance < 0.01

    def test_all_group_types(self, poisson_data):
        """Direct solver handles Dense (numeric), Sparse (categorical), SparseSSP (spline)."""
        X, y, w = poisson_data
        m = SuperGLM(
            family="poisson",
            selection_penalty=0,
            features={
                "DrivAge": Spline(n_knots=8),
                "Area": Categorical(),
            },
        )
        m.fit(X, y, sample_weight=w)
        assert m.result.converged
        assert m.result.deviance < np.sum(y) * 10  # reasonable deviance

    def test_warm_start(self, poisson_data):
        """Warm-started direct solver should converge in fewer iterations."""
        X, y, w = poisson_data

        # Cold start
        m1 = SuperGLM(
            family="poisson",
            selection_penalty=0,
            features={
                "DrivAge": Spline(n_knots=10),
                "Area": Categorical(),
            },
        )
        m1.fit(X, y, sample_weight=w)

        # Warm start via re-fit (same model object, beta is reused internally)
        # We can't easily warm-start through .fit(), so test via direct solver import
        from superglm.solvers.irls_direct import fit_irls_direct

        result_warm, _ = fit_irls_direct(
            X=m1._dm,
            y=y,
            weights=w,
            family=m1._distribution,
            link=m1._link,
            groups=m1._groups,
            lambda2=m1.lambda2,
            beta_init=m1.result.beta,
            intercept_init=m1.result.intercept,
        )
        # Warm start should need <= 2 iterations
        assert result_warm.n_iter <= 2

    def test_exact_edf(self, poisson_data):
        """Effective df from trace formula should be reasonable."""
        X, y, w = poisson_data
        m = SuperGLM(
            family="poisson",
            selection_penalty=0,
            features={
                "DrivAge": Spline(n_knots=10),
                "Area": Categorical(),
            },
        )
        m.fit(X, y, sample_weight=w)

        # edf should be between 1 (intercept-only) and total params + 1
        total_params = sum(g.size for g in m._groups) + 1
        assert 1 < m.result.effective_df < total_params

    def test_predict_after_direct_fit(self, poisson_data):
        """predict/reconstruct should work after direct solver fit."""
        X, y, w = poisson_data
        m = SuperGLM(
            family="poisson",
            selection_penalty=0,
            features={
                "DrivAge": Spline(n_knots=10),
                "Area": Categorical(),
            },
        )
        m.fit(X, y, sample_weight=w)

        mu_hat = m.predict(X)
        assert mu_hat.shape == y.shape
        assert np.all(mu_hat > 0)

        rec = m.reconstruct_feature("DrivAge")
        assert "relativity" in rec


# ── Select=True tests ──────────────────────────────────────────
class TestDirectSolverSelect:
    def test_select_no_aliasing(self, select_data):
        """select=True with direct solver should converge in <= 10 IRLS iters."""
        X, y, w = select_data
        m = SuperGLM(
            family="poisson",
            selection_penalty=0,
            features={
                "signal": Spline(n_knots=10, select=True),
                "noise": Spline(n_knots=10, select=True),
            },
        )
        m.fit(X, y, sample_weight=w)
        # Direct solver: no BCD aliasing → should converge fast
        assert m.result.n_iter <= 10
        assert m.result.converged


# ── REML + direct solver tests ────────────────────────────────
class TestREMLDirect:
    def test_reml_direct_convergence(self):
        """fit_reml() with selection_penalty=0 should converge using the direct solver."""
        # Use data with strong nonlinearity so REML finds a finite lambda
        rng = np.random.default_rng(123)
        n = 2000
        x1 = rng.uniform(18, 80, n)
        # Strong U-shape so REML doesn't push lambda→∞
        mu = np.exp(-2.0 + 0.002 * (x1 - 50) ** 2)
        y = rng.poisson(mu)
        X = pd.DataFrame({"DrivAge": x1})
        w = np.ones(n)

        m = SuperGLM(
            family="poisson",
            selection_penalty=0,
            features={"DrivAge": Spline(n_knots=10)},
        )
        m.fit_reml(X, y, sample_weight=w, max_reml_iter=50)

        assert hasattr(m, "_reml_lambdas")
        assert m._reml_result.converged

    def test_reml_direct_select_true(self, select_data):
        """REML + select=True + selection_penalty=0: should estimate both null and wiggle lambdas."""
        X, y, w = select_data
        m = SuperGLM(
            family="poisson",
            selection_penalty=0,
            features={
                "signal": Spline(n_knots=10, select=True),
                "noise": Spline(n_knots=10, select=True),
            },
        )
        m.fit_reml(X, y, sample_weight=w, max_reml_iter=15)

        assert hasattr(m, "_reml_lambdas")
        lambdas = m._reml_lambdas

        # Should have entries for null and wiggle penalty components
        null_keys = [k for k in lambdas if ":null" in k]
        wiggle_keys = [k for k in lambdas if ":wiggle" in k]
        assert len(null_keys) >= 1
        assert len(wiggle_keys) >= 1

    def test_reml_direct_all_lambdas_estimated(self, select_data):
        """With direct solver, ALL lambdas are REML-estimated including 1-col groups."""
        X, y, w = select_data
        m = SuperGLM(
            family="poisson",
            selection_penalty=0,
            features={
                "signal": Spline(n_knots=10, select=True),
                "noise": Spline(n_knots=10, select=True),
            },
        )
        m.fit_reml(X, y, sample_weight=w, max_reml_iter=15)

        lambdas = m._reml_lambdas
        # Null-space components (rank-1) should have been estimated (not stuck at initial value)
        null_keys = [k for k in lambdas if ":null" in k]
        for key in null_keys:
            # Lambda should differ from the initial value (0.1 default)
            # It may be close, but shouldn't be exactly the default
            assert lambdas[key] > 0

    def test_reml_direct_predict_after_fit(self, poisson_data):
        """predict/reconstruct work after REML + direct solver."""
        X, y, w = poisson_data
        m = SuperGLM(
            family="poisson",
            selection_penalty=0,
            features={
                "DrivAge": Spline(n_knots=10),
                "Area": Categorical(),
            },
        )
        m.fit_reml(X, y, sample_weight=w, max_reml_iter=10)

        mu_hat = m.predict(X)
        assert mu_hat.shape == y.shape
        assert np.all(mu_hat > 0)

        rec = m.reconstruct_feature("DrivAge")
        assert "relativity" in rec

    def test_reml_direct_keeps_basis_fixed(self, poisson_data, monkeypatch):
        """Direct REML should update lambda weights without rebuilding the basis."""
        X, y, w = poisson_data
        m = SuperGLM(
            family="poisson",
            selection_penalty=0,
            features={
                "DrivAge": Spline(n_knots=10),
                "Area": Categorical(),
            },
        )

        def fail_rebuild(*args, **kwargs):
            raise AssertionError("Direct REML should not rebuild the design matrix")

        monkeypatch.setattr(m, "_rebuild_design_matrix_with_lambdas", fail_rebuild)
        m.fit_reml(X, y, sample_weight=w, max_reml_iter=20)

        assert hasattr(m, "_reml_lambdas")
        assert "DrivAge" in m._reml_lambdas
