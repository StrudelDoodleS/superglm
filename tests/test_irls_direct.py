"""Tests for the direct IRLS solver (no BCD)."""

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from superglm.features.categorical import Categorical
from superglm.features.spline import Spline
from superglm.group_matrix import DenseGroupMatrix, DesignMatrix, SparseGroupMatrix
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


def _force_constant_working_rows(monkeypatch, response, weights=None):
    """Every PIRLS solve sees the same working rows, whatever the state.

    The solve is then a function of ``(W, z)`` alone, so its second iteration
    returns the first one's coefficients bit for bit on any BLAS: a forced
    float64 fixed point.  For Gaussian/identity ``z = y`` is the working
    response ``eta + (y - eta)`` without its rounding.
    """
    from dataclasses import replace

    import superglm.solvers.irls_direct._fit as irls_direct

    real = irls_direct.coefficient_working_rows

    def constant_rows(**kwargs):
        rows = real(**kwargs)
        return replace(
            rows,
            response=np.array(response, dtype=np.float64),
            weights=rows.weights if weights is None else np.array(weights, dtype=np.float64),
        )

    monkeypatch.setattr(irls_direct, "coefficient_working_rows", constant_rows)


def _force_unresolved_score(monkeypatch, *, resolved=False):
    """The capped score reads twice the bar (a fit to rounding), the floor-resolved one is real.

    Returns the list of the real verdicts read at ``resolve_cap = inf``.
    """
    import math
    from dataclasses import replace

    import superglm.solvers.irls_direct._fit as irls_direct

    real = irls_direct.penalized_mode_residual
    floor_verdicts: list[float] = []

    def residual(**kwargs):
        result = real(**kwargs)
        if math.isinf(kwargs["resolve_cap"]):
            floor_verdicts.append(result.ratio())
            return result
        return replace(
            result,
            relative=np.full_like(result.relative, 2.0 * result.bar),
            bar_effective=np.full_like(result.bar_effective, result.bar),
            resolved=resolved,
        )

    monkeypatch.setattr(irls_direct, "penalized_mode_residual", residual)
    return floor_verdicts


# ── Basic solver tests ─────────────────────────────────────────
class TestDirectSolverBasic:
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

        retained, retained_inverse, retained_gram = fit_irls_direct(
            **common, weight_semantics="frequency"
        )
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
            weight_semantics="frequency",
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
            weight_semantics="frequency",
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
            weight_semantics="frequency",
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
            weight_semantics="frequency",
        )

        assert result.reml_hessian_rank == 2  # intercept plus one identifiable slope

    def test_reml_working_system_avoids_rebuilding_one_step_crossproducts(self, monkeypatch):
        import superglm.solvers.irls_direct._fit as irls_direct
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
            weight_semantics="frequency",
        )

        assert build_calls == 1
        assert result.rank_info is None
        # Stats-skipping fits poison the unevaluated statistics so accidental
        # publication fails visibly instead of presenting 0 EDF or unit
        # dispersion as if either had been computed.
        assert np.isnan(result.effective_df)
        assert np.isnan(result.phi)
        assert family.deviance_calls == 1
        assert np.all(np.isfinite(result.beta))
        assert np.all(np.isfinite(inverse))
        assert np.all(np.isfinite(gram))

    def test_scop_candidate_can_skip_unused_generic_reml_decomposition(self, monkeypatch):
        """A private SCOP candidate may omit geometry replaced by its joint Hessian."""
        import superglm.solvers.irls_direct._finalise as irls_finalise
        import superglm.solvers.irls_direct._fit as irls_direct
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

        decomposition_calls = 0

        # A terminal decomposition the candidate fit skips is skipped whether
        # or not the retained fit would have had to certify it against the
        # observation factor.
        def _counting(name):
            original = getattr(irls_finalise, name)

            def counted(matrix, *args, **kwargs):
                nonlocal decomposition_calls
                decomposition_calls += 1
                return original(matrix, *args, **kwargs)

            return counted

        # The data rank's entry point (``rank.decide_data_rank``) counts too.
        for name in ("decompose_gram_if_authoritative", "decide_data_rank"):
            monkeypatch.setattr(irls_finalise, name, _counting(name))
        retained, _, _ = irls_direct.fit_irls_direct(**common, weight_semantics="frequency")
        retained_geometry_calls = decomposition_calls
        assert retained.rank_info is not None

        with pytest.raises(
            ValueError,
            match="requires rank metadata and fit statistics to be disabled",
        ):
            irls_direct.fit_irls_direct(
                **common, _compute_reml_geometry=False, weight_semantics="frequency"
            )

        decomposition_calls = 0
        candidate, omitted_inverse, candidate_gram = irls_direct.fit_irls_direct(
            **common,
            compute_rank_info=False,
            _compute_fit_statistics=False,
            _compute_reml_geometry=False,
            weight_semantics="frequency",
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
        import superglm.solvers.irls_direct._fit as irls_direct
        import superglm.solvers.irls_direct._recorder as irls_recorder
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
        # the fit and its recorder, which takes the extrema
        monkeypatch.setattr(irls_direct, "np", extrema_spy)
        monkeypatch.setattr(irls_recorder, "np", extrema_spy)

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
            weight_semantics="frequency",
        )

        assert stats_calls == 1
        assert extrema_spy.min_calls == 0
        assert extrema_spy.max_calls == 0

    def test_level_two_recorder_keeps_extrema_without_iteration_diagnostics(self):
        import superglm.solvers.irls_direct._fit as irls_direct
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
            weight_semantics="frequency",
        )

        assert len(recorder.rows) == 1
        assert {
            "eta_min_unclipped",
            "eta_max_unclipped",
            "working_eta_min_unclipped",
            "working_eta_max_unclipped",
        } <= recorder.rows[0].keys()

    def test_level_two_recorder_decision_is_frozen_before_iterations(self):
        import superglm.solvers.irls_direct._fit as irls_direct
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
            weight_semantics="frequency",
        )

        assert result.n_iter == 2
        assert len(recorder.rows) == 2
        assert recorder.level_reads == 1

    def test_one_step_return_xtwx_returns_h_inverse_and_gram(self):
        """A one-step fit with ``return_xtwx`` returns the final H inverse and weighted Gram."""
        import superglm.solvers.irls_direct._fit as irls_direct
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
            weight_semantics="frequency",
        )

        assert result.n_iter == 1
        assert H_inv.shape == (2, 2)
        assert XtWX.shape == (2, 2)

    def test_safe_decompose_h_uses_unpivoted_cholesky_for_spd(self, monkeypatch):
        import superglm.solvers.irls_direct._fit as irls_direct

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

    def test_gram_decomposition_uses_cholesky_for_spd(self, monkeypatch):
        import superglm.solvers.irls_direct._fit as irls_direct
        from superglm.solvers.rank import decompose_gram

        rng = np.random.default_rng(13)
        A = rng.standard_normal((6, 6))
        M = A @ A.T + np.eye(6)
        rhs = rng.standard_normal(6)

        def fail_svd(*args, **kwargs):
            raise AssertionError("SVD should not be used for SPD fast path")

        monkeypatch.setattr(irls_direct.np.linalg, "svd", fail_svd)

        decomposition = decompose_gram(M)

        np.testing.assert_allclose(
            decomposition.solve(rhs), np.linalg.solve(M, rhs), rtol=1e-10, atol=1e-10
        )
        assert decomposition.pre_truncation_condition > 0
        assert decomposition.method in {"cholesky", "pivoted_cholesky"}

    def test_constant_weight_fit_reuses_weighted_gram(self, monkeypatch):
        """Gaussian identity fits reuse X'WX across IRLS iterations."""
        import superglm.solvers.irls_direct._fit as irls_direct
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
            weight_semantics="frequency",
        )

        assert result.n_iter > 1
        assert centered_calls == 1

    def test_constant_weight_cache_preserves_solution(self, monkeypatch):
        """Reusing X'WX does not change the fitted coefficients."""
        import superglm.solvers.irls_direct._fit as irls_direct
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
            weight_semantics="frequency",
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
            weight_semantics="frequency",
        )

        np.testing.assert_allclose(cached.beta, uncached.beta, rtol=0, atol=1e-12)
        assert cached.intercept == pytest.approx(uncached.intercept, abs=1e-12)
        assert cached.deviance == pytest.approx(uncached.deviance, abs=1e-12)

    def test_poisson_sqrt_reuses_constant_weighted_gram(self, monkeypatch):
        """Exact Poisson/sqrt Fisher weights reuse X'WX without changing the fit."""
        import superglm.solvers.irls_direct._fit as irls_direct
        from superglm.distributions import Poisson
        from superglm.links import SqrtLink

        rng = np.random.default_rng(20260804)
        n = 160
        x = np.linspace(-1.0, 1.0, n)
        X_raw = np.column_stack((x, x**2))
        offset = 0.12 * np.sin(2.0 * x)
        eta = 1.4 + 0.25 * x - 0.08 * x**2 + offset
        y = rng.poisson(eta**2).astype(np.float64)
        weights = rng.uniform(0.5, 2.0, size=n)
        dm = DesignMatrix([DenseGroupMatrix(X_raw)], n=n, p=2)
        groups = [GroupSlice(name="x", start=0, end=2)]

        original_centered = irls_direct.build_centered_system
        centered_calls = 0

        def counting_centered_system(*args, **kwargs):
            nonlocal centered_calls
            centered_calls += 1
            return original_centered(*args, **kwargs)

        assert irls_direct._has_constant_irls_weights(Poisson(), SqrtLink())
        monkeypatch.setattr(irls_direct, "build_centered_system", counting_centered_system)
        cached, _ = irls_direct.fit_irls_direct(
            X=dm,
            y=y,
            weights=weights,
            family=Poisson(),
            link=SqrtLink(),
            groups=groups,
            lambda2=0.0,
            offset=offset,
            max_iter=20,
            tol=1e-10,
            direct_solve="gram",
            weight_semantics="frequency",
        )
        cached_centered_calls = centered_calls

        monkeypatch.setattr(irls_direct, "_has_constant_irls_weights", lambda *_: False)
        uncached, _ = irls_direct.fit_irls_direct(
            X=dm,
            y=y,
            weights=weights,
            family=Poisson(),
            link=SqrtLink(),
            groups=groups,
            lambda2=0.0,
            offset=offset,
            max_iter=20,
            tol=1e-10,
            direct_solve="gram",
            weight_semantics="frequency",
        )

        assert cached.n_iter > 1
        assert cached_centered_calls == 1
        assert centered_calls > cached_centered_calls + 1
        np.testing.assert_allclose(cached.beta, uncached.beta, rtol=0.0, atol=1e-12)
        assert cached.intercept == pytest.approx(uncached.intercept, abs=1e-12)
        assert cached.deviance == pytest.approx(uncached.deviance, abs=1e-12)

    def test_poisson_sqrt_constant_weight_cache_is_exact_type_gated(self):
        import superglm.solvers.irls_direct._fit as irls_direct
        from superglm.distributions import Poisson
        from superglm.links import SqrtLink

        class DerivedPoisson(Poisson):
            pass

        class DerivedSqrtLink(SqrtLink):
            pass

        assert not irls_direct._has_constant_irls_weights(DerivedPoisson(), SqrtLink())
        assert not irls_direct._has_constant_irls_weights(Poisson(), DerivedSqrtLink())

    def test_sqrt_null_shortcut_skips_only_unused_terminal_working_rows(self, monkeypatch):
        import superglm.solvers.irls_direct._fit as irls_direct
        from superglm.distributions import Poisson
        from superglm.links import SqrtLink

        offset = np.tile([-0.4, -0.2, 0.0, 0.2, 0.4], 12)
        y = np.tile([1.0, 2.0, 4.0, 3.0, 1.0], 12)
        weights = np.linspace(0.5, 1.5, len(y))
        dm = DesignMatrix([], n=len(y), p=0)
        original_rows = irls_direct.coefficient_working_rows
        row_calls = 0

        def counted_rows(*args, **kwargs):
            nonlocal row_calls
            row_calls += 1
            return original_rows(*args, **kwargs)

        monkeypatch.setattr(irls_direct, "coefficient_working_rows", counted_rows)
        shortcut, _ = irls_direct.fit_irls_direct(
            X=dm,
            y=y,
            weights=weights,
            family=Poisson(),
            link=SqrtLink(),
            groups=[],
            lambda2=0.0,
            offset=offset,
            max_iter=100,
            tol=1e-10,
            direct_solve="gram",
            compute_rank_info=False,
            _return_working_system=True,
            _compute_fit_statistics=False,
            _compute_reml_geometry=False,
            _compute_scop_postfit_inference=False,
            weight_semantics="frequency",
        )
        shortcut_row_calls = row_calls

        ordinary, _ = irls_direct.fit_irls_direct(
            X=dm,
            y=y,
            weights=weights,
            family=Poisson(),
            link=SqrtLink(),
            groups=[],
            lambda2=0.0,
            offset=offset,
            max_iter=100,
            tol=1e-10,
            direct_solve="gram",
            compute_rank_info=False,
            _compute_fit_statistics=False,
            _compute_reml_geometry=False,
            _compute_scop_postfit_inference=False,
            weight_semantics="frequency",
        )
        ordinary_row_calls = row_calls - shortcut_row_calls

        assert shortcut_row_calls == shortcut.n_iter
        assert ordinary_row_calls == ordinary.n_iter + 1
        assert shortcut.n_iter == ordinary.n_iter
        assert shortcut.converged == ordinary.converged
        assert shortcut.intercept == ordinary.intercept
        assert shortcut.deviance == ordinary.deviance

    def test_distribution_subclass_cannot_inherit_constant_weight_cache(self, monkeypatch):
        """Behavior-changing family subclasses must rebuild their weighted Gram."""
        import superglm.solvers.irls_direct._fit as irls_direct
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
            weight_semantics="frequency",
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
            weight_semantics="frequency",
        )

        assert actual.converged and reference.converged
        np.testing.assert_allclose(actual.beta, reference.beta, rtol=0.0, atol=1e-13)
        assert actual.intercept == pytest.approx(reference.intercept, abs=1e-13)
        assert actual.deviance == pytest.approx(reference.deviance, abs=1e-13)

    def test_gamma_log_reuses_centered_gram_when_working_response_changes(self, monkeypatch):
        """Constant Gamma-log weights must not rebuild the invariant Gram."""
        import superglm.solvers.irls_direct._fit as irls_direct
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
            weight_semantics="frequency",
        )

        assert result.n_iter == 4
        assert centered_calls == 1
        assert "irls_observed_newton_rescues" not in profile
        assert "irls_observed_newton_iters" not in profile

    def test_gamma_log_mode_score_solve_that_reaches_the_bar_keeps_fisher_and_one_gram(
        self, monkeypatch
    ):
        """A ``mode_score`` Gamma/log solve whose Fisher score reaches the bar never switches.

        The switch to Newton is for a Fisher iteration measured to stall or
        to contract too slowly for its budget.  One that reaches the bar
        keeps Fisher steps, and with them its constant-weight Gram, built
        once.  Mutation check: a rate test that fired on every resolved,
        unconverged score (``rate_exhausts_budget`` returning True) switched
        this solve and rebuilt the Gram on each Newton iteration.
        """
        import superglm.solvers.irls_direct._fit as irls_direct
        from superglm.distributions import Gamma
        from superglm.links import LogLink

        rng = np.random.default_rng(458)
        n = 120
        X_raw = rng.normal(size=(n, 3))
        y = rng.gamma(shape=2.0, scale=np.exp(0.2 + X_raw @ np.array([0.1, -0.2, 0.3])) / 2.0)
        original_centered = irls_direct.build_centered_system
        centered_calls = 0

        def counting_centered_system(*args, **kwargs):
            nonlocal centered_calls
            centered_calls += 1
            return original_centered(*args, **kwargs)

        monkeypatch.setattr(irls_direct, "build_centered_system", counting_centered_system)
        profile: dict = {}
        latch = [False]
        result, _ = irls_direct.fit_irls_direct(
            X=DesignMatrix([DenseGroupMatrix(X_raw)], n=n, p=3),
            y=y,
            weights=np.ones(n),
            family=Gamma(),
            link=LogLink(),
            groups=[GroupSlice(name="x", start=0, end=3)],
            lambda2=0.1,
            convergence="mode_score",
            profile=profile,
            weight_semantics="frequency",
            _newton_latch=latch,
        )
        assert result.converged
        assert result.n_iter > 2
        assert "irls_fisher_newton_switches" not in profile
        assert "irls_observed_newton_iters" not in profile
        assert latch == [False]
        assert centered_calls == 1

    def test_exact_reml_passes_one_newton_latch_to_every_pirls_solve(self, monkeypatch):
        """One exact Gamma/log REML fit shares a single ``_newton_latch`` across its PIRLS solves.

        The latch is what carries a Newton switch from one solve of the fit
        to the next (bootstrap, candidates, line-search trials), so every
        solve the optimizer makes must hold the same list.  Mutation check:
        dropping ``_newton_latch=newton_latch`` at any call site in
        ``reml.direct`` leaves that solve with ``None``.
        """
        import superglm.reml.direct as reml_direct
        from superglm.features.spline import Spline

        real = reml_direct.fit_irls_direct
        latches: list = []

        def spy(*args, **kwargs):
            latches.append(kwargs.get("_newton_latch"))
            return real(*args, **kwargs)

        monkeypatch.setattr(reml_direct, "fit_irls_direct", spy)
        rng = np.random.default_rng(11)
        n = 600
        x = rng.uniform(0.0, 1.0, n)
        level = rng.choice(["a", "b", "c"], n)
        mu = np.exp(0.5 + np.sin(4.0 * x) + 0.3 * (level == "b"))
        y = rng.gamma(2.0, mu / 2.0)
        model = SuperGLM(
            family="gamma",
            link="log",
            features={"x": Spline(n_knots=8), "c": Categorical()},
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit_reml(pd.DataFrame({"x": x, "c": level}), y)
        assert len(latches) >= 3
        assert latches[0] is not None
        assert all(latch is latches[0] for latch in latches)

    def test_gamma_log_switches_to_newton_when_fisher_cannot_reach_the_bar(self):
        """Fisher scoring that contracts too slowly for the budget finishes on Newton steps.

        The response is built so the mode is known (beta = 0.2) and the
        observed information ``y / mu`` is 0.01 on the six high-leverage rows:
        along the slope it is about 0.14 of the Fisher information, so
        Fisher's error map ``1 - F^-1 H`` contracts at about 0.86 an
        iteration and its 100-iteration budget ends short of the bar (before
        the fix: ``max_iter``, not converged).  The resolved score's measured
        rate shows it, and the iterations left take Newton steps.
        """
        from superglm.distributions import Gamma
        from superglm.links import LogLink
        from superglm.solvers.irls_direct import fit_irls_direct

        n, far = 200, 6
        x = np.zeros(n)
        x[:far] = 8.0
        x[far:] = np.linspace(-1.0, 1.0, n - far)
        design = np.column_stack([np.ones(n), x])
        ratio = np.ones(n)
        ratio[:far] = 0.01
        # the near rows take the correction that makes the mode exact
        correction = np.linalg.lstsq(
            design[far:].T, -design[:far].T @ (ratio[:far] - 1.0), rcond=None
        )[0]
        ratio[far:] = 1.0 + correction
        assert np.all(ratio > 0.0)
        y = ratio * np.exp(1.0 + 0.2 * x)
        profile: dict = {}
        result, _ = fit_irls_direct(
            X=DesignMatrix([DenseGroupMatrix(x[:, None])], n=n, p=1),
            y=y,
            weights=np.ones(n),
            family=Gamma(),
            link=LogLink(),
            groups=[GroupSlice(name="x", start=0, end=1)],
            lambda2=0.0,
            convergence="mode_score",
            profile=profile,
            weight_semantics="frequency",
        )
        assert result.converged
        assert profile["irls_fisher_newton_switches"] == 1
        np.testing.assert_allclose(result.beta, [0.2], rtol=1e-7)

        # One REML fit's later solves start where the switch left them: a set
        # latch begins on Newton steps, and a switching solve sets it.
        def solve(latch, profile):
            return fit_irls_direct(
                X=DesignMatrix([DenseGroupMatrix(x[:, None])], n=n, p=1),
                y=y,
                weights=np.ones(n),
                family=Gamma(),
                link=LogLink(),
                groups=[GroupSlice(name="x", start=0, end=1)],
                lambda2=0.0,
                convergence="mode_score",
                profile=profile,
                weight_semantics="frequency",
                _newton_latch=latch,
            )[0]

        latch = [False]
        solve(latch, {})
        assert latch == [True]
        latched: dict = {}
        started = solve(latch, latched)
        assert started.converged
        assert "irls_fisher_newton_switches" not in latched
        assert latched["irls_observed_newton_iters"] == started.n_iter < result.n_iter

    def test_a_switch_on_a_stagnated_score_takes_newton_steps_judged_afresh(self, monkeypatch):
        """The Gamma/log switch that stagnation triggers goes on to Newton steps, on a new history.

        The fixture is the slow-Fisher one above.  Until the switch the capped
        score is forced to twice the bar, resolved, and the rate test never
        fires, so Fisher's score stagnates after the window and the switch is
        the stagnation's; from the switch on the score is the real one.  The
        stagnation that triggered the switch must not end the solve, and the
        Newton iterates' ratios must not join Fisher's flat history: a Newton
        iterate is resolved and above the bar before the last, and a history
        still holding Fisher's ratios would call it stagnated.  Mutation
        checks: keeping ``score_stagnated`` ended the solve at the switch, and
        keeping the ratios at that Newton iterate, neither converged.
        """
        import math
        from dataclasses import replace

        import superglm.solvers.irls_direct._convergence as irls_direct_convergence
        import superglm.solvers.irls_direct._fit as irls_direct
        from superglm.distributions import Gamma
        from superglm.links import LogLink

        n, far = 200, 6
        x = np.zeros(n)
        x[:far] = 8.0
        x[far:] = np.linspace(-1.0, 1.0, n - far)
        design = np.column_stack([np.ones(n), x])
        ratio = np.ones(n)
        ratio[:far] = 0.01
        correction = np.linalg.lstsq(
            design[far:].T, -design[:far].T @ (ratio[:far] - 1.0), rcond=None
        )[0]
        ratio[far:] = 1.0 + correction
        y = ratio * np.exp(1.0 + 0.2 * x)

        real_residual = irls_direct.penalized_mode_residual
        real_switch = irls_direct._IRLSDirectCurvaturePolicy.switch_to_newton
        switched: list[bool] = []
        newton_scores: list[tuple[float, bool]] = []

        def residual(**kwargs):
            result = real_residual(**kwargs)
            capped = math.isfinite(kwargs["resolve_cap"])
            if switched:
                if capped:
                    newton_scores.append((result.ratio(), result.resolved))
                return result
            if not capped:
                return result
            return replace(
                result,
                relative=np.full_like(result.relative, 2.0 * result.bar),
                bar_effective=np.full_like(result.bar_effective, result.bar),
                resolved=True,
            )

        def switch(self, recorder):
            switched.append(True)
            real_switch(self, recorder)

        monkeypatch.setattr(irls_direct, "penalized_mode_residual", residual)
        monkeypatch.setattr(irls_direct._IRLSDirectCurvaturePolicy, "switch_to_newton", switch)
        monkeypatch.setattr(irls_direct_convergence, "rate_exhausts_budget", lambda *_args: False)
        profile: dict = {}
        result, _ = irls_direct.fit_irls_direct(
            X=DesignMatrix([DenseGroupMatrix(x[:, None])], n=n, p=1),
            y=y,
            weights=np.ones(n),
            family=Gamma(),
            link=LogLink(),
            groups=[GroupSlice(name="x", start=0, end=1)],
            lambda2=0.0,
            convergence="mode_score",
            profile=profile,
            weight_semantics="frequency",
        )
        assert profile["irls_fisher_newton_switches"] == 1
        assert result.converged
        assert result.termination_reason == "converged"
        assert profile["irls_observed_newton_iters"] == len(newton_scores)
        # what made the fresh history matter: an unconverged, resolved Newton score
        assert any(resolved and score > 1.0 for score, resolved in newton_scores[:-1])

    def test_inverse_link_line_search_keeps_every_row_inside_the_domain(self):
        """A Gamma/inverse step that would carry a row to eta <= 0 is halved, not clipped.

        ``stabilize_eta`` would turn a crossing row into a mean of 1e12 whose
        deviance grows only like ``log mu``, so the line search accepted it
        and the row stayed pinned there (before the fix: two rows at
        eta < 0, ``step_rejected``).  Every accepted state now keeps
        ``eta > 0`` on every row, and the fit converges.
        """
        from superglm.distributions import Gamma
        from superglm.links import InverseLink
        from superglm.solvers.irls_direct import fit_irls_direct

        rng = np.random.default_rng(0)
        n = 300
        x = rng.uniform(0.0, 1.0, n)
        y = rng.gamma(0.3, (1.0 / (0.05 + 2.0 * x)) / 0.3)
        columns = np.column_stack([x, x**2])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result, _ = fit_irls_direct(
                X=DesignMatrix([DenseGroupMatrix(columns)], n=n, p=2),
                y=y,
                weights=np.ones(n),
                family=Gamma(),
                link=InverseLink(),
                groups=[GroupSlice(name="x", start=0, end=2)],
                lambda2=0.0,
                convergence="mode_score",
                weight_semantics="frequency",
            )
        eta = result.intercept + columns @ result.beta
        assert np.all(eta > 0.0)
        assert result.converged

    def test_the_fixed_point_stop_reads_the_centred_intercept(self, monkeypatch):
        """A step that moves ``alpha`` but not the raw intercept's reading is not a fixed point.

        A column at 1e8 puts ``|c' beta|`` near 5e7, where the raw intercept
        ``alpha - c' beta`` has a spacing of 7.5e-9, so a start whose
        ``alpha`` is 1e-9 off the solve's own reads the same raw intercept.
        The solve is forced to one answer and the score above its bar, so
        the first iteration moves ``alpha`` by that 1e-9 and leaves ``beta``
        and the raw intercept bitwise unchanged; only the second repeats
        itself.  Mutation check: comparing the raw intercept alone stopped
        on the first iteration.
        """
        import math

        from superglm.distributions import Gaussian
        from superglm.links import IdentityLink
        from superglm.solvers.irls_direct import fit_irls_direct
        from superglm.solvers.mode_score import centred_warm_start

        n = 64
        x = 1e8 + np.random.default_rng(1).uniform(0.0, 1.0, n)
        y = 2.0 + 0.5 * (x - 1e8)
        _force_constant_working_rows(monkeypatch, y)
        floor_verdicts = _force_unresolved_score(monkeypatch)

        def solve(**start):
            return fit_irls_direct(
                X=DesignMatrix([DenseGroupMatrix(x[:, None])], n=n, p=1),
                y=y,
                weights=np.ones(n),
                family=Gaussian(),
                link=IdentityLink(),
                groups=[GroupSlice(name="x", start=0, end=1)],
                lambda2=0.0,
                convergence="mode_score",
                weight_semantics="frequency",
                **start,
            )[0]

        cold = solve()
        assert cold.converged and cold.n_iter == 2
        alpha, centre = centred_warm_start(cold)
        moved = alpha + 1e-9
        assert moved != alpha
        assert moved - math.fsum(centre * cold.beta) == cold.intercept
        warm = solve(
            beta_init=cold.beta.copy(),
            intercept_init=cold.intercept,
            _centred_init=(moved, centre),
        )
        assert warm.centred_intercept == alpha
        assert warm.n_iter == 2
        assert warm.converged
        assert floor_verdicts and max(floor_verdicts) <= 1.0

    def test_a_newton_switch_and_a_fixed_point_in_one_iteration_take_the_newton_step(
        self, monkeypatch
    ):
        """The iteration that switches Gamma/log to Newton is not a fixed point of the new map.

        The working rows are forced constant, so the second iteration repeats
        the first bit for bit, and the rate test is forced to fire on that
        same iteration.  The switch clears the score history; the fixed-point
        stop then wrote its last entry (before the fix: ``IndexError``), and
        would have ended the solve before the Newton map it chose ran.  The
        solve now takes the Newton iteration and stops at its fixed point.
        """
        import superglm.solvers.irls_direct._convergence as irls_direct_convergence
        from superglm.distributions import Gamma
        from superglm.links import LogLink
        from superglm.solvers.irls_direct import fit_irls_direct

        rng = np.random.default_rng(5)
        n = 120
        x = np.linspace(-1.0, 1.0, n)
        y = np.exp(0.5 + 0.3 * x) * rng.gamma(50.0, 1.0 / 50.0, n)
        _force_constant_working_rows(monkeypatch, np.log(y), weights=np.ones(n))
        floor_verdicts = _force_unresolved_score(monkeypatch, resolved=True)
        monkeypatch.setattr(
            irls_direct_convergence,
            "rate_exhausts_budget",
            lambda ratios, window, done, max_iter: done >= 2,
        )
        profile: dict = {}
        result, _ = fit_irls_direct(
            X=DesignMatrix([DenseGroupMatrix(x[:, None])], n=n, p=1),
            y=y,
            weights=np.ones(n),
            family=Gamma(),
            link=LogLink(),
            groups=[GroupSlice(name="x", start=0, end=1)],
            lambda2=0.0,
            convergence="mode_score",
            profile=profile,
            weight_semantics="frequency",
        )
        assert profile["irls_fisher_newton_switches"] == 1
        assert profile["irls_observed_newton_iters"] == 1
        assert result.n_iter == 3
        assert result.converged
        assert floor_verdicts and max(floor_verdicts) <= 1.0

    @pytest.mark.parametrize("convergence", ["deviance", "coefficients"])
    def test_a_rejected_step_is_never_recorded_as_converged(self, monkeypatch, convergence):
        """The iteration that rejects its step records no stop, though its criterion reads zero.

        Every trial is refused, so the retained state is the committed one and
        the criterion reads a change of exactly zero, below any tolerance.  A
        rejected step bounds nothing, so the stop rule vetoes the stop
        (``_IRLSDirectConvergence._criterion``): the iteration's published
        verdict is not converged, and its reason is ``step_rejected``.
        Mutation check: without the veto the trace recorded the rejected
        iteration as converged.
        """
        import superglm.solvers.irls_direct._search as irls_direct_search
        from superglm._fit_trace import MemoryTraceSink, TraceRun
        from superglm.distributions import Gaussian
        from superglm.links import IdentityLink
        from superglm.solvers.irls_direct import fit_irls_direct
        from superglm.solvers.irls_state import _IRLSStepDecision

        def reject_every_trial(**_kwargs):
            return _IRLSStepDecision(
                alpha=0.0, step_halvings=20, step_rejected=True, trials_attempted=20
            )

        monkeypatch.setattr(irls_direct_search, "_select_irls_trial", reject_every_trial)
        rng = np.random.default_rng(11)
        n = 60
        x = rng.uniform(-1.0, 1.0, (n, 2))
        y = 0.5 + x @ np.array([0.7, -0.4]) + 0.1 * rng.normal(size=n)
        sink = MemoryTraceSink()
        result, _ = fit_irls_direct(
            DesignMatrix([DenseGroupMatrix(x)], n=n, p=2),
            y,
            np.ones(n),
            Gaussian(),
            IdentityLink(),
            [GroupSlice("x", 0, 2)],
            lambda2=0.1,
            convergence=convergence,
            weight_semantics="frequency",
            trace_run=TraceRun("rejected-step", sink=sink),
        )
        assert not result.converged
        assert result.termination_reason == "step_rejected"
        (decision,) = [e.payload for e in sink.events if e.event_kind == "step_decision"]
        assert decision["step_rejected"]
        assert decision["convergence_value"] == 0.0
        assert not decision["fit_converged"]
        assert decision["termination_reason"] == "step_rejected"

    def test_a_zero_crossing_proposal_takes_the_first_halving_inside_the_domain(self):
        """The line search accepts the first halving that keeps every weighted row at eta > 0.

        The committed state has eta = 1 on every row and the proposal
        ``1 - 3x`` crosses zero for x > 1/3.  The merit is made to accept
        every trial, so only the domain test decides: without it the
        crossing full step is taken; with it alpha = 1/2 (``1 - 1.5x``, still
        crossing for x > 2/3) is refused and alpha = 1/4 accepted.  A row
        with zero weight may sit outside.  Only the two inverse links
        declare the domain.
        """
        from superglm.distributions import Gamma
        from superglm.links import (
            CauchitLink,
            CloglogLink,
            IdentityLink,
            InverseLink,
            InverseSquaredLink,
            LogitLink,
            LogLink,
            NegativeBinomialLink,
            ProbitLink,
            SqrtLink,
        )
        from superglm.solvers.irls_state import (
            _evaluate_irls_state,
            _select_irls_trial,
            link_domain_violation,
        )

        n = 41
        x = np.linspace(0.0, 1.0, n)
        dm = DesignMatrix([DenseGroupMatrix(x[:, None])], n=n, p=1)
        y = np.full(n, 0.8)
        weights = np.ones(n)
        weights[-1] = 0.0
        offset = np.zeros(n)
        offset[-1] = -5.0  # a zero-weight row outside the domain throughout

        def state(slope: float):
            return _evaluate_irls_state(
                dm, y, weights, Gamma(), InverseLink(), offset, np.array([slope]), 1.0
            )

        committed, proposal = state(0.0), state(-3.0)
        outside = link_domain_violation(InverseLink())
        assert outside is not None
        assert not outside(committed.eta_unclipped, weights)
        assert outside(proposal.eta_unclipped, weights)
        evaluated: list[float] = []

        def evaluate(alpha: float):
            evaluated.append(alpha)
            return state(-3.0 * alpha)

        def select(invalid_state):
            evaluated.clear()
            return _select_irls_trial(
                committed=committed,
                proposal=proposal,
                evaluate_state=evaluate,
                invalid_state=invalid_state,
                merit_delta=lambda candidate, base: -1.0,
            )

        assert select(None).alpha == 1.0
        decision = select(lambda candidate: outside(candidate.eta_unclipped, weights))
        assert decision.alpha == 0.25 and decision.step_halvings == 2
        assert not decision.step_rejected
        assert evaluated == [0.5, 0.25]
        assert outside(state(-1.5).eta_unclipped, weights)
        assert not outside(state(-0.75).eta_unclipped, weights)

        assert link_domain_violation(InverseSquaredLink()) is not None
        for link in (
            LogLink(),
            IdentityLink(),
            LogitLink(),
            ProbitLink(),
            CloglogLink(),
            CauchitLink(),
            SqrtLink(),
            NegativeBinomialLink(theta=1.0),
        ):
            assert link_domain_violation(link) is None

    def test_inverse_link_default_start_is_raised_inside_the_domain(self):
        """A negative offset that puts the default start at eta <= 0 has its intercept raised.

        The default intercept ``1 / mean(y)`` is chosen before the offset, so
        an offset of -10 on 30% of the rows starts them at eta < 0, outside
        the inverse link's domain, where the domain guard is not installed:
        the clipped rows (mean 1e12) are accepted and the state crawls back
        from the clip (before the fix: ``max_iter`` at 20 iterations, every
        such row at eta ~ 1e-6; 45 iterations to converge without a cap).
        The start is now raised until the smallest eta is the default
        intercept, as the log-binomial start is lowered.
        """
        from superglm.distributions import Gamma
        from superglm.links import InverseLink, LogLink
        from superglm.solvers.irls_direct import fit_irls_direct
        from superglm.solvers.irls_state import interior_start_intercept

        rng = np.random.default_rng(0)
        n = 300
        x = rng.uniform(0.0, 1.0, n)
        offset = np.where(rng.uniform(size=n) < 0.3, -10.0, 0.0)
        y = rng.gamma(0.5, (1.0 / (10.5 + x + offset)) / 0.5)
        start = 1.0 / float(np.mean(y))
        assert np.any(start + offset <= 0.0)
        raised = interior_start_intercept(
            Gamma(), InverseLink(), start + offset, np.ones(n), start, level=start
        )
        assert raised == start + 10.0
        # a start already inside, and a link without the domain, are left alone
        assert (
            interior_start_intercept(
                Gamma(), InverseLink(), start + 0 * offset, np.ones(n), start, level=start
            )
            == start
        )
        assert (
            interior_start_intercept(
                Gamma(), LogLink(), start + offset, np.ones(n), start, level=start
            )
            == start
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result, _ = fit_irls_direct(
                X=DesignMatrix([DenseGroupMatrix(x[:, None])], n=n, p=1),
                y=y,
                weights=np.ones(n),
                family=Gamma(),
                link=InverseLink(),
                groups=[GroupSlice(name="x", start=0, end=1)],
                lambda2=0.0,
                offset=offset,
                max_iter=20,
                convergence="mode_score",
                weight_semantics="frequency",
            )
        assert result.converged
        assert np.all(result.intercept + x * result.beta[0] + offset > 0.0)

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
            weight_semantics="frequency",
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
            weight_semantics="frequency",
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

    def test_tweedie_log_takes_observed_newton_steps_from_the_first_iteration(self):
        """Tweedie/log PIRLS is full Newton throughout: quadratic, not Fisher's linear rate."""
        from superglm._tweedie import generate_tweedie_cpg
        from superglm.distributions import Tweedie
        from superglm.links import LogLink
        from superglm.solvers.irls_direct import fit_irls_direct

        rng = np.random.default_rng(1818)
        n = 4000
        X_raw = rng.normal(size=(n, 3))
        mu = np.exp(0.3 + X_raw @ np.array([0.4, -0.3, 0.2]))
        # p near 2 has many small positive responses and the slowest Fisher
        # contraction: observed/Fisher row curvature is (2-p) + (p-1) y/mu.
        y = generate_tweedie_cpg(n, mu, 1.8, 1.8, rng=rng)
        dm = DesignMatrix([DenseGroupMatrix(X_raw)], n=n, p=3)
        groups = [GroupSlice(name="x", start=0, end=3)]

        def fit(use_observed_newton: bool):
            profile: dict[str, float | int] = {}
            result, _ = fit_irls_direct(
                X=dm,
                y=y,
                weights=np.ones(n),
                family=Tweedie(1.8),
                link=LogLink(),
                groups=groups,
                lambda2=0.0,
                tol=1e-10,
                convergence="coefficients",
                profile=profile,
                _use_observed_newton=use_observed_newton,
                weight_semantics="frequency",
            )
            return result, profile

        newton, newton_profile = fit(True)
        fisher, fisher_profile = fit(False)

        assert newton.converged and fisher.converged
        # Every iteration, the first included, used the observed rows.
        assert newton_profile["irls_observed_newton_iters"] == newton.n_iter
        assert "irls_observed_newton_iters" not in fisher_profile
        assert newton.n_iter < fisher.n_iter

        # Both stop on a last step s with |s_i| <= tol max(1, |b_i|). Near the mode
        # Fisher scoring maps the error by G = I - F^-1 H (F, H the Fisher and
        # observed information), so the error left after s is -(H^-1 F - I) s;
        # Newton's is second order in s. Each fixed point also rounds by F^-1
        # times the rounding of A'Wz, eps log2(n) sum |A| W |z|.
        design = np.column_stack([np.ones(n), X_raw])
        coefficients = np.r_[newton.intercept, newton.beta]
        mu_hat = np.exp(design @ coefficients)
        fisher_w = mu_hat ** (2.0 - 1.8)
        observed_w = mu_hat ** (1.0 - 1.8) * ((2.0 - 1.8) * mu_hat + 0.8 * y)
        fisher_info = design.T @ (fisher_w[:, None] * design)
        observed_info = design.T @ (observed_w[:, None] * design)
        contraction = np.linalg.solve(observed_info, fisher_info) - np.eye(4)
        step_bound = 1e-10 * np.abs(contraction) @ np.maximum(1.0, np.abs(coefficients))
        working = np.log(mu_hat) + (y - mu_hat) / mu_hat
        eps = np.finfo(np.float64).eps
        rounded = eps * np.log2(n) * np.abs(design).T @ np.abs(fisher_w * working)
        round_off = 2.0 * np.abs(np.linalg.inv(fisher_info)) @ rounded
        gap = np.abs(coefficients - np.r_[fisher.intercept, fisher.beta])
        np.testing.assert_array_less(gap, step_bound + round_off)

    @pytest.mark.parametrize(
        ("family_name", "declared"), [("gamma", False), ("tweedie", True)], ids=["gamma", "tweedie"]
    )
    def test_the_curvature_is_the_pairs_declaration_and_never_switches(
        self, monkeypatch, family_name, declared
    ):
        """One-engine design §3.11: Gamma/log takes Fisher rows and Tweedie/log
        observed rows on every iteration; a rejected step ends the fit in the
        same curvature, with no observed rescue and no Fisher latch."""
        import superglm.solvers.irls_direct._fit as irls_direct
        import superglm.solvers.irls_direct._search as irls_direct_search
        from superglm.distributions import Gamma, Tweedie
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

        original_select = irls_direct_search._select_irls_trial
        selection_calls = 0

        def reject_the_second_proposal(*args, **kwargs):
            nonlocal selection_calls
            selection_calls += 1
            if selection_calls == 2:
                return _IRLSStepDecision(0.0, 0, True, trials_attempted=21)
            return original_select(*args, **kwargs)

        monkeypatch.setattr(irls_direct, "coefficient_working_rows", recording_rows)
        monkeypatch.setattr(irls_direct_search, "_select_irls_trial", reject_the_second_proposal)

        profile: dict[str, float | int] = {}
        result, _ = irls_direct.fit_irls_direct(
            X=dm,
            y=y,
            weights=np.ones(n),
            family=Gamma() if family_name == "gamma" else Tweedie(p=1.5),
            link=LogLink(),
            groups=groups,
            lambda2=0.0,
            tol=1e-10,
            max_iter=30,
            profile=profile,
            record_diagnostics=True,
            weight_semantics="frequency",
        )

        # Gamma/log never asks for observed rows (no rescue); Tweedie/log takes
        # them on both iterations the loop runs (no Fisher latch)
        assert (True in curvature_requests) is declared
        assert profile.get("irls_observed_newton_iters", 0) == (2 if declared else 0)
        assert not any(key.startswith("irls_observed_newton_re") for key in profile)
        assert "irls_observed_newton_fallbacks" not in profile
        assert result.iteration_log is not None
        assert [row.termination_reason for row in result.iteration_log] == [
            "continue",
            "step_rejected",
        ]

    @staticmethod
    def _tweedie_log_problem():
        rng = np.random.default_rng(459)
        n = 160
        X_raw = rng.normal(size=(n, 3))
        mu = np.exp(0.1 + X_raw @ np.array([0.2, -0.3, 0.4]))
        y = rng.gamma(shape=2.0, scale=mu / 2.0)
        dm = DesignMatrix([DenseGroupMatrix(X_raw)], n=n, p=3)
        return dm, y, [GroupSlice(name="x", start=0, end=3)]

    def test_a_trial_without_finite_observed_rows_is_rejected_by_the_line_search(self, monkeypatch):
        """One-engine design §3.11: finiteness is the one condition observed rows
        keep, so a full step to a state whose rows the declared curvature cannot
        form is halved, never accepted and never answered with Fisher rows."""
        import superglm.solvers.irls_direct._fit as irls_direct
        from superglm.distributions import Tweedie
        from superglm.links import LogLink
        from superglm.solvers.working_rows import CoefficientWorkingRows

        dm, y, groups = self._tweedie_log_problem()
        original_rows = irls_direct.coefficient_working_rows
        committed_eta: list[np.ndarray] = []

        def rows(*args, **kwargs):
            result = original_rows(*args, **kwargs)
            if kwargs["prefer_observed"] and not committed_eta:
                committed_eta.append(np.array(kwargs["eta"]))
                return result
            if kwargs["prefer_observed"] and not np.array_equal(kwargs["eta"], committed_eta[0]):
                # every state other than the start is declared unrepresentable
                weights = np.array(result.weights)
                weights[0] = np.inf
                return CoefficientWorkingRows(
                    weights, result.response, "observed", "nonfinite_observed_rows"
                )
            return result

        monkeypatch.setattr(irls_direct, "coefficient_working_rows", rows)
        result, _ = irls_direct.fit_irls_direct(
            X=dm,
            y=y,
            weights=np.ones(dm.n),
            family=Tweedie(p=1.5),
            link=LogLink(),
            groups=groups,
            lambda2=0.0,
            max_iter=5,
            record_diagnostics=True,
            weight_semantics="frequency",
        )
        first = result.iteration_log[0]
        assert first.step_rejected
        assert first.trials_attempted > 1
        assert first.termination_reason == "step_rejected"

    def test_non_finite_observed_rows_at_the_start_raise(self, monkeypatch):
        """No Fisher substitute: a declared observed pair whose rows are not finite
        at the starting coefficients has no step, and says so."""
        import superglm.solvers.irls_direct._fit as irls_direct
        from superglm.distributions import Tweedie
        from superglm.links import LogLink
        from superglm.solvers.working_rows import CoefficientWorkingRows

        dm, y, groups = self._tweedie_log_problem()
        original_rows = irls_direct.coefficient_working_rows

        def rows(*args, **kwargs):
            result = original_rows(*args, **kwargs)
            if kwargs["prefer_observed"]:
                return CoefficientWorkingRows(
                    result.weights, result.response, "observed", "nonfinite_observed_rows"
                )
            return result

        monkeypatch.setattr(irls_direct, "coefficient_working_rows", rows)
        with pytest.raises(ValueError, match="observed-Newton rows .* not finite"):
            irls_direct.fit_irls_direct(
                X=dm,
                y=y,
                weights=np.ones(dm.n),
                family=Tweedie(p=1.5),
                link=LogLink(),
                groups=groups,
                lambda2=0.0,
                weight_semantics="frequency",
            )

    def test_qp_constraints_are_assembled_once_across_iterations(self, monkeypatch):
        """Large constrained fits dispatch raw moments through their cached plan."""
        import superglm.solvers.irls_direct._fit as irls_direct
        import superglm.solvers.irls_direct._qp as irls_direct_qp
        from superglm._group_matrix._group_matrix_execution import MatrixExecutionPlan
        from superglm.distributions import Gaussian
        from superglm.links import IdentityLink

        rng = np.random.default_rng(457)
        n = 50_000
        X_raw = np.column_stack([rng.normal(size=n), rng.normal(size=n)])
        y = 0.3 + X_raw @ np.array([0.1, -0.2]) + rng.normal(scale=0.1, size=n)
        # a type the raw-moment rung admits: a dense column never takes it
        dm = DesignMatrix([SparseGroupMatrix(sparse.csr_matrix(X_raw))], n=n, p=2)
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

        original_qp = irls_direct_qp.solve_constrained_qp
        original_moments = MatrixExecutionPlan._moments_prevalidated
        constraint_matrices = []
        moment_plan_ids = []

        def recording_qp(H, g, A, b, *args, **kwargs):
            constraint_matrices.append(A)
            return original_qp(H, g, A, b, *args, **kwargs)

        def recording_moments(self, *args, **kwargs):
            moment_plan_ids.append(id(self))
            return original_moments(self, *args, **kwargs)

        monkeypatch.setattr(irls_direct_qp, "solve_constrained_qp", recording_qp)
        monkeypatch.setattr(
            MatrixExecutionPlan,
            "_moments_prevalidated",
            recording_moments,
        )

        profile = {}
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
            profile=profile,
            weight_semantics="frequency",
        )

        assert dm.n * dm.p == 100_000
        assert result.n_iter == 3
        assert len({id(A) for A in constraint_matrices}) == 1
        assert moment_plan_ids == [id(dm.execution_plan)]
        assert profile["centered_raw_moment_hits"] == 1

        # Compare the dispatched path with the stable chunked baseline. The
        # Gaussian working Gram is fit-invariant, so both paths build it once.
        original_centered = irls_direct.build_centered_system

        def force_chunked_centering(*args, **kwargs):
            kwargs["_force_chunked"] = True
            return original_centered(*args, **kwargs)

        monkeypatch.setattr(irls_direct_qp, "solve_constrained_qp", original_qp)
        monkeypatch.setattr(
            irls_direct,
            "build_centered_system",
            force_chunked_centering,
        )
        baseline_profile = {}
        baseline, _ = irls_direct.fit_irls_direct(
            X=dm,
            y=y,
            weights=np.ones(n),
            family=Gaussian(),
            link=IdentityLink(),
            groups=groups,
            lambda2=0.0,
            max_iter=3,
            tol=0.0,
            profile=baseline_profile,
            weight_semantics="frequency",
        )

        assert moment_plan_ids == [id(dm.execution_plan)]
        assert baseline_profile.get("centered_raw_moment_hits", 0) == 0
        np.testing.assert_allclose(result.beta, baseline.beta, rtol=2e-12, atol=2e-13)
        assert result.intercept == pytest.approx(baseline.intercept, rel=2e-12, abs=2e-13)
        assert result.deviance == pytest.approx(baseline.deviance, rel=2e-12, abs=2e-13)

    def test_variable_weight_fit_rebuilds_weighted_gram(self, monkeypatch):
        """Poisson log fits do not reuse X'WX because W changes with mu."""
        import superglm.solvers.irls_direct._fit as irls_direct
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
            weight_semantics="frequency",
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
            weight_semantics="frequency",
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

    def test_an_edf_below_the_traces_resolution_is_published_as_zero(self, monkeypatch):
        """A coefficient whose penalty swamps its data publishes an EDF of exactly zero.

        A ridge of 1e20 on one of two columns gives that coefficient an EDF of
        about ``D_bb / 1e20``, near 1e-19: positive, but far below the rounding
        of a trace whose other entry is of order one.  The finaliser publishes
        every EDF within ``100 eps`` of zero as zero, so the coefficient, its
        group and the effective degrees of freedom read no data in it.
        Mutation check: without the cut the rank metadata published the 1e-19.
        """
        import superglm.solvers.irls_direct._finalise as irls_direct_finalise
        from superglm.distributions import Gaussian
        from superglm.links import IdentityLink
        from superglm.solvers.irls_direct import fit_irls_direct
        from superglm.types import PenaltyComponent

        traces = []
        real_edf = irls_direct_finalise._feature_edf

        def recorded(*args, **kwargs):
            edf = real_edf(*args, **kwargs)
            traces.append(edf.copy())
            return edf

        monkeypatch.setattr(irls_direct_finalise, "_feature_edf", recorded)
        rng = np.random.default_rng(3)
        n = 50
        a, b = rng.uniform(-1.0, 1.0, n), rng.uniform(-1.0, 1.0, n)
        y = 0.5 + a + 0.3 * b + 0.1 * rng.normal(size=n)
        ridge = PenaltyComponent(
            name="b",
            group_name="b",
            group_index=1,
            group_sl=slice(1, 2),
            omega_raw=None,
            penalty_kind="identity",
        )
        result, _ = fit_irls_direct(
            DesignMatrix([DenseGroupMatrix(a[:, None]), DenseGroupMatrix(b[:, None])], n=n, p=2),
            y,
            np.ones(n),
            Gaussian(),
            IdentityLink(),
            [GroupSlice("a", 0, 1), GroupSlice("b", 1, 2)],
            lambda2={"b": 1e20},
            reml_penalties=[ridge],
            weight_semantics="frequency",
        )
        (trace,) = traces
        assert 0.0 < trace[1] < 100.0 * np.finfo(np.float64).eps
        assert result.rank_info is not None
        assert result.rank_info.feature_edf[1] == 0.0
        assert result.rank_info.group_edf["b"] == 0.0
        assert result.effective_df == 1.0 + result.rank_info.feature_edf[0]

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
        """The direct optimizer updates lambda weights without rebuilding the design matrix."""
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
            raise AssertionError("The direct optimizer should not rebuild the design matrix")

        monkeypatch.setattr("superglm.model.base.rebuild_dm_with_lambdas", fail_rebuild)
        m.fit_reml(X, y, sample_weight=w, max_reml_iter=20)

        assert hasattr(m, "_reml_lambdas")
        assert "DrivAge" in m._reml_lambdas

    def test_exact_fit_gaussian_stops_at_its_fixed_point(self, monkeypatch):
        """A response the model fits to rounding ends each PIRLS solve in two iterations.

        Gaussian/identity is one penalised least-squares solve.  When the
        model reproduces its response exactly (a level effect plus the
        rounding a log-ratio leaves, |noise| <= 2.2e-16) the residuals and so
        the score's own scale are rounding, the relative score sits near 1,
        and the floors that would read it are never evaluated.  Both events
        are forced, so no BLAS or libm rounding decides them: the working
        response is ``y`` itself (``eta + (y - eta)`` without its rounding),
        so the second iteration of every solve returns the first one's
        coefficients bit for bit, and the capped score reads twice its bar.
        That second iteration is a fixed point: the solve stops there, with
        the floors resolved (``resolve_cap = inf``), instead of running every
        iteration of its budget at each smoothing parameter (2,802 iterations
        over 29 solves before the fix).
        """
        from superglm.features.numeric import Numeric
        from superglm.features.random_effect import RandomEffect

        rng = np.random.default_rng(0)
        n = 48
        level = np.tile(["x", "y"], n // 2)
        base = rng.uniform(1.0, 2.0, n)
        noise = np.log(base * 3.0) - np.log(3.0) - np.log(base)
        y = np.where(level == "y", np.log(0.6), 0.0) + noise
        assert np.max(np.abs(noise)) <= 4 * np.finfo(float).eps
        _force_constant_working_rows(monkeypatch, y)
        floor_verdicts = _force_unresolved_score(monkeypatch)
        X = pd.DataFrame(
            {
                "num": np.tile([4.0, 5.5, 6.5], n // 3),
                "a": level,
                "fam": np.repeat(["f1", "f2"], n // 2),
            }
        )
        m = SuperGLM(
            family="gaussian",
            features={"num": Numeric(), "a": Categorical(), "fam": RandomEffect()},
        )
        with warnings.catch_warnings():
            # the smoothing parameter of a family with no effect is not
            # identified by rounding residuals; its disclosure is not tested here
            warnings.simplefilter("ignore")
            m.fit_reml(X, y)
        profile = m._reml_profile
        # every solve ended at its fixed point, at most two iterations in (a
        # solve warm-started at its own fixed point stops on the first)
        assert len(floor_verdicts) == profile["irls_calls"]
        assert profile["irls_iters"] <= 2 * profile["irls_calls"]
        # the published verdict is the floor-resolved one: certified within
        # the floor, the limiting accuracy (not converged) above it
        np.testing.assert_allclose(
            profile["irls_mode_score"], profile["irls_mode_bar"] * floor_verdicts[-1], rtol=1e-15
        )
        stagnated = m._reml_result.terminal_refit_termination == "score_stagnated"
        assert stagnated == (floor_verdicts[-1] > 1.0)
