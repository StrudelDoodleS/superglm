"""NB2 theta estimation correctness fixtures (2026-08-20 audit, findings B1/B2).

Every test here fails against v0.28.0 and passes with the safeguarded,
data-started theta solve (B2) and the post-REML joint theta/lambda fixed
point (B1). The mgcv 1.9.3 oracle values are pinned in
``docs/audit/2026-08-20-distribution-estimation/README.md`` (sections 3.3-3.4);
mgcv was run strictly as a black-box oracle on the committed fixture CSVs.

Fixture provenance (all synthetic):
- ``nb_clamp005.csv``: n=4000, mu = exp(1.2 + 0.8 sin(2 pi x)), theta_true=0.05
  (heavy but realistic overdispersion). mgcv ``nb()`` theta: 0.05068.
  v0.28.0 published theta_hat = 50.0 with converged=True and no warning.
- ``nb_worst.csv``: n=3000, mu = exp(0.8 + 1.5 sin(6 pi x)), theta_true=1.0
  (high-frequency truth the pre-REML calibration smoothing cannot follow).
  mgcv ``nb()`` theta: 0.99574. v0.28.0 published theta_hat = 0.5541 (-45%).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from superglm import SuperGLM
from superglm.distributions import NegativeBinomial
from superglm.features.spline import CubicRegressionSpline

FIXTURES = Path(__file__).parent / "fixtures"

MGCV_THETA_CLAMP005 = 0.05068
MGCV_THETA_HIFREQ = 0.99574


def _fit_auto_theta(csv_name: str, n_knots: int) -> SuperGLM:
    data = pd.read_csv(FIXTURES / csv_name)
    model = SuperGLM(
        features={"x": CubicRegressionSpline(n_knots=n_knots)},
        family=NegativeBinomial("auto"),
    )
    model.fit_reml(data[["x"]], data["y"].to_numpy(dtype=np.float64))
    return model


class TestThetaWrongRootAndClamp:
    """Finding B2: the fixed-start Newton took the wrong root and the
    (0.1, 50.0) clamp published it silently with converged=True."""

    def test_heavy_overdispersion_recovers_mgcv_theta(self):
        """theta_true=0.05 must publish ~0.0507, not the 50.0 upper clamp.

        On v0.28.0 this exact fit publishes theta_hat = 50.0 (the wrong end
        of the parameter space, three orders of magnitude off, 17209 nats
        worse than the profile optimum) with converged=True and no warning.
        """
        model = _fit_auto_theta("nb_clamp005.csv", n_knots=10)
        theta_hat = float(model._distribution.theta)
        assert theta_hat == pytest.approx(MGCV_THETA_CLAMP005, rel=0.10)
        result = model._nb_profile_result
        assert result.converged
        # Published family and profile result must describe the same fit.
        assert float(result.theta_hat) == theta_hat

    def test_near_poisson_data_escapes_the_legacy_ceiling(self):
        """theta_true=150: the free profile optimum (~340) must be reachable.

        On v0.28.0 the (0.1, 50.0) clamp stops at theta_hat = 50.0 exactly,
        overstating V(mu_bar) by ~6.5% and stopping the NB fit ~9 AIC short
        of the Poisson fit it should approach (audit section 3.4).
        """
        n, seed, theta_true = 4000, 29, 150.0
        rng = np.random.default_rng(seed)
        x = rng.uniform(0, 1, n)
        mu_true = np.exp(1.2 + 0.8 * np.sin(2 * np.pi * x))
        y = rng.negative_binomial(theta_true, theta_true / (theta_true + mu_true))
        model = SuperGLM(
            features={"x": CubicRegressionSpline(n_knots=10)},
            family=NegativeBinomial("auto"),
        )
        model.fit_reml(pd.DataFrame({"x": x}), y.astype(np.float64))
        theta_hat = float(model._distribution.theta)
        # The audit's wide-bounds reference optimum on this design is ~340.
        assert theta_hat > 100.0
        assert theta_hat == pytest.approx(340.0, rel=0.35)
        assert model._nb_profile_result.converged

    def test_theta_ml_walks_downhill_from_the_legacy_start(self):
        """The inner solve from theta=1.0 must descend toward the optimum.

        On v0.28.0 the unsafeguarded Newton from 1.0 ascends the NLL and the
        clip converts the runaway into the 50.0 upper bound (returned here as
        a bare float), so the value assertion below reads 50.0.
        """
        from superglm.profiling.nb import _theta_ml

        data = pd.read_csv(FIXTURES / "nb_clamp005.csv")
        y = data["y"].to_numpy(dtype=np.float64)
        weights = np.ones(len(y))
        mu = np.full(len(y), y.mean())
        solve = _theta_ml(y, mu, weights, 1.0)
        theta = float(getattr(solve, "theta", solve))
        assert theta < 0.2
        assert bool(getattr(solve, "converged", False))
        assert not bool(getattr(solve, "at_upper", True))

    def test_user_bounds_bind_loudly_and_honestly(self):
        """A user-supplied bound that binds must warn and report converged=False.

        On v0.28.0 the same call publishes theta_hat = 50.0 (the *upper*
        bound, on the wrong side of the optimum) with converged=True and no
        warning; NBThetaBoundWarning does not exist there.
        """
        from superglm.profiling.nb import NBThetaBoundWarning, estimate_nb_theta

        data = pd.read_csv(FIXTURES / "nb_clamp005.csv")
        y = data["y"].to_numpy(dtype=np.float64)
        model = SuperGLM(
            features={"x": CubicRegressionSpline(n_knots=10)},
            family=NegativeBinomial(theta=1.0),
        )
        with pytest.warns(NBThetaBoundWarning, match="lower search bound"):
            result = estimate_nb_theta(model, data[["x"]], y, theta_bounds=(0.1, 50.0))
        # The bracketed solve stops at the *near* end of the constraint.
        assert result.theta_hat == pytest.approx(0.1, rel=1e-6)
        assert not result.converged


class TestLargeThetaScoreStability:
    """Review round 2, P1: the naive profile score cancels catastrophically
    in the near-Poisson decades the widened bounds newly admit."""

    @staticmethod
    def _poisson_rows():
        rng = np.random.default_rng(29)
        n = 4000
        mu = np.exp(1.2 + 0.8 * np.sin(2 * np.pi * rng.uniform(0, 1, n)))
        y = rng.poisson(mu).astype(np.float64)
        return y, mu, np.ones(n)

    def test_large_theta_score_is_likelihood_geometry_not_roundoff(self):
        """theta^2 * score must be a stable positive constant on Poisson data.

        The true profile score on exactly-Poisson data is positive for every
        theta (the likelihood increases toward the Poisson limit) and decays
        as theta^-2. The naive digamma/log/ratio form obtains that O(th^-2)
        value by cancelling O(th^-1) pieces computed from O(log th)-sized
        intermediates: measured on this fixture it drifts to +661/th^2 at
        1e7 and flips sign to -4854/th^2 at 1e8 - a bracketing solve reads
        that flip as a root and publishes an arbitrary interior estimate
        with converged=True, silently, exactly in the regime the widened
        bounds enabled. The stable large-theta expansion holds
        theta^2 * score at +316.88 through 1e9.
        """
        from superglm.profiling.nb import _theta_profile_score

        y, mu, weights = self._poisson_rows()
        scaled = {
            theta: theta * theta * _theta_profile_score(y, mu, weights, theta)
            for theta in (1e6, 1e7, 1e8)
        }
        for theta, value in scaled.items():
            assert value > 0.0, f"score sign lost to roundoff at theta={theta:g}"
        reference = scaled[1e6]
        for theta, value in scaled.items():
            assert value == pytest.approx(reference, rel=0.02), (
                f"theta^2 * score is not stable at theta={theta:g}"
            )

    def test_branches_agree_where_both_are_accurate(self):
        """The expansion must join the naive form seamlessly at the switch."""
        from scipy.special import digamma

        from superglm.profiling import nb as nb_module

        y, mu, weights = self._poisson_rows()
        theta = 2.0e4  # naive form still accurate; force the expansion here

        def naive(theta_value):
            return float(
                np.sum(
                    weights
                    * (
                        digamma(y + theta_value)
                        - digamma(theta_value)
                        + np.log(theta_value)
                        + 1.0
                        - np.log(theta_value + mu)
                        - (y + theta_value) / (mu + theta_value)
                    )
                )
            )

        original = nb_module._THETA_SCORE_ASYMPTOTIC_MIN
        try:
            nb_module._THETA_SCORE_ASYMPTOTIC_MIN = 1.0
            expansion = nb_module._theta_profile_score(y, mu, weights, theta)
        finally:
            nb_module._THETA_SCORE_ASYMPTOTIC_MIN = original
        assert expansion == pytest.approx(naive(theta), rel=1e-6)

    def test_poisson_data_reports_the_ceiling_honestly(self):
        """With a trustworthy sign the solve walks to the bound and says so."""
        from superglm.profiling.nb import _theta_ml

        y, mu, weights = self._poisson_rows()
        solve = _theta_ml(y, mu, weights, 1.0)
        assert solve.at_upper
        assert not solve.converged


class TestAlternationToleranceIsRelative:
    """Review round 2, P2: an absolute xatol=1e-2 accepts order-of-magnitude
    jumps below theta=0.01 now that the search range admits them."""

    def test_small_theta_steps_do_not_satisfy_the_absolute_reading(self, monkeypatch):
        from types import SimpleNamespace

        from superglm.features import Numeric
        from superglm.profiling import nb as nb_module

        X = pd.DataFrame({"x": np.linspace(-1.0, 1.0, 24)})
        y = np.resize(np.array([1.0, 2.0, 3.0]), len(X))
        model = SuperGLM(
            family=NegativeBinomial(theta=1.0),
            selection_penalty=0.0,
            features={"x": Numeric()},
        )

        def result_for(dm):
            return SimpleNamespace(
                beta=np.zeros(dm.p),
                intercept=float(np.log(np.mean(y))),
                n_iter=1,
                converged=True,
            )

        monkeypatch.setattr(
            nb_module, "fit_irls_direct", lambda **kwargs: (result_for(kwargs["X"]), None)
        )
        monkeypatch.setattr(nb_module, "fit_pirls", lambda **kwargs: result_for(kwargs["X"]))
        # A scripted alternation still moving 60% per step at theta ~ 0.002:
        # the absolute reading (|0.002 - 0.005| = 0.003 < 0.01) stops on the
        # second iterate; the relative reading continues to the settled one.
        script = iter([0.005, 0.002, 0.0019998, 0.0019998])
        monkeypatch.setattr(
            nb_module,
            "_theta_ml",
            lambda *args, **kwargs: nb_module._ThetaSolve(
                theta=next(script),
                converged=True,
                at_lower=False,
                at_upper=False,
                n_score_evaluations=1,
            ),
        )
        result = nb_module.estimate_nb_theta(model, X, y, maxiter=10)
        assert result.theta_hat == pytest.approx(0.0019998, rel=1e-9)


class TestProfilePlotSmallTheta:
    """Review round 2, P2: the plot grid's fixed 0.01 floor made every
    estimate in the newly admitted (1e-8, 0.01) band unplottable."""

    def test_profile_plot_reaches_a_small_theta_estimate(self):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")

        from superglm.features import Numeric
        from superglm.profiling.nb import estimate_nb_theta

        rng = np.random.default_rng(41)
        n = 4000
        x = rng.uniform(-1.0, 1.0, n)
        mu_true = np.exp(1.0 + 0.2 * x)
        theta_true = 0.003
        y = rng.negative_binomial(theta_true, theta_true / (theta_true + mu_true)).astype(
            np.float64
        )
        model = SuperGLM(
            family=NegativeBinomial(theta=1.0),
            selection_penalty=0.0,
            features={"x": Numeric()},
        )
        result = estimate_nb_theta(model, pd.DataFrame({"x": x}), y)
        assert result.theta_hat < 0.01, "fixture must land in the sub-0.01 band"
        figure = result.profile_plot()
        try:
            axis = figure.axes[0]
            # The profile CURVE itself must reach below the estimate; the
            # axes' xlim is no instrument (the MLE axvline extends it even
            # when the curve's grid never gets there).
            curve_grid_lo = float(np.min(axis.get_lines()[0].get_xdata()))
            assert curve_grid_lo < float(result.theta_hat)
        finally:
            import matplotlib.pyplot as plt

            plt.close(figure)


class TestThetaFrozenBeforeReml:
    """Finding B1: theta was calibrated before REML at the configured
    smoothing and never revisited, converting lack-of-fit at lambda2=0.1
    into spurious overdispersion."""

    def test_hifreq_design_reaches_the_joint_fixed_point(self):
        """theta_true=1 with a high-frequency truth must land near mgcv.

        On v0.28.0 the pre-REML freeze publishes theta_hat = 0.5541 (-45%):
        the calibration fit at lambda2=0.1 cannot follow sin(6 pi x) and the
        misfit is absorbed into theta. mgcv nb() on the same CSV: 0.99574.
        """
        model = _fit_auto_theta("nb_worst.csv", n_knots=20)
        theta_hat = float(model._distribution.theta)
        assert theta_hat == pytest.approx(MGCV_THETA_HIFREQ, rel=0.05)
        result = model._nb_profile_result
        assert result.converged
        assert float(result.theta_hat) == theta_hat
        # The profile CI must be evaluated at the REML fit and bracket the
        # published estimate.
        ci_lo, ci_hi = result.ci()
        assert ci_lo < theta_hat < ci_hi
