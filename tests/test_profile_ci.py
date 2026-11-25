"""Tests for profile likelihood CIs (NB theta and Tweedie p)."""

import numpy as np
import pandas as pd
import pytest

from superglm import SuperGLM, generate_tweedie_cpg
from superglm.distributions import NegativeBinomial, Tweedie
from superglm.features.numeric import Numeric
from superglm.profiling._scalar import RecordedObjective
from superglm.profiling.tweedie import TweedieProfileResult, _Candidate


class TestNBThetaProfileCI:
    def test_ci_contains_true_theta(self):
        """CI should contain the true theta for well-specified model."""
        rng = np.random.default_rng(42)
        n = 3000
        true_theta = 5.0
        x = rng.standard_normal(n)
        mu = np.exp(0.5 + 0.3 * x)
        p_nb = true_theta / (mu + true_theta)
        y = rng.negative_binomial(true_theta, p_nb).astype(float)
        X = pd.DataFrame({"x": x})

        model = SuperGLM(
            family=NegativeBinomial(theta=1.0),
            selection_penalty=0.001,
            features={"x": Numeric()},
        )
        result = model.estimate_theta(X, y)
        ci_lo, ci_hi = result.ci(alpha=0.05)

        assert ci_lo < true_theta < ci_hi
        assert ci_lo > 0
        assert ci_hi > ci_lo

    def test_ci_endpoints_invert_the_profile_likelihood_ratio(self):
        """The 95% interval's endpoints solve 2 (l(theta_opt) - l(theta)) = chi2_1(0.95) on
        the NB2 log-likelihood in theta alone, at a mean fitted once and then frozen, as
        ``NBProfileResult.interval()`` documents ("the fixed-mean profile" at the published
        mean). The mean is statsmodels' joint NB2 maximum-likelihood fit and the likelihood is
        SciPy's ``nbinom.logpmf`` (Venables and Ripley, MASS, 4th ed., section 7.4); an
        interval of the wrong width would still contain the true theta, and a reference that
        refits the mean at each theta moves these endpoints by about 4e-5 relative."""
        sm = pytest.importorskip("statsmodels.api")
        from scipy import optimize, stats

        rng = np.random.default_rng(42)
        n = 3000
        x = rng.standard_normal(n)
        mu = np.exp(0.5 + 0.3 * x)
        y = rng.negative_binomial(5.0, 5.0 / (mu + 5.0)).astype(float)
        design = np.column_stack([np.ones(n), x])

        joint = sm.NegativeBinomial(y, design, loglike_method="nb2").fit(
            method="newton", tol=1e-12, maxiter=100, disp=0
        )
        mu_frozen = np.exp(design @ joint.params[:2])

        def fixed_mean_ll(log_theta):
            theta = np.exp(log_theta)
            return stats.nbinom.logpmf(y, theta, theta / (theta + mu_frozen)).sum()

        best = optimize.minimize_scalar(
            lambda t: -fixed_mean_ll(t),
            bounds=(0.0, 4.0),
            method="bounded",
            options={"xatol": 1e-10},
        )
        cut = fixed_mean_ll(best.x) - 0.5 * stats.chi2.isf(0.05, 1)
        lower = np.exp(optimize.brentq(lambda t: fixed_mean_ll(t) - cut, -2.0, best.x, xtol=1e-13))
        upper = np.exp(optimize.brentq(lambda t: fixed_mean_ll(t) - cut, best.x, 6.0, xtol=1e-13))

        model = SuperGLM(
            family=NegativeBinomial(theta=1.0),
            selection_penalty=0.0,
            features={"x": Numeric()},
        )
        ci_lo, ci_hi = model.estimate_theta(pd.DataFrame({"x": x}), y).ci(alpha=0.05)
        # The library roots each endpoint in log theta to xtol = 1e-6, so an endpoint
        # errs by at most exp(1e-6) - 1 ~ 1e-6 relative; the reference roots to 1e-13.
        # The second 1e-6 covers the published mean, which matches the joint maximum-
        # likelihood mean only to the alternation's convergence (5e-8 relative here).
        np.testing.assert_allclose([ci_lo, ci_hi], [lower, upper], rtol=2e-6)

    def test_ci_is_interval(self):
        """Lower bound should be less than upper bound."""
        rng = np.random.default_rng(123)
        n = 1000
        x = rng.standard_normal(n)
        mu = np.exp(0.5 + 0.2 * x)
        y = rng.negative_binomial(3, 3 / (mu + 3)).astype(float)
        X = pd.DataFrame({"x": x})

        model = SuperGLM(
            family=NegativeBinomial(theta=1.0),
            selection_penalty=0.001,
            features={"x": Numeric()},
        )
        result = model.estimate_theta(X, y)
        ci_lo, ci_hi = result.ci()

        assert ci_lo < result.theta_hat < ci_hi

    def test_narrower_alpha_gives_wider_ci(self):
        """alpha=0.01 should give a wider CI than alpha=0.05."""
        rng = np.random.default_rng(42)
        n = 2000
        x = rng.standard_normal(n)
        mu = np.exp(0.5 + 0.3 * x)
        y = rng.negative_binomial(5, 5 / (mu + 5)).astype(float)
        X = pd.DataFrame({"x": x})

        model = SuperGLM(
            family=NegativeBinomial(theta=1.0),
            selection_penalty=0.001,
            features={"x": Numeric()},
        )
        result = model.estimate_theta(X, y)

        ci_95_lo, ci_95_hi = result.ci(alpha=0.05)
        ci_99_lo, ci_99_hi = result.ci(alpha=0.01)

        assert ci_99_lo <= ci_95_lo
        assert ci_99_hi >= ci_95_hi

    def test_profile_plot_returns_axes(self):
        """profile_plot() draws the likelihood-ratio statistic and returns its Axes."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rng = np.random.default_rng(42)
        n = 1000
        x = rng.standard_normal(n)
        mu = np.exp(0.5 + 0.2 * x)
        y = rng.negative_binomial(5, 5 / (mu + 5)).astype(float)
        X = pd.DataFrame({"x": x})

        model = SuperGLM(
            family=NegativeBinomial(theta=1.0),
            selection_penalty=0.001,
            features={"x": Numeric()},
        )
        result = model.estimate_theta(X, y)
        ax = result.profile_plot()

        assert isinstance(ax, plt.Axes)
        assert ax.get_xlabel() == "theta"
        assert len(ax.lines) >= 1  # at least the profile curve
        plt.close(ax.figure)

    def test_profile_plot_on_existing_ax(self):
        """profile_plot() should work with a provided Axes."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rng = np.random.default_rng(42)
        n = 1000
        x = rng.standard_normal(n)
        mu = np.exp(0.5 + 0.2 * x)
        y = rng.negative_binomial(5, 5 / (mu + 5)).astype(float)
        X = pd.DataFrame({"x": x})

        model = SuperGLM(
            family=NegativeBinomial(theta=1.0),
            selection_penalty=0.001,
            features={"x": Numeric()},
        )
        result = model.estimate_theta(X, y)

        fig, ax = plt.subplots()
        assert result.profile_plot(ax=ax) is ax
        plt.close(fig)


class TestTweedieProfileCI:
    @staticmethod
    def _estimated(n=1000, seed=42):
        rng = np.random.default_rng(seed)
        x = rng.standard_normal(n)
        mu = np.exp(1.0 + 0.3 * x)
        y = generate_tweedie_cpg(n, mu, phi=1.0, p=1.5, rng=rng)
        model = SuperGLM(
            family=Tweedie(p=1.5),
            selection_penalty=0.001,
            features={"x": Numeric()},
        )
        return model.estimate_p(pd.DataFrame({"x": x}), y)

    def test_ci_works(self):
        """Tweedie profile CI should produce a valid interval."""
        result = self._estimated()
        ci_lo, ci_hi = result.ci(alpha=0.05)

        # Should be a valid interval containing p_hat
        assert ci_lo < result.p_hat < ci_hi
        # Interval should be within the valid range
        assert ci_lo >= 1.0
        assert ci_hi <= 2.0

    @pytest.mark.parametrize(
        ("p_true", "seed", "side"),
        [pytest.param(1.057, 1, "lower", id="lower"), pytest.param(1.93, 3, "upper", id="upper")],
    )
    def test_an_interior_estimate_reaches_its_crossing_past_the_search_bound(
        self, p_true, seed, side
    ):
        """The interval searches (1.02, 1.98), past the default bounds (1.05, 1.95).

        An interior p_hat near a bound can have its likelihood-ratio crossing
        beyond it; stopping at the bound would report that genuine endpoint as
        censored. Both crossings sit at least 5e-3 past the bound, fifty times
        the interval's 1e-4 root tolerance.
        """
        rng = np.random.default_rng(seed)
        n = 400
        x = rng.uniform(-1.0, 1.0, n)
        y = generate_tweedie_cpg(n, np.exp(0.3 + 0.5 * x), 1.0, p_true, rng=rng)
        model = SuperGLM(family=Tweedie(p=1.5), selection_penalty=0.0, features={"x": Numeric()})
        result = model.estimate_p(pd.DataFrame({"x": x}), y)

        interval = result.interval(0.05)

        assert 1.05 < result.p_hat < 1.95
        assert not getattr(interval, f"{side}_censored")
        end = getattr(interval, side)
        assert 1.02 < end < 1.05 if side == "lower" else 1.95 < end < 1.98
        assert result.warnings == []

    def test_profile_plot_returns_axes(self):
        """profile_plot() draws the evaluated powers and returns its Axes."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        result = self._estimated(n=500)
        ax = result.profile_plot()

        assert isinstance(ax, plt.Axes)
        assert ax.get_xlabel() == "p"
        assert len(ax.lines) >= 1
        plt.close(ax.figure)

    def test_profile_plot_fits_nothing_and_shades_only_a_computed_interval(self, monkeypatch):
        """The plot reads the recorded profile; the interval stays the caller's explicit cost."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        result = self._estimated(n=500)
        evaluated = dict(result._objective.values)
        result.interval(0.05)

        def unexpected_interval(*args, **kwargs):
            raise AssertionError("profile_plot must not compute an interval")

        monkeypatch.setattr(result, "interval", unexpected_interval)
        monkeypatch.setattr(result, "ci", unexpected_interval)
        recorded = dict(result._objective.values)
        shaded = result.profile_plot(alpha=0.05)
        unshaded = result.profile_plot(alpha=0.10)

        assert len(recorded) > len(evaluated)  # the interval's own evaluations
        assert result._objective.values == recorded
        assert len(shaded.patches) == 1  # the interval band
        assert len(unshaded.patches) == 0
        plt.close("all")

    def test_a_censored_side_is_warned_once_per_alpha(self):
        """Censoring is recorded when the interval is computed, not on every read."""
        objective = RecordedObjective(lambda p: 0.01 * (p - 0.5) ** 2)
        objective(0.5)
        result = TweedieProfileResult(
            p_hat=0.5,
            phi_hat=1.0,
            nll=0.0,
            converged=True,
            fit_mode="fit",
            evaluations=pd.DataFrame({"p": [0.5], "nll": [0.0]}),
            warnings=[],
            search_nll=0.0,
            _objective=objective,
            _ll_scale=1.0,
            _ci_bounds=(0.0, 1.0),
            # An exact objective: its recorded values carry no evaluation error.
            _candidates={0.5: _Candidate(1.0, True, error=0.0)},
        )

        # The statistic 0.02 (p - 0.5)^2 never reaches the cutoff inside the bounds.
        assert result.ci() == (0.0, 1.0)
        assert result.ci() == (0.0, 1.0)
        censored = [warning for warning in result.warnings if "censored" in warning]
        assert len(censored) == 2
        assert "lower end p=0" in censored[0] and "upper end p=1" in censored[1]
