"""Tests for profile likelihood CIs (NB theta and Tweedie p)."""

import numpy as np
import pandas as pd

from superglm import SuperGLM
from superglm.distributions import NegativeBinomial
from superglm.features.numeric import Numeric
from superglm.nb_profile import estimate_nb_theta, profile_ci_theta
from superglm.tweedie_profile import estimate_tweedie_p


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
            lambda1=0.001,
            features={"x": Numeric()},
        )
        result = estimate_nb_theta(model, X, y)
        ci_lo, ci_hi = result.ci(alpha=0.05)

        assert ci_lo < true_theta < ci_hi
        assert ci_lo > 0
        assert ci_hi > ci_lo

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
            lambda1=0.001,
            features={"x": Numeric()},
        )
        result = estimate_nb_theta(model, X, y)
        ci_lo, ci_hi = result.ci()

        assert ci_lo < result.theta_hat < ci_hi

    def test_standalone_function(self):
        """profile_ci_theta works directly with y/mu/weights."""
        rng = np.random.default_rng(42)
        n = 1000
        mu = np.full(n, 2.0)
        theta = 5.0
        p_nb = theta / (mu + theta)
        y = rng.negative_binomial(theta, p_nb).astype(float)
        weights = np.ones(n)

        ci_lo, ci_hi = profile_ci_theta(y, mu, weights, theta)
        assert ci_lo < theta < ci_hi

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
            lambda1=0.001,
            features={"x": Numeric()},
        )
        result = estimate_nb_theta(model, X, y)

        ci_95_lo, ci_95_hi = result.ci(alpha=0.05)
        ci_99_lo, ci_99_hi = result.ci(alpha=0.01)

        assert ci_99_lo <= ci_95_lo
        assert ci_99_hi >= ci_95_hi

    def test_profile_plot_returns_figure(self):
        """profile_plot() should return a matplotlib Figure."""
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
            lambda1=0.001,
            features={"x": Numeric()},
        )
        result = estimate_nb_theta(model, X, y)
        fig = result.profile_plot()

        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        assert ax.get_xlabel() == r"$\theta$"
        assert len(ax.lines) >= 1  # at least the profile curve
        plt.close(fig)

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
            lambda1=0.001,
            features={"x": Numeric()},
        )
        result = estimate_nb_theta(model, X, y)

        fig, ax = plt.subplots()
        returned_fig = result.profile_plot(ax=ax)
        assert returned_fig is fig
        plt.close(fig)


class TestTweedieProfileCI:
    def test_ci_works(self):
        """Tweedie profile CI should produce a valid interval."""
        from superglm.tweedie_profile import generate_tweedie_cpg

        rng = np.random.default_rng(42)
        n = 1000
        true_p = 1.5
        x = rng.standard_normal(n)
        mu = np.exp(1.0 + 0.3 * x)
        y = generate_tweedie_cpg(n, mu, phi=1.0, p=true_p, rng=rng)
        # Ensure some positive values
        y = np.maximum(y, 0.0)
        X = pd.DataFrame({"x": x})

        model = SuperGLM(
            family="tweedie",
            lambda1=0.001,
            features={"x": Numeric()},
        )
        result = estimate_tweedie_p(model, X, y)
        ci_lo, ci_hi = result.ci(alpha=0.05)

        # Should be a valid interval containing p_hat
        assert ci_lo < result.p_hat < ci_hi
        # Interval should be within the valid range
        assert ci_lo >= 1.0
        assert ci_hi <= 2.0

    def test_profile_plot_returns_figure(self):
        """profile_plot() should return a matplotlib Figure."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from superglm.tweedie_profile import generate_tweedie_cpg

        rng = np.random.default_rng(42)
        n = 500
        x = rng.standard_normal(n)
        mu = np.exp(1.0 + 0.3 * x)
        y = generate_tweedie_cpg(n, mu, phi=1.0, p=1.5, rng=rng)
        y = np.maximum(y, 0.0)
        X = pd.DataFrame({"x": x})

        model = SuperGLM(
            family="tweedie",
            lambda1=0.001,
            features={"x": Numeric()},
        )
        result = estimate_tweedie_p(model, X, y)
        fig = result.profile_plot(n_points=20)

        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        assert ax.get_xlabel() == "p"
        assert len(ax.lines) >= 1
        plt.close(fig)
