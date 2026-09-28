"""Tests for Negative Binomial (NB2) distribution."""

import math
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.stats import nbinom

from superglm import (
    FactorSmooth,
    LambdaPolicy,
    NegativeBinomial,
    RandomEffect,
    Spline,
    SuperGLM,
    SuperGLMRegressor,
)
from superglm._frame import EagerFrame
from superglm.distributions import resolve_distribution
from superglm.features.numeric import Numeric
from superglm.penalties.group_lasso import GroupLasso
from superglm.profiling.nb import (
    NBProfileResult,
    NBThetaBoundWarning,
    estimate_nb_theta,
    solve_theta,
)
from superglm.solvers.dispersion import PRIOR_WEIGHTS

# =====================================================================
# Helpers
# =====================================================================


def _generate_nb2(n, mu, theta, rng=None):
    """Simulate NB2(mu, theta) using scipy's nbinom."""
    if rng is None:
        rng = np.random.default_rng()
    mu = np.broadcast_to(np.asarray(mu, dtype=np.float64), (n,)).copy()
    p = theta / (mu + theta)
    return rng.negative_binomial(theta, p).astype(np.float64)


# =====================================================================
# TestNB2Distribution
# =====================================================================


class TestNB2VarianceFunction:
    def test_basic(self):
        nb = NegativeBinomial(theta=5.0)
        mu = np.array([1.0, 2.0, 5.0, 10.0])
        expected = mu + mu**2 / 5.0
        np.testing.assert_allclose(nb.variance(mu), expected)

    def test_large_theta_approaches_poisson(self):
        """V(mu) = mu + mu^2/theta -> mu as theta -> inf."""
        nb = NegativeBinomial(theta=1e8)
        mu = np.array([1.0, 5.0, 10.0])
        np.testing.assert_allclose(nb.variance(mu), mu, rtol=1e-6)

    def test_small_theta_large_variance(self):
        nb = NegativeBinomial(theta=0.5)
        mu = np.array([5.0])
        expected = 5.0 + 25.0 / 0.5  # 55
        np.testing.assert_allclose(nb.variance(mu), expected)


class TestNB2DevianceUnit:
    def test_positive_y(self):
        nb = NegativeBinomial(theta=5.0)
        y = np.array([3.0, 7.0, 1.0])
        mu = np.array([2.0, 5.0, 3.0])
        d = nb.deviance_unit(y, mu)
        # All unit deviances should be non-negative
        assert np.all(d >= 0)

    def test_y_equals_mu(self):
        """Unit deviance at y=mu should be zero."""
        nb = NegativeBinomial(theta=5.0)
        mu = np.array([2.0, 5.0, 10.0])
        d = nb.deviance_unit(mu, mu)
        np.testing.assert_allclose(d, 0.0, atol=1e-12)

    def test_y_zero(self):
        """y=0 case uses special formula."""
        nb = NegativeBinomial(theta=5.0)
        y = np.array([0.0, 0.0])
        mu = np.array([2.0, 5.0])
        d = nb.deviance_unit(y, mu)
        expected = 2 * 5.0 * np.log((mu + 5.0) / 5.0)
        np.testing.assert_allclose(d, expected)
        assert np.all(d >= 0.0)

    def test_total_deviance_positive(self):
        nb = NegativeBinomial(theta=3.0)
        rng = np.random.default_rng(42)
        y = _generate_nb2(1000, mu=5.0, theta=3.0, rng=rng)
        mu = np.full_like(y, 5.0)
        d = nb.deviance_unit(y, mu)
        assert np.sum(d) > 0


class TestNB2LogLikelihood:
    def test_matches_scipy(self):
        """Log-likelihood should match scipy.stats.nbinom.logpmf."""
        nb = NegativeBinomial(theta=5.0)
        y = np.array([0, 1, 2, 5, 10], dtype=float)
        mu = np.array([3.0, 3.0, 3.0, 3.0, 3.0])
        weights = np.ones_like(y)

        ll_ours = nb.log_likelihood(y, mu, weights)

        # scipy nbinom: n=theta, p=theta/(mu+theta)
        p_nb = 5.0 / (3.0 + 5.0)
        ll_scipy = np.sum(nbinom.logpmf(y.astype(int), n=5.0, p=p_nb))

        np.testing.assert_allclose(ll_ours, ll_scipy, rtol=1e-10)

    def test_weighted(self):
        nb = NegativeBinomial(theta=5.0)
        y = np.array([1.0, 2.0, 3.0])
        mu = np.array([2.0, 2.0, 2.0])
        w1 = np.ones(3)
        w2 = np.array([2.0, 2.0, 2.0])
        ll1 = nb.log_likelihood(y, mu, w1)
        ll2 = nb.log_likelihood(y, mu, w2)
        np.testing.assert_allclose(ll2, 2.0 * ll1)


class TestNB2PoissonLimit:
    def test_large_theta_coefficients(self):
        """With very large theta, NB2 fit should give similar results to Poisson."""
        rng = np.random.default_rng(42)
        n = 5000
        x = rng.normal(0, 1, n)
        log_mu = 1.0 + 0.5 * x
        mu = np.exp(log_mu)
        y = rng.poisson(mu).astype(float)
        X = pd.DataFrame({"x": x})

        # Poisson fit
        m_pois = SuperGLM(
            family="poisson", penalty=GroupLasso(lambda1=0.0), features={"x": Numeric()}
        )
        m_pois.fit(X, y)

        # NB2 with large theta
        m_nb = SuperGLM(
            family=NegativeBinomial(theta=1e6),
            penalty=GroupLasso(lambda1=0.0),
            features={"x": Numeric()},
        )
        m_nb.fit(X, y)

        np.testing.assert_allclose(m_nb.result.intercept, m_pois.result.intercept, atol=0.05)
        np.testing.assert_allclose(m_nb.result.beta, m_pois.result.beta, atol=0.05)


# =====================================================================
# TestNB2Fitting
# =====================================================================


class TestNB2FixedThetaFit:
    def test_convergence(self):
        """NB2 model with fixed theta converges on synthetic data."""
        rng = np.random.default_rng(42)
        n = 3000
        theta = 5.0
        x = rng.normal(0, 1, n)
        mu = np.exp(1.0 + 0.3 * x)
        y = _generate_nb2(n, mu=mu, theta=theta, rng=rng)
        X = pd.DataFrame({"x": x})

        model = SuperGLM(
            family=NegativeBinomial(theta=theta),
            penalty=GroupLasso(lambda1=0.0),
            features={"x": Numeric()},
        )
        model.fit(X, y)

        assert model.result.converged
        # Check intercept near 1.0 and coef near 0.3
        np.testing.assert_allclose(model.result.intercept, 1.0, atol=0.15)

    def test_prediction_reasonable(self):
        rng = np.random.default_rng(42)
        n = 2000
        theta = 3.0
        mu_true = 5.0
        y = _generate_nb2(n, mu=mu_true, theta=theta, rng=rng)
        X = pd.DataFrame({"dummy": np.ones(n)})

        model = SuperGLM(
            family=NegativeBinomial(theta=theta),
            penalty=GroupLasso(lambda1=0.0),
            features={"dummy": Numeric()},
        )
        model.fit(X, y)

        pred = model.predict(X)
        np.testing.assert_allclose(pred.mean(), mu_true, rtol=0.1)


# =====================================================================
# TestNB2Profile
# =====================================================================


class TestNB2ProfileTheta:
    @pytest.mark.parametrize(
        ("selection_penalty", "expected_route"),
        [
            pytest.param(None, "direct", id="none-disabled"),
            pytest.param(0.0, "direct", id="zero-disabled"),
            pytest.param("auto", "pirls", id="explicit-auto"),
        ],
    )
    def test_profile_resolves_selection_intent_numerically(
        self,
        monkeypatch,
        selection_penalty,
        expected_route,
    ):
        from types import SimpleNamespace

        from superglm.model import fit_ops
        from superglm.profiling import nb as nb_module

        X = pd.DataFrame({"x": np.linspace(-1.0, 1.0, 24)})
        y = np.resize(np.array([1.0, 2.0, 3.0]), len(X))
        model = SuperGLM(
            family=NegativeBinomial(theta=1.0),
            selection_penalty=selection_penalty,
            features={"x": Numeric()},
        )
        calls = []

        def result_for(dm):
            return SimpleNamespace(
                beta=np.zeros(dm.p),
                intercept=float(np.log(np.mean(y))),
                n_iter=1,
                converged=True,
            )

        def fake_direct(**kwargs):
            calls.append(("direct", None))
            return result_for(kwargs["X"]), None

        def fake_pirls(**kwargs):
            calls.append(("pirls", kwargs["penalty"].lambda1))
            return result_for(kwargs["X"])

        # The alternation's mean fits follow the ordinary fit policy's solver route.
        monkeypatch.setattr(fit_ops, "fit_irls_direct", fake_direct)
        monkeypatch.setattr(fit_ops, "fit_pirls", fake_pirls)
        monkeypatch.setattr(
            nb_module,
            "solve_theta",
            lambda *args, **kwargs: nb_module.ThetaSolve(theta=1.0, at_lower=False, at_upper=False),
        )

        estimate_nb_theta(model, X, y, maxiter=1)

        assert calls[0][0] == expected_route
        if expected_route == "pirls":
            assert isinstance(calls[0][1], float)
            assert calls[0][1] > 0.0

    def test_recovers_theta(self):
        """Profile estimation recovers theta from synthetic data."""
        rng = np.random.default_rng(42)
        n = 3000
        theta_true = 5.0
        x = rng.normal(0, 1, n)
        mu = np.exp(1.0 + 0.3 * x)
        y = _generate_nb2(n, mu=mu, theta=theta_true, rng=rng)
        X = pd.DataFrame({"x": x})

        model = SuperGLM(
            family=NegativeBinomial(theta=1.0),  # initial guess
            penalty=GroupLasso(lambda1=0.0),
            features={"x": Numeric()},
        )

        result = estimate_nb_theta(
            model,
            X,
            y,
            theta_bounds=(0.5, 20.0),
        )
        assert isinstance(result, NBProfileResult)
        np.testing.assert_allclose(result.theta_hat, theta_true, atol=2.0)

    def test_result_records_each_alternation_step(self):
        rng = np.random.default_rng(42)
        n = 2000
        y = _generate_nb2(n, mu=5.0, theta=3.0, rng=rng)
        X = pd.DataFrame({"dummy": np.ones(n)})

        model = SuperGLM(
            family=NegativeBinomial(theta=1.0),
            penalty=GroupLasso(lambda1=0.0),
            features={"dummy": Numeric()},
        )

        result = estimate_nb_theta(model, X, y, theta_bounds=(0.5, 15.0))
        assert list(result.evaluations.columns) == ["theta", "nll"]
        assert len(result.evaluations) >= 1  # alternating alg converges in few iters

    def test_family_must_be_nb(self):
        model = SuperGLM(
            family="poisson", penalty=GroupLasso(lambda1=0.0), features={"x": Numeric()}
        )
        X = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        y = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="NegativeBinomial"):
            model.estimate_theta(X, y)

    def test_design_matrix_error_restores_temporary_family(self, monkeypatch):
        model = SuperGLM(
            family=NegativeBinomial(theta=2.5),
            penalty=GroupLasso(lambda1=0.0),
            features={"x": Numeric()},
        )
        X = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        y = np.array([1.0, 2.0, 3.0])

        def fail_build(*args, **kwargs):
            raise RuntimeError("build failed")

        monkeypatch.setattr(model, "_build_design_matrix", fail_build)

        with pytest.raises(RuntimeError, match="build failed"):
            estimate_nb_theta(model, X, y)

        assert model.family.theta == pytest.approx(2.5)


def _prior_score_error_bound(y, mu, w, theta):
    """Float64 error of the branch ``theta_score`` takes, plus the expansion's truncation.

    A row's rounding scales with the terms it combines (``rounding``), so 8 eps
    of their sizes bounds it; summing the n row values (``summands``) adds at
    most n eps of their absolute total (Higham 2002, sec. 4.2). The expansion
    also drops psi's 1/(120 z^4) term, which moves psi(a + b) - psi(a) by at
    most 4 b / (120 a^5) for a = w theta, b = w y.
    """
    from scipy.special import digamma

    eps = np.finfo(np.float64).eps
    if theta * w.min() >= 1e5:
        shifted = theta + y
        x = (y - mu) / (theta + mu)
        tail = np.abs(0.5 * y / (theta * shifted) / w) + np.abs(
            (1.0 / theta**2 - 1.0 / shifted**2) / 12.0 / w**2
        )
        # log1p(x) - x cancels parts of size |x| to a value of size x^2 / 2.
        rounding, summands = np.abs(x) + tail, np.abs(np.log1p(x) - x) + tail
        truncation = 4.0 * (w * y) / (120.0 * (w * theta) ** 5)
    else:
        terms = np.stack(
            [
                digamma(w * (y + theta)),
                -digamma(w * theta),
                np.full_like(y, np.log(theta)),
                np.ones_like(y),
                -np.log(theta + mu),
                -(y + theta) / (mu + theta),
            ]
        )
        rounding, summands = np.abs(terms).sum(axis=0), np.abs(terms.sum(axis=0))
        truncation = 0.0
    return float(
        np.sum(w * (8.0 * eps * rounding + truncation)) + y.size * eps * np.sum(w * summands)
    )


def _nb_profile(y, mu, weights, semantics):
    from superglm.profiling.nb import nb_nll, solve_theta

    theta = solve_theta(y, mu, weights, 1.0, weight_semantics=semantics, bounds=(1e-8, 1e8)).theta
    return NBProfileResult(
        theta,
        nb_nll(y, mu, weights, theta, weight_semantics=semantics),
        True,
        _y=y,
        _mu=mu,
        _weights=weights,
        _weight_semantics=semantics,
    )


class TestNB2WeightedProfile:
    """Prior weights and frequency counts in the theta score and the interval."""

    @pytest.mark.parametrize(
        ("weights", "mu_range", "theta"),
        [
            pytest.param((0.5, 4.0), (0.5, 4.0), 1e6, id="expansion-w-theta-5e5"),
            pytest.param((1e-4,), (5e3, 2e4), 1e5, id="direct-w-theta-10"),
        ],
    )
    def test_prior_weight_score_is_the_exact_digamma_score(self, weights, mu_range, theta):
        """Under prior weights the psi arguments are w (y + theta) and w theta.

        The large-theta expansion's two psi corrections carry 1/w and 1/w^2,
        and the switch to it follows the smallest psi argument w theta, not
        theta: at w theta = 10 its dropped Bernoulli term is 1e5 times this
        bound. Each count w y is Poisson, the regime a large theta describes.
        """
        mpmath = pytest.importorskip("mpmath")
        from superglm.profiling.nb import theta_score

        rng = np.random.default_rng(3)
        n = 300
        mu = rng.uniform(*mu_range, n)
        w = rng.choice(weights, n)
        y = rng.poisson(w * mu) / w
        mpmath.mp.dps = 60
        t = mpmath.mpf(theta)
        exact = float(
            sum(
                mpmath.mpf(wi)
                * (
                    mpmath.digamma(mpmath.mpf(wi) * (mpmath.mpf(yi) + t))
                    - mpmath.digamma(mpmath.mpf(wi) * t)
                    + mpmath.log(t)
                    + 1
                    - mpmath.log(t + mpmath.mpf(mi))
                    - (mpmath.mpf(yi) + t) / (mpmath.mpf(mi) + t)
                )
                for yi, mi, wi in zip(y, mu, w)
            )
        )

        score = theta_score(y, mu, w, theta, weight_semantics="prior")

        assert abs(score - exact) <= _prior_score_error_bound(y, mu, w, theta)

    def test_frequency_counts_give_the_interval_of_the_replicated_rows(self):
        """The likelihood-ratio scale is the total count, not the row count."""
        rng = np.random.default_rng(17)
        n = 400
        mu = rng.uniform(1.0, 4.0, n)
        y = _generate_nb2(n, mu=mu, theta=2.0, rng=rng)
        counts = rng.integers(1, 5, n)
        compressed = _nb_profile(y, mu, counts.astype(np.float64), "frequency")
        replicated = _nb_profile(
            np.repeat(y, counts), np.repeat(mu, counts), np.ones(counts.sum()), "frequency"
        )

        # Each endpoint is rooted to 1e-6 in log theta.
        np.testing.assert_allclose(
            np.log(compressed.ci(0.05)), np.log(replicated.ci(0.05)), rtol=0.0, atol=2e-6
        )

    def test_zero_prior_weights_leave_the_interval_as_if_their_rows_were_deleted(self):
        rng = np.random.default_rng(19)
        n = 400
        mu = rng.uniform(1.0, 4.0, n)
        y = _generate_nb2(n, mu=mu, theta=2.0, rng=rng)
        weights = rng.uniform(0.5, 2.0, n)
        weights[::4] = 0.0
        carried = weights > 0.0
        with_zeros = _nb_profile(y, mu, weights, "prior")
        deleted = _nb_profile(y[carried], mu[carried], weights[carried], "prior")

        # Each endpoint is rooted to 1e-6 in log theta.
        np.testing.assert_allclose(
            np.log(with_zeros.ci(0.05)), np.log(deleted.ci(0.05)), rtol=0.0, atol=2e-6
        )


class TestNB2AutoTheta:
    def test_reml_mode_rejects_selection_before_profile_work(self, monkeypatch):
        from superglm.profiling import nb as nb_module

        X = pd.DataFrame({"x": np.linspace(-1.0, 1.0, 24)})
        y = np.resize(np.array([1.0, 2.0, 3.0]), len(X))
        model = SuperGLM(
            family=NegativeBinomial(theta="auto"),
            selection_penalty="auto",
            features={"x": Numeric()},
        )
        profile_calls = []

        def unexpected_profile(*args, **kwargs):
            profile_calls.append(True)
            raise AssertionError("profile work must not start")

        monkeypatch.setattr(nb_module, "estimate_nb_theta", unexpected_profile)

        with pytest.raises(ValueError, match="does not support selection penalties"):
            model.estimate_theta(X, y, fit_mode="reml")

        assert profile_calls == []

    def test_a_mutating_callback_cannot_rewrite_the_recorded_evaluations(self):
        rng = np.random.default_rng(11)
        X = pd.DataFrame({"x": rng.uniform(-1.0, 1.0, 400)})
        mu = np.exp(0.5 + 0.4 * X["x"].to_numpy())
        y = rng.negative_binomial(3.0, 3.0 / (3.0 + mu)).astype(float)
        model = SuperGLM(
            family=NegativeBinomial(theta="auto"),
            penalty=GroupLasso(lambda1=0.0),
            features={"x": Numeric()},
        )
        seen = []

        def vandal(row):
            seen.append(dict(row))
            row["theta"], row["nll"] = -1.0, float("nan")

        result = estimate_nb_theta(model, X, y, on_evaluation=vandal)

        assert seen
        assert result.evaluations.to_dict("records") == seen

    @pytest.mark.parametrize("bounds, side", [((0.5, 1.0), "upper"), ((20.0, 100.0), "lower")])
    def test_interval_is_censored_at_the_estimation_bound_that_held_theta(self, bounds, side):
        # True theta 4 lies outside both windows, so the estimate stops on a bound.
        rng = np.random.default_rng(12)
        X = pd.DataFrame({"x": rng.uniform(-1.0, 1.0, 2000)})
        mu = np.exp(0.5 + 0.4 * X["x"].to_numpy())
        y = rng.negative_binomial(4.0, 4.0 / (4.0 + mu)).astype(float)
        model = SuperGLM(
            family=NegativeBinomial(theta="auto"),
            penalty=GroupLasso(lambda1=0.0),
            features={"x": Numeric()},
        )
        with pytest.warns(NBThetaBoundWarning):
            result = model.estimate_theta(X, y, theta_bounds=bounds)
        interval = result.interval(0.05)

        held = bounds[0] if side == "lower" else bounds[1]
        assert result.theta_hat == held
        if side == "upper":
            assert interval.upper == pytest.approx(held, rel=1e-12) and interval.upper_censored
            assert interval.lower < held and not interval.lower_censored
        else:
            assert interval.lower == pytest.approx(held, rel=1e-12) and interval.lower_censored
            assert interval.upper > held and not interval.upper_censored

    def test_an_unsettled_alternation_inverts_from_the_fixed_mean_optimum(self, monkeypatch):
        """One mean fit: theta_hat is the alternation's first iterate, not the optimum.

        Imperfect convergence is disclosed, not refused: the interval is inverted
        from the published mean's own profile optimum, and a caution says so.
        """
        import functools

        import superglm.profiling.nb as nb_module

        monkeypatch.setattr(
            nb_module,
            "estimate_nb_theta",
            functools.partial(nb_module.estimate_nb_theta, maxiter=1),
        )
        rng = np.random.default_rng(7)
        X = pd.DataFrame({"x": rng.normal(size=600)})
        mu = np.exp(0.4 + 0.5 * X["x"].to_numpy())
        y = rng.negative_binomial(2.0, 2.0 / (2.0 + mu)).astype(float)
        model = SuperGLM(
            family=NegativeBinomial(theta="auto"),
            penalty=GroupLasso(lambda1=0.0),
            features={"x": Numeric()},
        )
        with pytest.warns(UserWarning, match="not the optimum at the published mean"):
            result = model.estimate_theta(X, y, ci_alpha=0.05)
        assert not result.converged
        published = model._nb_profile_result
        root = solve_theta(
            published._y,
            published._mu,
            published._weights,
            published.theta_hat,
            weight_semantics=published._weight_semantics,
            bounds=(1e-8, 1e8),
        ).theta
        # The interval is inverted from the fixed-mean score root, to its tolerance,
        # which the unsettled theta_hat is not.
        assert math.exp(published._optimum()[0]) == pytest.approx(root, rel=4e-8)
        assert published.theta_hat != pytest.approx(root, rel=4e-8)
        with pytest.warns(UserWarning, match="not the optimum at the published mean"):
            interval = published.interval(0.05)
        assert interval.lower < root < interval.upper
        assert "CI not computed" not in str(model.summary())

    def test_a_near_poisson_upper_side_is_reported_censored(self):
        from superglm.export.summary import build_summary_export_payload

        # True theta 50 on 400 rows: theta_hat is interior, but the data cannot
        # reject Poisson, so the statistic stays under its cutoff up to the end
        # of the searched range, max(500, 100 theta_hat).
        rng = np.random.default_rng(3)
        X = pd.DataFrame({"x": rng.uniform(-1.0, 1.0, 400)})
        mu = np.exp(0.5 + 0.4 * X["x"].to_numpy())
        y = rng.negative_binomial(50.0, 50.0 / (50.0 + mu)).astype(float)
        model = SuperGLM(
            family=NegativeBinomial(theta="auto"),
            penalty=GroupLasso(lambda1=0.0),
            features={"x": Numeric()},
        )
        result = model.estimate_theta(X, y)
        assert result.converged and not result.warnings

        # The summary reports the censoring; it does not raise it.
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message=".*censored.*")
            summary = model.summary()
        with pytest.warns(UserWarning, match="censored at its upper end") as raised:
            interval = model._nb_profile_result.interval(0.05)
        assert raised[0].filename == __file__
        assert interval.upper_censored and not interval.lower_censored
        assert interval.upper == pytest.approx(max(500.0, 100.0 * result.theta_hat), rel=1e-12)
        assert model._nb_profile_result.warnings == [
            f"the 95% interval for theta is censored at its upper end theta={interval.upper:.6g}, "
            "where its search stopped: the likelihood-ratio statistic does not reach its cutoff "
            "there, so the interval may extend beyond it."
        ]
        assert summary._info["nb_theta_ci_status"] == "censored"
        theta_row = next(line for line in str(summary).splitlines() if "Theta:" in line)
        assert "] censored" in theta_row
        assert "] censored" in summary._repr_html_()
        assert model.metrics(X, y).summary()._info["nb_theta_ci_status"] == "censored"
        overview = {
            (row.section, row.metric): row.value
            for row in build_summary_export_payload(model).overview
        }
        assert overview[("Distribution Profile", "NB2 Theta CI Status")] == "censored"

    def test_an_alternation_out_of_steps_warns_and_reports_unconverged(self):
        rng = np.random.default_rng(12)
        X = pd.DataFrame({"x": rng.uniform(-1.0, 1.0, 2000)})
        mu = np.exp(0.5 + 0.4 * X["x"].to_numpy())
        y = rng.negative_binomial(4.0, 4.0 / (4.0 + mu)).astype(float)
        model = SuperGLM(
            family=NegativeBinomial(theta="auto"),
            penalty=GroupLasso(lambda1=0.0),
            features={"x": Numeric()},
        )
        # The first step moves theta from its seed 1 to near 4, far past xatol.
        with pytest.warns(UserWarning, match="did not settle in 1 mean fits") as caught:
            result = estimate_nb_theta(model, X, y, maxiter=1)
        assert not result.converged
        assert result.warnings == [str(w.message) for w in caught]

    @pytest.mark.parametrize("xatol", [0.0, -1e-3, np.nan, np.inf])
    def test_invalid_xatol_is_rejected_before_profile_work(self, monkeypatch, xatol):
        from superglm.profiling import nb as nb_module

        X = pd.DataFrame({"x": np.linspace(-1.0, 1.0, 24)})
        y = np.resize(np.array([1.0, 2.0, 3.0]), len(X))
        model = SuperGLM(
            family=NegativeBinomial(theta="auto"),
            penalty=GroupLasso(lambda1=0.0),
            features={"x": Numeric()},
        )

        def unexpected_profile(*args, **kwargs):
            raise AssertionError("profile work must not start")

        monkeypatch.setattr(nb_module, "estimate_nb_theta", unexpected_profile)

        with pytest.raises(ValueError, match="xatol must be finite and strictly positive"):
            model.estimate_theta(X, y, xatol=xatol)

    def test_estimate_theta_inherit_preserves_reml_final_refit(self, monkeypatch):
        X = pd.DataFrame({"x": np.linspace(-1.0, 1.0, 80)})
        y = np.resize(np.array([1.0, 2.0, 3.0, 4.0]), len(X))
        model = SuperGLM(
            family=NegativeBinomial(theta="auto"),
            penalty=GroupLasso(lambda1=0.0),
            features={"x": Numeric()},
        )
        model._last_fit_meta = {"method": "fit_reml"}
        result = NBProfileResult(theta_hat=2.5, nll=1.2, converged=True)
        configured_family = model._family_config
        configured_penalty = model._penalty_config
        configured_model = model._config
        configured_revision = model._config_revision

        def fake_profile(model_arg, X_arg, y_arg, sample_weight=None, offset=None, **kwargs):
            assert model_arg is not model
            assert model_arg.family.theta == "auto"
            assert isinstance(X_arg, EagerFrame)
            assert X_arg.native is X
            np.testing.assert_array_equal(y_arg, y)
            return result

        monkeypatch.setattr("superglm.profiling.nb.estimate_nb_theta", fake_profile)

        returned = model.estimate_theta(X, y, fit_mode="inherit")

        assert returned is not result
        assert model._last_fit_meta["method"] == "fit_reml"
        assert model._config is not configured_model
        assert model._family_config is not configured_family
        assert model._penalty_config is configured_penalty
        assert model._config_revision == configured_revision + 1
        assert model._config.family.theta == pytest.approx(2.5)
        assert model.family.theta == pytest.approx(2.5)
        assert model.theta_ == pytest.approx(2.5)
        refit = model.clone_unfitted()
        assert refit.family.theta == pytest.approx(2.5)
        refit.fit(X, y)
        assert refit.family.theta == pytest.approx(2.5)
        assert refit.theta_ == pytest.approx(2.5)
        assert model._nb_profile_result is not result
        assert returned is not model._nb_profile_result
        assert model._fit_state.projections["_nb_profile_result"] is model._nb_profile_result

    def test_auto_theta_reml_uses_zero_selection_regime_when_unconfigured(self):
        rng = np.random.default_rng(20260718)
        x = np.linspace(-1.0, 1.0, 100)
        mu = np.exp(0.2 + 0.25 * x)
        y = rng.poisson(rng.gamma(shape=3.0, scale=mu / 3.0)).astype(np.float64)
        X = pd.DataFrame({"x": x})
        model = SuperGLM(
            family=NegativeBinomial(theta="auto"),
            selection_penalty=None,
            features={"x": Spline(n_knots=5)},
        )

        model.fit_reml(X, y, max_reml_iter=1, max_pirls_iter=20)

        assert model.selection_penalty is None
        assert model.selection_penalty_ == pytest.approx(0.0)
        assert model.theta_ > 0.0

    def test_auto_theta_reml_with_random_effect_keeps_structured_credibility(self):
        rng = np.random.default_rng(20260727)
        n_levels = 60
        repeats = 8
        codes = np.repeat(np.arange(n_levels), repeats)
        x = rng.normal(size=len(codes))
        effects = rng.normal(scale=0.3, size=n_levels)
        mu = np.exp(0.2 + 0.15 * x + effects[codes])
        theta = 2.5
        y = rng.negative_binomial(theta, theta / (theta + mu)).astype(np.float64)
        X = pd.DataFrame(
            {
                "x": x,
                "group": np.array([f"g{code}" for code in codes], dtype=object),
            }
        )
        model = SuperGLM(
            family=NegativeBinomial(theta="auto"),
            selection_penalty=None,
            features={"x": Numeric(), "group": RandomEffect()},
            direct_solve="auto",
        ).fit_reml(
            X,
            y,
            max_reml_iter=2,
            max_pirls_iter=60,
            runtime_validation="skip",
        )

        assert np.isfinite(model.theta_)
        assert model.theta_ > 0.0
        assert model.result.direct_backend == "structured"

    def test_auto_theta_reml_with_factor_smooth_matches_gram_credibility(self):
        rng = np.random.default_rng(20260727)
        n_levels = 8
        repeats = 30
        codes = np.repeat(np.arange(n_levels), repeats)
        x = np.tile(np.linspace(-1.0, 1.0, repeats), n_levels)
        slopes = rng.normal(scale=0.2, size=n_levels)
        mu = np.exp(0.1 + 0.25 * x + slopes[codes] * x)
        theta = 3.0
        y = rng.negative_binomial(theta, theta / (theta + mu)).astype(np.float64)
        X = pd.DataFrame(
            {
                "x": x,
                "group": np.array([f"g{code}" for code in codes], dtype=object),
            }
        )
        policies = {
            "wiggle": LambdaPolicy.fixed(1.2),
            "null_0": LambdaPolicy.fixed(0.8),
            "null_1": LambdaPolicy.fixed(0.8),
        }

        def fit(direct_solve: str) -> SuperGLM:
            return SuperGLM(
                family=NegativeBinomial(theta="auto"),
                selection_penalty=None,
                features={
                    "x": Spline(
                        k=5,
                        lambda_policy=LambdaPolicy.fixed(1.1),
                    )
                },
                interactions=[
                    FactorSmooth(
                        "x",
                        group="group",
                        k=5,
                        lambda_policy=policies,
                    )
                ],
                direct_solve=direct_solve,
            ).fit_reml(
                X,
                y,
                max_reml_iter=2,
                max_pirls_iter=60,
                runtime_validation="skip",
            )

        structured = fit("auto")
        gram = fit("gram")

        assert structured.result.direct_backend == "structured"
        assert structured.theta_ == pytest.approx(gram.theta_, rel=1.0e-7)
        np.testing.assert_allclose(
            structured.predict(X),
            gram.predict(X),
            rtol=2.0e-6,
            atol=2.0e-7,
        )

    def test_nb_profile_bcd_forwards_configured_smoothing(self, monkeypatch):
        rng = np.random.default_rng(20260719)
        x = np.linspace(-1.0, 1.0, 90)
        y = rng.poisson(np.exp(0.1 + 0.2 * x)).astype(np.float64)
        X = pd.DataFrame({"x": x})
        model = SuperGLM(
            family=NegativeBinomial(theta=2.0),
            penalty=GroupLasso(lambda1=0.02),
            spline_penalty=0.75,
            features={"x": Spline(n_knots=5)},
        )
        from superglm.model import fit_ops

        real_fit_pirls = fit_ops.fit_pirls
        seen_lambda2 = []

        def recording_fit_pirls(*args, **kwargs):
            seen_lambda2.append(kwargs.get("lambda2"))
            return real_fit_pirls(*args, **kwargs)

        monkeypatch.setattr(fit_ops, "fit_pirls", recording_fit_pirls)

        estimate_nb_theta(model, X, y, maxiter=1)

        assert seen_lambda2 == [pytest.approx(0.75)]

    def test_estimate_theta_publishes_owned_detached_profile_result(self):
        rng = np.random.default_rng(20260720)
        x = np.linspace(-1.0, 1.0, 100)
        y = rng.poisson(np.exp(0.1 + 0.2 * x)).astype(np.float64)
        weights = np.linspace(0.5, 1.5, len(x))
        X = pd.DataFrame({"x": x})
        model = SuperGLM(
            family=NegativeBinomial(theta="auto"),
            selection_penalty=0.0,
            features={"x": Numeric()},
        )

        returned = model.estimate_theta(X, y, sample_weight=weights)
        installed = model._nb_profile_result
        installed_y = installed._y.copy()
        installed_weights = installed._weights.copy()

        assert returned is not installed
        assert returned.theta_hat == pytest.approx(installed.theta_hat)
        assert not np.shares_memory(installed._y, y)
        assert not np.shares_memory(installed._weights, weights)
        np.testing.assert_allclose(installed._mu, model._fit_mu, rtol=0.0, atol=0.0)
        for values in (installed._y, installed._mu, installed._weights):
            assert not values.flags.writeable
        # An interval computed through the returned result stays out of the model.
        returned.interval(0.2)
        assert 0.2 not in installed._ci_cache

        y[0] += 20.0
        weights[0] *= 20.0
        np.testing.assert_array_equal(installed._y, installed_y)
        np.testing.assert_array_equal(installed._weights, installed_weights)

    def test_auto_theta_flow(self):
        """nb_theta='auto' triggers profile estimation in fit()."""
        rng = np.random.default_rng(42)
        n = 2000
        theta_true = 5.0
        y = _generate_nb2(n, mu=5.0, theta=theta_true, rng=rng)
        X = pd.DataFrame({"dummy": np.ones(n)})

        model = SuperGLM(
            family=NegativeBinomial(theta="auto"),
            penalty=GroupLasso(lambda1=0.0),
            features={"dummy": Numeric()},
        )
        model.fit(X, y)

        # Configuration intent stays automatic; the learned value is fitted state.
        assert model.family.theta == "auto"
        assert model.theta_ > 0
        assert model.result.converged

    @pytest.mark.parametrize("direct_solve", ["auto", "structured"])
    def test_reml_theta_is_the_score_root_at_the_penalized_mean(self, direct_solve):
        """The alternation's mean fits carry the RandomEffect penalty that fit_reml publishes."""
        rng = np.random.default_rng(422)
        g = np.repeat(np.arange(20), 10)
        x = rng.normal(size=g.size)
        mu = np.exp(0.1 + 0.2 * x + rng.normal(0.0, 0.8, 20)[g])
        y = rng.negative_binomial(3.0, 3.0 / (3.0 + mu)).astype(float)
        X = pd.DataFrame({"x": x, "g": g.astype(str)})
        # The alternation penalises every REML component at spline_penalty; set
        # equal to g's fixed lambda, fit_reml publishes the alternation's own mean.
        model = SuperGLM(
            family=NegativeBinomial(1.0),
            selection_penalty=0.0,
            spline_penalty=50.0,
            direct_solve=direct_solve,
            features={"x": Numeric(), "g": RandomEffect(lambda_policy=LambdaPolicy.fixed(50.0))},
        )

        result = model.estimate_theta(X, y, fit_mode="reml", xatol=1e-8)

        # theta_hat is the root at that mean rounded to six significant digits; the
        # published mean sits at the rounded theta, which moves the root by less
        # than the rounding itself (the alternation contracts), plus xatol.
        rounding = 0.5 * 10.0 ** (math.floor(math.log10(result.theta_hat)) - 5) / result.theta_hat
        root = solve_theta(
            y,
            model.predict(X),
            np.ones(y.size),
            result.theta_hat,
            weight_semantics=PRIOR_WEIGHTS,
            bounds=(1e-8, 1e8),
        )
        assert result.converged
        assert root.theta == pytest.approx(result.theta_hat, rel=2.0 * rounding + 1e-8)


# =====================================================================
# TestNB2QuantileResiduals
# =====================================================================


class TestNB2QuantileResiduals:
    def test_approx_normal(self):
        """Quantile residuals should be ~N(0,1) for well-specified NB2."""
        rng = np.random.default_rng(42)
        n = 5000
        theta = 5.0
        mu_true = 5.0
        y = _generate_nb2(n, mu=mu_true, theta=theta, rng=rng)
        X = pd.DataFrame({"dummy": np.ones(n)})

        model = SuperGLM(
            family=NegativeBinomial(theta=theta),
            penalty=GroupLasso(lambda1=0.0),
            features={"dummy": Numeric()},
        )
        model.fit(X, y)

        metrics = model.metrics(X, y)
        qr = metrics.residuals("quantile")

        # Should be approximately N(0,1)
        assert abs(qr.mean()) < 0.15
        assert abs(qr.std() - 1.0) < 0.15


# =====================================================================
# TestNB2MetricsSummary
# =====================================================================


class TestNB2MetricsSummary:
    def test_summary_works(self):
        rng = np.random.default_rng(42)
        n = 1000
        y = _generate_nb2(n, mu=5.0, theta=3.0, rng=rng)
        X = pd.DataFrame({"dummy": np.ones(n)})

        model = SuperGLM(
            family=NegativeBinomial(theta=3.0),
            penalty=GroupLasso(lambda1=0.0),
            features={"dummy": Numeric()},
        )
        model.fit(X, y)

        metrics = model.metrics(X, y)
        summary = metrics.summary()
        text = str(summary)
        assert "NegativeBinomial" in text or "Neg. Binomial" in text


# =====================================================================
# TestNB2Sklearn
# =====================================================================


class TestNB2Sklearn:
    def test_fit_predict(self):
        rng = np.random.default_rng(42)
        n = 1000
        x = rng.normal(0, 1, n)
        mu = np.exp(1.0 + 0.3 * x)
        y = _generate_nb2(n, mu=mu, theta=5.0, rng=rng)
        X = pd.DataFrame({"x": x})

        reg = SuperGLMRegressor(
            family=NegativeBinomial(theta=5.0),
            selection_penalty=0.0,
        )
        reg.fit(X, y)
        pred = reg.predict(X)

        assert pred.shape == (n,)
        assert np.all(pred > 0)


# =====================================================================
# TestNB2Validation
# =====================================================================


class TestNB2InvalidTheta:
    def test_zero(self):
        with pytest.raises(ValueError, match="must be > 0"):
            NegativeBinomial(theta=0.0)

    def test_negative(self):
        with pytest.raises(ValueError, match="must be > 0"):
            NegativeBinomial(theta=-1.0)


class TestNB2ResolveDistribution:
    def test_resolve_object(self):
        dist = resolve_distribution(NegativeBinomial(theta=5.0))
        assert isinstance(dist, NegativeBinomial)
        assert dist.theta == 5.0

    def test_resolve_missing_theta(self):
        with pytest.raises(ValueError, match="requires parameters"):
            resolve_distribution("negative_binomial")

    def test_resolve_passthrough(self):
        nb = NegativeBinomial(theta=3.0)
        assert resolve_distribution(nb) is nb
