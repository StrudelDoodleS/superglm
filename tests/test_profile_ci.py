"""Tests for profile likelihood CIs (NB theta and Tweedie p)."""

import warnings
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2

import superglm.profiling.tweedie as tweedie_module
from superglm import SuperGLM
from superglm.distributions import NegativeBinomial, Tweedie
from superglm.features.numeric import Numeric
from superglm.profiling.nb import estimate_nb_theta, profile_ci_theta
from superglm.profiling.tweedie import estimate_tweedie_p


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
            selection_penalty=0.001,
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

        ci_lo, ci_hi = profile_ci_theta(y, mu, weights, theta, weight_semantics="frequency")
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
            selection_penalty=0.001,
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
            selection_penalty=0.001,
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
            selection_penalty=0.001,
            features={"x": Numeric()},
        )
        result = estimate_nb_theta(model, X, y)

        fig, ax = plt.subplots()
        returned_fig = result.profile_plot(ax=ax)
        assert returned_fig is fig
        plt.close(fig)


class TestTweedieProfileCI:
    @staticmethod
    def _bare_result(objective, *, p_hat=0.5, nll=0.0, ll_scale=1.0):
        trace = pd.DataFrame({"p": [p_hat], "nll": [nll]})
        result = tweedie_module.TweedieProfileResult(
            p_hat=p_hat,
            phi_hat=1.0,
            nll=nll,
            n_evaluations=1,
            converged=True,
            method="brent",
            phi_method="mle",
            search_trace=trace,
            _objective=objective,
            _ll_scale=ll_scale,
        )
        result._ci_p_range = (0.0, 1.0)
        result._ci_seed_points = (p_hat,)
        return result

    @staticmethod
    def _trace_plot_result(*, phi_method="mle"):
        def unexpected_objective(p):
            raise AssertionError(f"trace_plot must not evaluate the objective at p={p}")

        return tweedie_module.TweedieProfileResult(
            p_hat=1.5,
            phi_hat=1.0,
            nll=2.0,
            n_evaluations=3,
            converged=True,
            method="brent",
            phi_method=phi_method,
            search_trace=pd.DataFrame(
                {
                    "p": [1.8, 1.2, 1.5],
                    "nll": [2.1, 2.2, 2.0],
                }
            ),
            _objective=unexpected_objective,
            _ll_scale=10.0,
            _evaluation_count=lambda: 3,
        )

    @staticmethod
    def _trace_curve(ax):
        curves = [line for line in ax.lines if line.get_label().startswith("Search evaluations")]
        assert len(curves) == 1
        return curves[0]

    def test_trace_plot_uses_sorted_cached_trace_without_side_effects(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        result = self._trace_plot_result()
        result._ci_cache[0.05] = (1.4, 1.6)
        result._ci_details_cache[0.05] = SimpleNamespace(source="sentinel")
        trace = result.search_trace
        trace_snapshot = trace.copy(deep=True)
        ci_cache = result._ci_cache
        ci_cache_snapshot = ci_cache.copy()
        ci_details_cache = result._ci_details_cache
        ci_details_cache_snapshot = ci_details_cache.copy()
        n_total_evaluations = result.n_total_evaluations

        fig = result.trace_plot()

        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        curve = self._trace_curve(ax)
        np.testing.assert_allclose(curve.get_xdata(), [1.2, 1.5, 1.8])
        np.testing.assert_allclose(curve.get_ydata(), [4.0, 0.0, 2.0])
        assert ax.get_xlabel() == "p"
        assert ax.get_ylabel() == "Profile deviance"
        assert result.n_total_evaluations == n_total_evaluations
        assert result.search_trace is trace
        pd.testing.assert_frame_equal(result.search_trace, trace_snapshot)
        assert result._ci_cache is ci_cache
        assert result._ci_cache == ci_cache_snapshot
        assert result._ci_details_cache is ci_details_cache
        assert result._ci_details_cache == ci_details_cache_snapshot
        plt.close(fig)

    def test_trace_plot_uses_supplied_axes_and_neutral_pearson_wording(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        result = self._trace_plot_result(phi_method="pearson")
        fig, ax = plt.subplots()

        returned_fig = result.trace_plot(ax=ax)

        assert returned_fig is fig
        assert ax.get_ylabel() == "Profile objective difference"
        assert ax.get_title()
        assert "likelihood" not in ax.get_title().lower()
        plt.close(fig)

    @pytest.mark.parametrize(
        "nonfinite_column",
        ["p", "nll"],
        ids=["nonfinite-p", "nonfinite-nll"],
    )
    def test_trace_plot_filters_nonfinite_trace_rows(self, nonfinite_column):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        result = self._trace_plot_result()
        result.search_trace.loc[1, nonfinite_column] = np.nan
        trace_snapshot = result.search_trace.copy(deep=True)

        fig = result.trace_plot()

        curve = self._trace_curve(fig.axes[0])
        np.testing.assert_allclose(curve.get_xdata(), [1.5, 1.8])
        np.testing.assert_allclose(curve.get_ydata(), [0.0, 2.0])
        pd.testing.assert_frame_equal(result.search_trace, trace_snapshot)
        plt.close(fig)

    @pytest.mark.parametrize(
        "trace",
        [
            pd.DataFrame({"p": pd.Series(dtype=float), "nll": pd.Series(dtype=float)}),
            pd.DataFrame({"p": [np.nan, np.inf], "nll": [np.inf, np.nan]}),
        ],
        ids=["empty", "all-nonfinite"],
    )
    def test_trace_plot_requires_finite_p_and_nll(self, trace):
        result = self._trace_plot_result()
        result.search_trace = trace

        with pytest.raises(RuntimeError, match="finite p/nll"):
            result.trace_plot()

    def test_tweedie_ci_detail_records_are_public(self):
        from superglm import (
            TweedieProfileCIDensityProvenance,
            TweedieProfileCIDetails,
            TweedieProfileCIEndpoint,
            TweedieProfileCIEvaluation,
        )

        assert TweedieProfileCIDetails is tweedie_module.TweedieProfileCIDetails
        assert TweedieProfileCIDensityProvenance is tweedie_module.TweedieProfileCIDensityProvenance
        assert TweedieProfileCIEndpoint is tweedie_module.TweedieProfileCIEndpoint
        assert TweedieProfileCIEvaluation is tweedie_module.TweedieProfileCIEvaluation

    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"alpha": 0.0}, "alpha"),
            ({"alpha": 1.0}, "alpha"),
            ({"alpha": np.nan}, "alpha"),
            ({"p_hat": np.nan}, "p_hat"),
            ({"nll_hat": np.inf}, "nll_hat"),
            ({"ll_scale": 0.0}, "ll_scale"),
            ({"ll_scale": np.inf}, "ll_scale"),
            ({"p_range": (0.5, 0.5)}, "p_range"),
            ({"p_range": (0.8, 0.2)}, "p_range"),
            ({"p_range": (np.nan, 1.0)}, "p_range"),
            ({"p_range": (0.0, 0.4)}, "contain p_hat"),
        ],
    )
    def test_tweedie_ci_validates_inputs_before_objective(self, overrides, match):
        calls = []

        def objective(p):
            calls.append(p)
            return 0.0

        kwargs = {
            "objective": objective,
            "p_hat": 0.5,
            "nll_hat": 0.0,
            "ll_scale": 1.0,
            "alpha": 0.05,
            "p_range": (0.0, 1.0),
        }
        kwargs.update(overrides)

        with pytest.raises(ValueError, match=match):
            tweedie_module.profile_ci_p(**kwargs)

        assert calls == []

    def test_detailed_ci_distinguishes_truncation_from_boundary_root(self):
        shallow = tweedie_module._profile_ci_p_detailed(
            lambda p: 0.01 * (p - 0.5) ** 2,
            p_hat=0.5,
            nll_hat=0.0,
            ll_scale=1.0,
            p_range=(0.0, 1.0),
        )

        assert shallow.interval == (0.0, 1.0)
        assert shallow.lower.status == "truncated"
        assert shallow.upper.status == "truncated"
        assert shallow.lower.at_range_boundary
        assert shallow.upper.at_range_boundary

        cutoff = float(chi2.ppf(0.95, 1))

        def root_at_bounds(p):
            if p in (0.0, 1.0):
                return cutoff / 2.0
            return 0.0

        exact = tweedie_module._profile_ci_p_detailed(
            root_at_bounds,
            p_hat=0.5,
            nll_hat=0.0,
            ll_scale=1.0,
            p_range=(0.0, 1.0),
        )

        assert exact.interval == (0.0, 1.0)
        assert exact.lower.status == "root_found"
        assert exact.upper.status == "root_found"
        assert exact.lower.at_range_boundary
        assert exact.upper.at_range_boundary

    def test_ci_without_authoritative_records_marks_density_provenance_unavailable(self):
        details = tweedie_module._profile_ci_p_detailed(
            lambda p: 0.01 * (p - 0.5) ** 2,
            p_hat=0.5,
            nll_hat=0.0,
            ll_scale=1.0,
            p_range=(0.0, 1.0),
        )

        assert details.density_provenance == ()
        assert details.density_method is None
        assert details.density_exact is None

    def test_detailed_ci_finds_nearest_connected_crossing_on_both_sides(self):
        cutoff = float(chi2.ppf(0.95, 1))

        def disconnected_profile(p):
            distance = abs(p - 0.5)
            triangle = max(0.0, 1.0 - abs(distance - 0.2) / 0.1)
            lr_statistic = 2.0 * cutoff * triangle
            return lr_statistic / 2.0

        details = tweedie_module._profile_ci_p_detailed(
            disconnected_profile,
            p_hat=0.5,
            nll_hat=0.0,
            ll_scale=1.0,
            p_range=(0.0, 1.0),
        )

        assert details.lower.status == "root_found"
        assert details.upper.status == "root_found"
        assert details.lower.value == pytest.approx(0.35, abs=2e-6)
        assert details.upper.value == pytest.approx(0.65, abs=2e-6)
        assert details.lower.value != 0.0
        assert details.upper.value != 1.0

    def test_connected_ci_does_not_probe_invalid_remote_bounds_after_crossing(self):
        cutoff = float(chi2.ppf(0.95, 1))
        calls = []

        def objective(p):
            calls.append(float(p))
            if p in (0.0, 1.0):
                raise RuntimeError("remote bound is unstable")
            return 0.5 * cutoff * ((p - 0.5) / 0.15) ** 2

        details = tweedie_module._profile_ci_p_detailed(
            objective,
            p_hat=0.5,
            nll_hat=0.0,
            ll_scale=1.0,
            p_range=(0.0, 1.0),
        )

        assert calls[0] == 0.5
        assert 0.0 not in calls
        assert 1.0 not in calls
        assert details.lower.value == pytest.approx(0.35, abs=2e-4)
        assert details.upper.value == pytest.approx(0.65, abs=2e-4)

    def test_truncated_ci_reaches_and_propagates_invalid_remote_bound(self):
        calls = []

        def objective(p):
            calls.append(float(p))
            if p == 0.0:
                raise RuntimeError("lower bound is unstable")
            return 0.0

        with pytest.raises(RuntimeError, match=r"p=0.*lower bound is unstable"):
            tweedie_module._profile_ci_p_detailed(
                objective,
                p_hat=0.5,
                nll_hat=0.0,
                ll_scale=1.0,
                p_range=(0.0, 1.0),
            )

        assert calls[0] == 0.5
        assert calls[-1] == 0.0

    def test_ci_nonfinite_and_objective_failures_are_not_truncation(self):
        def nonfinite_at_lower_bound(p):
            return np.nan if p == 0.0 else 0.0

        with pytest.raises(ValueError, match=r"non-finite.*p=0(?:\.0)?"):
            tweedie_module._profile_ci_p_detailed(
                nonfinite_at_lower_bound,
                0.5,
                0.0,
                1.0,
                p_range=(0.0, 1.0),
            )

        def failure_at_probe(p):
            if p == 0.25:
                raise ValueError("deliberate fit failure")
            return 0.0

        with pytest.raises(ValueError, match=r"p=0\.25.*deliberate fit failure"):
            tweedie_module._profile_ci_p_detailed(
                failure_at_probe,
                0.5,
                0.0,
                1.0,
                p_range=(0.0, 1.0),
            )

        def runtime_failure_at_probe(p):
            if p == 0.25:
                raise RuntimeError("solver exploded")
            return 0.0

        with pytest.raises(RuntimeError, match=r"p=0\.25.*solver exploded"):
            tweedie_module._profile_ci_p_detailed(
                runtime_failure_at_probe,
                0.5,
                0.0,
                1.0,
                p_range=(0.0, 1.0),
            )

    def test_ci_brentq_failure_after_sign_change_is_numerical_error(self, monkeypatch):
        cutoff = float(chi2.ppf(0.95, 1))

        def objective(p):
            return 0.5 * cutoff * ((p - 0.5) / 0.23) ** 2

        def fail_brentq(*args, **kwargs):
            raise ValueError("forced root failure")

        monkeypatch.setattr(tweedie_module, "brentq", fail_brentq)

        with pytest.raises(RuntimeError, match=r"numerical CI root.*forced root failure"):
            tweedie_module._profile_ci_p_detailed(
                objective,
                0.5,
                0.0,
                1.0,
                p_range=(0.0, 1.0),
            )

    def test_ci_objective_failure_inside_brentq_stays_an_objective_error(self, monkeypatch):
        cutoff = float(chi2.ppf(0.95, 1))

        def objective(p):
            if p == 0.37:
                raise RuntimeError("interpolated fit failed")
            return 0.5 * cutoff * ((p - 0.5) / 0.23) ** 2

        def probe_failing_point(function, *args, **kwargs):
            function(0.37)
            raise AssertionError("unreachable")

        monkeypatch.setattr(tweedie_module, "brentq", probe_failing_point)

        with pytest.raises(
            RuntimeError,
            match=r"^Tweedie profile CI objective failed at p=0\.37.*interpolated fit failed",
        ):
            tweedie_module._profile_ci_p_detailed(
                objective,
                0.5,
                0.0,
                1.0,
                p_range=(0.0, 1.0),
            )

    def test_ci_rejects_materially_better_profile_probe_and_caches_nothing(self):
        def objective(p):
            if p == 0.25:
                return -0.01
            return 0.0

        result = self._bare_result(objective)

        with pytest.raises(RuntimeError, match="found a better profile value; rerun/expand search"):
            result.ci()

        assert result._ci_cache == {}
        assert result._ci_details_cache == {}

    @pytest.mark.parametrize(
        ("center_nll", "match"),
        [(0.01, "inconsistent.*nll_hat"), (-0.01, "better profile value")],
    )
    def test_ci_rejects_center_objective_mismatch_and_caches_nothing(self, center_nll, match):
        calls = []

        def objective(p):
            calls.append(float(p))
            return center_nll if p == 0.5 else 0.0

        result = self._bare_result(objective)

        with pytest.raises(RuntimeError, match=match):
            result.ci()

        assert calls == [0.5]
        assert result._ci_cache == {}
        assert result._ci_details_cache == {}

    def test_ci_ignores_only_immaterial_better_profile_difference(self):
        def objective(p):
            if p == 0.25:
                return -1e-4
            return 0.0

        details = tweedie_module._profile_ci_p_detailed(
            objective,
            0.5,
            0.0,
            1.0,
            p_range=(0.0, 1.0),
        )

        assert details.interval == (0.0, 1.0)

    def test_ci_scan_has_one_full_range_probe_budget(self):
        completed = {0.1: 0.0, 0.3: 0.0, 0.5: 0.0, 0.7: 0.0, 0.9: 0.0}

        def objective(p):
            completed.setdefault(float(p), 0.0)
            return completed[float(p)]

        before = len(completed)
        details = tweedie_module._profile_ci_p_detailed(
            objective,
            0.5,
            0.0,
            1.0,
            p_range=(0.0, 1.0),
            seed_points=tuple(completed),
            evaluation_count=lambda: len(completed),
        )

        assert details.n_new_evaluations == len(completed) - before
        assert details.n_new_evaluations <= 16
        evaluated_p = sorted(point.p for point in details.evaluations)
        assert max(np.diff(evaluated_p)) <= 1.0 / 16.0 + 1e-12

    def test_ci_count_snapshots_tuple_identity_and_failed_cache(self, monkeypatch):
        completed = {0.5: 0.0}

        def objective(p):
            completed.setdefault(float(p), 0.0)
            return completed[float(p)]

        result = self._bare_result(objective)
        result._evaluation_count = lambda: len(completed)
        trace_snapshot = result.search_trace.copy(deep=True)

        interval = (0.25, 0.75)
        details = SimpleNamespace(
            alpha=0.05,
            interval=interval,
            n_new_evaluations=2,
            evaluations=(),
            warnings=(),
        )

        def two_probe_ci(objective, *args, **kwargs):
            objective(0.25)
            objective(0.75)
            return details

        monkeypatch.setattr(tweedie_module, "_profile_ci_p_detailed", two_probe_ci, raising=False)

        assert result.n_total_evaluations == 1
        first = result.ci(alpha=0.05)
        assert first is interval
        assert result._ci_cache[0.05] is first
        assert result.ci(alpha=0.05) is first
        assert result.ci_details(alpha=0.05) is details
        assert result.n_evaluations == 1
        assert result.n_total_evaluations == 3
        assert result.n_post_search_evaluations == 2
        pd.testing.assert_frame_equal(result.search_trace, trace_snapshot)

        failed = self._bare_result(objective)
        failed._evaluation_count = lambda: len(completed)

        def failing_ci(objective, *args, **kwargs):
            objective(0.2)
            raise RuntimeError("CI failed")

        monkeypatch.setattr(tweedie_module, "_profile_ci_p_detailed", failing_ci)
        with pytest.raises(RuntimeError, match="CI failed"):
            failed.ci(alpha=0.1)
        assert failed._ci_cache == {}
        assert failed._ci_details_cache == {}

    def test_ci_details_rejects_legacy_tuple_without_detail_record(self):
        result = self._bare_result(lambda p: 0.0)
        result._ci_cache[0.05] = (0.25, 0.75)

        with pytest.raises(RuntimeError, match="details.*pre-populated"):
            result.ci_details(alpha=0.05)

    @pytest.mark.parametrize("method_name", ["ci", "ci_details"])
    def test_pearson_ci_methods_reject_stale_likelihood_ratio_caches(self, method_name):
        def unexpected_objective(p):
            raise AssertionError(f"Pearson CI guard must run before objective access at p={p}")

        result = self._bare_result(unexpected_objective)
        result.phi_method = "pearson"
        stale_interval = (0.25, 0.75)
        stale_details = SimpleNamespace(interval=stale_interval)
        result._ci_cache[0.05] = stale_interval
        result._ci_details_cache[0.05] = stale_details

        with pytest.raises(RuntimeError) as exc_info:
            getattr(result, method_name)(alpha=0.05)

        message = str(exc_info.value)
        assert "exact MLE" in message
        assert "bootstrap/sandwich" in message
        assert result._ci_cache[0.05] is stale_interval
        assert result._ci_details_cache[0.05] is stale_details

    @pytest.mark.parametrize("method_name", ["ci", "ci_details"])
    def test_pearson_ci_guard_runs_before_any_cache_lookup(self, method_name):
        class UnexpectedCacheAccess(dict):
            def __contains__(self, key):
                raise AssertionError(f"Pearson CI guard must run before cache lookup for {key}")

            def __getitem__(self, key):
                raise AssertionError(f"Pearson CI guard must run before cache access for {key}")

        result = self._bare_result(
            lambda p: (_ for _ in ()).throw(
                AssertionError(f"Pearson CI guard must run before objective access at p={p}")
            )
        )
        result.phi_method = "pearson"
        result._ci_cache = UnexpectedCacheAccess({0.05: (0.25, 0.75)})
        result._ci_details_cache = UnexpectedCacheAccess({0.05: SimpleNamespace()})

        with pytest.raises(RuntimeError, match="exact MLE.*bootstrap/sandwich"):
            getattr(result, method_name)(alpha=0.05)

    @pytest.mark.parametrize(
        "field",
        ["objective_finite", "fit_converged", "phi_converged"],
    )
    def test_result_ci_rejects_invalid_winning_record(self, field):
        result = self._bare_result(lambda p: 0.0)
        setattr(result, field, False)

        with pytest.raises(RuntimeError, match=field):
            result.ci()

    def test_cached_ci_cannot_bypass_invalid_winning_record(self):
        calls = []
        result = self._bare_result(lambda p: calls.append(float(p)) or 0.0)
        cached = (0.25, 0.75)
        result._ci_cache[0.05] = cached
        result.phi_converged = False

        with pytest.raises(RuntimeError, match="phi_converged"):
            result.ci(alpha=0.05)

        assert calls == []
        assert result._ci_cache[0.05] is cached

    @pytest.mark.parametrize(
        "invalid_field",
        ["objective_finite", "fit_converged", "phi_converged"],
    )
    def test_result_ci_rejects_invalid_probed_record(self, invalid_field):
        records = {}

        def objective(p):
            phi_result = SimpleNamespace(objective_finite=True, converged=True)
            record = SimpleNamespace(
                nll=0.0,
                fit_converged=True,
                phi_result=phi_result,
            )
            if p != 0.5:
                if invalid_field == "objective_finite":
                    phi_result.objective_finite = False
                elif invalid_field == "fit_converged":
                    record.fit_converged = False
                else:
                    phi_result.converged = False
            records[float(p)] = record
            return 0.0

        result = self._bare_result(objective)
        objective(0.5)
        result._evaluation_record = lambda p: records.get(float(p))

        with pytest.raises(RuntimeError, match=rf"{invalid_field}.*p=0"):
            result.ci()

        assert result._ci_cache == {}
        assert result._ci_details_cache == {}

    def test_successful_truncation_warning_is_cached_and_emitted_once(self):
        result = self._bare_result(lambda p: 0.01 * (p - 0.5) ** 2)

        with pytest.warns(UserWarning, match="truncated"):
            interval = result.ci()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert result.ci() is interval
        assert caught == []

    def test_ci_rejects_discontinuous_false_root(self):
        cutoff = float(chi2.ppf(0.95, 1))

        def discontinuous_profile(p):
            lr_statistic = 0.0 if 0.35 < p < 0.65 else 5.0
            return lr_statistic / 2.0

        with pytest.raises(RuntimeError, match="unresolved or discontinuous LR cutoff"):
            tweedie_module._profile_ci_p_detailed(
                discontinuous_profile,
                p_hat=0.5,
                nll_hat=0.0,
                ll_scale=1.0,
                p_range=(0.0, 1.0),
            )

        assert cutoff < 5.0

    def test_ci_refines_a_steep_continuous_root(self):
        cutoff = float(chi2.ppf(0.95, 1))
        root_distance = 0.147321

        def steep_continuous_profile(p):
            lr_statistic = cutoff * np.exp(500.0 * (abs(p - 0.5) - root_distance))
            return lr_statistic / 2.0

        details = tweedie_module._profile_ci_p_detailed(
            steep_continuous_profile,
            p_hat=0.5,
            nll_hat=steep_continuous_profile(0.5),
            ll_scale=1.0,
            p_range=(0.0, 1.0),
        )

        assert details.lower.value == pytest.approx(0.5 - root_distance, abs=2e-6)
        assert details.upper.value == pytest.approx(0.5 + root_distance, abs=2e-6)
        assert details.lower.status == details.upper.status == "root_found"

    @pytest.mark.parametrize(
        ("root", "converged", "match"),
        [
            (np.nan, True, "finite"),
            (1.5, True, "bracket"),
            (0.3, False, "converge"),
        ],
    )
    def test_ci_validates_root_candidate_before_objective_evaluation(
        self, monkeypatch, root, converged, match
    ):
        cutoff = float(chi2.ppf(0.95, 1))
        calls = []

        def objective(p):
            calls.append(float(p))
            return 0.5 * cutoff * ((p - 0.5) / 0.23) ** 2

        monkeypatch.setattr(
            tweedie_module,
            "brentq",
            lambda *args, **kwargs: (root, SimpleNamespace(converged=converged)),
        )

        with pytest.raises(RuntimeError, match=match):
            tweedie_module._profile_ci_p_detailed(
                objective,
                0.5,
                0.0,
                1.0,
                p_range=(0.0, 1.0),
            )

        assert not any(
            (not np.isfinite(p)) or p == 1.5 or (not converged and p == root) for p in calls
        )

    @pytest.mark.parametrize(
        ("failure_kind", "public_error"),
        [
            ("invalid_scalar", ValueError),
            ("nonfinite", ValueError),
            ("nonfinite_lr", ValueError),
            ("invalid_record", RuntimeError),
            ("malformed_record", RuntimeError),
            ("better_basin", RuntimeError),
        ],
    )
    def test_ci_evaluation_failures_inside_brent_are_not_relabelled(
        self, monkeypatch, failure_kind, public_error
    ):
        cutoff = float(chi2.ppf(0.95, 1))

        def objective(p):
            if p == 0.37:
                if failure_kind == "invalid_scalar":
                    return np.array([1.0, 2.0])
                if failure_kind == "nonfinite":
                    return np.nan
                if failure_kind == "nonfinite_lr":
                    return np.finfo(np.float64).max
                if failure_kind == "better_basin":
                    return -0.01
            return 0.5 * cutoff * ((p - 0.5) / 0.23) ** 2

        def evaluation_record(p):
            valid = not (failure_kind == "invalid_record" and p == 0.37)
            return SimpleNamespace(
                nll=np.array([0.0, 1.0])
                if failure_kind == "malformed_record" and p == 0.37
                else 0.0,
                fit_converged=True,
                phi_result=SimpleNamespace(objective_finite=True, converged=valid),
            )

        def probe_failure(function, *args, **kwargs):
            function(0.37)
            raise AssertionError("unreachable")

        monkeypatch.setattr(tweedie_module, "brentq", probe_failure)

        with pytest.raises(public_error) as caught:
            tweedie_module._profile_ci_p_detailed(
                objective,
                0.5,
                0.0,
                1.0,
                p_range=(0.0, 1.0),
                evaluation_record=evaluation_record,
            )

        assert isinstance(caught.value, tweedie_module._TweedieProfileCIEvaluationError)
        assert "p=0.37" in str(caught.value)
        assert "numerical CI root" not in str(caught.value)

    def test_public_ci_docs_disclose_finite_connected_scan_limitation(self):
        result_doc = tweedie_module.TweedieProfileResult.ci.__doc__ or ""
        function_doc = tweedie_module.profile_ci_p.__doc__ or ""

        for doc in (result_doc, function_doc):
            assert "nearest detected connected" in doc
            assert "narrower unsampled" in doc

    def test_ci_works(self):
        """Tweedie profile CI should produce a valid interval."""
        from superglm.profiling.tweedie import generate_tweedie_cpg

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
            family=Tweedie(p=1.5),
            selection_penalty=0.001,
            features={"x": Numeric()},
        )
        result = estimate_tweedie_p(model, X, y, phi_method="mle")
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

        from superglm.profiling.tweedie import generate_tweedie_cpg

        rng = np.random.default_rng(42)
        n = 500
        x = rng.standard_normal(n)
        mu = np.exp(1.0 + 0.3 * x)
        y = generate_tweedie_cpg(n, mu, phi=1.0, p=1.5, rng=rng)
        y = np.maximum(y, 0.0)
        X = pd.DataFrame({"x": x})

        model = SuperGLM(
            family=Tweedie(p=1.5),
            selection_penalty=0.001,
            features={"x": Numeric()},
        )
        result = estimate_tweedie_p(model, X, y, phi_method="pearson")
        fig = result.profile_plot(n_points=20)

        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        assert ax.get_xlabel() == "p"
        assert len(ax.lines) >= 1
        plt.close(fig)


class TestTweedieDensityCIProvenance:
    @staticmethod
    def _profile_fixture():
        cutoff = float(chi2.ppf(0.95, 1))
        records = {}
        objective_calls = []
        record_calls = []

        def objective(p):
            key = float(p)
            objective_calls.append(key)
            nll = 0.5 * cutoff * ((key - 1.5) / 0.2) ** 2
            n_saddlepoint = 2 if np.isclose(key, 1.55, rtol=0.0, atol=1e-14) else 0
            records[key] = SimpleNamespace(
                p=key,
                nll=nll,
                source="ci_fixture",
                fit_converged=True,
                phi_result=SimpleNamespace(
                    objective_finite=True,
                    converged=True,
                    diagnostics=tweedie_module._TweedieLogpdfDiagnostics(
                        n_positive=10,
                        n_saddlepoint=n_saddlepoint,
                    ),
                ),
            )
            return nll

        # A cached remote search record must not taint the connected LR interval.
        records[1.85] = SimpleNamespace(
            p=1.85,
            nll=100.0,
            source="remote_search",
            fit_converged=True,
            phi_result=SimpleNamespace(
                objective_finite=True,
                converged=True,
                diagnostics=tweedie_module._TweedieLogpdfDiagnostics(
                    n_positive=10,
                    n_saddlepoint=10,
                ),
            ),
        )

        def record(p):
            key = float(p)
            record_calls.append(key)
            return records.get(key)

        return objective, record, objective_calls, record_calls

    def test_connected_lr_density_provenance_excludes_remote_records(self):
        objective, record, objective_calls, record_calls = self._profile_fixture()

        details = tweedie_module._profile_ci_p_detailed(
            objective,
            p_hat=1.5,
            nll_hat=0.0,
            ll_scale=1.0,
            p_range=(1.1, 1.9),
            seed_points=(1.5, 1.55, 1.85),
            evaluation_record=record,
        )

        assert details.density_method == "hybrid_exact_saddlepoint"
        assert not details.density_exact
        assert details.any_saddlepoint
        assert details.max_saddlepoint_fraction == pytest.approx(0.2)
        assert details.max_saddlepoint_p == pytest.approx(1.55)
        assert details.n_density_records == len(details.density_provenance)
        assert details.n_saddlepoint_records == 1
        assert details.n_positive == 10 * details.n_density_records
        assert details.n_saddlepoint == 2
        assert all(
            details.lower.value <= item.p <= details.upper.value
            for item in details.density_provenance
        )
        assert all(
            item.lr_statistic <= details.cutoff + 1e-8 for item in details.density_provenance
        )
        assert all(item.p != 1.85 for item in details.density_provenance)
        hybrid = next(item for item in details.density_provenance if item.p == pytest.approx(1.55))
        assert hybrid.source == "ci_fixture"
        assert hybrid.n_positive == 10
        assert hybrid.n_saddlepoint == 2
        assert hybrid.method == "hybrid_exact_saddlepoint"
        with pytest.raises(FrozenInstanceError):
            hybrid.fraction = 0.0

        # Provenance is derived from records retained by the objective probes;
        # it does not trigger another fit/profile evaluation.
        assert len(record_calls) == len(objective_calls)

    def test_invalid_ci_density_counts_are_explicit_and_do_not_report_zero_totals(self):
        cutoff = float(chi2.ppf(0.95, 1))
        records = {}

        def objective(p):
            key = float(p)
            nll = 0.5 * cutoff * ((key - 0.5) / 0.2) ** 2
            records[key] = SimpleNamespace(
                nll=nll,
                source="invalid_counts" if key == 0.5 else "valid_counts",
                fit_converged=True,
                phi_result=SimpleNamespace(
                    objective_finite=True,
                    converged=True,
                    diagnostics=tweedie_module._TweedieLogpdfDiagnostics(
                        n_positive="10" if key == 0.5 else 10,
                        n_saddlepoint=0,
                    ),
                ),
            )
            return nll

        details = tweedie_module._profile_ci_p_detailed(
            objective,
            p_hat=0.5,
            nll_hat=0.0,
            ll_scale=1.0,
            p_range=(0.0, 1.0),
            evaluation_record=lambda p: records.get(float(p)),
        )

        assert details.n_invalid_density_records == 1
        assert details.n_positive is None
        assert details.n_saddlepoint is None
        assert details.density_method == "hybrid_exact_saddlepoint"
        assert details.density_exact is False

    def test_ci_rejects_cross_record_positive_count_mismatch(self):
        cutoff = float(chi2.ppf(0.95, 1))
        records = {}

        def objective(p):
            key = float(p)
            nll = 0.5 * cutoff * ((key - 0.5) / 0.2) ** 2
            records[key] = SimpleNamespace(
                nll=nll,
                source="winner" if key == 0.5 else "mismatched_positive_count",
                fit_converged=True,
                phi_result=SimpleNamespace(
                    objective_finite=True,
                    converged=True,
                    diagnostics=tweedie_module._TweedieLogpdfDiagnostics(
                        n_positive=10 if key == 0.5 else 11,
                        n_saddlepoint=0,
                    ),
                ),
            )
            return nll

        result = TestTweedieProfileCI._bare_result(
            objective,
            p_hat=0.5,
            nll=0.0,
            ll_scale=1.0,
        )
        result._evaluation_record = lambda p: records.get(float(p))

        with pytest.warns(UserWarning, match="positive-response count"):
            result.ci()
        details = result.ci_details()

        assert details.density_method == "hybrid_exact_saddlepoint"
        assert details.density_exact is False
        assert details.density_warning_severity == "high"
        assert details.n_positive is None
        assert details.n_saddlepoint is None
        assert details.n_invalid_density_records > 0
        assert any(not item.counts_valid for item in details.density_provenance)

    def test_ci_density_warning_is_emitted_once_and_cached(self):
        objective, record, _, _ = self._profile_fixture()
        result = TestTweedieProfileCI._bare_result(
            objective,
            p_hat=1.5,
            nll=0.0,
            ll_scale=1.0,
        )
        result._ci_p_range = (1.1, 1.9)
        result._ci_seed_points = (1.5, 1.55, 1.85)
        result._evaluation_record = record

        with pytest.warns(UserWarning, match="evaluated LR region"):
            interval = result.ci()

        details = result.ci_details()
        assert details.density_warning_severity == "warning"
        assert sum("evaluated LR region" in message for message in details.warnings) == 1
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert result.ci() is interval
            assert result.ci_details() is details
        assert caught == []

    def test_ci_density_warning_is_semantically_deduplicated_across_alpha(self):
        objective, record, _, _ = self._profile_fixture()
        result = TestTweedieProfileCI._bare_result(
            objective,
            p_hat=1.5,
            nll=0.0,
            ll_scale=1.0,
        )
        result._ci_p_range = (1.1, 1.9)
        result._ci_seed_points = (1.5, 1.55, 1.85)
        result._evaluation_record = record

        with pytest.warns(UserWarning, match="evaluated LR region"):
            result.ci(alpha=0.05)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result.ci(alpha=0.10)

        assert caught == []
        assert any("evaluated LR region" in message for message in result.ci_details(0.10).warnings)


class TestTweedieProfilePlotLabels:
    @staticmethod
    def _result(*, phi_method="mle", final_exact=True, ci_exact=True, truncated=False):
        interval = (1.4, 1.6)
        lower = tweedie_module.TweedieProfileCIEndpoint(
            value=interval[0],
            status="truncated" if truncated else "root_found",
            at_range_boundary=truncated,
            lr_statistic=1.0,
        )
        upper = tweedie_module.TweedieProfileCIEndpoint(
            value=interval[1],
            status="root_found",
            at_range_boundary=False,
            lr_statistic=1.0,
        )
        details = tweedie_module.TweedieProfileCIDetails(
            alpha=0.05,
            cutoff=float(chi2.ppf(0.95, 1)),
            p_range=(1.1, 1.9),
            lower=lower,
            upper=upper,
            interval=interval,
            n_new_evaluations=0,
            evaluations=(),
            warnings=(),
            density_method="exact" if ci_exact else "hybrid_exact_saddlepoint",
            density_exact=ci_exact,
        )
        result = tweedie_module.TweedieProfileResult(
            p_hat=1.5,
            phi_hat=1.0,
            nll=0.0,
            n_evaluations=1,
            converged=True,
            method="brent",
            phi_method=phi_method,
            search_trace=pd.DataFrame({"p": [1.5], "nll": [0.0]}),
            density_method="exact" if final_exact else "hybrid_exact_saddlepoint",
            density_exact=final_exact,
            _objective=lambda p: (float(p) - 1.5) ** 2,
            _ll_scale=1.0,
        )
        result._ci_cache[0.05] = interval
        result._ci_details_cache[0.05] = details
        return result

    @pytest.mark.parametrize(
        ("kwargs", "expected", "forbidden"),
        [
            ({}, ("MLE", "LR interval"), ("approximation-based",)),
            (
                {"final_exact": False},
                ("approximation-based profile estimate", "approximation-based LR interval"),
                ("MLE =",),
            ),
            (
                {"ci_exact": False},
                ("MLE", "approximation-based LR interval"),
                ("approximation-based profile estimate",),
            ),
            (
                {"phi_method": "pearson"},
                ("profile estimate",),
                ("MLE", "LR interval", "profile interval", "cutoff"),
            ),
            (
                {"truncated": True},
                ("truncated at configured bound",),
                (),
            ),
        ],
    )
    def test_profile_plot_uses_honest_estimate_and_interval_labels(
        self, monkeypatch, kwargs, expected, forbidden
    ):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        result = self._result(**kwargs)

        def unexpected_ci(*args, **kwargs):
            raise AssertionError("profile_plot must never call ci() or ci_details()")

        monkeypatch.setattr(result, "ci", unexpected_ci)
        monkeypatch.setattr(result, "ci_details", unexpected_ci)
        fig = result.profile_plot(n_points=3)
        labels = [text.get_text() for text in fig.axes[0].get_legend().get_texts()]
        joined = " | ".join(labels)
        for value in expected:
            assert value in joined
        for value in forbidden:
            assert value not in joined
        if kwargs.get("phi_method") == "pearson":
            ax = fig.axes[0]
            assert "likelihood" not in ax.get_title().lower()
            assert ax.get_ylabel() == "Profile objective difference"
        plt.close(fig)

    def test_mle_profile_plot_uses_tuple_only_cached_interval_without_ci_calls(self, monkeypatch):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        result = self._result()
        result._ci_details_cache.clear()

        def unexpected_ci(*args, **kwargs):
            raise AssertionError("profile_plot must never call ci() or ci_details()")

        monkeypatch.setattr(result, "ci", unexpected_ci)
        monkeypatch.setattr(result, "ci_details", unexpected_ci)

        fig = result.profile_plot(n_points=3)
        ax = fig.axes[0]
        labels = " | ".join(text.get_text() for text in ax.get_legend().get_texts())
        assert "cutoff" in labels
        assert "profile interval (density provenance unavailable)" in labels
        assert ax.get_ylabel() == "Profile deviance"
        assert "likelihood" in ax.get_title().lower()
        plt.close(fig)

    @pytest.mark.parametrize(
        ("alpha", "clear_cache"),
        [(0.05, True), (0.10, False)],
        ids=["uncached", "different-alpha-cache"],
    )
    def test_mle_profile_plot_never_computes_ci_and_uses_only_matching_cache(
        self, monkeypatch, alpha, clear_cache
    ):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        result = self._result()
        if clear_cache:
            result._ci_cache.clear()
            result._ci_details_cache.clear()

        def unexpected_ci(*args, **kwargs):
            raise AssertionError("profile_plot must not compute a profile CI")

        monkeypatch.setattr(result, "ci", unexpected_ci)
        monkeypatch.setattr(result, "ci_details", unexpected_ci)

        fig = result.profile_plot(alpha=alpha, n_points=3)
        ax = fig.axes[0]
        labels = " | ".join(text.get_text() for text in ax.get_legend().get_texts())
        assert "cutoff" not in labels
        assert "interval" not in labels
        assert ax.get_ylabel() == "Profile objective difference"
        assert "likelihood" not in ax.get_title().lower()
        plt.close(fig)

    def test_pearson_profile_plot_ignores_stale_lr_cache_without_ci_calls(self, monkeypatch):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        result = self._result(phi_method="pearson")

        def unexpected_ci(*args, **kwargs):
            raise AssertionError("Pearson profile_plot must not access LR CI methods")

        monkeypatch.setattr(result, "ci", unexpected_ci)
        monkeypatch.setattr(result, "ci_details", unexpected_ci)

        fig = result.profile_plot(n_points=3)
        ax = fig.axes[0]
        labels = " | ".join(text.get_text() for text in ax.get_legend().get_texts())
        assert "MLE" not in labels
        assert "LR" not in labels
        assert "cutoff" not in labels
        assert "interval" not in labels
        assert ax.get_ylabel() == "Profile objective difference"
        assert "likelihood" not in ax.get_title().lower()
        plt.close(fig)
