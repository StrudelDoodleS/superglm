"""Tests for Davies' algorithm and Wood (2013) Bayesian p-values."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2 as chi2_dist
from scipy.stats import exponnorm
from scipy.stats import f as f_dist

from superglm.stats import wood_pvalue as wood_pvalue_module
from superglm.stats.davies import psum_chisq, satterthwaite
from superglm.stats.wood_pvalue import (
    _fallback_tail,
    _fractional_tail,
    wood_test_smooth,
)

# ── Davies algorithm ─────────────────────────────────────────────


class TestDavies:
    def test_single_weight_one_is_standard_chi2(self):
        """A single chi²(1) with weight 1 should match scipy chi²(1)."""
        q = 3.84
        p, ifault = psum_chisq(q, np.array([1.0]))
        expected = 1.0 - chi2_dist.cdf(q, 1)
        assert ifault == 0
        np.testing.assert_allclose(p, expected, atol=0.01)

    def test_single_weight_matches_scaled_chi2(self):
        """Single weight w: P[w*chi²(1) > q] = P[chi²(1) > q/w]."""
        w = 2.5
        q = 5.0
        p, ifault = psum_chisq(q, np.array([w]))
        expected = 1.0 - chi2_dist.cdf(q / w, 1)
        assert ifault == 0
        np.testing.assert_allclose(p, expected, atol=0.01)

    def test_equal_weights_is_scaled_chi2(self):
        """k equal weights w: P[w*chi²(k) > q] = P[chi²(k) > q/w]."""
        k = 5
        w = 1.5
        q = 12.0
        weights = np.full(k, w)
        p, ifault = psum_chisq(q, weights)
        expected = 1.0 - chi2_dist.cdf(q / w, k)
        assert ifault == 0
        np.testing.assert_allclose(p, expected, atol=0.01)

    def test_unit_weights_chi2_k(self):
        """k unit weights = chi²(k) distribution."""
        k = 10
        q = 18.31  # chi²(10) 95th percentile
        weights = np.ones(k)
        p, ifault = psum_chisq(q, weights)
        expected = 1.0 - chi2_dist.cdf(q, k)
        assert ifault == 0
        np.testing.assert_allclose(p, expected, atol=0.01)

    def test_zero_q_gives_one(self):
        """P[Q > 0] should be 1 (or very close)."""
        p, ifault = psum_chisq(0.0, np.array([1.0, 2.0]))
        assert p >= 0.99

    def test_large_q_gives_zero(self):
        """Very large q should give p ≈ 0."""
        p, ifault = psum_chisq(1000.0, np.array([1.0, 0.5]))
        assert p < 0.001

    def test_empty_weights(self):
        """The degenerate zero form respects the strict upper-tail event."""
        p1, _ = psum_chisq(0.0, np.array([]))
        assert p1 == 0.0
        p0, _ = psum_chisq(-1.0, np.array([]))
        assert p0 == 1.0
        p2, _ = psum_chisq(5.0, np.array([]))
        assert p2 == 0.0

    def test_higher_df(self):
        """Weights with df > 1."""
        q = 10.0
        weights = np.array([1.0, 1.0])
        df = np.array([3.0, 2.0])
        p, ifault = psum_chisq(q, weights, df=df)
        # sum of chi²(3) + chi²(2) = chi²(5)
        expected = 1.0 - chi2_dist.cdf(q, 5)
        assert ifault == 0
        np.testing.assert_allclose(p, expected, atol=0.01)

    def test_mixed_positive_negative_weights(self):
        """Mixed sign weights should still produce valid probability."""
        q = 2.0
        weights = np.array([2.0, -1.0, 1.5])
        p, ifault = psum_chisq(q, weights)
        assert 0.0 <= p <= 1.0

    def test_equal_unit_weight_far_tail_is_exact_and_monotone(self):
        """Equal weights are the certified chi-square oracle across the tail."""
        q_values = np.logspace(-1.0, np.log10(5e4), 80)
        actual = []
        for q in q_values:
            p_value, ifault = psum_chisq(q, np.ones(5))
            assert ifault == 0
            actual.append(p_value)

        actual = np.asarray(actual)
        expected = chi2_dist.sf(q_values, 5)
        assert np.all(np.diff(actual) <= 0.0)
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=100.0 * np.finfo(np.float64).eps,
            atol=0.0,
        )

    def test_unequal_weight_oscillatory_tail_matches_hypoexponential_oracle(self):
        """Two chi-square(2) terms give a closed-form convolution oracle."""
        weights = np.array([1.0, 0.5])
        df = np.array([2.0, 2.0])
        accuracy = 1e-11
        for q in (1.0, 10.0, 30.0):
            p_value, ifault = psum_chisq(q, weights, df=df, acc=accuracy)
            expected = 2.0 * np.exp(-q / 2.0) - np.exp(-q)
            assert ifault == 0
            assert abs(p_value - expected) <= 10.0 * accuracy

    def test_ifault_marks_a_tail_that_cannot_be_resolved(self):
        """A tiny unequal-weight tail may be returned only if exact or flagged."""
        q = 100.0
        p_value, ifault = psum_chisq(
            q,
            np.array([1.0, 0.5]),
            df=np.array([2.0, 2.0]),
        )
        expected = 2.0 * np.exp(-q / 2.0) - np.exp(-q)
        if ifault == 0:
            np.testing.assert_allclose(
                p_value,
                expected,
                rtol=100.0 * np.finfo(np.float64).eps,
                atol=0.0,
            )
        else:
            assert ifault == 1

    def test_ifault_marks_an_unresolved_lower_tail_of_a_mixed_form(self):
        """Gross error in the complement must also prevent a success flag."""
        positive_weight = 1.0
        negative_weight = 1e5
        q = -4.3e7
        p_value, ifault = psum_chisq(
            q,
            np.array([positive_weight, -negative_weight]),
            df=np.array([2.0, 2.0]),
        )
        expected = 1.0 - negative_weight / (positive_weight + negative_weight) * np.exp(
            q / (2.0 * negative_weight)
        )

        if ifault == 0:
            np.testing.assert_allclose(
                p_value,
                expected,
                rtol=100.0 * np.finfo(np.float64).eps,
                atol=0.0,
            )
        else:
            assert ifault == 1

    @pytest.mark.parametrize(
        ("weights", "df", "sigma", "lim", "acc"),
        [
            (np.array([1.0]), np.array([1.0, 1.0]), 0.0, 100, 1e-6),
            (np.array([1.0]), np.array([0.0]), 0.0, 100, 1e-6),
            (np.array([np.nan]), None, 0.0, 100, 1e-6),
            (np.array([1.0]), None, -1.0, 100, 1e-6),
            (np.array([1.0]), None, 0.0, 2, 1e-6),
            (np.array([1.0]), None, 0.0, 3.5, 1e-6),
            (np.array([1.0]), None, 0.0, np.inf, 1e-6),
            (np.array([1.0]), None, 0.0, 100, 0.0),
        ],
    )
    def test_invalid_inputs_have_documented_ifault_four(
        self,
        weights,
        df,
        sigma,
        lim,
        acc,
    ):
        p_value, ifault = psum_chisq(
            1.0,
            weights,
            df=df,
            sigma=sigma,
            lim=lim,
            acc=acc,
        )
        assert np.isnan(p_value)
        assert ifault == 4

    @pytest.mark.parametrize("q", [-5.0, -1.0, 0.0, 1.0, 5.0, 20.0])
    def test_mixed_sign_tail_orientation_matches_asymmetric_laplace(self, q):
        """A difference of chi-square(2) terms has an exact two-sided oracle."""
        positive_weight = 2.0
        negative_weight = 0.7
        p_value, ifault = psum_chisq(
            q,
            np.array([positive_weight, -negative_weight]),
            df=np.array([2.0, 2.0]),
            acc=1e-12,
        )
        if q < 0.0:
            expected = 1.0 - negative_weight / (positive_weight + negative_weight) * np.exp(
                q / (2.0 * negative_weight)
            )
        else:
            expected = (
                positive_weight
                / (positive_weight + negative_weight)
                * np.exp(-q / (2.0 * positive_weight))
            )

        assert ifault == 0
        np.testing.assert_allclose(p_value, expected, rtol=2e-11, atol=2e-13)

    @pytest.mark.parametrize("direction", [-1.0, 1.0])
    def test_mixed_sign_tail_is_continuous_at_numerically_nonzero_q(self, direction):
        """Tiny nonzero q must not enter an ill-conditioned Fourier limit."""
        positive_weight = 0.15
        negative_weight = 7.0
        q = direction * 32.0 * np.finfo(np.float64).eps * negative_weight
        p_value, ifault = psum_chisq(
            q,
            np.array([positive_weight, -negative_weight]),
            df=np.array([2.0, 2.0]),
            acc=1e-12,
        )
        if q < 0.0:
            expected = 1.0 - negative_weight / (positive_weight + negative_weight) * np.exp(
                q / (2.0 * negative_weight)
            )
        else:
            expected = (
                positive_weight
                / (positive_weight + negative_weight)
                * np.exp(-q / (2.0 * positive_weight))
            )

        assert ifault == 0
        np.testing.assert_allclose(p_value, expected, rtol=2e-11, atol=2e-13)

    def test_mixed_sign_small_q_between_direct_and_fourier_scales(self):
        """A small Fourier frequency must not be reported as a successful tail."""
        positive_weight = 1.0
        negative_weight = 0.1
        q = 5e-8
        p_value, ifault = psum_chisq(
            q,
            np.array([positive_weight, -negative_weight]),
            df=np.array([2.0, 2.0]),
        )
        expected = (
            positive_weight
            / (positive_weight + negative_weight)
            * np.exp(-q / (2.0 * positive_weight))
        )

        assert ifault == 0
        np.testing.assert_allclose(p_value, expected, rtol=2e-11, atol=2e-13)

    @pytest.mark.parametrize("q", [-3.168e-4, -1e-4, 1e-4, 3.168e-4])
    def test_mixed_sign_small_q_matches_asymmetric_laplace_oracle(self, q):
        positive_weight = 2.0
        negative_weight = 0.7
        p_value, ifault = psum_chisq(
            q,
            np.array([positive_weight, -negative_weight]),
            df=np.array([2.0, 2.0]),
            acc=1e-12,
        )
        if q < 0.0:
            expected = 1.0 - negative_weight / (positive_weight + negative_weight) * np.exp(
                q / (2.0 * negative_weight)
            )
        else:
            expected = (
                positive_weight
                / (positive_weight + negative_weight)
                * np.exp(-q / (2.0 * positive_weight))
            )

        assert ifault == 0
        np.testing.assert_allclose(
            p_value,
            expected,
            rtol=100.0 * np.finfo(np.float64).eps,
            atol=0.0,
        )

    @pytest.mark.parametrize("q", [-1e308, 1e308])
    def test_asymmetric_laplace_exact_path_avoids_finite_scale_overflow(self, q):
        weight = 1e308
        p_value, ifault = psum_chisq(
            q,
            np.array([weight, -weight]),
            df=np.array([2.0, 2.0]),
        )
        if q < 0.0:
            expected = 1.0 - 0.5 * np.exp(0.5 * q / weight)
        else:
            expected = 0.5 * np.exp(-0.5 * q / weight)

        assert ifault == 0
        np.testing.assert_allclose(p_value, expected, rtol=1e-15, atol=0.0)

    def test_mixed_sign_zero_tail_resolves_widely_separated_component_scales(self):
        """Direct inversion must visit the reciprocal scale of every component."""
        positive_weight = 1.0
        negative_weight = 1e-5
        p_value, ifault = psum_chisq(
            0.0,
            np.array([positive_weight, -negative_weight]),
            df=np.array([2.0, 2.0]),
        )
        expected = positive_weight / (positive_weight + negative_weight)

        assert ifault == 0
        np.testing.assert_allclose(p_value, expected, rtol=2e-11, atol=2e-13)

    @pytest.mark.parametrize("negative_weight", [1e-12, 1e-6, 1.0, 1e6, 1e12])
    @pytest.mark.parametrize(("positive_df", "negative_df"), [(4.0, 4.0), (3.0, 90.0)])
    def test_zero_tail_two_component_mixture_matches_exact_f_ratio(
        self,
        negative_weight,
        positive_df,
        negative_df,
    ):
        """The exact q=0 F identity remains stable across component scales."""
        positive_weight = 1.0
        p_value, ifault = psum_chisq(
            0.0,
            np.array([positive_weight, -negative_weight]),
            df=np.array([positive_df, negative_df]),
            acc=1e-12,
        )
        threshold = negative_weight * negative_df / (positive_weight * positive_df)
        expected = f_dist.sf(threshold, positive_df, negative_df)

        assert ifault == 0
        np.testing.assert_allclose(
            p_value,
            expected,
            rtol=1e-12,
            atol=0.0,
        )

    @pytest.mark.parametrize(
        ("q", "weights"),
        [
            (5e-5, np.array([1.0, 0.9])),
            (-5e-5, np.array([-1.0, -0.9])),
        ],
    )
    def test_one_sided_near_support_tail_is_not_silently_certified(self, q, weights):
        """Direct inversion must flag a support-boundary tail it cannot resolve."""
        _, ifault = psum_chisq(
            q,
            weights,
            df=np.array([1.0, 1.0]),
            acc=1e-12,
        )

        assert ifault == 1

    @pytest.mark.parametrize("q", [-3.0, 0.0, 1.0, 3.0, 10.0])
    def test_normal_component_matches_exponentially_modified_normal(self, q):
        """chi-square(2) plus a Gaussian is scipy's exponnorm exactly."""
        weight = 1.3
        sigma = 0.8
        p_value, ifault = psum_chisq(
            q,
            np.array([weight]),
            df=np.array([2.0]),
            sigma=sigma,
            acc=1e-12,
        )
        expected = exponnorm.sf(q, 2.0 * weight / sigma, loc=0.0, scale=sigma)

        assert ifault == 0
        np.testing.assert_allclose(p_value, expected, rtol=2e-10, atol=2e-13)


# ── Satterthwaite fallback ───────────────────────────────────────


class TestSatterthwaite:
    def test_unit_weights_matches_chi2(self):
        """Unit weights: c*chi²(d) where c=1, d=k."""
        k = 5
        q = 11.07
        p, c, d = satterthwaite(q, np.ones(k))
        expected = 1.0 - chi2_dist.cdf(q, k)
        np.testing.assert_allclose(p, expected, atol=0.02)
        np.testing.assert_allclose(c, 1.0, rtol=0.01)
        np.testing.assert_allclose(d, float(k), rtol=0.01)

    def test_equal_weights(self):
        """Equal weights w: c=w, d=k."""
        w = 3.0
        k = 4
        _, c, d = satterthwaite(5.0, np.full(k, w))
        np.testing.assert_allclose(c, w, rtol=0.01)
        np.testing.assert_allclose(d, float(k), rtol=0.01)


# ── Wood (2013) testStat ─────────────────────────────────────────


class TestWoodTestSmooth:
    def test_full_rank_similar_to_naive_wald(self):
        """When edf1 = p_g, Wood test should approximately equal naive Wald."""
        rng = np.random.default_rng(42)
        p_g = 5
        n = 200

        # Generate a simple design
        X_j = rng.standard_normal((n, p_g))
        beta_j = rng.standard_normal(p_g) * 0.5

        # Build a plausible covariance
        XtX = X_j.T @ X_j
        V_b_j = np.linalg.inv(XtX + 0.01 * np.eye(p_g))

        # Full rank = p_g
        stat, p_val, ref_df = wood_test_smooth(beta_j, X_j, V_b_j, edf1_j=float(p_g))

        # Naive Wald
        wald = float(beta_j @ np.linalg.solve(V_b_j, beta_j))

        # They should be in the same order of magnitude
        assert stat > 0
        assert 0 <= p_val <= 1
        np.testing.assert_allclose(ref_df, float(p_g), atol=0.5)
        # Stat should be close to Wald when rank = full
        np.testing.assert_allclose(stat, wald, rtol=0.5)

    def test_reduced_edf_smaller_stat(self):
        """With edf1 < p_g, the test stat should generally be smaller."""
        rng = np.random.default_rng(42)
        p_g = 10
        n = 500

        X_j = rng.standard_normal((n, p_g))
        beta_j = rng.standard_normal(p_g) * 0.3

        XtX = X_j.T @ X_j
        V_b_j = np.linalg.inv(XtX + 0.1 * np.eye(p_g))

        # Full rank
        stat_full, _, _ = wood_test_smooth(beta_j, X_j, V_b_j, edf1_j=float(p_g))
        # Reduced rank
        stat_reduced, _, ref_df = wood_test_smooth(beta_j, X_j, V_b_j, edf1_j=5.0)

        assert ref_df == 5.0
        # reduced stat may differ from full, but p-value should be valid
        assert stat_reduced > 0

    def test_fractional_edf(self):
        """Fractional edf1 should produce valid results."""
        rng = np.random.default_rng(123)
        p_g = 8
        n = 300

        X_j = rng.standard_normal((n, p_g))
        beta_j = rng.standard_normal(p_g) * 0.4

        XtX = X_j.T @ X_j
        V_b_j = np.linalg.inv(XtX + 0.05 * np.eye(p_g))

        stat, p_val, ref_df = wood_test_smooth(beta_j, X_j, V_b_j, edf1_j=5.7)

        assert stat > 0
        assert 0 <= p_val <= 1
        np.testing.assert_allclose(ref_df, 5.7, atol=0.01)

    def test_edf_zero_returns_no_effect(self):
        """edf1=0 should return stat=0, p=1."""
        beta_j = np.array([0.1, 0.2])
        X_j = np.eye(2)
        V_b_j = np.eye(2)

        stat, p_val, ref_df = wood_test_smooth(beta_j, X_j, V_b_j, edf1_j=0.0)
        assert stat == 0.0
        assert p_val == 1.0
        assert ref_df == 0.0

    def test_f_test_with_res_df(self):
        """Positive res_df should use F-test (estimated scale)."""
        rng = np.random.default_rng(42)
        p_g = 5
        n = 200

        X_j = rng.standard_normal((n, p_g))
        beta_j = rng.standard_normal(p_g) * 0.5
        XtX = X_j.T @ X_j
        V_b_j = np.linalg.inv(XtX + 0.01 * np.eye(p_g))

        stat, p_val, ref_df = wood_test_smooth(beta_j, X_j, V_b_j, edf1_j=float(p_g), res_df=190.0)

        assert stat > 0
        assert 0 <= p_val <= 1

    def test_single_eigenvalue(self):
        """Single parameter (p_g=1) should still work."""
        rng = np.random.default_rng(42)
        n = 100

        X_j = rng.standard_normal((n, 1))
        beta_j = np.array([0.5])
        XtX = X_j.T @ X_j
        V_b_j = np.linalg.inv(XtX + 0.01 * np.eye(1))

        stat, p_val, ref_df = wood_test_smooth(beta_j, X_j, V_b_j, edf1_j=1.0)

        assert stat > 0
        assert 0 <= p_val <= 1

    def test_all_results_finite(self):
        """All outputs should be finite for typical inputs."""
        rng = np.random.default_rng(99)
        for p_g in [3, 7, 15]:
            n = 500
            X_j = rng.standard_normal((n, p_g))
            beta_j = rng.standard_normal(p_g) * 0.2
            XtX = X_j.T @ X_j
            V_b_j = np.linalg.inv(XtX + 0.1 * np.eye(p_g))
            edf1 = float(p_g) * 0.6

            stat, p_val, ref_df = wood_test_smooth(beta_j, X_j, V_b_j, edf1_j=edf1)

            assert np.isfinite(stat), f"Non-finite stat for p_g={p_g}"
            assert np.isfinite(p_val), f"Non-finite p_val for p_g={p_g}"
            assert np.isfinite(ref_df), f"Non-finite ref_df for p_g={p_g}"


# ── Mixture failure handling ─────────────────────────────────────


class TestMixtureFailureHandling:
    def test_known_scale_fractional_tail_uses_satterthwaite_on_each_fault(
        self,
        monkeypatch,
    ):
        calls = []

        def failed_tail(q, weights, df=None, **kwargs):
            calls.append((q, np.asarray(weights), df))
            return np.nan, 1

        monkeypatch.setattr(wood_pvalue_module, "psum_chisq", failed_tail)
        stat = 9.0
        stat_alt = 7.0
        weights = np.array([1.0, 0.65, 0.2])
        actual = _fractional_tail(stat, stat_alt, weights, res_df=-1.0)
        primary, _, _ = satterthwaite(stat, weights)
        alternate, _, _ = satterthwaite(stat_alt, weights)

        assert len(calls) == 2
        assert actual == pytest.approx(0.5 * (primary + alternate))

    def test_estimated_scale_fractional_tail_falls_back_if_either_call_faults(
        self,
        monkeypatch,
    ):
        calls = []

        def failed_tail(q, weights, df=None, **kwargs):
            calls.append((q, np.asarray(weights), df))
            return np.nan, 1 if len(calls) == 1 else 0

        monkeypatch.setattr(wood_pvalue_module, "psum_chisq", failed_tail)
        stat = 8.0
        stat_alt = 6.5
        weights = np.array([1.0, 0.55])
        res_df = 90.0
        actual = _fractional_tail(stat, stat_alt, weights, res_df)
        expected = _fallback_tail(
            stat,
            stat_alt,
            float(np.sum(weights)),
            res_df,
        )

        assert len(calls) == 2
        assert all(call[2] is not None for call in calls)
        assert actual == pytest.approx(expected)

    def test_estimated_scale_extreme_component_ratio_triggers_live_fallback(self):
        residual_df = 90.0
        stat = residual_df * 1e-12
        weights = np.array([1.0, 1.0, 1.20707142, 0.49292858])

        actual = _fractional_tail(stat, stat, weights, residual_df)
        expected = _fallback_tail(
            stat,
            stat,
            float(np.sum(weights)),
            residual_df,
        )

        assert actual == pytest.approx(expected)


# ── F-correction for estimated scale ─────────────────────────────


class TestFCorrection:
    """Verify Wood's live estimated-scale paths use an F reference."""

    def test_fractional_residual_df_is_not_rounded(self):
        """The denominator chi-square must retain its fractional fitted df."""
        from scipy.stats import f as f_dist

        numerator_df = 3.0
        residual_df = 2.5
        stat = 12.0
        actual = _fractional_tail(
            stat,
            stat,
            np.ones(int(numerator_df)),
            residual_df,
        )
        expected = f_dist.sf(stat / numerator_df, numerator_df, residual_df)
        np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-13)

    def test_subunit_rank_fault_falls_back_to_the_live_one_df_mixture(
        self,
        monkeypatch,
    ):
        """A displayed fractional rank must not replace the mixture's test df."""
        from scipy.stats import f as f_dist

        monkeypatch.setattr(
            wood_pvalue_module,
            "psum_chisq",
            lambda *args, **kwargs: (np.nan, 1),
        )
        stat = 8.0
        residual_df = 90.0
        beta = np.array([np.sqrt(stat)])
        X = np.ones((12, 1))
        covariance = np.ones((1, 1))

        actual_stat, actual_p, displayed_ref_df = wood_test_smooth(
            beta,
            X,
            covariance,
            edf1_j=0.5,
            res_df=residual_df,
        )

        assert actual_stat == pytest.approx(stat)
        assert displayed_ref_df == pytest.approx(0.5)
        assert actual_p == pytest.approx(f_dist.sf(stat, 1.0, residual_df))

    def test_wood_test_integer_rank_f_matches_scipy(self):
        """Integer rank path: F(k, res_df) should match scipy directly."""
        from scipy.stats import f as f_dist

        rng = np.random.default_rng(42)
        p_g = 5
        n = 200

        X_j = rng.standard_normal((n, p_g))
        beta_j = rng.standard_normal(p_g) * 0.5
        XtX = X_j.T @ X_j
        V_b_j = np.linalg.inv(XtX + 0.01 * np.eye(p_g))

        # Known scale
        stat_chi, p_chi, _ = wood_test_smooth(beta_j, X_j, V_b_j, edf1_j=float(p_g), res_df=-1.0)
        # Estimated scale
        stat_f, p_f, _ = wood_test_smooth(beta_j, X_j, V_b_j, edf1_j=float(p_g), res_df=190.0)

        # Same test statistic (doesn't depend on scale)
        np.testing.assert_allclose(stat_f, stat_chi, rtol=1e-10)
        # F p-value should be more conservative
        assert p_f >= p_chi - 1e-6

        # Verify F p-value directly: stat/k ~ F(k, res_df)
        expected_f = 1.0 - f_dist.cdf(stat_f / p_g, p_g, 190.0)
        np.testing.assert_allclose(p_f, expected_f, rtol=1e-6)

    def test_wood_test_fractional_rank_f_more_conservative(self):
        """Fractional rank path: F p-value should be >= chi² p-value."""
        rng = np.random.default_rng(123)
        p_g = 8
        n = 300

        X_j = rng.standard_normal((n, p_g))
        beta_j = rng.standard_normal(p_g) * 0.4
        XtX = X_j.T @ X_j
        V_b_j = np.linalg.inv(XtX + 0.05 * np.eye(p_g))

        _, p_chi, _ = wood_test_smooth(beta_j, X_j, V_b_j, edf1_j=5.7, res_df=-1.0)
        _, p_f, _ = wood_test_smooth(beta_j, X_j, V_b_j, edf1_j=5.7, res_df=290.0)

        assert p_f >= p_chi - 1e-6, (
            f"F p={p_f:.6f} should be >= chi² p={p_chi:.6f} for fractional rank"
        )


# ── Integration: summary shows fractional ref_df ─────────────────


class TestWoodIntegration:
    @pytest.fixture
    def spline_model(self):
        """Fit a spline model for integration testing."""
        from superglm import SuperGLM
        from superglm.features.spline import Spline

        rng = np.random.default_rng(42)
        n = 500
        x = rng.uniform(0, 10, n)
        y = rng.poisson(np.exp(0.1 * x)).astype(float)
        X = pd.DataFrame({"x": x})
        model = SuperGLM(
            family="poisson",
            selection_penalty=0.01,
            features={"x": Spline(n_knots=8, penalty="ssp")},
        )
        model.fit(X, y)
        return model, X, y

    def test_summary_shows_fractional_ref_df(self, spline_model):
        """Summary should show fractional ref_df like chi2(5.3)=..."""
        model, X, y = spline_model
        m = model.metrics(X, y)
        text = str(m.summary())

        # Should contain chi2( with a decimal ref_df
        assert "chi2(" in text
        # Should NOT contain "Wald chi2" anymore
        assert "Wald chi2" not in text

    def test_summary_html_shows_chi2(self, spline_model):
        """HTML should show chi² with fractional ref_df."""
        model, X, y = spline_model
        html = model.metrics(X, y).summary()._repr_html_()
        assert "chi2" in html or "χ²" in html or "&chi;" in html

    def test_poisson_uses_chi2_reference(self, spline_model):
        """Poisson (known scale) should use chi² reference dist."""
        model, X, y = spline_model
        m = model.metrics(X, y)
        # Known scale check
        assert m._known_scale is True

    def test_p_values_finite_and_valid(self, spline_model):
        """All p-values in summary should be finite and in [0,1]."""
        model, X, y = spline_model
        m = model.metrics(X, y)
        summary = m.summary()
        for row in summary._coef_rows:
            if row.is_spline and row.active:
                assert np.isfinite(row.wald_chi2), f"Non-finite stat for {row.name}"
                assert np.isfinite(row.wald_p), f"Non-finite p for {row.name}"
                assert 0 <= row.wald_p <= 1, f"Invalid p for {row.name}"
                assert row.ref_df is not None
                assert row.ref_df > 0

    def test_ref_df_less_than_n_params(self, spline_model):
        """ref_df (edf1) should be <= n_params for penalised splines."""
        model, X, y = spline_model
        m = model.metrics(X, y)
        summary = m.summary()
        for row in summary._coef_rows:
            if row.is_spline and row.active:
                assert row.ref_df <= row.n_params + 0.1

    def test_edf_properties_available(self, spline_model):
        """_influence_edf should return valid edf and edf1 vectors."""
        model, X, y = spline_model
        m = model.metrics(X, y)
        edf, edf1 = m._influence_edf
        assert len(edf) > 0
        assert len(edf1) > 0
        assert np.all(np.isfinite(edf))
        assert np.all(np.isfinite(edf1))
        # EDF should be between 0 and 1 per coefficient
        assert np.all(edf >= -0.01)
        assert np.all(edf <= 1.01)

    def test_gamma_uses_f_reference(self):
        """Gamma (estimated scale) should use F reference dist."""
        from superglm import SuperGLM
        from superglm.features.spline import Spline

        rng = np.random.default_rng(42)
        n = 500
        x = rng.uniform(1, 5, n)
        mu = np.exp(0.3 * x)
        y = rng.gamma(shape=2.0, scale=mu / 2.0)
        X = pd.DataFrame({"x": x})
        model = SuperGLM(
            family="gamma",
            selection_penalty=0.01,
            features={"x": Spline(n_knots=6, penalty="ssp")},
        )
        model.fit(X, y)
        m = model.metrics(X, y)
        assert m._known_scale is False

        summary = m.summary()
        for row in summary._coef_rows:
            if row.is_spline and row.active:
                assert np.isfinite(row.wald_p)

    def test_wood_note_in_footer(self, spline_model):
        """Footer should reference Wood (2013)."""
        model, X, y = spline_model
        text = str(model.metrics(X, y).summary())
        assert "Wood (2013)" in text

    def test_multi_spline_model(self):
        """Multiple splines should each get their own Wood test."""
        from superglm import SuperGLM
        from superglm.features.spline import Spline

        rng = np.random.default_rng(42)
        n = 500
        x1 = rng.uniform(0, 10, n)
        x2 = rng.uniform(0, 5, n)
        mu = np.exp(0.1 * x1 - 0.2 * x2)
        y = rng.poisson(mu).astype(float)
        X = pd.DataFrame({"x1": x1, "x2": x2})

        model = SuperGLM(
            family="poisson",
            selection_penalty=0.001,
            features={
                "x1": Spline(n_knots=8, penalty="ssp"),
                "x2": Spline(n_knots=6, penalty="ssp"),
            },
        )
        model.fit(X, y)
        m = model.metrics(X, y)
        summary = m.summary()

        spline_rows = [r for r in summary._coef_rows if r.is_spline and r.active]
        assert len(spline_rows) == 2
        for row in spline_rows:
            assert np.isfinite(row.wald_chi2)
            assert np.isfinite(row.wald_p)
            assert row.ref_df is not None
            assert row.ref_df > 0
