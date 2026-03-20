"""Safety validation tests for the cached-W discrete fREML optimizer.

Four test groups:
1. Exact vs discrete agreement on small/medium n
2. Restart / bad-start robustness
3. Cached-W sensitivity (W-refresh frequency)
4. Large-n stability smoke

These tests ensure the fast cached-W path (used when discrete=True,
selection_penalty=0) is trustworthy before merge.
"""

import numpy as np
import pandas as pd
import pytest

from superglm import SuperGLM
from superglm.distributions import clip_mu
from superglm.features.categorical import Categorical
from superglm.features.spline import CubicRegressionSpline
from superglm.links import stabilize_eta

# ── Helpers ──────────────────────────────────────────────────────


def _make_poisson_data(n, seed=42):
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0, 1, n)
    x2 = rng.uniform(0, 1, n)
    eta = 0.5 + np.sin(2 * np.pi * x1) + 0.5 * x2
    mu = np.exp(eta)
    y = rng.poisson(mu).astype(float)
    return pd.DataFrame({"x1": x1, "x2": x2}), y, np.ones(n)


def _make_gamma_data(n, seed=42):
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0, 1, n)
    x2 = rng.uniform(0, 1, n)
    eta = 1.0 + np.sin(2 * np.pi * x1) + 0.5 * x2
    mu = np.exp(eta)
    y = rng.gamma(shape=5.0, scale=mu / 5.0)
    y = np.maximum(y, 1e-4)
    return pd.DataFrame({"x1": x1, "x2": x2}), y, np.ones(n)


def _make_mtpl2_style_data(n, seed=42):
    """Synthetic MTPL2-like data: 3 smooth + 1 categorical, Poisson."""
    rng = np.random.default_rng(seed)
    driv_age = rng.uniform(18, 90, n)
    veh_age = rng.uniform(0, 20, n)
    bonus = rng.uniform(50, 150, n)
    area = rng.choice(["A", "B", "C", "D", "E", "F"], n)
    area_effect = {"A": 0.0, "B": -0.1, "C": 0.05, "D": -0.2, "E": 0.1, "F": -0.05}
    eta = (
        -2.0
        + 0.3 * np.sin(2 * np.pi * (driv_age - 18) / 72)
        - 0.1 * np.cos(np.pi * veh_age / 20)
        + 0.002 * (bonus - 100)
        + np.array([area_effect[a] for a in area])
    )
    mu = np.exp(eta)
    sample_weight = rng.uniform(0.1, 1.0, n)
    y_freq = rng.poisson(mu * sample_weight).astype(float) / sample_weight
    df = pd.DataFrame({"DrivAge": driv_age, "VehAge": veh_age, "BonusMalus": bonus, "Area": area})
    return df, y_freq, sample_weight


_SPLINE_FEATURES = {
    "x1": CubicRegressionSpline(n_knots=8),
    "x2": CubicRegressionSpline(n_knots=8),
}

_MTPL2_FEATURES = {
    "DrivAge": CubicRegressionSpline(n_knots=10),
    "VehAge": CubicRegressionSpline(n_knots=8),
    "BonusMalus": CubicRegressionSpline(n_knots=8),
    "Area": Categorical(base="most_exposed"),
}


def _fit_reml(family, features, discrete, df, y, w, **kwargs):
    model = SuperGLM(family=family, selection_penalty=0, features=features, discrete=discrete)
    model.fit_reml(df, y, sample_weight=w, max_reml_iter=30, **kwargs)
    return model


def _predict_mu(model, df):
    """Get fitted mu values."""
    eta = stabilize_eta(model._dm.matvec(model.result.beta) + model.result.intercept, model._link)
    return clip_mu(model._link.inverse(eta), model._distribution)


# ══════════════════════════════════════════════════════════════════
# Group 1: Exact vs Discrete Agreement
# ══════════════════════════════════════════════════════════════════


class TestExactVsDiscreteAgreement:
    """Verify cached-W discrete path agrees with exact REML oracle."""

    @pytest.mark.parametrize("n", [2000, 20000])
    def test_poisson_agreement(self, n):
        df, y, w = _make_poisson_data(n)

        exact = _fit_reml("poisson", _SPLINE_FEATURES, False, df, y, w)
        disc = _fit_reml("poisson", _SPLINE_FEATURES, True, df, y, w)

        assert exact._reml_result.converged
        assert disc._reml_result.converged

        # Deviance: relative diff <= 5e-4
        dev_rel = abs(exact.result.deviance - disc.result.deviance) / abs(exact.result.deviance)
        assert dev_rel <= 5e-4, f"Poisson n={n} deviance rel diff {dev_rel:.6f}"

        # EDF: At n=2k, the cached-W approximation (holding IRLS weights W
        # fixed during analytical lambda updates) introduces more EDF
        # divergence because W is less stable with fewer observations.
        # At n>=20k the approximation tightens to within 0.15.
        edf_tol = 0.30 if n <= 2000 else 0.15
        edf_diff = abs(exact.result.effective_df - disc.result.effective_df)
        assert edf_diff <= edf_tol, f"Poisson n={n} EDF diff {edf_diff:.4f}"

        # Predictions: corr >= 0.999
        mu_exact = _predict_mu(exact, df)
        mu_disc = _predict_mu(disc, df)
        corr = float(np.corrcoef(mu_exact, mu_disc)[0, 1])
        assert corr >= 0.999, f"Poisson n={n} mu corr {corr:.6f}"

    @pytest.mark.parametrize("n", [2000, 20000])
    def test_gamma_agreement(self, n):
        df, y, w = _make_gamma_data(n)

        exact = _fit_reml("gamma", _SPLINE_FEATURES, False, df, y, w)
        disc = _fit_reml("gamma", _SPLINE_FEATURES, True, df, y, w)

        assert exact._reml_result.converged
        assert disc._reml_result.converged

        # Deviance: relative diff <= 1e-3
        dev_rel = abs(exact.result.deviance - disc.result.deviance) / abs(exact.result.deviance)
        assert dev_rel <= 1e-3, f"Gamma n={n} deviance rel diff {dev_rel:.6f}"

        # EDF: abs diff <= 0.25
        edf_diff = abs(exact.result.effective_df - disc.result.effective_df)
        assert edf_diff <= 0.25, f"Gamma n={n} EDF diff {edf_diff:.4f}"

        # Phi: relative diff <= 5e-3
        phi_rel = abs(exact.result.phi - disc.result.phi) / abs(exact.result.phi)
        assert phi_rel <= 5e-3, f"Gamma n={n} phi rel diff {phi_rel:.6f}"

        # Predictions: corr >= 0.999
        mu_exact = _predict_mu(exact, df)
        mu_disc = _predict_mu(disc, df)
        corr = float(np.corrcoef(mu_exact, mu_disc)[0, 1])
        assert corr >= 0.999, f"Gamma n={n} mu corr {corr:.6f}"


# ══════════════════════════════════════════════════════════════════
# Group 2: Restart / Bad-Start Robustness
# ══════════════════════════════════════════════════════════════════


class TestRestartRobustness:
    """Verify discrete REML converges from adverse starts and restarts."""

    @pytest.mark.parametrize("family", ["poisson", "gamma"])
    def test_bad_starts_converge(self, family):
        """Very small, very large, and restart all converge to same point."""
        if family == "poisson":
            df, y, w = _make_poisson_data(5000)
        else:
            df, y, w = _make_gamma_data(5000)

        # Baseline: default start
        baseline = _fit_reml(family, _SPLINE_FEATURES, True, df, y, w)
        assert baseline._reml_result.converged

        # Very small lambda2_init
        small = _fit_reml(family, _SPLINE_FEATURES, True, df, y, w, lambda2_init=1e-6)
        assert small._reml_result.converged

        # Very large lambda2_init
        large = _fit_reml(family, _SPLINE_FEATURES, True, df, y, w, lambda2_init=1e5)
        assert large._reml_result.converged

        # Restart from returned lambdas
        restart_model = SuperGLM(
            family=family, selection_penalty=0, features=_SPLINE_FEATURES, discrete=True
        )
        # Seed the initial lambdas from baseline result
        baseline_lam = baseline._reml_lambdas
        avg_lam = float(np.mean(list(baseline_lam.values())))
        restart_model.fit_reml(df, y, sample_weight=w, max_reml_iter=30, lambda2_init=avg_lam)
        assert restart_model._reml_result.converged

        # All should agree on deviance within 1e-4
        devs = [
            baseline.result.deviance,
            small.result.deviance,
            large.result.deviance,
            restart_model.result.deviance,
        ]
        ref_dev = devs[0]
        for i, d in enumerate(devs):
            rel = abs(d - ref_dev) / abs(ref_dev)
            assert rel < 1e-4, f"Start {i} deviance rel diff {rel:.6f}"

        # EDF within 0.05
        edfs = [
            baseline.result.effective_df,
            small.result.effective_df,
            large.result.effective_df,
            restart_model.result.effective_df,
        ]
        ref_edf = edfs[0]
        for i, e in enumerate(edfs):
            assert abs(e - ref_edf) < 0.05, f"Start {i} EDF diff {abs(e - ref_edf):.4f}"

        # Predictions correlate >= 0.9999
        mu_ref = _predict_mu(baseline, df)
        for i, m in enumerate([small, large, restart_model]):
            mu_i = _predict_mu(m, df)
            corr = float(np.corrcoef(mu_ref, mu_i)[0, 1])
            assert corr >= 0.9999, f"Start {i + 1} mu corr {corr:.6f}"


# ══════════════════════════════════════════════════════════════════
# Group 3: Cached-W Sensitivity
# ══════════════════════════════════════════════════════════════════


class TestCachedWSensitivity:
    """Verify W-refresh frequency does not materially change results.

    Uses the private _max_analytical_per_w test hook to control how many
    analytical lambda updates run between IRLS W-refreshes.
    """

    @pytest.mark.parametrize("family", ["poisson", "gamma"])
    def test_refresh_strategies_agree(self, family):
        """refresh-every-step, refresh-every-2, and default cached-W agree."""
        if family == "poisson":
            df, y, w = _make_poisson_data(5000)
        else:
            df, y, w = _make_gamma_data(5000)

        results = {}
        for label, max_anal in [
            ("every_step", 1),
            ("every_2", 2),
            ("default", 30),
        ]:
            model = SuperGLM(
                family=family, selection_penalty=0, features=_SPLINE_FEATURES, discrete=True
            )
            model._max_analytical_per_w = max_anal
            model.fit_reml(df, y, sample_weight=w, max_reml_iter=50)
            assert model._reml_result.converged, f"{label} did not converge"
            results[label] = model

        ref = results["every_step"]  # most conservative strategy
        ref_dev = ref.result.deviance
        ref_edf = ref.result.effective_df
        ref_phi = ref.result.phi
        mu_ref = _predict_mu(ref, df)

        for label in ["every_2", "default"]:
            m = results[label]

            # Deviance
            if family == "poisson":
                dev_tol = 5e-4
            else:
                dev_tol = 1e-3
            dev_rel = abs(m.result.deviance - ref_dev) / abs(ref_dev)
            assert dev_rel <= dev_tol, f"{family} {label} deviance rel diff {dev_rel:.6f}"

            # EDF
            edf_diff = abs(m.result.effective_df - ref_edf)
            assert edf_diff <= 0.15, f"{family} {label} EDF diff {edf_diff:.4f}"

            # Phi (Gamma only)
            if family == "gamma":
                phi_rel = abs(m.result.phi - ref_phi) / abs(ref_phi)
                assert phi_rel <= 5e-3, f"Gamma {label} phi rel diff {phi_rel:.6f}"

            # Predictions
            mu_i = _predict_mu(m, df)
            corr = float(np.corrcoef(mu_ref, mu_i)[0, 1])
            assert corr >= 0.999, f"{family} {label} mu corr {corr:.6f}"


# ══════════════════════════════════════════════════════════════════
# Group 4: Large-n Stability Smoke
# ══════════════════════════════════════════════════════════════════


class TestLargeNStability:
    """Stability smoke test on MTPL2-style synthetic data."""

    def test_restart_reproduces(self):
        """Restarting from returned lambdas reproduces the same endpoint."""
        df, y, w = _make_mtpl2_style_data(50000)

        # First fit
        m1 = SuperGLM(
            family="poisson", selection_penalty=0, features=_MTPL2_FEATURES, discrete=True
        )
        m1.fit_reml(df, y, sample_weight=w, max_reml_iter=30)
        assert m1._reml_result.converged

        # Second fit starting from returned lambdas
        avg_lam = float(np.mean(list(m1._reml_lambdas.values())))
        m2 = SuperGLM(
            family="poisson", selection_penalty=0, features=_MTPL2_FEATURES, discrete=True
        )
        m2.fit_reml(df, y, sample_weight=w, max_reml_iter=30, lambda2_init=avg_lam)
        assert m2._reml_result.converged

        # Should agree exactly (same data, warm start near optimum)
        dev_rel = abs(m1.result.deviance - m2.result.deviance) / abs(m1.result.deviance)
        assert dev_rel < 1e-6, f"Restart deviance rel diff {dev_rel:.8f}"
        assert abs(m1.result.effective_df - m2.result.effective_df) < 0.01

        # Lambdas should agree
        for name in m1._reml_lambdas:
            lam1 = m1._reml_lambdas[name]
            lam2 = m2._reml_lambdas[name]
            log_diff = abs(np.log(lam1) - np.log(lam2))
            assert log_diff < 0.01, f"{name} log-lambda diff {log_diff:.4f}"

    def test_stable_across_seeds(self):
        """Discrete REML is stable across 3 data seeds."""
        deviances = []
        edfs = []
        irls_calls = []
        irls_iters = []

        for seed in [42, 123, 999]:
            df, y, w = _make_mtpl2_style_data(50000, seed=seed)
            model = SuperGLM(
                family="poisson",
                selection_penalty=0,
                features=_MTPL2_FEATURES,
                discrete=True,
            )
            model.fit_reml(df, y, sample_weight=w, max_reml_iter=30)
            assert model._reml_result.converged

            deviances.append(model.result.deviance)
            edfs.append(model.result.effective_df)
            # Get IRLS stats from profile if available
            p = getattr(model, "_reml_profile", {})
            irls_calls.append(p.get("irls_calls", "?"))
            irls_iters.append(p.get("irls_iters", "?"))

        # All seeds should converge; deviance/EDF should be in similar range
        # (not identical since data differs, but same order of magnitude)
        dev_arr = np.array(deviances)
        edf_arr = np.array(edfs)

        # Coefficient of variation of deviance < 10% (same DGP, different realizations)
        dev_cv = float(np.std(dev_arr) / np.mean(dev_arr))
        assert dev_cv < 0.10, f"Deviance CV across seeds: {dev_cv:.4f}"

        # EDF spread < 5
        edf_spread = float(np.max(edf_arr) - np.min(edf_arr))
        assert edf_spread < 5.0, f"EDF spread across seeds: {edf_spread:.2f}"
