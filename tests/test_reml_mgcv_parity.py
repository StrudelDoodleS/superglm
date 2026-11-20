"""mgcv parity tests for REML smoothing parameter estimation.

Compares SuperGLM's fit_reml() against mgcv::gam(method="REML") on the same
model, knots and data: cubic P-splines with a second-order difference penalty
and ten basis functions a smooth (mgcv ``bs="ps", k=10``; SuperGLM
``Spline(kind="ps", n_knots=6)``), mgcv given SuperGLM's knots. The references
are printed by tests/oracles/reml_parity_reference.R (R 4.5.0, mgcv 1.9.3).
mgcv reports its smoothing parameters for penalties it scaled by ``S.scale``
(16 here); the references divide that out, so they are the smoothing
parameters of the penalties as built, SuperGLM's own.

The bars are a few times the measured gaps, and the gaps are where mgcv's REML
search stops. SuperGLM's stop leaves its smoothing parameters where a refit at
``reml_tol=1e-10`` finds them, to nine digits. mgcv stops with a Newton step
left, ``H^-1 g`` from its outer gradient and Hessian (``"step"`` below, in log
lambda), and each resolved smoothing parameter is that step from SuperGLM's:
1.37e-3 on the Poisson ``x2``, at most 2.0e-5 on the others. A flat direction
(the Gamma ``x2``, nearly linear, EDF 1.0008) leaves its lambda unresolved by
either engine; its EDF is compared.
"""

import os

import numpy as np
import pandas as pd

from superglm.features.spline import Spline
from superglm.model import SuperGLM

DATA_DIR = os.path.join(os.path.dirname(__file__), "fixtures")

# ── mgcv references (tests/oracles/reml_parity_reference.R) ─────────

MGCV_POISSON = {
    "lambda": (1.914576, 628.8693),
    "edf": (5.751871, 1.952097),
    "sum_edf": 8.703968,
    "deviance": 961.8452598,
    "step": (-1.63e-05, -1.37e-03),
}

MGCV_GAMMA = {
    "lambda": (0.3138612, 757125.9),
    "edf": (6.598275, 1.000814),
    "sum_edf": 8.599088,
    "deviance": 157.6276400,
    "reml_scale": 0.192928517,
    "scale": 0.192102876,
    "step": (2.0e-05, -1.0),  # x2 is flat: a unit step leaves its EDF where it is
}

# A smooth's EDF within 2e-3 of mgcv's, their sum within 4e-3: about three times
# the largest gap measured, 6.7e-4 (empirical: mgcv's step moves EDF through
# both smooths and the working weights).
EDF_BAR = 2e-3
# A resolved smoothing parameter within 3e-3 in log lambda: about twice the
# largest gap measured, the 1.37e-3 step mgcv leaves on the Poisson ``x2``.
LOG_LAMBDA_BAR = 3e-3
# Deviance within 1e-5 relative, measured 1.4e-6 (Poisson, from its ``x2``
# step) and 2.6e-8 (Gamma); phi_likelihood within 1e-6 of mgcv's REML scale, measured
# 9e-9. phi within 3e-6 of mgcv's scale, measured 9.7e-7: about three times the gap, which
# the two engines' EDF and coefficient gaps set through Pearson's residual d.f.
DEVIANCE_BAR = 1e-5
PHI_BAR = 1e-6
PEARSON_BAR = 3e-6


def _superglm_knots(x: np.ndarray) -> np.ndarray:
    """The knots the oracle gives mgcv: SuperGLM's rule for ``Spline(kind="ps", n_knots=6)``."""
    lo, hi = x.min(), x.max()
    pad = 1e-3 * (hi - lo)
    inner = np.concatenate([[lo - pad], np.linspace(lo, hi, 8)[1:-1], [hi + pad]])
    below = inner[0] - (inner[1] - inner[0]) * np.arange(3, 0, -1)
    above = inner[-1] + (inner[-1] - inner[-2]) * np.arange(1, 4)
    return np.concatenate([below, inner, above])


def _fit(family: str) -> SuperGLM:
    df = pd.read_csv(os.path.join(DATA_DIR, f"reml_parity_data_{family}.csv"))
    m = SuperGLM(
        features={"x1": Spline(kind="ps", n_knots=6), "x2": Spline(kind="ps", n_knots=6)},
        family=family,
        selection_penalty=0,
    )
    m.fit_reml(df, df["y"].values.astype(float), max_reml_iter=30)
    for name in ("x1", "x2"):  # every comparison is on the basis mgcv fitted
        np.testing.assert_allclose(
            m._specs[name]._knots, _superglm_knots(df[name].values), rtol=1e-14, atol=1e-14
        )
    return m


class TestMgcvParity:
    """Compare SuperGLM REML against mgcv on the same model and data."""

    def test_poisson_deviance(self):
        """Deviance within ``DEVIANCE_BAR`` of mgcv's, relative; measured 1.4e-6."""
        m = _fit("poisson")
        assert m._reml_result.converged
        rel = abs(m.result.deviance - MGCV_POISSON["deviance"]) / MGCV_POISSON["deviance"]
        assert rel < DEVIANCE_BAR, (
            f"deviance {m.result.deviance:.7f}, mgcv {MGCV_POISSON['deviance']}"
        )

    def test_poisson_edf(self):
        """Each smooth's EDF within ``EDF_BAR`` of mgcv's; measured 1.1e-5 and 6.1e-4."""
        m = _fit("poisson")
        edf = m._group_edf
        np.testing.assert_allclose(
            [edf["x1"], edf["x2"]], MGCV_POISSON["edf"], rtol=0, atol=EDF_BAR
        )
        assert abs(m.result.effective_df - MGCV_POISSON["sum_edf"]) < 2 * EDF_BAR

    def test_poisson_lambdas(self):
        """Both smoothing parameters within ``LOG_LAMBDA_BAR`` of mgcv's in log lambda.

        Measured 1.6e-5 (``x1``) and 1.37e-3 (``x2``), mgcv's remaining steps.
        """
        m = _fit("poisson")
        lambdas = [m._reml_lambdas["x1"], m._reml_lambdas["x2"]]
        gaps = np.abs(np.log(lambdas) - np.log(MGCV_POISSON["lambda"]))
        assert np.all(gaps < LOG_LAMBDA_BAR), gaps

    def test_poisson_convergence_speed(self):
        """Newton REML should converge within 15 outer iterations."""
        m = _fit("poisson")
        assert m._reml_result.converged
        assert m._reml_result.n_reml_iter <= 15

    def test_gamma_deviance(self):
        """Deviance within ``DEVIANCE_BAR`` of mgcv's, relative; measured 2.6e-8."""
        m = _fit("gamma")
        assert m._reml_result.converged
        rel = abs(m.result.deviance - MGCV_GAMMA["deviance"]) / MGCV_GAMMA["deviance"]
        assert rel < DEVIANCE_BAR, (
            f"deviance {m.result.deviance:.7f}, mgcv {MGCV_GAMMA['deviance']}"
        )

    def test_gamma_edf(self):
        """Each smooth's EDF within ``EDF_BAR`` of mgcv's; measured 1.5e-5 and 6.7e-4.

        The nearly linear ``x2`` sits at EDF 1.0001 here and 1.0008 in mgcv, where
        its smoothing parameter is unresolved by either engine (4.3e6 here, 7.6e5).
        """
        m = _fit("gamma")
        edf = m._group_edf
        np.testing.assert_allclose([edf["x1"], edf["x2"]], MGCV_GAMMA["edf"], rtol=0, atol=EDF_BAR)
        assert abs(m.result.effective_df - MGCV_GAMMA["sum_edf"]) < 2 * EDF_BAR

    def test_gamma_scale(self):
        """The likelihood scale within ``PHI_BAR`` of mgcv's REML scale, relative; measured 9e-9.

        Both maximise the REML criterion over the scale (mgcv's ``reml.scale``). A null
        space miscounted by one moves it by 1.2e-3.
        """
        from superglm.export.summary import build_summary_export_payload

        m = _fit("gamma")
        overview = {r.metric: r.value for r in build_summary_export_payload(m).overview}
        phi = overview.get("Likelihood phi")
        assert phi is not None, "the summary publishes the likelihood scale"
        assert phi == m.result.phi_likelihood
        rel = abs(phi - MGCV_GAMMA["reml_scale"]) / MGCV_GAMMA["reml_scale"]
        assert rel < PHI_BAR, f"phi_likelihood {phi:.9f}, mgcv {MGCV_GAMMA['reml_scale']}"

    def test_gamma_standard_error_scale(self):
        """``phi``, the scale SEs read, within ``PEARSON_BAR`` of mgcv's ``scale``; measured 9.7e-7.

        mgcv's ``Vp`` carries ``scale`` (``sig2``), its Fletcher (2012) estimate, which a
        log-link Gamma fit with an intercept makes Pearson's; it is 0.43% below the REML
        scale here.
        """
        m = _fit("gamma")
        rel = abs(m.result.phi - MGCV_GAMMA["scale"]) / MGCV_GAMMA["scale"]
        assert rel < PEARSON_BAR, f"phi {m.result.phi:.9f}, mgcv {MGCV_GAMMA['scale']}"

    def test_gamma_lambda(self):
        """The resolved smoothing parameter within ``LOG_LAMBDA_BAR`` of mgcv's; measured 2.0e-5."""
        m = _fit("gamma")
        gap = abs(np.log(m._reml_lambdas["x1"]) - np.log(MGCV_GAMMA["lambda"][0]))
        assert gap < LOG_LAMBDA_BAR, gap

    def test_poisson_predictions_sane(self):
        """Predicted mu should be positive and correlate with observed counts."""
        df = pd.read_csv(os.path.join(DATA_DIR, "reml_parity_data_poisson.csv"))
        y = df["y"].values.astype(float)
        mu_hat = _fit("poisson").predict(df)

        # Sanity check: mu positive and reasonably correlated with y
        assert np.all(mu_hat > 0)
        r = np.corrcoef(y, mu_hat)[0, 1]
        assert r > 0.5, f"Prediction-response correlation {r:.3f} too low"
