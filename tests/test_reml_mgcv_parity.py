"""mgcv parity tests for REML smoothing parameter estimation.

Compares SuperGLM's fit_reml() against mgcv::gam(method="REML") on the same
model and data: cubic P-splines with a second-order difference penalty and ten
basis functions a smooth (mgcv ``bs="ps", k=10``; SuperGLM
``Spline(kind="ps", n_knots=6)``). The references are printed by
tests/oracles/reml_parity_reference.R (R 4.5.0, mgcv 1.9.3). mgcv reports its
smoothing parameters for penalties it scaled by ``S.scale`` (16 here); the
references divide that out, so they are the smoothing parameters of the
penalties as built, SuperGLM's own.

The bars follow from where each engine's REML search stops. SuperGLM's stop
leaves its smoothing parameters where a refit at ``reml_tol=1e-10`` finds them,
to nine digits. mgcv stops with a Newton step left, ``H^-1 g`` from its outer
gradient and Hessian (``"step"`` below, in log lambda). A smooth's EDF is
``sum_k 1 / (1 + lambda d_k)`` over its penalised directions, each of which
moves by at most 1/4 per unit of log lambda, so its eight penalised
directions move it by at most 2 per unit: mgcv's stop alone accounts for an EDF
gap up to ``2 * step``. A flat direction (the Gamma ``x2``, nearly linear,
EDF 1.0008) leaves its lambda unresolved by either engine; its EDF is compared.
"""

import os

import numpy as np
import pandas as pd

from superglm.features.spline import Spline
from superglm.model import SuperGLM

DATA_DIR = os.path.join(os.path.dirname(__file__), "fixtures")

# ── mgcv references (tests/oracles/reml_parity_reference.R) ─────────

MGCV_POISSON = {
    "lambda": (1.913327, 626.6924),
    "edf": (5.752225, 1.953474),
    "sum_edf": 8.705699,
    "deviance": 961.8344994,
    "step": (1.6e-05, 1.32e-03),
}

MGCV_GAMMA = {
    "lambda": (0.313665, 758383.0),
    "edf": (6.598888, 1.000812),
    "sum_edf": 8.599701,
    "deviance": 157.6268501,
    "scale": 0.1921034,
    "step": (2.04e-05, 1.0),  # x2 is flat: a unit step leaves its EDF where it is
}

# A smooth's EDF within 5e-3 of mgcv's: twice the 2.6e-3 mgcv's largest step
# left (1.3e-3 in log lambda) can account for; their sum within 1e-2.
EDF_BAR = 5e-3
# A resolved smoothing parameter within 1e-2 in log lambda. mgcv's stop accounts
# for at most 1.3e-3 of the gap and SuperGLM's for none; the Poisson ``x2`` is
# 4.8e-3 apart, 3.5e-3 beyond mgcv's step: a difference between the two engines'
# fits of this model that the bar bounds and the stops do not explain.
LOG_LAMBDA_BAR = 1e-2


def _fit(family: str) -> SuperGLM:
    df = pd.read_csv(os.path.join(DATA_DIR, f"reml_parity_data_{family}.csv"))
    m = SuperGLM(
        features={"x1": Spline(kind="ps", n_knots=6), "x2": Spline(kind="ps", n_knots=6)},
        family=family,
        selection_penalty=0,
    )
    return m.fit_reml(df, df["y"].values.astype(float), max_reml_iter=30)


class TestMgcvParity:
    """Compare SuperGLM REML against mgcv on the same model and data."""

    def test_poisson_deviance(self):
        """Deviance within 1e-4 of mgcv's, relative; measured 1.3e-5, from the lambda gap."""
        m = _fit("poisson")
        assert m._reml_result.converged
        rel = abs(m.result.deviance - MGCV_POISSON["deviance"]) / MGCV_POISSON["deviance"]
        assert rel < 1e-4, f"deviance {m.result.deviance:.6f} vs mgcv {MGCV_POISSON['deviance']}"

    def test_poisson_edf(self):
        """Each smooth's EDF within ``EDF_BAR`` of mgcv's; measured 3.6e-4 and 2.0e-3."""
        m = _fit("poisson")
        edf = m._group_edf
        np.testing.assert_allclose(
            [edf["x1"], edf["x2"]], MGCV_POISSON["edf"], rtol=0, atol=EDF_BAR
        )
        assert abs(m.result.effective_df - MGCV_POISSON["sum_edf"]) < 2 * EDF_BAR

    def test_poisson_lambdas(self):
        """Both smoothing parameters within ``LOG_LAMBDA_BAR`` of mgcv's in log lambda.

        Measured 6.7e-4 (``x1``) and 4.8e-3 (``x2``).
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
        """Deviance within 1e-4 of mgcv's, relative; measured 5.0e-6."""
        m = _fit("gamma")
        assert m._reml_result.converged
        rel = abs(m.result.deviance - MGCV_GAMMA["deviance"]) / MGCV_GAMMA["deviance"]
        assert rel < 1e-4, f"deviance {m.result.deviance:.6f} vs mgcv {MGCV_GAMMA['deviance']}"

    def test_gamma_edf(self):
        """Each smooth's EDF within ``EDF_BAR`` of mgcv's; measured 6.0e-4 and 6.7e-4.

        The nearly linear ``x2`` sits at EDF 1.0001 here and 1.0008 in mgcv, where
        its smoothing parameter is unresolved by either engine (4.3e6 here, 7.6e5).
        """
        m = _fit("gamma")
        edf = m._group_edf
        np.testing.assert_allclose([edf["x1"], edf["x2"]], MGCV_GAMMA["edf"], rtol=0, atol=EDF_BAR)
        assert abs(m.result.effective_df - MGCV_GAMMA["sum_edf"]) < 2 * EDF_BAR

    def test_gamma_scale(self):
        """Profiled phi within 1% of mgcv's scale.

        mgcv's scale is the Pearson estimate of this fit; SuperGLM profiles phi
        on its REML criterion. The two estimators differ by 0.43% here
        (measured), far more than the fits do (deviance 5e-6 apart).
        """
        m = _fit("gamma")
        rel = abs(m.result.phi - MGCV_GAMMA["scale"]) / MGCV_GAMMA["scale"]
        assert rel < 0.01, f"phi {m.result.phi:.6f} vs mgcv scale {MGCV_GAMMA['scale']}"

    def test_gamma_lambda(self):
        """The resolved smoothing parameter within ``LOG_LAMBDA_BAR`` of mgcv's; measured 6.0e-4."""
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
