"""Neutral high-precision reference for a real Tweedie power profile."""

import hashlib

import numpy as np
import pandas as pd
import pytest

from superglm import SuperGLM, Tweedie
from superglm.features.numeric import Numeric
from superglm.profiling.tweedie import estimate_tweedie_p, generate_tweedie_cpg


@pytest.mark.slow
def test_joint_profile_matches_neutral_reference() -> None:
    rng = np.random.default_rng(101)
    x = rng.standard_normal(800)
    mu = np.exp(0.3 + 0.45 * x)
    y = generate_tweedie_cpg(800, mu=mu, phi=0.8, p=1.2, rng=rng)
    digest = hashlib.sha256(np.ascontiguousarray(y, dtype="<f8").tobytes()).hexdigest()
    assert digest == "7d2c5cf30a0d8f3c1a7fb281adb2c864900f1ec16e59fdfff536d197f3186477"
    model = SuperGLM(features={"x": Numeric()}, family=Tweedie(p=1.5))

    result = estimate_tweedie_p(
        model,
        pd.DataFrame({"x": x}),
        y,
        p_bounds=(1.05, 1.95),
        xatol=1.0e-4,
        maxiter=30,
        phi_method="mle",
    )

    assert result.p_hat == pytest.approx(1.1968971098776182, abs=2.0e-4)
    assert result.phi_hat == pytest.approx(0.8068142191615686, rel=5.0e-4)
    assert result.converged
    assert result.method == "joint_ml"
    assert result.density_exact
    assert result.n_saddlepoint == 0
    assert result.n_evaluations <= 4
    assert int(result.search_trace["phi_n_evaluations"].sum()) <= 12
    assert set(result.search_trace["phi_optimizer"]) == {"exact-newton"}
    assert "fell back" not in result.outer_message
