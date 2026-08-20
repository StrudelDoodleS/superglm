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
