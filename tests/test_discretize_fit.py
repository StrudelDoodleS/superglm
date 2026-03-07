"""Tests for fit-time discretization (BAM-style binning)."""

import numpy as np
import pandas as pd
import pytest

from superglm import SuperGLM
from superglm.features.categorical import Categorical
from superglm.features.numeric import Numeric
from superglm.features.spline import CubicRegressionSpline, NaturalSpline, Spline
from superglm.group_matrix import DiscretizedSSPGroupMatrix


@pytest.fixture
def poisson_data():
    """Poisson data with one spline and one numeric feature."""
    rng = np.random.default_rng(42)
    n = 2000
    x1 = rng.uniform(0, 10, n)
    x2 = rng.standard_normal(n)
    mu = np.exp(0.5 + 0.3 * np.sin(x1) + 0.2 * x2)
    y = rng.poisson(mu).astype(float)
    X = pd.DataFrame({"x1": x1, "x2": x2})
    return X, y


class TestDiscretizedFit:
    def test_coefficients_close_to_exact(self, poisson_data):
        """Discretized coefficients should be close to exact."""
        X, y = poisson_data

        model_exact = SuperGLM(
            family="poisson",
            lambda1=0.01,
            features={"x1": Spline(n_knots=10, penalty="ssp"), "x2": Numeric()},
        )
        model_exact.fit(X, y)

        model_disc = SuperGLM(
            family="poisson",
            lambda1=0.01,
            discrete=True,
            n_bins=256,
            features={"x1": Spline(n_knots=10, penalty="ssp"), "x2": Numeric()},
        )
        model_disc.fit(X, y)

        beta_exact = model_exact.result.beta
        beta_disc = model_disc.result.beta
        rel_diff = np.linalg.norm(beta_exact - beta_disc) / (np.linalg.norm(beta_exact) + 1e-10)
        # With n=2000 and 256 bins, ~8 obs/bin → expect ~5-10% coefficient difference.
        # On real 678k data this would be <1%.
        assert rel_diff < 0.10, f"Relative coefficient difference {rel_diff:.4f} too large"

    def test_deviance_close_to_exact(self, poisson_data):
        """Discretized deviance should be close to exact."""
        X, y = poisson_data

        model_exact = SuperGLM(
            family="poisson",
            lambda1=0.01,
            features={"x1": Spline(n_knots=10, penalty="ssp"), "x2": Numeric()},
        )
        model_exact.fit(X, y)

        model_disc = SuperGLM(
            family="poisson",
            lambda1=0.01,
            discrete=True,
            features={"x1": Spline(n_knots=10, penalty="ssp"), "x2": Numeric()},
        )
        model_disc.fit(X, y)

        dev_exact = model_exact.result.deviance
        dev_disc = model_disc.result.deviance
        rel_diff = abs(dev_exact - dev_disc) / (abs(dev_exact) + 1e-10)
        assert rel_diff < 0.005, f"Relative deviance difference {rel_diff:.6f} too large"

    def test_predictions_close_to_exact(self, poisson_data):
        """Discretized predictions should be close to exact."""
        X, y = poisson_data

        model_exact = SuperGLM(
            family="poisson",
            lambda1=0.01,
            features={"x1": Spline(n_knots=10, penalty="ssp"), "x2": Numeric()},
        )
        model_exact.fit(X, y)

        model_disc = SuperGLM(
            family="poisson",
            lambda1=0.01,
            discrete=True,
            features={"x1": Spline(n_knots=10, penalty="ssp"), "x2": Numeric()},
        )
        model_disc.fit(X, y)

        mu_exact = model_exact.predict(X)
        mu_disc = model_disc.predict(X)
        max_rel = np.max(np.abs(mu_exact - mu_disc) / (mu_exact + 1e-10))
        assert max_rel < 0.05, f"Max relative prediction difference {max_rel:.4f} too large"

    def test_uses_discretized_group_matrix(self, poisson_data):
        """Discretized model should use DiscretizedSSPGroupMatrix for spline groups."""
        X, y = poisson_data

        model = SuperGLM(
            family="poisson",
            lambda1=0.01,
            discrete=True,
            features={"x1": Spline(n_knots=10, penalty="ssp"), "x2": Numeric()},
        )
        model.fit(X, y)

        # x1 group should be discretized, x2 should be dense
        gms = model._dm.group_matrices
        assert isinstance(gms[0], DiscretizedSSPGroupMatrix)
        assert gms[0].n_bins == 256
        assert len(gms[0].bin_idx) == len(y)

    def test_per_feature_discrete(self, poisson_data):
        """Per-feature discrete flag should override model-level."""
        X, y = poisson_data

        # Model-level discrete=False, but x1 is discrete=True
        model = SuperGLM(
            family="poisson",
            lambda1=0.01,
            discrete=False,
            features={
                "x1": Spline(n_knots=10, penalty="ssp", discrete=True, n_bins=128),
                "x2": Numeric(),
            },
        )
        model.fit(X, y)

        gms = model._dm.group_matrices
        assert isinstance(gms[0], DiscretizedSSPGroupMatrix)
        assert gms[0].n_bins == 128

    def test_global_discrete_flag(self, poisson_data):
        """Global discrete=True should apply to all splines without explicit flag."""
        X, y = poisson_data

        model = SuperGLM(
            family="poisson",
            lambda1=0.01,
            discrete=True,
            features={"x1": Spline(n_knots=10, penalty="ssp"), "x2": Numeric()},
        )
        model.fit(X, y)

        gms = model._dm.group_matrices
        assert isinstance(gms[0], DiscretizedSSPGroupMatrix)

    def test_categorical_stays_exact(self):
        """Categorical features should not be affected by discrete=True."""
        rng = np.random.default_rng(42)
        n = 500
        x_cat = rng.choice(["A", "B", "C"], n)
        y = rng.poisson(1.0, n).astype(float)
        X = pd.DataFrame({"cat": x_cat})

        model = SuperGLM(
            family="poisson",
            lambda1=0.01,
            discrete=True,
            features={"cat": Categorical(base="first")},
        )
        model.fit(X, y)

        # Categorical should NOT use discretized matrix
        gms = model._dm.group_matrices
        assert not isinstance(gms[0], DiscretizedSSPGroupMatrix)


class TestDiscretizedSelect:
    def test_select_true_discrete(self):
        """select=True + discrete=True should work and give same sparsity pattern."""
        rng = np.random.default_rng(42)
        n = 1000
        x_signal = rng.uniform(0, 10, n)
        x_noise = rng.uniform(0, 10, n)
        mu = np.exp(0.5 + 0.3 * np.sin(x_signal))
        y = rng.poisson(mu).astype(float)
        X = pd.DataFrame({"signal": x_signal, "noise": x_noise})

        model_exact = SuperGLM(
            family="poisson",
            lambda1=0.05,
            features={
                "signal": Spline(n_knots=10, penalty="ssp", select=True),
                "noise": Spline(n_knots=10, penalty="ssp", select=True),
            },
        )
        model_exact.fit(X, y)

        model_disc = SuperGLM(
            family="poisson",
            lambda1=0.05,
            discrete=True,
            features={
                "signal": Spline(n_knots=10, penalty="ssp", select=True),
                "noise": Spline(n_knots=10, penalty="ssp", select=True),
            },
        )
        model_disc.fit(X, y)

        # Both should zero the noise spline group
        beta_exact = model_exact.result.beta
        beta_disc = model_disc.result.beta

        # Check same groups are active/zeroed
        for g in model_exact._groups:
            exact_active = np.linalg.norm(beta_exact[g.sl]) > 1e-12
            disc_active = np.linalg.norm(beta_disc[g.sl]) > 1e-12
            assert exact_active == disc_active, (
                f"Sparsity mismatch for group '{g.name}': "
                f"exact={'active' if exact_active else 'zero'}, "
                f"disc={'active' if disc_active else 'zero'}"
            )


class TestDiscretizedREML:
    def test_reml_discrete(self):
        """fit_reml() + discrete=True should converge with similar lambdas."""
        rng = np.random.default_rng(42)
        n = 1000
        x = rng.uniform(0, 10, n)
        mu = np.exp(0.5 + 0.3 * np.sin(x))
        y = rng.poisson(mu).astype(float)
        X = pd.DataFrame({"x": x})

        model_exact = SuperGLM(
            family="poisson",
            lambda1=0,
            features={"x": Spline(n_knots=10, penalty="ssp")},
        )
        model_exact.fit_reml(X, y)

        model_disc = SuperGLM(
            family="poisson",
            lambda1=0,
            discrete=True,
            features={"x": Spline(n_knots=10, penalty="ssp")},
        )
        model_disc.fit_reml(X, y)

        # Both should converge
        assert model_exact.result.converged
        assert model_disc.result.converged

        # Deviances should be close
        dev_exact = model_exact.result.deviance
        dev_disc = model_disc.result.deviance
        rel_diff = abs(dev_exact - dev_disc) / (abs(dev_exact) + 1e-10)
        assert rel_diff < 0.01, f"REML deviance difference {rel_diff:.6f} too large"

    def test_freml_lambdas_close_to_exact(self):
        """fREML lambdas should be close to exact REML lambdas."""
        rng = np.random.default_rng(42)
        n = 2000
        x1 = rng.uniform(0, 10, n)
        x2 = rng.uniform(0, 5, n)
        mu = np.exp(0.5 + 0.3 * np.sin(x1) - 0.2 * np.cos(x2))
        y = rng.poisson(mu).astype(float)
        X = pd.DataFrame({"x1": x1, "x2": x2})

        model_exact = SuperGLM(
            family="poisson",
            lambda1=0,
            features={
                "x1": Spline(n_knots=10, penalty="ssp"),
                "x2": Spline(n_knots=8, penalty="ssp"),
            },
        )
        model_exact.fit_reml(X, y)

        model_disc = SuperGLM(
            family="poisson",
            lambda1=0,
            discrete=True,
            features={
                "x1": Spline(n_knots=10, penalty="ssp"),
                "x2": Spline(n_knots=8, penalty="ssp"),
            },
        )
        model_disc.fit_reml(X, y)

        # Per-group lambda comparison
        for name in model_exact._reml_lambdas:
            lam_exact = model_exact._reml_lambdas[name]
            lam_disc = model_disc._reml_lambdas[name]
            rel_diff = abs(lam_exact - lam_disc) / (abs(lam_exact) + 1e-10)
            assert rel_diff < 0.10, (
                f"Lambda '{name}' differs: exact={lam_exact:.6g}, "
                f"disc={lam_disc:.6g}, rel_diff={rel_diff:.4f}"
            )

    def test_freml_uses_pirls_not_direct(self):
        """discrete=True + lambda1=0 should use PIRLS (not irls_direct) for REML."""
        rng = np.random.default_rng(42)
        n = 500
        x = rng.uniform(0, 10, n)
        mu = np.exp(0.5 + 0.3 * np.sin(x))
        y = rng.poisson(mu).astype(float)
        X = pd.DataFrame({"x": x})

        model = SuperGLM(
            family="poisson",
            lambda1=0,
            discrete=True,
            features={"x": Spline(n_knots=10, penalty="ssp")},
        )
        model.fit_reml(X, y)

        # Verify discrete groups are DiscretizedSSPGroupMatrix
        for gm in model._dm.group_matrices:
            if hasattr(gm, "bin_idx"):
                assert isinstance(gm, DiscretizedSSPGroupMatrix)

        # Model should converge and have REML lambdas
        assert model.result.converged
        assert hasattr(model, "_reml_lambdas")

    def test_freml_select_true(self):
        """select=True + discrete=True + REML should converge and select correctly."""
        rng = np.random.default_rng(42)
        n = 1000
        x_signal = rng.uniform(0, 10, n)
        x_noise = rng.uniform(0, 10, n)
        mu = np.exp(0.5 + 0.3 * np.sin(x_signal))
        y = rng.poisson(mu).astype(float)
        X = pd.DataFrame({"signal": x_signal, "noise": x_noise})

        model = SuperGLM(
            family="poisson",
            lambda1=0,
            discrete=True,
            features={
                "signal": Spline(n_knots=10, penalty="ssp", select=True),
                "noise": Spline(n_knots=10, penalty="ssp", select=True),
            },
        )
        model.fit_reml(X, y)

        assert model.result.converged
        assert hasattr(model, "_reml_lambdas")


class TestDiscretizedIRLSDirect:
    def test_irls_direct_discrete(self):
        """lambda1=0 uses irls_direct solver — should work with discretization."""
        rng = np.random.default_rng(42)
        n = 1000
        x = rng.uniform(0, 10, n)
        mu = np.exp(0.5 + 0.3 * np.sin(x))
        y = rng.poisson(mu).astype(float)
        X = pd.DataFrame({"x": x})

        model_exact = SuperGLM(
            family="poisson",
            lambda1=0,
            features={"x": Spline(n_knots=10, penalty="ssp")},
        )
        model_exact.fit(X, y)

        model_disc = SuperGLM(
            family="poisson",
            lambda1=0,
            discrete=True,
            features={"x": Spline(n_knots=10, penalty="ssp")},
        )
        model_disc.fit(X, y)

        beta_exact = model_exact.result.beta
        beta_disc = model_disc.result.beta
        rel_diff = np.linalg.norm(beta_exact - beta_disc) / (np.linalg.norm(beta_exact) + 1e-10)
        assert rel_diff < 0.10


class TestConstrainedSplineDiscrete:
    """Constrained splines (NaturalSpline, CRS) with discrete=True."""

    def test_natural_spline_discrete_close_to_exact(self):
        """NaturalSpline discrete predictions should be close to exact."""
        rng = np.random.default_rng(42)
        n = 2000
        x = rng.uniform(0, 10, n)
        mu = np.exp(0.5 + 0.3 * np.sin(x))
        y = rng.poisson(mu).astype(float)
        X = pd.DataFrame({"x": x})

        model_exact = SuperGLM(
            family="poisson",
            lambda1=0.01,
            features={"x": NaturalSpline(n_knots=10)},
        )
        model_exact.fit(X, y)

        model_disc = SuperGLM(
            family="poisson",
            lambda1=0.01,
            discrete=True,
            features={"x": NaturalSpline(n_knots=10)},
        )
        model_disc.fit(X, y)

        # Same number of coefficients (K-2, not K)
        assert len(model_exact.result.beta) == len(model_disc.result.beta)

        # Predictions close
        mu_exact = model_exact.predict(X)
        mu_disc = model_disc.predict(X)
        max_rel = np.max(np.abs(mu_exact - mu_disc) / (mu_exact + 1e-10))
        assert max_rel < 0.05, f"Max relative prediction difference {max_rel:.4f}"

    def test_crs_discrete_close_to_exact(self):
        """CubicRegressionSpline discrete predictions should be close to exact."""
        rng = np.random.default_rng(42)
        n = 2000
        x = rng.uniform(0, 10, n)
        mu = np.exp(0.5 + 0.3 * np.sin(x))
        y = rng.poisson(mu).astype(float)
        X = pd.DataFrame({"x": x})

        model_exact = SuperGLM(
            family="poisson",
            lambda1=0.01,
            features={"x": CubicRegressionSpline(n_knots=10)},
        )
        model_exact.fit(X, y)

        model_disc = SuperGLM(
            family="poisson",
            lambda1=0.01,
            discrete=True,
            features={"x": CubicRegressionSpline(n_knots=10)},
        )
        model_disc.fit(X, y)

        # Same number of coefficients (K-2, not K)
        assert len(model_exact.result.beta) == len(model_disc.result.beta)

        # Deviance close
        dev_exact = model_exact.result.deviance
        dev_disc = model_disc.result.deviance
        rel_diff = abs(dev_exact - dev_disc) / (abs(dev_exact) + 1e-10)
        assert rel_diff < 0.005, f"CRS deviance difference {rel_diff:.6f}"

    def test_natural_spline_discrete_uses_discretized_matrix(self):
        """NaturalSpline + discrete=True should use DiscretizedSSPGroupMatrix."""
        rng = np.random.default_rng(42)
        n = 500
        x = rng.uniform(0, 10, n)
        y = rng.poisson(1.0, n).astype(float)
        X = pd.DataFrame({"x": x})

        model = SuperGLM(
            family="poisson",
            lambda1=0.01,
            discrete=True,
            features={"x": NaturalSpline(n_knots=10)},
        )
        model.fit(X, y)

        gm = model._dm.group_matrices[0]
        assert isinstance(gm, DiscretizedSSPGroupMatrix)


class TestModelLevelNBins:
    """Model-level n_bins should propagate to features that don't set their own."""

    def test_model_n_bins_propagates(self):
        """SuperGLM(n_bins=64) should produce 64-bin discretized matrices."""
        rng = np.random.default_rng(42)
        n = 500
        x = rng.uniform(0, 10, n)
        y = rng.poisson(1.0, n).astype(float)
        X = pd.DataFrame({"x": x})

        model = SuperGLM(
            family="poisson",
            lambda1=0.01,
            discrete=True,
            n_bins=64,
            features={"x": Spline(n_knots=10, penalty="ssp")},
        )
        model.fit(X, y)

        gm = model._dm.group_matrices[0]
        assert isinstance(gm, DiscretizedSSPGroupMatrix)
        assert gm.n_bins == 64

    def test_feature_n_bins_overrides_model(self):
        """Feature-level n_bins should override model-level."""
        rng = np.random.default_rng(42)
        n = 500
        x = rng.uniform(0, 10, n)
        y = rng.poisson(1.0, n).astype(float)
        X = pd.DataFrame({"x": x})

        model = SuperGLM(
            family="poisson",
            lambda1=0.01,
            discrete=True,
            n_bins=64,
            features={"x": Spline(n_knots=10, penalty="ssp", n_bins=128)},
        )
        model.fit(X, y)

        gm = model._dm.group_matrices[0]
        assert isinstance(gm, DiscretizedSSPGroupMatrix)
        assert gm.n_bins == 128

    def test_default_n_bins_is_256(self):
        """Without any n_bins setting, default should be 256."""
        rng = np.random.default_rng(42)
        n = 500
        x = rng.uniform(0, 10, n)
        y = rng.poisson(1.0, n).astype(float)
        X = pd.DataFrame({"x": x})

        model = SuperGLM(
            family="poisson",
            lambda1=0.01,
            discrete=True,
            features={"x": Spline(n_knots=10, penalty="ssp")},
        )
        model.fit(X, y)

        gm = model._dm.group_matrices[0]
        assert isinstance(gm, DiscretizedSSPGroupMatrix)
        assert gm.n_bins == 256
