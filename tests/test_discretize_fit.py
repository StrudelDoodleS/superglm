"""Tests for fit-time discretization (BAM-style binning)."""

import numpy as np
import pandas as pd
import pytest

from superglm import SuperGLM
from superglm.features.categorical import Categorical
from superglm.features.numeric import Numeric
from superglm.features.spline import CubicRegressionSpline, NaturalSpline, Spline
from superglm.group_matrix import DiscretizedSSPGroupMatrix, DiscretizedTensorGroupMatrix


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


@pytest.fixture
def tensor_interaction_data():
    """Poisson data with two spline parents and a nonlinear interaction."""
    rng = np.random.default_rng(123)
    n = 2500
    age = rng.uniform(18, 80, n)
    bm = rng.uniform(15, 45, n)
    log_mu = (
        -0.8
        + 0.18 * np.sin(age / 8.5)
        - 0.12 * np.cos(bm / 5.0)
        + 0.20 * np.sin(age / 11.0) * np.cos(bm / 6.0)
    )
    y = rng.poisson(np.exp(log_mu)).astype(float)
    X = pd.DataFrame({"age": age, "bm": bm})
    return X, y


class TestDiscretizedFit:
    def test_close_to_exact(self, poisson_data):
        """Discretized coefficients, deviance, and predictions close to exact."""
        X, y = poisson_data

        model_exact = SuperGLM(
            family="poisson",
            selection_penalty=0.01,
            features={"x1": Spline(n_knots=10, penalty="ssp"), "x2": Numeric()},
        )
        model_exact.fit(X, y)

        model_disc = SuperGLM(
            family="poisson",
            selection_penalty=0.01,
            discrete=True,
            n_bins=256,
            features={"x1": Spline(n_knots=10, penalty="ssp"), "x2": Numeric()},
        )
        model_disc.fit(X, y)

        # Coefficients: ~5-10% difference at n=2000/256 bins
        beta_exact = model_exact.result.beta
        beta_disc = model_disc.result.beta
        rel_diff = np.linalg.norm(beta_exact - beta_disc) / (np.linalg.norm(beta_exact) + 1e-10)
        assert rel_diff < 0.10, f"Relative coefficient difference {rel_diff:.4f} too large"

        # Deviance
        dev_exact = model_exact.result.deviance
        dev_disc = model_disc.result.deviance
        dev_rel = abs(dev_exact - dev_disc) / (abs(dev_exact) + 1e-10)
        assert dev_rel < 0.005, f"Relative deviance difference {dev_rel:.6f} too large"

        # Predictions
        mu_exact = model_exact.predict(X)
        mu_disc = model_disc.predict(X)
        max_rel = np.max(np.abs(mu_exact - mu_disc) / (mu_exact + 1e-10))
        assert max_rel < 0.05, f"Max relative prediction difference {max_rel:.4f} too large"

    def test_uses_discretized_group_matrix(self, poisson_data):
        """Discretized model should use DiscretizedSSPGroupMatrix for spline groups."""
        X, y = poisson_data

        model = SuperGLM(
            family="poisson",
            selection_penalty=0.01,
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
            selection_penalty=0.01,
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
            selection_penalty=0.01,
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
            selection_penalty=0.01,
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
            selection_penalty=0.05,
            features={
                "signal": Spline(n_knots=10, penalty="ssp", select=True),
                "noise": Spline(n_knots=10, penalty="ssp", select=True),
            },
        )
        model_exact.fit(X, y)

        model_disc = SuperGLM(
            family="poisson",
            selection_penalty=0.05,
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
            selection_penalty=0,
            features={"x": Spline(n_knots=10, penalty="ssp")},
        )
        model_exact.fit_reml(X, y)

        model_disc = SuperGLM(
            family="poisson",
            selection_penalty=0,
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
            selection_penalty=0,
            features={
                "x1": Spline(n_knots=10, penalty="ssp"),
                "x2": Spline(n_knots=8, penalty="ssp"),
            },
        )
        model_exact.fit_reml(X, y)

        model_disc = SuperGLM(
            family="poisson",
            selection_penalty=0,
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
        """discrete=True + selection_penalty=0 should use PIRLS (not irls_direct) for REML."""
        rng = np.random.default_rng(42)
        n = 500
        x = rng.uniform(0, 10, n)
        mu = np.exp(0.5 + 0.3 * np.sin(x))
        y = rng.poisson(mu).astype(float)
        X = pd.DataFrame({"x": x})

        model = SuperGLM(
            family="poisson",
            selection_penalty=0,
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
            selection_penalty=0,
            discrete=True,
            features={
                "signal": Spline(n_knots=10, penalty="ssp", select=True),
                "noise": Spline(n_knots=10, penalty="ssp", select=True),
            },
        )
        model.fit_reml(X, y)

        assert model.result.converged
        assert hasattr(model, "_reml_lambdas")

    def test_fit_reml_rejects_nonpositive_n_bins(self):
        """fit_reml() should validate per-feature n_bins before discretizing."""
        rng = np.random.default_rng(42)
        n = 200
        x = rng.uniform(0, 10, n)
        y = rng.poisson(np.exp(0.5 + 0.2 * np.sin(x))).astype(float)
        X = pd.DataFrame({"x": x})

        model = SuperGLM(
            family="poisson",
            selection_penalty=0,
            discrete=True,
            n_bins={"x": 0},
            features={"x": Spline(n_knots=8, penalty="ssp")},
        )

        with pytest.raises(ValueError, match="n_bins for feature 'x' must be >= 1"):
            model.fit_reml(X, y)


class TestDiscretizedIRLSDirect:
    def test_irls_direct_discrete(self):
        """selection_penalty=0 uses irls_direct solver — should work with discretization."""
        rng = np.random.default_rng(42)
        n = 1000
        x = rng.uniform(0, 10, n)
        mu = np.exp(0.5 + 0.3 * np.sin(x))
        y = rng.poisson(mu).astype(float)
        X = pd.DataFrame({"x": x})

        model_exact = SuperGLM(
            family="poisson",
            selection_penalty=0,
            features={"x": Spline(n_knots=10, penalty="ssp")},
        )
        model_exact.fit(X, y)

        model_disc = SuperGLM(
            family="poisson",
            selection_penalty=0,
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

    @pytest.mark.parametrize(
        "spline_cls",
        [
            pytest.param(NaturalSpline, id="natural"),
            pytest.param(CubicRegressionSpline, id="crs"),
        ],
    )
    def test_constrained_spline_discrete_close_to_exact(self, spline_cls):
        """Constrained spline discrete fit should match exact in beta count and deviance."""
        rng = np.random.default_rng(42)
        n = 2000
        x = rng.uniform(0, 10, n)
        mu = np.exp(0.5 + 0.3 * np.sin(x))
        y = rng.poisson(mu).astype(float)
        X = pd.DataFrame({"x": x})

        model_exact = SuperGLM(
            family="poisson", selection_penalty=0.01, features={"x": spline_cls(n_knots=10)}
        )
        model_exact.fit(X, y)

        model_disc = SuperGLM(
            family="poisson",
            selection_penalty=0.01,
            discrete=True,
            features={"x": spline_cls(n_knots=10)},
        )
        model_disc.fit(X, y)

        # Same number of coefficients (K-2, not K)
        assert len(model_exact.result.beta) == len(model_disc.result.beta)

        # Deviance close
        dev_exact = model_exact.result.deviance
        dev_disc = model_disc.result.deviance
        rel_diff = abs(dev_exact - dev_disc) / (abs(dev_exact) + 1e-10)
        assert rel_diff < 0.005, f"Deviance difference {rel_diff:.6f}"

        # Predictions close
        mu_exact = model_exact.predict(X)
        mu_disc = model_disc.predict(X)
        max_rel = np.max(np.abs(mu_exact - mu_disc) / (mu_exact + 1e-10))
        assert max_rel < 0.05, f"Max relative prediction difference {max_rel:.4f}"

    def test_natural_spline_discrete_uses_discretized_matrix(self):
        """NaturalSpline + discrete=True should use DiscretizedSSPGroupMatrix."""
        rng = np.random.default_rng(42)
        n = 500
        x = rng.uniform(0, 10, n)
        y = rng.poisson(1.0, n).astype(float)
        X = pd.DataFrame({"x": x})

        model = SuperGLM(
            family="poisson",
            selection_penalty=0.01,
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
            selection_penalty=0.01,
            discrete=True,
            n_bins=64,
            features={"x": Spline(n_knots=10, penalty="ssp")},
        )
        model.fit(X, y)

        gm = model._dm.group_matrices[0]
        assert isinstance(gm, DiscretizedSSPGroupMatrix)
        assert gm.n_bins == 64

    def test_model_n_bins_dict_propagates(self):
        """Model-level n_bins dict should apply per feature."""
        rng = np.random.default_rng(42)
        n = 800
        x1 = rng.uniform(0, 10, n)
        x2 = rng.uniform(-2, 3, n)
        y = rng.poisson(1.0, n).astype(float)
        X = pd.DataFrame({"x1": x1, "x2": x2})

        model = SuperGLM(
            family="poisson",
            selection_penalty=0.01,
            discrete=True,
            n_bins={"x1": 64, "x2": 32},
            features={
                "x1": Spline(n_knots=10, penalty="ssp"),
                "x2": Spline(n_knots=8, penalty="ssp"),
            },
        )
        model.fit(X, y)

        gm1, gm2 = model._dm.group_matrices[:2]
        assert isinstance(gm1, DiscretizedSSPGroupMatrix)
        assert isinstance(gm2, DiscretizedSSPGroupMatrix)
        assert gm1.n_bins == 64
        assert gm2.n_bins == 32

    def test_feature_n_bins_overrides_model(self):
        """Feature-level n_bins should override model-level."""
        rng = np.random.default_rng(42)
        n = 500
        x = rng.uniform(0, 10, n)
        y = rng.poisson(1.0, n).astype(float)
        X = pd.DataFrame({"x": x})

        model = SuperGLM(
            family="poisson",
            selection_penalty=0.01,
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
            selection_penalty=0.01,
            discrete=True,
            features={"x": Spline(n_knots=10, penalty="ssp")},
        )
        model.fit(X, y)

        gm = model._dm.group_matrices[0]
        assert isinstance(gm, DiscretizedSSPGroupMatrix)
        assert gm.n_bins == 256


class TestLowUniqueCompression:
    def test_low_unique_values_use_exact_support(self):
        """If unique support is smaller than n_bins, discrete fit should be exact."""
        rng = np.random.default_rng(42)
        n = 1500
        age = rng.integers(18, 81, size=n).astype(float)
        mu = np.exp(-1.0 + 0.03 * (age - 45) + 0.18 * np.sin(age / 7.0))
        y = rng.poisson(mu).astype(float)
        X = pd.DataFrame({"age": age})

        model_exact = SuperGLM(
            family="poisson",
            selection_penalty=0,
            features={"age": Spline(n_knots=10, penalty="ssp")},
        )
        model_exact.fit(X, y)

        model_disc = SuperGLM(
            family="poisson",
            selection_penalty=0,
            discrete=True,
            n_bins=256,
            features={"age": Spline(n_knots=10, penalty="ssp")},
        )
        model_disc.fit(X, y)

        gm = model_disc._dm.group_matrices[0]
        assert isinstance(gm, DiscretizedSSPGroupMatrix)
        assert gm.n_bins == len(np.unique(age))
        np.testing.assert_allclose(
            model_disc.predict(X), model_exact.predict(X), rtol=1e-8, atol=1e-10
        )


class TestDiscretizedTensorInteraction:
    def test_tensor_interaction_predictions_close_to_exact(self, tensor_interaction_data):
        """Discrete tensor interaction should stay close to the exact fit."""
        X, y = tensor_interaction_data

        model_exact = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            features={
                "age": Spline(n_knots=10, penalty="ssp"),
                "bm": Spline(n_knots=8, penalty="ssp"),
            },
            interactions=[("age", "bm")],
        )
        model_exact.fit(X, y)

        model_disc = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            discrete=True,
            n_bins={"age": 64, "bm": 48},
            features={
                "age": Spline(n_knots=10, penalty="ssp"),
                "bm": Spline(n_knots=8, penalty="ssp"),
            },
            interactions=[("age", "bm")],
        )
        model_disc.fit(X, y)

        pred_exact = model_exact.predict(X)
        pred_disc = model_disc.predict(X)
        mean_rel = np.mean(np.abs(pred_exact - pred_disc) / (pred_exact + 1e-10))
        rel_dev = abs(model_exact.result.deviance - model_disc.result.deviance) / (
            abs(model_exact.result.deviance) + 1e-10
        )
        assert mean_rel < 0.03, f"Mean relative prediction difference {mean_rel:.4f} too large"
        assert rel_dev < 0.002, f"Relative deviance difference {rel_dev:.6f} too large"

    def test_tensor_interaction_uses_discretized_group_matrix(self, tensor_interaction_data):
        """Discrete tensor interaction should reuse DiscretizedSSPGroupMatrix."""
        X, y = tensor_interaction_data

        model = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            discrete=True,
            n_bins={"age": 32, "bm": 24},
            features={
                "age": Spline(n_knots=10, penalty="ssp"),
                "bm": Spline(n_knots=8, penalty="ssp"),
            },
            interactions=[("age", "bm")],
        )
        model.fit(X, y)

        gm_age, gm_bm, gm_inter = model._dm.group_matrices[:3]
        assert isinstance(gm_age, DiscretizedSSPGroupMatrix)
        assert isinstance(gm_bm, DiscretizedSSPGroupMatrix)
        assert isinstance(gm_inter, DiscretizedSSPGroupMatrix)
        assert gm_age.n_bins == 32
        assert gm_bm.n_bins == 24
        assert gm_inter.n_bins <= 32 * 24

    def test_tensor_interaction_low_unique_support_is_exact(self):
        """Low-unique margins should compress the tensor support exactly."""
        ages = np.arange(18, 30, dtype=np.float64)
        bms = np.arange(20, 28, dtype=np.float64)
        grid = np.array(np.meshgrid(ages, bms)).reshape(2, -1).T
        X = pd.DataFrame(
            {
                "age": np.repeat(grid[:, 0], 6),
                "bm": np.repeat(grid[:, 1], 6),
            }
        )
        age = X["age"].to_numpy()
        bm = X["bm"].to_numpy()
        log_mu = -1.1 + 0.04 * (age - 23) - 0.06 * (bm - 24) + 0.01 * (age - 23) * (bm - 24)
        rng = np.random.default_rng(321)
        y = rng.poisson(np.exp(log_mu)).astype(float)

        model_exact = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            features={
                "age": Spline(n_knots=6, penalty="ssp"),
                "bm": Spline(n_knots=5, penalty="ssp"),
            },
            interactions=[("age", "bm")],
        )
        model_exact.fit(X, y)

        model_disc = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            discrete=True,
            n_bins={"age": 256, "bm": 256},
            features={
                "age": Spline(n_knots=6, penalty="ssp"),
                "bm": Spline(n_knots=5, penalty="ssp"),
            },
            interactions=[("age", "bm")],
        )
        model_disc.fit(X, y)

        gm_inter = model_disc._dm.group_matrices[2]
        assert isinstance(gm_inter, DiscretizedSSPGroupMatrix)
        assert gm_inter.n_bins == len(ages) * len(bms)
        np.testing.assert_allclose(
            model_disc.predict(X), model_exact.predict(X), rtol=1e-8, atol=1e-10
        )

    def test_decomposed_tensor_interaction_discrete_smoke(self, tensor_interaction_data):
        """Decomposed discrete tensors should fit without huge disc-disc histograms."""
        X, y = tensor_interaction_data

        model = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            discrete=True,
            n_bins={"age": 48, "bm": 36},
            features={
                "age": Spline(n_knots=10, penalty="ssp"),
                "bm": Spline(n_knots=8, penalty="ssp"),
            },
        )
        model._add_interaction("age", "bm", decompose=True)
        model.fit(X, y)

        interaction_gms = model._dm.group_matrices[2:4]
        assert len(interaction_gms) == 2
        assert all(isinstance(gm, DiscretizedSSPGroupMatrix) for gm in interaction_gms)
        assert [g.name for g in model._groups if g.feature_name == "age:bm"] == [
            "age:bm:bilinear",
            "age:bm:wiggly",
        ]

    def test_tensor_uses_discretized_tensor_group_matrix(self, tensor_interaction_data):
        """Discrete tensor interaction must use DiscretizedTensorGroupMatrix subclass."""
        X, y = tensor_interaction_data
        model = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            discrete=True,
            features={
                "age": Spline(n_knots=10, penalty="ssp"),
                "bm": Spline(n_knots=8, penalty="ssp"),
            },
            interactions=[("age", "bm")],
        )
        model.fit(X, y)
        gm_inter = model._dm.group_matrices[2]
        assert isinstance(gm_inter, DiscretizedTensorGroupMatrix)
        assert hasattr(gm_inter, "B1_unique_t")
        assert hasattr(gm_inter, "B2_unique_t")
        assert hasattr(gm_inter, "idx1")
        assert hasattr(gm_inter, "idx2")
        assert hasattr(gm_inter, "tensor_id")

    def test_decomposed_tensor_shares_tensor_id(self, tensor_interaction_data):
        """Decomposed discrete tensor subgroups must share the same tensor_id."""
        X, y = tensor_interaction_data
        model = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            discrete=True,
            features={
                "age": Spline(n_knots=10, penalty="ssp"),
                "bm": Spline(n_knots=8, penalty="ssp"),
            },
        )
        model._add_interaction("age", "bm", decompose=True)
        model.fit(X, y)
        tensor_gms = [
            gm
            for gm, g in zip(model._dm.group_matrices, model._groups)
            if g.feature_name == "age:bm"
        ]
        assert len(tensor_gms) == 2
        assert all(isinstance(gm, DiscretizedTensorGroupMatrix) for gm in tensor_gms)
        assert tensor_gms[0].tensor_id == tensor_gms[1].tensor_id

    def test_rebuild_design_matrix_preserves_tensor_type(self, tensor_interaction_data):
        """rebuild_design_matrix_with_lambdas must preserve DiscretizedTensorGroupMatrix."""
        X, y = tensor_interaction_data
        model = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            discrete=True,
            features={
                "age": Spline(n_knots=10, penalty="ssp"),
                "bm": Spline(n_knots=8, penalty="ssp"),
            },
            interactions=[("age", "bm")],
        )
        model.fit_reml(X, y, max_reml_iter=3)

        # After REML, the DM was rebuilt with updated lambdas
        gm_inter = model._dm.group_matrices[2]
        assert isinstance(gm_inter, DiscretizedTensorGroupMatrix)
        assert gm_inter.B1_unique_t is not None
        assert gm_inter.B2_unique_t is not None
        assert gm_inter.idx1 is not None
        assert gm_inter.idx2 is not None

    def test_build_discrete_returns_dataclass(self):
        """TensorInteraction.build_discrete() must return DiscreteTensorBuildResult."""
        from superglm.features.interaction import TensorInteraction
        from superglm.types import DiscreteTensorBuildResult

        rng = np.random.default_rng(42)
        n = 200
        x1 = rng.uniform(0, 10, n)
        x2 = rng.uniform(0, 10, n)
        spec1 = Spline(n_knots=6, penalty="ssp")
        spec2 = Spline(n_knots=5, penalty="ssp")
        spec1.build(x1)
        spec2.build(x2)

        ti = TensorInteraction("s1", "s2")
        result = ti.build_discrete(x1, x2, {"s1": spec1, "s2": spec2}, (64, 48))

        assert isinstance(result, DiscreteTensorBuildResult)
        assert result.B_joint.ndim == 2
        assert result.pair_idx.shape == (n,)
        assert result.B1_unique.ndim == 2
        assert result.B2_unique.ndim == 2
        assert result.idx1.shape == (n,)
        assert result.idx2.shape == (n,)
