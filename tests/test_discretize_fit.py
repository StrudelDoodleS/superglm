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


@pytest.fixture
def shared_marginal_tensor_data():
    """Insurance-shaped data for two ``ti()`` terms sharing one marginal.

    ``bm`` and ``veh`` are both driven by ``age``, so each joint support is
    holey and each tensor block is numerically rank deficient — the regime
    where a cross-tensor projection can silently collapse a whole term.
    """
    rng = np.random.default_rng(3)
    n = 1500
    age = rng.integers(18, 90, n).astype(float)
    bm = np.clip(150.0 - age - rng.integers(0, 40, n), 50.0, 130.0)
    veh = np.clip((age - 18) * rng.uniform(0.0, 0.7, n), 0.0, 40.0).round()
    X = pd.DataFrame({"age": age, "bm": bm, "veh": veh})
    exposure = rng.uniform(0.05, 1.0, n)
    mu = np.exp(-2.0 + 0.01 * age - 0.004 * bm + 0.01 * veh)
    y = (rng.poisson(mu * exposure) / exposure).astype(float)
    return X, y, exposure


def _shared_marginal_tensor_model(discrete):
    model = SuperGLM(
        family="poisson",
        selection_penalty=None,
        discrete=discrete,
        features={name: Spline(kind="ps", k=8) for name in ("age", "bm", "veh")},
    )
    model._add_interaction("age", "bm")
    model._add_interaction("age", "veh")
    return model


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


class TestDiscretePredictParity:
    def test_exact_spline_prediction_evaluates_repeated_values_once(self, monkeypatch):
        values = np.repeat(np.linspace(0.0, 10.0, 20), 50)
        rng = np.random.default_rng(601)
        y = rng.poisson(np.exp(-0.5 + 0.1 * np.sin(values))).astype(float)
        X = pd.DataFrame({"x": values})
        model = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            discrete=True,
            n_bins=256,
            features={"x": Spline(n_knots=8, penalty="ssp")},
        )
        model.fit(X, y)

        spec = model._specs["x"]
        original = spec._basis_matrix

        def bounded_basis(x):
            if len(x) > 20:
                raise AssertionError("exact spline scoring materialized repeated training rows")
            return original(x)

        monkeypatch.setattr(spec, "_basis_matrix", bounded_basis)

        prediction = model.predict(X)
        assert prediction.shape == (len(X),)
        assert np.all(np.isfinite(prediction))

    def test_fast_discrete_predict_matches_exact_canonical_predict_for_main_effects(
        self, poisson_data
    ):
        X, y = poisson_data
        model = SuperGLM(
            family="poisson",
            selection_penalty=0.01,
            discrete=True,
            n_bins=256,
            features={"x1": Spline(n_knots=10, penalty="ssp"), "x2": Numeric()},
        )
        model.fit(X, y)

        eta_exact = model._predict_eta_exact(X)
        eta_fast = model._predict_eta_fast_discrete(X)
        mu_exact = model.predict(X)
        mu_fast = model._predict_fast_discrete(X)

        assert np.max(np.abs(eta_exact - eta_fast)) < 3e-2
        assert np.max(np.abs(mu_exact - mu_fast)) < 7e-2

    def test_fast_discrete_predict_matches_exact_canonical_predict_for_tensor_terms(
        self, tensor_interaction_data
    ):
        X, y = tensor_interaction_data
        model = SuperGLM(
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
        model.fit_reml(X, y, max_reml_iter=6)

        eta_exact = model._predict_eta_exact(X)
        eta_fast = model._predict_eta_fast_discrete(X)
        mu_exact = model.predict(X)
        mu_fast = model._predict_fast_discrete(X)

        assert np.max(np.abs(eta_exact - eta_fast)) < 3e-2
        assert np.max(np.abs(mu_exact - mu_fast)) < 3e-2

    def test_fast_discrete_tensor_predict_matches_exact_on_shifted_holdout(
        self, tensor_interaction_data
    ):
        X, y = tensor_interaction_data
        model = SuperGLM(
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
        model.fit_reml(X, y, max_reml_iter=6)

        holdout = pd.DataFrame(
            {
                "age": np.linspace(20.0, 78.0, 1200),
                "bm": np.linspace(17.0, 43.0, 1200)[::-1],
            }
        )

        eta_exact = model._predict_eta_exact(holdout)
        eta_fast = model._predict_eta_fast_discrete(holdout)
        mu_exact = model._predict_exact(holdout)
        mu_fast = model._predict_fast_discrete(holdout)

        assert np.max(np.abs(eta_exact - eta_fast)) < 3e-2
        assert np.max(np.abs(mu_exact - mu_fast)) < 2e-2

    def test_fast_discrete_tensor_metadata_is_frozen_at_fit_time(self, tensor_interaction_data):
        X, y = tensor_interaction_data
        model = SuperGLM(
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
        model.fit_reml(X, y, max_reml_iter=6)

        assert model._prediction_plan is not None
        interaction_plan = next(
            term for term in model._prediction_plan["interactions"] if term["name"] == "age:bm"
        )
        assert interaction_plan["fast_discrete"] is not None

        holdout = pd.DataFrame(
            {
                "age": np.linspace(20.0, 78.0, 1200),
                "bm": np.linspace(17.0, 43.0, 1200)[::-1],
            }
        )

        model._fit_X_ref = pd.DataFrame(
            {
                "age": np.linspace(30.0, 35.0, len(X)),
                "bm": np.linspace(22.0, 24.0, len(X)),
            }
        )
        model._n_bins = {"age": 5, "bm": 4}
        model._discrete = False

        eta_exact = model._predict_eta_exact(holdout)
        eta_fast = model._predict_eta_fast_discrete(holdout)
        mu_exact = model._predict_exact(holdout)
        mu_fast = model._predict_fast_discrete(holdout)

        assert np.max(np.abs(eta_exact - eta_fast)) < 3e-2
        assert np.max(np.abs(mu_exact - mu_fast)) < 2e-2


class TestSharedMarginalTensors:
    """Two ``ti()`` terms sharing a marginal, as in the reference implementation functional ANOVA.

    ``s(a)+s(b)+s(c)+ti(a,b)+ti(a,c)`` is a mainstream modelling pattern.  The
    ``ti()`` marginals are already centered, so the two blocks are identifiable
    side by side and the discrete path must build the same spans as the exact
    path rather than constraining one tensor against the other.
    """

    def test_shared_marginal_tensors_keep_exact_path_widths(self, shared_marginal_tensor_data):
        """Discrete tensor blocks must not shrink relative to the exact path."""
        X, y, exposure = shared_marginal_tensor_data

        widths = {}
        for discrete in (False, True):
            model = _shared_marginal_tensor_model(discrete)
            model._build_design_matrix(X, y, exposure, None)
            widths[discrete] = [gm.shape[1] for gm in model._dm.group_matrices]

        assert widths[True] == widths[False], (
            "discrete design collapsed a shared-marginal tensor: "
            f"{widths[True]} vs exact {widths[False]}"
        )

    def test_shared_marginal_tensors_reml_matches_exact(self, shared_marginal_tensor_data):
        """fit_reml(discrete=True) must fit, and agree with the exact path."""
        X, y, exposure = shared_marginal_tensor_data

        model_exact = _shared_marginal_tensor_model(False)
        model_exact.fit_reml(X, y, sample_weight=exposure)

        model_disc = _shared_marginal_tensor_model(True)
        model_disc.fit_reml(X, y, sample_weight=exposure)

        dev_exact = model_exact.result.deviance
        dev_disc = model_disc.result.deviance
        rel_dev = abs(dev_exact - dev_disc) / abs(dev_exact)
        assert rel_dev < 0.005, f"Relative deviance difference {rel_dev:.6f} too large"

        edf_exact = model_exact.metrics(X, y, sample_weight=exposure).effective_df
        edf_disc = model_disc.metrics(X, y, sample_weight=exposure).effective_df
        assert abs(edf_exact - edf_disc) < 0.5, (
            f"Effective df {edf_disc:.3f} differs from exact {edf_exact:.3f}"
        )

        pred_exact = model_exact.predict(X)
        pred_disc = model_disc.predict(X)
        mean_rel = np.mean(np.abs(pred_exact - pred_disc) / (pred_exact + 1e-10))
        assert mean_rel < 0.03, f"Mean relative prediction difference {mean_rel:.4f} too large"


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

    def test_exact_tensor_prediction_evaluates_repeated_margins_once(self, monkeypatch):
        """Exact tensor scoring should not rebuild marginal bases on repeated rows."""
        ages = np.arange(18, 30, dtype=np.float64)
        bms = np.arange(20, 28, dtype=np.float64)
        grid = np.array(np.meshgrid(ages, bms)).reshape(2, -1).T
        X = pd.DataFrame(
            {
                "age": np.repeat(grid[:, 0], 6),
                "bm": np.repeat(grid[:, 1], 6),
            }
        )
        rng = np.random.default_rng(322)
        y = rng.poisson(0.5, len(X)).astype(float)
        model = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            discrete=True,
            n_bins=256,
            features={
                "age": Spline(n_knots=6, penalty="ssp"),
                "bm": Spline(n_knots=5, penalty="ssp"),
            },
            interactions=[("age", "bm")],
        )
        model.fit(X, y)

        spec = model._interaction_specs["age:bm"]
        original = spec._centered_marginal_basis

        def bounded_marginal_basis(values, info):
            limit = len(ages) if info is spec._marginal1 else len(bms)
            if len(values) > limit:
                raise AssertionError("exact tensor scoring materialized repeated training rows")
            return original(values, info)

        monkeypatch.setattr(spec, "_centered_marginal_basis", bounded_marginal_basis)

        prediction = model.predict(X)
        assert prediction.shape == (len(X),)
        assert np.all(np.isfinite(prediction))

    def test_exact_tensor_prediction_batches_large_observed_support(self, monkeypatch):
        import scipy.sparse as sp

        from superglm.features import interaction as interaction_module
        from superglm.features.interaction import TensorInteraction

        support = np.arange(5, dtype=float)
        x1 = np.repeat(support, len(support))
        x2 = np.tile(support, len(support))
        spec = TensorInteraction("x1", "x2")
        spec._p1 = 10
        spec._p2 = 10
        spec._marginal1 = object()
        spec._marginal2 = object()

        def marginal_basis(values, _info):
            values = np.asarray(values, dtype=float)
            columns = np.arange(1.0, 11.0)
            return sp.csr_matrix(np.sin(values[:, None] + columns[None, :]))

        monkeypatch.setattr(spec, "_centered_marginal_basis", marginal_basis)
        monkeypatch.setattr(interaction_module, "_MAX_TENSOR_SCORE_SUPPORT_CELLS", 100)
        original_einsum = interaction_module.np.einsum
        batch_sizes: list[int] = []

        def bounded_einsum(signature, left, coefficients, right, **kwargs):
            batch_sizes.append(len(left))
            if len(left) > 3:
                raise AssertionError("observed tensor support exceeded its batch memory bound")
            return original_einsum(signature, left, coefficients, right, **kwargs)

        monkeypatch.setattr(interaction_module.np, "einsum", bounded_einsum)
        beta = np.linspace(-0.5, 0.75, 100)

        actual = spec.score(x1, x2, beta)

        B1 = marginal_basis(x1, spec._marginal1).toarray()
        B2 = marginal_basis(x2, spec._marginal2).toarray()
        expected = original_einsum("ij,jk,ik->i", B1, beta.reshape(10, 10), B2)
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
        assert batch_sizes
        assert max(batch_sizes) <= 3

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

    def test_discrete_tensor_retains_only_support_sized_marginal_bases(
        self, tensor_interaction_data
    ):
        """Discrete tensor metadata must not retain full observation-row marginal bases."""
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

        tensor = model._dm.group_matrices[2]
        spec = model._interaction_specs["age:bm"]
        assert isinstance(tensor, DiscretizedTensorGroupMatrix)
        assert spec._marginal1.basis.shape[0] == tensor.n_bins1
        assert spec._marginal2.basis.shape[0] == tensor.n_bins2
        np.testing.assert_allclose(
            np.bincount(tensor.idx1, minlength=tensor.n_bins1) @ spec._marginal1.basis,
            0.0,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            np.bincount(tensor.idx2, minlength=tensor.n_bins2) @ spec._marginal2.basis,
            0.0,
            atol=1e-10,
        )

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

    def test_rebuild_design_matrix_freezes_unprojected_tensor_basis(
        self, tensor_interaction_data, monkeypatch
    ):
        """Changing tensor component lambdas should not rebuild the packed tensor basis."""
        import superglm.dm_builder as dm_builder

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

        tensor_idx = next(i for i, g in enumerate(model._groups) if g.feature_name == "age:bm")
        tensor_group = model._groups[tensor_idx]
        tensor_gm = model._dm.group_matrices[tensor_idx]
        assert isinstance(tensor_gm, DiscretizedTensorGroupMatrix)
        assert tensor_gm.projection is None
        assert tensor_gm.omega_components is not None

        lambdas = {
            f"{tensor_group.name}:{suffix}": 2.0 + i
            for i, (suffix, _omega) in enumerate(tensor_gm.omega_components)
        }

        def fail_reparam(*args, **kwargs):
            raise AssertionError("unprojected tensor basis should stay fixed")

        monkeypatch.setattr(dm_builder, "compute_R_inv", fail_reparam)
        monkeypatch.setattr(dm_builder, "compute_projected_R_inv", fail_reparam)

        rebuilt = dm_builder.rebuild_design_matrix_with_lambdas(
            model._dm,
            model._groups,
            lambdas,
            np.ones(len(X)),
            lambdas,
        )

        assert rebuilt.group_matrices[tensor_idx] is tensor_gm
        assert model._dm._centered_pattern_plan is not None
        assert rebuilt._centered_pattern_plan is model._dm._centered_pattern_plan

    def test_discrete_tensor_reml_reports_rebuild_phase_profile(self, tensor_interaction_data):
        """Discrete REML profile should expose rebuild and tensor-summary phase timings."""
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
        model.fit_reml(X, y, max_reml_iter=2)

        profile = model._reml_profile
        for key in (
            "reml_rebuild_dm_s",
            "reml_map_beta_s",
            "reml_penalty_context_s",
            "reml_tensor_summary_s",
            "irls_eta_s",
            "irls_deviance_eval_s",
        ):
            assert key in profile
            assert profile[key] >= 0.0

    def test_fast_candidate_interaction_mode_caps_reml_outer_iterations(
        self, tensor_interaction_data
    ):
        """Candidate interaction fits should cap REML updates but still finalize the fit."""
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
        model.fit_reml(X, y, max_reml_iter=12, interaction_mode="fast_candidate")

        profile = model._reml_profile
        assert profile["interaction_mode"] == "fast_candidate"
        assert profile["interaction_candidate_active"] is True
        assert profile["requested_max_reml_iter"] == 12
        assert profile["effective_max_reml_iter"] == 5
        assert profile["fit_runtime_canonicalize_validate"] is False
        assert profile["n_reml_iter"] <= 5
        assert model.result.converged
        assert np.all(np.isfinite(model.predict(X.iloc[:25])))
        assert model._runtime_canonical_state["diagnostics"]["skipped"] is True

    def test_fast_candidate_interaction_mode_does_not_cap_main_effect_models(
        self, tensor_interaction_data
    ):
        """The candidate cap is interaction-specific and leaves main-only REML untouched."""
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
        model.fit_reml(X, y, max_reml_iter=7, interaction_mode="fast_candidate")

        profile = model._reml_profile
        assert profile["interaction_mode"] == "fast_candidate"
        assert profile["interaction_candidate_active"] is False
        assert profile["requested_max_reml_iter"] == 7
        assert profile["effective_max_reml_iter"] == 7
        assert profile["fit_runtime_canonicalize_validate"] is True

    def test_fit_reml_rejects_unknown_interaction_mode(self, tensor_interaction_data):
        """Unknown interaction candidate modes should fail before fitting."""
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

        with pytest.raises(ValueError, match="interaction_mode"):
            model.fit_reml(X, y, max_reml_iter=5, interaction_mode="fast")

    def test_runtime_validation_auto_skips_large_fit(self, tensor_interaction_data, monkeypatch):
        """Auto runtime validation should skip the full training-row diagnostic on large fits."""
        from superglm.model import fit_ops

        X, y = tensor_interaction_data
        monkeypatch.setattr(fit_ops, "_AUTO_RUNTIME_VALIDATION_MAX_ROWS", 10)
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

        model.fit_reml(X, y, max_reml_iter=2, runtime_validation="auto")

        assert model._reml_profile["fit_runtime_canonicalize_validate"] is False
        assert model._reml_profile["fit_runtime_canonicalize_validate_reason"] == "large_fit"
        assert model._runtime_canonical_state["diagnostics"]["skipped"] is True
        assert np.all(np.isfinite(model.predict(X.iloc[:25])))

    def test_runtime_validation_full_overrides_large_fit_auto_skip(
        self, tensor_interaction_data, monkeypatch
    ):
        """Explicit full runtime validation should remain available for large fits."""
        from superglm.model import fit_ops

        X, y = tensor_interaction_data
        monkeypatch.setattr(fit_ops, "_AUTO_RUNTIME_VALIDATION_MAX_ROWS", 10)
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

        model.fit_reml(X, y, max_reml_iter=2, runtime_validation="full")

        assert model._reml_profile["fit_runtime_canonicalize_validate"] is True
        assert model._reml_profile["fit_runtime_canonicalize_validate_reason"] == "explicit_full"
        assert "skipped" not in model._runtime_canonical_state["diagnostics"]

    def test_fit_reml_rejects_unknown_runtime_validation(self, tensor_interaction_data):
        """Unknown runtime validation modes should fail before fitting."""
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

        with pytest.raises(ValueError, match="runtime_validation"):
            model.fit_reml(X, y, max_reml_iter=2, runtime_validation="sometimes")

    def test_discrete_reml_forwards_public_pirls_controls(
        self, tensor_interaction_data, monkeypatch
    ):
        """Discrete REML should honor public PIRLS tolerance and iteration controls."""
        from superglm.reml import discrete as discrete_reml

        X, y = tensor_interaction_data
        calls = []
        original = discrete_reml.fit_irls_direct

        def spy_fit_irls_direct(*args, **kwargs):
            calls.append({"tol": kwargs.get("tol"), "max_iter": kwargs.get("max_iter")})
            return original(*args, **kwargs)

        monkeypatch.setattr(discrete_reml, "fit_irls_direct", spy_fit_irls_direct)

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
        model.fit_reml(X, y, max_reml_iter=2, pirls_tol=1e-5, max_pirls_iter=7)

        assert calls
        assert all(call["tol"] == 1e-5 for call in calls)
        assert calls[0]["max_iter"] == 7
        assert calls[-1]["max_iter"] == 7
        assert {call["max_iter"] for call in calls[1:-1]} <= {1}

    def test_discrete_reml_tol_can_be_relaxed(self, tensor_interaction_data):
        """A loose REML tolerance should actually loosen discrete REML convergence."""
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

        model.fit_reml(X, y, max_reml_iter=8, reml_tol=1e9)

        assert model._reml_result.converged
        assert model._reml_result.n_reml_iter == 2

    def test_penalty_context_cache_reuses_frozen_tensor_components(
        self, tensor_interaction_data, monkeypatch
    ):
        """Penalty context rebuilds should reuse static tensor eigensystems."""
        from superglm.reml import penalty_algebra

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
        tensor_idx = next(i for i, g in enumerate(model._groups) if g.feature_name == "age:bm")
        reml_groups = [(tensor_idx, model._groups[tensor_idx])]
        cache = {}

        penalties_first, _, _ = penalty_algebra.build_penalty_context(
            model._dm.group_matrices,
            reml_groups,
            cache=cache,
        )
        assert any("age:bm" in pc.group_name for pc in penalties_first)

        def fail_eigvalsh(*args, **kwargs):
            raise AssertionError("cached penalty context should not recompute eigensystems")

        monkeypatch.setattr(penalty_algebra.np.linalg, "eigvalsh", fail_eigvalsh)

        penalties_second, _, _ = penalty_algebra.build_penalty_context(
            model._dm.group_matrices,
            reml_groups,
            cache=cache,
        )

        assert [pc.name for pc in penalties_second] == [pc.name for pc in penalties_first]

    def test_tensor_pair_summary_cache_reuses_static_marginal_eigenvalues(
        self, tensor_interaction_data, monkeypatch
    ):
        """Tensor logdet summary rebuilds should reuse static marginal eigensystems."""
        from superglm.model.reml_setup import collect_reml_groups
        from superglm.reml import penalty_algebra

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
        reml_groups = collect_reml_groups(model._groups, model._dm.group_matrices)
        cache = {}
        penalties, _, _ = penalty_algebra.build_penalty_context(
            model._dm.group_matrices,
            reml_groups,
            cache=cache,
        )
        summaries_first = penalty_algebra.build_tensor_pair_logdet_summaries(
            model._dm.group_matrices,
            penalties,
            cache=cache,
        )
        assert summaries_first

        def fail_eigvalsh(*args, **kwargs):
            raise AssertionError("cached tensor summaries should not recompute eigensystems")

        monkeypatch.setattr(penalty_algebra.np.linalg, "eigvalsh", fail_eigvalsh)

        summaries_second = penalty_algebra.build_tensor_pair_logdet_summaries(
            model._dm.group_matrices,
            penalties,
            cache=cache,
        )

        assert summaries_second.keys() == summaries_first.keys()

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

    @pytest.mark.parametrize("discrete", [False, True])
    def test_tensor_marginal_legacy_override_remains_compatible(self, discrete, monkeypatch):
        from types import MethodType

        from superglm.features.interaction import TensorInteraction
        from superglm.features.spline import PSpline

        x1 = np.linspace(0.0, 1.0, 80)
        x2 = np.linspace(-1.0, 2.0, 80)
        spec1 = PSpline(n_knots=6)
        spec2 = PSpline(n_knots=5)
        spec1.build(x1)
        spec2.build(x2)
        original = spec1.tensor_marginal_ingredients
        calls: list[int] = []

        def legacy_tensor_marginal(_self, values):
            calls.append(len(values))
            return original(values)

        monkeypatch.setattr(
            spec1,
            "tensor_marginal_ingredients",
            MethodType(legacy_tensor_marginal, spec1),
        )
        interaction = TensorInteraction("x1", "x2")

        if discrete:
            result = interaction.build_discrete(
                x1,
                x2,
                {"x1": spec1, "x2": spec2},
                (16, 12),
            )
            assert result.B1_unique.shape[0] == 16
        else:
            result = interaction.build(x1, x2, {"x1": spec1, "x2": spec2})
            assert result.n_cols > 0
        assert calls == [len(x1)]

    def test_discrete_tensor_compacts_override_that_ignores_support_kwargs(self, monkeypatch):
        from types import MethodType

        from superglm.features.interaction import TensorInteraction
        from superglm.features.spline import PSpline

        x1 = np.linspace(0.0, 1.0, 80)
        x2 = np.linspace(-1.0, 2.0, 80)
        spec1 = PSpline(n_knots=6)
        spec2 = PSpline(n_knots=5)
        spec1.build(x1)
        spec2.build(x2)
        original = spec1.tensor_marginal_ingredients

        def ignores_compact_kwargs(_self, values, **_kwargs):
            return original(values)

        monkeypatch.setattr(
            spec1,
            "tensor_marginal_ingredients",
            MethodType(ignores_compact_kwargs, spec1),
        )
        interaction = TensorInteraction("x1", "x2")

        result = interaction.build_discrete(
            x1,
            x2,
            {"x1": spec1, "x2": spec2},
            (16, 12),
        )

        assert result.B1_unique.shape[0] == 16
        assert result.B2_unique.shape[0] == 12
