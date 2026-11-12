"""Tests for fit-time discretization (BAM-style binning)."""

import warnings

import numpy as np
import pandas as pd
import pytest

from superglm import SuperGLM
from superglm._group_matrix import _group_matrix_algebra as algebra
from superglm._group_matrix._group_matrix_bins import discretize_column
from superglm.features import interaction as interaction_module
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

    @pytest.mark.parametrize("kind", [None, "bs"])
    def test_freml_select_true_on_heavy_tailed_quantile_knots(self, kind):
        """The default (cr) and bs select pairs raised PenaltyNumericalError
        ("stored derivative factor disagrees ...") at the discrete start, where the
        null penalty sits at its cap beside a small wiggle weight; 0.39's kindless
        P-spline fitted this column."""
        rng = np.random.default_rng(0)
        x = rng.lognormal(0.0, 2.0, 5000)
        y = rng.poisson(np.exp(-1.0 + 0.2 * np.tanh(np.log(x)))).astype(float)
        kw = {} if kind is None else {"kind": kind}
        model = SuperGLM(
            family="poisson",
            selection_penalty=0,
            discrete=True,
            features={"x": Spline(n_knots=10, knot_strategy="quantile_rows", select=True, **kw)},
        )
        model.fit_reml(pd.DataFrame({"x": x}), y)

        assert set(model._reml_lambdas) == {"x:null", "x:wiggle"}
        assert np.all(np.isfinite(model.predict(pd.DataFrame({"x": x}))))

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


class TestSharedMarginalTensors:
    """Two ``ti()`` terms sharing a marginal, as in mgcv functional ANOVA.

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
                "age": Spline(kind="ps", n_knots=10, penalty="ssp"),
                "bm": Spline(kind="ps", n_knots=8, penalty="ssp"),
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
                "age": Spline(kind="ps", n_knots=10, penalty="ssp"),
                "bm": Spline(kind="ps", n_knots=8, penalty="ssp"),
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
        assert profile["n_reml_iter"] <= 5
        assert model.result.converged
        assert np.all(np.isfinite(model.predict(X.iloc[:25])))

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

    @pytest.mark.parametrize("value", ["auto", "full", "skip", False])
    def test_runtime_validation_is_an_accepted_no_op_that_warns(
        self, tensor_interaction_data, value
    ):
        """The retired post-fit check's switch is still accepted, warns, and changes nothing."""
        X, y = tensor_interaction_data

        def model():
            return SuperGLM(
                family="poisson",
                selection_penalty=0.0,
                discrete=True,
                features={
                    "age": Spline(n_knots=10, penalty="ssp"),
                    "bm": Spline(n_knots=8, penalty="ssp"),
                },
            )

        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            plain = model().fit_reml(X, y, max_reml_iter=2)
        with pytest.warns(FutureWarning, match="runtime_validation"):
            legacy = model().fit_reml(X, y, max_reml_iter=2, runtime_validation=value)

        np.testing.assert_array_equal(legacy.predict(X), plain.predict(X))
        assert "diagnostics" not in legacy._runtime_canonical_state

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


def _two_pair_frame():
    """Poisson counts with two planted tensor pairs over four P-spline margins."""
    rng = np.random.default_rng(2026)
    n = 20_000
    x = {name: rng.uniform(-1.0, 1.0, n) for name in ("x0", "x1", "x2", "x3")}
    eta = (
        -0.5
        + 0.6 * np.sin(2.5 * x["x0"])
        + 0.4 * x["x1"] ** 2
        - 0.3 * np.cos(2.0 * x["x3"])
        + 0.5 * x["x0"] * x["x1"]
        + 0.4 * np.sin(2.0 * x["x2"]) * x["x3"]
    )
    y = rng.poisson(np.exp(eta)).astype(float)
    return pd.DataFrame(x), y


def _fit_two_pairs(X, y, *, reml, decline_channel=False, decline_raw=False):
    """Fit the two-pair model and count the tensor x tensor routes it took.

    The solvers do not hand a profile dict down to the block assembler, so the
    routes are observed the way the dispatch tests observe them: through the
    module-level names the assembler resolves at call time.  The control arms
    force a decline at its source: the channel arm patches the helper itself
    rather than the aggregate-cell budget, which other routes in the same fit
    read; the raw arm patches the band attachment in ``build_discrete``, so
    every constructor site carries ``raw_channels=None`` and the channel
    route runs its dense stage.
    """
    routes = {"channel": 0, "declined": 0, "rows": 0, "raw": 0}
    helper = algebra._cross_gram_tensor_tensor_channels
    rows = algebra._support_support_raw_cross
    raw_kernel = algebra._cell_hist_raw_kron

    def counted_helper(*args, **kwargs):
        result = None if decline_channel else helper(*args, **kwargs)
        routes["channel" if result is not None else "declined"] += 1
        return result

    def counted_rows(*args, **kwargs):
        routes["rows"] += 1
        return rows(*args, **kwargs)

    def counted_raw(*args):
        routes["raw"] += 1
        return raw_kernel(*args)

    # 64 bins per margin: each tensor observes about 4,000 of its 4,096 cells,
    # so the tensor x tensor block's n_joint (about 1.6e7) is above the
    # histogram cell cap and the displaced route is the row-expanding one.
    model = SuperGLM(
        family="poisson",
        discrete=True,
        n_bins=64,
        features={name: Spline(kind="ps", k=8) for name in X.columns},
        interactions=[("x0", "x1"), ("x2", "x3")],
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(algebra, "_cross_gram_tensor_tensor_channels", counted_helper)
        patch.setattr(algebra, "_support_support_raw_cross", counted_rows)
        patch.setattr(algebra, "_cell_hist_raw_kron", counted_raw)
        if decline_raw:
            patch.setattr(interaction_module, "_tensor_raw_channels", lambda *args: None)
        if reml:
            model.fit_reml(X, y, max_reml_iter=30)
        else:
            model.fit(X, y)
    return model, routes


def _assert_routes_differ(channel_routes, dense_routes):
    assert channel_routes["channel"] > 0
    assert channel_routes["declined"] == 0
    assert channel_routes["rows"] == 0
    assert dense_routes["channel"] == 0
    assert dense_routes["declined"] > 0
    # Every decline lands on the row route the channel route displaced.
    assert dense_routes["rows"] == dense_routes["declined"]


def test_distinct_margin_tensor_route_reproduces_the_dense_fit():
    """The channel route changes summation order, not the fit.

    At fixed smoothing the IRLS fixed point contracts a Gram perturbation:
    the two routes agree to 1e-14 on predictions and 1e-11 on coefficients,
    the same as re-chunking the dense route's own row sum gives (measured).
    """
    X, y = _two_pair_frame()
    channel, channel_routes = _fit_two_pairs(X, y, decline_channel=False, reml=False)
    dense, dense_routes = _fit_two_pairs(X, y, decline_channel=True, reml=False)
    _assert_routes_differ(channel_routes, dense_routes)

    np.testing.assert_allclose(channel.predict(X), dense.predict(X), rtol=1e-12, atol=0)
    assert channel.result.deviance == pytest.approx(dense.result.deviance, rel=1e-12)
    assert channel.result.effective_df == pytest.approx(dense.result.effective_df, rel=1e-12)
    assert channel.result.phi == pytest.approx(dense.result.phi, rel=1e-12)
    np.testing.assert_allclose(channel.result.beta, dense.result.beta, rtol=1e-10, atol=0)


def test_distinct_margin_tensor_route_reaches_the_same_reml_optimum():
    """Through REML the routes reach the same optimum, not the same coordinates.

    The smoothing parameters are flat to about 2.5e-7 relative at this
    fixture's optimum, so a 1e-15 change in the Gram moves the fitted lambdas
    by that much, the coefficients by 3e-9 and the predictions by 1e-11,
    while the objective moves by 3e-13 and the deviance by 5e-15 (measured).
    That spread is the fit's own, not the route's: re-chunking the dense
    route's row sum, a summation-order change inside the displaced route,
    moves them by 9e-8, 7e-9, 6e-12, 7e-13 and 3e-15.  So the optimum is
    asserted and the coordinates are not.
    """
    X, y = _two_pair_frame()
    channel, channel_routes = _fit_two_pairs(X, y, decline_channel=False, reml=True)
    dense, dense_routes = _fit_two_pairs(X, y, decline_channel=True, reml=True)
    _assert_routes_differ(channel_routes, dense_routes)

    channel_reml = channel.reml_diagnostics()
    dense_reml = dense.reml_diagnostics()
    assert channel_reml["termination_reason"] != "max_reml_iter"
    assert channel_reml["termination_reason"] == dense_reml["termination_reason"]
    assert channel_reml["n_reml_iter"] == dense_reml["n_reml_iter"]
    assert channel_reml["objective"] == pytest.approx(dense_reml["objective"], rel=1e-11)
    assert channel.result.deviance == pytest.approx(dense.result.deviance, rel=1e-12)
    assert channel.result.phi == pytest.approx(dense.result.phi, rel=1e-12)
    np.testing.assert_allclose(channel.predict(X), dense.predict(X), rtol=1e-10, atol=0)
    assert channel.result.effective_df == pytest.approx(dense.result.effective_df, rel=1e-10)


def _assert_raw_stage_differs(raw_routes, dense_routes):
    # Every channel block of the raw arm ran the raw kernel, across all the
    # rebuilds of the REML loop, so a constructor site dropping raw_channels
    # would show as a shortfall here; the dense arm never reaches the kernel.
    assert raw_routes["channel"] > 0
    assert raw_routes["raw"] == raw_routes["channel"]
    assert dense_routes["channel"] > 0
    assert dense_routes["raw"] == 0


def test_raw_channel_stage_reproduces_the_dense_fit():
    """The raw band changes the arithmetic of one stage, not the fit.

    Same tolerances as the channel-versus-dense-route test above: the raw
    stage forms ``(w * raw1) * raw2`` per row in cell order and projects
    after the grid contraction, a round-off perturbation of the Gram that the
    IRLS fixed point contracts.
    """
    X, y = _two_pair_frame()
    raw, raw_routes = _fit_two_pairs(X, y, reml=False)
    dense, dense_routes = _fit_two_pairs(X, y, reml=False, decline_raw=True)
    _assert_raw_stage_differs(raw_routes, dense_routes)

    np.testing.assert_allclose(raw.predict(X), dense.predict(X), rtol=1e-12, atol=0)
    assert raw.result.deviance == pytest.approx(dense.result.deviance, rel=1e-12)
    assert raw.result.effective_df == pytest.approx(dense.result.effective_df, rel=1e-12)
    assert raw.result.phi == pytest.approx(dense.result.phi, rel=1e-12)
    np.testing.assert_allclose(raw.result.beta, dense.result.beta, rtol=1e-10, atol=0)


def test_raw_channel_stage_reaches_the_same_reml_optimum():
    """Through REML the raw and dense stages reach the same optimum."""
    X, y = _two_pair_frame()
    raw, raw_routes = _fit_two_pairs(X, y, reml=True)
    dense, dense_routes = _fit_two_pairs(X, y, reml=True, decline_raw=True)
    _assert_raw_stage_differs(raw_routes, dense_routes)

    raw_reml = raw.reml_diagnostics()
    dense_reml = dense.reml_diagnostics()
    assert raw_reml["termination_reason"] != "max_reml_iter"
    assert raw_reml["termination_reason"] == dense_reml["termination_reason"]
    assert raw_reml["n_reml_iter"] == dense_reml["n_reml_iter"]
    assert raw_reml["objective"] == pytest.approx(dense_reml["objective"], rel=1e-11)
    assert raw.result.deviance == pytest.approx(dense.result.deviance, rel=1e-12)
    assert raw.result.phi == pytest.approx(dense.result.phi, rel=1e-12)
    np.testing.assert_allclose(raw.predict(X), dense.predict(X), rtol=1e-10, atol=0)
    assert raw.result.effective_df == pytest.approx(dense.result.effective_df, rel=1e-10)


def _wide_tensor_frame(n: int = 6000) -> tuple[pd.DataFrame, np.ndarray, dict[str, int]]:
    """A tensor beside five categoricals: the pattern key's radix product, about 1.3e17, passes 2**53."""
    rng = np.random.default_rng(5)
    levels = {"a": 300, "b": 200, "c": 150, "d": 100, "e": 60}
    X = pd.DataFrame(
        {
            "x1": rng.integers(0, 40, n).astype(float),
            "x2": rng.integers(0, 40, n).astype(float),
            **{
                name: [f"{name}{code}" for code in rng.integers(0, size, n)]
                for name, size in levels.items()
            },
        }
    )
    y = rng.poisson(np.exp(0.02 * X.x1 - 0.01 * X.x2)).astype(np.float64)
    return X, y, levels


def _wide_tensor_fit(X: pd.DataFrame, y: np.ndarray, levels: dict[str, int]) -> SuperGLM:
    return SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        discrete=True,
        features={
            "x1": Spline(n_knots=6),
            "x2": Spline(n_knots=6),
            **{name: Categorical() for name in levels},
        },
        interactions=[("x1", "x2")],
    ).fit(X, y)


def test_a_pattern_key_past_two_to_the_53_decodes_every_row():
    """Row patterns keyed past ``2**53`` decode to each row's own codes, and their Gram is the dense one.

    The key packs every group's code in mixed radix into a uint64.  Adding the
    signed codes ran in float64 (NumPy's uint64 + int64 loop), which drops the
    key's low digits past ``2**53``: rows took other rows' patterns and the
    centred Gram was wrong.  The pattern route is called directly at fixed
    weights, so no fit trajectory decides whether it runs.  Mutation: without
    the uint64 cast most rows decode to other codes and the Gram is off.
    """
    from superglm._group_matrix import _group_matrix_centered as centered
    from superglm.group_matrix import CategoricalGroupMatrix

    X, y, levels = _wide_tensor_frame()
    dm = _wide_tensor_fit(X, y, levels)._dm
    plan = centered._build_pattern_plan(dm)
    assert plan is not None
    assert np.prod(np.asarray(plan.sizes, dtype=np.float64)) > 2.0**53
    codes = np.column_stack(
        [
            gm.codes if isinstance(gm, CategoricalGroupMatrix) else gm.bin_idx
            for gm in dm.group_matrices
        ]
    )
    decoded = np.asarray(plan.unique_codes)[np.asarray(plan.row_patterns)]
    np.testing.assert_array_equal(decoded, codes)

    rng = np.random.default_rng(11)
    w = rng.uniform(0.5, 1.5, dm.n)
    z = rng.normal(size=dm.n)
    z_centered = z - (w @ z) / w.sum()
    attempted, centred = centered._try_pattern_tensor_centering(
        dm=dm, W=w, weighted_z=w * z_centered, sum_w=float(w.sum())
    )
    assert attempted and centred is not None
    _mean, gram, rhs = centred
    dense = dm.toarray()
    xc = dense - (w @ dense) / w.sum()
    reference_gram = xc.T @ (w[:, None] * xc)
    reference_rhs = xc.T @ (w * z_centered)
    # well-conditioned weights and a 0/1 or B-spline design: a few hundred
    # roundings per entry, far inside 1e-10 of the largest
    assert np.max(np.abs(gram - reference_gram)) <= 1e-10 * np.max(np.abs(reference_gram))
    assert np.max(np.abs(rhs - reference_rhs)) <= 1e-10 * np.max(np.abs(reference_rhs))


def test_a_loaded_design_rebuilds_its_pattern_plan():
    """A design read back from a pickle rebuilds its pattern plan from its codes.

    The plan is a cache of the codes, and a pickle's copy may come from a build
    whose keys lost digits past ``2**53``, so the pickle carries no plan and a
    load drops whatever plan it finds.  Mutations: keeping the plan in
    ``__getstate__``, or ``setdefault`` in ``__setstate__``, carries it through.
    """
    from superglm._group_matrix import _group_matrix_centered as centered

    X, y, levels = _wide_tensor_frame()
    dm = _wide_tensor_fit(X, y, levels)._dm
    plan = centered._pattern_plan(dm)
    assert plan is not None
    state = dm.__getstate__()
    assert state["_centered_pattern_plan"] is None
    state["_centered_pattern_plan"] = plan  # as a pickle from an older build carries it
    loaded = type(dm).__new__(type(dm))
    loaded.__setstate__(state)
    assert loaded._centered_pattern_plan is None
    rebuilt = centered._pattern_plan(loaded)
    np.testing.assert_array_equal(rebuilt.unique_codes, plan.unique_codes)
    np.testing.assert_array_equal(rebuilt.row_patterns, plan.row_patterns)


# ── Bins that resolve the knots and the rows ───────────────────────────────
# A spline's binned fit evaluates its basis at the bins' support points and
# predict at the rows' own values. A bin wider than a knot span hides the
# pieces of the curve inside it from the fit, and a bin holding most of the
# rows reads them all at one value, so there the bins follow the knots and the
# rows and each is read at its rows' mean (``discretize_column``). The noise sd
# is 0.3 throughout; 0.05 is a sixth of it.


def _equal_width_bins(x, n_bins):
    """``discretize_column`` as it was before it read knots: the reference for bit-identity."""
    x = np.asarray(x, dtype=np.float64).ravel()
    unique_vals, inverse = np.unique(x, return_inverse=True)
    if len(unique_vals) <= n_bins:
        return unique_vals.astype(np.float64), inverse.astype(np.intp)
    lo, hi = float(x.min()), float(x.max())
    if lo == hi:
        return np.array([lo]), np.zeros(len(x), dtype=np.intp)
    edges = np.linspace(lo, hi, n_bins + 1)
    bin_idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, n_bins - 1)
    return 0.5 * (edges[:-1] + edges[1:]), bin_idx


def _fit_design_eta(model):
    """The linear predictor on the fit's own (binned) design."""
    beta = np.asarray(model.result.beta, dtype=np.float64)
    return float(model.result.intercept) + np.asarray(model._dm.toarray(), dtype=np.float64) @ beta


def _rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def _lognormal_frame(n=10_000, seed=0):
    """85% of rows in the first of 256 equal-width bins, which holds several quantile knots."""
    rng = np.random.default_rng(seed)
    x = rng.lognormal(0.0, 1.5, n)
    y = np.sin(2.0 * np.log1p(x)) + rng.normal(0.0, 0.3, n)
    return pd.DataFrame({"x": x}), y


def _gaussian_spline_fit(X, y, knot_strategy, **kwargs):
    model = SuperGLM(
        family="gaussian",
        features={"x": Spline(kind="ps", k=12, knot_strategy=knot_strategy)},
        **kwargs,
    )
    model.fit_reml(X, y)
    return model


@pytest.mark.parametrize("knot_strategy", ["quantile", "quantile_tempered"])
def test_binned_fit_predicts_what_it_fitted_on_a_skewed_column(knot_strategy):
    X, y = _lognormal_frame()
    exact = _gaussian_spline_fit(X, y, knot_strategy)
    binned = _gaussian_spline_fit(X, y, knot_strategy, discrete=True, n_bins=256)
    prediction = np.asarray(binned.predict(X), dtype=np.float64)
    assert _rmse(prediction, _fit_design_eta(binned)) < 0.05
    assert _rmse(prediction, np.asarray(exact.predict(X), dtype=np.float64)) < 0.01


@pytest.fixture(scope="module")
def lognormal_uniform_fits():
    """Uniform knots on the skewed column: every span is wide, but 85% of the rows
    share the first equal-width bin. 50,000 rows, where those bins misread the
    fit by an rmse of about 0.25 and the maximum row by about 40."""
    X, y = _lognormal_frame(n=50_000)
    exact = _gaussian_spline_fit(X, y, "uniform")
    binned = _gaussian_spline_fit(X, y, "uniform", discrete=True, n_bins=256)
    return X, exact, binned


def test_binned_fit_predicts_what_it_fitted_when_one_bin_holds_most_rows(lognormal_uniform_fits):
    X, exact, binned = lognormal_uniform_fits
    prediction = np.asarray(binned.predict(X), dtype=np.float64)
    assert _rmse(prediction, _fit_design_eta(binned)) < 0.05
    assert _rmse(prediction, np.asarray(exact.predict(X), dtype=np.float64)) < 0.01


def test_binned_fit_predicts_what_it_fitted_at_the_column_maximum(lognormal_uniform_fits):
    X, _, binned = lognormal_uniform_fits
    row = int(np.argmax(X["x"].to_numpy()))
    prediction = np.asarray(binned.predict(X.iloc[[row]]), dtype=np.float64)[0]
    assert abs(prediction - _fit_design_eta(binned)[row]) < 0.05


def test_binned_fit_predicts_what_it_fitted_beside_piles_of_equal_values():
    """69% zeros, 2% ones and a tail towards 1 that clusters the tempered knots below 1."""
    rng = np.random.default_rng(1)
    n = 10_000
    u = rng.uniform(size=n)
    tail = np.clip(1.0 - rng.exponential(0.003, n), 0.0, 1.0)
    x = np.where(u < 0.69, 0.0, np.where(u < 0.71, 1.0, tail))
    f = np.where(x == 0.0, 0.0, np.sin(2.0 * np.log1p((1.0 - x) / 0.003)))
    y = f + rng.normal(0.0, 0.3, n)
    X = pd.DataFrame({"x": x})
    model = _gaussian_spline_fit(X, y, "quantile_tempered", discrete=True, n_bins=256)
    prediction = np.asarray(model.predict(X), dtype=np.float64)
    eta = _fit_design_eta(model)
    knotted = x > 0.0
    # Every knot sits below 1, inside one equal-width bin: the rows there are
    # the ones a fit on equal-width bins misreads (by about 77 at the pile of ones).
    assert _rmse(prediction[knotted], eta[knotted]) < 0.1
    assert np.max(np.abs(prediction[x == 1.0] - eta[x == 1.0])) < 0.05
    # The pile at 0 sits on its cell's edge; it is read at its own value.
    assert np.max(np.abs(prediction[x == 0.0] - eta[x == 0.0])) < 0.02
    assert _rmse(prediction, eta) < 0.05
    exact = _gaussian_spline_fit(X, y, "quantile_tempered")
    assert _rmse(prediction, np.asarray(exact.predict(X), dtype=np.float64)) < 0.02


def test_binned_tensor_of_skewed_columns_matches_the_exact_fit():
    rng = np.random.default_rng(2)
    n = 10_000
    a = rng.lognormal(0.0, 1.5, n)
    b = rng.lognormal(0.0, 1.2, n)
    f = (
        np.sin(2.0 * np.log1p(a))
        + 0.5 * np.cos(2.0 * np.log1p(b))
        + 0.3 * np.sin(np.log1p(a)) * np.cos(np.log1p(b))
    )
    y = f + rng.normal(0.0, 0.3, n)
    X = pd.DataFrame({"a": a, "b": b})
    fits = {}
    for discrete in (False, True):
        model = SuperGLM(
            family="gaussian",
            features={name: Spline(kind="ps", k=10, knot_strategy="quantile") for name in "ab"},
            interactions=[("a", "b")],
            discrete=discrete,
            n_bins=256,
        )
        model.fit_reml(X, y)
        fits[discrete] = model
    prediction = np.asarray(fits[True].predict(X), dtype=np.float64)
    assert _rmse(prediction, _fit_design_eta(fits[True])) < 0.05
    assert _rmse(prediction, np.asarray(fits[False].predict(X), dtype=np.float64)) < 0.05


def _assert_bins_unchanged(x, n_bins, knots):
    support, index = discretize_column(x, n_bins, knots)
    old_support, old_index = _equal_width_bins(x, n_bins)
    np.testing.assert_array_equal(support, old_support)
    np.testing.assert_array_equal(index, old_index)


def test_bins_that_already_resolve_the_knots_are_unchanged(poisson_data):
    rng = np.random.default_rng(5)
    x = rng.uniform(0.0, 10.0, 5000)
    _assert_bins_unchanged(x, 256, np.linspace(0.0, 10.0, 12))
    _assert_bins_unchanged(x, 64, np.linspace(0.0, 10.0, 12))
    # An existing fixture, through the fit's own design.
    X, y = poisson_data
    model = SuperGLM(
        family="poisson",
        selection_penalty=0,
        discrete=True,
        n_bins=64,
        features={"x1": Spline(n_knots=10, penalty="ssp"), "x2": Numeric()},
    )
    model.fit(X, y)
    gm = model._dm.group_matrices[0]
    assert isinstance(gm, DiscretizedSSPGroupMatrix)
    old_support, old_index = _equal_width_bins(X["x1"].to_numpy(), 64)
    np.testing.assert_array_equal(gm.bin_idx, old_index)
    np.testing.assert_array_equal(gm.B_unique, model._specs["x1"]._raw_basis_matrix(old_support))


def test_adaptive_cells_follow_the_rule():
    # Ten zeros and ten other rows on [0, 10] with a knot at 1: span [0, 1) has
    # a width share of 0.8 of the 8 equal-width bins, under b_min = 8 // 4 = 2.
    x = np.array([0.0] * 10 + [1.0, 2.0, 3.0, 3.0, 3.0, 7.0, 8.0, 8.5, 9.5, 10.0])
    support, index = discretize_column(x, 8, np.array([1.0]))
    # n_w = 4 cells on the knots: b_min = 4 // 4 = 1 per span, the other 2 by
    # width 1:9 (0.2, 1.8 -> 0, 1, then the larger remainder to [1, 10]), so
    # 1 and 3 cells: left edges 0, 1, 4, 7. n_q = 4: the lower quartiles are
    # the 5th, 10th and 15th smallest rows, 0, 0 and 3; 0 is the minimum and
    # is dropped. Cells [0, 1), [1, 3), [3, 4), [4, 7), [7, 10].
    np.testing.assert_allclose(support, [0.0, 1.5, 3.0, 5.5, 8.6], rtol=0, atol=1e-14)
    assert support[0] == 0.0 and support[2] == 3.0
    # The pile is read at its own value, the rows on the quantile edge 3 belong
    # to the cell on its right, and the empty cell [4, 7) keeps its midpoint.
    np.testing.assert_array_equal(index, [0] * 10 + [1, 1, 2, 2, 2, 4, 4, 4, 4, 4])
    # A crowded equal-width bin with no knot inside the column: n_w = 8
    # equal-width cells plus the row quantiles, so the pile's cell holds it alone.
    crowded = np.concatenate([np.zeros(60), np.linspace(2.0, 10.0, 40)])
    support, index = discretize_column(crowded, 16, np.array([-1.0, 11.0]))
    assert len(support) <= 16
    assert support[0] == 0.0 and np.all(index[:60] == 0)
    # Neither condition fails: the equal-width bins are kept.
    _assert_bins_unchanged(np.linspace(0.0, 10.0, 100), 16, np.array([5.0]))


def test_binned_support_never_exceeds_n_bins():
    rng = np.random.default_rng(11)
    for trial in range(200):
        n = int(rng.integers(50, 3000))
        x = rng.lognormal(0.0, float(rng.uniform(0.1, 3.0)), n)
        if trial % 3 == 0:
            x[rng.uniform(size=n) < 0.5] = x.min()
        lo, hi = float(x.min()), float(x.max())
        n_knots = int(rng.integers(0, 400))
        if trial % 2:
            knots = np.quantile(x, rng.uniform(size=n_knots))
        else:
            knots = rng.uniform(lo - 1.0, hi + 1.0, n_knots)
        n_bins = int(rng.choice([1, 2, 3, 4, 8, 32, 64, 256]))
        if trial % 5 == 1:
            x[rng.uniform(size=n) < 0.3] = x.max()
        if trial % 5 == 2:
            x[rng.uniform(size=n) < 0.4] = np.median(x)
        support, index = discretize_column(x, n_bins, knots)
        assert len(support) <= n_bins
        assert np.all(np.diff(support) > 0)
        assert index.shape == x.shape
        assert index.min() >= 0 and index.max() < len(support)
        # Nor below: like the equal-width bins, the grid never reads the
        # minimum and the maximum at one point.
        occupied = np.unique(index).size
        assert occupied >= min(n_bins, 2)
        inner = np.unique(knots[(knots > lo) & (knots < hi)])
        span = np.searchsorted(inner, x, side="right")
        n_spans = np.unique(span).size
        equal_width = _equal_width_bins(x, n_bins)[0]
        adaptive = len(support) != len(equal_width) or np.any(support != equal_width)
        if not adaptive:
            continue
        # The adaptive grid gives every span that holds rows a cell, so it is
        # used only when there are fewer such spans than bins, and it reads
        # every row inside its own knot span.
        assert n_spans < n_bins
        np.testing.assert_array_equal(np.searchsorted(inner, support[index], side="right"), span)
        # Each distinct lower quantile j / n_q (the smallest row with at least
        # ceil(j n / n_q) rows at or below it) starts an occupied cell, as the
        # minimum does; n_q is half the budget, or what one cell per span leaves.
        n_q = n_bins - n_bins // 2 if n_spans <= n_bins // 2 else n_bins - n_spans
        j = np.arange(1, n_q)
        quantiles = np.sort(x)[(j * n + n_q - 1) // n_q - 1]
        assert occupied >= np.unique(np.append(quantiles, lo)).size


def test_more_spans_than_the_knot_half_get_one_cell_each_and_keep_the_resolution():
    """One cell per span that holds rows, and the rows' quantiles for the rest of the budget.

    39 uniform knots and one at 5.01 make 41 spans on a regular column: the
    narrow span [5, 5.01) fails the equal-width test, and 41 spans are more
    than the 32 cells of the knot half. Those 32 cells used to be equal-width,
    and the quantiles of a regular column land on the same edges, so the fit
    saw 32 support points where the equal-width bins it replaced give 64, and
    misread the exact fit by an rmse of 0.030 against 0.008.
    """
    x = np.linspace(0.0, 10.0, 20001)
    knots = np.sort(np.append(np.linspace(0.0, 10.0, 41)[1:-1], 5.01))
    support, index = discretize_column(x, 64, knots)
    # 41 knot cells and 23 quantile cells less the one starting at the
    # minimum: the grid uses n_bins - 1 cells, against the 64 the equal-width
    # bins occupy.
    assert np.unique(index).size == 63
    inner = knots[(knots > 0.0) & (knots < 10.0)]
    np.testing.assert_array_equal(
        np.searchsorted(inner, support[index], side="right"),
        np.searchsorted(inner, x, side="right"),
    )
    rng = np.random.default_rng(7)
    y = np.sin(3.0 * x) + rng.normal(0.0, 0.3, x.size)
    X = pd.DataFrame({"x": x})
    predictions = []
    for discrete in (False, True):
        model = SuperGLM(
            family="gaussian",
            features={"x": Spline(kind="bs", knots=knots)},
            discrete=discrete,
            n_bins=64,
        )
        model.fit_reml(X, y)
        predictions.append(np.asarray(model.predict(X), dtype=np.float64))
    # 0.015 here; a twentieth of the noise sd bounds it.
    assert _rmse(predictions[1], predictions[0]) < 0.02


def test_a_budget_below_four_keeps_the_equal_width_bins():
    """A knot half of one cell would read every row at one point.

    With ``n_bins=2`` the adaptive grid had one knot cell and no quantile
    edge, so the fit below read all ten rows at one support point and
    predicted their mean, 4.5, everywhere.
    """
    x = np.arange(10.0)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0,
        discrete=True,
        n_bins=2,
        features={"x": Spline(kind="ps", k=8)},
    )
    model.fit(pd.DataFrame({"x": x}), x)
    knots = model._specs["x"]._knots
    for n_bins in (2, 3):
        _assert_bins_unchanged(x, n_bins, knots)
    np.testing.assert_array_equal(model._dm.group_matrices[0].bin_idx, _equal_width_bins(x, 2)[1])
    # The two bins, centred at 2.25 and 6.75, hold rows averaging 2 and 7.
    centres = model.predict(pd.DataFrame({"x": [2.25, 6.75]}))
    np.testing.assert_allclose(centres, [2.0, 7.0], rtol=0, atol=1e-2)


def test_a_column_near_the_float64_limit_bins_without_overflow():
    """The width shares of a column near ``1e308`` are formed without overflow.

    ``remaining * widths`` overflowed to infinity for the wide span, and the
    cast to a cell count made ``np.linspace`` refuse a negative count, in
    ``discretize_column`` and in the fit.
    """
    x = np.linspace(0.0, 1e307, 1000)
    knots = np.array([1e300])
    support, index = discretize_column(x, 256, knots)
    assert len(support) < 256  # the adaptive grid: the span [0, 1e300) has no bin
    # Scaling by a power of two is exact, so the column gets the bins its
    # scaled-down twin gets, scaled back.
    twin_support, twin_index = discretize_column(np.ldexp(x, -1000), 256, np.ldexp(knots, -1000))
    np.testing.assert_array_equal(support, np.ldexp(twin_support, 1000))
    np.testing.assert_array_equal(index, twin_index)
    model = SuperGLM(
        family="gaussian", discrete=True, features={"x": Spline(kind="ps", knots=[1e300])}
    )
    model.fit(pd.DataFrame({"x": x}), x / 1e307)
    np.testing.assert_allclose(model.predict(pd.DataFrame({"x": x})), x / 1e307, atol=1e-3)


# A frequency weight stands for that many literal rows, so the rule counts each
# row by its weight: the crowding test, the quantile edges and the cell means.


def _assert_weighted_bins_match_replication(x, weights, n_bins, knots):
    replicas = np.repeat(np.arange(x.size), weights.astype(np.intp))
    support, index = discretize_column(x, n_bins, knots, weights)
    replicated_support, replicated_index = discretize_column(x[replicas], n_bins, knots)
    np.testing.assert_array_equal(support, replicated_support)
    np.testing.assert_array_equal(index[replicas], replicated_index)
    equal_width = _equal_width_bins(x, n_bins)[0]
    assert len(support) != len(equal_width) or np.any(support != equal_width)
    # Counting each row once instead gives other bins.
    unweighted = discretize_column(x, n_bins, knots)[0]
    assert len(unweighted) != len(support) or np.any(unweighted != support)


def test_integer_weights_bin_like_their_replicated_rows():
    rng = np.random.default_rng(21)
    x = rng.lognormal(0.0, 1.5, 4000)
    weights = rng.integers(1, 7, x.size).astype(np.float64)
    knots = np.quantile(x, np.linspace(0.0, 1.0, 12))
    _assert_weighted_bins_match_replication(x, weights, 64, knots)
    # A pile of zeros that is crowded only once the rows count by weight: the
    # equal-width bins resolve the uniform knots and each holds about 1/64 of
    # the rows, until the zeros, a tenth of the rows, weigh 30 each.
    x = np.where(rng.uniform(size=x.size) < 0.1, 0.0, rng.uniform(0.0, 1.0, x.size))
    weights = np.where(x == 0.0, 30.0, rng.integers(1, 3, x.size).astype(np.float64))
    knots = np.linspace(0.0, 1.0, 8)
    _assert_bins_unchanged(x, 64, knots)
    _assert_weighted_bins_match_replication(x, weights, 64, knots)


def test_fractional_weights_read_each_cell_at_its_weighted_mean():
    rng = np.random.default_rng(22)
    x = rng.lognormal(0.0, 1.5, 5000)
    weights = rng.gamma(0.5, 2.0, x.size) + 1e-3
    knots = np.quantile(x, np.linspace(0.0, 1.0, 12))
    support, index = discretize_column(x, 64, knots, weights)
    assert len(support) <= 64
    assert np.all(np.diff(support) > 0)
    for cell in np.unique(index):
        rows = index == cell
        expected = np.sum(weights[rows] * x[rows]) / np.sum(weights[rows])
        np.testing.assert_allclose(support[cell], expected, rtol=1e-12, atol=0.0)
    # Rows of zero weight still have a support point inside their own range,
    # and a cell holding only such rows is not read at a 0/0 mean.
    weights[x > np.quantile(x, 0.995)] = 0.0
    support, index = discretize_column(x, 64, knots, weights)
    assert np.all(np.isfinite(support)) and np.all(np.diff(support) > 0)
    for cell in np.unique(index):
        rows = x[index == cell]
        assert rows.min() <= support[cell] <= rows.max()
