"""Tests for spline identifiability constraint (unweighted sum-to-zero).

Verifies acceptance criteria from the identifiability fix:
1. Unweighted mean contribution ≈ 0 for bs/ns/cr
2. Exact vs discrete path agreement
3. SE/CI are well-behaved (no runaway values)
4. SplineCategorical interaction compatibility
"""

import numpy as np
import pytest

from superglm import SuperGLM
from superglm.features.categorical import Categorical
from superglm.features.spline import BasisSpline, Spline


def _make_poisson_data(n=2000, seed=42):
    """Generate synthetic Poisson data with known exposure."""
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(18, 90, n)
    x2 = rng.uniform(0, 20, n)
    cat = rng.choice(["A", "B", "C", "D"], n)
    exposure = rng.uniform(0.1, 1.0, n)
    eta = -2.0 + 0.01 * (x1 - 40) ** 2 / 100 - 0.05 * x2
    mu = exposure * np.exp(eta)
    y = rng.poisson(mu).astype(np.float64)
    import pandas as pd

    df = pd.DataFrame({"x1": x1, "x2": x2, "cat": cat})
    return df, y, exposure


# ── Acceptance criterion 1: unweighted mean ≈ 0 ──────────────────


class TestUnweightedMeanZero:
    """For each spline kind, the unweighted mean contribution must be ≈ 0."""

    @pytest.fixture
    def data(self):
        return _make_poisson_data()

    @pytest.mark.parametrize("kind", ["bs", "ns", "cr"])
    def test_unweighted_mean_zero_reml(self, data, kind):
        df, y, exposure = data
        model = SuperGLM(
            family="poisson",
            lambda1=0.0,
            features={"x1": Spline(kind=kind, n_knots=8, penalty="ssp")},
        )
        model.fit_reml(df, y, exposure=exposure, max_reml_iter=10)

        spec = model._specs["x1"]
        fgroups = [g for g in model._groups if g.feature_name == "x1"]
        beta = np.concatenate([model.result.beta[g.sl] for g in fgroups])
        x_train = df["x1"].to_numpy(dtype=np.float64)
        B_train = spec.transform(x_train)
        f_train = B_train @ beta

        umean = np.mean(f_train)
        assert abs(umean) < 1e-10, f"Unweighted mean {umean:.2e} not zero for kind={kind}"

    @pytest.mark.parametrize("kind", ["bs", "ns", "cr"])
    def test_unweighted_mean_zero_fit(self, data, kind):
        """Also works with plain fit() (no REML)."""
        df, y, exposure = data
        model = SuperGLM(
            family="poisson",
            lambda1=0.0,
            features={"x1": Spline(kind=kind, n_knots=8, penalty="ssp")},
        )
        model.fit(df, y, exposure=exposure)

        spec = model._specs["x1"]
        fgroups = [g for g in model._groups if g.feature_name == "x1"]
        beta = np.concatenate([model.result.beta[g.sl] for g in fgroups])
        f_train = spec.transform(df["x1"].to_numpy(dtype=np.float64)) @ beta

        umean = np.mean(f_train)
        assert abs(umean) < 1e-10, f"Unweighted mean {umean:.2e} not zero for kind={kind}"

    @pytest.mark.parametrize("kind", ["bs", "ns", "cr"])
    def test_no_exposure_model(self, kind):
        """Identifiability works without exposure too."""
        df, y, _ = _make_poisson_data()
        model = SuperGLM(
            family="poisson",
            lambda1=0.0,
            features={"x1": Spline(kind=kind, n_knots=8, penalty="ssp")},
        )
        model.fit_reml(df, y, max_reml_iter=10)

        spec = model._specs["x1"]
        fgroups = [g for g in model._groups if g.feature_name == "x1"]
        beta = np.concatenate([model.result.beta[g.sl] for g in fgroups])
        f_train = spec.transform(df["x1"].to_numpy(dtype=np.float64)) @ beta

        umean = np.mean(f_train)
        assert abs(umean) < 1e-10, f"Unweighted mean {umean:.2e} not zero for kind={kind}"


class TestMultipleSplineMeanZero:
    """Unweighted mean ≈ 0 for each term in a multi-spline model."""

    def test_multi_spline_unweighted_mean(self):
        df, y, exposure = _make_poisson_data(n=3000)
        model = SuperGLM(
            family="poisson",
            lambda1=0.0,
            features={
                "x1": Spline(kind="bs", n_knots=10, penalty="ssp"),
                "x2": Spline(kind="bs", n_knots=8, penalty="ssp"),
                "cat": Categorical(base="first"),
            },
        )
        model.fit_reml(df, y, exposure=exposure, max_reml_iter=10)

        for name in ["x1", "x2"]:
            spec = model._specs[name]
            fgroups = [g for g in model._groups if g.feature_name == name]
            beta = np.concatenate([model.result.beta[g.sl] for g in fgroups])
            f_train = spec.transform(df[name].to_numpy(dtype=np.float64)) @ beta
            umean = np.mean(f_train)
            assert abs(umean) < 1e-10, f"Unweighted mean {umean:.2e} not zero for {name}"


# ── Acceptance criterion 2: exact vs discrete agreement ───────────


class TestExactVsDiscreteAgreement:
    """Exact and discretized paths should produce similar results."""

    @pytest.mark.parametrize("kind", ["bs", "ns", "cr"])
    def test_deviance_agreement(self, kind):
        df, y, exposure = _make_poisson_data(n=3000)
        features_exact = {"x1": Spline(kind=kind, n_knots=8, penalty="ssp", discrete=False)}
        features_disc = {
            "x1": Spline(kind=kind, n_knots=8, penalty="ssp", discrete=True, n_bins=256)
        }

        m_exact = SuperGLM(family="poisson", lambda1=0.0, features=features_exact)
        m_exact.fit_reml(df, y, exposure=exposure, max_reml_iter=10)

        m_disc = SuperGLM(family="poisson", lambda1=0.0, features=features_disc)
        m_disc.fit_reml(df, y, exposure=exposure, max_reml_iter=10)

        # Deviance should be close (within ~1% for 256 bins)
        dev_exact = m_exact.result.deviance
        dev_disc = m_disc.result.deviance
        rel_diff = abs(dev_exact - dev_disc) / max(abs(dev_exact), 1.0)
        assert rel_diff < 0.02, f"Deviance mismatch: exact={dev_exact:.2f}, disc={dev_disc:.2f}"

    @pytest.mark.parametrize("kind", ["bs", "ns", "cr"])
    def test_discrete_unweighted_mean_zero(self, kind):
        """Discretized path also has unweighted mean ≈ 0."""
        df, y, exposure = _make_poisson_data(n=3000)
        model = SuperGLM(
            family="poisson",
            lambda1=0.0,
            features={"x1": Spline(kind=kind, n_knots=8, penalty="ssp", discrete=True, n_bins=256)},
        )
        model.fit_reml(df, y, exposure=exposure, max_reml_iter=10)

        spec = model._specs["x1"]
        fgroups = [g for g in model._groups if g.feature_name == "x1"]
        beta = np.concatenate([model.result.beta[g.sl] for g in fgroups])
        f_train = spec.transform(df["x1"].to_numpy(dtype=np.float64)) @ beta
        umean = np.mean(f_train)
        assert abs(umean) < 1e-10, f"Discrete: unweighted mean {umean:.2e} not zero"


# ── Acceptance criterion 3: SE/CI behavior ────────────────────────


class TestSEBehavior:
    """SEs should be well-behaved: no runaway values, no NaN."""

    @pytest.mark.parametrize("kind", ["bs", "ns", "cr"])
    def test_se_finite_and_small(self, kind):
        df, y, exposure = _make_poisson_data(n=3000)
        model = SuperGLM(
            family="poisson",
            lambda1=0.0,
            features={"x1": Spline(kind=kind, n_knots=8, penalty="ssp")},
        )
        model.fit_reml(df, y, exposure=exposure, max_reml_iter=10)
        rels = model.relativities(with_se=True)
        se = rels["x1"]["se_log_relativity"].to_numpy()
        assert np.all(np.isfinite(se)), f"SEs contain non-finite values for kind={kind}"
        assert np.all(se >= 0), f"SEs contain negative values for kind={kind}"
        # SEs should not be runaway (> 10 would indicate something wrong)
        assert np.max(se) < 10, f"Max SE {np.max(se):.2f} is too large for kind={kind}"

    @pytest.mark.parametrize("kind", ["bs", "ns", "cr"])
    def test_simultaneous_bands_finite(self, kind):
        df, y, exposure = _make_poisson_data(n=3000)
        model = SuperGLM(
            family="poisson",
            lambda1=0.0,
            features={"x1": Spline(kind=kind, n_knots=8, penalty="ssp")},
        )
        model.fit_reml(df, y, exposure=exposure, max_reml_iter=10)
        bands = model.simultaneous_bands("x1")
        assert np.all(np.isfinite(bands["se"].to_numpy()))
        assert np.all(np.isfinite(bands["ci_lower_simultaneous"].to_numpy()))
        assert np.all(np.isfinite(bands["ci_upper_simultaneous"].to_numpy()))
        # Simultaneous bands should be wider than pointwise
        assert np.all(
            bands["ci_upper_simultaneous"].to_numpy()
            >= bands["ci_upper_pointwise"].to_numpy() - 1e-10
        )


# ── Acceptance criterion 4: SplineCategorical interaction ─────────


class TestInteractionCompatibility:
    """SplineCategorical should work with identifiability constraint."""

    def test_spline_categorical_interaction_fits(self):
        df, y, exposure = _make_poisson_data(n=3000)
        model = SuperGLM(
            family="poisson",
            lambda1=0.0,
            features={
                "x1": Spline(kind="bs", n_knots=8, penalty="ssp"),
                "cat": Categorical(base="first"),
            },
            interactions=[("x1", "cat")],
        )
        model.fit_reml(df, y, exposure=exposure, max_reml_iter=10)
        assert model.result is not None
        assert model.result.deviance > 0

    def test_spline_categorical_relativities(self):
        df, y, exposure = _make_poisson_data(n=3000)
        model = SuperGLM(
            family="poisson",
            lambda1=0.0,
            features={
                "x1": Spline(kind="bs", n_knots=8, penalty="ssp"),
                "cat": Categorical(base="first"),
            },
            interactions=[("x1", "cat")],
        )
        model.fit_reml(df, y, exposure=exposure, max_reml_iter=10)
        rels = model.relativities()
        # Per-level curves should exist for non-base levels
        for level in ["B", "C", "D"]:
            key = f"x1:cat[{level}]"
            assert key in rels, f"Missing interaction relativity for {key}"
            assert len(rels[key]) > 0


# ── Column count tests ────────────────────────────────────────────


class TestColumnCounts:
    """Verify k → n_cols = k-1 for all spline kinds."""

    @pytest.mark.parametrize(
        "kind,k,expected_ncols",
        [
            ("bs", 14, 13),
            ("ns", 10, 9),
            ("cr", 10, 9),
        ],
    )
    def test_k_produces_k_minus_1_cols(self, kind, k, expected_ncols):
        s = Spline(kind=kind, k=k)
        x = np.linspace(0, 1, 100)
        info = s.build(x)
        assert info.n_cols == expected_ncols, (
            f"kind={kind}, k={k}: expected {expected_ncols} cols, got {info.n_cols}"
        )


# ── Projection stored correctly ──────────────────────────────────


class TestInteractionProjection:
    """Verify _interaction_projection is set and has correct shape."""

    @pytest.mark.parametrize("kind", ["bs", "ns", "cr"])
    def test_interaction_projection_exists(self, kind):
        s = Spline(kind=kind, n_knots=8)
        x = np.linspace(0, 1, 100)
        s.build(x)
        assert s._interaction_projection is not None
        # Projection should be orthogonal
        P = s._interaction_projection
        PtP = P.T @ P
        assert np.allclose(PtP, np.eye(PtP.shape[0]), atol=1e-12)

    def test_select_stores_interaction_projection(self):
        """select=True stores interaction projection for interaction support."""
        s = BasisSpline(n_knots=8, select=True)
        x = np.linspace(0, 1, 100)
        s.build(x)
        # BS has no boundary constraints, so _interaction_projection is
        # just the identifiability projection (K, K-1)
        assert s._interaction_projection is not None
        assert s._interaction_projection.shape == (s._n_basis, s._n_basis - 1)
