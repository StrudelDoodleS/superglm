"""Tests for the Polynomial feature spec."""

import numpy as np
import pandas as pd
import pytest
from numpy.polynomial.legendre import legvander

from superglm import OrderedCategorical, Polynomial, SuperGLM
from superglm.features.polynomial import Polynomial as PolynomialDirect


def _heaped_exposure(n: int = 1500, seed: int = 31):
    """x with exposure heaped at low values — the defect-triggering shape."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 100.0, n)
    w = np.exp(-x / 15.0) + 0.02 * rng.uniform(0.5, 1.0, n)
    return x, w


class TestPolynomialSpec:
    def test_group_size_equals_degree(self):
        for deg in [1, 2, 3, 4]:
            info = PolynomialDirect(degree=deg).build(np.linspace(0, 100, 200))
            assert info.n_cols == deg

    def test_no_penalty_matrix(self):
        info = PolynomialDirect(degree=3).build(np.linspace(0, 1, 50))
        assert info.penalty_matrix is None
        assert info.reparametrize is False

    def test_columns_shape(self):
        x = np.linspace(0, 100, 300)
        info = PolynomialDirect(degree=3).build(x)
        assert info.columns.shape == (300, 3)

    def test_columns_near_orthogonal(self):
        """Legendre polynomials on uniformly spaced data should be nearly orthogonal."""
        x = np.linspace(0, 100, 1000)
        info = PolynomialDirect(degree=3).build(x)
        G = info.columns.T @ info.columns
        # Off-diagonal elements should be much smaller than diagonal
        diag = np.diag(G)
        off_diag = G - np.diag(diag)
        assert np.max(np.abs(off_diag)) < 0.1 * np.min(diag)

    def test_degree_zero_raises(self):
        with pytest.raises(ValueError, match="degree must be >= 1"):
            PolynomialDirect(degree=0)

    def test_transform_same_as_build(self):
        x = np.linspace(0, 50, 100)
        spec = PolynomialDirect(degree=2)
        info = spec.build(x)
        transformed = spec.transform(x)
        np.testing.assert_allclose(info.columns, transformed)

    def test_transform_new_data(self):
        spec = PolynomialDirect(degree=2)
        spec.build(np.linspace(0, 100, 200))
        new_x = np.array([25.0, 50.0, 75.0])
        result = spec.transform(new_x)
        assert result.shape == (3, 2)

    def test_reconstruct_keys(self):
        spec = PolynomialDirect(degree=3)
        spec.build(np.linspace(0, 100, 200))
        rec = spec.reconstruct(np.array([0.1, -0.05, 0.01]))
        assert "x" in rec
        assert "log_relativity" in rec
        assert "relativity" in rec
        assert "degree" in rec
        assert "coefficients" in rec
        assert rec["degree"] == 3
        assert rec["powers"] == (1, 2, 3)

    def test_reconstruct_curve_shape(self):
        spec = PolynomialDirect(degree=2)
        spec.build(np.linspace(0, 100, 200))
        rec = spec.reconstruct(np.array([0.0, 0.1]), n_points=50)
        assert rec["x"].shape == (50,)
        assert rec["relativity"].shape == (50,)
        assert np.all(rec["relativity"] > 0)

    def test_constant_feature_raises(self):
        """All-same values cannot identify any polynomial component.

        Before the data-orthogonal basis this silently emitted a constant,
        intercept-collinear block; the distinct-support guard now refuses.
        """
        spec = PolynomialDirect(degree=2)
        with pytest.raises(ValueError, match="distinct x values with positive weight"):
            spec.build(np.full(100, 5.0))


class TestPolynomialIntegration:
    @pytest.fixture
    def sample_data(self):
        rng = np.random.default_rng(42)
        n = 500
        age = rng.uniform(18, 85, n)
        region = rng.choice(["A", "B", "C"], n, p=[0.3, 0.3, 0.4])
        sample_weight = rng.uniform(0.3, 1.0, n)
        mu = np.exp(-2.0 + 0.01 * (age - 50) ** 2 / 100 + (region == "A") * 0.3)
        y = rng.poisson(mu * sample_weight).astype(float)
        X = pd.DataFrame({"age": age, "region": region})
        return X, y, sample_weight

    def test_fit_predict(self, sample_data):
        X, y, sample_weight = sample_data
        from superglm.features.categorical import Categorical

        model = SuperGLM(
            penalty="group_lasso",
            selection_penalty=0.01,
            features={
                "age": Polynomial(degree=3),
                "region": Categorical(base="first"),
            },
        )
        model.fit(X, y, sample_weight=sample_weight)
        preds = model.predict(X)
        assert preds.shape == (len(X),)
        assert np.all(preds > 0)

    def test_reconstruct_feature(self, sample_data):
        X, y, sample_weight = sample_data
        from superglm.features.categorical import Categorical

        model = SuperGLM(
            penalty="group_lasso",
            selection_penalty=0.01,
            features={
                "age": Polynomial(degree=2),
                "region": Categorical(base="first"),
            },
        )
        model.fit(X, y, sample_weight=sample_weight)
        rec = model.reconstruct_feature("age")
        assert "x" in rec
        assert "relativity" in rec

    def test_summary_shows_polynomial(self, sample_data):
        X, y, sample_weight = sample_data
        from superglm.features.categorical import Categorical

        model = SuperGLM(
            penalty="group_lasso",
            selection_penalty=0.01,
            features={
                "age": Polynomial(degree=2),
                "region": Categorical(base="first"),
            },
        )
        model.fit(X, y, sample_weight=sample_weight)
        s = model.diagnostics()
        assert "age" in s
        assert s["age"]["n_params"] == 2


class TestDataOrthogonalBasis:
    """The basis is orthonormalized against the training exposure weights."""

    def test_weighted_orthonormality_convention(self):
        # (1/sum w) * Phi' diag(w) Phi = I, and each column is w-orthogonal
        # to the constant.  The uniform-measure Legendre basis fails both
        # on heaped exposure.
        x, w = _heaped_exposure()
        info = PolynomialDirect(degree=4).build(x, sample_weight=w)
        Phi = info.columns
        total = float(w.sum())
        G = (Phi * w[:, None]).T @ Phi / total
        np.testing.assert_allclose(G, np.eye(4), atol=1e-10)
        col_means = (Phi * w[:, None]).sum(axis=0) / total
        np.testing.assert_allclose(col_means, 0.0, atol=1e-10)

    def test_per_power_covariance_diagonal_on_heaped_exposure(self):
        # Defect catcher: under Gaussian identity-link fixed-weight fitting,
        # Cov(beta_hat) ∝ inv(X' W X).  With the data-orthogonal basis the
        # per-power block is diagonal; the old Legendre basis on the same
        # fixture is measurably non-diagonal (the unreliable-z defect).
        x, w = _heaped_exposure()
        spec = PolynomialDirect(degree=4)
        info = spec.build(x, sample_weight=w)

        X = np.column_stack([np.ones(len(x)), info.columns])
        C = np.linalg.inv((X * w[:, None]).T @ X)
        block = C[1:, 1:]
        off = block - np.diag(np.diag(block))
        assert np.abs(off).max() < 1e-8 * np.diag(block).max()

        # The same fixture under the old Legendre construction: documented
        # measurably non-diagonal, which is why z-based drop tests lied.
        L = legvander(spec._scale(x), 4)[:, 1:]
        XL = np.column_stack([np.ones(len(x)), L])
        CL = np.linalg.inv((XL * w[:, None]).T @ XL)
        blockL = CL[1:, 1:]
        offL = blockL - np.diag(np.diag(blockL))
        assert np.abs(offL).max() > 0.1 * np.diag(blockL).max()

    def test_transform_pushes_through_stored_factor(self):
        # transform on new x = seed basis through the STORED triangular
        # factor; never re-orthogonalized against the new sample.
        x, w = _heaped_exposure()
        spec = PolynomialDirect(powers=[1, 2, 4])
        spec.build(x, sample_weight=w)

        new_x = np.array([5.0, 12.5, 40.0, 99.0])
        result = spec.transform(new_x)
        assert result.shape == (4, 3)
        assert np.all(np.isfinite(result))

        seed = legvander(spec._scale(new_x), spec.degree)
        expected = np.linalg.solve(spec._R.T, seed.T).T[:, [1, 2, 4]]
        np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)

        # Mutation check: transform actually reads the stored factor.
        spec._R = spec._R * 2.0
        assert np.abs(spec.transform(new_x) - result).max() > 1e-3

    def test_degree_is_sugar_for_full_ladder(self):
        x, w = _heaped_exposure()
        a = PolynomialDirect(degree=3).build(x, sample_weight=w)
        b = PolynomialDirect(powers=[1, 2, 3]).build(x, sample_weight=w)
        np.testing.assert_allclose(a.columns, b.columns, atol=1e-14)

    def test_unweighted_build_defaults_to_unit_weights(self):
        x = np.linspace(0.0, 100.0, 400)
        info = PolynomialDirect(degree=3).build(x)
        Phi = info.columns
        G = Phi.T @ Phi / len(x)
        np.testing.assert_allclose(G, np.eye(3), atol=1e-10)


class TestPredictionEquivalence:
    """Same column space modulo the constant: unpenalized fits are pinned."""

    def _fixture(self):
        rng = np.random.default_rng(42)
        n = 800
        x = rng.uniform(0.0, 100.0, n)
        w = np.exp(-x / 20.0) + 0.05
        xs = (x - 50.0) / 50.0
        y = rng.normal(1.0 + 0.8 * xs - 0.5 * xs**2 + 0.3 * xs**3, 0.4)
        return x, w, y

    def test_unpenalized_gaussian_predictions_match_monomial_span(self):
        # With the model intercept present and no selection penalty, the
        # polynomial block spans {x, x^2, x^3} modulo the constant, so
        # predictions are basis-independent.  Coefficients are NOT: they sit
        # in the data-orthogonal coordinates, not the Legendre ones.
        x, w, y = self._fixture()
        xs = x / 100.0

        model_poly = SuperGLM(
            family="gaussian",
            selection_penalty=None,
            features={"x": Polynomial(degree=3)},
        )
        model_poly.fit(pd.DataFrame({"x": x}), y, sample_weight=w)

        from superglm.features.numeric import Numeric

        model_raw = SuperGLM(
            family="gaussian",
            selection_penalty=None,
            features={"x1": Numeric(), "x2": Numeric(), "x3": Numeric()},
        )
        X_raw = pd.DataFrame({"x1": xs, "x2": xs**2, "x3": xs**3})
        model_raw.fit(X_raw, y, sample_weight=w)

        pred_poly = model_poly.predict(pd.DataFrame({"x": x}))
        pred_raw = model_raw.predict(X_raw)
        np.testing.assert_allclose(pred_poly, pred_raw, rtol=1e-8, atol=1e-8)

        # Coefficients moved to the data-orthogonal coordinates: they differ
        # from the WLS solution in the old Legendre basis.
        (g,) = model_poly._feature_groups("x")
        beta_poly = np.asarray(model_poly.result.beta[g.sl], dtype=np.float64)

        spec = model_poly._specs["x"]
        L = legvander(spec._scale(x), 3)[:, 1:]
        XL = np.column_stack([np.ones(len(x)), L])
        beta_leg = np.linalg.lstsq(np.sqrt(w)[:, None] * XL, np.sqrt(w) * y, rcond=None)[0]
        assert np.abs(beta_poly - beta_leg[1:]).max() > 1e-3


class TestPowerSubsets:
    """powers= selects orthogonal components built up to max(powers)."""

    def _fit_beta(self, powers, x, w, y):
        model = SuperGLM(
            family="gaussian",
            selection_penalty=None,
            features={"x": Polynomial(powers=powers)},
        )
        model.fit(pd.DataFrame({"x": x}), y, sample_weight=w)
        (g,) = model._feature_groups("x")
        beta = np.asarray(model.result.beta[g.sl], dtype=np.float64)
        return dict(zip(powers, beta, strict=True)), model

    def test_drop_top_power_leaves_lower_unchanged(self):
        x, w = _heaped_exposure(n=1000, seed=5)
        rng = np.random.default_rng(11)
        xs = (x - 50.0) / 50.0
        y = rng.normal(0.5 + 0.7 * xs - 0.4 * xs**2 + 0.2 * xs**4, 0.3)

        full, _ = self._fit_beta([1, 2, 3, 4], x, w, y)
        dropped, _ = self._fit_beta([1, 2, 3], x, w, y)
        for p in (1, 2, 3):
            np.testing.assert_allclose(dropped[p], full[p], rtol=1e-6, atol=1e-8)

    def test_drop_middle_power_leaves_retained_unchanged(self):
        x, w = _heaped_exposure(n=1000, seed=5)
        rng = np.random.default_rng(11)
        xs = (x - 50.0) / 50.0
        y = rng.normal(0.5 + 0.7 * xs - 0.4 * xs**2 + 0.2 * xs**4, 0.3)

        full, _ = self._fit_beta([1, 2, 3, 4], x, w, y)
        subset, _ = self._fit_beta([1, 2, 4], x, w, y)
        for p in (1, 2, 4):
            np.testing.assert_allclose(subset[p], full[p], rtol=1e-6, atol=1e-8)

    def test_group_size_equals_number_of_powers(self):
        x, w = _heaped_exposure(n=400, seed=9)
        info = PolynomialDirect(powers=[2, 5]).build(x, sample_weight=w)
        assert info.n_cols == 2
        assert info.columns.shape == (400, 2)

    def test_summary_rows_labelled_by_stated_power(self):
        rng = np.random.default_rng(123)
        n = 400
        age = rng.uniform(18, 90, n)
        sample_weight = rng.uniform(0.3, 1.0, n)
        age_s = (age - 50.0) / 20.0
        mu = np.exp(-1.8 + 0.35 * age_s - 0.25 * age_s**2 + 0.02 * age_s**4)
        y = rng.poisson(mu * sample_weight).astype(float)

        model = SuperGLM(
            family="poisson",
            selection_penalty=0.001,
            features={"age": Polynomial(powers=[1, 2, 4])},
        )
        model.fit(pd.DataFrame({"age": age}), y, sample_weight=sample_weight)

        text = str(model.summary())
        assert "age P(1,2,4)" in text
        assert "age[P1]" in text
        assert "age[P2]" in text
        assert "age[P4]" in text
        assert "age[P3]" not in text

    def test_reconstruct_carries_powers(self):
        x, w = _heaped_exposure(n=400, seed=9)
        spec = PolynomialDirect(powers=[1, 2, 4])
        spec.build(x, sample_weight=w)
        rec = spec.reconstruct(np.array([0.1, -0.05, 0.01]), n_points=50)
        assert rec["powers"] == (1, 2, 4)
        assert rec["degree"] == 4
        assert rec["x"].shape == (50,)

    def test_repr_shows_powers_when_not_a_ladder(self):
        assert repr(PolynomialDirect(powers=[4, 1, 2])) == "Polynomial(powers=[1, 2, 4])"
        assert repr(PolynomialDirect(degree=3)) == "Polynomial(degree=3)"
        assert repr(PolynomialDirect(powers=[1, 2, 3])) == "Polynomial(degree=3)"


class TestPolynomialGuards:
    def test_too_few_distinct_x_raises(self):
        x = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="distinct x values with positive weight"):
            PolynomialDirect(degree=3).build(x)

    def test_zero_weight_points_do_not_count_as_support(self):
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        w = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="distinct x values with positive weight"):
            PolynomialDirect(degree=3).build(x, sample_weight=w)

    def test_rank_guard_refuses_negligible_pivot(self):
        # Passes the distinct-support count (weights are positive) but the
        # effective support is two points: the pivot check must refuse
        # rather than silently regularize.
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        w = np.array([1.0, 1.0, 1e-300, 1e-300, 1e-300, 1e-300])
        with pytest.raises(ValueError, match="rank-deficient"):
            PolynomialDirect(degree=4).build(x, sample_weight=w)

    def test_degree_and_powers_mutually_exclusive(self):
        with pytest.raises(ValueError, match="not both"):
            PolynomialDirect(degree=3, powers=[1, 2])

    def test_powers_validation(self):
        with pytest.raises(ValueError, match=">= 1"):
            PolynomialDirect(powers=[0, 1])
        with pytest.raises(ValueError, match="distinct"):
            PolynomialDirect(powers=[1, 2, 2])
        with pytest.raises(ValueError, match="integers"):
            PolynomialDirect(powers=[1, 2.5])
        with pytest.raises(ValueError, match="integers"):
            PolynomialDirect(powers=[1, True])
        with pytest.raises(ValueError, match="at least one"):
            PolynomialDirect(powers=[])

    def test_negative_weight_raises(self):
        x = np.linspace(0, 10, 50)
        w = np.ones(50)
        w[3] = -0.5
        with pytest.raises(ValueError, match="nonnegative"):
            PolynomialDirect(degree=2).build(x, sample_weight=w)

    def test_transform_before_build_raises(self):
        with pytest.raises(ValueError, match="not fitted"):
            PolynomialDirect(degree=2).transform(np.array([1.0, 2.0]))

    def test_model_level_error_names_the_feature(self):
        y = np.ones(50)
        X = pd.DataFrame({"flat": np.full(50, 7.0)})
        model = SuperGLM(
            family="gaussian",
            selection_penalty=None,
            features={"flat": Polynomial(degree=2)},
        )
        with pytest.raises(ValueError, match="Feature 'flat'"):
            model.fit(X, y)


class TestWeightPolicyAndRestore:
    """Weight-family policy, support-scoped scaling, and pickle migration."""

    def test_tweedie_basis_orthonormal_under_prior_weights(self):
        # Deliberate policy pin: unlike spline knot geometry (physical rows
        # only under Tweedie), the Polynomial standardization follows
        # sample_weight under EVERY family, Tweedie EDM prior weights
        # included — orthonormalization is inference/selection geometry,
        # and the spanned column space is weight-invariant.
        from superglm import Tweedie

        rng = np.random.default_rng(21)
        n = 400
        x = rng.uniform(0.0, 100.0, n)
        w = np.exp(-x / 25.0) + 0.2 * rng.uniform(0.5, 1.0, n)
        mu = np.exp(0.3 + 0.004 * x)
        y = np.where(rng.uniform(size=n) < 0.75, rng.gamma(2.0, mu / 2.0), 0.0)

        model = SuperGLM(
            family=Tweedie(p=1.5),
            selection_penalty=None,
            features={"x": Polynomial(degree=3)},
        )
        model.fit(pd.DataFrame({"x": x}), y, sample_weight=w)

        spec = model._specs["x"]
        Phi = spec.transform(x)
        G = (Phi * w[:, None]).T @ Phi / float(w.sum())
        np.testing.assert_allclose(G, np.eye(3), atol=1e-10)

        # And it is genuinely the prior-weight basis, not the unit-weight one.
        G_unit = Phi.T @ Phi / n
        assert np.abs(G_unit - np.eye(3)).max() > 0.05
        unit_cols = PolynomialDirect(degree=3).build(x).columns
        assert np.abs(Phi - unit_cols).max() > 1e-3

    def test_scalar_ordered_polynomial_keeps_prior_weight_geometry(self):
        """The shared scalar compiler preserves its historical QR stream."""
        levels = [f"L{index}" for index in range(6)]
        labels = np.repeat(levels, 12)
        positions = np.repeat(np.arange(len(levels), dtype=np.float64), 12)
        weights = np.geomspace(0.1, 4.0, labels.size)
        response = 0.3 + positions - 0.15 * positions**2

        model = SuperGLM(
            family="gaussian",
            selection_penalty=None,
            features={
                "band": OrderedCategorical(
                    order=levels,
                    basis=Polynomial(degree=2),
                )
            },
        )
        model.fit(pd.DataFrame({"band": labels}), response, sample_weight=weights)

        inner = model._specs["band"]._basis_spline
        columns = inner.transform(positions)
        weighted_moments = (columns * weights[:, None]).sum(axis=0) / weights.sum()
        tolerance = 64.0 * np.finfo(np.float64).eps * max(columns.shape)
        np.testing.assert_allclose(weighted_moments, 0.0, atol=tolerance, rtol=0.0)

        # Mutation catcher: routing physical-row geometry through the shared
        # scalar path instead gives unit-row moments and misses this condition.
        unit_moments = columns.mean(axis=0)
        assert np.linalg.norm(unit_moments, ord=np.inf) > 0.1

    def test_zero_weight_outlier_does_not_stretch_scaling(self):
        x, w = _heaped_exposure(n=600, seed=17)
        x_out = np.append(x, 1e6)
        w_out = np.append(w, 0.0)

        ref = PolynomialDirect(degree=4)
        ref.build(x, sample_weight=w)

        spec = PolynomialDirect(degree=4)
        info = spec.build(x_out, sample_weight=w_out)

        # Scale bounds ignore the zero-weight outlier ...
        assert spec._lo == ref._lo
        assert spec._hi == ref._hi
        # ... so conditioning and the stored factor are unaffected ...
        np.testing.assert_allclose(spec._R, ref._R, rtol=1e-10, atol=1e-12)
        Phi = info.columns
        total = float(w_out.sum())
        G = (Phi * w_out[:, None]).T @ Phi / total
        np.testing.assert_allclose(G, np.eye(4), atol=1e-10)
        # ... and the outlier's own x still evaluates (plain polynomial
        # extrapolation on the scaled seed).
        assert np.all(np.isfinite(spec.transform(np.array([1e6]))))

    def test_sample_weight_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length"):
            PolynomialDirect(degree=2).build(np.linspace(0, 10, 50), sample_weight=np.ones(10))

    def test_powers_non_sequence_raises(self):
        with pytest.raises(ValueError, match="sequence"):
            PolynomialDirect(powers=3)

    def test_pre_022_state_restores_and_refuses_with_migration_message(self):
        # Pre-0.22 pickles restore __dict__ without __init__: no _R, no
        # powers.  __setstate__ defaults them and transform refuses with
        # the migration message instead of an AttributeError.
        old_state = {"degree": 3, "_lo": 0.0, "_hi": 100.0}
        spec = PolynomialDirect.__new__(PolynomialDirect)
        spec.__setstate__(old_state)
        assert spec.powers == (1, 2, 3)
        assert spec.degree == 3
        with pytest.raises(ValueError, match="refit"):
            spec.transform(np.array([1.0, 2.0]))

    def test_fitted_spec_pickle_round_trip(self):
        import pickle

        x, w = _heaped_exposure(n=400, seed=23)
        spec = PolynomialDirect(powers=[1, 2, 4])
        spec.build(x, sample_weight=w)
        restored = pickle.loads(pickle.dumps(spec))
        assert restored.powers == (1, 2, 4)
        np.testing.assert_allclose(restored.transform(x), spec.transform(x), atol=1e-12)
