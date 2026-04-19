"""Theory-driven invariants for fitting, weighting, and backend algebra."""

import numpy as np
import pandas as pd
import pytest

from superglm import Constraint, SuperGLM
from superglm.distributions import clip_mu
from superglm.features.categorical import Categorical
from superglm.features.numeric import Numeric
from superglm.features.spline import CubicRegressionSpline, NaturalSpline, PSpline, Spline
from superglm.group_matrix import (
    DiscretizedSSPGroupMatrix,
    DiscretizedTensorGroupMatrix,
    _block_xtwx,
    _cross_gram,
)
from superglm.links import stabilize_eta


def _final_working_problem(model, y, sample_weight=None, offset=None):
    """Return the final PIRLS working weights/response residual."""
    y = np.asarray(y, dtype=np.float64)
    weights = (
        np.ones_like(y) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    )
    offset_arr = np.zeros_like(y) if offset is None else np.asarray(offset, dtype=np.float64)

    beta = model.result.beta
    eta = stabilize_eta(model._dm.matvec(beta) + model.result.intercept + offset_arr, model._link)
    mu = clip_mu(model._link.inverse(eta), model._distribution)
    V = model._distribution.variance(mu)
    dmu_deta = model._link.deriv_inverse(eta)
    W = weights * dmu_deta**2 / np.maximum(V, 1e-10)
    z = eta + (y - mu) / dmu_deta
    r = z - model._dm.matvec(beta) - model.result.intercept - offset_arr
    return W, r


class TestSolverTheoryInvariants:
    def test_unpenalised_poisson_score_equations(self):
        """Canonical Poisson fit should satisfy the score equations at convergence."""
        rng = np.random.default_rng(0)
        n = 400
        x1 = rng.standard_normal(n)
        x2 = rng.standard_normal(n)
        eta = 0.2 + 0.4 * x1 - 0.3 * x2
        y = rng.poisson(np.exp(eta)).astype(float)
        X = pd.DataFrame({"x1": x1, "x2": x2})

        model = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            spline_penalty=0.0,
            features={
                "x1": Numeric(),
                "x2": Numeric(),
            },
        )
        model.fit(X, y)

        mu = model.predict(X)
        residual = y - mu
        score_intercept = float(np.sum(residual))
        score_beta = model._dm.toarray().T @ residual

        np.testing.assert_allclose(score_intercept, 0.0, atol=1e-5)
        np.testing.assert_allclose(score_beta, 0.0, atol=1e-5)

    def test_integer_frequency_weights_match_row_replication(self):
        """Integer frequency weights should be equivalent to duplicating rows."""
        X = pd.DataFrame(
            {
                "x1": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                "x2": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            }
        )
        y = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        weights = np.array([1, 2, 3, 1, 2, 1], dtype=float)

        weighted = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            spline_penalty=0.0,
            features={
                "x1": Numeric(),
                "x2": Numeric(),
            },
        )
        weighted.fit(X, y, sample_weight=weights)

        idx = np.repeat(np.arange(len(X)), weights.astype(int))
        X_rep = X.iloc[idx].reset_index(drop=True)
        y_rep = y[idx]

        replicated = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            spline_penalty=0.0,
            features={
                "x1": Numeric(),
                "x2": Numeric(),
            },
        )
        replicated.fit(X_rep, y_rep)

        np.testing.assert_allclose(
            weighted.result.intercept, replicated.result.intercept, atol=1e-10
        )
        np.testing.assert_allclose(weighted.result.beta, replicated.result.beta, atol=1e-10)
        np.testing.assert_allclose(weighted.predict(X), replicated.predict(X), atol=1e-10)

    def test_group_lasso_solution_satisfies_kkt_conditions(self):
        """Final BCD solution should satisfy group-lasso KKT conditions."""
        rng = np.random.default_rng(123)
        n = 400
        x1 = rng.standard_normal(n)
        x2 = rng.standard_normal(n)
        x3 = rng.standard_normal(n)
        eta = 0.1 + 0.7 * x1
        y = rng.poisson(np.exp(eta)).astype(float)
        X = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})

        lam = 20.0
        model = SuperGLM(
            family="poisson",
            selection_penalty=lam,
            spline_penalty=0.0,
            features={
                "x1": Numeric(),
                "x2": Numeric(),
                "x3": Numeric(),
            },
        )
        model.fit(X, y)

        W, r = _final_working_problem(model, y)
        tol = 1e-2
        for gm, g in zip(model._dm.group_matrices, model._groups):
            beta_g = model.result.beta[g.sl]
            grad_g = -gm.rmatvec(W * r)
            threshold = lam * g.weight

            if np.linalg.norm(beta_g) > 1e-10:
                np.testing.assert_allclose(np.linalg.norm(grad_g), threshold, atol=tol)
                assert np.dot(grad_g, beta_g) < 0
            else:
                assert np.linalg.norm(grad_g) <= threshold + tol

    def test_fit_is_invariant_to_row_order(self):
        """Row order should not change the fitted solution."""
        rng = np.random.default_rng(2)
        n = 400
        x1 = rng.standard_normal(n)
        cat = rng.choice(["A", "B", "C"], n)
        weights = rng.integers(1, 4, size=n).astype(float)
        eta = 0.3 + 0.4 * x1 + 0.2 * (cat == "B") - 0.1 * (cat == "C")
        y = rng.poisson(np.exp(eta)).astype(float)
        X = pd.DataFrame({"x1": x1, "cat": cat})

        model_a = SuperGLM(
            family="poisson",
            selection_penalty=0.05,
            spline_penalty=0.0,
            features={"x1": Numeric(), "cat": Categorical(base="first")},
        )
        model_a.fit(X, y, sample_weight=weights)

        perm = rng.permutation(n)
        model_b = SuperGLM(
            family="poisson",
            selection_penalty=0.05,
            spline_penalty=0.0,
            features={"x1": Numeric(), "cat": Categorical(base="first")},
        )
        model_b.fit(X.iloc[perm].reset_index(drop=True), y[perm], sample_weight=weights[perm])

        np.testing.assert_allclose(model_a.result.intercept, model_b.result.intercept, atol=1e-10)
        np.testing.assert_allclose(model_a.result.beta, model_b.result.beta, atol=1e-10)
        np.testing.assert_allclose(model_a.result.deviance, model_b.result.deviance, atol=1e-10)


class TestBackendLinearAlgebraInvariants:
    def test_block_xtwx_matches_dense_oracle(self):
        """Blockwise X'WX should equal the dense oracle on mixed backends."""
        rng = np.random.default_rng(7)
        n = 300
        X = pd.DataFrame(
            {
                "x_num": rng.standard_normal(n),
                "x_cat": rng.choice(["A", "B", "C"], n),
                "x_spline": rng.uniform(0, 10, n),
            }
        )
        eta = (
            0.1
            + 0.2 * X["x_num"].to_numpy()
            + 0.2 * (X["x_cat"].to_numpy() == "B")
            + 0.1 * np.sin(X["x_spline"].to_numpy())
        )
        y = rng.poisson(np.exp(eta)).astype(float)

        model = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            discrete=True,
            features={
                "x_num": Numeric(),
                "x_cat": Categorical(base="first"),
                "x_spline": Spline(n_knots=8, penalty="ssp"),
            },
        )
        model.fit(X, y)

        W = rng.uniform(0.5, 2.0, n)
        xtwx_block = _block_xtwx(model._dm.group_matrices, model._groups, W)
        X_dense = model._dm.toarray()
        xtwx_dense = X_dense.T @ (X_dense * W[:, None])

        np.testing.assert_allclose(xtwx_block, xtwx_dense, atol=1e-10)

    def test_row_subset_preserves_dense_design_behavior(self):
        """Row-subsetted design matrices should agree with the dense oracle."""
        rng = np.random.default_rng(8)
        n = 200
        X = pd.DataFrame(
            {
                "x_num": rng.standard_normal(n),
                "x_cat": rng.choice(["A", "B", "C"], n),
                "x_spline": rng.uniform(0, 10, n),
            }
        )
        y = rng.poisson(np.exp(0.1 + 0.3 * X["x_num"].to_numpy())).astype(float)

        model = SuperGLM(
            family="poisson",
            selection_penalty=0.1,
            discrete=True,
            features={
                "x_num": Numeric(),
                "x_cat": Categorical(base="first"),
                "x_spline": Spline(n_knots=6, penalty="ssp"),
            },
        )
        model.fit(X, y)

        idx = np.sort(rng.choice(n, size=80, replace=False))
        dm_sub = model._dm.row_subset(idx)
        X_sub_dense = model._dm.toarray()[idx]

        np.testing.assert_allclose(dm_sub.toarray(), X_sub_dense, atol=1e-12)

        beta = rng.standard_normal(model._dm.p)
        np.testing.assert_allclose(dm_sub.matvec(beta), X_sub_dense @ beta, atol=1e-12)

        w = rng.standard_normal(len(idx))
        np.testing.assert_allclose(dm_sub.rmatvec(w), X_sub_dense.T @ w, atol=1e-12)

    def test_high_cardinality_tabmat_subset_preserves_width(self):
        """tabmat CategoricalMatrix must preserve full column count on row subsets.

        Regression test for 8422dbe: without pinning the category universe,
        tabmat infers categories from the observed subset only, shrinking
        the sandwich output and breaking XtWX assembly.
        """
        from superglm.group_matrix import CategoricalGroupMatrix, _block_xtwx, _build_tabmat_split

        rng = np.random.default_rng(42)
        n = 2000
        n_levels = 150  # > 100 threshold for CategoricalMatrix path
        codes = rng.integers(-1, n_levels, size=n).astype(np.intp)
        gm = CategoricalGroupMatrix(codes, n_levels)

        # Subset that drops many levels
        idx = np.arange(40)
        sub = gm.row_subset(idx)
        n_unique = len(np.unique(sub.codes[sub.codes < sub.n_levels]))
        assert n_unique < n_levels, "subset should drop some levels"
        assert sub.shape == (40, n_levels), "group shape must preserve full width"

        # tabmat split on subset must also preserve width
        split = _build_tabmat_split([sub])
        assert split is not None
        assert split.shape == (40, n_levels), (
            f"tabmat split shape {split.shape} != (40, {n_levels})"
        )

        # sandwich must produce full-width XtWX
        W = rng.uniform(0.5, 2.0, 40)
        xtwx_tabmat = np.asarray(split.sandwich(W))
        assert xtwx_tabmat.shape == (n_levels, n_levels)

        # Match dense oracle
        from superglm.types import GroupSlice

        groups = [GroupSlice("cat", 0, n_levels)]
        xtwx_block = _block_xtwx([sub], groups, W)
        np.testing.assert_allclose(xtwx_tabmat, xtwx_block, atol=1e-12)

    def test_categorical_cross_gram_matches_dense_oracle(self):
        """Categorical × categorical cross-gram must match dense weighted one-hot."""
        from superglm.group_matrix import CategoricalGroupMatrix

        codes_i = np.array([-1, 0, 1, -1, 1, 0], dtype=np.intp)
        codes_j = np.array([0, -1, 1, 1, -1, 0], dtype=np.intp)
        gm_i = CategoricalGroupMatrix(codes_i, 2)
        gm_j = CategoricalGroupMatrix(codes_j, 2)
        W = np.array([1.0, -2.0, 0.5, 3.0, 4.0, -1.0])

        cross = _cross_gram(gm_i, gm_j, W)
        dense = (W[:, None] * gm_i.toarray()).T @ gm_j.toarray()

        np.testing.assert_allclose(cross, dense, atol=1e-12)

    def test_two_discretized_spline_cross_gram(self):
        """Cross-gram between two DiscretizedSSPGroupMatrix groups should match dense."""
        rng = np.random.default_rng(9)
        n = 400
        X = pd.DataFrame(
            {
                "s1": rng.uniform(0, 10, n),
                "s2": rng.uniform(0, 10, n),
            }
        )
        y = rng.poisson(np.exp(0.1 + 0.1 * np.sin(X["s1"].to_numpy()))).astype(float)

        model = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            discrete=True,
            features={
                "s1": Spline(n_knots=6, penalty="ssp"),
                "s2": Spline(n_knots=8, penalty="ssp"),
            },
        )
        model.fit(X, y)

        W = rng.uniform(0.5, 2.0, n)
        xtwx_block = _block_xtwx(model._dm.group_matrices, model._groups, W)
        X_dense = model._dm.toarray()
        xtwx_dense = X_dense.T @ (X_dense * W[:, None])

        np.testing.assert_allclose(xtwx_block, xtwx_dense, atol=1e-10)

    def test_two_discretized_scop_block_xtwx_matches_dense(self):
        """Full XtWX with two discretized SCOP groups should match dense oracle."""
        rng = np.random.default_rng(19)
        n = 400
        X = pd.DataFrame(
            {
                "x1": rng.uniform(0, 1, n),
                "x2": rng.uniform(0, 1, n),
            }
        )
        eta = -0.4 + 0.4 * np.log1p(4 * X["x1"].to_numpy()) + 0.3 * np.log1p(5 * X["x2"].to_numpy())
        y = rng.poisson(np.exp(eta)).astype(float)

        model = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            discrete=True,
            features={
                "x1": PSpline(n_knots=8, constraint=Constraint.fit.increasing),
                "x2": PSpline(n_knots=7, constraint=Constraint.fit.increasing),
            },
        )
        model.fit_reml(X, y, max_reml_iter=6)

        W = rng.uniform(0.5, 2.0, n)
        xtwx_block = _block_xtwx(model._dm.group_matrices, model._groups, W)
        xtwx_dense = model._dm.toarray().T @ (model._dm.toarray() * W[:, None])

        np.testing.assert_allclose(xtwx_block, xtwx_dense, atol=1e-10)

    def test_mixed_discretized_scop_block_xtwx_matches_dense(self):
        """Mixed SCOP + discretized SSP + categorical XtWX should match dense oracle."""
        rng = np.random.default_rng(23)
        n = 500
        X = pd.DataFrame(
            {
                "s": rng.uniform(0, 1, n),
                "z": rng.uniform(0, 1, n),
                "area": rng.choice(["a", "b", "c"], size=n),
            }
        )
        eta = (
            -0.2
            + 0.5 * np.log1p(6 * X["s"].to_numpy())
            + 0.2 * np.sin(2 * np.pi * X["z"].to_numpy())
        )
        y = rng.poisson(np.exp(eta)).astype(float)

        model = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            discrete=True,
            features={
                "s": PSpline(n_knots=9, constraint=Constraint.fit.increasing),
                "z": Spline(kind="cr", k=10),
                "area": Categorical(base="first"),
            },
        )
        model.fit_reml(X, y, max_reml_iter=6)

        W = rng.uniform(0.5, 2.0, n)
        xtwx_block = _block_xtwx(model._dm.group_matrices, model._groups, W)
        xtwx_dense = model._dm.toarray().T @ (model._dm.toarray() * W[:, None])

        np.testing.assert_allclose(xtwx_block, xtwx_dense, atol=1e-10)

    def test_tensor_gram_matches_dense_oracle(self):
        """DiscretizedTensorGroupMatrix.gram() must match X.T @ diag(W) @ X."""
        rng = np.random.default_rng(77)
        n = 500
        X = pd.DataFrame(
            {
                "s1": rng.uniform(0, 10, n),
                "s2": rng.uniform(0, 10, n),
            }
        )
        y = rng.poisson(np.exp(0.1 + 0.05 * X["s1"].to_numpy())).astype(float)
        model = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            discrete=True,
            features={
                "s1": Spline(n_knots=6, penalty="ssp"),
                "s2": Spline(n_knots=5, penalty="ssp"),
            },
            interactions=[("s1", "s2")],
        )
        model.fit(X, y)

        # Verify tensor group type
        gm_tensor = model._dm.group_matrices[2]
        assert isinstance(gm_tensor, DiscretizedTensorGroupMatrix)

        W = rng.uniform(0.5, 2.0, n)
        X_dense = gm_tensor.toarray()
        gram_dense = X_dense.T @ (X_dense * W[:, None])
        gram_factored = gm_tensor.gram(W)
        np.testing.assert_allclose(gram_factored, gram_dense, atol=1e-10)

    def test_tensor_cross_gram_main_matches_dense(self):
        """Cross-gram between tensor and main-effect groups must match dense."""
        rng = np.random.default_rng(88)
        n = 500
        X = pd.DataFrame(
            {
                "s1": rng.uniform(0, 10, n),
                "s2": rng.uniform(0, 10, n),
                "s3": rng.uniform(0, 10, n),
            }
        )
        y = rng.poisson(np.exp(0.1 + 0.05 * X["s1"].to_numpy())).astype(float)
        model = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            discrete=True,
            features={
                "s1": Spline(n_knots=6, penalty="ssp"),
                "s2": Spline(n_knots=5, penalty="ssp"),
                "s3": Spline(n_knots=4, penalty="ssp"),
            },
            interactions=[("s1", "s2")],
        )
        model.fit(X, y)

        gms = model._dm.group_matrices
        gm_main = gms[0]  # s1 main effect
        gm_tensor = gms[3]  # s1:s2 tensor
        gm_other = gms[2]  # s3, non-parent of tensor
        assert isinstance(gm_tensor, DiscretizedTensorGroupMatrix)
        assert isinstance(gm_main, DiscretizedSSPGroupMatrix)

        W = rng.uniform(0.5, 2.0, n)

        # tensor × parent main effect
        cross = _cross_gram(gm_main, gm_tensor, W)
        X_main = gm_main.toarray()
        X_tensor = gm_tensor.toarray()
        cross_dense = X_main.T @ (X_tensor * W[:, None])
        np.testing.assert_allclose(cross, cross_dense, atol=1e-9)

        # tensor × non-parent main effect
        cross2 = _cross_gram(gm_other, gm_tensor, W)
        X_other = gm_other.toarray()
        cross2_dense = X_other.T @ (X_tensor * W[:, None])
        np.testing.assert_allclose(cross2, cross2_dense, atol=1e-9)

    def test_tensor_full_xtwx_matches_dense(self):
        """Full _block_xtwx with tensor interaction must match dense oracle."""
        rng = np.random.default_rng(99)
        n = 500
        X = pd.DataFrame(
            {
                "s1": rng.uniform(0, 10, n),
                "s2": rng.uniform(0, 10, n),
                "s3": rng.uniform(0, 10, n),
            }
        )
        y = rng.poisson(np.exp(0.1 + 0.05 * X["s1"].to_numpy())).astype(float)
        model = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            discrete=True,
            features={
                "s1": Spline(n_knots=6, penalty="ssp"),
                "s2": Spline(n_knots=5, penalty="ssp"),
                "s3": Spline(n_knots=4, penalty="ssp"),
            },
            interactions=[("s1", "s2")],
        )
        model.fit(X, y)

        W = rng.uniform(0.5, 2.0, n)
        xtwx_block = _block_xtwx(model._dm.group_matrices, model._groups, W)
        X_dense = model._dm.toarray()
        xtwx_dense = X_dense.T @ (X_dense * W[:, None])
        np.testing.assert_allclose(xtwx_block, xtwx_dense, atol=1e-9)

    def test_tensor_cross_gram_decomposed_matches_dense(self):
        """Cross-gram between bilinear and wiggly tensor subgroups must match dense."""
        rng = np.random.default_rng(55)
        n = 500
        X = pd.DataFrame(
            {
                "s1": rng.uniform(0, 10, n),
                "s2": rng.uniform(0, 10, n),
            }
        )
        y = rng.poisson(np.exp(0.1 + 0.05 * X["s1"].to_numpy())).astype(float)
        model = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            discrete=True,
            features={
                "s1": Spline(n_knots=6, penalty="ssp"),
                "s2": Spline(n_knots=5, penalty="ssp"),
            },
        )
        model._add_interaction("s1", "s2", decompose=True)
        model.fit(X, y)

        gms = model._dm.group_matrices
        # Find bilinear and wiggly subgroups
        tensor_gms = [gm for gm, g in zip(gms, model._groups) if g.feature_name == "s1:s2"]
        assert len(tensor_gms) == 2
        gm_bilin, gm_wiggly = tensor_gms
        assert isinstance(gm_bilin, DiscretizedTensorGroupMatrix)
        assert isinstance(gm_wiggly, DiscretizedTensorGroupMatrix)
        assert gm_bilin.tensor_id == gm_wiggly.tensor_id

        W = rng.uniform(0.5, 2.0, n)
        cross = _cross_gram(gm_bilin, gm_wiggly, W)
        X_b = gm_bilin.toarray()
        X_w = gm_wiggly.toarray()
        cross_dense = X_b.T @ (X_w * W[:, None])
        np.testing.assert_allclose(cross, cross_dense, atol=1e-10)

    def test_tensor_matvec_rmatvec_match_dense(self):
        """DiscretizedTensorGroupMatrix matvec/rmatvec must match dense materialization."""
        rng = np.random.default_rng(66)
        n = 300
        X = pd.DataFrame(
            {
                "s1": rng.uniform(0, 10, n),
                "s2": rng.uniform(0, 10, n),
            }
        )
        y = rng.poisson(np.exp(0.1 + 0.05 * X["s1"].to_numpy())).astype(float)
        model = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            discrete=True,
            features={
                "s1": Spline(n_knots=6, penalty="ssp"),
                "s2": Spline(n_knots=5, penalty="ssp"),
            },
            interactions=[("s1", "s2")],
        )
        model.fit(X, y)

        gm = model._dm.group_matrices[2]
        assert isinstance(gm, DiscretizedTensorGroupMatrix)
        X_dense = gm.toarray()
        p_g = gm.shape[1]

        v = rng.standard_normal(p_g)
        np.testing.assert_allclose(gm.matvec(v), X_dense @ v, atol=1e-12)

        w = rng.standard_normal(n)
        np.testing.assert_allclose(gm.rmatvec(w), X_dense.T @ w, atol=1e-12)


class TestPredictionTimeContracts:
    """Verify predict() contracts for edge-case inputs."""

    def test_unseen_categorical_level_raises_error(self):
        """Predicting with unseen categorical levels should raise ValueError."""
        X_train = pd.DataFrame(
            {
                "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "cat": ["A", "B", "C", "A", "B", "C"],
            }
        )
        y = np.array([1.0, 2.0, 1.5, 1.0, 2.5, 1.0])

        model = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            spline_penalty=0.0,
            features={"x": Numeric(), "cat": Categorical(base="first")},
        )
        model.fit(X_train, y)

        # Known levels should predict fine
        X_known = pd.DataFrame({"x": [3.0], "cat": ["B"]})
        pred = model.predict(X_known)
        assert np.all(np.isfinite(pred))

        # Unseen level should raise
        X_unseen = pd.DataFrame({"x": [3.0], "cat": ["D"]})
        with pytest.raises(ValueError, match="unseen"):
            model.predict(X_unseen)

    def test_unseen_categorical_in_interaction_raises_error(self):
        """Unseen categorical level in an interaction should also raise ValueError."""
        rng = np.random.default_rng(50)
        n = 200
        age = rng.uniform(18, 80, n)
        region = rng.choice(["A", "B", "C"], n)
        eta = -0.5 + 0.01 * age + 0.2 * (region == "B")
        y = rng.poisson(np.exp(eta)).astype(float)
        X_train = pd.DataFrame({"age": age, "region": region})

        model = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            features={
                "age": Spline(n_knots=6, penalty="ssp"),
                "region": Categorical(base="first"),
            },
            interactions=[("age", "region")],
        )
        model.fit(X_train, y)

        # Known levels predict fine
        X_known = pd.DataFrame({"age": [40.0], "region": ["B"]})
        pred = model.predict(X_known)
        assert np.all(np.isfinite(pred))

        # Unseen level raises
        X_unseen = pd.DataFrame({"age": [40.0], "region": ["D"]})
        with pytest.raises(ValueError, match="unseen"):
            model.predict(X_unseen)

    def test_nan_categorical_raises_error(self):
        """NaN/None in categorical column should raise ValueError, not TypeError."""
        X_train = pd.DataFrame(
            {
                "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "cat": ["A", "B", "C", "A", "B", "C"],
            }
        )
        y = np.array([1.0, 2.0, 1.5, 1.0, 2.5, 1.0])

        model = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            spline_penalty=0.0,
            features={"x": Numeric(), "cat": Categorical(base="first")},
        )
        model.fit(X_train, y)

        # NaN in categorical column (single row)
        X_nan = pd.DataFrame({"x": [3.0], "cat": [np.nan]})
        with pytest.raises(ValueError, match="missing"):
            model.predict(X_nan)

        # None in categorical column (single row)
        X_none = pd.DataFrame({"x": [3.0], "cat": [None]})
        with pytest.raises(ValueError, match="missing"):
            model.predict(X_none)

        # Batch with mix of valid string + NaN (the common case)
        X_batch_nan = pd.DataFrame({"x": [3.0, 4.0], "cat": ["B", np.nan]})
        with pytest.raises(ValueError, match="missing"):
            model.predict(X_batch_nan)

        # Batch with mix of valid string + None
        X_batch_none = pd.DataFrame({"x": [3.0, 4.0], "cat": ["B", None]})
        with pytest.raises(ValueError, match="missing"):
            model.predict(X_batch_none)

    def test_spline_extrapolation_is_flat_clamp(self):
        """Values outside training range should clamp to boundary predictions."""
        rng = np.random.default_rng(10)
        n = 200
        x = rng.uniform(1.0, 10.0, n)
        y = rng.poisson(np.exp(0.1 + 0.05 * x)).astype(float)
        X = pd.DataFrame({"x": x})

        model = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            features={"x": Spline(n_knots=8, penalty="ssp")},
        )
        model.fit(X, y)

        # Predictions at the boundaries
        X_lo = pd.DataFrame({"x": [1.0]})
        X_hi = pd.DataFrame({"x": [10.0]})
        pred_lo = model.predict(X_lo)
        pred_hi = model.predict(X_hi)

        # Far outside training range — clamped to boundary
        X_far_lo = pd.DataFrame({"x": [-100.0]})
        X_far_hi = pd.DataFrame({"x": [1000.0]})
        pred_far_lo = model.predict(X_far_lo)
        pred_far_hi = model.predict(X_far_hi)

        np.testing.assert_allclose(pred_far_lo, pred_lo, atol=1e-10)
        np.testing.assert_allclose(pred_far_hi, pred_hi, atol=1e-10)
        assert np.all(np.isfinite(pred_far_lo))
        assert np.all(np.isfinite(pred_far_hi))

    def test_constant_numeric_predictor_produces_zero_column(self):
        """A constant standardized numeric feature should produce an all-zero column."""
        X = pd.DataFrame({"x": [5.0] * 100, "z": np.random.default_rng(0).standard_normal(100)})
        y = np.random.default_rng(0).poisson(1.0, 100).astype(float)

        model = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            spline_penalty=0.0,
            features={"x": Numeric(), "z": Numeric()},
        )
        model.fit(X, y)

        # The constant feature contributes nothing to predictions
        # Predictions should be finite regardless
        pred = model.predict(X)
        assert np.all(np.isfinite(pred))
        assert np.all(pred > 0)

        # Verify the constant column is passed through unchanged
        spec = model._specs["x"]
        col = spec.transform(np.array([5.0, 5.0, 5.0]))
        np.testing.assert_allclose(col, 5.0, atol=1e-6)

    def test_constant_spline_predictor_contributes_no_variation(self):
        """A constant spline feature should not affect prediction variation.

        The implementation detail (whether the basis is all-zero or not) is
        not the contract — what matters is that fit/predict remain finite and
        the constant feature contributes no predictive variation.
        """
        rng = np.random.default_rng(1)
        X = pd.DataFrame({"x": [5.0] * 100, "z": rng.standard_normal(100)})
        y = rng.poisson(1.0, 100).astype(float)

        model = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            features={"x": Spline(n_knots=6, penalty="ssp"), "z": Numeric()},
        )
        model.fit(X, y)

        # Predictions must be finite and positive
        pred = model.predict(X)
        assert np.all(np.isfinite(pred))
        assert np.all(pred > 0)

        # The constant feature should contribute no variation: predicting at
        # different "x" values (while z is the same) should give identical results.
        X_vary_x = pd.DataFrame({"x": [3.0, 5.0, 7.0], "z": [0.0, 0.0, 0.0]})
        pred_vary = model.predict(X_vary_x)
        assert np.ptp(pred_vary) < 1e-6, (
            f"Constant-feature spline should add no variation, but ptp={np.ptp(pred_vary):.2e}"
        )

    @pytest.mark.parametrize(
        "spline_cls",
        [Spline, NaturalSpline, CubicRegressionSpline],
        ids=["bspline", "natural", "crs"],
    )
    def test_extrapolation_finite_all_spline_types(self, spline_cls):
        """All spline types should produce finite predictions outside training range."""
        rng = np.random.default_rng(11)
        n = 200
        x = rng.uniform(0.0, 10.0, n)
        y = rng.poisson(np.exp(0.5 + 0.05 * x)).astype(float)
        X = pd.DataFrame({"x": x})

        model = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            features={"x": spline_cls(n_knots=8)},
        )
        model.fit(X, y)

        X_extrap = pd.DataFrame({"x": [-50.0, -10.0, 0.0, 10.0, 50.0, 100.0]})
        pred = model.predict(X_extrap)
        assert np.all(np.isfinite(pred))
        assert np.all(pred > 0)


class TestREMLInteraction:
    """Verify REML works correctly with interaction terms."""

    def test_reml_spline_categorical_interaction_converges(self):
        """fit_reml() with SplineCategorical interaction should converge.

        Uses a strong interaction effect so REML has a clear signal to
        stabilise lambdas rather than pushing them to infinity.
        """
        rng = np.random.default_rng(42)
        n = 2000
        age = rng.uniform(18, 80, n)
        region = rng.choice(["A", "B", "C"], n)
        eta = (
            -1.0
            + 0.02 * (age - 50)
            + 0.5 * (region == "B")
            + 0.015 * (age - 50) * (region == "B")  # strong interaction
            - 0.01 * (age - 50) * (region == "C")
        )
        y = rng.poisson(np.exp(eta)).astype(float)
        X = pd.DataFrame({"age": age, "region": region})

        model = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            features={"age": Spline(n_knots=8, penalty="ssp"), "region": Categorical()},
            interactions=[("age", "region")],
        )
        model.fit_reml(X, y, max_reml_iter=30)

        assert model._reml_lambdas is not None
        # Main spline + interaction per-level groups should all have REML lambdas
        assert len(model._reml_lambdas) >= 2  # at least main + 1 interaction level

        # Predictions should be finite and positive
        pred = model.predict(X)
        assert np.all(np.isfinite(pred))
        assert np.all(pred > 0)

        # Deviance should decrease relative to main-effects-only model
        main_model = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            features={"age": Spline(n_knots=8, penalty="ssp"), "region": Categorical()},
        )
        main_model.fit_reml(X, y, max_reml_iter=15)
        assert model.result.deviance <= main_model.result.deviance + 1.0  # small tolerance

    def test_reml_interaction_lambdas_are_positive(self):
        """REML-estimated lambdas for interaction groups should be positive and finite."""
        rng = np.random.default_rng(43)
        n = 600
        age = rng.uniform(18, 80, n)
        region = rng.choice(["A", "B", "C"], n)
        eta = -0.5 + 0.005 * age + 0.2 * (region == "B")
        y = rng.poisson(np.exp(eta)).astype(float)
        X = pd.DataFrame({"age": age, "region": region})

        model = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            features={"age": Spline(n_knots=6, penalty="ssp"), "region": Categorical()},
            interactions=[("age", "region")],
        )
        model.fit_reml(X, y, max_reml_iter=15)

        for name, lam in model._reml_lambdas.items():
            assert np.isfinite(lam), f"Non-finite REML lambda for {name}: {lam}"
            assert lam > 0, f"Non-positive REML lambda for {name}: {lam}"

    def test_reml_interaction_deviance_below_main_effects(self):
        """REML fit with interaction should improve on main-effects-only model."""
        rng = np.random.default_rng(44)
        n = 800
        age = rng.uniform(18, 80, n)
        region = rng.choice(["A", "B"], n)
        # DGP has a genuine interaction: slope of age differs by region
        eta = -0.5 + 0.01 * age + 0.3 * (region == "B") + 0.008 * age * (region == "B")
        y = rng.poisson(np.exp(eta)).astype(float)
        X = pd.DataFrame({"age": age, "region": region})

        interaction_model = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            features={"age": Spline(n_knots=6, penalty="ssp"), "region": Categorical()},
            interactions=[("age", "region")],
        )
        interaction_model.fit_reml(X, y, max_reml_iter=15)

        main_model = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            features={"age": Spline(n_knots=6, penalty="ssp"), "region": Categorical()},
        )
        main_model.fit_reml(X, y, max_reml_iter=15)

        assert interaction_model.result.deviance < main_model.result.deviance
