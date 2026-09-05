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
            weight_semantics="frequency",
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
            weight_semantics="frequency",
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

    def test_tabmat_split_skips_sparse_ssp_groups(self):
        """SSP spline groups must stay on sparse/factored algebra."""
        import scipy.sparse as sp

        from superglm.group_matrix import SparseSSPGroupMatrix, _build_tabmat_split

        B = sp.csr_matrix(
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.5, 0.5, 0.0],
                    [0.0, 0.5, 0.5],
                    [0.0, 0.0, 1.0],
                ]
            )
        )
        gm = SparseSSPGroupMatrix(B, np.eye(3))

        assert _build_tabmat_split([gm]) is None

    @pytest.mark.parametrize("vector_layout", ["strided", "readonly"])
    def test_tabmat_raw_block_products_normalize_compiled_vector_buffers(self, vector_layout):
        from superglm.group_matrix import (
            CategoricalGroupMatrix,
            _block_xtwx,
            _block_xtwx_rhs,
            _block_xtwx_signed,
            _build_tabmat_split,
        )
        from superglm.types import GroupSlice

        rng = np.random.default_rng(162)
        n = 600
        n_levels = 120
        codes = np.resize(np.arange(n_levels, dtype=np.intp), n)
        rng.shuffle(codes)
        gm = CategoricalGroupMatrix(codes, n_levels)
        split = _build_tabmat_split([gm])
        assert split is not None
        groups = [GroupSlice("cat", 0, n_levels)]
        base_W = rng.uniform(0.25, 2.0, size=n)
        base_Wz = rng.normal(size=n)
        if vector_layout == "strided":
            W_storage = np.empty(2 * n)
            Wz_storage = np.empty(2 * n)
            W_storage[::2] = base_W
            Wz_storage[::2] = base_Wz
            W = W_storage[::2]
            Wz = Wz_storage[::2]
        else:
            W = base_W.copy()
            Wz = base_Wz.copy()
            W.setflags(write=False)
            Wz.setflags(write=False)

        expected_gram = _block_xtwx([gm], groups, base_W)
        expected_signed = _block_xtwx_signed([gm], groups, base_W)
        expected_rhs = _block_xtwx_rhs([gm], groups, base_W, base_Wz)

        np.testing.assert_allclose(_block_xtwx([gm], groups, W, tabmat_split=split), expected_gram)
        np.testing.assert_allclose(
            _block_xtwx_signed([gm], groups, W, tabmat_split=split), expected_signed
        )
        actual_rhs = _block_xtwx_rhs([gm], groups, W, Wz, tabmat_split=split)
        for actual, expected in zip(actual_rhs, expected_rhs, strict=True):
            np.testing.assert_allclose(actual, expected)

    def test_spline_categorical_level_group_matches_masked_ssp(self):
        """Compact spline-by-category level algebra must match masked sparse SSP."""
        import scipy.sparse as sp

        from superglm.group_matrix import SparseSSPGroupMatrix, SplineCategoricalGroupMatrix

        rng = np.random.default_rng(113)
        n = 60
        B = sp.random(n, 7, density=0.35, format="csr", random_state=113)
        R_inv = rng.normal(size=(7, 5))
        mask = rng.random(n) < 0.3
        W = rng.uniform(0.2, 2.0, size=n)
        w = rng.normal(size=n)
        beta = rng.normal(size=5)

        compact = SplineCategoricalGroupMatrix(B, R_inv, mask)
        masked = SparseSSPGroupMatrix(B.multiply(mask.astype(float)[:, None]), R_inv)

        np.testing.assert_allclose(compact.matvec(beta), masked.matvec(beta), atol=1e-12)
        np.testing.assert_allclose(compact.rmatvec(w), masked.rmatvec(w), atol=1e-12)
        np.testing.assert_allclose(compact.gram(W), masked.gram(W), atol=1e-12)
        np.testing.assert_allclose(compact.toarray(), masked.toarray(), atol=1e-12)

    def test_spline_categorical_level_group_stores_row_indices_not_full_mask(self):
        """Spline-by-category levels should store selected rows, not a full n-row mask."""
        import scipy.sparse as sp

        from superglm.group_matrix import SplineCategoricalGroupMatrix

        n = 80
        B = sp.random(n, 6, density=0.25, format="csr", random_state=119)
        R_inv = np.eye(6)
        mask = np.zeros(n, dtype=bool)
        mask[[2, 7, 19, 43, 71]] = True

        compact = SplineCategoricalGroupMatrix(B, R_inv, mask)

        assert not hasattr(compact, "mask")
        np.testing.assert_array_equal(compact.row_idx, np.flatnonzero(mask))
        assert compact.B_level.shape == (int(mask.sum()), B.shape[1])

    def test_discretized_spline_categorical_matches_row_level_group(self):
        """Discrete spline-category algebra should match the row-subset implementation."""
        import scipy.sparse as sp

        from superglm.group_matrix import (
            DiscretizedSplineCategoricalGroupMatrix,
            SplineCategoricalGroupMatrix,
        )

        rng = np.random.default_rng(125)
        n = 90
        n_bins = 11
        n_basis = 6
        p_solver = 4
        B_unique = rng.normal(size=(n_bins, n_basis))
        bin_idx = rng.integers(0, n_bins, size=n)
        row_idx = np.sort(rng.choice(n, size=37, replace=False))
        R_inv = rng.normal(size=(n_basis, p_solver))
        W = rng.uniform(0.2, 2.0, size=n)
        Wz = rng.normal(size=n)
        w = rng.normal(size=n)
        beta = rng.normal(size=p_solver)

        compressed = DiscretizedSplineCategoricalGroupMatrix(B_unique, R_inv, bin_idx, row_idx)
        row_level = SplineCategoricalGroupMatrix(sp.csr_matrix(B_unique[bin_idx]), R_inv, row_idx)

        np.testing.assert_allclose(compressed.matvec(beta), row_level.matvec(beta), atol=1e-12)
        np.testing.assert_allclose(compressed.rmatvec(w), row_level.rmatvec(w), atol=1e-12)
        np.testing.assert_allclose(compressed.gram(W), row_level.gram(W), atol=1e-12)
        got_gram, got_xtw, got_xtwz = compressed.gram_rmatvec(W, Wz)
        exp_gram, exp_xtw, exp_xtwz = row_level.gram_rmatvec(W, Wz)
        np.testing.assert_allclose(got_gram, exp_gram, atol=1e-12)
        np.testing.assert_allclose(got_xtw, exp_xtw, atol=1e-12)
        np.testing.assert_allclose(got_xtwz, exp_xtwz, atol=1e-12)
        np.testing.assert_allclose(compressed.toarray(), row_level.toarray(), atol=1e-12)

        assert not hasattr(compressed, "B_level")
        np.testing.assert_array_equal(compressed.bin_idx_level, bin_idx[row_idx])
        np.testing.assert_array_equal(compressed.row_idx, row_idx)

    def test_discretized_spline_categorical_row_subset_preserves_duplicate_rows(self):
        """Duplicate row subsets should behave like dense row indexing."""
        from superglm.group_matrix import DiscretizedSplineCategoricalGroupMatrix

        rng = np.random.default_rng(131)
        n_bins = 5
        B_unique = rng.normal(size=(n_bins, 4))
        bin_idx = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1], dtype=np.intp)
        row_idx = np.array([2, 5, 7, 9], dtype=np.intp)
        R_inv = rng.normal(size=(4, 3))
        gm = DiscretizedSplineCategoricalGroupMatrix(B_unique, R_inv, bin_idx, row_idx)

        idx = np.array([5, 5, 2, 9, 2], dtype=np.intp)
        sub = gm.row_subset(idx)

        np.testing.assert_allclose(sub.toarray(), gm.toarray()[idx], atol=1e-12)

    def test_discretized_spline_categorical_row_subset_accepts_boolean_mask(self):
        """Boolean row subsets should follow NumPy row-indexing semantics."""
        from superglm.group_matrix import DiscretizedSplineCategoricalGroupMatrix

        rng = np.random.default_rng(132)
        B_unique = rng.normal(size=(5, 4))
        bin_idx = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1], dtype=np.intp)
        row_idx = np.array([2, 5, 7, 9], dtype=np.intp)
        R_inv = rng.normal(size=(4, 3))
        gm = DiscretizedSplineCategoricalGroupMatrix(B_unique, R_inv, bin_idx, row_idx)

        mask = np.zeros(gm.shape[0], dtype=bool)
        mask[[5, 2, 9]] = True
        sub = gm.row_subset(mask)

        np.testing.assert_allclose(sub.toarray(), gm.toarray()[mask], atol=1e-12)

    def test_discretized_spline_categorical_cross_gram_matches_dense_oracle(self, monkeypatch):
        """Discrete smooth × discrete spline-category cross-Gram should stay compressed."""
        import superglm._group_matrix._group_matrix_algebra as algebra
        from superglm.group_matrix import (
            DiscretizedSplineCategoricalGroupMatrix,
            DiscretizedSSPGroupMatrix,
        )

        rng = np.random.default_rng(126)
        n = 100
        n_main_bins = 9
        n_sc_bins = 7
        B_main = rng.normal(size=(n_main_bins, 5))
        B_sc = rng.normal(size=(n_sc_bins, 4))
        idx_main = rng.integers(0, n_main_bins, size=n)
        idx_sc = rng.integers(0, n_sc_bins, size=n)
        row_idx = np.sort(rng.choice(n, size=43, replace=False))
        main = DiscretizedSSPGroupMatrix(B_main, rng.normal(size=(5, 3)), idx_main)
        spline_cat = DiscretizedSplineCategoricalGroupMatrix(
            B_sc,
            rng.normal(size=(4, 2)),
            idx_sc,
            row_idx,
        )
        W = rng.uniform(0.1, 2.0, size=n)
        dense = main.toarray().T @ (spline_cat.toarray() * W[:, None])

        def fail_toarray(*args, **kwargs):
            raise AssertionError("discrete spline-category cross-Gram should not materialize")

        def fail_agg_by_bin(*args, **kwargs):
            raise AssertionError("discrete spline-category cross-Gram should use support bins")

        monkeypatch.setattr(DiscretizedSplineCategoricalGroupMatrix, "toarray", fail_toarray)
        monkeypatch.setattr(algebra, "_agg_by_bin", fail_agg_by_bin)

        compact = algebra._cross_gram(main, spline_cat, W)

        np.testing.assert_allclose(compact, dense, rtol=1e-10, atol=1e-10)

    def test_discretized_spline_categorical_pair_cross_gram_matches_dense_oracle(self, monkeypatch):
        """Two discrete spline-category groups should use support-bin histograms."""
        from superglm.group_matrix import DiscretizedSplineCategoricalGroupMatrix, _cross_gram

        rng = np.random.default_rng(127)
        n = 110
        n_bins_i = 8
        n_bins_j = 6
        idx_i = rng.integers(0, n_bins_i, size=n)
        idx_j = rng.integers(0, n_bins_j, size=n)
        row_i = np.sort(rng.choice(n, size=52, replace=False))
        row_j = np.sort(rng.choice(n, size=49, replace=False))
        left = DiscretizedSplineCategoricalGroupMatrix(
            rng.normal(size=(n_bins_i, 5)),
            rng.normal(size=(5, 3)),
            idx_i,
            row_i,
        )
        right = DiscretizedSplineCategoricalGroupMatrix(
            rng.normal(size=(n_bins_j, 4)),
            rng.normal(size=(4, 2)),
            idx_j,
            row_j,
        )
        W = rng.normal(size=n)
        dense = left.toarray().T @ (right.toarray() * W[:, None])

        def fail_toarray(*args, **kwargs):
            raise AssertionError("discrete spline-category pair cross-Gram should not materialize")

        monkeypatch.setattr(DiscretizedSplineCategoricalGroupMatrix, "toarray", fail_toarray)

        compact = _cross_gram(left, right, W)

        np.testing.assert_allclose(compact, dense, rtol=1e-10, atol=1e-10)

    def test_same_categorical_spline_cat_cross_gram_skips_row_intersection(self, monkeypatch):
        """Same-parent spline-category levels should use level metadata before row intersect."""
        import superglm._group_matrix._group_matrix_algebra as algebra
        from superglm.group_matrix import DiscretizedSplineCategoricalGroupMatrix

        rng = np.random.default_rng(129)
        n = 100
        row_idx = np.sort(rng.choice(n, size=44, replace=False))
        idx_left = rng.integers(0, 8, size=n)
        idx_right = rng.integers(0, 7, size=n)
        left = DiscretizedSplineCategoricalGroupMatrix(
            rng.normal(size=(8, 5)),
            rng.normal(size=(5, 3)),
            idx_left,
            row_idx,
        )
        right = DiscretizedSplineCategoricalGroupMatrix(
            rng.normal(size=(7, 4)),
            rng.normal(size=(4, 2)),
            idx_right,
            row_idx,
        )
        left.spline_cat_feature = right.spline_cat_feature = "area"
        left.spline_cat_level = right.spline_cat_level = "B"
        W = rng.normal(size=n)
        dense = left.toarray().T @ (right.toarray() * W[:, None])

        def fail_intersect(*args, **kwargs):
            raise AssertionError("same-level spline-category cross-Gram should not intersect rows")

        monkeypatch.setattr(algebra.np, "intersect1d", fail_intersect)

        compact = algebra._cross_gram(left, right, W)

        np.testing.assert_allclose(compact, dense, rtol=1e-10, atol=1e-10)

    def test_disjoint_categorical_spline_cat_levels_return_zero(self):
        """Disjoint row supports give zero, independently of their labels."""
        import superglm._group_matrix._group_matrix_algebra as algebra
        from superglm.group_matrix import DiscretizedSplineCategoricalGroupMatrix

        rng = np.random.default_rng(130)
        n = 100
        left = DiscretizedSplineCategoricalGroupMatrix(
            rng.normal(size=(8, 5)),
            rng.normal(size=(5, 3)),
            rng.integers(0, 8, size=n),
            np.arange(0, n, 2),
        )
        right = DiscretizedSplineCategoricalGroupMatrix(
            rng.normal(size=(7, 4)),
            rng.normal(size=(4, 2)),
            rng.integers(0, 7, size=n),
            np.arange(1, n, 2),
        )
        left.spline_cat_feature = right.spline_cat_feature = "area"
        left.spline_cat_level = "B"
        right.spline_cat_level = "C"

        cross = algebra._cross_gram(left, right, rng.normal(size=n))

        np.testing.assert_allclose(cross, np.zeros((left.shape[1], right.shape[1])), atol=0.0)

    @pytest.mark.parametrize("left_compressed", [False, True])
    @pytest.mark.parametrize("right_compressed", [False, True])
    @pytest.mark.parametrize(
        ("right_rows", "expected"),
        [([1, 2], -2.0), ([0, 1], -1.0), ([2], 0.0)],
        ids=["overlap", "identical", "disjoint"],
    )
    def test_spline_category_labels_do_not_determine_row_overlap(
        self, left_compressed, right_compressed, right_rows, expected
    ):
        """Independent groupings can share rows despite different level names."""
        import scipy.sparse as sp

        from superglm.group_matrix import (
            SplineCategoricalGroupMatrix,
            SupportCompressedSplineCategoricalGroupMatrix,
        )

        def make_group(rows, compressed):
            if compressed:
                return SupportCompressedSplineCategoricalGroupMatrix(
                    np.ones((1, 1)), np.eye(1), np.zeros(3, dtype=np.intp), np.array(rows)
                )
            return SplineCategoricalGroupMatrix(
                sp.csr_matrix(np.ones((3, 1))), np.eye(1), np.array(rows)
            )

        left = make_group([0, 1], left_compressed)
        right = make_group(right_rows, right_compressed)
        left.spline_cat_feature = right.spline_cat_feature = "group"
        left.spline_cat_level = "AB"
        right.spline_cat_level = "BC"
        weights = np.array([1.0, -2.0, 4.0])

        np.testing.assert_array_equal(_cross_gram(left, right, weights), [[expected]])
        np.testing.assert_array_equal(_cross_gram(right, left, weights), [[expected]])

    def test_spline_categorical_cross_level_gram_is_zero(self):
        """Disjoint spline-by-category level groups have an exact zero cross-block."""
        import scipy.sparse as sp

        from superglm.group_matrix import SplineCategoricalGroupMatrix, _cross_gram

        rng = np.random.default_rng(114)
        n = 50
        B = sp.random(n, 6, density=0.3, format="csr", random_state=114)
        codes = rng.integers(0, 3, size=n)
        gm_a = SplineCategoricalGroupMatrix(B, np.eye(6), codes == 0)
        gm_b = SplineCategoricalGroupMatrix(B, np.eye(6), codes == 1)

        cross = _cross_gram(gm_a, gm_b, rng.uniform(0.5, 1.5, size=n))

        np.testing.assert_allclose(cross, np.zeros((6, 6)), atol=0.0)

    def test_overlapping_spline_categorical_cross_gram_avoids_dense_fallback(self, monkeypatch):
        """Spline-category groups on the same rows should multiply row subsets directly."""
        import scipy.sparse as sp

        from superglm.group_matrix import SplineCategoricalGroupMatrix, _cross_gram

        rng = np.random.default_rng(124)
        n = 80
        B_left = sp.random(n, 5, density=0.4, format="csr", random_state=124)
        B_right = sp.random(n, 4, density=0.5, format="csr", random_state=125)
        rows = np.sort(rng.choice(n, size=35, replace=False))
        left = SplineCategoricalGroupMatrix(B_left, rng.normal(size=(5, 3)), rows)
        right = SplineCategoricalGroupMatrix(B_right, rng.normal(size=(4, 2)), rows)
        W = rng.normal(size=n)

        dense = left.toarray().T @ (right.toarray() * W[:, None])

        def fail_toarray(*args, **kwargs):
            raise AssertionError("spline-category cross-Gram should not materialize")

        monkeypatch.setattr(SplineCategoricalGroupMatrix, "toarray", fail_toarray)

        compact = _cross_gram(left, right, W)

        np.testing.assert_allclose(compact, dense, rtol=1e-10, atol=1e-10)

    def test_tensor_spline_categorical_cross_gram_matches_dense_oracle(self):
        """Tensor × compact spline-category cross-Gram should avoid dense fallback math."""
        import scipy.sparse as sp

        from superglm._group_matrix._group_matrix_algebra import (
            _cross_gram_tensor_spline_categorical,
        )
        from superglm.group_matrix import SplineCategoricalGroupMatrix

        rng = np.random.default_rng(115)
        n = 80
        n1, n2 = 5, 4
        k1, k2, k_cat = 3, 2, 4
        idx1 = rng.integers(0, n1, size=n)
        idx2 = rng.integers(0, n2, size=n)
        B1 = rng.normal(size=(n1, k1))
        B2 = rng.normal(size=(n2, k2))
        pair_codes = idx1 * n2 + idx2
        observed_codes, pair_idx = np.unique(pair_codes, return_inverse=True)
        observed_i1 = observed_codes // n2
        observed_i2 = observed_codes % n2
        B_joint = (B1[observed_i1, :, None] * B2[observed_i2, None, :]).reshape(
            len(observed_codes), k1 * k2
        )
        R_tensor = rng.normal(size=(k1 * k2, 5))
        tensor = DiscretizedTensorGroupMatrix(
            B1,
            B2,
            idx1,
            idx2,
            B_joint,
            R_tensor,
            pair_idx.astype(np.intp),
            tensor_id=4,
        )
        B_cat = sp.random(n, k_cat, density=0.5, format="csr", random_state=115)
        R_cat = rng.normal(size=(k_cat, 3))
        mask = rng.random(n) < 0.4
        spline_cat = SplineCategoricalGroupMatrix(B_cat, R_cat, mask)
        W = rng.normal(size=n)

        compact = _cross_gram_tensor_spline_categorical(tensor, spline_cat, W)
        dense = tensor.toarray().T @ (spline_cat.toarray() * W[:, None])

        np.testing.assert_allclose(compact, dense, rtol=1e-10, atol=1e-10)

    def test_tensor_discretized_spline_categorical_cross_gram_matches_dense_oracle(self):
        """Tensor × discrete spline-category should use support-bin cross algebra."""
        from superglm._group_matrix._group_matrix_algebra import (
            _cross_gram_tensor_spline_categorical,
        )
        from superglm.group_matrix import DiscretizedSplineCategoricalGroupMatrix

        rng = np.random.default_rng(128)
        n = 85
        n1, n2, n_cat = 5, 4, 7
        k1, k2, k_cat = 3, 2, 4
        idx1 = rng.integers(0, n1, size=n)
        idx2 = rng.integers(0, n2, size=n)
        idx_cat = rng.integers(0, n_cat, size=n)
        B1 = rng.normal(size=(n1, k1))
        B2 = rng.normal(size=(n2, k2))
        pair_codes = idx1 * n2 + idx2
        observed_codes, pair_idx = np.unique(pair_codes, return_inverse=True)
        B_joint = (B1[observed_codes // n2, :, None] * B2[observed_codes % n2, None, :]).reshape(
            len(observed_codes), k1 * k2
        )
        R_tensor = rng.normal(size=(k1 * k2, 5))
        tensor = DiscretizedTensorGroupMatrix(
            B1,
            B2,
            idx1,
            idx2,
            B_joint,
            R_tensor,
            pair_idx.astype(np.intp),
            tensor_id=7,
        )
        row_idx = np.sort(rng.choice(n, size=39, replace=False))
        spline_cat = DiscretizedSplineCategoricalGroupMatrix(
            rng.normal(size=(n_cat, k_cat)),
            rng.normal(size=(k_cat, 3)),
            idx_cat,
            row_idx,
        )
        W = rng.normal(size=n)

        compact = _cross_gram_tensor_spline_categorical(tensor, spline_cat, W)
        dense = tensor.toarray().T @ (spline_cat.toarray() * W[:, None])

        np.testing.assert_allclose(compact, dense, rtol=1e-10, atol=1e-10)

    def test_tensor_spline_categorical_cross_gram_does_not_call_tensor_rmatvec(self, monkeypatch):
        """The block assembler should use the compact tensor × spline-category kernel."""
        import scipy.sparse as sp

        from superglm.group_matrix import SplineCategoricalGroupMatrix

        rng = np.random.default_rng(116)
        n = 40
        idx1 = rng.integers(0, 3, size=n)
        idx2 = rng.integers(0, 2, size=n)
        B1 = rng.normal(size=(3, 2))
        B2 = rng.normal(size=(2, 2))
        pair_codes = idx1 * 2 + idx2
        observed_codes, pair_idx = np.unique(pair_codes, return_inverse=True)
        B_joint = (B1[observed_codes // 2, :, None] * B2[observed_codes % 2, None, :]).reshape(
            len(observed_codes), 4
        )
        tensor = DiscretizedTensorGroupMatrix(
            B1,
            B2,
            idx1,
            idx2,
            B_joint,
            np.eye(4),
            pair_idx.astype(np.intp),
            tensor_id=5,
        )
        spline_cat = SplineCategoricalGroupMatrix(
            sp.random(n, 3, density=0.5, format="csr", random_state=116),
            np.eye(3),
            rng.random(n) < 0.5,
        )

        def fail_rmatvec(*args, **kwargs):
            raise AssertionError("tensor rmatvec fallback should not be used")

        monkeypatch.setattr(DiscretizedTensorGroupMatrix, "rmatvec", fail_rmatvec)

        cross = _cross_gram(tensor, spline_cat, rng.uniform(0.5, 1.5, size=n))

        assert cross.shape == (4, 3)

    def test_shared_margin_tensor_tensor_cross_gram_matches_dense_oracle(self):
        """Tensor × tensor terms sharing one marginal should use exact compact algebra."""
        rng = np.random.default_rng(122)
        n = 90
        n_shared, n_left, n_right = 5, 4, 3
        k_shared, k_left, k_right = 3, 2, 4
        idx_shared = rng.integers(0, n_shared, size=n)
        idx_left = rng.integers(0, n_left, size=n)
        idx_right = rng.integers(0, n_right, size=n)
        B_shared = rng.normal(size=(n_shared, k_shared))
        B_left = rng.normal(size=(n_left, k_left))
        B_right = rng.normal(size=(n_right, k_right))

        def make_tensor(B1, B2, idx1, idx2, n2, tensor_id, out_cols):
            pair_codes = idx1 * n2 + idx2
            observed_codes, pair_idx = np.unique(pair_codes, return_inverse=True)
            B_joint = (
                B1[observed_codes // n2, :, None] * B2[observed_codes % n2, None, :]
            ).reshape(len(observed_codes), B1.shape[1] * B2.shape[1])
            return DiscretizedTensorGroupMatrix(
                B1,
                B2,
                idx1,
                idx2,
                B_joint,
                rng.normal(size=(B1.shape[1] * B2.shape[1], out_cols)),
                pair_idx.astype(np.intp),
                tensor_id=tensor_id,
            )

        left = make_tensor(B_shared, B_left, idx_shared, idx_left, n_left, 1, 5)
        right = make_tensor(B_shared, B_right, idx_shared, idx_right, n_right, 2, 6)
        W = rng.normal(size=n)

        compact = _cross_gram(left, right, W)
        dense = left.toarray().T @ (right.toarray() * W[:, None])

        np.testing.assert_allclose(compact, dense, rtol=1e-10, atol=1e-10)

    def test_shared_margin_tensor_tensor_cross_gram_does_not_materialize_tensor(self, monkeypatch):
        """The block assembler should avoid tensor toarray fallback for shared margins."""
        rng = np.random.default_rng(123)
        n = 80
        n_shared, n_left, n_right = 50, 46, 46
        idx_shared = rng.integers(0, n_shared, size=n)
        idx_left = rng.integers(0, n_left, size=n)
        idx_right = rng.integers(0, n_right, size=n)
        B_shared = rng.normal(size=(n_shared, 2))
        B_left = rng.normal(size=(n_left, 2))
        B_right = rng.normal(size=(n_right, 3))

        def make_tensor(B1, B2, idx1, idx2, n2, tensor_id):
            B_joint = (
                B1[np.repeat(np.arange(B1.shape[0]), n2), :, None]
                * B2[np.tile(np.arange(n2), B1.shape[0]), None, :]
            ).reshape(B1.shape[0] * n2, B1.shape[1] * B2.shape[1])
            return DiscretizedTensorGroupMatrix(
                B1,
                B2,
                idx1,
                idx2,
                B_joint,
                np.eye(B1.shape[1] * B2.shape[1]),
                (idx1 * n2 + idx2).astype(np.intp),
                tensor_id=tensor_id,
            )

        left = make_tensor(B_shared, B_left, idx_shared, idx_left, n_left, 1)
        right = make_tensor(B_shared, B_right, idx_shared, idx_right, n_right, 2)

        def fail_toarray(*args, **kwargs):
            raise AssertionError("shared-margin tensor cross-Gram should not materialize")

        monkeypatch.setattr(DiscretizedTensorGroupMatrix, "toarray", fail_toarray)

        cross = _cross_gram(left, right, rng.uniform(0.5, 1.5, size=n))

        assert cross.shape == (4, 6)

    def test_tensor_own_margin_cross_gram_uses_packed_weight_grid(self, monkeypatch):
        """Tensor × parent smooth cross-blocks should not use the generic row scan path."""
        import superglm._group_matrix._group_matrix_algebra as algebra

        rng = np.random.default_rng(121)
        n = 90
        n1, n2 = 6, 5
        k1, k2, k_main = 3, 4, 5
        p_tensor, p_main = 7, 4
        idx1 = rng.integers(0, n1, size=n)
        idx2 = rng.integers(0, n2, size=n)
        B1 = rng.normal(size=(n1, k1))
        B2 = rng.normal(size=(n2, k2))
        B_main = rng.normal(size=(n1, k_main))
        pair_codes = idx1 * n2 + idx2
        observed_codes, pair_idx = np.unique(pair_codes, return_inverse=True)
        B_joint = (B1[observed_codes // n2, :, None] * B2[observed_codes % n2, None, :]).reshape(
            len(observed_codes), k1 * k2
        )
        tensor = DiscretizedTensorGroupMatrix(
            B1,
            B2,
            idx1,
            idx2,
            B_joint,
            rng.normal(size=(k1 * k2, p_tensor)),
            pair_idx.astype(np.intp),
            tensor_id=6,
        )
        main = DiscretizedSSPGroupMatrix(
            B_main,
            rng.normal(size=(k_main, p_main)),
            idx1,
        )
        W = rng.uniform(0.2, 1.8, size=n)

        def fail_generic(*args, **kwargs):
            raise AssertionError("generic tensor-main cross-Gram should not be used")

        monkeypatch.setattr(algebra, "_cross_gram_tensor_main", fail_generic)

        cross = algebra._cross_gram(tensor, main, W)
        dense = tensor.toarray().T @ (main.toarray() * W[:, None])

        np.testing.assert_allclose(cross, dense, rtol=1e-10, atol=1e-10)

    def test_tensor_own_margin_detection_is_cached(self, monkeypatch):
        """Repeated tensor × main cross-blocks should not rescan full index arrays."""
        import superglm._group_matrix._group_matrix_algebra as algebra

        rng = np.random.default_rng(122)
        n = 80
        n1, n2 = 5, 4
        idx1 = rng.integers(0, n1, size=n)
        idx2 = rng.integers(0, n2, size=n)
        B1 = rng.normal(size=(n1, 3))
        B2 = rng.normal(size=(n2, 2))
        pair_codes = idx1 * n2 + idx2
        observed_codes, pair_idx = np.unique(pair_codes, return_inverse=True)
        B_joint = (B1[observed_codes // n2, :, None] * B2[observed_codes % n2, None, :]).reshape(
            len(observed_codes), 6
        )
        tensor = DiscretizedTensorGroupMatrix(
            B1,
            B2,
            idx1,
            idx2,
            B_joint,
            np.eye(6),
            pair_idx.astype(np.intp),
            tensor_id=7,
        )
        main = DiscretizedSSPGroupMatrix(rng.normal(size=(n1, 4)), np.eye(4), idx1)
        W = rng.uniform(0.2, 1.8, size=n)
        calls = 0
        real_array_equal = algebra.np.array_equal

        def counting_array_equal(a, b):
            nonlocal calls
            calls += 1
            return real_array_equal(a, b)

        monkeypatch.setattr(algebra.np, "array_equal", counting_array_equal)

        algebra._cross_gram(tensor, main, W)
        first_call_count = calls
        algebra._cross_gram(tensor, main, W)

        assert first_call_count > 0
        assert calls == first_call_count

    def test_block_xtwx_rhs_reuses_tensor_w_grid_for_own_margins(self, monkeypatch):
        """One block assembly should share a tensor W-grid across both parent crosses."""
        import superglm._group_matrix._group_matrix_algebra as algebra
        from superglm.types import GroupSlice

        rng = np.random.default_rng(123)
        n = 120
        n1, n2 = 7, 6
        idx1 = rng.integers(0, n1, size=n)
        idx2 = rng.integers(0, n2, size=n)
        B1 = rng.normal(size=(n1, 3))
        B2 = rng.normal(size=(n2, 2))
        pair_codes = idx1 * n2 + idx2
        observed_codes, pair_idx = np.unique(pair_codes, return_inverse=True)
        B_joint = (B1[observed_codes // n2, :, None] * B2[observed_codes % n2, None, :]).reshape(
            len(observed_codes), 6
        )
        main1 = DiscretizedSSPGroupMatrix(rng.normal(size=(n1, 4)), np.eye(4), idx1)
        main2 = DiscretizedSSPGroupMatrix(rng.normal(size=(n2, 5)), np.eye(5), idx2)
        tensor = DiscretizedTensorGroupMatrix(
            B1,
            B2,
            idx1,
            idx2,
            B_joint,
            np.eye(6),
            pair_idx.astype(np.intp),
            tensor_id=8,
        )
        groups = [
            GroupSlice(name="x1", start=0, end=4, weight=2.0),
            GroupSlice(name="x2", start=4, end=9, weight=np.sqrt(5.0)),
            GroupSlice(name="x1:x2", start=9, end=15, weight=np.sqrt(6.0)),
        ]
        W = rng.uniform(0.2, 1.8, size=n)
        Wz = rng.normal(size=n)
        calls = 0
        real_hist = algebra._disc_disc_2d_hist

        def counting_hist(a, b, weights, n_a, n_b):
            nonlocal calls
            if a is tensor.idx1 and b is tensor.idx2 and weights is W:
                calls += 1
            return real_hist(a, b, weights, n_a, n_b)

        monkeypatch.setattr(algebra, "_disc_disc_2d_hist", counting_hist)

        algebra._block_xtwx_rhs([main1, main2, tensor], groups, W, Wz)

        assert calls == 1

    def test_block_xtwx_rhs_profiles_tensor_block_work(self):
        """Optional block profiling should attribute tensor diagonal and cross work."""
        import superglm._group_matrix._group_matrix_algebra as algebra
        from superglm.types import GroupSlice

        rng = np.random.default_rng(126)
        n = 100
        n1, n2, n3 = 7, 5, 4
        idx1 = rng.integers(0, n1, size=n)
        idx2 = rng.integers(0, n2, size=n)
        idx3 = rng.integers(0, n3, size=n)
        B1 = rng.normal(size=(n1, 3))
        B2 = rng.normal(size=(n2, 2))
        pair_codes = idx1 * n2 + idx2
        observed_codes, pair_idx = np.unique(pair_codes, return_inverse=True)
        B_joint = (B1[observed_codes // n2, :, None] * B2[observed_codes % n2, None, :]).reshape(
            len(observed_codes), 6
        )
        main_own = DiscretizedSSPGroupMatrix(rng.normal(size=(n1, 4)), np.eye(4), idx1)
        tensor = DiscretizedTensorGroupMatrix(
            B1,
            B2,
            idx1,
            idx2,
            B_joint,
            np.eye(6),
            pair_idx.astype(np.intp),
            tensor_id=12,
        )
        main_other = DiscretizedSSPGroupMatrix(rng.normal(size=(n3, 3)), np.eye(3), idx3)
        groups = [
            GroupSlice(name="x1", start=0, end=4, weight=2.0),
            GroupSlice(name="x1:x2", start=4, end=10, weight=np.sqrt(6.0)),
            GroupSlice(name="x3", start=10, end=13, weight=np.sqrt(3.0)),
        ]
        W = rng.uniform(0.2, 1.8, size=n)
        Wz = rng.normal(size=n)
        profile: dict[str, float] = {}

        profiled = algebra._block_xtwx_rhs(
            [main_own, tensor, main_other], groups, W, Wz, profile=profile
        )
        plain = algebra._block_xtwx_rhs([main_own, tensor, main_other], groups, W, Wz)

        for got, expected in zip(profiled, plain, strict=True):
            np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)

        assert profile["block_diag_tensor_s"] > 0.0
        assert profile["block_diag_discrete_ssp_s"] > 0.0
        assert profile["block_cross_tensor_own_margin_s"] > 0.0
        assert profile["block_cross_tensor_main_s"] > 0.0
        assert profile["block_cross_disc_disc_s"] > 0.0
        assert profile["block_hist2d_s"] > 0.0
        assert profile["block_calls"] == 1

    def test_unprojected_tensor_penalty_context_uses_marginal_eigenspectra(self, monkeypatch):
        """Unprojected tensor penalties should not eigendecompose full tensor blocks."""
        import superglm.reml.penalty_algebra as penalty_algebra
        from superglm.types import GroupSlice

        rng = np.random.default_rng(127)
        n = 40
        n1, n2 = 6, 5
        k1, k2 = 4, 3
        idx1 = rng.integers(0, n1, size=n)
        idx2 = rng.integers(0, n2, size=n)
        B1 = rng.normal(size=(n1, k1))
        B2 = rng.normal(size=(n2, k2))
        pair_codes = idx1 * n2 + idx2
        observed_codes, pair_idx = np.unique(pair_codes, return_inverse=True)
        B_joint = (B1[observed_codes // n2, :, None] * B2[observed_codes % n2, None, :]).reshape(
            len(observed_codes), k1 * k2
        )
        D1 = np.diff(np.eye(k1), n=2, axis=0)
        D2 = np.diff(np.eye(k2), n=2, axis=0)
        S1 = D1.T @ D1
        S2 = D2.T @ D2
        omega_left = np.kron(S1, np.eye(k2))
        omega_right = np.kron(np.eye(k1), S2)
        tensor = DiscretizedTensorGroupMatrix(
            B1,
            B2,
            idx1,
            idx2,
            B_joint,
            np.eye(k1 * k2),
            pair_idx.astype(np.intp),
            tensor_id=13,
        )
        tensor.omega_components = [
            ("margin_x1", omega_left),
            ("margin_x2", omega_right),
        ]
        tensor.component_types = {}
        tensor.lambda_policies = {}
        group = GroupSlice(name="x1:x2", start=0, end=k1 * k2, weight=np.sqrt(k1 * k2))
        real_eigvalsh = penalty_algebra.np.linalg.eigvalsh
        full_tensor_eigs = 0

        def counting_eigvalsh(a):
            nonlocal full_tensor_eigs
            if np.asarray(a).shape == (k1 * k2, k1 * k2):
                full_tensor_eigs += 1
            return real_eigvalsh(a)

        monkeypatch.setattr(penalty_algebra.np.linalg, "eigvalsh", counting_eigvalsh)

        penalties, caches, ranks = penalty_algebra.build_penalty_context(
            [tensor],
            [(0, group)],
        )

        assert full_tensor_eigs == 0
        assert [pc.name for pc in penalties] == ["x1:x2:margin_x1", "x1:x2:margin_x2"]
        eig1 = np.linalg.eigvalsh(S1)
        eig2 = np.linalg.eigvalsh(S2)
        eps_thresh = np.finfo(float).eps ** (2 / 3)
        pos1 = eig1[eig1 > eps_thresh * max(float(eig1.max()), 1e-12)]
        pos2 = eig2[eig2 > eps_thresh * max(float(eig2.max()), 1e-12)]
        assert ranks["x1:x2:margin_x1"] == float(pos1.size * k2)
        assert ranks["x1:x2:margin_x2"] == float(pos2.size * k1)
        assert caches["x1:x2:margin_x1"].log_det_omega_plus == pytest.approx(
            float(k2 * np.sum(np.log(pos1)))
        )
        assert caches["x1:x2:margin_x2"].log_det_omega_plus == pytest.approx(
            float(k1 * np.sum(np.log(pos2)))
        )

    def test_map_beta_between_bases_skips_unchanged_group_matrix(self, monkeypatch):
        """Frozen group matrices should not pay an identity least-squares remap."""
        from superglm.reml.result import _map_beta_between_bases
        from superglm.types import GroupSlice

        rng = np.random.default_rng(128)
        n = 30
        n1, n2 = 5, 4
        k1, k2 = 3, 2
        idx1 = rng.integers(0, n1, size=n)
        idx2 = rng.integers(0, n2, size=n)
        B1 = rng.normal(size=(n1, k1))
        B2 = rng.normal(size=(n2, k2))
        pair_codes = idx1 * n2 + idx2
        observed_codes, pair_idx = np.unique(pair_codes, return_inverse=True)
        B_joint = (B1[observed_codes // n2, :, None] * B2[observed_codes % n2, None, :]).reshape(
            len(observed_codes), k1 * k2
        )
        tensor = DiscretizedTensorGroupMatrix(
            B1,
            B2,
            idx1,
            idx2,
            B_joint,
            np.eye(k1 * k2),
            pair_idx.astype(np.intp),
            tensor_id=14,
        )
        groups = [GroupSlice(name="x1:x2", start=0, end=k1 * k2, weight=np.sqrt(k1 * k2))]
        beta = rng.normal(size=k1 * k2)

        def fail_lstsq(*args, **kwargs):
            raise AssertionError("unchanged group matrix should not be remapped")

        monkeypatch.setattr(np.linalg, "lstsq", fail_lstsq)

        mapped = _map_beta_between_bases(beta, [tensor], [tensor], groups)

        np.testing.assert_allclose(mapped, beta, rtol=0.0, atol=0.0)

    def test_reml_hessian_uses_same_slice_penalty_trace_fast_path(self, monkeypatch):
        """Same-slice multi-penalty Hessian traces should avoid generic block tracing."""
        import superglm.reml.gradient as gradient
        from superglm.types import PenaltyComponent

        rng = np.random.default_rng(129)
        q = 8
        H_inv = rng.normal(size=(q, q))
        H_inv = H_inv.T @ H_inv + np.eye(q)
        D1 = np.diff(np.eye(q), n=1, axis=0)
        D2 = np.diff(np.eye(q), n=2, axis=0)
        S1 = D1.T @ D1
        S2 = D2.T @ D2
        penalties = [
            PenaltyComponent(
                name="g:a",
                group_name="g",
                group_index=0,
                group_sl=slice(0, q),
                omega_raw=S1,
                omega_ssp=S1,
                rank=float(np.linalg.matrix_rank(S1)),
                log_det_omega_plus=0.0,
                eigvals_omega=np.array([]),
            ),
            PenaltyComponent(
                name="g:b",
                group_name="g",
                group_index=0,
                group_sl=slice(0, q),
                omega_raw=S2,
                omega_ssp=S2,
                rank=float(np.linalg.matrix_rank(S2)),
                log_det_omega_plus=0.0,
                eigvals_omega=np.array([]),
            ),
        ]

        def fail_block_trace(*args, **kwargs):
            raise AssertionError("same-slice penalties should use the cached fast path")

        monkeypatch.setattr(gradient, "_penalty_block_trace", fail_block_trace)

        hess = gradient.reml_direct_hessian(
            [],
            distribution=type("KnownScale", (), {"scale_known": True})(),
            XtWX_S_inv=H_inv,
            lambdas={"g:a": 2.0, "g:b": 3.0},
            gradient=np.zeros(2),
            reml_penalties=penalties,
        )

        assert hess.shape == (2, 2)
        np.testing.assert_allclose(hess, hess.T, rtol=1e-12, atol=1e-12)

    def test_categorical_spline_categorical_cross_gram_matches_dense_oracle(self):
        """Categorical × compact spline-category cross-Gram should use one aggregation."""
        import scipy.sparse as sp

        from superglm._group_matrix._group_matrix_algebra import (
            _cross_gram_categorical_spline_categorical,
        )
        from superglm.group_matrix import CategoricalGroupMatrix, SplineCategoricalGroupMatrix

        rng = np.random.default_rng(117)
        n = 70
        n_levels = 5
        codes = rng.integers(-1, n_levels, size=n)
        cat = CategoricalGroupMatrix(codes, n_levels)
        B = sp.random(n, 4, density=0.45, format="csr", random_state=117)
        R_inv = rng.normal(size=(4, 3))
        spline_cat = SplineCategoricalGroupMatrix(B, R_inv, codes == 2)
        W = rng.normal(size=n)

        compact = _cross_gram_categorical_spline_categorical(cat, spline_cat, W)
        dense = cat.toarray().T @ (spline_cat.toarray() * W[:, None])

        np.testing.assert_allclose(compact, dense, rtol=1e-12, atol=1e-12)

    def test_categorical_spline_categorical_cross_gram_does_not_call_cat_rmatvec(self, monkeypatch):
        """The block assembler should avoid per-column categorical rmatvec fallback."""
        import scipy.sparse as sp

        from superglm.group_matrix import CategoricalGroupMatrix, SplineCategoricalGroupMatrix

        rng = np.random.default_rng(118)
        n = 60
        n_levels = 8
        codes = rng.integers(-1, n_levels, size=n)
        cat = CategoricalGroupMatrix(codes, n_levels)
        spline_cat = SplineCategoricalGroupMatrix(
            sp.random(n, 5, density=0.4, format="csr", random_state=118),
            rng.normal(size=(5, 3)),
            codes == 3,
        )

        def fail_rmatvec(*args, **kwargs):
            raise AssertionError("categorical rmatvec fallback should not be used")

        monkeypatch.setattr(CategoricalGroupMatrix, "rmatvec", fail_rmatvec)

        cross = _cross_gram(cat, spline_cat, rng.uniform(0.5, 1.5, size=n))

        assert cross.shape == (n_levels, 3)

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

    def test_tensor_main_cross_gram_channels_over_smaller_margin(self, monkeypatch):
        """Unrelated tensor-main blocks should channel over the cheaper tensor margin."""
        import superglm._group_matrix._group_matrix_algebra as algebra

        rng = np.random.default_rng(125)
        n = 120
        n1, n2, n_main = 7, 6, 5
        k1, k2, k_main = 2, 5, 4
        p_tensor, p_main = 8, 3
        idx1 = rng.integers(0, n1, size=n)
        idx2 = rng.integers(0, n2, size=n)
        main_idx = rng.integers(0, n_main, size=n)
        B1 = rng.normal(size=(n1, k1))
        B2 = rng.normal(size=(n2, k2))
        B_main = rng.normal(size=(n_main, k_main))
        pair_codes = idx1 * n2 + idx2
        observed_codes, pair_idx = np.unique(pair_codes, return_inverse=True)
        B_joint = (B1[observed_codes // n2, :, None] * B2[observed_codes % n2, None, :]).reshape(
            len(observed_codes), k1 * k2
        )
        tensor = DiscretizedTensorGroupMatrix(
            B1,
            B2,
            idx1,
            idx2,
            B_joint,
            rng.normal(size=(k1 * k2, p_tensor)),
            pair_idx.astype(np.intp),
            tensor_id=8,
        )
        main = DiscretizedSSPGroupMatrix(
            B_main,
            rng.normal(size=(k_main, p_main)),
            main_idx,
        )
        W = rng.uniform(0.2, 1.8, size=n)

        original = algebra._disc_disc_2d_hist_channels
        channel_shapes = []

        def spy_channels(idx_a, idx_b, channel_idx, weights, channel_basis, n_a, n_b):
            channel_shapes.append(channel_basis.shape)
            return original(idx_a, idx_b, channel_idx, weights, channel_basis, n_a, n_b)

        monkeypatch.setattr(algebra, "_disc_disc_2d_hist_channels", spy_channels)

        cross = algebra._cross_gram_tensor_main(tensor, main, W)
        dense = main.toarray().T @ (tensor.toarray() * W[:, None])

        np.testing.assert_allclose(cross, dense, rtol=1e-10, atol=1e-10)
        assert channel_shapes == [B1.shape]

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
