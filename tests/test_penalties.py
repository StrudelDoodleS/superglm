"""Tests for penalty objects, flavors, and warm starting."""

import numpy as np
import pandas as pd
import pytest

from superglm.penalties.base import penalty_targets_group, validate_penalty_features
from superglm.penalties.flavors import Adaptive
from superglm.penalties.group_elastic_net import GroupElasticNet
from superglm.penalties.group_lasso import GroupLasso
from superglm.penalties.ridge import Ridge
from superglm.penalties.sparse_group_lasso import SparseGroupLasso
from superglm.sklearn import SuperGLMRegressor
from superglm.types import GroupSlice


@pytest.fixture
def groups():
    return [GroupSlice("a", 0, 3, weight=np.sqrt(3)), GroupSlice("b", 3, 4, weight=1.0)]


@pytest.fixture
def sample_data():
    rng = np.random.default_rng(42)
    n = 500
    age = rng.uniform(18, 85, n)
    region = rng.choice(["A", "B", "C"], n, p=[0.3, 0.3, 0.4])
    density = rng.normal(5, 2, n)
    exposure = rng.uniform(0.3, 1.0, n)
    mu = np.exp(-2.0 + 0.01 * (age - 50) ** 2 / 100 + (region == "A") * 0.3)
    y = rng.poisson(mu * exposure).astype(float)
    X = pd.DataFrame({"age": age, "region": region, "density": density})
    return X, y, exposure


class TestGroupLasso:
    def test_prox_shrinks(self, groups):
        beta = np.array([0.5, 0.5, 0.5, 2.0])
        pen = GroupLasso(lambda1=0.1)
        result = pen.prox(beta, groups, step=1.0)
        # Group a should shrink, group b should shrink
        assert np.linalg.norm(result[:3]) < np.linalg.norm(beta[:3])
        assert abs(result[3]) < abs(beta[3])

    def test_eval_positive(self, groups):
        beta = np.array([1.0, 2.0, 3.0, 4.0])
        pen = GroupLasso(lambda1=0.5)
        assert pen.eval(beta, groups) > 0

    def test_eval_zero_for_zero_beta(self, groups):
        pen = GroupLasso(lambda1=0.5)
        assert pen.eval(np.zeros(4), groups) == 0.0

    def test_prox_does_not_modify_input(self, groups):
        beta = np.array([1.0, 2.0, 3.0, 4.0])
        beta_copy = beta.copy()
        GroupLasso(lambda1=0.1).prox(beta, groups, step=1.0)
        np.testing.assert_array_equal(beta, beta_copy)

    def test_feature_filter(self):
        groups = [
            GroupSlice("age", 0, 2, weight=np.sqrt(2), feature_name="age"),
            GroupSlice("region", 2, 4, weight=np.sqrt(2), feature_name="region"),
        ]
        beta = np.array([2.0, 1.0, 2.0, 1.0])
        pen = GroupLasso(lambda1=1.0, features=["region"])
        result = pen.prox(beta, groups, step=1.0)
        np.testing.assert_allclose(result[:2], beta[:2])
        assert np.linalg.norm(result[2:]) < np.linalg.norm(beta[2:])


class TestGroupLassoProxGroup:
    def test_matches_full_prox(self, groups):
        """prox_group on each group should match full prox."""
        beta = np.array([0.5, 0.5, 0.5, 2.0])
        pen = GroupLasso(lambda1=0.1)
        full_result = pen.prox(beta, groups, step=1.0)
        for g in groups:
            group_result = pen.prox_group(beta[g.sl], g, step=1.0)
            np.testing.assert_allclose(full_result[g.sl], group_result)

    def test_zeroing(self):
        g = GroupSlice("g", 0, 5, weight=1.0)
        pen = GroupLasso(lambda1=0.1)
        bg = np.full(5, 0.01)
        result = pen.prox_group(bg, g, step=1.0)
        np.testing.assert_allclose(result, 0.0)

    def test_nonzeroing(self):
        g = GroupSlice("g", 0, 3, weight=1.0)
        pen = GroupLasso(lambda1=0.1)
        bg = np.array([3.0, 4.0, 0.0])
        result = pen.prox_group(bg, g, step=1.0)
        assert np.linalg.norm(result) > 0


class TestSparseGroupLasso:
    def test_alpha_zero_matches_group_lasso(self, groups):
        """alpha=0 should behave like pure group lasso."""
        beta = np.array([1.0, 2.0, 3.0, 4.0])
        gl = GroupLasso(lambda1=0.1).prox(beta, groups, step=1.0)
        sgl = SparseGroupLasso(lambda1=0.1, alpha=0.0).prox(beta, groups, step=1.0)
        np.testing.assert_allclose(gl, sgl, atol=1e-10)

    def test_alpha_one_is_elementwise(self, groups):
        """alpha=1 should be pure L1 soft-thresholding (no group structure)."""
        beta = np.array([0.5, -0.3, 0.1, 2.0])
        pen = SparseGroupLasso(lambda1=0.2, alpha=1.0)
        result = pen.prox(beta, groups, step=1.0)
        # Pure L1: each element soft-thresholded by 0.2
        expected = np.sign(beta) * np.maximum(np.abs(beta) - 0.2, 0.0)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_within_group_sparsity(self, groups):
        """SGL should zero individual elements within a surviving group."""
        # Group a has one large and two small entries
        beta = np.array([5.0, 0.01, 0.01, 3.0])
        pen = SparseGroupLasso(lambda1=0.1, alpha=0.8)
        result = pen.prox(beta, groups, step=1.0)
        # Group a survives (large element), but small elements may be zeroed
        assert result[0] != 0.0  # large element survives
        assert abs(result[1]) < abs(beta[1])  # small elements shrink

    def test_eval_positive(self, groups):
        beta = np.array([1.0, 2.0, 3.0, 4.0])
        pen = SparseGroupLasso(lambda1=0.5, alpha=0.5)
        assert pen.eval(beta, groups) > 0

    def test_feature_filter_skips_untargeted_group(self):
        groups = [
            GroupSlice("age", 0, 2, weight=np.sqrt(2), feature_name="age"),
            GroupSlice("region", 2, 4, weight=np.sqrt(2), feature_name="region"),
        ]
        beta = np.array([0.8, -0.7, 0.8, -0.7])
        pen = SparseGroupLasso(lambda1=0.5, alpha=1.0, features=["region"])
        result = pen.prox(beta, groups, step=1.0)
        np.testing.assert_allclose(result[:2], beta[:2])
        assert np.linalg.norm(result[2:]) < np.linalg.norm(beta[2:])


class TestSparseGroupLassoProxGroup:
    def test_matches_full_prox(self, groups):
        beta = np.array([0.5, 0.5, 0.5, 2.0])
        pen = SparseGroupLasso(lambda1=0.1, alpha=0.5)
        full_result = pen.prox(beta, groups, step=1.0)
        for g in groups:
            group_result = pen.prox_group(beta[g.sl], g, step=1.0)
            np.testing.assert_allclose(full_result[g.sl], group_result)

    def test_zeroing(self):
        g = GroupSlice("g", 0, 5, weight=1.0)
        pen = SparseGroupLasso(lambda1=0.1, alpha=0.5)
        bg = np.full(5, 0.01)
        result = pen.prox_group(bg, g, step=1.0)
        np.testing.assert_allclose(result, 0.0)


class TestRidge:
    def test_prox_closed_form(self):
        groups = [GroupSlice("g", 0, 3, 1.0)]
        beta = np.array([1.0, 2.0, 3.0])
        pen = Ridge(lambda1=0.5)
        result = pen.prox(beta, groups, step=1.0)
        expected = beta / (1.0 + 0.5)
        np.testing.assert_allclose(result, expected)

    def test_no_zeroing(self):
        """Ridge never zeros coefficients."""
        groups = [GroupSlice("g", 0, 3, 1.0)]
        beta = np.array([0.001, 0.001, 0.001])
        pen = Ridge(lambda1=10.0)
        result = pen.prox(beta, groups, step=1.0)
        assert np.all(result != 0.0)

    def test_eval(self):
        groups = [GroupSlice("g", 0, 2, 1.0)]
        beta = np.array([3.0, 4.0])
        pen = Ridge(lambda1=1.0)
        assert pen.eval(beta, groups) == pytest.approx(12.5)  # 1.0 * 25 / 2

    def test_no_flavor(self):
        assert Ridge(lambda1=1.0).flavor is None

    def test_feature_filter(self):
        groups = [
            GroupSlice("age", 0, 2, weight=np.sqrt(2), feature_name="age"),
            GroupSlice("region", 2, 4, weight=np.sqrt(2), feature_name="region"),
        ]
        beta = np.array([2.0, 1.0, 2.0, 1.0])
        pen = Ridge(lambda1=1.0, features=["region"])
        result = pen.prox(beta, groups, step=1.0)
        np.testing.assert_allclose(result[:2], beta[:2])
        np.testing.assert_allclose(result[2:], beta[2:] / 2.0)


class TestRidgeProxGroup:
    def test_matches_full_prox(self):
        groups = [GroupSlice("g", 0, 3, 1.0)]
        beta = np.array([1.0, 2.0, 3.0])
        pen = Ridge(lambda1=0.5)
        full_result = pen.prox(beta, groups, step=1.0)
        group_result = pen.prox_group(beta, groups[0], step=1.0)
        np.testing.assert_allclose(full_result, group_result)


class TestGroupElasticNet:
    def test_alpha_one_matches_group_lasso(self, groups):
        """alpha=1 should equal GroupLasso."""
        beta = np.array([1.0, 2.0, 3.0, 4.0])
        gl = GroupLasso(lambda1=0.1).prox(beta, groups, step=1.0)
        gen = GroupElasticNet(lambda1=0.1, alpha=1.0).prox(beta, groups, step=1.0)
        np.testing.assert_allclose(gen, gl, atol=1e-10)

    def test_alpha_one_eval_matches_group_lasso(self, groups):
        beta = np.array([1.0, 2.0, 3.0, 4.0])
        gl_val = GroupLasso(lambda1=0.1).eval(beta, groups)
        gen_val = GroupElasticNet(lambda1=0.1, alpha=1.0).eval(beta, groups)
        assert gen_val == pytest.approx(gl_val, abs=1e-10)

    def test_alpha_zero_matches_ridge(self, groups):
        """alpha=0 should equal Ridge."""
        beta = np.array([1.0, 2.0, 3.0, 4.0])
        ridge = Ridge(lambda1=0.1).prox(beta, groups, step=1.0)
        gen = GroupElasticNet(lambda1=0.1, alpha=0.0).prox(beta, groups, step=1.0)
        np.testing.assert_allclose(gen, ridge, atol=1e-10)

    def test_alpha_zero_eval_matches_ridge(self, groups):
        beta = np.array([1.0, 2.0, 3.0, 4.0])
        ridge_val = Ridge(lambda1=0.1).eval(beta, groups)
        gen_val = GroupElasticNet(lambda1=0.1, alpha=0.0).eval(beta, groups)
        assert gen_val == pytest.approx(ridge_val, abs=1e-10)

    def test_prox_shrinks(self, groups):
        beta = np.array([0.5, 0.5, 0.5, 2.0])
        pen = GroupElasticNet(lambda1=0.1, alpha=0.5)
        result = pen.prox(beta, groups, step=1.0)
        assert np.linalg.norm(result[:3]) < np.linalg.norm(beta[:3])
        assert abs(result[3]) < abs(beta[3])

    def test_eval_positive(self, groups):
        beta = np.array([1.0, 2.0, 3.0, 4.0])
        pen = GroupElasticNet(lambda1=0.5, alpha=0.5)
        assert pen.eval(beta, groups) > 0

    def test_eval_zero_for_zero_beta(self, groups):
        pen = GroupElasticNet(lambda1=0.5, alpha=0.5)
        assert pen.eval(np.zeros(4), groups) == 0.0

    def test_feature_filter(self):
        groups = [
            GroupSlice("age", 0, 2, weight=np.sqrt(2), feature_name="age"),
            GroupSlice("region", 2, 4, weight=np.sqrt(2), feature_name="region"),
        ]
        beta = np.array([2.0, 1.0, 2.0, 1.0])
        pen = GroupElasticNet(lambda1=1.0, alpha=0.5, features=["region"])
        result = pen.prox(beta, groups, step=1.0)
        np.testing.assert_allclose(result[:2], beta[:2])
        assert np.linalg.norm(result[2:]) < np.linalg.norm(beta[2:])


class TestPenaltyFeatureHelpers:
    def test_matches_feature_name_and_group_name(self):
        groups = [
            GroupSlice(
                "age:linear",
                0,
                1,
                weight=1.0,
                feature_name="age",
                subgroup_type="linear",
            ),
            GroupSlice(
                "age:spline",
                1,
                4,
                weight=np.sqrt(3),
                feature_name="age",
                subgroup_type="spline",
            ),
            GroupSlice("region", 4, 6, weight=np.sqrt(2), feature_name="region"),
        ]
        pen_feature = GroupLasso(lambda1=0.1, features=["age"])
        assert penalty_targets_group(pen_feature, groups[0])
        assert penalty_targets_group(pen_feature, groups[1])
        assert not penalty_targets_group(pen_feature, groups[2])

        pen_group = GroupLasso(lambda1=0.1, features=["age:spline"])
        assert not penalty_targets_group(pen_group, groups[0])
        assert penalty_targets_group(pen_group, groups[1])
        assert not penalty_targets_group(pen_group, groups[2])

    def test_validate_penalty_features_raises_on_unknown_name(self):
        groups = [GroupSlice("region", 0, 2, feature_name="region")]
        pen = GroupLasso(lambda1=0.1, features=["missing"])
        with pytest.raises(ValueError, match="Unknown penalty feature/group filter"):
            validate_penalty_features(pen, groups)


class TestGroupElasticNetProxGroup:
    def test_matches_full_prox(self, groups):
        """prox_group on each group should match full prox."""
        beta = np.array([0.5, 0.5, 0.5, 2.0])
        pen = GroupElasticNet(lambda1=0.1, alpha=0.5)
        full_result = pen.prox(beta, groups, step=1.0)
        for g in groups:
            group_result = pen.prox_group(beta[g.sl], g, step=1.0)
            np.testing.assert_allclose(full_result[g.sl], group_result)

    def test_zeroing(self):
        """Small groups get zeroed (group lasso part)."""
        g = GroupSlice("g", 0, 5, weight=1.0)
        pen = GroupElasticNet(lambda1=0.1, alpha=0.8)
        bg = np.full(5, 0.01)
        result = pen.prox_group(bg, g, step=1.0)
        np.testing.assert_allclose(result, 0.0)

    def test_no_full_zeroing_with_low_alpha(self):
        """Low alpha (ridge-dominated) doesn't zero groups easily."""
        g = GroupSlice("g", 0, 5, weight=1.0)
        pen = GroupElasticNet(lambda1=0.1, alpha=0.1)
        bg = np.full(5, 0.05)
        result = pen.prox_group(bg, g, step=1.0)
        assert np.linalg.norm(result) > 0


class TestAdaptive:
    def test_weights_inversely_proportional(self):
        groups = [
            GroupSlice("big", 0, 3, weight=np.sqrt(3)),
            GroupSlice("small", 3, 6, weight=np.sqrt(3)),
        ]
        beta_init = np.array([5.0, 5.0, 5.0, 0.01, 0.01, 0.01])
        flavor = Adaptive(expon=1)
        new_groups = flavor.adjust_weights(groups, beta_init)

        # "big" group gets smaller weight (penalised less)
        # "small" group gets larger weight (penalised more)
        assert new_groups[0].weight < new_groups[1].weight

    def test_expon_effect(self):
        groups = [GroupSlice("g", 0, 2, weight=np.sqrt(2))]
        beta_init = np.array([2.0, 0.0])
        w1 = Adaptive(expon=1).adjust_weights(groups, beta_init)[0].weight
        w2 = Adaptive(expon=2).adjust_weights(groups, beta_init)[0].weight
        # Higher expon = more extreme weighting
        assert w2 < w1  # stronger inverse for larger expon

    def test_does_not_modify_original(self):
        groups = [GroupSlice("g", 0, 2, weight=1.0)]
        beta_init = np.array([1.0, 1.0])
        Adaptive().adjust_weights(groups, beta_init)
        assert groups[0].weight == 1.0  # original unchanged


class TestWarmStart:
    def test_adaptive_flavor_runs(self, sample_data):
        """End-to-end: GroupLasso with Adaptive flavor should fit."""
        X, y, exposure = sample_data
        model = SuperGLMRegressor(
            spline_features=["age"],
            n_knots=10,
            penalty=GroupLasso(lambda1=0.01, flavor=Adaptive()),
        )
        model.fit(X, y, sample_weight=exposure)
        preds = model.predict(X)
        assert preds.shape == (len(X),)
        assert np.all(preds > 0)


class TestSklearnShorthand:
    def test_string_group_lasso(self, sample_data):
        X, y, _ = sample_data
        model = SuperGLMRegressor(penalty="group_lasso", lambda1=0.01)
        model.fit(X, y)
        assert model.coef_ is not None

    def test_string_ridge(self, sample_data):
        X, y, _ = sample_data
        model = SuperGLMRegressor(penalty="ridge", lambda1=0.01)
        model.fit(X, y)
        assert model.coef_ is not None
        # Ridge never zeros — all coefficients non-zero
        assert np.all(model.coef_ != 0)

    def test_string_sparse_group_lasso(self, sample_data):
        X, y, _ = sample_data
        model = SuperGLMRegressor(
            penalty="sparse_group_lasso",
            lambda1=0.01,
            spline_features=["age"],
            n_knots=10,
        )
        model.fit(X, y)
        assert model.coef_ is not None

    def test_object_penalty(self, sample_data):
        X, y, exposure = sample_data
        model = SuperGLMRegressor(
            penalty=GroupLasso(lambda1=0.01, flavor=Adaptive()),
            spline_features=["age"],
            n_knots=10,
        )
        model.fit(X, y, sample_weight=exposure)
        assert model.coef_ is not None

    def test_string_group_elastic_net(self, sample_data):
        X, y, _ = sample_data
        model = SuperGLMRegressor(
            penalty="group_elastic_net",
            lambda1=0.01,
            spline_features=["age"],
            n_knots=10,
        )
        model.fit(X, y)
        assert model.coef_ is not None

    def test_unknown_string_raises(self):
        with pytest.raises(ValueError, match="Unknown penalty"):
            model = SuperGLMRegressor(penalty="unknown")
            model.fit(
                pd.DataFrame({"x": [1, 2, 3]}),
                np.array([1, 2, 3]),
            )
