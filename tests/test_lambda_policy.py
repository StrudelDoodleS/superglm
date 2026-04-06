"""Tests for LambdaPolicy type and plumbing."""

import numpy as np
import pandas as pd
import pytest

from superglm import SuperGLM
from superglm.features.spline import Spline
from superglm.types import GroupInfo, LambdaPolicy


class TestLambdaPolicyConstruction:
    def test_estimate_factory(self):
        p = LambdaPolicy.estimate()
        assert p.mode == "estimate"
        assert p.value is None

    def test_fixed_factory(self):
        p = LambdaPolicy.fixed(0.5)
        assert p.mode == "fixed"
        assert p.value == 0.5

    def test_off_factory(self):
        p = LambdaPolicy.off()
        assert p.mode == "fixed"
        assert p.value == 0.0

    def test_frozen(self):
        p = LambdaPolicy.estimate()
        with pytest.raises(AttributeError):
            p.mode = "fixed"

    def test_fixed_requires_value(self):
        with pytest.raises(ValueError):
            LambdaPolicy(mode="fixed", value=None)

    def test_invalid_mode(self):
        with pytest.raises(ValueError):
            LambdaPolicy(mode="off")


class TestGroupInfoLambdaPolicies:
    def test_default_none(self):
        info = GroupInfo(columns=np.eye(3), n_cols=3)
        assert info.lambda_policies is None

    def test_accepts_dict(self):
        policies = {"d1": LambdaPolicy.estimate(), "d2": LambdaPolicy.fixed(1.0)}
        info = GroupInfo(columns=np.eye(3), n_cols=3, lambda_policies=policies)
        assert info.lambda_policies == policies


class TestPublicImport:
    def test_importable(self):
        from superglm import LambdaPolicy

        p = LambdaPolicy.estimate()
        assert p.mode == "estimate"


class TestSplineLambdaPolicy:
    def test_single_policy_broadcast(self):
        """Single LambdaPolicy broadcasts to all penalty components."""
        rng = np.random.default_rng(42)
        x = rng.uniform(0, 1, 200)
        spec = Spline(kind="cr", n_knots=6, m=(1, 2), lambda_policy=LambdaPolicy.fixed(0.5))
        info = spec.build(x)
        assert info.lambda_policies is not None
        assert all(p.mode == "fixed" and p.value == 0.5 for p in info.lambda_policies.values())

    def test_dict_policy_per_component(self):
        """Dict lambda_policy maps to named components."""
        rng = np.random.default_rng(42)
        x = rng.uniform(0, 1, 200)
        spec = Spline(
            kind="cr",
            n_knots=6,
            m=(1, 2),
            lambda_policy={"d1": LambdaPolicy.estimate(), "d2": LambdaPolicy.fixed(1.0)},
        )
        info = spec.build(x)
        assert info.lambda_policies["d1"].mode == "estimate"
        assert info.lambda_policies["d2"].mode == "fixed"
        assert info.lambda_policies["d2"].value == 1.0

    def test_unknown_component_key_raises(self):
        """Dict lambda_policy with unknown key raises ValueError."""
        with pytest.raises(ValueError, match="unknown"):
            Spline(
                kind="cr",
                n_knots=6,
                m=(1, 2),
                lambda_policy={"d1": LambdaPolicy.estimate(), "bogus": LambdaPolicy.fixed(1.0)},
            ).build(np.linspace(0, 1, 200))

    def test_no_policy_default(self):
        """Spline without lambda_policy leaves GroupInfo.lambda_policies as None."""
        rng = np.random.default_rng(42)
        x = rng.uniform(0, 1, 200)
        spec = Spline(kind="bs", n_knots=6)
        info = spec.build(x)
        assert info.lambda_policies is None

    def test_single_penalty_emits_component(self):
        """Single-penalty spline with lambda_policy emits explicit penalty_components."""
        rng = np.random.default_rng(42)
        x = rng.uniform(0, 1, 200)
        spec = Spline(kind="bs", n_knots=6, lambda_policy=LambdaPolicy.fixed(2.0))
        info = spec.build(x)
        assert info.lambda_policies is not None
        assert info.penalty_components is not None
        assert len(info.penalty_components) == 1
        name, omega = info.penalty_components[0]
        assert name == "wiggle"
        np.testing.assert_allclose(omega, info.penalty_matrix)
        assert info.lambda_policies["wiggle"].value == 2.0


# ---------------------------------------------------------------------------
# fit_reml() integration tests for fixed lambda policies
# ---------------------------------------------------------------------------


@pytest.fixture
def poisson_data():
    rng = np.random.default_rng(42)
    n = 500
    x = rng.uniform(0, 10, n)
    eta = 0.5 + 0.3 * np.sin(x)
    y = rng.poisson(np.exp(eta)).astype(float)
    return pd.DataFrame({"x": x}), y, np.ones(n)


class TestFitRemlLambdaPolicy:
    @pytest.mark.slow
    def test_fixed_lambda_unchanged(self, poisson_data):
        """Fixed lambda stays at its value after fit_reml."""
        X, y, w = poisson_data
        model = SuperGLM(
            family="poisson",
            features={"x": Spline(kind="cr", n_knots=8, lambda_policy=LambdaPolicy.fixed(1.0))},
        )
        model.fit_reml(X, y, sample_weight=w)
        reml_lambdas = model._reml_lambdas
        lam_val = next(iter(reml_lambdas.values()))
        assert lam_val == 1.0

    @pytest.mark.slow
    def test_mixed_policy_multi_m(self, poisson_data):
        """Mixed policy: d1 estimated, d2 fixed."""
        X, y, w = poisson_data
        model = SuperGLM(
            family="poisson",
            features={
                "x": Spline(
                    kind="cr",
                    n_knots=8,
                    m=(1, 2),
                    lambda_policy={"d1": LambdaPolicy.estimate(), "d2": LambdaPolicy.fixed(0.5)},
                ),
            },
        )
        model.fit_reml(X, y, sample_weight=w)
        reml_lambdas = model._reml_lambdas
        d2_key = [k for k in reml_lambdas if k.endswith(":d2")][0]
        assert reml_lambdas[d2_key] == 0.5

    @pytest.mark.slow
    def test_off_lambda(self, poisson_data):
        """LambdaPolicy.off() gives higher edf than estimated."""
        X, y, w = poisson_data
        model_off = SuperGLM(
            family="poisson",
            features={"x": Spline(kind="cr", n_knots=8, lambda_policy=LambdaPolicy.off())},
        )
        model_off.fit_reml(X, y, sample_weight=w)
        model_est = SuperGLM(
            family="poisson",
            features={"x": Spline(kind="cr", n_knots=8)},
        )
        model_est.fit_reml(X, y, sample_weight=w)
        assert model_off._result.effective_df > model_est._result.effective_df

    @pytest.mark.slow
    def test_no_policy_unchanged(self, poisson_data):
        """Plain Spline without lambda_policy behaves as before."""
        X, y, w = poisson_data
        model1 = SuperGLM(
            family="poisson",
            features={"x": Spline(kind="cr", n_knots=8)},
        )
        model1.fit_reml(X, y, sample_weight=w)
        model2 = SuperGLM(
            family="poisson",
            features={"x": Spline(kind="cr", n_knots=8)},
        )
        model2.fit_reml(X, y, sample_weight=w)
        for k in model1._reml_lambdas:
            np.testing.assert_allclose(model1._reml_lambdas[k], model2._reml_lambdas[k], rtol=1e-6)
        np.testing.assert_allclose(model1._result.beta, model2._result.beta, rtol=1e-6)
