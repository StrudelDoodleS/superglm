"""Tests for LambdaPolicy type and plumbing."""

import numpy as np
import pytest

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
