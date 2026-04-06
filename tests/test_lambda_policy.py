"""Tests for LambdaPolicy type and plumbing."""

import numpy as np
import pytest

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
