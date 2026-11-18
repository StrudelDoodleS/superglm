"""Tests for LambdaPolicy type and plumbing."""

import numpy as np
import pandas as pd
import pytest

from superglm import Numeric, OrderedCategorical, SuperGLM
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

    def test_rebuild_spline_categorical_preserves_lambda_policies(self):
        """Lambda rebuild should not drop per-component policy metadata."""
        import scipy.sparse as sp

        from superglm.dm_builder import rebuild_design_matrix_with_lambdas
        from superglm.group_matrix import DesignMatrix, SplineCategoricalGroupMatrix
        from superglm.types import GroupSlice

        n = 20
        B = sp.random(n, 4, density=0.5, format="csr", random_state=82)
        mask = np.zeros(n, dtype=bool)
        mask[::3] = True
        gm = SplineCategoricalGroupMatrix(B, np.eye(4), mask)
        gm.omega = np.eye(4)
        gm.omega_components = [("wiggle", np.eye(4))]
        gm.component_types = {"wiggle": "smooth"}
        gm.lambda_policies = {"wiggle": LambdaPolicy.fixed(0.25)}
        group = GroupSlice(name="x:area[A]", start=0, end=4, feature_name="x:area")
        dm = DesignMatrix([gm], n=n, p=4)

        rebuilt = rebuild_design_matrix_with_lambdas(
            dm,
            [group],
            {"x:area[A]:wiggle": 0.25},
            sample_weight=np.ones(n),
            lambda2=1.0,
        )

        rebuilt_gm = rebuilt.group_matrices[0]
        assert rebuilt_gm.lambda_policies == gm.lambda_policies

    def test_rebuild_discretized_spline_categorical_preserves_lambda_policies(self):
        """Lambda rebuild should preserve policy metadata on compressed spline-cat groups."""
        from superglm.dm_builder import rebuild_design_matrix_with_lambdas
        from superglm.group_matrix import DesignMatrix, DiscretizedSplineCategoricalGroupMatrix
        from superglm.types import GroupSlice

        n = 24
        B_unique = np.eye(5, 4)
        bin_idx = np.arange(n) % B_unique.shape[0]
        row_idx = np.arange(0, n, 3)
        gm = DiscretizedSplineCategoricalGroupMatrix(B_unique, np.eye(4), bin_idx, row_idx)
        gm.omega = np.eye(4)
        gm.omega_components = [("wiggle", np.eye(4))]
        gm.component_types = {"wiggle": "smooth"}
        gm.lambda_policies = {"wiggle": LambdaPolicy.fixed(0.25)}
        group = GroupSlice(name="x:area[A]", start=0, end=4, feature_name="x:area")
        dm = DesignMatrix([gm], n=n, p=4)

        rebuilt = rebuild_design_matrix_with_lambdas(
            dm,
            [group],
            {"x:area[A]:wiggle": 0.25},
            sample_weight=np.ones(n),
            lambda2=1.0,
        )

        rebuilt_gm = rebuilt.group_matrices[0]
        assert isinstance(rebuilt_gm, DiscretizedSplineCategoricalGroupMatrix)
        assert rebuilt_gm.lambda_policies == gm.lambda_policies


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

    @pytest.mark.slow
    def test_off_lambda_is_exactly_zero(self, poisson_data):
        """LambdaPolicy.off() yields exactly 0.0 in _reml_lambdas, not 1e-6."""
        X, y, w = poisson_data
        model = SuperGLM(
            family="poisson",
            features={"x": Spline(kind="cr", n_knots=8, lambda_policy=LambdaPolicy.off())},
        )
        model.fit_reml(X, y, sample_weight=w)
        lam_val = next(iter(model._reml_lambdas.values()))
        assert lam_val == 0.0, f"off() lambda should be 0.0, got {lam_val}"

    @pytest.mark.parametrize("route", ["fit", "fit_path"])
    def test_fit_rejects_lambda_policy(self, poisson_data, route):
        """fit() and fit_path() refuse lambda_policy, naming the route and fit_reml()."""
        X, y, w = poisson_data
        model = SuperGLM(
            family="poisson",
            selection_penalty=0.01,
            features={"x": Spline(kind="cr", n_knots=8, lambda_policy=LambdaPolicy.fixed(1.0))},
        )
        with pytest.raises(NotImplementedError, match=rf"fit_reml\(\), not {route}\(\)"):
            getattr(model, route)(X, y, sample_weight=w)

    @pytest.mark.parametrize("route", ["fit", "fit_path"])
    def test_fit_rejects_lambda_policy_on_an_ordered_basis(self, route):
        """A policy on an OrderedCategorical's spline basis is refused, naming the feature."""
        levels = list("abcdefgh")
        codes = np.tile(np.arange(len(levels)), 15)
        basis = Spline(kind="ps", n_knots=5, lambda_policy=LambdaPolicy.fixed(100.0))
        model = SuperGLM(
            family="gaussian",
            selection_penalty=0.01,
            features={"grade": OrderedCategorical(order=levels, basis=basis)},
        )
        with pytest.raises(NotImplementedError, match=rf"'grade'.*fit_reml\(\), not {route}\(\)"):
            getattr(model, route)(pd.DataFrame({"grade": np.asarray(levels)[codes]}), np.sin(codes))

    @pytest.mark.parametrize("route", ["fit", "fit_path"])
    def test_a_structure_applied_to_a_declaration_keeps_its_policy_refused(self, route):
        """``Structure.apply`` copies the declared smoothing, which leaves a policy unresolved."""
        from superglm import Structure

        x = np.linspace(0.0, 1.0, 120)
        declared = SuperGLM(
            family="gaussian",
            selection_penalty=0.01,
            spline_penalty=0.1,
            features={"x": Spline(kind="ps", n_knots=6, lambda_policy=LambdaPolicy.fixed(1000.0))},
        )
        applied = Structure(features={}).apply(declared)
        with pytest.raises(NotImplementedError, match=rf"'x'.*fit_reml\(\), not {route}\(\)"):
            getattr(applied, route)(pd.DataFrame({"x": x}), np.sin(6.0 * x))

    def test_a_refit_keeps_a_policy_term_smoothing_beside_its_interaction(self):
        """The resolved lambda of a policy term stays on it when an interaction shares its name.

        The interaction's own keys (``"x:z:..."``) start with the term's name; the
        refit still keys the term's one penalty by the term, at the fixed 3.0.
        """
        rng = np.random.default_rng(1)
        x, z = rng.uniform(size=300), rng.uniform(size=300)
        frame = pd.DataFrame({"x": x, "z": z})
        y = np.sin(4.0 * x) + z + x * z + rng.normal(0.0, 0.1, 300)
        model = SuperGLM(
            family="gaussian",
            selection_penalty=0,
            features={
                "x": Spline(kind="ps", n_knots=5, lambda_policy=LambdaPolicy.fixed(3.0)),
                "z": Spline(kind="ps", n_knots=5),
            },
            interactions=[("x", "z")],
        )
        model.fit_reml(frame, y)
        refit = model.refit_unpenalised(frame, y, keep_smoothing=True)
        assert refit.lambda2["x"] == 3.0
        assert "x:wiggle" not in refit.lambda2
        assert any(key.startswith("x:z") for key in refit.lambda2)

    @pytest.mark.parametrize("term", ["spline", "ordered"])
    def test_refits_after_fit_reml_take_the_resolved_lambda(self, term):
        """drop1 and refit_unpenalised refit a fixed policy at the lambda fit_reml resolved.

        Their clones are fitted at the source's smoothing, so they carry no policy, and
        neither does an unfitted clone of the model a refit returns. Each refit is the
        policy-free model declared at the same scalar lambda (3.0, or 0.0 when the
        smoothing is dropped): one Gaussian solve each, equal to rounding, which 1e-10
        bounds on these values near 2.
        """
        levels = list("abcdefgh")
        rng = np.random.default_rng(0)
        codes, z, x = rng.integers(0, 8, 300), rng.uniform(size=300), rng.uniform(size=300)
        column = x if term == "spline" else np.asarray(levels)[codes]
        frame = pd.DataFrame({"s": column, "z": z})
        y = np.sin(codes) + z + np.sin(4.0 * x) + rng.normal(0.0, 0.1, 300)

        def features(policy=None):
            if term == "spline":
                smooth = Spline(kind="ps", n_knots=6, lambda_policy=policy)
            else:
                basis = Spline(kind="ps", n_knots=4, lambda_policy=policy)
                smooth = OrderedCategorical(order=levels, basis=basis)
            return {"s": smooth, "z": Numeric()}

        model = SuperGLM(
            family="gaussian", selection_penalty=0, features=features(LambdaPolicy.fixed(3.0))
        )
        model.fit_reml(frame, y)

        def declared(lam):
            return SuperGLM(
                family="gaussian", selection_penalty=0, spline_penalty=lam, features=features()
            ).fit(frame, y)

        for keep, lam in ((True, 3.0), (False, 0.0)):
            refit = model.refit_unpenalised(frame, y, keep_smoothing=keep)
            expected = declared(lam).predict(frame)
            np.testing.assert_allclose(refit.predict(frame), expected, rtol=0, atol=1e-10)
            again = refit.clone_unfitted().fit(frame, y)
            np.testing.assert_allclose(again.predict(frame), expected, rtol=0, atol=1e-10)
        np.testing.assert_allclose(
            model.drop1(frame, y).set_index("feature")["deviance_reduced"].sort_index(),
            declared(3.0).drop1(frame, y).set_index("feature")["deviance_reduced"].sort_index(),
            rtol=1e-10,
        )
