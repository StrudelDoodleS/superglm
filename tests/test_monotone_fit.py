"""End-to-end tests for monotone-constrained model fitting.

Tests that SuperGLM with monotone BSplineSmooth / CubicRegressionSpline terms
produces actually-monotone predictions via the constrained QP solver path.
"""

import warnings
from contextlib import nullcontext

import numpy as np
import pandas as pd
import pytest

from superglm import Constraint, SuperGLM
from superglm.families import Gaussian
from superglm.features.spline import BSplineSmooth, CubicRegressionSpline, PSpline
from superglm.types import LambdaPolicy


class TestMonotoneFitBSplineSmooth:
    """BSplineSmooth with monotone_mode='fit' produces monotone predictions."""

    @pytest.fixture
    def monotone_data(self):
        """Data with a clearly increasing relationship."""
        rng = np.random.default_rng(42)
        n = 500
        x = np.sort(rng.uniform(0, 1, n))
        # True function: monotone increasing sigmoid
        y_true = 1 / (1 + np.exp(-10 * (x - 0.5)))
        y = y_true + rng.normal(0, 0.1, n)
        return x, y

    @pytest.mark.slow
    def test_predictions_are_monotone(self, monotone_data):
        x, y = monotone_data

        df = pd.DataFrame({"x": x, "y": y})
        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            features={
                "x": BSplineSmooth(
                    n_knots=10,
                    constraint=Constraint.fit.increasing,
                ),
            },
        )
        model.fit(df[["x"]], df["y"])

        x_grid = np.linspace(0, 1, 200)
        df_grid = pd.DataFrame({"x": x_grid})
        pred = model.predict(df_grid)
        # Predictions must be monotone increasing
        diffs = np.diff(pred)
        assert np.all(diffs >= -1e-8), f"Predictions not monotone: min diff = {diffs.min():.2e}"

    @pytest.mark.slow
    def test_unconstrained_unchanged(self, monotone_data):
        """monotone=None does not enter the constrained QP path."""
        x, y = monotone_data

        df = pd.DataFrame({"x": x, "y": y})
        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            features={"x": BSplineSmooth(n_knots=10)},
        )
        model.fit(df[["x"]], df["y"])
        assert model._result.n_iter > 0
        assert model._result.deviance < 100

    @pytest.mark.slow
    def test_weighted_fit(self, monotone_data):
        x, y = monotone_data

        weights = np.ones(len(x))
        weights[:100] = 2.0

        df = pd.DataFrame({"x": x, "y": y})
        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            features={
                "x": BSplineSmooth(
                    n_knots=10,
                    constraint=Constraint.fit.increasing,
                ),
            },
        )
        model.fit(df[["x"]], df["y"], sample_weight=weights)

        x_grid = np.linspace(0, 1, 200)
        df_grid = pd.DataFrame({"x": x_grid})
        pred = model.predict(df_grid)
        assert np.all(np.diff(pred) >= -1e-8)

    @pytest.mark.slow
    def test_decreasing(self):
        rng = np.random.default_rng(42)
        n = 300
        x = np.sort(rng.uniform(0, 1, n))
        y = np.exp(-3 * x) + rng.normal(0, 0.05, n)

        df = pd.DataFrame({"x": x, "y": y})
        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            features={
                "x": BSplineSmooth(
                    n_knots=8,
                    constraint=Constraint.fit.decreasing,
                ),
            },
        )
        model.fit(df[["x"]], df["y"])

        x_grid = np.linspace(0, 1, 200)
        df_grid = pd.DataFrame({"x": x_grid})
        pred = model.predict(df_grid)
        assert np.all(np.diff(pred) <= 1e-8)


class TestMonotoneFitCRS:
    """CubicRegressionSpline with monotone_mode='fit'."""

    @pytest.mark.slow
    def test_predictions_are_monotone(self):
        rng = np.random.default_rng(42)
        n = 500
        x = np.sort(rng.uniform(0, 1, n))
        y = 2 * x + rng.normal(0, 0.2, n)

        df = pd.DataFrame({"x": x, "y": y})
        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            features={
                "x": CubicRegressionSpline(
                    n_knots=10,
                    constraint=Constraint.fit.increasing,
                ),
            },
        )
        model.fit(df[["x"]], df["y"])

        x_grid = np.linspace(0, 1, 200)
        df_grid = pd.DataFrame({"x": x_grid})
        pred = model.predict(df_grid)
        assert np.all(np.diff(pred) >= -1e-8)


class TestMonotoneMixedModel:
    """Model with both monotone and unconstrained terms."""

    @pytest.mark.slow
    def test_mixed_terms(self):
        rng = np.random.default_rng(42)
        n = 500
        x1 = np.sort(rng.uniform(0, 1, n))
        x2 = rng.uniform(0, 1, n)
        y = 2 * x1 + np.sin(2 * np.pi * x2) + rng.normal(0, 0.2, n)

        df = pd.DataFrame({"x1": x1, "x2": x2, "y": y})
        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            features={
                "x1": BSplineSmooth(
                    n_knots=8,
                    constraint=Constraint.fit.increasing,
                ),
                "x2": BSplineSmooth(n_knots=8),  # unconstrained
            },
        )
        model.fit(df[["x1", "x2"]], df["y"])

        # x1 predictions should be monotone when x2 is held fixed
        x_grid = np.linspace(0, 1, 200)
        df_grid = pd.DataFrame(
            {
                "x1": x_grid,
                "x2": np.full(200, 0.5),
            }
        )
        pred = model.predict(df_grid)
        assert np.all(np.diff(pred) >= -1e-8)


class TestMonotoneRegression:
    """Regression tests: unconstrained behavior is unchanged."""

    @pytest.mark.slow
    def test_no_performance_regression(self):
        """Unconstrained model runs at the same speed (no QP overhead)."""
        rng = np.random.default_rng(42)
        n = 1000
        x = rng.uniform(0, 1, n)
        y = np.sin(2 * np.pi * x) + rng.normal(0, 0.3, n)

        df = pd.DataFrame({"x": x, "y": y})
        model = SuperGLM(
            family=Gaussian(),
            features={"x": BSplineSmooth(n_knots=10)},
        )
        model.fit(df[["x"]], df["y"])
        assert model._result.converged


class TestSCOPPenaltyInEDF:
    """SCOP penalty contributes to EDF and information criteria."""

    @pytest.mark.slow
    def test_group_edf_changes_with_lambda(self):
        """Per-group EDF should also respond to penalty changes."""
        rng = np.random.default_rng(42)
        n = 300
        x = np.sort(rng.uniform(0, 1, n))
        y = 2 * x + rng.normal(0, 0.2, n)
        df = pd.DataFrame({"x": x, "y": y})

        model_low = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            spline_penalty=0.01,
            features={
                "x": PSpline(n_knots=10, constraint=Constraint.fit.increasing, shape_engine="scop")
            },
        )
        model_low.fit(df[["x"]], df["y"])

        model_high = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            spline_penalty=100.0,
            features={
                "x": PSpline(n_knots=10, constraint=Constraint.fit.increasing, shape_engine="scop")
            },
        )
        model_high.fit(df[["x"]], df["y"])

        # Group EDF should also respond (not pinned)
        edf_low = sum(model_low._group_edf.values())
        edf_high = sum(model_high._group_edf.values())
        assert edf_high < edf_low, (
            f"Group EDF should decrease with higher penalty: low={edf_low:.2f}, high={edf_high:.2f}"
        )

    @pytest.mark.slow
    def test_edf_changes_with_lambda(self):
        """Increasing lambda should reduce effective degrees of freedom."""
        rng = np.random.default_rng(42)
        n = 300
        x = np.sort(rng.uniform(0, 1, n))
        y = 2 * x + rng.normal(0, 0.2, n)
        df = pd.DataFrame({"x": x, "y": y})

        model_low = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            spline_penalty=0.01,
            features={
                "x": PSpline(n_knots=10, constraint=Constraint.fit.increasing, shape_engine="scop")
            },
        )
        model_low.fit(df[["x"]], df["y"])

        model_high = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            spline_penalty=100.0,
            features={
                "x": PSpline(n_knots=10, constraint=Constraint.fit.increasing, shape_engine="scop")
            },
        )
        model_high.fit(df[["x"]], df["y"])

        # Higher penalty should give lower EDF
        assert model_high._result.effective_df < model_low._result.effective_df


class TestMonotoneUnsupportedCombinations:
    """Unsupported combinations raise NotImplementedError."""

    def test_monotone_with_selection_penalty_raises(self):
        rng = np.random.default_rng(42)
        n = 200
        x = rng.uniform(0, 1, n)
        y = 2 * x + rng.normal(0, 0.2, n)
        df = pd.DataFrame({"x": x, "y": y})

        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0.1,
            features={
                "x": BSplineSmooth(n_knots=8, constraint=Constraint.fit.increasing),
            },
        )
        with pytest.raises(NotImplementedError, match="selection_penalty"):
            model.fit(df[["x"]], df["y"])

    def test_monotone_with_select_true_raises(self):
        s = BSplineSmooth(n_knots=8, constraint=Constraint.fit.increasing, select=True)
        x = np.linspace(0, 1, 200)
        with pytest.raises(NotImplementedError, match="select=True"):
            s.build(x)

    def test_monotone_fit_reml_qp_passthrough(self):
        """QP monotone in fit_reml works via passthrough heuristic."""
        rng = np.random.default_rng(42)
        n = 200
        x = np.sort(rng.uniform(0, 1, n))
        y = 2 * x + rng.normal(0, 0.2, n)
        df = pd.DataFrame({"x": x})

        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            features={
                "x": BSplineSmooth(n_knots=8, constraint=Constraint.fit.increasing),
            },
        )
        model.fit_reml(df[["x"]], y)
        assert model._result.converged

    def test_scop_monotone_with_discrete_works(self):
        """discrete=True + SCOP monotone_mode='fit' is now supported."""
        rng = np.random.default_rng(42)
        n = 200
        x = rng.uniform(0, 1, n)
        y = 2 * x + rng.normal(0, 0.2, n)
        df = pd.DataFrame({"x": x, "y": y})

        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            discrete=True,
            features={
                "x": PSpline(n_knots=8, constraint=Constraint.fit.increasing, shape_engine="scop"),
            },
        )
        model.fit(df[["x"]], df["y"])
        x_grid = np.linspace(0, 1, 50)
        pred = model.predict(pd.DataFrame({"x": x_grid}))
        assert np.all(np.diff(pred) >= -1e-8)


# ── PSpline SCOP engine tests ─────────────────────────────────────────────────


class TestMonotoneFitPSpline:
    """PSpline with monotone_mode='fit' uses SCOP engine."""

    @pytest.mark.slow
    def test_predictions_are_monotone(self):
        rng = np.random.default_rng(42)
        n = 500
        x = np.sort(rng.uniform(0, 1, n))
        y = 1 / (1 + np.exp(-10 * (x - 0.5))) + rng.normal(0, 0.1, n)
        df = pd.DataFrame({"x": x, "y": y})
        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            features={"x": PSpline(n_knots=10, constraint=Constraint.fit.increasing)},
        )
        model.fit(df[["x"]], df["y"])
        x_grid = np.linspace(0, 1, 200)
        pred = model.predict(pd.DataFrame({"x": x_grid}))
        assert np.all(np.diff(pred) >= -1e-8), f"min diff = {np.diff(pred).min():.2e}"

    @pytest.mark.slow
    def test_decreasing(self):
        rng = np.random.default_rng(42)
        n = 300
        x = np.sort(rng.uniform(0, 1, n))
        y = np.exp(-3 * x) + rng.normal(0, 0.05, n)
        df = pd.DataFrame({"x": x, "y": y})
        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            features={"x": PSpline(n_knots=8, constraint=Constraint.fit.decreasing)},
        )
        model.fit(df[["x"]], df["y"])
        x_grid = np.linspace(0, 1, 200)
        pred = model.predict(pd.DataFrame({"x": x_grid}))
        assert np.all(np.diff(pred) <= 1e-8)

    @pytest.mark.slow
    def test_mixed_scop_and_unconstrained(self):
        rng = np.random.default_rng(42)
        n = 500
        x1 = np.sort(rng.uniform(0, 1, n))
        x2 = rng.uniform(0, 1, n)
        y = 2 * x1 + np.sin(2 * np.pi * x2) + rng.normal(0, 0.2, n)
        df = pd.DataFrame({"x1": x1, "x2": x2, "y": y})
        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            features={
                "x1": PSpline(n_knots=8, constraint=Constraint.fit.increasing, shape_engine="scop"),
                "x2": PSpline(n_knots=8),
            },
        )
        model.fit(df[["x1", "x2"]], df["y"])
        x_grid = np.linspace(0, 1, 200)
        pred = model.predict(pd.DataFrame({"x1": x_grid, "x2": np.full(200, 0.5)}))
        assert np.all(np.diff(pred) >= -1e-8)

    @pytest.mark.slow
    def test_unconstrained_pspline_unchanged(self):
        rng = np.random.default_rng(42)
        n = 300
        x = rng.uniform(0, 1, n)
        y = np.sin(2 * np.pi * x) + rng.normal(0, 0.2, n)
        df = pd.DataFrame({"x": x, "y": y})
        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            features={"x": PSpline(n_knots=8)},
        )
        model.fit(df[["x"]], df["y"])
        assert model._result.converged

    @pytest.mark.parametrize("fit_method", ["fit", "fit_reml"])
    @pytest.mark.parametrize("ordered", [False, True], ids=["spline", "ordered"])
    def test_kindless_constraint_beside_scop_builds_ps(self, fit_method, ordered):
        """A constrained spline with no kind was a SCOP ``ps`` term on 0.39. As a
        ``cr`` QP term beside a ``ps`` one on SCOP, 0.40.0 refused the model; it builds as
        ``ps`` on SCOP again, bit for bit as the explicit declaration. Fails on e7b0df17."""
        from superglm import OrderedCategorical, Spline

        rng = np.random.default_rng(42)
        df = pd.DataFrame({"x1": rng.uniform(0, 1, 200), "x2": rng.uniform(0, 1, 200)})
        df["band"] = np.array(list("ABCDE"))[(df["x2"].to_numpy() * 5).astype(int)]
        y = df["x1"] + df["x2"] + rng.normal(0, 0.1, 200)
        inc = Constraint.fit.increasing
        scop = (
            {
                "band": OrderedCategorical(
                    order=list("ABCDE"),
                    basis=Spline(kind="ps", k=5, constraint=inc, shape_engine="scop"),
                )
            }
            if ordered
            else {"x2": Spline(kind="ps", k=8, constraint=inc, shape_engine="scop")}
        )
        X = df[["x1", *scop]]
        fits = []
        for x1, warned in (
            (Spline(constraint=inc), True),
            (Spline(kind="ps", constraint=inc, shape_engine="scop"), False),
        ):
            model = SuperGLM(family=Gaussian(), selection_penalty=0, features={"x1": x1, **scop})
            with pytest.warns(UserWarning, match="x1.*names no kind") if warned else nullcontext():
                getattr(model, fit_method)(X, y)
            fits.append(model)
        assert isinstance(fits[0]._specs["x1"], PSpline)
        np.testing.assert_array_equal(fits[0].result.beta, fits[1].result.beta)
        np.testing.assert_array_equal(fits[0].predict(X), fits[1].predict(X))

    @pytest.mark.parametrize("fit_method", ["fit", "fit_reml"])
    def test_kindless_ordered_categorical_beside_scop_keeps_its_levels(self, fit_method):
        """An ordered categorical whose basis names no kind, beside a SCOP term, takes the
        ``ps`` basis on SCOP and keeps its level map: string levels fit, bit for bit as the
        explicit ``kind='ps', shape_engine='scop'`` basis (and as 0.39). Fails on eedfa48e,
        which replaced the wrapper by a bare P-spline on the band column."""
        from superglm import OrderedCategorical, Spline

        rng = np.random.default_rng(42)
        df = pd.DataFrame({"x2": rng.uniform(0, 1, 200)})
        df["band"] = np.array(list("ABCDE"))[(df["x2"].to_numpy() * 5).astype(int)]
        y = df["x2"] + rng.normal(0, 0.1, 200)
        inc = Constraint.fit.increasing
        fits = []
        for kind, warned in ((None, True), ("ps", False)):
            features = {
                "x2": Spline(kind="ps", k=8, constraint=inc, shape_engine="scop"),
                "band": OrderedCategorical(
                    order=list("ABCDE"),
                    basis=Spline(
                        kind=kind, k=5, constraint=inc, shape_engine="scop" if kind else "qp"
                    ),
                ),
            }
            model = SuperGLM(family=Gaussian(), selection_penalty=0, features=features)
            with (
                pytest.warns(UserWarning, match="band.*names no kind") if warned else nullcontext()
            ):
                getattr(model, fit_method)(df, y)
            fits.append(model)
        assert isinstance(fits[0]._specs["band"], OrderedCategorical)
        assert isinstance(fits[0]._specs["band"]._spline, PSpline)
        np.testing.assert_array_equal(fits[0].result.beta, fits[1].result.beta)
        np.testing.assert_array_equal(fits[0].predict(df), fits[1].predict(df))

    @pytest.mark.parametrize("fit_method", ["fit", "fit_reml"])
    def test_kindless_constraint_without_scop_stays_cr(self, fit_method, recwarn):
        """No SCOP term, no rewrite: a kindless ``k=4`` monotone spline is the QP ``cr``
        term it was on 0.40, and nothing is built as ``ps`` (which needs ``k >= 5``).
        Fails on eedfa48e with ``k=4 is too small for kind='ps'``."""
        from superglm import Spline

        rng = np.random.default_rng(7)
        df = pd.DataFrame({"x": rng.uniform(0, 1, 200)})
        y = df["x"] + rng.normal(0, 0.1, 200)
        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            features={"x": Spline(k=4, constraint=Constraint.fit.increasing)},
        )
        getattr(model, fit_method)(df, y)
        assert isinstance(model._specs["x"], CubicRegressionSpline)
        assert not [w for w in recwarn if "names no kind" in str(w.message)]
        assert np.all(np.diff(model.predict(df.sort_values("x"))) >= -1e-12)

    @staticmethod
    def _kindless_beside_scop(x1):
        """A Gaussian fit of ``x1`` beside a SCOP ``ps`` term, with its data."""
        from superglm import Spline

        rng = np.random.default_rng(42)
        df = pd.DataFrame({"x1": rng.uniform(0, 1, 200), "x2": rng.uniform(0, 1, 200)})
        y = (df["x1"] + df["x2"] + rng.normal(0, 0.1, 200)).to_numpy()
        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            features={
                "x1": x1,
                "x2": Spline(
                    kind="ps", k=8, constraint=Constraint.fit.increasing, shape_engine="scop"
                ),
            },
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(df, y)
        return model, df, y

    def test_an_editor_knot_change_on_a_kindless_term_beside_scop_reaches_the_refit(self):
        """The editor edits the ``ps`` the fit used, as an explicit spline: a change to the
        knots refits those knots and keeps the mark, with no kindless rewrite left to apply.
        Fails on df503618, where the replacement was rebuilt from the arguments the spline
        was first declared with (10 knots, no mark)."""
        from superglm import Spline
        from superglm.features.rebuild import (
            EDITOR_KNOTS_ATTRIBUTE,
            declared_spline,
            respaced_spline,
        )

        inc = Constraint.fit.increasing
        first, df, y = self._kindless_beside_scop(Spline(constraint=inc))
        edited = respaced_spline(declared_spline(first, "x1"), n_knots=4)
        setattr(edited, EDITOR_KNOTS_ATTRIBUTE, True)
        refit, *_ = self._kindless_beside_scop(edited)
        assert isinstance(refit._specs["x1"], PSpline)
        assert refit._specs["x1"].n_knots == 4
        assert getattr(refit._specs["x1"], EDITOR_KNOTS_ATTRIBUTE, False)

    def test_an_editor_rule_edit_of_a_kindless_k_term_beside_scop_refits_the_asked_count(self):
        """``k=8`` is 6 knots as ``cr`` and 4 as ``ps``; the knots tool shows the 4. A rule
        edit to 6 evenly spaced knots refits a ``ps`` with 6. Fails on 4ff7a63d: 4."""
        from superglm import Spline
        from superglm.editor import EditorSession

        inc = Constraint.fit.increasing
        opened, df, y = self._kindless_beside_scop(Spline(k=8, constraint=inc))
        assert opened._specs["x1"].n_knots == 4
        session = EditorSession.from_model(opened, train_data=(df, y, np.ones(200)))
        session.stage_structural("knots", "x1", {"count": 6, "strategy": "uniform"})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            session.refit_pending()
        assert isinstance(session.model._specs["x1"], PSpline)
        assert session.model._specs["x1"].n_knots == 6

    def test_a_waiting_basis_change_beside_scop_is_not_rewritten_to_ps_by_a_knot_reset(self):
        """A basis change to ``bs`` waits, then a knot edit and a reset. The refit is the ``bs``
        the user asked for, which beside SCOP is refused as an explicit ``bs`` is; it is not
        quietly fitted as ``ps``. Fails on 4ff7a63d: the reset carried the kindless arguments
        onto the ``bs`` and the refit fitted ``ps``."""
        from superglm import Spline
        from superglm.editor import EditorSession

        inc = Constraint.fit.increasing
        opened, df, y = self._kindless_beside_scop(Spline(k=8, constraint=inc))
        session = EditorSession.from_model(opened, train_data=(df, y, np.ones(200)))
        session.stage_structural("basis", "x1", {"kind": "bs"})
        session.stage_structural("knots", "x1", {"count": 6, "strategy": "uniform"})
        session.stage_structural("knots", "x1", {"reset": True})
        draft = next(step.draft_spec for step in session.pending if step.term == "x1")
        assert type(draft).__name__ == "BSplineSmooth"
        assert getattr(draft, "_kindless_args", None) is None
        with pytest.raises(NotImplementedError, match="SCOP.*QP"):
            session.refit_pending()

    def test_an_editor_reset_of_a_kindless_k_term_beside_scop_refits_the_original(self):
        """``k=8`` is 6 knots as ``cr`` and 4 as ``ps``. The opened fit is the ``ps``; an edit
        to 5 knots and a reset refit it bit for bit, and so does an explicit ``ps`` of ``k=8``."""
        from superglm import Spline
        from superglm.editor.knots import _reset_basis
        from superglm.features.rebuild import declared_spline, respaced_spline

        inc = Constraint.fit.increasing
        opened, *_ = self._kindless_beside_scop(Spline(k=8, constraint=inc))
        explicit, *_ = self._kindless_beside_scop(
            Spline(kind="ps", k=8, constraint=inc, shape_engine="scop")
        )
        assert opened._specs["x1"].n_knots == 4
        edited, *_ = self._kindless_beside_scop(
            respaced_spline(declared_spline(opened, "x1"), n_knots=5)
        )
        assert edited._specs["x1"].n_knots == 5
        reset, _ = _reset_basis(opened, "x1", declared_spline(edited, "x1"))
        refit, *_ = self._kindless_beside_scop(reset)
        assert refit._specs["x1"].n_knots == 4
        np.testing.assert_array_equal(refit.result.beta, opened.result.beta)
        np.testing.assert_array_equal(refit.result.beta, explicit.result.beta)

    @pytest.mark.parametrize("fit_method", ["fit", "fit_reml"])
    def test_a_kindless_ordered_basis_clamped_to_its_levels_warns_of_the_clamp_once(
        self, fit_method
    ):
        """The wrapper warned of the clamp when it was declared; installing the ``ps`` basis
        beside a SCOP term must not warn of it again. Fails on adf356cd: two at the fit."""
        from superglm import OrderedCategorical, Spline

        rng = np.random.default_rng(42)
        df = pd.DataFrame({"x2": rng.uniform(0, 1, 200)})
        df["band"] = np.array(list("ABCDE"))[(df["x2"].to_numpy() * 5).astype(int)]
        y = df["x2"] + rng.normal(0, 0.1, 200)
        inc = Constraint.fit.increasing
        with pytest.warns(UserWarning, match="n_knots exceeds the level count"):
            band = OrderedCategorical(order=list("ABCDE"), basis=Spline(constraint=inc))
        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            features={
                "x2": Spline(kind="ps", k=8, constraint=inc, shape_engine="scop"),
                "band": band,
            },
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            getattr(model, fit_method)(df, y)
        texts = [str(w.message) for w in caught]
        assert [t for t in texts if "names no kind" in t]
        assert not [t for t in texts if "exceeds the level count" in t]

    @pytest.mark.parametrize("fit_method", ["fit", "fit_reml"])
    def test_a_kindless_spline_ps_cannot_build_beside_scop_keeps_the_scop_qp_refusal(
        self, fit_method
    ):
        """``k=4`` is a ``cr`` the QP takes and a ``ps`` cannot be (``k >= 5``). Beside a
        SCOP term 0.40 refused the model with the SCOP + QP error; so does this, rather
        than the ``ps`` construction error. Fails on df503618 with ``ValueError``."""
        from superglm import Spline

        rng = np.random.default_rng(7)
        df = pd.DataFrame({"x1": rng.uniform(0, 1, 200), "x2": rng.uniform(0, 1, 200)})
        y = df["x1"] + df["x2"] + rng.normal(0, 0.1, 200)
        inc = Constraint.fit.increasing
        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            features={
                "x1": Spline(k=4, constraint=inc),
                "x2": Spline(kind="ps", k=8, constraint=inc, shape_engine="scop"),
            },
        )
        with pytest.raises(NotImplementedError, match="SCOP.*QP"):
            getattr(model, fit_method)(df, y)

    @pytest.mark.parametrize("fit_method", ["fit", "fit_reml"])
    def test_scop_plus_qp_raises(self, fit_method):
        """Every fit entry point rejects mixed SCOP and QP engines."""
        rng = np.random.default_rng(42)
        n = 200
        x1 = rng.uniform(0, 1, n)
        x2 = rng.uniform(0, 1, n)
        y = x1 + x2 + rng.normal(0, 0.1, n)
        df = pd.DataFrame({"x1": x1, "x2": x2, "y": y})
        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            features={
                "x1": PSpline(n_knots=5, constraint=Constraint.fit.increasing, shape_engine="scop"),
                "x2": BSplineSmooth(n_knots=5, constraint=Constraint.fit.increasing),
            },
        )
        with pytest.raises(NotImplementedError, match="SCOP.*QP"):
            getattr(model, fit_method)(df[["x1", "x2"]], df["y"])


# ── fit_reml with fixed lambda tests ────────────────────────────────────────────


class TestMonotoneFixedLambdaREML:
    """fit_reml() works with monotone terms when all lambdas are fixed."""

    @pytest.mark.slow
    def test_fit_reml_with_fixed_lambda_policy_qp(self):
        """BSplineSmooth monotone with fixed lambda_policy works in fit_reml."""
        rng = np.random.default_rng(42)
        n = 500
        x = np.sort(rng.uniform(0, 1, n))
        y = 1 / (1 + np.exp(-10 * (x - 0.5))) + rng.normal(0, 0.1, n)
        df = pd.DataFrame({"x": x, "y": y})

        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            features={
                "x": BSplineSmooth(
                    n_knots=8,
                    constraint=Constraint.fit.increasing,
                    lambda_policy=LambdaPolicy(mode="fixed", value=1.0),
                ),
            },
        )
        model.fit_reml(df[["x"]], df["y"])

        # Predictions must be monotone increasing
        x_grid = np.linspace(0, 1, 200)
        pred = model.predict(pd.DataFrame({"x": x_grid}))
        assert np.all(np.diff(pred) >= -1e-8), f"min diff = {np.diff(pred).min():.2e}"

        # Model should have converged
        assert model._result.converged

        # REML lambdas should be set
        assert model._reml_lambdas is not None

    @pytest.mark.slow
    def test_fit_reml_with_fixed_lambda_policy_scop(self):
        """PSpline monotone with fixed lambda_policy works in fit_reml."""
        rng = np.random.default_rng(42)
        n = 500
        x = np.sort(rng.uniform(0, 1, n))
        y = 1 / (1 + np.exp(-10 * (x - 0.5))) + rng.normal(0, 0.1, n)
        df = pd.DataFrame({"x": x, "y": y})

        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            features={
                "x": PSpline(
                    n_knots=8,
                    constraint=Constraint.fit.increasing,
                    lambda_policy=LambdaPolicy(mode="fixed", value=1.0),
                    shape_engine="scop",
                ),
            },
        )
        model.fit_reml(df[["x"]], df["y"])

        # Predictions must be monotone increasing
        x_grid = np.linspace(0, 1, 200)
        pred = model.predict(pd.DataFrame({"x": x_grid}))
        assert np.all(np.diff(pred) >= -1e-8), f"min diff = {np.diff(pred).min():.2e}"

        assert model._result.converged
        assert model._reml_lambdas is not None

    def test_fit_reml_without_fixed_lambdas_works_qp(self):
        """fit_reml works for QP monotone via passthrough heuristic."""
        rng = np.random.default_rng(42)
        n = 200
        x = np.sort(rng.uniform(0, 1, n))
        y = 2 * x + rng.normal(0, 0.2, n)
        df = pd.DataFrame({"x": x})

        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            features={
                "x": BSplineSmooth(
                    n_knots=8,
                    constraint=Constraint.fit.increasing,
                ),
            },
        )
        model.fit_reml(df[["x"]], y)
        assert model._result.converged

    @pytest.mark.slow
    def test_fit_reml_without_fixed_lambdas_works_scop(self):
        """fit_reml works for SCOP monotone with auto lambda (Phase 5a)."""
        rng = np.random.default_rng(42)
        n = 300
        x = np.sort(rng.uniform(0, 1, n))
        y = 2 * x + rng.normal(0, 0.2, n)
        df = pd.DataFrame({"x": x})

        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            discrete=True,
            features={
                "x": PSpline(n_knots=8, constraint=Constraint.fit.increasing, shape_engine="scop"),
            },
        )
        model.fit_reml(df[["x"]], y)
        assert model._result.converged
        assert model._reml_lambdas is not None

    @pytest.mark.slow
    def test_fit_reml_unchanged_without_monotone(self):
        """Normal REML still works without monotone terms."""
        rng = np.random.default_rng(42)
        n = 500
        x = rng.uniform(0, 1, n)
        y = np.sin(2 * np.pi * x) + rng.normal(0, 0.3, n)
        df = pd.DataFrame({"x": x, "y": y})

        model = SuperGLM(
            family=Gaussian(),
            features={"x": BSplineSmooth(n_knots=10)},
        )
        model.fit_reml(df[["x"]], df["y"])
        assert model._result.converged


# ── discrete + monotone + fit_reml intersection tests ─────────────────────────


class TestDiscreteMonotoneREML:
    """The full intersection: discrete=True + monotone + fit_reml(fixed lambda).

    Tests at 250k rows to exercise the discretization performance path
    at meaningful scale.
    """

    @pytest.mark.slow
    def test_discrete_qp_monotone_fit_reml(self):
        """BSplineSmooth: discrete + monotone + fit_reml with fixed lambda."""
        rng = np.random.default_rng(42)
        n = 250_000
        x = np.sort(rng.uniform(0, 1, n))
        y = 2 * x + rng.normal(0, 0.3, n)
        df = pd.DataFrame({"x": x, "y": y})

        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            discrete=True,
            features={
                "x": BSplineSmooth(
                    n_knots=10,
                    constraint=Constraint.fit.increasing,
                    lambda_policy=LambdaPolicy(mode="fixed", value=1.0),
                ),
            },
        )
        model.fit_reml(df[["x"]], df["y"])

        x_grid = np.linspace(0, 1, 200)
        pred = model.predict(pd.DataFrame({"x": x_grid}))
        assert np.all(np.diff(pred) >= -1e-8), f"min diff = {np.diff(pred).min():.2e}"
        assert model._result.converged
        assert model._reml_lambdas is not None

        # Verify the term is actually discretized
        from superglm.group_matrix import DiscretizedSSPGroupMatrix

        gm = model._dm.group_matrices[0]
        assert isinstance(gm, DiscretizedSSPGroupMatrix), (
            f"Expected DiscretizedSSPGroupMatrix, got {type(gm).__name__}"
        )

    @pytest.mark.slow
    def test_discrete_scop_monotone_fit_reml(self):
        """PSpline: discrete + monotone + fit_reml with fixed lambda."""
        rng = np.random.default_rng(42)
        n = 250_000
        x = np.sort(rng.uniform(0, 1, n))
        y = 2 * x + rng.normal(0, 0.3, n)
        df = pd.DataFrame({"x": x, "y": y})

        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            discrete=True,
            features={
                "x": PSpline(
                    n_knots=10,
                    constraint=Constraint.fit.increasing,
                    lambda_policy=LambdaPolicy(mode="fixed", value=1.0),
                    shape_engine="scop",
                ),
            },
        )
        model.fit_reml(df[["x"]], df["y"])

        x_grid = np.linspace(0, 1, 200)
        pred = model.predict(pd.DataFrame({"x": x_grid}))
        assert np.all(np.diff(pred) >= -1e-8), f"min diff = {np.diff(pred).min():.2e}"
        assert model._result.converged
        assert model._reml_lambdas is not None

        # Verify the term used the discretized SCOP path
        from superglm.group_matrix import DiscretizedSCOPGroupMatrix

        gm = model._dm.group_matrices[0]
        assert isinstance(gm, DiscretizedSCOPGroupMatrix), (
            f"Expected DiscretizedSCOPGroupMatrix, got {type(gm).__name__}"
        )


# ── summary() monotone engine display tests ──────────────────────────────────────


class TestSummaryMonotoneEngine:
    """summary() shows monotone engine type (qp/scop) alongside direction."""

    @pytest.mark.slow
    def test_summary_shows_engine_for_qp(self):
        """BSplineSmooth monotone summary shows 'qp' engine."""
        rng = np.random.default_rng(42)
        n = 300
        x = np.sort(rng.uniform(0, 1, n))
        y = 2 * x + rng.normal(0, 0.2, n)
        df = pd.DataFrame({"x": x, "y": y})

        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            features={
                "x": BSplineSmooth(
                    n_knots=8,
                    constraint=Constraint.fit.increasing,
                ),
            },
        )
        model.fit(df[["x"]], df["y"])
        summary_str = str(model.summary())

        assert "qp" in summary_str.lower()
        assert "mono=increasing (qp)" in summary_str

    @pytest.mark.slow
    def test_summary_shows_engine_for_scop(self):
        """PSpline monotone summary shows 'scop' engine."""
        rng = np.random.default_rng(42)
        n = 300
        x = np.sort(rng.uniform(0, 1, n))
        y = 2 * x + rng.normal(0, 0.2, n)
        df = pd.DataFrame({"x": x, "y": y})

        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            features={
                "x": PSpline(
                    n_knots=8,
                    constraint=Constraint.fit.increasing,
                    shape_engine="scop",
                ),
            },
        )
        model.fit(df[["x"]], df["y"])
        summary_str = str(model.summary())

        assert "scop" in summary_str.lower()
        assert "mono=increasing (scop)" in summary_str


class TestDiscreteQPMonotone:
    """discrete=True with QP monotone (BSplineSmooth/CRS)."""

    @pytest.mark.slow
    def test_discrete_bsplinesmooth_monotone(self):
        rng = np.random.default_rng(42)
        n = 1000
        x = np.sort(rng.uniform(0, 1, n))
        y = 2 * x + rng.normal(0, 0.2, n)
        df = pd.DataFrame({"x": x, "y": y})
        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            discrete=True,
            features={"x": BSplineSmooth(n_knots=8, constraint=Constraint.fit.increasing)},
        )
        model.fit(df[["x"]], df["y"])
        x_grid = np.linspace(0, 1, 200)
        pred = model.predict(pd.DataFrame({"x": x_grid}))
        assert np.all(np.diff(pred) >= -1e-8)

    @pytest.mark.slow
    def test_discrete_crs_monotone(self):
        rng = np.random.default_rng(42)
        n = 1000
        x = np.sort(rng.uniform(0, 1, n))
        y = 2 * x + rng.normal(0, 0.2, n)
        df = pd.DataFrame({"x": x, "y": y})
        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            discrete=True,
            features={"x": CubicRegressionSpline(n_knots=8, constraint=Constraint.fit.increasing)},
        )
        model.fit(df[["x"]], df["y"])
        x_grid = np.linspace(0, 1, 200)
        pred = model.predict(pd.DataFrame({"x": x_grid}))
        assert np.all(np.diff(pred) >= -1e-8)

    @pytest.mark.slow
    def test_discrete_qp_mixed_model(self):
        """Monotone discrete + ordinary discrete in same model."""
        rng = np.random.default_rng(42)
        n = 1000
        x1 = np.sort(rng.uniform(0, 1, n))
        x2 = rng.uniform(0, 1, n)
        y = 2 * x1 + np.sin(2 * np.pi * x2) + rng.normal(0, 0.2, n)
        df = pd.DataFrame({"x1": x1, "x2": x2, "y": y})
        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            discrete=True,
            features={
                "x1": BSplineSmooth(n_knots=8, constraint=Constraint.fit.increasing),
                "x2": PSpline(n_knots=8),
            },
        )
        model.fit(df[["x1", "x2"]], df["y"])
        x_grid = np.linspace(0, 1, 200)
        pred = model.predict(pd.DataFrame({"x1": x_grid, "x2": np.full(200, 0.5)}))
        assert np.all(np.diff(pred) >= -1e-8)

        # Verify both terms are discretized
        from superglm.group_matrix import DiscretizedSSPGroupMatrix

        gms = model._dm.group_matrices
        groups = model._groups
        for gm, g in zip(gms, groups):
            assert isinstance(gm, DiscretizedSSPGroupMatrix), (
                f"Term {g.feature_name} should be discretized, got {type(gm).__name__}"
            )

    @pytest.mark.slow
    def test_discrete_vs_nondiscrete_parity(self):
        """Discrete and non-discrete monotone fits should give similar results."""
        rng = np.random.default_rng(42)
        n = 500
        x = np.sort(rng.uniform(0, 1, n))
        y = 2 * x + rng.normal(0, 0.2, n)
        df = pd.DataFrame({"x": x, "y": y})
        x_grid = np.linspace(0, 1, 200)
        df_grid = pd.DataFrame({"x": x_grid})

        model_dense = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            discrete=False,
            features={"x": BSplineSmooth(n_knots=8, constraint=Constraint.fit.increasing)},
        )
        model_dense.fit(df[["x"]], df["y"])
        pred_dense = model_dense.predict(df_grid)

        model_disc = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            discrete=True,
            features={"x": BSplineSmooth(n_knots=8, constraint=Constraint.fit.increasing)},
        )
        model_disc.fit(df[["x"]], df["y"])
        pred_disc = model_disc.predict(df_grid)

        np.testing.assert_allclose(pred_dense, pred_disc, atol=0.05)


class TestDiscreteSCOPMonotone:
    @pytest.mark.slow
    def test_discrete_pspline_monotone(self):
        rng = np.random.default_rng(42)
        n = 1000
        x = np.sort(rng.uniform(0, 1, n))
        y = 2 * x + rng.normal(0, 0.2, n)
        df = pd.DataFrame({"x": x, "y": y})
        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            discrete=True,
            features={"x": PSpline(n_knots=8, constraint=Constraint.fit.increasing)},
        )
        model.fit(df[["x"]], df["y"])
        x_grid = np.linspace(0, 1, 200)
        pred = model.predict(pd.DataFrame({"x": x_grid}))
        assert np.all(np.diff(pred) >= -1e-8)

    @pytest.mark.slow
    def test_discrete_scop_vs_nondiscrete_parity(self):
        rng = np.random.default_rng(42)
        n = 500
        x = np.sort(rng.uniform(0, 1, n))
        y = 2 * x + rng.normal(0, 0.2, n)
        df = pd.DataFrame({"x": x, "y": y})
        x_grid = np.linspace(0, 1, 200)
        df_grid = pd.DataFrame({"x": x_grid})

        model_dense = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            discrete=False,
            features={
                "x": PSpline(n_knots=8, constraint=Constraint.fit.increasing, shape_engine="scop")
            },
        )
        model_dense.fit(df[["x"]], df["y"])
        pred_dense = model_dense.predict(df_grid)

        model_disc = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            discrete=True,
            features={
                "x": PSpline(n_knots=8, constraint=Constraint.fit.increasing, shape_engine="scop")
            },
        )
        model_disc.fit(df[["x"]], df["y"])
        pred_disc = model_disc.predict(df_grid)

        np.testing.assert_allclose(pred_dense, pred_disc, atol=0.05)


QP_TERMS = {
    "ps": lambda: PSpline(n_knots=6, constraint=Constraint.fit.increasing),
    "cr": lambda: CubicRegressionSpline(n_knots=6, constraint=Constraint.fit.increasing),
    "bs": lambda: BSplineSmooth(n_knots=6, constraint=Constraint.fit.increasing),
}


def _qp_fit(kind: str, x: np.ndarray, y: np.ndarray, constrained: bool = True) -> SuperGLM:
    term = QP_TERMS[kind]() if constrained else type(QP_TERMS[kind]())(n_knots=6)
    model = SuperGLM(
        family=Gaussian(), selection_penalty=0, spline_penalty=1.0, features={"x": term}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.fit(pd.DataFrame({"x": x}), y)


@pytest.mark.parametrize("kind", sorted(QP_TERMS))
def test_a_term_the_constraint_holds_flat_publishes_its_face_statistics(kind):
    """A decreasing truth under ``increasing`` binds every row: the term's face is ``{0}``.

    The fit is then the intercept alone (Meyer and Woodroofe 2000: the degrees
    of freedom of a constrained fit are its face's), so the EDF is 1, the
    Gaussian scale is ``RSS / (n - 1)``, the AIC charges one parameter, and
    the term's covariance is zero.  Mutation: the export's face dropped
    (``_active_face_basis`` returning ``None``) publishes the unconstrained
    trace, 4.7 for ``ps``, and its scale.
    """
    from superglm._roundoff import gamma
    from superglm.model.state_ops import coef_covariance

    n = 60
    x = np.linspace(0.0, 1.0, n)
    y = 1.0 - 2.0 * x
    model = _qp_fit(kind, x, y)
    mu = model.predict(pd.DataFrame({"x": x}))
    solver = model._solver_pirls_result()
    assert solver.converged
    assert np.ptp(mu) <= gamma(n) * np.max(np.abs(y))
    p = solver.beta.size
    assert abs(solver.effective_df - 1.0) <= gamma(p) * p
    rss = float(np.sum((y - mu) ** 2))
    np.testing.assert_allclose(solver.phi, rss / (n - 1.0), rtol=gamma(n))
    metrics = model.metrics(pd.DataFrame({"x": x}), y)
    np.testing.assert_allclose(metrics.aic + 2.0 * metrics.log_likelihood, 2.0, rtol=gamma(n))
    covariance, _ = coef_covariance(model)
    assert np.max(np.abs(covariance)) <= gamma(p) * solver.phi


@pytest.mark.parametrize("kind", sorted(QP_TERMS))
def test_a_term_held_flat_beside_a_fitted_one_adds_no_degrees_of_freedom(kind):
    """A term the constraint holds flat beside a fitted term adds no degrees of freedom.

    Its coefficients sit at round-off, not at zero, so its rows bind only on a
    scale shared by the whole fit; its face is ``{0}`` and the EDF is the
    model's without it.  Mutation: zero slack read on each row's own scale
    counts the held directions (9.3 against 5.6 for ``ps``).
    """
    from superglm._roundoff import gamma

    rng = np.random.default_rng(0)
    n = 200
    x, z = rng.uniform(0.0, 1.0, n), rng.uniform(0.0, 1.0, n)
    y = 1.0 - 2.0 * x + np.sin(6.0 * z) + rng.normal(0.0, 0.1, n)
    free = type(QP_TERMS[kind]())
    features = {"x": QP_TERMS[kind](), "z": free(n_knots=6)}
    model = SuperGLM(family=Gaussian(), selection_penalty=0, spline_penalty=1.0, features=features)
    alone = SuperGLM(
        family=Gaussian(), selection_penalty=0, spline_penalty=1.0, features={"z": free(n_knots=6)}
    )
    frame = pd.DataFrame({"x": x, "z": z})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(frame, y)
        alone.fit(frame[["z"]], y)
    solver = model._solver_pirls_result()
    assert solver.converged
    reference = alone._solver_pirls_result()
    # each trace entry is within gamma_p kappa of its value, kappa either
    # fit's penalised Hessian condition (the two traces round differently);
    # pre_truncation_condition is the factor's, the square root of it
    kappa = (
        max(
            solver.rank_info.augmented.pre_truncation_condition,
            reference.rank_info.augmented.pre_truncation_condition,
        )
        ** 2
    )
    p = solver.beta.size
    bound = gamma(p) * kappa * p
    assert bound < 1.0  # the per-row reading moves the EDF by about 3.7
    assert abs(solver.effective_df - reference.effective_df) <= bound


@pytest.mark.parametrize(
    ("beta", "column_scale", "row_scale", "held"),
    [
        # a flat term at round-off beside a fitted column: both rows bind
        ([1e-16, 3e-16, 6e-16, 1.0], [1.0, 1.0, 1.0, 1.0], 1.0, 2),
        # a free slope of 1e-3 beside a covariate in units of 1e-9 (coefficient 1e9)
        ([0.0, 1e-3, 2e-3, 1e9], [1.0, 1.0, 1.0, 1e-9], 1.0, 0),
        # a free slope in large units (contribution 1e-3) beside a fitted column
        ([0.0, 1e-12, 2e-12, 1.0], [1e9, 1e9, 1e9, 1.0], 1.0, 0),
        # binding rows at large magnitudes: 1e150 rows on 1e160 coefficients
        ([1e160, 1e160, 1e160, 1e160], [2e-150] * 4, 1e150, 2),
        # free rows of 1e154 on coefficients near 1e152
        ([0.0, 1e152, 2e152, 1.0], [5e-155] * 3 + [1.0], 1e154, 0),
        # a flat term at round-off on rows of 1e154: both rows bind
        ([1e138, 3e138, 6e138, 1.0], [5e-155] * 3 + [1.0], 1e154, 2),
        # a flat term at round-off on rows of 1e-200, whose squared norms underflow
        ([1e-16, 3e-16, 6e-16, 1.0], [1.0] * 4, 1e-200, 2),
        # binding rows on columns with no data norm: the fit's scale is zero and abstains
        ([1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0], 1.0, 2),
        # free rows whose fit-scale denominator overflows after row scaling (2 * 0.9 * 1e308)
        ([0.0, 1.0, 2.0, 1.0], [1e-308] * 3 + [1.0], 1.8, 0),
    ],
)
def test_the_face_reads_zero_slack_against_the_largest_column_contribution(
    beta, column_scale, row_scale, held
):
    """A row binds when its slack is zero on its own scale or on ``max_j |beta_j| ||x_j||``.

    Mutations, and the cases each fails: the row's own scale alone (1, 6, 7);
    the fit's scale alone (8, columns with no data norm); the largest raw
    coefficient as the reference (2, 3, 6); the reference left in coefficient
    units (3, 6); no power-of-two row scaling (6 frees a flat term, 7 fails
    in the null-space SVD); no finiteness check on the fit's scale (9).  Cases
    4 and 5 check that rows and coefficients near float64's range read as they
    do at unit scale.
    """
    from superglm.solvers.irls_direct import _active_face_basis

    A = np.zeros((2, 4))
    A[0, :2], A[1, 1:3] = [-row_scale, row_scale], [-row_scale, row_scale]
    face = _active_face_basis(A, np.zeros(2), np.asarray(beta), [0, 1], np.asarray(column_scale))
    assert (4 if face is None else face.shape[1]) == 4 - held


def test_a_bound_far_above_its_row_leaves_the_row_in_range():
    """A row of order one with a bound of 1e200 that binds holds its direction.

    Mutation: the scaling exponent taken from the bound as well shrinks the row
    to about 1e-200, its squared norm underflows, and the face keeps a
    spurious free direction.
    """
    from superglm.solvers.irls_direct import _active_face_basis

    face = _active_face_basis(
        np.array([[1.0, 0.0]]),
        np.array([1e200]),
        np.array([1e200, 1.0]),
        [0],
        np.array([1e-100, 1.0]),
    )
    assert face is not None and face.shape[1] == 1


def test_an_overflowing_unit_scale_on_a_column_no_row_touches_reads_the_rows():
    """A flat term beside a fitted column of 1e150 binds, whatever an untouched column's norm.

    The fifth column's norm, 1e-160 against a contribution of 1e150, overflows
    its unit scale.  Mutation: the fit's scale summed over every column
    multiplies that by the rows' zero entries (NaN), the fit's reading
    abstains, and the flat term reads as free.
    """
    from superglm.solvers.irls_direct import _active_face_basis

    A = np.zeros((2, 5))
    A[0, :2], A[1, 1:3] = [-1.0, 1.0], [-1.0, 1.0]
    face = _active_face_basis(
        A,
        np.zeros(2),
        np.array([1e134, 3e134, 6e134, 1e150, 0.0]),
        [0, 1],
        np.array([1.0, 1.0, 1.0, 1.0, 1e-160]),
    )
    assert face is not None and face.shape[1] == 3


@pytest.mark.parametrize("kind", sorted(QP_TERMS))
def test_a_partly_binding_constraint_publishes_the_divergence_of_its_fit(kind):
    """The EDF of a fit some rows bind is the divergence of its fitted values.

    Stein's degrees of freedom are ``E[div mu_hat(y)]`` (Tibshirani and Taylor
    2012, Theorem 3): near ``y`` the constrained fit is the fit with its
    binding rows held as equalities, affine in ``y``, so its divergence is
    the trace on that face and a one-sided difference of size ``h`` is exact
    but for rounding.  Each fitted value carries at most ``gamma_p kappa
    max|mu|`` of it, ``kappa`` the equilibrated Hessian's condition, so the
    sum of ``n`` quotients is within ``2 n gamma_p kappa max|mu| / h``.  The
    face is partial: the EDF lies strictly between the intercept's 1 and the
    unconstrained fit's.  Mutation: the export's face dropped publishes the
    unconstrained trace.
    """
    from superglm._roundoff import gamma

    rng = np.random.default_rng(0)
    n = 30
    x = np.sort(rng.uniform(0.0, 1.0, n))
    y = np.where(x < 0.5, 1.0 - 0.6 * x, 0.7 + 1.5 * (x - 0.5)) + rng.normal(0.0, 0.05, n)
    frame = pd.DataFrame({"x": x})
    model = _qp_fit(kind, x, y)
    solver = model._solver_pirls_result()
    assert solver.converged
    free = _qp_fit(kind, x, y, constrained=False)._solver_pirls_result().effective_df
    assert 1.0 < solver.effective_df < free - 0.5
    mu = model.predict(frame)
    h = 1e-6
    divergence = 0.0
    for i in range(n):
        bumped = y.copy()
        bumped[i] += h
        divergence += (_qp_fit(kind, x, bumped).predict(frame)[i] - mu[i]) / h
    kappa = solver.rank_info.augmented.pre_truncation_condition
    bound = 2 * n * gamma(solver.beta.size) * kappa * np.max(np.abs(mu)) / h
    assert abs(solver.effective_df - divergence) <= bound
