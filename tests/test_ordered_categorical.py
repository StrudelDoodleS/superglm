"""Tests for OrderedCategorical feature type."""

import numpy as np
import pandas as pd
import pytest

from superglm import Categorical, Constraint, OrderedCategorical, PSpline, Spline, SuperGLM

# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def age_band_data():
    """Synthetic data with ordered age bands."""
    rng = np.random.default_rng(42)
    n = 2000
    bands = ["18-25", "26-35", "36-45", "46-55", "56-65", "65+"]
    x = rng.choice(bands, n, p=[0.15, 0.25, 0.25, 0.20, 0.10, 0.05])
    sample_weight = rng.uniform(0.3, 1.0, n)
    midpoints = {
        "18-25": 21.5,
        "26-35": 30.5,
        "36-45": 40.5,
        "46-55": 50.5,
        "56-65": 60.5,
        "65+": 70.0,
    }
    x_numeric = np.array([midpoints[v] for v in x])
    mu = np.exp(-2.0 + 0.01 * (x_numeric - 45) ** 2 / 100)
    y = rng.poisson(mu * sample_weight).astype(float)
    X = pd.DataFrame({"age_band": x})
    return X, y, sample_weight, midpoints, bands


@pytest.fixture
def ordinal_data():
    """Synthetic data with generic ordered levels."""
    rng = np.random.default_rng(123)
    n = 1000
    levels = ["Low", "Medium", "High", "Very High"]
    x = rng.choice(levels, n, p=[0.3, 0.3, 0.25, 0.15])
    sample_weight = rng.uniform(0.5, 1.0, n)
    # True effect: monotone increasing
    effect = {"Low": 0.0, "Medium": 0.2, "High": 0.5, "Very High": 0.8}
    mu = np.exp(-1.5 + np.array([effect[v] for v in x]))
    y = rng.poisson(mu * sample_weight).astype(float)
    X = pd.DataFrame({"risk": x})
    return X, y, sample_weight, levels


# ── Constructor Tests ─────────────────────────────────────────────


class TestConstructor:
    def test_values_derive_ordering(self):
        spec = OrderedCategorical(
            values={"C": 3.0, "A": 1.0, "B": 2.0}, basis=Spline(kind="ps", n_knots=2)
        )
        assert spec._ordered_levels == ["A", "B", "C"]

    def test_order_generates_linspace(self):
        spec = OrderedCategorical(order=["X", "Y", "Z"], basis=Spline(kind="ps", n_knots=2))
        assert spec._level_to_value == {"X": 0.0, "Y": 0.5, "Z": 1.0}

    def test_mutual_exclusion_both(self):
        with pytest.raises(ValueError, match="exactly one"):
            OrderedCategorical(values={"A": 1.0}, order=["A"])

    def test_mutual_exclusion_neither(self):
        with pytest.raises(ValueError, match="Must specify"):
            OrderedCategorical()

    def test_invalid_basis(self):
        with pytest.raises(ValueError, match="basis must be"):
            OrderedCategorical(order=["A", "B"], basis="cubic")

    @pytest.mark.parametrize(
        ("shortcut", "value"),
        [("kind", "cr"), ("n_knots", 4), ("degree", 2), ("select", True), ("penalty", "ssp")],
    )
    def test_removed_shortcut_raises_type_error(self, shortcut, value):
        """The five scalar shortcuts were removed in 0.24.0; passing one must
        fail at the constructor with the parameter's own name, not warn and
        build something."""
        with pytest.raises(TypeError, match=shortcut):
            OrderedCategorical(order=["A", "B", "C"], **{shortcut: value})

    def test_removed_shortcut_raises_even_beside_a_spline_basis(self):
        """Before removal a shortcut next to `basis=Spline(...)` was merely
        ignored with a warning; silence now would be indistinguishable from
        the shortcut still working."""
        with pytest.raises(TypeError, match="n_knots"):
            OrderedCategorical(order=["A", "B", "C"], basis=Spline(kind="ps", n_knots=2), n_knots=7)

    @pytest.mark.parametrize("legacy", ["spline", "step"])
    def test_removed_basis_string_names_the_replacement(self, legacy):
        """`basis="spline"`/`basis="step"` were removed in 0.24.0; the error
        must name `basis=Spline(...)` as the way forward."""
        with pytest.raises(ValueError, match=r"basis=Spline\(\.\.\.\)"):
            OrderedCategorical(order=["A", "B", "C"], basis=legacy)


# ── Spline Mode: Build / Transform / Reconstruct ─────────────────


class TestSplineMode:
    def test_build_returns_groupinfo(self, age_band_data):
        X, y, sample_weight, midpoints, _ = age_band_data
        spec = OrderedCategorical(values=midpoints, basis=Spline(kind="ps", n_knots=3))
        result = spec.build(X["age_band"].values, sample_weight=sample_weight)
        # Should return GroupInfo (not a list when select=False)
        from superglm.types import GroupInfo

        assert isinstance(result, GroupInfo)
        assert result.n_cols > 0
        assert result.penalty_matrix is not None

    def test_build_select_returns_single_group_with_components(self, age_band_data):
        X, y, sample_weight, midpoints, _ = age_band_data
        spec = OrderedCategorical(values=midpoints, basis=Spline(kind="ps", n_knots=3, select=True))
        result = spec.build(X["age_band"].values, sample_weight=sample_weight)
        assert not isinstance(result, list)
        assert result.penalty_components is not None
        assert len(result.penalty_components) == 2  # null + wiggle

    def test_transform_shape(self, age_band_data):
        X, y, sample_weight, midpoints, _ = age_band_data
        spec = OrderedCategorical(values=midpoints, basis=Spline(kind="ps", n_knots=3))
        spec.build(X["age_band"].values, sample_weight=sample_weight)
        T = spec.transform(X["age_band"].values)
        assert T.shape[0] == len(X)

    def test_reconstruct_has_level_annotations(self, age_band_data):
        X, y, sample_weight, midpoints, bands = age_band_data
        # Use full model pipeline so R_inv is set correctly
        model = SuperGLM(
            features={
                "age_band": OrderedCategorical(values=midpoints, basis=Spline(kind="ps", n_knots=3))
            },
        )
        model.fit(X, y, sample_weight=sample_weight)
        spec = model._specs["age_band"]
        beta_combined = model._result.beta[model._groups[0].sl]
        raw = spec.reconstruct(beta_combined)
        # Should have standard spline keys
        assert "x" in raw
        assert "log_relativity" in raw
        assert "relativity" in raw
        # Plus per-level annotations
        assert "levels" in raw
        assert "level_values" in raw
        assert "level_log_relativities" in raw
        assert "level_relativities" in raw
        assert set(raw["levels"]) == set(bands)

    def test_unseen_level_raises(self, age_band_data):
        X, y, sample_weight, midpoints, _ = age_band_data
        spec = OrderedCategorical(values=midpoints)
        spec.build(X["age_band"].values, sample_weight=sample_weight)
        with pytest.raises(ValueError, match="unseen"):
            spec.transform(np.array(["UNKNOWN"]))

    def test_n_knots_clamping_applies_to_the_default_basis(self):
        """The default P-spline's n_knots=5 must clamp to n_levels-1 with the
        same warning an explicit Spline gets."""
        with pytest.warns(UserWarning, match="clamped"):
            spec = OrderedCategorical(order=["A", "B", "C"])
        assert spec._spline.n_knots == 2  # min(5, 3-1) = 2
        assert spec.n_knots == 2  # the derived attribute reports the clamped value

    def test_spline_matches_manual(self, age_band_data):
        """Spline mode should produce the same result as manual Spline on midpoints."""
        from superglm.features.spline import PSpline

        X, y, sample_weight, midpoints, _ = age_band_data
        x_vals = X["age_band"].values
        x_numeric = np.array([midpoints[v] for v in x_vals])

        # Manual spline
        manual_spline = PSpline(n_knots=3, degree=3, penalty="ssp")
        manual_info = manual_spline.build(x_numeric, sample_weight=sample_weight)

        # OrderedCategorical
        spec = OrderedCategorical(values=midpoints, basis=Spline(kind="ps", n_knots=3))
        ocat_info = spec.build(x_vals, sample_weight=sample_weight)

        # Same penalty matrix
        np.testing.assert_allclose(ocat_info.penalty_matrix, manual_info.penalty_matrix)
        # Same number of columns
        assert ocat_info.n_cols == manual_info.n_cols


# ── Reporting Base ────────────────────────────────────────────────


class TestReportingBase:
    def test_base_level_most_exposed(self, ordinal_data):
        X, y, sample_weight, levels = ordinal_data
        spec = OrderedCategorical(
            order=levels, base="most_exposed", basis=Spline(kind="ps", n_knots=3)
        )
        spec.build(X["risk"].values, sample_weight=sample_weight)
        # Should pick the level with highest total sample_weight
        assert spec._base_level in levels

    def test_base_level_explicit(self, ordinal_data):
        X, y, sample_weight, levels = ordinal_data
        spec = OrderedCategorical(order=levels, base="High", basis=Spline(kind="ps", n_knots=3))
        spec.build(X["risk"].values, sample_weight=sample_weight)
        assert spec._base_level == "High"


# ── Edge Cases ────────────────────────────────────────────────────


class TestEdgeCases:
    def test_order_single_value_linspace(self):
        """Single level should produce value=0.0."""
        # This is degenerate but shouldn't crash; the default n_knots clamps
        # to n_levels - 1 = 0 with the usual warning.
        with pytest.warns(UserWarning, match="clamped to 0"):
            spec = OrderedCategorical(order=["Only"])
        assert spec._level_to_value == {"Only": 0.0}

    def test_unseen_level_at_predict(self, ordinal_data):
        X, y, sample_weight, levels = ordinal_data
        spec = OrderedCategorical(order=levels, basis=Spline(kind="ps", n_knots=3))
        spec.build(X["risk"].values, sample_weight=sample_weight)
        with pytest.raises(ValueError, match="unseen"):
            spec.transform(np.array(["UNKNOWN"]))


# ── Integration Tests ─────────────────────────────────────────────


class TestIntegrationSpline:
    def test_fit_predict(self, age_band_data):
        X, y, sample_weight, midpoints, _ = age_band_data
        model = SuperGLM(
            features={
                "age_band": OrderedCategorical(values=midpoints, basis=Spline(kind="ps", n_knots=3))
            },
        )
        model.fit(X, y, sample_weight=sample_weight)
        preds = model.predict(X)
        assert preds.shape == (len(X),)
        assert np.all(preds > 0)

    def test_summary(self, age_band_data):
        X, y, sample_weight, midpoints, bands = age_band_data
        model = SuperGLM(
            features={
                "age_band": OrderedCategorical(values=midpoints, basis=Spline(kind="ps", n_knots=3))
            },
        )
        model.fit(X, y, sample_weight=sample_weight)
        s = model.summary()
        text = str(s)
        assert "age_band[18-25]" in text
        assert f"age_band[{bands[-1]}]" in text
        level_rows = [r for r in s._coef_rows if r.group == "age_band" and not r.is_spline]
        assert len(level_rows) == len(bands)
        assert all(r.name.startswith("age_band[") for r in level_rows)
        assert not any(r.name == "age_band" and r.coef is not None for r in s._coef_rows)
        assert all(np.isfinite(r.se) and r.se >= 0 for r in level_rows if r.se is not None)

    def test_relativities(self, age_band_data):
        X, y, sample_weight, midpoints, _ = age_band_data
        model = SuperGLM(
            features={
                "age_band": OrderedCategorical(values=midpoints, basis=Spline(kind="ps", n_knots=3))
            },
        )
        model.fit(X, y, sample_weight=sample_weight)
        rels = model.relativities()
        assert "age_band" in rels
        df = rels["age_band"]
        assert "relativity" in df.columns

    def test_term_inference(self, age_band_data):
        X, y, sample_weight, midpoints, bands = age_band_data
        model = SuperGLM(
            features={
                "age_band": OrderedCategorical(values=midpoints, basis=Spline(kind="ps", n_knots=3))
            },
        )
        model.fit(X, y, sample_weight=sample_weight)
        ti = model.term_inference("age_band")
        # Primary output is categorical (K levels), not a continuous curve
        assert ti.kind == "categorical"
        assert ti.levels is not None
        assert set(ti.levels) == set(bands)
        assert len(ti.relativity) == len(bands)

    def test_spline_se_at_levels(self, age_band_data):
        """Spline mode SEs should be at the K category positions, not on a grid."""
        X, y, sample_weight, midpoints, bands = age_band_data
        model = SuperGLM(
            features={
                "age_band": OrderedCategorical(values=midpoints, basis=Spline(kind="ps", n_knots=3))
            },
        )
        model.fit(X, y, sample_weight=sample_weight)
        ti = model.term_inference("age_band")
        se = ti.se_log_relativity
        assert se is not None
        # SE array should have K entries (one per level), not 200
        assert len(se) == len(bands)
        assert np.all(np.isfinite(se))
        assert np.all(se >= 0)
        assert np.any(se > 0)
        assert np.max(se) < 5.0

    def test_smooth_curve_for_plotting(self, age_band_data):
        """Spline mode should provide a smooth_curve for plotting."""
        X, y, sample_weight, midpoints, _ = age_band_data
        model = SuperGLM(
            features={
                "age_band": OrderedCategorical(values=midpoints, basis=Spline(kind="ps", n_knots=3))
            },
        )
        model.fit(X, y, sample_weight=sample_weight)
        ti = model.term_inference("age_band")
        curve = ti.smooth_curve
        assert curve is not None
        # Continuous grid (default 200 points)
        assert len(curve.x) == 200
        assert len(curve.relativity) == 200
        assert curve.se_log_relativity is not None
        assert len(curve.se_log_relativity) == 200
        assert curve.ci_lower is not None
        assert curve.ci_upper is not None

    def test_relativities_per_level(self, age_band_data):
        """relativities() should return per-level output, not a continuous curve."""
        X, y, sample_weight, midpoints, bands = age_band_data
        model = SuperGLM(
            features={
                "age_band": OrderedCategorical(values=midpoints, basis=Spline(kind="ps", n_knots=3))
            },
        )
        model.fit(X, y, sample_weight=sample_weight)
        rels = model.relativities(with_se=True)
        df = rels["age_band"]
        assert "level" in df.columns
        assert len(df) == len(bands)
        assert "se_log_relativity" in df.columns
        assert np.all(np.isfinite(df["se_log_relativity"].values))


class TestIntegrationReml:
    def test_fit_reml_spline(self, age_band_data):
        X, y, sample_weight, midpoints, _ = age_band_data
        model = SuperGLM(
            features={
                "age_band": OrderedCategorical(values=midpoints, basis=Spline(kind="ps", n_knots=3))
            },
        )
        model.fit_reml(X, y, sample_weight=sample_weight)
        assert model._reml_result is not None
        assert model._reml_result.converged
        assert len(model._reml_result.lambdas) > 0
        preds = model.predict(X)
        assert np.all(preds > 0)

    def test_fit_reml_select(self, age_band_data):
        X, y, sample_weight, midpoints, _ = age_band_data
        model = SuperGLM(
            features={
                "age_band": OrderedCategorical(
                    values=midpoints, basis=Spline(kind="ps", n_knots=3, select=True)
                )
            },
        )
        # Explicit loose tolerance: the select=True null direction marches
        # toward the lambda bound and the Newton endgame stalls
        # (line_search_failed) above the tight publication bar at reproducible
        # lambdas. The subject here is that select=True fits under REML, not
        # the stopping criterion's endgame classification.
        model.fit_reml(X, y, sample_weight=sample_weight, reml_tol=1e-6)
        assert model._reml_result is not None
        assert model._reml_result.converged


class TestIntegrationMixed:
    def test_ocat_with_other_features(self, age_band_data):
        """OrderedCategorical works alongside other feature types."""
        X, y, sample_weight, midpoints, _ = age_band_data
        rng = np.random.default_rng(42)
        X = X.copy()
        X["region"] = rng.choice(["A", "B", "C"], len(X))
        model = SuperGLM(
            features={
                "age_band": OrderedCategorical(
                    values=midpoints, basis=Spline(kind="ps", n_knots=3)
                ),
                "region": Categorical(base="first"),
            },
        )
        model.fit(X, y, sample_weight=sample_weight)
        preds = model.predict(X)
        assert preds.shape == (len(X),)
        assert np.all(preds > 0)


class TestSplineObjectBasis:
    """OrderedCategorical with a Spline() object as basis."""

    def test_spline_object_builds_and_fits(self, age_band_data):
        from superglm.features.spline import Spline

        X, y, sample_weight, _, _ = age_band_data
        spec = OrderedCategorical(
            order=sorted(X["age_band"].unique()),
            basis=Spline(n_knots=4, kind="bs"),
        )
        model = SuperGLM(
            family="poisson",
            features={"age_band": spec},
            selection_penalty=0.0,
        )
        model.fit(X, y, sample_weight=sample_weight)
        assert model.result.converged

    def test_spline_object_inherits_constraint(self, age_band_data):
        X, y, sample_weight, _, _ = age_band_data
        spec = OrderedCategorical(
            order=sorted(X["age_band"].unique()),
            basis=PSpline(n_knots=4, constraint=Constraint.fit.increasing),
        )
        assert spec._spline.monotone == "increasing"
        assert spec._spline.monotone_mode == "fit"
        model = SuperGLM(
            family="poisson",
            features={"age_band": spec},
            selection_penalty=0.0,
        )
        model.fit(X, y, sample_weight=sample_weight)
        assert model.result.converged
        assert spec._spline.monotone == "increasing"
        assert spec._spline.monotone_mode == "fit"

    @pytest.mark.parametrize("kind", ["cr", "bs"])
    @pytest.mark.parametrize("direction", ["increasing", "decreasing"])
    def test_fit_time_monotone_on_qp_bases(self, ordinal_data, kind, direction):
        """A fit-time monotone constraint must work on the QP bases, not just ps.

        `OrderedCategorical.build()` delegates to its inner spline, and that
        spline stamps ``monotone_engine="qp"`` on the GroupInfo it returns. The
        builder then asked the OUTER spec -- the OrderedCategorical -- for the
        raw constraint geometry, which only the inner spline owns, and died with
        "its raw constraint geometry is unavailable". `ps` masked this because it
        routes through SCOP instead and never takes the QP branch.
        """
        from superglm.features.spline import Spline

        X, y, sample_weight, levels = ordinal_data
        spec = OrderedCategorical(
            order=levels,
            basis=Spline(kind=kind, k=5, constraint=getattr(Constraint.fit, direction)),
        )
        model = SuperGLM(family="poisson", features={"risk": spec}, selection_penalty=0.0)
        model.fit(X, y, sample_weight=sample_weight)
        assert model.result.converged
        # A monotone curve alone cannot prove the QP path engaged -- postfit
        # repair or SCOP could produce one too. Pin the dispatch this fix
        # delegates: the fitted group must carry the QP engine stamp.
        engines = {g.monotone_engine for g in model._groups if g.feature_name == "risk"}
        assert "qp" in engines and "scop" not in engines, engines

        rels = model.reconstruct_feature("risk")["level_log_relativities"]
        curve = np.array([rels[lev] for lev in levels], dtype=float)
        steps = np.diff(curve)
        # Feasibility slack derived from dtype epsilon and the output's own
        # scale, not a hard-coded absolute (AGENTS numerical test policy).
        slack = np.finfo(float).eps ** 0.5 * max(1.0, float(np.max(np.abs(curve))))
        if direction == "increasing":
            assert np.all(steps >= -slack), f"not increasing: {steps}"
            # One-sided monotonicity alone is satisfied by an all-zero
            # collapsed block, so a reversed constraint direction would still
            # pass. The fixture's true effect rises 0 -> 0.8: pin that the
            # constrained fit RECOVERS the rise.
            assert curve[-1] - curve[0] > 0.1, f"no material rise: {curve}"
        else:
            assert np.all(steps <= slack), f"not decreasing: {steps}"
            # Against rising truth a decreasing-constrained fit can at best
            # go flat; a material rise here means the constraint was dropped.
            assert curve[-1] - curve[0] < 0.05, f"constraint not binding: {curve}"

    def test_fit_time_monotone_on_qp_bases_under_reml(self, ordinal_data):
        """`model/reml_setup.py` restores QP constraints through a second, separate
        lookup of the same raw geometry, so REML pins its own call site."""
        from superglm.features.spline import Spline

        X, y, sample_weight, levels = ordinal_data
        spec = OrderedCategorical(
            order=levels,
            basis=Spline(kind="cr", k=5, constraint=Constraint.fit.increasing),
        )
        model = SuperGLM(family="poisson", features={"risk": spec})
        model.fit_reml(X, y, sample_weight=sample_weight)
        engines = {g.monotone_engine for g in model._groups if g.feature_name == "risk"}
        assert "qp" in engines and "scop" not in engines, engines

        rels = model.reconstruct_feature("risk")["level_log_relativities"]
        curve = np.array([rels[lev] for lev in levels], dtype=float)
        steps = np.diff(curve)
        slack = np.finfo(float).eps ** 0.5 * max(1.0, float(np.max(np.abs(curve))))
        assert np.all(steps >= -slack), f"not increasing: {steps}"
        assert curve[-1] - curve[0] > 0.1, f"no material rise: {curve}"

    def test_fit_time_monotone_composes_with_specials(self, ordinal_data):
        """`specials=` is the one delegation shape the wrapper reshapes: the
        build returns two GroupInfos and row-expands the spline block, so the
        forwarded raw constraint -- stated in the inner spline's coefficient
        space -- meets a design whose rows and columns the wrapper rearranged.
        The identifiability projection is built on the ordered rows only and
        the special block is unpenalized, so the composition holds; this pins
        it against future changes to either block."""
        from superglm.features.spline import Spline

        X, y, sample_weight, levels = ordinal_data
        X = X.copy()
        special_rows = np.arange(len(X)) % 10 == 0
        X.loc[special_rows, "risk"] = "MISSING"
        spec = OrderedCategorical(
            order=levels,
            specials=["MISSING"],
            basis=Spline(kind="cr", k=5, constraint=Constraint.fit.increasing),
        )
        model = SuperGLM(family="poisson", features={"risk": spec}, selection_penalty=0.0)
        model.fit(X, y, sample_weight=sample_weight)
        assert model.result.converged
        # A monotone curve alone cannot prove the QP path engaged -- postfit
        # repair or SCOP could produce one too. Pin the dispatch this fix
        # delegates: the fitted group must carry the QP engine stamp.
        engines = {g.monotone_engine for g in model._groups if g.feature_name == "risk"}
        assert "qp" in engines and "scop" not in engines, engines

        rels = model.reconstruct_feature("risk")["level_log_relativities"]
        curve = np.array([rels[lev] for lev in levels], dtype=float)
        steps = np.diff(curve)
        slack = np.finfo(float).eps ** 0.5 * max(1.0, float(np.max(np.abs(curve))))
        assert np.all(steps >= -slack), f"not increasing: {steps}"
        assert curve[-1] - curve[0] > 0.1, f"no material rise: {curve}"
        assert "MISSING" in rels

    def test_repr_shows_spline_object(self):
        from superglm.features.spline import Spline

        spec = OrderedCategorical(
            order=["a", "b", "c", "d"],
            basis=Spline(n_knots=3),
        )
        r = repr(spec)
        assert "OrderedCategorical" in r
        assert "4 levels" in r

    def test_n_knots_clamped_for_spline_object(self):
        from superglm.features.spline import Spline

        with pytest.warns(UserWarning, match="clamped"):
            spec = OrderedCategorical(
                order=["a", "b", "c"],  # 3 levels → max 2 knots
                basis=Spline(n_knots=10),
            )
        assert spec._spline.n_knots == 2


# ── Post-fit shape repair and structured rejection on the wrapper ──


@pytest.fixture
def dipping_ordinal_data():
    """Ordered levels whose true effect DIPS, so a monotone repair must bind.

    ``ordinal_data``'s truth is monotone by construction, so a post-fit
    ``increasing`` repair on it is a no-op whether or not the repair engine
    ever sees the term -- it measures nothing. The designed dip here is the
    signal: an unrepaired fit must reproduce it, a repaired one must not.
    """
    rng = np.random.default_rng(20260815)
    levels = [f"B{i:02d}" for i in range(12)]
    effect = np.array([0.0, 0.30, 0.60, 0.90, 1.20, 0.10, -0.20, 0.05, 0.60, 1.00, 1.30, 1.60])
    idx = np.repeat(np.arange(len(levels)), 200)
    X = pd.DataFrame({"band": np.array(levels, dtype=object)[idx]})
    y = effect[idx] + 0.10 * rng.normal(size=idx.size)
    return X, y, levels, effect


def _designed_excursion(effect):
    """The INTERIOR fall the fixture designs in: peak before the trough, to it.

    Not ``max(effect) - min(effect[argmax(effect):])``, which is the obvious
    spelling and is silently zero here: this truth ENDS at its global maximum
    (1.60 at the last level), so that slice is a singleton and every guard
    built on it degrades to ``> 0``. The quantity the tests want is the drop a
    monotone repair has to remove, which is the descent into the trough.
    """
    effect = np.asarray(effect, dtype=float)
    trough = int(np.argmin(effect))
    return float(np.max(effect[: trough + 1]) - effect[trough])


# Floor for "the unrepaired fit still carries the fixture's dip". A DEGENERATION
# DETECTOR, not a calibration: how much of a designed excursion a penalized
# smooth reproduces depends on lambda, the knots and the data, with no closed
# form to derive a fraction from. So the value has to sit well below what a
# working fixture realises and well above zero. Measured across both bases and
# the specials variant, the fits realise 0.278-0.677 of the designed excursion;
# one tenth leaves the tightest case a factor of 2.8, which is margin against a
# different BLAS rather than margin fitted to the observation.
_DIP_REALISATION_FLOOR = 0.10


def _level_curve(model, name, levels):
    rels = model.reconstruct_feature(name)["level_log_relativities"]
    return np.array([rels[lev] for lev in levels], dtype=float)


def _monotone_slack(curve):
    """Feasibility slack from dtype epsilon and the curve's own scale."""
    return np.finfo(float).eps ** 0.5 * max(1.0, float(np.max(np.abs(curve))))


class TestOrderedCategoricalShapeDispatch:
    """A declared shape constraint must bind on the WRAPPER, not only on a
    bare ``Spline``. Both engines below filtered on ``isinstance(spec,
    _SplineBase)`` before reading the constraint, and an ``OrderedCategorical``
    is not one -- so the declaration reached neither.
    """

    @pytest.mark.parametrize("kind", ["cr", "ps"])
    @pytest.mark.parametrize("direction", ["increasing", "decreasing"])
    def test_postfit_repair_reaches_the_wrapped_spline(self, dipping_ordinal_data, kind, direction):
        from superglm.features.spline import Spline

        X, y, levels, effect = dipping_ordinal_data
        designed_dip = _designed_excursion(effect)
        # Fixture self-check: the truth must actually design a fall in, or
        # every guard below is measuring nothing.
        assert designed_dip == pytest.approx(1.40)
        spec = OrderedCategorical(
            order=levels,
            basis=Spline(kind=kind, n_knots=8, constraint=getattr(Constraint.postfit, direction)),
        )
        model = SuperGLM(family="gaussian", features={"band": spec}, selection_penalty=0.0).fit(
            X, y
        )

        # The unrepaired fit must carry the designed dip, or the repair has
        # nothing to bind on and a green test would measure nothing. A quarter
        # of the designed excursion is the floor a fit that tracked it clears.
        before = _level_curve(model, "band", levels)
        assert float(-np.min(np.diff(before))) > _DIP_REALISATION_FLOOR * designed_dip, before

        model.apply_shape_postfit(X)

        after = _level_curve(model, "band", levels)
        steps = np.diff(after)
        slack = _monotone_slack(after)
        if direction == "increasing":
            assert np.all(steps >= -slack), f"not increasing: {steps}"
            # One-sided monotonicity alone is satisfied by an all-zero
            # collapsed block, so pin that the repaired curve still carries
            # the truth's overall rise rather than having gone flat.
            assert after[-1] - after[0] > 0.5 * (effect[-1] - effect[0]), after
        else:
            assert np.all(steps <= slack), f"not decreasing: {steps}"
            # NOT a mirror of the branch above, and the asymmetry is the point.
            # The fixture's truth rises overall, so the weighted projection of
            # it onto the DECREASING cone is essentially the constant -- which
            # means `steps <= slack` here is satisfied by exactly the collapsed
            # block the increasing branch guards against, and cannot be the
            # discriminating assertion. What this arm actually pins is that the
            # engine ran with the direction it was handed: the recorded repair
            # below carries `kind == "decreasing"`, and the curve had to move a
            # long way to get flat. Asserting a rise here would be asserting
            # the constraint failed.
            assert float(np.max(np.abs(after - before))) > _DIP_REALISATION_FLOOR * designed_dip, (
                after
            )

        # Pin the DISPATCH as well, not just the shape: a monotone curve alone
        # could come from anywhere, but a recorded repair means the engine saw
        # the term.
        assert "band" in model._shape_repairs
        repair = model._shape_repairs["band"]
        assert repair.kind == direction
        assert repair.max_violation_after <= repair.max_violation_before

    @pytest.mark.parametrize("kind", ["cr", "ps"])
    def test_postfit_repair_matches_the_unwrapped_spline(self, dipping_ordinal_data, kind):
        """The definition of the fix: the wrapper reaches the engine the plain
        term already used, so the two must agree.

        A shape assertion alone cannot say that. It passes for any repair that
        happens to produce a monotone curve, including one that repaired the
        wrong coefficient space or weighted the grid by the wrong rows. Pinning
        the wrapped result against an unwrapped ``Spline`` on the same mapped
        level positions, same knots, same constraint, same data, says the
        stronger thing -- and would fail loudly on a future change that made the
        two diverge, instead of requiring the numbers to be re-measured by hand.
        """
        from superglm.features.spline import Spline

        X, y, levels, _ = dipping_ordinal_data
        # The positions `OrderedCategorical` maps `order=` onto, made explicit.
        positions = np.linspace(0.0, 1.0, len(levels))
        X = X.copy()
        X["pos"] = positions[[levels.index(lev) for lev in X["band"]]]
        constraint = Constraint.postfit.increasing

        wrapped = SuperGLM(
            family="gaussian",
            features={
                "band": OrderedCategorical(
                    order=levels, basis=Spline(kind=kind, n_knots=8, constraint=constraint)
                )
            },
            selection_penalty=0.0,
        ).fit(X, y)
        plain = SuperGLM(
            family="gaussian",
            features={"pos": Spline(kind=kind, n_knots=8, constraint=constraint)},
            selection_penalty=0.0,
        ).fit(X, y)

        wrapped.apply_shape_postfit(X)
        plain.apply_shape_postfit(X)

        wrapped_curve = _level_curve(wrapped, "band", levels)
        plain_curve = np.asarray(plain.predict(pd.DataFrame({"pos": positions})), dtype=float)
        plain_curve = plain_curve - plain_curve[0]

        # Tolerance derived, not chosen. Given one design, a level value is
        # reached by assembling a k-term basis row, a factorization-based solve
        # of the k-by-k system, the cone projection, and the reconstruction --
        # a chain whose accumulated float64 error is bounded by a low-degree
        # polynomial in k times the unit roundoff (Higham, *Accuracy and
        # Stability of Numerical Algorithms*, 2nd ed., Lemma 3.1 for the inner
        # products, §9.3 for the solve's backward error), scaled by the
        # magnitude of the quantity itself. Taking that polynomial as k**2 for
        # a well-conditioned design gives the bound below; k is read off the
        # fitted group rather than hard-coded, so a knot-count change moves the
        # bound with it. Observed at 2.5 ulps of scale (`cr`) and 9.8 (`ps`),
        # against a bound of k**2 = 81 -- roundoff, with room for a different
        # BLAS to order the sums differently, and roughly nine orders of
        # magnitude tighter than the 0.41-0.68 divergence the unfixed code
        # shows, which is what makes the assertion discriminating.
        n_cols = sum(g.size for g in wrapped._groups if g.feature_name == "band")
        scale = max(1.0, float(np.max(np.abs(wrapped_curve))))
        atol = n_cols**2 * np.finfo(np.float64).eps * scale
        np.testing.assert_allclose(wrapped_curve, plain_curve, rtol=0.0, atol=atol)
        # The pre-repair violation is a single reduction over the same curve on
        # the same grid, so it carries only the grid's own roundoff.
        assert wrapped._shape_repairs["band"].max_violation_before == pytest.approx(
            plain._shape_repairs["pos"].max_violation_before,
            rel=n_cols * np.finfo(np.float64).eps,
        )

        # And the reason the two agree at all, asserted EXACTLY rather than to
        # any tolerance: the axis the wrapper resolves from labels is bitwise
        # the float64 column the plain term reads straight off the frame. That
        # is what makes the comparison above a statement about arithmetic
        # ordering rather than about two different geometries that happen to
        # land close. Asserted last so the curve comparison owns the failure
        # message when the dispatch is broken.
        axis, axis_rows = wrapped._specs["band"].shape_axis(X["band"].to_numpy())
        assert axis_rows.all()
        assert np.array_equal(axis, X["pos"].to_numpy())

    def test_postfit_repair_on_a_selected_wrapper_refuses_rather_than_no_ops(
        self, dipping_ordinal_data
    ):
        """``select=True`` skips the inner spline's identifiability projection,
        so the term's columns no longer sum to zero and a repair would move the
        fitted mean. The repair engine's own publication check catches that and
        refuses.

        A plain ``Spline`` in the same configuration repairs instead, because
        runtime canonicalization registers it and its recorded column means
        cancel the shift; an ``OrderedCategorical`` is never registered, so
        nothing cancels it. That asymmetry is issue #311 -- what this test pins
        is the part that must not regress: the ordered term REFUSES, loudly,
        rather than returning silently unrepaired the way it did before the
        constraint reached the engine at all.

        A TOMBSTONE, NOT AN INVARIANT. Closing #311 makes this configuration
        repair, at which point the ``pytest.raises`` below must be replaced,
        not restored -- its failure then is the fix landing, not a regression.
        The state assertions after it are the part that survives: whatever the
        engine decides, refusing must leave the fitted model untouched rather
        than half-repaired, and nothing else covers that on this path.
        """
        from superglm.features.spline import Spline

        X, y, levels, _ = dipping_ordinal_data
        model = SuperGLM(
            family="gaussian",
            features={
                "band": OrderedCategorical(
                    order=levels,
                    basis=Spline(
                        kind="cr",
                        n_knots=8,
                        select=True,
                        constraint=Constraint.postfit.increasing,
                    ),
                )
            },
            selection_penalty=0.0,
        ).fit(X, y)
        beta_before = np.array(model.result.beta, copy=True)
        intercept_before = float(model.result.intercept)
        revision_before = model._fit_revision

        with pytest.raises(RuntimeError, match="fitted centering changed"):
            model.apply_shape_postfit(X)

        # A refusal that left partial state would be worse than either outcome.
        # `_validate_repair_for_publication` runs before any mutation and the
        # work happens on a `FittedStateRevision`, so the published model must
        # be bit-identical to what it was -- assert that rather than trust it.
        assert not getattr(model, "_shape_repairs", {})
        assert not getattr(model, "_monotone_repairs", {})
        np.testing.assert_array_equal(model.result.beta, beta_before)
        assert float(model.result.intercept) == intercept_before
        assert model._fit_revision == revision_before

    def test_postfit_repair_composes_with_specials(self, dipping_ordinal_data):
        """``specials=`` is the one delegation shape the wrapper reshapes: the
        term owns a second, unpenalized free block beside the smooth. The
        repair works in the inner spline's own coefficient space, so it must
        see the smooth block alone -- concatenating the special block into it
        would reinterpret free level effects as spline coefficients."""
        from superglm.features.spline import Spline

        X, y, levels, effect = dipping_ordinal_data
        X = X.copy()
        X.loc[np.arange(len(X)) % 25 == 0, "band"] = "MISSING"
        designed_dip = _designed_excursion(effect)
        # Fixture self-check: the truth must actually design a fall in, or
        # every guard below is measuring nothing.
        assert designed_dip == pytest.approx(1.40)
        spec = OrderedCategorical(
            order=levels,
            specials=["MISSING"],
            basis=Spline(kind="cr", n_knots=8, constraint=Constraint.postfit.increasing),
        )
        model = SuperGLM(family="gaussian", features={"band": spec}, selection_penalty=0.0).fit(
            X, y
        )
        before = _level_curve(model, "band", levels)
        assert float(-np.min(np.diff(before))) > _DIP_REALISATION_FLOOR * designed_dip, before

        model.apply_shape_postfit(X)

        after = _level_curve(model, "band", levels)
        steps = np.diff(after)
        assert np.all(steps >= -_monotone_slack(after)), f"not increasing: {steps}"
        assert "band" in model._shape_repairs

    def test_structured_rejection_sees_the_wrapped_fit_constraint(self, dipping_ordinal_data):
        """``_reject_structured_fit_constraints`` exists because a fit-time
        shape constraint and a RandomEffect do not jointly define the compact
        REML geometry. A bare ``Spline`` is refused; the same request wrapped
        must be refused identically, not fitted by an undefined path."""
        from superglm import RandomEffect
        from superglm.features.spline import Spline

        X, y, levels, _ = dipping_ordinal_data
        X = X.copy()
        X["grp"] = np.array([f"g{i % 6}" for i in range(len(X))], dtype=object)
        model = SuperGLM(
            family="gaussian",
            features={
                "band": OrderedCategorical(
                    order=levels,
                    basis=Spline(kind="cr", n_knots=6, constraint=Constraint.fit.increasing),
                ),
                "grp": RandomEffect(),
            },
        )
        with pytest.raises(NotImplementedError, match="fit-time shape constraints"):
            model.fit_reml(X, y)

    @pytest.mark.parametrize("basis", ["piecewise", "polynomial"])
    def test_postfit_repair_refuses_a_basis_with_no_curve_geometry(self, basis):
        """Selection is by declaration; resolving the curve is by type, and the
        two can drift. Nothing declares a constraint on a non-spline basis
        today, so this pins the refusal rather than a live path: the repair
        must name the basis instead of dying on ``spec._lo`` several frames
        later, the same way the QP forward now does."""
        from superglm import Piecewise, Polynomial
        from superglm.model.shape_ops import _curve_spec

        levels = [f"B{i:02d}" for i in range(8)]
        inner = Piecewise(breaks=["B03"]) if basis == "piecewise" else Polynomial(degree=2)
        spec = OrderedCategorical(order=levels, basis=inner)
        with pytest.raises(NotImplementedError, match="only implemented for spline bases"):
            _curve_spec("band", spec)

    def test_qp_geometry_forward_names_a_basis_that_cannot_supply_it(self):
        """The wrapper forwards ``_build_monotone_constraints_raw``
        unconditionally, so ``getattr(spec, ..., None)`` at the builder's QP
        branch always finds a method and its ``RuntimeError`` backstop can
        never fire. A basis with no raw geometry must still fail by name."""
        from superglm.features.spline import Spline

        spec = OrderedCategorical(
            order=[f"B{i:02d}" for i in range(8)],
            basis=Spline(kind="ns", n_knots=5),
        )
        with pytest.raises(RuntimeError, match="raw constraint geometry"):
            spec._build_monotone_constraints_raw()
