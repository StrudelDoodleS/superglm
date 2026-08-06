"""Tests for the Piecewise feature spec: hat basis, validation, coefficient meaning.

Two of the four properties the design rests on are pinned here: the coefficient
meaning (``v_j == f(t_j) - f(t_r)``) and linear extrapolation past the boundary
knots.  Export exactness and edit locality belong to the later stages.

Tolerances are derived, not chosen: every quantity compared below comes out of a
dot product with exactly two non-zero terms (a hat basis row has two non-zero
entries), so ``_dot_atol`` bounds the round-off at the magnitude actually in
play.  It evaluates to ~1e-15 at unit magnitude, comfortably inside the 1e-12
bar the design asks for, and the observed error on every case is 0.0.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from superglm import Categorical, Piecewise, SuperGLM
from superglm.features.piecewise import _SMALL_SEGMENT_WEIGHT_FRACTION, _STRATEGIES
from tests._piecewise_cases import CASE_NAMES, make_case

_EPS = float(np.finfo(np.float64).eps)
_NONZEROS_PER_ROW = 2


def _dot_atol(magnitude: float) -> float:
    """Round-off bound for a two-term dot product at the given magnitude."""
    return _NONZEROS_PER_ROW * 4.0 * _EPS * max(float(magnitude), 1.0)


def _fitted(name: str):
    """Fit one fixture case and return (spec, beta, knots)."""
    case = make_case(name)
    model = SuperGLM(
        features={"x": case.spec, "region": Categorical(base="first")},
    )
    model.fit(case.X, case.y, sample_weight=case.sample_weight)
    spec = model._specs["x"]
    group = next(g for g in model._groups if g.feature_name == "x")
    beta = np.asarray(model._result.beta[group.start : group.end], dtype=np.float64)
    return spec, beta, spec._knots


def _spec_on(x, breaks, **kwargs) -> Piecewise:
    """Build a spec on *x* and return it."""
    spec = Piecewise(breaks, **kwargs)
    spec.build(np.asarray(x, dtype=np.float64))
    return spec


class TestHatBasis:
    def test_hat_basis_at_the_knots_is_exactly_the_identity(self):
        """Handles land on knots and the editor recovers coefficients exactly."""
        spec = _spec_on(np.linspace(0.0, 10.0, 41), [1.0, 4.0], lower=0.0, upper=10.0)
        np.testing.assert_array_equal(spec._hat_basis(spec._knots), np.eye(spec._knots.size))

    def test_basis_rows_sum_to_one_inside_and_in_both_tails(self):
        """Partition of unity survives extrapolation, which the drop-a-knot algebra needs."""
        spec = _spec_on(np.linspace(0.0, 10.0, 41), [1.0, 4.0], lower=0.0, upper=10.0)
        t = spec._knots
        below = np.array([t[0] - 7.5, t[0] - 0.25])
        above = np.array([t[-1] + 0.25, t[-1] + 7.5])
        inside = np.array([0.5, 2.0, 4.0, 9.75])
        rows = spec._raw_basis_matrix(np.concatenate([below, inside, above]))
        np.testing.assert_allclose(rows.sum(axis=1), 1.0, rtol=0.0, atol=_dot_atol(8.0))

    def test_build_reports_j_plus_one_unpenalised_columns(self):
        info = Piecewise([1.0, 4.0], lower=0.0, upper=10.0).build(np.linspace(0.0, 10.0, 41))
        assert info.n_cols == 3
        assert info.columns.shape == (41, 3)
        assert info.penalty_matrix is None
        assert info.reparametrize is False
        assert info.penalized is True

    @pytest.mark.parametrize("case_name", CASE_NAMES)
    def test_transform_equals_build_columns(self, case_name):
        """Section 9 property 2 in its load-bearing form: fit and predict share columns.

        Swept over the whole matrix rather than run on one convenient shape.  A
        ``build``/``transform`` disagreement about WHICH hat is dropped is
        invisible wherever the base resolves to knot 0, and that disagreement is
        exactly what makes ``model.predict`` score through a different basis than
        the fit used -- the deviance is unchanged, the coefficients stop meaning
        ``f(t_j) - f(t_base)``, and predictions move by more than 100%.
        """
        case = make_case(case_name)
        x = case.X["x"].to_numpy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            info = case.spec.build(x, sample_weight=case.sample_weight)
        np.testing.assert_array_equal(case.spec.transform(x), info.columns)

    def test_the_case_matrix_moves_the_base_off_column_zero(self):
        """Guard for the sweep above: on a base at knot 0 that property is vacuous."""
        indices = set()
        for case_name in CASE_NAMES:
            case = make_case(case_name)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                case.spec.build(case.X["x"].to_numpy(), sample_weight=case.sample_weight)
            indices.add(int(case.spec._base_index))
        assert indices - {0}, (
            "every fixture resolved its base to knot 0, so a build/transform "
            f"disagreement about the dropped hat would pass unseen. Base indices: {indices}"
        )

    def test_score_equals_transform_at_beta(self):
        x = np.linspace(0.0, 10.0, 41)
        spec = _spec_on(x, [1.0, 4.0], lower=0.0, upper=10.0)
        beta = np.array([0.3, -0.2, 0.7])
        np.testing.assert_array_equal(spec.score(x, beta), spec.transform(x) @ beta)

    def test_raw_basis_matrix_keeps_the_base_column(self):
        """The editor duck-types on this hook and needs a handle at every knot."""
        x = np.linspace(0.0, 10.0, 41)
        spec = _spec_on(x, [1.0, 4.0], lower=0.0, upper=10.0, base=1.0)
        raw = spec._raw_basis_matrix(x)
        assert raw.shape == (41, spec._knots.size)
        assert raw.shape[1] == spec.transform(x).shape[1] + 1
        np.testing.assert_array_equal(raw[:, spec._non_base_indices], spec.transform(x))


class TestCoefficientMeaning:
    """Spec section 9 property 2: each coefficient is f(t_j) - f(t_r)."""

    @pytest.mark.parametrize("case_name", CASE_NAMES)
    def test_coefficients_are_log_relativities_against_the_base_knot(self, case_name):
        spec, beta, knots = _fitted(case_name)
        # Independent route to the same numbers: evaluate the fitted function at
        # the knots through score(), then difference against the base knot.
        at_knots = spec.score(knots, beta)
        assert at_knots[spec._base_index] == 0.0
        expected = at_knots[spec._non_base_indices] - at_knots[spec._base_index]
        atol = _dot_atol(np.max(np.abs(beta)))
        assert atol <= 1e-12
        np.testing.assert_allclose(beta, expected, rtol=0.0, atol=atol)


class TestLinearExtrapolation:
    """Spec section 9 property 4: two points a side, so a slope error cannot pass."""

    @pytest.mark.parametrize("case_name", CASE_NAMES)
    def test_predictions_beyond_the_boundary_knots_stay_on_the_boundary_lines(self, case_name):
        spec, beta, knots = _fitted(case_name)
        at_knots = spec.score(knots, beta)
        width_lo = float(knots[1] - knots[0])
        width_hi = float(knots[-1] - knots[-2])
        slope_lo = (at_knots[1] - at_knots[0]) / width_lo
        slope_hi = (at_knots[-1] - at_knots[-2]) / width_hi

        below = knots[0] - np.array([0.5, 3.0]) * width_lo
        above = knots[-1] + np.array([0.5, 3.0]) * width_hi
        got = spec.score(np.concatenate([below, above]), beta)
        want = np.concatenate(
            [
                at_knots[0] + slope_lo * (below - knots[0]),
                at_knots[-1] + slope_hi * (above - knots[-1]),
            ]
        )
        # The tail rows carry basis entries up to 4 in magnitude, so the
        # round-off bound scales with the largest product actually formed.
        atol = _dot_atol(4.0 * np.max(np.abs(beta)))
        assert atol <= 1e-12
        np.testing.assert_allclose(got, want, rtol=0.0, atol=atol)


class TestValidationRules:
    """One test per rule, each naming the rule it pins."""

    def test_rule_1_non_finite_x_raises(self):
        with pytest.raises(ValueError, match="requires finite x"):
            Piecewise([2.0]).build(np.array([1.0, np.nan, 3.0]))

    def test_rule_1_non_finite_x_raises_at_transform_too(self):
        spec = _spec_on([1.0, 2.0, 3.0], [2.0])
        with pytest.raises(ValueError, match="requires finite x"):
            spec.transform(np.array([1.0, np.inf]))

    def test_rule_1_sample_weight_must_match_x(self):
        with pytest.raises(ValueError, match="sample_weight must have length 3"):
            Piecewise([2.0]).build(np.array([1.0, 2.0, 3.0]), sample_weight=np.ones(2))

    def test_rule_2_empty_breaks_points_at_numeric(self):
        with pytest.raises(ValueError, match="Use Numeric"):
            Piecewise([]).build(np.linspace(0.0, 10.0, 11))

    def test_rule_2_zero_int_breaks_points_at_numeric(self):
        with pytest.raises(ValueError, match="Use Numeric"):
            Piecewise(0).build(np.linspace(0.0, 10.0, 11))

    def test_rule_3_unknown_strategy_names_the_supported_set(self):
        with pytest.raises(ValueError, match=r"strategy must be one of \['quantile'\]"):
            Piecewise([2.0], strategy="kmeans").build(np.linspace(0.0, 10.0, 11))

    def test_rule_3_default_strategy_is_the_only_supported_one(self):
        """The seam is reserved, not opened: adding a strategy is an API decision."""
        assert _STRATEGIES == frozenset({"quantile"})
        assert Piecewise([2.0]).strategy == "quantile"

    def test_rule_4_sequence_breaks_must_strictly_increase(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            Piecewise([4.0, 2.0]).build(np.linspace(0.0, 10.0, 11))

    def test_rule_4_duplicate_breaks_are_a_zero_width_segment(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            Piecewise([2.0, 2.0]).build(np.linspace(0.0, 10.0, 11))

    def test_rule_5_int_breaks_warn_when_ties_collapse_the_request(self):
        case = make_case("heaped_int_x")
        x = case.X["x"].to_numpy()
        with pytest.warns(UserWarning, match="8 requested, 6 realised"):
            info = case.spec.build(x, sample_weight=case.sample_weight)
        realised = case.spec._knots.size - 2
        assert realised == 6
        assert info.n_cols == realised + 1
        assert case.spec._n_breaks_requested == 8

    def test_rule_5_int_breaks_raise_when_nothing_lands_inside(self):
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        with pytest.raises(ValueError, match="realised no breakpoints"):
            Piecewise(3, lower=1.4, upper=1.6).build(x)

    def test_rule_6a_lower_must_be_below_upper(self):
        with pytest.raises(ValueError, match="lower < upper"):
            Piecewise([2.0], lower=5.0, upper=1.0).build(np.linspace(0.0, 10.0, 11))

    def test_rule_6b_breaks_must_sit_strictly_inside_the_range(self):
        with pytest.raises(ValueError, match="strictly inside"):
            Piecewise([10.0], lower=0.0, upper=10.0).build(np.linspace(0.0, 10.0, 11))

    def test_rule_7_a_float_base_must_name_exactly_one_knot(self):
        with pytest.raises(ValueError, match="must equal exactly one knot"):
            Piecewise([2.0, 5.0], base=3.0, lower=0.0, upper=10.0).build(np.linspace(0.0, 10.0, 11))

    def test_rule_7_an_unknown_base_keyword_names_the_alternatives(self):
        with pytest.raises(ValueError, match="base must be 'most_exposed'"):
            Piecewise([2.0], base="middle").build(np.linspace(0.0, 10.0, 11))

    def test_rule_7_most_exposed_without_weights_falls_back_to_the_first_knot(self):
        """Mirrors Categorical: with no exposure there is nothing to be most of."""
        spec = _spec_on(np.linspace(0.0, 10.0, 11), [2.0, 5.0], lower=0.0, upper=10.0)
        assert spec._base_index == 0

    def test_rule_7_most_exposed_picks_the_heaviest_knot(self):
        x = np.concatenate([np.full(50, 5.0), np.array([0.0, 2.0, 10.0])])
        w = np.concatenate([np.full(50, 3.0), np.array([1.0, 1.0, 1.0])])
        spec = Piecewise([2.0, 5.0], lower=0.0, upper=10.0)
        spec.build(x, sample_weight=w)
        assert float(spec._knots[spec._base_index]) == 5.0

    def test_rule_8_a_knot_with_zero_hat_mass_raises(self):
        """breaks=[2] with data only at {1, 3} leaves knot 2 with an empty column."""
        with pytest.raises(ValueError, match="zero hat mass"):
            Piecewise([2.0]).build(np.array([1.0, 1.0, 3.0, 3.0]))

    def test_rule_9_a_segment_with_no_weight_raises_and_reports_every_segment(self):
        with pytest.raises(ValueError, match="carry no in-range weight") as excinfo:
            Piecewise([1.0, 2.0], lower=0.0, upper=3.0).build(np.array([0.5, 0.5, 2.5, 2.5]))
        assert "Per-segment weight" in str(excinfo.value)

    def test_rule_9_does_not_credit_a_segment_with_rows_from_outside_the_range(self):
        """A rated range bracketing no observation at all must not build.

        ``_segment_index`` clamps out-of-range rows into the boundary segments,
        which is what continues the boundary line; counting those clamped rows
        as support is what let ``[45, 55]`` pass on data living entirely in
        ``[0, 10] u [90, 100]``.
        """
        x = np.concatenate([np.linspace(0.0, 10.0, 300), np.linspace(90.0, 100.0, 300)])
        with pytest.raises(ValueError, match="carry no in-range weight") as excinfo:
            Piecewise([50.0], lower=45.0, upper=55.0, base="first").build(x)
        message = str(excinfo.value)
        # Both segments are named, and the clamped weight is reported as what it
        # is rather than folded into the per-segment counts.
        assert "[45, 50]" in message and "[50, 55]" in message
        assert "[0, 0]" in message
        assert "lies outside [45, 55]" in message

    def test_rule_11_counts_only_in_range_weight_for_a_boundary_segment(self):
        """The thin-segment diagnostic must see through the clamp too.

        A boundary segment holding two in-range rows is credited with the whole
        out-of-range tail unless the clamped rows are excluded, which silences
        the warning about the segment whose slope those rows actually set.
        """
        x = np.concatenate(
            [np.full(400, 5.0), np.array([26.0, 27.0]), np.linspace(30.0, 60.0, 600)]
        )
        with pytest.warns(UserWarning, match="of the in-range weight") as record:
            Piecewise([30.0], lower=25.0, upper=60.0, base="first").build(x)
        message = str(record[0].message)
        assert "[25, 30]" in message
        assert "[2, 600]" in message
        assert "lies outside [25, 60]" in message

    def test_rule_10_rank_deficiency_is_caught_where_rules_8_and_9_stay_silent(self):
        """The discriminating fixture: every column has mass, every segment has data."""
        x = np.array([0.0, 1.5, 3.0])
        spec = Piecewise([1.0, 2.0], base="first", lower=0.0, upper=3.0)
        # Rules 8 and 9 are provably silent here, so a rank failure is the only
        # thing this fixture can report.
        probe = Piecewise([1.0, 2.0], base="first", lower=0.0, upper=3.0)
        probe._knots = np.array([0.0, 1.0, 2.0, 3.0])
        basis = probe._hat_basis(x)
        assert np.all(np.abs(basis).sum(axis=0) > 0.0)
        assert np.all(np.bincount(probe._segment_index(x), minlength=3) > 0)
        with pytest.raises(ValueError, match="rank deficient"):
            spec.build(x)

    def test_rule_10_measures_the_retained_columns_against_the_intercept(self):
        """One distinct x per segment: the exact degeneracy the rule exists for.

        The retained columns are independent OF EACH OTHER here, so a rank check
        confined to them passes; it is the intercept they are collinear with.
        Fitting this design reports a coefficient with an SE of order 1e13.
        """
        x = np.concatenate([np.full(50, 0.5), np.full(50, 1.5)])
        spec = Piecewise([1.0], lower=0.0, upper=2.0, base="first")
        probe = Piecewise([1.0], lower=0.0, upper=2.0, base="first")
        probe._knots = np.array([0.0, 1.0, 2.0])
        probe._non_base_indices = np.array([1, 2], dtype=np.intp)
        retained = probe._hat_basis(x)[:, probe._non_base_indices]
        assert np.linalg.matrix_rank(retained) == retained.shape[1]
        assert np.linalg.matrix_rank(np.column_stack([np.ones(x.size), retained])) == 2
        with pytest.raises(ValueError, match="rank deficient against the intercept"):
            spec.build(x)

    def test_rule_11_a_thin_segment_warns_and_reports_every_segment(self):
        x = np.concatenate([np.array([0.5]), np.linspace(1.01, 2.0, 1000)])
        with pytest.warns(UserWarning, match="of the in-range weight") as record:
            Piecewise([1.0], lower=0.0, upper=2.0).build(x)
        assert "Per-segment weight" in str(record[0].message)

    def test_rule_11_threshold_is_a_named_constant(self):
        assert _SMALL_SEGMENT_WEIGHT_FRACTION == 0.005


class TestSpecState:
    def test_transform_before_build_names_the_missing_step(self):
        with pytest.raises(RuntimeError, match="call build\\(\\)"):
            Piecewise([2.0]).transform(np.array([1.0]))

    def test_repr_before_and_after_build(self):
        spec = Piecewise([1.0, 2.0])
        assert repr(spec) == "Piecewise(breaks=[1, 2], base='most_exposed')"
        spec.build(np.linspace(0.0, 3.0, 7))
        assert "4 knots" in repr(spec)
        assert "ref=0" in repr(spec)

    def test_a_refit_keeps_the_base_knot_it_already_chose(self):
        """A CV fold must not silently redefine what every coefficient means."""
        x = np.concatenate([np.full(50, 5.0), np.array([0.0, 2.0, 10.0])])
        w = np.concatenate([np.full(50, 3.0), np.array([1.0, 1.0, 1.0])])
        spec = Piecewise([2.0, 5.0], lower=0.0, upper=10.0)
        spec.build(x, sample_weight=w)
        assert float(spec._knots[spec._base_index]) == 5.0
        # Second fit whose exposure now sits on knot 2.
        w2 = np.concatenate([np.full(50, 0.01), np.array([1.0, 100.0, 1.0])])
        spec.build(x, sample_weight=w2)
        assert float(spec._knots[spec._base_index]) == 5.0

    def test_reconstruct_returns_knot_relativities_and_derived_slopes(self):
        spec = _spec_on(np.linspace(0.0, 10.0, 41), [1.0, 4.0], lower=0.0, upper=10.0, base=1.0)
        rec = spec.reconstruct(np.array([0.2, -0.1, 0.5]))
        np.testing.assert_array_equal(rec["knots"], np.array([0.0, 1.0, 4.0, 10.0]))
        assert rec["base_knot"] == 1.0
        assert rec["base_index"] == 1
        np.testing.assert_array_equal(rec["log_relativity"], np.array([0.2, 0.0, -0.1, 0.5]))
        np.testing.assert_allclose(rec["relativity"], np.exp(rec["log_relativity"]))
        expected = np.array([(0.0 - 0.2) / 1.0, (-0.1 - 0.0) / 3.0, (0.5 - -0.1) / 6.0])
        np.testing.assert_allclose(rec["slopes"], expected, rtol=0.0, atol=_dot_atol(1.0))
        assert rec["boundary_slopes"] == (expected[0], expected[-1])


class TestPiecewiseIntegration:
    def test_a_model_with_a_piecewise_term_fits_and_predicts(self):
        """No dm_builder change: the spec builds polymorphically like any other."""
        case = make_case("interior_base")
        model = SuperGLM(
            features={"x": case.spec, "region": Categorical(base="first")},
        )
        model.fit(case.X, case.y, sample_weight=case.sample_weight)
        preds = model.predict(case.X)
        assert preds.shape == (len(case.X),)
        assert np.all(preds > 0)
        # The model copies the spec before fitting, so fitted knots live on
        # model._specs, not on the spec the caller handed in.
        group = next(g for g in model._groups if g.feature_name == "x")
        assert group.end - group.start == model._specs["x"]._knots.size - 1
        assert case.spec._knots.size == 0

    def test_predictions_agree_with_the_piecewise_contribution(self):
        """model.predict routes the term through score(), so the two must agree."""
        case = make_case("unequal_widths")
        model = SuperGLM(features={"x": case.spec})
        model.fit(case.X, case.y, sample_weight=case.sample_weight)
        spec = model._specs["x"]
        group = next(g for g in model._groups if g.feature_name == "x")
        beta = np.asarray(model._result.beta[group.start : group.end], dtype=np.float64)
        grid = pd.DataFrame({"x": np.array([-5.0, 0.0, 7.5, 60.0, 130.0])})
        contribution = spec.score(grid["x"].to_numpy(), beta)
        ratio = model.predict(grid) / np.exp(contribution)
        np.testing.assert_allclose(ratio, ratio[0], rtol=1e-12, atol=0.0)
