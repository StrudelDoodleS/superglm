"""Tests for the Piecewise feature spec: hat basis, validation, coefficient meaning.

Two of the four properties the design rests on are pinned here: the coefficient
meaning (``v_j == f(t_j) - f(t_r)``) and the extrapolation contract past the
boundary knots (flat under the ``"clip"`` default, the boundary lines under
``"extend"``, a refusal under ``"error"``).  Export exactness and edit locality
belong to the later stages.

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
import scipy.sparse as sp

from superglm import Categorical, Piecewise, SuperGLM
from superglm.features.piecewise import _SMALL_SEGMENT_WEIGHT_FRACTION, _STRATEGIES
from tests._piecewise_cases import CASE_NAMES, make_case

_EPS = float(np.finfo(np.float64).eps)
_NONZEROS_PER_ROW = 2


def _dot_atol(magnitude: float) -> float:
    """Round-off bound for a two-term dot product at the given magnitude."""
    return _NONZEROS_PER_ROW * 4.0 * _EPS * max(float(magnitude), 1.0)


def _fitted(name: str, extrapolation: str | None = None):
    """Fit one fixture case and return (spec, beta, knots)."""
    case = make_case(name, extrapolation=extrapolation)
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

    def test_build_columns_are_sparse_with_at_most_two_nonzeros_per_row(self):
        """The design is emitted CSR so the builder can dedup the repeated rows.

        At most two non-zeros per row is the hat locality stated on the storage
        itself; the bound is what makes the sparse route cheaper than dense at
        any knot count.
        """
        x = np.linspace(0.0, 10.0, 41)
        spec = Piecewise([1.0, 4.0], lower=0.0, upper=10.0)
        info = spec.build(x)
        assert sp.issparse(info.columns)
        assert info.columns.format == "csr"
        per_row = np.diff(info.columns.indptr)
        assert per_row.max() <= 2
        np.testing.assert_array_equal(info.columns.toarray(), spec.transform(x))

    def test_a_heaped_fit_stores_the_design_one_distinct_row_deep(self):
        """The fit-time representation collapses repeated rows without binning.

        Heaped x is the rating-variable norm, so thousands of rows carry a few
        dozen distinct basis rows; the builder's dedup gate must accept them
        and the block must stay unpenalized (``omega is None``) -- the compressed
        container is SSP-shaped, and an identity reparameterisation with no
        penalty is exactly a fixed-df design stored one distinct row deep.
        """
        from superglm.group_matrix import SupportCompressedSSPGroupMatrix

        rng = np.random.default_rng(31)
        levels = np.arange(6.0, 126.0, 6.0)
        x = rng.choice(levels, 4000)
        x[: levels.size] = levels  # every level present, endpoints pinned
        y = rng.poisson(1.0, x.size).astype(np.float64)
        model = SuperGLM(features={"x": Piecewise([42.0, 60.0, 66.0])})
        model.fit(pd.DataFrame({"x": x}), y)

        idx = next(i for i, g in enumerate(model._groups) if g.feature_name == "x")
        gm = model._dm.group_matrices[idx]
        assert isinstance(gm, SupportCompressedSSPGroupMatrix)
        assert gm.is_lossless_support
        assert gm.omega is None
        assert gm.omega_components is None
        assert gm.n_bins <= levels.size

    def test_a_small_fit_declines_compression_and_stays_plain_sparse(self):
        """Below the calibrated-rows floor the dedup gate declines by design."""
        from superglm.group_matrix import SparseGroupMatrix

        rng = np.random.default_rng(33)
        x = np.linspace(0.0, 10.0, 41)
        model = SuperGLM(features={"x": Piecewise([2.0, 5.0], lower=0.0, upper=10.0)})
        model.fit(pd.DataFrame({"x": x}), rng.poisson(1.0, x.size).astype(np.float64))

        idx = next(i for i, g in enumerate(model._groups) if g.feature_name == "x")
        assert isinstance(model._dm.group_matrices[idx], SparseGroupMatrix)

    @pytest.mark.parametrize("case_name", CASE_NAMES)
    def test_the_emitted_design_is_bit_identical_to_the_dense_hat_path(self, case_name):
        """build() emits CSR straight from the two hat entries per row.

        The historical path went through a dense (n_unique, J+2) basis and
        ``sp.csr_matrix(dense)``.  The direct emission must reproduce that
        matrix bit for bit -- values, indices and indptr -- because the
        builder's dedup gate groups rows by their raw float bits, and a
        last-ulp difference would silently change which rows merge.
        """
        case = make_case(case_name)
        x = case.X["x"].to_numpy(dtype=np.float64)
        spec = case.spec
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            info = spec.build(x, sample_weight=case.sample_weight)

        reference = sp.csr_matrix(spec.transform(x))
        emitted = info.columns
        assert emitted.has_sorted_indices
        np.testing.assert_array_equal(emitted.indptr, reference.indptr)
        np.testing.assert_array_equal(emitted.indices, reference.indices)
        np.testing.assert_array_equal(emitted.data, reference.data)

    def test_a_large_distinct_x_build_probes_rank_through_the_gram(self, monkeypatch):
        """Above the factor ceiling the probe switches to the tridiagonal Gram."""
        import superglm.features.piecewise as pw

        calls = {"gram": 0}
        original = Piecewise._weighted_gram

        def spy(self, seg, frac, weights):
            calls["gram"] += 1
            return original(self, seg, frac, weights)

        monkeypatch.setattr(Piecewise, "_weighted_gram", spy)
        x = np.linspace(0.0, 100.0, pw._RANK_PROBE_MAX_FACTOR_ROWS + 1)
        spec = Piecewise([25.0, 50.0, 75.0], lower=0.0, upper=100.0)
        info = spec.build(x)

        assert calls["gram"] == 1
        np.testing.assert_array_equal(info.columns.toarray(), spec.transform(x))

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
        np.testing.assert_array_equal(case.spec.transform(x), info.columns.toarray())

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


class TestExtrapolationModes:
    """Spec section 9 property 4, per mode: two points a side, so a slope error cannot pass."""

    def test_the_default_is_clip_and_a_typo_fails_where_it_is_written(self):
        assert Piecewise([2.0]).extrapolation == "clip"
        with pytest.raises(
            ValueError, match=r"extrapolation must be one of \('clip', 'extend', 'error'\)"
        ):
            Piecewise([2.0], extrapolation="linear")

    @pytest.mark.parametrize("case_name", CASE_NAMES)
    def test_extend_predictions_beyond_the_boundary_knots_stay_on_the_boundary_lines(
        self, case_name
    ):
        spec, beta, knots = _fitted(case_name, extrapolation="extend")
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

    @pytest.mark.parametrize("case_name", CASE_NAMES)
    def test_clip_predictions_beyond_the_boundary_knots_hold_the_boundary_values(self, case_name):
        """The default: beyond the knot span the term is exactly flat.

        Asserted as exact equality with the boundary-knot scores, not as
        closeness, because clip evaluates the identical basis row at the
        identical clamped point -- any difference at all is a policy leak.
        """
        spec, beta, knots = _fitted(case_name)
        assert spec.extrapolation == "clip"
        at_knots = spec.score(knots, beta)
        width_lo = float(knots[1] - knots[0])
        width_hi = float(knots[-1] - knots[-2])
        below = knots[0] - np.array([0.5, 3.0]) * width_lo
        above = knots[-1] + np.array([0.5, 3.0]) * width_hi
        got = spec.score(np.concatenate([below, above]), beta)
        np.testing.assert_array_equal(got[:2], np.full(2, at_knots[0]))
        np.testing.assert_array_equal(got[2:], np.full(2, at_knots[-1]))

    def test_clip_and_extend_agree_on_and_inside_the_boundary_knots(self):
        """The policy is a tail rule only: in-range predictions are mode-independent."""
        x = np.linspace(0.0, 10.0, 41)
        beta = np.array([0.4, -0.2, 0.7])
        clip = _spec_on(x, [1.0, 4.0], lower=0.0, upper=10.0)
        extend = _spec_on(x, [1.0, 4.0], lower=0.0, upper=10.0, extrapolation="extend")
        probe = np.array([0.0, 0.5, 1.0, 3.9, 4.0, 9.99, 10.0])
        np.testing.assert_array_equal(clip.score(probe, beta), extend.score(probe, beta))

    def test_error_mode_refuses_out_of_range_at_transform_and_scores_in_range(self):
        spec = _spec_on(
            np.linspace(0.0, 10.0, 41), [1.0, 4.0], lower=0.0, upper=10.0, extrapolation="error"
        )
        np.testing.assert_allclose(
            spec._raw_basis_matrix(np.array([0.0, 10.0])).sum(axis=1),
            [1.0, 1.0],
            rtol=0.0,
            atol=1e-15,
        )
        with pytest.raises(
            ValueError, match=r"outside the rated range \[0, 10\] with extrapolation='error'"
        ):
            spec.transform(np.array([5.0, 10.5]))

    def test_error_mode_refuses_a_narrower_pin_at_build(self):
        """The policy binds at build() too: rows the term will not rate refuse loudly."""
        with pytest.raises(
            ValueError, match=r"outside the rated range \[20, 80\] with extrapolation='error'"
        ):
            Piecewise([50.0], lower=20.0, upper=80.0, extrapolation="error").build(
                np.linspace(0.0, 100.0, 201)
            )

    def test_clip_mode_narrow_pins_fit_identically_to_a_precomputed_clip(self):
        """``Piecewise(breaks, upper=u)`` IS the tail-grouping idiom under clip.

        Exact equality, not closeness: the policy clamps x before the identical
        hat arithmetic, so both routes must produce the same bits.  This is the
        contract that lets a caller delete the ``x.clip(...)`` preprocessing
        column and state the grouping on the term instead.
        """
        x = np.concatenate([np.linspace(0.0, 100.0, 601), [120.0, 140.0, -10.0]])
        pinned = Piecewise([40.0, 60.0], lower=20.0, upper=80.0)
        manual = Piecewise([40.0, 60.0], lower=20.0, upper=80.0)
        info_pinned = pinned.build(x)
        info_manual = manual.build(np.clip(x, 20.0, 80.0))
        np.testing.assert_array_equal(pinned._knots, manual._knots)
        assert pinned._base_index == manual._base_index
        np.testing.assert_array_equal(info_pinned.columns.toarray(), info_manual.columns.toarray())
        probe = np.array([-10.0, 20.0, 45.0, 80.0, 140.0])
        np.testing.assert_array_equal(pinned.transform(probe), manual.transform(probe))


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

    def test_rule_7_most_exposed_without_weights_picks_the_most_rows_knot(self):
        """No weights means weights of one: hat-carried mass is the row count.

        The model API always hands ``build()`` an explicit weight vector (ones
        when the caller gave none), so a weights-absent fallback to the first
        knot was unreachable through any fit and made the direct spec API
        disagree with both the fitted behaviour and the documentation.
        """
        x = np.linspace(0.0, 10.0, 11)
        spec = _spec_on(x, [2.0, 5.0], lower=0.0, upper=10.0)
        mass = spec._raw_basis_matrix(x).sum(axis=0)
        assert spec._base_index == int(np.argmax(mass))
        assert spec._base_index != 0
        ones = Piecewise([2.0, 5.0], lower=0.0, upper=10.0)
        ones.build(x, sample_weight=np.ones(x.size))
        assert ones._base_index == spec._base_index

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

        Extend mode, because only extend has out-of-range rows at fit time.
        ``_segment_index`` clamps them into the boundary segments, which is
        what continues the boundary line; counting those clamped rows as
        support is what let ``[45, 55]`` pass on data living entirely in
        ``[0, 10] u [90, 100]``.
        """
        x = np.concatenate([np.linspace(0.0, 10.0, 300), np.linspace(90.0, 100.0, 300)])
        with pytest.raises(ValueError, match="carry no in-range weight") as excinfo:
            Piecewise([50.0], lower=45.0, upper=55.0, base="first", extrapolation="extend").build(x)
        message = str(excinfo.value)
        # Both segments are named, and the clamped weight is reported as what it
        # is rather than folded into the per-segment counts.
        assert "[45, 50]" in message and "[50, 55]" in message
        assert "[0, 0]" in message
        assert "lies outside [45, 55]" in message

    def test_rule_8_catches_the_same_fixture_under_clip_because_grouping_is_real_support(self):
        """The clip twin of the rule-9 test above, failing for the clip reason.

        Under clip the tail rows are genuinely grouped onto the boundary knots
        -- they support the boundary values, so neither boundary segment is
        empty -- and what is actually indefensible about breaks=[50] on data
        living in ``[0, 10] u [90, 100]`` is that no row is anywhere near 50:
        the interior knot's column is identically zero.
        """
        x = np.concatenate([np.linspace(0.0, 10.0, 300), np.linspace(90.0, 100.0, 300)])
        with pytest.raises(ValueError, match="carry zero hat mass") as excinfo:
            Piecewise([50.0], lower=45.0, upper=55.0, base="first").build(x)
        assert "[50]" in str(excinfo.value)

    def test_rule_11_counts_only_in_range_weight_for_a_boundary_segment(self):
        """The thin-segment diagnostic must see through the clamp too.

        Extend mode, because only extend has out-of-range rows at fit time.  A
        boundary segment holding two in-range rows is credited with the whole
        out-of-range tail unless the clamped rows are excluded, which silences
        the warning about the segment whose slope those rows actually set.
        """
        x = np.concatenate(
            [np.full(400, 5.0), np.array([26.0, 27.0]), np.linspace(30.0, 60.0, 600)]
        )
        with pytest.warns(UserWarning, match="of the in-range weight") as record:
            Piecewise([30.0], lower=25.0, upper=60.0, base="first", extrapolation="extend").build(x)
        message = str(record[0].message)
        assert "[25, 30]" in message
        assert "[2, 600]" in message
        assert "lies outside [25, 60]" in message

    def test_rule_11_stays_silent_on_the_same_fixture_under_clip(self):
        """The clip twin: grouping the tail onto the boundary knot feeds the segment.

        The 400 rows heaped at 5.0 land on the 25.0 knot, so the ``[25, 30]``
        segment carries 402 of 1002 rows and there is nothing thin to warn
        about.  This is the semantic difference between the modes stated as
        behaviour: clip turns tail exposure into boundary support, extend turns
        it into boundary-slope leverage.
        """
        x = np.concatenate(
            [np.full(400, 5.0), np.array([26.0, 27.0]), np.linspace(30.0, 60.0, 600)]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            Piecewise([30.0], lower=25.0, upper=60.0, base="first").build(x)

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

    def test_rule_10_the_gram_backstop_still_refuses_exact_deficiency(self, monkeypatch):
        """Driving the deficient fixtures through the large-n Gram arm.

        The Gram probe runs with a deliberately generous tolerance because it
        only has to catch EXACT deficiency (rules 8/9/11 catch graded
        near-deficiency structurally first); these are exactly-deficient, so
        the verdict must survive the route change.
        """
        import superglm.features.piecewise as pw

        monkeypatch.setattr(pw, "_RANK_PROBE_MAX_FACTOR_ROWS", 0)
        with pytest.raises(ValueError, match="rank deficient against the intercept"):
            Piecewise([1.0], lower=0.0, upper=2.0, base="first").build(
                np.concatenate([np.full(50, 0.5), np.full(50, 1.5)])
            )
        with pytest.raises(ValueError, match="rank deficient"):
            Piecewise([1.0, 2.0], base="first", lower=0.0, upper=3.0).build(
                np.array([0.0, 1.5, 3.0])
            )

    def test_rule_11_a_thin_segment_warns_and_reports_every_segment(self):
        x = np.concatenate([np.array([0.5]), np.linspace(1.01, 2.0, 1000)])
        with pytest.warns(UserWarning, match="of the in-range weight") as record:
            Piecewise([1.0], lower=0.0, upper=2.0).build(x)
        assert "Per-segment weight" in str(record[0].message)

    def test_rule_11_threshold_is_a_named_constant(self):
        assert _SMALL_SEGMENT_WEIGHT_FRACTION == 0.005

    def test_non_finite_breaks_fail_at_construction(self):
        """NaN passes every ordering comparison, so it must fail where it is written.

        Left to build(), ``Piecewise([np.nan])`` passes the strict-order and
        in-range checks (both comparisons are false on NaN) and surfaces as a
        rank/SVD failure with nothing pointing at the constructor call.
        """
        with pytest.raises(ValueError, match="breaks must be finite"):
            Piecewise([np.nan])
        with pytest.raises(ValueError, match="breaks must be finite"):
            Piecewise([1.0, np.inf, 2.0])

    def test_non_finite_bounds_fail_at_construction(self):
        """``lower=-inf`` passes ``lower < upper`` but makes every hat width infinite."""
        with pytest.raises(ValueError, match="lower must be finite"):
            Piecewise([2.0], lower=-np.inf)
        with pytest.raises(ValueError, match="upper must be finite"):
            Piecewise([2.0], upper=np.nan)


class TestZeroWeightRows:
    """A zero weight is zero replicated rows: predictable, but never geometry.

    The stated rule is ``_spline_knots.knot_geometry_data``'s, and the spline
    path already follows it; these tests hold ``Piecewise`` to the same
    contract, mirroring ``test_spline_weight_geometry``.
    """

    def test_the_fit_is_identical_with_and_without_the_zero_weight_rows(self):
        case = make_case("zero_weight_rows")
        keep = case.sample_weight > 0.0
        assert not np.all(keep), "fixture must actually carry zero-weight rows"

        def fitted(X, y, w):
            model = SuperGLM(
                features={
                    "x": Piecewise([25.0, 50.0, 75.0], base=50.0),
                    "region": Categorical(base="first"),
                },
            )
            model.fit(X, y, sample_weight=w)
            return model

        full = fitted(case.X, case.y, case.sample_weight)
        dropped = fitted(
            case.X.loc[keep].reset_index(drop=True),
            case.y[keep],
            case.sample_weight[keep],
        )

        np.testing.assert_array_equal(full._specs["x"]._knots, dropped._specs["x"]._knots)
        assert full._specs["x"]._base_index == dropped._specs["x"]._base_index
        np.testing.assert_allclose(
            np.asarray(full.result.beta),
            np.asarray(dropped.result.beta),
            rtol=1e-10,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            full.predict(case.X), dropped.predict(case.X), rtol=1e-10, atol=0.0
        )

    def test_zero_weight_rows_remain_predictable(self):
        """Out of geometry is not out of the model: their x still evaluates."""
        case = make_case("zero_weight_rows")
        model = SuperGLM(features={"x": case.spec, "region": Categorical(base="first")})
        model.fit(case.X, case.y, sample_weight=case.sample_weight)
        spec = model._specs["x"]

        outside = np.array([-30.0, 250.0])
        # Under the default clip policy the zero-weight tails group onto the
        # boundary knots, exactly like any other out-of-range prediction row.
        np.testing.assert_array_equal(
            spec.transform(outside),
            spec.transform(np.array([spec._knots[0], spec._knots[-1]])),
        )
        preds = model.predict(case.X)
        assert preds.shape == (len(case.X),)
        assert np.all(np.isfinite(preds))

    def test_a_zero_weight_outlier_does_not_widen_the_default_range(self):
        """Regression mirroring the spline path: x=100, w=0 must not widen [-1, 1]."""
        x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0, 100.0])
        w = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0])

        weighted = Piecewise([0.0])
        weighted.build(x, sample_weight=w)
        omitted = Piecewise([0.0])
        omitted.build(x[w > 0.0], sample_weight=w[w > 0.0])

        np.testing.assert_array_equal(weighted._knots, np.array([-1.0, 0.0, 1.0]))
        np.testing.assert_array_equal(weighted._knots, omitted._knots)

    def test_int_mode_places_quantiles_on_positive_weight_rows_only(self):
        rng = np.random.default_rng(41)
        x_pos = np.round(rng.uniform(0.0, 10.0, 400), 1)
        x = np.concatenate([x_pos, np.array([500.0, 800.0])])
        w = np.concatenate([np.ones(400), np.zeros(2)])

        weighted = Piecewise(3)
        weighted.build(x, sample_weight=w)
        omitted = Piecewise(3)
        omitted.build(x_pos, sample_weight=np.ones(400))

        np.testing.assert_array_equal(weighted._knots, omitted._knots)

    def test_negative_weights_are_refused_with_a_clear_message(self):
        """A negative weight otherwise reaches sqrt in the rank probe as NaN."""
        with pytest.raises(ValueError, match="non-negative sample_weight"):
            Piecewise([2.0]).build(
                np.linspace(0.0, 10.0, 11), sample_weight=np.linspace(-1.0, 9.0, 11)
            )

    def test_non_finite_weights_are_refused(self):
        w = np.ones(11)
        w[3] = np.nan
        with pytest.raises(ValueError, match="finite sample_weight"):
            Piecewise([2.0]).build(np.linspace(0.0, 10.0, 11), sample_weight=w)

    def test_all_zero_weights_are_refused(self):
        with pytest.raises(ValueError, match="positive sample_weight"):
            Piecewise([2.0]).build(np.linspace(0.0, 10.0, 11), sample_weight=np.zeros(11))


class TestSpecState:
    def test_transform_before_build_names_the_missing_step(self):
        with pytest.raises(RuntimeError, match="call build\\(\\)"):
            Piecewise([2.0]).transform(np.array([1.0]))

    def test_repr_before_and_after_build(self):
        spec = Piecewise([1.0, 2.0])
        assert repr(spec) == "Piecewise(breaks=[1, 2], base='most_exposed')"
        spec.build(np.linspace(0.0, 3.0, 7))
        assert "4 knots" in repr(spec)
        # Unweighted most_exposed is hat-carried row count: the interior knots
        # each carry 2.0 of the 7 rows' mass against 1.5 at either boundary,
        # and argmax breaks the tie toward the first of them.
        assert "ref=1" in repr(spec)

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
    def test_a_categorical_interaction_keeps_its_plain_sparse_representation(self):
        """Row compression is opt-in via ``GroupInfo.supports_row_compression``.

        Before the scoping, ``_build_unpenalized_sparse_group`` re-routed EVERY
        unpenalized sparse group: a categorical interaction became a
        ``DiscretizedSSPGroupMatrix`` subclass, which disables the tabmat
        split for the whole model and flips its design_summary representation
        -- a behaviour change to a shipped term type that never asked for it.
        """
        from superglm.group_matrix import SparseGroupMatrix

        rng = np.random.default_rng(37)
        n = 4000
        frame = pd.DataFrame(
            {
                "region": rng.choice(["A", "B", "C"], n),
                "ptype": rng.choice(["X", "Y", "Z"], n),
            }
        )
        y = rng.poisson(1.0, n).astype(np.float64)
        model = SuperGLM(
            features={"region": Categorical(base="first"), "ptype": Categorical(base="first")},
            interactions=[("region", "ptype")],
        )
        model.fit(frame, y)

        idx = next(i for i, g in enumerate(model._groups) if g.feature_name == "region:ptype")
        assert type(model._dm.group_matrices[idx]) is SparseGroupMatrix
        # The pre-PR representation and its consequences, pinned: the split
        # survives because no group in this model is a compressed one.
        assert model._dm._get_or_build_tabmat_split() is not None
        summary = model.design_summary()
        row = summary.loc[summary["feature"] == "region:ptype"].iloc[0]
        assert row["representation"] == "sparse-csr"
        assert not row["compressed"]

    def test_build_warnings_name_the_feature_that_raised_them(self):
        """Two Piecewise terms, one thin segment: the warning must say which.

        The build-time errors already carry ``Feature {name!r}:``; without the
        same prefix on warnings, a model with two terms of one spec type emits
        two identical-looking messages with nothing saying which column to fix.
        """
        a = np.linspace(0.0, 10.0, 1001)
        b = np.concatenate([np.array([0.5]), np.linspace(1.01, 2.0, 1000)])
        rng = np.random.default_rng(43)
        y = rng.poisson(1.0, a.size).astype(np.float64)
        model = SuperGLM(
            features={
                "a": Piecewise([5.0], lower=0.0, upper=10.0),
                "b": Piecewise([1.0], lower=0.0, upper=2.0),
            },
        )
        with pytest.warns(UserWarning) as record:
            model.fit(pd.DataFrame({"a": a, "b": b}), y)

        messages = [str(w.message) for w in record]
        thin = [m for m in messages if "of the in-range weight" in m]
        assert thin, messages
        assert all(m.startswith("Feature 'b': ") for m in thin)
        assert not any("Feature 'a'" in m for m in messages)

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
