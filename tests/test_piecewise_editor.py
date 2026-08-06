"""Editing, plotting and screening a ``Piecewise`` term.

The design's third property lives here: an editor handle sits on a knot, its
value *is* the coefficient, and moving one moves nothing else.  That is only
true because ``term.x`` is the knot vector and the raw hat basis evaluated at
the knots is the identity, so the editor's least-squares recovery returns the
coefficient vector exactly rather than approximately -- the tests assert the
exact form on purpose.

Two things had to be measured before an assertion could be written, and both
came out against the plan's phrasing:

* **Locality is a statement about the basis column, not about the two adjacent
  knot intervals.**  Rows outside ``[lower, upper]`` are clamped into the outer
  segment, so the hats at ``t_1`` and ``t_J`` keep non-zero entries out in the
  linear tails.  Measured on ``pinned_narrower``: moving the handle at ``t_J``
  changes predictions for rows above ``upper`` by 2.2e-01, which the interval
  form of the claim would have called a violation.  The support is taken from
  the basis column throughout, and one test pins the discrepancy itself.
* **The base-handle re-base keeps predictions local to round-off, not to the
  bit.**  Every coefficient shifts by ``-d`` and the intercept by ``+d``, and
  the two cancel through ``sum_j h_j = 1`` -- a sum that is not exactly 1 in
  floating point.  Measured worst relative move outside the base hat's support
  over the whole fixture matrix: 5.4e-16, about 2.4 eps.  A non-base handle
  move *is* bit-identical outside its support, because there the other
  coefficients are untouched and the hat is exactly 0.0.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

import superglm.editor
from superglm import Categorical, Numeric, Polynomial, Spline, SuperGLM
from superglm.editor import EditorSession
from superglm.editor.apply import _apply_term_edit, apply_edits_to_model_copy_with_data
from superglm.editor.controls import _control_handle_count, _control_handle_limits
from superglm.editor.payloads import session_payload
from superglm.editor.summaries import _compact_summary_row
from superglm.editor.terms import term_type_from_spec, term_weights_from_data
from superglm.export.summary import build_summary_export_payload
from superglm.features.piecewise import Piecewise
from superglm.plotting.comparison import _build_term_comparison_data, _resolve_comparable_terms
from tests._piecewise_cases import CASE_NAMES, PiecewiseCase, make_case

_EPS = float(np.finfo(np.float64).eps)

# The editor's own limits, restated so a change to either is a test failure
# rather than a silent change of behaviour.
_DEFAULT_HANDLE_CAP = 12
_HARD_HANDLE_CAP = 24


def _prediction_rtol(n_cols: int) -> float:
    """Round-off floor for comparing two prediction vectors of these models.

    Both sides build ``eta`` as an intercept plus a dot product over ``n_cols``
    piecewise columns plus two other term contributions, then exponentiate:
    ``2 * n_cols + 6`` roundings each.  Doubling covers both sides and the
    factor of 8 leaves headroom for a differently ordered BLAS.  A relative
    error on the response equals an absolute error on ``eta``, and every
    log-scale quantity here is order 1, so nothing inflates this beyond the
    flop count.
    """
    return 8 * (n_cols + 2) * _EPS


def _fit(case_name: str) -> tuple[SuperGLM, PiecewiseCase]:
    """Fit the named fixture.  Callers only read from it or deep-copy it."""
    case = make_case(case_name)
    model = SuperGLM(
        features={
            "x": case.spec,
            "region": Categorical(base="first"),
            "density": Numeric(),
        },
    )
    model.fit(case.X, case.y, sample_weight=case.sample_weight)
    return model, case


_FITTED: dict[str, tuple[SuperGLM, PiecewiseCase]] = {}


def _fitted(case_name: str) -> tuple[SuperGLM, PiecewiseCase]:
    if case_name not in _FITTED:
        _FITTED[case_name] = _fit(case_name)
    return _FITTED[case_name]


def _spec(model: SuperGLM) -> Piecewise:
    """The FITTED spec.  ``SuperGLM`` deep-copies specs, so the caller's is unbuilt."""
    return model._specs["x"]


def _beta(model: SuperGLM) -> np.ndarray:
    group = next(g for g in model._groups if g.feature_name == "x")
    return np.asarray(model.result.beta)[group.sl].copy()


def _hat_column(model: SuperGLM, case: PiecewiseCase, knot_index: int) -> np.ndarray:
    x_values = np.asarray(case.X["x"], dtype=np.float64)
    return _spec(model)._raw_basis_matrix(x_values)[:, knot_index]


def _most_balanced_non_base_knot(model: SuperGLM, case: PiecewiseCase) -> int:
    """Pick the non-base knot whose support splits the rows most evenly.

    A locality assertion is vacuous if either side of the split is empty, and
    the outermost hats cover nearly everything on a narrow pin.  Choosing the
    most balanced knot keeps both halves populated on every fixture without
    hand-picking an index per case.
    """
    spec = _spec(model)
    x_values = np.asarray(case.X["x"], dtype=np.float64)
    basis = spec._raw_basis_matrix(x_values)
    best, best_score = -1, -1
    for j in range(spec._knots.size):
        if j == spec._base_index:
            continue
        n_in = int(np.count_nonzero(basis[:, j] != 0.0))
        score = min(n_in, x_values.size - n_in)
        if score > best_score:
            best, best_score = j, score
    return best


def _force_apply(model: SuperGLM, session: EditorSession):
    """Materialize edits, forcing the apply path to run even for a null edit.

    ``apply_edits_to_model_copy_with_data`` skips any term whose edited values
    still match its originals, so a genuinely null session never reaches
    ``_apply_piecewise_term`` at all.  Perturbing ``original_log_effect`` alone
    makes the term register as changed while leaving the *targets* -- which are
    read from ``edited_log_effect`` -- at the fitted values, which is the only
    way to measure what the apply branch does with a null edit.
    """
    for term in session.terms.values():
        term.original_log_effect = term.original_log_effect - 1.0
    return apply_edits_to_model_copy_with_data(model, session.terms)


# ══════════════════════════════════════════════════════════════════
# Control handles
# ══════════════════════════════════════════════════════════════════


class TestControlHandles:
    def test_the_editor_names_a_piecewise_spec_piecewise(self):
        assert term_type_from_spec(Piecewise([1.0, 2.0])) == "piecewise"

    @pytest.mark.parametrize("case_name", CASE_NAMES)
    def test_one_handle_sits_on_every_knot_carrying_its_log_relativity(self, case_name):
        model, _ = _fitted(case_name)
        spec = _spec(model)
        session = EditorSession.from_model(model, terms=["x"])

        controls = session.control_points("x")

        # atol=0: the handles are not near the knots, they ARE the knots. The
        # raw basis at the knot vector is the identity, so the least-squares
        # recovery and the support-centre calculation are both exact.
        assert controls["x"].size == spec._knots.size
        np.testing.assert_allclose(controls["x"], spec._knots, rtol=0.0, atol=0.0)
        assert_array_equal(controls["basis_index"], np.arange(spec._knots.size))
        assert_array_equal(controls["log_effect"], model.term_inference("x").log_relativity)
        assert controls["log_effect"][spec._base_index] == 0.0

    def test_the_many_knots_case_beats_the_twelve_handle_default(self):
        model, _ = _fitted("many_knots")
        spec = _spec(model)
        session = EditorSession.from_model(model, terms=["x"])

        controls = session.control_points("x")

        # Without the opt-in this term would show 12 of its 15 knots: the
        # default is what the opt-in exists to override, so assert both halves.
        assert spec._knots.size > _DEFAULT_HANDLE_CAP
        assert _control_handle_count(spec._knots.size, None) == _DEFAULT_HANDLE_CAP
        assert controls["x"].size == spec._knots.size

    def test_a_term_past_the_hard_cap_subsamples_its_handles(self):
        # The hard cap is real and is documented in the Piecewise docstring
        # rather than pretended away, so it gets an assertion of its own.
        model = _wide_piecewise_model()
        spec = _spec(model)
        session = EditorSession.from_model(model, terms=["x"])

        controls = session.control_points("x")

        assert spec._knots.size > _HARD_HANDLE_CAP
        assert _control_handle_limits(spec._knots.size)[1] == _HARD_HANDLE_CAP
        assert controls["x"].size == _HARD_HANDLE_CAP
        # Every displayed handle is still exactly on a knot -- subsampling
        # drops handles, it never moves them off the knots.
        assert np.all(np.isin(controls["x"], spec._knots))
        assert controls["max_handles"] == _HARD_HANDLE_CAP

    def test_moving_a_handle_lands_exactly_on_the_requested_value(self):
        model, _ = _fitted("interior_base")
        spec = _spec(model)
        session = EditorSession.from_model(model, terms=["x"])
        controls = session.control_points("x")
        before = session.terms["x"].edited_log_effect.copy()
        knot_index = _most_balanced_non_base_knot(model, _fitted("interior_base")[1])

        session.move_control_point("x", knot_index, float(controls["log_effect"][knot_index] + 0.3))

        expected = before.copy()
        expected[knot_index] = before[knot_index] + 0.3
        assert_array_equal(session.terms["x"].edited_log_effect, expected)
        assert session.terms["x"].edited_log_effect[spec._base_index] == 0.0

    def test_the_gate_was_widened_by_exactly_one_term_type(self):
        # A categorical term is refused by the levels guard, which sits ahead of
        # the type gate, so it cannot tell whether the gate itself is still
        # narrow. A Polynomial term has an x grid and no levels, so it reaches
        # the gate and only the gate can refuse it -- that is the assertion
        # that would notice the tuple being opened too far.
        case = make_case("interior_base")
        model = SuperGLM(
            features={
                "x": case.spec,
                "region": Categorical(base="first"),
                "density": Polynomial(degree=2),
            },
        )
        model.fit(case.X, case.y, sample_weight=case.sample_weight)
        session = EditorSession.from_model(model, terms=["x", "region", "density"])

        assert session.terms["density"].x is not None
        assert session.terms["density"].levels is None
        with pytest.raises(TypeError, match="control handles"):
            session.control_points("density")
        with pytest.raises(TypeError, match="control handles"):
            session.control_points("region")

    def test_the_frontend_payload_carries_the_piecewise_handles(self):
        model, _ = _fitted("interior_base")
        spec = _spec(model)
        session = EditorSession.from_model(model, terms=["x", "region"])

        payload = session_payload(session)

        assert payload["x"]["term_type"] == "piecewise"
        controls = payload["x"]["controls"]
        assert controls is not None
        assert controls["count"] == spec._knots.size
        assert controls["x"] == [float(v) for v in spec._knots]
        # The gate is widened, not removed: a categorical term still gets none.
        assert payload["region"]["controls"] is None


def _wide_piecewise_model() -> SuperGLM:
    """A 31-knot term, past the editor's hard cap of 24 handles."""
    rng = np.random.default_rng(99)
    n = 1200
    x = rng.uniform(0.0, 120.0, n)
    x[0], x[-1] = 0.0, 120.0
    weights = rng.uniform(0.5, 1.5, n)
    frame = pd.DataFrame({"x": x, "region": rng.choice(["A", "B"], n)})
    y = rng.poisson(np.exp(-1.5 + 0.01 * x) * weights).astype(np.float64)
    breaks = [float(v) for v in range(4, 120, 4)]
    model = SuperGLM(
        features={
            "x": Piecewise(breaks, base=60.0, lower=0.0, upper=120.0),
            "region": Categorical(base="first"),
        },
    )
    model.fit(frame, y, sample_weight=weights)
    return model


# ══════════════════════════════════════════════════════════════════
# Design section 9, property 3a -- the null edit (#236)
# ══════════════════════════════════════════════════════════════════


class TestNullEdit:
    @pytest.mark.parametrize("case_name", CASE_NAMES)
    def test_closing_a_session_with_no_edits_returns_an_untouched_copy(self, case_name):
        # The session-level half of the property. It does NOT reach the
        # piecewise apply branch -- an unedited term is filtered out before
        # `_apply_term_edit` is called -- which is exactly why the next test
        # exists. What this one pins is that materializing yields a separate
        # model whose predictions and intercept are bit-identical, so a copy
        # that aliased the source could not pass it for the wrong reason.
        model, case = _fitted(case_name)
        before = model.predict(case.X)
        session = EditorSession.from_model(model, terms=["x"])

        edited = session.to_model()

        assert edited is not model
        assert_array_equal(edited.predict(case.X), before)
        assert edited.result.intercept == model.result.intercept
        assert_array_equal(model.predict(case.X), before)

    @pytest.mark.parametrize("case_name", CASE_NAMES)
    def test_the_apply_path_moves_nothing_when_the_targets_are_the_fitted_values(self, case_name):
        # This is the #236 property itself, and the previous test does NOT
        # cover it: an unedited term never reaches the apply branch. Here the
        # branch runs against targets equal to the fitted knot values, which is
        # exactly the state a user leaves after opening and closing a panel.
        model, case = _fitted(case_name)
        session = EditorSession.from_model(model, terms=["x"])
        before = model.predict(case.X)
        beta_before = _beta(model)

        edited = _force_apply(model, session)

        assert_array_equal(edited.predict(case.X), before)
        assert_array_equal(_beta(edited), beta_before)
        assert edited.result.intercept == model.result.intercept

    def test_the_apply_path_refuses_an_editor_grid_that_is_not_the_knot_vector(self):
        model, _ = _fitted("interior_base")
        session = EditorSession.from_model(model, terms=["x"])
        term = session.terms["x"]
        term.edited_log_effect = term.edited_log_effect[:-1]

        with pytest.raises(ValueError, match="knot vector"):
            _apply_term_edit(model, term)


# ══════════════════════════════════════════════════════════════════
# Design section 9, property 3b -- locality
# ══════════════════════════════════════════════════════════════════


class TestLocality:
    @pytest.mark.parametrize("case_name", CASE_NAMES)
    def test_moving_one_non_base_handle_moves_one_coefficient_and_two_segments(self, case_name):
        model, case = _fitted(case_name)
        spec = _spec(model)
        knot_index = _most_balanced_non_base_knot(model, case)
        delta = 0.3
        before = model.predict(case.X)
        beta_before = _beta(model)
        intercept_before = model.result.intercept
        session = EditorSession.from_model(model, terms=["x"])
        session.terms["x"].edited_log_effect[knot_index] += delta

        edited = apply_edits_to_model_copy_with_data(model, session.terms)

        column = _hat_column(model, case, knot_index)
        support = column != 0.0
        assert support.any() and (~support).any(), "fixture no longer splits the rows"

        # Exactly one coefficient moves, by exactly the delta.
        coef_index = int(np.flatnonzero(spec._non_base_indices == knot_index)[0])
        expected = beta_before.copy()
        expected[coef_index] = beta_before[coef_index] + delta
        assert_array_equal(_beta(edited), expected)
        assert edited.result.intercept == intercept_before

        # Bit-identical off the moved hat: every other coefficient is untouched
        # and the hat is exactly 0.0 there, so no rounding can leak across.
        after = edited.predict(case.X)
        assert_array_equal(after[~support], before[~support])
        # Non-vacuity: the rows on the moved hat move by a visible amount, not
        # by a last-bit wobble that an equality test would also accept.
        assert np.max(np.abs(after[support] / before[support] - 1.0)) > 0.01
        np.testing.assert_allclose(
            after[support],
            before[support] * np.exp(delta * column[support]),
            rtol=_prediction_rtol(spec._non_base_indices.size),
            atol=0.0,
        )

    def test_a_hat_next_to_the_boundary_keeps_support_out_in_the_linear_tail(self):
        # MEASURED, and it contradicts the interval form of the locality claim:
        # with `upper` pinned inside the data, rows above it are clamped into
        # the last segment, so the hat at t_J still weights them. Locality is a
        # statement about the basis column; (t_{j-1}, t_{j+1}) is only the same
        # set while no row extrapolates.
        model, case = _fitted("pinned_narrower")
        spec = _spec(model)
        knots = spec._knots
        knot_index = knots.size - 2
        x_values = np.asarray(case.X["x"], dtype=np.float64)

        support = _hat_column(model, case, knot_index) != 0.0
        interval = (x_values > knots[knot_index - 1]) & (x_values < knots[knot_index + 1])

        assert np.any(support & ~interval)
        assert np.all(x_values[support & ~interval] >= knots[-1])
        # The reverse containment does hold: nothing inside the two adjacent
        # segments is outside the support.
        assert not np.any(interval & ~support)

    @pytest.mark.parametrize("case_name", CASE_NAMES)
    def test_a_handle_move_is_reported_back_as_the_value_that_was_set(self, case_name):
        model, case = _fitted(case_name)
        knot_index = _most_balanced_non_base_knot(model, case)
        delta = 0.3
        session = EditorSession.from_model(model, terms=["x"])
        session.terms["x"].edited_log_effect[knot_index] += delta
        target = session.terms["x"].edited_log_effect.copy()

        edited = apply_edits_to_model_copy_with_data(model, session.terms)

        # The round trip is exact, not approximate: the assignment is direct,
        # so what the editor shows is what the refitted term reports.
        assert_array_equal(edited.term_inference("x").log_relativity, target)


# ══════════════════════════════════════════════════════════════════
# Design section 9, property 3c -- the base-handle re-base
# ══════════════════════════════════════════════════════════════════


class TestBaseHandleRebase:
    @pytest.mark.parametrize("case_name", CASE_NAMES)
    def test_dragging_the_base_handle_shifts_every_coefficient_and_the_intercept(self, case_name):
        model, case = _fitted(case_name)
        spec = _spec(model)
        delta = 0.4
        beta_before = _beta(model)
        intercept_before = model.result.intercept
        session = EditorSession.from_model(model, terms=["x"])
        session.terms["x"].edited_log_effect[spec._base_index] += delta

        edited = apply_edits_to_model_copy_with_data(model, session.terms)

        # Re-basing, exactly: the term is now measured against a base knot
        # whose value is `delta`, so every coefficient falls by `delta` and the
        # intercept absorbs it. Coefficient locality does not survive this and
        # is not asserted; prediction locality does, below.
        assert_array_equal(_beta(edited), beta_before - delta)
        assert edited.result.intercept == intercept_before + delta

    @pytest.mark.parametrize("case_name", CASE_NAMES)
    def test_the_re_base_still_moves_only_the_base_hat_s_own_rows(self, case_name):
        model, case = _fitted(case_name)
        spec = _spec(model)
        delta = 0.4
        before = model.predict(case.X)
        session = EditorSession.from_model(model, terms=["x"])
        session.terms["x"].edited_log_effect[spec._base_index] += delta

        edited = apply_edits_to_model_copy_with_data(model, session.terms)

        # f'(x) = f(x) + delta * h_base(x): the +delta on the intercept and the
        # -delta on every coefficient cancel through the partition of unity.
        # That cancellation is exact in algebra and not in binary, so rows off
        # the base hat move by round-off rather than by nothing -- measured at
        # worst 5.4e-16 relative over the whole fixture matrix, against a
        # tolerance derived from the flop count.
        column = _hat_column(model, case, spec._base_index)
        support = column != 0.0
        assert support.any() and (~support).any(), "fixture no longer splits the rows"
        rtol = _prediction_rtol(spec._non_base_indices.size)

        after = edited.predict(case.X)
        np.testing.assert_allclose(after, before * np.exp(delta * column), rtol=rtol, atol=0.0)
        np.testing.assert_allclose(after[~support], before[~support], rtol=rtol, atol=0.0)
        assert np.max(np.abs(after[support] / before[support] - 1.0)) > rtol


# ══════════════════════════════════════════════════════════════════
# Surfaces an edit reaches after the edit
# ══════════════════════════════════════════════════════════════════


class TestEditedModelReporting:
    @pytest.mark.parametrize("case_name", CASE_NAMES)
    def test_the_edited_summary_still_reports_every_knot(self, case_name):
        """The edited knot values are the numbers an actuary files.

        The edited model's rows come from a second builder
        (``report_ops._build_editor_stale_coef_rows``), and its generic tail is
        the spline fallback: the per-knot rows disappear, the surviving row is
        labelled a spline, and the whole point of the edit becomes invisible on
        the console and in the exported Summary sheet.
        """
        model, case = _fitted(case_name)
        spec = _spec(model)
        knot_index = _most_balanced_non_base_knot(model, case)

        session = EditorSession.from_model(model, terms=["x"])
        controls = session.control_points("x")
        session.move_control_point("x", knot_index, float(controls["log_effect"][knot_index]) + 0.4)
        edited = apply_edits_to_model_copy_with_data(model, session.terms)

        text = str(edited.summary())
        for knot in spec._knots:
            assert f"x[{float(knot):.10g}]" in text
        assert "[piecewise, " in text
        assert "[spline, " not in text

        payload = build_summary_export_payload(edited)
        rows = [row for row in payload.terms if row.group == "x"]
        coefficients = [row for row in rows if row.kind == "coefficient"]
        assert [row.term for row in coefficients] == [
            f"x[{float(spec._knots[j]):.10g}]" for j in spec._non_base_indices
        ]
        assert [row.kind for row in rows if row.kind == "group"] == ["group"]
        # The edited coefficient really is the value the handle was moved to.
        moved = next(
            row for row in coefficients if row.term == f"x[{float(spec._knots[knot_index]):.10g}]"
        )
        assert moved.estimate == pytest.approx(float(controls["log_effect"][knot_index]) + 0.4)

    def test_the_edited_summary_reports_no_smoothing_parameter(self):
        """§4 makes this term unpenalized; the fallback printed the global ridge.

        ``spline_group_enrichment`` reads ``fitted_lambda2(model)`` for any group
        it is handed, so the fallback published ``lambda = 0.1`` in an exported
        workbook for a term whose ``GroupInfo.penalty_matrix`` is ``None``.
        """
        model, case = _fitted("interior_base")
        knot_index = _most_balanced_non_base_knot(model, case)

        session = EditorSession.from_model(model, terms=["x"])
        controls = session.control_points("x")
        session.move_control_point("x", knot_index, float(controls["log_effect"][knot_index]) + 0.4)
        edited = apply_edits_to_model_copy_with_data(model, session.terms)

        assert "lam=" not in str(edited.summary())
        payload = build_summary_export_payload(edited)
        group_row = next(row for row in payload.terms if row.group == "x" and row.kind == "group")
        assert group_row.smoothing_lambda is None

    def test_the_browser_payload_calls_the_group_row_piecewise(self):
        """The console renderer was fixed to say "piecewise"; this one said "spline".

        Two summary surfaces disagreeing about what the term is, for the one
        term type where smooth-versus-not is the entire point of the feature.
        """
        model, _ = _fitted("interior_base")
        rows = [_compact_summary_row(row) for row in model.summary()._display_rows]

        group_row = next(row for row in rows if row["name"] == "x" and row["stat_label"] == "chi2")
        assert group_row["kind"] == "piecewise"
        # The JS renders the label from the kind rather than hard-coding
        # "spline", so an unlisted kind would print nothing at all instead of
        # the wrong word.  Both halves have to move together.
        source = (Path(superglm.editor.__file__).parent / "app" / "summary.js").read_text()
        listed = source.split("const GROUP_ROW_KINDS = ", 1)[1].split(";", 1)[0]
        assert '"piecewise"' in listed and '"spline"' in listed
        assert "GROUP_ROW_KINDS.has(row.kind)" in source


class TestOffsetAndExposure:
    @pytest.mark.parametrize("case_name", CASE_NAMES)
    def test_the_editor_offset_reproduces_the_term_outside_the_knot_span(self, case_name):
        """``refit_with_edited_offset`` must condition on the term that was edited.

        ``term_offset_values`` interpolates over the editor grid with
        ``left=``/``right=`` clamping, which holds the term FLAT past the
        boundary knots -- contradicting ``Piecewise.score``, the plotted curve
        and the boundary slopes the exported workbook publishes. ``Piecewise``
        is the first spec whose grid is deliberately allowed to be narrower than
        the data, so the shared helper's clamp is newly load-bearing here.
        """
        model, case = _fitted(case_name)
        spec = _spec(model)
        session = EditorSession.from_model(model, terms=["x"])

        offset = np.asarray(session.edited_offset(["x"], X=case.X), dtype=np.float64).ravel()
        expected = spec.score(case.X["x"].to_numpy(dtype=np.float64), _beta(model))

        np.testing.assert_allclose(
            offset, expected, rtol=0.0, atol=_prediction_rtol(spec._non_base_indices.size)
        )

    def test_the_narrower_pin_really_does_put_rows_outside_the_knot_span(self):
        """Guard: on a fixture whose grid spans the data the clamp is invisible."""
        model, case = _fitted("pinned_narrower")
        spec = _spec(model)
        x_values = case.X["x"].to_numpy(dtype=np.float64)

        outside = (x_values < spec._knots[0]) | (x_values > spec._knots[-1])
        assert int(np.count_nonzero(outside)) > 100

    def test_the_exposure_layer_keeps_the_weight_behind_the_boundary_segments(self):
        """``np.histogram`` drops anything outside the outermost grid edge.

        On the narrower pin that silently deleted a fifth of total exposure --
        and precisely the fifth sitting behind the two boundary segments, which
        is what a user is looking at when deciding whether to drag ``t_0`` or
        ``t_{J+1}``.
        """
        model, case = _fitted("pinned_narrower")
        session = EditorSession.from_model(model, terms=["x"])

        weights = term_weights_from_data(case.X, case.sample_weight, "x", session.terms["x"])

        assert float(np.sum(weights)) == pytest.approx(float(np.sum(case.sample_weight)), rel=1e-12)


class TestModelComparison:
    def test_a_piecewise_term_is_comparable_across_two_models(self):
        """``_comparison_family`` admitted only Numeric / Polynomial / spline.

        The term was reported as "missing or unsupported in one or more models"
        while present and identically specified in both, sending the reader
        after a column that is not absent.
        """
        first, case = _fitted("interior_base")
        second_case = make_case("interior_base")
        second = SuperGLM(
            features={
                "x": second_case.spec,
                "region": Categorical(base="first"),
                "density": Numeric(),
            },
        )
        second.fit(
            second_case.X,
            second_case.y * 0 + np.roll(second_case.y, 3),
            sample_weight=second_case.sample_weight,
        )

        terms, skipped = _resolve_comparable_terms({"a": first, "b": second})

        assert "x" in terms
        assert "x" not in skipped

        payload = _build_term_comparison_data(
            models={"a": first, "b": second},
            terms=["x"],
            X=case.X,
            sample_weight=case.sample_weight,
        )
        entry = next(term for term in payload["terms"] if term["name"] == "x")
        assert entry["family"] == "continuous"
        # The continuous path scores through `spec.score`, which a piecewise term
        # answers exactly, so the overlay is the fitted function and not a resample.
        grid = np.asarray(entry["domain"]["x"], dtype=np.float64)
        np.testing.assert_allclose(
            np.asarray(entry["series"]["a"]["link"], dtype=np.float64),
            _spec(first).score(grid, _beta(first)),
            rtol=0.0,
            atol=1e-12,
        )


# ══════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════


class TestPlotting:
    def test_the_matplotlib_grid_draws_the_piecewise_panel(self):
        # Before the kind tuples admitted "piecewise" this completed without
        # raising and drew NOTHING: the panel fell to `ax.set_visible(False)`.
        # A plot that silently omits a fitted term is the defect being pinned,
        # so the assertion is on drawn content, not on completion.
        model, case = _fitted("interior_base")

        fig = model.plot(X=case.X, sample_weight=case.sample_weight)

        panels = [ax for ax in fig.axes if ax.get_title() == "x"]
        assert len(panels) == 1
        assert panels[0].get_visible()
        assert len(panels[0].lines) > 0

    def test_the_grid_lays_out_a_density_strip_under_a_lone_piecewise_term(self):
        # The grid decides whether to build the lower strip row from its own
        # tuple of continuous kinds, separately from the panel dispatch. With a
        # Numeric or a Spline also in the model another term keeps that row
        # alive and the omission is invisible, so this model has exactly one
        # continuous term and it is the piecewise one.
        case = make_case("interior_base")
        model = SuperGLM(features={"x": case.spec, "region": Categorical(base="first")})
        model.fit(case.X, case.y, sample_weight=case.sample_weight)

        fig = model.plot(X=case.X, sample_weight=case.sample_weight)

        strips = [
            ax
            for ax in fig.axes
            if not ax.get_title() and "density" in ax.get_ylabel() and ax.lines and ax.collections
        ]
        assert len(strips) == 1

    def test_the_single_term_matplotlib_plot_draws_the_curve(self):
        # The single-term path had its own fallback: a figure containing the
        # text "Unknown term kind: 'piecewise'" and no curve.
        model, case = _fitted("interior_base")
        spec = _spec(model)

        fig = model.plot("x", X=case.X, sample_weight=case.sample_weight)

        panels = [ax for ax in fig.axes if ax.get_title() == "x"]
        assert len(panels) == 1
        assert not any(ax.texts for ax in fig.axes)
        curves = [line for line in panels[0].lines if line.get_label() == "Relativity"]
        assert len(curves) == 1
        assert_array_equal(curves[0].get_xdata(), spec._knots)

    def test_the_plot_data_payload_carries_the_knot_grid_and_a_density(self):
        # `_main_effect_density_dataframe` dispatches on the same kind tuple
        # and would fall through to `list(ti.levels)`, which is None here.
        model, case = _fitted("interior_base")
        spec = _spec(model)

        payload = model.plot_data("x", X=case.X, sample_weight=case.sample_weight)

        entry = payload["terms"][0]
        assert entry["term_kind"] == "piecewise"
        assert_array_equal(entry["effect"]["x"].to_numpy(), spec._knots)
        assert entry["density"] is not None
        assert_array_equal(entry["density"]["x"].to_numpy(), spec._knots)

    def test_the_plotly_figure_draws_the_piecewise_curve(self):
        go = pytest.importorskip("plotly.graph_objects", reason="plotly is an optional extra")
        model, case = _fitted("interior_base")
        spec = _spec(model)

        fig = model.plot(engine="plotly", X=case.X, sample_weight=case.sample_weight)

        traces = [
            trace
            for trace in fig.data
            if isinstance(trace, go.Scatter)
            and trace.x is not None
            and len(trace.x) == spec._knots.size
            and np.array_equal(np.asarray(trace.x, dtype=np.float64), spec._knots)
        ]
        assert traces, "no plotly trace was drawn on the piecewise knot grid"


# ══════════════════════════════════════════════════════════════════
# PSST screening
# ══════════════════════════════════════════════════════════════════


class TestScreeningDeferral:
    def test_screening_reports_a_bespoke_piecewise_deferral(self):
        case = make_case("interior_base")
        model = SuperGLM(
            family="poisson",
            features={
                "x": case.spec,
                "region": Categorical(base="first"),
                "density": Spline(kind="ps", n_knots=6),
            },
        )
        model.fit_reml(case.X, case.y, sample_weight=case.sample_weight)

        table = model.screen_interactions(case.X, case.y, sample_weight=case.sample_weight)

        deferred = table.attrs["deferred_features"]
        assert set(deferred) == {"x"}
        reason = deferred["x"]
        assert reason.startswith("Piecewise margins are deferred")
        assert "hat basis" in reason
        # Not the generic fallback, which names the class and stops there.
        assert reason != "Piecewise margins are deferred: no screenable margin"

    def test_naming_the_piecewise_term_in_candidates_raises_with_the_reason(self):
        case = make_case("interior_base")
        model = SuperGLM(
            family="poisson",
            features={
                "x": case.spec,
                "region": Categorical(base="first"),
                "density": Spline(kind="ps", n_knots=6),
            },
        )
        model.fit_reml(case.X, case.y, sample_weight=case.sample_weight)

        with pytest.raises(ValueError, match="Piecewise margins are deferred") as excinfo:
            model.screen_interactions(
                case.X,
                case.y,
                sample_weight=case.sample_weight,
                candidates=[("x", "region")],
            )
        assert "hat basis" in str(excinfo.value)
