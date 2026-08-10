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


def _fit(case_name: str, extrapolation: str | None = None) -> tuple[SuperGLM, PiecewiseCase]:
    """Fit the named fixture.  Callers only read from it or deep-copy it."""
    case = make_case(case_name, extrapolation=extrapolation)
    model = SuperGLM(
        features={
            "x": case.spec,
            "region": Categorical(base="first"),
            "density": Numeric(),
        },
    )
    model.fit(case.X, case.y, sample_weight=case.sample_weight)
    return model, case


_FITTED: dict[tuple[str, str | None], tuple[SuperGLM, PiecewiseCase]] = {}


def _fitted(case_name: str, extrapolation: str | None = None) -> tuple[SuperGLM, PiecewiseCase]:
    key = (case_name, extrapolation)
    if key not in _FITTED:
        _FITTED[key] = _fit(case_name, extrapolation=extrapolation)
    return _FITTED[key]


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
        # MEASURED under extrapolation="extend", and it contradicts the
        # interval form of the locality claim: with `upper` pinned inside the
        # data, rows above it are clamped into the last segment, so the hat at
        # t_J still weights them. Locality is a statement about the basis
        # column; (t_{j-1}, t_{j+1}) is only the same set while no row
        # extrapolates.
        model, case = _fitted("pinned_narrower", extrapolation="extend")
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

    def test_the_same_hat_is_local_again_under_the_clip_default(self):
        # The clip twin: rows above `upper` are grouped onto the boundary knot
        # itself, where the second-to-last hat is exactly zero, so the interval
        # form of the locality claim is restored.
        model, case = _fitted("pinned_narrower")
        assert _spec(model).extrapolation == "clip"
        spec = _spec(model)
        knots = spec._knots
        knot_index = knots.size - 2
        x_values = np.asarray(case.X["x"], dtype=np.float64)

        support = _hat_column(model, case, knot_index) != 0.0
        interval = (x_values > knots[knot_index - 1]) & (x_values < knots[knot_index + 1])

        assert not np.any(support & ~interval)
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
    @pytest.mark.parametrize("extrapolation", ["clip", "extend"])
    @pytest.mark.parametrize("case_name", CASE_NAMES)
    def test_the_editor_offset_reproduces_the_term_outside_the_knot_span(
        self, case_name, extrapolation
    ):
        """``refit_with_edited_offset`` must condition on the term that was edited.

        ``term_offset_values`` interpolates over the editor grid with
        ``left=``/``right=`` clamping, which holds the term FLAT past the
        boundary knots.  Under the clip default that clamp IS the model; under
        ``extrapolation="extend"`` it contradicts ``Piecewise.score``, the
        plotted curve and the boundary slopes the exported workbook publishes,
        so the helper has to switch to the boundary lines.  Both modes are
        swept: ``Piecewise`` is the first spec whose grid is deliberately
        allowed to be narrower than the data, so which tail rule the shared
        helper applies is newly load-bearing here.
        """
        model, case = _fitted(case_name, extrapolation=extrapolation)
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

    def test_offset_scoring_refuses_out_of_range_rows_under_error_mode(self):
        """Silently clamping rows the model itself refuses is a wrong offset.

        Under ``extrapolation='error'`` the model's ``score()`` raises on
        out-of-range rows; ``term_offset_values`` manufacturing a clamped
        offset for them would let ``refit_with_edited_offset`` rate exactly
        the rows the term's stated policy excludes.  The refusal mirrors the
        model's own message, tolerance included.
        """
        from superglm.editor.terms import term_offset_values

        model, _ = _fitted("interior_base", extrapolation="error")
        session = EditorSession.from_model(model, terms=["x"])
        term = session.terms["x"]
        spec = _spec(model)

        boundary = np.array([float(spec._knots[0]), float(spec._knots[-1])])
        assert term_offset_values(term, boundary).shape == (2,)
        with pytest.raises(ValueError, match="outside the rated range") as excinfo:
            term_offset_values(term, np.array([float(spec._knots[-1]) + 1.0]))
        assert "extrapolation='error'" in str(excinfo.value)

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
        # text "Unknown term kind: 'piecewise'" and no curve.  The curve is
        # drawn on a dense display grid that contains every knot -- the CI
        # band between knots is a quadratic form of the adjacent hats, not an
        # interpolation of the knot limits, so the knot grid alone cannot
        # carry the band.
        model, case = _fitted("interior_base")
        spec = _spec(model)

        fig = model.plot("x", X=case.X, sample_weight=case.sample_weight)

        panels = [ax for ax in fig.axes if ax.get_title() == "x"]
        assert len(panels) == 1
        assert not any(ax.texts for ax in fig.axes)
        curves = [line for line in panels[0].lines if line.get_label() == "Relativity"]
        assert len(curves) == 1
        drawn_x = np.asarray(curves[0].get_xdata(), dtype=np.float64)
        assert drawn_x.size > 50
        assert np.all(np.isin(spec._knots, drawn_x))
        # At the knots the drawn curve is the fitted relativity, exactly.
        ti = model.term_inference("x")
        drawn_y = np.asarray(curves[0].get_ydata(), dtype=np.float64)
        knot_positions = np.searchsorted(drawn_x, spec._knots)
        np.testing.assert_allclose(drawn_y[knot_positions], ti.relativity, rtol=1e-12, atol=0.0)

    def test_the_plot_data_payload_carries_the_knot_grid_and_a_density(self):
        # `_main_effect_density_dataframe` dispatches on the same kind tuple
        # and would fall through to `list(ti.levels)`, which is None here.
        # The effect grid stays the knot vector -- it is the exact
        # representation -- while the density gets an independent dense grid:
        # a KDE sampled at as few as three knots can miss every mode.
        model, case = _fitted("interior_base")
        spec = _spec(model)

        payload = model.plot_data("x", X=case.X, sample_weight=case.sample_weight)

        entry = payload["terms"][0]
        assert entry["term_kind"] == "piecewise"
        assert_array_equal(entry["effect"]["x"].to_numpy(), spec._knots)
        assert entry["density"] is not None
        density_x = entry["density"]["x"].to_numpy(dtype=np.float64)
        assert density_x.size > 50
        assert density_x.min() == spec._knots[0]
        assert density_x.max() == spec._knots[-1]

    def test_the_exposure_density_is_dense_for_a_three_knot_term(self):
        """codex: a 3-knot term fed the KDE exactly three evaluation points."""
        case = make_case("interior_base")
        model = SuperGLM(features={"x": Piecewise([50.0]), "region": Categorical(base="first")})
        model.fit(case.X, case.y, sample_weight=case.sample_weight)
        assert model._specs["x"]._knots.size == 3

        payload = model.plot_data("x", X=case.X, sample_weight=case.sample_weight)

        assert len(payload["terms"][0]["density"]) > 50

    def test_the_plotly_figure_draws_the_piecewise_curve(self):
        go = pytest.importorskip("plotly.graph_objects", reason="plotly is an optional extra")
        model, case = _fitted("interior_base")
        spec = _spec(model)

        fig = model.plot(engine="plotly", X=case.X, sample_weight=case.sample_weight)

        # Same display policy as matplotlib: traces live on a dense grid that
        # contains every knot, and the effect trace passes through the fitted
        # knot relativities exactly.
        ti = model.term_inference("x")
        knot_rel = np.asarray(ti.relativity, dtype=np.float64)
        matched = []
        for trace in fig.data:
            if not isinstance(trace, go.Scatter) or trace.x is None or len(trace.x) <= 50:
                continue
            x_arr = np.asarray(trace.x, dtype=np.float64)
            if not np.all(np.isin(spec._knots, x_arr)):
                continue
            y_arr = np.asarray(trace.y, dtype=np.float64)
            positions = np.searchsorted(x_arr, spec._knots)
            if positions.max() < y_arr.size and np.allclose(
                y_arr[positions], knot_rel, rtol=1e-12, atol=0.0
            ):
                matched.append(trace)
        assert matched, "no dense plotly trace passes through the fitted knot relativities"

    def test_the_display_band_is_the_exact_quadratic_form_between_knots(self):
        """codex: interpolating the knot CI endpoints misstates the band off-knot.

        Constructed so the two MUST differ: strong negative covariance between
        adjacent knots makes the true mid-segment SE far smaller than the
        interpolated one.  The display helper has to match the direct
        quadratic form ``var f(x) = h1^2 V11 + 2 h1 h2 V12 + h2^2 V22`` and
        not the straight line between the knot SEs.
        """
        from scipy.stats import norm

        from superglm.inference import TermInference
        from superglm.plotting.common import piecewise_display_term

        knots = np.array([0.0, 1.0, 2.0])
        log_rel = np.array([0.0, 0.5, 1.0])
        V = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.04, -0.038],
                [0.0, -0.038, 0.04],
            ]
        )
        se = np.sqrt(np.diag(V))
        z = float(norm.ppf(0.975))
        ti = TermInference(
            name="x",
            kind="piecewise",
            active=True,
            x=knots,
            log_relativity=log_rel,
            relativity=np.exp(log_rel),
            se_log_relativity=se,
            ci_lower=np.exp(log_rel - z * se),
            ci_upper=np.exp(log_rel + z * se),
            knot_covariance=V,
        )

        display = piecewise_display_term(ti, n_points=201)
        j = int(np.argmin(np.abs(np.asarray(display.x) - 1.5)))
        assert float(display.x[j]) == pytest.approx(1.5, abs=1e-12)

        expected_var = 0.25 * V[1, 1] + 2 * 0.25 * V[1, 2] + 0.25 * V[2, 2]
        expected_se = float(np.sqrt(expected_var))
        assert float(display.se_log_relativity[j]) == pytest.approx(expected_se, rel=1e-12)
        # The correct band really does differ from interpolating the knot SEs.
        interpolated = float(np.interp(1.5, knots, se))
        assert abs(expected_se - interpolated) > 0.1
        # And the drawn limits are exp(log +- z * se) evaluated on the grid.
        mid_log = float(np.interp(1.5, knots, log_rel))
        assert float(display.ci_lower[j]) == pytest.approx(
            float(np.exp(mid_log - z * expected_se)), rel=1e-12
        )
        assert float(display.ci_upper[j]) == pytest.approx(
            float(np.exp(mid_log + z * expected_se)), rel=1e-12
        )

    def test_term_inference_carries_the_knot_covariance(self):
        """The covariance that makes the exact band computable at plot time."""
        model, _ = _fitted("interior_base")
        spec = _spec(model)
        ti = model.term_inference("x")

        V = ti.knot_covariance
        assert V is not None
        assert V.shape == (spec._knots.size, spec._knots.size)
        base = spec._base_index
        assert np.all(V[base, :] == 0.0)
        assert np.all(V[:, base] == 0.0)
        np.testing.assert_allclose(V, V.T, rtol=0.0, atol=1e-18)
        np.testing.assert_allclose(
            np.sqrt(np.maximum(np.diag(V), 0.0)),
            ti.se_log_relativity,
            rtol=0.0,
            atol=1e-15,
        )

    def test_the_matplotlib_band_edges_are_dense(self):
        """The pointwise band and its edges follow the display grid."""
        model, case = _fitted("interior_base")

        fig = model.plot("x", X=case.X, sample_weight=case.sample_weight)

        panels = [ax for ax in fig.axes if ax.get_title() == "x"]
        assert panels[0].collections, "no filled CI band was drawn"
        dense_dashed = [
            line
            for line in panels[0].lines
            if line.get_linestyle() == "--" and len(line.get_xdata()) > 50
        ]
        assert len(dense_dashed) >= 2, "CI edge lines still sit on the knot grid"


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
