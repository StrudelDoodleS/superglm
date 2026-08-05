"""End-to-end reporting for an OrderedCategorical special declared as a non-str.

Every other specials fixture in the suite declares a STRING special ("MISSING",
"VR00"), where the coerced string form and the domain's own label for the level
are the same object, so the four independent membership tests downstream of
``reconstruct`` agree by accident. The two non-str tests in
``test_ordered_categorical_specials.py`` stop at ``build``/``transform`` and
never reach ``reconstruct``, which is exactly why a namespace split between
``_specials`` (coerced ``"9"``) and the reported label (``9`` / ``9.0``) shipped
as a KeyError out of ``term_inference``.

These fixtures fit and then walk the whole reporting chain -- term inference,
the ASCII/HTML summary rows, and the Excel export payload -- for both an INT
domain (``order=[1, 2, 3, 4, 5, 9]``) and a FLOAT domain
(``order=[1.0, 2.0, 9.0]``), which fail differently: the int domain crashes on
``raw["level_values"][9]`` while the float domain also desynchronises the
canonical row name (``"band[9]"`` vs ``"band[9.0]"``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from superglm import OrderedCategorical, Spline, SuperGLM
from superglm.editor import EditorSession
from superglm.export.summary import build_summary_export_payload

INT_DOMAIN: list[int] = [1, 2, 3, 4, 5, 9]
FLOAT_DOMAIN: list[float] = [1.0, 2.0, 9.0]


def _fit(domain, dtype):
    """Fit `band` over `domain` with the last level free, on a `dtype` column."""
    rng = np.random.default_rng(20260805)
    codes = np.repeat(np.arange(len(domain)), 260)
    band = np.asarray(domain, dtype=dtype)[codes]
    smooth_x = codes / (len(domain) - 1.0)
    eta = -1.2 + 1.1 * smooth_x - 0.4 * smooth_x**2
    # The special sits well off the curve so its free coefficient is real.
    eta = np.where(codes == len(domain) - 1, -1.2 - 0.6, eta)
    weights = rng.uniform(0.7, 1.5, band.size)
    y = rng.poisson(np.exp(eta) * weights).astype(float)
    X = pd.DataFrame({"band": band})
    model = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        features={
            "band": OrderedCategorical(
                order=list(domain),
                specials=[9],
                base="first",
                basis=Spline(kind="ps", k=5),
            )
        },
    )
    model.fit(X, y, sample_weight=weights)
    return model


@pytest.fixture(scope="module")
def int_model():
    return _fit(INT_DOMAIN, np.int64)


@pytest.fixture(scope="module")
def float_model():
    return _fit(FLOAT_DOMAIN, np.float64)


def _cases():
    return [
        pytest.param("int_model", INT_DOMAIN, 9, id="int-domain"),
        pytest.param("float_model", FLOAT_DOMAIN, 9.0, id="float-domain"),
    ]


@pytest.mark.parametrize(("fixture", "domain", "special"), _cases())
def test_term_inference_reports_a_non_string_special_as_a_free_level(
    request, fixture, domain, special
):
    # Crashes today: `reconstruct` reports the special under its DOMAIN label
    # (9) while `_term_ops` matches on the coerced `spec._specials` ("9"), so
    # `level_is_special` comes back all-False, the special stays in
    # `smooth_levels`, and `raw["level_values"][9]` raises KeyError -- it is
    # keyed on the smooth levels only.
    model = request.getfixturevalue(fixture)
    ti = model.term_inference("band")

    assert list(ti.levels) == list(domain)
    assert ti.level_is_special is not None
    # Assert the FLAG, not just that the call returned: the crash and the
    # silent mislabelling share one cause, and only this pins the second.
    np.testing.assert_array_equal(
        np.asarray(ti.level_is_special),
        np.array([level == special for level in domain]),
    )
    # The curve is a statement about the smooth levels alone, so the special
    # must not have taken a coordinate on the spline axis.
    assert ti.smooth_curve is not None
    assert len(ti.levels) == len(domain)


@pytest.mark.parametrize(("fixture", "domain", "special"), _cases())
def test_summary_marks_a_non_string_special_free_and_the_rest_smooth(
    request, fixture, domain, special
):
    # False today: `coef_tables` tests `level in set(spec._specials)` = {"9"},
    # and `9 in {"9"}` is False, so the free level's Fit cell reads "smooth".
    model = request.getfixturevalue(fixture)
    rows = [row for row in model.summary()._coef_rows if row.group == "band" and not row.is_spline]

    assert [row.name for row in rows] == [f"band[{level}]" for level in domain]
    assert [row.level_fit for row in rows] == ["smooth"] * (len(domain) - 1) + ["free"]

    free_row = rows[-1]
    assert free_row.se is not None
    assert np.isfinite(free_row.se) and free_row.se > 0.0


@pytest.mark.parametrize(("fixture", "domain", "special"), _cases())
def test_editor_stale_rows_mark_a_non_string_special_free(request, fixture, domain, special):
    # The FOURTH site with the same membership test, and the one no other test
    # reaches with a non-str special: `report_ops._build_editor_stale_coef_rows`
    # builds its own OC level rows for an edited model, and it too matched on
    # the string-coerced `spec._specials`. An edited model's summary would then
    # report the free level as "smooth" even after the live path was fixed.
    from superglm.model.report_ops import _build_editor_stale_coef_rows

    model = request.getfixturevalue(fixture)
    rows = _build_editor_stale_coef_rows(model)
    level_rows = [row for row in rows if row.group == "band" and not row.is_spline]

    assert [row.name for row in level_rows] == [f"band[{level}]" for level in domain]
    assert [row.level_fit for row in level_rows] == ["smooth"] * (len(domain) - 1) + ["free"]


@pytest.mark.parametrize(("fixture", "domain", "special"), _cases())
def test_export_payload_kinds_a_non_string_special_as_a_free_level(
    request, fixture, domain, special
):
    # False today on BOTH counts: `level_fit` is wrong upstream (so `kind`
    # would be "level"), and on the float domain `_canonical_level_row_names`
    # spells the row "band[9]" from `_ordered_levels` while the coef row is
    # named "band[9.0]" from `raw["levels"]`, so the row is not recognised as a
    # level row at all and falls through to kind="coefficient".
    model = request.getfixturevalue(fixture)
    payload = build_summary_export_payload(model)
    by_term = {row.term: row for row in payload.terms}

    for level in domain[:-1]:
        assert by_term[f"band[{level}]"].kind == "level"
    assert by_term[f"band[{special}]"].kind == "free level"


@pytest.mark.parametrize(("fixture", "domain", "special"), _cases())
def test_a_non_string_special_predicts_and_scores_through_the_free_block(
    request, fixture, domain, special
):
    # The free level's reported relativity must be the one predict() uses; a
    # namespace split that leaves the special inside the smooth would report a
    # curve value here while the design still scores it off the indicator.
    model = request.getfixturevalue(fixture)
    ti = model.term_inference("band")
    rels = dict(zip(list(ti.levels), np.asarray(ti.log_relativity).tolist()))

    frame = pd.DataFrame({"band": np.asarray(domain, dtype=type(domain[0]))})
    eta = np.log(np.asarray(model.predict(frame), dtype=np.float64))
    base_eta = eta[0]
    np.testing.assert_allclose(
        eta - base_eta,
        np.array([rels[level] for level in domain]),
        atol=1e-8,
    )


# ── Editor: the namespace split stopped here ──────────────────────────────────


@pytest.mark.parametrize(
    ("domain", "dtype"), [(INT_DOMAIN, np.int64), (FLOAT_DOMAIN, np.float64)], ids=["int", "float"]
)
def test_a_non_string_special_survives_an_editor_round_trip(domain, dtype):
    # False before this fix on the FLOAT domain. `_apply_ordered_spline_term`
    # matched the str-coerced `spec._specials` ("9") against the editable term's
    # own row labels, which are in the display namespace ("9.0"), so the level was
    # reported missing and the edit refused on data that plainly contains it.
    # The int domain happened to work because both spellings agree there -- which
    # is exactly why the earlier non-string tests missed it.
    model = _fit(domain, dtype)
    X = pd.DataFrame({"band": np.asarray(domain, dtype=dtype)})
    levels = [str(level) for level in model.term_inference("band").levels]

    session = EditorSession.from_model(model, terms=["band"])
    session.select_levels("band", [levels[-1]])
    session.shift("band", 0.3)
    edited = session.to_model()

    delta = np.asarray(edited._predict_eta_exact(X)) - np.asarray(model._predict_eta_exact(X))
    is_special = (X["band"].astype(str) == levels[-1]).to_numpy()
    assert delta[is_special] == pytest.approx(0.3, abs=1e-9)
    assert delta[~is_special] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize(
    ("domain", "dtype"), [(INT_DOMAIN, np.int64), (FLOAT_DOMAIN, np.float64)], ids=["int", "float"]
)
def test_collapsing_a_non_string_special_is_still_refused(domain, dtype):
    # `_require_no_special_members` is a guard, and on a float domain it was INERT:
    # it compared the str-coerced `_specials` ("9") against member labels in the
    # display namespace ("9.0"), matched nothing, and let a special through.
    #
    # The match is deliberately anchored on THIS guard's wording. A looser
    # "free level" pattern passes even with the guard neutered, because the
    # grouping-coverage check downstream raises its own ValueError that also
    # contains that phrase -- so the test would report a guard as working while
    # measuring a different one entirely. (That downstream check does mean an
    # inert guard is not exploitable today; this is defence in depth, and the
    # point is that it must fail for its OWN reason.)
    model = _fit(domain, dtype)
    levels = [str(level) for level in model.term_inference("band").levels]

    session = EditorSession.from_model(model, terms=["band"])
    with pytest.raises(ValueError, match="collapse .* cannot include free level"):
        session.select_levels("band", [levels[-2], levels[-1]])
        session.replace_with_collapsed_levels("band")
