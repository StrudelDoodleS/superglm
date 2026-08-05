"""Editor behaviour for OrderedCategorical terms that declare specials."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from superglm import OrderedCategorical, Spline, SuperGLM
from superglm.editor import EditorSession

SMOOTH_LEVELS = ["1", "2", "3", "4", "5", "6"]
SMOOTH_EFFECT = {"1": -0.30, "2": -0.18, "3": -0.05, "4": 0.06, "5": 0.15, "6": 0.20}
ONE_SPECIAL = (["MISSING"], [0.14] * 6 + [0.16], {**SMOOTH_EFFECT, "MISSING": 0.55})
TWO_SPECIALS = (
    ["MISSING", "UNKNOWN"],
    [0.14] * 6 + [0.10, 0.06],
    {**SMOOTH_EFFECT, "MISSING": 0.55, "UNKNOWN": -0.40},
)


def _fit(specials, probabilities, effects, *, select=False):
    rng = np.random.default_rng(20260805)
    labels = rng.choice(SMOOTH_LEVELS + specials, 900, p=probabilities)
    X = pd.DataFrame({"band": labels})
    y = np.array([effects[label] for label in labels]) + rng.normal(0.0, 0.15, 900)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={
            "band": OrderedCategorical(
                order=SMOOTH_LEVELS,
                specials=specials,
                basis=Spline(kind="ps", k=8, select=select),
            )
        },
    )
    model.fit(X, y)
    return model, X, y


@pytest.fixture
def specials_model():
    return _fit(*ONE_SPECIAL)


@pytest.fixture
def two_specials_model():
    return _fit(*TWO_SPECIALS)


def _band_blocks(model):
    groups = {str(group.name): group for group in model._groups if group.feature_name == "band"}
    return groups["band"], groups["band:special"]


def _edit_special(model, level, delta):
    session = EditorSession.from_model(model, terms=["band"])
    session.select_levels("band", [level])
    session.shift("band", delta)
    edited = session.to_model()
    spline_group, special_group = _band_blocks(edited)
    return (
        float(edited.result.intercept),
        np.asarray(edited.result.beta[spline_group.sl], dtype=np.float64).copy(),
        np.asarray(edited.result.beta[special_group.sl], dtype=np.float64).copy(),
    )


# `select=True` is the discriminating shape.  Without it the spline block is
# exactly orthogonal to the level-weight vector (its columns are weight-centered
# at build time), so the constant vector never lies in the smooth block's column
# space, the ordered rows pin the intercept on their own, and the joint solve
# already lands on the exact-assignment answer.  `select=True` adds an
# unpenalized null-space column that is NOT weight-centered; the constant then
# IS in the span, the intercept goes free, and today's `_apply_projected_term`
# min-norm solve splits the special's edit across the intercept and the spline.
@pytest.mark.parametrize("select", [False, True])
def test_special_edit_leaves_the_ordered_projection_untouched(select):
    model, _, _ = _fit(*ONE_SPECIAL, select=select)

    small = _edit_special(model, "MISSING", 0.5)
    large = _edit_special(model, "MISSING", 1.5)

    # Only the ordered levels go through the least-squares projection, so the
    # SIZE of a special's edit may not move the intercept or the spline block.
    assert small[0] == pytest.approx(large[0], abs=1e-12)
    np.testing.assert_allclose(small[1], large[1], atol=1e-12)
    # ...and the special's own coefficient carries the whole difference.
    np.testing.assert_allclose(large[2] - small[2], 1.0, atol=1e-12)


@pytest.mark.parametrize("select", [False, True])
def test_editing_one_special_leaves_the_other_special_coefficient_alone(select):
    model, _, _ = _fit(*TWO_SPECIALS, select=select)

    small = _edit_special(model, "UNKNOWN", 0.5)
    large = _edit_special(model, "UNKNOWN", 1.5)

    assert small[0] == pytest.approx(large[0], abs=1e-12)
    np.testing.assert_allclose(small[1], large[1], atol=1e-12)
    # Special indicator column j belongs to _specials[j] = ["MISSING", "UNKNOWN"].
    # MISSING was not edited and its coefficient is set from its own effect, so
    # it is identical across the two runs. Under the joint min-norm solve the
    # shared intercept column drags it with UNKNOWN's edit.
    np.testing.assert_allclose(small[2][0], large[2][0], atol=1e-12)
    np.testing.assert_allclose(large[2][1] - small[2][1], 1.0, atol=1e-12)


def test_a_special_with_no_editable_row_is_refused(specials_model):
    from superglm.editor.apply import apply_edits_to_model_copy

    model, _, _ = specials_model
    session = EditorSession.from_model(model, terms=["band"])
    term = session.terms["band"].copy()
    terms = {"band": term}
    assert term.levels is not None
    keep = [i for i, level in enumerate(term.levels) if level != "MISSING"]
    term.levels = [term.levels[i] for i in keep]
    term.metadata.pop("native_levels", None)
    term.original_log_effect = term.original_log_effect[keep]
    term.edited_log_effect = term.edited_log_effect[keep]
    if term.weights is not None:
        term.weights = term.weights[keep]
    # Only terms whose effects actually moved are applied, so make one.
    term.edited_log_effect = term.edited_log_effect + 0.25

    # Without a row for 'MISSING' the special's indicator column is all-zero, so
    # the joint least-squares solve silently assigns it a min-norm 0.0 and wipes
    # the fitted free-level effect instead of refusing the edit.
    with pytest.raises(ValueError, match="MISSING"):
        apply_edits_to_model_copy(model, terms)


def _term_and_indices(model, levels):
    session = EditorSession.from_model(model, terms=["band"])
    term = session.terms["band"]
    return term, np.array([term.levels.index(level) for level in levels], dtype=np.intp)


def test_collapse_refuses_a_group_that_mixes_a_special_with_ordered_levels(specials_model):
    from superglm.editor.collapse import collapsed_feature_spec

    model, X, _ = specials_model
    term, idx = _term_and_indices(model, ["6", "MISSING"])

    # "MISSING" sits last in _ordered_levels, so it is *adjacent* to "6" and the
    # contiguity check at collapse.py:58-64 waves this selection through today.
    with pytest.raises(ValueError, match="free level"):
        collapsed_feature_spec(model, term, idx, X=X)


def test_collapsing_ordered_levels_keeps_the_special_free(specials_model):
    from superglm.editor.collapse import collapsed_feature_spec

    model, X, _ = specials_model
    term, idx = _term_and_indices(model, ["2", "3"])

    replacement, metadata = collapsed_feature_spec(model, term, idx, X=X)

    assert metadata["group_label"] == "2+3"
    # _ordered_spec_with_grouping rebuilds the spec from an explicit argument
    # list (collapse.py:367-372); without specials= the free level is silently
    # smoothed back into the curve.
    assert replacement._specials == ["MISSING"]
    assert replacement._smooth_levels == ["1", "2+3", "4", "5", "6"]
    assert replacement._ordered_levels == ["1", "2+3", "4", "5", "6", "MISSING"]
    assert "MISSING" not in replacement._level_to_value
