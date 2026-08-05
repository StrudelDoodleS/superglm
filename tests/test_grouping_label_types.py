"""Issue #237: a grouping declared with one label spelling against a column that
renders levels differently (int order, float column) canonicalises both sides
independently and they never meet."""

import numpy as np
import pandas as pd
import pytest

from superglm import OrderedCategorical, Spline, SuperGLM
from superglm.editor import EditorSession
from superglm.features.grouping import collapse_levels


def _model(order, dtype):
    rng = np.random.default_rng(4)
    codes = rng.choice(np.asarray(order, dtype=dtype), 600)
    X = pd.DataFrame({"band": codes})
    y = 0.1 * codes.astype(float) + rng.normal(0.0, 0.15, 600)
    m = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"band": OrderedCategorical(order=list(order), basis=Spline(kind="ps", k=5))},
    )
    m.fit(X, y)
    return m, X


def test_collapse_levels_reconciles_declared_labels_with_the_data_spelling():
    # `order=` and `groups=` are declared as ints; the column is float, so the
    # data spells the same levels "1.0".."9.0". Both sides were canonicalised
    # independently and never reconciled.
    data = np.array([1.0, 2.0, 3.0, 4.0, 9.0, 9.0])
    g = collapse_levels(data, groups={"2+3": [2, 3]}, order=[1, 2, 3, 4, 9])
    assert set(g.all_original_levels) == {"1.0", "2.0", "3.0", "4.0", "9.0"}
    assert g.original_to_group["2.0"] == "2+3"
    assert g.original_to_group["3.0"] == "2+3"
    assert g.original_to_group["1.0"] == "1.0"


def test_a_genuinely_absent_level_is_still_rejected():
    # The reconciliation must not become a catch-all: a level that really is not
    # in the data must still raise, or the check stops meaning anything.
    data = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="not found in data"):
        collapse_levels(data, groups={"g": [7]}, order=[1, 2, 3])


@pytest.mark.parametrize("dtype", [np.float64, np.int64], ids=["float-col", "int-col"])
def test_an_ordered_categorical_with_numeric_labels_survives_a_collapse(dtype):
    # The end-to-end shape from issue #237: numeric bands are a natural way to
    # declare an ordered factor and collapsing levels is a core editor action.
    model, X = _model([1, 2, 3, 4, 9], dtype)
    session = EditorSession.from_model(model, terms=["band"])
    session.select_levels("band", ["2", "3"])
    refit = session.replace_with_collapsed_levels("band")

    # A grouped term DISPLAYS expanded back to its original levels, so the merged
    # label is not in `.levels`. What shows the merge is the grouping itself, and
    # the two members sharing one fitted effect.
    spec = refit._specs["band"]
    assert spec._grouping is not None
    merged = [members for members in spec._grouping.group_to_originals.values() if len(members) > 1]
    assert len(merged) == 1 and len(merged[0]) == 2, spec._grouping.group_to_originals

    ti = refit.term_inference("band")
    by_level = dict(zip([str(lv) for lv in ti.levels], np.asarray(ti.log_relativity)))
    a, b = (str(m) for m in merged[0])
    assert by_level[a] == pytest.approx(by_level[b]), "collapsed levels must share one effect"
    assert np.all(np.isfinite(np.asarray(refit._predict_eta_exact(X))))
