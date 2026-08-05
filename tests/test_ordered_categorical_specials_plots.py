"""Plot rendering of OrderedCategorical special (free) levels, both backends."""

import importlib.util

import numpy as np
import pandas as pd
import pytest

from superglm import Numeric, OrderedCategorical, Spline, SuperGLM

PLOTLY_AVAILABLE = importlib.util.find_spec("plotly") is not None

BANDS = [str(i) for i in range(1, 11)]


@pytest.fixture
def specials_model():
    """Ten ordered bands on a saturating curve plus an 18% free MISSING level."""
    rng = np.random.default_rng(20260805)
    n = 4000
    band = rng.choice([*BANDS, "MISSING"], n, p=[0.082] * 10 + [0.18])
    idx = np.array([BANDS.index(b) if b in BANDS else -1 for b in band], dtype=np.float64)
    log_effect = np.where(idx >= 0, 0.6 * np.sqrt(np.maximum(idx, 0.0) / 9.0), np.log(0.577))
    x = rng.normal(size=n)
    sample_weight = rng.uniform(0.5, 1.0, n)
    mu = np.exp(-2.0 + log_effect + 0.1 * x)
    y = rng.poisson(mu * sample_weight).astype(float)
    frame = pd.DataFrame({"band": band, "x": x})
    model = SuperGLM(
        features={
            "band": OrderedCategorical(
                order=BANDS, specials=["MISSING"], basis=Spline(kind="ps", k=6)
            ),
            "x": Numeric(),
        }
    )
    model.fit(frame, y, sample_weight=sample_weight)
    return model, frame, sample_weight


def test_term_inference_marks_specials_and_keeps_level_x_smooth_only(specials_model):
    # False today: TermInference has no level_is_special at all, and _term_ops.py
    # builds level_x by looking every level up in raw["level_values"], which has no
    # entry for MISSING — so today this raises KeyError before any figure exists.
    model, _, _ = specials_model
    ti = model.term_inference("band")

    assert list(ti.levels) == [*BANDS, "MISSING"]
    assert ti.level_is_special is not None
    np.testing.assert_array_equal(ti.level_is_special, [False] * 10 + [True])
    assert len(ti.relativity) == 11
    assert len(ti.smooth_curve.level_x) == 10
    # with_se defaults to True (api.py:1017) and both plot backends need the band:
    # the curve SE must exist and be finite, not silently vanish or crash.
    assert ti.smooth_curve.se_log_relativity is not None
    assert np.all(np.isfinite(ti.smooth_curve.se_log_relativity))
    assert len(ti.smooth_curve.se_log_relativity) == len(ti.smooth_curve.x)


def test_predict_reproduces_the_reported_relativities_for_bands_and_specials(specials_model):
    # The only end-to-end guard on build -> fit -> score: model/base.py prefers
    # `spec.score` over `spec.transform`, so score() is the whole of predict()
    # for an OrderedCategorical. Nothing else asserts that the fitted smooth
    # survives the split — dropping the ordered branch of score() would leave
    # every band predicting the same thing while the unit tests stayed green.
    model, _, _ = specials_model
    ti = model.term_inference("band")
    rel = dict(zip([str(lv) for lv in ti.levels], np.asarray(ti.relativity, dtype=float)))

    probe = pd.DataFrame({"band": ["1", "10", "MISSING"], "x": [0.0, 0.0, 0.0]})
    pred = np.asarray(model.predict(probe), dtype=float)

    # Two ordered bands must differ from each other in the ratio the term
    # reports for them, and the special in its own.
    assert pred[1] / pred[0] == pytest.approx(rel["10"] / rel["1"], rel=1e-8)
    assert pred[2] / pred[0] == pytest.approx(rel["MISSING"] / rel["1"], rel=1e-8)
    # ...and that ratio is not 1, so the assertions above are not vacuous.
    assert abs(rel["10"] / rel["1"] - 1.0) > 0.1
    assert abs(rel["MISSING"] / rel["1"] - 1.0) > 0.1


@pytest.mark.xfail(reason="display of free levels lands in Task 8", strict=True)
def test_plotting_a_specials_model_lands_in_task_8(specials_model):
    # Task 4 shortens SmoothCurve.level_x to the smooth levels while relativity
    # still carries all of them, so main_effects.py pairs an 11-vector with 10
    # positions and matplotlib raises. Task 8 owns the display change; this
    # xfail is strict so that landing it flips this test to a failure and the
    # gap cannot merge unnoticed.
    model, _, _ = specials_model
    assert model.plot("band") is not None


def test_term_inference_level_is_special_is_none_without_specials():
    # False today: the attribute does not exist, so this is an AttributeError.
    rng = np.random.default_rng(7)
    n = 1500
    band = rng.choice(BANDS, n)
    idx = np.array([BANDS.index(b) for b in band], dtype=np.float64)
    sample_weight = rng.uniform(0.5, 1.0, n)
    y = rng.poisson(np.exp(-2.0 + 0.4 * idx / 9.0) * sample_weight).astype(float)
    frame = pd.DataFrame({"band": band})
    model = SuperGLM(
        features={"band": OrderedCategorical(order=BANDS, basis=Spline(kind="ps", k=5))}
    )
    model.fit(frame, y, sample_weight=sample_weight)

    ti = model.term_inference("band")
    assert ti.level_is_special is None
    assert len(ti.smooth_curve.level_x) == 10
