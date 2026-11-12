"""Reporting regressions from the #626 follow-up handoff (items 14–15)."""

from contextlib import nullcontext

import numpy as np
import pandas as pd
import pytest

from superglm import Categorical, Numeric, SuperGLM, collapse_levels
from superglm.distributional import Predictor
from superglm.distributional import terms as lss_terms
from superglm.distributional.families.gaussian import GaussianLS
from tests.bound_predictor_fixtures import model_from_templates


def _interaction_model(second="special"):
    rng = np.random.default_rng(20261112)
    x = rng.uniform(-1, 1, 180)
    z = rng.normal(size=len(x))
    frame = pd.DataFrame({"x": x, second: z})
    y = 0.3 + 0.7 * x - 0.2 * z + 0.5 * x * z + rng.normal(scale=0.2, size=len(x))
    model = model_from_templates(
        family=GaussianLS(),
        predictors=[
            Predictor(
                "location", {"x": Numeric(), second: Numeric()}, interactions=(("x", second),)
            ),
            Predictor("scale", {}),
        ],
    ).fit(frame, y)
    return model, frame


def test_lss_summary_keeps_an_interaction_named_special():
    model, frame = _interaction_model()
    reference, _ = _interaction_model(second="z")
    actual = model.summary()
    expected = reference.summary()
    expected["term"] = expected["term"].replace({"z": "special", "x:z": "x:special"})
    pd.testing.assert_frame_equal(actual, expected)
    assert "x:special" in set(actual.term)
    assert "x (special level)" not in set(actual.term)
    np.testing.assert_array_equal(
        model.predict(frame), reference.predict(frame.rename(columns={"special": "z"}))
    )


def test_lss_numeric_effect_does_not_absorb_a_special_named_interaction(monkeypatch):
    model, frame = _interaction_model()
    fitted = model._require_fitted()
    effect = model.term_inference("location", "x", simultaneous=False)
    block = fitted.layout.term_slices["location:x"]
    beta = float(fitted.coefficients[block][0])
    variance = float(fitted.inference.covariance[block, block][0, 0])
    delta = effect.x - np.median(frame.x)
    np.testing.assert_allclose(effect.effect, beta * delta, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(effect.se, np.sqrt(variance) * np.abs(delta), rtol=1e-12, atol=1e-14)
    assert effect.special is None

    # Repairing the main effect does not withhold an unrelated interaction's test.
    monkeypatch.setattr(lss_terms, "repaired_terms", lambda _: ("location:x",))
    assert lss_terms.repair_covering(fitted, "location:x:special") is None


def test_lss_summary_keeps_a_main_column_named_special():
    rng = np.random.default_rng(73)
    frame = pd.DataFrame({"x": rng.normal(size=120), "x:special": rng.normal(size=120)})
    y = frame.x + 0.7 * frame["x:special"] + rng.normal(scale=0.2, size=len(frame))
    model = model_from_templates(
        family=GaussianLS(),
        predictors=[
            Predictor("location", {"x": Numeric(), "x:special": Numeric()}),
            Predictor("scale", {}),
        ],
    ).fit(frame, y)
    table = model.summary()
    assert "x:special" in set(table.term)
    assert "x (special level)" not in set(table.term)


@pytest.mark.parametrize("with_se", [False, True])
@pytest.mark.parametrize("centering", ["native", "mean"])
@pytest.mark.parametrize("declared", [False, True])
def test_grouped_term_inference_omits_only_groups_absent_from_the_fit(with_se, centering, declared):
    # A grouping can be built on the book and reused on a subset. B is absent
    # but belongs to fitted AB; the missing level's entire group has no rows.
    grouping = collapse_levels(
        ["base", "A", "B", "missing"], groups={"AB": ["A", "B"], "empty": ["missing"]}
    )
    frame = pd.DataFrame({"g": ["base", "A"] * 60})
    y = np.tile([0.3, 1.0], 60) + np.random.default_rng(19).normal(scale=0.1, size=len(frame))
    levels = ["base", "A", "B", "missing"] if declared else None
    pin_warning = pytest.warns(UserWarning, match="pinned to base") if declared else nullcontext()
    with pin_warning:
        model = SuperGLM(
            family="gaussian",
            selection_penalty=0,
            features={"g": Categorical(base="base", grouping=grouping, levels=levels)},
        ).fit(frame, y)
    effect = model.term_inference("g", with_se=with_se, centering=centering)
    assert effect.levels == ["A", "B", "base", *(["missing"] if declared else [])]
    beta = float(y[1::2].mean() - y[::2].mean())
    contrast = np.array([1.0, 1.0, 0.0, *([0.0] if declared else [])])
    if centering == "mean":
        contrast -= 1 / (3 if declared else 2)  # center the fitted universe before expansion
    np.testing.assert_allclose(effect.log_relativity, beta * contrast, rtol=1e-12, atol=1e-14)
    if with_se:
        residual = y - np.tile([y[::2].mean(), y[1::2].mean()], 60)
        variance = residual @ residual / (len(y) - 2)
        se_beta = np.sqrt(variance * (1 / 60 + 1 / 60))
        expected_se = se_beta * np.abs(contrast)
        np.testing.assert_allclose(effect.se_log_relativity, expected_se, rtol=1e-12, atol=1e-14)
    else:
        assert effect.se_log_relativity is None
