"""A level the fit does not determine, drawn in the browser."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from tests.editor.conftest import _browser_rows

from superglm import Categorical, Numeric, OrderedCategorical, Spline, SuperGLM

pytest.importorskip("playwright.sync_api")
pytestmark = pytest.mark.browser


@pytest.fixture(scope="module")
def aliased_level_model() -> SuperGLM:
    """The browser book with ``t10``, the indicator of territory ``T10``, beside ``territory``.

    ``T10``'s relativity and ``t10`` trade along a direction the rows do not identify; the
    published constraint holds ``t10`` (the later term), so ``T10``'s point leans on that alias.
    """
    rng, frame, eta, age_band_levels = _browser_rows()
    y = rng.poisson(np.exp(eta)).astype(np.float64)
    frame = frame.assign(t10=(frame["territory"] == "T10").astype(np.float64))
    model = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        spline_penalty=0.1,
        features={
            "curve": Spline(kind="ps", n_knots=7),
            "territory": Categorical(base="first"),
            "age_band": OrderedCategorical(
                order=age_band_levels, basis=Spline(kind="ps", k=5), base="first"
            ),
            "long_category": Categorical(base="first"),
            "t10": Numeric(),
        },
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(frame, y)
    return model


def test_a_level_the_fit_does_not_determine_is_hollow_and_has_no_interval(
    open_editor_page, aliased_level_model
):
    """``T10`` is drawn hollow, names its kind and says why on hover; it has no interval bar.

    The payload gives ``T10`` the kind ``"alias"`` and no interval (``null`` ends); the other
    nine levels keep their bars.  At ``8cf0bca5d`` the point was drawn as any other, and the
    chart drew a bar for it from its null ends.
    """
    with open_editor_page(selected_term="territory", model=aliased_level_model) as (page, _):
        page.locator("#ciToggle").click()
        page.locator("#chart .ci-whisker").first.wait_for(state="attached")
        flagged = page.locator("#chart .point.non-estimable")
        assert flagged.count() == 1
        assert flagged.get_attribute("data-non-estimable") == "alias"
        assert flagged.get_attribute("data-index") == "9"
        at = float(flagged.get_attribute("cx"))
        bars = page.locator("#chart .ci-whisker").evaluate_all(
            "els => els.filter(e => e.getAttribute('x1') === e.getAttribute('x2'))"
            ".map(e => Number(e.getAttribute('x1')))"
        )
        assert len(bars) == 9
        assert all(abs(x - at) > 1.0 for x in bars)
        flagged.hover()
        tooltip = page.locator("#chart .point-tooltip")
        tooltip.wait_for(state="attached")
        assert "Not determined: no interval" in tooltip.text_content()
        assert "aliased: held by the constraint" in tooltip.text_content()
