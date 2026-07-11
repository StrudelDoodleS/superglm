from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from playwright.sync_api import sync_playwright

from superglm import Categorical, Spline, SuperGLM
from superglm.editor import EditorSession


@pytest.fixture
def browser_editor_widget():
    rng = np.random.default_rng(20260711)
    n = 120
    X = pd.DataFrame(
        {
            "age": rng.uniform(18.0, 80.0, n),
            "region": rng.choice(["A", "B", "C"], n),
        }
    )
    y = 0.3 + 0.01 * X["age"].to_numpy() + 0.2 * (X["region"] == "B")
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"age": Spline(n_knots=7), "region": Categorical(base="first")},
    )
    model.fit(X, y)
    widget = EditorSession.from_model(model, terms=["age", "region"]).widget()
    try:
        yield widget
    finally:
        widget.close()


@pytest.mark.browser
def test_editor_browser_loads_authoritative_state(browser_editor_widget):
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1180, "height": 720})
        page.goto(browser_editor_widget.app_url)
        page.locator("#chart .edited").first.wait_for()
        assert page.locator("#term").input_value() == "age"
        assert page.locator("#status").get_attribute("style") in (None, "")
        browser.close()
