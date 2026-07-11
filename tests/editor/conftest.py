from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager

import numpy as np
import pandas as pd
import pytest

from superglm import Categorical, Spline, SuperGLM
from superglm.editor import EditorSession


@pytest.fixture(scope="session")
def editor_browser_model() -> SuperGLM:
    rng = np.random.default_rng(20260711)
    territory_levels = [f"T{i:02d}" for i in range(1, 11)]
    long_levels = [
        "MyReallyLongCategoryNameThatWouldNeverFit",
        "CommercialVehicleWithSpecialistUsage",
        "PrivateMotorStandard",
        "AgriculturalMachinery",
        "MotorcycleAndScooter",
        "TaxiAndPrivateHire",
        "FleetLightCommercial",
        "FleetHeavyCommercial",
        "ClassicAndCollectable",
        "Family👨‍👩‍👧‍👦DriverCaféCategory",
    ]
    n = 500
    curve = rng.uniform(0.0, 10.0, n)
    territory = rng.choice(territory_levels, n)
    long_category = rng.choice(long_levels, n)
    y = (
        0.5
        + 0.12 * np.sin(curve)
        + 0.03 * np.array([territory_levels.index(value) for value in territory])
        + rng.normal(0.0, 0.04, n)
    )
    frame = pd.DataFrame(
        {
            "curve": curve,
            "territory": territory,
            "long_category": long_category,
        }
    )
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        spline_penalty=0.1,
        features={
            "curve": Spline(n_knots=7),
            "territory": Categorical(base="first"),
            "long_category": Categorical(base="first"),
        },
    )
    model.fit(frame, y)
    return model


@pytest.fixture(scope="session")
def chromium_browser():
    playwright_api = pytest.importorskip("playwright.sync_api")
    with playwright_api.sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        yield browser
        browser.close()


@pytest.fixture
def open_editor_page(chromium_browser, editor_browser_model):
    opened: list[ExitStack] = []

    @contextmanager
    def open_page(
        *,
        viewport: dict[str, int] | None = None,
        selected_term: str = "curve",
    ) -> Iterator[tuple[object, EditorSession]]:
        resources = ExitStack()
        opened.append(resources)
        try:
            session = EditorSession.from_model(
                editor_browser_model,
                terms=["curve", "territory", "long_category"],
            )
            widget = session.widget()
            resources.callback(widget.close)
            page = chromium_browser.new_page(viewport=viewport or {"width": 1180, "height": 720})
            resources.callback(page.close)
            page.goto(f"{widget.app_url}&test=1", wait_until="domcontentloaded")
            page.locator("#chart path.edited").first.wait_for()
            page.wait_for_function(
                "term => document.querySelector('#status')?.dataset.term === term",
                arg="curve",
            )
            if selected_term != "curve":
                with page.expect_response(
                    lambda response: (
                        response.request.method == "POST"
                        and response.url.split("?", maxsplit=1)[0].endswith("/term")
                    )
                ):
                    page.select_option("#term", selected_term)
                page.wait_for_function(
                    "term => document.querySelector('#status')?.dataset.term === term",
                    arg=selected_term,
                )
                page.locator("#chart path.edited").first.wait_for()
            yield page, session
        finally:
            if resources in opened:
                opened.remove(resources)
            resources.close()

    try:
        yield open_page
    finally:
        with ExitStack() as teardown:
            for resources in opened:
                teardown.callback(resources.close)
            opened.clear()
