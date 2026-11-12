from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager

import numpy as np
import pandas as pd
import pytest
from tests._lss_editor_fixtures import lognormal_case, open_case
from tests.editor._chart_layout import wait_for_chart_to_fit

from superglm import Categorical, OrderedCategorical, Spline, SuperGLM
from superglm.editor import EditorSession


def _browser_rows():
    """The browser book's rows, its signal on the linear predictor, and its generator."""
    rng = np.random.default_rng(20260711)
    territory_levels = [f"T{i:02d}" for i in range(1, 11)]
    age_band_levels = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
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
    age_band = rng.choice(age_band_levels, n)
    long_category = rng.choice(long_levels, n)
    eta = (
        0.5
        + 0.12 * np.sin(curve)
        + 0.03 * np.array([territory_levels.index(value) for value in territory])
        + 0.08 * np.array([age_band_levels.index(value) for value in age_band])
    )
    frame = pd.DataFrame(
        {
            "curve": curve,
            "territory": territory,
            "age_band": age_band,
            "long_category": long_category,
        }
    )
    return rng, frame, eta, age_band_levels


def _browser_model(family: str, frame: pd.DataFrame, y: np.ndarray, age_band_levels) -> SuperGLM:
    model = SuperGLM(
        family=family,
        selection_penalty=0.0,
        spline_penalty=0.1,
        features={
            "curve": Spline(kind="ps", n_knots=7),
            "territory": Categorical(base="first"),
            "age_band": OrderedCategorical(
                order=age_band_levels,
                basis=Spline(kind="ps", k=5),
                base="first",
            ),
            "long_category": Categorical(base="first"),
        },
    )
    model.fit(frame, y)
    return model


@pytest.fixture(scope="session")
def editor_browser_model() -> SuperGLM:
    """The model behind every browser editor page.

    Poisson on a log link, not gaussian: one of these tests drives the export
    dialog, and the rating-table export it reaches is multiplicative -- base
    times one relativity per block -- so it accepts log-link models only.  The
    same signal is carried through ``exp`` into a count, so the terms, the
    level labels and the shape of every curve the page draws are unchanged.
    """
    rng, frame, eta, age_band_levels = _browser_rows()
    y = rng.poisson(np.exp(eta)).astype(np.float64)
    return _browser_model("poisson", frame, y, age_band_levels)


@pytest.fixture(scope="session")
def editor_identity_model() -> SuperGLM:
    """The same book fitted as a gaussian on the identity link, a response in the thousands.

    Its curves are additive: each term's effect is drawn as an amount around zero.
    """
    rng, frame, eta, age_band_levels = _browser_rows()
    y = 1000.0 * eta + rng.normal(0.0, 20.0, len(frame))
    return _browser_model("gaussian", frame, y, age_band_levels)


@pytest.fixture(scope="session")
def chromium_browser():
    playwright_api = pytest.importorskip("playwright.sync_api")
    with playwright_api.sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        yield browser
        browser.close()


def choose_feature(page, term: str) -> None:
    """Pick a term from the feature list as an analyst would.

    Below 1278px the list opens collapsed to a strip, so open it, click the
    row, and collapse it again to leave the layout under test as it was.
    """
    collapsed = page.locator("#featureList").get_attribute("data-open") == "false"
    if collapsed:
        page.locator("#featureListToggle").click()
    page.locator(f'#featureList [data-term="{term}"]').click()
    if collapsed:
        page.locator("#featureListToggle").click()


@pytest.fixture(name="choose_feature")
def choose_feature_fixture():
    return choose_feature


def _open_session_page(
    resources: ExitStack,
    chromium_browser,
    session: EditorSession,
    *,
    viewport: dict[str, int] | None,
    prepare: Callable[[object], None] | None,
    first_term: str,
    selected_term: str,
):
    """Serve ``session``, open it in a page, and wait until ``selected_term`` is drawn."""
    widget = session.widget()
    resources.callback(widget.close)
    page = chromium_browser.new_page(viewport=viewport or {"width": 1180, "height": 720})
    resources.callback(page.close)
    # A route handler still fetching when the page closes would raise on the next page opened;
    # the handlers go first, and their in-flight requests are dropped.
    resources.callback(page.unroute_all, behavior="ignoreErrors")
    if prepare is not None:
        prepare(page)
    page.goto(f"{widget.app_url}&test=1", wait_until="domcontentloaded")
    page.locator("#chart path.edited").first.wait_for()
    page.wait_for_function(
        "term => document.querySelector('#status')?.dataset.term === term",
        arg=first_term,
    )
    if selected_term != first_term:
        # Tests go on to read and change the session, so Python has to have the term.
        with page.expect_response(
            lambda response: (
                response.request.method == "POST"
                and response.url.split("?", maxsplit=1)[0].endswith("/select")
            )
        ):
            choose_feature(page, selected_term)
        page.wait_for_function(
            "term => document.querySelector('#status')?.dataset.term === term",
            arg=selected_term,
        )
        page.locator("#chart path.edited").first.wait_for()
    # The chart is drawn at its panel's size, and the panel settles a
    # frame after the feature list toggles or the metrics render.
    wait_for_chart_to_fit(page)
    return page


@pytest.fixture
def open_editor_page(chromium_browser, editor_browser_model):
    opened: list[ExitStack] = []

    @contextmanager
    def open_page(
        *,
        viewport: dict[str, int] | None = None,
        selected_term: str = "curve",
        n_points: int = 200,
        collapsed_levels: tuple[str, tuple[str, ...]] | None = None,
        prepare: Callable[[object], None] | None = None,
        model: SuperGLM | None = None,
    ) -> Iterator[tuple[object, EditorSession]]:
        resources = ExitStack()
        opened.append(resources)
        try:
            session = EditorSession.from_model(
                editor_browser_model if model is None else model,
                terms=["curve", "territory", "age_band", "long_category"],
                n_points=n_points,
            )
            if collapsed_levels is not None:
                collapsed_term, levels = collapsed_levels
                session.select_levels(collapsed_term, levels)
                session.replace_with_collapsed_levels(collapsed_term, method="fit")
            page = _open_session_page(
                resources,
                chromium_browser,
                session,
                viewport=viewport,
                prepare=prepare,
                first_term="curve",
                selected_term=selected_term,
            )
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


@pytest.fixture(scope="session")
def lss_browser_case():
    """The fitted log-normal SuperLSS and its rows; sessions open over copies."""
    return lognormal_case()


@pytest.fixture
def open_lss_editor_page(chromium_browser, lss_browser_case):
    opened: list[ExitStack] = []

    @contextmanager
    def open_page(
        *,
        viewport: dict[str, int] | None = None,
        selected_term: str = "mean:age",
        prepare: Callable[[object], None] | None = None,
        cv=None,
        case=None,
    ) -> Iterator[tuple[object, EditorSession]]:
        resources = ExitStack()
        opened.append(resources)
        try:
            session = open_case(lss_browser_case if case is None else case, cv=cv)
            page = _open_session_page(
                resources,
                chromium_browser,
                session,
                viewport=viewport,
                prepare=prepare,
                first_term=next(iter(session.terms)),
                selected_term=selected_term,
            )
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
