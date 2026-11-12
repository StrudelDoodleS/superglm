"""The Explain tab of a SuperLSS session, driven from the browser."""

from __future__ import annotations

import math
import re
from contextlib import ExitStack
from urllib.parse import urlsplit

import pytest
from tests._lss_editor_fixtures import nb_case, open_case, scalar_family_case
from tests.editor._chart_layout import wait_for_chart_to_fit
from tests.editor._editor_waits import wait_for_editor_idle, wait_for_interception
from tests.editor.conftest import _open_session_page

from superglm.editor.lss.panel import NOT_DRAWN_YET

pytest.importorskip("playwright.sync_api")
pytestmark = pytest.mark.browser

_TWO_FRAMES = "() => new Promise(done => requestAnimationFrame(() => requestAnimationFrame(done)))"
# The table as {row label: {column label: cell}}: a row per risk, a column per figure.
_TABLE = """() => {
    const heads = [...document.querySelectorAll('#riskPanel .risk-table thead th')].slice(1).map(h => h.textContent);
    return Object.fromEntries([...document.querySelectorAll('#riskPanel .risk-table tbody tr')].map(row => [
        row.querySelector('th').textContent,
        Object.fromEntries([...row.querySelectorAll('td')].map((cell, i) => [heads[i], cell.textContent])),
    ]));
}"""
_TABLE_DIFFERS = f"before => JSON.stringify(({_TABLE})()) !== before"
_ROWS = "() => [...document.querySelectorAll('#riskPanel .risk-rows tbody tr')].map(r => r.dataset.riskRow)"
_WALK_TITLE = "() => document.querySelector('#riskPanel .risk-drivers-title').textContent"
_WALK = """() => [...document.querySelectorAll('#riskPanel .risk-drivers-table tbody tr')].map(row => ({
    label: [...row.querySelector('th').childNodes].filter(n => n.nodeType === 3).map(n => n.textContent).join(''),
    detail: row.querySelector('.risk-drivers-detail')?.textContent ?? '',
    cells: [...row.querySelectorAll('td:not(.risk-bar-cell)')].map(c => c.firstChild?.textContent ?? ''),
    bars: [...row.querySelectorAll('.risk-bar')].map(bar => parseFloat(bar.style.width)),
}))"""


def _is_risk(request) -> bool:
    return request.method == "POST" and urlsplit(request.url).path == "/risk"


def _open_explain(page) -> list:
    """Open the Explain tab and wait for its table; returns the /risk requests seen from here on."""
    asked: list = []
    page.on("request", lambda request: _is_risk(request) and asked.append(request))
    page.get_by_role("tab", name="Explain", exact=True).click()
    page.locator("#riskPanel .risk-table tbody tr").first.wait_for()
    page.evaluate(_TWO_FRAMES)
    return asked


def _table(page) -> dict[str, dict[str, str]]:
    return page.evaluate(_TABLE)


def _changes_table(page, act) -> None:
    """Run ``act`` and wait for the table to read what it changed."""
    before = page.evaluate(f"() => JSON.stringify(({_TABLE})())")
    act()
    page.wait_for_function(_TABLE_DIFFERS, arg=before)


def _set(page, risk: str, column: str, value: str) -> None:
    """Type a value into a risk's box and wait for the table to read the new risk."""
    box = page.locator(f'#riskPanel [data-risk="{risk}"][data-column="{column}"]')

    def act():
        box.fill(value)
        box.press("Enter")

    _changes_table(page, act)


def _add(page) -> None:
    """Add a risk with the plus button: a training row drawn at random."""
    _changes_table(page, page.locator("#riskPanel .risk-add").click)


def _copy(page, risk: str) -> None:
    """Add a copy of ``risk`` with its row's copy action."""
    _changes_table(
        page, page.locator(f'#riskPanel [data-risk="{risk}"][data-risk-action="copy"]').click
    )


def test_the_explain_tab_is_a_page_of_its_own_and_starts_at_the_reference_risk(
    open_lss_editor_page,
):
    with open_lss_editor_page(selected_term="scale:age") as (page, session):
        asked = _open_explain(page)

        assert page.locator("#explainTab").get_attribute("aria-selected") == "true"
        assert page.locator("#editorView").is_hidden()
        assert page.locator("#reportPanel").is_hidden()
        # The tab draws from its own request, not from a report.
        assert [request.post_data_json for request in asked] == [{"risks": [{}]}]

        reference = session.model.reference_row().iloc[0]
        age = page.locator('#riskPanel [data-risk="a"][data-column="age"]')
        assert float(age.input_value()) == pytest.approx(float(reference["age"]), rel=1e-3)
        region = page.locator('#riskPanel [data-risk="a"][data-column="region"]')
        assert region.input_value() == str(reference["region"])
        table = _table(page)
        assert list(table) == ["A"]
        assert list(table["A"])[:5] == ["Mean", "Median", "90th", "99th", "99.5th"]
        assert page.locator('#riskPanel .risk-plots .risk-line[data-risk="a"]').count() == 2

        page.get_by_role("tab", name="Editor", exact=True).click()
        assert page.locator("#riskView").is_hidden()
        assert page.locator("#editorView").is_visible()


def test_a_superglm_session_has_no_explain_tab(open_editor_page):
    with open_editor_page() as (page, _session):
        assert page.locator("#explainTab").is_hidden()
        assert page.get_by_role("tab", name="Diagnose", exact=True).is_visible()


@pytest.mark.parametrize(
    "viewport",
    [{"width": 1600, "height": 900}, {"width": 1100, "height": 720}],
    ids=["16:9", "notebook-cell"],
)
def test_the_tab_fits_its_window_and_the_plots_take_the_height(open_lss_editor_page, viewport):
    with open_lss_editor_page(selected_term="scale:age", viewport=viewport) as (page, _session):
        _open_explain(page)
        _add(page)
        page.evaluate(_TWO_FRAMES)
        fits = page.evaluate(
            """() => {
                const view = document.querySelector('#riskView');
                const plots = document.querySelector('#riskPanel .risk-plots').getBoundingClientRect();
                const side = document.querySelector('#riskPanel .risk-side').getBoundingClientRect();
                const table = document.querySelector('#riskPanel .risk-table').getBoundingClientRect();
                return {
                    scrolls: view.scrollHeight - view.clientHeight,
                    plots: plots.height, beside: side.left >= plots.right, bottom: plots.bottom,
                    clipped: table.right - side.right,
                };
            }"""
        )
        # Nothing to scroll: the rows, the plots and the table are all on screen.
        assert fits["scrolls"] <= 0
        assert fits["beside"]
        assert fits["clipped"] <= 0.5
        # The plots take what the rows of risks leave of the window.
        assert fits["plots"] >= viewport["height"] - 250
        assert fits["bottom"] <= viewport["height"]


def test_the_plus_button_adds_risks_that_keep_their_letters(open_lss_editor_page):
    with open_lss_editor_page(selected_term="scale:age") as (page, _session):
        asked = _open_explain(page)
        add = page.locator("#riskPanel .risk-add")

        _copy(page, "a")
        assert page.evaluate(_ROWS) == ["a", "b"]
        picked = page.locator('#riskPanel .risk-pick[data-risk="b"]')
        assert picked.get_attribute("aria-pressed") == "true"
        # A copy of risk A reads as risk A does, so every ratio is one.
        table = _table(page)
        assert list(table) == ["A", "B", "B ÷ A"]
        assert set(table["B ÷ A"].values()) == {"×1.00"}
        # The two curves coincide, and their letters stand side by side over the one peak.
        letters = page.evaluate(
            "[...document.querySelectorAll('#riskPanel .risk-plots text.dist-readout')]"
            ".map((node) => [node.textContent, Number(node.getAttribute('x'))])"
        )
        assert sorted(letter for letter, _x in letters) == ["A", "B"]
        assert abs(letters[0][1] - letters[1][1]) >= 12

        _set(page, "b", "age", "22")
        assert asked[-1].post_data_json == {"risks": [{}, {"age": 22}]}
        table = _table(page)
        assert table["B"]["Mean"] != table["A"]["Mean"]
        assert table["B ÷ A"]["CV"] != "×1.00"
        # The selected risk is filled; the other is a line.
        assert page.locator('#riskPanel .risk-plots .risk-fill[data-risk="b"]').count() == 1
        assert page.locator('#riskPanel .risk-plots .risk-fill[data-risk="a"]').count() == 0

        # The plus button adds a training row drawn at random, so its curve is its own.
        with page.expect_response(lambda r: urlsplit(r.url).path == "/risk_row") as drawn:
            _add(page)
        row = drawn.value.json()["values"]
        assert asked[-1].post_data_json == {"risks": [{}, {"age": 22}, row]}
        table = _table(page)
        assert list(table) == ["A", "B", "C"]
        assert table["C"] != table["A"] and table["C"] != table["B"]
        # A row's actions show under the pointer; risk C is the one selected.
        page.locator('#riskPanel tr[data-risk-row="b"]').hover()
        _changes_table(
            page, page.locator('#riskPanel [data-risk="b"][data-risk-action="remove"]').click
        )
        # Risk C keeps its letter and its colour when risk B goes.
        assert page.evaluate(_ROWS) == ["a", "c"]
        assert list(_table(page)) == ["A", "C", "C ÷ A"]
        assert page.locator('#riskPanel .risk-plots .risk-line[data-risk="c"]').count() == 2
        assert page.locator('#riskPanel .risk-plots .risk-line[data-risk="b"]').count() == 0

        for _ in range(6):
            _add(page)
        assert page.evaluate(_ROWS) == ["a", "c", "b", "d", "e", "f", "g", "h"]
        assert add.is_disabled()
        assert "up to 8" in add.get_attribute("title")
        # The copy action is off as well, with the same words, and does nothing when pressed.
        copies = page.locator('#riskPanel [data-risk-action="copy"]')
        assert copies.count() == 8
        for index in range(8):
            assert copies.nth(index).is_disabled()
            assert copies.nth(index).get_attribute("title") == add.get_attribute("title")
        # Removing a risk gives the action back.
        page.locator('#riskPanel tr[data-risk-row="h"]').hover()
        _changes_table(
            page, page.locator('#riskPanel [data-risk="h"][data-risk-action="remove"]').click
        )
        assert copies.first.is_enabled()
        assert "copy of this risk" in copies.first.get_attribute("title")


def test_the_walk_multiplies_from_the_reference_risk_or_risk_a_to_the_selected_risk(
    open_lss_editor_page,
):
    with open_lss_editor_page(selected_term="scale:age") as (page, _session):
        _open_explain(page)
        assert page.evaluate(_WALK_TITLE) == "Reference → A"
        # At the reference risk no feature moves anything: one line says so, not two equal rows.
        assert page.locator("#riskPanel .risk-drivers-table").count() == 0
        assert page.locator("#riskPanel .risk-drivers-empty").inner_text() == (
            "Risk A is the reference risk. Change a value or add a risk to see what moves it."
        )

        with page.expect_response(lambda r: urlsplit(r.url).path == "/risk") as answered:
            _set(page, "a", "age", "22")
        reply = answered.value.json()
        columns = [feature["column"] for feature in reply["features"]]
        moves = reply["risks"][0]["effects"]["edited"][columns.index("age")]
        start, age, end = page.evaluate(_WALK)
        assert age["label"] == "age" and age["detail"].endswith(" → 22")
        for cell, move in zip(age["cells"], moves, strict=True):
            assert cell.startswith("×")
            assert float(cell[1:]) == pytest.approx(math.exp(move), rel=5e-3)
        assert max(age["bars"]) > 0
        assert start["cells"] != end["cells"]

        _copy(page, "a")
        page.wait_for_function(f"() => ({_WALK_TITLE})() === 'A → B'")
        _set(page, "b", "age", "70")
        start, age, end = page.evaluate(_WALK)
        assert (start["label"], age["label"], age["detail"], end["label"]) == (
            "A",
            "age",
            "22 → 70",
            "B",
        )
        # Selecting risk A again walks to it from the reference risk.
        page.locator('#riskPanel .risk-pick[data-risk="a"]').click()
        page.wait_for_function(f"() => ({_WALK_TITLE})() === 'Reference → A'")
        assert page.locator('#riskPanel .risk-plots .risk-fill[data-risk="a"]').count() == 1


def test_a_model_of_counts_draws_each_risks_chance_of_a_number_of_claims(chromium_browser):
    session = open_case(nb_case())
    with ExitStack() as resources:
        first = next(iter(session.terms))
        page = _open_session_page(
            resources,
            chromium_browser,
            session,
            viewport={"width": 1400, "height": 900},
            prepare=None,
            first_term=first,
            selected_term=first,
        )
        _open_explain(page)

        figures = _table(page)["A"]
        assert list(figures)[:5] == ["Mean", "No claim", "1 or more", "2 or more", "3 or more"]
        none = float(figures["No claim"].rstrip("%"))
        some = float(figures["1 or more"].rstrip("%"))
        assert none + some == pytest.approx(100.0, abs=0.11)
        # A count has no line to place: only how its chances are written.
        assert page.locator("#riskPanel .distribution-line-input").is_hidden()
        assert page.locator('#riskPanel [data-style="one_in"]').is_visible()
        # The distribution itself: a bar for no claim, one claim, two and so on.
        assert page.locator('#riskPanel .risk-plots .risk-count-bar[data-risk="a"]').count() >= 2
        assert page.evaluate(_WALK_TITLE) == "Reference → A"

        page.locator('#riskPanel [data-style="one_in"]').click()
        # A chance of one half or more stays a percentage; the smaller one reads 1 in N.
        likely, rare = ("No claim", "1 or more") if none >= some else ("1 or more", "No claim")
        page.wait_for_function(f"() => ({_TABLE})().A[{rare!r}].startsWith('1 in ')")
        assert _table(page)["A"][likely].endswith("%")


def test_a_row_from_the_book_fills_the_risk_with_a_training_rows_values(open_lss_editor_page):
    with open_lss_editor_page(selected_term="scale:age") as (page, session):
        _open_explain(page)
        with page.expect_response(lambda r: urlsplit(r.url).path == "/risk_row") as drawn:
            page.locator('#riskPanel [data-risk="a"][data-risk-action="book"]').click()
        row = drawn.value.json()
        # The box shows the row's value to four significant figures. Rounding that again to
        # fewer would differ from rounding the value itself whenever the shown value ends in 5.
        page.wait_for_function(
            'age => Number(document.querySelector(\'#riskPanel [data-risk="a"][data-column="age"]\')'
            ".value) === Number(Number(age).toPrecision(4))",
            arg=row["values"]["age"],
        )

        train = session.model.X_train.iloc[row["index"]]
        assert row["values"]["age"] == pytest.approx(float(train["age"]))
        assert row["values"]["region"] == str(train["region"])
        region = page.locator('#riskPanel [data-risk="a"][data-column="region"]')
        assert region.input_value() == row["values"]["region"]

        page.locator('#riskPanel [data-risk="a"][data-risk-action="reset"]').click()
        reference = session.model.reference_row().iloc[0]
        page.wait_for_function(
            'level => document.querySelector(\'#riskPanel [data-risk="a"][data-column="region"]\')'
            ".value === level",
            arg=str(reference["region"]),
        )


def test_after_an_edit_the_risk_is_read_beside_the_original_model(open_lss_editor_page):
    with open_lss_editor_page(selected_term="scale:age") as (page, _session):
        page.get_by_role("radiogroup", name="Chart tools").get_by_role(
            "radio", name="Handles", exact=True
        ).click()
        wait_for_chart_to_fit(page)
        handle = page.locator("#chart .control-handle").nth(2)
        handle.wait_for()
        box = handle.bounding_box()
        assert box is not None
        x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
        page.mouse.move(x, y)
        page.mouse.down()
        page.mouse.move(x, y - 60, steps=6)
        page.mouse.up()
        wait_for_editor_idle(page)

        _open_explain(page)
        table = _table(page)
        assert list(table) == ["A, original", "A", "Edited ÷ original"]
        # The edit moved the CV's age curve, so the mean stays and the CV does not.
        assert table["Edited ÷ original"]["Mean"] == "×1.00"
        assert table["Edited ÷ original"]["CV"] != "×1.00"
        assert page.locator("#riskPanel .risk-line-original").count() == 2


def test_a_request_that_fails_keeps_the_risks_controls_and_offers_retry(open_lss_editor_page):
    with open_lss_editor_page(selected_term="scale:age") as (page, _session):
        _open_explain(page)
        wait_for_interception(page)
        page.route("**/risk", lambda route: route.abort())
        box = page.locator('#riskPanel [data-risk="a"][data-column="age"]')
        box.fill("30")
        box.press("Enter")

        note = page.locator("#riskPanel .risk-note")
        note.wait_for(state="visible")
        assert "did not answer" in note.inner_text()
        assert page.locator("#riskPanel .risk-plots").is_hidden()
        assert box.is_visible() and box.input_value() == "30"

        page.unroute("**/risk")
        note.get_by_role("button", name="Retry").click()
        note.wait_for(state="hidden")
        assert page.locator("#riskPanel .risk-plots").is_visible()


def test_the_line_box_holds_a_number_alone_and_a_press_on_the_plots_moves_the_line(
    open_lss_editor_page,
):
    def above(page) -> str:
        return next(label for label in _table(page)["A"] if label.startswith("Above "))

    with open_lss_editor_page(selected_term="scale:age") as (page, _session):
        _open_explain(page)
        line = page.locator("#riskPanel .distribution-line-input")
        before = line.input_value()
        assert re.fullmatch(r"[\d,.]+", before)
        assert above(page) == f"Above {before}"

        plots = page.locator("#riskPanel .risk-plots").bounding_box()
        assert plots is not None
        page.mouse.click(plots["x"] + plots["width"] * 0.45, plots["y"] + 100)
        page.wait_for_function(
            "before => document.querySelector('#riskPanel .distribution-line-input').value !== before",
            arg=before,
        )
        assert re.fullmatch(r"[\d,.]+", line.input_value())
        assert above(page) == f"Above {line.input_value()}"

        line.fill("2500")
        line.press("Enter")
        page.wait_for_function(f"() => 'Above 2,500' in ({_TABLE})().A")
        # The distribution panel under the chart reads the same line, a number alone as well.
        page.get_by_role("tab", name="Editor", exact=True).click()
        page.evaluate("document.querySelector('.plot-column').scrollTo(0, 100000)")
        panel_line = page.locator("#distributionPanel .distribution-line-input")
        page.wait_for_function(
            "() => document.querySelector('#distributionPanel .distribution-line-input')?.value === '2,500'"
        )
        assert panel_line.input_value() == "2,500"


def test_a_value_outside_the_training_rows_is_accepted_and_flagged(open_lss_editor_page):
    with open_lss_editor_page(selected_term="scale:age") as (page, _session):
        _open_explain(page)
        box = page.locator('#riskPanel [data-risk="a"][data-column="age"]')
        low, high = float(box.get_attribute("min")), float(box.get_attribute("max"))
        assert box.get_attribute("data-outside") is None
        assert "in the training rows" in box.get_attribute("title")

        _set(page, "a", "age", str(math.floor(low) - 10))
        assert box.get_attribute("data-outside") is not None
        assert box.get_attribute("aria-invalid") == "false"
        assert box.get_attribute("title") == (
            f"Outside the training rows ({low:.4g} to {high:.4g}): the model extrapolates here"
        )
        # The flag is a visible state, not only a title.
        assert box.evaluate("node => getComputedStyle(node).borderColor") != page.locator(
            '#riskPanel [data-risk="a"][data-column="region"]'
        ).evaluate("node => getComputedStyle(node).borderColor")

        _set(page, "a", "age", str(math.ceil(high) + 5))
        assert box.get_attribute("data-outside") is not None
        _set(page, "a", "age", str(round((low + high) / 2)))
        assert box.get_attribute("data-outside") is None
        assert "in the training rows" in box.get_attribute("title")


def test_a_risk_that_is_loading_offers_no_retry(open_lss_editor_page):
    """Retry belongs to a failed request: the note that says loading has no such button."""
    answer = []

    def hold(route) -> None:
        answer.append(route)

    def prepare(page) -> None:
        page.route("**/risk", hold)

    with open_lss_editor_page(selected_term="scale:age", prepare=prepare) as (page, _session):
        page.get_by_role("tab", name="Explain", exact=True).click()
        note = page.locator("#riskPanel .risk-note")
        note.wait_for(state="visible")
        assert "Loading" in note.inner_text()
        assert note.get_by_role("button", name="Retry").is_hidden()
        assert page.locator("#riskPanel").get_attribute("aria-busy") == "true"

        answer[0].continue_()
        page.locator("#riskPanel .risk-table tbody tr").first.wait_for()
        assert note.is_hidden()


def test_a_family_without_browser_maths_says_so_in_the_panel_and_in_explain(chromium_browser):
    session = open_case(scalar_family_case("GaussianLS"))
    reason = NOT_DRAWN_YET.format(family="GaussianLS")
    shows = "([root, text]) => document.querySelector(root)?.textContent.includes(text)"
    with ExitStack() as resources:
        errors: list = []
        page = _open_session_page(
            resources,
            chromium_browser,
            session,
            viewport={"width": 1400, "height": 900},
            prepare=lambda page: page.on("pageerror", lambda error: errors.append(error)),
            first_term="location:age",
            selected_term="location:age",
        )
        page.locator("#distributionPanel").scroll_into_view_if_needed()
        page.wait_for_function(shows, arg=["#distributionPanel", reason])
        page.get_by_role("tab", name="Explain", exact=True).click()
        page.wait_for_function(shows, arg=["#riskPanel", reason])
        assert errors == []
