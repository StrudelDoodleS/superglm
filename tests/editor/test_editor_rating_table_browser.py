from __future__ import annotations

from urllib.parse import urlsplit

import pytest

pytest.importorskip("playwright.sync_api")
pytestmark = pytest.mark.browser


def _rating_table_for(term: str):
    return lambda response: (
        response.request.method == "POST"
        and urlsplit(response.url).path == "/rating_table"
        and response.request.post_data_json == {"term": term}
    )


def _term_view(page, name: str):
    return page.get_by_role("radiogroup", name="Term view").get_by_role(
        "radio", name=name, exact=True
    )


def test_table_shows_the_terms_rating_table_block_in_place_of_the_chart(
    open_editor_page, choose_feature
):
    with open_editor_page(selected_term="age_band") as (page, _session):
        with page.expect_response(_rating_table_for("age_band")) as response_info:
            _term_view(page, "Table").click()
        block = response_info.value.json()

        frame = page.locator("#ratingTableFrame")
        frame.locator("table.rating-table").wait_for()
        assert page.locator("#chart").is_hidden()
        assert page.locator("#ciToggle").is_hidden()
        assert _term_view(page, "Table").get_attribute("aria-checked") == "true"
        assert block["available"] is True
        assert frame.locator("thead th").all_inner_texts() == block["columns"]
        relativity = block["columns"].index("Relativity")
        weight = block["columns"].index("Weight")
        first = frame.locator("tbody tr").first.locator("td").all_inner_texts()
        assert first[0] == str(block["rows"][0][0])
        assert first[relativity] == f"{block['rows'][0][relativity]:.6f}"
        assert first[weight] == f"{block['rows'][0][weight]:,.2f}"
        assert frame.locator("tbody tr").count() == len(block["rows"])

        with page.expect_response(_rating_table_for("territory")):
            choose_feature(page, "territory")
        page.wait_for_function(
            "() => document.querySelector('#ratingTableFrame thead th')?.textContent"
            " === 'territory'"
        )

        _term_view(page, "Chart").click()
        page.locator("#chart path.edited").first.wait_for()
        assert frame.is_hidden()
        assert page.locator("#chart").is_visible()


def test_table_is_rebuilt_for_each_model_revision_while_it_is_shown(open_editor_page):
    # Mutation check: keying the table's request on the view and the term
    # alone sends no request after an edit, an Undo or a Redo, and the table
    # keeps showing the block of a model that is no longer in force.
    with open_editor_page(selected_term="territory") as (page, session):
        frame = page.locator("#ratingTableFrame")

        def shown_relativity(block, level: str) -> str:
            # The cell the table shows for `level`, once it shows the reply's.
            row = [str(cells[0]) for cells in block["rows"]].index(level)
            relativity = block["columns"].index("Relativity")
            expected = f"{block['rows'][row][relativity]:.6f}"
            page.wait_for_function(
                "([row, column, text]) => document.querySelector("
                "`#ratingTableFrame tbody tr:nth-child(${row}) td:nth-child(${column})`"
                ")?.textContent === text",
                arg=[row + 1, relativity + 1, expected],
            )
            return expected

        with page.expect_response(_rating_table_for("territory")) as response_info:
            _term_view(page, "Table").click()
        frame.locator("table.rating-table").wait_for()
        fitted = shown_relativity(response_info.value.json(), "T03")

        # An edit made in the notebook, pulled in with Refresh.
        session.select_levels("territory", ["T03"])
        session.shift("territory", 0.2)
        page.wait_for_function("() => !document.querySelector('#refreshAction').disabled")
        with page.expect_response(_rating_table_for("territory")) as response_info:
            page.locator("#refreshAction").click()
        edited = shown_relativity(response_info.value.json(), "T03")
        assert edited != fitted

        page.wait_for_function("() => !document.querySelector('#undoAction').disabled")
        with page.expect_response(_rating_table_for("territory")) as response_info:
            page.locator("#undoAction").click()
        assert shown_relativity(response_info.value.json(), "T03") == fitted

        page.wait_for_function("() => !document.querySelector('#redoAction').disabled")
        with page.expect_response(_rating_table_for("territory")) as response_info:
            page.locator("#redoAction").click()
        assert shown_relativity(response_info.value.json(), "T03") == edited
        assert page.locator("#chart").is_hidden()
