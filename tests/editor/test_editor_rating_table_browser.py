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
