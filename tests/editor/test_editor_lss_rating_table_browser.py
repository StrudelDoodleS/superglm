"""The Chart / Table switch in a SuperLSS session, driven from the browser."""

from __future__ import annotations

from urllib.parse import urlsplit

import pytest
from tests._lss_editor_fixtures import lognormal_case
from tests.test_editor_lss_rating_table import refusing_case

pytest.importorskip("playwright.sync_api")
pytestmark = pytest.mark.browser


@pytest.fixture(scope="session")
def lss_browser_case(request):
    """The fitted case this test is parametrised with (see the parametrisation below)."""
    return request.param()


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


@pytest.mark.parametrize("lss_browser_case", [lognormal_case], indirect=True)
def test_table_shows_a_mean_term_and_a_cv_term(open_lss_editor_page, choose_feature):
    with open_lss_editor_page(selected_term="mean:region") as (page, _session):
        frame = page.locator("#ratingTableFrame")
        with page.expect_response(_rating_table_for("mean:region")) as response_info:
            _term_view(page, "Table").click()
        block = response_info.value.json()
        frame.locator("table.rating-table").wait_for()

        assert block["available"] is True and page.locator("#chart").is_hidden()
        assert frame.locator("thead th").all_inner_texts() == block["columns"]
        assert frame.locator("tbody tr").count() == len(block["rows"])
        assert frame.locator(".rating-table-note").inner_text().startswith("Mean = base x ")

        with page.expect_response(_rating_table_for("scale:region")) as response_info:
            choose_feature(page, "scale:region")
        block = response_info.value.json()
        page.wait_for_function(
            "() => document.querySelector('#ratingTableFrame')?.dataset.term === 'scale:region'"
            " && document.querySelector('#ratingTableFrame table')"
        )
        assert block["available"] is True
        note = frame.locator(".rating-table-note").inner_text()
        assert note.startswith("Sigma = floor + base x product of the relativities")
        assert "the floor is 0.01" in note
        assert frame.locator("tbody tr").count() == len(block["rows"])


@pytest.mark.parametrize("lss_browser_case", [refusing_case], indirect=True)
def test_table_gives_the_librarys_sentence_for_a_term_it_refuses(
    open_lss_editor_page, choose_feature
):
    with open_lss_editor_page(selected_term="scale:region") as (page, _session):
        frame = page.locator("#ratingTableFrame")
        with page.expect_response(_rating_table_for("scale:region")) as response_info:
            _term_view(page, "Table").click()
        block = response_info.value.json()
        message = frame.locator(".rating-table-message")
        message.wait_for()

        assert block["available"] is False and "Piecewise" in block["reason"]
        assert message.inner_text() == block["reason"]
        assert frame.locator("table").count() == 0

        # No other term's table is lost.
        with page.expect_response(_rating_table_for("mean:region")):
            choose_feature(page, "mean:region")
        frame.locator("table.rating-table").wait_for()
        assert frame.locator(".rating-table-message").count() == 0
