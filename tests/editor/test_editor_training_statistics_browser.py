import re
from contextlib import closing

import pytest
from tests._no_mean_fixture import CELLS, CELLS_Y, TRAIN, TRAIN_Y, editor_fit

from superglm.editor import edit

expect = pytest.importorskip("playwright.sync_api").expect
pytestmark = pytest.mark.browser


def test_training_summary_follows_redo_while_validation_prices_its_own_rows(chromium_browser):
    session = edit(editor_fit(), train_data=(TRAIN, TRAIN_Y), validation_data=(CELLS, CELLS_Y))
    session.select_indices("a", [1])
    session.shift("a", 0.01).shift("a", -0.31).shift("a", 0.90)
    session.undo().undo()
    with closing(session.widget()) as widget:
        page = chromium_browser.new_page(viewport={"width": 1440, "height": 900})
        try:
            page.goto(f"{widget.app_url}&test=1", wait_until="domcontentloaded")
            summary = page.locator("#summaryFrame")
            deviance = page.locator("#summaryTiles .summary-tile").filter(has_text="Deviance")
            notice = page.locator("#appAlertMessage")
            metrics = page.locator("#metricGrid")
            chart = page.locator("#chart path.edited").first
            redo = page.get_by_role("button", name="Redo edit")
            expect(deviance).to_contain_text("8.65")
            expect(deviance.locator("strong")).to_have_attribute("title", re.compile(r"^8\.646047"))
            expect(metrics).to_contain_text("1 of 4 rows have no quote")
            priced_curve = chart.get_attribute("d")
            assert priced_curve

            redo.click()
            expect(summary).to_contain_text("10 of 30 rows have no quote")
            expect(notice).to_contain_text("Adjust the curve or undo the edit")
            expect(page.locator("#appAlertRetry")).to_be_hidden()
            expect(chart).not_to_have_attribute("d", priced_curve)
            assert len(session.history) == 2

            redo.click()
            expect(deviance).to_contain_text("32.2")
            expect(deviance.locator("strong")).to_have_attribute(
                "title", re.compile(r"^32\.154953")
            )
            expect(notice).to_be_hidden()
            expect(metrics).to_contain_text("Validation data")
            expect(metrics).to_contain_text("The original model has no quote")
            validation_deviance = metrics.locator(".metric-item").filter(has_text="Deviance").first
            expect(validation_deviance.locator(".metric-item-value")).to_have_text("3.36")
        finally:
            page.close()
