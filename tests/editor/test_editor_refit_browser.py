from __future__ import annotations

import json

import numpy as np
import pytest

from superglm.editor.payloads import session_payload, timeline_payload

pytest.importorskip("playwright.sync_api")
pytestmark = pytest.mark.browser


def _path(url: str) -> str:
    from urllib.parse import urlsplit

    return urlsplit(url).path


def _reload_editor(page, term: str) -> None:
    page.reload(wait_until="domcontentloaded")
    page.locator("#chart path.edited").first.wait_for()
    page.wait_for_function(
        "term => document.querySelector('#status')?.dataset.term === term",
        arg=term,
    )
    _wait_for_editor_idle(page)


def _wait_for_editor_idle(page) -> None:
    page.wait_for_function(
        """() => {
            const metrics = document.querySelector('#metricGrid');
            const summary = document.querySelector('#summaryFrame');
            return metrics?.getAttribute('aria-busy') === 'false'
                && summary?.getAttribute('aria-busy') === 'false'
                && document.querySelector('#appBusyOverlay')?.hidden;
        }"""
    )


def _complete_metric_payload(request, *, edited_deviance: float) -> str:
    request_payload = request.post_data_json
    keys = (
        "deviance",
        "aic",
        "bic",
        "log_likelihood",
        "explained_deviance",
        "pearson_chi2",
        "effective_df",
    )
    original = {key: float(index + 1) for index, key in enumerate(keys)}
    edited = dict(original)
    edited["deviance"] = edited_deviance
    payload = {
        "status": "complete",
        "available": True,
        "model_revision": request_payload["model_revision"],
        "request_sequence": request_payload["request_sequence"],
        "metric": "deviance",
        "label": "Deviance",
        "dataset": "training",
        "dataset_label": "Training",
        "n_obs": 500,
        "original": original["deviance"],
        "edited": edited_deviance,
        "delta": edited_deviance - original["deviance"],
        "metrics": {"original": original, "edited": edited},
    }
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    assert len(body.encode()) < 1024
    return body


def test_hidden_summary_refreshes_when_reopened_after_an_edit(open_editor_page):
    with open_editor_page() as (page, session):
        _wait_for_editor_idle(page)
        inspector = page.get_by_role("complementary", name="Model inspector")
        inspector.get_by_role("tab", name="Help").click()
        summary_requests: list[object] = []

        def record_summary(request) -> None:
            if request.method == "POST" and _path(request.url) == "/summary":
                summary_requests.append(request)

        page.on("request", record_summary)
        page.locator('button[data-op="select_all"]').click()
        page.locator("#selectionMenu").wait_for(state="visible")
        with page.expect_response(
            lambda response: response.request.method == "POST" and _path(response.url) == "/op"
        ):
            page.get_by_role("button", name="Increase selection", exact=True).click()

        page.wait_for_function(
            "() => document.querySelector('#summaryFrame')?.dataset.freshness === 'stale'"
        )
        assert summary_requests == []

        with page.expect_request(
            lambda request: request.method == "POST" and _path(request.url) == "/summary"
        ) as summary_info:
            inspector.get_by_role("tab", name="Summary").click()

        assert summary_info.value.post_data_json["model_revision"] == session.model_revision
        page.wait_for_function(
            "() => document.querySelector('#summaryFrame')?.dataset.freshness === 'current'"
        )


def test_editor_evidence_catches_up_after_an_edit_in_report_view(open_editor_page):
    with open_editor_page() as (page, session):
        _wait_for_editor_idle(page)
        page.locator('button[data-op="select_all"]').click()
        page.locator("#selectionMenu").wait_for(state="visible")
        with page.expect_response(
            lambda response: response.request.method == "POST" and _path(response.url) == "/op"
        ):
            page.get_by_role("button", name="Increase selection", exact=True).click()
        _wait_for_editor_idle(page)
        # The edit's debounced metrics and summary refresh fires 150ms later
        # whichever view is showing; let it land before leaving the editor, so
        # the only stale evidence below is the undo's.
        page.wait_for_function(
            "() => document.querySelector('#metricGrid')?.dataset.freshness === 'current'"
            " && document.querySelector('#summaryFrame')?.dataset.freshness === 'current'"
        )

        with page.expect_response(
            lambda response: response.request.method == "POST" and _path(response.url) == "/report"
        ):
            page.get_by_role("tab", name="Validation", exact=True).click()

        with page.expect_response(
            lambda response: response.request.method == "POST" and _path(response.url) == "/op"
        ):
            page.get_by_role("button", name="Undo edit").click()
        page.wait_for_function(
            "() => document.querySelector('#metricGrid')?.dataset.freshness === 'stale'"
        )

        with (
            page.expect_request(
                lambda request: request.method == "POST" and _path(request.url) == "/metrics"
            ) as metrics_info,
            page.expect_request(
                lambda request: request.method == "POST" and _path(request.url) == "/summary"
            ) as summary_info,
        ):
            page.get_by_role("tab", name="Editor", exact=True).click()

        assert metrics_info.value.post_data_json["model_revision"] == session.model_revision
        assert summary_info.value.post_data_json["model_revision"] == session.model_revision
        _wait_for_editor_idle(page)


def test_structural_refit_commits_atomically_before_held_metrics(open_editor_page):
    with open_editor_page(selected_term="territory") as (page, session):
        _wait_for_editor_idle(page)
        points = page.locator("#chart .point[data-index]")
        with page.expect_response(
            lambda response: response.request.method == "POST" and _path(response.url) == "/select"
        ):
            points.nth(1).click()
        with page.expect_response(
            lambda response: response.request.method == "POST" and _path(response.url) == "/select"
        ):
            points.nth(2).click(modifiers=["Control"])
        page.wait_for_function(
            "() => document.querySelectorAll('#chart .point.selected[data-index]').length === 2"
        )

        # The collapse waits: staging fits nothing, so nothing blocks the page.
        with page.expect_response(
            lambda response: response.request.method == "POST" and _path(response.url) == "/stage"
        ) as stage_info:
            page.get_by_role("button", name="Collapse", exact=True).click()
        assert stage_info.value.status == 200
        assert page.locator("#appBusyOverlay").is_hidden()
        page.wait_for_function("() => !document.querySelector('#refitPendingAction').disabled")

        requests: list[object] = []
        held_metrics: list[object] = []

        def record_request(request) -> None:
            requests.append(request)

        def hold_metrics(route) -> None:
            held_metrics.append(route)
            page.evaluate("count => { window.__heldMetricRouteCount = count; }", len(held_metrics))

        page.evaluate("window.__heldMetricRouteCount = 0")
        page.on("request", record_request)
        page.route("**/metrics", hold_metrics)
        try:
            with page.expect_request(
                lambda request: request.method == "POST" and _path(request.url) == "/metrics"
            ):
                with page.expect_response(
                    lambda response: (
                        response.request.method == "POST"
                        and _path(response.url) == "/refit_pending"
                    )
                ) as refit_info:
                    page.locator("#refitPendingAction").click()
                    page.locator("#appBusyOverlay").wait_for(state="visible")
                    assert page.locator("#editorView").get_attribute("inert") == ""
                    assert page.evaluate("document.activeElement?.id") == "appBusyAnnouncement"

            assert refit_info.value.status == 200
            page.wait_for_function(
                """revision => {
                    const overlay = document.querySelector('#appBusyOverlay');
                    const chart = document.querySelector('#chart');
                    const summary = document.querySelector('#summaryFrame');
                    const metrics = document.querySelector('#metricGrid');
                    return overlay?.hidden
                        && window.__heldMetricRouteCount === 1
                        && chart?.dataset.modelRevision === revision
                        && summary?.dataset.modelRevision === revision
                        && metrics?.dataset.freshness === 'updating';
                }""",
                arg=str(session.model_revision),
            )

            chart_revision = page.locator("#chart").get_attribute("data-model-revision")
            summary_revision = page.locator("#summaryFrame").get_attribute("data-model-revision")
            assert chart_revision == summary_revision == str(session.model_revision)
            assert page.locator("#appBusyOverlay").is_hidden()
            assert page.locator("#metricGrid").get_attribute("data-freshness") == "updating"
            assert len(held_metrics) == 1
            request_paths = [_path(request.url) for request in requests]
            assert request_paths.count("/refit_pending") == 1
            assert request_paths.count("/state") == 0
        finally:
            for route in held_metrics:
                route.abort()
            page.unroute("**/metrics", hold_metrics)


def test_older_metrics_response_cannot_replace_newer_revision(open_editor_page):
    with open_editor_page(selected_term="territory") as (page, session):
        _wait_for_editor_idle(page)
        with page.expect_response(
            lambda response: response.request.method == "POST" and _path(response.url) == "/select"
        ):
            page.locator('#chart .point[data-index="3"]').click()
        page.locator("#selectionMenu").wait_for(state="visible")

        held_metrics: list[object] = []
        pending_metrics: list[object] = []

        def hold_metrics(route) -> None:
            held_metrics.append(route)
            pending_metrics.append(route)
            page.evaluate("count => { window.__heldMetricRouteCount = count; }", len(held_metrics))

        page.evaluate(
            """() => {
                window.__heldMetricRouteCount = 0;
                window.__settledMetricSequences = [];
                const responseJSON = Response.prototype.json;
                window.__restoreMetricResponseJSON = () => {
                    Response.prototype.json = responseJSON;
                    delete window.__restoreMetricResponseJSON;
                };
                Response.prototype.json = async function() {
                    const payload = await responseJSON.call(this);
                    if (new URL(this.url).pathname.endsWith('/metrics')) {
                        const sequence = Number(payload?.request_sequence);
                        window.setTimeout(() => {
                            window.__settledMetricSequences.push(sequence);
                        }, 0);
                    }
                    return payload;
                };
            }"""
        )
        page.route("**/metrics", hold_metrics)
        try:
            increase = page.get_by_role("button", name="Increase selection", exact=True)
            metric_requests: list[object] = []
            for expected_revision in (1, 2):
                with page.expect_request(
                    lambda request: request.method == "POST" and _path(request.url) == "/metrics"
                ) as metric_info:
                    with page.expect_response(
                        lambda response: (
                            response.request.method == "POST" and _path(response.url) == "/op"
                        )
                    ) as edit_info:
                        increase.click()

                assert edit_info.value.status == 200
                metric_request = metric_info.value
                metric_requests.append(metric_request)
                assert metric_request.post_data_json["model_revision"] == expected_revision
                assert session.model_revision == expected_revision

            page.wait_for_function("() => window.__heldMetricRouteCount === 2")
            assert len(held_metrics) == 2
            first_request, second_request = metric_requests
            first_route, second_route = held_metrics
            first_payload = first_request.post_data_json
            second_payload = second_request.post_data_json
            assert first_payload["model_revision"] < second_payload["model_revision"]
            assert first_payload["request_sequence"] < second_payload["request_sequence"]

            newer_value = 222.2
            with page.expect_response(
                lambda response: (
                    _path(response.url) == "/metrics"
                    and response.request.post_data_json["request_sequence"]
                    == second_payload["request_sequence"]
                )
            ) as newer_response:
                second_route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=_complete_metric_payload(
                        second_request,
                        edited_deviance=newer_value,
                    ),
                )
                pending_metrics.remove(second_route)
            newer_response.value.finished()
            page.wait_for_function(
                "sequence => window.__settledMetricSequences.includes(sequence)",
                arg=second_payload["request_sequence"],
            )

            expected_current = {
                "revision": str(second_payload["model_revision"]),
                "value": str(newer_value),
            }
            page.wait_for_function(
                """expected => {
                    const metrics = document.querySelector('#metricGrid');
                    const value = metrics?.querySelector('.metric-item-value');
                    const chart = document.querySelector('#chart');
                    const summary = document.querySelector('#summaryFrame');
                    return metrics?.dataset.freshness === 'current'
                        && value?.textContent === expected.value
                        && chart?.dataset.modelRevision === expected.revision
                        && summary?.dataset.modelRevision === expected.revision;
                }""",
                arg=expected_current,
            )

            older_value = 111.1
            with page.expect_response(
                lambda response: (
                    _path(response.url) == "/metrics"
                    and response.request.post_data_json["request_sequence"]
                    == first_payload["request_sequence"]
                )
            ) as older_response:
                first_route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=_complete_metric_payload(
                        first_request,
                        edited_deviance=older_value,
                    ),
                )
                pending_metrics.remove(first_route)
            older_response.value.finished()
            page.wait_for_function(
                "sequence => window.__settledMetricSequences.includes(sequence)",
                arg=first_payload["request_sequence"],
            )

            page.wait_for_function(
                """expected => {
                    const metrics = document.querySelector('#metricGrid');
                    const value = metrics?.querySelector('.metric-item-value');
                    const chart = document.querySelector('#chart');
                    const summary = document.querySelector('#summaryFrame');
                    return metrics?.dataset.freshness === 'current'
                        && value?.textContent === expected.value
                        && chart?.dataset.modelRevision === expected.revision
                        && summary?.dataset.modelRevision === expected.revision;
                }""",
                arg=expected_current,
            )
            assert page.locator("#metricFreshness").get_attribute("data-freshness") == "current"
            assert page.locator("#metricGrid .metric-item-value").first.text_content() == str(
                newer_value
            )
        finally:
            for route in pending_metrics:
                route.abort()
            page.unroute("**/metrics", hold_metrics)
            page.evaluate("window.__restoreMetricResponseJSON?.()")


def test_a_structural_refit_over_live_edits_runs_at_once_and_undo_brings_them_back(
    open_editor_page,
):
    term = "long_category"
    labels = [
        "MyReallyLongCategoryNameThatWouldNeverFit",
        "Family👨‍👩‍👧‍👦DriverCaféCategory",
    ]
    with open_editor_page(selected_term=term) as (page, session):
        session.select_levels(term, labels)
        session.shift(term, 0.05)
        session.shift(term, -0.01)
        _reload_editor(page, term)
        edited = session.terms[term].edited_log_effect.copy()
        records = list(session.history)
        model = session.model
        stages: list[object] = []
        page.on(
            "request",
            lambda request: _path(request.url) == "/stage" and stages.append(request),
        )
        collapse = page.get_by_role("button", name="Collapse", exact=True)

        # A change still waits for a running action to finish.
        page.evaluate("window.__superglmTest.setAppBusy(true, 'Testing busy guard', 'Waiting')")
        collapse.evaluate("node => node.click()")
        page.evaluate("() => new Promise(resolve => requestAnimationFrame(() => resolve()))")
        assert stages == []
        page.evaluate("window.__superglmTest.setAppBusy(false)")

        with page.expect_response(
            lambda response: response.request.method == "POST" and _path(response.url) == "/stage"
        ) as response_info:
            collapse.click()
        assert response_info.value.status == 200
        # Staged: the model and the edits stand until Refit.
        assert session.model is model
        assert [id(record) for record in session.history] == [id(record) for record in records]
        # R refits once the page has committed the waiting change and enabled Refit.
        page.wait_for_function("() => !document.querySelector('#refitPendingAction').disabled")

        with page.expect_response(
            lambda response: (
                response.request.method == "POST" and _path(response.url) == "/refit_pending"
            )
        ) as refit_info:
            page.keyboard.press("r")
        assert refit_info.value.status == 200
        page.locator("#appBusyOverlay").wait_for(state="hidden")
        # Nothing is lost, so nothing asked first. The collapse restructured the
        # edited term, so its edits are set aside (D2).
        assert page.locator("dialog[open]").count() == 0 and len(stages) == 1
        assert session.history == [] and session.edited_terms() == []

        with page.expect_response(
            lambda response: response.request.method == "POST" and _path(response.url) == "/op"
        ):
            page.locator("#undoAction").click()
        _wait_for_editor_idle(page)
        np.testing.assert_array_equal(session.terms[term].edited_log_effect, edited)
        assert [id(record) for record in session.history] == [id(record) for record in records]
        # Undo of the Refit brings the collapse back as waiting.
        assert len(session.pending) == 1


def _stage_response(response) -> bool:
    return response.request.method == "POST" and _path(response.url) == "/stage"


def _refit_response(response) -> bool:
    return response.request.method == "POST" and _path(response.url) == "/refit_pending"


def test_two_staged_collapses_wait_and_one_refit_applies_both(open_editor_page):
    with open_editor_page(selected_term="territory") as (page, session):
        _wait_for_editor_idle(page)
        original = session.model
        revision = session.model_revision
        refit = page.locator("#refitPendingAction")
        assert refit.is_disabled()
        assert refit.get_attribute("aria-label") == "Refit, nothing waiting"
        paths: list[str] = []
        page.on(
            "request",
            lambda request: request.method == "POST" and paths.append(_path(request.url)),
        )

        for levels in (["T02", "T03"], ["T06", "T07"]):
            session.select_levels("territory", levels)
            _reload_editor(page, "territory")
            page.locator("#selectionMenu").wait_for(state="visible")
            assert page.locator("#selectionRefitLabel").text_content() == "Structure"
            with page.expect_response(_stage_response) as staged:
                page.get_by_role("button", name="Collapse", exact=True).click()
            assert staged.value.status == 200
            # Staging fits nothing: the page never blocks and the model stands.
            assert page.locator("#appBusyOverlay").is_hidden()
            assert session.model is original and session.model_revision == revision

        page.wait_for_function(
            "() => document.querySelector('#refitPendingCount')?.textContent === '2'"
        )
        assert refit.get_attribute("aria-label") == "Refit, 2 changes waiting"
        assert len(session.pending) == 2

        with page.expect_response(_refit_response) as refit_info:
            page.keyboard.press("r")
        assert refit_info.value.status == 200
        _wait_for_editor_idle(page)
        assert paths.count("/stage") == 2 and paths.count("/refit_pending") == 1
        assert not {"/collapse_levels", "/ungroup_levels"} & set(paths)
        assert session.pending == []
        groups = session_payload(session)["territory"]["level_groups"]
        assert sorted(group["levels"] for group in groups) == [["T02", "T03"], ["T06", "T07"]]
        page.wait_for_function("() => document.querySelector('#refitPendingAction').disabled")


# The routes that refit: Refit itself, and each operation's own route, which
# stages the change and refits at once.
_REFIT_ROUTES = frozenset(
    {"/refit_pending", "/collapse_levels", "/ungroup_levels", "/set_reference", "/shape_range"}
)


def test_refit_after_every_change_is_one_step(open_editor_page):
    with open_editor_page(selected_term="territory") as (page, session):
        _wait_for_editor_idle(page)
        inspector = page.get_by_role("complementary", name="Model inspector")
        inspector.get_by_role("tab", name="Settings").click()
        switch = inspector.get_by_role("switch", name="Refit after every structural change")
        switch.click()
        assert switch.get_attribute("aria-checked") == "true"

        session.select_levels("territory", ["T04", "T05"])
        _reload_editor(page, "territory")
        page.locator("#selectionMenu").wait_for(state="visible")
        # With the setting on, the menu's structural row says its icons refit.
        assert page.locator("#selectionRefitLabel").text_content() == "Refit"
        before = session_payload(session)["territory"]
        entries = len(timeline_payload(session))
        paths: list[str] = []
        page.on(
            "response",
            lambda response: (
                response.request.method == "POST" and paths.append(_path(response.url))
            ),
        )
        with page.expect_response(
            lambda response: (
                response.request.method == "POST" and _path(response.url) == "/collapse_levels"
            )
        ) as refit_info:
            page.get_by_role("button", name="Collapse", exact=True).click()
        assert refit_info.value.status == 200
        # The change goes to the operation's own route, with Settings' keep-reference.
        assert refit_info.value.request.post_data_json["keep_reference"] is True
        _wait_for_editor_idle(page)

        # One refit and nothing staged first.
        assert [path for path in paths if path in _REFIT_ROUTES | {"/stage"}] == [
            "/collapse_levels"
        ]
        assert session.pending == []
        groups = session_payload(session)["territory"]["level_groups"]
        assert [group["levels"] for group in groups] == [["T04", "T05"]]
        # One new entry on the timeline, and nothing waiting.
        timeline = timeline_payload(session)
        assert len(timeline) == entries + 1
        assert [entry for entry in timeline if entry.get("status") == "waiting"] == []
        page.wait_for_function("() => document.querySelector('#refitPendingAction').disabled")

        # One Undo takes the whole change back.
        with page.expect_response(
            lambda response: response.request.method == "POST" and _path(response.url) == "/op"
        ):
            page.locator("#undoAction").click()
        _wait_for_editor_idle(page)
        after = session_payload(session)["territory"]
        assert after["levels"] == before["levels"]
        assert after["level_groups"] == before["level_groups"] == []
        assert session.pending == []
