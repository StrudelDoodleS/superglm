"""The pytest side of a test-map build (``-p test_map_plugin``): it names the recorder's contexts.

A test file is the context while its module and its classes are collected and while each of its
tests runs (setup, call and teardown), and the session is again once its collection is done; a
fixture broader than a function is its own context while it is set up, and each test file notes
the ones its tests use, so the tests that read a shared fixture's result reach what built it. Each test file's count of tests and their seconds are kept for the map.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
import test_map_recorder as recorder


def _test_file(path: Path) -> str:
    return path.resolve().relative_to(recorder.root()).as_posix()


def _fixture_context(fixturedef) -> str:
    return f"{recorder.FIXTURE}{fixturedef.baseid}::{fixturedef.argname}"


@pytest.hookimpl(wrapper=True)
def pytest_make_collect_report(collector):
    module = collector.getparent(pytest.Module)
    if module is None:
        return (yield)
    # Back to the enclosing context afterwards: the hooks after collection
    # (``pytest_collection_modifyitems``) decide for every test file, so they are the session's.
    outer = recorder._context
    recorder.switch(_test_file(module.path))
    try:
        return (yield)
    finally:
        recorder.switch(outer)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_protocol(item, nextitem):
    test_file = _test_file(item.path)
    recorder.switch(test_file)
    used = recorder.fixtures.setdefault(test_file, set())
    # The request outlives the protocol (pytest drops the item's reference at teardown). It holds
    # every fixture the test resolved, including those asked for by name while it ran
    # (``request.getfixturevalue``), which ``item.fixturenames`` does not list beforehand.
    request = getattr(item, "_request", None)
    started = time.perf_counter()
    try:
        return (yield)
    finally:
        resolved = list(getattr(request, "_fixture_defs", {}).values())
        for name in item.fixturenames:
            resolved += item._fixtureinfo.name2fixturedefs.get(name, ())[-1:]
        used.update(_fixture_context(d) for d in resolved if d.scope != "function")
        count = recorder.tests.setdefault(test_file, [0, 0.0])
        count[0] += 1
        count[1] += time.perf_counter() - started


@pytest.hookimpl(wrapper=True)
def pytest_fixture_setup(fixturedef, request):
    if fixturedef.scope == "function":
        return (yield)
    outer = recorder._context
    recorder.switch(_fixture_context(fixturedef))
    try:
        return (yield)
    finally:
        recorder.switch(outer)


def pytest_configure(config):
    recorder.own_contexts()


def pytest_unconfigure(config):
    recorder.save()
