"""The pytest side of a test-map build (``-p test_map_plugin``): it names the recorder's contexts.

A test file is the context while its module is collected and while each of its tests runs (setup,
call and teardown); a fixture broader than a function is its own context while it is set up, and
each test file notes the ones its tests use, so the tests that read a shared fixture's result reach
what built it. Each test file's count of tests and their seconds are kept for the map.
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
    if isinstance(collector, pytest.Module):
        recorder.switch(_test_file(collector.path))
    return (yield)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_protocol(item, nextitem):
    test_file = _test_file(item.path)
    recorder.switch(test_file)
    used = recorder.fixtures.setdefault(test_file, set())
    for name in item.fixturenames:
        definitions = item._fixtureinfo.name2fixturedefs.get(name)
        if definitions and definitions[-1].scope != "function":
            used.add(_fixture_context(definitions[-1]))
    started = time.perf_counter()
    try:
        return (yield)
    finally:
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
