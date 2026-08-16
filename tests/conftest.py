"""Shared pytest fixtures and hooks."""

import pytest


def pytest_addoption(parser):
    parser.addoption("--run-browser", action="store_true", help="run Playwright editor tests")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-browser"):
        return
    skip = pytest.mark.skip(reason="pass --run-browser to run Playwright editor tests")
    for item in items:
        if "browser" in item.keywords:
            item.add_marker(skip)


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    """Under ``SUPERGLM_REQUIRE_DATA``, turn a dataset skip into a FAILURE.

    Scoped to the items actually carrying a dataset skip, which is the whole
    point of doing it here rather than raising from :mod:`tests._datasets`.
    Raising at import time collapses the importing module's entire collection,
    so one unreadable parquet deleted ~33 synthetic tests with no stake in it --
    a worse diagnostic than the silent skip it replaced, and one that would fire
    on a transient fetch failure once CI carries the dataset.

    ``tryfirst`` so this runs before pytest's own skipping plugin evaluates the
    ``skipif`` and turns the item into a skip we can no longer see.
    """
    from . import _datasets

    if not _datasets.require_data():
        return
    for mark in item.iter_markers(name="skipif"):
        reason = str(mark.kwargs.get("reason", ""))
        if any(mark.args) and reason.startswith(_datasets.SKIP_SENTINEL):
            pytest.fail(f"SUPERGLM_REQUIRE_DATA is set, but {reason}", pytrace=False)


@pytest.fixture(autouse=True)
def _close_matplotlib_figures():
    """Close all matplotlib figures after each test to prevent resource leaks."""
    yield
    try:
        import matplotlib.pyplot as plt

        plt.close("all")
    except ImportError:
        pass
