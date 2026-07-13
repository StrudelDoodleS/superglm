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


@pytest.fixture(autouse=True)
def _close_matplotlib_figures():
    """Close all matplotlib figures after each test to prevent resource leaks."""
    yield
    try:
        import matplotlib.pyplot as plt

        plt.close("all")
    except ImportError:
        pass
