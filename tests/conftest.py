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


def skipif_is_active(mark) -> bool:
    """Whether pytest's own evaluation would skip on this ``skipif`` *mark*.

    Mirrors ``_pytest.skipping.evaluate_skip_marks``: the condition may be
    positional *or* passed as ``condition=``, and a ``skipif`` carrying no
    condition at all is an unconditional skip.  Reading only ``mark.args`` --
    which this hook did -- silently escalated one of those three forms and let
    the other two skip with nothing to say so.

    Measured on pytest 9.1.1, three marks carrying the same sentinel reason:
    pytest skipped all three, ``any(mark.args)`` escalated one.  That is the
    single failure mode of this design that leaves no trace, and the trace is
    the whole point -- a real-data job that goes green having skipped is
    exactly what #261 was.

    A *string* condition is pytest's deferred-eval form, evaluated against the
    test module's globals.  Reproducing that here would be a second
    implementation of the thing this function exists to agree with, so it
    refuses rather than guessing; a dataset guard is built from
    ``_datasets.skip_reason`` and has never needed one.
    """
    conditions = (mark.kwargs["condition"],) if "condition" in mark.kwargs else mark.args
    if any(isinstance(condition, str) for condition in conditions):
        pytest.fail(
            "SUPERGLM_REQUIRE_DATA cannot evaluate the string skipif condition "
            f"{conditions!r} on a dataset guard; pass the condition as a value.",
            pytrace=False,
        )
    return not conditions or any(conditions)


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
    ``skipif`` and turns the item into a skip we can no longer see.  Load-bearing
    and measured: dropping it turns the escalation back into a silent skip.

    The sentinel is checked before the condition, so the string-condition
    refusal in :func:`skipif_is_active` can only ever reach a mark this project
    built.
    """
    from . import _datasets

    if not _datasets.require_data():
        return
    for mark in item.iter_markers(name="skipif"):
        reason = str(mark.kwargs.get("reason", ""))
        if reason.startswith(_datasets.SKIP_SENTINEL) and skipif_is_active(mark):
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
