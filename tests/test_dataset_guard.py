"""The real-data guard, and the switch that stops it passing silently.

These suites skip wherever the gitignored parquet is absent -- CI included --
and a skip reads identically to a pass in the summary.  That is how the guide
anchor drifted out of tolerance on 2026-08-07 and went unreported.  The
enforcement switch is what makes "did this actually run?" answerable, so the
switch itself needs coverage: without it, the guard can silently stop applying
to a suite and nothing says so.

Nothing here touches a dataset.  ``usable`` is stubbed, so these run identically
with the parquet present or absent.
"""

from __future__ import annotations

import pytest

from . import _datasets
from .conftest import pytest_runtest_setup


class _Marker:
    """Enough of a pytest mark for the hook to read."""

    def __init__(self, condition, reason):
        self.args = (condition,)
        self.kwargs = {"reason": reason}


class _Item:
    """Enough of a pytest item for the hook to read."""

    def __init__(self, *markers):
        self._markers = markers

    def iter_markers(self, name):
        assert name == "skipif"
        return iter(self._markers)


def _dataset_item(condition=True):
    return _Item(_Marker(condition, _datasets.skip_reason("nope.parquet")))


@pytest.fixture
def _absent(monkeypatch):
    """No dataset is loadable, whatever the machine actually has."""
    monkeypatch.setattr(_datasets, "usable", lambda name: None)


# ── the switch is parsed as an allow-list, not as "anything truthy" ──────────


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " on "])
def test_the_switch_is_on_for_affirmative_spellings(monkeypatch, value):
    monkeypatch.setenv("SUPERGLM_REQUIRE_DATA", value)
    assert _datasets.require_data()


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "NO"])
def test_the_switch_is_off_for_negative_spellings(monkeypatch, value):
    """``no`` and ``off`` mean OFF.

    An earlier revision tested ``not in ("", "0", "false")``, which turned every
    one of these except the first three INTO the enabled state -- the opposite
    of what the person typing them meant.
    """
    monkeypatch.setenv("SUPERGLM_REQUIRE_DATA", value)
    assert not _datasets.require_data()


def test_the_switch_is_off_when_unset(monkeypatch):
    monkeypatch.delenv("SUPERGLM_REQUIRE_DATA", raising=False)
    assert not _datasets.require_data()


# ── skip_reason never raises, so a module-scope call cannot kill collection ──


def test_skip_reason_is_pure_and_tagged(_absent, monkeypatch):
    """It must NOT raise, at any switch setting.

    It is called at module scope in three suites.  An earlier revision raised
    here, and one unreadable parquet then errored the whole module at
    COLLECTION -- 33 synthetic tests with no stake in the dataset went with the
    2 that had one.
    """
    for value in ("1", "0", ""):
        monkeypatch.setenv("SUPERGLM_REQUIRE_DATA", value)
        reason = _datasets.skip_reason("nope.parquet")
        assert reason is not None
        assert reason.startswith(_datasets.SKIP_SENTINEL)
        assert "parquet engine" in reason


def test_skip_reason_is_none_when_the_dataset_is_usable(monkeypatch):
    monkeypatch.setattr(_datasets, "usable", lambda name: "/somewhere/x.parquet")
    assert _datasets.skip_reason("x.parquet") is None


# ── the switch escalates the marked ITEMS, and only those ───────────────────


def test_a_dataset_skip_becomes_a_failure_under_the_switch(_absent, monkeypatch):
    monkeypatch.setenv("SUPERGLM_REQUIRE_DATA", "1")
    # ``pytest.fail`` raises ``Failed``, which derives from ``BaseException`` and
    # is therefore NOT caught by ``pytest.raises(Exception)``.
    with pytest.raises(pytest.fail.Exception) as excinfo:
        pytest_runtest_setup(_dataset_item())
    assert "SUPERGLM_REQUIRE_DATA is set" in str(excinfo.value)


def test_a_dataset_skip_stays_a_skip_without_the_switch(_absent, monkeypatch):
    monkeypatch.delenv("SUPERGLM_REQUIRE_DATA", raising=False)
    pytest_runtest_setup(_dataset_item())  # must not raise


def test_the_switch_leaves_unrelated_skips_alone(_absent, monkeypatch):
    """Scope is the point.

    The hook exists instead of a raise in ``_datasets`` precisely so that one
    unreadable dataset cannot take unrelated tests with it.  A skipif that is
    not a dataset skip must be untouched.
    """
    monkeypatch.setenv("SUPERGLM_REQUIRE_DATA", "1")
    unrelated = _Item(_Marker(True, "pass --run-browser to run Playwright editor tests"))
    pytest_runtest_setup(unrelated)  # must not raise


def test_the_switch_leaves_a_satisfied_condition_alone(_absent, monkeypatch):
    """A dataset mark whose condition is False is not skipping, so nothing to escalate."""
    monkeypatch.setenv("SUPERGLM_REQUIRE_DATA", "1")
    pytest_runtest_setup(_dataset_item(condition=False))  # must not raise


# ── every suite that guards on a dataset must route through skip_reason ─────


@pytest.mark.parametrize(
    "module",
    [
        "tests.test_screening_guide_numbers",
        "tests.test_mixed_interaction_screening",
        "tests.test_realdata_parity",
    ],
)
def test_every_data_guarded_suite_carries_the_sentinel(module):
    """The enforcement switch must reach every suite that skips on a dataset.

    ``test_realdata_parity`` shipped a revision where the helpers moved to
    ``usable()`` but the ``skipif`` marks kept hardcoded reasons, so all 29 of
    its tests still skipped silently under the switch -- the exact hole the
    switch exists to close -- and the message then mislabelled a present file
    with no engine as "not found".  A suite that guards on a dataset without
    routing through ``skip_reason`` evades enforcement, so assert the tag.
    """
    import importlib

    mod = importlib.import_module(module)
    marks = [(k, v) for k, v in vars(mod).items() if k.endswith("_SKIP")]
    assert marks, f"{module} declares no dataset skip mark"

    for name, dec in marks:
        mark = getattr(dec, "mark", None)
        assert mark is not None and mark.name == "skipif", f"{module}.{name} is not a skipif mark"
        if not any(mark.args):
            continue  # dataset usable here; the mark is not skipping, nothing to escalate
        assert str(mark.kwargs.get("reason", "")).startswith(_datasets.SKIP_SENTINEL), (
            f"{module}.{name} builds a skip reason that bypasses _datasets.skip_reason, "
            "so SUPERGLM_REQUIRE_DATA cannot reach it"
        )
