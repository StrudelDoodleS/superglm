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

import os
import subprocess
import sys
from pathlib import Path

import pytest

from . import _datasets
from .conftest import pytest_runtest_setup, skipif_is_active

_ROOT = Path(__file__).resolve().parents[1]


def _mark(*args, **kwargs):
    """A REAL ``skipif`` ``Mark``, not a stand-in for one.

    A hand-rolled stub fixes the mark's shape to whatever the hook happens to
    read today, which is how this file previously pinned ``args`` and missed
    the ``condition=`` form pytest supports equally.  Building the mark through
    ``pytest.mark.skipif`` means the three forms below are the three forms
    pytest itself accepts.
    """
    return pytest.mark.skipif(*args, **kwargs).mark


class _Item:
    """Enough of a pytest item for the hook to read.

    ``iter_markers`` filters by name the way a real item does, rather than
    asserting the hook only ever asks for ``skipif`` -- a hook that also looked
    at ``skip`` marks should fail on its behaviour, not on the stub.
    """

    def __init__(self, *marks):
        self._marks = marks

    def iter_markers(self, name):
        return iter(mark for mark in self._marks if mark.name == name)


def _dataset_item(condition=True):
    return _Item(_mark(condition, reason=_datasets.skip_reason("nope.parquet")))


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
    unrelated = _Item(_mark(True, reason="pass --run-browser to run Playwright editor tests"))
    pytest_runtest_setup(unrelated)  # must not raise


def test_the_switch_leaves_a_satisfied_condition_alone(_absent, monkeypatch):
    """A dataset mark whose condition is False is not skipping, so nothing to escalate."""
    monkeypatch.setenv("SUPERGLM_REQUIRE_DATA", "1")
    pytest_runtest_setup(_dataset_item(condition=False))  # must not raise


# ── every form pytest skips on, the switch must escalate ────────────────────


#: The three ways pytest accepts a ``skipif``.  ``_pytest.skipping`` takes the
#: condition from ``kwargs["condition"]`` when it is there and from ``args``
#: otherwise, and treats no condition at all as an unconditional skip.
_SKIPPING_FORMS = {
    "positional condition": lambda reason: _mark(True, reason=reason),
    "condition= keyword": lambda reason: _mark(condition=True, reason=reason),
    "no condition at all": lambda reason: _mark(reason=reason),
}


@pytest.mark.parametrize("form", list(_SKIPPING_FORMS), ids=list(_SKIPPING_FORMS))
def test_every_form_pytest_skips_on_is_escalated(_absent, monkeypatch, form):
    """Reading only ``mark.args`` escalated one of these three and missed two.

    Measured before the fix, one item per form under the armed switch: the
    positional form failed, the ``condition=`` and no-condition forms were let
    through -- while a real pytest run skipped all three.  A suite written in
    either of the missed forms carries the sentinel, reads as guarded, skips as
    intended, and is silently exempt from the switch.  That leaves no trace
    anywhere, which is worse than the hole it re-opens.
    """
    monkeypatch.setenv("SUPERGLM_REQUIRE_DATA", "1")
    item = _Item(_SKIPPING_FORMS[form](_datasets.skip_reason("nope.parquet")))
    with pytest.raises(pytest.fail.Exception) as excinfo:
        pytest_runtest_setup(item)
    assert "SUPERGLM_REQUIRE_DATA is set" in str(excinfo.value)


def test_a_string_condition_on_a_dataset_guard_is_refused_not_guessed(_absent, monkeypatch):
    """pytest ``eval``s a string condition; this hook must not pretend to.

    ``any(("False",))`` is ``True`` and ``any(("True",))`` is also ``True``, so
    a naive read gets the answer right half the time by accident.  Refusing
    names the problem at the mark that has it.
    """
    monkeypatch.setenv("SUPERGLM_REQUIRE_DATA", "1")
    guard = _Item(_mark("1 > 2", reason=_datasets.skip_reason("nope.parquet")))
    with pytest.raises(pytest.fail.Exception, match="string skipif condition"):
        pytest_runtest_setup(guard)


def test_a_string_condition_on_an_unrelated_skip_is_left_alone(_absent, monkeypatch):
    """The refusal above is scoped by the sentinel, so it cannot reach other suites."""
    monkeypatch.setenv("SUPERGLM_REQUIRE_DATA", "1")
    unrelated = _Item(_mark("sys.platform == 'win32'", reason="windows only"))
    pytest_runtest_setup(unrelated)  # must not raise


# ── the hook has to be registered, and has to win the ordering ──────────────


_HOOK_PROBE = """\
import pytest

from tests import _datasets

# No dataset, whatever this machine actually holds.
_datasets.usable = lambda name: None
REASON = _datasets.skip_reason("nothing.parquet")


@pytest.mark.skipif(True, reason=REASON)
def test_guarded():
    raise AssertionError("the body must never run; this item can only skip or error")
"""


def _run_hook_probe(tmp_path: Path, require_data: str) -> subprocess.CompletedProcess[str]:
    """Run one sentinel-guarded item in a real pytest, with ``tests.conftest`` loaded."""
    ini = tmp_path / "pytest.ini"
    ini.write_text("[pytest]\naddopts =\n", encoding="utf-8")
    probe = tmp_path / "test_hook_probe.py"
    probe.write_text(_HOOK_PROBE, encoding="utf-8")
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "tests.conftest",
            "-p",
            "no:cacheprovider",
            "-c",
            str(ini),
            "-q",
            str(probe),
        ],
        cwd=_ROOT,
        env={**os.environ, "PYTHONPATH": str(_ROOT), "SUPERGLM_REQUIRE_DATA": require_data},
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_the_switch_works_through_a_real_pytest_run(tmp_path):
    """Everything above calls the hook as a function; this runs it as a hook.

    That gap is not theoretical.  ``tryfirst`` is what puts this impl ahead of
    ``_pytest.skipping``'s own ``tryfirst`` ``pytest_runtest_setup`` -- pluggy
    appends ``tryfirst`` impls and iterates the list reversed, so the later
    registration runs first, and the first impl to raise ends the call.  Drop
    the decorator and skipping wins: measured, the same probe goes from
    ``1 error`` to ``1 skipped`` and every direct-call test above stays green.
    Renaming the function, or unregistering it, is invisible to them too.

    A ``-p`` plugin is registered EARLIER than a conftest, so this is the
    pessimistic case for the ordering: if the hook wins here it wins as a
    conftest.
    """
    armed = _run_hook_probe(tmp_path, "1")
    assert "1 error" in armed.stdout, armed.stdout + armed.stderr
    assert "SUPERGLM_REQUIRE_DATA is set" in armed.stdout, armed.stdout

    disarmed = _run_hook_probe(tmp_path, "0")
    assert "1 skipped" in disarmed.stdout, disarmed.stdout + disarmed.stderr


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

    The "is this mark skipping here?" question is answered by the hook's own
    :func:`~tests.conftest.skipif_is_active`, deliberately, and not by a second
    reading of ``mark.args``: this check had that reading too, so a suite that
    moved to ``skipif(condition=...)`` would have gone unchecked here at the
    same moment it went unescalated there.
    """
    import importlib

    mod = importlib.import_module(module)
    marks = [(k, v) for k, v in vars(mod).items() if k.endswith("_SKIP")]
    assert marks, f"{module} declares no dataset skip mark"

    for name, dec in marks:
        mark = getattr(dec, "mark", None)
        assert mark is not None and mark.name == "skipif", f"{module}.{name} is not a skipif mark"
        if not skipif_is_active(mark):
            continue  # dataset usable here; the mark is not skipping, nothing to escalate
        assert str(mark.kwargs.get("reason", "")).startswith(_datasets.SKIP_SENTINEL), (
            f"{module}.{name} builds a skip reason that bypasses _datasets.skip_reason, "
            "so SUPERGLM_REQUIRE_DATA cannot reach it"
        )
