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

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

from . import _datasets
from .conftest import pytest_runtest_setup, skip_mark_reason, skipif_is_active
from .test_fetch_fremtpl import data_guarded_suites

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
    with pytest.raises(pytest.fail.Exception, match="cannot decide whether this skipif skips"):
        pytest_runtest_setup(guard)


def test_a_string_condition_on_an_unrelated_skip_is_left_alone(_absent, monkeypatch):
    """The refusal above is scoped by the sentinel, so it cannot reach other suites."""
    monkeypatch.setenv("SUPERGLM_REQUIRE_DATA", "1")
    unrelated = _Item(_mark("sys.platform == 'win32'", reason="windows only"))
    pytest_runtest_setup(unrelated)  # must not raise


def test_the_refusal_names_the_mark_and_not_the_env_var(_absent, monkeypatch):
    """``skipif_is_active`` has two callers and only one of them is the switch.

    ``tests/test_dataset_guard.py`` calls it with ``SUPERGLM_REQUIRE_DATA``
    unset and irrelevant, so a message opening with that name sends whoever
    reads it to the wrong place.
    """
    monkeypatch.setenv("SUPERGLM_REQUIRE_DATA", "1")
    guard = _Item(_mark("1 > 2", reason=_datasets.skip_reason("nope.parquet")))
    with pytest.raises(pytest.fail.Exception) as excinfo:
        pytest_runtest_setup(guard)
    assert "SUPERGLM_REQUIRE_DATA" not in str(excinfo.value), str(excinfo.value)
    assert "evaluates against the test module's globals" in str(excinfo.value)


# ── a plain ``skip`` skips too, so the switch has to reach it ────────────────


#: ``pytest.mark.skip`` accepts its reason positionally or by keyword, and
#: ``_pytest.skipping`` builds the same ``Skip(reason=...)`` from either.
_PLAIN_SKIP_FORMS = {
    "reason= keyword": lambda reason: pytest.mark.skip(reason=reason).mark,
    "positional reason": lambda reason: pytest.mark.skip(reason).mark,
}


@pytest.mark.parametrize("form", list(_PLAIN_SKIP_FORMS), ids=list(_PLAIN_SKIP_FORMS))
def test_a_sentinel_carrying_plain_skip_is_escalated(_absent, monkeypatch, form):
    """The hook read ``skipif`` only, and a ``skip`` is the same silence.

    ``@pytest.mark.skip(reason=_datasets.skip_reason(...))`` carries the
    sentinel, reads as a dataset guard to any human, skips exactly like the
    ``skipif`` beside it -- and went unescalated.  Measured before the fix
    through a real pytest run with the switch armed: ``1 skipped``.  A ``skip``
    has no condition, so escalating it is unconditional and cannot be wrong.
    """
    monkeypatch.setenv("SUPERGLM_REQUIRE_DATA", "1")
    item = _Item(_PLAIN_SKIP_FORMS[form](_datasets.skip_reason("nope.parquet")))
    with pytest.raises(pytest.fail.Exception) as excinfo:
        pytest_runtest_setup(item)
    assert "SUPERGLM_REQUIRE_DATA is set" in str(excinfo.value)


def test_the_switch_leaves_an_unrelated_plain_skip_alone(_absent, monkeypatch):
    """Widening to ``skip`` must not widen past the sentinel.

    ``tests/conftest.py`` adds exactly such a mark to every browser item on
    every run without ``--run-browser``; escalating those would fail the suite
    on a machine with no Chromium.
    """
    monkeypatch.setenv("SUPERGLM_REQUIRE_DATA", "1")
    browser = _Item(pytest.mark.skip(reason="pass --run-browser to run Playwright tests").mark)
    pytest_runtest_setup(browser)  # must not raise


def test_the_reason_reader_agrees_with_pytests_own_skip_dataclass():
    """Both spellings of a ``skip`` reason are the same mark to pytest."""
    from _pytest.skipping import Skip

    for form in _PLAIN_SKIP_FORMS.values():
        mark = form("why")
        assert Skip(*mark.args, **mark.kwargs).reason == skip_mark_reason(mark) == "why"


# ── the hook has to be registered, and has to win the ordering ──────────────


_HOOK_PROBE = """\
import pytest

from tests import _datasets

# No dataset, whatever this machine actually holds.
_datasets.usable = lambda name: None
REASON = _datasets.skip_reason("nothing.parquet")


@pytest.mark.{decorator}
def test_guarded():
    raise AssertionError("the body must never run; this item can only skip or error")
"""

#: Both mark names pytest turns into a skip.  The ``skip`` case is the one that
#: was unescalated, and it needs the end-to-end form for the same reason the
#: ``skipif`` case does: ``_pytest.skipping`` evaluates BOTH in its own
#: ``tryfirst`` ``pytest_runtest_setup``, so the ordering that makes this hook
#: win is a property of the registration and not of the function.
_HOOK_PROBE_MARKS = {
    "skipif": "skipif(True, reason=REASON)",
    "skip": "skip(reason=REASON)",
}


def _run_hook_probe(
    tmp_path: Path, require_data: str, decorator: str
) -> subprocess.CompletedProcess[str]:
    """Run one sentinel-guarded item in a real pytest, with ``tests.conftest`` loaded."""
    ini = tmp_path / "pytest.ini"
    ini.write_text("[pytest]\naddopts =\n", encoding="utf-8")
    probe = tmp_path / "test_hook_probe.py"
    probe.write_text(_HOOK_PROBE.format(decorator=decorator), encoding="utf-8")
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


@pytest.mark.parametrize("mark", list(_HOOK_PROBE_MARKS), ids=list(_HOOK_PROBE_MARKS))
def test_the_switch_works_through_a_real_pytest_run(tmp_path, mark):
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
    decorator = _HOOK_PROBE_MARKS[mark]
    armed = _run_hook_probe(tmp_path, "1", decorator)
    assert "1 error" in armed.stdout, armed.stdout + armed.stderr
    assert "SUPERGLM_REQUIRE_DATA is set" in armed.stdout, armed.stdout

    disarmed = _run_hook_probe(tmp_path, "0", decorator)
    assert "1 skipped" in disarmed.stdout, disarmed.stdout + disarmed.stderr


# ── every suite that guards on a dataset must route through skip_reason ─────


def _suite_imported_with_the_dataset_absent(module: str, monkeypatch, directory: Path = None):
    """Import *module*'s source afresh, with nothing readable, under a private name.

    A suite decides its ``skipif`` marks at import.  On a machine that HAS the
    parquet every condition is false, so reading the marks says nothing at all
    -- and the real-data job is *precisely* a machine that has it, which makes
    the data-present run the one where this check must not go quiet.  Forcing
    the absent branch is what makes it bite on every machine.

    Under a private name and NOT via ``importlib.reload``, deliberately: reload
    rebinds the classes in ``sys.modules[module]``, and pytest has already
    collected items that hold the originals.  ``sys.modules[module]`` is left
    exactly as it was found.

    The ``tests.`` prefix on ``probe_name`` is load-bearing, not cosmetic: it is
    what makes ``spec.parent == "tests"``, so the suite's own
    ``from . import _datasets`` resolves to the ALREADY-PATCHED ``tests._datasets``
    rather than to a fresh copy.  Drop it and the import raises
    ``attempted relative import with no known parent package``.

    *directory* defaults to this package and exists so the two properties above
    -- the ``Skipped`` catch and the relative-import resolution -- can be tested
    on a throwaway module instead of only through a mutation of a real suite.
    """
    leaf = module.rsplit(".", 1)[-1]
    probe_name = f"tests._{leaf}__dataset_absent_probe"
    source = (directory or Path(__file__).parent) / f"{leaf}.py"
    spec = importlib.util.spec_from_file_location(probe_name, source)
    assert spec is not None and spec.loader is not None, module
    probe = importlib.util.module_from_spec(spec)

    monkeypatch.setattr(_datasets, "usable", lambda name: None)
    sys.modules[probe_name] = probe
    try:
        spec.loader.exec_module(probe)
    except pytest.skip.Exception as exc:
        # A module-scope ``pytest.importorskip`` -- plausible for pyarrow --
        # raises ``Skipped`` straight through this helper, and pytest then
        # reports THIS check as skipped rather than failed.  It carries no
        # sentinel, so the switch cannot escalate it either: the guard would go
        # quiet again by a different door from the one it just closed.
        # Measured before this catch, switch armed: ``27 passed, 1 skipped``.
        raise AssertionError(
            f"{module} skipped at import ({exc}), so this check asserted nothing about it. "
            f"A data-guarded suite must be importable without its dataset."
        ) from exc
    finally:
        sys.modules.pop(probe_name, None)
    return probe


def _unpack_marks(obj):
    """The ``Mark`` objects on *obj*, whatever spelling ``pytestmark`` carries.

    ``pytestmark`` may be a single ``MarkDecorator`` or a list of either
    ``MarkDecorator`` or ``Mark`` -- pytest normalises all of them, so a reader
    that assumed one shape would be fail-open on the others.
    """
    marks = getattr(obj, "pytestmark", []) or []
    if not isinstance(marks, (list, tuple)):
        marks = [marks]
    return [getattr(mark, "mark", mark) for mark in marks]


def _dataset_skip_marks(module: str, mod) -> list[tuple[str, object]]:
    """Every ``skipif`` the freshly-imported *mod* would hand pytest.

    Read two ways, because each alone is fail-open in the other's direction:

    * **by name** -- module-level bindings ending ``_SKIP``.  This is the only
      reading that catches a guard DECLARED and never applied to anything.
    * **by shape** -- every ``pytestmark`` in the module namespace, plus the
      module's own.  This is the only reading that catches a mark written
      INLINE on a new test function or class.

    The shape reading is the one that was missing, and it re-opened #261 one
    level down.  Measured on this tree: a single new test carrying
    ``@pytest.mark.skipif(_datasets.find(...) is None, reason="no other data")``
    left all 52 gate tests green, and the real-data job's own command then
    reported ``29 passed, 1 skipped`` with the switch armed -- a silent skip
    inside the job whose entire purpose is that a skip cannot read as a pass.
    """
    found: list[tuple[str, object]] = []
    for name, value in vars(mod).items():
        if name.endswith("_SKIP"):
            found.append((f"{module}.{name}", getattr(value, "mark", value)))
    for holder, label in [(mod, module), *((v, f"{module}.{k}") for k, v in vars(mod).items())]:
        for mark in _unpack_marks(holder):
            if getattr(mark, "name", None) == "skipif":
                found.append((label, mark))
    return found


def _fake_suite(tmp_path: Path, name: str, body: str) -> str:
    """Write a throwaway suite under *tmp_path* and return its dotted name."""
    (tmp_path / f"{name}.py").write_text(body, encoding="utf-8")
    return f"tests.{name}"


def test_a_suite_that_skips_at_import_is_a_failure_not_a_skip(tmp_path, monkeypatch):
    """``Skipped`` derives from ``BaseException`` and rode straight out of the helper.

    pytest then reported THIS check as skipped, so the guard went quiet in a
    summary line that reads exactly like a pass -- with no sentinel on the item,
    so ``SUPERGLM_REQUIRE_DATA`` could not escalate it either.  Measured before
    the catch, switch armed: ``27 passed, 1 skipped``.
    """
    module = _fake_suite(
        tmp_path,
        "test_skips_at_import",
        "import pytest\n\npytest.importorskip('a_module_that_is_definitely_not_installed')\n",
    )
    try:
        _suite_imported_with_the_dataset_absent(module, monkeypatch, tmp_path)
    except AssertionError as exc:
        assert "skipped at import" in str(exc), str(exc)
    except BaseException as exc:  # noqa: BLE001 - the point is that nothing else may escape
        raise AssertionError(
            f"a module-scope skip left the helper as {type(exc).__name__}, which pytest "
            f"reports as a skipped guard rather than a failed one"
        ) from exc
    else:
        raise AssertionError("the import did not fail at all")


def test_the_freshly_imported_suite_sees_the_patched_datasets_module(tmp_path, monkeypatch):
    """The ``tests.`` prefix on the probe name is load-bearing, not cosmetic.

    It is what makes ``spec.parent == "tests"``, so a suite's own
    ``from . import _datasets`` binds the ALREADY-PATCHED module.  Without it the
    import raises ``attempted relative import with no known parent package``;
    with a prefix but a fresh copy, every guard would evaluate against the real
    filesystem and the check would say nothing on a machine holding the data.
    """
    module = _fake_suite(
        tmp_path,
        "test_reads_datasets",
        "from . import _datasets\n\nSEEN = _datasets.usable('freMTPL2freq.parquet')\n",
    )
    probe = _suite_imported_with_the_dataset_absent(module, monkeypatch, tmp_path)
    assert probe.SEEN is None
    assert probe._datasets is _datasets


def test_the_discovery_sees_a_skipif_written_inline_on_a_test(tmp_path, monkeypatch):
    """The name scan cannot see this one, and it is how #261 re-opens.

    A new test carrying its own ``@pytest.mark.skipif`` needs no ``*_SKIP``
    binding at all, so a discovery that reads names finds nothing to check.
    """
    module = _fake_suite(
        tmp_path,
        "test_inline_only",
        "import pytest\n\n\n"
        '@pytest.mark.skipif(True, reason="not routed through skip_reason")\n'
        "def test_guarded():\n    pass\n",
    )
    probe = _suite_imported_with_the_dataset_absent(module, monkeypatch, tmp_path)
    found = _dataset_skip_marks(module, probe)
    assert [label for label, _ in found] == [f"{module}.test_guarded"], found


def test_the_discovery_still_sees_a_guard_declared_and_never_applied(tmp_path, monkeypatch):
    """And the shape scan cannot see THAT one, which is why both readings stay."""
    module = _fake_suite(
        tmp_path,
        "test_declared_only",
        "import pytest\n\n"
        'UNUSED_SKIP = pytest.mark.skipif(True, reason="declared, never applied")\n',
    )
    probe = _suite_imported_with_the_dataset_absent(module, monkeypatch, tmp_path)
    found = _dataset_skip_marks(module, probe)
    assert [label for label, _ in found] == [f"{module}.UNUSED_SKIP"], found


def _assert_is_a_dataset_guard(label: str, mark) -> None:
    """The three questions, in the order the design depends on.

    The sentinel is asked BEFORE the condition, and that ordering is the same
    invariant ``tests/conftest.py``'s hook keeps: ``skipif_is_active`` refuses a
    string condition, so anything that reaches it must already be known to be a
    mark this project built.  Asked the other way round -- which this check did
    -- an unrelated ``skipif("sys.platform == 'win32'", ...)`` was reported as
    "SUPERGLM_REQUIRE_DATA cannot evaluate the string skipif condition ... on a
    dataset guard", naming an env var that is unset here and a role the mark
    does not have.
    """
    assert getattr(mark, "name", None) == "skipif", f"{label} is not a skipif mark"
    assert skip_mark_reason(mark).startswith(_datasets.SKIP_SENTINEL), (
        f"{label} builds a skip reason that bypasses _datasets.skip_reason, "
        "so SUPERGLM_REQUIRE_DATA cannot reach it"
    )
    assert skipif_is_active(mark), (
        f"{label} does not skip even with no dataset readable, so its "
        f"condition is not the dataset guard it is named for"
    )


def test_a_non_dataset_mark_is_reported_for_what_it_is_not_for_its_condition():
    """Ordering, pinned where it can be read.

    Under the old order this raised ``Failed`` from the string-condition
    refusal, not ``AssertionError`` -- measured on a real ``WIN_SKIP`` added to
    ``tests/test_screening_guide_numbers.py``.
    """
    unrelated = _mark("sys.platform == 'win32'", reason="windows only")
    with pytest.raises(AssertionError, match="bypasses _datasets.skip_reason"):
        _assert_is_a_dataset_guard("some_suite.WIN_SKIP", unrelated)


#: Derived, never restated: ``tests/test_fetch_fremtpl.py`` already discovers
#: these from the source for the workflow contract, and two hardcoded lists that
#: must agree are a drift waiting to happen.
_GUARDED_MODULES = sorted(f"tests.{Path(name).stem}" for name in data_guarded_suites())


def test_the_parametrized_module_list_is_not_empty():
    """An empty ``parametrize`` list is ONE SKIPPED ITEM, not a failure.

    So a discovery that quietly matched nothing would delete the check below
    from coverage and report it as a skip -- this branch's own defect, wearing
    this branch's own clothes.
    """
    assert _GUARDED_MODULES, "the data-guarded suite discovery found nothing"


@pytest.mark.parametrize("module", _GUARDED_MODULES)
def test_every_data_guarded_suite_carries_the_sentinel(module, monkeypatch):
    """The enforcement switch must reach every suite that skips on a dataset.

    ``test_realdata_parity`` shipped a revision where the helpers moved to
    ``usable()`` but the ``skipif`` marks kept hardcoded reasons, so all 29 of
    its tests still skipped silently under the switch -- the exact hole the
    switch exists to close -- and the message then mislabelled a present file
    with no engine as "not found".  A suite that guards on a dataset without
    routing through ``skip_reason`` evades enforcement, so assert the tag.

    Three ways this check has already been vacuous, all closed here:

    * it read module variables named ``*_SKIP_REASON``, and the bug it is named
      for had none -- the reason was inlined into the mark;
    * it read the marks of the ALREADY-imported module, so with the parquet
      present every condition was false and the loop skipped every mark.
      Measured: a ``FREQ_SKIP`` whose reason bypasses ``skip_reason`` passed on
      a data-present machine and failed only on a data-absent one.  The
      real-data job is a data-present machine.
    * it discovered guards by NAME only, so a ``skipif`` written inline on a new
      test evaded it entirely -- see :func:`_dataset_skip_marks`.

    The module list is the one ``tests/test_fetch_fremtpl.py`` derives from the
    source, not a second hardcoded copy: a fourth data-guarded suite has to be
    covered here the moment it exists, and a list that has to be remembered is
    a list that goes stale silently.

    The "is this mark skipping here?" question is answered by the hook's own
    :func:`~tests.conftest.skipif_is_active`, deliberately, and not by a second
    reading of ``mark.args``: this check had that reading too, so a suite that
    moved to ``skipif(condition=...)`` would have gone unchecked here at the
    same moment it went unescalated there.  It is asked SECOND, after the
    sentinel: ``skipif_is_active`` refuses a string condition, and asking it
    first pointed that refusal at marks this project never built.

    Reserved by this check, and stated because it is stronger than "route
    through ``skip_reason``": in these suites a module-level ``*_SKIP`` name,
    and any inline ``skipif``, is a dataset guard.  A ``WIN_SKIP`` for an
    unrelated platform condition belongs in a differently named binding.
    """
    mod = _suite_imported_with_the_dataset_absent(module, monkeypatch)
    marks = _dataset_skip_marks(module, mod)
    assert marks, f"{module} declares no dataset skip mark"

    for label, mark in marks:
        _assert_is_a_dataset_guard(label, mark)
