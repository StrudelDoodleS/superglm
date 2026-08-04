"""Tests for ``scripts/mutation_gate.py``.

The gate decides whether a change's regression tests constrain it, so a defect
here is invisible: it either blesses a change nothing demonstrates, or accuses a
correct one.  Every scenario below pins a verdict that was wrong in a reviewed
revision of the script, with the mechanism named in the test.

Two layers, deliberately:

- unit tests over the pure helpers, fast enough to run in the default suite;
- end-to-end scenarios marked ``slow``, each building real commits in a throwaway
  **clone** and invoking the gate as a subprocess.  They clone rather than branch
  in place because the harness commits with ``git add -A``; run against a
  developer's checkout it would sweep uncommitted work into a scratch branch and
  delete it.
"""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "scripts" / "mutation_gate.py"

_spec = importlib.util.spec_from_file_location("mutation_gate", GATE)
assert _spec and _spec.loader
mutation_gate = importlib.util.module_from_spec(_spec)
# Registered before execution: @dataclass resolves its owning module through
# sys.modules, and raises AttributeError on None if it is not there yet.
sys.modules["mutation_gate"] = mutation_gate
_spec.loader.exec_module(mutation_gate)


# --------------------------------------------------------------------------
# Unit tests over the pure helpers
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("tests/test_links.py", True),
        # pytest's python_files default is "test_*.py" *and* "*_test.py"; the
        # suffix form was rejected, so a change whose regression test used it
        # was reported NO EVIDENCE.
        ("tests/regression_test.py", True),
        ("tests/editor/test_nested.py", True),
        ("tests/helpers.py", False),
        ("src/superglm/test_thing.py", False),
        ("tests/test_notpython.txt", False),
    ],
)
def test_is_test_module_matches_both_discovery_patterns(path, expected):
    assert mutation_gate.is_test_module(path) is expected


@pytest.mark.parametrize(
    ("node_id", "expected"),
    [
        ("tests/test_links.py::test_bound", ("tests.test_links", "test_bound")),
        (
            "tests/test_links.py::TestBand::test_bound",
            ("tests.test_links.TestBand", "test_bound"),
        ),
        (
            "tests/editor/test_x.py::TestA::TestB::test_y",
            ("tests.editor.test_x.TestA.TestB", "test_y"),
        ),
    ],
)
def test_junit_key_matches_pytest_classnames(node_id, expected):
    assert mutation_gate.junit_key(node_id) == expected


@pytest.mark.parametrize(
    ("item", "expected"),
    [
        ("tests/t.py::test_a[1]", "tests/t.py::test_a"),
        ("tests/t.py::test_a[a[b]]", "tests/t.py::test_a"),
        ("tests/t.py::test_a", "tests/t.py::test_a"),
    ],
)
def test_owning_function_strips_the_parametrise_suffix(item, expected):
    assert mutation_gate.owning_function(item) == expected


_ADDED = frozenset({"_new", "_helper", "_x", "thing"})


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        # A call-phase exception is a JUnit <failure>, tag-identical to an
        # assertion failure. Only the message separates "the fix is absent" from
        # "the behaviour is wrong".
        ("ModuleNotFoundError: No module named 'superglm._new'", True),
        ("ImportError: cannot import name '_helper' from 'superglm.solvers'", True),
        ("AttributeError: module 'superglm.links' has no attribute '_x'", True),
        ("NameError: name 'thing' is not defined", True),
        ("  ImportError: cannot import name '_helper' from 'superglm.x'", True),
        ("assert 1 == 2", False),
        ("AssertionError: bound not enforced", False),
        # Same shapes, but naming something the change did not add: the symbol
        # was always there, so the failure is about behaviour.
        ("ModuleNotFoundError: No module named 'superglm.links'", False),
        ("NameError: name 'unrelated' is not defined", False),
    ],
)
def test_is_symbol_error_requires_a_name_this_change_added(message, expected):
    assert mutation_gate.is_symbol_error(message, _ADDED) is expected


def test_runtime_attribute_error_is_behaviour_not_absence():
    """Matching on the exception class alone discarded real evidence.

    When the defect *is* a null-handling bug, the regression test fails against
    the mutant with a genuine ``AttributeError`` -- a kill, not a missing
    symbol. The same message *is* absence when the change added that attribute,
    so the name decides, not the exception class.
    """
    message = "AttributeError: 'NoneType' object has no attribute 'lower'"
    assert mutation_gate.is_symbol_error(message, frozenset({"unrelated"})) is False
    assert mutation_gate.is_symbol_error(message, frozenset({"lower"})) is True


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("tests/conftest.py", True),
        ("tests/_wood_reml_oracles.py", True),
        ("tests/editor/conftest.py", True),
        ("tests/test_links.py", False),
        ("tests/regression_test.py", False),
        ("tests/data/fixture.json", False),
        ("src/superglm/links.py", False),
    ],
)
def test_is_test_support_matches_the_files_pytest_will_not_collect(path, expected):
    assert mutation_gate.is_test_support(path) is expected


def test_module_shell_sees_data_a_stable_parametrise_id_hides():
    """The remaining attribution blind spot, and how the gate detects it.

    73% of this repository's ``pytest.param`` sites carry an explicit ``id=``.
    Retuning an expected value for an existing case leaves both the function's
    dump and its collected node ID untouched -- only the module shell moves.
    """
    before = (
        "CASES = ['a']\nEXPECTED = {'a': 1}\n\n\n"
        "@pytest.mark.parametrize('c', CASES)\ndef test_x(c):\n    assert EXPECTED[c]\n"
    )
    after = before.replace("{'a': 1}", "{'a': 2}")

    assert mutation_gate.test_functions(before) == mutation_gate.test_functions(after)
    assert mutation_gate.module_shell(before) != mutation_gate.module_shell(after)


def test_module_shell_ignores_the_test_functions_themselves():
    """Otherwise every strengthened test would also read as unattributable."""
    before = "X = 1\n\n\ndef test_a():\n    assert 1\n\n\nclass TestG:\n    def test_b(self):\n        assert 1\n"
    after = before.replace("assert 1\n\n\nclass", "assert 1 and True\n\n\nclass")

    assert before != after
    assert mutation_gate.module_shell(before) == mutation_gate.module_shell(after)


def test_module_shell_survives_a_syntax_error():
    assert mutation_gate.module_shell("def test_(:\n") == ""


def test_defined_names_reports_module_and_binding_names(tmp_path):
    package = tmp_path / "superglm"
    package.mkdir()
    (package / "_new_helper.py").write_text(
        "CONSTANT = 1\n\n\ndef helper():\n    inner = 2\n    return inner\n\n\nclass Band:\n    lower = 0\n"
    )
    names = mutation_gate.defined_names(package)

    assert {"_new_helper", "CONSTANT", "helper", "inner", "Band", "lower"} <= names


def test_test_functions_reports_class_qualified_names():
    source = (
        "def test_top():\n    pass\n\n\n"
        "class TestGroup:\n    def test_inner(self):\n        pass\n\n\n"
        "def helper():\n    pass\n"
    )
    assert set(mutation_gate.test_functions(source)) == {"test_top", "TestGroup::test_inner"}


def test_test_functions_ignores_a_pure_move_but_not_a_changed_body():
    original = "def test_a():\n    assert 1\n\n\ndef test_b():\n    assert 2\n"
    moved = "def test_b():\n    assert 2\n\n\ndef test_a():\n    assert 1\n"
    tightened = "def test_a():\n    assert 1 and True\n\n\ndef test_b():\n    assert 2\n"

    assert mutation_gate.test_functions(original) == mutation_gate.test_functions(moved)
    assert (
        mutation_gate.test_functions(original)["test_a"]
        != (mutation_gate.test_functions(tightened)["test_a"])
    )


def test_test_functions_cannot_see_a_data_driven_case():
    """The blind spot the collected-item diff exists to cover."""
    before = "CASES = [1]\n\n\n@pytest.mark.parametrize('c', CASES)\ndef test_x(c):\n    assert c\n"
    after = (
        "CASES = [1, 2]\n\n\n@pytest.mark.parametrize('c', CASES)\ndef test_x(c):\n    assert c\n"
    )
    assert mutation_gate.test_functions(before) == mutation_gate.test_functions(after)


def test_test_functions_survives_a_syntax_error():
    assert mutation_gate.test_functions("def test_(:\n") == {}


# --------------------------------------------------------------------------
# End-to-end scenarios
# --------------------------------------------------------------------------

# A scratch production module the scenarios own outright, committed in the
# baseline so it exists at *both* revisions. Mutating a real constant instead
# coupled every scenario to that constant's value: a pull request legitimately
# retuning it would fail `bump`'s assertion, and since the workflow runs this
# suite before invoking the gate, the whole job would go red for an unrelated
# reason.
FIXTURE = "src/superglm/_gate_fixture.py"
IMPORT = "from superglm._gate_fixture import ETA_MIN\n"
PARAM = "import pytest\n\n" + IMPORT
CASE = "tests/test_gate_case.py"
NEWMOD = "src/superglm/_gatefix.py"
STATUS = re.compile(r"mutation gate: (SKIP|PASS|FAIL|NO EVIDENCE|INCONCLUSIVE)")


@pytest.fixture(scope="module")
def clone(tmp_path_factory):
    """A throwaway clone with one baseline commit the scenarios branch from."""
    dest = tmp_path_factory.mktemp("gate-repo") / "repo"
    subprocess.run(
        ["git", "clone", "--quiet", "--no-hardlinks", str(ROOT), str(dest)],
        check=True,
        capture_output=True,
    )
    for key, value in (("user.email", "gate@test"), ("user.name", "Gate Test")):
        subprocess.run(["git", "config", key, value], cwd=dest, check=True, capture_output=True)
    # `git clone` takes committed content, so the clone would otherwise run the
    # *last committed* gate and silently ignore the working tree. Every scenario
    # would then pass or fail for the wrong revision of the thing under test.
    shutil.copy2(GATE, dest / "scripts" / "mutation_gate.py")
    write(dest, FIXTURE, '"""Scratch module owned by the gate tests."""\n\nETA_MIN = -80.0\n')
    write(
        dest,
        CASE,
        IMPORT + "\n\ndef test_pin():\n    assert isinstance(ETA_MIN, float)\n",
    )
    commit(dest, "baseline")
    subprocess.run(["git", "branch", "gate-base"], cwd=dest, check=True, capture_output=True)
    return dest


def write(repo: Path, path: str, text: str) -> None:
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)


def bump(repo: Path) -> None:
    """The production change every scenario mutates against."""
    target = repo / FIXTURE
    source = target.read_text()
    assert "ETA_MIN = -80.0" in source
    target.write_text(source.replace("ETA_MIN = -80.0", "ETA_MIN = -70.0", 1))


def commit(repo: Path, message: str) -> None:
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-qm", message], cwd=repo, check=True, capture_output=True)


def branch(repo: Path, name: str, start: str = "gate-base") -> None:
    subprocess.run(
        ["git", "checkout", "-q", "-b", name, start], cwd=repo, check=True, capture_output=True
    )


def run_gate(repo: Path, base: str, head: str = "HEAD", **extra: str) -> tuple[int, str]:
    # Inherit the platform environment rather than replacing it. A hardcoded
    # "/usr/bin:/bin" PATH left the gate unable to find `git` anywhere but
    # Unix, so the suite crashed before reaching a verdict. Only what a
    # scenario depends on is overridden, and MUTATION_GATE_EXEMPT is cleared
    # explicitly so a developer's ambient export cannot turn every scenario
    # into a SKIP that still reads as green.
    env = {**os.environ, "HOME": str(repo), "MUTATION_GATE_EXEMPT": "", **extra}
    proc = subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "mutation_gate.py"),
            "--base-ref",
            base,
            "--head-ref",
            head,
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    out = proc.stdout + proc.stderr
    found = STATUS.findall(out)
    return proc.returncode, (found[-1] if found else f"NO STATUS: {out[-400:]}")


@pytest.mark.slow
def test_weak_added_test_does_not_constrain_the_change(clone):
    """The failure mode the gate exists for: a test that passes at both ends."""
    branch(clone, "s-weak")
    bump(clone)
    write(
        clone,
        CASE,
        IMPORT + "\n\ndef test_pin():\n    assert isinstance(ETA_MIN, float)"
        "\n\n\ndef test_weak():\n    assert isinstance(ETA_MIN, float)\n",
    )
    commit(clone, "weak")
    assert run_gate(clone, "gate-base") == (1, "FAIL")


@pytest.mark.slow
def test_added_test_that_fails_against_the_mutant_passes(clone):
    branch(clone, "s-strong")
    bump(clone)
    write(clone, CASE, IMPORT + "\n\ndef test_strong():\n    assert ETA_MIN == -70.0\n")
    commit(clone, "strong")
    assert run_gate(clone, "gate-base") == (0, "PASS")


@pytest.mark.slow
def test_parametrised_test_is_judged_on_its_items(clone):
    """One function, three items.

    Comparing pytest's passed-item count against a count of ``ast`` function
    definitions reported "Only 3 of 1 added test(s) pass" and refused every
    parametrised regression test -- the house style in this repository.
    """
    branch(clone, "s-param")
    bump(clone)
    write(
        clone,
        CASE,
        PARAM + '\n\n@pytest.mark.parametrize("bound", [-70.0, -70.0, -70.0])\n'
        "def test_param(bound):\n    assert ETA_MIN == bound\n",
    )
    commit(clone, "param")
    assert run_gate(clone, "gate-base") == (0, "PASS")


@pytest.mark.slow
def test_new_parametrise_case_counts_even_when_the_function_is_unchanged(clone):
    """A case appended to a module-level list feeding an unchanged decorator.

    The function's ``ast.dump`` is byte identical, so AST comparison alone
    reported NO EVIDENCE for a genuinely new regression case.
    """
    branch(clone, "s-data-before")
    write(
        clone,
        CASE,
        PARAM + '\n\nCASES = [-80.0]\n\n\n@pytest.mark.parametrize("bound", CASES)\n'
        "def test_bounds(bound):\n    assert ETA_MIN == bound\n",
    )
    commit(clone, "data before")
    branch(clone, "s-data-after", "s-data-before")
    bump(clone)
    write(
        clone,
        CASE,
        PARAM + '\n\nCASES = [-70.0]\n\n\n@pytest.mark.parametrize("bound", CASES)\n'
        "def test_bounds(bound):\n    assert ETA_MIN == bound\n",
    )
    commit(clone, "data after")
    assert run_gate(clone, "s-data-before") == (0, "PASS")


@pytest.mark.slow
def test_strengthened_existing_test_counts_as_evidence(clone):
    branch(clone, "s-tighten")
    bump(clone)
    write(clone, CASE, IMPORT + "\n\ndef test_pin():\n    assert ETA_MIN == -70.0\n")
    commit(clone, "tighten")
    assert run_gate(clone, "gate-base") == (0, "PASS")


@pytest.mark.slow
def test_skipped_sibling_is_excluded_not_fatal(clone):
    branch(clone, "s-partial")
    bump(clone)
    write(
        clone,
        CASE,
        PARAM + "\n\ndef test_killer():\n    assert ETA_MIN == -70.0\n\n\n"
        '@pytest.mark.skip(reason="not evaluable here")\n'
        "def test_skipped():\n    assert ETA_MIN == -70.0\n",
    )
    commit(clone, "partial")
    assert run_gate(clone, "gate-base") == (0, "PASS")


@pytest.mark.slow
def test_only_a_skipped_added_test_is_inconclusive(clone):
    branch(clone, "s-allskip")
    bump(clone)
    write(
        clone,
        CASE,
        PARAM + '\n\n@pytest.mark.skip(reason="not evaluable here")\n'
        "def test_only_skipped():\n    assert ETA_MIN == -70.0\n",
    )
    commit(clone, "allskip")
    assert run_gate(clone, "gate-base") == (1, "INCONCLUSIVE")


@pytest.mark.slow
def test_uncollectable_node_id_does_not_poison_its_siblings(clone):
    """``pytest <uncollectable-id>`` is a usage error: exit 4, nothing runs."""
    branch(clone, "s-uncollect")
    bump(clone)
    write(
        clone,
        CASE,
        IMPORT + "\n\nclass HelperThing:\n"
        "    def test_not_collected(self):\n        assert False\n\n\n"
        "def test_killer():\n    assert ETA_MIN == -70.0\n",
    )
    commit(clone, "uncollect")
    assert run_gate(clone, "gate-base") == (0, "PASS")


@pytest.mark.slow
def test_added_test_red_at_head_is_a_defect_in_the_change(clone):
    branch(clone, "s-broken")
    bump(clone)
    write(
        clone,
        CASE,
        IMPORT + "\n\ndef test_always_fails():\n    assert ETA_MIN == 12345\n",
    )
    commit(clone, "broken")
    assert run_gate(clone, "gate-base") == (1, "INCONCLUSIVE")


@pytest.mark.slow
def test_fix_in_a_new_module_is_inconclusive_not_passed(clone):
    """At base the module is absent, so its tests error rather than fail."""
    branch(clone, "s-newmod")
    write(clone, NEWMOD, "def answer() -> int:\n    return 42\n")
    write(
        clone,
        CASE,
        "from superglm._gatefix import answer\n\n\ndef test_answer():\n    assert answer() == 42\n",
    )
    commit(clone, "new module")
    assert run_gate(clone, "gate-base") == (1, "INCONCLUSIVE")


@pytest.mark.slow
def test_in_body_import_error_is_not_a_kill(clone):
    """Moving the import inside the test turned INCONCLUSIVE into PASS.

    pytest records a call-phase ``ModuleNotFoundError`` as ``<failure>``,
    tag-identical to an assertion failure, so the missing-symbol policy was
    defeated by where the import was written.
    """
    branch(clone, "s-inbody")
    write(clone, NEWMOD, "def answer() -> int:\n    return 42\n")
    write(
        clone,
        CASE,
        "def test_answer():\n    from superglm._gatefix import answer\n\n"
        "    assert answer() == 42\n",
    )
    commit(clone, "in-body import")
    assert run_gate(clone, "gate-base") == (1, "INCONCLUSIVE")


@pytest.mark.slow
def test_base_side_skip_is_not_survival(clone):
    """An ``importorskip`` skips at base; that is unmeasured, not survived."""
    branch(clone, "s-baseskip")
    write(clone, NEWMOD, "def answer() -> int:\n    return 42\n")
    write(
        clone,
        CASE,
        "import pytest\n\n\ndef test_needs_new_module():\n"
        '    pytest.importorskip("superglm._gatefix")\n'
        "    from superglm._gatefix import answer\n\n    assert answer() == 42\n",
    )
    commit(clone, "base skip")
    assert run_gate(clone, "s-baseskip~1") == (1, "INCONCLUSIVE")


@pytest.mark.slow
def test_runtime_attribute_error_against_the_mutant_is_a_kill(clone):
    """A null-handling fix, whose regression test fails at base with AttributeError.

    Classifying every ``AttributeError`` as a missing symbol reported
    INCONCLUSIVE here and discarded real evidence. ``lower`` exists at both
    revisions, so the failure is behavioural.
    """
    branch(clone, "s-null-before")
    write(
        clone,
        FIXTURE,
        "ETA_MIN = -80.0\n\n\nclass Band:\n    lower = -80.0\n\n\ndef band():\n    return None\n",
    )
    commit(clone, "null-returning band")
    branch(clone, "s-null-after", "s-null-before")
    write(
        clone,
        FIXTURE,
        "ETA_MIN = -80.0\n\n\nclass Band:\n    lower = -80.0\n\n\ndef band():\n    return Band()\n",
    )
    write(
        clone,
        CASE,
        "from superglm._gate_fixture import band\n\n\n"
        "def test_band_has_a_lower_bound():\n    assert band().lower == -80.0\n",
    )
    commit(clone, "fix and regression test")
    assert run_gate(clone, "s-null-before") == (0, "PASS")


@pytest.mark.slow
def test_support_file_change_is_inconclusive_not_absent_evidence(clone):
    """An oracle table can carry a regression no discovery-named module records."""
    branch(clone, "s-support")
    bump(clone)
    write(clone, "tests/_gate_oracles.py", "EXPECTED_BOUND = -70.0\n")
    commit(clone, "oracle strengthened")
    assert run_gate(clone, "gate-base") == (1, "INCONCLUSIVE")


@pytest.mark.slow
def test_retuned_value_behind_a_stable_id_is_inconclusive_not_absent(clone):
    """Neither the function dump nor the node ID moves; the module shell does.

    Reported NO EVIDENCE, which reads as "you wrote no test" for a change whose
    regression genuinely constrains it. The gate cannot attribute it, and now
    says so rather than claiming an absence it did not establish.
    """
    body = (
        PARAM + '\n\nEXPECTED = {{"bound": {value}}}\n\n\n'
        '@pytest.mark.parametrize("case", ["bound"])\n'
        "def test_case(case):\n    assert ETA_MIN == EXPECTED[case]\n"
    )
    branch(clone, "s-stable-before")
    write(clone, CASE, body.format(value="-80.0"))
    commit(clone, "stable id, old value")
    branch(clone, "s-stable-after", "s-stable-before")
    bump(clone)
    write(clone, CASE, body.format(value="-70.0"))
    commit(clone, "stable id, retuned value")
    assert run_gate(clone, "s-stable-before") == (1, "INCONCLUSIVE")


@pytest.mark.slow
def test_production_change_with_no_test_change_has_no_evidence(clone):
    branch(clone, "s-noev")
    bump(clone)
    commit(clone, "noev")
    assert run_gate(clone, "gate-base") == (1, "NO EVIDENCE")


@pytest.mark.slow
def test_change_outside_the_package_is_skipped(clone):
    branch(clone, "s-docs")
    write(clone, "README.md", (clone / "README.md").read_text() + "\n")
    commit(clone, "docs")
    assert run_gate(clone, "gate-base") == (0, "SKIP")


@pytest.mark.slow
def test_uncommitted_work_is_refused_rather_than_silently_unmeasured(clone):
    """The diff is ``merge_base..head``; a fix in the working tree is invisible.

    Without this the gate reported SKIP with exit 0 for a branch whose fix was
    real but uncommitted, which reads as a pass.
    """
    branch(clone, "s-dirty")
    bump(clone)
    try:
        assert run_gate(clone, "gate-base") == (1, "INCONCLUSIVE")
    finally:
        subprocess.run(["git", "checkout", "--", "src"], cwd=clone, check=True, capture_output=True)


@pytest.mark.slow
def test_exemption_label_short_circuits_everything(clone):
    branch(clone, "s-exempt")
    bump(clone)
    commit(clone, "exempt")
    assert run_gate(clone, "gate-base", MUTATION_GATE_EXEMPT="mutation-gate-exempt label") == (
        0,
        "SKIP",
    )
