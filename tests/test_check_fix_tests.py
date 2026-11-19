"""scripts/check_fix_tests.py: which tests it selects, and what it says about each one.

The base run takes ``scripts/`` from the base as well, so a test here fails on base only
where the pull request changes the script. Most carry ``pins_behaviour``, since they pin
what the script already did, and a test added here later must say so too; a test that fails
on the script before the change it guards is a regression, and carries none.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_fix_tests.py"
_spec = importlib.util.spec_from_file_location("check_fix_tests", SCRIPT)
check_fix_tests = importlib.util.module_from_spec(_spec)
sys.modules["check_fix_tests"] = check_fix_tests  # dataclasses resolve their module by name
_spec.loader.exec_module(check_fix_tests)


def _tests(source: str):
    return check_fix_tests.collect_tests(textwrap.dedent(source))


@pytest.mark.pins_behaviour
def test_a_test_is_selected_only_when_its_own_source_changes():
    before = _tests(
        '''
        import pytest
        HELPER = 1

        @pytest.mark.parametrize("n", [1, 2])
        def test_cases(n):
            """Old words."""
            assert n  # old comment

        class TestGroup:
            def test_method(self):
                assert HELPER
        '''
    )
    same = _tests(
        '''
        import pytest
        HELPER = 2

        @pytest.mark.parametrize("n", [1, 2])
        def test_cases(n):
            """New words."""
            assert n  # new comment

        class TestGroup:
            def test_method(self):
                assert HELPER
        '''
    )
    changed = _tests(
        """
        import pytest

        @pytest.mark.parametrize("n", [1, 2, 3])
        def test_cases(n):
            assert n

        class TestGroup:
            def test_method(self):
                assert not HELPER

        def test_new():
            pass
        """
    )

    assert before == same  # helper, docstring and comment edits select nothing
    assert set(changed) == {"test_cases", "TestGroup::test_method", "test_new"}
    assert {name for name in before if before[name].digest != changed[name].digest} == set(before)


@pytest.mark.pins_behaviour
def test_the_marker_counts_on_a_test_a_class_or_the_module():
    on_test = _tests("@pytest.mark.pins_behaviour\ndef test_a(): pass\ndef test_b(): pass")
    on_class = _tests("@pytest.mark.pins_behaviour\nclass TestC:\n    def test_a(self): pass")
    on_module = _tests(
        "pytestmark = [pytest.mark.slow, pytest.mark.pins_behaviour]\ndef test_a(): pass"
    )
    other_mark = _tests("pytestmark = pytest.mark.slow\n@pytest.mark.slow\ndef test_a(): pass")

    assert (on_test["test_a"].pinned, on_test["test_b"].pinned) == (True, False)
    assert on_class["TestC::test_a"].pinned and on_module["test_a"].pinned
    assert not other_mark["test_a"].pinned


def test_a_control_is_documented_by_whatever_carries_its_mark():
    """The docstring that counts is the marked test's, class's or module's, never an outer one's."""
    on_test = _tests(
        '@pytest.mark.pins_behaviour\ndef test_a():\n    """Pins x."""\n'
        "@pytest.mark.pins_behaviour\ndef test_b(): pass\n"
        '@pytest.mark.pins_behaviour\ndef test_c():\n    """ \n    """'
    )
    on_class = _tests(
        '@pytest.mark.pins_behaviour\nclass TestC:\n    """Pins y."""\n    def test_a(self): pass'
    )
    on_module = _tests('"""Pins z."""\npytestmark = pytest.mark.pins_behaviour\ndef test_a(): pass')
    inner = _tests(
        '"""Pins z."""\n@pytest.mark.pins_behaviour\nclass TestC:\n    def test_a(self):\n'
        '        """Says nothing of the class."""'
    )

    assert [on_test[name].documented for name in ("test_a", "test_b", "test_c")] == [
        True,
        False,
        False,  # a blank docstring says nothing
    ]
    assert on_class["TestC::test_a"].documented and on_module["test_a"].documented
    assert not inner["TestC::test_a"].documented


@pytest.mark.pins_behaviour
def test_the_marker_is_read_from_the_marks_a_test_has_not_from_what_a_decorator_mentions():
    tests = _tests(
        """
        import pytest

        class TestBody:
            pytestmark = [pytest.mark.slow, pytest.mark.pins_behaviour]
            def test_a(self): pass

        class TestAnnotated:
            pytestmark: pytest.MarkDecorator = pytest.mark.pins_behaviour
            def test_a(self): pass

        class TestOther:
            pytestmark = pytest.mark.slow
            def test_a(self): pass

        @pytest.mark.parametrize("n", [pytest.param(0, marks=pytest.mark.pins_behaviour), 1])
        def test_one_case_marked(n): pass

        @pytest.mark.pins_behaviour("a reason")
        def test_called_marker(): pass
        """
    )
    annotated_module = _tests("pytestmark: list = [pytest.mark.pins_behaviour]\ndef test_a(): pass")

    assert {name: test.pinned for name, test in tests.items()} == {
        "TestBody::test_a": True,
        "TestAnnotated::test_a": True,
        "TestOther::test_a": False,
        "test_one_case_marked": False,  # a mark on one pytest.param pins no test
        "test_called_marker": True,
    }
    assert annotated_module["test_a"].pinned


_BASE_SRC = "def double(x):\n    return x + x + 1  # the bug\n"
_FIXED_SRC = "def double(x):\n    return x + x\n\n\ndef triple(x):\n    return 3 * x\n"
_OLD_TEST = "from tiny import double\n\n\ndef test_old():\n    assert double(0) == 1\n"
_NEW_TESTS = """
import pytest
from tiny import double

def test_fix_fails_without_it():
    assert double(2) == 4


def test_vacuous_passes_without_the_fix():
    assert double(0) in (0, 1)


@pytest.mark.pins_behaviour
def test_deliberate_pin():
    "A control: double returns an int before and after the fix."
    assert isinstance(double(1), int)


@pytest.mark.pins_behaviour
def test_undocumented_pin():
    assert double(1) > 0


@pytest.mark.parametrize("n", [0, 1, 2])
def test_one_case_needs_the_fix(n):
    assert double(n) != 3  # base gives 1, 3, 5: only n = 1 fails


@pytest.mark.parametrize("n", [0, 1])
def test_every_case_passes_without_it(n):
    assert double(n) >= 0


def test_skipped_on_base():
    pytest.skip("no data")


def test_needs_a_new_name():
    from tiny import triple

    assert triple(1) == 3


class TestGroup:
    def test_method_fails_without_it(self):
        assert double(1) == 2
"""
_NEW_MODULE = (
    "from tiny import double, triple\n\n\ndef test_import_fails_on_base():\n"
    "    assert triple(2) == 6\n\n\ndef test_unchanged_beside_it():\n"
    "    assert double(0) in (0, 1)\n"
)
# A tool the pull request adds beside its src/ change, and its test: the tool is absent on base.
_NEW_TOOL = {
    "scripts/tool.py": 'ANSWER = "ok"\n',
    "tests/test_tool.py": (
        "import runpy\n\n\ndef test_the_new_tool_answers():\n"
        '    assert runpy.run_path("scripts/tool.py")["ANSWER"] == "ok"\n'
    ),
}


def _git(repo: Path, *args: str) -> str:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.com",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.com",
    }
    done = subprocess.run(
        ["git", *args], cwd=repo, env=env, capture_output=True, text=True, check=True, timeout=60
    )
    return done.stdout.strip()


def _write(repo: Path, files: dict[str, str]) -> None:
    for name, text in files.items():
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text(text)


def _repo(
    tmp_path: Path, head_files: dict[str, str], base_files: dict[str, str] | None = None
) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    _git(repo, "init", "-q", "-b", "master")
    _write(
        repo,
        {"src/tiny/__init__.py": _BASE_SRC, "tests/test_old.py": _OLD_TEST, **(base_files or {})},
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "base")
    base = _git(repo, "rev-parse", "HEAD")
    _write(repo, head_files)
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "head")
    return repo, base


def _run(repo: Path, base: str, *extra: str, env=None):
    command = [
        sys.executable,
        str(SCRIPT),
        "--root",
        str(repo),
        "--base",
        base,
        "--package",
        "tiny",
    ]
    return subprocess.run(
        [*command, *extra], capture_output=True, text=True, check=False, timeout=240, env=env
    )


def test_the_base_run_names_its_tests_through_a_symlinked_temporary_directory(tmp_path):
    """macOS reaches its temporary directory through a symlink (/var to /private/var). The
    base tree must be named by its real path, or pytest lists its tests without their file
    and the check cannot judge them."""
    real, link = tmp_path / "real", tmp_path / "link"
    real.mkdir()
    try:
        link.symlink_to(real, target_is_directory=True)
    except OSError:
        pytest.skip("this platform does not let the test create a symlink")
    project = '[project]\nname = "tiny"\n'
    repo, base = _repo(
        tmp_path / "work",
        {
            "src/tiny/__init__.py": _FIXED_SRC,
            "tests/test_new.py": _NEW_TESTS,
            # The head's pytest table makes the run name the base tree as its --rootdir.
            "pyproject.toml": project
            + '\n[tool.pytest.ini_options]\nmarkers = ["pins_behaviour: a control"]\n',
        },
        base_files={"pyproject.toml": project},
    )
    report = tmp_path / "out.json"
    done = _run(repo, base, "--json", str(report), env={**os.environ, "TMPDIR": str(link)})

    assert done.returncode == 1, done.stdout + done.stderr
    names = [row["test"] for row in json.loads(report.read_text())["tests"]]
    assert names and all(name.startswith("tests/test_new.py::") for name in names)


@pytest.mark.pins_behaviour
def test_a_vacuous_test_is_flagged_and_every_other_outcome_is_told_apart(tmp_path):
    repo, base = _repo(
        tmp_path,
        {
            "src/tiny/__init__.py": _FIXED_SRC,
            "tests/test_old.py": _OLD_TEST.replace("== 1", "== 0"),
            "tests/test_new.py": _NEW_TESTS,
            "tests/test_new_module.py": _NEW_MODULE,
            "tests/_helpers.py": "VALUE = 1\n",
            **_NEW_TOOL,
        },
    )
    report = tmp_path / "out.json"
    done = _run(repo, base, "--json", str(report))
    verdicts = {row["test"]: row for row in json.loads(report.read_text())["tests"]}
    status = {name.split("::", 1)[1]: row["status"] for name, row in verdicts.items()}
    flagged = {name.split("::", 1)[1] for name, row in verdicts.items() if row["flagged"]}

    assert done.returncode == 1, done.stdout + done.stderr
    assert flagged == {
        "test_vacuous_passes_without_the_fix",
        "test_every_case_passes_without_it",
        "test_undocumented_pin",  # marked, but says nothing of what it pins
    }
    assert status["test_fix_fails_without_it"] == status["TestGroup::test_method_fails_without_it"]
    assert status["test_fix_fails_without_it"] == "fails"
    assert status["test_one_case_needs_the_fix"] == "fails"  # one failing case is enough
    assert status["test_deliberate_pin"] == "passes"
    assert verdicts["tests/test_new.py::test_deliberate_pin"]["pinned"]
    assert status["test_skipped_on_base"] == "skipped"
    assert status["test_needs_a_new_name"] == "fails"
    assert verdicts["tests/test_new.py::test_needs_a_new_name"]["kind"] == "ImportError"
    # a module that lacks a name on base fails there, each of its tests, counted by module
    assert status["test_import_fails_on_base"] == status["test_unchanged_beside_it"] == "fails"
    assert verdicts["tests/test_new_module.py::test_unchanged_beside_it"]["kind"] == "ImportError"
    assert "tests/test_new_module.py: ImportError x 2" in done.stdout
    assert status["test_the_new_tool_answers"] == "fails"  # the tool is the base's: absent
    assert status["test_old"] == "fails"  # the changed assertion fails on base's code
    assert "tests/test_new.py::test_vacuous_passes_without_the_fix (added)" in done.stdout
    assert "control (@pytest.mark.pins_behaviour): tests/test_new.py::test_deliberate_pin" in (
        done.stdout
    )
    assert "whatever its reason" in done.stdout  # the caveat travels with the report

    report_only = _run(repo, base, "--report-only")
    assert report_only.returncode == 0 and "FLAGGED: 3 test(s)" in report_only.stdout


@pytest.mark.pins_behaviour
def test_a_pull_request_with_nothing_to_check_passes(tmp_path):
    no_tests, base = _repo(tmp_path / "a", {"src/tiny/__init__.py": _FIXED_SRC})
    no_src, base_two = _repo(tmp_path / "b", {"tests/test_new.py": _NEW_TESTS})
    unchanged, base_three = _repo(
        tmp_path / "c",
        {"src/tiny/__init__.py": _FIXED_SRC, "tests/test_old.py": _OLD_TEST + "\n# a comment\n"},
    )

    assert "no test added or changed" in _run(no_tests, base).stdout
    assert "no src/ change" in _run(no_src, base_two).stdout
    assert "no test added or changed" in _run(unchanged, base_three).stdout
    assert all(_run(r, b).returncode == 0 for r, b in [(no_tests, base), (no_src, base_two)])
    assert _run(unchanged, base_three).returncode == 0


@pytest.mark.pins_behaviour
def test_the_base_run_imports_the_merge_bases_src_and_refuses_any_other(tmp_path):
    repo, base = _repo(
        tmp_path,
        {
            "src/tiny/__init__.py": _FIXED_SRC,
            "tests/test_where.py": "from tiny import double\n\n\ndef test_x():\n    assert double(2) == 5\n",
        },
    )
    report = tmp_path / "out.json"
    done = _run(repo, base, "--json", str(report))
    refused = _run(repo, base, "--package", "json")  # a package that is not under src/

    # double(2) is 5 only on base's src/, so a pass means the base tree's src/ was imported
    assert json.loads(report.read_text())["tests"][0]["status"] == "passes"
    assert done.returncode == 1
    assert refused.returncode == 2 and "resolves to" in refused.stderr


@pytest.mark.pins_behaviour
def test_the_repositorys_own_pytest_options_do_not_change_what_the_check_reads(tmp_path):
    """``addopts = -v`` turns ``--collect-only -q`` into a tree of modules, not one id per line."""
    repo, base = _repo(
        tmp_path,
        {
            "src/tiny/__init__.py": _FIXED_SRC,
            "pytest.ini": "[pytest]\naddopts = -v --tb=short\n",
            "tests/test_fresh.py": "from tiny import double\n\n\ndef test_x():\n    assert double(2) == 4\n",
        },
    )
    done = _run(repo, base, "--json", str(tmp_path / "out.json"))

    assert json.loads((tmp_path / "out.json").read_text())["tests"][0]["status"] == "fails"
    assert done.returncode == 0


@pytest.mark.pins_behaviour
def test_a_module_that_skips_itself_at_collection_is_reported_and_never_counted(tmp_path):
    repo, base = _repo(
        tmp_path,
        {
            "src/tiny/__init__.py": _FIXED_SRC,
            "tests/test_optional.py": (
                'import pytest\n\npytest.importorskip("a_package_nobody_installs")\n\n\n'
                "def test_needs_it():\n    assert True\n"
            ),
            "tests/test_fresh.py": "from tiny import double\n\n\ndef test_x():\n    assert double(2) == 4\n",
        },
    )
    done = _run(repo, base)

    assert done.returncode == 0, done.stdout + done.stderr
    assert "tests/test_optional.py: skipped at collection on base" in done.stdout
    assert "tests/test_optional.py::test_needs_it (added)" in done.stdout


# Collects on the fixed code only, failing on base for a reason that is not a missing name.
_KEYED_MODULE = (
    "from tiny import double\n\nCASE = {0: 'fixed'}[double(0)]  # double(0) is 1 on base\n\n\n"
    "def test_reads_its_case():\n    assert CASE == 'fixed'\n"
)


def test_a_test_the_base_cannot_judge_fails_the_check(tmp_path):
    """A module that fails to collect on base for a reason other than a missing name leaves its
    tests undecided, and an undecided test fails the check, listed. A test the run does not
    judge by design (``threads``, a skip for want of data) is listed as not run, and a pull
    request whose only new test is one of those passes."""
    undecided, base = _repo(
        tmp_path / "a",
        {
            "src/tiny/__init__.py": _FIXED_SRC,
            "tests/test_fresh.py": "from tiny import double\n\n\ndef test_x():\n    assert double(2) == 4\n",
            "tests/test_keyed.py": _KEYED_MODULE,
        },
    )
    by_design = {
        "skipped": "import pytest\n\n\ndef test_needs_data():\n    pytest.skip('[dataset] no data')\n",
        "not_run": (
            "import pytest\nfrom tiny import double\n\n\n@pytest.mark.threads\n"
            "def test_needs_the_pools():\n    assert double(2) == 4\n"
        ),
    }
    not_judged = {
        status: _repo(
            tmp_path / status,
            {
                "src/tiny/__init__.py": _FIXED_SRC,
                "pytest.ini": "[pytest]\nmarkers = threads: needs the default pools\n",
                "tests/test_later.py": test,
            },
        )
        for status, test in by_design.items()
    }
    report = tmp_path / "out.json"
    done = _run(undecided, base, "--json", str(report))
    status = {row["test"]: row["status"] for row in json.loads(report.read_text())["tests"]}
    judged_nothing = {
        status: _run(*repo, "--json", str(tmp_path / f"{status}.json"))
        for status, repo in not_judged.items()
    }

    assert done.returncode == 1, done.stdout + done.stderr
    assert status == {
        "tests/test_fresh.py::test_x": "fails",
        "tests/test_keyed.py::test_reads_its_case": "collection_error",
    }
    assert "UNDECIDED: 1 test(s)" in done.stdout and "FLAGGED" not in done.stdout
    assert "tests/test_keyed.py::test_reads_its_case (added): KeyError" in done.stdout
    assert _run(undecided, base, "--report-only").returncode == 0
    for status, run in judged_nothing.items():
        rows = json.loads((tmp_path / f"{status}.json").read_text())["tests"]
        assert run.returncode == 0, run.stdout + run.stderr
        assert [row["status"] for row in rows] == [status]
        assert "no test was judged on base" in run.stdout and "UNDECIDED" not in run.stdout


def test_the_base_run_reads_the_bases_project_metadata_and_the_heads_pytest_options(tmp_path):
    """A test of project metadata meets the base's ``pyproject.toml``, while the head's
    ``[tool.pytest.ini_options]`` still reaches the run (here a filter that makes a warning an
    error, and a relative ``pythonpath`` that resolves inside the base tree), so both floor tests
    fail on base and the helper's control passes there."""
    project = '[project]\nname = "tiny"\nrequires-python = ">=3.12"\n'
    repo, base = _repo(
        tmp_path,
        {
            "src/tiny/__init__.py": _FIXED_SRC,
            "pyproject.toml": project.replace("3.12", "3.13")
            + "\n[tool.pytest.ini_options]\n"
            + 'filterwarnings = ["error::UserWarning"]\npythonpath = ["tests/support"]\n'
            + 'markers = ["pins_behaviour: a control"]\n',
            "tests/support/fixture_helper.py": "ONE = 1\n",
            "tests/test_helper.py": (
                "import pytest\nfrom fixture_helper import ONE\n\n\n@pytest.mark.pins_behaviour\n"
                "def test_reads_a_helper_on_the_pythonpath():\n"
                '    "A control: the head\'s relative pythonpath resolves in the base tree."\n'
                "    assert ONE == 1\n"
            ),
            "tests/test_floor.py": (
                "import tomllib\nimport warnings\nfrom pathlib import Path\n\n\n"
                "def test_the_floor_is_raised():\n"
                '    project = tomllib.loads(Path("pyproject.toml").read_text())["project"]\n'
                '    assert project["requires-python"] == ">=3.13"\n\n\n'
                "def test_a_warning_is_an_error():\n"
                '    warnings.warn("an error under the head\'s options", UserWarning)\n'
            ),
        },
        base_files={"pyproject.toml": project},
    )
    report = tmp_path / "out.json"
    done = _run(repo, base, "--json", str(report))
    status = {row["test"]: row["status"] for row in json.loads(report.read_text())["tests"]}

    assert done.returncode == 0, done.stdout + done.stderr
    assert status == {
        "tests/test_floor.py::test_the_floor_is_raised": "fails",
        "tests/test_floor.py::test_a_warning_is_an_error": "fails",
        "tests/test_helper.py::test_reads_a_helper_on_the_pythonpath": "passes",
    }


@pytest.mark.pins_behaviour
def test_a_setup_error_is_filed_under_its_exception_and_not_as_an_assertion(tmp_path):
    repo, base = _repo(
        tmp_path,
        {
            "src/tiny/__init__.py": _FIXED_SRC,
            "tests/test_kinds.py": (
                "import dataclasses\n\nimport pytest\n\n\n@pytest.fixture\ndef triple():\n"
                "    from tiny import triple\n\n    return triple\n\n\n"
                "def test_setup_error(triple):\n    assert triple(1) == 3\n\n\n"
                'def test_custom_message():\n    ok = False\n    assert ok, "coefficients: drifted"\n\n\n'
                'def test_multiline_message():\n    raise ValueError("no fit\\nreason: singular")\n\n\n'
                "def test_approx():\n    assert 1.0 == pytest.approx(2.0)\n\n\n"
                "@dataclasses.dataclass\nclass Pair:\n    a: int\n\n\n"
                "def test_dataclass():\n    assert Pair(1) == Pair(2)\n"
            ),
        },
    )
    report = tmp_path / "out.json"
    _run(repo, base, "--json", str(report))
    kinds = {
        row["test"].split("::")[1]: row["kind"] for row in json.loads(report.read_text())["tests"]
    }

    assert kinds == {
        "test_setup_error": "ImportError",
        "test_custom_message": "AssertionError",
        "test_multiline_message": "ValueError",  # not "reason", the tail of the message
        "test_approx": "AssertionError",  # not "Obtained", a line of the explanation
        "test_dataclass": "AssertionError",  # not "a", a line of the drill-down
    }


@pytest.mark.pins_behaviour
def test_threads_tests_are_not_run_and_the_pools_are_pinned(tmp_path):
    pools = "OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS NUMBA_NUM_THREADS".split()
    pools += "VECLIB_MAXIMUM_THREADS BLIS_NUM_THREADS NUMEXPR_NUM_THREADS SUPERGLM_N_JOBS".split()
    repo, base = _repo(
        tmp_path,
        {
            "src/tiny/__init__.py": _FIXED_SRC,
            "pytest.ini": "[pytest]\nmarkers = threads: needs the default pools\n",
            "tests/test_pools.py": (
                "import os\n\nimport pytest\n\n\n@pytest.mark.threads\n"
                'def test_needs_the_default_pools():\n    assert os.environ["OMP_NUM_THREADS"] != "1"\n\n\n'
                f"@pytest.mark.pins_behaviour\ndef test_every_pool_is_pinned():\n"
                '    "A control: the base run pins every pool."\n'
                f"    assert [os.environ.get(name) for name in {pools!r}] == ['1'] * {len(pools)}\n"
            ),
        },
    )
    report = tmp_path / "out.json"
    unpinned = {
        **os.environ,
        "SUPERGLM_N_JOBS": "8",
        "BLIS_NUM_THREADS": "8",
        "OMP_NUM_THREADS": "8",
    }
    done = _run(repo, base, "--json", str(report), env=unpinned)
    status = {
        row["test"].split("::")[1]: row["status"] for row in json.loads(report.read_text())["tests"]
    }

    assert done.returncode == 0, done.stdout + done.stderr
    assert status == {
        "test_needs_the_default_pools": "not_run",
        "test_every_pool_is_pinned": "passes",
    }


@pytest.mark.pins_behaviour
def test_a_collection_that_fails_on_base_says_why_on_stderr(tmp_path):
    repo, base = _repo(
        tmp_path,
        {
            "src/tiny/__init__.py": _FIXED_SRC,
            "tests/conftest.py": "from tiny import triple  # absent on base\n",
            "tests/test_new.py": "def test_x():\n    assert True\n",
        },
    )
    done = _run(repo, base)

    assert done.returncode == 2
    assert (
        "collection on base exited" in done.stderr and "cannot import name 'triple'" in done.stderr
    )


@pytest.mark.pins_behaviour
@pytest.mark.parametrize("separator", ["/", "\\"], ids=["posix", "windows"])
def test_a_skip_at_collection_is_matched_to_gits_path_whichever_slash_pytest_writes(
    monkeypatch, separator
):
    """Pytest prints the location of a skip as a path, so a Windows run writes ``tests\\sub\\x.py``."""
    location = separator.join(["tests", "sub", "test_optional.py"])
    output = f"SKIPPED [1] {location}:3: could not import 'nothing': No module named 'nothing'\n"
    monkeypatch.setattr(
        check_fix_tests,
        "pytest_run",
        lambda *args, **kwargs: subprocess.CompletedProcess([], 0, output, ""),
    )

    ids, broken, skipped = check_fix_tests.collect_on_base(
        Path("."), {}, ["tests/sub/test_optional.py"]
    )

    assert (ids, broken) == (set(), {})
    assert skipped == {
        "tests/sub/test_optional.py": "could not import 'nothing': No module named 'nothing'"
    }


@pytest.mark.pins_behaviour
def test_a_collection_that_outlives_its_limit_is_a_check_that_could_not_run(monkeypatch):
    timed_out = subprocess.CompletedProcess([], check_fix_tests.TIMED_OUT, "", "")
    monkeypatch.setattr(check_fix_tests, "pytest_run", lambda *args, **kwargs: timed_out)

    with pytest.raises(check_fix_tests.CheckError, match="collection on base timed out"):
        check_fix_tests.collect_on_base(Path("."), {}, ["tests/test_x.py"])


@pytest.mark.pins_behaviour
def test_the_exception_is_named_by_the_header_of_the_last_block_of_error_lines():
    head = check_fix_tests._error_head
    context = 'E     File "m.py", line 1\nE       x = (\nE           ^\nE   SyntaxError: invalid'

    assert head("E   ValueError: no fit\nE   reason: singular") == "ValueError: no fit"
    assert (
        head("E   KeyError: 'a'\n\nDuring handling\n> raise\nE   ValueError: b") == "ValueError: b"
    )
    assert head(context) == "SyntaxError: invalid"  # its source lines come first
    assert head("E   assert 1 == 2\nE    +  where 1 = f()") == "assert 1 == 2"
    approx = "E   assert 1.0 == 2.0 ± 2e-06\nE     \nE     comparison failed\nE     Obtained: 1.0"
    assert head(approx) == "assert 1.0 == 2.0 ± 2e-06"  # an explanation can hold "name: value"
    assert head("E   assert_failed: a\nE   b: 1") == "assert_failed: a"  # not an assert
    assert head("E   AssertionError: drifted\nE   assert False") == "AssertionError: drifted"
    assert (
        head("E   ValueError: a\nE\nE   tail: b") == "ValueError: a"
    )  # a blank line of the message
    assert head("nothing marked") == ""


def _is_running(pid: int) -> bool:
    if sys.platform == "win32":
        listing = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True, check=False
        ).stdout
        return str(pid) in listing.split()
    state = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True, check=False
    ).stdout.strip()
    return bool(state) and not state.startswith("Z")  # a zombie has already died


_HANGS = """
import os
import pathlib
import subprocess
import sys
import time


def test_never_finishes():
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
    pathlib.Path(os.environ["FIX_TESTS_PIDFILE"]).write_text(str(sleeper.pid))
    time.sleep(600)
"""


@pytest.mark.pins_behaviour
def test_a_run_that_outlives_its_timeout_is_reported_and_leaves_no_process_behind(tmp_path):
    """The test sleeps and starts a process of its own that sleeps too: both hold the run's
    output, and both must be gone when the check returns."""
    repo, base = _repo(
        tmp_path, {"src/tiny/__init__.py": _FIXED_SRC, "tests/test_hangs.py": _HANGS}
    )
    pidfile, report = tmp_path / "sleeper.pid", tmp_path / "out.json"
    environment = {**os.environ, "FIX_TESTS_PIDFILE": str(pidfile)}

    done = _run(repo, base, "--timeout", "10", "--json", str(report), env=environment)
    verdict = json.loads(report.read_text())["tests"][0]
    sleeper = int(pidfile.read_text())
    for _ in range(50):  # a killed process takes a moment to leave the process table
        if not _is_running(sleeper):
            break
        time.sleep(0.1)

    assert done.returncode == 2, done.stdout + done.stderr
    assert "the run on base timed out after 10 s" in done.stdout  # the report is printed first
    assert "tests/test_hangs.py::test_never_finishes (added)" in done.stdout
    assert "fix-test check could not run: the run on base timed out" in done.stderr
    assert verdict["status"] == "timed_out" and not verdict["flagged"]
    assert not _is_running(sleeper)


@pytest.mark.pins_behaviour
def test_the_scripts_own_limits_end_inside_the_workflows_timeout_wrapper():
    workflow = yaml.safe_load((ROOT / ".github/workflows/dev-ci.yml").read_text(encoding="utf-8"))
    steps = workflow["jobs"]["fix-tests"]["steps"]
    run = next(step["run"] for step in steps if "check_fix_tests" in step.get("run", ""))
    wrapper = 60 * int(re.search(r"timeout --kill-after=\d+s (\d+)m", run).group(1))

    assert "--timeout" not in run  # the script's default applies
    assert check_fix_tests.COLLECT_TIMEOUT + check_fix_tests.RUN_TIMEOUT < wrapper
