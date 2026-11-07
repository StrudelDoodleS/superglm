"""scripts/check_fix_tests.py: which tests it selects, and what it says about each one."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_fix_tests.py"
_spec = importlib.util.spec_from_file_location("check_fix_tests", SCRIPT)
check_fix_tests = importlib.util.module_from_spec(_spec)
sys.modules["check_fix_tests"] = check_fix_tests  # dataclasses resolve their module by name
_spec.loader.exec_module(check_fix_tests)


def _tests(source: str):
    return check_fix_tests.collect_tests(textwrap.dedent(source))


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
    assert isinstance(double(1), int)


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
    "from tiny import triple\n\n\ndef test_import_fails_on_base():\n    assert triple(2) == 6\n"
)


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


def _repo(tmp_path: Path, head_files: dict[str, str]) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    _git(repo, "init", "-q", "-b", "master")
    _write(repo, {"src/tiny/__init__.py": _BASE_SRC, "tests/test_old.py": _OLD_TEST})
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


def test_a_vacuous_test_is_flagged_and_every_other_outcome_is_told_apart(tmp_path):
    repo, base = _repo(
        tmp_path,
        {
            "src/tiny/__init__.py": _FIXED_SRC,
            "tests/test_old.py": _OLD_TEST.replace("== 1", "== 0"),
            "tests/test_new.py": _NEW_TESTS,
            "tests/test_new_module.py": _NEW_MODULE,
            "tests/_helpers.py": "VALUE = 1\n",
        },
    )
    report = tmp_path / "out.json"
    done = _run(repo, base, "--json", str(report))
    verdicts = {row["test"]: row for row in json.loads(report.read_text())["tests"]}
    status = {name.split("::", 1)[1]: row["status"] for name, row in verdicts.items()}
    flagged = {name.split("::", 1)[1] for name, row in verdicts.items() if row["flagged"]}

    assert done.returncode == 1, done.stdout + done.stderr
    assert flagged == {"test_vacuous_passes_without_the_fix", "test_every_case_passes_without_it"}
    assert status["test_fix_fails_without_it"] == status["TestGroup::test_method_fails_without_it"]
    assert status["test_fix_fails_without_it"] == "fails"
    assert status["test_one_case_needs_the_fix"] == "fails"  # one failing case is enough
    assert status["test_deliberate_pin"] == "passes"
    assert verdicts["tests/test_new.py::test_deliberate_pin"]["pinned"]
    assert status["test_skipped_on_base"] == "skipped"
    assert status["test_needs_a_new_name"] == "fails"
    assert verdicts["tests/test_new.py::test_needs_a_new_name"]["kind"] == "ImportError"
    assert verdicts["tests/test_new_module.py::test_import_fails_on_base"]["kind"] == "ImportError"
    assert status["test_old"] == "fails"  # the changed assertion fails on base's code
    assert "tests/test_new.py::test_vacuous_passes_without_the_fix (added)" in done.stdout
    assert "whatever its reason" in done.stdout  # the caveat travels with the report

    report_only = _run(repo, base, "--report-only")
    assert report_only.returncode == 0 and "FLAGGED: 2 test(s)" in report_only.stdout


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


def test_a_module_that_skips_itself_at_collection_is_reported_and_never_counted(tmp_path):
    repo, base = _repo(
        tmp_path,
        {
            "src/tiny/__init__.py": _FIXED_SRC,
            "tests/test_optional.py": (
                'import pytest\n\npytest.importorskip("a_package_nobody_installs")\n\n\n'
                "def test_needs_it():\n    assert True\n"
            ),
        },
    )
    done = _run(repo, base)

    assert done.returncode == 0, done.stdout + done.stderr
    assert "tests/test_optional.py: skipped at collection on base" in done.stdout
    assert "tests/test_optional.py::test_needs_it (added)" in done.stdout


def test_a_setup_error_is_filed_under_its_exception_and_not_as_an_assertion(tmp_path):
    repo, base = _repo(
        tmp_path,
        {
            "src/tiny/__init__.py": _FIXED_SRC,
            "tests/test_kinds.py": (
                "import pytest\n\n\n@pytest.fixture\ndef triple():\n"
                "    from tiny import triple\n\n    return triple\n\n\n"
                "def test_setup_error(triple):\n    assert triple(1) == 3\n\n\n"
                'def test_custom_message():\n    ok = False\n    assert ok, "coefficients: drifted"\n'
            ),
        },
    )
    report = tmp_path / "out.json"
    _run(repo, base, "--json", str(report))
    kinds = {
        row["test"].split("::")[1]: row["kind"] for row in json.loads(report.read_text())["tests"]
    }

    assert kinds == {"test_setup_error": "ImportError", "test_custom_message": "AssertionError"}


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
