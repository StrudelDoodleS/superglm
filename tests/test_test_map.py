"""scripts/test_map.py: what a recorded run maps, and which test files a change selects."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("test_map", ROOT / "scripts" / "test_map.py")
test_map = importlib.util.module_from_spec(_spec)
sys.modules["test_map"] = test_map
_spec.loader.exec_module(test_map)

KERNEL = """from numba import njit


@njit(cache=True)
def kernel(x):
    return x + 1
"""
CALLER = """from superglm.kernels import kernel


def call(x):
    return kernel(x)
"""
CONSTANTS = """LIMIT = 3


def limit():
    return LIMIT


CAP = limit()
"""
USER = """from superglm.constants import CAP


def use():
    return CAP
"""
# A package that re-exports lazily, as superglm's root does.
PACKAGE = """_LAZY_EXPORTS = {"superglm.caller": ("call",)}
"""


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def _repo(root: Path) -> Path:
    files = {
        "pyproject.toml": "[project]\nname = 'x'\n",
        "src/superglm/__init__.py": PACKAGE,
        "src/superglm/kernels.py": KERNEL,
        "src/superglm/caller.py": CALLER,
        "src/superglm/constants.py": CONSTANTS,
        "src/superglm/user.py": USER,
        "tests/__init__.py": "",
        "tests/test_caller.py": "from superglm import call\n\n\ndef test_call():\n    call(1)\n",
        "tests/test_user.py": "def test_use():\n    from superglm.user import use\n\n    use()\n",
        "tests/test_other.py": "def test_nothing():\n    pass\n",
        "tests/test_reader.py": "def test_read():\n    pass\n",
        "tests/fixtures/table.json": "{}\n",
        "tests/fixtures/unread.json": "{}\n",
        "tests/test_names_it.py": 'NAME = "unread.json"\n\n\ndef test_it():\n    pass\n',
    }
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text, encoding="utf-8")
    _git(root, "init", "-q", "-b", "master")
    _git(root, "config", "user.email", "test@example.invalid")
    _git(root, "config", "user.name", "test")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "base")
    return root


TESTS = {
    "tests/test_caller.py": [1, 1.0],
    "tests/test_user.py": [1, 1.0],
    "tests/test_other.py": [1, 10.0],
    "tests/test_reader.py": [1, 1.0],
    "tests/test_names_it.py": [1, 1.0],
}
# Who ran what: numba bodies never show, the Python caller of a kernel does.
RAN = {
    "src/superglm/caller.py": {"call": ["tests/test_caller.py"]},
    "src/superglm/user.py": {"use": ["tests/test_user.py"]},
    "src/superglm/constants.py": {"limit": ["tests/test_reader.py"]},
}


def _map(root: Path, ran: dict | None = None, **overrides) -> dict:
    """A map as ``merge`` builds it."""
    files = sorted(TESTS)
    functions = {
        path: {name: [files.index(test) for test in tests] for name, tests in names.items()}
        for path, names in (RAN if ran is None else ran).items()
    }
    data = {
        "version": test_map.VERSION,
        "commit": _git(root, "rev-parse", "HEAD"),
        "tests": TESTS,
        "files": files,
        "functions": functions,
        "opened": {"tests/fixtures/table.json": ["tests/test_reader.py"]},
        "listed": {},
        "session": {},
        "import_time": {},
        "untracked": [],
    }
    return {**data, **overrides}


def _change(root: Path, path: str, text: str) -> list[tuple[str, str]]:
    (root / path).write_text(text, encoding="utf-8")
    return test_map.changes(_map(root)["commit"], root)


def _selected(selection) -> set[str]:
    return set(selection.files) - set(test_map.ALWAYS)


def test_a_numba_kernel_change_selects_the_tests_that_reach_its_python_caller(tmp_path):
    root = _repo(tmp_path)
    data = _map(root)
    changed = _change(root, "src/superglm/kernels.py", KERNEL.replace("x + 1", "x + 2"))
    selection = test_map.select(data, changed, root)
    assert selection.full is None
    # The map holds nothing under kernels.py: only the numba rule reaches test_caller.
    assert _selected(selection) == {"tests/test_caller.py"}
    assert "kernels.py" in selection.files["tests/test_caller.py"]


def test_without_the_numba_rule_a_kernel_change_selects_nothing(tmp_path, monkeypatch):
    """The mutation: a selector blind to compiled bodies misses the kernel's tests."""
    root = _repo(tmp_path)
    monkeypatch.setattr(test_map.Modules, "compiles", lambda self, path: False)
    changed = _change(root, "src/superglm/kernels.py", KERNEL.replace("x + 1", "x + 2"))
    assert _selected(test_map.select(_map(root), changed, root)) == set()


def test_a_body_change_selects_who_ran_it_and_a_constant_follows_its_name(tmp_path):
    root = _repo(tmp_path)
    data = _map(root)
    # Only test_reader ran limit(); user.py reads CAP, which limit() computes at import.
    body = _change(root, "src/superglm/constants.py", CONSTANTS.replace("return LIMIT", "return 4"))
    assert _selected(test_map.select(data, body, root)) == {
        "tests/test_reader.py",
        "tests/test_user.py",
    }
    # LIMIT feeds limit(), which feeds CAP at import, which user.py imports.
    constant = _change(root, "src/superglm/constants.py", CONSTANTS.replace("= 3", "= 4"))
    assert _selected(test_map.select(data, constant, root)) == {
        "tests/test_reader.py",
        "tests/test_user.py",
    }


def test_a_function_nothing_calls_at_import_selects_only_who_ran_it(tmp_path):
    root = _repo(tmp_path)
    text = CONSTANTS.replace("CAP = limit()", "CAP = 3")
    (root / "src/superglm/constants.py").write_text(text, encoding="utf-8")
    _git(root, "commit", "-qam", "no import-time call")
    changed = _change(root, "src/superglm/constants.py", text.replace("return LIMIT", "return 4"))
    assert _selected(test_map.select(_map(root), changed, root)) == {"tests/test_reader.py"}


def _commit_module(root: Path, files: dict[str, str]) -> None:
    for name, text in files.items():
        (root / name).write_text(text, encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "more modules")


LINKS = "class Link:\n    domain = None\n\n    def inverse(self, eta):\n        return eta\n"
READER = "def bounds(link):\n    return getattr(link, 'domain', None)\n"
REGISTRY = "HANDLERS = {}\n\n\ndef register(name, handler):\n    HANDLERS[name] = handler\n"
PLUGIN = "from superglm.registry import register\n\n\ndef handle():\n    return 1\n\n\nregister('a', handle)\n"


def test_a_class_attribute_reaches_whoever_reads_it_by_name_anywhere(tmp_path):
    root = _repo(tmp_path)
    _commit_module(root, {"src/superglm/link.py": LINKS, "src/superglm/reader.py": READER})
    ran = {
        "src/superglm/reader.py": {"bounds": ["tests/test_other.py"]},
        "src/superglm/link.py": {"Link.inverse": ["tests/test_reader.py"]},
    }
    changed = _change(root, "src/superglm/link.py", LINKS.replace("None", "(0, 1)"))
    # reader.py never imports Link: it reads the attribute through getattr, by its name.
    selected = _selected(test_map.select(_map(root, ran=ran), changed, root))
    assert selected == {"tests/test_other.py"}


def test_a_changed_registration_selects_the_registry_and_the_module(tmp_path):
    root = _repo(tmp_path)
    _commit_module(root, {"src/superglm/registry.py": REGISTRY, "src/superglm/plugin.py": PLUGIN})
    ran = {
        "src/superglm/registry.py": {"register": ["tests/test_user.py"]},
        "src/superglm/plugin.py": {"handle": ["tests/test_reader.py"]},
    }
    changed = _change(root, "src/superglm/plugin.py", PLUGIN.replace("'a'", "'b'"))
    selected = _selected(test_map.select(_map(root, ran=ran), changed, root))
    assert selected == {"tests/test_user.py", "tests/test_reader.py"}


def test_a_comment_only_change_selects_nothing(tmp_path):
    root = _repo(tmp_path)
    changed = _change(root, "src/superglm/user.py", USER + "# a note\n")
    assert _selected(test_map.select(_map(root), changed, root)) == set()


def test_a_changed_name_passes_through_a_lazy_package_reexport(tmp_path):
    root = _repo(tmp_path)
    # call() becomes a constant: test_caller imports it from the package root, which re-exports it
    # lazily, and nothing recorded ran the new value.
    changed = _change(root, "src/superglm/caller.py", CALLER + "\n\ncall = 3\n")
    assert _selected(test_map.select(_map(root, ran={}), changed, root)) == {"tests/test_caller.py"}


@pytest.mark.parametrize("path", ["pyproject.toml", "tests/conftest.py"])
def test_a_change_to_what_configures_every_test_runs_the_whole_suite(tmp_path, path):
    root = _repo(tmp_path)
    (root / path).write_text("# changed\n", encoding="utf-8")
    _git(root, "add", path)
    selection = test_map.select(_map(root), test_map.changes(_map(root)["commit"], root), root)
    assert selection.full is not None and path in selection.full


def test_data_files_select_their_readers_or_the_files_that_name_them(tmp_path):
    root = _repo(tmp_path)
    read = _change(root, "tests/fixtures/table.json", '{"a": 1}\n')
    assert _selected(test_map.select(_map(root), read, root)) == {"tests/test_reader.py"}
    unread = _change(root, "tests/fixtures/unread.json", '{"a": 1}\n')
    assert _selected(test_map.select(_map(root), unread, root)) >= {"tests/test_names_it.py"}


def test_a_selection_that_is_most_of_the_suite_runs_the_whole_suite(tmp_path):
    root = _repo(tmp_path)
    data = _map(root, ran={"src/superglm/user.py": {"use": ["tests/test_other.py"]}})
    changed = _change(root, "src/superglm/user.py", USER.replace("return CAP", "return 0"))
    selection = test_map.select(data, changed, root)
    assert selection.full is not None and "recorded seconds" in selection.full


def test_a_deleted_test_file_leaves_the_selection_but_whoever_ran_its_code_stays(tmp_path):
    root = _repo(tmp_path)
    ran = {"tests/test_caller.py": {"test_call": ["tests/test_caller.py", "tests/test_reader.py"]}}
    (root / "tests/test_caller.py").unlink()
    changed = test_map.changes(_map(root)["commit"], root)
    # pytest refuses a path that is not there, so the deleted file is never handed to it.
    assert _selected(test_map.select(_map(root, ran=ran), changed, root)) == {
        "tests/test_reader.py"
    }


INHERITS = "class Base:\n    def value(self):\n        return 1\n\n\nclass Child(Base):\n    pass\n"
CHILD_TEST = (
    "from superglm.shapes import Child\n\n\ndef test_value():\n    assert Child().value() == 1\n"
)


def test_an_added_override_reaches_who_used_the_class_through_its_inherited_method(tmp_path):
    root = _repo(tmp_path)
    _commit_module(root, {"src/superglm/shapes.py": INHERITS, "tests/test_user.py": CHILD_TEST})
    # The map records Base.value: the test ran the inherited method, nothing of Child's own.
    ran = {"src/superglm/shapes.py": {"Base.value": ["tests/test_user.py"]}}
    override = INHERITS.replace("    pass\n", "    def value(self):\n        return 200\n")
    changed = _change(root, "src/superglm/shapes.py", override)
    assert _selected(test_map.select(_map(root, ran=ran), changed, root)) == {"tests/test_user.py"}


def test_merge_folds_a_shared_fixtures_reach_into_each_test_file_that_uses_it(tmp_path):
    record = {
        "reached": {
            "tests/test_a.py": {"src/a.py": ["f"]},
            "fixture:tests::model": {"src/fit.py": ["fit"]},
        },
        "opened": {},
        "listed": {},
        "fixtures": {"tests/test_a.py": [], "tests/test_b.py": ["fixture:tests::model"]},
        "import_time": {},
        "tests": {"tests/test_a.py": [2, 0.5], "tests/test_b.py": [1, 0.25]},
        "untracked": [],
    }
    (tmp_path / "1.json").write_text(json.dumps(record), encoding="utf-8")
    (tmp_path / "2.json").write_text(json.dumps(record), encoding="utf-8")
    data = test_map.merge(tmp_path)
    assert data["files"] == ["tests/test_a.py", "tests/test_b.py"]
    assert data["functions"] == {"src/a.py": {"f": [0]}, "src/fit.py": {"fit": [1]}}
    assert data["tests"]["tests/test_a.py"] == [4, 1.0]


RECORDED_MODULE = """def helper():
    return 1


def in_child():
    return 2


def dynamic():
    return 3
"""
RECORDED_TESTS = """import os
import subprocess
import sys

import pytest

from superglm import recorded


@pytest.fixture(scope="session")
def shared():
    return recorded.helper()


@pytest.fixture(scope="module")
def lazily():
    return recorded.dynamic()


def test_uses_the_fixture(shared):
    assert shared == 1


def test_asks_for_a_fixture_while_it_runs(request):
    assert request.getfixturevalue("lazily") == 3


def test_runs_a_child():
    code = "from superglm import recorded; recorded.in_child()"
    subprocess.run([sys.executable, "-c", code], check=True, env=dict(os.environ))
"""


def test_a_recorded_run_maps_fixtures_and_subprocesses_to_their_test_file(tmp_path):
    """The recorder end to end: a session fixture's code and a child process's code both map."""
    root = tmp_path / "project"
    (root / "src" / "superglm").mkdir(parents=True)
    (root / "src" / "superglm" / "__init__.py").write_text("", encoding="utf-8")
    (root / "src" / "superglm" / "recorded.py").write_text(RECORDED_MODULE, encoding="utf-8")
    (root / "pyproject.toml").write_text("[tool.pytest.ini_options]\n", encoding="utf-8")
    (root / "tests").mkdir()
    (root / "tests" / "test_recorded.py").write_text(RECORDED_TESTS, encoding="utf-8")
    out = tmp_path / "out"
    variables = test_map.recording_environment(out, root)
    variables["PYTHONPATH"] = os.pathsep.join([variables["PYTHONPATH"], str(root / "src")])
    run = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "no:randomly"]
        + [*test_map.RECORDING_ARGUMENTS, "tests"],
        cwd=root,
        env={**os.environ, **variables},
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stdout[-2000:]
    data = test_map.merge(out)
    # A fixture asked for by name while a test runs counts like one the test lists.
    functions = {"helper": [0], "in_child": [0], "dynamic": [0]}
    assert data["functions"]["src/superglm/recorded.py"] == functions
    assert data["tests"]["tests/test_recorded.py"][0] == 3
    # Each function shows under its own context: the fixture's and the child's.
    records = [json.loads(path.read_text(encoding="utf-8")) for path in out.glob("*.json")]
    contexts = {
        context
        for record in records
        for context, files in record["reached"].items()
        if "src/superglm/recorded.py" in files
    }
    fixtures = {f"fixture:tests/test_recorded.py::{name}" for name in ("shared", "lazily")}
    assert contexts == {"tests/test_recorded.py", *fixtures}


def test_build_records_no_map_when_the_checkout_moved_during_its_run(tmp_path, monkeypatch):
    """A tracked file edited while ``build`` runs the suite leaves no map under the old commit.

    Mutation: ``build`` checked the tree only before the run, and saved the map anyway.
    """
    root = _repo(tmp_path)
    monkeypatch.setattr(test_map, "ROOT", root)
    commit = _git(root, "rev-parse", "HEAD")

    def suite(*args, **kwargs):
        (root / "tests" / "test_other.py").write_text("x = 2\n", encoding="utf-8")
        return 0

    monkeypatch.setattr(test_map.subprocess, "call", suite)
    monkeypatch.setattr(test_map, "merge", lambda out: {"tests": {}})
    with pytest.raises(SystemExit, match="the checkout moved during the run"):
        test_map.main(["build"])
    assert not (test_map.map_directory(root) / f"{commit}.json").exists()
