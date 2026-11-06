"""Check that the tests a pull request adds or changes fail on the code before the change.

    uv run python scripts/check_fix_tests.py --base origin/master
    uv run python scripts/check_fix_tests.py --base origin/master --report-only --json out.json -- -n 4

This is the fail-to-pass criterion of SWT-Bench (arXiv:2406.12952): a test that pins
a fix fails on the unfixed code and passes on the fixed code. The script takes the
head commit's tree with ``src/`` replaced by the merge base's, runs only the tests
the pull request added or changed, and judges each one. It compares committed
revisions; uncommitted changes are not seen. Each selected test is judged:

* fails or errors on base: counts as a pass for this check. The check cannot tell
  the right reason from the wrong one, so the output counts the failures by
  exception type; an ``ImportError`` or ``AttributeError`` says the test fails
  for want of a name, not for the behaviour it asserts.
* passes on base: the test does not need the change. It is flagged unless it
  carries ``@pytest.mark.pins_behaviour`` on the test, its class or the module
  (``pytestmark``), which says "this deliberately pins behaviour that already worked".
* skipped on base, deselected (``docs``) or not collected: reported, never counted
  as failing.

A parametrized test counts as passing on base only if every case that ran passes.

Limits. A test is selected when it is new or when its own decorators, signature
or body changed (found by comparing ``ast`` dumps of the merge base's and the head's
module, so comments, docstrings and formatting do not count; a renamed file is
compared with its old self). A changed fixture, helper, module constant or
``conftest.py`` selects nothing, and neither does a changed ``src/`` helper that
tests import: a test that only reads a changed helper is not re-run. The marker is
read from decorators, ``pytestmark`` and enclosing classes, not from
``pytest.param(marks=...)``. A pull request with no ``src/`` change has nothing to
check, and exits 0.

Exit status: 0 nothing to check or nothing flagged (or ``--report-only``), 1 a test is
flagged, 2 the check itself could not run.
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

MARKER = "pins_behaviour"
ROOT = Path(__file__).resolve().parents[1]
TEST_FILES = ("test_*.py", "*_test.py")
PYTEST_ERRORS = {2, 3, 4}  # interrupted, internal error, usage error; 5 is "nothing collected"
SHOWN = 20


class CheckError(Exception):
    """The check could not run; the message says why."""


@dataclass(frozen=True)
class Test:
    digest: str
    pinned: bool


def _mentions_marker(*nodes: ast.AST) -> bool:
    return any(
        isinstance(n, ast.Attribute) and n.attr == MARKER for node in nodes for n in ast.walk(node)
    )


def _digest(node: ast.FunctionDef | ast.AsyncFunctionDef, context: str) -> str:
    body = node.body
    first = body[0]
    if (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    ):
        body = body[1:]
    return context + "|".join(ast.dump(part) for part in (*node.decorator_list, node.args, *body))


def collect_tests(source: str) -> dict[str, Test]:
    """The test functions of a module by pytest id (``name`` or ``Class::name``)."""
    tree = ast.parse(source)
    module_pinned = any(
        isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in node.targets)
        and _mentions_marker(node.value)
        for node in tree.body
    )
    found: dict[str, Test] = {}

    def visit(body, classes, context, pinned):
        for node in body:
            if isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
                header = [*node.decorator_list, *node.bases]
                visit(
                    node.body,
                    (*classes, node.name),
                    context + "".join(ast.dump(part) for part in header),
                    pinned or _mentions_marker(*header),
                )
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith(
                "test"
            ):
                found["::".join((*classes, node.name))] = Test(
                    _digest(node, context), pinned or _mentions_marker(*node.decorator_list)
                )

    visit(tree.body, (), "", module_pinned)
    return found


def git(root: Path, *args: str, binary: bool = False):
    done = subprocess.run(["git", *args], cwd=root, capture_output=True, check=False)
    if done.returncode:
        raise CheckError(f"git {' '.join(args)} failed: {done.stderr.decode(errors='replace')}")
    return done.stdout if binary else done.stdout.decode()


def selected_tests(root: Path, merge_base: str, head: str):
    """``(src_changed, {(path, id): "added" | "changed"}, pinned ids, notes)`` of the diff."""
    src_changed, selected, pinned, notes = False, {}, set(), []
    for line in git(root, "diff", "--name-status", "-M", merge_base, head).splitlines():
        status, *paths = line.split("\t")
        src_changed |= any(path.startswith("src/") for path in paths)
        old, new = paths[0], paths[-1]
        if status.startswith("D") or not new.startswith("tests/"):
            continue
        if not any(fnmatch.fnmatch(Path(new).name, pattern) for pattern in TEST_FILES):
            continue
        try:
            before = {} if status.startswith("A") else _tests_at(root, merge_base, old)
            after = _tests_at(root, head, new)
        except SyntaxError as error:
            notes.append(f"{new}: not parsed ({error.msg}); its tests were not selected")
            continue
        for name, test in after.items():
            if name not in before:
                selected[(new, name)] = "added"
            elif before[name].digest != test.digest:
                selected[(new, name)] = "changed"
            if test.pinned:
                pinned.add((new, name))
    return src_changed, selected, pinned, notes


def _tests_at(root: Path, rev: str, path: str) -> dict[str, Test]:
    return collect_tests(git(root, "show", f"{rev}:{path}"))


def base_tree(root: Path, merge_base: str, head: str, where: Path, package: str) -> dict[str, str]:
    """A worktree of ``head`` whose ``src/`` is the merge base's, and the environment to run it in."""
    git(root, "worktree", "add", "--detach", str(where), head)
    shutil.rmtree(where / "src")
    archive = git(root, "archive", merge_base, "src", binary=True)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(where, filter="data")
    path = os.pathsep.join(filter(None, [str(where / "src"), os.environ.get("PYTHONPATH", "")]))
    env = {**os.environ, "PYTHONPATH": path}
    probe = "import importlib.util as u; print(u.find_spec(sys.argv[1]).origin)"
    origin = subprocess.run(
        [sys.executable, "-c", "import sys;" + probe, package],
        cwd=where,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    if not Path(origin).resolve().is_relative_to((where / "src").resolve()):
        raise CheckError(f"`import {package}` resolves to {origin!r}, not the base tree's src/")
    return env


def pytest_run(where: Path, env, *args: str) -> subprocess.CompletedProcess:
    # The repository's addopts carry -v, which turns -q's one id per line into a tree.
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-p",
        "no:cacheprovider",
        "-o",
        "addopts=",
        "-q",
        *args,
    ]
    return subprocess.run(command, cwd=where, env=env, capture_output=True, text=True, check=False)


def _last_error_line(text: str) -> str:
    lines = [line[1:].strip() for line in text.splitlines() if line.startswith("E ")]
    return lines[-1] if lines else "collection failed"


def collect_on_base(where, env, paths):
    """The function ids pytest finds per module, and the modules that fail to import."""
    done = pytest_run(where, env, "--collect-only", "--continue-on-collection-errors", *paths)
    if done.returncode in PYTEST_ERRORS:
        raise CheckError(f"collection on base exited {done.returncode}:\n{done.stdout[-2000:]}")
    ids = {
        tuple(re.sub(r"\[.*", "", line, flags=re.S).split("::", 1))
        for line in done.stdout.splitlines()
        if re.match(r"\S+\.py::", line)
    }
    parts = re.split(r"_+ ERROR collecting (\S+) _+\n", done.stdout)
    broken = {path: _last_error_line(text) for path, text in zip(parts[1::2], parts[2::2])}
    if not ids and len(broken) < len(
        paths
    ):  # a changed output format would otherwise pass everything
        raise CheckError(f"pytest listed no tests on base:\n{done.stdout[-2000:]}")
    return ids, broken


def _kind(message: str) -> str:
    head = message.split(":", 1)[0].strip()
    return head if re.fullmatch(r"[A-Za-z_][\w.]*", head) else "AssertionError"


def run_on_base(where, env, ids, pytest_args):
    """``{(path, id): (status, kind)}`` for ``ids`` from one junit report; status is
    ``fails``, ``passes`` or ``skipped``."""
    report = where / "base-junit.xml"
    modules = {path[:-3].replace("/", "."): path for path, _ in ids}
    nodes = sorted(f"{path}::{name}" for path, name in ids)
    done = pytest_run(
        where, env, f"--junitxml={report}", "-m", "not docs and not browser", *pytest_args, *nodes
    )
    if done.returncode in PYTEST_ERRORS or not report.exists():
        raise CheckError(f"pytest on base exited {done.returncode}:\n{done.stdout[-2000:]}")
    cases: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for case in ET.parse(report).iter("testcase"):
        owner = case.get("classname", "")
        for dotted, path in modules.items():
            if owner == dotted or owner.startswith(dotted + "."):
                classes = [c for c in owner[len(dotted) + 1 :].split(".") if c]
                name = re.sub(r"\[.*", "", case.get("name", ""), flags=re.S)
                outcome = next((c for c in case if c.tag in ("failure", "error", "skipped")), None)
                status = (
                    "passes"
                    if outcome is None
                    else {"skipped": "skipped"}.get(outcome.tag, "fails")
                )
                kind = _kind(outcome.get("message", "")) if status == "fails" else ""
                cases.setdefault((path, "::".join([*classes, name])), []).append((status, kind))
    found = {}
    for key, outcomes in cases.items():
        statuses = {status for status, _ in outcomes}
        if "fails" in statuses:
            found[key] = ("fails", next(kind for status, kind in outcomes if status == "fails"))
        else:
            found[key] = ("passes" if "passes" in statuses else "skipped", "")
    return found


def judge(selected, pinned, collected, broken, ran):
    """``{(path, id): {"how": ..., "status": ..., "kind": ...}}`` for every selected test."""
    verdicts = {}
    for key, how in selected.items():
        path = key[0]
        if path in broken:
            status, kind = "fails", _kind(broken[path])
        elif key not in collected:
            status, kind = "not_collected", ""
        elif key not in ran:
            status, kind = "not_run", ""
        else:
            status, kind = ran[key]
        flagged = status == "passes" and key not in pinned
        verdicts[key] = {
            "how": how,
            "status": status,
            "kind": kind,
            "pinned": key in pinned,
            "flagged": flagged,
        }
    return verdicts


def render(verdicts, notes, merge_base, head) -> str:
    def label(key):
        return f"{key[0]}::{key[1]} ({verdicts[key]['how']})"

    by_status: dict[str, list] = {}
    for key, verdict in verdicts.items():
        by_status.setdefault(verdict["status"], []).append(key)
    flagged = [key for key, verdict in verdicts.items() if verdict["flagged"]]
    kinds: dict[str, int] = {}
    for key in by_status.get("fails", []):
        kinds[verdicts[key]["kind"]] = kinds.get(verdicts[key]["kind"], 0) + 1
    lines = [
        f"fix-test check: head {head[:10]} against src/ of {merge_base[:10]}",
        f"{len(verdicts)} tests added or changed",
        f"  fail or error on base ......... {len(by_status.get('fails', []))}",
        f"  pass on base, pins_behaviour .. {sum(verdicts[k]['pinned'] and verdicts[k]['status'] == 'passes' for k in verdicts)}",
        f"  pass on base, unmarked ........ {len(flagged)}",
        f"  skipped on base ............... {len(by_status.get('skipped', []))}",
        f"  not run or not collected ...... "
        f"{len(by_status.get('not_run', [])) + len(by_status.get('not_collected', []))}",
        *notes,
    ]
    if kinds:
        counted = ", ".join(f"{kind} {n}" for kind, n in sorted(kinds.items(), key=lambda p: -p[1]))
        lines.append(f"fail or error on base, by exception: {counted}")
        lines.append(
            "  caveat: a failure on base counts as a pass here, whatever its reason; a name that "
            "is missing on base (ImportError, AttributeError, NameError) is not the behaviour asserted."
        )
        wrong: dict[tuple[str, str], int] = {}
        for key in by_status["fails"]:
            if verdicts[key]["kind"] != "AssertionError":
                where = (key[0], verdicts[key]["kind"])
                wrong[where] = wrong.get(where, 0) + 1
        lines += [f"    {path}: {kind} x {n}" for (path, kind), n in sorted(wrong.items())[:SHOWN]]
    for status, title in (("skipped", "skipped on base"), ("not_run", "not run (docs/browser)")):
        for key in by_status.get(status, [])[:SHOWN]:
            lines.append(f"  {title}: {label(key)}")
    for key in by_status.get("not_collected", [])[:SHOWN]:
        lines.append(f"  not collected by pytest: {label(key)}")
    if flagged:
        lines.append(
            f"FLAGGED: {len(flagged)} test(s) pass on the code before the change and carry no "
            f"@pytest.mark.{MARKER}. Either they do not need the change, or they pin behaviour "
            "that already worked: mark them."
        )
        lines += [f"  {label(key)}" for key in flagged]
    else:
        lines.append("no flagged tests")
    return "\n".join(lines)


def check(args) -> int:
    root = Path(args.root).resolve()
    head = git(root, "rev-parse", args.head).strip()
    merge_base = git(root, "merge-base", args.base, head).strip()
    src_changed, selected, pinned, notes = selected_tests(root, merge_base, head)
    if not src_changed:
        print("no src/ change: nothing to check")
        return 0
    if not selected:
        print("no test added or changed: nothing to check")
        return 0
    with tempfile.TemporaryDirectory(prefix="fix-tests-") as scratch:
        where = Path(scratch) / "tree"
        try:
            env = base_tree(root, merge_base, head, where, args.package)
            paths = sorted({path for path, _ in selected})
            collected, broken = collect_on_base(where, env, paths)
            good = {key for key in selected if key in collected and key[0] not in broken}
            ran = run_on_base(where, env, good, args.pytest_args) if good else {}
        finally:
            if where.exists():
                git(root, "worktree", "remove", "--force", str(where))
    verdicts = judge(selected, pinned, collected, broken, ran)
    print(render(verdicts, notes, merge_base, head))
    if args.json:
        rows = [{"test": f"{p}::{n}", **v} for (p, n), v in sorted(verdicts.items())]
        Path(args.json).write_text(
            json.dumps({"head": head, "merge_base": merge_base, "tests": rows}, indent=1)
        )
    flagged = any(verdict["flagged"] for verdict in verdicts.values())
    return 1 if flagged and not args.report_only else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base", required=True, help="the pull request's base revision")
    parser.add_argument("--head", default="HEAD")
    parser.add_argument("--root", default=str(ROOT), help="the git repository to check")
    parser.add_argument("--package", default="superglm", help="the package under src/")
    parser.add_argument("--report-only", action="store_true", help="print the flags, exit 0")
    parser.add_argument("--json", help="write every verdict to this file")
    parser.add_argument("pytest_args", nargs="*", help="after `--`: extra arguments for pytest")
    try:
        return check(parser.parse_args(argv))
    except CheckError as error:
        print(f"fix-test check could not run: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
