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
* skipped on base, deselected (``docs``, ``browser`` and ``threads``: the last needs
  the default thread pools, which this run pins) or not collected (a module that skips
  itself at collection, say for a missing optional package, is one): reported, never
  counted as failing.

A parametrized test counts as passing on base only if every case that ran passes.

Limits. A test is selected when it is new or when its own decorators, signature
or body changed (found by comparing ``ast`` dumps of the merge base's and the head's
module, so comments, docstrings and formatting do not count; a renamed file is
compared with its old self). A changed fixture, helper, module constant or
``conftest.py`` selects nothing, and neither does a changed ``src/`` helper that
tests import: a test that only reads a changed helper is not re-run. The marker is
read from the test's own decorators, its enclosing classes' decorators and ``pytestmark``
(in a class body or at module level), and not from inside ``parametrize``: a mark on one
``pytest.param`` pins no test, so mark the whole test. A pull request with no ``src/``
change has nothing to check, and exits 0.

The run on base stops after ``--timeout`` seconds (default 1200, inside the workflow's
``timeout`` wrapper) with its process tree killed. The check then prints what it collected,
none of it judged, and exits 2.

Exit status: 0 nothing to check or nothing flagged (or ``--report-only``), 1 a test is
flagged, 2 the check itself could not run (a timeout included).
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import fnmatch
import io
import json
import os
import re
import shutil
import signal
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
RUN_TIMEOUT = 1200  # seconds for the run on base; the workflow's `timeout` wrapper is 30 minutes
COLLECT_TIMEOUT = 300  # seconds for collecting on base
TIMED_OUT = 124  # the status coreutils' `timeout` returns; pytest's own are 0 to 5


class CheckError(Exception):
    """The check could not run; the message says why."""


class RunTimeoutError(CheckError):
    """The run on base outlived its limit."""


@dataclass(frozen=True)
class Test:
    digest: str
    pinned: bool


def _names_marker(expression: ast.AST) -> bool:
    """Whether ``expression`` is the marker itself, bare or called: not one named inside its arguments."""
    if isinstance(expression, ast.Call):
        expression = expression.func
    return isinstance(expression, ast.Attribute) and expression.attr == MARKER


def _pytestmark_pins(statement: ast.AST) -> bool:
    """Whether ``statement`` assigns ``pytestmark`` a mark or a list of marks that includes the marker."""
    if isinstance(statement, ast.Assign):
        targets, value = statement.targets, statement.value
    elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
        targets, value = [statement.target], statement.value
    else:
        return False
    if not any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in targets):
        return False
    marks = value.elts if isinstance(value, (ast.List, ast.Tuple)) else [value]
    return any(_names_marker(mark) for mark in marks)


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
    module_pinned = any(_pytestmark_pins(node) for node in tree.body)
    found: dict[str, Test] = {}

    def visit(body, classes, context, pinned):
        for node in body:
            if isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
                header = [*node.decorator_list, *node.bases]
                visit(
                    node.body,
                    (*classes, node.name),
                    context + "".join(ast.dump(part) for part in header),
                    pinned
                    or any(_names_marker(d) for d in node.decorator_list)
                    or any(_pytestmark_pins(statement) for statement in node.body),
                )
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith(
                "test"
            ):
                found["::".join((*classes, node.name))] = Test(
                    _digest(node, context), pinned or any(map(_names_marker, node.decorator_list))
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


def pinned_pools(base) -> dict[str, str]:
    """``base`` with every thread pool pinned to one as ``run_test_suite`` pins them, so that
    each xdist worker runs serially and starts no pool of its own."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from run_test_suite import stage_environment
    finally:
        sys.path.pop(0)
    return stage_environment(True, base)


def base_tree(root: Path, merge_base: str, head: str, where: Path, package: str) -> dict[str, str]:
    """A worktree of ``head`` whose ``src/`` is the merge base's, and the environment to run it in."""
    git(root, "worktree", "add", "--detach", str(where), head)
    shutil.rmtree(where / "src")
    archive = git(root, "archive", merge_base, "src", binary=True)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(where, filter="data")
    path = os.pathsep.join(filter(None, [str(where / "src"), os.environ.get("PYTHONPATH", "")]))
    env = {**pinned_pools(os.environ), "PYTHONPATH": path}
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


def _own_group() -> dict:
    """``Popen`` options that put the child at the root of a process group it can be killed by."""
    return {} if sys.platform == "win32" else {"start_new_session": True}


def _kill_tree(child: subprocess.Popen) -> None:
    """Kill ``child`` and every process it started, then reap it."""
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(child.pid)], capture_output=True, check=False
        )
    else:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(child.pid, signal.SIGKILL)
    if child.poll() is None:
        child.kill()
    child.wait()


def pytest_run(where: Path, env, *args: str, timeout: float) -> subprocess.CompletedProcess:
    """Run pytest in ``where``; ``returncode`` is ``TIMED_OUT`` when it outlived ``timeout`` seconds.

    The output goes to files, not pipes: a pipe stays open while any process the run started
    holds it (an xdist worker, a test's own subprocess), so reading it after a kill would wait
    for them. The whole process tree is killed, so none outlives the check, and the run is
    unbuffered, so what it printed before the kill is not lost.
    """
    # The repository's addopts carry -v, which turns -q's one id per line into a tree.
    command = [
        sys.executable,
        "-u",
        "-m",
        "pytest",
        "-p",
        "no:cacheprovider",
        "-o",
        "addopts=",
        "-q",
        *args,
    ]
    with (
        tempfile.TemporaryFile("w+", errors="replace") as out,
        tempfile.TemporaryFile("w+", errors="replace") as err,
    ):
        child = subprocess.Popen(
            command, cwd=where, env=env, stdout=out, stderr=err, **_own_group()
        )
        try:
            status = child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            status = TIMED_OUT
        finally:
            if child.poll() is None:  # a timeout or an interrupt: nothing it started outlives it
                _kill_tree(child)
        out.seek(0)
        err.seek(0)
        return subprocess.CompletedProcess(command, status, out.read(), err.read())


def _output(done: subprocess.CompletedProcess) -> str:
    """The tail of what pytest printed on each stream; a conftest import error is on stderr."""
    return "\n".join(part[-2000:] for part in (done.stdout, done.stderr) if part.strip())


def _error_head(text: str) -> str:
    """The line of a report that names its exception: ``Type: message``, else ``assert ...``.

    Pytest marks every line of the exception text with ``E``, a message that spans lines
    included, so the last ``E`` line may be the tail of a message. The final exception is the
    last unbroken run of ``E`` lines, and the first of its lines that starts ``Type:`` is its
    header (a ``SyntaxError`` prints its source line before it); a plain ``assert`` has none.
    """
    runs: list[list[str]] = []
    follows = False
    for line in text.splitlines():
        marked = re.match(r"E(\s|$)", line) is not None
        if marked:
            if not follows:
                runs.append([])
            runs[-1].append(line[1:].strip())
        follows = marked
    if not runs:
        return ""
    return next((line for line in runs[-1] if _names_exception(line)), runs[-1][0])


def _names_exception(line: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z_][\w.]*", line.split(":", 1)[0].strip()))


def collect_on_base(where, env, paths):
    """The function ids pytest finds per module, the modules that fail to import, and the
    modules that skip themselves at collection (with the reason)."""
    done = pytest_run(
        where,
        env,
        "--collect-only",
        "--continue-on-collection-errors",
        "-rs",
        *paths,
        timeout=COLLECT_TIMEOUT,
    )
    if done.returncode == TIMED_OUT:
        raise CheckError(
            f"collection on base timed out after {COLLECT_TIMEOUT} s:\n{_output(done)}"
        )
    if done.returncode in PYTEST_ERRORS:
        raise CheckError(f"collection on base exited {done.returncode}:\n{_output(done)}")
    ids = {
        tuple(re.sub(r"\[.*", "", line, flags=re.S).split("::", 1))
        for line in done.stdout.splitlines()
        if re.match(r"\S+\.py::", line)
    }
    parts = re.split(r"_+ ERROR collecting (\S+) _+\n", done.stdout)
    broken = {
        path: _error_head(text) or "collection failed"
        for path, text in zip(parts[1::2], parts[2::2])
    }
    skipped = {
        # a node id always has "/", but the location of a skip is a path, and a Windows one has "\"
        match[1].replace("\\", "/"): match[2]
        for match in re.finditer(r"^SKIPPED \[\d+\] (\S+?\.py):\d+: (.*)$", done.stdout, re.M)
    }
    if not ids and not set(paths) <= broken.keys() | skipped.keys():
        # a changed output format would otherwise pass everything
        raise CheckError(f"pytest listed no tests on base:\n{_output(done)}")
    return ids, broken, skipped


def _kind(message: str) -> str:
    """The exception type a report line starts with, else ``AssertionError``: a plain assert
    reports ``assert x == y``, which no type precedes."""
    head = re.sub(r'^failed on (?:setup|teardown) with "', "", message).split(":", 1)[0].strip()
    return head if re.fullmatch(r"[A-Za-z_][\w.]*", head) else "AssertionError"


def _failure_kind(outcome: ET.Element) -> str:
    """The kind of a failing or erroring case, from the exception text of its report: that text
    names the exception of a setup error and ends a custom assertion message, which pytest's
    one-line ``message`` attribute does not do for either."""
    return _kind(_error_head(outcome.text or "") or outcome.get("message", ""))


def run_on_base(where, env, ids, pytest_args, timeout):
    """``{(path, id): (status, kind)}`` for ``ids`` from one junit report; status is
    ``fails``, ``passes`` or ``skipped``. Raises ``RunTimeoutError`` after ``timeout`` seconds."""
    report = where / "base-junit.xml"
    modules = {path[:-3].replace("/", "."): path for path, _ in ids}
    nodes = sorted(f"{path}::{name}" for path, name in ids)
    done = pytest_run(
        where,
        env,
        f"--junitxml={report}",
        "-m",
        "not docs and not browser and not threads",
        *pytest_args,
        *nodes,
        timeout=timeout,
    )
    if done.returncode == TIMED_OUT:
        raise RunTimeoutError(f"the run on base timed out after {timeout:g} s:\n{_output(done)}")
    if done.returncode in PYTEST_ERRORS or not report.exists():
        raise CheckError(f"pytest on base exited {done.returncode}:\n{_output(done)}")
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
                kind = _failure_kind(outcome) if status == "fails" else ""
                cases.setdefault((path, "::".join([*classes, name])), []).append((status, kind))
    found = {}
    for key, outcomes in cases.items():
        statuses = {status for status, _ in outcomes}
        if "fails" in statuses:
            found[key] = ("fails", next(kind for status, kind in outcomes if status == "fails"))
        else:
            found[key] = ("passes" if "passes" in statuses else "skipped", "")
    return found


def judge(selected, pinned, collected, broken, ran, unfinished=frozenset()):
    """``{(path, id): {"how": ..., "status": ..., "kind": ...}}`` for every selected test.

    ``unfinished`` holds the tests a run that timed out was sent, which it did not judge."""
    verdicts = {}
    for key, how in selected.items():
        path = key[0]
        if path in broken:
            status, kind = "fails", _kind(broken[path])
        elif key not in collected:
            status, kind = "not_collected", ""
        elif key in unfinished:
            status, kind = "timed_out", ""
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
        f"  not run, not collected, timed out {sum(len(by_status.get(s, [])) for s in ('not_run', 'not_collected', 'timed_out'))}",
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
    for status, title in (
        ("skipped", "skipped on base"),
        ("not_run", "not run (docs, browser, threads)"),
        ("timed_out", "collected, not judged: the run on base timed out"),
    ):
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
    timed_out = None
    with tempfile.TemporaryDirectory(prefix="fix-tests-") as scratch:
        where = Path(scratch) / "tree"
        try:
            env = base_tree(root, merge_base, head, where, args.package)
            paths = sorted({path for path, _ in selected})
            collected, broken, skipped = collect_on_base(where, env, paths)
            notes += [
                f"{path}: skipped at collection on base ({reason}); its tests were not run"
                for path, reason in sorted(skipped.items())
            ]
            good = {key for key in selected if key in collected and key[0] not in broken}
            ran = {}
            if good:
                try:
                    ran = run_on_base(where, env, good, args.pytest_args, args.timeout)
                except RunTimeoutError as error:
                    timed_out = error
                    notes.append(
                        f"the run on base timed out after {args.timeout:g} s: no test was judged"
                    )
        finally:
            if where.exists():
                git(root, "worktree", "remove", "--force", str(where))
    verdicts = judge(selected, pinned, collected, broken, ran, good if timed_out else frozenset())
    print(render(verdicts, notes, merge_base, head))
    if args.json:
        rows = [{"test": f"{p}::{n}", **v} for (p, n), v in sorted(verdicts.items())]
        Path(args.json).write_text(
            json.dumps({"head": head, "merge_base": merge_base, "tests": rows}, indent=1)
        )
    if timed_out:
        raise timed_out  # after the report: the check ends as one that could not run
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
    parser.add_argument(
        "--timeout",
        type=float,
        default=RUN_TIMEOUT,
        help="seconds the run of the selected tests on base may take (default: %(default)s); "
        "the check then reports what it collected and exits 2",
    )
    parser.add_argument("pytest_args", nargs="*", help="after `--`: extra arguments for pytest")
    try:
        return check(parser.parse_args(argv))
    except CheckError as error:
        print(f"fix-test check could not run: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
