"""Run CI's quick checks before a push, so a change that would fail them never leaves the machine.

    uv run python scripts/pre_push.py              # every check
    uv run python scripts/pre_push.py --full       # the whole quick suite, not a selection
    uv run python scripts/pre_push.py --no-suite   # skip the quick test suite
    uv run python scripts/pre_push.py --install    # refuse pushes of commits it has not passed

The checks, in order, stopping at the first failure:

1. ``uv lock --check``, then ``uv sync`` with the extras the type-check job
   installs, which every later check runs in.
2. ``ruff format --check`` and ``ruff check`` on ``src/`` and ``tests/``.
3. ``ty`` on ``src/superglm``, its diagnostic count against the backlog the
   type-check job accepts (read from ``.github/workflows/dev-ci.yml``).
4. The test files the branch changes since its merge base with
   ``origin/master``, under OpenBLAS's Haswell and SkylakeX kernels
   (``OPENBLAS_CORETYPE``): a test that asserts one kernel's rounding fails
   here instead of on a CI runner with another CPU. A kernel this CPU cannot
   execute is skipped; with neither (ARM64), the tests run once as dispatched.
5. The quick suite, ``scripts/run_test_suite.py -m "not slow and not browser
   and not docs"``: the test files the change reaches, by the test map
   ``scripts/test_map.py`` keeps (its docstring gives every rule), or the whole
   quick suite when there is no map, when a file that configures every test
   changed, when the selection would take most of the suite's time anyway, or
   with ``--full``. A whole-suite run on a clean tree records a new map.

When every check passes on a clean tree, the commit is stamped as passed, with
which suite passed (a selection of N of the map's M tests, or the whole quick
suite) and the commit's series of patch ids over its merge base with
``origin/master``. The hook ``--install`` writes only checks that every commit
being pushed carries a stamp, and says which suite each passed. A commit with
no stamp of its own carries one whose series is the same (``git patch-id
--stable``, commit by commit): a rebase that changed no commit's diff. A rebase
that resolved a conflict changes a patch id and needs the checks again. The
hook takes a moment: git opens its connection to the remote before running
the hook, and a remote drops a connection left idle for the length of the
checks. ``git push --no-verify`` skips the hook.

On a machine with ``benchboard`` on the path, every pytest run goes through
it. Native Windows, macOS and ARM64 behaviour still needs CI's runners; no
local check stands in for them.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUICK = "not slow and not browser and not docs"
# The kernels GitHub's x86-64 runners dispatch to (AMD EPYC runs Haswell's,
# Intel Xeon SkylakeX's), each with the CPU features it needs, as NumPy names
# them. A forced kernel the host cannot execute crashes, so it is skipped.
KERNELS = {"Haswell": ("AVX2", "FMA3"), "SkylakeX": ("AVX512_SKX",)}
NO_TESTS_COLLECTED = 5
SELECTION_SECONDS_PER_WORKER = 30
TY_EXTRAS = ("dev", "bench", "plotting")
# Git runs the hook from the top of the worktree being pushed. A branch from
# before the gate has no script, and one from before the stamps has a script
# without --verify-push; both push as they did then.
HOOK_MARK = "# Installed by scripts/pre_push.py --install."
HOOK = f"""#!/bin/sh
{HOOK_MARK}
[ -f scripts/pre_push.py ] || exit 0
grep -q -- --verify-push scripts/pre_push.py || exit 0
exec uv run --no-sync python scripts/pre_push.py --verify-push
"""
# A pre-push hook reads "<local ref> <local sha> <remote ref> <remote sha>" lines;
# a deletion pushes this all-zero sha.
NO_COMMIT = "0" * 40


def _run(
    label: str, command: list[str], env: dict[str, str] | None = None, passing=frozenset({0})
) -> None:
    print(f"pre-push: {label}", flush=True)
    result = subprocess.run(command, cwd=ROOT, env=env)
    if result.returncode not in passing:
        sys.exit(f"pre-push: {label} failed (exit {result.returncode})")


def _pytest(label: str, arguments: list[str], cores: int | None = None) -> list[str]:
    """``arguments`` run through benchboard where it is installed."""
    if shutil.which("benchboard") is None:
        return arguments
    core_option = [] if cores is None else ["--cores", str(cores)]
    return ["benchboard", "heavy", f"pre-push-{label}", *core_option, "--", *arguments]


def _with_variables(
    command: list[str], variables: dict[str, str]
) -> tuple[list[str], dict[str, str] | None]:
    """``command`` and the environment that sets ``variables`` for the suite alone.

    Under benchboard they go after its ``--`` through ``env``, so benchboard's own Python does
    not see them (the map recorder would start in it).
    """
    if not variables:
        return command, None
    if command[0] != "benchboard":
        return command, {**os.environ, **variables}
    split = command.index("--") + 1
    setting = [f"{name}={value}" for name, value in variables.items()]
    return [*command[:split], "env", *setting, *command[split:]], None


def _ty_backlog() -> int:
    workflow = (ROOT / ".github" / "workflows" / "dev-ci.yml").read_text(encoding="utf-8")
    found = re.search(r"diagnostics > (\d+)", workflow)
    if found is None:
        sys.exit("pre-push: no ty backlog in .github/workflows/dev-ci.yml")
    return int(found.group(1))


def _ty() -> None:
    print("pre-push: ty diagnostics against the accepted backlog", flush=True)
    output = subprocess.run(
        ["uv", "run", "--no-sync", "ty", "check", "src/superglm", "--output-format", "concise"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    ).stdout
    if "All checks passed!" in output:
        count = 0
    else:
        counts = re.findall(r"^Found (\d+) diagnostics?$", output, re.MULTILINE)
        if not counts:
            sys.exit("pre-push: ty printed no diagnostic count")
        count = int(counts[-1])
    backlog = _ty_backlog()
    if count > backlog:
        sys.exit(f"pre-push: ty diagnostics rose from the accepted {backlog} to {count}")


def _changed_tests() -> list[str]:
    base = subprocess.run(
        ["git", "merge-base", "HEAD", "origin/master"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    names = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=d", base, "--", "tests/"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    return [name for name in names if re.fullmatch(r"tests/(.*/)?test_[^/]*\.py", name)]


def _runnable_kernels() -> list[str | None]:
    """The kernels this host can execute; ``[None]`` (native dispatch) off x86-64 or with none."""
    from numpy._core._multiarray_umath import __cpu_features__ as features

    kernels = [name for name, needs in KERNELS.items() if all(features.get(f) for f in needs)]
    skipped = sorted(set(KERNELS) - set(kernels))
    if skipped:
        print(f"pre-push: this CPU cannot run OpenBLAS {', '.join(skipped)}; CI's runners do")
    return kernels or [None]


def _install() -> None:
    hooks = subprocess.run(
        ["git", "rev-parse", "--git-path", "hooks"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    hook = (ROOT / hooks / "pre-push").resolve()
    # Any hook this script wrote, by any version of it, carries the mark and is replaced.
    if hook.exists() and HOOK_MARK not in hook.read_text(encoding="utf-8").splitlines():
        sys.exit(f"pre-push: {hook} exists and is not this script's; leaving it alone")
    hook.parent.mkdir(parents=True, exist_ok=True)
    hook.write_text(HOOK, encoding="utf-8")
    hook.chmod(0o755)
    print(f"pre-push: installed {hook}")


# Git exports these to the hook (``git rev-parse --local-env-vars``), naming the
# repository being pushed; every check below runs without them, so nothing it
# starts (a test's throwaway ``git init``, books.py's git calls) acts on that
# repository. tests/conftest.py drops them too, for a suite started elsewhere.
GIT_REPOSITORY_ENV = (
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_CONFIG",
    "GIT_CONFIG_PARAMETERS",
    "GIT_CONFIG_COUNT",
    "GIT_OBJECT_DIRECTORY",
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_IMPLICIT_WORK_TREE",
    "GIT_GRAFT_FILE",
    "GIT_INDEX_FILE",
    "GIT_NO_REPLACE_OBJECTS",
    "GIT_REPLACE_REF_BASE",
    "GIT_PREFIX",
    "GIT_SHALLOW_FILE",
    "GIT_COMMON_DIR",
)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def _stamp(sha: str) -> Path:
    """Where the gate records that ``sha`` passed: shared by every worktree of the repository."""
    return Path(_git("rev-parse", "--path-format=absolute", "--git-common-dir")) / (
        f"superglm-pre-push/{sha}"
    )


def _clean_head() -> str | None:
    """HEAD's sha when no tracked file differs from it, else ``None``."""
    return (
        None if _git("status", "--porcelain", "--untracked-files=no") else _git("rev-parse", "HEAD")
    )


def _series(commit: str) -> list[str] | None:
    """``git patch-id --stable`` of each commit from the merge base with ``origin/master`` to
    ``commit``, oldest first; ``None`` when there is none, or a merge commit among them.

    Two commits with the same series apply the same diffs, commit by commit, to their bases: one
    is a rebase of the other that resolved no conflict.
    """
    try:
        base = _git("merge-base", commit, "origin/master")
        if _git("rev-list", "--merges", f"{base}..{commit}"):
            return None
        commits = _git("rev-list", "--reverse", f"{base}..{commit}").split()
    except subprocess.CalledProcessError:
        return None
    series = []
    for each in commits:
        diff = subprocess.run(
            ["git", "diff-tree", "-p", "--no-commit-id", "-r", each],
            cwd=ROOT,
            capture_output=True,
            check=True,
        ).stdout
        patch_id = subprocess.run(
            ["git", "patch-id", "--stable"], cwd=ROOT, input=diff, capture_output=True, check=True
        ).stdout.split()
        series.append(patch_id[0].decode() if patch_id else "empty")
    return series or None


def _stamp_text(passed: str, series: list[str] | None) -> str:
    return f"passed: {passed}\n" + (f"series: {' '.join(series)}\n" if series else "")


def _stamp_fields(path: Path) -> dict[str, str]:
    """A stamp's ``passed`` and ``series`` lines; a stamp from before they existed is empty."""
    text = path.read_text(encoding="utf-8")
    return dict(line.split(": ", 1) for line in text.splitlines() if ": " in line)


def _carried(commit: str) -> tuple[str, dict[str, str]] | None:
    """A stamped commit whose patch series ``commit`` repeats, and its stamp."""
    series = _series(commit)
    directory = _stamp(commit).parent
    if series is None or not directory.is_dir():
        return None
    wanted = " ".join(series)
    for path in sorted(directory.iterdir()):
        fields = _stamp_fields(path)
        if fields.get("series") == wanted:
            return path.name, fields
    return None


def _verify_push(lines) -> None:
    """Refuse the push unless every commit it sends has passed the gate, or carries a stamp."""
    unchecked = []
    for line in lines:
        fields = line.split()
        if len(fields) != 4 or fields[1] == NO_COMMIT:
            continue
        # An annotated tag arrives as its tag object; the commit it names is what passed.
        # A tag of a tree or blob names no commit, so there is nothing to check.
        try:
            commit = _git("rev-parse", "--verify", "--quiet", f"{fields[1]}^{{commit}}")
        except subprocess.CalledProcessError:
            continue
        if _stamp(commit).exists():
            passed = _stamp_fields(_stamp(commit)).get("passed", "the checks")
            print(f"pre-push: {commit[:12]} passed {passed}")
            continue
        carried = _carried(commit)
        if carried is None:
            unchecked.append(commit[:12])
            continue
        source, stamp = carried
        print(
            f"pre-push: {commit[:12]} carries the stamp of {source[:12]}, which passed "
            f"{stamp.get('passed', 'the checks')}: the same patches over origin/master, "
            "a rebase that changed no commit's diff"
        )
    if unchecked:
        sys.exit(
            f"pre-push: {', '.join(unchecked)} has not passed the checks; run "
            "`uv run python scripts/pre_push.py` on a clean tree, then push again "
            "(`git push --no-verify` skips this)"
        )


def _test_map():
    # Beside this script, wherever ROOT points: the selector is the one this gate ships with.
    path = Path(__file__).resolve().parent / "test_map.py"
    spec = importlib.util.spec_from_file_location("test_map", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["test_map"] = module  # dataclasses look their module up here
    spec.loader.exec_module(module)
    return module


def _junit_tests(report: Path) -> int:
    """The tests a run's junit reports list: stage 1's and, beside it, stage 2's."""
    reports = [report, report.with_name(f"{report.stem}-threads{report.suffix}")]
    return sum(sum(1 for _ in ET.parse(path).iter("testcase")) for path in reports if path.exists())


def _quick_suite(full: bool, start: str | None) -> str:
    """Run the quick suite, or the test map's selection of it; what passed, for the stamp."""
    test_map = _test_map()
    selection = test_map.Selection(full="--full") if full else test_map.choose(ROOT)
    with tempfile.TemporaryDirectory() as scratch:
        junit = Path(scratch) / "quick-suite.xml"
        suite = [
            *["uv", "run", "--no-sync", "python", "scripts/run_test_suite.py"],
            *["-m", QUICK, "-q", f"--junitxml={junit}"],
        ]
        variables: dict[str, str] = {}
        label, cores = "quick-suite", None
        if selection.full is None:
            listed = Path(scratch) / "selected.txt"
            listed.write_text("\n".join(sorted(selection.files)) + "\n", encoding="utf-8")
            # A worker per SELECTION_SECONDS_PER_WORKER of recorded test time, so a small
            # selection leaves the rest of a shared machine free.
            cores = min(
                os.cpu_count() or 1,
                max(2, math.ceil(selection.seconds / SELECTION_SECONDS_PER_WORKER)),
            )
            suite += ["--files-from", str(listed), "-n", str(cores)]
            label = "quick-selection"
            print(
                f"pre-push: the test map selects {len(selection.files)} test files, "
                f"{selection.seconds:.0f} of the quick suite's {selection.total_seconds:.0f} "
                "recorded seconds (`uv run python scripts/test_map.py select` says why)"
            )
        else:
            print(f"pre-push: the whole quick suite: {selection.full}")
            if start is not None:  # a clean tree: this run records a map of the commit
                out = Path(scratch) / "map"
                variables = test_map.recording_environment(out, ROOT)
                suite += list(test_map.RECORDING_ARGUMENTS)
        command, env = _with_variables(_pytest(label, suite, cores), variables)
        _run("quick suite", command, env=env, passing=frozenset({0, NO_TESTS_COLLECTED}))
        tests = _junit_tests(junit)
        if variables:
            data = test_map.merge(out)
            print(
                f"pre-push: recorded a test map of {start[:12]} in {test_map.save(data, start, ROOT)}"
            )
    if selection.full is None:
        return (
            f"a selection of {tests} tests; the quick suite of the map "
            f"({selection.map_commit[:12]}) has {selection.map_tests}"
        )
    return f"the whole quick suite ({tests} tests)"


def main(argv: list[str] | None = None) -> None:
    for name in GIT_REPOSITORY_ENV:
        os.environ.pop(name, None)
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--install", action="store_true", help="install the git pre-push hook")
    parser.add_argument("--no-suite", action="store_true", help="skip the quick test suite")
    parser.add_argument(
        "--full",
        action="store_true",
        help="run the whole quick suite, not the test map's selection",
    )
    parser.add_argument(
        "--verify-push",
        action="store_true",
        help="the hook: refuse commits the checks have not passed",
    )
    options = parser.parse_args(argv)
    if options.install:
        _install()
        return
    if options.verify_push:
        _verify_push(sys.stdin)
        return
    start = _clean_head()

    _run("uv lock", ["uv", "lock", "--check"])
    # Every later check runs in the environment the type-check job installs,
    # whatever this worktree had synced before.
    extras = [option for extra in TY_EXTRAS for option in ("--extra", extra)]
    _run("uv sync with CI's extras", ["uv", "sync", "--locked", "--quiet", *extras])
    _run("ruff format", ["uv", "run", "--no-sync", "ruff", "format", "--check", "src", "tests"])
    _run("ruff check", ["uv", "run", "--no-sync", "ruff", "check", "src", "tests"])
    _ty()
    pytest = ["uv", "run", "--no-sync", "python", "-m", "pytest", "-q", "-p", "no:randomly"]
    changed = _changed_tests()
    for kernel in _runnable_kernels() if changed else ():
        env = dict(os.environ) if kernel is None else {**os.environ, "OPENBLAS_CORETYPE": kernel}
        # Every selected test skipping (an optional dependency missing) collects nothing.
        _run(
            f"changed tests on OpenBLAS {kernel or 'native dispatch'}",
            _pytest(
                f"kernel-{kernel or 'native'}".lower(), [*pytest, "-n", "4", *changed], cores=4
            ),
            env=env,
            passing=frozenset({0, NO_TESTS_COLLECTED}),
        )
    if options.no_suite:
        print("pre-push: every check passed")
        return
    passed = _quick_suite(options.full, start)
    print("pre-push: every check passed")
    if start is None or _clean_head() != start:
        print("pre-push: the tree had uncommitted changes, so no commit is stamped as passed")
        return
    _stamp(start).parent.mkdir(exist_ok=True)
    _stamp(start).write_text(_stamp_text(passed, _series(start)), encoding="utf-8")
    print(f"pre-push: stamped {start[:12]} as passed: {passed}")


if __name__ == "__main__":
    main()
