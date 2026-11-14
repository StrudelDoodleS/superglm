"""Run CI's quick checks before a push, so a change that would fail them never leaves the machine.

    uv run python scripts/pre_push.py              # every check
    uv run python scripts/pre_push.py --no-suite   # skip the quick test suite
    uv run python scripts/pre_push.py --install    # run it from git's pre-push hook

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
   and not docs"``.

On a machine with ``benchboard`` on the path, every pytest run goes through
it. Native Windows, macOS and ARM64 behaviour still needs CI's runners; no
local check stands in for them.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUICK = "not slow and not browser and not docs"
# The kernels GitHub's x86-64 runners dispatch to (AMD EPYC runs Haswell's,
# Intel Xeon SkylakeX's), each with the CPU features it needs, as NumPy names
# them. A forced kernel the host cannot execute crashes, so it is skipped.
KERNELS = {"Haswell": ("AVX2", "FMA3"), "SkylakeX": ("AVX512_SKX",)}
NO_TESTS_COLLECTED = 5
TY_EXTRAS = ("dev", "bench", "plotting")
# Git runs the hook from the top of the worktree being pushed. A branch from
# before the gate has no script, so it pushes as it did then.
HOOK_MARK = "# Installed by scripts/pre_push.py --install."
HOOK = f"""#!/bin/sh
{HOOK_MARK}
[ -f scripts/pre_push.py ] || exit 0
exec uv run --no-sync python scripts/pre_push.py
"""


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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--install", action="store_true", help="install the git pre-push hook")
    parser.add_argument("--no-suite", action="store_true", help="skip the quick test suite")
    options = parser.parse_args(argv)
    if options.install:
        _install()
        return

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
    if not options.no_suite:
        suite = ["uv", "run", "--no-sync", "python", "scripts/run_test_suite.py", "-m", QUICK, "-q"]
        _run("quick suite", _pytest("quick-suite", suite))
    print("pre-push: every check passed")


if __name__ == "__main__":
    main()
