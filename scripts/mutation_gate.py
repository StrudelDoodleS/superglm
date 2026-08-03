"""Assert that a change's own regression tests fail against the unfixed code.

`AGENTS.md` requires that "adversarial regressions include a mutation check or a
demonstration against the unfixed implementation".  This script enforces that
sentence mechanically, so it stops being a convention nobody can verify.

The check is two questions, and both must hold:

1. do the tests this change ADDED **pass** at the head revision, and
2. do they **fail** when only ``src/`` is rolled back to the merge base?

A test that passes against the unfixed implementation does not constrain the fix
-- it observes what the code already did.  A test that fails at *both* revisions
is simply broken, and must not be mistaken for evidence.  That failure mode is
invisible to review and has been found repeatedly in this repository:

- ``tests/test_shape_fit.py`` asserts one-sided second-derivative bounds that an
  identically zero curve satisfies, so it stayed green while the constrained
  smooth it covers collapsed to a flat line;
- a per-iteration feasibility guard survived mutation against all 272 tests in
  its own five test files, while the same harness killed nine sibling mutations;
- a BLAS-sampler dedup guard shipped with a regression test whose fixture never
  reached the deduplicating path.

Design notes, each answering a way an earlier revision of this script could be
satisfied for the wrong reason:

**Isolated worktrees.**  The caller's checkout is never modified.  Two throwaway
``git worktree`` trees are built -- one at the merge base, one at the head -- and
pytest runs inside them.  Overlaying a base ``src/`` onto the head checkout was
not a revert: ``git checkout <tree> -- <dir>`` does not delete files that exist
only at the head, so a fix landing in a *new* module was never mutated at all and
the gate accused a correct change.  In-place mutation also risked destroying
uncommitted work, and a signal during the restore could leave the tree
half-mutated.

**Attribution, not counting.**  The base and head revisions of each changed test
module are parsed with ``ast``; the set difference gives the test functions this
change added, class-qualified.  Those node IDs are what must fail.  Requiring
merely that *something* in the changed modules fails is satisfied by a
neighbouring test the same change reparameterised.

**Test-call failures only.**  pytest exits non-zero for interruptions, usage
errors, collection errors and "no tests ran" as well as for failures, and a
skipped or errored test is not evidence.  Only node IDs reported as ``FAILED``
count as a kill; ``ERROR`` is reported as inconclusive.

**The whole test tree is overlaid**, not just ``test_*.py``, so ``conftest.py``
and shared helpers stay consistent with the tests being run.

Two known limitations, both reported honestly rather than papered over:

- A change containing several independent fixes is accepted once *any* added
  test is killed.  Mapping each production hunk to its own killed mutation would
  need per-hunk mutants and is deliberately out of scope.
- A fix landing entirely in a **new** module cannot be mutation-tested this way:
  at the base revision the module does not exist, so its tests error rather than
  fail, and an ``ImportError`` is not evidence that a test constrains behaviour.
  That is reported ``INCONCLUSIVE``, not ``PASS``.

Outcomes
--------
``SKIP``          no production change to mutate.                     exit 0
``PASS``          an added test fails at base and passes at head.     exit 0
``FAIL``          every added test passes against the unfixed code.   exit 1
``NO EVIDENCE``   production change with nothing that demonstrates.   exit 1
``INCONCLUSIVE``  the mutant could not be evaluated.                  exit 1

The last two are non-zero deliberately: a green badge must not be shown when
nothing was demonstrated.  The check is advisory until it is added to the
ruleset, so a non-zero result is informational rather than blocking.

Usage
-----
    uv run python scripts/mutation_gate.py --base-ref origin/master

Exempt a change that genuinely cannot satisfy this by adding the
``mutation-gate-exempt`` label to the pull request; the justification belongs in
the PR body where review can see it.
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = "src/superglm"
TESTS = "tests"

SKIP, PASS, FAIL, NO_EVIDENCE, INCONCLUSIVE = (
    "SKIP",
    "PASS",
    "FAIL",
    "NO EVIDENCE",
    "INCONCLUSIVE",
)
_FAILED = re.compile(r"^FAILED\s+(\S+)")


def git(*args: str, cwd: Path | None = None, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args], cwd=cwd or ROOT, capture_output=True, text=True, check=False
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed:\n{result.stderr.strip()}")
    return result.stdout.strip()


def is_test_module(path: str) -> bool:
    return (
        path.startswith(f"{TESTS}/")
        and path.endswith(".py")
        and Path(path).name.startswith("test_")
    )


def file_at(rev: str, path: str) -> str | None:
    """A file's contents at ``rev``, or None when it does not exist there."""
    result = subprocess.run(
        ["git", "show", f"{rev}:{path}"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout if result.returncode == 0 else None


def test_qualnames(source: str | None) -> set[str]:
    """Class-qualified names of the test functions declared in ``source``."""
    if source is None:
        return set()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()

    names: set[str] = set()

    def walk(node: ast.AST, prefix: str) -> None:
        for child in getattr(node, "body", []):
            if isinstance(child, ast.ClassDef):
                walk(child, f"{prefix}{child.name}::")
            elif isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                if child.name.startswith("test"):
                    names.add(f"{prefix}{child.name}")

    walk(tree, "")
    return names


def added_node_ids(merge_base: str, head: str, test_files: list[str]) -> list[str]:
    """Node IDs of the test functions this change introduced."""
    node_ids: list[str] = []
    for path in test_files:
        before = test_qualnames(file_at(merge_base, path))
        after = test_qualnames(file_at(head, path))
        node_ids += [f"{path}::{name}" for name in sorted(after - before)]
    return node_ids


def run_pytest(tree: Path, targets: list[str]) -> tuple[set[str], int, str]:
    """Run pytest inside ``tree``; return (failed node ids, passed count, output).

    ``-ra`` is required, not ``-rf``: the short-summary flag ``f`` reports only
    failures, so collection ERRORs, skips and XPASSes never appear and would be
    silently read as absent.

    ``PYTHONPATH`` must point at this tree's ``src`` or the editable install's
    ``.pth`` would import the caller's checkout and every run would measure the
    same unmutated code.
    """
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            *targets,
            "-p",
            "no:cacheprovider",
            "-q",
            "--no-header",
            "-ra",
            "--tb=no",
        ],
        cwd=tree,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONPATH": str(tree / "src")},
    )
    out = proc.stdout + proc.stderr
    failed = {match.group(1) for line in out.splitlines() if (match := _FAILED.match(line.strip()))}
    passed_match = re.search(r"(\d+) passed", out)
    return failed, int(passed_match.group(1)) if passed_match else 0, out


def report(status: str, headline: str, detail: str = "") -> int:
    marker = {SKIP: "-", PASS: "+", FAIL: "x", NO_EVIDENCE: "?", INCONCLUSIVE: "?"}[status]
    print(f"\n{marker} mutation gate: {status}\n  {headline}")
    if detail:
        print("\n" + detail.rstrip())
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write(f"### Mutation gate - {status}\n\n{headline}\n")
            if detail:
                handle.write(f"\n~~~\n{detail.rstrip()}\n~~~\n")
    return 0 if status in (SKIP, PASS) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-ref", default="origin/master")
    parser.add_argument("--head-ref", default="HEAD")
    args = parser.parse_args()

    if os.environ.get("MUTATION_GATE_EXEMPT", "").strip():
        return report(
            SKIP,
            "Exempted by the mutation-gate-exempt label.",
            "The justification belongs in the pull-request body, where review can see it.",
        )

    merge_base = git("merge-base", args.base_ref, args.head_ref)
    head = git("rev-parse", args.head_ref)
    changed = [
        line for line in git("diff", "--name-only", f"{merge_base}..{head}").splitlines() if line
    ]
    src_changed = [p for p in changed if p.startswith(SRC)]
    tests_changed = [p for p in changed if is_test_module(p) and file_at(head, p) is not None]

    if not src_changed:
        return report(SKIP, "No change under src/superglm - nothing to mutate.")

    if not tests_changed:
        return report(
            NO_EVIDENCE,
            f"{len(src_changed)} production file(s) changed, no test module added or changed.",
            "AGENTS.md requires focused regression tests for new solver, REML, family,\n"
            "input-boundary or feature behaviour. Nothing here demonstrates that this\n"
            "change is constrained. A pure refactor may legitimately add none - in that\n"
            "case apply the mutation-gate-exempt label and say why in the PR body.\n\n"
            + "\n".join(f"  {p}" for p in src_changed),
        )

    added = added_node_ids(merge_base, head, tests_changed)
    print(f"mutation gate: base {merge_base[:12]}  head {head[:12]}")
    print(f"  production files changed: {len(src_changed)}")
    print(f"  test modules changed:     {len(tests_changed)}")
    print(f"  test functions added:     {len(added)}")

    if not added:
        return report(
            NO_EVIDENCE,
            "Test modules changed but no test function was added.",
            "Attribution needs an added test. Modifying an existing test can strengthen\n"
            "it, but this gate cannot tell a strengthened assertion from a rename, so it\n"
            "claims no evidence either way. Add a regression test, or apply the\n"
            "mutation-gate-exempt label.\n\n" + "\n".join(f"  {p}" for p in tests_changed),
        )

    # Two isolated trees. The caller's checkout is never touched.
    scratch = Path(tempfile.mkdtemp(prefix="mutation-gate-"))
    base_tree, head_tree = scratch / "base", scratch / "head"
    try:
        git("worktree", "add", "--detach", "--quiet", str(base_tree), merge_base)
        git("worktree", "add", "--detach", "--quiet", str(head_tree), head)

        # base src + HEAD tests: overlay the whole tests tree so conftest and
        # shared helpers match the tests being run.
        git("checkout", head, "--", TESTS, cwd=base_tree)

        base_failed, base_passed, base_out = run_pytest(base_tree, added)
        head_failed, head_passed, head_out = run_pytest(head_tree, added)

        # Every added test must genuinely PASS at head. Counting passes catches
        # failures, collection errors and skips in one check -- a skipped or
        # errored test is not evidence of anything.
        if head_passed != len(added):
            return report(
                INCONCLUSIVE,
                f"Only {head_passed} of {len(added)} added test(s) pass at the head revision.",
                "A test that does not pass at head demonstrates nothing about this change:\n"
                "it may be failing, erroring or skipped. Make the added tests pass at head\n"
                "first.\n\n" + head_out[-1500:],
            )

        killed = base_failed
        if killed:
            return report(
                PASS,
                f"{len(killed)} of {len(added)} added test(s) fail against the unfixed "
                "code and pass at head, as required.",
                "\n".join(f"  {n}" for n in sorted(killed)),
            )

        if base_passed != len(added):
            return report(
                INCONCLUSIVE,
                f"Added tests neither passed nor failed at base "
                f"({base_passed} of {len(added)} passed, none failed).",
                "They errored or were skipped against the unfixed code. An error is not a\n"
                "demonstrated behavioural constraint - it is usually a missing symbol,\n"
                "which a genuinely new API produces legitimately. Verify by hand that\n"
                "these fail for the intended reason.\n\n" + base_out[-1500:],
            )

        return report(
            FAIL,
            f"All {len(added)} added test(s) PASS against the unfixed code.",
            "They observe behaviour the code already had, so they do not constrain this\n"
            "change. Make at least one fail at the merge base.\n\n"
            "Added tests:\n" + "\n".join(f"  {n}" for n in added) + "\n\n" + base_out[-1200:],
        )
    finally:
        for tree in (base_tree, head_tree):
            git("worktree", "remove", "--force", str(tree), check=False)
        shutil.rmtree(scratch, ignore_errors=True)
        git("worktree", "prune", check=False)


if __name__ == "__main__":
    raise SystemExit(main())
