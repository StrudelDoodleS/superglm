"""Assert that a change's own regression tests fail against the unfixed code.

`AGENTS.md` requires that "adversarial regressions include a mutation check or a
demonstration against the unfixed implementation".  This script enforces that
sentence mechanically, so it stops being a convention nobody can verify.

The check is two questions, and both must hold:

1. do the tests this change added or strengthened **pass** at the head revision,
   and
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

**The mutant is the head tree with ``src/`` rolled back.**  Both worktrees are
created at the head revision; the base tree then has ``src/`` deleted outright
and restored from the merge base.  Building the base tree *at the merge base*
instead left every non-``src`` file -- ``pyproject.toml``, ``conftest.py``,
shared helpers, ``scripts/`` -- at the base revision, so the two runs differed by
more than the fix.  Overlaying with ``git checkout <rev> -- <dir>`` is not a
replace either: it writes the source revision's paths but does not delete paths
that exist only in the target, so a fix landing in a *new* module was never
mutated and a deleted ``conftest.py`` stayed live.  Deleting first is what makes
it an actual revert.

**The caller's checkout is never modified.**  In-place mutation risked destroying
uncommitted work, and a signal during the restore could leave the tree
half-mutated.  Uncommitted changes under ``src/`` or ``tests/`` are refused up
front rather than silently ignored: the diff is ``merge_base..head``, so a fix
still sitting in the working tree is invisible to it and would otherwise be
reported ``SKIP`` with exit 0.

**Attribution, not counting.**  The base and head revisions of each changed test
module are parsed with ``ast``.  A test function counts as evidence if it was
added, or if its body or decorators changed -- comparing ``ast.dump`` output
detects a strengthened assertion without any diff-line bookkeeping, and ignores
a pure move.  Requiring merely that *something* in the changed modules fails is
satisfied by an untouched neighbour.

Crediting a *modified* test is a deliberate trade with a cost worth stating: the
AST cannot distinguish an assertion that was tightened adversarially from one
mechanically updated to match the new behaviour, and both fail against the
rolled-back ``src/``.  So a change that edits an expected constant earns a
``PASS`` on that edit alone.  The alternative -- crediting only added tests --
was worse: tightening an existing assertion is the textbook fix for a test that
was too weak (it is how two of the three examples above should have been fixed),
and refusing it left the blanket ``mutation-gate-exempt`` label as the only
remedy, which disables the check for the whole change including the production
edits that do need constraining.

**Outcomes come from ``--junit-xml``, not from the terminal summary.**  Scraping
``N passed`` counts *items*, while ``ast`` counts *function definitions*; the two
differ for every parametrised test, and this repository has over 400 parametrise
sites.  The mismatch made a correct change report "Only 3 of 1 added test(s)
pass".  The XML carries one record per collected item with its own
failure/error/skipped state, so each test function is judged on its own items and
a teardown error can no longer hide behind a passing call.

**Targets are intersected with what pytest actually collects.**  ``ast`` sees any
``test``-prefixed function, but pytest only collects classes matching ``Test*``
and skips those defining ``__init__``.  Passing an uncollectable node ID is a
*usage* error: pytest exits 4 having run nothing, so one bad ID would zero the
result for every other test in the change.

**Non-evaluable tests are partitioned out, not fatal.**  A test that is skipped
on this runner -- ``browser``-marked, or ``skipif``-guarded on a platform -- is
not evidence, but neither does it invalidate its siblings.  Those are excluded
and named; the remaining tests are still judged.

One known limitation, reported honestly rather than papered over: a change
containing several independent fixes is accepted once *any* qualifying test is
killed.  Mapping each production hunk to its own killed mutation would need
per-hunk mutants and is deliberately out of scope.  A fix landing entirely in a
**new** module is a related case -- at the base revision the module does not
exist, so its tests error rather than fail, and an ``ImportError`` is not
evidence that a test constrains behaviour.  That is reported ``INCONCLUSIVE``,
never ``PASS``.

Outcomes
--------
``SKIP``          no production change to mutate.                     exit 0
``PASS``          a qualifying test fails at base and passes at head. exit 0
``FAIL``          every qualifying test passes against the unfixed code. exit 1
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
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = "src"
PACKAGE = "src/superglm"
TESTS = "tests"

SKIP, PASS, FAIL, NO_EVIDENCE, INCONCLUSIVE = (
    "SKIP",
    "PASS",
    "FAIL",
    "NO EVIDENCE",
    "INCONCLUSIVE",
)


def git(*args: str, cwd: Path | None = None, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args], cwd=cwd or ROOT, capture_output=True, text=True, check=False
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed:\n{result.stderr.strip()}")
    return result.stdout.strip()


def is_test_module(path: str) -> bool:
    """Match pytest's default discovery patterns, both of them.

    ``python_files`` defaults to ``test_*.py`` *and* ``*_test.py`` and this
    repository does not override it, so recognising only the prefix form would
    report NO EVIDENCE for a change whose regression test is named the other way.
    """
    name = Path(path).name
    return (
        path.startswith(f"{TESTS}/")
        and name.endswith(".py")
        and (name.startswith("test_") or name.endswith("_test.py"))
    )


def file_at(rev: str, path: str) -> str | None:
    """A file's contents at ``rev``, or None when it does not exist there."""
    result = subprocess.run(
        ["git", "show", f"{rev}:{path}"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout if result.returncode == 0 else None


def test_functions(source: str | None) -> dict[str, str]:
    """Map each test function's class-qualified name to its structural dump.

    ``ast.dump`` omits line numbers, so a function that merely moved compares
    equal while a changed assertion, body or decorator does not.  That is what
    lets a *strengthened* existing test count as evidence without mapping diff
    hunks back to node IDs.
    """
    if source is None:
        return {}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}

    found: dict[str, str] = {}

    def walk(node: ast.AST, prefix: str) -> None:
        for child in getattr(node, "body", []):
            if isinstance(child, ast.ClassDef):
                walk(child, f"{prefix}{child.name}::")
            elif isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                if child.name.startswith("test"):
                    found[f"{prefix}{child.name}"] = ast.dump(child)

    walk(tree, "")
    return found


def candidate_node_ids(merge_base: str, head: str, test_files: list[str]) -> tuple[list[str], int]:
    """Node IDs this change added or strengthened, and how many were added.

    A modified test counts: tightening an assertion so it fails against the
    unfixed code is the textbook fix for a test that was too weak, and refusing
    to credit it would leave the blanket ``mutation-gate-exempt`` label as the
    only remedy -- turning the gate off for the whole change, including the
    production edits that do need constraining.
    """
    node_ids: list[str] = []
    added_count = 0
    for path in test_files:
        before = test_functions(file_at(merge_base, path))
        after = test_functions(file_at(head, path))
        qualifying = sorted(name for name, dump in after.items() if before.get(name) != dump)
        added_count += sum(1 for name in qualifying if name not in before)
        node_ids += [f"{path}::{name}" for name in qualifying]
    return node_ids, added_count


def collectable(
    tree: Path, test_files: list[str], candidates: list[str]
) -> tuple[list[str], list[str]]:
    """Split ``candidates`` into those pytest collects and those it does not.

    Passing an uncollectable node ID makes pytest exit 4 as a usage error
    without running anything, which would discard every other result in the run.

    ``-o addopts=`` is load-bearing.  This repository sets ``addopts = "-v
    --tb=short"``, and pytest *prepends* those, so the ``-v`` cancels ``-q`` to
    net verbosity 0 -- at which point ``--collect-only`` prints its indented
    tree of ``<Module>``/``<Function>`` nodes instead of flat node IDs, and this
    parser found nothing collectable at all.
    """
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            *test_files,
            "--collect-only",
            "-o",
            "addopts=",
            "-q",
            "--no-header",
            "-p",
            "no:cacheprovider",
        ],
        cwd=tree,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONPATH": str(tree / SRC)},
    )
    collected = {line.strip().split("[")[0] for line in proc.stdout.splitlines() if "::" in line}
    keep = [c for c in candidates if c in collected]
    drop = [c for c in candidates if c not in collected]
    return keep, drop


@dataclass
class Outcome:
    """Per-item results for one test function, from the JUnit report."""

    collected: int = 0
    passed: int = 0
    failed: list[str] = field(default_factory=list)
    errored: list[str] = field(default_factory=list)
    skipped: int = 0

    @property
    def clean(self) -> bool:
        """Every collected item ran and passed."""
        return self.collected > 0 and self.passed == self.collected

    @property
    def red(self) -> bool:
        return bool(self.failed or self.errored)


def junit_key(node_id: str) -> tuple[str, str]:
    """The (classname, function) pair JUnit will report for a node ID."""
    path, *parts = node_id.split("::")
    module = path[:-3].replace("/", ".") if path.endswith(".py") else path.replace("/", ".")
    return ".".join([module, *parts[:-1]]), parts[-1]


def run_pytest(tree: Path, targets: list[str]) -> tuple[dict[str, Outcome], str]:
    """Run ``targets`` inside ``tree`` and attribute every item to its function.

    ``PYTHONPATH`` must point at this tree's ``src`` or the editable install's
    ``.pth`` would import the caller's checkout and every run would measure the
    same unmutated code.  The import is asserted below rather than assumed:
    ``site`` appends ``.pth`` paths *after* ``PYTHONPATH`` today, but a packaging
    backend that installed a meta-path finder instead would silently make both
    runs measure head code and report FAIL on every change.
    """
    report_path = tree / ".mutation-gate-report.xml"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            *targets,
            "-o",
            "addopts=",
            "-p",
            "no:cacheprovider",
            "--no-header",
            "--tb=no",
            "-q",
            f"--junit-xml={report_path}",
        ],
        cwd=tree,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONPATH": str(tree / SRC)},
    )
    out = proc.stdout + proc.stderr

    outcomes = {t: Outcome() for t in targets}
    if not report_path.exists():
        return outcomes, out

    index = {junit_key(t): t for t in targets}
    for case in ET.parse(report_path).getroot().iter("testcase"):
        key = (case.get("classname", ""), (case.get("name") or "").split("[")[0])
        target = index.get(key)
        if target is None:
            continue
        item = case.get("name", "")
        outcome = outcomes[target]
        outcome.collected += 1
        if case.find("failure") is not None:
            outcome.failed.append(item)
        elif case.find("error") is not None:
            outcome.errored.append(item)
        elif case.find("skipped") is not None:
            outcome.skipped += 1
        else:
            outcome.passed += 1
    report_path.unlink(missing_ok=True)
    return outcomes, out


def imported_from(tree: Path) -> str | None:
    """Where ``import superglm`` resolves inside ``tree``, or None if it fails."""
    proc = subprocess.run(
        [sys.executable, "-c", "import superglm; print(superglm.__file__)"],
        cwd=tree,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONPATH": str(tree / SRC)},
    )
    # A genuinely absent package at the base revision is handled downstream.
    return proc.stdout.strip() if proc.returncode == 0 else None


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
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Run even with uncommitted changes under src/ or tests/ (they are NOT measured).",
    )
    args = parser.parse_args()

    if os.environ.get("MUTATION_GATE_EXEMPT", "").strip():
        return report(
            SKIP,
            "Exempted by the mutation-gate-exempt label.",
            "The justification belongs in the pull-request body, where review can see it.",
        )

    dirty = [
        line for line in git("status", "--porcelain", "--", SRC, TESTS).splitlines() if line.strip()
    ]
    if dirty and not args.allow_dirty:
        return report(
            INCONCLUSIVE,
            f"{len(dirty)} uncommitted change(s) under src/ or tests/.",
            "This gate compares two committed revisions, so uncommitted work is invisible\n"
            "to it -- a fix still in the working tree would be reported SKIP with exit 0,\n"
            "which reads as a pass. Commit first, or pass --allow-dirty to measure only\n"
            "what is committed.\n\n" + "\n".join(f"  {line}" for line in dirty),
        )

    merge_base = git("merge-base", args.base_ref, args.head_ref)
    head = git("rev-parse", args.head_ref)
    changed = [
        line for line in git("diff", "--name-only", f"{merge_base}..{head}").splitlines() if line
    ]
    src_changed = [p for p in changed if p.startswith(PACKAGE)]
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

    candidates, added_count = candidate_node_ids(merge_base, head, tests_changed)
    print(f"mutation gate: base {merge_base[:12]}  head {head[:12]}")
    print(f"  production files changed: {len(src_changed)}")
    print(f"  test modules changed:     {len(tests_changed)}")
    print(f"  tests added:              {added_count}")
    print(f"  tests strengthened:       {len(candidates) - added_count}")

    if not candidates:
        return report(
            NO_EVIDENCE,
            "Test modules changed but no test function was added or strengthened.",
            "Attribution needs a test whose body or decorators this change touched.\n"
            "Renames, moves and formatting do not qualify. Add or tighten a regression\n"
            "test, or apply the mutation-gate-exempt label.\n\n"
            + "\n".join(f"  {p}" for p in tests_changed),
        )

    # Two isolated trees, both at head. The caller's checkout is never touched.
    scratch = Path(tempfile.mkdtemp(prefix="mutation-gate-"))
    base_tree, head_tree = scratch / "base", scratch / "head"
    try:
        git("worktree", "add", "--detach", "--quiet", str(base_tree), head)
        git("worktree", "add", "--detach", "--quiet", str(head_tree), head)

        # Roll src/ back by REPLACING it: delete first so head-only modules go
        # away, then restore the merge base's tree exactly. Everything else --
        # tests, conftest, pyproject - stays at head, so the only difference
        # between the two runs is the fix itself.
        shutil.rmtree(base_tree / SRC, ignore_errors=True)
        git("checkout", merge_base, "--", SRC, cwd=base_tree, check=False)

        # Total, silent failure mode if it ever breaks: both runs would measure
        # head code and every change would be reported FAIL.
        resolved = imported_from(head_tree)
        if resolved is not None and not resolved.startswith(str(head_tree)):
            return report(
                INCONCLUSIVE,
                "pytest would import superglm from outside the mutant tree.",
                f"PYTHONPATH was ignored; superglm resolved to:\n  {resolved}\n\n"
                "An installed distribution is shadowing the worktree, so neither run\n"
                "would measure the revision it claims to.",
            )

        targets, uncollectable = collectable(head_tree, tests_changed, candidates)
        if not targets:
            return report(
                INCONCLUSIVE,
                f"None of the {len(candidates)} qualifying test(s) are collected by pytest.",
                "ast sees any test-prefixed function, but pytest only collects classes\n"
                "matching Test* and skips those defining __init__.\n\n"
                + "\n".join(f"  {n}" for n in uncollectable),
            )

        head_outcomes, head_out = run_pytest(head_tree, targets)
        base_outcomes, base_out = run_pytest(base_tree, targets)

        # Partition rather than bail: a test skipped on this runner is not
        # evidence, but it does not invalidate its siblings either.
        evaluable = [t for t in targets if head_outcomes[t].clean]
        red_at_head = [t for t in targets if head_outcomes[t].red]
        unmeasurable = [
            t for t in targets if not head_outcomes[t].clean and not head_outcomes[t].red
        ]

        if red_at_head:
            return report(
                INCONCLUSIVE,
                f"{len(red_at_head)} qualifying test(s) do not pass at the head revision.",
                "That is a defect in this change, not a limit of the gate: a test that is\n"
                "red at head demonstrates nothing. Make them pass first.\n\n"
                + "\n".join(f"  {n}" for n in red_at_head)
                + "\n\n"
                + head_out[-1500:],
            )

        if not evaluable:
            return report(
                INCONCLUSIVE,
                f"None of the {len(targets)} qualifying test(s) could be evaluated.",
                "They were skipped or never ran on this runner - browser-marked or\n"
                "platform-guarded tests cannot demonstrate anything here.\n\n"
                + "\n".join(f"  {n}" for n in unmeasurable + uncollectable)
                + "\n\n"
                + head_out[-1500:],
            )

        excluded = unmeasurable + uncollectable
        note = (
            "\n\nExcluded as non-evaluable:\n" + "\n".join(f"  {n}" for n in excluded)
            if excluded
            else ""
        )

        killed = [t for t in evaluable if base_outcomes[t].failed]
        if killed:
            return report(
                PASS,
                f"{len(killed)} of {len(evaluable)} evaluable test(s) fail against the "
                "unfixed code and pass at head, as required.",
                "\n".join(
                    f"  {n}"
                    + "".join(
                        f"\n      {i}"
                        for i in base_outcomes[n].failed
                        if i != n.rsplit("::", 1)[-1]
                    )
                    for n in killed
                )
                + note,
            )

        # A target with NO record at base did not survive the mutant -- it was
        # never measured. A module that fails to import against the rolled-back
        # src raises a *collection* error, which JUnit files against the module
        # rather than the test, so it matches no target and leaves the outcome
        # empty. Reading that as "passed at base" is how an unobserved result
        # becomes a FAIL against a correct change.
        unmeasured = [
            t for t in evaluable if base_outcomes[t].errored or base_outcomes[t].collected == 0
        ]
        if unmeasured:
            return report(
                INCONCLUSIVE,
                f"{len(unmeasured)} evaluable test(s) errored or never ran against the "
                "unfixed code, rather than failing.",
                "An error is not a demonstrated behavioural constraint - it is usually a\n"
                "missing symbol, which a genuinely new module produces legitimately.\n"
                "Verify by hand that these fail for the intended reason.\n\n"
                + "\n".join(f"  {n}" for n in unmeasured)
                + note
                + "\n\n"
                + base_out[-1500:],
            )

        return report(
            FAIL,
            f"All {len(evaluable)} evaluable test(s) PASS against the unfixed code.",
            "They observe behaviour the code already had, so they do not constrain this\n"
            "change. Make at least one fail at the merge base.\n\n"
            "Evaluated:\n" + "\n".join(f"  {n}" for n in evaluable) + note,
        )
    finally:
        for tree in (base_tree, head_tree):
            git("worktree", "remove", "--force", str(tree), check=False)
        shutil.rmtree(scratch, ignore_errors=True)
        git("worktree", "prune", check=False)


if __name__ == "__main__":
    raise SystemExit(main())
