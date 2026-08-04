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

**Scope, and what this is not.**  The underlying rule -- a regression test must
fail before the fix and pass after it -- is standard practice, taught directly in
Django's and Qt's contribution guides.  What is unusual here is the automation:
this builds a single coarse *historical mutant* by reverting all of ``src/`` to
the merge base.  That is cheap (two focused test runs), and it measures the
genuine unfixed implementation rather than a synthetic fault, which is what makes
it good at catching hollow assertions and tests that merely execute code.  It is
also blunt in two directions: one unrelated failure can kill the mutant, and one
valid test can make a change carrying several independent fixes look fully
constrained.  Real mutation-testing systems -- mutmut and Cosmic Ray in Python,
Stryker, PIT, cargo-mutants elsewhere -- instead inject many small targeted
faults, and cargo-mutants can scope them to a diff; Google has reported running
diff-filtered mutation results into review at scale.  Per-function mutation of
the changed code would be strictly more precise and considerably more expensive.
This script is deliberately the cheap version, and it is advisory.

**The mutant is the head tree with ``src/`` rolled back.**  Three worktrees are
built, all from throwaway temporary directories.  ``head_tree`` and ``base_tree``
are both created at the head revision, and ``base_tree`` then has ``src/``
deleted outright and restored from the merge base; ``base_tests_tree`` sits at
the merge base and is used only to enumerate which test items existed before.
Building the base tree *at the merge base*
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

**A missing symbol is never a kill.**  pytest records an exception raised in the
test *body* as a JUnit ``<failure>``, indistinguishable by tag from an assertion
failure -- so a test whose only claim against the mutant is ``ImportError`` or
``AttributeError`` on a name the change introduced would otherwise be counted as
evidence, and moving an import from module scope into the body would flip
``INCONCLUSIVE`` to ``PASS``.  Failures whose message names an import, attribute
or name error are tracked separately and never count.  This is stricter than it
first looks: a real shipped fix in this repository (``dc0dd57``) adds a private
helper and tests it directly, so every failing item at base is a missing-symbol
error and the honest verdict is ``INCONCLUSIVE`` -- those tests constrain the
helper's existence, not the behaviour of the fix.  Driving the public API instead
would fail behaviourally and earn a ``PASS``.

Two known limitations, reported honestly rather than papered over: a change
containing several independent fixes is accepted once *any* qualifying test is
killed -- mapping each production hunk to its own killed mutation would need
per-hunk mutants and is out of scope, per the note on prior art above.  And a fix
landing entirely in a **new** module cannot be measured this way at all: at the
base revision the module does not exist, so its tests error or raise rather than
fail.  That is reported ``INCONCLUSIVE``, never ``PASS``.

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


def collect_items(tree: Path, test_files: list[str]) -> tuple[set[str], int, str]:
    """Every item ID pytest collects from ``test_files``, with its exit status.

    Item IDs carry the parametrise suffix, which is what makes a *data-driven*
    strengthening visible: appending a case to a module-level list consumed by an
    unchanged ``@pytest.mark.parametrize(..., CASES)`` leaves the function's AST
    byte-identical, so comparing ASTs alone reports NO EVIDENCE for a real new
    regression case.  Comparing collected items catches it.

    The exit code must be checked by the caller.  A module that fails to collect
    contributes nothing here, and silently dropping its candidates would let a
    sibling module's killer carry a change whose other test module is broken.

    ``-o addopts=`` is load-bearing.  This repository sets ``addopts = "-v
    --tb=short"``, and pytest *prepends* those, so the ``-v`` cancels ``-q`` to
    net verbosity 0 -- at which point ``--collect-only`` prints its indented
    tree of ``<Module>``/``<Function>`` nodes instead of flat node IDs, and this
    parser found nothing collectable at all.
    """
    if not test_files:
        return set(), 0, ""
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
    items = {line.strip() for line in proc.stdout.splitlines() if "::" in line}
    return items, proc.returncode, proc.stdout + proc.stderr


def owning_function(item_id: str) -> str:
    """The test function node ID that owns a collected item."""
    return item_id.split("[")[0]


@dataclass
class Outcome:
    """Per-item results for one test function, from the JUnit report."""

    collected: int = 0
    passed: int = 0
    failed: list[str] = field(default_factory=list)
    symbol_errors: list[str] = field(default_factory=list)
    errored: list[str] = field(default_factory=list)
    skipped: int = 0

    @property
    def clean(self) -> bool:
        """Every collected item ran and passed."""
        return self.collected > 0 and self.passed == self.collected

    @property
    def red(self) -> bool:
        return bool(self.failed or self.symbol_errors or self.errored)


# A call-phase exception is a JUnit <failure>, tag-identical to an assertion
# failure -- only the message distinguishes them. These names are what a test
# raises when the symbol it needs does not exist at the base revision, which is
# the new-module case wearing a different hat: not behavioural evidence.
_SYMBOL_ERRORS = (
    "ModuleNotFoundError",
    "ImportError",
    "AttributeError",
    "NameError",
)


def is_symbol_error(message: str) -> bool:
    return message.lstrip().startswith(_SYMBOL_ERRORS)


def junit_key(node_id: str) -> tuple[str, str]:
    """The (classname, function) pair JUnit will report for a node ID."""
    path, *parts = node_id.split("::")
    module = path[:-3].replace("/", ".") if path.endswith(".py") else path.replace("/", ".")
    return ".".join([module, *parts[:-1]]), parts[-1]


def run_pytest(tree: Path, targets: list[str]) -> tuple[dict[str, Outcome], int, str]:
    """Run ``targets`` inside ``tree`` and attribute every item to its function.

    The process status is returned alongside the per-item outcomes and must be
    checked: a session-finish, plugin or internal error leaves every target's
    record clean, so reading only the JUnit report would call a red run green.

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
        return outcomes, proc.returncode, out

    index = {junit_key(t): t for t in targets}
    for case in ET.parse(report_path).getroot().iter("testcase"):
        key = (case.get("classname", ""), (case.get("name") or "").split("[")[0])
        target = index.get(key)
        if target is None:
            continue
        item = case.get("name", "")
        outcome = outcomes[target]
        outcome.collected += 1
        failure = case.find("failure")
        if failure is not None:
            message = failure.get("message") or failure.text or ""
            if is_symbol_error(message):
                outcome.symbol_errors.append(f"{item}: {message.splitlines()[0][:100]}")
            else:
                outcome.failed.append(item)
        elif case.find("error") is not None:
            outcome.errored.append(item)
        elif case.find("skipped") is not None:
            outcome.skipped += 1
        else:
            outcome.passed += 1
    report_path.unlink(missing_ok=True)
    return outcomes, proc.returncode, out


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

    # No early return on an empty AST candidate set: a case appended to a
    # module-level list feeding an unchanged @parametrize leaves every function
    # dump identical, and only the collected-item diff below can see it.

    # Three isolated trees. The caller's checkout is never touched.
    #   head_tree       head src + head tests -- the change as proposed
    #   base_tree       base src + head tests -- the mutant
    #   base_tests_tree the merge base untouched, used only to enumerate which
    #                   test items existed before, so a new parametrise case is
    #                   attributable even when the function's AST is unchanged
    tests_at_base = [p for p in tests_changed if file_at(merge_base, p) is not None]
    scratch = Path(tempfile.mkdtemp(prefix="mutation-gate-"))
    base_tree, head_tree = scratch / "base", scratch / "head"
    base_tests_tree = scratch / "base-tests"
    try:
        git("worktree", "add", "--detach", "--quiet", str(base_tree), head)
        git("worktree", "add", "--detach", "--quiet", str(head_tree), head)
        git("worktree", "add", "--detach", "--quiet", str(base_tests_tree), merge_base)

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

        head_items, collect_status, collect_out = collect_items(head_tree, tests_changed)
        if collect_status != 0:
            return report(
                INCONCLUSIVE,
                "pytest could not collect every changed test module at head.",
                "A module that fails to collect contributes no candidates, and a sibling\n"
                "module's killer must not carry a change whose other test module is\n"
                "broken. Fix collection first.\n\n" + collect_out[-1500:],
            )

        # Item-level attribution catches what the AST cannot: a case appended to
        # a module-level list feeding an unchanged @parametrize decorator.
        base_items, _, _ = collect_items(base_tests_tree, tests_at_base)
        new_item_owners = {owning_function(i) for i in head_items - base_items}

        collected_functions = {owning_function(i) for i in head_items}
        qualifying = sorted(
            (set(candidates) | new_item_owners) & collected_functions,
        )
        uncollectable = sorted(set(candidates) - collected_functions)
        data_driven = sorted(new_item_owners - set(candidates))
        if data_driven:
            print(f"  tests with new cases:     {len(data_driven)}")

        targets = qualifying
        if not targets and not candidates:
            return report(
                NO_EVIDENCE,
                "Test modules changed but nothing was added, strengthened or given a new case.",
                "Attribution needs a test whose body or decorators this change touched, or\n"
                "a new collected item. Renames, moves and formatting do not qualify. Add\n"
                "or tighten a regression test, or apply the mutation-gate-exempt label.\n\n"
                + "\n".join(f"  {p}" for p in tests_changed),
            )
        if not targets:
            return report(
                INCONCLUSIVE,
                f"None of the {len(candidates)} qualifying test(s) are collected by pytest.",
                "ast sees any test-prefixed function, but pytest only collects classes\n"
                "matching Test* and skips those defining __init__.\n\n"
                + "\n".join(f"  {n}" for n in uncollectable),
            )

        head_outcomes, head_status, head_out = run_pytest(head_tree, targets)
        base_outcomes, base_status, base_out = run_pytest(base_tree, targets)

        # Exit status is evidence in its own right. A clean JUnit record says
        # nothing about a session-finish, plugin or internal error, and pytest
        # reports those only through the process status.
        if head_status != 0:
            return report(
                INCONCLUSIVE,
                f"pytest exited {head_status} at the head revision.",
                "Only exit 0 means every target was collected and passed. A non-zero\n"
                "status with clean per-test records is a session, plugin or internal\n"
                "error, and nothing measured under it can be trusted.\n\n" + head_out[-1500:],
            )
        if base_status not in (0, 1):
            return report(
                INCONCLUSIVE,
                f"pytest exited {base_status} against the unfixed code.",
                "Only 0 (all passed) and 1 (tests failed) are meaningful here; 2-5 are\n"
                "interruption, internal error, usage error and no-tests-collected, none\n"
                "of which measures the mutant.\n\n" + base_out[-1500:],
            )

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

        # Only a real call-phase failure is a kill. Anything that did not
        # cleanly pass at base and was not killed is UNMEASURED, not survived --
        # enumerating the ways it can be unmeasured and defaulting the rest to
        # "passed" is how an unobserved outcome becomes a verdict. A module that
        # fails to import against the rolled-back src raises a *collection*
        # error, which JUnit files against the module rather than the test and
        # leaves the record empty; an in-test importorskip skips instead; a
        # teardown error leaves the call passing. Only `.clean` proves survival.
        killed = [t for t in evaluable if base_outcomes[t].failed]
        unmeasured = [t for t in evaluable if not base_outcomes[t].clean and t not in killed]

        def why_unmeasured(node: str) -> str:
            outcome = base_outcomes[node]
            if outcome.symbol_errors:
                return "; ".join(outcome.symbol_errors)
            parts = []
            if outcome.errored:
                parts.append(f"{len(outcome.errored)} errored")
            if outcome.skipped:
                parts.append(f"{outcome.skipped} skipped")
            if not outcome.collected:
                parts.append("never collected")
            return ", ".join(parts) or "did not run"

        pending = (
            "\n\nNot measured against the unfixed code:\n"
            + "\n".join(f"  {n}\n      {why_unmeasured(n)}" for n in unmeasured)
            if unmeasured
            else ""
        )

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
                + note
                + pending,
            )

        if unmeasured:
            reasons = "\n".join(f"  {n}\n      {why_unmeasured(n)}" for n in unmeasured)
            return report(
                INCONCLUSIVE,
                f"{len(unmeasured)} evaluable test(s) neither passed nor failed against "
                "the unfixed code.",
                "They errored, were skipped, or never ran. None of those is a demonstrated\n"
                "behavioural constraint - a missing symbol is what a genuinely new module\n"
                "produces legitimately, and a call-phase ImportError is the same case\n"
                "wearing the costume of a failure. Verify by hand that these fail for the\n"
                "intended reason.\n\n" + reasons + note + "\n\n" + base_out[-1500:],
            )

        return report(
            FAIL,
            f"All {len(evaluable)} evaluable test(s) PASS against the unfixed code.",
            "They observe behaviour the code already had, so they do not constrain this\n"
            "change. Make at least one fail at the merge base.\n\n"
            "Evaluated:\n" + "\n".join(f"  {n}" for n in evaluable) + note,
        )
    finally:
        for tree in (base_tree, head_tree, base_tests_tree):
            git("worktree", "remove", "--force", str(tree), check=False)
        shutil.rmtree(scratch, ignore_errors=True)
        git("worktree", "prune", check=False)


if __name__ == "__main__":
    raise SystemExit(main())
