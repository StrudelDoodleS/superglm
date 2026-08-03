# Fixing a tracked issue

A repeatable process for turning an issue into a merged fix, and a prompt you can
hand to an agent to run it.

The shape of it is: **reproduce, constrain, fix, enumerate, verify.** The two
steps people skip are *constrain* (write a test that fails first) and *enumerate*
(find the siblings), and those are the two that decide whether the fix holds.

---

## 1. Read the issue properly

- Read the **comments as well as the body**. Evidence and corrections accumulate
  there, and review bots post findings as summary comments rather than inline
  threads — zero unresolved threads does not mean zero findings.
- **Treat the issue as evidence, not ground truth.** Check whether its premises
  still hold. Real examples from this tracker: #200 deferred work on the grounds
  that no serialization path existed, when `pickle` is the documented deployment
  artifact; #206 deferred on the grounds that another issue would subsume it,
  and that issue closed without introducing the subsuming change.
- **Line numbers in issue bodies go stale.** Re-locate every symbol at `HEAD`
  before you trust a citation.

## 2. Reproduce before you change anything

Confirm the defect exists at `HEAD` and capture the exact command and output. If
it does not reproduce, that is the finding — say so in the issue rather than
fixing something that is already gone.

Run under the audited source, and prove which source you are on:

```bash
uv run python -c "import superglm; print(superglm.__file__)"
```

Always run a **control** beside the failing case: the same input on a sibling
family, link, backend or code path that should work. Without a control you
cannot tell a real defect from a bad fixture. Several candidate findings in the
2026-08 audit died at this step — the numbers were real, the mechanism was
misattributed.

## 3. Write the failing test first, and prove it fails

This is the step that makes the fix durable, and it is checked by CI
(`scripts/mutation_gate.py`, `.github/workflows/mutation-gate.yml`) — advisory
until that check is added to the ruleset.

```bash
# with your test written but the fix NOT yet applied
uv run pytest tests/test_your_area.py -q      # must FAIL
```

Then, once the fix is in, check the test still constrains it:

```bash
uv run python scripts/mutation_gate.py --base-ref origin/master
```

That reverts `src/` to the merge base, keeps your tests, and requires at least
one to fail. **A test that passes against the unfixed code is not a regression
test** — it observes what the code already did. Three shipped examples:

- `tests/test_shape_fit.py` asserts one-sided second-derivative bounds that an
  identically zero curve satisfies, so it stayed green while the constrained
  smooth collapsed to a flat line;
- a per-iteration feasibility guard survived mutation against all 272 tests in
  its own five test files, while the same harness killed nine sibling mutations;
- a BLAS dedup guard shipped with a test whose fixture never reached the path.

Follow the numerical test policy in `AGENTS.md`: assert certified invariants —
rank, subspace, residual, reconstruction, prediction, backward error — not the
sign or magnitude of BLAS roundoff. Derive tolerances from dimensions, dtype
epsilon, norms and conditioning.

## 4. Fix it

Match the surrounding code's idiom, naming and comment density. Preserve
mathematical names where they make the numerics clearer.

## 5. Enumerate the siblings — mechanically

This is where the yield is. When you fix one call site, **grep every sibling and
classify each one**; do not reason about which ones "probably" matter.

- one unguarded call site → enumerate every call of that function
- one wrong threshold → find every place deriving or comparing a related one
- one unbudgeted allocation → inspect every analogous loop and list the
  *simultaneously live* buffers, not just the one being budgeted
- one missing `isinstance` branch → assert the registry is exhaustive, so the
  next addition cannot silently inherit the bug

The last one is the pattern worth copying: a *completeness test* over a registry
prevents recurrence in a way that fixing the two known members does not.

## 6. Verify before you claim anything

```bash
uv run pytest tests/ -q -m "not slow"
uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/
uv run python scripts/mutation_gate.py --base-ref origin/master
```

Performance-sensitive work compares complete-fit timing, peak memory, numerical
outputs **and actual backend dispatch** against the baseline. Measure serially on
a quiet machine — confirm `uptime` load is below 1.0 first and record it, pin
BLAS threads, report best-of-N with the spread. A number taken on a loaded box
measures contention. Never claim a speedup from a nonconverged or behaviourally
different result.

## 7. Open the PR

- Declare **exactly one** release impact with a rationale: `release:none`,
  `release:patch`, or `release:minor`. `patch`/`minor` include the exact version
  in the same PR. Only one release-bearing PR may advance from a published
  version at a time — check `git ls-remote --tags origin` against the version in
  `pyproject.toml` before assuming you are clear.
- Reference the issue with **`Refs #N`**, not `Fixes #N`, unless you intend the
  merge to close it. Negated phrasing does not help: "Does not fix #N" still
  closes #N. Verify the issue's state after merging.
- Request review from **both** bots, as two separate comments. A PR reviewed by
  only one reads as clean when it is half-reviewed.
- Post the reproduction and the mutation-gate output in the PR body. That is the
  evidence a reviewer needs and cannot reconstruct.

---

## Prompt template

Paste this at an agent, substituting the issue number.

```text
Fix issue #N in this repository.

Work on a branch off origin/master. Do not push or open a PR without asking.

1. READ
   gh issue view N --repo StrudelDoodleS/superglm --json title,body,comments
   Read every comment, not just the body. Treat the issue as evidence, not
   ground truth: check whether its premises still hold at HEAD, and re-locate
   every cited symbol, because line numbers in issue bodies are stale.

2. REPRODUCE
   Reproduce the defect at HEAD before changing anything. Print
   superglm.__file__ to prove which source you are on. Run a CONTROL beside the
   failing case — the same input on a sibling family/link/backend/path that
   should work. If it does not reproduce, stop and report that instead.

3. CONSTRAIN
   Write the regression test FIRST and show me it fails against the unfixed
   code. Assert certified invariants (rank, subspace, residual, reconstruction,
   prediction, backward error), not the sign or magnitude of roundoff. Derive
   tolerances from dimensions, dtype epsilon, norms and conditioning.

4. FIX
   Match the surrounding code's idiom and naming.

5. ENUMERATE
   Mechanically grep every sibling of whatever you fixed — every call site of
   that function, every related threshold, every analogous loop — and classify
   each. Report the enumeration, including the sites you judged safe and why.
   Where the defect was a missing case in a registry, add a completeness test
   over that registry rather than only fixing the known members.

6. VERIFY
   uv run pytest tests/ -q -m "not slow"
   uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/
   uv run python scripts/mutation_gate.py --base-ref origin/master
   Paste the real output. Do not claim anything passes without showing it.

Report: what reproduced, the control, the enumeration and its verdicts, the
mutation-gate result, and anything you could NOT verify. Negative results and
"I could not confirm X" are wanted, not failures.
```

The last paragraph matters more than it looks. Most bad agent output comes from
the model wanting to close the loop; asking explicitly for negative results and
unverified claims is what makes them appear.
