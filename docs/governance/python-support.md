# Python support policy

**The floor tracks NumPy.** `requires-python` is whatever the current NumPy
release requires, and the CI matrix covers that floor through the newest
released CPython. When NumPy raises its floor, superglm raises its floor in the
same release cycle.

Current: **`>=3.12`** (NumPy 2.5, SciPy 1.18).

## Why this rule and not a calendar

superglm's numerical results *are* NumPy and SciPy. Coefficients come out of
LAPACK, and acceptance decisions are made against thresholds near machine
precision. A Python version we support but that upstream has abandoned is not a
version lag — it is a **second numerical platform**, frozen forever, that every
correctness claim would have to be re-established on.

That is not hypothetical. Python 3.10 caps at NumPy 2.2.6 / SciPy 1.15.3 /
pandas 2.3.3, permanently. On that stack a SCOP fit lands on the opposite side
of a certification threshold from the same fit on NumPy 2.4.2 — see issue #179,
where identical code passes on one and raises on the other. Supporting 3.10
meant claiming a platform we were not validating.

Compiled libraries can afford a much wider window — LightGBM supports `>=3.10`
because its numbers come from its own C++ and NumPy only marshals arrays there.
The width of the window a project can afford is proportional to how little of
its correctness lives in its dependencies' numerics. Ours lives almost entirely
there.

Tracking NumPy also makes the frozen-platform failure structurally impossible:
there is only ever one numerical stack in the support matrix.

## How this compares to the ecosystem

[SPEC 0](https://scientific-python.org/specs/spec-0000/), the scientific-Python
convention that replaced NEP 29, recommends dropping a Python version three
years after its release. Following NumPy is very close to SPEC 0 in practice and
never lags it — as of 2026-07 both give `>=3.12`.

CPython's own support window is five years, which is why an EOL date is a poor
trigger: by the time CPython retires a version, the numerical stack has been
frozen on it for roughly two years.

## The annual change

Each October a new CPython ships and NumPy typically raises its floor. The
change is mechanical:

1. Check NumPy's `requires-python` on PyPI.
2. If it moved, set `requires-python` in `pyproject.toml` to match.
3. Update `[tool.ruff] target-version`, the trove classifiers, the README badge,
   and the `ci.yml` matrix (floor → newest release).
4. Run the full suite, then `ruff check` — a raised `target-version` enables
   lint rules for newer syntax and may surface modernisations.
5. Declare it `release:minor`: dropping a Python version is a compatibility
   change.

Dropping a version is not a break for anyone already on it. Their installed
superglm keeps working; a resolver simply stops offering them newer releases,
which is what NumPy did to 3.10 users at 2.3 and pandas at 3.0.

## History

| date | floor | trigger |
|---|---|---|
| 2026-07-31 | `>=3.12` | Policy adopted. Dropped 3.10 and 3.11 together — 3.11 was already past SPEC 0 (released 2022-10), and keeping it would have repeated this in October. Prompted by issue #179, an intermittent 3.10-only failure caused by frozen numerics. |
