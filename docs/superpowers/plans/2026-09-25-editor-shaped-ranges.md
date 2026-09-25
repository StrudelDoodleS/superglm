# Shaped Ranges Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the editor's Breaks mode with range-pinned polynomials: select a range, choose Flat/Line/Quadratic/Cubic, refit; the rest of the term stays the REML-penalised spline. Also add a searchable left-hand feature list.

**Architecture:** A new library option `Spline(polynomial_ranges=[PolynomialRange(lo, hi, degree)])` for `bs`/`cr` splines: range edges become knots (repeated for a kink), knots inside a range are dropped so each range is one polynomial piece, linear equality rows pin that piece's degree and are absorbed by the existing `_SplineBase._apply_constraints` null-space hook, and the integrated-derivative penalty skips the pinned intervals. The editor builds these specs through the existing structural path (`_refit_replacing` → `StructuralStep` → Restore). The frontend adds four palette icons and a range overlay and deletes Breaks mode.

**Tech Stack:** Python 3.13, numpy/scipy (`scipy.interpolate.BSpline`), FastAPI editor server, vanilla ES-module frontend, pytest + xdist, Playwright browser tests, node test runner for `tests/editor_frontend`.

**Spec:** `docs/superpowers/specs/2026-09-25-editor-shaped-ranges-design.md` (read it first).

## Global Constraints

- Work only in `/home/max/projects/superglm/.worktrees/editor-structural-tools` (branch `feat/editor-structural-tools`). Never `cd` to `/home/max/projects/superglm`. Never `git stash`. `git add` explicit paths only. Never touch `pyproject.toml`'s version, `superglm.__version__` or `uv.lock`.
- Surgical edits (Edit tool / in-place replacements); never re-emit a whole existing file.
- Code shape: no nested loops outside compiled kernels, no meaningless guards, shallow nesting. Source lines must buy behaviour; tests may be verbose. Report source and test LOC separately.
- Editor browser responses carry only intentional messages (`editor/errors.py`): a library `ValueError` becomes one fixed sentence; backend exception text never reaches the browser.
- Editor UI must not intrude: no selection-triggered popups; each action is its own compact palette icon; caveats in hover and Help.
- Tests assert mathematical invariants with tolerances derived from dimensions and float64 epsilon; no wall-clock assertions. Every regression test must fail against the unfixed code (mutation check).
- Focused tests per task; never run the suite serially. Full suite once at the end: `uv run python scripts/run_test_suite.py -m "not browser and not docs"`.
- Cite papers and public docs only (Wood 2016 arXiv:1605.02446; de Boor, *A Practical Guide to Splines*; Wood 2017 *GAMs* §1.8.1). Never read GPL source (mgcv).
- The plan/spec directory is gitignored: `git add -f docs/superpowers/...`.

## Review Focus

1. A range whose edge is the data boundary (the user selects through the last point) — the edge must not be inserted as a knot, and on `cr` a Flat/Line range at the boundary must not stack a dependent natural-boundary row. Test in Task 2.
2. A selection with fewer than `degree + 1` distinct x values (one point; two points for a Quadratic) — a named refusal, never a singular fit or a crash. Tests in Task 2 (library) and Task 5 (editor message).
3. A second shape on an already-shaped term — the free part's knots must not move (build from `fitted_base_knots`), an overlapping range is refused by name, and the same range with a new degree replaces the old one. Tests in Task 5.
4. Restore after two shapes, then Revert to original — each step undoes exactly one shape; revert restores the opened model's predictions bit-for-bit. Test in Task 5.
5. Binned (`discrete=True`) and weighted fits — the pinned piece must still be an exact polynomial (the demo book fits with `discrete=True, n_bins=512` and exposure weights). Test in Task 2.

---

### Task 1: Pure range geometry — `PolynomialRange`, edge knots, pinning rows, null space

**Files:**
- Create: `src/superglm/features/_spline_ranges.py`
- Test: `tests/test_spline_polynomial_ranges.py`

**Interfaces:**
- Produces:
  - `PolynomialRange(lo: float | str, hi: float | str, degree: int, join: str = "kink")` frozen dataclass (public; exported from `superglm` in Task 2).
  - `merged_interior_knots(base: NDArray, ranges: Sequence[PolynomialRange], degree: int, lo: float, hi: float) -> NDArray` — base interior knots with every knot inside a closed range (within `1e-9 * (hi - lo)`) dropped, plus each interior range edge repeated `degree` times (`kink`) or once (`smooth`); a shared edge takes the larger multiplicity; edges on `lo`/`hi` are not added.
  - `pinning_rows(knots: NDArray, degree: int, ranges: Sequence[PolynomialRange]) -> NDArray` shape `(n_rows, n_basis)`.
  - `pinned_intervals(ranges) -> list[tuple[float, float]]`.
  - `constraint_null_space(C: NDArray) -> NDArray` — orthonormal Z with `C @ Z = 0`, from the complete QR of `C.T`; raises `ValueError("polynomial range constraints are dependent ...")` if `C` is not certified full row rank (smallest |R_ii| below `max(C.shape) * eps * max|R_ii|`).
  - `validate_ranges(ranges, degree, lo, hi) -> tuple[PolynomialRange, ...]` sorted by `lo`; refusals below.

- [ ] **Step 1: Write the failing tests**

```python
"""Polynomial ranges on B-splines: geometry, pinning rows and their null space."""

import numpy as np
import pytest
from scipy.interpolate import BSpline

from superglm.features._spline_ranges import (
    PolynomialRange,
    constraint_null_space,
    merged_interior_knots,
    pinning_rows,
    validate_ranges,
)

LO, HI, DEGREE = 0.0, 10.0, 3


def _clamped(interior):
    return np.concatenate([[LO] * (DEGREE + 1), interior, [HI] * (DEGREE + 1)])


def _random_member(knots, Z, seed):
    theta = np.random.default_rng(seed).normal(size=Z.shape[1])
    return BSpline(knots, Z @ theta, DEGREE)


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
@pytest.mark.parametrize("join", ["kink", "smooth"])
def test_every_member_is_a_polynomial_of_the_range_degree(degree, join):
    ranges = validate_ranges([PolynomialRange(3.0, 6.0, degree, join)], DEGREE, LO, HI)
    interior = merged_interior_knots(np.linspace(1, 9, 9), ranges, DEGREE, LO, HI)
    knots = _clamped(interior)
    Z = constraint_null_space(pinning_rows(knots, DEGREE, ranges))
    grid = np.linspace(3.0, 6.0, 41)
    for seed in range(5):
        values = _random_member(knots, Z, seed)(grid)
        coeffs = np.polynomial.polynomial.polyfit(grid, values, degree)
        residual = values - np.polynomial.polynomial.polyval(grid, coeffs)
        scale = max(1.0, float(np.max(np.abs(values))))
        assert np.max(np.abs(residual)) <= 1e3 * np.finfo(float).eps * scale


def test_knots_inside_a_range_are_dropped_and_kink_edges_repeat():
    ranges = validate_ranges([PolynomialRange(3.0, 6.0, 1)], DEGREE, LO, HI)
    interior = merged_interior_knots(np.array([2.0, 3.0, 4.0, 5.0, 7.0]), ranges, DEGREE, LO, HI)
    np.testing.assert_array_equal(interior, [2.0, 3.0, 3.0, 3.0, 6.0, 6.0, 6.0, 7.0])


def test_boundary_edges_are_not_inserted():
    ranges = validate_ranges([PolynomialRange(LO, 4.0, 0)], DEGREE, LO, HI)
    interior = merged_interior_knots(np.array([2.0, 5.0, 8.0]), ranges, DEGREE, LO, HI)
    np.testing.assert_array_equal(interior, [4.0, 4.0, 4.0, 5.0, 8.0])


def test_kink_join_is_continuous_and_may_change_slope():
    ranges = validate_ranges([PolynomialRange(3.0, 6.0, 1)], DEGREE, LO, HI)
    knots = _clamped(merged_interior_knots(np.linspace(1, 9, 9), ranges, DEGREE, LO, HI))
    Z = constraint_null_space(pinning_rows(knots, DEGREE, ranges))
    spline = _random_member(knots, Z, 3)
    step = 1e-7
    assert spline(3.0 - step) == pytest.approx(spline(3.0 + step), abs=1e-5)
    slopes = spline.derivative()([3.0 - 1e-3, 3.0 + 1e-3])
    assert abs(slopes[0] - slopes[1]) > 1e-3  # a kink is allowed, not forced to vanish


@pytest.mark.parametrize(
    ("ranges", "message"),
    [
        ([PolynomialRange(4.0, 2.0, 1)], "lo must be below hi"),
        ([PolynomialRange(-1.0, 2.0, 1)], "inside the fitted range"),
        ([PolynomialRange(2.0, 5.0, 1), PolynomialRange(4.0, 7.0, 0)], "overlap"),
        ([PolynomialRange(2.0, 5.0, 1, "smooth"), PolynomialRange(5.0, 7.0, 0)], "meet at a kink"),
        ([PolynomialRange(2.0, 5.0, 4)], "degree"),
    ],
)
def test_invalid_ranges_are_refused_by_name(ranges, message):
    with pytest.raises(ValueError, match=message):
        validate_ranges(ranges, DEGREE, LO, HI)


def test_dependent_rows_are_refused_not_silently_truncated():
    row = np.arange(1.0, 6.0)
    with pytest.raises(ValueError, match="dependent"):
        constraint_null_space(np.vstack([row, 2.0 * row]))
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_spline_polynomial_ranges.py -q`
Expected: FAIL (`ModuleNotFoundError: superglm.features._spline_ranges`).

- [ ] **Step 3: Implement `_spline_ranges.py`**

```python
"""Polynomial ranges on a spline: edge knots, pinning rows and their null space.

A range pins the spline to a polynomial of ``degree`` on ``[lo, hi]``. Its
edges become knots -- repeated ``degree`` times for a kink (C0, Curry-Schoenberg;
de Boor, *A Practical Guide to Splines*) or once for the spline's own
continuity -- and the spline's other knots inside the range are dropped, so
the range is ONE polynomial piece. Pinning that piece to degree d is then
"its (d+1)-th derivative vanishes", which is ``spline_degree - d`` rows at
distinct points of the piece. The rows are absorbed as ``beta = Z theta``
(Wood 2017, section 1.8.1).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import BSpline

JOINS = ("kink", "smooth")
SHAPE_NAMES = ("Flat", "Line", "Quadratic", "Cubic")


@dataclass(frozen=True)
class PolynomialRange:
    """Pin a spline to a polynomial of ``degree`` (0-3) on ``[lo, hi]``.

    ``join="kink"`` keeps the curve continuous at the edges and lets its slope
    change there; ``"smooth"`` keeps the spline's own continuity. On an
    ordered term ``lo`` and ``hi`` may be band names.
    """

    lo: float | str
    hi: float | str
    degree: int
    join: str = "kink"

    def __post_init__(self) -> None:
        if isinstance(self.degree, bool) or not isinstance(self.degree, (int, np.integer)):
            raise ValueError(f"PolynomialRange degree must be an integer, got {self.degree!r}")
        if not 0 <= int(self.degree) <= 3:
            raise ValueError(f"PolynomialRange degree must be 0-3, got {self.degree}")
        if self.join not in JOINS:
            raise ValueError(f"PolynomialRange join must be one of {JOINS}, got {self.join!r}")

    @property
    def label(self) -> str:
        return SHAPE_NAMES[int(self.degree)]


def validate_ranges(
    ranges: Sequence[PolynomialRange], degree: int, lo: float, hi: float
) -> tuple[PolynomialRange, ...]:
    """Return numeric ranges sorted by ``lo``, refusing anything ill-posed."""
    ordered = tuple(sorted(ranges, key=lambda r: float(r.lo)))
    for r in ordered:
        if not float(r.lo) < float(r.hi):
            raise ValueError(f"PolynomialRange lo must be below hi, got [{r.lo}, {r.hi}]")
        if float(r.lo) < lo or float(r.hi) > hi:
            raise ValueError(
                f"PolynomialRange [{r.lo}, {r.hi}] must lie inside the fitted range [{lo}, {hi}]"
            )
        if int(r.degree) > degree:
            raise ValueError(f"PolynomialRange degree {r.degree} exceeds the spline degree {degree}")
    for left, right in zip(ordered[:-1], ordered[1:]):
        if float(right.lo) < float(left.hi):
            raise ValueError(f"PolynomialRanges [{left.lo}, {left.hi}] and [{right.lo}, {right.hi}] overlap")
        if float(right.lo) == float(left.hi) and "smooth" in (left.join, right.join):
            raise ValueError("Adjacent PolynomialRanges must meet at a kink")
    return ordered


def merged_interior_knots(
    base: NDArray, ranges: Sequence[PolynomialRange], degree: int, lo: float, hi: float
) -> NDArray:
    """Base interior knots outside every range, plus each range's edge knots."""
    tolerance = 1e-9 * (hi - lo)
    base = np.asarray(base, dtype=np.float64)
    inside = np.zeros(base.size, dtype=bool)
    multiplicity: dict[float, int] = {}
    for r in ranges:
        inside |= (base >= float(r.lo) - tolerance) & (base <= float(r.hi) + tolerance)
        for edge in (float(r.lo), float(r.hi)):
            if lo < edge < hi:
                copies = degree if r.join == "kink" else 1
                multiplicity[edge] = max(multiplicity.get(edge, 0), copies)
    edges = [edge for edge, copies in multiplicity.items() for _ in range(copies)]
    return np.sort(np.concatenate([base[~inside], np.asarray(edges, dtype=np.float64)]))


def pinned_intervals(ranges: Sequence[PolynomialRange]) -> list[tuple[float, float]]:
    return [(float(r.lo), float(r.hi)) for r in ranges]


def pinning_rows(knots: NDArray, degree: int, ranges: Sequence[PolynomialRange]) -> NDArray:
    """Rows C with ``C @ beta = 0`` iff each range's piece has at most its degree."""
    n_basis = len(knots) - degree - 1
    blocks = [np.zeros((0, n_basis))]
    for r in ranges:
        n_points = degree - int(r.degree)
        if n_points == 0:
            continue
        fractions = np.arange(1, n_points + 1) / (n_points + 1)
        points = float(r.lo) + (float(r.hi) - float(r.lo)) * fractions
        blocks.append(derivative_design(knots, degree, points, int(r.degree) + 1))
    return np.vstack(blocks)


def derivative_design(knots: NDArray, degree: int, points: NDArray, order: int) -> NDArray:
    """Order-``order`` derivative of every basis function at ``points``."""
    identity = np.eye(len(knots) - degree - 1)
    return BSpline(knots, identity, degree)(points, nu=order)


def constraint_null_space(C: NDArray) -> NDArray:
    """Orthonormal Z with ``C @ Z = 0`` for a certified full-row-rank C."""
    n_rows, n_basis = C.shape
    if n_rows == 0:
        return np.eye(n_basis)
    Q, R = np.linalg.qr(C.T, mode="complete")
    diagonal = np.abs(np.diag(R))
    if diagonal.min() <= max(C.shape) * np.finfo(float).eps * diagonal.max():
        raise ValueError(
            "polynomial range constraints are dependent; ranges this close need to meet at a kink"
        )
    return Q[:, n_rows:]
```

Note: `BSpline(knots, identity, degree)` evaluates all basis functions at once (a vector-valued spline), replacing the per-column loop the natural-boundary code uses.

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_spline_polynomial_ranges.py -q`
Expected: PASS. Then mutation-check: change `n_points = degree - int(r.degree)` to `degree - int(r.degree) - 1` and confirm `test_every_member_is_a_polynomial_of_the_range_degree` fails; restore.

- [ ] **Step 5: Commit**

```bash
git add src/superglm/features/_spline_ranges.py tests/test_spline_polynomial_ranges.py
git commit -m "Add polynomial-range geometry for splines"
```

---

### Task 2: `Spline(polynomial_ranges=...)` for `bs` and `cr`

**Files:**
- Modify: `src/superglm/features/_spline_factory.py` (new `polynomial_ranges` keyword, passed to `bs`/`cr`; refusal for other kinds)
- Modify: `src/superglm/features/_spline_config.py` (`initialize_spec` stores `self._polynomial_ranges`; refusals for `constraint=` and `select=True` with ranges)
- Modify: `src/superglm/features/spline.py` (`_SplineBase`: knot placement, `_constraint_rows`, one `_apply_constraints`, `fitted_base_knots`; `CubicRegressionSpline`/`NaturalSpline` stop overriding `_apply_constraints` and supply natural rows via `_constraint_rows`)
- Modify: `src/superglm/features/_spline_constraints.py` (`build_natural_constraint_rows` returning the 2×K C; `build_natural_constraint_null_space` becomes `constraint_null_space(build_natural_constraint_rows(...))`)
- Modify: `src/superglm/features/_spline_penalties.py` (`build_integrated_derivative_penalty(..., excluded=())`)
- Modify: `src/superglm/features/_spline_runtime.py` (`place_knots`: insert range edges via `merged_interior_knots`, keep the base interior in `self._base_interior_knots`, check `degree + 1` distinct observed x per range)
- Modify: `src/superglm/__init__.py` (export `PolynomialRange`)
- Test: `tests/test_spline_polynomial_ranges.py` (append)

**Interfaces:**
- Consumes: Task 1 functions.
- Produces:
  - `Spline(kind="bs"|"cr", ..., polynomial_ranges: Sequence[PolynomialRange] | None = None)`; refuses `kind` in `("ps", "ns", "cr_cardinal")` with `ValueError("polynomial_ranges needs kind='bs' or kind='cr' ...")`, and refuses `constraint=` or `select=True` together with ranges.
  - `spec.polynomial_ranges -> tuple[PolynomialRange, ...]` (resolved, numeric, sorted; empty tuple when none).
  - `spec.fitted_base_knots -> NDArray | None` — interior knots before range edges were inserted (what the editor passes back as `knots=` for the next step).
  - `spec.fitted_knots` — unchanged meaning (all interior knots of the fitted basis), now including repeated edge knots.
  - `_SplineBase._constraint_rows(self) -> NDArray` (K columns; default: pinning rows only; `cr`/`ns` prepend natural rows, omitting a natural row at an end covered by a range of degree <= 1 — that row is implied and would make C dependent).

- [ ] **Step 1: Write the failing tests (append to `tests/test_spline_polynomial_ranges.py`)**

```python
import pandas as pd

from superglm import PolynomialRange as PublicRange
from superglm import Spline, SuperGLM


def _book(n=6_000, seed=5):
    rng = np.random.default_rng(seed)
    x = rng.uniform(18.0, 80.0, n)
    weight = rng.gamma(4.0, 0.25, n) + 0.05
    eta = 0.1 + 0.3 * np.sin((x - 18.0) / 9.0)
    y = rng.poisson(np.exp(eta) * weight) / weight
    return pd.DataFrame({"age": x}), y, weight


def _fitted(kind, ranges, *, discrete=False):
    X, y, w = _book()
    model = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        features={"age": Spline(kind=kind, k=12, polynomial_ranges=ranges)},
        discrete=discrete,
    )
    model.fit_reml(X, y, sample_weight=w)
    return model


def _curve(model, lo, hi):
    grid = np.linspace(lo, hi, 61)
    eta = model._predict_eta_exact(pd.DataFrame({"age": grid})) - model.result.intercept
    return grid, eta


@pytest.mark.parametrize("kind", ["bs", "cr"])
@pytest.mark.parametrize("discrete", [False, True])
@pytest.mark.parametrize("degree", [0, 1, 2])
def test_fitted_curve_is_the_pinned_polynomial_on_the_range(kind, discrete, degree):
    model = _fitted(kind, [PublicRange(30.0, 45.0, degree)], discrete=discrete)
    grid, eta = _curve(model, 30.0, 45.0)
    coeffs = np.polynomial.polynomial.polyfit(grid, eta, degree)
    residual = eta - np.polynomial.polynomial.polyval(grid, coeffs)
    assert np.max(np.abs(residual)) <= 1e4 * np.finfo(float).eps * max(1.0, np.max(np.abs(eta)))


def test_curve_outside_the_range_stays_smooth_and_penalised():
    model = _fitted("bs", [PublicRange(30.0, 45.0, 1)])
    lambdas = dict(model._reml_result.lambdas)
    assert 0.0 < lambdas["age"] < np.inf


def test_restricted_penalty_is_the_curvature_integral_over_free_intervals_only():
    """beta' S beta must equal the integral of f''**2 outside the pinned range."""
    from scipy.interpolate import BSpline

    from superglm.features._spline_penalties import build_integrated_derivative_penalty
    from superglm.features._spline_ranges import (
        PolynomialRange as Range,
        constraint_null_space,
        merged_interior_knots,
        pinning_rows,
    )

    ranges = (Range(3.0, 6.0, 2),)
    interior = merged_interior_knots(np.linspace(1, 9, 9), ranges, 3, 0.0, 10.0)
    knots = np.concatenate([[0.0] * 4, interior, [10.0] * 4])
    Z = constraint_null_space(pinning_rows(knots, 3, ranges))
    omega = build_integrated_derivative_penalty(knots, 3, 2, excluded=[(3.0, 6.0)])
    beta = Z @ np.random.default_rng(1).normal(size=Z.shape[1])
    curvature = BSpline(knots, beta, 3).derivative(2)
    nodes, weights = np.polynomial.legendre.leggauss(8)
    free = [(a, b) for a, b in zip(np.unique(knots)[:-1], np.unique(knots)[1:]) if not (a >= 3.0 and b <= 6.0)]
    integral = sum(
        0.5 * (b - a) * float(np.sum(weights * curvature(0.5 * (b - a) * nodes + 0.5 * (a + b)) ** 2))
        for a, b in free
    )
    assert float(beta @ omega @ beta) == pytest.approx(integral, rel=1e-10)
    inside = BSpline(knots, beta, 3).derivative(2)(np.linspace(3.0, 6.0, 5))
    assert np.ptp(inside) <= 1e-8 * max(1.0, float(np.max(np.abs(inside))))  # quadratic: constant f''


def test_whole_axis_range_equals_an_unpenalised_polynomial_fit():
    X, y, w = _book()
    lo, hi = float(X["age"].min()), float(X["age"].max())
    ranged = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        features={"age": Spline(kind="bs", k=12, polynomial_ranges=[PublicRange(lo, hi, 2)])},
    )
    ranged.fit_reml(X, y, sample_weight=w)
    Xq = pd.DataFrame({"age": X["age"], "age2": X["age"] ** 2})
    from superglm import Numeric

    poly = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        features={"age": Numeric(), "age2": Numeric()},
    )
    poly.fit(Xq, y, sample_weight=w)
    # Agreement is bounded by the two fits' IRLS convergence tolerance, not round-off.
    np.testing.assert_allclose(ranged.predict(X), poly.predict(Xq), rtol=1e-6, atol=0.0)


def test_range_at_the_boundary_on_cr_fits_without_dependent_rows():
    model = _fitted("cr", [PublicRange(70.0, 80.0, 0)])
    grid, eta = _curve(model, 70.0, model._feature_specs["age"].fitted_boundary[1])
    assert np.ptp(eta) <= 1e4 * np.finfo(float).eps * max(1.0, np.max(np.abs(eta)))


def test_fitted_knots_report_edges_and_base_knots_reproduce_placement():
    model = _fitted("bs", [PublicRange(30.0, 45.0, 1)])
    spec = model._feature_specs["age"]
    assert np.count_nonzero(spec.fitted_knots == 30.0) == 3
    assert np.count_nonzero(spec.fitted_knots == 45.0) == 3
    assert not np.any((spec.fitted_knots > 30.0) & (spec.fitted_knots < 45.0))
    assert 30.0 not in spec.fitted_base_knots and 45.0 not in spec.fitted_base_knots


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"kind": "ps"}, "kind='bs' or kind='cr'"),
        ({"kind": "cr_cardinal"}, "kind='bs' or kind='cr'"),
        ({"kind": "bs", "select": True}, "select"),
    ],
)
def test_unsupported_combinations_are_refused_by_name(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Spline(k=12, polynomial_ranges=[PublicRange(30.0, 45.0, 1)], **kwargs)


def test_too_few_distinct_x_in_a_range_is_refused_by_name():
    X, y, w = _book()
    X.loc[(X["age"] > 50) & (X["age"] < 52), "age"] = 51.0
    X = X[~((X["age"] > 50.0) & (X["age"] < 52.0)) | (X["age"] == 51.0)]
    model = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        features={"age": Spline(kind="bs", k=12, polynomial_ranges=[PublicRange(50.5, 51.5, 1)])},
    )
    with pytest.raises(ValueError, match="distinct"):
        model.fit_reml(X, y[X.index], sample_weight=w[X.index])


def test_ppform_export_reproduces_a_kinked_ranged_spline():
    from superglm.export._ppform import extract_ppform

    model = _fitted("bs", [PublicRange(30.0, 45.0, 1)])
    block = extract_ppform(model, "age")
    grid = np.linspace(18.5, 79.5, 400)
    _, eta = _curve(model, 18.5, 79.5)
    np.testing.assert_allclose(block.evaluate(np.linspace(18.5, 79.5, 61)), eta, atol=1e-10)
```

Adjust the last test's `extract_ppform(...)` call and the returned block's evaluation method to the real signature in `src/superglm/export/_ppform.py:111` (read it; keep the assertion: ppform values equal the model's link-scale term curve to 1e-10). Replace `model._feature_specs[...]` with the accessor the codebase uses for a fitted feature spec if that attribute name differs (`rg -n "_feature_specs|def feature_spec" src/superglm/model`).

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_spline_polynomial_ranges.py -q -n 4`
Expected: FAIL (`TypeError: Spline() got an unexpected keyword argument 'polynomial_ranges'`, `ImportError: PolynomialRange`).

- [ ] **Step 3: Implement**

1. `_spline_penalties.build_integrated_derivative_penalty(knots, degree, order, excluded=())`: skip an interval `(a, b)` when `any(lo <= a and b <= hi for lo, hi in excluded)`. Every `_build_penalty_for_order` on `BSplineSmooth` and `CubicRegressionSpline` passes `excluded=pinned_intervals(self._polynomial_ranges)`.
2. `_spline_constraints.build_natural_constraint_rows(knots, degree, *, lo, hi) -> NDArray` (the 2×K f'' rows, built with `derivative_design`); `build_natural_constraint_null_space` returns `constraint_null_space(build_natural_constraint_rows(...))` so existing callers keep working.
3. `_SplineBase`:

```python
    def _constraint_rows(self) -> NDArray:
        """Linear equality rows on the raw coefficients; empty when unconstrained."""
        return pinning_rows(self._knots, self.degree, self._polynomial_ranges)

    def _apply_constraints(self, B, omega: NDArray) -> tuple[Any, NDArray, int, NDArray | None]:
        C = self._constraint_rows()
        if C.shape[0] == 0:
            return B, omega, self._n_basis, None
        Z = constraint_null_space(C)
        self._Z = Z
        return B, Z.T @ omega @ Z, Z.shape[1], Z
```

   `CubicRegressionSpline` and `NaturalSpline` (degree 3) replace their `_apply_constraints` overrides with:

```python
    def _constraint_rows(self) -> NDArray:
        natural = build_natural_constraint_rows(self._knots, self.degree, lo=self._lo, hi=self._hi)
        # A Flat or Line range touching an end already makes f'' vanish there;
        # the natural row at that end would be a dependent duplicate.
        pinned_ends = {
            end
            for r in self._polynomial_ranges
            if r.degree <= 1
            for end, edge, boundary in ((0, r.lo, self._lo), (1, r.hi, self._hi))
            if float(edge) == boundary
        }
        keep = [end for end in (0, 1) if end not in pinned_ends]
        return np.vstack([natural[keep], super()._constraint_rows()])
```

   `NaturalSpline` keeps its `degree < 3` early return (no natural rows, only `super()._constraint_rows()`).
4. `_spline_runtime.place_knots`: after the base interior is resolved, store `spec._base_interior_knots = interior.copy()`; when `spec._polynomial_ranges` is non-empty, resolve them with `validate_ranges(..., spec.degree, spec._lo, spec._hi)`, refuse a range holding fewer than `degree + 1` distinct training x values (`ValueError(f"PolynomialRange [{lo}, {hi}] needs at least {d + 1} distinct values of the feature inside it; it has {n}")`), then `interior = merged_interior_knots(interior, ranges, spec.degree, spec._lo, spec._hi)` before `_assemble_knot_vector`.
5. `fitted_base_knots` property next to `fitted_knots`: `None` before fit, else `self._base_interior_knots.copy()`.
6. Factory/config: thread `polynomial_ranges` through `Spline()` into `initialize_spec` (store `tuple(polynomial_ranges or ())` as `self._polynomial_ranges`); refusals: kinds other than `bs`/`cr`; `constraint is not None`; `select=True`.
7. Export `PolynomialRange` from `superglm/__init__.py` next to `Spline`.
8. A range covering every knot interval leaves an identically zero penalty. REML must then treat the group as unpenalised rather than estimate a lambda for a zero matrix: find how `fit_reml` collects penalised groups (`rg -n "penalty_matrix|reml_groups" src/superglm/model/reml_setup.py`) and drop a group whose projected penalty is exactly zero from the REML set (the same way an unpenalised parametric group is excluded). `test_whole_axis_range_equals_an_unpenalised_polynomial_fit` pins it.

- [ ] **Step 4: Run to verify they pass, plus the spline neighbourhood**

Run: `uv run pytest tests/test_spline_polynomial_ranges.py tests/ -q -n 8 -k "spline or natural or cr_ or ppform or export"`
Expected: PASS. Mutation-check: remove `excluded=` from the `bs` penalty call and confirm `test_pinned_quadratic_is_not_shrunk_by_the_penalty` or the whole-axis test fails; restore. Remove the natural-row omission and confirm `test_range_at_the_boundary_on_cr_fits_without_dependent_rows` fails; restore.

- [ ] **Step 5: Complete-fit timing and commit**

Time `fit_reml` on the 105k-row demo book (`docs/examples/editor_demo.ipynb` cells 2-6 data) with `age` as `Spline(kind="bs", k=14, knot_strategy="quantile_tempered")` vs the same plus `polynomial_ranges=[PolynomialRange(18, 25, 1), PolynomialRange(70, 80, 0)]`, threads pinned to 1; record wall time and the in-range polynomial residual in the commit message.

```bash
git add src/superglm/features/_spline_factory.py src/superglm/features/_spline_config.py src/superglm/features/spline.py src/superglm/features/_spline_constraints.py src/superglm/features/_spline_penalties.py src/superglm/features/_spline_runtime.py src/superglm/__init__.py tests/test_spline_polynomial_ranges.py
git commit -m "Pin bs and cr splines to polynomials on chosen ranges"
```

---

### Task 3: Ordered categoricals take band-name ranges

**Files:**
- Modify: `src/superglm/features/ordered_categorical.py` (a `_resolve_spline_named_ranges(spline)` beside `_resolve_spline_named_knots` at `:1112`, called from the same place)
- Test: `tests/test_spline_polynomial_ranges.py` (append)

**Interfaces:**
- Consumes: `PolynomialRange` with `lo`/`hi` as band names; `spec.polynomial_ranges`.
- Produces: `OrderedCategorical(order=..., basis=Spline(kind="cr", k=5, polynomial_ranges=[PolynomialRange("25-34", "50-64", 1)]))` resolves names to level values (the same `_level_to_value[...]` the named knots use) and replaces the spline's ranges with numeric ones.

- [ ] **Step 1: Failing test**

```python
def test_ordered_term_pins_whole_bands_named_by_label():
    from superglm import OrderedCategorical

    rng = np.random.default_rng(9)
    bands = ["A", "B", "C", "D", "E", "F", "G"]
    X = pd.DataFrame({"band": rng.choice(bands, 8_000)})
    effect = dict(zip(bands, [0.0, 0.1, 0.25, 0.3, 0.2, 0.35, 0.5]))
    y = rng.poisson(np.exp(X["band"].map(effect).to_numpy()))
    model = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        features={
            "band": OrderedCategorical(
                order=bands,
                basis=Spline(kind="cr", k=6, polynomial_ranges=[PublicRange("B", "E", 1)]),
            )
        },
    )
    model.fit_reml(X, y)
    rel = model.predict(pd.DataFrame({"band": bands}))
    log_rel = np.log(rel[1:5])
    second_difference = np.diff(log_rel, n=2)
    assert np.max(np.abs(second_difference)) <= 1e4 * np.finfo(float).eps * max(1.0, np.max(np.abs(log_rel)))
```

- [ ] **Step 2: Run, expect FAIL** (`ValueError` from `float("B")` in `validate_ranges`). Run: `uv run pytest tests/test_spline_polynomial_ranges.py -q -k ordered`
- [ ] **Step 3: Implement** `_resolve_spline_named_ranges`, mirroring `_resolve_spline_named_knots`: for each range, a `str` edge resolves through `_resolve_declared_position` → `_grouped_break_position` → `_level_to_value[self._smooth_levels[pos]]`; numeric edges stay; then `spline._polynomial_ranges = tuple(replace(r, lo=lo_value, hi=hi_value) for ...)`.
- [ ] **Step 4: Run, expect PASS**; mutation: skip the resolution call → test fails.
- [ ] **Step 5: Commit** `git add src/superglm/features/ordered_categorical.py tests/test_spline_polynomial_ranges.py && git commit -m "Let ordered terms pin whole bands by name"`

---

### Task 4: Editor disclosure names shaped ranges

**Files:**
- Modify: `src/superglm/editor/transform.py:30` (moved in Task 5; here only rename the constant) → `EDITOR_CHOSEN_SHAPE_ATTRIBUTE = "_editor_chosen_shape"`
- Modify: `src/superglm/model/report_ops.py:253-257,717` (`editor_break_terms` → `editor_shape_terms`; `_editor_chose_breaks` → `_editor_chose_shape`)
- Modify: `src/superglm/inference/summary.py:1354` (`editor_break_notes` → `editor_shape_notes`, text below), `:872`, `:1308`
- Modify: `src/superglm/export/summary.py:441`, `src/superglm/editor/summaries.py:85`
- Test: update the existing note tests (`rg -n "editor_break|placed in the editor" tests/`)

**Interfaces:**
- Produces: `model_info["editor_shape_terms"]: list[str]`; note text exactly: `"Shaped ranges for {term} were chosen in the editor from this data. Tests are conditional on them; judge them on validation deviance."`

- [ ] **Step 1:** Update the existing tests' expected text and keys to the new names (they fail).
- [ ] **Step 2:** Run `uv run pytest -q -n 8 $(rg -l "editor_break|placed in the editor" tests/)` — FAIL.
- [ ] **Step 3:** Rename across the five source sites (one constant, one key, one helper, the note text).
- [ ] **Step 4:** Run again — PASS.
- [ ] **Step 5:** Commit with explicit paths: `git commit -m "Name shaped ranges in the editor disclosure"`

---

### Task 5: Editor backend — `/shape_range` replaces `/transform_term`

**Files:**
- Create: `src/superglm/editor/shapes.py` (replaces `editor/transform.py`; delete that file with `git rm`)
- Modify: `src/superglm/editor/session.py:919-957` (`replace_with_transformed_term` → `replace_with_shaped_range`)
- Modify: `src/superglm/editor/widget.py:1012-1030` (`_transform_term` → `_shape_range`)
- Modify: `src/superglm/editor/server.py:300-315` (`/transform_term` → `/shape_range`; drop `_break_list` if now unused)
- Modify: `src/superglm/editor/payloads.py` (per-term `shape` payload; remove the `transform` payload)
- Modify: `src/superglm/editor/errors.py` or the session module constant (`_SHAPE_REFUSED` message)
- Test: `tests/test_editor_structure.py` (replace the transform tests with shape tests)

**Interfaces:**
- Consumes: Tasks 2-4 (`Spline(polynomial_ranges=...)`, `fitted_base_knots`, `PolynomialRange`, band-name ranges, `EDITOR_CHOSEN_SHAPE_ATTRIBUTE`).
- Produces:
  - `shapes.shape_availability(model, editable) -> tuple[bool, str | None]` — reasons (exact strings, shown in hover): `"Shapes need a spline term."` (categorical, linear Numeric, ordered term without a spline basis); `"Shapes are not available for cardinal cubic regression splines."`; `"Remove the term's shape constraint to add shaped ranges."`; `"A term used by an interaction cannot be reshaped."`
  - `shapes.shaped_feature_spec(model, editable, *, lo, hi, degree, X) -> tuple[FeatureSpec, str]` — label like `"Line 18–25"`. After snapping, `lo >= hi` raises `EditorValueError("Select at least two points to shape a range.")` before any library call. Numeric: `lo`/`hi` snapped to 3 significant figures of the fitted span (`snap_edge(value, span)`); source kinds `bs`/`cr` keep their kind; `ps`/`ns` become `bs` with the same `fitted_base_knots`, `fitted_boundary`, degree and `m` (the help text says so). Existing ranges are kept; the same `(lo, hi)` with a new degree replaces; any other overlap raises `EditorValueError(f"This range overlaps the {label} range {lo}–{hi}. Restore it or choose a range outside it.")`. Ordered with a spline basis: `lo`/`hi` are band labels; rebuild through `rebuilt_ordered_spec` with the basis spline carrying the named range. Sets `EDITOR_CHOSEN_SHAPE_ATTRIBUTE` on the built spline.
  - `EditorSession.replace_with_shaped_range(term, *, lo, hi, degree, **refit_kwargs)` → `_refit_replacing` + `_push_structure(operation="shape_range", ...)`; a library `ValueError` becomes `EditorValueError(_SHAPE_REFUSED)` with `_SHAPE_REFUSED = "That range cannot be shaped. Choose a range with more distinct values, or a lower degree."`.
  - `POST /shape_range` payload `{term: str, lo: number|string, hi: number|string, degree: 0-3, method?: str, level_display?: str}` → the structural-step envelope.
  - Term payload `shape: {available: bool, reason: str|null, ranges: [{lo, hi, degree, label}]}`.

- [ ] **Step 1: Write the failing tests** (in `tests/test_editor_structure.py`, replacing every `transform_term` / `replace_with_transformed_term` test; use the file's existing session fixtures)

```python
def test_line_range_refits_and_restore_undoes_it(age_session):
    session = age_session
    before = session.to_model().predict(session.validation_X)
    session.replace_with_shaped_range("age", lo=30.0, hi=45.0, degree=1)
    spec = session.model._feature_specs["age"]
    assert [(r.lo, r.hi, r.degree) for r in spec.polynomial_ranges] == [(30.0, 45.0, 1)]
    assert session.structure_history[-1].operation == "shape_range"
    session.uncollapse_levels()
    np.testing.assert_array_equal(session.to_model().predict(session.validation_X), before)


def test_second_shape_keeps_the_free_knots(age_session):
    session = age_session
    session.replace_with_shaped_range("age", lo=30.0, hi=45.0, degree=1)
    first = session.model._feature_specs["age"].fitted_base_knots
    session.replace_with_shaped_range("age", lo=70.0, hi=80.0, degree=0)
    np.testing.assert_array_equal(session.model._feature_specs["age"].fitted_base_knots, first)


def test_overlapping_range_is_refused_by_name_and_same_range_replaces(age_session):
    session = age_session
    session.replace_with_shaped_range("age", lo=30.0, hi=45.0, degree=1)
    with pytest.raises(EditorValueError, match="overlaps the Line range 30–45"):
        session.replace_with_shaped_range("age", lo=40.0, hi=50.0, degree=0)
    session.replace_with_shaped_range("age", lo=30.0, hi=45.0, degree=2)
    spec = session.model._feature_specs["age"]
    assert [(r.lo, r.hi, r.degree) for r in spec.polynomial_ranges] == [(30.0, 45.0, 2)]


def test_single_point_selection_is_refused_before_the_library(age_session):
    with pytest.raises(EditorValueError, match="^Select at least two points to shape a range.$"):
        age_session.replace_with_shaped_range("age", lo=33.0, hi=33.0, degree=1)


def test_library_refusal_reaches_the_browser_as_the_fixed_sentence(age_session):
    # Two distinct x values cannot carry a quadratic: the library refuses by name,
    # and the editor turns that into its one intentional sentence.
    lo, hi = age_session.two_value_range("age")
    with pytest.raises(EditorValueError) as caught:
        age_session.replace_with_shaped_range("age", lo=lo, hi=hi, degree=2)
    assert str(caught.value) == _SHAPE_REFUSED


def test_revert_after_two_shapes_restores_the_opened_model(age_session):
    session = age_session
    opened = session.to_model().predict(session.validation_X)
    session.replace_with_shaped_range("age", lo=30.0, hi=45.0, degree=1)
    session.replace_with_shaped_range("age", lo=70.0, hi=80.0, degree=0)
    session.revert_to_reference_model()
    np.testing.assert_array_equal(session.to_model().predict(session.validation_X), opened)


def test_shape_payload_reports_availability_and_ranges(age_session, widget_client):
    response = widget_client.post("/shape_range", json={"term": "age", "lo": 30.0, "hi": 45.0, "degree": 1})
    assert response.status_code == 200
    term = response.json()["state"]["terms"]["age"]
    assert term["shape"]["available"] is True
    assert term["shape"]["ranges"] == [{"lo": 30.0, "hi": 45.0, "degree": 1, "label": "Line"}]


def test_categorical_term_reports_shapes_unavailable(widget_client):
    term = widget_client.get("/state").json()["terms"]["region"]
    assert term["shape"] == {"available": False, "reason": "Shapes need a spline term.", "ranges": []}
```

`two_value_range` is a test helper, not session API: write it in the test module as a function returning an interval between two adjacent distinct training `age` values (so it holds exactly two). Build the `age_session` fixture from the file's existing fixture pattern (a Poisson model with `age` as `Spline(kind="bs", k=12)` fitted on a few thousand rows, with validation data), and `widget_client` from the existing FastAPI test-client fixture used by the `/set_reference` tests. Align payload key paths (`["state"]["terms"]`) with what the existing set-reference endpoint test reads.

- [ ] **Step 2: Run, expect FAIL** (`AttributeError: replace_with_shaped_range`). Run: `uv run pytest tests/test_editor_structure.py -q -n 4`
- [ ] **Step 3: Implement** `shapes.py`, the session/widget/server/payload changes; `git rm src/superglm/editor/transform.py` and remove its imports; remove `replace_with_transformed_term`, `_transform_term`, `/transform_term`, the `transform` payload and `_TRANSFORM_REFUSED`. Keep `collapse.rebuilt_ordered_spec` as the ordered rebuild.
- [ ] **Step 4: Run, expect PASS**, plus `uv run pytest tests/ -q -n 8 -k editor -m "not browser"`. Mutation: make `shaped_feature_spec` rebuild from `fitted_knots` instead of `fitted_base_knots` and confirm `test_second_shape_keeps_the_free_knots` fails; restore.
- [ ] **Step 5: Commit** with explicit paths (including the `git rm`): `git commit -m "Shape editor ranges through /shape_range"`

---

### Task 6: Frontend — shape icons and range overlay replace Breaks mode

**Files:**
- Create: `src/superglm/editor/app/shapes.js` (pure helpers), `src/superglm/editor/app/chart/shape_overlay.js`
- Modify: `src/superglm/editor/app/index.html` (four palette buttons after `#setReference`; remove the Breaks tool-rail button and Breaks controls)
- Modify: `src/superglm/editor/app/main.js` (enablement beside `setReference` at `:1246`; click handlers beside `:1592`; remove Breaks wiring)
- Modify: `src/superglm/editor/app/summary.js` (`shapeRangeTransition`; delete `transformTransition`)
- Modify: `src/superglm/editor/app/chart.js` (draw the overlay; remove break overlay calls), `src/superglm/editor/app/interactions.js` (remove Breaks gestures), `src/superglm/editor/app/views/help_content.js`, `src/superglm/editor/app/views/tool_rail.js`, styles (`styles/chart.css`, `styles/panels.css`)
- Delete (`git rm`): `app/breaks.js`, `app/chart/break_overlay.js`, `app/views/breaks_controls.js`, `tests/editor_frontend/breaks.test.js`
- Test: `tests/editor_frontend/shapes.test.js`

**Interfaces:**
- Consumes: term payload `shape: {available, reason, ranges}`, `POST /shape_range`.
- Produces (`shapes.js`):
  - `shapeRangeForSelection(term, selectedIndices: Set<number>) -> {lo: number|string, hi: number|string} | null` — null unless the selection is non-empty and contiguous in the displayed order; numeric terms return `{lo: min x, hi: max x}`; ordered terms return the first and last selected band labels (a selected collapsed group contributes all of its bands).
  - `shapeButtonState(term, selectedIndices) -> {visible: boolean, enabled: boolean, reason: string|null}` — hidden for categorical terms; visible but disabled with `term.shape.reason` when unavailable; disabled with `"Select a continuous run of points."` when the selection is not contiguous.
- `summary.js`: `shapeRangeTransition(term, lo, hi, degree) -> {name: "shape and refit", path: "/shape_range", payload: {term, lo, hi, degree, method: "auto"}}`.
- Palette buttons: `id` `shapeFlat|shapeLine|shapeQuadratic|shapeCubic`, `class="selection-item"`, `data-shape-degree="0".."3"`, `data-help-operation="shape_flat"...`, `aria-label` "Flat and refit" / "Line and refit" / "Quadratic and refit" / "Cubic and refit"; 24×24 stroked SVG icons (horizontal line; rising line; parabola; S-curve) matching the existing palette icons' stroke style.
- Help (`OPERATION_HELP`): `shape_flat`: "Make the selected range flat and refit. The rest of the curve stays smooth. Restore undoes it."; same pattern for line/quadratic/cubic; add to `shape_*`: "A P-spline term becomes a B-spline with a derivative penalty so its penalty can skip the shaped range." only in the Help drawer section text, not every hover. `STRUCTURE_HELP.restore_structure.body`: "Undo the latest collapse, ungroup, shape or reference change." Remove `TOOL_HELP.breaks` and `STRUCTURE_HELP.transform_term`; remove `"breaks"` from the Modes section.
- Overlay: for each `term.shape.ranges` entry, a light shaded band spanning `[sx(lo), sx(hi)]` (ordered terms: from the first band's left gap to the last band's right gap) with a small label at the top ("Line"); `data-popover-title="Line"`, `data-popover-body="Pinned to a straight line from 30 to 45."`. Drawn beneath the curve, above the grid; not drawn while Build is animating.

- [ ] **Step 1: Failing node tests** (`tests/editor_frontend/shapes.test.js`, using the repo's node test runner conventions from the neighbouring tests)

```js
import test from "node:test";
import assert from "node:assert/strict";
import { shapeRangeForSelection, shapeButtonState } from "../../src/superglm/editor/app/shapes.js";

const numeric = {
  x: [18, 20, 25, 30, 40, 50],
  levels: null,
  shape: { available: true, reason: null, ranges: [] },
};

test("contiguous numeric selection gives its x extent", () => {
  assert.deepEqual(shapeRangeForSelection(numeric, new Set([1, 2, 3])), { lo: 20, hi: 30 });
});

test("a gap in the selection gives no range", () => {
  assert.equal(shapeRangeForSelection(numeric, new Set([1, 3])), null);
});

test("unavailable terms show disabled icons with the backend reason", () => {
  const term = { ...numeric, shape: { available: false, reason: "Shapes need a spline term.", ranges: [] } };
  assert.deepEqual(shapeButtonState(term, new Set([1, 2])), {
    visible: true,
    enabled: false,
    reason: "Shapes need a spline term.",
  });
});

test("categorical terms hide the icons", () => {
  const term = { levels: ["N", "S"], ordered: false, shape: { available: false, reason: "Shapes need a spline term.", ranges: [] } };
  assert.equal(shapeButtonState(term, new Set([0])).visible, false);
});
```

Match the term payload field names (`x`, `levels`, ordered flag) to what `selectors.js`/`chart.js` read today (`rg -n "term.levels|term.ordered|term.kind" src/superglm/editor/app`).

- [ ] **Step 2: Run, expect FAIL** — `node --test tests/editor_frontend/shapes.test.js` (module not found).
- [ ] **Step 3: Implement** `shapes.js`, `shape_overlay.js`, the palette buttons, enablement and click handlers (`runStructuralRefit(shapeRangeTransition(selectedTerm(), range.lo, range.hi, degree))`), help text; delete Breaks mode files and wiring; run `node --test tests/editor_frontend` to catch any import of a deleted module.
- [ ] **Step 4: Run, expect PASS** — `node --test tests/editor_frontend` (all files).
- [ ] **Step 5: Commit** with explicit paths including the `git rm`s: `git commit -m "Replace Breaks mode with shape icons and a range overlay"`

---

### Task 7: Frontend — searchable left-hand feature list

**Files:**
- Create: `src/superglm/editor/app/views/feature_list.js`
- Modify: `src/superglm/editor/app/index.html` (a `<nav id="featureList">` left of the chart, collapsible; remove `<select id="term">` from the context bar), `src/superglm/editor/app/main.js` (`renderTermPickerState` at `:841` renders the list instead of `<option>`s; the `change` handler at `:1456` becomes the list's selection handler), `src/superglm/editor/app/styles/panels.css`
- Test: `tests/editor_frontend/feature_list.test.js`

**Interfaces:**
- Produces (`feature_list.js`):
  - `filterFeatures(groups: [string, string[]][], query: string) -> [string, string[]][]` — case-insensitive substring match on the feature name; empty groups dropped; empty query returns `groups` unchanged.
  - `renderFeatureList(root, {groups, activeTerm, query, termMeta}, onSelect)` — one row per feature with name, kind chip and EDF (from `termMeta[name] = {kind, edf}`), `aria-current="true"` on the active row, which is scrolled into view with `scrollIntoView({block: "nearest"})`; ArrowUp/ArrowDown move focus between visible rows, Enter selects.
  - Collapsed state: a 28px strip with the active feature's name rotated; toggle button `#featureListToggle` (`aria-expanded`), remembered in `localStorage` under `superglm.editor.featureList` wrapped in try/catch.

- [ ] **Step 1: Failing node tests**

```js
import test from "node:test";
import assert from "node:assert/strict";
import { filterFeatures } from "../../src/superglm/editor/app/views/feature_list.js";

const groups = [["Smooth", ["age", "mileage"]], ["Categorical", ["region", "territory"]]];

test("empty query keeps every feature", () => {
  assert.deepEqual(filterFeatures(groups, ""), groups);
});

test("query filters case-insensitively and drops empty groups", () => {
  assert.deepEqual(filterFeatures(groups, "TER"), [["Categorical", ["territory"]]]);
});
```

- [ ] **Step 2: Run, expect FAIL.** `node --test tests/editor_frontend/feature_list.test.js`
- [ ] **Step 3: Implement** the list, search box (`<input type="search" placeholder="Search features">`), active-row focus, keyboard, collapse; keep keyboard shortcut and focus-return behaviour (`main.js:688` lists `termSelect` among focus-return candidates — replace with the list's search box).
- [ ] **Step 4: Run, expect PASS** — `node --test tests/editor_frontend`.
- [ ] **Step 5: Commit** `git commit -m "Add a searchable feature list to the editor"`

---

### Task 8: Browser tests, docs, demo and full verification

**Files:**
- Modify: `tests/editor/test_editor_structure_browser.py` (replace the Breaks-mode test with: select a run of `age` points → click Line → the overlay shows "Line" and the in-range curve is straight → Restore removes it; feature search filters and selects `territory`)
- Modify: `docs/tutorials/edit-a-model-in-the-browser.md` (replace "Give a Term a Shape with Breaks" with "Shape a Range": gesture, the four shapes, kink joins, whole-axis polynomial = select all, P-spline note, disclosure; add "Find a Feature" for the list; update "Restore and Revert" wording)
- Modify: `docs/how-to/specify-features.md` (a short `polynomial_ranges` section with the `PolynomialRange` example and the kinds it supports)

- [ ] **Step 1:** Write the browser tests (they fail until Tasks 5-7 are in). Run: `uv run pytest tests/editor/test_editor_structure_browser.py -q -m browser`
- [ ] **Step 2:** Docs edits.
- [ ] **Step 3:** Run the demo notebook end to end with the per-cell timer (`docs/examples/editor_demo.ipynb`) and confirm it completes; record per-cell times.
- [ ] **Step 4:** Full verification: `uv run python scripts/run_test_suite.py -m "not browser and not docs"`, `node --test tests/editor_frontend`, the structure browser file, `uv run ruff check src/ tests/`, `uv run ruff format --check src/ tests/`. Report source and test LOC separately (`git diff --numstat 99241b0c -- src` vs `-- tests`).
- [ ] **Step 5:** Commit with explicit paths: `git commit -m "Document shaped ranges and cover them in the browser"`
