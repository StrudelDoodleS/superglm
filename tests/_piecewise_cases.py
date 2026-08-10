"""Shared ``Piecewise`` fixture matrix, swept by every stage of the prototype.

One fixture per real shape, not one convenient shape: this repository has more
than once produced a green suite that measured nothing because every test ran
against the same easy geometry.  Stages 1, 2 and 3 all import from here so a
shape that breaks one surface cannot be quietly absent from another.

Every case carries the same three columns -- ``x`` (the piecewise feature),
``region`` (categorical) and ``density`` (numeric) -- so a caller can assemble
whichever model it needs without inventing new data.

``heaped_int_x`` is the one case that legitimately warns at ``build()``: tied
weighted quantiles collapse on heaped integer data, which is exactly the
behaviour it exists to cover.  Callers sweeping the whole matrix should expect
that warning rather than treat it as a failure.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from superglm.features.piecewise import Piecewise


class PiecewiseCase(NamedTuple):
    """A fitted-model-ready fixture: unpacks as ``(X, y, sample_weight, spec)``."""

    X: pd.DataFrame
    y: NDArray[np.float64]
    sample_weight: NDArray[np.float64]
    spec: Piecewise


def _frame(
    x: NDArray[np.float64],
    seed: int,
    *,
    kink: float,
) -> tuple[pd.DataFrame, NDArray[np.float64], NDArray[np.float64]]:
    """Build a Poisson dataset whose mean really is piecewise linear in x."""
    rng = np.random.default_rng(seed)
    n = x.size
    region = rng.choice(["A", "B", "C"], n)
    density = rng.uniform(0.0, 1.0, n)
    sample_weight = rng.uniform(0.5, 1.5, n)
    span = float(x.max() - x.min())
    lin = (
        -1.5
        + 0.02 * np.minimum(x, kink)
        - 0.01 * np.maximum(x - kink, 0.0)
        + 0.3 * (region == "A")
        + 0.2 * density
    )
    y = rng.poisson(np.exp(lin) * sample_weight).astype(np.float64)
    frame = pd.DataFrame({"x": x, "region": region, "density": density})
    assert span > 0.0
    return frame, y, sample_weight


def _uniform_x(n: int, lo: float, hi: float, seed: int) -> NDArray[np.float64]:
    """Deterministic near-uniform coverage of [lo, hi], endpoints included."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(lo, hi, n)
    x[0] = lo
    x[-1] = hi
    return x


def interior_base() -> PiecewiseCase:
    """Base at an interior knot -- the ordinary case."""
    x = _uniform_x(600, 0.0, 100.0, seed=11)
    X, y, w = _frame(x, seed=12, kink=50.0)
    return PiecewiseCase(X, y, w, Piecewise([25.0, 50.0, 75.0], base=50.0))


def end_base_lower() -> PiecewiseCase:
    """Base at t_0: every coefficient is measured against the bottom knot."""
    x = _uniform_x(600, 0.0, 100.0, seed=13)
    X, y, w = _frame(x, seed=14, kink=50.0)
    return PiecewiseCase(
        X, y, w, Piecewise([25.0, 50.0, 75.0], base="first", lower=0.0, upper=100.0)
    )


def end_base_upper() -> PiecewiseCase:
    """Base at t_{J+1}: the mirror image, which a one-sided bug survives."""
    x = _uniform_x(600, 0.0, 100.0, seed=15)
    X, y, w = _frame(x, seed=16, kink=50.0)
    return PiecewiseCase(X, y, w, Piecewise([25.0, 50.0, 75.0], base=100.0, lower=0.0, upper=100.0))


def unequal_widths() -> PiecewiseCase:
    """Segments of visibly different widths -- 5, 5, 50 and 40 units."""
    x = _uniform_x(600, 0.0, 100.0, seed=17)
    X, y, w = _frame(x, seed=18, kink=60.0)
    return PiecewiseCase(X, y, w, Piecewise([5.0, 10.0, 60.0], base=10.0, lower=0.0, upper=100.0))


def pinned_wider() -> PiecewiseCase:
    """Rated range wider than the data: the tariff covers ages nobody insured yet."""
    x = _uniform_x(600, 0.0, 100.0, seed=19)
    X, y, w = _frame(x, seed=20, kink=50.0)
    return PiecewiseCase(
        X, y, w, Piecewise([25.0, 50.0, 75.0], base=50.0, lower=-20.0, upper=140.0)
    )


def pinned_narrower() -> PiecewiseCase:
    """Rated range narrower than the data.

    Under the default ``extrapolation="clip"`` the rows outside
    ``[lower, upper]`` are grouped onto the boundary knots at fit time.  Under
    ``make_case(name, extrapolation="extend")`` they load the linear tails
    instead: their basis entries exceed 1 on the near knot and go negative on
    the far one -- the configuration that makes the signed zero-column test a
    false negative, and the only case in the matrix that exercises
    extrapolating rows at fit time.
    """
    x = _uniform_x(600, 0.0, 100.0, seed=21)
    X, y, w = _frame(x, seed=22, kink=50.0)
    return PiecewiseCase(X, y, w, Piecewise([40.0, 60.0], base=40.0, lower=20.0, upper=80.0))


def heaped_int_x() -> PiecewiseCase:
    """Integer x heaped on multiples of 5, with int-mode breaks and tied quantiles.

    Insurance rating variables are heaped (ages ending 0/5, whole-year
    tenures), so collapsing quantiles are the norm rather than the exception.
    ``build()`` warns for this case; that warning is the point of the fixture.
    """
    rng = np.random.default_rng(23)
    heaped = rng.choice(np.array([10.0, 20.0, 30.0]), 540, p=[0.4, 0.35, 0.25])
    off_grid = rng.choice(np.array([0.0, 5.0, 15.0, 25.0, 35.0, 40.0]), 60)
    x = np.concatenate([heaped, off_grid])
    x[0] = 0.0
    x[-1] = 40.0
    X, y, w = _frame(x, seed=24, kink=20.0)
    return PiecewiseCase(X, y, w, Piecewise(8, base="most_exposed"))


def many_knots() -> PiecewiseCase:
    """Fifteen knots, above the editor's default 12-handle display budget."""
    x = _uniform_x(900, 0.0, 112.0, seed=25)
    X, y, w = _frame(x, seed=26, kink=48.0)
    breaks = [float(v) for v in range(8, 112, 8)]
    return PiecewiseCase(X, y, w, Piecewise(breaks, base=48.0, lower=0.0, upper=112.0))


def zero_weight_rows() -> PiecewiseCase:
    """Zero-exposure rows, two of them far outside the positive-weight range.

    The library rule (``_spline_knots.knot_geometry_data``): a zero frequency
    weight represents zero replicated rows, so it must not affect learned
    geometry.  The far rows at x = -30 and x = 250 widen the default rated
    range to [-30, 250] if the boundaries ever read zero-weight rows -- moving
    both outer knots and with them every boundary value the term rates.  The
    fit must be identical with these rows present and with them dropped; the
    rows stay predictable (the default ``clip`` groups them onto the boundary
    knots), they just carry no geometry.
    """
    x = _uniform_x(600, 0.0, 100.0, seed=27)
    X, y, w = _frame(x, seed=28, kink=50.0)
    x_zero = np.array([-30.0, 15.5, 62.5, 250.0])
    zero_frame = pd.DataFrame(
        {
            "x": x_zero,
            "region": np.array(["A", "B", "C", "B"]),
            "density": np.array([0.25, 0.5, 0.75, 0.5]),
        }
    )
    X = pd.concat([X, zero_frame], ignore_index=True)
    y = np.concatenate([y, np.zeros(x_zero.size)])
    w = np.concatenate([w, np.zeros(x_zero.size)])
    return PiecewiseCase(X, y, w, Piecewise([25.0, 50.0, 75.0], base=50.0))


CASES = {
    "interior_base": interior_base,
    "end_base_lower": end_base_lower,
    "end_base_upper": end_base_upper,
    "unequal_widths": unequal_widths,
    "pinned_wider": pinned_wider,
    "pinned_narrower": pinned_narrower,
    "heaped_int_x": heaped_int_x,
    "many_knots": many_knots,
    "zero_weight_rows": zero_weight_rows,
}

# Cases whose build() is expected to warn, and why.  Anything else warning is a
# finding, not a fixture to quietly adjust.
WARNING_CASES = {"heaped_int_x": "tied weighted quantiles collapse on heaped x"}

CASE_NAMES = tuple(CASES)


def make_case(name: str, extrapolation: str | None = None) -> PiecewiseCase:
    """Return the named fixture, rebuilt fresh so specs are never shared.

    ``extrapolation`` overrides the spec's mode by rebuilding it through
    ``__init__`` (so the override is validated, not smuggled past it); ``None``
    keeps the spec's own default.
    """
    case = CASES[name]()
    if extrapolation is not None:
        spec = case.spec
        case = case._replace(
            spec=Piecewise(
                spec.breaks,
                base=spec.base,
                strategy=spec.strategy,
                lower=spec.lower,
                upper=spec.upper,
                extrapolation=extrapolation,
            )
        )
    return case


__all__ = ["CASES", "CASE_NAMES", "WARNING_CASES", "PiecewiseCase", "make_case"]
