"""Exact piecewise-polynomial form of a fitted smooth term.

A fitted spline is a piecewise polynomial exactly, not approximately.  Whatever
basis a spline spec uses internally -- B-spline for ``PSpline``/``BSplineSmooth``,
value-parameterised for the ``cr``/``ns`` family -- the fitted curve lies in the
clamped B-spline space on the knots the model already reports publicly.  So the
coefficients are recoverable from the knots and the curve alone, by one solve,
uniformly across every kind.  Measured on all five: residual 1.9e-15 to 2.9e-15,
``cond(B) = 5.16``.

That uniformity is the whole reason this module is short.  Reading each spec's
internal parameterisation would need five derivations; reading its knots needs
one.  See ``docs/superpowers/specs/2026-08-16-continuous-block-ppform-design.md``
section 4.1 for the measurements.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import BSpline, PPoly

# The solve needs comfortably more curve samples than basis functions.  1201 is
# ~100x the basis size of a typical rating spline, which keeps the least-squares
# problem strongly over-determined without making ``term_inference`` expensive.
_DEFAULT_GRID_POINTS = 1201

# The recovered coefficients are only usable if they reproduce the curve.  The
# measured residual across every spline kind and configuration is <= 2.9e-15;
# 1e-11 is four orders of margin over that and still far tighter than any
# approximation this replaces.
_EXACTNESS_TOLERANCE = 1e-11

_PPFORM_DEGREE = 3
_N_COEFFICIENTS = _PPFORM_DEGREE + 1


class PpformNotExactError(ValueError):
    """A term's fitted curve is not the piecewise polynomial its knots imply."""


@dataclass(frozen=True)
class PpformSegments:
    """One fitted smooth as exact polynomial pieces.

    ``coefficients[i]`` are ascending powers of the NORMALISED local variable
    ``u = (x - breaks[i]) / (breaks[i + 1] - breaks[i])``, so segment ``i``
    evaluates as ``a + b*u + c*u**2 + d*u**3`` on ``[breaks[i], breaks[i+1])``.

    Normalised rather than ``x - breaks[i]`` deliberately: a raw local variable
    on a covariate ranging to 1e5 loses enough precision in a fixed-scale
    DECIMAL column to produce a 3.3x relativity error, which is worse than the
    binning this replaces.  It is also de Boor's own shifted-and-scaled form.

    A term fitted below cubic degree still reports four coefficients, with the
    unused high powers exactly zero, so every block this feeds has one column
    shape regardless of the degree the caller happened to fit.
    """

    breaks: NDArray[np.float64]
    coefficients: NDArray[np.float64]
    residual: float
    degree: int
    extrapolation: str

    @property
    def n_segments(self) -> int:
        return len(self.breaks) - 1

    def evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Log relativity at ``x``, clipped to the outermost breaks.

        Clipping here matches ``extrapolation="clip"`` and makes the emitted
        tail rows of the export block agree with this evaluator, so a test can
        compare the two without reimplementing either.
        """
        x = np.asarray(x, dtype=np.float64)
        lo, hi = self.breaks[0], self.breaks[-1]
        idx = np.clip(np.searchsorted(self.breaks, x, side="right") - 1, 0, self.n_segments - 1)
        width = self.breaks[idx + 1] - self.breaks[idx]
        u = np.clip((np.clip(x, lo, hi) - self.breaks[idx]) / width, 0.0, 1.0)
        c = self.coefficients[idx]
        return c[:, 0] + u * (c[:, 1] + u * (c[:, 2] + u * c[:, 3]))


def _clamped_knot_vector(interior: NDArray, boundary: tuple[float, float], degree: int) -> NDArray:
    """The full knot vector, with the boundary repeated ``degree + 1`` times.

    ``SplineMetadata.interior_knots`` is interior only by contract, but a spec
    that reports a boundary value among them would silently produce a repeated
    knot and a discontinuous piece, so they are filtered rather than trusted.
    """
    lo, hi = float(boundary[0]), float(boundary[1])
    interior = np.asarray(interior, dtype=np.float64)
    interior = np.sort(interior[(interior > lo) & (interior < hi)])
    return np.concatenate([np.full(degree + 1, lo), interior, np.full(degree + 1, hi)])


def extract_ppform(
    model,
    name,
    *,
    centering: str = "native",
    n_points: int = _DEFAULT_GRID_POINTS,
) -> PpformSegments:
    """Recover the exact polynomial pieces of a fitted smooth term.

    Raises ``PpformNotExactError`` if the recovered pieces do not reproduce the
    fitted curve.  That is a hard error rather than a warning: the entire value
    of this block is that it is exact, so a block that is not exact must not be
    written at all.
    """
    ti = model.term_inference(name, n_points=n_points, with_se=False, centering=centering)
    meta = getattr(ti, "spline", None)
    if meta is None:
        raise PpformNotExactError(
            f"Term {name!r} reports no spline metadata, so its knots are unknown and "
            "its piecewise-polynomial form cannot be recovered."
        )

    degree = int(meta.degree)
    if degree > _PPFORM_DEGREE:
        # Four coefficients cannot carry a quartic, and truncating one silently
        # is precisely the approximation this module exists to remove.
        raise PpformNotExactError(
            f"Term {name!r} is degree {degree}, above the cubic form's "
            f"{_N_COEFFICIENTS} coefficients. Its rating block cannot be exported "
            "as ppform."
        )
    knots = _clamped_knot_vector(meta.interior_knots, meta.boundary, degree)
    x_grid = np.asarray(ti.x, dtype=np.float64)
    f_grid = np.asarray(ti.log_relativity, dtype=np.float64)

    inside = (x_grid >= knots[0]) & (x_grid <= knots[-1])
    x_fit, f_fit = x_grid[inside], f_grid[inside]

    basis = BSpline.design_matrix(x_fit, knots, degree, extrapolate=False).toarray()
    coef, *_ = np.linalg.lstsq(basis, f_fit, rcond=None)
    residual = float(np.abs(basis @ coef - f_fit).max())
    if not np.isfinite(residual) or residual > _EXACTNESS_TOLERANCE:
        raise PpformNotExactError(
            f"Term {name!r} is not the piecewise polynomial its {len(meta.interior_knots)} "
            f"reported knots imply: the recovered form misses the fitted curve by "
            f"{residual:.3e}, above the {_EXACTNESS_TOLERANCE:.0e} tolerance. Its rating "
            "block cannot be exported as ppform."
        )

    pp = PPoly.from_spline(BSpline(knots, coef, degree), extrapolate=False)
    # ``from_spline`` keeps degenerate zero-width pieces at the clamped ends.
    keep = np.diff(pp.x) > 0
    breaks = np.concatenate([pp.x[:-1][keep], [pp.x[-1]]])
    # PPoly stores DESCENDING powers of ``x - breaks[i]``; the export wants
    # ASCENDING powers of the NORMALISED ``u``.  Rescaling by the segment width
    # is what converts between the two, and is the step that makes the emitted
    # coefficients safe in a fixed-scale numeric column.
    raw = pp.c[:, keep].T  # (n_segments, degree + 1), descending powers of (x - break)
    widths = np.diff(breaks)
    ascending = raw[:, ::-1]
    # A sub-cubic fit reports fewer than four powers; the missing high powers are
    # zero rather than absent, so every block downstream has one column shape.
    padded = np.zeros((ascending.shape[0], _N_COEFFICIENTS), dtype=np.float64)
    padded[:, : ascending.shape[1]] = ascending
    powers = np.arange(_N_COEFFICIENTS)
    coefficients = padded * widths[:, None] ** powers[None, :]

    return PpformSegments(
        breaks=breaks,
        coefficients=np.ascontiguousarray(coefficients, dtype=np.float64),
        residual=residual,
        degree=degree,
        extrapolation=str(meta.extrapolation),
    )
