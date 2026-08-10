"""Piecewise-linear feature: a hat basis on stated breakpoints.

Continuous but deliberately not smooth.  Free joins are already a binned
``Categorical`` and smooth joins are already a ``Spline``; this occupies the one
remaining cell of that lattice.  Every coefficient is the log-relativity of a
knot against the base knot, so the summary row, the editor handle and the
workbook cell are the same number.

Hosted inside an ``OrderedCategorical`` (``basis=Piecewise(...)``) the same
term runs on the LEVEL axis: breaks may be stated as band names, and
``degrees=`` states a per-segment polynomial degree -- the classical grafted /
segmented polynomial (Fuller 1969; Gallant & Fuller 1973, JASA 68:144-147)
with the degree-0 plateau tail of Anderson & Nelson (1975, Biometrics
31:303-318).  On the numeric axis both stay refused: the exported workbook is
exact under linear interpolation only at degree 1, and band names have no
numeric meaning.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from superglm.features._spline_knots import weighted_quantile_knots
from superglm.types import GroupInfo

# Placement rules for int-mode ``breaks``.  Kept as a frozenset so a second rule
# can land without an API change.  It stays a set of one on current evidence:
# the reported gains for k-means binning over equal-frequency binning come from
# a single-author histogram-GBT study (Labovich 2025, arXiv:2505.12460), and
# nothing located addresses *weighted* quantiles at all, which is the only kind
# an exposure-weighted tariff ever computes.  Reserve the seam, do not act on it.
_STRATEGIES = frozenset({"quantile"})

# A segment carrying less than this fraction of the total weight warns (rule 11).
# What degrades on a starved segment is the conditioning of X'WX -- a property of
# the data, not of the basis.  The hat basis itself stays well conditioned under
# any knot placement, so this is a data-support diagnostic, not a numerical one,
# and the two must not be conflated in the message.
_SMALL_SEGMENT_WEIGHT_FRACTION = 0.005

# Rule-10 rank probe routing.  Up to this many distinct x values the probe runs
# as an SVD on the weighted FACTOR sqrt(W)H -- the reliable rank object -- and
# the (n_unique, J+2) dense factor it needs is bounded and small.  Above it the
# probe falls back to the (J+2)x(J+2) weighted Gram accumulated by bincount;
# see the inline rationale at the use site for why that backstop is safe there
# and only there.  Module globals rather than defaults so a test can lower the
# ceiling and drive a small fixture through the Gram arm.
_RANK_PROBE_MAX_FACTOR_ROWS = 10_000
_GRAM_CHUNK_ROWS = 1 << 20
_GRAM_PROBE_RELATIVE_TOL = 1e-8


def _finite_x(x: NDArray) -> NDArray[np.float64]:
    """Coerce *x* to a flat float64 array, rejecting NaN and inf (rule 1)."""
    values = np.asarray(x, dtype=np.float64).ravel()
    if not np.all(np.isfinite(values)):
        n_nan = int(np.count_nonzero(np.isnan(values)))
        n_inf = int(values.size - n_nan - np.count_nonzero(np.isfinite(values)))
        raise ValueError(
            f"Piecewise requires finite x, got {n_nan} NaN and {n_inf} infinite value(s). "
            "A genuine missing band is a level, not a point on a line: put it in an "
            "OrderedCategorical(specials=[...]) rather than a Piecewise term."
        )
    return values


def _conforming_weights(
    sample_weight: NDArray | None,
    n_rows: int,
) -> NDArray[np.float64]:
    """Return per-row weights as float64, defaulting to ones (part of rule 1).

    Mirrors ``_spline_knots.knot_geometry_data``: a frequency weight counts
    replicated rows, so a negative or non-finite count is refused here, where
    the input is written -- otherwise a negative weight reaches ``np.sqrt`` in
    the rule-10 rank probe and surfaces as a NaN or LinAlgError far from the
    cause.  Zero weights are legal (zero replicated rows) and are excluded
    from learned geometry in ``_resolve_knots``.
    """
    if sample_weight is None:
        return np.ones(n_rows, dtype=np.float64)
    weights = np.asarray(sample_weight, dtype=np.float64).ravel()
    if weights.size != n_rows:
        raise ValueError(f"sample_weight must have length {n_rows}, got {weights.size}.")
    if not np.all(np.isfinite(weights)):
        n_bad = int(np.count_nonzero(~np.isfinite(weights)))
        raise ValueError(
            f"Piecewise requires finite sample_weight, got {n_bad} non-finite value(s)."
        )
    if np.any(weights < 0.0):
        n_neg = int(np.count_nonzero(weights < 0.0))
        raise ValueError(
            f"Piecewise requires non-negative sample_weight, got {n_neg} negative "
            "value(s). A frequency weight counts replicated rows, so a negative "
            "count has no meaning here."
        )
    if not np.any(weights > 0.0):
        raise ValueError(
            "Piecewise requires at least one row with positive sample_weight: an "
            "all-zero weight vector represents no replicated rows at all, so there "
            "is no data to place knots on."
        )
    return weights


def _format_values(values: NDArray) -> str:
    """Render an array of knot or weight values for an error message."""
    return "[" + ", ".join(f"{float(v):.10g}" for v in np.asarray(values).ravel()) + "]"


def _breaks_contain_names(breaks: Any) -> bool:
    """Whether a breaks sequence states any break as a level name."""
    if isinstance(breaks, int | np.integer):
        return False
    return any(isinstance(entry, str) for entry in breaks)


def _validate_degrees(degrees: Sequence[int] | None, breaks: Any) -> tuple[int, ...] | None:
    """Validate a ``degrees=`` declaration at construction time."""
    if degrees is None:
        return None
    if isinstance(breaks, int | np.integer):
        raise ValueError(
            "Piecewise degrees= requires stated breaks: degrees are per-segment "
            "statements, and int-mode breaks are placed from the data, so there "
            "are no stated segments to attach them to."
        )
    items = list(degrees)
    if len(list(breaks)) == 0:
        # Without the early refusal this falls through to build()'s rule 2,
        # whose "use Numeric() instead" advice is wrong for the one caller
        # degrees= exists for: a single global polynomial segment is a
        # Polynomial, on either axis.
        raise ValueError(
            "Piecewise(breaks=[], degrees=...) states one global polynomial "
            "segment with no breaks, which is a Polynomial: use "
            "Polynomial(degree=d) on a numeric axis, or "
            "OrderedCategorical(basis=Polynomial(...)) on a band axis."
        )
    n_segments = len(list(breaks)) + 1
    if len(items) != n_segments:
        raise ValueError(
            f"Piecewise degrees= must state one degree per segment: "
            f"{len(list(breaks))} break(s) make {n_segments} segments, got "
            f"{len(items)} degree(s)."
        )
    cleaned: list[int] = []
    for d in items:
        if isinstance(d, bool) or not isinstance(d, int | np.integer):
            raise ValueError(f"Piecewise degrees must be integers >= 0, got {d!r}")
        if d < 0:
            raise ValueError(f"Piecewise degrees must be >= 0, got {int(d)}")
        cleaned.append(int(d))
    if all(d == 0 for d in cleaned):
        raise ValueError(
            "Piecewise degrees are all 0: every segment flat is a constant, "
            "which the model intercept already carries. State at least one "
            "non-flat segment, or drop the term."
        )
    adjacent_flat = [i for i in range(len(cleaned) - 1) if cleaned[i] == 0 and cleaned[i + 1] == 0]
    if adjacent_flat:
        i = adjacent_flat[0]
        raise ValueError(
            f"Piecewise degrees {cleaned} state consecutive flat segments "
            f"({i} and {i + 1}): value continuity makes them one plateau, so the "
            "break between them states no kink. Remove that break, or give one "
            "side a degree."
        )
    return tuple(cleaned)


@dataclass(frozen=True)
class StructuralContrastRow:
    """One reported structural contrast of a (possibly segmented) piecewise term.

    ``kind == "slope_change"`` carries a contrast vector over the retained
    columns whose value is the change of slope at the stated break --
    the truncated-power coefficient of the fixed-knot spline parameterization,
    whose t/Wald row is ordinary linear-model inference (Smith 1979,
    Am. Statist. 33(2):57-62; the known-changeover case is Sprent 1961).

    ``kind == "curvature"`` carries the retained column indices of one
    segment's within-segment curvature freedoms (degree >= 2), tested jointly:
    "is this stretch actually curved?".  Under C0 seams these are the only
    honest per-segment questions -- the segments are coupled, so per-segment
    orthogonal per-power z-statistics do not exist in this design.
    """

    kind: str  # "slope_change" | "curvature"
    # Knot index of the break (slope_change) or segment index (curvature).
    index: int
    contrast: NDArray[np.float64] | None = None
    column_indices: tuple[int, ...] = ()


class Piecewise:
    """Continuous piecewise-linear feature on stated breakpoints.

    Given ``J`` breakpoints the knot vector is ``[lower, *breaks, upper]``
    (``J + 2`` knots) and the basis is the degree-1 B-spline (hat) basis on
    those knots, evaluated directly rather than through ``scipy`` so that the
    outer segments extend linearly past the boundary knots.  The hat at a base
    knot ``t_r`` is dropped for identifiability against the model intercept, so
    the group has ``J + 1`` columns and each retained coefficient is exactly

    ``v_j = f(t_j) - f(t_r)`` -- the log relativity at knot ``j``, against base.

    Parameters
    ----------
    breaks : sequence of float, or int
        The breakpoints.  A sequence is the primary mode and the defensible
        one: you state where the kinks are.  An int places that many
        breakpoints at exposure-weighted quantiles of *x*, snapped to observed
        values -- a convenience for exploration, not for a filed tariff.  Heaped
        data (ages ending 0/5, whole-year tenures) routinely collapses tied
        quantiles, so int mode can realise fewer breakpoints than requested; it
        warns rather than raising, because that outcome is the library's doing
        and not the caller's.

        As the ``basis=`` of an :class:`OrderedCategorical` the breaks may
        also be stated as BAND NAMES (``breaks=["Mi060", "Mi066"]``), resolved
        to level positions when the ordered term is constructed; integer
        positions on the level axis are the escape hatch.  On the numeric axis
        a name has nothing to resolve against, so string breaks refuse at
        ``build()``.
    degrees : sequence of int, optional
        One polynomial degree per segment (``len(breaks) + 1`` entries),
        default all 1 -- the plain kinked line.  ``0`` states a flat segment
        (the grouped/plateau tail); ``2`` and above add within-segment
        curvature.  Seams stay value-continuous by construction (the grafted-
        polynomial device: Gallant & Fuller 1973; plateau tails: Anderson &
        Nelson 1975).  Legal only when this Piecewise is the ``basis=`` of an
        :class:`OrderedCategorical`: there the export contract is one row per
        band, so the table is exact at any degree, while on the numeric axis
        the exported workbook is exact under linear interpolation only at
        degree 1 -- a numeric-axis build with any degree != 1 refuses loudly.
        Requires stated breaks (int-mode placement has no stated segments).
    base : float or {'most_exposed', 'first'}
        Reference knot, mirroring ``Categorical``.  ``'most_exposed'`` picks
        the knot carrying the largest hat-carried mass under the fit weights;
        with no ``sample_weight`` the weights are ones, so it is the knot
        carrying the most rows.  A float must equal exactly one knot.  The
        resolved base is sticky: ``_base_knot`` persists across ``build()``
        calls, so after a refit whose knot set changed, ``base='first'`` can
        legitimately remain on a surviving interior knot rather than move to
        the new first knot.
    strategy : {'quantile'}
        Placement rule, consulted only when *breaks* is an int.
    lower, upper : float, optional
        Pin the outermost knots.  Default ``min(x)`` / ``max(x)``.  Pinning
        **wider** than the data states a rated range the tariff must cover.
        Pinning **narrower** is allowed too, and what happens to the rows
        outside ``[lower, upper]`` is the ``extrapolation`` parameter's call:
        under ``"clip"`` they are grouped onto the boundary knots, so
        ``Piecewise(breaks, upper=u)`` fits identically to precomputing
        ``x.clip(max=u)`` -- the tail-grouping idiom stated as a term
        parameter instead of a preprocessing step; under ``"extend"`` they
        load the linear tails, and their leverage lands entirely on the two
        boundary segments' slopes; under ``"error"`` the build refuses.
    extrapolation : {'clip', 'extend', 'error'}
        Behaviour outside ``[lower, upper]``, mirroring ``Spline``.  ``"clip"``
        (default) holds the boundary knot's value.  ``"extend"`` continues the
        boundary segments at their fitted slopes.  ``"error"`` raises on
        out-of-range values.  The policy binds at ``build()`` as well as at
        prediction, so the design that is fitted is the design the tariff
        states.

    Notes
    -----
    Outside ``[t_0, t_{J+1}]`` the term follows ``extrapolation``, and the
    default mirrors this library's splines: hold the boundary knot's value
    flat.  Under ``"extend"`` the boundary segments' slopes continue past the
    boundary knots -- and the slope is *stated*: ``lower`` / ``upper`` pin
    where the boundary segments start and the exported table prints the two
    boundary slopes, so the rule outside the tabulated range is reproducible
    by hand.  The exported workbook states whichever rule is in force.

    In the editor the term gets one control handle per knot -- the handle *is*
    the coefficient, because the raw basis evaluated at the knots is the
    identity.  The editor's hard cap of 24 handles still applies, so a term
    with more than 24 knots displays a subsampled set of them and the knots
    without a handle cannot be dragged.  State fewer breakpoints if every knot
    has to be editable.
    """

    # Opt out of the editor's 12-handle default (see `editor/controls.py`).
    # Every column here is a reported coefficient with a knot to sit on, so
    # thinning the handles would drop model parameters out of the editor rather
    # than thin a redundant display grid.
    _editor_wants_all_handles = True

    def __init__(
        self,
        breaks: Sequence[float | str] | int,
        *,
        base: float | str = "most_exposed",
        strategy: str = "quantile",
        lower: float | None = None,
        upper: float | None = None,
        extrapolation: str = "clip",
        degrees: Sequence[int] | None = None,
    ):
        # Validated here, not in build(): a typo'd mode must fail where it is
        # written.  The message mirrors Spline's for the same parameter.
        if extrapolation not in {"clip", "extend", "error"}:
            raise ValueError(
                f"extrapolation must be one of ('clip', 'extend', 'error'), got {extrapolation!r}"
            )
        # Finiteness is validated here for the same reason: NaN passes every
        # ordering comparison (rule 4's strict-increase check and rule 6's
        # in-range check are both false-negative on NaN), so a non-finite
        # break or bound would otherwise surface at build() as a low-level
        # rank/SVD failure with nothing pointing at the line that wrote it.
        # Name-mode breaks defer instead: a level name is resolved (and
        # validated) against an OrderedCategorical's declared levels, and a
        # numeric-axis build refuses it loudly before any arithmetic.
        if not isinstance(breaks, int | np.integer) and not _breaks_contain_names(breaks):
            requested = np.asarray(breaks, dtype=np.float64).ravel()
            if not np.all(np.isfinite(requested)):
                raise ValueError(
                    f"Piecewise breaks must be finite, got {_format_values(requested)}."
                )
        for bound_name, bound in (("lower", lower), ("upper", upper)):
            if bound is not None and not np.isfinite(float(bound)):
                raise ValueError(f"Piecewise {bound_name} must be finite, got {float(bound):.10g}.")
        self.breaks = breaks if isinstance(breaks, int | np.integer) else list(breaks)
        self.base = base
        self.strategy = strategy
        self.lower = lower
        self.upper = upper
        self.extrapolation = extrapolation
        self.degrees = _validate_degrees(degrees, breaks)
        # Set by OrderedCategorical on ITS deep copy when this spec is hosted
        # as an inner basis: the axis is then level positions 0..L-1, where
        # name resolution has happened and per-segment degrees are table-exact.
        self._on_level_axis = False
        # Fitted state, all resolved in build().
        self._knots: NDArray[np.float64] = np.empty(0, dtype=np.float64)
        self._base_index: int = 0
        self._base_knot: float | None = None
        self._non_base_indices: NDArray[np.intp] = np.empty(0, dtype=np.intp)
        self._strategy_actual: str = "explicit"
        self._n_breaks_requested: int | None = None
        # Segmented (degrees != all-1) fitted structure; None on the legacy path.
        self._seg_value_groups: list[NDArray[np.intp]] | None = None
        self._seg_base_group: int = 0
        self._seg_bubbles: list[tuple[int, int]] = []
        self._seg_retained: NDArray[np.intp] | None = None

    def __setstate__(self, state: dict) -> None:
        # Pickles predating segmented degrees lack the new state; default it
        # state so a restored spec keeps transforming on the legacy path.
        state = dict(state)
        state.setdefault("degrees", None)
        state.setdefault("_on_level_axis", False)
        state.setdefault("_seg_value_groups", None)
        state.setdefault("_seg_base_group", 0)
        state.setdefault("_seg_bubbles", [])
        state.setdefault("_seg_retained", None)
        self.__dict__.update(state)

    @property
    def _degrees_active(self) -> bool:
        """Whether any stated degree differs from 1 (the segmented build path).

        ``degrees=[1, ..., 1]`` deliberately routes through the legacy hat
        path: it states the default, and the default's basis is contractually
        bit-identical to the un-stated form.
        """
        return self.degrees is not None and any(d != 1 for d in self.degrees)

    def __repr__(self) -> str:
        breaks = self.breaks
        if isinstance(breaks, int | np.integer):
            shown = str(breaks)
        elif _breaks_contain_names(breaks):
            shown = "[" + ", ".join(repr(entry) for entry in breaks) + "]"
        else:
            shown = _format_values(np.asarray(breaks, dtype=np.float64))
        head = f"Piecewise(breaks={shown}, base={self.base!r})"
        if self.degrees is not None:
            head = f"{head[:-1]}, degrees={list(self.degrees)})"
        if self._knots.size:
            ref = float(self._knots[self._base_index])
            return f"{head[:-1]}, {self._knots.size} knots, ref={ref:.10g})"
        return head

    # ── basis ────────────────────────────────────────────────────────

    def _require_fitted(self) -> None:
        if self._knots.size == 0:
            raise RuntimeError("Piecewise has no knots yet: call build() before transform().")

    def _segment_index(self, x: NDArray[np.float64]) -> NDArray[np.intp]:
        """Index of the segment each row falls in, clamped to the outer segments."""
        n_seg = self._knots.size - 1
        return np.clip(np.searchsorted(self._knots, x, side="right") - 1, 0, n_seg - 1)

    def _policy_x(self, x: NDArray[np.float64], lo: float, hi: float) -> NDArray[np.float64]:
        """Apply the extrapolation policy against a resolved ``[lo, hi]`` range.

        Mirrors ``_spline_runtime.prepare_eval_points``: ``clip`` clamps,
        ``extend`` passes through, ``error`` raises with a scale-aware
        tolerance so a boundary value reconstructed in floating point does not
        refuse its own rated range.
        """
        if self.extrapolation == "clip":
            return np.clip(x, lo, hi)
        if self.extrapolation == "extend":
            return x
        scale = max(1.0, abs(lo), abs(hi), abs(hi - lo))
        tol = 1e-12 * scale
        if np.any(x < lo - tol) or np.any(x > hi + tol):
            raise ValueError(
                f"Piecewise received values outside the rated range "
                f"[{lo:.6g}, {hi:.6g}] with extrapolation='error'."
            )
        return x

    def _hat_values(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate all ``J+2`` hats at *x*, with the policy already applied.

        Three properties this construction is relied on for, all of them tested:
        the rows sum to exactly 1 everywhere *including* outside the knot span,
        evaluating at the knots gives exactly the identity, and outside the
        span the values continue the boundary segment's line -- which is the
        ``"extend"`` behaviour, and which ``"clip"`` never reaches because its
        x has already been clamped onto the boundary knots.
        """
        t = self._knots
        seg = self._segment_index(x)
        # UNCLIPPED on purpose: w < 0 below t_0 and w > 1 above t_{J+1}.  That is
        # precisely what continues the boundary segment's line past the last
        # knot; clipping w to [0, 1] here would silently re-impose "clip" on a
        # term whose stated policy is "extend", and would also make the outer
        # columns of a narrower-than-the-data pin misreport their support.
        w = (x - t[seg]) / (t[seg + 1] - t[seg])
        H = np.zeros((x.size, t.size), dtype=np.float64)
        rows = np.arange(x.size)
        H[rows, seg] = 1.0 - w
        H[rows, seg + 1] = w
        return H

    def _hat_basis(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate the hats at *x* under the extrapolation policy."""
        self._require_fitted()
        t = self._knots
        return self._hat_values(self._policy_x(x, float(t[0]), float(t[-1])))

    def _raw_basis_matrix(self, x: NDArray) -> NDArray[np.float64]:
        """Return the dense ``(n, J+2)`` raw hat basis, base column included.

        This is the duck-typed hook the editor's control-handle recovery looks
        for.  It returns **all** ``J + 2`` raw columns -- not the identifiable
        ``J + 1`` subset that ``transform`` returns -- so that a handle exists
        at every knot, the base knot included.
        """
        if self._degrees_active:
            # Handle-per-knot recovery assumes a coefficient per knot, which a
            # merged flat run and a curvature column both break.  Unreachable
            # from shipped surfaces (a segmented term exists only inside an
            # OrderedCategorical, whose editor display is level-based), so any
            # future caller fails loudly rather than dragging wrong handles.
            raise RuntimeError(
                "Piecewise with per-segment degrees has no per-knot handle basis; "
                "its columns are knot-value groups plus curvature columns."
            )
        return self._hat_basis(_finite_x(x))

    # ── segmented (degrees=) structure ───────────────────────────────

    def _resolve_segmented_structure(self) -> None:
        """Derive the segmented column structure from ``_knots``/``degrees``.

        Runs after ``_resolve_base``, which fixed the base KNOT; here the base
        COLUMN becomes the knot-value group containing it.  The structure is
        the grafted-polynomial space (per-segment degrees, C0 seams) spanned in
        a local form:

        - one knot-value column per group of knots, where a maximal run of
          consecutive degree-0 segments merges its knots into one group (the
          plateau: the segment is flat because its endpoint values are one
          coefficient);
        - for each segment of degree d >= 2, columns ``u**p - u`` (p = 2..d,
          ``u`` the within-segment coordinate), supported on that segment and
          vanishing at both seams.

        Every column is continuous and the curvature columns are zero at every
        knot, so seams are value-continuous BY CONSTRUCTION and evaluating at
        the knots still reads off the knot-value coefficients -- the same
        span as the one-sided truncated-power spelling (Smith 1979), in a
        better-conditioned local basis.
        """
        degrees = self.degrees
        assert degrees is not None
        n_knots = self._knots.size
        if len(degrees) != n_knots - 1:
            raise ValueError(
                f"Piecewise degrees state {len(degrees)} segment(s) but the resolved "
                f"knots make {n_knots - 1}."
            )
        groups: list[list[int]] = [[0]]
        for segment, degree in enumerate(degrees):
            right_knot = segment + 1
            if degree == 0:
                groups[-1].append(right_knot)
            else:
                groups.append([right_knot])
        self._seg_value_groups = [np.asarray(g, dtype=np.intp) for g in groups]
        knot_to_group = {int(k): gi for gi, g in enumerate(groups) for k in g}
        self._seg_base_group = knot_to_group[int(self._base_index)]
        self._seg_bubbles = [
            (segment, power)
            for segment, degree in enumerate(degrees)
            for power in range(2, degree + 1)
        ]
        n_value = len(groups)
        n_full = n_value + len(self._seg_bubbles)
        retained = [j for j in range(n_value) if j != self._seg_base_group]
        retained.extend(range(n_value, n_full))
        self._seg_retained = np.asarray(retained, dtype=np.intp)

    def _segmented_basis_full(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """All segmented columns (knot-value groups first, then curvature).

        *x* must already have the extrapolation policy applied; on the level
        axis every value is in range, so the policy never binds anyway.
        """
        groups = self._seg_value_groups
        assert groups is not None
        H = self._hat_values(x)
        cols = [H[:, g].sum(axis=1) for g in groups]
        t = self._knots
        for segment, power in self._seg_bubbles:
            t0, t1 = float(t[segment]), float(t[segment + 1])
            u = (x - t0) / (t1 - t0)
            inside = (x >= t0) & (x <= t1)
            cols.append(np.where(inside, u**power - u, 0.0))
        return np.column_stack(cols)

    def _segmented_transform(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Retained segmented columns under the extrapolation policy."""
        self._require_fitted()
        assert self._seg_retained is not None
        t = self._knots
        policy_x = self._policy_x(x, float(t[0]), float(t[-1]))
        return self._segmented_basis_full(policy_x)[:, self._seg_retained]

    # ── knot, base and support resolution ────────────────────────────

    def _resolve_knots(
        self,
        x: NDArray[np.float64],
        weights: NDArray[np.float64],
        sample_weight: NDArray | None,
    ) -> NDArray[np.float64]:
        """Resolve the knot vector (validation rules 2-6).

        Returns *x* with the extrapolation policy applied against the resolved
        range, because the policy binds at build() too: placement, support
        counting and the basis must all see the same rows the tariff rates.
        """
        breaks = self.breaks
        int_mode = isinstance(breaks, int | np.integer)

        # Rule 2 -- no breakpoints at all.  A piecewise term with no breaks is a
        # straight line, which is a feature this library already ships.
        if int_mode:
            if int(breaks) < 1:
                raise ValueError(
                    f"Piecewise(breaks={int(breaks)}) requests no breakpoints: a piecewise "
                    "term with no breaks is a straight line. Use Numeric() instead."
                )
            requested = np.empty(0, dtype=np.float64)
        else:
            requested = np.asarray(breaks, dtype=np.float64).ravel()
            if requested.size == 0:
                raise ValueError(
                    "Piecewise(breaks=[]) has no breakpoints: a piecewise term with no "
                    "breaks is a straight line. Use Numeric() instead."
                )

        # Rule 3 -- strategy is an extension point, validated even in sequence
        # mode (where it is not consulted) so a typo'd keyword still fails loud.
        if self.strategy not in _STRATEGIES:
            raise ValueError(
                f"strategy must be one of {sorted(_STRATEGIES)}, got {self.strategy!r}."
            )

        # Rule 4 -- a sequence must be strictly increasing.  This one is the
        # caller's own mistake, so it raises with the offending pair named.
        if not int_mode:
            bad = np.flatnonzero(np.diff(requested) <= 0.0)
            if bad.size:
                j = int(bad[0])
                raise ValueError(
                    "Piecewise breaks must be strictly increasing with no duplicates; "
                    f"breaks[{j}]={requested[j]:.10g} is not below breaks[{j + 1}]="
                    f"{requested[j + 1]:.10g}. Got {_format_values(requested)}."
                )

        # Rule 6a -- the rated range.  Resolved before int-mode placement, which
        # needs it to know which rows are inside the range.  Learned geometry
        # comes from positive-weight rows only (the documented
        # ``knot_geometry_data`` rule: a zero frequency weight is zero
        # replicated rows, so it must not widen the default boundaries or move
        # data-adaptive knots).  The fixed rules downstream are weight-linear,
        # so a zero-weight row already contributes nothing to them; it is only
        # the defaults and int-mode placement that must be scoped here.
        positive = weights > 0.0
        lo = float(x[positive].min()) if self.lower is None else float(self.lower)
        hi = float(x[positive].max()) if self.upper is None else float(self.upper)
        if not lo < hi:
            raise ValueError(
                f"Piecewise needs lower < upper, got lower={lo:.10g}, upper={hi:.10g}."
            )

        # The extrapolation policy binds here, before placement.  Under "clip"
        # rows outside [lo, hi] are grouped onto the boundary knots, so
        # Piecewise(breaks, upper=u) fits identically to a precomputed
        # x.clip(max=u); under "error" a narrower-than-the-data pin refuses
        # before any downstream rule can report on rows the term will not rate.
        x = self._policy_x(x, lo, hi)

        # Rule 5 -- int mode places breakpoints at exposure-weighted quantiles
        # of the positive-weight rows: a zero-weight row must not pull a
        # quantile, and snapping must not target an x value that only a
        # zero-weight row exhibits.
        if int_mode:
            n_requested = int(breaks)
            inside = (x >= lo) & (x <= hi) & positive
            x_q = x[inside]
            w_q = None if sample_weight is None else weights[inside]
            placed = weighted_quantile_knots(x_q, n_requested, 1.0, sample_weight=w_q)
            # Snap each quantile onto an observed value before deduplicating.
            # The quantile helper interpolates a weighted CDF, so on heaped data
            # it happily returns breakpoints in the gaps *between* the heaps --
            # measured on x heaped at multiples of 5, breaks=8 placed four
            # breakpoints inside gaps that bracket no rows at all, which rule 9
            # then rejects.  Turning a bare `breaks=8` into a hard error is the
            # library's mistake reported as the caller's.  Snapping also makes
            # every segment provably non-empty (each breakpoint is a value that
            # occurs) and is what actually collapses tied quantiles, which is the
            # documented reason realised J can fall short of the request.
            distinct = np.unique(x_q)
            if placed.size and distinct.size > 1:
                right = np.clip(np.searchsorted(distinct, placed), 1, distinct.size - 1)
                below, above = distinct[right - 1], distinct[right]
                placed = np.unique(np.where(placed - below <= above - placed, below, above))
            # A break on or outside a boundary is dropped: at a boundary it is a
            # knot that is already there, outside it is not in the rated range.
            placed = placed[(placed > lo) & (placed < hi)]
            if placed.size == 0:
                raise ValueError(
                    f"Piecewise(breaks={n_requested}) realised no breakpoints strictly "
                    f"inside (lower, upper) = ({lo:.10g}, {hi:.10g}): x has too few "
                    "distinct values in the rated range. State the breakpoints explicitly."
                )
            if placed.size < n_requested:
                warnings.warn(
                    f"Piecewise(breaks={n_requested}) realised only {placed.size} distinct "
                    f"breakpoint(s) ({n_requested} requested, {placed.size} realised): "
                    "tied weighted quantiles collapse on heaped x. The term therefore has "
                    f"edf = {placed.size + 1}, not {n_requested + 1}. "
                    f"Breakpoints: {_format_values(placed)}.",
                    UserWarning,
                    stacklevel=3,
                )
            requested = placed
            self._strategy_actual = self.strategy
            self._n_breaks_requested = n_requested
        else:
            self._strategy_actual = "explicit"
            self._n_breaks_requested = None

        # Rule 6b -- every break strictly inside the rated range.
        outside = requested[(requested <= lo) | (requested >= hi)]
        if outside.size:
            raise ValueError(
                f"Piecewise breaks must lie strictly inside (lower, upper) = "
                f"({lo:.10g}, {hi:.10g}); {_format_values(outside)} do not. A break at or "
                "past a boundary is a knot that is already there."
            )

        self._knots = np.concatenate(([lo], requested, [hi])).astype(np.float64)
        return x

    def _knot_mass(
        self,
        seg: NDArray[np.intp],
        frac: NDArray[np.float64],
        weights: NDArray[np.float64],
        *,
        signed: bool,
    ) -> NDArray[np.float64]:
        """Per-knot hat-carried mass, ``H.T @ w``, without materialising ``H``.

        A hat row has exactly two entries -- ``1 - frac`` at column ``seg`` and
        ``frac`` at column ``seg + 1`` -- so the column sums are two bincounts.
        ``signed=False`` takes ``|h|`` first, which is rule 8's form.
        """
        k = self._knots.size
        left = 1.0 - frac
        right = frac
        if not signed:
            left = np.abs(left)
            right = np.abs(right)
        mass = np.bincount(seg, weights=weights * left, minlength=k) + np.bincount(
            seg + 1, weights=weights * right, minlength=k
        )
        # asarray, not astype: with weights= the bincounts are already float64,
        # so this is a no-copy identity that states the dtype the type
        # checker's bincount stubs get wrong.
        return np.asarray(mass, dtype=np.float64)

    def _weighted_gram(
        self,
        seg: NDArray[np.intp],
        frac: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Accumulate the ``(J+2, J+2)`` weighted Gram ``H'WH`` from the hat pairs.

        Hat locality makes the Gram symmetric tridiagonal (a row touches only
        columns ``seg`` and ``seg + 1``), so three bincounts per chunk build it
        with no array taller than the chunk and nothing wider than ``J + 2``.
        """
        k = self._knots.size
        diag = np.zeros(k, dtype=np.float64)
        off = np.zeros(k - 1, dtype=np.float64)
        for start in range(0, seg.size, _GRAM_CHUNK_ROWS):
            block = slice(start, start + _GRAM_CHUNK_ROWS)
            s = seg[block]
            left = 1.0 - frac[block]
            right = frac[block]
            w = weights[block]
            diag += np.bincount(s, weights=w * left * left, minlength=k)
            diag += np.bincount(s + 1, weights=w * right * right, minlength=k)
            off += np.bincount(s, weights=w * left * right, minlength=k - 1)[: k - 1]
        gram = np.diag(diag)
        idx = np.arange(k - 1)
        gram[idx, idx + 1] = off
        gram[idx + 1, idx] = off
        return gram

    def _resolve_base(self, signed_mass: NDArray[np.float64]) -> None:
        """Resolve the base knot and the retained column indices (rule 7).

        *signed_mass* is the signed hat-carried mass ``H.T @ w`` per knot.
        """
        t = self._knots

        # Reuse a base fixed by an earlier build() when it still names a knot,
        # so a refit or a CV fold cannot silently redefine every coefficient.
        r: int | None = None
        if self._base_knot is not None:
            prior = np.flatnonzero(t == self._base_knot)
            if prior.size == 1:
                r = int(prior[0])

        if r is None:
            if isinstance(self.base, str):
                if self.base == "first":
                    r = 0
                elif self.base == "most_exposed":
                    # Signed mass, not |h|: the partition of unity makes
                    # h_j(x_i) row i's share of knot j, and a tail row's
                    # negative share is genuinely negative support.  With no
                    # sample_weight the weights are ones, so this is the knot
                    # carrying the most rows -- the model API always passes
                    # explicit weights (ones when the caller gave none), so a
                    # weights-absent special case would be unreachable there
                    # and would make the direct spec API disagree with it.
                    r = int(np.argmax(signed_mass))
                else:
                    raise ValueError(
                        f"base must be 'most_exposed', 'first', or a knot value, got "
                        f"{self.base!r}. Knots: {_format_values(t)}."
                    )
            else:
                target = float(self.base)
                matches = np.flatnonzero(t == target)
                if matches.size != 1:
                    raise ValueError(
                        f"base={target:.10g} must equal exactly one knot, matched "
                        f"{matches.size}. Knots: {_format_values(t)}."
                    )
                r = int(matches[0])

        self._base_index = r
        self._base_knot = float(t[r])
        self._non_base_indices = np.array([j for j in range(t.size) if j != r], dtype=np.intp)

    def _validate_support(
        self,
        x: NDArray[np.float64],
        seg: NDArray[np.intp],
        frac: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> None:
        """Check the data can carry the knots that were asked for (rules 8-11).

        Callers pass unique x values with per-value aggregated weights, plus
        the segment index and within-segment fraction that determine each
        row's two hat entries: every rule here is linear in the weights, and
        the rank probe sees the same weighted Gram either way.  Nothing in
        here materialises an ``(n, J+2)`` array except the rank probe's
        factor, whose row count is capped.
        """
        t = self._knots

        # Rule 8 -- a knot whose column is identically zero.  Written with |h_j|
        # rather than the signed sum: the signed sum is the true zero-column test
        # only where the hats are non-negative, and with lower/upper pinned
        # narrower than the data the tail rows carry negative entries, so a
        # non-zero column can sum to zero by cancellation.  |h_j| is that same
        # condition stated correctly on the mode this feature explicitly supports.
        mass = self._knot_mass(seg, frac, weights, signed=False)
        empty_knots = np.flatnonzero(mass == 0.0)
        if empty_knots.size:
            raise ValueError(
                f"Piecewise knot(s) {_format_values(t[empty_knots])} carry zero hat mass, "
                "so their design columns are identically zero and their coefficients are "
                f"unidentifiable. Per-knot mass over knots {_format_values(t)}: "
                f"{_format_values(mass)}."
            )

        # Rules 9 and 11 count IN-RANGE support only.  `_segment_index` clamps a
        # row outside [t_0, t_{J+1}] into the nearest boundary segment, which is
        # right for evaluating the basis -- that clamp is what continues the
        # boundary line -- and wrong for counting support: a row at x = 95 is not
        # evidence about the segment [45, 50].  Binning the clamped index lets a
        # rated range holding no observation at all pass both rules while the
        # message reports full weight, so the extrapolating rows are separated
        # out here and reported rather than credited.  Out-of-range rows exist
        # only under extrapolation="extend": under "clip" x arrives already
        # grouped onto the boundary knots -- those rows genuinely support the
        # boundary values, count as such, and the tails report never fires --
        # and under "error" the build has already refused them.
        n_seg = t.size - 1
        inside = (x >= t[0]) & (x <= t[-1])
        seg_weight = np.bincount(seg[inside], weights=weights[inside], minlength=n_seg)
        extrapolating = float(weights[~inside].sum())
        tails = (
            ""
            if extrapolating <= 0.0
            else (
                f" A further {extrapolating:.10g} of weight lies outside "
                f"[{t[0]:.10g}, {t[-1]:.10g}]; those rows load the boundary segments' "
                "slopes but support no segment, so they are excluded from the counts above."
            )
        )

        # Rule 9 -- a segment bracketing no data.  This is a DATA-SUPPORT rule,
        # not an identifiability rule: it rates a distinction the data cannot
        # support.  Positive weight per segment is *not* sufficient for
        # identifiability -- that is what rule 10 is for.
        starved = np.flatnonzero(seg_weight <= 0.0)
        if starved.size:
            edges = ", ".join(f"[{t[j]:.10g}, {t[j + 1]:.10g}]" for j in starved)
            raise ValueError(
                f"Piecewise segment(s) {edges} carry no in-range weight, so the term rates a "
                "distinction the data cannot support. Per-segment weight over segments "
                f"{_format_values(t)}: {_format_values(seg_weight)}.{tails}"
            )

        # Rule 10 -- the rule that actually catches degeneracy.  Rules 8 and 9
        # both pass on configurations whose retained columns are still rank
        # deficient (one distinct x per segment, for instance).
        #
        # The rank is taken on ALL J+2 weighted hat columns, not on the J+1
        # retained ones, because the intercept is part of the matrix the fit
        # inverts and the dropped hat is what makes room for it.  The hats are a
        # partition of unity, so the weighted intercept column sqrt(w) is exactly
        # their row sum: rank(sqrt(w) H) == rank([sqrt(w) 1, sqrt(w) H_retained]).
        # Checking only the retained columns misses the very degeneracy this rule
        # was added for -- one distinct x per segment gives J+1 independent
        # retained columns that are still collinear with the intercept.
        if self._degrees_active:
            self._validate_segmented_rank(x, weights)
            self._warn_thin_segments(t, seg_weight, tails)
            return
        n_cols = self._non_base_indices.size
        if x.size <= _RANK_PROBE_MAX_FACTOR_ROWS:
            # The FACTOR, never the Gram, is the reliable rank object: this
            # repository's recorded lesson is that eigensolver rank probes on a
            # Gram are not driver-stable near the cutoff.  The SVD on
            # sqrt(W)H therefore stays authoritative wherever its dense factor
            # is affordable, and 10_000 distinct values covers every banded
            # rating factor this feature was built for.
            H = self._hat_values(x)
            scaled = np.sqrt(weights)[:, None] * H
            rank = int(np.linalg.matrix_rank(scaled))
        else:
            # Backstop above the factor ceiling, not a peer of it.  The Gram
            # squares the condition number and its small eigenvalues are
            # eigensolver-driver-sensitive, so this arm runs with a GENEROUS
            # relative tolerance and only needs to catch EXACT deficiency --
            # an empty or single-point segment -- because rules 8/9/11 catch
            # graded near-deficiency structurally first.  For exact deficiency
            # the null eigenvalue is exact up to accumulation round-off,
            # orders below this tolerance under any LAPACK driver, so the
            # loose cut cannot flip a verdict either way.
            gram = self._weighted_gram(seg, frac, weights)
            eigenvalues = np.linalg.eigvalsh(gram)
            cutoff = _GRAM_PROBE_RELATIVE_TOL * max(float(eigenvalues[-1]), 0.0)
            rank = int(np.count_nonzero(eigenvalues > cutoff))
        if rank < n_cols + 1:
            raise ValueError(
                f"Piecewise design is rank deficient against the intercept: rank {rank} of "
                f"{n_cols + 1} ({n_cols} retained column(s) plus the intercept they are "
                f"identified against, deficiency {n_cols + 1 - rank}). The knots ask for more "
                f"distinct positions than x supplies. Knots: {_format_values(t)}."
            )

        # Rule 11 -- positive but thin. A warning, not an error: it is the
        # failure mode most likely to reach production silently, and the weight
        # per segment is the diagnostic that tells an actuary which breakpoint
        # to move.
        self._warn_thin_segments(t, seg_weight, tails)

    @staticmethod
    def _warn_thin_segments(
        t: NDArray[np.float64],
        seg_weight: NDArray[np.float64],
        tails: str,
    ) -> None:
        total = float(seg_weight.sum())
        thin = np.flatnonzero(seg_weight < _SMALL_SEGMENT_WEIGHT_FRACTION * total)
        if thin.size:
            edges = ", ".join(f"[{t[j]:.10g}, {t[j + 1]:.10g}]" for j in thin)
            warnings.warn(
                f"Piecewise segment(s) {edges} carry under "
                f"{_SMALL_SEGMENT_WEIGHT_FRACTION:.1%} of the in-range weight. "
                f"Per-segment weight over segments {_format_values(t)}: "
                f"{_format_values(seg_weight)}.{tails}",
                UserWarning,
                stacklevel=4,
            )

    def _validate_segmented_rank(
        self,
        x: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> None:
        """Segmented rule 10: refuse a degenerate segmented basis loudly.

        Two layers, both refusals and never regularization (the Item A
        discipline): a per-segment distinct-support count with a message that
        names the starved segment and its degree, then an SVD rank probe on
        the weighted FACTOR -- the driver-stable rank object this repository
        standardises on -- over ALL segmented columns.  The knot-value groups
        are a partition of unity, so the intercept is their row sum and full
        column rank is exactly identifiability against the intercept, the same
        argument as the legacy probe.  The row count is the distinct level
        positions (<= the level count), far under the factor ceiling, so the
        Gram fallback is never needed here.
        """
        degrees = self.degrees
        assert degrees is not None
        t = self._knots
        for segment, degree in enumerate(degrees):
            if degree < 2:
                continue
            inside = (x >= t[segment]) & (x <= t[segment + 1]) & (weights > 0.0)
            n_distinct = int(np.unique(x[inside]).size)
            if n_distinct < degree + 1:
                raise ValueError(
                    f"Piecewise segment [{t[segment]:.10g}, {t[segment + 1]:.10g}] states "
                    f"degree {degree}, which needs at least {degree + 1} distinct "
                    f"positions with positive weight in the segment; the data carry "
                    f"{n_distinct}. Lower the degree or move the break."
                )
        B = self._segmented_basis_full(x)
        scaled = np.sqrt(weights)[:, None] * B
        rank = int(np.linalg.matrix_rank(scaled))
        n_full = B.shape[1]
        if rank < n_full:
            assert self._seg_retained is not None
            raise ValueError(
                f"Piecewise segmented design is rank deficient against the intercept: "
                f"rank {rank} of {n_full} ({self._seg_retained.size} retained column(s) "
                f"plus the intercept they are identified against, deficiency "
                f"{n_full - rank}). The stated degrees ask for more distinct positions "
                f"than x supplies. Knots: {_format_values(t)}; degrees: "
                f"{list(degrees)}."
            )

    # ── feature-spec contract ────────────────────────────────────────

    def build(
        self,
        x: NDArray[np.floating],
        sample_weight: NDArray[np.floating] | None = None,
    ) -> GroupInfo:
        """Resolve knots and base from *x*, validate, and return the ``J+1`` columns.

        Validation runs in a fixed order, and the order is part of the contract
        because several rules fire together on a degenerate input: (1) finite x,
        (2) non-empty breaks, (3) known strategy, (4) increasing sequence breaks,
        (6a) lower < upper, then the extrapolation policy binds (clip groups the
        out-of-range rows onto the boundary knots, error refuses them), (5)
        int-mode placement, (6b) breaks strictly inside, (7) base resolves to
        one knot, (8) no zero-mass knot, (9) no in-range empty segment, (10)
        full column rank against the intercept, (11) thin-segment warning.

        Level-axis-only declarations refuse before any of that: band-name
        breaks and per-segment ``degrees=`` exist only where an
        ``OrderedCategorical`` hosts this spec as its ``basis=``.
        """
        if not self._on_level_axis:
            if _breaks_contain_names(self.breaks):
                names = [entry for entry in self.breaks if isinstance(entry, str)]
                raise ValueError(
                    f"Piecewise breaks {names!r} are stated as band names, which only "
                    "resolve against the declared levels of an OrderedCategorical "
                    "(pass this spec as its basis=). On a numeric axis, state the "
                    "breaks as numbers."
                )
            if self._degrees_active:
                raise ValueError(
                    f"Piecewise degrees={list(self.degrees or ())} on a numeric axis: "
                    "per-segment degrees are only legal when this Piecewise is the "
                    "basis= of an OrderedCategorical, where the export contract is "
                    "one row per band and the table is exact at any degree. On a "
                    "numeric axis the exported workbook is exact under linear "
                    "interpolation only at degree 1; for smooth curvature use "
                    "Spline, or the hinge composition documented in the guide."
                )
        x = _finite_x(x)
        weights = _conforming_weights(sample_weight, x.size)
        x = self._resolve_knots(x, weights, sample_weight)
        # Deduplicate before anything scales with n.  Rating variables are
        # heaped, so the distinct values are typically dozens; every rule below
        # is weight-linear, so unique values with per-value aggregated weights
        # give the same masses, the same segment counts and the same weighted
        # Gram -- hence the same rank verdict -- as the row-level design.  This
        # is deduplication, not binning: no discretisation error exists to
        # introduce.  The policy is already applied to x, so the raw segment
        # arithmetic is the right one here; _hat_basis would just clamp again.
        x_unique, inverse = np.unique(x, return_inverse=True)
        # asarray is a no-copy identity here (bincount with weights= already
        # returns float64); it exists because the checker's bincount stubs
        # report an integer array, which every float64-annotated consumer of
        # w_agg would then trip over.
        w_agg = np.asarray(
            np.bincount(inverse, weights=weights, minlength=x_unique.size),
            dtype=np.float64,
        )
        # Everything downstream works from the two hat entries per row -- the
        # segment index and the within-segment fraction -- rather than a dense
        # (n_unique, J+2) basis.  On a genuinely continuous x (n_unique ~ n)
        # the dense route peaked at three full-height arrays for a basis with
        # two non-zeros per row; these two vectors are the same information at
        # O(n) and the identical arithmetic (`_hat_values` computes exactly
        # `1 - frac` and `frac`), so every emitted float is bit-identical.
        t = self._knots
        seg = self._segment_index(x_unique)
        frac = (x_unique - t[seg]) / (t[seg + 1] - t[seg])
        self._resolve_base(self._knot_mass(seg, frac, w_agg, signed=True))
        if self._degrees_active:
            self._resolve_segmented_structure()
            self._validate_support(x_unique, seg, frac, w_agg)
            assert self._seg_retained is not None
            # Dense on purpose: the segmented path exists only on a level axis,
            # where the distinct rows number at most the level count.
            columns_dense = self._segmented_basis_full(x_unique)[:, self._seg_retained][inverse]
            return GroupInfo(
                columns=columns_dense,
                n_cols=int(columns_dense.shape[1]),
            )
        self._seg_value_groups = None
        self._seg_bubbles = []
        self._seg_retained = None
        self._validate_support(x_unique, seg, frac, w_agg)
        # Emitted sparse (a hat row has at most two non-zeros) so the builder's
        # sparse branch can re-detect the repeated rows and store the design
        # one distinct row deep.  The gather reconstructs the exact per-row
        # values: equal x means bit-identical hat arithmetic.
        columns = self._emit_unique_csr(seg, frac)[inverse]
        return GroupInfo(
            columns=columns,
            n_cols=int(columns.shape[1]),
            supports_row_compression=True,
        )

    def _emit_unique_csr(
        self,
        seg: NDArray[np.intp],
        frac: NDArray[np.float64],
    ) -> sp.csr_matrix:
        """Emit the retained hat columns over the unique rows as canonical CSR.

        Built straight from the two per-row entries, never via a dense
        intermediate.  ``eliminate_zeros`` matches what a dense->CSR
        conversion produced historically: an x sitting exactly on a knot puts
        an exact ``0.0`` in its neighbour knot's entry, and the dense route
        never stored it.
        """
        n = seg.size
        r = self._base_index
        row = np.tile(np.arange(n, dtype=np.intp), 2)
        col = np.concatenate([seg, seg + 1])
        data = np.concatenate([1.0 - frac, frac])
        keep = col != r
        row, col, data = row[keep], col[keep], data[keep]
        col = np.where(col > r, col - 1, col)
        emitted = sp.csr_matrix(
            (data, (row, col)),
            shape=(n, self._knots.size - 1),
            dtype=np.float64,
        )
        emitted.eliminate_zeros()
        return emitted

    def transform(self, x: NDArray) -> NDArray[np.float64]:
        """Evaluate the identifiable retained columns on new data.

        Out-of-range x follows the term's ``extrapolation`` policy.  On the
        legacy (all-degree-1) path these are the ``J+1`` non-base hats; on the
        segmented path, the non-base knot-value groups then the curvature
        columns.
        """
        if self._degrees_active:
            return self._segmented_transform(_finite_x(x))
        return self._hat_basis(_finite_x(x))[:, self._non_base_indices]

    def score(self, x: NDArray, beta: NDArray[np.floating]) -> NDArray[np.floating]:
        """Score the fitted piecewise contribution directly on new data."""
        beta = np.asarray(beta, dtype=np.float64).ravel()
        return self.transform(x) @ beta

    def reconstruct(self, beta: NDArray[np.floating]) -> dict[str, Any]:
        """Coefficients -> per-knot relativities and derived segment slopes."""
        self._require_fitted()
        beta = np.asarray(beta, dtype=np.float64).ravel()
        if self._degrees_active:
            return self._reconstruct_segmented(beta)
        t = self._knots
        log_rel = np.zeros(t.size, dtype=np.float64)
        log_rel[self._non_base_indices] = beta
        # Derived, not fitted: the slopes carry no independent p-value.
        slopes = np.diff(log_rel) / np.diff(t)
        return {
            # "x" is the generic contract key every x-bearing spec returns, and
            # the dispatchers that render a fitted curve -- `model.relativities()`
            # above all -- are if/elif chains with no else, so a spec that omits
            # it is dropped from the output with no error and no warning.
            # "knots" is the same vector under the name that means something
            # here; both are exported so neither caller has to know about the
            # other.
            "x": t.copy(),
            "knots": t.copy(),
            "base_knot": float(t[self._base_index]),
            "base_index": int(self._base_index),
            "log_relativity": log_rel,
            "relativity": np.exp(log_rel),
            "slopes": slopes,
            "boundary_slopes": (float(slopes[0]), float(slopes[-1])),
            # The out-of-range rule travels with the numbers: the exported
            # workbook and the editor's offset scoring both need to know
            # whether the boundary slopes continue or the end values hold.
            "extrapolation": self.extrapolation,
        }

    def _reconstruct_segmented(self, beta: NDArray[np.float64]) -> dict[str, Any]:
        """Segmented reconstruct: a display grid dense enough for curvature.

        Reachable only from an ``OrderedCategorical`` host, which overwrites
        the relativities with base-shifted values and derives its own per-level
        table; the keys here serve the shared curve-display contract ("x",
        "log_relativity", "relativity") plus the knot bookkeeping.  Per-segment
        slopes are deliberately absent: a curved segment has no single slope.
        """
        degrees = self.degrees
        assert degrees is not None
        t = self._knots
        pieces = []
        for segment, degree in enumerate(degrees):
            n_pts = 2 if degree <= 1 else 25
            pieces.append(np.linspace(float(t[segment]), float(t[segment + 1]), n_pts))
        x_grid = np.unique(np.concatenate(pieces))
        log_rel = self.score(x_grid, beta)
        knot_log_rel = self.score(t, beta)
        return {
            "x": x_grid,
            "knots": t.copy(),
            "base_knot": float(t[self._base_index]),
            "base_index": int(self._base_index),
            "log_relativity": log_rel,
            "relativity": np.exp(log_rel),
            "knot_log_relativity": knot_log_rel,
            "degrees": list(degrees),
            "extrapolation": self.extrapolation,
        }

    # ── structural contrast rows (level-axis reporting) ──────────────

    def ordered_structural_rows(self) -> list[StructuralContrastRow]:
        """Structural contrasts of a fitted piecewise term, for summaries.

        One slope-change contrast per stated break and one curvature family
        per segment of degree >= 2.  This is the fixed-knot truncated-power
        inference vocabulary: with the knots stated as inputs the design is
        linear in every parameter, the slope change at a stated join is the
        coefficient of its plus-function, and both hypotheses are ordinary
        Wald/F tests (Smith 1979, "Splines as a useful and convenient
        statistical tool", Am. Statist. 33(2):57-62; Sprent 1961 for the
        known-changeover two-phase case).  Deliberately NOT per-segment
        per-power z-rows: under C0 seams the segments share their joint
        values, so within-segment orthogonal components are not free
        parameters and that clean-z geometry does not exist here.

        Contrast vectors are stated over the retained columns of
        ``transform`` and are invariant to the base-column choice.
        """
        self._require_fitted()
        t = self._knots
        n_segments = t.size - 1
        degrees = self.degrees if self.degrees is not None else (1,) * n_segments
        if self._degrees_active:
            groups = self._seg_value_groups
            retained = self._seg_retained
            bubbles = self._seg_bubbles
            base_group = self._seg_base_group
            assert groups is not None and retained is not None
        else:
            groups = [np.asarray([k], dtype=np.intp) for k in range(t.size)]
            bubbles = []
            base_group = int(self._base_index)
            retained = np.asarray([g for g in range(t.size) if g != base_group], dtype=np.intp)
        knot_to_group = {int(k): gi for gi, g in enumerate(groups) for k in g}
        n_value = len(groups)
        # Retained position of each full column (value groups then bubbles).
        position_of = {int(full): pos for pos, full in enumerate(retained)}
        bubble_position = {
            (segment, power): position_of[n_value + i] for i, (segment, power) in enumerate(bubbles)
        }

        n_retained = len(position_of)

        def add_value(c: NDArray[np.float64], knot: int, coefficient: float) -> None:
            group = knot_to_group[knot]
            if group != base_group:
                c[position_of[group]] += coefficient

        rows: list[StructuralContrastRow] = []
        for k in range(1, t.size - 1):
            c = np.zeros(n_retained, dtype=np.float64)
            if degrees[k] > 0:
                h = float(t[k + 1] - t[k])
                add_value(c, k + 1, 1.0 / h)
                add_value(c, k, -1.0 / h)
                for segment, power in bubbles:
                    if segment == k:
                        # d/dx (u**p - u) at u=0 is -1/h for every p >= 2.
                        c[bubble_position[(segment, power)]] += -1.0 / h
            if degrees[k - 1] > 0:
                h = float(t[k] - t[k - 1])
                add_value(c, k, -1.0 / h)
                add_value(c, k - 1, 1.0 / h)
                for segment, power in bubbles:
                    if segment == k - 1:
                        # d/dx (u**p - u) at u=1 is (p - 1)/h; subtracted as
                        # part of the left slope.
                        c[bubble_position[(segment, power)]] -= (power - 1.0) / h
            rows.append(StructuralContrastRow(kind="slope_change", index=k, contrast=c))
        for segment, degree in enumerate(degrees):
            if degree < 2:
                continue
            columns = tuple(bubble_position[(segment, power)] for power in range(2, degree + 1))
            rows.append(
                StructuralContrastRow(kind="curvature", index=segment, column_indices=columns)
            )
        return rows
