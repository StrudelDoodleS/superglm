"""Piecewise-linear feature: a hat basis on stated breakpoints.

Continuous but deliberately not smooth.  Free joins are already a binned
``Categorical`` and smooth joins are already a ``Spline``; this occupies the one
remaining cell of that lattice.  Every coefficient is the log-relativity of a
knot against the base knot, so the summary row, the editor handle and the
workbook cell are the same number.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
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
    """Return per-row weights as float64, defaulting to ones (part of rule 1)."""
    if sample_weight is None:
        return np.ones(n_rows, dtype=np.float64)
    weights = np.asarray(sample_weight, dtype=np.float64).ravel()
    if weights.size != n_rows:
        raise ValueError(f"sample_weight must have length {n_rows}, got {weights.size}.")
    return weights


def _format_values(values: NDArray) -> str:
    """Render an array of knot or weight values for an error message."""
    return "[" + ", ".join(f"{float(v):.10g}" for v in np.asarray(values).ravel()) + "]"


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
    base : float or {'most_exposed', 'first'}
        Reference knot, mirroring ``Categorical``.  ``'most_exposed'`` picks the
        knot carrying the largest share of the weight; with no ``sample_weight``
        it falls back to the first knot.  A float must equal exactly one knot.
    strategy : {'quantile'}
        Placement rule, consulted only when *breaks* is an int.
    lower, upper : float, optional
        Pin the outermost knots.  Default ``min(x)`` / ``max(x)``.  Pinning
        **wider** than the data states a rated range the tariff must cover.
        Pinning **narrower** is allowed too, and then rows outside
        ``[lower, upper]`` load the linear tails: their leverage lands entirely
        on the two boundary segments and therefore dominates the boundary
        slopes.  That is a modelling choice, not an error, so it is documented
        rather than blocked.

    Notes
    -----
    Beyond ``[t_0, t_{J+1}]`` the function continues at the boundary slope.
    That is not a differentiator over this library's splines, which already
    extrapolate linearly, and holding flat beyond the boundary is an
    established alternative.  What ``Piecewise`` adds is that the slope is
    *stated*: ``lower`` / ``upper`` pin where the boundary segments start and
    the exported table prints the two boundary slopes, so the rule outside the
    tabulated range is reproducible by hand.

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
        breaks: Sequence[float] | int,
        *,
        base: float | str = "most_exposed",
        strategy: str = "quantile",
        lower: float | None = None,
        upper: float | None = None,
    ):
        self.breaks = breaks
        self.base = base
        self.strategy = strategy
        self.lower = lower
        self.upper = upper
        # Fitted state, all resolved in build().
        self._knots: NDArray[np.float64] = np.empty(0, dtype=np.float64)
        self._base_index: int = 0
        self._base_knot: float | None = None
        self._non_base_indices: NDArray[np.intp] = np.empty(0, dtype=np.intp)
        self._strategy_actual: str = "explicit"
        self._n_breaks_requested: int | None = None

    def __repr__(self) -> str:
        breaks = self.breaks
        shown = breaks if isinstance(breaks, int | np.integer) else _format_values(breaks)
        head = f"Piecewise(breaks={shown}, base={self.base!r})"
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

    def _hat_basis(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate all ``J+2`` hats at *x* (assumed already finite float64).

        Three properties this construction is relied on for, all of them tested:
        the rows sum to exactly 1 everywhere *including* outside the knot span,
        ``_hat_basis(knots)`` is exactly the identity, and outside the span the
        values continue the boundary segment's line.
        """
        self._require_fitted()
        t = self._knots
        seg = self._segment_index(x)
        # UNCLIPPED on purpose: w < 0 below t_0 and w > 1 above t_{J+1}.  That is
        # precisely what continues the boundary segment's line past the last
        # knot; clipping w to [0, 1] would hold the function flat outside the
        # span instead, and would also make the outer columns of a
        # narrower-than-the-data pin misreport their support.
        w = (x - t[seg]) / (t[seg + 1] - t[seg])
        H = np.zeros((x.size, t.size), dtype=np.float64)
        rows = np.arange(x.size)
        H[rows, seg] = 1.0 - w
        H[rows, seg + 1] = w
        return H

    def _raw_basis_matrix(self, x: NDArray) -> NDArray[np.float64]:
        """Return the dense ``(n, J+2)`` raw hat basis, base column included.

        This is the duck-typed hook the editor's control-handle recovery looks
        for.  It returns **all** ``J + 2`` raw columns -- not the identifiable
        ``J + 1`` subset that ``transform`` returns -- so that a handle exists
        at every knot, the base knot included.
        """
        return self._hat_basis(_finite_x(x))

    # ── knot, base and support resolution ────────────────────────────

    def _resolve_knots(
        self,
        x: NDArray[np.float64],
        weights: NDArray[np.float64],
        sample_weight: NDArray | None,
    ) -> None:
        """Resolve the knot vector (validation rules 2-6)."""
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
        # needs it to know which rows are inside the range.
        lo = float(x.min()) if self.lower is None else float(self.lower)
        hi = float(x.max()) if self.upper is None else float(self.upper)
        if not lo < hi:
            raise ValueError(
                f"Piecewise needs lower < upper, got lower={lo:.10g}, upper={hi:.10g}."
            )

        # Rule 5 -- int mode places breakpoints at exposure-weighted quantiles.
        if int_mode:
            n_requested = int(breaks)
            inside = (x >= lo) & (x <= hi)
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

    def _resolve_base(
        self,
        H: NDArray[np.float64],
        weights: NDArray[np.float64],
        sample_weight: NDArray | None,
    ) -> None:
        """Resolve the base knot and the retained column indices (rule 7)."""
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
                    if sample_weight is None:
                        # Mirror Categorical: with no weights there is no exposure
                        # to be most of, so fall back to the first knot.
                        r = 0
                    else:
                        # Signed mass, not |h|: the partition of unity makes
                        # h_j(x_i) row i's share of knot j, and a tail row's
                        # negative share is genuinely negative support.
                        r = int(np.argmax(H.T @ weights))
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
        H: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> None:
        """Check the data can carry the knots that were asked for (rules 8-11)."""
        t = self._knots

        # Rule 8 -- a knot whose column is identically zero.  Written with |h_j|
        # rather than the signed sum: the signed sum is the true zero-column test
        # only where the hats are non-negative, and with lower/upper pinned
        # narrower than the data the tail rows carry negative entries, so a
        # non-zero column can sum to zero by cancellation.  |h_j| is that same
        # condition stated correctly on the mode this feature explicitly supports.
        mass = np.abs(H).T @ weights
        empty_knots = np.flatnonzero(mass == 0.0)
        if empty_knots.size:
            raise ValueError(
                f"Piecewise knot(s) {_format_values(t[empty_knots])} carry zero hat mass, "
                "so their design columns are identically zero and their coefficients are "
                f"unidentifiable. Per-knot mass over knots {_format_values(t)}: "
                f"{_format_values(mass)}."
            )

        # Rule 9 -- a segment bracketing no data.  This is a DATA-SUPPORT rule,
        # not an identifiability rule: it rates a distinction the data cannot
        # support.  Positive weight per segment is *not* sufficient for
        # identifiability -- that is what rule 10 is for.
        seg = self._segment_index(x)
        n_seg = t.size - 1
        seg_weight = np.bincount(seg, weights=weights, minlength=n_seg)
        starved = np.flatnonzero(seg_weight <= 0.0)
        if starved.size:
            edges = ", ".join(f"[{t[j]:.10g}, {t[j + 1]:.10g}]" for j in starved)
            raise ValueError(
                f"Piecewise segment(s) {edges} carry no weight, so the term rates a "
                "distinction the data cannot support. Per-segment weight over segments "
                f"{_format_values(t)}: {_format_values(seg_weight)}."
            )

        # Rule 10 -- the rule that actually catches degeneracy.  Rules 8 and 9
        # both pass on configurations whose retained columns are still rank
        # deficient (one distinct x per segment, for instance).
        n_cols = self._non_base_indices.size
        scaled = np.sqrt(weights)[:, None] * H[:, self._non_base_indices]
        rank = int(np.linalg.matrix_rank(scaled))
        if rank < n_cols:
            raise ValueError(
                f"Piecewise design is rank deficient: rank {rank} of {n_cols} retained "
                f"columns (deficiency {n_cols - rank}). The knots ask for more distinct "
                f"positions than x supplies. Knots: {_format_values(t)}."
            )

        # Rule 11 -- positive but thin. A warning, not an error: it is the
        # failure mode most likely to reach production silently, and the weight
        # per segment is the diagnostic that tells an actuary which breakpoint
        # to move.
        total = float(seg_weight.sum())
        thin = np.flatnonzero(seg_weight < _SMALL_SEGMENT_WEIGHT_FRACTION * total)
        if thin.size:
            edges = ", ".join(f"[{t[j]:.10g}, {t[j + 1]:.10g}]" for j in thin)
            warnings.warn(
                f"Piecewise segment(s) {edges} carry under "
                f"{_SMALL_SEGMENT_WEIGHT_FRACTION:.1%} of the total weight. "
                f"Per-segment weight over segments {_format_values(t)}: "
                f"{_format_values(seg_weight)}.",
                UserWarning,
                stacklevel=3,
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
        (6a) lower < upper, (5) int-mode placement, (6b) breaks strictly inside,
        (7) base resolves to one knot, (8) no zero-mass knot, (9) no empty
        segment, (10) full column rank, (11) thin-segment warning.
        """
        x = _finite_x(x)
        weights = _conforming_weights(sample_weight, x.size)
        self._resolve_knots(x, weights, sample_weight)
        H = self._hat_basis(x)
        self._resolve_base(H, weights, sample_weight)
        self._validate_support(x, H, weights)
        columns = H[:, self._non_base_indices]
        return GroupInfo(columns=columns, n_cols=int(columns.shape[1]))

    def transform(self, x: NDArray) -> NDArray[np.float64]:
        """Evaluate the identifiable ``J+1`` hat columns on new data."""
        return self._hat_basis(_finite_x(x))[:, self._non_base_indices]

    def score(self, x: NDArray, beta: NDArray[np.floating]) -> NDArray[np.floating]:
        """Score the fitted piecewise contribution directly on new data."""
        beta = np.asarray(beta, dtype=np.float64).ravel()
        return self.transform(x) @ beta

    def reconstruct(self, beta: NDArray[np.floating]) -> dict[str, Any]:
        """Coefficients -> per-knot relativities and derived segment slopes."""
        self._require_fitted()
        beta = np.asarray(beta, dtype=np.float64).ravel()
        t = self._knots
        log_rel = np.zeros(t.size, dtype=np.float64)
        log_rel[self._non_base_indices] = beta
        # Derived, not fitted: the slopes carry no independent p-value.
        slopes = np.diff(log_rel) / np.diff(t)
        return {
            "knots": t.copy(),
            "base_knot": float(t[self._base_index]),
            "base_index": int(self._base_index),
            "log_relativity": log_rel,
            "relativity": np.exp(log_rel),
            "slopes": slopes,
            "boundary_slopes": (float(slopes[0]), float(slopes[-1])),
        }
