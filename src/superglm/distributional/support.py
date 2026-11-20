"""Predictions that lean on what a SuperLSS fit does not determine (``diagnostics.estimability``).

A row's linear predictor for parameter ``b`` is ``r_b beta_b``, with ``r_b`` its row of
``[1, X_b]`` (its intercept column, if the predictor has one).  Along a direction ``v`` the fit
does not determine, it moves by ``r_b v_b``, so the row is off support for that kind of
direction when ``|| r_b Q_kb ||_2 > tol || r_b ||_2``, read in the Jacobi scale of the fit's
terminal rank (the dual row ``r_b / s``; ``Q_kb`` an orthonormal basis of the kind's part in
block ``b``), as SuperGLM's rows are read (R's ``predict.lm``, ``rankdeficient = "warnif"``).
The kinds are the fit's ``design_aliases`` (``"alias"``) and ``unfitted`` directions
(``DistributionalFitState``).  A fit with neither reads no row.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

import superglm.distributional.prediction_design as prediction_design
from superglm._frame import EagerFrame, FrameLike, as_eager_frame
from superglm._roundoff import gamma_up
from superglm._warn import _PACKAGE_PREFIX
from superglm.diagnostics.estimability import (
    KINDS,
    NonEstimablePredictionWarning,
    disclosure,
    off_support,
)

# The owner's share floor (``identifiability._SHARE_FLOOR``): a block part below it is rounding.
_SHARE_FLOOR = float(np.sqrt(np.finfo(np.float64).eps))


def _jacobi(fitted: Any) -> NDArray:
    """The Jacobi scale of the fit's terminal rank, a zero column read at one."""
    scale = np.asarray(fitted.fit_state.solver_result.terminal_rank.column_scale, dtype=float)
    return np.where(scale > 0.0, scale, 1.0)


def _block_bases(fitted: Any) -> dict[str, list[NDArray]] | None:
    """Each kind's orthonormal basis per predictor block, in the Jacobi scale; ``None`` if none."""
    state = fitted.fit_state
    spans = {
        kind: span
        for kind, span in (
            ("alias", getattr(state, "design_aliases", None)),
            ("unfitted", getattr(state, "unfitted", None)),
        )
        if span is not None and span.shape[1]
    }
    if not spans:
        return None
    scale = _jacobi(fitted)
    bases: dict[str, list[NDArray]] = {}
    for kind, span in spans.items():
        scaled = np.asarray(span, dtype=np.float64) * scale[:, None]
        bases[kind] = []
        for predictor in state.layout.predictors:
            block = scaled[predictor.coefficient_slice]
            left, singular, _ = np.linalg.svd(block, full_matrices=False)
            bases[kind].append(left[:, singular > _SHARE_FLOOR])
    return bases


def _unfitted_blocks(fitted: Any) -> list[NDArray] | None:
    """The unfitted span per predictor block on the coordinates it reaches; ``None`` if none.

    The span is held on the coordinates it reaches above the share floor
    (``DistributionalFitState.unfitted_coordinates``), the ones whose standard errors the
    summary withholds; below it an entry is the rounding of the basis, not the direction.
    """
    state = fitted.fit_state
    span = getattr(state, "unfitted", None)
    if span is None or span.shape[1] == 0:
        return None
    held = np.where(state.unfitted_coordinates[:, None], np.asarray(span, dtype=np.float64), 0.0)
    return [held[predictor.coefficient_slice] for predictor in state.layout.predictors]


def _reaches(design: NDArray, span: NDArray) -> NDArray[np.bool_]:
    """Which rows of ``design`` reach ``span``: ``|r'U|`` above the rounding of forming it.

    ``r'U`` sums ``k`` products, ``k`` the coordinates ``span`` holds, so its rounding is at most
    ``gamma_k |r|'|U|`` (Higham 2002, eq. 3.5), a bound no diagonal rescaling moves; each row is
    first scaled by the power of two of its largest entry, which is exact.  Such a row's
    prediction has a variance along the direction (``(r'u)^2`` over its curvature) that the
    fit's covariance leaves out.
    """
    count = int(np.count_nonzero(np.any(span != 0.0, axis=1)))
    if count == 0:
        return np.zeros(design.shape[0], dtype=bool)
    exponent = np.frexp(np.abs(design).max(axis=1, initial=0.0))[1]
    scaled = np.ldexp(design, -exponent[:, None])
    bound = gamma_up(count) * (np.abs(scaled) @ np.abs(span))
    return np.any(np.abs(scaled @ span) > bound, axis=1)


@dataclass(frozen=True)
class SupportRows:
    """What each row's prediction leans on, per parameter (``n x P``).

    ``labels`` is its off-support kind (``off_support``, R's ``1e-6``, ``""`` on support), and
    ``unfitted`` whether it reaches an unfitted direction at all (``_reaches``): any variance
    of such a row's predictor, a standard error, a posterior draw or a band, is withheld.
    """

    labels: NDArray
    unfitted: NDArray[np.bool_]


def support_rows(fitted: Any, X: FrameLike | EagerFrame) -> SupportRows | None:
    """The rows' ``SupportRows``; ``None`` on a fit with no alias and no unfitted direction.

    One pass over the rows' designs, a block of rows at a time (``map_design_blocks``), and none
    on a fit with neither.
    """
    bases = _block_bases(fitted)
    if bases is None:
        return None
    unfitted = _unfitted_blocks(fitted)
    layout = fitted.layout
    scale = _jacobi(fitted)

    def read(rows: slice, local: dict[str, NDArray]) -> tuple[slice, NDArray, NDArray]:
        width = (rows.stop - rows.start, len(layout.predictors))
        out, reach = np.full(width, "", dtype=object), np.zeros(width, dtype=bool)
        for index, predictor in enumerate(layout.predictors):
            design = np.asarray(local[predictor.name], dtype=np.float64)
            # each row by the power of two of its largest entry, which is exact and leaves the
            # share's ratio as it was, so a covariate of 1e300 squares to no infinity
            exponent = np.frexp(np.abs(design).max(axis=1, initial=0.0))[1]
            dual = np.ldexp(design, -exponent[:, None]) / scale[predictor.coefficient_slice]
            projections = {
                kind: dual @ blocks[index] for kind, blocks in bases.items() if blocks[index].size
            }
            if projections:
                out[:, index] = off_support(projections, np.linalg.norm(dual, axis=1))
            if unfitted is not None:
                reach[:, index] = _reaches(design, unfitted[index])
        return rows, out, reach

    frame = as_eager_frame(X)
    labels = np.full((len(frame), len(layout.predictors)), "", dtype=object)
    reached = np.zeros(labels.shape, dtype=bool)
    for rows, out, reach in prediction_design.map_design_blocks(
        frame, fitted.compiled_predictors, layout, read
    ):
        labels[rows], reached[rows] = out, reach
    return SupportRows(labels, reached)


def row_labels(labels: NDArray) -> NDArray:
    """Each row's kind: the first in ``KINDS`` of its parameters' kinds, ``""`` on support."""
    labels = np.asarray(labels, dtype=object)
    rows = np.full(labels.shape[0], "", dtype=object)
    for kind in KINDS:
        rows[(rows == "") & np.any(labels == kind, axis=1)] = kind
    return rows


def disclose(
    support: SupportRows | None,
    no_mean: NDArray | None = None,
    *,
    withheld: str | None = None,
) -> NDArray[np.bool_]:
    """Warn once for the rows ``support`` puts off support; return which rows they are.

    ``no_mean`` marks the rows with no mean at the returned predictor: the off-support ones are
    named in this warning alone (``estimability.disclosure``).  ``withheld`` names what a
    variance-reading caller sets to NaN on the off-support rows and on every row that reaches
    an unfitted direction; the same warning names those rows, so no NaN is silent.
    """
    if support is None:
        return np.zeros(0 if no_mean is None else np.shape(no_mean)[0], dtype=bool)
    labels = row_labels(support.labels)
    off = labels != ""
    message, positions, kinds, absent = "", np.zeros(0, dtype=np.intp), (), 0
    if np.any(off):
        base = disclosure(labels, no_mean)
        message, positions, kinds, absent = str(base), base.rows, base.kinds, base.no_mean
        if withheld is not None:
            message += f" Their {withheld} are NaN."
    further = np.flatnonzero(support.unfitted.any(axis=1) & ~off)
    if withheld is not None and further.size:
        shown = ", ".join(str(int(row)) for row in further[:5]) + (
            ", ..." if further.size > 5 else ""
        )
        plural = "row reaches" if further.size == 1 else "rows reach"
        message = (
            f"{message} {further.size} {'further ' if message else ''}{plural} a direction the "
            f"data identify but the fit did not fit, whose variance the fit does not have: their "
            f"{withheld} are NaN (rows {shown})."
        ).strip()
    if message:
        warning = NonEstimablePredictionWarning(message, positions, kinds, absent)
        warnings.warn(warning, skip_file_prefixes=(_PACKAGE_PREFIX,))
    return off


def withheld_cells(fitted: Any, X: FrameLike | EagerFrame, what: str) -> NDArray | None:
    """The (row, parameter) cells whose variance a reader must withhold, warned once as ``what``.

    A cell is withheld where the row's predictor for that parameter is off support or reaches
    an unfitted direction (``SupportRows``); ``None`` on a fit with neither kind of direction,
    which reads no row.
    """
    support = support_rows(fitted, X)
    if support is None:
        return None
    disclose(support, withheld=what)
    return (support.labels != "") | support.unfitted


__all__ = ["SupportRows", "disclose", "row_labels", "support_rows", "withheld_cells"]
