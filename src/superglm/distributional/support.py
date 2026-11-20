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
from typing import Any

import numpy as np
from numpy.typing import NDArray

import superglm.distributional.prediction_design as prediction_design
from superglm._frame import EagerFrame, FrameLike, as_eager_frame
from superglm._warn import _PACKAGE_PREFIX
from superglm.diagnostics.estimability import KINDS, disclosure, off_support

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


def support_labels(fitted: Any, X: FrameLike | EagerFrame) -> NDArray | None:
    """Each row's off-support kind per parameter (``off_support``, ``n x P``); ``None`` if none.

    One pass over the rows' designs, a block of rows at a time (``map_design_blocks``), and only
    on a fit with an alias or an unfitted direction.
    """
    bases = _block_bases(fitted)
    if bases is None:
        return None
    layout = fitted.layout
    scale = _jacobi(fitted)

    def labelled(rows: slice, local: dict[str, NDArray]) -> tuple[slice, NDArray]:
        out = np.full((rows.stop - rows.start, len(layout.predictors)), "", dtype=object)
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
        return rows, out

    frame = as_eager_frame(X)
    labels = np.full((len(frame), len(layout.predictors)), "", dtype=object)
    for rows, out in prediction_design.map_design_blocks(
        frame, fitted.compiled_predictors, layout, labelled
    ):
        labels[rows] = out
    return labels


def row_labels(labels: NDArray) -> NDArray:
    """Each row's kind: the first in ``KINDS`` of its parameters' kinds, ``""`` on support."""
    labels = np.asarray(labels, dtype=object)
    rows = np.full(labels.shape[0], "", dtype=object)
    for kind in KINDS:
        rows[(rows == "") & np.any(labels == kind, axis=1)] = kind
    return rows


def disclose(labels: NDArray | None, no_mean: NDArray | None = None) -> NDArray[np.bool_]:
    """Warn once for the rows ``labels`` puts off support; return which rows they are.

    ``no_mean`` marks the rows with no mean at the returned predictor: the off-support ones are
    named in this warning alone (``estimability.disclosure``).
    """
    if labels is None:
        return np.zeros(0 if no_mean is None else np.shape(no_mean)[0], dtype=bool)
    rows = row_labels(labels)
    off = rows != ""
    if np.any(off):
        warnings.warn(disclosure(rows, no_mean), skip_file_prefixes=(_PACKAGE_PREFIX,))
    return off


__all__ = ["disclose", "row_labels", "support_labels"]
