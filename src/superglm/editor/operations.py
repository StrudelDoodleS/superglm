"""Pure array operations used by the editor session."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def anchored_isotonic_values(
    y_all: NDArray,
    idx: NDArray[np.intp],
    weights_all: NDArray | None,
    direction: str,
) -> NDArray:
    return _apply_runs(idx, lambda run: _anchored_isotonic_run(y_all, run, weights_all, direction))


def anchored_smooth_values(
    y_all: NDArray,
    idx: NDArray[np.intp],
    strength: float,
) -> NDArray:
    return _apply_runs(idx, lambda run: _anchored_smooth_run(y_all, run, strength))


def monotone_clamp_values(
    y_all: NDArray,
    idx: NDArray[np.intp],
    direction: str,
) -> NDArray:
    return _apply_runs(idx, lambda run: _monotone_clamp_run(y_all, run, direction))


def _apply_runs(idx: NDArray[np.intp], transform) -> NDArray:
    # Selection can be disjoint; each contiguous run is edited independently so
    # anchors and monotonic constraints do not bleed across gaps.
    idx = np.asarray(idx, dtype=np.intp)
    after = np.empty(idx.size, dtype=np.float64)
    offset = 0
    for run in _contiguous_runs(idx):
        values = np.asarray(transform(run), dtype=np.float64)
        after[offset : offset + run.size] = values
        offset += run.size
    return after


def _isotonic_values(y: NDArray, weights: NDArray | None, direction: str) -> NDArray:
    from sklearn.isotonic import IsotonicRegression

    increasing = direction == "increasing"
    x = np.arange(len(y), dtype=np.float64)
    ir = IsotonicRegression(increasing=increasing, out_of_bounds="clip")
    return np.asarray(ir.fit_transform(x, y, sample_weight=weights), dtype=np.float64)


def _monotone_clamp_run(y_all: NDArray, run: NDArray[np.intp], direction: str) -> NDArray:
    left = int(run[0]) - 1
    current = float(y_all[left] if left >= 0 else y_all[run[0]])
    out = np.empty(run.size, dtype=np.float64)
    for i, idx in enumerate(run):
        value = float(y_all[idx])
        current = max(current, value) if direction == "increasing" else min(current, value)
        out[i] = current
    return out


def _anchored_smooth_run(y_all: NDArray, run: NDArray[np.intp], strength: float) -> NDArray:
    left = int(run[0]) - 1
    right = int(run[-1]) + 1
    has_left = left >= 0
    has_right = right < y_all.size

    values = []
    if has_left:
        values.append(float(y_all[left]))
    values.extend(float(v) for v in y_all[run])
    if has_right:
        values.append(float(y_all[right]))

    # Smooth the selected run plus neighbor anchors, then restore the anchors
    # and taper the edge movement. This gives aggressive smoothing without a
    # visible discontinuity where the selected region meets the untouched curve.
    extended = np.asarray(values, dtype=np.float64)
    if extended.size < 3:
        smoothed = extended.copy()
    else:
        local_smooth = _wide_gaussian_smooth(extended)
        smoothed = (1.0 - strength) * extended + strength * local_smooth
    if has_left:
        smoothed[0] = float(y_all[left])
    if has_right:
        smoothed[-1] = float(y_all[right])
    start = 1 if has_left else 0
    run_after = smoothed[start : start + run.size].copy()
    return _taper_smooth_edges(
        y_all[run],
        run_after,
        float(y_all[left]) if has_left else None,
        float(y_all[right]) if has_right else None,
    )


def _wide_gaussian_smooth(values: NDArray) -> NDArray:
    values = np.asarray(values, dtype=np.float64)
    if values.size < 3:
        return values.copy()
    sigma = min(max(values.size / 24.0, 1.75), 14.0)
    radius = int(max(2, np.ceil(3.0 * sigma)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= np.sum(kernel)
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _taper_smooth_edges(
    original: NDArray,
    smoothed: NDArray,
    left_anchor: float | None,
    right_anchor: float | None,
) -> NDArray:
    original = np.asarray(original, dtype=np.float64)
    smoothed = np.asarray(smoothed, dtype=np.float64)
    if smoothed.size == 0 or (left_anchor is None and right_anchor is None):
        return smoothed.copy()

    factors = np.ones(smoothed.size, dtype=np.float64)
    edge = min(max(3, smoothed.size // 10), max(smoothed.size // 2, 1))
    edge_floor = 0.35
    left_ramp = np.array([edge_floor]) if edge == 1 else np.linspace(edge_floor, 1.0, edge)
    right_ramp = np.array([edge_floor]) if edge == 1 else np.linspace(1.0, edge_floor, edge)
    if left_anchor is not None:
        factors[:edge] = np.minimum(factors[:edge], left_ramp)
    if right_anchor is not None:
        factors[-edge:] = np.minimum(factors[-edge:], right_ramp)

    out = original + factors * (smoothed - original)
    if left_anchor is not None:
        out[0] = _clamp_boundary_jump(out[0], original[0], left_anchor)
    if right_anchor is not None:
        out[-1] = _clamp_boundary_jump(out[-1], original[-1], right_anchor)
    return out


def _clamp_boundary_jump(value: float, original: float, anchor: float) -> float:
    max_jump = abs(original - anchor)
    return float(np.clip(value, anchor - max_jump, anchor + max_jump))


def _anchored_isotonic_run(
    y_all: NDArray,
    run: NDArray[np.intp],
    weights_all: NDArray | None,
    direction: str,
) -> NDArray:
    left = int(run[0]) - 1
    right = int(run[-1]) + 1
    has_left = left >= 0
    has_right = right < y_all.size
    run_weights = None if weights_all is None else np.asarray(weights_all[run], dtype=np.float64)
    anchor_weight = _anchor_weight(run_weights, run.size)

    values = []
    weights = []
    if has_left:
        values.append(float(y_all[left]))
        weights.append(anchor_weight)
    values.extend(float(v) for v in y_all[run])
    weights.extend([1.0] * run.size if run_weights is None else [float(v) for v in run_weights])
    if has_right:
        values.append(float(y_all[right]))
        weights.append(anchor_weight)

    fitted = _isotonic_values(
        np.asarray(values, dtype=np.float64),
        np.asarray(weights, dtype=np.float64),
        direction,
    )
    start = 1 if has_left else 0
    run_after = fitted[start : start + run.size].copy()
    return _project_to_anchors(
        run_after,
        float(y_all[left]) if has_left else None,
        float(y_all[right]) if has_right else None,
        direction,
    )


def _contiguous_runs(idx: NDArray[np.intp]) -> list[NDArray[np.intp]]:
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) != 1) + 1
    return [run.astype(np.intp, copy=False) for run in np.split(idx, breaks)]


def _anchor_weight(weights: NDArray | None, size: int) -> float:
    # The anchors are synthetic observations. They get very high weight so the
    # selected run respects neighboring untouched values without hard-pinning
    # every interior point to those neighbors.
    if weights is None:
        return float(max(size, 1) * 1_000_000)
    finite_positive = weights[np.isfinite(weights) & (weights > 0)]
    total = float(np.sum(finite_positive)) if finite_positive.size else float(max(size, 1))
    return max(total, 1.0) * 1_000_000


def _project_to_anchors(
    values: NDArray,
    left_value: float | None,
    right_value: float | None,
    direction: str,
) -> NDArray:
    # IsotonicRegression handles monotonicity but not external boundary
    # constraints. Projecting afterward keeps the fitted run compatible with
    # the nearest untouched values when those anchors are directionally valid.
    values = np.asarray(values, dtype=np.float64).copy()
    if values.size == 0:
        return values

    if direction == "increasing":
        lower = -np.inf if left_value is None else left_value
        upper = np.inf if right_value is None else right_value
        if lower <= upper:
            values = np.clip(values, lower, upper)
        values = np.maximum.accumulate(values)
        if right_value is not None and lower <= upper:
            values = np.minimum(values, right_value)
        return values

    upper = np.inf if left_value is None else left_value
    lower = -np.inf if right_value is None else right_value
    if lower <= upper:
        values = np.clip(values, lower, upper)
    values = np.minimum.accumulate(values)
    if right_value is not None and lower <= upper:
        values = np.maximum(values, right_value)
    return values
