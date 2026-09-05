"""Instance-local, opt-in timing for stable distributional fit phases."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from types import MappingProxyType

PHASE_NAMES = (
    "frame_normalization",
    "predictor_compilation",
    "layout_penalty_assembly",
    "dense_predictor_matrices",
    "initialization",
    "likelihood_evaluation",
    "curvature_gradient_assembly",
    "coefficient_decomposition_solve",
    "efs_update_backtracking",
    "newton_endgame",
    "terminal_observed_retry_fallback",
    "inference_edf",
    "serialization",
    "fit_total",
)

_clock = time.perf_counter


def _validate_phase(name: str) -> str:
    if name not in PHASE_NAMES:
        raise ValueError(f"unknown distributional fit phase: {name!r}")
    return name


@dataclass(frozen=True)
class FitPhaseSnapshot:
    """Immutable cumulative seconds and observation counts for every phase."""

    seconds: Mapping[str, float]
    counts: Mapping[str, int]

    def __post_init__(self) -> None:
        seconds = dict(self.seconds)
        counts = dict(self.counts)
        if tuple(seconds) != PHASE_NAMES or tuple(counts) != PHASE_NAMES:
            raise ValueError("phase snapshot must contain every phase in canonical order")
        for name in PHASE_NAMES:
            elapsed = float(seconds[name])
            count = counts[name]
            if not math.isfinite(elapsed) or elapsed < 0.0:
                raise ValueError(f"phase {name!r} seconds must be finite and non-negative")
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError(f"phase {name!r} count must be a non-negative integer")
            seconds[name] = elapsed
        object.__setattr__(self, "seconds", MappingProxyType(seconds))
        object.__setattr__(self, "counts", MappingProxyType(counts))

    def as_dict(self) -> dict[str, dict[str, float | int]]:
        """Return a JSON-safe owned representation."""

        return {
            "seconds": dict(self.seconds),
            "counts": dict(self.counts),
        }


class FitPhaseRecorder:
    """Mutable per-fit accumulator; never shared through module-level state."""

    def __init__(self, *, clock: Callable[[], float] | None = None) -> None:
        self._clock = _clock if clock is None else clock
        if not callable(self._clock):
            raise TypeError("clock must be callable")
        self._seconds = {name: 0.0 for name in PHASE_NAMES}
        self._counts = {name: 0 for name in PHASE_NAMES}

    def add(self, name: str, seconds: float) -> None:
        """Add one completed observation to a phase."""

        phase = _validate_phase(name)
        elapsed = float(seconds)
        if not math.isfinite(elapsed) or elapsed < 0.0:
            raise ValueError("phase seconds must be finite and non-negative")
        self._seconds[phase] += elapsed
        self._counts[phase] += 1

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        """Measure one inclusive phase observation using this recorder's clock."""

        phase = _validate_phase(name)
        started = float(self._clock())
        try:
            yield
        finally:
            finished = float(self._clock())
            elapsed = finished - started
            if not math.isfinite(started) or not math.isfinite(finished) or elapsed < 0.0:
                raise RuntimeError("phase clock must return finite monotonic values")
            self.add(phase, elapsed)

    def snapshot(self) -> FitPhaseSnapshot:
        """Return an owned immutable view of the current accumulators."""

        return FitPhaseSnapshot(seconds=self._seconds, counts=self._counts)


@contextmanager
def measure_phase(
    recorder: FitPhaseRecorder | None,
    name: str,
) -> Iterator[None]:
    """Measure *name* when enabled without consulting a clock when disabled."""

    if recorder is None:
        yield
        return
    with recorder.measure(name):
        yield


__all__ = [
    "PHASE_NAMES",
    "FitPhaseRecorder",
    "FitPhaseSnapshot",
    "measure_phase",
]
