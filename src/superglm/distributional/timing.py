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


def _owned_seconds(values: Mapping[str, float], label: str) -> dict[str, float]:
    seconds = dict(values)
    if tuple(seconds) != PHASE_NAMES:
        raise ValueError("phase snapshot must contain every phase in canonical order")
    for name in PHASE_NAMES:
        elapsed = float(seconds[name])
        if not math.isfinite(elapsed) or elapsed < 0.0:
            raise ValueError(f"phase {name!r} {label} must be finite and non-negative")
        seconds[name] = elapsed
    return seconds


@dataclass(frozen=True)
class FitPhaseSnapshot:
    """Immutable cumulative seconds and observation counts for every phase.

    ``seconds`` is each phase's inclusive time, nested phases included.
    ``exclusive_seconds`` is the clock time a phase spent with no other phase
    open inside it, so the exclusive seconds of different phases never
    overlap.  ``manual_seconds`` holds samples added with
    ``FitPhaseRecorder.add``: they have no interval on the clock and may
    overlap measured time, so they stay out of the exclusive partition.
    """

    seconds: Mapping[str, float]
    counts: Mapping[str, int]
    exclusive_seconds: Mapping[str, float]
    manual_seconds: Mapping[str, float]

    def __post_init__(self) -> None:
        seconds = _owned_seconds(self.seconds, "seconds")
        exclusive = _owned_seconds(self.exclusive_seconds, "exclusive seconds")
        manual = _owned_seconds(self.manual_seconds, "manual seconds")
        counts = dict(self.counts)
        if tuple(counts) != PHASE_NAMES:
            raise ValueError("phase snapshot must contain every phase in canonical order")
        for name in PHASE_NAMES:
            count = counts[name]
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError(f"phase {name!r} count must be a non-negative integer")
        object.__setattr__(self, "seconds", MappingProxyType(seconds))
        object.__setattr__(self, "counts", MappingProxyType(counts))
        object.__setattr__(self, "exclusive_seconds", MappingProxyType(exclusive))
        object.__setattr__(self, "manual_seconds", MappingProxyType(manual))

    def as_dict(self) -> dict[str, dict[str, float | int]]:
        """Return a JSON-safe owned representation."""

        return {
            "seconds": dict(self.seconds),
            "counts": dict(self.counts),
            "exclusive_seconds": dict(self.exclusive_seconds),
            "manual_seconds": dict(self.manual_seconds),
        }


class FitPhaseRecorder:
    """Mutable per-fit accumulator; never shared through module-level state.

    Phases may nest.  Each clock reading closes the interval since the
    previous reading, and that interval is charged to the innermost open
    phase alone, so a nested phase's time is excluded from its parent's
    exclusive seconds.  This is the self time of a call-graph profiler
    (gprof's self seconds; ``tottime`` in Python's ``profile`` module); the
    inclusive ``seconds`` correspond to ``cumtime``.
    """

    def __init__(self, *, clock: Callable[[], float] | None = None) -> None:
        self._clock = _clock if clock is None else clock
        if not callable(self._clock):
            raise TypeError("clock must be callable")
        self._seconds = {name: 0.0 for name in PHASE_NAMES}
        self._exclusive = {name: 0.0 for name in PHASE_NAMES}
        self._manual = {name: 0.0 for name in PHASE_NAMES}
        self._counts = {name: 0 for name in PHASE_NAMES}
        self._open: list[str] = []
        self._first_reading: float | None = None
        self._last_reading = 0.0

    def add(self, name: str, seconds: float) -> None:
        """Add one completed observation to a phase.

        A manual sample has no interval on this recorder's clock, and the
        clock time it describes may already be charged to an open phase.  It
        therefore adds to the phase's inclusive ``seconds`` and to its
        ``manual_seconds``, never to the exclusive partition.
        """

        phase = _validate_phase(name)
        elapsed = float(seconds)
        if not math.isfinite(elapsed) or elapsed < 0.0:
            raise ValueError("phase seconds must be finite and non-negative")
        self._seconds[phase] += elapsed
        self._manual[phase] += elapsed
        self._counts[phase] += 1

    def _read_clock(self) -> float:
        """Read the clock and charge the interval since the last reading."""

        reading = float(self._clock())
        self._advance(reading)
        return reading

    def _advance(self, reading: float) -> None:
        """Charge the interval up to ``reading`` to the innermost open phase."""

        if not math.isfinite(reading):
            raise RuntimeError("phase clock must return finite monotonic values")
        if self._open:
            interval = reading - self._last_reading
            # Two finite readings can still be infinitely far apart.
            if not math.isfinite(interval) or interval < 0.0:
                raise RuntimeError("phase clock must return finite monotonic values")
            self._exclusive[self._open[-1]] += interval
        if self._first_reading is None:
            self._first_reading = reading
        self._last_reading = reading

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        """Measure one phase observation using this recorder's clock."""

        phase = _validate_phase(name)
        started = self._read_clock()
        self._open.append(phase)
        try:
            yield
        finally:
            try:
                finished = self._read_clock()
            finally:
                # Context managers close in reverse order, so this observation
                # is the innermost open one.
                self._open.pop()
            # Python's profiler counts cumulative time only at the outermost
            # call of a recursive function; a phase nested in itself does too,
            # so its inclusive seconds cannot exceed the time it was open.
            elapsed = finished - started
            if not math.isfinite(elapsed):
                raise RuntimeError("phase clock must return finite monotonic values")
            if phase not in self._open:
                self._seconds[phase] += elapsed
            self._counts[phase] += 1

    def _absorb(self, timed: FitPhaseRecorder) -> None:
        """Fold in a finished recorder that read this clock, as one nested interval.

        A fit is timed on a recorder of its own, so its profile holds the fit
        alone whatever this recorder has open.  Folding it in charges the time
        before the fit's first reading to the phase open here, adds the fit's
        observations, and leaves the time after its last reading for this
        recorder's next reading.
        """

        if timed._first_reading is not None:
            self._advance(timed._first_reading)
            self._last_reading = timed._last_reading
        for name in PHASE_NAMES:
            if name not in self._open:
                self._seconds[name] += timed._seconds[name]
            self._exclusive[name] += timed._exclusive[name]
            self._manual[name] += timed._manual[name]
            self._counts[name] += timed._counts[name]

    def snapshot(self) -> FitPhaseSnapshot:
        """Return an owned immutable view of the current accumulators."""

        return FitPhaseSnapshot(
            seconds=self._seconds,
            counts=self._counts,
            exclusive_seconds=self._exclusive,
            manual_seconds=self._manual,
        )


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
