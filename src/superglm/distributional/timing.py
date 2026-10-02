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


def _accumulate(totals: dict[str, float], name: str, seconds: float) -> None:
    """Add ``seconds`` to ``totals[name]``, refusing a total that overflows.

    Each addend is finite, but finite addends can still sum past the largest
    float; refusing here keeps a bad clock from surfacing later as an invalid
    snapshot after the fit has finished.
    """
    total = totals[name] + seconds
    if not math.isfinite(total):
        raise RuntimeError("phase timing totals must stay finite")
    totals[name] = total


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


class _FitWindow:
    """One fit's stretch of a recorder: its accumulators at the fit's first reading and after it.

    The fit opens ``fit_total`` with its first reading, which first charges
    the interval before it to whatever phase was already open; the window
    starts after that charge, so a caller's earlier time stays out.  It stops
    without reading the clock, so the interval after the fit's last reading
    stays out as well.  A caller's own measurements made while the fit runs
    sit on the same stack in clock order and fall inside the window once.
    """

    def __init__(self) -> None:
        self.start: tuple[dict[str, float], ...] | None = None
        self.stop: tuple[dict[str, float], ...] | None = None
        self.start_reading = 0.0
        self.stop_reading = 0.0

    def snapshot(self) -> FitPhaseSnapshot:
        """The window's own timings; ``fit_total`` is the fit's span, whatever enclosed it."""

        zero = {name: 0.0 for name in PHASE_NAMES}
        start = self.start if self.start is not None else (zero, zero, zero, zero)
        stop = self.stop if self.start is not None else start
        seconds, exclusive, manual, counts = (
            {name: after[name] - before[name] for name in PHASE_NAMES}
            for before, after in zip(start, stop, strict=True)
        )
        seconds["fit_total"] = self.stop_reading - self.start_reading
        return FitPhaseSnapshot(
            seconds=seconds,
            counts={name: int(counts[name]) for name in PHASE_NAMES},
            exclusive_seconds=exclusive,
            manual_seconds=manual,
        )


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
        self._last_reading = 0.0
        self._starting: list[_FitWindow] = []

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
        _accumulate(self._seconds, phase, elapsed)
        _accumulate(self._manual, phase, elapsed)
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
            _accumulate(self._exclusive, self._open[-1], interval)
        self._last_reading = reading
        if self._starting:
            state = self._state()
            for window in self._starting:
                window.start, window.start_reading = state, reading
            self._starting.clear()

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
                _accumulate(self._seconds, phase, elapsed)
            self._counts[phase] += 1

    def _state(self) -> tuple[dict[str, float], ...]:
        return (
            dict(self._seconds),
            dict(self._exclusive),
            dict(self._manual),
            {name: float(count) for name, count in self._counts.items()},
        )

    @contextmanager
    def _fit_window(self) -> Iterator[_FitWindow]:
        """Delimit one fit on this recorder; see ``_FitWindow``.  Never raises on exit."""

        window = _FitWindow()
        self._starting.append(window)
        try:
            yield window
        finally:
            if window in self._starting:
                self._starting.remove(window)
            else:
                window.stop, window.stop_reading = self._state(), self._last_reading

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
