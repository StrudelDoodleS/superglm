"""Process measurements for benchmark receipts on Linux, macOS and Windows."""

from __future__ import annotations

import os
import sys
from typing import NamedTuple


class PeakRSS(NamedTuple):
    bytes: int
    source: str
    ru_maxrss_unit: str | None


def peak_rss() -> PeakRSS:
    """Read the whole-process high-water mark, outside the measured fit region."""
    if sys.platform == "win32":
        import psutil

        # peak_wset is bytes, unlike current RSS (wset). No sampling thread.
        # https://psutil.io/7.2/#psutil.Process.memory_info
        return PeakRSS(int(psutil.Process().memory_info().peak_wset), "psutil.peak_wset", None)

    import resource

    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return PeakRSS(raw, "resource.ru_maxrss", "bytes")
    return PeakRSS(raw * 1024, "resource.ru_maxrss", "kib")


def load_average() -> tuple[float, float, float] | None:
    """Unknown load stays unknown; psutil's Windows emulation needs a thread."""
    getloadavg = getattr(os, "getloadavg", None)
    return None if getloadavg is None else getloadavg()


def cpu_affinity() -> list[int] | None:
    getaffinity = getattr(os, "sched_getaffinity", None)
    return None if getaffinity is None else sorted(getaffinity(0))


def available_cpu_count() -> int:
    affinity = cpu_affinity()
    if affinity is not None:
        return len(affinity)
    count = getattr(os, "process_cpu_count", os.cpu_count)()
    return count or 1
