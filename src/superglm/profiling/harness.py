"""Reusable profiling harness helpers for benchmark investigations."""

from __future__ import annotations

import csv
import gc
import io
import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SystemSample:
    """One process/system telemetry sample."""

    t_rel_s: float
    rss_bytes: int
    vms_bytes: int
    uss_bytes: int
    child_rss_bytes: int
    process_cpu_percent: float
    load_avg_1m: float
    load_avg_5m: float
    load_avg_15m: float
    thread_count: int
    voluntary_ctx_switches: int
    involuntary_ctx_switches: int
    read_bytes: int
    write_bytes: int
    available_memory_bytes: int
    gc_gen0: int
    gc_gen1: int
    gc_gen2: int
    cpu_percent_per_core: tuple[float, ...]


def flatten_system_sample(sample: SystemSample) -> dict[str, float | int]:
    """Flatten a telemetry sample into CSV-friendly scalar columns."""
    row: dict[str, float | int] = {
        "t_rel_s": sample.t_rel_s,
        "rss_bytes": sample.rss_bytes,
        "vms_bytes": sample.vms_bytes,
        "uss_bytes": sample.uss_bytes,
        "child_rss_bytes": sample.child_rss_bytes,
        "process_cpu_percent": sample.process_cpu_percent,
        "load_avg_1m": sample.load_avg_1m,
        "load_avg_5m": sample.load_avg_5m,
        "load_avg_15m": sample.load_avg_15m,
        "thread_count": sample.thread_count,
        "voluntary_ctx_switches": sample.voluntary_ctx_switches,
        "involuntary_ctx_switches": sample.involuntary_ctx_switches,
        "read_bytes": sample.read_bytes,
        "write_bytes": sample.write_bytes,
        "available_memory_bytes": sample.available_memory_bytes,
        "gc_gen0": sample.gc_gen0,
        "gc_gen1": sample.gc_gen1,
        "gc_gen2": sample.gc_gen2,
    }
    for i, value in enumerate(sample.cpu_percent_per_core):
        row[f"cpu_core_{i}_percent"] = value
    return row


def summarize_system_samples(samples: list[SystemSample]) -> dict[str, float | int]:
    """Summarize a time series of process/system telemetry."""
    if not samples:
        return {"n_samples": 0}

    summary: dict[str, float | int] = {
        "n_samples": len(samples),
        "wall_time_s": samples[-1].t_rel_s - samples[0].t_rel_s,
        "rss_peak_bytes": max(s.rss_bytes for s in samples),
        "rss_delta_bytes": samples[-1].rss_bytes - samples[0].rss_bytes,
        "uss_peak_bytes": max(s.uss_bytes for s in samples),
        "child_rss_peak_bytes": max(s.child_rss_bytes for s in samples),
        "process_cpu_mean_percent": sum(s.process_cpu_percent for s in samples) / len(samples),
        "process_cpu_peak_percent": max(s.process_cpu_percent for s in samples),
        "load_avg_1m_peak": max(s.load_avg_1m for s in samples),
        "thread_count_peak": max(s.thread_count for s in samples),
        "read_bytes_delta": samples[-1].read_bytes - samples[0].read_bytes,
        "write_bytes_delta": samples[-1].write_bytes - samples[0].write_bytes,
    }
    n_cores = max(len(s.cpu_percent_per_core) for s in samples)
    for core in range(n_cores):
        core_vals = [
            s.cpu_percent_per_core[core] for s in samples if core < len(s.cpu_percent_per_core)
        ]
        summary[f"cpu_core_{core}_peak_percent"] = max(core_vals)
        summary[f"cpu_core_{core}_mean_percent"] = sum(core_vals) / len(core_vals)
    return summary


def write_system_samples_csv(samples: list[SystemSample], path: str | Path) -> None:
    """Write the sampled process/system telemetry to CSV."""
    rows = [flatten_system_sample(sample) for sample in samples]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_tracemalloc_report(
    path: str | Path,
    *,
    start_snapshot,
    end_snapshot,
    peak_bytes: int,
    limit: int = 40,
) -> None:
    """Write a tracemalloc summary and top allocation diffs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"peak_traced_bytes={peak_bytes}", ""]
    if start_snapshot is None or end_snapshot is None:
        lines.append("No tracemalloc snapshots captured.")
        path.write_text("\n".join(lines))
        return
    lines.append("Top allocation deltas (lineno):")
    for stat in end_snapshot.compare_to(start_snapshot, "lineno")[:limit]:
        lines.append(str(stat))
    lines.append("")
    lines.append("Top allocation totals (lineno):")
    for stat in end_snapshot.statistics("lineno")[:limit]:
        lines.append(str(stat))
    path.write_text("\n".join(lines))


def write_pstats_summary(profile, path: str | Path, *, limit: int = 80) -> None:
    """Write human-readable cProfile stats sorted by cumulative time and total time."""
    import pstats

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = io.StringIO()
    stats = pstats.Stats(profile, stream=stream)
    stats.strip_dirs().sort_stats("cumtime").print_stats(limit)
    stream.write("\n=== sort: tottime ===\n")
    stats = pstats.Stats(profile, stream=stream)
    stats.strip_dirs().sort_stats("tottime").print_stats(limit)
    path.write_text(stream.getvalue())


def write_cprofile_stats(profile, path: str | Path, *, limit: int = 80) -> None:
    """Backward-compatible alias for writing human-readable cProfile stats."""
    write_pstats_summary(profile, path, limit=limit)


def capture_system_sample(proc, *, t_rel_s: float) -> SystemSample:
    """Capture one process/system telemetry sample via psutil."""
    import psutil

    with proc.oneshot():
        mem = proc.memory_info()
        try:
            full_mem = proc.memory_full_info()
            uss = int(getattr(full_mem, "uss", mem.rss))
        except (psutil.AccessDenied, AttributeError, OSError):
            uss = int(mem.rss)
        cpu_percent = float(proc.cpu_percent(interval=None))
        thread_count = int(proc.num_threads())
        ctx = proc.num_ctx_switches()
        io_counters = proc.io_counters()

    child_rss = 0
    for child in proc.children(recursive=True):
        try:
            child_rss += int(child.memory_info().rss)
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            continue

    try:
        load1, load5, load15 = os.getloadavg()
    except OSError:
        load1 = load5 = load15 = 0.0

    vm = psutil.virtual_memory()
    gc0, gc1, gc2 = gc.get_count()

    return SystemSample(
        t_rel_s=t_rel_s,
        rss_bytes=int(mem.rss),
        vms_bytes=int(mem.vms),
        uss_bytes=uss,
        child_rss_bytes=child_rss,
        process_cpu_percent=cpu_percent,
        load_avg_1m=float(load1),
        load_avg_5m=float(load5),
        load_avg_15m=float(load15),
        thread_count=thread_count,
        voluntary_ctx_switches=int(ctx.voluntary),
        involuntary_ctx_switches=int(ctx.involuntary),
        read_bytes=int(getattr(io_counters, "read_bytes", 0)),
        write_bytes=int(getattr(io_counters, "write_bytes", 0)),
        available_memory_bytes=int(vm.available),
        gc_gen0=int(gc0),
        gc_gen1=int(gc1),
        gc_gen2=int(gc2),
        cpu_percent_per_core=tuple(
            float(v) for v in psutil.cpu_percent(interval=None, percpu=True)
        ),
    )


class SystemSampler:
    """Background psutil sampler for process and per-core telemetry."""

    def __init__(self, *, pid: int | None = None, interval_s: float = 0.5):
        self.pid = pid or os.getpid()
        self.interval_s = interval_s
        self.samples: list[SystemSample] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0 = 0.0
        self._error: str | None = None

    @property
    def error(self) -> str | None:
        """Any sampler-side failure string."""
        return self._error

    def start(self) -> None:
        """Start the background sampler."""
        import psutil

        if self._thread is not None:
            raise RuntimeError("SystemSampler already started")

        proc = psutil.Process(self.pid)
        proc.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None, percpu=True)
        self._t0 = time.perf_counter()

        def run() -> None:
            while not self._stop.is_set():
                try:
                    sample = capture_system_sample(proc, t_rel_s=time.perf_counter() - self._t0)
                    self.samples.append(sample)
                except Exception as exc:  # pragma: no cover - defensive
                    self._error = repr(exc)
                    return
                self._stop.wait(self.interval_s)

        self._thread = threading.Thread(target=run, name="system-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the sampler and join the background thread."""
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=max(2.0, self.interval_s * 4))
        self._thread = None


def dump_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write one JSON artifact with stable formatting."""
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def dataclass_payload(obj: Any) -> dict[str, Any]:
    """Convert a dataclass-like object into a JSON-serializable mapping."""
    try:
        return asdict(obj)
    except TypeError:
        if hasattr(obj, "__dict__"):
            return dict(obj.__dict__)
        raise
