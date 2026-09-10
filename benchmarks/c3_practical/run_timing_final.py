"""Prepare one new six-worker timing window; quietness remains subject to review."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HARNESS = ROOT / "benchmarks/c3_c1_complete_fit.py"
FIXTURES = ROOT / "benchmarks/_c3_c1_fixtures.py"
DATA = ROOT.parent / "c3-c1-completion/data"
SOURCES = {
    "baseline": ROOT.parent / "c3-c1-baseline",
    "candidate": ROOT.parent / "c3-pragmatic-final",
}
EXPECTED_SHA = {
    "baseline": "8962c4520cad948aa20c480a238b7bb1e276e9cd",
    "candidate": "5f994c8f6ac0501606594e2f36bfc0cd24050ec1",
}
ORDER = ("baseline", "candidate", "candidate", "baseline", "baseline", "candidate")
THREAD_VARIABLES = (
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
)
ACTIVITY_LIMITATION = (
    "Endpoint CPU samples miss processes born and exited entirely inside the interval, and cannot "
    "measure peak contention or memory-bandwidth competition. PID reuse is not detectable in the "
    "worker's existing endpoint snapshots. Categories are command-name substring heuristics; "
    "command arguments are examined transiently and never persisted. CPU interval clocks only "
    "normalize CPU ticks; fit times come exclusively from each worker's perf_counter receipt."
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def snapshot():
    processes = []
    for entry in Path("/proc").glob("[0-9]*"):
        try:
            name = (entry / "comm").read_text().strip()
            command = (entry / "cmdline").read_bytes().lower()
            categories = [
                label for label in ("pylance", "pytest", "node", "python")
                if label.encode() in command
            ]
            stat = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            processes.append({
                "pid": int(entry.name), "name": name, "categories": categories,
                "cpu_ticks": int(stat[11]) + int(stat[12]),
                "rss_bytes": int(stat[21]) * os.sysconf("SC_PAGE_SIZE"),
            })
        except (OSError, ValueError, IndexError):
            continue
    return {
        "time_ns": time.time_ns(), "load_average": list(os.getloadavg()),
        "cpu_count": os.cpu_count(), "affinity": sorted(os.sched_getaffinity(0)),
        "process_activity": processes,
    }


def primary_category(process):
    categories = set(process["categories"])
    if "pytest" in categories:
        return "pytest"
    if categories & {"node", "pylance"}:
        return "node_or_pylance"
    if "python" in categories:
        return "other_python"
    return "other_system_and_tools"


def activity_audit(start, end, worker_pid=None):
    duration = (end["time_ns"] - start["time_ns"]) / 1e9
    if duration <= 0:
        raise ValueError("activity snapshots must have increasing clocks")
    hz = os.sysconf("SC_CLK_TCK")
    before = {process["pid"]: process for process in start["process_activity"]}
    after = {process["pid"]: process for process in end["process_activity"]}
    activity, invalid = [], []
    for pid in before.keys() & after.keys():
        ticks = after[pid]["cpu_ticks"] - before[pid]["cpu_ticks"]
        if ticks < 0:
            invalid.append(pid)
            continue
        if ticks or pid == worker_pid:
            activity.append({
                **after[pid], "delta_ticks": ticks, "cpu_seconds": ticks / hz,
                "average_cores": ticks / hz / duration,
            })
    activity.sort(key=lambda process: process["delta_ticks"], reverse=True)
    external = [process for process in activity if process["pid"] != worker_pid]
    categories = defaultdict(float)
    for process in external:
        categories[primary_category(process)] += process["average_cores"]
    return {
        "activity_interval_seconds": duration, "clock_ticks_per_second": hz,
        "load_start": start["load_average"], "load_end": end["load_average"],
        "worker_pid": worker_pid,
        "worker": next((process for process in activity if process["pid"] == worker_pid), None),
        "external_average_cores": sum(process["average_cores"] for process in external),
        "external_average_cores_by_category": dict(categories),
        "external_process_deltas": external,
        "started": [after[pid] for pid in sorted(after.keys() - before.keys())],
        "exited": [before[pid] for pid in sorted(before.keys() - after.keys())],
        "negative_tick_delta_pids": invalid,
        "limitations": ACTIVITY_LIMITATION,
    }


def source_sha(path):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--quiet-seconds", type=float, default=10.0)
    parser.add_argument("--max-preflight-external-cores", type=float, default=0.1)
    args = parser.parse_args()
    if not 1 <= args.quiet_seconds <= 60 or args.max_preflight_external_cores < 0:
        parser.error("quiet-seconds must be 1..60 and the preflight CPU threshold nonnegative")
    for label, source in SOURCES.items():
        if source_sha(source) != EXPECTED_SHA[label]:
            raise RuntimeError(f"{label} source is not its pinned revision: {source}")
    for filename in ("freMTPL2freq.parquet", "freMTPL2sev.parquet"):
        if not (DATA / filename).is_file():
            raise FileNotFoundError(DATA / filename)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    out = (args.out or Path(__file__).parent / f"final-timing-{timestamp}").resolve()
    out.mkdir(parents=True, exist_ok=False)
    controller = {
        "schema": 1, "audit_status": "pending_current_window_review", "root": str(ROOT),
        "sources": {label: str(source) for label, source in SOURCES.items()},
        "expected_source_sha": EXPECTED_SHA, "order": list(ORDER), "data": str(DATA.resolve()),
        "data_sha256": {filename: digest(DATA / filename) for filename in (
            "freMTPL2freq.parquet", "freMTPL2sev.parquet"
        )},
        "executable": sys.executable, "controller_pid": os.getpid(),
        "controller_sha256": digest(__file__), "harness_sha256": digest(HARNESS),
        "fixtures_sha256": digest(FIXTURES), "runs": [],
        "cache_policy": "Caches retained; fresh processes, no warmup, no cache flushing.",
        "preflight_external_core_threshold": args.max_preflight_external_cores,
    }
    write_json(out / "controller.json", controller)
    before = snapshot()
    time.sleep(args.quiet_seconds)
    after = snapshot()
    preflight = {"before": before, "after": after, "activity": activity_audit(before, after)}
    write_json(out / "quiet-window.json", preflight)
    if preflight["activity"]["external_average_cores"] > args.max_preflight_external_cores:
        controller["preflight_status"] = "obvious_contention_refused_before_workers"
        write_json(out / "controller.json", controller)
        raise SystemExit(f"Preflight CPU activity exceeds the configured threshold; evidence: {out}")
    controller["preflight_status"] = "screen_passed_not_quietness_approval"
    env = dict(os.environ)
    env.pop("PYTEST_ADDOPTS", None)
    for name in THREAD_VARIABLES:
        env[name] = "1"
    profile = "current sampled serial window; final quietness review pending"
    for index, label in enumerate(ORDER):
        source = SOURCES[label]
        if source_sha(source) != EXPECTED_SHA[label] or digest(HARNESS) != controller["harness_sha256"]:
            raise RuntimeError("source or harness changed before a worker")
        run_directory = out / f"{index}-{label}"
        run_directory.mkdir(exist_ok=False)
        output = run_directory / f"{label}-0.json"
        command = [
            sys.executable, str(HARNESS), "--source", f"{label}={source}",
            "--worker-source", str(source), "--worker-output", str(output),
            "--out", str(run_directory), "--data", str(DATA.resolve()),
            "--fixture", "severity-gaussian", "--n", "0", "--replicate", "20", "--knots", "12",
            "--outer", "efs+newton", "--no-practical-reml", "--initial-lambda", "0.1",
            "--max-reml-iter", "100", "--max-inner-iter", "100", "--inner-tol", "1e-7",
            "--reml-tol", "1e-6", "--max-lambda", "1e10", "--holdout-every", "10",
            "--no-warmup", "--threads", "1", "--repeat", "1", "--n-bins", "256",
            "--measure-time", "--quiet-profile", profile,
        ]
        if label == "candidate":
            command.append("--discrete")
        env["PYTHONPATH"] = str(source / "src")
        record = {"index": index, "label": label, "json": str(output), "returncode": None}
        with output.with_suffix(".log").open("w") as log:
            worker = subprocess.Popen(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT)
            record["worker_pid"] = worker.pid
            controller["runs"].append(record)
            write_json(out / "controller.json", controller)
            print(json.dumps({"starting": index, "arm": label, "worker_pid": worker.pid}), flush=True)
            record["returncode"] = worker.wait()
        record["json_sha256"] = digest(output) if output.exists() else None
        record["log_sha256"] = digest(output.with_suffix(".log"))
        record["npz_sha256"] = digest(output.with_suffix(".npz")) if output.with_suffix(".npz").exists() else None
        write_json(run_directory / "manifest.json", {"schema": 1, "runs": [record]})
        record["manifest_sha256"] = digest(run_directory / "manifest.json")
        write_json(out / "controller.json", controller)
        print(json.dumps({"finished": index, "arm": label, "returncode": record["returncode"]}), flush=True)
        if record["returncode"]:
            raise SystemExit(record["returncode"])
    controller["harness_stable"] = digest(HARNESS) == controller["harness_sha256"]
    controller["fixtures_stable"] = digest(FIXTURES) == controller["fixtures_sha256"]
    write_json(out / "controller.json", controller)
    print(f"Raw timing window ready for neutral audit: {out}", flush=True)


if __name__ == "__main__":
    main()
