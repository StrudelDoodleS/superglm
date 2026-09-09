"""Public serial complete-fit controller; raw worker receipts own timing evidence.

Example (use the same Python environment for both source roots)::

    python benchmarks/discrete_performance.py --baseline /path/to/base \
        --candidate /path/to/candidate --out /tmp/discrete-run \
        --fixture gaussian --n 10000 --knots 12 --n-bins 256 \
        --quiet-profile 'dedicated local session; other compute stopped'

Use a different output directory with --instrument for dispatch profiling.
Profiling runs never request or summarize fit timings. This Linux controller
uses its own monotonic clock and /proc, not shell/tool elapsed-time proxies.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import c3_c1_complete_fit as harness

THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
BACKGROUND_LIMIT = 0.25
PREFLIGHT_SECONDS = 2.0
CPU_CAVEAT = (
    "Endpoint accounting is a lower bound: CPU used after a process's last "
    "observation before exit, and by processes born and exited between snapshots, "
    "is unavailable. Newly observed processes contribute all observed lifetime "
    "ticks; PID reuse is distinguished by start ticks. Descendants that reparent "
    "between snapshots may not be identifiable. This is a screening check, "
    "not proof that the machine stayed quiet throughout a fit."
)


def write_json(path, value, *, exclusive=False):
    """Keep all endpoint/process records uncompressed and preserve old receipts."""
    path = Path(path)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if exclusive:
        with path.open("x") as stream:
            stream.write(payload)
    else:
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("x") as stream:
            stream.write(payload)
        temporary.replace(path)


def process_snapshot():
    """Read each identity and its own user+system CPU ticks (not child ticks)."""
    started = time.monotonic_ns()
    processes = {}
    unreadable = []
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        try:
            raw = (directory / "stat").read_text()
            left, right = raw.index("("), raw.rindex(")")
            fields = raw[right + 2 :].split()
            pid = int(directory.name)
            start_ticks = int(fields[19])
            processes[f"{pid}:{start_ticks}"] = {
                "pid": pid,
                "start_ticks": start_ticks,
                "ppid": int(fields[1]),
                "comm": raw[left + 1 : right],
                "state": fields[0],
                "user_ticks": int(fields[11]),
                "system_ticks": int(fields[12]),
                "cpu_ticks": int(fields[11]) + int(fields[12]),
            }
        except (OSError, ValueError, IndexError):
            unreadable.append(int(directory.name))
    cpu = Path("/proc/stat").read_text().splitlines()[0].split()
    return {
        "monotonic_ns": started,
        "snapshot_finished_monotonic_ns": time.monotonic_ns(),
        "clock_ticks_per_second": os.sysconf("SC_CLK_TCK"),
        # guest times are already included in user/nice; omit the extra fields.
        "system_total_cpu_ticks": sum(map(int, cpu[1:9])),
        "observed_process_cpu_ticks": sum(p["cpu_ticks"] for p in processes.values()),
        "affinity": sorted(os.sched_getaffinity(0)),
        "load_average": list(os.getloadavg()),
        "unreadable_or_exited_pids": unreadable,
        "processes": processes,
    }


def descendant_identities(snapshot, roots):
    pids = set(roots)
    processes = snapshot["processes"]
    while True:
        expanded = pids | {p["pid"] for p in processes.values() if p["ppid"] in pids}
        if expanded == pids:
            break
        pids = expanded
    return {key for key, process in processes.items() if process["pid"] in pids}


def background_audit(before, after, roots):
    elapsed = (after["monotonic_ns"] - before["monotonic_ns"]) / 1e9
    excluded = descendant_identities(before, roots) | descendant_identities(after, roots)
    first, last = before["processes"], after["processes"]
    active = []
    for identity, process in last.items():
        if identity in excluded:
            continue
        old_ticks = first.get(identity, {}).get("cpu_ticks", 0)
        delta = max(0, process["cpu_ticks"] - old_ticks)
        if delta:
            active.append(
                {
                    **process,
                    "identity": identity,
                    "delta_ticks": delta,
                    "newly_observed": identity not in first,
                }
            )
    ticks = sum(p["delta_ticks"] for p in active)
    cores = ticks / before["clock_ticks_per_second"] / elapsed
    disappeared = sorted(set(first) - set(last) - excluded)
    return {
        "elapsed_monotonic_seconds": elapsed,
        "external_cpu_ticks_lower_bound": ticks,
        "external_cpu_cores_lower_bound": cores,
        "external_cpu_limit_cores": BACKGROUND_LIMIT,
        "passes_activity_screen": cores <= BACKGROUND_LIMIT,
        "endpoint_audit_flagged": cores > BACKGROUND_LIMIT,
        "excluded_identities": sorted(excluded),
        "external_active_processes": sorted(active, key=lambda p: -p["delta_ticks"]),
        "disappeared_external_identities": disappeared,
        "new_external_identities": sorted(set(last) - set(first) - excluded),
        "caveat": CPU_CAVEAT,
    }


def file_receipt(path):
    path = Path(path)
    return {"path": str(path), "sha256": harness.digest(path) if path.is_file() else None}


def worker_summary(receipt):
    result = receipt.get("result", {})
    fits = receipt.get("coefficient_fits", [])
    return {
        "status": receipt.get("status"),
        "timing_status": receipt.get("timing_status"),
        "fit_seconds": receipt.get("fit_seconds"),
        "fit_process_cpu_seconds": receipt.get("fit_process_cpu_seconds"),
        "peak_rss_mib": receipt.get("peak_fit_process_rss_bytes", 0) / (1024 * 1024)
        if "peak_fit_process_rss_bytes" in receipt
        else None,
        "peak_process_rss_bytes": receipt.get("peak_process_rss_bytes"),
        "source_tree_sha256": receipt.get("source", {}).get("source_tree_sha256"),
        "source_sha": receipt.get("source", {}).get("sha"),
        "source_stable": receipt.get("source_stable"),
        "result": {
            key: result.get(key)
            for key in (
                "converged",
                "coefficient_converged",
                "smoothing_converged",
                "smoothing_reason",
                "smoothing_certified",
                "n_inner_iter",
                "n_smoothing_iter",
            )
        },
        "coefficient_fit_backends": [fit.get("execution_backend_identifier") for fit in fits],
        "coefficient_fit_iterations": [fit.get("iterations") for fit in fits],
        "phase_snapshot": receipt.get("phase_snapshot"),
        "instrumentation": receipt.get("instrumentation"),
    }


def compare_arrays(reference, other):
    """Descriptive errors only: no inferred equivalence or tolerance assertions."""
    import numpy as np

    differences = {}
    with (
        np.load(reference, allow_pickle=False) as left,
        np.load(other, allow_pickle=False) as right,
    ):
        for name in sorted(set(left.files) | set(right.files)):
            if name not in left.files or name not in right.files:
                differences[name] = {"status": "missing_array"}
                continue
            a, b = np.asarray(left[name]), np.asarray(right[name])
            if a.shape != b.shape:
                differences[name] = {
                    "status": "shape_mismatch",
                    "reference_shape": list(a.shape),
                    "other_shape": list(b.shape),
                }
                continue
            if not (np.isfinite(a).all() and np.isfinite(b).all()):
                differences[name] = {"status": "nonfinite", "shape": list(a.shape)}
                continue
            delta = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
            denominator = float(np.linalg.norm(a.ravel()))
            absolute_norm = float(np.linalg.norm(delta.ravel()))
            relative = absolute_norm / denominator if denominator else None
            differences[name] = {
                "status": "descriptive_difference",
                "shape": list(a.shape),
                "max_absolute_difference": float(np.max(np.abs(delta))) if delta.size else 0.0,
                "difference_norm": absolute_norm,
                "reference_norm": denominator,
                "norm_relative_difference": relative,
                "zero_reference_norm": denominator == 0,
            }
    return differences


def comparison_report(manifest):
    runs = manifest["runs"]
    lookup = {(r["repeat"], r["source_label"], r["representation"]): r for r in runs}
    comparisons = []
    for repeat in sorted({r["repeat"] for r in runs}):
        pairs = [
            (
                "baseline_candidate_same_representation",
                (repeat, "baseline", mode),
                (repeat, "candidate", mode),
            )
            for mode in ("exact", "discrete")
        ] + [
            (
                "candidate_discrete_exact_comparison",
                (repeat, "candidate", "exact"),
                (repeat, "candidate", "discrete"),
            )
        ]
        for kind, akey, bkey in pairs:
            if akey not in lookup or bkey not in lookup:
                continue
            left, right = lookup[akey], lookup[bkey]
            item = {
                "kind": kind,
                "repeat": repeat,
                "reference": left["run_id"],
                "other": right["run_id"],
                "reference_result": left.get("worker_summary"),
                "other_result": right.get("worker_summary"),
            }
            a, b = left["artifacts"]["arrays"], right["artifacts"]["arrays"]
            if a["sha256"] and b["sha256"]:
                item["array_differences"] = compare_arrays(a["path"], b["path"])
            else:
                item["status"] = "missing_worker_arrays"
            comparisons.append(item)
    aggregates = []
    for label in ("baseline", "candidate"):
        for mode in ("exact", "discrete"):
            subset = [r for r in runs if r["source_label"] == label and r["representation"] == mode]
            item = {
                "source_label": label,
                "representation": mode,
                "runs": len(subset),
                "endpoint_flagged_runs": [
                    r["run_id"] for r in subset if r["cpu_audit"]["endpoint_audit_flagged"]
                ],
            }
            for key in ("fit_seconds", "fit_process_cpu_seconds", "peak_rss_mib"):
                values = [r.get("worker_summary", {}).get(key) for r in subset]
                values = [v for v in values if isinstance(v, (float, int)) and math.isfinite(v)]
                item[key] = (
                    {
                        "count": len(values),
                        "median": statistics.median(values),
                        "min": min(values),
                        "max": max(values),
                    }
                    if values
                    else None
                )
            aggregates.append(item)
    return {
        "interpretation": (
            "Differences are descriptive. Discrete execution may preserve the observed "
            "support exactly when the bin budget covers that support, or approximate "
            "a larger support through binning. Classifying a run requires evidence "
            "about its actual support and constructed representation; fixture names "
            "alone are insufficient. Floating-point differences may occur with exact "
            "support, and small errors do not establish equivalence."
        ),
        "timing_caveat": "All available worker timings are retained, including endpoint-flagged runs; inspect audit flags before interpretation.",
        "aggregates": aggregates,
        "comparisons": comparisons,
    }


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--repeat", type=int, default=3)
    fixture_action = harness.parser()._option_string_actions["--fixture"]
    p.add_argument("--fixture", choices=fixture_action.choices, default=fixture_action.default)
    p.add_argument("--n", type=int)
    p.add_argument("--knots", type=int)
    p.add_argument("--n-bins", type=int, default=256)
    p.add_argument("--support-size", type=int)
    p.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    p.add_argument("--replicate", type=int, default=1)
    p.add_argument(
        "--instrument", action="store_true", help="Separate untimed dispatch profiling run"
    )
    p.add_argument("--quiet-profile", help="Operator description of the quiet measurement session")
    return p


def main():
    args = parser().parse_args()
    if args.repeat < 1 or args.replicate < 1 or args.n_bins < 1:
        raise SystemExit("--repeat, --replicate and --n-bins must be positive")
    if not args.instrument and not (args.quiet_profile and args.quiet_profile.strip()):
        raise SystemExit("Timed runs require --quiet-profile identifying the operator's session")
    if not Path("/proc/self/stat").is_file() or not hasattr(os, "waitid"):
        raise SystemExit("Linux /proc and waitid are required for process endpoint accounting")
    roots = {"baseline": args.baseline.resolve(), "candidate": args.candidate.resolve()}
    for source in roots.values():
        if not (source / "src/superglm").is_dir():
            raise SystemExit(f"Missing source tree: {source}")
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    environment.update(dict.fromkeys(THREAD_VARIABLES, "1"))
    worker = Path(harness.__file__).resolve()
    manifest = {
        "schema": 1,
        "controller": file_receipt(__file__),
        "worker": file_receipt(worker),
        "fixtures": file_receipt(worker.with_name("_c3_c1_fixtures.py")),
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "python_executable": sys.executable,
        "python_version": sys.version,
        "controller_pid": os.getpid(),
        "thread_environment": {key: environment[key] for key in THREAD_VARIABLES},
        "inherited_numba_environment": {
            key: value
            for key, value in environment.items()
            if key.startswith("NUMBA_") and key not in THREAD_VARIABLES
        },
        "cache_policy": "Retain inherited Numba cache settings and existing caches; no cache clearing or private cache directory.",
        "warmup_policy": "Fresh interpreter per fit; existing worker calls superglm.warmup() before fit timing. Other cache/JIT misses may occur during fit. Process peak RSS includes imports, fixture construction, warmup and fit.",
        "order_policy": "baseline exact, baseline discrete, candidate exact, candidate discrete; reverse on odd repetition indices (forward/reverse/forward for default repeat=3)",
        "source_before": {key: harness.source_receipt(root) for key, root in roots.items()},
        "status": "preflight",
        "runs": [],
    }
    manifest_path = output / "manifest.json"
    write_json(manifest_path, manifest, exclusive=True)
    before = process_snapshot()
    time.sleep(PREFLIGHT_SECONDS)
    after = process_snapshot()
    audit = background_audit(before, after, {os.getpid()})
    preflight_path = output / "preflight.json"
    write_json(preflight_path, {"before": before, "after": after, "audit": audit}, exclusive=True)
    manifest["preflight"] = {**file_receipt(preflight_path), "audit": audit}
    overloaded = after["load_average"][0] > 2 * len(after["affinity"])
    if not audit["passes_activity_screen"] or overloaded:
        manifest["status"] = "aborted_preflight"
        manifest["preflight"]["load_guard_failed"] = overloaded
        write_json(manifest_path, manifest)
        print(f"Preflight rejected activity/headroom; see {manifest_path}", file=sys.stderr)
        return 2
    manifest["status"] = "running"
    write_json(manifest_path, manifest)
    arms = [(label, mode) for label in roots for mode in ("exact", "discrete")]
    for repeat in range(args.repeat):
        for label, mode in arms if repeat % 2 == 0 else list(reversed(arms)):
            run_id = f"r{repeat:02d}-{label}-{mode}"
            receipt_path = output / f"{run_id}.json"
            log_path = output / f"{run_id}.log"
            cpu_path = output / f"{run_id}.cpu.json"
            cmd = [
                sys.executable,
                str(worker),
                "--worker-source",
                str(roots[label]),
                "--worker-output",
                str(receipt_path),
                "--fixture",
                args.fixture,
                "--n-bins",
                str(args.n_bins),
                "--replicate",
                str(args.replicate),
                "--data",
                str(args.data.resolve()),
                "--threads",
                "1",
                "--warmup",
                "--discrete" if mode == "discrete" else "--no-discrete",
            ]
            for option in ("n", "knots", "support_size"):
                value = getattr(args, option)
                if value is not None:
                    cmd.extend(["--" + option.replace("_", "-"), str(value)])
            cmd.extend(
                ["--instrument"]
                if args.instrument
                else ["--measure-time", "--quiet-profile", args.quiet_profile]
            )
            run_env = {**environment, "PYTHONPATH": str(roots[label] / "src")}
            before = process_snapshot()
            started = time.monotonic_ns()
            with log_path.open("x") as log:
                child = subprocess.Popen(
                    cmd, cwd=roots[label], env=run_env, stdout=log, stderr=subprocess.STDOUT
                )
                spawned = process_snapshot()
                # Observe the completed worker while it is still a zombie, so its
                # identity and total CPU ticks survive until the endpoint read.
                os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOWAIT)
                after = process_snapshot()
                returncode = child.wait()
            finished = time.monotonic_ns()
            audit = background_audit(before, after, {os.getpid(), child.pid})
            write_json(
                cpu_path,
                {
                    "before_spawn": before,
                    "after_spawn": spawned,
                    "after_exit_before_reap": after,
                    "audit": audit,
                    "worker_pid": child.pid,
                    "worker_processes_at_spawn": sorted(
                        descendant_identities(spawned, {child.pid})
                    ),
                    "worker_processes_at_exit": sorted(descendant_identities(after, {child.pid})),
                    "process_elapsed_monotonic_seconds": (finished - started) / 1e9,
                },
                exclusive=True,
            )
            run = {
                "run_id": run_id,
                "repeat": repeat,
                "source_label": label,
                "representation": mode,
                "command": cmd,
                "cwd": str(roots[label]),
                "worker_pid": child.pid,
                "returncode": returncode,
                "cpu_audit": audit,
                "artifacts": {
                    "receipt": file_receipt(receipt_path),
                    "log": file_receipt(log_path),
                    "arrays": file_receipt(receipt_path.with_suffix(".npz")),
                    "cpu": file_receipt(cpu_path),
                },
            }
            if receipt_path.is_file():
                receipt = json.loads(receipt_path.read_text())
                run["worker_summary"] = worker_summary(receipt)
            manifest["runs"].append(run)
            write_json(manifest_path, manifest)
            print(
                f"{run_id}: returncode={returncode}; external CPU lower bound={audit['external_cpu_cores_lower_bound']:.3f} cores",
                flush=True,
            )
    manifest["source_after"] = {key: harness.source_receipt(root) for key, root in roots.items()}
    manifest["source_freeze_verified"] = all(
        manifest["source_before"][key]["source_tree_sha256"]
        == manifest["source_after"][key]["source_tree_sha256"]
        and manifest["source_before"][key]["sha"] == manifest["source_after"][key]["sha"]
        and all(
            r.get("worker_summary", {}).get("source_tree_sha256")
            == manifest["source_before"][key]["source_tree_sha256"]
            and r.get("worker_summary", {}).get("source_sha")
            == manifest["source_before"][key]["sha"]
            and r.get("worker_summary", {}).get("source_stable") is True
            for r in manifest["runs"]
            if r["source_label"] == key
        )
        for key in roots
    )
    manifest["status"] = (
        "complete"
        if all(r["returncode"] == 0 for r in manifest["runs"])
        and manifest["source_freeze_verified"]
        else "failed_or_source_changed"
    )
    comparison_path = output / "comparison.json"
    write_json(comparison_path, comparison_report(manifest), exclusive=True)
    manifest["comparison"] = file_receipt(comparison_path)
    write_json(manifest_path, manifest)
    print(f"Raw receipts: {manifest_path}")
    return 0 if manifest["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
