from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from superglm._fit_trace import SCHEMA_VERSION, JSONLTraceSink, TraceRun

TraceRow = dict[str, Any]
TRACE_SUFFIXES = ("reml", "pirls", "scop")


@dataclass
class REMLIterRow:
    run_id: str
    iteration: int
    objective_before: float
    objective_after: float
    lambda_max_delta: float


@dataclass(frozen=True)
class REMLDebugRun:
    """Loaded trace bundle for one ``fit_reml()`` debug run."""

    run_id: str
    base_dir: Path
    metadata: TraceRow
    reml_rows: list[TraceRow]
    pirls_rows: list[TraceRow]
    scop_rows: list[TraceRow]
    events: list[TraceRow]

    def artifact_path(self, suffix: str) -> Path:
        """Return the artifact path for one run-local suffix."""
        return self.base_dir / f"{self.run_id}_{suffix}"

    @property
    def trace_paths(self) -> dict[str, Path]:
        """Return the expected artifact paths for this run."""
        return {
            "run": self.artifact_path("run.json"),
            "reml": self.artifact_path("reml.jsonl"),
            "pirls": self.artifact_path("pirls.jsonl"),
            "scop": self.artifact_path("scop.jsonl"),
        }

    @property
    def lambda_names(self) -> list[str]:
        """Return sorted lambda component names seen in REML rows."""
        names: set[str] = set()
        for row in self.reml_rows:
            lambdas = row.get("lambdas")
            if isinstance(lambdas, dict):
                names.update(str(name) for name in lambdas)
        return sorted(names)


class REMLDebugRecorder:
    def __init__(self, enabled_level: int, base_dir: Path, run_id: str):
        self.enabled_level = enabled_level
        self.base_dir = base_dir
        self.run_id = run_id
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.trace_run = (
            TraceRun(run_id, sink=JSONLTraceSink(base_dir, run_id)) if enabled_level >= 2 else None
        )

    def write_run_metadata(self, payload: dict) -> None:
        path = self.base_dir / f"{self.run_id}_run.json"
        existing = _load_json(path) if path.exists() else {}
        metadata = {
            **existing,
            **payload,
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
        }
        path.write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

    def append_jsonl(self, suffix: str, payload: dict) -> None:
        path = self.base_dir / f"{self.run_id}_{suffix}.jsonl"
        # Existing recorders do not have evaluated state identities.  Preserve
        # their flat compatibility shape, but make their non-authoritative
        # status explicit rather than pretending they are canonical events.
        payload = {
            **payload,
            "purpose": "legacy_compatibility",
            "authoritative": False,
        }
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload) + "\n")


def list_reml_debug_run_ids(base_dir: Path) -> list[str]:
    """Return sorted run IDs discovered in ``base_dir``."""
    if not base_dir.exists():
        return []
    return sorted(path.name.removesuffix("_run.json") for path in base_dir.glob("*_run.json"))


def _load_json(path: Path) -> TraceRow:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def load_jsonl_rows(path: Path) -> list[TraceRow]:
    """Load JSONL rows from ``path``. Missing files yield an empty list."""
    if not path.exists():
        return []
    rows: list[TraceRow] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _is_canonical_event(row: TraceRow) -> bool:
    return "schema_version" in row


def _validate_canonical_event(row: TraceRow, *, path: Path, run_id: str) -> None:
    required = {
        "schema_version",
        "run_id",
        "sequence",
        "timestamp",
        "event_kind",
        "channel",
        "purpose",
        "authoritative",
        "payload",
    }
    missing = sorted(required.difference(row))
    if missing:
        raise ValueError(f"Malformed canonical trace row in {path}: missing {missing}")
    schema_version = row["schema_version"]
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version <= 0
    ):
        raise ValueError(f"Canonical trace schema_version in {path} must be a positive integer")
    if row["run_id"] != run_id:
        raise ValueError(
            f"Canonical trace row in {path} belongs to run {row['run_id']!r}, expected {run_id!r}"
        )
    sequence = row["sequence"]
    if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence <= 0:
        raise ValueError(f"Canonical trace sequence in {path} must be a positive integer")
    if not isinstance(row["payload"], dict):
        raise ValueError(f"Canonical trace payload in {path} must be an object")


def _compatibility_rows(rows: list[TraceRow]) -> list[TraceRow]:
    projected: list[TraceRow] = []
    for row in rows:
        if not _is_canonical_event(row):
            projected.append(row)
            continue
        payload = dict(row["payload"])
        payload.setdefault("trace_sequence", row["sequence"])
        payload.setdefault("trace_state_id", payload.get("state_id"))
        payload.setdefault("trace_authoritative", row["authoritative"])
        payload.setdefault("trace_purpose", row["purpose"])
        projected.append(payload)
    return projected


def load_reml_debug_run(base_dir: Path, run_id: str) -> REMLDebugRun:
    """Load one REML debug run and its known trace files."""
    run_path = base_dir / f"{run_id}_run.json"
    if not run_path.exists():
        raise FileNotFoundError(f"Missing REML debug metadata: {run_path}")
    channel_rows = {
        suffix: load_jsonl_rows(base_dir / f"{run_id}_{suffix}.jsonl") for suffix in TRACE_SUFFIXES
    }
    events: list[TraceRow] = []
    seen_sequences: set[int] = set()
    for path in sorted(base_dir.glob(f"{run_id}_*.jsonl")):
        for row in load_jsonl_rows(path):
            if not _is_canonical_event(row):
                continue
            # ``run`` and ``run_child`` share the filesystem glob ``run_*``.
            # The embedded run identity, not a filename-prefix guess, owns a
            # canonical event.
            if row.get("run_id") not in (None, run_id):
                continue
            _validate_canonical_event(row, path=path, run_id=run_id)
            sequence = int(row["sequence"])
            if sequence in seen_sequences:
                raise ValueError(
                    f"duplicate canonical trace sequence {sequence} for run {run_id!r}"
                )
            seen_sequences.add(sequence)
            events.append(row)
    events.sort(key=lambda row: int(row["sequence"]))
    sequences = [int(row["sequence"]) for row in events]
    if sequences != list(range(1, len(sequences) + 1)):
        raise ValueError(f"noncontiguous canonical trace sequence for run {run_id!r}: {sequences}")

    return REMLDebugRun(
        run_id=run_id,
        base_dir=base_dir,
        metadata=_load_json(run_path),
        reml_rows=_compatibility_rows(channel_rows["reml"]),
        pirls_rows=_compatibility_rows(channel_rows["pirls"]),
        scop_rows=_compatibility_rows(channel_rows["scop"]),
        events=events,
    )


def load_reml_debug_runs(base_dir: Path) -> list[REMLDebugRun]:
    """Load all discovered REML debug runs from ``base_dir``."""
    return [load_reml_debug_run(base_dir, run_id) for run_id in list_reml_debug_run_ids(base_dir)]


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _int_or_none(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def summarize_reml_debug_run(run: REMLDebugRun) -> TraceRow:
    """Build a machine-readable summary row for one debug run."""
    reml_last = run.reml_rows[-1] if run.reml_rows else {}
    final_lambdas = reml_last.get("lambdas") if isinstance(reml_last.get("lambdas"), dict) else {}
    lambda_deltas = [
        float(row["lambda_max_delta"])
        for row in run.reml_rows
        if row.get("lambda_max_delta") is not None
    ]
    scop_step_norms = [
        float(row["step_norm"]) for row in run.scop_rows if row.get("step_norm") is not None
    ]
    final_phase_rows = [
        row
        for row in run.pirls_rows
        if isinstance(row.get("phase"), str) and row["phase"] == "final"
    ]
    final_pirls_row = (
        final_phase_rows[-1] if final_phase_rows else (run.pirls_rows[-1] if run.pirls_rows else {})
    )

    summary: TraceRow = {
        "run_id": run.run_id,
        "method": run.metadata.get("method"),
        "family": run.metadata.get("family"),
        "link": run.metadata.get("link"),
        "discrete": bool(run.metadata.get("discrete", False)),
        "n_obs": _int_or_none(run.metadata.get("n_obs")),
        "n_columns": _int_or_none(run.metadata.get("n_columns")),
        "n_groups": _int_or_none(run.metadata.get("n_groups")),
        "has_constraints": bool(run.metadata.get("has_constraints", False)),
        "has_qp_constraints": bool(run.metadata.get("has_qp_constraints", False)),
        "has_scop_constraints": bool(run.metadata.get("has_scop_constraints", False)),
        "reml_iterations": _int_or_none(reml_last.get("iteration")) or 0,
        "final_objective_before": _float_or_none(reml_last.get("objective_before")),
        "final_objective_after": _float_or_none(reml_last.get("objective_after")),
        "final_lambda_max_delta": _float_or_none(reml_last.get("lambda_max_delta")),
        "max_lambda_max_delta": max(lambda_deltas) if lambda_deltas else None,
        "strict_converged": bool(reml_last.get("strict_converged", False)),
        "plateau_converged": bool(reml_last.get("plateau_converged", False)),
        "n_reml_rows": len(run.reml_rows),
        "n_pirls_rows": len(run.pirls_rows),
        "n_scop_rows": len(run.scop_rows),
        "final_pirls_iteration": _int_or_none(final_pirls_row.get("iteration")),
        "final_pirls_deviance": _float_or_none(final_pirls_row.get("deviance")),
        "max_scop_step_norm": max(scop_step_norms) if scop_step_norms else None,
        "scop_fisher_fallbacks": sum(
            bool(row.get("used_fisher_fallback", False)) for row in run.scop_rows
        ),
        "final_lambdas_json": json.dumps(final_lambdas, sort_keys=True),
        "estimated_names_json": json.dumps(reml_last.get("estimated_names", [])),
    }
    return summary


def summarize_reml_debug_runs(runs: Iterable[REMLDebugRun]) -> list[TraceRow]:
    """Build machine-readable summary rows for a sequence of runs."""
    return [summarize_reml_debug_run(run) for run in runs]


def write_reml_debug_summary_csv(rows: Iterable[TraceRow], output_path: Path) -> Path:
    """Write summary rows to CSV, preserving first-seen column order."""
    row_list = list(rows)
    fieldnames: list[str] = []
    seen: set[str] = set()

    for row in row_list:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        if row_list:
            writer.writerows(row_list)
    return output_path


def plot_reml_debug_trajectory(
    run: REMLDebugRun,
    output_path: Path,
    *,
    title: str | None = None,
) -> Path:
    """Write a two-panel objective/lambda convergence plot for one run."""
    if not run.reml_rows:
        raise ValueError(f"Run {run.run_id} has no REML rows to plot")

    import matplotlib.pyplot as plt

    iterations = [int(row["iteration"]) for row in run.reml_rows]
    objective_before = [float(row["objective_before"]) for row in run.reml_rows]
    objective_after = [float(row["objective_after"]) for row in run.reml_rows]
    lambda_names = run.lambda_names

    fig, axes = plt.subplots(2, 1, figsize=(10.0, 7.2), sharex=True)

    ax = axes[0]
    ax.plot(iterations, objective_before, marker="o", color="#264653", label="objective before")
    ax.plot(iterations, objective_after, marker="o", color="#e76f51", label="objective after")
    ax.set_ylabel("objective")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[1]
    if lambda_names:
        for name in lambda_names:
            values = []
            for row in run.reml_rows:
                lambdas = row.get("lambdas")
                value = None
                if isinstance(lambdas, dict):
                    value = lambdas.get(name)
                values.append(max(float(value), 1e-12) if value is not None else float("nan"))
            ax.plot(iterations, values, marker="o", label=name)
        ax.set_yscale("log")
        ax.legend(frameon=False)
    else:
        ax.text(
            0.5,
            0.5,
            "No lambda rows recorded",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    ax.set_xlabel("REML iteration")
    ax.set_ylabel("lambda")
    ax.grid(alpha=0.25)

    fig.suptitle(title or f"REML debug trajectory: {run.run_id}")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


__all__ = [
    "REMLDebugRecorder",
    "REMLDebugRun",
    "REMLIterRow",
    "TRACE_SUFFIXES",
    "list_reml_debug_run_ids",
    "load_jsonl_rows",
    "load_reml_debug_run",
    "load_reml_debug_runs",
    "plot_reml_debug_trajectory",
    "summarize_reml_debug_run",
    "summarize_reml_debug_runs",
    "write_reml_debug_summary_csv",
]
