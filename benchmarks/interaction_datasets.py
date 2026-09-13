"""Fetch and validate the public interaction-research corpus without fitting models.

    uv run python benchmarks/interaction_datasets.py list
    uv run python benchmarks/interaction_datasets.py fetch
    uv run python benchmarks/interaction_datasets.py validate uci_abalone

The registry pins bytes and schemas. Receipts describe actual availability.
Loading preserves missing values and the raw target scale. Preprocessing,
feature encoding and implementation of the documented split belong to trial
adapters, which must fit preprocessing only on their training partition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import time
import urllib.request
import uuid
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

MANIFEST = Path(__file__).with_suffix(".json")
DEFAULT_ROOT = Path(__file__).resolve().parents[1] / ".benchmark-artifacts/interaction-datasets"
DEFAULT_BUDGET = 300 * 1024**2


def read_manifest(path=MANIFEST):
    """Read an extensible registry, rejecting ambiguous dataset identities."""
    manifest = json.loads(Path(path).read_text())
    if manifest["schema_version"] != 1:
        raise ValueError("Unsupported corpus schema_version")
    entries = manifest["datasets"]
    names = [entry["id"] for entry in entries]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate dataset IDs in manifest")
    if any(not re.fullmatch(r"[a-z0-9_]+", name) for name in names):
        raise ValueError("Dataset IDs must contain only lowercase letters, digits and underscores")
    return entries


def source_path(entry, root):
    """Resolve a downloaded artifact or an explicitly registered local reference."""
    source = entry["source"]
    if entry["availability"] == "local_reference":
        return Path(source["reference_path"]).expanduser()
    filename = source["filename"]
    if Path(filename).name != filename or filename in {".", ".."}:
        raise ValueError("source.filename must be a plain filename")
    if not re.fullmatch(r"[a-z0-9_]+", entry["id"]):
        raise ValueError("Invalid dataset ID")
    return Path(root) / entry["id"] / filename


def verify_bytes(source, path):
    """Require a complete file with the exact recorded content identity."""
    size = path.stat().st_size
    if size != source["bytes"]:
        raise ValueError(f"Artifact bytes {size} != expected {source['bytes']}: {path}")
    if size > source["max_bytes"]:
        raise ValueError(f"Artifact exceeds byte limit: {path}")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != source["sha256"]:
        raise ValueError(f"Artifact SHA256 mismatch: {path}")
    return digest


def download(source, destination, *, timeout=30):
    """Publish a complete pinned download atomically, without replacing any file."""
    if not source["url"].startswith("https://"):
        raise ValueError("Downloads require an explicit HTTPS URL")
    if source["bytes"] > source["max_bytes"]:
        raise ValueError("Declared artifact bytes exceed its byte limit")
    destination.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=".download-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            request = urllib.request.Request(
                source["url"],
                headers={
                    "User-Agent": "SuperGLM-public-dataset-research/1",
                    "Accept-Encoding": "identity",
                },
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                reported_size = response.headers.get("Content-Length")
                if reported_size is not None and int(reported_size) > source["max_bytes"]:
                    raise ValueError("HTTP Content-Length exceeds byte limit")
                size = 0
                while chunk := response.read(1024**2):
                    size += len(chunk)
                    if size > source["max_bytes"]:
                        raise ValueError("Downloaded bytes exceed byte limit")
                    if time.monotonic() - started > 4 * timeout:
                        raise TimeoutError("Total download deadline exceeded")
                    stream.write(chunk)
                if reported_size is not None and size != int(reported_size):
                    raise ValueError("Downloaded bytes disagree with HTTP Content-Length")
            stream.flush()
            os.fsync(stream.fileno())
        verify_bytes(source, temporary)
        # link() is atomic and refuses an existing destination, including a
        # concurrent writer. replace() would discard an existing user's file.
        os.link(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_frame(source, path):
    """Parse only declared tables. ZIP members are streamed, never extracted."""
    options = {"na_values": ["?"], **source.get("read_csv", {})}
    kind = source["format"]
    if kind == "csv":
        frame = pd.read_csv(path, **options)
    elif kind == "parquet":
        frame = pd.read_parquet(path)
    elif kind == "zip_csv":
        cap = source.get("max_expanded_bytes", 512 * 1024**2)
        with zipfile.ZipFile(path) as archive:

            def member(name, csv_options):
                info = archive.getinfo(name)
                if info.file_size > cap:
                    raise ValueError(f"ZIP member {name} exceeds expanded byte limit")
                with archive.open(info) as stream:
                    return pd.read_csv(stream, **csv_options)

            if "members" in source:
                parts = source["members"]
                if sum(archive.getinfo(part["member"]).file_size for part in parts) > cap:
                    raise ValueError("ZIP tables exceed the total expanded byte limit")
                frames = []
                for part in parts:
                    table = member(part["member"], options)
                    column = source["partition_column"]
                    if column in table:
                        raise ValueError("ZIP partition label would overwrite a column")
                    label = pd.Series(part["label"], index=table.index, name=column)
                    frames.append(pd.concat([table, label], axis=1))
                frame = pd.concat(frames, ignore_index=True)
            else:
                frame = member(source["member"], options)
            companion = source.get("companion")
            if companion is not None:
                extra = member(companion["member"], {"na_values": ["?"]})
                if len(extra) != len(frame):
                    raise ValueError("ZIP companion has a different row count")
                for column in companion["align_on"]:
                    if not frame[column].equals(extra[column]):
                        raise ValueError(f"ZIP companion alignment failed for {column}")
                for column in companion["columns"]:
                    if column in frame:
                        raise ValueError(f"ZIP companion would overwrite {column}")
                    frame[column] = extra[column]
    else:
        raise ValueError(f"Unsupported artifact format {kind!r}")
    if source.get("strip_column_names", False):
        frame.columns = frame.columns.str.strip()
    return frame


def validate_frame(entry, frame):
    """Check the observed table contract and the selected target's support."""
    schema = entry["schema"]
    if len(frame) != schema["rows"]:
        raise ValueError(f"Table rows {len(frame)} != expected {schema['rows']}")
    if list(frame.columns) != schema["columns"] or not frame.columns.is_unique:
        raise ValueError("Table columns differ from the pinned ordered schema")
    missing = {column: int(n) for column, n in frame.isna().sum().items() if n}
    if missing != schema["missing_counts"]:
        raise ValueError("Table missing-value counts differ from the pinned schema")
    target_name = entry["primary_target"]
    features = entry["features"]
    if len(features) != len(set(features)) or set(features) - set(frame.columns):
        raise ValueError("Predictor columns are duplicated or absent")
    if target_name in features or set(entry.get("exclude_columns", {})) & set(features):
        raise ValueError("The target or an excluded column appears in the predictors")
    if set(entry.get("categorical_columns", [])) - set(features):
        raise ValueError("Categorical columns must be selected predictors")
    text_columns = set(frame.select_dtypes(include=["object", "string", "category"]))
    if text_columns & set(features) - set(entry.get("categorical_columns", [])):
        raise ValueError("String/category predictors need an explicit categorical declaration")
    if set(entry.get("split", {}).get("columns", [])) - set(frame.columns):
        raise ValueError("Split columns are absent from the table")
    rule = entry.get("target_rule", {})
    target = frame[target_name]
    if target.isna().any() and not rule.get("allow_missing", False):
        raise ValueError("Missing target values are not allowed")
    observed = target.dropna()
    if not len(observed):
        raise ValueError("No observed target values")
    if "classes" in rule:
        if set(observed.unique()) != set(rule["classes"]):
            raise ValueError("Observed target classes differ from the declared classes")
    else:
        values = pd.to_numeric(observed, errors="raise").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite target values")
        if "minimum" in rule and (values < rule["minimum"]).any():
            raise ValueError("Observed target is below its declared minimum")
        if rule.get("integer", False) and (values != np.floor(values)).any():
            raise ValueError("Count target contains noninteger values")
    for column in frame.select_dtypes(include="number"):
        if np.isinf(frame[column].to_numpy(dtype=float)).any():
            raise ValueError(f"Nonfinite numeric values in {column}")
    return missing


def load_dataset(dataset_id, *, root=DEFAULT_ROOT, manifest=MANIFEST):
    """Return the verified raw frame, without encoding, imputation or splitting."""
    entries = {entry["id"]: entry for entry in read_manifest(manifest)}
    entry = entries[dataset_id]
    if entry["availability"] not in {"fetchable", "local_reference"}:
        raise ValueError(f"{dataset_id} is {entry['availability']}; it is not a ready table")
    path = source_path(entry, root)
    verify_bytes(entry["source"], path)
    frame = read_frame(entry["source"], path)
    validate_frame(entry, frame)
    return frame


def fetch_one(entry, root, *, allow_download=True, timeout=30):
    """Fetch if absent, then require every data-integrity check before readiness."""
    if entry["availability"] not in {"fetchable", "local_reference"}:
        raise ValueError(f"Dataset is {entry['availability']}")
    path = source_path(entry, root)
    downloaded = False
    if not path.exists():
        if not allow_download or entry["availability"] == "local_reference":
            raise FileNotFoundError(path)
        download(entry["source"], path, timeout=timeout)
        downloaded = True
    digest = verify_bytes(entry["source"], path)
    frame = read_frame(entry["source"], path)
    missing = validate_frame(entry, frame)
    target = frame[entry["primary_target"]]
    target_summary = {"observed": int(target.notna().sum()), "missing": int(target.isna().sum())}
    if pd.api.types.is_numeric_dtype(target):
        target_summary.update(minimum=float(target.min()), maximum=float(target.max()))
    else:
        target_summary["classes"] = target.value_counts().to_dict()
    return {
        "id": entry["id"],
        "independent_unit": entry.get("independent_unit", entry["id"]),
        "counts_as_real_source": entry.get("counts_as_real_source", True),
        "status": "ready",
        "storage": "local_reference" if entry["availability"] == "local_reference" else "download",
        "path": str(path.resolve()),
        "sha256": digest,
        "bytes": path.stat().st_size,
        "downloaded_this_run": downloaded,
        "rows": len(frame),
        "columns": list(frame.columns),
        "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "feature_count": len(entry["features"]),
        "missing_counts": missing,
        "target": target_summary,
        "adapter_ready": False,
        "adapter_boundary": "Raw data verified; apply the declared split and training-only preprocessing.",
    }


def run_collection(entries, root, *, download, budget_bytes=DEFAULT_BUDGET, timeout=30):
    """Archive successes and failures together; absence never becomes readiness."""
    root = Path(root)
    records = []
    reserved = 0
    for entry in entries:
        if entry["availability"] not in {"fetchable", "local_reference"}:
            records.append(
                {
                    "id": entry["id"],
                    "status": entry["availability"],
                    "reason": entry.get("reason", ""),
                }
            )
            continue
        try:
            path = source_path(entry, root)
            if download and not path.exists() and entry["availability"] == "fetchable":
                # Reserve the cap even when a download fails. An incomplete
                # response still consumed network bytes from this run's budget.
                cap = entry["source"]["max_bytes"]
                if reserved + cap > budget_bytes:
                    raise ValueError("Collection download byte budget exceeded")
                reserved += cap
            record = fetch_one(entry, root, allow_download=download, timeout=timeout)
        except Exception as exc:
            record = {
                "id": entry["id"],
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        records.append(record)
    receipt = {
        "schema_version": 1,
        "validated_utc": datetime.now(UTC).isoformat(),
        "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "registry_entries_sha256": hashlib.sha256(
            json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "download_budget_bytes": budget_bytes,
        "reserved_download_bytes": reserved,
        "ready_count": sum(record["status"] == "ready" for record in records),
        "independent_ready_real_sources": len(
            {
                record["independent_unit"]
                for record in records
                if record["status"] == "ready" and record["counts_as_real_source"]
            }
        ),
        "failed_count": sum(record["status"] == "failed" for record in records),
        "datasets": records,
    }
    receipt_dir = root / "receipts"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    path = receipt_dir / f"{datetime.now(UTC):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:8]}.json"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=receipt_dir, prefix=".receipt-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write((json.dumps(receipt, indent=2, allow_nan=False) + "\n").encode())
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path, receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["list", "fetch", "validate"])
    parser.add_argument("ids", nargs="*")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--budget-mib", type=float, default=300)
    parser.add_argument("--timeout", type=float, default=30)
    args = parser.parse_args(argv)
    if args.budget_mib <= 0 or args.timeout <= 0:
        parser.error("budget and timeout must be positive")
    entries = read_manifest(args.manifest)
    if args.ids:
        unknown = set(args.ids) - {entry["id"] for entry in entries}
        if unknown:
            parser.error(f"Unknown dataset IDs: {sorted(unknown)}")
        entries = [entry for entry in entries if entry["id"] in args.ids]
    if args.command == "list":
        for entry in entries:
            print(f"{entry['id']:32s} {entry['availability']:16s} {entry.get('title', '')}")
        return 0
    if not args.ids:
        entries = [
            entry for entry in entries if entry["availability"] in {"fetchable", "local_reference"}
        ]
    path, receipt = run_collection(
        entries,
        args.root,
        download=args.command == "fetch",
        budget_bytes=int(args.budget_mib * 1024**2),
        timeout=args.timeout,
    )
    for record in receipt["datasets"]:
        suffix = f" ({record['error']})" if "error" in record else ""
        print(f"{record['id']}: {record['status']}{suffix}")
    print(f"Ready {receipt['ready_count']}/{len(entries)}; receipt: {path}")
    return 0 if receipt["ready_count"] == len(entries) else 1


if __name__ == "__main__":
    sys.exit(main())
