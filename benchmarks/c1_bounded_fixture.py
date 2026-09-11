"""Replay the existing fragmented Gaussian fixture without a resident full frame.

Column generation follows gaussian_fragmented_fixture's RNG order exactly.
The directory is benchmark input, not a SuperGLM fitted artifact or an authority
boundary. Prepared fitting must validate/copy these batches through its own API.
Use a disk filesystem for capacity measurements, not a memory-backed /tmp.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pandas as pd

NUMERIC_COLUMNS = tuple([f"linear{i}" for i in range(6)] + ["smooth0", "smooth1", "curve"])
CATEGORY_LEVELS = {"category0": 3, "category1": 3, "category2": 4, "group": 3}
DTYPES = dict.fromkeys(NUMERIC_COLUMNS, "<f8") | dict.fromkeys(CATEGORY_LEVELS, "<i4")
DTYPES["y"] = "<f8"


def _read_columns(handles, rows):
    values = {}
    for name, handle in handles.items():
        dtype = np.dtype(DTYPES[name])
        payload = handle.read(rows * dtype.itemsize)
        if len(payload) != rows * dtype.itemsize:
            raise ValueError(f"Truncated benchmark column: {name}")
        values[name] = np.frombuffer(payload, dtype=dtype)
    return values


def write_fixture(path: Path, n: int, *, batch_rows: int = 65536) -> dict:
    """Write the same rows as gaussian_fragmented_fixture(n, 4), in bounded RAM."""
    if type(n) is not int or n <= 0 or type(batch_rows) is not int or batch_rows <= 0:
        raise ValueError("n and batch_rows must be positive integers")
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(28109)
    hashes = {}
    start_time = time.perf_counter()
    start_cpu = time.process_time()

    def write_column(name, generate):
        digest = hashlib.sha256()
        with (path / f"{name}.bin").open("xb") as handle:
            for start in range(0, n, batch_rows):
                count = min(batch_rows, n - start)
                payload = generate(count).astype(DTYPES[name], copy=False).tobytes()
                handle.write(payload)
                digest.update(payload)
        hashes[name] = digest.hexdigest()

    for name in NUMERIC_COLUMNS:
        write_column(name, lambda count: rng.uniform(-1, 1, count))
    for name, levels in CATEGORY_LEVELS.items():
        write_column(name, lambda count, levels=levels: rng.integers(levels, size=count))

    # Formula and arithmetic order match c3_c1_complete_fit.py. Generate all
    # feature columns first, because its response noise consumes the same RNG.
    with ExitStack() as stack:
        handles = {
            name: stack.enter_context((path / f"{name}.bin").open("rb"))
            for name in DTYPES
            if name != "y"
        }

        def response(count):
            frame = _read_columns(handles, count)
            linear = sum((0.12 + i * 0.035) * frame[f"linear{i}"] for i in range(6))
            effects = np.asarray([-0.15, 0.0, 0.15, 0.3])
            categorical = sum(effects[frame[f"category{i}"]] for i in range(3))
            group = np.asarray([-1.0, 0.0, 1.0])[frame["group"]]
            curve = frame["curve"]
            smooth0, smooth1 = frame["smooth0"], frame["smooth1"]
            group_curve = (0.7 + 0.25 * group) * np.sin(2.5 * curve) + 0.2 * group * curve
            mean = (
                0.4
                + linear
                + categorical
                + 0.6 * np.sin(2.7 * smooth0)
                + 0.4 * smooth1**2
                + group_curve
            )
            log_sigma = (
                -0.45
                + 0.2 * linear
                + 0.3 * categorical
                + 0.18 * np.cos(2.4 * smooth0)
                + 0.15 * np.sin(2.2 * smooth1)
                + 0.18 * (1 + 0.25 * group) * np.cos(2.1 * curve)
                + 0.08 * group
            )
            return mean + np.exp(log_sigma) * rng.normal(size=count)

        write_column("y", response)

    receipt = {
        "format": "c1-fragmented-gaussian-input/v1",
        "rows": n,
        "seed": 28109,
        "generation_batch_rows": batch_rows,
        "dtypes": DTYPES,
        "sha256": hashes,
        "payload_bytes": n * sum(np.dtype(dtype).itemsize for dtype in DTYPES.values()),
        "generation_seconds": time.perf_counter() - start_time,
        "generation_cpu_seconds": time.process_time() - start_cpu,
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "original_generator_sha256": hashlib.sha256(
            Path(__file__).with_name("c3_c1_complete_fit.py").read_bytes()
        ).hexdigest(),
    }
    (path / "manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def iter_fixture(path: Path, *, batch_rows: int = 65536):
    """Yield ordinary column batches, including the response column named y."""
    if type(batch_rows) is not int or batch_rows <= 0:
        raise ValueError("batch_rows must be a positive integer")
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest["format"] != "c1-fragmented-gaussian-input/v1" or manifest["dtypes"] != DTYPES:
        raise ValueError("Unsupported benchmark input")
    n = manifest["rows"]
    with ExitStack() as stack:
        handles = {name: stack.enter_context((path / f"{name}.bin").open("rb")) for name in DTYPES}
        for start in range(0, n, batch_rows):
            count = min(batch_rows, n - start)
            columns = _read_columns(handles, count)
            for name, levels in CATEGORY_LEVELS.items():
                columns[name] = np.asarray([f"level{j}" for j in range(levels)])[columns[name]]
            yield pd.DataFrame(columns, index=pd.RangeIndex(start, start + count))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--batch-rows", type=int, default=65536)
    args = parser.parse_args()
    print(json.dumps(write_fixture(args.out, args.n, batch_rows=args.batch_rows)))


if __name__ == "__main__":
    main()
