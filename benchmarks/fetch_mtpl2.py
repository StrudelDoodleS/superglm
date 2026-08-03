"""Fetch freMTPL2freq (OpenML dataset 41214) into data/freMTPL2freq.parquet.

Used by the local performance workflow to populate the git-ignored data
directory on demand. Skips the download when the parquet already exists
unless ``--force`` is given.

Usage::

    uv run python benchmarks/fetch_mtpl2.py
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

OPENML_DATASET_ID = 41214
API_URL = f"https://api.openml.org/api/v1/json/data/{OPENML_DATASET_ID}"
ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "data" / "freMTPL2freq.parquet"

NUMERIC_INT_COLUMNS = ("ClaimNb", "VehPower", "VehAge", "DrivAge", "BonusMalus", "Density")
NOMINAL_COLUMNS = ("Area", "VehBrand", "VehGas", "Region")


def _download(url: str, attempts: int = 3) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "superglm-local-perf"})
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == attempts:
                raise
            wait = 10 * attempt
            print(f"download attempt {attempt} failed ({exc}); retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def fetch(out_path: Path = OUT_PATH, force: bool = False) -> Path:
    if out_path.exists() and not force:
        print(f"{out_path} already exists; skipping fetch")
        return out_path

    description = json.loads(_download(API_URL))["data_set_description"]
    arff_url = description["url"]
    print(f"downloading {arff_url}")
    raw = _download(arff_url).decode("utf-8")

    # scipy.io.arff rejects @attribute ... string, so parse the ARFF by hand:
    # the @data section is plain CSV with single-quoted nominals/strings.
    header, _, data_section = raw.partition("@data")
    columns = [
        line.split()[1] for line in header.splitlines() if line.lower().startswith("@attribute")
    ]
    df = pd.read_csv(io.StringIO(data_section), names=columns, quotechar="'")
    for column in NOMINAL_COLUMNS:
        df[column] = df[column].astype("category")
    for column in NUMERIC_INT_COLUMNS:
        df[column] = df[column].astype("int64")
    df["IDpol"] = df["IDpol"].astype("float64")
    df["Exposure"] = df["Exposure"].astype("float64")

    if len(df) != 678013:
        raise RuntimeError(f"unexpected row count {len(df)} (expected 678013)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    print(f"wrote {out_path} ({len(df):,} rows)")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    parser.add_argument("--out", default=str(OUT_PATH), help="output parquet path")
    args = parser.parse_args()
    fetch(Path(args.out), force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
