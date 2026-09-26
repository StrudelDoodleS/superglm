"""Fetch the DVSA 2024 MOT results as a car-test credibility dataset.

Every MOT test on a car in Great Britain in 2024, one row per normal (first)
test, with a pass/fail response and the vehicle hierarchy make > make_model >
variant.  That hierarchy is the point: nested random effects at thousands of
levels over tens of millions of rows, which is where fitting more than one
large ``RandomEffect`` with REML is expected to get expensive.

Source: Driver and Vehicle Standards Agency, "MOT testing data results (2024)",
published on data.gov.uk as
https://edh-dvsa-data-gov-uk-files-prod.s3.eu-west-1.amazonaws.com/MOT+testing+data+results+(2024).zip
Contains public sector information licensed under the Open Government Licence
v3.0 (https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).
Field meanings follow the DVSA "MOT testing data user guide" v5.1.

The archive is pinned by SHA-256 and byte length (measured 2026-09-26; the
object's Last-Modified is 2025-05-13).  A mismatch is a hard error: the
cleaning rules below were measured on these bytes and nothing else.

What the pinned files actually are, which is not what the user guide says (it
describes pipe-delimited files with DD-MM-YYYY dates):

* twelve monthly CSVs ``test_result_2024MM.csv`` (7.7 GB unpacked), comma
  delimited, ``"``-quoted with backslash escapes (``"SERIES 3 109\\""``), ISO
  dates, ``test_mileage`` written as a float, and a ``completed_date`` column
  the guide does not list (not read here);
* each monthly file is fifteen parts concatenated, so its header line recurs
  fifteen times inside it;
* a test appears on several rows (66,857,355 rows for 42,598,624 tests).  Most
  repeats are identical; 743,965 tests (1.7%) have repeats that disagree, mostly on
  ``first_use_date`` (``1999-09-01`` against ``1999-12-31``) but also on class,
  vehicle_id, model, fuel, capacity, mileage and postcode.  The file cannot say
  which row is right, so those tests are dropped rather than resolved;
* no test_id appears in two monthly files and every test_date lies in its
  file's month, so repeats are collapsed one month at a time.  That is exact,
  and it bounds memory by one month of parsed columns, streamed from the
  archive in 64 MB blocks;
* ``fuel_type`` mixes the guide's codes with two full names,
  ``Hybrid Electric (Clean)`` and ``Electric``, which are mapped to HY and EL.

Filters, applied in this order; every run prints what each one removed:

1. the recurring header lines;
2. repeated rows of the same test, leaving one row per test;
3. tests whose repeated rows disagree on any column read;
4. tests not of class 4 (cars and light passenger vehicles);
5. tests that are not a normal test (``test_type`` NT), i.e. retests and appeals;
6. results other than P, F and PRS, i.e. abandoned and aborted tests (DVSA's
   own failure-rate statistics use the same three);
7. make ``UNCLASSIFIED``, a vehicle with no DVSA vehicle record;
8. first use on 1971-01-01, the DVLA's sentinel for an unknown date;
9. first use missing, before 1900 (keying errors such as ``1010-03-01``) or
   after the test.

Model ``UNCLASSIFIED`` under a known make is kept: the make is real, and the
unknown models of one make pool into a single make_model level.

Columns of the output:

``fail``
    1 if the test started in a failing state, result F or PRS (failed, then
    repaired at the station within the hour); 0 for a pass, P.  This is
    DVSA's initial failure rate.
``vehicle_age_years``
    Days from first use to the test, over 365.25.
``test_mileage``
    Odometer reading at the test in miles; NaN where none was taken (blank or 0).
``cylinder_capacity``
    Engine capacity in cc as the tester recorded it; NaN where blank or 0,
    which is nearly always an electric vehicle.
``fuel_type``
    Two-letter DVSA code: PE petrol, DI diesel, HY hybrid electric, EL
    electric, ED electric diesel, LP LPG, GB gas bi-fuel, OT other, and rarer
    gas, fuel-cell and steam codes.
``postcode_area``
    Postcode area of the testing station, not of the keeper; XX pools areas
    with fewer than five stations.
``make``
    Vehicle make.
``make_model``
    ``make|model``, so a model name shared by two makes is two levels.
``variant``
    ``make|model|fuel_type|capacity``, capacity rounded half-up to the nearest
    100 cc, or ``na`` where the capacity is missing.
``vehicle_id``
    DVSA's vehicle identifier (from registration and VIN), for a holdout split
    that keeps one vehicle's tests together.
``test_month``
    Calendar month of the test, 1 to 12.

Measured on the 2026-09-26 run: 32,549,581 tests on 31,078,052 vehicles, fail
rate 0.2820, with 7,294 makes, 38,366 make_models, 64,761 variants, 119 postcode
areas and 14 fuel codes.  ``make`` is free text with a long tail (6,776 makes
have fewer than ten tests, such as ``0PEL`` and ``dmc   de lorean``); it is
kept as recorded, not cleaned.  The run took 151 s at a peak RSS of 4.5 GB.

Usage::

    python scripts/fetch_dvsa_mot.py --dest data/

Exit status is 0 only when the archive is verified and the parquet is written.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
from pandas.api.types import union_categoricals

URL = (
    "https://edh-dvsa-data-gov-uk-files-prod.s3.eu-west-1.amazonaws.com/"
    "MOT+testing+data+results+(2024).zip"
)
#: SHA-256 and byte length of the bytes at ``URL``, measured 2026-09-26.
SHA256 = "1432996ede0e7fe89947ea5063ed13ef8b19a8f2e1268aaf261e9fec9bea1d6c"
SIZE = 1_531_265_132
ARCHIVE_NAME = "mot_results_2024.zip"
OUTPUT_NAME = "dvsa_mot_2024.parquet"

_TIMEOUT_SECONDS = 300
_BLOCK_BYTES = 64 << 20

_CATEGORY = pa.dictionary(pa.int32(), pa.string())
#: The columns read, spelled as the pinned files spell them, and their types.
COLUMN_TYPES = {
    "test_id": pa.int64(),
    "vehicle_id": pa.int64(),
    "test_date": pa.date32(),
    "test_class_id": pa.int8(),
    "test_type": _CATEGORY,
    "test_result": _CATEGORY,
    "test_mileage": pa.float64(),
    "postcode_area": _CATEGORY,
    "make": _CATEGORY,
    "model": _CATEGORY,
    "fuel_type": _CATEGORY,
    "cylinder_capacity": pa.float64(),
    "first_use_date": pa.date32(),
}
#: Full names some rows carry in place of the two-letter fuel code.
FUEL_CODES = {"Hybrid Electric (Clean)": "HY", "Electric": "EL"}
KEPT_RESULTS = ("P", "F", "PRS")
FAILED_RESULTS = ("F", "PRS")
UNKNOWN_FIRST_USE = pd.Timestamp("1971-01-01")
EARLIEST_FIRST_USE = pd.Timestamp("1900-01-01")
CATEGORY_COLUMNS = ("fuel_type", "postcode_area", "make", "make_model", "variant")


class FetchError(RuntimeError):
    """The archive could not be fetched or did not match its pin."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify(path: Path) -> None:
    """Delete *path* and raise unless it is the pinned archive.

    Deleting matters for a resumed download: bad bytes left in the partial
    file would be resumed into, and fail, on every later run.
    """
    found = _sha256(path)
    if found != SHA256:
        size = path.stat().st_size
        path.unlink()
        raise FetchError(
            f"{path.name}: sha256 {found[:12]} at {size} bytes does not match the pinned "
            f"{SHA256[:12]} at {SIZE} bytes; deleted it. Re-run to download afresh, and if "
            f"the mismatch repeats the upstream archive changed: re-measure before re-pinning."
        )


def _download(partial: Path, offset: int) -> None:
    """Fetch the archive from byte *offset* onwards into *partial*."""
    request = urllib.request.Request(URL, headers={"Range": f"bytes={offset}-"})
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
            # A server that ignores the range answers 200 with the whole body;
            # appending that to the partial file would corrupt it.
            mode = "ab" if response.status == 206 else "wb"
            with partial.open(mode) as handle:
                shutil.copyfileobj(response, handle, 1 << 20)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise FetchError(f"download interrupted ({exc}); re-run to resume") from exc
    # CPython returns a short body without raising, so the length is checked here.
    size = partial.stat().st_size
    if size != SIZE:
        raise FetchError(f"download stopped at {size} of {SIZE} bytes; re-run to resume")


def obtain(raw_dir: Path) -> Path:
    """Return the verified archive in *raw_dir*, downloading only what is missing.

    The body streams to ``<name>.part`` and a rerun resumes it with a range
    request.  Only a complete, verified file is renamed into place.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / ARCHIVE_NAME
    if archive.exists():
        _verify(archive)
        print(f"{archive}: cached copy matches pin {SHA256[:12]}", flush=True)
        return archive
    partial = raw_dir / f"{ARCHIVE_NAME}.part"
    offset = partial.stat().st_size if partial.exists() else 0
    if offset < SIZE:
        print(f"downloading {URL} from byte {offset}", flush=True)
        _download(partial, offset)
    _verify(partial)
    partial.replace(archive)
    print(f"{archive}: verified sha256 {SHA256[:12]}", flush=True)
    return archive


def read_month(archive: zipfile.ZipFile, member: str) -> pa.Table:
    """Parse the read columns of one monthly CSV, streamed in 64 MB blocks.

    The header tokens double as null markers, so a recurring header line parses
    as a row with a null test_id instead of failing the int64 conversion.
    ``strings_can_be_null=False`` stops those markers nulling a real make or
    model that happened to be spelled like a column name.
    """
    convert = pacsv.ConvertOptions(
        column_types=COLUMN_TYPES,
        include_columns=list(COLUMN_TYPES),
        null_values=["", *COLUMN_TYPES],
        strings_can_be_null=False,
    )
    with archive.open(member) as handle:
        reader = pacsv.open_csv(
            handle,
            read_options=pacsv.ReadOptions(block_size=_BLOCK_BYTES),
            parse_options=pacsv.ParseOptions(escape_char="\\"),
            convert_options=convert,
        )
        return reader.read_all()


def clean_month(table: pa.Table) -> tuple[pd.DataFrame, Counter]:
    """Reduce one month to kept car tests, counting what each filter removed.

    Returns the output columns and a counter whose first entry is the raw row
    count and whose later entries are rows (steps 1-2) or tests (from step 3,
    when rows and tests coincide) removed, in filter order.
    """
    counts = Counter({"raw rows": table.num_rows})
    counts["repeated header lines"] = table["test_id"].null_count
    rows = table.filter(pc.is_valid(table["test_id"])).to_pandas(date_as_object=False)

    # Normalise before comparing repeats, so "HY" against its full name or a
    # blank against a 0 capacity is not mistaken for a disagreement.
    rows["fuel_type"] = rows["fuel_type"].astype(str).replace(FUEL_CODES).astype("category")
    rows["test_mileage"] = rows["test_mileage"].where(rows["test_mileage"] > 0)
    rows["cylinder_capacity"] = rows["cylinder_capacity"].where(rows["cylinder_capacity"] > 0)

    distinct = rows.drop_duplicates()
    n_tests = distinct["test_id"].nunique()
    counts["repeated rows of the same test"] = len(rows) - n_tests
    tests = distinct[~distinct["test_id"].duplicated(keep=False)]
    counts["tests whose repeated rows disagree"] = n_tests - len(tests)

    first_use = tests["first_use_date"]
    drops = {
        "not class 4 (cars)": tests["test_class_id"] != 4,
        "not a normal test (NT)": tests["test_type"] != "NT",
        "result not P, F or PRS": ~tests["test_result"].isin(KEPT_RESULTS),
        "make UNCLASSIFIED": tests["make"] == "UNCLASSIFIED",
        "first use 1971-01-01 (unknown date)": first_use == UNKNOWN_FIRST_USE,
        "first use missing, pre-1900 or after the test": ~first_use.between(
            EARLIEST_FIRST_USE, tests["test_date"]
        ),
    }
    keep = np.ones(len(tests), dtype=bool)
    for label, drop in drops.items():
        removed = keep & drop.to_numpy()
        counts[label] = int(removed.sum())
        keep &= ~removed
    return derive(tests[keep]), counts


def derive(tests: pd.DataFrame) -> pd.DataFrame:
    """Build the output columns from kept tests, one row per test."""
    make = tests["make"].astype(str)
    make_model = make + "|" + tests["model"].astype(str)
    capacity = np.floor(tests["cylinder_capacity"] / 100 + 0.5) * 100
    capacity_band = capacity.astype("Int64").astype("string").fillna("na")
    variant = make_model + "|" + tests["fuel_type"].astype(str) + "|" + capacity_band
    age_days = (tests["test_date"] - tests["first_use_date"]).dt.days
    return pd.DataFrame(
        {
            "fail": tests["test_result"].isin(FAILED_RESULTS).astype("int8"),
            "vehicle_age_years": age_days / 365.25,
            "test_mileage": tests["test_mileage"],
            "cylinder_capacity": tests["cylinder_capacity"],
            "fuel_type": tests["fuel_type"].astype(str).astype("category"),
            "postcode_area": tests["postcode_area"].astype(str).astype("category"),
            "make": make.astype("category"),
            "make_model": make_model.astype("category"),
            "variant": variant.astype(str).astype("category"),
            "vehicle_id": tests["vehicle_id"],
            "test_month": tests["test_date"].dt.month.astype("int8"),
        }
    ).reset_index(drop=True)


def concat_months(months: list[pd.DataFrame]) -> pd.DataFrame:
    """Stack monthly frames, merging each categorical's levels across months.

    A plain ``pd.concat`` turns categoricals with different levels into
    strings, which at this size costs gigabytes.
    """
    frame = pd.concat(
        [month.drop(columns=list(CATEGORY_COLUMNS)) for month in months], ignore_index=True
    )
    for column in CATEGORY_COLUMNS:
        frame[column] = union_categoricals(
            [month[column] for month in months], sort_categories=True
        )
    return frame[months[0].columns]


def filter_table(counts: Counter) -> pd.DataFrame:
    """Tabulate *counts* as rows removed and remaining after each step."""
    steps = pd.Series(counts)
    removed = steps.copy()
    removed.iloc[0] = 0
    return pd.DataFrame({"removed": removed, "remaining": steps.iloc[0] - removed.cumsum()})


def build(archive_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the cleaned tests and the filter table for the pinned archive."""
    months = []
    counts = Counter()
    with zipfile.ZipFile(archive_path) as archive:
        # The archive also carries macOS resource forks named ``._*.csv``.
        members = sorted(
            name
            for name in archive.namelist()
            if name.endswith(".csv") and not name.startswith("__MACOSX/")
        )
        for member in members:
            month, month_counts = clean_month(read_month(archive, member))
            months.append(month)
            counts.update(month_counts)
            print(f"{member}: {len(month):,} tests kept", flush=True)
    return concat_months(months), filter_table(counts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dest", type=Path, default=Path("data"), help="directory to write the parquet to"
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="directory holding the pinned archive (defaults to <dest>/raw)",
    )
    args = parser.parse_args(argv)

    started = time.perf_counter()
    try:
        archive = obtain(args.raw_dir or args.dest / "raw")
    except FetchError as exc:
        print(f"DVSA MOT fetch failed: {exc}", file=sys.stderr)
        return 1
    frame, filters = build(archive)
    print(filters.to_string(formatters={c: "{:,}".format for c in filters.columns}))

    args.dest.mkdir(parents=True, exist_ok=True)
    out = args.dest / OUTPUT_NAME
    # Written aside and moved into place, so an interrupted write never leaves
    # a truncated parquet under the real name.
    tmp = args.dest / f".{OUTPUT_NAME}.partial"
    try:
        frame.to_parquet(tmp, engine="pyarrow", index=False)
        tmp.replace(out)
    finally:
        tmp.unlink(missing_ok=True)

    levels = ", ".join(f"{c} {frame[c].cat.categories.size:,}" for c in CATEGORY_COLUMNS)
    print(
        f"wrote {out}: {len(frame):,} tests, {out.stat().st_size / 1e6:.1f} MB; "
        f"fail rate {frame['fail'].mean():.4f}; levels: {levels}; "
        f"{time.perf_counter() - started:.0f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
