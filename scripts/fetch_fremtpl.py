"""Fetch the public freMTPL2 parquet artifacts the real-data suites need.

The suites in ``tests/test_realdata_parity.py``,
``tests/test_screening_guide_numbers.py`` and
``tests/test_mixed_interaction_screening.py`` are anchored to freMTPL2, which
is gitignored.  Without a copy they skip, and a skip reads identically to a
pass in a CI summary -- that is how the guide anchor drifted out of tolerance
on 2026-08-07 and went unreported for four days (#261).

Both datasets are CC0 on OpenML (ids 41214 / 41215), mirrored as parquet.  The
upstream is C. Dutang and A. Charpentier (2018), *CASdatasets: Insurance
datasets*, distributed with A. Charpentier (ed.), *Computational Actuarial
Science with R*, CRC 2018.  We read only the published data, never that
package's source.

Every download is pinned by SHA-256, so this script answers "is this the
artifact we measured against?" rather than "did some bytes arrive?".  A
mismatch is a hard error: silently proceeding on a re-encoded upstream would
re-pin the suites' anchors to data nobody checked.

Measured 2026-08-11 against a developer's copy of ``freMTPL2freq.parquet``,
because a pinned fetch is worthless if it feeds the suites different numbers
from the copy the anchors were taken on.  Same 678,013 rows in the same order;
after the normalisation below, the same dtypes.  Two columns differ before the
suites touch them, and neither reaches a result:

* ``Exposure`` disagrees on 6,877 rows (1.01%), by at most 2.6e-14 relative and
  161 ulp -- a text export carrying 14 significant digits against 15.  *Every*
  one of those rows is below 0.01, and all three suites clip at 0.01, so after
  that clip the two copies are bit-identical.
* ``VehGas`` carries literal ARFF quotes in the older copy.  No real-data test
  reads the column; :func:`normalise` strips them regardless.

The guide's anchors then agree exactly: ``sum_sample_weight``,
``mains_deviance``, ``mains_edf`` 52.867789482508, ``phi`` and the Pearson sum
all reproduce at 0.000e+00 relative difference, against a 1e-5 test tolerance.

Usage::

    python scripts/fetch_fremtpl.py --dest data/
    python scripts/fetch_fremtpl.py --cache-key   # for actions/cache

Exit status is 0 only when every artifact is present, verified and normalised.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

#: Bump when the normalisation below changes, so a cache from the previous
#: encoding cannot be reused.  The raw downloads are pinned by hash and so
#: cannot drift, but the schema we derive from them can.
SCHEMA_REVISION = 1

_RETRY_DELAYS = (2, 5, 15, 30)
_TIMEOUT_SECONDS = 300


class FetchError(RuntimeError):
    """A dataset could not be fetched or did not match its pin.

    Deliberately distinct from anything a test raises: the caller turns this
    into a CI step whose failure is unmistakably the download, not the suite.
    """


@dataclass(frozen=True)
class Artifact:
    """One pinned upstream parquet and the schema we normalise it to."""

    #: Filename ``tests/_datasets.py`` looks for.
    name: str
    openml_id: int
    url: str
    #: SHA-256 of the bytes at ``url``, measured 2026-08-11.
    sha256: str
    rows: int
    #: Full expected column order after normalisation.
    columns: tuple[str, ...]
    #: Columns the mirror stores narrower than a developer's copy (``uint8``,
    #: or ``float64`` for an integer count).  Cast so a discrete-binning or
    #: dtype-dispatched path cannot see a different type in CI than locally.
    integer_columns: tuple[str, ...] = ()
    #: Columns an ARFF round-trip can leave wrapped in literal quotes.
    string_columns: tuple[str, ...] = ()


ARTIFACTS: tuple[Artifact, ...] = (
    Artifact(
        name="freMTPL2freq.parquet",
        openml_id=41214,
        url="https://data.openml.org/datasets/0004/41214/dataset_41214.pq",
        sha256="aead80a9ac68baf2c78fc1beaa287441d88d06cc11be60a1f226784d164c6dd7",
        rows=678013,
        columns=(
            "IDpol",
            "ClaimNb",
            "Exposure",
            "Area",
            "VehPower",
            "VehAge",
            "DrivAge",
            "BonusMalus",
            "VehBrand",
            "VehGas",
            "Density",
            "Region",
        ),
        integer_columns=("ClaimNb", "VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"),
        string_columns=("VehGas",),
    ),
    Artifact(
        name="freMTPL2sev.parquet",
        openml_id=41215,
        url="https://data.openml.org/datasets/0004/41215/dataset_41215.pq",
        sha256="c721d570c42eeaf4a70cc12f4c1e04095a6046f6cdbc01fad919274879783e60",
        rows=26639,
        columns=("IDpol", "ClaimAmount"),
    ),
)


def cache_key() -> str:
    """A key that changes exactly when the pinned bytes or the schema change.

    The pins live only in this file, so the workflow never restates a hash it
    could drift from.
    """
    digest = hashlib.sha256()
    digest.update(f"schema{SCHEMA_REVISION}".encode())
    for art in ARTIFACTS:
        digest.update(art.sha256.encode())
    return f"fremtpl-raw-{digest.hexdigest()[:16]}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(art: Artifact, target: Path) -> None:
    """Download *art* to *target*, retrying only what a retry can fix.

    The body streams to a sibling temporary file and is moved into place only
    once it is complete, so *target* is either absent or whole.  That matters
    because the raw directory is what ``actions/cache`` stores: a half-written
    file cached under a key that claims a verified hash would be restored, and
    trusted, on every later run.
    """
    last: Exception | None = None
    for attempt, delay in enumerate((*_RETRY_DELAYS, None)):
        partial: Path | None = None
        try:
            with urllib.request.urlopen(art.url, timeout=_TIMEOUT_SECONDS) as response:
                with tempfile.NamedTemporaryFile(
                    dir=target.parent, delete=False, suffix=".part"
                ) as tmp:
                    partial = Path(tmp.name)
                    shutil.copyfileobj(response, tmp)
            partial.replace(target)
            return
        except urllib.error.HTTPError as exc:
            # A 404 or a 403 will still be a 404 or a 403 in thirty seconds.
            # Only 429 and the 5xx family are worth waiting out.
            if exc.code < 500 and exc.code != 429:
                raise FetchError(f"{art.name}: {art.url} returned HTTP {exc.code}") from exc
            last = exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = exc
        finally:
            if partial is not None:
                partial.unlink(missing_ok=True)
        if delay is None:
            break
        print(f"  attempt {attempt + 1} failed ({last}); retrying in {delay}s", flush=True)
        time.sleep(delay)
    raise FetchError(f"{art.name}: could not download {art.url}: {last}")


def obtain(art: Artifact, raw_dir: Path) -> Path:
    """Return a verified raw copy of *art*, downloading only if needed.

    A cached file that fails its pin is discarded rather than trusted: a
    truncated write is exactly what a cache restore can hand back.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw = raw_dir / f"dataset_{art.openml_id}.pq"
    if raw.exists():
        found = _sha256(raw)
        if found == art.sha256:
            print(f"{art.name}: cached copy matches pin {art.sha256[:12]}", flush=True)
            return raw
        print(f"{art.name}: cached copy has sha256 {found[:12]}, re-downloading", flush=True)
        raw.unlink()
    print(f"{art.name}: downloading {art.url}", flush=True)
    _download(art, raw)
    found = _sha256(raw)
    if found != art.sha256:
        raw.unlink(missing_ok=True)
        raise FetchError(
            f"{art.name}: sha256 {found} does not match the pinned {art.sha256}. "
            f"The upstream artifact changed; re-pin deliberately after re-measuring "
            f"the suites' anchors against it."
        )
    print(f"{art.name}: verified sha256 {found[:12]}", flush=True)
    return raw


def normalise(art: Artifact, raw: Path, dest: Path) -> Path:
    """Write *raw* to *dest* in the schema a developer's local copy carries.

    The mirror stores the counts as ``uint8`` and ``Density`` as ``float64``,
    where a copy taken from the CASdatasets release carries ``int64``; and an
    ARFF round-trip can leave string levels wrapped in literal quotes.  Neither
    difference moves a number, but both would make CI and a local run disagree
    about dtypes, and a dtype is exactly what a discrete or categorical path
    dispatches on.
    """
    import pandas as pd

    frame = pd.read_parquet(raw)
    if tuple(frame.columns) != art.columns:
        raise FetchError(f"{art.name}: columns {tuple(frame.columns)} != expected {art.columns}")
    if len(frame) != art.rows:
        raise FetchError(f"{art.name}: {len(frame)} rows, expected {art.rows}")
    for column in art.integer_columns:
        frame[column] = frame[column].astype("int64")
    for column in art.string_columns:
        frame[column] = frame[column].astype(str).str.strip("'")
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / art.name
    frame.to_parquet(out, index=False)
    print(f"{art.name}: wrote {out} ({len(frame)} rows)", flush=True)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("data"),
        help="directory to write the normalised parquet files to",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="directory holding the pinned raw downloads (defaults to <dest>/../.fremtpl-raw)",
    )
    parser.add_argument(
        "--cache-key",
        action="store_true",
        help="print the actions/cache key implied by the pins and exit",
    )
    args = parser.parse_args(argv)

    if args.cache_key:
        print(cache_key())
        return 0

    raw_dir = args.raw_dir or args.dest.parent / ".fremtpl-raw"
    try:
        for art in ARTIFACTS:
            normalise(art, obtain(art, raw_dir), args.dest)
    except FetchError as exc:
        print(f"::error title=freMTPL2 fetch failed::{exc}", file=sys.stderr)
        return 1
    print(f"all {len(ARTIFACTS)} artifacts ready in {args.dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
