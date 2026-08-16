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

``freMTPL2sev.parquet`` gets the same statement, because a pin without a schema
contract is only half the guarantee and the asymmetry read as an oversight.
There is no older developer copy of it to compare against -- the pair the suites
ran on IS the fetched pair, both raw hashes matching the pins above -- so what
is recorded is what was measured on the mirror: 26,639 rows, ``IDpol`` and
``ClaimAmount`` both ``float64``, nothing to cast.  ``IDpol`` is the key
``tests/test_realdata_parity._load_gamma_data`` joins the two files on, and it
is ``float64`` on the freq side too, so that join is same-dtype and lossless.
:data:`Artifact.float_columns` now asserts it on both sides rather than leaving
it as a property of two separate silences.

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
#:
#: Deliberate belt and braces, and it costs something, so it is written down
#: rather than left to look like an oversight: the cache holds RAW bytes, which
#: this revision provably cannot change, so bumping it forces one re-download of
#: bytes that are identical.  That is ~2s once.  The alternative -- a key that
#: cannot express a normalisation change -- is a cache that silently survives
#: one, and the workflow header would then be describing a guarantee the key
#: does not give.  Cheap insurance beats a correct-today comment.
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
    #: Byte length of the bytes at ``url``.  Pinned beside the hash because the
    #: two answer different questions: a hash says "not the artifact", a length
    #: says WHY not.  See :func:`obtain`.
    size: int
    rows: int
    #: Full expected column order after normalisation.
    columns: tuple[str, ...]
    #: Columns the mirror stores narrower than a developer's copy (``uint8``,
    #: or ``float64`` for an integer count).  Cast so a discrete-binning or
    #: dtype-dispatched path cannot see a different type in CI than locally.
    integer_columns: tuple[str, ...] = ()
    #: Columns an ARFF round-trip can leave wrapped in literal quotes.
    string_columns: tuple[str, ...] = ()
    #: Columns that must still be ``float64`` after normalisation.  ASSERTED,
    #: never cast: a cast would hide the change this exists to report.  This is
    #: what makes the sev artifact carry a schema contract and not only a pin --
    #: ``IDpol`` is the key ``test_realdata_parity`` joins the two files on, and
    #: it appears here on both sides, so "the join key agrees" is one line to
    #: read rather than a property of two separate absences.
    float_columns: tuple[str, ...] = ()


ARTIFACTS: tuple[Artifact, ...] = (
    Artifact(
        name="freMTPL2freq.parquet",
        openml_id=41214,
        url="https://data.openml.org/datasets/0004/41214/dataset_41214.pq",
        sha256="aead80a9ac68baf2c78fc1beaa287441d88d06cc11be60a1f226784d164c6dd7",
        size=7469711,
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
        float_columns=("IDpol", "Exposure"),
    ),
    Artifact(
        name="freMTPL2sev.parquet",
        openml_id=41215,
        url="https://data.openml.org/datasets/0004/41215/dataset_41215.pq",
        sha256="c721d570c42eeaf4a70cc12f4c1e04095a6046f6cdbc01fad919274879783e60",
        size=277195,
        rows=26639,
        columns=("IDpol", "ClaimAmount"),
        # Measured, not assumed: the mirror stores both of these as float64
        # already, so there is nothing here to CAST -- which is why this
        # artifact carried no schema contract at all and freq carried one.
        # Casting ``IDpol`` to int64 "for symmetry" would be actively wrong: it
        # would manufacture the int64-against-float64 join that
        # ``_load_gamma_data`` does not currently do.
        float_columns=("IDpol", "ClaimAmount"),
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


def _diagnose_pin_miss(art: Artifact, raw: Path, found: str) -> FetchError:
    """Say WHICH failure a freshly downloaded, wrongly hashed file is.

    Two very different things reach this point, and they need opposite actions:
    a damaged transfer (re-run) and a re-encoded upstream (re-measure, then
    re-pin).  Reporting the first as the second is the worst outcome this script
    has, because the advice is "re-pin" and the bytes are corrupt.

    A damaged transfer is not exotic and does not raise.  ``_download``'s retry
    loop only covers *raised* transport errors, and CPython does not raise on a
    short body: ``http.client.HTTPResponse.read(amt)`` returns short and closes
    the connection instead, carrying the comment "Ideally, we would raise
    IncompleteRead if the content-length wasn't satisfied, but it might break
    compatibility".  ``shutil.copyfileobj`` reads with an ``amt``, so a proxy or
    a mid-body close lands a complete-LOOKING file.  Measured: it reached the
    re-pin message.

    Two discriminators, because neither alone is complete:

    * a second, independent download -- decides same-length corruption, which a
      length check cannot see.  Two bodies that disagree with each other cannot
      both be upstream, so that is transport, whatever the length says;
    * the pinned byte length -- separates a re-encode this script may advise
      re-pinning against from one it may not.  A proxy truncating at the same
      offset every time survives a repeat download, so agreement alone does not
      make bytes upstream.

    The second download runs FIRST, and the length only LABELS the answer.  The
    other way round -- which this did -- the length decided, and a real
    re-encode almost always changes the length, so the realistic re-encode
    landed in a branch asserting as fact that it was not one, never took the
    second download that could have told the difference, and repeated that on
    every rerun.  Measured on the unfixed code, a re-encoded body against the
    real freq pin: ONE download taken, and "A short or overlong body is a
    DAMAGED TRANSFER ... Do not re-pin against these bytes."  The documented
    recovery was reachable only by a re-encode that preserved the byte length
    exactly, and the workflow routes that wording to "re-run this job".  The
    suite shared the blind spot by construction: its only re-encode case set
    the PIN's length to the re-encoded length.

    So three outcomes, not two, and the third is named rather than guessed at:
    two agreeing downloads at an unpinned length are genuinely ambiguous
    between a deterministic truncation and a re-encode, and nothing available
    here separates them.  It says so, and sends the operator to the artifact's
    published length instead of to a re-pin.
    """
    size = raw.stat().st_size
    raw.unlink(missing_ok=True)
    second = raw.parent / f"{raw.name}.recheck"
    try:
        _download(art, second)
        again = _sha256(second)
        again_size = second.stat().st_size
    finally:
        second.unlink(missing_ok=True)
    if again != found:
        return FetchError(
            f"{art.name}: two downloads disagreed ({found[:12]} then {again[:12]}, at "
            f"{size} then {again_size} bytes against the pinned {art.size}). Bodies that "
            f"disagree with each other cannot both be upstream: this is a DAMAGED TRANSFER, "
            f"corrupted in transit -- re-run the fetch. Do not re-pin against these bytes."
        )
    if size == art.size:
        return FetchError(
            f"{art.name}: sha256 {found} does not match the pinned {art.sha256}, and two "
            f"independent downloads agree on it at the pinned length. The upstream artifact "
            f"changed; re-pin deliberately after re-measuring the suites' anchors against it."
        )
    return FetchError(
        f"{art.name}: sha256 {found[:12]} does not match the pinned {art.sha256[:12]}, and "
        f"both downloads agree on it at {size} bytes rather than the pinned {art.size}. "
        f"That is AMBIGUOUS and is not routed either way: a proxy truncating at the same "
        f"offset every time is indistinguishable from a re-encoded upstream from this end. "
        f"Check the artifact's published byte length at {art.url} before touching anything "
        f"-- only if upstream really is {size} bytes may you re-measure the suites' anchors "
        f"and re-pin."
    )


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
        raise _diagnose_pin_miss(art, raw, found)
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
    for column in art.float_columns:
        if frame[column].dtype != "float64":
            raise FetchError(
                f"{art.name}: {column} is {frame[column].dtype} after normalisation, expected "
                f"float64. The raw bytes are pinned, so this is the reader disagreeing with "
                f"the copy the suites' anchors were measured on; a dtype is what a discrete "
                f"or categorical path dispatches on."
            )
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / art.name
    # Written aside and moved into place, like the raw download and for the same
    # reason: ``tests/_datasets.usable`` accepts a path on existence alone, so a
    # write interrupted here leaves a truncated parquet that the next run does
    # NOT re-fetch and that raises from inside a test body instead.
    tmp = dest / f".{art.name}.partial"
    try:
        frame.to_parquet(tmp, index=False)
        tmp.replace(out)
    finally:
        tmp.unlink(missing_ok=True)
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
