"""Fetch the two public Spanish books the held-out smoothing guide is drawn from.

The guide ``docs/how-to/choose-smoothing-by-holdout.md`` and its figure scripts
``docs/scripts/ncv_health_*.py`` and ``docs/scripts/ncv_motor_agree.py`` read
two books published on Mendeley Data under CC BY 4.0.  Nothing is
redistributed: this script downloads each file from its record, checks it
against the SHA-256 pinned below (the hash Mendeley publishes on the record),
and writes a parquet copy to ``data/`` (git-ignored) for the figure scripts.

Spanish health insurance portfolio
    Lledó, J., Espinosa Adamez, P. and Perez Gimenez, V. (2025). *Dataset of
    health insurance portfolio*, version 4. Mendeley Data,
    doi:10.17632/386vmj2tbk.4, https://data.mendeley.com/datasets/386vmj2tbk/4.
    One row per insured person and calendar year, 2017 to 2019: 228,711 rows
    by 42 columns.

Spanish motor book by coverage
    Espinosa, P., Lledó, J. and Atance, D. (2026). *A detailed dataset of
    motor insurance policies with coverage-specific financial information*,
    version 1. Mendeley Data, doi:10.17632/sw4jmdb2sm.1,
    https://data.mendeley.com/datasets/sw4jmdb2sm/1.  One row per insured and
    year, 2022 to 2024: 354,140 rows by 47 columns.

Licence of both, as stated on each record: CC BY 4.0,
http://creativecommons.org/licenses/by/4.0 -- "You can share, copy and modify
this dataset so long as you give appropriate credit, provide a link to the CC
BY license, and indicate if changes were made, but you may not do so in a way
that suggests the rights holder has endorsed you or your use of the dataset.
Note that further permission may be required for any content within the
dataset that is identified as belonging to a third party."

Changes made.  This script changes the format only: the health spreadsheet's
first sheet is read cell by cell as stored (shared strings as text, numbers as
written, dates as the spreadsheet's serial day numbers) and the motor file is
read as ``;``-separated text with ``NA`` as the missing value; both are
written to parquet with every row and column.  The figure scripts then remove
rows before fitting (``docs/scripts/spanish_books.py`` states which and why):
from the health book the rows with zero exposure and the dental-only product,
and from the motor book the rows with zero exposure or without a vehicle value,
vehicle age or licence age; they also cap ages at fixed limits.

Usage::

    python scripts/fetch_spanish_books.py --dest data/

Exit status is 0 only when both books are present and verified.
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.etree.ElementTree import iterparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.fetch_fremtpl as fremtpl  # noqa: E402
from scripts.fetch_fremtpl import FetchError  # noqa: E402

_SHEET_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
#: Mendeley answers Python's default User-Agent with HTTP 403, so downloads name a generic client.
USER_AGENT = "superglm-fetch-spanish-books"


@dataclass(frozen=True)
class Book:
    """One pinned upstream file and the parquet copy written from it."""

    #: Name of the parquet copy under ``--dest``.
    name: str
    #: Name of the verified raw download under the raw directory.
    raw_name: str
    url: str
    #: SHA-256 of the bytes at ``url``, as published on the Mendeley record.
    sha256: str
    #: Byte length of the bytes at ``url``; reported when the hash misses.
    size: int
    rows: int
    columns: int
    #: ``"xlsx"`` (first sheet) or ``"csv;"`` (``;``-separated text).
    form: str


BOOKS: tuple[Book, ...] = (
    Book(
        name="spanish_health.parquet",
        raw_name="Dataset of health insurance portfolio.xlsx",
        url=(
            "https://data.mendeley.com/public-files/datasets/386vmj2tbk/files/"
            "621bbef0-6ef0-4683-9427-394c2ca4fe7c/file_downloaded"
        ),
        sha256="2bd9ffced5198a48a2d9fc1a19ad748b20c7c038896ba627a147412cec03836b",
        size=45451594,
        rows=228711,
        columns=42,
        form="xlsx",
    ),
    Book(
        name="spanish_motor_coverage.parquet",
        raw_name="Dataset of motor insurance portfolio.csv",
        url=(
            "https://data.mendeley.com/public-files/datasets/sw4jmdb2sm/files/"
            "3dc64675-b6bd-4e8e-bfba-b64e585c6599/file_downloaded"
        ),
        sha256="6a47d19d5278a049ea0aeaf39c955cc26068639bdc58cb4523b201e740f0faf4",
        size=94710312,
        rows=354140,
        columns=47,
        form="csv;",
    ),
)


def obtain(book: Book, raw_dir: Path) -> Path:
    """Return a verified raw copy of *book*, downloading only if needed.

    A copy that fails its pin is discarded, never trusted; a download that
    fails its pin is removed and the miss is diagnosed as in
    ``scripts/fetch_fremtpl.py`` (a damaged transfer, a changed upstream, or
    the ambiguous case between them).
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw = raw_dir / book.raw_name
    if raw.exists():
        found = fremtpl._sha256(raw)
        if found == book.sha256:
            print(f"{book.name}: cached copy matches pin {book.sha256[:12]}", flush=True)
            return raw
        print(f"{book.name}: cached copy has sha256 {found[:12]}, re-downloading", flush=True)
        raw.unlink()
    print(f"{book.name}: downloading {book.url}", flush=True)
    fremtpl._download(book, raw)
    found = fremtpl._sha256(raw)
    if found != book.sha256:
        raise fremtpl._diagnose_pin_miss(book, raw, found)
    print(f"{book.name}: verified sha256 {found[:12]}", flush=True)
    return raw


def _column_index(ref: str) -> int:
    """Zero-based column of a cell reference such as ``"AB12"``."""
    index = 0
    for letter in re.match(r"[A-Z]+", ref).group():
        index = index * 26 + ord(letter) - 64
    return index - 1


def xlsx_first_sheet_as_csv(path: Path) -> str:
    """The first sheet of an .xlsx file as CSV text, cells as stored.

    Streams the sheet's XML with the standard library: shared strings become
    their text, inline strings their text, and every other cell its stored
    value, so numbers keep the digits written and dates stay serial numbers.
    A row is as wide as the header row; a missing cell is an empty field.
    """
    with zipfile.ZipFile(path) as archive:
        strings: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            with archive.open("xl/sharedStrings.xml") as handle:
                for _, element in iterparse(handle):
                    if element.tag == _SHEET_NS + "si":
                        strings.append("".join(t.text or "" for t in element.iter(_SHEET_NS + "t")))
                        element.clear()
        out = io.StringIO()
        writer = csv.writer(out, lineterminator="\n")
        width = None
        with archive.open("xl/worksheets/sheet1.xml") as handle:
            for _, element in iterparse(handle):
                if element.tag != _SHEET_NS + "row":
                    continue
                cells = {}
                for cell in element:
                    value = cell.find(_SHEET_NS + "v")
                    if cell.get("t") == "s":
                        text = strings[int(value.text)]
                    elif cell.get("t") == "inlineStr":
                        text = "".join(t.text or "" for t in cell.iter(_SHEET_NS + "t"))
                    else:
                        text = "" if value is None or value.text is None else value.text
                    cells[_column_index(cell.get("r"))] = text
                width = width or max(cells) + 1
                writer.writerow([cells.get(i, "") for i in range(width)])
                element.clear()
    return out.getvalue()


def read_raw(book: Book, raw: Path):
    """The raw file as a data frame, every row and column, types inferred by pandas."""
    import pandas as pd

    if book.form == "xlsx":
        return pd.read_csv(io.StringIO(xlsx_first_sheet_as_csv(raw)), low_memory=False)
    return pd.read_csv(raw, sep=";", na_values=["NA"], keep_default_na=False, low_memory=False)


def convert(book: Book, raw: Path, dest: Path) -> Path:
    """Write *raw* to ``dest / book.name`` as parquet, after checking its shape.

    The copy is written aside and moved into place, so the name the figure
    scripts read is either absent or whole.
    """
    frame = read_raw(book, raw)
    if frame.shape != (book.rows, book.columns):
        raise FetchError(
            f"{book.name}: {frame.shape[0]} rows by {frame.shape[1]} columns, expected "
            f"{book.rows} by {book.columns}"
        )
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / book.name
    tmp = dest / f".{book.name}.partial"
    try:
        frame.to_parquet(tmp, index=False)
        tmp.replace(out)
    finally:
        tmp.unlink(missing_ok=True)
    print(f"{book.name}: wrote {out} ({len(frame)} rows)", flush=True)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dest", type=Path, default=Path("data"), help="directory for the parquet copies"
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="directory holding the verified raw downloads (defaults to <dest>/.spanish-raw)",
    )
    args = parser.parse_args(argv)
    raw_dir = args.raw_dir or args.dest / ".spanish-raw"
    opener = urllib.request.build_opener()
    opener.addheaders = [("User-Agent", USER_AGENT)]
    urllib.request.install_opener(opener)
    try:
        for book in BOOKS:
            convert(book, obtain(book, raw_dir), args.dest)
    except FetchError as exc:
        print(f"Spanish books fetch failed: {exc}", file=sys.stderr)
        return 1
    print(f"both books ready in {args.dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
