"""The fetch of the two Spanish books the held-out smoothing guide is drawn from.

Nothing here touches the network or the real files: every download is
stubbed, and the conversion runs on spreadsheets built in the test.
"""

from __future__ import annotations

import dataclasses
import hashlib
import zipfile

import openpyxl
import pandas as pd
import pytest
from scripts import fetch_spanish_books as fetch
from scripts.fetch_fremtpl import FetchError

HEALTH, MOTOR = fetch.BOOKS


def _stub_download(monkeypatch, payload: bytes, log: list[str] | None = None):
    """Every download, the first and any second opinion, writes *payload*."""

    def fake(book, target):
        if log is not None:
            log.append(book.name)
        target.write_bytes(payload)

    monkeypatch.setattr("scripts.fetch_fremtpl._download", fake)


def _pinned_to(book: fetch.Book, payload: bytes) -> fetch.Book:
    return dataclasses.replace(book, sha256=hashlib.sha256(payload).hexdigest(), size=len(payload))


def test_every_book_is_pinned_to_a_mendeley_file_and_credited():
    for book, record in ((HEALTH, "386vmj2tbk"), (MOTOR, "sw4jmdb2sm")):
        assert book.url.startswith(
            f"https://data.mendeley.com/public-files/datasets/{record}/files/"
        )
        assert len(book.sha256) == 64 and int(book.sha256, 16) >= 0
        assert f"doi:10.17632/{record}" in fetch.__doc__
    assert "CC BY 4.0" in fetch.__doc__
    assert "http://creativecommons.org/licenses/by/4.0" in fetch.__doc__


def test_a_download_that_misses_its_pin_is_refused_and_removed(tmp_path, monkeypatch):
    wrong = b"not the pinned spreadsheet"
    _stub_download(monkeypatch, wrong)
    book = dataclasses.replace(HEALTH, size=len(wrong))
    with pytest.raises(FetchError, match="does not match the pinned"):
        fetch.obtain(book, tmp_path)
    assert list(tmp_path.iterdir()) == [], (
        "an unverified download was left where the next run would read it"
    )


def test_a_failed_download_leaves_nothing_under_the_real_name(tmp_path, monkeypatch):
    def fail(book, target):
        raise FetchError(f"{book.name}: could not download")

    monkeypatch.setattr("scripts.fetch_fremtpl._download", fail)
    with pytest.raises(FetchError, match="could not download"):
        fetch.obtain(MOTOR, tmp_path)
    assert not (tmp_path / MOTOR.raw_name).exists()


def test_a_cached_copy_that_fails_its_pin_is_downloaded_again(tmp_path, monkeypatch):
    good = b"the published bytes"
    book = _pinned_to(MOTOR, good)
    (tmp_path / book.raw_name).write_bytes(b"truncated")
    log: list[str] = []
    _stub_download(monkeypatch, good, log)
    assert fetch.obtain(book, tmp_path).read_bytes() == good
    assert log == [book.name]


def test_a_cached_copy_that_matches_its_pin_is_used(tmp_path, monkeypatch):
    good = b"the published bytes"
    book = _pinned_to(MOTOR, good)
    (tmp_path / book.raw_name).write_bytes(good)
    log: list[str] = []
    _stub_download(monkeypatch, b"should not be read", log)
    assert fetch.obtain(book, tmp_path).read_bytes() == good
    assert log == []


def _openpyxl_spreadsheet(path, rows):
    """A spreadsheet as openpyxl writes it: text as inline strings."""
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    for row in rows:
        sheet.append(row)
    workbook.save(path)
    return path


def _shared_strings_spreadsheet(path, rows):
    """A spreadsheet as the published file is written: text in a shared-string table."""
    ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    strings: list[str] = []
    sheet_rows = []
    for r, row in enumerate(rows, start=1):
        cells = []
        for c, value in enumerate(row):
            ref = f"{chr(65 + c)}{r}"
            if value is None:
                continue
            if isinstance(value, str):
                strings.append(value)
                cells.append(f'<c r="{ref}" t="s"><v>{len(strings) - 1}</v></c>')
            else:
                cells.append(f'<c r="{ref}"><v>{value!r}</v></c>')
        sheet_rows.append(f'<row r="{r}">{"".join(cells)}</row>')
    shared = "".join(f"<si><t>{text}</t></si>" for text in strings)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("xl/sharedStrings.xml", f'<sst xmlns="{ns}">{shared}</sst>')
        archive.writestr(
            "xl/worksheets/sheet1.xml",
            f'<worksheet xmlns="{ns}"><sheetData>{"".join(sheet_rows)}</sheetData></worksheet>',
        )
    return path


@pytest.mark.parametrize("write", [_shared_strings_spreadsheet, _openpyxl_spreadsheet])
def test_the_spreadsheet_is_converted_cell_for_cell(tmp_path, write):
    raw = write(
        tmp_path / "book.xlsx",
        [
            ["ID", "ID_policy", "exposure_time", "gender", "C_H", "date_effect"],
            ["1_1", 1, 0.96974885844748848, "M", "H1", 34135],
            ["2_1", 2, 1, "F", None, 34200],
            ["3_1", 2, 0.5, "F", "H2", 34201],
        ],
    )
    book = dataclasses.replace(HEALTH, rows=3, columns=6)
    written = pd.read_parquet(fetch.convert(book, raw, tmp_path / "out"))
    expected = pd.DataFrame(
        {
            "ID": ["1_1", "2_1", "3_1"],
            "ID_policy": [1, 2, 2],
            "exposure_time": [0.96974885844748848, 1.0, 0.5],
            "gender": ["M", "F", "F"],
            "C_H": ["H1", None, "H2"],
            "date_effect": [34135, 34200, 34201],
        }
    )
    pd.testing.assert_frame_equal(written, expected, check_dtype=False)
    assert written["ID_policy"].dtype == "int64"
    assert written["exposure_time"].dtype == "float64"


def test_the_motor_text_is_read_with_its_separator_and_missing_value(tmp_path):
    raw = tmp_path / "motor.csv"
    raw.write_text(
        '"insured_id";"year";"fuel_type";"vehicle_value"\n1;2022;"G";149511.4425\n2;2022;"";NA\n',
        encoding="utf-8",
    )
    book = dataclasses.replace(MOTOR, rows=2, columns=4)
    written = pd.read_parquet(fetch.convert(book, raw, tmp_path / "out"))
    assert written["vehicle_value"].isna().tolist() == [False, True]
    assert written["fuel_type"].tolist() == ["G", ""]


def test_a_file_of_the_wrong_shape_is_refused_and_nothing_is_written(tmp_path):
    raw = _shared_strings_spreadsheet(tmp_path / "book.xlsx", [["a", "b"], [1, 2]])
    book = dataclasses.replace(HEALTH, rows=228711, columns=42)
    with pytest.raises(FetchError, match="1 rows by 2 columns, expected 228711 by 42"):
        fetch.convert(book, raw, tmp_path / "out")
    assert not (tmp_path / "out").exists() or list((tmp_path / "out").iterdir()) == []
