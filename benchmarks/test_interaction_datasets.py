"""Data-integrity boundaries for the research corpus, without network access."""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import zipfile
from pathlib import Path

import interaction_datasets as corpus
import pandas as pd
import pytest


def test_committed_registry_matches_the_frozen_corpus_receipt():
    root = Path(__file__).resolve().parents[1]
    receipt = json.loads(
        (root / "notes/research/2026-09-13-interaction-dataset-corpus-receipt.json").read_text()
    )
    assert hashlib.sha256(corpus.MANIFEST.read_bytes()).hexdigest() == receipt["manifest_sha256"]


def entry_for(raw=b"x,y\n1,2\n3,4\n"):
    return {
        "id": "example",
        "availability": "fetchable",
        "primary_target": "y",
        "features": ["x"],
        "categorical_columns": [],
        "exclude_columns": {},
        "split": {"strategy": "random", "columns": []},
        "source": {
            "url": "https://example.org/data.csv",
            "filename": "source.csv",
            "format": "csv",
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
            "max_bytes": 1024,
        },
        "schema": {"rows": 2, "columns": ["x", "y"], "missing_counts": {}},
        "target_rule": {"minimum": 0, "integer": True},
    }


def write_source(root, entry, raw):
    path = root / entry["id"] / entry["source"]["filename"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return path


def test_load_requires_pinned_bytes_and_preserves_rows(tmp_path):
    raw = b"x,y\n1,2\n3,4\n"
    entry = entry_for(raw)
    write_source(tmp_path, entry, raw)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "datasets": [entry]}))
    frame = corpus.load_dataset("example", root=tmp_path, manifest=manifest)
    pd.testing.assert_frame_equal(frame, pd.DataFrame({"x": [1, 3], "y": [2, 4]}))


@pytest.mark.parametrize("corrupt", [b"incomplete", b"x,y\n1,9\n3,4\n"])
def test_corrupt_existing_data_is_preserved_and_never_refetched(tmp_path, monkeypatch, corrupt):
    entry = entry_for()
    path = write_source(tmp_path, entry, corrupt)

    def unexpected_network(*args, **kwargs):
        pytest.fail("A corrupt existing file must not trigger replacement")

    monkeypatch.setattr(corpus.urllib.request, "urlopen", unexpected_network)
    with pytest.raises(ValueError, match="bytes|SHA256"):
        corpus.fetch_one(entry, tmp_path)
    assert path.read_bytes() == corrupt


@pytest.mark.parametrize(
    "field,value,match", [("rows", 3, "rows"), ("columns", ["z", "y"], "columns")]
)
def test_shape_contract_catches_an_incomplete_or_wrong_table(tmp_path, field, value, match):
    raw = b"x,y\n1,2\n3,4\n"
    entry = entry_for(raw)
    entry["schema"][field] = value
    write_source(tmp_path, entry, raw)
    with pytest.raises(ValueError, match=match):
        corpus.fetch_one(entry, tmp_path, allow_download=False)


@pytest.mark.parametrize("raw", [b"x,y\n1,\n3,4\n", b"x,y\n1,2.5\n3,4\n", b"x,y\n1,inf\n3,4\n"])
def test_target_contract_refuses_missing_fractional_or_infinite_counts(tmp_path, raw):
    entry = entry_for(raw)
    if b"1,\n" in raw:
        entry["schema"]["missing_counts"] = {"y": 1}
    write_source(tmp_path, entry, raw)
    with pytest.raises(ValueError, match="target"):
        corpus.fetch_one(entry, tmp_path, allow_download=False)


class Response(io.BytesIO):
    headers = {}
    url = "https://example.org/data.csv"


@pytest.mark.parametrize("raw,limit", [(b"too long for the cap", 4), (b"x,y\n1,2\n", 1024)])
def test_download_failure_leaves_no_published_or_partial_file(tmp_path, monkeypatch, raw, limit):
    entry = entry_for()
    entry["source"]["max_bytes"] = limit
    if limit < entry["source"]["bytes"]:
        entry["source"]["bytes"] = limit
    monkeypatch.setattr(corpus.urllib.request, "urlopen", lambda *a, **k: Response(raw))
    with pytest.raises(ValueError, match="bytes|limit"):
        corpus.fetch_one(entry, tmp_path)
    assert not (tmp_path / "example" / "source.csv").exists()
    assert not list(tmp_path.rglob(".download-*"))


def test_failed_validation_is_archived_and_never_reported_ready(tmp_path):
    entry = entry_for()
    receipt_path, receipt = corpus.run_collection([entry], tmp_path, download=False)
    assert receipt["ready_count"] == 0
    assert receipt["failed_count"] == 1
    assert receipt["datasets"][0]["status"] == "failed"
    assert json.loads(receipt_path.read_text()) == receipt


def test_explicit_catalogue_request_is_not_ready(tmp_path):
    entry = {"id": "unfetched", "availability": "catalogued", "reason": "No pinned artifact"}
    _, receipt = corpus.run_collection([entry], tmp_path, download=True)
    assert receipt["ready_count"] == 0
    assert receipt["datasets"][0]["status"] == "catalogued"


def test_budget_refusal_happens_before_network(tmp_path, monkeypatch):
    def unexpected_network(*args, **kwargs):
        pytest.fail("The declared download exceeds the collection budget")

    monkeypatch.setattr(corpus.urllib.request, "urlopen", unexpected_network)
    _, receipt = corpus.run_collection([entry_for()], tmp_path, download=True, budget_bytes=1)
    assert receipt["ready_count"] == 0
    assert receipt["failed_count"] == 1
    assert "budget" in receipt["datasets"][0]["error"]


def test_zip_companion_must_match_the_target_before_alignment(tmp_path):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("train.csv", "x,y\n1,2\n3,4\n")
        archive.writestr("groups.csv", "material,y\nA,4\nB,2\n")
    raw = buffer.getvalue()
    entry = entry_for(raw)
    entry["source"].update(
        format="zip_csv",
        member="train.csv",
        companion={"member": "groups.csv", "columns": ["material"], "align_on": ["y"]},
    )
    entry["schema"]["columns"].append("material")
    entry["exclude_columns"]["material"] = "Group identifier"
    write_source(tmp_path, entry, raw)
    with pytest.raises(ValueError, match="companion"):
        corpus.fetch_one(entry, tmp_path, allow_download=False)


def test_reference_is_verified_in_place_without_copying(tmp_path):
    raw = b"x,y\n1,2\n3,4\n"
    reference = tmp_path / "user-data.csv"
    reference.write_bytes(raw)
    entry = entry_for(raw)
    entry["availability"] = "local_reference"
    entry["source"]["reference_path"] = str(reference)
    record = corpus.fetch_one(entry, tmp_path / "corpus", allow_download=False)
    assert record["status"] == "ready"
    assert record["storage"] == "local_reference"
    assert reference.read_bytes() == raw
    assert not (tmp_path / "corpus" / "example").exists()


def test_provided_zip_partitions_remain_distinguishable(tmp_path):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("train.csv", "x,y\n1,2\n")
        archive.writestr("test.csv", "x,y\n3,4\n")
    raw = buffer.getvalue()
    entry = entry_for(raw)
    entry["source"].update(
        format="zip_csv",
        members=[
            {"member": "train.csv", "label": "train"},
            {"member": "test.csv", "label": "test"},
        ],
        partition_column="partition",
    )
    path = write_source(tmp_path, entry, raw)
    frame = corpus.read_frame(entry["source"], path)
    assert frame["partition"].tolist() == ["train", "test"]
    assert frame["y"].tolist() == [2, 4]


def test_related_tables_do_not_inflate_independent_source_count(tmp_path):
    raw = b"x,y\n1,2\n3,4\n"
    first = entry_for(raw)
    second = entry_for(raw)
    second["id"] = "second_target"
    first["independent_unit"] = second["independent_unit"] = "one_source"
    write_source(tmp_path, first, raw)
    write_source(tmp_path, second, raw)
    _, receipt = corpus.run_collection([first, second], tmp_path, download=False)
    assert receipt["ready_count"] == 2
    assert receipt["independent_ready_real_sources"] == 1


def test_string_predictor_requires_an_explicit_categorical_declaration(tmp_path):
    raw = b"x,y\nred,2\nblue,4\n"
    entry = entry_for(raw)
    write_source(tmp_path, entry, raw)
    with pytest.raises(ValueError, match="categorical"):
        corpus.fetch_one(entry, tmp_path, allow_download=False)


def competition_entry(raw):
    entry = entry_for(raw)
    entry["availability"] = "kaggle_competition"
    entry["source"].pop("url")
    entry["source"].update(competition="example-competition", member_file="source.csv")
    return entry


def recording_cli(written, calls):
    """Stand in for the Kaggle CLI, which chooses its own output filename."""

    def run(command, **kwargs):
        calls.append(command)
        staging = Path(command[command.index("-p") + 1])
        (staging / "downloaded-name.csv").write_bytes(written)
        return subprocess.CompletedProcess(command, 0)

    return run


def test_competition_download_publishes_the_pinned_member(tmp_path, monkeypatch):
    raw = b"x,y\n1,2\n3,4\n"
    entry = competition_entry(raw)
    calls = []
    monkeypatch.setattr(corpus.subprocess, "run", recording_cli(raw, calls))
    record = corpus.fetch_one(entry, tmp_path)
    assert record["status"] == "ready"
    assert record["storage"] == "kaggle_cli"
    assert record["downloaded_this_run"] is True
    assert (tmp_path / "example" / "source.csv").read_bytes() == raw
    assert calls == [
        [
            "uv",
            "run",
            "--with",
            "kaggle==2.2.4",
            "kaggle",
            "competitions",
            "download",
            "-c",
            "example-competition",
            "-f",
            "source.csv",
            "-p",
            calls[0][-1],
        ]
    ]
    assert Path(calls[0][-1]).parent == tmp_path / "example"


def test_competition_download_of_wrong_bytes_publishes_nothing(tmp_path, monkeypatch):
    entry = competition_entry(b"x,y\n1,2\n3,4\n")
    monkeypatch.setattr(corpus.subprocess, "run", recording_cli(b"x,y\n9,9\n9,9\n", []))
    with pytest.raises(ValueError, match="bytes|SHA256"):
        corpus.fetch_one(entry, tmp_path)
    assert not (tmp_path / "example" / "source.csv").exists()
    assert not list(tmp_path.rglob(".kaggle-*"))


def test_existing_competition_member_is_never_refetched(tmp_path, monkeypatch):
    raw = b"x,y\n1,2\n3,4\n"
    entry = competition_entry(raw)
    write_source(tmp_path, entry, raw)

    def unexpected_cli(*args, **kwargs):
        pytest.fail("An existing pinned member must not trigger the Kaggle CLI")

    monkeypatch.setattr(corpus.subprocess, "run", unexpected_cli)
    record = corpus.fetch_one(entry, tmp_path)
    assert record["status"] == "ready"
    assert record["downloaded_this_run"] is False
