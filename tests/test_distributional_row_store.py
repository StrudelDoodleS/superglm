"""Bounded typed row storage verifies content before replay or cache reuse."""

import gc
import hashlib
import json
import os
import pickle
import tracemalloc
import weakref
from dataclasses import FrozenInstanceError

import numpy as np
import pytest


def _module():
    from superglm.distributional import row_store

    return row_store


def _store(path, *, n=13, block_rows=4):
    mod = _module()
    writer = mod.RowStoreWriter(
        path, (mod.RowColumn("y", "<f8"), mod.RowColumn("i", "<i4")), block_rows=block_rows
    )
    writer.append({"y": np.arange(n, dtype="<f8"), "i": np.arange(n, dtype="<i4")})
    return writer.finish()


def _rewrite_manifest(path, change):
    mod = _module()
    manifest = json.loads((path / "manifest.json").read_bytes())
    change(manifest)
    (path / "manifest.json").write_bytes(mod._json_bytes(manifest))


def _mutate_preserving_time(path):
    info = path.stat()
    original = path.read_bytes()
    changed = bytearray(original)
    changed[len(changed) // 2] ^= 1
    path.write_bytes(changed)
    os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns))
    return original


def test_arbitrary_append_boundaries_and_unaligned_verified_range(tmp_path):
    from superglm.distributional.row_store import PreparedRowStore, RowColumn, RowStoreWriter

    schema = (RowColumn("y", "<f8"), RowColumn("code", "<i4"))
    writer = RowStoreWriter(tmp_path / "rows", schema, block_rows=4)
    writer.append({"y": np.arange(3, dtype="<f8"), "code": np.arange(3, dtype="<i4")})
    writer.append({"y": np.arange(3, 9, dtype="<f8"), "code": np.arange(3, 9, dtype="<i4")})
    store = writer.finish()
    block = store.read_verified(3, 7)
    np.testing.assert_array_equal(block.columns["y"], np.arange(3, 7))
    assert [(b.start, b.stop) for b in store.iter_verified()] == [(0, 4), (4, 8), (8, 9)]
    assert (
        PreparedRowStore.open(tmp_path / "rows", expected_generation=store.generation).generation
        == store.generation
    )


def test_content_generation_preserves_bits_strides_and_input_ownership(tmp_path):
    mod = _module()
    schema = tuple(
        mod.RowColumn(name, dtype)
        for name, dtype in (("float", "<f8"), ("big", "<i8"), ("small", "<i4"), ("byte", "|u1"))
    )
    floating = np.array(
        [0, 1 << 63, 0x7FF0000000000000, 0x7FF8000000000001, 0xFFF0000000000000], dtype="<u8"
    ).view("<f8")
    columns = {
        "float": floating,
        "big": np.arange(10, dtype="<i8")[::2],
        "small": np.arange(5, dtype="<i4")[::-1],
        "byte": np.arange(5, dtype="|u1"),
    }
    metadata = {"labels": ["α", "b"], "nested": {"finite": 0.5, "null": None}}
    first = mod.RowStoreWriter(tmp_path / "one", schema, metadata=metadata, block_rows=3)
    second = mod.RowStoreWriter(tmp_path / "two", schema, metadata=metadata, block_rows=3)
    first.append(columns)
    for start, stop in ((0, 1), (1, 3), (3, 5)):
        second.append({name: value[start:stop] for name, value in columns.items()})
    expected = {name: value.tobytes() for name, value in columns.items()}
    for value in columns.values():
        value[:] = 7
    metadata["labels"][0] = "changed"
    left, right = first.finish(), second.finish()
    assert left.generation == right.generation
    assert left.metadata["labels"] == ("α", "b")
    for name in columns:
        assert (
            b"".join(block.columns[name].tobytes() for block in left.iter_verified())
            == expected[name]
        )
    with pytest.raises(TypeError):
        left.metadata["nested"]["finite"] = 1
    with pytest.raises(FrozenInstanceError):
        left.schema[0].name = "new"


@pytest.mark.parametrize("start,stop", [(0, 1), (3, 4), (4, 8), (3, 7), (8, 12), (12, 13)])
def test_exact_range_snapshots_and_immutable_backing(tmp_path, start, stop):
    store = _store(tmp_path / "rows")
    block = store.read_verified(start, stop)
    assert block.authority == store.verify_range(start, stop)
    for name, dtype in (("y", "<f8"), ("i", "<i4")):
        value = block.columns[name]
        np.testing.assert_array_equal(value, np.arange(start, stop))
        assert value.dtype.str == dtype
        owner = value
        while isinstance(owner, np.ndarray):
            with pytest.raises(ValueError):
                owner.flags.writeable = True
            owner = owner.base
        assert type(owner) is bytes
    with pytest.raises(TypeError):
        block.columns["other"] = np.zeros(1)
    _mutate_preserving_time(tmp_path / "rows" / "blocks" / f"{start // 4:08d}.bin")
    np.testing.assert_array_equal(block.columns["y"], np.arange(start, stop))


@pytest.mark.parametrize(
    "start,stop",
    [
        (-1, 1),
        (2, 1),
        (0, 0),
        (0, 5),
        (12, 14),
        (True, 2),
        (0, False),
        (0.0, 2),
        (0, 1.5),
        (np.bool_(True), 2),
    ],
)
@pytest.mark.parametrize("operation", ["read_verified", "verify_range"])
def test_invalid_ranges_refuse_before_io(tmp_path, monkeypatch, start, stop, operation):
    mod = _module()
    store = _store(tmp_path / "rows")
    monkeypatch.setattr(mod, "_read_file", lambda *args: pytest.fail("invalid range performed I/O"))
    with pytest.raises(
        TypeError
        if isinstance(start, (bool, float, np.bool_)) or isinstance(stop, (bool, float))
        else ValueError
    ):
        getattr(store, operation)(start, stop)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"block_rows": 0},
        {"block_rows": 65537},
        {"block_rows": True},
        {"block_rows": 1.5},
        {"metadata": {"bad": float("nan")}},
        {"metadata": {"big": "x" * (64 << 10)}},
        {"metadata": {1: "bad"}},
    ],
)
def test_invalid_writer_options_do_not_create_destination(tmp_path, kwargs):
    mod = _module()
    path = tmp_path / "rows"
    with pytest.raises((TypeError, ValueError)):
        mod.RowStoreWriter(path, (mod.RowColumn("y", "<f8"),), **kwargs)
    assert not path.exists()


@pytest.mark.parametrize("dtype", [">f8", "<f4", "object", "<u8", "f8"])
def test_invalid_schema_dtype(dtype):
    with pytest.raises(ValueError):
        _module().RowColumn("y", dtype)


@pytest.mark.parametrize(
    "bad", ["missing", "length", "dtype", "object", "matrix", "proxy", "subclass"]
)
def test_append_validates_all_arrays_before_consumption(tmp_path, bad):
    mod = _module()
    writer = mod.RowStoreWriter(
        tmp_path / "rows", (mod.RowColumn("y", "<f8"), mod.RowColumn("i", "<i4")), block_rows=4
    )
    columns = {"y": np.arange(4, dtype="<f8"), "i": np.arange(4, dtype="<i4")}
    if bad == "missing":
        del columns["i"]
    elif bad == "length":
        columns["i"] = columns["i"][:3]
    elif bad == "dtype":
        columns["i"] = columns["i"].astype("<i8")
    elif bad == "object":
        columns["y"] = np.array([object()] * 4, dtype=object)
    elif bad == "matrix":
        columns["i"] = columns["i"].reshape(2, 2)
    elif bad == "proxy":

        class Proxy:
            def __array__(self, *args, **kwargs):
                pytest.fail("append must not coerce input proxies")

        columns["y"] = Proxy()
    else:

        class Subclass(np.ndarray):
            pass

        columns["y"] = columns["y"].view(Subclass)
    with pytest.raises((TypeError, ValueError)):
        writer.append(columns)
    assert writer._n_rows == 0
    assert list((tmp_path / "rows" / "blocks").iterdir()) == []
    writer.abort()


def test_publication_abort_and_persistence(tmp_path):
    mod = _module()
    path = tmp_path / "rows"
    writer = mod.RowStoreWriter(path, (mod.RowColumn("x", "<f8"),), block_rows=3)
    writer.append({"x": np.arange(4, dtype="<f8")})
    with pytest.raises(mod.RowStoreIntegrityError):
        mod.PreparedRowStore.open(path)
    writer.abort()
    assert not path.exists()
    store = _store(path)
    generation = store.generation
    reference = weakref.ref(store)
    del store, writer
    gc.collect()
    assert reference() is None
    assert mod.PreparedRowStore.open(path, expected_generation=generation).generation == generation
    with pytest.raises(ValueError, match="exists"):
        mod.RowStoreWriter(path, (mod.RowColumn("x", "<f8"),))
    with pytest.raises(mod.RowStoreIntegrityError, match="generation"):
        mod.PreparedRowStore.open(path, expected_generation="0" * 64)


@pytest.mark.parametrize(
    "file", ["blocks/00000000.bin", "blocks/00000001.bin", "index.bin", "manifest.json"]
)
def test_mutation_after_apparent_cache_hit_requires_fresh_verification(tmp_path, file):
    mod = _module()
    path = tmp_path / "rows"
    store = _store(path)
    cached = store.read_verified(3, 7)
    assert cached.authority == store.verify_range(3, 7)
    original = _mutate_preserving_time(path / file)
    with pytest.raises(mod.RowStoreIntegrityError):
        store.verify_range(cached.start, cached.stop)
    (path / file).write_bytes(original)
    restored = mod.PreparedRowStore.open(path, expected_generation=store.generation)
    assert restored.verify_range(3, 7) == cached.authority


def test_final_metadata_check_detects_mutation_during_payload_io(tmp_path, monkeypatch):
    mod = _module()
    path = tmp_path / "rows"
    store = _store(path)
    original = mod._read_file

    def read(directory, name, *args):
        result = original(directory, name, *args)
        if name == "00000000.bin":
            _mutate_preserving_time(path / "manifest.json")
        return result

    monkeypatch.setattr(mod, "_read_file", read)
    with pytest.raises(mod.RowStoreIntegrityError):
        store.read_verified(0, 4)


@pytest.mark.parametrize("file", ["blocks/00000000.bin", "index.bin", "manifest.json"])
def test_truncation_after_open_is_refused(tmp_path, file):
    mod = _module()
    path = tmp_path / "rows"
    store = _store(path)
    target = path / file
    target.write_bytes(target.read_bytes()[:-1])
    with pytest.raises(mod.RowStoreIntegrityError):
        store.read_verified(0, 4)


@pytest.mark.parametrize(
    "field,value",
    [
        ("format", "wrong"),
        ("n_rows", 100_000_001),
        ("n_rows", True),
        ("n_blocks", 50000),
        ("block_rows", 0),
        ("block_rows", 65537),
        ("schema", [{"name": "y", "dtype": "<f4"}]),
    ],
)
def test_forged_manifest_refuses_before_payload_read(tmp_path, monkeypatch, field, value):
    mod = _module()
    path = tmp_path / "rows"
    _store(path)
    _rewrite_manifest(path, lambda manifest: manifest.update({field: value}))
    original = mod._read_file

    def read(directory, name, *args):
        assert not name.endswith(".bin") or name == "index.bin"
        return original(directory, name, *args)

    monkeypatch.setattr(mod, "_read_file", read)
    with pytest.raises(mod.RowStoreIntegrityError):
        mod.PreparedRowStore.open(path)


def test_duplicate_json_oversized_files_and_bad_index_coverage(tmp_path):
    mod = _module()
    path = tmp_path / "rows"
    _store(path)
    manifest_path = path / "manifest.json"
    saved = manifest_path.read_bytes()
    manifest_path.write_bytes(saved[:-1] + b',"n_rows":13}')
    with pytest.raises(mod.RowStoreIntegrityError):
        mod.PreparedRowStore.open(path)
    manifest_path.write_bytes(b"x" * ((64 << 10) + 1))
    with pytest.raises(mod.RowStoreIntegrityError):
        mod.PreparedRowStore.open(path)
    manifest_path.write_bytes(saved)
    index = bytearray((path / "index.bin").read_bytes())
    first, stop, size, digest = mod._RECORD.unpack_from(index)
    mod._RECORD.pack_into(index, 0, first + 1, stop, size, digest)
    (path / "index.bin").write_bytes(index)
    _rewrite_manifest(
        path, lambda manifest: manifest.update(index_sha256=hashlib.sha256(index).hexdigest())
    )
    with pytest.raises(mod.RowStoreIntegrityError, match="cover"):
        mod.PreparedRowStore.open(path)


@pytest.mark.parametrize(
    "target", ["manifest.json", "index.bin", "blocks/00000000.bin", "blocks", "root"]
)
def test_symlinks_are_refused(tmp_path, target):
    mod = _module()
    path = tmp_path / "rows"
    _store(path)
    original = path if target == "root" else path / target
    moved = tmp_path / "moved"
    original.rename(moved)
    original.symlink_to(moved, target_is_directory=moved.is_dir())
    with pytest.raises(mod.RowStoreIntegrityError):
        mod.PreparedRowStore.open(path)


def test_index_limit_is_checked_before_append_and_index_stays_raw(tmp_path):
    mod = _module()
    writer = mod.RowStoreWriter(tmp_path / "rows", (mod.RowColumn("x", "|u1"),), block_rows=1)
    with pytest.raises(ValueError, match="index"):
        writer.append({"x": np.zeros((1 << 20) // 56 + 1, dtype="|u1")})
    assert writer._n_rows == 0
    writer.append({"x": np.arange(3, dtype="|u1")})
    store = writer.finish()
    assert type(store._index_bytes) is bytes
    assert store.retained_index_bytes == 3 * 56
    assert not any(isinstance(value, list) for value in vars(store).values())


def test_actual_read_traffic_and_array_lifetimes(tmp_path, monkeypatch):
    mod = _module()
    store = _store(tmp_path / "rows", n=17)
    original = mod._read_file
    reads = []

    def read(directory, name, *args):
        result = original(directory, name, *args)
        reads.append((name, len(result)))
        return result

    monkeypatch.setattr(mod, "_read_file", read)
    for operation in (store.read_verified, store.verify_range, store.verify_range):
        reads.clear()
        operation(3, 7)
        assert reads == [
            ("manifest.json", len(store._manifest_bytes)),
            ("index.bin", store.retained_index_bytes),
            ("00000000.bin", 48),
            ("00000001.bin", 48),
            ("manifest.json", len(store._manifest_bytes)),
            ("index.bin", store.retained_index_bytes),
        ]
    iterator = store.iter_verified()
    first = next(iterator)
    reference = weakref.ref(first.columns["y"])
    del first
    assert reference() is None
    retained = next(iterator)
    store_ref = weakref.ref(store)
    iterator.close()
    del iterator, store, operation
    assert store_ref() is None
    np.testing.assert_array_equal(retained.columns["y"], np.arange(4, 8))


@pytest.mark.parametrize("width", [1, 128])
def test_actual_byte_allocation_is_bounded_for_narrow_and_wide_rows(
    tmp_path, monkeypatch, record_property, width
):
    mod = _module()
    schema = tuple(mod.RowColumn(f"c{i}", "<f8") for i in range(width))
    writer = mod.RowStoreWriter(tmp_path / "rows", schema)
    effective = min(65536, (8 << 20) // (8 * width))
    assert writer.block_rows == effective
    source = np.arange(2 * effective, dtype="<f8")[::2]
    for _ in range(4):
        writer.append({column.name: source for column in schema})
    store = writer.finish()

    def forbidden(*args, **kwargs):
        pytest.fail("verified row access must not map or unpickle files")

    monkeypatch.setattr(np, "memmap", forbidden)
    monkeypatch.setattr(np, "load", forbidden)
    monkeypatch.setattr(pickle, "loads", forbidden)
    requested = []
    original = mod._read_file

    def read(directory, name, limit, exact=None):
        requested.append((name, limit, exact))
        return original(directory, name, limit, exact)

    monkeypatch.setattr(mod, "_read_file", read)
    actual_requests = []
    original_fdopen = mod.os.fdopen

    class ReadSpy:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            self.stream.__enter__()
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def read(self, size):
            actual_requests.append(size)
            return self.stream.read(size)

        def fileno(self):
            return self.stream.fileno()

    monkeypatch.setattr(
        mod.os, "fdopen", lambda *args, **kwargs: ReadSpy(original_fdopen(*args, **kwargs))
    )
    tracemalloc.start()
    try:
        block = store.read_verified(1, effective + 1)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak + len(store._manifest_bytes) + store.retained_index_bytes <= 3 * (8 << 20) + 2 * (
        (1 << 20) + (64 << 10)
    )
    assert sum(array.nbytes for array in block.columns.values()) == effective * width * 8
    assert [exact for name, _, exact in requested if name.startswith("000")] == [
        effective * width * 8
    ] * 2
    assert all(limit <= (8 << 20) for _, limit, _ in requested)
    assert actual_requests == [exact for _, _, exact in requested]
    record_property("peak_read_bytes", peak)
    record_property(
        "pinned_metadata_bytes", len(store._manifest_bytes) + store.retained_index_bytes
    )
    record_property("actual_read_request_bytes", json.dumps(actual_requests))


@pytest.mark.parametrize("size", [0, 129])
def test_schema_column_count_refused_before_creation(tmp_path, size):
    mod = _module()
    schema = tuple(mod.RowColumn(str(i), "|u1") for i in range(size))
    with pytest.raises(ValueError):
        mod.RowStoreWriter(tmp_path / "rows", schema)
    assert not (tmp_path / "rows").exists()


def test_duplicate_schema_and_row_budget_refused_before_consumption(tmp_path, monkeypatch):
    mod = _module()
    with pytest.raises(ValueError, match="unique"):
        mod.RowStoreWriter(tmp_path / "rows", (mod.RowColumn("x", "<f8"),) * 2)
    assert mod._MAX_ROWS == 100_000_000
    monkeypatch.setattr(mod, "_MAX_ROWS", 8)
    writer = mod.RowStoreWriter(tmp_path / "rows", (mod.RowColumn("x", "|u1"),), block_rows=4)
    with pytest.raises(ValueError, match="budget"):
        writer.append({"x": np.zeros(9, dtype="|u1")})
    assert writer._n_rows == 0
    writer.abort()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_rows": 0},
        {"batch_rows": 5},
        {"batch_rows": True},
        {"batch_rows": 1.5},
        {"start": 4, "stop": 4},
        {"start": -1},
        {"stop": 14},
    ],
)
def test_iterator_rejects_invalid_ranges_and_batch_sizes(tmp_path, kwargs):
    store = _store(tmp_path / "rows")
    with pytest.raises((TypeError, ValueError)):
        next(store.iter_verified(**kwargs))


def test_replaced_generation_refuses_old_store_and_expected_generation(tmp_path):
    mod = _module()
    path = tmp_path / "rows"
    store = _store(path)
    store.read_verified(0, 4)
    path.rename(tmp_path / "old")
    replacement = _store(path, n=17)
    assert replacement.generation != store.generation
    with pytest.raises(mod.RowStoreIntegrityError):
        store.verify_range(0, 4)
    with pytest.raises(mod.RowStoreIntegrityError):
        mod.PreparedRowStore.open(path, expected_generation=store.generation)


def test_generation_validation_does_not_claim_payload_verification(tmp_path):
    mod = _module()
    path = tmp_path / "rows"
    store = _store(path)
    _mutate_preserving_time(path / "blocks" / "00000000.bin")
    store.validate_generation()
    with pytest.raises(mod.RowStoreIntegrityError):
        store.verify_range(0, 4)


@pytest.mark.parametrize("target", ["root", "blocks"])
def test_writer_refuses_replaced_directories_without_touching_replacement(tmp_path, target):
    mod = _module()
    path = tmp_path / "rows"
    writer = mod.RowStoreWriter(path, (mod.RowColumn("x", "|u1"),), block_rows=2)
    changed = path if target == "root" else path / "blocks"
    changed.rename(tmp_path / "original")
    changed.mkdir()
    sentinel = changed / "sentinel"
    sentinel.write_bytes(b"preserve")
    with pytest.raises(mod.RowStoreIntegrityError):
        writer.append({"x": np.zeros(2, dtype="|u1")})
    with pytest.raises(mod.RowStoreIntegrityError):
        writer.abort()
    assert sentinel.read_bytes() == b"preserve"
    assert sorted(item.name for item in changed.iterdir()) == ["sentinel"]


def test_abort_preserves_unowned_files_and_published_store(tmp_path):
    mod = _module()
    path = tmp_path / "rows"
    writer = mod.RowStoreWriter(path, (mod.RowColumn("x", "|u1"),), block_rows=2)
    writer.append({"x": np.arange(3, dtype="|u1")})
    (path / "extra").write_bytes(b"preserve")
    writer.abort()
    assert sorted(item.name for item in path.iterdir()) == ["extra"]
    writer = mod.RowStoreWriter(tmp_path / "published", (mod.RowColumn("x", "|u1"),))
    writer.append({"x": np.arange(1, dtype="|u1")})
    store = writer.finish()
    writer.abort()
    np.testing.assert_array_equal(store.read_verified(0, 1).columns["x"], [0])
