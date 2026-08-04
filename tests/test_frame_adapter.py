from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import polars as pl
import pytest

import superglm._frame as frame_module
from superglm._frame import as_eager_frame, is_supported_eager_frame


def test_as_eager_frame_wraps_pandas_once() -> None:
    X = pd.DataFrame({"x": [1.0, 2.0], "label": ["a", "b"]})

    frame = as_eager_frame(X)

    assert frame.native is X
    assert frame.backend == "pandas"
    assert frame.columns == ("x", "label")
    assert len(frame) == 2
    assert as_eager_frame(frame) is frame


def test_pandas_fast_path_does_not_enter_narwhals(monkeypatch: pytest.MonkeyPatch) -> None:
    X = pd.DataFrame({"x": [1.0, 2.0], "label": ["a", "b"]})

    monkeypatch.setattr(
        frame_module.nw,
        "from_native",
        lambda *_args, **_kwargs: pytest.fail("pandas must not pay Narwhals dispatch"),
    )

    frame = as_eager_frame(X)

    assert frame.columns == ("x", "label")
    assert len(frame) == 2
    np.testing.assert_array_equal(frame.column_array("x"), [1.0, 2.0])


def test_unrelated_inputs_do_not_enter_narwhals_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    X = np.ones((2, 1))

    monkeypatch.setattr(
        frame_module.nw,
        "from_native",
        lambda *_args, **_kwargs: pytest.fail("non-frame inputs must bypass Narwhals dispatch"),
    )

    assert not is_supported_eager_frame(X)
    with pytest.raises(ValueError, match="pandas or eager Polars DataFrame"):
        as_eager_frame(X)


def test_pandas_string_extraction_keeps_the_array_protocol_fast_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    X = pd.DataFrame({"label": ["a", "b"]})

    monkeypatch.setattr(
        pd.Series,
        "to_numpy",
        lambda *_args, **_kwargs: pytest.fail("StringDtype.to_numpy allocates on this path"),
    )

    np.testing.assert_array_equal(as_eager_frame(X).column_array("label"), ["a", "b"])


def test_as_eager_frame_wraps_eager_polars_once() -> None:
    X = pl.DataFrame({"x": [1.0, 2.0], "label": ["a", "b"]})

    frame = as_eager_frame(X)

    assert frame.native is X
    assert frame.backend == "polars"
    assert frame.columns == ("x", "label")
    assert len(frame) == 2
    assert as_eager_frame(frame) is frame


def test_polars_schema_is_cached_across_classification_and_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    X = pl.DataFrame({f"x{index}": [index, index + 1] for index in range(12)})
    frame = as_eager_frame(X)
    frame_type = type(frame._polars_frame)
    schema_property = frame_type.schema
    calls = 0

    def counted_schema(self):
        nonlocal calls
        calls += 1
        return schema_property.__get__(self, frame_type)

    monkeypatch.setattr(frame_type, "schema", property(counted_schema))

    for name in frame.columns:
        assert frame.column_kind(name) == "numeric"
        assert frame.column_dtype(name) == "Int64"
    frame.digest()

    assert calls == 1


def test_polars_column_extraction_bypasses_narwhals_column_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    X = pl.DataFrame({f"x{index}": [index, index + 1] for index in range(12)})
    frame = as_eager_frame(X)

    monkeypatch.setattr(
        type(frame._polars_frame),
        "__getitem__",
        lambda *_args, **_kwargs: pytest.fail(
            "Polars extraction must use native constant-time column lookup"
        ),
    )

    for index, name in enumerate(frame.columns):
        np.testing.assert_array_equal(frame.column_array(name), [index, index + 1])


@pytest.mark.parametrize("value", [np.ones((2, 1)), {"x": [1.0, 2.0]}, object()])
def test_as_eager_frame_rejects_unrelated_objects(value: object) -> None:
    with pytest.raises(ValueError, match="pandas or eager Polars DataFrame"):
        as_eager_frame(value)

    assert not is_supported_eager_frame(value)


def test_as_eager_frame_rejects_lazy_polars_with_collection_guidance() -> None:
    X = pl.DataFrame({"x": [1.0, 2.0]}).lazy()

    with pytest.raises(ValueError, match="eager.*collect"):
        as_eager_frame(X)

    assert not is_supported_eager_frame(X)


def test_as_eager_frame_rejects_duplicate_pandas_columns() -> None:
    X = pd.DataFrame(np.ones((2, 2)), columns=["x", "x"])

    with pytest.raises(ValueError, match="columns must be unique"):
        as_eager_frame(X)


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_require_columns_reports_all_missing_names(backend: str) -> None:
    native = (
        pd.DataFrame({"x": [1.0, 2.0]}) if backend == "pandas" else pl.DataFrame({"x": [1.0, 2.0]})
    )
    frame = as_eager_frame(native)

    with pytest.raises(ValueError, match=r"missing required columns: \['y', 'z'\]"):
        frame.require_columns(("x", "y", "z", "y"))


def test_pandas_column_kinds_preserve_current_autodetection_semantics() -> None:
    X = pd.DataFrame(
        {
            "float": [1.0, 2.0],
            "integer": [1, 2],
            "boolean": [True, False],
            "object": ["a", "b"],
            "string": pd.Series(["a", "b"], dtype="string"),
            "categorical": pd.Series(["a", "b"], dtype="category"),
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        }
    )
    frame = as_eager_frame(X)

    assert frame.column_kind("float") == "numeric"
    assert frame.column_kind("integer") == "numeric"
    assert frame.column_kind("boolean") == "boolean"
    assert frame.column_kind("object") == "categorical"
    assert frame.column_kind("string") == "categorical"
    assert frame.column_kind("categorical") == "categorical"
    assert frame.column_kind("date") == "unsupported"


def test_polars_column_kinds_and_logical_categorical_values() -> None:
    X = pl.DataFrame(
        {
            "float": [1.0, 2.0],
            "integer": [1, 2],
            "boolean": [True, False],
            "string": ["a", "b"],
            "categorical": pl.Series(["a", "b"], dtype=pl.Categorical),
            "enum": pl.Series(["silver", "bronze"], dtype=pl.Enum(["bronze", "silver"])),
            "date": pl.Series(["2024-01-01", "2024-01-02"]).str.to_date(),
        }
    )
    frame = as_eager_frame(X)

    assert frame.column_kind("float") == "numeric"
    assert frame.column_kind("integer") == "numeric"
    assert frame.column_kind("boolean") == "boolean"
    assert frame.column_kind("string") == "categorical"
    assert frame.column_kind("categorical") == "categorical"
    assert frame.column_kind("enum") == "categorical"
    assert frame.column_kind("date") == "unsupported"
    np.testing.assert_array_equal(frame.column_array("categorical"), ["a", "b"])
    np.testing.assert_array_equal(frame.column_array("enum"), ["silver", "bronze"])


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_column_array_extracts_once_and_coerces_from_cached_raw_values(
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native = (
        pd.DataFrame({"x": [1, 2, 3]}) if backend == "pandas" else pl.DataFrame({"x": [1, 2, 3]})
    )
    frame = as_eager_frame(native)
    calls: list[object] = []
    original = type(frame)._extract_column

    def counted(self, name: object):
        calls.append(name)
        return original(self, name)

    monkeypatch.setattr(type(frame), "_extract_column", counted)

    raw_first = frame.column_array("x")
    converted = frame.column_array("x", dtype=np.float64)
    raw_second = frame.column_array("x")

    assert calls == ["x"]
    assert raw_second is raw_first
    assert converted.dtype == np.float64
    np.testing.assert_array_equal(converted, [1.0, 2.0, 3.0])


def test_polars_column_extraction_never_converts_the_frame_to_pandas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    X = pl.DataFrame({"x": [1.0, 2.0]})

    monkeypatch.setattr(
        pl.DataFrame,
        "to_pandas",
        lambda *_args, **_kwargs: pytest.fail("whole-frame conversion is forbidden"),
    )

    np.testing.assert_array_equal(as_eager_frame(X).column_array("x"), [1.0, 2.0])


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_take_rows_is_positional_and_preserves_native_backend(backend: str) -> None:
    native = (
        pd.DataFrame({"x": [10, 20, 30], "label": ["a", "b", "c"]}, index=[7, 5, 9])
        if backend == "pandas"
        else pl.DataFrame({"x": [10, 20, 30], "label": ["a", "b", "c"]})
    )

    selected = as_eager_frame(native).take_rows(np.array([2, 0, 2], dtype=np.intp))
    selected_frame = as_eager_frame(selected)

    assert selected_frame.backend == backend
    np.testing.assert_array_equal(selected_frame.column_array("x"), [30, 10, 30])
    np.testing.assert_array_equal(selected_frame.column_array("label"), ["c", "a", "c"])


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_select_native_preserves_backend_order_and_rows(backend: str) -> None:
    native = (
        pd.DataFrame({"x": [1, 2], "y": [3, 4], "z": [5, 6]})
        if backend == "pandas"
        else pl.DataFrame({"x": [1, 2], "y": [3, 4], "z": [5, 6]})
    )

    selected = as_eager_frame(native).select_native(("z", "x"))
    selected_frame = as_eager_frame(selected)

    assert selected_frame.backend == backend
    assert selected_frame.columns == ("z", "x")
    assert len(selected_frame) == 2


def test_pandas_selection_and_digest_preserve_falsey_hashable_column_labels() -> None:
    native = pd.DataFrame(
        {
            0: [1.0, 2.0],
            None: [3.0, 4.0],
            "": [5.0, 6.0],
            ("tuple", "label"): [7.0, 8.0],
        }
    )
    frame = as_eager_frame(native)
    labels = (None, "", ("tuple", "label"))

    selected = frame.select_native(labels)

    assert list(selected.columns) == list(labels)
    assert frame.digest(labels) == as_eager_frame(selected).digest()


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_select_native_rejects_empty_selection_without_losing_row_count(backend: str) -> None:
    native = (
        pd.DataFrame({"unused": [1, 2]})
        if backend == "pandas"
        else pl.DataFrame({"unused": [1, 2]})
    )

    with pytest.raises(ValueError, match="empty native column selection"):
        as_eager_frame(native).select_native(())


def _legacy_pandas_digest(frame: pd.DataFrame) -> bytes:
    digest = hashlib.blake2b(digest_size=16, person=b"superglm-fit-v1")
    metadata = tuple((repr(name), str(dtype)) for name, dtype in frame.dtypes.items())
    digest.update(repr((frame.shape, metadata)).encode("utf-8"))
    row_hashes = pd.util.hash_pandas_object(frame, index=True, categorize=True).to_numpy(
        dtype=np.uint64,
        copy=False,
    )
    digest.update(np.ascontiguousarray(row_hashes).data)
    return digest.digest()


def test_pandas_digest_matches_the_existing_fit_guard_bytes() -> None:
    X = pd.DataFrame(
        {"x": [1.0, 2.0], "category": pd.Series(["a", "b"], dtype="category")},
        index=[10, 20],
    )

    assert as_eager_frame(X).digest() == _legacy_pandas_digest(X)
    assert as_eager_frame(X).digest(("category",)) == _legacy_pandas_digest(X[["category"]])


def test_selected_digest_ignores_unused_columns_for_both_backends() -> None:
    pandas_left = pd.DataFrame({"used": [1.0, 2.0], "unused": [[1], [2]]})
    pandas_right = pd.DataFrame({"used": [1.0, 2.0], "unused": [[9], [8]]})
    polars_left = pl.DataFrame({"used": [1.0, 2.0], "unused": ["a", "b"]})
    polars_right = pl.DataFrame({"used": [1.0, 2.0], "unused": ["x", "y"]})

    assert as_eager_frame(pandas_left).digest(("used",)) == as_eager_frame(pandas_right).digest(
        ("used",)
    )
    assert as_eager_frame(polars_left).digest(("used",)) == as_eager_frame(polars_right).digest(
        ("used",)
    )


def test_polars_digest_is_deterministic_and_sensitive_to_values_order_and_schema() -> None:
    X = pl.DataFrame({"x": [1, 2, 3], "label": ["a", "b", "c"]})
    equal = pl.DataFrame({"x": [1, 2, 3], "label": ["a", "b", "c"]})
    changed = pl.DataFrame({"x": [1, 9, 3], "label": ["a", "b", "c"]})
    reordered = X[[2, 1, 0]]
    changed_schema = X.with_columns(pl.col("x").cast(pl.Float64))

    digest = as_eager_frame(X).digest()

    assert as_eager_frame(equal).digest() == digest
    assert as_eager_frame(changed).digest() != digest
    assert as_eager_frame(reordered).digest() != digest
    assert as_eager_frame(changed_schema).digest() != digest


@pytest.mark.parametrize("backend", ["pandas", "polars"])
@pytest.mark.parametrize("include_index", [True, False])
def test_empty_selected_digest_preserves_row_count_without_native_selection(
    backend: str,
    include_index: bool,
) -> None:
    two_rows = (
        pd.DataFrame({"unused": [1, 2]})
        if backend == "pandas"
        else pl.DataFrame({"unused": [1, 2]})
    )
    three_rows = (
        pd.DataFrame({"unused": [1, 2, 3]})
        if backend == "pandas"
        else pl.DataFrame({"unused": [1, 2, 3]})
    )

    assert as_eager_frame(two_rows).digest(
        (),
        include_index=include_index,
    ) != as_eager_frame(three_rows).digest(
        (),
        include_index=include_index,
    )
