"""The retained-data guard of a Polars frame survives a change of Polars version.

Polars documents its row hashes as unstable across versions (Polars 2.0 even
removed three of the four seeds), so a fit-time fingerprint built on them cannot
be recomputed after an upgrade.  The current fingerprint hashes the column values
with an encoding of our own, which is the same under every Polars.

``fixtures/saved_polars_guard`` holds one small Gaussian random-effect model per
fingerprint, both pickled under Polars 1.43.2 by
``scripts/make_polars_guard_fixtures.py``: ``legacy`` by commit 931548cd (the
fingerprint is Polars' four-seed ``hash_rows``, with no scheme tag) and
``values`` by the release that introduced the scheme tag.  Run this file under
Polars 1.x and 2.x; each test says which it expects.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

import superglm as sg
import superglm._frame as frame_module
from superglm._frame import POLARS_DIGEST_SCHEME, as_eager_frame
from superglm.model import fit_data_guard as guard_module
from superglm.model.fit_data_guard import FitDataGuard, require_unchanged_fit_data

FIXTURES = Path(__file__).parent / "fixtures" / "saved_polars_guard"
POLARS_2 = int(pl.__version__.split(".")[0]) >= 2
NOTICE = "fit-time digest was written by Polars 1.x"


def _load(kind: str) -> dict:
    with open(FIXTURES / f"re_retained_{kind}.pkl", "rb") as handle:
        return pickle.load(handle)


def _mixed_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "i": pl.Series([1, None, -3], dtype=pl.Int32),
            "f": [0.5, float("nan"), None],
            "b": [True, None, False],
            "s": ["a", None, "héé"],
            "c": pl.Series(["x", "y", None], dtype=pl.Categorical),
            "e": pl.Series(["y", "x", "x"], dtype=pl.Enum(["x", "y"])),
            "d": [dt.date(2020, 1, 2), None, dt.date(1999, 12, 31)],
            "t": pl.Series(
                [dt.datetime(2020, 1, 1, 5), None, dt.datetime(2021, 3, 4)],
                dtype=pl.Datetime("ms", "UTC"),
            ),
            "l": [[1], [2, 3], None],
            "n": pl.Series([None, None, None], dtype=pl.Null),
            "bin": [b"a", None, b"cc"],
            "dec": pl.Series(["1.5", None, "2.25"]).cast(pl.Decimal(10, 2)),
        }
    )


def test_the_digest_of_a_frame_is_the_same_under_every_polars() -> None:
    """Pinned bytes, recorded under Polars 1.43.2 and 2.0.0 alike: nothing in the
    digest comes from Polars' hash, so no Polars version can move it."""
    frame = as_eager_frame(_mixed_frame())

    assert frame.digest().hex() == "b7454a2df64f6ada464dda5987a8d9c4"
    assert frame.digest(("i", "s")).hex() == "f0b904fa66e06f99b4ebd11b4b26846b"
    assert as_eager_frame(pl.DataFrame({"u": [1, 2]})).digest(()).hex() == (
        "0e63eee0ae32ed826aa9c1498f51d1b3"
    )


def test_the_digest_separates_values_nulls_text_and_types() -> None:
    base = pl.DataFrame({"x": [1, 2, 3], "t": ["ab", "c", None]})
    digest = as_eager_frame(base).digest()

    for changed in (
        base.with_columns(pl.col("x").cast(pl.Int32)),
        base.with_columns(pl.Series("x", [1, 2, None])),
        base.with_columns(pl.Series("t", ["a", "bc", None])),
        base.with_columns(pl.Series("t", ["ab", "c", ""])),
        base.with_columns(pl.col("t").cast(pl.Categorical)),
    ):
        assert as_eager_frame(changed).digest() != digest


def test_a_guard_written_now_names_its_scheme_and_verifies() -> None:
    X = pl.DataFrame({"x": np.linspace(-1.0, 1.0, 20), "g": ["a", "b"] * 10})
    y = np.arange(20.0)

    guard = FitDataGuard.capture(X, y)

    assert guard.x_digest_scheme == POLARS_DIGEST_SCHEME
    assert guard.matches_retained_values(X.clone(), y)
    assert not guard.matches_retained_values(X.with_columns(pl.col("x") + 1.0), y)


def test_a_guard_written_under_polars_1_verifies_under_this_polars() -> None:
    """The cross-version regression for the scheme: a model fitted and pickled
    under Polars 1.43.2 reports under this Polars, without a warning, from the
    digest it stored, and refuses a retained frame that was edited."""
    record = _load("values")
    model = record["model"]
    guard = model._fit_data_guard
    assert record["polars"].startswith("1.")
    assert guard.x_digest_scheme == POLARS_DIGEST_SCHEME
    assert as_eager_frame(model._fit_X_ref).digest(guard.x_columns) == guard.x_digest

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        table = model.random_effects("group").table
        edited = model._fit_X_ref.with_columns(pl.Series("group", ["a"] * 60))
        with pytest.raises(RuntimeError, match="retained fit data were mutated"):
            require_unchanged_fit_data(model, edited, model._fit_y_ref)

    pd.testing.assert_frame_equal(table, record["table"], rtol=1e-10)


def test_a_legacy_guard_verifies_on_polars_1_and_warns_once_on_polars_2(monkeypatch) -> None:
    """A guard from before the scheme tag holds Polars' four-seed row hash.  On
    Polars 1.x it is recomputed, as before (and still refuses an edited frame);
    on Polars 2.x, which cannot recompute it, the report is produced from the
    unchanged data with one warning that names the cause."""
    monkeypatch.setattr(guard_module, "_UNVERIFIABLE_WARNED", False)
    record = _load("legacy")
    model = record["model"]
    assert model._fit_data_guard.x_digest_scheme is None
    assert record["polars"].startswith("1.")

    if POLARS_2:
        with pytest.warns(UserWarning, match=NOTICE) as caught:
            table = model.random_effects("group").table
        assert len([w for w in caught if NOTICE in str(w.message)]) == 1
        message = str(caught[0].message)
        assert "cannot be checked under this Polars" in message
        assert "results assume the retained data are unchanged" in message
        assert "refit to restore the check" in message
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            table = model.random_effects("group").table
            edited = model._fit_X_ref.with_columns(pl.Series("group", ["a"] * 60))
            with pytest.raises(RuntimeError, match="retained fit data were mutated"):
                require_unchanged_fit_data(model, edited, model._fit_y_ref)

    pd.testing.assert_frame_equal(table, record["table"], rtol=1e-10)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.random_effects("group")


def test_a_legacy_guard_that_cannot_be_recomputed_proceeds_with_one_warning(monkeypatch) -> None:
    """The same path on any Polars: where the old digest is unavailable the guard
    must not report the unchanged data as mutated."""
    monkeypatch.setattr(guard_module, "_UNVERIFIABLE_WARNED", False)

    def unavailable(self, columns=None):
        raise frame_module.LegacyDigestUnavailableError("no four-seed hash_rows")

    monkeypatch.setattr(frame_module.EagerFrame, "legacy_polars_digest", unavailable)
    model = _load("legacy")["model"]

    with pytest.warns(UserWarning, match=NOTICE):
        model.random_effects("group")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.random_effects("group")


def test_a_guard_from_a_scheme_this_release_does_not_know_is_not_a_mutation(monkeypatch) -> None:
    monkeypatch.setattr(guard_module, "_UNVERIFIABLE_WARNED", False)
    X = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
    y = np.array([1.0, 2.0, 3.0])
    guard = FitDataGuard.capture(X, y)
    future = FitDataGuard(
        x_backend=guard.x_backend,
        x_digest=b"\x00" * 16,
        y_snapshot=guard.y_snapshot,
        x_columns=guard.x_columns,
        x_digest_scheme="polars-values-v9",
    )

    with pytest.warns(UserWarning, match="polars-values-v9"):
        assert future.matches_retained_values(X, y, own_frame=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert not future.matches_retained_values(X.clone(), y)


def test_a_legacy_guard_does_not_vouch_for_a_frame_the_caller_supplies(monkeypatch) -> None:
    """The unverifiable-digest allowance covers the model's own retained frame
    only.  Rows the caller passes to ``random_effects`` can be anything, so where
    the old digest cannot be recomputed they are refused as before, not read
    with the fitted posterior errors (here 58/1/1 groups against the 20/20/20
    that were fitted).  The retained frame still reports, with its one warning."""
    monkeypatch.setattr(guard_module, "_UNVERIFIABLE_WARNED", False)
    model = _load("legacy")["model"]
    changed = model._fit_X_ref.with_columns(pl.Series("group", ["a"] * 58 + ["b", "c"]))

    with pytest.raises(ValueError, match="must reproduce fitted training rows"):
        model.random_effects(
            "group",
            X=changed,
            y=model._fit_y_ref,
            sample_weight=model._fit_weights,
            offset=model._fit_offset,
        )
    with pytest.warns(UserWarning, match=NOTICE) if POLARS_2 else warnings.catch_warnings():
        model.random_effects("group")


def _legacy_digest_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(guard_module, "_UNVERIFIABLE_WARNED", False)

    def unavailable(self, columns=None):
        raise frame_module.LegacyDigestUnavailableError("no four-seed hash_rows")

    monkeypatch.setattr(frame_module.EagerFrame, "legacy_polars_digest", unavailable)


def _as_legacy(guard):
    return dataclasses.replace(guard, x_digest_scheme=None)


# Each caller below passes the guard ``own_frame=X is <retained frame>``.  The
# tests hand it an equal-length frame that is not the retained one, which an
# unverifiable digest must not vouch for.


def test_a_legacy_guard_refuses_a_copy_of_the_retained_frame(monkeypatch) -> None:
    _legacy_digest_unavailable(monkeypatch)
    model = _load("legacy")["model"]

    with pytest.raises(RuntimeError, match="retained fit data were mutated"):
        require_unchanged_fit_data(model, model._fit_X_ref.clone(), model._fit_y_ref)
    with pytest.warns(UserWarning, match=NOTICE):
        require_unchanged_fit_data(model, model._fit_X_ref, model._fit_y_ref)


def test_a_legacy_geometry_guard_does_not_vouch_for_rows_of_a_released_fit(monkeypatch) -> None:
    """With the design released, only a guard match lets ``metrics`` read the
    fit's covariance; rows it cannot verify are refused, not scored with it."""
    _legacy_digest_unavailable(monkeypatch)
    rng = np.random.default_rng(518)
    X = pl.DataFrame({"x": np.linspace(0.0, 1.0, 120)})
    y = 1.0 + X["x"].to_numpy() + rng.normal(0.0, 0.1, 120)
    model = sg.SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        retain_fit_state=False,
        features={"x": sg.Spline(n_knots=6)},
    ).fit(X, y)
    model._fit_geometry_guard = _as_legacy(model._fit_geometry_guard)

    with pytest.raises(RuntimeError, match="does not match the fit geometry"):
        model.metrics(X.with_columns(pl.col("x").reverse()), y).leverage


def test_a_legacy_geometry_guard_does_not_lend_the_fit_covariance_to_other_rows(
    monkeypatch,
) -> None:
    """A structured fit reads its retained covariance and design for rows the
    guard matches.  Rows it cannot verify are evaluated on their own design,
    exactly as a guard that can be checked evaluates them."""
    _legacy_digest_unavailable(monkeypatch)
    rng = np.random.default_rng(518)
    X = pl.DataFrame({"group": ["a"] * 58 + ["b"] * 31 + ["c"] * 31})
    y = rng.normal(size=120)
    model = sg.SuperGLM(
        family="gaussian", direct_solve="structured", features={"group": sg.RandomEffect()}
    ).fit_reml(X, y)
    changed = X.with_columns(pl.col("group").reverse())
    reference = model.metrics(changed, y).leverage

    model._fit_geometry_guard = _as_legacy(model._fit_geometry_guard)

    np.testing.assert_array_equal(model.metrics(changed, y).leverage, reference)


def test_a_legacy_guard_does_not_pair_inherited_weights_with_other_rows(monkeypatch) -> None:
    _legacy_digest_unavailable(monkeypatch)
    rng = np.random.default_rng(518)
    X = pl.DataFrame({"x": rng.uniform(size=200), "z": rng.uniform(size=200)})
    y = rng.normal(size=200)
    model = sg.SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"x": sg.Spline(n_knots=6), "z": sg.Spline(n_knots=6)},
    ).fit(X, y, sample_weight=rng.uniform(0.5, 2.0, size=200))
    model._fit_data_guard = _as_legacy(model._fit_data_guard)

    with pytest.raises(ValueError, match="inherited from the fit"):
        model.screen_interactions(X.with_columns(pl.col("x").reverse()), y)


def test_fitting_on_a_polars_frame_writes_a_guard_that_round_trips() -> None:
    rng = np.random.default_rng(3)
    X = pl.DataFrame({"group": ["a", "b", "c"] * 20})
    y = rng.normal(size=60)
    model = sg.SuperGLM(family="gaussian", features={"group": sg.RandomEffect()}).fit_reml(X, y)

    restored = pickle.loads(pickle.dumps(model))

    assert restored._fit_data_guard.x_digest_scheme == POLARS_DIGEST_SCHEME
    assert restored._fit_geometry_guard.x_digest_scheme == POLARS_DIGEST_SCHEME
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        restored.random_effects("group")


def test_a_fit_digests_each_retained_column_once_for_both_guards(monkeypatch) -> None:
    """The data guard and the geometry guard fingerprint the same columns of the
    same frame, so the second reuses the first's digest."""
    seen: list[str] = []
    original = frame_module._update_polars_column

    def counting(digest, name, column, dtype):
        seen.append(name)
        return original(digest, name, column, dtype)

    monkeypatch.setattr(frame_module, "_update_polars_column", counting)
    X = pl.DataFrame({"group": ["a", "b", "c"] * 20})
    model = sg.SuperGLM(family="gaussian", features={"group": sg.RandomEffect()})
    model.fit_reml(X, np.arange(60.0) % 7)

    assert seen == ["group"]
    assert model._fit_data_guard.x_digest == model._fit_geometry_guard.x_digest


def test_a_text_column_digests_the_same_in_any_chunk_size(monkeypatch) -> None:
    """Text is fed in slices of rows so a huge column is never joined whole; the
    byte stream, and so the digest, must not depend on the slice size."""
    words = ["", "a", "bc", "déjà vu", "日本語", "x" * 40]
    cycle = [words[i % len(words)] if i % 5 else None for i in range(23)]  # 23 rows, nulls
    X = pl.DataFrame(
        {
            "s": pl.Series(cycle, dtype=pl.String),
            "c": pl.Series(cycle, dtype=pl.Categorical),
            "e": pl.Series(cycle, dtype=pl.Enum(words)),
        }
    )
    whole = as_eager_frame(X).digest()
    assert as_eager_frame(X).digest() == whole

    for rows in (1, 2, 5, 7, 22, 23, 24):  # a partial last slice for all but 1, 23
        monkeypatch.setattr(frame_module, "_TEXT_CHUNK_ROWS", rows)
        assert as_eager_frame(X).digest() == whole, rows
