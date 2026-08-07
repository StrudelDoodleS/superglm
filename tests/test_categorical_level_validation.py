"""Which sentinels `_validate_categorical_levels` owns, and which it reports.

The validator answers two different questions with two different errors, and the
boundary between them is narrower than "is this value missing". It owns exactly
`None` and a float NaN -- those are "the column has missing values". Every other
null-ish sentinel a column can carry (`pd.NA`, `pd.NaT`, a NaN that is not a
Python float) is not missingness it claims: it is a value that was not seen at
fit, and it comes back as an unseen-level report naming the offender.

That distinction is worth pinning because the cheap way to find missing values is
`pd.isna`, which is true for the whole family. Using it as the *answer* silently
relabels three sentinels; using it only as a pre-filter -- a clean vector proves
the narrow test is also clean, a dirty one still pays for the narrow test -- keeps
the boundary where it is. These tests fail against the first and pass against the
second, so they are what makes that a checked claim rather than a comment.

The membership half is pinned here too: the validator needs a level *set*, never
an ordering, and a domain that mixes types has no ordering to give it.
"""

from __future__ import annotations

import decimal

import numpy as np
import pandas as pd
import pytest

from superglm.features.categorical import _validate_categorical_levels

KNOWN = {"A", "B"}


# ── The sentinels the validator owns ────────────────────────────────


@pytest.mark.parametrize(
    "sentinel",
    [
        pytest.param(None, id="none"),
        pytest.param(float("nan"), id="python-float-nan"),
        pytest.param(np.float64("nan"), id="numpy-float64-nan"),
    ],
)
def test_none_and_float_nan_are_reported_as_missing_values(sentinel):
    x = np.array(["A", "B", sentinel], dtype=object)
    with pytest.raises(ValueError, match="missing values"):
        _validate_categorical_levels(x, KNOWN, context="f")


def test_a_float_column_of_nan_is_reported_as_missing_values():
    # Not an object column: the pre-filter has to answer for the homogeneous
    # float case too, where `pd.isna` is a plain `np.isnan` rather than an
    # element-wise null check.
    x = np.array([1.0, 2.0, np.nan])
    with pytest.raises(ValueError, match="missing values"):
        _validate_categorical_levels(x, {1.0, 2.0}, context="f")


# ── The sentinels it does not own ───────────────────────────────────


@pytest.mark.parametrize(
    "sentinel",
    [
        pytest.param(pd.NA, id="pd-NA"),
        pytest.param(pd.NaT, id="pd-NaT"),
        pytest.param(np.float32("nan"), id="numpy-float32-nan"),
        pytest.param(decimal.Decimal("NaN"), id="decimal-nan"),
    ],
)
def test_other_null_sentinels_are_reported_as_unseen_levels(sentinel):
    # `pd.isna` is true for every one of these. None of them is `None` or a
    # Python float, so none is missingness this validator claims -- each is an
    # unseen level, and the report has to name it rather than blame the column.
    x = np.array(["A", "B", sentinel], dtype=object)
    with pytest.raises(ValueError, match="unseen categorical levels") as excinfo:
        _validate_categorical_levels(x, KNOWN, context="f")
    assert "missing values" not in str(excinfo.value)


def test_a_null_sentinel_inside_the_fitted_domain_is_accepted():
    # The strongest form of the same boundary: when the sentinel IS a fitted
    # level, the validator must not object at all. A `pd.isna`-as-answer
    # implementation rejects a domain it was itself built from.
    x = np.array(["A", "B", pd.NA], dtype=object)
    _validate_categorical_levels(x, KNOWN | {pd.NA}, context="f")


# ── Membership needs no ordering ────────────────────────────────────


def test_a_mixed_type_domain_validates_without_an_ordering():
    # `order=[1..6], specials=["MISSING"]` in column form. There is no order
    # between an int and a str, and the validator does not need one.
    x = np.array([1, 2, 3, "MISSING"], dtype=object)
    _validate_categorical_levels(x, {1, 2, 3, 4, 5, 6, "MISSING"}, context="f")


def test_a_mixed_type_domain_still_rejects_an_unseen_level():
    x = np.array([1, 2, "NOT_A_LEVEL"], dtype=object)
    with pytest.raises(ValueError, match="unseen categorical levels") as excinfo:
        _validate_categorical_levels(x, {1, 2, "MISSING"}, context="f")
    assert "NOT_A_LEVEL" in str(excinfo.value)


@pytest.mark.parametrize(
    "x",
    [
        pytest.param(np.array(["b", "a", "b"]), id="unicode"),
        pytest.param(np.array(["b", "a", "b"], dtype=object), id="object"),
        pytest.param(pd.Series(["b", "a", "b"]), id="series"),
        pytest.param(pd.Series(["b", "a", "b"]).astype("category"), id="categorical-series"),
        pytest.param(np.array([], dtype=object), id="empty"),
    ],
)
def test_every_clean_column_shape_passes(x):
    _validate_categorical_levels(x, {"a", "b"}, context="f")


def test_a_repeated_level_is_reported_once():
    # The observed side is a set, so a level unseen on 10,000 rows is named once.
    x = np.array(["A"] * 5 + ["Z"] * 5, dtype=object)
    with pytest.raises(ValueError, match="unseen categorical levels") as excinfo:
        _validate_categorical_levels(x, KNOWN, context="f")
    assert str(excinfo.value).count("'Z'") == 1
