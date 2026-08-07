"""Which sentinels `_validate_categorical_levels` owns, and which it reports.

The validator answers two different questions with two different errors, and the
boundary between them is narrower than "is this value missing". It owns exactly
`None` and a float NaN -- those are "the column has missing values". Every other
null-ish sentinel a column can carry (`pd.NA`, `pd.NaT`, a NaN that is not a
Python float) is not missingness it claims: it is a value that was not seen at
fit, and it comes back as an unseen-level report naming the offender.

That distinction is worth pinning because the cheap way to find missing values is
`pd.isna`, which is true for the whole family. Using it as the *answer* silently
relabels three sentinels, and using it even as a pre-filter widens the surface in
the other direction: pandas decides nullness by asking `v != v`, so a float
subclass that compares equal to itself is a NaN this validator owns and `pd.isna`
waves through. Both mistakes are pinned below.

The membership half is pinned here too, and it is the half with a mechanism
behind it: the validator needs a level *set*, never an ordering. A scan that
reaches for the set by sorting works on the homogeneous columns and fails on the
domains this library actually builds -- `order=[1..6], specials=["MISSING"]` has
no order between its ints and its str.
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


class _SelfEqualNan(float):
    """A NaN that says it equals itself, so `pd.isna` calls it clean."""

    def __new__(cls):
        return super().__new__(cls, float("nan"))

    def __eq__(self, other):
        return self is other

    def __ne__(self, other):
        return self is not other

    def __hash__(self):
        return hash("_SelfEqualNan")

    def __repr__(self):
        return "_SelfEqualNan()"


@pytest.mark.parametrize("in_domain", [False, True], ids=["outside-domain", "inside-domain"])
def test_a_nan_pandas_does_not_recognise_is_still_missing(in_domain):
    # The other direction of the `pd.isna` mistake, and the reason it cannot be
    # used even as a pre-filter. `pd.isna` asks `v != v`; this value answers
    # False and is nonetheless a float NaN, which is squarely what this
    # validator owns. Gating the narrow scan on `pd.isna` skips it and the
    # column is accepted -- silently, when the value is also a fitted level.
    sentinel = _SelfEqualNan()
    assert not pd.isna(sentinel), "fixture no longer reproduces the pd.isna blind spot"
    assert isinstance(sentinel, float) and np.isnan(sentinel)

    x = np.array(["A", "B", sentinel], dtype=object)
    known = KNOWN | {sentinel} if in_domain else KNOWN
    with pytest.raises(ValueError, match="missing values"):
        _validate_categorical_levels(x, known, context="f")


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


class _Unorderable:
    """A level that hashes and compares equal, and refuses to be ordered.

    `int` beside `str` is the mixed-type domain this library really builds, but
    it only raises TypeError, which a `try: sort / except TypeError` scan can
    swallow and recover from. This one refuses out of band, so a scan that
    sorts is caught doing it rather than quietly falling back.
    """

    def __init__(self, tag):
        self.tag = tag

    def __hash__(self):
        return hash(self.tag)

    def __eq__(self, other):
        return isinstance(other, _Unorderable) and other.tag == self.tag

    def __lt__(self, other):
        raise AssertionError("the level scan ordered the column; membership needs no order")

    __gt__ = __lt__

    def __repr__(self):
        return f"_Unorderable({self.tag!r})"


def test_a_domain_is_never_ordered_to_answer_membership():
    a, b = _Unorderable("a"), _Unorderable("b")
    x = np.array([a, b, a, b, a], dtype=object)
    _validate_categorical_levels(x, {a, b}, context="f")


def test_an_unseen_level_is_named_without_ordering_the_domain():
    # The reporting path has to survive it too: an unorderable domain must give
    # a clean unseen-level report, not a crash from the error path itself.
    a, b, c = _Unorderable("a"), _Unorderable("b"), _Unorderable("c")
    x = np.array([a, b, c], dtype=object)
    with pytest.raises(ValueError, match="unseen categorical levels") as excinfo:
        _validate_categorical_levels(x, {a, b}, context="f")
    assert "_Unorderable('c')" in str(excinfo.value)


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
