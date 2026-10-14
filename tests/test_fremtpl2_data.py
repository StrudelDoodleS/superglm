"""The Wüthrich & Merz claim recount behind every corrected freMTPL2 frame.

``examples/fremtpl2_data.py`` is the one copy of the rule; the docs and the
real-data suites load it through ``tests/_datasets.load_freq_corrected``.  The
synthetic tests pin each clause of the rule; the real-book test pins the
published totals (678,007 policies, 26,383 claims).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from . import _datasets

_DATA = _datasets._fremtpl2_data()

_FREQ_SKIP_REASON = _datasets.skip_reason("freMTPL2freq.parquet")
_SEV_SKIP_REASON = _datasets.skip_reason("freMTPL2sev.parquet")

BOOK_SKIP = pytest.mark.skipif(
    _FREQ_SKIP_REASON is not None or _SEV_SKIP_REASON is not None,
    reason=_FREQ_SKIP_REASON or _SEV_SKIP_REASON or "",
)


def _freq() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "IDpol": [1.0, 2.0, 3.0, 30000.0, 30001.0],
            "ClaimNb": [1, 2, 1, 0, 1],
            "Exposure": [0.5, 1.3, 0.2, 2.0, 0.9],
            "VehAge": [0, 1, 2, 3, 4],
        }
    )


def _sev() -> pd.DataFrame:
    # policy 2: one record; 30000: six records; 30001: one; 99999 has no policy
    ids = [2.0, 30000.0, 30000.0, 30000.0, 30000.0, 30000.0, 30000.0, 30001.0, 99999.0]
    return pd.DataFrame({"IDpol": ids, "ClaimAmount": [100.0] + [10.0] * 6 + [7.5, 1e6]})


def test_counts_come_from_the_severity_file_for_every_policy():
    out = _DATA.recount_claims(_freq(), _sev())
    # 1 and 3 had a recorded claim and no severity record: now 0.  2 keeps one
    # (not the recorded 2).  30000 has six records and is dropped.  The
    # severity record with no policy is ignored.
    assert out["IDpol"].tolist() == [1.0, 2.0, 3.0, 30001.0]
    assert out["ClaimNb"].tolist() == [0, 1, 0, 1]
    assert out["ClaimNb"].dtype == np.int64
    assert out["ClaimTotal"].tolist() == [0.0, 100.0, 0.0, 7.5]


def test_exposure_is_capped_at_one_year_and_the_inputs_are_untouched():
    freq, sev = _freq(), _sev()
    before = freq.copy()
    out = _DATA.recount_claims(freq, sev)
    assert out["Exposure"].tolist() == [0.5, 1.0, 0.2, 0.9]
    pd.testing.assert_frame_equal(freq, before)
    assert out.index.tolist() == list(range(len(out)))


def test_a_repeated_policy_is_refused():
    freq = pd.concat([_freq(), _freq().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="repeated IDpol"):
        _DATA.recount_claims(freq, _sev())


@BOOK_SKIP
def test_the_corrected_book_has_the_published_totals():
    book = _datasets.load_freq_corrected()
    assert len(book) == 678_007
    assert int(book["ClaimNb"].sum()) == 26_383
    assert book["ClaimNb"].max() == 5
    assert book["Exposure"].max() == 1.0
    block = book["IDpol"] <= _DATA.CORRUPTED_IDPOL_MAX
    # The corrupted block keeps its 9,139 policies but only the claims the
    # severity file records for them.
    assert int(block.sum()) == 9_139
    assert int(book.loc[block, "ClaimNb"].sum()) == 302
