"""Binomial/log models saved by superglm v0.36.0 load, predict and disclose their truncated rows.

The fixtures in ``fixtures/saved_v0_36_0`` were written by the v0.36.0 release
(commit f8e5ac01) with ``scripts/make_saved_binomlog_fixtures.py``.  Each holds
a fit whose factorization truncated a light cut, so its ``PIRLSResult`` carries
v0.36.0's ``TruncatedDirection`` records, which stored every moved row as
``rows``: at the cut's own maximum (converged, weakly identified) and away from
it (not converged, unresolved).  The records now hold their rows as runs; a
saved one must still load, list its rows and predict exactly as v0.36.0 did.
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import numpy as np
import pytest

FIXTURES = Path(__file__).parent / "fixtures" / "saved_v0_36_0"


@pytest.mark.parametrize("name", ["binomlog_light_cut_at_maximum", "binomlog_light_cut_unresolved"])
def test_a_saved_truncated_record_loads_and_predicts_exactly(name: str) -> None:
    with open(FIXTURES / f"{name}.pkl", "rb") as handle:
        record = pickle.load(handle)
    assert record["version"] == "0.36.0"
    model, frame = record["model"], record["frame"]
    predictions = np.asarray(model.predict(frame[["A", "B"]], offset=frame["off"].to_numpy()))
    np.testing.assert_array_equal(predictions, record["predictions"])
    assert model.result.converged is record["converged"]
    records = model.result.truncated_directions
    assert [list(r.rows) for r in records] == record["truncated_rows"]
    for saved in records:
        assert saved.row_count == len(saved.rows)
        assert not saved.boundary
        assert not saved.earlier
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        diagnostics = model.diagnostics()["_model"]
    key = "weakly_identified_rows" if record["at_maximum"][0] else "unresolved_rows"
    assert [entry["rows"] for entry in diagnostics[key]] == record["truncated_rows"]
    assert diagnostics["boundary_rows"] == []
    # saved again, the records round-trip in the current layout
    again = pickle.loads(pickle.dumps(model)).result.truncated_directions
    assert again == records
