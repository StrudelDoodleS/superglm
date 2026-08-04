"""A governed fit must record the rank rule that chose its zeros.

``RankPolicy.version`` was stamped onto every ``RankDecomposition`` and
``RankInfo`` and read by nothing, so a recorded fit could not be traced back to
the rule deciding which coefficients are reported as exact zeros.  That rule is
the one thing a stored decomposition cannot re-derive for itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from superglm import Numeric, SuperGLM
from superglm.solvers.rank import SHARED_RANK_POLICY


def _fitted() -> SuperGLM:
    rng = np.random.default_rng(20260805)
    n = 200
    frame = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "z": rng.normal(size=n),
        }
    )
    y = 0.4 + 0.8 * frame["x"].to_numpy() + 0.2 * rng.normal(size=n)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"x": Numeric(), "z": Numeric()},
    )
    model.fit(frame, y)
    return model


def test_training_telemetry_records_the_rank_policy_version() -> None:
    telemetry = _fitted().training_telemetry()

    assert "rank_policy" in telemetry
    assert telemetry["rank_policy"]["version"] == SHARED_RANK_POLICY.version


def test_recorded_version_comes_from_the_fit_not_the_live_policy() -> None:
    """Read off the result, so a carried-over fit reports its own version.

    Asserting the payload tracks ``rank_info.policy_version`` rather than
    re-reading the module global is the whole point: the two agree today, and
    the test pins which one is the source.
    """
    model = _fitted()
    telemetry = model.training_telemetry()
    result = model._result

    assert telemetry["rank_policy"]["version"] == result.rank_info.policy_version
    assert telemetry["rank_policy"]["coordinate_space"] == result.rank_info.coordinate_space


def test_rank_policy_payload_is_json_serializable() -> None:
    """Telemetry is a governance artifact; a numpy scalar here breaks callers."""
    import json

    payload = _fitted().training_telemetry()["rank_policy"]
    json.dumps(payload)
    assert isinstance(payload["version"], int)
