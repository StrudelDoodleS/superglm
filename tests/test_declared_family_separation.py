"""Family-owned recession declarations must reach the public separation policy."""

import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from superglm import LogLogisticLSS, LomaxLSS, SuperLSS, cat
from superglm.diagnostics.separation import SeparationError, SeparationWarning
from superglm.distributional.families._links import BoundedLogitLink
from superglm.distributional.observation import ResponseBounds
from superglm.distributional.separation import censored_recession
from superglm.links import LogitLink, LogLink, PowerLink, SqrtLink


@pytest.mark.parametrize("cls", [LomaxLSS, LogLogisticLSS])
@pytest.mark.parametrize("side", ["right", "left"])
def test_all_positive_censored_intercept_policy(cls, side):
    family = cls()
    frame = pd.DataFrame({"unused": np.zeros(12)})
    y = np.ones(12)

    def model(policy):
        return SuperLSS(family, family.scale(), family.shape(), separation=policy)

    # At every finite positive scale the one-sided probability is below its
    # supremum. Removing the family hook used to report silent convergence.
    with pytest.raises(SeparationError, match=rf"'scale'.*{side}-censored"):
        model("error").fit(frame, y, censored=side)
    with pytest.warns(SeparationWarning, match=rf"'scale'.*{side}-censored"):
        model("warn").fit(frame, y, censored=side, max_inner_iter=25)
    with warnings.catch_warnings():
        warnings.simplefilter("error", SeparationWarning)
        model("ignore").fit(frame, y, censored=side, max_inner_iter=25)


@pytest.mark.parametrize("cls", [LomaxLSS, LogLogisticLSS])
@pytest.mark.parametrize("side", ["right", "left"])
@pytest.mark.parametrize("truncation", [None, (0.2, 5.0), (0.2, np.inf)])
def test_censored_categorical_scale_level_with_truncation(cls, side, truncation):
    family = cls()
    frame = pd.DataFrame({"g": ["a"] * 12 + ["b"] * 6})
    y = np.r_[np.tile([0.5, 1.0, 2.0], 4), np.ones(6)]
    model = SuperLSS(family, family.scale(cat("g", base="a")), family.shape(), separation="error")
    with pytest.raises(SeparationError, match=rf"'scale:g'.*only {side}-censored rows: 'b'"):
        model.fit(frame, y, censored=["none"] * 12 + [side] * 6, truncation=truncation)


@pytest.mark.parametrize("cls", [LomaxLSS, LogLogisticLSS])
@pytest.mark.parametrize(
    "link,expected",
    [
        (LogLink(), [True, True, False]),
        (LogitLink(), [True, True, False]),
        # The configured/clipped link is outside the scan's audited open-link
        # classes. It is left unflagged, as for the existing families.
        (BoundedLogitLink(0.1, 10.0), [False, False, False]),
        (SqrtLink(), [False, True, False]),
        (PowerLink(-1.0), [True, False, False]),
    ],
)
def test_censoring_recession_respects_link_endpoints(cls, link, expected):
    bounds = ResponseBounds.from_arrays(np.full(3, 0.2), np.full(3, 5.0), np.array([-1, 1, 0]))
    compiled = [
        SimpleNamespace(parameter_index=j, link=lnk) for j, lnk in enumerate((link, LogLink()))
    ]
    actual = censored_recession(cls(), bounds, compiled)
    np.testing.assert_array_equal(actual[:, 0], expected)
    assert not actual[:, 1].any()


@pytest.mark.parametrize("cls", [LomaxLSS, LogLogisticLSS])
def test_exact_rows_anchor_a_censored_level(cls):
    family = cls()
    frame = pd.DataFrame({"g": ["a"] * 12 + ["b"] * 12})
    y = np.tile([0.3, 1.0, 4.0, 2.0], 6)
    with warnings.catch_warnings():
        warnings.simplefilter("error", SeparationWarning)
        SuperLSS(family, family.scale(cat("g")), family.shape(), separation="error").fit(
            frame, y, censored=np.tile(["none", "right", "left", "none"], 6)
        )
