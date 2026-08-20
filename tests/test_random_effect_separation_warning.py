"""Warning when a REML RandomEffect sits beside a separating unpenalised level.

A ``Categorical`` level with exposure but no positive response has no finite
MLE under a log link with a zero-mass family, and the REML criterion goes
nearly flat in a neighbouring random-effect variance (issue #339). fit_reml
warns on that configuration; these tests pin the trigger and its negative
space.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from superglm import Categorical, LambdaPolicy, RandomEffect, SuperGLM, Tweedie

_MATCH = "claim-free"


def _frame(claim_free_level: bool) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(7)
    n = 60
    band = np.array(["u", "v", "w"])[rng.integers(0, 3, size=n)]
    band[:6] = "z"  # a thin level; claim-free unless given a positive row
    cell = np.array([f"c{i}" for i in rng.integers(0, 6, size=n)])
    y = np.where(rng.random(n) < 0.35, rng.gamma(1.0, 50.0, size=n), 0.0)
    y[:6] = 0.0
    if not claim_free_level:
        y[0] = 25.0
    y[6] = 40.0  # keep the model as a whole fittable
    return pd.DataFrame({"band": band, "cell": cell}), y


def _fit(df, y, *, family=None, features=None):
    model = SuperGLM(
        family=family if family is not None else Tweedie(p=1.5),
        link="log",
        selection_penalty=0.0,
        features=features
        if features is not None
        else {"band": Categorical(base="most_exposed"), "cell": RandomEffect()},
    )
    model.fit_reml(df, y, max_reml_iter=2, max_pirls_iter=400)
    return model


def _messages(caught):
    return [str(item.message) for item in caught if _MATCH in str(item.message)]


def test_warns_on_tweedie_random_effect_beside_claim_free_level():
    df, y = _frame(claim_free_level=True)
    with pytest.warns(UserWarning, match=_MATCH) as caught:
        _fit(df, y)
    message = _messages(caught)[0]
    assert "'cell'" in message
    assert "'band'" in message
    assert "Tweedie dispersion" in message


def test_warns_for_poisson_without_the_tweedie_scale_clause():
    df, y = _frame(claim_free_level=True)
    with pytest.warns(UserWarning, match=_MATCH) as caught:
        _fit(df, np.rint(y / 25.0), family="poisson")
    message = _messages(caught)[0]
    assert "Tweedie dispersion" not in message


def test_no_warning_when_every_level_has_a_positive_row():
    df, y = _frame(claim_free_level=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _fit(df, y)
    assert _messages(caught) == []


def test_no_warning_when_the_random_effect_lambda_is_fixed():
    df, y = _frame(claim_free_level=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _fit(
            df,
            y,
            features={
                "band": Categorical(base="most_exposed"),
                "cell": RandomEffect(lambda_policy=LambdaPolicy.fixed(5.0)),
            },
        )
    assert _messages(caught) == []


def test_no_warning_for_gaussian_family():
    df, y = _frame(claim_free_level=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _fit(df, y, family="gaussian")
    assert _messages(caught) == []


def test_the_hazard_is_a_separation_warning_and_obeys_the_separation_seam():
    """This release added ``SeparationWarning`` for exactly this proposition.

    A model-risk pipeline filtering on it should catch all three separation
    diagnostics, and ``separation="ignore"`` should quiet all three.  Raised as
    a bare ``UserWarning`` outside the seam, this one is invisible to both.
    """
    from superglm import SeparationWarning

    df, y = _frame(claim_free_level=True)

    with pytest.warns(SeparationWarning, match=_MATCH):
        _fit(df, y)

    model = SuperGLM(
        family=Tweedie(p=1.5),
        link="log",
        selection_penalty=0.0,
        separation="ignore",
        features={"band": Categorical(base="most_exposed"), "cell": RandomEffect()},
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit_reml(df, y, max_reml_iter=2, max_pirls_iter=400)
    assert _messages(caught) == [], 'separation="ignore" must quiet the hazard too'


def test_the_hazard_reads_levels_after_grouping_is_applied():
    """Taking the warning's own advice must silence it.

    The message tells the caller to group the claim-free level.  Scanning the
    RAW column ignores ``grouping=``, so the collapsed level is still counted
    and the warning fires again -- on a model where it no longer has a
    coefficient.
    """
    from superglm import collapse_levels

    df, y = _frame(claim_free_level=True)
    grouping = collapse_levels(df["band"], groups={"uz": ["u", "z"], "v": ["v"], "w": ["w"]})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _fit(
            df,
            y,
            features={
                "band": Categorical(base="most_exposed", grouping=grouping),
                "cell": RandomEffect(),
            },
        )
    assert _messages(caught) == [], "a collapsed level no longer separates and must not be named"
