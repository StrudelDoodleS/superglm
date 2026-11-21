"""Warning when a REML RandomEffect sits beside a separating unpenalised level.

A ``Categorical`` level with exposure but no positive response has no finite
MLE under a log link with a zero-mass family. ``fit_reml`` emits one
``SeparationWarning`` (from the post-fit level disclosure), appending an
estimated-dispersion REML clause only when the family has an estimated scale
(such as Tweedie 1 < p < 2) and a REML-estimated ``RandomEffect`` is present
(#694). Known-scale families (Poisson, NegativeBinomial) and fixed-lambda
random effects keep the single level disclosure without that clause.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from superglm import (
    Categorical,
    LambdaPolicy,
    RandomEffect,
    SeparationError,
    SeparationWarning,
    SuperGLM,
    Tweedie,
)


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


def _fit(df, y, *, family=None, features=None, separation="warn"):
    model = SuperGLM(
        family=family if family is not None else Tweedie(p=1.5),
        link="log",
        selection_penalty=0.0,
        separation=separation,
        features=features
        if features is not None
        else {"band": Categorical(base="most_exposed"), "cell": RandomEffect()},
    )
    model.fit_reml(df, y, max_reml_iter=2, max_pirls_iter=400)
    return model


def _sep_messages(caught):
    return [str(item.message) for item in caught if issubclass(item.category, SeparationWarning)]


def test_warns_on_tweedie_random_effect_beside_claim_free_level():
    """#694b: estimated-scale Tweedie REML emits one warning with level totals and RE clause."""
    df, y = _frame(claim_free_level=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _fit(df, y)
    messages = _sep_messages(caught)
    assert len(messages) == 1
    message = messages[0]
    assert "'band' level 'z'" in message
    assert "With an estimated dispersion" in message
    assert "'cell'" in message
    assert "Gaussian-style" not in message
    assert "RandomEffect variance component(s)" not in message


def test_warns_for_poisson_without_the_tweedie_scale_clause():
    """#694a: known-scale Poisson REML emits one warning with no dispersion/RE clause."""
    df, y = _frame(claim_free_level=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _fit(df, np.rint(y / 25.0), family="poisson")
    messages = _sep_messages(caught)
    assert len(messages) == 1
    message = messages[0]
    assert "'band' level 'z'" in message
    assert "estimated dispersion" not in message
    assert "variance component" not in message


def test_poisson_reml_reproducer_warns_once_and_keeps_lambda_unchanged():
    """#694 reproducer: one SeparationWarning on Poisson REML and invariant RE lambda."""
    rng = np.random.default_rng(0)
    n = 20000
    lev = rng.choice(list("ABCDE"), n, p=[0.4, 0.3, 0.2, 0.098, 0.002])
    grp = rng.integers(0, 30, n).astype(str)
    u = rng.normal(0, 0.2, 30)
    y = rng.poisson(0.08 * np.where(lev == "B", 1.3, 1.0) * np.exp(u[grp.astype(int)])).astype(
        float
    )
    y[lev == "E"] = 0.0
    X = pd.DataFrame({"lev": lev, "grp": grp})

    m = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        features={"lev": Categorical(base="A"), "grp": RandomEffect()},
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m.fit_reml(X, y)
    messages = _sep_messages(caught)
    assert len(messages) == 1
    assert "'lev' level 'E'" in messages[0]
    assert "variance component" not in messages[0]
    assert "estimated dispersion" not in messages[0]

    keep = lev != "E"
    m_no_e = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        features={"lev": Categorical(base="A"), "grp": RandomEffect()},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m_no_e.fit_reml(X[keep].reset_index(drop=True), y[keep])

    def re_lambda(model):
        return [r.smoothing_lambdas[0][1] for r in model.summary()._coef_rows if r.structured_kind][
            0
        ]

    assert re_lambda(m) == pytest.approx(re_lambda(m_no_e), rel=1e-7)


@pytest.mark.pins_behaviour
def test_no_warning_when_every_level_has_a_positive_row():
    """Control: pins that a REML fit with no claim-free levels emits no SeparationWarning."""
    df, y = _frame(claim_free_level=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _fit(df, y)
    assert _sep_messages(caught) == []


@pytest.mark.pins_behaviour
def test_no_warning_when_the_random_effect_lambda_is_fixed():
    """Control: pins that a fixed-lambda RandomEffect emits one level warning without the RE clause."""
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
    messages = _sep_messages(caught)
    assert len(messages) == 1
    assert "'band' level 'z'" in messages[0]
    assert "estimated dispersion" not in messages[0]
    assert "variance component" not in messages[0]


@pytest.mark.pins_behaviour
def test_no_warning_for_gaussian_family():
    """Control: pins that Gaussian identity-link REML emits no SeparationWarning on zero levels."""
    df, y = _frame(claim_free_level=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _fit(df, y, family="gaussian")
    assert _sep_messages(caught) == []


def test_the_hazard_is_a_separation_warning_and_obeys_the_separation_seam():
    """``warn`` emits one warning, ``ignore`` emits none, and ``error`` refuses without warning."""
    df, y = _frame(claim_free_level=True)

    with warnings.catch_warnings(record=True) as caught_warn:
        warnings.simplefilter("always")
        _fit(df, y, separation="warn")
    messages = _sep_messages(caught_warn)
    assert len(messages) == 1 and "With an estimated dispersion" in messages[0]

    with warnings.catch_warnings(record=True) as caught_ign:
        warnings.simplefilter("always")
        _fit(df, y, separation="ignore")
    assert _sep_messages(caught_ign) == []

    with warnings.catch_warnings(record=True) as caught_err:
        warnings.simplefilter("always")
        with pytest.raises(SeparationError, match="'band' level 'z'"):
            _fit(df, y, separation="error")
    assert _sep_messages(caught_err) == []


@pytest.mark.pins_behaviour
def test_the_hazard_reads_levels_after_grouping_is_applied():
    """Control: pins that collapsing the claim-free level with grouping= silences SeparationWarning."""
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
    assert _sep_messages(caught) == []
