"""SlowFitWarning: an exact REML fit says in advance when it will take minutes."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from superglm import Categorical, RandomEffect, SlowFitWarning, Spline, SuperGLM
from superglm.diagnostics import fit_cost
from superglm.diagnostics.fit_cost import exact_reml_cost_message
from superglm.distributions import Binomial, Gaussian, Poisson
from superglm.links import IdentityLink, LogitLink, LogLink

# freMTPL2 (542,410 training rows), exact fit times after the one-pass
# weight-derivative correction: mains only, then mains + 1, 3 and 10 pairs.
BOOK_MAINS = dict(n_rows=542_410, n_coef=67)  # 7-12 s
BOOK_ONE_PAIR = dict(n_rows=542_410, n_coef=192)  # 27 s, 9 s binned
BOOK_THREE_PAIRS = dict(n_rows=542_410, n_coef=478)  # 95 s, 19 s binned
BOOK_TEN_PAIRS = dict(n_rows=542_410, n_coef=1_565)  # 405 s, 90 s binned


@pytest.mark.parametrize("book", [BOOK_THREE_PAIRS, BOOK_TEN_PAIRS])
def test_rule_fires_on_the_book_fits_that_take_minutes(book):
    message = exact_reml_cost_message(**book, distribution=Poisson(), link=LogLink())
    assert message is not None
    assert "542,410 rows" in message and f"{book['n_coef']} coefficients" in message
    assert "discrete=True" in message


@pytest.mark.parametrize("book", [BOOK_MAINS, BOOK_ONE_PAIR])
def test_rule_is_silent_on_the_book_fits_that_take_seconds(book):
    assert exact_reml_cost_message(**book, distribution=Poisson(), link=LogLink()) is None


def test_rule_skips_gaussian_identity_and_covers_other_families():
    # constant working weights: the weighted cross-product is formed once
    assert (
        exact_reml_cost_message(**BOOK_TEN_PAIRS, distribution=Gaussian(), link=IdentityLink())
        is None
    )
    assert (
        exact_reml_cost_message(**BOOK_THREE_PAIRS, distribution=Binomial(), link=LogitLink())
        is not None
    )


def _small_book(n=2_000, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "x": rng.uniform(0.0, 1.0, n),
            "z": rng.uniform(0.0, 1.0, n),
            "g": pd.Categorical(rng.choice(list("abcd"), n)),
        }
    )
    mu = np.exp(-1.0 + np.sin(3.0 * df["x"]) + 0.3 * df["z"])
    return df, rng.poisson(mu).astype(float)


def _model(discrete):
    return SuperGLM(
        family="poisson",
        features={"x": Spline(n_knots=8), "z": Spline(n_knots=8), "g": Categorical()},
        discrete=discrete,
    )


def _slow_fit_warnings(model, df, y):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit_reml(df, y)
    return [w for w in caught if issubclass(w.category, SlowFitWarning)]


def test_ordinary_small_exact_fit_is_silent():
    df, y = _small_book()
    assert _slow_fit_warnings(_model(discrete=False), df, y) == []


@pytest.mark.parametrize("discrete", [False, True])
def test_fit_reml_warns_on_the_exact_path_only(monkeypatch, discrete):
    # Lower the threshold so a small fit counts as large; the wiring is under test.
    monkeypatch.setattr(fit_cost, "EXACT_REML_WORK_WARN", 1.0)
    df, y = _small_book()
    caught = _slow_fit_warnings(_model(discrete=discrete), df, y)
    if discrete:
        assert caught == []
    else:
        assert len(caught) == 1
        assert "2,000 rows" in str(caught[0].message)
        # attributed to the caller's fit_reml line, not to library internals
        assert caught[0].filename == __file__


def test_structured_fit_is_silent(monkeypatch):
    # The structured solver eliminates the random-effect block and never forms
    # the full cross-product the rule counts.
    monkeypatch.setattr(fit_cost, "EXACT_REML_WORK_WARN", 1.0)
    df, y = _small_book()
    df["re"] = pd.Categorical(np.arange(len(df)) % 400)
    model = SuperGLM(
        family="poisson",
        features={"x": Spline(n_knots=8), "re": RandomEffect()},
        direct_solve="structured",
    )
    assert _slow_fit_warnings(model, df, y) == []
    assert model._reml_profile["structured_chain"] == ("re",)
