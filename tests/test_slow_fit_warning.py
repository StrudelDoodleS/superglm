"""SlowFitWarning: an exact REML fit says in advance when its smoothing search will be slow."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from superglm import Categorical, SlowFitWarning, Spline, SuperGLM
from superglm.diagnostics import fit_cost
from superglm.diagnostics.fit_cost import exact_reml_cost_message
from superglm.distributions import Binomial, Gaussian, Poisson
from superglm.links import IdentityLink, LogitLink, LogLink

# freMTPL2 (542,410 training rows): mains only, and mains + one spline-by-factor pair
BOOK_MAINS = dict(n_rows=542_410, n_smoothing=3, n_coef=67)  # 7.8 s exact
BOOK_ONE_PAIR = dict(n_rows=542_410, n_smoothing=13, n_coef=197)  # 149 s exact, 8 s binned


def test_rule_fires_on_the_slow_book_fit_and_not_on_the_fast_one():
    message = exact_reml_cost_message(**BOOK_ONE_PAIR, distribution=Poisson(), link=LogLink())
    assert message is not None
    assert "542,410 rows" in message and "13 smoothing parameters" in message
    assert "discrete=True" in message
    assert exact_reml_cost_message(**BOOK_MAINS, distribution=Poisson(), link=LogLink()) is None


def test_rule_skips_families_without_per_parameter_weight_derivatives():
    # Gaussian identity: constant working weights, no derivative Grams at all
    assert (
        exact_reml_cost_message(**BOOK_ONE_PAIR, distribution=Gaussian(), link=IdentityLink())
        is None
    )
    # binomial logit with >= 4 smoothing parameters takes the one-pass leverage route
    assert (
        exact_reml_cost_message(**BOOK_ONE_PAIR, distribution=Binomial(), link=LogitLink()) is None
    )
    few = dict(n_rows=10_000_000, n_smoothing=3, n_coef=67)
    assert exact_reml_cost_message(**few, distribution=Binomial(), link=LogitLink()) is not None


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
