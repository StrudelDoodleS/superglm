"""Small fitted SuperLSS cases for the editor tests.

Each case has 600 rows with ``age`` (uniform 18-90), ``region`` (seven levels of
unequal frequency) and ``power`` (uniform 4-12). The mean has a smooth of
``age``, region levels and a smooth of ``power``; the scale (or theta, or the
Tweedie dispersion) has a smooth of ``age`` and region levels. The Tweedie
case has a constant power and over half of its responses are zero. The cases are cached, so tests must
revise copies and never fit or mutate the shared model.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from superglm import (
    Constraint,
    GammaLS,
    LogNormalLS,
    NegativeBinomialLS,
    SuperLSS,
    TweedieLSS,
    cat,
    s,
)
from superglm.editor import EditorSession, edit
from superglm.editor.evaluation import EvaluationDataset

EPS = np.finfo(np.float64).eps

N_ROWS = 600
REGIONS = tuple(f"R{index}" for index in range(7))
_REGION_FREQUENCIES = np.array([0.34, 0.22, 0.15, 0.11, 0.09, 0.05, 0.04])
_REGION_LOCATION = np.array([0.0, 0.25, -0.2, 0.4, -0.35, 0.15, 0.55])
_REGION_SPREAD = np.array([0.0, 0.1, -0.1, 0.2, 0.05, -0.15, 0.25])


@dataclass(frozen=True)
class FitCase:
    """A fitted model with the data it was fitted on."""

    model: SuperLSS
    X: pd.DataFrame
    y: np.ndarray
    sample_weight: np.ndarray | None
    offsets: dict[str, np.ndarray] | None


def _covariates(seed: int, n: int = N_ROWS) -> tuple[np.random.Generator, pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    region = rng.choice(len(REGIONS), size=n, p=_REGION_FREQUENCIES)
    frame = pd.DataFrame(
        {
            "age": rng.uniform(18.0, 90.0, n),
            "region": np.array(REGIONS)[region],
            "power": rng.uniform(4.0, 12.0, n),
        }
    )
    return rng, frame, region


def _log_mean(frame: pd.DataFrame, region: np.ndarray) -> np.ndarray:
    hump = np.sin(np.pi * (frame["age"].to_numpy() - 18.0) / 72.0)
    return 0.5 * hump + 0.05 * (frame["power"].to_numpy() - 8.0) + _REGION_LOCATION[region]


def _log_spread(frame: pd.DataFrame, region: np.ndarray) -> np.ndarray:
    centred = (frame["age"].to_numpy() - 54.0) / 36.0
    return 0.5 * centred**2 + _REGION_SPREAD[region]


_SEEDS = {
    LogNormalLS: 20261103,
    GammaLS: 20261104,
    NegativeBinomialLS: 20261105,
    TweedieLSS: 20261110,
}


def _lognormal_rows(seed: int, n: int):
    rng, X, region = _covariates(seed, n)
    mean = np.exp(7.0 + _log_mean(X, region))
    sigma = np.exp(-0.7 + _log_spread(X, region))
    y = np.exp(np.log(mean) - 0.5 * sigma**2 + sigma * rng.standard_normal(n))
    return X, y, None


@functools.cache
def lognormal_case() -> FitCase:
    X, y, _ = _lognormal_rows(_SEEDS[LogNormalLS], N_ROWS)
    family = LogNormalLS()
    model = SuperLSS(
        family,
        family.mean(s("age", k=8), cat("region"), s("power", k=6)),
        family.scale(s("age", k=6), cat("region")),
    ).fit_reml(X, y, outer="efs+newton")
    return FitCase(model, X, y, None, None)


def _gamma_rows(seed: int, n: int):
    rng, X, region = _covariates(seed, n)
    mean = np.exp(7.0 + _log_mean(X, region))
    cv = np.exp(-0.6 + _log_spread(X, region))
    return X, rng.gamma(shape=1.0 / cv**2, scale=mean * cv**2), None


@functools.cache
def gamma_case() -> FitCase:
    X, y, _ = _gamma_rows(_SEEDS[GammaLS], N_ROWS)
    family = GammaLS()
    model = SuperLSS(
        family,
        family.mean(s("age", k=8), cat("region"), s("power", k=6)),
        family.scale(s("age", k=6), cat("region")),
    ).fit_reml(X, y, outer="efs+newton")
    return FitCase(model, X, y, None, None)


@functools.cache
def ruled_case() -> FitCase:
    """The gamma case with fit-time shape rules on a term of each parameter.

    The mean rises in ``power`` and the CV is convex in ``age``, as the rows are
    drawn, so the rules bind only where the data wanders; the CV's region
    relativities are capped to 0.8-1.25.
    """
    X, y, _ = _gamma_rows(_SEEDS[GammaLS], N_ROWS)
    family = GammaLS()
    model = SuperLSS(
        family,
        family.mean(
            s("age", k=8),
            cat("region"),
            s("power", kind="cr", k=6, constraint=Constraint.fit.increasing),
        ),
        family.scale(
            s("age", kind="cr", k=6, constraint=Constraint.fit.convex),
            cat("region", constraint=Constraint.fit.cap(0.8, 1.25)),
        ),
    ).fit_reml(X, y, outer="efs+newton")
    return FitCase(model, X, y, None, None)


def _nb_rows(seed: int, n: int):
    rng, X, region = _covariates(seed, n)
    X["exposure"] = rng.uniform(0.2, 1.0, n)
    mean = X["exposure"].to_numpy() * np.exp(1.2 + _log_mean(X, region))
    theta = np.exp(1.0 - 0.5 * _log_spread(X, region))
    y = rng.negative_binomial(theta, theta / (theta + mean)).astype(np.float64)
    return X, y, {"mean": np.log(X["exposure"].to_numpy())}


@functools.cache
def nb_case() -> FitCase:
    X, y, offsets = _nb_rows(_SEEDS[NegativeBinomialLS], N_ROWS)
    family = NegativeBinomialLS()
    model = SuperLSS(
        family,
        family.mean(s("age", k=8), cat("region"), s("power", k=6), offset=offsets["mean"]),
        family.theta(s("age", k=6), cat("region")),
    ).fit_reml(X, y, outer="efs+newton")
    return FitCase(model, X, y, None, offsets)


def _tweedie_rows(seed: int, n: int):
    """Compound Poisson-gamma responses at power 1.5 (Jorgensen 1987), zero when no jump occurs."""
    rng, X, region = _covariates(seed, n)
    power = 1.5
    mean = np.exp(2.5 + _log_mean(X, region))
    dispersion = np.exp(2.5 + _log_spread(X, region))
    rate = mean ** (2.0 - power) / (dispersion * (2.0 - power))
    shape = (2.0 - power) / (power - 1.0)
    scale = dispersion * (power - 1.0) * mean ** (power - 1.0)
    jumps = rng.poisson(rate)
    y = np.where(jumps > 0, rng.gamma(shape * np.maximum(jumps, 1), scale), 0.0)
    return X, y, None


@functools.cache
def tweedie_case() -> FitCase:
    X, y, _ = _tweedie_rows(_SEEDS[TweedieLSS], N_ROWS)
    family = TweedieLSS()
    model = SuperLSS(
        family,
        family.mu(s("age", k=8), cat("region"), s("power", k=6)),
        family.phi(s("age", k=6), cat("region")),
        family.p(),
    ).fit_reml(X, y, outer="efs+newton")
    return FitCase(model, X, y, None, None)


_ROW_DRAWS = {
    LogNormalLS: _lognormal_rows,
    GammaLS: _gamma_rows,
    NegativeBinomialLS: _nb_rows,
    TweedieLSS: _tweedie_rows,
}


def holdout_rows(case: FitCase, n: int) -> EvaluationDataset:
    """``n`` fresh rows from the case's generator, with weights uniform on 0.5-2."""
    family = type(case.model.family)
    seed = _SEEDS[family] + 1000
    X, y, offset = _ROW_DRAWS[family](seed, n)
    weights = np.random.default_rng(seed + 1).uniform(0.5, 2.0, n)
    return EvaluationDataset("holdout", "Holdout", X, y, weights, offset, "supplied")


def open_case(case: FitCase, **kw) -> EditorSession:
    """An editor session over ``case``, opened with the rows it was fitted on."""
    return edit(case.model, train_data=(case.X, case.y, case.sample_weight, case.offsets), **kw)


def assert_cv_tab_draws_the_editor_curves(session, items) -> None:
    """Each Cross-validation entry's in-force curve is the Editor tab's, value for value.

    ``fit`` is the Editor's ``original_y`` and ``edited`` its ``y`` (absent while
    the term is unedited); every fold's curve sits at the in-force curve's
    exposure-weighted mean log, so folds compare on shape at the level drawn.
    """
    from superglm.editor.payloads import session_payload

    payload = session_payload(session)
    assert items
    for item in items:
        term = payload[item["name"]]
        if item["levels"] is None:
            index = np.arange(len(item["x"]))
            assert item["x"] == pytest.approx(term["x"])
        else:
            position = {str(level): i for i, level in enumerate(term["levels"])}
            index = np.array([position[str(level)] for level in item["levels"]])
        fit, edited = np.asarray(term["original_y"])[index], np.asarray(term["y"])[index]
        np.testing.assert_allclose(item["fit"], fit, rtol=1e-12, atol=0.0)
        if item["edited"] is None:
            np.testing.assert_allclose(edited, fit, rtol=1e-12, atol=0.0)
        else:
            assert not np.allclose(edited, fit, rtol=0.0, atol=1e-14)
            np.testing.assert_allclose(item["edited"], edited, rtol=1e-12, atol=0.0)
        assert item["value_label"] == term["y_label"]
        weights = np.asarray(item["weights"], dtype=np.float64)
        anchor = np.log(edited if item["held"] else fit)
        for fold in item["folds"]:
            values = np.log(np.asarray(fold["values"], dtype=np.float64))
            points = ~np.isnan(values)
            np.testing.assert_allclose(
                np.average(values[points], weights=weights[points]),
                np.average(anchor[points], weights=weights[points]),
                rtol=0.0,
                atol=1e-9,
            )
