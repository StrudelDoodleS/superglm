"""Small fitted SuperLSS cases for the editor tests.

Each case has 600 rows with ``age`` (uniform 18-90), ``region`` (seven levels of
unequal frequency) and ``power`` (uniform 4-12). The mean has a smooth of
``age``, region levels and a smooth of ``power``; the scale (or theta) has a
smooth of ``age`` and region levels. The cases are cached, so tests must
revise copies and never fit or mutate the shared model.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from superglm import GammaLS, LogNormalLS, NegativeBinomialLS, SuperLSS, cat, s

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


def _covariates(seed: int) -> tuple[np.random.Generator, pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    region = rng.choice(len(REGIONS), size=N_ROWS, p=_REGION_FREQUENCIES)
    frame = pd.DataFrame(
        {
            "age": rng.uniform(18.0, 90.0, N_ROWS),
            "region": np.array(REGIONS)[region],
            "power": rng.uniform(4.0, 12.0, N_ROWS),
        }
    )
    return rng, frame, region


def _log_mean(frame: pd.DataFrame, region: np.ndarray) -> np.ndarray:
    hump = np.sin(np.pi * (frame["age"].to_numpy() - 18.0) / 72.0)
    return 0.5 * hump + 0.05 * (frame["power"].to_numpy() - 8.0) + _REGION_LOCATION[region]


def _log_spread(frame: pd.DataFrame, region: np.ndarray) -> np.ndarray:
    centred = (frame["age"].to_numpy() - 54.0) / 36.0
    return 0.5 * centred**2 + _REGION_SPREAD[region]


@functools.cache
def lognormal_case() -> FitCase:
    rng, X, region = _covariates(seed=20261103)
    mean = np.exp(7.0 + _log_mean(X, region))
    sigma = np.exp(-0.7 + _log_spread(X, region))
    y = np.exp(np.log(mean) - 0.5 * sigma**2 + sigma * rng.standard_normal(N_ROWS))
    family = LogNormalLS()
    model = SuperLSS(
        family,
        family.mean(s("age", k=8), cat("region"), s("power", k=6)),
        family.scale(s("age", k=6), cat("region")),
    ).fit_reml(X, y, outer="efs+newton")
    return FitCase(model, X, y, None, None)


@functools.cache
def gamma_case() -> FitCase:
    rng, X, region = _covariates(seed=20261104)
    mean = np.exp(7.0 + _log_mean(X, region))
    cv = np.exp(-0.6 + _log_spread(X, region))
    y = rng.gamma(shape=1.0 / cv**2, scale=mean * cv**2)
    family = GammaLS()
    model = SuperLSS(
        family,
        family.mean(s("age", k=8), cat("region"), s("power", k=6)),
        family.scale(s("age", k=6), cat("region")),
    ).fit_reml(X, y, outer="efs+newton")
    return FitCase(model, X, y, None, None)


@functools.cache
def nb_case() -> FitCase:
    rng, X, region = _covariates(seed=20261105)
    X["exposure"] = rng.uniform(0.2, 1.0, N_ROWS)
    mean = X["exposure"].to_numpy() * np.exp(1.2 + _log_mean(X, region))
    theta = np.exp(1.0 - 0.5 * _log_spread(X, region))
    y = rng.negative_binomial(theta, theta / (theta + mean)).astype(np.float64)
    offsets = {"mean": np.log(X["exposure"].to_numpy())}
    family = NegativeBinomialLS()
    model = SuperLSS(
        family,
        family.mean(s("age", k=8), cat("region"), s("power", k=6)),
        family.theta(s("age", k=6), cat("region")),
    ).fit_reml(X, y, offsets=offsets, outer="efs+newton")
    return FitCase(model, X, y, None, offsets)
