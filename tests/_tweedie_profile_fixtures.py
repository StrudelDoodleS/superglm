"""Tweedie frames shared by the power-search and REML-tolerance tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from superglm import Categorical, OrderedCategorical, Spline

CAT_LEVELS = {"f0": 6, "f1": 9, "f2": 4, "f3": 11, "f4": 13, "f5": 11}
OC_LEVELS = {"f6": 16, "f7": 24}


def flat_lambda_fixture(n: int = 12_000, seed: int = 4, *, informative_smooths: bool = False):
    """Tweedie frame whose saturated cr terms leave log-lambda nearly flat.

    On this realisation the exact-Newton optimizer at reml_tol=1e-6 stops five
    iterations before the tight answer, moving a published SE by ~92%. The
    same conditioning makes penalized modes uncertifiable toward p=2 at a few
    thousand rows, which the power-search tests use as a natural wall.

    ``informative_smooths=True`` adds penalty-visible curvature to the two
    ordinal smooths' level profiles (one sine cycle on f6, two on f7). The
    default realisation's ordinal signal is purely linear in the level index,
    which lies in the cr penalty's null space: nothing in the data then ties
    those smoothing parameters down, so whether their log-lambda directions
    read as "informative" was decided entirely by the criterion's error
    terms. Under the pre-0.29.0 reduced Tweedie scale profile (which charged
    this fixture's 83% zero rows a log-phi the exact saturated likelihood
    does not contain) they measured informative; under the exact criterion
    they are genuinely null — the 1-D exact-criterion profile decreases
    monotonically toward the lambda cap and the optimizer's terminal beats
    the reduced-criterion answer by 0.7 (12k) to 9.0 (400k). Tests about
    *informative* directions must therefore opt into curvature the penalty
    can see; tests about flat directions use the default.
    """
    rng = np.random.default_rng(seed)
    cols: dict[str, np.ndarray] = {}
    eta = np.full(n, -1.0)
    for name, k in CAT_LEVELS.items():
        levels = [f"{name}_{j:02d}" for j in range(k)]
        idx = rng.integers(0, k, n)
        cols[name] = np.array(levels)[idx]
        eta += rng.normal(0, 0.2, k)[idx]
    orders: dict[str, list[str]] = {}
    for name, k in OC_LEVELS.items():
        levels = [f"{name}_{j:02d}" for j in range(k)]
        idx = rng.integers(0, k, n)
        cols[name] = np.array(levels)[idx]
        eta += 0.02 * (idx - k / 2)
        if informative_smooths:
            cycles = 1.0 if name == "f6" else 2.0
            amplitude = 0.10 if name == "f6" else 0.25
            eta += amplitude * np.sin(2.0 * np.pi * cycles * idx / (k - 1))
        orders[name] = levels
    frame = pd.DataFrame(cols)
    weights = rng.uniform(1.19e-5, 1.0, n)
    offset = np.where(rng.random(n) < 0.35, 0.0, 1.0986)
    y = np.where(rng.random(n) < 0.83, 0.0, rng.gamma(1.5, np.exp(eta) * 900, n))
    features: dict = {name: Categorical() for name in CAT_LEVELS}
    for name, k in OC_LEVELS.items():
        features[name] = OrderedCategorical(order=orders[name], basis=Spline(kind="cr", k=k))
    return frame, y, weights, offset, features


def search_fixture(n: int = 1_200, seed: int = 7):
    """Small two-term Tweedie frame for power-search tests."""
    rng = np.random.default_rng(seed)
    cat_levels = [f"c{j}" for j in range(4)]
    cat = np.array(cat_levels)[rng.integers(0, 4, n)]
    oc_levels = [f"o{j:02d}" for j in range(8)]
    oc_idx = rng.integers(0, 8, n)
    eta = 0.3 * (cat == "c1") + 0.05 * (oc_idx - 4) - 0.5
    y = np.where(rng.random(n) < 0.4, 0.0, rng.gamma(1.2, np.exp(eta) * 2.0, n))
    frame = pd.DataFrame({"c": cat, "o": np.array(oc_levels)[oc_idx]})
    features = {
        "c": Categorical(),
        "o": OrderedCategorical(order=oc_levels, basis=Spline(kind="cr", k=8)),
    }
    return frame, y, features
