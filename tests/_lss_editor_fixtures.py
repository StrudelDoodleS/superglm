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
from superglm.distributional.families import (
    GaussianLS,
    GeneralizedGammaLSS,
    GeneralizedParetoLSS,
    LogLogisticLSS,
    LomaxLSS,
    TwoPieceLogNormalLSS,
    TwoPieceNormalLSS,
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


def _negative_binomial_draws(rng: np.random.Generator, natural: np.ndarray) -> np.ndarray:
    """Counts with mean ``natural[:, 0]`` and size ``natural[:, 1]``; the family has no quantile."""
    size = natural[:, 1]
    return rng.negative_binomial(size, size / (size + natural[:, 0])).astype(np.float64)


# The example table: one entry per exported scalar SuperLSS family, each the natural
# parameters, in the family's declared order, of the intercept-only law its rows are
# drawn from (the first two vary with age, and the first with region when asked). A
# family without ``quantile`` brings a sampler ``(rng, natural) -> y``.
_SCALAR_LAWS = {
    "GaussianLS": (GaussianLS, {}, (50.0, 200.0), None),
    "GeneralizedGammaLSS": (GeneralizedGammaLSS, {}, (1000.0, 0.6, 1.0), None),
    "GeneralizedParetoLSS": (GeneralizedParetoLSS, {}, (1000.0, 0.3), None),
    "LogLogisticLSS": (LogLogisticLSS, {}, (1000.0, 3.0), None),
    "LomaxLSS": (LomaxLSS, {}, (1000.0, 3.0), None),
    "TwoPieceLogNormalLSS": (TwoPieceLogNormalLSS, {}, (1000.0, 0.5, 0.2), None),
    "TwoPieceNormalLSS": (TwoPieceNormalLSS, {}, (50.0, 200.0, 0.2), None),
    "LogNormalLS-location": (LogNormalLS, {"parametrisation": "location"}, (7.0, 0.5), None),
    "LogNormalLS": (LogNormalLS, {}, (1000.0, 0.6), None),
    "GammaLS": (GammaLS, {}, (1000.0, 0.6), None),
    "NegativeBinomialLS": (NegativeBinomialLS, {}, (3.0, 2.0), _negative_binomial_draws),
    "TweedieLSS": (TweedieLSS, {}, (100.0, 2.0, 1.5), None),
}
EXAMPLE_FAMILIES = tuple(_SCALAR_LAWS)
# The browser draws these four in the distribution panel and Explain tab.
_DRAWN_IN_THE_BROWSER = frozenset({"LogNormalLS", "GammaLS", "NegativeBinomialLS", "TweedieLSS"})
SCALAR_FAMILIES = tuple(name for name in _SCALAR_LAWS if name not in _DRAWN_IN_THE_BROWSER)
EXAMPLE_FAMILY_CLASSES = frozenset(law[0].__name__ for law in _SCALAR_LAWS.values())


@functools.cache
def scalar_family_case(
    name: str, links: tuple[str | None, ...] = (), *, levels: bool = False, third: bool = False
) -> FitCase:
    """400 rows of one of ``EXAMPLE_FAMILIES``, drawn from its own ``quantile``.

    The first parameter has ``s("age", k=5)``, and ``cat("region")`` too with
    ``levels``; the second has ``s("age", k=4)``, and so does the third with
    ``third``; any other is an intercept. ``links`` overrides the parameters'
    links in order.
    """
    cls, options, natural, sampler = _SCALAR_LAWS[name]
    family = cls(**options)
    rng = np.random.default_rng(20261112)
    n = 400
    X = pd.DataFrame({"age": rng.uniform(18.0, 90.0, n)})
    wave = np.sin(np.pi * (X["age"].to_numpy() - 18.0) / 72.0)
    theta = np.tile(np.asarray(natural, dtype=np.float64), (n, 1))
    theta[:, 0] *= np.exp(0.3 * wave)
    theta[:, 1] *= np.exp(0.2 * wave)
    if levels:
        # Drawn from a stream of their own, so the rows without levels stay as they were.
        region = np.random.default_rng(20261113).choice(len(REGIONS), size=n, p=_REGION_FREQUENCIES)
        X["region"] = np.array(REGIONS)[region]
        theta[:, 0] *= np.exp(0.5 * _REGION_LOCATION[region])
    if sampler is None:
        y = np.asarray(family.quantile(rng.uniform(0.005, 0.995, n), theta), dtype=np.float64)
    else:
        y = sampler(rng, theta)
    terms = (
        (s("age", k=5), cat("region")) if levels else (s("age", k=5),),
        (s("age", k=4),),
        *(((s("age", k=4),),) if third else ()),
    )
    # A family may name its predictor helpers apart from its parameters (Tweedie's ``mu``).
    helpers = getattr(family, "_predictor_helpers", {})
    predictors = [
        getattr(family, helpers.get(spec.name, spec.name))(
            *(terms[i] if i < len(terms) else ()), link=links[i] if i < len(links) else None
        )
        for i, spec in enumerate(family.parameters)
    ]
    model = SuperLSS(family, *predictors).fit_reml(X, y, outer="efs+newton")
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
        # The chart's no-effect line follows the curve's own unit.
        assert item["display"]["kind"] == term.get("display", {"kind": "relativity"})["kind"]
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
