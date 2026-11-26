"""A severity distribution fitted as three pieces at one fixed threshold."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any, TypedDict, Unpack
from uuid import UUID, uuid4

import numpy as np
import pandas as pd
import scipy.linalg
from numpy.typing import ArrayLike, NDArray
from scipy import integrate, optimize, special

from superglm._experimental import warn_experimental
from superglm._frame import EagerFrame, FrameLike, as_eager_frame
from superglm.distributional.api import SuperLSS, _renderer
from superglm.distributional.families.gamma import GammaLS
from superglm.distributional.families.generalized_pareto import GeneralizedParetoLSS
from superglm.distributional.families.log_normal import LogNormalLS
from superglm.distributional.kernels._gamma_windows import gamma_shape_scale_window_jets
from superglm.distributional.kernels._severity_primitives import logratio
from superglm.distributional.offsets import shape_fit_offsets, sources_of, take_offset
from superglm.distributional.response import unvalidated_response, validated_response_array
from superglm.distributional.row_law import (
    LEVELS,
    DistributionCurve,
    check_curve_request,
    curve_grid,
    finite_amounts,
    refuse_missing,
    row_log_density,
)
from superglm.distributional.weights import WeightContract, _weight_values
from superglm.distributions import Binomial, resolve_distribution
from superglm.model.api import SuperGLM
from superglm.solvers.dispersion import model_weight_semantics

_VERSION = 1
_LOCATION_LOG_NORMAL = LogNormalLS(parametrisation="location", scale_floor=0)
_FREQUENCY = WeightContract("frequency")
_LOG_TINY = np.log(np.finfo(np.float64).tiny)
_LOG_HALF_SMALLEST = np.log(np.finfo(np.float64).smallest_subnormal) - np.log(2.0)
_LOG_UNIT_ROUNDOFF = -53.0 * np.log(2.0)

type _Body = GammaLS | LogNormalLS
type _Components = tuple[SuperLSS, SuperGLM, SuperLSS]


@dataclass(frozen=True)
class _Split:
    identity: UUID
    threshold: float


@dataclass(frozen=True)
class _FitEvidence:
    split: _Split
    role: str
    revision: int


class _Offsets(TypedDict, total=False):
    body_offsets: Mapping[str, NDArray]
    exceedance_offset: NDArray
    tail_offsets: Mapping[str, NDArray]


def _revision(model: SuperLSS | SuperGLM) -> int:
    if isinstance(model, SuperLSS):
        return 0 if model._model is None else model._model.fit_state.revision
    return getattr(model, "_fit_revision", 0)


def _verify_split(models: _Components) -> _Split:
    split = None
    for role, model in zip(("body", "exceedance", "tail"), models, strict=True):
        evidence = getattr(model, "_split_splice_evidence", None)
        if (
            not isinstance(evidence, _FitEvidence)
            or evidence.role != role
            or evidence.revision != _revision(model)
        ):
            raise ValueError(f"{role} must carry a current fit from SplitSplice.fit_components")
        if split is not None and split != evidence.split:
            raise ValueError("body, exceedance and tail must come from the same split")
        split = evidence.split
    assert split is not None
    return split


def _aligned(frame: EagerFrame, values: object, name: str) -> None:
    if isinstance(values, pd.Series) and isinstance(frame.native, pd.DataFrame):
        if not values.index.equals(frame.native.index):
            raise ValueError(f"{name} index must match X in row order")


@contextmanager
def _split_offsets(model: SuperLSS, rows: NDArray) -> Iterator[None]:
    """Give the component fit its row subset and keep the original declarations."""
    templates = model._predictors
    model._predictors = tuple(
        replace(predictor, offset=take_offset(sources_of(predictor), rows))
        if any(not isinstance(source, str) for source in sources_of(predictor))
        else predictor
        for predictor in templates
    )
    try:
        yield
    finally:
        model._predictors = templates


def _row_values(X: FrameLike, values: ArrayLike, name: str) -> NDArray[np.float64]:
    frame = as_eager_frame(X)
    _aligned(frame, values, name)
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0:
        return np.broadcast_to(array, (len(frame),))
    if array.shape != (len(frame),):
        raise ValueError(f"{name} must be a scalar or one value per row of X")
    return array


def _log_window(family: _Body, a: ArrayLike, b: ArrayLike, theta: NDArray) -> NDArray:
    return family.log_window_probability(np.asarray(a), np.asarray(b), theta)


def _body_quantile(family: _Body, log_p: NDArray, theta: NDArray, u: float) -> NDArray:
    if isinstance(family, LogNormalLS):
        mu, sigma = _log_normal_coordinates(family, theta)
        return np.exp(np.minimum(np.log(u), mu + sigma * special.ndtri_exp(log_p)))

    mean, cv = theta.T
    shape = 1.0 / (cv * cv)
    lower = log_p <= -np.log(2.0)
    quantile = np.empty(len(log_p))
    quantile[lower] = special.gammaincinv(shape[lower], np.exp(log_p[lower]))
    quantile[~lower] = special.gammainccinv(shape[~lower], -np.expm1(log_p[~lower]))
    with np.errstate(over="ignore", invalid="ignore"):
        quantile *= mean / shape
    # The parent probability, Gamma scale or scaled inverse can fall outside float64 range.
    use_log = (log_p < _LOG_TINY) | (quantile == 0) | ~np.isfinite(quantile)
    for i in np.flatnonzero(use_log):
        log_scale = np.log(mean[i]) - np.log(shape[i])
        log_u = np.log(u)
        log_normalizer = special.gammaln(shape[i] + 1.0)

        def residual(log_y):
            log_x = log_y - log_scale
            if log_x < _LOG_UNIT_ROUNDOFF:
                # e**(-x) <= P(shape,x)*Gamma(shape+1)/x**shape <= 1.
                return shape[i] * log_x - log_normalizer - log_p[i]
            y = u if log_y == log_u else np.exp(log_y)
            return _log_window(family, 0.0, y, theta[i : i + 1])[0] - log_p[i]

        if residual(_LOG_HALF_SMALLEST) >= 0.0:
            quantile[i] = 0.0
        elif residual(log_u) <= 0.0:
            quantile[i] = u
        else:
            quantile[i] = np.exp(optimize.brentq(residual, _LOG_HALF_SMALLEST, log_u, xtol=2e-13))
    return np.minimum(quantile, u)


def _log_normal_coordinates(family: LogNormalLS, theta: NDArray) -> tuple[NDArray, NDArray]:
    first, sigma = theta.T
    mu = np.log(first) - 0.5 * sigma**2 if family.parametrisation == "mean" else first
    return mu, sigma


def _log_body_moment(family: _Body, a: ArrayLike, b: ArrayLike, theta: NDArray) -> NDArray:
    """Log of E[Y 1(a < Y <= b)] by exponentially tilting the density."""
    if isinstance(family, GammaLS):
        mean, cv = theta.T
        v = cv * cv
        with np.errstate(over="ignore"):
            shifted = np.column_stack((mean * (1.0 + v), cv / np.sqrt(1.0 + v)))
        finite = np.isfinite(shifted[:, 0]) & (shifted[:, 1] > 0.0)
        if np.all(finite):
            return np.log(mean) + _log_window(family, a, b, shifted)
        lo, hi, _ = np.broadcast_arrays(a, b, mean)
        log_mass = np.empty(len(mean))
        if np.any(finite):
            log_mass[finite] = family.log_window_probability(
                lo[finite], hi[finite], shifted[finite]
            )
        rows = ~finite
        log_mass[rows] = gamma_shape_scale_window_jets(
            lo[rows],
            hi[rows],
            1.0 / v[rows] + 1.0,
            np.log(mean[rows]) + 2.0 * np.log(cv[rows]),
            order=0,
        )[0]
        return np.log(mean) + log_mass

    mu, sigma = _log_normal_coordinates(family, theta)
    shifted = np.column_stack((mu + sigma**2, sigma))
    log_mean = np.log(theta[:, 0]) if family.parametrisation == "mean" else mu + 0.5 * sigma**2
    return log_mean + _log_window(_LOCATION_LOG_NORMAL, a, b, shifted)


def _log_ramp_mean(family: _Body, a: float, w: float, theta: NDArray) -> float:
    """Log of E[Y-a | a < Y <= a+w] when subtracting moments loses precision."""
    # In log response the normalizing mass is proportional to a Gamma
    # density times y, or a normal density for the lognormal.
    ratio = w / a
    end = np.log1p(ratio) if np.isfinite(ratio) else np.log(w) - np.log(a)
    first, second = theta
    if isinstance(family, GammaLS):
        shape = 1.0 / (second * second)
        mode = logratio(first, a)
        peak = np.clip(mode, 0.0, end)
        offset = peak - mode
        rate = np.exp(np.log(shape) + offset)
        slope = -shape * np.expm1(offset) if abs(offset) < 0.5 else shape - rate
        scale = 1.0 / max(1.0, abs(slope), np.sqrt(rate))

        def log_density(delta):
            # exp(d)-1-d = d² * 1F1(1;3;d)/2, stable near the peak.
            if abs(delta) < 0.5:
                curvature = (rate * delta) * delta * (0.5 * special.hyp1f1(1, 3, delta))
            else:
                curvature = rate * (np.expm1(delta) - delta)
            return slope * delta - curvature
    else:
        mu, sigma = _log_normal_coordinates(family, theta[None, :])
        mode = mu[0] - np.log(a)
        peak = np.clip(mode, 0.0, end)
        z_peak = (peak - mode) / sigma[0]
        scale = min(1.0, sigma[0] / max(1.0, abs(z_peak)))

        def log_density(delta):
            shift = delta / sigma[0]
            return -shift * (z_peak + 0.5 * shift)

    # The slope and curvature at the clipped mode set the local width.
    # Map each side separately by distance = scale*t/(1-t), so even a
    # window many widths across spends quadrature nodes near the mode.
    mass = ramp = 0.0
    pivot = min(end, peak + scale)
    log_expm1_pivot = np.log(-np.expm1(-pivot))
    for sign, extent in ((-1, peak), (1, end - peak)):
        if extent <= 0.0:
            continue
        bound = extent / (scale + extent)

        def integrand(t, payout):
            delta = sign * scale * (t / (1.0 - t))
            log_weight = log_density(delta) - 2.0 * np.log1p(-t)
            if payout:
                x = min(peak + delta, end)
                if x <= 0.0:
                    return 0.0
                # Divide expm1(x) by expm1(pivot) before integrating.
                log_weight += x - pivot + np.log(-np.expm1(-x)) - log_expm1_pivot
            return np.exp(log_weight)

        mass += integrate.quad(integrand, 0.0, bound, args=(False,), epsabs=0, epsrel=1e-10)[0]
        ramp += integrate.quad(integrand, 0.0, bound, args=(True,), epsabs=0, epsrel=1e-10)[0]
    if not (0.0 < mass < np.inf and 0.0 < ramp < np.inf):
        raise ArithmeticError("body layer moment could not be resolved in float64")
    return np.log(a) + pivot + log_expm1_pivot + np.log(ramp / mass)


def _body_layer(family: _Body, a: NDArray, w: NDArray, u: float, theta: NDArray) -> NDArray:
    b = np.minimum(a + w, u)
    log_norm = _log_window(family, 0.0, u, theta)
    above = np.exp(np.log(w) + _log_window(family, b, u, theta) - log_norm)
    ramp = np.zeros(len(a))
    rows = b > a
    if np.any(rows):
        lo, hi, width, params = a[rows], b[rows], w[rows], theta[rows]
        log_mass = _log_window(family, lo, hi, params)
        log_moment = _log_body_moment(family, lo, hi, params)
        norm = log_norm[rows]
        with np.errstate(divide="ignore", invalid="ignore"):
            gap = np.log(lo) + log_mass - log_moment
            log_value = log_moment - norm + np.log(-np.expm1(np.minimum(gap, 0.0)))
        value = np.exp(log_value)
        # The ramp is at most width times its conditional probability.
        log_bound = log_mass - norm + np.log(width)
        negligible = log_bound <= _LOG_HALF_SMALLEST
        value[negligible] = 0.0
        fallback = (lo > 0) & ~negligible & ((gap > -0.01) | ~np.isfinite(value))
        for i in np.flatnonzero(fallback):
            log_ramp = _log_ramp_mean(family, float(lo[i]), float(width[i]), params[i])
            value[i] = np.exp(log_mass[i] - norm[i] + log_ramp)
        ramp[rows] = value
    return np.minimum(above + ramp, w)


def _tail_layer(family: GeneralizedParetoLSS, a: NDArray, w: NDArray, theta: NDArray, q: NDArray):
    """Integrate q*S over [a, a+w] after shifting the GPD scale to a."""
    scale, shape = theta.T
    log_width = np.log(w)
    with np.errstate(divide="ignore", over="ignore"):
        log_xi = np.log(shape)
        log_scale = np.logaddexp(np.log(scale), log_xi + np.log(a))
        log_r = log_width - log_scale
        log_z = log_xi + log_r
        # h = log1p(xi*w/scale_at_a)/xi. Its small-argument limit is w/scale_at_a.
        log_h = log_r.copy()
        ordinary = log_z >= -36.0
        log_h[ordinary] = np.log(np.logaddexp(0.0, log_z[ordinary])) - log_xi[ordinary]
        log_decay = np.log1p(-shape)
        log_k = log_h + log_decay
        log_area = np.where(
            log_k < -36.0,
            np.where(ordinary, log_scale + log_h, log_width),
            log_scale - log_decay + np.log(-np.expm1(-np.exp(log_k))),
        )
        log_survival = family.log_window_probability(a, np.inf, theta)
        return np.minimum(np.exp(np.log(q) + log_survival + log_area), w)


def _restore_splice(version: int, models: _Components) -> SplitSplice:
    if version != _VERSION:
        raise ValueError(f"unsupported SplitSplice artifact version {version}")
    return SplitSplice(*models)


def _window_probability(u: float, y: NDArray, survival: bool, law: tuple) -> NDArray:
    """``P(Y > y)`` or ``P(Y <= y)`` per row of predicted components ``law``."""
    family, tail_family, body, tail, q, valid = law
    out = np.full(len(q), np.nan)
    out[valid & (y <= 0.0)] = float(survival)
    boundary = valid & (y == u)
    out[boundary] = q[boundary] if survival else 1.0 - q[boundary]
    below = valid & (y > 0.0) & (y < u)
    if np.any(below):
        params, z, p = body[below], y[below], q[below]
        log_norm = _log_window(family, 0.0, u, params)
        lo, hi = (z, u) if survival else (0.0, z)
        log_ratio = _log_window(family, lo, hi, params) - log_norm
        conditional = np.exp(np.minimum(log_ratio, 0.0))
        out[below] = (1.0 - p) * conditional + (p if survival else 0.0)
    above = valid & (y > u) & np.isfinite(y) & (q > 0.0)
    if np.any(above):
        log_sf = np.log(q[above]) + tail_family.log_window_probability(
            y[above] - u, np.inf, tail[above]
        )
        out[above] = np.exp(log_sf) if survival else -np.expm1(log_sf)
    out[valid & (y > u) & ((q == 0.0) | (y == np.inf))] = float(not survival)
    return out


@dataclass(frozen=True, init=False)
class SplitSplice:
    """A fixed-threshold loss law with verified body, exceedance and excess fits.

    Use :meth:`fit` with the original loss rows and three configured models.
    To assemble the models later, pass the three results of :meth:`fit_components`
    to this constructor in body, exceedance, tail order. The threshold is part
    of their fit evidence. Refit the splice if a component's fit changes.

    A standard pickle or joblib round trip preserves the law and its threshold.
    Each prediction accepts the same frame for all three components. Pass
    ``body_offsets``, ``exceedance_offset`` and ``tail_offsets`` for models
    that require prediction-time offset arrays.
    """

    body: SuperLSS
    exceedance: SuperGLM
    tail: SuperLSS
    _split: _Split

    def __init__(self, body: SuperLSS, exceedance: SuperGLM, tail: SuperLSS):
        split = _verify_split((body, exceedance, tail))
        object.__setattr__(self, "body", body)
        object.__setattr__(self, "exceedance", exceedance)
        object.__setattr__(self, "tail", tail)
        object.__setattr__(self, "_split", split)

    @property
    def threshold(self) -> float:
        """Threshold separating the body from strictly larger losses."""
        return self._split.threshold

    def __reduce__(self):
        self._check_fits()
        return _restore_splice, (_VERSION, (self.body, self.exceedance, self.tail))

    @staticmethod
    def fit_components(
        X: FrameLike,
        y: ArrayLike,
        *,
        threshold: float,
        body: SuperLSS,
        exceedance: SuperGLM,
        tail: SuperLSS,
        sample_weight: ArrayLike | None = None,
        censored: ArrayLike = False,
        reml: bool = False,
        body_fit_options: Mapping[str, Any] | None = None,
        exceedance_fit_options: Mapping[str, Any] | None = None,
        tail_fit_options: Mapping[str, Any] | None = None,
    ) -> _Components:
        """Fit and return the three supplied models on one verified split.

        ``body`` must be a :class:`SuperLSS` with ``GammaLS`` or
        ``LogNormalLS``; ``exceedance`` must be a binomial :class:`SuperGLM`;
        ``tail`` must be a :class:`SuperLSS` with ``GeneralizedParetoLSS``.

        ``y`` contains positive loss amounts, or the limit for a right-censored
        row. ``censored`` is a boolean flag per row (or one flag for all rows).
        A right-censored limit must be at or above the threshold, so the loss
        is known to exceed it. A complete loss equal to the threshold goes
        in the body. Arrays are positional. Pandas Series must match X's index
        in order. Nonunit weights require frequency semantics on all three
        models and give exact nonnegative integer replication counts. Rows
        with zero counts are dropped before validating response values.

        ``reml=True`` calls each model's ``fit_reml``. Each options mapping is
        forwarded to the corresponding fit, with response bounds and weights
        supplied by this split.
        """
        u = float(threshold)
        if not np.isfinite(u) or u <= 0.0:
            raise ValueError("threshold must be finite and positive")
        if not isinstance(body, SuperLSS) or not isinstance(body.family, GammaLS | LogNormalLS):
            raise TypeError("body must be a SuperLSS with GammaLS or LogNormalLS")
        if not isinstance(exceedance, SuperGLM) or not isinstance(
            resolve_distribution(exceedance.family), Binomial
        ):
            raise TypeError("exceedance must be a binomial SuperGLM")
        if not isinstance(tail, SuperLSS) or not isinstance(tail.family, GeneralizedParetoLSS):
            raise TypeError("tail must be a SuperLSS with GeneralizedParetoLSS")

        frame = as_eager_frame(X)
        for name, values in (("y", y), ("censored", censored), ("sample_weight", sample_weight)):
            _aligned(frame, values, name)
        right = np.asarray(censored, dtype=object if isinstance(censored, list | tuple) else None)
        if right.ndim != 0 and right.shape != (len(frame),):
            raise ValueError("censored must be a boolean flag per row or one boolean scalar")
        right = np.broadcast_to(right, (len(frame),))
        weights = None
        active = None
        if sample_weight is not None:
            weights = _weight_values(np.asarray(sample_weight), len(frame), _FREQUENCY)
            if (
                np.any(weights != 1)
                and (
                    body.weight_semantics,
                    model_weight_semantics(exceedance),
                    tail.weight_semantics,
                )
                != ("frequency",) * 3
            ):
                raise ValueError(
                    "nonunit replication weights require frequency semantics on all fits"
                )
            if np.any(weights == 0):
                active = np.flatnonzero(weights > 0)
                y = unvalidated_response(y, family=body.family, n_rows=len(frame))[active]
                right = right[active]
                weights = weights[active]

        split_error = "the split must have positive-weight rows on both sides of the threshold"
        if len(right) == 0:
            raise ValueError(split_error)
        loss = validated_response_array(y, family=body.family, n_rows=len(right))
        if np.any(loss <= 0.0):
            raise ValueError("loss amounts must be positive")
        if right.dtype.kind != "b":
            if right.dtype.kind != "O" or any(not isinstance(v, bool | np.bool_) for v in right):
                raise ValueError("censored must be a boolean flag per row or one boolean scalar")
            right = right.astype(bool)
        if np.any(right & (loss < u)):
            raise ValueError(
                "a censored limit below the threshold leaves exceedance membership unknown"
            )
        is_tail = (loss > u) | right
        if not (np.any(is_tail) and np.any(~is_tail)):
            raise ValueError(split_error)

        options = [
            dict(opts or {})
            for opts in (body_fit_options, exceedance_fit_options, tail_fit_options)
        ]
        for opts in options:
            for key in ("truncation", "censored", "sample_weight"):
                if key in opts:
                    raise ValueError(f"SplitSplice supplies {key} to the component fits")
        offset = options[1].get("offset")
        _aligned(frame, offset, "exceedance offset")
        if active is not None and offset is not None:
            offset = np.asarray(offset, dtype=object if isinstance(offset, list | tuple) else None)
            if offset.shape != (len(frame),):
                raise ValueError("exceedance offset must have one value per row of X")
            options[1]["offset"] = offset[active]
        for component in (body, tail):
            shape_fit_offsets(component._predictors, frame)

        rows_body, rows_tail = np.flatnonzero(~is_tail), np.flatnonzero(is_tail)
        input_body = rows_body if active is None else active[rows_body]
        input_tail = rows_tail if active is None else active[rows_tail]
        fit_body, fit_binary, fit_tail = (
            (body.fit_reml, exceedance.fit_reml, tail.fit_reml)
            if reml
            else (body.fit, exceedance.fit, tail.fit)
        )
        with _split_offsets(body, input_body):
            fit_body(
                frame.take_rows(input_body),
                loss[rows_body],
                None if weights is None else weights[rows_body],
                truncation=(None, u),
                **options[0],
            )
        fit_binary(
            X if active is None else frame.take_rows(active),
            is_tail.astype(np.float64),
            weights,
            **options[1],
        )
        with _split_offsets(tail, input_tail):
            fit_tail(
                frame.take_rows(input_tail),
                loss[rows_tail] - u,
                None if weights is None else weights[rows_tail],
                censored=right[rows_tail],
                **options[2],
            )
        split = _Split(uuid4(), u)
        for role, model in (("body", body), ("exceedance", exceedance), ("tail", tail)):
            setattr(model, "_split_splice_evidence", _FitEvidence(split, role, _revision(model)))
        return body, exceedance, tail

    @classmethod
    def fit(cls, X: FrameLike, y: ArrayLike, **kwargs: Any) -> SplitSplice:
        """Fit a split splice. See :meth:`fit_components` for the fit arguments."""
        return cls(*cls.fit_components(X, y, **kwargs))

    def _check_fits(self) -> None:
        if _verify_split((self.body, self.exceedance, self.tail)) != self._split:
            raise ValueError("the components have been refitted on a different split")

    def _predict_components(self, X: FrameLike, offsets: _Offsets):
        accepted = _Offsets.__optional_keys__
        unknown = offsets.keys() - accepted
        if unknown:
            raise TypeError(
                f"Unknown prediction offset keywords {', '.join(sorted(unknown))}. "
                f"Accepted keywords are {', '.join(sorted(accepted))}."
            )
        self._check_fits()
        body = self.body.predict_parameters(X, offsets=offsets.get("body_offsets")).to_numpy()
        tail = self.tail.predict_parameters(X, offsets=offsets.get("tail_offsets")).to_numpy()
        offset = offsets.get("exceedance_offset")
        _aligned(as_eager_frame(X), offset, "exceedance_offset")
        q = self.exceedance.predict(X, offset=offset)
        family = self.body._require_fitted().family
        tail_family = self.tail._require_fitted().family
        assert isinstance(family, GammaLS | LogNormalLS) and isinstance(
            tail_family, GeneralizedParetoLSS
        )
        valid = (q >= 0.0) & (q <= 1.0) & np.all(np.isfinite(body) & np.isfinite(tail), axis=1)
        return family, tail_family, body, tail, q, valid

    def _probability(
        self, X: FrameLike, y: ArrayLike, survival: bool, offsets: _Offsets
    ) -> NDArray:
        y = _row_values(X, y, "y")
        return _window_probability(
            self.threshold, y, survival, self._predict_components(X, offsets)
        )

    def predict_cdf(self, X: FrameLike, y: ArrayLike, **offsets: Unpack[_Offsets]) -> NDArray:
        """Return P(Y <= y) per row for a scalar or row-aligned y."""
        return self._probability(X, y, False, offsets)

    def predict_survival(self, X: FrameLike, y: ArrayLike, **offsets: Unpack[_Offsets]) -> NDArray:
        """Return P(Y > y) per row, including probabilities below CDF precision."""
        return self._probability(X, y, True, offsets)

    def predict_quantile(self, X: FrameLike, p: ArrayLike, **offsets: Unpack[_Offsets]) -> NDArray:
        """Return each row's p-quantile for probabilities strictly inside (0, 1)."""
        p = _row_values(X, p, "p")
        if np.any(~((p > 0.0) & (p < 1.0))):
            raise ValueError("quantile probabilities must lie strictly inside (0, 1)")
        family, _, body, tail, q, valid = self._predict_components(X, offsets)
        u = self.threshold
        cut = 1.0 - q
        out = np.full(len(q), np.nan)
        out[valid & (p == cut)] = u
        below = valid & (p < cut)
        if np.any(below):
            probability, mass = p[below], cut[below]
            relative = probability / mass
            log_relative = np.log(relative)
            near = relative > 0.5
            log_relative[near] = np.log1p((probability[near] - mass[near]) / mass[near])
            params = body[below]
            log_p = log_relative + _log_window(family, 0.0, u, params)
            out[below] = _body_quantile(family, log_p, params, u)
        above = valid & (p > cut)
        scale, shape = tail[above].T
        remaining = 1.0 - p[above]
        q_above = q[above]
        distance = -np.log(remaining / q_above)
        near = remaining > 0.5 * q_above
        # For q >= 1/2 the cutoff is exact, so subtract from p before it
        # disappears in 1-p. For smaller q, use the exact complement of p.
        delta = np.where(q_above >= 0.5, p[above] - cut[above], q_above - remaining)
        distance[near] = -np.log1p(-delta[near] / q_above[near])
        with np.errstate(over="ignore"):
            out[above] = u + scale * (distance * special.exprel(shape * distance))
        return out

    def predict(self, X: FrameLike, **offsets: Unpack[_Offsets]) -> NDArray:
        """Return each row's mean loss on the original scale."""
        family, _, body, tail, q, valid = self._predict_components(X, offsets)
        u = self.threshold
        out = np.full(len(q), np.nan)
        out[valid] = q[valid] * u
        in_body = valid & (q < 1.0)
        if np.any(in_body):
            params = body[in_body]
            log_mean = _log_body_moment(family, 0.0, u, params) - _log_window(
                family, 0.0, u, params
            )
            out[in_body] += (1.0 - q[in_body]) * np.exp(np.minimum(log_mean, np.log(u)))
        in_tail = valid & (q > 0.0)
        scale, shape = tail[in_tail].T
        with np.errstate(over="ignore"):
            out[in_tail] += (q[in_tail] * scale) / (1.0 - shape)
        return out

    def predict_layer(
        self, X: FrameLike, attachment: ArrayLike, limit: ArrayLike, **offsets: Unpack[_Offsets]
    ) -> NDArray:
        """Return E[min(max(Y - attachment, 0), limit)] for each row."""
        a = _row_values(X, attachment, "attachment")
        w = _row_values(X, limit, "limit")
        if np.any(~np.isfinite(a) | ~np.isfinite(w) | (w < 0.0)):
            raise ValueError("attachment and limit must be finite and limit nonnegative")
        family, tail_family, body, tail, q, valid = self._predict_components(X, offsets)
        u = self.threshold
        # Keep widths explicitly: a small layer can have a+limit rounded to a.
        certain = np.minimum(np.maximum(-a, 0.0), w)
        a = np.maximum(a, 0.0)
        remaining = w - certain
        below_u = np.minimum(np.maximum(u - a, 0.0), remaining)
        tail_width = remaining - below_u
        out = np.where(valid, certain + q * below_u, np.nan)
        below = valid & (below_u > 0.0) & (q < 1.0)
        if np.any(below):
            out[below] += (1.0 - q[below]) * _body_layer(
                family, a[below], below_u[below], u, body[below]
            )
        above = valid & (tail_width > 0.0) & (q > 0.0)
        if np.any(above):
            out[above] += _tail_layer(
                tail_family, np.maximum(a[above] - u, 0.0), tail_width[above], tail[above], q[above]
            )
        return np.minimum(out, w)

    def predict_pdf(self, X: FrameLike, y: ArrayLike, **offsets: Unpack[_Offsets]) -> NDArray:
        """Return each row's spliced probability density at ``y``.

        Evaluates ``(1 - q) f_body(y) / F_body(u)`` for ``0 < y <= u`` and
        ``q g(y - u)`` for ``y > u``, where ``u`` is ``threshold`` and ``q`` is
        the row's exceedance probability. A loss equal to ``u`` belongs to the
        body, so the density at ``u`` is the body's left limit and the spliced
        density is left-continuous at the step. Amounts at or below 0 or
        infinite return ``0.0``; rows with an invalid component prediction
        return ``NaN``. Experimental: its name, arguments and return layout may
        change in a minor release without a deprecation period; its numbers are
        tested like the rest of superglm.

        Parameters
        ----------
        X : DataFrame
            Rows to evaluate.
        y : float or array-like of shape (n_rows,)
            Unit-free loss amount(s).
        **offsets
            Optional prediction offsets: ``body_offsets``,
            ``exceedance_offset``, ``tail_offsets``.

        Returns
        -------
        ndarray of shape (n_rows,)
            Spliced density values.

        Raises
        ------
        ValueError
            If ``y`` contains missing (``NaN``) values, has the wrong length,
            cannot be evaluated in float64 at an in-support amount (naming the
            rows), or if a component was refitted on a different split.
        TypeError
            If an unknown offset keyword is supplied.
        """
        warn_experimental("predict_pdf")
        y = _row_values(X, y, "y")
        return self._density(y, y > self.threshold, self._predict_components(X, offsets))

    def predict_mean_excess(
        self, X: FrameLike, v: ArrayLike, **offsets: Unpack[_Offsets]
    ) -> NDArray:
        """Return each row's expected excess ``E[Y - v | Y > v]`` at ``v``.

        At or above ``threshold`` ``u``, evaluates the generalized Pareto tail's
        exact linear form ``(scale + shape * (v - u)) / (1 - shape)``. Below
        ``u``, combines the truncated body layer and the tail contribution over
        the spliced survival ``P(Y > v)``; for ``v < 0``, returns
        ``predict(X) - v``. Rows with an invalid component prediction return
        ``NaN``. Experimental: its name, arguments and return layout may change
        in a minor release without a deprecation period; its numbers are tested
        like the rest of superglm.

        Parameters
        ----------
        X : DataFrame
            Rows to evaluate.
        v : float or array-like of shape (n_rows,)
            Finite unit-free threshold amount(s).
        **offsets
            Optional prediction offsets: ``body_offsets``,
            ``exceedance_offset``, ``tail_offsets``.

        Returns
        -------
        ndarray of shape (n_rows,)
            Mean excess values.

        Raises
        ------
        ValueError
            If ``v`` contains missing (``NaN``) or infinite values, has the
            wrong length, or if a component was refitted on a different split.
        TypeError
            If an unknown offset keyword is supplied.
        """
        warn_experimental("predict_mean_excess")
        v = finite_amounts(_row_values(X, v, "v"), "predict_mean_excess")
        return self._mean_excess(v, self._predict_components(X, offsets))

    def distribution_curve(
        self,
        row: FrameLike,
        *,
        kind: str = "density",
        y: ArrayLike | None = None,
        n_points: int = 200,
        **offsets: Unpack[_Offsets],
    ) -> DistributionCurve:
        """Return one row's spliced law on a grid split at the threshold.

        Builds the ``DistributionCurve`` payload behind :meth:`plot_distribution`
        for a single row: its ``"density"``, ``"survival"``, or
        ``"mean_excess"``. Without ``y``, the body receives
        ``max(n_points // 2, 2)`` points from its conditional 0.001 quantile
        (0.01 for survival) up to ``threshold``, and the tail receives
        ``max(n_points - n_body, 2)`` points from ``threshold`` to
        ``threshold`` plus its 0.999 quantile (``1 - 1e-6`` for survival), so
        even a rare tail gets its own stretch of the grid. Whenever the grid
        spans ``threshold``, ``threshold`` appears twice: as the last
        ``"body"`` point (left limit) and as the first ``"tail"`` point (right
        limit, ``q * g(0)`` for density). Experimental: its name, arguments and
        return layout may change in a minor release without a deprecation
        period; its numbers are tested like the rest of superglm.

        Parameters
        ----------
        row : DataFrame
            A single row to evaluate (for example ``X.iloc[[0]]``).
        kind : {"density", "survival", "mean_excess"}, default="density"
            Which curve to evaluate.
        y : array-like of shape (n_points,), optional
            Explicit grid of at least two finite, strictly increasing amounts.
        n_points : int, default=200
            Target grid size when ``y`` is ``None`` (at least 2; each piece
            receives at least 2 points, so ``n_points`` of 2 or 3 returns 4
            points).
        **offsets
            Optional prediction offsets: ``body_offsets``,
            ``exceedance_offset``, ``tail_offsets``.

        Returns
        -------
        DistributionCurve
            Frozen curve payload with ``parameters`` keyed ``body_<name>``,
            ``"exceedance"``, and ``tail_<name>``.

        Raises
        ------
        ValueError
            If ``row`` does not have length 1, ``kind`` is unknown,
            ``n_points < 2`` (when ``y`` is ``None``), ``y`` is not a 1-D
            strictly increasing finite array of at least two amounts, or a
            component was refitted on a different split.
        TypeError
            If an unknown offset keyword is supplied.
        """
        warn_experimental("distribution_curve")
        return self._curve(row, kind, y, n_points, offsets, "distribution_curve")

    def plot_distribution(
        self,
        row: FrameLike,
        *,
        kind: str = "density",
        engine: str = "matplotlib",
        y: ArrayLike | None = None,
        n_points: int = 200,
        **offsets: Unpack[_Offsets],
    ) -> Any:
        """Draw one row's spliced density, survival or mean excess curve.

        Calls :meth:`distribution_curve` and draws the body and tail in two
        colours, marks the threshold, and connects the two one-sided values at
        the threshold with a vertical step (visible on density plots; survival
        and mean excess are continuous there). Survival is drawn on log-log
        axes. Experimental: its name, arguments and return layout may change in
        a minor release without a deprecation period; its numbers are tested
        like the rest of superglm.

        Parameters
        ----------
        row : DataFrame
            A single row to plot (for example ``X.iloc[[0]]``).
        kind : {"density", "survival", "mean_excess"}, default="density"
            Which curve to draw.
        engine : {"matplotlib", "plotly"}, default="matplotlib"
            Plotting backend.
        y : array-like of shape (n_points,), optional
            Explicit grid of at least two finite, strictly increasing amounts.
        n_points : int, default=200
            Target grid size when ``y`` is ``None`` (at least 2).
        **offsets
            Optional prediction offsets: ``body_offsets``,
            ``exceedance_offset``, ``tail_offsets``.

        Returns
        -------
        Figure
            Matplotlib or Plotly figure for the row's spliced law.

        Raises
        ------
        ValueError
            If ``row`` does not have length 1, ``kind`` or ``engine`` is
            unknown, ``y`` / ``n_points`` is invalid, or a component was
            refitted on a different split.
        TypeError
            If an unknown offset keyword is supplied.
        """
        warn_experimental("plot_distribution")
        curve = self._curve(row, kind, y, n_points, offsets, "plot_distribution")
        return _renderer(engine, "distribution")(curve)

    def _density(self, y: NDArray, in_tail: NDArray, law: tuple) -> NDArray:
        """The density at ``y``, read from the tail where ``in_tail`` and the body elsewhere."""
        family, tail_family, body, tail, q, valid = law
        u = self.threshold
        refuse_missing(y, "predict_pdf")
        out = np.where(valid, 0.0, np.nan)
        below = valid & ~in_tail & (y > 0.0)
        if np.any(below):
            rows = np.flatnonzero(below)
            log_f = row_log_density(family, y[rows], body[rows], positions=rows)
            log_body = log_f - _log_window(family, 0.0, u, body[rows])
            with np.errstate(over="ignore"):
                out[rows] = (1.0 - q[rows]) * np.exp(log_body)
        above = valid & in_tail & np.isfinite(y)
        if np.any(above):
            rows = np.flatnonzero(above)
            log_g = row_log_density(tail_family, y[rows] - u, tail[rows], positions=rows)
            with np.errstate(over="ignore"):
                out[rows] = q[rows] * np.exp(log_g)
        return out

    def _mean_excess(self, v: NDArray, law: tuple) -> NDArray:
        """``E[Y - v | Y > v]``: the tail's line from the threshold, a ratio of positive terms below."""
        family, _, body, tail, q, valid = law
        u = self.threshold
        scale, shape = tail.T
        out = np.full(len(q), np.nan)
        above = valid & (v >= u)
        with np.errstate(over="ignore"):
            out[above] = (scale[above] + shape[above] * (v[above] - u)) / (1.0 - shape[above])
        below = valid & (v < u)
        if np.any(below):
            # S(v) = 1 for v below 0, so e(v) = e(0) - v there.
            a = np.maximum(v[below], 0.0)
            params, p = body[below], q[below]
            log_ratio = _log_window(family, a, u, params) - _log_window(family, 0.0, u, params)
            survival = (1.0 - p) * np.exp(np.minimum(log_ratio, 0.0)) + p
            layer = (
                (1.0 - p) * _body_layer(family, a, u - a, u, params)
                + p * (u - a)
                + p * scale[below] / (1.0 - shape[below])
            )
            out[below] = layer / survival + np.maximum(-v[below], 0.0)
        return out

    def _curve(
        self,
        row: FrameLike,
        kind: str,
        y: ArrayLike | None,
        n_points: int,
        offsets: _Offsets,
        caller: str,
    ) -> DistributionCurve:
        grid = check_curve_request(len(as_eager_frame(row)), kind, y, n_points, caller)
        law = self._predict_components(row, offsets)
        family, tail_family, body, tail, q, valid = law
        u = self.threshold
        empty = np.empty(0)
        if grid is None:
            p_low, p_high = LEVELS[kind]
            low = _body_quantile(
                family, np.log([p_low]) + _log_window(family, 0.0, u, body), body, u
            )
            high = u + tail_family.quantile(np.array([p_high]), tail)
            n_body = max(n_points // 2, 2)
            body_grid = curve_grid(kind, n_body, low[0], u, positive=True)
            tail_grid = curve_grid(kind, max(n_points - n_body, 2), u, high[0], positive=True)
        elif grid[0] <= u < grid[-1]:
            body_grid = np.append(grid[grid < u], u)
            tail_grid = np.insert(grid[grid > u], 0, u)
        else:
            body_grid, tail_grid = (grid, empty) if grid[-1] <= u else (empty, grid)
        points = np.concatenate([body_grid, tail_grid])
        in_tail = np.arange(len(points)) >= len(body_grid)
        take = np.zeros(len(points), dtype=np.intp)
        rows = (family, tail_family, body[take], tail[take], q[take], valid[take])
        if kind == "density":
            values = self._density(points, in_tail, rows)
        elif kind == "survival":
            values = _window_probability(u, points, True, rows)
        else:
            values = self._mean_excess(points, rows)
        parameters = {
            **{
                f"body_{name}": float(value)
                for name, value in zip(self.body.parameter_names_, body[0])
            },
            "exceedance": float(q[0]),
            **{
                f"tail_{name}": float(value)
                for name, value in zip(self.tail.parameter_names_, tail[0])
            },
        }
        return DistributionCurve(
            kind=kind,
            y=points,
            values=values,
            piece=("body",) * len(body_grid) + ("tail",) * len(tail_grid),
            threshold=u,
            mean=float(self.predict(row, **offsets)[0]),
            family=f"SplitSplice({type(family).__name__}, {type(tail_family).__name__})",
            parameters=parameters,
            amount_scale="log" if kind == "survival" else "linear",
        )
