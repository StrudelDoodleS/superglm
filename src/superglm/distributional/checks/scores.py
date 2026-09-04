"""Proper scores for a fitted distributional (location-scale-shape) model.

Two scores carry the checking story.  The **log score** ``-log f(y | theta_hat)``
is the local proper score the fit already optimises, read back per row.  The
**continuous ranked probability score** is the global one: it compares the whole
predictive distribution against the realised value, is reported in the units of
the response, and is the score of Gneiting and Raftery (2007), *Journal of the
American Statistical Association* 102(477), 359-378.

Two routes compute the CRPS and the tests hold them against each other.  The
closed forms are the catalogue of Jordan, Krueger and Lerch (2019), *Journal of
Statistical Software* 90(12) -- Gaussian from Gneiting and Raftery (2007), gamma
from Scheuerer and Moeller (2015), *Annals of Applied Statistics* 9(3),
1328-1349, log-normal from Baran and Lerch (2015), *Quarterly Journal of the
Royal Meteorological Society* 141(691), 2289-2299.  The general route is the
quantile-score integral of Laio and Tamea (2007), *Hydrology and Earth System
Sciences* 11(4), 1267-1277,

    CRPS = 2 * integral_0^1 (1{y < Q(p)} - p) (Q(p) - y) dp,

which asks a family for nothing but its quantile function.  Restricting the
integrand to ``Q(p) > t`` gives the threshold-weighted CRPS of Gneiting and
Ranjan (2011), *Journal of Business and Economic Statistics* 29(3), 411-422, with
the indicator weight -- the tail-story score, chosen before looking at the data.

The closed forms live here rather than on the families: a family owns its
likelihood and its distribution functions, and a score catalogue keyed by family
name keeps a scoring convention out of the fitting boundary.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, NamedTuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import special

from superglm._frame import EagerFrame, FrameLike, as_eager_frame
from superglm.distributional.family import (
    COMPLETE_OBSERVATION,
    DistributionFunctionFamily,
    PriorWeightedDistributionFunctionFamily,
)
from superglm.distributional.kernels.log_normal import location_of_mean

# These helpers preserve the model boundary's shape-only-then-slice rule, so a
# row omitted by zero likelihood weight never reaches offset value validation.
# ``_row_index`` is the posterior primitive's rule for naming payload rows.
from superglm.distributional.model import (
    _prediction_offsets,
    _take_unvalidated_offsets,
    _unvalidated_offset_shapes,
)
from superglm.distributional.posterior import _row_index
from superglm.distributional.weights import (
    ResolvedLikelihoodWeights,
    UnsupportedLikelihoodContractError,
    resolve_likelihood_weights,
)

CrpsMethod = Literal["auto", "closed", "numeric"]

#: The quantile-score integral is mapped to ``t`` by ``p = Phi(t)`` and truncated
#: here.  Measured against the closed forms on 400 seeded rows per family, the
#: truncation costs 1e-10 relative at 5.0 and 1e-14 at 6.0, so 6.0 is the point
#: where the tail stops mattering and the node budget starts to.
_TAIL_LIMIT = 6.0
#: Probabilities are pulled inside ``(0, 1)`` by this margin before the probit.
_PROBABILITY_MARGIN = 1.0e-15
_INVERSE_SQRT_PI = float(1.0 / np.sqrt(np.pi))
_SQRT_TWO_PI = float(np.sqrt(2.0 * np.pi))


def _standard_normal_pdf(values: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.exp(-0.5 * values * values) / _SQRT_TWO_PI


def _validated_scoring_inputs(
    y: NDArray, theta: NDArray
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(response, theta)`` with one finite response value per row."""
    parameters = np.asarray(theta, dtype=np.float64)
    if parameters.ndim != 2 or parameters.shape[0] < 1:
        raise ValueError("theta must be a (rows, parameters) matrix with at least one row")
    response = np.asarray(y, dtype=np.float64)
    if response.ndim == 0:
        response = np.full(parameters.shape[0], float(response), dtype=np.float64)
    if response.shape != (parameters.shape[0],):
        raise ValueError("a score needs one response value per row of theta")
    return response, parameters


# --------------------------------------------------------------------------
# Closed forms, keyed by family class name
# --------------------------------------------------------------------------


def _gaussian_crps(family: Any, y: NDArray, theta: NDArray) -> NDArray[np.float64]:
    """Gneiting and Raftery (2007), equation (21), in ``(location, scale)``."""
    response, parameters = _validated_scoring_inputs(y, theta)
    location, scale = parameters[:, 0], parameters[:, 1]
    z = (response - location) / scale
    return scale * (
        z * (2.0 * special.ndtr(z) - 1.0) + 2.0 * _standard_normal_pdf(z) - _INVERSE_SQRT_PI
    )


def _gamma_crps(family: Any, y: NDArray, theta: NDArray) -> NDArray[np.float64]:
    """Scheuerer and Moeller (2015) in the family's ``(mean, cv)`` coordinates.

    With shape ``k = 1 / cv^2`` and scale ``s = mean * cv^2``,
    ``CRPS = y (2 F_k(y/s) - 1) - k s (2 F_{k+1}(y/s) - 1) - s / B(1/2, k)``,
    where ``F_k`` is the regularised lower incomplete gamma in shape ``k``.
    """
    response, parameters = _validated_scoring_inputs(y, theta)
    squared_cv = parameters[:, 1] * parameters[:, 1]
    shape = 1.0 / squared_cv
    scale = parameters[:, 0] * squared_cv
    ratio = response / scale
    return (
        response * (2.0 * special.gammainc(shape, ratio) - 1.0)
        - shape * scale * (2.0 * special.gammainc(shape + 1.0, ratio) - 1.0)
        - scale / special.beta(0.5, shape)
    )


def _log_normal_crps(family: Any, y: NDArray, theta: NDArray) -> NDArray[np.float64]:
    """Baran and Lerch (2015) in ``(mu, sigma)`` of ``log Y``.

    The family reports either ``E[Y]`` or ``mu`` first depending on its
    parametrisation; ``mu = log m - sigma^2 / 2`` converts the mean form, which
    is the kernel's own conversion rather than a restatement of it.
    """
    response, parameters = _validated_scoring_inputs(y, theta)
    sigma = parameters[:, 1]
    location = (
        np.asarray(location_of_mean(parameters[:, 0], sigma), dtype=np.float64)
        if family.parametrisation == "mean"
        else parameters[:, 0]
    )
    z = (np.log(response) - location) / sigma
    return response * (2.0 * special.ndtr(z) - 1.0) - 2.0 * np.exp(
        location + 0.5 * sigma * sigma
    ) * (special.ndtr(z - sigma) + special.ndtr(sigma / np.sqrt(2.0)) - 1.0)


#: Closed-form CRPS by family class name.  A family absent from the catalogue is
#: scored by the quantile-score integral, which needs only its quantile function.
_CLOSED_FORMS: Mapping[str, Callable[[Any, NDArray, NDArray], NDArray[np.float64]]] = {
    "GaussianLS": _gaussian_crps,
    "GammaLS": _gamma_crps,
    "LogNormalLS": _log_normal_crps,
}


def has_closed_form_crps(family: Any) -> bool:
    """Report whether the catalogue holds a closed-form CRPS for this family."""
    return type(family).__name__ in _CLOSED_FORMS


def crps_closed_form(family: Any, y: NDArray, theta: NDArray) -> NDArray[np.float64]:
    """Evaluate the catalogued closed-form CRPS per row.

    ``theta`` is the ``(n, k)`` natural-parameter matrix in the family's own
    coordinates and must lie inside its support; ``y`` broadcasts to ``n``.
    """
    name = type(family).__name__
    formula = _CLOSED_FORMS.get(name)
    if formula is None:
        raise NotImplementedError(
            f"{name} has no catalogued closed-form CRPS; score it with the numeric "
            "quantile-score integral (crps_numeric or method='numeric')"
        )
    return formula(family, y, theta)


# --------------------------------------------------------------------------
# The quantile-score integral
# --------------------------------------------------------------------------


def _panel_boundary(
    family: Any, points: NDArray[np.float64], theta: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return ``t = Phi^-1(F(points))`` clipped into the truncated ``t`` range."""
    probability = np.clip(
        np.asarray(family.cdf(points, theta), dtype=np.float64),
        _PROBABILITY_MARGIN,
        1.0 - _PROBABILITY_MARGIN,
    )
    return np.clip(special.ndtri(probability), -_TAIL_LIMIT, _TAIL_LIMIT)


def crps_numeric(
    family: Any,
    y: NDArray,
    theta: NDArray,
    *,
    n_nodes: int = 64,
    threshold: float | None = None,
) -> NDArray[np.float64]:
    """Integrate the quantile score, optionally weighted by ``1{Q(p) > threshold}``.

    The substitution ``p = Phi(t)`` on ``t`` in ``[-6, 6]`` turns the integral
    into a Gaussian-weighted one that Gauss-Legendre handles well, *except* at
    the kink where ``Q(p)`` crosses ``y``: the indicator turns over there and a
    single panel converges only algebraically (measured on the Gaussian: 8e-2
    relative at 32 nodes, still 4e-4 at 512).  Splitting the range at
    ``t* = Phi^-1(F(y))`` per row puts the kink on a panel edge and the same 32
    nodes reach 2e-15.  A finite ``threshold`` adds its own edge at
    ``Phi^-1(F(threshold))``, the point where the weight turns over, so the
    weighted integrand is smooth inside every panel as well.

    ``threshold=-inf`` restores the unweighted CRPS exactly: the extra edge is
    not added and the mask passes every node through unchanged.
    """
    if not isinstance(family, DistributionFunctionFamily):
        raise NotImplementedError(
            "the numeric CRPS needs a family with a cdf and a quantile function; this one has neither"
        )
    order = int(n_nodes)
    if order < 1:
        raise ValueError("n_nodes must be a positive Gauss-Legendre order")
    response, parameters = _validated_scoring_inputs(y, theta)
    n_observations = parameters.shape[0]

    interior = [_panel_boundary(family, response, parameters)]
    if threshold is not None and np.isfinite(threshold):
        interior.append(
            _panel_boundary(
                family, np.full(n_observations, float(threshold), dtype=np.float64), parameters
            )
        )
    edges = np.column_stack(
        (
            np.full(n_observations, -_TAIL_LIMIT),
            np.sort(np.column_stack(interior), axis=1),
            np.full(n_observations, _TAIL_LIMIT),
        )
    )

    nodes, node_weights = np.polynomial.legendre.leggauss(order)
    total = np.zeros(n_observations, dtype=np.float64)
    for panel in range(edges.shape[1] - 1):
        half = 0.5 * (edges[:, panel + 1] - edges[:, panel])
        centre = 0.5 * (edges[:, panel + 1] + edges[:, panel])
        for node, node_weight in zip(nodes, node_weights, strict=True):
            abscissa = centre + half * node
            probability = special.ndtr(abscissa)
            quantile = np.asarray(family.quantile(probability, parameters), dtype=np.float64)
            integrand = (
                (np.where(response < quantile, 1.0, 0.0) - probability)
                * (quantile - response)
                * _standard_normal_pdf(abscissa)
            )
            if threshold is not None:
                integrand = np.where(quantile > threshold, integrand, 0.0)
            total += (half * node_weight) * integrand
    return 2.0 * total


# --------------------------------------------------------------------------
# Model-facing scores
# --------------------------------------------------------------------------


class _ScoringRows(NamedTuple):
    """Retained call rows with predictive-law and aggregation weights separated."""

    frame: EagerFrame
    response: NDArray[np.float64]
    offsets: Mapping[str, NDArray[np.float64]]
    resolved: ResolvedLikelihoodWeights
    positions: NDArray[np.intp]
    n_observations: int
    aggregation_mass: NDArray[np.float64]
    prior_law: NDArray[np.float64] | None


class _PriorWeightedRowLaw:
    """Bind a family's prior-weighted CDF and quantile to retained row weights."""

    def __init__(
        self, family: PriorWeightedDistributionFunctionFamily, weights: NDArray[np.float64]
    ) -> None:
        self._family = family
        self._weights = weights

    def cdf(self, y: NDArray, theta: NDArray) -> NDArray:
        return self._family.cdf_prior_weighted(y, theta, self._weights)

    def quantile(self, p: NDArray, theta: NDArray) -> NDArray:
        return self._family.quantile_prior_weighted(p, theta, self._weights)


def _scoring_rows(
    fitted: Any,
    X: FrameLike | EagerFrame,
    y: NDArray,
    *,
    sample_weight: NDArray | None,
    offsets: Mapping[str, NDArray] | None,
) -> _ScoringRows:
    """Resolve one scoring call and omit zero-weight rows before any prediction."""
    frame = as_eager_frame(X)
    n_observations = len(frame)
    response = np.asarray(y, dtype=np.float64)
    if response.ndim == 0:
        response = np.full(n_observations, float(response), dtype=np.float64)
    if response.shape != (n_observations,):
        raise ValueError("a score needs one response value per row of X")

    shaped_offsets = _unvalidated_offset_shapes(offsets, n_observations)
    resolved = resolve_likelihood_weights(
        sample_weight,
        n_observations=n_observations,
        contract=fitted.fit_state.weight_contract,
    )
    positions = np.asarray(resolved.input_positions, dtype=np.intp)
    retained_frame = as_eager_frame(frame.take_rows(positions))
    retained_response = np.array(response[positions], copy=True)
    predictor_names = tuple(state.name for state in fitted.layout.predictors)
    retained_offsets = _prediction_offsets(
        _take_unvalidated_offsets(shaped_offsets, positions),
        predictor_names,
        len(positions),
    )
    semantics = resolved.provenance.contract.semantics
    aggregation_mass = (
        np.ones(len(positions), dtype=np.float64)
        if semantics == "prior"
        else np.asarray(resolved.values, dtype=np.float64)
    )
    prior_law = (
        np.asarray(resolved.values, dtype=np.float64)
        if semantics == "prior" and not resolved.provenance.all_unit
        else None
    )
    return _ScoringRows(
        frame=retained_frame,
        response=retained_response,
        offsets=retained_offsets,
        resolved=resolved,
        positions=positions,
        n_observations=n_observations,
        aggregation_mass=aggregation_mass,
        prior_law=prior_law,
    )


def _row_law(fitted: Any, rows: _ScoringRows) -> DistributionFunctionFamily:
    """Return the retained rows' unit or prior-weighted predictive law."""
    family = fitted.family
    if not isinstance(family, DistributionFunctionFamily):
        raise NotImplementedError(
            "a proper distribution score needs a family with a cdf and a quantile function; "
            "this one has neither"
        )
    if rows.prior_law is None:
        return family
    if not isinstance(family, PriorWeightedDistributionFunctionFamily):
        # Let the family state its own likelihood-contract refusal when it has
        # one.  LogNormalLS, for example, explains why its non-unit prior law
        # does not exist rather than being reduced to a generic score error.
        fitted.family.bind_likelihood(rows.response, rows.resolved, COMPLETE_OBSERVATION)
        raise UnsupportedLikelihoodContractError(
            f"{type(family).__name__} has no prior-weighted distribution function, so "
            "non-unit prior weights cannot be scored"
        )
    return _PriorWeightedRowLaw(family, rows.prior_law)


def _closed_form_parameters(
    family: Any,
    theta: NDArray[np.float64],
    prior_law: NDArray[np.float64] | None,
) -> NDArray[np.float64] | None:
    """Return natural parameters for the same prior-weighted reproductive law."""
    if prior_law is None:
        return theta
    if type(family).__name__ not in {"GaussianLS", "GammaLS"}:
        return None
    parameters = np.array(theta, dtype=np.float64, copy=True)
    parameters[:, 1] /= np.sqrt(prior_law)
    return parameters


def _retained_log_score(fitted: Any, rows: _ScoringRows) -> NDArray[np.float64]:
    """Return compressed likelihood contributions on retained physical rows."""
    plan = fitted.family.bind_likelihood(rows.response, rows.resolved, COMPLETE_OBSERVATION)
    theta = np.asarray(
        fitted.predict_parameters(rows.frame, offsets=rows.offsets), dtype=np.float64
    )
    evaluation = fitted.family.evaluate_natural(rows.response, theta, plan, derivative_order=0)
    return -np.asarray(evaluation.reported_log_likelihood, dtype=np.float64)


def _retained_crps(
    fitted: Any,
    rows: _ScoringRows,
    *,
    method: CrpsMethod,
    n_nodes: int,
    threshold: float | None = None,
) -> NDArray[np.float64]:
    """Return the unaggregated CRPS of each retained row's predictive law."""
    theta = np.asarray(
        fitted.predict_parameters(rows.frame, offsets=rows.offsets), dtype=np.float64
    )
    row_law = _row_law(fitted, rows)
    if threshold is not None:
        return crps_numeric(
            row_law,
            rows.response,
            theta,
            n_nodes=n_nodes,
            threshold=threshold,
        )
    if method == "numeric":
        return crps_numeric(row_law, rows.response, theta, n_nodes=n_nodes)

    closed_theta = _closed_form_parameters(fitted.family, theta, rows.prior_law)
    if method == "closed":
        if closed_theta is None:
            raise NotImplementedError(
                f"{type(fitted.family).__name__} has no catalogued prior-weighted closed-form CRPS"
            )
        return crps_closed_form(fitted.family, rows.response, closed_theta)
    if has_closed_form_crps(fitted.family) and closed_theta is not None:
        return crps_closed_form(fitted.family, rows.response, closed_theta)
    return crps_numeric(row_law, rows.response, theta, n_nodes=n_nodes)


def _aligned_contributions(
    rows: _ScoringRows, retained_scores: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Restore retained compressed contributions to the input row positions."""
    values = np.asarray(retained_scores, dtype=np.float64)
    if values.shape != rows.aggregation_mass.shape:
        raise ValueError("a retained score must provide one value per retained row")
    scores = np.full(rows.n_observations, np.nan, dtype=np.float64)
    scores[rows.positions] = rows.aggregation_mass * values
    return scores


def log_score(
    fitted: Any,
    X: FrameLike | EagerFrame,
    y: NDArray,
    *,
    sample_weight: NDArray | None = None,
    offsets: Mapping[str, NDArray] | None = None,
) -> NDArray[np.float64]:
    """Return ``-log f(y | theta_hat)`` per row under the fit's weight contract.

    Weights are read exactly as the fit reads them: ``"prior"`` semantics scale
    a row's dispersion, ``"frequency"`` semantics replicate it, and a zero weight
    drops the row from the likelihood altogether.  Dropped rows score ``nan``
    rather than silently disappearing, so the result always aligns with ``X``.
    """
    rows = _scoring_rows(
        fitted,
        X,
        y,
        sample_weight=sample_weight,
        offsets=offsets,
    )
    scores = np.full(rows.n_observations, np.nan, dtype=np.float64)
    # The likelihood already reports count × unit log score under frequency
    # semantics, so restoring it directly is essential: multiplying by the
    # aggregation mass here would square the replication count.
    scores[rows.positions] = _retained_log_score(fitted, rows)
    return scores


def crps(
    fitted: Any,
    X: FrameLike | EagerFrame,
    y: NDArray,
    *,
    sample_weight: NDArray | None = None,
    offsets: Mapping[str, NDArray] | None = None,
    method: CrpsMethod = "auto",
    n_nodes: int = 64,
) -> NDArray[np.float64]:
    """Return the CRPS per row of ``X``.

    ``method="auto"`` takes a catalogued closed form for the row law when one is
    available and the quantile-score integral otherwise; ``"closed"`` refuses
    rather than falling back, and ``"numeric"`` forces the integral (which is
    how the tests hold the two routes against each other).  Prior weights alter
    the row law; frequency weights multiply the unit-law score by their literal
    replication count.  Zero-weight rows are returned as ``nan``.
    """
    if method not in ("auto", "closed", "numeric"):
        raise ValueError("method must be 'auto', 'closed' or 'numeric'")
    rows = _scoring_rows(
        fitted,
        X,
        y,
        sample_weight=sample_weight,
        offsets=offsets,
    )
    return _aligned_contributions(
        rows,
        _retained_crps(fitted, rows, method=method, n_nodes=n_nodes),
    )


def threshold_weighted_crps(
    fitted: Any,
    X: FrameLike | EagerFrame,
    y: NDArray,
    threshold: float,
    *,
    sample_weight: NDArray | None = None,
    offsets: Mapping[str, NDArray] | None = None,
    n_nodes: int = 64,
) -> NDArray[np.float64]:
    """Return the tail-weighted CRPS with the indicator weight ``1{z > threshold}``.

    Gneiting and Ranjan (2011).  The weight is a modelling choice made before the
    data are seen; ``threshold=-inf`` recovers the unweighted CRPS exactly.
    Prior weights select the prior-weighted predictive law, while frequency
    weights compress the repeated unit-law score into its source row.
    """
    rows = _scoring_rows(
        fitted,
        X,
        y,
        sample_weight=sample_weight,
        offsets=offsets,
    )
    return _aligned_contributions(
        rows,
        _retained_crps(
            fitted,
            rows,
            method="numeric",
            n_nodes=n_nodes,
            threshold=float(threshold),
        ),
    )


def score_table(
    fitted: Any,
    X: FrameLike | EagerFrame,
    y: NDArray,
    *,
    which: Sequence[str] = ("log", "crps"),
    thresholds: Sequence[float] = (),
    sample_weight: NDArray | None = None,
    offsets: Mapping[str, NDArray] | None = None,
    method: CrpsMethod = "auto",
    n_nodes: int = 64,
) -> pd.DataFrame:
    """Score every row into one frame under the fitted model's weight contract.

    Columns follow ``which`` in ``("log", "crps")`` order and then one
    ``twcrps_<threshold>`` column per entry of ``thresholds``.  A frequency row
    contains its compressed ``count × unit_score`` contribution; a zero-weight
    row is ``nan`` in every column.
    """
    names = tuple(which)
    if not names:
        raise ValueError("score_table needs at least one score name")
    unknown = tuple(name for name in names if name not in ("log", "crps"))
    if unknown:
        raise ValueError(f"unknown score name: {', '.join(unknown)}")

    rows = _scoring_rows(
        fitted,
        X,
        y,
        sample_weight=sample_weight,
        offsets=offsets,
    )
    columns: dict[str, NDArray[np.float64]] = {}
    if "log" in names:
        log_values = np.full(rows.n_observations, np.nan, dtype=np.float64)
        log_values[rows.positions] = _retained_log_score(fitted, rows)
        columns["log"] = log_values
    if "crps" in names:
        if method not in ("auto", "closed", "numeric"):
            raise ValueError("method must be 'auto', 'closed' or 'numeric'")
        columns["crps"] = _aligned_contributions(
            rows,
            _retained_crps(fitted, rows, method=method, n_nodes=n_nodes),
        )
    for threshold in thresholds:
        value = float(threshold)
        columns[f"twcrps_{value:g}"] = _aligned_contributions(
            rows,
            _retained_crps(
                fitted,
                rows,
                method="numeric",
                n_nodes=n_nodes,
                threshold=value,
            ),
        )
    return pd.DataFrame(columns, index=_row_index(X, rows.n_observations))


__all__ = [
    "crps",
    "crps_closed_form",
    "crps_numeric",
    "has_closed_form_crps",
    "log_score",
    "score_table",
    "threshold_weighted_crps",
]
