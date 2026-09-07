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

with an established bound or correction for the omitted tails. Clamping both
``y`` and ``Q(p)`` to ``t`` gives the threshold-weighted CRPS with indicator
weight, ``integral_t^infinity (F(z) - 1{y <= z})^2 dz`` (Gneiting and Ranjan,
2011; Allen, Ginsbourger and Ziegel, 2023, Proposition 1).

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
from superglm.distributional.families.generalized_gamma import GeneralizedGammaLSS
from superglm.distributional.families.generalized_pareto import GeneralizedParetoLSS
from superglm.distributional.family import (
    COMPLETE_OBSERVATION,
    DistributionFunctionFamily,
    PriorWeightedDistributionFunctionFamily,
    PriorWeightedVarianceFamily,
    VarianceFamily,
    validated_parameter_matrix,
)
from superglm.distributional.kernels.generalized_gamma import (
    location_of_mean as generalized_gamma_location_of_mean,
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

#: Finite-variance rows bound the omitted tails at this probit endpoint.
#: Known heavy tails use 4 on the right plus an analytic correction, avoiding
#: the loss of relative precision in probabilities very close to one.
_TAIL_LIMIT = 6.0
_CORRECTED_TAIL_LIMIT = 4.0
_NUMERIC_RTOL = float(np.sqrt(np.finfo(np.float64).eps))
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
    if not np.all(np.isfinite(response)):
        raise ValueError("scoring responses must be finite")
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
    ratio = np.maximum(response, 0.0) / scale
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
    z = np.full(response.shape, -np.inf)
    positive = response > 0.0
    z[positive] = (np.log(response[positive]) - location[positive]) / sigma[positive]
    return response * (2.0 * special.ndtr(z) - 1.0) - 2.0 * np.exp(
        location + 0.5 * sigma * sigma
    ) * (special.ndtr(z - sigma) + special.ndtr(sigma / np.sqrt(2.0)) - 1.0)


#: Closed-form CRPS by family class name.  A family absent from the catalogue is
#: scored numerically when an established tail bound or correction is available.
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
    values = validated_parameter_matrix(
        theta, n_observations=None, parameters=family.parameters, family_name=name
    )
    return formula(family, y, values)


# --------------------------------------------------------------------------
# The quantile-score integral
# --------------------------------------------------------------------------


def _panel_boundary(
    probability: NDArray[np.float64], upper: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return the probit CDF edge clipped to this row's finite panels."""
    probability = np.clip(probability, _PROBABILITY_MARGIN, 1.0 - _PROBABILITY_MARGIN)
    return np.clip(special.ndtri(probability), -_TAIL_LIMIT, upper)


def _score_values(values: Any, n: int, name: str) -> NDArray[np.float64]:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (n,) or not np.all(np.isfinite(result)):
        raise ValueError(
            f"numeric CRPS needs finite row-shaped {name}; numerical evaluation unresolved"
        )
    if name == "CDF" and np.any((result < 0.0) | (result > 1.0)):
        raise ValueError("numeric CRPS CDF values must lie in [0, 1]")
    return result


def _variance_for_score(family: Any, theta: NDArray) -> NDArray[np.float64]:
    """Read the variance of the same predictive law as the quantile panels."""
    if isinstance(family, _PriorWeightedRowLaw):
        if not isinstance(family._family, PriorWeightedVarianceFamily):
            raise NotImplementedError("numeric CRPS needs an established prior-law tail bound")
        variance = family._family.variance_prior_weighted(theta, family._weights)
    elif isinstance(family, VarianceFamily):
        variance = family.variance(theta)
    else:
        raise NotImplementedError(
            "numeric CRPS needs finite variance or a known-law tail correction"
        )
    values = np.asarray(variance, dtype=np.float64)
    if values.shape != (len(theta),) or np.any(np.isnan(values) | (values < 0)):
        raise ValueError("numeric CRPS variance bound is numerically unresolved")
    if not np.all(np.isfinite(values)):
        raise NotImplementedError(
            "numeric CRPS has no established tail bound for infinite variance"
        )
    return values


def _generalized_gamma_tail_gap(theta: NDArray) -> NDArray[np.float64]:
    """Return 2 - sigma*abs(Q), with exact represented-product classification.

    Integer ratios also preserve the small positive denominator of a finite
    tail when the rounded floating product would equal two.
    """
    gap = np.full(len(theta), 2.0)
    for row in np.flatnonzero(theta[:, 2] < 0):
        sn, sd = float(theta[row, 1]).as_integer_ratio()
        qn, qd = float(-theta[row, 2]).as_integer_ratio()
        denominator = sd * qd
        numerator = 2 * denominator - sn * qn
        gap[row] = numerator / denominator if numerator > 0 else 0.0
    return gap


def _positive_score_tail(
    family: Any, theta: NDArray, endpoint: NDArray, gap: NDArray
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Squared-survival integral R(B) and a bounded series/rounding allowance.

    For negative-Q generalized gamma, S(x)=P(k,z(x)), k=Q^-2. Writing
    H=Gamma(k+1) P(k,z)/z^k gives c_j=(-1)^j k/((k+j) j!). Integrating
    H_16^2 termwise weights each coefficient pair by h/(h+i+j), h=k*gap.
    For z<=1 the alternating remainder is at most z^17/17!, so the tail
    error is bounded by U*(2*r+r*r), plus a floating arithmetic allowance.
    This is a local tail formula, not a certificate for black-box quadrature.
    """
    eps = np.finfo(np.float64).eps
    if type(family) is GeneralizedParetoLSS:
        psi, xi = theta.T
        # Remove response units before multiplying by a tiny shape: xi*B can
        # underflow although the dimensionless xi*(B/psi) is representable.
        log_base = np.log1p(xi * (endpoint / psi))
        # Divide the small log1p first: 2/xi can overflow near the exponential
        # limit even though the log tail and score remain representable.
        log_tail = np.log(psi) - np.log(2.0 - xi) + (xi - 2.0) * (log_base / xi)
        value = np.exp(log_tail)
        error = value * (32 * eps * (1.0 + np.abs(log_tail) + np.abs(np.log(psi))))
    else:
        location = (
            generalized_gamma_location_of_mean(theta[:, 0], theta[:, 1], theta[:, 2])
            if family.parametrisation == "mean"
            else theta[:, 0]
        )
        value, error = np.empty(len(theta)), np.empty(len(theta))
        for row, (mu, sigma, q) in enumerate(zip(location, theta[:, 1], theta[:, 2], strict=True)):
            k = 1.0 / (q * q)
            log_b = np.log(endpoint[row])
            log_z = np.log(k) + abs(q) / sigma * (mu - log_b)
            z = np.exp(log_z)
            if not np.isfinite(z) or z > 1.0:
                raise ValueError(
                    "numeric CRPS generalized-gamma tail is unresolved (requires z<=1)"
                )
            h = k * gap[row]
            log_denominator = np.log(gap[row]) - np.log(sigma) - np.log(abs(q))
            log_u = log_b + 2.0 * k * log_z - log_denominator - 2.0 * special.gammaln(k + 1.0)
            u = np.exp(log_u)
            terms = np.ones(17)
            for j in range(1, len(terms)):
                terms[j] = -terms[j - 1] * z / j * ((k + j - 1) / (k + j))
            degrees = np.arange(len(terms))
            summands = np.outer(terms, terms) * (h / (h + degrees[:, None] + degrees[None, :]))
            remainder = z**17 / float(special.factorial(17, exact=True))
            arithmetic = (
                128
                * eps
                * (
                    1.0
                    + abs(log_b)
                    + abs(2 * k * log_z)
                    + abs(log_denominator)
                    + abs(2 * special.gammaln(k + 1))
                )
            )
            value[row] = u * np.sum(summands)
            error[row] = u * (2 * remainder + remainder**2 + arithmetic * np.sum(np.abs(summands)))
    if np.any(~np.isfinite(value) | (value <= 0) | ~np.isfinite(error)):
        raise ValueError("numeric CRPS finite tail is outside the representable numerical range")
    return value, error


def _quantile_panels(
    family: Any,
    response: NDArray,
    theta: NDArray,
    edges: NDArray,
    order: int,
    threshold: float | None,
) -> NDArray[np.float64]:
    nodes, node_weights = np.polynomial.legendre.leggauss(order)
    total = np.zeros(len(theta), dtype=np.float64)
    for panel in range(edges.shape[1] - 1):
        half = 0.5 * (edges[:, panel + 1] - edges[:, panel])
        centre = 0.5 * (edges[:, panel + 1] + edges[:, panel])
        for node, node_weight in zip(nodes, node_weights, strict=True):
            abscissa = centre + half * node
            probability = special.ndtr(abscissa)
            quantile = _score_values(family.quantile(probability, theta), len(theta), "quantiles")
            if threshold is not None:
                quantile = np.maximum(quantile, threshold)
            # Evaluate 1-Phi(t) without cancellation in the upper panels.
            residual = np.where(response < quantile, special.ndtr(-abscissa), -probability)
            integrand = residual * (quantile - response) * _standard_normal_pdf(abscissa)
            total += (2.0 * half * node_weight) * integrand
    return _score_values(total, len(theta), "panel integrals")


def crps_numeric(
    family: Any,
    y: NDArray,
    theta: NDArray,
    *,
    n_nodes: int = 64,
    threshold: float | None = None,
) -> NDArray[np.float64]:
    """Integrate CRPS, optionally with CDF weight ``1{z > threshold}``.

    Both response and quantiles are clamped to a finite threshold. Minus
    infinity is exactly the unweighted path; plus infinity scores zero.
    Finite probit panels split at F(y) and F(threshold), with up to three
    doublings of ``n_nodes``. Successive estimates are a convergence diagnostic,
    not a rigorous quadrature enclosure. Independent omitted-tail bounds use
    finite predictive variance, or explicit GPD/negative-Q generalized-gamma
    formulas. Tail and panel budgets are respectively 1/4 and 1/2 of sqrt(eps)
    times max(IQR, distance from the clamped median, current score).

    A law without an established tail bound raises NotImplementedError; an
    unresolved finite numerical evaluation raises ValueError. Negative-Q
    generalized gamma has infinite CRPS exactly when sigma*abs(Q)>=2, even
    though its mean already diverges at one. Extreme finite rows outside the
    compact tail evaluator's domain can refuse explicitly.
    """
    if not isinstance(family, DistributionFunctionFamily):
        raise NotImplementedError(
            "the numeric CRPS needs a family with a cdf and a quantile function; this one has neither"
        )
    order = int(n_nodes)
    if order < 1:
        raise ValueError("n_nodes must be a positive Gauss-Legendre order")
    response, parameters = _validated_scoring_inputs(y, theta)
    if threshold is not None and np.isnan(threshold):
        raise ValueError("threshold must not be NaN")
    if threshold == -np.inf:
        threshold = None
    n = len(parameters)
    # Calling the actual law validates natural parameters (and the mean-form
    # GG domain), including for +infinity thresholds. Responses outside the
    # predictive support are legitimate here; likelihood validation is not.
    cdf_y = _score_values(family.cdf(response, parameters), n, "CDF")
    if threshold == np.inf:
        return np.zeros(n)

    gap = np.full(n, 2.0)
    corrected = np.full(n, type(family) is GeneralizedParetoLSS)
    if type(family) is GeneralizedGammaLSS:
        q = parameters[:, 2]
        if np.any((q != 0) & (np.abs(q) < 1e-8)):
            raise ValueError("numeric CRPS tail is unresolved for the near-zero-Q approximation")
        gap = _generalized_gamma_tail_gap(parameters)
        divergent = gap <= 0
        if np.any(divergent):
            scores = np.full(n, np.inf)
            if np.any(~divergent):
                scores[~divergent] = crps_numeric(
                    family,
                    response[~divergent],
                    parameters[~divergent],
                    n_nodes=order,
                    threshold=threshold,
                )
            return scores
        corrected = (q < 0) & (gap <= 1.5)  # infinite second moment, not infinite CRPS

    # Numeric overflow is an explicit refusal, never a replacement infinity.
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        try:
            return _finite_crps_numeric(
                family, response, parameters, cdf_y, corrected, gap, order, threshold
            )
        except FloatingPointError as exc:
            raise ValueError(
                "numeric CRPS finite evaluation is unresolved in the numerical range"
            ) from exc


def _finite_crps_numeric(
    family: Any,
    response: NDArray,
    parameters: NDArray,
    cdf_y: NDArray,
    corrected: NDArray,
    gap: NDArray,
    order: int,
    threshold: float | None,
) -> NDArray[np.float64]:
    n = len(parameters)
    upper = np.where(corrected, _CORRECTED_TAIL_LIMIT, _TAIL_LIMIT)
    delta_l = float(special.ndtr(-_TAIL_LIMIT))
    delta_u = 1.0 - special.ndtr(upper)
    quantiles = [
        _score_values(family.quantile(np.full(n, p), parameters), n, "quantiles")
        for p in (0.25, 0.5, 0.75, delta_l)
    ]
    q25, median, q75, a = quantiles
    b = _score_values(family.quantile(1.0 - delta_u, parameters), n, "quantiles")
    if np.any((a > q25) | (q25 > median) | (median > q75) | (q75 > b)):
        raise ValueError("numeric CRPS quantiles are not ordered; numerical evaluation unresolved")
    response_scale = q75 - q25
    interior = [_panel_boundary(cdf_y, upper)]
    lower_cdf_bound = np.zeros(n)
    upper_cdf_bound = np.zeros(n)
    ordinary = ~corrected
    if np.any(ordinary):
        # Wrappers retain all row weights, so never slice their theta here.
        variance = _variance_for_score(
            family, parameters if np.all(ordinary) else parameters[ordinary]
        )
        root_variance = np.sqrt(variance)
        lower_cdf_bound[ordinary] = root_variance * np.sqrt(delta_l**3 / (1.0 - delta_l))
        upper_cdf_bound[ordinary] = root_variance * np.sqrt(
            delta_u[ordinary] ** 3 / (1.0 - delta_u[ordinary])
        )
    lower_cdf_bound[corrected] = np.maximum(a[corrected], 0.0) * delta_l**2
    if threshold is not None:
        cdf_t = _score_values(family.cdf(np.full(n, threshold), parameters), n, "CDF")
        interior.append(_panel_boundary(cdf_t, upper))
        lower_cdf_bound[a <= threshold] = 0.0
        a, b = np.maximum(a, threshold), np.maximum(b, threshold)
        response, median = np.maximum(response, threshold), np.maximum(median, threshold)
    edges = np.column_stack(
        (np.full(n, -_TAIL_LIMIT), np.sort(np.column_stack(interior), axis=1), upper)
    )
    correction = np.zeros(n)
    # These omission bounds remain valid when y lies outside [A,B], including
    # below-support responses and endpoints at the clamp's atom.
    tail_error = (
        lower_cdf_bound
        + delta_l**2 * np.abs(response - a)
        + 2 * delta_l * np.maximum(a - response, 0)
    )
    tail_error[ordinary] += (
        upper_cdf_bound[ordinary]
        + delta_u[ordinary] ** 2 * np.abs(b[ordinary] - response[ordinary])
        + 2 * delta_u[ordinary] * np.maximum(response[ordinary] - b[ordinary], 0)
    )
    if np.any(corrected):
        if np.any(response[corrected] > b[corrected]):
            raise ValueError(
                "numeric CRPS tail is unresolved for a response beyond the upper endpoint"
            )
        tail, error = _positive_score_tail(
            family, parameters[corrected], b[corrected], gap[corrected]
        )
        # R(B) alone is NOT the omitted quantile integral. This endpoint term
        # is essential even when the mean is infinite. Clamped B=t is an atom;
        # y<=t then makes the endpoint term zero and R(t) is the entire score.
        correction[corrected] = tail + delta_u[corrected] ** 2 * (
            b[corrected] - response[corrected]
        )
        tail_error[corrected] += error

    previous = _quantile_panels(family, response, parameters, edges, order, threshold)
    for _ in range(3):
        order *= 2
        current = _quantile_panels(family, response, parameters, edges, order, threshold)
        total = current + correction
        scale = np.maximum.reduce((response_scale, np.abs(response - median), total))
        target = _NUMERIC_RTOL * scale
        if not np.all(np.isfinite(total)) or np.any(tail_error > 0.25 * target):
            raise ValueError(
                "numeric CRPS omitted-tail bound is unresolved at the numerical tolerance"
            )
        if np.all(np.abs(current - previous) <= 0.5 * target):
            return total
        previous = current
    raise ValueError("numeric CRPS finite-panel quadrature is unresolved after bounded refinement")


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
    if not np.all(np.isfinite(retained_response)):
        raise ValueError("scoring responses must be finite")
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
        if closed_theta is None or not has_closed_form_crps(fitted.family):
            raise NotImplementedError(
                f"{type(fitted.family).__name__} has no catalogued prior-weighted closed-form CRPS"
            )
        # Prediction validated raw parameters; prior-law scales may legitimately
        # fall below the fitted family's raw parameter floor.
        return _CLOSED_FORMS[type(fitted.family).__name__](
            fitted.family, rows.response, closed_theta
        )
    if has_closed_form_crps(fitted.family) and closed_theta is not None:
        return _CLOSED_FORMS[type(fitted.family).__name__](
            fitted.family, rows.response, closed_theta
        )
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
    thresholds = tuple(float(value) for value in thresholds)
    threshold_names = [f"twcrps_{value:g}" for value in thresholds]
    if len(threshold_names) != len(set(threshold_names)):
        raise ValueError("thresholds produce duplicate score column names")

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
