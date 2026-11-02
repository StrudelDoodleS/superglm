"""Pure validation and normalization for public fit entry points."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from superglm._count_lattice import _not_a_whole_number
from superglm._frame import EagerFrame, as_eager_frame
from superglm._utils import contract_weights
from superglm.distributions import (
    Binomial,
    Distribution,
    NegativeBinomial,
    Poisson,
    validate_response,
)
from superglm.solvers.dispersion import FREQUENCY_WEIGHTS, PRIOR_WEIGHTS


class PriorWeightLatticeWarning(UserWarning):
    """A counting response or binomial weight lacks an exact probability law."""


class FractionalFrequencyWeightWarning(UserWarning):
    """A replication count that is not a whole number of observations."""


#: Absolute slack when testing a weight for unity.  Values sit near 1 here, so
#: magnitude scaling is not in play and a plain tolerance is the right shape.
_UNIT_WEIGHT_TOLERANCE = 1e-9

#: What the caller is doing with theta, which decides how far an interpolated
#: density reaches. The family alone cannot say: a finished auto-theta fit
#: carries a numeric theta, and scoring it refits nothing.
THETA_FIXED = "fixed"  #: given constant, or an already-fitted model being scored
THETA_ESTIMATED = "estimated"  #: profiled and fed back into IRLS (an auto fit)


def _interpolated_density_reach(family, *, theta_role: str) -> str:
    """How far an interpolated counting density reaches into the results.

    Wherever a counting family evaluates ``gammaln`` at a fractional argument.
    Under ``THETA_FIXED`` (a given theta, or a fitted model being scored) only
    the reported likelihood moves; under ``THETA_ESTIMATED`` theta is profiled
    and re-enters ``V(mu)`` and the IRLS weights, so the estimates move too.
    """
    if not isinstance(family, NegativeBinomial):
        return " Coefficients, fitted means and deviance are unaffected."
    if theta_role == THETA_FIXED:
        # No theta_hat here, but the interpolated NB factor
        # Gamma(wy + w theta) / Gamma(w theta) is theta-dependent, so any theta
        # inference drawn from this likelihood inherits the interpolation.
        return (
            " Coefficients, fitted means and deviance are unaffected. The "
            "interpolated factor is theta-dependent, so any profile of theta "
            "taken from this likelihood moves with it."
        )
    return (
        " theta_hat is profiled from that interpolated density and then "
        "enters the variance and the IRLS weights, so the coefficients, "
        "fitted means and deviance move as well."
    )


def _theta_role_for(family) -> str:
    """Whether this family will have theta profiled back into the fit.

    ``theta="auto"`` means the fitter will profile it and feed it back, so the
    estimates move.  Any numeric theta -- given by the caller, or stamped in by
    a finished auto fit -- is a constant from here on.
    """
    if isinstance(family, NegativeBinomial) and getattr(family, "theta", None) in ("auto", None):
        return THETA_ESTIMATED
    return THETA_FIXED


def _warn_frequency_counting_response(
    y: NDArray, weights: NDArray, family, theta_role: str | None = None
) -> None:
    """Warn when a replicated counting response is not a whole count.

    Under ``"frequency"`` the likelihood is ``w log f(y; mu)``, so the weight
    cannot rescue a fractional response to Poisson or the negative binomial:
    ``gammaln`` at a fractional argument is not a probability, and the reported
    log-likelihood, AIC and BIC are a quasi-likelihood.
    """
    # Carried rows only: a zero replication count contributes nothing.
    carried = weights > 0.0
    values = y[carried]
    count = int(np.count_nonzero(_not_a_whole_number(values)))
    if count == 0:
        return
    import warnings

    warnings.warn(
        f"{count} of {int(carried.sum())} carried rows have a non-integral response under "
        f'weight_semantics="frequency" with a {type(family).__name__} family, '
        "whose support is the non-negative integers. A replication count "
        "multiplies the per-row density rather than entering it, so it cannot "
        "put a fractional response back on the lattice: the density evaluates "
        "gammaln at a fractional argument, and the reported log-likelihood, "
        "AIC and BIC are a quasi-likelihood rather than an exact density."
        + _interpolated_density_reach(family, theta_role=theta_role or _theta_role_for(family))
        + ' Pass weight_semantics="prior" with y = count / exposure and '
        "sample_weight = exposure if the response is a rate.",
        PriorWeightLatticeWarning,
        stacklevel=5,
    )


def _warn_prior_weighted_binomial(weights: NDArray) -> None:
    """Warn when a prior-weighted binomial response cannot normalise.

    The prior construction is ``w Y ~ Binomial(w, mu)``, but ``y`` is pinned
    to ``{0, 1}``, so only the outcomes with masses ``(1 - mu)**w`` and
    ``mu**w`` are reachable, and they sum to 1 only at ``w == 1``. Unlike
    Poisson and the negative binomial, every non-unit prior weight on a
    binomial response therefore reports a sub- or super-probability.
    """
    # Carried rows only: a zero-weight binomial row has log density exactly 0.0.
    off = (weights > 0.0) & (np.abs(weights - 1.0) > _UNIT_WEIGHT_TOLERANCE)
    count = int(np.count_nonzero(off))
    if count == 0:
        return
    import warnings

    warnings.warn(
        f"{count} of {len(weights)} rows carry a non-unit sample_weight under "
        'weight_semantics="prior" with a Binomial family. The prior '
        "construction reads w as a trial count, but the response is pinned to "
        "{0, 1}, so only the all-failure and all-success outcomes are "
        "reachable and their masses ((1-mu)**w and mu**w) do not sum to one. "
        "Coefficients, fitted means and deviance are unaffected, but the "
        "reported log-likelihood, AIC and BIC are not an exact density. Use "
        'weight_semantics="frequency" if the weights are replication counts.',
        PriorWeightLatticeWarning,
        stacklevel=5,
    )


def _check_frequency_counts(weights: NDArray, weight_semantics: str) -> None:
    """Warn when a declared replication count is not a whole number.

    ``"frequency"`` says row ``i`` appears ``w_i`` times, which a fractional
    count cannot, so the likelihood criteria and the ``sum(w) - edf``
    residual d.f. are no longer exact. It warns rather than raises because
    the contract has always accepted fractional weights; under ``"prior"``
    they are well defined and nothing fires.
    """
    if weight_semantics != FREQUENCY_WEIGHTS:
        return
    carried = weights > 0.0
    values = weights[carried]
    count = int(np.count_nonzero(_not_a_whole_number(values)))
    if count == 0:
        return
    import warnings

    warnings.warn(
        f"{count} of {int(carried.sum())} carried rows have a non-integral "
        'sample_weight under weight_semantics="frequency", which declares a '
        "replication count. A row cannot appear a fractional number of times, "
        "so the reported log-likelihood, AIC, BIC and the sum(w) - edf "
        "residual degrees of freedom are a quasi-likelihood rather than exact. "
        'Pass weight_semantics="prior" if the weights are precisions or '
        "exposure -- fractional values are well defined there.",
        FractionalFrequencyWeightWarning,
        stacklevel=4,
    )


def _check_counting_lattice(
    y: NDArray,
    weights: NDArray,
    family,
    weight_semantics: str,
    *,
    theta_role: str | None = None,
) -> None:
    """Warn about replicated counting responses and prior binomial weights.

    Prior Poisson/NB rates may be deliberately adjusted, or carry round-off
    from dividing counts by exposure. They are accepted without an integrality
    warning; fitting and likelihood evaluation use the supplied values.
    """
    if weight_semantics == FREQUENCY_WEIGHTS:
        # Replication leaves each row's counting support unchanged.
        if not isinstance(family, Poisson | NegativeBinomial):
            return
        _warn_frequency_counting_response(y, weights, family, theta_role)
        return
    if weight_semantics != PRIOR_WEIGHTS:
        return
    if isinstance(family, Binomial):
        _warn_prior_weighted_binomial(weights)


def check_weight_contract(
    y: NDArray,
    weights: NDArray,
    family,
    weight_semantics: str,
    *,
    theta_role: str | None = None,
) -> None:
    """Check frequency replication and binomial likelihood interpretations.

    Fit and evaluation boundaries share this warning policy. Prior-weighted
    Poisson/NB rates are accepted quietly, including fractional adjustments.
    """
    _check_frequency_counts(weights, weight_semantics)
    _check_counting_lattice(y, weights, family, weight_semantics, theta_role=theta_role)


@dataclass(frozen=True)
class ValidatedFitInput:
    """Normalized arrays that are safe to pass into design construction."""

    X: EagerFrame
    y: NDArray[np.float64]
    sample_weight: NDArray[np.float64]
    offset: NDArray[np.float64] | None


def _finite_vector(
    name: str,
    value,
    n_rows: int,
    *,
    require_nonempty: bool = False,
    check_finite: bool = True,
) -> NDArray[np.float64]:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a numeric one-dimensional array") from exc
    if require_nonempty and raw.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if raw.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real-valued")
    if getattr(raw.dtype, "kind", None) in {"M", "m"}:
        raise ValueError(f"{name} must contain only real numeric values")
    if len(raw) != n_rows:
        raise ValueError(f"{name} must have length {n_rows}, got {len(raw)}")
    try:
        normalized = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain only real numeric values") from exc
    if check_finite and not np.all(np.isfinite(normalized)):
        raise ValueError(f"{name} must contain only finite values")
    return normalized


def _reject_unhashable_values(name: object, values: NDArray) -> None:
    """Raise unless every element of an object column can serve as a level.

    Level tables, encoders and interaction keys all hash these values, so the
    test is `hash` itself; dtype inference cannot see a `str` subclass with
    `__hash__ = None` or a tuple nesting an unhashable member. `deque(...,
    maxlen=0)` drives the map in C, one `hash` call per element.
    """
    try:
        deque(map(hash, values), maxlen=0)
    except (TypeError, ValueError) as exc:
        # `hash` reports refusal as TypeError for the unhashable types and as
        # ValueError for the few objects that hash conditionally, such as a
        # memoryview over a writable buffer. Both mean the same thing here.
        raise ValueError(
            f"X column {name!r} must contain only scalar values or hashable tuple levels"
        ) from exc


def validate_x_columns(frame: EagerFrame, columns: Iterable[object]) -> None:
    """Validate configured model columns before feature construction or scoring."""
    for name in tuple(dict.fromkeys(columns)):
        values = frame.column_array(name)
        dtype_kind = getattr(values.dtype, "kind", None)
        inferred_dtype = ""
        object_has_complex = False
        if dtype_kind == "O":
            _reject_unhashable_values(name, values)
            inferred_dtype = pd.api.types.infer_dtype(values, skipna=True)
            object_has_complex = inferred_dtype == "complex" or (
                inferred_dtype.startswith("mixed")
                and any(isinstance(value, complex | np.complexfloating) for value in values)
            )
        if dtype_kind == "c" or object_has_complex:
            raise ValueError(f"X column {name!r} must be real-valued")
        numeric_like_object = inferred_dtype in {
            "decimal",
            "floating",
            "integer",
            "mixed-integer-float",
        }
        physical_numeric = dtype_kind in {"b", "i", "u", "f"}
        if (
            frame.column_kind(name) in {"numeric", "boolean"}
            or physical_numeric
            or numeric_like_object
        ):
            try:
                numeric = np.asarray(values, dtype=np.float64)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"X column {name!r} must contain only real numeric values"
                ) from exc
            if not np.all(np.isfinite(numeric)):
                raise ValueError(
                    f"X column {name!r} must contain only finite values; "
                    "non-finite or missing values are not allowed"
                )


def validate_prediction_offset(value, n_rows: int) -> NDArray[np.float64] | None:
    """Validate a public prediction offset against the scored frame."""
    if value is None:
        return None
    return _finite_vector("offset", value, n_rows)


def validate_fit_input(
    X,
    y,
    sample_weight,
    offset,
    *,
    family: Distribution,
    weight_semantics: str,
    required_columns: Iterable[object],
    check_all_columns: bool = False,
    theta_role: str | None = None,
) -> ValidatedFitInput:
    """Validate a public fit call before any feature is built or learned."""
    frame = as_eager_frame(X)
    if len(frame) == 0:
        raise ValueError("X must be non-empty")

    required = tuple(dict.fromkeys(required_columns))
    frame.require_columns(required)
    columns_to_check = frame.columns if check_all_columns else required
    validate_x_columns(frame, columns_to_check)

    n_rows = len(frame)
    # validate_response() performs the universal finite check together with the
    # family-domain check, so avoid scanning the response twice here.
    y_arr = _finite_vector("y", y, n_rows, require_nonempty=True, check_finite=False)
    weight_arr = contract_weights(
        sample_weight, n_rows, family=family, weight_semantics=weight_semantics
    )
    offset_arr = None if offset is None else _finite_vector("offset", offset, n_rows)
    validate_response(y_arr, family)
    # Contract warnings run after every ordinary validation, so under -W error
    # they never mask the ValueError for invalid input.
    check_weight_contract(y_arr, weight_arr, family, weight_semantics, theta_role=theta_role)
    return ValidatedFitInput(frame, y_arr, weight_arr, offset_arr)
