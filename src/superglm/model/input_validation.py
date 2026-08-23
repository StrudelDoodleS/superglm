"""Pure validation and normalization for public fit entry points."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from superglm._frame import EagerFrame, as_eager_frame
from superglm._utils import _validate_strict_prior_weights
from superglm.distributions import (
    Binomial,
    Distribution,
    NegativeBinomial,
    Poisson,
    Tweedie,
    validate_response,
)
from superglm.solvers.dispersion import PRIOR_WEIGHTS


class PriorWeightLatticeWarning(UserWarning):
    """A discrete family's prior-weighted response left its own support."""


#: Relative slack when testing ``w * y`` for integrality.  The product is
#: formed in floating point from two user arrays, so an exactly-integral
#: intent (``count / exposure`` times ``exposure``) can land a few ulps away.
_LATTICE_RELATIVE_TOLERANCE = 1e-9


def _warn_prior_weighted_binomial(weights: NDArray) -> None:
    """Warn when a prior-weighted binomial response cannot normalise.

    The prior construction is ``w Y ~ Binomial(w, mu)``, whose support is
    ``{0, 1/w, ..., 1}`` -- but ``validate_response`` pins ``y`` to ``{0, 1}``,
    so the only reachable outcomes are "no successes" and "all ``w``
    successes". Their masses are ``(1 - mu)**w`` and ``mu**w``, which sum to 1
    **only at w == 1**: at ``w = 3, mu = 0.4`` they sum to 0.28, and at
    fractional ``w`` they exceed 1 (``sqrt(mu) + sqrt(1 - mu)``). The binomial
    coefficient is exactly 1 at both endpoints, which makes each term look
    exact in isolation and is why this is easy to miss -- but an exact
    coefficient is not a normalised distribution.

    So unlike Poisson and the negative binomial, where only a non-integral
    ``w * y`` leaves the lattice, EVERY non-unit prior weight on a binomial
    response reports a sub- or super-probability.
    """
    off = np.abs(weights - 1.0) > _LATTICE_RELATIVE_TOLERANCE
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


def _check_counting_lattice(y: NDArray, weights: NDArray, family, weight_semantics: str) -> None:
    """Warn when a prior-weighted counting response is off its own lattice.

    The prior construction for Poisson and the negative binomial is
    ``w Y ~ Poisson(w mu)`` and ``w Y ~ NB2(w mu, w theta)``, both supported on
    the non-negative integers.  Where ``w * y`` is not integral the reported
    density evaluates ``gammaln`` at a fractional argument, which interpolates
    the counting density: the value is finite and smooth, but it is not a
    probability.  The log-likelihood, AIC and BIC are then a quasi-likelihood,
    and for the negative binomial the interpolated ``Gamma(w y + w theta) /
    Gamma(w theta)`` factor is theta-dependent, so it reaches ``theta_hat``
    and its profile interval too.

    This warns rather than raises deliberately.  The canonical case is
    on-lattice by construction -- ``y = count / exposure`` weighted by
    ``exposure`` -- and where it is not, the two contracts still share a score
    equation, so ``beta``, the fitted means and the deviance are unaffected.
    Refusing an otherwise valid fit over a defect confined to its reported
    likelihood would cost more than it protects; saying so plainly does not.
    """
    if weight_semantics != PRIOR_WEIGHTS:
        return
    if isinstance(family, Binomial):
        _warn_prior_weighted_binomial(weights)
        return
    if not isinstance(family, Poisson | NegativeBinomial):
        return
    scaled = weights * y
    slack = _LATTICE_RELATIVE_TOLERANCE * np.maximum(1.0, np.abs(scaled))
    off_lattice = np.abs(scaled - np.rint(scaled)) > slack
    count = int(np.count_nonzero(off_lattice))
    if count == 0:
        return
    import warnings

    name = type(family).__name__
    # At FIXED theta the two contracts share a score equation, so only the
    # reported likelihood moves. With ``theta="auto"`` that promise fails:
    # theta is profiled from the interpolated density and then enters V(mu),
    # the IRLS working weights and the unit deviance, so the coefficients
    # themselves move too. Say which case the reader is in.
    auto_theta = isinstance(family, NegativeBinomial) and getattr(family, "theta", None) in (
        "auto",
        None,
    )
    if isinstance(family, NegativeBinomial):
        reach = (
            " theta_hat is profiled from that interpolated density and then "
            "enters the variance and the IRLS weights, so the coefficients, "
            "fitted means and deviance move as well."
            if auto_theta
            else " theta_hat and its profile interval are affected as well; "
            "coefficients, fitted means and deviance are not, because at "
            "fixed theta the two contracts share a score equation."
        )
    else:
        reach = (
            " Coefficients, fitted means and deviance are unaffected -- the "
            "two weight contracts share a score equation."
        )
    warnings.warn(
        f"{count} of {len(scaled)} rows have a non-integral sample_weight * y "
        f'under weight_semantics="prior", which puts them off the {name} '
        "lattice, so the reported log-likelihood, AIC and BIC are a "
        "quasi-likelihood rather than an exact density, and randomized "
        "quantile residuals round those rows onto a neighbouring count."
        + reach
        + " The canonical weighting (y = count / exposure with "
        "sample_weight = exposure) is on-lattice; pass "
        'weight_semantics="frequency" if the weights are replication counts.',
        PriorWeightLatticeWarning,
        stacklevel=4,
    )


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

    Everything downstream keys on these values -- level tables, encoders,
    interaction keys -- so "does it hash" is the whole question, and asking it
    with `hash` itself is the only answer that cannot drift from what those
    callers will do. Column-wide dtype inference is not a substitute: pandas
    labels by `isinstance`, so a `string` column may still hold a `str`
    subclass that sets `__hash__ = None`, and a tuple of scalars may still
    nest an unhashable member.

    How the answer is collected is what makes it affordable. `deque(...,
    maxlen=0)` drives the map in C and discards each result, so a column costs
    one `hash` call per element and runs no Python bytecode at all, where the
    scan this replaced ran a Python-level predicate per row. A power search
    re-validates every configured column on every candidate fit, which is
    enough repetition for that difference to outweigh the fits it guards.
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
    if sample_weight is None:
        weight_arr = np.ones(n_rows, dtype=np.float64)
    elif isinstance(family, Tweedie) and weight_semantics == PRIOR_WEIGHTS:
        # Strict positivity is a Tweedie density requirement, not a statement
        # about the weight contract: the compound-Poisson normalizer carries
        # ``log w``, so a zero prior weight is an unevaluable density rather
        # than an uninformative row.  Read as a replication count the weight
        # never enters that normalizer, and zero means what it means for every
        # other family -- a row that appears no times.
        weight_arr = _validate_strict_prior_weights(sample_weight, n_rows)
    else:
        weight_arr = _finite_vector("sample_weight", sample_weight, n_rows)
    if np.any(weight_arr < 0.0):
        raise ValueError("sample_weight must be nonnegative")
    if not np.any(weight_arr > 0.0):
        raise ValueError("sample_weight must not be all zero")
    offset_arr = None if offset is None else _finite_vector("offset", offset, n_rows)
    validate_response(y_arr, family)
    _check_counting_lattice(y_arr, weight_arr, family, weight_semantics)
    return ValidatedFitInput(frame, y_arr, weight_arr, offset_arr)
