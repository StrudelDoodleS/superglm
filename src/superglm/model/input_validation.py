"""Pure validation and normalization for public fit entry points."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from superglm import _count_lattice as _count_lattice_module
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
from superglm.solvers.dispersion import FREQUENCY_WEIGHTS, PRIOR_WEIGHTS

_LATTICE_MAXIMUM_SLACK = _count_lattice_module._LATTICE_MAXIMUM_SLACK
_LATTICE_ULP_SLACK = _count_lattice_module._LATTICE_ULP_SLACK
_is_exact_power_of_two = _count_lattice_module._is_exact_power_of_two
_not_a_whole_number = _count_lattice_module._not_a_whole_number
_off_integer_lattice = _count_lattice_module._off_integer_lattice
_product_was_exact = _count_lattice_module._product_was_exact


class PriorWeightLatticeWarning(UserWarning):
    """A discrete family's prior-weighted response left its own support."""


class FractionalFrequencyWeightWarning(UserWarning):
    """A replication count that is not a whole number of observations."""


#: Absolute slack when testing a weight for unity.  Values sit near 1 here, so
#: magnitude scaling is not in play and a plain tolerance is the right shape.
_UNIT_WEIGHT_TOLERANCE = 1e-9

#: What the caller is doing with theta, which decides how far an interpolated
#: density reaches.  Deriving this from the family alone is not enough: an
#: auto-theta fit replaces ``theta`` with a number once it finishes, so the
#: fitted family is indistinguishable from a fixed-theta one at evaluation --
#: and at evaluation nothing is refitted, so nothing but the likelihood moves.
THETA_FIXED = "fixed"  #: given constant, or an already-fitted model being scored
THETA_PROFILED = "profiled"  #: profiled with mu held fixed (a standalone interval)
THETA_ESTIMATED = "estimated"  #: profiled and fed back into IRLS (an auto fit)


def _interpolated_density_reach(family, *, theta_role: str) -> str:
    """How far an interpolated counting density reaches into the results.

    The same question wherever a counting family evaluates ``gammaln`` at a
    fractional argument, so it is answered in one place -- but the answer needs
    to know what the caller is doing, not only which family it holds.

    ``THETA_FIXED``
        theta is a given constant, or the model is already fitted and is only
        being scored.  The score equation is unchanged and nothing is refitted,
        so only the reported likelihood moves.  Claiming that ``theta_hat`` and
        its interval are affected here is false twice over: a fixed-theta model
        has no ``theta_hat``, and an evaluation does not produce one.
    ``THETA_PROFILED``
        theta is profiled against the interpolated density with ``mu`` held
        fixed, as in a standalone confidence interval.  The interval moves; the
        coefficients cannot, because nothing is refitted.
    ``THETA_ESTIMATED``
        theta is profiled and then re-enters ``V(mu)``, the IRLS working
        weights and the unit deviance, as in an auto-theta fit.  The estimates
        move as well.
    """
    if not isinstance(family, NegativeBinomial):
        return " Coefficients, fitted means and deviance are unaffected."
    if theta_role == THETA_FIXED:
        # Nothing is estimated here, so there is no theta_hat to be affected --
        # but the interpolated NB factor Gamma(wy + w theta) / Gamma(w theta)
        # is theta-dependent regardless, so any theta inference later drawn
        # from this likelihood inherits the interpolation. Say that without
        # asserting an estimate this call never produced.
        return (
            " Coefficients, fitted means and deviance are unaffected. The "
            "interpolated factor is theta-dependent, so any profile of theta "
            "taken from this likelihood moves with it."
        )
    if theta_role == THETA_ESTIMATED:
        return (
            " theta_hat is profiled from that interpolated density and then "
            "enters the variance and the IRLS weights, so the coefficients, "
            "fitted means and deviance move as well."
        )
    return (
        " The profiled theta and its interval are taken from that interpolated "
        "density and move with it; coefficients, fitted means and deviance are "
        "not, because mu is held fixed and nothing is refitted."
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

    Under ``"frequency"`` the likelihood is ``w log f(y; mu)``: the weight
    multiplies an ordinary per-row density rather than entering it, so it
    cannot rescue a response the family does not support.  Poisson and the
    negative binomial live on the non-negative integers, and a fractional
    ``y`` evaluates ``gammaln`` at a fractional argument -- finite, smooth, and
    not a probability.  The reported log-likelihood, AIC and BIC are then a
    quasi-likelihood, and randomized quantile residuals invert the CDF at a
    count that cannot occur.
    """
    # Carried rows only, as every sibling check in this module already does.
    # A zero replication count is a row that appears no times and contributes
    # exactly zero to the likelihood, so its response cannot put anything off
    # the lattice -- warning about it would reject an otherwise valid fit
    # under -W error over a row the fit ignores.
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
    # Carried rows only. A zero weight is admitted for every non-Tweedie
    # family, and the binomial branch of `prior_weight_log_density` returns
    # exactly 0.0 there -- so a fit whose every carried row is w == 1 has an
    # ordinary Bernoulli likelihood, and warning that it is "not an exact
    # density" would be false about that fit. Counting the zero rows would also
    # inflate the figure when the warning does fire legitimately.
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

    ``"frequency"`` says row ``i`` appears ``w_i`` times, so the likelihood is
    ``w_i log f(y_i; mu_i, phi)`` and the residual d.f. are ``sum(w) - edf``.
    Neither statement survives a fractional count: a row cannot appear 0.4
    times, and ``sum(w)`` stops being a number of observations. The arithmetic
    still evaluates, so the fit publishes likelihood criteria and REML results
    that look exact and are not.

    This warns rather than raises. The frequency contract reproduces the
    pre-``weight_semantics`` behaviour bit for bit, and that behaviour accepted
    fractional weights -- refusing them now would break code that worked before
    the contract was declared, to protect a criterion rather than an estimate.
    Under ``"prior"`` a fractional weight is entirely well defined, which is
    the whole point of the default, so this fires only on the other reading.
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
    if weight_semantics == FREQUENCY_WEIGHTS:
        # Replication does not move the support.  "This row appeared w times"
        # leaves each appearance an ordinary draw from the counting family, so
        # ``y`` itself must still be a whole count -- and unlike the prior arm
        # the weight never enters that question, so an integral or omitted
        # weight says nothing about it.  A fractional ``y`` here reaches the
        # same interpolated ``gammaln`` the prior arm warns about, and did so
        # with nothing to mark it because this function returned early for
        # every frequency model.
        if not isinstance(family, Poisson | NegativeBinomial):
            return
        _warn_frequency_counting_response(y, weights, family, theta_role)
        return
    if weight_semantics != PRIOR_WEIGHTS:
        return
    if isinstance(family, Binomial):
        _warn_prior_weighted_binomial(weights)
        return
    if not isinstance(family, Poisson | NegativeBinomial):
        return
    scaled = weights * y
    # The product tolerance is only earned where a product was actually formed.
    # IEEE-754 multiplication by an exact power of two -- and w == 1 above all,
    # which is what an unweighted fit passes -- is exact, so `scaled` is the
    # caller's response bit for bit and the supplied-value rule applies. Lending
    # those rows the round-off allowance let a directly supplied response like
    # 2**49 + 0.125 through, in the most common fit there is.
    exact_product = _product_was_exact(weights, y, scaled)
    count = int(
        np.count_nonzero(
            np.where(
                exact_product,
                _not_a_whole_number(scaled),
                _off_integer_lattice(scaled),
            )
        )
    )
    if count == 0:
        return
    import warnings

    name = type(family).__name__
    reach = _interpolated_density_reach(family, theta_role=theta_role or _theta_role_for(family))
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


def check_weight_contract(
    y: NDArray,
    weights: NDArray,
    family,
    weight_semantics: str,
    *,
    theta_role: str | None = None,
) -> None:
    """Both halves of the declared-contract check, behind one call.

    Each contract has its own way of being unhonourable, and each is silent
    about the other's: ``_check_counting_lattice`` returns immediately under
    ``"frequency"``, and ``_check_frequency_counts`` returns immediately under
    ``"prior"``. A caller that wires in only one therefore looks correct and
    covers half the models it sees -- which is exactly what happened when the
    evaluation boundaries were first given the lattice check alone, leaving
    every frequency-weighted evaluation unwarned.

    Call this at any point where a likelihood, an information criterion or a
    residual degrees-of-freedom is about to be computed from caller-supplied
    weights. It is the single seam; do not call the halves directly.
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
    # Both contract checks run only once every ordinary validation has passed.
    # They describe a likelihood that is about to be computed, so on input that
    # will never reach one they are noise -- and under ``-W error`` a warning
    # raised here would surface *instead of* the ValueError the caller needs to
    # see, reporting a quasi-likelihood for a negative Poisson response rather
    # than the negative response.
    check_weight_contract(y_arr, weight_arr, family, weight_semantics)
    return ValidatedFitInput(frame, y_arr, weight_arr, offset_arr)
