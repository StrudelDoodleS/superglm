"""The two weight contracts, and the likelihood sizes they imply.

``"prior"``
    ``w_i`` is the precision of row ``i``: ``Var(Y_i) = phi V(mu_i) / w_i``
    and the contribution is ``log f(y_i; mu_i, phi / w_i)`` (McCullagh &
    Nelder 1989, Sec. 2.2.2; statsmodels ``var_weights``, Stata ``aweight``).
    The default, as in R's ``glm`` and glum.
``"frequency"``
    Row ``i`` stands for ``w_i`` identical rows: ``w_i log f(y_i; mu_i, phi)``
    (statsmodels ``freq_weights``, Stata ``fweight``, SAS ``FREQ``).

Both give the same score equations and ``beta_hat``, but they coincide only at
``w == 1`` (integer weights do not make them agree), and they differ in the
likelihood's size, which this module returns. The size sets the dispersion,
residual degrees of freedom, Wald standard errors, the ``n`` of AIC/BIC and,
through REML, the smoothing parameters. Only the prior reading is a
likelihood for fractional ``w``.

Under ``"prior"`` a zero weight means infinite variance, so the size counts
the rows with positive weight, as R's ``df.residual`` does. Tweedie forbids
``w <= 0``. See ``docs/explanation/families-and-weights.md``.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from superglm.distributions import Distribution, Tweedie

PRIOR_WEIGHTS = "prior"
FREQUENCY_WEIGHTS = "frequency"
WEIGHT_SEMANTICS = (PRIOR_WEIGHTS, FREQUENCY_WEIGHTS)


def validate_weight_semantics(value: object) -> str:
    """Return a recognised weight contract, or raise naming both."""
    if value not in WEIGHT_SEMANTICS:
        raise ValueError(
            f"weight_semantics must be 'prior' or 'frequency', got {value!r}",
        )
    return str(value)


def family_default_weight_semantics(distribution: Distribution | None) -> str:
    """Return the contract a family carried before it became declarable.

    SuperGLM read Tweedie weights as EDM prior weights and every other
    family's as frequency weights, with no way to say otherwise.  This is
    retained for one purpose: a model or configuration captured before
    ``weight_semantics`` existed must restore the behaviour it was fitted
    under rather than adopt the new default.  It is reached only from the two
    migration paths -- ``model_weight_semantics`` when the attribute is absent,
    and ``ModelConfig.__setstate__`` -- and never from a model that carries a
    resolved contract of its own.
    """
    return PRIOR_WEIGHTS if isinstance(distribution, Tweedie) else FREQUENCY_WEIGHTS


def model_weight_semantics(model: object) -> str:
    """Return the contract a fitted or configured model resolved at build time.

    ``_distribution`` is only populated once a model has been fitted, so an
    UNFITTED legacy pickle would fall through the migration with ``None`` and
    be handed the non-Tweedie default -- silently flipping a pre-change Tweedie
    model, which always read prior weights, onto the replication reading the
    moment it was fitted. The configured family is consulted when there is no
    fitted one, so the migration answers on what the model actually is.
    """
    stored = getattr(model, "_weight_semantics", None)
    if stored is not None:
        return validate_weight_semantics(stored)
    distribution = getattr(model, "_distribution", None)
    if distribution is None:
        distribution = _configured_distribution(model)
    return family_default_weight_semantics(distribution)


def _configured_distribution(model: object) -> Distribution | None:
    """Best-effort family of an unfitted model, for the pickle migration only."""
    from superglm.distributions import resolve_distribution

    for source in ("_family_config", "_config", None):
        if source is None:
            family = getattr(model, "family", None)
        else:
            config = getattr(model, source, None)
            family = getattr(config, "family", None) if config is not None else None
        if isinstance(family, Distribution):
            return family
        if isinstance(family, str):
            try:
                return resolve_distribution(family)
            except ValueError:
                # Parameterised families cannot be named by string, so a string
                # here is never Tweedie and the non-Tweedie default is right.
                return None
    return None


def dispersion_likelihood_size(
    sample_weight: NDArray,
    *,
    weight_semantics: str,
) -> float:
    """Return the likelihood size the declared weight contract implies."""
    weights = np.asarray(sample_weight, dtype=np.float64)
    if validate_weight_semantics(weight_semantics) == PRIOR_WEIGHTS:
        return float(np.count_nonzero(weights > 0.0))
    return float(np.sum(weights, dtype=np.float64))


def lazy_likelihood_size(sample_weight: NDArray, *, weight_semantics: str) -> Callable[[], float]:
    """``dispersion_likelihood_size``, computed at the first call and reused after it.

    Every REML dispersion denominator and the scale term's
    ``0.5 * (n - M_p) * log(D)`` read the declared contract's size, never the
    physical row count, so the objective, gradient and Hessian stay
    consistent. Owner: the calling optimizer's frame; lifetime: one optimizer
    call. ``sample_weight`` is captured, so it must not change in place while
    the memo is alive.
    """
    return functools.cache(
        lambda: dispersion_likelihood_size(sample_weight, weight_semantics=weight_semantics)
    )


def pearson_residual_degrees_of_freedom(
    sample_weight: NDArray,
    effective_df: float,
    *,
    weight_semantics: str,
) -> float:
    """Return residual d.f. under the declared weight contract."""
    likelihood_size = dispersion_likelihood_size(
        sample_weight,
        weight_semantics=weight_semantics,
    )
    return max(likelihood_size - float(effective_df), 1.0)


# The two published scales (Wedderburn 1974; McCullagh & Nelder 1989, Sec. 8.3.6).
# ``phi`` is what standard errors, intervals and Wald tests read: Pearson's moment
# estimate, which needs only the mean-variance relation. ``phi_likelihood`` is what a
# density reads and what a variance component pairs with lambda: Wood's (2011) profile of
# the family's exact likelihood, the REML profile on the REML routes and ML at the fitted
# mean (zero nullity) elsewhere. mgcv publishes the same pair as ``sig2`` and ``reml.scale``.
PEARSON = "pearson"
KNOWN = "known"
LEGACY = "legacy"
REML_PROFILE = "reml_profile"
ML_AT_MEAN = "ml"
DEVIANCE_FORM = "deviance"
SCALE_METHOD_LABELS = {
    PEARSON: "Pearson",
    KNOWN: "known",
    LEGACY: "as saved",
    REML_PROFILE: "REML profile",
    ML_AT_MEAN: "ML",
    DEVIANCE_FORM: "deviance",
}


def profile_likelihood_scale(
    distribution: Distribution,
    y: NDArray,
    sample_weight: NDArray,
    *,
    weight_semantics: str,
    deviance: float,
    nullity: float = 0.0,
) -> float | None:
    """Wood's profile of the exact likelihood's phi; ML at the fitted mean at zero nullity.

    ``deviance`` is the (penalized) deviance at the fitted mean. None where the family has
    no profile or the profile has no interior optimum (a zero deviance, no residual size).
    """
    from superglm.distributions import Gamma, Gaussian
    from superglm.reml.scale import (
        gaussian_reml_scale_phi,
        prepare_gamma_reml_scale_data,
        prepare_tweedie_reml_scale_data,
        profile_gamma_reml_scale,
        profile_tweedie_reml_scale,
    )

    deviance, nullity = float(deviance), float(nullity)
    if not (math.isfinite(deviance) and deviance > 0.0):
        return None
    try:
        if isinstance(distribution, Gaussian):
            size = dispersion_likelihood_size(sample_weight, weight_semantics=weight_semantics)
            if size - nullity <= 0.0:
                return None
            return gaussian_reml_scale_phi(deviance, size, nullity)
        if isinstance(distribution, Gamma):
            data = prepare_gamma_reml_scale_data(
                y, sample_weight, weight_semantics=weight_semantics
            )
            if 2.0 * data.sum_weight <= nullity:
                return None
            return profile_gamma_reml_scale(data, deviance, nullity).phi
        if isinstance(distribution, Tweedie):
            data = prepare_tweedie_reml_scale_data(
                y, sample_weight, distribution.p, weight_semantics=weight_semantics
            )
            return profile_tweedie_reml_scale(data, deviance, nullity).phi
    except (ValueError, FloatingPointError, OverflowError):
        # Rows the exact likelihood cannot profile (no positive response, a weight the
        # contract forbids): the caller publishes its Pearson pair and says so.
        return None
    return None


def lifted_ml_scale(
    distribution: Distribution,
    y: NDArray,
    sample_weight: NDArray,
    mu: NDArray,
    *,
    weight_semantics: str,
    exponent: int,
) -> float | None:
    """ML at the fitted mean of a fit in a lifted weight unit, in that solver unit.

    The profile runs on the weights times ``2**exponent`` (exact), so a deviance or a
    scale that underflows in the caller's unit keeps its digits. A prior weight enters
    the density only through ``phi / w``, so that profile is the solver-unit scale; a
    replication count's ML does not depend on the weights' unit, so it is lifted after.
    """
    from superglm._utils import weighted_row_sum

    weights = np.ldexp(np.asarray(sample_weight, dtype=np.float64), exponent)
    phi = profile_likelihood_scale(
        distribution,
        y,
        weights,
        weight_semantics=weight_semantics,
        deviance=weighted_row_sum(weights, distribution.deviance_unit, y, mu),
    )
    if phi is None or validate_weight_semantics(weight_semantics) == PRIOR_WEIGHTS:
        return phi
    with np.errstate(over="ignore"):
        return float(np.ldexp(phi, exponent))


def _normal(value: float | None) -> bool:
    return value is not None and math.isfinite(value) and value >= np.finfo(np.float64).tiny


def scale_fields(
    result,
    *,
    distribution: Distribution,
    likelihood_phi: float | None,
    likelihood_method: str,
    solver_likelihood_phi: float | None = None,
) -> dict:
    """The field values that publish both scales on ``result``.

    ``result.phi`` (and ``solver_phi``) already hold Pearson's estimate at the fitting
    design's mean, as every solver publishes it, or 1 for a known scale. The likelihood
    scale is ``likelihood_phi`` from ``likelihood_method``, a physical value. On a fit in a
    lifted weight unit (``solver_phi`` set: weights below ``2**-512``) it is also carried
    in the solver unit, ``ldexp(likelihood_phi, weight_exponent)`` exactly, unless the
    caller profiled it there (``solver_likelihood_phi``) because the physical value
    underflows. The Pearson pair stands in, and the method says so, where neither is a
    normal positive float.
    """
    if getattr(distribution, "scale_known", True):
        # Standard errors of a known-scale family read 1 whatever ``phi`` holds.
        return {
            "phi_likelihood": 1.0,
            "solver_phi_likelihood": None,
            "phi_method": KNOWN,
            "phi_likelihood_method": KNOWN,
        }
    solver_phi = getattr(result, "solver_phi", None)
    value = None if likelihood_phi is None else float(likelihood_phi)
    pearson_pair = {
        "phi_likelihood": float(result.phi),
        "solver_phi_likelihood": solver_phi,
        "phi_method": PEARSON,
        "phi_likelihood_method": PEARSON,
    }
    if solver_phi is not None:
        exponent = int(result.weight_exponent)
        solver_value = solver_likelihood_phi
        if solver_value is None and _normal(value):
            try:
                solver_value = math.ldexp(value, exponent)
            except OverflowError:
                solver_value = None
        if not _normal(None if solver_value is None else float(solver_value)):
            return pearson_pair
        return {
            "phi_likelihood": math.ldexp(float(solver_value), -exponent),
            "solver_phi_likelihood": float(solver_value),
            "phi_method": PEARSON,
            "phi_likelihood_method": likelihood_method,
        }
    if not _normal(value):
        return pearson_pair
    return {
        "phi_likelihood": value,
        "solver_phi_likelihood": None,
        "phi_method": PEARSON,
        "phi_likelihood_method": likelihood_method,
    }


def publish_scales(
    result,
    *,
    distribution: Distribution,
    likelihood_phi: float | None,
    likelihood_method: str,
    solver_likelihood_phi: float | None = None,
):
    """Return ``result`` with both published scales set (``scale_fields``)."""
    from dataclasses import replace

    return replace(
        result,
        **scale_fields(
            result,
            distribution=distribution,
            likelihood_phi=likelihood_phi,
            likelihood_method=likelihood_method,
            solver_likelihood_phi=solver_likelihood_phi,
        ),
    )


def scale_method(result) -> str:
    """The estimator behind ``result.phi``; ``"legacy"`` for a result saved before the split."""
    return getattr(result, "phi_method", None) or LEGACY


def likelihood_scale_method(result) -> str:
    """The estimator behind the likelihood scale; ``"legacy"`` before the split."""
    return getattr(result, "phi_likelihood_method", None) or LEGACY
