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
