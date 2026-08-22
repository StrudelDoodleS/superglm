"""The two weight contracts, and the likelihood sizes they imply.

An observation weight can enter an exponential-dispersion likelihood in two
ways, and they are different models rather than different scalings of one
model:

``"prior"``
    ``w_i`` states the *precision* of row ``i``: ``Var(Y_i) = phi V(mu_i) / w_i``
    and the contribution is ``log f(y_i; mu_i, phi / w_i)``.  This is what you
    have when the response is an average -- an aggregated ``incurred /
    exposure`` with the exposure as weight, or an average severity with the
    claim count as weight.  McCullagh & Nelder (*Generalized Linear Models*,
    2nd ed., 1989, Sec. 2.2.2) call these prior weights; statsmodels and Stata
    call the same thing variance or analytic weights.
``"frequency"``
    ``w_i`` states that row ``i`` stands in for ``w_i`` identical rows, so the
    contribution is ``w_i log f(y_i; mu_i, phi)``.  The data are a compression
    of a longer table.

The two coincide only at integer ``w``, where "measured ``w`` times as
precisely" and "appears ``w`` times" happen to agree.  At fractional ``w`` only
the prior reading is a likelihood at all: a row cannot be replicated 0.4 times,
and the frequency reading's ``sum(w) - edf`` residual d.f. stops counting
anything.

Both contracts give the *same* score equations and therefore the same
``beta_hat``; they part company over the likelihood's size, which is what this
module returns.  Everything downstream of the size moves with it: the Pearson
dispersion, residual degrees of freedom, Wald standard errors and intervals,
the effective ``n`` in AIC/BIC, and -- through the REML criterion -- the
smoothing parameters and the fitted surface.

Established practice, and where SuperGLM sits in it
---------------------------------------------------

Every major system either offers both contracts as separate declared inputs or
defaults to the prior reading:

=================  ====================================  ======================
system             prior / variance / analytic            frequency / case
=================  ====================================  ======================
R ``glm``          ``weights=`` (the only weight input)   replicate the rows
statsmodels GLM    ``var_weights=``                       ``freq_weights=``
Stata              ``aweight``                            ``fweight``
SAS ``GENMOD``     ``WEIGHT`` (fractional allowed)        ``FREQ`` (integers)
glum               ``sample_weight=``                     replicate the rows
=================  ====================================  ======================

R documents ``weights`` as "prior weights", with the values "inversely
proportional to the dispersions"; SAS is explicit that ``FREQ`` changes the
sample size used in downstream formulas and ``WEIGHT`` does not, which is
exactly the distinction below.  SuperGLM's default is ``"prior"`` for the same
reason R's and glum's is: exposure is continuous, and in the setting this
library targets the aggregated response is the ordinary case.

Zero weights
------------

Under ``"frequency"`` a zero-weight row drops out of ``sum(w)`` on its own.
Under ``"prior"`` it has to be counted out deliberately, because ``w = 0`` says
the row carries no information at all -- infinite variance -- rather than that
it was observed once.  The prior size is therefore the number of rows carrying
positive weight, which is what R reports: a twelve-row Gamma fit with four zero
weights and rank 2 returns ``df.residual = 6``, not ``10``, alongside the
explicit note that "observations with zero weight not used for calculating
dispersion".  Tweedie forbids ``w <= 0`` outright (its compound-Poisson
normalizer carries ``log w``), so the count and the row total agree there and
this rule changes nothing for it.
"""

from __future__ import annotations

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
