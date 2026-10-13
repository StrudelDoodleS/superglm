"""Advance notice for exact REML fits that will take minutes.

On the exact path (``discrete=False``) every coefficient update, every
line-search refit and the one-pass weight-derivative correction (Wood 2011,
section 3.5.1) each form a ``p x p`` weighted cross-product over all ``n``
rows, so the fit costs about ``n p^2`` per such product, and a smoothing search
needs dozens of them. ``discrete=True`` forms those products from the binned
values instead and chooses the smoothing on the working model
(performance-oriented iteration, Wood 2011 section 6.2, as mgcv's ``bam``
does).
"""

from __future__ import annotations

from typing import Any

from superglm.distributions import Gaussian
from superglm.links import IdentityLink


class SlowFitWarning(UserWarning):
    """An exact REML fit is large enough that it will take minutes.

    ``SuperGLM.fit_reml`` warns before the smoothing search starts when the
    model is fitted with ``discrete=False`` and rows x coefficients squared
    reaches 1e11, for a family whose working weights change with the fit
    (every family except Gaussian with the identity link, whose weighted
    cross-product is formed once and reused). A fit on the structured solver
    for a random effect or factor smooth does not warn: it never forms the
    full cross-product. The fit still runs and its result is unaffected. Pass ``discrete=True`` to the constructor for the
    fast path, or silence the warning with
    ``warnings.filterwarnings("ignore", category=SlowFitWarning)``.
    """


# rows x coefficients^2: the multiply-adds of one dense weighted cross-product.
# At 1e11 one product costs seconds on one thread and a search forms dozens.
# On the 542,410-row freMTPL2 book the exact fit took 7-12 s at 2.4e9 (main
# effects, 67 coefficients), 27 s at 2.0e10 (one interaction pair, 192), 95 s
# at 1.2e11 (three pairs, 478) and 405 s at 1.3e12 (ten pairs, 1,565);
# block-sparse designs such as these cost less than the dense count, dense ones
# more, so at the threshold a fit takes a minute or more.
EXACT_REML_WORK_WARN = 1.0e11


def exact_reml_cost_message(
    *,
    n_rows: int,
    n_coef: int,
    distribution: Any,
    link: Any,
) -> str | None:
    """Return the ``SlowFitWarning`` text for an exact REML fit, or ``None``.

    The caller has established that the fit is on the exact path and that
    its smoothing is being estimated; this decides whether it is slow.
    """
    if type(distribution) is Gaussian and type(link) is IdentityLink:
        return None  # constant working weights: the cross-product is reused
    if float(n_rows) * n_coef * n_coef < EXACT_REML_WORK_WARN:
        return None
    return (
        f"fit_reml is fitting {n_rows:,} rows exactly (discrete=False) with "
        f"{n_coef} coefficients. Each coefficient update forms a {n_coef} x "
        f"{n_coef} weighted cross-product over every row, and choosing the "
        "smoothing takes dozens of them, so this fit may take minutes. On a "
        "542,000-row claim-frequency book, about 480 coefficients took 1.6 "
        "minutes and about 1,560 took 7 minutes; SuperGLM(discrete=True) took "
        "19 s and 90 s. It groups each spline's values into n_bins equal-width "
        "bins, which changes nothing for a column with no more distinct values "
        "than n_bins, and it chooses the smoothing on the working model, as "
        "mgcv's bam does. Its fitted values there differed from the exact fit's "
        "by at most 1.3% on 99% of rows."
    )


__all__ = ["EXACT_REML_WORK_WARN", "SlowFitWarning", "exact_reml_cost_message"]
