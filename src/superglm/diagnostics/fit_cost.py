"""Advance notice for exact REML fits that will be slow.

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
    """An exact REML fit is large enough to be slow: most of a minute or more.

    ``SuperGLM.fit_reml`` warns before the smoothing search starts when the model is fitted with
    ``discrete=False`` and rows x coefficients squared reaches 1e11, for a family whose working
    weights change with the fit (every family except Gaussian with the identity link). It does
    not warn for a structured-solver fit with a random effect or factor smooth (it never forms
    the full cross-product) or a model with a monotone term (different engine). The fit still
    runs and its result is unaffected. Pass ``discrete=True`` to the constructor for the fast
    path, or silence with ``warnings.filterwarnings("ignore", category=SlowFitWarning)``.
    """


# rows x coefficients^2: the multiply-adds of one dense weighted cross-product. At 1e11 one
# product costs seconds on one thread and a search forms dozens; block-sparse designs cost
# less than the dense count, dense ones more. Exact wall time grows about as (n p^2)^0.64,
# which puts a minute near 1.8e11 (measured on the 542,410-row freMTPL2 frequency book,
# one pinned thread: 48 s at 1.2e11, 214 s at 1.3e12). Binning with `discrete=True` is
# faster but changes fitted values (99th percentile 0.46-0.99% on the training rows).
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
        "542,000-row claim-frequency book, about 480 coefficients took 48 s "
        "and about 1,540 took 3.6 minutes; SuperGLM(discrete=True) took 17 s "
        "and 76 s. It groups each spline's values into n_bins equal-width "
        "bins, which changes nothing for a column with no more distinct values "
        "than n_bins, and it chooses the smoothing on the working model, as "
        "mgcv's bam does. Its fitted values there were within 0.5% and 1.0% of "
        "the exact fit's on 99% of rows, though up to 28% and 40% apart on the "
        "single worst row."
    )


__all__ = ["EXACT_REML_WORK_WARN", "SlowFitWarning", "exact_reml_cost_message"]
