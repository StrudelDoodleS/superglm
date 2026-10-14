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
    full cross-product. Nor does a model with a monotone (shape-constrained)
    term: its smoothing is chosen by a different engine, whose cost has not
    been measured. The fit still runs and its result is unaffected. Pass ``discrete=True`` to the constructor for the
    fast path, or silence the warning with
    ``warnings.filterwarnings("ignore", category=SlowFitWarning)``.
    """


# rows x coefficients^2: the multiply-adds of one dense weighted cross-product.
# At 1e11 one product costs seconds on one thread and a search forms dozens;
# block-sparse designs cost less than the dense count, dense ones more, so at
# the threshold a fit takes a minute or more.
#
# Evidence: the 542,410-row freMTPL2 frequency book (Poisson, exposure
# weights), one session on the PR #487 head (f71dc03b), one thread each on a
# shared machine; wall times of the same fit varied up to 1.5x between
# sessions. "Coefficients" is the design width ``dm.p``, intercept excluded,
# after the spline-by-factor separation hold (before it: 197, 483 and 1,565).
#
#   model              coefs  n p^2    exact   discrete=True
#   main effects          67  2.4e9    8-9 s   3 s
#   + 1 pair             192  2.0e10   40 s    9 s   (27 s in another session)
#   + 3 pairs            478  1.2e11   86 s    20 s
#   + 10 pairs         1,540  1.3e12   422 s   84 s
#
# Gamma with the log link reuses its constant Fisher data Gram across the
# coefficient fits but is not exempt: the observed-curvature geometry, the
# weight-derivative pass and the line search still cross every row, and the
# same designs with a Gamma response took 119 s (3 pairs, 21 s binned) and
# 554 s (10 pairs).
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
        "542,000-row claim-frequency book, about 480 coefficients took 1.4 "
        "minutes and about 1,540 took 7 minutes; SuperGLM(discrete=True) took "
        "20 s and 84 s. It groups each spline's values into n_bins equal-width "
        "bins, which changes nothing for a column with no more distinct values "
        "than n_bins, and it chooses the smoothing on the working model, as "
        "mgcv's bam does. Its fitted values there differed from the exact fit's "
        "by at most 1.1% on 99% of rows."
    )


__all__ = ["EXACT_REML_WORK_WARN", "SlowFitWarning", "exact_reml_cost_message"]
