"""Advance notice for exact REML fits whose smoothing search will be slow.

On the exact path (``discrete=False``) each Newton step of the smoothing
search differentiates the working weights with respect to every smoothing
parameter (Wood 2011, section 3.4), and each derivative is a weighted Gram
over every row. A step therefore costs about rows x smoothing parameters x
coefficients, and that term dominates large fits (measured figures are in the
message below). ``discrete=True`` chooses the smoothing on the working model
instead (performance-oriented iteration, Wood 2011 section 6.2, as mgcv's
``bam`` does), which needs no weight derivatives, and bins the splines'
values.
"""

from __future__ import annotations

from typing import Any

from superglm.distributions import Binomial, Gaussian
from superglm.links import IdentityLink, LogitLink


class SlowFitWarning(UserWarning):
    """An exact REML fit is large enough that choosing its smoothing will be slow.

    ``SuperGLM.fit_reml`` warns before the smoothing search starts when the
    model is fitted with ``discrete=False`` and rows x smoothing parameters x
    coefficients reaches 1e9, for a family whose working weights change with
    the fit: every family except Gaussian with the identity link, and except
    binomial with the logit link once it has four or more smoothing
    parameters, whose search takes a cheaper route. The fit still runs and
    its result is unaffected. Pass ``discrete=True`` to the constructor for
    the fast path, or silence the warning with
    ``warnings.filterwarnings("ignore", category=SlowFitWarning)``.
    """


# rows x smoothing parameters x coefficients at which the weight-derivative
# Grams cost about 9 s per smoothing-search step on one thread: measured at
# 8.7e-9 s per row x coefficient per Gram on the freMTPL2 book.
EXACT_REML_WORK_WARN = 1.0e9

# Binomial-logit searches with this many smoothing parameters replace the
# per-parameter Grams with one leverage pass (``reml.direct``).
_LEVERAGE_ROUTE_MIN_PENALTIES = 4


def exact_reml_cost_message(
    *,
    n_rows: int,
    n_smoothing: int,
    n_coef: int,
    distribution: Any,
    link: Any,
) -> str | None:
    """Return the ``SlowFitWarning`` text for an exact REML fit, or ``None``.

    The caller has established that the fit is on the exact path and that
    its smoothing is being estimated; this decides whether it is slow.
    """
    if type(distribution) is Gaussian and type(link) is IdentityLink:
        return None  # constant working weights: no weight derivatives
    if (
        type(distribution) is Binomial
        and type(link) is LogitLink
        and n_smoothing >= _LEVERAGE_ROUTE_MIN_PENALTIES
    ):
        return None
    if n_rows * n_smoothing * n_coef < EXACT_REML_WORK_WARN:
        return None
    return (
        f"fit_reml is fitting {n_rows:,} rows exactly (discrete=False) with "
        f"{n_smoothing} smoothing parameters and {n_coef} coefficients. Each step "
        "of the smoothing search reads every row once per smoothing parameter, so "
        "this fit may take many minutes. On a 542,000-row claim-frequency book, 13 "
        "smoothing parameters took 2.5 minutes, 17 took 15 minutes and 102 did not "
        "finish in 30 minutes; SuperGLM(discrete=True) took 8 s, 16 s and 84 s. "
        "It groups each spline's values into n_bins equal-width bins, which changes "
        "nothing for a column with no more distinct values than n_bins, and it "
        "chooses the smoothing on the working model, as mgcv's bam does. Its fitted "
        "values there differed from the exact fit's by under 1% on 99% of rows."
    )


__all__ = ["EXACT_REML_WORK_WARN", "SlowFitWarning", "exact_reml_cost_message"]
