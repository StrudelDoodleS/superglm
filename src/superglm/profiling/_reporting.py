"""Profile intervals as summaries and editor payloads report them."""

from __future__ import annotations

from typing import Any, Literal

ProfileCIStatus = Literal[
    "available", "censored", "caution", "censored with caution", "not computed"
]


def reported_interval(
    interval: Any, *, caution: bool = False
) -> tuple[tuple[float, float], ProfileCIStatus]:
    """``(lower, upper)`` and its status: "censored" when either side is where its
    search stopped, "caution" when the interval carries a caution
    (``result.warnings`` says which), "censored with caution" for both, else
    "available".

    A censored endpoint is not a likelihood-ratio crossing, so the interval
    may extend beyond it.
    """
    censored = interval.lower_censored or interval.upper_censored
    if censored:
        status = "censored with caution" if caution else "censored"
    else:
        status = "caution" if caution else "available"
    return (interval.lower, interval.upper), status


def profile_cautioned(result: Any, alpha: float) -> bool:
    """Whether the interval at ``alpha`` carries a caution: the estimate's own, or
    one its computation found."""
    cautions = getattr(result, "_ci_cautions", {}).get(float(alpha))
    return getattr(result, "_caution", None) is not None or bool(cautions)


def cached_tweedie_profile_ci(
    result: Any, alpha: float = 0.05
) -> tuple[tuple[float, float] | None, ProfileCIStatus]:
    """The interval already computed at ``alpha``; reporting never evaluates the profile.

    Each point of the p profile is a model refit, so an interval nobody asked
    for is "not computed".
    """
    interval = result._ci_cache.get(float(alpha))
    if interval is None:
        return None, "not computed"
    return reported_interval(interval, caution=profile_cautioned(result, alpha))


def tweedie_profile_report_identity(result: Any, alpha: float) -> tuple[Any, ...]:
    """What a cached summary shows of the profile; an interval computed later refreshes it."""
    interval, status = cached_tweedie_profile_ci(result, alpha)
    return (id(result), result.p_hat, result.phi_hat, result.nll, status, interval)


__all__ = [
    "ProfileCIStatus",
    "cached_tweedie_profile_ci",
    "profile_cautioned",
    "reported_interval",
    "tweedie_profile_report_identity",
]
