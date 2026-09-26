"""Read-only Tweedie profile state for summaries and editor payloads."""

from __future__ import annotations

from typing import Any, Literal

TweedieCIStatus = Literal["available", "censored", "not computed"]


def cached_tweedie_profile_ci(
    result: Any, alpha: float = 0.05
) -> tuple[tuple[float, float] | None, TweedieCIStatus]:
    """The interval already computed at ``alpha``; reporting never evaluates the profile.

    A side that stopped at a search bound or next to an infeasible power makes
    the interval "censored": that endpoint is where the search stopped, not a
    likelihood-ratio crossing.
    """
    interval = result._ci_cache.get(float(alpha))
    if interval is None:
        return None, "not computed"
    censored = interval.lower_censored or interval.upper_censored
    return (interval.lower, interval.upper), "censored" if censored else "available"


def tweedie_profile_report_identity(result: Any, alpha: float) -> tuple[Any, ...]:
    """What a cached summary shows of the profile; an interval computed later refreshes it."""
    interval, status = cached_tweedie_profile_ci(result, alpha)
    return (id(result), result.p_hat, result.phi_hat, result.nll, status, interval)


__all__ = [
    "TweedieCIStatus",
    "cached_tweedie_profile_ci",
    "tweedie_profile_report_identity",
]
