"""Read-only Tweedie profile state for summaries and editor payloads."""

from __future__ import annotations

import math
from typing import Any, Literal

TweedieCIStatus = Literal[
    "available",
    "not computed",
    "unavailable for Pearson plug-in",
]

_CI_AVAILABLE: TweedieCIStatus = "available"
_CI_NOT_COMPUTED: TweedieCIStatus = "not computed"
_CI_PEARSON_UNAVAILABLE: TweedieCIStatus = "unavailable for Pearson plug-in"
_MISSING = object()


def cached_tweedie_profile_ci(
    result: Any,
    alpha: float = 0.05,
) -> tuple[tuple[Any, Any] | None, TweedieCIStatus]:
    """Return one exact cached MLE interval without evaluating the profile.

    Pearson profiles cannot support likelihood-ratio intervals, so even a
    legacy or manually populated tuple is deliberately ignored. Missing
    reporting attributes on legacy/fake results are treated as uncached.
    """
    phi_method = _normalise_phi_method(getattr(result, "phi_method", None))
    if phi_method == "pearson":
        return None, _CI_PEARSON_UNAVAILABLE
    if phi_method != "mle":
        return None, _CI_NOT_COMPUTED

    try:
        alpha_value = float(alpha)
    except (TypeError, ValueError, OverflowError):
        return None, _CI_NOT_COMPUTED
    cache = getattr(result, "_ci_cache", None)
    if not isinstance(cache, dict):
        return None, _CI_NOT_COMPUTED
    interval = cache.get(alpha_value, _MISSING)
    if interval is _MISSING or not _is_finite_interval_tuple(interval):
        return None, _CI_NOT_COMPUTED
    return interval, _CI_AVAILABLE


def tweedie_profile_method_label(result: Any) -> str:
    """Describe the statistical and density methods without overstating them."""
    search_method = _search_method_label(getattr(result, "method", None))
    phi_method = _normalise_phi_method(getattr(result, "phi_method", None))
    density_approximation = _uses_density_approximation(result)

    if phi_method == "mle":
        qualifiers = [search_method]
        if density_approximation:
            qualifiers.append("density approximation")
        return f"Profile MLE ({'; '.join(qualifiers)})"
    if phi_method == "pearson":
        qualifiers = [search_method, "Pearson plug-in"]
        if density_approximation:
            qualifiers.append("density approximation")
        return f"Approximate profile ({'; '.join(qualifiers)})"

    qualifiers = [search_method]
    if density_approximation:
        qualifiers.append("density approximation")
    return f"Profile ({'; '.join(qualifiers)})"


def tweedie_profile_report_identity(result: Any, alpha: float) -> tuple[Any, ...]:
    """Return hashable state that invalidates summaries after explicit CI work."""
    interval, status = cached_tweedie_profile_ci(result, alpha)
    interval_key = None if interval is None else (float(interval[0]), float(interval[1]))
    return (
        id(result),
        str(getattr(result, "method", "")),
        _normalise_phi_method(getattr(result, "phi_method", None)),
        _density_identity(result),
        status,
        interval_key,
    )


def _normalise_phi_method(value: Any) -> str:
    return str(value or "").strip().lower()


def _search_method_label(value: Any) -> str:
    method = str(value or "unknown").strip().lower()
    labels = {
        "brent": "Brent",
        "brentq": "BrentQ",
        "grid": "Grid",
        "grid_refine": "Grid Refine",
        "lbfgsb": "L-BFGS-B",
        "powell": "Powell",
    }
    return labels.get(method, method.replace("_", " ").title())


def _is_finite_interval_tuple(value: Any) -> bool:
    if not isinstance(value, tuple) or len(value) != 2:
        return False
    try:
        return all(math.isfinite(float(endpoint)) for endpoint in value)
    except (TypeError, ValueError, OverflowError):
        return False


def _uses_density_approximation(result: Any) -> bool:
    density_exact = getattr(result, "density_exact", None)
    if density_exact is None:
        return False
    try:
        return not bool(density_exact)
    except (TypeError, ValueError):
        return False


def _density_identity(result: Any) -> bool | None:
    density_exact = getattr(result, "density_exact", None)
    if density_exact is None:
        return None
    try:
        return bool(density_exact)
    except (TypeError, ValueError):
        return None


__all__ = [
    "TweedieCIStatus",
    "cached_tweedie_profile_ci",
    "tweedie_profile_method_label",
    "tweedie_profile_report_identity",
]
