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
    if _has_exact_phi_method(result, "pearson"):
        return None, _CI_PEARSON_UNAVAILABLE
    if not _has_exact_phi_method(result, "mle"):
        return None, _CI_NOT_COMPUTED

    try:
        alpha_value = float(alpha)
    except (TypeError, ValueError, OverflowError):
        return None, _CI_NOT_COMPUTED
    cache = getattr(result, "_ci_cache", None)
    if type(cache) is not dict:
        return None, _CI_NOT_COMPUTED
    interval = dict.get(cache, alpha_value, _MISSING)
    if interval is _MISSING or not _is_finite_interval_tuple(interval):
        return None, _CI_NOT_COMPUTED
    return interval, _CI_AVAILABLE


def tweedie_profile_method_label(result: Any) -> str:
    """Describe the statistical and density methods without overstating them."""
    search_method = _search_method_label(getattr(result, "method", None))
    density_approximation = _uses_density_approximation(result)

    if _has_exact_phi_method(result, "mle"):
        qualifiers = [search_method]
        if density_approximation:
            qualifiers.append("density approximation")
        return f"Profile MLE ({'; '.join(qualifiers)})"
    if _has_exact_phi_method(result, "pearson"):
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
    interval_key = None if interval is None else _interval_report_identity(interval)
    return (
        id(result),
        _report_value_identity(getattr(result, "p_hat", _MISSING)),
        _report_value_identity(getattr(result, "phi_hat", _MISSING)),
        _report_value_identity(getattr(result, "nll", _MISSING)),
        _report_value_identity(getattr(result, "method", _MISSING)),
        _report_value_identity(getattr(result, "phi_method", _MISSING)),
        _report_value_identity(getattr(result, "density_exact", _MISSING)),
        status,
        interval_key,
    )


def _has_exact_phi_method(result: Any, expected: str) -> bool:
    value = getattr(result, "phi_method", None)
    return type(value) is str and value == expected


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
    if type(value) is not tuple or tuple.__len__(value) != 2:
        return False
    try:
        return all(math.isfinite(float(tuple.__getitem__(value, index))) for index in range(2))
    except (TypeError, ValueError, OverflowError):
        return False


def _interval_report_identity(interval: tuple[Any, Any]) -> tuple[Any, ...]:
    """Fingerprint an already validated exact tuple without subclass dispatch."""
    return (
        id(interval),
        float.hex(float(tuple.__getitem__(interval, 0))),
        float.hex(float(tuple.__getitem__(interval, 1))),
    )


def _report_value_identity(value: Any) -> tuple[Any, ...]:
    """Return a hashable identity without hashing or formatting arbitrary values."""
    value_type = type(value)
    if value is None:
        return ("none",)
    if value_type is bool:
        return ("bool", value)
    if value_type is int:
        return ("int", value)
    if value_type is float:
        return ("float", float.hex(value))
    if value_type is str:
        return ("str", value)
    return ("object", id(value_type), id(value))


def _uses_density_approximation(result: Any) -> bool:
    """Whether ANY story behind the estimate is approximation-based.

    Publication re-profiles dispersion and repoints the live density
    fields at itself, stashing the searched story as ``search_density_*``;
    ``p_hat`` was selected on the searched curve, so a saddlepoint-scored
    search must qualify the label even under an exact publication -- and
    a saddlepoint publication must qualify it even after an exact search.
    """
    for field in ("density_exact", "search_density_exact"):
        value = getattr(result, field, None)
        if value is None:
            continue
        try:
            if not bool(value):
                return True
        except (TypeError, ValueError):
            continue
    return False


__all__ = [
    "TweedieCIStatus",
    "cached_tweedie_profile_ci",
    "tweedie_profile_method_label",
    "tweedie_profile_report_identity",
]
