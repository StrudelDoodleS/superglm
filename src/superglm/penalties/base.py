"""Penalty and Flavor protocols."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, runtime_checkable

from numpy.typing import NDArray

from superglm.types import GroupSlice


def normalize_penalty_features(features: str | Iterable[str] | None) -> frozenset[str] | None:
    """Normalize optional penalty feature filters.

    Parameters
    ----------
    features
        ``None`` for all groups, a single feature/group name, or an iterable of
        names. Exact feature names (``group.feature_name``) and exact group names
        (``group.name``) are both supported.
    """
    if features is None:
        return None
    if isinstance(features, str):
        tokens = [features]
    else:
        tokens = list(features)
    cleaned = [str(token).strip() for token in tokens if str(token).strip()]
    return frozenset(cleaned) if cleaned else None


def penalty_targets_group(penalty: object, group: GroupSlice) -> bool:
    """Whether the lambda1-style penalty should apply to this group."""
    if not group.penalized:
        return False
    features = getattr(penalty, "features", None)
    if features is None:
        return True
    return group.feature_name in features or group.name in features


def validate_penalty_features(
    penalty: object,
    groups: list[GroupSlice],
) -> None:
    """Raise on unknown penalty feature/group filters after DM construction."""
    features = getattr(penalty, "features", None)
    if features is None:
        return
    available = {g.feature_name for g in groups} | {g.name for g in groups}
    missing = sorted(features - available)
    if missing:
        available_names = sorted(available)
        raise ValueError(
            "Unknown penalty feature/group filter(s): "
            f"{missing}. Available names: {available_names}"
        )


def penalty_has_targets(penalty: object, groups: list[GroupSlice]) -> bool:
    """Return True when the penalty applies to at least one fitted group."""
    return any(penalty_targets_group(penalty, g) for g in groups)


@runtime_checkable
class Penalty(Protocol):
    """Protocol for penalty functions.

    The solver calls prox() in the inner loop and eval() for diagnostics.
    Penalties are passed as objects so different penalties can be swapped
    without changing the solver.
    """

    @property
    def lambda1(self) -> float: ...

    @property
    def flavor(self) -> Flavor | None: ...

    @property
    def features(self) -> frozenset[str] | None: ...

    def prox(self, beta: NDArray, groups: list[GroupSlice], step: float) -> NDArray:
        """Proximal operator: argmin_z  step * P(z) + 0.5 * ||beta - z||^2."""
        ...

    def prox_group(self, bg: NDArray, group: GroupSlice, step: float) -> NDArray:
        """Proximal operator for a single group's coefficients.

        Parameters
        ----------
        bg : NDArray of shape (p_g,)
            Candidate coefficients for this group (already gradient-stepped).
        group : GroupSlice
            Group metadata (weight, size, etc.).
        step : float
            Step size (1 / L_g for this group).

        Returns
        -------
        NDArray of shape (p_g,)
        """
        ...

    def eval(self, beta: NDArray, groups: list[GroupSlice]) -> float:
        """Evaluate penalty value P(beta)."""
        ...


@runtime_checkable
class Flavor(Protocol):
    """Protocol for penalty modifiers (e.g. adaptive weighting).

    Flavors adjust group weights based on an initial estimate before
    the solver runs its main fit.
    """

    def adjust_weights(
        self,
        groups: list[GroupSlice],
        beta_init: NDArray,
        group_matrices: list | None = None,
    ) -> list[GroupSlice]:
        """Return new GroupSlice list with modified weights.

        Parameters
        ----------
        groups : list of GroupSlice
        beta_init : (p,) coefficient vector from initial fit.
        group_matrices : list of GroupMatrix, optional
            Per-group design matrices. When provided, implementations can
            compute fitted-value norms ``||X_g beta_g||`` instead of raw
            coefficient norms, giving scale-invariant adaptive weights.
        """
        ...
