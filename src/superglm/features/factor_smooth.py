"""Fully penalized smooth deviations by factor level."""

from __future__ import annotations

from typing import Any, Literal

from superglm.types import LambdaPolicy


class FactorSmooth:
    """An mgcv-style ``bs="fs"`` P-spline interaction.

    The fitted representation is populated by the design-matrix builder.  One
    shared marginal basis is repeated implicitly across every fitted factor
    level, with no reference level and no centering side condition.
    """

    structured_kind = "factor_smooth"
    requires_reml = True

    def __init__(
        self,
        variable: str,
        *,
        group: str,
        kind: str = "ps",
        k: int = 6,
        m: int = 2,
        unseen: Literal["population", "error"] = "population",
        missing: Literal["error"] = "error",
        lambda_policy: LambdaPolicy | dict[str, LambdaPolicy] | None = None,
        name: str | None = None,
    ):
        if not isinstance(variable, str) or not variable:
            raise ValueError("variable must be a non-empty column name")
        if not isinstance(group, str) or not group:
            raise ValueError("group must be a non-empty column name")
        if variable == group:
            raise ValueError("variable and group must name different columns")
        if kind != "ps":
            raise NotImplementedError("FactorSmooth currently supports only kind='ps'.")
        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError("k must be an integer")
        if k < 5:
            raise ValueError("k must be at least 5 for a cubic P-spline basis")
        if isinstance(m, bool) or not isinstance(m, int):
            raise TypeError("m must be an integer")
        if not 1 <= m < k:
            raise ValueError(f"m must be between 1 and k - 1, got m={m}, k={k}")
        if unseen not in ("population", "error"):
            raise ValueError(f"unseen must be 'population' or 'error', got {unseen!r}")
        if missing != "error":
            raise ValueError(f"missing must be 'error', got {missing!r}")
        if name is not None and (not isinstance(name, str) or not name):
            raise ValueError("name must be a non-empty string when supplied")

        valid_components = {"wiggle", *(f"null_{index}" for index in range(m))}
        if isinstance(lambda_policy, dict):
            unknown = set(lambda_policy) - valid_components
            if unknown:
                raise ValueError(
                    "lambda_policy contains unknown component names "
                    f"{sorted(unknown)!r}; valid names are {sorted(valid_components)!r}"
                )
            invalid = {
                component
                for component, policy in lambda_policy.items()
                if not isinstance(policy, LambdaPolicy)
            }
            if invalid:
                raise TypeError(
                    "lambda_policy values must be LambdaPolicy instances; "
                    f"invalid components: {sorted(invalid)!r}"
                )
        elif lambda_policy is not None and not isinstance(lambda_policy, LambdaPolicy):
            raise TypeError("lambda_policy must be a LambdaPolicy, a component mapping, or None")

        self.variable = variable
        self.group = group
        self.kind = kind
        self.k = k
        self.m = m
        self.unseen = unseen
        self.missing = missing
        self._lambda_policy = lambda_policy
        self.name = name or f"{variable}:{group}:fs"

        self._levels: list[Any] = []
        self._level_to_code: dict[Any, int] = {}
        self._spline = None
        self._natural_map = None
        self._base_penalty_components: tuple[tuple[str, Any], ...] = ()

    @property
    def parent_names(self) -> tuple[str, str]:
        """The numeric marginal and grouping columns read by this interaction."""
        return (self.variable, self.group)


__all__ = ["FactorSmooth"]
