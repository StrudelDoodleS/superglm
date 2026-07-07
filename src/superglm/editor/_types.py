"""Shared editor data types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class EditableTerm:
    """Editable link-scale representation of one fitted 1D main effect."""

    name: str
    kind: str
    original_log_effect: NDArray
    edited_log_effect: NDArray
    x: NDArray | None = None
    levels: list[str] | None = None
    weights: NDArray | None = None
    ci_lower_log_effect: NDArray | None = None
    ci_upper_log_effect: NDArray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return int(self.edited_log_effect.size)

    @property
    def relativity(self) -> NDArray:
        return np.exp(self.edited_log_effect)

    def copy(self) -> EditableTerm:
        return EditableTerm(
            name=self.name,
            kind=self.kind,
            original_log_effect=self.original_log_effect.copy(),
            edited_log_effect=self.edited_log_effect.copy(),
            x=None if self.x is None else self.x.copy(),
            levels=None if self.levels is None else list(self.levels),
            weights=None if self.weights is None else self.weights.copy(),
            ci_lower_log_effect=(
                None if self.ci_lower_log_effect is None else self.ci_lower_log_effect.copy()
            ),
            ci_upper_log_effect=(
                None if self.ci_upper_log_effect is None else self.ci_upper_log_effect.copy()
            ),
            metadata=dict(self.metadata),
        )


@dataclass
class EditRecord:
    """One reversible edit to a term."""

    term: str
    operation: str
    indices: NDArray[np.intp]
    before: NDArray
    after: NDArray
    params: dict[str, Any] = field(default_factory=dict)
