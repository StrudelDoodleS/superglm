"""Display-order helpers for categorical editor terms."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


def level_order_for_target(
    size: int,
    selected_indices: NDArray[np.intp],
    target_index: int,
) -> list[int]:
    """Return an order that moves selected levels as one block to a target slot."""
    selected = [int(i) for i in np.unique(selected_indices)]
    selected_set = set(selected)
    remaining = [i for i in range(size) if i not in selected_set]
    target = max(0, min(size, int(target_index)))
    insert_at = sum(1 for i in remaining if i < target)
    return remaining[:insert_at] + selected + remaining[insert_at:]


def level_order_for_direction(
    size: int,
    selected_indices: NDArray[np.intp],
    direction: str,
) -> list[int]:
    """Return an order that nudges selected levels one slot left or right."""
    if direction not in {"left", "right"}:
        raise ValueError(f"direction must be 'left' or 'right', got {direction!r}")
    selected = set(int(i) for i in selected_indices)
    order = list(range(size))
    if direction == "left":
        for pos in range(1, size):
            if order[pos] in selected and order[pos - 1] not in selected:
                order[pos - 1], order[pos] = order[pos], order[pos - 1]
    else:
        for pos in range(size - 2, -1, -1):
            if order[pos] in selected and order[pos + 1] not in selected:
                order[pos], order[pos + 1] = order[pos + 1], order[pos]
    return order


def level_order_for_labels(levels: Sequence[str], labels: Sequence[str]) -> list[int]:
    """Return current display indices ordered by a persisted label list."""
    label_to_idx = {str(level): i for i, level in enumerate(levels)}
    order = [label_to_idx[str(label)] for label in labels if str(label) in label_to_idx]
    seen = set(order)
    order.extend(i for i in range(len(levels)) if i not in seen)
    return order
