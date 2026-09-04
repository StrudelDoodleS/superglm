from __future__ import annotations

from typing import Any, cast

import pytest

from superglm.distributional.packing import packed_pairs


@pytest.mark.parametrize("k_parameters", range(1, 5))
def test_packed_pairs_use_canonical_upper_triangle(k_parameters: int) -> None:
    expected = tuple(
        (left, right) for left in range(k_parameters) for right in range(left, k_parameters)
    )

    assert packed_pairs(k_parameters) == expected


@pytest.mark.parametrize("k_parameters", [0, -1, True, 2.5])
def test_packed_pairs_rejects_invalid_parameter_count(k_parameters: object) -> None:
    with pytest.raises((TypeError, ValueError), match="k_parameters"):
        packed_pairs(cast(Any, k_parameters))
