from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class ExactSCOPSupport:
    B_unique: NDArray
    row_to_support: NDArray

    @property
    def n_unique(self) -> int:
        return self.B_unique.shape[0]

    def weighted_products(self, W: NDArray, z: NDArray) -> tuple[NDArray, NDArray]:
        W_agg = np.bincount(self.row_to_support, weights=W, minlength=self.n_unique)
        Wz_agg = np.bincount(self.row_to_support, weights=W * z, minlength=self.n_unique)
        BtWB = self.B_unique.T @ (self.B_unique * W_agg[:, None])
        BtWz = self.B_unique.T @ Wz_agg
        return BtWB, BtWz


def build_exact_scop_support(B_scop: NDArray) -> ExactSCOPSupport | None:
    B = np.asarray(B_scop, dtype=np.float64)
    B_unique, row_to_support = np.unique(B, axis=0, return_inverse=True)
    if B_unique.shape[0] == B.shape[0]:
        return None
    return ExactSCOPSupport(B_unique=B_unique, row_to_support=row_to_support)
