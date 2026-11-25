"""The tensor penalty's root for PSST profiling, from the pair's margins.

The overlap span is what the pair's own mains already cover: ``[intercept | A | B]``
with ``A``/``B`` the centered marginal menus. The tensor block's column order is
C-order ``p * k_b + q``, matching the row-Kronecker in ``features/interaction.py``.
The assembled Gram producers (``pair_overlap_moments``, ``tensor_penalty``) live in
``tests/_overlap_oracle.py`` as the definition the factors are graded against; an
import guard keeps them out of production.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from superglm.screening._factor_kernels import _penalty_spectrum


def tensor_penalty_root(S1: NDArray, S2: NDArray) -> NDArray:
    """``rootS`` with ``rootS' rootS`` the tensor penalty, assembled from the margins.

    ``kron(S1, I) + kron(I, S2)`` is a Kronecker sum, so with
    ``S1 = Qa diag(a) Qa'`` and ``S2 = Qb diag(b) Qb'`` its eigenvectors are
    ``kron(Qa, Qb)`` and its eigenvalues ``a_p + b_q`` in C-order
    ``p * k_b + q``. Assembling the root from the marginal spectra avoids
    resolving exact null directions at the assembled dimension's ``eigh``
    round-off ``n eps ||S||_2``, which large smoothing parameters amplify.

    Each margin is rooted by
    :func:`superglm.screening._factor_kernels._penalty_root`'s policy, matching
    the structured path. Unlike the structured path, the dense path reports a
    dropped direction as free rather than refusing. A margin without a penalty
    passes a zero block of its own width.
    """
    a, Qa, _, _ = _penalty_spectrum(np.asarray(S1, dtype=np.float64))
    b, Qb, _, _ = _penalty_spectrum(np.asarray(S2, dtype=np.float64))
    values = (a[:, None] + b[None, :]).reshape(-1)
    keep = values > 0.0
    if not np.any(keep):
        return np.zeros((0, a.size * b.size), dtype=np.float64)
    vectors = np.kron(Qa, Qb)
    return (vectors[:, keep] * np.sqrt(values[keep])).T
