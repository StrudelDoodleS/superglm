"""The tensor penalty's root for PSST profiling, from the pair's margins.

The overlap span is what the candidate pair's own mains already cover:
``[intercept | A | B]`` with ``A``/``B`` the centered marginal menus evaluated
on each covariate's support.  Column order of the tensor block is C-order
``p * k_b + q``, matching ``features/interaction.py``'s row-Kronecker and
``pair_score_curvature``.

Issue #257 moved the dense screening path onto design factors: ``_pair_factor``
reduces the pair's cell tables to one triangular factor and the ladder reads
``V_eff`` off a block of it, so no Gram is formed and the spectrum the
statistic works in is the design's rather than its square.  The assembled
Gram producers that route replaced (``pair_overlap_moments`` and
``tensor_penalty``) are the definition that factor is graded against, so they
live in ``tests/_overlap_oracle.py``.  An import guard keeps them from
becoming production again by accident:
``test_no_production_module_imports_a_retired_gram_producer``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from superglm.screening._factor_kernels import _penalty_spectrum


def tensor_penalty_root(S1: NDArray, S2: NDArray) -> NDArray:
    """``rootS`` with ``rootS' rootS`` the tensor penalty, from the MARGINS.

    ``kron(S1, I) + kron(I, S2)`` is a Kronecker SUM, so with
    ``S1 = Qa diag(a) Qa'`` and ``S2 = Qb diag(b) Qb'`` its eigenvectors are
    ``kron(Qa, Qb)`` and its eigenvalues are ``a_p + b_q`` -- exactly, and in
    the C-order ``p * k_b + q`` the tensor block already uses.  The root is
    therefore assembled rather than computed, and the only eigendecompositions
    taken are of the two MARGIN penalties.

    **THAT IS WHERE THE DENSE LADDER'S HIGH-EDGE ACCURACY COMES FROM, AND IT
    IS MEASURED.**  ``eigh`` is accurate to ``n eps ||S||_2`` ABSOLUTELY, so a
    direction that is exactly null in exact arithmetic comes back at the
    dimension's round-off -- and the ladder's high edge multiplies it by
    ``1e10``.  On ``moderate_pair`` the penalty is ``kron(S_a, I_23)`` with
    ``S_a`` nine wide and one exactly-null direction; its assembly is 207 wide
    and ``eigh`` resolves that direction only to 6.94e-13 of ``||S_ti||_2``,
    where on ``S_a`` itself it resolves to 4.3e-17.  Through the ladder, the
    dense arm's high-edge ``edf`` reads 8.26e-07 from the stacked-QR arbiter
    when the assembly is rooted and 4.99e-13 when the margins are -- and the
    statistic 7.38e-07 against 3.13e-11.  The same six orders separate the
    dense arm from the structured one, which has always rooted ``S_a``.

    ONE ROOTING POLICY, and it is
    :func:`superglm.screening._factor_kernels._penalty_root`'s: an eigenvalue
    inside ``eigh``'s own bar is taken at its MAGNITUDE, one outside it and
    negative is dropped.  Applied to each margin separately, which is what the
    structured path already does to ``S_a`` -- so the two paths root the same
    matrices by the same rule, and issue #323's "two policies deciding on
    different bars" cannot reappear between them.  The dense path does NOT
    inherit :func:`superglm.screening._structured._profile`'s refusal on a
    material drop: it has no numerical-certificate refusal contract, only a
    budget one, and adding one would delete rows the moment route published.
    A dropped direction is reported FREE, which is the direction
    :func:`_penalty_root` documents.

    A margin that carries no penalty passes a zero block of its own width -- a categorical contrast block is
    unpenalized and its zeros then contribute nothing to any eigenvalue sum.
    """
    a, Qa, _, _ = _penalty_spectrum(np.asarray(S1, dtype=np.float64))
    b, Qb, _, _ = _penalty_spectrum(np.asarray(S2, dtype=np.float64))
    values = (a[:, None] + b[None, :]).reshape(-1)
    keep = values > 0.0
    if not np.any(keep):
        return np.zeros((0, a.size * b.size), dtype=np.float64)
    vectors = np.kron(Qa, Qb)
    return (vectors[:, keep] * np.sqrt(values[keep])).T
