"""Retired: the range-space sum-to-zero block factor.

``basis="sz"`` factor smooths use the balance tree
(``superglm.solvers._structured.balance_tree``). Names pickled under this
module stay importable as inert stand-ins (PEP 562), so the model loads and
rebuilds its inference state on first use (``superglm.model.retired_state``).
"""

from __future__ import annotations

from superglm.solvers._structured.retired import module_getattr

__getattr__ = module_getattr(
    __name__,
    frozenset(
        {
            "ProfiledSumToZeroBlockFactor",
            "SumToZeroBlockFactor",
            "_LocalPSD",
            "_SymmetricBorderFactor",
        }
    ),
)
