"""Retired: the range-space sum-to-zero block factor (one-engine design §3.5, §3.12).

``basis="sz"`` factor smooths are factored on the balance tree
(``superglm.solvers._structured.balance_tree``).  The names a model pickled by
an earlier build stores under this module stay importable as inert stand-ins
(PEP 562), so the model loads and rebuilds its inference state with the
current engine on first use (``superglm.model.retired_state``).  Release
0.37.0 may drop these names.
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
