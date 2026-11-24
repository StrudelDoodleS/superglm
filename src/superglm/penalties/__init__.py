"""Penalties for SuperGLM fits."""

import typing as _typing

from superglm._lazy import attach

if _typing.TYPE_CHECKING:
    from superglm.penalties.base import Flavor, Penalty
    from superglm.penalties.flavors import Adaptive
    from superglm.penalties.group_elastic_net import GroupElasticNet
    from superglm.penalties.group_lasso import GroupLasso
    from superglm.penalties.ridge import Ridge
    from superglm.penalties.sparse_group_lasso import SparseGroupLasso

# Each name is read from its module on first access (``superglm._lazy``): the package's body
# imports none of its modules, so a thread that imports one of them by path while another thread
# runs this body never waits on a sibling that thread holds.
_LAZY_EXPORTS: dict[str, tuple[str, ...]] = {
    "superglm.penalties.base": ("Flavor", "Penalty"),
    "superglm.penalties.flavors": ("Adaptive",),
    "superglm.penalties.group_elastic_net": ("GroupElasticNet",),
    "superglm.penalties.group_lasso": ("GroupLasso",),
    "superglm.penalties.ridge": ("Ridge",),
    "superglm.penalties.sparse_group_lasso": ("SparseGroupLasso",),
}

__getattr__, __dir__ = attach(__name__, __path__, _LAZY_EXPORTS)

__all__ = [
    "Penalty",
    "Flavor",
    "GroupElasticNet",
    "GroupLasso",
    "SparseGroupLasso",
    "Ridge",
    "Adaptive",
]
