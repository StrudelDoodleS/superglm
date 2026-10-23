"""The pickler behind ``SuperLSS``'s pickle and joblib support.

The standard pickler refuses the read-only mappings a model keeps its
configuration in; :func:`pickle_state` carries them. ``_restore_mapping_proxy``
stays at this path because files pickled by earlier releases name it.
"""

from __future__ import annotations

import copyreg
import io
import pickle
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

_MAPPING_PROXY_TYPE = type(MappingProxyType({}))


def _restore_mapping_proxy(values: Mapping[Any, Any]) -> MappingProxyType:
    return MappingProxyType(dict(values))


def _reduce_mapping_proxy(value: Mapping[Any, Any]) -> tuple[Any, tuple[dict[Any, Any]]]:
    return _restore_mapping_proxy, (dict(value),)


class _DistributionalPickler(pickle.Pickler):
    dispatch_table = copyreg.dispatch_table.copy()
    dispatch_table[_MAPPING_PROXY_TYPE] = _reduce_mapping_proxy


def pickle_state(value: Any) -> bytes:
    """Pickle ``value`` with the read-only mappings the standard pickler refuses."""
    stream = io.BytesIO()
    _DistributionalPickler(stream, protocol=pickle.HIGHEST_PROTOCOL).dump(value)
    return stream.getvalue()
