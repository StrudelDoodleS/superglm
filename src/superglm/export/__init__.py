"""Rating table export helpers."""

import typing as _typing

from superglm._lazy import attach

if _typing.TYPE_CHECKING:
    from superglm.export.rating_tables import (
        RatingTableBaseNotRepresentableError,
        export_rating_tables,
    )

# Each name is read from its module on first access (``superglm._lazy``): the package's body
# imports none of its modules, so a thread that imports one of them by path while another thread
# runs this body never waits on a sibling that thread holds.
_LAZY_EXPORTS: dict[str, tuple[str, ...]] = {
    "superglm.export.rating_tables": (
        "RatingTableBaseNotRepresentableError",
        "export_rating_tables",
    ),
}

__getattr__, __dir__ = attach(__name__, __path__, _LAZY_EXPORTS)

__all__ = ["RatingTableBaseNotRepresentableError", "export_rating_tables"]
