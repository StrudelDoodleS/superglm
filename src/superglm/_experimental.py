"""The experimental label: features whose spelling may still change.

An experimental feature's name, its arguments and the layout of what it returns may change in a
minor release without a deprecation period. Its numbers are held to the same tests as the rest of
the library. The first call of any experimental feature in a session warns once; the ordinary
warnings filter silences it by class. There is no switch to turn a feature on or off.
"""

from __future__ import annotations

from superglm._warn import warn_caller

__all__ = ["ExperimentalWarning", "warn_experimental"]


class ExperimentalWarning(UserWarning):
    """Issued once per session on the first call to an experimental feature.

    An experimental feature's name, arguments and return layout may change in a
    minor release without a deprecation period. Its numbers are held to the same
    tests as the rest of the library.

    Silence this warning with the standard warnings filter::

        import warnings
        import superglm

        warnings.filterwarnings("ignore", category=superglm.ExperimentalWarning)
    """


# One flag for every experimental feature, not Python's per-location registry, which
# ``warnings.catch_warnings`` resets. It is set only after the warning returns, so under
# ``-W error`` each call raises until the class is filtered.
_announced = False


def warn_experimental(name: str) -> None:
    """Warn at the caller's line, once per session, that ``name`` is experimental."""
    global _announced
    if _announced:
        return
    warn_caller(
        f"{name} is experimental: its name, its arguments and the layout of what it returns may "
        "change in a minor release without a deprecation period. Its numbers are tested like the "
        "rest of superglm. This warning is shown once per session; to silence it, call "
        "warnings.filterwarnings('ignore', category=superglm.ExperimentalWarning).",
        ExperimentalWarning,
    )
    _announced = True
