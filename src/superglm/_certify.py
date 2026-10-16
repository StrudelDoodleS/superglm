"""The model-wide certification dial: tolerance-based decisions by default, certificates on request.

A fitted model publishes some decisions that a cheap rule settles almost
always and an expensive certificate settles always.  The data rank is the
first: by default it is read off the data Gram the fit has already formed, at
the eigensolver's tolerance (``rank.decide_data_rank``); with certification
on, a Gram that cannot certify its own rank is decided again on a factor of
every row, as before.  The dial is one switch so that later certificates can
move behind it without a second option.

``SuperGLM._parallel_scope`` enters :func:`certification_scope` with the
model's ``_certify`` attribute around every fit, refit and inference call, so
code below the model reads :func:`certification_requested` instead of taking
a parameter through every solver signature.  Outside any model scope the dial
is off.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_CERTIFY: ContextVar[bool] = ContextVar("superglm_certify", default=False)


@contextmanager
def certification_scope(enabled: bool) -> Iterator[None]:
    """Turn certification on or off for the code run inside the scope."""
    token = _CERTIFY.set(bool(enabled))
    try:
        yield
    finally:
        _CERTIFY.reset(token)


def certification_requested() -> bool:
    """Whether the enclosing scope asked for certified decisions."""
    return _CERTIFY.get()
