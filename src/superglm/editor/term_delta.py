"""State replies that carry only the terms their client lacks.

The browser keeps a copy of every term and an edit usually changes one, so a
whole state is mostly what the browser already holds. This is state transfer per
object (Saito and Shapiro, *Optimistic Replication*, ACM Computing Surveys 37(1),
2005): every reply names each term's content by a hash, and a client that says
which reply it last merged gets the terms whose hash differs from that reply's.

The hash compares what a reply would send, so a term that changes because another
one did, as a display anchored on other terms can, is sent like any other. A
client whose reply has been forgotten, or that names none, gets every term. One
that sends no header gets the plain state.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from typing import Any

from superglm.editor.io import jsonable

KNOWN_STATE_HEADER = "X-SuperGLM-Editor-Known-State"

# Replies a slower page or a retried request may still be working from.
_REMEMBERED_REPLIES = 64


def _is_state(value: Any) -> bool:
    return isinstance(value, dict) and "terms" in value and "state_generation" in value


class TermDeltas:
    """The term hashes of recent replies, by state generation, across a server's clients."""

    def __init__(self) -> None:
        self._versions: OrderedDict[int, dict[str, str]] = OrderedDict()
        self._lock = threading.Lock()

    def reply(self, payload: Any, known: str) -> Any:
        """``payload`` with its state cut to what a client holding reply ``known`` lacks."""
        if _is_state(payload):
            return self._cut(payload, known)
        if isinstance(payload, dict) and _is_state(payload.get("state")):
            return {**payload, "state": self._cut(payload["state"], known)}
        return payload

    def _cut(self, state: dict[str, Any], known: str) -> dict[str, Any]:
        terms = jsonable(state["terms"])
        versions = {
            name: hashlib.blake2b(
                json.dumps(term, allow_nan=False).encode("utf-8"), digest_size=8
            ).hexdigest()
            for name, term in terms.items()
        }
        with self._lock:
            base = self._versions.get(int(known)) if known.isdecimal() else None
            self._versions[int(state["state_generation"])] = versions
            while len(self._versions) > _REMEMBERED_REPLIES:
                self._versions.popitem(last=False)
        changed = (
            terms
            if base is None
            else {name: term for name, term in terms.items() if base.get(name) != versions[name]}
        )
        return {**state, "terms": changed, "term_versions": versions}


__all__ = ["KNOWN_STATE_HEADER", "TermDeltas"]
