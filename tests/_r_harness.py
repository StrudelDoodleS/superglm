"""Lazy availability gate for the R oracle harness.

Every R-oracle suite needs the same answer -- is there an ``Rscript`` on the
path, and does it have ``mgcv`` and ``jsonlite`` available -- and
each used to work it out with a module-level ``subprocess.run``, either as a
bare ``HAS_R_HARNESS`` constant or inside the condition of a
``@pytest.mark.skipif``.  A ``skipif`` condition is an ordinary expression
evaluated when the decorator is applied, so both spellings ran at *import*
time.  That made merely COLLECTING the suite spawn one R subprocess per module,
and on a box with no R at all the probe raised ``FileNotFoundError`` from
``subprocess.run`` before any test existed to skip -- a collection error rather
than the skip the gate was written to produce.

The probe therefore lives here, behind :func:`functools.cache`, and is reached
only from inside a test body via :func:`require_r_harness`.  Collection no
longer runs R; the first test that actually needs R pays for the probe once and
the rest reuse the cached answer.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from functools import cache
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

_AVAILABILITY_PROBE = (
    'quit(status=if (requireNamespace("mgcv", quietly=TRUE) && '
    'requireNamespace("jsonlite", quietly=TRUE)) 0 else 1)'
)

SKIP_REASON = "requires R with mgcv and jsonlite"


def r_environment() -> dict[str, str]:
    """The environment R harness subprocesses run under."""
    return os.environ.copy()


@cache
def r_harness_available() -> bool:
    """Whether ``Rscript`` plus ``mgcv`` and ``jsonlite`` are usable.

    Cached, so the probe costs one R start-up per session rather than one per
    guarded test.  The ``shutil.which`` check comes first so that a box without
    R returns ``False`` instead of raising out of ``subprocess.run``.
    """
    if shutil.which("Rscript") is None:
        return False
    completed = subprocess.run(
        ["Rscript", "-e", _AVAILABILITY_PROBE],
        cwd=ROOT,
        env=r_environment(),
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0


def require_r_harness() -> None:
    """Skip the calling test unless the R oracle harness is available.

    Call this from inside the test body.  Using it as a ``skipif`` condition
    would reintroduce the import-time subprocess this module exists to remove.
    """
    if not r_harness_available():
        pytest.skip(SKIP_REASON)
