"""freMTPL2 frequency data with claim counts recounted from the severity file.

``freMTPL2freq`` (OpenML 41214) carries a ``ClaimNb`` column that is wrong for
the policies with ``IDpol <= 24500``: those 9,139 policies all show at least
one claim (claim frequency 2.09 per year against 0.075 elsewhere), and almost
none of their claims has a record in ``freMTPL2sev`` (OpenML 41215).
Wüthrich & Merz therefore take the claim counts from the severity file
instead (M. V. Wüthrich and M. Merz, *Statistical Foundations of Actuarial
Learning and its Applications*, Springer 2023, open access, Section 13.1 and
Listing 13.1). Their cleaned book has 678,007 policies and 26,383 claims.

:func:`recount_claims` applies that rule:

1. ``ClaimNb`` becomes the number of ``freMTPL2sev`` records with the
   policy's ``IDpol`` (0 for a policy with none), and ``ClaimTotal`` the sum of
   their ``ClaimAmount``. This replaces every policy's count, not only those
   up to 24,500. Severity records whose ``IDpol`` is not in the frequency file
   are ignored.
2. Policies with more than five recounted claims are dropped (six policies,
   with 6, 8, 9, 11, 11 and 16 claims).
3. ``Exposure`` is capped at one year.

The row order of the frequency file is kept. The result matches the authors'
own published cleaned file (``freMTPL2freq.rda``, linked from Richman,
Scognamiglio & Wüthrich, *The Credibility Transformer*, 2024, footnote 1) on
all 678,007 rows: the same ``IDpol``, ``ClaimNb``, capped ``Exposure`` and
covariates, in the same order, and the same ``ClaimTotal`` up to summation
order (nine multi-claim policies differ by at most 2.2e-16 relative). That check was run once by
hand on 2026-10-14; it is not part of the test suite, because that file is not
one of the pinned artifacts.

This module lives with the examples on purpose: it is a data recipe for the
documentation and the real-data tests, not part of the ``superglm`` API.
``tests/_datasets.py`` loads it by path.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

#: The largest recounted claim count a policy may carry and stay in the book.
MAX_CLAIMS = 5
#: Exposure, in years, above which a policy's exposure is capped.
EXPOSURE_CAP = 1.0
#: Policies up to this ``IDpol`` carry the wrong ``ClaimNb`` in freMTPL2freq.
CORRUPTED_IDPOL_MAX = 24500

FREQ_FILE = "freMTPL2freq.parquet"
SEV_FILE = "freMTPL2sev.parquet"


def recount_claims(freq: pd.DataFrame, sev: pd.DataFrame) -> pd.DataFrame:
    """Return ``freq`` with ``ClaimNb`` recounted from ``sev`` (Wüthrich & Merz).

    ``freq`` needs ``IDpol``, ``ClaimNb`` and ``Exposure``; ``sev`` needs
    ``IDpol`` and ``ClaimAmount``. The returned frame has a fresh index, an
    integer ``ClaimNb``, a float ``ClaimTotal`` and ``Exposure`` capped at
    :data:`EXPOSURE_CAP`; policies with more than :data:`MAX_CLAIMS` claims
    are dropped. Neither input is modified.
    """
    if freq["IDpol"].duplicated().any():
        raise ValueError("freMTPL2freq has a repeated IDpol; the recount needs one row per policy")
    by_policy = sev.groupby("IDpol", sort=False)["ClaimAmount"].agg(["size", "sum"])
    out = freq.copy()
    ids = out["IDpol"]
    out["ClaimNb"] = ids.map(by_policy["size"]).fillna(0).to_numpy(dtype=np.int64)
    out["ClaimTotal"] = ids.map(by_policy["sum"]).fillna(0.0).to_numpy(dtype=np.float64)
    out = out.loc[out["ClaimNb"] <= MAX_CLAIMS].reset_index(drop=True)
    out["Exposure"] = np.minimum(out["Exposure"].to_numpy(dtype=np.float64), EXPOSURE_CAP)
    return out


def default_data_dirs() -> list[Path]:
    """Where :func:`load_corrected_freq` looks, in order (as ``tests/_datasets.py``)."""
    dirs = [os.environ.get("SUPERGLM_DATA_DIR", ""), Path.home() / ".cache" / "superglm"]
    dirs.append(Path(__file__).resolve().parent.parent / "data")
    return [Path(d) for d in dirs if d]


def find_data_dir(data_dir: str | Path | None = None) -> Path:
    """The first directory holding both freMTPL2 parquet files."""
    candidates = [Path(data_dir)] if data_dir is not None else default_data_dirs()
    for d in candidates:
        if (d / FREQ_FILE).exists() and (d / SEV_FILE).exists():
            return d
    raise FileNotFoundError(
        f"{FREQ_FILE} and {SEV_FILE} not found together in any of "
        f"{[str(d) for d in candidates]}; fetch both with "
        "`uv run python scripts/fetch_fremtpl.py --dest data/`"
    )


def load_corrected_freq(data_dir: str | Path | None = None) -> pd.DataFrame:
    """Read both freMTPL2 files from ``data_dir`` and apply :func:`recount_claims`."""
    d = find_data_dir(data_dir)
    return recount_claims(pd.read_parquet(d / FREQ_FILE), pd.read_parquet(d / SEV_FILE))
