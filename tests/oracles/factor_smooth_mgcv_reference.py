"""Export the fs mgcv parity rows for ``factor_smooth_mgcv_reference.R``, and merge its fits.

    uv run python tests/oracles/factor_smooth_mgcv_reference.py export DIR
    Rscript tests/oracles/factor_smooth_mgcv_reference.R DIR
    uv run python tests/oracles/factor_smooth_mgcv_reference.py merge DIR

The rows stay as pinned in the fixture; only each case's ``reference`` and the
metadata are rewritten.
"""

from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

import pandas as pd

FIXTURE = Path(__file__).parents[1] / "fixtures" / "factor_smooth_mgcv_reference.json"
CASES = ("gaussian", "poisson", "poisson_global", "poisson_global_discrete")
MODEL = (
    'paraPen on s(x, bs="ps", k=6, m=2) per level of f: the second-difference penalty and '
    "one penalty per null_j on degree j, orthonormal over the rows"
)


def main(command: str, directory: str) -> None:
    out = Path(directory)
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    if command == "export":
        for name in CASES:
            for part in ("data", "prediction_data", "unseen_data"):
                frame = pd.DataFrame(fixture[name][part])
                frame.to_csv(out / f"{name}_{part}.csv", index=False, float_format="%.17g")
        return
    if command != "merge":
        raise SystemExit(f"unknown command {command!r}; use export or merge")
    for name in CASES:
        reference = (out / f"{name}_reference.json").read_text(encoding="utf-8")
        fixture[name]["reference"] = json.loads(reference)
    versions = json.loads((out / "versions.json").read_text(encoding="utf-8"))
    fixture["metadata"].update(
        generated_at=datetime.date.today().isoformat(), factor_smooth=MODEL, **versions
    )
    FIXTURE.write_text(json.dumps(fixture, separators=(",", ":")), encoding="utf-8")


if __name__ == "__main__":
    main(*sys.argv[1:])
