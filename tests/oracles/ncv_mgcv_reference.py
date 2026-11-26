"""Export the library's NCV designs for ``ncv_mgcv_reference.R``, and merge mgcv's fits into the fixture.

    uv run python tests/oracles/ncv_mgcv_reference.py export DIR
    Rscript tests/oracles/ncv_mgcv_reference.R DIR
    uv run python tests/oracles/ncv_mgcv_reference.py merge DIR

``export`` simulates the rows of ``tests/test_fit_ncv_mgcv.py`` (fixed seeds), fits its model once
to build the design, and writes per family the dense design with the intercept first, one
``(p + 1) x (p + 1)`` penalty per smoothing parameter in the order of the library's
``log lambda``, the response and weights, and per hold-out form the library's neighbourhoods as
mgcv's ``nei = list(a, ma, d, md)`` (1-based, ``ma`` and ``md`` the end of each neighbourhood).
It writes the fixture with the rows, those neighbourhoods and a fingerprint of the design; ``merge``
adds each case's mgcv reference and the R and mgcv versions.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

from tests.test_fit_ncv_mgcv import (  # noqa: E402
    FAMILIES,
    FIXTURE,
    FORMS,
    design,
    fingerprint,
    library_problem,
    mgcv_nei,
    neighbourhoods_of,
    simulate,
)

ROW_COLUMNS = ("x1", "x2", "z", "f", "group", "w", "y")


def export(out: Path) -> None:
    fixture: dict = {"metadata": {}}
    for family in FAMILIES:
        frame = simulate(family)
        _, problem = library_problem(family, frame)
        X, penalties = design(problem)
        np.savetxt(out / f"{family}_X.csv", X, delimiter=",", fmt="%.17g")
        for j, S in enumerate(penalties, start=1):
            np.savetxt(out / f"{family}_S{j}.csv", S, delimiter=",", fmt="%.17g")
        frame[["y", "w"]].to_csv(out / f"{family}_rows.csv", index=False, float_format="%.17g")
        entry = {
            "rows": {c: frame[c].tolist() for c in ROW_COLUMNS},
            "lambda_names": list(problem.names),
            "design": fingerprint(X, penalties),
            "forms": {},
        }
        for form in FORMS:
            nei = mgcv_nei(neighbourhoods_of(form, frame))
            for part, values in nei.items():
                np.savetxt(out / f"{family}_{form}_{part}.txt", values, fmt="%d")
            entry["forms"][form] = {"nei": nei}
        fixture[family] = entry
    FIXTURE.write_text(json.dumps(fixture, separators=(",", ":")), encoding="utf-8")


def merge(out: Path) -> None:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    for family in FAMILIES:
        for form in FORMS:
            text = (out / f"{family}_{form}_reference.json").read_text(encoding="utf-8")
            fixture[family]["forms"][form]["mgcv"] = json.loads(text)
    versions = json.loads((out / "versions.json").read_text(encoding="utf-8"))
    fixture["metadata"] = {
        "oracle": "mgcv gam(y ~ X - 1, weights = w, paraPen = list(X = S), method = 'NCV', nei)",
        **versions,
    }
    FIXTURE.write_text(json.dumps(fixture, separators=(",", ":")), encoding="utf-8")


if __name__ == "__main__":
    command, directory = sys.argv[1:]
    {"export": export, "merge": merge}[command](Path(directory))
