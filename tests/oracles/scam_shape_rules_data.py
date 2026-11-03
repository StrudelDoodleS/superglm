"""Write the four seeded datasets ``scam_shape_rules.R`` fits, one CSV per shape.

Usage: ``uv run python tests/oracles/scam_shape_rules_data.py DATA_DIR``.  The
fixture ``tests/fixtures/scam_shape_rules.json`` stores these data next to
scam's outputs, so the test never regenerates them; see the R script's header
for the full regeneration recipe.  ``x`` is rounded to six decimals and ``y``
is a count, so every value reads back exactly in R and in Python.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SEED = 20261103
N_ROWS = 400

# scam's shape name -> log-mean on [0, 1] with that shape
LOG_MEANS = {
    "micv": lambda x: 0.2 + 1.6 * (1.0 - np.exp(-3.0 * x)),  # increasing, concave
    "micx": lambda x: 0.2 + 1.5 * x**2,  # increasing, convex
    "mdcv": lambda x: 1.7 - 1.5 * x**2,  # decreasing, concave
    "mdcx": lambda x: 0.2 + 1.6 * np.exp(-3.0 * x),  # decreasing, convex
}


def write_datasets(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    for name, log_mean in LOG_MEANS.items():
        x = np.round(rng.uniform(0.0, 1.0, N_ROWS), 6)
        y = rng.poisson(np.exp(log_mean(x)))
        rows = "".join(f"{xi:.6f},{yi:d}\n" for xi, yi in zip(x, y, strict=True))
        (directory / f"{name}.csv").write_text("x,y\n" + rows, encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    write_datasets(Path(sys.argv[1]))
