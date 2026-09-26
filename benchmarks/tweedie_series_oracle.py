"""50-digit reference values of the Dunn-Smyth series log W(t) = log sum_j t^j / (j! Gamma(a j)).

Run with ``uv run --with mpmath python benchmarks/tweedie_series_oracle.py
tests/fixtures/tweedie_series_oracle.json``. The float64 (log_t, a) pairs are
the inputs the compiled kernel receives; the reference is their exact log W,
so the fixture measures the kernel's float64 evaluation error alone.
"""

import itertools
import json
import math
import sys
from pathlib import Path

import mpmath as mp

mp.mp.dps = 50

POWERS = (1.01, 1.05, 1.1, 1.3, 1.5, 1.7, 1.9, 1.95, 1.99)
PHIS = (1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0)
RESPONSES = (1e-3, 0.1, 1.0, 10.0, 1e3)
# Past this peak index a 50-digit sum costs seconds per row.
MAX_MODE = 3e5
# Terms 120 log-units below the peak are below the 50-digit resolution.
LOG_FLOOR = 120


def _side(term, j: int, step: int, peak):
    """Sum exp(term(j) - peak) from j in one direction until a term falls below the floor."""
    total = mp.mpf(0)
    while j >= 1:
        q = term(j)
        total += mp.exp(q - peak)
        if q < peak - LOG_FLOOR:
            break
        j += step
    return total


def reference_log_w(log_t: float, a: float) -> tuple[str, int]:
    lt, am = mp.mpf(log_t), mp.mpf(a)

    def term(j: int):
        return j * lt - mp.loggamma(j + 1) - mp.loggamma(am * j)

    mode = max(1, int(math.exp((log_t - a * math.log(a)) / (a + 1))))
    peak = max(term(j) for j in range(max(1, mode - 2), mode + 3))
    total = _side(term, mode, 1, peak) + _side(term, mode - 1, -1, peak)
    return mp.nstr(peak + mp.log(total), 40), mode


def main(out: Path) -> None:
    rows = []
    for p, phi, y in itertools.product(POWERS, PHIS, RESPONSES):
        a = (2 - p) / (p - 1)
        log_t = a * (math.log(y) - math.log(p - 1)) - math.log(2 - p) - (a + 1) * math.log(phi)
        if math.exp((log_t - a * math.log(a)) / (a + 1)) > MAX_MODE:
            continue
        value, mode = reference_log_w(log_t, a)
        rows.append({"p": p, "a": a, "log_t": log_t, "mode": mode, "log_w": value})
    payload = {"generator": "benchmarks/tweedie_series_oracle.py", "digits": 50, "rows": rows}
    out.write_text(json.dumps(payload, indent=1) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
