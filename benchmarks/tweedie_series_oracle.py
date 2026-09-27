"""50-digit reference values of the Dunn-Smyth series log W(t) = log sum_j t^j / (j! Gamma(a j)).

Run with ``uv run --with mpmath python benchmarks/tweedie_series_oracle.py
tests/fixtures/tweedie_series_oracle.json``. The float64 (log_t, a) pairs are
the inputs the compiled kernel receives; the reference is their exact log W,
so the fixture measures the kernel's float64 evaluation error alone.

The ``saturated`` rows are end to end: exact l_sat, T = -d l_sat / d log phi
and T' = dT / d log phi of one positive row from its float64 (p, y, w, phi), at
peak indices 1e2 to 1e7 that straddle the series-to-saddlepoint switch for
every power. Each is a direct sum of every term within 120 log-units of the
peak, up to about 1e5 terms at j = 1e7.
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
SATURATED_POWERS = (1.01, 1.05, 1.2, 1.5, 1.8, 1.95, 1.99)
SATURATED_PEAKS = (1e2, 3e2, 1e3, 3e3, 1e4, 3e4, 1e5, 1e6, 1e7)


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


def _moments(lt, am, j_s):
    """Peak, then the sums of exp(q - peak) times 1, (j - mode) and (j - mode)^2."""

    def term(j: int):
        return j * lt - mp.loggamma(j + 1) - mp.loggamma(am * j)

    mode = max(1, int(j_s))
    peak = term(mode)
    for step in (1, -1):
        while mode + step >= 1 and term(mode + step) > peak:
            mode += step
            peak = term(mode)
    sums = [mp.mpf(0)] * 3
    for j, step in ((mode, 1), (mode - 1, -1)):
        while j >= 1:
            q = term(j)
            weight = mp.exp(q - peak)
            sums = [s + weight * (j - mode) ** k for k, s in enumerate(sums)]
            if q < peak - LOG_FLOOR:
                break
            j += step
    return peak, mode, sums


def reference_saturated(p: float, y: float, w: float, phi: float) -> dict:
    """Exact l_sat, T and T' of one positive row from its float64 inputs."""
    P, Y, W, F = (mp.mpf(v) for v in (p, y, w, phi))
    am = (2 - P) / (P - 1)
    lt = am * (mp.log(Y) - mp.log(P - 1)) - mp.log(2 - P) + (am + 1) * (mp.log(W) - mp.log(F))
    j_s = mp.exp((lt - am * mp.log(am)) / (am + 1))
    peak, mode, (s0, s1, s2) = _moments(lt, am, j_s)
    mean, var = mode + s1 / s0, s2 / s0 - (s1 / s0) ** 2
    canonical = W * Y ** (2 - P) / ((1 - P) * (2 - P) * F)
    return {
        "peak_index": float(j_s),
        "l_sat": mp.nstr(peak + mp.log(s0) - mp.log(Y) + canonical, 30),
        "score": mp.nstr((am + 1) * mean + canonical, 30),
        "slope": mp.nstr(-((am + 1) ** 2) * var - canonical, 30),
    }


def saturated_rows() -> list[dict]:
    y, w = 1.7, 2.5
    rows = []
    for p, peak in itertools.product(SATURATED_POWERS, SATURATED_PEAKS):
        phi = w * y ** (2 - p) / ((2 - p) * peak)
        rows.append({"p": p, "y": y, "w": w, "phi": phi, **reference_saturated(p, y, w, phi)})
    return rows


def main(out: Path) -> None:
    rows = []
    for p, phi, y in itertools.product(POWERS, PHIS, RESPONSES):
        a = (2 - p) / (p - 1)
        log_t = a * (math.log(y) - math.log(p - 1)) - math.log(2 - p) - (a + 1) * math.log(phi)
        if math.exp((log_t - a * math.log(a)) / (a + 1)) > MAX_MODE:
            continue
        value, mode = reference_log_w(log_t, a)
        rows.append({"p": p, "a": a, "log_t": log_t, "mode": mode, "log_w": value})
    payload = {
        "generator": "benchmarks/tweedie_series_oracle.py",
        "digits": 50,
        "rows": rows,
        "saturated": saturated_rows(),
    }
    out.write_text(json.dumps(payload, indent=1) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
