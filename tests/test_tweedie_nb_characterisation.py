"""The rebuilt Tweedie/NB2 code against the outputs recorded from the pre-rebuild code.

The fixture comes from ``benchmarks/tweedie_nb_characterisation.py`` run on the
master commit named in its provenance.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from superglm._tweedie import tweedie_logpdf, tweedie_unit_deviance

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "tweedie_nb_characterisation.json").read_text()
)
EPS = np.finfo(np.float64).eps


def _logpdf_float64_bound(row: dict) -> float:
    """Round-off envelope of the new evaluator, from the row's own term magnitudes.

    log f = log W - log y + c w / phi - w d / (2 phi). The series carries 16 eps per
    unit of its peak-term magnitude (tests/test_tweedie_series.py). The canonical
    term is a power and four roundings, the unit deviance's regular branch forms
    g = first - second with |first| + |second| <= 7 |g| at mu in {y/2, 2y} and each
    side within 3 eps, and the three additions round at these magnitudes: 32 eps
    per unit covers every one of them.
    """
    y, mu, phi, p, w = (row[key] for key in ("y", "mu", "phi", "p", "w"))
    half_deviance = w * float(tweedie_unit_deviance(np.array([y]), np.array([mu]), p)[0])
    half_deviance /= 2.0 * phi
    if y == 0.0:
        return 32.0 * EPS * max(1.0, half_deviance)
    a = (2.0 - p) / (p - 1.0)
    log_t = (
        a * (math.log(y) - math.log(p - 1.0)) - math.log(2.0 - p) + (a + 1.0) * math.log(w / phi)
    )
    mode = max(1, math.floor(math.exp((log_t - a * math.log(a)) / (a + 1.0))))
    peak = abs(mode * log_t) + abs(math.lgamma(mode + 1.0)) + abs(math.lgamma(a * mode))
    canonical = w * y ** (2.0 - p) / ((p - 1.0) * (2.0 - p) * phi)
    return 16.0 * EPS * max(1.0, peak) + 32.0 * EPS * (abs(math.log(y)) + canonical + half_deviance)


@pytest.mark.parametrize(
    "row",
    FIXTURE["logpdf"],
    ids=lambda r: f"p{r['p']}-phi{r['phi']}-y{r['y']}-mu{r['mu']}-w{r['w']}",
)
def test_logpdf_matches_the_exact_density_or_master(row):
    value = tweedie_logpdf(
        np.array([row["y"]]),
        np.array([row["mu"]]),
        row["phi"],
        row["p"],
        weights=np.array([row["w"]]),
    )[0]
    bound = _logpdf_float64_bound(row)
    if row["logpdf_exact"] is not None:
        # The fixture's 50-digit log density of the same float64 inputs.
        assert abs(value - row["logpdf_exact"]) <= bound
        return
    # Past the reference's mode cap only master's value exists. Its series shares
    # this evaluator's float64 conditioning, and its measured error on the route
    # is the fixture's old_route_max_rel_error_by_route.
    old_route_error = FIXTURE["old_route_max_rel_error_by_route"][row["route"]]
    assert abs(value - row["logpdf"]) <= bound + old_route_error * max(1.0, abs(row["logpdf"]))
