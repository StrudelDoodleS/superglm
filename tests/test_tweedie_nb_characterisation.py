"""The rebuilt Tweedie/NB2 code against the outputs recorded from the pre-rebuild code.

The fixture comes from ``benchmarks/tweedie_nb_characterisation.py`` run on the
master commit named in its provenance.
"""

import json
import math
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import chi2

from superglm._tweedie import tweedie_logpdf, tweedie_unit_deviance
from superglm.profiling.tweedie import profile_phi_at
from superglm.reml.observed_geometry import ObservedModeNotCertifiedError

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "tweedie_nb_characterisation.json").read_text()
)
EPS = np.finfo(np.float64).eps
# estimate_p's default Brent resolution.
XATOL = 1e-3


@pytest.fixture
def characterisation_case():
    """The builder that generated the fixture: ``(model, X, y)`` for a case name."""
    from benchmarks.tweedie_nb_characterisation import build_case

    return build_case


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


@pytest.mark.parametrize(
    "row", FIXTURE["reml_phi"] + FIXTURE["reml_phi_books"], ids=lambda r: r["case"]
)
def test_reml_phi_matches_master(row):
    from benchmarks.tweedie_nb_characterisation import build_case

    model, X, y = build_case(row["case"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_reml(X, y)
    # The master saturated likelihood used the series (or i1e at p=1.5) to <=1e-13;
    # phi moves by that over the profile curvature, well under 1e-9 relative.
    assert model.result.phi == pytest.approx(row["phi"], rel=1e-9)


def _brent_reach(p: float) -> float:
    """How far scipy's bounded Brent can leave p_hat from the minimiser it brackets.

    fminbound stops once |x - m| <= 2 tol1 - (b - a) / 2, tol1 = sqrt(eps) |x| + xatol / 3,
    so x is within 2 tol1 of both ends of a bracket that holds the minimiser.
    """
    return 2.0 * (math.sqrt(EPS) * abs(p) + XATOL / 3.0)


def _nll_bound(row: dict) -> float:
    """The mean NLL's change from the density alone at a fixed mean and phi.

    Each row's master value is within old_route_max_rel_error of exact per unit
    max(1, |logpdf|), and the series within 64 eps of it; averaged over the rows
    that is the fixture's mean_logpdf_scale times their sum.
    """
    return (FIXTURE["old_route_max_rel_error"] + 64.0 * EPS) * row["mean_logpdf_scale"]


def _candidate_determination(model, result, X, y, fit_mode: str) -> float:
    """Mean NLL change from where a warm-started candidate fit stops.

    ML candidates start from the previous candidate and stop once their
    objective changes by under tol relative, so the deviance term D / (2 phi n)
    carries that much; REML candidates start cold and repeat master's exactly.
    """
    if fit_mode != "fit":
        return 0.0
    mu = np.asarray(model.predict(X), dtype=np.float64)
    deviance = float(np.sum(tweedie_unit_deviance(np.asarray(y, float), mu, result.p_hat)))
    return model._tol * deviance / (2.0 * result.phi_hat * len(y))


def _censored_on_master(row: dict) -> bool:
    return any("certifiable-region boundary" in w for w in row["warnings"])


@pytest.mark.slow
@pytest.mark.parametrize("row", FIXTURE["estimate_p"], ids=lambda r: f"{r['case']}-{r['fit_mode']}")
def test_estimate_p_matches_master(row, characterisation_case):
    model, X, y = characterisation_case(row["case"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.estimate_p(X, y, fit_mode=row["fit_mode"], ci_alpha=0.05)
        at_master_p = result._objective(row["p_hat"])
    nll_bound = _nll_bound(row)
    determination = _candidate_determination(model, result, X, y, row["fit_mode"])
    # The same profile: one candidate at master's p_hat moves only by the density
    # and by where its fit stops.
    assert abs(at_master_p - row["search_nll"]) <= nll_bound + determination

    if _censored_on_master(row):
        # Master stopped next to a power whose REML mode it could not certify.
        # Certification is a knife edge (a mode score against a 1e-9 bar), so the
        # rebuilt search may certify that power and route past it: it must land
        # no worse, and still disclose any censoring it meets.
        assert result.search_nll <= row["search_nll"] + nll_bound
        assert any("censored" in w for w in result.warnings)
        return

    reach = _brent_reach(row["p_hat"])
    # Both searches bracket the minimiser of profiles that differ by at most
    # nll_bound + determination; master's joint ML (unpen, fit) is Newton-precise.
    assert abs(result.p_hat - row["p_hat"]) <= 2.0 * reach
    # Each estimate's NLL is within c reach^2 / 2 of the profile minimum, with
    # c = d2 nll / dp2 recovered from master's interval (endpoint slope over
    # its distance from p_hat, exact for a quadratic profile).
    curvature = max(
        slope / abs(end - row["p_hat"])
        for slope, end in zip(row["ci95_slopes"], row["ci95"], strict=True)
    )
    assert abs(result.search_nll - row["search_nll"]) <= (
        0.5 * curvature * reach**2 + nll_bound + determination
    )
    # The published phi is the ML dispersion at the published mean (pinned by
    # test_published_phi_is_profiled_at_the_published_mean). Taken at master's
    # p_hat, that dispersion moves only by a score perturbation of
    # nll_bound + determination over its curvature in log phi: the published
    # mean is master's where p_hat agrees, and on the joint-ML row the design is
    # unpenalized, so the deviance is stationary in the coefficients.
    mu = np.asarray(model.predict(X), dtype=np.float64)
    phi_at_master_p = profile_phi_at(np.asarray(y, float), mu, np.ones(len(y)), row["p_hat"]).phi
    log_phi_bound = (nll_bound + determination) / row["phi_log_curvature"]
    assert abs(math.log(phi_at_master_p / row["phi_hat"])) <= log_phi_bound
    shift = abs(result.p_hat - row["p_hat"])
    # Interval endpoints: master accepted an LR residual of 1e-3 chi2 cutoff and
    # the rebuild's brentq stops within 1e-12 + 1e-6 |end|; an NLL change of d at
    # the endpoint and at p_hat moves the crossing by d / slope.
    cutoff = chi2.ppf(0.95, 1)
    ends = result.ci(0.05)
    for end, master_end, slope in zip(ends, row["ci95"], row["ci95_slopes"], strict=True):
        master_root = 1e-3 * cutoff / (2.0 * row["n"] * slope)
        moved = 2.0 * (nll_bound + determination) + 0.5 * curvature * shift**2
        assert abs(end - master_end) <= master_root + 1e-12 + 1e-6 * abs(end) + moved / slope


def test_boundary_optimum_is_reported_not_raised(characterisation_case):
    model, X, y = characterisation_case("zeros90")
    result = model.estimate_p(X, y, p_bounds=(1.6, 1.9))  # true p = 1.5 lies below the window
    assert result.p_hat == 1.6
    assert any("bound" in w for w in result.warnings)
    assert result.interval(0.05).lower_censored


def test_uncertifiable_power_is_skipped(characterisation_case, monkeypatch):
    model, X, y = characterisation_case("positive96")
    original = type(model).fit_reml

    def flaky(self, *args, **kwargs):
        if self._family_config.p > 1.9:
            raise ObservedModeNotCertifiedError(1e-3, 1e-9)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(type(model), "fit_reml", flaky)
    result = model.estimate_p(X, y, fit_mode="reml")
    assert math.isinf(result.evaluations.set_index("p").loc[1.95, "nll"])
    assert any("skipped" in w for w in result.warnings)
