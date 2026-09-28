"""The rebuilt Tweedie/NB2 code against the outputs recorded from the pre-rebuild code.

The fixture comes from ``benchmarks/tweedie_nb_characterisation.py`` run on the
master commit named in its provenance.
"""

import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import gammaln
from scipy.stats import chi2

from superglm._tweedie import tweedie_logpdf, tweedie_unit_deviance
from superglm.profiling.nb import NBThetaBoundWarning
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


def _route_error_by_power() -> dict[float, float]:
    """Master's measured density error at each grid power, per unit max(1, |log f|)."""
    errors: dict[float, float] = {}
    for row in FIXTURE["logpdf"]:
        if row["logpdf_exact"] is None:
            continue
        error = abs(row["logpdf"] - row["logpdf_exact"]) / max(1.0, abs(row["logpdf"]))
        errors[row["p"]] = max(errors.get(row["p"], 0.0), error)
    return errors


ROUTE_ERROR_BY_POWER = _route_error_by_power()


def _nll_bound(row: dict) -> float:
    """The mean NLL's change from the density alone at a fixed mean and phi.

    Master's routes are measured against the 50-digit density at the fixture's
    grid powers; the case evaluates between the grid powers that bracket p_hat
    and its interval, so its rows are within the largest of those errors per
    unit max(1, |logpdf|), and the series within 64 eps. Averaged over the
    rows that is the fixture's mean_logpdf_scale times their sum.
    """
    reach = [row["p_hat"], *(row["ci95"] or ())]
    powers = sorted(ROUTE_ERROR_BY_POWER)
    lowest = max(p for p in powers if p <= min(reach))
    highest = min(p for p in powers if p >= max(reach))
    route_error = max(ROUTE_ERROR_BY_POWER[p] for p in powers if lowest <= p <= highest)
    return (route_error + 64.0 * EPS) * row["mean_logpdf_scale"]


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
    nll_bound = _nll_bound(row)

    if _censored_on_master(row):
        # Master stopped next to a power whose REML mode it could not certify.
        # Its candidates there ran through uncertified line-search trials -- the
        # unit deviance's eps * mu / y error for 0 < y < mu/2 swamped the PIRLS
        # merit -- so their REML paths, and where they stopped, are no reference.
        # The rebuilt search must land no worse and disclose any censoring it
        # still meets: an infeasible power beside p_hat.
        assert result.search_nll <= row["search_nll"] + nll_bound
        searched = result.evaluations.sort_values("p")["nll"].to_numpy()
        at = int(np.searchsorted(np.sort(result.evaluations["p"].to_numpy()), result.p_hat))
        beside_infeasible = bool(np.isinf(searched[max(at - 1, 0) : at + 2]).any())
        assert beside_infeasible == any("censored estimate" in w for w in result.warnings)
        return

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        at_master_p = result._objective(row["p_hat"])
    determination = _candidate_determination(model, result, X, y, row["fit_mode"])
    # The same profile: one candidate at master's p_hat moves only by the density
    # and by where its fit stops.
    assert abs(at_master_p - row["search_nll"]) <= nll_bound + determination

    reach = _brent_reach(row["p_hat"])
    # Both searches bracket the minimiser of profiles that differ by at most
    # nll_bound + determination; master's joint ML (unpen, fit) is Newton-precise.
    assert abs(result.p_hat - row["p_hat"]) <= 2.0 * reach
    shift = abs(result.p_hat - row["p_hat"])
    if len(result.evaluations) == row["n_evaluations"] and shift <= 1e-9:
        # The same Brent path: each candidate refits from the same predecessor
        # as master's did, so where its fit stops is master's too, and the
        # searched values differ by the density and by the candidate's slope
        # over the shift (at most the endpoint slopes, which bound it inside
        # the interval).
        slope_bound = 2.0 * max(row["ci95_slopes"]) * shift
        assert abs(result.search_nll - row["search_nll"]) <= nll_bound + slope_bound
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
    # Interval endpoints: master accepted an LR residual of 1e-3 chi2 cutoff and
    # the rebuild's brentq stops within its xtol of 1e-4 plus 4 eps |end|; an
    # NLL change of d at the endpoint and at p_hat moves the crossing by d / slope.
    cutoff = chi2.ppf(0.95, 1)
    ends = result.ci(0.05)
    for end, master_end, slope in zip(ends, row["ci95"], row["ci95_slopes"], strict=True):
        master_root = 1e-3 * cutoff / (2.0 * row["n"] * slope)
        moved = 2.0 * (nll_bound + determination) + 0.5 * curvature * shift**2
        assert abs(end - master_end) <= master_root + 1e-4 + 4 * EPS * abs(end) + moved / slope


def test_boundary_optimum_is_reported_not_raised(characterisation_case):
    model, X, y = characterisation_case("zeros90")
    result = model.estimate_p(X, y, p_bounds=(1.6, 1.9))  # true p = 1.5 lies below the window
    assert result.p_hat == 1.6
    assert any("search bound" in w for w in result.warnings)
    assert any("rounded responses" in w for w in result.warnings)
    assert result.interval(0.05).lower_censored
    # Spec section 6: the censored side is recorded in the warnings too.
    assert any("95% interval for p is censored at its lower end" in w for w in result.warnings)


def test_optimum_at_the_upper_bound_is_reported(characterisation_case):
    model, X, y = characterisation_case("zeros90")
    result = model.estimate_p(X, y, p_bounds=(1.2, 1.4))  # true p = 1.5 lies above the window
    assert result.p_hat == 1.4
    assert any("search bound" in w for w in result.warnings)
    # The rounded-responses artefact is a maximum as p -> 1, never at an upper bound.
    assert not any("rounded responses" in w for w in result.warnings)
    interval = result.interval(0.05)
    assert interval.upper_censored and interval.upper == 1.4
    assert any("censored at its upper end" in w for w in result.warnings)


def test_an_interval_computed_on_the_returned_result_does_not_reach_the_model(
    characterisation_case,
):
    model, X, y = characterisation_case("zeros90")
    result = model.estimate_p(X, y, p_bounds=(1.6, 1.9), ci_alpha=0.05)
    installed_warnings = list(model._tweedie_profile_result.warnings)
    result.interval(0.1)
    assert model.summary(alpha=0.1)._info["tweedie_p_ci_status"] == "not computed"
    assert model._tweedie_profile_result.warnings == installed_warnings
    assert len(result.warnings) > len(installed_warnings)


def test_estimate_p_profiles_the_prior_weighted_likelihood():
    # Exposure-weighted books are the ordinary case: the prior weight enters the
    # density, so phi_hat and the published NLL are the maximum of the weighted
    # log-likelihood at the published mean, found here by brute force.
    from scipy.optimize import minimize_scalar

    from superglm import Categorical, Spline, SuperGLM, families, generate_tweedie_cpg

    rng = np.random.default_rng(5)
    n = 4000
    x, level = rng.uniform(0.0, 1.0, n), rng.integers(0, 4, n)
    weights = rng.uniform(0.2, 3.0, n)
    mu = np.exp(0.2 + 0.5 * np.sin(2 * np.pi * x) + np.array([0.0, 0.2, -0.2, 0.1])[level])
    y = generate_tweedie_cpg(n, mu, 1.5 / weights, 1.4, rng=rng)
    X = pd.DataFrame({"x": x, "level": level.astype(str)})
    model = SuperGLM(
        family=families.tweedie(p=1.5),
        features={"x": Spline(n_knots=8), "level": Categorical()},
    )
    result = model.estimate_p(X, y, sample_weight=weights, p_bounds=(1.2, 1.7))
    fitted = np.asarray(model.predict(X), dtype=np.float64)

    def weighted_nll(log_phi):
        return -np.sum(tweedie_logpdf(y, fitted, math.exp(log_phi), result.p_hat, weights)) / n

    assert result.nll == pytest.approx(weighted_nll(math.log(result.phi_hat)), rel=1e-12)
    brute = minimize_scalar(
        weighted_nll, bounds=(-3.0, 3.0), method="bounded", options={"xatol": 1e-10}
    )
    # The bounded search resolves log phi only to where the NLL's round-off,
    # 64 eps per unit of the mean |log f|, matches its quadratic rise c du^2 / 2.
    scale = float(np.mean(np.abs(tweedie_logpdf(y, fitted, result.phi_hat, result.p_hat, weights))))
    step = 1e-3
    u = math.log(result.phi_hat)
    curvature = (weighted_nll(u + step) - 2.0 * weighted_nll(u) + weighted_nll(u - step)) / step**2
    resolution = math.sqrt(2.0 * 64.0 * EPS * max(1.0, scale) / curvature) + 1e-10
    assert abs(u - brute.x) <= resolution


def test_censored_interval_is_reported_in_the_summary(characterisation_case):
    model, X, y = characterisation_case("zeros90")
    result = model.estimate_p(X, y, p_bounds=(1.6, 1.9), ci_alpha=0.05)
    summary = model.summary(alpha=0.05)
    assert summary._info["tweedie_p_ci"] == result.ci(0.05)
    assert summary._info["tweedie_p_ci_status"] == "censored"
    assert "censored" in str(summary)


def test_published_phi_is_profiled_at_the_published_mean(characterisation_case):
    model, X, y = characterisation_case("zeros90")
    result = model.estimate_p(X, y, fit_mode="reml")
    expected = profile_phi_at(np.asarray(y, float), model.predict(X), np.ones(len(y)), result.p_hat)
    assert result.phi_hat == pytest.approx(expected.phi, rel=1e-12)
    assert model.result.phi == result.phi_hat
    assert result.nll == pytest.approx(expected.criterion / len(y), rel=1e-12)


def test_estimate_p_refuses_nonunit_frequency_weights(characterisation_case):
    model, X, y = characterisation_case("zeros90", weight_semantics="frequency")
    with pytest.raises(ValueError, match="frequency"):
        model.estimate_p(X, y, sample_weight=np.full(len(y), 2.0))


def test_progress_reports_each_candidate_while_the_search_runs(characterisation_case):
    model, X, y = characterisation_case("zeros90")
    events = []
    result = model.estimate_p(
        X, y, p_bounds=(1.4, 1.6), progress_callback=lambda *event: events.append(event)
    )
    rows = [payload["profile_trace"][0] for phase, payload in events if phase == "profiling"]
    assert rows == result.evaluations.to_dict("records")
    assert [phase for phase, _ in events[len(rows) :]] == ["best_found", "final_refit"]
    # The interval evaluates the same profile after the caller's display is done.
    result.ci(0.05)
    assert len(events) == len(rows) + 2


def test_estimate_p_signature_is_the_slim_one():
    import inspect

    from superglm import SuperGLM

    parameters = set(inspect.signature(SuperGLM.estimate_p).parameters)
    assert {"method", "phi_method", "kwargs"}.isdisjoint(parameters)


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


def test_an_unconverged_candidate_fit_leaves_the_estimate_unconverged(
    characterisation_case, monkeypatch
):
    # The publication refit converges; only the search's candidate fits do not.
    import dataclasses

    import superglm.profiling.tweedie as tweedie_module

    solve = tweedie_module._solve_coefficients

    def unconverged(*args, **kwargs):
        return dataclasses.replace(solve(*args, **kwargs), converged=False)

    monkeypatch.setattr(tweedie_module, "_solve_coefficients", unconverged)
    model, X, y = characterisation_case("zeros90")
    result = model.estimate_p(X, y, p_bounds=(1.4, 1.6))
    assert model.result.converged
    assert not result.evaluations["fit_converged"].any()
    assert not result.converged


def test_an_unconverged_reml_candidate_leaves_the_estimate_unconverged(
    characterisation_case, monkeypatch
):
    import dataclasses

    model, X, y = characterisation_case("zeros90")
    original = type(model).fit_reml

    def unconverged(self, *args, **kwargs):
        fitted = original(self, *args, **kwargs)
        self._reml_result = dataclasses.replace(self._reml_result, converged=False)
        return fitted

    # The publication refit does not go through fit_reml, so it still converges.
    monkeypatch.setattr(type(model), "fit_reml", unconverged)
    result = model.estimate_p(X, y, fit_mode="reml", p_bounds=(1.4, 1.6))
    assert model.result.converged
    assert not result.evaluations["fit_converged"].any()
    assert not result.converged


def test_theta_steps_reach_the_progress_callback_while_the_search_runs(characterisation_case):
    model, X, y = characterisation_case("nb_worst")
    events = []
    result = model.estimate_theta(X, y, progress_callback=lambda *event: events.append(event))
    rows = [payload["profile_trace"][0] for phase, payload in events if phase == "profiling"]
    assert rows == result.evaluations.to_dict("records")
    assert [phase for phase, _ in events[len(rows) :]] == ["best_found", "final_refit"]


def _one_published_digit(value: float) -> float:
    """One unit in the sixth significant digit, the resolution theta_hat is published at."""
    return 10.0 ** (math.floor(math.log10(abs(value))) - 5)


def _statistic_slope(result, theta: float, n: int) -> float:
    """d/d log theta of the likelihood-ratio statistic 2 n (nll - nll_hat), central difference."""
    h = 1e-4
    upper = result._profile_nll(theta * math.exp(h))
    lower = result._profile_nll(theta * math.exp(-h))
    return 2.0 * n * (upper - lower) / (2.0 * h)


@pytest.mark.parametrize(
    "row", FIXTURE["estimate_theta"], ids=lambda r: f"{r['case']}-{r['fit_mode']}"
)
def test_estimate_theta_matches_master(row, characterisation_case):
    import dataclasses

    model, X, y = characterisation_case(row["case"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = model.estimate_theta(X, y, fit_mode=row["fit_mode"])
    messages = [str(w.message) for w in caught if issubclass(w.category, NBThetaBoundWarning)]
    assert messages == row["warnings"] == result.warnings
    assert result.converged == row["converged"]
    # The alternation repeats master's: the same moment start, and direct-route
    # mean fits stopped at master's 1e-8. Each run locates the score root to
    # brentq's rtol of 1e-8, so the unrounded iterate lies in the rounding cell of
    # master's six published digits, widened by twice that.
    unrounded = float(result.evaluations["theta"].iloc[-1])
    cell = 0.5 * _one_published_digit(row["theta_hat"]) + 2e-8 * unrounded
    assert abs(unrounded - row["theta_hat"]) <= cell
    # At master's theta_hat the fixed-mean profile is the same family density on
    # the new publication fit. Each run's fit stops within tol (D + 1) of its
    # optimum (relative objective change below tol at a linear rate under 1/2),
    # so the two NLLs differ by at most tol (D + 1) / n.
    n = len(y)
    band = model._tol * (model.result.deviance + 1.0) / n
    at_master = dataclasses.replace(
        result,
        theta_hat=row["theta_hat"],
        nll=result._profile_nll(row["theta_hat"]),
        _ci_cache={},
    )
    assert abs(at_master.nll - row["nll"]) <= band
    if row["fit_mode"] == "reml":
        # Master inverted from theta_hat, which a REML publication's mean does
        # not maximise: theta_hat is estimated at the configured penalty. The
        # rebuild inverts from that mean's own profile optimum, so each end is
        # checked as that curve's crossing instead.
        _assert_crossings_from_the_optimum(at_master, 0.05, n)
        return
    # Each side is rooted to 1e-6 relative (master in theta, the rebuild in
    # log theta); an NLL offset of `band` at the estimate and at the endpoint
    # moves the statistic by 4 n band, hence the crossing by that over its slope.
    for end, master_end in zip(at_master.ci(0.05), row["ci95"]):
        allowed = 2e-6 + 4.0 * n * band / abs(_statistic_slope(at_master, end, n))
        assert abs(math.log(end) - math.log(master_end)) <= allowed


def _assert_crossings_from_the_optimum(result, alpha, n):
    """Each uncensored end is where 2 n (nll - nll at the published mean's optimum)
    crosses the cutoff, located to _CI_LOG_XTOL in log theta: the statistic there
    is within that step times its slope of the cutoff."""
    from superglm.profiling.nb import _CI_LOG_XTOL

    _, optimum_nll = result._optimum()
    interval = result._interval(alpha)
    cutoff = chi2.ppf(1.0 - alpha, 1)
    for end, censored in (
        (interval.lower, interval.lower_censored),
        (interval.upper, interval.upper_censored),
    ):
        if censored:
            continue
        statistic = 2.0 * n * (result._profile_nll(end) - optimum_nll)
        assert abs(statistic - cutoff) <= 2.0 * abs(_statistic_slope(result, end, n)) * _CI_LOG_XTOL


@pytest.mark.parametrize(
    "row", FIXTURE["estimate_theta_refused"], ids=lambda r: f"{r['case']}-{r['fit_mode']}"
)
def test_estimate_theta_refusals_match_master(row, characterisation_case):
    from superglm import PublicationModeError

    model, X, y = characterisation_case(row["case"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NBThetaBoundWarning)
        with pytest.raises(PublicationModeError, match="theta=1e\\+08"):
            model.estimate_theta(X, y, fit_mode=row["fit_mode"])


def _nb_log_density_50_digits(mpmath, y: float, mu: float, theta: float) -> float:
    """The NB2 log density of the float64 inputs, with 50 significant digits."""
    y, mu, theta = mpmath.mpf(y), mpmath.mpf(mu), mpmath.mpf(theta)
    return float(
        mpmath.loggamma(y + theta)
        - mpmath.loggamma(theta)
        - mpmath.loggamma(y + 1)
        + theta * mpmath.log(theta / (mu + theta))
        + y * mpmath.log(mu / (mu + theta))
    )


@pytest.mark.parametrize("theta", [1e2, 1e6, 1e8])
def test_nb_nll_matches_the_50_digit_density_at_large_theta(theta):
    """Near the Poisson limit the log-gamma pair cancels to O(y log theta) from O(theta log theta).

    The float64 density adds at most six formed terms: betaln(theta, y) or its
    log1p recurrence, log y, y log1p(theta / mu), theta log1p(mu / theta),
    log Gamma(y + 1), y log mu and y log1p(mu / theta). With theta >= mu and
    theta >= y, every one is at most S = y (|log theta| + |log mu| + 2) +
    log Gamma(y + 1) + mu + 1 in size and within 4 eps of S (a rounded ratio,
    log1p or a log-gamma routine, and one product), and each of the five
    additions rounds a partial sum of at most 6 S by eps / 2: 24 + 15 < 64 eps S
    per row. The recurrence's own sum is of terms below y^2 / theta <= y.

    Below the recurrence (theta < eps^-1/2) SciPy's betaln can itself form
    log Gamma(theta) + log Gamma(y) - log Gamma(theta + y), two log-gamma values
    within eps of log Gamma(theta + y) each: 4 eps log Gamma(theta + y) more.
    The weighted mean adds n eps of sum(w |log f|) for its summation. A density
    formed from the log-gamma pair at theta = 1e8 errs by ulp(log Gamma(theta)),
    about 2e-7.
    """
    mpmath = pytest.importorskip("mpmath")
    from superglm.profiling.nb import nb_nll

    rng = np.random.default_rng(5)
    mu = rng.uniform(0.5, 3.0, 400)
    y = rng.poisson(mu).astype(float)
    w = rng.integers(1, 4, 400).astype(float)
    with mpmath.workdps(50):
        exact_rows = np.array(
            [_nb_log_density_50_digits(mpmath, *row, theta) for row in zip(y, mu, strict=True)]
        )
    exact = -np.sum(w * exact_rows) / w.sum()
    size = y * (abs(math.log(theta)) + np.abs(np.log(mu)) + 2.0) + gammaln(y + 1.0) + mu + 1.0
    betaln_rows = (y > 0.0) & (theta < 1.0 / math.sqrt(EPS))
    row_error = 64.0 * EPS * size + np.where(betaln_rows, 4.0 * EPS * gammaln(theta + y), 0.0)
    bound = np.sum(w * row_error) + y.size * EPS * np.sum(w * np.abs(exact_rows))
    bound /= w.sum()
    assert abs(nb_nll(y, mu, w, theta, weight_semantics="frequency") - exact) <= bound


def test_prior_weight_moment_start_solves_the_prior_weight_moment_equation():
    from superglm.profiling.nb import _theta_moment_start

    # w Y ~ NB2(w mu, w theta), so Var(Y) = (mu + mu^2 / theta) / w and
    # d = w (y - mu)^2 - mu has mean mu^2 / theta on every row.
    rng = np.random.default_rng(17)
    n, theta = 40_000, 2.0
    mu = rng.uniform(1.0, 5.0, n)
    w = rng.choice([0.5, 1.0, 4.0], n)
    y = rng.negative_binomial(w * theta, theta / (theta + mu)) / w
    start = _theta_moment_start(y, mu, w, weight_semantics="prior")
    d = w * (y - mu) ** 2 - mu
    # Delta method on the denominator of sum(mu^2) / sum(d).
    standard_error = start * np.std(d) * math.sqrt(n) / np.sum(d)
    assert abs(start - theta) <= 4.0 * standard_error


def test_frequency_moment_start_is_free_of_the_counts_common_scale():
    from superglm.profiling.nb import _theta_moment_start, solve_theta

    # Counts near 1e307 overflow sum(w mu^2) and sum(w ((y - mu)^2 - mu)), whose
    # ratio is the start at unit counts; the score solve only needs signs.
    mu = np.array([5.0, 8.0, 11.0, 14.0])
    y = np.array([0.0, 21.0, 3.0, 30.0])
    unit = _theta_moment_start(y, mu, np.ones(4), weight_semantics="frequency")
    start = _theta_moment_start(y, mu, np.full(4, 1e307), weight_semantics="frequency")
    # w / max w is exactly 1 here, so the two sums are the same floats.
    assert start == unit
    # The score's own sum overflows at these counts; its sign, which is what
    # brackets the root, survives.
    with np.errstate(over="ignore"):
        solved = solve_theta(
            y, mu, np.full(4, 1e307), start, weight_semantics="frequency", bounds=(1e-3, 1e6)
        )
    assert math.isfinite(solved.theta) and not (solved.at_lower or solved.at_upper)
    # A mean past sqrt(max float) makes both sums infinite: that ratio is no start,
    # and the walk begins at the upper bound instead.
    with np.errstate(over="ignore"):
        huge = _theta_moment_start(
            y, np.append(mu[:3], 1e160), np.ones(4), weight_semantics="frequency"
        )
    assert huge == math.inf


def test_estimate_theta_interval_is_the_crossing_at_the_published_mean(characterisation_case):
    from superglm.profiling.nb import nb_nll

    model, X, y = characterisation_case("nb_worst")
    with pytest.warns(UserWarning, match="lies outside its interval"):
        result = model.estimate_theta(X, y, fit_mode="reml", ci_alpha=0.1)
    installed = model._nb_profile_result
    assert result is not installed
    assert installed._ci_cache[0.1] == result._ci_cache[0.1]
    cutoff = chi2.ppf(0.9, 1)
    y = np.asarray(y, dtype=float)
    # The published REML mean's own profile optimum, which the interval is
    # inverted from; theta_hat, estimated at the configured penalty, is not it.
    _, optimum_nll = result._optimum()

    def excess(theta):
        nll = nb_nll(y, model._fit_mu, np.ones_like(y), theta, weight_semantics="prior")
        return 2.0 * y.size * (nll - optimum_nll) - cutoff

    # Each endpoint is the crossing to 1e-6 in log theta: the excess changes
    # sign across a 1e-5 relative neighbourhood of it.
    for end in result.ci(0.1):
        assert excess(end * (1.0 - 1e-5)) * excess(end * (1.0 + 1e-5)) < 0.0
    # Both ends are crossings, but theta_hat lies outside them: a caution.
    assert model.summary(alpha=0.1)._info["nb_theta_ci_status"] == "caution"


def test_estimate_theta_signature_is_the_slim_one():
    import inspect

    from superglm import SuperGLM

    parameters = set(inspect.signature(SuperGLM.estimate_theta).parameters)
    assert {"kwargs", "maxiter", "verbose", "trace_callback"}.isdisjoint(parameters)
    assert {"fit_mode", "theta_bounds", "xatol", "ci_alpha", "progress_callback"} <= parameters


def test_the_reml_alternation_records_each_joint_refit(characterisation_case):
    from superglm.distributions import NegativeBinomial

    model, X, y = characterisation_case("nb_worst")
    model.family = NegativeBinomial("auto")
    model.fit_reml(X, y)
    result = model._nb_profile_result
    # The calibration alternation stops near 0.55; the joint REML refits move
    # theta to the published estimate, and each refit is one more row.
    assert result.evaluations["theta"].iloc[0] < 0.6
    assert result.evaluations["theta"].iloc[-1] == result.theta_hat == model._distribution.theta
    assert result.evaluations["nll"].iloc[-1] == pytest.approx(result.nll, rel=1e-15)
