"""Fixtures for the editor's browser maths: SciPy values and condition numbers for the special
functions, and the predicted laws of the log-normal and gamma families."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from scipy import special, stats

from superglm.distributional.families import GammaLS, LogNormalLS, LowerBoundedLogLink
from superglm.links import LogLink

EPS = float(np.finfo(np.float64).eps)
MATH_FIXTURE = Path(__file__).parent / "editor_frontend" / "fixtures" / "distribution_math.json"
LAWS_FIXTURE = Path(__file__).parent / "editor_frontend" / "fixtures" / "distribution_laws.json"
# The browser tests allow 1e4 * eps * max(1, kappa) relative error; a stored value that moves by
# the same amount under another SciPy build would already make those tests meaningless.
TOLERANCE = 1e4 * EPS

TAILS = [1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.25, 0.4]
PROBABILITIES = sorted([*TAILS, 0.5, *(1.0 - t for t in TAILS)])
GAMMA_TAILS = [1e-9, 1e-6, 1e-3, 1e-2, 0.1, 0.3, 0.5]
GAMMA_SHAPES = [0.04, 0.25, 1.0, 4.0, 30.0, 400.0]
ERFC_POINTS = [
    -6.0, -3.0, -1.0, -0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.46875, 0.5, 1.0, 2.0, 3.0,
    3.999, 4.0, 4.001, 6.0, 10.0, 20.0, 26.0,
]  # fmt: skip
LNGAMMA_POINTS = [
    0.04, 0.25, 0.5, 0.9, 1.01, 1.5, 1.99, 2.5, 4.0, 9.5, 10.0, 30.0, 400.0, 1e4, 1e6,
]  # fmt: skip
NB_MEANS = [0.05, 0.3, 2.0]
NB_THETAS = [0.5, 3.0, 50.0]
NB_COUNTS = range(5)


def _tail_condition(x, density, tail):
    """kappa = |x f(x) / F(x)|: relative change of a tail probability per relative change of x."""
    return float(np.abs(x * density / tail))


def _quantile_condition(x, x_density, tail):
    """kappa_q = tail / |x f(x)|: relative change of a quantile per relative change of the
    probability of its smaller tail, the reciprocal of `_tail_condition`. A quantile of exactly 0
    has no relative condition number; its row demands the exact 0."""
    return 1.0 if x == 0.0 else float(tail / abs(x_density))


def _normal_rows() -> dict:
    z = special.ndtri(np.array(PROBABILITIES))
    density = stats.norm.pdf(z)
    cdf = special.ndtr(z)
    sf = special.ndtr(-z)
    rows = {"norm_cdf": [], "norm_sf": [], "norm_quantile": []}
    for z_i, d_i, c_i, s_i, p_i in zip(z, density, cdf, sf, PROBABILITIES):
        z_i = float(z_i)
        rows["norm_cdf"].append(
            {"z": z_i, "value": float(c_i), "kappa": _tail_condition(z_i, d_i, c_i)}
        )
        rows["norm_sf"].append(
            {"z": z_i, "value": float(s_i), "kappa": _tail_condition(z_i, d_i, s_i)}
        )
        kappa = _quantile_condition(z_i, z_i * d_i, c_i if p_i <= 0.5 else s_i)
        rows["norm_quantile"].append({"p": p_i, "value": z_i, "kappa": kappa})
    return rows


def _erfc_rows() -> list[dict]:
    rows = []
    for x in ERFC_POINTS:
        value = float(special.erfc(x))
        # d erfc / dx = -2 exp(-x^2) / sqrt(pi); erfcx = exp(x^2) erfc stays finite for large x.
        if x > 0.0:
            kappa = x * 2.0 / np.sqrt(np.pi) / float(special.erfcx(x))
        else:
            kappa = abs(x) * 2.0 / np.sqrt(np.pi) * np.exp(-x * x) / value
        rows.append({"x": x, "value": value, "kappa": float(kappa)})
    return rows


def _gamma_x_grid(a: float) -> list[float]:
    """x at the lower and upper `GAMMA_TAILS` quantiles of Gamma(a)."""
    lower = [float(special.gammaincinv(a, t)) for t in GAMMA_TAILS]
    upper = [float(special.gammainccinv(a, t)) for t in GAMMA_TAILS if t < 0.5]
    return lower + upper


def _gamma_x_density(a: float, x: float) -> float:
    """x f(x) = exp(a ln x - x - lnGamma(a)): the numerator of kappa for P and Q."""
    return float(np.exp(a * np.log(x) - x - special.gammaln(a)))


def _gamma_quantile(a: float, p: float) -> float:
    """The quantile of the smaller tail; 1 - p is exact for p >= 1/2."""
    if p <= 0.5:
        return float(special.gammaincinv(a, p))
    return float(special.gammainccinv(a, 1.0 - p))


def _gamma_rows() -> dict:
    rows = {"gamma_p": [], "gamma_q": [], "gamma_quantile": []}
    probabilities = sorted([*GAMMA_TAILS, *(1.0 - t for t in GAMMA_TAILS if t < 0.5)])
    for a in GAMMA_SHAPES:
        for x in _gamma_x_grid(a):
            p_value = float(special.gammainc(a, x))
            q_value = float(special.gammaincc(a, x))
            xf = _gamma_x_density(a, x)
            rows["gamma_p"].append({"a": a, "x": x, "value": p_value, "kappa": xf / p_value})
            rows["gamma_q"].append({"a": a, "x": x, "value": q_value, "kappa": xf / q_value})
        for p in probabilities:
            x = _gamma_quantile(a, p)
            # The quantile inverts the smaller tail; kappa_q is that tail's.
            tail = float(special.gammainc(a, x)) if p <= 0.5 else float(special.gammaincc(a, x))
            kappa = _quantile_condition(x, _gamma_x_density(a, x), tail)
            rows["gamma_quantile"].append({"a": a, "p": p, "value": x, "kappa": kappa})
    return rows


def _lngamma_rows() -> list[dict]:
    rows = []
    for a in LNGAMMA_POINTS:
        value = float(special.gammaln(a))
        kappa = abs(a * float(special.digamma(a)) / value)
        rows.append({"a": a, "value": value, "kappa": kappa})
    return rows


def _nb_rows() -> dict:
    rows = {"nb_pmf": [], "nb_sf": []}
    for mean in NB_MEANS:
        for theta in NB_THETAS:
            law = stats.nbinom(theta, theta / (theta + mean))
            for k in NB_COUNTS:
                pmf = float(law.pmf(k))
                rows["nb_pmf"].append({"k": k, "mean": mean, "theta": theta, "value": pmf})
                # P(Y >= k) = 1 - sum_{j<k} pmf: kappa is the amplification of the subtraction.
                sf = 1.0 if k == 0 else float(law.sf(k - 1))
                kappa = (1.0 - sf) / sf
                rows["nb_sf"].append(
                    {"k": k, "mean": mean, "theta": theta, "value": sf, "kappa": kappa}
                )
    return rows


def _math_content() -> dict:
    return {
        "erfc": _erfc_rows(),
        **_normal_rows(),
        "lngamma": _lngamma_rows(),
        **_gamma_rows(),
        **_nb_rows(),
    }


def test_math_fixture_is_current():
    content = _math_content()
    if os.environ.get("SUPERGLM_WRITE_FIXTURES") == "1":
        MATH_FIXTURE.write_text(json.dumps(content, indent=1) + "\n")
    stored = json.loads(MATH_FIXTURE.read_text())
    assert stored.keys() == content.keys()
    for name, rows in content.items():
        assert len(stored[name]) == len(rows), name
        for old, new in zip(stored[name], rows):
            assert old.keys() == new.keys(), name
            kappa = max(1.0, new.get("kappa", 1.0))
            for field, value in new.items():
                if field in ("kappa", "value"):
                    scale = kappa if field == "value" else 1.0
                    assert old[field] == pytest.approx(value, rel=TOLERANCE * scale, abs=0.0), (
                        name,
                        new,
                        field,
                    )
                else:
                    assert old[field] == pytest.approx(value, rel=TOLERANCE, abs=0.0), (name, new)


# ── predicted laws ──────────────────────────────────────────────────────────────────────────────

LAW_SEED = 20261103
LAW_PROBABILITIES = [1e-12, 1e-6, 1e-2, 0.1, 0.5, 0.9, 1.0 - 1e-2, 1.0 - 1e-6, 1.0 - 1e-12]
DENSITY_Z = [-3.0, -1.0, 0.0, 1.0, 3.0]
LOG_NORMAL_FLOOR = float(LogNormalLS().scale_floor)
# The browser switches the gamma to the Wilson-Hilferty approximation above this shape.
EXACT_SHAPE_LIMIT = 1e4
EXACT_EDGE_SHAPE = 9990.0
LARGE_SHAPES = [1.01e4, 1e5, 1e7]


def _link_spec(link) -> dict:
    if isinstance(link, LowerBoundedLogLink):
        return {"kind": "floored_log", "floor": float(link.floor)}
    if isinstance(link, LogLink):
        return {"kind": "log"}
    raise TypeError(f"no browser link for {type(link).__name__}")


def _exact_specs() -> list[tuple[str, float, float]]:
    """(family, mean, spread): 20 log-normal and 20 gamma rows drawn over a wide range, then
    the edges: a log-normal scale at the floor and at 2.5, a gamma with CV exactly 1 (mode 0) and
    one just inside the exact-shape limit."""
    rng = np.random.default_rng(LAW_SEED)
    means = 10.0 ** rng.uniform(1.0, 5.0, size=40)
    excess = 10.0 ** rng.uniform(-3.0, np.log10(2.5 - LOG_NORMAL_FLOOR), size=20)
    cvs = 10.0 ** rng.uniform(np.log10(0.02), np.log10(3.0), size=20)
    specs = [("lognormal_mean", m, LOG_NORMAL_FLOOR + e) for m, e in zip(means[:20], excess)]
    specs += [("gamma", m, cv) for m, cv in zip(means[20:], cvs)]
    specs += [
        ("lognormal_mean", 1.0e4, LOG_NORMAL_FLOOR + 1e-9),
        ("lognormal_mean", 3.0e2, 2.5),
        ("gamma", 5.0e3, 1.0),
        ("gamma", 2.0e2, 1.0 / np.sqrt(EXACT_EDGE_SHAPE)),
    ]
    return [(name, float(m), float(spread)) for name, m, spread in specs]


def _mp_tails(mp, name, floor, eta, y):
    """(F, S, x f(x)) at y, in 50-digit arithmetic from the link-scale eta."""
    y = mp.mpf(y)
    if name == "lognormal_mean":
        scale = floor + mp.exp(eta[1])
        z = (mp.log(y) - (mp.mpf(eta[0]) - scale * scale / 2)) / scale
        return mp.ncdf(z), mp.ncdf(-z), mp.npdf(z) / scale
    shape = mp.exp(-2 * mp.mpf(eta[1]))
    x = y * shape / mp.exp(eta[0])
    density = mp.exp(shape * mp.log(x) - x - mp.loggamma(shape))
    lower = mp.gammainc(shape, 0, x, regularized=True)
    return lower, mp.gammainc(shape, x, mp.inf, regularized=True), density


def _mp_log10_density(mp, name, floor, log10_y, e0, e1):
    """log10 of the density of log10 Y: ln(ln 10 * y f_Y(y)) / ln 10."""
    ln10 = mp.log(10)
    if name == "lognormal_mean":
        scale = floor + mp.exp(e1)
        z = (log10_y * ln10 - (e0 - scale * scale / 2)) / scale
        return (mp.log(ln10 / scale) - mp.log(2 * mp.pi) / 2 - z * z / 2) / ln10
    shape = mp.exp(-2 * e1)
    log_scaled = log10_y * ln10 - (e0 + 2 * e1)  # ln (y / theta)
    return (mp.log(ln10) + shape * log_scaled - mp.exp(log_scaled) - mp.loggamma(shape)) / ln10


def _mp_density_condition(mp, name, floor, log10_y, eta):
    """sum_i |a_i d v / d a_i| over the inputs a = (log10 y, eta0, eta1) of v = log10 density: the
    absolute error that a relative rounding of each input costs."""
    args = [mp.mpf(log10_y), mp.mpf(eta[0]), mp.mpf(eta[1])]
    total = mp.mpf(0)
    for i, arg in enumerate(args):

        def along(value, i=i):
            moved = [*args]
            moved[i] = value
            return _mp_log10_density(mp, name, floor, *moved)

        total += abs(arg * mp.diff(along, arg))
    return total


def _mp_mode(mp, name, floor, eta):
    """The mode by the algebra mean * exp(-1.5 s^2) or mean * (1 - CV^2), which differs from the
    browser's mu - s^2 and (k - 1) theta routes; and its condition number."""
    mean = mp.exp(eta[0])
    if name == "lognormal_mean":
        scale = floor + mp.exp(eta[1])
        return mean * mp.exp(-1.5 * scale * scale), 1 + abs(mp.mpf(eta[0])) + 3 * scale * scale
    cv2 = mp.exp(2 * mp.mpf(eta[1]))
    if cv2 >= 1:
        return mp.mpf(0), mp.mpf(1)
    return mean * (1 - cv2), 1 / (1 - cv2)


def _assert_family_agrees(label, got, tail, kappa_tail):
    """The family's value, as a tail probability, equals the 50-digit tail to the fixture bound."""
    bound = TOLERANCE * max(1.0, kappa_tail) * tail
    assert abs(got - tail) <= bound, (label, got, tail, bound)


def _law_row(mp, name: str, mean: float, spread: float, *, with_tails: bool) -> dict:
    family = LogNormalLS() if name == "lognormal_mean" else GammaLS()
    links = [spec.default_link for spec in family.parameters]
    natural = np.array([mean, spread])
    eta = [float(links[j].link(natural[j : j + 1])[0]) for j in range(2)]
    # The family sees the natural parameters its links give back, as a fitted model's do.
    theta = np.array([[float(links[j].inverse(np.array([eta[j]]))[0]) for j in range(2)]])
    specs = [_link_spec(link) for link in links]
    floor = specs[1].get("floor", 0.0)
    mode, mode_kappa = _mp_mode(mp, name, floor, eta)
    row = {
        "family": name,
        "links": specs,
        "natural": theta[0].tolist(),
        "eta": eta,
        "mean": {"value": float(family.default_prediction(theta)[0]), "kappa": 1.0},
        "mode": {"value": float(mode), "kappa": float(mode_kappa)},
    }
    if with_tails:
        row.update(_law_tails(mp, family, name, floor, eta, theta))
    row["log10_density"] = []
    for point in _density_points(mp, name, floor, eta):
        value = _mp_log10_density(mp, name, floor, mp.mpf(point), mp.mpf(eta[0]), mp.mpf(eta[1]))
        condition = _mp_density_condition(mp, name, floor, point, eta)
        row["log10_density"].append(
            {"l": point, "value": float(value), "condition": float(condition)}
        )
    return row


def _density_points(mp, name, floor, eta) -> list[float]:
    """log10 y at the log-scale mean plus DENSITY_Z log-scale standard deviations (the log-normal
    scale; the gamma's CV, capped at 1), in 50-digit arithmetic so no platform moves them."""
    spread = floor + mp.exp(eta[1]) if name == "lognormal_mean" else min(mp.exp(eta[1]), 1)
    return [float((mp.mpf(eta[0]) + z * spread) / mp.log(10)) for z in DENSITY_Z]


def _law_tails(mp, family, name, floor, eta, theta) -> dict:
    """cdf, sf and quantile rows at the family's quantiles of LAW_PROBABILITIES. The evaluation
    points are those quantiles to 10 significant digits, so that the last digits of SciPy's
    inversion do not move them; the family's value is checked against the 50-digit tail."""
    grid = np.array(LAW_PROBABILITIES)
    repeated = np.repeat(theta, len(grid), axis=0)
    quantiles = np.asarray(family.quantile(grid, repeated))
    points = np.array([float(f"{q:.10g}") for q in quantiles])
    cdf = np.asarray(family.cdf(points, repeated))
    rows = {"cdf": [], "sf": [], "quantile": []}
    for i, p in enumerate(LAW_PROBABILITIES):
        lower, upper, x_density = _mp_tails(mp, name, floor, eta, points[i])
        exact_lower, exact_upper, exact_density = _mp_tails(mp, name, floor, eta, quantiles[i])
        small_tail = exact_lower if p <= 0.5 else exact_upper
        target = p if p <= 0.5 else 1.0 - p
        label = (name, eta, p)
        _assert_family_agrees(
            ("quantile", *label), float(small_tail), target, exact_density / small_tail
        )
        _assert_family_agrees(("cdf", *label), float(cdf[i]), float(lower), x_density / lower)
        tail = lower if p <= 0.5 else upper
        rows["cdf"].append(
            {"y": float(points[i]), "value": float(cdf[i]), "kappa": float(x_density / lower)}
        )
        rows["sf"].append(
            {"y": float(points[i]), "value": float(upper), "kappa": float(x_density / upper)}
        )
        rows["quantile"].append(
            {
                "p": p,
                "value": float(quantiles[i]),
                "kappa": _quantile_condition(points[i], x_density, tail),
            }
        )
    return rows


def _laws_content() -> dict:
    mp = pytest.importorskip("mpmath")
    with mp.workdps(50):
        rows = [_law_row(mp, *spec, with_tails=True) for spec in _exact_specs()]
        large = [
            _law_row(mp, "gamma", mean, 1.0 / np.sqrt(shape), with_tails=False)
            for mean, shape in zip([1.5e3, 1.0e3, 1.0e2], LARGE_SHAPES)
        ]
    return {
        "exact_shape_limit": EXACT_SHAPE_LIMIT,
        "probabilities": LAW_PROBABILITIES,
        "rows": rows,
        "large_shape_rows": large,
    }


def _assert_current(old, new, where: str = "") -> None:
    """Stored fixture equals the regenerated one: a `value` to TOLERANCE * max(1, kappa), every
    other number to TOLERANCE, everything else exactly."""
    if isinstance(new, dict):
        assert isinstance(old, dict) and old.keys() == new.keys(), where
        scale = max(1.0, new.get("kappa", 1.0))
        for key, value in new.items():
            if key == "value":
                assert old[key] == pytest.approx(value, rel=TOLERANCE * scale, abs=0.0), (
                    where,
                    key,
                )
            else:
                _assert_current(old[key], value, f"{where}/{key}")
    elif isinstance(new, list):
        assert isinstance(old, list) and len(old) == len(new), where
        for i, (old_item, new_item) in enumerate(zip(old, new)):
            _assert_current(old_item, new_item, f"{where}[{i}]")
    elif isinstance(new, str):
        assert old == new, where
    else:
        assert old == pytest.approx(new, rel=TOLERANCE, abs=0.0), where


def test_laws_fixture_is_current():
    content = _laws_content()
    if os.environ.get("SUPERGLM_WRITE_FIXTURES") == "1":
        LAWS_FIXTURE.write_text(json.dumps(content, indent=1) + "\n")
    _assert_current(json.loads(LAWS_FIXTURE.read_text()), content)
