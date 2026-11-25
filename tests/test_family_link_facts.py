"""Truth table of what generic code decides from a family, a link, or the two together.

Every fact a fit, a REML search, inference or a report reads off the family and
the link is written out here once, as the set of shipped classes it holds for,
and checked on every shipped family and link (parameterised ones at several
parameters), an empty subclass of each, and user-defined objects with and
without the optional hooks.

Two kinds of fact are kept apart, because the library keeps them apart:

- *exact* facts hold for the shipped class itself and for nothing else. A
  subclass may change a likelihood or an inverse link, so it never inherits a
  closed form, a curvature claim or a cache key;
- *inherited* facts (a mean's range, a response's support, a predictor's clip,
  the dispersion's profile) hold for a subclass too, and a user-defined object
  that declares nothing takes the generic default.

``PROBES`` names where each fact is read. A change that moves a fact re-points
its probe; the expectations do not change.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

import superglm.links as links_module
from superglm._family_link import pair_of
from superglm._utils import contract_weights
from superglm._weight_units import _fisher_weight_power_bound
from superglm.diagnostics.fit_cost import exact_reml_cost_message
from superglm.diagnostics.separation import response_boundaries
from superglm.distributions import (
    Binomial,
    Gamma,
    Gaussian,
    NegativeBinomial,
    Poisson,
    Tweedie,
    clip_mu,
    initial_mean,
    scale_is_known,
    validate_response,
)
from superglm.inference.metrics import ModelMetrics
from superglm.links import (
    CauchitLink,
    CloglogLink,
    IdentityLink,
    InverseLink,
    InverseSquaredLink,
    LogitLink,
    LogLink,
    NegativeBinomialLink,
    PowerLink,
    ProbitLink,
    SqrtLink,
    declared_eta_domain,
    is_multiplicative,
    stabilize_eta,
)
from superglm.model.fit_ops import _log_power_null_intercept, _null_curvature_kind
from superglm.reml.observed_geometry import (
    classify_binned_reml_curvature,
    classify_reml_curvature,
)
from superglm.reml.scale import (
    GammaScaleProfileData,
    GaussianScaleProfile,
    TweedieScaleProfile,
    prepare_reml_scale_data,
)
from superglm.solvers.dispersion import family_default_weight_semantics
from superglm.solvers.irls_state import (
    _irls_objective_scale,
    mean_space_boundary_rows,
    mean_space_clipped_rows,
)
from superglm.solvers.working_rows import (
    coefficient_initial_intercept,
)

# ── The objects ──────────────────────────────────────────────────


@dataclass(frozen=True)
class Case:
    """One family or link: the shipped class it is exactly, and the one it inherits from."""

    label: str
    obj: object
    exact: str | None
    lineage: str | None


class BareFamily:
    """A user family with the required members only, and no ``scale_known``."""

    default_link = "log"

    def variance(self, mu):
        return mu.copy()

    def deviance_unit(self, y, mu):
        return (y - mu) ** 2

    def log_likelihood(self, y, mu, weights, phi=1.0):
        return 0.0


class HookedFamily(BareFamily):
    """A user family that declares every optional hook."""

    scale_known = False

    def variance_derivative(self, mu):
        return np.ones_like(mu)

    def variance_second_derivative(self, mu):
        return np.zeros_like(mu)

    def variance_third_derivative(self, mu):
        return np.zeros_like(mu)

    def reml_curvature(self, link):
        return "observed"

    def validate_response(self, y):
        if np.any(y > 100.0):
            raise ValueError("HookedFamily requires y <= 100")


class BareLink:
    """A user link with the required methods only."""

    def link(self, mu):
        return np.log(mu)

    def inverse(self, eta):
        return np.exp(eta)

    def deriv(self, mu):
        return 1.0 / mu

    def deriv_inverse(self, eta):
        return np.exp(eta)


class HookedLink(BareLink):
    """A user link that declares every optional hook."""

    eta_domain = (0.0, np.inf)

    def deriv2_inverse(self, eta):
        return np.exp(eta)

    def deriv3_inverse(self, eta):
        return np.exp(eta)

    def deriv4_inverse(self, eta):
        return np.exp(eta)

    def reml_curvature(self, distribution):
        return "observed"


def _empty_subclass(obj):
    klass = type(f"User{type(obj).__name__}", (type(obj),), {})
    twin = klass.__new__(klass)
    twin.__dict__.update(vars(obj))
    return twin


def _cases(shipped: dict[str, tuple[str, object]], custom: dict[str, object]) -> list[Case]:
    cases = [Case(label, obj, kind, kind) for label, (kind, obj) in shipped.items()]
    cases += [
        Case(f"sub({label})", _empty_subclass(obj), None, kind)
        for label, (kind, obj) in shipped.items()
    ]
    return cases + [Case(label, obj, None, None) for label, obj in custom.items()]


FAMILIES = _cases(
    {
        "gaussian": ("gaussian", Gaussian()),
        "poisson": ("poisson", Poisson()),
        "gamma": ("gamma", Gamma()),
        "binomial": ("binomial", Binomial()),
        "nb(1.3)": ("nb", NegativeBinomial(theta=1.3)),
        "nb(auto)": ("nb", NegativeBinomial(theta="auto")),
        "tweedie(1.5)": ("tweedie", Tweedie(p=1.5)),
        "tweedie(1.25)": ("tweedie", Tweedie(p=1.25)),
    },
    {"bare family": BareFamily(), "hooked family": HookedFamily()},
)
LINKS = _cases(
    {
        "identity": ("identity", IdentityLink()),
        "log": ("log", LogLink()),
        "logit": ("logit", LogitLink()),
        "probit": ("probit", ProbitLink()),
        "cloglog": ("cloglog", CloglogLink()),
        "cauchit": ("cauchit", CauchitLink()),
        "inverse": ("inverse", InverseLink()),
        "inverse_squared": ("inverse_squared", InverseSquaredLink()),
        "sqrt": ("sqrt", SqrtLink()),
        "power(1)": ("power", PowerLink(1.0)),
        "power(-1)": ("power", PowerLink(-1.0)),
        "power(-2)": ("power", PowerLink(-2.0)),
        "power(-0.5)": ("power", PowerLink(-0.5)),
        "power(-0.25)": ("power", PowerLink(-0.25)),
        "power(0.5)": ("power", PowerLink(0.5)),
        "nb_link(1.3)": ("nb_link", NegativeBinomialLink(1.3)),
        "nb_link(2)": ("nb_link", NegativeBinomialLink(2.0)),
    },
    {"bare link": BareLink(), "hooked link": HookedLink()},
)
PAIRS = list(itertools.product(FAMILIES, LINKS))

_PROBE_Y = np.array([1.0, 2.0])
_PROBE_W = np.ones(2)

# ── Where each fact is read today ────────────────────────────────


def _raises(call, *args, **kwargs) -> str | None:
    try:
        call(*args, **kwargs)
    except (ValueError, NotImplementedError) as error:
        return str(error)
    return None


def _reml_curvature(family, link) -> str:
    try:
        return classify_reml_curvature(family, link)
    except NotImplementedError:
        return "refused"


def _scale_profile(family) -> str | None:
    profile = prepare_reml_scale_data(family, _PROBE_Y, _PROBE_W, weight_semantics="frequency")
    names = {
        GaussianScaleProfile: "gaussian",
        GammaScaleProfileData: "gamma",
        TweedieScaleProfile: "tweedie",
    }
    return None if profile is None else names[type(profile)]


def _domain_rows(family, link) -> tuple[bool, ...] | None:
    rows = pair_of(family, link).link_domain_rows
    return None if rows is None else tuple(bool(r) for r in rows(np.array([-1.0, 0.5, 2.0])))


def _zero_start(family, link) -> float:
    with np.errstate(divide="ignore"):
        return coefficient_initial_intercept(
            distribution=family, link=link, y=np.zeros(2), sample_weight=_PROBE_W
        )


FAMILY_PROBES = {
    "mean_clip": lambda f: tuple(clip_mu(np.array([-1.0, 0.5, 2.0]), f)),
    "start_floor": lambda f: float(initial_mean(np.zeros(2), _PROBE_W, f)),
    "start_cap": lambda f: float(initial_mean(np.ones(2), _PROBE_W, f)),
    "response_refusal": lambda f: tuple(
        _raises(validate_response, np.array([y]), f) for y in (-1.0, 0.0, 0.5, 1000.0)
    ),
    "zero_prior_weight_refused": lambda f: (
        _raises(contract_weights, np.array([0.0, 1.0]), 2, family=f, weight_semantics="prior")
        is not None
    ),
    "zero_frequency_weight_refused": lambda f: (
        _raises(contract_weights, np.array([0.0, 1.0]), 2, family=f, weight_semantics="frequency")
        is not None
    ),
    "legacy_weight_semantics": family_default_weight_semantics,
    "scale_known": scale_is_known,
    "scale_known_in_metrics": lambda f: ModelMetrics._known_scale.fget(SimpleNamespace(_family=f)),
    "scale_profile": _scale_profile,
}
LINK_PROBES = {
    "eta_clip": lambda k: tuple(float(v) for v in stabilize_eta(np.array([-1e300, 0.0, 1e300]), k)),
    "multiplicative": is_multiplicative,
    "eta_domain": declared_eta_domain,
}
PAIR_PROBES = {
    "constant_fisher_weight": lambda f, k: pair_of(f, k).constant_fisher_weight is not None,
    "observed_newton": lambda f, k: pair_of(f, k).observed_newton,
    "observed_rows_concave": lambda f, k: pair_of(f, k).observed_rows_concave,
    "log_link_variance_power": lambda f, k: pair_of(f, k).log_link_variance_power,
    "presolve_eligible": lambda f, k: pair_of(f, k).presolve_eligible,
    "fisher_equals_observed": lambda f, k: pair_of(f, k).fisher_equals_observed,
    "reml_curvature": _reml_curvature,
    "binned_reml_curvature": classify_binned_reml_curvature,
    "deviance_convex_in_eta": lambda f, k: pair_of(f, k).deviance_convex_in_eta,
    "null_curvature_kind": _null_curvature_kind,
    "null_closed_form": lambda f, k: (
        _log_power_null_intercept(_PROBE_Y, _PROBE_W, np.array([0.0, 0.1]), f, k) is not None
    ),
    "linear_model": lambda f, k: (
        exact_reml_cost_message(n_rows=10**7, n_coef=10**3, distribution=f, link=k) is None
    ),
    "objective_scale": lambda f, k: _irls_objective_scale(
        y=np.array([2.0, 3.0]), weights=_PROBE_W, family=f, link=k
    ),
    "zero_start": _zero_start,
    "weight_power_bound": _fisher_weight_power_bound,
    "mean_space_bounded_by_link": lambda f, k: pair_of(f, k).mean_space_violation is not None,
    "link_domain_guarded": lambda f, k: pair_of(f, k).link_domain_violation is not None,
    "link_domain_rows": _domain_rows,
    "mean_space_boundary_rows": lambda f, k: mean_space_boundary_rows(
        f, k, np.array([-1e-9, -3.0, 0.5, 0.25]), np.ones(4)
    ),
    "mean_space_clipped_rows": lambda f, k: mean_space_clipped_rows(
        f, k, np.array([-1e-9, -30.0, -1.0]), np.ones(3)
    ),
    "clip_can_hold_against": lambda f, k: pair_of(f, k).clip_can_hold_against,
    "tail_rows": lambda f, k: pair_of(f, k).tail_rows,
    "response_boundaries": response_boundaries,
}

# ── What each fact is, written out ───────────────────────────────

CANONICAL = {("gaussian", "identity"), ("poisson", "log"), ("binomial", "logit"), ("gamma", "inverse")}  # fmt: skip
CONSTANT_FISHER_WEIGHT = {("gaussian", "identity"), ("gamma", "log"), ("poisson", "sqrt")}
OBSERVED_NEWTON = {("gamma", "log"), ("tweedie", "log")}
CONCAVE_OBSERVED_ROWS = OBSERVED_NEWTON | {("nb", "log"), ("binomial", "probit"), ("binomial", "cloglog")}  # fmt: skip
PRESOLVE = CONCAVE_OBSERVED_ROWS | {("poisson", "log"), ("binomial", "logit")}
CONVEX_DEVIANCE = {
    ("poisson", "log"), ("poisson", "identity"), ("gamma", "inverse"), ("gamma", "log"),
    ("nb", "log"), ("tweedie", "log"), ("binomial", "logit"), ("binomial", "probit"),
    ("binomial", "cloglog"), ("binomial", "log"),
}  # fmt: skip
TAIL_ROWS = {("binomial", "logit"), ("binomial", "probit"), ("binomial", "cloglog")}
LOG_VARIANCE_POWER = {"gaussian": 0.0, "poisson": 1.0, "gamma": 2.0}
BINARY_LINKS = {"logit", "probit", "cloglog", "cauchit"}
ESTIMATED_SCALE = {"gaussian", "gamma", "tweedie"}
RESPONSE_REFUSALS = {
    # at y = -1, 0, 0.5 and 1000; a subclass keeps its parent's rule and wording
    "poisson": ("Poisson family requires nonnegative y", None, None, None),
    "nb": ("NegativeBinomial family requires nonnegative y", None, None, None),
    "tweedie": ("Tweedie family requires nonnegative y", None, None, None),
    "gamma": ("Gamma family requires strictly positive y",) * 2 + (None, None),
    "gaussian": (None, None, None, None),
}
# 2 * (binary exponent of the largest |dmu/deta| on the link's clip) + 1
LINK_SLOPE_POWER = {
    "identity": 3, "log": 233, "logit": -1, "probit": -1, "cloglog": -1, "cauchit": -1,
    "inverse": 161, "inverse_squared": 119, "sqrt": 1025, "power(1)": 3, "power(-1)": 161,
    "power(-2)": 119, "power(-0.5)": 243, "power(-0.25)": 405, "power(0.5)": 83,
    "nb_link(1.3)": 135, "nb_link(2)": 137,
}  # fmt: skip
LOG_LINK_WEIGHT_POWER = {
    "gaussian": 232, "poisson": 117, "nb(1.3)": 2, "tweedie(1.5)": 59, "tweedie(1.25)": 88
}  # fmt: skip
ETA_CLIP = {
    "identity": (-1e300, 0.0, 1e300),
    "log": (-80.0, 0.0, 80.0),
    "logit": (-20.0, 0.0, 20.0),
    "probit": (-20.0, 0.0, 20.0),
    "cloglog": (-20.0, 0.0, links_module._CLOGLOG_LINK_ETA_MAX),
    "cauchit": (-links_module._CAUCHIT_LINK_ETA_MAX, 0.0, links_module._CAUCHIT_LINK_ETA_MAX),
    "inverse": (1e-12, 1e-12, 1e12),
    "inverse_squared": (1e-12, 1e-12, 1e12),
    "sqrt": (-links_module._SQRT_LINK_ETA_MAX, 0.0, links_module._SQRT_LINK_ETA_MAX),
    "power": (1e-12, 1e-12, 1e12),
    "nb_link": (-30.0, -1e-10, -1e-10),
    None: (-20.0, 0.0, 20.0),
}


def _exact(family: Case, link: Case) -> tuple[str | None, str | None]:
    return family.exact, link.exact


def _matches(left: float, right: float) -> bool:
    return abs(left - right) <= 1e-12


def _canonical(family: Case, link: Case) -> bool:
    """Observed information equals Fisher's: a canonical link, by exact class and parameter."""
    if family.exact is None or link.exact is None:
        return False
    if _exact(family, link) in CANONICAL:
        return True
    if link.exact == "power":
        target = {"gaussian": 1.0, "gamma": -1.0}.get(family.exact)
        if family.exact == "tweedie":
            target = 1.0 - family.obj.p
        return target is not None and _matches(link.obj.power, target)
    if _exact(family, link) == ("nb", "nb_link"):
        return family.obj.theta != "auto" and _matches(family.obj.theta, link.obj.theta)
    return False


def _declares_curvature(case: Case) -> bool:
    return case.label.startswith("hooked")


def _expected_reml_curvature(family: Case, link: Case) -> str:
    if family.exact is not None and link.exact is not None:
        return "fisher" if _canonical(family, link) else "observed"
    return "observed" if _declares_curvature(family) or _declares_curvature(link) else "refused"


def _expected_binned_curvature(family: Case, link: Case) -> str:
    concave = _exact(family, link) in CONCAVE_OBSERVED_ROWS
    return "observed" if concave and not _canonical(family, link) else "fisher"


def _expected_log_variance_power(family: Case, link: Case) -> float | None:
    if link.exact != "log":
        return None
    return family.obj.p if family.exact == "tweedie" else LOG_VARIANCE_POWER.get(family.exact)


def _expected_null_kind(family: Case, link: Case) -> str | None:
    """The null model reads the family by inheritance and the link by exact class."""
    if (family.lineage, link.exact) in CANONICAL:
        return "canonical"
    if family.lineage in ("gamma", "tweedie", "nb") and link.exact == "log":
        return "convex"
    return None


def _has_eta_domain(link: Case) -> bool:
    negative_power = link.lineage == "power" and link.obj.power < 0.0
    return (
        link.lineage in ("inverse", "inverse_squared")
        or negative_power
        or (link.label == "hooked link")
    )


def _expected_domain_rows(family: Case, link: Case) -> tuple[bool, ...] | None:
    """Rows outside the link's domain at eta = -1, 0.5 and 2."""
    if _has_eta_domain(link):
        return (True, False, False)
    if link.lineage != "identity" or family.lineage == "gaussian":
        return None
    return (True, False, family.lineage == "binomial")


def _expected_boundaries(family: Case, link: Case) -> tuple[str, ...]:
    if family.lineage == "binomial":
        if link.lineage in BINARY_LINKS:
            return ("zero", "one")
        return ("zero",) if link.lineage == "log" else ()
    mass_at_zero = family.lineage in ("poisson", "nb", "tweedie")
    return ("zero",) if mass_at_zero and link.lineage == "log" else ()


def _expected_weight_power(family: Case, link: Case) -> int | None:
    if _exact(family, link) in CONSTANT_FISHER_WEIGHT:
        return 2 if family.exact == "poisson" else 0
    if link.exact is None:
        return None
    if link.exact == "log" and family.label in LOG_LINK_WEIGHT_POWER:
        return LOG_LINK_WEIGHT_POWER[family.label]
    floor = 0 if family.exact == "gaussian" else 333
    return max(0, LINK_SLOPE_POWER[link.label] + floor)


def _expected_zero_start(family: Case, link: Case) -> float:
    if _exact(family, link) == ("poisson", "sqrt"):
        return 0.0
    floor = {"gaussian": 0.0, "binomial": 1e-3}.get(family.lineage, 1e-12)
    with np.errstate(divide="ignore"):
        return float(link.obj.link(np.atleast_1d(floor))[0])


def _expected_refusals(family: Case) -> tuple[str | None, ...]:
    if family.lineage == "binomial":
        return tuple(
            None
            if y in (0.0, 1.0)
            else "Binomial family requires y in {0, 1}, but found 1 invalid values "
            f"(e.g. {np.array([y])})."
            for y in (-1.0, 0.0, 0.5, 1000.0)
        )
    if family.label == "hooked family":
        return (None, None, None, "HookedFamily requires y <= 100")
    return RESPONSE_REFUSALS.get(family.lineage, (None, None, None, None))


FAMILY_FACTS = {
    "mean_clip": lambda f: {
        "gaussian": (-1.0, 0.5, 2.0),
        "binomial": (1e-7, 0.5, 1.0 - 1e-7),
    }.get(f.lineage, (1e-50, 0.5, 2.0)),
    "start_floor": lambda f: {"gaussian": 0.0, "binomial": 1e-3}.get(f.lineage, 1e-12),
    "start_cap": lambda f: 1.0 - 1e-3 if f.lineage == "binomial" else 1.0,
    "response_refusal": _expected_refusals,
    "zero_prior_weight_refused": lambda f: f.lineage == "tweedie",
    "zero_frequency_weight_refused": lambda f: False,
    "legacy_weight_semantics": lambda f: "prior" if f.lineage == "tweedie" else "frequency",
    # a family that declares nothing fits as scale-known and reports as estimated
    "scale_known": lambda f: f.lineage not in ESTIMATED_SCALE and f.label != "hooked family",
    "scale_known_in_metrics": lambda f: f.lineage in ("poisson", "nb", "binomial"),
    "scale_profile": lambda f: f.lineage if f.lineage in ESTIMATED_SCALE else None,
}
LINK_FACTS = {
    "eta_clip": lambda k: (-1e300, 0.0, 1e300) if "power(1)" in k.label else ETA_CLIP[k.lineage],
    "multiplicative": lambda k: k.lineage == "log",
    "eta_domain": lambda k: (0.0, np.inf) if _has_eta_domain(k) else None,
}
PAIR_FACTS = {
    "constant_fisher_weight": lambda f, k: _exact(f, k) in CONSTANT_FISHER_WEIGHT,
    "observed_newton": lambda f, k: _exact(f, k) in OBSERVED_NEWTON,
    "observed_rows_concave": lambda f, k: _exact(f, k) in CONCAVE_OBSERVED_ROWS,
    "log_link_variance_power": _expected_log_variance_power,
    "presolve_eligible": lambda f, k: _exact(f, k) in PRESOLVE,
    "fisher_equals_observed": _canonical,
    "reml_curvature": _expected_reml_curvature,
    "binned_reml_curvature": _expected_binned_curvature,
    "deviance_convex_in_eta": lambda f, k: _exact(f, k) in CONVEX_DEVIANCE,
    "null_curvature_kind": _expected_null_kind,
    "null_closed_form": lambda f, k: (
        k.exact == "log" and f.lineage in ("poisson", "gamma", "tweedie")
    ),
    "linear_model": lambda f, k: _exact(f, k) == ("gaussian", "identity"),
    "objective_scale": lambda f, k: 5.0 if _exact(f, k) == ("poisson", "sqrt") else 1.0,
    "zero_start": _expected_zero_start,
    "weight_power_bound": _expected_weight_power,
    "mean_space_bounded_by_link": lambda f, k: (f.lineage, k.lineage) == ("binomial", "log"),
    "link_domain_guarded": lambda f, k: _expected_domain_rows(f, k) is not None,
    "link_domain_rows": _expected_domain_rows,
    "mean_space_boundary_rows": lambda f, k: (
        {"log": 3, "identity": 2}.get(k.lineage, 0) if f.lineage == "binomial" else 0
    ),
    "mean_space_clipped_rows": lambda f, k: (
        2 if (f.lineage, k.lineage) == ("binomial", "log") else 0
    ),
    "clip_can_hold_against": lambda f, k: (
        f.lineage == "binomial" and k.lineage not in ("log", "identity")
    ),
    "tail_rows": lambda f, k: _exact(f, k) in TAIL_ROWS,
    "response_boundaries": _expected_boundaries,
}
# Facts whose probe evaluates a theta the family has not estimated yet.
_NEEDS_THETA = {"weight_power_bound", "zero_start"}


def _same(found, expected) -> bool:
    if isinstance(found, np.generic):
        found = found.item()
    if isinstance(expected, float) and isinstance(found, float):
        return found == expected or bool(np.isclose(found, expected, rtol=1e-12, atol=0.0))
    return type(found) is type(expected) and found == expected


# ── The table ────────────────────────────────────────────────────


@pytest.mark.pins_behaviour
@pytest.mark.parametrize("fact", sorted(FAMILY_FACTS))
def test_family_facts_hold_for_every_family(fact):
    """Pins what each family-only decision answers today, before the families declare it.

    Every shipped family, an empty subclass of each (which keeps its parent's
    answer) and two user-defined families (which take the generic default).
    """
    wrong = [
        f"{family.label}: found {FAMILY_PROBES[fact](family.obj)!r}, "
        f"expected {FAMILY_FACTS[fact](family)!r}"
        for family in FAMILIES
        if not _same(FAMILY_PROBES[fact](family.obj), FAMILY_FACTS[fact](family))
    ]
    assert not wrong, "\n".join(wrong)


@pytest.mark.pins_behaviour
@pytest.mark.parametrize("fact", sorted(LINK_FACTS))
def test_link_facts_hold_for_every_link(fact):
    """Pins what each link-only decision answers today, before the links declare it.

    Every shipped link, an empty subclass of each (which keeps its parent's
    answer) and two user-defined links (which take the generic default).
    """
    wrong = [
        f"{link.label}: found {LINK_PROBES[fact](link.obj)!r}, expected {LINK_FACTS[fact](link)!r}"
        for link in LINKS
        if not _same(LINK_PROBES[fact](link.obj), LINK_FACTS[fact](link))
    ]
    assert not wrong, "\n".join(wrong)


@pytest.mark.pins_behaviour
@pytest.mark.parametrize("fact", sorted(PAIR_FACTS))
def test_pair_facts_hold_for_every_family_and_link(fact):
    """Pins what each family-and-link decision answers today, before one object resolves it.

    Every family against every link: 612 pairs. An exact fact is false for any
    subclass or user-defined object; an inherited one follows the parent class.
    """
    wrong = []
    for family, link in PAIRS:
        if fact in _NEEDS_THETA and getattr(family.obj, "theta", None) == "auto":
            continue
        found = PAIR_PROBES[fact](family.obj, link.obj)
        expected = PAIR_FACTS[fact](family, link)
        if not _same(found, expected):
            wrong.append(f"{family.label} / {link.label}: found {found!r}, expected {expected!r}")
    assert not wrong, "\n".join(wrong[:40])


@pytest.mark.pins_behaviour
def test_resolving_a_fact_never_raises_for_user_objects():
    """Pins that only the REML curvature refuses an undeclared pair, and only when asked."""
    for family, link in PAIRS:
        if family.lineage is not None and link.lineage is not None:
            continue
        for fact, probe in PAIR_PROBES.items():
            if fact in _NEEDS_THETA and getattr(family.obj, "theta", None) == "auto":
                continue
            probe(family.obj, link.obj)
    with pytest.raises(NotImplementedError, match="reml_curvature"):
        classify_reml_curvature(BareFamily(), LogLink())
