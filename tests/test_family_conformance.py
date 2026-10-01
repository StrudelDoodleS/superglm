"""Family conformance (one-engine design §9, test T5; stage 1, §3.3 and §3.11).

A family provides rows, the derivatives the exact REML Hessian needs, a
diagonal ``W``, deviance and dispersion; no pair is routed by an audit table.
Parametrised from the registry (every built-in REML family, parameterised
ones at two parameters, times every built-in link), this module checks

- the row contract: finite observed rows and an error scale ``e >= |w|``
  (``observed_row_error_scale``), ``e = w`` on the closed-form log-link rows,
  non-negative Fisher rows, and observed rows equal to Fisher rows wherever
  the pair's curvature is classified Fisher;
- the dispatch: the structured decision a REML fit takes, recorded where the
  fit makes it, is one decision for every pair, and invariant under weights
  times 1e6, a numeric column offset by 1e8, a random-effect penalty fixed at
  1e-7 and a duplicated column (T1 for the signed classes);
- a mini trigger battery: the observed REML geometry of the signed classes and
  a custom family, built on the chain from the family's own rows at a
  penalized mode, against the exact rational ``H`` of the same rows;
- the custom-family protocol: a conforming custom family with signed observed
  rows fits on the chain and matches the dense backend; one without a
  curvature declaration is refused before any fitting work.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import superglm.reml.direct as direct_module
import superglm.reml.discrete as discrete_module
from superglm import Categorical, LambdaPolicy, Numeric, RandomEffect, SuperGLM
from superglm.distributions import (
    Binomial,
    Gamma,
    Gaussian,
    NegativeBinomial,
    Poisson,
    Tweedie,
    clip_mu,
)
from superglm.group_matrix import (
    CategoricalGroupMatrix,
    DenseGroupMatrix,
    DesignMatrix,
    RandomEffectGroupMatrix,
)
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
    stabilize_eta,
)
from superglm.reml.observed_geometry import (
    _BUILTIN_REML_DISTRIBUTIONS,
    _BUILTIN_REML_LINKS,
    ObservedGeometryInfeasibleError,
    build_observed_reml_geometry,
    classify_reml_curvature,
    compute_observed_information_weights,
    observed_row_error_scale,
)
from superglm.reml.penalty_algebra import build_penalty_matrix
from superglm.solvers.irls_direct import fit_irls_direct
from superglm.solvers.mode_score import linear_predictor
from superglm.solvers.structured import NestedSchurFactor, ProfiledNestedSchurFactor
from superglm.solvers.working_rows import fisher_working_weights
from superglm.types import GroupSlice, PenaltyComponent
from tests.test_signed_rows import _companion_bound, _exact_hessian, _exact_reference

EPS = np.finfo(np.float64).eps

FAMILIES = {
    "gaussian": Gaussian(),
    "poisson": Poisson(),
    "binomial": Binomial(),
    "gamma": Gamma(),
    "nb2": NegativeBinomial(theta=1.5),
    "tweedie125": Tweedie(p=1.25),
    "tweedie175": Tweedie(p=1.75),
}
LINKS = {
    "log": LogLink(),
    "identity": IdentityLink(),
    "logit": LogitLink(),
    "probit": ProbitLink(),
    "cloglog": CloglogLink(),
    "cauchit": CauchitLink(),
    "inverse": InverseLink(),
    "inverse_squared": InverseSquaredLink(),
    "sqrt": SqrtLink(),
    "power": PowerLink(0.5),
    "nb": NegativeBinomialLink(1.5),
}
PAIRS = [pytest.param(family, link, id=f"{family}-{link}") for family in FAMILIES for link in LINKS]


def test_the_registry_is_covered() -> None:
    """Every registered built-in family and link type appears in the parametrisation."""
    assert {type(value) for value in FAMILIES.values()} == set(_BUILTIN_REML_DISTRIBUTIONS)
    assert {type(value) for value in LINKS.values()} == set(_BUILTIN_REML_LINKS)


def _grid(family) -> tuple[np.ndarray, np.ndarray]:
    """Responses in the family's support against means in (0.05, 0.95), every link's range."""
    mu = np.linspace(0.05, 0.95, 7)
    if isinstance(family, Binomial):
        y = np.array([0.0, 1.0, 0.5])
    elif isinstance(family, Gaussian):
        y = np.array([-0.4, 0.3, 2.5])
    elif isinstance(family, Gamma):
        y = np.array([0.01, 0.4, 3.0])
    else:
        y = np.array([0.0, 1.0, 4.0])
    return np.repeat(y, len(mu)), np.tile(mu, len(y))


@pytest.mark.parametrize(("family_name", "link_name"), PAIRS)
def test_rows_meet_the_contract(family_name, link_name) -> None:
    family, link = FAMILIES[family_name], LINKS[link_name]
    y, mu = _grid(family)
    eta = link.link(mu)
    weight = np.full(len(y), 2.0)
    observed = compute_observed_information_weights(family, link, y, mu, eta, weight)
    error = observed_row_error_scale(family, link, y, mu, eta, weight, observed)
    fisher = fisher_working_weights(
        distribution=family, link=link, mu=mu, eta=eta, sample_weight=weight
    )
    assert np.all(np.isfinite(observed)) and np.all(np.isfinite(error))
    assert np.all(error >= np.abs(observed))
    assert np.all(fisher >= 0.0)
    closed_form = type(link) is LogLink and type(family) in (
        Gamma,
        Poisson,
        Tweedie,
        NegativeBinomial,
    )
    if closed_form:
        np.testing.assert_array_equal(error, np.abs(observed))
    if classify_reml_curvature(family, link) == "fisher":
        # the residual term vanishes identically: observed rows are Fisher rows
        # up to the rounding their error scale states.  The observed row takes
        # 11 operations and at most 5 elementary-function values (u, v, V, V'
        # and the link's inverse), each within one rounding of terms bounded by
        # the error scale; the Fisher row at most 8 more: gamma_24 of it
        # (Higham 2002, Lemma 3.1).
        roundings = 24 * EPS / 2
        assert np.all(np.abs(observed - fisher) <= roundings / (1 - roundings) * error)


# ── dispatch ──────────────────────────────────────────────────────────────


class _DispatchedError(Exception):
    """Raised once the fit has taken its structured decision."""


def _dispatch_data(variant: str, family) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict]:
    """A 2,000-row fit beside a 40-level random effect, as the pricing fits of
    ``test_structured_irls`` (a shape whose chain auto prices ahead of gram)."""
    rng = np.random.default_rng(17)
    n = 2000
    region = rng.integers(0, 40, n)
    frame = pd.DataFrame(
        {
            "age": rng.uniform(18.0, 80.0, n),
            "cover": [f"c{code}" for code in rng.integers(0, 5, n)],
            "region": [f"r{code:02d}" for code in region],
        }
    )
    mean = rng.uniform(0.2, 0.8, n)
    if isinstance(family, Binomial):
        y = (rng.uniform(size=n) < mean).astype(float)
    elif isinstance(family, Gaussian):
        y = mean + rng.normal(0.0, 0.05, n)
    elif isinstance(family, Gamma):
        y = rng.gamma(4.0, mean / 4.0)
    elif isinstance(family, Tweedie):
        counts = rng.poisson(mean)
        y = np.array([rng.gamma(2.0, 0.5, count).sum() for count in counts])
    else:
        y = rng.poisson(mean).astype(float)
    weight = np.ones(n)
    features = {"age": Numeric(), "cover": Categorical(), "region": RandomEffect()}
    if variant == "weights_1e6":
        weight = weight * 1e6
    elif variant == "offset_1e8":
        frame["age"] = frame["age"] + 1e8
    elif variant == "lambda_1e-7":
        features["region"] = RandomEffect(lambda_policy=LambdaPolicy.fixed(1e-7))
    elif variant == "duplicated":
        frame["age_copy"] = frame["age"]
        features["age_copy"] = Numeric()
    return frame, y, weight, features


_RESOLVERS = {
    module: module.resolve_structured_backend for module in (direct_module, discrete_module)
}


def _recorded_decision(monkeypatch, family, link, variant: str, discrete: bool):
    decisions = []
    for module, real in _RESOLVERS.items():

        def record(*args, _real=real, **kwargs):
            decisions.append(_real(*args, **kwargs))
            raise _DispatchedError

        monkeypatch.setattr(module, "resolve_structured_backend", record)
    frame, y, weight, features = _dispatch_data(variant, family)
    model = SuperGLM(
        family=family,
        link=link,
        features=features,
        selection_penalty=0,
        direct_solve="auto",
        discrete=discrete,
    )
    with warnings.catch_warnings(), pytest.raises(_DispatchedError):
        warnings.simplefilter("ignore")
        model.fit_reml(frame, y, sample_weight=weight)
    (decision,) = decisions
    return (
        decision.use_structured,
        decision.group_name,
        decision.chain_group_indices,
        decision.nested_fallback_reason,
        decision.fallback_reason,
    )


VARIANTS = ("base", "weights_1e6", "offset_1e8", "lambda_1e-7", "duplicated")


@pytest.mark.parametrize("discrete", [False, True], ids=["exact", "discrete"])
@pytest.mark.parametrize(("family_name", "link_name"), PAIRS)
def test_the_dispatch_is_one_decision_for_every_pair(
    monkeypatch, family_name, link_name, discrete
) -> None:
    """The decision reads the model's terms and sizes only (design §3.3, §6).

    Every pair, whatever the sign of its observed rows, takes the chain of one
    that Gaussian/identity takes on this shape, with no decline and no reason
    for gram, and no weight scale, column offset, tiny penalty or duplicated column moves
    it.
    """
    family, link = FAMILIES[family_name], LINKS[link_name]
    decisions = {
        variant: _recorded_decision(monkeypatch, family, link, variant, discrete)
        for variant in VARIANTS
    }
    for decision in decisions.values():
        assert decision == (True, "region", decisions["base"][2], None, None)
    assert len(decisions["base"][2]) == 1


# ── the mini trigger battery ──────────────────────────────────────────────


def _battery_design(seed: int):
    """A 3-level chain of 4 / 10 / 24 levels beside a normal, a mean-10 column
    offset by 1e6 and a base-coded 3-level categorical, 240 rows."""
    rng = np.random.default_rng(seed)
    n = 240
    parents = [np.concatenate([np.arange(4), rng.integers(0, 4, 6)])]
    parents.append(np.concatenate([np.arange(10), rng.integers(0, 10, 14)]))
    leaf = rng.integers(0, 24, n)
    model = parents[1][leaf]
    make = parents[0][model]
    dense = np.column_stack([rng.normal(size=n), 1e6 + 10.0 + 3.0 * rng.normal(size=n)])
    matrices = [
        DenseGroupMatrix(dense),
        CategoricalGroupMatrix(rng.integers(-1, 2, n), n_levels=2),
        RandomEffectGroupMatrix(make, 4),
        RandomEffectGroupMatrix(model, 10),
        RandomEffectGroupMatrix(leaf, 24),
    ]
    groups, start = [], 0
    for name, matrix in zip(("x", "cat", "make", "model", "variant"), matrices, strict=True):
        end = start + matrix.shape[1]
        groups.append(
            GroupSlice(name=name, start=start, end=end, penalized=name not in ("x", "cat"))
        )
        start = end
    dm = DesignMatrix(matrices, n=n, p=start)
    components = [
        PenaltyComponent(
            name=group.name,
            group_name=group.name,
            group_index=index,
            group_sl=group.sl,
            omega_raw=None,
            penalty_kind="identity",
        )
        for index, group in enumerate(groups)
        if group.penalized
    ]
    signal = 0.2 * dense[:, 0] + rng.normal(0.0, 0.2, 24)[leaf] + rng.normal(0.0, 0.3, 4)[make]
    return dm, groups, components, signal


def _battery_response(family, link, signal, rng):
    mean = 0.5 + 0.15 * np.tanh(signal)
    if isinstance(family, Binomial):
        return (rng.uniform(size=len(mean)) < mean).astype(float)
    if isinstance(family, Gaussian):
        # rows w mu (2 mu - y) are negative for y > 2 mu, about 5% here
        return mean + rng.normal(0.0, 0.3, len(mean))
    if isinstance(family, Gamma | InverseGaussianLike):
        return rng.gamma(4.0, mean / 4.0)
    if isinstance(family, Tweedie):
        counts = rng.poisson(2.0 * mean)
        return np.array([rng.gamma(2.0, 0.25, count).sum() for count in counts])
    return rng.poisson(mean).astype(float)


def _penalized_mode(dm, groups, components, lambdas, family, link, y):
    """A penalized mode of the pair: the dense backend's damped Fisher PIRLS at these lambdas."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result, _ = fit_irls_direct(
            X=dm,
            y=y,
            weights=np.ones(dm.n),
            family=family,
            link=link,
            groups=groups,
            lambda2=lambdas,
            reml_penalties=components,
            direct_solve="gram",
            max_iter=100,
            weight_semantics="prior",
        )
    assert np.all(np.isfinite(result.beta)) and np.isfinite(result.intercept)
    return result


class InverseGaussianLike:
    """A custom family by protocol alone: ``V = mu^3`` (the inverse Gaussian variance).

    Its observed log-link rows ``w (2 y - mu) / mu^2`` are negative for ``y <
    mu / 2``, the case the family table sent to gram.
    """

    @property
    def scale_known(self) -> bool:
        return False

    @property
    def default_link(self) -> str:
        return "log"

    def variance(self, mu):
        return mu**3

    def variance_derivative(self, mu):
        return 3.0 * mu**2

    def variance_second_derivative(self, mu):
        return 6.0 * mu

    def variance_third_derivative(self, mu):
        return np.full_like(mu, 6.0)

    def reml_curvature(self, link) -> str:
        return "observed"

    def deviance_unit(self, y, mu):
        return (y - mu) ** 2 / (y * mu**2)

    def log_likelihood(self, y, mu, weights, phi: float = 1.0) -> float:
        return float(
            np.sum(
                weights
                * (
                    -0.5 * np.log(2.0 * np.pi * phi * y**3)
                    - (y - mu) ** 2 / (2.0 * phi * y * mu**2)
                )
            )
        )


SIGNED = [
    pytest.param(Gaussian(), LogLink(), id="gaussian-log"),
    pytest.param(Gamma(), IdentityLink(), id="gamma-identity"),
    pytest.param(Poisson(), SqrtLink(), id="poisson-sqrt"),
    pytest.param(Binomial(), CauchitLink(), id="binomial-cauchit"),
    pytest.param(Binomial(), LogLink(), id="binomial-log"),
    pytest.param(Tweedie(p=1.75), SqrtLink(), id="tweedie175-sqrt"),
    pytest.param(InverseGaussianLike(), LogLink(), id="custom-inverse-gaussian-log"),
]


@pytest.mark.parametrize("lam", [1.0, 1e-7], ids=["moderate", "tiny"])
@pytest.mark.parametrize(("family", "link"), SIGNED)
def test_the_observed_geometry_on_the_chain_is_exact(family, link, lam) -> None:
    """The family's own observed rows at a penalized mode, on the chain, against exact.

    The geometry is built as the REML driver builds it (the chain and the
    rows' error scale): its ``log|H|`` is within the companion bound of ``test_signed_rows`` of the
    exact rational ``log det`` of the same rows, and where those rows make the
    exact ``H`` indefinite the build refuses the point as infeasible.
    """
    dm, groups, components, signal = _battery_design(seed=5)
    rng = np.random.default_rng(6)
    y = _battery_response(family, link, signal, rng)
    lambdas = {"make": lam, "model": 2.0 * lam, "variant": 3.0 * lam}
    result = _penalized_mode(dm, groups, components, lambdas, family, link, y)
    sample_weight, offset = np.ones(dm.n), np.zeros(dm.n)
    arguments = dict(
        dm=dm,
        distribution=family,
        link=link,
        y=y,
        sample_weight=sample_weight,
        offset_arr=offset,
        result=result,
        penalty=None,
        groups=groups,
        lambdas=lambdas,
        reml_penalties=components,
        structured_group_index=4,
        structured_chain_group_indices=(2, 3, 4),
    )
    eta = stabilize_eta(linear_predictor(dm, result, None), link)
    mu = clip_mu(link.inverse(eta), family, link)
    rows = compute_observed_information_weights(family, link, y, mu, eta, sample_weight)
    S = build_penalty_matrix(dm.group_matrices, groups, lambdas, dm.p, components)
    X = dm.toarray()
    reference = _exact_reference(_exact_hessian(X, rows, S))
    if reference is None:
        with pytest.raises(ObservedGeometryInfeasibleError):
            build_observed_reml_geometry(**arguments)
        return
    geometry = build_observed_reml_geometry(**arguments)
    assert isinstance(geometry.hessian_inverse, ProfiledNestedSchurFactor)
    assert isinstance(geometry.hessian_inverse.augmented_factor, NestedSchurFactor)
    np.testing.assert_array_equal(geometry.weights, rows)
    error = observed_row_error_scale(family, link, y, mu, eta, sample_weight, rows)
    bound = _companion_bound(X, rows, S, error)
    assert abs(geometry.log_det_H - reference[0]) <= dm.p * bound


def test_the_battery_reaches_signed_rows() -> None:
    """At least the Gaussian/log, Gamma/identity and custom modes carry negative rows."""
    dm, groups, components, signal = _battery_design(seed=5)
    lambdas = {"make": 1.0, "model": 2.0, "variant": 3.0}
    for family, link in (
        (Gaussian(), LogLink()),
        (Gamma(), IdentityLink()),
        (InverseGaussianLike(), LogLink()),
    ):
        y = _battery_response(family, link, signal, np.random.default_rng(6))
        result = _penalized_mode(dm, groups, components, lambdas, family, link, y)
        eta = stabilize_eta(linear_predictor(dm, result, None), link)
        mu = clip_mu(link.inverse(eta), family)
        rows = compute_observed_information_weights(family, link, y, mu, eta, np.ones(dm.n))
        assert np.any(rows < 0.0), type(family).__name__


# ── the custom-family protocol ────────────────────────────────────────────


def _custom_fit(direct_solve: str) -> SuperGLM:
    rng = np.random.default_rng(23)
    n = 1500
    region = rng.integers(0, 60, n)
    frame = pd.DataFrame({"x": rng.normal(size=n), "region": [f"r{code:02d}" for code in region]})
    mean = np.exp(0.2 + 0.2 * frame["x"].to_numpy() + rng.normal(0.0, 0.2, 60)[region])
    # inverse Gaussian draws by Michael, Schucany and Haas (1976), shape 4
    nu = rng.normal(size=n) ** 2
    shape = 4.0
    root = (
        mean
        + mean**2 * nu / (2 * shape)
        - mean / (2 * shape) * np.sqrt(4 * mean * shape * nu + mean**2 * nu**2)
    )
    y = np.where(rng.uniform(size=n) <= mean / (mean + root), root, mean**2 / root)
    model = SuperGLM(
        family=InverseGaussianLike(),
        link="log",
        features={"x": Numeric(), "region": RandomEffect()},
        selection_penalty=0,
        direct_solve=direct_solve,
    )
    model.fit_reml(frame, y, reml_tol=1e-9, pirls_tol=1e-10)
    return model


def test_a_conforming_custom_family_fits_on_the_chain_and_matches_gram() -> None:
    """A custom family with signed observed rows keeps the chain (no audit table)
    and reaches the dense backend's REML optimum within ``reml_tol (1 + |V|)``."""
    chain = _custom_fit("structured")
    gram = _custom_fit("gram")
    profile = chain._reml_profile
    assert profile["direct_backend"] == "structured"
    assert profile["structured_chain"] == ("region",)
    assert profile["structured_nested_fallback_reason"] is None
    assert isinstance(chain._linear_system_state.profiled_factor, ProfiledNestedSchurFactor)
    assert chain._reml_result.converged and gram._reml_result.converged
    objective = gram._reml_result.objective
    assert abs(chain._reml_result.objective - objective) <= 1e-9 * (1.0 + abs(objective))


def test_a_custom_family_without_a_curvature_declaration_is_refused_before_fitting() -> None:
    """The protocol's one required declaration: without ``reml_curvature`` the
    REML fit refuses before any fit state exists (no route is guessed)."""

    class Undeclared(InverseGaussianLike):
        reml_curvature = None  # type: ignore[assignment]

    model = SuperGLM(
        family=Undeclared(),
        link="log",
        features={"x": Numeric(), "region": RandomEffect()},
        selection_penalty=0,
    )
    frame = pd.DataFrame({"x": np.linspace(0.0, 1.0, 50), "region": ["a", "b"] * 25})
    with pytest.raises(NotImplementedError, match="explicit ordinary REML curvature"):
        model.fit_reml(frame, np.linspace(1.0, 2.0, 50))
    assert model._fit_state is None
