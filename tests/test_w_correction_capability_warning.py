"""The Fisher-path W(rho) drop must warn, not vanish silently (RFC-13, audit J.4)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from superglm import CubicRegressionSpline, SuperGLM
from superglm.distributions import clip_mu
from superglm.links import stabilize_eta
from superglm.reml import build_penalty_caches
from superglm.reml.w_derivatives import compute_dW_deta, reml_w_correction
from superglm.solvers.irls_direct import fit_irls_direct

_SKIP_MESSAGE = "REML W(rho) correction skipped"


class _LinkWithoutDeriv2:
    """Delegates to a real link but hides deriv2_inverse."""

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        if name in {"deriv2_inverse", "deriv3_inverse"}:
            raise AttributeError(name)
        return getattr(self._inner, name)


class _DistributionWithoutVarianceDerivative:
    """Delegates to a real distribution but hides variance_derivative."""

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        if name in {"variance_derivative", "variance_second_derivative"}:
            raise AttributeError(name)
        return getattr(self._inner, name)


def _build_setup(family: str, seed: int):
    """Fit a two-spline model and assemble the pieces reml_w_correction needs."""
    rng = np.random.default_rng(seed)
    n = 400
    x1 = rng.uniform(0, 1, n)
    x2 = rng.uniform(0, 1, n)
    mu = np.exp(0.5 + np.sin(2 * np.pi * x1) + 0.5 * x2)
    if family == "poisson":
        y = rng.poisson(mu).astype(float)
    elif family == "gamma":
        y = np.maximum(rng.gamma(shape=5.0, scale=mu / 5.0), 1e-4)
    else:  # pragma: no cover - guard against a typo in a future parametrization
        raise ValueError(family)
    df = pd.DataFrame({"x1": x1, "x2": x2})

    model = SuperGLM(
        features={
            "x1": CubicRegressionSpline(n_knots=8),
            "x2": CubicRegressionSpline(n_knots=8),
        },
        family=family,
    )
    model.fit(df, y)

    sample_weight = np.ones(n)
    offset_arr = np.zeros(n)
    lambdas = {"x1": 10.0, "x2": 0.5}

    reml_groups = [(i, group) for i, group in enumerate(model._groups) if group.penalized]
    penalty_caches = build_penalty_caches(model._dm.group_matrices, reml_groups)

    pirls_result, XtWX_S_inv, _ = fit_irls_direct(
        X=model._dm,
        y=y,
        weights=sample_weight,
        family=model._distribution,
        link=model._link,
        groups=model._groups,
        lambda2=lambdas,
        offset=offset_arr,
        return_xtwx=True,
        weight_semantics="frequency",
    )

    return {
        "model": model,
        "dm": model._dm,
        "link": model._link,
        "distribution": model._distribution,
        "groups": model._groups,
        "pirls_result": pirls_result,
        "XtWX_S_inv": XtWX_S_inv,
        "lambdas": lambdas,
        "reml_groups": reml_groups,
        "penalty_caches": penalty_caches,
        "sample_weight": sample_weight,
        "offset_arr": offset_arr,
    }


@pytest.fixture(scope="module")
def poisson_setup():
    """A fitted Poisson/log spline model plus the pieces reml_w_correction needs."""
    return _build_setup("poisson", seed=42)


@pytest.fixture(scope="module")
def gamma_setup():
    """A fitted Gamma/log spline model, whose W(rho) correction is structurally zero."""
    return _build_setup("gamma", seed=7)


def _call(setup, *, link=None, distribution=None):
    return reml_w_correction(
        dm=setup["dm"],
        link=link if link is not None else setup["link"],
        groups=setup["groups"],
        pirls_result=setup["pirls_result"],
        XtWX_S_inv=setup["XtWX_S_inv"],
        lambdas=setup["lambdas"],
        reml_groups=setup["reml_groups"],
        penalty_caches=setup["penalty_caches"],
        sample_weight=setup["sample_weight"],
        offset_arr=setup["offset_arr"],
        distribution=(distribution if distribution is not None else setup["distribution"]),
    )


def _eta_mu(setup, link, distribution):
    """The linear predictor and mean at which ``reml_w_correction`` evaluates W."""
    pirls_result = setup["pirls_result"]
    eta = stabilize_eta(
        setup["dm"].matvec(pirls_result.beta) + pirls_result.intercept + setup["offset_arr"],
        link,
    )
    return eta, clip_mu(link.inverse(eta), distribution)


def _dW_deta(setup, *, link=None, distribution=None):
    """Recompute dW/deta exactly as ``reml_w_correction``'s Fisher path does."""
    link = link if link is not None else setup["link"]
    distribution = distribution if distribution is not None else setup["distribution"]
    eta, mu = _eta_mu(setup, link, distribution)
    return compute_dW_deta(link, distribution, mu, eta, setup["sample_weight"])


def test_link_without_deriv2_inverse_warns(poisson_setup) -> None:
    link = _LinkWithoutDeriv2(poisson_setup["link"])
    # The branch under test is reached only when dW/deta comes back None.
    assert _dW_deta(poisson_setup, link=link) is None
    with pytest.warns(UserWarning, match="deriv2_inverse"):
        result = _call(poisson_setup, link=link)
    assert result is None


def test_distribution_without_variance_derivative_warns(poisson_setup) -> None:
    distribution = _DistributionWithoutVarianceDerivative(poisson_setup["distribution"])
    assert _dW_deta(poisson_setup, distribution=distribution) is None
    with pytest.warns(UserWarning, match="variance_derivative"):
        result = _call(poisson_setup, distribution=distribution)
    assert result is None


def test_warns_once_per_class_pair_not_once_per_iteration(poisson_setup) -> None:
    """The stdlib default filter must dedup per class pair, not per call.

    Both variants are raised from a *single* call site so that the filter's
    ``(text, category, lineno)`` key differs only in the message text.  That
    is what makes this discriminating: drop the class names from the message
    and six calls collapse to one warning instead of two.
    """
    variants = [
        {"link": _LinkWithoutDeriv2(poisson_setup["link"])},
        {"distribution": _DistributionWithoutVarianceDerivative(poisson_setup["distribution"])},
    ]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("default")  # what an unconfigured process uses
        for _ in range(3):
            for kwargs in variants:
                _call(poisson_setup, **kwargs)

    messages = [str(w.message) for w in caught if _SKIP_MESSAGE in str(w.message)]
    assert len(messages) == 2, messages
    assert sum("deriv2_inverse" in message for message in messages) == 1
    assert sum("variance_derivative" in message for message in messages) == 1


def test_one_custom_link_still_warns_for_each_counterpart(poisson_setup, gamma_setup) -> None:
    """Same missing method, two counterpart distributions => two warnings.

    The complementary axis to the test above: there the two variants differ in
    *which* method is missing, here they differ only in the *counterpart* that
    is not missing anything.  Both are raised from the single call site inside
    ``_call``, so the filter's ``(text, category, lineno)`` key again differs
    only in message text.  Name only the missing method and the two pairs
    collapse to one warning, silently hiding the second degraded fit.
    """
    link = _LinkWithoutDeriv2(poisson_setup["link"])
    distributions = [poisson_setup["distribution"], gamma_setup["distribution"]]
    assert {type(d).__name__ for d in distributions} == {"Poisson", "Gamma"}
    # Only the link is deficient; both distributions have variance_derivative.
    for distribution in distributions:
        assert hasattr(distribution, "variance_derivative")
        assert _dW_deta(poisson_setup, link=link, distribution=distribution) is None

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("default")
        for distribution in distributions:
            for _ in range(2):
                _call(poisson_setup, link=link, distribution=distribution)

    messages = [str(w.message) for w in caught if _SKIP_MESSAGE in str(w.message)]
    assert len(messages) == 2, messages
    assert sum("Poisson" in message for message in messages) == 1
    assert sum("Gamma" in message for message in messages) == 1


def _deficient_link_class(module_name: str) -> type:
    """A link-wrapper class called ``MyLink`` that claims to live in ``module_name``.

    Two of these differ *only* in ``__module__``.  Two ``class`` statements in
    this file could not express that -- they would share ``__module__`` -- so
    the type is built here and its identity attributes set explicitly.
    ``__qualname__`` is overridden too: defined in a function it would
    otherwise be ``_deficient_link_class.<locals>.MyLink``, which already
    differs from nothing and would let a ``__name__``-keyed message pass.
    """

    class MyLink(_LinkWithoutDeriv2):
        pass

    MyLink.__module__ = module_name
    MyLink.__qualname__ = "MyLink"
    return MyLink


def test_same_class_name_in_different_modules_warns_twice(poisson_setup) -> None:
    """Two distinct ``MyLink`` classes from different modules => two warnings.

    The third dedup axis.  A bare ``type(x).__name__`` key collides here: both
    pairs render identically, the filter drops the second, and one degraded
    fit goes unreported.  Keying on ``__module__`` + ``__qualname__`` separates
    them.
    """
    alpha = _deficient_link_class("acme.alpha_links")(poisson_setup["link"])
    beta = _deficient_link_class("acme.beta_links")(poisson_setup["link"])

    # The collision is real: same bare name, same qualname, different classes.
    assert type(alpha) is not type(beta)
    assert type(alpha).__name__ == type(beta).__name__ == "MyLink"
    assert type(alpha).__qualname__ == type(beta).__qualname__ == "MyLink"
    assert type(alpha).__module__ != type(beta).__module__

    links = [alpha, beta]
    for link in links:
        assert _dW_deta(poisson_setup, link=link) is None

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("default")
        for link in links:
            for _ in range(2):
                _call(poisson_setup, link=link)

    messages = [str(w.message) for w in caught if _SKIP_MESSAGE in str(w.message)]
    assert len(messages) == 2, messages
    assert sum("acme.alpha_links" in message for message in messages) == 1
    assert sum("acme.beta_links" in message for message in messages) == 1


def test_builtin_link_does_not_warn(poisson_setup) -> None:
    """Poisson/log runs the whole correction, so neither early return may fire."""
    dW_deta = _dW_deta(poisson_setup)
    assert dW_deta is not None
    assert np.any(dW_deta)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _call(poisson_setup)

    assert result is not None, "the correction must actually be computed here"
    assert [str(w.message) for w in caught if _SKIP_MESSAGE in str(w.message)] == []


def test_compute_dW_deta_itself_is_silent(poisson_setup) -> None:
    """The warning lives in ``reml_w_correction``, never in ``compute_dW_deta``.

    ``compute_dW_deta`` has a second public entry point --
    ``model_compute_dW_deta`` (``model/reml_ops.py:13``), surfaced as
    ``Model._compute_dW_deta`` and re-exported from ``reml/__init__.py``.  That
    is a bare derivative query making no REML claim, so a warning about skipped
    smoothing-parameter gradients does not belong on it.

    Without this test the placement is unpinned: moving the ``warnings.warn``
    call from ``reml_w_correction`` into ``compute_dW_deta`` leaves every other
    test in this file green.
    """
    variants = [
        {"link": _LinkWithoutDeriv2(poisson_setup["link"])},
        {"distribution": _DistributionWithoutVarianceDerivative(poisson_setup["distribution"])},
    ]

    for kwargs in variants:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert _dW_deta(poisson_setup, **kwargs) is None

        assert [str(w.message) for w in caught if _SKIP_MESSAGE in str(w.message)] == [], kwargs


def test_model_compute_dW_deta_is_silent(poisson_setup, monkeypatch) -> None:
    """The same guard through the public ``Model._compute_dW_deta`` wrapper.

    This is the path the placement decision exists to protect, so assert it
    directly rather than only on the underlying function.
    """
    model = poisson_setup["model"]
    link = _LinkWithoutDeriv2(poisson_setup["link"])
    monkeypatch.setattr(model, "_link", link)
    eta, mu = _eta_mu(poisson_setup, link, poisson_setup["distribution"])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = model._compute_dW_deta(mu, eta, poisson_setup["sample_weight"])

    assert result is None, "the capability gate is what must be exercised here"
    assert [str(w.message) for w in caught if _SKIP_MESSAGE in str(w.message)] == []


def test_gamma_log_structural_zero_does_not_warn(gamma_setup) -> None:
    """Gamma/log has a genuinely zero correction; it must stay silent."""
    dW_deta = _dW_deta(gamma_setup)
    # Not None -- the capability branch is *not* how Gamma/log returns None.
    assert dW_deta is not None
    assert not np.any(dW_deta), "Gamma/log must reach the structural-zero branch"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _call(gamma_setup)

    assert result is None
    assert [str(w.message) for w in caught if _SKIP_MESSAGE in str(w.message)] == []
