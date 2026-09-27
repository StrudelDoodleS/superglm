"""Models pickled by superglm 0.35 still load after the Tweedie/NB2 profiling rebuild.

A 0.35 Tweedie model fitted by ``fit_reml`` holds its REML-scale memo, and one
that ran ``estimate_p`` holds its profile result, and both name classes the
rebuild retired. A 0.35 ``NBProfileResult`` has a theta cache and bare interval
pairs where the rebuilt result has evaluations and ``Interval`` records.

Each test writes a stream the way 0.35 did: the retired classes are registered
under their 0.35 names only while pickling, and the result states copy the
0.35.0 field layout. The restore is then checked on the loaded model.
"""

import pickle
from contextlib import contextmanager

import numpy as np
import pandas as pd
import pytest

import superglm.profiling.tweedie as profiling_tweedie
import superglm.reml.scale as reml_scale
from superglm import NegativeBinomial, Spline, SuperGLM, Tweedie, generate_tweedie_cpg
from superglm.model.fit_state import FrozenMapping
from superglm.profiling import NBProfileResult, TweedieProfileResult
from superglm.profiling._scalar import Interval

_RETIRED_NAMES = (
    (reml_scale, "TweedieScaleProfileData"),
    (profiling_tweedie, "TweedieProfileCIDensityProvenance"),
    (profiling_tweedie, "TweedieProfileCIDetails"),
    (profiling_tweedie, "TweedieProfileCIEndpoint"),
    (profiling_tweedie, "TweedieProfileCIEvaluation"),
    (profiling_tweedie, "_PhiProfileResult"),
    (profiling_tweedie, "_PreparedTweedieDensity"),
    (profiling_tweedie, "_ProfileContext"),
    (profiling_tweedie, "_ProfileContextREML"),
    (profiling_tweedie, "_ProfileEvaluation"),
    (profiling_tweedie, "_TweedieLogpdfDiagnostics"),
)


# 0.35's search contexts; a pickled bound method records its function's name.
def evaluate(self, p, source=""):
    return 0.0


def evaluation_count(self):
    return 0


def evaluation_record(self, p):
    return None


@contextmanager
def _classes_named_as_in_0_35():
    """Stand-ins that pickle under 0.35's names, registered only while writing."""
    retired = {}
    for module, name in _RETIRED_NAMES:
        methods = (evaluate, evaluation_count, evaluation_record)
        cls = type(name, (), {method.__name__: method for method in methods})
        cls.__module__, cls.__qualname__ = module.__name__, name
        vars(module)[name] = retired[name] = cls
    try:
        yield retired
    finally:
        for module, name in _RETIRED_NAMES:
            del vars(module)[name]


def _tweedie_result_as_pickled_by_0_35(retired):
    """A result in 0.35's layout: its profile evaluated through a retired search context."""
    context = retired["_ProfileContextREML"]()
    record = retired["_ProfileEvaluation"]()
    record.phi_profile = retired["_PhiProfileResult"]()
    record.density = retired["_TweedieLogpdfDiagnostics"]()
    context.records = {1.5126: record}
    context.prepared = retired["_PreparedTweedieDensity"]()
    details = retired["TweedieProfileCIDetails"]()
    details.lower = retired["TweedieProfileCIEndpoint"]()
    details.upper = retired["TweedieProfileCIEndpoint"]()
    details.evaluations = (retired["TweedieProfileCIEvaluation"](),)
    details.density_provenance = (retired["TweedieProfileCIDensityProvenance"](),)
    trace = pd.DataFrame(
        {
            "step": [0, 1, 2],
            "p": [1.05, 1.95, 1.5126],
            "phi": [0.989, 8.595, 2.397],
            "nll": [4.2519, 3.1425, 2.3257],
            "n_iter": [4, 8, 5],
            "fit_converged": [True, True, True],
            "source": ["brent", "brent", "brent"],
        }
    )
    state = {
        "p_hat": 1.5126,
        "phi_hat": 2.3968,
        "nll": 2.32568,
        "n_evaluations": 3,
        "converged": True,
        "method": "brent",
        "phi_method": "mle",
        "search_trace": trace,
        "warnings": [],
        "density_method": "exact",
        "requested_method": "auto",
        "fit_mode": "fit_reml",
        "search_fit_mode": "fit_reml",
        "search_nll": 2.32567,
        "_objective": context.evaluate,
        "_ll_scale": 300.0,
        # The 0.01 interval reached the old search range at its lower end.
        "_ci_cache": {0.05: (1.4874, 1.5382), 0.01: (1.02, 1.5501)},
        "_ci_details_cache": {0.05: details},
        "_ci_p_range": (1.02, 1.98),
        "_evaluation_count": context.evaluation_count,
        "_evaluation_record": context.evaluation_record,
        "_infeasible_reason": {}.get,
    }
    result = TweedieProfileResult.__new__(TweedieProfileResult)
    result.__dict__.update(state)
    return result


def test_a_tweedie_reml_model_pickled_by_0_35_keeps_its_published_profile():
    rng = np.random.default_rng(7)
    n = 300
    X = pd.DataFrame({"x": rng.uniform(0.0, 1.0, n)})
    y = generate_tweedie_cpg(n, np.exp(0.5 + X["x"].to_numpy()), 2.0, 1.5, rng=rng)
    model = SuperGLM(family=Tweedie(p=1.5), features={"x": Spline(n_knots=5)})
    model.fit_reml(X, y)
    with _classes_named_as_in_0_35() as retired:
        memo = retired["TweedieScaleProfileData"]()
        memo.prepared_positive = retired["_PreparedTweedieDensity"]()
        model._reml_result.tweedie_scale_data = memo
        model._tweedie_profile_result = _tweedie_result_as_pickled_by_0_35(retired)
        stream = pickle.dumps(model)

    restored = pickle.loads(stream)

    np.testing.assert_array_equal(restored.predict(X), model.predict(X))
    result = restored._tweedie_profile_result
    assert (result.p_hat, result.phi_hat, result.nll) == (1.5126, 2.3968, 2.32568)
    assert (result.converged, result.fit_mode, result.search_nll) == (True, "fit_reml", 2.32567)
    assert result.evaluations.to_dict("list") == {
        "p": [1.05, 1.95, 1.5126],
        "nll": [4.2519, 3.1425, 2.3257],
        "phi": [0.989, 8.595, 2.397],
        "fit_converged": [True, True, True],
    }
    info = restored.summary()._info
    assert (info["tweedie_p"], info["tweedie_phi"]) == (1.5126, 2.3968)
    assert (info["tweedie_p_ci"], info["tweedie_p_ci_status"]) == ((1.4874, 1.5382), "available")
    assert result.interval(0.01) == Interval(1.02, 1.5501, True, False)
    with pytest.raises(RuntimeError, match="pickled by superglm 0.35"):
        result.ci(0.1)


def test_an_nb_result_pickled_by_0_35_recomputes_its_interval_on_the_same_profile():
    rng = np.random.default_rng(11)
    n = 600
    X = pd.DataFrame({"x": rng.uniform(0.0, 1.0, n)})
    mu = np.exp(0.6 + X["x"].to_numpy())
    y = rng.negative_binomial(2.0, 2.0 / (2.0 + mu)).astype(np.float64)
    model = SuperGLM(family=NegativeBinomial(theta=1.0), features={"x": Spline(n_knots=5)})
    model.estimate_theta(X, y)
    published = model._nb_profile_result
    state = {
        "theta_hat": published.theta_hat,
        "nll": published.nll,
        "n_evaluations": 2,
        "converged": True,
        "cache": FrozenMapping({1.0: 1.9, published.theta_hat: published.nll}),
        "_y": published._y,
        "_mu": published._mu,
        "_weights": published._weights,
        "_weight_semantics": published._weight_semantics,
        # 0.35 cached bare pairs, and its summary always computed one.
        "_ci_cache": {0.05: (1.0, 3.0)},
        "_publication_locked": True,
    }
    legacy = NBProfileResult.__new__(NBProfileResult)
    legacy.__dict__.update(state)
    model._nb_profile_result = legacy

    restored = pickle.loads(pickle.dumps(model))

    expected = published.ci(0.05)
    assert restored.summary()._info["nb_theta_ci"] == expected
    result = restored._nb_profile_result
    assert result.evaluations.to_dict("list") == {
        "theta": [1.0, published.theta_hat],
        "nll": [1.9, published.nll],
    }
    assert (result.theta_hat, result.nll, result.warnings) == (
        published.theta_hat,
        published.nll,
        [],
    )
