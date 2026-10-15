"""Fits that are discarded skip the retained-fit rank metadata; the published fit keeps it.

The data-subspace rank decomposition (a TSQR over every row once the data
Gram cannot certify itself) is read only by retained-fit inference.  The
Tweedie power search's candidate fits and the NB2 auto-theta mean fits are
never published, and an auto-theta REML refit is published only if no later
refit replaces it, so only the published fit may pay for it.  The tests
force the factor route and count the data-rank factorisations; master paid
one per candidate and per mean fit, and fails both counts.
"""

import pickle

import numpy as np
import pandas as pd
import pytest

import superglm.model.reml_finalize as reml_finalize
import superglm.solvers.irls_direct as irls_direct
from superglm import NegativeBinomial, SuperGLM, Tweedie
from superglm.features.categorical import Categorical
from superglm.features.spline import Spline


@pytest.fixture
def data_rank_factorisations(monkeypatch):
    """Send the data rank to its TSQR route on every fit and count the factorisations."""
    calls = []
    decide = irls_direct.decompose_gram_if_authoritative
    factor = irls_direct.grouped_weighted_factor

    def factor_route(matrix, **kwargs):
        # Only the data-rank call passes ``psd_by_construction``.
        return None if "psd_by_construction" in kwargs else decide(matrix, **kwargs)

    def counted(*args, **kwargs):
        calls.append(1)
        return factor(*args, **kwargs)

    monkeypatch.setattr(irls_direct, "decompose_gram_if_authoritative", factor_route)
    monkeypatch.setattr(irls_direct, "grouped_weighted_factor", counted)
    return calls


def _frame(seed, n):
    rng = np.random.default_rng(seed)
    levels = np.array([f"L{j:02d}" for j in range(6)])
    idx = rng.integers(0, len(levels), n)
    frame = pd.DataFrame({"band": levels[idx], "z": rng.uniform(0.0, 10.0, n)})
    eta = -0.5 + 0.08 * idx + 0.3 * np.sin(frame["z"].to_numpy())
    return rng, frame, np.exp(eta)


def _features():
    return {"band": Categorical(), "z": Spline(kind="ps", k=6)}


def test_power_search_candidates_skip_the_data_rank_factorisation(data_rank_factorisations):
    rng, frame, mu = _frame(11, 400)
    y = np.where(rng.random(len(mu)) < 0.6, 0.0, rng.gamma(1.5, mu))
    weights = rng.uniform(0.2, 1.0, len(mu))
    model = SuperGLM(family=Tweedie(p=1.5), features=_features())
    result = model.estimate_p(frame, y, sample_weight=weights, fit_mode="reml")
    assert len(result.evaluations) > 2
    # The published refit at p_hat, and nothing else.
    assert len(data_rank_factorisations) == 1
    assert model.result.rank_info is not None


def _nb_auto(seed=3, n=600):
    rng, frame, mu = _frame(seed, n)
    y = rng.negative_binomial(1.5, 1.5 / (1.5 + 2.0 * mu)).astype(np.float64)
    return frame, y


def test_nb_auto_theta_pays_the_data_rank_factorisation_once(data_rank_factorisations):
    frame, y = _nb_auto()
    model = SuperGLM(family=NegativeBinomial(theta="auto"), features=_features())
    model.fit_reml(frame, y)
    trace = model._nb_profile_result.evaluations
    # Mean fits plus at least one REML fit ran; only the published one factors.
    assert len(trace) >= 2
    assert len(data_rank_factorisations) == 1
    assert model.result.rank_info is not None
    assert "_deferred_rank_info" not in model.__dict__
    assert "_defer_rank_info" not in model.__dict__


def test_deferred_rank_metadata_is_what_the_fit_computes_inline(monkeypatch):
    """The published fit's deferred RankInfo is bit-for-bit the inline one.

    Mutation check: computing the deferred metadata from any state but the
    published fit's own (the first REML fit's, say) changes its bytes here.
    """

    def fit(inline):
        frame, y = _nb_auto(seed=8)
        model = SuperGLM(family=NegativeBinomial(theta="auto"), features=_features())
        with monkeypatch.context() as patch:
            if inline:
                original = reml_finalize.fit_irls_direct

                def no_deferral(*args, _defer_rank_info=None, **kwargs):
                    kwargs.pop("compute_rank_info", None)
                    return original(*args, compute_rank_info=True, **kwargs)

                patch.setattr(reml_finalize, "fit_irls_direct", no_deferral)
            model.fit_reml(frame, y)
        return model

    deferred, inline = fit(inline=False), fit(inline=True)
    assert len(deferred._nb_profile_result.evaluations) >= 2
    np.testing.assert_array_equal(deferred.result.beta, inline.result.beta)
    assert deferred.result.rank_info is not None
    assert pickle.dumps(deferred.result.rank_info) == pickle.dumps(inline.result.rank_info)
    assert deferred._solver_result.rank_info is deferred.result.rank_info
    assert deferred._reml_result.pirls_result.rank_info is deferred.result.rank_info
