"""The coupled power search builds the candidate design matrix once.

Every candidate ``fit_reml`` at a new power rebuilds a design matrix that is
bit-identical across candidates -- the design depends on the frame, features,
and weights, never on ``p`` -- at ~19% of each candidate fit. A profile-scoped
cache serves later candidates from the first build, recomputing only the
family-dependent distribution/link pair, and self-verifies with a fixed-probe
matvec digest so a mutated design can never be silently served: on mismatch it
drops the cache and rebuilds. The published refit never uses the cache.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from superglm import Categorical, OrderedCategorical, Spline, SuperGLM, families


def _search_fixture(n: int = 1_200, seed: int = 7):
    rng = np.random.default_rng(seed)
    cat_levels = [f"c{j}" for j in range(4)]
    cat = np.array(cat_levels)[rng.integers(0, 4, n)]
    oc_levels = [f"o{j:02d}" for j in range(8)]
    oc_idx = rng.integers(0, 8, n)
    eta = 0.3 * (cat == "c1") + 0.05 * (oc_idx - 4) - 0.5
    y = np.where(rng.random(n) < 0.4, 0.0, rng.gamma(1.2, np.exp(eta) * 2.0, n))
    frame = pd.DataFrame({"c": cat, "o": np.array(oc_levels)[oc_idx]})
    features = {
        "c": Categorical(),
        "o": OrderedCategorical(order=oc_levels, basis=Spline(kind="cr", k=8)),
    }
    return frame, y, features


def _count_builds(monkeypatch) -> list[str]:
    import superglm.model.base as base_module

    real = base_module.model_build_design_matrix
    calls: list[str] = []

    def counting(model, X, y, sample_weight, offset):
        calls.append(type(model).__name__)
        return real(model, X, y, sample_weight, offset)

    monkeypatch.setattr(base_module, "model_build_design_matrix", counting)
    return calls


class TestSearchBuildsOnce:
    def test_coupled_search_builds_the_design_twice_in_total(self, monkeypatch):
        """One build fills the candidate cache; the publication builds fresh."""
        calls = _count_builds(monkeypatch)
        frame, y, features = _search_fixture()

        model = SuperGLM(family=families.tweedie(p=1.5), features=features)
        model.estimate_p(frame, y, fit_mode="reml")

        assert len(calls) == 2

    def test_cached_search_matches_an_uncached_search_bitwise(self, monkeypatch):
        """The cache is a pure cost optimization: every number is identical."""
        frame, y, features = _search_fixture()

        def run():
            model = SuperGLM(family=families.tweedie(p=1.5), features=features)
            result = model.estimate_p(frame, y, fit_mode="reml")
            return result, np.asarray(model.result.beta, dtype=float).copy()

        cached_result, cached_beta = run()

        import superglm.profiling.tweedie as tweedie_module

        monkeypatch.setattr(tweedie_module, "_SEARCH_DM_CACHE", False)
        uncached_result, uncached_beta = run()

        assert float(cached_result.p_hat) == float(uncached_result.p_hat)
        assert float(cached_result.phi_hat) == float(uncached_result.phi_hat)
        assert list(cached_result.search_trace["nll"]) == list(uncached_result.search_trace["nll"])
        assert np.array_equal(cached_beta, uncached_beta)


class TestCacheMechanism:
    def test_repeat_fits_are_served_from_the_cache(self, monkeypatch):
        calls = _count_builds(monkeypatch)
        frame, y, features = _search_fixture(n=600)

        model = SuperGLM(family=families.tweedie(p=1.5), features=features)
        model._profile_design_cache = {}
        model.fit_reml(frame, y, runtime_validation="skip")
        assert len(calls) == 1
        model.fit_reml(frame, y, runtime_validation="skip")
        assert len(calls) == 1

    def test_a_tampered_cache_is_dropped_and_rebuilt(self, monkeypatch):
        """The digest check fails closed: never serve a mutated design."""
        calls = _count_builds(monkeypatch)
        frame, y, features = _search_fixture(n=600)

        model = SuperGLM(family=families.tweedie(p=1.5), features=features)
        model._profile_design_cache = {}
        model.fit_reml(frame, y, runtime_validation="skip")
        assert len(calls) == 1

        model._profile_design_cache["expected_probe"] = (
            model._profile_design_cache["expected_probe"] + 1.0
        )
        model.fit_reml(frame, y, runtime_validation="skip")
        assert len(calls) == 2

    def test_the_cache_recomputes_the_family_per_fit(self, monkeypatch):
        """A cached design must never pin the first candidate's power."""
        frame, y, features = _search_fixture(n=600)

        model = SuperGLM(family=families.tweedie(p=1.3), features=features)
        model._profile_design_cache = {}
        model.fit_reml(frame, y, runtime_validation="skip")
        assert float(model._distribution.p) == pytest.approx(1.3)

        model.family = families.tweedie(p=1.7)
        model.fit_reml(frame, y, runtime_validation="skip")
        assert float(model._distribution.p) == pytest.approx(1.7)
