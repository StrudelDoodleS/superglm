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

    def test_cached_search_matches_uncached_on_a_tensor_interaction(self, monkeypatch):
        """Interaction specs are taught at build time like main-effect specs;
        a cache hit that restores only _specs hands later candidates fresh
        untaught interaction templates -- measured as a KeyError('x1:x2') at
        the second candidate, while the uncached search completes."""
        rng = np.random.default_rng(21)
        n = 700
        x1 = rng.uniform(0, 1, n)
        x2 = rng.uniform(0, 1, n)
        eta = 0.5 + np.sin(2 * np.pi * x1) + 0.3 * x2
        y = np.where(rng.random(n) < 0.4, 0.0, rng.gamma(1.3, np.exp(eta), n))
        frame = pd.DataFrame({"x1": x1, "x2": x2})

        def run():
            model = SuperGLM(
                family=families.tweedie(p=1.5),
                features={
                    "x1": Spline(kind="cr", n_knots=5),
                    "x2": Spline(kind="cr", n_knots=5),
                },
                interactions=[("x1", "x2")],
            )
            result = model.estimate_p(frame, y, fit_mode="reml", maxiter=6)
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


class TestCacheScope:
    def test_direct_public_search_does_not_hijack_a_later_refit(self):
        """The exported search must not leave its design cache on the model.

        A leaked cache is not a stale attribute, it is a wrong answer: the
        next fit_reml sees a nonempty cache, probe-verifies the cached
        design against itself -- the probe certifies integrity, not input
        identity -- and silently fits the search's dataset instead of the
        one the caller just passed.
        """
        from superglm.profiling.tweedie import estimate_tweedie_p

        frame1, y1, features = _search_fixture(n=900, seed=3)
        model = SuperGLM(family=families.tweedie(p=1.5), features=features)
        estimate_tweedie_p(model, frame1, y1, fit_mode="fit_reml", maxiter=6)

        assert not hasattr(model, "_profile_design_cache")

        frame2, y2, _ = _search_fixture(n=1_300, seed=11)
        model.fit_reml(frame2, y2, runtime_validation="skip")
        assert len(model._fit_weights) == len(y2)

        reference = SuperGLM(family=families.tweedie(p=1.5), features=features)
        reference.fit_reml(frame2, y2, runtime_validation="skip")
        np.testing.assert_allclose(
            np.asarray(model.result.beta, dtype=float),
            np.asarray(reference.result.beta, dtype=float),
            rtol=1e-10,
            atol=0,
        )
        assert float(model.result.intercept) == pytest.approx(
            float(reference.result.intercept), rel=1e-10
        )

    def test_an_abandoned_search_still_removes_the_cache(self, monkeypatch):
        """Cleanup must survive a search that dies mid-candidate."""
        import superglm.profiling.tweedie as tweedie_module
        from superglm.profiling.tweedie import estimate_tweedie_p

        def exploding(*args, **kwargs):
            raise RuntimeError("forced mid-search failure")

        monkeypatch.setattr(tweedie_module, "_search_brent", exploding)
        frame, y, features = _search_fixture(n=900, seed=3)
        model = SuperGLM(family=families.tweedie(p=1.5), features=features)
        with pytest.raises(RuntimeError, match="forced mid-search failure"):
            estimate_tweedie_p(model, frame, y, fit_mode="fit_reml", maxiter=6)

        assert not hasattr(model, "_profile_design_cache")


class TestConstrainedGroupsDisableTheCache:
    def test_a_constrained_search_never_serves_from_the_cache(self, monkeypatch):
        """GroupSlice objects are mutated in place by the constrained REML
        path (strip_qp_constraints / restore_qp_constraints recompose against
        the current group matrix), and the fixed-probe check certifies only
        the design's matvec -- it cannot see a mutated group. Until the
        bitwise-equivalence evidence covers constrained fixtures, the cache
        stands down and every candidate builds fresh."""
        calls = _count_builds(monkeypatch)
        rng = np.random.default_rng(5)
        n = 900
        x = rng.uniform(0.0, 1.0, n)
        eta = 0.8 * x - 0.6
        y = np.where(rng.random(n) < 0.4, 0.0, rng.gamma(1.2, np.exp(eta) * 2.0, n))
        frame = pd.DataFrame({"x": x})
        from superglm import Constraint

        features = {"x": Spline(kind="ps", n_knots=8, constraint=Constraint.fit.increasing)}

        model = SuperGLM(family=families.tweedie(p=1.5), features=features)
        model.estimate_p(frame, y, fit_mode="reml", maxiter=8)

        # One build per candidate fit plus the publication: strictly more
        # than the cached search's two, proving no candidate was served a
        # possibly-mutated design.
        assert len(calls) > 2
