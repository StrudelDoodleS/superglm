"""The published REML fit must be lambda-determined; searches may run looser.

Wood's compound stopping criterion accepts when the projected gradient and the
objective change both fall below ``reml_tol * (1 + |objective|)``. That bar
scales with the magnitude of the REML objective -- which grows with the data --
while the gradient along a flat log-lambda direction does not. At the
historical default of 1e-6 a fit can stop with ``converged=True`` while the
smoothing parameter, and with it every published standard error, is still
moving: on the fixture below the worst coefficient SE shifts by ~92% between
the default fit and a tight one, with predictions essentially unchanged.

The resolution is engine-scoped. The Newton engines (exact and discrete
cached-W) use the compound bar and get a tight default. The EFS-family engines
(the step-criterion loops in efs.py / runner.py / scop_efs.py) already stop on
a lambda-change bound -- tightening them buys no determination and their
linear convergence would pay heavily -- so their default stays put. Power
search candidate fits only rank powers (their objective is determined to ~1e-8
even at the loose bar), so they keep the loose tolerance explicitly and the
publication refit repays determination once.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from superglm import Spline, SuperGLM, families
from superglm.model import state_ops

from ._tweedie_profile_fixtures import flat_lambda_fixture as _flat_lambda_fixture
from ._tweedie_profile_fixtures import search_fixture as _small_search_fixture


def _standard_errors(model: SuperGLM) -> np.ndarray:
    cov = state_ops.coef_covariance(model)
    cov = cov[0] if isinstance(cov, tuple) else cov
    return np.sqrt(np.clip(np.diag(np.asarray(cov, dtype=float)), 0.0, None))


def _spy_optimizer_tols(monkeypatch) -> list[float]:
    """Record the reml_tol every Newton-engine invocation actually receives."""
    from superglm.model import reml_ops

    real = reml_ops.optimize_direct_reml
    seen: list[float] = []

    def wrapper(*args, **kwargs):
        seen.append(float(kwargs["reml_tol"]))
        return real(*args, **kwargs)

    monkeypatch.setattr(reml_ops, "optimize_direct_reml", wrapper)
    return seen


def _replace_dc(instance, **changes):
    from dataclasses import replace

    return replace(instance, **changes)


class TestPublishedFitDetermination:
    def test_default_fit_publishes_determined_standard_errors(self):
        """A converged default fit's SEs must match a tight fit's to <0.5%.

        At reml_tol=1e-6 this fixture publishes a worst-coefficient SE 92%
        away from the tight answer, converged=True. The default must sit past
        the determination elbow (1e-9 measures 0.011% here).
        """
        frame, y, weights, offset, features = _flat_lambda_fixture()

        default_fit = SuperGLM(family=families.tweedie(p=1.5), features=features)
        default_fit.fit_reml(
            frame, y, sample_weight=weights, offset=offset, runtime_validation="skip"
        )
        tight = SuperGLM(family=families.tweedie(p=1.5), features=features)
        tight.fit_reml(
            frame,
            y,
            sample_weight=weights,
            offset=offset,
            runtime_validation="skip",
            reml_tol=1e-11,
        )

        assert default_fit._reml_result.converged
        assert tight._reml_result.converged
        se_default = _standard_errors(default_fit)
        se_tight = _standard_errors(tight)
        worst = float(np.max(np.abs(se_default - se_tight) / np.maximum(se_tight, 1e-300)))
        assert worst < 5e-3


class TestEngineScopedTolerance:
    def test_resolver_maps_the_sentinel_per_engine(self):
        from superglm.model.reml_execute import (
            NEWTON_REML_TOL_DEFAULT,
            STEP_REML_TOL_DEFAULT,
            resolve_reml_tol,
        )

        assert NEWTON_REML_TOL_DEFAULT == 1e-9
        assert STEP_REML_TOL_DEFAULT == 1e-6
        assert resolve_reml_tol(None, engine="newton") == NEWTON_REML_TOL_DEFAULT
        assert resolve_reml_tol(None, engine="step") == STEP_REML_TOL_DEFAULT
        assert resolve_reml_tol(2.5e-7, engine="newton") == 2.5e-7
        assert resolve_reml_tol(2.5e-7, engine="step") == 2.5e-7
        with pytest.raises(ValueError):
            resolve_reml_tol(None, engine="brent")

    def test_newton_engine_receives_the_tight_default(self, monkeypatch):
        seen = _spy_optimizer_tols(monkeypatch)
        frame, y, features = _small_search_fixture()

        model = SuperGLM(family=families.tweedie(p=1.5), features=features)
        model.fit_reml(frame, y, runtime_validation="skip")

        assert seen == [1e-9]

    def test_discrete_engine_floors_explicit_tolerances_at_1e_12(self):
        """The discrete engine clamps reml_tol at 1e-12 (disclosed in the
        docstring); pinned here on the real backend, not a wrapper spy."""
        from superglm import Spline as _Spline

        rng = np.random.default_rng(5)
        n = 1_500
        frame = pd.DataFrame({"x1": rng.uniform(0, 1, n), "x2": rng.uniform(0, 1, n)})
        eta = 0.4 * np.sin(5.0 * frame["x1"].to_numpy()) + 0.2 * frame["x2"].to_numpy()
        y = rng.poisson(np.exp(eta)).astype(float)

        def fit(tol):
            model = SuperGLM(
                family="poisson",
                selection_penalty=0,
                discrete=True,
                n_bins=32,
                features={
                    "x1": _Spline(kind="cr", n_knots=6),
                    "x2": _Spline(kind="cr", n_knots=6),
                },
            )
            model.fit_reml(frame, y, runtime_validation="skip", reml_tol=tol)
            return model._reml_result

        floored = fit(1e-15)
        at_floor = fit(1e-12)

        assert floored.n_reml_iter == at_floor.n_reml_iter
        assert floored.lambdas == at_floor.lambdas

    def test_an_explicit_tolerance_is_honored_verbatim(self, monkeypatch):
        seen = _spy_optimizer_tols(monkeypatch)
        frame, y, features = _small_search_fixture()

        model = SuperGLM(family=families.tweedie(p=1.5), features=features)
        model.fit_reml(frame, y, runtime_validation="skip", reml_tol=2.5e-7)

        assert seen == [2.5e-7]


class TestCandidateGradePaths:
    """Every candidate-grade fit runs at the search tolerance, not the default.

    interaction_mode='fast_candidate' caps outer iterations at 5 and exists to
    rank interaction candidates; under that cap the tight publication default
    cannot buy determination -- it only burns an extra Newton iteration and
    flips converged flags in screening logs.
    """

    @staticmethod
    def _interaction_fixture(n: int = 800, seed: int = 3):
        rng = np.random.default_rng(seed)
        frame = pd.DataFrame({"x1": rng.uniform(0, 1, n), "x2": rng.uniform(0, 1, n)})
        eta = 0.3 * np.sin(6.0 * frame["x1"].to_numpy()) + 0.2 * frame["x2"].to_numpy()
        y = rng.poisson(np.exp(eta)).astype(float)
        return frame, y

    def test_fast_candidate_screening_runs_at_search_tolerance(self, monkeypatch):
        from superglm import Spline as _Spline

        seen = _spy_optimizer_tols(monkeypatch)
        frame, y = self._interaction_fixture()
        model = SuperGLM(
            family="poisson",
            selection_penalty=0,
            features={"x1": _Spline(kind="cr", n_knots=6), "x2": _Spline(kind="cr", n_knots=6)},
            interactions=[("x1", "x2")],
        )
        model.fit_reml(frame, y, interaction_mode="fast_candidate", runtime_validation="skip")

        assert seen == [1e-6]

    def test_fast_candidate_honors_an_explicit_tolerance(self, monkeypatch):
        from superglm import Spline as _Spline

        seen = _spy_optimizer_tols(monkeypatch)
        frame, y = self._interaction_fixture()
        model = SuperGLM(
            family="poisson",
            selection_penalty=0,
            features={"x1": _Spline(kind="cr", n_knots=6), "x2": _Spline(kind="cr", n_knots=6)},
            interactions=[("x1", "x2")],
        )
        model.fit_reml(
            frame, y, interaction_mode="fast_candidate", runtime_validation="skip", reml_tol=3e-8
        )

        assert seen == [3e-8]


class TestEngineSeamSentinels:
    """Every model-bound engine wrapper resolves the sentinel for the engine
    it actually forwards to.

    Each forwards to exactly one engine whose loop would TypeError on a
    verbatim None (discrete.py floats the tolerance; efs.py compares
    against it), so the wrapper owns the resolution.

    **Two of the three seams below are production-reached; one is not.**
    ``model_optimize_direct_reml`` and ``model_optimize_efs_reml`` are live --
    ``fit_ops.py:2011-2012`` injects both into ``optimize_reml_best``, which
    calls them at ``reml_execute.py:343,387`` and ``:366,410``.
    ``model_optimize_discrete_reml_cached_w`` is NOT: its only inbound
    reference anywhere is ``SuperGLM._optimize_discrete_reml_cached_w``
    (``api.py:2081``), which nothing calls, plus the test below.  What runs the
    discrete cached-W optimizer in production is ``reml/direct.py:150``
    reaching ``reml.discrete.optimize_discrete_reml_cached_w`` directly, past
    this adapter chain entirely.

    That is the other half of audit finding S1
    (``notes/audit/2026-07-28/subsystems/model-orchestration.md``), which names
    the ``run_reml_once`` chain and this one together.  The first half was
    deleted by the PR "Delete the covariance chain no production fit reaches",
    which also dropped ``TestRunnerPathSentinel``: that class drove the deleted
    ``model_run_reml_once`` wrapper with a monkeypatched engine, so it pinned
    the wrapper and nothing else.  The sentinel contract it asserted --
    ``resolve_reml_tol`` picking the engine the seam forwards to -- survives at
    the two live seams below.  The discrete-cached-W seam is left in place and
    tracked as the follow-up rather than deleted alongside it, so that the
    remaining half of S1 is retired on its own evidence.
    """

    @staticmethod
    def _seam_model():
        import types

        return types.SimpleNamespace(
            _dm="DM0",
            _distribution=None,
            _link=None,
            _groups=[],
            _active_set=None,
            _discrete=False,
        )

    def test_discrete_cached_w_seam_resolves_the_newton_default(self, monkeypatch):
        import superglm.reml.discrete as discrete_module
        from superglm.model import reml_ops

        captured: list[float] = []

        def fake_optimize(*args, **kwargs):
            captured.append(kwargs["reml_tol"])
            return "RESULT"

        monkeypatch.setattr(discrete_module, "optimize_discrete_reml_cached_w", fake_optimize)
        for reml_tol in (None, 3e-7):
            reml_ops.model_optimize_discrete_reml_cached_w(
                self._seam_model(),
                None,
                None,
                None,
                [],
                {},
                {},
                max_reml_iter=1,
                reml_tol=reml_tol,
                verbose=False,
            )
        assert captured == [1e-9, 3e-7]

    def test_efs_seam_resolves_the_step_default(self, monkeypatch):
        import superglm.reml.efs as efs_module
        from superglm.model import reml_ops

        captured: list[float] = []

        def fake_optimize(*args, **kwargs):
            captured.append(kwargs["reml_tol"])
            return "RESULT", "DM"

        monkeypatch.setattr(efs_module, "optimize_efs_reml", fake_optimize)
        monkeypatch.setattr(reml_ops, "configured_penalty", lambda model: None)
        for reml_tol in (None, 3e-7):
            reml_ops.model_optimize_efs_reml(
                self._seam_model(),
                None,
                None,
                None,
                [],
                {},
                {},
                max_reml_iter=1,
                reml_tol=reml_tol,
                verbose=False,
            )
        assert captured == [1e-6, 3e-7]

    def test_direct_seam_resolves_the_newton_default(self, monkeypatch):
        from superglm.model import reml_ops

        captured: list[float] = []

        def fake_optimize(*args, **kwargs):
            captured.append(kwargs["reml_tol"])
            return "RESULT"

        monkeypatch.setattr(reml_ops, "optimize_direct_reml", fake_optimize)
        for reml_tol in (None, 3e-7):
            reml_ops.model_optimize_direct_reml(
                self._seam_model(),
                None,
                None,
                None,
                [],
                {},
                {},
                max_reml_iter=1,
                reml_tol=reml_tol,
                verbose=False,
            )
        assert captured == [1e-9, 3e-7]


class TestPublicationDispersion:
    def test_discrete_publication_describes_the_public_mean(self, subtests):
        """On a binned model the internal design's mean is an approximation;
        the published dispersion must be profiled at the mean callers get
        from predict(), not at the binned matvec. One published fit, one
        mean: the summary statistics must be computed at that same public
        mean, not left as a hybrid of public-mean phi inside binned-mean
        likelihood/deviance. Both are claims about one search."""
        from superglm import Spline as _Spline
        from superglm.model.fit_ops import _compute_fit_stats, _compute_null_mu
        from superglm.profiling.tweedie import profile_phi_at

        rng = np.random.default_rng(11)
        # 1500 rows keep the binned and public means ~1e-2 apart, and the
        # dispersion profiled at each ~5e-5 apart: far above either claim's
        # resolution (measured 2026-09-23).
        n = 1_500
        frame = pd.DataFrame({"x1": rng.uniform(0.0, 1.0, n), "x2": rng.uniform(0.0, 1.0, n)})
        eta = 0.4 * np.sin(4.0 * frame["x1"].to_numpy()) + 0.3 * frame["x2"].to_numpy() - 0.6
        y = np.where(rng.random(n) < 0.5, 0.0, rng.gamma(1.4, np.exp(eta) * 3.0, n))

        model = SuperGLM(
            family=families.tweedie(p=1.5),
            selection_penalty=0,
            discrete=True,
            n_bins=64,
            features={"x1": _Spline(kind="cr", n_knots=8), "x2": _Spline(kind="cr", n_knots=8)},
        )
        result = model.estimate_p(frame, y, fit_mode="reml")
        mu = np.asarray(model.predict(frame), dtype=float)
        y_arr = np.asarray(y, dtype=float)
        ones = np.ones(n)

        with subtests.test("the published phi is profiled at the public mean"):
            oracle = profile_phi_at(y_arr, mu, ones, float(result.p_hat))
            assert float(result.phi_hat) == pytest.approx(float(oracle.phi), rel=1e-8)

        with subtests.test("the published statistics describe the public mean"):
            null_mu = _compute_null_mu(
                y_arr, ones, None, model._distribution, model._link, weight_semantics="prior"
            )
            oracle = _compute_fit_stats(
                y_arr,
                mu,
                ones,
                None,
                model._distribution,
                model._link,
                float(result.phi_hat),
                null_mu=null_mu,
                weight_semantics="prior",
            )

            np.testing.assert_allclose(model._fit_mu, mu, rtol=0, atol=0)
            assert model._fit_stats.log_likelihood == pytest.approx(
                oracle.log_likelihood, rel=1e-12
            )
            assert model._fit_stats.pearson_chi2 == pytest.approx(oracle.pearson_chi2, rel=1e-12)
            assert model._fit_stats.explained_deviance == pytest.approx(
                oracle.explained_deviance, rel=1e-12
            )

    def test_the_coupled_publication_runs_tight_and_owns_its_dispersion_story(self, subtests):
        """One coupled search, then every claim about what it published.

        The search is the expensive part and is identical for every claim, so
        it runs once. Each scenario below re-profiles a private deep copy of
        the published (model, result) under its own monkeypatch, so no
        scenario's mutations reach the next.
        """
        with pytest.MonkeyPatch.context() as mp:
            seen = _spy_optimizer_tols(mp)
            frame, y, features = _small_search_fixture()
            model = SuperGLM(family=families.tweedie(p=1.5), features=features)
            result = model.estimate_p(frame, y, fit_mode="reml")

        with subtests.test("coupled candidates run loose and the publication runs tight"):
            # Candidate fits rank powers; only the published refit pays for
            # determination. Every optimizer call before the last must carry
            # the loose search tolerance, and the last -- the publication fit
            # at p_hat -- the tight default.
            assert len(seen) >= 3
            assert set(seen[:-1]) == {1e-6}
            assert seen[-1] == 1e-9

        with subtests.test("the aggregate judges the publication refit, not the candidate"):
            # Candidates run at the loose search bar and the publication runs
            # tight, so the two can disagree on exactly the flat-lambda designs
            # the tolerance split was built for; the candidate's green flags
            # must not mask a stalled publication.
            from superglm.model import profile_ops

            published, searched = copy.deepcopy((model, result))
            assert searched.converged
            published._reml_result = _replace_dc(published._reml_result, converged=False)
            profile_ops._install_tweedie_profile(
                published, X=frame, y=y, offset=None, result=searched
            )
            assert searched.converged is False

    def test_a_decoupled_publication_reports_its_reml_convergence(self):
        """A decoupled run's search is ML, but it publishes a REML fit: the
        published convergence must describe that fit, not the ML candidates."""
        frame, y, features = _small_search_fixture()
        model = SuperGLM(family=families.tweedie(p=1.5), features=features)
        result = model.estimate_p(frame, y, fit_mode="reml", search_fit_mode="fit")

        assert result.fit_mode == "fit_reml"
        assert model._reml_result is not None and model._reml_result.converged
        assert result.converged is True

    def test_coupled_publication_profiles_phi_against_the_published_fit(self):
        """The published phi must describe the published fit, not the candidate.

        Candidates run at the search tolerance; the publication refit runs
        tight. Carrying the candidate's phi onto the tight refit scales every
        published SE by sqrt(phi) of the wrong fit: a relative phi gap of
        2.1e-6 on this fixture, 200x what the assertion resolves (measured
        2026-09-23). Weights and an offset keep the re-profile's weighted,
        offset mean in play.
        """
        from superglm.profiling.tweedie import profile_phi_at

        frame, y, features = _small_search_fixture()
        rng = np.random.default_rng(12)
        weights = rng.uniform(0.2, 1.0, len(y))
        offset = np.log(rng.choice([0.5, 1.0, 2.0], len(y)))
        model = SuperGLM(family=families.tweedie(p=1.5), features=features)
        result = model.estimate_p(frame, y, sample_weight=weights, offset=offset, fit_mode="reml")

        mu = np.asarray(model.predict(frame, offset=offset), dtype=float)
        oracle = profile_phi_at(np.asarray(y, dtype=float), mu, weights, float(result.p_hat))
        searched_phi = result.evaluations.set_index("p").loc[result.p_hat, "phi"]

        assert float(result.phi_hat) == pytest.approx(float(oracle.phi), rel=1e-12)
        assert float(result.phi_hat) != float(searched_phi)
        # The searched objective's value survives for the CI and the plots,
        # and the published nll refers to the published dispersion.
        assert np.isfinite(result.search_nll)
        assert result.nll == pytest.approx(oracle.criterion / len(y), rel=1e-12)
        assert model.result.phi == pytest.approx(float(result.phi_hat), rel=1e-12)


class TestSearchPublishSplit:
    def test_decoupled_publication_is_the_only_reml_fit_and_is_tight(self, monkeypatch):
        seen = _spy_optimizer_tols(monkeypatch)
        frame, y, features = _small_search_fixture()

        model = SuperGLM(family=families.tweedie(p=1.5), features=features)
        model.estimate_p(frame, y, fit_mode="reml", search_fit_mode="fit")

        assert seen == [1e-9]


class TestCertificationBar:
    def test_the_bar_moves_with_the_reml_tolerance_and_nothing_else(self):
        """The certificate's bar is what the REML stop needs (one-engine design §3.8).

        ``REML_TOL_BAR_RATIO * reml_tol`` (the forcing-term scaling of an inexact
        Newton method): a caller who asks the outer iteration for more asks
        PIRLS for proportionally more, and the tensor endgame below at
        ``reml_tol=1e-11`` needs it (with a fixed bar its line search failed).
        ``pirls_tol`` does not move it: a point that certified as a candidate
        cannot fail publication because the caller tightened ``pirls_tol``.
        The observed-geometry reading is the bar at the default tolerance.
        """
        import inspect

        from superglm.reml.observed_geometry import observed_mode_certification_bar
        from superglm.solvers.mode_score import (
            MODE_CERTIFICATION_BAR,
            REML_TOL_BAR_RATIO,
            mode_certification_bar,
        )

        assert mode_certification_bar(1e-9) == REML_TOL_BAR_RATIO * 1e-9
        assert mode_certification_bar(1e-11) == REML_TOL_BAR_RATIO * 1e-11
        assert mode_certification_bar() == MODE_CERTIFICATION_BAR
        assert observed_mode_certification_bar() == MODE_CERTIFICATION_BAR
        assert not inspect.signature(observed_mode_certification_bar).parameters
        # never below what float64 can express, never looser than a candidate fit's
        assert mode_certification_bar(1e-20) == 100.0 * np.finfo(float).eps
        assert mode_certification_bar(1.0) == mode_certification_bar(1e-6)


class TestFreezeDiagnostics:
    def test_the_profile_records_the_last_freeze_decision(self):
        """The active-set freeze is the mechanism that separates informative
        directions from inferentially flat ones; calibrating its bar needs
        the per-direction gradient and curvature it judged, recorded on the
        profile the way the resolved tolerance already is."""
        frame, y, features = _small_search_fixture()
        model = SuperGLM(family=families.tweedie(p=1.5), features=features)
        model.fit_reml(frame, y, runtime_validation="skip")

        profile = model._reml_profile
        freeze = profile["reml_freeze_decision"]
        assert set(freeze) == {
            "names",
            "proj_grad",
            "hess_diag",
            "row_curvature",
            "penalty_rank",
            "normalized_curvature",
            "curvature_bar",
            "score_scale",
            "estimated",
            "frozen",
        }
        assert len(freeze["names"]) == len(freeze["proj_grad"]) == len(freeze["hess_diag"])
        assert len(freeze["frozen"]) == len(freeze["names"])
        assert len(freeze["row_curvature"]) == len(freeze["penalty_rank"]) == len(freeze["names"])
        assert len(freeze["normalized_curvature"]) == len(freeze["names"])
        assert float(freeze["score_scale"]) > 0.0
        assert float(freeze["curvature_bar"]) > 0.0
        assert all(np.isfinite(v) for v in freeze["proj_grad"])
        # The audit reconstructs the verdict from the recorded quantities:
        # fixed directions freeze definitionally; estimated ones by the
        # judged symmetric per-dimension curvature against the bar and the
        # gradient against scale.
        for g, norm, est, fz in zip(
            freeze["proj_grad"],
            freeze["normalized_curvature"],
            freeze["estimated"],
            freeze["frozen"],
        ):
            gradient_flat = g < 1e-7 * float(freeze["score_scale"])
            curvature_flat = norm < float(freeze["curvature_bar"])
            assert fz == ((not est) or (gradient_flat and curvature_flat))


class TestFreezeRevalidation:
    def test_the_tolerance_exit_revalidates_a_live_masked_gradient(self):
        """benign_3k's frozen x2 keeps a raw gradient far above the default
        reml_tol*scale, so the accepting iteration cannot trust the stale
        mask blindly: a coupled partner's update can re-activate a masked
        direction through its CURVATURE, which the mask's gradient arm
        cannot see. The engine recomputes the freeze decision against the
        current Hessian before accepting, records that it did, and stops
        only because the mask survives -- with the calibrated behavior
        unchanged."""
        rng = np.random.default_rng(5)
        n = 3_000
        frame = pd.DataFrame({"x1": rng.uniform(0, 1, n), "x2": rng.uniform(0, 1, n)})
        eta = 0.4 * np.sin(5.0 * frame["x1"].to_numpy()) + 0.2 * frame["x2"].to_numpy()
        y = rng.poisson(np.exp(eta)).astype(float)
        model = SuperGLM(
            family="poisson",
            features={
                "x1": Spline(kind="cr", n_knots=8),
                "x2": Spline(kind="cr", n_knots=8),
            },
        )
        model.fit_reml(frame, y, runtime_validation="skip", max_reml_iter=200)

        r = model._reml_result
        assert r.converged
        assert model._reml_profile.get("reml_freeze_revalidated") is True
        assert int(r.n_reml_iter) <= 12
        assert float(r.lambdas["x1"]) == pytest.approx(0.0809, rel=0.05)

    def test_the_published_record_is_the_revalidation_itself(self, monkeypatch):
        """The revalidation is the last freeze decision made -- the one
        that authorized score_objective_tolerance -- so the published
        record must carry ITS quantities, not iteration k-1's."""
        import superglm.reml.direct as direct_module

        captured = []
        real = direct_module.freeze_flat_directions

        def spy(*args, **kwargs):
            decision = real(*args, **kwargs)
            captured.append(decision)
            return decision

        monkeypatch.setattr(direct_module, "freeze_flat_directions", spy)
        rng = np.random.default_rng(5)
        n = 3_000
        frame = pd.DataFrame({"x1": rng.uniform(0, 1, n), "x2": rng.uniform(0, 1, n)})
        eta = 0.4 * np.sin(5.0 * frame["x1"].to_numpy()) + 0.2 * frame["x2"].to_numpy()
        y = rng.poisson(np.exp(eta)).astype(float)
        model = SuperGLM(
            family="poisson",
            features={
                "x1": Spline(kind="cr", n_knots=8),
                "x2": Spline(kind="cr", n_knots=8),
            },
        )
        model.fit_reml(frame, y, runtime_validation="skip", max_reml_iter=200)

        assert model._reml_profile.get("reml_freeze_revalidated") is True
        freeze = model._reml_profile["reml_freeze_decision"]
        last = captured[-1]
        np.testing.assert_array_equal(freeze["frozen"], [bool(v) for v in last.frozen])
        np.testing.assert_array_equal(
            freeze["normalized_curvature"], [float(v) for v in last.normalized_curvature]
        )
        assert float(freeze["curvature_bar"]) == float(last.curvature_bar)

    def test_an_all_null_discrete_fit_settles_before_the_stationary_exit(self):
        """POI performs one working-model update per outer iteration, so an
        all-frozen verdict can be measured at a nonstationary coefficient
        mode. The stationary exit now also requires the objective arm's
        agreement; on this all-null fixture the settle costs one further
        iteration and the compound criterion takes the exit honestly
        (measured: identical lambdas either way; pre-fix stopped
        active_set_stationary at iteration 11 with the objective still
        moving)."""
        rng = np.random.default_rng(21)
        n = 2_000
        frame = pd.DataFrame({"x1": rng.uniform(0, 1, n), "x2": rng.uniform(0, 1, n)})
        y = rng.poisson(1.4, n).astype(float)
        model = SuperGLM(
            family="poisson",
            selection_penalty=0,
            discrete=True,
            n_bins=32,
            features={
                "x1": Spline(kind="cr", n_knots=8),
                "x2": Spline(kind="cr", n_knots=8),
            },
        )
        model.fit_reml(frame, y, runtime_validation="skip")

        r = model._reml_result
        assert r.converged
        assert str(r.termination_reason) == "score_objective_tolerance"
        assert int(r.n_reml_iter) >= 12

    def test_the_exact_stationary_exit_requires_a_stationary_mode(self):
        """The Fisher path admits a PIRLS-exhausted candidate, so the
        all-frozen exit could publish converged=True from a nonstationary
        beta. The gate requires a converged current mode with a stable
        objective; measured on this all-null fixture, warm-start chaining
        settles the mode across outer iterations, so even max_iter=1
        reaches the reference answer and the gated exit fires identically
        (lambdas match the default-budget run to <0.1%)."""
        rng = np.random.default_rng(21)
        n = 1_500
        frame = pd.DataFrame({"x1": rng.uniform(0, 1, n), "x2": rng.uniform(0, 1, n)})
        y = rng.poisson(1.4, n).astype(float)

        def fit(**kwargs):
            model = SuperGLM(
                family="poisson",
                features={
                    "x1": Spline(kind="cr", n_knots=8),
                    "x2": Spline(kind="cr", n_knots=8),
                },
                **kwargs,
            )
            model.fit_reml(frame, y, runtime_validation="skip")
            return model._reml_result

        exhausted = fit(max_iter=1)
        reference = fit()

        assert exhausted.converged and reference.converged
        # The gate defers the stationary exit until the mode is honest;
        # here the settle hands the exit to the compound criterion.
        assert str(exhausted.termination_reason) in {
            "score_objective_tolerance",
            "active_set_stationary",
        }
        for name in reference.lambdas:
            assert float(exhausted.lambdas[name]) == pytest.approx(
                float(reference.lambdas[name]), rel=1e-2
            )

    def test_the_compound_exit_requires_a_certified_candidate_mode(self):
        """A loose reml_tol accepts only at a certified candidate mode.

        Under the deviance stop, max_pirls_iter=1 with pirls_tol below the
        achievable floor left every candidate uncertified, and the gate
        deferred the compound exit to a warm-started settle.  Every Fisher
        REML PIRLS now stops on the certificate's centred score (one-engine
        design §3.8), which pirls_tol does not move: here one Newton step
        certifies the late candidates, so the accept is a certified one and
        the published lambdas sit within the precision reml_tol=1e-5 asks
        for.  At a stationary accept the gradient is within ``tol (1 + |V|)``
        of zero, so ``rho`` is within that over the criterion's curvature of
        the optimum (the terminal freeze decision's ``hess_diag``); the
        reference fit adds its own, tighter share."""
        rng = np.random.default_rng(9)
        n = 2_000
        frame = pd.DataFrame({"x1": rng.uniform(0, 1, n), "x2": rng.uniform(0, 1, n)})
        eta = (
            0.3
            + 0.8 * np.sin(4.0 * frame["x1"].to_numpy())
            + 0.5 * (frame["x2"].to_numpy() - 0.5) ** 2
        )
        y = rng.poisson(np.exp(eta)).astype(float)

        def fit(**kwargs):
            model = SuperGLM(
                family="poisson",
                features={
                    "x1": Spline(kind="cr", n_knots=8),
                    "x2": Spline(kind="cr", n_knots=8),
                },
            )
            model.fit_reml(frame, y, runtime_validation="skip", **kwargs)
            return model._reml_result

        reference = fit()
        starved_model = SuperGLM(
            family="poisson",
            features={
                "x1": Spline(kind="cr", n_knots=8),
                "x2": Spline(kind="cr", n_knots=8),
            },
        )
        starved_model.fit_reml(
            frame, y, runtime_validation="skip", reml_tol=1e-5, max_pirls_iter=1, pirls_tol=1e-15
        )
        starved = starved_model._reml_result

        assert starved.converged and reference.converged
        assert str(starved.termination_reason) == "score_objective_tolerance"
        decision = starved_model._reml_profile["reml_freeze_decision"]
        for position, name in enumerate(decision["names"]):
            bound = 2.0 * 1e-5 * decision["score_scale"] / decision["hess_diag"][position]
            gap = abs(np.log(starved.lambdas[name]) - np.log(reference.lambdas[name]))
            assert gap <= bound, (name, gap, bound)

    def test_a_mixed_policy_fit_records_the_estimated_status(self):
        """A fixed direction freezes definitionally: its recorded gradient
        is projected to zero while its coupled curvature can exceed the
        bar, so without the estimated flag the published quantities imply
        it should be active. The record carries the flag and the audit
        reconstructs the verdict."""
        from superglm import LambdaPolicy

        rng = np.random.default_rng(9)
        n = 500
        frame = pd.DataFrame({"x1": rng.uniform(0, 1, n), "x2": rng.uniform(0, 1, n)})
        eta = 0.3 + 0.6 * frame["x1"].to_numpy() + 0.4 * np.sin(4.0 * frame["x2"].to_numpy())
        y = rng.poisson(np.exp(eta)).astype(float)
        model = SuperGLM(
            family="poisson",
            features={
                "x1": Spline(
                    kind="cr", n_knots=6, lambda_policy=LambdaPolicy(mode="fixed", value=1.5)
                ),
                "x2": Spline(kind="cr", n_knots=6),
            },
        )
        model.fit_reml(frame, y, runtime_validation="skip")

        freeze = model._reml_profile["reml_freeze_decision"]
        status = dict(zip(freeze["names"], zip(freeze["estimated"], freeze["frozen"])))
        # The fixed policy attaches to the spline's wiggle component.
        assert status["x1:wiggle"] == (False, True)
        assert status["x2"][0] is True


class TestAllFixedLambdaDiagnostics:
    """An all-fixed fit exits before the Newton machinery, but the public
    contract promises the freeze decision on the profile: fixed lambdas
    freeze definitionally (the projection zeroes their scores), and the
    record must exist for that path too."""

    @pytest.mark.parametrize("discrete", [False, True])
    def test_all_fixed_lambdas_still_record_the_freeze_decision(self, discrete):
        from superglm import LambdaPolicy

        rng = np.random.default_rng(9)
        n = 400
        frame = pd.DataFrame({"x": rng.uniform(0.0, 1.0, n)})
        y = rng.poisson(np.exp(0.3 + 0.8 * frame["x"].to_numpy())).astype(float)
        kwargs = {"discrete": True, "n_bins": 32, "selection_penalty": 0} if discrete else {}
        model = SuperGLM(
            family="poisson",
            features={
                "x": Spline(
                    kind="cr", n_knots=6, lambda_policy=LambdaPolicy(mode="fixed", value=1.5)
                )
            },
            **kwargs,
        )
        model.fit_reml(frame, y, runtime_validation="skip")

        assert model._reml_result.termination_reason == "fixed_lambdas"
        freeze = model._reml_profile["reml_freeze_decision"]
        assert set(freeze) == {
            "names",
            "proj_grad",
            "hess_diag",
            "row_curvature",
            "penalty_rank",
            "normalized_curvature",
            "curvature_bar",
            "score_scale",
            "estimated",
            "frozen",
        }
        assert freeze["frozen"] == [True] * len(freeze["names"])
        assert float(freeze["score_scale"]) > 0.0


class TestFlatDirectionFloor:
    """The freeze bar classifies geometry, not precision.

    freeze_tol = 0.1 * reml_tol coupled "is this direction informative"
    (a property of the curvature) to "how precisely locate the optimum".
    Tightening the default to 1e-9 dragged the bar to 1e-10 and un-froze
    the inferentially flat directions the historical 1e-6 default froze at
    1e-7 -- which then march geometrically toward the lambda cap, paying
    8-15 extra iterations, publishing platform-dependent lambda values, and
    exhausting the line search at tight tolerances. Measured separation at
    the endgame: null directions |H_ii|/scale <= 3.3e-9, the tightest
    informative direction 1.5e-6 -- three orders of magnitude around the
    1e-7 floor.
    """

    def test_null_directions_freeze_at_the_default_tolerance(self):
        """benign_3k's x2 smooth is inferentially null (max SE identical at
        every tolerance); it must freeze instead of marching to the cap."""
        rng = np.random.default_rng(5)
        n = 3_000
        frame = pd.DataFrame({"x1": rng.uniform(0, 1, n), "x2": rng.uniform(0, 1, n)})
        eta = 0.4 * np.sin(5.0 * frame["x1"].to_numpy()) + 0.2 * frame["x2"].to_numpy()
        y = rng.poisson(np.exp(eta)).astype(float)

        model = SuperGLM(
            family="poisson",
            features={
                "x1": Spline(kind="cr", n_knots=8),
                "x2": Spline(kind="cr", n_knots=8),
            },
        )
        model.fit_reml(frame, y, runtime_validation="skip", max_reml_iter=200)

        r = model._reml_result
        freeze = model._reml_profile["reml_freeze_decision"]
        frozen = dict(zip(freeze["names"], freeze["frozen"]))
        assert r.converged
        assert frozen["x2"], "the null direction must freeze, not march"
        assert not frozen["x1"]
        # No march: the loose-default iteration count, not 16+.
        assert int(r.n_reml_iter) <= 12
        # The informative lambda is where every tolerance rung puts it.
        assert float(r.lambdas["x1"]) == pytest.approx(0.0809, rel=0.05)

    def test_the_tensor_endgame_no_longer_exhausts_the_line_search(self):
        """tensor_600 at reml_tol=1e-11 previously marched its null margins
        until line_search_failed with converged=False; with the flat
        directions frozen the active set is determined and the fit
        converges cleanly.

        At this tolerance the endgame is decided by the last bits of the
        iterates.  Over seeds 80-111 the one-engine stop rule converges on 31
        of 32 (the stage-1 tree failed seed 104 among 97-104), and removing
        the freeze floor (``FLAT_DIRECTION_FREEZE_FLOOR = 0``) fails 9 of them,
        seed 100 among them: that is the draw pinned here.  The one draw the
        stop rule does not converge (seed 99, this test's former draw) ends a
        different way, not by marching: every trial along the still-moving
        margin, at lambda about 3e8, raises ``PenaltyNumericalError`` (the
        penalty determinant's certified resolution) before its gradient
        reaches the freeze bar.  That exit is recorded as a follow-up."""
        rng = np.random.default_rng(100)
        n = 600
        x1 = rng.uniform(0, 1, n)
        x2 = rng.uniform(0, 1, n)
        eta = 0.5 + np.sin(2 * np.pi * x1) + 0.3 * x2
        y = rng.poisson(np.exp(eta)).astype(float)
        frame = pd.DataFrame({"x1": x1, "x2": x2})

        model = SuperGLM(
            family="poisson",
            features={"x1": Spline(kind="cr", n_knots=6), "x2": Spline(kind="cr", n_knots=6)},
            interactions=[("x1", "x2")],
        )
        model.fit_reml(
            frame,
            y,
            sample_weight=np.ones(n),
            runtime_validation="skip",
            reml_tol=1e-11,
            max_reml_iter=200,
        )

        r = model._reml_result
        assert r.converged
        assert str(getattr(r, "termination_reason", "")) != "line_search_failed"

    @pytest.mark.slow
    def test_large_n_keeps_the_informative_directions_active(self):
        """score_scale = 1+|objective| grows with the row count while
        log-lambda curvature saturates (measured f6: 0.25 at 12k, 0.62 at
        1e6, on the pre-0.29.0 criterion). Judged against score_scale, the
        old bar froze f7 at 400k rows and everything at 1e6 rows by
        iteration 3, publishing lambdas a factor e^5.6 from the floor-off
        optimum with SEs off by up to 87%. The curvature-relative arm keeps
        the bar n-free.

        Re-derived for 0.29.0: the informative_smooths realisation gives the
        two smooths penalty-visible curvature, because the default
        realisation's smooth directions are informative only under the
        reduced Tweedie criterion this release removes (see the fixture's
        docstring; per-dimension curvature measured here 1.3e-1 and 2.6e-1
        against the 1e-3 bar). The lambda pins are the exact-criterion
        determined answer.

        Marked slow for its 400k rows: the n-free bar itself is pinned in
        microseconds by test_reml_convergence.py's
        test_freeze_judges_curvature_relative_to_the_strongest_direction."""
        frame, y, weights, offset, features = _flat_lambda_fixture(
            400_000, informative_smooths=True
        )
        model = SuperGLM(family=families.tweedie(p=1.5), features=features)
        model.fit_reml(frame, y, sample_weight=weights, offset=offset, runtime_validation="skip")

        r = model._reml_result
        freeze = model._reml_profile["reml_freeze_decision"]
        frozen = dict(zip(freeze["names"], freeze["frozen"]))
        assert r.converged
        assert not frozen["f6"]
        assert not frozen["f7"]
        # The determined answer, pinned. Timing/memory/dispatch comparisons
        # live in the complete-fit baseline (PR record), tested
        # separately from numerical correctness per the test policy.
        assert str(r.termination_reason) == "score_objective_tolerance"
        assert float(r.lambdas["f6"]) == pytest.approx(16.928, rel=0.05)
        assert float(r.lambdas["f7"]) == pytest.approx(0.47106, rel=0.05)
        assert int(r.n_reml_iter) <= 25

    def test_a_high_rank_random_effect_does_not_freeze_the_low_rank_spline(self):
        """Row curvature scales with penalty rank (measured: a random
        effect's curvature goes 112 -> 255 -> 391 across 300 -> 600 -> 1000
        levels, ~0.4 per rank, while an informative cr-5 spline holds
        ~2.5). At 600 levels the raw relative bar swallowed the spline --
        real signal, frozen. Per-rank judgment keeps the two commensurate
        at any level count."""
        from superglm import RandomEffect

        rng = np.random.default_rng(17)
        n, n_levels = 100_000, 600
        levels = [f"g{j:03d}" for j in range(n_levels)]
        idx = rng.integers(0, n_levels, n)
        re_effects = rng.normal(0.0, 0.3, n_levels)
        x = rng.uniform(0.0, 1.0, n)
        eta = -0.2 + re_effects[idx] + 0.5 * np.sin(4.0 * x)
        y = rng.poisson(np.exp(eta)).astype(float)
        frame = pd.DataFrame({"g": np.array(levels)[idx], "x": x})

        model = SuperGLM(
            family="poisson",
            features={"g": RandomEffect(), "x": Spline(kind="cr", n_knots=5)},
        )
        model.fit_reml(frame, y, runtime_validation="skip")

        r = model._reml_result
        freeze = model._reml_profile["reml_freeze_decision"]
        frozen = dict(zip(freeze["names"], freeze["frozen"]))
        assert r.converged
        assert not frozen["x"]
        assert float(r.lambdas["x"]) == pytest.approx(0.182, rel=0.25)

    def test_informative_slow_directions_do_not_freeze(self):
        """The tightest informative curvature must stay active: it is exactly
        what the determination work exists to pin, and the floor must not
        freeze it.

        Re-derived for 0.29.0 with the informative_smooths realisation: on
        the default realisation these directions were informative only under
        the reduced Tweedie criterion (see the fixture's docstring). Here f6
        measures 5.9e-3 per penalty dimension against the 1e-3 bar — the
        informative-but-slow band this class characterises as the tightest
        real signal — and f7 5.5e-2."""
        frame, y, weights, offset, features = _flat_lambda_fixture(informative_smooths=True)
        model = SuperGLM(family=families.tweedie(p=1.5), features=features)
        model.fit_reml(frame, y, sample_weight=weights, offset=offset, runtime_validation="skip")

        r = model._reml_result
        freeze = model._reml_profile["reml_freeze_decision"]
        frozen = dict(zip(freeze["names"], freeze["frozen"]))
        assert r.converged
        assert not frozen["f6"]
        assert not frozen["f7"]


class TestSCOPPlateauExit:
    """The EFS plateau exit may not pre-empt an actively contracting fit.

    The plateau road (``obj_rel < 1e-6 and max_change < 0.01``) used fixed
    thresholds with no notion of progress, so it granted ``converged=True``
    at the identical point for reml_tol 1e-6 through 1e-11. Measured on the
    4000-row monotone fixture (2026-08-07): the EFS tail contracts at ratio
    ~0.6 per iteration with lambda still walking one percent per iteration
    -- and the plateau fired mid-walk at iteration 5. Its honest role is
    the step-engine analog of ``converged_at_precision``: classify the
    endgame where steps have STOPPED contracting (noise-floor stall,
    measured ratio ~1.05 on the 400-row variant), never an exit taken
    while iterations are still buying precision.
    """

    @staticmethod
    def _monotone_fixture(n):
        from superglm import Constraint, CubicRegressionSpline

        rng = np.random.default_rng(11)
        x1 = rng.uniform(0, 1, n)
        x2 = rng.uniform(0, 1, n)
        eta = 0.3 + 1.1 * np.log1p(3.0 * x1) + 0.35 * np.sin(2 * np.pi * x2)
        y = rng.poisson(np.exp(eta)).astype(float)
        frame = pd.DataFrame({"x1": x1, "x2": x2})
        features = {
            "x1": Spline(kind="ps", n_knots=8, constraint=Constraint.fit.increasing),
            "x2": CubicRegressionSpline(n_knots=8),
        }
        return frame, y, features

    def test_the_stall_verdict_bounds_the_extrapolated_remaining_movement(self):
        """A ratio just under 1 is still a geometric tail: at r=0.95 with
        max_change at the 0.01 plateau cap, max_change*r/(1-r) says ~19% of
        the lambda movement remains -- 'stalled' may not forfeit that. The
        bound only defers the plateau in the r in [0.9, 1) band; at r >= 1
        no geometric extrapolation exists and the non-contracting steps are
        the machinery noise floor itself, the plateau's honest target."""
        from superglm.reml.scop_efs import _scop_plateau_remaining_movement_bounded

        assert _scop_plateau_remaining_movement_bounded(0.009, 1.05)
        assert not _scop_plateau_remaining_movement_bounded(0.009, 0.95)
        assert _scop_plateau_remaining_movement_bounded(0.002, 0.95)
        assert _scop_plateau_remaining_movement_bounded(0.004, 0.9)
        assert not _scop_plateau_remaining_movement_bounded(0.009, 0.9)

    def test_the_stall_verdict_is_a_bounded_noise_band(self):
        """Two trajectories the consecutive-ratio counter got wrong: an
        oscillating noise floor (1e-5, 2e-5, 1e-5) resets the counter on
        every down-leg and never plateaus, exhausting max_reml_iter on a
        flat fit -- while an expanding tail (0.002, 0.004, 0.008; the
        same-sign adaptive alpha deliberately grows) counts every ratio
        above 0.9 as a stall and plateaus while movement accelerates.
        Stall evidence is a bounded band: the last three accepted steps
        within 2x of their own minimum, with the geometric tail still
        gated by the extrapolated remaining movement."""
        from superglm.reml.scop_efs import _scop_plateau_steps_stalled

        assert _scop_plateau_steps_stalled([1e-5, 2e-5, 1e-5], 0.5)
        assert not _scop_plateau_steps_stalled([0.002, 0.004, 0.008], 2.0)
        assert not _scop_plateau_steps_stalled([0.01, 0.006, 0.0036], 0.6)
        assert not _scop_plateau_steps_stalled([2e-5, 1e-5], 0.5)
        assert not _scop_plateau_steps_stalled([0.01, 0.0095, 0.009], 0.95)
        assert _scop_plateau_steps_stalled([0.002, 0.0019, 0.0018], 0.95)
        # A gradual expander INSIDE the band is still expansion, not noise:
        # every step grew, so movement is accelerating and the stall is
        # deferred until the trajectory turns. An increasing noise window
        # (chance ordering at the floor) is deferred the same single
        # iteration and stalls once it turns.
        assert not _scop_plateau_steps_stalled([0.004, 0.005, 0.006], 1.2)
        assert not _scop_plateau_steps_stalled([1e-5, 1.5e-5, 2e-5], 1.33)
        assert _scop_plateau_steps_stalled([2e-5, 1.9e-5, 1.85e-5], 0.97)
        # A sawtooth expander -- one transient down-step inside the band,
        # then +54% -- is still material recent growth, not a stall.
        assert not _scop_plateau_steps_stalled([0.004, 0.0039, 0.006], 1.54)
        # Persistent 10%-per-iteration growth (+21% across the window) is
        # BELOW the single-window resolution limit: the multi-SCOP cleanup
        # endgame's legitimate limit-cycle stall carries measured growth
        # legs at ratio 1.199 per step, so a bar tight enough to reject
        # this trajectory defers that real stall to max_reml_iter. In the
        # sub-resolution band the remaining-movement cap is the guarantee.
        assert _scop_plateau_steps_stalled([0.004, 0.0044, 0.00484], 1.1)
        # The cleanup cycle's measured stalling window -- post-excursion,
        # growth leg at ratio 1.199 -- stalls; its pure three-leg growth
        # window (net 1.437) correctly defers within the same cycle.
        assert _scop_plateau_steps_stalled([2.628e-4, 1.574e-4, 1.887e-4], 1.199)
        assert not _scop_plateau_steps_stalled([1.054e-4, 1.264e-4, 1.515e-4], 1.199)
        # A single noise up-leg at the measured 1.05 ratio still stalls.
        assert _scop_plateau_steps_stalled([1.9e-5, 1.85e-5, 1.95e-5], 1.05)
        # Equal steps carry 1e-16-relative exp/log jitter that can order
        # itself increasingly; a jitter-increase is a stall, not expansion.
        jittered = [0.0049999999999998969, 0.0049999999999999958, 0.0050000000000000582]
        assert _scop_plateau_steps_stalled(jittered, 1.0)

    def test_the_plateau_does_not_preempt_a_contracting_tail(self):
        """Pre-fix, this fit exited objective_plateau at iteration 5 with
        the lambda still moving a percent per iteration; the gated plateau
        keeps iterating until progress genuinely stops (or the strict road
        is reached), whichever the machinery supports."""
        frame, y, features = self._monotone_fixture(4_000)
        model = SuperGLM(family="poisson", features=features)
        model.fit_reml(frame, y, runtime_validation="skip", reml_tol=1e-9, max_reml_iter=40)

        r = model._reml_result
        assert r.converged
        assert str(r.termination_reason) in {"lambda_tolerance", "objective_plateau"}
        assert int(r.n_reml_iter) > 6

    def test_the_strict_road_wins_when_the_tolerance_is_reachable(self):
        """Exit ordering: a reachable reml_tol terminates as lambda_tolerance,
        not as a plateau classification."""
        frame, y, features = self._monotone_fixture(400)
        model = SuperGLM(family="poisson", features=features)
        model.fit_reml(frame, y, runtime_validation="skip", reml_tol=1e-3)

        r = model._reml_result
        assert r.converged
        assert str(r.termination_reason) == "lambda_tolerance"

    def test_an_unreachable_tolerance_classifies_as_plateau(self, monkeypatch):
        """Below the machinery noise floor (steps stall near 2e-5 on this
        fixture), the honest exit is the plateau classification with
        converged=True -- the step-engine converged_at_precision. The EFS
        step's floor: the Newton step reaches this tolerance on this fixture."""
        import functools

        import superglm.reml.scop_efs as scop_efs

        monkeypatch.setattr(
            scop_efs,
            "optimize_scop_efs_reml",
            functools.partial(scop_efs.optimize_scop_efs_reml, _outer_step="efs"),
        )
        frame, y, features = self._monotone_fixture(400)
        model = SuperGLM(family="poisson", features=features)
        model.fit_reml(frame, y, runtime_validation="skip", reml_tol=1e-11, max_reml_iter=60)

        r = model._reml_result
        assert r.converged
        assert str(r.termination_reason) == "objective_plateau"


class TestPublicationREMLBudget:
    """estimate_p owns its publication refit budget.

    The publication REML refit ran at a fixed max_reml_iter=20 no caller
    could change: passing max_reml_iter into estimate_p died with a
    TypeError inside the search machinery instead of reaching the refit.
    The budget routes to the PUBLICATION refit alone -- candidate search
    fits keep their own loose-bar budget -- and a non-REML publication
    mode refuses it rather than letting it sit inert.
    """

    @pytest.mark.parametrize("search_fit_mode", ["fit", "reml"])
    def test_the_budget_reaches_the_publication_refit(self, search_fit_mode, monkeypatch):
        """The unconverged publication refit is a fit_reml fit the caller
        receives, so it warns once, at the caller, as fit_reml does; the
        search's REML candidates (``search_fit_mode="reml"``, capped here at one
        iteration too) are discarded and stay silent. Mutation check: on
        af53c8d4 the publication installed silently."""
        from superglm import ConvergenceWarning

        candidates = []
        real_fit_reml = SuperGLM.fit_reml

        def capped_candidates(self, *args, **kwargs):
            if getattr(self, "_suppress_convergence_warning", False):
                candidates.append(1)
                kwargs["max_reml_iter"] = 1
            return real_fit_reml(self, *args, **kwargs)

        monkeypatch.setattr(SuperGLM, "fit_reml", capped_candidates)
        frame, y, features = _small_search_fixture()
        model = SuperGLM(family=families.tweedie(p=1.5), features=features)
        with pytest.warns(ConvergenceWarning, match="max_reml_iter") as record:
            result = model.estimate_p(
                frame, y, fit_mode="reml", search_fit_mode=search_fit_mode, max_reml_iter=1
            )

        # One outer iteration can never satisfy the two-evaluation
        # convergence contract: the budget provably bound the refit.
        assert int(model._reml_result.n_reml_iter) == 1
        assert result.converged is False
        disclosed = [w for w in record if issubclass(w.category, ConvergenceWarning)]
        assert len(disclosed) == 1
        assert disclosed[0].filename == __file__
        assert bool(candidates) == (search_fit_mode == "reml")

    @pytest.mark.parametrize("estimate", ["p", "theta"])
    def test_an_unset_budget_resolves_per_engine(self, estimate):
        """An unset budget is ``fit_reml``'s: 100 outer iterations for a SCOP model.

        It read as 20, so the publication refit of a model with a monotone
        (SCOP) term stopped at 20 outer iterations where ``fit_reml`` and the
        search's own REML candidates run to 100, and warned of a cap the
        caller never set; ``estimate_theta`` takes no budget at all and had
        the same 20. Mutation check: 7ed61b05 published 20 for both.
        """
        from superglm import Constraint

        rng = np.random.default_rng(3)
        n = 500
        frame = pd.DataFrame({"x": rng.uniform(0.0, 1.0, n), "z": rng.uniform(0.0, 1.0, n)})
        mean = np.exp(0.2 + 0.8 * frame["x"] + 0.3 * np.sin(6.0 * frame["z"])).to_numpy()
        features = {
            "x": Spline(kind="ps", k=8, constraint=Constraint.fit.increasing),
            "z": Spline(kind="ps", k=8),
        }
        if estimate == "p":
            counts = rng.poisson(mean)
            y = np.array([rng.gamma(2.0, 0.5, count).sum() for count in counts])
            model = SuperGLM(
                family=families.tweedie(p=1.5),
                selection_penalty=0.0,
                discrete=True,
                features=features,
            )
            model.estimate_p(frame, y, fit_mode="reml", search_fit_mode="fit")
        else:
            y = rng.poisson(mean * rng.gamma(2.0, 0.5, n)).astype(float)
            model = SuperGLM(
                family=families.nb2(theta=1.0),
                selection_penalty=0.0,
                discrete=True,
                features=features,
            )
            model.estimate_theta(frame, y, fit_mode="reml")
        assert model.reml_diagnostics()["profile"]["effective_max_reml_iter"] == 100

    def test_a_pure_ml_publication_refuses_the_reml_budget(self):
        frame, y, features = _small_search_fixture()
        model = SuperGLM(family=families.tweedie(p=1.5), features=features)
        with pytest.raises(ValueError, match=r"fit_mode='fit'"):
            model.estimate_p(frame, y, fit_mode="fit", max_reml_iter=5)

    def test_the_budget_rejects_non_integral_counts(self):
        """int() before validation silently truncated 1.9 to one iteration
        and accepted True and '5' as budgets -- a shortened publication
        refit with a changed convergence verdict, not an error. The budget
        is a non-boolean integer via the integer-index protocol."""
        frame, y, features = _small_search_fixture()
        model = SuperGLM(family=families.tweedie(p=1.5), features=features)
        # np.bool_ is not a Python bool but carries __index__ on the
        # supported NumPy floor (1.24); it must not become a one-iteration
        # budget there either.
        for bad in (1.9, True, "5", np.bool_(True)):
            with pytest.raises(ValueError, match="max_reml_iter"):
                model.estimate_p(frame, y, fit_mode="reml", max_reml_iter=bad)

    def test_the_budget_rejects_a_nonpositive_iteration_count(self):
        """max_reml_iter=0 slid through int() into the Newton loop and died
        as an internal RuntimeError; the budget's floor is one iteration
        and the refusal belongs at the API boundary."""
        frame, y, features = _small_search_fixture()
        model = SuperGLM(family=families.tweedie(p=1.5), features=features)
        with pytest.raises(ValueError, match=r"max_reml_iter.*>= 1"):
            model.estimate_p(frame, y, fit_mode="reml", max_reml_iter=0)
        with pytest.raises(ValueError, match=r"max_reml_iter.*>= 1"):
            model.estimate_p(frame, y, fit_mode="reml", max_reml_iter=-3)


def _panel_459(ordering: int | None, *, xshift: float):
    """#439's review panel book (Poisson, 4000 rows, exposure offset).

    ``x`` sits at ``xshift`` plus noise, so with ``xshift=1e4`` the centred
    columns of ``s2`` and ``x:s2`` correlate at 0.99999998. Rows are permuted
    by ``default_rng(ordering).permutation``; None keeps the drawn order.
    """
    rng = np.random.default_rng(4242)
    n = 4000
    s1 = rng.uniform(18, 80, n)
    s2 = rng.gamma(2.0, 3.0, n)
    s3 = rng.uniform(0, 1, n)
    c1 = rng.integers(0, 4, n)
    c2 = rng.integers(0, 3, n)
    make = rng.integers(0, 20, n)
    model_code = make * 8 + rng.integers(0, 8, n)
    region = rng.integers(0, 50, n)
    z = rng.normal(0.0, 1.0, n)
    effects = np.random.default_rng(4242 + 17)
    make_effect = effects.normal(0, 0.25, 20)
    model_effect = effects.normal(0, 0.15, 160)
    region_effect = effects.normal(0, 0.1, 50)
    eta = (
        -1.0
        + 0.4 * np.sin((s1 - 18) / 20)
        + 0.05 * np.log1p(s2)
        + 0.3 * (s3 - 0.5) ** 2
        + 0.1 * c1
        - 0.1 * c2
        + make_effect[make]
        + model_effect[model_code]
        + region_effect[region]
        + 0.15 * z
    )
    offset = np.log(rng.uniform(0.1, 1.0, n))
    y = rng.poisson(np.exp(eta + 0.8 + offset)).astype(float)
    frame = pd.DataFrame(
        {
            "s1": s1,
            "s2": s2,
            "c1": np.array([f"a{v}" for v in c1], dtype=object),
            "make": np.array([f"m{v:02d}" for v in make], dtype=object),
            "x": xshift + z,
        }
    )
    order = np.arange(n) if ordering is None else np.random.default_rng(ordering).permutation(n)
    return frame.iloc[order].reset_index(drop=True), y[order], offset[order]


class TestDeadSearchNewtonDecrement:
    """#459: a dead line search whose Newton model predicts a decrease below
    the stop resolution is a resolved optimum, not a failure."""

    @pytest.mark.parametrize("ordering", [None, 9, 20])
    def test_a_negligible_predicted_decrease_converges_with_unchanged_numbers(
        self, monkeypatch, ordering
    ):
        """Fails unfixed: on 0.37.1 these row orders of the panel model end
        ``line_search_failed`` with ``converged=False`` and a
        ConvergenceWarning (measured on Linux x86-64 OpenBLAS, identical at 1
        and 4 threads; 6 of 22 orders), while the other orders converge to the
        same answer. At iteration 6-8 the active gradient sits 1.009 to 2.03
        times its bar, but the Newton model predicts a decrease of at most
        1.2e-8 against a resolution of 2.06e-6 and evaluation noise of
        4.2e-7, so rounding decides whether the full step is accepted.
        Mutation: dropping the ``predicted_decrease`` arm from
        ``classify_dead_feasible_exit`` fails the same way.

        The second fit runs the 0.37.1 rule in place of the classifier: every
        published number must match bitwise, since the rule only names the
        exit. Where rounding takes another platform's fit through the compound
        criterion instead, both fits converge, the comparison still holds,
        and the test skips: the arm was not exercised there.
        """
        import warnings

        import superglm.reml.direct as direct
        from superglm import Categorical, ConvergenceWarning, Numeric, RandomEffect

        X, y, offset = _panel_459(ordering, xshift=1e4)
        real = direct.classify_dead_feasible_exit

        def gradient_rule_only(*args, **kwargs):
            kwargs.pop("predicted_decrease", None)
            return real(*args, **kwargs)

        def fit():
            model = SuperGLM(
                family="poisson",
                features={
                    "x": Numeric(),
                    "s2": Numeric(),
                    "c1": Categorical(),
                    "s1": Spline(kind="ps", k=8),
                    "make": RandomEffect(),
                },
                interactions=[("x", "s2"), ("x", "c1")],
                discrete=False,
                selection_penalty=0,
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model.fit_reml(X, y, offset=offset)
            return model, [w for w in caught if issubclass(w.category, ConvergenceWarning)]

        fixed, convergence_warnings = fit()
        with monkeypatch.context() as patch:
            patch.setattr(direct, "classify_dead_feasible_exit", gradient_rule_only)
            reference, _ = fit()

        assert fixed._reml_result.converged, fixed._reml_result.termination_reason
        assert fixed.reml_diagnostics()["converged"]
        assert not convergence_warnings
        assert fixed._reml_result.lambdas == reference._reml_result.lambdas
        assert fixed._reml_result.objective == reference._reml_result.objective
        np.testing.assert_array_equal(fixed.result.beta, reference.result.beta)
        np.testing.assert_array_equal(
            fixed.predict(X, offset=offset), reference.predict(X, offset=offset)
        )
        if reference._reml_result.termination_reason != "line_search_failed":
            pytest.skip(
                "the gradient-only rule ended this row order "
                f"{reference._reml_result.termination_reason!r} here, not at a dead line "
                "search, so the decrement arm was not exercised"
            )
        assert fixed._reml_result.termination_reason == "converged_at_precision"

    def test_a_capped_newton_step_withholds_the_decrement(self, monkeypatch):
        """The decrement is the quadratic model's prediction for the Newton
        step; a step longer than the solver's cap of 5 log-lambda units
        extrapolates the model past where the solver trusts it, so the arm is
        withheld (``predicted_decrease=None``).

        Every lambda move is rejected by a stand-in objective, from
        ``lambda=1e-3`` where the first Newton step is capped: the first
        trial sits exactly 5 units from the candidate (measured; the
        uncapped step from ``lambda=0.1`` is 0.48). The search evaluates and
        rejects its trials, so the cap guard alone withholds the decrement.
        Mutation: dropping the ``max_delta > max_newton_step`` guard passes
        the decrement of the capped step instead of None.
        """
        import superglm.reml.direct as direct
        from superglm import ConvergenceWarning

        rng = np.random.default_rng(20260727)
        x = rng.uniform(0.0, 1.0, 240)
        y = rng.poisson(np.exp(0.2 + np.sin(2.0 * np.pi * x))).astype(float)
        evaluated: list[dict[str, float]] = []
        classified: list[dict] = []
        real = direct.classify_dead_feasible_exit

        def reject_every_move(*args, **kwargs):
            evaluated.append(dict(args[6]))
            return 0.0 if evaluated[-1] == evaluated[0] else 1.0

        def spy(*args, **kwargs):
            classified.append(kwargs)
            return real(*args, **kwargs)

        monkeypatch.setattr(direct, "reml_laml_objective", reject_every_move)
        monkeypatch.setattr(direct, "classify_dead_feasible_exit", spy)
        model = SuperGLM(family="poisson", features={"x": Spline(k=7)}, selection_penalty=0)
        with pytest.warns(ConvergenceWarning, match="no smoothing step improved"):
            model.fit_reml(
                pd.DataFrame({"x": x}),
                y,
                lambda2_init={"x": 1e-3},
                max_reml_iter=5,
                runtime_validation="skip",
            )

        first_trial = max(abs(np.log(evaluated[1][k] / evaluated[0][k])) for k in evaluated[0])
        assert first_trial == pytest.approx(5.0, rel=1e-9)
        assert len(classified) == 1
        assert classified[0]["evaluated_trial"] is True
        assert classified[0]["predicted_decrease"] is None
        assert model._reml_result.termination_reason == "line_search_failed"

    def test_an_undetermined_dead_search_still_reports_not_converged(self):
        """#439's documented undetermined model: the panel book under a
        noncanonical sqrt link. Its search dies at the first iteration with
        the active gradient at 9.5 against a bar of 3.3e-4, the Newton step
        over the step cap and no trial objective evaluated (measured), so
        neither arm of the classifier has anything to grant: it stays
        ``line_search_failed`` with ``converged=False`` and warns."""
        from superglm import ConvergenceWarning, Numeric, RandomEffect

        X, y, offset = _panel_459(None, xshift=0.0)
        model = SuperGLM(
            family="poisson",
            link="sqrt",
            features={"x": Numeric(), "s1": Spline(kind="ps", k=8), "make": RandomEffect()},
            selection_penalty=0,
        )
        with pytest.warns(ConvergenceWarning, match="no smoothing step improved"):
            model.fit_reml(X, y, offset=offset)

        assert model._reml_result.termination_reason == "line_search_failed"
        assert not model._reml_result.converged
        assert not model.reml_diagnostics()["converged"]
