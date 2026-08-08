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

import numpy as np
import pandas as pd
import pytest

from superglm import Categorical, OrderedCategorical, Spline, SuperGLM, families
from superglm.model import state_ops

CAT_LEVELS = {"f0": 6, "f1": 9, "f2": 4, "f3": 11, "f4": 13, "f5": 11}
OC_LEVELS = {"f6": 16, "f7": 24}


def _flat_lambda_fixture(n: int = 12_000, seed: int = 4):
    """Tweedie frame whose saturated cr terms leave log-lambda nearly flat.

    On this realisation the exact-Newton optimizer at reml_tol=1e-6 stops five
    iterations before the tight answer, moving a published SE by ~92%.
    """
    rng = np.random.default_rng(seed)
    cols: dict[str, np.ndarray] = {}
    eta = np.full(n, -1.0)
    for name, k in CAT_LEVELS.items():
        levels = [f"{name}_{j:02d}" for j in range(k)]
        idx = rng.integers(0, k, n)
        cols[name] = np.array(levels)[idx]
        eta += rng.normal(0, 0.2, k)[idx]
    orders: dict[str, list[str]] = {}
    for name, k in OC_LEVELS.items():
        levels = [f"{name}_{j:02d}" for j in range(k)]
        idx = rng.integers(0, k, n)
        cols[name] = np.array(levels)[idx]
        eta += 0.02 * (idx - k / 2)
        orders[name] = levels
    frame = pd.DataFrame(cols)
    weights = rng.uniform(1.19e-5, 1.0, n)
    offset = np.where(rng.random(n) < 0.35, 0.0, 1.0986)
    y = np.where(rng.random(n) < 0.83, 0.0, rng.gamma(1.5, np.exp(eta) * 900, n))
    features: dict = {name: Categorical() for name in CAT_LEVELS}
    for name, k in OC_LEVELS.items():
        features[name] = OrderedCategorical(order=orders[name], basis=Spline(kind="cr", k=k))
    return frame, y, weights, offset, features


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


def _small_search_fixture(n: int = 1_200, seed: int = 7):
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


class TestRunnerPathSentinel:
    """The single-pass runner wrapper implements the None convention too.

    No live caller reaches it today, but it is a public-adjacent seam
    (SuperGLM._run_reml_once) and runner.py compares max_change < reml_tol,
    which raises TypeError on None if the sentinel is forwarded verbatim.

    The engine classification follows the stopping criterion, not the inner
    solver: runner.py gates on the per-step lambda change in BOTH branches
    (use_direct only selects the coefficient solver), so the sentinel must
    resolve to the step default regardless of use_direct. Resolving 1e-9
    here would hand a Newton-grade bar to a linear-rate fixed point --
    exactly the combination the engine scoping exists to prevent.
    """

    def test_model_run_reml_once_resolves_the_sentinel_as_a_step_engine(self, monkeypatch):
        import types

        from superglm.model import reml_ops

        captured: list[float] = []

        def fake_run_reml_once(*args, **kwargs):
            captured.append(kwargs["reml_tol"])
            return "RESULT", "DM"

        monkeypatch.setattr(reml_ops, "run_reml_once", fake_run_reml_once)
        monkeypatch.setattr(reml_ops, "configured_penalty", lambda model: None)
        model = types.SimpleNamespace(
            _dm="DM0",
            _distribution=None,
            _link=None,
            _groups=[],
            _active_set=None,
            _direct_solve="auto",
            _reml_penalties=None,
        )
        for use_direct, reml_tol, expected in (
            (True, None, 1e-6),
            (False, None, 1e-6),
            (True, 3e-7, 3e-7),
        ):
            reml_ops.model_run_reml_once(
                model,
                None,
                None,
                None,
                [],
                {},
                {},
                max_reml_iter=1,
                reml_tol=reml_tol,
                verbose=False,
                use_direct=use_direct,
            )
        assert captured == [1e-6, 1e-6, 3e-7]


class TestEngineSeamSentinels:
    """Every model-bound engine wrapper resolves the sentinel for the engine
    it actually forwards to.

    These are the same public-adjacent seam class as the runner wrapper:
    each forwards to exactly one engine whose loop would TypeError on a
    verbatim None (discrete.py floats the tolerance; efs.py compares
    against it), so the wrapper owns the resolution.
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
    def test_discrete_publication_profiles_phi_at_the_public_mean(self):
        """On a binned model the internal design's mean is an approximation;
        the published dispersion must be profiled at the mean callers get
        from predict(), not at the binned matvec."""
        from superglm import Spline as _Spline
        from superglm.profiling.tweedie import _profile_phi_detailed

        rng = np.random.default_rng(11)
        n = 4_000
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
        edf = float(model.result.effective_df)
        # Warm-start from the SEARCH winner's phi, not from the published
        # answer: starting at result.phi_hat only proves the answer is a
        # stationary point; starting where the old code would have published
        # from proves the re-profile moved to the published fit's optimum.
        trace = result.search_trace
        gap = (trace["p"] - float(result.p_hat)).abs()
        search_phi = float(trace.loc[gap.idxmin(), "phi"])
        oracle = _profile_phi_detailed(
            np.asarray(y, dtype=float),
            mu,
            float(result.p_hat),
            weights=np.ones(n),
            df_resid=max(float(n) - edf, 1.0),
            phi_method="mle",
            phi_start=search_phi,
        )

        assert float(result.phi_hat) == pytest.approx(float(oracle.phi), rel=1e-8)

    def test_discrete_publication_stats_describe_the_public_mean(self):
        """One published fit, one mean: the summary statistics must be
        computed at the same public mean the published dispersion was
        profiled at, not left as a hybrid of public-mean phi inside
        binned-mean likelihood/deviance."""
        from superglm import Spline as _Spline
        from superglm.model.fit_ops import _compute_fit_stats, _compute_null_mu

        rng = np.random.default_rng(11)
        n = 4_000
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

        mu_pub = np.asarray(model.predict(frame), dtype=float)
        y_arr = np.asarray(y, dtype=float)
        ones = np.ones(n)
        null_mu = _compute_null_mu(y_arr, ones, None, model._distribution, model._link)
        oracle = _compute_fit_stats(
            y_arr,
            mu_pub,
            ones,
            None,
            model._distribution,
            model._link,
            float(result.phi_hat),
            null_mu=null_mu,
        )

        np.testing.assert_allclose(model._fit_mu, mu_pub, rtol=0, atol=0)
        assert model._fit_stats.log_likelihood == pytest.approx(oracle.log_likelihood, rel=1e-12)
        assert model._fit_stats.pearson_chi2 == pytest.approx(oracle.pearson_chi2, rel=1e-12)
        assert model._fit_stats.explained_deviance == pytest.approx(
            oracle.explained_deviance, rel=1e-12
        )

    def test_the_aggregate_judges_the_publication_refit_not_the_candidate(self, monkeypatch):
        """fit/solver/reml convergence on the result must describe the
        publication refit. Candidates run at the loose search bar and the
        publication runs tight, so the two can disagree on exactly the
        flat-lambda designs the tolerance split was built for -- and the
        candidate's green flags must not mask a stalled publication."""

        from superglm.model import profile_ops

        frame, y, features = _small_search_fixture()
        model = SuperGLM(family=families.tweedie(p=1.5), features=features)
        result = model.estimate_p(frame, y, fit_mode="reml")
        assert result.converged and result.fit_converged
        assert result.search_fit_converged is True

        model._reml_result = _replace_dc(model._reml_result, converged=False)
        mu = np.asarray(model.predict(frame), dtype=float)
        profile_ops._reprofile_published_dispersion(
            model, np.asarray(y, dtype=float), np.ones(len(y)), mu, result, "mle"
        )

        assert result.reml_converged is False
        assert result.fit_converged is False
        assert result.converged is False
        # The searched winner's flag survives for the CI guard.
        assert result.search_fit_converged is True

    def test_a_decoupled_publication_reports_its_reml_convergence(self):
        """The docstring promises fit_converged covers REML and solver
        convergence for a REML publication. A decoupled run's search is ML
        (reml_converged=None there), but it publishes a REML fit -- the
        published flags must describe that fit, not the ML candidates."""
        frame, y, features = _small_search_fixture()
        model = SuperGLM(family=families.tweedie(p=1.5), features=features)
        result = model.estimate_p(frame, y, fit_mode="reml", search_fit_mode="fit")

        assert result.reml_converged is True
        assert result.fit_converged is True
        assert result.search_fit_converged is True
        assert result.converged is True

    def test_a_troubled_reprofile_is_disclosed_on_the_result(self, monkeypatch):
        """A boundary, fallback, or non-convergent published re-profile must
        not hide behind the search's clean record."""
        from dataclasses import replace as _replace

        import superglm.profiling.tweedie as tweedie_module
        from superglm.model import profile_ops

        frame, y, features = _small_search_fixture()
        model = SuperGLM(family=families.tweedie(p=1.5), features=features)
        result = model.estimate_p(frame, y, fit_mode="reml")
        baseline = len(result.warnings)

        real = tweedie_module._profile_phi_detailed

        def troubled(*args, **kwargs):
            return _replace(
                real(*args, **kwargs),
                converged=False,
                used_fallback=True,
                message="forced for the test",
            )

        monkeypatch.setattr(tweedie_module, "_profile_phi_detailed", troubled)
        mu = np.asarray(model.predict(frame), dtype=float)
        profile_ops._reprofile_published_dispersion(
            model, np.asarray(y, dtype=float), np.ones(len(y)), mu, result, "mle"
        )

        assert not result.phi_converged
        assert result.phi_used_fallback
        assert len(result.warnings) == baseline + 1
        assert "re-profile" in result.warnings[-1]

    def test_the_searched_density_provenance_survives_publication(self):
        """p_hat, search_nll, plots and the profile CI come from the
        SEARCHED curve. When the search scored its winner with saddlepoint
        density but the publication re-profile evaluates exactly, replacing
        the density story wholesale would label an approximation-based
        power estimate as exact -- the searched provenance survives beside
        the published one, disclosed in warnings."""
        from superglm.model import profile_ops

        frame, y, features = _small_search_fixture()
        model = SuperGLM(family=families.tweedie(p=1.5), features=features)
        result = model.estimate_p(frame, y, fit_mode="reml")

        # Rewind to first-reprofile state with a saddlepoint-scored search.
        result.search_nll = None
        result.density_method = "saddlepoint"
        result.density_exact = False
        result.saddlepoint_fraction = 0.4
        mu = np.asarray(model.predict(frame), dtype=float)
        profile_ops._reprofile_published_dispersion(
            model, np.asarray(y, dtype=float), np.ones(len(y)), mu, result, "mle"
        )

        # The live fields describe the published dispersion...
        assert result.density_exact is True
        # ...while the searched story survives beside it.
        assert result.search_density_method == "saddlepoint"
        assert result.search_density_exact is False
        assert result.search_saddlepoint_fraction == pytest.approx(0.4)
        assert any("saddlepoint" in w and "search" in w for w in result.warnings), (
            result.warnings
        )

    def test_a_reprofile_rewrites_the_whole_dispersion_story(self, monkeypatch):
        """The re-profile IS the published dispersion, so the aggregate
        convergence flag, the density classification and the phi warnings
        must all describe it -- not the search winner it replaced."""
        from dataclasses import replace as _replace

        import superglm.profiling.tweedie as tweedie_module
        from superglm.model import profile_ops
        from superglm.profiling.tweedie import _TweedieLogpdfDiagnostics

        frame, y, features = _small_search_fixture()
        model = SuperGLM(family=families.tweedie(p=1.5), features=features)
        result = model.estimate_p(frame, y, fit_mode="reml")
        assert result.converged and result.phi_converged

        # Stale entries from the search winner's phi: the rebuild must
        # remove them, not stack publication entries on top of them.
        result.warnings = list(result.warnings) + [
            "Winning inner phi profile did not converge.",
            "Winning phi estimate is at the lower dispersion boundary.",
        ]

        real = tweedie_module._profile_phi_detailed

        def troubled(*args, **kwargs):
            return _replace(
                real(*args, **kwargs),
                converged=False,
                diagnostics=_TweedieLogpdfDiagnostics(n_positive=7, n_saddlepoint=7),
                message="forced for the test",
            )

        monkeypatch.setattr(tweedie_module, "_profile_phi_detailed", troubled)
        mu = np.asarray(model.predict(frame), dtype=float)
        with pytest.warns(UserWarning):
            profile_ops._reprofile_published_dispersion(
                model, np.asarray(y, dtype=float), np.ones(len(y)), mu, result, "mle"
            )

        # The aggregate flag is recomputed from the published dispersion.
        assert not result.phi_converged
        assert not result.converged
        assert result.converged == (
            result.objective_finite
            and result.outer_converged
            and result.fit_converged
            and result.phi_converged
        )
        # The density block classifies the re-profile's own evaluation.
        assert result.density_method == "saddlepoint"
        assert result.density_exact is False
        assert result.n_saddlepoint == 7 and result.n_positive == 7
        assert result.saddlepoint_fraction == pytest.approx(1.0)
        # Search-phi warnings are gone; the entries describe the re-profile.
        assert "Winning inner phi profile did not converge." not in result.warnings
        assert not any(w.startswith("Winning phi estimate is at the ") for w in result.warnings)
        assert any("Saddlepoint approximation used for 7/7" in w for w in result.warnings)
        assert any("re-profile did not converge" in w for w in result.warnings)

    def test_a_published_boundary_dispersion_is_disclosed(self, monkeypatch):
        """The published dispersion landing on the hard phi bound must not
        be silent: label recomputed AND a warning entry on the result."""
        from dataclasses import replace as _replace

        import superglm.profiling.tweedie as tweedie_module
        from superglm.model import profile_ops

        frame, y, features = _small_search_fixture()
        model = SuperGLM(family=families.tweedie(p=1.5), features=features)
        result = model.estimate_p(frame, y, fit_mode="reml")
        assert result.phi_boundary == ""

        real = tweedie_module._profile_phi_detailed

        def pinned(*args, **kwargs):
            return _replace(real(*args, **kwargs), lower_boundary=True)

        monkeypatch.setattr(tweedie_module, "_profile_phi_detailed", pinned)
        mu = np.asarray(model.predict(frame), dtype=float)
        profile_ops._reprofile_published_dispersion(
            model, np.asarray(y, dtype=float), np.ones(len(y)), mu, result, "mle"
        )

        assert result.phi_boundary == "lower"
        assert any("dispersion boundary" in w for w in result.warnings)

    def test_coupled_publication_profiles_phi_against_the_published_fit(self):
        """The published phi must describe the published fit, not the candidate.

        Candidates run at the search tolerance; the publication refit runs
        tight. Carrying the candidate's phi onto the tight refit scales every
        published SE by sqrt(phi) of the wrong fit (4.7e-6 here, 2.3e-3 on the
        12k stress realisation).
        """
        from superglm.profiling.tweedie import _profile_phi_detailed

        frame, y, weights, offset, features = _flat_lambda_fixture(n=6_000)
        model = SuperGLM(family=families.tweedie(p=1.5), features=features)
        result = model.estimate_p(frame, y, sample_weight=weights, offset=offset, fit_mode="reml")

        mu = np.asarray(model.predict(frame, offset=offset), dtype=float)
        edf = float(model.result.effective_df)
        # Warm-start from the search winner's phi (what the old code would
        # have published), so the oracle is independent of the answer.
        trace = result.search_trace
        gap = (trace["p"] - float(result.p_hat)).abs()
        search_phi = float(trace.loc[gap.idxmin(), "phi"])
        oracle = _profile_phi_detailed(
            np.asarray(y, dtype=float),
            mu,
            float(result.p_hat),
            weights=np.asarray(weights, dtype=float),
            df_resid=max(float(len(y)) - edf, 1.0),
            phi_method="mle",
            phi_start=search_phi,
        )

        assert float(result.phi_hat) == pytest.approx(float(oracle.phi), rel=1e-8)
        # The searched objective's value survives for the CI and the plots,
        # and the published nll refers to the published dispersion.
        assert result.search_nll is not None
        assert model.result.phi == pytest.approx(float(result.phi_hat), rel=1e-12)


class TestSearchPublishSplit:
    def test_coupled_candidates_run_loose_and_the_publication_runs_tight(self, monkeypatch):
        """Candidate fits rank powers; only the published refit pays for
        determination. Every optimizer call before the last must carry the
        loose search tolerance, and the last -- the publication fit at p_hat
        -- the tight default."""
        seen = _spy_optimizer_tols(monkeypatch)
        frame, y, features = _small_search_fixture()

        model = SuperGLM(family=families.tweedie(p=1.5), features=features)
        model.estimate_p(frame, y, fit_mode="reml")

        assert len(seen) >= 3
        assert set(seen[:-1]) == {1e-6}
        assert seen[-1] == 1e-9

    def test_decoupled_publication_is_the_only_reml_fit_and_is_tight(self, monkeypatch):
        seen = _spy_optimizer_tols(monkeypatch)
        frame, y, features = _small_search_fixture()

        model = SuperGLM(family=families.tweedie(p=1.5), features=features)
        model.estimate_p(frame, y, fit_mode="reml", search_fit_mode="fit")

        assert seen == [1e-9]

    def test_ci_guard_judges_the_searched_winner_not_the_reprofile(self):
        """ci() inverts the searched curve, so its guard must read the
        searched winner's certification flags. The publication re-profile
        overwrites objective_finite/phi_converged with its own dispersion
        status; judging those would refuse a clean search because a
        publication re-profile stalled on a curve the interval never
        touches -- and accept a stalled search whenever the publication
        re-profile happens to converge."""
        frame, y, features = _small_search_fixture()
        model = SuperGLM(family=families.tweedie(p=1.5), features=features)
        result = model.estimate_p(frame, y, fit_mode="reml", search_fit_mode="fit")

        assert result.search_objective_finite is True
        assert result.search_phi_converged is True

        # A troubled PUBLICATION re-profile must not refuse an interval on
        # the clean searched curve.
        result.phi_converged = False
        result.objective_finite = False
        lo, hi = result.ci()
        assert lo < float(result.p_hat) < hi

        # A troubled SEARCH winner must refuse, however clean the
        # publication re-profile looks.
        result.phi_converged = True
        result.objective_finite = True
        result.search_phi_converged = False
        with pytest.raises(RuntimeError, match="phi_converged"):
            result.ci(alpha=0.10)


class TestCertificationBar:
    def test_the_bar_is_one_fixed_number_with_nothing_to_move_it(self):
        """The certification bar must be the same fixed number at the
        candidate gate and the terminal publication refit. The error message
        promises that changing `tol` cannot move the bar; a terminal
        expression reading min(pirls_tol, ceiling) breaks that promise below
        the ceiling, where a point that certified as a candidate at 1e-9
        fails publication solely because the caller tightened pirls_tol."""
        import inspect

        from superglm.reml.observed_geometry import observed_mode_certification_bar

        bar = observed_mode_certification_bar()
        assert bar == max(1e-9, 100.0 * np.finfo(float).eps)
        assert not inspect.signature(observed_mode_certification_bar).parameters
