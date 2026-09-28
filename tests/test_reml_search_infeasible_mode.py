"""A candidate power whose penalized mode does not converge must be routed
around by the power search, not raised out of it.

``optimize_direct_reml`` reports "this power has no usable penalized mode" two
different ways from the same loop: ``ObservedModeNotCertifiedError`` when the
mode is found but cannot be differentiated through, and a bare ``RuntimeError``
when PIRLS did not converge to a mode at all. The Tweedie power search catches
only the first. The second escapes and kills the whole search -- from the
bracket endpoint p=1.95, which the search probes second and never selects.

The natural-failure test below is sized so the failure lands on a power the
search only probes, never returns. Tests that inject their failure run on a
small frame instead: the data cannot matter to them, and a frame with walls of
its own would let a natural failure stand in for the injected one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from superglm import Spline, SuperGLM, families
from superglm.reml.observed_geometry import ObservedModeNotCertifiedError

from ._tweedie_profile_fixtures import flat_lambda_fixture as _fixture
from ._tweedie_profile_fixtures import search_fixture


def _plumbing_fixture():
    """A small frame for tests whose infeasible powers are injected.

    Searches over [1.05, 1.95] on it meet no uncertifiable power (measured at
    600 and 1,200 rows), so the only walls a test sees are the ones it made.
    """
    frame, y, features = search_fixture(n=600)
    return frame, y, None, None, features


def _model(features, p=1.5):
    return SuperGLM(family=families.tweedie(p=p), features=features)


def test_bracket_endpoint_without_a_converged_mode_is_routed_around(recwarn, monkeypatch):
    """The search must survive a probe power whose penalized mode fails.

    One realisation carries three claims about the one coupled search: it
    routes around the wall, it lands where the ML-mode search that never fits
    REML at the wall lands, and it does not warn about censoring merely
    because a wall exists far from p_hat.
    """
    frame, y, weights, offset, features = _fixture(5_000)

    # This realisation used to meet a natural wall at p=1.95 -- the second
    # point Brent probes -- scoring 6.7e-5 against the 1e-9 bar: Fisher
    # scoring stopped short of the mode. Observed-Newton PIRLS now certifies
    # it (test_tweedie_p_recovery pins that), so the wall is injected at the
    # same power.
    real_fit_reml = SuperGLM.fit_reml

    def wall_from_195(self, X, yv, **kwargs):
        if float(getattr(self.family, "p", 0.0)) >= 1.95:
            raise ObservedModeNotCertifiedError(6.7e-5, 1e-9)
        return real_fit_reml(self, X, yv, **kwargs)

    monkeypatch.setattr(SuperGLM, "fit_reml", wall_from_195)

    coupled = _model(features).estimate_p(
        frame, y, sample_weight=weights, offset=offset, fit_mode="reml"
    )
    assert 1.05 < float(coupled.p_hat) < 1.95
    assert np.isfinite(float(coupled.phi_hat))

    # The fail-closed benchmark treats any warning as a failure, so censoring
    # must not fire merely because an infeasible power exists. The wall has to
    # be in the search's own record for that to be tested at all: the 6000-row
    # version of this check met no wall and passed with the distance test gone.
    assert np.isinf(coupled.evaluations.set_index("p").loc[1.95, "nll"])
    assert not [w for w in coupled.warnings if "censored" in w]
    assert not [w for w in recwarn.list if "censored" in str(w.message)]

    # The ML-mode search never fits REML at the failing power, so it completes
    # without routing; the coupled search must route around the wall to reach
    # the same answer.
    decoupled = _model(features).estimate_p(
        frame,
        y,
        sample_weight=weights,
        offset=offset,
        fit_mode="reml",
        search_fit_mode="fit",
    )
    assert 1.05 < float(decoupled.p_hat) < 1.95
    assert float(coupled.p_hat) == pytest.approx(float(decoupled.p_hat), rel=1e-2)


class TestTypedModeFailureContract:
    """Every no-usable-mode raise shares one routable type family."""

    def test_not_converged_routes_through_the_certification_catch(self):
        """A handler written for the certification failure must also see this.

        The two conditions -- mode found but uncertifiable, and no mode found
        at all -- are one physical situation to a power search. Subclassing is
        what guarantees no future catch site handles one and crashes on the
        other; a blanket ``except RuntimeError`` is not an option because
        optimize_direct_reml raises bare RuntimeError for genuine invariant
        violations that must propagate.
        """
        from superglm.reml.observed_geometry import (
            ObservedModeNotCertifiedError,
            ObservedModeNotConvergedError,
        )

        assert issubclass(ObservedModeNotConvergedError, ObservedModeNotCertifiedError)
        exc = ObservedModeNotConvergedError()
        assert "converged penalized coefficient mode" in str(exc)
        # No mode exists, so no score was achieved: the attributes the parent's
        # handlers format must exist and be safely non-finite.
        assert not np.isfinite(exc.relative_max)

    def test_an_injected_unconverged_mode_is_scored_infeasible(self, monkeypatch):
        """The search must route around the new type wherever it is raised.

        The terminal-refit raise sites in reml_finalize.py cannot be reached on
        a small fixture without loosening the certification gate, so the
        routing contract is pinned by injection: any candidate fit that raises
        the typed error is an infeasible point, not a dead search.
        """
        from superglm.reml.observed_geometry import ObservedModeNotConvergedError

        frame, y, weights, offset, features = _plumbing_fixture()
        real_fit_reml = SuperGLM.fit_reml

        def failing_above_19(self, X, yv, **kwargs):
            if float(getattr(self.family, "p", 0.0)) > 1.9:
                raise ObservedModeNotConvergedError()
            return real_fit_reml(self, X, yv, **kwargs)

        monkeypatch.setattr(SuperGLM, "fit_reml", failing_above_19)

        result = _model(features).estimate_p(
            frame, y, sample_weight=weights, offset=offset, fit_mode="reml"
        )

        assert float(result.p_hat) < 1.9


class TestPublicationModeFailure:
    """A publish refit that cannot certify must explain itself."""

    def test_the_typed_error_is_public_api(self):
        """The docstrings and the guide promise a typed routable error; a
        caller must be able to import it without reaching into a private
        module path."""
        import superglm
        from superglm.model import profile_ops

        assert superglm.PublicationModeError is profile_ops.PublicationModeError
        assert "PublicationModeError" in superglm.__all__

    def test_the_coupled_failure_does_not_recommend_itself(self):
        """A coupled caller already passed fit_mode='reml'; the options list
        must offer only the ways out that remain, while the decoupled branch
        keeps recommending the coupled search."""
        from superglm.model import profile_ops

        coupled = str(
            profile_ops._publication_mode_failure(
                RuntimeError("score 3.9e-8 exceeds bar"),
                parameter="p",
                value=1.61,
                decoupled=False,
            )
        )
        assert "fit_mode='reml' searches only certifiable points" not in coupled
        assert "fit_mode='fit'" in coupled
        assert "p_bounds" in coupled

        decoupled = str(
            profile_ops._publication_mode_failure(
                RuntimeError("score 3.9e-8 exceeds bar"),
                parameter="p",
                value=1.61,
                decoupled=True,
            )
        )
        assert "fit_mode='reml' searches only certifiable points" in decoupled

    def test_publish_failure_names_the_power_and_the_options(self, monkeypatch):
        from superglm.model import fit_ops
        from superglm.reml.observed_geometry import ObservedModeNotCertifiedError

        frame, y, weights, offset, features = _fixture(3_000)

        def refuse(*args, **kwargs):
            raise ObservedModeNotCertifiedError(3.9e-8, 1e-9)

        monkeypatch.setattr(fit_ops, "_fit_reml_in_workspace", refuse)

        with pytest.raises(RuntimeError) as excinfo:
            _model(features).estimate_p(
                frame,
                y,
                sample_weight=weights,
                offset=offset,
                fit_mode="reml",
                search_fit_mode="fit",
            )

        message = str(excinfo.value)
        # It must name the power it failed at and every real way out.
        assert "p=" in message
        assert "search_fit_mode" in message or "fit_mode='fit'" in message
        assert "p_bounds" in message

    def test_publish_failure_is_a_typed_routable_error(self, monkeypatch):
        """Callers route the recoverable certifiability condition by type.

        RuntimeError compatibility is kept for pre-existing broad handlers,
        and the certification detail that caused the failure stays chained.
        """
        from superglm.model import fit_ops, profile_ops
        from superglm.reml.observed_geometry import ObservedModeNotCertifiedError

        frame, y, weights, offset, features = _fixture(3_000)

        def refuse(*args, **kwargs):
            raise ObservedModeNotCertifiedError(3.9e-8, 1e-9)

        monkeypatch.setattr(fit_ops, "_fit_reml_in_workspace", refuse)

        with pytest.raises(profile_ops.PublicationModeError) as excinfo:
            _model(features).estimate_p(
                frame,
                y,
                sample_weight=weights,
                offset=offset,
                fit_mode="reml",
                search_fit_mode="fit",
            )

        assert isinstance(excinfo.value, RuntimeError)
        assert isinstance(excinfo.value.__cause__, ObservedModeNotCertifiedError)

    def test_theta_publish_failure_names_theta_controls(self, monkeypatch):
        """The theta search is alternating ML fits, not REML certification;
        its failure guidance must name theta's actual controls, not p's."""
        from superglm.distributions import NegativeBinomial
        from superglm.model import fit_ops
        from superglm.reml.observed_geometry import ObservedModeNotCertifiedError

        frame, _, weights, offset, features = _fixture(3_000)
        rng = np.random.default_rng(9)
        counts = rng.poisson(1.2, len(frame)).astype(float)

        def refuse(*args, **kwargs):
            raise ObservedModeNotCertifiedError(3.9e-8, 1e-9)

        monkeypatch.setattr(fit_ops, "_fit_reml_in_workspace", refuse)

        model = SuperGLM(family=NegativeBinomial(theta="auto"), features=features)
        with pytest.raises(RuntimeError) as excinfo:
            model.estimate_theta(frame, counts, fit_mode="reml")

        message = str(excinfo.value)
        assert "theta=" in message
        assert "theta_bounds" in message
        assert "p_bounds" not in message


class TestBoundaryCensoringWarning:
    """A p_hat pinned against the certifiable boundary is disclosed."""

    def test_an_optimum_pinned_against_a_wall_is_censored(self, monkeypatch):
        """The profile beyond an infeasible neighbour is unknown, so p_hat is censored.

        The fixture's REML optimum is near 1.48; a wall above 1.3 leaves the
        search's best feasible power next to an infeasible one.
        """
        from superglm.reml.observed_geometry import ObservedModeNotConvergedError

        frame, y, weights, offset, features = _plumbing_fixture()
        real_fit_reml = SuperGLM.fit_reml

        def failing_above_13(self, X, yv, **kwargs):
            if float(getattr(self.family, "p", 0.0)) > 1.3:
                raise ObservedModeNotConvergedError()
            return real_fit_reml(self, X, yv, **kwargs)

        monkeypatch.setattr(SuperGLM, "fit_reml", failing_above_13)
        with pytest.warns(UserWarning, match="censored estimate") as raised:
            result = _model(features).estimate_p(
                frame, y, sample_weight=weights, offset=offset, fit_mode="reml"
            )

        assert float(result.p_hat) <= 1.3
        assert any("censored estimate" in warning for warning in result.warnings)
        # Raised at the caller's line, not inside superglm.
        censored = [w for w in raised if "censored estimate" in str(w.message)]
        assert censored[0].filename == __file__
        # So is a censored side of the interval.
        with pytest.warns(UserWarning, match="interval for p is censored"):
            result.interval(0.05)


class TestCIAtTheCertifiabilityWall:
    """The interval treats uncertifiable powers as the profile's boundary.

    Exercised on a synthetic deterministic objective: the wall logic is pure
    and pins exactly here. The upper-wall pair is
    test_profile_scalar::test_interval_side_ending_at_infeasible_region_is_censored
    and ::test_a_crossing_just_before_an_infeasible_region_is_not_censored.
    """

    @staticmethod
    def _interval(wall: float):
        from superglm.profiling._scalar import RecordedObjective, likelihood_ratio_interval

        p_hat = 1.5

        def objective(p: float) -> float:
            # LR = 2 * 1000 * (nll - nll_hat) crosses the 95% cutoff (3.841)
            # at |p - p_hat| ~ 0.05.
            return np.inf if p < wall else 1.0 + 0.768 * (p - p_hat) ** 2

        recorded = RecordedObjective(objective)
        return likelihood_ratio_interval(
            recorded, p_hat, recorded(p_hat), (1.05, 1.95), alpha=0.05, scale=1000.0, xtol=1e-4
        )[0]

    def test_a_lower_wall_is_bisected_not_left_at_a_scan_point(self):
        """Powers DECREASE toward a lower wall; the censored end is the wall itself."""
        interval = self._interval(wall=1.48)

        assert interval.lower_censored
        assert 1.48 <= interval.lower <= 1.48 + 1e-4
        assert not interval.upper_censored
        assert interval.upper == pytest.approx(1.55, abs=2e-3)

    def test_a_crossing_above_a_lower_wall_still_roots_normally(self):
        interval = self._interval(wall=1.40)

        assert not interval.lower_censored
        assert interval.lower == pytest.approx(1.45, abs=2e-3)


class TestSCOPModeFailuresAreRoutable:
    def test_a_scop_constrained_profile_search_completes(self):
        """SCOP mode failures now raise the routable typed family instead of
        bare RuntimeError, so a coupled search over a SCOP-constrained model
        scores a failing candidate infeasible rather than dying. The healthy
        path is pinned end-to-end here; the routing of the typed family is
        pinned by the injection tests above."""
        from superglm import Constraint

        rng = np.random.default_rng(31)
        n = 900
        x = rng.uniform(0.0, 1.0, n)
        eta = 0.9 * x - 0.6
        y = np.where(rng.random(n) < 0.4, 0.0, rng.gamma(1.2, np.exp(eta) * 2.0, n))
        frame = pd.DataFrame({"x": x})
        features = {"x": Spline(kind="ps", n_knots=8, constraint=Constraint.fit.increasing)}

        model = SuperGLM(family=families.tweedie(p=1.5), features=features)
        result = model.estimate_p(frame, y, fit_mode="reml")

        assert 1.05 < float(result.p_hat) < 1.95
        assert np.isfinite(float(result.phi_hat))

    def test_the_scop_certification_raise_is_typed_and_scored(self):
        """The SCOP latent-mode certification failure carries its achieved
        score like the observed-geometry gates, so a search formats a real
        infeasibility reason instead of 'no mode found'."""
        import superglm.reml.scop_efs as scop_module
        from superglm.reml.observed_geometry import ObservedModeNotCertifiedError

        assert scop_module.ObservedModeNotCertifiedError is ObservedModeNotCertifiedError

    def test_the_certification_failure_reports_the_metric_that_failed(self):
        """The failing condition is mode_newton_relative > mode_tolerance,
        but the exception carried mode_score.relative_max -- a deliberately
        distinct metric. An ill-conditioned mode can hold a sub-threshold
        componentwise score with an excessive Newton correction, so the
        search's infeasibility reason and PublicationModeError claimed a
        score that never exceeded the bar."""
        from superglm.reml.scop_efs import _scop_certification_failure

        exc = _scop_certification_failure(3.2e-5, 1e-8, 4.1e-12)

        assert exc.relative_max == pytest.approx(3.2e-5)
        assert exc.tolerance == pytest.approx(1e-8)
        assert "componentwise" in exc.hint
        assert "4.1e-12" in exc.hint


class TestQPModeFailuresAreRoutable:
    def test_the_qp_infeasibility_raise_is_typed_and_routable(self, monkeypatch):
        """A QP candidate with no feasible mode (or no complete KKT
        certificate) is this point's infeasibility, not a dead search:
        both constraint terminations raise the routable typed family,
        exactly like the SCOP and observed-geometry gates. A bare
        RuntimeError here aborted an entire coupled power search on one
        bad candidate."""
        from types import SimpleNamespace

        import superglm.model.reml_execute as reml_execute
        from superglm import Constraint
        from superglm.reml.observed_geometry import ObservedModeNotCertifiedError

        rng = np.random.default_rng(3)
        n = 400
        x = rng.uniform(0, 1, n)
        y = rng.poisson(np.exp(0.4 + 0.9 * x)).astype(float)
        frame = pd.DataFrame({"x": x})
        model = SuperGLM(
            family="poisson",
            features={"x": Spline(kind="cr", n_knots=6, constraint=Constraint.fit.increasing)},
        )
        model.fit_reml(frame, y, runtime_validation="skip")
        lambdas = {k: float(v) for k, v in model._reml_result.lambdas.items()}

        for reason in ("constraint_infeasible", "constraint_kkt_incomplete"):
            monkeypatch.setattr(
                reml_execute,
                "fit_irls_direct",
                lambda *a, _reason=reason, **k: (
                    SimpleNamespace(termination_reason=_reason),
                    None,
                ),
            )
            with pytest.raises(ObservedModeNotCertifiedError):
                reml_execute.run_fixed_monotone_reml(
                    model,
                    y=np.asarray(y, dtype=float),
                    sample_weight=np.ones(n),
                    offset=None,
                    pirls_tol=1e-8,
                    max_pirls_iter=50,
                    lambdas=lambdas,
                    reml_penalties=getattr(model, "_reml_penalties", None) or [],
                    compute_fit_stats=lambda *a, **k: None,
                )
