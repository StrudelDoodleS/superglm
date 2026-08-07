"""A candidate power whose penalized mode does not converge must be routed
around by the power search, not raised out of it.

``optimize_direct_reml`` reports "this power has no usable penalized mode" two
different ways from the same loop: ``ObservedModeNotCertifiedError`` when the
mode is found but cannot be differentiated through, and a bare ``RuntimeError``
when PIRLS did not converge to a mode at all. The Tweedie power search catches
only the first. The second escapes and kills the whole search -- from the
bracket endpoint p=1.95, which the search probes second and never selects.

Both fixtures below are sized so the failure lands on a power the search only
probes, never returns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from superglm import Categorical, OrderedCategorical, Spline, SuperGLM, families

CAT_LEVELS = {"f0": 6, "f1": 9, "f2": 4, "f3": 11, "f4": 13, "f5": 11}
OC_LEVELS = {"f6": 16, "f7": 24}


def _fixture(n: int, seed: int = 4):
    """Small Tweedie/log frame with saturated cr bases on both ordered terms."""
    rng = np.random.default_rng(seed)
    columns: dict[str, np.ndarray] = {}
    eta = np.full(n, -1.0)
    for name, k in CAT_LEVELS.items():
        levels = [f"{name}_{j:02d}" for j in range(k)]
        idx = rng.integers(0, k, n)
        columns[name] = np.array(levels)[idx]
        eta += rng.normal(0, 0.2, k)[idx]
    orders: dict[str, list[str]] = {}
    for name, k in OC_LEVELS.items():
        levels = [f"{name}_{j:02d}" for j in range(k)]
        idx = rng.integers(0, k, n)
        columns[name] = np.array(levels)[idx]
        eta += 0.02 * (idx - k / 2)
        orders[name] = levels
    frame = pd.DataFrame(columns)
    weights = rng.uniform(1.19e-5, 1.0, n)
    offset = np.where(rng.random(n) < 0.35, 0.0, 1.0986)
    y = np.where(rng.random(n) < 0.83, 0.0, rng.gamma(1.5, np.exp(eta) * 900, n))
    features = {name: Categorical() for name in CAT_LEVELS}
    for name, k in OC_LEVELS.items():
        features[name] = OrderedCategorical(order=orders[name], basis=Spline(kind="cr", k=k))
    return frame, y, weights, offset, features


def _model(features, p=1.5):
    return SuperGLM(family=families.tweedie(p=p), features=features)


@pytest.mark.parametrize("n", [6_000, 4_000])
def test_bracket_endpoint_without_a_converged_mode_is_routed_around(n):
    """The search must survive a probe power whose penalized mode fails."""
    frame, y, weights, offset, features = _fixture(n)

    # Precondition: p=1.95 -- the second point Brent probes -- has no usable
    # penalized mode under REML. Without this the test proves nothing.
    with pytest.raises(RuntimeError) as excinfo:
        _model(features, p=1.95).fit_reml(frame, y, sample_weight=weights, offset=offset)
    assert "converged penalized coefficient mode" in str(excinfo.value)

    result = _model(features).estimate_p(
        frame, y, sample_weight=weights, offset=offset, fit_mode="reml"
    )
    assert 1.05 < float(result.p_hat) < 1.95
    assert np.isfinite(float(result.phi_hat))


def test_decoupled_search_already_survives_what_the_coupled_search_does_not():
    """The ML-mode search never fits REML at the failing power, so it completes.

    This pins the asymmetry: the documented "approximation" is strictly more
    robust here than the regime it approximates.
    """
    frame, y, weights, offset, features = _fixture(6_000)

    decoupled = _model(features).estimate_p(
        frame,
        y,
        sample_weight=weights,
        offset=offset,
        fit_mode="reml",
        search_fit_mode="fit",
    )
    assert 1.05 < float(decoupled.p_hat) < 1.95

    coupled = _model(features).estimate_p(
        frame, y, sample_weight=weights, offset=offset, fit_mode="reml"
    )
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

        frame, y, weights, offset, features = _fixture(3_000)
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


class TestInitializationSearchesRouteAroundInfeasiblePoints:
    """grid_refine and profile_opt read every initialization record back.

    An uncertifiable initialization power leaves no evaluation record, so an
    unconditional cache read raises KeyError and kills the search that was
    designed to route around exactly this.
    """

    @pytest.mark.parametrize(
        ("method", "p_bounds"),
        [("grid_refine", (1.05, 1.95)), ("profile_opt", (1.05, 1.95))],
    )
    def test_infeasible_initialization_points_are_routed_around(
        self, monkeypatch, method, p_bounds
    ):
        from superglm.reml.observed_geometry import ObservedModeNotConvergedError

        frame, y, weights, offset, features = _fixture(3_000)
        real_fit_reml = SuperGLM.fit_reml

        def failing_above_19(self, X, yv, **kwargs):
            if float(getattr(self.family, "p", 0.0)) > 1.9:
                raise ObservedModeNotConvergedError()
            return real_fit_reml(self, X, yv, **kwargs)

        monkeypatch.setattr(SuperGLM, "fit_reml", failing_above_19)

        result = _model(features).estimate_p(
            frame,
            y,
            sample_weight=weights,
            offset=offset,
            fit_mode="reml",
            method=method,
            p_bounds=p_bounds,
        )

        assert p_bounds[0] < float(result.p_hat) < 1.9


class TestPublicationModeFailure:
    """A publish refit that cannot certify must explain itself."""

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

    def test_a_pinned_optimum_warns(self):
        from superglm.profiling.tweedie import _boundary_censoring_message

        message = _boundary_censoring_message(
            1.6152, {1.6158: "not certifiable", 1.95: "not certifiable"}, xatol=1e-3
        )
        assert message is not None
        assert "censored" in message
        assert "search_fit_mode" in message

    def test_grid_censoring_uses_grid_spacing_as_resolution(self, monkeypatch):
        """method='grid' resolves p only to its spacing, so censoring must be
        judged against that spacing, not against Brent's xatol."""
        import superglm.profiling.tweedie as tweedie_module

        captured = {}
        real = tweedie_module._boundary_censoring_message

        def spy(p_hat, infeasible, *, xatol):
            captured["resolution"] = xatol
            return real(p_hat, infeasible, xatol=xatol)

        monkeypatch.setattr(tweedie_module, "_boundary_censoring_message", spy)
        frame, y, weights, offset, features = _fixture(3_000)

        _model(features).estimate_p(
            frame, y, sample_weight=weights, offset=offset, fit_mode="reml", method="grid"
        )

        assert captured["resolution"] == pytest.approx((1.95 - 1.05) / 19.0)

    def test_a_grid_step_gap_warns_at_grid_resolution(self):
        from superglm.profiling.tweedie import _boundary_censoring_message

        spacing = (1.95 - 1.05) / 19.0
        message = _boundary_censoring_message(
            1.6184, {1.6658: "not certifiable"}, xatol=spacing
        )
        assert message is not None
        assert "censored" in message

    def test_a_distant_boundary_stays_silent(self):
        from superglm.profiling.tweedie import _boundary_censoring_message

        assert (
            _boundary_censoring_message(
                1.5006, {1.05: "not certifiable", 1.95: "not certifiable"}, xatol=1e-3
            )
            is None
        )

    def test_routed_around_endpoints_do_not_warn_end_to_end(self, recwarn):
        """The k-sweep fixture routes around p=1.95; its p_hat sits far away.

        The fail-closed benchmark treats any warning as a failure, so the
        warning must not fire merely because infeasible powers exist.
        """
        frame, y, weights, offset, features = _fixture(6_000)

        _model(features).estimate_p(frame, y, sample_weight=weights, offset=offset, fit_mode="reml")

        assert not [w for w in recwarn.list if "censored" in str(w.message)]
