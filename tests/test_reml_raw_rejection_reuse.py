"""Only rejected raw centering may survive a fixed-design REML coefficient fit."""

import inspect
import weakref

import numpy as np
import pandas as pd
import pytest

from superglm import Constraint, CubicRegressionSpline, Numeric, SuperGLM
from superglm._fit_trace import TraceRun
from superglm._group_matrix import _group_matrix_centered as raw_centering
from superglm.distributions import Gamma, Gaussian, Poisson
from superglm.group_matrix import DenseGroupMatrix, DesignMatrix
from superglm.links import IdentityLink, LogLink
from superglm.reml import direct, observed_geometry
from superglm.solvers import centered_system, irls_direct
from superglm.solvers.scop import build_scop_solver_reparam
from superglm.types import GroupSlice, LinearConstraintSet, PenaltyComponent


def _fixture(x=(1.0, 2.0, 3.0)):
    dm = DesignMatrix([DenseGroupMatrix(np.array(x)[:, None])], n=len(x), p=1)
    groups = [
        GroupSlice(
            name="BonusMalus",
            start=0,
            end=1,
            constraints=LinearConstraintSet(np.ones((1, 1)), np.zeros(1)),
        )
    ]
    return dm, groups, Poisson(), LogLink()


def _fit(fixture, policy=None, *, weights=None, penalty=1.0, **kwargs):
    dm, groups, family, link = fixture
    # Execute the unfixed solver too, so RED observes repeated real assemblies.
    if "_raw_moment_policy" in inspect.signature(irls_direct.fit_irls_direct).parameters:
        kwargs["_raw_moment_policy"] = policy
    return irls_direct.fit_irls_direct(
        X=dm,
        y=np.ones(dm.n),
        weights=np.ones(dm.n) if weights is None else weights,
        family=family,
        link=link,
        groups=groups,
        lambda2=penalty,
        S_override=np.array([[penalty]]),
        beta_init=np.zeros(1),
        intercept_init=0.0,
        direct_solve=kwargs.pop("direct_solve", "gram"),
        return_xtwx=True,
        weight_semantics="frequency",
        **kwargs,
    )


@pytest.fixture
def attempts(monkeypatch):
    # Change only the size crossover, not the rejection/accuracy certificate.
    monkeypatch.setattr(raw_centering, "_MIN_MIXED_RAW_MOMENT_CELLS", 0)
    seen = []
    original = centered_system.try_raw_moment_centering

    def counted(**kwargs):
        result = original(**kwargs)
        seen.append(result is not None)
        return result

    monkeypatch.setattr(centered_system, "try_raw_moment_centering", counted)
    return seen


def test_constrained_fits_reuse_only_rejection_and_rebuild_changed_weights(attempts, monkeypatch):
    # Removing the negative handoff repeats raw attempts; stale data breaks the
    # independently known centered Gram/inverse when weights or penalty change.
    fixture = _fixture()
    policy = centered_system.TabmatCenteringState()
    systems = []
    original = irls_direct.build_centered_system

    def recorded(**kwargs):
        system = original(**kwargs)
        systems.append(system)
        return system

    monkeypatch.setattr(irls_direct, "build_centered_system", recorded)
    for weights, penalty, gram, raw_gram in (
        (np.ones(3), 1.0, 2.0, 14.0),
        (np.array([2.0, 1.0, 2.0]), 3.0, 4.0, 24.0),
    ):
        result, inverse, raw = _fit(fixture, policy, weights=weights, penalty=penalty)
        assert result.converged
        np.testing.assert_array_equal(result.beta, [0.0])
        assert result.intercept == 0.0
        np.testing.assert_array_equal(raw, [[raw_gram]])
        np.testing.assert_allclose(inverse, [[1 / (gram + penalty)]], rtol=8 * np.finfo(float).eps)
        np.testing.assert_array_equal(systems[-1].data_gram, [[gram]])
        np.testing.assert_array_equal(systems[-1].rhs, [0.0])
    assert attempts == [False]
    with pytest.raises(ValueError, match="finite and non-negative"):
        _fit(fixture, policy, weights=np.array([1.0, np.nan, 1.0]))


def test_bonus_malus_cubic_qp_representation_is_eligible(attempts):
    # The French recipe uses a constrained cubic spline, not a SCOP group.
    # Exercise its real built representation, alongside an ill-located numeric.
    x = np.linspace(50.0, 230.0, 36)
    frame = pd.DataFrame({"BonusMalus": x, "power": 9.0 + np.sin(x)})
    model = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        discrete=False,
        features={
            "BonusMalus": CubicRegressionSpline(n_knots=4, constraint=Constraint.fit.increasing),
            "power": Numeric(),
        },
    )
    model._build_design_matrix(frame, np.ones(len(x)), np.ones(len(x)), None)
    dm, groups = model._dm, model._groups
    assert any(group.constraints is not None for group in groups)
    assert all(group.monotone_engine != "scop" for group in groups)
    family, link = Poisson(), LogLink()
    policy = centered_system.TabmatCenteringState()
    for penalty in (1.0, 2.0):
        result, _ = irls_direct.fit_irls_direct(
            dm,
            np.ones(len(x)),
            np.ones(len(x)),
            family,
            link,
            groups,
            lambda2=penalty,
            S_override=penalty * np.eye(dm.p),
            beta_init=np.zeros(dm.p),
            intercept_init=0.0,
            direct_solve="gram",
            weight_semantics="frequency",
            _raw_moment_policy=policy,
        )
        assert result.converged
        np.testing.assert_array_equal(result.beta, np.zeros(dm.p))
        assert result.intercept == 0.0
    assert attempts == [False]


def test_optimizer_owns_one_rejection_but_outside_refits_stay_fresh(attempts, monkeypatch):
    # Missing any of the three optimizer call-site handoffs repeats the rejected
    # assembly. Both surrounding fits deliberately receive no policy.
    x = np.linspace(-1.0, 1.0, 36)
    dm, groups, family, link = _fixture(2.0 + x)
    phases = []
    owners = []
    original = direct.fit_irls_direct

    def recorded(**kwargs):
        if kwargs.get("_raw_moment_policy") is not None:
            owners.append(weakref.ref(kwargs["_raw_moment_policy"]))
        result = original(**kwargs)
        phases.append(kwargs["debug_context"]["phase"])
        return result

    monkeypatch.setattr(direct, "fit_irls_direct", recorded)
    _fit((dm, groups, family, link))
    before = len(attempts)
    direct.optimize_direct_reml(
        dm,
        family,
        link,
        groups,
        False,
        np.rint(2.0 + 8.0 * np.sin(np.pi * (x + 1.0) / 2.0) ** 2),
        np.ones(len(x)),
        np.zeros(len(x)),
        [(0, groups[0])],
        {"BonusMalus": 1.0},
        {"BonusMalus": 1.0},
        weight_semantics="frequency",
        max_reml_iter=3,
        reml_tol=1e-6,
        verbose=False,
        direct_solve="gram",
        reml_penalties=[
            PenaltyComponent(
                "BonusMalus",
                "BonusMalus",
                0,
                slice(0, 1),
                None,
                rank=1.0,
                penalty_kind="identity",
            )
        ],
    )
    assert {"bootstrap", "candidate", "line_search"} <= set(phases)
    assert len(attempts) - before == 1
    assert owners and all(owner() is None for owner in owners)
    _fit((dm, groups, family, link))
    assert attempts == [False, False, False]


def test_accepted_route_is_certified_with_fresh_fit_state(attempts, monkeypatch):
    # Carrying a positive/preflight state or skipping its fresh certificate is
    # forbidden even when two fits happen to use identical weights.
    fixture = _fixture((-1.0, 0.0, 1.0))
    policy = centered_system.TabmatCenteringState()
    original = irls_direct.build_centered_system
    starts = []
    states = []

    def recorded(**kwargs):
        state = kwargs["tabmat_state"]
        if not states or state is not states[-1]:
            states.append(state)
            starts.append((state.eligible, state.raw_spline_eligible, state.raw_moment_eligible))
        return original(**kwargs)

    monkeypatch.setattr(irls_direct, "build_centered_system", recorded)
    for penalty in (1.0, 2.0):
        result, inverse, _ = _fit(fixture, policy, penalty=penalty)
        assert result.converged
        np.testing.assert_array_equal(result.beta, [0.0])
        np.testing.assert_allclose(inverse, [[1 / (2 + penalty)]], rtol=8 * np.finfo(float).eps)
    assert starts == [(None, None, None), (None, None, None)]
    assert len(attempts) >= 2 and all(attempts)
    assert policy.raw_moment_eligible is None


def test_later_raw_eligibility_does_not_reuse_a_system_or_positive_policy(attempts):
    # The first W rejects but uniform W restores raw eligibility. Remembering
    # the refusal may change the route, never the weighted answer.
    fixture = _fixture((-1.0, 0.0, 1.0))
    policy = centered_system.TabmatCenteringState()
    _fit(fixture, policy, weights=np.array([1.0, 1.0, 20.0]))
    assert attempts == [False]
    result, inverse, raw = _fit(fixture, policy)
    assert attempts == [False]
    assert result.converged
    np.testing.assert_array_equal(result.beta, [0.0])
    np.testing.assert_array_equal(raw, [[2.0]])
    np.testing.assert_allclose(inverse, [[1 / 3]], rtol=8 * np.finfo(float).eps)
    _fit(fixture)
    assert attempts[1:] and all(attempts[1:])


@pytest.mark.parametrize(
    "change",
    [
        "design",
        "plan",
        "family",
        "link",
        "custom_group",
        "custom_constraint",
        "trace",
        "debug",
        "qr",
    ],
)
def test_changed_or_unsupported_owner_drops_negative_policy(attempts, change):
    # Reusing the old refusal after leaving its supported owner suppresses a
    # fresh raw attempt when the original route returns.
    fixture = _fixture()
    policy = centered_system.TabmatCenteringState()
    if change == "qr":
        # A constrained fit always resolves to QP/Gram, even when QR is asked
        # for. Remove the constraint so this exercises a real backend change.
        fixture[1][0].constraints = None
    _fit(fixture, policy)
    dm, groups, family, link = fixture
    kwargs = {}
    old_constraint = groups[0].constraints
    if change == "design":
        dm = _fixture()[0]
    elif change == "plan":
        dm._execution_plan = None
    elif change == "family":

        class CustomPoisson(Poisson):
            pass

        family = CustomPoisson()
    elif change == "link":

        class CustomLog(LogLink):
            pass

        link = CustomLog()
    elif change == "custom_group":

        class CustomDense(DenseGroupMatrix):
            pass

        dm = DesignMatrix([CustomDense(np.array([[1.0], [2.0], [3.0]]))], n=3, p=1)
    elif change == "trace":
        kwargs["trace_run"] = TraceRun("no-reuse")
    elif change == "debug":
        kwargs["debug_recorder"] = object()  # Disabled recorder, still a callback boundary.
    elif change == "custom_constraint":

        class CustomConstraint(LinearConstraintSet):
            pass

        groups[0].constraints = CustomConstraint(np.ones((1, 1)), np.zeros(1))
    else:
        kwargs["direct_solve"] = "qr"
    _fit((dm, groups, family, link), policy, **kwargs)
    if change == "custom_constraint":
        groups[0].constraints = old_constraint
    if change == "plan":
        dm._execution_plan = None
    before_return = len(attempts)
    _fit(fixture, policy)
    assert len(attempts) > before_return
    assert not any(attempts)


def test_positive_initial_data_reuse_remains_local_and_validated(attempts, monkeypatch):
    # The existing exact-W/z initial-data shortcut may carry its own certified
    # system, but must not turn that acceptance into a cross-fit raw policy.
    dm, groups, family, link = _fixture((-1.0, 0.0, 1.0))
    groups[0].constraints = None
    policy = centered_system.TabmatCenteringState()
    initial = centered_system._InitialDataReuse()
    hits = []
    original = centered_system._InitialDataReuse.take

    def recorded(self, *args):
        result = original(self, *args)
        hits.append(result is not None)
        return result

    monkeypatch.setattr(centered_system._InitialDataReuse, "take", recorded)
    for penalty in (1.0, 2.0):
        _, inverse, _ = _fit(
            (dm, groups, family, link), policy, penalty=penalty, _initial_data_reuse=initial
        )
        np.testing.assert_allclose(inverse, [[1 / (2 + penalty)]], rtol=8 * np.finfo(float).eps)
    assert hits == [False, True]
    assert policy.raw_moment_eligible is None
    assert all(attempts)


def test_scop_cannot_retain_the_fixed_coordinate_policy(attempts):
    fixture = _fixture()
    policy = centered_system.TabmatCenteringState()
    _fit(fixture, policy)
    reparam = build_scop_solver_reparam(q_raw=2, kind="increasing")
    irls_direct.fit_irls_direct(
        np.array([[-1.0], [1.0]]),
        np.array([-1.0, 1.0]),
        np.ones(2),
        Gaussian(),
        IdentityLink(),
        [GroupSlice("shape", 0, 1, monotone_engine="scop", scop_reparameterization=reparam)],
        lambda2=0.0,
        beta_init=np.ones(1),
        intercept_init=0.0,
        max_iter=1,
        S_override=np.zeros((1, 1)),
        _scop_joint=False,
        scop_state_init={0: {"beta_eff": np.zeros(1), "S_scop": np.zeros((1, 1))}},
        _compute_scop_postfit_inference=False,
        weight_semantics="frequency",
        _raw_moment_policy=policy,
    )
    assert policy._raw_moment_owners == ()
    before = len(attempts)
    _fit(fixture, policy)
    assert len(attempts) > before


def test_gamma_observed_geometry_does_not_share_fisher_policy(attempts, monkeypatch):
    # Observed curvature must still use each current y/mu, not any remembered
    # Fisher weights, data Gram, or raw-rejection policy from coefficient fits.
    x = np.linspace(-1.0, 1.0, 24)
    dm, groups, _, link = _fixture(2.0 + x)
    groups[0].constraints = None
    family = Gamma()
    y = np.exp(0.2 * x) * (1.0 + 0.1 * np.cos(4 * np.pi * x))
    originals = direct.fit_irls_direct, observed_geometry.build_centered_system
    policies, observed_states = [], []

    def coefficient(**kwargs):
        policies.append(kwargs["_raw_moment_policy"])
        return originals[0](**kwargs)

    def observed(**kwargs):
        observed_states.append(kwargs["tabmat_state"])
        system = originals[1](**kwargs)
        W = kwargs["W"]
        # This direct dense oracle uses no production centering helper.
        xc = x - np.dot(W, x) / W.sum()
        expected = np.dot(W * xc, xc)
        bound = 32 * len(x) * np.finfo(float).eps * max(expected, 1.0)
        assert abs(system.data_gram[0, 0] - expected) <= bound
        return system

    monkeypatch.setattr(direct, "fit_irls_direct", coefficient)
    monkeypatch.setattr(observed_geometry, "build_centered_system", observed)
    direct.optimize_direct_reml(
        dm,
        family,
        link,
        groups,
        False,
        y,
        np.ones(len(x)),
        np.zeros(len(x)),
        [(0, groups[0])],
        {"BonusMalus": 1.0},
        {"BonusMalus": 1.0},
        weight_semantics="frequency",
        max_reml_iter=2,
        reml_tol=1e-6,
        verbose=False,
        direct_solve="gram",
        reml_penalties=[
            PenaltyComponent(
                "BonusMalus",
                "BonusMalus",
                0,
                slice(0, 1),
                None,
                rank=1.0,
                penalty_kind="identity",
            )
        ],
    )
    assert len(observed_states) >= 2 and policies
    assert all(state is not policy for state in observed_states for policy in policies)
