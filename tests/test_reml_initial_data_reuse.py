"""Reused initial data carries no penalty, and serves only a build at the same weights."""

import gc
import weakref

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from superglm import SuperGLM
from superglm._group_matrix import _lookup_tables
from superglm.distributions import Poisson
from superglm.features.categorical import Categorical
from superglm.features.spline import Spline
from superglm.group_matrix import (
    CategoricalGroupMatrix,
    DenseGroupMatrix,
    DesignMatrix,
    SparseSSPGroupMatrix,
    SupportCompressedSSPGroupMatrix,
)
from superglm.links import LogLink
from superglm.reml import direct, observed_geometry
from superglm.solvers import centered_system
from superglm.solvers.irls_direct import _fit as irls_direct
from superglm.types import GroupSlice


@pytest.mark.parametrize(
    "kind", ["dense", "sparse_ssp", "support_compressed", "mixed_categorical", "custom_categorical"]
)
def test_initial_data_is_built_once_but_both_final_systems_are_fresh(monkeypatch, kind):
    # A final system is built at its own iterate, whatever an entry holds.
    # Removing the initial-only handoff repeats this real numerical assembly.
    x = np.array([[-1.0], [0.0], [1.0]])
    mixed = kind in ("mixed_categorical", "custom_categorical")
    if mixed:
        x = np.tile(x, (2, 1))
    if kind == "dense" or mixed:
        group = DenseGroupMatrix(x)
    elif kind == "sparse_ssp":
        group = SparseSSPGroupMatrix(sparse.csr_matrix(x), np.ones((1, 1)))
    else:
        group = SupportCompressedSSPGroupMatrix(x, np.ones((1, 1)), np.arange(3))
    matrix_groups = [group]
    groups = [GroupSlice(name="x", start=0, end=1)]
    if mixed:

        class CustomCategorical(CategoricalGroupMatrix):
            pass

        categorical = CustomCategorical if kind == "custom_categorical" else CategoricalGroupMatrix
        matrix_groups.append(categorical(np.array([0, 0, 0, -1, -1, -1]), n_levels=1))
        groups.append(GroupSlice(name="category", start=1, end=2))
    dm = DesignMatrix(matrix_groups, n=len(x), p=len(matrix_groups))
    family, link = Poisson(), LogLink()
    builds = []
    original = irls_direct.build_centered_system

    def counted(**kwargs):
        result = original(**kwargs)
        builds.append(result)
        return result

    monkeypatch.setattr(irls_direct, "build_centered_system", counted)
    # Keep the regression runnable against the unfixed implementation.
    entry_type = getattr(centered_system, "_InitialDataReuse", None)
    reuse = {} if entry_type is None else {"_initial_data_reuse": entry_type()}
    for penalty in (1.0, 2.0):
        result, inverse, gram = irls_direct.fit_irls_direct(
            X=dm,
            y=np.tile([0.0, 3.0, 0.0], 2 if mixed else 1),
            weights=np.ones(dm.n),
            family=family,
            link=link,
            groups=groups,
            lambda2=penalty,
            S_override=np.diag([penalty, 2 * penalty]) if mixed else np.array([[penalty]]),
            beta_init=np.zeros(dm.p),
            intercept_init=0.0,
            direct_solve="gram",
            return_xtwx=True,
            weight_semantics="frequency",
            **reuse,
        )
        assert result.converged
        np.testing.assert_array_equal(result.beta, np.zeros(dm.p))
        # In the mixed fixture the numeric column sums to zero within each
        # category. Raw Gram is diag(4, 3); centering gives diag(4, 3/2).
        expected_gram = np.diag([4.0, 3.0]) if mixed else np.array([[2.0]])
        expected_inverse = (
            np.diag([1 / (4 + penalty), 1 / (1.5 + 2 * penalty)])
            if mixed
            else np.array([[1 / (2 + penalty)]])
        )
        np.testing.assert_array_equal(gram, expected_gram)
        np.testing.assert_allclose(inverse, expected_inverse, rtol=8 * np.finfo(float).eps)
    assert len(builds) == (4 if kind == "custom_categorical" else 3)


def test_custom_group_does_not_enter_the_fixed_design_reuse_path(monkeypatch):
    # A subclass can expose mutable coordinates or callback side effects.
    class CustomDense(DenseGroupMatrix):
        pass

    dm = DesignMatrix([CustomDense(np.array([[-1.0], [0.0], [1.0]]))], n=3, p=1)
    family, link = Poisson(), LogLink()
    reuse = centered_system._InitialDataReuse()
    builds = []
    original = irls_direct.build_centered_system

    def counted(**kwargs):
        result = original(**kwargs)
        builds.append(result)
        return result

    monkeypatch.setattr(irls_direct, "build_centered_system", counted)
    for penalty in (1.0, 2.0):
        irls_direct.fit_irls_direct(
            X=dm,
            y=np.array([0.0, 3.0, 0.0]),
            weights=np.ones(3),
            family=family,
            link=link,
            groups=[GroupSlice(name="x", start=0, end=1)],
            lambda2=penalty,
            S_override=np.array([[penalty]]),
            beta_init=np.zeros(1),
            intercept_init=0.0,
            direct_solve="gram",
            weight_semantics="frequency",
            _initial_data_reuse=reuse,
        )
    assert len(builds) == 4


def _seed_entry():
    # a type the raw-moment rung admits, so its latch moves: a dense column
    # never takes that rung (issue #430)
    column = sparse.csr_matrix(np.array([[-1.0], [0.0], [1.0]]))
    dm = DesignMatrix([SparseSSPGroupMatrix(column, np.ones((1, 1)))], n=3, p=1)
    W, z = np.array([1.0, 2.0, 1.0]), np.array([-1.0, 0.0, 1.0])
    state = centered_system.TabmatCenteringState()
    before = centered_system.TabmatCenteringState()
    system = centered_system.build_centered_system(
        dm=dm,
        W=W,
        z_off=z,
        penalty=np.ones((1, 1)),
        tabmat_state=state,
    )
    entry = centered_system._InitialDataReuse()
    entry.remember(("warm and coordinates",), W, z, before, state, system)
    return entry, W, z, before, state


@pytest.mark.parametrize(
    "change", ["weights", "rhs", "signed_zero", "shape", "dtype", "centering", "key", "equal_key"]
)
def test_changed_initial_contract_must_miss(change):
    # Bypassing either the context/state or actual binary64 operand guard is unsafe.
    entry, W, z, state, _ = _seed_entry()
    key = entry.base.key
    if change == "weights":
        W[0] = 2.0
    elif change == "rhs":
        z[0] = 0.0
    elif change == "signed_zero":
        z[1] = -0.0
    elif change == "shape":
        z = z[:, None]
    elif change == "dtype":
        z = z.astype(np.float32)
    elif change == "centering":
        state.raw_moment_eligible = False
    elif change == "equal_key":  # an entry asks for the same objects, not equal ones
        key = ("".join(["warm and ", "coordinates"]),)
        assert key == entry.base.key
    else:
        key = ("changed warm state or coordinates",)
    assert entry.take(key, W, z, state, np.ones((1, 1))) is None


@pytest.mark.parametrize(
    "field, built_under, read_under, served",
    [
        # "try the rung" is None or True; only False skips it
        ("raw_moment_eligible", True, None, True),
        ("raw_moment_eligible", None, False, False),
        ("tensor_raw_eligible", False, None, False),
        # no build leaves it True, so nothing is served across that value
        ("tensor_raw_eligible", True, None, False),
        # a preflight runs for None and not for True: another build
        ("eligible", True, None, False),
        ("raw_spline_eligible", True, None, False),
    ],
)
def test_an_entry_serves_a_build_that_takes_the_routes_it_was_built_by(
    field, built_under, read_under, served
):
    # Serving across states that differ in a decision no route reads would keep a build that
    # the rung ladder runs again at every fit's start; serving across one a route reads
    # would return data another route forms.
    entry, W, z, _, _ = _seed_entry()
    entry.base.before = centered_system.TabmatCenteringState(**{field: built_under})
    state = centered_system.TabmatCenteringState(**{field: read_under})

    taken = entry.take(entry.base.key, W, z, state, np.ones((1, 1)))

    assert (taken is not None) is served


def test_none_and_true_of_the_raw_moment_flag_build_the_same_system_by_the_same_route(monkeypatch):
    # ``routes_as`` serves one from the other, which is sound only while the rungs read them alike.
    # Mutation check: a raw-moment rung that runs only for ``None`` takes another route here.
    tried = []
    try_rung = centered_system.try_raw_moment_centering

    def counted(**kwargs):
        result = try_rung(**kwargs)
        tried.append(result is not None)
        return result

    monkeypatch.setattr(centered_system, "try_raw_moment_centering", counted)
    column = sparse.csr_matrix(np.array([[-1.0], [0.0], [1.0]]))
    dm = DesignMatrix([SparseSSPGroupMatrix(column, np.ones((1, 1)))], n=3, p=1)
    W, z = np.array([1.0, 2.0, 1.0]), np.array([-1.0, 0.0, 1.0])
    states = [
        centered_system.TabmatCenteringState(),
        centered_system.TabmatCenteringState(raw_moment_eligible=True),
    ]
    first, second = (
        centered_system.build_centered_system(
            dm=dm, W=W, z_off=z, penalty=np.ones((1, 1)), tabmat_state=state
        )
        for state in states
    )
    assert len(tried) == 2 and tried[0] == tried[1]
    for name in ("data_gram", "rhs", "mean_x"):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
    assert states[0].routes_as(states[1])


@pytest.mark.parametrize("subclass", [False, True])
def test_the_geometry_owner_serves_exactly_the_built_in_designs(monkeypatch, subclass):
    # A subclass can expose mutable coordinates or call back between an entry's making and its use.
    # Mutation check: a geometry owner with no check on the design serves the subclass.
    class Custom(DesignMatrix):
        pass

    kind = Custom if subclass else DesignMatrix
    dm = kind([DenseGroupMatrix(np.array([[-1.0], [0.0], [1.0]]))], n=3, p=1)
    builds = []
    build = observed_geometry.build_centered_system

    def counted(**kwargs):
        builds.append(kwargs["W"])
        return build(**kwargs)

    monkeypatch.setattr(observed_geometry, "build_centered_system", counted)
    owner = centered_system._InitialDataReuse()
    for _ in range(2):
        observed_geometry._observed_centered_system(
            dm,
            np.array([1.0, 2.0, 1.0]),
            np.ones((1, 1)),
            centered_system.TabmatCenteringState(),
            owner,
        )
    assert len(builds) == (2 if subclass else 1)


@pytest.mark.parametrize("penalty, expected_hessian", [(3.0, 5.0), (-3.0, 0.0)])
def test_reused_data_gets_fresh_penalty_repair_and_centering_state(penalty, expected_hessian):
    # Retaining an old Hessian or skipping the existing PSD repair fails this.
    entry, W, z, state, after = _seed_entry()
    assert state != after  # This real fixture changes the raw-moment safety latch.
    reused = entry.take(entry.base.key, W, z, state, np.array([[penalty]]))
    assert state == after
    assert reused.sum_w == 4.0
    assert reused.mean_z == 0.0
    np.testing.assert_array_equal(reused.mean_x, [0.0])
    np.testing.assert_array_equal(reused.data_gram, [[2.0]])
    np.testing.assert_array_equal(reused.rhs, [2.0])
    np.testing.assert_array_equal(reused.penalty, [[penalty]])
    np.testing.assert_array_equal(reused.hessian, [[expected_hessian]])
    assert not reused.data_gram.flags.writeable
    assert not reused.hessian.flags.writeable


@pytest.mark.parametrize("change", ["warm", "coordinates"])
def test_solver_does_not_reuse_changed_initial_context(monkeypatch, change):
    entry = centered_system._InitialDataReuse()
    dm = DesignMatrix([DenseGroupMatrix(np.array([[-1.0], [0.0], [1.0]]))], n=3, p=1)
    family, link = Poisson(), LogLink()
    hits = []
    original = centered_system._InitialDataReuse.take

    def recorded(self, key, *args):
        result = original(self, key, *args)
        hits.append(result is not None)
        return result

    monkeypatch.setattr(centered_system._InitialDataReuse, "take", recorded)
    for i in range(2):
        if i and change == "coordinates":
            dm = DesignMatrix([DenseGroupMatrix(np.array([[-2.0], [0.0], [2.0]]))], n=3, p=1)
        result, inverse, gram = irls_direct.fit_irls_direct(
            X=dm,
            y=np.array([0.0, 3.0, 0.0]),
            weights=np.ones(3),
            family=family,
            link=link,
            groups=[GroupSlice(name="x", start=0, end=1)],
            lambda2=1.0,
            S_override=np.ones((1, 1)),
            beta_init=np.zeros(1),
            intercept_init=0.25 if i and change == "warm" else 0.0,
            direct_solve="gram",
            weight_semantics="frequency",
            _initial_data_reuse=entry,
            return_xtwx=True,
        )
        assert result.converged
        if change == "coordinates":
            expected_gram = 8.0 if i else 2.0
            np.testing.assert_array_equal(gram, [[expected_gram]])
            np.testing.assert_allclose(
                inverse, [[1 / (expected_gram + 1)]], rtol=8 * np.finfo(float).eps
            )
    assert hits == [False, False]  # Neither later iterations nor final export look up the entry.


@pytest.mark.parametrize("interrupt", [False, True])
def test_the_optimizer_shares_one_entry_between_its_fits_and_releases_it_on_return_or_exception(
    monkeypatch, interrupt
):
    # Real tiny REML scheduling, without asserting any platform-specific
    # terminal-trial count or Armijo rounding outcome.
    created = []
    owners = []
    entry_type = centered_system._InitialDataReuse
    original_fit = direct.fit_irls_direct

    def entry():
        value = entry_type()
        created.append(weakref.ref(value))
        return value

    class InterruptedTrialError(RuntimeError):
        pass

    def coefficient_fit(**kwargs):
        initial = kwargs.get("_initial_data_reuse")
        if initial is not None:
            owners.append(next(i for i, reference in enumerate(created) if reference() is initial))
        result = original_fit(**kwargs)
        if interrupt and len(owners) == 3:
            raise InterruptedTrialError
        return result

    monkeypatch.setattr(direct, "_InitialDataReuse", entry)
    monkeypatch.setattr(direct, "fit_irls_direct", coefficient_fit)
    x = np.linspace(0.0, 1.0, 36)
    y = np.rint(1.0 + 8.0 * np.sin(np.pi * x) ** 2)
    model = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        discrete=False,
        features={"x": Spline(n_knots=5, penalty="ssp")},
    )
    if interrupt:
        with pytest.raises(InterruptedTrialError):
            model.fit_reml(pd.DataFrame({"x": x}), y, max_reml_iter=3)
    else:
        model.fit_reml(pd.DataFrame({"x": x}), y, max_reml_iter=3)
    assert len(created) == 1  # one owner for the run, however many fits it makes
    assert len(owners) >= 3 and set(owners) == {0}  # the candidates and the trials share it
    gc.collect()
    assert all(reference() is None for reference in created)


def _poisson_problem():
    # Two spline-like columns: the Poisson IRLS takes several steps from zero.
    x = np.linspace(-1.0, 1.0, 40)
    support = np.column_stack([x, x**2 - np.mean(x**2)])
    # a support whose builds leave the centring state as it began, so a fresh fit's first
    # build equals the previous fit's last
    group = SupportCompressedSSPGroupMatrix(support, np.eye(2), np.arange(len(x)))
    dm = DesignMatrix([group], n=len(x), p=2)
    y = np.rint(2.0 + 5.0 * np.exp(-3.0 * (x - 0.3) ** 2))
    return dm, [GroupSlice(name="x", start=0, end=2)], y


FAMILY, LINK = Poisson(), LogLink()  # one pair for every fit: an entry names them


def _fit_from(dm, groups, y, penalty, start, owner, builds):
    beta, intercept = start
    return irls_direct.fit_irls_direct(
        X=dm,
        y=y,
        weights=np.ones(dm.n),
        family=FAMILY,
        link=LINK,
        groups=groups,
        lambda2=penalty,
        S_override=penalty * np.eye(2),
        beta_init=beta,
        intercept_init=intercept,
        direct_solve="gram",
        return_xtwx=True,
        weight_semantics="frequency",
        _initial_data_reuse=owner,
    )


@pytest.fixture
def recorded_builds(monkeypatch):
    builds = []
    original = irls_direct.build_centered_system

    def recorded(**kwargs):
        builds.append(np.array(kwargs["W"], copy=True))
        return original(**kwargs)

    monkeypatch.setattr(irls_direct, "build_centered_system", recorded)
    return builds


def test_a_fit_started_where_another_ended_reads_the_data_that_fit_built_there(recorded_builds):
    # Removing the final system's hand-over rebuilds the weighted Gram at the weights the
    # first fit ended at, a pass over every row for data already in hand.
    dm, groups, y = _poisson_problem()
    owner = centered_system._InitialDataReuse()
    first, _, _ = _fit_from(dm, groups, y, 1.0, (np.zeros(2), 0.0), owner, recorded_builds)
    ended_at = recorded_builds[-1]  # the first fit's final system
    start = (first.beta.copy(), float(first.intercept))
    before = len(recorded_builds)
    second, inverse, gram = _fit_from(dm, groups, y, 3.0, start, owner, recorded_builds)

    assert second.n_iter >= 1
    assert not any(np.array_equal(W, ended_at) for W in recorded_builds[before:-1])
    # the same answer as a fit that rebuilds everything
    expected, expected_inverse, expected_gram = _fit_from(
        dm, groups, y, 3.0, start, None, recorded_builds
    )
    np.testing.assert_array_equal(second.beta, expected.beta)
    assert second.intercept == expected.intercept and second.n_iter == expected.n_iter
    np.testing.assert_array_equal(inverse, expected_inverse)
    np.testing.assert_array_equal(gram, expected_gram)


def test_trials_that_start_from_one_point_all_read_it_after_a_sibling_ended_elsewhere(
    recorded_builds,
):
    # A rejected trial's final system must not displace the starting point's system that the
    # trials after it start from.
    dm, groups, y = _poisson_problem()
    owner = centered_system._InitialDataReuse()
    base, _, _ = _fit_from(dm, groups, y, 1.0, (np.zeros(2), 0.0), owner, recorded_builds)
    start = (base.beta.copy(), float(base.intercept))
    starting_weights = recorded_builds[-1]
    for penalty in (3.0, 9.0, 27.0):  # siblings: each ends elsewhere, none is taken
        before = len(recorded_builds)
        _fit_from(dm, groups, y, penalty, start, owner, recorded_builds)
        assert not any(np.array_equal(W, starting_weights) for W in recorded_builds[before:-1])


def test_a_fit_at_other_weights_does_not_read_a_retained_final_system(recorded_builds):
    dm, groups, y = _poisson_problem()
    owner = centered_system._InitialDataReuse()
    _fit_from(dm, groups, y, 1.0, (np.zeros(2), 0.0), owner, recorded_builds)
    before = len(recorded_builds)
    other = (np.array([0.05, -0.02]), 0.1)
    result, _, gram = _fit_from(dm, groups, y, 3.0, other, owner, recorded_builds)
    expected, _, expected_gram = _fit_from(dm, groups, y, 3.0, other, None, recorded_builds)

    assert len(recorded_builds) - before >= 1
    np.testing.assert_array_equal(result.beta, expected.beta)
    np.testing.assert_array_equal(gram, expected_gram)


@pytest.mark.parametrize(
    "design", ["sparse_ssp", "support_compressed", "spline_by_category_tables"]
)
def test_an_exact_reml_fit_builds_no_centred_system_twice_at_the_same_weights(
    monkeypatch, recorded_builds, design
):
    # By the weights and working response themselves, not by a ledger's flag: a fit's first
    # system is the final system of the fit it started from, on each design type the owner
    # serves, and on the table route as on the rungs.
    rng = np.random.default_rng(3)
    n = 1500
    x = rng.uniform(size=n)
    x = x if design == "sparse_ssp" else np.round(x, 2)  # 101 values: a lossless support
    level = rng.choice(list("abcd"), n)
    y = rng.poisson(np.exp(0.3 + np.sin(3 * x) + 0.3 * (level == "b"))).astype(float)
    responses = []
    original = irls_direct.build_centered_system

    def recorded_with_response(**kwargs):
        responses.append(np.array(kwargs["z_off"], copy=True))
        return original(**kwargs)

    monkeypatch.setattr(irls_direct, "build_centered_system", recorded_with_response)
    model = SuperGLM(
        family="poisson",
        discrete=False,
        features={"x": Spline(n_knots=6), "level": Categorical()},
        interactions=[("x", "level")] if design == "spline_by_category_tables" else [],
    )
    frame = pd.DataFrame({"x": x, "level": level})
    forced = _lookup_tables.table_gram_config(design == "spline_by_category_tables" or None)
    with forced:
        model.fit_reml(frame, y)

    kinds = {type(group).__name__ for group in model._dm.group_matrices}
    assert {
        "sparse_ssp": "SparseSSPGroupMatrix",
        "support_compressed": "SupportCompressedSSPGroupMatrix",
        "spline_by_category_tables": "SupportCompressedSplineCategoricalGroupMatrix",
    }[design] in kinds
    systems = [(W.tobytes(), z.tobytes()) for W, z in zip(recorded_builds, responses, strict=True)]
    assert len(systems) > 5
    assert len(set(systems)) == len(systems)
