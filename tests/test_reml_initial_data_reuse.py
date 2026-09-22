"""Initial trial data reuse must not reuse a penalty or a final system."""

import gc
import weakref

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from superglm import SuperGLM
from superglm.distributions import Poisson
from superglm.features.spline import Spline
from superglm.group_matrix import (
    CategoricalGroupMatrix,
    DenseGroupMatrix,
    DesignMatrix,
    SparseSSPGroupMatrix,
    SupportCompressedSSPGroupMatrix,
)
from superglm.links import LogLink
from superglm.reml import direct
from superglm.solvers import centered_system, irls_direct
from superglm.types import GroupSlice


@pytest.mark.parametrize(
    "kind", ["dense", "sparse_ssp", "support_compressed", "mixed_categorical", "custom_categorical"]
)
def test_initial_data_is_built_once_but_both_final_systems_are_fresh(monkeypatch, kind):
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
    dm = DesignMatrix([DenseGroupMatrix(np.array([[-1.0], [0.0], [1.0]]))], n=3, p=1)
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
    "change", ["weights", "rhs", "signed_zero", "shape", "dtype", "centering", "key"]
)
def test_changed_initial_contract_must_miss(change):
    # Bypassing either the context/state or actual binary64 operand guard is unsafe.
    entry, W, z, state, _ = _seed_entry()
    key = entry.key
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
    else:
        key = ("changed warm state or coordinates",)
    assert entry.take(key, W, z, state, np.ones((1, 1))) is None


@pytest.mark.parametrize("penalty, expected_hessian", [(3.0, 5.0), (-3.0, 0.0)])
def test_reused_data_gets_fresh_penalty_repair_and_centering_state(penalty, expected_hessian):
    # Retaining an old Hessian or skipping the existing PSD repair fails this.
    entry, W, z, state, after = _seed_entry()
    assert state != after  # This real fixture changes the raw-moment safety latch.
    reused = entry.take(entry.key, W, z, state, np.array([[penalty]]))
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
def test_line_search_owner_releases_entry_on_return_or_exception(monkeypatch, interrupt):
    # Real tiny REML scheduling, without asserting any platform-specific
    # terminal-trial count or Armijo rounding outcome.
    created = []
    trials = []
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
            owner = next(i for i, reference in enumerate(created) if reference() is initial)
            trials.append((kwargs["debug_context"]["reml_iteration"], owner))
        result = original_fit(**kwargs)
        if interrupt and kwargs.get("_initial_data_reuse") is not None:
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
    assert created  # The fixture enters a line search, without fixing its trial count.
    assert trials
    by_iteration = {}
    for iteration, owner in trials:
        assert by_iteration.setdefault(iteration, owner) == owner
    assert len(set(by_iteration.values())) == len(by_iteration)
    gc.collect()
    assert all(reference() is None for reference in created)
