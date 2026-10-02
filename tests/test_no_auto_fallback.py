"""No fallback reason is reachable under ``direct_solve="auto"`` (one-engine design §6, §15 stage 4).

The routing inventory of the one-engine work (43 sites, 2026-09-29; site 44,
the exact line search's refused trial, added at review) listed every place a
direct fit could change solver or arithmetic route.  ``INVENTORY`` records
where each site's disposition is checked; the checks are the behavioural tests
it names, here and in the files it cites, not the table itself.  Each site now
has one of these dispositions:

- ``structure``: a decision the model's terms, their sizes, the penalties the
  user fixed or an override's pattern make, before any row is read.  The
  ``fallback_reason`` it publishes says why ``auto`` takes gram for those
  terms; it never changes for weights, values or lambda values, which the
  test below pins.
- ``pattern``: the level-code pattern as symbolic input (decision 11), like a
  sparse Cholesky's symbolic analysis, or a summation order it sets.
- ``deleted``: the net is gone.  What remains when the engine cannot proceed
  is one clear error naming the cause (decision 6), never another solver.
- ``one method``: a data-dependent step inside one method (a pivot, a
  Levenberg shift, a halved step), with no change of solver.
- ``post-fit``: inference after the fit; it does not route the fit.
- ``stage 5``: gram's own internals, which the empty tree replaces in stage 5.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import scipy.sparse as sp

import superglm.solvers.irls_direct as irls_direct
from superglm.distributions import Poisson
from superglm.group_matrix import (
    DenseGroupMatrix,
    DesignMatrix,
    FactorSmoothGroupMatrix,
    GroupMatrix,
    RandomEffectGroupMatrix,
)
from superglm.links import LogLink
from superglm.solvers.structured import NestedSchurFactor, resolve_structured_backend
from superglm.types import GroupSlice, LinearConstraintSet, PenaltyComponent

# site number in the inventory: (site, disposition, where it is checked)
INVENTORY: dict[int, tuple[str, str, str]] = {
    1: ("constraints or SCOP on a group", "structure", "structural[constraints]"),
    2: ("term mix and dominant block", "structure", "structural[two-factor-smooths]"),
    3: ("dominant width != matrix width", "deleted", "invariant_violation_is_an_error"),
    4: ("Rule B nested chain", "pattern", "test_random_effect_nesting.py"),
    5: ("zero-penalty random effect leaves the chain", "structure", "structural[zero-penalty]"),
    6: ("family table declines signed rows", "deleted", "test_structured_irls.py signed"),
    7: ("Tweedie power refits route by p", "deleted", "follows from site 6"),
    8: ("observed_rows flag", "deleted", "follows from site 6"),
    9: ("override declines a longer chain", "structure", "structural[override-chain]"),
    10: ("random-effect price", "structure", "structural[below-crossover]"),
    11: ("fs price", "structure", "structural[fs-below-crossover]"),
    12: ("sz price", "structure", "structural[sz]"),
    13: ("dense memory budget", "deleted", "never committed: no commit's src reads memory"),
    14: ("override couples the leaf", "structure", "structural[override-leaf]"),
    15: ("zero penalty on the leaf", "structure", "structural[zero-penalty-leaf]"),
    16: ("level weight under an override", "deleted", "resolver_reads_no_rows"),
    17: ("fs override local blocks", "structure", "structural[fs-override-singular]"),
    18: ("fs singular local block", "structure", "structural[fs-zero-component]"),
    19: ("structured without compact penalties", "structure", "no_compact_penalties"),
    20: ("layout dispatch to the scalar factor", "deleted", "a_random_effect_is_always_a_chain"),
    21: ("refusal retry onto gram", "deleted", "refusal_is_one_clear_error"),
    22: ("REML driver latch", "deleted", "no_reml_latch"),
    23: ("finalize latch", "deleted", "no_reml_latch"),
    24: ("discrete refused trial", "one method", "test_structured_irls.py discrete trial"),
    25: ("observed geometry refusal", "one method", "step halved or not converged"),
    26: ("W(rho) correction refusal", "one method", "not converged, no solver change"),
    27: ("nested Cholesky to eigh", "one method", "stage 0 verified pivoted Cholesky"),
    28: ("scalar and block SVD fallbacks", "deleted", "factors retired with shims"),
    29: ("sz border LDL fallback", "deleted", "stage 3 balance tree"),
    30: ("sz public-rank certificate", "deleted", "stage 3 balance tree"),
    31: ("border centre spread rule", "deleted", "stage 0 prior-weighted centre"),
    32: ("sparse indicator pass", "pattern", "summation order of one pass"),
    33: ("weighted scatter sparsity", "pattern", "summation order of one pass"),
    34: ("signed moment split", "deleted", "signed by type (stage 1, stage 4)"),
    35: ("cancelled column norms", "post-fit", "estimability numbers only"),
    36: ("structured estimability check", "post-fit", "inference only"),
    37: ("gram centring rungs", "stage 5", "gram internals"),
    38: ("gram penalty projection", "stage 5", "gram internals"),
    39: ("decompose_gram ladder", "stage 5", "gram internals"),
    40: ("observed to Fisher switch", "deleted", "stage 1 curvature by type"),
    41: ("BLAS threads by width", "structure", "thread count, not a route"),
    42: ("direct or coordinate-descent solver", "structure", "configuration"),
    43: ("selected inverse block cap", "structure", "post-fit request size"),
    44: ("exact refused trial", "one method", "test_structured_irls.py exact trial"),
}


# ── structure: decisions that read no row ────────────────────────────────


def _groups(matrices: list[GroupMatrix], names: list[str]) -> list[GroupSlice]:
    groups, start = [], 0
    for matrix, name in zip(matrices, names, strict=True):
        penalized = isinstance(matrix, RandomEffectGroupMatrix | FactorSmoothGroupMatrix)
        groups.append(GroupSlice(name, start, start + matrix.shape[1], penalized=penalized))
        start += matrix.shape[1]
    return groups


def _factor_smooth(n_levels: int, basis: str, rows: int = 6, k: int = 2) -> FactorSmoothGroupMatrix:
    x = np.tile(np.linspace(-1.0, 1.0, rows), n_levels)
    local = np.column_stack([x**power for power in range(k)])
    return FactorSmoothGroupMatrix(
        sp.csr_matrix(local),
        np.repeat(np.arange(n_levels, dtype=np.intp), rows),
        n_levels,
        natural_map=np.eye(k),
        levels=tuple(f"g{level}" for level in range(n_levels)),
        repeated_penalty_components=(("wiggle", np.diag([0.0] * (k - 1) + [1.0])),),
        factor_basis=basis,
    )


def _case(name: str):
    """``(matrices, groups, lambda2, S_override)`` that decide gram, or decline a chain, by structure."""
    rng = np.random.default_rng(0)
    n = 240
    if name in ("two-factor-smooths", "fs-below-crossover", "sz", "fs-zero-component"):
        levels = {"sz": 40, "fs-below-crossover": 3}.get(name, 40)
        smooth = _factor_smooth(levels, "sz" if name == "sz" else "fs")
        rows = smooth.shape[0]
        matrices: list[GroupMatrix] = [DenseGroupMatrix(rng.normal(size=(rows, 3))), smooth]
        names = ["x", f"s:{smooth.factor_basis}"]
        if name == "two-factor-smooths":
            matrices.append(_factor_smooth(levels, "fs"))
            names.append("t:fs")
        groups = _groups(matrices, names)
        lambda2 = {f"{group.name}:wiggle": 1.0 for group in groups[1:]}
        if name == "fs-zero-component":
            lambda2[f"{names[1]}:wiggle"] = 0.0
        return matrices, groups, lambda2, None
    if name == "fs-override-singular":
        smooth = _factor_smooth(40, "fs")
        matrices = [DenseGroupMatrix(rng.normal(size=(smooth.shape[0], 3))), smooth]
        groups = _groups(matrices, ["x", "s:fs"])
        override = np.zeros((groups[-1].end, groups[-1].end))
        diagonal = np.arange(groups[1].start, groups[1].end)
        override[diagonal, diagonal] = 1.0
        override[diagonal[:2], diagonal[:2]] = 0.0  # level g0's local block is zero
        return matrices, groups, None, override
    leaf = rng.integers(0, 60, n)
    parent = leaf // 6
    if name == "below-crossover":
        matrices = [
            DenseGroupMatrix(rng.normal(size=(n, 40))),
            RandomEffectGroupMatrix(leaf % 4, 4),
        ]
        groups = _groups(matrices, ["x", "g"])
        return matrices, groups, {"g": 1.0}, None
    matrices = [
        DenseGroupMatrix(rng.normal(size=(n, 3))),
        RandomEffectGroupMatrix(parent, 10),
        RandomEffectGroupMatrix(leaf, 60),
    ]
    groups = _groups(matrices, ["x", "p", "g"])
    lambda2 = {"p": 1.0, "g": 1.0}
    if name == "constraints":
        groups[0].constraints = LinearConstraintSet(A=np.eye(1, 3), b=np.zeros(1))
        return matrices, groups, lambda2, None
    if name in ("zero-penalty", "zero-penalty-leaf"):
        lambda2["p" if name == "zero-penalty" else "g"] = 0.0
        return matrices, groups, lambda2, None
    if name == "override-chain":
        # couples the two coarser levels of a three-level chain: the chain
        # declines to its leaf, and they join the border, which may be dense
        matrices.insert(1, RandomEffectGroupMatrix(parent // 2, 5))
        groups = _groups(matrices, ["x", "q", "p", "g"])
    override = np.zeros((groups[-1].end, groups[-1].end))
    override[np.arange(3, groups[-1].end), np.arange(3, groups[-1].end)] = 1.0
    if name == "override-chain":
        coarse, middle = groups[1].start, groups[2].start
        override[coarse, middle] = override[middle, coarse] = 0.1
    else:
        # couples a leaf level to the border: the leaf itself is not structured
        override[0, groups[2].start] = override[groups[2].start, 0] = 0.1
        override[0, 0] = 1.0
    return matrices, groups, None, override


# name: (auto takes the structured solver, its chain, the reason it publishes)
_STRUCTURAL = {
    "constraints": (False, (), "constraints"),
    "two-factor-smooths": (False, (), "at most one FactorSmooth"),
    "zero-penalty": (True, (2,), None),  # the parent leaves the chain for the border
    "override-chain": (True, (3,), None),  # declined to the leaf: nested_fallback_reason
    "below-crossover": (False, (1,), "crossover"),
    "fs-below-crossover": (False, (1,), "crossover"),
    "sz": (True, (1,), None),  # the balance tree on the size rule (retired: sz to gram)
    "override-leaf": (False, (), "S_override"),
    "zero-penalty-leaf": (False, (), "zero penalty"),
    "fs-override-singular": (False, (), "singular local penalty block"),
    "fs-zero-component": (False, (), "zero penalty component"),
}


def _scaled(lambda2, factor):
    if lambda2 is None:
        return None
    return {name: value * factor for name, value in lambda2.items()}


@pytest.mark.parametrize("name", _STRUCTURAL)
def test_structural(name: str) -> None:
    """Every gram decision, or declined chain, ``auto`` makes is one of structure.

    It is the same for lambda values scaled by 1e-7 and 1e6 (a lambda the user
    fixed at zero stays zero: that is the specification), and forced
    ``structured`` either refuses the same terms or keeps the same chain.
    """
    matrices, groups, lambda2, override = _case(name)
    width = groups[-1].end
    decisions = [
        resolve_structured_backend(
            matrices,
            groups,
            direct_solve="auto",
            coefficient_width=width,
            lambda2=_scaled(lambda2, factor),
            S_override=override,
        )
        for factor in (1.0, 1e-7, 1e6)
    ]
    first = decisions[0]
    structured, chain, reason = _STRUCTURAL[name]
    assert first.use_structured is structured
    if chain:
        assert first.chain_group_indices == chain
    if reason is not None:
        assert reason in first.fallback_reason
    if name == "override-chain":
        assert "must be diagonal" in first.nested_fallback_reason
    for decision in decisions[1:]:
        assert decision.use_structured == first.use_structured
        assert decision.fallback_reason == first.fallback_reason
        assert decision.chain_group_indices == first.chain_group_indices
        assert decision.nested_fallback_reason == first.nested_fallback_reason
    if not first.use_structured and first.auto_cost_ratio is None:
        with pytest.raises(ValueError, match="ineligible"):
            resolve_structured_backend(
                matrices,
                groups,
                direct_solve="structured",
                coefficient_width=width,
                lambda2=lambda2,
                S_override=override,
            )


def test_resolver_reads_no_rows() -> None:
    """The resolver takes no rows at all: no weight, family, link or response."""
    parameters = inspect.signature(resolve_structured_backend).parameters
    for name in ("row_weights", "weights", "family", "link", "y"):
        assert name not in parameters


def test_invariant_violation_is_an_error() -> None:
    """Site 3: inconsistent coefficient geometry was a gram route under auto; it is a bug."""
    matrices, groups, lambda2, _ = _case("zero-penalty")
    groups[2] = GroupSlice("g", groups[2].start, groups[2].end - 1, penalized=True)
    with pytest.raises(RuntimeError, match="inconsistent coefficient geometry"):
        resolve_structured_backend(
            matrices, groups, direct_solve="auto", coefficient_width=groups[-1].end
        )


def test_a_random_effect_is_always_a_chain() -> None:
    """Site 20: the scalar layout is gone; every random-effect leaf resolves a chain."""
    from superglm.solvers.structured import get_structured_layout

    matrices, groups, _lambda2, _ = _case("zero-penalty")
    lone = [matrices[0], RandomEffectGroupMatrix(np.arange(240) % 60, 60)]
    lone_groups = _groups(lone, ["x", "g"])
    decision = resolve_structured_backend(
        lone, lone_groups, direct_solve="auto", coefficient_width=lone_groups[-1].end
    )
    assert decision.chain_group_indices == (1,)
    with pytest.raises(ValueError, match="chain of one"):
        get_structured_layout(DesignMatrix(lone, n=240, p=63), lone_groups, dominant_group_index=1)


# ── deleted: runtime nets ───────────────────────────────────────────────


def _poisson_problem():
    rng = np.random.default_rng(701)
    n, levels = 420, 60
    codes = rng.integers(0, levels, size=n, dtype=np.intp)
    numeric = rng.normal(size=(n, 2))
    eta = -0.25 + numeric @ np.array([0.3, -0.18]) + rng.normal(scale=0.22, size=levels)[codes]
    matrices = [DenseGroupMatrix(numeric), RandomEffectGroupMatrix(codes, levels)]
    groups = [
        GroupSlice(name="x", start=0, end=2, penalized=False),
        GroupSlice(name="g", start=2, end=2 + levels, penalized=True),
    ]
    penalties = [
        PenaltyComponent(
            name="g",
            group_name="g",
            group_index=1,
            group_sl=groups[1].sl,
            omega_raw=None,
            penalty_kind="identity",
        )
    ]
    return dict(
        X=DesignMatrix(matrices, n=n, p=2 + levels),
        y=rng.poisson(np.exp(eta)).astype(float),
        weights=rng.uniform(0.4, 2.2, size=n),
        family=Poisson(),
        link=LogLink(),
        groups=groups,
        lambda2={"g": 2.75},
        reml_penalties=penalties,
        weight_semantics="prior",
    )


def test_no_compact_penalties() -> None:
    """Site 19: without compact penalties or an override the call path is gram, by structure."""
    arguments = _poisson_problem() | {"reml_penalties": None}
    result, _ = irls_direct.fit_irls_direct(direct_solve="auto", **arguments)
    assert result.direct_backend == "gram"
    assert "reml_penalties" in result.direct_fallback_reason


@pytest.mark.parametrize("site", ["build", "solve", "logdet", "trace"])
def test_refusal_is_one_clear_error(monkeypatch, site: str) -> None:
    """Sites 21 and 28: a factor that cannot proceed raises ``StructuredSolverError`` under
    auto, naming the cause and the gram escape, and no fit runs on another solver.

    Mutation: restoring the retry turns the error into a gram fit and a second call.
    """
    cause = "Nested chain 'g' has a tree pivot 0 within its certified uncertainty 1e-16."

    def refuse(*_args, **_kwargs):
        raise np.linalg.LinAlgError(cause)

    if site == "build":
        monkeypatch.setattr(irls_direct, "build_augmented_structured_factor", refuse)
    else:
        from superglm.solvers.structured import ProfiledNestedSchurFactor

        owner, method = {
            "solve": (NestedSchurFactor, "solve_data"),
            "logdet": (NestedSchurFactor, "logdet"),
            "trace": (ProfiledNestedSchurFactor, "trace_inverse_operator"),
        }[site]
        monkeypatch.setattr(owner, method, refuse)
    calls: list[str] = []
    fit_once = irls_direct._fit_irls_direct_once

    def spy(**kwargs):
        calls.append(kwargs["direct_solve"])
        return fit_once(**kwargs)

    monkeypatch.setattr(irls_direct, "_fit_irls_direct_once", spy)
    with pytest.raises(irls_direct.StructuredSolverError) as refused:
        irls_direct.fit_irls_direct(direct_solve="auto", **_poisson_problem())
    assert cause.rstrip(".") in str(refused.value)
    assert "direct_solve='gram'" in str(refused.value)
    assert calls == ["auto"]


def test_a_refusals_notes_reach_the_clear_error() -> None:
    """``_structured_solver_errors`` keeps the notes a factor's refusal carries.

    ``_build_iterate_factor`` attaches the largest Levenberg shift's refusal
    to an observed iterate's own refusal as a note, and ``str(error)`` leaves
    notes out, so the clear error named only the unshifted cause (#433).
    Fails without the notes in the message.
    """
    refusal = np.linalg.LinAlgError(
        "Nested chain 'g' has a tree pivot 0 within its certified uncertainty 1e-16."
    )
    refusal.add_note(
        "No Levenberg shift up to 1e+08 certified the iterate either; the largest "
        "shift's factor refused with: border curvature -0.03 is material."
    )
    with pytest.raises(irls_direct.StructuredSolverError) as refused:
        with irls_direct._structured_solver_errors():
            raise refusal
    message = str(refused.value)
    assert "tree pivot 0 within its certified uncertainty 1e-16." in message
    assert (
        "No Levenberg shift up to 1e+08 certified the iterate either; the largest "
        "shift's factor refused with: border curvature -0.03 is material." in message
    )
    assert "direct_solve='gram'" in message


def test_a_non_finite_outer_hessian_is_the_clear_error(monkeypatch) -> None:
    """A structured REML outer Hessian out of float64 range has no Newton step
    (Opus review of #425: a uniform prior weight of 1e-200 does it): the fit
    raises the structured path's one clear error, naming the gram escape,
    instead of numpy's "Eigenvalues did not converge" or a later non-finite
    smoothing parameter.

    The Hessian is made non-finite directly, so the test pins the guard
    whatever the factor does at extreme weights.  Fails without the
    finiteness check before the modified Newton step.
    """
    import pandas as pd

    import superglm.reml.direct as direct_reml
    from superglm import Numeric, RandomEffect, SuperGLM

    hessian = direct_reml.reml_direct_hessian

    def overflowing(*args, **kwargs):
        return np.full_like(hessian(*args, **kwargs), np.inf)

    monkeypatch.setattr(direct_reml, "reml_direct_hessian", overflowing)
    rng = np.random.default_rng(0)
    n = 600
    codes = rng.integers(0, 60, n)
    frame = pd.DataFrame({"x": rng.normal(size=n), "g": [f"g{c:02d}" for c in codes]})
    y = rng.poisson(np.exp(0.2 * frame["x"] + rng.normal(0, 0.3, 60)[codes])).astype(float)
    model = SuperGLM(
        family="poisson",
        features={"x": Numeric(), "g": RandomEffect()},
        selection_penalty=0,
        direct_solve="structured",
    )
    with pytest.raises(irls_direct.StructuredSolverError, match="direct_solve='gram'"):
        model.fit_reml(frame, y)


@pytest.mark.parametrize("discrete", [False, True], ids=["exact", "discrete"])
def test_no_reml_latch(monkeypatch, discrete: bool) -> None:
    """Sites 22 and 23: PIRLS results doctored onto gram with a reason (what the
    retry produced) move no later REML fit, and the terminal refit keeps the
    model's own direct_solve.

    Mutation: restoring the driver latch turns the later driver modes into
    "gram"; restoring the finalize latch turns the terminal refit's.
    """
    import pandas as pd

    import superglm.model.reml_finalize as reml_finalize
    import superglm.reml.direct as direct_reml
    import superglm.reml.discrete as discrete_reml
    from superglm import Numeric, RandomEffect, SuperGLM

    driver = discrete_reml if discrete else direct_reml
    modes: dict[str, list[str]] = {"driver": [], "finalize": []}
    for key, module in (("driver", driver), ("finalize", reml_finalize)):
        original = module.fit_irls_direct

        def doctored(*args, _fit=original, _modes=modes[key], **kwargs):
            _modes.append(kwargs["direct_solve"])
            result = _fit(*args, **kwargs)
            if _modes is modes["driver"]:
                result[0].direct_backend = "gram"
                result[0].direct_fallback_reason = "a doctored refusal"
            return result

        monkeypatch.setattr(module, "fit_irls_direct", doctored)
    rng = np.random.default_rng(3)
    n = 600
    codes = rng.integers(0, 60, n)
    frame = pd.DataFrame({"x": rng.normal(size=n), "g": [f"g{c:02d}" for c in codes]})
    y = rng.poisson(np.exp(0.2 * frame["x"] + rng.normal(0, 0.3, 60)[codes])).astype(float)
    model = SuperGLM(
        family="poisson",
        features={"x": Numeric(), "g": RandomEffect()},
        selection_penalty=0,
        discrete=discrete,
        direct_solve="auto",
    )
    model.fit_reml(frame, y)
    assert len(modes["driver"]) > 1 and set(modes["driver"]) == {"auto"}
    assert modes["finalize"] == ["auto"]
    assert model.result.direct_backend == "structured"
    assert model.result.direct_fallback_reason is None
