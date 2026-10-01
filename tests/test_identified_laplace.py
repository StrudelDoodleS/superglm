"""The identified part of the REML criterion and the score stop (one-engine design §3.8, §3.9).

The stage-0 verifier found REML path-dependent on the §3.9 fixtures: ``xt`` is
5 everywhere except two rows of prior weight 1e-15, so the likelihood is flat
along it to within float64, yet its log-curvature entered ``log|H|`` at full
weight and ``log|H|`` moved with wherever PIRLS happened to stop.  REML's
smoothing parameters then depended on the warm-start path (``offset_rare``
Poisson: lambda_u 19.0 against 10.47, silently).  And on canonical links a
deviance-stopped PIRLS left coefficients whose rows carry a small share of the
likelihood short of their mode, so the criterion drifted with every warm start
and REML never converged (``raw_1e8`` binomial).  Each test fails under the
mutation named in its docstring, which the change record demonstrates.

Tolerances: two criteria at the same smoothing parameters and certified modes
agree to the precision the REML stop resolves them, ``reml_tol (1 + |V|)``
(``reml.convergence``), which is what the fits ask for.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from superglm.reml.identified import WeakIdentificationWarning
from tests.test_one_engine_stage0 import _adversarial_base, _fit, _response

REML_TOL = 1e-9


def _rare_column(base: float, family: str, *, positive: bool = False):
    """``offset_rare`` (``base`` 5) or ``tiny_weight_rare`` (``base`` 0): ``xt`` carried by two rows of weight 1e-15."""
    frame, eta, weight, rng, levels = _adversarial_base()
    rows = np.flatnonzero(weight > 0)[:2]
    frame["xt"] = base
    frame.loc[rows, "xt"] = [base + 1.0, base + 2.0]
    weight = weight.copy()
    weight[rows] = 1e-15
    if positive:
        weight = np.where(weight == 0.0, 0.01, weight)
    y = _response(family, eta, rng) if family != "binomial" else _binomial(eta, rng)
    return frame, y, weight, levels


def _binomial(eta, rng):
    return (rng.uniform(size=len(eta)) < 1.0 / (1.0 + np.exp(-eta))).astype(float)


def _fixed(model, frame, y, weight, family, numerics, levels, *, solve):
    """A fresh fit at ``model``'s smoothing parameters, from the default start."""
    lambdas = model._reml_lambdas
    return _fit(
        frame,
        y,
        weight,
        family,
        numerics,
        levels,
        lam_g=lambdas["g"],
        lam_u=lambdas["u"],
        solve=solve,
    )


@pytest.mark.parametrize("solve", ["auto", "gram"])
def test_the_reml_criterion_does_not_move_with_the_flat_coefficient(solve):
    """Section 3.9, T3 row "weakly identified coefficient inside log|H|".

    The REML fit reaches its smoothing parameters along a warm-started path
    that leaves ``xt`` wherever it stopped; a fresh fit at the same smoothing
    parameters starts ``xt`` elsewhere.  With ``xt`` left out of the Laplace
    approximation (``reml.identified``) the two criteria agree to the REML
    precision.  With it inside, the engine, which resolves and moves ``xt``,
    ends unconverged (at stage 0 the criteria differed by ~4.5 and the free fit
    ended at lambda_u 19.0 against 10.47): the ``auto`` case fails with
    ``IdentifiedLaplace`` returning the full ``H``.  Gram, whose factor does
    not move ``xt``, is the control.
    """
    frame, y, weight, levels = _rare_column(5.0, "poisson")
    numerics = ["x1", "xt"]
    free = _fit(frame, y, weight, "poisson", numerics, levels, solve=solve)
    fixed = _fixed(free, frame, y, weight, "poisson", numerics, levels, solve=solve)
    xt = next(group for group in free._groups if group.name == "xt").start
    assert free._reml_profile["reml_laplace_excluded"] == (xt,)
    assert free._reml_result.converged
    value = free._reml_result.objective
    assert abs(value - fixed._reml_result.objective) <= REML_TOL * (1.0 + abs(value))


def test_both_backends_select_the_same_smoothing_parameters_on_the_identified_part():
    """Section 3.9 and T2's identified part: the smoothing parameters of the other terms.

    Fails with ``IdentifiedLaplace`` returning the full ``H`` (auto lambda_u
    19.0 against gram's 10.47).  The bound on the log smoothing parameters is
    the REML stop's: at a stationary point the gradient is within
    ``reml_tol (1 + |V|)`` of zero, so each fit's ``rho`` is within that over
    the criterion's curvature of the optimum (``hess_diag`` of the terminal
    freeze decision), twice for two fits.
    """
    frame, y, weight, levels = _rare_column(5.0, "poisson")
    numerics = ["x1", "xt"]
    auto = _fit(frame, y, weight, "poisson", numerics, levels, solve="auto")
    gram = _fit(frame, y, weight, "poisson", numerics, levels, solve="gram")
    decision = auto._reml_profile["reml_freeze_decision"]
    for position, name in enumerate(decision["names"]):
        curvature = decision["hess_diag"][position]
        bound = 2.0 * REML_TOL * decision["score_scale"] / curvature
        gap = abs(np.log(auto._reml_lambdas[name]) - np.log(gram._reml_lambdas[name]))
        assert gap <= bound, (name, gap, bound)


@pytest.mark.parametrize(
    ("family", "base"), [("poisson", 0.0), ("binomial", 5.0), ("gaussian", 0.0), ("tweedie", 5.0)]
)
def test_weak_identification_is_disclosed_for_every_family(family, base):
    """Section 3.9's disclosure, for canonical links as well as observed ones.

    The stage-0 verifier found canonical-link fits carried no flag at all.
    The finished fit names ``xt`` in the profile, in ``diagnostics()`` and in
    one ``WeakIdentificationWarning``.  Fails with the terminal disclosure
    (``reml_finalize._disclose_weak_identification``) removed.
    """
    frame, y, weight, levels = _rare_column(base, family, positive=family == "tweedie")
    numerics = ["x1", "xt"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = _fit_recording(frame, y, weight, family, numerics, levels)
    xt = next(group for group in model._groups if group.name == "xt").start
    assert xt in model._reml_profile["reml_weakly_identified"]
    assert model.diagnostics()["xt"]["weakly_identified"] == [0]
    assert "xt[0]" in model.diagnostics()["_model"]["weakly_identified"]
    named = [w for w in caught if issubclass(w.category, WeakIdentificationWarning)]
    assert len(named) == 1 and "xt[0]" in str(named[0].message)


def test_a_fit_without_rare_rows_flags_nothing():
    """The negative control of the disclosure: the same design at ordinary weights."""
    frame, eta, weight, rng, levels = _adversarial_base()
    frame["xt"] = frame["x1"] ** 2
    y = _response("poisson", eta, rng)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = _fit_recording(frame, y, weight, "poisson", ["x1", "xt"], levels)
    assert model._reml_profile["reml_weakly_identified"] == ()
    assert model._reml_profile["reml_laplace_excluded"] == ()
    assert not [w for w in caught if issubclass(w.category, WeakIdentificationWarning)]


def test_the_score_stop_publishes_the_criterion_at_the_mode():
    """Section 3.8 for canonical links; T3 row "deviance stop for Fisher REML fits".

    The verifier's ``raw_1e8`` binomial at lambda_g 1e-7 and weights x1e4: the
    one-row separated levels creep towards their penalty bound, a few
    thousandths of a unit of log-odds per PIRLS iteration (every full Newton
    step overshoots and the line search keeps 1/256 of it), which no deviance
    change sees, while their log-curvature enters ``log|H|``.  A
    deviance-stopped fit published a criterion 21.3 above its value at the
    mode, as converged.  Stopping every REML PIRLS and the terminal refit on
    the certificate's centred score publishes such a fit as what it is: its
    score stops contracting short of the bar (``stagnation_window``), and the
    mode is published as not converged, never refused.  Fails with
    ``convergence="deviance"`` for the Fisher REML fits (``reml.direct``) and
    their terminal refit (``reml_finalize``), which publish it converged.
    """
    frame, eta, weight, rng, levels = _adversarial_base()
    n = len(frame)
    frame["xr"] = 1e8 + rng.normal(size=n)
    frame["vr"] = 1e3 * (1e8 + rng.normal(size=n))
    eta = eta + 0.1 * (frame["xr"].to_numpy() - 1e8)
    y = _binomial(eta, rng)
    weight = weight * 1e4
    numerics = ["x1", "xr", "vr"]
    engine = _fit(frame, y, weight, "binomial", numerics, levels, lam_g=1e-7)
    assert engine._reml_profile["direct_backend"] == "structured"
    assert engine._reml_profile["reml_terminal_mode_certified"] is False
    assert engine._reml_profile["reml_terminal_mode_termination"] == "score_stagnated"
    assert engine._reml_result.converged is False


def _fit_recording(frame, y, weight, family, numerics, levels):
    """``_fit`` with the caller's warning filters in force."""
    from superglm import Categorical, Numeric, RandomEffect, Spline, SuperGLM, Tweedie

    spec = {name: Numeric() for name in numerics}
    spec["u"] = Spline(kind="ps", k=8)
    spec["cat"] = Categorical()
    spec["g"] = RandomEffect(levels=levels)
    model = SuperGLM(
        family=Tweedie(p=1.5) if family == "tweedie" else family,
        features=spec,
        selection_penalty=0,
    )
    model.fit_reml(frame, y, sample_weight=weight, pirls_tol=1e-10, reml_tol=REML_TOL)
    return model


# ── one factorization of H_II (review of #425) ──────────────────────────────


def _aliased_weak_frame(case: str):
    """A 40-level random effect beside columns two rows of weight 1e-15 carry.

    ``sol``: ``b = r - a`` with ``r`` weak, so the full ``H`` is singular
    (``a + b - r = 0``) and so is ``H_II`` (``a + b`` lives on the weak rows).
    ``dup``: two identical weak columns, so ``(H^+)_WW`` has rank one.
    Returns the frame, the response, the weights, the full and the reduced
    numeric columns.
    """
    import pandas as pd

    rng = np.random.default_rng(1)
    n = 240
    g = np.tile(np.arange(40), 6)
    a = rng.integers(-5, 6, n).astype(float)
    weight = np.ones(n)
    weight[:2] = 1e-15
    y = 0.3 * a + 0.1 * np.sin(g) + 0.1 * rng.normal(size=n)
    r = np.zeros(n)
    r[:2] = [1.0, 2.0]
    frame = pd.DataFrame({"a": a, "g": [f"g{c:02d}" for c in g]})
    if case == "sol":
        frame["b"], frame["r"] = r - a, r
        return frame, y, weight, ["a", "b", "r"], ["a", "b"]
    frame["r1"], frame["r2"] = r, r.copy()
    return frame, y, weight, ["a", "r1", "r2"], ["a"]


def _numeric_fit(frame, y, weight, numerics, solve, **features):
    from superglm import Numeric, RandomEffect, SuperGLM

    spec = {name: Numeric() for name in numerics} | {"g": RandomEffect()} | features
    model = SuperGLM(family="gaussian", features=spec, selection_penalty=0, direct_solve=solve)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_reml(frame, y, sample_weight=weight, reml_tol=REML_TOL)
    return model


@pytest.mark.parametrize("solve", ["gram", "structured"])
@pytest.mark.parametrize("case", ["sol", "dup"])
def test_the_identified_part_is_the_criterion_without_the_weak_columns(case, solve):
    """Sol P2 and Codex P2 on #425: ``log|H_II|``, rank and inverse from ONE factorization.

    For a Gaussian identity fit ``H_II`` is exactly the Hessian of the model
    without the excluded columns (the rows that carry them weigh 1e-15), so
    the two REML criteria are one function of the smoothing parameter and
    agree at their optima to the REML stop's ``reml_tol (1 + |V|)``.
    Jacobi's identity on the singular full factor biased the criterion by
    0.2027 (``sol``), and an excluded block of rank one left the retained
    weak direction in ``log|H|`` while the inverse dropped it (``dup``: 14.7
    on gram, 3.1 structured, and ``lambda_g`` moved by 0.7%).
    """
    frame, y, weight, full, reduced = _aliased_weak_frame(case)
    model = _numeric_fit(frame, y, weight, full, solve)
    control = _numeric_fit(frame, y, weight, reduced, solve)
    assert model.result.direct_backend == control.result.direct_backend == solve
    weak = [group.start for group in model._groups if group.name in set(full) - set(reduced)]
    assert model._reml_profile["reml_laplace_excluded"] == tuple(weak)
    assert control._reml_profile["reml_laplace_excluded"] == ()
    value = control._reml_result.objective
    assert abs(model._reml_result.objective - value) <= REML_TOL * (1.0 + abs(value))


@pytest.mark.parametrize("solve", ["gram", "structured"])
def test_a_penalty_fixed_at_zero_identifies_nothing(solve):
    """Codex P2 on #425: a component whose policy is ``off()`` adds nothing to ``S``.

    Level ``u4`` of a random effect fixed at lambda 0 lives only on two rows
    of weight 1e-15: it is weakly identified and left out of the Laplace
    approximation, which the component's mere presence used to prevent.  The
    structured rebuild cannot leave out a column of a border random-effect
    block (its complete one-hot sum is a structural generator), so that
    backend keeps the full ``H`` for inverse, determinant and rank alike and
    counts it.  Fails with ``penalized_columns`` reading component presence.
    """
    import pandas as pd

    from superglm import RandomEffect
    from superglm.types import LambdaPolicy

    rng = np.random.default_rng(3)
    n = 400
    g = np.tile(np.arange(40), 10)
    u = rng.integers(0, 4, n)
    u[:2] = 4
    weight = np.ones(n)
    weight[:2] = 1e-15
    a = rng.normal(size=n)
    y = 0.3 * a + 0.2 * np.sin(g) + 0.1 * u + 0.1 * rng.normal(size=n)
    frame = pd.DataFrame({"a": a, "g": [f"g{c:02d}" for c in g], "u": [f"u{c}" for c in u]})
    model = _numeric_fit(
        frame, y, weight, ["a"], solve, u=RandomEffect(lambda_policy=LambdaPolicy.off())
    )
    assert model._reml_profile["reml_laplace_excluded_labels"] == ("u[4]",)
    unsupported = model._reml_profile["reml_laplace_exclusion_unsupported"]
    assert (unsupported > 0) if solve == "structured" else (unsupported == 0)


def test_a_refusal_in_the_identified_part_is_the_clear_error(monkeypatch):
    """Claude Low on #425: the rebuild of the identified factor refuses as every
    structured operation does, as ``StructuredSolverError`` naming the gram escape.

    Fails with the rebuild outside ``_structured_solver_errors`` (a bare
    ``LinAlgError``).
    """
    from superglm.solvers._structured.nested import NestedSchurFactor
    from superglm.solvers.irls_direct import StructuredSolverError

    build = NestedSchurFactor.__init__

    def refusing(self, *args, excluded=(), **kwargs):
        if excluded:
            raise np.linalg.LinAlgError("Nested chain 'g' border column 1 is refused.")
        build(self, *args, excluded=excluded, **kwargs)

    monkeypatch.setattr(NestedSchurFactor, "__init__", refusing)
    frame, y, weight, full, _ = _aliased_weak_frame("sol")
    with pytest.raises(StructuredSolverError, match="direct_solve='gram'"):
        _numeric_fit(frame, y, weight, full, "structured")


def test_a_refused_identified_part_at_a_discrete_trial_halves_the_step(monkeypatch):
    """Claude Low on #425: in the discrete line search the trial factor's identified
    part is read inside the refused-trial handling, so a refusal there rejects
    the trial (counted, step halved) and the fit completes.

    Fails with the read after the ``try`` (the refusal fails the fit).
    """
    from superglm import Numeric, RandomEffect, SuperGLM
    from superglm.solvers._structured.nested import NestedSchurFactor

    build = NestedSchurFactor.__init__
    restricted: list[int] = []

    def refuse_second(self, *args, excluded=(), **kwargs):
        if excluded:
            restricted.append(len(restricted))
            if len(restricted) == 2:  # the first candidate's is the first
                raise np.linalg.LinAlgError("Nested chain 'g' border column 1 is refused.")
        build(self, *args, excluded=excluded, **kwargs)

    monkeypatch.setattr(NestedSchurFactor, "__init__", refuse_second)
    frame, y, weight, full, _ = _aliased_weak_frame("sol")
    model = SuperGLM(
        family="poisson",
        features={name: Numeric() for name in full} | {"g": RandomEffect()},
        selection_penalty=0,
        discrete=True,
        direct_solve="structured",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_reml(frame, np.round(np.exp(y)), sample_weight=weight)
    assert model._reml_profile["reml_n_refused_structured_trials"] >= 1
    assert len(restricted) > 2
