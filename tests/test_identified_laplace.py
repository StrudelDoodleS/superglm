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
