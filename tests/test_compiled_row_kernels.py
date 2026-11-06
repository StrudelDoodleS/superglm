"""The compiled rows of every SuperLSS family's likelihood against the array rows they replace.

``gamma``, ``log_normal`` (mean form), ``generalized_pareto`` and ``two_piece`` evaluate ordinary
rows with a generated kernel of ``kernels._declared`` and ``negative_binomial`` with a per-row port
of its array code.  Each declines a chunk holding a row it cannot represent, and the array route,
whose domain is wider and whose refusals are the package's, evaluates it.

The array route is the oracle here.  Both routes are backward stable, so they agree to the
roundings of a channel's terms.  ``ROUNDINGS = 2048 u`` of the row's largest entry at that order is
the generated kernels' own gate, ``C Q u`` with ``C = 2`` and ``Q = 1024``
(``tests/test_declared_severity_families.py``): the composed bound of a generated entry is at most
``Q`` times its channel scale, and the gamma's digamma series and the array code's differ by
hundreds of ``u`` there.  The two-piece and log-normal declarations are also held against mpmath,
and no worse than ``8`` times the array code's error.
"""

from __future__ import annotations

import importlib
import pkgutil

import numpy as np
import pandas as pd
import pytest

import superglm
from superglm.distributional.kernels import _declared
from superglm.distributional.kernels import gamma as gamma_kernel
from superglm.distributional.kernels import generalized_pareto as pareto_kernel
from superglm.distributional.kernels import log_normal as log_normal_kernel
from superglm.distributional.kernels import negative_binomial as nb_kernel
from superglm.distributional.kernels import two_piece as two_piece_kernel

U = 2.0**-53
ROUNDINGS = 2048.0 * U

_ROW_ROUTES = (
    (gamma_kernel, "density_rows"),
    (pareto_kernel, "density_rows"),
    (log_normal_kernel, "density_rows"),
    (two_piece_kernel, "density_rows"),
    (nb_kernel, "_compiled_rows"),
)


def _decline(*args, **kwargs):
    return None


def _routes(monkeypatch, call):
    """``call()`` through the compiled routes, which are checked to serve every row, then the array routes."""
    served = []
    with monkeypatch.context() as patch:
        for module, name in _ROW_ROUTES:
            route = getattr(module, name)

            def recorded(*args, _route=route, **kwargs):
                result = _route(*args, **kwargs)
                served.append(result is not None)
                return result

            patch.setattr(module, name, recorded)
        compiled = call()
    assert served and all(served), "a compiled route declined rows it should serve"
    with monkeypatch.context() as patch:
        for module, name in _ROW_ROUTES:
            patch.setattr(module, name, _decline)
        array = call()
    return compiled, array


def _channels(evaluated, order):
    channels = [evaluated.optimizing_log_likelihood[:, None]]
    if order >= 1:
        channels.append(evaluated.score)
    if order >= 2:
        channels.append(evaluated.hessian_packed)
    return channels


def _assert_agree(compiled, array, order):
    """Each entry within ``ROUNDINGS`` of the largest entry of its row at that order, and of one
    for the value, whose terms cancel to a smaller number."""
    assert np.array_equal(compiled.valid, array.valid)
    for level, (got, expected) in enumerate(
        zip(_channels(compiled, order), _channels(array, order), strict=True)
    ):
        scale = np.max(np.abs(expected), axis=1, keepdims=True)
        scale = np.maximum(scale, 1.0) if level == 0 else scale
        assert np.all(np.abs(got - expected) <= ROUNDINGS * scale), np.max(
            np.abs(got - expected) / np.maximum(scale, 1e-300)
        )


def _rows(seed, n=480):
    """A grid of positive responses and parameters with ties at the joins and the means."""
    rng = np.random.default_rng(seed)
    mean = np.exp(rng.uniform(-2.0, 3.0, n))
    ratio = np.exp(rng.uniform(-4.0, 4.0, n))
    ratio[::17] = 1.0
    scale = np.exp(rng.uniform(-3.0, 1.0, n))
    weight = rng.uniform(0.4, 3.0, n)
    return rng, mean, ratio * mean, scale, weight


def _gamma(semantics):
    _, mean, y, cv, weight = _rows(1)
    weight = np.ceil(weight) if semantics == "frequency" else weight
    return lambda order: gamma_kernel.evaluate_gamma_rows(
        y, mean, np.sqrt(cv), weight, semantics, derivative_order=order
    )


def _pareto():
    rng, scale, y, _, weight = _rows(3)
    shape = rng.uniform(-0.15, 0.9, len(y))
    shape[::11] = 0.0
    y = np.where(shape < 0.0, np.minimum(y, -0.95 * scale / np.minimum(shape, -1e-300)), y)
    return lambda order: pareto_kernel.scale_rows(
        y, scale, shape, np.ceil(weight), derivative_order=order
    )


def _log_normal():
    _, mean, y, scale, weight = _rows(4)
    return lambda order: log_normal_kernel.mean_rows(
        y, mean, scale, np.ceil(weight), derivative_order=order
    )


def _two_piece(form):
    rng, mean, y, scale, weight = _rows(5)
    skew = rng.uniform(-0.8, 0.8, len(y))
    skew[::13] = 0.0
    if form == "mean":
        return lambda order: two_piece_kernel.mean_rows(
            y, mean, scale, skew, np.ceil(weight), derivative_order=order
        )
    location = np.log(mean)
    return lambda order: two_piece_kernel.location_rows(
        np.where(np.arange(len(y)) % 7 == 0, location, np.log(y)),  # rows on the join
        location,
        scale,
        skew,
        np.ceil(weight),
        derivative_order=order,
    )


def _negative_binomial(semantics):
    rng, mean, _, _, weight = _rows(6)
    size = np.exp(rng.uniform(-1.5, 4.0, len(mean)))
    counts = rng.poisson(mean * rng.gamma(size, 1.0 / size)).astype(np.float64)
    weight = np.ceil(weight) if semantics == "frequency" else weight
    return lambda order: nb_kernel.evaluate_negative_binomial_rows(
        counts, mean, size, weight, semantics, derivative_order=order
    )


_CASES = {
    "gamma-prior": lambda: _gamma("prior"),
    "gamma-frequency": lambda: _gamma("frequency"),
    "generalized-pareto": _pareto,
    "log-normal-mean": _log_normal,
    "two-piece-mean": lambda: _two_piece("mean"),
    "two-piece-location": lambda: _two_piece("location"),
    "negative-binomial-prior": lambda: _negative_binomial("prior"),
    "negative-binomial-frequency": lambda: _negative_binomial("frequency"),
}


@pytest.mark.parametrize("order", [0, 1, 2])
@pytest.mark.parametrize("case", _CASES)
def test_compiled_rows_agree_with_the_array_rows(monkeypatch, case, order):
    compiled, array = _routes(monkeypatch, lambda: _CASES[case]()(order))
    assert compiled.valid.all() and array.valid.all()
    _assert_agree(compiled, array, order)


def test_the_comparison_rejects_a_kernel_that_is_wrong_in_one_channel(monkeypatch):
    """Mutation check: a Hessian entry off by ``1e-9`` of itself, which a looser comparison passes."""
    real = _declared.gamma.density

    def broken(*arguments):
        real(*arguments)
        arguments[-3][:, 1] *= 1.0 + 1e-9  # J2's mean-CV entry

    monkeypatch.setattr(_declared.gamma, "density", broken)
    compiled, array = _routes(monkeypatch, lambda: _CASES["gamma-prior"]()(2))
    with pytest.raises(AssertionError):
        _assert_agree(compiled, array, 2)


def test_a_chunk_with_a_row_the_kernel_cannot_represent_takes_the_array_route_whole(monkeypatch):
    """The compiled route declines it, and the result is the array route's, bit for bit."""
    rng, scale, y, _, weight = _rows(7)
    shape = rng.uniform(0.0, 0.5, len(y))
    y[5], shape[5] = 3.0 * scale[5], -0.5  # past the support's end: 1 + shape y/scale < 0
    declined = []

    def call():
        return pareto_kernel.scale_rows(y, scale, shape, np.ceil(weight), derivative_order=2)

    with monkeypatch.context() as patch:
        route = pareto_kernel.density_rows
        patch.setattr(
            pareto_kernel,
            "density_rows",
            lambda *a, **k: declined.append(route(*a, **k) is None) or None,
        )
        fallback = call()
    assert declined == [True]
    with monkeypatch.context() as patch:
        patch.setattr(pareto_kernel, "density_rows", _decline)
        array = call()
    assert not fallback.valid[5]
    for name in ("optimizing_log_likelihood", "score", "hessian_packed", "valid"):
        np.testing.assert_array_equal(getattr(fallback, name), getattr(array, name))


@pytest.mark.parametrize(
    ("mean", "theta", "refusal", "message"),
    [
        (2.0**460, 1.0, nb_kernel.NegativeBinomialNumericalDomainError, "outside the numerical"),
        (2.0**440, 2.0**-440, nb_kernel.NegativeBinomialDerivativeRepresentationError, "log-link"),
    ],
)
def test_a_negative_binomial_row_the_kernel_declines_raises_what_the_array_route_raises(
    mean, theta, refusal, message
):
    """The first is outside the parameter domain, the second cannot return its natural channels to
    the direct log-link ones within the roundoff bound; both are the kernel's to decline."""
    counts, means, thetas, weights = (np.array([v]) for v in (3.0, mean, theta, 1.0))
    declined = nb_kernel._compiled_rows(counts.astype(np.int64), (means, thetas, weights), True, 2)
    assert declined is None
    with pytest.raises(refusal, match=message):
        nb_kernel.evaluate_negative_binomial_rows(
            counts, means, thetas, weights, "prior", derivative_order=2
        )


def test_inputs_past_the_compiled_range_take_the_array_route():
    """Inputs past ``2^200`` can leave a unit-mass entry subnormal; the array rows rescale it."""
    scale, mass = np.array([2.0**250]), np.full(1, 2.0**50)
    zero = np.zeros(1)
    declined = _declared.density_rows(
        _declared.two_piece_location, 2, scale, (zero, scale, zero), (), mass
    )
    assert declined is None
    evaluated = two_piece_kernel.location_rows(scale, zero, scale, zero, mass, derivative_order=2)
    assert np.all(evaluated.hessian_packed[:, [0, 1, 3]] < 0.0)  # -2^-1050 and -2^-1049: subnormal


def test_the_poisson_limit_is_evaluated_by_the_compiled_rows(monkeypatch):
    """Size far past the counts: theta channels are exactly zero, as in the array route."""
    counts = np.array([0.0, 1.0, 4.0, 9.0])
    mean = np.array([0.5, 1.0, 3.0, 8.0])
    theta = np.full(4, 2.0**70)

    def call():
        return nb_kernel.evaluate_negative_binomial_rows(
            counts, mean, theta, np.ones(4), "prior", derivative_order=2
        )

    compiled, array = _routes(monkeypatch, call)
    _assert_agree(compiled, array, 2)
    assert np.all(compiled.score[:, 1] == 0.0) and np.all(compiled.hessian_packed[:, 1:] == 0.0)


# ---------------------------------------------------------------- the new declarations, in mpmath


def _exact_error(got, exact):
    """The gap to an mpmath value in ``u`` of the largest entry of its row."""
    return np.abs(np.asarray(got) - np.asarray(exact)) / max(np.max(np.abs(exact)), 1e-300) / U


def _held_against_mpmath(monkeypatch, evaluate, exact, rows, order=2):
    """The compiled error is at most ``8`` times the array code's, or ``64 u`` of the row scale."""
    compiled, array = _routes(monkeypatch, lambda: evaluate(order))
    for i in rows:
        for name, index in (("score", 1), ("hessian_packed", 2)):
            reference = exact(i)[index - 1]
            ours = _exact_error(getattr(compiled, name)[i], reference).max()
            theirs = _exact_error(getattr(array, name)[i], reference).max()
            assert ours <= max(8.0 * theirs, 64.0), (name, i, ours, theirs)


def test_log_normal_mean_form_rows_against_mpmath(monkeypatch):
    mp = pytest.importorskip("mpmath")
    from tests._log_normal_ls_oracles import mp_mean_derivatives

    _, mean, y, scale, weight = _rows(8, n=24)
    mass = np.ones(len(y))
    _held_against_mpmath(
        monkeypatch,
        lambda order: log_normal_kernel.mean_rows(y, mean, scale, mass, derivative_order=order),
        lambda i: mp_mean_derivatives(mp, y[i], mean[i], scale[i]),
        range(len(y)),
    )


@pytest.mark.parametrize("form", ["location", "mean"])
def test_two_piece_rows_against_mpmath(monkeypatch, form):
    pytest.importorskip("mpmath")
    from tests import _two_piece_lss_oracles as oracle

    rng, mean, y, scale, _ = _rows(9, n=16)
    skew = rng.uniform(-0.7, 0.7, len(y))
    t = np.where(np.arange(len(y)) % 4 == 0, np.log(mean), y)  # rows on the join and off it
    mass = np.ones(len(y))
    if form == "mean":

        def evaluate(order):
            return two_piece_kernel.mean_rows(y, mean, scale, skew, mass, derivative_order=order)

        def exact(i):
            point = (mean[i], scale[i], skew[i])
            return oracle.mp_packed(
                lambda m, s, e: oracle.mp_mean_log_density(y[i], m, s, e), point
            )
    else:

        def evaluate(order):
            return two_piece_kernel.location_rows(
                t, np.log(mean), scale, skew, mass, derivative_order=order
            )

        def exact(i):
            point = (np.log(mean[i]), scale[i], skew[i])
            return oracle.mp_packed(lambda m, s, e: oracle.mp_log_density(t[i], m, s, e), point)

    _held_against_mpmath(monkeypatch, evaluate, exact, range(len(y)))


# ---------------------------------------------------------------- what the kernels promise


def _declared_modules():
    return [
        importlib.import_module(f"{_declared.__name__}.{module.name}")
        for module in pkgutil.iter_modules(_declared.__path__)
    ]


def test_every_generated_kernel_releases_the_gil_and_is_registered_as_pooled():
    """Row chunks on the worker pool run these side by side; the registry is what
    ``test_only_the_pooled_block_kernels_release_the_gil_and_none_is_parallel`` checks."""
    for module in _declared_modules():
        kernels = [
            getattr(module, name)
            for name in (*module.LAWS, "carrier")
            if hasattr(getattr(module, name), "targetoptions")
        ]
        assert len(kernels) == len(module.LAWS) + 1
        assert set(map(id, kernels)) == set(map(id, module._POOLED_ROW_KERNELS))
        assert all(kernel.targetoptions.get("nogil") is True for kernel in kernels)


def test_a_piece_is_constant_between_the_joins_so_the_join_row_takes_the_right_piece():
    """``side`` is ``-1`` below zero and ``1`` at and above it, with no derivative."""
    from superglm.distributional.kernels._severity_primitives import side

    assert [side(z) for z in (-1e-300, -0.0, 0.0, 1e-300)] == [-1.0, 1.0, 1.0, 1.0]
    t = np.array([0.0])  # exactly on the join: the right piece, as the array code takes it
    left = two_piece_kernel.location_rows(
        t, t, np.ones(1), np.full(1, 0.5), np.ones(1), derivative_order=2
    )
    assert left.hessian_packed[0, 0] == pytest.approx(-1.0 / 1.5**2)


# ------------------------------------------------------------------ the range gate, in a fit


def _gate_fit(family, helpers, response, frame, gate):
    """A fit and its row-evaluation counts, with the compiled routes' range gate at ``gate``."""
    from superglm import SuperLSS

    served = [0, 0]  # calls the compiled route served, calls it declined
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_declared, "_MAX_INPUT", gate)
        for module, name in _ROW_ROUTES:
            route = getattr(module, name)

            def counted(*args, _route=route, **kwargs):
                result = _route(*args, **kwargs)
                served[result is None] += 1
                return result

            patch.setattr(module, name, counted)
        model = SuperLSS(family, *(getattr(family, name)("x") for name in helpers))
        model.fit_reml(frame, response)
    return model, served


def _gate_data(case, n=1500):
    rng = np.random.default_rng(20261106)
    x = rng.uniform(-1.0, 1.0, n)
    mean, spread, noise = np.exp(0.6 * np.sin(np.pi * x)), 0.35, rng.standard_normal(n)
    if case == "gamma":
        shape = 1.0 / spread**2
        return x, rng.gamma(shape, mean / shape)
    if case == "generalized_pareto":
        return x, mean * ((1.0 - rng.uniform(size=n)) ** -0.2 - 1.0) / 0.2
    if case == "log_normal":
        return x, np.exp(np.log(mean) + spread * noise)
    skewed = spread * np.where(noise > 0, 1.4, 0.8) * noise
    return x, (np.exp(np.log(mean) + skewed) if case == "two_piece_log_normal" else mean + skewed)


_GATE_CASES = {
    "gamma": (superglm.GammaLS, ("mean", "scale")),
    "generalized_pareto": (superglm.GeneralizedParetoLSS, ("scale", "shape")),
    "log_normal": (superglm.LogNormalLS, ("mean", "scale")),
    "two_piece_log_normal": (superglm.TwoPieceLogNormalLSS, ("mean", "scale", "skew")),
    "two_piece_normal": (superglm.TwoPieceNormalLSS, ("location", "scale", "skew")),
}
# A response in other units: the log-link families keep their fit, with predictions in those units.
_UNIT_FREE = ("gamma", "generalized_pareto", "log_normal", "two_piece_log_normal")


@pytest.mark.parametrize("case", sorted(_GATE_CASES))
@pytest.mark.parametrize("units", [1.0, _declared._MAX_INPUT / 2.0])
def test_a_fit_reads_the_same_down_either_route_of_the_range_gate(case, units):
    """The gate at ``2^200`` picks the route of a call, not the fit: on a response inside it and on
    one past it, a fit with the gate shipped and with it forced the other way agrees on its stop,
    iterations, EDF, predictions and standard errors.  Mutation checks: a gate that admits every
    input fails the route counts, and a compiled Hessian 1e-4 too large fails the comparison."""
    if units != 1.0 and case not in _UNIT_FREE:
        pytest.skip("an identity-link location is not unit-free: its penalty and bounds are not")
    make, helpers = _GATE_CASES[case]
    x, y = _gate_data(case)
    frame = pd.DataFrame({"x": x})
    # Inside the gate the shipped route is compiled and forcing the gate shut sends every call to the
    # array route; past it the shipped route declines, and opening the gate declines nothing.
    inside = np.abs(y * units).max() * 4.0 <= _declared._MAX_INPUT
    shipped, served = _gate_fit(make(), helpers, y * units, frame, _declared._MAX_INPUT)
    forced, other = _gate_fit(make(), helpers, y * units, frame, 0.0 if inside else np.inf)
    if inside:
        assert served[0] > 0 and served[1] == 0 and other[0] == 0 and other[1] > 0
    else:
        assert served[1] > 0 and other[0] > 0 and other[1] == 0
    for field in ("converged", "smoothing_converged", "n_smoothing_iter", "n_inner_iter"):
        assert getattr(shipped.result_, field) == getattr(forced.result_, field)
    assert shipped.result_.total_effective_df == pytest.approx(
        forced.result_.total_effective_df, rel=1e-6
    )
    a, b = shipped.predict_parameters(frame), forced.predict_parameters(frame)
    for name in a:
        scale = np.max(np.abs(a[name]))
        assert np.max(np.abs(np.asarray(a[name]) - np.asarray(b[name]))) <= 1e-6 * scale
    se_a, se_b = (np.sqrt(np.diag(m.covariance_)) for m in (shipped, forced))
    assert np.max(np.abs(se_a - se_b)) <= 1e-6 * np.max(se_a)
