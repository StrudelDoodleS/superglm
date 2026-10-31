"""Generate numba row kernels from a severity family's declared log density and log tails.

Prototype for notes/research/2026-10-31-severity-family-framework.md; not production code.

A :class:`Declaration` gives the closed-form log density in the family's natural
parameters, and the log survival and log CDF where they are closed form, all
written with :mod:`sevprims` primitives.  :func:`build` differentiates them with
SymPy (packed orders 1-4 for the density, 1-2 for the tails), shares common
subexpressions, prints one ``@njit`` row loop per output set and imports the
module.  Additive terms of the log density free of every parameter become the
plan's parameter-independent carrier, as the hand-written families split them.
"""

from __future__ import annotations

import importlib.util
import itertools
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import sympy as sp
from sympy.codegen.cfunctions import expm1, log1p
from sympy.printing.pycode import PythonCodePrinter

import sevprims

HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Declaration:
    name: str
    parameters: tuple[str, ...]
    log_density: Callable[..., sp.Expr]
    log_survival: Callable[..., sp.Expr] | None = None
    log_cdf: Callable[..., sp.Expr] | None = None
    scipy_law: Callable[..., object] | None = None
    real_parameters: tuple[str, ...] = ()
    notes: str = ""


@dataclass
class Derived:
    declaration: Declaration
    y: sp.Symbol
    theta: tuple[sp.Symbol, ...]
    optimizing: sp.Expr
    carrier: sp.Expr
    density: dict[int, list[sp.Expr]]
    tails: dict[str, dict[int, list[sp.Expr]]] = field(default_factory=dict)
    seconds: float = 0.0


def packed(k: int, order: int) -> list[tuple[int, ...]]:
    return list(itertools.combinations_with_replacement(range(k), order))


def symbols(declaration: Declaration) -> tuple[sp.Symbol, tuple[sp.Symbol, ...]]:
    y = sp.Symbol("y", positive=True)
    theta = tuple(
        sp.Symbol(name, real=True)
        if name in declaration.real_parameters
        else sp.Symbol(name, positive=True)
        for name in declaration.parameters
    )
    return y, theta


def derivatives(expression: sp.Expr, theta, max_order: int) -> dict[int, list[sp.Expr]]:
    """Packed derivatives of each order, every entry differentiated once from its parent."""
    cache = {(): expression}
    out = {0: [expression]}
    for order in range(1, max_order + 1):
        out[order] = []
        for key in packed(len(theta), order):
            if key not in cache:
                cache[key] = sp.diff(cache[key[:-1]], theta[key[-1]])
            out[order].append(cache[key])
    return out


def derive(declaration: Declaration, *, density_order: int = 4) -> Derived:
    start = time.perf_counter()
    y, theta = symbols(declaration)
    expression = declaration.log_density(y, *theta)
    terms = sp.Add.make_args(expression)
    depends = [term for term in terms if term.free_symbols & set(theta)]
    carrier = sp.Add(*[term for term in terms if not term.free_symbols & set(theta)])
    optimizing = sp.Add(*depends)
    derived = Derived(
        declaration,
        y,
        theta,
        optimizing,
        carrier,
        derivatives(optimizing, theta, density_order),
    )
    for kind, law in (("logsf", declaration.log_survival), ("logcdf", declaration.log_cdf)):
        if law is not None:
            derived.tails[kind] = derivatives(law(y, *theta), theta, 2)
    derived.seconds = time.perf_counter() - start
    return derived


# ---------------------------------------------------------------- printing

_NUMBA_NAMES = {
    "softplus": "nb_softplus",
    "expit": "nb_expit",
    "log1mexp": "nb_log1mexp",
    "invexpm1": "nb_invexpm1",
    "dev": "nb_dev",
    "gdev": "nb_gdev",
    "gnorm": "nb_gnorm",
}


class _Printer(PythonCodePrinter):
    def _print_Rational(self, expr):
        return f"({expr.p}.0 / {expr.q}.0)"

    def _print_Integer(self, expr):
        return f"{int(expr)}.0"

    def _print_Pow(self, expr, rational=False):
        base, exponent = expr.as_base_exp()
        if exponent.is_Integer and 0 < exponent <= 4:
            return "(" + " * ".join([self.parenthesize(base, 100)] * int(exponent)) + ")"
        if exponent.is_Integer and -4 <= exponent < 0:
            return "(1.0 / (" + " * ".join([self.parenthesize(base, 100)] * int(-exponent)) + "))"
        if exponent == sp.S.Half:
            return f"sqrt({self._print(base)})"
        return f"({self.parenthesize(base, 100)} ** {self.parenthesize(exponent, 100)})"

    def _print_log(self, expr):
        return f"log({self._print(expr.args[0])})"

    def _print_exp(self, expr):
        return f"exp({self._print(expr.args[0])})"

    def _print_log1p(self, expr):
        return f"log1p({self._print(expr.args[0])})"

    def _print_expm1(self, expr):
        return f"expm1({self._print(expr.args[0])})"

    def _print_loggamma(self, expr):
        return f"lgamma({self._print(expr.args[0])})"

    def _print_polygamma(self, expr):
        n, x = expr.args
        if n == 0:
            return f"nb_digamma({self._print(x)})"
        return f"nb_polygamma({int(n)}, {self._print(x)})"

    def _print_psires(self, expr):
        n, x = expr.args
        return f"nb_psires({int(n)}, {self._print(x)})"

    def _print_Function(self, expr):
        name = type(expr).__name__
        if name in _NUMBA_NAMES:
            arguments = ", ".join(self._print(argument) for argument in expr.args)
            return f"{_NUMBA_NAMES[name]}({arguments})"
        raise ValueError(f"no numba rule for {name}")


_PRINTER = _Printer({"fully_qualified_modules": False})

_HEADER = '''"""Generated by sevgen.py from the {name} declaration; do not edit."""

# ruff: noqa
from math import exp, expm1, lgamma, log, log1p, sqrt

from numba import njit

from sevprims import (
    nb_digamma, nb_dev, nb_expit, nb_gdev, nb_gnorm, nb_invexpm1, nb_log1mexp,
    nb_polygamma, nb_psires, nb_softplus,
)
'''


def _local(derived: Derived, response: str) -> dict:
    mapping = {derived.y: sp.Symbol(response)}
    mapping.update({symbol: sp.Symbol(f"th{index}") for index, symbol in enumerate(derived.theta)})
    return mapping


def emit_rows(
    name: str,
    derived: Derived,
    targets: list[tuple[str, sp.Expr]],
    *,
    weighted: bool,
    outputs: tuple[str, ...],
) -> str:
    """One ``@njit`` row loop writing each ``(target, expression)``, CSE shared across them."""
    mapping = _local(derived, "y_")
    expressions = [expression.xreplace(mapping) for _, expression in targets]
    replacements, reduced = sp.cse(
        expressions, symbols=sp.numbered_symbols("t"), optimizations="basic"
    )
    k = len(derived.theta)
    parameters = ", ".join(f"P{index}" for index in range(k))
    lines = [
        "@njit(cache=False, fastmath=False, error_model='numpy')",
        f"def {name}(Y, {parameters}, W, {', '.join(outputs)}):",
        "    for i in range(Y.shape[0]):",
        "        y_ = Y[i]",
    ]
    lines += [f"        th{index} = P{index}[i]" for index in range(k)]
    lines.append("        w_ = W[i]" if weighted else "        w_ = 1.0")
    lines += [f"        {symbol} = {_PRINTER.doprint(value)}" for symbol, value in replacements]
    for (target, _), expression in zip(targets, reduced, strict=True):
        lines.append(f"        {target} = w_ * ({_PRINTER.doprint(expression)})")
    return "\n".join(lines) + "\n"


def _density_targets(derived: Derived, order: int) -> list[tuple[str, sp.Expr]]:
    targets = [("value[i]", derived.density[0][0])]
    if order >= 1:
        targets += [(f"score[i, {j}]", e) for j, e in enumerate(derived.density[1])]
    if order >= 2:
        targets += [(f"hess[i, {j}]", e) for j, e in enumerate(derived.density[2])]
    return targets


def render(derived: Derived) -> str:
    chunks = [_HEADER.format(name=derived.declaration.name)]
    outputs = {0: ("value",), 1: ("value", "score"), 2: ("value", "score", "hess")}
    for order in (0, 1, 2):
        chunks.append(
            emit_rows(
                f"density_o{order}",
                derived,
                _density_targets(derived, order),
                weighted=True,
                outputs=outputs[order],
            )
        )
    higher = [(f"d3[i, {j}]", e) for j, e in enumerate(derived.density.get(3, []))]
    higher += [(f"d4[i, {j}]", e) for j, e in enumerate(derived.density.get(4, []))]
    if higher:
        chunks.append(emit_rows("density_d34", derived, higher, weighted=True, outputs=("d3", "d4")))
    chunks.append(
        emit_rows(
            "carrier_rows",
            derived,
            [("value[i]", derived.carrier)],
            weighted=True,
            outputs=("value",),
        )
    )
    for kind, channels in derived.tails.items():
        for order in (0, 2):
            targets = [("value[i]", channels[0][0])]
            if order == 2:
                targets += [(f"score[i, {j}]", e) for j, e in enumerate(channels[1])]
                targets += [(f"hess[i, {j}]", e) for j, e in enumerate(channels[2])]
            chunks.append(
                emit_rows(
                    f"{kind}_o{order}", derived, targets, weighted=False, outputs=outputs[order]
                )
            )
    return "\n\n".join(chunks)


def build(declaration: Declaration, directory: Path, *, density_order: int = 4):
    """Derive, render, write and import the kernels; returns ``(GeneratedKernel, report)``."""
    derived = derive(declaration, density_order=density_order)
    start = time.perf_counter()
    source = render(derived)
    render_seconds = time.perf_counter() - start
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"gen_{declaration.name}.py"
    path.write_text(source)
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location(f"gen_{declaration.name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = {
        "family": declaration.name,
        "derive_seconds": derived.seconds,
        "render_seconds": render_seconds,
        "generated_lines": source.count("\n"),
        "path": str(path),
    }
    return GeneratedKernel(module, derived), report


class GeneratedKernel:
    """Array-level calls into one generated module (contiguous float64 inputs)."""

    def __init__(self, module, derived: Derived):
        self.module = module
        self.derived = derived
        self.k = len(derived.theta)

    def _columns(self, theta):
        theta = np.ascontiguousarray(theta, dtype=np.float64)
        return tuple(np.ascontiguousarray(theta[:, j]) for j in range(self.k))

    def evaluate(self, y, theta, weights, order: int):
        y = np.ascontiguousarray(y, dtype=np.float64)
        columns = self._columns(theta)
        weights = np.ascontiguousarray(weights, dtype=np.float64)
        n = len(y)
        value = np.empty(n)
        if order == 0:
            self.module.density_o0(y, *columns, weights, value)
            return value, None, None
        score = np.empty((n, self.k))
        if order == 1:
            self.module.density_o1(y, *columns, weights, value, score)
            return value, score, None
        hess = np.empty((n, self.k * (self.k + 1) // 2))
        self.module.density_o2(y, *columns, weights, value, score, hess)
        return value, score, hess

    def carrier(self, y, weights):
        y = np.ascontiguousarray(y, dtype=np.float64)
        value = np.empty(len(y))
        dummy = tuple(np.ones(len(y)) for _ in range(self.k))
        self.module.carrier_rows(y, *dummy, np.ascontiguousarray(weights, dtype=np.float64), value)
        return value

    def third_fourth(self, y, theta, weights):
        y = np.ascontiguousarray(y, dtype=np.float64)
        n = len(y)
        d3 = np.empty((n, len(packed(self.k, 3))))
        d4 = np.empty((n, len(packed(self.k, 4))))
        self.module.density_d34(
            y, *self._columns(theta), np.ascontiguousarray(weights, dtype=np.float64), d3, d4
        )
        return d3, d4

    def tail(self, kind: str, c, theta, order: int = 2):
        c = np.ascontiguousarray(c, dtype=np.float64)
        n = len(c)
        value = np.empty(n)
        ones = np.ones(n)
        if order == 0:
            getattr(self.module, f"{kind}_o0")(c, *self._columns(theta), ones, value)
            return value, None, None
        score = np.empty((n, self.k))
        hess = np.empty((n, self.k * (self.k + 1) // 2))
        getattr(self.module, f"{kind}_o2")(c, *self._columns(theta), ones, value, score, hess)
        return value, score, hess


def mpmath_function(derived: Derived, expression: sp.Expr):
    """The expression as an mpmath callable of ``(y, *theta)``, for the high-precision oracle."""
    namespace = dict(sevprims.MPMATH)
    return sp.lambdify(
        (derived.y, *derived.theta),
        expression,
        modules=[namespace, "mpmath"],
    )


__all__ = ["Declaration", "GeneratedKernel", "build", "derive", "expm1", "log1p", "packed"]
