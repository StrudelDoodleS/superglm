"""Generate the declared severity families' Numba row kernels with SymPy.

    uv run python tools/generate_severity_families.py           # rewrite the modules
    uv run python tools/generate_severity_families.py --check   # exit 1 if any is stale

A :class:`Declaration` gives a family's log density in its natural parameters,
and its log survival and log CDF where they are closed form, written with the
primitives below.  Each law is differentiated to order 4, every packed entry
once from its parent (no expansion, which would reopen cancellations the
primitives closed), and printed as one ``@njit`` row loop per law and order
writing the plan's packed jet ``(J0, ..., Jp)`` (``Jk`` columns in
``itertools.combinations_with_replacement`` order), common subexpressions
shared.  Additive terms of the log density free of every parameter become the
``carrier`` kernel, as the hand-written families split them.  The output,
``src/superglm/distributional/kernels/_declared/<name>.py``, imports only Numba,
NumPy, ``math`` and ``kernels._severity_primitives``; SymPy runs here only.

Kernel signatures: ``<law>_<p>(Y, P0..PK-1, D0..Dd-1[, M], J0..Jp)`` with
``law`` one of ``density``, ``logsf``, ``logcdf``; ``P`` the natural
parameters, ``D`` the declaration's per-row data (not differentiated; the
gamma's prior weight), ``M`` the density's row multiplier (frequency weights);
``carrier(Y, D.., M, J0)``.  The loops use ``error_model="numpy"``: a row whose
channels are not representable gets a non-finite entry, which marks it
invalid, and never raises.

Primitives (SymPy rule here, Numba body in ``_severity_primitives``):
``softplus``, ``expit``, ``log1mexp``, ``invexpm1``, ``gdev``, ``gnorm``,
``psires``, ``sres`` (``gnorm``'s derivatives), and SymPy's ``loggamma`` and
``polygamma``.  ``polygamma``, ``psires`` and ``sres`` of orders ``n <= 3``
at one argument share one call of the jet functions ``polygamma_0123``,
``psires_0123`` and ``sres_0123``.  An :class:`Engine` (plan
interface I2) is a two-argument primitive ``engine(u, v, upper, order, out)``:
each generated row calls it once per argument pair, and SymPy treats its
partials as functions whose derivatives are the next partials.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import importlib
import inspect
import itertools
import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import mpmath as mp
import sympy as sp
from sympy.core.function import ArgumentIndexError
from sympy.printing.pycode import PythonCodePrinter

KERNELS = Path(__file__).resolve().parents[1] / "src" / "superglm" / "distributional" / "kernels"
OUTPUT = KERNELS / "_declared"
PRIMITIVES = KERNELS / "_severity_primitives.py"
PRIMITIVES_MODULE = "superglm.distributional.kernels._severity_primitives"
MAX_ORDER = 4
LAWS = ("density", "logsf", "logcdf")

# ---------------------------------------------------------------- primitives


class softplus(sp.Function):  # noqa: N801
    nargs = 1

    def fdiff(self, argindex=1):
        return expit(self.args[0])


class expit(sp.Function):  # noqa: N801
    nargs = 1

    def fdiff(self, argindex=1):
        z = self.args[0]
        return expit(z) * expit(-z)


class log1mexp(sp.Function):  # noqa: N801
    nargs = 1

    def fdiff(self, argindex=1):
        return invexpm1(self.args[0])


class invexpm1(sp.Function):  # noqa: N801
    nargs = 1

    def fdiff(self, argindex=1):
        r = invexpm1(self.args[0])
        return -r * (1 + r)


class gdev(sp.Function):  # noqa: N801
    """``y/m - 1 - log(y/m)``; its partials never form ``1 + t`` from a rounded ``t``."""

    nargs = 2

    def fdiff(self, argindex=1):
        y, m = self.args
        return (y - m) / (y * m) if argindex == 1 else (m - y) / m**2


class gnorm(sp.Function):  # noqa: N801
    nargs = 1

    def fdiff(self, argindex=1):
        a = self.args[0]
        return -sres(0, a) / a


class sres(sp.Function):  # noqa: N801
    """``a^(n+1) psires(n, a)``: bounded where ``psires`` is singular, so derivatives keep it."""

    nargs = 2

    def fdiff(self, argindex=2):
        if argindex != 2:
            raise ArgumentIndexError(self, argindex)
        n, a = self.args
        return ((n + 1) * sres(n, a) + sres(n + 1, a)) / a


class psires(sp.Function):  # noqa: N801
    nargs = 2

    def fdiff(self, argindex=2):
        if argindex != 2:
            raise ArgumentIndexError(self, argindex)
        n, a = self.args
        return psires(n + 1, a)


def _mp_psires(n, a):
    n = int(n)
    if n == 0:
        return mp.digamma(a) - mp.log(a)
    return mp.polygamma(n, a) - (-1) ** (n - 1) * mp.factorial(n - 1) / a**n


MPMATH = {
    "softplus": lambda z: mp.log1p(mp.exp(z)),
    "expit": lambda z: 1 / (1 + mp.exp(-z)),
    "log1mexp": lambda t: mp.log(-mp.expm1(-t)) if t < 1 else mp.log1p(-mp.exp(-t)),
    "invexpm1": lambda t: 1 / mp.expm1(t),
    "gdev": lambda y, m: y / m - 1 - mp.log(y / m),
    "gnorm": lambda a: a * mp.log(a) - a - mp.loggamma(a),
    "psires": _mp_psires,
    "sres": lambda n, a: a ** (int(n) + 1) * _mp_psires(n, a),
    "loggamma": mp.loggamma,
    "polygamma": lambda n, x: mp.polygamma(int(n), x),
}
"""mpmath references of the primitives, by SymPy function name, for the oracles."""


@dataclass(frozen=True)
class Engine:
    """A two-argument special-function engine (plan interface I2) used as a primitive.

    ``name`` is an ``@njit`` function in ``module``,
    ``engine(u, v, upper, order, out)``, writing the value and then the partials
    in ``(u, v)`` of total order ``1..order`` (each block in
    ``combinations_with_replacement`` order) to ``out``, and NaN on rows outside
    its domain.  ``reference(u, v, upper)`` is its mpmath value, for oracles.
    An engine with ``work``, the name of a size constant in ``module``, takes a
    trailing scratch array of that size, allocated once per kernel call.
    """

    name: str
    module: str
    reference: Callable | None = None
    work: str | None = None

    def __call__(self, u, v, *, upper: bool):
        return engine_partial(self, upper, 0, 0)(u, v)


@functools.cache
def engine_partial(engine: Engine, upper: bool, p: int, q: int) -> type[sp.Function]:
    """SymPy function for ``d^(p+q) engine / du^p dv^q`` on one tail."""

    def fdiff(self, argindex=1):
        return engine_partial(engine, upper, p + (argindex == 1), q + (argindex == 2))(*self.args)

    side = "upper" if upper else "lower"
    attributes = {"nargs": 2, "fdiff": fdiff, "engine": engine, "upper": upper, "orders": (p, q)}
    return type(f"{engine.name}_{side}_{p}_{q}", (sp.Function,), attributes)


def engine_mpmath(function: type[sp.Function]) -> Callable:
    """mpmath value of an engine partial, by differentiating its reference."""
    reference, upper = function.engine.reference, function.upper
    return lambda u, v: mp.diff(lambda a, b: reference(a, b, upper), (u, v), function.orders)


# ---------------------------------------------------------------- declarations


@dataclass(frozen=True)
class Declaration:
    """A severity law in log space: callables of ``(y, *parameters, *data)`` giving SymPy expressions.

    With ``coordinates="log"`` the kernels differentiate in ``eta = log(theta)``
    and take ``eta`` as their parameters: a log-link family's predictor jets,
    with no chain rule to cancel through.
    """

    name: str
    parameters: tuple[str, ...]
    log_density: Callable[..., sp.Expr]
    log_survival: Callable[..., sp.Expr] | None = None
    log_cdf: Callable[..., sp.Expr] | None = None
    data: tuple[str, ...] = ()
    real: tuple[str, ...] = ()
    coordinates: str = "natural"
    doc: str = ""


def _gamma_log_density(y, mean, cv, weight):
    shape = weight / cv**2
    return gnorm(shape) - shape * gdev(y, mean) - sp.log(y)


def _mp_log_gamma_tail(a, x, upper):
    if upper:
        return mp.log(mp.gammainc(a, x, mp.inf, regularized=True))
    return mp.log(mp.gammainc(a, 0, x, regularized=True))


GAMMA_TAIL = Engine(
    "log_gamma_tail_into",
    "superglm.distributional.kernels._incomplete_gamma",
    reference=_mp_log_gamma_tail,
    work="WORK_SIZE",
)


def _gamma_tail(upper: bool):
    def tail(y, mean, cv, weight):
        shape = weight / cv**2
        return GAMMA_TAIL(shape, shape * y / mean, upper=upper)

    return tail


GAMMA = Declaration(
    name="gamma",
    parameters=("mean", "cv"),
    data=("weight",),
    log_density=_gamma_log_density,
    log_survival=_gamma_tail(True),
    log_cdf=_gamma_tail(False),
    doc=(
        "``GammaLS``'s row law in its own parameters: mean ``mean``, shape\n"
        "``weight/cv^2`` (prior weights reweight the law; pass ones and the\n"
        "weights as ``M`` for frequency weights).  The log tails are\n"
        "``log Q`` and ``log P`` of the incomplete-gamma engine at shape\n"
        "``weight/cv^2`` and point ``weight y/(mean cv^2)``."
    ),
)

GAMMA_LOG = replace(
    GAMMA,
    name="gamma_log",
    coordinates="log",
    doc=GAMMA.doc + "\nKernels in ``(log mean, log cv)``: ``GammaLS``'s predictor jets.",
)

DECLARATIONS = (GAMMA, GAMMA_LOG)


@dataclass
class Derived:
    declaration: Declaration
    y: sp.Symbol
    theta: tuple[sp.Symbol, ...]
    data: tuple[sp.Symbol, ...]
    carrier: sp.Expr
    laws: dict[str, list[list[sp.Expr]]]
    seconds: float


def packed(k: int, order: int) -> list[tuple[int, ...]]:
    return list(itertools.combinations_with_replacement(range(k), order))


def jet(expression: sp.Expr, theta, max_order: int = MAX_ORDER) -> list[list[sp.Expr]]:
    """Packed derivatives of orders ``0..max_order``, each entry differentiated once from its parent."""
    cache = {(): expression}
    out = [[expression]]
    for order in range(1, max_order + 1):
        for key in packed(len(theta), order):
            cache[key] = sp.diff(cache[key[:-1]], theta[key[-1]])
        out.append([cache[key] for key in packed(len(theta), order)])
    return out


def derive(declaration: Declaration) -> Derived:
    start = time.perf_counter()
    y = sp.Symbol("y", positive=True)
    theta = tuple(
        sp.Symbol(name, real=True) if name in declaration.real else sp.Symbol(name, positive=True)
        for name in declaration.parameters
    )
    data = tuple(sp.Symbol(name, positive=True) for name in declaration.data)
    if declaration.coordinates == "log":
        theta = tuple(sp.Symbol(f"log_{name}", real=True) for name in declaration.parameters)
    law = tuple(sp.exp(t) for t in theta) if declaration.coordinates == "log" else theta
    density = declaration.log_density(y, *law, *data)
    terms = sp.Add.make_args(density)
    free = [term for term in terms if not term.free_symbols & set(theta)]
    laws = {"density": jet(sp.Add(*[term for term in terms if term not in free]), theta)}
    for kind, function in (("logsf", declaration.log_survival), ("logcdf", declaration.log_cdf)):
        if function is not None:
            laws[kind] = jet(function(y, *law, *data), theta)
    return Derived(declaration, y, theta, data, sp.Add(*free), laws, time.perf_counter() - start)


# ---------------------------------------------------------------- printing


class _Printer(PythonCodePrinter):
    def _print_Rational(self, expr):
        return f"({expr.p}.0 / {expr.q}.0)"

    def _print_Integer(self, expr):
        return f"{int(expr)}.0"

    def _print_Pow(self, expr, rational=False):
        base, exponent = expr.as_base_exp()
        if exponent.is_Integer and exponent != 0:
            factors = " * ".join([self.parenthesize(base, 100)] * abs(int(exponent)))
            return f"({factors})" if exponent > 0 else f"(1.0 / ({factors}))"
        if exponent == sp.S.Half:
            return f"sqrt({self._print(base)})"
        return f"({self.parenthesize(base, 100)} ** {self.parenthesize(exponent, 100)})"

    def _print_Function(self, expr):
        name = type(expr).__name__
        if name not in _PRIMITIVE_NAMES:
            raise ValueError(f"no Numba rule for {name}")
        return f"{name}({', '.join(self._print(argument) for argument in expr.args)})"


_PRIMITIVE_NAMES = ("softplus", "expit", "log1mexp", "invexpm1", "gdev", "gnorm")
_JETS = {sp.polygamma: "polygamma_0123", psires: "psires_0123", sres: "sres_0123"}
_PRINTER = _Printer({"fully_qualified_modules": False})


def _engine_index(p: int, q: int) -> int:
    order = p + q
    return order * (order + 1) // 2 + q


@dataclass
class _Statement:
    defines: tuple[sp.Symbol, ...]
    uses: frozenset
    lines: list[str]


def _calls(expressions: list[sp.Expr]):
    """Group jet-function and engine atoms by argument: ``{(callee, upper, args): {atom: index}}``."""
    groups: dict[tuple, dict[sp.Expr, int]] = {}
    for expression in expressions:
        for atom in expression.atoms(sp.Function):
            if type(atom) in _JETS:
                n, argument = atom.args
                if not (n.is_Integer and 0 <= n <= 3):
                    raise ValueError(f"{atom}: the jet functions stop at n = 3")
                key = (_JETS[type(atom)], None, (argument,))
                groups.setdefault(key, {})[atom] = int(n)
            elif hasattr(type(atom), "engine"):
                key = (type(atom).engine, type(atom).upper, atom.args)
                groups.setdefault(key, {})[atom] = _engine_index(*type(atom).orders)
    for key in groups:
        if any(argument.atoms(sp.Function) & set().union(*groups.values()) for argument in key[2]):
            raise ValueError("a jet or engine argument may not itself call a jet or engine")
    return groups


def emit_kernel(
    name: str,
    derived: Derived,
    targets: list[tuple[str, sp.Expr]],
    *,
    multiplied: bool,
    parametrised: bool = True,
):
    """One ``@njit`` row loop writing every ``(target, expression)``, CSE shared across them."""
    k, d = len(derived.theta) if parametrised else 0, len(derived.data)
    local = {derived.y: sp.Symbol("y")}
    local.update({s: sp.Symbol(f"p{j}") for j, s in enumerate(derived.theta)})
    local.update({s: sp.Symbol(f"d{j}") for j, s in enumerate(derived.data)})
    expressions = [expression.xreplace(local) for _, expression in targets]
    groups = _calls(expressions)
    outputs, arguments, preamble, engines = {}, [], [], set()
    for g, ((callee, upper, args), atoms) in enumerate(groups.items()):
        outputs.update({atom: sp.Symbol(f"c{g}_{index}") for atom, index in atoms.items()})
        arguments.extend(args)
        if isinstance(callee, Engine):
            engines.add(callee)
            order = max(sum(type(atom).orders) for atom in atoms)
            preamble.append(f"    out{g} = np.empty({(order + 1) * (order + 2) // 2})")
            if callee.work is not None and "    work = " not in "".join(preamble):
                size = getattr(importlib.import_module(callee.module), callee.work)
                preamble.append(f"    work = np.empty({int(size)})")
    expressions = [expression.xreplace(outputs) for expression in expressions]
    replacements, reduced = sp.cse(
        arguments + expressions, symbols=sp.numbered_symbols("t"), optimizations="basic"
    )
    reduced_arguments, reduced = reduced[: len(arguments)], reduced[len(arguments) :]
    statements = [
        _Statement(
            (symbol,), frozenset(value.free_symbols), [f"{symbol} = {_PRINTER.doprint(value)}"]
        )
        for symbol, value in replacements
    ]
    position = 0
    for g, ((callee, upper, args), atoms) in enumerate(groups.items()):
        printed = [_PRINTER.doprint(a) for a in reduced_arguments[position : position + len(args)]]
        uses = frozenset().union(
            *(a.free_symbols for a in reduced_arguments[position : position + len(args)])
        )
        position += len(args)
        if isinstance(callee, Engine):
            order = max(sum(type(atom).orders) for atom in atoms)
            scratch = ", work" if callee.work is not None else ""
            lines = [f"{callee.name}({', '.join(printed)}, {upper}, {order}, out{g}{scratch})"]
            lines += [f"c{g}_{index} = out{g}[{index}]" for index in sorted(set(atoms.values()))]
            defines = tuple(sp.Symbol(f"c{g}_{index}") for index in sorted(set(atoms.values())))
        else:
            defines = tuple(sp.Symbol(f"c{g}_{n}") for n in range(4))
            lines = [f"{', '.join(map(str, defines))} = {callee}({printed[0]})"]
        statements.append(_Statement(defines, uses, lines))
    body = _ordered(statements)
    parameters = [f"P{j}" for j in range(k)] + [f"D{j}" for j in range(d)]
    signature = ["Y", *parameters, *(["M"] if multiplied else []), *_outputs(targets)]
    lines = [
        "@njit(cache=True, nogil=True, error_model='numpy')",
        f"def {name}({', '.join(signature)}):",
        *preamble,
        "    for i in range(Y.shape[0]):",
        "        y = Y[i]",
        *[f"        p{j} = P{j}[i]" for j in range(k)],
        *[f"        d{j} = D{j}[i]" for j in range(d)],
        *(["        m = M[i]"] if multiplied else []),
        *[f"        {line}" for line in body],
    ]
    for (target, _), expression in zip(targets, reduced, strict=True):
        printed = _PRINTER.doprint(expression)
        lines.append(
            f"        {target} = m * ({printed})" if multiplied else f"        {target} = {printed}"
        )
    return "\n".join(lines) + "\n", engines


def _outputs(targets) -> list[str]:
    return list(dict.fromkeys(target.split("[", 1)[0] for target, _ in targets))


def _ordered(statements: list[_Statement]) -> list[str]:
    """Statement lines with every definition before its first use, else in CSE order."""
    defined_by = {symbol: index for index, s in enumerate(statements) for symbol in s.defines}
    done: set[int] = set()
    lines: list[str] = []

    def visit(index: int) -> None:
        if index in done:
            return
        done.add(index)
        for symbol in statements[index].uses:
            if symbol in defined_by:
                visit(defined_by[symbol])
        lines.extend(statements[index].lines)

    for index in range(len(statements)):
        visit(index)
    return lines


def _jet_targets(channels: list[list[sp.Expr]], order: int) -> list[tuple[str, sp.Expr]]:
    targets = [("J0[i]", channels[0][0])]
    for level in range(1, order + 1):
        targets += [(f"J{level}[i, {j}]", e) for j, e in enumerate(channels[level])]
    return targets


def _callee_digest(dispatchers) -> str:
    """Digest of compiled callees' sources and the module constants they read."""
    parts = []
    for dispatcher in dispatchers:
        function = dispatcher.py_func
        parts.append(inspect.getsource(function))
        for name in function.__code__.co_names:
            value = function.__globals__.get(name)
            if hasattr(value, "py_func"):
                parts.append(_callee_digest([value]))
            elif isinstance(value, int | float | tuple) or hasattr(value, "tolist"):
                parts.append(f"{name}={getattr(value, 'tolist', lambda: value)()!r}")
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


_CALLEE_LINE = re.compile(r'^_CALLEE_DIGEST = "[^"]*"$', re.MULTILINE)
# Kernel modules outside ``_declared`` that call compiled primitives: each
# carries the digest of the primitives it imports, so editing one recompiles it.
PRIMITIVE_CALLERS = ("_incomplete_gamma", "_tweedie_numba")


def stamped_callers() -> dict[Path, str]:
    """Each primitive caller's text with its ``_CALLEE_DIGEST`` brought up to date."""
    files = {}
    for name in PRIMITIVE_CALLERS:
        module = importlib.import_module(f"superglm.distributional.kernels.{name}")
        callees = [
            value
            for _, value in sorted(vars(module).items())
            if hasattr(value, "py_func") and value.py_func.__module__ == PRIMITIVES_MODULE
        ]
        path = KERNELS / f"{name}.py"
        text = path.read_text()
        if not _CALLEE_LINE.search(text):
            raise RuntimeError(f"{path} has no _CALLEE_DIGEST line")
        files[path] = _CALLEE_LINE.sub(f'_CALLEE_DIGEST = "{_callee_digest(callees)}"', text)
    return files


def render(derived: Derived, primitives_text: str) -> str:
    """The module text: every law at orders ``0..MAX_ORDER`` and the carrier."""
    chunks, engines = [], set()
    for law, channels in derived.laws.items():
        for order in range(MAX_ORDER + 1):
            text, used = emit_kernel(
                f"{law}_{order}",
                derived,
                _jet_targets(channels, order),
                multiplied=law == "density",
            )
            chunks.append(text)
            engines |= used
    carrier, _ = emit_kernel(
        "carrier", derived, [("J0[i]", derived.carrier)], multiplied=True, parametrised=False
    )
    chunks.append(carrier)
    body = "\n\n".join(chunks)
    primitives = sorted(
        name for name in (*_PRIMITIVE_NAMES, *_JETS.values()) if re.search(rf"\b{name}\(", body)
    )
    declaration = derived.declaration
    engine_digest = (
        _callee_digest([getattr(importlib.import_module(e.module), e.name) for e in engines])
        if engines
        else ""
    )
    header = [
        f'"""Generated by ``tools/generate_severity_families.py`` from the ``{declaration.name}``'
        " declaration; do not edit by hand.",
        "",
        *declaration.doc.splitlines(),
        '"""',
        "",
        "# ruff: noqa",
        "# fmt: off",
        "",
        "from math import exp, expm1, lgamma, log, log1p, sqrt",
        "",
        "import numpy as np",
        "from numba import njit",
        "",
    ]
    if primitives:
        header.append(f"from {PRIMITIVES_MODULE} import {', '.join(primitives)}")
    for engine in sorted(engines, key=lambda e: (e.module, e.name)):
        header.append(f"from {engine.module} import {engine.name}")
    header += [
        "",
        f"PARAMETERS = {declaration.parameters!r}",
        f"DATA = {declaration.data!r}",
        f"COORDINATES = {declaration.coordinates!r}",
        f"LAWS = {tuple(derived.laws)!r}",
        f"MAX_ORDER = {MAX_ORDER}",
        f'PRIMITIVES_DIGEST = "{hashlib.sha256(primitives_text.encode()).hexdigest()}"',
        f'ENGINES_DIGEST = "{engine_digest}"',
        "",
        "",
    ]
    return "\n".join(header) + body


def outputs() -> dict[Path, str]:
    """Every file the generator owns, with its current text."""
    primitives = PRIMITIVES.read_text()
    files = stamped_callers()
    for declaration in DECLARATIONS:
        files[OUTPUT / f"{declaration.name}.py"] = render(derive(declaration), primitives)
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="exit 1 if a module is stale")
    arguments = parser.parse_args()
    files = outputs()
    stale = [path for path, text in files.items() if not path.exists() or path.read_text() != text]
    if arguments.check:
        for path in stale:
            print(f"{path} is stale; rerun the generator", file=sys.stderr)
        return 1 if stale else 0
    for path in stale:
        path.write_text(files[path])
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
