"""Every numba dispatcher the superglm package defines, found two ways.

``package_dispatchers`` imports each superglm module and keeps the numba
dispatchers bound in it.  ``entry_point_sites`` reads each source file for the
numba entry points it uses: a decorator on a module-level function, or a
module-level ``name = njit(...)(function)``.  The two must name the same
dispatchers, so a kernel made where the import walk cannot see it (inside a
function, or through an entry point the census does not classify) fails the
census instead of escaping it.  Names are ``module.attribute`` at the
definition site.
"""

from __future__ import annotations

import ast
import importlib
import pkgutil
from pathlib import Path

# numba's public ways to make compiled code.  Only ``njit`` and ``jit`` make the
# dispatchers this census classifies; a superglm module using any other fails it
# until the census learns to classify that kind.
ENTRY_POINTS = frozenset(
    {
        "njit",
        "jit",
        "vectorize",
        "guvectorize",
        "cfunc",
        "stencil",
        "generated_jit",
        "overload",
        "overload_method",
        "overload_attribute",
        "intrinsic",
        "register_jitable",
    }
)
DISPATCHER_ENTRY_POINTS = frozenset({"njit", "jit"})


def _numba_bindings(tree: ast.Module) -> tuple[dict[str, str], set[str]]:
    """Local names bound to numba entry points, and local names of numba modules."""
    entry_points: dict[str, str] = {}
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "numba":
            for alias in node.names:
                if alias.name in ENTRY_POINTS:
                    entry_points[alias.asname or alias.name] = alias.name
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "numba":
                    modules.add(alias.asname or "numba")
    return entry_points, modules


def _reference(node: ast.AST, entry_points: dict[str, str], modules: set[str]) -> str | None:
    """The entry point a name or attribute refers to, if any."""
    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
        return entry_points.get(node.id)
    if isinstance(node, ast.Attribute) and node.attr in ENTRY_POINTS:
        base = node.value
        while isinstance(base, ast.Attribute):
            base = base.value
        if isinstance(base, ast.Name) and base.id in modules:
            return node.attr
    return None


def _applied(node: ast.expr) -> ast.expr:
    """The entry point reference of ``@ep``, ``@ep(...)`` or ``ep(...)(f)``."""
    return node.func if isinstance(node, ast.Call) else node


def _module_name(path: Path, package_dir: Path, package: str) -> str:
    parts = path.relative_to(package_dir).with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join((package, *parts))


def entry_point_sites(package_dir: Path, package: str = "superglm"):
    """``(sites, problems)`` read from the package's source.

    ``sites`` maps ``(module, attribute)`` to the entry point that binds a
    dispatcher there.  ``problems`` names every other use of an entry point.
    """
    sites: dict[tuple[str, str], str] = {}
    problems: list[str] = []
    for path in sorted(package_dir.rglob("*.py")):
        module = _module_name(path, package_dir, package)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        entry_points, modules = _numba_bindings(tree)
        if not entry_points and not modules:
            continue
        claimed: set[int] = set()

        def claim(name: str, node: ast.expr, module: str = module) -> bool:
            reference = _applied(node)
            kind = _reference(reference, entry_points, modules)
            if kind is None:
                return False
            sites[(module, name)] = kind
            claimed.add(id(reference))
            return True

        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                for decorator in node.decorator_list:
                    claim(node.name, decorator)
            # A module-level ``if`` (``TYPE_CHECKING``) may bind the kernel in a branch.
            for statement in [*node.body, *node.orelse] if isinstance(node, ast.If) else [node]:
                if (
                    isinstance(statement, ast.Assign)
                    and len(statement.targets) == 1
                    and isinstance(statement.targets[0], ast.Name)
                    and isinstance(statement.value, ast.Call)
                    and statement.value.args
                ):
                    claim(statement.targets[0].id, statement.value.func)
        for node in ast.walk(tree):
            kind = _reference(node, entry_points, modules)
            if kind is not None and id(node) not in claimed:
                line = getattr(node, "lineno", "?")
                problems.append(f"{module}:{line}: {kind} outside a module-level kernel definition")
    for (module, name), kind in sites.items():
        if kind not in DISPATCHER_ENTRY_POINTS:
            problems.append(f"{module}.{name}: {kind} makes no dispatcher the census classifies")
    return sites, problems


def package_dispatchers(package: str = "superglm"):
    """``(dispatchers, problems)``: every dispatcher by name, and census failures.

    Imports every module of the package.  A module that fails to import for an
    optional dependency is skipped unless its source defines a kernel.
    """
    from numba.core.dispatcher import Dispatcher

    root = importlib.import_module(package)
    sites, problems = entry_point_sites(Path(root.__file__).parent, package)
    modules = {package: root}
    for info in pkgutil.walk_packages(root.__path__, f"{package}."):
        try:
            modules[info.name] = importlib.import_module(info.name)
        except ImportError:
            continue
    dispatchers = {}
    for module_name, attribute in sorted(sites):
        value = getattr(modules.get(module_name), attribute, None)
        if isinstance(value, Dispatcher):
            dispatchers[f"{module_name}.{attribute}"] = value
        else:
            problems.append(f"{module_name}.{attribute}: no dispatcher bound after import")
    named = {id(value) for value in dispatchers.values()}
    for module in modules.values():
        for value in vars(module).values():
            if (
                isinstance(value, Dispatcher)
                and (value.py_func.__module__ or "").split(".")[0] == package
                and id(value) not in named
            ):
                named.add(id(value))
                problems.append(
                    f"{value.py_func.__module__}.{value.py_func.__qualname__}: "
                    "a dispatcher no definition in the source accounts for"
                )
    return dispatchers, problems


def is_inlined(dispatcher) -> bool:
    """Inlined into each caller's IR, so it never compiles a signature of its own."""
    return dispatcher.targetoptions.get("inline") == "always"


def signatures(dispatchers) -> dict[str, list[str]]:
    return {name: [str(sig) for sig in value.signatures] for name, value in dispatchers.items()}
