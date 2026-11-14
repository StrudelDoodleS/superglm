"""Print the hash that keys CI's numba kernel cache: the files that can change compiled code.

    uv run python scripts/numba_cache_key.py           # the hash
    uv run python scripts/numba_cache_key.py --list    # the files it covers

Numba (``cache=True``) checks a cached kernel against its own source file's content, the numba
version, the signature, the CPU name and features, and the function's bytecode and closure cells.
It does not notice a change to a callee defined in another file, and it freezes a global's value
at compile time without checking it again (Numba, *Compiling Python code with @jit: cache*). So a
restored cache is safe only while every file a kernel can reach is unchanged. That is:

- every superglm module that compiles with numba: one that names a compile entry point
  (``njit``, ``jit``, ``vectorize``, ``guvectorize``, ``overload``, ``register_jitable`` and the
  rest of ``COMPILERS``), imported at module level or inside a function. A module that only
  reads numba's state (its compile events, cache locator or thread count) compiles nothing, and
  no superglm module sets a numba option that changes generated code;
- every superglm module those import, transitively, since callees and frozen constants come from
  there: ``import``, ``from ... import`` (absolute or relative), imports inside functions, and a
  dotted string literal naming a superglm module, which is how lazy re-export tables load one;
- ``uv.lock``, which pins numba, llvmlite, NumPy and SciPy.

A parent package's ``__init__`` runs when a submodule is imported but supplies none of its
names, so it is in the key only when a covered module imports from it. Imports under
``if TYPE_CHECKING:`` never run and are skipped. The host CPU, Python version and shard are in the
workflow's key beside this hash.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def module_paths(src: Path) -> dict[str, Path]:
    """Each module under ``src`` (the directory holding the ``superglm`` package) by dotted name."""
    modules = {}
    for path in sorted((src / "superglm").rglob("*.py")):
        parts = path.relative_to(src).with_suffix("").parts
        modules[".".join(parts[:-1] if parts[-1] == "__init__" else parts)] = path
    return modules


def _is_type_checking(test: ast.expr) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _nodes(tree: ast.AST):
    """Every node of ``tree`` except the bodies of ``if TYPE_CHECKING:`` blocks."""
    stack = [tree]
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, ast.If) and _is_type_checking(node.test):
            stack.extend(node.orelse)
            continue
        stack.extend(ast.iter_child_nodes(node))


# numba's compile entry points: a module that names one defines compiled code.
COMPILERS = frozenset(
    {
        "jit",
        "njit",
        "vectorize",
        "guvectorize",
        "cfunc",
        "stencil",
        "generated_jit",
        "jitclass",
        "intrinsic",
        "overload",
        "overload_method",
        "overload_attribute",
        "overload_classmethod",
        "register_jitable",
    }
)


def _is_numba(name: str | None) -> bool:
    return name is not None and (name == "numba" or name.startswith("numba."))


def _dotted(node: ast.expr) -> list[str]:
    """``a.b.c`` as ``["a", "b", "c"]``, or ``[]`` for anything but a dotted name."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    return [node.id, *reversed(parts)] if isinstance(node, ast.Name) else []


def scan(name: str, path: Path, modules: dict[str, Path]) -> tuple[bool, set[str]]:
    """Whether module ``name`` compiles with numba, and the superglm modules it imports."""
    package = name if path.name == "__init__.py" else name.rpartition(".")[0]
    nodes = list(_nodes(ast.parse(path.read_bytes(), filename=str(path))))
    compiles, imported, numba_names = False, set(), set()
    for node in nodes:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_numba(alias.name):
                    numba_names.add(alias.asname or "numba")
                imported.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.rsplit(".", node.level - 1)[0] if node.level > 1 else package
                base = f"{base}.{node.module}" if node.module else base
            else:
                base = node.module or ""
            for alias in node.names:
                if _is_numba(base):
                    compiles |= alias.name in COMPILERS
                    numba_names.add(alias.asname or alias.name)
                submodule = f"{base}.{alias.name}"
                imported.add(submodule if submodule in modules else base)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and "." in node.value:
            imported.add(node.value)
    # ``numba.njit``, ``nb.vectorize``, ``numba.extending.overload``: an attribute of numba
    compiles |= any(
        len(parts) > 1 and parts[0] in numba_names and parts[-1] in COMPILERS
        for parts in (_dotted(node) for node in nodes if isinstance(node, ast.Attribute))
    )
    return compiles, imported & modules.keys()


def covered_files(root: Path = ROOT) -> list[Path]:
    """The files the key hashes: numba modules, their import closure, and ``uv.lock``."""
    modules = module_paths(root / "src")
    scans = {name: scan(name, path, modules) for name, path in modules.items()}
    pending = [name for name, (compiles, _) in scans.items() if compiles]
    covered = set(pending)
    while pending:
        for imported in scans[pending.pop()][1] - covered:
            covered.add(imported)
            pending.append(imported)
    return sorted(modules[name] for name in covered) + [root / "uv.lock"]


def cache_key(root: Path = ROOT) -> str:
    """SHA-256 over each covered file's path relative to ``root`` and content."""
    digest = hashlib.sha256()
    for path in covered_files(root):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=ROOT, help="repository root")
    parser.add_argument("--list", action="store_true", help="print the covered files instead")
    args = parser.parse_args(argv)
    if args.list:
        for path in covered_files(args.root):
            print(path.relative_to(args.root).as_posix())
    else:
        print(cache_key(args.root))


if __name__ == "__main__":
    main()
