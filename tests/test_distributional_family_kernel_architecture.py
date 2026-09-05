from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

ROOT = Path(__file__).parents[1] / "src/superglm/distributional"
KERNELS = ROOT / "kernels"
ADAPTERS = ROOT / "families"


def _modules() -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in ROOT.rglob("*.py"):
        parts = path.relative_to(ROOT).with_suffix("").parts
        module_parts = parts[:-1] if parts[-1] == "__init__" else parts
        result[".".join(module_parts)] = path
    return result


def _distributional_imports(path: Path) -> Iterator[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    current = ".".join(path.relative_to(ROOT).with_suffix("").parts)
    package = current.rsplit(".", 1)[0] if "." in current else ""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                prefix = "superglm.distributional."
                if alias.name.startswith(prefix):
                    yield alias.name[len(prefix) :]
                elif alias.name == "superglm.distributional":
                    yield ""
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = package.split(".") if package else []
                base = parts[: len(parts) - node.level + 1]
                if node.module:
                    base.extend(node.module.split("."))
                yield ".".join(base)
            elif node.module == "superglm.distributional":
                yield ""
            elif node.module and node.module.startswith("superglm.distributional."):
                yield node.module.removeprefix("superglm.distributional.")


def _reachable(graph: dict[str, set[str]], start: str) -> set[str]:
    reached: set[str] = set()
    pending = [start]
    while pending:
        node = pending.pop()
        if node in reached:
            continue
        reached.add(node)
        pending.extend(graph.get(node, set()) - reached)
    return reached


def _strong_components(graph: dict[str, set[str]]) -> tuple[tuple[str, ...], ...]:
    remaining = set(graph)
    components: list[tuple[str, ...]] = []
    while remaining:
        root = min(remaining)
        reachable_from_root = _reachable(graph, root)
        reaches_root = {node for node in remaining if root in _reachable(graph, node)}
        component = remaining & reachable_from_root & reaches_root
        components.append(tuple(sorted(component)))
        remaining -= component
    return tuple(sorted(components))


def test_distributional_internal_imports_are_acyclic_and_never_use_aggregates() -> None:
    modules = _modules()
    graph = {
        module: {
            imported
            for imported in _distributional_imports(path)
            if imported in modules and imported != module
        }
        for module, path in modules.items()
    }
    cycles = [component for component in _strong_components(graph) if len(component) > 1]
    assert cycles == []

    forbidden = {"", "families", "kernels"}
    offenders = {
        str(path.relative_to(ROOT)): sorted(forbidden & set(_distributional_imports(path)))
        for path in ROOT.rglob("*.py")
        if path.name != "__init__.py" and forbidden & set(_distributional_imports(path))
    }
    assert offenders == {}


def test_kernels_import_no_distributional_module_or_contract() -> None:
    expected = {"_tweedie_numba.py", "negative_binomial.py", "tweedie.py"}
    assert KERNELS.is_dir()
    assert expected <= {path.name for path in KERNELS.glob("*.py")}

    allowed = {
        "gaussian.py": {"kernels._common"},
        "gamma.py": {"kernels._common"},
        "generalized_gamma.py": {
            "kernels._common",
            "kernels.gamma",
            "kernels.log_normal",
        },
        "generalized_pareto.py": {"kernels._common"},
        "log_normal.py": {"kernels._common", "kernels.gaussian"},
        "negative_binomial.py": {"kernels._common"},
        "two_piece.py": {"kernels._common"},
        "tweedie.py": {"kernels._tweedie_numba", "kernels._common"},
    }
    offenders = {
        str(path.relative_to(ROOT)): sorted(
            set(_distributional_imports(path)) - allowed.get(path.name, set())
        )
        for path in KERNELS.glob("*.py")
        if set(_distributional_imports(path)) - allowed.get(path.name, set())
    }
    assert offenders == {}


def test_kernels_share_their_primitive_helpers() -> None:
    duplicated = (
        "WeightSemantics",
        "_readonly",
        "_readonly_bool",
        "_semantics",
        "_weights",
        "_derivative_order",
        "_order",
    )
    offenders = {}
    for path in sorted(KERNELS.glob("*.py")):
        if path.name in {"__init__.py", "_common.py", "_tweedie_numba.py"}:
            continue
        tree = ast.parse(path.read_text())
        names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)} | {
            target.id
            for node in tree.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        hits = sorted(names & set(duplicated))
        if hits:
            offenders[path.name] = hits
    assert offenders == {}


def test_contracts_and_family_adapters_follow_the_one_way_edge_table() -> None:
    allowed = {
        ROOT / "family.py": {"weights"},
        ROOT / "weights.py": set(),
        ADAPTERS / "_base.py": {"weights"},
        ADAPTERS / "gaussian.py": {"family", "weights", "kernels.gaussian", "families._base"},
        ADAPTERS / "gamma.py": {"family", "weights", "kernels.gamma", "families._base"},
        ADAPTERS / "generalized_gamma.py": {
            "family",
            "weights",
            "kernels.generalized_gamma",
            "families._base",
            "families.gaussian",
        },
        ADAPTERS / "negative_binomial.py": {
            "family",
            "weights",
            "kernels.negative_binomial",
            "families._base",
        },
        ADAPTERS / "tweedie.py": {"family", "weights", "kernels.tweedie", "families._base"},
        ADAPTERS / "generalized_pareto.py": {
            "family",
            "weights",
            "kernels.generalized_pareto",
            "families._base",
            "families._links",
        },
        ADAPTERS / "two_piece.py": {
            "family",
            "weights",
            "kernels.two_piece",
            "families._base",
            "families._links",
            "families.gaussian",
        },
        ADAPTERS / "log_normal.py": {
            "family",
            "weights",
            "kernels.log_normal",
            "families._base",
            "families.gaussian",
        },
        ADAPTERS / "_links.py": set(),
    }
    offenders = {
        str(path.relative_to(ROOT)): sorted(set(_distributional_imports(path)) - admitted)
        for path, admitted in allowed.items()
        if set(_distributional_imports(path)) - admitted
    }
    assert offenders == {}

    for relative in ("api.py", "model.py", "solver/chunks.py", "solver/solver.py"):
        family_edges = {
            imported
            for imported in _distributional_imports(ROOT / relative)
            if imported == "families" or imported.startswith("families.")
        }
        assert family_edges == set(), (relative, sorted(family_edges))


def test_adapters_share_their_plumbing() -> None:
    duplicated = ("_immutable", "_readonly", "_array_digest", "_response_row_count")
    offenders = {}
    for path in sorted(ADAPTERS.glob("*.py")):
        if path.name in {"__init__.py", "_base.py"}:
            continue
        tree = ast.parse(path.read_text())
        names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
        hits = sorted(names & set(duplicated))
        if hits:
            offenders[path.name] = hits
    assert offenders == {}


def test_validators_are_public() -> None:
    from superglm.distributional import validated_derivative_order, validated_parameter_matrix

    assert callable(validated_derivative_order) and callable(validated_parameter_matrix)


def test_package_initializers_do_not_eagerly_import_implementations() -> None:
    for relative in ("__init__.py", "families/__init__.py", "kernels/__init__.py"):
        tree = ast.parse((ROOT / relative).read_text(), filename=relative)
        assert not any(isinstance(node, ast.Import | ast.ImportFrom) for node in tree.body), (
            relative
        )
