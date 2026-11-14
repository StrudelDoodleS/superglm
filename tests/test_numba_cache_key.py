"""scripts/numba_cache_key.py: CI's numba cache key covers every file a kernel can reach."""

from __future__ import annotations

import gc
import importlib
import os
import pkgutil
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from scripts.numba_cache_key import ROOT, cache_key, covered_files

_PACKAGE = {
    "__init__.py": "",
    "_constants.py": "TOL = 1e-12\n",
    "_lazy_constants.py": "SCALE = 2.0\n",
    "_typing_only.py": "class Shape: ...\n",
    "kernels/__init__.py": "",
    "kernels/_callee.py": "def g(x):\n    return x\n",  # compiled by smooth.py
    "kernels/smooth.py": (
        "from typing import TYPE_CHECKING\n"
        "import numba as nb\n"
        "from superglm._constants import TOL\n"
        "from ._callee import g as _g\n"
        "if TYPE_CHECKING:\n"
        "    from superglm._typing_only import Shape\n"
        "\n"
        "g = nb.njit(cache=True)(_g)\n"
        "\n"
        "@nb.njit(cache=True)\n"
        "def f(x):\n"
        "    return g(x) + TOL\n"
        "\n"
        "def scale():\n"
        "    from superglm._lazy_constants import SCALE\n"
        "    return SCALE\n"
    ),
    "api.py": "import numpy as np\nfrom superglm.kernels.smooth import f\n",
    "_events.py": "from numba.core import event\n",
}


def _tree(root: Path) -> Path:
    for name, source in {**_PACKAGE, "../../uv.lock": "lock\n"}.items():
        path = root / "src/superglm" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    return root


def _covered(root: Path) -> set[str]:
    return {path.relative_to(root).as_posix() for path in covered_files(root)}


def test_the_key_holds_kernels_and_what_they_import_and_nothing_else(tmp_path: Path) -> None:
    assert _covered(_tree(tmp_path)) == {
        "src/superglm/kernels/smooth.py",
        "src/superglm/kernels/_callee.py",  # a callee's source, imported relatively
        "src/superglm/_constants.py",  # a frozen global from another module
        "src/superglm/_lazy_constants.py",  # imported inside a function
        "uv.lock",
    }
    # api.py imports a kernel but no kernel imports it; _events.py reads numba state
    # without compiling; the TYPE_CHECKING import never runs; __init__ supplies no names.


@pytest.mark.parametrize(
    ("changed", "moves"),
    [
        ("_constants.py", True),
        ("_lazy_constants.py", True),
        ("uv.lock", True),
        ("api.py", False),
        ("_typing_only.py", False),
    ],
)
def test_the_key_moves_only_with_a_covered_file(tmp_path: Path, changed: str, moves: bool) -> None:
    root = _tree(tmp_path)
    before = cache_key(root)
    path = root / ("uv.lock" if changed == "uv.lock" else f"src/superglm/{changed}")
    path.write_text(path.read_text() + "# edited\n")
    assert (cache_key(root) != before) is moves


def test_a_lazy_export_table_brings_its_modules_into_the_key(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    (root / "src/superglm/kernels/smooth.py").write_text(
        "from numba import njit\nfrom superglm.registry import LIMIT\n"
    )
    (root / "src/superglm/registry.py").write_text(
        '_EXPORTS = {"LIMIT": ("superglm._constants", "TOL")}\n'
    )
    assert {"src/superglm/registry.py", "src/superglm/_constants.py"} <= _covered(root)


def test_every_compiled_superglm_function_is_defined_in_a_covered_module() -> None:
    """Import the package and find every numba dispatcher and ufunc it built: each one's
    Python source must be in the key, or a change to it would reuse a stale kernel."""
    from numba.core.dispatcher import Dispatcher
    from numba.np.ufunc.dufunc import DUFunc

    import superglm

    for module in pkgutil.walk_packages(superglm.__path__, "superglm."):
        if module.name.endswith(".__main__"):
            continue
        try:
            importlib.import_module(module.name)
        except ImportError as error:  # an optional extra this environment lacks
            if (error.name or "").startswith("superglm"):
                raise
    covered = {path.resolve() for path in covered_files(ROOT)}
    defined = set()
    for obj in gc.get_objects():  # type(), not isinstance(): a dead weakref proxy raises
        if issubclass(type(obj), Dispatcher):
            function = obj.py_func
        elif issubclass(type(obj), DUFunc):
            function = obj._dispatcher.py_func
        else:
            continue
        source = Path(function.__code__.co_filename).resolve()
        if source.is_relative_to(ROOT / "src/superglm"):
            defined.add(source)
    assert defined, "found no superglm kernels to check"
    assert not defined - covered, sorted(defined - covered)


def test_ci_keys_the_numba_cache_on_this_script_and_no_partial_match() -> None:
    import yaml

    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
    steps = workflow["jobs"]["test-compatibility"]["steps"]
    (hashing,) = [step for step in steps if step.get("id") == "numba-sources"]
    (restore,) = [step for step in steps if "numba" in step.get("with", {}).get("key", "")]
    assert "python scripts/numba_cache_key.py" in hashing["run"]
    assert restore["with"]["key"].endswith("-${{ steps.numba-sources.outputs.hash }}")
    assert "restore-keys" not in restore["with"], "a partial match restores stale kernels"
    assert steps.index(hashing) < steps.index(restore)


@pytest.mark.skipif(sys.platform == "win32" or shutil.which("bash") is None, reason="needs bash")
@pytest.mark.parametrize("fails", [False, True], ids=["hashes", "script-fails"])
def test_a_failing_key_script_fails_the_step(tmp_path: Path, fails: bool) -> None:
    """Without this, an empty hash would give every commit the same key."""
    import yaml

    steps = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))["jobs"][
        "test-compatibility"
    ]["steps"]
    (hashing,) = [step for step in steps if step.get("id") == "numba-sources"]
    fake_uv = tmp_path / "uv"
    fake_uv.write_text(f"#!/bin/sh\nprintf 'abc\\r\\n'\nexit {int(fails)}\n")
    fake_uv.chmod(0o755)
    output = tmp_path / "github_output"
    env = {**os.environ, "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}"}
    env["GITHUB_OUTPUT"] = str(output)
    # GitHub runs ``shell: bash`` as ``bash --noprofile --norc -eo pipefail {0}``.
    result = subprocess.run(
        ["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", hashing["run"]], env=env
    )
    assert (result.returncode != 0) is fails
    assert (output.read_text() if output.exists() else "") == ("" if fails else "hash=abc\n")
