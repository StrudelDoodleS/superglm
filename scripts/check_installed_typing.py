"""Build and check root API typing from an isolated wheel installation.

Run with ``uv run python scripts/check_installed_typing.py``. The temporary
environment installs wheel dependencies and the project's pinned ty version.
No source-path override or editable install is used.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import tomllib
import zipfile
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    config = tomllib.loads((root / "pyproject.toml").read_text())
    checker = next(
        dep for dep in config["project"]["optional-dependencies"]["dev"] if dep.startswith("ty==")
    )
    env = {
        key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "VIRTUAL_ENV"}
    }

    def run(*args: str, cwd: Path, success: bool = True) -> str:
        result = subprocess.run(args, cwd=cwd, env=env, text=True, capture_output=True)
        output = result.stdout + result.stderr
        print(output, end="", flush=True)
        if success and result.returncode:
            raise RuntimeError(f"Command failed ({result.returncode}): {args}")
        if not success and result.returncode != 1:
            raise AssertionError(f"Expected typing failure, got {result.returncode}: {args}")
        return output

    with tempfile.TemporaryDirectory(prefix="superglm-wheel-typing-") as directory:
        work = Path(directory)
        run("uv", "build", "--wheel", "--out-dir", str(work / "dist"), cwd=root)
        (wheel,) = (work / "dist").glob("*.whl")
        with zipfile.ZipFile(wheel) as archive:
            assert "superglm/py.typed" in archive.namelist(), "wheel lacks PEP 561 marker"
        run("uv", "venv", "--python", "3.13", str(work / "venv"), cwd=work)
        python = work / "venv/bin/python"
        run("uv", "pip", "install", "--python", str(python), str(wheel), checker, cwd=work)
        run(
            str(python),
            "-I",
            "-c",
            (
                "import pathlib, sys, superglm; "
                "p = pathlib.Path(superglm.__file__).resolve(); print('Installed import:', p); "
                "assert p.is_relative_to(pathlib.Path(sys.prefix).resolve()); "
                "assert 'site-packages' in p.parts"
            ),
            cwd=work,
        )
        for name in ("valid.py", "invalid.py"):
            shutil.copyfile(root / "tests/typing" / name, work / name)
        ty = str(work / "venv/bin/ty")
        run(
            ty, "check", "--python", str(python), "--output-format", "concise", "valid.py", cwd=work
        )
        output = run(
            ty,
            "check",
            "--python",
            str(python),
            "--output-format",
            "concise",
            "invalid.py",
            cwd=work,
            success=False,
        )
        for line in (4, 5, 7, 8, 9):
            assert f"invalid.py:{line}:" in output, f"Missing negative diagnostic at line {line}"
        assert output.count("error[") == 5, "Unexpected negative diagnostics"
        assert output.count("unresolved-attribute") == 2
        assert output.count("invalid-argument-type") == 3
        print("Installed-wheel typing passed: valid imports/types and all five negatives.")


if __name__ == "__main__":
    main()
