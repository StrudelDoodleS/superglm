import re
from pathlib import Path

import superglm

ROOT = Path(__file__).resolve().parents[1]


def _version_from_pyproject() -> str:
    match = re.search(
        r'^version = "([^"]+)"$',
        (ROOT / "pyproject.toml").read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    assert match is not None
    return match.group(1)


def _local_package_version_from_lock() -> str:
    lock_text = (ROOT / "uv.lock").read_text(encoding="utf-8")
    match = re.search(
        r'\[\[package\]\]\s+name = "superglm"\s+version = "([^"]+)"',
        lock_text,
        flags=re.MULTILINE,
    )
    assert match is not None
    return match.group(1)


def test_package_version_metadata_is_consistent():
    pyproject_version = _version_from_pyproject()

    assert superglm.__version__ == pyproject_version
    assert _local_package_version_from_lock() == pyproject_version
