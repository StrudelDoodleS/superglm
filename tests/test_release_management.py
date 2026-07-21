from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from scripts.bump_version import VersionBumpError, bump_version, expected_next_version

ROOT = Path(__file__).resolve().parents[1]


def _write_version_fixture(
    root: Path,
    *,
    project_version: str = "0.12.3",
    source_version: str | None = None,
) -> tuple[Path, Path]:
    source_version = project_version if source_version is None else source_version
    pyproject = root / "pyproject.toml"
    source = root / "src/superglm/__init__.py"
    source.parent.mkdir(parents=True)
    pyproject.write_text(
        '[project]\nname = "superglm"\nversion = '
        f'"{project_version}"\n\n[tool.ruff]\nline-length = 100\n',
        encoding="utf-8",
    )
    source.write_text(
        f'PUBLIC = True\n__version__ = "{source_version}"\n',
        encoding="utf-8",
    )
    return pyproject, source


@pytest.mark.parametrize(
    ("current", "impact", "expected"),
    [
        ("0.12.0", "patch", "0.12.1"),
        ("0.12.9", "patch", "0.12.10"),
        ("0.12.3", "minor", "0.13.0"),
        ("0.99.8", "minor", "0.100.0"),
    ],
)
def test_expected_next_version_is_deterministic(
    current: str,
    impact: str,
    expected: str,
) -> None:
    assert expected_next_version(current, impact) == expected


@pytest.mark.parametrize("impact", ["none", "needs-human-decision", "major", ""])
def test_expected_next_version_rejects_non_releasable_impacts(impact: str) -> None:
    with pytest.raises(VersionBumpError, match="patch or minor"):
        expected_next_version("0.12.3", impact)


def test_bump_version_updates_both_source_versions(tmp_path: Path) -> None:
    pyproject, source = _write_version_fixture(tmp_path)

    result = bump_version(tmp_path, requested="0.12.4", impact="patch")

    assert result == "0.12.4"
    assert 'version = "0.12.4"' in pyproject.read_text(encoding="utf-8")
    assert '__version__ = "0.12.4"' in source.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("requested", "impact", "message"),
    [
        ("0.12.3", "patch", "expected 0.12.4"),
        ("0.12.5", "patch", "expected 0.12.4"),
        ("0.13.0", "patch", "expected 0.12.4"),
        ("0.12.4", "minor", "expected 0.13.0"),
        ("1.0.0", "minor", "pre-1.0"),
        ("0.13", "minor", "pre-1.0"),
        ("v0.13.0", "minor", "pre-1.0"),
    ],
)
def test_invalid_bumps_leave_both_files_unchanged(
    tmp_path: Path,
    requested: str,
    impact: str,
    message: str,
) -> None:
    pyproject, source = _write_version_fixture(tmp_path)
    before = (pyproject.read_bytes(), source.read_bytes())

    with pytest.raises(VersionBumpError, match=message):
        bump_version(tmp_path, requested=requested, impact=impact)

    assert (pyproject.read_bytes(), source.read_bytes()) == before


def test_mismatched_source_versions_fail_without_writes(tmp_path: Path) -> None:
    pyproject, source = _write_version_fixture(tmp_path, source_version="0.12.2")
    before = (pyproject.read_bytes(), source.read_bytes())

    with pytest.raises(VersionBumpError, match="do not agree"):
        bump_version(tmp_path, requested="0.12.4", impact="patch")

    assert (pyproject.read_bytes(), source.read_bytes()) == before


def test_cli_updates_an_explicit_fixture_root(tmp_path: Path) -> None:
    pyproject, source = _write_version_fixture(tmp_path)

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/bump_version.py"),
            "0.13.0",
            "--impact",
            "minor",
            "--root",
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "Updated SuperGLM version: 0.12.3 -> 0.13.0"
    assert 'version = "0.13.0"' in pyproject.read_text(encoding="utf-8")
    assert '__version__ = "0.13.0"' in source.read_text(encoding="utf-8")
