"""Validate and apply the approved next pre-1.0 SuperGLM version."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

_PRE_ONE_VERSION = re.compile(r"0\.(0|[1-9]\d*)\.(0|[1-9]\d*)\Z")
_SOURCE_VERSION = re.compile(r'^__version__ = "([^"]+)"$', flags=re.MULTILINE)


class VersionBumpError(ValueError):
    """Raised when a requested release transition violates project policy."""


def _parse_pre_one_version(value: str) -> tuple[int, int, int]:
    match = _PRE_ONE_VERSION.fullmatch(value)
    if match is None:
        raise VersionBumpError(f"version {value!r} must be a canonical pre-1.0 0.x.y version")
    return 0, int(match.group(1)), int(match.group(2))


def expected_next_version(current: str, impact: str) -> str:
    """Return the only permitted next version for a patch or minor impact."""
    _, minor, patch = _parse_pre_one_version(current)
    if impact == "patch":
        return f"0.{minor}.{patch + 1}"
    if impact == "minor":
        return f"0.{minor + 1}.0"
    raise VersionBumpError("release impact must be patch or minor")


def _project_version(pyproject: str) -> str:
    _, marker, remainder = pyproject.partition("[project]")
    if not marker:
        raise VersionBumpError("pyproject.toml has no [project] section")
    next_section = re.search(r"^\[", remainder, flags=re.MULTILINE)
    project = remainder[: next_section.start()] if next_section else remainder
    match = re.search(r'^version\s*=\s*"([^"]+)"$', project, flags=re.MULTILINE)
    if match is None:
        raise VersionBumpError("pyproject.toml [project] has no string version")
    return match.group(1)


def _replace_once(text: str, old: str, new: str, *, path: Path) -> str:
    if text.count(old) != 1:
        raise VersionBumpError(f"expected exactly one version marker in {path}")
    return text.replace(old, new, 1)


def bump_version(root: Path, *, requested: str, impact: str) -> str:
    """Validate the transition fully, then update both source version files."""
    root = root.resolve()
    pyproject_path = root / "pyproject.toml"
    source_path = root / "src/superglm/__init__.py"
    pyproject = pyproject_path.read_text(encoding="utf-8")
    source = source_path.read_text(encoding="utf-8")

    project_version = _project_version(pyproject)
    source_match = _SOURCE_VERSION.search(source)
    if source_match is None:
        raise VersionBumpError("src/superglm/__init__.py has no __version__ marker")
    source_version = source_match.group(1)
    if source_version != project_version:
        raise VersionBumpError("pyproject.toml and src/superglm/__init__.py versions do not agree")

    _parse_pre_one_version(requested)
    expected = expected_next_version(project_version, impact)
    if requested != expected:
        raise VersionBumpError(
            f"requested {requested}; {impact} impact from {project_version} expected {expected}"
        )

    updated_pyproject = _replace_once(
        pyproject,
        f'version = "{project_version}"',
        f'version = "{requested}"',
        path=pyproject_path,
    )
    updated_source = _replace_once(
        source,
        f'__version__ = "{source_version}"',
        f'__version__ = "{requested}"',
        path=source_path,
    )

    pyproject_path.write_text(updated_pyproject, encoding="utf-8")
    source_path.write_text(updated_source, encoding="utf-8")
    return requested


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", help="approved next version in canonical 0.x.y form")
    parser.add_argument("--impact", choices=("patch", "minor"), required=True)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root; defaults to the parent of scripts/",
    )
    args = parser.parse_args()
    try:
        current = _project_version((args.root / "pyproject.toml").read_text(encoding="utf-8"))
        updated = bump_version(args.root, requested=args.version, impact=args.impact)
    except (OSError, VersionBumpError) as exc:
        parser.error(str(exc))
    print(f"Updated SuperGLM version: {current} -> {updated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
