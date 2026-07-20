"""Validate the contents and shape of SuperGLM release archives."""

from __future__ import annotations

import argparse
import tarfile
import zipfile
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

NATIVE_SUFFIXES = (".dll", ".dylib", ".exe", ".so")
REPOSITORY_ONLY_SDIST_ENTRIES = {
    ".github",
    "benchmarks",
    "docs",
    "jsconfig.json",
    "package-lock.json",
    "package.json",
    "scratch",
    "uv.lock",
}


def _required_editor_assets(source_root: Path) -> set[str]:
    editor_root = source_root / "src/superglm/editor/app"
    assets = {
        f"superglm/editor/app/{path.relative_to(editor_root).as_posix()}"
        for path in editor_root.rglob("*")
        if path.is_file() and (path.name == "index.html" or path.suffix in {".js", ".css"})
    }
    if not assets:
        raise ValueError(f"No editor assets found below {editor_root}")
    return assets


def _reject_bad_archive_names(names: Iterable[str], archive: Path) -> None:
    bad: list[str] = []
    for name in names:
        parts = PurePosixPath(name).parts
        if (
            name.startswith(("/", "\\"))
            or "\\" in name
            or ".." in parts
            or name.lower().endswith(NATIVE_SUFFIXES)
        ):
            bad.append(name)
    if bad:
        raise ValueError(f"Unexpected entries in {archive}: {bad[:20]}")


def _require_editor_assets(names: set[str], archive: Path, required_suffixes: set[str]) -> None:
    if archive.name.endswith(".whl"):
        source_prefix = ""
    elif archive.name.endswith(".tar.gz"):
        expected_root = archive.name.removesuffix(".tar.gz")
        roots = {parts[0] for name in names if (parts := PurePosixPath(name).parts)}
        if roots != {expected_root}:
            raise ValueError(
                f"Unexpected source roots in {archive}: "
                f"expected {[expected_root]}, found {sorted(roots)}"
            )
        source_prefix = f"{expected_root}/src/"
    else:  # pragma: no cover - callers accept only wheel and sdist suffixes
        raise ValueError(f"Unsupported release archive: {archive}")

    required = {f"{source_prefix}{suffix}" for suffix in required_suffixes}
    missing = required.difference(names)
    if missing:
        raise ValueError(f"Missing editor assets in {archive}: {sorted(missing)}")


def _wheel_names(wheel: Path) -> set[str]:
    with zipfile.ZipFile(wheel) as archive:
        return set(archive.namelist())


def _sdist_names(sdist: Path) -> set[str]:
    with tarfile.open(sdist) as archive:
        links = [member.name for member in archive.getmembers() if member.issym() or member.islnk()]
        if links:
            raise ValueError(f"Unexpected links in {sdist}: {links[:20]}")
        return {member.name for member in archive.getmembers()}


def _reject_repository_only_sdist_entries(names: set[str], sdist: Path) -> None:
    root = sdist.name.removesuffix(".tar.gz")
    bad: list[str] = []
    for name in names:
        parts = PurePosixPath(name).parts
        if len(parts) > 1 and parts[0] == root and parts[1] in REPOSITORY_ONLY_SDIST_ENTRIES:
            bad.append(name)
    if bad:
        raise ValueError(f"Repository-only entries found in {sdist}: {bad[:20]}")


def verify_release_artifacts(dist_dir: Path, *, source_root: Path) -> None:
    wheels = sorted(dist_dir.glob("*.whl"))
    sdists = sorted(dist_dir.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ValueError(
            f"Expected one wheel and one sdist in {dist_dir}; "
            f"found {len(wheels)} wheel(s) and {len(sdists)} sdist(s)"
        )

    wheel = wheels[0]
    sdist = sdists[0]
    if not wheel.name.endswith("-py3-none-any.whl"):
        raise ValueError(f"Expected a universal pure-Python wheel, found {wheel.name}")

    required_assets = _required_editor_assets(source_root)
    wheel_names = _wheel_names(wheel)
    sdist_names = _sdist_names(sdist)
    for archive, names in ((wheel, wheel_names), (sdist, sdist_names)):
        _reject_bad_archive_names(names, archive)
        _require_editor_assets(names, archive, required_assets)
    _reject_repository_only_sdist_entries(sdist_names, sdist)

    print(f"Verified {wheel.name} and {sdist.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dist_dir", type=Path)
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    verify_release_artifacts(args.dist_dir, source_root=args.source_root)


if __name__ == "__main__":
    main()
