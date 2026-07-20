import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = (ROOT / "pyproject.toml").read_text(encoding="utf-8")


def _toml_section(name: str) -> str:
    marker = f"[{name}]"
    _, found, remainder = PYPROJECT.partition(marker)
    assert found, f"missing {marker}"
    next_section = re.search(r"^\[", remainder, flags=re.MULTILINE)
    return remainder[: next_section.start()] if next_section else remainder


def test_editor_dependencies_are_part_of_the_normal_install() -> None:
    project = _toml_section("project")
    optional = _toml_section("project.optional-dependencies")

    assert '"fastapi>=0.115"' in project
    assert '"uvicorn>=0.30"' in project
    assert "ipykernel" not in project

    assert re.search(r"^editor\s*=", optional, flags=re.MULTILINE) is None
    assert re.search(r"^all\s*=", optional, flags=re.MULTILINE) is None
    assert "fastapi" not in optional
    assert "uvicorn" not in optional
    assert "ipykernel" not in optional


def test_dataframe_boundary_dependencies_are_explicit() -> None:
    project = _toml_section("project")
    optional = _toml_section("project.optional-dependencies")

    assert '"narwhals>=2.17.0"' in project
    assert "polars" not in project
    assert '"polars>=1.42.1"' in optional
    assert "pyarrow" not in project


def test_project_exposes_useful_pypi_metadata() -> None:
    project = _toml_section("project")
    urls = _toml_section("project.urls")

    assert 'license = "MIT"' in project
    assert 'license-files = ["LICENSE"]' in project
    assert "keywords = [" in project
    assert "classifiers = [" in project
    assert '"Programming Language :: Python :: 3.14"' in project

    for label in ("Documentation", "Homepage", "Issues", "Repository"):
        assert re.search(rf"^{label}\s*=", urls, flags=re.MULTILINE)


def test_sdist_uses_a_small_explicit_allowlist() -> None:
    sdist = _toml_section("tool.hatch.build.targets.sdist")

    assert "only-include = [" in sdist
    for required in ("src", "LICENSE", "README.md", "pyproject.toml"):
        assert f'"{required}"' in sdist

    for repository_only in (
        "benchmarks",
        "docs",
        "run_test.py",
        "scratch",
        "tests",
        "uv.lock",
    ):
        assert f'"{repository_only}"' not in sdist


def test_installation_docs_are_pypi_first() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    installation = (ROOT / "docs/getting-started/installation.md").read_text(encoding="utf-8")

    assert "pip install superglm" in readme
    assert "pip install superglm" in installation
    assert "pip install git+" not in readme
    assert "pip install git+" not in installation
    assert '<img src="https://raw.githubusercontent.com/' in readme
