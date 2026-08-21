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


def test_the_two_pyarrow_floors_are_declared_identically() -> None:
    """Because Dependabot resolves a package to ONE requirement string.

    ``pyarrow`` is declared in both ``dev`` and ``bench``.  When those two
    strings disagree, Dependabot rewrites the higher one to the lower on every
    pull request that touches the package -- it did so on #318, and again on
    #354 after #318 was superseded, whose entire ``pyproject.toml`` diff was
    that one line.  The value it lands on is whichever is lower, so the
    disagreement is what creates the rewrite, not the specific numbers.

    ``dev``'s floor is the one with a reason attached and the one that binds:
    ``dev-ci.yml``'s 3.12 job installs ``dev`` and ``bench`` together and runs
    pytest with no out-of-band ``--with pyarrow``, so the intersection of the
    two is its effective constraint.

    Aligning them once is not durable -- nothing stops the next bump moving one
    and not the other, and then this recurs for a third time.  This asserts the
    invariant instead, so CI remembers rather than a reader having to.
    """
    optional = _toml_section("project.optional-dependencies")

    floors = re.findall(r'"pyarrow([^"]*)"', optional)
    assert len(floors) == 2, f"expected pyarrow in exactly two extras, found {floors}"
    assert floors[0] == floors[1], (
        "the dev and bench pyarrow specifiers have diverged, which is what lets "
        f"Dependabot normalise the higher one down to the lower: {floors}"
    )


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


def test_dataframe_boundary_documentation_is_discoverable() -> None:
    installation = (ROOT / "docs/getting-started/installation.md").read_text(encoding="utf-8")
    quickstart = (ROOT / "docs/getting-started/quickstart.md").read_text(encoding="utf-8")
    model_api = (ROOT / "docs/api/model.md").read_text(encoding="utf-8")
    mkdocs = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    developer_path = ROOT / "docs/development/data-and-solver-boundaries.md"

    assert developer_path.exists()
    developer = developer_path.read_text(encoding="utf-8")
    assert "pandas.DataFrame" in installation
    assert "eager Polars DataFrame" in installation
    assert "pip install polars" in installation
    assert "LazyFrame.collect()" in installation
    assert "Outputs remain pandas" in installation
    assert "import polars as pl" in quickstart
    assert "model = SuperGLM" in quickstart
    assert "model.predict(X)" in quickstart
    assert "design_summary" in model_api
    assert "development/data-and-solver-boundaries.md" in mkdocs
    assert "User layer" in developer
    assert "Developer layer" in developer
    assert "Where to make a change" in developer
    assert "future AFT" in developer
    assert "adds no AFT API" in developer
    assert "construction-time eligibility" in developer
    assert "traces" in developer
