from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
FEATURES_HTML = SITE / "api" / "features" / "index.html"
FAMILIES_HTML = SITE / "api" / "families" / "index.html"


def build_docs() -> None:
    subprocess.run(
        ["uv", "run", "--group", "docs", "mkdocs", "build", "-q"],
        cwd=ROOT,
        check=True,
    )


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def require(text: str, needle: str, label: str) -> None:
    if needle not in text:
        raise AssertionError(f"{label}: missing {needle!r}")


def forbid(text: str, needle: str, label: str) -> None:
    if needle in text:
        raise AssertionError(f"{label}: unexpected {needle!r}")


def check_features_chrome(text: str) -> None:
    forbid(text, "doc-symbol-heading", "features/chrome")
    forbid(text, "doc-symbol-toc", "features/chrome")


def check_features_structure(text: str) -> None:
    require(text, "Feature specs define how raw columns become model terms.", "features/structure")
    require(text, "Factory", "features/structure")
    require(text, "Spline Classes", "features/structure")
    require(text, "Other Feature Classes", "features/structure")
    require(
        text,
        'Use <code>kind="ps"</code> for a difference-penalized P-spline and '
        '<code>kind="bs"</code> for an integrated-derivative B-spline smooth.',
        "features/structure",
    )
    require(text, "PSpline", "features/structure")
    require(text, "BSplineSmooth", "features/structure")
    forbid(text, "BasisSpline", "features/structure")
    forbid(text, "module-attribute", "features/structure")


def check_families_chrome(text: str) -> None:
    forbid(text, "doc-symbol-heading", "families/chrome")
    forbid(text, "doc-symbol-toc", "families/chrome")


def check_families_structure(text: str) -> None:
    require(
        text,
        "Family objects define the response distribution used during fitting, scoring, and inference.",
        "families/structure",
    )
    require(text, "Factories", "families/structure")
    require(text, "Family Classes", "families/structure")
    require(text, "NegativeBinomial", "families/structure")
    require(text, "families.nb2", "families/structure")
    require(
        text,
        "Known-scale families keep <code>phi=1</code>. Negative binomial overdispersion is controlled by "
        "<code>theta</code>, not by a meaningful fitted <code>phi</code>.",
        "families/structure",
    )


def run_checks(page: str, mode: str) -> None:
    build_docs()
    if page in ("features", "all"):
        features = read_text(FEATURES_HTML)
        if mode in ("chrome", "all"):
            check_features_chrome(features)
        if mode in ("structure", "all"):
            check_features_structure(features)
    if page in ("families", "all"):
        families = read_text(FAMILIES_HTML)
        if mode in ("chrome", "all"):
            check_families_chrome(families)
        if mode in ("structure", "all"):
            check_families_structure(families)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--page", choices=["features", "families", "all"], default="all")
    parser.add_argument("--mode", choices=["chrome", "structure", "all"], default="all")
    args = parser.parse_args()
    run_checks(args.page, args.mode)
    print(f"API docs checks passed for page={args.page}, mode={args.mode}")


if __name__ == "__main__":
    main()
