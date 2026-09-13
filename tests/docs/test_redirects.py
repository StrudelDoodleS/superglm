"""Every URL that was live on the MkDocs site redirects to an existing page.

The MkDocs site served directory URLs (``guide/monotone/``). The redirect
map lists the source document rediraffe writes for each one
(``guide/monotone/index.md``) and the page it forwards to.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEGACY = ROOT / "tests" / "docs" / "fixtures" / "legacy_urls.txt"
REDIRECTS = ROOT / "docs" / "redirects.txt"
DOCS = ROOT / "docs"


def redirect_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for raw in REDIRECTS.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        old, new = line.split()
        mapping[old] = new
    return mapping


def legacy_urls() -> list[str]:
    return [line.strip() for line in LEGACY.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_every_legacy_url_has_a_redirect() -> None:
    mapping = redirect_map()
    missing = [url for url in legacy_urls() if f"{url}index.md" not in mapping]
    assert missing == [], f"live URLs with no redirect: {missing}"


def test_every_redirect_target_exists() -> None:
    dangling = [new for new in redirect_map().values() if not (DOCS / new).exists()]
    assert dangling == [], f"redirect targets that are not pages: {dangling}"
