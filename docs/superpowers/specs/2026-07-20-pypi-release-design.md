# PyPI release design

## Goal

Publish SuperGLM as a normal pure-Python package using PyPI Trusted Publishing,
then create a GitHub Release from the exact same checked artifacts. The first
successful upload will create and claim the pending `superglm` PyPI project.

## Package contract

- Ship one `py3-none-any` wheel and one source distribution.
- Make the editor available from a normal `pip install superglm`: FastAPI and
  Uvicorn become core dependencies.
- Remove the `editor` compatibility extra and its unused `ipykernel` runtime
  dependency. Keep `ipykernel` only in the documentation dependency group.
- Remove the misleading `all` extra, which currently installs contributor and
  benchmark tooling for end users.
- Add PyPI-facing project URLs, classifiers, keywords, and SPDX license
  metadata.
- Replace Git-only installation instructions with PyPI-first instructions and
  use a PyPI-safe absolute logo URL.
- Restrict the sdist to release inputs, source, tests, and top-level legal/readme
  files so generated notebooks and benchmark records are not shipped.

## Release flow

A pushed semantic version tag such as `v0.12.0` starts `.github/workflows/release.yml`.

1. A read-only build job verifies that the tag, `pyproject.toml`, and
   `superglm.__version__` agree and that the tagged commit belongs to master.
2. It builds the wheel and sdist once, validates metadata/archive contents, and
   smoke-tests installation from the wheel.
3. A `pypi` environment job downloads those artifacts and publishes them with
   OIDC. It receives only `id-token: write` permission.
4. Only after PyPI succeeds, a separate `contents: write` job creates the
   GitHub Release with generated notes and attaches the same wheel and sdist.

Actions and the uv tool version are pinned. No PyPI token is stored in GitHub.
The release page cannot claim success before PyPI accepts the artifacts.

## Validation

Packaging tests assert dependency/extra metadata, sdist exclusions, universal
wheel tags, required editor assets, and workflow permissions/order. Local
verification builds from a clean tree, runs Twine and wheel-content checks,
installs the built wheel in an isolated environment, and exercises imports.
