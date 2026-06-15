# Security Policy

## Supported Versions

Only the latest released minor version is supported for security fixes.

## Reporting a vulnerability

Please report suspected vulnerabilities privately to the repository owner. Do not open public
issues for suspected vulnerabilities until a fix or disclosure plan is agreed.

## Security Controls

This repository uses a supply-chain security evidence workflow on pull requests, `master`
pushes, scheduled runs, and manual dispatch. The controls are intended to produce auditable
evidence for model-risk and software-governance review. They do not mathematically prove the
absence of malicious code.

Current controls include:

- CodeQL code scanning for common vulnerable code patterns.
- Dependency vulnerability scanning with `pip-audit`.
- Dependency review on pull requests.
- OpenSSF Scorecard reporting.
- CycloneDX SBOM generation.
- Package build/content checks for wheel and source distributions.
- Restricted GitHub Actions token permissions.
- CODEOWNERS coverage for workflow, packaging, and package-source changes.

## Release Provenance

Release artifact attestation and signed provenance should be added when the project has a
formal package publishing workflow. Until then, the security workflow builds and inspects local
wheel and source-distribution artifacts as governance evidence.
