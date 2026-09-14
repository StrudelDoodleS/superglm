# Local research tool installation

Installed with user approval on 2026-09-13. No additional skills or paper MCP
were installed. The historic Lean compiler and MCP receipts remain unchanged.

## Lean MCP

The [configuration template](lean-gaussian-certificate/codex-mcp.toml.example)
was installed at both locations:

- `/home/max/projects/superglm/.codex/config.toml`
- `/home/max/projects/superglm/.worktrees/adaptive-interactions/.codex/config.toml`

Neither file existed before installation. Both use upstream
`lean-lsp-mcp==0.30.0` with `leanclient==0.13.2` through `uvx` and point to the
research worktree's `docs/research/lean-gaussian-certificate`. Keep that worktree
available or update both absolute paths when relocating the proof project.
The shared Git `info/exclude` gained `/.codex/config.toml`; existing rules remain.
The user-level Codex configuration was unchanged, verified by SHA-256 before
and after installation.

From each checkout, these commands exited 0:

```sh
codex mcp get lean_research --json
codex mcp list --json
```

The inspected output was restricted to `lean_research` to avoid exposing other
server configuration. Both reported `enabled: true`, `disabled_reason: null`,
stdio transport, the pinned launch arguments and the intended proof directory.
`get` reported exactly `lean_goal`, `lean_diagnostic_messages`, `lean_hover_info`,
`lean_local_search` and `lean_multi_attempt` in `enabled_tools`. Environment
values include empty `LEAN_MCP_INSTRUCTIONS`, `LEAN_REPL=false`,
`LEAN_LOOGLE_LOCAL=false` and all 18 other tools disabled. These commands verify
configuration resolution, not a fresh server handshake. The prior
[MCP smoke receipt](lean-gaussian-certificate/mcp-validation.json) records the
handshake and all five working calls with the same package pins and settings.
The running agent tool catalog still contains no Lean tools; reconnect to load
the configured server.

## SymPy

The environment is independent of the project's `.venv`, `pyproject.toml` and
`uv.lock`. Create it once and install the research pins from the worktree root.
These commands use the current `notes/research` location; the original
installation used `docs/research` in the preserved archival worktree:

```sh
uv venv --python 3.13 /home/max/.local/share/superglm-research/.venv
uv pip install --python /home/max/.local/share/superglm-research/.venv/bin/python -r notes/research/sympy-requirements.txt
/home/max/.local/share/superglm-research/.venv/bin/python notes/research/check_sympy_quadratic_gap.py
uv pip check --python /home/max/.local/share/superglm-research/.venv/bin/python
```

The checks exited 0 with Python 3.13.14, SymPy 1.14.0 and mpmath 1.3.0.
`uv pip check` checked two packages and reported all packages compatible.
The symbolic script returned exact zero remainders for the scalar stationary
quadratic gap, the symmetric 2-by-2 gap and its inverse-residual form.
The inverse expression assumes `a*d-c**2 != 0`. Stationarity is imposed by
substitution and symmetry by construction. These symbolic checks do not certify
floating-point computation or establish positive definiteness.

Use the environment's Python executable directly for later experiments; shell
activation is optional. No SymPy MCP wrapper is needed.

The current symbolic check scripts refuse `-O` and `PYTHONOPTIMIZE` before
importing optional dependencies, so a disabled assertion cannot report a
successful check. The symbolic identities are unchanged. The original
validation receipts retain the script hashes and outputs from before this
guard was added; they describe the preserved `639f499e` source checkpoint.
