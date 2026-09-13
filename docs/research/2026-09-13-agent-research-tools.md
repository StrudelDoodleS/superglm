# Research tools without additional skill packs

Date: 2026-09-13. This records the user's request for useful agent tools while
avoiding additional workflow bundles and custom skills. It does not change
the current agent instructions or register another MCP server.

## Selected tool candidate

Use upstream [lean-lsp-mcp v0.30.0](https://github.com/oOo0oOo/lean-lsp-mcp/releases/tag/v0.30.0)
with `leanclient==0.13.2`, restricted to five local tools:

| Tool | Evidence it returns |
| --- | --- |
| `lean_goal` | The current proof goal at a source position |
| `lean_diagnostic_messages` | Lean errors, warnings and information |
| `lean_hover_info` | Declaration types and documentation |
| `lean_local_search` | Matching declarations in local sources |
| `lean_multi_attempt` | Goals and diagnostics for candidate tactics |

The [upstream tool reference](https://github.com/oOo0oOo/lean-lsp-mcp/blob/v0.30.0/docs/tools.md)
describes these interfaces. The persistent language server can provide feedback
between full compiler checks. Whether it reduces agent time or improves proof
success on our research is an empirical question; no Astra/Fable comparison has
been run. `lake build` and direct compiler checks remain the recorded validation
for completed proof files.

The [configuration example](lean-gaussian-certificate/codex-mcp.toml.example)
clears server instructions with `LEAN_MCP_INSTRUCTIONS=""`, disables the other
18 tools at server startup, and applies the same five-name client allowlist.
It disables optional REPL and local Loogle setup. The selected tools do not need
AI-provider keys or remote theorem-search services. This is not network
isolation; package/dependency downloads remain possible.

Use a project configuration for proof work. Its absolute paths are for this
machine and checkout. The example has not been installed as a live Codex config.
Only the server and Lean client versions are pinned by that launch command;
it does not lock all transitive Python packages.

[Codex's MCP documentation](https://developers.openai.com/codex/mcp)
supports project-scoped configuration and tool allowlists, and states that server
instructions are used as guidance. Filtering tools alone would not remove that
guidance. The empty environment override is supported by
[upstream startup code](https://github.com/oOo0oOo/lean-lsp-mcp/blob/v0.30.0/src/lean_lsp_mcp/server.py).
An empty CLI `--instructions` argument does not have the same effect because of
the [CLI truthiness check](https://github.com/oOo0oOo/lean-lsp-mcp/blob/v0.30.0/src/lean_lsp_mcp/__init__.py).

## Local validation

A temporary stdio server ran against our pinned Lean 4.33.1 / Mathlib project.
Its initialization response identified version 0.30.0 and contained an empty
instructions string. `tools/list` advertised exactly the five selected tools.
All five calls completed without an MCP error:

- Goal inspection returned the square-expansion theorem's unsolved goal.
- Diagnostics reported no errors, timeouts or failed dependencies.
- Hover returned the theorem's type.
- Local search found our quadratic-gap declarations.
- Tactic trials closed the goal with `ring` and rejected `rfl` with an explicit
  error and remaining goal.

The [MCP validation receipt](lean-gaussian-certificate/mcp-validation.json)
contains the calls and results. The failed `rfl` trial also produced an
error-recovery `sorryAx` diagnostic in the temporary language-server buffer.
It is a rejected attempt, not an accepted proof. Both archived Lean source hashes
were unchanged afterwards; a subsequent direct check of `ProofTour.lean` again
reported only the standard foundational axioms. The separate
[compiler receipt](lean-gaussian-certificate/validation.json) records completed
proofs and the deliberately false square-expansion example.

This smoke test establishes these calls work with this project; it does not
validate every server feature, certify its resource usage, or measure model
reasoning quality. The temporary process was stopped after the test. `uvx`
cached its Python dependencies; production dependencies, skill installations
and agent configuration were unchanged.

## Assessment of the other recommendations

The attached recommendation links the
[Numina fork](https://github.com/project-numina/lean-lsp-mcp/tree/5c0eddf0a67881aae10589e9c399538f90f1eff6),
but its plain `uvx lean-lsp-mcp` command resolves the upstream distribution.
The inspected fork registers 21 tools, including four Gemini/OpenAI helpers,
and supplies unconditional workflow instructions. Upstream v0.30.0 registers 23
tools before filtering and supplies controls to reduce that exposure. These are
source counts, not measured context-token costs.

Do not add the Lean skill bundles, Theorist Toolbox workflows or a custom
SuperGLM skill for this task. The user wants tools, and the existing research
documents already record the mathematical contracts. The official
[Lean skills repository](https://github.com/leanprover/skills) describes testing
skills against a baseline; its existence does not establish a benefit on our
models and workloads. The [community Lean workflows](https://github.com/cameronfreer/lean4-skills)
and [Theorist Toolbox](https://github.com/morankor/theorist-toolbox) prescribe
additional proof/project workflows. They are optional choices, not prerequisites
for using Lean or its language server.

Use the alphaXiv discovery and raw-paper tools already exposed in this session
before adding another literature-search MCP. A Semantic Scholar integration
would need a concrete capability gap, such as a required citation-graph query.

[SymPy](https://docs.sympy.org/latest/tutorials/intro-tutorial/intro.html) is a useful
ordinary Python package for symbolic expansions, derivatives and exact small
examples. Use it in an isolated research environment when an experiment needs
it; no MCP wrapper or skill is necessary. Numerical exploration and symbolic
checks supply evidence, while a Lean proof checks a precisely stated theorem
under its explicit assumptions. Neither validates an unstated claim about the
floating-point SuperGLM implementation.
