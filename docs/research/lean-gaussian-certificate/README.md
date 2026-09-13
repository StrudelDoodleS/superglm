# A first Lean proof and the Gaussian identities

Verified on 2026-09-13. These proof sources accompany the
[Gaussian error-bound research](../2026-09-13-adaptive-gaussian-error-bounds.md).
The exact statements checked by Lean are narrower than the complete proposed
numerical certificate.

## Reading a proof

This example is in [ProofTour.lean](ProofTour.lean):

```lean
import Mathlib

theorem square_sum (a b : ℝ) :
    (a + b) ^ 2 = a ^ 2 + 2 * a * b + b ^ 2 := by
  ring
```

`import Mathlib` loads the mathematical library. `theorem` introduces a claim
named `square_sum`. The arguments `a b : ℝ` are arbitrary real numbers. The
expression after the colon is the claim. `:= by` starts its proof, and `ring`
constructs a proof by normalizing the polynomial expressions.

Lean checks the resulting proof term. This establishes the statement for every
real `a` and `b` in Lean's mathematical foundations; it does not test a sample
of floating-point values. Tactics help construct proofs, and Lean's kernel
checks the result. The official [tactics chapter](https://lean-lang.org/theorem_proving_in_lean4/Tactics/)
explains that distinction.

The same file includes a scalar version of the Gaussian result:

```lean
theorem scalar_quadratic_gap (h b x xStar : ℝ)
    (stationary : h * xStar = b) :
    (h * x ^ 2 / 2 - b * x) - (h * xStar ^ 2 / 2 - b * xStar) =
      h * (x - xStar) ^ 2 / 2 := by
  rw [← stationary]
  ring
```

Here `stationary` is a required assumption. `rw` rewrites the expression using
that equality, then `ring` proves the remaining algebra. Positive `h` would make
this stationary reference a minimum. The algebraic identity itself does not
need that assumption. The general matrix version appears in `Certificate.lean`.

## Seeing a rejected attempt

[ProofTourInvalid.lean.txt](ProofTourInvalid.lean.txt) deliberately removes the
`2 * a * b` term while keeping `ring` as the proposed proof. Lean exits with
status 1 and leaves this goal:

```text
a b : ℝ
⊢ a * b * 2 + a ^ 2 + b ^ 2 = a ^ 2 + b ^ 2
```

The turnstile `⊢` means "still to prove". In this example the proposed claim is
false: setting `a=b=1` gives `4=2`. In general, a failed tactic only says that
this proof attempt did not establish the claim; a true theorem can need a more
capable proof. The invalid example is stored as text so normal builds pass.

## What the research proof establishes

[Certificate.lean](Certificate.lean) proves these statements for a general
finite index type over the real numbers:

| Theorem | Required assumptions and conclusion |
| --- | --- |
| `SuperGLM.residual_identity` | If `H xStar = b`, then `H (xStar-x) = b-H x`. |
| `SuperGLM.quadratic_gap` | For symmetric `H` and stationary `xStar`, the quadratic objective gap is half the error energy. An arbitrary additive constant cancels. |
| `SuperGLM.residual_energy_of_left_inverse` | Given an exact supplied left inverse `M H = I` and stationarity, error energy equals `r.T M r`. |
| `SuperGLM.quadratic_gap_residual` | Combines symmetry, stationarity and the supplied left inverse to express the objective gap as `r.T M r / 2`. |

The inverse lemmas assume the exact left-inverse property. They do not prove
that a computed inverse has that property. The proof does not establish positive
definiteness, the penalty Schur bound, covariance claims, floating-point rounding,
REML convergence or a correct implementation of the SuperGLM solver.

The checked sources have no proof placeholders or newly declared axioms.
`#print axioms` reports the standard dependencies `propext`, `Classical.choice`
and `Quot.sound`, with no `sorryAx`. These are part of the mathematical
foundations used by these proofs, not additional solver assumptions hidden by
this project.

## Reproduce the checks

This project pins Lean `leanprover/lean4:v4.33.1` and Mathlib commit
`0df444a360eaa60ab8c11dca51a86af692955474`. The manifest pins transitive
dependencies too. Lean reports commit
`819816b2e0a3bf405af45ae5c7af2491d8f5bee6`; Lake reports `5.0.0-src+819816b`.

From this directory on the current machine:

```sh
/home/max/.elan/bin/lake exe cache get
/home/max/.elan/bin/lake build
/home/max/.elan/bin/lake env lean Certificate.lean
/home/max/.elan/bin/lake env lean ProofTour.lean
/home/max/.elan/bin/lake env lean --stdin < ProofTourInvalid.lean.txt
```

The last command is expected to fail with exit status 1; the others should
succeed. Preserve the committed manifest when fetching dependencies. A clean
checkout needs network access and disk space for the prebuilt Mathlib cache.
Local generated `.lake`
files are ignored. This machine reuses the already downloaded cache through an
ignored local symlink; the proof sources and dependency pins do not depend on it.

Elan is installed at `/home/max/.elan`. Shell startup files were preserved.
To use short commands in a shell session, add its `bin` directory to that
session's `PATH`. On another machine use the corresponding `lake` executable
after installing [Lean through the official instructions](https://lean-lang.org/install/manual/).

The [validation receipt](validation.json) records the commands, source hashes,
exit statuses and compiler output from this checkout.
