# Numerical refusals in the frozen broad interaction trial

The 2026-09-14 broad trial preserves two Concrete failures for investigation.
Both occur in the one-pair model, `Cement × Age`, at k=4 and k=6. The additive
controls and the two- and four-pair models converge on the same data. The
larger models contain this pair, so these failures do not establish that the
pair lacks predictive value or that its model space is intrinsically invalid.

| Arm | Minimum equilibrated eigenvalue | Maximum absolute eigenvalue | Outcome |
| --- | ---: | ---: | --- |
| `k4_s1` | −2.428e−3 | 5.696 | Materially indefinite data Gram; fit refused |
| `k6_s1` | −9.729e−3 | 7.013 | Materially indefinite data Gram; fit refused |

Values above are rounded as printed by the exceptions. The complete errors,
tracebacks, data/source identities and timing receipts remain under
`.benchmark-artifacts/broad-interactions/frozen-20260914/uci_concrete/` in the
respective arm's `result.json`. Neither failed arm has a validation score,
saved fitted model or test prediction. The completed-fit convergence
telemetry was not reached.

## Where the refusal happens

This is the final full coefficient refit called by
[`reml/discrete.py`](../../src/superglm/reml/discrete.py), after the outer
smoothing optimization. In
[`solvers/irls_direct.py`](../../src/superglm/solvers/irls_direct.py), the call
`decompose_gram_if_authoritative(centered_final.data_gram)` requests data-rank
metadata. The PSD guard in
[`solvers/rank.py`](../../src/superglm/solvers/rank.py) rejects the equilibrated
matrix when

\[
\lambda_{\min} < -\max(100,m)\epsilon\max(\max_i|\lambda_i|,1),
\]

where m is the active matrix order. The exception occurs before the later
observation-factor fallback. That fallback therefore supplied no certificate
for either failed matrix. This matrix is the centered, unpenalized data Gram;
it is not the Hessian with respect to smoothing parameters.

The centered-system builder can project a penalized Hessian while retaining
the original data Gram. An earlier accepted penalized solve consequently
does not certify this unpenalized matrix. These observations identify the
refusal path, not the source of its numerical error.

## What the exact mathematics does and does not establish

For nonnegative weights and an exact centered design X_c,

\[
v^T X_c^T W X_c v = \|W^{1/2}X_c v\|_2^2\geq0.
\]

Thus the intended exact Gram is PSD. If its computed, equilibrated form is
\(\widehat G=G+E\), with symmetric E, the Rayleigh quotient gives
\(\lambda_{\min}(\widehat G)\geq-\|E\|_2\). A useful certificate must bound
the construction error E, including centering and any change of coordinates.
An eigensolver-scale tolerance alone does not supply that construction bound.
No such bound was measured for these two failures. The negative eigenvalues
alone cannot distinguish a construction bug from amplified rounding error.

A follow-up reproduction should capture the failing Gram, weights, centering
vector, final basis coordinates, smoothing parameters and actual construction
path. Compare these with an independently formed weighted-design factor,
using a dimension-, norm- and conditioning-dependent error bound. Any change
to fallback behavior needs a focused regression and a demonstration against
the original failure. This batch changes no production solver or tolerance.

## Other unfinished models

Nonconverged fits are separate from the two matrix refusals. The broad trial
records their coefficient and REML status and excludes them from model
selection. The accompanying generated bilinear integration fixture also
reaches the unchanged 100-iteration REML limit near a curvature-penalty
boundary; its preserved receipt is described in the
[trial protocol](2026-09-14-broad-interaction-trials.md). That fixture checks
exclusion and replay behavior and is not an additional real dataset.

This note is read-only failure triage of source frozen at the trial's
`protocol.json`; its aggregate package-source SHA-256 is
`7151ca3bdf15181144e935c937434a0924d1cce28940b3ef584f3a49a5bfcfb5`.
The trial report and measurement receipt record the complete failure counts.
