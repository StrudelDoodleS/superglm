# Gaussian Newton integration repair

CI exposed two omissions in the Gaussian root-carrier preparation change.
Six public Newton cases evaluated the carrier-free root directly when forming
second curvature differences. An architecture test also rejected a concrete
Gaussian import in the shared model module.

The model now uses the existing exact-type reuse contract. Curvature evaluation
checks complete row shapes and declared storage, then evaluates fresh owned
children of at most 8,192 rows. The eager fallback, root refusal, stencil
formulas and numerical certificates remain unchanged. Children do not retain
the root carrier or repeat full-response validation inside the loop.

Both original failures were reproduced before editing. Five new numerical and
dispatch cases also fail against the unfixed source. The final affected suite
passes **295 tests**, including all seven failing CI nodes. It covers independent
Gaussian curvature formulas, eager equivalence, bounded children, lifetime and
malformed-source refusal. Independent review, Ruff, formatting and targeted
type checks pass.

A serial complete-fit control uses the public 384-row Gaussian frequency-weight
fixture with `discrete=True`, 32 bins, an 8 KiB chunk-selector budget and
`outer="efs+newton"`. The working baseline is
`141497cf`, before root-carrier removal. Three baseline/candidate pairs take
0.283/0.529, 0.287/0.282 and 0.303/0.246 seconds. All fits converge, and all saved
outputs, full smoothing histories, covariances and predictions agree exactly.
Backend and iteration counts also agree in both separate profile runs.

The first pair has peak process RSS of 644/758 MB; the subsequent pairs use
449/449 MB. Public warmup is outside the fit clock. All samples are retained;
these small controls establish no speedup or general memory guarantee.
Profiles retain 250 natural evaluations and 234 likelihood-cache accesses.
The correction creates 13 additional owned children for 13 curvature evaluations;
their cumulative curvature time changes from 6.8 to 10.6 milliseconds.

The [receipt](c1_gaussian_newton_repair_receipt.json) pins both source trees,
the driver and fixture, each measurement, exact comparisons and profile calls.
The earlier 10-million-row measurement uses `outer="efs"` and remains bound to
its original source receipt. This repair adds no out-of-core fitting capability.

## Follow-up review

The [Fable follow-up](https://github.com/StrudelDoodleS/superglm/pull/381#issuecomment-5637374448)
found no correctness defect in the Fisher handoff, Gaussian preparation or
Newton correction. CI serving-model records confirm Fable 5.1 at max effort.
Codex also reported no major issues at `508dd144`. All twelve Python numerical
shards, the aggregate, required real-data, development and security checks pass
on that solver source.

Four optional improvements remain outside this closeout:

1. Reuse observed geometry when an explicit Fisher retry returns the same state.
2. Consider a bounded child cache shared by Newton curvature evaluations, with
   a larger opt-in Newton measurement before claiming a performance benefit.
3. Use the shared derived-field contract for the temporary null-plan rebind
   before extending derived preparation to another family.
4. Release the temporary null-model owner earlier to shorten its carrier's
   overlap with final row copies.

These are recorded limits and follow-ups, not additional C1 acceptance gates.
The ordinary Opus 5 workflow is restored after the temporary review.
