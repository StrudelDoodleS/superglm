# Fitted interaction plots from the broad trial

The figures show all ten interaction terms in the selected Airfoil,
Concrete and King County models from the
[frozen broad trial](2026-09-14-broad-interaction-trials.md).

- [Airfoil, two terms](figures/2026-09-14-broad-interactions/airfoil.png)
- [Concrete, four terms](figures/2026-09-14-broad-interactions/concrete.png)
- [King County, four terms](figures/2026-09-14-broad-interactions/king-county.png)
- [All figures as a three-page PDF](figures/2026-09-14-broad-interactions/interaction-surfaces.pdf)

Red is a positive contribution and blue a negative contribution within the
jointly fitted model. Main effects, other interactions and the intercept
also contribute to its prediction. A plotted term is not the difference
between this model and a separately fitted additive model. The reported test
MSE reduction belongs to the selected group of terms, not to each panel.

All three models use Gaussian identity links. The plotter evaluates the
saved tensor's `score` in its fitted coefficient coordinates, without
exponentiating or adding a new centering constraint. The contribution is
in the outcome's units: scaled sound-pressure level in dB, concrete strength
in MPa, and sale price in dollars. Housing colour bars show thousands of
dollars. Airfoil displacement thickness is displayed in millimetres.

The axes use original predictor units, reversing the frozen training
preprocessor's centering and scaling. Frequency, thickness, curing age and
living area use logarithmic display axes; the fitted models are unchanged.
The plotting window spans training marginal first-to-99th percentiles.
Building grades are evaluated at integer values. This display window does
not remove observations from a fit or change any test result.

Dots show training observations. Grey masks grid points outside the convex
hull of the observed raw predictor pair. Inside that hull, some combinations
still have few or no nearby observations. This mask is a display aid, not
a density threshold, numerical certificate, confidence interval or causal
identification condition. Panels have separately labelled colour scales.

Airfoil illustrates a sign-changing fitted interaction: the frequency
pattern reverses between thin and thick boundary layers. Concrete has more
localized curvature, including regions with sparse support. These are
descriptions of the fitted terms, without new significance or causal claims.

The [plot script](plot_broad_interaction_surfaces.py) verifies the frozen
production and benchmark source identities, data preparation, and saved
model hashes. Before extracting terms, it exactly reproduces every saved
test prediction for the three models, 5,173 predictions in total. No model
is fitted, selected, or altered for these figures. Ruff check and format
checks pass, and each rendered figure was visually inspected.

The archived receipt's `plot_script_sha256` identifies the
[original plotter at `639f499e`](https://github.com/StrudelDoodleS/superglm/blob/639f499eaec04967de0b7c31090276671144b99f/docs/research/plot_broad_interaction_surfaces.py),
whose SHA-256 is `2698a7fa17e2412eec526268f98a224906b209d02edbed511e70f67891b53aa9`.
It predates the replay and plotting corrections made during PR review. The
current script has a different hash; the archived receipt and figure bytes
remain unchanged.

The [receipt](figures/2026-09-14-broad-interactions/receipt.json) records input
and output hashes, axis bounds, effect ranges and grid shapes. PNG and SVG
figures, raw-axis/contribution NPZ grids and the combined PDF are retained
in the same directory. NPZ contributions are in original outcome units,
before the housing display divisor. Prepare the frozen source checkout and
environment in the [replay instructions](2026-09-14-interaction-review-validation.md#replaying-a-historical-source-tree),
then replay to a separate output directory:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run --project .worktrees/interaction-frozen python notes/research/plot_broad_interaction_surfaces.py \
  --source-root .worktrees/interaction-frozen \
  --run-root /path/to/frozen-20260914 \
  --data-root /path/to/interaction-datasets \
  --output /tmp/replayed-interaction-surfaces
```

The NPZ values reproduce exactly in this environment. Image/container bytes
may vary with rendering metadata; the receipt binds the delivered files.
Without `--output`, the current plotter writes to the ignored
`.benchmark-artifacts/broad-interaction-surfaces-replay/` directory. It refuses
an output inside the tracked frozen figure directory.
