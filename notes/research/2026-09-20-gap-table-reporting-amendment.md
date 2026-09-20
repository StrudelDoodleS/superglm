# Gap-table reporting and provenance amendment

This amendment changes interpretation and receipt handling after the original
September 19 runs. It changes no fitted model, loss, capacity selection, arm,
data split, numerical tolerance, or cost threshold.

Closure is defined only when the additive smooth model's test loss exceeds
the validation-selected unrestricted boosting model's test loss. A zero or
negative difference gives no positive headroom to close. The summary now
reports that difference, labels the reason, and returns null closure values.
R1 still measures the interaction signal between the two boosting controls.
Its admission does not imply positive smooth-to-boosting headroom.

R3 requires all admitted datasets to pass. A measured failure establishes
false even if other datasets are missing. Without a measured failure, missing
required results leave the rule undecided, represented by null. No admitted
datasets also leaves it undecided. R2 still counts its passing datasets against
the entire eligible population and lists missing results separately.

Future runs use summary schema 2 and an amended protocol identifier. The
launcher saves dataset, arm and source identity before starting the worker.
The worker records its own source identity and actual runtime before fitting,
then checkpoints prepared data and pair selection before the fit. A timeout
before worker startup retains explicitly labelled launcher identity, not an
invented worker runtime. Receipt replacement is atomic, so an interrupted
write leaves the last complete checkpoint readable. The launcher refuses to
overwrite an existing arm receipt; use a fresh output directory.

The original Kaggle and corpus receipts and summaries remain historical
evidence. Any amended summary derived from them belongs in a new directory
with input hashes and the summarizer's identity. Missing historical source
hashes stay missing. Recomputing a summary is not a new fit or evidence that
the Gram repair improved those models.
