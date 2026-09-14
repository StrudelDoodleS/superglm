# Interaction dataset corpus

The core corpus contains 34 validated tables representing 32 independent real
sources. It includes 31 new downloads: 30 real sources and one CASP structural-
decoy control. Three existing local tables contribute two additional real
sources, California housing and freMTPL2. Frequency and severity belong to the
same insurance book. The decoy control is available but excluded from real-source
counts.

The downloads total 290,258,438 bytes, or 290.3 MB. This fits the expanded 300 MiB
payload budget. No core download is failed, pending or merely catalogued.
The [tracked receipt](2026-09-13-interaction-dataset-corpus-receipt.json) preserves
the full scan, file hashes, ordered columns, observed dtypes, missing counts and
target ranges. The [registry](../../benchmarks/interaction_datasets.json) records
exact URLs, publisher metadata snapshots, source versions, byte limits, predictor
roles and proposed split policies. Dataset readiness is a property of a receipt,
not a promise made by the registry.

The registry is a frozen input: its whole-file SHA-256 is recorded in this
receipt and the broad-trial data identities. Its `related_research` value
therefore keeps the original `docs/research` path after the documentation move.
Use the current Kaggle-corpus link below; changing that metadata inside the
registry would change the pinned input bytes.

The separate [Kaggle corpus](2026-09-13-interaction-kaggle-corpus.md) and
[registry](../../benchmarks/interaction_kaggle_datasets.json) add five verified real
sources and record unavailable competitions, including Allstate, separately.
Together, the two receipts cover 37 independent real sources and 38 real tables,
or 39 tables including the decoy control. Kaggle mirrors of an already counted
source, alternate targets and repeated worktree copies would not increase this
count. The [scaling design](2026-09-13-real-data-scaling-design.md) distinguishes
row count, raw predictors and expanded model coefficients.

All rows below are observed counts for the pinned artifacts. The predictor
count d excludes the selected target, identifiers, related outcomes and the
other exclusions in the registry. It precedes categorical encoding, spline
expansion or interaction construction. A larger d does not itself establish a
large fitted coefficient count or a performance result.

| Source | Rows | d | Primary target | Split rule or preparation constraint |
| --- | ---: | ---: | --- | --- |
| [Abalone](https://archive.ics.uci.edu/dataset/1/abalone) | 4,177 | 8 | `Rings` | Group duplicate profiles; random control |
| [Auto MPG](https://archive.ics.uci.edu/dataset/9/auto+mpg) | 398 | 7 | `mpg` | Latest model years; group car names |
| [Automobile](https://archive.ics.uci.edu/dataset/10/automobile) | 205 | 23 | `price` | Manufacturer groups; 4 missing prices |
| [Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) | 569 | 30 | `Diagnosis` | Stratified by diagnosis; group ID |
| [Computer Hardware](https://archive.ics.uci.edu/dataset/29/computer+hardware) | 209 | 6 | `PRP` | Vendor groups; exclude ERP |
| [Statlog (German Credit Data)](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) | 1,000 | 20 | `class` | Stratified, duplicate-profile groups |
| [Forest Fires](https://archive.ics.uci.edu/dataset/162/forest+fires) | 517 | 12 | `area` | Spatial grid-cell groups |
| [Concrete Compressive Strength](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength) | 1,030 | 8 | `Concrete compressive strength` | Mixture groups across curing ages |
| [Communities and Crime](https://archive.ics.uci.edu/dataset/183/communities+and+crime) | 1,994 | 118 | `ViolentCrimesPerPop` | State groups; missing predictors retained |
| [Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality) | 6,497 | 12 | `quality` | Duplicate-profile groups; both colors count once |
| [Parkinsons Telemonitoring](https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring) | 5,875 | 19 | `total_UPDRS` | Subject groups across recordings |
| [Yacht Hydrodynamics](https://archive.ics.uci.edu/dataset/243/yacht+hydrodynamics) | 308 | 6 | `residuary_resistance` | Hull groups across speeds |
| [Physicochemical Properties of Protein Tertiary Structure; decoy control](https://archive.ics.uci.edu/dataset/265/physicochemical+properties+of+protein+tertiary+structure) | 45,730 | 9 | `RMSD` | Exploratory only; protein IDs absent |
| [Bike Sharing](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) | 17,379 | 12 | `cnt` | Chronological whole days |
| [Airfoil Self-Noise](https://archive.ics.uci.edu/dataset/291/airfoil+self+noise) | 1,503 | 5 | `scaled-sound-pressure` | Tunnel-configuration groups |
| [Combined Cycle Power Plant](https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant) | 9,568 | 4 | `PE` | Duplicate-profile groups; timestamps absent |
| [Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance) | 649 | 30 | `G3` | Outer school holdout; Portuguese table |
| [Online News Popularity](https://archive.ics.uci.edu/dataset/332/online+news+popularity) | 39,644 | 46 | `shares` | Chronological publication days |
| [Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) | 30,000 | 23 | `Y` | Stratified by default; group ID |
| [Air Quality](https://archive.ics.uci.edu/dataset/360/air+quality) | 9,357 | 8 | `CO(GT)` | Chronological days; -200 sentinel handled |
| [Appliances Energy Prediction](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction) | 19,735 | 24 | `Appliances` | Chronological days in one building |
| [Beijing PM2.5](https://archive.ics.uci.edu/dataset/381/beijing+pm2+5+data) | 43,824 | 11 | `pm2.5` | 2010-12 train, 2013 validation, 2014 test |
| [SGEMM GPU kernel performance](https://archive.ics.uci.edu/dataset/440/sgemm+gpu+kernel+performance) | 241,600 | 14 | `Run1 (ms)` | Tiling-block groups; other runtimes excluded |
| [Superconductivty Data](https://archive.ics.uci.edu/dataset/464/superconductivty+data) | 21,263 | 81 | `critical_temp` | Chemical-formula groups |
| [Real Estate Valuation](https://archive.ics.uci.edu/dataset/477/real+estate+valuation+data+set) | 414 | 6 | `Y house price of unit area` | Later months; coordinate-group control |
| [Metro Interstate Traffic Volume](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume) | 48,204 | 7 | `traffic_volume` | Chronological days; repeated hours stay together |
| [Seoul Bike Sharing Demand](https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand) | 8,760 | 11 | `Rented Bike Count` | Chronological days; operating-hours estimand |
| [Ames Housing](https://jse.amstat.org/v19n3/decock.pdf) | 2,930 | 74 | `SalePrice` | 2006-08 train, 2009 validation, 2010 test |
| [Relative location of CT slices on axial axis](https://archive.ics.uci.edu/dataset/206/relative+location+of+ct+slices+on+axial+axis) | 53,500 | 384 | `reference` | Patient groups across slices |
| [BlogFeedback](https://archive.ics.uci.edu/dataset/304/blogfeedback) | 60,021 | 280 | `comments_next_24h` | Original train; February/March 2012 validation/test |
| [Year Prediction MSD](https://archive.ics.uci.edu/dataset/203/yearpredictionmsd) | 515,345 | 90 | `year` | Original 463,715/51,630 artist-separated split |
| [California Housing](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset) | 20,640 | 8 | `MedHouseVal` | Existing random control plus spatial holdout |
| [freMTPL2 frequency](https://www.openml.org/d/41214) | 678,013 | 9 | `ClaimNb` | Policy groups; exposure offset |
| [freMTPL2 severity](https://www.openml.org/d/41215) | 26,639 | join | `ClaimAmount` | Policy groups; join frequency features |

The split rules are research choices except where the source explicitly supplies
an outer split. They define the intended generalization question before model
selection. They are not automatically executed by the data loader.

The wider complete tables now include CT slices with 384 predictors, BlogFeedback
with 280, Communities and Crime with 118 after conservative exclusions,
YearPredictionMSD with 90 and superconductivity with 81. The largest core table
has 678,013 insurance policies. SGEMM contains 241,600 measured hardware
configurations. These are source dimensions, not evidence that an interaction
fit is affordable at every resulting coefficient count.

Several details materially change a future comparison:

- Bike sharing has 17,379 rows, while the current UCI metadata says 17,389.
  `casual + registered` reconstructs the count target; neither enters the
  predictors. The downloaded normalized CSV and its exact hash are pinned.
- Online News has 39,644 rows, compared with 39,797 in current metadata.
  The 46-predictor default omits the URL, acquisition-relative age and all
  keyword/self-reference share summaries. Those summaries require a separate
  prediction-time availability audit before admission.
- The Wine Quality CSV includes 6,497 red and white observations and a color
  column. UCI metadata reports the 4,898-row white subset. Both colors form one
  corpus here. Student Performance uses the complete 649-row Portuguese table;
  the mathematics subject and G1/G2/G3 are not independent sources. G1 and G2 are
  excluded from the final-grade baseline.
- Seoul's API labels `Functioning Day` as its target. The demand task instead
  uses `Rented Bike Count`. Closed hours form structural zeros. The registry
  declares an operating-hours analysis after chronological partitioning, with
  an all-hours analysis requiring a separate estimand.
- The Air Quality file contains 9,357 parsed rows, compared with 9,358 in
  metadata. Its documented -200 sentinel becomes missing through explicit CSV
  options. There are 1,683 missing CO targets. Beijing has 2,067 missing PM2.5
  targets, and Automobile has four missing prices. The loader retains these
  rows; an adapter must assign partitions before removing unlabeled cases.
- Metro's literal `None` holiday label is a valid category. Ames also uses
  absence tokens in categorical fields. Explicit parser options preserve these
  tokens. Ames numeric NA tokens remain missing. Its 2,930 PIDs are unique, so
  the proposed year split yields 1,941/648/341 train/validation/test rows without
  cross-year property duplicates. The physical-property baseline excludes sale
  date, sale condition/type and miscellaneous monetary valuation fields.
- Parkinson recordings stay with their subject, CT slices stay with their
  patient, and superconductivity measurements stay with their chemical formula.
  The superconductivity ZIP supplies formula labels in `unique_m.csv`; the
  loader checks row counts and exact target alignment before attaching them.
- BlogFeedback includes the complete training file and all 60 supplied daily
  test files. The loader attaches `__source_file` as split metadata and excludes
  it from predictors. Training windows overlap, so randomly repartitioning that
  file is unsuitable. February and March 2012 supply separate validation and
  test periods. YearPredictionMSD preserves the publisher's first 463,715 versus
  last 51,630 rows. Artist IDs for a similarly certified inner validation split
  are not available.

The existing distributional fixture directory was absent from this worktree.
One older copy at `task6-diag-653/tests/fixtures/distributional_datasets/` has an
explicit README stating that every row is hand-written. Its eight- or nine-row
files mimic freMTPL2, the pricing game, SGEMM and YearPredictionMSD schemas. They
contribute no real observations and are not real-data subsets. Other worktree
copies of the same fixtures were not counted. No genuine local subset was
counted as a complete source. The UCI Servo source was also excluded because its
contributor describes a simulation.

Existing loaders mostly cover freMTPL2 and California housing.
`tests/_datasets.py` searches configured data directories;
`scripts/fetch_fremtpl.py` and `benchmarks/fetch_mtpl2.py` provide the insurance
fetch paths. `benchmarks/benchmark_housing_tensor.py` uses a complete California
frame and a content-fingerprint guard. The new registry references the existing
full parquet files in place and pins their observed byte identities. freMTPL2
severity supplies only policy ID and claim amount; modelling it requires a
many-to-one join to policy features after policy-level split assignment.
[OpenML frequency metadata](https://api.openml.org/api/v1/json/data/41214) and
[severity metadata](https://api.openml.org/api/v1/json/data/41215), checked on
2026-09-13, identify both as version 1 under CC0.

Every fetched UCI source page explicitly supplied CC BY 4.0 at the audit date.
Creator names and DOIs are retained in the registry for attribution. Metadata
JSON and source-page bytes are archived under the ignored source-audit directory,
and their SHA256 values are recorded. Ames comes directly from Dean De Cock's
[publisher supplement](https://jse.amstat.org/v19n3/decock/AmesHousing.txt) and
[paper](https://jse.amstat.org/v19n3/decock.pdf). No explicit dataset license was
located there; the registry records that limit instead of substituting a mirror's
license. The California entry records the sklearn/StatLib provenance and does
not assert a newly verified dataset license. No raw data is committed.

The fetcher uses only the existing environment and standard-library download
and archive tools. It enforces exact file sizes and SHA256 pins, bounded socket
and total download time, per-source byte limits and an aggregate download budget.
ZIP members are read directly without filesystem extraction and have expanded
size caps. Downloads and receipts are published atomically. Existing files are
verified or rejected; a corrupt file is preserved and never silently replaced.
Missing files, malformed tables, target violations and failed downloads produce
failure records and a nonzero CLI exit. An explicitly requested catalogue-only
entry cannot be reported as ready.

From this worktree:

```bash
uv run python benchmarks/interaction_datasets.py list
uv run python benchmarks/interaction_datasets.py fetch
uv run python benchmarks/interaction_datasets.py validate
uv run python benchmarks/interaction_datasets.py validate uci_breast_cancer uci_credit_default ames_housing uci_bike_sharing
```

The loader is `load_dataset(id, root=..., manifest=...)` in
[interaction_datasets.py](../../benchmarks/interaction_datasets.py). It returns a
verified pandas frame and accepts an alternate registry, including the Kaggle
registry. It does not encode categories, impute values, choose a family, transform
the target, select interactions or implement the split. Each receipt explicitly
sets `adapter_ready` to false. Fit imputation, category vocabularies, knot choices
and any feature selection on the training partition only. Preserve the declared
IDs, dates and source-file labels until partitions are fixed. The severity join,
Ames absence semantics, repeated Metro hours and structural-zero policies remain
adapter responsibilities.

The initial full scan returned 34/34 ready and zero failures in
`.benchmark-artifacts/interaction-datasets/receipts/20260913T211943-c918db1c.json`.
Its full contents are preserved in the tracked receipt. After that scan, Ruff
formatting and a guard requiring explicit categorical declarations for string
predictors changed the utility hash. The tracked record includes both hashes;
all 34 stored dtype schemas satisfy the added guard. A final full scan under the
current utility returned 34/34 ready with zero failures in
`.benchmark-artifacts/interaction-datasets/receipts/20260913T213337-ae0767e8.json`.
Its data records are exactly equal to the initial scan. The tracked receipt
preserves this final validation identity and references the shared data records.
The final 18 focused tests, Ruff checks and formatting checks pass. Tests cover changed bytes of equal length,
truncation, byte-budget refusal, target support, missing files, preserved ZIP
partitions, companion misalignment, in-place local references and independent
source counting. Removing the SHA256 comparison in an in-memory mutation made
the equal-length corruption regression fail; the unchanged implementation passes
all 18 tests. No production fit or production dependency/version change was made
for this corpus task.
