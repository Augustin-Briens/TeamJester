# Roadmap — what a *prospective* validation of this pipeline requires

This dataset was analysed retrospectively. To run the same machinery
as a defensible prospective protocol — where new batches arrive and
verdicts have consequences — the following must be frozen, gated, and
unblinded in order. Each step lists inputs → outputs → responsible
party → **pass/stop criteria**.

## Phase 0 — Freeze everything (before any new data)

| Step | Inputs | Outputs | Owner | Pass / stop |
|---|---|---|---|---|
| 0.1 Freeze recipe | `validated_comparison/recipe.json` (vcompare-1.0) | signed recipe file: thresholds, size floors, noise model, classifier params | analyst + reviewer | recipe md5 recorded; **stop** if refit needed — a refit is a new study |
| 0.2 Freeze reference | 17 Batch_3 images + md5s | locked reference manifest | analyst | all md5s match `dataset_manifest.csv` |
| 0.3 Freeze classifier | t_core rule + solidity cut | classifier version string | analyst | versioned in recipe |
| 0.4 Freeze model assumptions | `assumption_table.csv`, `dfn_input_bands.csv` | assumption band file + 17-point sweep grid | analyst + modeler | every unmeasured input has a declared band + source |
| 0.5 Freeze primary comparisons | 9 declared features + pairs | `primary_comparisons.json` | analyst | **no feature may be added after seeing outcomes** |
| 0.6 Freeze decision rules | thresholds for ACCEPT/INVESTIGATE/REJECT + meaningful-difference bars (0.25 Ah / 0.5 µm / 0.5 MPa / 0.01 V) | signed decision-rules doc | analyst + stakeholder | rules exist before outcomes are inspected |
| 0.7 Measured baseline calibration | reference LOO exceedance rate | recorded false-alarm estimate (1/17 ≈ 5.9% [CI 1.0–26.9%] — never "zero") | analyst | calibration computed with reference-only weights + held-out T² |

## Phase 1 — Acquisition gates (each incoming batch)

| Step | Inputs | Outputs | Owner | Pass / stop |
|---|---|---|---|---|
| 1.1 Acquisition metadata | SEM settings, pixel tag, detector, milling record | acquisition record | imaging tech | scale present; **stop→review** if detector/conditions outside supported envelope |
| 1.2 Scale verification | TIFF resolution tags | verified pixel size | pipeline | within tolerance of 0.025 µm/px; else stop |
| 1.3 Specimen identity | sample ID chain, image↔specimen map | preserved provenance | imaging tech | every image traceable to a physical specimen + location; else stop |
| 1.4 Image-quality review | contrast, noise_MAD, sharpness, problem-photo flag | per-image guard report | pipeline | guards inside reference envelope → proceed; **fails → imaging review, NOT a material verdict** |
| 1.5 Coverage adequacy | number of FOVs, sampling locations | FOV plan assessment | reviewer | enough independent regions for the declared test; else the batch is "under-sampled", not "accepted" |

## Phase 2 — Frozen analysis (no peeking)

| Step | Inputs | Outputs | Owner | Pass / stop |
|---|---|---|---|---|
| 2.1 Execute frozen recipe | new images + frozen recipe | per-image features (same 9 + diagnostics) | pipeline | identical code path as recipe; recipe version logged |
| 2.2 Save results BEFORE unblinding | feature table, flags | timestamped results artefact | pipeline | results file sealed before any batch label/outcome shown |
| 2.3 Evaluate at the image unit | sealed features | exact permutation + BH + CIs + MDDs | pipeline | only the frozen comparisons computed |
| 2.4 Verdict machinery | flags + consensus exceedance + T² | verdict under frozen decision rules | pipeline | verdict produced by rule, not by inspection |
| 2.5 Model indicators (if declared) | marker bands + assumption sweep | paired indicator differences vs declared bars | pipeline | differences at matched sweep indices only; absolutes labelled illustrative |
| 2.6 Unblind + record | sealed results ↔ labels | audit record | reviewer | any post-unblinding analysis explicitly marked exploratory |

## Phase 3 — Stop/review triggers

Review instead of verdict whenever:

- acquisition conditions outside the supported envelope (new
  detector, changed contrast, different milling);
- an image-quality guard fails on >1 region — investigate imaging
  first;
- specimen identity or scale is uncertain;
- the batch is outside the feature envelope of the reference in ways
  the frozen comparisons were not designed to test;
- the recipe would need refitting to proceed — that is a *new*
  study, not a verdict.

## Phase 4 — Prioritized follow-up measurements

| Priority | Measurement | What it resolves | What it leaves open |
|---|---|---|---|
| 1 | **Controlled repeat imaging** (same specimens, defined settings, multiple detectors) | separates acquisition drift from material change — the Batch_1 dim-contrast question | only the imaging-vs-material split, not chemistry |
| 2 | **Blind expert annotation** of ambiguous bright objects (multiple raters, blind to batch) | gives the classifier an eval target it currently lacks; quantifies human-machine agreement | experts can also disagree; not chemistry either |
| 3 | **Targeted EDS / elemental mapping** on flagged regions | chemical identity of the disputed bright material — fine Si vs SiOx vs contaminant | one technique, sampling-limited; bulk chemistry still inferred |
| 4 | **Measured thickness + independent porosity** (cross-section metrology, He pycnometry, mercury intrusion) | shrinks the two largest assumed bands (75 µm, +0.15–0.30) | still 2-D/aggregate, not full 3-D pore topology |
| 5 | **3-D imaging (FIB-SEM or X-ray CT)** | through-plane connectivity, true tortuosity | cost + throughput; resolves different length scales |
| 6 | **Electrochemical testing** (coin cells, matched protocols) | grounds the DFN indicators in measured performance | model still needs its own parameterisation validation |

## Standing commitments

- A "zero false alarms" claim is never made — the measured LOO
  exceedance (5.9%, wide CI) is the honest baseline statement.
- "No clear difference" is reported with an MDD, never as
  equivalence.
- New batches outside supported acquisition conditions are flagged
  for review, not forced into a verdict.
- Envelope overlap is similarity, not provenance.
