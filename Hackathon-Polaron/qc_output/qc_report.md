# Polaron QC Report — SUPERSEDED VERDICTS

> **Status (acquisition analysis, 2026-10-04):** the verdicts in this
> report predate the acquisition-group analysis. The batch
> "differences" they rest on are count-type markers that the variance
> partition shows track the imaging session, not the batch. Current
> position — see `AGENTS.md` "Current state" and
> `paper/micro2dfn_paper.pdf`: composition indistinguishable, all
> count-based batch claims "can't tell" until session-controlled
> re-imaging. Kept for provenance.

## The question

Did the supplier's material change before it becomes a manufacturing problem? Each incoming batch is photographed by electron microscope, every pixel is sorted into **pore / graphite bulk / silicon**, ~40 physical measurements are taken per image, and the batch is compared against the approved baseline (**Batch_3**) using the baseline's own natural variation as the ruler.

## What the verdicts mean

| Verdict | Meaning | Action |
|---|---|---|
| **ACCEPT** | Indistinguishable from the approved batch within measurement noise | Release to production |
| **INVESTIGATE** | Something moved but the evidence is partial or not uniquely attributable | Sample more, ask the supplier |
| **REJECT** | Changed beyond anything the baseline ever produced, in a pattern matching a known defect | Quarantine + root-cause. *Not* 'the battery will fail' — 'do not release without an explanation' |

## Verdict summary

| Batch | Verdict | Driving signature | Anomalous regions | T² p |
|---|---|---|---|---|
| Batch_1 | **REJECT** | E1 Excessive silicon content | 3/7 | 0.0000 |
| Batch_2 | **INVESTIGATE** | - | 0/7 | 0.1875 |

Baseline: **17 images** · frozen multi-Otsu thresholds t_pore=0.7109, t_si=1.3354 (fitted on baseline only, applied identically to every image — the ruler never moves) · pixel = 25 nm

Calibration (leave-one-image-out on baseline): worst honest self-score **3.36** → flag ≥ 2.73, investigate ≥ 2.20, reject ≥ 3.30

![fingerprint scores](fig_fingerprints.png)

![per-region anomaly scores](fig_image_scores.png)

_Dots above the dashed line are regions more anomalous than anything the approved batch produced against itself._


---

## Batch_1 — **REJECT**

**Why this verdict:**
- region img_4ih2ggld exceeds the baseline envelope (E1, 2.74) and 2 further regions share the E1 signature above the baseline 90th percentile — corroborated heterogeneous defect (consensus p≈0.026)
- joint multivariate drift unusual at p=0.0000 — no single dominant fingerprint but the batch is off-nominal as a whole

Multivariate drift (T2) bootstrap p-value: **0.0000**

**Regions outside the baseline envelope:**
| Image | Top signature | Score |
|---|---|---|
| img_4ih2ggld | E1 | 2.74 |

**Near-envelope regions (above baseline 90th percentile, below strict flag line):**
| Image | Top signature | Score |
|---|---|---|
| img_5n1q8atc | E1 | 2.43 |
| img_f1vzngrs | E1 | 2.42 |

![raw comparison](fig_raw_compare_Batch_1.png)

_Same microscope images at equal contrast — the flagged regions visibly carry more bright silicon._


![img_4ih2ggld](seg_Batch_1_img_4ih2ggld.png)

![budget](fig_budget_Batch_1_img_4ih2ggld.png)

![img_5n1q8atc](seg_Batch_1_img_5n1q8atc.png)

![budget](fig_budget_Batch_1_img_5n1q8atc.png)

![img_f1vzngrs](seg_Batch_1_img_f1vzngrs.png)

![budget](fig_budget_Batch_1_img_f1vzngrs.png)

**Primary signature (consensus of anomalous regions):** E1 — Excessive silicon content

- Likely cause: Formulation / dosing error
- Mechanism: More Si than the pore space can absorb at full charge -> electrode thickens, SEI grows.
- Expected consequence: Swelling, SEI growth, dead Si

**Fingerprint evidence (z vs baseline):**
- Silicon fraction: z=+3.9 expected up  *(matches)*
- Swelling budget (P01): z=-3.4 expected down  *(matches)*
- Area share with negative swelling budget: z=+0.7 expected up

**Also plausible:** E7 Manufacturing tears / handling damage (0.60)

> ⚠ **Contrast caveat:** the flagged regions also run dimmer silicon (Si:bulk contrast z=-4.1). If this is a real grade change (e.g. finer/oxidised Si), the extra bright material is real silicon and E1 stands — the independent micro2dfn object-classifier adjudication supports this reading (see reconcile_output/adjudication.csv). If it is an acquisition-contrast change, part of the silicon increase could be over-counted; either way the batch is non-uniform and stays quarantined.

**Significant feature shifts (Benjamini–Hochberg, FDR=0.1):**

| Feature | Baseline median | Batch median | z | p | bigger than noise? |
|---|---|---|---|---|---|
| Pore count density (per Mpx) | 108.3 | 142.8 | +3.6 | 0.0002 | — |

**Watch list** — raw p<0.05 but not significant after multiplicity control (reported honestly, not claimed):
- Si/bulk BSE contrast ratio: z=-1.7, p=0.007
- Si D90, area-weighted (um): z=-1.2, p=0.024
- Si-to-pore contact share: z=+1.8, p=0.010
- Vertical/oblique crack fraction: z=-1.4, p=0.037
- InLens crack area fraction: z=-2.6, p=0.010
- Si contact out-of-band share: z=-1.0, p=0.029
- Fully enclosed Si share: z=-0.9, p=0.045

> ⚠ **InLens caveat:** this channel measures surface texture and is sensitive to imaging conditions (focus, contrast, drift). Verify acquisition settings before treating an InLens shift as a material change.

**Predicted consequence bands** (risk proxies from cited physics — not capacity predictions; computed on the **anomalous regions (n=3)**):

| Risk | What it means | Baseline | This batch | Direction | Source |
|---|---|---|---|---|---|
| swelling interval | Breathing room left after silicon swells. Range brackets worst case (pure Si, ~3x) to mild case (nanoporous Si, ~1x). Negative = electrode must thicken or crack on charge. | [-0.19 .. -0.03], 98% of area in deficit | [-0.32 .. -0.07], 100% of area in deficit | higher risk | M3 Obrovac/Chevrier; R8 Lu 2025 |
| Si surface / SEI proxy | Silicon edge length per unit silicon area + share of Si in large (cracking-prone) particles. More surface and a fatter tail = more SEI growth and lithium loss on cycling. | 4.01 um-1, big-Si share 47% | 6.09 um-1, big-Si share 15% | higher risk | R1 Liu 2012; R4 Müller 2018; R17 Son 2018 |
| ion-detour / plating proxy | How much longer the real pore path is vs a straight line, and how flat pores are. Bigger detours concentrate current -> lithium plating risk at fast charge. | tau=1.80, pore flatness=1.19 | tau=1.42 (0.78x baseline), pore flatness=1.11 | unchanged | M1 Ebner 2014; M2 Billaud 2016; R10 Lu 2023 |
| dead-Si penalty | Share of silicon sealed inside graphite with no pore nearby — material that can't charge (or swells with nowhere to go). | 87% enclosed, 0.63 um to nearest pore | 76% enclosed, 0.62 um to nearest pore | unchanged | R8 Lu 2025 |
| pre-existing damage | Tears already in the material act as failure-starting points. Vertical/oblique cracks and InLens surface cracks are the reliable signatures. | vert cracks 0.0132, InLens density 0.9323 | vert cracks 0.0114, InLens density 1.0219 | unchanged | dictionary E7; polishing-artefact caveat applies |

![distributions](fig_dist_Batch_1.png)


---

## Batch_2 — **INVESTIGATE**

**Why this verdict:**
- 2 BH-significant feature shifts (bright_aspect, inlens_crack_frac) — real movement, but not matching a single known-defect pattern

Multivariate drift (T2) bootstrap p-value: **0.1875**

No effect fingerprint matched above the calibrated threshold.

**Significant feature shifts (Benjamini–Hochberg, FDR=0.1):**

| Feature | Baseline median | Batch median | z | p | bigger than noise? |
|---|---|---|---|---|---|
| Si particle elongation | 1.912 | 2.01 | +1.3 | 0.0037 | — |
| InLens crack area fraction | 0.2216 | 0.1981 | -3.2 | 0.0009 | — |

**Watch list** — raw p<0.05 but not significant after multiplicity control (reported honestly, not claimed):
- Si-to-pore contact share: z=+1.5, p=0.018
- Pore count density (per Mpx): z=+1.3, p=0.010

> ⚠ **InLens caveat:** this channel measures surface texture and is sensitive to imaging conditions (focus, contrast, drift). Verify acquisition settings before treating an InLens shift as a material change.

**Predicted consequence bands** (risk proxies from cited physics — not capacity predictions; computed on the **whole batch**):

| Risk | What it means | Baseline | This batch | Direction | Source |
|---|---|---|---|---|---|
| swelling interval | Breathing room left after silicon swells. Range brackets worst case (pure Si, ~3x) to mild case (nanoporous Si, ~1x). Negative = electrode must thicken or crack on charge. | [-0.19 .. -0.03], 98% of area in deficit | [-0.21 .. -0.03], 98% of area in deficit | unchanged | M3 Obrovac/Chevrier; R8 Lu 2025 |
| Si surface / SEI proxy | Silicon edge length per unit silicon area + share of Si in large (cracking-prone) particles. More surface and a fatter tail = more SEI growth and lithium loss on cycling. | 4.01 um-1, big-Si share 47% | 5.04 um-1, big-Si share 44% | unchanged | R1 Liu 2012; R4 Müller 2018; R17 Son 2018 |
| ion-detour / plating proxy | How much longer the real pore path is vs a straight line, and how flat pores are. Bigger detours concentrate current -> lithium plating risk at fast charge. | tau=1.80, pore flatness=1.19 | tau=2.06 (1.14x baseline), pore flatness=1.15 | unchanged | M1 Ebner 2014; M2 Billaud 2016; R10 Lu 2023 |
| dead-Si penalty | Share of silicon sealed inside graphite with no pore nearby — material that can't charge (or swells with nowhere to go). | 87% enclosed, 0.63 um to nearest pore | 80% enclosed, 0.56 um to nearest pore | unchanged | R8 Lu 2025 |
| pre-existing damage | Tears already in the material act as failure-starting points. Vertical/oblique cracks and InLens surface cracks are the reliable signatures. | vert cracks 0.0132, InLens density 0.9323 | vert cracks 0.0155, InLens density 0.9012 | unchanged | dictionary E7; polishing-artefact caveat applies |

![distributions](fig_dist_Batch_2.png)


---

## Feature scorecard — which metrics earned their keep

![scorecard](fig_scorecard.png)

Every metric is graded: how much it wobbles on approved material (stability), whether it moved when something was wrong (discrimination), and whether it duplicates a stronger metric (redundancy). Weights used in fingerprint scoring penalise noisy/redundant metrics — and are computed **without** looking at the batch under test.

| Metric | Tier | baseline spread | max batch |z| | max image |z| | weight |
|---|---|---|---|---|---|
| Pore count density (per Mpx) | decisive | 0.1297 | 3.61 | 8.87 | 1.0 |
| InLens crack area fraction | decisive | 0.0387 | 3.19 | 3.62 | 1.0 |
| Si area share >3 um | decisive | 0.0731 | 1.76 | 10.46 | 1.0 |
| Si/bulk BSE contrast ratio | decisive | 0.0481 | 1.68 | 4.61 | 1.0 |
| Pore verticality | decisive | 0.0269 | 1.48 | 4.16 | 1.0 |
| Vertical/oblique crack fraction | supporting | 0.1503 | 1.44 | 1.93 | 1.0 |
| Silicon fraction | decisive | 0.1409 | 1.33 | 4.44 | 1.0 |
| Mean Si diameter (um) | decisive | 0.0919 | 1.32 | 2.68 | 1.0 |
| Si particle elongation | supporting | 0.0379 | 1.31 | 1.47 | 1.0 |
| Si D90, area-weighted (um) | decisive | 0.1479 | 1.23 | 3.63 | 1.0 |
| Si particle density (per Mpx) | decisive | 0.3465 | 1.19 | 7.59 | 1.0 |
| Swelling budget, worst 5% window | decisive | 0.1358 | 1.06 | 4.24 | 1.0 |
| InLens vertical/oblique crack share | supporting | 0.1097 | 1.04 | 1.8 | 1.0 |
| Swelling budget (P01) | decisive | 0.1468 | 1.03 | 4.0 | 1.0 |
| Si area share >5 um | decisive | 0.3768 | 1.0 | 2.65 | 1.0 |
| InLens crack length density (um-1) | supporting | 0.0757 | 0.85 | 2.48 | 1.0 |
| Pore horizontal chord (um) | decisive | 0.1051 | 0.76 | 3.03 | 1.0 |
| Si-to-pore distance, mean (um) | supporting | 0.1673 | 0.75 | 1.91 | 1.0 |
| Si clustering index (Clark-Evans R) | supporting | 0.0731 | 0.72 | 2.26 | 1.0 |
| Bulk/flake alignment (h/v) | decisive | 0.0411 | 0.68 | 2.97 | 1.0 |
| Largest Si particle (um) | supporting | 0.1629 | 0.56 | 2.27 | 1.0 |
| Pore-solid boundary density (um-1) | supporting | 0.1017 | 0.46 | 1.18 | 1.0 |
| Pore verticality (area-weighted) | supporting | 0.0925 | 0.38 | 1.4 | 1.0 |
| Porosity | supporting | 0.1961 | 0.36 | 1.56 | 1.0 |
| 2D tortuosity proxy (through-plane) | supporting | 0.2555 | 0.26 | 1.34 | 1.0 |
| Horizontal crack fraction | supporting | 0.3121 | 0.25 | 1.97 | 1.0 |
| Si patchiness (strip CV) | supporting | 0.2767 | 0.15 | 2.01 | 1.0 |
| Bulk (graphite+binder) fraction | decisive | 0.0214 | 0.14 | 3.14 | 1.0 |
| Pore correlation length (um) | decisive | 0.1235 | 0.13 | 2.7 | 1.0 |
| Si-to-pore contact share | moved-but-noisy | 0.6046 | 1.76 | 3.86 | 0.3 |


---

## Segmentation health

- **Frozen vs floating thresholds:** per-image Otsu (the independent recipe) was also run as a diagnostic — see `bright_frac_drift_diag` in the Diagnostics data. A large positive drift means a per-image ruler would have absorbed real silicon signal into its moving thresholds.
- **Instrument drift:** raw-intensity signatures of every image were compared against the baseline band (warnings above if any image breached it).
- **Uncertainty:** every claimed shift is annotated against tile error bars and threshold-sensitivity ranges where available.


---

![z heatmap](fig_z_heatmap.png)


## What we cannot tell from these images

2-D BSE sections cannot see: binder migration, true 3-D pore connectivity (electrolyte starvation), SEI itself, lithium inventory, or exact capacity/cycle-life outcomes. Crack metrics may partly reflect polishing artefacts — vertical/oblique cracks are the more reliable signature. The consequence bands above are directional risk proxies, not battery-life predictions.


---

## Appendix

**Baseline self-test (leave-one-image-out) top scores:** 0.55, 0.61, 0.62, 0.63, 0.89, 0.97, 1.13, 1.21, 1.29, 1.46, 1.48, 1.48, 1.56, 1.86, 1.95, 2.57, 3.36 — all below the flag threshold 2.73, i.e. zero false alarms on known-good material.

**Metric glossary:**

| Metric | What it means (plain terms) | Unit | Used in |
|---|---|---|---|
| Large-void fraction | Share of the image taken by very large holes — torn-out or delaminated regions rather than normal pores. | fraction | E7 |
| Si particle elongation | How elongated silicon particles are on average — round is normal, stretched shapes hint at smearing or agglomerates. | ratio | - |
| Si-bulk boundary density (um-1) | Total silicon/graphite contact length per unit area — the interface where expansion stress concentrates. | um-1 | E6 |
| Si/bulk BSE contrast ratio | How bright the silicon signal is relative to the grey body. A contrast-quality check — helps separate a real composition change from a microscope-settings change. | ratio | - |
| Si clustering index (Clark-Evans R) | Whether silicon clumps together (below 1 = clustered, 1 = evenly spread, above 1 = suspiciously uniform). A clump swells like one giant particle. | ratio | E3 |
| Si correlation length (um) | Size scale over which silicon distribution repeats — another view of clumping. | um | - |
| Si size spread (CV) | Spread of silicon particle sizes relative to the mean — a wide spread means inconsistent milling or grading. | ratio | E2 |
| Si D50, area-weighted (um) | Median silicon particle size (weighted by area). | um | - |
| Si D90, area-weighted (um) | 90th-percentile silicon particle size — the size of the biggest 10% of silicon mass, i.e. the risky tail. | um | E2 |
| Largest Si particle (um) | Largest silicon chunk in the image — the single most dangerous particle, since the biggest one cracks first. | um | - |
| Mean Si diameter (um) | Average silicon particle size. Oversized silicon chunks are the ones that shatter on the first charge (literature: fracture risk grows with size). | um | E2 |
| Silicon fraction | Share of the image that is silicon. Silicon stores lots of energy but swells ~3x when charging — too much and the electrode physically tears itself apart. | fraction | E1 |
| Si nearest-neighbour dist (um) | Typical distance from each silicon particle to its nearest neighbour. | um | - |
| Si patchiness (strip CV) | How unevenly silicon is spread across the image — patchy distribution means the slurry was mixed unevenly. | CV | E3 |
| Si particle density (per Mpx) | How many separate silicon particles per million pixels — more smaller pieces vs fewer big chunks. | count/Mpx | - |
| Si-to-pore contact share | Share of silicon edges that sit close enough to a pore to expand into it — silicon with no nearby pore has nowhere to go. | 0-1 | E6 |
| Si particle solidity | How solid and regular silicon particles are (1 = smooth blob, lower = jagged or cracked-looking shapes). | 0-1 | E2 |
| Bulk/flake alignment (h/v) | How much wider graphite flakes are than tall — the flake orientation that drives ion detours when too extreme. | ratio | E4, E5 |
| Bulk (graphite+binder) fraction | Share of the image that is the graphite/binder body — the main framework everything else sits in. | fraction | - |
| Crack fraction | Total share of pores shaped like cracks (long and thin) rather than round holes. | fraction | - |
| Horizontal crack fraction | Share of cracks lying flat along the electrode — these separate layers and suggest crushing or drying stress. | fraction | E4 |
| Vertical/oblique crack fraction | Share of cracks running across the electrode thickness — the more reliable sign of tearing, since polishing rarely makes these. | fraction | E7 |
| InLens crack length density (um-1) | Length of fine surface cracks per unit area, seen in the surface-sensitive channel — tears the chemistry image can't resolve. | um-1 | E4, E7 |
| InLens crack area fraction | Share of solid material covered by fine cracks in the surface channel. | fraction | - |
| InLens vertical/oblique crack share | Share of fine cracks running across the electrode thickness — the orientation linked to real tearing. | 0-1 | E7 |
| Pore anisotropy (h/v) | How much flatter pores are than they are tall (1 = round). Flat pores force lithium ions to take detours — slower charging, plating risk. | ratio | E4, E5 |
| Pore horizontal chord (um) | Typical width of pores measured horizontally. | um | - |
| Pore vertical chord (um) | Typical height of pores measured vertically (the direction ions must travel to reach the current collector). | um | - |
| Pore correlation length (um) | Size scale over which pore structure repeats — coarse = blocky morphology, fine = well-dispersed. | um | - |
| Porosity | Share of the image that is empty space. Pores are the electrode's breathing room — they hold electrolyte and give silicon somewhere to swell into. | fraction | E4 |
| Pore-solid boundary density (um-1) | Total length of pore walls per unit area — the internal surface exposed to electrolyte. | um-1 | - |
| Pore phase connected to top edge | Share of pore space actually connected to the electrode surface — disconnected pores can't carry electrolyte. | 0-1 | - |
| Pore verticality | Average 'uprightness' of pores — higher means pores point toward the current collector, good for ion flow. | 0-1 | - |
| Pore verticality (area-weighted) | Same as pore uprightness but weighted by pore size — dominated by the big transport channels. | 0-1 | E5 |
| Pore count density (per Mpx) | How many separate pores per million pixels — fragmentation of the pore network. | count/Mpx | E7 |
| Round pore fraction | Share of pores that are roundish — the healthy shape expected from the manufacturing process. | fraction | - |
| Si area share >3 um | Share of silicon mass sitting in particles bigger than 3 um — the cracking-prone tail. | 0-1 | E2 |
| Si area share >5 um | Share of silicon mass in particles bigger than 5 um — extreme tail, the most dangerous chunks. | 0-1 | E2 |
| Si contact out-of-band share | Share of silicon particles whose pore access falls outside the acceptable band — either smothered or floating in a void. | 0-1 | E6 |
| Fully enclosed Si share | Share of silicon particles completely sealed inside graphite with no pore nearby — likely dead material that never charges and adds pure swelling stress. | 0-1 | E6 |
| Si-pore boundary density (um-1) | Direct silicon-to-pore contact length — the seams where silicon can actually expand into empty space. | um-1 | - |
| Si-to-pore distance, mean (um) | Typical distance from a silicon edge to the nearest pore — how far expansion has to travel to find space. | um | E6 |
| Si boundary per unit Si area (um-1) | Silicon edge length per unit silicon area — a proxy for how much silicon surface could grow SEI or expose fresh surface if it cracks. | um-1 | - |
| Swelling budget (P01) | Empty space left after reserving room for silicon's worst-case expansion (~3x) plus graphite's (~10%). Negative = the material must thicken or crack on charge. | fraction | E1, E4 |
| Swelling budget, worst 5% window | Worst-case local breathing room — the 5% of windows most starved of expansion space. Where the electrode would fail first. | fraction | - |
| Area share with negative swelling budget | Share of the electrode where the swelling budget is negative — the fraction that must thicken on charge. | 0-1 | E1 |
| 2D tortuosity proxy (through-plane) | How much longer the actual path through connected pores is versus a straight line down — the ion-detour estimate behind plating risk. | ratio | E5 |
