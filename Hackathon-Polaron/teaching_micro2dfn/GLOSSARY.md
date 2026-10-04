# Glossary — plain-words definitions

| Term | Plain meaning |
|---|---|
| **anode** | the battery's negative electrode — here a thin graphite+silicon coating on a copper foil, photographed in cross-section. |
| **cross-section** | a slice through the coating, polished flat, so the internal structure faces the camera. |
| **SEM** | scanning electron microscope — the instrument that took the photos. |
| **BSE** | Back-scattered electrons — the imaging mode where brightness tracks mean atomic number. Brighter ≈ heavier elements. The segmentation channel. |
| **µm** | micrometre, 1/1000 of a millimetre. A human hair is ~70 µm; our smallest visible pores are ~0.13 µm. |
| **nm** | nanometre, 1/1000 of a µm. Pixel pitch here is 25 nm. |
| **Mpx** | million pixels — the unit for object densities (e.g. pores per Mpx). |
| **QC** | quality control — here: comparing a new batch against an approved reference batch. |
| **reference / baseline** | Batch_3 — the shipment-approved batch everything is compared against. A chosen ruler, not proven-perfect material. |
| **ETD / InLens** | Secondary-electron detector channels; ETD sees surface-sensitive contrast, InLens topography. Used in the adjudication texture tests, not for thresholds. |
| **FOV** | Field of view — one image = one specimen region. The statistical unit. |
| **pixel pitch (25 nm/px)** | spacing between pixel samples — NOT the same as resolution; real object floor ~0.13 µm (pores) / ~0.35 µm (Si). |
| **flat-field correction** | divide the image by a heavily blurred copy of itself to remove the illumination gradient before thresholding. |
| **multi-Otsu threshold** | automatic cut-points that minimise within-class variance; turns greys into pore/bulk/bright labels. A criterion, not a truth. |
| **t_pore / t_si / t_core** | the three frozen brightness cut-points: below t_pore = pore, above t_si = bright, above t_core = deep-bright (used by the classifier). |
| **frozen vs floating ruler** | thresholds fitted once on the reference and reused (frozen = comparable) vs refit per image (floating = the ruler bends to the anomaly). |
| **watershed** | splits fused objects by flooding from interior peaks; needed because particles touch. Can over-split — a stated size-metric bias. |
| **si_particle / bright_fine** | the classifier's two bright-object labels: clear particles vs ambiguous material (fines, rims, dim Si). Membership is a recipe decision. |
| **t_core** | interior-intensity bar for si_particle = 5th percentile of clear reference-particle interiors. Recipe assumption. |
| **solidity** | object area / convex-hull area; crescents and ragged objects score low. |
| **equivalent diameter** | diameter of the circle with the object's area. |
| **area-weighted D50/D90** | size percentiles weighted by object area — what matters to the model (big particles dominate volume). Number-median is ~3x smaller. |
| **Clark–Evans R** | nearest-neighbour clustering index; <1 clumped, ~1 random, >1 spaced. |
| **distance transform** | every pixel's distance to the nearest pore → Si-to-pore proximity (2-D wetting proxy). |
| **swelling budget** | pore fraction minus the expansion demand of Si (~280%) and graphite (~10%): a derived proxy, not a model input, not independent evidence. |
| **robust z-score** | (value − baseline median) / MAD; how unusual one image is vs the reference envelope. |
| **MAD** | median absolute deviation — a robust spread measure; noise_MAD doubles as the image-noise guard. |
| **correlation (r)** | how strongly two metrics move together, −1 to +1; 0 = unrelated, 1 = identical ranking. |
| **envelope** | the reference batch's own range of values; "inside the envelope" = looks like the reference. |
| **exact permutation test** | enumerate every way to split the images into fake "batches"; the p-value is where the observed difference ranks among all of them. No distribution assumed. |
| **BH / FDR 0.10** | Benjamini–Hochberg correction across the 9 features per pair — limits the expected share of false "significant" hits to 10%. |
| **MDD** | minimum detectable difference — how big a real difference must be before this dataset could likely see it. Every "no difference" is bounded by its MDD. |
| **LOO sweep** | leave-one-out: drop each image, re-test. If one photo carries the result, it shows here. |
| **consensus exceedance** | counting how many regions in a batch individually exceed the baseline p90 on the same signature — detects heterogeneous batches medians can't see. |
| **EDS / EDX** | energy-dispersive X-ray spectroscopy — measures which chemical elements are present. The one measurement that could settle "is the bright stuff really silicon". |
| **SEI** | solid electrolyte interphase — the thin protective film that forms on electrode surfaces; its growth consumes lithium. |
| **LAM** | loss of active material — a fade mechanism the model couples to particle stress. |
| **DFN model** | Doyle–Fuller–Newman: the standard porous-electrode battery model — lithium through electrolyte + inside particles, coupled to mechanics. Here via PyBaMM with a graphite+silicon negative. |
| **PyBaMM** | the open-source battery-modelling package that runs the DFN here. |
| **NMC** | nickel-manganese-cobalt — the cathode chemistry in the reference parameter set (NMC622). |
| **1C rate** | a full charge or discharge in about one hour; the test protocol is 5 cycles at 1C. |
| **CV hold** | constant-voltage hold — a top-up step at the end of charge. |
| **Ah** | amp-hours — battery charge; the reference cell is ~5.4 Ah. |
| **MPa** | megapascals — a unit of mechanical stress inside the silicon particles. |
| **IQR** | interquartile range — the middle-half spread of a set of values; used here to size the assumption-only fog. |
| **N/P ratio** | usable anode capacity / usable cathode capacity. Held at the reference value across the sweep by resizing the cathode. |
| **assumption sweep (17-point)** | running the model at the band centre + corners of the unmeasured inputs, so every indicator is a band, not a fake point. |
| **paired difference** | batch A minus batch B at the SAME assumption index — removes the shared-assumption axis to first order. |
| **declared threshold** | meaningful-difference bar fixed BEFORE looking (0.25 Ah / 0.5 µm / 0.5 MPa / 0.01 V) — prevents post-hoc storytelling. |
| **evidence class** | measured / inferred / assumed / simulated / illustrative — every number's epistemic label. |
| **MDD-critical vs 80%** | smallest difference detectable at the chosen power levels; pairwise table reports the 80% value. |
