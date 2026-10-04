# Metric watchlist — one page

The ~30 markers reduce to **8 things worth watching** on any new
image. Full definitions: `metric_dictionary.csv`. Guard metrics gate
trust; content metrics carry interpretation.

| # | Watch | Why it matters | Red flag |
|---|---|---|---|
| 1 | **Si:bulk contrast** + noise_MAD + sharpness | acquisition guards — if the photography changed, no material comparison is valid | contrast < ~1.9 or z < −2 vs baseline; noise > ~11 |
| 2 | **Resolved pore fraction** | the reliable, recipe-robust geometry number; feeds DFN porosity (plus assumed sub-res) | shifts beyond baseline envelope under the FROZEN recipe |
| 3 | **Pore count + noise-adjusted count** | the supported Batch_1 finding — but noise inflates counts (correlation ~0.91) | raw shift that vanishes after adjustment = photography, not material |
| 4 | **Si candidate fraction AND uncertain-bright fraction — always together** | the classifier boundary between them is a recipe choice; quoting either alone overstates | reporting Si without the uncertain-bright share |
| 5 | **Si size: area-weighted D50 / D90** | feeds the DFN particle radius; watershed-sensitive | border-touching objects dominating; D50 ≠ number-median by ~3x |
| 6 | **Si→pore proximity + low-coverage share** | 2-D wetting proxy; scales the DFN Si-share bracket | quoting as real 3-D wetting |
| 7 | **Model stress + swelling budget** | consequence indicators — assumption-dominated | reading them as measured properties; Si share drives them more than geometry |
| 8 | **Envelope coverage for new images** | similarity assignment only | treating "inside envelope" as provenance or quality |

## Never quote without a class label

`measured` (from pixels, recipe-dependent) · `inferred` (derived,
assumption-carrying) · `assumed` (declared inputs, swept) ·
`simulated` (model outputs, conditional) · `illustrative` (examples).

Units: µm = micrometre (hair ≈ 70 µm) · Mpx = million pixels ·
MPa = megapascals (stress) · Ah = amp-hours (charge) · z = standard
distance from the baseline median · D50 = median size.

## Known traps on this dataset

- **Pore count ≠ porosity.** The supported finding is a *count*
  density shift; the model takes porosity (the fraction), which is
  not significantly different.
- **"No clear difference" ≠ equivalence** — check the MDD
  (minimum detectable difference) column before saying "same".
- **The classifier IS the disagreement.** Between pipelines r ≈ 0.27
  on classified Si, 0.8–1.0 on everything else. Report
  candidate + uncertain together.
- **2-D sections cannot see thickness, 3-D connectivity, or
  sub-resolution porosity** — all three are declared assumptions in
  the model, not measurements.
- **Verdicts are similarity to a selected reference** — Batch_3 is
  not proven defect-free.
