# micro2dfn — microstructure markers & DFN consequence report

**What this is.** SEM cross-section photos of a silicon–graphite battery anode were colour-coded into three materials — pores (black), graphite (grey), silicon (bright) — using **one shared set of brightness cutoffs** for every photo, so no batch can re-calibrate itself into looking normal. From those maps we measured ~30 structural markers per image, then fed them into the standard PyBaMM Doyle–Fuller–Newman battery model as consequence *indicators*.

**Honesty rules.** Numbers are reported as batch *differences* with 95% CIs — the defensible signal. Absolute DFN outputs carry stated assumptions and are illustrative. Nothing here predicts cycle life or guarantees failure. Every value traces back to a marker, a formula, and an overlay image.

## Photo-quality check

**2 low-contrast photos flagged** — silicon may be over-counted in them (the object classifier separates classified Si particles from uncertain-bright material; check the overlays):

- `Batch_1/img_4ih2ggld` (si_bulk_contrast=1.78, bright_fine_frac=0.081)
- `Batch_1/img_5n1q8atc` (si_bulk_contrast=1.93, bright_fine_frac=0.081)

## Segmentation settings (frozen)

- Shared thresholds: `t_pore=0.7109`, `t_si=1.3354` — fitted on the baseline batch **Batch_3** only, saved in `recipe.json`, and reused verbatim on later runs. A newly added batch can never change how existing batches are measured; `run_manifest.json` records the recipe md5, code md5s and every input image md5 for this exact run.
- Si/bright-fine classifier cutoff: `t_core=1.550` (5th percentile interior brightness of large, compact particles, calibrated on baseline objects only).
- Resolution floor: Si objects < 150 px and pores < 20 px (~0.35 um) are merged into bulk — same rule for every image.
- **Cached measurements reused** (`--reuse`): markers and objects loaded from `markers_all.csv`/`objects_all.csv` written by the run recorded in `run_manifest.json`.

## Marker summary per batch

Median [IQR] per batch. Full per-image values in `markers_<batch>.csv`; column meanings in `data_dictionary.csv`.

| Marker | Batch_1 | Batch_2 | Batch_3 |
|---|---|---|---|
| Resolved porosity (image-visible pores) | 0.1038 [0.0915,0.11] | 0.1083 [0.0971,0.117] | 0.109 [0.0893,0.115] |
| Silicon fraction (of coating) | 0.05954 [0.0381,0.0622] | 0.06141 [0.0587,0.0685] | 0.06034 [0.0504,0.0642] |
| Ambiguous bright-material fraction | 0.01253 [0.0097,0.0491] | 0.01248 [0.0122,0.0166] | 0.01374 [0.00979,0.02] |
| Si particle diameter, median (area-weighted) | 2.588 [1.89,2.76] | 2.673 [2.43,2.75] | 2.528 [2.3,2.66] |
| Si particle diameter, 90th pct (area-weighted) | 5.457 [4.65,5.77] | 5.898 [4.86,6.22] | 5.271 [4.67,5.76] |
| Si particle radius (3D estimate) | 1.735 [1.6,1.89] | 1.815 [1.76,1.95] | 1.825 [1.7,1.9] |
| Si surface area per volume | 0.4255 [0.364,0.736] | 0.4811 [0.464,0.534] | 0.3767 [0.333,0.487] |
| Si clustering index (Clark-Evans R) | 0.6767 [0.658,0.679] | 0.6928 [0.655,0.698] | 0.6649 [0.619,0.695] |
| Graphite effective particle radius | 3.108 [2.96,3.14] | 2.991 [2.87,3.3] | 3.156 [2.96,3.51] |
| Graphite flake alignment (h/v) | 1.173 [1.12,1.21] | 1.163 [1.16,1.17] | 1.196 [1.16,1.2] |
| Pore anisotropy (flat vs upright) | 1.168 [1.12,1.21] | 1.152 [1.14,1.18] | 1.189 [1.15,1.2] |
| Vertical crack fraction | 0.01101 [0.00892,0.0115] | 0.01551 [0.00882,0.0167] | 0.01322 [0.0119,0.0143] |
| Pore connectivity (2D reach) | 0.01737 [0.0136,0.0185] | 0.03005 [0.0237,0.0339] | 0.02291 [0.0149,0.043] |
| Largest connected pore region | 0.03838 [0.0353,0.0497] | 0.0548 [0.0407,0.056] | 0.05891 [0.0486,0.0874] |
| Effective Bruggeman exponent | 1.229 [1.18,1.39] | 1.306 [1.2,1.34] | 1.282 [1.21,1.35] |
| Si-to-pore distance, mean | 0.4853 [0.461,0.519] | 0.5346 [0.478,0.573] | 0.5675 [0.536,0.625] |
| Low-coverage Si share | 0.8614 [0.796,0.875] | 0.8635 [0.838,0.894] | 0.9101 [0.872,0.979] |

## Batch differences

No pair–marker difference survives Benjamini–Hochberg correction (FDR 0.10).

Nominal (uncorrected) p<0.05 count: 1 of 51 tests — compare against ~2.6 expected by chance. `mdd` is the smallest median difference detectable at ~5%: smaller observed gaps are under-powered, not proven absent. These are UNBLOCKED tests — acquisition session can organise them; the stratified within-session test is in `build_report.py` (`blocked_permutation.csv`).

All pairwise differences are in `batch_differences.csv`.

## DFN consequence indicators

**Plain conclusion.** Paired same-seed comparisons (same cathode at each sweep index for every batch) shift predicted cycle-5 discharge capacity by Batch_1-Batch_3 +0.109 Ah; Batch_2-Batch_3 +0.149 Ah; Batch_1-Batch_2 -0.0466 Ah — while the swept assumptions alone shift it by ~1.14 Ah (pooled IQR). Judged against the declared up-front thresholds and the paired ranges: 3 of 12 pair-indicator differences are meaningful, 9 sit fully below threshold, the rest are inconclusive. Lithium loss and capacity retention are reported only as diagnostics — they cannot move measurably in 5 cycles and are not part of the comparison. Pore-count differences are image findings only — they are not a DFN input and the model verdict does not cover them. See `dfn_paired_differences.csv` for every verdict.

PyBaMM (rebuilt report; solves unchanged) DFN (composite Si/Gr anode, swelling + SEI + stress-driven LAM), 5 x 1C cycles, 17 points spanning the marker uncertainty bands, downsample x2, seed fixed (0). Median [min, max] across runs of the primary `accessible` variant (Si share bracketed by classified-x-accessible to all-uncertain-bright). Full settings in `dfn_validation.md`.

| Indicator | Batch_1 | Batch_2 | Batch_3 |
|---|---|---|---|
| Cycle-1 discharge capacity [A.h] | 5.59 [4.24,6.91] | 5.68 [4.26,7.02] | 5.48 [4.14,6.79] |
| Capacity retention first->last cycle [%] (diagnostic only — not compared) | 100 [100,100] | 100 [100,100] | 100 [100,100] |
| Loss of lithium inventory [%] (diagnostic only — not compared) | 0.0105 [0.01,0.0109] | 0.0105 [0.00996,0.0108] | 0.0105 [0.00995,0.0108] |
| Mean coulombic efficiency | 0.999 [0.999,1] | 0.999 [0.999,1] | 0.999 [0.999,1] |
| Peak cell thickness change during discharge [um] | 12.3 [9.49,13.7] | 12.2 [9.38,13.5] | 12.1 [9.3,13.5] |
| Peak Si surface stress during discharge [MPa] | 10 [7.29,13.9] | 9.53 [6.65,13.7] | 10.8 [7.99,14.8] |
| Min negative-electrode surface potential [V] | -0.00824 [-0.0181,-0.000866] | -0.00824 [-0.0181,-0.000625] | -0.00855 [-0.0185,-0.000968] |
| Usable N/P ratio | 0.935 [0.924,0.942] | 0.943 [0.923,0.953] | 0.913 [0.913,0.913] |

Convergence: Batch_1 17/17; Batch_2 17/17; Batch_3 17/17 runs.
Plating (reversed rule): a run dipping below 0 V anode surface potential counts as a warning ONLY where the reference cell under the same protocol stays >= 0 V; if the reference also dips, sub-zero is a protocol/geometry artefact and is not attributed to any batch. Per-run warning counts are in `dfn_paired_differences.csv` (`plating_warnings_*`); reference min = n/a V — see `dfn_validation.md`.

*These are model-predicted indicators, not cell-test data. Compare the bands between batches — do not read absolute values as predictions.*

## Assumed (not measured) parameters

| Parameter | Value | Why assumed |
|---|---|---|
| Electrode thickness | 75 um [65-85 swept] | foil & coating top not in frame |
| Sub-resolution porosity | +0.15-0.30 | pores below ~0.35 um are invisible |
| Si active share | classified Si x accessible share -> all uncertain bright counted | label uncertainty treated as an explicit bracket; 'all-Si-active' variant run separately |
| Si Young's modulus / Poisson ratio | 50 GPa [35-90] / 0.25 [0.22-0.28] | Bonkile2024 cited values, swept over range |
| Si mechanics block | Bonkile2024 | swappable — if EDS shows the bright phase is SiOx, refit (`_si_block` in simulate.py) |
| Bruggeman default | 1.5 | literature value when tau proxy degenerates |
| Cathode thickness | rescaled per run to hold reference N/P | keeps min-anode-potential meaningful under assumed geometry |
| Degradation constants | OKane2022 published (LAM 2.7778e-7 s-1, SEI k 1e-12 m/s) | audit table in `dfn_validation.md` |
| Other material constants | Chen2020_composite | diffusivities, kinetics, densities not image-measurable |

## Not measurable from these images

Binder distribution (no BSE contrast), SEI layer, lithium inventory, 3-D pore connectivity / electrolyte wetting (2-D sections only — see the optional reconstruction branch), foil corrosion.

**Note on tortuosity.** In every image the resolved pore network does not form a connected path across the section, so a diffusion-solve tortuosity cannot be measured on 2-D cuts — the through-plane network lives largely in sub-resolution pores. Directional markers (pore anisotropy, geodesic proxy, Bruggeman-equivalent exponent) carry the usable signal.

## How every number was produced

```
BSE TIFF -> flat-field (Gaussian background)
  -> shared multi-Otsu thresholds (frozen to thresholds.json)
  -> watershed split + Si/bright-fine object classification
  -> per-image stereological markers (this report)
  -> per-batch median/IQR + bootstrap CIs
  -> PyBaMM DFN sweep over marker bands -> indicators
```

Files: `markers_*.csv` (per image), `objects_*.csv` (per Si object), `data_dictionary.csv` (every column explained), `dfn_params_*.json` (exact simulator inputs), `dfn_results.csv` (every sweep run), `figs/` (distributions, overlays, bands, tornado), `micro2dfn_results.xlsx` (all tables).
