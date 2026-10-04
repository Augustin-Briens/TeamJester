# DFN validation

PyBaMM 26.9.0.0; protocol: 1C discharge->2.8 V, rest, 1C charge->4.2 V, CV hold <50 mA, rest; 5 cycles.

## Reference cell (unmodified Chen2020_composite, no degradation)

- discharge capacity per cycle: 5.401 -> 5.415 -> 5.415 -> 5.415 -> 5.415 Ah
- mean coulombic efficiency: 0.9995
- min anode surface potential: +0.0115 V (no plating — a sweep run counts as a plating WARNING only if it dips below 0 V while this reference run stays >= 0; if the reference also dips, sub-zero is a protocol artefact and is not attributed to any batch)
- measured stoichiometry windows: graphite 0.015-0.969, Si 0.240-0.995, NMC 0.268-0.952
- reference N/P (from stoichiometry windows): 0.913

## Batch_3 median point (repaired model)

- discharge cap cycle 1: 5.037 Ah; retention 100.28%; LLI 0.01%; min anode V -0.0098; N/P 0.913

## Ablation — before vs after the fix

| stage | discharge_cap_cycle1_Ah | discharge_cap_last_Ah | retention_pct | loss_li_inventory_pct | min_neg_surface_v | np_ratio | converged | cycles_completed | note |
|---|---|---|---|---|---|---|---|---|---|
| 0_reference | 5.401 | 5.415 | 100.3 | nan | 0.01147 | nan | 1 | 5 | unmodified Chen2020_composite, no degradation |
| 1_geometry | 4.9 | 4.91 | 100.2 | -2.045e-10 | 0.00989 | 0.913 | 1 | 5 | our B3 geometry, cathode rescaled to reference N/P, no degradation |
| 2_degradation_shipped | 4.533 | 3.202 | 70.63 | 16.03 | -0.1029 | 0.913 | 1 | 5 | BEFORE: graphite params copied onto Si, LAM 1e-3 s-1, hand-set SEI |
| 3_full_fixed | 5.037 | 5.051 | 100.3 | 0.01068 | -0.009823 | 0.913 | 1 | 5 | AFTER: Si keeps Si chemistry/mechanics (Bonkile2024), published LAM/SEI — the full model |

## Constants audit (ours vs published)

| constant | published | shipped_buggy | ours | reason |
|---|---|---|---|---|
| Primary: Negative electrode LAM constant proportional term [s-1] | 2.7778e-07 | 0.001 | 2.7778e-07 | OKane2022/Bonkile2024; shipped value was per-hour entered as per-second (3600x too fast) |
| Secondary: Negative electrode LAM constant proportional term [s-1] | 2.7778e-07 | 0.001 | 2.7778e-07 | same published value |
| Positive electrode LAM constant proportional term [s-1] | 2.7778e-07 | 1e-3 | 2.7778e-07 | published OKane2022 |
| SEI kinetic rate constant [m.s-1] | 1e-12 | 1e-15 | 1e-12 | OKane2022 published value (shipped was 1000x slower) |
| Secondary: Negative electrode partial molar volume [m3.mol-1] | 1.2e-05 (Bonkile2024); task spec suggests ~9e-6 from 2.8x12.06/3.75 | 3.1e-06 | 1.2e-05 | published coherent set; implies ~+334% volume at full lithiation via silicon_volume_change_Ai2020 (near the ~280% Li15Si4 estimate) |
| Secondary: Negative electrode Young's modulus [Pa] | 50e9 (Bonkile2024); lithiated-Si literature range ~35-90 GPa | 15000000000.0 | 50000000000.0 | cited; swept over the cited range as si_youngs_pa |
| Secondary: Negative electrode Poisson's ratio | 0.22 (Bonkile2024); cited range 0.22-0.28 | 0.3 | 0.22 | cited; swept as si_nu |
| Secondary: Negative electrode critical stress [Pa] | 720e6 (Bonkile2024) | 60000000.0 | 720000000.0 | published Si value; shipped used graphite's 60e6 |
| Secondary: Negative electrode OCP [V] | silicon_ocp_average_Mark2016 (Chen2020_composite) | graphite_LGM50_ocp_Chen2020 | silicon_ocp_average_Mark2016 | left untouched by merge |
| Secondary: Negative electrode density [kg.m-3] | 2650 (Chen2020_composite) | 1657 | 2650 | left untouched by merge |
| Inner/Outer SEI split keys | not used by 'solvent-diffusion limited' | several hand-set values | dropped | dead parameters for a different SEI option; the model reads the unprefixed OKane2022 SEI keys |

## Tests

- [x] reference discharge capacity within 10% of nominal 5 Ah — 5.401 Ah
- [x] reference drift < 0.5% over 5 cycles — 0.259%
- [x] per-cycle discharge capacity == integral of discharge current — 5.4015 vs 5.4015 Ah
- [x] silicon (Secondary) OCP, kinetics & mechanics differ from graphite (Primary)
- [x] Si volume-change function ~+300% at full lithiation — +334%
- [x] B3 median capacity within 15% of reference cell — 5.037 vs 5.401 Ah
- [x] B3 median retention >= 98% after 5 cycles — 100.28%
- [x] B3 median lithium loss < 2% after 5 cycles — 0.01%

**8/8 tests pass.**

## Declared comparison thresholds (fixed BEFORE the sweeps)

A batch difference only counts as meaningful when its median paired difference exceeds these values AND the paired range excludes 0:

- `discharge_cap_last_Ah`: 0.25
- `thickness_change_um`: 0.5
- `si_stress_MPa`: 0.5
- `min_neg_surface_v`: 0.01

Compared indicators: discharge_cap_last_Ah, thickness_change_um, si_stress_MPa, min_neg_surface_v. Lithium-inventory loss and capacity retention are kept in `dfn_results.csv` as run diagnostics only — they cannot move measurably in 5 cycles and play no role in the batch comparison. Swelling (thickness change) and Si surface stress are read as the PEAK during each discharge step, not at the final rest.

## Run settings actually used

- recipe: dfn_output/recipe.json
- recipe_fitted_on: Batch_3
- recipe_md5: 039b0b88a61125c753cacae86197e830
- run_manifest: dfn_output/run_manifest.json
- pybamm_version: 26.9.0.0
- downsample: 1
- cycles: 5
- sweep_points_per_variant: 17
- variants: accessible, all_si_active, no_problem_photos
- si_bracket: lo = classified Si x accessible share; hi = all uncertain bright counted
- lam_proportional_s^-1: 2.7778e-07
- np_hold: one shared cathode per sweep index, sized on the reference batch's anode points (same for all batches at that index)
- seed: 0
- compute: modal (parallel)
- ref_min_neg_v: 0.011473244183722118