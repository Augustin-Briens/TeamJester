# New-image batch assignment — ALL CALLS: "CAN'T TELL"

**Status correction (acquisition analysis, 2026-10-04):** each new
photo's frame height occurs in only one batch's acquisition group, and
each call followed that session — the assignments cannot separate
batch-match from session-match. The protocol (predictions saved blind
before any ground truth) is kept and remains the right pattern; the
verdicts are relabelled **"can't tell"** until session-controlled data
exists. Do not quote the assigned column below as material evidence.

Method (v2): primary rule = envelope coverage — count of the 9 declared features inside each batch's observed min-max range (unbiased by batch spread). Exactly one batch containing 9/9 => that batch; several => inconclusive; none => best coverage wins unless median robust-z >2.5 => out-of-distribution. Median-z kept as a secondary score (it favours wide batches — see `medianz_winner` in assignments.csv). Recipe frozen on Batch_3 (`validated_comparison/recipe.json`).

| image | assigned (superseded) | inside env. (B1/B2/B3) | median-z best | confidence |
|---|---|---|---|---|
| img_3e122cbj | Batch_1 → can't tell | 7/3/3 | Batch_1 2.20 | best envelope coverage 7/9 — session-matched (group 2316 = B1-only session) |
| img_fn0mhxef | Batch_2 → can't tell | 8/9/8 | Batch_1 0.38 | 9/9 envelope — session-matched (group 2048 = B2-only session) |
| img_xrv9xvzb | Batch_3 → can't tell | 6/7/9 | Batch_1 0.66 | 9/9 envelope — session-matched (group 2088 = B3-only session) |

## Per-image notes

### img_3e122cbj → Batch_1 (best envelope coverage 7/9 (median-z 2.20))
- vs Batch_1: 7/9 features inside envelope; median-z 2.20
- vs Batch_2: 3/9 features inside envelope; median-z 3.61
- vs Batch_3: 3/9 features inside envelope; median-z 2.61
- largest deviations vs Batch_1: uncertain_bright_frac z=10.5; pores_per_mpx_noise_adj z=2.4; si_d50_um z=2.4
- key features: pore_frac=0.098, pores/mpx=206 (noise-adj 193), Si cand=0.036, uncertain=0.086, Si d50=1.21 um, R=0.69

### img_fn0mhxef → Batch_2 (envelope-consistent (9/9 features inside Batch_2; Batch_1 8/9; Batch_3 8/9))
- vs Batch_1: 8/9 features inside envelope; median-z 0.38
- vs Batch_2: 9/9 features inside envelope; median-z 0.51
- vs Batch_3: 8/9 features inside envelope; median-z 0.87
- largest deviations vs Batch_2: pore_anisotropy z=1.5; si_d50_um z=1.5; si_candidate_frac z=1.0
- key features: pore_frac=0.102, pores/mpx=132 (noise-adj 126), Si cand=0.068, uncertain=0.019, Si d50=3.05 um, R=0.71

### img_xrv9xvzb → Batch_3 (envelope-consistent (9/9 features inside Batch_3; Batch_1 6/9; Batch_2 7/9))
- vs Batch_1: 6/9 features inside envelope; median-z 0.66
- vs Batch_2: 7/9 features inside envelope; median-z 0.82
- vs Batch_3: 9/9 features inside envelope; median-z 0.70
- largest deviations vs Batch_3: pores_per_mpx_noise_adj z=2.1; si_clustering_R z=1.8; si_candidate_frac z=1.7
- key features: pore_frac=0.124, pores/mpx=115 (noise-adj 128), Si cand=0.040, uncertain=0.018, Si d50=2.66 um, R=0.56

## Caveats

- Assignment is similarity, not provenance: batches overlap on every feature, so weak-confidence calls are expected for images near the overlap region.
- With 7 images in Batch_1/2 and 17 in Batch_3, envelope estimates are noisy; scores carry that uncertainty.
- The frozen recipe segments BSE only; InLens/ETD channels are recorded but not used (channel comparability not yet established).