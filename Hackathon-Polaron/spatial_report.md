# Spatial heterogeneity — the defect-risk signals

All values from the frozen vcompare recipe — **diagnostic/exploratory layer**: these markers depend on where object boundaries land and none survive acquisition blocking, so they are not batch markers (see `marker_session_partition.csv`). `top`/`bottom` are frame edges; the foil is not in frame and the direction-to-physical-axis mapping is unconfirmed (the earlier FFT-based orientation claim was withdrawn — see `archive/ARCHIVE.md`). Batch medians [min-max]:

| signal | Batch_1 | Batch_2 | Batch_3 | New |
|---|---|---|---|---|
| si_topbot_ratio | 0.903 [0.433,1.323] | 1.099 [0.580,2.377] | 0.945 [0.657,2.238] | 1.256 [0.958,1.460] |
| pore_topbot_ratio | 1.032 [0.774,1.294] | 0.970 [0.659,1.360] | 0.899 [0.515,1.829] | 1.333 [0.520,1.728] |
| si_depth_slope | -0.002 [-0.006,0.015] | -0.001 [-0.015,0.005] | -0.000 [-0.015,0.007] | -0.005 [-0.008,-0.002] |
| pore_depth_slope | -0.002 [-0.006,0.005] | -0.002 [-0.006,0.008] | 0.001 [-0.010,0.013] | -0.003 [-0.007,0.014] |
| si_hotspot_ratio | 2.723 [2.310,4.920] | 2.907 [2.457,4.260] | 3.119 [2.241,5.428] | 2.509 [1.992,4.522] |
| si_maxpatch_frac | 0.236 [0.177,0.283] | 0.237 [0.184,0.281] | 0.205 [0.147,0.431] | 0.204 [0.185,0.244] |
| si_lr_asym | 0.004 [0.001,0.035] | 0.008 [0.003,0.024] | 0.009 [0.001,0.047] | 0.014 [0.007,0.023] |
| pore_hotspot_ratio | 1.792 [1.657,2.288] | 2.151 [1.594,2.296] | 2.029 [1.576,2.663] | 1.814 [1.602,1.838] |
| largest_si_cluster_frac | 0.046 [0.037,0.076] | 0.047 [0.038,0.066] | 0.055 [0.043,0.079] | 0.049 [0.040,0.053] |
| cracklike_frac | 0.194 [0.068,0.301] | 0.173 [0.106,0.312] | 0.209 [0.098,0.457] | 0.148 [0.125,0.199] |
| si_near_pore_frac | 0.031 [0.016,0.052] | 0.030 [0.015,0.047] | 0.020 [0.004,0.039] | 0.021 [0.010,0.060] |

## Reading the signals

- `si_hotspot_ratio` high = one patch carries much more Si than typical -> uneven loading, local swelling stress.
- `largest_si_cluster_frac` = share of Si that is ONE connected region — the 'enormous continuous cluster' case.
- `si_topbot_ratio` != 1 = Si segregated top-to-bottom in the frame (drying/calendering hypothesis only — physical direction unconfirmed).
- `cracklike_frac` = share of pore area in thin frame-vertical objects — crack-like rather than round voids. Caveat: streaking can also be a sectioning artefact, and it is session-sensitive.
- `si_near_pore_frac` = Si within 0.15 µm of a resolved pore in this 2-D slice — a measured blind spot (sub-resolution pores dominate real porosity), not a batch marker.