"""Constants for the recon3d pipeline.

Everything derived from the images inherits micro2dfn's frozen recipe;
everything here is a modelling choice and is written into report.md.
"""

# ---------------------------------------------------------------------------
# Training data
# ---------------------------------------------------------------------------
# CPU budget (Modal GPU is paywalled): 100 nm voxels keep the 6.4 um
# RVE at 64^3 — an 8x cheaper volume than 128^3 @ 50 nm, at the cost of
# dropping sub-0.35 um pore throats (already acknowledged unresolved).
DOWNSAMPLE: int = 4            # 25 nm -> 100 nm/voxel for the 3-D grid
VOXEL_NM: float = 100.0        # = micro2dfn PIXEL_NM * DOWNSAMPLE

# ---------------------------------------------------------------------------
# SliceGAN
# ---------------------------------------------------------------------------
VOLUME: int = 64               # cube edge in voxels -> 64*100nm = 6.4 um
N_PHASES: int = 3              # pore / bulk / bright
LATENT: int = 32               # noise depth per channel (z-dim of G input)
G_CH: int = 32                 # generator channel width
D_CH: int = 32                 # discriminator channel width
N_CRITIC: int = 5              # D steps per G step (WGAN-GP)
LAMBDA_GP: float = 10.0
LR: float = 1e-4
BETA1, BETA2 = 0.5, 0.9
BATCH: int = 4                 # real 2-D crops vs generated slices per step
ITERS: int = 6000              # G updates (full run, ~55 min on 8 CPU)
ITERS_FAST: int = 300          # smoke-test only — NOT a trained model
PRINT_EVERY: int = 100

# Discriminators see only slices that contain the through-plane axis
# (volume axis 0).  In-plane (axis-1/axis-2) statistics are assumed equal
# — the single-view anisotropy limitation documented in report.md.
SLICE_AXES: tuple = (1, 2)

# Ensemble: seeds x volumes-per-seed realizations per batch
SEEDS: tuple = (0, 1, 2)
VOLUMES_PER_SEED: int = 4
SEEDS_FAST: tuple = (0,)
VOLUMES_FAST: int = 2

# ---------------------------------------------------------------------------
# Transport solve (steady-state Laplace on the pore phase)
# ---------------------------------------------------------------------------
CG_TOL: float = 1e-7
CG_MAXITER: int = 4000
TAU_DIRECTIONS: tuple = ("through-plane", "in-plane")
# "accessible Si": bright voxels face-adjacent to the spanning pore cluster
CONTACT_VOX: int = 1

# ---------------------------------------------------------------------------
# Validation — generated slices must match held-out real slices on these
# two-point statistics (reported, not gated)
# ---------------------------------------------------------------------------
S2_MAX_R: int = 64             # px — two-point correlation range
N_VAL_REAL: int = 48           # held-out real slices per batch
N_VAL_FAKE: int = 48           # generated slices per batch
