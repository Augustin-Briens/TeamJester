"""Constants for the micro2dfn pipeline.

Recipe follows the validated polaron_qc segmentation/feature settings;
physical constants are traceable to Polaron_Physics_Dictionary.md.
"""

# Acquisition
PIXEL_NM: float = 25.0            # BSE pixel pitch (TIFF resolution tags)
PIXEL_UM: float = PIXEL_NM / 1000.0

# Phase labels
PORE, BULK, SI = 0, 1, 2

# Segmentation (one shared threshold pair, fitted on pooled pixels)
FLATFIELD_SIGMA_PX: float = 200.0
SEG_SMOOTH_SIGMA_PX: float = 1.5
THRESH_SWEEP_DELTA: float = 0.02
SEG_SUBSAMPLE: int = 2_000_000

# Object / cleanup rules
MIN_BRIGHT_PX: int = 150          # Si specks below ~0.35 um -> bulk
MIN_FEATURE_PX: int = 20          # pore specks below -> bulk
BIG_VOID_IMAGE_FRAC: float = 0.005
CRACK_ASPECT: float = 3.0
CRACK_ANGLE_DEG: float = 30.0
N_PATCH_STRIPS: int = 6
CONTACT_DILATION_PX: int = 3      # ~75 nm Si<->pore contact band
LARGE_SI_CUTOFFS_UM: tuple = (3.0, 5.0)
WATERSHED_MIN_DIST_PX: int = 8
TORT_DOWNSAMPLE: int = 4
TILE_GRID: int = 5

# Si object classifier (si_particle vs bright_fine)
SI_CALIB_MIN_DIAM_UM: float = 2.0     # calibration set: clear particles
SI_CALIB_MIN_SOLIDITY: float = 0.85
SI_CORE_PERCENTILE: float = 5.0       # t_core = p5 of interiors
SI_MIN_SOLIDITY: float = 0.75
OBJECT_ERODE_PX: int = 2              # interior = median after 2 px erosion

# Stereology
SECTION_FACTOR: float = 0.785         # mean 2D circle diam = 0.785 * 3D sphere diam
CROFTON: float = 3.141592653589793 / 4.0  # S_V = (4/pi) * L_A

LOW_COVERAGE_CUT: float = 0.05        # particle with <5% boundary at pore

# ---------------------------------------------------------------------------
# DFN-facing parameters — ASSUMED, not measured (always flagged in outputs)
# ---------------------------------------------------------------------------
ELECTRODE_THICKNESS_M: float = 75e-6  # foil & coating top not in frame
SUBRES_POROSITY_LO: float = 0.15      # pores below resolution: literature band
SUBRES_POROSITY_HI: float = 0.30      #   -> porosity_est = resolved + delta
BRUGGEMAN_DEFAULT: float = 1.5        # used when tau proxy is degenerate
