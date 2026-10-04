"""Marker cards — the anti-black-box layer.

Every marker carries a documentation block: plain name, what it measures,
how it is measured, its unit, which DFN input it feeds (if any), and what
a high value warns about. Rendered into report.md, data_dictionary.csv,
and the Excel "Dictionary" sheet.
"""
from __future__ import annotations

CARDS: dict[str, dict] = {
    # --- A. composition ----------------------------------------------------
    "pore_frac_resolved": dict(
        name="Resolved porosity (image-visible pores)",
        unit="fraction", group="A. Composition",
        measures="Share of the image that is empty space (dark regions) "
                 "large enough to see.",
        method="Pixel counting on the segmented map (Delesse: area "
               "fraction = volume fraction).",
        dfn=None,
        warns="Low -> fewer channels for lithium; high can mean loose "
              "coating or pull-out artefacts.",
        caveat="Only pores above ~0.35 um are counted; true electrode "
               "porosity is typically 25-40 %."),
    "porosity_est": dict(
        name="Estimated total porosity",
        unit="fraction", group="A. Composition",
        measures="Resolved porosity plus a correction for pores too small "
                 "to see.",
        method="pore_frac_resolved + delta_lit (0.15-0.30 literature band). "
               "ASSUMED component — swept as uncertainty.",
        dfn="Negative electrode porosity",
        warns="Feeds electrolyte transport; treated as a range, not a point.",
        caveat="The delta is assumed from literature, not measured."),
    "si_frac_total": dict(
        name="Silicon fraction (of coating)",
        unit="fraction", group="A. Composition",
        measures="Share of the image that is classified silicon particles.",
        method="Area of si_particle objects / image area.",
        dfn="(with si_frac_of_solid) Secondary active-material fraction",
        warns="High -> more swelling demand, more SEI surface.",
        caveat=None),
    "si_frac_of_solid": dict(
        name="Silicon share of the solid material",
        unit="fraction", group="A. Composition",
        measures="Silicon area divided by all solid area (Si + graphite).",
        method="si_frac_total / (1 - pore_frac_resolved).",
        dfn="Secondary: Negative electrode active material volume fraction",
        warns="High -> greater lithiation swelling per unit coating.",
        caveat=None),
    "gr_frac": dict(
        name="Graphite (+binder) fraction",
        unit="fraction", group="A. Composition",
        measures="Share of the image that is grey bulk material.",
        method="Pixel counting (binder has no BSE contrast — merged with "
               "graphite by design).",
        dfn="Primary: Negative electrode active material volume fraction",
        warns=None,
        caveat="Includes invisible carbon-binder phase."),
    "bright_fine_frac": dict(
        name="Ambiguous bright-material fraction",
        unit="fraction", group="A. Composition",
        measures="Bright objects that failed the silicon test (edges, "
                 "carbon-binder, fines).",
        method="Object classifier: interior brightness + compactness.",
        dfn=None,
        warns="Elevated -> low-contrast photo overcounting Si; trust "
              "si_frac_total instead of the raw bright fraction.",
        caveat="Guard rail — not a physical phase claim."),

    # --- B. silicon particles ----------------------------------------------
    "si_d50_num_um": dict(
        name="Si particle diameter, median (count-weighted)",
        unit="um", group="B. Silicon particles",
        measures="Typical silicon particle size counting every particle "
                 "equally.",
        method="Equivalent-circle diameter of each watershed-split "
               "si_particle; 50th percentile.",
        dfn=None,
        warns="Rising -> coarser Si grade.",
        caveat="2D sections under-read true 3D size (~0.785x)."),
    "si_d50_aw_um": dict(
        name="Si particle diameter, median (area-weighted)",
        unit="um", group="B. Silicon particles",
        measures="Particle size where most of the silicon *mass* lives.",
        method="Area-weighted quantile of equivalent diameters.",
        dfn="(basis for) Secondary particle radius",
        warns="Rising -> coarser Si grade / worse cracking risk.",
        caveat=None),
    "si_d90_aw_um": dict(
        name="Si particle diameter, 90th pct (area-weighted)",
        unit="um", group="B. Silicon particles",
        measures="Size of the large end of the silicon distribution.",
        method="Area-weighted 90th percentile.",
        dfn=None,
        warns="High -> large-particle tail; crack/pulverization risk.",
        caveat=None),
    "si_dmax_um": dict(
        name="Largest Si particle",
        unit="um", group="B. Silicon particles",
        measures="Biggest silicon particle fully inside the frame.",
        method="Max equivalent diameter of non-border objects.",
        dfn=None,
        warns="Very large -> single-particle cracking hazard.",
        caveat=None),
    "si_frac_gt3um": dict(
        name="Si area share in particles > 3 um",
        unit="fraction", group="B. Silicon particles",
        measures="How much silicon sits in large particles.",
        method="Area-weighted tail of the size distribution.",
        dfn=None,
        warns="High -> larger fraction at cracking risk on lithiation.",
        caveat=None),
    "si_frac_gt5um": dict(
        name="Si area share in particles > 5 um",
        unit="fraction", group="B. Silicon particles",
        measures="How much silicon sits in very large particles.",
        method="Area-weighted tail of the size distribution.",
        dfn=None,
        warns="High -> severe cracking/pulverization risk.",
        caveat=None),
    "si_radius_3d_um": dict(
        name="Si particle radius (3D estimate)",
        unit="um", group="B. Silicon particles",
        measures="Sphere-equivalent radius corrected for slice bias.",
        method="si_dmean_aw / (2 * 0.785) — Saltykov sectioning correction.",
        dfn="Secondary: Negative electrode particle radius [m]",
        warns="Feeds lithium diffusion length inside Si.",
        caveat="Assumes roughly spherical particles."),
    "si_sv_um_inv": dict(
        name="Si surface area per volume",
        unit="um^-1", group="B. Silicon particles",
        measures="Silicon boundary exposed to the rest of the coating, "
                 "per unit volume.",
        method="Crofton stereology: S_V = transitions * px / area.",
        dfn="(reported; can scale SEI exchange current)",
        warns="High -> more Si-electrolyte contact -> more SEI growth.",
        caveat=None),
    "si_solidity_mean": dict(
        name="Si particle compactness",
        unit="-", group="B. Silicon particles",
        measures="How solid/compact the silicon particles are (1 = smooth "
                 "convex).",
        method="Mean solidity over si_particle objects.",
        dfn=None,
        warns="Low -> jagged particles; sphere assumption degrades.",
        caveat=None),
    "si_aspect_mean": dict(
        name="Si particle elongation",
        unit="-", group="B. Silicon particles",
        measures="Mean major/minor axis ratio of silicon particles.",
        method="regionprops ellipse axes.",
        dfn=None,
        warns="High -> elongated particles; sphere assumption degrades.",
        caveat=None),
    "si_clustering_R": dict(
        name="Si clustering index (Clark-Evans R)",
        unit="-", group="B. Silicon particles",
        measures="How clumped silicon particles are (R<1 clumped, R~1 "
                 "random, R>1 evenly spaced).",
        method="Nearest-neighbour distance vs random expectation, "
               "Donnelly edge-corrected.",
        dfn=None,
        warns="Low -> clusters swell as one big particle -> local stress.",
        caveat="Graphite pockets force R<1 structurally."),
    "si_patchiness": dict(
        name="Si patchiness (strip CV)",
        unit="-", group="B. Silicon particles",
        measures="How unevenly silicon is distributed across the image.",
        method="CV of Si fraction across 6 vertical strips.",
        dfn=None,
        warns="High -> mixing/dispersion problem.",
        caveat=None),

    # --- C. graphite matrix -------------------------------------------------
    "gr_sv_um_inv": dict(
        name="Graphite surface per volume",
        unit="um^-1", group="C. Graphite matrix",
        measures="Bulk material boundary exposed to pores, per unit volume.",
        method="Crofton stereology on bulk-pore transitions.",
        dfn="(basis for) Primary particle effective radius",
        warns=None,
        caveat=None),
    "gr_radius_eff_um": dict(
        name="Graphite effective particle radius",
        unit="um", group="C. Graphite matrix",
        measures="Sphere-equivalent radius giving the measured surface-"
                 "to-volume ratio.",
        method="R_eff = 3 * V_AM / S_V.",
        dfn="Primary: Negative electrode particle radius [m]",
        warns="Feeds lithium diffusion length inside graphite.",
        caveat="Flakes are not spheres — diffusion-equivalent "
               "approximation."),
    "gr_chord_ratio_xz": dict(
        name="Graphite flake alignment (h/v)",
        unit="-", group="C. Graphite matrix",
        measures="Mean horizontal vs vertical chord through bulk; >1 = "
                 "flakes lie flat.",
        method="Chord-length statistics.",
        dfn=None,
        warns="High -> strong layering -> anisotropic ion/electron paths.",
        caveat=None),

    # --- D. pores & transport ----------------------------------------------
    "pore_d50_um": dict(
        name="Pore diameter, median",
        unit="um", group="D. Pores & transport",
        measures="Typical size of visible pores.",
        method="Median equivalent-circle diameter of pore objects.",
        dfn=None,
        warns=None,
        caveat="Resolved pores only."),
    "pore_count_per_mpx": dict(
        name="Pore count density",
        unit="per Mpx", group="D. Pores & transport",
        measures="Number of pore objects per million pixels.",
        method="Count of pore objects >= 20 px.",
        dfn=None,
        warns="High -> more fragmented pore network or finer pores.",
        caveat=None),
    "pore_anisotropy": dict(
        name="Pore anisotropy (flat vs upright)",
        unit="-", group="D. Pores & transport",
        measures=">1 means pores lie flat along the foil.",
        method="Horizontal / vertical chord ratio on the pore mask.",
        dfn=None,
        warns="High -> flat pores lengthen the through-plane ion path -> "
              "plating risk at fast charge.",
        caveat=None),
    "crack_frac_v": dict(
        name="Vertical crack fraction",
        unit="fraction", group="D. Pores & transport",
        measures="Area share of elongated pores pointing across the "
                 "coating (tearing signature).",
        method="Pore objects with aspect > 3 oriented > 30 deg from foil.",
        dfn=None,
        warns="High -> mechanical tears; also express ion lanes.",
        caveat="May partly reflect polishing/pull-out artefacts."),
    "crack_frac_h": dict(
        name="Horizontal crack fraction",
        unit="fraction", group="D. Pores & transport",
        measures="Area share of elongated pores lying along the foil.",
        method="Pore objects with aspect > 3 within 30 deg of foil.",
        dfn=None,
        warns="High -> delamination-type damage.",
        caveat=None),
    "big_void_frac": dict(
        name="Large-void fraction",
        unit="fraction", group="D. Pores & transport",
        measures="Area share of very large holes (>0.5 % of image).",
        method="Pore objects above the large-void area cutoff.",
        dfn=None,
        warns="High -> manufacturing voids / contamination damage.",
        caveat=None),
    "round_pore_frac": dict(
        name="Round pore fraction",
        unit="fraction", group="D. Pores & transport",
        measures="Area share of compact, non-elongated pores (control "
                 "metric).",
        method="Pore objects with aspect <= 3.",
        dfn=None,
        warns="Stable value = segmentation behaving normally.",
        caveat=None),
    "pore_verticality_aw": dict(
        name="Pore verticality (area-weighted)",
        unit="-", group="D. Pores & transport",
        measures="How upright the pore area is overall (1 = vertical).",
        method="Area-weighted |sin(angle vs foil)| over pore objects.",
        dfn=None,
        warns="Low -> pores lie flat -> longer ion path.",
        caveat=None),
    "tau_vertical_proxy": dict(
        name="Tortuosity proxy, vertical",
        unit="-", group="D. Pores & transport",
        measures="How much longer a path through the pore network is than "
                 "a straight line, top-to-bottom.",
        method="Geodesic (maze-solving) distance through pores vs straight "
               "depth.",
        dfn="(basis for) Bruggeman coefficient",
        warns="High -> sluggish ion transport through thickness.",
        caveat="Image top is NOT the separator (edges unknown) — use the "
               "anisotropy ratio, not the absolute."),
    "tau_horizontal_proxy": dict(
        name="Tortuosity proxy, horizontal",
        unit="-", group="D. Pores & transport",
        measures="Same measure along the in-plane direction.",
        method="Geodesic distance left-to-right.",
        dfn=None,
        warns=None,
        caveat=None),
    "tau_anisotropy": dict(
        name="Tortuosity anisotropy (vertical/horizontal)",
        unit="-", group="D. Pores & transport",
        measures="Directional penalty of the pore network.",
        method="tau_vertical_proxy / tau_horizontal_proxy.",
        dfn=None,
        warns="High -> ions detour much more through-thickness than "
              "in-plane -> plating risk.",
        caveat=None),
    "percolation_reach_2d": dict(
        name="Pore connectivity (2D reach)",
        unit="fraction", group="D. Pores & transport",
        measures="Fraction of pore pixels connected to the top edge "
                 "through the pore network.",
        method="Geodesic flood from the edge; connected share.",
        dfn=None,
        warns="Low -> dead-end pores carry no ions.",
        caveat="2D proxy only — true connectivity is 3D (needs tomography "
               "or the bonus reconstruction branch)."),
    "bruggeman_b_eff": dict(
        name="Effective Bruggeman exponent",
        unit="-", group="D. Pores & transport",
        measures="Tortuosity-porosity exponent implied by the measured "
                 "tau proxy.",
        method="b = 1 - ln(tau) / ln(eps), clipped to [1, 3]; default 1.5 "
               "if degenerate.",
        dfn="Negative electrode Bruggeman coefficient (electrolyte)",
        warns="High -> steeper transport penalty as porosity drops.",
        caveat="Heuristic calibration — replaced by TauFactor tau_z if "
               "the 3D bonus branch runs."),
    "pore_largest_region_frac": dict(
        name="Largest connected pore region",
        unit="fraction", group="D. Pores & transport",
        measures="Biggest connected pore blob as a share of all pore "
                 "pixels.",
        method="Connected components on the pore mask.",
        dfn=None,
        warns="Low -> fragmented pore network.",
        caveat=None),
    "si_coverage_mean": dict(
        name="Si boundary coverage by pores, mean",
        unit="fraction", group="D. Pores & transport",
        measures="Mean share of each silicon particle's boundary within "
                 "~75 nm of a pore.",
        method="Per-object boundary-coverage (ASSB-style coverage%).",
        dfn=None,
        warns="Low -> silicon poorly wetted by electrolyte.",
        caveat=None),
    "si_coverage_p10": dict(
        name="Si boundary coverage, worst 10%",
        unit="fraction", group="D. Pores & transport",
        measures="Coverage of the least-connected silicon particles.",
        method="10th percentile of per-object coverage.",
        dfn=None,
        warns="Near 0 -> a real tail of isolated silicon exists.",
        caveat=None),
    "si_lowcoverage_share": dict(
        name="Low-coverage Si share",
        unit="fraction", group="D. Pores & transport",
        measures="Share of silicon particles with <5% of their boundary "
                 "near a pore — effectively isolated.",
        method="Share of objects below the coverage cutoff.",
        dfn=None,
        warns="High -> substantial dead silicon.",
        caveat=None),
    "si_pore_dist_mean_um": dict(
        name="Si-to-pore distance, mean",
        unit="um", group="D. Pores & transport",
        measures="How far silicon boundaries sit from the nearest pore.",
        method="Distance transform from pore mask over Si boundary pixels.",
        dfn=None,
        warns="High -> silicon far from electrolyte -> poorly wetted, "
              "dead capacity.",
        caveat=None),
    "si_pore_contact": dict(
        name="Si-pore contact share",
        unit="fraction", group="D. Pores & transport",
        measures="Share of Si boundary within ~75 nm of a pore.",
        method="3-px dilation contact band (1-px test is degenerate here — "
               "Si sits ~0.6 um away).",
        dfn=None,
        warns="Low -> silicon isolated from electrolyte.",
        caveat=None),
    "si_enclosed_share": dict(
        name="Fully-enclosed Si share",
        unit="fraction", group="D. Pores & transport",
        measures="Share of silicon particles with zero pore contact.",
        method="Per-object contact test.",
        dfn=None,
        warns="High -> much silicon is electrochemically isolated.",
        caveat="Runs high (~0.7-0.99) in every image — relative "
               "comparison only."),
    "si_accessible_frac": dict(
        name="Accessible silicon fraction",
        unit="fraction", group="D. Pores & transport",
        measures="Share of silicon within reach of a pore.",
        method="1 - si_enclosed_share.",
        dfn="Scales Secondary active-material fraction (dead-Si proxy)",
        warns="Low -> much of the silicon may be inactive.",
        caveat=None),
    "corr_len_pore_um": dict(
        name="Pore correlation length",
        unit="um", group="D. Pores & transport",
        measures="Characteristic size of pore regions.",
        method="Two-point correlation (FFT) decay to 1/e.",
        dfn=None,
        warns=None,
        caveat=None),
    "corr_len_bulk_um": dict(
        name="Bulk correlation length",
        unit="um", group="D. Pores & transport",
        measures="Characteristic size of graphite domains.",
        method="Two-point correlation (FFT) decay to 1/e.",
        dfn=None,
        warns=None,
        caveat=None),
    "corr_len_si_um": dict(
        name="Si correlation length",
        unit="um", group="D. Pores & transport",
        measures="Characteristic spacing/size of silicon regions.",
        method="Two-point correlation (FFT) decay to 1/e.",
        dfn=None,
        warns=None,
        caveat=None),

    # --- E. swelling & heterogeneity — removed ----------------------------
    # swelling_budget / swelling_budget_min / swelling_deficit_area_frac:
    # negative by construction (resolved ~10% porosity vs real 25-40%);
    # "deficit tracks Si" was true by definition. See archive/ARCHIVE.md.
    # --- F. imaging guards ---------------------------------------------------
    "si_bulk_contrast": dict(
        name="Si vs bulk contrast ratio",
        unit="-", group="F. Imaging guards",
        measures="How much brighter silicon is than bulk, normalised to "
                 "the pore-bulk gap.",
        method="(I_si - I_pore) / (I_bulk - I_pore) on raw intensities.",
        dfn=None,
        warns="Low (~<1.9 here) -> low-contrast photo; Si may be "
              "overcounted (see bright_fine_frac).",
        caveat="Guard rail, not a material property."),
    "noise_mad": dict(
        name="Image noise level",
        unit="gray", group="F. Imaging guards",
        measures="Pixel noise on bulk material.",
        method="1.4826 * MAD of (raw - Gaussian(raw, 2 px)).",
        dfn=None, warns="High -> segmentation less reliable.",
        caveat="Guard rail."),
    "sharpness_lapvar": dict(
        name="Image sharpness",
        unit="gray^2", group="F. Imaging guards",
        measures="Focus quality (variance of Laplacian on bulk).",
        method="Var(|d2I|) over bulk pixels.",
        dfn=None, warns="Low -> blurry photo.", caveat="Guard rail."),
    "shading_slope": dict(
        name="Top-to-bottom shading slope",
        unit="gray/px", group="F. Imaging guards",
        measures="Brightness drift across the frame.",
        method="Linear slope of the row-median of bulk pixels.",
        dfn=None, warns="Strong -> uneven illumination left over.",
        caveat="Guard rail."),
    "pore_frac_tile_err": dict(
        name="Porosity tile uncertainty",
        unit="fraction", group="Uncertainty",
        measures="How much the measured porosity could differ just because "
                 "the image is finite.",
        method="95% CI half-width across a 5x5 tile jackknife.",
        dfn=None, warns=None, caveat=None),
    "si_frac_tile_err": dict(
        name="Si fraction tile uncertainty",
        unit="fraction", group="Uncertainty",
        measures="Same finite-image uncertainty for the silicon fraction.",
        method="95% CI half-width across a 5x5 tile jackknife.",
        dfn=None, warns=None, caveat=None),
    "is_problem_photo": dict(
        name="Problem-photo flag",
        unit="0/1", group="F. Imaging guards",
        measures="Low-contrast photo where Si may be overcounted.",
        method="si_bulk_contrast below the pooled 5th percentile across "
               "all images.",
        dfn=None,
        warns="Treat this photo's Si-derived markers with care; results "
              "reported with and without problem photos.",
        caveat="Guard rail."),

    # --- housekeeping / misc -----------------------------------------------
    "bright_frac_raw": dict(
        name="Raw bright fraction (all bright objects)",
        unit="fraction", group="A. Composition",
        measures="Bright pixels BEFORE the si_particle/bright_fine split — "
                 "what naive thresholding would call silicon.",
        method="Pixel counting on the raw bright phase.",
        dfn=None,
        warns="Compare with si_frac_total — a big gap means overcounting.",
        caveat=None),
    "si_per_mpx": dict(
        name="Si particle count density",
        unit="per Mpx", group="B. Silicon particles",
        measures="Number of silicon particles per million pixels.",
        method="Count of classified si_particle objects.",
        dfn=None, warns="High -> finer Si distribution.", caveat=None),
    "bright_fine_per_mpx": dict(
        name="Bright-fine count density",
        unit="per Mpx", group="B. Silicon particles",
        measures="Number of ambiguous bright objects per million pixels.",
        method="Count of bright_fine objects.",
        dfn=None, warns=None, caveat=None),
    "si_d10_num_um": dict(
        name="Si particle diameter, 10th pct (count-weighted)",
        unit="um", group="B. Silicon particles",
        measures="Small end of the silicon size distribution.",
        method="10th percentile of equivalent diameters.",
        dfn=None, warns=None, caveat=None),
    "si_d90_num_um": dict(
        name="Si particle diameter, 90th pct (count-weighted)",
        unit="um", group="B. Silicon particles",
        measures="Large end of the size distribution counting all "
                 "particles equally.",
        method="90th percentile of equivalent diameters.",
        dfn=None, warns=None, caveat=None),
    "si_dmean_aw_um": dict(
        name="Si particle diameter, mean (area-weighted)",
        unit="um", group="B. Silicon particles",
        measures="Average size where the silicon mass sits.",
        method="Area-weighted mean of equivalent diameters.",
        dfn="(basis for) Secondary particle radius",
        warns=None, caveat=None),
    "si_nn_dist_um": dict(
        name="Si nearest-neighbour distance",
        unit="um", group="B. Silicon particles",
        measures="Mean distance to the closest silicon particle.",
        method="KD-tree nearest neighbour on particle centroids.",
        dfn=None, warns="Low -> tight packing.", caveat=None),
    "pore_chord_h_um": dict(
        name="Pore chord, horizontal",
        unit="um", group="D. Pores & transport",
        measures="Mean uninterrupted pore run along image rows.",
        method="Line-intercept (chord) statistics.",
        dfn=None, warns=None, caveat=None),
    "pore_chord_v_um": dict(
        name="Pore chord, vertical",
        unit="um", group="D. Pores & transport",
        measures="Mean uninterrupted pore run along image columns.",
        method="Line-intercept (chord) statistics.",
        dfn=None, warns=None, caveat=None),
    "img_p1": dict(
        name="Raw intensity, 1st pct", unit="gray",
        group="F. Imaging guards", measures="Dark end of the raw "
        "histogram.", method="Percentile of raw pixels.",
        dfn=None, warns=None, caveat="Guard rail."),
    "img_p50": dict(
        name="Raw intensity, median", unit="gray",
        group="F. Imaging guards", measures="Middle of the raw histogram.",
        method="Percentile of raw pixels.",
        dfn=None, warns=None, caveat="Guard rail."),
    "img_p99": dict(
        name="Raw intensity, 99th pct", unit="gray",
        group="F. Imaging guards", measures="Bright end of the raw "
        "histogram.", method="Percentile of raw pixels.",
        dfn=None, warns=None, caveat="Guard rail."),
    "has_inlens": dict(
        name="InLens channel present", unit="0/1",
        group="F. Imaging guards", measures="Secondary-electron image "
        "exists for this location.", method="File existence check.",
        dfn=None, warns=None, caveat="Guard rail."),
    "has_etd_or_se": dict(
        name="ETD/SE channel present", unit="0/1",
        group="F. Imaging guards", measures="Secondary detector image "
        "exists.", method="File existence check.",
        dfn=None, warns=None, caveat="Guard rail."),
}


def card(name: str) -> dict:
    return CARDS.get(name, dict(name=name, unit="?", group="Other",
                                measures="", method="", dfn=None,
                                warns=None, caveat=None))
