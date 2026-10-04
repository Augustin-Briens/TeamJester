"""Explainable outputs: report.md, data dictionary, Excel, figures.

Design rule: every number in the report chains back to a marker -> formula
-> source image -> overlay. Nothing is a black box.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .marker_cards import CARDS, card

KEY_MARKERS = [
    "pore_frac_resolved", "si_frac_total", "bright_fine_frac",
    "si_d50_aw_um", "si_d90_aw_um", "si_radius_3d_um", "si_sv_um_inv",
    "si_clustering_R", "gr_radius_eff_um", "gr_chord_ratio_xz",
    "pore_anisotropy", "crack_frac_v", "percolation_reach_2d",
    "pore_largest_region_frac", "bruggeman_b_eff",
    "si_pore_dist_mean_um", "si_lowcoverage_share",
]

INDICATORS = [
    ("discharge_cap_cycle1_Ah", "Cycle-1 discharge capacity [A.h]"),
    ("retention_pct", "Capacity retention first->last cycle [%] "
                      "(diagnostic only — not compared)"),
    ("loss_li_inventory_pct", "Loss of lithium inventory [%] "
                              "(diagnostic only — not compared)"),
    ("mean_coulombic_eff", "Mean coulombic efficiency"),
    ("thickness_change_um", "Peak cell thickness change during "
                            "discharge [um]"),
    ("si_stress_MPa", "Peak Si surface stress during discharge [MPa]"),
    ("min_neg_surface_v", "Min negative-electrode surface potential [V]"),
    ("np_ratio", "Usable N/P ratio"),
]


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------
def batch_summary(markers: pd.DataFrame) -> pd.DataFrame:
    """Per batch x marker: n, median, IQR, 95% CI (bootstrap on images)."""
    rng = np.random.default_rng(0)
    rows = []
    for batch, sub in markers.groupby("batch"):
        for col in KEY_MARKERS:
            if col not in sub:
                continue
            v = sub[col].replace([np.inf, -np.inf], np.nan
                                 ).dropna().to_numpy()
            if len(v) == 0:
                continue
            boots = [rng.choice(v, len(v), replace=True).mean()
                     for _ in range(2000)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            rows.append({"batch": batch, "marker": col, "n": len(v),
                         "median": np.median(v),
                         "iqr_lo": np.percentile(v, 25),
                         "iqr_hi": np.percentile(v, 75),
                         "ci95_lo": lo, "ci95_hi": hi})
    return pd.DataFrame(rows)


def batch_deltas(markers: pd.DataFrame, n_perm: int = 20000,
                 fdr: float = 0.10) -> pd.DataFrame:
    """Pairwise batch differences: median diff + two-sided permutation
    p-value + minimum detectable difference (95th pct of the null),
    Benjamini–Hochberg across the whole tested family.

    `mdd` = smallest |median diff| distinguishable at ~5% two-sided:
    differences below it are invisible at these sample sizes.
    Session caveat: unblocked only — the within-session stratified test
    lives in build_report.py (blocked_permutation.csv).
    """
    rng = np.random.default_rng(0)
    batches = sorted(markers["batch"].unique())
    rows = []
    for i, b1 in enumerate(batches):
        for b2 in batches[i + 1:]:
            for col in KEY_MARKERS:
                if col not in markers:
                    continue
                v1 = markers.loc[markers["batch"] == b1, col].replace(
                    [np.inf, -np.inf], np.nan).dropna().to_numpy()
                v2 = markers.loc[markers["batch"] == b2, col].replace(
                    [np.inf, -np.inf], np.nan).dropna().to_numpy()
                if len(v1) < 3 or len(v2) < 3:
                    continue
                obs = float(np.median(v1) - np.median(v2))
                pooled = np.concatenate([v1, v2])
                n1 = len(v1)
                idx = rng.permuted(np.tile(
                    np.arange(len(pooled)), (n_perm, 1)), axis=1)
                stat = (np.median(pooled[idx[:, :n1]], axis=1)
                        - np.median(pooled[idx[:, n1:]], axis=1))
                p = float((1 + (np.abs(stat) >= abs(obs)).sum())
                          / (n_perm + 1))
                rows.append({"pair": f"{b1} - {b2}", "marker": col,
                             "median_diff": obs, "p_perm": p,
                             "mdd": float(np.percentile(
                                 np.abs(stat), 95))})
    out = pd.DataFrame(rows)
    if len(out):
        o = out.p_perm.argsort().to_numpy()
        q = np.empty(len(out))
        run_min = 1.0
        for rank in range(len(out) - 1, -1, -1):
            idx = o[rank]
            run_min = min(run_min,
                          out.p_perm.iloc[idx] * len(out) / (rank + 1))
            q[idx] = run_min
        out["q_bh"] = np.clip(q, 0, 1)
        out["clear"] = out.q_bh < fdr
    return out


def write_data_dictionary(markers: pd.DataFrame, out_dir: str) -> str:
    rows = []
    for col in markers.columns:
        c = card(col)
        rows.append({
            "column": col,
            "plain_name": c["name"],
            "group": c["group"],
            "unit": c["unit"],
            "what_it_measures": c["measures"],
            "how_it_is_measured": c["method"],
            "feeds_dfn_parameter": c["dfn"] or "",
            "high_value_warns": c["warns"] or "",
            "caveat": c["caveat"] or "",
        })
    path = os.path.join(out_dir, "data_dictionary.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def fig_marker_distributions(markers: pd.DataFrame, out_dir: str) -> str:
    # drop markers with no finite data (e.g. tau_fdm_* when resolved
    # pores never span the section)
    cols = [m for m in KEY_MARKERS
            if np.isfinite(markers[m].replace([np.inf, -np.inf], np.nan)
                           ).any()]
    n = len(cols)
    ncols = 5
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.6 * ncols, 3 * nrows))
    axes = np.atleast_1d(axes).ravel()
    batches = sorted(markers["batch"].unique())
    palette = {"Batch_1": "#d62728", "Batch_2": "#1f77b4",
               "Batch_3": "#2ca02c", "Batch_4": "#9467bd"}
    for ax in axes[n:]:
        ax.axis("off")
    for ax, colm in zip(axes, cols):
        for i, b in enumerate(batches):
            v = markers.loc[markers["batch"] == b, colm].replace(
                [np.inf, -np.inf], np.nan).dropna()
            if len(v) == 0:
                continue
            ax.scatter(np.full(len(v), i) + np.random.default_rng(0)
                       .normal(0, 0.06, len(v)), v, s=12,
                       color=palette.get(b, "k"), alpha=0.7)
            ax.hlines(v.median(), i - 0.25, i + 0.25,
                      color=palette.get(b, "k"), lw=2)
        ax.set_title(card(colm)["name"], fontsize=8)
        ax.set_xticks(range(len(batches)))
        ax.set_xticklabels([b.replace("Batch_", "B") for b in batches],
                           fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle("Structural markers per batch (dots = images, "
                 "bar = median)", fontsize=11)
    fig.tight_layout()
    path = os.path.join(out_dir, "figs", "fig_marker_distributions.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_classification_overlay(raw, si_part, fine, pore,
                               title: str, out_path: str) -> str:
    """RGB overlay: red=si_particle, yellow=bright_fine, blue=pore."""
    rgb = (np.clip(raw / np.percentile(raw, 99.5), 0, 1)[..., None]
           * np.array([0.35, 0.35, 0.35])).astype(np.float64)
    rgb[..., 0] = np.maximum(rgb[..., 0], si_part)          # red
    rgb[..., 1] = np.maximum(rgb[..., 1], si_part * 0.15)
    rgb[..., 0] = np.maximum(rgb[..., 0], fine)             # yellow
    rgb[..., 1] = np.maximum(rgb[..., 1], fine)
    rgb[..., 2] = np.maximum(rgb[..., 2], pore)             # blue
    g = max(1, int(np.ceil(raw.shape[1] / 1400)))
    plt.imsave(out_path, rgb[::g, ::g].clip(0, 1))
    return out_path


def fig_consequence_bands(summaries: pd.DataFrame, out_dir: str) -> str:
    fig, axes = plt.subplots(1, len(INDICATORS), figsize=(18, 4))
    batches = summaries["batch"].tolist()
    for ax, (col, label) in zip(axes, INDICATORS):
        med = summaries[col + "_med"]
        lo = summaries[col + "_lo"]
        hi = summaries[col + "_hi"]
        ax.errorbar(range(len(batches)), med,
                    yerr=[med - lo, hi - med], fmt="o-", capsize=5)
        ax.set_xticks(range(len(batches)))
        ax.set_xticklabels(batches, rotation=30, fontsize=8)
        ax.set_title(label, fontsize=9)
        ax.grid(alpha=0.3)
    fig.suptitle("DFN consequence indicators — median with marker-"
                 "uncertainty band (indicators, not cell predictions)",
                 fontsize=11)
    fig.tight_layout()
    path = os.path.join(out_dir, "figs", "fig_consequence_bands.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_tornado(sweep_all: pd.DataFrame, out_dir: str) -> str | None:
    """Sensitivity: rank correlation of each sweep param vs the compared
    indicator with the most headroom (peak Si stress)."""
    ok = sweep_all[sweep_all["converged"] == 1]
    if len(ok) < 5:
        return None
    from scipy.stats import spearmanr
    target = "si_stress_MPa" if "si_stress_MPa" in ok \
        else "discharge_cap_last_Ah"
    rows = []
    for p in ("porosity_total", "si_active_frac", "si_radius_m",
              "si_youngs_pa", "si_nu", "bruggeman", "thickness_m"):
        if p not in ok:
            continue
        r, _ = spearmanr(ok[p], ok[target])
        rows.append((p, abs(r), np.sign(r)))
    rows.sort(key=lambda t: -t[1])
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.barh([r[0] for r in rows],
            [r[1] * r[2] for r in rows],
            color=["#d62728" if r[2] > 0 else "#1f77b4" for r in rows])
    ax.set_xlabel(f"Spearman corr. with {target}  "
                  "(+ increases, - decreases)")
    ax.set_title("What drives the compared indicator — sensitivity")
    fig.tight_layout()
    path = os.path.join(out_dir, "figs", "fig_tornado.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Excel workbook
# ---------------------------------------------------------------------------
def write_excel(out_dir: str, markers: pd.DataFrame,
                summary: pd.DataFrame, deltas: pd.DataFrame,
                sweep_all: pd.DataFrame | None,
                dfn_summary: pd.DataFrame | None) -> str:
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill
    from openpyxl.utils.dataframe import dataframe_to_rows

    wb = Workbook()
    header_font = Font(bold=True)
    header_fill = PatternFill("solid", fgColor="DDEBF7")

    def add_df(ws, df):
        for r in dataframe_to_rows(df, index=False, header=True):
            ws.append(r)
        for c in ws[1]:
            c.font = header_font
            c.fill = header_fill
        for col_cells in ws.columns:
            w = max((len(str(c.value)) for c in col_cells), default=8)
            ws.column_dimensions[col_cells[0].column_letter].width = \
                min(w + 2, 60)
        ws.freeze_panes = "A2"

    ws = wb.active
    ws.title = "Batch summary"
    add_df(ws, summary)

    ws = wb.create_sheet("Markers (per image)")
    add_df(ws, markers.round(5))

    ws = wb.create_sheet("Batch differences")
    add_df(ws, deltas)

    if dfn_summary is not None and len(dfn_summary):
        ws = wb.create_sheet("DFN indicators")
        add_df(ws, dfn_summary.round(6))
    if sweep_all is not None and len(sweep_all):
        ws = wb.create_sheet("DFN sweep runs")
        add_df(ws, sweep_all.round(6))

    ws = wb.create_sheet("Dictionary")
    dict_rows = []
    for col in markers.columns:
        c = card(col)
        dict_rows.append({"column": col, "plain_name": c["name"],
                          "unit": c["unit"], "group": c["group"],
                          "measures": c["measures"], "method": c["method"],
                          "dfn_input": c["dfn"] or "",
                          "warns": c["warns"] or "",
                          "caveat": c["caveat"] or ""})
    add_df(ws, pd.DataFrame(dict_rows))

    path = os.path.join(out_dir, "micro2dfn_results.xlsx")
    wb.save(path)
    return path


# ---------------------------------------------------------------------------
# report.md
# ---------------------------------------------------------------------------
def write_report(out_dir: str, markers: pd.DataFrame,
                 summary: pd.DataFrame, deltas: pd.DataFrame,
                 dfn_summary: pd.DataFrame | None,
                 t_pore: float, t_si: float, t_core: float,
                 settings: dict | None = None,
                 paired: pd.DataFrame | None = None) -> str:
    L = []
    A = L.append
    batches = sorted(markers["batch"].unique())

    A("# micro2dfn — microstructure markers & DFN consequence report\n")
    A("**What this is.** SEM cross-section photos of a silicon–graphite "
      "battery anode were colour-coded into three materials — pores "
      "(black), graphite (grey), silicon (bright) — using **one shared "
      "set of brightness cutoffs** for every photo, so no batch can "
      "re-calibrate itself into looking normal. From those maps we "
      "measured ~30 structural markers per image, then fed them into the "
      "standard PyBaMM Doyle–Fuller–Newman battery model as consequence "
      "*indicators*.\n")
    A("**Honesty rules.** Numbers are reported as batch *differences* "
      "with 95% CIs — the defensible signal. Absolute DFN outputs carry "
      "stated assumptions and are illustrative. Nothing here predicts "
      "cycle life or guarantees failure. Every value traces back to a "
      "marker, a formula, and an overlay image.\n")

    A("## Photo-quality check\n")
    probs = markers[markers["is_problem_photo"] == 1]
    if len(probs):
        A(f"**{len(probs)} low-contrast photos flagged** — silicon may be "
          "over-counted in them (the object classifier separates "
          "classified Si particles from uncertain-bright material; "
          "check the overlays):\n")
        for _, r in probs.iterrows():
            A(f"- `{r['batch']}/{r['image_id']}` "
              f"(si_bulk_contrast={r['si_bulk_contrast']:.2f}, "
              f"bright_fine_frac={r['bright_fine_frac']:.3f})")
    else:
        A("No low-contrast photos detected.")
    A("")

    A("## Segmentation settings (frozen)\n")
    prov = settings or {}
    A(f"- Shared thresholds: `t_pore={t_pore:.4f}`, `t_si={t_si:.4f}` — "
      f"fitted on the baseline batch **{prov.get('recipe_fitted_on', '?')}** "
      "only, saved in `recipe.json`, and reused verbatim on later runs. "
      "A newly added batch can never change how existing batches are "
      "measured; `run_manifest.json` records the recipe md5, code md5s "
      "and every input image md5 for this exact run.")
    A(f"- Si/bright-fine classifier cutoff: `t_core={t_core:.3f}` "
      "(5th percentile interior brightness of large, compact particles, "
      "calibrated on baseline objects only).")
    A(f"- Resolution floor: Si objects < 150 px and pores < 20 px "
      "(~0.35 um) are merged into bulk — same rule for every image.")
    if prov.get("reuse_cached_markers"):
        A("- **Cached measurements reused** (`--reuse`): markers and "
          "objects loaded from `markers_all.csv`/`objects_all.csv` "
          "written by the run recorded in `run_manifest.json`.")
    A("")

    A("## Marker summary per batch\n")
    A("Median [IQR] per batch. Full per-image values in "
      "`markers_<batch>.csv`; column meanings in `data_dictionary.csv`.\n")
    piv = summary.pivot(index="marker", columns="batch", values="median")
    iqr_lo = summary.pivot(index="marker", columns="batch",
                           values="iqr_lo")
    iqr_hi = summary.pivot(index="marker", columns="batch",
                           values="iqr_hi")
    A("| Marker | " + " | ".join(batches) + " |")
    A("|" + "---|" * (len(batches) + 1))
    for m in KEY_MARKERS:
        if m not in piv.index:
            continue
        c = card(m)
        cells = []
        for b in batches:
            if pd.notna(piv.loc[m, b]):
                cells.append(f"{piv.loc[m, b]:.4g} "
                             f"[{iqr_lo.loc[m, b]:.3g},"
                             f"{iqr_hi.loc[m, b]:.3g}]")
            else:
                cells.append("—")
        A(f"| {c['name']} | " + " | ".join(cells) + " |")
    A("")

    A("## Batch differences\n")
    clear = deltas[deltas["clear"]]
    if len(clear):
        A("**Differences surviving Benjamini–Hochberg (FDR 0.10)** "
          "across all tested pair–marker combinations (two-sided "
          "permutation test on the median, 20 000 resamples):\n")
        for _, r in clear.iterrows():
            A(f"- `{r['pair']}` — {card(r['marker'])['name']}: "
              f"**{r['median_diff']:+.4g}** "
              f"(p={r['p_perm']:.3g}, q={r['q_bh']:.3g}, "
              f"MDD={r['mdd']:.3g})")
    else:
        A("No pair–marker difference survives Benjamini–Hochberg "
          "correction (FDR 0.10).")
    n_nom = int((deltas["p_perm"] < 0.05).sum()) if len(deltas) else 0
    A(f"\nNominal (uncorrected) p<0.05 count: {n_nom} of "
      f"{len(deltas)} tests — compare against ~{0.05*len(deltas):.1f} "
      "expected by chance. `mdd` is the smallest median difference "
      "detectable at ~5%: smaller observed gaps are under-powered, not "
      "proven absent. These are UNBLOCKED tests — acquisition session "
      "can organise them; the stratified within-session test is in "
      "`build_report.py` (`blocked_permutation.csv`).")
    A("\nAll pairwise differences are in `batch_differences.csv`.\n")

    if dfn_summary is not None and len(dfn_summary):
        A("## DFN consequence indicators\n")
        # plain-English conclusion first: measured batch differences vs
        # the spread the assumptions create on their own
        if paired is not None and len(paired):
            capr = paired[paired["indicator"] ==
                          "discharge_cap_last_Ah"]
            n_meaningful = int((paired["verdict"] ==
                                "meaningful_difference").sum()) \
                if "verdict" in paired else 0
            n_below = int((paired["verdict"] ==
                           "below_threshold").sum()) \
                if "verdict" in paired else 0
            if len(capr):
                cspr = capr["assumption_only_spread_iqr"].iloc[0]
                cbits = "; ".join(
                    f"{r['pair']} {r['median_paired_diff']:+.3g} Ah"
                    for _, r in capr.iterrows())
                A("**Plain conclusion.** Paired same-seed comparisons "
                  "(same cathode at each sweep index for every batch) "
                  f"shift predicted cycle-5 discharge capacity by "
                  f"{cbits} — while the swept assumptions alone shift "
                  f"it by ~{cspr:.3g} Ah (pooled IQR). Judged against "
                  "the declared up-front thresholds and the paired "
                  "ranges: "
                  f"{n_meaningful} of {len(paired)} pair-indicator "
                  f"differences are meaningful, {n_below} sit fully "
                  "below threshold, the rest are inconclusive. "
                  "Lithium loss and capacity retention are reported "
                  "only as diagnostics — they cannot move measurably "
                  "in 5 cycles and are not part of the comparison. "
                  "Pore-count differences are image findings only — "
                  "they are not a DFN input and the model verdict "
                  "does not cover them. See `dfn_paired_differences."
                  "csv` for every verdict.\n")
        st = settings or {}
        A(f"PyBaMM {st.get('pybamm_version', '?')} DFN "
          "(composite Si/Gr anode, swelling + SEI + stress-driven LAM), "
          f"{st.get('cycles', '?')} x 1C cycles, "
          f"{st.get('sweep_points_per_variant', '?')} points spanning "
          "the marker uncertainty bands, downsample "
          f"x{st.get('downsample', 1)}, seed fixed "
          f"({st.get('seed', 0)}). Median [min, max] across runs of "
          "the primary `accessible` variant (Si share bracketed by "
          "classified-x-accessible to all-uncertain-bright). Full "
          "settings in `dfn_validation.md`.\n")
        A("| Indicator | " + " | ".join(dfn_summary["batch"]) + " |")
        A("|" + "---|" * (len(dfn_summary) + 1))
        for col, lab in INDICATORS:
            if col + "_med" not in dfn_summary:
                continue
            cells = []
            for _, r in dfn_summary.iterrows():
                m, lo, hi = r[col + "_med"], r[col + "_lo"], \
                    r[col + "_hi"]
                cells.append(f"{m:.3g} [{lo:.3g},{hi:.3g}]"
                             if np.isfinite(m) else "n/a")
            A(f"| {lab} | " + " | ".join(cells) + " |")
        A("")
        conv = dfn_summary[["batch", "n_runs", "n_converged"]]
        A("Convergence: " + "; ".join(
            f"{r['batch']} {r['n_converged']}/{r['n_runs']}"
            for _, r in conv.iterrows()) + " runs.")
        if "plating_flag_any" in dfn_summary:
            A("Plating (reversed rule): a run dipping below 0 V anode "
              "surface potential counts as a warning ONLY where the "
              "reference cell under the same protocol stays >= 0 V; "
              "if the reference also dips, sub-zero is a protocol/"
              "geometry artefact and is not attributed to any batch. "
              "Per-run warning counts are in `dfn_paired_differences."
              "csv` (`plating_warnings_*`); reference min = "
              f"{st.get('ref_min_neg_v', 'n/a')} V — see "
              "`dfn_validation.md`.\n")
        A("*These are model-predicted indicators, not cell-test data. "
          "Compare the bands between batches — do not read absolute "
          "values as predictions.*\n")

    A("## Assumed (not measured) parameters\n")
    A("| Parameter | Value | Why assumed |")
    A("|---|---|---|")
    A("| Electrode thickness | 75 um [65-85 swept] | foil & coating top "
      "not in frame |")
    A("| Sub-resolution porosity | +0.15-0.30 | pores below ~0.35 um are "
      "invisible |")
    A("| Si active share | classified Si x accessible share -> all "
      "uncertain bright counted | label uncertainty treated as an "
      "explicit bracket; 'all-Si-active' variant run separately |")
    A("| Si Young's modulus / Poisson ratio | 50 GPa [35-90] / 0.25 "
      "[0.22-0.28] | Bonkile2024 cited values, swept over range |")
    A("| Si mechanics block | Bonkile2024 | swappable — if EDS shows "
      "the bright phase is SiOx, refit (`_si_block` in simulate.py) |")
    A("| Bruggeman default | 1.5 | literature value when tau proxy "
      "degenerates |")
    A("| Cathode thickness | rescaled per run to hold reference N/P | "
      "keeps min-anode-potential meaningful under assumed geometry |")
    A("| Degradation constants | OKane2022 published (LAM 2.7778e-7 "
      "s-1, SEI k 1e-12 m/s) | audit table in `dfn_validation.md` |")
    A("| Other material constants | Chen2020_composite | "
      "diffusivities, kinetics, densities not image-measurable |\n")

    A("## Not measurable from these images\n")
    A("Binder distribution (no BSE contrast), SEI layer, lithium "
      "inventory, 3-D pore connectivity / electrolyte wetting "
      "(2-D sections only — see the optional reconstruction branch), "
      "foil corrosion.\n")
    A("**Note on tortuosity.** In every image the resolved pore network "
      "does not form a connected path across the section, so a "
      "diffusion-solve tortuosity cannot be measured on 2-D cuts — the "
      "through-plane network lives largely in sub-resolution pores. "
      "Directional markers (pore anisotropy, geodesic proxy, "
      "Bruggeman-equivalent exponent) carry the usable signal.\n")

    A("## How every number was produced\n")
    A("```\nBSE TIFF -> flat-field (Gaussian background)\n"
      "  -> shared multi-Otsu thresholds (frozen to thresholds.json)\n"
      "  -> watershed split + Si/bright-fine object classification\n"
      "  -> per-image stereological markers (this report)\n"
      "  -> per-batch median/IQR + bootstrap CIs\n"
      "  -> PyBaMM DFN sweep over marker bands -> indicators\n```\n")
    A("Files: `markers_*.csv` (per image), `objects_*.csv` (per Si "
      "object), `data_dictionary.csv` (every column explained), "
      "`dfn_params_*.json` (exact simulator inputs), `dfn_results.csv` "
      "(every sweep run), `figs/` (distributions, overlays, bands, "
      "tornado), `micro2dfn_results.xlsx` (all tables).\n")

    path = os.path.join(out_dir, "report.md")
    with open(path, "w") as fh:
        fh.write("\n".join(L))
    return path


def write_readme(out_dir: str) -> str:
    text = """# dfn_output/ — file inventory

| file | what it is |
|---|---|
| `report.md` | the readable report — start here |
| `micro2dfn_results.xlsx` | all tables in one Excel workbook |
| `markers_<batch>.csv` | one row per image, all structural markers |
| `objects_<batch>.csv` | one row per detected Si/bright-fine object |
| `data_dictionary.csv` | plain-English meaning of every column |
| `batch_summary.csv` | per batch x marker: n, median, IQR, 95% CI |
| `batch_differences.csv` | pairwise batch differences + CI |
| `thresholds.json` | the frozen segmentation cutoffs |
| `dfn_params_<batch>.json` | exact PyBaMM inputs per sweep point |
| `dfn_results.csv` | every simulation run's outputs |
| `figs/` | marker distributions, classification overlays, DFN bands, tornado |

Generated by `run_dfn.py` — one command regenerates everything.
"""
    path = os.path.join(out_dir, "README.md")
    with open(path, "w") as fh:
        fh.write(text)
    return path
