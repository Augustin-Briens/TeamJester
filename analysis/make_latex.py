"""Generate final_report.tex (full report, all figures/tables) and build it.

Usage: python3 make_latex.py            # writes + compiles via tectonic
"""
import os
import glob
import subprocess
import numpy as np
import pandas as pd
import common as C

T = C.TABLE_DIR
FIG = C.FIG_DIR
OVL = C.OVL_DIR
BUILD = os.path.join(C.OUT, "report_build")
os.makedirs(BUILD, exist_ok=True)


def esc(s):
    s = str(s)
    for a, b in (("\\", r"\textbackslash "), ("_", r"\_"), ("%", r"\%"),
                 ("&", r"\&"), ("#", r"\#"), ("~", r"\textasciitilde "),
                 ("^", r"\textasciicircum ")):
        s = s.replace(a, b)
    for a, b in (("µ", r"$\mu$"), ("×", r"$\times$"), ("±", r"$\pm$"),
                 ("≈", r"$\approx$"), ("≥", r"$\geq$"), ("≤", r"$\leq$"),
                 ("→", r"$\to$"), ("–", "--"), ("—", "---"),
                 ("°", r"$^\circ$"), ("₂", "2"), ("₃", "3")):
        s = s.replace(a, b)
    return s


def fnum(x, nd=3):
    try:
        x = float(x)
    except Exception:
        return esc(x)
    if not np.isfinite(x):
        return "--"
    if abs(x) >= 100:
        return f"{x:,.1f}"
    if abs(x) >= 1:
        return f"{x:.3f}".rstrip("0").rstrip(".")
    if x == 0:
        return "0"
    return f"{x:.4f}".rstrip("0")


def tab(df, cols, headers=None, fmt=None, size=r"\scriptsize",
        colspec=None, caption=None, label=None):
    headers = headers or cols
    colspec = colspec or "l" + "r" * (len(cols) - 1)
    lines = [f"{size}\n\\begin{{longtable}}{{{colspec}}}"]
    if caption:
        lines.append(f"\\caption{{{caption}}}\\label{{{label}}}\\\\")
    lines.append("\\toprule")
    lines.append(" & ".join(f"\\textbf{{{esc(h)}}}" for h in headers)
                 + r" \\" + "\n\\midrule\n\\endfirsthead")
    lines.append("\\toprule " + " & ".join(
        f"\\textbf{{{esc(h)}}}" for h in headers) + r" \\" +
        "\n\\midrule\n\\endhead")
    lines.append("\\midrule\\multicolumn{" + str(len(cols)) +
                 "}{r}{\\footnotesize cont.}\\\\\\midrule\\endfoot")
    lines.append("\\bottomrule\\endlastfoot")
    for _, r in df.iterrows():
        cells = []
        for j, c in enumerate(cols):
            v = r[c]
            if fmt and c in fmt:
                cells.append(fmt[c](v))
            elif j == 0:
                cells.append(esc(v))
            else:
                cells.append(fnum(v))
        lines.append(" & ".join(cells) + r" \\")
    lines.append("\\end{longtable}\n" + r"\normalsize")
    return "\n".join(lines)


def fig(path, caption, width="0.95\\textwidth", label=None):
    if not os.path.exists(path):
        return f"% missing figure {path}\n"
    p = os.path.relpath(path, BUILD)
    lab = f"\\label{{{label}}}" if label else ""
    return (f"\\begin{{figure}}[H]\\centering\n"
            f"\\includegraphics[width={width}]{{{p}}}\n"
            f"\\caption{{{caption}}}{lab}\n\\end{{figure}}\n")


ASSUME = ("the electrode is believed to be graphite (grey in BSE) with "
          "brighter silicon-based particles; black is pore or crack. This "
          "comes from image appearance and is unconfirmed.")


def load(name):
    p = os.path.join(T, name)
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()


def build():
    seg = load("segment_info.csv")
    sens = load("threshold_sensitivity.csv")
    half = load("half_agreement.csv")
    xdet = load("cross_detector.csv")
    q = load("quality_guards.csv")
    fmeta = load("feature_meta.csv")
    rep = load("repeatability.csv")
    dropped = load("dropped_features.csv")
    bs = load("baseline_stats.csv")
    cons = load("baseline_consistency.csv")
    D = load("deltas.csv")
    Dnf = load("deltas_noflag.csv")
    dinp = load("dfn_inputs.csv")
    dcap = load("dfn_capacity.csv")
    ddel = load("dfn_deltas.csv")
    dsw = load("dfn_thickness_sweep.csv")
    dass = load("dfn_assumptions.csv")
    loo_s = load("loo_summary.csv")
    conf = {v: load(f"loo_confusion_{v}.csv") for v in
            ("allfeatures", "safe", "safe_noflag")}
    need = load("images_needed.csv")
    need_s = load("images_needed_safe.csv")
    unseencsv = "/tmp/unseen_test/categorise_results.csv"
    unse = load(unseencsv) if os.path.exists(unseencsv) else pd.DataFrame()

    n_b = {b: int((seg.batch == b).sum()) for b in C.BATCHES}
    S = []  # story

    S.append(r"""
\begin{titlepage}
\centering
\vspace*{3cm}
{\LARGE\bfseries SEM Baseline--Delta Analysis\par}
\vspace{0.6cm}
{\Large Hackathon-Polaron electrode batches\par}
\vspace{1.2cm}
{\large Batch\_1 vs Batch\_2 against the Batch\_3 promised baseline\par}
\vspace{0.8cm}
{\large Full technical report --- all stages\par}
\vspace{2cm}
\begin{minipage}{0.85\textwidth}\small
\textbf{Standing assumption (every output):} """ + ASSUME + r"""
\end{minipage}
\vfill
{\large \today\par}
\end{titlepage}
\tableofcontents
\newpage
""")

    # ---------------- exec summary ----------------
    S.append(r"""
\section{Executive summary}
\begin{itemize}
\item \textbf{Batch\_1 is the outlier}, but its single strongest "difference"
(bulk BSE texture, +18\%, the only delta surviving Holm correction) tracks the
noise guard at $r{=}0.91$ --- it is an acquisition signature, not electrode
physics. The physically meaningful signals are pore\_frac $-$13.6\% and
large\_gap\_frac $-$71\% (both suggestive: bootstrap intervals exclude zero).
\item \textbf{Batch\_2 is close to baseline}: nothing survives Holm
correction; the best candidate is graphite structure-tensor coherence
($-$8.4\%, suggestive).
\item \textbf{Cause (inference, not proof)}: Batch\_1's pore chords shrink
$\sim$5\% in \emph{both} directions --- an isotropic shrink favours denser
particle packing over harder calendering, which would compress pores
vertically only.
\item \textbf{Categorisation}: single images are unreliable (leave-one-out
$\approx$45\% vs 37\% shuffled control). Groups of 5--7 images reach
$\sim$97--100\% for Batch\_1-like calls; \texttt{categorise.py} therefore
scores a folder as one group.
\item \textbf{DFN} (PyBaMM \texttt{BasicDFNComposite}, Chen2020): under
measured porosity the model predicts $\sim$3\% capacity loss for B1/B2 at 2C;
under the literature-corrected porosity (0.30) batches are indistinguishable.
Absolute values are unvalidated.
\end{itemize}
""")

    # ---------------- data ----------------
    S.append(r"""
\section{Data, pixel size, alignment}
\begin{itemize}
""" + f"""\\item {n_b['Batch_1']}/{n_b['Batch_2']}/{n_b['Batch_3']} imaged
locations (Batch\\_1/2/3), each with img\\_$<$id$>$\\_BSE.tif plus Inlens and
ETD (or SE) detectors.""" + r"""
\item Pixel size read from TIFF XResolution: \textbf{25.0 nm/px} for every
image (field of view $\sim$175$\times$54~µm).
\item The three detectors are pixel-aligned: normalised cross-correlation
peaks at zero shift, so Inlens dark pixels refine pores/cracks.
\item RGB TIFFs are converted to greyscale; the green channel is used (best
contrast across detectors).
\end{itemize}
""" + tab(seg[["batch", "image_id", "um_per_px", "t_lo", "t_bright",
               "pore_frac", "silicon_frac"]].round(4),
          ["batch", "image_id", "um_per_px", "t_lo", "t_bright",
           "pore_frac", "silicon_frac"],
          ["batch", "image", "µm/px", "t_pore", "t_bright",
           "pore frac", "Si frac"],
          caption="Per-image segmentation thresholds and phase fractions",
          label="tab:seginfo") + "\n\\newpage\n")

    # ---------------- phase 1 ----------------
    sB = sens[sens["param"] == "t_lo(pore)"].groupby(
        "shift")[["pore", "silicon", "graphite"]].mean()
    sT = sens[sens["param"] == "t_b(silicon)"].groupby(
        "shift")[["pore", "silicon", "graphite"]].mean()
    dB = abs(sB.loc[-0.1, "pore"] - sB.loc[0.0, "pore"])
    dBp = abs(sB.loc[0.1, "pore"] - sB.loc[0.0, "pore"])
    dTm = abs(sT.loc[-0.1, "silicon"] - sT.loc[0.0, "silicon"])
    dTp = abs(sT.loc[0.1, "silicon"] - sT.loc[0.0, "silicon"])
    ha = half.assign(
        pore_diff=(half.pore_L - half.pore_R).abs(),
        si_diff=(half.silicon_L - half.silicon_R).abs()).groupby(
        "batch")[["pore_diff", "si_diff"]].mean()
    S.append(r"""
\section{Phase 1 --- Segmentation and evidence of accuracy}
Method (identical for every image, no per-batch tuning): clip 1st--99.5th
percentile normalisation, mild Gaussian denoise, Otsu valley threshold for
pore (dark) and multi-Otsu for the bright silicon tail --- the bright tail
has no second histogram mode on any image, so the multi-Otsu fallback is used
everywhere (honest limitation); morphology cleanup; Inlens dark pixels (the
detectors are aligned) refine fine cracks.

\subsection{Validation}
""" + "\\begin{itemize}\n" + rf"""\item \textbf{{Threshold sensitivity}}
($\pm$10\% shift of both thresholds, averaged over all images): pore
threshold moves pore fraction by {dB:.4f}/{dBp:.4f} (down/up);
bright threshold moves silicon fraction by {dTm:.4f}/{dTp:.4f}.
\item \textbf{{Left/right repeatability}} (segment halves independently):
mean absolute pore-fraction difference = """ + ", ".join(
        f"{esc(b)}: {v:.4f}" for b, v in ha.pore_diff.items()) + r"""; silicon
""" + ", ".join(f"{esc(b)}: {v:.4f}" for b, v in ha.si_diff.items()) + r""".
\item \textbf{Cross-detector agreement}: mean pore-mask IoU BSE-vs-Inlens =
""" + f"{xdet.iou_pore.mean():.3f}" + r""", Dice = """ +
            f"{xdet.dice_pore.mean():.3f}" + r""" (Inlens sees finer cracks,
so partial agreement is expected).
\item \textbf{Point-count sheet}: 300 random locations, one crop each, empty
human column (in \texttt{outputs/pointcount/} +
\texttt{score\_pointcount.py}).
\end{itemize}

\subsection{Image-quality guards}
Absolute thresholds: Si/bulk contrast $<$2.8, noise-MAD $>$11.2, sharpness
$>$11.0, shading range $>$55 flag an image. Two Batch\_1 images are flagged;
all results are also reported without them.
""" + tab(q.sort_values(["batch", "image_id"]),
          ["batch", "image_id", "si_bulk_contrast", "noise_mad",
           "sharpness", "shading_range", "flagged"],
          caption="Quality guards per image (flagged = exceeded a guard)",
          label="tab:guards") + "\n")

    for b in C.BATCHES:
        S.append(fig(os.path.join(OVL, f"contact_{b}.png"),
                     f"{esc(b)}: segmentation contact sheet (pore=blue, "
                     f"silicon=red overlay on BSE).",
                     label=f"fig:contact{b}"))
    for iid in ("4ih2ggld", "5n1q8atc"):
        p = os.path.join(OVL, f"overlay_Batch_1_{iid}.png")
        if os.path.exists(p):
            S.append(fig(p, f"Flagged image Batch\\_1/{iid} overlay.",
                         label=f"fig:ovl{iid}"))
    S.append("\\newpage\n")

    # ---------------- phase 2 ----------------
    fm = fmeta.merge(rep[["feature", "lr_corr"]], on="feature",
                     how="left")
    kept = bs.feature.tolist()
    fm_k = fm[fm.feature.isin(kept)]
    S.append(r"""
\section{Phase 2 --- Feature dictionary}
Every feature is explainable in one sentence; references were actually opened
(citations in \texttt{features.xlsx}, FeatureDict sheet). Retained after
pruning (constant, left/right corr $<$0.5, or duplicate $|$r$|>$0.9):
""" + tab(fm_k[["feature", "group", "units", "detector", "definition",
                "meaning", "lr_corr"]],
          ["feature", "group", "units", "detector", "definition", "meaning",
           "lr_corr"],
          colspec="llllp{4.6cm}p{4.2cm}r",
          caption="Retained features with left/right repeatability",
          label="tab:featdict") + "\n" +
        tab(dropped, ["feature", "reason"],
            colspec="lp{10cm}", caption="Dropped features and why",
            label="tab:dropped") + "\n\\newpage\n")

    # ---------------- phase 3 ----------------
    S.append(r"""
\section{Phase 3 --- Batch\_3 baseline}
Statistics over 17 images; bootstrap CI on the mean (2{,}000 resamples);
envelope = 5th--95th percentile. Internal consistency uses robust z-scores
(median/MAD) per feature.
""" + tab(bs, ["feature", "mean", "median", "std", "p5", "p95",
               "ci95_lo", "ci95_hi"],
          caption="Batch\\_3 baseline statistics", label="tab:baseline") +
        "\nBatch\\_3 outlier images (robust z $>$3.5 on at least one "
        "feature): " + esc(", ".join(cons[cons.outlier].image_id)) +
        ". These are within-batch scatter, not exclusions.\n")
    for p in sorted(glob.glob(os.path.join(FIG, "features_*.png"))):
        if "_unseen" in p:
            continue
        S.append(fig(p, os.path.basename(p).replace("features_", "")
                     .replace(".png", "").replace("_", " ") +
                     " --- every image as a point against the Batch\\_3 "
                     "5--95\\% band.", label="fig:" +
                     os.path.basename(p)[:-4]))
    S.append(fig(os.path.join(FIG, "distributions.png"),
                 "Size and pore distributions overlaid per batch.",
                 label="fig:dists"))
    S.append(fig(os.path.join(FIG, "pore_depth.png"),
                 "Pore fraction depth profile across the field of view.",
                 label="fig:depth"))
    S.append("\\newpage\n")

    # ---------------- phase 4 ----------------
    di = dinp.groupby("batch")[["porosity", "gr_amf", "si_amf",
                                "gr_radius_m", "si_radius_m"]].mean(
        ).reset_index()
    di["gr_radius_m"] = (di.gr_radius_m * 1e6).round(3)
    di["si_radius_m"] = (di.si_radius_m * 1e6).round(3)
    di = di.rename(columns={"gr_radius_m": "gr_radius_um",
                            "si_radius_m": "si_radius_um"})
    S.append(r"""
\section{Phase 4 --- DFN simulation}
PyBaMM \texttt{BasicDFNComposite} + published \texttt{Chen2020\_composite}
(LGM50 NMC811 / graphite-SiO$_x$). Image-derived inputs per batch below;
everything else at published values and identical for all batches.
Uncertainty bands resample the image-derived inputs (30 draws). Porosity is
run twice: the measured visible value \emph{and} corrected to 0.30
(literature range for calendered anodes), because pores below the
$\sim$50~nm pixel size are unresolved. Absolute values are NOT validated ---
only batch-to-batch differences are meaningful; the model includes no
swelling, cracking or ageing.
""" + tab(di, ["batch", "porosity", "gr_amf", "si_amf", "gr_radius_um",
               "si_radius_um"], ["batch", "porosity", "graphite amf",
                                "silicon amf", "r_graphite µm",
                                "r_Si µm"],
          caption="Image-derived DFN inputs (batch means)",
          label="tab:dfninputs") + "\n" +
        tab(dass, ["parameter", "value", "note"], colspec="lp{5cm}p{5.5cm}",
            caption="DFN assumptions", label="tab:dfnass") + "\n" +
        tab(dcap, ["batch", "porosity_mode", "c_rate", "n", "cap_mean",
                   "cap_std", "cap_p5", "cap_p95"],
            caption="Discharge capacity (Ah) per batch / porosity mode / "
                    "C-rate --- 30 resamples", label="tab:dfncap") + "\n" +
        tab(ddel, ["batch", "porosity_mode", "c_rate", "delta_ah",
                   "pct_delta", "ci_lo", "ci_hi", "sig"],
            caption="DFN output deltas vs Batch\\_3 (bootstrap CI)",
            label="tab:dfndeltas") + "\n" +
        tab(dsw, ["thickness_m", "c_rate", "capacity_ah"],
            fmt={"thickness_m": lambda v: f"{float(v)*1e6:.0f} µm"},
            caption="Thickness sweep (Batch\\_3, corrected porosity)",
            label="tab:dfnsweep") + "\n" +
        fig(os.path.join(FIG, "dfn.png"),
            "Discharge curves with resample bands, capacity vs rate, and "
            "delta vs Batch\\_3.", label="fig:dfn") + "\n" +
        fig(os.path.join(FIG, "dfn_profiles.png"),
            "End-of-discharge electrolyte and particle concentration "
            "profiles (median + 5--95\\% band).", label="fig:dfnprof") +
        "\n\\newpage\n")

    # ---------------- phase 5 ----------------
    S.append(r"""
\section{Phase 5 --- Deltas vs Batch\_3}
Per feature: absolute/\%/standardised delta (B3 std units), bootstrap 95\%
CI, Mann--Whitney p, Holm-corrected p across all 42 tests, and the minimum
detectable difference at $n{=}7$. Classes: \textbf{different} (CI excludes 0
\emph{and} survives Holm), \textbf{suggestive} (CI excludes 0 only),
\textbf{equivalent} (CI inside $\pm$10\%), \textbf{undetermined}.
""")
    for b in ("Batch_1", "Batch_2"):
        Db = D[D.batch == b].sort_values("abs_std_delta", ascending=False)
        S.append(tab(Db,
                     ["feature", "delta", "pct_delta", "std_delta",
                      "ci_lo", "ci_hi", "p_mw", "p_holm", "mdd",
                      "classification"],
                     colspec="lrrrrrrrrl",
                     caption=f"{esc(b)} deltas vs Batch\_3 (ranked by "
                             "abs. std delta)", label=f"tab:delta{b}"))
        S.append(fig(os.path.join(FIG, f"ranked_deltas_{b}.png"),
                     f"{esc(b)}: ranked standardised deltas with intervals.",
                     label=f"fig:rank{b}"))
    S.append(r"""
\subsection{Without the two flagged Batch\_1 images}
""" + tab(Dnf[(Dnf.batch == "Batch_1") &
              (Dnf.feature.isin(["silicon_frac", "pore_frac",
                                 "large_gap_frac", "bse_bulk_texture"]))],
          ["feature", "pct_delta", "ci_lo", "ci_hi", "p_holm",
           "classification"],
          caption="Flag-sensitive features without the flagged images",
          label="tab:noflag") + "\n\\newpage\n")

    # ---------------- phase 6 ----------------
    S.append(r"""
\section{Phase 6 --- Categorising unseen samples}
\texttt{categorise.py} segments an unseen folder with the identical Phase~1
settings, computes the Phase~2 features, then scores the \textbf{group
median} on the 4--8 most repeatable artefact-safe features: a robust
Mahalanobis distance to the Batch\_3 envelope (threshold = 95th percentile
of a group-size-matched bootstrap of B3) decides IN/OUT; the nearest batch
centroid (with a softmax confidence) decides most-like. Top-3 drivers are
reported in plain words and a one-page PDF shows the overlay and the
sample's position on the key feature plots. Missing detectors degrade
gracefully (detector-specific features drop out of the model).
""")
    S.append(tab(loo_s, list(loo_s.columns),
                 colspec="lrrrrrr",
                 caption="Leave-one-out validation by feature variant "
                         "(plus shuffled-label control)",
                 label="tab:loo"))
    for v, ttl in (("allfeatures", "all retained features"),
                   ("safe", "artefact-safe features"),
                   ("safe_noflag", "artefact-safe, flagged images removed")):
        c = conf[v]
        if len(c):
            S.append(tab(c, list(c.columns),
                         caption=f"LOO confusion --- {ttl}",
                         label=f"tab:conf{v}"))
    S.append(tab(need, ["batch", "n_images", "group_accuracy"],
                 caption="Group-size calibration (all features)",
                 label="tab:need") +
             tab(need_s, ["batch", "n_images", "group_accuracy"],
                 caption="Group-size calibration (artefact-safe)",
                 label="tab:needsafe"))
    S.append(fig(os.path.join(FIG, "categorisation_map.png"),
                 "Categorisation map: every image in the robust-PCA feature "
                 "plane with the Batch\\_3 envelope; colour=true batch, "
                 "marker shape=LOO prediction.",
                 label="fig:catmap"))
    if len(unse):
        S.append(r"""
\subsection{End-to-end test on an unseen folder}
A copy of Batch\_2's images was passed through \texttt{categorise.py} as a
held-out sample (this also verifies the tool runs on a plain folder).
The group call was IN / most-like Batch\_3 --- consistent with Batch\_2's
genuine closeness to baseline; the per-image rows show why single-image
calls are unreliable.
""" + tab(unse,
          ["level", "image_id", "dist", "threshold", "call_inout",
           "most_like", "confidence"],
          caption="categorise\\_results.csv from the Batch\\_2-as-unseen "
                  "test", label="tab:unseen"))

    # ---------------- phase 7 ----------------
    S.append(r"""
\section{Phase 7 --- Why the differences exist (inference)}
Images show \emph{what} differs; they cannot prove cause. Candidate
mechanisms were scored by their predicted side-effects (e.g.\ harder
calendering should compress pores vertically --- anisotropic shrink --- and
align flakes; a different powder lot would shift silicon size; worse mixing
leaves larger voids).
\begin{itemize}
\item \textbf{Batch\_1}: pore chords shrink $\sim$5\% in both directions
(isotropic) and large gaps collapse $-$71\% while silicon size, flake
alignment and composition stay put. Harder pressing is rejected (predicts
anisotropic shrink and flake reorientation); \textbf{denser packing /
less mixing damage is the best-supported explanation}, confidence medium,
an inference. Both flagged images were excluded before this verdict.
\item \textbf{Batch\_2}: only suggestive, weaker signals (graphite
structure-tensor coherence $-$8.4\%); consistent with ordinary
batch-to-batch scatter; no mechanism claimable.
\item \textbf{Artefact check}: the strongest deltas (BSE texture, ETD
roughness) correlate with the noise guard at $r{>}0.85$ --- acquisition
signatures, reported via the artefact-safe variant, not interpreted
physically.
\end{itemize}
""")
    for p in sorted(glob.glob(os.path.join(FIG, "evidence_*.png"))):
        S.append(fig(p, os.path.basename(p).replace("evidence_", "")
                     .replace(".png", "").replace("_", " ") +
                     " --- side-by-side evidence (left: baseline, right: "
                     "batch) with marked regions.",
                     label="fig:" + os.path.basename(p)[:-4]))

    # ---------------- limits ----------------
    S.append(r"""
\section{Limitations and assumptions}
\begin{itemize}
\item """ + ASSUME + r"""
\item Bright tail has no separate histogram mode on any image --- the
silicon threshold is a multi-Otsu fallback, honest but calibrated to the
tail shape rather than a valley.
\item Visible porosity ($\sim$10\%) underestimates true porosity; DFN
results under corrected porosity 0.30 show the batch signal is much smaller
than the correction itself.
\item 2D sections cannot give 3D tortuosity at 9\% porosity --- reported as
proxy only and dropped for non-repeatability.
\item No binder class claimed: not separable at this resolution.
\item Batch sizes are small ($n{=}7$ for B1/B2): several real differences
may sit in `undetermined'. MDDs are listed per feature.
\end{itemize}

\subsection{References actually opened}
\begin{itemize}
\item Chen2020 composite parameter set, J.\ Electrochem.\ Soc.\ 167
080534 (IOP open access) --- DFN parameters.
\item Clark \& Evans 1954 --- nearest-neighbour (Clark--Evans) clustering
index with edge correction (Ecological Society of America page).
\item Torquato, Annu.\ Rev.\ Mater.\ Res.\ --- two-point correlation /
lineal-path statistical descriptors (Princeton PDF).
\item stereology.info --- point-count and stereological validation
methodology.
\item BoneJ thickness --- local-thickness pore sizing.
\item Wikipedia: tortuosity, structure tensor, power spectral density ---
definitions for transport/texture proxies.
\item Profatilova et al., ACS AEM 2020 --- paywalled; only abstract + SI
(figshare 13311418) used.
\end{itemize}
""")

    doc = (r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=1.9cm]{geometry}
\usepackage{graphicx,longtable,booktabs,xcolor,float,caption,hyperref}
\usepackage[T1]{fontenc}
\captionsetup{font=small,labelfont=bf}
\hypersetup{colorlinks=true,linkcolor=blue!40!black,urlcolor=blue!40!black}
\setlength{\parskip}{4pt}
\begin{document}
""" + "\n".join(S) + "\n\\end{document}\n")

    tex = os.path.join(BUILD, "final_report.tex")
    with open(tex, "w") as f:
        f.write(doc)
    return tex


if __name__ == "__main__":
    tex = build()
    subprocess.run(["/tmp/tectonic", tex, "--outdir", BUILD], check=True,
                   cwd=BUILD)
    print(os.path.join(BUILD, "final_report.pdf"))
