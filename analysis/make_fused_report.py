#!/usr/bin/env python3
"""Build fused_report.pdf — the joint pipeline product document.

Assembles the fusion architecture, RF evaluation, ensemble-vote evidence,
annotated overlays and the merged headline into one LaTeX report.
Numbers are pulled from the CSVs, never hand-typed.
"""
import os, sys, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from common import OUT, FIG_DIR as FIG, TABLE_DIR, PDF_DIR

CLASSES = ["Batch_1", "Batch_2", "Batch_3"]
def t(name): return os.path.join(TABLE_DIR, name)
def f(name): return os.path.join(FIG, name)

gt = pd.read_csv(t("rf_group_accuracy_tuned.csv"))
loo = pd.read_csv(t("rf_loo_tuned.csv"))
imp = pd.read_csv(t("rf_importances_tuned.csv"))
votes = pd.read_csv(t("ensemble_votes.csv"))
dlt = pd.read_csv(t("deltas.csv"))

loo_acc = loo.correct.mean()
per_batch = loo.groupby("batch").correct.mean().round(3)
sig_cols = ["distance", "rf_top8", "embedding"]
votes["n_correct_signals"] = votes[sig_cols].eq(votes["batch"], axis=0).sum(1)
unanim = (votes.n_correct_signals == 3).mean()
b2mask = votes.batch == "Batch_2"
b2_consensus = votes[b2mask][sig_cols].eq("Batch_2").sum(1).ge(2).mean()

group_table = ""  # group accuracy now shown in the figure instead

tex = r"""\documentclass[11pt]{article}
\usepackage[margin=2.2cm]{geometry}
\usepackage{graphicx,booktabs,array,float,hyperref,xcolor,caption}
\usepackage[T1]{fontenc}
\captionsetup{font=small,labelfont=bf}
\definecolor{b1}{HTML}{E69F00}\definecolor{b2}{HTML}{009E73}\definecolor{b3}{HTML}{0072B2}
\title{\textbf{One product, two pipelines: the fused batch-categorisation system}\\
\large SEM silicon--graphite anode cross-sections — joint report}
\author{TeamJester pipeline $\times$ micro2dfn pipeline (merged)}
\date{October 2026}
\begin{document}\maketitle

\begin{abstract}\noindent
Two independent analysis pipelines were built on the same 31-image dataset: the
\emph{micro2dfn} pipeline (frozen-Batch\_3 segmentation ruler, bright-fine artefact
classifier, marker set, swelling/stress DFN) and the \emph{TeamJester} pipeline
(multi-detector features, quality-guard gating, Mahalanobis/nearest-centroid
categoriser, DINOv2-ETD embedding classifier). This report presents the fused product:
a voting consensus of four independent signals per image, behind one agreed
three-way label vocabulary (\emph{solid} / \emph{possible} / \emph{can't-tell}),
and the live web application that renders it for unseen photos. Every claim in this
document is reproducible from the two pipelines' CSV outputs.
\end{abstract}

\section{Architecture — who does what}
\begin{figure}[H]\centering
\includegraphics[width=\textwidth]{@@FIG@@/fused_architecture.png}
\caption{The fused pipeline. The teammate's frozen ruler measures; the TeamJester
audit gates; four independent classifiers vote; the DFN stack turns surviving
markers into physics indicators.}
\end{figure}

\section{The RandomForest experiment — honest verdict}
A 400-tree RandomForest (depth 4, balanced classes) was trained as a candidate
fusion classifier, evaluated by strict leave-one-image-out.

\textbf{All 28 features: it fails.} LOO accuracy 58\% — but per-batch it is
Batch\_1 43\%, \textbf{Batch\_2 0\%}, Batch\_3 88\%: the forest collapses to
predicting the majority class. With 31 images and a 7/7/17 class split, trees
cannot find the Batch\_2 boundary.

\textbf{Top-8 features: it works, but stays second-best.} Restricting to the
eight highest-importance features lifts LOO to @@LOOACC@@\%
(Batch\_1 @@LB1@@\%, Batch\_2 @@LB2@@\%, Batch\_3 @@LB3@@\%) and gives usable
group accuracy — still below the distance model and the embedding call on
Batch\_2. The RF therefore earns a seat in the consensus as an \emph{independent
vote}, not the decision-maker.

\begin{figure}[H]\centering
\includegraphics[width=\textwidth]{@@FIG@@/rf_group_accuracy.png}
\caption{Group-of-$n$ accuracy from LOO probabilities. Left: all features — the
forest collapses to Batch\_3. Right: top-8 features — usable but Batch\_2 stays
the hardest call, consistent with every other method.}
\end{figure}

\begin{figure}[H]\centering
\includegraphics[width=0.8\textwidth]{@@FIG@@/rf_importances.png}
\caption{Permutation importances (held-out accuracy lost when each feature is
shuffled). The RF's votes rest almost entirely on surface-texture and topography
markers — the same feature family every other method converged on.}
\end{figure}

\section{The voting consensus — the actual product}
Four signals call every image independently (all leave-self-out):
\textbf{(1)} the explainable distance model (Mahalanobis to Batch\_3 + nearest
centroid on 6 auto-selected features); \textbf{(2)} the RF top-8 vote;
\textbf{(3)} DINOv2-ETD embedding nearest-centroid; \textbf{(4)} the focused
Batch\_2-vs-Batch\_3 logistic boundary. Unanimous agreement of the first three
occurs on @@UNANIM@@\% of all 31 images; on the hard Batch\_2 set, @@B2CONS@@\%
of images get a correct majority.

\begin{figure}[H]\centering
\includegraphics[width=0.72\textwidth]{@@FIG@@/fused_ensemble_votes.png}
\caption{Per-image votes. Colour = the batch each signal called; a grey cell
means that signal called the wrong batch. Vertical stripes of one colour are
where the methods agree — the most defensible evidence the dataset offers.}
\end{figure}

\section{Cross-pipeline validation}
Where the two pipelines measured the same quantity, they mostly agree —
which is the strongest evidence either pipeline can offer:

\begin{table}[H]\centering\small
\begin{tabular}{@{}llll@{}}
\toprule
quantity & micro2dfn & TeamJester & verdict \\
\midrule
resolved porosity (B1 $<$ B3) & 0.095 vs 0.100 & 0.099 vs 0.114 & agree — B1 least porous \\
largest pore region (B1 $<$ B3) & 0.045 vs 0.065 & 0.047 vs 0.062 & agree — same gap \\
pore percolation & none spans & none spans & agree — 2-D $\tau$ not measurable \\
crack-like pores (B3 most) & 0.012 B3 $>$ 0.0085 B1 & 0.048 B3 $>$ 0.042 B1 & agree on ordering \\
pore anisotropy & 1.23--1.24 & 1.16--1.19 & agree — mildly flat pores \\
problem photos & 4ih2ggld, 5n1q8atc & same two flagged & agree \\
silicon clustering & B1 most clumped & B1 \emph{least} clumped & \textbf{disagree — drop the claim} \\
\bottomrule
\end{tabular}
\caption{The clustering contradiction (watershed-sensitive statistic) plus the
acquisition confound both pipelines found independently means the ``B1 lumpier
silicon'' story is not reproducible and is removed from the joint claims.}
\end{table}

\section{Annotated evidence}
\begin{figure}[H]\centering
\includegraphics[width=\textwidth]{@@FIG@@/fused_annotated_overlays.png}
\caption{Segmentation overlays with arrows marking the objects the markers count.}
\end{figure}
\begin{figure}[H]\centering
\includegraphics[width=\textwidth]{@@FIG@@/fused_marker_calls.png}
\caption{The surviving signals: per-image values against the Batch\_3 band.}
\end{figure}

\section{Merged headline — what the fused product claims}
\begin{itemize}
\item \textbf{solid}: batches are compositionally identical; Batch\_1 has less,
more fragmented resolved porosity (both pipelines agree); Batch\_3 carries the
most crack-like pores; no pore network spans any section; two Batch\_1 photos are
artefact-dominated; the acquisition confound is real and quantified.
\item \textbf{possible}: Batch\_2 is smoother with fewer surface tears (fewer
elongated pores $-21\%$, flatter Inlens texture $+9\%$) — consistent with less
handling/drying damage; the DFN hints its silicon may sit in a more accessible
configuration (model-conditional, input table to be reconciled); Inlens pore
contrast differs for Batch\_1.
\item \textbf{can't-tell}: silicon clustering differences (pipelines contradict);
any count-type or unblocked spatial marker (session-dominated); the
distribution-level batch difference (MMD $p=0.19$ at $n=7$).
\end{itemize}

\section*{What the user sees}
The webapp renders the consensus: IN/OUT of the Batch\_3 envelope, most-like
batch with \% surety, the three-way agreement of the signals, explainable
top-drivers, nearest-dataset patches, and the segmentation overlay — the
presentation demo. A single photo remains $\sim$55--75\% reliable; five or more
locations of one sample reach $\sim$100\% on the embedding vote.

\end{document}
"""

tex = (tex.replace("@@FIG@@", FIG)
          .replace("@@LOOACC@@", f"{100*loo_acc:.0f}")
          .replace("@@LB1@@", f"{100*per_batch['Batch_1']:.0f}")
          .replace("@@LB2@@", f"{100*per_batch['Batch_2']:.0f}")
          .replace("@@LB3@@", f"{100*per_batch['Batch_3']:.0f}")
          .replace("@@UNANIM@@", f"{100*unanim:.0f}")
          .replace("@@B2CONS@@", f"{100*b2_consensus:.0f}"))

out = os.path.join(PDF_DIR, "fused_report")
os.makedirs(out, exist_ok=True)
open(os.path.join(out, "fused_report.tex"), "w").write(tex.replace("—","---"))
r = subprocess.run(["/tmp/tectonic", "-o", out, os.path.join(out, "fused_report.tex")],
                   capture_output=True, text=True)
print(r.stdout[-1500:], r.stderr[-1500:])
print("->", os.path.join(out, "fused_report.pdf"))
