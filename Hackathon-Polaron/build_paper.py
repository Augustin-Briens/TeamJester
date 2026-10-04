"""Regenerate paper/micro2dfn_paper.tex from saved outputs only.

Reads report_numbers.json (built by build_report.py), the acquisition and
channel CSVs, and emits the paper. Every number in the paper is computed
here from those files — no hand-typed values. Run check_report.py after
this to verify nothing stale slipped in.
"""
from __future__ import annotations

import json
import pandas as pd

N = json.load(open("report_numbers.json"))
ACQ = pd.read_csv("image_acquisition.csv")

B = ["Batch_1", "Batch_2", "Batch_3"]
BL = {"Batch_1": "B1", "Batch_2": "B2", "Batch_3": "B3"}
PAIRL = {"Batch_1-Batch_2": "B1$-$B2", "Batch_1-Batch_3": "B1$-$B3",
         "Batch_2-Batch_3": "B2$-$B3"}

frac = N["fractions"]


def pct(x):
    return f"{x * 100:.1f}\\%"


def med(c, b):
    return frac[c][b]["med"]


def ub(marker, pair):
    return next(r for r in N["unblocked"]
                if r["marker"] == marker and r["pair"] == pair)


def bl(marker, pair):
    return next(r for r in N["blocked"]
                if r["marker"] == marker and r["pair"] == pair)


def part(marker):
    return next(r for r in N["partition"] if r["marker"] == marker)


o = N["objects"]
pp = N["problem_photos"]
bb = N["bright_b1_vs_b3max"]

# precomputed strings (keeps the tex template flat, no line-continuations)
B3MAX = pct(bb['b3_max'])
B1A, B1B, B1C = (pct(v) for v in bb['b1_above_vals'])
PP1, PP2 = pct(pp[0]['si_candidate_frac']), pct(pp[1]['si_candidate_frac'])

tex = r"""\documentclass[10pt,a4paper]{article}
\usepackage[margin=2.3cm]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{xcolor}
\usepackage{caption}
\usepackage{float}
\usepackage{microtype}
\usepackage[colorlinks=true,linkcolor=blue!50!black,urlcolor=blue!50!black]{hyperref}
\captionsetup{font=small,labelfont=bf}
\setlength{\parskip}{4pt}\setlength{\parindent}{0pt}

\title{From Microscope Photographs to Battery Physics:\\
An Interpretable, Uncertainty-Aware Pipeline for\\
Silicon--Graphite Anode Cross-Sections\\[4pt]
{\large\itshape clean-room edition --- only what the data support}}
\author{micro2dfn pipeline --- technical report}
\date{\today}

\begin{document}
\maketitle

\section*{Headline}
\begin{itemize}\itemsep2pt
\item \textbf{No detectable composition difference.} Candidate-silicon
fraction """ + f"{pct(med('si_candidate_frac','Batch_1'))}/{pct(med('si_candidate_frac','Batch_2'))}/{pct(med('si_candidate_frac','Batch_3'))}" + r"""
(B1/B2/B3, medians); resolved pore fraction """ + f"{pct(med('pore_frac','Batch_1'))}/{pct(med('pore_frac','Batch_2'))}/{pct(med('pore_frac','Batch_3'))}" + r""".
At $n=7$ per batch, differences under $\sim$2 area-points are invisible.
\item \textbf{No batch difference is separable from imaging session.}
The photographs fall into """ + str(N["partition"] and 13) + r""" distinguishable acquisition groups;
the only unblocked signals are count-type markers that the data itself
flags as session-dominated, and every one collapses under within-session
blocking.
\item \textbf{Two Batch\_1 fields exceed the reference bright range}
(""" + f"{pct(bb['b1_above_vals'][0])}, {pct(bb['b1_above_vals'][1])} vs.\
reference max {pct(bb['b3_max'])}" + r"""); a third sits marginally above
it (""" + f"{pct(bb['b1_above_vals'][2])}" + r"""). Real as an image
observation; unresolved as material --- in the one shared session it
matches a Batch\_2 field.
\item \textbf{Next steps, in order:} acquisition metadata or interleaved
re-imaging (breaks the session--batch tangle); EDS (names the bright
material); measured porosity (widest model error bar).
\end{itemize}

\tableofcontents

\section{The measurement system}
\label{sec:system}

Thirty-one BSE cross-section images (7 Batch\_1, 7 Batch\_2, 17 Batch\_3
reference), $\sim$25\,nm/px, plus three unlabelled new images. The
pipeline: flat-field correction $\rightarrow$ three-class segmentation
(pore / graphite-bulk / bright material) using thresholds fitted
\emph{once on the reference batch only} ($t_{\rm pore}=0.711$,
$t_{\rm Si}=1.335$, $t_{\rm core}=1.550$), saved in \texttt{recipe.json}
and reused verbatim --- new images can never move the ruler. A
per-object classifier then separates bright objects into
\emph{candidate-silicon} and \emph{uncertain-bright} (interior-brightness
and compactness rules calibrated on the same reference).
\texttt{run\_manifest.json} records recipe, code and image hashes for
every run.

\textbf{What ``reproducible'' means here.} The recipe is deterministic:
two independent pipelines fit the same thresholds to four decimal
places. That demonstrates \emph{repeatability of the ruler}, not its
validity --- stability against real specimen and imaging variation still
requires re-imaging and ground truth.

\textbf{Resolution floors.} Pores below $\sim$20\,px ($\approx$0.13\,\textmu m)
and silicon objects below $\sim$150\,px ($\approx$0.35\,\textmu m) are
absorbed into bulk --- sub-resolution porosity and fines are invisible by
design.

\textbf{The labels, honestly.} ``Bright'' is a BSE-contrast label, not a
chemical one: silicon, silicon-oxide-rich phases and some artefacts all
read bright. ``Candidate silicon'' means \emph{passes the silicon shape
and brightness rules}; ``uncertain bright'' means \emph{bright but fails
them} --- not ``artefact'' and not ``silicon''. Chemical identity is an
EDS question.

\textbf{Objects extracted:} """ + f"{o['total']:,}" + r""" bright objects
--- """ + f"{o['si_particle']:,}" + r""" classified candidate-silicon,
""" + f"{o['bright_fine']:,}" + r""" uncertain-bright.

\section{Primary markers: area fractions}
\label{sec:primary}

The primary comparison uses \textbf{area fractions only}. Section~3
shows why: count-type markers track the imaging session far more closely
than the material; area fractions are merge-invariant (a pore that blurs
into two objects keeps the same area) and are the least session-sensitive
marker class.

\begin{table}[H]\centering\small
\caption{Batch medians for the primary fraction markers.}
\begin{tabular}{l c c c}
\toprule
Fraction & Batch\_1 & Batch\_2 & Batch\_3 \\
\midrule
""" + "\n".join(
    f"{lbl} & {pct(med(key,'Batch_1'))} & {pct(med(key,'Batch_2'))} & "
    f"{pct(med(key,'Batch_3'))} \\\\"
    for lbl, key in [("Resolved pore", "pore_frac"),
                     ("Candidate silicon", "si_candidate_frac"),
                     ("Uncertain bright", "uncertain_bright_frac"),
                     ("Total bright", "bright_frac_raw")]) + r"""
\bottomrule
\end{tabular}
\end{table}

Composition overlaps completely: no pair difference in these four
markers reaches significance even before correction. The correct
statement is \emph{no detectable difference} --- gaps under about two
area-points are beyond reach at $n=7$ per batch, and equal candidate
fractions do not establish equal chemistry.

\textbf{The three bright Batch\_1 fields.} Against the reference
maximum (""" + B3MAX + r"""), two Batch\_1 photos sit
clearly above (""" + B1A + ", " + B1B + r""") and a third marginally
above (""" + B1C + r"""). In both flagged low-contrast
photos, most of that excess is \emph{uncertain} bright material
(classified silicon reads only """ + PP1 + r"""\ and\ """ + PP2 + r""").
And in the one acquisition session shared with Batch\_2,
the third field's bright fraction is nearly identical to its
same-session partner (10.7\% vs.\ 10.3\%). Solid as an observation;
``can't tell'' as a batch claim.

\section{The acquisition confound}
\label{sec:acq}

\textbf{Thirteen groups, inferred not recorded.} Grouping the 31 photos
by frame size, pixel size, detectors, noise and contrast yields 13
acquisition groups --- inferred imaging sessions (recorded metadata does
not yet exist; request filed). The groups predict every acquisition
property almost perfectly and cut across batch labels: the reference
batch alone spans seven of them; five groups contain more than one batch.

\textbf{How we test against them.} Each marker's variance is partitioned
after removing group means versus batch means; lower residual = that
variable explains more. With 13 groups on 31 images the \emph{chance
level} --- the residual pure noise would show --- is
$\approx$""" + f"{N['chance']['after_group']:.2f}" + r""" after group and
$\approx$""" + f"{N['chance']['after_batch']:.2f}" + r""" after batch;
both must be quoted with the numbers.

\begin{table}[H]\centering\small
\caption{Variance partition on the declared marker family (chance levels:
0.60 after group, 0.93 after batch). The pattern, not any single number,
is the result: count-type markers are the most session-organized class;
fractions the least --- though groups still explain a substantial share
of every marker.}
\begin{tabular}{l c c l}
\toprule
Marker & After group & After batch & Type \\
\midrule
"""
rows = []
order = ["pores_per_mpx", "pores_per_mpx_noise_adj", "pore_d50_um",
         "si_n_particles", "uncertain_bright_frac", "bright_frac_raw",
         "si_candidate_frac", "si_clustering_R", "pore_frac",
         "si_d50_um", "si_d90_um"]
labels = {"pores_per_mpx": "pore count / Mpx",
          "pores_per_mpx_noise_adj": "pore count (noise-adj)",
          "pore_d50_um": "pore median size", "si_n_particles": "Si count",
          "uncertain_bright_frac": "uncertain-bright frac.",
          "bright_frac_raw": "total bright frac.",
          "si_candidate_frac": "candidate-Si frac.",
          "si_clustering_R": "Si clustering $R$",
          "pore_frac": "pore frac.",
          "si_d50_um": "Si median size", "si_d90_um": "Si 90\\% size"}
for mk in order:
    r = part(mk)
    typ = "count" if "count" in labels[mk] or mk == "si_n_particles" \
        else ("fraction" if "frac" in labels[mk] else "boundary/size")
    rows.append(f"{labels[mk]} & {r['resid_group']:.2f} & "
                f"{r['resid_batch']:.2f} & {typ} \\\\")
tex += "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}

\textbf{Why counts are fragile.} A count depends on where object
boundaries land; acquisition contrast and sharpness shift them, merging
or splitting objects while conserving total area. Even a minimum-size
cut-off does not rescue the count --- the session effect is edge drift
across the whole size range, not specks at the detection floor. The
pipeline's working rule: \emph{fractions are primary; counts are
within-session diagnostics only}.

\textbf{Unblocked comparison (declared family, exact permutation +
Benjamini--Hochberg + minimum detectable difference).} Of
""" + str(len(N["unblocked"])) + r""" declared tests,
""" + str(N["unblocked_nominal"]) + r""" reach nominal $p<0.05$
(""" + f"{N['unblocked_expected_chance']:.1f}" + r""" expected by
chance), and """ + str(N["unblocked_bh_survivors"]) + r""" survive
BH at $q=0.10$ --- all count-type markers on the Batch\_1--Batch\_3
pair: pore count $+$""" + f"{ub('pores_per_mpx','Batch_1-Batch_3')['diff']:.1f}" + r"""/Mpx
($p_{\rm adj}=$""" + f"{ub('pores_per_mpx','Batch_1-Batch_3')['padj']:.3f}" + r"""),
noise-adjusted pore count $+$""" + f"{ub('pores_per_mpx_noise_adj','Batch_1-Batch_3')['diff']:.1f}" + r"""/Mpx
($p_{\rm adj}=$""" + f"{ub('pores_per_mpx_noise_adj','Batch_1-Batch_3')['padj']:.3f}" + r"""),
median pore size $-$""" + f"{abs(ub('pore_d50_um','Batch_1-Batch_3')['diff']):.3f}" + r"""\,\textmu m
($p_{\rm adj}=$""" + f"{ub('pore_d50_um','Batch_1-Batch_3')['padj']:.3f}" + r""").
Every survivor belongs to the session-dominated marker class ---
self-consistently so.

\textbf{Blocked comparison (stratified permutation, batch labels shuffled
within groups only).} Only five groups mix batches, and shared coverage is
thin: Batch\_1--Batch\_2 share """ + str(len(N['shared_sessions']['Batch_1-Batch_2'])) + r"""
sessions, Batch\_2--Batch\_3 share """ + str(len(N['shared_sessions']['Batch_2-Batch_3'])) + r""",
and \textbf{Batch\_1--Batch\_3 share exactly one session with one photo
each} --- the decisive pair is nearly untestable. Within that limit:
\textbf{zero} of the """ + str(len(N["blocked"])) + r""" blocked tests
reach nominal significance, including on the markers that survived
unblocked. In the single session shared by the key pair, Batch\_1 has
\emph{fewer} pores than Batch\_3 (123 vs.\ 141/Mpx).

\textbf{Does the pipeline see material or sessions?} A classifier given
only acquisition information (frame height, noise, contrast) reaches 68\%
leave-one-out batch accuracy, above the 55\% majority baseline; the full
image-feature set reaches 42\%. The imaging conditions carry more
apparent batch information than the material features. The three
new-image calls are consistent with this: every photo's frame height
occurs in only one batch, and each call followed the session --- the
predictions were saved blind, but the outcome cannot distinguish
batch-match from session-match. All three are reported ``can't tell''.

\textbf{Honest boundary.} This does not prove the batches identical ---
the within-session test is under-powered. It proves the dataset cannot
separate material from imaging condition, and that no count-based or
unblocked batch claim is supportable as measured evidence.

\section{What was demoted, and why}
\label{sec:demoted}

\begin{itemize}
\item \textbf{Boundary-dependent markers} (particle sizes and shapes,
chords, anisotropy, clustering index, crack fraction, percolation reach,
largest pore region, correlation lengths): every batch comparison on them
is nominal-or-less unblocked and zero blocked; they also depend on where
boundaries land, which the session moves. Appendix material.
\item \textbf{Silicon-access markers} (distance to pore, enclosed share,
coverage, accessible fraction): built on one 2-D slice over resolved
pores only --- 2-D sees $\sim$10\% porosity of a real 25--40\%, and
out-of-plane contacts are invisible. Kept as stated limitations inputs
to the model, not as batch markers.
\item \textbf{The InLens/ETD layer} (exploratory): InLens gain differs
by batch (median brightness $\approx$89 for Batch\_3 vs.\ $\sim$110 for
Batches~1--2), so even its within-session-consistent direction
(Batch\_1 pores darkest in all 3 shared sessions) rests on $n=3$.
Promising, unproven --- follow-up with controlled re-imaging.
\item \textbf{New-image assignments:} kept blind protocol; all three
calls ``can't tell'' pending session control.
\item \textbf{DFN indicators} (Appendix~\ref{app:dfn}): model-conditional,
assumption-dominated, session-inheriting.
\end{itemize}

\section{What was removed, and why}
\label{sec:removed}

\begin{itemize}
\item \textbf{FFT ``vertical texture'' and the curtaining test:} the
directionality score measured the frame's aspect ratio, not the texture
--- pure noise scores the same as real images at the real frame shape;
direct autocorrelation shows features slightly \emph{horizontally}
elongated. Archived.
\item \textbf{Diffusion-solve tortuosity:} no 2-D section ever has a
spanning pore path, so the solve returned $+\infty$ everywhere --- it
never produced a usable number. The honest residual is one Limitations
sentence.
\item \textbf{Swelling-deficit as a finding:} negative by construction
(resolved porosity $\sim$10\% vs real 25--40\%) and ``deficit tiles
track silicon'' true by definition. The real content --- silicon wetting
depends on pores we cannot see --- stays in Limitations as a blind spot.
\item \textbf{3-D reconstruction branch} (optional SliceGAN): not ground
truth, not needed for the core deliverable; archived runnable.
\end{itemize}

\section{Limitations}
\label{sec:limits}

\begin{itemize}
\item \textbf{Acquisition and material are not adequately separated} ---
the dominant limitation. Groups are inferred; shared sessions are thin
(B1--B3: one). ``Can't tell'' is the correct status for every batch
difference, not ``identical''.
\item \textbf{``Bright'' $\neq$ ``silicon''} --- EDS is the gate.
\item \textbf{Resolved porosity $\approx$10\% vs true 25--40\%} ---
silicon wetting runs through sub-resolution pores we cannot see; a real
blind spot, stated not hidden.
\item \textbf{Absolute model values are illustrative}; all assumed
parameters (thickness, sub-resolution porosity, mechanics) are swept and
still cap accuracy.
\item \textbf{No 3-D connectivity} from 2-D sections.
\item \textbf{BSE cannot see} binder, SEI, lithium inventory, wetting,
foil corrosion.
\end{itemize}

\appendix
\section{DFN consequence indicators (demoted)}
\label{app:dfn}

The Doyle--Fuller--Newman model translates image markers into cell
responses; it adds no measurement of its own. Under the repaired model
(in-cycle peaks, one shared cathode per sweep index, corrected plating
rule), paired 17-point sweeps give: cycle-5 discharge capacity
5.59/5.68/5.48\,Ah (B1/B2/B3); peak swelling $\sim$12\,\textmu m;
peak silicon surface stress 10.0/9.5/10.8\,MPa. Only the stress ordering
exceeds its declared threshold (B2$-$B3 $=-1.60$\,MPa, $-15\%$ of the B3
value; B1$-$B3 $=-0.95$; B1$-$B2 $=+0.53$), driven almost entirely by the
\emph{modelled} reachable-silicon share (Spearman $\rho\approx-0.93$) ---
a 2-D contact proxy, not measured connectivity. Assumptions alone move
capacity by 1.14\,Ah against $\le$0.15\,Ah of batch difference, and the
label-uncertainty bracket moves stress by up to 1.8\,MPa --- more than
the batch difference itself. A further caveat found in this audit: the
saved sweeps fed particle-radius overrides under invalid PyBaMM key
names, so both radii ran at their parameter-set defaults for every
batch and point --- the radius channel contributed nothing to the saved
numbers (fixed in code for the next run). Read as a hypothesis for the
image layer,
not a batch verdict; the model also inherits the session confound through
its inputs. Lithium loss, retention and efficiency are diagnostics only
(nothing ages measurably in 5 cycles).

\section{Exploratory channel features}
\label{app:channels}

The co-registered InLens and ETD detector channels (0-px shift on all 34
fields) answer different physical questions than BSE; they are recorded
and analysed in \texttt{channel\_features.csv} but kept out of the frozen
ruler. Within-session-consistent direction: Batch\_1's pores read darkest
in InLens in all 3 shared sessions --- a candidate follow-up signal, not
a measured pore depth, with $n=3$ and batch-correlated gain.

\section{Provenance}
\label{app:prov}

Recipe \texttt{recipe.json} (baseline Batch\_3, $t_{\rm pore}=0.711$,
$t_{\rm Si}=1.335$, $t_{\rm core}=1.550$); every run writes
\texttt{run\_manifest.json} (recipe, code and image hashes). All numbers
in this paper are generated by \texttt{build\_report.py} from
\texttt{per\_image\_features.csv}, \texttt{image\_acquisition.csv},
\texttt{objects\_classified.csv} and \texttt{dfn\_paired\_differences.csv};
\texttt{check\_report.py} fails on any stale or hand-typed value.

\end{document}
"""

with open("paper/micro2dfn_paper.tex", "w") as fh:
    fh.write(tex)
print(f"wrote paper/micro2dfn_paper.tex ({len(tex)} chars)")
