#!/usr/bin/env python3
"""
build_deck.py — assemble micro2dfn_masterclass.pptx.

Rebuild:  cd teaching_micro2dfn && ../.venv/bin/python scripts/build_deck.py
Requires: teaching_micro2dfn/figures/*.png  (run scripts/build_figures.py first)
All slide numbers are traceable to claim_evidence.csv (claim ids in notes).
"""
import os

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

HERE = os.path.dirname(os.path.abspath(__file__))
TEACH = os.path.dirname(HERE)
FIG = os.path.join(TEACH, "figures")
OUT = os.path.join(TEACH, "micro2dfn_masterclass.pptx")

# ---------------------------------------------------------------- style
INK = RGBColor(0x22, 0x28, 0x2E)
GREY = RGBColor(0x6B, 0x74, 0x80)
LIGHT = RGBColor(0xF4, 0xF6, 0xF7)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
TEAL = RGBColor(0x0F, 0x7B, 0x7B)
WARN = RGBColor(0xB3, 0x44, 0x1E)
B1 = RGBColor(0xD5, 0x62, 0x3A)
B2 = RGBColor(0x2E, 0x7D, 0xC2)
B3 = RGBColor(0x43, 0x9A, 0x68)
FONT = "Arial"

PAGE_W, PAGE_H = Inches(13.333), Inches(7.5)
TITLE_Y = Inches(0.42)
BODY_TOP = Inches(1.30)
FOOT_Y = Inches(7.02)

prs = Presentation()
prs.slide_width = PAGE_W
prs.slide_height = PAGE_H
BLANK = prs.slide_layouts[6]


def _box(slide, x, y, w, h):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    return tf


def _run(p, text, size, color=INK, bold=False, italic=False):
    r = p.add_run()
    r.text = text
    f = r.font
    f.name = FONT
    f.size = Pt(size)
    f.bold = bold
    f.italic = italic
    f.color.rgb = color
    return r


def add_title(slide, kicker, title, route=False):
    """kicker = small coloured section label; title = one-line idea."""
    tf = _box(slide, Inches(0.55), Inches(0.28), Inches(11.6), Inches(0.4))
    p = tf.paragraphs[0]
    _run(p, kicker.upper(), 12.5, TEAL, bold=True)
    tf = _box(slide, Inches(0.55), Inches(0.62), Inches(12.0), Inches(0.75))
    p = tf.paragraphs[0]
    _run(p, title, 25, INK, bold=True)
    if route:
        chip = slide.shapes.add_shape(
            1, Inches(12.15), Inches(0.34), Inches(0.85), Inches(0.34))
        chip.fill.solid()
        chip.fill.fore_color.rgb = TEAL
        chip.line.fill.background()
        ctf = chip.text_frame
        ctf.paragraphs[0].alignment = PP_ALIGN.CENTER
        r = ctf.paragraphs[0].add_run()
        r.text = "ROUTE"
        r.font.name = FONT
        r.font.size = Pt(9)
        r.font.bold = True
        r.font.color.rgb = WHITE


def add_bullets(slide, x, y, w, h, bullets, size=15):
    tf = _box(slide, x, y, w, h)
    tf.word_wrap = True
    first = True
    for item in bullets:
        lvl, txt = (item if isinstance(item, tuple) else (0, item))
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.level = lvl
        p.space_after = Pt(7)
        bold = txt.startswith("**") and "**" not in txt[2:]
        if lvl == 0:
            _run(p, "—  ", size, TEAL, bold=True)
        else:
            _run(p, "    ·  ", size - 1, GREY, bold=True)
        if txt.startswith("**"):
            if "**" in txt[2:]:
                head, txt = txt[2:].split("**", 1)
                _run(p, head, size if lvl == 0 else size - 1,
                     INK if lvl == 0 else GREY, bold=True)
            else:
                txt = txt[2:]
        if txt:
            _run(p, txt, size if lvl == 0 else size - 1,
                 INK if lvl == 0 else GREY, bold=bold)
    return tf


def add_pic(slide, fname, x, y, w=None, h=None, border=True):
    path = os.path.join(FIG, fname)
    kw = {}
    if w is not None:
        kw["width"] = w
    if h is not None:
        kw["height"] = h
    pic = slide.shapes.add_picture(path, x, y, **kw)
    if border:
        pic.line.color.rgb = RGBColor(0xD8, 0xDE, 0xE3)
        pic.line.width = Pt(0.75)
    return pic


def pic_fit(slide, fname, x, y, max_w, max_h):
    """Fit image inside box, centred, preserving aspect."""
    from PIL import Image
    path = os.path.join(FIG, fname)
    im = Image.open(path)
    ar = im.width / im.height
    w = max_w
    h = int(w / ar)
    if h > max_h:
        h = max_h
        w = int(h * ar)
    px = x + int((max_w - w) / 2)
    py = y + int((max_h - h) / 2)
    return add_pic(slide, fname, px, py, w=w)


def add_foot(slide, text, warn=False):
    tf = _box(slide, Inches(0.55), FOOT_Y, Inches(12.3), Inches(0.42))
    p = tf.paragraphs[0]
    _run(p, text, 10.5, WARN if warn else GREY, italic=True)


def add_notes(slide, notes):
    slide.notes_slide.notes_text_frame.text = notes


def new_slide(kicker, title, route=False):
    s = prs.slides.add_slide(BLANK)
    bg = s.shapes.add_shape(1, 0, 0, PAGE_W, PAGE_H)
    bg.fill.solid()
    bg.fill.fore_color.rgb = WHITE
    bg.line.fill.background()
    bg.shadow.inherit = False
    add_title(s, kicker, title, route)
    return s


# ================================================================ SLIDES
N = []  # slide-number → route marker, filled as we go

# ---- 1 title
s = prs.slides.add_slide(BLANK)
bg = s.shapes.add_shape(1, 0, 0, PAGE_W, PAGE_H)
bg.fill.solid(); bg.fill.fore_color.rgb = INK; bg.line.fill.background()
tf = _box(s, Inches(0.9), Inches(1.7), Inches(11.5), Inches(0.5))
_run(tf.paragraphs[0], "TEACHING MASTERCLASS — micro2dfn", 15, TEAL, bold=True)
tf = _box(s, Inches(0.9), Inches(2.25), Inches(11.8), Inches(1.9))
p = tf.paragraphs[0]
_run(p, "From pixels to predictions:\n", 40, WHITE, bold=True)
p = tf.add_paragraph()
_run(p, "reading an anode cross-section honestly", 40, WHITE, bold=True)
tf = _box(s, Inches(0.9), Inches(4.35), Inches(11.4), Inches(1.3))
for i, t in enumerate([
        "The object: a thin graphite+silicon coating — the negative "
        "electrode of a lithium-ion battery — sliced open and photographed.",
        "How that photo becomes material labels, how labels become "
        "quality-control comparisons and model inputs, and exactly "
        "where the evidence stops — worked end-to-end on real data."]):
    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
    _run(p, t, 16, RGBColor(0xC9, 0xD2, 0xDA))
tf = _box(s, Inches(0.9), Inches(6.5), Inches(11.4), Inches(0.5))
_run(tf.paragraphs[0],
     "Data: 34 electron-microscope fields of view · image-level "
     "comparisons only · all "
     "numbers traceable via claim_evidence.csv", 11.5, GREY)
add_notes(s, """Welcome and framing (90 s).

This deck teaches ONE thing: the full evidential chain from a raw
back-scattered-electron image to (a) a QC comparison against a baseline
batch and (b) physics-model consequence indicators — with every step
executed on the real images in this repository, and every number
traceable to a file.

Two systems exist here and they do DIFFERENT jobs:
- polaron_qc  — anomaly detection against a frozen baseline (Batch_3).
  Its verdicts (ACCEPT / INVESTIGATE / REJECT) are statements about
  similarity to a reference, not about chemistry or cell life.
- micro2dfn   — structural measurement (pores, silicon objects,
  proximity) plus exploratory PyBaMM-DFN consequence indicators.

House rules we will keep honouring all the way through:
- Batch_3 is a shipment-approved SELECTED reference, not proven
  defect-free ground truth.
- Every quantitative claim carries a class: measured / inferred /
  assumed / simulated / illustrative (see claim_evidence.csv).
- The image, not the pixel or the particle, is the independent unit —
  34 images is a small dataset, so several honest answers are "we
  cannot tell with this data".

A 10-slide fast route is marked ROUTE in the corner; everything else is
the teaching path (technical depth lives in the notes + appendix).""")

# ---- 2 route map
s = new_slide("how to use this deck", "Two routes through the material")
add_bullets(s, Inches(0.55), Inches(1.5), Inches(7.2), Inches(4.6), [
    "**Fast route — 10 slides, ~15 min (marked ROUTE)",
    (1, "title → the question → one image → the two systems → the "
        "frozen ruler → measurements → the one supported finding → "
        "the model's answer → where evidence stops → take-home"),
    "**Teaching path — full deck + appendix",
    (1, "worked image transformations, statistics, reconciliation, "
        "model mechanics, failure modes, roadmap"),
    "**Companion files",
    (1, "presenter_script.md — verbatim narration"),
    (1, "metric_dictionary.csv — every metric: units, meaning, limits"),
    (1, "claim_evidence.csv — every number → its source file"),
    (1, "roadmap.md — what a prospective validation would require"),
], size=14.5)
add_foot(s, "Speaker notes carry the technical depth; appendix slides "
            "carry the reference tables.")
add_notes(s, """Orientation slide.

The fast route exists so a time-poor reviewer gets the honest story in
15 minutes: what the data are, what the pipelines do, the ONE supported
batch difference (noise-adjusted resolved-pore count, Batch_1 vs
Batch_3), the unresolved Batch_1 bright-material question, what the DFN
adds (conditional indicators only), and the evidence wall.

Everything outside the route is where a student learns HOW: the
histogram where thresholds come from, why thresholds are frozen, the
classifier that caused the two pipelines to disagree, the permutation
test mechanics, the 17-point assumption sweep, and the roadmap for
doing this prospectively.""")

# ---- 3 the question (ROUTE)
s = new_slide("the question", "Three batches, three new photos, one question",
              route=True)
add_bullets(s, Inches(0.55), Inches(1.55), Inches(6.6), Inches(4.9), [
    "**The setup",
    (1, "Thin graphite+silicon electrode coatings, sliced open "
        "(ion-milled — polished flat with an ion beam) and imaged by "
        "SEM back-scattered electrons (BSE)."),
    (1, "Batch_3 = the shipment-approved reference. Batch_1 and "
        "Batch_2 = candidates. Plus 3 unlabelled new photos."),
    "**The practical questions",
    (1, "Do the candidate batches look like the reference? "
        "(the quality-control, QC, job)"),
    (1, "Where they differ, what could it mean downstream? "
        "(the consequence job)"),
    "**What this deck will NOT claim",
    (1, "no chemistry, no measured cell performance, no lifetime, "
        "no safety verdicts — images cannot supply those"),
], size=14)
pic_fit(s, "fig_dataset_tree.png", Inches(7.4), Inches(1.5),
        Inches(5.4), Inches(4.9))
add_foot(s, "BSE = back-scattered electrons; SEM = scanning electron "
            "microscope — the workhorse instrument. Brightness rises "
            "with average atomic number: heavier atoms glow brighter, "
            "so carbon shows dark and silicon bright.")
add_notes(s, """Claims used here: C01 (34 FOVs), dataset counts.

Set the stakes plainly. These are graphite+silicon composite anodes.
A QC engineer receives new batches and photos; the questions are
similarity and consequence. What we CAN answer: whether the resolved
structure differs from the reference under a frozen measurement
recipe, and — under declared assumptions — how those structural
differences could move model indicators. What we CANNOT: chemistry
(the disputed bright material needs EDS), cell performance (no cells
were tested), lifetime, safety.

Notice the asymmetry already: the reference was chosen because it
passed shipment review — it is a ruler, not a promise of perfection.
Everything downstream is 'different from Batch_3', never 'good/bad'
in absolute terms.""")

# ---- 4 what one image is (ROUTE)
s = new_slide("the data", "One field of view: what BSE brightness means",
              route=True)
pic_fit(s, "fig_ref_raw.png", Inches(0.5), Inches(1.35),
        Inches(7.1), Inches(5.4))
add_bullets(s, Inches(7.85), Inches(1.6), Inches(5.0), Inches(4.9), [
    "**Reading the greys",
    (1, "near-black = open pore / resin"),
    (1, "mid-grey = graphite + binder bulk"),
    (1, "bright = silicon-family (higher mean Z)"),
    "**The worked example",
    (1, "img_hawkfj64 — the Batch_3 image nearest the batch median on "
        "all 9 declared features (a documented selection rule, not "
        "cherry-picked)"),
    "**25 nm per pixel",
    (1, "a sampling pitch, not a resolution — the effective object "
        "floor is ~0.13 µm pores / ~0.35 µm Si"),
    (1, "scale check: a human hair ≈ 70 µm — the smallest pores we "
        "can see are ~1/500th of that"),
], size=13.5)
add_foot(s, "C02–C04 in claim_evidence.csv. Crop reused for every "
            "transformation that follows — same region throughout. "
            "µm = micrometre (1/1000 mm).")
add_notes(s, """Claims: C03 (25 nm pitch), C04 (selection rule for
img_hawkfj64), C01.

BSE contrast is compositional: heavier average atomic number =
brighter. Carbon (graphite, binder) is dark-to-mid; silicon is
brighter; vacuum-filled pores are darkest. This is why BSE — and not
the secondary-electron channels — is the workhorse for phase
segmentation, and also its weakness: contrast says 'mean atomic
number', not 'chemical species'. Silicon vs silicon-oxide vs a
bright contaminant can overlap; that ambiguity becomes the central
plot twist for Batch_1 later.

The 25 nm/px figure comes from the TIFF resolution tags (manifest).
Stress: pitch is not resolution — beam broadening, polishing smear,
and the pixel-count floor set a practical limit, which is why the
recipe uses min-object sizes (20 px pores ~0.13 µm, 150 px bright
~0.35 µm).""")

# ---- 5 dataset inventory
s = new_slide("the data", "The dataset: 34 fields of view, three channels")
add_bullets(s, Inches(0.55), Inches(1.55), Inches(5.7), Inches(4.9), [
    "**Batch_1 — 7 fields of view (FOVs)",
    "**Batch_2 — 7 FOVs",
    "**Batch_3 — 17 FOVs (the reference)",
    "**New_Images_Batch — 3 unlabelled FOVs",
    "**Channels",
    (1, "BSE: all 34 — the segmentation channel"),
    (1, "ETD (Everhart–Thornley, surface-sensitive): 30"),
    (1, "InLens (topography): all 34"),
    "**Images are the unit",
    (1, "34 is small; image-to-image variance sets every bound below"),
], size=13.5)
pic_fit(s, "fig_inventory.png", Inches(6.5), Inches(1.45),
        Inches(6.3), Inches(5.2))
add_foot(s, "dataset_manifest.csv lists every file + md5. C01–C02.")
add_notes(s, """Claims: C01, C02.

Point out the design constraint that drives ALL statistics later: the
image is the independent unit, and we have 7/7/17. A single image is
~1500x2500 px and may contain ~10^3 objects — but those pixels and
objects are correlated inside one specimen, one polish, one imaging
session. Treating them as independent would overstate the evidence by
orders of magnitude. That is why the comparison uses batch-level
permutation on image-level summaries.

Channels matter for adjudication: ETD/InLens give texture/topography
that help decide whether a bright object is a real particle or an
edge artefact (used in the six-test adjudication later).""")

# ---- 6 what we cannot see
s = new_slide("the data", "What these images cannot tell us")
add_bullets(s, Inches(0.55), Inches(1.55), Inches(6.4), Inches(5.0), [
    "**Below the object floor",
    (1, "sub-resolution porosity (~0.15–0.30 fraction exists but is "
        "invisible — enters the model only as an ASSUMED add-on)"),
    "**Not in a 2-D section",
    (1, "3-D pore connectivity, true through-plane tortuosity"),
    (1, "electrode thickness (foil never in frame → assumed 75 µm)"),
    "**Not in BSE at all",
    (1, "binder distribution, the SEI film (the thin protective "
        "layer that forms on electrodes), lithium inventory"),
    (1, "chemical identity of bright material → needs EDS "
        "(elemental spectroscopy)"),
    "**Not in a photo",
    (1, "measured capacity, cycle life, safety"),
], size=14)
add_bullets(s, Inches(7.3), Inches(1.7), Inches(5.5), Inches(4.6), [
    "**Consequence for honesty",
    (1, "every model input that is not measured is declared and "
        "swept over a band (assumption_table.csv)"),
    (1, "the wall between measured and assumed is drawn on every "
        "slide from here on"),
], size=14)
add_foot(s, "This list IS part of the result set: it defines the "
            "boundary of every claim that follows.", warn=True)
add_notes(s, """Claims: C37 (assumed inputs), C39 (tortuosity limit).

This is the most important honesty slide in the deck. Students must
internalise: a cross-section image is a thin, 2-D, sub-micron-resolved
snapshot. Everything about the third dimension, the sub-resolution
phases, and the chemistry is INFERRED or ASSUMED.

The DFN model later needs ~10 inputs the images cannot provide; we
handle that by declaring them, sweeping them over published/physical
bands, and comparing batches at matched assumption points — but the
batch difference can never exceed the assumption spread (we will show
that it does not).""")

# ---- 7 two pipelines (ROUTE)
s = new_slide("the two systems", "Same pixels, two different jobs", route=True)
add_bullets(s, Inches(0.55), Inches(1.5), Inches(6.1), Inches(5.2), [
    "**polaron_qc — anomaly detection",
    (1, "question: does this batch break the reference's own "
        "envelope?"),
    (1, "frozen baseline thresholds → robust z (a 'how unusual' "
        "score) → fingerprints → verdict (ACCEPT / INVESTIGATE / "
        "REJECT)"),
    (1, "verdict = similarity to baseline, not material truth"),
    "**micro2dfn — measurement → consequence",
    (1, "question: what structure is resolved, and what could it do "
        "in a physics model?"),
    (1, "shared segmentation → OBJECT classifier → ~30 markers → "
        "a battery-physics simulation (PyBaMM DFN)"),
], size=13.5)
add_bullets(s, Inches(7.0), Inches(1.7), Inches(5.8), Inches(4.6), [
    "**They share the ruler",
    (1, "same flat-field, same multi-Otsu idea, same pixel scale"),
    "**They differ at ONE decision",
    (1, "what counts as a silicon PARTICLE vs ambiguous bright "
        "material — and that single difference explains every "
        "downstream disagreement (r = 0.27 on classified Si, "
        "0.8–1.0 on everything else)"),
    "**No claim of which is 'better'",
    (1, "accuracy needs common ground truth — which does not exist "
        "yet; see roadmap.md"),
], size=13.5)
add_foot(s, "Reconciliation detail: RECONCILIATION.md, "
            "reconcile_output/metric_crosswalk.csv. C21–C22.")
add_notes(s, """Claims: C21 (crosswalk correlations), C22 (verdict
agreement 13/14).

Key teaching point: two systems can share a segmentation front-end
yet answer different questions, so comparing their 'verdicts'
directly is a category error. polaron_qc flags departure from a
baseline (all bright material counted generously); micro2dfn must
assign a mechanism-relevant class to every bright object (particle
vs fine vs rim), because only particles belong in the DFN particle
model.

The empirical fact that makes this vivid: across 31 images the two
pipelines correlate at r~0.8-1.0 on every shared metric EXCEPT
classified silicon (r=0.27). They see the same pixels; they disagree
only about what to call the borderline objects. This is the single
most instructive disagreement in the whole project.""")

# ---- 8 flat field
s = new_slide("image → labels", "Step 1: remove the illumination gradient")
pic_fit(s, "fig_flatfield.png", Inches(0.5), Inches(1.4),
        Inches(8.2), Inches(5.3))
add_bullets(s, Inches(8.95), Inches(1.7), Inches(4.0), Inches(4.9), [
    "**The problem",
    (1, "large-area intensity drift (beam, detector, charging) makes "
        "one global threshold unfair"),
    "**The fix",
    (1, "divide by a heavily-smoothed copy of itself (Gaussian "
        "sigma-200 on a 16x downsample)"),
    "**What stays honest",
    (1, "thresholds are still fit on the ORIGINAL intensities of the "
        "reference — the flat field only de-trends"),
], size=13)
add_foot(s, "polaron_qc/io.py flat_field; same idea in both pipelines.")
add_notes(s, """Flat-field correction divides the image by its own
heavily low-passed version, removing gradients larger than the
features we care about while leaving particle-scale contrast intact.

Why it matters for trust: a vignette can shift a region across a
global threshold and masquerade as composition change. After
flat-fielding, residual drift still exists (the si_bulk_contrast
guard metric watches for it — two Batch_1 photos later fail this
guard).""")

# ---- 9 histogram
s = new_slide("image → labels", "Step 2: the histogram is where labels come from")
pic_fit(s, "fig_histogram.png", Inches(0.55), Inches(1.45),
        Inches(12.2), Inches(5.0))
add_foot(s, "Multi-Otsu finds the two valleys that split pore / bulk / "
            "bright. Thresholds are RECIPE CHOICES — different choices, "
            "different labels, different findings.")
add_notes(s, """Claims: C05–C06 (threshold values).

Walk the histogram left to right: the pore mode (near black), the big
bulk mode (graphite+binder), and the long bright tail (silicon
family). Otsu's method chooses cut points minimising within-class
variance — it is a criterion, not a truth. On this reference image
the frozen thresholds (brightness cut-points on the corrected scale) are t_pore=0.7109, t_si=1.3354, t_core=1.5495
(intensity units after flat-field normalisation).

Teaching point: every 'material fraction' downstream is a function
of where these vertical lines sit. When we later say '6% silicon',
what we mean is '6% of pixels/objects fall right of t_si under the
frozen recipe'. Different threshold = different measured world. That
is why the recipe is frozen and versioned (recipe.json).""")

# ---- 10 frozen vs floating (ROUTE)
s = new_slide("image → labels", "Step 3: freeze the ruler, or it absorbs the anomaly",
              route=True)
pic_fit(s, "fig_frozen_floating.png", Inches(0.55), Inches(1.5),
        Inches(12.2), Inches(4.4))
add_foot(s, "Frozen thresholds are fit ONCE on Batch_3 pixels and reused "
            "on every image. A per-image (floating) threshold re-centres "
            "on each image and can hide exactly the shift we are looking for.",
         warn=True)
add_notes(s, """Claims: C05, C18 (floating Otsu counts MORE bright on
flagged images: 20.7% vs 11.9%).

This is the project's core metrology lesson: 'new data must not
silently recalibrate their own ruler'. If the threshold is refit per
image, then a batch with genuinely more bright material just moves
the cut higher and reports a similar fraction — the ruler bends to
fit the anomaly. The vcompare recipe therefore fits thresholds on the
baseline (Batch_3) pixels only, freezes them in recipe.json, and
applies them unchanged to every image.

Caveat taught here: freezing can OVER-detect too (on the dim
Batch_1 photos, floating Otsu actually finds MORE bright material
than frozen — direction of bias is not guaranteed). The defence is
not that frozen is unbiased, it is that frozen is COMPARABLE —
every image is judged by the same ruler, and the ruler's own
uncertainty (threshold sweep) is reported.""")

# ---- 11 segmentation + watershed
s = new_slide("image → labels", "Steps 4–5: threshold, then split touching objects")
pic_fit(s, "fig_seg_steps.png", Inches(0.5), Inches(1.4),
        Inches(7.0), Inches(5.3))
add_bullets(s, Inches(7.75), Inches(1.6), Inches(5.1), Inches(5.0), [
    "**Thresholding gives blobs",
    (1, "but adjacent particles fuse into one object"),
    "**Watershed splits on the distance transform",
    (1, "peaks inside the blob become seeds; boundaries flood "
        "between them"),
    "**Why we must split",
    (1, "particle size and clustering metrics are meaningless "
        "while objects are welded"),
    "**The cost",
    (1, "over-splitting is possible → size metrics carry a "
        "splitting bias (a stated limitation)"),
], size=13.5)
add_foot(s, "Objects below the size floor merge into bulk before "
            "measurement: Si < 150 px (~0.35 µm), pore < 20 px (~0.13 µm).")
add_notes(s, """Claims: C07 (size floors).

The watershed step converts a binary mask into countable objects.
Two teaching cautions: (a) splitting quality sets the floor of every
geometry metric — if two particles stay fused, D50 is inflated; if a
particle is over-split, D50 is deflated; (b) the minimum-size floor
is a deliberate recipe choice that removes sub-resolution dust, and the
same floor must apply to every image or counts are incomparable.

The distance-transform visual (shown in fig) makes the mechanism
intuitive: each bright blob is a landscape, watershed floods from the
peaks, and ridges become object boundaries.""")

# ---- 12 classifier
s = new_slide("image → labels", "Step 6: the object classifier — where the two pipelines part")
pic_fit(s, "fig_classifier.png", Inches(0.5), Inches(1.4),
        Inches(8.0), Inches(5.3))
add_bullets(s, Inches(8.7), Inches(1.55), Inches(4.2), Inches(5.1), [
    "**Three labels for bright objects",
    (1, "si_particle — interior >= t_core AND solidity >= 0.75 "
        "(solidity = how filled-in the shape is: area ÷ tight-fit "
        "outline area)"),
    (1, "bright_fine — fails the cut: small, low-solidity, or "
        "low-contrast ambiguous material"),
    (1, "rim/edge — bright halo at a pore edge (touched-border "
        "crescents)"),
    "**Why it exists",
    (1, "the DFN needs PARTICLES; rims and dust are not particles"),
    "**Why it matters later",
    (1, "this is the ONLY metric where the two pipelines disagree "
        "(r = 0.27)"),
], size=12.5)
add_foot(s, "t_core = 5th percentile of clear-particle interior "
            "intensity, calibrated on the reference. Classifier choice, "
            "not ground truth. C08.")
add_notes(s, """Claims: C08 (classifier rule), C21 (the r=0.27
disagreement), C44 (up to ~11 pt disagreement on disputed photos).

The classifier is a two-gate rule: an object counts as a silicon
particle only if (a) its interior (eroded core) is bright enough to
clear t_core — removing rim/edge highlights whose cores are bulk —
and (b) it is solid enough — removing ragged dust and crescents.
t_core is calibrated as the 5th-percentile interior intensity of
unambiguous reference particles: a recipe assumption, versioned, not
measured truth.

On most images this changes little (~6% classified Si everywhere).
On the three anomalous Batch_1 regions it decides the entire
interpretation: count the ambiguous material as silicon (QC's
generous convention) and Si jumps to ~11-12%; apply the particle
gates (micro2dfn) and most of it drops to 'bright_fine'. SAME pixels,
different label, different story — the perfect illustration that
classification is a model, not a measurement.""")

# ---- 13 measurements (ROUTE)
s = new_slide("labels → numbers", "Step 7: objects become measurements", route=True)
pic_fit(s, "fig_measure.png", Inches(0.5), Inches(1.4),
        Inches(7.3), Inches(5.3))
add_bullets(s, Inches(8.0), Inches(1.6), Inches(4.85), Inches(5.1), [
    "**Per object",
    (1, "eq. diameter = sqrt(4A/pi); solidity; orientation"),
    "**Distances",
    (1, "distance transform from pore mask → Si-to-pore proximity "
        "(a 2-D wetting proxy)"),
    "**Topology",
    (1, "largest connected pore region share — network "
        "fragmentation"),
    "**Then aggregate",
    (1, "per-image summaries (medians, area-weighted D50) — the "
        "image stays the unit"),
], size=13)
add_foot(s, "Cyan = largest connected pore region. Agent annotations "
            "over raw pixels — not expert labels. metric_dictionary.csv "
            "defines every term.")
add_notes(s, """Worked example: the circled particle is ~2.97 µm
equivalent diameter; its boundary sits ~0.07 µm from the nearest pore
(essentially pore-adjacent — good for lithium access in the model's
language).

Three measurement ideas to teach:
1. Equivalent diameter converts an arbitrary blob to the diameter of
   the equal-area circle — area-weighted D50 then summarises a batch
   without letting dust dominate.
2. The distance transform answers 'how far is each Si boundary from
   electrolyte space' — a 2-D stand-in for wetting (real wetting is
   3-D; flagged as a limitation).
3. Largest-connected-region share measures whether pores form one
   network or many fragments — 2-D connectivity only; a section can
   slice through a connected 3-D network and look fragmented.

Every one of ~30 markers in metric_dictionary.csv is built from these
three primitives: sizes, distances, topology.""")

# ---- 14 metric dictionary concept
s = new_slide("labels → numbers", "~30 markers, each with units, limits, and a job")
add_bullets(s, Inches(0.55), Inches(1.5), Inches(6.15), Inches(5.2), [
    "**Every marker declares",
    (1, "units · equation · worked number · what an increase means"),
    (1, "limitations · recipe-sensitivity · where it feeds the model"),
    "**Classes (the honesty axis)",
    (1, "image-derived — measurable from pixels (pore fraction)"),
    (1, "geometry-derived — computed on objects (D50, Clark-Evans)"),
    (1, "physical proxy — heuristics (Bruggeman exponent)"),
    (1, "assumed input — declared, swept (thickness, sub-res "
        "porosity)"),
    (1, "simulation output — model results (stress, capacity)"),
    "**The watchlist",
    (1, "8 metrics that carry the whole interpretation — one-page "
        "watchlist in the repo"),
], size=13.5)
add_bullets(s, Inches(7.0), Inches(1.55), Inches(5.8), Inches(5.0), [
    "**Watchlist (grouped)",
    (1, "acquisition guards: Si:bulk contrast, noise_MAD "
        "(image-noise meter), sharpness"),
    (1, "resolved pore fraction (reliable, DFN-adjacent)"),
    (1, "pore count + noise-adjusted count (the supported finding)"),
    (1, "Si candidate + uncertain-bright fractions (report TOGETHER)"),
    (1, "Si size: area-weighted D50/D90 → DFN radius"),
    (1, "Si→pore proximity + low-coverage share"),
    (1, "model stress + swelling budget (assumption-dependent)"),
], size=13.5)
add_foot(s, "The dictionary is the contract between the picture and "
            "the claim — read the limitations column before quoting a "
            "number.")
add_notes(s, """The metric dictionary turns 'the pipeline outputs
numbers' into 'each number has a declared meaning and a declared
failure mode'.

The five evidence classes are the deck's honesty grammar. If a
student quotes 'silicon fraction 6%', the dictionary asks: measured
under which recipe? Answer: image-derived under frozen thresholds,
classifier-dependent — the same pixels give ~11% under the QC
convention.

The watchlist exists because 30 markers is too many to hold in your
head — the 8 grouped metrics carry essentially all the
interpretation in this dataset. Everything else is supporting cast.""")

# ---- 15 comparison design (ROUTE via next slide; this is design)
s = new_slide("comparison", "Comparing batches: image is the unit, recipe is frozen")
add_bullets(s, Inches(0.55), Inches(1.55), Inches(6.3), Inches(5.1), [
    "**The design (validated_comparison/recipe.json)",
    (1, "fit thresholds + noise model on Batch_3 ONLY, freeze, apply "
        "to all 31 images"),
    (1, "summarise each image to 9 declared features"),
    (1, "compare batches on difference of medians, image unit"),
    "**The tests",
    (1, "exact permutation — every label assignment enumerated: "
        "C(24,7)=346,104 (B1–B3), C(14,7)=3,432 (B2–B3)"),
    (1, "Benjamini–Hochberg FDR 0.10 — a correction that caps the "
        "expected share of false 'significant' hits at 10% across "
        "the 9 tests per pair"),
    (1, "bootstrap confidence intervals, seed 0; a leave-one-out "
        "(LOO) sweep for fragility"),
    "**And for every null result",
    (1, "a minimum detectable difference (MDD) is reported — "
        "'no clear difference' != equivalence"),
], size=13)
add_bullets(s, Inches(7.2), Inches(1.7), Inches(5.6), Inches(4.8), [
    "**Why so defensive?",
    (1, "the earlier QC run claimed 'zero baseline false alarms' — "
        "the corrected LOO calibration measures 1/17 ≈ 5.9% "
        "[CI 1.0–26.9%]"),
    (1, "every safeguard below exists because a check FAILED once"),
], size=14)
add_foot(s, "vcompare/stats.py; recipe vcompare-1.0 fitted on 17 "
            "Batch_3 images, seed 0. C09, C25.")
add_notes(s, """Claims: C09 (design), C25 (the corrected 1/17 false
alarm), C48 (MDD on all 24 non-sig rows).

Exact permutation is the right test here because n is small and the
data need no distribution assumption: if batch labels meant nothing,
any of the 346,104 ways to split 24 images into 7+17 is equally
likely — the observed median difference is ranked against ALL of
them. No normality, no asymptotics.

The honesty hook for students: the original analysis reported a
'flag >= 2.73' threshold with zero reference exceedances, implying a
perfectly clean baseline. Re-running with reference-only weights and
leave-one-out calibration found 1 of 17 reference images DOES exceed
its own flag threshold. With n=17 that is a measured exceedance rate
of 5.9% with a wide CI — and it downgraded the Batch_1 verdict from
REJECT to INVESTIGATE under the stricter calibration. Small-n
calibration is fragile; this deck treats the corrected version as
authoritative.""")

# ---- 16 dotgrid = the result (ROUTE)
s = new_slide("comparison", "The result: one supported difference out of nine",
              route=True)
pic_fit(s, "fig_dotgrid.png", Inches(0.55), Inches(1.45),
        Inches(8.1), Inches(5.2))
add_bullets(s, Inches(8.9), Inches(1.55), Inches(4.0), Inches(5.2), [
    "**Supported (BH-significant)",
    (1, "Batch_1 vs Batch_3: resolved pore count +34.5/Mpx, "
        "exact p = 0.0005"),
    (1, "… and after noise adjustment: +18.8/Mpx, p = 0.002"),
    "**Everything else",
    (1, "Si fractions, sizes, clustering, porosity, anisotropy — "
        "no clear difference at these sample sizes"),
    "**Read the MDD column",
    (1, "differences SMALLER than the MDD would be invisible — "
        "not absent"),
], size=12.5)
add_foot(s, "validated_comparison/pairwise_comparisons.csv — 9 "
            "features x 3 pairs. Mpx = million pixels. C10–C13.")
add_notes(s, """Claims: C10, C11 (the supported finding), C12 (B2's
raw-only shift), C13 (Si MDD ~0.024).

Read the dot grid row by row. Only two cells survive the FDR
correction — both are the pore-count family for Batch_1 vs Batch_3.
Everything else is 'not established at this n', with MDDs quoted so
the reader knows how big a difference could still be hiding.

The most instructive non-result: classified Si fraction is flat
across batches (~6%) — under THIS recipe. The Batch_1 anomaly that
dominates later slides is NOT a batch-level Si-fraction shift; it is
three-of-seven heterogeneous regions with unusual bright material —
a different statistical object entirely (consensus exceedance, not a
median shift).""")

# ---- 17 permutation test
s = new_slide("comparison", "How the p-value is computed — every label shuffle")
pic_fit(s, "fig_permtest.png", Inches(0.55), Inches(1.45),
        Inches(8.3), Inches(5.1))
add_bullets(s, Inches(9.05), Inches(1.6), Inches(3.85), Inches(5.0), [
    "**Enumerate all splits",
    (1, "346,104 ways to choose 7 of 24 images as 'Batch_1'"),
    "**Rank the observed gap",
    (1, "+34.5/Mpx sits above every shuffled split → p = 165/346104"),
    "**Why exact",
    (1, "no distribution assumed; small n handled honestly"),
], size=13)
add_foot(s, "Figure recomputes the null live and reproduces the saved "
            "p = 0.000477 — the plot itself is a traceability check.")
add_notes(s, """Claims: C10 — and the figure is self-verifying: it
recomputes the permutation distribution from the per-image table and
lands on the identical p-value stored in pairwise_comparisons.csv.

Explain the histogram: each bar is one way the world could look if
batch labels were meaningless. The observed +34.5 difference (orange
line) exceeds essentially all of them — 165 splits out of 346,104
reach or beat it, hence p = 0.000477.

Contrast with parametric tests: a t-test would assume approximate
normality of the median difference; with n=7 in one group that
assumption is unverifiable. Enumeration replaces assumption with
arithmetic.""")

# ---- 18 noise adjustment
s = new_slide("comparison", "Noise inflates pore counts — adjust, then re-test")
pic_fit(s, "fig_noise_adj.png", Inches(0.5), Inches(1.45),
        Inches(8.3), Inches(5.15))
add_bullets(s, Inches(9.0), Inches(1.55), Inches(3.9), Inches(5.1), [
    "**The confound",
    (1, "noisier photo → more tiny threshold-crossing specks → "
        "higher pore count (correlation ≈ 0.91 with the noise_MAD "
        "guard metric)"),
    "**The fix",
    (1, "regress pore count on image noise using REFERENCE images "
        "only (slope 6.18, fit explains ~51% of variation), subtract "
        "the noise-predicted part"),
    "**The outcome",
    (1, "Batch_1 survives (+18.8, p=0.002)"),
    (1, "Batch_2's raw shift evaporates (+4.2, p=0.35) — it was "
        "mostly photography"),
], size=12)
add_foot(s, "recipe.json noise model; fitted on Batch_3 only so the "
            "correction itself cannot leak batch information. C11–C12.")
add_notes(s, """Claims: C11, C12; noise model parameters from
recipe.json (slope 6.18 pores per MAD unit, R²~0.51).

This slide teaches confounding at its purest: the 'extra pores' in
noisier photos are partly counting noise specks, not real voids. The
adjustment regresses raw pore count on the noise guard metric —
fitted on the 17 reference images ONLY, so the correction model
cannot be tuned to make the batches look similar or different — then
re-runs the identical permutation test on adjusted counts.

Two different lessons land at once:
- Batch_1's excess is NOT explained by noise → the finding survives.
- Batch_2's raw-count 'difference' was largely a photography effect →
  an honest pipeline must report this as 'possible before, gone after
  adjustment', which is exactly what START_HERE.md does.

Also note R²~0.51: noise explains about half the count variance, so
the residual is still a noisy measure — another reason the result is
reported as 'resolved pore COUNT differs', not 'porosity differs'.""")

# ---- 19 LOO robustness
s = new_slide("comparison", "Could one image be carrying the result?")
pic_fit(s, "fig_loo.png", Inches(0.55), Inches(1.5),
        Inches(8.1), Inches(4.9))
add_bullets(s, Inches(8.9), Inches(1.65), Inches(4.0), Inches(4.8), [
    "**Leave-one-out sweep",
    (1, "drop each Batch_1 image in turn, re-run the exact test"),
    "**Result",
    (1, "p stays <= 0.0016 no matter which image is dropped"),
    (1, "all 7 Batch_1 images sit above the Batch_3 median — it is "
        "a batch-level shift, not one weird photo"),
    "**Separate question",
    (1, "the 3-of-7 bright-material anomaly is a DIFFERENT, "
        "heterogeneous signal — next section"),
], size=12.5)
add_foot(s, "loo_sweep.csv — the pore-count shift is robust; the "
            "bright-material anomaly is heterogeneous (3 of 7), tested "
            "by consensus exceedance instead.")
add_notes(s, """Claims: C26 (LOO p<=0.0016).

Two distinct signals live in Batch_1 and students must keep them
separate:
1. A UNIFORM shift — resolved pore counts elevated in ALL SEVEN
   Batch_1 images relative to Batch_3. The LOO sweep proves no single
   image drives it. This is the supported comparison finding.
2. A HETEROGENEOUS anomaly — three of the seven regions carry
   unusual bright material. Median-based tests are blind to
   3-of-7 heterogeneity (medians absorb minorities); that is why the
   QC layer uses per-region flags + consensus-exceedance counting
   instead.

The LOO figure's flat p-line near the floor is what 'robust' looks
like; if dropping one image swung the p-value, the finding would be
fragile.""")

# ---- 20 the disputed three
s = new_slide("the Batch_1 anomaly", "Three of seven Batch_1 regions carry extra bright material",
              route=False)
pic_fit(s, "fig_disputed3.png", Inches(0.5), Inches(1.4),
        Inches(8.2), Inches(5.3))
add_bullets(s, Inches(8.95), Inches(1.55), Inches(4.0), Inches(5.15), [
    "**What was flagged",
    (1, "img_4ih2ggld, img_5n1q8atc, img_f1vzngrs"),
    (1, "bright material 10.7–12.5% (all-bright convention) vs "
        "baseline median 7.7%"),
    "**The complication",
    (1, "2 of 3 photos are DIMMER than baseline — their silicon "
        "glows fainter vs bulk (contrast 1.8 vs ~2.3). Real "
        "material or a contrast artefact?"),
    (1, "img_f1vzngrs shows the excess at NORMAL contrast — the "
        "interesting one"),
], size=12)
add_foot(s, "Consensus-exceedance detection: same-signature regions "
            "above the baseline's p90 (its 90th percentile). C14, "
            "C41.")
add_notes(s, """Claims: C41 (10.7-12.5% vs 7.7%), C14 (contrast
values), C20 (f1vzngrs normal contrast, z=+2.4 bright).

Set up the mystery honestly. Under the QC convention (all bright
material counted), these three regions carry 10.7-12.5% bright
material against a baseline median of 7.7% (max 10.7%). Two of the
three photos are measurably dimmer in silicon contrast — which cuts
BOTH ways: dimmer Si could mean a real composition change (e.g. a
SiOx-richer grade) OR an acquisition/contrast drift that smears
borderline pixels into the bright bin. The third region shows the
excess at normal contrast, which is why the anomaly cannot be waved
away as 'just the photography'.

What we do NOT say: 'Batch_1 has more silicon'. Batch-level median
Si fraction is ~6% like the others; this is a 3-of-7 regional
anomaly, and its chemical identity is unresolved.""")

# ---- 21 adjudication
s = new_slide("the Batch_1 anomaly", "Six tests adjudicate what the bright material probably is")
pic_fit(s, "fig_adjudication.png", Inches(0.5), Inches(1.4),
        Inches(12.3), Inches(5.25))
add_foot(s, "reconcile_output/adjudication.csv — every test is an "
            "algorithm check on pixels, not an expert label. C16–C20.")
add_notes(s, """Claims: C16–C20 — walk the six tests.

T1 (contrast-corrected re-classification): after correcting each
flagged region's contrast to baseline, 94% of ambiguous bright
objects pass the particle cut — vs only 45% in matched baseline
controls. The objects WANT to be particles.

T2 (morphology): ambiguous objects are compact discs ~1.05 µm,
solidity 0.82 (vs 0.88 for classified Si), only 8% touch image
borders. Particle-like, slightly softer-edged — consistent with fine
or dimmer silicon.

T3 (rim check): adjacent objects are compact discs, not crescent
rims — rules out 'pore-edge halo artefact'.

T4 (floating threshold): per-image Otsu finds MORE bright material
(20.7% vs 11.9% frozen) — the frozen threshold is if anything
conservative on these photos.

T5 (texture): in ETD and InLens the ambiguous material textures like
classified silicon (ratios 1.20/1.23), not like polished-flat resin.

T6 (distribution): whole-image median shift ~z=0.0 but bright-tail
shift z=1.8 — the anomaly lives in the tail, like a sub-population
of extra-bright objects.

Verdict of the adjudication: LIKELY FINE SILICON (or silicon-family).
What remains open: chemical identity — EDS or equivalent is the only
way to close it. And the dimmer contrast on 2/3 photos is a real
secondary anomaly (acquisition drift or dimmer-Si grade) that no
amount of re-thresholding fully explains.""")

# ---- 22 crosswalk / verdict semantics
s = new_slide("the Batch_1 anomaly", "Same pixels, same flag — different word for it")
pic_fit(s, "fig_crosswalk.png", Inches(0.5), Inches(1.45),
        Inches(7.6), Inches(5.1))
add_bullets(s, Inches(8.35), Inches(1.55), Inches(4.55), Inches(5.2), [
    "**Where the pipelines agree",
    (1, "correlation 0.8–1.0 (1.0 = identical ranking) on raw bright "
        "fraction, porosity, anisotropy, flake alignment"),
    (1, "both flag the SAME three regions"),
    "**Where they disagree",
    (1, "classified Si only: correlation 0.27 — the classifier "
        "decision"),
    "**What the verdicts mean",
    (1, "REJECT = beyond the baseline's own envelope on a matched "
        "defect signature → quarantine + root-cause; NOT 'cells will "
        "fail'"),
    (1, "under the stricter recheck (held-out calibration) Batch_1 "
        "reads INVESTIGATE — same anomaly, stricter ruler"),
], size=12)
add_foot(s, "reconcile_output/metric_crosswalk.csv + verdict_matrix.csv "
            "(13/14 images agree). C21–C24.")
add_notes(s, """Claims: C21, C22, C23 (original REJECT), C24 (corrected
INVESTIGATE), C47 (verdict semantics).

The crosswalk figure is the empirical heart of the reconciliation:
the two pipelines are not 'two answers', they are one measurement
plus one naming disagreement. r=1.00 on porosity, 0.96-0.92 on
anisotropy/alignment/raw bright — and 0.27 on classified Si. The
divergence is exactly the si_particle-vs-bright_fine gate, nothing
else.

Verdict semantics need care. QC 'REJECT' is a similarity statement
under a specific calibration — 'beyond the baseline envelope on the
matched defect signature' — which triggered quarantine + root-cause
in the workflow, not a product-failure claim. The corrected
calibration (reference-only weights, held-out T², measured 1/17
baseline exceedance) demotes it to INVESTIGATE: the anomaly is real,
the confidence in the severity label was overstated. Both statements
are true; the second is the defensible one.""")

# ---- 23 honest verdict
s = new_slide("the Batch_1 anomaly", "Where the Batch_1 evidence actually lands")
add_bullets(s, Inches(0.55), Inches(1.5), Inches(11.9), Inches(4.6), [
    "**Established",
    (1, "3 of 7 Batch_1 regions carry extra bright material (both "
        "pipelines flag the same regions)"),
    (1, "the objects behave like fine/dimmer silicon on six pixel-"
        "level tests — LIKELY fine silicon is the best-supported "
        "reading"),
    (1, "batch-level resolved pore count is higher in all 7 Batch_1 "
        "images (supported comparison finding — survives noise "
        "adjustment and dropping any single image)"),
    "**Not established",
    (1, "chemical identity — needs EDS/equivalent; 'silicon-family' "
        "is morphology+texture, not spectroscopy"),
    (1, "whether 2/3 dim-contrast photos are acquisition drift or a "
        "real dimmer-Si grade — either way it is batch "
        "non-uniformity"),
    (1, "any consequence for cells — no electrochemical data exist"),
], size=14.5)
add_foot(s, "EDS = energy-dispersive X-ray spectroscopy — it reads "
            "chemical elements, which BSE brightness alone cannot. "
            "Recommended next step: targeted EDS on the three flagged "
            "regions + re-image under controlled conditions.")
add_notes(s, """Claims: synthesis of C16–C24, C41, C45.

This is the slide a student should be able to reproduce from memory
— the discipline of separating 'established', 'best-supported
interpretation', and 'open'.

Notice what is deliberately absent: 'Batch_1 is worse'. The evidence
says 'Batch_1 is heterogeneous and carries a likely-fine-silicon
excess in 3 of 7 regions plus a uniform pore-count shift'. Whether
that is worse for a cell is a different question the images cannot
answer, and the DFN section will show the model indicators mostly
sit inside the assumption band anyway.""")

# ---- 24 DFN schematic
s = new_slide("structure → model", "What a DFN model is (in one picture)")
pic_fit(s, "fig_dfn_schematic.png", Inches(0.5), Inches(1.45),
        Inches(8.3), Inches(5.15))
add_bullets(s, Inches(9.0), Inches(1.6), Inches(3.9), Inches(5.0), [
    "**Doyle–Fuller–Newman",
    (1, "two porous electrodes + separator; lithium tracked through "
        "electrolyte and inside solid particles"),
    "**Our two-material negative",
    (1, "primary phase = graphite; secondary phase = silicon with "
        "swelling + stress coupling"),
    "**Why bother",
    (1, "it turns 'more/finer/coarser structure' into signed "
        "changes in model indicators — under declared assumptions"),
], size=12.5)
add_foot(s, "PyBaMM 26.9.0.0 DFN, Chen2020_composite parameter set. "
            "Citations in notes. C49.")
add_notes(s, """Claims: C49 (model+parameter sets).

The DFN model (Doyle, Fuller & Newman 1993) treats each electrode as
a porous continuum: liquid-phase lithium transport across the
thickness, solid-state diffusion inside representative spherical
particles at every position, Butler-Volmer kinetics at the
interface. PyBaMM's Chen2020_composite set adds a SECOND active
material (silicon) alongside graphite in the negative electrode —
the right skeleton for a Si/Gr blend.

What the image contributes: porosity, Si fraction, Si particle
radius, and connectivity proxies — i.e., the GEOMETRY inputs. What
the image cannot contribute: kinetics, electrolyte properties,
electrode thickness, sub-resolution porosity — declared assumptions
swept over bands.

Primary citations for notes: Doyle/Fuller/Newman JECS 140(6) 1993;
Chen et al. PhysChemChemPhys 2020 (parameter set); OKane et al.
PhysChemChemPhys 2022 (degradation constants); Bonkile & Ramadesigan
J. Power Sources 2024 (silicon mechanics).""")

# ---- 25 silicon swelling mechanism
s = new_slide("structure → model", "Why silicon is special: it swells ~3x on lithiation")
pic_fit(s, "fig_si_swelling.png", Inches(0.5), Inches(1.45),
        Inches(7.6), Inches(5.1))
add_bullets(s, Inches(8.35), Inches(1.55), Inches(4.5), Inches(5.2), [
    "**The numbers",
    (1, "Si: ~+280% volume at full lithiation (Li15Si4)"),
    (1, "graphite: ~+10%"),
    "**The consequences in the model",
    (1, "particle stress → mechanical fade coupling "
        "(stress-LAM = stress-driven loss of active material)"),
    (1, "electrode-thickness growth"),
    (1, "pore space is the expansion buffer → the 'swelling "
        "budget' proxy"),
    "**The image link",
    (1, "Si fraction + resolved porosity set how much buffer the "
        "structure shows — within the resolution floor"),
], size=12.5)
add_foot(s, "C43 — literature values, not measured here. Swelling "
            "budget is a derived proxy, not a DFN input.")
add_notes(s, """Claims: C43.

The mechanism everyone must own: silicon's capacity advantage comes
with ~3x volume change on lithiation vs ~10% for graphite. In the
model this drives (a) surface stress on Si particles (a stress-LAM
term converts stress into active-material loss), (b) electrode
thickness growth, and (c) consumption of pore volume.

The 'swelling budget' proxy = resolved pore fraction minus the
expansion demand of the measured Si and graphite fractions. It is a
DIAGNOSTIC made from the same pixels as the inputs — useful for
interpretation, explicitly NOT independent evidence and NOT a DFN
input. On this dataset it is negative everywhere (the resolved pore
space alone never covers nominal Si expansion) — another reason
sub-resolution porosity has to be assumed.""")

# ---- 26 input map (ROUTE)
s = new_slide("structure → model", "Which measurements feed which model inputs")
pic_fit(s, "fig_inputmap.png", Inches(0.5), Inches(1.4),
        Inches(7.9), Inches(5.3))
add_bullets(s, Inches(8.65), Inches(1.5), Inches(4.25), Inches(5.2), [
    "**Measured → model",
    (1, "resolved porosity → porosity (plus assumed sub-res add-on)"),
    (1, "Si D50/2 → Si particle radius (swept over the batch band)"),
    (1, "classified Si x accessible share → Si active fraction, LOW "
        "end of bracket"),
    (1, "ALL uncertain bright counted → HIGH end of bracket"),
    "**Assumed → swept",
    (1, "thickness 65–85 µm; sub-res porosity +0.15–0.30; Si "
        "modulus 35–90 GPa; nu 0.22–0.28"),
    "**Not a model input",
    (1, "pore COUNT — the one supported finding is invisible to the "
        "DFN entirely"),
], size=12)
add_foot(s, "17-point sweep per batch over the measurement band. "
            "dfn_input_bands.csv + assumption_table.csv. C37–C38.")
add_notes(s, """Claims: C37, C38.

Read the map left to right: measured quantities (solid edges) become
model inputs only through declared transforms — each with its own
uncertainty. The Si-active-fraction BRACKET is the elegant part: the
low end counts only classifier-passed Si x pore-accessible share;
the high end counts ALL ambiguous bright material as active silicon.
The classifier disagreement is thus carried into the model as an
explicit uncertainty bracket rather than being silently resolved.

The uncomfortable fact to state plainly: the one supported batch
finding (pore count) does not enter the model at all — pore count is
a count, while porosity (the fraction) is the input, and porosity is
not significantly different between batches. So the DFN comparison
answers a DIFFERENT question than the statistical comparison — it
asks 'given the measured geometry bands, how much do consequence
indicators move?'""")

# ---- 27 trace = one real run
s = new_slide("structure → model", "One sweep run, fully real: the protocol and the indicators")
pic_fit(s, "fig_trace.png", Inches(0.55), Inches(1.35),
        Inches(12.2), Inches(5.4))
add_foot(s, "Actual PyBaMM solution at the Batch_3 band centre — "
            "5 x (1C discharge → rest → 1C charge → CV hold → rest). "
            "1C = a full charge/discharge in ~1 hour; CV hold = "
            "constant-voltage top-up. Indicators read as in-cycle "
            "peaks/deltas.")
add_notes(s, """Claims: C33 (reference cell values), model protocol
from dfn_validation.md.

Walk the four panels: voltage follows the 1C protocol (discharge to
2.8 V, rest, charge to 4.2 V with CV hold to 50 mA, rest); current is
the square-wave drive; the Si surface-stress panel shows the in-cycle
peak during discharge (max lithiation) — THAT peak is the compared
indicator, not a rest value; the thickness panel shows the electrode
breathing ~12 µm on a ~75 µm stack.

How 'capacity' is extracted: PyBaMM accumulates a discharge-capacity
counter; per-cycle capacity is the counter's last-minus-first within
the DISCHARGE step only (the figure annotates this). A bug in this
diffing was found and fixed during validation — worth mentioning as
an example of why the model layer needs self-checks.""")

# ---- 28 sweep bands
s = new_slide("structure → model", "The assumption sweep: 17 runs per batch")
pic_fit(s, "fig_sweepbands.png", Inches(0.5), Inches(1.45),
        Inches(8.3), Inches(5.15))
add_bullets(s, Inches(9.0), Inches(1.6), Inches(3.9), Inches(5.0), [
    "**Why sweep",
    (1, "unmeasured inputs are bands, not points — the honest output "
        "is a band of indicators"),
    "**17 points",
    (1, "band centre + corners + edges over porosity / Si share / "
        "radius / mechanics / thickness"),
    "**The picture",
    (1, "batch medians sit inside overlapping bands — the "
        "assumptions dominate the differences"),
], size=12.5)
add_foot(s, "Assumption-only spread ~1.16 Ah IQR (the typical "
            "middle-half spread across sweep runs) on cycle-5 "
            "capacity — dwarfs every batch difference. Ah = "
            "amp-hours, battery charge. C27–C29.")
add_notes(s, """Claims: C27 (17/17 converged), C29 (1.16 Ah IQR).

This slide is the model honesty contract: since ~10 inputs are
assumed bands, a single run would imply false precision. 17 points
per batch sample the assumption hyper-rectangle around each batch's
measured band. The output is therefore a BAND of predicted
indicators per batch, and the honest comparison asks whether the
bands separate — mostly they do not.

The headline: pooled assumption-only spread on cycle-5 capacity is
~1.16 Ah (IQR); the largest batch median difference is 0.15 Ah. The
model cannot see batch differences through that fog — which is
exactly why the comparison uses PAIRED differences at matched sweep
indices (next slide), not absolute values.""")

# ---- 29 paired differences
s = new_slide("structure → model", "Paired differences: what moves, what doesn't", route=True)
pic_fit(s, "fig_paired.png", Inches(0.5), Inches(1.45),
        Inches(8.4), Inches(5.15))
add_bullets(s, Inches(9.1), Inches(1.55), Inches(3.8), Inches(5.15), [
    "**Declared thresholds first",
    (1, "capacity 0.25 Ah (amp-hours) · swelling 0.5 µm · stress "
        "0.5 MPa (megapascals) · min anode voltage 0.01 V — fixed "
        "before looking"),
    "**The reads",
    (1, "capacity: B1–B3 +0.099, B2–B3 +0.150 Ah — below threshold"),
    (1, "Si stress: B1–B3 −0.81, B2–B3 −1.59 MPa — crosses the "
        "declared bar"),
    (1, "swelling & plating-margin: below threshold"),
    "**The catch",
    (1, "the stress shift rides on assumed Si share + modulus — "
        "model-conditional, not measured"),
], size=11.5)
add_foot(s, "dfn_paired_differences.csv — same cathode per sweep "
            "index; differences at matched assumptions only. C28–C32.")
add_notes(s, """Claims: C28–C32.

Explain the pairing trick: at each of the 17 sweep indices the
cathode is re-sized to hold the reference N/P constant, so
differences reflect ANODE-geometry differences at matched
assumptions — it is a paired design that removes the shared
assumption axis to first order.

Results:
- capacity differences (+0.099, +0.150 Ah) are real model outputs but
  sit below the pre-declared 0.25 Ah threshold — under the declared
  rules they do NOT count as meaningful;
- peak Si surface stress differs (−0.81, −1.59 MPa) beyond the 0.5
  MPa bar — the one indicator that 'resolves' batches;
- swelling (~0.1 µm) and min-anode-V stay under threshold.

The honesty suffix: post-hoc, the stress difference correlates with
assumed active-Si share (rho ~ −0.93 across the 51 runs) and ~0 with
measured Si radius — i.e. the model is largely re-expressing the
Si-share assumption bracket, not discovering new physics. All of
this is model-conditional: no cell was ever built or tested.""")

# ---- 30 ablation + validation meaning
s = new_slide("structure → model", "Validation means self-consistency — and it caught real bugs")
pic_fit(s, "fig_ablation.png", Inches(0.5), Inches(1.45),
        Inches(7.9), Inches(5.15))
add_bullets(s, Inches(8.6), Inches(1.5), Inches(4.25), Inches(5.2), [
    "**What was wrong",
    (1, "shipped constants had the fade rate (LAM = loss of active "
        "material) 3600x too fast and the protective-film rate "
        "(SEI = solid electrolyte interphase) 1000x too slow vs the "
        "cited sources"),
    "**What it did",
    (1, "buggy model lost ~29% capacity in 5 cycles — physically "
        "absurd, and it would have shipped"),
    "**What validation is",
    (1, "8/8 self-checks now pass — reference sanity, ablation, "
        "convergence, protocol audit"),
    "**What it is NOT",
    (1, "no experimental data exist → nothing here validates "
        "predictive accuracy"),
], size=12.5)
add_foot(s, "dfn_validation.md — constants audit + ablation table. "
            "C34–C36.")
add_notes(s, """Claims: C34, C35, C36.

Two teaching points.

1. Validation ≠ accuracy. The 8/8 checks prove the code does what it
   claims (protocol correct, counters diffed right, converged,
   reference sane). They say nothing about whether the model predicts
   real cells — that needs experimental data we do not have.

2. The ablation is the cautionary tale: as shipped, the LAM
   proportional constant was 1e-3/s vs the published 2.7778e-7/s
   (3,600x too fast) and the SEI rate 1e-15 vs 1e-12 m/s (1,000x too
   slow). The buggy model shredded ~29% capacity in 5 cycles —
   obviously wrong only because we checked. Without the ablation the
   pipeline would have produced confident, wrong, smooth-looking
   numbers. Every 'validated' badge in this project means 'internally
   consistent', never 'true'.""")

# ---- 31 where evidence stops (ROUTE)
s = new_slide("the evidence wall", "Where the evidence stops — the honest ledger", route=True)
add_bullets(s, Inches(0.55), Inches(1.45), Inches(5.95), Inches(5.4), [
    "**Established (measured)",
    (1, "Batch_1 pore-count shift vs Batch_3 (uniform, robust)"),
    (1, "3-of-7 bright-material anomaly; likely fine silicon"),
    (1, "both pipelines detect; they differ only on naming"),
    "**Best-supported but open",
    (1, "stress-indicator differences (model-conditional)"),
    (1, "dim-contrast on 2 flagged photos (drift or grade?)"),
    "**Not established (can't be, with this data)",
    (1, "chemical identity · any cell-level consequence"),
    (1, "general batch ranking · equivalence on the other metrics"),
], size=13.5)
add_bullets(s, Inches(6.9), Inches(1.45), Inches(5.95), Inches(5.4), [
    "**The honest ledger",
    (1, "assumed inputs: thickness, sub-res porosity, Si share "
        "bracket, mechanics, degradation constants"),
    (1, "assumption spread > every batch difference on capacity"),
    (1, "34 images is small — MDDs bound what 'no difference' "
        "means"),
    "**What would close the gaps (roadmap.md)",
    (1, "EDS (elemental spectroscopy) on flagged regions → "
        "chemistry"),
    (1, "controlled repeat imaging → acquisition effects"),
    (1, "blind expert annotation → labels for classifier eval"),
    (1, "measured thickness/porosity → shrink the assumed bands"),
    (1, "electrochemical testing → ground the indicators"),
], size=13.5)
add_foot(s, "Every line above maps to a row in claim_evidence.csv.",
         warn=True)
add_notes(s, """The ledger slide — the whole project on one screen.
The left column is the evidence-graded answer to the original
question; the right column is the audit trail a reviewer should
demand.

If asked 'so is Batch_1 bad?' the honest answer is: 'Batch_1 is
detectably different — a uniform pore-count shift plus a 3-of-7
likely-fine-silicon anomaly — under a frozen recipe against the
selected reference. Whether that difference matters for cells is
unanswered; the model indicators that do resolve batches are
assumption-dominated.'""")

# ---- 32 new images
s = new_slide("incoming data", "Three unlabelled photos: similarity, not provenance")
pic_fit(s, "fig_newbatch.png", Inches(0.5), Inches(1.45),
        Inches(8.2), Inches(5.1))
add_bullets(s, Inches(8.95), Inches(1.55), Inches(4.0), Inches(5.1), [
    "**Envelope coverage on the 9 features",
    (1, "img_3e122cbj → inside Batch_1 envelope on 7/9 — including "
        "the uncertain-bright signature → assigned Batch_1 "
        "(cautious)"),
    (1, "img_fn0mhxef → 9/9 inside Batch_2"),
    (1, "img_xrv9xvzb → 9/9 inside Batch_3"),
    "**Low confidence by design",
    (1, "overlapping envelopes = similarity classes, not identity; "
        "the labels are hypothesis-generating only"),
], size=12)
add_foot(s, "new_image_assignment/assignments.csv — median robust-z "
            "assignment; C40.")
add_notes(s, """Claims: C40.

The new-image exercise demonstrates the correct posture for incoming
data: run the FROZEN recipe, compute the 9 declared features, and ask
which batch envelope the image falls inside — no re-fitting, no
verdict. img_3e122cbj is the interesting one: it lands inside the
Batch_1 envelope on the uncertain-bright fraction (0.086, vs the B1
anomaly signature) plus the elevated noise-adjusted pore count (193)
— i.e. it resembles the Batch_1 phenotype, a hypothesis worth
following with EDS/re-imaging, not a provenance call.""")

# ---- 33 roadmap
s = new_slide("prospective use", "If this were run prospectively: the frozen protocol")
pic_fit(s, "fig_roadmap.png", Inches(0.5), Inches(1.4),
        Inches(8.6), Inches(5.3))
add_bullets(s, Inches(9.35), Inches(1.55), Inches(3.6), Inches(5.2), [
    "**Freeze BEFORE new data",
    (1, "recipe, reference, classifier, model assumptions, "
        "comparisons, decision rules — all versioned"),
    "**Gates that stop the line",
    (1, "acquisition outside supported conditions → review, don't "
        "force a verdict"),
    (1, "image-quality guards fail → investigate imaging first"),
    "**Unblind last",
    (1, "save results before revealing labels; evaluate at the "
        "image unit"),
], size=12)
add_foot(s, "roadmap.md has the full stepwise table: inputs, outputs, "
            "owner, pass/stop criteria.")
add_notes(s, """roadmap.md operationalises every lesson in this deck
as a stage gate: each step lists inputs, outputs, a responsible
agent, and a pass/stop criterion. The three principles:

1. Everything that can move is frozen and versioned before new data
   arrive — including the decision rules, so 'what counts as a
   difference' cannot be negotiated after seeing outcomes.
2. Guards first: scale verification, contrast/noise checks, specimen
   identity — a failed guard sends the batch to imaging review, not
   into the verdict machinery.
3. Evaluation at the image unit with results saved before unblinding
   — so the statistics stay honest.

Then the prioritized follow-ups: controlled repeat imaging (resolves
acquisition-vs-material), blind expert annotation (gives the
classifier an eval target it currently lacks), targeted EDS (closes
the chemistry question), measured thickness/porosity (shrinks the
assumption bands), electrochemical testing (grounds the
indicators).""")

# ---- 34 orientation open question
s = new_slide("open questions", "Bonus open question: through-plane pore texture")
pic_fit(s, "fig_orientation.png", Inches(0.5), Inches(1.45),
        Inches(8.6), Inches(4.9))
add_bullets(s, Inches(9.35), Inches(1.6), Inches(3.6), Inches(4.8), [
    "**What we claimed",
    (1, "pores and texture prefer the vertical/through-plane "
        "direction (FFT directionality + object shape)"),
    "**What the audit found",
    (1, "the FFT score measures the frame's aspect ratio — pure "
        "noise scores the same at the real 1034x3500 frame shape"),
    (1, "direct autocorrelation shows features ~1.4x HORIZONTAL — "
        "the claim was withdrawn"),
    "**Why show a retracted result",
    (1, "this is what honest reporting looks like: the finding was "
        "in orientation_features.csv for weeks before anyone ran the "
        "noise control"),
], size=12)
add_foot(s, "claim C42 withdrawn — see archive/ARCHIVE.md and "
            "archive/orientation_features.csv")
add_notes(s, """Claims: C42 — WITHDRAWN (was exploratory).

The retraction slide. We reported an orientation direction for weeks:
pores preferred the vertical/through-plane direction by object shape,
and an FFT energy ratio 'agreed'. The audit found the FFT metric reads
the frame's aspect ratio — pure noise scores +0.7 at the images'
1034x3500 shape — and a direct autocorrelation shows slightly
HORIZONTALLY elongated features, the opposite of the claim. The lesson
is the process: every metric needs a null control (what does it score
on pure noise?), and an 'exploratory' flag does not exempt a claim from
a mechanism check. Kept in the deck because the retraction is better
teaching material than the original slide was.""")

# ---- 35 take-home (ROUTE)
s = new_slide("take-home", "Five things to keep", route=True)
add_bullets(s, Inches(0.55), Inches(1.55), Inches(12.0), Inches(5.0), [
    "**1. Labels are model outputs.** Every 'material fraction' is a "
    "threshold+classifier choice — freeze and version the recipe.",
    "**2. The image is the unit.** 34 images is small; 'no clear "
    "difference' carries a minimum-detectable-difference, never "
    "equivalence.",
    "**3. One supported finding.** Batch_1 > Batch_3 on noise-adjusted "
    "resolved pore count (uniform, robust to dropping any image) — "
    "plus a separate 3-of-7 likely-fine-silicon anomaly still "
    "awaiting chemical confirmation.",
    "**4. Model outputs are conditional.** DFN (battery-physics) "
    "indicators move with assumed inputs more than with batches — "
    "report paired differences against declared thresholds, never "
    "absolute claims.",
    "**5. Verdicts are similarity statements.** ACCEPT/INVESTIGATE/"
    "REJECT describe distance from a SELECTED reference — not "
    "chemistry, performance, or safety.",
], size=15)
add_foot(s, "Knowledge check + answers in the repo; every number "
            "traces via claim_evidence.csv.")
add_notes(s, """Close on the five transferable lessons — each one a
general analysis principle demonstrated by this specific dataset.
Offer the knowledge check as self-assessment. Remind the audience
that every number quoted tonight has a row in claim_evidence.csv —
that file, not the slides, is the contract.""")

# ---- 36 knowledge check pointer / references
s = new_slide("wrap-up", "Traceability and further reading")
add_bullets(s, Inches(0.55), Inches(1.5), Inches(6.1), Inches(5.2), [
    "**Trace your numbers",
    (1, "claim_evidence.csv — 50 claims → source file + class"),
    (1, "metric_dictionary.csv — units, equations, limits"),
    (1, "dataset_manifest.csv — every image + md5"),
    "**Self-test",
    (1, "knowledge_check.md — 8 questions with answers"),
    "**Rebuild everything",
    (1, "BUILD.md — one command regenerates figures + deck"),
    "**Expert review",
    (1, "expert_review_sheet.csv — what a human expert should "
        "verify and sign off"),
], size=14)
add_bullets(s, Inches(7.0), Inches(1.5), Inches(5.85), Inches(5.2), [
    "**Primary sources",
    (1, "Doyle, Fuller, Newman — JECS 140:1526 (1993): the DFN model"),
    (1, "Chen et al. — PhysChemChemPhys 22 (2020): parameter set"),
    (1, "OKane et al. — PhysChemChemPhys 24 (2022): degradation "
        "constants"),
    (1, "Bonkile & Ramadesigan — J. Power Sources 581 (2024): Si "
        "mechanics"),
    (1, "Otsu — IEEE SMC-9 (1979); Saltykov stereology; Delesse "
        "(1848); Clark & Evans (1954); Benjamini & Hochberg JRSS-B "
        "57 (1995)"),
], size=13)
add_foot(s, "Appendix slides follow: pairwise table, calibration "
            "detail, assumption table, stage gallery, environment.")
add_notes(s, """Close with the reproducibility contract and the
primary literature. The citations on the right are the mechanism and
method sources; the repository files on the left are where every
number lives.

Suggested closing exercise: give the audience claim_evidence.csv and
ask them to pick any three numbers shown tonight and trace them to
source files — the trace IS the lesson. The expert_review_sheet.csv
lists what a domain expert should verify before this pack's
interpretations are trusted: the ambiguous-bright classification,
the contrast-anomaly reading, the assumption bands, and five
spot-checks on the claim ledger. The knowledge_check.md questions
map one-to-one to the take-home points; answer key included.

References in full (for the notes): Doyle M., Fuller T.F., Newman J.
J. Electrochem. Soc. 140:1526 (1993); Chen C.-H. et al.
Phys.Chem.Chem.Phys. 22 (2020); O'Kane S.E.J. et al.
Phys.Chem.Chem.Phys. 24 (2022); Bonkile M.P. & Ramadesigan V.
J. Power Sources (2024); Otsu N. IEEE Trans. SMC-9:62 (1979);
Benjamini Y. & Hochberg Y. JRSS-B 57:289 (1995); Clark P.J. &
Evans F.C. Ecology 35:445 (1954); Delesse A. C.R. Acad. Sci. 25:544
(1848); Saltykov S.A. stereological sectioning correction.""")

# ============================ APPENDIX ============================

# A1 pairwise table
s = new_slide("appendix A1", "All 9 features x 3 batch pairs (permutation + BH)")
import pandas as pd
pw = pd.read_csv(os.path.join(os.path.dirname(TEACH),
                              "validated_comparison",
                              "pairwise_comparisons.csv"))
hdr = ["feature", "median diff", "exact p", "BH", "MDD(80%)"]
pairs = sorted(pw.pair.unique())
for t_i, pair in enumerate(pairs):
    sub = pw[pw.pair == pair]
    x = Inches(0.45 + t_i * 4.3)
    tf = _box(s, x, Inches(1.35), Inches(4.2), Inches(0.4))
    _run(tf.paragraphs[0], pair.replace("Batch_", "B"), 13, TEAL,
         bold=True)
    tab = s.shapes.add_table(len(sub) + 1, len(hdr), x, Inches(1.75),
                             Inches(4.15), Inches(4.9)).table
    tab.columns[0].width = Inches(1.6)
    for j in (1, 2, 3, 4):
        tab.columns[j].width = Inches(0.62)
    for j, h in enumerate(hdr):
        c = tab.cell(0, j); c.text = h
        c.text_frame.paragraphs[0].font.size = Pt(9)
        c.text_frame.paragraphs[0].font.bold = True
        c.text_frame.paragraphs[0].font.name = FONT
    for i, (_, r) in enumerate(sub.iterrows()):
        vals = [str(r["feature"]),
                f"{r['median_diff']:+.3g}",
                f"{r['p_exact']:.4g}",
                "YES" if r["significant_bh"] else "—",
                f"{r['mdd_80pct']:.3g}"]
        for j, v in enumerate(vals):
            c = tab.cell(i + 1, j); c.text = v
            fp = c.text_frame.paragraphs[0].font
            fp.size = Pt(8); fp.name = FONT
            if j == 3 and v == "YES":
                fp.bold = True; fp.color.rgb = WARN
add_foot(s, "validated_comparison/pairwise_comparisons.csv — the "
            "authoritative table; 'no clear difference' rows carry MDDs.")
add_notes(s, """Appendix reference table — every declared feature x
pair. Only rows marked YES survive BH at FDR 0.10. The MDD column is
the answer to 'could a real difference be hiding?' — differences
below it are invisible at n=7 (or 17).""")

# A2 calibration detail
s = new_slide("appendix A2", "Corrected calibration: what changed and why it matters")
add_bullets(s, Inches(0.55), Inches(1.55), Inches(12.0), Inches(4.6), [
    "**Original QC verdicts**: Batch_1 REJECT · Batch_2 INVESTIGATE",
    (1, "the flag weights + T² reference (a multivariate "
        "'how-unusual' statistic) were fitted on data that saw the "
        "batches under test"),
    "**Corrected recheck** (reference-only weights + held-out T²): "
    "BOTH batches → INVESTIGATE",
    (1, "Batch_1: 1/7 regions exceeds flag threshold; 2 more above "
        "the baseline's p90 (90th percentile, same signature); "
        "held-out T² unusual (p ≈ 0.00005)"),
    "**Baseline honesty check**: LOO exceedance = 1/17 ≈ 5.9% "
    "[CI 1.0–26.9%] — the earlier 'zero false alarms' claim is "
    "withdrawn",
    "**Standing today**: the anomaly EXISTS; the severity label is "
    "calibration-dependent → INVESTIGATE is the defensible call",
], size=14)
add_foot(s, "qc_recheck.csv · calibration_counts.csv · checks.csv — "
            "C23–C25, C50.")
add_notes(s, """Claims: C23–C25, C50.

Appendix-level detail on the verdict correction for reviewers who
want the mechanics: the original flag weights and T² null were fitted
with exposure to the test batches, inflating confidence. The
reference-only recheck keeps the SAME flag thresholds but derives
weights and the multivariate null from the reference alone — that is
the change that demotes Batch_1 to INVESTIGATE and produces the
measured 1/17 baseline exceedance.""")

# A3 assumption table
s = new_slide("appendix A3", "The declared assumptions (full sweep table)")
at_path = os.path.join(os.path.dirname(TEACH), "dfn_output",
                       "assumption_table.csv")
at = pd.read_csv(at_path)
at = at[at.batch == "Batch_3"]          # same assumed inputs every batch
cols_at = ["input", "value", "why assumed"]
rows_at = at[cols_at].astype(str).values.tolist()
hdr_at = cols_at
tab = s.shapes.add_table(len(rows_at) + 1, len(hdr_at), Inches(0.5),
                         Inches(1.4), Inches(12.4), Inches(4.9)).table
tab.columns[0].width = Inches(2.6)
tab.columns[1].width = Inches(3.6)
tab.columns[2].width = Inches(6.2)
for j, h in enumerate(hdr_at):
    c = tab.cell(0, j); c.text = h
    c.text_frame.paragraphs[0].font.size = Pt(11)
    c.text_frame.paragraphs[0].font.bold = True
    c.text_frame.paragraphs[0].font.name = FONT
for i, row in enumerate(rows_at):
    for j, v in enumerate(row):
        c = tab.cell(i + 1, j); c.text = v[:110]
        c.text_frame.paragraphs[0].font.size = Pt(9.5)
        c.text_frame.paragraphs[0].font.name = FONT
add_foot(s, "dfn_output/assumption_table.csv — every unmeasured input "
            "is declared with its source and sweep band.")
add_notes(s, """Appendix table: the complete assumption ledger —
what is assumed, the default value, the sweep band, and the source
(literature citation or declared convention). This is the file a
reviewer checks when asking 'what did you just make up?' — the honest
answer is 'everything in this table, and only this table'.""")

# A4 stage gallery
s = new_slide("appendix A4", "The transformation chain on the reference crop")
pic_fit(s, "stage_4.png", Inches(0.6), Inches(1.45), Inches(12.1),
        Inches(0.95))
ANN_DIR = os.path.join(TEACH, "annotated_images")
for i, (f, lab) in enumerate([
        ("ref_raw_crop.png", "raw"),
        ("ref_flat_crop.png", "flat-fielded"),
        ("ref_seg_crop.png", "labelled")]):
    path = os.path.join(ANN_DIR, f)
    from PIL import Image as _IM
    ar = _IM.open(path).size
    h = Inches(3.6)
    w = Inches(3.6)
    pic = s.shapes.add_picture(path, Inches(0.9 + i * 4.15),
                               Inches(2.7), width=w)
    pic.line.color.rgb = RGBColor(0xD8, 0xDE, 0xE3)
    pic.line.width = Pt(0.75)
    tf = _box(s, Inches(0.9 + i * 4.15), Inches(6.35), Inches(3.6),
              Inches(0.4))
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    _run(p, lab, 12, GREY, italic=True)
add_foot(s, "Annotated crops in annotated_images/ — agent-produced "
            "annotations, NOT expert labels (annotation_provenance.md).")
add_notes(s, """Gallery slide — the pipeline strip on top shows where
in the chain these crops sit; the three images below are the SAME
1024x1024 reference window at three stages (raw, flat-fielded,
labelled). Useful when a reviewer asks 'show me the data before and
after each step' — the crops are byte-for-byte the annotated_images/
assets, and annotation_provenance.md states plainly that they are
algorithm-produced, not expert labels. If an expert later annotates
the same crop blind, this is the before/after comparison target.""")

# A5 environment / reproducibility
s = new_slide("appendix A5", "Reproduce it: environment, seeds, commands")
import json as _json
_bs_path = os.path.join(TEACH, "build_state.json")
if os.path.exists(_bs_path):
    _bs = _json.load(open(_bs_path))
    _stamp = (f"build stamp — sync {_bs.get('last_sync_utc','?')} · "
              f"{len(_bs.get('fingerprints',{}))} fingerprinted inputs "
              f"· stages: {', '.join(_bs.get('built_stages',[])) or 'up-to-date'}")
else:
    _stamp = ("unstamped build — run scripts/sync.py to fingerprint "
              "inputs and stamp this pack")
add_bullets(s, Inches(0.55), Inches(1.55), Inches(12.0), Inches(4.9), [
    "**Environment (version_manifest)",
    (1, "Python 3.13.5 · NumPy 2.5.3 · pandas 3.0.6 · SciPy 1.18.1 · "
        "scikit-image 0.26.0 · matplotlib 3.11.2 · PyBaMM 26.9.0.0"),
    "**Self-updating pack",
    (1, "scripts/sync.py fingerprints all inputs, rebuilds only "
        "changed stages, and re-audits every claim"),
    (1, _stamp),
    "**Seeds and recipes",
    (1, "comparison recipe vcompare-1.0, seed 0, fitted on 17 "
        "Batch_3 images (recipe.json, md5'd inputs)"),
    "**Rebuild this teaching pack",
    (1, "scripts/sync.py — one command does manifest → figures → "
        "deck → PDF → claim audit"),
    (1, "audit_claims.py — re-verifies every file-checkable claim; "
        "STALE = source no longer matches the deck"),
    "**Traceability",
    (1, "claim ids (C01–C50) in speaker notes ↔ claim_evidence.csv"),
], size=14)
add_foot(s, "Version pins in version_manifest.txt; commands in "
            "BUILD.md.")
add_notes(s, """Appendix: reproducibility block. One command chain
rebuilds manifest → figures → deck → PDF (BUILD.md); figure build
includes a self-check that recomputes the exact permutation p-value
and compares it to the saved table (must print 0.000477). All
versions pinned in version_manifest.txt; the frozen recipe itself
(md5'd inputs, seed 0, fitted on the 17 Batch_3 images) is the
auditable anchor — any rebuild changing a number means either the
data or the code changed, both of which are detectable.""")

prs.save(OUT)
print("wrote", OUT, "with", len(prs.slides.__iter__.__self__._sldIdLst),
      "slides")
