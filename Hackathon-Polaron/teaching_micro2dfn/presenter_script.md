# Presenter script — 10-slide route (~15 min)

Verbatim narration for the ROUTE-tagged slides. Slide numbers refer to
the built deck. Claim ids (C…) point into `claim_evidence.csv`.

---

## Slide 1 — Title

> "This is a worked tour of one real dataset: 34 electron-microscope
> cross-sections of battery anodes. By the end you'll see how a grey
> image becomes material labels, how labels become a batch comparison
> and inputs to a physics model — and, just as importantly, where the
> evidence stops. Two things to hold onto throughout: the reference
> batch is a *chosen* baseline, not proven-perfect material; and every
> number I show is labelled measured, assumed, or simulated."

## Slide 3 — The question

> "Three batches. Batch_3 is the shipment-approved reference; Batch_1
> and Batch_2 are candidates; plus three unlabelled photos. The
> practical questions are: do the candidates look like the reference —
> a QC question — and where they differ, what could it mean downstream
> — a consequence question. What I will *not* be claiming: chemistry,
> cell performance, lifetime, or safety. The images cannot supply
> those."

## Slide 4 — One field of view

> "This is one field of view, 25 nanometres per pixel — a sampling
> pitch, not a resolution. In back-scattered electrons, brightness
> rises with mean atomic number: near-black is open pore, mid-grey is
> the graphite+binder bulk, and bright is silicon-family material.
> We'll follow this exact image — img_hawkfj64, chosen because it's
> the Batch_3 image nearest the batch median on all nine declared
> features — through every transformation. Nothing is cherry-picked;
> the selection rule is documented."

## Slide 7 — Two systems

> "Two pipelines share the same pixels but do different jobs.
> polaron_qc is anomaly detection: does this batch break the
> reference's own envelope? Its verdicts are similarity statements.
> micro2dfn measures structure — pores, silicon objects, their
> distances — and feeds exploratory model runs. They differ at exactly
> one decision: what counts as a silicon *particle* versus ambiguous
> bright material. That single disagreement explains every downstream
> difference between them. Neither is claimed 'better' — comparing
> accuracy needs ground truth that doesn't exist yet."

## Slide 10 — Frozen ruler

> "Here's the metrology choice everything depends on. The segmentation
> thresholds were fitted once on Batch_3 pixels and frozen; every image
> is measured by that same ruler. The alternative — refitting
> thresholds per image — sounds adaptive but actually lets the ruler
> bend to the anomaly: on the disputed photos, a floating threshold
> counts *more* bright material, not less. Freezing isn't unbiased;
> it's *comparable* — and it makes the ruler's own uncertainty
> reportable."

## Slide 13 — Measurements

> "Objects become numbers in three ways: sizes — equivalent diameter
> of each particle; distances — how far is each silicon boundary from
> pore space, a two-dimensional stand-in for wetting; and topology —
> is the pore space one network or fragments? The cyan overlay marks
> the largest connected pore region. Every one of the ~30 markers in
> the metric dictionary is built from these three primitives, and
> each is aggregated per image — the image stays the unit of
> analysis."

## Slide 16 — The result

> "Nine declared features, three batch pairs, exact permutation tests
> with a false-discovery-rate correction — and the honest headline is
> boring: *one* supported difference. Batch_1 resolves more pore
> objects per area than Batch_3 — and it survives a noise adjustment.
> Everything else — silicon fractions, sizes, clustering, porosity —
> shows no clear difference *at these sample sizes*; the
> minimum-detectable-difference column tells you how big a real
> difference could still be hiding. No clear difference is not
> equivalence."

## Slide 29 — Paired differences

> "First, what actually feeds the model — because it matters for how
> to read this. Measured inputs: resolved porosity, the silicon
> particle radius. The silicon *share* is a bracket, not a number:
> the low end counts only classifier-confirmed silicon, the high end
> counts all the ambiguous bright material — the disagreement between
> the pipelines enters as explicit uncertainty. And note: pore
> *count* — the one supported finding — is not a model input at all.
> Now the comparison: paired at matched assumption points, with
> meaningful-difference thresholds declared *before* looking: 0.25
> amp-hours capacity, half a micron swelling, half a megapascal
> stress. Result: capacity differences — plus 0.10 and 0.15
> amp-hours — are real model outputs but sit *below* the declared
> bar. Peak silicon stress does differ — about minus 0.8 and minus
> 1.6 megapascals — the one indicator that resolves batches. The
> catch: that stress difference rides mostly on the assumed silicon
> share, not on anything the images measured. These are conditional
> indicators, not measured performance — no cell was ever built or
> tested."

## Slide 31 — The evidence wall

> "Here's the honest ledger. Established: the pore-count shift, and
> the three-region anomaly — three of seven Batch_1 regions carry
> extra bright material; both pipelines flag the same regions, and
> six pixel-level tests read it as likely fine silicon. Note the
> verdict story too: the original automatic REJECT was corrected to
> INVESTIGATE under stricter calibration — the anomaly is real, the
> severity label was overstated. Open: the chemistry — that needs
> elemental spectroscopy — whether two dim photos are drift or a
> dimmer silicon grade, and every cell-level consequence —
> assumption spread dominates the model. What would close the gaps:
> elemental analysis on the flagged regions, controlled repeat
> imaging, blind expert annotation for a classifier target, measured
> thickness and porosity to shrink the assumed bands, and eventually
> electrochemical testing."

## Slide 35 — Take-home

> "Five things to keep. Labels are model outputs — freeze and version
> the recipe. The image is the unit — and 'no clear difference' is
> bounded by a detectable-difference, never equivalence. One
> supported finding, one unresolved anomaly. Model outputs are
> conditional on declared assumptions. And verdicts are similarity
> statements — distance from a selected reference — not chemistry,
> performance, or safety. Every number tonight traces to a file
> through claim_evidence.csv — that file, not the slides, is the
> contract."

---

### If asked…

- *"Is Batch_1 bad?"* — "Detectably different under a frozen recipe:
  a uniform pore-count shift plus a 3-of-7 likely-fine-silicon
  anomaly. 'Bad for cells' is unanswered — no electrochemical data."
- *"Why not just re-threshold each image?"* — "Then the ruler bends
  to the anomaly. Frozen keeps images comparable; floating stays a
  diagnostic."
- *"Which pipeline is right about silicon?"* — "They agree on the
  pixels (r ≥ 0.8) and disagree on the naming of borderline objects.
  The adjudication tests favour 'fine silicon'; chemistry needs EDS."
- *"Can the model settle it?"* — "No. The assumption spread dwarfs
  the batch differences; the model answers 'what could geometry do'
  not 'what will cells do'."
